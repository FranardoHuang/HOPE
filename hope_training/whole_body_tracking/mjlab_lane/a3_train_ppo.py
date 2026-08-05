#!/usr/bin/env python3
"""PPO (rsl-rl) on the A3 court scene -- the "is it actually learning?" run.

WHAT THIS IS (plain language)
-----------------------------
The plant step proved we can carry the vendor's robot physics into MuJoCo Warp
field by field.  The scene step put a calibrated table, net and ball around it.
This step closes the loop: a real policy, real PPO updates, on GPU, with the
same plant -- so we can say "it trains", not just "it steps".

It deliberately does **not** try to match the Isaac A211/C211 observation and
reward columns.  That is the next step (cross-engine parity).  What is claimed
here is narrower and checkable:

  * our robot + our table + our plant, batched at nworld = 4096,
  * driven by rsl-rl 5.4.0 PPO through mjlab's own runner class,
  * with a reward that a human can read off a curve: stay in the split-ready
    stance, stay upright, and get the racket to the ball.

WHAT IS *NOT* mjlab-DEFAULT HERE (and why)
------------------------------------------
mjlab's ManagerBasedRlEnv is not used.  Its managers would re-open exactly the
three doors the plant step nailed shut: ``ActionCfg`` installs mjlab actuators
over the vendor's 31 pure-torque motors, ``EventCfg``/``CollisionCfg`` rewrite
geoms, and the observation manager assumes an mjlab ``Entity``.  So this file
implements the rsl-rl ``VecEnv`` interface *directly* on the ``A3PlantEnv``
handle that ``a3_court_env.build_court_env`` returns.  Everything above the
env boundary (PPO, storage, logger, runner, checkpointing) is stock rsl-rl /
mjlab -- ``MjlabOnPolicyRunner``, not a hand-rolled loop.

CONTROL ABI
-----------
The vendor's actuators are pure torque motors and the deployed controller
computes the PD itself.  We do the same, once per *physics* step (1 kHz):

    q_des = clamp(q_ready + action_scale * a, joint_range)      # 50 Hz
    tau   = clamp(kp*(q_des - q) - kd*qd, ctrlrange)            # 1 kHz

so the policy's action is a residual joint-position target around the
split-ready pose, which is the same shape the Isaac lane uses.

Usage
-----
  python a3_train_ppo.py --smoke                       # 64 worlds, 3 iters
  python a3_train_ppo.py --nworld 4096 --iterations 60 --seed 0 --tag s0
  python a3_train_ppo.py --analyze RUN_s0.jsonl RUN_s1.jsonl --out BAND.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# a3_court_env pulls in a3_plant_env, calibrate_restitution and geometry.
import a3_court_env as court  # noqa: E402

plant = court.plant
geom = court.geom


# ==========================================================================
# Task configuration.  Everything a human would want to argue about is here.
# ==========================================================================


@dataclass
class TaskCfg:
  """The reward/termination recipe.  Deliberately small and readable."""

  # --- episode -----------------------------------------------------------
  episode_length_s: float = 3.0
  action_scale: float = 0.25          # rad of joint-target residual per unit action
  action_scale_mode: str = "flat"
  """``flat`` = one number for all 31 joints (this file's own default).

  ``vendor`` = the Isaac/deploy decoder scale, per joint
  ``0.25 * effort_limit / kp`` (``robots/agibot_a3.py::AGIBOT_A3_ACTION_SCALE``),
  i.e. ``a = 1`` asks for a quarter of that joint's torque budget at zero
  velocity.  Ranges from 0.0375 rad (head, 6/40) to 0.647 rad (waist yaw,
  220/85), so it is NOT a rescaling of ``flat`` -- it re-weights which joints
  the policy can move.  Provided because A211 parity will need it.
  """
  action_clip: float = 4.0            # hard clip on the raw policy output

  # --- observation scaling ----------------------------------------------
  obs_scale_lin_vel: float = 0.5
  obs_scale_ang_vel: float = 0.25
  obs_scale_joint_vel: float = 0.05
  obs_scale_ball_vel: float = 0.1
  obs_clip: float = 100.0

  # --- reward weights ----------------------------------------------------
  # NOTE on `w_alive`: a large constant alive bonus is a *reporting* hazard, not
  # a learning one -- it adds the same number to every step of every policy and
  # so compresses the visible dynamic range of the reward curve.  The pilot run
  # (w_alive = 1.0) sat at r/step = 2.99 -> 3.03 while the racket->ball distance
  # improved 0.41 -> 0.33 m: real learning, invisible curve.  Kept small.
  w_alive: float = 0.25
  w_pose: float = 1.0                 # split-ready joint tracking
  w_upright: float = 0.5
  w_height: float = 0.5
  w_reach: float = 2.0                # smooth racket->ball distance shaping
  w_touch: float = 4.0                # sharp bonus for actually being on the ball
  w_action_rate: float = -0.02
  w_joint_vel: float = -0.002
  w_torque: float = -0.05             # on the *normalized* torque
  r_termination: float = -5.0

  # reward kernels
  k_pose: float = 2.0                 # exp(-k * mean square joint error)
  k_upright: float = 4.0
  k_height: float = 40.0
  reach_len_m: float = 0.8            # exp(-d / reach_len)
  touch_sigma_m: float = 0.15         # exp(-(d/sigma)^2)

  # --- termination -------------------------------------------------------
  min_pelvis_z: float = 0.70          # ready pelvis sits at 1.0684
  max_tilt_proj_g: float = -0.5       # projected gravity z; -1 = perfectly upright

  # --- reset randomization ----------------------------------------------
  reset_joint_noise_rad: float = 0.05
  reset_joint_vel_noise: float = 0.0
  reset_root_xy_noise_m: float = 0.02
  reset_root_yaw_noise_rad: float = 0.05

  # --- ball ---------------------------------------------------------------
  ball_reserve_after_s: float = 2.0   # re-serve if the rally has clearly died
  ball_dead_z_hope: float = -0.35     # below the table plane == dead
  ball_dead_x_lo_hope: float = -1.2   # past the robot
  ball_dead_x_hi_hope: float = 3.4


@dataclass
class SimCfg:
  """Plant/scene knobs.  Defaults are the shipped, calibrated values."""

  nworld: int = 4096
  cone: str = "elliptic"              # calibrated tangential channel
  add_pairs: bool = True
  njmax: int = 572                    # measured, not the warp heuristic (64)
  nconmax: int = 128
  ball_spawn_hope: tuple = (2.0, -0.7625, 0.68)


# ==========================================================================
# Small math helpers (kept local so this file has no mjlab-internal deps).
# ==========================================================================


def quat_rotate_inverse(q, v):
  """Rotate ``v`` (world) into the frame of quaternion ``q`` (w, x, y, z)."""
  import torch

  q_w = q[:, 0:1]
  q_vec = q[:, 1:4]
  a = v * (2.0 * q_w * q_w - 1.0)
  b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
  c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
  return a - b + c


# ==========================================================================
# The environment.
# ==========================================================================


class A3ReadyBallVecEnv:
  """rsl-rl ``VecEnv`` over the A3 court scene.

  Registered as a duck-typed VecEnv rather than a subclass so that importing
  this module never depends on rsl_rl being importable (the --analyze path
  runs anywhere).
  """

  def __init__(self, sim_cfg: SimCfg, task_cfg: TaskCfg, device: str,
               xml_path: Path | None = None,
               ready_pose_path: Path | None = None,
               seed: int = 0, count_contacts: bool = False) -> None:
    import mujoco
    import torch

    self.cfg = task_cfg
    self.sim_cfg = sim_cfg
    self.device = torch.device(device)
    self._torch = torch

    xml = xml_path or plant.default_xml()
    self.env = court.build_court_env(
      xml_path=xml,
      num_envs=sim_cfg.nworld,
      device=device,
      ball_pos_hope=sim_cfg.ball_spawn_hope,
      cone=sim_cfg.cone,
      add_pairs=sim_cfg.add_pairs,
      njmax=sim_cfg.njmax,
      nconmax=sim_cfg.nconmax,
    )
    self.sim = self.env.sim
    m = self.env.mj_model
    self.mj_model = m
    self.num_envs = int(self.sim.num_envs)
    self.decimation = int(self.env.decimation)
    self.step_dt = float(self.env.step_dt)
    self.physics_dt = float(self.env.physics_dt)
    self.max_episode_length = int(round(task_cfg.episode_length_s / self.step_dt))

    # ---- ready pose --------------------------------------------------
    rp = ready_pose_path or (_HERE / "ready_pose.json")
    if not Path(rp).is_file():
      rp = Path("/workspace/mjlab_lane/ready_pose.json")
    self.pose = court.load_ready_pose(Path(rp))
    qpos0, qvel0, idx = court.ready_qpos(self.env, self.pose)
    self.root_qadr = int(idx["root_qadr"])
    self.row_map_agrees = bool(idx["consistency"]["agree"])

    # root dof address: the free joint's dof address.
    root_jid = int(np.argmin(np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE,
                                      np.arange(m.njnt), m.njnt)))
    self.root_vadr = int(m.jnt_dofadr[root_jid])

    # ---- actuator wiring (vendor PD, computed outside the plant) ------
    kp_np, kd_np, q_adr_act, v_adr_act = plant._pd_wiring(self.env)
    self.num_actions = int(m.nu)
    jnt_of_act = m.actuator_trnid[:, 0].astype(int)
    jrange = m.jnt_range[jnt_of_act]

    T = lambda x, dt=torch.float32: torch.as_tensor(  # noqa: E731
      np.asarray(x), dtype=dt, device=self.device)
    self.kp = T(kp_np)
    self.kd = T(kd_np)
    self.q_adr_act = T(q_adr_act, torch.long)
    self.v_adr_act = T(v_adr_act, torch.long)
    # Contiguity lets us use slices instead of gathers in the 1 kHz inner loop.
    self._q_slice = (slice(int(q_adr_act[0]), int(q_adr_act[0]) + len(q_adr_act))
                     if np.all(np.diff(q_adr_act) == 1) else None)
    self._v_slice = (slice(int(v_adr_act[0]), int(v_adr_act[0]) + len(v_adr_act))
                     if np.all(np.diff(v_adr_act) == 1) else None)
    self.tau_lo = T(m.actuator_ctrlrange[:, 0])
    self.tau_hi = T(m.actuator_ctrlrange[:, 1])
    self.tau_scale = torch.maximum(self.tau_hi.abs(), self.tau_lo.abs())
    self.jnt_lo = T(jrange[:, 0])
    self.jnt_hi = T(jrange[:, 1])
    self.q_ready = T(qpos0[q_adr_act])
    if task_cfg.action_scale_mode == "vendor":
      # AGIBOT_A3_ACTION_SCALE = 0.25 * effort_limit_sim / stiffness, per joint.
      # effort_limit_sim is bit-identical to the MJCF ctrlrange (verified in the
      # plant step), and the kp here is the same VENDOR_KP table, so this is the
      # Isaac decoder reproduced from the compiled model rather than copied.
      self.act_scale = 0.25 * self.tau_hi / self.kp
    else:
      self.act_scale = torch.full_like(self.kp, float(task_cfg.action_scale))

    # ---- ball wiring ---------------------------------------------------
    bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, court.BALL_JOINT)
    assert bj >= 0, "ball free joint missing from the court model"
    self.b_q = int(m.jnt_qposadr[bj])
    self.b_v = int(m.jnt_dofadr[bj])
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot/right_racket")
    if sid < 0:
      sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_racket")
    assert sid >= 0, "right_racket site missing from the court model"
    self.racket_sid = int(sid)

    # ---- base reset state ----------------------------------------------
    self.qpos_init = T(qpos0)               # (nq,)
    self.qvel_init = T(qvel0)
    self.hope_to_scene = T(np.asarray(court.HOPE_TO_SCENE))
    self.ready_pelvis_z = float(qpos0[self.root_qadr + 2])

    # ---- serve recipe (validated narrow returner serve) -----------------
    sc = geom.ServeConfig.reachable_returner()
    self.serve_cfg = sc
    self.serve_pos_lo = T([sc.pos_x_range[0], sc.pos_y_range[0], sc.pos_z_range[0]])
    self.serve_pos_hi = T([sc.pos_x_range[1], sc.pos_y_range[1], sc.pos_z_range[1]])
    self.serve_vel_lo = T([sc.vel_x_range[0], sc.vel_y_range[0], sc.vel_z_range[0]])
    self.serve_vel_hi = T([sc.vel_x_range[1], sc.vel_y_range[1], sc.vel_z_range[1]])

    # ---- buffers ---------------------------------------------------------
    N = self.num_envs
    self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
    self.ball_age_buf = torch.zeros(N, dtype=torch.long, device=self.device)
    self.actions = torch.zeros(N, self.num_actions, device=self.device)
    self.last_actions = torch.zeros_like(self.actions)
    self.gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(N, 1)
    self.common_step_counter = 0
    self._ball_reserve_steps = int(round(task_cfg.ball_reserve_after_s / self.step_dt))

    # statistics accumulators (GPU-side; one sync per iteration)
    self._acc = {k: torch.zeros((), device=self.device) for k in (
      "ep_ret_sum", "ep_len_sum", "ep_cnt", "ep_min_d_sum",
      "term_fall_h", "term_tilt", "term_nonfinite", "term_timeout",
      "steps", "rew_sum", "reserves",
    )}
    self._rew_terms = ("alive", "pose", "upright", "height", "reach", "touch",
                       "action_rate", "joint_vel", "torque", "termination")
    for k in self._rew_terms:
      self._acc["r_" + k] = torch.zeros((), device=self.device)
    self._cur_ret = torch.zeros(N, device=self.device)
    self._cur_min_d = torch.full((N,), 1e3, device=self.device)

    self.generator = torch.Generator(device=self.device)
    self.generator.manual_seed(int(seed))

    # Optional per-substep contact probe.  OFF during the timed training runs
    # (it adds ~5 kernels over the naconmax array per physics step); ON for the
    # untimed policy evaluations, where "did the racket ever touch the ball"
    # is the whole question.
    self.count_contacts = bool(count_contacts)
    self._contact_ok = False
    if self.count_contacts:
      self._setup_contact_probe(mujoco)

    # first full reset
    self.reset()
    self.num_obs = int(self._obs_buf.shape[1])

  # ---- rsl-rl VecEnv surface ------------------------------------------

  @property
  def unwrapped(self):
    return self

  def get_observations(self):
    from tensordict import TensorDict

    return TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs])

  def close(self):
    return None

  # ---- internals --------------------------------------------------------

  def _setup_contact_probe(self, mujoco):
    """Wire a sync-free ball<->racket contact counter over mjwarp's contact array."""
    torch = self._torch
    m = self.mj_model
    d = self.sim.data
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    ball = gid(court.BALL_GEOM)
    rackets = [gid(n) for n in court.RACKET_GEOMS]
    table = gid("court_table_top")
    if ball < 0 or any(r < 0 for r in rackets):
      print("[a3_train_ppo] contact probe disabled: geoms not found")
      return
    try:
      contact = d.contact
      geom = contact.geom
      worldid = contact.worldid
      nacon = getattr(d, "nacon")
      self._con_geom, self._con_world = geom, worldid
      self._nacon = nacon
      self._naconmax = int(geom.shape[0])
      self._con_idx = torch.arange(self._naconmax, device=self.device)
    except Exception as exc:
      print(f"[a3_train_ppo] contact probe disabled: {exc!r}")
      return
    self._ball_gid = int(ball)
    self._racket_gids = torch.as_tensor(rackets, dtype=torch.long,
                                        device=self.device)
    self._table_gid = int(table)
    self._contact_ok = True
    self._acc["contact_ball_racket_substeps"] = torch.zeros((), device=self.device)
    self._acc["contact_ball_table_substeps"] = torch.zeros((), device=self.device)
    self._acc["ep_touched_racket"] = torch.zeros((), device=self.device)
    self._cur_touched = torch.zeros(self.num_envs, device=self.device)

  def _probe_contacts(self):
    torch = self._torch
    g = self._con_geom[:]
    valid = self._con_idx < self._nacon[0]
    g0, g1 = g[:, 0].long(), g[:, 1].long()
    is_ball = (g0 == self._ball_gid) | (g1 == self._ball_gid)
    other = torch.where(g0 == self._ball_gid, g1, g0)
    is_racket = (other.unsqueeze(-1) == self._racket_gids).any(dim=-1)
    hit = valid & is_ball & is_racket
    tab = valid & is_ball & (other == self._table_gid)
    self._acc["contact_ball_racket_substeps"] += hit.sum()
    self._acc["contact_ball_table_substeps"] += tab.sum()
    w = self._con_world[:].long().clamp_(0, self.num_envs - 1)
    self._cur_touched.scatter_add_(0, w, hit.float())

  def _rand(self, *shape, lo=0.0, hi=1.0):
    torch = self._torch
    r = torch.rand(*shape, device=self.device, generator=self.generator)
    return lo + (hi - lo) * r

  def _qpos_act(self):
    d = self.sim.data
    if self._q_slice is not None:
      return d.qpos[:, self._q_slice]
    return d.qpos[:, self.q_adr_act]

  def _qvel_act(self):
    d = self.sim.data
    if self._v_slice is not None:
      return d.qvel[:, self._v_slice]
    return d.qvel[:, self.v_adr_act]

  def _serve(self, ids):
    """Write a fresh serve into the ball's free joint for ``ids``."""
    torch = self._torch
    n = int(ids.numel()) if hasattr(ids, "numel") else len(ids)
    if n == 0:
      return
    u = torch.rand(n, 3, device=self.device, generator=self.generator)
    pos_hope = self.serve_pos_lo + (self.serve_pos_hi - self.serve_pos_lo) * u
    u2 = torch.rand(n, 3, device=self.device, generator=self.generator)
    vel = self.serve_vel_lo + (self.serve_vel_hi - self.serve_vel_lo) * u2
    pos = pos_hope + self.hope_to_scene
    d = self.sim.data
    bq, bv = self.b_q, self.b_v
    d.qpos[ids, bq:bq + 3] = pos
    d.qpos[ids, bq + 3:bq + 7] = torch.tensor(
      [1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4)
    d.qvel[ids, bv:bv + 3] = vel
    d.qvel[ids, bv + 3:bv + 6] = 0.0
    self.ball_age_buf[ids] = 0

  def _reset_idx(self, ids):
    torch = self._torch
    n = int(ids.numel())
    if n == 0:
      return
    # mjwarp's own masked reset first: clears qacc, warmstart, contacts, act.
    self.sim.reset(ids)
    d = self.sim.data
    cfg = self.cfg

    qpos = self.qpos_init.unsqueeze(0).repeat(n, 1)
    qvel = self.qvel_init.unsqueeze(0).repeat(n, 1)
    if cfg.reset_joint_noise_rad > 0:
      noise = self._rand(n, self.num_actions,
                         lo=-cfg.reset_joint_noise_rad,
                         hi=cfg.reset_joint_noise_rad)
      q = torch.clamp(self.q_ready.unsqueeze(0) + noise, self.jnt_lo, self.jnt_hi)
      qpos[:, self.q_adr_act] = q
    if cfg.reset_joint_vel_noise > 0:
      qvel[:, self.v_adr_act] = self._rand(
        n, self.num_actions, lo=-cfg.reset_joint_vel_noise,
        hi=cfg.reset_joint_vel_noise)
    rq = self.root_qadr
    if cfg.reset_root_xy_noise_m > 0:
      qpos[:, rq:rq + 2] += self._rand(n, 2, lo=-cfg.reset_root_xy_noise_m,
                                       hi=cfg.reset_root_xy_noise_m)
    if cfg.reset_root_yaw_noise_rad > 0:
      yaw = self._rand(n, lo=-cfg.reset_root_yaw_noise_rad,
                       hi=cfg.reset_root_yaw_noise_rad)
      # compose a yaw-only quaternion onto the (near-identity) ready quat
      cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
      q0 = qpos[:, rq + 3:rq + 7]
      w0, x0, y0, z0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
      qpos[:, rq + 3] = cy * w0 - sy * z0
      qpos[:, rq + 4] = cy * x0 - sy * y0
      qpos[:, rq + 5] = cy * y0 + sy * x0
      qpos[:, rq + 6] = cy * z0 + sy * w0

    d.qpos[ids] = qpos
    d.qvel[ids] = qvel
    d.ctrl[ids] = 0.0
    self._serve(ids)
    self.episode_length_buf[ids] = 0
    self.actions[ids] = 0.0
    self.last_actions[ids] = 0.0
    self._cur_ret[ids] = 0.0
    self._cur_min_d[ids] = 1e3

  def reset(self):
    torch = self._torch
    ids = torch.arange(self.num_envs, device=self.device)
    self._reset_idx(ids)
    self.sim.forward()
    self._compute_obs()
    return self.get_observations(), {}

  # ---- observation ------------------------------------------------------

  def _state(self):
    torch = self._torch
    d = self.sim.data
    rq, rv = self.root_qadr, self.root_vadr
    qpos, qvel = d.qpos, d.qvel
    base_quat = qpos[:, rq + 3:rq + 7]
    base_pos = qpos[:, rq:rq + 3]
    base_lin_w = qvel[:, rv:rv + 3]
    base_ang_b = qvel[:, rv + 3:rv + 6]          # free joint: already body-local
    proj_g = quat_rotate_inverse(base_quat, self.gravity_w)
    ball_pos = qpos[:, self.b_q:self.b_q + 3]
    ball_vel = qvel[:, self.b_v:self.b_v + 3]
    racket = d.site_xpos[:, self.racket_sid]
    return dict(qpos=qpos, qvel=qvel, base_quat=base_quat, base_pos=base_pos,
                base_lin_w=base_lin_w, base_ang_b=base_ang_b, proj_g=proj_g,
                ball_pos=ball_pos, ball_vel=ball_vel, racket=racket)

  def _compute_obs(self, st=None):
    torch = self._torch
    cfg = self.cfg
    st = st or self._state()
    q = st["base_quat"]
    lin_b = quat_rotate_inverse(q, st["base_lin_w"])
    d_ball_racket = st["ball_pos"] - st["racket"]
    d_ball_pelvis = st["ball_pos"] - st["base_pos"]
    d_racket_pelvis = st["racket"] - st["base_pos"]
    obs = torch.cat([
      lin_b * cfg.obs_scale_lin_vel,
      st["base_ang_b"] * cfg.obs_scale_ang_vel,
      st["proj_g"],
      self._qpos_act() - self.q_ready.unsqueeze(0),
      self._qvel_act() * cfg.obs_scale_joint_vel,
      self.actions,
      quat_rotate_inverse(q, d_ball_racket),
      quat_rotate_inverse(q, st["ball_vel"]) * cfg.obs_scale_ball_vel,
      quat_rotate_inverse(q, d_ball_pelvis),
      quat_rotate_inverse(q, d_racket_pelvis),
    ], dim=-1)
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    self._obs_buf = torch.clamp(obs, -cfg.obs_clip, cfg.obs_clip)
    return self._obs_buf

  # ---- step -------------------------------------------------------------

  def step(self, actions):
    torch = self._torch
    cfg = self.cfg
    d = self.sim.data

    actions = torch.clamp(actions.to(self.device), -cfg.action_clip, cfg.action_clip)
    self.last_actions = self.actions
    self.actions = actions
    q_des = torch.clamp(self.q_ready.unsqueeze(0) + self.act_scale * actions,
                        self.jnt_lo, self.jnt_hi)

    tau_sq = torch.zeros(self.num_envs, device=self.device)
    for _ in range(self.decimation):
      tau = self.kp * (q_des - self._qpos_act()) - self.kd * self._qvel_act()
      tau = torch.clamp(tau, self.tau_lo, self.tau_hi)
      d.ctrl[:] = tau
      tau_n = tau / self.tau_scale
      tau_sq += (tau_n * tau_n).mean(dim=-1)
      self.sim.step()
      if self._contact_ok:
        self._probe_contacts()
    tau_sq /= self.decimation

    self.episode_length_buf += 1
    self.ball_age_buf += 1
    self.common_step_counter += 1

    st = self._state()
    rew, terms = self._reward(st, tau_sq)
    terminated, reasons = self._terminate(st)
    truncated = self.episode_length_buf >= self.max_episode_length
    truncated = truncated & (~terminated)
    rew = rew + cfg.r_termination * terminated.float()
    terms["termination"] = cfg.r_termination * terminated.float()
    dones = terminated | truncated

    # -- ball housekeeping: re-serve a dead rally without ending the episode
    ball_hope = st["ball_pos"] - self.hope_to_scene
    dead = ((ball_hope[:, 2] < cfg.ball_dead_z_hope)
            | (ball_hope[:, 0] < cfg.ball_dead_x_lo_hope)
            | (ball_hope[:, 0] > cfg.ball_dead_x_hi_hope)
            | (self.ball_age_buf > self._ball_reserve_steps)
            | (~torch.isfinite(ball_hope).all(dim=1)))
    dead = dead & (~dones)
    reserve_ids = dead.nonzero(as_tuple=False).squeeze(-1)

    self._accumulate(rew, terms, dones, terminated, truncated, reasons,
                     st, reserve_ids.numel())

    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
      self._reset_idx(reset_ids)
    if reserve_ids.numel() > 0:
      self._serve(reserve_ids)
    if reset_ids.numel() > 0 or reserve_ids.numel() > 0:
      self.sim.forward()

    self._compute_obs()
    extras: dict[str, Any] = {"time_outs": truncated}
    return (self.get_observations(), rew, dones.long(), extras)

  # ---- reward / termination --------------------------------------------

  def _reward(self, st, tau_sq):
    torch = self._torch
    cfg = self.cfg
    qerr = self._qpos_act() - self.q_ready.unsqueeze(0)
    pose = torch.exp(-cfg.k_pose * (qerr * qerr).mean(dim=-1))
    upright = torch.exp(-cfg.k_upright * (1.0 + st["proj_g"][:, 2]) ** 2)
    dz = st["base_pos"][:, 2] - self.ready_pelvis_z
    height = torch.exp(-cfg.k_height * dz * dz)
    d = torch.linalg.norm(st["ball_pos"] - st["racket"], dim=-1)
    self._last_d = d
    reach = torch.exp(-d / cfg.reach_len_m)
    touch = torch.exp(-(d / cfg.touch_sigma_m) ** 2)
    da = self.actions - self.last_actions
    action_rate = (da * da).mean(dim=-1)
    qd = self._qvel_act()
    joint_vel = (qd * qd).mean(dim=-1)
    terms = {
      "alive": torch.full_like(pose, cfg.w_alive),
      "pose": cfg.w_pose * pose,
      "upright": cfg.w_upright * upright,
      "height": cfg.w_height * height,
      "reach": cfg.w_reach * reach,
      "touch": cfg.w_touch * touch,
      "action_rate": cfg.w_action_rate * action_rate,
      "joint_vel": cfg.w_joint_vel * joint_vel,
      "torque": cfg.w_torque * tau_sq,
    }
    rew = sum(terms.values())
    rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
    return rew, terms

  def _terminate(self, st):
    torch = self._torch
    cfg = self.cfg
    fall_h = st["base_pos"][:, 2] < cfg.min_pelvis_z
    tilt = st["proj_g"][:, 2] > cfg.max_tilt_proj_g
    finite = (torch.isfinite(st["qpos"]).all(dim=1)
              & torch.isfinite(st["qvel"]).all(dim=1))
    nonfinite = ~finite
    terminated = fall_h | tilt | nonfinite
    return terminated, {"fall_h": fall_h & finite, "tilt": tilt & finite & (~fall_h),
                        "nonfinite": nonfinite}

  # ---- statistics --------------------------------------------------------

  def _accumulate(self, rew, terms, dones, terminated, truncated, reasons, st,
                  n_reserve):
    torch = self._torch
    a = self._acc
    a["steps"] += self.num_envs
    a["rew_sum"] += rew.sum()
    for k in terms:
      a["r_" + k] += terms[k].sum()
    a["term_fall_h"] += reasons["fall_h"].sum()
    a["term_tilt"] += reasons["tilt"].sum()
    a["term_nonfinite"] += reasons["nonfinite"].sum()
    a["term_timeout"] += truncated.sum()
    a["reserves"] += n_reserve
    self._cur_ret += rew
    self._cur_min_d = torch.minimum(self._cur_min_d, self._last_d)
    df = dones.float()
    a["ep_ret_sum"] += (self._cur_ret * df).sum()
    a["ep_len_sum"] += (self.episode_length_buf.float() * df).sum()
    a["ep_min_d_sum"] += (torch.clamp(self._cur_min_d, max=10.0) * df).sum()
    a["ep_cnt"] += df.sum()
    if self._contact_ok:
      a["ep_touched_racket"] += ((self._cur_touched > 0).float() * df).sum()
      self._cur_touched *= (1.0 - df)

  def pop_stats(self) -> dict:
    torch = self._torch
    keys = list(self._acc.keys())
    vals = torch.stack([self._acc[k].float() for k in keys]).cpu().numpy()
    out = {k: float(v) for k, v in zip(keys, vals)}
    for k in keys:
      self._acc[k].zero_()
    steps = max(out["steps"], 1.0)
    ep = max(out["ep_cnt"], 1.0)
    stats = {
      "env_steps": out["steps"],
      "mean_step_reward": out["rew_sum"] / steps,
      "episodes_finished": out["ep_cnt"],
      "mean_episode_return": out["ep_ret_sum"] / ep,
      "mean_episode_length": out["ep_len_sum"] / ep,
      "mean_episode_min_racket_ball_dist_m": out["ep_min_d_sum"] / ep,
      "termination_rate_per_env_step": (
        out["term_fall_h"] + out["term_tilt"] + out["term_nonfinite"]) / steps,
      "timeout_rate_per_env_step": out["term_timeout"] / steps,
      "terminations": {
        "fall_height": out["term_fall_h"],
        "fall_tilt": out["term_tilt"],
        "nonfinite_state": out["term_nonfinite"],
        "timeout_truncation": out["term_timeout"],
      },
      "ball_reserves": out["reserves"],
      "reward_terms_mean": {k: out["r_" + k] / steps for k in self._rew_terms},
    }
    if self._contact_ok:
      stats["contact"] = {
        "ball_racket_contact_substeps": out["contact_ball_racket_substeps"],
        "ball_table_contact_substeps": out["contact_ball_table_substeps"],
        "episodes_with_a_racket_touch": out["ep_touched_racket"],
        "fraction_of_episodes_with_a_racket_touch": out["ep_touched_racket"] / ep,
      }
    return stats


# ==========================================================================
# Runner glue.
# ==========================================================================


def _rsl_rl_version() -> str:
  try:
    from importlib.metadata import version

    return version("rsl-rl-lib")
  except Exception:  # pragma: no cover
    return "?"


def build_agent_cfg(seed: int, iterations: int, num_steps_per_env: int,
                    experiment: str, entropy_coef: float = 0.002,
                    init_std: float = 1.0) -> dict:
  from mjlab.rl.config import (RslRlModelCfg, RslRlOnPolicyRunnerCfg,
                               RslRlPpoAlgorithmCfg)

  cfg = RslRlOnPolicyRunnerCfg(
    seed=seed,
    num_steps_per_env=num_steps_per_env,
    max_iterations=iterations,
    save_interval=max(iterations, 1),
    experiment_name=experiment,
    logger="tensorboard",
    obs_groups={"actor": ["policy"], "critic": ["policy"]},
    clip_actions=None,
    upload_model=False,
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution",
                        "init_std": init_std, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu",
                         obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      # rsl-rl's default 0.005 is a *per-dimension* entropy bonus.  With 31
      # action dims the bonus gradient is 31 * coef per unit log-std, which on
      # this reward scale outbid the whole stance-quality budget: the pilot's
      # policy std GREW 1.00 -> 1.16 over 60 iterations, i.e. PPO was paid to
      # stay noisy.  Measured, then halved-and-a-bit.
      entropy_coef=entropy_coef,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
  )
  return asdict(cfg)


def train(args) -> int:
  import torch
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  device = args.device
  sim_cfg = SimCfg(nworld=args.nworld, cone=args.cone,
                   add_pairs=not args.no_pairs,
                   njmax=args.njmax, nconmax=args.nconmax)
  task_cfg = TaskCfg(episode_length_s=args.episode_s,
                     action_scale_mode=args.action_scale_mode)

  t_build = time.perf_counter()
  env = A3ReadyBallVecEnv(sim_cfg, task_cfg, device=device, seed=args.seed)
  build_s = time.perf_counter() - t_build
  print(f"[a3_train_ppo] scene built in {build_s:.1f}s: nworld={env.num_envs} "
        f"nq={env.mj_model.nq} nu={env.num_actions} obs={env.num_obs} "
        f"decimation={env.decimation} step_dt={env.step_dt:.4f} "
        f"max_ep_len={env.max_episode_length}", flush=True)

  agent_cfg = build_agent_cfg(args.seed, args.iterations, args.num_steps_per_env,
                              args.experiment, entropy_coef=args.entropy_coef,
                              init_std=args.init_std)
  log_dir = Path(args.log_root) / args.experiment / f"{args.tag}_seed{args.seed}"
  log_dir.mkdir(parents=True, exist_ok=True)

  runner = MjlabOnPolicyRunner(env, agent_cfg, str(log_dir), device)

  # --- receipt hook: one JSON line per PPO iteration ---------------------
  records: list[dict] = []
  jsonl = Path(args.out_prefix + ".jsonl")
  jsonl.parent.mkdir(parents=True, exist_ok=True)
  jf = jsonl.open("w")
  orig_log = runner.logger.log
  collection_size = env.num_envs * args.num_steps_per_env
  t_start = time.perf_counter()

  def log_hook(**kw):
    stats = env.pop_stats()
    it = int(kw["it"])
    ct, lt = float(kw["collect_time"]), float(kw["learn_time"])
    rec = {
      "iter": it,
      "wall_s_total": time.perf_counter() - t_start,
      "collect_s": ct, "learn_s": lt, "iter_s": ct + lt,
      "env_steps_per_s": collection_size / max(ct + lt, 1e-9),
      "env_steps_per_s_collect_only": collection_size / max(ct, 1e-9),
      "physics_steps_per_s": collection_size * env.decimation / max(ct, 1e-9),
      "learning_rate": float(kw["learning_rate"]),
      "action_std": float(kw["action_std"].mean().item()),
      "losses": {k: float(v) for k, v in kw["loss_dict"].items()},
    }
    rec.update(stats)
    if runner.logger.rewbuffer:
      rec["rsl_rl_mean_reward"] = statistics.mean(runner.logger.rewbuffer)
      rec["rsl_rl_mean_ep_len"] = statistics.mean(runner.logger.lenbuffer)
    records.append(rec)
    jf.write(json.dumps(rec) + "\n")
    jf.flush()
    print(f"[it {it:4d}] R_ep={rec['mean_episode_return']:8.2f} "
          f"r_step={rec['mean_step_reward']:6.3f} "
          f"len={rec['mean_episode_length']:6.1f} "
          f"term/step={rec['termination_rate_per_env_step']:.4f} "
          f"minD={rec['mean_episode_min_racket_ball_dist_m']:.3f}m "
          f"fps={rec['env_steps_per_s']:.0f} "
          f"({ct:.2f}s+{lt:.2f}s)", flush=True)
    if args.rsl_rl_console:
      return orig_log(**kw)
    return None

  runner.logger.log = log_hook  # type: ignore[method-assign]

  smi_start = plant._nvidia_smi()
  runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
  smi_end = plant._nvidia_smi()
  jf.close()

  ckpt = log_dir / f"model_{args.iterations - 1}.pt"
  if not ckpt.is_file():
    runner.save(str(ckpt))

  summary = {
    "tag": args.tag,
    "seed": args.seed,
    "device": device,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "scene": {
      "nworld": env.num_envs, "cone": sim_cfg.cone, "pairs": sim_cfg.add_pairs,
      "njmax": int(getattr(env.sim.wp_data, "njmax", -1)),
      "naconmax": int(getattr(env.sim.wp_data, "naconmax", -1)),
      "cuda_graph": bool(env.sim.use_cuda_graph),
      "timestep": env.physics_dt, "decimation": env.decimation,
      "policy_dt": env.step_dt,
      "nq": int(env.mj_model.nq), "nv": int(env.mj_model.nv),
      "nu": int(env.mj_model.nu), "nbody": int(env.mj_model.nbody),
      "ngeom": int(env.mj_model.ngeom), "npair": int(env.mj_model.npair),
      "ready_row_map_agrees_with_json": env.row_map_agrees,
      "build_seconds": build_s,
    },
    "agent": {
      "num_steps_per_env": args.num_steps_per_env,
      "iterations": args.iterations,
      "obs_dim": env.num_obs, "action_dim": env.num_actions,
      "episode_length_s": task_cfg.episode_length_s,
      "max_episode_length_steps": env.max_episode_length,
      "entropy_coef": args.entropy_coef,
      "init_std": args.init_std,
      "action_scale_mode": task_cfg.action_scale_mode,
      "action_scale_rad_min_max": [float(env.act_scale.min()),
                                   float(env.act_scale.max())],
      "hidden_dims": [512, 256, 128],
      "obs_normalization": True,
      "rsl_rl_version": _rsl_rl_version(),
    },
    "task_cfg": asdict(task_cfg),
    "throughput": _throughput_summary(records, env, args),
    "learning": _learning_summary(records),
    "nvidia_smi_start": smi_start,
    "nvidia_smi_end": smi_end,
    "torch_cuda_mem_reserved_MiB": torch.cuda.memory_reserved() / 2**20,
    "torch_cuda_max_mem_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
    "log_dir": str(log_dir),
    "jsonl": str(jsonl),
    "records": records,
  }
  out = Path(args.out_prefix + ".json")
  out.write_text(json.dumps(summary, indent=2))
  print(json.dumps({k: summary[k] for k in
                    ("throughput", "learning")}, indent=2), flush=True)
  print(f"[a3_train_ppo] wrote {out}", flush=True)
  return 0


def evaluate(args) -> int:
  """Score a fixed policy on the same reward the training run optimizes.

  Two modes, and the pair is the point: ``--eval zero`` is the do-nothing
  policy (pure vendor PD holding split-ready, which the plant receipt says
  sags only 5 mm in 0.9 s), and ``--eval ckpt`` is a trained checkpoint,
  deterministic.  If the trained number does not beat the do-nothing number on
  the *same* reward, "the curve went up" would only mean PPO learned to stop
  shouting at its own actuators.
  """
  import torch
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  torch.manual_seed(args.seed)
  sim_cfg = SimCfg(nworld=args.nworld, cone=args.cone,
                   add_pairs=not args.no_pairs,
                   njmax=args.njmax, nconmax=args.nconmax)
  task_cfg = TaskCfg(episode_length_s=args.episode_s,
                     action_scale_mode=args.action_scale_mode)
  env = A3ReadyBallVecEnv(sim_cfg, task_cfg, device=args.device, seed=args.seed,
                          count_contacts=not args.no_contact_probe)

  policy = None
  if args.eval == "ckpt":
    assert args.eval_ckpt, "--eval ckpt needs --eval-ckpt PATH"
    agent_cfg = build_agent_cfg(args.seed, 1, args.num_steps_per_env,
                                args.experiment, entropy_coef=args.entropy_coef,
                                init_std=args.init_std)
    runner = MjlabOnPolicyRunner(env, agent_cfg, None, args.device)
    runner.load(args.eval_ckpt, map_location=args.device)
    policy = runner.get_inference_policy(args.device)

  obs = env.get_observations()
  zeros = torch.zeros(env.num_envs, env.num_actions, device=env.device)
  env.pop_stats()                      # discard the reset-only warmup
  t0 = time.perf_counter()
  with torch.inference_mode():
    for _ in range(args.eval_steps):
      act = zeros if policy is None else policy(obs)
      obs, _, _, _ = env.step(act)
  torch.cuda.synchronize()
  wall = time.perf_counter() - t0
  stats = env.pop_stats()
  out = {
    "mode": args.eval,
    "checkpoint": args.eval_ckpt,
    "nworld": env.num_envs,
    "policy_steps": args.eval_steps,
    "wall_s": wall,
    "env_steps_per_s": env.num_envs * args.eval_steps / wall,
    "stats": stats,
    "nvidia_smi": plant._nvidia_smi(),
  }
  Path(args.out_prefix + ".json").write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2), flush=True)
  return 0


def _throughput_summary(records, env, args) -> dict:
  if not records:
    return {}
  # iteration 0 pays CUDA-graph warmup / JIT; report both with and without.
  fps = [r["env_steps_per_s"] for r in records]
  it_s = [r["iter_s"] for r in records]
  col = [r["collect_s"] for r in records]
  lrn = [r["learn_s"] for r in records]
  tail = slice(1, None) if len(records) > 1 else slice(0, None)
  return {
    "env_steps_per_s_mean_incl_it0": float(np.mean(fps)),
    "env_steps_per_s_mean_excl_it0": float(np.mean(fps[tail])),
    "env_steps_per_s_median": float(np.median(fps[tail])),
    "physics_steps_per_s_collect_median": float(np.median(
      [r["physics_steps_per_s"] for r in records][tail])),
    "wall_s_per_iteration_mean_excl_it0": float(np.mean(it_s[tail])),
    "collect_s_mean_excl_it0": float(np.mean(col[tail])),
    "learn_s_mean_excl_it0": float(np.mean(lrn[tail])),
    "collect_fraction": float(np.mean(col[tail]) /
                              max(np.mean(it_s[tail]), 1e-9)),
    "total_wall_s": float(records[-1]["wall_s_total"]),
    "total_env_steps": float(sum(r["env_steps"] for r in records)),
  }


def _spearman(y) -> float:
  y = np.asarray(y, dtype=float)
  n = len(y)
  if n < 3:
    return float("nan")
  x = np.arange(n, dtype=float)
  rx = np.argsort(np.argsort(x)).astype(float)
  ry = np.argsort(np.argsort(y)).astype(float)
  rx -= rx.mean()
  ry -= ry.mean()
  den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
  return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _learning_summary(records) -> dict:
  if not records:
    return {}
  ret = [r["mean_episode_return"] for r in records]
  step_r = [r["mean_step_reward"] for r in records]
  ep_len = [r["mean_episode_length"] for r in records]
  term = [r["termination_rate_per_env_step"] for r in records]
  mind = [r["mean_episode_min_racket_ball_dist_m"] for r in records]
  k = max(1, len(records) // 10)
  return {
    "iterations": len(records),
    "mean_episode_return_first": float(np.mean(ret[:k])),
    "mean_episode_return_last": float(np.mean(ret[-k:])),
    "mean_episode_return_gain": float(np.mean(ret[-k:]) - np.mean(ret[:k])),
    "mean_step_reward_first": float(np.mean(step_r[:k])),
    "mean_step_reward_last": float(np.mean(step_r[-k:])),
    "mean_episode_length_first": float(np.mean(ep_len[:k])),
    "mean_episode_length_last": float(np.mean(ep_len[-k:])),
    "termination_rate_first": float(np.mean(term[:k])),
    "termination_rate_last": float(np.mean(term[-k:])),
    "min_racket_ball_dist_first_m": float(np.mean(mind[:k])),
    "min_racket_ball_dist_last_m": float(np.mean(mind[-k:])),
    "spearman_return_vs_iteration": _spearman(ret),
    "spearman_step_reward_vs_iteration": _spearman(step_r),
    "spearman_min_dist_vs_iteration": _spearman(mind),
    "monotone_rising": bool(np.mean(ret[-k:]) > np.mean(ret[:k])),
    "curve_mean_episode_return": [float(v) for v in ret],
    "curve_mean_step_reward": [float(v) for v in step_r],
    "curve_mean_episode_length": [float(v) for v in ep_len],
    "curve_termination_rate": [float(v) for v in term],
    "curve_min_racket_ball_dist_m": [float(v) for v in mind],
    "curve_action_std": [float(r["action_std"]) for r in records],
    "curve_reward_terms": {
      t: [float(r["reward_terms_mean"][t]) for r in records]
      for t in records[0]["reward_terms_mean"]
    },
    "reward_terms_first": {t: float(np.mean([r["reward_terms_mean"][t]
                                             for r in records[:k]]))
                           for t in records[0]["reward_terms_mean"]},
    "reward_terms_last": {t: float(np.mean([r["reward_terms_mean"][t]
                                            for r in records[-k:]]))
                          for t in records[0]["reward_terms_mean"]},
  }


# ==========================================================================
# N-seed band analysis (the only reproducibility claim warp supports).
# ==========================================================================


def analyze(paths, out_path) -> int:
  curves, names = [], []
  for p in paths:
    p = Path(p)
    if p.suffix == ".jsonl":
      recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    else:
      recs = json.loads(p.read_text())["records"]
    curves.append(recs)
    names.append(p.stem)
  n = min(len(c) for c in curves)
  band: dict[str, Any] = {"runs": names, "iterations_compared": n,
                          "n_seeds": len(curves)}
  for key in ("mean_episode_return", "mean_step_reward", "mean_episode_length",
              "termination_rate_per_env_step",
              "mean_episode_min_racket_ball_dist_m", "env_steps_per_s"):
    arr = np.array([[c[i][key] for i in range(n)] for c in curves], dtype=float)
    mean = arr.mean(axis=0)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    spread = hi - lo
    denom = np.maximum(np.abs(mean), 1e-9)
    k = max(1, n // 10)
    band[key] = {
      "per_seed_first": [float(a[:k].mean()) for a in arr],
      "per_seed_last": [float(a[-k:].mean()) for a in arr],
      "band_mean": [float(v) for v in mean],
      "band_lo": [float(v) for v in lo],
      "band_hi": [float(v) for v in hi],
      "abs_spread_mean": float(spread.mean()),
      "abs_spread_max": float(spread.max()),
      "rel_spread_mean_pct": float(100.0 * (spread / denom).mean()),
      "rel_spread_max_pct": float(100.0 * (spread / denom).max()),
      "final_values": [float(a[-1]) for a in arr],
      "final_abs_spread": float(spread[-1]),
      "final_rel_spread_pct": float(100.0 * spread[-1] / denom[-1]),
      "last_decile_band_overlaps_first_decile": bool(
        max(float(a[-k:].mean()) for a in arr) > min(float(a[:k].mean()) for a in arr)
        and min(float(a[-k:].mean()) for a in arr)
        < max(float(a[:k].mean()) for a in arr)),
    }
  ret = band["mean_episode_return"]
  band["verdict"] = {
    "all_seeds_rose": all(l > f for f, l in zip(ret["per_seed_first"],
                                                ret["per_seed_last"])),
    "learning_gain_vs_seed_spread": float(
      (np.mean(ret["per_seed_last"]) - np.mean(ret["per_seed_first"]))
      / max(ret["abs_spread_mean"], 1e-9)),
  }
  Path(out_path).write_text(json.dumps(band, indent=2))
  print(json.dumps({k: v for k, v in band.items()
                    if k in ("runs", "iterations_compared", "verdict")}, indent=2))
  print(json.dumps({"mean_episode_return": {
    k: v for k, v in band["mean_episode_return"].items()
    if not k.startswith("band_")}}, indent=2))
  print(f"[a3_train_ppo] wrote {out_path}")
  return 0


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--nworld", type=int, default=4096)
  p.add_argument("--iterations", type=int, default=60)
  p.add_argument("--num-steps-per-env", type=int, default=24)
  p.add_argument("--episode-s", type=float, default=3.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--cone", choices=("pyramidal", "elliptic"), default="elliptic")
  p.add_argument("--no-pairs", action="store_true")
  p.add_argument("--njmax", type=int, default=572)
  p.add_argument("--nconmax", type=int, default=128)
  p.add_argument("--action-scale-mode", choices=("flat", "vendor"), default="flat",
                 help="'vendor' = the Isaac/deploy per-joint 0.25*effort/kp decoder")
  p.add_argument("--entropy-coef", type=float, default=0.002)
  p.add_argument("--init-std", type=float, default=1.0)
  p.add_argument("--experiment", default="a3_court_ppo")
  p.add_argument("--tag", default="run")
  p.add_argument("--log-root", default="/workspace/mjlab_lane/logs")
  p.add_argument("--out-prefix", default=None)
  p.add_argument("--rsl-rl-console", action="store_true",
                 help="also print rsl-rl's own iteration block")
  p.add_argument("--smoke", action="store_true",
                 help="64 worlds / 3 iterations, for wiring checks")
  p.add_argument("--analyze", nargs="+", default=None,
                 help="two or more run .jsonl/.json files -> N-seed band")
  p.add_argument("--out", default="BAND.json")
  p.add_argument("--eval", choices=("zero", "ckpt"), default=None,
                 help="score a fixed policy instead of training")
  p.add_argument("--eval-ckpt", default=None)
  p.add_argument("--eval-steps", type=int, default=750)
  p.add_argument("--no-contact-probe", action="store_true",
                 help="eval only: skip the per-substep ball<->racket contact count")
  a = p.parse_args(argv)

  if a.analyze:
    return analyze(a.analyze, a.out)

  if a.smoke:
    a.nworld, a.iterations = 64, 3
    a.tag = a.tag if a.tag != "run" else "smoke"
  if a.out_prefix is None:
    a.out_prefix = f"/workspace/mjlab_lane/TRAIN_{a.tag}_seed{a.seed}"
  if a.eval:
    return evaluate(a)
  return train(a)


if __name__ == "__main__":
  raise SystemExit(main())
