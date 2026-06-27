"""Lightweight MuJoCo sim-to-sim runner for the HOPE A3 ping-pong BeyondMimic/HOPE ONNX policy.

Runs the EXACT policy contract used in Isaac training, but in MuJoCo (a different physics engine),
to verify the exported ONNX before hardware. NO retraining, NO reward changes, NO target-sampling
changes. This deliberately does NOT use the official Agibot 1570D HITTER-tokenizer C++ harness and
does NOT convert the policy to 29D — it runs the 31D BeyondMimic ONNX as-is.

=============================================================================================
POLICY CONTRACT (verified against the exported ONNX metadata + checkpoint weights, 2026-06-27)
=============================================================================================
ONNX (logs/.../basecouple03_resume/exported/policy.onnx):
  inputs : obs[1,180] (float32), time_step[1,1] (float32)
  outputs: actions[1,31], joint_pos[1,31], joint_vel[1,31],
           body_pos_w[1,14,3], body_quat_w[1,14,4], body_lin_vel_w[1,14,3], body_ang_vel_w[1,14,3]
  -> outputs[1:] are the REFERENCE motion (the BeyondMimic clip) indexed by `time_step`. We use them
     as the single source of truth for the reference command + anchor (NO npz body-order guessing).

ACTOR OBSERVATION = 180D, concatenated in THIS order (verified: actor.0.weight is (512,180)):
   1. command            62  = cat(ref_joint_pos[31], ref_joint_vel[31])   (generated_commands("motion"))
   2. motion_anchor_pos_b  3  ref torso pose in the ROBOT torso (anchor) frame
   3. motion_anchor_ori_b  6  same, orientation as first 2 columns of the rotation matrix (6D rot rep)
   4. base_ang_vel         3  pelvis angular velocity in the PELVIS BODY frame (IMU gyro)
   5. joint_pos           31  q - default_joint_pos   (Isaac articulation order)
   6. joint_vel           31  qdot                     (Isaac articulation order)
   7. actions             31  last raw policy action (the 31-vec from the previous ONNX call)
   8. projected_gravity    3  gravity unit vec in the PELVIS BODY frame (IMU)
   9. base_target_pos_b    2  desired base XY in the yaw-heading base frame
  10. racket_target_pos_b  3  desired racket pos in the yaw-heading base frame
  11. racket_target_vel_w  3  desired racket velocity in WORLD frame
  12. time_to_strike       1  seconds until the strike frame
  13. swing_type           1  +1 forehand / -1 backhand
  (NOTE: the user's brief said 129D/command=13/ori=4 — that is WRONG vs the actual weights.
   The real contract is 180D with command=62 and motion_anchor_ori_b=6. We build the real one.)

FRAME CONVENTIONS (critical — these are NOT all world frame):
  * world frame ........ MuJoCo global frame. env origin = (0,0,0) (single env, matches Isaac env 0).
  * pelvis BODY frame .. base_ang_vel and projected_gravity are expressed here (IMU-measurable).
                         base_ang_vel_b = R_pelvis^T @ omega_world ; proj_grav_b = R_pelvis^T @ [0,0,-1].
  * yaw-heading base ... racket_target_pos_b / base_target_pos_b use yaw_quat(pelvis_quat) (Z-rotation
                         ONLY), then quat_rotate_inverse. So they are heading-relative, not full-3D.
  * anchor (torso) frame motion_anchor_* express the REFERENCE torso pose relative to the ROBOT torso.

JOINT ORDER:
  * Isaac articulation order = ONNX metadata `joint_names` (31, interleaved L/R; e.g. left_hip_pitch,
    right_hip_pitch, waist_yaw, ...). The obs, action, default_joint_pos, action_scale, kp, kd are ALL
    in this order. We work internally in this order.
  * MuJoCo qpos/qvel/actuators are in MJCF declaration order (waist, head, L-arm, R-arm, L-leg, R-leg).
  * We build a name-based permutation between the two and only convert at the sim boundary.

ACTION DECODE (Isaac articulation order):  target_q = default_joint_pos + raw_action * action_scale
  (use_default_offset=True; no action clipping in training — clip_actions=null). default_joint_pos,
  action_scale, kp(stiffness), kd(damping) are all read from the ONNX metadata.

PD CONTROL (per physics substep, target_q held across the control step / decimation):
  torque = kp * (target_q - q) - kd * qdot   then clipped to the actuator effort limits.
  The PD gains ARE the official Agibot a3_kps/a3_kds (the Isaac config transcribes them). To avoid
  DOUBLE damping we zero the MJCF passive joint damping + frictionloss on the 31 actuated DOFs (Isaac's
  ImplicitActuator models damping via kd only); armature is kept (physical, present in both). Use
  --keep-passive to leave the MJCF damping/frictionloss in (a harder, less faithful test).

CONTROL FREQUENCY: 50 Hz. Isaac used sim_dt=0.005 * decimation=4. We mirror that (--sim-dt/--decimation).

REFERENCE-STATE-INIT (matches Isaac): each new swing (episode reset OR clip wrap) teleports the robot
  to the reference pose at that clip's first frame (root pose from ref body 0 = pelvis_link, joints from
  ref_joint_pos). Isaac does exactly this in MotionCommand._resample_command, so an episode contains
  several teleport-initialised swings until a termination (fall) or the 10 s timeout.

OBSERVATION NOISE: training adds small uniform obs corruption; deployment/sim-to-sim feeds CLEAN obs
  (the ONNX is deterministic). We feed clean obs and document it; sensor noise is a separate concern.

=============================================================================================
USAGE (motion env that has mujoco + onnxruntime, e.g. hope-motion-py310):
  python scripts/mujoco_eval_onnx.py \
      --onnx logs/rsl_rl/agibot_a3_hope/2026-06-27_18-14-06_basecouple03_resume/exported/policy.onnx \
      --mjcf agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
      --motion-files logs/rsl_rl/eval_motion/fh.npz logs/rsl_rl/eval_motion/bh.npz \
      --noise-scales 0.0 0.05 --steps 1200 --seed 0
  (paths are relative to the whole_body_tracking/ dir; --mjcf is relative to the repo root by default)
=============================================================================================
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

# -------------------------------------------------------------------------------------------------
# Constants pulled from the verified training config (HOPEPingPong.yaml uniform-mode + RacketTargetCmd).
# These are the RUNTIME values (YAML overrides applied), not the python cfg defaults.
# -------------------------------------------------------------------------------------------------
ANCHOR_BODY = "torso_Link"               # ONNX metadata anchor_body_name
# 14 tracked bodies, in ONNX metadata `body_names` order (index 0 = root pelvis, 7 = torso anchor).
TRACKED_BODIES = [
    "pelvis_link", "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
    "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link", "torso_Link",
    "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
    "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link",
]
ANCHOR_TRACKED_IDX = TRACKED_BODIES.index(ANCHOR_BODY)   # 7
ROOT_TRACKED_IDX = 0                                     # pelvis_link
# End-effector bodies for the ee_body_pos termination (indices INTO the 14 tracked list).
EE_TRACKED_IDX = [TRACKED_BODIES.index(n) for n in
                  ["left_ankle_roll_Link", "right_ankle_roll_Link",
                   "left_wrist_yaw_Link", "right_wrist_yaw_Link"]]
FEET_BODIES = ["left_ankle_roll_Link", "right_ankle_roll_Link"]

# Termination thresholds (tracking_env_cfg.TerminationsCfg).
TERM_ANCHOR_POS_Z = 0.25     # |ref_torso_z - robot_torso_z| > 0.25  -> fall
TERM_ANCHOR_ORI = 0.8        # |proj_grav_z(ref) - proj_grav_z(robot)| > 0.8 -> fall
TERM_EE_POS_Z = 0.25         # any ee |z(ref_relative) - z(robot)| > 0.25 -> fall

# RacketTargetCommand uniform-mode sampling (HOPEPingPong.yaml overrides).
RACKET_POS_X_RANGE = (0.40, 0.40)        # fixed strike plane (x), relative to env origin
RACKET_POS_Y_ABS_RANGE = (0.05, 0.45)    # |y|; sign set per clip
RACKET_POS_Z_RANGE = (0.70, 1.05)
RACKET_VEL_X_RANGE = (1.5, 3.5)
RACKET_VEL_Y_RANGE = (-1.0, 1.0)
RACKET_VEL_Z_RANGE = (0.0, 1.5)
BASE_TARGET_X_RANGE = (-0.10, 0.10)
BASE_TARGET_Y_RANGE = (-0.10, 0.10)
BASE_COUPLE_BLEND = 0.3                  # weak base->racket Y coupling
BASE_COUPLE_MAX_OFFSET = 0.20
FOREHAND_ON_NEGATIVE_Y = True            # forehand (clip 0) target on -y
STRIKE_PHASE_PER_CLIP = (0.36, 0.74)     # forehand / backhand contact phase
STRIKE_WINDOW_S = 0.12
# Strike-success acceptance thresholds (RacketTargetCommandCfg) — identical to Isaac's exact metric.
STRIKE_POS_THRESH = 0.075                 # m   strike_success_pos_thresh
STRIKE_VEL_THRESH = 0.5                   # m/s strike_success_vel_thresh  (||actual - target|| 3-vec)
STRIKE_NORMAL_THRESH_DEG = 15.0           # deg strike_success_normal_thresh_deg  (acos(dot) of unit normals)
# Racket face normal = local +Y axis of the racket frame (== wrist frame; mount_quat is identity).
MOUNT_NORMAL_AXIS = 1
MOUNT_NORMAL_SIGN = 1.0
WRIST_TRACKED_IDX = TRACKED_BODIES.index("right_wrist_yaw_Link")   # 13; racket frame == this body's frame
CLIP_NAMES = {0: "forehand", 1: "backhand"}


# =================================================================================================
# Quaternion / frame math (numpy, scalar-first w,x,y,z — MuJoCo and IsaacLab both use this order).
# =================================================================================================
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / (w * w + x * x + y * y + z * z)


def mat_from_quat(q):
    """3x3 rotation matrix R (world<-body) from quaternion (w,x,y,z). Matches IsaacLab matrix_from_quat."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_rotate(q, v):
    return mat_from_quat(q) @ v


def quat_rotate_inverse(q, v):
    return mat_from_quat(q).T @ v


def yaw_quat(q):
    """Quaternion of the yaw-only (Z) rotation of q. Matches IsaacLab yaw_quat."""
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def subtract_frame_transforms(t01, q01, t02, q02):
    """Pose of frame 2 expressed in frame 1. Returns (t12, q12). Matches IsaacLab."""
    q10 = quat_inv(q01)
    t12 = quat_rotate(q10, t02 - t01)
    q12 = quat_mul(q10, q02)
    return t12, q12


# =================================================================================================
# ONNX policy wrapper
# =================================================================================================
class OnnxPolicy:
    def __init__(self, onnx_path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ins = {i.name: i.shape for i in self.sess.get_inputs()}
        outs = [o.name for o in self.sess.get_outputs()]
        assert "obs" in ins and "time_step" in ins, f"unexpected ONNX inputs: {ins}"
        self.obs_dim = int(ins["obs"][1])
        assert self.obs_dim == 180, f"expected obs dim 180, got {self.obs_dim}"
        self.out_names = outs
        md = self.sess.get_modelmeta().custom_metadata_map
        # --- metadata (FAIL LOUDLY if anything is missing) -------------------------------------
        required = ["joint_names", "default_joint_pos", "action_scale", "joint_stiffness",
                    "joint_damping", "body_names", "anchor_body_name", "observation_names"]
        missing = [k for k in required if k not in md or not md[k].strip()]
        if missing:
            raise SystemExit(
                "[FATAL] ONNX is missing required metadata keys: " + ", ".join(missing) +
                "\nThis runner needs them to map joints/gains. Re-export with attach_onnx_metadata "
                "(scripts/play.py) so they are baked in.")
        self.joint_names = md["joint_names"].split(",")
        self.default_q = np.array([float(v) for v in md["default_joint_pos"].split(",")], np.float64)
        self.action_scale = np.array([float(v) for v in md["action_scale"].split(",")], np.float64)
        self.kp = np.array([float(v) for v in md["joint_stiffness"].split(",")], np.float64)
        self.kd = np.array([float(v) for v in md["joint_damping"].split(",")], np.float64)
        self.body_names = md["body_names"].split(",")
        n = len(self.joint_names)
        assert n == 31 and self.default_q.shape == (31,) and self.action_scale.shape == (31,), \
            f"expected 31 joints, got {n}"
        assert self.body_names == TRACKED_BODIES, \
            f"ONNX body_names != expected tracked order:\n {self.body_names}\n {TRACKED_BODIES}"

    def refs(self, time_step):
        """Reference motion at `time_step` (obs-independent). Returns dict of arrays in metadata order."""
        obs = np.zeros((1, self.obs_dim), np.float32)
        ts = np.array([[float(time_step)]], np.float32)
        o = self.sess.run(None, {"obs": obs, "time_step": ts})
        names = self.out_names
        d = {names[i]: o[i][0] for i in range(len(names))}
        return d

    def action(self, obs, time_step):
        ts = np.array([[float(time_step)]], np.float32)
        o = self.sess.run(None, {"obs": obs[None].astype(np.float32), "time_step": ts})
        return o[0][0].astype(np.float64)    # mean action (31,), Isaac articulation order


# =================================================================================================
# MuJoCo robot wrapper (handles the articulation<->MJCF joint permutation + FK reads)
# =================================================================================================
class MujocoRobot:
    def __init__(self, mjcf_path, joint_names, body_names, sim_dt, keep_passive, pd_mode, kd_for_implicit=None):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.opt.timestep = sim_dt
        self.pd_mode = pd_mode      # "explicit" (torque kp*e - kd*qd) or "implicit" (kp torque + kd as
                                    #  passive damping integrated by MuJoCo's implicitfast integrator,
                                    #  matching Isaac's ImplicitActuator semantics).
        self.data = mujoco.MjData(self.model)

        def jid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        def bid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        def aid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # Per actuated joint (in ARTICULATION order): qpos addr, qvel/dof addr, actuator id.
        self.qadr = np.array([self.model.jnt_qposadr[jid(n)] for n in joint_names], int)
        self.vadr = np.array([self.model.jnt_dofadr[jid(n)] for n in joint_names], int)
        self.act_id = np.array([aid(n + "_motor") for n in joint_names], int)
        assert (self.act_id >= 0).all(), "missing <joint>_motor actuator(s) in MJCF"
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_id, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_id, 1].copy()

        # Body ids for the 14 tracked bodies, the pelvis (free base) and torso (anchor).
        self.tracked_bid = np.array([bid(n) for n in body_names], int)
        self.pelvis_bid = bid("pelvis_link")
        self.torso_bid = bid(ANCHOR_BODY)
        self.racket_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_racket")
        self.feet_bid = [bid(n) for n in FEET_BODIES]
        self.feet_geoms = {g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] in self.feet_bid}

        # Zero MJCF passive damping + frictionloss on the 31 actuated DOFs (kd provides damping;
        # avoids double-damping vs Isaac's ImplicitActuator). Armature is kept (physical).
        if not keep_passive:
            self.model.dof_damping[self.vadr] = 0.0
            self.model.dof_frictionloss[self.vadr] = 0.0
        if pd_mode == "implicit":
            # Match Isaac's ImplicitActuator: the kd damping is integrated IMPLICITLY (stable + no
            # under-shoot of fast swings at a 5 ms step). Put kd into the passive joint damping and
            # use MuJoCo's implicitfast integrator; the control torque then applies kp only.
            assert kd_for_implicit is not None
            self.model.dof_damping[self.vadr] = kd_for_implicit
            self.model.opt.integrator = int(self.mj.mjtIntegrator.mjINT_IMPLICITFAST)

    # --- state reads (all returned in ARTICULATION order or world/body frame as named) ----------
    def q_artic(self):
        return self.data.qpos[self.qadr].copy()

    def qd_artic(self):
        return self.data.qvel[self.vadr].copy()

    def body_pos(self, bid):
        return self.data.xpos[bid].copy()

    def body_quat(self, bid):
        return self.data.xquat[bid].copy()

    def pelvis_ang_vel_body(self):
        """Pelvis angular velocity in the pelvis BODY frame (== IMU gyro). Uses mj_objectVelocity."""
        res = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY,
                                  self.pelvis_bid, res, 1)  # flg_local=1 -> body frame
        return res[:3].copy()   # [angular(3), linear(3)] -> angular

    def projected_gravity_body(self):
        R = self.data.xmat[self.pelvis_bid].reshape(3, 3)   # world<-body
        return R.T @ np.array([0.0, 0.0, -1.0])

    def tracked_pos(self):
        return self.data.xpos[self.tracked_bid].copy()      # (14,3) world

    def racket_pos(self):
        return self.data.site_xpos[self.racket_site].copy()  # world

    def racket_lin_vel_w(self):
        # World-frame linear velocity of the racket site (== pingpang_red_Link origin; coincident with
        # Isaac's racket body, whose data.body_lin_vel_w is the analytic rigid-body origin velocity).
        res = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_SITE,
                                  self.racket_site, res, 0)  # flg_local=0 -> world frame; [ang(3), lin(3)]
        return res[3:6].copy()

    def racket_normal_w(self):
        # Actual racket face normal in world = local +Y axis of the racket(=wrist) frame.
        # site has identity orientation rel. to the wrist, so site_xmat == wrist world rotation.
        R = self.data.site_xmat[self.racket_site].reshape(3, 3)
        return R[:, MOUNT_NORMAL_AXIS] * MOUNT_NORMAL_SIGN

    def foot_contact_frac(self):
        ncon = self.data.ncon
        feet_in_contact = set()
        for i in range(ncon):
            c = self.data.contact[i]
            for g in (c.geom1, c.geom2):
                if g in self.feet_geoms:
                    feet_in_contact.add(self.model.geom_bodyid[g])
        return len(feet_in_contact) / max(len(self.feet_bid), 1)

    def reset_to_reference(self, root_pos, root_quat, root_lin_w, root_ang_w, q_artic):
        """Reference-state-init: teleport base + joints to the reference pose/vel (world frame inputs)."""
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat
        self.data.qpos[self.qadr] = q_artic
        # MuJoCo free-joint qvel: linear in WORLD frame, angular in the BODY frame.
        R = mat_from_quat(root_quat)
        self.data.qvel[0:3] = root_lin_w
        self.data.qvel[3:6] = R.T @ root_ang_w
        self.data.qvel[self.vadr] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def apply_pd_and_step(self, target_q_artic, kp, kd, decimation):
        """Hold target_q across `decimation` physics substeps, recomputing PD torque each substep.
        explicit: tau = kp*(tgt-q) - kd*qd (full PD as motor force).
        implicit: tau = kp*(tgt-q) only; kd is the passive joint damping integrated by implicitfast."""
        for _ in range(decimation):
            q = self.data.qpos[self.qadr]
            qd = self.data.qvel[self.vadr]
            tau = kp * (target_q_artic - q)
            if self.pd_mode == "explicit":
                tau = tau - kd * qd
            tau = np.clip(tau, self.ctrl_lo, self.ctrl_hi)
            self.data.ctrl[self.act_id] = tau
            self.mj.mj_step(self.model, self.data)
        return tau   # last substep torque (for logging)


# =================================================================================================
# RacketTargetCommand port (uniform mode, unified 2-clip) — only the obs-feeding quantities.
# =================================================================================================
class RacketCommand:
    def __init__(self, seg_start, seg_len, step_dt, rng, target_normal_per_clip, origin=np.zeros(3)):
        self.seg_start = seg_start          # (num_clips,)
        self.seg_len = seg_len
        self.step_dt = step_dt
        self.rng = rng
        self.origin = origin                # env origin (world). Single env -> (0,0,0), like Isaac env 0.
        # Per-clip target paddle normal: the imitated swing's reference face normal at strike (unified
        # uniform mode uses this, NOT a velocity-derived normal). Precomputed from the ref wrist quat.
        self.target_normal_per_clip = target_normal_per_clip
        # state
        self.racket_target_pos_w = np.zeros(3)
        self.racket_target_vel_w = np.zeros(3)
        self.racket_target_normal_w = np.array([0.0, 1.0, 0.0])
        self.base_target_pos_w = np.zeros(2)
        self.swing_sign = 1.0
        self.time_to_strike = 0.0

    def _u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def resample(self, clip_id):
        """New swing: sample racket target (pos/vel), base target, swing sign — matches uniform mode."""
        o = self.origin
        # racket target position (world): fixed x-plane; |y| per clip sign; z range.
        px = o[0] + self._u(*RACKET_POS_X_RANGE)
        ymag = self._u(*RACKET_POS_Y_ABS_RANGE)
        fh_sign = -1.0 if FOREHAND_ON_NEGATIVE_Y else 1.0
        sign = fh_sign if clip_id == 0 else -fh_sign      # forehand clip0 on -y, backhand clip1 on +y
        py = o[1] + sign * ymag
        pz = o[2] + self._u(*RACKET_POS_Z_RANGE)
        self.racket_target_pos_w = np.array([px, py, pz])
        # racket target velocity (world): independent box sample.
        self.racket_target_vel_w = np.array([self._u(*RACKET_VEL_X_RANGE),
                                             self._u(*RACKET_VEL_Y_RANGE),
                                             self._u(*RACKET_VEL_Z_RANGE)])
        # swing sign: clip 0 -> +1 (forehand), clip 1 -> -1 (backhand).
        self.swing_sign = 1.0 if clip_id == 0 else -1.0
        # target paddle normal = the per-clip reference face normal at strike (unified uniform mode).
        self.racket_target_normal_w = self.target_normal_per_clip[clip_id]
        # base target XY (world): spawn + weak Y coupling to racket target + small jitter.
        base_xy = o[:2].copy()
        racket_y_off = self.racket_target_pos_w[1] - o[1]
        base_xy[1] += float(np.clip(BASE_COUPLE_BLEND * racket_y_off,
                                    -BASE_COUPLE_MAX_OFFSET, BASE_COUPLE_MAX_OFFSET))
        base_xy[0] += self._u(*BASE_TARGET_X_RANGE)
        base_xy[1] += self._u(*BASE_TARGET_Y_RANGE)
        self.base_target_pos_w = base_xy

    def update_strike_timing(self, clip_id, time_step):
        seg_start = self.seg_start[clip_id]
        seg_len = self.seg_len[clip_id]
        phase = STRIKE_PHASE_PER_CLIP[clip_id]
        strike_step = seg_start + int(round(phase * (seg_len - 1)))
        self.time_to_strike = (strike_step - time_step) * self.step_dt

    # --- base-relative observation projections (yaw-heading base frame) -------------------------
    def racket_target_pos_b(self, base_pos_w, base_quat_w):
        return quat_rotate_inverse(yaw_quat(base_quat_w), self.racket_target_pos_w - base_pos_w)

    def base_target_pos_b(self, base_pos_w, base_quat_w):
        delta = np.array([self.base_target_pos_w[0] - base_pos_w[0],
                          self.base_target_pos_w[1] - base_pos_w[1], 0.0])
        return quat_rotate_inverse(yaw_quat(base_quat_w), delta)[:2]


# =================================================================================================
# Observation builder (180D, exact training order)
# =================================================================================================
def build_obs(refs, robot: MujocoRobot, racket: RacketCommand, last_action, default_q):
    # robot base (pelvis = root) world pose
    base_pos_w = robot.body_pos(robot.pelvis_bid)
    base_quat_w = robot.body_quat(robot.pelvis_bid)
    # robot anchor (torso) world pose
    robot_anchor_pos_w = robot.body_pos(robot.torso_bid)
    robot_anchor_quat_w = robot.body_quat(robot.torso_bid)
    # reference anchor (torso) world pose, from ONNX side-outputs (body_names order)
    ref_anchor_pos_w = refs["body_pos_w"][ANCHOR_TRACKED_IDX]
    ref_anchor_quat_w = refs["body_quat_w"][ANCHOR_TRACKED_IDX]

    # 1. command (62): reference joint pos + vel
    command = np.concatenate([refs["joint_pos"], refs["joint_vel"]])
    # 2/3. motion_anchor_pos_b (3) + ori_b (6): ref anchor expressed in robot anchor frame
    pos_b, ori_q = subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w,
                                             ref_anchor_pos_w, ref_anchor_quat_w)
    ori_b6 = mat_from_quat(ori_q)[:, :2].reshape(-1)     # first 2 columns -> [R00,R01,R10,R11,R20,R21]
    # 4. base_ang_vel (3) body frame
    base_ang_vel = robot.pelvis_ang_vel_body()
    # 5/6. joint pos/vel rel (31 each), articulation order
    q = robot.q_artic()
    qd = robot.qd_artic()
    joint_pos_rel = q - default_q
    joint_vel_rel = qd
    # 7. last action (31)
    # 8. projected gravity (3) body frame
    proj_grav = robot.projected_gravity_body()
    # 9. base_target_pos_b (2)
    base_tgt_b = racket.base_target_pos_b(base_pos_w, base_quat_w)
    # 10. racket_target_pos_b (3)
    racket_tgt_b = racket.racket_target_pos_b(base_pos_w, base_quat_w)
    # 11. racket_target_vel_w (3)
    racket_vel_w = racket.racket_target_vel_w
    # 12. time_to_strike (1)
    tts = np.array([racket.time_to_strike])
    # 13. swing_type (1)
    swing = np.array([racket.swing_sign])

    obs = np.concatenate([
        command, pos_b, ori_b6, base_ang_vel, joint_pos_rel, joint_vel_rel,
        last_action, proj_grav, base_tgt_b, racket_tgt_b, racket_vel_w, tts, swing,
    ]).astype(np.float64)
    assert obs.shape == (180,), f"obs dim {obs.shape} != 180"
    return obs, base_quat_w, robot_anchor_pos_w, robot_anchor_quat_w, ref_anchor_pos_w, ref_anchor_quat_w


# =================================================================================================
# Termination checks (replicate tracking_env_cfg terminations, single env)
# =================================================================================================
def body_pos_relative_w(ref_body_pos, ref_anchor_pos, ref_anchor_quat, robot_anchor_pos, robot_anchor_quat):
    """Reference body positions re-anchored to the robot's actual (xy, yaw); z from the reference anchor.
    Mirrors MotionCommand._update_command body_pos_relative_w."""
    delta_pos = robot_anchor_pos.copy()
    delta_pos[2] = ref_anchor_pos[2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
    out = np.zeros_like(ref_body_pos)
    for i in range(ref_body_pos.shape[0]):
        out[i] = delta_pos + quat_rotate(delta_ori, ref_body_pos[i] - ref_anchor_pos)
    return out


def check_terminations(refs, robot, robot_anchor_pos, robot_anchor_quat, ref_anchor_pos, ref_anchor_quat):
    reasons = []
    # anchor_pos (z only): |ref torso z - robot torso z| > 0.25
    if abs(ref_anchor_pos[2] - robot_anchor_pos[2]) > TERM_ANCHOR_POS_Z:
        reasons.append("anchor_pos")
    # anchor_ori: |proj_grav_z(ref) - proj_grav_z(robot)| > 0.8
    g = np.array([0.0, 0.0, -1.0])
    pg_ref = quat_rotate_inverse(ref_anchor_quat, g)[2]
    pg_rob = quat_rotate_inverse(robot_anchor_quat, g)[2]
    if abs(pg_ref - pg_rob) > TERM_ANCHOR_ORI:
        reasons.append("anchor_ori")
    # ee_body_pos (z only): any ee |z(ref_relative) - z(robot)| > 0.25
    ref_rel = body_pos_relative_w(refs["body_pos_w"], ref_anchor_pos, ref_anchor_quat,
                                  robot_anchor_pos, robot_anchor_quat)
    robot_tracked = robot.tracked_pos()
    ee_err_z = np.abs(ref_rel[EE_TRACKED_IDX, 2] - robot_tracked[EE_TRACKED_IDX, 2])
    if np.any(ee_err_z > TERM_EE_POS_Z):
        reasons.append("ee_body_pos")
    return reasons


# =================================================================================================
# Strike-success accumulator (Isaac exact-strike metric, replicated as a simple mean over exact
# frames — the converged value of Isaac's decay=0.99 EMA. Tracks overall + per swing-type clip).
# =================================================================================================
class StrikeAcc:
    def __init__(self):
        self.n = 0
        self.pos_err = self.vel_err = self.nrm_err = 0.0
        self.pos_pass = self.vel_pass = self.nrm_pass = self.comp = 0

    def add(self, pos_err, vel_err, nrm_err_deg):
        pp = pos_err < STRIKE_POS_THRESH
        pv = vel_err < STRIKE_VEL_THRESH
        pn = nrm_err_deg < STRIKE_NORMAL_THRESH_DEG
        self.n += 1
        self.pos_err += pos_err; self.vel_err += vel_err; self.nrm_err += nrm_err_deg
        self.pos_pass += pp; self.vel_pass += pv; self.nrm_pass += pn; self.comp += (pp and pv and pn)

    def rate(self, k):
        return (getattr(self, k) / self.n) if self.n else float("nan")


# =================================================================================================
# Rollout for one noise scale
# =================================================================================================
def run_rollout(policy, robot, refs_table, seg_start, seg_len, num_clips, step_dt, decimation,
                noise_scale, std_vec, n_steps, max_ep_len, rng, csv_writer, mode_label,
                target_normal_per_clip):
    racket = RacketCommand(seg_start, seg_len, step_dt, rng, target_normal_per_clip)
    strike = {"all": StrikeAcc(), "forehand": StrikeAcc(), "backhand": StrikeAcc()}

    def fresh_swing():
        """Sample a clip, set time_step to its start, ref-state-init the robot, resample racket target."""
        clip = int(rng.integers(0, num_clips))
        ts = int(seg_start[clip])
        r = refs_table[ts]
        robot.reset_to_reference(
            root_pos=r["body_pos_w"][ROOT_TRACKED_IDX], root_quat=r["body_quat_w"][ROOT_TRACKED_IDX],
            root_lin_w=r["body_lin_vel_w"][ROOT_TRACKED_IDX], root_ang_w=r["body_ang_vel_w"][ROOT_TRACKED_IDX],
            q_artic=r["joint_pos"])
        racket.resample(clip)
        racket.update_strike_timing(clip, ts)
        return clip, ts

    clip, time_step = fresh_swing()
    last_action = np.zeros(31)
    ep_len = 0

    # accumulators
    ep_lengths, term_reasons = [], []
    n_term_early = n_timeout = 0
    roll_acc, pitch_acc, torquemax_acc, footc_acc, n_acc = 0.0, 0.0, 0.0, 0.0, 0
    racket_err_acc, racket_err_n = 0.0, 0        # mean over the ±strike_window_s gate
    racket_exact_acc, racket_exact_n = 0.0, 0    # pos err at the EXACT strike frame (|tts| <= 0.5*step_dt)
    racket_velerr_acc = 0.0                       # racket vel err at the exact strike frame (Isaac bottleneck)
    fell = 0
    exact_tol = 0.5 * step_dt + 1e-6

    for step in range(n_steps):
        refs = refs_table[time_step]
        obs, base_quat_w, ra_pos, ra_quat, refa_pos, refa_quat = build_obs(
            refs, robot, racket, last_action, policy.default_q)

        mean = policy.action(obs, time_step)
        action = mean if noise_scale <= 0.0 else mean + noise_scale * std_vec * rng.standard_normal(31)
        last_action = action.copy()

        target_q = policy.default_q + action * policy.action_scale
        tau = robot.apply_pd_and_step(target_q, policy.kp, policy.kd, decimation)
        ep_len += 1

        # --- metrics (post-step state) ---
        bq = robot.body_quat(robot.pelvis_bid)
        # roll/pitch from pelvis quat (deg)
        w, x, y, z = bq
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        roll_d, pitch_d = math.degrees(roll), math.degrees(pitch)
        torque_max = float(np.max(np.abs(tau)))
        foot_c = robot.foot_contact_frac()
        roll_acc += abs(roll_d); pitch_acc += abs(pitch_d)
        torquemax_acc += torque_max; footc_acc += foot_c; n_acc += 1
        # racket tracking error inside the strike window
        racket_err = float("nan")
        if abs(racket.time_to_strike) <= STRIKE_WINDOW_S:
            racket_err = float(np.linalg.norm(robot.racket_pos() - racket.racket_target_pos_w))
            racket_err_acc += racket_err; racket_err_n += 1
            if abs(racket.time_to_strike) <= exact_tol:   # exact strike frame (Isaac exact_strike mask)
                # All three error channels at the exact strike frame (same arithmetic as Isaac):
                #   pos_err = ||racket_pos_w - target_pos_w||
                #   vel_err = ||racket_lin_vel_w - target_vel_w||   (full 3-vec norm)
                #   nrm_err = acos(dot(unit normals)) in degrees
                pos_err = racket_err
                vel_err = float(np.linalg.norm(robot.racket_lin_vel_w() - racket.racket_target_vel_w))
                nrm = robot.racket_normal_w()
                tgt_nrm = racket.racket_target_normal_w
                cos_a = float(np.clip(np.dot(nrm, tgt_nrm), -1.0, 1.0))
                nrm_err_deg = math.degrees(math.acos(cos_a))
                strike["all"].add(pos_err, vel_err, nrm_err_deg)
                strike[CLIP_NAMES[clip]].add(pos_err, vel_err, nrm_err_deg)
                racket_exact_acc += pos_err; racket_exact_n += 1
                racket_velerr_acc += vel_err

        # --- terminations ---
        reasons = check_terminations(refs, robot, ra_pos, ra_quat, refa_pos, refa_quat)
        terminated = len(reasons) > 0
        timeout = ep_len >= max_ep_len

        if csv_writer is not None:
            csv_writer.writerow([
                mode_label, step, time_step, clip, f"{racket.swing_sign:+.0f}",
                f"{racket.time_to_strike:.4f}", f"{roll_d:.3f}", f"{pitch_d:.3f}",
                f"{ra_pos[2]:.4f}", f"{refa_pos[2]:.4f}",
                f"{np.mean(np.abs(target_q)):.4f}", f"{np.max(np.abs(target_q)):.4f}",
                f"{torque_max:.2f}", f"{foot_c:.2f}",
                ("" if math.isnan(racket_err) else f"{racket_err:.4f}"),
                ep_len, ("|".join(reasons) if terminated else ("timeout" if timeout else "")),
            ])

        if terminated or timeout:
            ep_lengths.append(ep_len)
            if terminated:
                n_term_early += 1; fell += 1; term_reasons.extend(reasons)
            else:
                n_timeout += 1
            ep_len = 0
            clip, time_step = fresh_swing()
            last_action = np.zeros(31)
            continue

        # --- advance the motion clock; wrap within the env's current segment (multi-swing per episode) ---
        time_step += 1
        seg_end = int(seg_start[clip]) + int(seg_len[clip])
        if time_step >= seg_end:
            # clip wrap mid-episode: sample a new swing + ref-state-init (Isaac teleports here too),
            # but do NOT reset ep_len (the episode continues across swings until a fall/timeout).
            clip = int(rng.integers(0, num_clips))
            time_step = int(seg_start[clip])
            r = refs_table[time_step]
            robot.reset_to_reference(
                root_pos=r["body_pos_w"][ROOT_TRACKED_IDX], root_quat=r["body_quat_w"][ROOT_TRACKED_IDX],
                root_lin_w=r["body_lin_vel_w"][ROOT_TRACKED_IDX], root_ang_w=r["body_ang_vel_w"][ROOT_TRACKED_IDX],
                q_artic=r["joint_pos"])
            racket.resample(clip)
            last_action = np.zeros(31)
        racket.update_strike_timing(clip, time_step)

    total_term = n_term_early + n_timeout
    from collections import Counter
    rc = Counter(term_reasons)
    return dict(
        mode=mode_label, noise_scale=noise_scale,
        mean_ep_len=(sum(ep_lengths) / len(ep_lengths)) if ep_lengths else float("nan"),
        n_episodes=len(ep_lengths), n_term_early=n_term_early, n_timeout=n_timeout,
        terminated_rate=(n_term_early / total_term) if total_term else float("nan"),
        term_breakdown=dict(rc), fell=fell,
        base_roll_deg=roll_acc / max(n_acc, 1), base_pitch_deg=pitch_acc / max(n_acc, 1),
        torque_max_mean=torquemax_acc / max(n_acc, 1), foot_contact_frac=footc_acc / max(n_acc, 1),
        racket_pos_err_strike=(racket_err_acc / racket_err_n) if racket_err_n else float("nan"),
        racket_pos_err_exact=(racket_exact_acc / racket_exact_n) if racket_exact_n else float("nan"),
        racket_vel_err_exact=(racket_velerr_acc / racket_exact_n) if racket_exact_n else float("nan"),
        n_strikes=racket_exact_n,
        # --- full Isaac-matching strike composite metrics (overall + fh/bh) ---
        strike_composite_success_exact=strike["all"].rate("comp"),
        strike_pos_pass_exact=strike["all"].rate("pos_pass"),
        strike_vel_pass_exact=strike["all"].rate("vel_pass"),
        strike_normal_pass_exact=strike["all"].rate("nrm_pass"),
        racket_normal_err_exact=strike["all"].rate("nrm_err"),
        strike_composite_forehand=strike["forehand"].rate("comp"),
        strike_composite_backhand=strike["backhand"].rate("comp"),
        n_strikes_fh=strike["forehand"].n, n_strikes_bh=strike["backhand"].n,
    )


# =================================================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    wbt = os.path.dirname(here)                 # whole_body_tracking/
    repo = os.path.dirname(os.path.dirname(wbt))  # HOPE/ (whole_body_tracking is hope_training/whole_body_tracking)
    default_run = os.path.join(
        wbt, "logs/rsl_rl/agibot_a3_hope/2026-06-27_18-14-06_basecouple03_resume")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", default=os.path.join(default_run, "exported/policy.onnx"))
    p.add_argument("--std", default=os.path.join(default_run, "exported/learned_std.npy"),
                   help="learned_std.npy sidecar for the dither mode (31,). Optional if only noise_scale=0.")
    p.add_argument("--mjcf", default=os.path.join(
        repo, "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"))
    p.add_argument("--motion-files", nargs="+", default=[
        os.path.join(wbt, "logs/rsl_rl/eval_motion/fh.npz"),
        os.path.join(wbt, "logs/rsl_rl/eval_motion/bh.npz")],
        help="motion clips in TRAINING order (clip0=forehand, clip1=backhand). Used for segment lengths.")
    p.add_argument("--noise-scales", nargs="+", type=float, default=[0.0, 0.05])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--sim-dt", type=float, default=0.005, help="MuJoCo physics dt (Isaac used 0.005)")
    p.add_argument("--decimation", type=int, default=4, help="physics substeps per 50 Hz control step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-passive", action="store_true",
                   help="keep MJCF joint damping+frictionloss (harder, less faithful to Isaac)")
    p.add_argument("--pd-mode", choices=["explicit", "implicit"], default="explicit",
                   help="explicit: torque=kp*e-kd*qd. implicit: kp torque + kd as passive damping via "
                        "MuJoCo implicitfast (matches Isaac's ImplicitActuator; less fast-swing undershoot).")
    p.add_argument("--out-dir", default=None, help="where to write the CSV (default: ONNX run dir)")
    args = p.parse_args()

    step_dt = args.sim_dt * args.decimation
    assert abs(step_dt - 0.02) < 1e-9, f"control dt {step_dt} != 0.02 (50 Hz). adjust --sim-dt/--decimation"
    max_ep_len = int(round(10.0 / step_dt))   # 10 s episode -> 500 steps

    print(f"[mj-sim2sim] onnx={args.onnx}")
    print(f"[mj-sim2sim] mjcf={args.mjcf}")
    policy = OnnxPolicy(args.onnx)
    print(f"[mj-sim2sim] obs_dim={policy.obs_dim} joints={len(policy.joint_names)} "
          f"control={1/step_dt:.0f}Hz (sim_dt={args.sim_dt}, decim={args.decimation})")

    # std sidecar (only needed if a noise_scale > 0 is requested)
    std_vec = None
    if any(s > 0 for s in args.noise_scales):
        if not os.path.isfile(args.std):
            raise SystemExit(f"[FATAL] dither mode requested but std sidecar not found: {args.std}\n"
                             f"        Create it from the checkpoint: np.save(.../learned_std.npy, "
                             f"torch.load(model.pt)['model_state_dict']['std'])")
        std_vec = np.load(args.std).astype(np.float64).reshape(-1)
        assert std_vec.shape == (31,), f"std sidecar shape {std_vec.shape} != (31,)"
        print(f"[mj-sim2sim] learned std: mean={std_vec.mean():.4f} min={std_vec.min():.4f} max={std_vec.max():.4f}")

    # segment lengths from the motion npz frame counts (clip0 then clip1, like MotionLoader)
    seg_len = []
    for f in args.motion_files:
        if not os.path.isfile(f):
            raise SystemExit(f"[FATAL] motion file not found: {f}")
        seg_len.append(int(np.load(f)["joint_pos"].shape[0]))
    seg_len = np.array(seg_len, int)
    seg_start = np.zeros(len(seg_len), int)
    if len(seg_len) > 1:
        seg_start[1:] = np.cumsum(seg_len)[:-1]
    num_clips = len(seg_len)
    T = int(seg_len.sum())
    print(f"[mj-sim2sim] motion: {num_clips} clips, seg_len={seg_len.tolist()}, "
          f"seg_start={seg_start.tolist()}, T={T}")

    robot = MujocoRobot(args.mjcf, policy.joint_names, policy.body_names, args.sim_dt, args.keep_passive,
                        args.pd_mode, kd_for_implicit=policy.kd)
    print(f"[mj-sim2sim] PD mode: {args.pd_mode}"
          + ("  (kd as passive damping + implicitfast integrator)" if args.pd_mode == "implicit" else ""))

    # Precompute the reference table (refs depend only on time_step) -> one ONNX call per frame, once.
    refs_table = [policy.refs(ts) for ts in range(T)]

    # Per-clip TARGET paddle normal (unified uniform mode): the imitated swing's reference face normal
    # at its strike frame = local +Y of the reference wrist(=racket) frame at strike_step.
    target_normal_per_clip = []
    for c in range(num_clips):
        strike_step = int(seg_start[c]) + int(round(STRIKE_PHASE_PER_CLIP[c] * (seg_len[c] - 1)))
        ref_wrist_quat = refs_table[strike_step]["body_quat_w"][WRIST_TRACKED_IDX]
        target_normal_per_clip.append(mat_from_quat(ref_wrist_quat)[:, MOUNT_NORMAL_AXIS] * MOUNT_NORMAL_SIGN)
    target_normal_per_clip = np.array(target_normal_per_clip)

    out_dir = args.out_dir or default_run
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "mujoco_sim2sim_log.csv")
    csv_f = open(csv_path, "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["mode", "step", "time_step", "clip", "swing_sign", "time_to_strike",
                 "base_roll_deg", "base_pitch_deg", "torso_z", "ref_torso_z",
                 "target_q_mean_abs", "target_q_max_abs", "torque_max", "foot_contact_frac",
                 "racket_pos_err_strike", "episode_len", "term_reason"])

    results = []
    for ns in args.noise_scales:
        rng = np.random.default_rng(args.seed)   # same seed per mode -> identical target/clip sequence
        print(f"\n[mj-sim2sim] >>> rollout noise_scale={ns}")
        res = run_rollout(policy, robot, refs_table, seg_start, seg_len, num_clips, step_dt,
                          args.decimation, ns, std_vec, args.steps, max_ep_len, rng, cw,
                          mode_label=f"ns={ns}", target_normal_per_clip=target_normal_per_clip)
        results.append(res)
    csv_f.close()

    # ---- summary table ----
    print("\n" + "=" * 92)
    print(f"MuJoCo sim-to-sim | {os.path.basename(args.onnx)} | {args.steps} steps | seed {args.seed}")
    print("-" * 92)
    cols = [f"{r['mode']:>16s}" for r in results]
    print(f"{'metric':28s}" + "".join(cols))
    def row(label, key, fmt="{:16.4f}"):
        print(f"{label:28s}" + "".join(fmt.format(r[key]) if isinstance(r[key], float) else f"{str(r[key]):>16s}"
                                       for r in results))
    row("mean_episode_length", "mean_ep_len")
    row("terminated_rate", "terminated_rate")
    row("n_episodes", "n_episodes")
    row("early/timeout", "n_term_early", "{:16d}")
    print(f"{'  (timeouts)':28s}" + "".join(f"{r['n_timeout']:16d}" for r in results))
    row("base_roll_deg(|mean|)", "base_roll_deg")
    row("base_pitch_deg(|mean|)", "base_pitch_deg")
    row("torque_max(mean)", "torque_max_mean")
    row("foot_contact_frac", "foot_contact_frac")
    row("racket_err@window(m)", "racket_pos_err_strike")
    print("-" * 92)
    row("strike_composite_succ_exact", "strike_composite_success_exact")
    row("  forehand", "strike_composite_forehand")
    row("  backhand", "strike_composite_backhand")
    row("strike_pos_pass_exact", "strike_pos_pass_exact")
    row("strike_vel_pass_exact", "strike_vel_pass_exact")
    row("strike_normal_pass_exact", "strike_normal_pass_exact")
    row("racket_pos_err@exact(m)", "racket_pos_err_exact")
    row("racket_vel_err@exact(m/s)", "racket_vel_err_exact")
    row("racket_normal_err@exact(deg)", "racket_normal_err_exact")
    row("n_strikes (exact)", "n_strikes", "{:16d}")
    print(f"{'  fh/bh strikes':28s}" + "".join(f"{str(str(r['n_strikes_fh'])+'/'+str(r['n_strikes_bh'])):>16s}"
                                               for r in results))
    print("-" * 92)
    row("fell(count)", "fell", "{:16d}")
    print(f"{'term_breakdown':28s}" + "".join(f"{str(r['term_breakdown']):>16s}" for r in results))
    print("=" * 92)
    print(f"[mj-sim2sim] per-step CSV -> {csv_path}\n")


if __name__ == "__main__":
    main()
