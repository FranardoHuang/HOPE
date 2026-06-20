"""HOPE-specific command term: racket / base target tracking on top of BeyondMimic.

This adds the HITTER (arXiv:2508.21043) racket-target objective to the BeyondMimic motion
tracker. The base ``MotionCommand`` (in ``commands.py``) drives the imitation reward and owns the
per-env motion clock (``time_steps``). ``RacketTargetCommand`` rides on top of it:

* it samples a *desired* racket state (position, velocity, face normal) and a *desired* base XY
  position each swing — exactly the quantities the model-based planner emits at deploy time via
  ``hope_msgs/RacketCommand`` (position, velocity, normal). No ball is needed in simulation;
  training samples targets, the planner supplies them at runtime.
* it computes the *actual* racket state in simulation by forward kinematics through the fixed
  racket mount ``T_mount`` (wrist -> paddle center), so the reward can compare actual vs desired.
* it derives the strike timing from the motion clip phase, exposing ``time_to_strike`` plus the
  ``pre_strike`` / ``strike_window`` masks that gate the goal rewards.

Per the HOPE racket-tracking prohibition, there is NO measured racket feedback at deploy time:
``r_racket`` is a simulation-only signal; on hardware the policy runs open-loop on racket pose.

HITTER alignment notes (see the project HITTER verification):
* racket *position* is observed relative to the base; racket *velocity* is observed in world.
* racket *normal* is a REWARD target only (not an actor observation in HITTER).
* swing type is a *sampled* variable used here to (a) flag forehand/backhand and (b) select the
  reference clip; it is not required in the actor observation when separate forehand/backhand
  policies are trained (the HOPE default, reimplement.md step 17).
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_mul,
    quat_rotate_inverse,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RacketTargetCommand(CommandTerm):
    """Samples desired racket/base targets and computes the actual racket state by FK."""

    cfg: RacketTargetCommandCfg

    def __init__(self, cfg: RacketTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Resolve the racket FK source: prefer the racket body if it survived the physics import,
        # otherwise fall back to (wrist body pose) * (constant mount offset).
        # NOTE: Articulation.find_bodies() RAISES (resolve_matching_names) when a name matches no
        # body — it does not return []. So gate on body_names membership before calling it, or the
        # wrist-offset fallback below becomes unreachable.
        if cfg.racket_body_name in self.robot.body_names:
            self._racket_mode = "body"
            self._racket_body_index = self.robot.find_bodies(cfg.racket_body_name, preserve_order=True)[0][0]
            self._wrist_body_index = -1
        else:
            self._racket_mode = "wrist_offset"
            self._racket_body_index = -1
            assert cfg.wrist_body_name in self.robot.body_names, (
                f"RacketTargetCommand: neither racket body '{cfg.racket_body_name}' nor wrist body "
                f"'{cfg.wrist_body_name}' found on asset '{cfg.asset_name}'."
            )
            self._wrist_body_index = self.robot.find_bodies(cfg.wrist_body_name, preserve_order=True)[0][0]

        self._mount_offset = torch.tensor(cfg.mount_offset, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        # Fixed wrist->racket rotation (used only in wrist_offset fallback mode so the face normal is
        # taken in the racket frame, not the bare wrist frame). Identity for the A3 mount (all mount
        # joints are rpy=0); set non-identity if your mount tilts the paddle relative to the wrist.
        self._mount_quat = torch.tensor(cfg.mount_quat, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )

        # The motion command (resolved lazily on first update; not guaranteed to exist at __init__).
        self._motion_term: MotionCommand | None = None
        # Per-env motion phase at the last target resample; used to detect clip wraps (new swings).
        # Stamped on every resample so a reset-time resample is not double-triggered next step.
        self._prev_motion_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Desired (sampled) targets, world frame.
        self.racket_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w[:, 2] = 1.0
        self.base_target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.swing_sign = torch.ones(self.num_envs, device=self.device)

        # Actual racket state, world frame (from FK).
        self.racket_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.racket_quat_w[:, 0] = 1.0
        self.racket_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w[:, 2] = 1.0

        # Strike timing / gating.
        self.time_to_strike = torch.zeros(self.num_envs, device=self.device)
        self.pre_strike = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.strike_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["racket_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_pos_error"] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ #
    # CommandTerm API
    # ------------------------------------------------------------------ #
    @property
    def command(self) -> torch.Tensor:
        """Raw desired-target vector (world frame): [pos(3), vel(3), normal(3), t_left(1), base_xy(2), swing(1)]."""
        return torch.cat(
            [
                self.racket_target_pos_w,
                self.racket_target_vel_w,
                self.racket_target_normal_w,
                self.time_to_strike.unsqueeze(-1),
                self.base_target_pos_w,
                self.swing_sign.unsqueeze(-1),
            ],
            dim=-1,
        )

    @property
    def base_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w

    @property
    def base_quat_w(self) -> torch.Tensor:
        return self.robot.data.root_quat_w

    def _motion(self) -> MotionCommand:
        if self._motion_term is None:
            self._motion_term = self._env.command_manager.get_term(self.cfg.motion_command_name)
        return self._motion_term

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]

        # Desired racket position (world): a reachable offset from the per-env nominal stance.
        # NOTE: ranges are PLACEHOLDERS — set them from the A3 arm IK reachability at the nominal
        # stance (reimplement.md step 13, "reachable racket target").
        pos = origins.clone()
        pos[:, 0] += sample_uniform(*self.cfg.racket_pos_x_range, (n,), self.device)
        pos[:, 1] += sample_uniform(*self.cfg.racket_pos_y_range, (n,), self.device)
        pos[:, 2] += sample_uniform(*self.cfg.racket_pos_z_range, (n,), self.device)
        self.racket_target_pos_w[env_ids] = pos

        # Desired racket velocity (world): biased toward the opponent (+X) with lateral/vertical spread.
        vel = torch.empty(n, 3, device=self.device)
        vel[:, 0] = sample_uniform(*self.cfg.racket_vel_x_range, (n,), self.device)
        vel[:, 1] = sample_uniform(*self.cfg.racket_vel_y_range, (n,), self.device)
        vel[:, 2] = sample_uniform(*self.cfg.racket_vel_z_range, (n,), self.device)
        self.racket_target_vel_w[env_ids] = vel

        # Desired racket face normal (world).
        if self.cfg.normal_mode == "velocity":
            normal = vel / (torch.norm(vel, dim=-1, keepdim=True) + 1e-6)
        else:  # "sampled"
            normal = torch.empty(n, 3, device=self.device)
            normal[:, 0] = sample_uniform(*self.cfg.racket_normal_x_range, (n,), self.device)
            normal[:, 1] = sample_uniform(*self.cfg.racket_normal_y_range, (n,), self.device)
            normal[:, 2] = sample_uniform(*self.cfg.racket_normal_z_range, (n,), self.device)
            normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        self.racket_target_normal_w[env_ids] = normal

        # Desired base XY (world): where the feet should step to before the strike.
        base_xy = origins[:, :2].clone()
        base_xy[:, 0] += sample_uniform(*self.cfg.base_target_x_range, (n,), self.device)
        base_xy[:, 1] += sample_uniform(*self.cfg.base_target_y_range, (n,), self.device)
        self.base_target_pos_w[env_ids] = base_xy

        # Swing type from target Y relative to the nominal base Y (right arm holds the paddle).
        base_y_nom = origins[:, 1] + self.cfg.base_nominal_offset[1]
        dy = pos[:, 1] - base_y_nom
        if self.cfg.forehand_on_negative_y:
            self.swing_sign[env_ids] = torch.where(dy <= 0.0, 1.0, -1.0)
        else:
            self.swing_sign[env_ids] = torch.where(dy >= 0.0, 1.0, -1.0)

        # Stamp the motion phase baseline for these envs so the per-swing wrap detector in
        # _update_command does not immediately re-trigger after this (e.g. reset-time) resample.
        self._prev_motion_steps[env_ids] = self._motion().time_steps[env_ids]

    def _compute_racket_state(self):
        data = self.robot.data
        if self._racket_mode == "body":
            idx = self._racket_body_index
            self.racket_pos_w = data.body_pos_w[:, idx]
            self.racket_quat_w = data.body_quat_w[:, idx]
            self.racket_lin_vel_w = data.body_lin_vel_w[:, idx]
        else:
            widx = self._wrist_body_index
            wpos = data.body_pos_w[:, widx]
            wquat = data.body_quat_w[:, widx]
            wlin = data.body_lin_vel_w[:, widx]
            wang = data.body_ang_vel_w[:, widx]
            offset_w = quat_apply(wquat, self._mount_offset)
            self.racket_pos_w = wpos + offset_w
            self.racket_lin_vel_w = wlin + torch.cross(wang, offset_w, dim=-1)
            self.racket_quat_w = quat_mul(wquat, self._mount_quat)
        # Face normal = chosen local axis of the racket frame, mapped to world.
        # TODO(asset): confirm mount_normal_axis/sign against pingpang_red_Link.STL (see hope-a3-racket-mount).
        self.racket_normal_w = (
            matrix_from_quat(self.racket_quat_w)[:, :, self.cfg.mount_normal_axis] * self.cfg.mount_normal_sign
        )

    def _update_command(self):
        motion = self._motion()
        total = max(int(motion.motion.time_step_total), 1)
        strike_step = round(self.cfg.strike_phase * (total - 1))
        self.time_to_strike = (strike_step - motion.time_steps).float() * self._env.step_dt
        self.pre_strike = self.time_to_strike > 0.0
        self.strike_window = self.time_to_strike.abs() <= self.cfg.strike_window_s

        # Re-sample the target at each new swing (the motion clip wrapped to an earlier frame).
        # _resample_command re-stamps _prev_motion_steps for the affected envs; the full clone below
        # keeps every env current. Targets for fresh episodes are sampled by the manager's reset.
        wrapped = torch.where(motion.time_steps < self._prev_motion_steps)[0]
        if len(wrapped) > 0:
            self._resample_command(wrapped)
        self._prev_motion_steps = motion.time_steps.clone()

    def _update_metrics(self):
        # CommandTerm.compute() runs _update_metrics() BEFORE _update_command(), so refresh the
        # actual racket FK here (once per step) — metrics, rewards, and observations then all read
        # the same fresh buffers (rewards/obs read them after the full command_manager.compute()).
        self._compute_racket_state()
        self.metrics["racket_pos_error"] = torch.norm(self.racket_pos_w - self.racket_target_pos_w, dim=-1)
        self.metrics["racket_vel_error"] = torch.norm(self.racket_lin_vel_w - self.racket_target_vel_w, dim=-1)
        cos_ang = torch.sum(self.racket_normal_w * self.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
        self.metrics["racket_normal_error_deg"] = torch.acos(cos_ang) * (180.0 / math.pi)
        self.metrics["base_pos_error"] = torch.norm(self.base_pos_w[:, :2] - self.base_target_pos_w, dim=-1)

    # ------------------------------------------------------------------ #
    # Observation helpers (base-relative quantities)
    # ------------------------------------------------------------------ #
    def racket_target_pos_b(self) -> torch.Tensor:
        """Desired racket position relative to the base (yaw-heading frame). HITTER actor obs."""
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_target_pos_w - self.base_pos_w)

    def base_target_pos_b(self) -> torch.Tensor:
        """Desired base XY position relative to the current base (yaw-heading frame). HITTER actor obs."""
        delta_xy = self.base_target_pos_w - self.base_pos_w[:, :2]
        delta = torch.cat([delta_xy, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), delta)[:, :2]

    # ------------------------------------------------------------------ #
    # Debug visualization (no-op stubs; targets are world-frame buffers).
    # ------------------------------------------------------------------ #
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class RacketTargetCommandCfg(CommandTermCfg):
    """Configuration for :class:`RacketTargetCommand`."""

    class_type: type = RacketTargetCommand

    asset_name: str = MISSING
    motion_command_name: str = "motion"

    # The target is re-sampled per swing (on clip wrap / reset), not on a fixed time schedule,
    # so disable the base CommandTerm time-based resampling.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # --- racket mount FK ---
    racket_body_name: str = "pingpang_red_Link"
    wrist_body_name: str = "right_wrist_yaw_Link"
    mount_offset: tuple[float, float, float] = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
    # Fixed wrist->racket rotation (w, x, y, z); only used in the wrist_offset FK fallback. Identity
    # for the A3 ping-pong URDF (all mount joints are rpy=0). Set non-identity if the mount tilts the
    # paddle relative to the wrist frame.
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mount_normal_axis: int = 2  # racket-local +Z is the face normal (CONFIRM against the A3 paddle mesh)
    mount_normal_sign: float = 1.0

    # --- strike timing (fraction of the reference clip where the paddle meets the ball) ---
    strike_phase: float = 0.46  # HITTER clip: strike at frame 43/94 ≈ 0.46
    strike_window_s: float = 0.1  # half-window; goal-racket reward active within ±strike_window_s

    # --- nominal stance (offset of the base from the env origin) ---
    base_nominal_offset: tuple[float, float, float] = (0.0, 0.0, 0.93)

    # --- reachable racket-target workspace (offsets from the env origin, world frame, meters) ---
    # PLACEHOLDER ranges — measure from A3 arm IK reachability at the nominal stance.
    racket_pos_x_range: tuple[float, float] = (0.25, 0.55)
    racket_pos_y_range: tuple[float, float] = (-0.45, 0.45)
    racket_pos_z_range: tuple[float, float] = (0.70, 1.15)

    # --- desired racket velocity (world frame, m/s) ---
    racket_vel_x_range: tuple[float, float] = (1.5, 4.0)
    racket_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    racket_vel_z_range: tuple[float, float] = (0.0, 1.5)

    # --- desired racket face normal ---
    normal_mode: str = "velocity"  # "velocity" (n = v/|v|) or "sampled"
    racket_normal_x_range: tuple[float, float] = (0.5, 1.0)
    racket_normal_y_range: tuple[float, float] = (-0.3, 0.3)
    racket_normal_z_range: tuple[float, float] = (-0.3, 0.3)

    # --- desired base XY target (offsets from the env origin, world frame, meters) ---
    base_target_x_range: tuple[float, float] = (-0.10, 0.10)
    base_target_y_range: tuple[float, float] = (-0.35, 0.35)

    # --- swing-type convention ---
    forehand_on_negative_y: bool = True  # right arm holds the paddle: target on -Y side -> forehand (+1)
