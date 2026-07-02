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
* HOPE currently also observes desired racket *normal* in the actor so the policy can respond to
  normal targets; actual racket normal remains a privileged simulation-only critic/reward signal.
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
    euler_xyz_from_quat,
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

        # Reference racket state at the strike frame (CONSTANT per clip): pos (env-origin relative),
        # world linear velocity, and face normal, computed by the SAME FK as the actual racket
        # (_compute_racket_state) but fed the reference MOTION's body poses. Used by the
        # "reference_perturbed" target mode so a sampled target is one the imitated swing can actually
        # reach (a perfect imitator hits it exactly). Cached lazily on first resample, after the motion
        # term is resolved and its motion_file is loaded.
        self._ref_strike_cached = False
        # Unified multi-clip: per-clip strike phase as a [num_segments] tensor (built lazily once the
        # motion term is resolved). None until then; falls back to the scalar strike_phase.
        self._strike_phase_per_clip_t = None
        # Per-clip reference paddle FACE NORMAL at the strike frame ([num_segments, 3], built lazily). In
        # uniform mode the target normal is set to the imitated swing's actual paddle normal (which the
        # policy can achieve) — NOT the racket-velocity direction, which is ~18-110 deg off the +Y blade
        # face and makes the normal goal (and thus the composite success metric) unsatisfiable.
        self._ref_normal_per_clip = None
        # Optional per-clip racket target-velocity boxes (uniform mode). Built ONCE from cfg; stays None
        # when the shared box is used (backward compatible). Shape (num_clips, 3, 2): [clip][x/y/z][lo/hi].
        self._vel_range_per_clip_t = None
        if self.cfg.racket_vel_range_per_clip is not None:
            self._vel_range_per_clip_t = torch.tensor(
                [[[float(lo), float(hi)] for (lo, hi) in clip_rng]
                 for clip_rng in self.cfg.racket_vel_range_per_clip],
                dtype=torch.float32,
                device=self.device,
            )
        # Optional per-clip racket target-POSITION boxes (uniform mode). Same shape/semantics as the
        # velocity one above; None -> shared pos box (backward compatible). (num_clips, 3, 2): [clip][x/y/z][lo/hi].
        self._pos_range_per_clip_t = None
        if self.cfg.racket_pos_range_per_clip is not None:
            self._pos_range_per_clip_t = torch.tensor(
                [[[float(lo), float(hi)] for (lo, hi) in clip_rng]
                 for clip_rng in self.cfg.racket_pos_range_per_clip],
                dtype=torch.float32,
                device=self.device,
            )
        self._ref_racket_pos_rel = torch.zeros(3, device=self.device)
        self._ref_racket_vel_w = torch.zeros(3, device=self.device)
        self._ref_racket_normal_w = torch.zeros(3, device=self.device)
        # Reference base (root) XY at the strike + the base->racket horizontal offset. Used to COUPLE
        # base_target to racket_target so standing at base_target keeps the racket reachable.
        self._ref_base_pos_rel = torch.zeros(3, device=self.device)
        self._ref_reach_offset_xy = torch.zeros(2, device=self.device)

        # Success-gated curriculum: running perturbation scale, advanced only when the smoothed
        # exact-strike composite success clears the threshold (see _perturb_scale / _update_metrics).
        self._curr_perturb_scale = float(cfg.ref_perturb_curriculum_start)

        # Decayed accumulators for the CONDITIONAL exact-strike pass rates (see _update_metrics). The
        # logged strike_*_pass_exact / strike_composite_success_exact report the fraction of *exact-strike
        # samples* that clear each acceptance threshold — NOT a per-env value held over the long
        # non-strike portion of every episode (which diluted the old metric ~10x at reset-logging time).
        # Sample-weighted EMA: acc = decay*acc + this-step-count; rate = pass_acc / sample_acc.
        self._exact_n_acc = 0.0
        self._exact_pass_comp_acc = 0.0
        self._exact_pass_pos_acc = 0.0
        self._exact_pass_vel_acc = 0.0
        self._exact_pass_normal_acc = 0.0
        # 5 cm / 10 cm position-accuracy buckets on the SAME exact-strike mask + EMA denominator as the
        # pass metrics above (so they are comparable with the composite, unlike the old window-exit-held
        # strike_success_5cm/10cm which sampled the racket ~0.26 m past target).
        self._exact_pass_5cm_acc = 0.0
        self._exact_pass_10cm_acc = 0.0
        self._exact_composite_rate = 0.0

        # Per-clip (forehand=clip 0 / backhand=clip 1) breakdown of the exact-strike metrics, so wandb
        # shows each swing separately (the aggregate composite can hide one swing lagging). Same
        # sample-weighted EMA as the global accumulators above, but each clip's exact-strike samples are
        # accumulated separately (selected by the motion command's clip_id). Populated in multiseg only.
        self._clip_names = {0: "forehand", 1: "backhand"}
        self._exact_n_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pass_pos_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pass_vel_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pass_normal_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pass_comp_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pos_err_sum_c = {c: 0.0 for c in self._clip_names}
        self._exact_vel_err_sum_c = {c: 0.0 for c in self._clip_names}
        self._exact_nrm_err_sum_c = {c: 0.0 for c in self._clip_names}

        # --- UNCONDITIONAL swing accounting (Phase A wandb fix) ------------------------------------
        # strike_composite_success_exact is CONDITIONAL: its denominator is exact-strike SAMPLES, so
        # an env that falls BEFORE the strike frame contributes nothing — composite ~1.0 coexists
        # with any pre-strike fall rate (exactly what happened in deploy). These accumulators count
        # every swing START (episode reset or clip wrap assigns a new swing) with the same EMA decay
        # as the exact accumulators, so:
        #   swing_completion_rate = exact_n_acc / swing_starts_acc   (unconditional; falls count)
        #   pre_strike_fall_rate  = pre-strike terminations / swing starts
        self._swing_starts_acc = 0.0
        self._swing_starts_acc_c = {c: 0.0 for c in self._clip_names}
        self._prestrike_fall_acc = 0.0
        # True only while _resample_command is invoked from the intra-episode WRAP path (see
        # _update_command): wraps start a new swing but never count a pre-strike fall (a wrapped
        # env necessarily passed its strike frame alive).
        self._resample_is_wrap = False

        # Strike timing / gating.
        self.time_to_strike = torch.zeros(self.num_envs, device=self.device)
        self.pre_strike = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.strike_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Episode-wide tracking errors (instantaneous; averaged over terminating envs at reset).
        self.metrics["racket_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        # How far the (coupled) base target sits from spawn — i.e. how much repositioning is commanded.
        # ~0 when base_couple_blend=0; grows with the coupling toward far racket targets.
        self.metrics["base_target_offset_norm"] = torch.zeros(self.num_envs, device=self.device)
        # Strike-window metrics: hold the value from the MOST RECENT strike — these map directly to
        # the acceptance criteria (racket pos < 7.5 cm, vel < 0.5 m/s, normal < 15 deg AT strike) and
        # are the real "is the policy learning to hit" signal (the episode-wide ones above are diluted
        # by the long non-strike portion of each swing).
        self.metrics["racket_pos_error_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_success"] = torch.zeros(self.num_envs, device=self.device)
        # Exact-strike metrics: sampled only on the nearest control frame to the configured strike
        # step. These avoid the "within-window" dilution from the +/- strike reward window.
        self.metrics["racket_pos_error_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_composite_success_exact"] = torch.zeros(self.num_envs, device=self.device)
        # Per-axis position error AT the exact strike frame (which axis is the miss?).
        self.metrics["racket_pos_error_x_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_y_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_z_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Conditional exact-strike pass rates (broadcast scalar; the trustworthy success signal). Each is
        # the fraction of exact-strike samples that clear that acceptance threshold, undiluted by the
        # non-strike steps that wrecked the old held metric. strike_composite_success_exact requires all
        # three (pos & vel & normal) and drives the success-gated perturbation curriculum.
        self.metrics["strike_pos_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_vel_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_normal_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        # Position-accuracy buckets + error distribution on the exact-strike sample (comparable w/ composite).
        self.metrics["exact_strike_pos_success_5cm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_pos_success_10cm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_pos_err_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_pos_err_p90"] = torch.zeros(self.num_envs, device=self.device)
        # Per-clip (forehand/backhand) versions of the exact-strike pass rates + errors (multiseg only;
        # stay 0 for a single-clip run). Updated in _update_metrics.
        for _cname in self._clip_names.values():
            for _key in (
                "strike_pos_pass_exact", "strike_vel_pass_exact", "strike_normal_pass_exact",
                "strike_composite_success_exact", "racket_pos_error_exact_strike",
                "racket_vel_error_exact_strike", "racket_normal_error_deg_exact_strike",
            ):
                self.metrics[f"{_key}_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_sample_count_decayed"] = torch.zeros(self.num_envs, device=self.device)
        # UNCONDITIONAL swing accounting (Phase A): completion_rate = exact-strike arrivals / swing
        # STARTS (falls count against it, unlike the conditional composite above); fall rate before
        # the strike frame. Broadcast scalars like the pass rates.
        self.metrics["swing_completion_rate"] = torch.zeros(self.num_envs, device=self.device)
        for _cname in self._clip_names.values():
            self.metrics[f"swing_completion_rate_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_strike_fall_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_window_hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Base-position error while the base target is active (pre-strike), held at its last value.
        self.metrics["base_pos_error_pre_strike"] = torch.zeros(self.num_envs, device=self.device)
        # Swing-quality detail at the most recent strike: actual paddle speed, per-axis position error,
        # and success at tighter/looser thresholds (5 cm / 10 cm) for a fuller accuracy distribution.
        self.metrics["racket_speed_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_target_speed_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_x_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_y_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_z_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        # DEPRECATED semantics: these hold the value at the WINDOW-EXIT frame (racket ~0.26 m past target),
        # NOT at contact, and use the diluting reset-mean denominator. Renamed so they stop reading as
        # "success". Use exact_strike_pos_success_5cm/10cm above for the real contact-frame accuracy.
        self.metrics["strike_success_5cm_window_exit"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_success_10cm_window_exit"] = torch.zeros(self.num_envs, device=self.device)
        # Robot-health diagnostics (episode-wide, instantaneous) — logged here because this term already
        # holds ``self.robot``. Useful for sim2real: standing height, peak joint speed, actuator effort.
        self.metrics["base_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_upright"] = torch.zeros(self.num_envs, device=self.device)
        # Stability diagnostics (instability shows up here BEFORE a termination): absolute base roll/pitch
        # (deg; 0 = level — base_upright only gives the combined tilt), and foot contact + slip (a planted
        # foot should be ~still; horizontal foot speed while in contact = slip = the robot shuffling/sliding).
        self.metrics["base_roll_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_pitch_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_contact_frac"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_slip_speed"] = torch.zeros(self.num_envs, device=self.device)
        # Foot-slip magnitude (sum over feet of horizontal speed WHILE in contact) used by the
        # pre_strike_foot_slip reward. Recomputed each step in _update_metrics; stays 0 if the feet /
        # contact sensor cannot be resolved, so the reward is a safe no-op in that case.
        self.foot_slip_in_contact = torch.zeros(self.num_envs, device=self.device)
        # Fraction of feet in contact (mean over the 2 feet): 1.0 = both planted, 0.5 = one foot
        # unloaded, 0.0 = airborne. Clean attribute for the feet_contact stance reward (real_sensor
        # variant). Same value as metrics["foot_contact_frac"]; 0 (safe) if the sensor cannot resolve.
        self.feet_contact_frac = torch.zeros(self.num_envs, device=self.device)
        # ---- footwork-to-strike signals (base-FREE; reward/metric only, NEVER in the observation) ----
        # racket_target_distance = ||racket_FK - racket_target|| (frame-invariant, no base position).
        # racket_progress = prev_distance - current_distance (>0 = the WHOLE body moved the racket closer
        # to the target). This dense progress term is what drives footwork WITHOUT a base target.
        z = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self.racket_target_distance = z()
        self.racket_progress = z()
        self._prev_racket_dist = z()
        self.foot_slip_sq = z()  # sum_feet contact * ||foot_xy_vel||^2
        self.foot_vel_sq = z()  # sum_feet ||foot_vel||^2 (excessive/violent foot motion)
        self.foot_drag = z()  # sum_feet ||foot_xy_vel|| while the foot is LOW (near ground -> dragging)
        self.arm_overreach_frac = z()  # fraction of ARM joints within 10% of a position limit
        self.waist_twist = z()  # |waist_yaw - default| + |waist_roll - default| (anti twist-instead-of-step)
        self.proj_grav_xy = z()  # ||projected_gravity_xy|| = base tilt (strike-window stability)
        self.base_ang_vel_xy_norm = z()  # ||base_ang_vel_xy|| (strike-window stability)
        self.vertical_speed = z()  # |base_lin_vel_z| (vertical bob)
        self._drag_height = 0.10  # m: foot below this counts as "near ground" for the drag penalty
        self.metrics["joint_vel_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["time_to_strike_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_strike_flag"] = torch.zeros(self.num_envs, device=self.device)
        # Curriculum perturbation scale (reference_perturbed mode): 0 at start ramping to 1; lets you
        # watch the reachable target ball widen in wandb. Stays 0 in "uniform" mode.
        self.metrics["ref_perturb_scale"] = torch.zeros(self.num_envs, device=self.device)
        self._has_jpos_limits = hasattr(self.robot.data, "soft_joint_pos_limits") or hasattr(
            self.robot.data, "joint_pos_limits"
        )
        if self._has_jpos_limits:
            self.metrics["joint_pos_near_limit_frac"] = torch.zeros(self.num_envs, device=self.device)
        self._has_torque = hasattr(self.robot.data, "applied_torque")
        if self._has_torque:
            self.metrics["joint_torque_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["joint_torque_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        # Policy action magnitude (saturation check for sim2real).
        self.metrics["action_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_delta_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_delta_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y"):
            self.metrics[f"base_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"base_pos_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y", "z"):
            self.metrics[f"racket_pos_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"racket_vel_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)

        # --- DEBUG: swing-through sign check + raw/gated reward kernels (cfg.debug_reward_logging) ---
        # err_minus uses the CURRENT (correct) swing-through form target - vel*t_to_strike; err_plus uses
        # the FLIPPED form target + vel*t_to_strike. In-window we expect err_minus < err_plus (sign OK) and
        # at the exact strike (t_to_strike~0) the two collapse together. raw/gated are written by the reward
        # terms in hope_rewards.py. All held over the relevant mask so the reset-mean is the in-window value.
        if self.cfg.debug_reward_logging:
            for _k in (
                "dbg_err_minus_win", "dbg_err_plus_win", "dbg_err_minus_exact", "dbg_err_plus_exact",
                "dbg_racket_pos_raw", "dbg_racket_pos_gated",
                "dbg_racket_vel_raw", "dbg_racket_vel_gated",
                "dbg_racket_normal_raw", "dbg_racket_normal_gated",
                "dbg_base_raw", "dbg_base_gated",
            ):
                self.metrics[_k] = torch.zeros(self.num_envs, device=self.device)

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

    def _ref_racket_pos_at(self, motion, f: int) -> torch.Tensor:
        """Racket-center FK position (env-origin rel) at reference frame ``f``.

        Uses the SAME FK as :meth:`_compute_racket_state` / :meth:`_ensure_reference_strike_state`
        (racket body, or wrist + constant mount offset) but reads the reference MOTION's body poses.
        ``f`` is clamped to the valid frame range. Used by the clean-strike-velocity centered difference.
        """
        total = max(int(motion.time_step_total), 1)
        f = int(max(0, min(total - 1, f)))
        if self._racket_mode == "body":
            return motion._body_pos_w[f, self._racket_body_index]
        widx = self._wrist_body_index
        wpos = motion._body_pos_w[f, widx]
        wquat = motion._body_quat_w[f, widx]
        offset_w = quat_apply(wquat.unsqueeze(0), self._mount_offset[0:1]).squeeze(0)
        return wpos + offset_w

    def _ensure_reference_strike_state(self):
        """Cache the reference racket state at the strike frame (once, after the motion is loaded).

        Uses the SAME FK as :meth:`_compute_racket_state` (racket body, or wrist+mount offset) but
        reads the reference MOTION's body poses instead of the live robot, so that a robot perfectly
        tracking the clip would land its racket exactly on this state. The motion's raw body arrays
        are indexed in live-articulation order (``MotionLoader`` selects tracked bodies from them via
        the live ``body_indexes``), so the live ``_racket_body_index`` / ``_wrist_body_index`` index
        them directly. Positions are env-origin relative (as the motion stores them).
        """
        if self._ref_strike_cached:
            return
        motion = self._motion().motion  # MotionLoader
        total = max(int(motion.time_step_total), 1)
        strike_step = round(self.cfg.strike_phase * (total - 1))
        if self._racket_mode == "body":
            idx = self._racket_body_index
            pos, quat, lin, _ = self._reference_body_state(motion, strike_step, idx)
        else:
            widx = self._wrist_body_index
            wpos, wquat, wlin, wang = self._reference_body_state(motion, strike_step, widx, require_ang_vel=True)
            offset_w = quat_apply(wquat.unsqueeze(0), self._mount_offset[0:1]).squeeze(0)
            pos = wpos + offset_w
            lin = wlin + torch.cross(wang, offset_w, dim=-1)
            quat = quat_mul(wquat.unsqueeze(0), self._mount_quat[0:1]).squeeze(0)
        normal = matrix_from_quat(quat.unsqueeze(0))[0, :, self.cfg.mount_normal_axis] * self.cfg.mount_normal_sign

        # --- clean reference strike velocity ---------------------------------------------------------
        # Recompute the strike target velocity from the FINAL racket FK position by a centered finite
        # difference, so it is consistent with the position the policy actually tracks (the stored
        # body_lin_vel_w is FD'd/interpolated and ~1 m/s inconsistent at the racket tip — see cfg docs).
        # raw_lin = legacy single-frame stored velocity (kept for the flag-off path and the diagnostics).
        raw_lin = lin.detach().clone()
        dt = float(self._env.step_dt)
        # single-frame centered difference of the FK position (the consistency probe vs the stored vel)
        fd1 = (self._ref_racket_pos_at(motion, strike_step + 1) - self._ref_racket_pos_at(motion, strike_step - 1)) / (
            2.0 * dt
        )
        # windowed centered difference (the clean target velocity): wider baseline rejects single-frame jitter
        W = max(1, int(self.cfg.clean_strike_vel_window))
        clean_lin = (
            self._ref_racket_pos_at(motion, strike_step + W) - self._ref_racket_pos_at(motion, strike_step - W)
        ) / (2.0 * W * dt)
        if self.cfg.clean_reference_strike_velocity:
            lin = clean_lin

        self._ref_racket_pos_rel = pos.detach().clone()
        self._ref_racket_vel_w = lin.detach().clone()
        self._ref_racket_normal_w = (normal / (torch.norm(normal) + 1e-6)).detach().clone()
        # Reference base (root) XY at the strike — root is articulation body index 0 (same order the
        # motion arrays use). The base->racket horizontal offset couples base_target to racket_target.
        self._ref_base_pos_rel = self._reference_body_state(motion, strike_step, 0)[0].detach().clone()
        self._ref_reach_offset_xy = (self._ref_racket_pos_rel[:2] - self._ref_base_pos_rel[:2]).detach().clone()
        self._ref_strike_cached = True
        p, v, nrm = self._ref_racket_pos_rel, self._ref_racket_vel_w, self._ref_racket_normal_w
        b, off = self._ref_base_pos_rel, self._ref_reach_offset_xy
        raw_strike_speed = float(torch.norm(raw_lin))
        clean_strike_speed = float(torch.norm(clean_lin))
        raw_clean_vel_diff = float(torch.norm(raw_lin - clean_lin))
        raw_vs_fd_vel_diff = float(torch.norm(raw_lin - fd1))
        print(
            f"[RacketTargetCommand] reference_perturbed: strike frame {strike_step}/{total - 1} "
            f"(phase {self.cfg.strike_phase}); reference racket @ strike (env-origin rel): "
            f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
            f"vel=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f}) |v|={float(torch.norm(v)):.2f} "
            f"normal=({nrm[0]:.3f},{nrm[1]:.3f},{nrm[2]:.3f}); "
            f"reference base XY=({b[0]:.3f},{b[1]:.3f}) base->racket offset XY=({off[0]:.3f},{off[1]:.3f})",
            flush=True,
        )
        print(
            f"[RacketTargetCommand] strike-velocity denoise (clean_reference_strike_velocity="
            f"{self.cfg.clean_reference_strike_velocity}, window=+-{W}): "
            f"raw_strike_speed={raw_strike_speed:.3f} clean_strike_speed={clean_strike_speed:.3f} "
            f"raw_clean_vel_diff={raw_clean_vel_diff:.3f} raw_vs_fd_vel_diff={raw_vs_fd_vel_diff:.3f} "
            f"(target uses {'CLEAN' if self.cfg.clean_reference_strike_velocity else 'RAW'} velocity)",
            flush=True,
        )

    def _reference_body_state(self, motion, step: int, body_index: int, require_ang_vel: bool = False):
        """Return reference body state from MotionLoader's full-articulation private arrays.

        This is the current, intentional coupling point to upstream ``MotionLoader`` internals. Public
        ``MotionCommand`` buffers expose only the tracking subset, while the racket FK needs the full
        articulation body order so it can read the racket body or wrist mount. Keep direct private-field
        access centralized here until MotionLoader grows a public full-body state API.
        """
        required = ["_body_pos_w", "_body_quat_w", "_body_lin_vel_w"]
        if require_ang_vel:
            required.append("_body_ang_vel_w")
        missing = [name for name in required if not hasattr(motion, name)]
        if missing:
            raise AttributeError(
                "RacketTargetCommand requires MotionLoader full-body reference arrays "
                f"{required}, but missing {missing}. This is the HOPE coupling point for "
                "reference_perturbed racket FK; update _reference_body_state if upstream MotionLoader changes."
            )

        pos = motion._body_pos_w[step, body_index]
        quat = motion._body_quat_w[step, body_index]
        lin = motion._body_lin_vel_w[step, body_index]
        ang = motion._body_ang_vel_w[step, body_index] if require_ang_vel else None
        return pos, quat, lin, ang

    def _perturb_scale(self) -> float:
        """Curriculum factor in [start, 1.0] that widens the reference perturbation over training.

        Success-gated mode (default): return the running ``_curr_perturb_scale``, which advances only
        when the policy demonstrates exact-strike success (see :meth:`_update_metrics`). Otherwise fall
        back to the legacy open-loop ramp keyed to ``env.common_step_counter``. The returned scale is
        clamped to ``[ref_perturb_curriculum_start, 1.0]``.
        """
        start = float(self.cfg.ref_perturb_curriculum_start)
        if self.cfg.target_mode == "reference_perturbed" and self.cfg.ref_perturb_success_gated:
            scale = self._curr_perturb_scale
        else:
            steps = float(getattr(self._env, "common_step_counter", 0))
            c = float(self.cfg.ref_perturb_curriculum_steps)
            frac = 1.0 if c <= 0.0 else min(1.0, steps / c)
            scale = start + (1.0 - start) * frac
        return min(1.0, max(start, scale))

    def _ensure_ref_normal_per_clip(self):
        """Cache the reference paddle face normal at each clip's strike frame ([num_segments, 3]).

        Uses the SAME FK as the live racket normal (racket-body quat, or wrist-quat * mount_quat) but reads
        the reference MOTION's body quats at that clip's strike frame, so a robot tracking the clip lands
        its paddle normal exactly on this target. Used as the uniform-mode normal target (the sampled
        racket velocity is not the paddle-face direction).
        """
        if self._ref_normal_per_clip is not None:
            return
        motion = self._motion()
        ml = motion.motion
        nseg = ml.num_segments
        spc = tuple(self.cfg.strike_phase_per_clip)
        normals = torch.zeros(nseg, 3, device=self.device)
        for c in range(nseg):
            ss = int(ml.seg_start[c])
            sl = int(ml.seg_len[c])
            ph = float(spc[c]) if spc and len(spc) == nseg else float(self.cfg.strike_phase)
            f = ss + round(ph * (sl - 1))
            if self._racket_mode == "body":
                quat = ml._body_quat_w[f, self._racket_body_index]
            else:
                wquat = ml._body_quat_w[f, self._wrist_body_index]
                quat = quat_mul(wquat.unsqueeze(0), self._mount_quat[0:1]).squeeze(0)
            nrm = matrix_from_quat(quat.unsqueeze(0))[0, :, self.cfg.mount_normal_axis] * self.cfg.mount_normal_sign
            normals[c] = nrm / (torch.norm(nrm) + 1e-6)
        self._ref_normal_per_clip = normals
        print(f"[RacketTargetCommand] per-clip reference strike paddle normals: {normals.tolist()}", flush=True)

    def _sample_targets_uniform(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Independent box sampling (legacy mode). Ranges are PLACEHOLDERS not tied to the swing."""
        pos = origins.clone()
        motion = self._motion()
        if self._pos_range_per_clip_t is not None and motion._multiseg:
            # PER-CLIP position box (unified policy): each env samples x/y/z from ITS clip's box (added to
            # the env origin). The y range is SIGNED per clip (forehand -y / backhand +y encoded directly in
            # the box), so this REPLACES the shared x-range + |y|-sign + z-range logic below. Lets each clip's
            # target track its own reference strike point (e.g. backhand z~1.2 when strike_phase=0.50).
            clip = motion.clip_id[env_ids]                      # (n,) long, 0=forehand / 1=backhand
            rng_e = self._pos_range_per_clip_t[clip]            # (n, 3, 2): [env][x/y/z][lo/hi]
            lo = rng_e[..., 0]                                  # (n, 3)
            hi = rng_e[..., 1]                                  # (n, 3)
            pos[:, :3] += lo + (hi - lo) * torch.rand(n, 3, device=self.device)
        else:
            # Shared box (legacy / single-clip): identical sampling to before — backward compatible.
            pos[:, 0] += sample_uniform(*self.cfg.racket_pos_x_range, (n,), self.device)
            if motion._multiseg:
                # Unified policy: the target Y region is conditioned on the swing TYPE (clip) so forehand and
                # backhand regions are non-overlapping (HITTER §IV). Sample |y| and set the sign per clip:
                # forehand (clip 0) on -y if forehand_on_negative_y, backhand (clip 1) on the opposite side.
                clip = motion.clip_id[env_ids]
                ymag = sample_uniform(*self.cfg.racket_pos_y_abs_range, (n,), self.device)
                fh_sign = -1.0 if self.cfg.forehand_on_negative_y else 1.0
                sign = torch.where(clip == 0, fh_sign, -fh_sign)
                pos[:, 1] = origins[:, 1] + sign * ymag
            else:
                pos[:, 1] += sample_uniform(*self.cfg.racket_pos_y_range, (n,), self.device)
            pos[:, 2] += sample_uniform(*self.cfg.racket_pos_z_range, (n,), self.device)
        self.racket_target_pos_w[env_ids] = pos

        if self._vel_range_per_clip_t is not None and motion._multiseg:
            # PER-CLIP velocity (unified policy): each env samples from ITS clip's box, so the slower
            # backhand gets a lower target speed than the forehand instead of one shared box that
            # overshoots the backhand. Vectorized: gather each env's clip range, then uniform-sample.
            clip = motion.clip_id[env_ids]                      # (n,) long, 0=forehand / 1=backhand
            rng_e = self._vel_range_per_clip_t[clip]            # (n, 3, 2): [env][x/y/z][lo/hi]
            lo = rng_e[..., 0]                                  # (n, 3)
            hi = rng_e[..., 1]                                  # (n, 3)
            vel = lo + (hi - lo) * torch.rand(n, 3, device=self.device)
        else:
            # Shared box (legacy / single-clip): identical sampling to before — backward compatible.
            vel = torch.empty(n, 3, device=self.device)
            vel[:, 0] = sample_uniform(*self.cfg.racket_vel_x_range, (n,), self.device)
            vel[:, 1] = sample_uniform(*self.cfg.racket_vel_y_range, (n,), self.device)
            vel[:, 2] = sample_uniform(*self.cfg.racket_vel_z_range, (n,), self.device)
        self.racket_target_vel_w[env_ids] = vel

        if motion._multiseg:
            # Unified policy: the target paddle normal is the imitated swing's actual face normal at
            # strike (reachable by the imitation). The sampled racket velocity is the SWING-PATH direction,
            # which is ~18-110 deg off the +Y blade face, so normal_mode=velocity makes the normal goal
            # unsatisfiable (normal_pass=0 -> composite success stuck at 0).
            self._ensure_ref_normal_per_clip()
            clip = motion.clip_id[env_ids]
            normal = self._ref_normal_per_clip[clip]
        elif self.cfg.normal_mode == "velocity":
            normal = vel / (torch.norm(vel, dim=-1, keepdim=True) + 1e-6)
        else:  # "sampled"
            normal = torch.empty(n, 3, device=self.device)
            normal[:, 0] = sample_uniform(*self.cfg.racket_normal_x_range, (n,), self.device)
            normal[:, 1] = sample_uniform(*self.cfg.racket_normal_y_range, (n,), self.device)
            normal[:, 2] = sample_uniform(*self.cfg.racket_normal_z_range, (n,), self.device)
            normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        self.racket_target_normal_w[env_ids] = normal

    def _sample_targets_reference_perturbed(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Target = reference racket state @ strike + curriculum-scaled uniform perturbation.

        Guarantees the target is reachable by the imitated swing (a perfect imitator scores exactly),
        with the perturbation ball widening over training (``_perturb_scale``) for generalization.
        """
        self._ensure_reference_strike_state()
        scale = self._perturb_scale()
        dev = self.device
        pos_h = torch.tensor(self.cfg.ref_perturb_pos, device=dev) * scale
        vel_h = torch.tensor(self.cfg.ref_perturb_vel, device=dev) * scale
        nrm_h = float(self.cfg.ref_perturb_normal) * scale

        dpos = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * pos_h
        self.racket_target_pos_w[env_ids] = origins + self._ref_racket_pos_rel.unsqueeze(0) + dpos

        dvel = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * vel_h
        self.racket_target_vel_w[env_ids] = self._ref_racket_vel_w.unsqueeze(0) * self.cfg.ref_vel_scale + dvel

        dnrm = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * nrm_h
        normal = self._ref_racket_normal_w.unsqueeze(0) + dnrm
        self.racket_target_normal_w[env_ids] = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)

        self.metrics["ref_perturb_scale"][env_ids] = scale

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]

        # UNCONDITIONAL swing accounting: every resample STARTS a new swing attempt. On the
        # true-reset path (not a wrap) it also ENDS the previous attempt — count a pre-strike
        # fall if the env terminated before reaching the strike frame.
        self._count_swing_starts(env_ids, count_prestrike_falls=not self._resample_is_wrap)

        # Desired racket pos/vel/normal — either independent box sampling (legacy) or coupled to the
        # reference swing's strike state (reachable-by-construction; reimplement.md step 13 / rank 5).
        if self.cfg.target_mode == "reference_perturbed":
            self._sample_targets_reference_perturbed(env_ids, origins, n)
        else:
            self._sample_targets_uniform(env_ids, origins, n)

        # Desired base XY (world): COUPLE it to the racket target so standing there keeps the racket
        # reachable by the imitated swing — base_target = racket_target_xy - (reference base->racket
        # offset). Independent sampling used to fight the arm's reach (the base_position reward pulled
        # the base away from where the racket needed it). base_target_*_range is now a SMALL JITTER
        # around the coupled point. Legacy "uniform" mode keeps the old origin-relative sampling.
        if self.cfg.target_mode == "reference_perturbed":
            self._ensure_reference_strike_state()
            base_xy = self.racket_target_pos_w[env_ids][:, :2] - self._ref_reach_offset_xy.unsqueeze(0)
        else:
            # uniform: start at spawn, then WEAKLY couple the base toward the racket target's SIDEWAYS
            # offset (Y only; X is the fixed strike plane, so no forward repositioning). The base shifts a
            # fraction (base_couple_blend) of the target's Y offset, clamped to ±base_couple_max_offset, so
            # the robot leans/steps slightly toward far targets instead of stretching in place. blend=0 ->
            # the old spawn-only behaviour. This only moves a REWARD target — the racket target distribution
            # is unchanged. (No walking reference exists, so keep the blend small: it fights leg imitation.)
            base_xy = origins[:, :2].clone()
            blend = float(self.cfg.base_couple_blend)
            if blend > 0.0:
                racket_y_off = self.racket_target_pos_w[env_ids][:, 1] - origins[:, 1]
                base_xy[:, 1] += (blend * racket_y_off).clamp(
                    -self.cfg.base_couple_max_offset, self.cfg.base_couple_max_offset
                )
        base_xy[:, 0] += sample_uniform(*self.cfg.base_target_x_range, (n,), self.device)
        base_xy[:, 1] += sample_uniform(*self.cfg.base_target_y_range, (n,), self.device)
        self.base_target_pos_w[env_ids] = base_xy

        # Swing type. Unified multi-clip: it IS the imitated clip (forehand=clip 0 -> +1, backhand=clip 1
        # -> -1), matching the swing_type observation. Single-clip legacy: infer from the target Y side.
        motion = self._motion()
        if motion._multiseg:
            clip = motion.clip_id[env_ids]
            self.swing_sign[env_ids] = torch.where(clip == 0, 1.0, -1.0)
        else:
            base_y_nom = origins[:, 1] + self.cfg.base_nominal_offset[1]
            dy = self.racket_target_pos_w[env_ids][:, 1] - base_y_nom
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

    def _compute_strike_timing(self):
        """Refresh time_to_strike / pre_strike / strike_window from the CURRENT motion phase.

        The ``motion`` command term computes before this one (it is a parent-class field, registered
        first), so by now ``motion.time_steps`` is already advanced for this control step. Computing the
        timing here — at the top of _update_metrics, alongside the fresh racket FK — keeps the strike
        masks ALIGNED with the racket pose they gate. Previously the masks were only set in
        _update_command (which runs AFTER _update_metrics), so _update_metrics read a 1-step-stale
        time_to_strike: ``exact_strike`` fired one control frame LATE, after the paddle had flown ~one
        step past the strike (~12 cm at a 6 m/s swing) — which collapsed the measured position accuracy
        (exact-strike pos<7.5cm read ~11% instead of the true ~68%) while barely moving velocity (flat
        near the peak). That made strike_composite_success_exact ~6x pessimistic vs the honest probe.
        """
        motion = self._motion()
        ml = motion.motion
        if motion._multiseg:
            # Per-clip strike frame on the concatenated time axis: the contact phase differs per swing
            # (forehand peak ~0.36, backhand ~0.74), so resolve strike_step per env from its clip.
            if self._strike_phase_per_clip_t is None:
                sp = tuple(self.cfg.strike_phase_per_clip)
                if sp and len(sp) == ml.num_segments:
                    self._strike_phase_per_clip_t = torch.tensor([float(x) for x in sp], device=self.device)
                else:
                    self._strike_phase_per_clip_t = torch.full(
                        (ml.num_segments,), float(self.cfg.strike_phase), device=self.device
                    )
            clip = motion.clip_id
            seg_start = ml.seg_start[clip]
            seg_len = ml.seg_len[clip]
            phase = self._strike_phase_per_clip_t[clip]
            strike_step = seg_start + (phase * (seg_len - 1).float()).round().long()
            self.time_to_strike = (strike_step - motion.time_steps).float() * self._env.step_dt
        else:
            total = max(int(ml.time_step_total), 1)
            strike_step = round(self.cfg.strike_phase * (total - 1))
            self.time_to_strike = (strike_step - motion.time_steps).float() * self._env.step_dt
        self.pre_strike = self.time_to_strike > 0.0
        self.strike_window = self.time_to_strike.abs() <= self.cfg.strike_window_s

    def _update_command(self):
        motion = self._motion()
        # Timing is refreshed in _update_metrics (aligned with the FK); recompute here too so a direct
        # _update_command call outside the compute() path stays correct. Idempotent within a step
        # (motion.time_steps is unchanged between the two calls).
        self._compute_strike_timing()

        # Re-sample the target at each new swing. Use the motion command's robust just_resampled signal
        # (set this same step when it wrapped a swing) instead of a time_steps<prev heuristic — the latter
        # fails for the unified policy when a wrap jumps the index to a HIGHER concatenated segment start
        # (forehand->backhand). Targets for fresh episodes are sampled by the manager's reset.
        wrapped = torch.where(motion.just_resampled)[0] if hasattr(motion, "just_resampled") else \
            torch.where(motion.time_steps < self._prev_motion_steps)[0]
        if len(wrapped) > 0:
            # Wrap path: a wrapped env passed its strike frame alive (strike < seg end), so no
            # pre-strike fall is counted — the flag only gates fall accounting inside
            # _resample_command's _count_swing_starts hook.
            self._resample_is_wrap = True
            try:
                self._resample_command(wrapped)
            finally:
                self._resample_is_wrap = False
        self._prev_motion_steps = motion.time_steps.clone()

    def _count_swing_starts(self, env_ids, count_prestrike_falls: bool) -> None:
        """UNCONDITIONAL swing accounting (Phase A wandb fix). Increment-only here; the decay is
        applied once per step in _update_metrics next to the exact accumulators, so
        swing_completion_rate = exact_n_acc / swing_starts_acc shares one EMA timescale.
        NOTE: an episode TIMEOUT mid-swing counts as an uncompleted start (slight deflation,
        ~one boundary swing per 10 s episode) but never as a fall (terminated excludes timeouts)."""
        n = int(len(env_ids))
        if n == 0:
            return
        self._swing_starts_acc += float(n)
        motion = self._motion()
        if motion._multiseg:
            clips = motion.clip_id[env_ids]
            for c in self._clip_names:
                self._swing_starts_acc_c[c] += float((clips == c).sum())
        if count_prestrike_falls:
            term = self._env.termination_manager.terminated[env_ids]
            self._prestrike_fall_acc += float((term & self.pre_strike[env_ids]).sum())

    def _update_footwork_signals(self, racket_dist: torch.Tensor) -> None:
        """Base-FREE footwork-to-strike signals (reward/metric only; NEVER observed). The legs are driven
        to move by racket PROGRESS (reducing the racket->target distance), not by any base target. All
        guards degrade to 0 if a body/sensor cannot resolve, so this can never crash training."""
        data = self.robot.data
        # --- racket-target distance + dense progress (the base-free movement driver) ---
        self.racket_target_distance = racket_dist
        # progress = previous - current distance; clamp to kill the spike when the target resamples / the
        # episode resets (a target jump would otherwise read as huge spurious progress).
        self.racket_progress = (self._prev_racket_dist - racket_dist).clamp(-0.15, 0.15)
        self._prev_racket_dist = racket_dist.detach()
        self.metrics["racket_target_distance"] = racket_dist
        self.metrics["racket_progress"] = self.racket_progress
        self.metrics["racket_progress_prestrike"] = torch.where(
            self.pre_strike, self.racket_progress, torch.zeros_like(self.racket_progress)
        )
        # --- base stability components (training-only) ---
        pg = getattr(data, "projected_gravity_b", None)
        if pg is not None:
            self.proj_grav_xy = torch.norm(pg[:, :2], dim=-1)
        self.base_ang_vel_xy_norm = torch.norm(data.root_ang_vel_b[:, :2], dim=-1)
        self.vertical_speed = torch.abs(data.root_lin_vel_b[:, 2])
        self.metrics["proj_grav_xy"] = self.proj_grav_xy
        self.metrics["base_ang_vel_xy"] = self.base_ang_vel_xy_norm
        self.metrics["base_vertical_speed"] = self.vertical_speed
        # --- foot footwork signals (slip² / velocity / drag); feet may STEP, so this is PENALTY-only ---
        if self._foot_idx_robot and self._contact_sensor is not None and self._foot_idx_contact:
            f_force = torch.norm(self._contact_sensor.data.net_forces_w[:, self._foot_idx_contact, :], dim=-1)
            in_contact = (f_force > 10.0).float()  # (E,2)
            f_vel = data.body_lin_vel_w[:, self._foot_idx_robot, :]  # (E,2,3)
            f_xy_speed = torch.norm(f_vel[..., :2], dim=-1)  # (E,2)
            f_speed = torch.norm(f_vel, dim=-1)  # (E,2)
            f_height = data.body_pos_w[:, self._foot_idx_robot, 2]  # (E,2)
            self.foot_slip_sq = (in_contact * f_xy_speed.square()).sum(dim=-1)  # contact * ||v_xy||²
            self.foot_vel_sq = f_speed.square().sum(dim=-1)  # excessive/violent foot motion
            # Phase A fix: the old height gate (f_height < 0.10 m) sat BELOW the planted ankle
            # origin (~0.07 m), so EVERY low step counted as "dragging" — stepping itself was
            # taxed, one of the reasons the policy never learned to move left/right. "Drag" now
            # means lateral speed while the foot is LOADED (in contact) = sliding under load;
            # airborne swing-leg motion is free (foot_vel_sq still bounds violent motion).
            self.foot_drag = (in_contact * f_xy_speed).sum(dim=-1)
            self.metrics["foot_slip_sq"] = self.foot_slip_sq
            self.metrics["foot_vel_mean"] = f_speed.mean(dim=-1)
            self.metrics["foot_lift_rate"] = (1.0 - in_contact).mean(dim=-1)  # 0 = both planted, 1 = airborne
            self.metrics["foot_vel_at_strike"] = torch.where(
                self.strike_window, f_speed.mean(dim=-1), torch.zeros(self.num_envs, device=self.device)
            )
        # --- anti-arm-only: ARM joints near a limit + arm joint velocity (resolve arm joint idx once) ---
        if not getattr(self, "_arm_resolved", False):
            self._arm_resolved = True
            self._arm_joint_idx, self._leg_joint_idx, self._waist_twist_idx = [], [], []
            try:
                self._arm_joint_idx = list(self.robot.find_joints([".*shoulder.*", ".*elbow.*", ".*wrist.*"])[0])
            except Exception:
                pass
            try:
                self._leg_joint_idx = list(self.robot.find_joints([".*hip.*", ".*knee.*", ".*ankle.*"])[0])
            except Exception:
                pass
            # waist YAW+ROLL: the "twist/lean instead of step" DOFs the policy uses to face a lateral
            # target without moving its feet. Penalized (pre-strike) so reaching a far target needs footwork.
            # waist_pitch is EXCLUDED (it is the swing wind-up / natural lean, not a lateral-reach cheat).
            try:
                self._waist_twist_idx = list(self.robot.find_joints(["waist_yaw_joint", "waist_roll_joint"])[0])
            except Exception:
                pass
        limits = getattr(data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(data, "joint_pos_limits", None)
        if self._arm_joint_idx and limits is not None:
            ai = self._arm_joint_idx
            half = ((limits[:, ai, 1] - limits[:, ai, 0]) * 0.5).clamp(min=1e-6)
            d = torch.minimum(
                data.joint_pos[:, ai] - limits[:, ai, 0], limits[:, ai, 1] - data.joint_pos[:, ai]
            ).clamp(min=0.0)
            self.arm_overreach_frac = ((d / half) < 0.1).float().mean(dim=-1)  # within 10% of a limit
            self.metrics["arm_overreach_frac"] = self.arm_overreach_frac
            self.metrics["arm_joint_vel_max"] = torch.max(torch.abs(data.joint_vel[:, ai]), dim=-1).values
        # --- diagnostic: do the LEGS actually move before the strike? (footwork is happening) ---
        if self._leg_joint_idx:
            leg_vel = torch.max(torch.abs(data.joint_vel[:, self._leg_joint_idx]), dim=-1).values
            self.metrics["leg_joint_vel_max"] = leg_vel
            self.metrics["leg_moving_prestrike"] = torch.where(
                self.pre_strike, (leg_vel > 0.2).float(), torch.zeros(self.num_envs, device=self.device)
            )
        # --- anti twist-instead-of-step: |waist_yaw|+|waist_roll| deviation from the default (neutral,
        #     facing-forward) pose. This is the magnitude the prestrike_waist_twist reward penalizes, so
        #     reaching a lateral target by turning the torso (feet planted) becomes costly -> step instead. ---
        if self._waist_twist_idx:
            wi = self._waist_twist_idx
            self.waist_twist = torch.abs(data.joint_pos[:, wi] - data.default_joint_pos[:, wi]).sum(dim=-1)
            self.metrics["waist_twist_abs"] = self.waist_twist
            self.metrics["waist_twist_prestrike"] = torch.where(
                self.pre_strike, self.waist_twist, torch.zeros(self.num_envs, device=self.device)
            )

    def _update_metrics(self):
        # CommandTerm.compute() runs _update_metrics() BEFORE _update_command(), so refresh the
        # actual racket FK AND the strike timing here (once per step) — metrics, rewards, and
        # observations then all read the same fresh, phase-aligned buffers (rewards/obs read them
        # after the full command_manager.compute()). _compute_strike_timing must run before any
        # exact_strike / strike_window gating below (see its docstring: the old stale read measured
        # the strike one control frame too late).
        self._compute_racket_state()
        self._compute_strike_timing()
        origins = self._env.scene.env_origins
        # commanded base repositioning = distance of the base target from spawn (0 if coupling disabled).
        self.metrics["base_target_offset_norm"] = torch.norm(self.base_target_pos_w - origins[:, :2], dim=-1)
        pos_err = torch.norm(self.racket_pos_w - self.racket_target_pos_w, dim=-1)
        vel_err = torch.norm(self.racket_lin_vel_w - self.racket_target_vel_w, dim=-1)
        cos_ang = torch.sum(self.racket_normal_w * self.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
        normal_err_deg = torch.acos(cos_ang) * (180.0 / math.pi)
        base_err = torch.norm(self.base_pos_w[:, :2] - self.base_target_pos_w, dim=-1)
        base_pos_rel = self.base_pos_w[:, :2] - origins[:, :2]
        base_err_xy = self.base_pos_w[:, :2] - self.base_target_pos_w
        racket_pos_err_vec = self.racket_pos_w - self.racket_target_pos_w
        racket_vel_err_vec = self.racket_lin_vel_w - self.racket_target_vel_w

        # Episode-wide (instantaneous) errors.
        self.metrics["racket_pos_error"] = pos_err
        self.metrics["racket_vel_error"] = vel_err
        self.metrics["racket_normal_error_deg"] = normal_err_deg
        self.metrics["base_pos_error"] = base_err
        self.metrics["time_to_strike_s"] = self.time_to_strike
        self.metrics["pre_strike_flag"] = self.pre_strike.float()
        self.metrics["strike_window_hit_rate"] = self.strike_window.float()
        if self.cfg.target_mode == "reference_perturbed":
            self.metrics["ref_perturb_scale"] = torch.full_like(pos_err, self._perturb_scale())
        else:
            self.metrics["ref_perturb_scale"].zero_()
        # Per-axis ERROR components only (which direction is the miss?). The per-axis actual/target
        # state and the speed/normal-cos scalars were dropped as redundant wandb clutter.
        for axis_idx, axis in enumerate(("x", "y")):
            self.metrics[f"base_pos_{axis}"] = base_pos_rel[:, axis_idx]
            self.metrics[f"base_pos_error_{axis}"] = base_err_xy[:, axis_idx]
        for axis_idx, axis in enumerate(("x", "y", "z")):
            self.metrics[f"racket_pos_error_{axis}"] = racket_pos_err_vec[:, axis_idx]
            self.metrics[f"racket_vel_error_{axis}"] = racket_vel_err_vec[:, axis_idx]

        # Strike-window-gated: hold the value sampled during the most recent strike window. The gating
        # masks come from the previous _update_command (<=1-step / 20 ms lag at 50 Hz — negligible vs
        # the ±strike_window_s window). Between strikes the held value carries to the next reset.
        in_win = self.strike_window
        exact_strike = torch.abs(self.time_to_strike) <= (0.5 * self._env.step_dt + 1e-6)
        self.metrics["exact_strike_hit_rate"] = exact_strike.float()

        # --- DEBUG: swing-through sign verification (cfg.debug_reward_logging) -----------------------
        # err_minus = ||racket_pos - (target - vel*t_to_strike)||  (the CURRENT form used by the reward)
        # err_plus  = ||racket_pos - (target + vel*t_to_strike)||  (the FLIPPED form the user suspected)
        # Held over the strike window / exact strike so the reset-mean reports the in-window value. Expect
        # err_minus_win < err_plus_win (sign correct) and err_minus_exact ~= err_plus_exact (t~0 collapse).
        if self.cfg.debug_reward_logging:
            _ttf = self.time_to_strike.unsqueeze(-1)
            _tp_minus = self.racket_target_pos_w - self.racket_target_vel_w * _ttf
            _tp_plus = self.racket_target_pos_w + self.racket_target_vel_w * _ttf
            _err_minus = torch.norm(self.racket_pos_w - _tp_minus, dim=-1)
            _err_plus = torch.norm(self.racket_pos_w - _tp_plus, dim=-1)
            self.metrics["dbg_err_minus_win"] = torch.where(in_win, _err_minus, self.metrics["dbg_err_minus_win"])
            self.metrics["dbg_err_plus_win"] = torch.where(in_win, _err_plus, self.metrics["dbg_err_plus_win"])
            self.metrics["dbg_err_minus_exact"] = torch.where(
                exact_strike, _err_minus, self.metrics["dbg_err_minus_exact"]
            )
            self.metrics["dbg_err_plus_exact"] = torch.where(
                exact_strike, _err_plus, self.metrics["dbg_err_plus_exact"]
            )
        self.metrics["racket_pos_error_exact_strike"] = torch.where(
            exact_strike, pos_err, self.metrics["racket_pos_error_exact_strike"]
        )
        self.metrics["racket_vel_error_exact_strike"] = torch.where(
            exact_strike, vel_err, self.metrics["racket_vel_error_exact_strike"]
        )
        self.metrics["racket_normal_error_deg_exact_strike"] = torch.where(
            exact_strike, normal_err_deg, self.metrics["racket_normal_error_deg_exact_strike"]
        )
        # --- CONDITIONAL exact-strike success (the trustworthy, undiluted metric) -------------------
        # Old bug: strike_composite_success_exact was a per-env HELD value (last exact-strike result,
        # else init 0). CommandTerm.reset() logs mean(metric[env_ids]) over the RESETTING envs then zeros
        # them, so every env that reset without ever registering an exact-strike frame contributed 0 ->
        # the logged value was ~10x diluted vs the true conditional pass rate (raw probe ~0.32 logged
        # ~0.03), and the success-gated curriculum never advanced off ref_perturb_curriculum_start.
        # Fix: report the fraction of *exact-strike samples* that pass each threshold as a sample-weighted
        # EMA, broadcast to every env, so the reset-mean, the curriculum's .mean(), and the per-env value
        # all equal the conditional rate. pos/vel/normal are also logged separately.
        pass_pos = (pos_err < self.cfg.strike_success_pos_thresh) & exact_strike
        pass_vel = (vel_err < self.cfg.strike_success_vel_thresh) & exact_strike
        pass_normal = (normal_err_deg < self.cfg.strike_success_normal_thresh_deg) & exact_strike
        pass_comp = pass_pos & pass_vel & pass_normal
        decay = float(self.cfg.exact_success_decay)
        self._exact_n_acc = decay * self._exact_n_acc + float(exact_strike.sum())
        self._exact_pass_comp_acc = decay * self._exact_pass_comp_acc + float(pass_comp.sum())
        self._exact_pass_pos_acc = decay * self._exact_pass_pos_acc + float(pass_pos.sum())
        self._exact_pass_vel_acc = decay * self._exact_pass_vel_acc + float(pass_vel.sum())
        # 5/10 cm position buckets on the exact-strike sample (NOT the window-exit frame).
        _pass_5cm = (pos_err < 0.05) & exact_strike
        _pass_10cm = (pos_err < 0.10) & exact_strike
        self._exact_pass_5cm_acc = decay * self._exact_pass_5cm_acc + float(_pass_5cm.sum())
        self._exact_pass_10cm_acc = decay * self._exact_pass_10cm_acc + float(_pass_10cm.sum())
        self._exact_pass_normal_acc = decay * self._exact_pass_normal_acc + float(pass_normal.sum())
        # UNCONDITIONAL swing accounting: decay the start/fall accumulators at the SAME per-step
        # rate as the exact accumulators (increments happen in _count_swing_starts), then report
        #   swing_completion_rate = exact-strike arrivals / swing starts   (falls count against it)
        #   pre_strike_fall_rate  = pre-strike terminations / swing starts
        # These are the honest companions to the CONDITIONAL composite below, whose denominator
        # only contains exact-strike samples (pre-strike falls are invisible to it).
        self._swing_starts_acc = decay * self._swing_starts_acc
        self._prestrike_fall_acc = decay * self._prestrike_fall_acc
        for _c in self._clip_names:
            self._swing_starts_acc_c[_c] = decay * self._swing_starts_acc_c[_c]
        _s_denom = max(self._swing_starts_acc, 1e-6)
        _s_enough = self._swing_starts_acc >= float(self.cfg.exact_success_min_count)
        self.metrics["swing_completion_rate"][:] = min(self._exact_n_acc / _s_denom, 1.0) if _s_enough else 0.0
        self.metrics["pre_strike_fall_rate"][:] = min(self._prestrike_fall_acc / _s_denom, 1.0) if _s_enough else 0.0
        for _c, _cn in self._clip_names.items():
            _cd = max(self._swing_starts_acc_c[_c], 1e-6)
            _ce = self._swing_starts_acc_c[_c] >= float(self.cfg.exact_success_min_count)
            self.metrics[f"swing_completion_rate_{_cn}"][:] = (
                min(self._exact_n_acc_c[_c] / _cd, 1.0) if _ce else 0.0
            )

        enough = self._exact_n_acc >= float(self.cfg.exact_success_min_count)
        denom = max(self._exact_n_acc, 1e-6)
        self._exact_composite_rate = (self._exact_pass_comp_acc / denom) if enough else 0.0
        # Broadcast in place so the entries reset() zeros are refreshed before the next reset logs them.
        self.metrics["strike_composite_success_exact"][:] = self._exact_composite_rate
        self.metrics["strike_pos_pass_exact"][:] = (self._exact_pass_pos_acc / denom) if enough else 0.0
        self.metrics["strike_vel_pass_exact"][:] = (self._exact_pass_vel_acc / denom) if enough else 0.0
        self.metrics["strike_normal_pass_exact"][:] = (self._exact_pass_normal_acc / denom) if enough else 0.0
        # Exact-strike position accuracy buckets (comparable with composite: same mask + EMA denominator).
        self.metrics["exact_strike_pos_success_5cm"][:] = (self._exact_pass_5cm_acc / denom) if enough else 0.0
        self.metrics["exact_strike_pos_success_10cm"][:] = (self._exact_pass_10cm_acc / denom) if enough else 0.0
        # Distribution of position error over THIS step's exact-strike samples (p90 + mean), broadcast.
        _ex_errs = pos_err[exact_strike]
        if _ex_errs.numel() > 0:
            self.metrics["exact_strike_pos_err_mean"][:] = _ex_errs.mean()
            self.metrics["exact_strike_pos_err_p90"][:] = torch.quantile(_ex_errs, 0.90)
        self.metrics["exact_strike_sample_count_decayed"][:] = self._exact_n_acc
        # --- per-clip (forehand/backhand) breakdown of the exact-strike pass rates + errors -----------
        # Same sample-weighted EMA as the global block above, selected by the motion command's clip_id so
        # wandb shows each swing separately. pass_pos/vel/normal already include `& exact_strike`. Multiseg
        # (unified forehand+backhand) only; single-clip leaves these at 0.
        _motion = self._motion()
        if getattr(_motion, "_multiseg", False):
            _clip = _motion.clip_id
            for _c, _cn in self._clip_names.items():
                _sel = exact_strike & (_clip == _c)
                _self_f = _sel.float()
                self._exact_n_acc_c[_c] = decay * self._exact_n_acc_c[_c] + float(_sel.sum())
                self._exact_pass_pos_acc_c[_c] = decay * self._exact_pass_pos_acc_c[_c] + float((pass_pos & _sel).sum())
                self._exact_pass_vel_acc_c[_c] = decay * self._exact_pass_vel_acc_c[_c] + float((pass_vel & _sel).sum())
                self._exact_pass_normal_acc_c[_c] = decay * self._exact_pass_normal_acc_c[_c] + float((pass_normal & _sel).sum())
                self._exact_pass_comp_acc_c[_c] = decay * self._exact_pass_comp_acc_c[_c] + float((pass_comp & _sel).sum())
                self._exact_pos_err_sum_c[_c] = decay * self._exact_pos_err_sum_c[_c] + float((pos_err * _self_f).sum())
                self._exact_vel_err_sum_c[_c] = decay * self._exact_vel_err_sum_c[_c] + float((vel_err * _self_f).sum())
                self._exact_nrm_err_sum_c[_c] = decay * self._exact_nrm_err_sum_c[_c] + float((normal_err_deg * _self_f).sum())
                _n = self._exact_n_acc_c[_c]
                # rate = acc / n once enough decayed samples accumulated (else 0). errors = decayed mean
                # error over THIS clip's exact-strike samples. _scale folds in the "enough" gate.
                _scale = (1.0 / max(_n, 1e-6)) if _n >= float(self.cfg.exact_success_min_count) else 0.0
                self.metrics[f"strike_pos_pass_exact_{_cn}"][:] = self._exact_pass_pos_acc_c[_c] * _scale
                self.metrics[f"strike_vel_pass_exact_{_cn}"][:] = self._exact_pass_vel_acc_c[_c] * _scale
                self.metrics[f"strike_normal_pass_exact_{_cn}"][:] = self._exact_pass_normal_acc_c[_c] * _scale
                self.metrics[f"strike_composite_success_exact_{_cn}"][:] = self._exact_pass_comp_acc_c[_c] * _scale
                self.metrics[f"racket_pos_error_exact_strike_{_cn}"][:] = self._exact_pos_err_sum_c[_c] * _scale
                self.metrics[f"racket_vel_error_exact_strike_{_cn}"][:] = self._exact_vel_err_sum_c[_c] * _scale
                self.metrics[f"racket_normal_error_deg_exact_strike_{_cn}"][:] = self._exact_nrm_err_sum_c[_c] * _scale
        # Per-axis position error AT the exact strike frame (which axis is the miss?). The position-only
        # strike_success_exact was dropped — strike_pos_pass_exact above is the same signal, undiluted.
        _axis_err_exact = torch.abs(self.racket_pos_w - self.racket_target_pos_w)
        for _ai, _ax in enumerate(("x", "y", "z")):
            self.metrics[f"racket_pos_error_{_ax}_exact_strike"] = torch.where(
                exact_strike, _axis_err_exact[:, _ai], self.metrics[f"racket_pos_error_{_ax}_exact_strike"]
            )
        # Success-gated curriculum: widen the perturbation only once the smoothed CONDITIONAL exact-strike
        # composite success (fraction of exact-strike samples passing all three thresholds) clears the bar.
        if self.cfg.target_mode == "reference_perturbed" and self.cfg.ref_perturb_success_gated:
            if (
                self._curr_perturb_scale < 1.0
                and enough
                and self._exact_composite_rate > self.cfg.ref_perturb_advance_threshold
            ):
                self._curr_perturb_scale = min(
                    1.0, self._curr_perturb_scale + float(self.cfg.ref_perturb_advance_rate)
                )
        self.metrics["racket_pos_error_at_strike"] = torch.where(
            in_win, pos_err, self.metrics["racket_pos_error_at_strike"]
        )
        self.metrics["racket_vel_error_at_strike"] = torch.where(
            in_win, vel_err, self.metrics["racket_vel_error_at_strike"]
        )
        self.metrics["racket_normal_error_deg_at_strike"] = torch.where(
            in_win, normal_err_deg, self.metrics["racket_normal_error_deg_at_strike"]
        )
        self.metrics["strike_success"] = torch.where(
            in_win, (pos_err < self.cfg.strike_success_pos_thresh).float(), self.metrics["strike_success"]
        )
        # Base target is tracked before the strike, so log that error during the pre-strike phase.
        self.metrics["base_pos_error_pre_strike"] = torch.where(
            self.pre_strike, base_err, self.metrics["base_pos_error_pre_strike"]
        )

        # Swing-quality detail held at the most recent strike: actual/target paddle speed and the
        # per-axis position error (which direction is the miss?).
        racket_speed = torch.norm(self.racket_lin_vel_w, dim=-1)
        target_speed = torch.norm(self.racket_target_vel_w, dim=-1)
        axis_err = torch.abs(self.racket_pos_w - self.racket_target_pos_w)
        self.metrics["racket_speed_at_strike"] = torch.where(
            in_win, racket_speed, self.metrics["racket_speed_at_strike"]
        )
        self.metrics["racket_target_speed_at_strike"] = torch.where(
            in_win, target_speed, self.metrics["racket_target_speed_at_strike"]
        )
        self.metrics["racket_pos_error_x_at_strike"] = torch.where(
            in_win, axis_err[:, 0], self.metrics["racket_pos_error_x_at_strike"]
        )
        self.metrics["racket_pos_error_y_at_strike"] = torch.where(
            in_win, axis_err[:, 1], self.metrics["racket_pos_error_y_at_strike"]
        )
        self.metrics["racket_pos_error_z_at_strike"] = torch.where(
            in_win, axis_err[:, 2], self.metrics["racket_pos_error_z_at_strike"]
        )
        # Window-exit-held (kept for continuity; the trustworthy contact-frame version is
        # exact_strike_pos_success_5cm/10cm, computed on the exact-strike mask above).
        self.metrics["strike_success_5cm_window_exit"] = torch.where(
            in_win, (pos_err < 0.05).float(), self.metrics["strike_success_5cm_window_exit"]
        )
        self.metrics["strike_success_10cm_window_exit"] = torch.where(
            in_win, (pos_err < 0.10).float(), self.metrics["strike_success_10cm_window_exit"]
        )

        # Robot-health diagnostics (episode-wide, instantaneous).
        data = self.robot.data
        self.metrics["base_height"] = data.root_pos_w[:, 2]
        self.metrics["base_upright"] = matrix_from_quat(self.base_quat_w)[:, 2, 2]  # 1.0 = perfectly upright
        self.metrics["joint_vel_abs_max"] = torch.max(torch.abs(data.joint_vel), dim=-1).values
        # --- stability diagnostics: absolute base roll/pitch + foot contact/slip --------------------
        # Resolve foot body indices + the contact sensor once (robust to USD Link/link casing; degrades
        # gracefully to 0 if absent so it can never crash training).
        if not getattr(self, "_stab_resolved", False):
            self._stab_resolved = True
            self._foot_idx_robot, self._foot_idx_contact, self._contact_sensor = [], [], None
            try:
                self._foot_idx_robot = list(self.robot.find_bodies([".*ankle_roll.*"])[0])
            except Exception:
                pass
            try:
                cs = self._env.scene.sensors["contact_forces"]
                self._contact_sensor = cs
                self._foot_idx_contact = list(cs.find_bodies([".*ankle_roll.*"])[0])
            except Exception:
                pass
        _roll, _pitch, _ = euler_xyz_from_quat(self.base_quat_w)
        # wrap to (-180, 180] so a level base reads ~0 (euler_xyz_from_quat can return [0, 2pi))
        self.metrics["base_roll_deg"] = torch.rad2deg(torch.atan2(torch.sin(_roll), torch.cos(_roll)))
        self.metrics["base_pitch_deg"] = torch.rad2deg(torch.atan2(torch.sin(_pitch), torch.cos(_pitch)))
        if self._foot_idx_robot and self._contact_sensor is not None and self._foot_idx_contact:
            f_force = torch.norm(self._contact_sensor.data.net_forces_w[:, self._foot_idx_contact, :], dim=-1)  # (E,2)
            in_contact = (f_force > 10.0).float()  # 10 N = the sensor's force_threshold
            self.metrics["foot_contact_frac"] = in_contact.mean(dim=-1)  # 1.0 = both feet planted
            self.feet_contact_frac = self.metrics["foot_contact_frac"]  # clean attr for feet_contact reward
            f_speed = torch.norm(data.body_lin_vel_w[:, self._foot_idx_robot, :2], dim=-1)  # horizontal (E,2)
            _slip_sum = (f_speed * in_contact).sum(dim=-1)  # sum over feet of horizontal speed while in contact
            # metric = MEAN slip over the contacting feet (m/s; planted foot should be ~0).
            self.metrics["foot_slip_speed"] = _slip_sum / in_contact.sum(dim=-1).clamp(min=1.0)
            # reward signal = SUM over feet (so both slipping feet are penalized); pre_strike-gated in the reward.
            self.foot_slip_in_contact = _slip_sum
        # footwork-to-strike signals (racket progress, foot slip²/vel/drag, arm overreach, strike stability)
        self._update_footwork_signals(pos_err)
        if self._has_jpos_limits:
            limits = getattr(data, "soft_joint_pos_limits", None)
            if limits is None:
                limits = data.joint_pos_limits
            half_span = ((limits[..., 1] - limits[..., 0]) * 0.5).clamp(min=1e-6)
            dist = torch.minimum(data.joint_pos - limits[..., 0], limits[..., 1] - data.joint_pos).clamp(min=0.0)
            self.metrics["joint_pos_near_limit_frac"] = ((dist / half_span) < 0.1).float().mean(dim=-1)
        if self._has_torque:
            tau_abs = torch.abs(data.applied_torque)
            self.metrics["joint_torque_abs_mean"] = torch.mean(tau_abs, dim=-1)
            self.metrics["joint_torque_abs_max"] = torch.max(tau_abs, dim=-1).values
        act = getattr(self._env.action_manager, "action", None)
        if act is not None:
            a_abs = torch.abs(act)
            self.metrics["action_abs_mean"] = torch.mean(a_abs, dim=-1)
            self.metrics["action_abs_max"] = torch.max(a_abs, dim=-1).values
            prev_act = getattr(self._env.action_manager, "prev_action", None)
            if prev_act is not None:
                delta_abs = torch.abs(act - prev_act)
                self.metrics["action_delta_abs_mean"] = torch.mean(delta_abs, dim=-1)
                self.metrics["action_delta_abs_max"] = torch.max(delta_abs, dim=-1).values
            else:
                self.metrics["action_delta_abs_mean"].zero_()
                self.metrics["action_delta_abs_max"].zero_()

    # ------------------------------------------------------------------ #
    # Observation helpers (base-relative quantities)
    # ------------------------------------------------------------------ #
    def racket_target_pos_b(self) -> torch.Tensor:
        """Desired racket position relative to the base (yaw-heading frame). HITTER actor obs.

        PRIVILEGED: uses ``base_pos_w`` (world base position), which is fabricated on hardware
        (no localizer). Used by the `full` obs mode; the deploy-parity mode (legacy task name:
        `real_sensor_only`) replaces it with :meth:`racket_target_pos_b_rel`.
        """
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_target_pos_w - self.base_pos_w)

    def racket_target_pos_b_rel(self) -> torch.Tensor:
        """Desired racket position relative to the CURRENT racket (FK), in the yaw-heading frame.

        DEPLOY-HONEST (no world base position): expanding the rotation, the base position cancels::

            R_yaw^T (target_w - racket_w) = R_yaw^T(target_w - base_w) - R_yaw^T(racket_w - base_w)
                                          = (target in base frame) - (racket FK in base frame)

        Both terms are computable on the real robot from the planner's target + racket forward
        kinematics (joint encoders), WITHOUT a fabricated base pose. Replaces
        :meth:`racket_target_pos_b` in the deploy-parity observation mode (legacy task name:
        ``real_sensor_only``).
        """
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_target_pos_w - self.racket_pos_w)

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
    mount_normal_axis: int = 1  # racket-local +Y is the face normal (red/hitting face; confirmed in Step 11)
    mount_normal_sign: float = 1.0  # +1 = red/forehand face; -1 = black/backhand face

    # --- strike timing (fraction of the reference clip where the paddle meets the ball) ---
    strike_phase: float = 0.46  # HITTER clip: strike at frame 43/94 ≈ 0.46
    # Unified multi-clip (HITTER forehand+backhand single policy): per-clip strike phase, aligned with the
    # MotionLoader segment order (i.e. the order of motion files: forehand, backhand). Empty -> use the
    # scalar strike_phase for every clip. e.g. (0.36, 0.74) for forehand_new + backhand_new.
    strike_phase_per_clip: tuple = ()
    strike_window_s: float = 0.1  # half-window; goal-racket reward active within ±strike_window_s
    strike_success_pos_thresh: float = 0.075  # m; "strike_success" metric = fraction of strikes with racket pos error below this
    strike_success_vel_thresh: float = 0.5  # m/s; exact-strike racket velocity acceptance threshold
    strike_success_normal_thresh_deg: float = 15.0  # deg; exact-strike face-normal acceptance threshold

    # --- nominal stance (offset of the base from the env origin) ---
    base_nominal_offset: tuple[float, float, float] = (0.0, 0.0, 0.93)

    # --- target generation mode ---
    # "uniform": independent box sampling from the *_range fields below (legacy; the boxes are
    #   PLACEHOLDERS not tied to the swing, so the imitated swing's racket may never pass through them).
    # "reference_perturbed": target = the reference swing's racket state AT the strike frame (pos/vel/
    #   normal, computed by the same FK as the actual racket) + a curriculum-scaled uniform perturbation.
    #   Reachable by construction (a perfect imitator scores exactly); the *_range fields are ignored.
    target_mode: str = "reference_perturbed"

    # reference_perturbed perturbation (final half-extents; scaled 0->1 by the curriculum below).
    ref_perturb_pos: tuple[float, float, float] = (0.15, 0.20, 0.15)  # m, per-axis half-range
    ref_perturb_vel: tuple[float, float, float] = (1.0, 1.0, 0.8)  # m/s, per-axis half-range
    ref_perturb_normal: float = 0.30  # face-normal jitter magnitude (added then renormalized)
    # Curriculum: perturbation half-extents ramp from `start`*final to 1.0*final.
    # Success-gated mode (default): `_curr_perturb_scale` starts at `ref_perturb_curriculum_start` and
    # only advances (by `ref_perturb_advance_rate` per control step) once the smoothed exact-strike
    # composite success exceeds `ref_perturb_advance_threshold` — keeps the strike error inside the
    # racket reward kernel's responsive band until the policy demonstrably hits, then widens.
    # Open-loop fallback (success_gated=False): ramp over `ref_perturb_curriculum_steps` control steps
    # (env.common_step_counter); set steps<=0 to disable the ramp (always full).
    ref_perturb_curriculum_steps: int = 30000
    ref_perturb_curriculum_start: float = 0.05
    ref_perturb_success_gated: bool = True
    ref_perturb_advance_threshold: float = 0.30  # widen once smoothed exact-strike composite success > this
    ref_perturb_advance_rate: float = 1.0e-5  # scale increment per control step while above threshold

    # --- reference velocity scaling (stage slow -> fast hitting) ---
    # The sampled racket-velocity target is `ref_vel_scale * reference_vel + perturbation`. The reference
    # forehand clip strikes at ~6 m/s; set <1.0 to train a slower, more controllable hit FIRST (e.g. 0.6
    # -> ~3.6 m/s) and ramp back to 1.0 once the slow strike is accurate. NOTE: at scale!=1.0 the target
    # velocity no longer equals the imitated swing's velocity, so a perfect imitator no longer matches it
    # exactly (the "reachable by construction" guarantee holds for position/normal, not scaled velocity).
    ref_vel_scale: float = 1.0

    # --- clean reference strike velocity (denoise the target velocity) ---
    # The motion's stored body_lin_vel_w is a finite-difference (torch.gradient) of the 30->50 fps
    # interpolated joint trajectory propagated through FK (see scripts/csv_to_npz.py). At the fast,
    # high-jerk racket tip those FD/interpolation errors accumulate to ~1 m/s and are INCONSISTENT with
    # the position trajectory (stored-vel vs central-diff-of-pos differ ~1.1 m/s near the strike). Since
    # the racket-velocity reward target is essentially the reference velocity, that ~1 m/s noise is the
    # floor on racket_vel_error_exact_strike (velEx parked ~0.74 regardless of reward tuning).
    # When True, the cached strike target velocity is recomputed from the FINAL racket FK position
    # (body_pos_w, the same FK as the actual racket) by a centered finite difference over +-window frames,
    # which is consistent with the position the policy actually tracks and rejects single-frame jitter.
    # False keeps the legacy single-frame stored-velocity path.
    clean_reference_strike_velocity: bool = False
    clean_strike_vel_window: int = 2  # half-window (frames) for the centered finite difference (try 2 or 3)

    # --- debug logging (sign verification + raw/gated reward kernels) ---
    # When True, RacketTargetCommand logs dbg_err_{minus,plus}_{win,exact} (swing-through sign check) and
    # the reward terms log dbg_{racket_pos,racket_vel,racket_normal,base}_{raw,gated}. Pure logging; no
    # behaviour change. Turn off for production runs (extra wandb scalars).
    debug_reward_logging: bool = False

    # --- conditional exact-strike success metric (logging + curriculum gating) ---
    # The logged strike_*_pass_exact / strike_composite_success_exact are a sample-weighted EMA of the
    # exact-strike pass rate: acc = decay*acc + this-step-count each control step. decay ~0.99 gives a
    # ~100-step (~2 s @ 50 Hz) memory; higher = smoother but slower to reflect the current policy. The
    # rate (and the curriculum) only trust it once `exact_success_min_count` decayed samples accumulate.
    exact_success_decay: float = 0.99
    exact_success_min_count: float = 50.0

    # --- reachable racket-target workspace (offsets from the env origin, world frame, meters) ---
    # Used only by target_mode="uniform". PLACEHOLDER ranges (not the reference strike point).
    racket_pos_x_range: tuple[float, float] = (0.25, 0.55)
    racket_pos_y_range: tuple[float, float] = (-0.45, 0.45)
    # Unified multi-clip: |y| sampling range; the SIGN is set per clip (forehand on -y, backhand on +y,
    # per forehand_on_negative_y) so forehand/backhand target regions are non-overlapping (HITTER §IV).
    racket_pos_y_abs_range: tuple[float, float] = (0.05, 0.45)
    racket_pos_z_range: tuple[float, float] = (0.70, 1.15)

    # --- desired racket velocity (world frame, m/s) ---
    racket_vel_x_range: tuple[float, float] = (1.5, 4.0)
    racket_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    racket_vel_z_range: tuple[float, float] = (0.0, 1.5)
    # Optional PER-CLIP velocity boxes (uniform mode + unified multi-clip only). None -> use the SHARED
    # racket_vel_*_range above for every clip (BACKWARD COMPATIBLE: old behavior, nothing changes). When
    # set, it is a tuple indexed by clip_id (0=forehand, 1=backhand — same order as strike_phase_per_clip /
    # the command's _clip_names), each entry ((x_lo,x_hi),(y_lo,y_hi),(z_lo,z_hi)). Reason: the forehand and
    # backhand reference clips have DIFFERENT natural strike speeds (~2.6 vs ~2.0 m/s at the racket), so a
    # single shared box overshoots the slower backhand and its strike can never satisfy the velocity gate.
    # Confirmed by the MuJoCo per-clip eval probe: lowering only the backhand target box raised backhand
    # composite 0.32->0.79 (deterministic) / 0.39->0.77 (dither) with forehand byte-identical.
    racket_vel_range_per_clip: tuple | None = None

    # OPTIONAL per-clip racket target-POSITION boxes (uniform mode, unified multi-clip policy). None ->
    # use the shared racket_pos_x_range + |y|-sign + racket_pos_z_range box for every clip (BACKWARD
    # COMPATIBLE: old behavior, nothing changes). When set, it is a tuple indexed by clip_id (0=forehand,
    # 1=backhand — same order as strike_phase_per_clip), each entry ((x_lo,x_hi),(y_lo,y_hi),(z_lo,z_hi))
    # added to the env origin. NOTE the y range is SIGNED here (encode forehand -y / backhand +y directly),
    # so it REPLACES the shared |y|-sign logic. Reason: when strike_phase changes the strike frame, the
    # racket sits at a DIFFERENT height/depth per clip (e.g. backhand @ phase 0.50 -> z~1.22, above the
    # shared z<=1.05 box), so a shared box makes that clip's strike-frame position unreachable. Per-clip
    # boxes let each clip's target track its own reference strike point.
    racket_pos_range_per_clip: tuple | None = None

    # --- desired racket face normal ---
    normal_mode: str = "velocity"  # "velocity" (n = v/|v|) or "sampled"
    racket_normal_x_range: tuple[float, float] = (0.5, 1.0)
    racket_normal_y_range: tuple[float, float] = (-0.3, 0.3)
    racket_normal_z_range: tuple[float, float] = (-0.3, 0.3)

    # --- desired base XY target (offsets from the env origin, world frame, meters) ---
    base_target_x_range: tuple[float, float] = (-0.10, 0.10)
    base_target_y_range: tuple[float, float] = (-0.35, 0.35)
    # Weak base->racket coupling (UNIFORM mode only). base_couple_blend = fraction of the racket target's
    # sideways (Y) offset that the base target shifts toward; clamped to ±base_couple_max_offset meters.
    # 0.0 = disabled (spawn-only). Conservative because no walking reference exists (it fights leg imitation).
    base_couple_blend: float = 0.0
    base_couple_max_offset: float = 0.20

    # --- swing-type convention ---
    forehand_on_negative_y: bool = True  # right arm holds the paddle: target on -Y side -> forehand (+1)
