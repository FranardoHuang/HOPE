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

        # --- Tier-1 virtual incoming ball (rewardDesign.md): per-swing sampled incoming state and
        # the at-strike outcome caches read by the one-shot virtual_* reward terms. vb_fired is
        # recomputed EVERY step in _update_metrics (true only on a gated exact-strike frame), so the
        # cached outcome is consumed exactly once per swing. Buffers exist even when the feature is
        # off (all-zero / all-False) so the reward terms are safely inert.
        self.vb_vel_in_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vb_spin_in_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vb_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_landing_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.vb_landing_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_on_opponent = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_depth_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_net_z = torch.zeros(self.num_envs, device=self.device)
        self.vb_net_clear = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_topspin = torch.zeros(self.num_envs, device=self.device)
        self.vb_spin_out_norm = torch.zeros(self.num_envs, device=self.device)
        self._vb_params = None  # lazy venue-yaml load on first evaluation
        # Derived table landmarks (env frame), from geometry.py ITTF constants.
        from whole_body_tracking.tasks.table_tennis import geometry as _tt_geom

        self._vb_net_x = float(cfg.vb_table_near_x) + _tt_geom.NET_X
        self._vb_far_x = float(cfg.vb_table_near_x) + _tt_geom.TABLE_LENGTH
        self._vb_half_w = _tt_geom.TABLE_WIDTH / 2.0
        self._vb_net_top_z = float(cfg.vb_table_surface_z) + _tt_geom.NET_HEIGHT
        self._vb_ball_r = _tt_geom.BALL_RADIUS
        self._vb_target_xy = torch.tensor(
            [float(cfg.vb_target_x), float(cfg.vb_target_y)], device=self.device
        )
        # Sample-weighted EMA accumulators (same decay/min-count discipline as the exact-strike block).
        self._vb_exact_acc = 0.0
        self._vb_hit_acc = 0.0
        self._vb_net_acc = 0.0
        self._vb_land_valid_acc = 0.0
        self._vb_inb_acc = 0.0

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
        self._ref_racket_pos_rel_per_clip = None
        self._ref_racket_vel_w_per_clip = None
        self._ref_racket_normal_w_per_clip = None
        self._ref_base_pos_rel_per_clip = None
        self._ref_reach_offset_xy_per_clip = None

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
        # P2.3 adaptive-sigma driver: global decayed error-magnitude sums over exact-strike samples
        # (mean = sum / _exact_n_acc). Per-clip variants exist below; sigma needs one global signal.
        self._exact_pos_err_sum = 0.0
        self._exact_vel_err_sum = 0.0
        # Live adaptive sigmas (start at the cfg maxima = the hand-tuned YAML stds; only applied to
        # the reward terms when cfg.adaptive_sigma is on).
        self._adaptive_sigma_pos = float(cfg.sigma_pos_max)
        self._adaptive_sigma_vel = float(cfg.sigma_vel_max)

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
        # POST-strike falls (fall AFTER reaching the strike frame — the follow-through/recovery fall that
        # swing_completion_rate + pre_strike_fall_rate are both blind to; it was the actual backhand
        # deploy failure mode). Same swing-starts denominator as pre_strike_fall_rate.
        self._poststrike_fall_acc = 0.0
        # Per-clip fall attribution. NOTE: at reset time the MOTION command has already resampled
        # clip_id to the NEW swing (motion resets before racket_target), so falls are attributed via
        # _prev_clip_id — the clip snapshot taken at the END of the previous _update_command, i.e. the
        # clip the env was actually swinging when it fell.
        self._prestrike_fall_acc_c = {c: 0.0 for c in self._clip_names}
        self._poststrike_fall_acc_c = {c: 0.0 for c in self._clip_names}
        self._prev_clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Post-wrap recovery latch: >=0 = the clip whose swing JUST finished while this env sits in the
        # post-wrap hold. A fall during that hold is physically the PREVIOUS swing's recovery fall, but
        # at the wrap the timing already describes the NEXT swing (pre_strike=True, clip_id=new random
        # clip) — without the latch such falls book as pre-strike falls of a 50%-wrong clip, which would
        # invert exactly the backhand-recovery diagnosis these metrics exist for. -1 = not recovering.
        self._recover_from_clip = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # True only while _resample_command is invoked from the intra-episode WRAP path (see
        # _update_command): wraps start a new swing but never count a pre-strike fall (a wrapped
        # env necessarily passed its strike frame alive).
        self._resample_is_wrap = False

        # --- HER-style achieved-target replay buffers (see RacketTargetCommandCfg) -----------------
        # Per-clip ring buffers of the racket state the policy ACTUALLY produced at exact-strike frames:
        # position env-origin-relative (world minus env origin), velocity world. Written in
        # _update_metrics on the exact_strike mask (alive envs only — terminated envs were reset before
        # the command computes); read in _sample_targets_uniform with prob achieved_target_mix_prob.
        _absize = max(int(cfg.achieved_buffer_size), 1)
        self._ach_pos = {c: torch.zeros(_absize, 3, device=self.device) for c in self._clip_names}
        self._ach_vel = {c: torch.zeros(_absize, 3, device=self.device) for c in self._clip_names}
        self._ach_fill = {c: 0 for c in self._clip_names}
        self._ach_ptr = {c: 0 for c in self._clip_names}
        # Decayed counters for the logged replay fraction (same EMA timescale as the exact accumulators).
        self._resample_n_acc = 0.0
        self._replay_n_acc = 0.0

        # --- A1 target latency & time-variance (mocap->planner->runner realism) --------------------
        # MOTIVATION: training previously handed the actor a PERFECT, instantly-updated target; the
        # real loop (mocap -> planner -> runner) delivers it LATE (transport + planning latency),
        # NOISY (ball-prediction error that SHRINKS as the strike approaches — SMASH Eq. 14), and
        # REFINED mid-swing (the planner re-plans WHERE while the swing clock keeps running — PACE
        # injects sensor delays for the same reason). Without modeling this, the mocap-closed-loop
        # deployment faces out-of-distribution target dynamics. ALL knobs default OFF and the default
        # path is byte-identical: delay==0 & jitter==0 make the actor-visible views ALIAS the live
        # tensors (zero overhead, no extra RNG); midswing_resample_prob==0 short-circuits before any
        # RNG draw. Only the ACTOR-visible view is degraded — rewards, metrics, the privileged critic,
        # and the achieved-target-replay WRITE always use the TRUE live target.
        self._delay_steps = max(int(cfg.target_delay_steps), 0)
        self._jitter_pos = max(float(cfg.target_jitter_pos_per_s), 0.0)
        self._jitter_vel = max(float(cfg.target_jitter_vel_per_s), 0.0)
        # The actor view is materialized (separate tensors) whenever latency OR jitter is on;
        # otherwise the delayed_* attributes ARE the live tensors (which are only ever index-assigned
        # after __init__, so the alias stays valid for the whole run).
        self._actor_view_active = self._delay_steps > 0 or self._jitter_pos > 0.0 or self._jitter_vel > 0.0
        if self._delay_steps > 0:
            # Ring buffers (length delay+1) over the ACTOR-VISIBLE target quantities: the slot
            # written this step is read back `delay` pushes later (see _push_actor_target).
            # time_to_strike is NOT buffered ON PURPOSE: the swing clock is generated robot-side by
            # the deploy runner, not by the mocap link, so it carries no mocap latency.
            _L = self._delay_steps + 1
            self._delay_buf_pos = torch.zeros(_L, self.num_envs, 3, device=self.device)
            self._delay_buf_vel = torch.zeros(_L, self.num_envs, 3, device=self.device)
            self._delay_buf_sign = torch.ones(_L, self.num_envs, device=self.device)
            self._delay_ptr = 0
        if self._actor_view_active:
            self.delayed_racket_target_pos_w = self.racket_target_pos_w.clone()
            self.delayed_racket_target_vel_w = self.racket_target_vel_w.clone()
            self.delayed_swing_sign = self.swing_sign.clone()
        else:
            # Flags off: zero-overhead aliases of the live tensors (byte-identical baseline).
            self.delayed_racket_target_pos_w = self.racket_target_pos_w
            self.delayed_racket_target_vel_w = self.racket_target_vel_w
            self.delayed_swing_sign = self.swing_sign
        # A1 metrics: per-step per-env redraw indicator (wandb reset-mean = per-step mid-swing
        # refinement fraction) + the constant delay-in-effect broadcast (refreshed every step in
        # _update_metrics because CommandTerm.reset() zeros metric entries of resetting envs).
        self.metrics["midswing_resample_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_delay_steps_in_effect"] = torch.full(
            (self.num_envs,), float(self._delay_steps), device=self.device
        )

        # Strike timing / gating.
        self.time_to_strike = torch.zeros(self.num_envs, device=self.device)
        self.pre_strike = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.strike_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Episode-wide tracking errors (instantaneous; averaged over terminating envs at reset).
        self.metrics["racket_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["adaptive_sigma_pos"] = torch.full((self.num_envs,), float(cfg.sigma_pos_max), device=self.device)
        self.metrics["adaptive_sigma_vel"] = torch.full((self.num_envs,), float(cfg.sigma_vel_max), device=self.device)
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
        # Tier-1 virtual-ball outcome rates (broadcast sample-weighted EMAs, exact-strike denominator
        # for hit rate; hit (captured) denominator for the outcome rates). Only logged when enabled.
        if cfg.virtual_ball:
            for _vk in (
                "virtual_hit_rate", "virtual_net_clear_rate", "virtual_land_valid_rate",
                "virtual_land_inbounds_rate", "virtual_land_err_m", "virtual_topspin_revs",
                "virtual_approach_speed",
            ):
                self.metrics[_vk] = torch.zeros(self.num_envs, device=self.device)
        # UNCONDITIONAL swing accounting (Phase A): completion_rate = exact-strike arrivals / swing
        # STARTS (falls count against it, unlike the conditional composite above); fall rate before
        # the strike frame. Broadcast scalars like the pass rates.
        self.metrics["swing_completion_rate"] = torch.zeros(self.num_envs, device=self.device)
        for _cname in self._clip_names.values():
            self.metrics[f"swing_completion_rate_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_strike_fall_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Post-strike (follow-through/recovery) falls + per-clip fall attribution: the multi-swing
        # episode's real recovery signal. pre_strike_fall_rate alone hides a policy that hits and THEN
        # falls (100% completion, 0% pre-strike falls — the actual backhand deploy failure signature).
        self.metrics["post_strike_fall_rate"] = torch.zeros(self.num_envs, device=self.device)
        for _cname in self._clip_names.values():
            self.metrics[f"pre_strike_fall_rate_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"post_strike_fall_rate_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
        # HER-style achieved-target replay diagnostics: fraction of resampled targets drawn from the
        # achieved buffer (EMA; ~achieved_target_mix_prob once the buffers are filled) + per-clip fill.
        self.metrics["achieved_replay_frac"] = torch.zeros(self.num_envs, device=self.device)
        for _cname in self._clip_names.values():
            self.metrics[f"achieved_buffer_fill_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
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
        self._progress_reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
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

    def _strike_frame_for_clip(self, motion, clip_id: int) -> tuple[int, float, int, int]:
        """Return (global strike frame, phase, segment start, segment len) for one reference clip."""
        nseg = int(motion.num_segments)
        if clip_id < 0 or clip_id >= nseg:
            raise IndexError(f"clip_id {clip_id} out of range for {nseg} segments")
        seg_start = int(motion.seg_start[clip_id].item())
        seg_len = int(motion.seg_len[clip_id].item())
        spc = tuple(self.cfg.strike_phase_per_clip)
        phase = float(spc[clip_id]) if spc and len(spc) == nseg else float(self.cfg.strike_phase)
        strike_step = seg_start + round(phase * (seg_len - 1))
        return int(strike_step), phase, seg_start, seg_len

    def _ref_racket_pos_at(
        self,
        motion,
        f: int,
        *,
        clip_start: int = 0,
        clip_end: int | None = None,
    ) -> torch.Tensor:
        """Racket-center FK position (env-origin rel) at reference frame ``f``.

        Uses the SAME FK as :meth:`_compute_racket_state` /
        :meth:`_ensure_reference_strike_state` (racket body, or wrist + constant mount offset) but reads
        the reference MOTION's body poses. ``f`` is clamped to the requested clip segment so clean
        centered-difference velocities never leak across a concatenated forehand/backhand boundary.
        """
        total = max(int(motion.time_step_total), 1)
        hi = total - 1 if clip_end is None else int(clip_end)
        lo = int(clip_start)
        f = int(max(lo, min(hi, f)))
        if self._racket_mode == "body":
            return motion._body_pos_w[f, self._racket_body_index]
        widx = self._wrist_body_index
        wpos = motion._body_pos_w[f, widx]
        wquat = motion._body_quat_w[f, widx]
        offset_w = quat_apply(wquat.unsqueeze(0), self._mount_offset[0:1]).squeeze(0)
        return wpos + offset_w

    def _ensure_reference_strike_state(self):
        """Cache per-clip reference racket/base states at each clip's strike frame.

        Target sampling in ``reference_perturbed`` mode must be centered on the exact teacher clip the
        env is imitating. The old single cached state was fine for one clip, but a unified forehand+
        backhand policy needs separate strike position, velocity, face normal, and base->racket reach
        offsets for each concatenated MotionLoader segment.
        """
        if self._ref_strike_cached:
            return
        motion = self._motion().motion  # MotionLoader
        nseg = int(motion.num_segments)
        pos_all = torch.zeros(nseg, 3, device=self.device)
        vel_all = torch.zeros(nseg, 3, device=self.device)
        nrm_all = torch.zeros(nseg, 3, device=self.device)
        base_all = torch.zeros(nseg, 3, device=self.device)
        reach_all = torch.zeros(nseg, 2, device=self.device)
        W = max(1, int(self.cfg.clean_strike_vel_window))
        dt = float(self._env.step_dt)
        report_lines = []

        for clip_id in range(nseg):
            strike_step, phase, seg_start, seg_len = self._strike_frame_for_clip(motion, clip_id)
            seg_end = seg_start + seg_len - 1
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

            # --- clean reference strike velocity --------------------------------------------------
            # Recompute the strike target velocity from the FINAL racket FK position by a centered
            # finite difference (clamped to this clip's segment), so it is consistent with the
            # position the policy actually tracks (the stored body_lin_vel_w is FD'd/interpolated and
            # ~1 m/s inconsistent at the racket tip — see cfg docs). raw_lin = legacy single-frame
            # stored velocity (kept for the flag-off path and the diagnostics).
            raw_lin = lin.detach().clone()
            fd1 = (
                self._ref_racket_pos_at(motion, strike_step + 1, clip_start=seg_start, clip_end=seg_end)
                - self._ref_racket_pos_at(motion, strike_step - 1, clip_start=seg_start, clip_end=seg_end)
            ) / (2.0 * dt)
            clean_lin = (
                self._ref_racket_pos_at(motion, strike_step + W, clip_start=seg_start, clip_end=seg_end)
                - self._ref_racket_pos_at(motion, strike_step - W, clip_start=seg_start, clip_end=seg_end)
            ) / (2.0 * W * dt)
            if self.cfg.clean_reference_strike_velocity:
                lin = clean_lin

            pos_all[clip_id] = pos.detach().clone()
            vel_all[clip_id] = lin.detach().clone()
            nrm_all[clip_id] = (normal / (torch.norm(normal) + 1e-6)).detach().clone()
            # Reference base (root) at the strike — root is articulation body index 0 (same order the
            # motion arrays use). The base->racket horizontal offset couples base_target to racket_target.
            base_all[clip_id] = self._reference_body_state(motion, strike_step, 0)[0].detach().clone()
            reach_all[clip_id] = (pos_all[clip_id, :2] - base_all[clip_id, :2]).detach().clone()

            cname = self._clip_names.get(clip_id, f"clip{clip_id}")
            p = pos_all[clip_id]
            v = vel_all[clip_id]
            nrm = nrm_all[clip_id]
            b = base_all[clip_id]
            off = reach_all[clip_id]
            report_lines.append(
                f"  {cname}: strike frame {strike_step}/{seg_end} (phase {phase:.3f}) "
                f"pos=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f}) "
                f"vel=({float(v[0]):.3f},{float(v[1]):.3f},{float(v[2]):.3f}) "
                f"|v|={float(torch.norm(v)):.2f} "
                f"normal=({float(nrm[0]):.3f},{float(nrm[1]):.3f},{float(nrm[2]):.3f}) "
                f"baseXY=({float(b[0]):.3f},{float(b[1]):.3f}) "
                f"reachXY=({float(off[0]):.3f},{float(off[1]):.3f}) "
                f"raw_speed={float(torch.norm(raw_lin)):.3f} "
                f"clean_speed={float(torch.norm(clean_lin)):.3f} "
                f"raw_clean_diff={float(torch.norm(raw_lin - clean_lin)):.3f} "
                f"raw_fd_diff={float(torch.norm(raw_lin - fd1)):.3f}"
            )

        self._ref_racket_pos_rel_per_clip = pos_all
        self._ref_racket_vel_w_per_clip = vel_all
        self._ref_racket_normal_w_per_clip = nrm_all
        self._ref_base_pos_rel_per_clip = base_all
        self._ref_reach_offset_xy_per_clip = reach_all
        # Legacy single-clip fields (no in-file consumers besides diagnostics): mirror clip 0.
        self._ref_racket_pos_rel = pos_all[0].detach().clone()
        self._ref_racket_vel_w = vel_all[0].detach().clone()
        self._ref_racket_normal_w = nrm_all[0].detach().clone()
        self._ref_base_pos_rel = base_all[0].detach().clone()
        self._ref_reach_offset_xy = reach_all[0].detach().clone()
        self._ref_strike_cached = True
        print(
            "[RacketTargetCommand] reference strike centers per clip "
            f"(clean_reference_strike_velocity={self.cfg.clean_reference_strike_velocity}, window=+-{W}):\n"
            + "\n".join(report_lines),
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
        """Cache the reference paddle face normal at each clip's strike frame ([num_segments, 3])."""
        if self._ref_normal_per_clip is not None:
            return
        self._ensure_reference_strike_state()
        assert self._ref_racket_normal_w_per_clip is not None
        self._ref_normal_per_clip = self._ref_racket_normal_w_per_clip

    def _sample_targets_uniform(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Independent box sampling (legacy mode). Ranges are PLACEHOLDERS not tied to the swing."""
        pos = origins.clone()
        motion = self._motion()
        if self._pos_range_per_clip_t is not None and motion._multiseg:
            # PER-CLIP position box (unified policy): each env samples x/y/z from ITS clip's box (added to
            # the env origin). The y range is SIGNED per clip (the configured box is used directly, so a
            # near-center backhand box is valid and does not go through the shared +/-|y| fallback). This
            # replaces the shared x-range + |y|-sign + z-range logic below and lets each clip track its own
            # reference strike point.
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

        # --- HER-style achieved-target replay (mixture) --------------------------------------------
        # With prob achieved_target_mix_prob, overwrite the freshly box-sampled pos+vel with a jittered
        # PREVIOUSLY-ACHIEVED strike state from this clip's ring buffer (written at exact-strike frames
        # in _update_metrics). Replayed targets are reachable-by-demonstration, so the target
        # distribution stops asking for points the taught swing never passes through (Ace/HER, adapted
        # forward-looking for on-policy PPO — retroactive relabel would be obs-inconsistent). Clamped
        # into the per-clip box inflated by achieved_clamp_inflate so replay can neither collapse the
        # target support nor drift outside the deploy runner's hand-synced target clips. Non-replayed
        # envs keep the pure box sample; the per-clip reference normal below is shared by both paths.
        if self.cfg.achieved_target_mix_prob > 0.0 and motion._multiseg:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            clip_all = motion.clip_id[env_ids_t]
            replay = torch.rand(n, device=self.device) < float(self.cfg.achieved_target_mix_prob)
            self._resample_n_acc += float(n)
            infl = 1.0 + float(self.cfg.achieved_clamp_inflate)
            for c in self._clip_names:
                fill = self._ach_fill.get(c, 0)
                if fill < int(self.cfg.achieved_min_fill):
                    continue
                sel = replay & (clip_all == c)
                m = int(sel.sum())
                if m == 0:
                    continue
                rows = torch.randint(0, fill, (m,), device=self.device)
                rpos = self._ach_pos[c][rows] + (torch.rand(m, 3, device=self.device) * 2.0 - 1.0) * float(
                    self.cfg.achieved_jitter_pos
                )
                rvel = self._ach_vel[c][rows] + (torch.rand(m, 3, device=self.device) * 2.0 - 1.0) * float(
                    self.cfg.achieved_jitter_vel
                )
                if self._pos_range_per_clip_t is not None and c < self._pos_range_per_clip_t.shape[0]:
                    lo, hi = self._pos_range_per_clip_t[c, :, 0], self._pos_range_per_clip_t[c, :, 1]
                    ctr, half = (lo + hi) * 0.5, (hi - lo) * 0.5 * infl
                    rpos = torch.min(torch.max(rpos, ctr - half), ctr + half)
                if self._vel_range_per_clip_t is not None and c < self._vel_range_per_clip_t.shape[0]:
                    lo, hi = self._vel_range_per_clip_t[c, :, 0], self._vel_range_per_clip_t[c, :, 1]
                    ctr, half = (lo + hi) * 0.5, (hi - lo) * 0.5 * infl
                    rvel = torch.min(torch.max(rvel, ctr - half), ctr + half)
                ids_sel = env_ids_t[sel]
                self.racket_target_pos_w[ids_sel] = self._env.scene.env_origins[ids_sel] + rpos
                self.racket_target_vel_w[ids_sel] = rvel
                self._replay_n_acc += float(m)

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
        """Target = this env's reference racket state @ strike + curriculum-scaled perturbation.

        The target center is selected by ``motion.clip_id`` for unified forehand+backhand training, so
        the policy no longer has to imitate one teacher strike while chasing a different sampled target.
        """
        self._ensure_reference_strike_state()
        assert self._ref_racket_pos_rel_per_clip is not None
        assert self._ref_racket_vel_w_per_clip is not None
        assert self._ref_racket_normal_w_per_clip is not None
        motion = self._motion()
        if motion._multiseg:
            clip = motion.clip_id[env_ids]
        else:
            clip = torch.zeros(n, dtype=torch.long, device=self.device)
        ref_pos = self._ref_racket_pos_rel_per_clip[clip]
        ref_vel = self._ref_racket_vel_w_per_clip[clip]
        ref_nrm = self._ref_racket_normal_w_per_clip[clip]

        scale = self._perturb_scale()
        dev = self.device
        pos_h = torch.tensor(self.cfg.ref_perturb_pos, device=dev) * scale
        vel_h = torch.tensor(self.cfg.ref_perturb_vel, device=dev) * scale
        nrm_h = float(self.cfg.ref_perturb_normal) * scale

        dpos = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * pos_h
        self.racket_target_pos_w[env_ids] = origins + ref_pos + dpos

        dvel = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * vel_h
        self.racket_target_vel_w[env_ids] = ref_vel * self.cfg.ref_vel_scale + dvel

        dnrm = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * nrm_h
        normal = ref_nrm + dnrm
        self.racket_target_normal_w[env_ids] = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)

        self.metrics["ref_perturb_scale"][env_ids] = scale

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]
        motion = self._motion()

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
            assert self._ref_reach_offset_xy_per_clip is not None
            if motion._multiseg:
                clip = motion.clip_id[env_ids]
            else:
                clip = torch.zeros(n, dtype=torch.long, device=self.device)
            base_xy = self.racket_target_pos_w[env_ids][:, :2] - self._ref_reach_offset_xy_per_clip[clip]
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

        # Tier-1 virtual incoming ball: one (v_in, omega_in) per swing. The ball's position at the
        # strike time is the racket target BY CONSTRUCTION (the sampler defines the ball to arrive
        # there), so only velocity + spin are sampled. Boxes stay inside the venue-fit envelope.
        if self.cfg.virtual_ball:
            self.vb_vel_in_w[env_ids, 0] = sample_uniform(*self.cfg.vb_vel_x_range, (n,), self.device)
            self.vb_vel_in_w[env_ids, 1] = sample_uniform(*self.cfg.vb_vel_y_range, (n,), self.device)
            self.vb_vel_in_w[env_ids, 2] = sample_uniform(*self.cfg.vb_vel_z_range, (n,), self.device)
            _s = float(self.cfg.vb_spin_abs_max)
            self.vb_spin_in_w[env_ids] = sample_uniform(-_s, _s, (n, 3), self.device)

        # Stamp the motion phase baseline for these envs so the per-swing wrap detector in
        # _update_command does not immediately re-trigger after this (e.g. reset-time) resample.
        self._prev_motion_steps[env_ids] = self._motion().time_steps[env_ids]
        self._prev_racket_dist[env_ids] = torch.norm(
            self.racket_pos_w[env_ids] - self.racket_target_pos_w[env_ids], dim=-1
        ).detach()
        self.racket_progress[env_ids] = 0.0
        self._progress_reset_mask[env_ids] = True

        # A1 target latency: a TRUE reset (not an intra-episode wrap) starts a fresh "deploy
        # session" — the runner latches the first planner target before the policy steps, so the
        # actor-visible view (and the whole ring buffer) is backfilled with the fresh target: no
        # cross-episode target leakage. Intra-episode WRAPS are deliberately NOT backfilled — the
        # next swing's target reaching the actor `delay` steps late is exactly the latency modeled.
        if self._actor_view_active and not self._resample_is_wrap:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self.delayed_racket_target_pos_w[ids] = self.racket_target_pos_w[ids]
            self.delayed_racket_target_vel_w[ids] = self.racket_target_vel_w[ids]
            self.delayed_swing_sign[ids] = self.swing_sign[ids]
            if self._delay_steps > 0:
                self._delay_buf_pos[:, ids] = self.racket_target_pos_w[ids].unsqueeze(0)
                self._delay_buf_vel[:, ids] = self.racket_target_vel_w[ids].unsqueeze(0)
                self._delay_buf_sign[:, ids] = self.swing_sign[ids].unsqueeze(0)

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
            # (v2 blade re-plane: forehand 0.47, backhand 0.333), so resolve strike_step per env from its clip.
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
            if motion._multiseg:
                # Latch the clip that JUST finished (before _prev_clip_id is re-snapshotted below):
                # while the post-wrap hold lasts, a fall belongs to THIS swing's recovery.
                self._recover_from_clip[wrapped] = self._prev_clip_id[wrapped]
            self._resample_is_wrap = True
            try:
                self._resample_command(wrapped)
            finally:
                self._resample_is_wrap = False
        # The recovery window ends when the post-wrap hold expires (the new swing's clock starts
        # advancing) — from then on falls are genuinely pre-strike of the new clip. in_hold is this
        # step's post-decrement hold state, so a zero-length hold clears the latch immediately.
        if motion._multiseg and hasattr(motion, "in_hold"):
            self._recover_from_clip[~motion.in_hold] = -1
        self._prev_motion_steps = motion.time_steps.clone()
        # Snapshot the clip each env is swinging THIS step: at the next true reset the motion command
        # will already have resampled clip_id, so fall attribution reads this snapshot instead.
        if motion._multiseg:
            self._prev_clip_id = motion.clip_id.clone()

        # --- A1 mid-swing target refinement (the planner refines WHERE, not WHEN) -------------------
        # Each step, envs still approaching the strike (pre_strike AND time_to_strike > tts floor)
        # re-draw their target with per-step prob p, exactly as the deploy planner refines its ball
        # prediction mid-swing. ONLY the target sampling runs (position/velocity/normal through the
        # existing uniform / per-clip-box / reference-perturbed path, including the HER achieved-target
        # mixture inside it):
        #   * strike timing untouched — same strike step, the swing clock keeps running;
        #   * NO _count_swing_starts — a refinement is not a new swing attempt (metrics denominators
        #     would otherwise be inflated);
        #   * base target / swing type / _prev_motion_steps untouched;
        #   * the racket-progress baseline is reset via _progress_reset_mask (same mechanism as the
        #     resample path) so the target jump creates no fake progress reward;
        #   * the achieved-target replay WRITE is unaffected (it stores the LIVE target state at
        #     exact-strike frames, and tts floor > 0 keeps refinement away from the strike frame).
        # prob==0 (default) short-circuits before any RNG draw — byte-identical baseline.
        _ms_prob = float(self.cfg.midswing_resample_prob)
        if _ms_prob > 0.0:
            eligible = self.pre_strike & (self.time_to_strike > float(self.cfg.midswing_resample_tts_floor))
            redraw = eligible & (torch.rand(self.num_envs, device=self.device) < _ms_prob)
            ids = torch.where(redraw)[0]
            if len(ids) > 0:
                origins = self._env.scene.env_origins[ids]
                if self.cfg.target_mode == "reference_perturbed":
                    self._sample_targets_reference_perturbed(ids, origins, len(ids))
                else:
                    self._sample_targets_uniform(ids, origins, len(ids))
                self._prev_racket_dist[ids] = torch.norm(
                    self.racket_pos_w[ids] - self.racket_target_pos_w[ids], dim=-1
                ).detach()
                self.racket_progress[ids] = 0.0
                self._progress_reset_mask[ids] = True
            # Per-env 0/1 indicator; the wandb reset-mean = per-step refinement fraction (~ prob *
            # eligible fraction). Written every step while the feature is on so zero-redraw steps count.
            self.metrics["midswing_resample_count"] = redraw.float()

        # A1 target latency/jitter: refresh the ACTOR-visible target view once per step (no-op alias
        # when the knobs are off). Runs LAST so it sees this step's wrap/refinement target updates.
        self._push_actor_target()

    def _push_actor_target(self):
        """A1: refresh the ACTOR-visible target view once per control step (latency + jitter).

        Applied on PUSH (not on read) so the jitter is drawn ONCE per step and every actor obs term
        reads the same tensor within the step (determinism). The jitter std decays with the time to
        strike (SMASH Eq. 14 — the mocap ball prediction converges as the strike approaches):
        per-step std = knob * clamp(time_to_strike, 0, 1). The ring buffer stores the jittered
        values, so a delayed read reproduces the prediction noise AS OF push time (what the mocap
        link actually emitted then). The TRUE live target is untouched — rewards, metrics, the
        privileged critic, and the achieved-target-replay write keep reading racket_target_pos_w /
        racket_target_vel_w / swing_sign. time_to_strike is never delayed: the swing clock is
        generated robot-side by the deploy runner, not by the mocap link.
        """
        if not self._actor_view_active:
            return  # default path: delayed_* alias the live tensors — nothing to compute, no RNG
        pos = self.racket_target_pos_w
        vel = self.racket_target_vel_w
        if self._jitter_pos > 0.0 or self._jitter_vel > 0.0:
            scale = self.time_to_strike.clamp(0.0, 1.0).unsqueeze(-1)
            if self._jitter_pos > 0.0:
                pos = pos + torch.randn_like(pos) * (self._jitter_pos * scale)
            if self._jitter_vel > 0.0:
                vel = vel + torch.randn_like(vel) * (self._jitter_vel * scale)
        if self._delay_steps > 0:
            # Write this step's (jittered) target into slot `w`; the next slot in the length-
            # (delay+1) ring was written exactly `delay` pushes ago — that is the actor's view.
            w = self._delay_ptr
            self._delay_buf_pos[w].copy_(pos)
            self._delay_buf_vel[w].copy_(vel)
            self._delay_buf_sign[w].copy_(self.swing_sign)
            r = (w + 1) % (self._delay_steps + 1)
            self._delay_ptr = r
            self.delayed_racket_target_pos_w.copy_(self._delay_buf_pos[r])
            self.delayed_racket_target_vel_w.copy_(self._delay_buf_vel[r])
            self.delayed_swing_sign.copy_(self._delay_buf_sign[r])
        else:
            # Jitter-only (delay==0): the actor view is live + this step's noise, no latency.
            self.delayed_racket_target_pos_w.copy_(pos)
            self.delayed_racket_target_vel_w.copy_(vel)
            self.delayed_swing_sign.copy_(self.swing_sign)

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
            pre = self.pre_strike[env_ids]
            # POST-strike fall = terminated at/after the strike frame (tts <= 0, follow-through) OR
            # during the post-wrap hold (_recover_from_clip latch >= 0: the previous swing's recovery,
            # even though the wrap already flipped pre_strike=True for the NEXT swing). Both are the
            # "hit, then fall while recovering" failure that completion + pre-strike metrics miss.
            rec = self._recover_from_clip[env_ids]
            recovering = rec >= 0
            true_pre = term & pre & ~recovering
            post = term & (~pre | recovering)
            self._prestrike_fall_acc += float(true_pre.sum())
            self._poststrike_fall_acc += float(post.sum())
            if motion._multiseg:
                # Attribute the fall to the clip the env was ON when it fell: pre-strike falls to the
                # _prev_clip_id snapshot (motion already resampled clip_id for the new episode);
                # post-wrap-hold falls to the latched clip whose swing caused the recovery.
                fall_clips = torch.where(recovering, rec, self._prev_clip_id[env_ids])
                for c in self._clip_names:
                    csel = fall_clips == c
                    self._prestrike_fall_acc_c[c] += float((true_pre & csel).sum())
                    self._poststrike_fall_acc_c[c] += float((post & csel).sum())
            # True reset: the new episode starts fresh (its stand-start/reset hold is genuine
            # pre-strike preparation, not recovery), so clear the latch for these envs.
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self._recover_from_clip[env_ids_t] = -1

    def _update_footwork_signals(self, racket_dist: torch.Tensor) -> None:
        """Base-FREE footwork-to-strike signals (reward/metric only; NEVER observed). The legs are driven
        to move by racket PROGRESS (reducing the racket->target distance), not by any base target. All
        guards degrade to 0 if a body/sensor cannot resolve, so this can never crash training."""
        data = self.robot.data
        # --- racket-target distance + dense progress (the base-free movement driver) ---
        self.racket_target_distance = racket_dist
        # progress = previous - current distance. Resample/reset steps are not learnable progress:
        # the target and/or reference clip jumped, so reset the baseline and emit exactly zero.
        motion = self._motion()
        reset_progress = self._progress_reset_mask.clone()
        if hasattr(motion, "just_resampled"):
            reset_progress |= motion.just_resampled
        progress = (self._prev_racket_dist - racket_dist).clamp(-0.15, 0.15)
        self.racket_progress = torch.where(reset_progress, torch.zeros_like(progress), progress)
        self._prev_racket_dist = racket_dist.detach()
        self._progress_reset_mask.zero_()
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

    def _vb_evaluate(self, exact_strike: torch.Tensor, pos_err: torch.Tensor):
        """Tier-1 at-strike virtual-ball evaluation (rewardDesign.md).

        Runs once per control step from ``_update_metrics`` (rewards/obs read the same fresh
        buffers after ``command_manager.compute()``). Whenever ANY env sits at its exact-strike
        frame, the FULL batch goes through contact + coarse rollout — the cost is kernel-launch
        bound and batch-size independent, so gathering the ~30 striking envs saves nothing
        (verify_tier1 (b)); ``vb_fired`` masks consumption. On strike-free steps the one-shot
        mask is cleared and nothing is computed.
        """
        from whole_body_tracking.tasks.tracking.mdp import virtual_ball as _vb

        if not bool(exact_strike.any()):
            self.vb_fired.zero_()
            return
        if self._vb_params is None:
            self._vb_params = _vb.load_venue_params()
            print(
                f"[RacketTargetCommand] virtual ball ON: venue constants from "
                f"{self._vb_params.source_path} (k_d={self._vb_params.k_d}, "
                f"k_m={self._vb_params.k_m}, e(u_n)={self._vb_params.paddle_e_g1}"
                f"*exp({self._vb_params.paddle_e_g2}*u_n), a_t={self._vb_params.paddle_a_t})",
                flush=True,
            )
        prm = self._vb_params

        v_in, w_in = self.vb_vel_in_w, self.vb_spin_in_w
        v_r, n_face = self.racket_lin_vel_w, self.racket_normal_w
        # CAPTURE GATE: close enough at the strike frame AND paddle moving INTO the ball along the
        # oriented contact normal (a stationary/retreating wall-block scores nothing — verify (c)3).
        n_or = _vb.orient_normal(n_face, v_in, v_r)
        approach = torch.sum(v_r * n_or, dim=-1)
        gate = (
            exact_strike
            & (pos_err < float(self.cfg.vb_capture_radius))
            & (approach > float(self.cfg.vb_min_approach_speed))
        )

        # Achieved-state contact (venue paddle model, e(u_n)) + coarse landing rollout.
        # The rollout must start in the ENV-LOCAL frame: the virtual table landmarks
        # (vb_table_near_x / net / far end) and vb_target_xy are per-env offsets from the env
        # origin, while racket_pos_w is TRUE world frame (env grids span tens of meters at 4096
        # envs — using it raw put every landing ~|env_origin| away from the target; caught in the
        # first vb_warmE14k run: virtual_land_err_m ~62 m). Env grids are pure translations, so
        # velocities, normals, and spins need no correction.
        v_plus, w_plus = _vb.predict_paddle_contact(v_in, v_r, n_face, w_in, prm)
        land = _vb.coarse_landing(
            self.racket_pos_w - self._env.scene.env_origins,
            v_plus,
            w_plus,
            prm,
            surface_z=float(self.cfg.vb_table_surface_z),
            net_x=self._vb_net_x,
            h=float(self.cfg.vb_rollout_h),
            n_steps=int(self.cfg.vb_rollout_steps),
        )
        lx, ly = land["land_xy"][:, 0], land["land_xy"][:, 1]
        on_opp = (
            land["land_valid"] & (lx > self._vb_net_x) & (lx <= self._vb_far_x) & (ly.abs() <= self._vb_half_w)
        )
        depth_ok = lx > (self._vb_net_x + float(self.cfg.vb_min_landing_depth))
        net_clear = land["net_valid"] & (land["net_z"] > self._vb_net_top_z + self._vb_ball_r)
        # Outgoing topspin component about t_hat = z_hat x d_hat of the outgoing horizontal
        # direction (Ace-style): omega . t_hat = -w_x*d_y + w_y*d_x.
        d_xy = v_plus[:, :2]
        d_hat = d_xy / (torch.linalg.norm(d_xy, dim=-1, keepdim=True) + 1e-9)
        topspin = -w_plus[:, 0] * d_hat[:, 1] + w_plus[:, 1] * d_hat[:, 0]

        # One-shot caches consumed by hope_rewards.virtual_* THIS step.
        self.vb_fired = gate
        self.vb_landing_xy = land["land_xy"]
        self.vb_landing_valid = land["land_valid"]
        self.vb_on_opponent = on_opp
        self.vb_depth_ok = depth_ok
        self.vb_net_z = land["net_z"]
        self.vb_net_clear = net_clear
        self.vb_topspin = topspin
        self.vb_spin_out_norm = torch.linalg.norm(w_plus, dim=-1)

        # Sample-weighted EMA rates (hit rate over exact-strike samples; outcome rates over captured
        # hits). NOTE: accumulators only decay on strike-carrying steps — exact at 4096 envs (a
        # strike happens ~every step), slightly stale at small env counts (diagnostics only).
        decay = float(self.cfg.exact_success_decay)
        self._vb_exact_acc = decay * self._vb_exact_acc + float(exact_strike.sum())
        self._vb_hit_acc = decay * self._vb_hit_acc + float(gate.sum())
        self._vb_net_acc = decay * self._vb_net_acc + float((gate & net_clear).sum())
        self._vb_land_valid_acc = decay * self._vb_land_valid_acc + float((gate & land["land_valid"]).sum())
        self._vb_inb_acc = decay * self._vb_inb_acc + float((gate & net_clear & on_opp).sum())
        enough_e = self._vb_exact_acc >= float(self.cfg.exact_success_min_count)
        enough_h = self._vb_hit_acc >= 1.0
        self.metrics["virtual_hit_rate"][:] = (self._vb_hit_acc / max(self._vb_exact_acc, 1e-6)) if enough_e else 0.0
        self.metrics["virtual_net_clear_rate"][:] = (self._vb_net_acc / max(self._vb_hit_acc, 1e-6)) if enough_h else 0.0
        self.metrics["virtual_land_valid_rate"][:] = (
            (self._vb_land_valid_acc / max(self._vb_hit_acc, 1e-6)) if enough_h else 0.0
        )
        self.metrics["virtual_land_inbounds_rate"][:] = (
            (self._vb_inb_acc / max(self._vb_hit_acc, 1e-6)) if enough_h else 0.0
        )
        self.metrics["virtual_approach_speed"] = torch.where(
            exact_strike, approach, self.metrics["virtual_approach_speed"]
        )
        fired_valid = gate & land["land_valid"]
        if bool(fired_valid.any()):
            derr = torch.linalg.norm(land["land_xy"] - self._vb_target_xy.unsqueeze(0), dim=-1)
            self.metrics["virtual_land_err_m"][:] = derr[fired_valid].mean()
            self.metrics["virtual_topspin_revs"][:] = (topspin[fired_valid] / (2.0 * math.pi)).mean()

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
        # Tier-1 virtual-ball at-strike evaluation (one-shot buffers consumed by the virtual_*
        # reward terms after this compute()); no-op (and vb_fired stays False) when disabled.
        if self.cfg.virtual_ball:
            self._vb_evaluate(exact_strike, pos_err)
        # GLOBAL error-magnitude EMAs (P2.3 adaptive sigma driver) — same decay/mask as the pass
        # counters above; per-clip variants exist further down but sigma needs one global signal.
        self._exact_pos_err_sum = decay * self._exact_pos_err_sum + float((pos_err * exact_strike).sum())
        self._exact_vel_err_sum = decay * self._exact_vel_err_sum + float((vel_err * exact_strike).sum())
        # UNCONDITIONAL swing accounting: decay the start/fall accumulators at the SAME per-step
        # rate as the exact accumulators (increments happen in _count_swing_starts), then report
        #   swing_completion_rate = exact-strike arrivals / swing starts   (falls count against it)
        #   pre_strike_fall_rate  = pre-strike terminations / swing starts
        # These are the honest companions to the CONDITIONAL composite below, whose denominator
        # only contains exact-strike samples (pre-strike falls are invisible to it).
        self._swing_starts_acc = decay * self._swing_starts_acc
        self._prestrike_fall_acc = decay * self._prestrike_fall_acc
        self._poststrike_fall_acc = decay * self._poststrike_fall_acc
        self._resample_n_acc = decay * self._resample_n_acc
        self._replay_n_acc = decay * self._replay_n_acc
        for _c in self._clip_names:
            self._swing_starts_acc_c[_c] = decay * self._swing_starts_acc_c[_c]
            self._prestrike_fall_acc_c[_c] = decay * self._prestrike_fall_acc_c[_c]
            self._poststrike_fall_acc_c[_c] = decay * self._poststrike_fall_acc_c[_c]
        _s_denom = max(self._swing_starts_acc, 1e-6)
        _s_enough = self._swing_starts_acc >= float(self.cfg.exact_success_min_count)
        self.metrics["swing_completion_rate"][:] = min(self._exact_n_acc / _s_denom, 1.0) if _s_enough else 0.0
        self.metrics["pre_strike_fall_rate"][:] = min(self._prestrike_fall_acc / _s_denom, 1.0) if _s_enough else 0.0
        self.metrics["post_strike_fall_rate"][:] = (
            min(self._poststrike_fall_acc / _s_denom, 1.0) if _s_enough else 0.0
        )
        # HER replay diagnostics: fraction of resampled targets drawn from the achieved buffer
        # (~achieved_target_mix_prob once the per-clip buffers pass achieved_min_fill).
        self.metrics["achieved_replay_frac"][:] = (
            self._replay_n_acc / max(self._resample_n_acc, 1e-6)
            if self._resample_n_acc >= float(self.cfg.exact_success_min_count)
            else 0.0
        )
        for _c, _cn in self._clip_names.items():
            _cd = max(self._swing_starts_acc_c[_c], 1e-6)
            _ce = self._swing_starts_acc_c[_c] >= float(self.cfg.exact_success_min_count)
            self.metrics[f"swing_completion_rate_{_cn}"][:] = (
                min(self._exact_n_acc_c[_c] / _cd, 1.0) if _ce else 0.0
            )
            # Fall attribution uses _prev_clip_id (the clip during the fall) while starts use the NEW
            # clip; with uniform clip resampling the denominators match in expectation.
            self.metrics[f"pre_strike_fall_rate_{_cn}"][:] = (
                min(self._prestrike_fall_acc_c[_c] / _cd, 1.0) if _ce else 0.0
            )
            self.metrics[f"post_strike_fall_rate_{_cn}"][:] = (
                min(self._poststrike_fall_acc_c[_c] / _cd, 1.0) if _ce else 0.0
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
            # --- HER achieved-target buffer WRITE ---------------------------------------------------
            # Record the racket state the policy ACTUALLY produced at this step's exact-strike frames
            # (pos env-origin-relative, vel world). Alive envs only by construction: terminated envs
            # were reset before the command computes, so their state never lands here. Gated on the mix
            # prob so the buffers cost nothing when replay is off.
            if self.cfg.achieved_target_mix_prob > 0.0:
                for _c in self._clip_names:
                    _bidx = torch.where(exact_strike & (_clip == _c))[0]
                    _m = int(_bidx.numel())
                    if _m == 0:
                        continue
                    _size = self._ach_pos[_c].shape[0]
                    _rows = (self._ach_ptr[_c] + torch.arange(_m, device=self.device)) % _size
                    self._ach_pos[_c][_rows] = self.racket_pos_w[_bidx] - origins[_bidx]
                    self._ach_vel[_c][_rows] = self.racket_lin_vel_w[_bidx]
                    self._ach_ptr[_c] = int((self._ach_ptr[_c] + _m) % _size)
                    self._ach_fill[_c] = min(self._ach_fill[_c] + _m, _size)
            for _c, _cn in self._clip_names.items():
                self.metrics[f"achieved_buffer_fill_{_cn}"][:] = float(self._ach_fill[_c])
        # Per-axis position error AT the exact strike frame (which axis is the miss?). The position-only
        # strike_success_exact was dropped — strike_pos_pass_exact above is the same signal, undiluted.
        _axis_err_exact = torch.abs(self.racket_pos_w - self.racket_target_pos_w)
        for _ai, _ax in enumerate(("x", "y", "z")):
            self.metrics[f"racket_pos_error_{_ax}_exact_strike"] = torch.where(
                exact_strike, _axis_err_exact[:, _ai], self.metrics[f"racket_pos_error_{_ax}_exact_strike"]
            )
        # P2.3 SMASH-style ADAPTIVE TRACKING SIGMA (coarse-to-fine): every sigma_update_every steps,
        # set the racket position/velocity reward stds to the clamped decayed MEAN exact-strike error,
        # so the kernel always brackets the current operating band instead of a hand-tuned constant
        # (SMASH Table IV: removing this collapses success 86.4 -> 22.6). Mutates the LIVE reward-term
        # params in place (read per compute() call); also keeps racket_strike_success's own std_pos/
        # std_vel in lockstep so the multiplicative bonus agrees with the additive terms. Same
        # placement/pattern as the success-gated perturb curriculum below.
        if (
            self.cfg.adaptive_sigma
            and enough
            and self._env.common_step_counter % int(self.cfg.sigma_update_every) == 0
        ):
            pos_mean = self._exact_pos_err_sum / denom
            vel_mean = self._exact_vel_err_sum / denom
            sigma_pos = min(max(float(self.cfg.sigma_ema_scale) * pos_mean, float(self.cfg.sigma_pos_min)),
                            float(self.cfg.sigma_pos_max))
            sigma_vel = min(max(float(self.cfg.sigma_ema_scale) * vel_mean, float(self.cfg.sigma_vel_min)),
                            float(self.cfg.sigma_vel_max))
            rm = self._env.reward_manager
            try:
                rm.get_term_cfg("racket_position").params["std"] = sigma_pos
                rm.get_term_cfg("racket_velocity").params["std"] = sigma_vel
                succ = rm.get_term_cfg("racket_strike_success").params
                succ["std_pos"] = sigma_pos
                succ["std_vel"] = sigma_vel
            except ValueError:
                pass  # a variant task without these terms: adaptive sigma is a no-op there
            self._adaptive_sigma_pos = sigma_pos
            self._adaptive_sigma_vel = sigma_vel
        if self.cfg.adaptive_sigma:
            self.metrics["adaptive_sigma_pos"][:] = self._adaptive_sigma_pos
            self.metrics["adaptive_sigma_vel"][:] = self._adaptive_sigma_vel
        # A1 target latency diagnostic: constant broadcast, refreshed every step because
        # CommandTerm.reset() zeros metric entries of resetting envs before logging them.
        # (midswing_resample_count is written per step in _update_command while the feature is on.)
        self.metrics["target_delay_steps_in_effect"][:] = float(self._delay_steps)

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

        PRIVILEGED: uses ``base_pos_w`` (world base position). Mocap streams the base pose at 300 Hz
        during play, but that link is not bridged into the deploy front-end, so this term is fabricated
        at deploy; base-position-freedom is a deliberate robustness choice. Used by the `full` obs mode;
        the deploy-parity mode (legacy task name: `real_sensor_only`) replaces it with
        :meth:`racket_target_pos_b_rel`.
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

        A1: reads the ACTOR-visible target view (delayed/jittered when the A1 knobs are on;
        the live tensor itself otherwise — byte-identical default). This method backs the
        deploy-parity ACTOR obs only; the critic's :meth:`racket_target_pos_b` stays live.
        """
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.actor_racket_target_pos_w() - self.racket_pos_w)

    # --- A1 ACTOR-visible target accessors (delayed/jittered view; live aliases when off) ------- #
    def actor_racket_target_pos_w(self) -> torch.Tensor:
        """ACTOR-visible desired racket position (world): the A1 delayed/jittered view when target
        latency/jitter is enabled, else the live tensor itself (zero-overhead alias). Rewards,
        metrics, and the privileged critic keep reading the TRUE live ``racket_target_pos_w``."""
        return self.delayed_racket_target_pos_w

    def actor_racket_target_vel_w(self) -> torch.Tensor:
        """ACTOR-visible desired racket velocity (world). See :meth:`actor_racket_target_pos_w`."""
        return self.delayed_racket_target_vel_w

    def actor_swing_sign(self) -> torch.Tensor:
        """ACTOR-visible swing sign (forehand +1 / backhand -1), delayed with the target when A1
        latency is on (the swing-type flag rides the same planner->runner message as the target)."""
        return self.delayed_swing_sign

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

    # --- P2.3 SMASH-style adaptive tracking sigma (coarse-to-fine reward kernel widths) ---
    # When on, every `sigma_update_every` control steps the racket_position/racket_velocity reward
    # stds (and racket_strike_success's std_pos/std_vel) are set to
    #   clamp(sigma_ema_scale * decayed_mean_exact_strike_error, sigma_min, sigma_max)
    # so the exp kernel always brackets the CURRENT operating error band. Replaces the hand-run
    # 1.8 -> 1.0 -> 0.8 -> 0.5 velocity-std curriculum. sigma_*_max should match the task YAML's
    # starting stds; sigma_*_min should sit at the acceptance thresholds (0.075 m / 0.5 m/s).
    adaptive_sigma: bool = False
    sigma_update_every: int = 500
    sigma_ema_scale: float = 1.0
    sigma_pos_min: float = 0.075
    sigma_pos_max: float = 0.20
    sigma_vel_min: float = 0.5
    sigma_vel_max: float = 1.0

    # --- A1 target latency & time-variance (mocap->planner->runner realism; roadmap A1) -------------
    # MOTIVATION: training otherwise hands the actor a PERFECT, instantly-updated target, while the
    # real loop (mocap -> planner -> runner) delivers it LATE (transport + planning latency), NOISY
    # (ball-prediction error that shrinks as the strike approaches — SMASH Eq. 14), and REFINED
    # mid-swing (the planner re-plans WHERE, not WHEN). PACE injects sensor delays for the same
    # reason. Without this, the mocap-closed-loop deployment faces out-of-distribution target
    # dynamics. Scope: ONLY the ACTOR-visible target view (pos/vel/swing_sign) is degraded; rewards,
    # metrics, the privileged critic, and the achieved-target-replay write use the TRUE live target.
    # time_to_strike is NEVER delayed: the swing clock is generated robot-side by the deploy runner,
    # not by the mocap link. ALL defaults OFF => byte-identical baseline (delay==0 aliases the live
    # tensors; jitter==0 / prob==0 short-circuit before any RNG draw).
    target_delay_steps: int = 0  # actor sees target pos/vel/swing_sign this many control steps (50 Hz) late
    # SMASH-style tts-decaying gaussian noise on the ACTOR-visible target, drawn ONCE per step on the
    # ring-buffer push (determinism within a step): per-step std = knob * clamp(time_to_strike, 0, 1),
    # i.e. the knob is the std at time_to_strike >= 1 s, decaying to 0 at the strike (prediction
    # convergence). Units: m (pos) / m/s (vel).
    target_jitter_pos_per_s: float = 0.0
    target_jitter_vel_per_s: float = 0.0
    # Mid-swing target refinement: each control step, envs with pre_strike AND time_to_strike >
    # midswing_resample_tts_floor re-draw their target (position/velocity/normal via the existing
    # sampling path) with this per-step probability. Strike timing is untouched (same strike step),
    # no swing start is counted, and the racket-progress baseline is reset so the target jump creates
    # no fake progress.
    midswing_resample_prob: float = 0.0
    midswing_resample_tts_floor: float = 0.3  # s; no refinement inside the last `floor` seconds before the strike

    # --- Tier-1 VIRTUAL INCOMING BALL + at-strike landing evaluation (rewardDesign.md) -----------
    # Per swing, a virtual incoming ball (v_in, omega_in) is sampled that BY CONSTRUCTION arrives at
    # the racket target point at the strike time. On the exact-strike frame, the achieved racket FK
    # state is pushed through the venue-fitted paddle contact model (virtual_ball.predict_paddle_
    # contact, e(u_n) restitution) and a coarse RK4 landing rollout; the cached outcome buffers feed
    # the one-shot virtual_* reward terms in hope_rewards.py. No obs change; 175-D contract untouched.
    virtual_ball: bool = False
    # Incoming-ball velocity box (world/env frame, m/s; -x = toward the robot). Kept inside the venue
    # fit's validity envelope (ball speed 1-7 m/s); vertical component ~near-apex-to-descending.
    vb_vel_x_range: tuple[float, float] = (-4.5, -2.0)
    vb_vel_y_range: tuple[float, float] = (-0.6, 0.6)
    vb_vel_z_range: tuple[float, float] = (-1.0, 0.5)
    # Incoming spin: per-axis uniform (rad/s). 50 rad/s ~ 8 rev/s per axis keeps |omega| inside the
    # quaternion-validated 0-15 rev/s envelope.
    vb_spin_abs_max: float = 50.0
    # virtual_spin reward semantics: "topspin" = Ace-style outgoing-topspin generation (ball
    # quality); "minimize" = stage-1 placement-first mode (franco 2026-07-04) — reward CANCELING
    # the incoming spin, kernel exp(-|omega_out|^2 / vb_spin_min_sigma^2) on the outgoing spin
    # magnitude, same legal-landing gate. Sigma in rad/s (10 ~ 1.6 rev/s residual).
    vb_spin_mode: str = "topspin"
    vb_spin_min_sigma: float = 10.0
    # CAPTURE GATE: the virtual contact only evaluates when (a) the racket center is within this
    # distance of the ball (= racket 0.075 + ball 0.020, the v0 real-hit radius) at the exact-strike
    # frame, and (b) the paddle is actively moving INTO the ball along the oriented contact normal
    # faster than vb_min_approach_speed (kills the phantom-block / retreating-racket exploit,
    # verify_tier1 (c)3 — a stationary wall-block scores nothing).
    vb_capture_radius: float = 0.095
    vb_min_approach_speed: float = 0.3
    # Virtual table placement in the env frame. The _hopex clips are HOPE +X aligned with the root at
    # the env origin, so the HOPE convention (robot ~0.5 m behind its table end, centered on the
    # width) puts the near table edge at x = +0.5 and the surface at z = +0.76 above the env origin.
    # Net/far-end/half-width follow from the ITTF table (geometry.py): net at near_x + 1.37 etc.
    vb_table_near_x: float = 0.5
    vb_table_surface_z: float = 0.76
    # Landing target on the opponent half (env frame). Default = P2 half center (near_x + 2.055, 0).
    vb_target_x: float = 2.555
    vb_target_y: float = 0.0
    # Reward shaping constants (read by hope_rewards.virtual_*).
    vb_landing_sigma: float = 0.3     # m — Gaussian width on ||landing_xy - target_xy|| (v0 parity)
    vb_net_margin: float = 0.12      # m — target clearance above the net top (v0 pass_net parity)
    vb_net_sigma: float = 0.10       # m — Gaussian width on the net-clearance error
    vb_spin_ref: float = 250.0       # rad/s (~40 rev/s) — full-credit outgoing topspin (Ace-style)
    vb_min_landing_depth: float = 0.3  # m past the net for the in-bounds bonus (dink guard, verify (c)1)
    # Coarse rollout resolution (verify_tier1 (b): h=10 ms, 1.0 s horizon covers 1-7 m/s shots).
    vb_rollout_h: float = 0.01
    vb_rollout_steps: int = 100

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
    # added to the env origin. NOTE the y range is SIGNED here and used directly, so it REPLACES the
    # shared |y|-sign logic. Reason: each clip's strike frame can sit at a different height/depth/lateral
    # offset, so a shared box can make one clip's strike-frame position unreachable. Per-clip boxes let
    # each clip's target track its own reference strike point.
    racket_pos_range_per_clip: tuple | None = None

    # --- HER-style achieved-target replay (uniform mode + unified multi-clip only) -------------------
    # On-policy-compatible hindsight relabeling (Ace/HER, adapted for PPO): true retroactive relabeling is
    # observation-inconsistent here (the target is in the actor obs every step), so instead the NEXT
    # swing's target is drawn, with probability `achieved_target_mix_prob`, from a per-clip ring buffer of
    # racket states the policy ACTUALLY produced at previous exact-strike frames (pos env-origin-relative,
    # vel world). Every replayed target is reachable-by-demonstration — it kills the "the box asks for a
    # point the taught swing never passes through" mismatch without moving the box. Mixture (not pure
    # replay) + jitter + clamping into the per-clip box inflated by `achieved_clamp_inflate` (clamp
    # applies only when per-clip boxes are configured — the unified task always sets them) prevent the
    # target distribution from collapsing onto what the policy already does or drifting far from the
    # training workspace. NOTE the deploy-side target clips must be re-synced to the training boxes
    # whenever the boxes change (they are hand-maintained in pp_policy.hpp / imitate_presets.py).
    # 0.0 = OFF (backward compatible: pure box sampling). TRAIN-ONLY: eval entry points force this to
    # 0.0 so checkpoints are always scored on the pure box distribution. The buffer only fills at
    # exact-strike frames of envs that are still alive, so fallen approaches never contribute targets.
    achieved_target_mix_prob: float = 0.0
    achieved_buffer_size: int = 4096  # per-clip ring buffer capacity (entries)
    achieved_min_fill: int = 256  # replay only once a clip's buffer holds at least this many entries
    achieved_jitter_pos: float = 0.03  # m, uniform per-axis jitter added to a replayed position
    achieved_jitter_vel: float = 0.15  # m/s, uniform per-axis jitter added to a replayed velocity
    achieved_clamp_inflate: float = 0.20  # clamp replayed targets into the per-clip box inflated by this fraction

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
