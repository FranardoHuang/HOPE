"""HOPE goal-tracking reward terms (HITTER r_goal).

These implement the racket/base target tracking rewards on top of the BeyondMimic imitation
reward (``r_imitation``, the ``motion_*`` terms already in ``rewards.py``) and the regularization
reward (``r_regularization``, ``action_rate_l2`` / ``joint_torques_l2`` / contact penalties).

Activation timing follows HITTER: the base-position reward is active **before** the strike; the
racket position/velocity/normal rewards are active only in a **short window around** the strike.
Because a ``RewardTermCfg`` weight is constant, the time gating is applied *inside* each term by
multiplying the exponential kernel by the command's ``pre_strike`` / ``strike_window`` mask.

The exponential kernel form (``exp(-error/std**2)``) mirrors the BeyondMimic motion-tracking
rewards. HITTER does not publish reward weights or kernel forms, so the weights in the env config
are HOPE choices to be tuned, not paper-sourced values.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


def _dbg_log(cmd: RacketTargetCommand, name: str, raw: torch.Tensor, mask: torch.Tensor) -> None:
    """Log raw (pre-mask) and gated (post-mask) kernel values, held over the active mask.

    No-op unless ``cmd.cfg.debug_reward_logging`` is set. The held value lets the reset-mean report the
    in-window reward, and lets you see how much reward the time-gate is killing (gated vs raw) and whether
    the raw kernel still has any gradient at the current error scale (raw ~0 => std too tight).
    """
    if not cmd.cfg.debug_reward_logging:
        return
    cmd.metrics[f"dbg_{name}_raw"] = torch.where(mask, raw, cmd.metrics[f"dbg_{name}_raw"])
    cmd.metrics[f"dbg_{name}_gated"] = torch.where(mask, raw * mask.float(), cmd.metrics[f"dbg_{name}_gated"])


def racket_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket center position near strike using the target's swing-through trajectory."""
    cmd = _cmd(env, command_name)
    target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_pos_now), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "racket_pos", raw, cmd.strike_window)
    return raw * cmd.strike_window.float()


def racket_position_tracking_static_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Ablation B: track the strike POINT itself (no swing-through), decoupling position from timing/velocity.

    Identical gating to ``racket_position_tracking_exp`` but compares against the bare ``racket_target_pos_w``
    instead of the moving swing-through point ``target - vel*t_to_strike``. Over a ±0.15 s window the
    swing-through point sweeps up to ~0.9 m at a 6 m/s target, so the standard term mostly rewards being on
    the moving line (timing/velocity); this variant gives a clean "get the paddle to the point" signal for
    early stable positioning. Select via ``rewards.racket_position_static: true`` in the task YAML.
    """
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_pos_w - cmd.racket_target_pos_w), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "racket_pos", raw, cmd.strike_window)
    return raw * cmd.strike_window.float()


def racket_velocity_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket linear velocity near the strike time (FK actual vs desired, world frame)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "racket_vel", raw, cmd.strike_window)
    return raw * cmd.strike_window.float()


def racket_normal_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket face-normal orientation near the strike time. ``std`` is in radians."""
    cmd = _cmd(env, command_name)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    raw = torch.exp(-(angle**2) / std**2)
    _dbg_log(cmd, "racket_normal", raw, cmd.strike_window)
    return raw * cmd.strike_window.float()


def base_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track desired base XY position before the strike (encourages repositioning footwork)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "base", raw, cmd.pre_strike)
    return raw * cmd.pre_strike.float()


def pre_strike_foot_slip(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize horizontal foot speed WHILE the foot is in contact, BEFORE the strike only.

    The robot was sliding/leaning to reach far racket targets while the base reward pinned it near spawn
    (foot_slip_speed high, foot_contact_frac low). This term teaches it to plant its feet and stabilize
    during the approach. It is gated by ``pre_strike`` ONLY (not ``strike_window``), so the strike swing's
    footwork is untouched. ``foot_slip_in_contact`` (sum over feet of horizontal speed * in_contact) is
    precomputed by the RacketTargetCommand each step (0 if the feet/contact sensor cannot be resolved).
    Returns a positive magnitude; the RewTerm weight is negative.
    """
    cmd = _cmd(env, command_name)
    return cmd.foot_slip_in_contact * cmd.pre_strike.float()


# ============================================================================================== #
# Footwork-to-strike (BASE-FREE). The legs move because moving the body REDUCES the racket->target
# distance (racket_progress), not because they track a base target. Footwork is penalized for being
# BAD (slip / drag / violent / unstable at strike), NOT for stepping — the feet are free to move.
# ============================================================================================== #
def racket_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """DENSE pre-strike reward for reducing the racket->target distance (prev - current, clamped). This
    is the base-free driver of whole-body footwork: the legs/waist/arms all get credit for moving the
    racket closer to the target, with NO base-position target. Gated to pre_strike (approach phase); the
    strike swing itself is scored by the racket pos/vel/normal terms. Positive when approaching; RewTerm
    weight is POSITIVE."""
    cmd = _cmd(env, command_name)
    return cmd.racket_progress * cmd.pre_strike.float()


def hold_ready(
    env: ManagerBasedRLEnv, command_name: str, std: float, reach: float = 0.65, reach_mode: str = "racket"
) -> torch.Tensor:
    """POSITIVE ready-stance reward during the pre-swing HOLD (the between-swing recovery phase).

    HITTER's balance recovery comes from a positive "prepare for the next target" signal (its pre-strike
    base-position reward), not from balance penalties. In the base-free deploy-parity design the hold
    phase (reference frozen at the next swing's first frame) already pulls the UPPER body to the ready
    pose via imitation, but the legs/base get zero positive signal — only penalties. This term fills that
    gap without a base-position target (deploy-honest: everything here is proprioceptive in spirit —
    stillness + planted feet): ``exp(-(|v_base|^2 + |w_base|^2)/std^2) * feet_contact_frac``, gated to
    the motion command's ``in_hold`` mask. Rewards arriving at the next windup calm, upright-by-stillness
    and with both feet planted — i.e. finishing the previous swing in a recoverable state.

    ``reach`` gate: stillness is only the CORRECT ready action when the robot is already where it can
    strike from. Without the gate this term pays ~weight/step for planted stillness, which out-earns the
    telescoping racket_progress for stepping during the hold — i.e. it would teach freeze-then-rush
    exactly when wide target boxes need footwork. Two gate modes (``reach_mode``):

    * ``"racket"`` (legacy default, base-free tasks): ``racket_target_distance < reach`` — the 3D
      FK-blade->target distance. CAVEAT (2026-07-05 footwork audit): this gate is NOT
      station-selective — the blade distance is arm-pose-controllable (arm imitation is swing-only,
      so reaching toward the target during the hold is reward-free), and for near-side targets it is
      SMALLER at the wrong station than at the correct one, inverting the settle income exactly where
      a step is required. Keep it only for base-free tasks that have no meaningful station.
    * ``"station"`` (HITTER footwork tasks): ``|base_xy − base_target_xy| < reach`` — the planar
      base->commanded-station error. Station-selective by construction and not arm-gameable: far
      station -> the term is silent (base_position/racket_progress drive the step, untaxed);
      arrived -> the stillness income switches on (move to the stance, THEN settle, then swing).

    Zero outside the hold (the swing itself is untouched) and a safe no-op if the motion command has
    no hold state. RewTerm weight is POSITIVE.
    """
    cmd = _cmd(env, command_name)
    in_hold = getattr(cmd._motion(), "in_hold", None)
    if in_hold is None:
        return torch.zeros(cmd.num_envs, device=cmd.device)
    data = cmd.robot.data
    motion_sq = torch.sum(torch.square(data.root_lin_vel_w), dim=-1) + torch.sum(
        torch.square(data.root_ang_vel_w), dim=-1
    )
    raw = torch.exp(-motion_sq / std**2) * cmd.feet_contact_frac
    if reach_mode == "station":
        station_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w, dim=-1)
        near = (station_err < reach).float()
    elif reach_mode == "racket":
        near = (cmd.racket_target_distance < reach).float()
    else:
        raise ValueError(f"hold_ready: unknown reach_mode '{reach_mode}' (expected 'racket' or 'station')")
    return raw * near * in_hold.float()


def base_decel_tracking(
    env: ManagerBasedRLEnv, command_name: str, v_gain: float = 2.0, v_max: float = 1.6, std: float = 0.4
) -> torch.Tensor:
    """P2.4 PACE-style smooth-deceleration shaping: track a pseudo base-velocity command that decays
    with the remaining planar racket->target error (G08: the robot rushes far targets reactively, with
    no deceleration profile, and arrives too hot to strike).

    PACE's remedy is a velocity command proportional to the remaining position error, so the DESIRED
    speed goes to ~0 exactly at arrival. Deploy-parity constraint: the 175-D actor obs contract is
    FROZEN, so this is a REWARD-side term only — nothing new is observed; the kernel reuses the task's
    own error measure (the planar racket->target distance, frame-invariant, no world base position):

        v_des = clamp(v_gain * ||(racket_target_xy - racket_xy)||, 0, v_max)
        reward = exp(-(||v_base_xy|| - v_des)^2 / std^2)

    Far target -> v_des saturates at v_max and the term pays for MOVING (it cooperates with
    racket_progress instead of taxing the approach); as the strike stance is reached v_des -> 0 and the
    term pays for a CALM base — a smooth taper instead of the bang-bang rush-then-slam. Gated to
    ``pre_strike`` ONLY: the strike swing and the post-strike recovery are untouched (post-strike the
    distance to the OLD swung-through target would otherwise command a bogus speed-up). Base velocity
    is the WORLD planar root velocity (same source as hold_ready); v_gain [1/s] is the P-gain of the
    pseudo velocity command, v_max [m/s] its cap, std [m/s] the kernel width. RewTerm weight is
    POSITIVE; default weight 0.0 = OFF (flag-gated via task.rewards.base_decel_weight)."""
    cmd = _cmd(env, command_name)
    planar_err = torch.norm(cmd.racket_target_pos_w[:, :2] - cmd.racket_pos_w[:, :2], dim=-1)
    v_des = (v_gain * planar_err).clamp(0.0, v_max)
    v_base = torch.norm(cmd.robot.data.root_lin_vel_w[:, :2], dim=-1)
    raw = torch.exp(-torch.square(v_base - v_des) / std**2)
    return raw * cmd.pre_strike.float()


def racket_strike_success(
    env: ManagerBasedRLEnv, command_name: str, std_pos: float, std_vel: float, std_normal: float
) -> torch.Tensor:
    """MULTIPLICATIVE strike success R_pos * R_vel * R_normal, gated to the strike window. Unlike the
    additive racket terms (which give partial credit for getting only position OR velocity right), the
    product is high ONLY when position AND velocity AND normal are all good at once — a true hit. RewTerm
    weight is POSITIVE."""
    # Each kernel already multiplies by strike_window internally, so the product is non-zero ONLY in the
    # window (no extra gate needed). The product is high only when pos AND vel AND normal are all good.
    rp = racket_position_tracking_exp(env, command_name, std_pos)
    rv = racket_velocity_tracking_exp(env, command_name, std_vel)
    rn = racket_normal_tracking_exp(env, command_name, std_normal)
    return rp * rv * rn


# ============================================================================================== #
# Tier-1 VIRTUAL-BALL outcome terms (rewardDesign.md). One-shot: non-zero ONLY on the exact-strike
# step of envs that passed the capture gate (cmd.vb_fired, set by RacketTargetCommand._vb_evaluate
# from the venue-fitted contact + coarse landing rollout). All are inert (all-zero) unless
# commands.racket_target.virtual_ball is enabled. Anti-farming gates follow the adversarial
# verification (verify_tier1-reward-soundness.md (c)):
#   1. the in-bounds bonus requires landing depth > net_x + vb_min_landing_depth (dink guard),
#   2. the capture gate requires a minimum paddle approach speed (phantom-block guard, in _vb_evaluate),
#   3. the pass_net CLEAR BONUS pays only for shots that also land legally (net-without-landing
#      guard); its height KERNEL is deliberately ungated shaping — see virtual_pass_net docstring.
# ============================================================================================== #
def virtual_pass_net(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Net-height shaping at the virtual net-plane crossing + fully-gated clear bonus.

    The Gaussian kernel on (net-crossing height - (net_top + margin)) pays for ANY shot that
    reaches the net plane inside the rollout horizon (v0 ``pass_net_margin`` semantics): it is the
    CLIMB gradient that teaches a flat-hitting policy to angle shots upward. Gating it on a legal
    landing (this term's original verify (c)4 reading) starved training completely — the E-champion
    warm-start crosses the net legally on only ~0.2% of strikes, so 2.5k iterations of vb_warmE14k3
    paid exactly zero virtual reward (2026-07-03 incident). The farming surface is bounded: the
    kernel requires an actual net-plane crossing, maxes only at the correct height, and is worth at
    most 1/swing; anti-farming gates stay in full on the +0.5 clear bonus here and on the
    landing/spin terms. RewTerm weight POSITIVE.
    """
    cmd = _cmd(env, command_name)
    target_z = cmd._vb_net_top_z + float(cmd.cfg.vb_net_margin)
    err = cmd.vb_net_z - target_z
    kernel = torch.exp(-(err**2) / float(cmd.cfg.vb_net_sigma) ** 2)
    legal = cmd.vb_net_clear & cmd.vb_landing_valid & cmd.vb_on_opponent
    raw = kernel * cmd.vb_net_crossed.float() + 0.5 * legal.float()
    return raw * cmd.vb_fired.float()


def virtual_landing(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Landing-accuracy kernel + fully-gated in-bounds bonus (v0 ``landing_in_opponent_half``).

    CLIMB-PHASE shape (2026-07-04): the Gaussian kernel on ||landing_xy - target_xy|| pays for any
    landing inside the rollout horizon — NOT gated on net clearance. The E-warm-started policy
    lands ~1.9 m short of the target and reaches the net plane on only a few % of strikes, so both
    net-gated terms stayed ~zero for 5k+ iterations (vb_warmE14k3/4); this kernel is the dense
    bottom rung that pays for hitting DEEPER. Net-farming risk is bounded: the rollout has no net
    collider, so the kernel is smooth through the net plane with its single max AT the target —
    drilling the net base (err ~0.75 m) always pays less than clearing and landing deeper. The
    +1.0 bonus keeps the full gate: net clearance AND on-opponent AND depth past
    net_x + vb_min_landing_depth (verify (c)1 dink guard). Re-tighten (restore the net_clear gate
    on the kernel, sigma back toward 0.3) once virtual_net_clear_rate is healthy. RewTerm weight
    POSITIVE.
    """
    cmd = _cmd(env, command_name)
    dist2 = torch.sum(torch.square(cmd.vb_landing_xy - cmd._vb_target_xy.unsqueeze(0)), dim=-1)
    kernel = torch.exp(-dist2 / float(cmd.cfg.vb_landing_sigma) ** 2)
    bonus = (cmd.vb_landing_valid & cmd.vb_net_clear & cmd.vb_on_opponent & cmd.vb_depth_ok).float()
    raw = kernel * cmd.vb_landing_valid.float() + bonus
    return raw * cmd.vb_fired.float()


def virtual_spin(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Outgoing-topspin reward (Ace's ws-term), only for shots that land legally.

    ``clamp(topspin / vb_spin_ref, 0, 1)`` where topspin is omega_plus projected on z_hat x d_hat
    of the outgoing direction; gated on a valid net-clearing in-bounds landing so brushing wild
    swipes that miss the table cannot farm spin. RewTerm weight POSITIVE (ramp toward parity with
    landing per the Ace precedent once the wiring is validated).
    """
    cmd = _cmd(env, command_name)
    legal = cmd.vb_landing_valid & cmd.vb_net_clear & cmd.vb_on_opponent
    if getattr(cmd.cfg, "vb_spin_mode", "topspin") == "minimize":
        # Stage-1 placement-first semantics (franco 2026-07-04): the BEST shot kills the incoming
        # spin — reward small outgoing |omega|, not topspin generation (which is ball quality and
        # deliberately unrewarded in stage 1).
        kernel = torch.exp(-(cmd.vb_spin_out_norm / float(cmd.cfg.vb_spin_min_sigma)) ** 2)
        raw = kernel * legal.float()
    else:
        raw = (cmd.vb_topspin / float(cmd.cfg.vb_spin_ref)).clamp(0.0, 1.0) * legal.float()
    return raw * cmd.vb_fired.float()


# --- footwork penalties (feet may STEP; we only punish BAD foot behaviour) --------------------- #
def foot_slip_sq(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot slip while in contact: sum over feet of contact * ||foot_xy_velocity||² (always on).
    A planted/landing foot should not skate. Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_slip_sq


def foot_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize excessive/violent foot velocity: sum over feet of ||foot_velocity||². Lets the foot step
    but discourages flailing. Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_vel_sq


def foot_drag(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot dragging: lateral foot speed while the foot is near the ground (skimming instead of
    lifting cleanly to step). Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_drag


def arm_overreach(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Anti-arm-only: penalize solving the target by maxing the arm out — fraction of ARM joints within
    10% of a position limit. Encourages using the body/legs to bring the target into a comfortable arm
    range instead of stretching. Positive in [0,1]; RewTerm weight is negative."""
    return _cmd(env, command_name).arm_overreach_frac


def prestrike_waist_twist(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Anti twist-instead-of-step: penalize |waist_yaw|+|waist_roll| deviation from neutral BEFORE the
    strike. Widening the racket-target box alone did NOT force footwork — the policy just rotated its
    torso (waist yaw/roll) to face a lateral target while its feet stayed planted (arm_overreach stayed
    ~0, legs frozen). This term makes that twist costly during the approach, so getting behind a far
    target requires STEPPING. Gated by ``pre_strike`` ONLY (the strike swing's rotation is untouched) and
    ``waist_pitch`` is excluded (that is the swing wind-up / lean, not a lateral-reach cheat). Returns a
    positive magnitude (radians); the RewTerm weight is negative."""
    cmd = _cmd(env, command_name)
    return cmd.waist_twist * cmd.pre_strike.float()


# --- strike-window stability (penalize wobble/bob/skate AT the hit; gated to the strike window) - #
def strike_proj_grav_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize base tilt (||projected_gravity_xy||) DURING the strike window — be upright at the hit."""
    cmd = _cmd(env, command_name)
    return cmd.proj_grav_xy * cmd.strike_window.float()


def strike_base_ang_vel(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize base roll/pitch rate (||base_ang_vel_xy||) DURING the strike window."""
    cmd = _cmd(env, command_name)
    return cmd.base_ang_vel_xy_norm * cmd.strike_window.float()


def prestrike_proj_grav_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Sim2real balance shaping (CHANGE 3): penalize base/torso forward TILT (||projected_gravity_xy||, a
    POSITION quantity) DURING the approach (pre_strike). Together with the existing strike-window
    ``strike_upright`` this keeps the CoM over the support base THROUGH the whole swing — the forward
    pitch-over is exactly the AGI-MuJoCo failure mode. Deliberately NOT an angular-velocity penalty: a
    base-ang-vel penalty is anti-correlated with swing power and is gameable; projected-gravity tilt is a
    pose, so it does not fight the swing. Gated by pre_strike ONLY (the strike window is covered by
    strike_upright). Positive magnitude; the RewTerm weight is NEGATIVE."""
    cmd = _cmd(env, command_name)
    return cmd.proj_grav_xy * cmd.pre_strike.float()


def strike_foot_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot motion (sum ||foot_velocity||²) DURING the strike window — plant for the hit."""
    cmd = _cmd(env, command_name)
    return cmd.foot_vel_sq * cmd.strike_window.float()


def strike_vertical_bob(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize vertical base velocity (|base_lin_vel_z|) DURING the strike window — no bob at the hit."""
    cmd = _cmd(env, command_name)
    return cmd.vertical_speed * cmd.strike_window.float()


# ============================================================================================== #
# Sim2real: torque-saturation penalty (CHANGE 2). Discourage the policy from demanding torque the
# EXPLICIT clipped-PD motor cannot deliver. Under IdealPDActuatorCfg the model computes the pre-clip
# effort (kp*(q_des-q)+kd*(-qd)) and clips it to ±effort_limit; the ratio |computed| / effort_limit >1
# is exactly the over-demand that lags on the real robot. Penalizing the mean over-limit fraction over
# the arm + waist joints teaches a swing that lives inside the torque envelope (the elbow was measured
# at ~6.7x its 24 Nm limit in the failing trace). Uses ``data.computed_torque`` (Isaac copies each
# actuator's PRE-clip computed_effort into it) and ``data.joint_effort_limits`` (the per-joint sim
# limit written from effort_limit_sim). Both degrade to a 0 reward if unavailable, so it can never crash.
# ============================================================================================== #
_TORQUE_SAT_JOINT_EXPR = [".*shoulder.*", ".*elbow.*", ".*wrist.*", "waist_.*_joint"]


def _torque_sat_joint_idx(env: ManagerBasedRLEnv, command_name: str):
    """Resolve+cache the arm+waist joint indices on the command term (once)."""
    cmd = _cmd(env, command_name)
    idx = getattr(cmd, "_torque_sat_joint_idx", None)
    if idx is None:
        try:
            idx = list(cmd.robot.find_joints(_TORQUE_SAT_JOINT_EXPR)[0])
        except Exception:
            idx = []
        cmd._torque_sat_joint_idx = idx  # cache (empty list means "unresolvable")
    return cmd, idx


def arm_torque_saturation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Mean over-limit fraction of the COMPUTED (pre-clip) effort over the arm + waist joints:
    ``mean_j relu(|computed_torque_j| / effort_limit_j - 1)``. 0 when every arm/waist joint is inside its
    torque envelope; grows as the swing demands un-deliverable torque (the explicit-PD saturation that
    tips the free base in AGI's MuJoCo). Positive magnitude; the RewTerm weight is NEGATIVE."""
    cmd, idx = _torque_sat_joint_idx(env, command_name)
    data = cmd.robot.data
    tau = getattr(data, "computed_torque", None)
    lim = getattr(data, "joint_effort_limits", None)
    if not idx or tau is None or lim is None:
        z = torch.zeros(cmd.num_envs, device=cmd.device)
        cmd.metrics["arm_torque_sat_frac"] = z
        return z
    tau_a = torch.abs(tau[:, idx])
    lim_a = lim[:, idx].clamp(min=1e-3)  # guard against a 0/inf limit
    over = (tau_a / lim_a - 1.0).clamp(min=0.0)  # relu(ratio - 1): the un-deliverable fraction
    frac = over.mean(dim=-1)
    cmd.metrics["arm_torque_sat_frac"] = frac  # watch-metric: should fall toward 0 during fine-tune
    return frac

def motion_body_pos_swing_only(env, command_name: str, std: float, body_names=None):
    """motion_relative_body_position_error_exp gated to ~in_hold (2026-07-05): during
    hold the joint reference is the default STAND (commands.joint_pos) while the frozen
    body refs still show clip frame 0's crouch — un-gated, the two imitation pulls
    fight and the policy settles into the splayed-feet crouch-stand. Swing-only."""
    from .rewards import motion_relative_body_position_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_relative_body_position_error_exp(env, command_name, std, body_names)
    return torch.where(cmd.in_hold, torch.zeros_like(r), r)


def motion_body_ori_swing_only(env, command_name: str, std: float, body_names=None):
    """See motion_body_pos_swing_only."""
    from .rewards import motion_relative_body_orientation_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_relative_body_orientation_error_exp(env, command_name, std, body_names)
    return torch.where(cmd.in_hold, torch.zeros_like(r), r)

def foot_orientation_discipline(env, command_name: str, asset_cfg):
    """L1 deviation of the foot-orientation joints (hip yaw/roll, ankle roll) from the
    REFERENCE joint positions — hold-aware via commands.joint_pos (default stand during
    hold, clip footwork during swings). 2026-07-05: with no joint-level imitation in
    the stack these DOF were reward-free, and the policy twisted the feet to
    -1.13/+0.90 rad during swings/side-switches vs a reference envelope of ±0.41
    (Gate 2.5 diag) — the 'weird foot placement' at strike/switch. Use a NEGATIVE
    weight (penalty); keep it small so it disciplines feet without taxing the lunge.
    """
    cmd = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    ref = cmd.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(q - ref), dim=1)

