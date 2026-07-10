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

import math
import torch
from typing import TYPE_CHECKING

from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand, face_tracking_pair

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


def _dbg_log(cmd: RacketTargetCommand, name: str, raw: torch.Tensor, mask: torch.Tensor) -> None:
    """Log the current pre-mask kernel and its actual post-mask value.

    No-op unless ``cmd.cfg.debug_reward_logging`` is set.  The old implementation updated both
    tensors only where ``mask`` was true, making ``dbg_*_gated`` identically equal to
    ``dbg_*_raw`` and unable to reveal how often the gate removed income.  These diagnostics now
    mirror the reward expression on every step: ``gated = raw * mask``.
    """
    if not cmd.cfg.debug_reward_logging:
        return
    cmd.metrics[f"dbg_{name}_raw"] = raw
    cmd.metrics[f"dbg_{name}_gated"] = raw * mask.float()


def _window_pos(cmd: RacketTargetCommand) -> torch.Tensor:
    """1c TIGHT window for the position channel (== strike_window unless racket.strike_window_pos_s)."""
    win = getattr(cmd, "strike_window_pos", None)
    return cmd.strike_window if win is None else win


def _window_wide(cmd: RacketTargetCommand) -> torch.Tensor:
    """1c WIDE window for the normal/velocity channels (== strike_window unless racket.strike_window_wide_s)."""
    win = getattr(cmd, "strike_window_wide", None)
    return cmd.strike_window if win is None else win


def _pos_gate(cmd: RacketTargetCommand, pos_gate_radius: float | None) -> torch.Tensor | float:
    """Proximity power-gate (reward_staged_design §② C2a): sigmoid((r_gate - pos_err)/0.05) with
    pos_err = ||racket_FK - target||. ~0 when the paddle cannot reach the target (no face/velocity
    money AND no face/velocity gradient noise while out of reach), ~1 once inside the gate; smooth
    so there is no bang-bang flicker at the gate edge. 人话:拍子够得着球才开始付拍面/拍速的钱。
    ``None`` (the default of every caller) returns 1.0 — byte-identical baseline."""
    if pos_gate_radius is None:
        return 1.0
    pos_err = torch.norm(cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1)
    return torch.sigmoid((float(pos_gate_radius) - pos_err) / 0.05)


def _pos_kernel_raw(cmd: RacketTargetCommand, std: float) -> torch.Tensor:
    """UNGATED swing-through position kernel (shared by racket_position / racket_strike_success)."""
    target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_pos_now), dim=-1)
    return torch.exp(-error / std**2)


def _vel_kernel_raw(cmd: RacketTargetCommand, std: float) -> torch.Tensor:
    """UNGATED velocity kernel (shared by racket_velocity / racket_strike_success)."""
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def _face_pair(cmd: RacketTargetCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """(measured, target) face normals for EVERY face-channel term — the single source of the
    face frame. Any new face reward/penalty MUST read through here, never pick buffers itself.

    face_command=True (question-bank demanded normals): the bank is a +Y-calibration-frame ("A"
    convention) product — gen_stage1_questions.py sign-aligns every demanded normal to the RAW +Y
    clip face and has no notion of the striking-face sign table. So the measured side must be the
    raw +Y axis (``racket_normal_raw_w``), NOT the striking-face-signed ``racket_normal_w``.
    Pairing the signed normal against the bank target was the M3c/M2f 单翻病 (2026-07-09 病因定案):
    on sign=-1 (backhand) clips the reward optimum sat ~180° from the physically correct face and
    both arms converged to a ~34° systematic face error. A-vs-A is bitwise identical to flipping
    both sides (dot(-a,-b) == dot(a,b)), so nothing is lost: the mount sign table keeps serving
    the metric / reference / diagnostic channels (B convention) untouched.

    face_command=False: unchanged clip-reference pairing (signed vs signed — both sides carry the
    same per-clip sign, so this path is flip-invariant and byte-identical to the baseline).
    """
    return face_tracking_pair(cmd)


def _normal_kernel_raw(cmd: RacketTargetCommand, std: float) -> torch.Tensor:
    """UNGATED face-normal kernel (shared by racket_normal / racket_strike_success).

    Stage-1 face command: the reference is the DEMANDED (inverse-solved, question-bank) normal
    instead of the clip-locked reference normal. face_command=False keeps the old tensor read —
    byte-identical baseline. racket_strike_success re-anchors through this helper automatically.
    The (measured, target) pair comes from ``_face_pair`` — see its docstring for the frame rules.
    """
    measured, target_normal = _face_pair(cmd)
    cos_ang = torch.sum(measured * target_normal, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    return torch.exp(-(angle**2) / std**2)


def racket_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket center position near strike using the target's swing-through trajectory.
    Gated by the TIGHT position window (1c): contact must be precise; == strike_window by default."""
    cmd = _cmd(env, command_name)
    raw = _pos_kernel_raw(cmd, std)
    win = _window_pos(cmd)
    _dbg_log(cmd, "racket_pos", raw, win)
    return raw * win.float()


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
    win = _window_pos(cmd)
    _dbg_log(cmd, "racket_pos", raw, win)
    return raw * win.float()


def racket_velocity_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, pos_gate_radius: float | None = None
) -> torch.Tensor:
    """Track racket linear velocity near the strike time (FK actual vs desired, world frame).
    Gated by the WIDE window (1c; == strike_window by default) and, when rewards.face_gate_by_pos
    is on, by the proximity power-gate (see ``_pos_gate``)."""
    cmd = _cmd(env, command_name)
    raw = _vel_kernel_raw(cmd, std)
    win = _window_wide(cmd)
    _dbg_log(cmd, "racket_vel", raw, win)
    return raw * win.float() * _pos_gate(cmd, pos_gate_radius)


def racket_normal_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, pos_gate_radius: float | None = None
) -> torch.Tensor:
    """Track racket face-normal orientation near the strike time. ``std`` is in radians.
    Gated by the WIDE window (1c; == strike_window by default) and, when rewards.face_gate_by_pos
    is on, by the proximity power-gate (see ``_pos_gate``)."""
    cmd = _cmd(env, command_name)
    raw = _normal_kernel_raw(cmd, std)
    win = _window_wide(cmd)
    _dbg_log(cmd, "racket_normal", raw, win)
    return raw * win.float() * _pos_gate(cmd, pos_gate_radius)


def base_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track desired base XY position before the strike (encourages repositioning footwork)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "base", raw, cmd.pre_strike)
    return raw * cmd.pre_strike.float()


def post_strike_brake(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """POSITIVE braking reward through the FOLLOW-THROUGH (2026-07-07 continuous-rally upgrade).

    Deploy P7 failure mode: the walk-and-strike lunge carries base momentum past the strike; with
    nothing positive active in the tts<0 segment (every goal term is pre_strike/strike_window gated)
    the policy has no incentive to arrest it, and over consecutive swings the displacement
    accumulates until a swing starts from an untrained stance and falls. This term pays
    ``exp(-(|v_base_xy|/std)^2)`` ONLY in the follow-through window::

        (~pre_strike) & (~strike_window)

    i.e. from strike-window EXIT (tts < -strike_window_s) to the clip wrap — it can never touch the
    strike itself (the swing's through-speed is strike_window-protected), and on the wrap step tts
    snaps positive for the next swing so the window closes exactly at the wrap. During a post-wrap
    HOLD, ``pre_strike`` is True (the hold freezes tts positive at the windup value), so braking
    there is ``hold_ready``'s job (stillness x planted feet), not this term's. The window length is
    clip-clocked (not policy-controllable), so the bounded positive income cannot be farmed by
    prolonging it. Deliberately NO position target here: pulling toward any station mid-follow-
    through fights the swing's natural momentum sink — position homing is ``base_position``'s job
    once the next station appears at the wrap.
    """
    cmd = _cmd(env, command_name)
    v_xy = torch.norm(cmd.robot.data.root_lin_vel_w[:, :2], dim=-1)
    raw = torch.exp(-torch.square(v_xy / std))
    gate = (~cmd.pre_strike) & (~cmd.strike_window)
    _dbg_log(cmd, "post_strike_brake", raw, gate)
    return raw * gate.float()


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


def hold_heading(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.6
) -> torch.Tensor:
    """Reward re-squaring to world +x during a recovery hold.

    A yawed stand-start distribution supplies the missing recovery states; this term is
    deliberately zero outside ``in_hold`` so it cannot reshape the strike itself.
    """
    if not math.isfinite(float(std)) or float(std) <= 0.0:
        raise ValueError(f"hold_heading std must be finite and > 0, got {std!r}")
    cmd = _cmd(env, command_name)
    in_hold = getattr(cmd._motion(), "in_hold", None)
    if in_hold is None:
        return torch.zeros(cmd.num_envs, device=cmd.device)
    q = cmd.base_quat_w  # scalar-first (w, x, y, z)
    forward_x = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
    forward_y = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    yaw = torch.atan2(forward_y, forward_x)
    return torch.exp(-torch.square(yaw) / std**2) * in_hold.float()


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
    active ``pre_strike`` motion but explicitly OFF during the frozen pre-swing hold: ``hold_ready``
    owns hold stillness, while paying ``base_decel``'s nonzero target speed there asks the base to move
    and creates a contradictory objective. The strike swing and post-strike recovery are untouched
    (post-strike the distance to the OLD swung-through target would otherwise command a bogus speed-up). Base velocity
    is the WORLD planar root velocity (same source as hold_ready); v_gain [1/s] is the P-gain of the
    pseudo velocity command, v_max [m/s] its cap, std [m/s] the kernel width. RewTerm weight is
    POSITIVE; default weight 0.0 = OFF (flag-gated via task.rewards.base_decel_weight)."""
    cmd = _cmd(env, command_name)
    planar_err = torch.norm(cmd.racket_target_pos_w[:, :2] - cmd.racket_pos_w[:, :2], dim=-1)
    v_des = (v_gain * planar_err).clamp(0.0, v_max)
    v_base = torch.norm(cmd.robot.data.root_lin_vel_w[:, :2], dim=-1)
    raw = torch.exp(-torch.square(v_base - v_des) / std**2)
    in_hold = getattr(cmd._motion(), "in_hold", None)
    active = cmd.pre_strike if in_hold is None else (cmd.pre_strike & ~in_hold)
    return raw * active.float()


def racket_strike_success(
    env: ManagerBasedRLEnv, command_name: str, std_pos: float, std_vel: float, std_normal: float
) -> torch.Tensor:
    """MULTIPLICATIVE strike success R_pos * R_vel * R_normal, gated to the LEGACY strike window
    (``strike_window_s``). Unlike the additive racket terms (which give partial credit for getting only
    position OR velocity right), the product is high ONLY when position AND velocity AND normal are all
    good at once — a true hit. RewTerm weight is POSITIVE.

    1c split windows deliberately do NOT narrow this term (R3b forensics 2026-07-08): this is a BONUS
    channel (reward_staged_design §D: landing/net/spin/success 不动——验证奖金不是引导), and the vrr
    scoring gates (exact_strike, vb capture) are window-independent too. The first 1c implementation
    reused the internally window-gated kernels, so the product's support collapsed to the window
    INTERSECTION = the ±0.02 s tight position window (3 frames @50 Hz instead of 13): the true-hit
    bonus lost ~17x income (R3b 0.0021 vs R1b 0.0350) exactly when the tight window had already cut the
    dense position money, the policy farmed the wide vel/normal channels instead of contact, and the
    forehand missed the vb capture gate every swing (hit rate 0, return rate ~0). The product now uses
    the UNGATED kernels x the legacy window: byte-identical when the windows are not split (bool win:
    win^3 == win), and under split windows contact precision is still enforced by the position kernel
    itself + the vb capture gate — not by shrinking the bonus support.
    The proximity power-gate is deliberately NOT passed down here: success is already multiplicative
    (the design keeps the big money on the ungated product)."""
    cmd = _cmd(env, command_name)
    raw = (
        _pos_kernel_raw(cmd, std_pos)
        * _vel_kernel_raw(cmd, std_vel)
        * _normal_kernel_raw(cmd, std_normal)
    )
    return raw * cmd.strike_window.float()


def racket_guidance(env: ManagerBasedRLEnv, command_name: str, d_max: float = 0.5) -> torch.Tensor:
    """Constant guidance penalty toward the racket target (reward_staged_design 2026-07-08 §② B2):
    ``min(||racket_FK - target||, d_max)``, paid every pre-strike AND in-window step (union: from
    swing start through the strike window; the post-strike follow-through is untouched). This is
    the "挥拍到指定位置" gradient that exists even when the paddle is far outside every exp
    kernel's responsive band (the exp-starvation antidote); ``min(·, d_max)`` caps the burden so a
    far target can never drown the imitation signal (risk ⑤-1). Returns a POSITIVE magnitude —
    the RewTerm weight is NEGATIVE (set via rewards.racket_guidance_weight; cfg default 0.0 = off,
    the term is skipped). 人话:挥不到球也天天有"往哪挥"的工资单,小而恒。"""
    cmd = _cmd(env, command_name)
    dist = torch.norm(cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1)
    active = cmd.pre_strike | cmd.strike_window
    return dist.clamp(max=float(d_max)) * active.float()


def racket_face_guidance(
    env: ManagerBasedRLEnv, command_name: str, theta_max: float = 1.5707963
) -> torch.Tensor:
    """Constant FACE-ANGLE guidance penalty (2026-07-10, M3c 死区解药): ``min(angle, theta_max)``
    between the achieved mount normal and the demanded face normal, paid every pre-strike AND
    in-window step (same active mask as ``racket_guidance``). The exp face kernel has ~zero
    gradient beyond ~3·std (M3c 卡在 33°、v5syn 反手起步 ~53° 都在死区里) — this linear term is
    the face-channel twin of the position guidance: a small constant "which way to turn the
    blade" wage that never starves. The (measured, target) pair comes from ``_face_pair`` — the
    SAME frame the exp kernel uses. It must, or the two face terms fight: the original inline pick
    read the sign-flipped ``racket_normal_w`` against the A-frame bank target, so on sign=-1
    (backhand) clips this linear term pulled the blade toward the WRONG face with live gradient —
    worse than the dead exp kernel it was meant to rescue (2026-07-09 病因定案 + R9u/M3d-live
    止损). Returns POSITIVE radians — the RewTerm weight is NEGATIVE
    (rewards.racket_face_guidance_weight; cfg default 0.0 = term skipped, byte-identical).
    NOTE theta_max defaults to pi/2: rescues starting deeper than 90° (M3b-type 116° dead-zone
    starts) must pass theta_max=pi, or the clamp zeroes the gradient exactly where it is needed.
    HOT-RESTART note (M3c/M2f-type checkpoints onto the fixed frame): with weight != 0 this term
    steps once at restart (pre-fix backhand read ~146°, clamped to the pi/2 constant with zero
    gradient; post-fix it reads the true ~0.6 rad with live gradient — a one-off reward-level
    shift of ~+0.98*|weight| per active step). Watch value_loss/KL and the new
    face_cmd_normal_error_deg metric for the first few hundred iterations.
    人话:拍面反了 90° 时 exp 核一分钱梯度都不给,这里每一度都扣一点——把反面的拍子一路拉回来。"""
    cmd = _cmd(env, command_name)
    measured, target_normal = _face_pair(cmd)
    cos_ang = torch.sum(measured * target_normal, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    active = cmd.pre_strike | cmd.strike_window
    return angle.clamp(max=float(theta_max)) * active.float()


def tracking_envelope_violation(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str],
    ignore_hold: bool = False,
) -> torch.Tensor:
    """R-b envelope-as-penalty (reward_staged_design 2026-07-08 §⑥): per-step indicator of the
    tracking-envelope violation that used to TERMINATE the episode — the union of the two removed
    terminations, with the SAME z-only expressions (terminations.bad_anchor_pos_z_only |
    bad_motion_body_pos_z_only over the feet+wrists list). Returns 1.0 while violating, else 0.0;
    the RewTerm weight is NEGATIVE (terminations.envelope_penalty_weight, e.g. -1.0 => -0.02/step
    @50 Hz), so standing in the violation zone costs money instead of ending the episode.
    ``command_name`` is the MOTION command ("motion"). 人话:跟丢参考不再判死,改成站在违规区里
    每秒扣钱。Weight 0.0 (cfg default) = term skipped, byte-identical."""
    from whole_body_tracking.tasks.tracking.mdp.terminations import (
        bad_anchor_pos_z_only,
        bad_motion_body_pos_z_only,
    )

    viol = bad_anchor_pos_z_only(env, command_name, threshold) | bad_motion_body_pos_z_only(
        env, command_name, threshold, body_names
    )
    if ignore_hold:
        viol = _ignore_hold(env.command_manager.get_term(command_name), viol, True)
    return viol.float()


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
# limit written from effort_limit_sim). This term is a hardware-envelope claim: when enabled, missing
# indices/data or invalid limits MUST stop the run rather than reporting a counterfeit zero saturation.
# ============================================================================================== #
_TORQUE_SAT_JOINT_EXPR = [".*shoulder.*", ".*elbow.*", ".*wrist.*", "waist_.*_joint"]


def _torque_sat_joint_idx(env: ManagerBasedRLEnv, command_name: str):
    """Resolve+cache the arm+waist joint indices on the command term (once)."""
    cmd = _cmd(env, command_name)
    idx = getattr(cmd, "_torque_sat_joint_idx", None)
    if idx is None:
        try:
            idx = list(cmd.robot.find_joints(_TORQUE_SAT_JOINT_EXPR)[0])
        except Exception as exc:
            raise RuntimeError(
                "arm_torque_saturation could not resolve shoulder/elbow/wrist/waist joints"
            ) from exc
        if not idx:
            raise RuntimeError(
                "arm_torque_saturation resolved zero shoulder/elbow/wrist/waist joints"
            )
        if len(idx) != len(set(idx)):
            raise RuntimeError(
                f"arm_torque_saturation resolved duplicate joint indices: {idx}"
            )
        cmd._torque_sat_joint_idx = idx
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
    if tau is None or lim is None:
        raise RuntimeError(
            "arm_torque_saturation requires robot.data.computed_torque (pre-clip) and "
            "robot.data.joint_effort_limits; the active actuator backend exposes neither/both "
            "incorrectly"
        )
    if tau.ndim != 2 or lim.ndim != 2 or tau.shape != lim.shape:
        raise RuntimeError(
            f"arm_torque_saturation expected matching [env,joint] tensors, got "
            f"computed_torque={tuple(tau.shape)}, limits={tuple(lim.shape)}"
        )
    if max(idx) >= tau.shape[1]:
        raise RuntimeError(
            f"arm_torque_saturation joint index {max(idx)} exceeds tensor width {tau.shape[1]}"
        )
    tau_a = torch.abs(tau[:, idx])
    lim_a = lim[:, idx]
    # The boolean checks synchronize a CUDA stream, so run the mechanism/data contract once,
    # not at every 50-Hz reward evaluation. Later non-finite torques still propagate into the
    # reward (and PPO's normal non-finite guard) rather than being converted into a fake zero.
    if not getattr(cmd, "_torque_sat_contract_checked", False):
        if not bool(torch.isfinite(tau_a).all()) or not bool(torch.isfinite(lim_a).all()):
            raise RuntimeError("arm_torque_saturation received non-finite torque/effort-limit data")
        if bool((lim_a <= 0.0).any()):
            raise RuntimeError("arm_torque_saturation requires strictly positive effort limits")
        cmd._torque_sat_contract_checked = True
    over = (tau_a / lim_a - 1.0).clamp(min=0.0)  # relu(ratio - 1): the un-deliverable fraction
    frac = over.mean(dim=-1)
    cmd.metrics["arm_torque_sat_frac"] = frac  # watch-metric: should fall toward 0 during fine-tune
    return frac

def motion_body_pos_swing_only(env, command_name: str, std: float, body_names=None,
                               window_scale: float = 1.0, window_command_name: str | None = None):
    """motion_relative_body_position_error_exp gated to ~in_hold (2026-07-05): during
    hold the joint reference is the default STAND (commands.joint_pos) while the frozen
    body refs still show clip frame 0's crouch — un-gated, the two imitation pulls
    fight and the policy settles into the splayed-feet crouch-stand. Swing-only.
    window_scale/window_command_name: V2 in-window imitation yield, forwarded to the base
    func (see rewards._apply_window_scale); defaults = no-op."""
    from .rewards import motion_relative_body_position_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_relative_body_position_error_exp(env, command_name, std, body_names,
                                                window_scale, window_command_name)
    return torch.where(cmd.in_hold, torch.zeros_like(r), r)


def motion_body_ori_swing_only(env, command_name: str, std: float, body_names=None,
                               window_scale: float = 1.0, window_command_name: str | None = None):
    """See motion_body_pos_swing_only."""
    from .rewards import motion_relative_body_orientation_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_relative_body_orientation_error_exp(env, command_name, std, body_names,
                                                   window_scale, window_command_name)
    return torch.where(cmd.in_hold, torch.zeros_like(r), r)


def motion_body_lin_vel_swing_only(env, command_name: str, std: float, body_names=None,
                                   window_scale: float = 1.0,
                                   window_command_name: str | None = None):
    """Body linear-velocity imitation with no income during a recovery hold.

    ``MotionCommand.body_lin_vel_w`` correctly exposes a stationary (zero-velocity)
    reference while held.  Paying the ordinary velocity kernel for that reference is still
    wrong for HitterPure rally recovery, though: it rewards *remaining still* while
    ``hold_heading`` asks the base/waist to turn back toward the table.  The video teacher is
    an imitation prior for the swing, not a hold controller, so the whole term is silent in
    hold just like the position/orientation terms above.
    """
    from .rewards import motion_global_body_linear_velocity_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_global_body_linear_velocity_error_exp(
        env, command_name, std, body_names, window_scale, window_command_name
    )
    return torch.where(cmd.in_hold, torch.zeros_like(r), r)


def motion_body_ang_vel_swing_only(env, command_name: str, std: float, body_names=None,
                                   window_scale: float = 1.0,
                                   window_command_name: str | None = None):
    """Angular-velocity counterpart of :func:`motion_body_lin_vel_swing_only`."""
    from .rewards import motion_global_body_angular_velocity_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_global_body_angular_velocity_error_exp(
        env, command_name, std, body_names, window_scale, window_command_name
    )
    return torch.where(cmd.in_hold, torch.zeros_like(r), r)


def _ignore_hold(command, value: torch.Tensor, ignore_hold: bool) -> torch.Tensor:
    """Mask a reference-relative termination during hold, failing loud on a bad command.

    The absolute fall guards remain separate termination terms.  This helper is deliberately
    not a permissive ``getattr(..., False)`` fallback: configuring ``ignore_hold=True`` on a
    command without an ``in_hold`` contract would silently reintroduce the reset-time bug.
    """
    if not ignore_hold:
        return value
    if not hasattr(command, "in_hold"):
        raise RuntimeError("ignore_hold=True requires the command to expose an in_hold mask")
    return value & ~command.in_hold.bool()


def bad_anchor_pos_z_only_hold_aware(
    env, command_name: str, threshold: float, ignore_hold: bool = False
) -> torch.Tensor:
    """Reference torso-height envelope with an explicit held-RSI exclusion."""
    from .terminations import bad_anchor_pos_z_only
    command = env.command_manager.get_term(command_name)
    return _ignore_hold(
        command, bad_anchor_pos_z_only(env, command_name, threshold), ignore_hold
    )


def bad_anchor_ori_hold_aware(
    env, asset_cfg, command_name: str, threshold: float, ignore_hold: bool = False
) -> torch.Tensor:
    """Reference orientation envelope with an explicit held-RSI exclusion."""
    from .terminations import bad_anchor_ori
    command = env.command_manager.get_term(command_name)
    return _ignore_hold(
        command,
        bad_anchor_ori(env, asset_cfg, command_name, threshold),
        ignore_hold,
    )


def bad_motion_body_pos_z_only_hold_aware(
    env, command_name: str, threshold: float, body_names=None,
    ignore_hold: bool = False,
) -> torch.Tensor:
    """Reference body-height envelope with an explicit held-RSI exclusion."""
    from .terminations import bad_motion_body_pos_z_only
    command = env.command_manager.get_term(command_name)
    return _ignore_hold(
        command,
        bad_motion_body_pos_z_only(env, command_name, threshold, body_names),
        ignore_hold,
    )

def foot_orientation_discipline(env, command_name: str, asset_cfg, hold_gate: bool = False):
    """L1 deviation of the foot-orientation joints (hip yaw/roll, ankle roll) from the
    REFERENCE joint positions — hold-aware via commands.joint_pos (default stand during
    hold, clip footwork during swings). 2026-07-05: with no joint-level imitation in
    the stack these DOF were reward-free, and the policy twisted the feet to
    -1.13/+0.90 rad during swings/side-switches vs a reference envelope of ±0.41
    (Gate 2.5 diag) — the 'weird foot placement' at strike/switch. Use a NEGATIVE
    weight (penalty); keep it small so it disciplines feet without taxing the lunge.
    When ``hold_gate`` is true, the term is zero during the recovery hold: otherwise the
    square-stand joint reference penalizes the hip-yaw motion needed to re-square the base.
    """
    cmd = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    ref = cmd.joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(q - ref), dim=1)
    if hold_gate:
        penalty = torch.where(cmd.in_hold, torch.zeros_like(penalty), penalty)
    return penalty
