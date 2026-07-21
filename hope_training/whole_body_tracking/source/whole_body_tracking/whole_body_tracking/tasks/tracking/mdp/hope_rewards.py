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


_BASE_DECEL_ACTIVATION_ATTR = "_hope_base_decel_activation_counters"
_BASE_DECEL_LAST_STEP_ATTR = "_hope_base_decel_activation_last_step"
_BASE_DECEL_LAST_SIGNATURE_ATTR = "_hope_base_decel_activation_last_signature"
_BASE_DECEL_ELIGIBLE_COUNT = "base_decel_eligible_sample_count"
_BASE_DECEL_RAW_KERNEL_SUM = "base_decel_raw_kernel_sum"
_BASE_DECEL_RAW_NONZERO_COUNT = "base_decel_raw_kernel_nonzero_sample_count"


def _base_decel_values(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_gain: float,
    v_max: float,
    std: float,
) -> tuple[RacketTargetCommand, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact base-decel command, raw kernel, eligibility mask, and reward.

    Both the actual RewardTerm and the weight-independent activation observer below use this
    function.  Keeping the arithmetic in one place prevents a zero-weight control from being
    measured with a mask or kernel that differs from the treatment's real reward execution.
    """

    cmd = _cmd(env, command_name)
    planar_err = torch.norm(
        cmd.racket_target_pos_w[:, :2] - cmd.racket_pos_w[:, :2], dim=-1
    )
    v_des = (v_gain * planar_err).clamp(0.0, v_max)
    v_base = torch.norm(cmd.robot.data.root_lin_vel_w[:, :2], dim=-1)
    raw = torch.exp(-torch.square(v_base - v_des) / std**2)
    in_hold = getattr(cmd._motion(), "in_hold", None)
    eligible = cmd.pre_strike if in_hold is None else (cmd.pre_strike & ~in_hold)
    reward = raw * eligible.float()
    return cmd, raw, eligible, reward


def _base_decel_counter_state(
    cmd: RacketTargetCommand, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Lazily allocate device scalar counters without changing command configuration."""

    state = getattr(cmd, _BASE_DECEL_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _BASE_DECEL_ELIGIBLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _BASE_DECEL_RAW_KERNEL_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _BASE_DECEL_RAW_NONZERO_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
        }
        setattr(cmd, _BASE_DECEL_ACTIVATION_ATTR, state)
    return state


def _base_decel_step_token(env: ManagerBasedRLEnv) -> int | None:
    """Return Isaac's host-side simulator-step token when the environment exposes it.

    The real manager environment owns an integer ``common_step_counter``.  Synthetic callers that
    do not expose it still get correct reward arithmetic, but cannot request same-step idempotence.
    Avoiding ``int(tensor)`` here is deliberate: activation accounting must never add a CUDA sync.
    """

    token = getattr(env, "common_step_counter", None)
    return token if type(token) is int else None


def _record_base_decel_activation(
    env: ManagerBasedRLEnv,
    cmd: RacketTargetCommand,
    raw: torch.Tensor,
    eligible: torch.Tensor,
    reward: torch.Tensor,
    *,
    signature: tuple[float, float, float],
) -> None:
    """Accumulate one unique simulator step of activation evidence.

    A future unconditional step hook must call :func:`observe_base_decel_activation` for both the
    zero-weight control and the treatment.  The treatment's RewardManager also calls
    :func:`base_decel_tracking`; the shared ``common_step_counter`` token makes those two calls
    idempotent.  A parameter mismatch in the same step fails loudly instead of letting the hook
    attest a different kernel than the reward used.
    """

    token = _base_decel_step_token(env)
    if token is not None:
        previous_token = getattr(cmd, _BASE_DECEL_LAST_STEP_ATTR, None)
        if previous_token == token:
            previous_signature = getattr(cmd, _BASE_DECEL_LAST_SIGNATURE_ATTR, None)
            if previous_signature != signature:
                raise RuntimeError(
                    "base-decel activation observer and RewardTerm used different "
                    "v_gain/v_max/std values in the same simulator step"
                )
            return

    state = _base_decel_counter_state(cmd, raw)
    state[_BASE_DECEL_ELIGIBLE_COUNT].add_(
        eligible.detach().sum(dtype=torch.long)
    )
    # ``reward`` is the exact raw kernel after the real ``pre_strike & ~in_hold`` gate.  Summing
    # this tensor (rather than recomputing or indexing ``raw``) preserves NaN/Inf evidence and the
    # treatment's executed arithmetic exactly.
    state[_BASE_DECEL_RAW_KERNEL_SUM].add_(reward.detach().sum())
    raw_detached = raw.detach()
    state[_BASE_DECEL_RAW_NONZERO_COUNT].add_(
        (
            eligible.detach()
            & torch.isfinite(raw_detached)
            & raw_detached.gt(0)
        ).sum(dtype=torch.long)
    )
    if token is not None:
        setattr(cmd, _BASE_DECEL_LAST_STEP_ATTR, token)
        setattr(cmd, _BASE_DECEL_LAST_SIGNATURE_ATTR, signature)


def observe_base_decel_activation(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_gain: float = 2.0,
    v_max: float = 1.6,
    std: float = 0.4,
) -> None:
    """Measure base-decel activation without applying any reward.

    Both arms call this observer from :func:`base_decel_activation_probe`, a nonzero-weight
    instrumentation RewardTerm.  Keeping the observer in RewardManager is essential: Isaac 2.1
    computes reward before reset/command updates, so a command-stage observer would measure the
    next command state while the treatment's real RewardTerm measured the previous state.  The
    probe performs no random draw or simulator write and shares the treatment's exact kernel/mask.
    If the treatment RewardTerm also runs in that reward stage, ``env.common_step_counter`` makes
    accounting idempotent.
    """

    cmd, raw, eligible, reward = _base_decel_values(
        env, command_name, v_gain, v_max, std
    )
    _record_base_decel_activation(
        env,
        cmd,
        raw,
        eligible,
        reward,
        signature=(float(v_gain), float(v_max), float(std)),
    )


def base_decel_activation_probe(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_gain: float = 2.0,
    v_max: float = 1.6,
    std: float = 0.4,
) -> torch.Tensor:
    """Reward-stage activation probe whose weighted reward contribution is identically zero.

    The experiment translator gives this term a nonzero manager weight in *both* control and
    treatment so IsaacLab cannot optimize it away.  Returning an environment-sized zero tensor
    means even a weight of ``1.0`` changes neither per-term reward nor total reward.  The treatment's
    later :func:`base_decel_tracking` call sees the same state and deduplicates on the shared step
    token.
    """

    observe_base_decel_activation(env, command_name, v_gain, v_max, std)
    cmd = _cmd(env, command_name)
    return torch.zeros_like(cmd.racket_target_pos_w[:, 0])


def consume_base_decel_activation_counters(
    env: ManagerBasedRLEnv, command_name: str
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update's base-decel activation device scalars.

    The last simulator-step token intentionally survives the reset.  A logger that consumes after
    rollout and then observes the same step cannot accidentally charge that step to the next PPO
    update.  A second consume therefore returns exact scalar zeros.
    """

    cmd = _cmd(env, command_name)
    state = _base_decel_counter_state(cmd, cmd.racket_target_pos_w)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    # RewardManager may lazily create these counters while Isaac is stepping under
    # ``torch.inference_mode()``.  PyTorch deliberately refuses to mutate such an inference
    # tensor from the logger's normal-mode context, even though cloning it for the snapshot is
    # allowed.  Reset in inference mode as well; this changes only the private accounting
    # scalars and preserves their device/dtype and the last-step deduplication token.
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


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
    cmd, raw, eligible, reward = _base_decel_values(
        env, command_name, v_gain, v_max, std
    )
    _record_base_decel_activation(
        env,
        cmd,
        raw,
        eligible,
        reward,
        signature=(float(v_gain), float(v_max), float(std)),
    )
    return reward


_QDOT_ACTIVATION_ATTR = "_hope_qdot_hinge_activation_counters"
_QDOT_OBSERVED_STEP_ATTR = "_hope_qdot_hinge_observed_step"
_QDOT_ACTIVE_STEP_ATTR = "_hope_qdot_hinge_active_step"
_QDOT_OBSERVED_COUNT = "observed_sample_count"
_QDOT_ACTIVE_COUNT = "hinge_active_sample_count"
_QDOT_EXCESS_COUNT = "excess_sample_count"
_QDOT_EXCESS_SQUARE_SUM = "normalized_excess_square_sum"


def _joint_velocity_limit_hinge_values(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin: float = 0.85,
    expected_joint_count: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env hinge value and the per-env above-margin mask.

    The first returned tensor is non-negative; the :class:`RewardTermCfg` weight must therefore
    be non-positive.  Unlike ``action_rate_l2``, this term measures the robot's *realized* joint
    velocity against the actual articulation limits, in the exact runtime articulation order::

        mean_j(relu(abs(qd_j) / qd_limit_j - margin) ** 2)

    This source gate is deliberately A3-specific: all 31 runtime joints must be selected in
    identity order.  Missing/reordered joints and zero/non-finite limits fail closed rather than
    silently changing the denominator.  The limit tensor is read on every call: Isaac may update
    articulation limits at runtime, so caching the first value would cease to measure the actual
    plant.  CUDA validity assertions stay asynchronous rather than forcing a host sync per step.
    """
    if type(expected_joint_count) is not int or expected_joint_count != 31:
        raise ValueError(
            "joint_velocity_limit_hinge requires the exact 31-joint A3 runtime order"
        )
    if isinstance(margin, bool):
        raise ValueError("joint_velocity_limit_hinge margin must be finite and in (0, 1)")
    margin = float(margin)
    if not math.isfinite(margin) or not 0.0 < margin < 1.0:
        raise ValueError("joint_velocity_limit_hinge margin must be finite and in (0, 1)")

    asset = env.scene[asset_cfg.name]
    data = asset.data
    joint_names = list(getattr(data, "joint_names", getattr(asset, "joint_names", ())))
    if (
        len(joint_names) != expected_joint_count
        or any(not str(name) for name in joint_names)
        or len(set(joint_names)) != expected_joint_count
    ):
        raise RuntimeError(
            "joint_velocity_limit_hinge requires exactly 31 unique runtime joint names"
        )

    raw_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if isinstance(raw_ids, slice):
        joint_ids = list(range(expected_joint_count))[raw_ids]
    else:
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        joint_ids = [int(value) for value in raw_ids]
    if joint_ids != list(range(expected_joint_count)):
        raise RuntimeError(
            "joint_velocity_limit_hinge requires identity 31-joint articulation order, "
            f"got joint_ids={joint_ids}"
        )

    joint_vel = data.joint_vel
    if joint_vel.ndim != 2 or tuple(joint_vel.shape)[1] != expected_joint_count:
        raise RuntimeError(
            "joint_velocity_limit_hinge requires joint_vel shaped [num_envs, 31], "
            f"got {tuple(joint_vel.shape)}"
        )

    runtime_limits = data.joint_vel_limits
    if runtime_limits.ndim == 1:
        if tuple(runtime_limits.shape) != (expected_joint_count,):
            raise RuntimeError(
                "joint_velocity_limit_hinge requires 31 articulation velocity limits, "
                f"got {tuple(runtime_limits.shape)}"
            )
        limits = runtime_limits
        limits_valid = torch.all(torch.isfinite(limits) & (limits > 0.0))
    elif runtime_limits.ndim == 2:
        if (
            tuple(runtime_limits.shape)[0] < 1
            or tuple(runtime_limits.shape)[1] != expected_joint_count
            or tuple(runtime_limits.shape)[0] not in (1, tuple(joint_vel.shape)[0])
        ):
            raise RuntimeError(
                "joint_velocity_limit_hinge requires joint_vel_limits shaped [31] or "
                f"[num_envs, 31], got {tuple(runtime_limits.shape)}"
            )
        limits = runtime_limits
        limits_valid = torch.all(torch.isfinite(limits) & (limits > 0.0))
        if tuple(runtime_limits.shape)[0] > 1:
            limits_valid = limits_valid & torch.all(
                runtime_limits == runtime_limits[0].unsqueeze(0)
            )
    else:
        raise RuntimeError(
            "joint_velocity_limit_hinge requires joint_vel_limits shaped [31] or "
            f"[num_envs, 31], got {tuple(runtime_limits.shape)}"
        )

    # CPU tests and CPU VecEnv paths get the precise diagnostic immediately.  On CUDA,
    # torch._assert_async fails the training stream without a per-step host synchronization.
    if limits_valid.device.type == "cpu":
        if not bool(limits_valid):
            raise RuntimeError(
                "joint_velocity_limit_hinge requires finite, positive, identical "
                "articulation velocity limits in every environment"
            )
    else:
        torch._assert_async(limits_valid)

    normalized_excess = torch.relu(torch.abs(joint_vel) / limits - margin)
    squared_mean = torch.mean(torch.square(normalized_excess), dim=-1)
    return squared_mean, torch.any(normalized_excess > 0.0, dim=-1)


def _qdot_activation_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _QDOT_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _QDOT_OBSERVED_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDOT_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDOT_EXCESS_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDOT_EXCESS_SQUARE_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
        }
        setattr(env, _QDOT_ACTIVATION_ATTR, state)
    return state


def _record_joint_velocity_limit_hinge_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    excess_mask: torch.Tensor,
    *,
    hinge_active: bool,
) -> None:
    """Book one simulator step, deduplicating probe and real RewardTerm calls."""

    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _qdot_activation_counter_state(env, values)
    if token is None or getattr(env, _QDOT_OBSERVED_STEP_ATTR, None) != token:
        state[_QDOT_OBSERVED_COUNT].add_(values.numel())
        state[_QDOT_EXCESS_COUNT].add_(
            excess_mask.detach().sum(dtype=torch.long)
        )
        state[_QDOT_EXCESS_SQUARE_SUM].add_(values.detach().sum())
        if token is not None:
            setattr(env, _QDOT_OBSERVED_STEP_ATTR, token)
    if hinge_active and (
        token is None or getattr(env, _QDOT_ACTIVE_STEP_ATTR, None) != token
    ):
        state[_QDOT_ACTIVE_COUNT].add_(values.numel())
        if token is not None:
            setattr(env, _QDOT_ACTIVE_STEP_ATTR, token)


def joint_velocity_limit_hinge_probe(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin: float = 0.85,
    expected_joint_count: int = 31,
) -> torch.Tensor:
    """Observe realized qdot without changing the scalar reward.

    A nonzero manager weight is safe because this function returns exact zeros.  The probe and the
    actual hinge share the validation/math helper and a simulator-step token, so treatment samples
    are observed once while hinge-active samples are booked separately.
    """

    values, excess_mask = _joint_velocity_limit_hinge_values(
        env, asset_cfg, margin, expected_joint_count
    )
    _record_joint_velocity_limit_hinge_activation(
        env, values, excess_mask, hinge_active=False
    )
    return torch.zeros_like(values)


def joint_velocity_limit_hinge(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin: float = 0.85,
    expected_joint_count: int = 31,
) -> torch.Tensor:
    """Penalize and attest the normalized joint-speed tail near articulation limits."""

    values, excess_mask = _joint_velocity_limit_hinge_values(
        env, asset_cfg, margin, expected_joint_count
    )
    _record_joint_velocity_limit_hinge_activation(
        env, values, excess_mask, hinge_active=True
    )
    return values


def consume_joint_velocity_limit_hinge_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update's qdot observer/activation ledger."""

    template = env.scene["robot"].data.joint_vel
    state = _qdot_activation_counter_state(env, template)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


# The deploy-space recovery smoother is deliberately narrower than action_rate_l2: it observes the
# affine-transformed, q_des-clamped target, only on the three waist + twelve leg joints, and only in
# the same swing's post-contact recovery window.  Keep the semantic joint set explicit so a renamed,
# missing, or accidentally arm-inclusive action contract fails closed instead of changing the mean.
_PROCESSED_QDES_RECOVERY_JOINT_NAMES = frozenset(
    {
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    }
)
_PROCESSED_QDES_CONTROL_DT_S = 0.02
_PROCESSED_QDES_ACTIVATION_ATTR = "_hope_processed_qdes_slew_activation_counters"
_PROCESSED_QDES_OBSERVED_STEP_ATTR = "_hope_processed_qdes_slew_observed_step"
_PROCESSED_QDES_ACTIVE_STEP_ATTR = "_hope_processed_qdes_slew_active_step"
_PROCESSED_QDES_SIGNATURE_ATTR = "_hope_processed_qdes_slew_signature"
_PROCESSED_QDES_OBSERVED_COUNT = "observed_sample_count"
_PROCESSED_QDES_VALID_COUNT = "previous_qdes_valid_sample_count"
_PROCESSED_QDES_INVALID_COUNT = "previous_qdes_invalid_first_step_sample_count"
_PROCESSED_QDES_ELIGIBLE_COUNT = "recovery_eligible_sample_count"
_PROCESSED_QDES_ACTIVE_COUNT = "reward_enabled_eligible_sample_count"
_PROCESSED_QDES_TAIL_ACTIVE_COUNT = "tail_active_sample_count"
_PROCESSED_QDES_EXCESS_JOINT_COUNT = "above_margin_joint_count"
_PROCESSED_QDES_TAIL_SUM = "gated_tail_value_sum"


def _processed_qdes_slew_hinge_values(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    command_name: str = "racket_target",
    margin: float = 0.85,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gated tail value, eligibility, history validity, and per-joint excess.

    For each selected joint, ``u = abs(q_des[t] - q_des[t-1]) / (qdot_limit * 0.02)`` and::

        tail = 1 - exp(-square(relu(u - margin) / (1 - margin)))

    The environment reward is the mean over the exact 15-joint waist/leg set.  Reset-invalid
    history and samples outside the same-attempt recovery window return exact zero.
    """

    if action_name != "joint_pos":
        raise ValueError("processed_qdes_slew_hinge action_name must be exactly 'joint_pos'")
    if command_name != "racket_target":
        raise ValueError(
            "processed_qdes_slew_hinge command_name must be exactly 'racket_target'"
        )
    if isinstance(margin, bool):
        raise ValueError("processed_qdes_slew_hinge margin must be finite and in (0, 1)")
    margin = float(margin)
    if not math.isfinite(margin) or not 0.0 < margin < 1.0:
        raise ValueError("processed_qdes_slew_hinge margin must be finite and in (0, 1)")
    if isinstance(recovery_start_s, bool) or isinstance(recovery_end_s, bool):
        raise ValueError(
            "processed_qdes_slew_hinge recovery window must be finite and satisfy 0 <= start < end"
        )
    recovery_start_s = float(recovery_start_s)
    recovery_end_s = float(recovery_end_s)
    if (
        not math.isfinite(recovery_start_s)
        or not math.isfinite(recovery_end_s)
        or recovery_start_s < 0.0
        or recovery_start_s >= recovery_end_s
    ):
        raise ValueError(
            "processed_qdes_slew_hinge recovery window must be finite and satisfy 0 <= start < end"
        )
    raw_control_dt = getattr(env, "step_dt", None)
    if (
        isinstance(raw_control_dt, bool)
        or not isinstance(raw_control_dt, (int, float))
        or not math.isfinite(float(raw_control_dt))
        or not math.isclose(
            float(raw_control_dt),
            _PROCESSED_QDES_CONTROL_DT_S,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires env.step_dt == 0.02 s control time"
        )

    action = env.action_manager.get_term(action_name)
    processed = getattr(action, "processed_actions", None)
    previous = getattr(action, "previous_processed_qdes", None)
    previous_valid = getattr(action, "previous_processed_qdes_valid", None)
    if (
        not torch.is_tensor(processed)
        or not torch.is_tensor(previous)
        or processed.ndim != 2
        or previous.shape != processed.shape
        or not torch.is_tensor(previous_valid)
        or previous_valid.dtype != torch.bool
        or tuple(previous_valid.shape) != (tuple(processed.shape)[0],)
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires reset-aware processed q_des history from joint_pos"
        )

    asset = getattr(action, "_asset", None)
    data = getattr(asset, "data", None)
    runtime_names = list(
        getattr(data, "joint_names", getattr(asset, "joint_names", ()))
    )
    action_names = list(getattr(action, "_joint_names", ()))
    joint_count = len(runtime_names)
    if (
        joint_count != 31
        or len(set(runtime_names)) != joint_count
        or action_names != runtime_names
        or tuple(processed.shape)[1] != joint_count
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires identity 31-joint A3 action/articulation order"
        )
    raw_joint_ids = getattr(action, "_joint_ids", slice(None))
    if isinstance(raw_joint_ids, slice):
        action_joint_ids = list(range(joint_count))[raw_joint_ids]
    else:
        if hasattr(raw_joint_ids, "tolist"):
            raw_joint_ids = raw_joint_ids.tolist()
        action_joint_ids = [int(value) for value in raw_joint_ids]
    if action_joint_ids != list(range(joint_count)):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires identity 31-joint A3 action/articulation order"
        )
    selected = [
        index
        for index, name in enumerate(runtime_names)
        if name in _PROCESSED_QDES_RECOVERY_JOINT_NAMES
    ]
    if (
        len(selected) != 15
        or {runtime_names[index] for index in selected}
        != _PROCESSED_QDES_RECOVERY_JOINT_NAMES
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires the exact 15 A3 waist/leg joints"
        )

    runtime_limits = getattr(data, "joint_vel_limits", None)
    if not torch.is_tensor(runtime_limits):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires runtime articulation joint velocity limits"
        )
    if runtime_limits.ndim == 1 and tuple(runtime_limits.shape) == (joint_count,):
        selected_limits = runtime_limits[selected]
    elif (
        runtime_limits.ndim == 2
        and tuple(runtime_limits.shape)[1] == joint_count
        and tuple(runtime_limits.shape)[0] in (1, tuple(processed.shape)[0])
    ):
        selected_limits = runtime_limits[:, selected]
    else:
        raise RuntimeError(
            "processed_qdes_slew_hinge requires joint_vel_limits shaped [31], [1,31], or [num_envs,31]"
        )
    limits_valid = torch.all(torch.isfinite(selected_limits) & selected_limits.gt(0.0))
    if limits_valid.device.type == "cpu":
        if not bool(limits_valid):
            raise RuntimeError(
                "processed_qdes_slew_hinge requires finite positive waist/leg velocity limits"
            )
    else:
        torch._assert_async(limits_valid)

    current_selected = processed[:, selected]
    previous_selected = previous[:, selected]
    normalized = torch.abs(current_selected - previous_selected) / (
        selected_limits * _PROCESSED_QDES_CONTROL_DT_S
    )
    normalized_excess = torch.relu(normalized - margin)
    per_joint_tail = 1.0 - torch.exp(
        -torch.square(normalized_excess / (1.0 - margin))
    )
    raw_value = torch.mean(per_joint_tail, dim=-1)

    cmd = _cmd(env, command_name)
    clock = getattr(cmd, "post_strike_age_and_same_attempt", None)
    if not callable(clock):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires racket_target's same-attempt post-strike clock"
        )
    age_s, same_attempt = clock()
    if (
        not torch.is_tensor(age_s)
        or not torch.is_tensor(same_attempt)
        or tuple(age_s.shape) != tuple(previous_valid.shape)
        or tuple(same_attempt.shape) != tuple(previous_valid.shape)
        or same_attempt.dtype != torch.bool
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge received an invalid same-attempt post-strike clock"
        )
    in_recovery = (
        same_attempt
        & age_s.ge(recovery_start_s)
        & age_s.le(recovery_end_s)
    )
    eligible = previous_valid & in_recovery
    value = torch.where(eligible, raw_value, torch.zeros_like(raw_value))
    excess = eligible.unsqueeze(-1) & normalized_excess.gt(0.0)
    return value, eligible, previous_valid, excess


def _processed_qdes_slew_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _PROCESSED_QDES_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _PROCESSED_QDES_OBSERVED_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_VALID_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_INVALID_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_ELIGIBLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_TAIL_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_EXCESS_JOINT_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_TAIL_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
        }
        setattr(env, _PROCESSED_QDES_ACTIVATION_ATTR, state)
    return state


def _record_processed_qdes_slew_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    eligible: torch.Tensor,
    previous_valid: torch.Tensor,
    excess: torch.Tensor,
    *,
    hinge_active: bool,
    signature: tuple[str, str, float, float, float],
) -> None:
    """Book one simulator step, idempotently sharing the probe and real term."""

    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _processed_qdes_slew_counter_state(env, values)
    already_observed = (
        token is not None
        and getattr(env, _PROCESSED_QDES_OBSERVED_STEP_ATTR, None) == token
    )
    if already_observed:
        if getattr(env, _PROCESSED_QDES_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError(
                "processed q_des slew probe and RewardTerm used different parameters in one step"
            )
    else:
        state[_PROCESSED_QDES_OBSERVED_COUNT].add_(values.numel())
        state[_PROCESSED_QDES_VALID_COUNT].add_(
            previous_valid.detach().sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_INVALID_COUNT].add_(
            (~previous_valid.detach()).sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_ELIGIBLE_COUNT].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_TAIL_ACTIVE_COUNT].add_(
            torch.any(excess.detach(), dim=-1).sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_EXCESS_JOINT_COUNT].add_(
            excess.detach().sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_TAIL_SUM].add_(values.detach().sum())
        if token is not None:
            setattr(env, _PROCESSED_QDES_OBSERVED_STEP_ATTR, token)
            setattr(env, _PROCESSED_QDES_SIGNATURE_ATTR, signature)
    if hinge_active and (
        token is None or getattr(env, _PROCESSED_QDES_ACTIVE_STEP_ATTR, None) != token
    ):
        state[_PROCESSED_QDES_ACTIVE_COUNT].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        if token is not None:
            setattr(env, _PROCESSED_QDES_ACTIVE_STEP_ATTR, token)


def processed_qdes_slew_hinge_probe(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    command_name: str = "racket_target",
    margin: float = 0.85,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> torch.Tensor:
    """Measure processed-q_des recovery slew while contributing exact zero reward."""

    values, eligible, previous_valid, excess = _processed_qdes_slew_hinge_values(
        env, action_name, command_name, margin, recovery_start_s, recovery_end_s
    )
    _record_processed_qdes_slew_activation(
        env,
        values,
        eligible,
        previous_valid,
        excess,
        hinge_active=False,
        signature=(
            action_name,
            command_name,
            float(margin),
            float(recovery_start_s),
            float(recovery_end_s),
        ),
    )
    return torch.zeros_like(values)


def processed_qdes_slew_hinge(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    command_name: str = "racket_target",
    margin: float = 0.85,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> torch.Tensor:
    """Penalize deploy-space waist/leg q_des slew only during same-swing recovery."""

    values, eligible, previous_valid, excess = _processed_qdes_slew_hinge_values(
        env, action_name, command_name, margin, recovery_start_s, recovery_end_s
    )
    _record_processed_qdes_slew_activation(
        env,
        values,
        eligible,
        previous_valid,
        excess,
        hinge_active=True,
        signature=(
            action_name,
            command_name,
            float(margin),
            float(recovery_start_s),
            float(recovery_end_s),
        ),
    )
    return values


def consume_processed_qdes_slew_hinge_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update's processed-q_des recovery ledger."""

    action = env.action_manager.get_term("joint_pos")
    template = action.processed_actions[:, 0]
    state = _processed_qdes_slew_counter_state(env, template)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


# Wave-Q qdes limit barrier ------------------------------------------------------------------- #
#
# All-joint deploy-space q_des position-limit barrier.  Unlike processed_qdes_slew_hinge (a rate
# hinge on a 15-joint subset inside the recovery window), this term charges every one of the 31
# joints, every control step, as soon as the commanded target enters the margin band next to a
# position limit.  人话:哪个关节的目标角贴到限位边上就罚哪个,全身 31 个关节全程盯着,
# 不挑"最狠的几个"。
_QDES_LIMIT_BARRIER_JOINT_COUNT = 31
_QDES_LIMIT_BARRIER_ACTIVATION_ATTR = "_hope_qdes_limit_barrier_activation_counters"
_QDES_LIMIT_BARRIER_OBSERVED_STEP_ATTR = "_hope_qdes_limit_barrier_observed_step"
_QDES_LIMIT_BARRIER_ACTIVE_STEP_ATTR = "_hope_qdes_limit_barrier_active_step"
_QDES_LIMIT_BARRIER_SIGNATURE_ATTR = "_hope_qdes_limit_barrier_signature"
_QDES_LIMIT_BARRIER_OBSERVED_COUNT = "observed_sample_count"
_QDES_LIMIT_BARRIER_ABOVE_MARGIN_JOINT_COUNT = "above_margin_joint_count"
_QDES_LIMIT_BARRIER_ABOVE_MARGIN_SAMPLE_COUNT = "above_margin_sample_count"
_QDES_LIMIT_BARRIER_MAX_INTRUSION = "max_intrusion_depth_frac"
_QDES_LIMIT_BARRIER_VALUE_SUM = "barrier_value_sum"
_QDES_LIMIT_BARRIER_ACTIVE_COUNT = "reward_enabled_sample_count"


def _qdes_limit_barrier_values(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-env barrier value and the per-joint intrusion depth.

    For EVERY joint j of the 31-joint A3 articulation (no top-k, no subset)::

        d_j = min(q_des_j - lo_j, hi_j - q_des_j) / (hi_j - lo_j)     # normalized limit distance
        t_j = relu(margin_frac - d_j) / margin_frac                    # intrusion depth in [0, 1+]
        value = sum_j(1 - exp(-t_j^2))                                 # per-joint bounded tails, summed

    SUM aggregation (Franco 2026-07-21 裁定): mean 是稀释器——单关节违规被 ÷31,top-k 的
    "挑几个狠罚"其实就是在补这个稀释;去 top-k 的正当性来自 sum:违规几个关节就罚几份,
    每个关节的 tail 有界 (t→∞ 时 →1),满违规单关节 tail≈1,权重 -0.65 下每步约 -0.65×dt.
    Dense: charged on every control step in every phase (no strike/recovery gate) — 对标 Jiayi
    v14 的全程 barrier.  q_des is the SAME processed (affine + clamp) target the PD controller
    receives, and lo/hi are the SAME soft_joint_pos_limits the deploy-parity clamp uses.
    """

    if action_name != "joint_pos":
        raise ValueError("qdes_limit_barrier action_name must be exactly 'joint_pos'")
    if isinstance(margin_frac, bool):
        raise ValueError(
            "qdes_limit_barrier margin_frac must be finite and in (0, 0.5)"
        )
    margin_frac = float(margin_frac)
    if not math.isfinite(margin_frac) or not 0.0 < margin_frac < 0.5:
        raise ValueError(
            "qdes_limit_barrier margin_frac must be finite and in (0, 0.5)"
        )

    action = env.action_manager.get_term(action_name)
    processed = getattr(action, "processed_actions", None)
    if not torch.is_tensor(processed) or processed.ndim != 2:
        raise RuntimeError(
            "qdes_limit_barrier requires processed q_des targets from joint_pos"
        )

    asset = getattr(action, "_asset", None)
    data = getattr(asset, "data", None)
    runtime_names = list(
        getattr(data, "joint_names", getattr(asset, "joint_names", ()))
    )
    action_names = list(getattr(action, "_joint_names", ()))
    joint_count = len(runtime_names)
    if (
        joint_count != _QDES_LIMIT_BARRIER_JOINT_COUNT
        or len(set(runtime_names)) != joint_count
        or action_names != runtime_names
        or tuple(processed.shape)[1] != joint_count
    ):
        raise RuntimeError(
            "qdes_limit_barrier requires identity 31-joint A3 action/articulation order"
        )
    raw_joint_ids = getattr(action, "_joint_ids", slice(None))
    if isinstance(raw_joint_ids, slice):
        action_joint_ids = list(range(joint_count))[raw_joint_ids]
    else:
        if hasattr(raw_joint_ids, "tolist"):
            raw_joint_ids = raw_joint_ids.tolist()
        action_joint_ids = [int(value) for value in raw_joint_ids]
    if action_joint_ids != list(range(joint_count)):
        raise RuntimeError(
            "qdes_limit_barrier requires identity 31-joint A3 action/articulation order"
        )

    limits = getattr(data, "soft_joint_pos_limits", None)
    if not torch.is_tensor(limits):
        raise RuntimeError(
            "qdes_limit_barrier requires runtime articulation soft joint position limits"
        )
    if limits.ndim == 2 and tuple(limits.shape) == (joint_count, 2):
        lower = limits[:, 0]
        upper = limits[:, 1]
    elif (
        limits.ndim == 3
        and tuple(limits.shape)[1] == joint_count
        and tuple(limits.shape)[2] == 2
        and tuple(limits.shape)[0] in (1, tuple(processed.shape)[0])
    ):
        lower = limits[:, :, 0]
        upper = limits[:, :, 1]
    else:
        raise RuntimeError(
            "qdes_limit_barrier requires soft_joint_pos_limits shaped [31,2], [1,31,2], or [num_envs,31,2]"
        )
    span = upper - lower
    limits_valid = torch.all(
        torch.isfinite(lower) & torch.isfinite(upper) & span.gt(0.0)
    )
    if limits_valid.device.type == "cpu":
        if not bool(limits_valid):
            raise RuntimeError(
                "qdes_limit_barrier requires finite joint position limits with lo < hi"
            )
    else:
        torch._assert_async(limits_valid)

    distance = torch.minimum(processed - lower, upper - processed) / span
    intrusion = torch.relu(margin_frac - distance) / margin_frac
    value = torch.sum(1.0 - torch.exp(-torch.square(intrusion)), dim=-1)
    return value, intrusion


def _qdes_limit_barrier_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _QDES_LIMIT_BARRIER_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _QDES_LIMIT_BARRIER_OBSERVED_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_LIMIT_BARRIER_ABOVE_MARGIN_JOINT_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_LIMIT_BARRIER_ABOVE_MARGIN_SAMPLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_LIMIT_BARRIER_MAX_INTRUSION: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _QDES_LIMIT_BARRIER_VALUE_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _QDES_LIMIT_BARRIER_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
        }
        setattr(env, _QDES_LIMIT_BARRIER_ACTIVATION_ATTR, state)
    return state


def _record_qdes_limit_barrier_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    intrusion: torch.Tensor,
    *,
    barrier_active: bool,
    signature: tuple[str, float],
) -> None:
    """Book one simulator step, idempotently sharing the probe and real term."""

    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _qdes_limit_barrier_counter_state(env, values)
    already_observed = (
        token is not None
        and getattr(env, _QDES_LIMIT_BARRIER_OBSERVED_STEP_ATTR, None) == token
    )
    if already_observed:
        if getattr(env, _QDES_LIMIT_BARRIER_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError(
                "qdes limit barrier probe and RewardTerm used different parameters in one step"
            )
    else:
        above = intrusion.detach().gt(0.0)
        state[_QDES_LIMIT_BARRIER_OBSERVED_COUNT].add_(values.numel())
        state[_QDES_LIMIT_BARRIER_ABOVE_MARGIN_JOINT_COUNT].add_(
            above.sum(dtype=torch.long)
        )
        state[_QDES_LIMIT_BARRIER_ABOVE_MARGIN_SAMPLE_COUNT].add_(
            torch.any(above, dim=-1).sum(dtype=torch.long)
        )
        state[_QDES_LIMIT_BARRIER_MAX_INTRUSION].copy_(
            torch.maximum(
                state[_QDES_LIMIT_BARRIER_MAX_INTRUSION], intrusion.detach().max()
            )
        )
        state[_QDES_LIMIT_BARRIER_VALUE_SUM].add_(values.detach().sum())
        if token is not None:
            setattr(env, _QDES_LIMIT_BARRIER_OBSERVED_STEP_ATTR, token)
            setattr(env, _QDES_LIMIT_BARRIER_SIGNATURE_ATTR, signature)
    if barrier_active and (
        token is None
        or getattr(env, _QDES_LIMIT_BARRIER_ACTIVE_STEP_ATTR, None) != token
    ):
        state[_QDES_LIMIT_BARRIER_ACTIVE_COUNT].add_(values.numel())
        if token is not None:
            setattr(env, _QDES_LIMIT_BARRIER_ACTIVE_STEP_ATTR, token)


def qdes_limit_barrier_probe(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
) -> torch.Tensor:
    """Measure the q_des limit barrier while contributing exact zero reward.

    Same design provenance as :func:`qdes_limit_barrier` — Jiayi V14 全关节 top-k qdes barrier
    思想,去 top-k 重做(Franco 指示),-0.65/margin 0.08 取自 v14 起点.  人话:零权重探针,
    只记账(贴限位的关节数、最大侵入深度),不改任何奖励。
    """

    values, intrusion = _qdes_limit_barrier_values(env, action_name, margin_frac)
    _record_qdes_limit_barrier_activation(
        env,
        values,
        intrusion,
        barrier_active=False,
        signature=(action_name, float(margin_frac)),
    )
    return torch.zeros_like(values)


def qdes_limit_barrier(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
) -> torch.Tensor:
    """Penalize processed q_des targets inside the margin band next to a position limit.

    Jiayi V14 全关节 top-k qdes barrier 思想,去 top-k 重做(Franco 指示):全部 31 个关节直接罚,
    不做 top-k、不做子集;-0.65/margin 0.08 取自 v14 起点.  Per-joint bounded tails are SUMMED,
    not averaged (Franco: mean 把单关节违规 ÷31 稀释掉;sum 让违规几个关节就罚几份,这也是
    去 top-k 的正当性所在;满违规单关节 tail≈1 → 每步约 -0.65×dt).  Dense — charged on every
    control step in every phase, matching V14's whole-episode barrier.  人话:目标角贴到限位
    边缘就开始扣钱,越贴越狠,几个关节贴就扣几份,全身关节一视同仁,全程有效。
    """

    values, intrusion = _qdes_limit_barrier_values(env, action_name, margin_frac)
    _record_qdes_limit_barrier_activation(
        env,
        values,
        intrusion,
        barrier_active=True,
        signature=(action_name, float(margin_frac)),
    )
    return values


def consume_qdes_limit_barrier_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update's q_des limit-barrier ledger.

    The reset MUST run under ``torch.inference_mode()``: the counters are created during rollout
    collection (inference mode), and an in-place ``zero_()`` on an inference tensor outside
    InferenceMode raises at the first PPO log flush.
    """

    action = env.action_manager.get_term("joint_pos")
    template = action.processed_actions[:, 0]
    state = _qdes_limit_barrier_counter_state(env, template)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


# Wave-B lower-body diagnostics --------------------------------------------------------------- #
#
# Both mechanisms below are intentionally default-off and share one phase gate: a bounded
# pre-contact support interval plus the existing reset/wrap/reveal-aware post-strike clock.  The
# clock is armed by the phase-aligned strike opportunity, not by racket contact or task success,
# so a failed attempt remains in the sample instead of being silently selected away.
_A3_RUNTIME_JOINT_ORDER = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
_LOWER_BODY_LEG_JOINT_NAMES = frozenset(_A3_RUNTIME_JOINT_ORDER[-12:])
_LOWER_BODY_FOOT_BODY_NAMES = ("left_ankle_roll_Link", "right_ankle_roll_Link")

_LOWER_BODY_POSE_COUNTER_ATTR = "_hope_lower_body_pose_activation_counters"
_LOWER_BODY_POSE_OBSERVED_STEP_ATTR = "_hope_lower_body_pose_observed_step"
_LOWER_BODY_POSE_ACTIVE_STEP_ATTR = "_hope_lower_body_pose_active_step"
_LOWER_BODY_POSE_SIGNATURE_ATTR = "_hope_lower_body_pose_signature"
_LOWER_BODY_BUNDLE_COUNTER_ATTR = "_hope_lower_body_bundle_activation_counters"
_LOWER_BODY_BUNDLE_OBSERVED_STEP_ATTR = "_hope_lower_body_bundle_observed_step"
_LOWER_BODY_BUNDLE_ACTIVE_STEP_ATTR = "_hope_lower_body_bundle_active_step"
_LOWER_BODY_BUNDLE_SIGNATURE_ATTR = "_hope_lower_body_bundle_signature"


def _finite_scalar(value, *, name: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return parsed


def _lower_body_support_gate(
    env: ManagerBasedRLEnv,
    racket_command_name: str,
    motion_command_name: str,
    support_pre_s: float,
    support_post_s: float,
) -> torch.Tensor:
    """Inclusive support gate without success-conditioned sample selection."""

    if racket_command_name != "racket_target" or motion_command_name != "motion":
        raise ValueError(
            "lower-body support rewards require racket_target and motion command names"
        )
    support_pre_s = _finite_scalar(
        support_pre_s, name="lower-body support_pre_s", nonnegative=True
    )
    support_post_s = _finite_scalar(
        support_post_s, name="lower-body support_post_s", nonnegative=True
    )
    racket = _cmd(env, racket_command_name)
    motion = env.command_manager.get_term(motion_command_name)
    time_to_strike = getattr(racket, "time_to_strike", None)
    pre_strike = getattr(racket, "pre_strike", None)
    in_hold = getattr(motion, "in_hold", None)
    clock = getattr(racket, "post_strike_age_and_same_attempt", None)
    if (
        not torch.is_tensor(time_to_strike)
        or not torch.is_tensor(pre_strike)
        or pre_strike.dtype != torch.bool
        or not torch.is_tensor(in_hold)
        or in_hold.dtype != torch.bool
        or not callable(clock)
    ):
        raise RuntimeError(
            "lower-body support rewards require live TTS/pre-strike, motion hold, and "
            "same-attempt post-strike clock tensors"
        )
    age_s, same_attempt = clock()
    expected_shape = tuple(time_to_strike.shape)
    if (
        time_to_strike.ndim != 1
        or tuple(pre_strike.shape) != expected_shape
        or tuple(in_hold.shape) != expected_shape
        or not torch.is_tensor(age_s)
        or tuple(age_s.shape) != expected_shape
        or not torch.is_tensor(same_attempt)
        or tuple(same_attempt.shape) != expected_shape
        or same_attempt.dtype != torch.bool
    ):
        raise RuntimeError("lower-body support reward phase tensors must be aligned per environment")

    pre_support = (
        (~in_hold)
        & pre_strike
        & time_to_strike.ge(0.0)
        & time_to_strike.le(support_pre_s)
    )
    post_support = same_attempt & age_s.ge(0.0) & age_s.le(support_post_s)
    return pre_support | post_support


def _lower_body_runtime_tensors(
    env: ManagerBasedRLEnv,
    motion_command_name: str,
    *,
    require_motion_reference: bool,
) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[int]]:
    """Resolve the exact current-main A3 reference/runtime order or fail closed."""

    if motion_command_name != "motion":
        raise ValueError("lower-body rewards require motion_command_name='motion'")
    robot = env.scene["robot"]
    data = robot.data
    runtime_names = tuple(
        str(name)
        for name in getattr(data, "joint_names", getattr(robot, "joint_names", ()))
    )
    # The live Isaac articulation enumerates joints breadth-first, not in the
    # deploy-runtime order; require the exact A3 name set (31, unique) and select
    # leg indices by name below, matching processed_qdes_slew_hinge's discipline.
    if (
        len(runtime_names) != len(_A3_RUNTIME_JOINT_ORDER)
        or len(set(runtime_names)) != len(runtime_names)
        or set(runtime_names) != set(_A3_RUNTIME_JOINT_ORDER)
    ):
        raise RuntimeError(
            "lower-body rewards require the exact 31-joint A3 runtime articulation order"
        )
    q = getattr(data, "joint_pos", None)
    qd = getattr(data, "joint_vel", None)
    default_q = getattr(data, "default_joint_pos", None)
    motion = env.command_manager.get_term(motion_command_name)
    if getattr(motion, "robot", None) is not robot:
        raise RuntimeError("lower-body rewards require motion and reward to reference the same robot")
    reference = None
    expected_width = len(_A3_RUNTIME_JOINT_ORDER)
    if (
        not torch.is_tensor(q)
        or q.ndim != 2
        or q.shape[1] != expected_width
        or not torch.is_tensor(qd)
        or tuple(qd.shape) != tuple(q.shape)
        or not torch.is_tensor(default_q)
        or tuple(default_q.shape) not in (tuple(q.shape), (1, expected_width))
    ):
        raise RuntimeError(
            "lower-body rewards require aligned [env,31] runtime/default tensors"
        )
    if require_motion_reference:
        reference = getattr(motion, "joint_pos", None)
        loaded_reference = getattr(getattr(motion, "motion", None), "joint_pos", None)
        if (
            not torch.is_tensor(reference)
            or tuple(reference.shape) != tuple(q.shape)
            or not torch.is_tensor(loaded_reference)
            or loaded_reference.ndim != 2
            or loaded_reference.shape[1] != expected_width
        ):
            raise RuntimeError(
                "lower-body pose imitation requires an aligned [env,31] current motion "
                "tensor and a 31-column loaded motion reference"
            )
    leg_ids = [
        index for index, name in enumerate(runtime_names) if name in _LOWER_BODY_LEG_JOINT_NAMES
    ]
    if (
        len(leg_ids) != 12
        or {runtime_names[index] for index in leg_ids} != _LOWER_BODY_LEG_JOINT_NAMES
    ):
        raise RuntimeError("lower-body rewards require exactly 12 leg joints")
    return robot, q, qd, default_q, reference, leg_ids


def _lower_body_pose_values(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    std: float = 0.35,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bounded 12-leg pose imitation and diagnostic raw magnitudes."""

    std = _finite_scalar(std, name="lower_body_pose_imitation std", positive=True)
    _, q, _, default_q, reference, leg_ids = _lower_body_runtime_tensors(
        env, motion_command_name, require_motion_reference=True
    )
    eligible = _lower_body_support_gate(
        env,
        racket_command_name,
        motion_command_name,
        support_pre_s,
        support_post_s,
    )
    delta = q[:, leg_ids] - reference[:, leg_ids]
    error_sq_mean = torch.mean(torch.square(delta), dim=-1)
    kernel = torch.exp(-error_sq_mean / (std * std))
    value = torch.where(eligible, kernel, torch.zeros_like(kernel))
    reference_motion_l1 = torch.mean(
        torch.abs(reference[:, leg_ids] - default_q[:, leg_ids]), dim=-1
    )
    return value, eligible, torch.mean(torch.abs(delta), dim=-1), reference_motion_l1


def _lower_body_pose_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _LOWER_BODY_POSE_COUNTER_ATTR, None)
    if state is None:
        state = {
            "observed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "support_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "reward_enabled_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "gated_kernel_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_joint_abs_error_mean_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_reference_motion_l1_mean_sum": torch.zeros((), dtype=template.dtype, device=template.device),
        }
        setattr(env, _LOWER_BODY_POSE_COUNTER_ATTR, state)
    return state


def _record_lower_body_pose_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    eligible: torch.Tensor,
    joint_error: torch.Tensor,
    reference_motion: torch.Tensor,
    *,
    reward_active: bool,
    signature: tuple,
) -> None:
    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _lower_body_pose_counter_state(env, values)
    already = token is not None and getattr(env, _LOWER_BODY_POSE_OBSERVED_STEP_ATTR, None) == token
    if already:
        if getattr(env, _LOWER_BODY_POSE_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError("lower-body pose probe and reward used different parameters in one step")
    else:
        mask = eligible.detach()
        state["observed_sample_count"].add_(values.numel())
        state["support_eligible_sample_count"].add_(mask.sum(dtype=torch.long))
        state["gated_kernel_sum"].add_(values.detach().sum())
        state["gated_joint_abs_error_mean_sum"].add_((joint_error.detach() * mask).sum())
        state["gated_reference_motion_l1_mean_sum"].add_((reference_motion.detach() * mask).sum())
        if token is not None:
            setattr(env, _LOWER_BODY_POSE_OBSERVED_STEP_ATTR, token)
            setattr(env, _LOWER_BODY_POSE_SIGNATURE_ATTR, signature)
    if reward_active and (
        token is None or getattr(env, _LOWER_BODY_POSE_ACTIVE_STEP_ATTR, None) != token
    ):
        state["reward_enabled_eligible_sample_count"].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        if token is not None:
            setattr(env, _LOWER_BODY_POSE_ACTIVE_STEP_ATTR, token)


def lower_body_pose_imitation_probe(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    std: float = 0.35,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> torch.Tensor:
    values, eligible, joint_error, reference_motion = _lower_body_pose_values(
        env, racket_command_name, motion_command_name, std, support_pre_s, support_post_s
    )
    _record_lower_body_pose_activation(
        env,
        values,
        eligible,
        joint_error,
        reference_motion,
        reward_active=False,
        signature=(
            racket_command_name,
            motion_command_name,
            float(std),
            float(support_pre_s),
            float(support_post_s),
        ),
    )
    return torch.zeros_like(values)


def lower_body_pose_imitation(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    std: float = 0.35,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> torch.Tensor:
    """Positive v4rg 12-leg pose imitation only in the same swing's support window."""

    values, eligible, joint_error, reference_motion = _lower_body_pose_values(
        env, racket_command_name, motion_command_name, std, support_pre_s, support_post_s
    )
    _record_lower_body_pose_activation(
        env,
        values,
        eligible,
        joint_error,
        reference_motion,
        reward_active=True,
        signature=(
            racket_command_name,
            motion_command_name,
            float(std),
            float(support_pre_s),
            float(support_post_s),
        ),
    )
    return values


def _lower_body_stability_bundle_values(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    min_stance_width_m: float = 0.22,
    stance_scale_m: float = 0.05,
    leg_velocity_margin_radps: float = 1.0,
    leg_velocity_scale_radps: float = 0.5,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a bounded, reference-free physical support bundle.

    This deliberately does not repeat motion-reference foot orientation, slip, base-upright, or
    dense all-joint velocity costs.  It charges only (1) collapse/crossover below an absolute
    physical support width and (2) a 12-leg realized-qdot tail above a free margin.
    """

    min_stance_width_m = _finite_scalar(
        min_stance_width_m, name="lower_body_stability min_stance_width_m", positive=True
    )
    stance_scale_m = _finite_scalar(
        stance_scale_m, name="lower_body_stability stance_scale_m", positive=True
    )
    leg_velocity_margin_radps = _finite_scalar(
        leg_velocity_margin_radps,
        name="lower_body_stability leg_velocity_margin_radps",
        nonnegative=True,
    )
    leg_velocity_scale_radps = _finite_scalar(
        leg_velocity_scale_radps,
        name="lower_body_stability leg_velocity_scale_radps",
        positive=True,
    )
    robot, q, qd, _, _, leg_ids = _lower_body_runtime_tensors(
        env, motion_command_name, require_motion_reference=False
    )
    eligible = _lower_body_support_gate(
        env,
        racket_command_name,
        motion_command_name,
        support_pre_s,
        support_post_s,
    )

    body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
    if (
        len(body_names) == 0
        or len(set(body_names)) != len(body_names)
        or any(name not in body_names for name in _LOWER_BODY_FOOT_BODY_NAMES)
    ):
        raise RuntimeError("lower-body stability requires unique left/right A3 ankle-roll bodies")
    foot_ids = [body_names.index(name) for name in _LOWER_BODY_FOOT_BODY_NAMES]
    body_pos_w = getattr(robot.data, "body_pos_w", None)
    if (
        not torch.is_tensor(body_pos_w)
        or body_pos_w.ndim != 3
        or body_pos_w.shape[0] != q.shape[0]
        or body_pos_w.shape[1] != len(body_names)
        or body_pos_w.shape[2] != 3
    ):
        raise RuntimeError("lower-body stability requires runtime body_pos_w shaped [env,body,3]")
    base_quat = getattr(_cmd(env, racket_command_name), "base_quat_w", None)
    if not torch.is_tensor(base_quat) or tuple(base_quat.shape) != (q.shape[0], 4):
        raise RuntimeError("lower-body stability requires per-environment base quaternion wxyz")
    quat = base_quat / torch.norm(base_quat, dim=-1, keepdim=True).clamp(min=1.0e-12)
    w, x, y, z = quat.unbind(dim=-1)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    left_minus_right = body_pos_w[:, foot_ids[0], :2] - body_pos_w[:, foot_ids[1], :2]
    signed_width = -sin_yaw * left_minus_right[:, 0] + cos_yaw * left_minus_right[:, 1]
    stance_excess = torch.relu(min_stance_width_m - signed_width) / stance_scale_m
    stance_tail = 1.0 - torch.exp(-torch.square(stance_excess))

    leg_velocity_excess = torch.relu(
        torch.abs(qd[:, leg_ids]) - leg_velocity_margin_radps
    ) / leg_velocity_scale_radps
    leg_velocity_tail = torch.mean(
        1.0 - torch.exp(-torch.square(leg_velocity_excess)), dim=-1
    )
    raw_bundle = (stance_tail + leg_velocity_tail) / 2.0
    value = torch.where(eligible, raw_bundle, torch.zeros_like(raw_bundle))
    return value, eligible, stance_tail, leg_velocity_tail, signed_width


def _lower_body_bundle_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _LOWER_BODY_BUNDLE_COUNTER_ATTR, None)
    if state is None:
        state = {
            "observed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "support_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "reward_enabled_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "narrow_or_crossed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "gated_bundle_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_stance_tail_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_leg_velocity_tail_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_signed_stance_width_m_sum": torch.zeros((), dtype=template.dtype, device=template.device),
        }
        setattr(env, _LOWER_BODY_BUNDLE_COUNTER_ATTR, state)
    return state


def _record_lower_body_bundle_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    eligible: torch.Tensor,
    stance_tail: torch.Tensor,
    velocity_tail: torch.Tensor,
    signed_width: torch.Tensor,
    *,
    min_stance_width_m: float,
    reward_active: bool,
    signature: tuple,
) -> None:
    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _lower_body_bundle_counter_state(env, values)
    already = token is not None and getattr(env, _LOWER_BODY_BUNDLE_OBSERVED_STEP_ATTR, None) == token
    if already:
        if getattr(env, _LOWER_BODY_BUNDLE_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError("lower-body stability probe and reward used different parameters in one step")
    else:
        mask = eligible.detach()
        state["observed_sample_count"].add_(values.numel())
        state["support_eligible_sample_count"].add_(mask.sum(dtype=torch.long))
        state["narrow_or_crossed_sample_count"].add_(
            (mask & signed_width.detach().lt(min_stance_width_m)).sum(dtype=torch.long)
        )
        state["gated_bundle_sum"].add_(values.detach().sum())
        state["gated_stance_tail_sum"].add_((stance_tail.detach() * mask).sum())
        state["gated_leg_velocity_tail_sum"].add_((velocity_tail.detach() * mask).sum())
        state["gated_signed_stance_width_m_sum"].add_((signed_width.detach() * mask).sum())
        if token is not None:
            setattr(env, _LOWER_BODY_BUNDLE_OBSERVED_STEP_ATTR, token)
            setattr(env, _LOWER_BODY_BUNDLE_SIGNATURE_ATTR, signature)
    if reward_active and (
        token is None or getattr(env, _LOWER_BODY_BUNDLE_ACTIVE_STEP_ATTR, None) != token
    ):
        state["reward_enabled_eligible_sample_count"].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        if token is not None:
            setattr(env, _LOWER_BODY_BUNDLE_ACTIVE_STEP_ATTR, token)


def lower_body_stability_bundle_probe(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    min_stance_width_m: float = 0.22,
    stance_scale_m: float = 0.05,
    leg_velocity_margin_radps: float = 1.0,
    leg_velocity_scale_radps: float = 0.5,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> torch.Tensor:
    args = (
        racket_command_name,
        motion_command_name,
        min_stance_width_m,
        stance_scale_m,
        leg_velocity_margin_radps,
        leg_velocity_scale_radps,
        support_pre_s,
        support_post_s,
    )
    values, eligible, stance, velocity, width = _lower_body_stability_bundle_values(
        env, *args
    )
    signature = tuple(float(value) if index >= 2 else value for index, value in enumerate(args))
    _record_lower_body_bundle_activation(
        env,
        values,
        eligible,
        stance,
        velocity,
        width,
        min_stance_width_m=float(min_stance_width_m),
        reward_active=False,
        signature=signature,
    )
    return torch.zeros_like(values)


def lower_body_stability_bundle(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    min_stance_width_m: float = 0.22,
    stance_scale_m: float = 0.05,
    leg_velocity_margin_radps: float = 1.0,
    leg_velocity_scale_radps: float = 0.5,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> torch.Tensor:
    """Negative-weight B2 bundle: fixed support-width floor plus realized leg-qdot tail."""

    args = (
        racket_command_name,
        motion_command_name,
        min_stance_width_m,
        stance_scale_m,
        leg_velocity_margin_radps,
        leg_velocity_scale_radps,
        support_pre_s,
        support_post_s,
    )
    values, eligible, stance, velocity, width = _lower_body_stability_bundle_values(
        env, *args
    )
    signature = tuple(float(value) if index >= 2 else value for index, value in enumerate(args))
    _record_lower_body_bundle_activation(
        env,
        values,
        eligible,
        stance,
        velocity,
        width,
        min_stance_width_m=float(min_stance_width_m),
        reward_active=True,
        signature=signature,
    )
    return values


def consume_lower_body_wave_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot/reset both Wave-B ledgers exactly once per PPO update."""

    template = env.scene["robot"].data.joint_pos[:, 0]
    pose = _lower_body_pose_counter_state(env, template)
    bundle = _lower_body_bundle_counter_state(env, template)
    snapshot = {
        **{f"pose/{name}": value.detach().clone() for name, value in pose.items()},
        **{f"bundle/{name}": value.detach().clone() for name, value in bundle.items()},
    }
    # inference_mode (not no_grad): the counters are inference tensors created inside
    # env.step, and in-place resets outside InferenceMode are rejected by torch.
    with torch.inference_mode():
        for value in (*pose.values(), *bundle.values()):
            value.zero_()
    return snapshot


# S1 post-swing settle debts ------------------------------------------------------------------- #
#
# Idea credit: Jiayi's V13 post-swing debts (an unmerged branch).  This is a clean re-derivation on
# main, NOT a copy: the five debt channels keep the V13 intent (quiet the base, stand upright and
# tall, plant the feet after the swing), while every margin/scale below is re-fixed by this repo's
# conventions — the V13 branch's unvalidated numbers are deliberately not reused.  The activation
# window deliberately REUSES the processed_qdes_slew_hinge recovery-window mechanism (the racket
# command's reset/wrap/reveal-aware same-attempt post-strike clock) instead of inventing a second
# clock, so both recovery mechanisms agree on what "the same swing's 0.20..1.55 s" means; a reset
# invalidates the attempt through that clock and the term returns exact zero.
# 人话:挥完拍 0.2–1.55 秒内要"稳稳站好"——身体别乱晃、别歪、别蹲矮、脚别滑;每项有免费额度,
# 超出的部分按有界"债务尾巴"扣钱,重置后的无效历史一分不扣。
_POST_SWING_SETTLE_FOOT_BODY_NAMES = _LOWER_BODY_FOOT_BODY_NAMES
# A3 stand-keyframe pelvis height: robots/agibot_a3.py init_state pos z = 1.0684 m (itself read
# from the vendor MuJoCo stand keyframe).  Kept as an explicit reward parameter so the training
# contract records the exact nominal height the run trained with.
_POST_SWING_SETTLE_NOMINAL_ROOT_Z_M = 1.0684
_POST_SWING_SETTLE_COUNTER_ATTR = "_hope_post_swing_settle_activation_counters"
_POST_SWING_SETTLE_OBSERVED_STEP_ATTR = "_hope_post_swing_settle_observed_step"
_POST_SWING_SETTLE_ACTIVE_STEP_ATTR = "_hope_post_swing_settle_active_step"
_POST_SWING_SETTLE_SIGNATURE_ATTR = "_hope_post_swing_settle_signature"
_POST_SWING_SETTLE_COMPONENT_NAMES = (
    "base_quiet_lin",
    "base_quiet_ang",
    "tilt_debt",
    "root_height_debt",
    "settle_foot_slip",
)


def _post_swing_settle_tail(
    magnitude: torch.Tensor, margin: float, scale: float
) -> torch.Tensor:
    """Bounded debt tail ``1 - exp(-square(relu(x - margin) / scale))`` in [0, 1)."""

    return 1.0 - torch.exp(-torch.square(torch.relu(magnitude - margin) / scale))


def _post_swing_settle_debt_values(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    base_lin_margin_mps: float = 0.30,
    base_lin_scale_mps: float = 0.20,
    base_ang_margin_radps: float = 0.50,
    base_ang_scale_radps: float = 0.30,
    tilt_margin_rad: float = 0.10,
    tilt_scale_rad: float = 0.10,
    nominal_root_z_m: float = _POST_SWING_SETTLE_NOMINAL_ROOT_Z_M,
    root_height_deadband_m: float = 0.05,
    root_height_scale_m: float = 0.05,
    foot_slip_margin_mps: float = 0.05,
    foot_slip_scale_mps: float = 0.10,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the gated five-debt settle value, eligibility, and per-component tails [env, 5].

    Component magnitudes (each turned into a bounded tail by ``_post_swing_settle_tail``):
    ``||root_lin_vel_w||``, ``||root_ang_vel_w||``, ``asin(||projected_gravity_b_xy||)`` (the same
    tilt quantity strike/prestrike upright terms read as a norm), ``relu(nominal_root_z -
    deadband - root_z)`` (deadband is the margin), and the two ankle-roll links' mean horizontal
    speed (the same bodies/tensors the foot-slip metrics read).  Samples outside the same-attempt
    recovery window — including everything after a reset until the next armed strike — are exact 0.
    """

    if racket_command_name != "racket_target":
        raise ValueError(
            "post_swing_settle_debt racket_command_name must be exactly 'racket_target'"
        )
    if motion_command_name != "motion":
        raise ValueError(
            "post_swing_settle_debt motion_command_name must be exactly 'motion'"
        )
    base_lin_margin_mps = _finite_scalar(
        base_lin_margin_mps, name="post_swing_settle_debt base_lin_margin_mps", nonnegative=True
    )
    base_lin_scale_mps = _finite_scalar(
        base_lin_scale_mps, name="post_swing_settle_debt base_lin_scale_mps", positive=True
    )
    base_ang_margin_radps = _finite_scalar(
        base_ang_margin_radps, name="post_swing_settle_debt base_ang_margin_radps", nonnegative=True
    )
    base_ang_scale_radps = _finite_scalar(
        base_ang_scale_radps, name="post_swing_settle_debt base_ang_scale_radps", positive=True
    )
    tilt_margin_rad = _finite_scalar(
        tilt_margin_rad, name="post_swing_settle_debt tilt_margin_rad", nonnegative=True
    )
    tilt_scale_rad = _finite_scalar(
        tilt_scale_rad, name="post_swing_settle_debt tilt_scale_rad", positive=True
    )
    nominal_root_z_m = _finite_scalar(
        nominal_root_z_m, name="post_swing_settle_debt nominal_root_z_m", positive=True
    )
    root_height_deadband_m = _finite_scalar(
        root_height_deadband_m,
        name="post_swing_settle_debt root_height_deadband_m",
        nonnegative=True,
    )
    root_height_scale_m = _finite_scalar(
        root_height_scale_m, name="post_swing_settle_debt root_height_scale_m", positive=True
    )
    foot_slip_margin_mps = _finite_scalar(
        foot_slip_margin_mps, name="post_swing_settle_debt foot_slip_margin_mps", nonnegative=True
    )
    foot_slip_scale_mps = _finite_scalar(
        foot_slip_scale_mps, name="post_swing_settle_debt foot_slip_scale_mps", positive=True
    )
    if isinstance(recovery_start_s, bool) or isinstance(recovery_end_s, bool):
        raise ValueError(
            "post_swing_settle_debt recovery window must be finite and satisfy 0 <= start < end"
        )
    recovery_start_s = float(recovery_start_s)
    recovery_end_s = float(recovery_end_s)
    if (
        not math.isfinite(recovery_start_s)
        or not math.isfinite(recovery_end_s)
        or recovery_start_s < 0.0
        or recovery_start_s >= recovery_end_s
    ):
        raise ValueError(
            "post_swing_settle_debt recovery window must be finite and satisfy 0 <= start < end"
        )

    robot = env.scene["robot"]
    data = robot.data
    motion = env.command_manager.get_term(motion_command_name)
    if getattr(motion, "robot", None) is not robot:
        raise RuntimeError(
            "post_swing_settle_debt requires motion and reward to reference the same robot"
        )
    root_lin_vel = getattr(data, "root_lin_vel_w", None)
    root_ang_vel = getattr(data, "root_ang_vel_w", None)
    root_pos = getattr(data, "root_pos_w", None)
    projected_gravity = getattr(data, "projected_gravity_b", None)
    if (
        not torch.is_tensor(root_lin_vel)
        or root_lin_vel.ndim != 2
        or root_lin_vel.shape[1] != 3
        or not torch.is_tensor(root_ang_vel)
        or tuple(root_ang_vel.shape) != tuple(root_lin_vel.shape)
        or not torch.is_tensor(root_pos)
        or tuple(root_pos.shape) != tuple(root_lin_vel.shape)
        or not torch.is_tensor(projected_gravity)
        or tuple(projected_gravity.shape) != tuple(root_lin_vel.shape)
    ):
        raise RuntimeError(
            "post_swing_settle_debt requires aligned [env,3] root velocity/position/"
            "projected-gravity tensors"
        )
    num_envs = root_lin_vel.shape[0]

    body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
    if (
        len(body_names) == 0
        or len(set(body_names)) != len(body_names)
        or any(name not in body_names for name in _POST_SWING_SETTLE_FOOT_BODY_NAMES)
    ):
        raise RuntimeError(
            "post_swing_settle_debt requires unique left/right A3 ankle-roll bodies"
        )
    foot_ids = [body_names.index(name) for name in _POST_SWING_SETTLE_FOOT_BODY_NAMES]
    body_lin_vel_w = getattr(data, "body_lin_vel_w", None)
    if (
        not torch.is_tensor(body_lin_vel_w)
        or body_lin_vel_w.ndim != 3
        or body_lin_vel_w.shape[0] != num_envs
        or body_lin_vel_w.shape[1] != len(body_names)
        or body_lin_vel_w.shape[2] != 3
    ):
        raise RuntimeError(
            "post_swing_settle_debt requires runtime body_lin_vel_w shaped [env,body,3]"
        )

    base_lin_speed = torch.norm(root_lin_vel, dim=-1)
    base_ang_speed = torch.norm(root_ang_vel, dim=-1)
    tilt_rad = torch.asin(
        torch.norm(projected_gravity[:, :2], dim=-1).clamp(max=1.0)
    )
    root_height_debt_m = torch.relu(
        nominal_root_z_m - root_height_deadband_m - root_pos[:, 2]
    )
    foot_speed_xy = torch.norm(body_lin_vel_w[:, foot_ids, :2], dim=-1).mean(dim=-1)
    tails = torch.stack(
        (
            _post_swing_settle_tail(base_lin_speed, base_lin_margin_mps, base_lin_scale_mps),
            _post_swing_settle_tail(base_ang_speed, base_ang_margin_radps, base_ang_scale_radps),
            _post_swing_settle_tail(tilt_rad, tilt_margin_rad, tilt_scale_rad),
            # The deadband already played the margin's role above, so the tail's margin is 0.
            _post_swing_settle_tail(root_height_debt_m, 0.0, root_height_scale_m),
            _post_swing_settle_tail(foot_speed_xy, foot_slip_margin_mps, foot_slip_scale_mps),
        ),
        dim=-1,
    )
    raw_value = torch.mean(tails, dim=-1)

    # Same-attempt recovery-window gate: identical mechanism (and default window) to
    # processed_qdes_slew_hinge — one shared clock, never a second bookkeeping of "post swing".
    cmd = _cmd(env, racket_command_name)
    clock = getattr(cmd, "post_strike_age_and_same_attempt", None)
    if not callable(clock):
        raise RuntimeError(
            "post_swing_settle_debt requires racket_target's same-attempt post-strike clock"
        )
    age_s, same_attempt = clock()
    if (
        not torch.is_tensor(age_s)
        or not torch.is_tensor(same_attempt)
        or tuple(age_s.shape) != (num_envs,)
        or tuple(same_attempt.shape) != (num_envs,)
        or same_attempt.dtype != torch.bool
    ):
        raise RuntimeError(
            "post_swing_settle_debt received an invalid same-attempt post-strike clock"
        )
    eligible = (
        same_attempt
        & age_s.ge(recovery_start_s)
        & age_s.le(recovery_end_s)
    )
    value = torch.where(eligible, raw_value, torch.zeros_like(raw_value))
    return value, eligible, tails


def _post_swing_settle_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _POST_SWING_SETTLE_COUNTER_ATTR, None)
    if state is None:
        state = {
            "observed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "recovery_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "reward_enabled_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "gated_debt_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            **{
                f"gated_{name}_tail_sum": torch.zeros(
                    (), dtype=template.dtype, device=template.device
                )
                for name in _POST_SWING_SETTLE_COMPONENT_NAMES
            },
        }
        setattr(env, _POST_SWING_SETTLE_COUNTER_ATTR, state)
    return state


def _record_post_swing_settle_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    eligible: torch.Tensor,
    tails: torch.Tensor,
    *,
    reward_active: bool,
    signature: tuple,
) -> None:
    """Book one simulator step, idempotently sharing the probe and real term."""

    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _post_swing_settle_counter_state(env, values)
    already = (
        token is not None
        and getattr(env, _POST_SWING_SETTLE_OBSERVED_STEP_ATTR, None) == token
    )
    if already:
        if getattr(env, _POST_SWING_SETTLE_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError(
                "post-swing settle probe and RewardTerm used different parameters in one step"
            )
    else:
        mask = eligible.detach()
        state["observed_sample_count"].add_(values.numel())
        state["recovery_eligible_sample_count"].add_(mask.sum(dtype=torch.long))
        state["gated_debt_sum"].add_(values.detach().sum())
        gated_tails = tails.detach() * mask.unsqueeze(-1)
        for index, name in enumerate(_POST_SWING_SETTLE_COMPONENT_NAMES):
            state[f"gated_{name}_tail_sum"].add_(gated_tails[:, index].sum())
        if token is not None:
            setattr(env, _POST_SWING_SETTLE_OBSERVED_STEP_ATTR, token)
            setattr(env, _POST_SWING_SETTLE_SIGNATURE_ATTR, signature)
    if reward_active and (
        token is None or getattr(env, _POST_SWING_SETTLE_ACTIVE_STEP_ATTR, None) != token
    ):
        state["reward_enabled_eligible_sample_count"].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        if token is not None:
            setattr(env, _POST_SWING_SETTLE_ACTIVE_STEP_ATTR, token)


def _post_swing_settle_signature(args: tuple) -> tuple:
    return tuple(
        float(value) if index >= 2 else value for index, value in enumerate(args)
    )


def post_swing_settle_debt_probe(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    base_lin_margin_mps: float = 0.30,
    base_lin_scale_mps: float = 0.20,
    base_ang_margin_radps: float = 0.50,
    base_ang_scale_radps: float = 0.30,
    tilt_margin_rad: float = 0.10,
    tilt_scale_rad: float = 0.10,
    nominal_root_z_m: float = _POST_SWING_SETTLE_NOMINAL_ROOT_Z_M,
    root_height_deadband_m: float = 0.05,
    root_height_scale_m: float = 0.05,
    foot_slip_margin_mps: float = 0.05,
    foot_slip_scale_mps: float = 0.10,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> torch.Tensor:
    """Measure the settle debts while contributing exact zero reward."""

    args = (
        racket_command_name,
        motion_command_name,
        base_lin_margin_mps,
        base_lin_scale_mps,
        base_ang_margin_radps,
        base_ang_scale_radps,
        tilt_margin_rad,
        tilt_scale_rad,
        nominal_root_z_m,
        root_height_deadband_m,
        root_height_scale_m,
        foot_slip_margin_mps,
        foot_slip_scale_mps,
        recovery_start_s,
        recovery_end_s,
    )
    values, eligible, tails = _post_swing_settle_debt_values(env, *args)
    _record_post_swing_settle_activation(
        env,
        values,
        eligible,
        tails,
        reward_active=False,
        signature=_post_swing_settle_signature(args),
    )
    return torch.zeros_like(values)


def post_swing_settle_debt(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    base_lin_margin_mps: float = 0.30,
    base_lin_scale_mps: float = 0.20,
    base_ang_margin_radps: float = 0.50,
    base_ang_scale_radps: float = 0.30,
    tilt_margin_rad: float = 0.10,
    tilt_scale_rad: float = 0.10,
    nominal_root_z_m: float = _POST_SWING_SETTLE_NOMINAL_ROOT_Z_M,
    root_height_deadband_m: float = 0.05,
    root_height_scale_m: float = 0.05,
    foot_slip_margin_mps: float = 0.05,
    foot_slip_scale_mps: float = 0.10,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> torch.Tensor:
    """Negative-weight S1 bundle: five bounded settle debts in the same-swing recovery window."""

    args = (
        racket_command_name,
        motion_command_name,
        base_lin_margin_mps,
        base_lin_scale_mps,
        base_ang_margin_radps,
        base_ang_scale_radps,
        tilt_margin_rad,
        tilt_scale_rad,
        nominal_root_z_m,
        root_height_deadband_m,
        root_height_scale_m,
        foot_slip_margin_mps,
        foot_slip_scale_mps,
        recovery_start_s,
        recovery_end_s,
    )
    values, eligible, tails = _post_swing_settle_debt_values(env, *args)
    _record_post_swing_settle_activation(
        env,
        values,
        eligible,
        tails,
        reward_active=True,
        signature=_post_swing_settle_signature(args),
    )
    return values


def consume_post_swing_settle_debt_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot/reset one PPO update's post-swing settle ledger."""

    template = env.scene["robot"].data.root_pos_w[:, 0]
    state = _post_swing_settle_counter_state(env, template)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    # inference_mode (not no_grad): see consume_lower_body_wave_activation_counters.
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


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


def racket_face_conditional_guidance(
    env: ManagerBasedRLEnv,
    command_name: str,
    theta_free: float = 0.262,
    theta_max: float = math.pi,
    pos_full: float = 0.075,
    pos_zero: float = 0.095,
    vel_full: float = 0.5,
    vel_zero: float = 1.0,
) -> torch.Tensor:
    """Fixed-budget signed-face correction that cannot reward abandoning readiness.

    The always-on linear face penalty helped angle while degrading position, velocity and completion.
    This term isolates the proposed remedy: within the wide strike window it spends one fixed cost
    budget, then converts that cost from a readiness deficit into signed-face error as the current
    swing-through position and velocity enter compact margins.  The face gradient is exactly zero
    outside either outer readiness margin; inside the corresponding acceptance margin the readiness
    gate is one, with a linear transition between the two:

    ``g(e; full, zero) = clamp((zero - e) / (zero - full), 0, 1)``

    ``face = clamp((face_angle - theta_free) / (theta_max - theta_free), 0, 1)``

    ``ready = g(pos_err) * g(vel_err)``

    ``penalty = window * (1 - ready * (1 - face))``

    Thus an unready strike-window state pays the fixed maximum cost, while a ready state pays only
    its face-error fraction.  Since ``d penalty / d ready = -(1 - face) <= 0``, improving position or
    velocity readiness can never increase the cost.  This avoids the inverse incentive in the naive
    ``ready * face`` formulation, where a negatively weighted policy could escape the face penalty by
    deliberately becoming less ready.  The term can still add readiness-aligned shaping; that is part
    of the integrated mechanism being ablated, not a claim of a face-only scalar reward.

    The defaults bind existing task contracts rather than adding tunable thresholds: 7.5 cm is the
    exact position-pass threshold, 9.5 cm is the virtual-ball capture radius, 0.5 m/s is the exact
    velocity-pass threshold, and 1.0 m/s is twice that tolerance.  The angular free band is 15 degrees.
    Therefore the return is dimensionless in ``[0, 1]`` and the non-positive RewTerm weight is the exact
    maximum per-strike-window-step penalty budget.  It reads the same ``_face_pair`` as every other
    face reward.

    This is task shaping, never a safety credit: it is non-positive after weighting and changes no
    termination, collision, joint/torque/qdot limit, observation, action, or plant contract.  Safety
    regressions remain non-compensable experiment failures.
    """
    values = {
        "theta_free": theta_free,
        "theta_max": theta_max,
        "pos_full": pos_full,
        "pos_zero": pos_zero,
        "vel_full": vel_full,
        "vel_zero": vel_zero,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError(f"racket_face_conditional_guidance requires finite bounds, got {values}")
    if not 0.0 <= float(theta_free) < float(theta_max) <= math.pi:
        raise ValueError(
            "racket_face_conditional_guidance requires 0 <= theta_free < theta_max <= pi"
        )
    if not 0.0 < float(pos_full) < float(pos_zero):
        raise ValueError("racket_face_conditional_guidance requires 0 < pos_full < pos_zero")
    if not 0.0 < float(vel_full) < float(vel_zero):
        raise ValueError("racket_face_conditional_guidance requires 0 < vel_full < vel_zero")

    cmd = _cmd(env, command_name)
    measured, target_normal = _face_pair(cmd)
    cos_ang = torch.sum(measured * target_normal, dim=-1).clamp(-1.0, 1.0)
    # atan2(sin, cos) keeps the near-180-degree gradient finite; acos has an infinite derivative
    # at +/-1 and can turn an otherwise-zero compact gate into 0*NaN during backpropagation.
    sin_ang = torch.linalg.vector_norm(torch.cross(measured, target_normal, dim=-1), dim=-1)
    angle = torch.atan2(sin_ang, cos_ang)
    face_fraction = ((angle - float(theta_free)) / (float(theta_max) - float(theta_free))).clamp(
        0.0, 1.0
    )

    target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    pos_err = torch.norm(cmd.racket_pos_w - target_pos_now, dim=-1)
    vel_err = torch.norm(cmd.racket_lin_vel_w - cmd.racket_target_vel_w, dim=-1)
    pos_gate = ((float(pos_zero) - pos_err) / (float(pos_zero) - float(pos_full))).clamp(0.0, 1.0)
    vel_gate = ((float(vel_zero) - vel_err) / (float(vel_zero) - float(vel_full))).clamp(0.0, 1.0)
    readiness = pos_gate * vel_gate
    event_window = _window_wide(cmd).float()
    active_face_gate = readiness * event_window

    # Spend a fixed budget in the exogenous strike-time window, then grant relief only when readiness
    # and face correctness coexist.  For every face_fraction in [0,1], greater readiness weakly lowers
    # this cost; leaving the readiness gate can never be a way to evade the penalty.
    penalty_fraction = event_window * (1.0 - readiness * (1.0 - face_fraction))

    # Directional observability for +200 screening.  These metrics exist only when the default-off
    # term is evaluated, so a purported treatment whose gate/reward never executes is visible.
    cmd.metrics["face_conditional_guidance_gate"] = active_face_gate
    cmd.metrics["face_conditional_guidance_error_fraction"] = face_fraction
    cmd.metrics["face_conditional_guidance_cost_fraction"] = penalty_fraction
    return penalty_fraction


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
