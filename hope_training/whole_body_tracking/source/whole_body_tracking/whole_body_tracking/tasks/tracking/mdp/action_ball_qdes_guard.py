"""One engine-neutral executable ``q_des`` guard for ActionBall.

The policy proposes a deploy-space joint target.  This module owns the pure
tensor transformation from that proposal plus current ``q/qdot`` to the target
that a simulator may execute.  It owns no Isaac/MuJoCo object, termination
manager, counter, receipt, or policy observation; each backend retains its own
plant write and telemetry while consuming the same numerical decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionBallQdesGuardResult:
    """Device-local result of one policy-step target projection."""

    executable_qdes: torch.Tensor
    nominal_projected_qdes: torch.Tensor
    nominal_projection_span: torch.Tensor
    finite_fallback_qdes: torch.Tensor
    sanitized_qdes: torch.Tensor
    target_lower: torch.Tensor
    target_upper: torch.Tensor
    hard_inner_lower: torch.Tensor
    hard_inner_upper: torch.Tensor
    safe_joint_pos: torch.Tensor
    safe_joint_vel: torch.Tensor
    ballistic_next: torch.Tensor
    qdes_nonfinite: torch.Tensor
    qdes_forbidden_request: torch.Tensor
    qdes_safety_violation: torch.Tensor
    crossing_violation: torch.Tensor
    per_joint_guard: torch.Tensor
    brake_target: torch.Tensor
    lower_crossing_risk: torch.Tensor
    upper_crossing_risk: torch.Tensor
    unambiguous_crossing_risk: torch.Tensor
    state_finite: torch.Tensor

    @property
    def hard_violation_env(self) -> torch.Tensor:
        return torch.any(
            self.qdes_safety_violation | self.crossing_violation, dim=1
        )


@dataclass(frozen=True)
class ActionBallQdesEnvelope:
    """Construction-owned immutable tensors for the executable target envelope.

    The tensors are detached clones when ``freeze=True`` is requested from the
    factory below.  Runtime code must treat them as read-only: they are derived
    exclusively from the robot/config limit contract and are not episode state.
    """

    soft_lower: torch.Tensor
    soft_upper: torch.Tensor
    hard_lower: torch.Tensor
    hard_upper: torch.Tensor
    hard_inner_lower: torch.Tensor
    hard_inner_upper: torch.Tensor
    target_lower: torch.Tensor
    target_upper: torch.Tensor
    target_span: torch.Tensor
    project_finite_without_termination: bool


def derive_action_ball_qdes_envelope(
    *,
    soft_lower: torch.Tensor,
    soft_upper: torch.Tensor,
    hard_lower: torch.Tensor,
    hard_upper: torch.Tensor,
    hard_margin_rad: float,
    hard_margin_fraction: float,
    project_finite_without_termination: bool,
    projection_soft_inset_fraction: float,
    freeze: bool = False,
    validate_values: bool = False,
) -> ActionBallQdesEnvelope:
    """Derive the static q_des envelope once at backend construction.

    ``validate_values`` is deliberately explicit.  The legacy public wrapper
    below keeps its historical host-sync-free behavior, while a backend that
    owns static URDF/config tensors asks for one construction-time validation
    and then reuses the returned envelope on every policy step.
    """

    if type(soft_lower) is not torch.Tensor or soft_lower.ndim != 2:
        raise ValueError("soft_lower must have shape [N,J]")
    shape = tuple(soft_lower.shape)
    for name, value in (
        ("soft_upper", soft_upper),
        ("hard_lower", hard_lower),
        ("hard_upper", hard_upper),
    ):
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != shape
            or value.dtype != soft_lower.dtype
            or value.device != soft_lower.device
        ):
            raise ValueError(f"{name} must match soft_lower")
    if type(project_finite_without_termination) is not bool:
        raise TypeError("project_finite_without_termination must be bool")
    if type(freeze) is not bool or type(validate_values) is not bool:
        raise TypeError("freeze and validate_values must be bool")

    if freeze:
        soft_lower = soft_lower.detach().clone()
        soft_upper = soft_upper.detach().clone()
        hard_lower = hard_lower.detach().clone()
        hard_upper = hard_upper.detach().clone()

    hard_travel = hard_upper - hard_lower
    hard_inset = hard_margin_rad + hard_margin_fraction * hard_travel
    hard_inner_lower = hard_lower + hard_inset
    hard_inner_upper = hard_upper - hard_inset
    target_lower = torch.maximum(soft_lower, hard_inner_lower)
    target_upper = torch.minimum(soft_upper, hard_inner_upper)
    if project_finite_without_termination:
        soft_travel = soft_upper - soft_lower
        projection_inset = projection_soft_inset_fraction * soft_travel
        target_lower = torch.maximum(
            target_lower, soft_lower + projection_inset
        )
        target_upper = torch.minimum(
            target_upper, soft_upper - projection_inset
        )
    target_span = target_upper - target_lower

    if validate_values:
        valid = torch.all(
            torch.isfinite(soft_lower)
            & torch.isfinite(soft_upper)
            & torch.isfinite(hard_lower)
            & torch.isfinite(hard_upper)
            & torch.isfinite(hard_inner_lower)
            & torch.isfinite(hard_inner_upper)
            & torch.isfinite(target_lower)
            & torch.isfinite(target_upper)
            & soft_lower.lt(soft_upper)
            & hard_lower.lt(hard_upper)
            & hard_inner_lower.lt(hard_inner_upper)
            & soft_lower.ge(hard_lower)
            & soft_upper.le(hard_upper)
            & target_span.gt(0.0)
        )
        if not bool(valid.detach().cpu()):
            raise ValueError(
                "q_des envelope requires finite positive soft/hard spans, "
                "soft inside hard, and a non-empty projected target"
            )

    return ActionBallQdesEnvelope(
        soft_lower=soft_lower,
        soft_upper=soft_upper,
        hard_lower=hard_lower,
        hard_upper=hard_upper,
        hard_inner_lower=hard_inner_lower,
        hard_inner_upper=hard_inner_upper,
        target_lower=target_lower,
        target_upper=target_upper,
        target_span=target_span,
        project_finite_without_termination=(
            project_finite_without_termination
        ),
    )


def action_ball_qdes_guard_with_envelope(
    *,
    pre_clamp_qdes: torch.Tensor,
    previous_executable_qdes: torch.Tensor,
    previous_executable_valid: torch.Tensor,
    default_qdes: torch.Tensor,
    envelope: ActionBallQdesEnvelope,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    policy_dt_s: float,
) -> ActionBallQdesGuardResult:
    """Apply only dynamic projection/braking against one frozen envelope."""

    shape = tuple(pre_clamp_qdes.shape)
    if len(shape) != 2:
        raise ValueError("pre_clamp_qdes must have shape [N,J]")
    for name, value in (
        ("previous_executable_qdes", previous_executable_qdes),
        ("default_qdes", default_qdes),
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
    ):
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != shape
            or value.dtype != pre_clamp_qdes.dtype
            or value.device != pre_clamp_qdes.device
        ):
            raise ValueError(f"{name} must match pre_clamp_qdes")
    if (
        type(previous_executable_valid) is not torch.Tensor
        or tuple(previous_executable_valid.shape) != (shape[0],)
        or previous_executable_valid.dtype != torch.bool
        or previous_executable_valid.device != pre_clamp_qdes.device
    ):
        raise ValueError("previous_executable_valid must have bool shape [N]")
    if type(envelope) is not ActionBallQdesEnvelope:
        raise TypeError("envelope must be ActionBallQdesEnvelope")
    hard_inner_lower = envelope.hard_inner_lower
    hard_inner_upper = envelope.hard_inner_upper
    target_lower = envelope.target_lower
    target_upper = envelope.target_upper

    previous_is_safe = (
        previous_executable_valid[:, None]
        & torch.isfinite(previous_executable_qdes)
    )
    finite_fallback = torch.where(
        previous_is_safe, previous_executable_qdes, default_qdes
    )
    sanitized = torch.where(
        torch.isfinite(pre_clamp_qdes), pre_clamp_qdes, finite_fallback
    )

    qdes_nonfinite = ~torch.isfinite(pre_clamp_qdes)
    qdes_forbidden_request = (
        qdes_nonfinite
        | pre_clamp_qdes.le(hard_inner_lower)
        | pre_clamp_qdes.ge(hard_inner_upper)
    )
    qdes_safety_violation = (
        qdes_nonfinite
        if envelope.project_finite_without_termination
        else qdes_forbidden_request
    )

    state_finite = torch.isfinite(joint_pos) & torch.isfinite(joint_vel)
    safe_joint_pos = torch.where(
        torch.isfinite(joint_pos), joint_pos, default_qdes
    )
    safe_joint_vel = torch.where(
        torch.isfinite(joint_vel), joint_vel, torch.zeros_like(joint_vel)
    )
    ballistic_next = safe_joint_pos + safe_joint_vel * policy_dt_s
    crossing_violation = (
        ~state_finite
        | safe_joint_pos.le(hard_inner_lower)
        | safe_joint_pos.ge(hard_inner_upper)
        | ballistic_next.le(hard_inner_lower)
        | ballistic_next.ge(hard_inner_upper)
    )
    per_joint_guard = qdes_safety_violation | crossing_violation
    brake_target = torch.clamp(
        safe_joint_pos - safe_joint_vel * policy_dt_s,
        min=target_lower,
        max=target_upper,
    )
    lower_crossing_risk = state_finite & (
        safe_joint_pos.le(hard_inner_lower)
        | ballistic_next.le(hard_inner_lower)
    )
    upper_crossing_risk = state_finite & (
        safe_joint_pos.ge(hard_inner_upper)
        | ballistic_next.ge(hard_inner_upper)
    )
    lower_only = lower_crossing_risk & ~upper_crossing_risk
    upper_only = upper_crossing_risk & ~lower_crossing_risk
    unambiguous_crossing_risk = lower_only | upper_only
    guard_target = brake_target
    nominal_source = torch.where(qdes_nonfinite, brake_target, sanitized)
    nominal_target = torch.clamp(
        nominal_source, min=target_lower, max=target_upper
    )
    executable = torch.where(per_joint_guard, guard_target, nominal_target)
    return ActionBallQdesGuardResult(
        executable_qdes=executable,
        nominal_projected_qdes=nominal_target,
        nominal_projection_span=envelope.target_span,
        finite_fallback_qdes=finite_fallback,
        sanitized_qdes=sanitized,
        target_lower=target_lower,
        target_upper=target_upper,
        hard_inner_lower=hard_inner_lower,
        hard_inner_upper=hard_inner_upper,
        safe_joint_pos=safe_joint_pos,
        safe_joint_vel=safe_joint_vel,
        ballistic_next=ballistic_next,
        qdes_nonfinite=qdes_nonfinite,
        qdes_forbidden_request=qdes_forbidden_request,
        qdes_safety_violation=qdes_safety_violation,
        crossing_violation=crossing_violation,
        per_joint_guard=per_joint_guard,
        brake_target=brake_target,
        lower_crossing_risk=lower_crossing_risk,
        upper_crossing_risk=upper_crossing_risk,
        unambiguous_crossing_risk=unambiguous_crossing_risk,
        state_finite=state_finite,
    )


def action_ball_qdes_guard(
    *,
    pre_clamp_qdes: torch.Tensor,
    previous_executable_qdes: torch.Tensor,
    previous_executable_valid: torch.Tensor,
    default_qdes: torch.Tensor,
    soft_lower: torch.Tensor,
    soft_upper: torch.Tensor,
    hard_lower: torch.Tensor,
    hard_upper: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    policy_dt_s: float,
    hard_margin_rad: float,
    hard_margin_fraction: float,
    project_finite_without_termination: bool,
    projection_soft_inset_fraction: float,
) -> ActionBallQdesGuardResult:
    """Project one batched proposal and brake dangerous measured state.

    All joint-valued tensors have shape ``[N,J]`` except
    ``previous_executable_valid`` which is ``[N]``.  Construction code in each
    backend validates static envelopes once; this function stays free of host
    synchronization and applies the shared per-row decision only.
    """

    envelope = derive_action_ball_qdes_envelope(
        soft_lower=soft_lower,
        soft_upper=soft_upper,
        hard_lower=hard_lower,
        hard_upper=hard_upper,
        hard_margin_rad=hard_margin_rad,
        hard_margin_fraction=hard_margin_fraction,
        project_finite_without_termination=(
            project_finite_without_termination
        ),
        projection_soft_inset_fraction=(
            projection_soft_inset_fraction
        ),
    )
    return action_ball_qdes_guard_with_envelope(
        pre_clamp_qdes=pre_clamp_qdes,
        previous_executable_qdes=previous_executable_qdes,
        previous_executable_valid=previous_executable_valid,
        default_qdes=default_qdes,
        envelope=envelope,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        policy_dt_s=policy_dt_s,
    )


__all__ = (
    "ActionBallQdesEnvelope",
    "ActionBallQdesGuardResult",
    "action_ball_qdes_guard",
    "action_ball_qdes_guard_with_envelope",
    "derive_action_ball_qdes_envelope",
)
