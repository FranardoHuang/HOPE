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
    maximum_inward_target: torch.Tensor
    state_finite: torch.Tensor

    @property
    def hard_violation_env(self) -> torch.Tensor:
        return torch.any(
            self.qdes_safety_violation | self.crossing_violation, dim=1
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

    shape = tuple(pre_clamp_qdes.shape)
    if len(shape) != 2:
        raise ValueError("pre_clamp_qdes must have shape [N,J]")
    for name, value in (
        ("previous_executable_qdes", previous_executable_qdes),
        ("default_qdes", default_qdes),
        ("soft_lower", soft_lower),
        ("soft_upper", soft_upper),
        ("hard_lower", hard_lower),
        ("hard_upper", hard_upper),
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
    if type(project_finite_without_termination) is not bool:
        raise TypeError("project_finite_without_termination must be bool")

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

    qdes_nonfinite = ~torch.isfinite(pre_clamp_qdes)
    qdes_forbidden_request = (
        qdes_nonfinite
        | pre_clamp_qdes.le(hard_inner_lower)
        | pre_clamp_qdes.ge(hard_inner_upper)
    )
    qdes_safety_violation = (
        qdes_nonfinite
        if project_finite_without_termination
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
    maximum_inward_target = torch.where(
        lower_only, target_upper, target_lower
    )
    guard_target = torch.where(
        unambiguous_crossing_risk, maximum_inward_target, brake_target
    )
    nominal_source = torch.where(qdes_nonfinite, brake_target, sanitized)
    nominal_target = torch.clamp(
        nominal_source, min=target_lower, max=target_upper
    )
    executable = torch.where(per_joint_guard, guard_target, nominal_target)
    return ActionBallQdesGuardResult(
        executable_qdes=executable,
        nominal_projected_qdes=nominal_target,
        nominal_projection_span=target_upper - target_lower,
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
        maximum_inward_target=maximum_inward_target,
        state_finite=state_finite,
    )


__all__ = ("ActionBallQdesGuardResult", "action_ball_qdes_guard")
