"""Engine-neutral continuous learning costs for ActionBall FullMDP.

The four functions in this module are pure tensor kernels.  They own the
numeric contract shared by Isaac Lab and MuJoCo GPU, but no manager, simulator,
termination, counter, receipt, or policy observation.  Each function returns a
non-positive learning value; RewardManager therefore uses the positive
absolute coefficient from :data:`REGULARIZATION_SPECS` while preserving one
auditable component row per objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from . import action_ball_full_mdp_reward_contract as reward_contract
except ImportError:
    import action_ball_full_mdp_reward_contract as reward_contract

JOINT_COUNT = reward_contract.REGULARIZATION_JOINT_COUNT
SOFT_LIMIT_MARGIN_FRAC = reward_contract.REGULARIZATION_SOFT_LIMIT_MARGIN_FRAC
SOFT_LIMIT_PENALTY_FLOOR = (
    reward_contract.REGULARIZATION_SOFT_LIMIT_PENALTY_FLOOR
)
PROJECTION_KNEE_FRAC = reward_contract.REGULARIZATION_PROJECTION_KNEE_FRAC
STANCE_EPS_FRAC = reward_contract.REGULARIZATION_STANCE_EPS_FRAC
MARGIN_FLOOR_FRAC = reward_contract.REGULARIZATION_MARGIN_FLOOR_FRAC
RegularizationRewardSpec = reward_contract.RegularizationRewardSpec
REGULARIZATION_SPECS = reward_contract.REGULARIZATION_SPECS
REGULARIZATION_NAMES = reward_contract.REGULARIZATION_NAMES


@dataclass(frozen=True)
class _PreparedSoftLimitBarrierV2:
    """Construction-static geometry shared by the two live barrier rows."""

    num_envs: int
    lower: torch.Tensor
    upper: torch.Tensor
    default_q: torch.Tensor
    hard_lower: torch.Tensor
    hard_upper: torch.Tensor
    span: torch.Tensor
    margin_eff: torch.Tensor
    band_rad: torch.Tensor
    penalty_floor: float


def _joint_rows(value: torch.Tensor, *, name: str) -> tuple[int, int]:
    if (
        type(value) is not torch.Tensor
        or value.ndim != 2
        or value.shape[1] != JOINT_COUNT
        or value.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError(f"{name} must be floating [N,{JOINT_COUNT}]")
    return int(value.shape[0]), int(value.shape[1])


def _matching_joint_rows(
    reference: torch.Tensor, value: torch.Tensor, *, name: str
) -> None:
    if (
        type(value) is not torch.Tensor
        or tuple(value.shape) != tuple(reference.shape)
        or value.dtype != reference.dtype
        or value.device != reference.device
    ):
        raise ValueError(f"{name} must exactly match the reference tensor")


def _expanded_limits(
    limits: torch.Tensor,
    *,
    reference: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(reference.shape[0])
    if type(limits) is not torch.Tensor or limits.device != reference.device:
        raise ValueError(f"{name} must be a tensor on the position device")
    if limits.dtype != reference.dtype:
        raise ValueError(f"{name} must use the position dtype")
    if limits.ndim == 2 and tuple(limits.shape) == (JOINT_COUNT, 2):
        lower, upper = limits[:, 0], limits[:, 1]
    elif (
        limits.ndim == 3
        and tuple(limits.shape)[1:] == (JOINT_COUNT, 2)
        and int(limits.shape[0]) in (1, n)
    ):
        lower, upper = limits[:, :, 0], limits[:, :, 1]
    else:
        raise ValueError(
            f"{name} must have shape [{JOINT_COUNT},2], [1,{JOINT_COUNT},2], "
            f"or [N,{JOINT_COUNT},2]"
        )
    torch._assert_async(torch.all(torch.isfinite(lower) & torch.isfinite(upper)))
    torch._assert_async(torch.all(upper > lower))
    return lower, upper


def _expanded_default(
    default_joint_pos: torch.Tensor, *, reference: torch.Tensor
) -> torch.Tensor:
    n = int(reference.shape[0])
    if (
        type(default_joint_pos) is not torch.Tensor
        or default_joint_pos.device != reference.device
        or default_joint_pos.dtype != reference.dtype
    ):
        raise ValueError("default_joint_pos must use the position device/dtype")
    if default_joint_pos.ndim == 1 and tuple(default_joint_pos.shape) == (JOINT_COUNT,):
        result = default_joint_pos
    elif (
        default_joint_pos.ndim == 2
        and int(default_joint_pos.shape[0]) in (1, n)
        and int(default_joint_pos.shape[1]) == JOINT_COUNT
    ):
        result = default_joint_pos
    else:
        raise ValueError(f"default_joint_pos must have shape [{JOINT_COUNT}], [1,{JOINT_COUNT}], or [N,{JOINT_COUNT}]")
    torch._assert_async(torch.all(torch.isfinite(result)))
    return result


def action_rate_l2(
    action: torch.Tensor, previous_action: torch.Tensor
) -> torch.Tensor:
    """Return ``-sum((a_t-a_(t-1))**2)`` with finite invalid-row fallback.

    The sum over all 31 action dimensions is exactly the Isaac Lab /
    BeyondMimic ``mdp.action_rate_l2`` convention.  Reset semantics remain the
    backend action manager's responsibility: both backends reset current and
    previous action to zero before the first post-reset policy action.
    """

    _joint_rows(action, name="action")
    _matching_joint_rows(action, previous_action, name="previous_action")
    row_finite = torch.all(
        torch.isfinite(action) & torch.isfinite(previous_action), dim=1
    )
    delta = torch.where(
        row_finite[:, None], action - previous_action, torch.zeros_like(action)
    )
    return -torch.sum(torch.square(delta), dim=1)


def _prepare_soft_limit_barrier_v2(
    reference: torch.Tensor,
    soft_limits: torch.Tensor,
    default_joint_pos: torch.Tensor,
    hard_limits: torch.Tensor,
    *,
    margin_frac: float = SOFT_LIMIT_MARGIN_FRAC,
    penalty_floor: float = SOFT_LIMIT_PENALTY_FLOOR,
    freeze: bool = False,
) -> _PreparedSoftLimitBarrierV2:
    """Validate geometry; optionally freeze its construction-owned sources."""

    num_envs, _ = _joint_rows(reference, name="reference")
    if not 0.0 < float(margin_frac) < 0.5:
        raise ValueError("margin_frac must be in (0,0.5)")
    if not 0.0 <= float(penalty_floor) < 1.0:
        raise ValueError("penalty_floor must be in [0,1)")
    lower, upper = _expanded_limits(
        soft_limits, reference=reference, name="soft_limits"
    )
    hard_lower, hard_upper = _expanded_limits(
        hard_limits, reference=reference, name="hard_limits"
    )
    default_q = _expanded_default(default_joint_pos, reference=reference)
    if freeze:
        lower = lower.clone()
        upper = upper.clone()
        default_q = default_q.clone()
        hard_lower = hard_lower.clone()
        hard_upper = hard_upper.clone()
    span = upper - lower
    default_distance = torch.minimum(default_q - lower, upper - default_q) / span
    margin_eff = torch.clamp(
        default_distance - STANCE_EPS_FRAC, max=float(margin_frac)
    )
    torch._assert_async(torch.all(margin_eff > MARGIN_FLOOR_FRAC))
    band_rad = margin_eff * span
    return _PreparedSoftLimitBarrierV2(
        num_envs=num_envs,
        lower=lower,
        upper=upper,
        default_q=default_q,
        hard_lower=hard_lower,
        hard_upper=hard_upper,
        span=span,
        margin_eff=margin_eff,
        band_rad=band_rad,
        penalty_floor=float(penalty_floor),
    )


def _soft_limit_barrier_v2_prepared(
    positions: torch.Tensor, geometry: _PreparedSoftLimitBarrierV2
) -> torch.Tensor:
    """Evaluate only the dynamic half of the v2 barrier."""

    num_envs, _ = _joint_rows(positions, name="positions")
    if (
        type(geometry) is not _PreparedSoftLimitBarrierV2
        or num_envs != geometry.num_envs
        or positions.dtype != geometry.lower.dtype
        or positions.device != geometry.lower.device
    ):
        raise ValueError("positions must match the prepared barrier geometry")

    row_finite = torch.all(torch.isfinite(positions), dim=1)
    safe_positions = torch.where(
        row_finite[:, None],
        positions,
        torch.broadcast_to(geometry.default_q, positions.shape),
    )
    distance = torch.minimum(
        safe_positions - geometry.lower, geometry.upper - safe_positions
    ) / geometry.span
    intrusion = torch.relu(geometry.margin_eff - distance) / geometry.margin_eff
    depth_rad = intrusion * geometry.band_rad
    ramp = torch.where(
        intrusion <= 1.0,
        torch.square(depth_rad) / (2.0 * geometry.band_rad),
        depth_rad - 0.5 * geometry.band_rad,
    )
    hard_excess = torch.relu(geometry.hard_lower - safe_positions) + torch.relu(
        safe_positions - geometry.hard_upper
    )
    floor_rad = torch.where(
        hard_excess > 0.0,
        geometry.penalty_floor * geometry.band_rad,
        torch.zeros_like(ramp),
    )
    per_joint = torch.where(
        intrusion > 0.0, ramp + floor_rad, torch.zeros_like(ramp)
    )
    value = torch.sum(per_joint, dim=1)
    return -torch.where(row_finite, value, torch.zeros_like(value))


def soft_limit_barrier_v2(
    positions: torch.Tensor,
    soft_limits: torch.Tensor,
    default_joint_pos: torch.Tensor,
    hard_limits: torch.Tensor,
    *,
    margin_frac: float = SOFT_LIMIT_MARGIN_FRAC,
    penalty_floor: float = SOFT_LIMIT_PENALTY_FLOOR,
) -> torch.Tensor:
    """Return the negative v2 soft-limit barrier in radians.

    Finite rows preserve the adopted C1 quadratic-band / linear-tail kernel
    exactly.  A row with a non-finite position returns finite zero because its
    independent terminal/plant mechanism owns the non-finite consequence.
    """

    geometry = _prepare_soft_limit_barrier_v2(
        positions,
        soft_limits,
        default_joint_pos,
        hard_limits,
        margin_frac=margin_frac,
        penalty_floor=penalty_floor,
    )
    return _soft_limit_barrier_v2_prepared(positions, geometry)


def qdes_projection_penalty(
    pre_clamp_qdes: torch.Tensor,
    nominal_projected_qdes: torch.Tensor,
    nominal_projection_span: torch.Tensor,
    pre_clamp_valid: torch.Tensor,
    projected_valid: torch.Tensor,
    *,
    knee_frac: float = PROJECTION_KNEE_FRAC,
) -> torch.Tensor:
    """Return the negative v2 projection distance with its linear tail."""

    n, _ = _joint_rows(pre_clamp_qdes, name="pre_clamp_qdes")
    _matching_joint_rows(
        pre_clamp_qdes, nominal_projected_qdes, name="nominal_projected_qdes"
    )
    _matching_joint_rows(
        pre_clamp_qdes, nominal_projection_span, name="nominal_projection_span"
    )
    if not 0.0 < float(knee_frac) < 0.5:
        raise ValueError("knee_frac must be in (0,0.5)")
    for name, value in (
        ("pre_clamp_valid", pre_clamp_valid),
        ("projected_valid", projected_valid),
    ):
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != (n,)
            or value.dtype is not torch.bool
            or value.device != pre_clamp_qdes.device
        ):
            raise ValueError(f"{name} must be bool [N] on the qdes device")
    valid = pre_clamp_valid & projected_valid
    structural = torch.all(
        torch.isfinite(nominal_projected_qdes)
        & torch.isfinite(nominal_projection_span)
        & nominal_projection_span.gt(0.0),
        dim=1,
    )
    torch._assert_async(torch.all(~valid | structural))
    safe_projected = torch.where(
        structural[:, None], nominal_projected_qdes, torch.zeros_like(nominal_projected_qdes)
    )
    safe_span = torch.where(
        structural[:, None], nominal_projection_span, torch.ones_like(nominal_projection_span)
    )
    nonfinite = ~torch.isfinite(pre_clamp_qdes)
    safe_pre = torch.where(nonfinite, safe_projected + safe_span, pre_clamp_qdes)
    distance_rad = torch.abs(safe_pre - safe_projected)
    knee_rad = float(knee_frac) * safe_span
    per_joint = torch.where(
        distance_rad <= knee_rad,
        torch.square(distance_rad) / (2.0 * knee_rad),
        distance_rad - 0.5 * knee_rad,
    )
    per_joint = torch.where(valid[:, None], per_joint, torch.zeros_like(per_joint))
    return -torch.sum(per_joint, dim=1)


__all__ = [
    "JOINT_COUNT",
    "SOFT_LIMIT_MARGIN_FRAC",
    "SOFT_LIMIT_PENALTY_FLOOR",
    "PROJECTION_KNEE_FRAC",
    "RegularizationRewardSpec",
    "REGULARIZATION_SPECS",
    "REGULARIZATION_NAMES",
    "action_rate_l2",
    "soft_limit_barrier_v2",
    "qdes_projection_penalty",
]
