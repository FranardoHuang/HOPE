"""Pure tensor kernels for the portable MuJoCo R06/R07 slice.

The caller owns live engine reads and lifecycle state.  These functions own
only deterministic geometry/reduction math and return ordinary tensors; they
mint no owner, receipt, identity, admission, or completion authority.
"""

from __future__ import annotations


OUTCOME_NONE = 0
OUTCOME_FLIGHT_EXPIRED = 1
OUTCOME_BALL_DEAD = 2
OUTCOME_LEGAL_LANDING = 3
OUTCOME_OWN_TABLE_LANDING = 4
OUTCOME_OUT = 5
OUTCOME_INVALID = 6
RECOVERY_START_AGE_TICK = 10
RECOVERY_END_AGE_TICK = 77
RECOVERY_REWARD_WEIGHT = 0.7
PLACEMENT_BROAD_SIGMA_M = ((2.90 - 2.20) ** 2 + (0.55 - -0.55) ** 2) ** 0.5 / 2.0
RECOVERY_COMPONENT_SCALES = (
    0.05, 0.10, 0.20, 0.30, 0.20, 1.00, 0.10,
    0.20, 0.30, 0.50, 0.10, 0.10, 1.00,
)
RECOVERY_READY_TOLERANCES = (
    0.10, 0.10, 0.30, 0.50, 0.25, 1.00, 0.15,
    0.25, 0.50, 1.00, 0.10, 0.05, 0.00,
)


def observe_flight_step(
    *,
    torch,
    previous,
    current,
    tracking,
    target_positive_x,
    net_x,
    net_clear_z,
    landing_plane_z,
    table_bounds,
):
    finite = torch.isfinite(previous).all(1) & torch.isfinite(current).all(1)
    tracking = tracking & finite
    dx = current[:, 0] - previous[:, 0]
    if target_positive_x:
        net = tracking & (previous[:, 0] < net_x) & (current[:, 0] >= net_x)
        net_fraction = (net_x - previous[:, 0]) / dx.clamp_min(1.0e-12)
    else:
        net = tracking & (previous[:, 0] > net_x) & (current[:, 0] <= net_x)
        net_fraction = (net_x - previous[:, 0]) / dx.clamp_max(-1.0e-12)
    net_z = previous[:, 2] + net_fraction * (current[:, 2] - previous[:, 2])
    landing = (
        tracking
        & (previous[:, 2] > landing_plane_z)
        & (current[:, 2] <= landing_plane_z)
    )
    landing_fraction = (landing_plane_z - previous[:, 2]) / (
        current[:, 2] - previous[:, 2]
    ).clamp_max(-1.0e-12)
    xy = previous[:, :2] + landing_fraction[:, None] * (
        current[:, :2] - previous[:, :2]
    )
    x0, x1, y0, y1 = table_bounds
    on_table = (
        landing
        & (xy[:, 0] >= x0)
        & (xy[:, 0] <= x1)
        & (xy[:, 1] >= y0)
        & (xy[:, 1] <= y1)
    )
    opponent_bound = landing & (
        (xy[:, 0] > net_x) if target_positive_x else (xy[:, 0] < net_x)
    )
    on_opponent = on_table & opponent_bound
    return (
        net,
        net & (net_z >= net_clear_z),
        landing,
        xy,
        on_table,
        opponent_bound,
        on_opponent,
    )


def classify_outcome(
    *,
    torch,
    active,
    selected_contact,
    finite,
    landing_present,
    landing_on_table,
    landing_on_opponent,
    net_crossed,
    net_clear,
    dead,
    expired,
    codes,
):
    """Classify one observed flight without creating lifecycle authority."""

    landing = active & landing_present
    invalid = active & ~finite
    legal = landing & landing_on_opponent & net_crossed & net_clear
    own = landing & landing_on_table & ~landing_on_opponent
    out = (landing & ~legal & ~own) | ((dead | expired) & selected_contact)
    outcome = torch.zeros_like(codes)
    outcome = torch.where(expired, torch.full_like(outcome, OUTCOME_FLIGHT_EXPIRED), outcome)
    outcome = torch.where(dead, torch.full_like(outcome, OUTCOME_BALL_DEAD), outcome)
    outcome = torch.where(out, torch.full_like(outcome, OUTCOME_OUT), outcome)
    outcome = torch.where(own, torch.full_like(outcome, OUTCOME_OWN_TABLE_LANDING), outcome)
    outcome = torch.where(legal, torch.full_like(outcome, OUTCOME_LEGAL_LANDING), outcome)
    outcome = torch.where(invalid, torch.full_like(outcome, OUTCOME_INVALID), outcome)
    return outcome.ne(OUTCOME_NONE), outcome


def r06_rows(
    *,
    torch,
    settled,
    selected_contact,
    invalid_outcome,
    crossing_present,
    crossing_xy,
    target_xy,
    opponent_bound,
    on_opponent,
    net_crossed,
    net_clear,
    broad_sigma,
    narrow_sigma,
):
    present = settled
    numeric = (
        (~crossing_present | torch.isfinite(crossing_xy).all(1))
        & torch.isfinite(target_xy).all()
    )
    valid = present & numeric & ~invalid_outcome
    contact = valid & selected_contact
    crossing = contact & crossing_present
    error = torch.linalg.vector_norm(crossing_xy - target_xy, dim=1)
    broad = torch.reciprocal(1.0 + torch.square(error / broad_sigma))
    narrow = torch.exp(-torch.square(error / narrow_sigma))
    score_eligible = (
        crossing & net_crossed & net_clear & opponent_bound
    )
    total = torch.where(
        score_eligible,
        0.5 * (broad + narrow) * torch.where(
            on_opponent, torch.ones_like(error), torch.full_like(error, 0.5)
        ),
        torch.zeros_like(error),
    )
    common = score_eligible & on_opponent
    facts = torch.zeros((present.shape[0], 32), dtype=error.dtype, device=error.device)
    facts[:, 0] = common.to(error.dtype)
    facts[:, 1] = total
    # A-family's placement treatment gain is exactly one for every ordinary
    # source-valid policy settlement, including the important no-contact zero.
    facts[:, 2] = valid.to(error.dtype)
    facts[:, 3:5] = torch.where(crossing[:, None], crossing_xy, torch.zeros_like(crossing_xy))
    facts[:, 5] = torch.where(crossing, error, torch.zeros_like(error))
    facts[:, 6:11] = torch.stack(
        (
            on_opponent,
            selected_contact,
            crossing,
            net_crossed,
            net_clear,
        ),
        dim=1,
    ).to(error.dtype)
    facts = torch.where(valid[:, None], facts, torch.zeros_like(facts))
    return present, valid, common, facts


def recovery_errors(
    *,
    torch,
    root_position,
    reference_root_position,
    root_orientation_error_sq,
    root_linear_velocity,
    root_angular_velocity,
    joint_position,
    reference_joint_position,
    joint_velocity,
    body_position,
    reference_body_position,
    body_orientation_error_sq,
    body_linear_velocity,
    body_angular_velocity,
    foot_support,
    foot_slip_xy,
):
    rms = lambda value: torch.sqrt(torch.mean(torch.square(value), dim=1))
    rms_l2 = lambda value: torch.sqrt(
        torch.mean(torch.sum(torch.square(value), dim=2), dim=1)
    )
    reference_root = reference_root_position
    supported_slip = torch.where(
        foot_support,
        torch.linalg.vector_norm(foot_slip_xy, dim=2),
        torch.zeros_like(foot_slip_xy[:, :, 0]),
    )
    return torch.stack(
        (
            torch.linalg.vector_norm(root_position - reference_root, dim=1),
            torch.sqrt(root_orientation_error_sq),
            torch.linalg.vector_norm(root_linear_velocity, dim=1),
            torch.linalg.vector_norm(root_angular_velocity, dim=1),
            rms(joint_position - reference_joint_position),
            rms(joint_velocity),
            rms_l2(body_position - reference_body_position),
            torch.sqrt(torch.mean(body_orientation_error_sq, dim=1)),
            rms_l2(body_linear_velocity),
            rms_l2(body_angular_velocity),
            torch.linalg.vector_norm(root_position[:, :2] - reference_root[:, :2], dim=1),
            supported_slip.amax(dim=1),
            (2 - foot_support.to(torch.long).sum(dim=1)).clamp_min(0).to(root_position.dtype),
        ),
        dim=1,
    )


def r07_rows(
    *, torch, expected, age, errors, hard_safety_ok, scales, ready_tolerances, weight
):
    valid = torch.isfinite(errors).all(1)
    eligible = expected & valid
    raw = torch.mean(torch.reciprocal(1.0 + torch.square(errors / scales)), dim=1)
    weighted = torch.where(eligible, raw * weight, torch.zeros_like(raw))
    ready = (
        eligible
        & hard_safety_ok
        & torch.all(errors <= ready_tolerances, dim=1)
    )
    facts = torch.zeros((expected.shape[0], 32), dtype=errors.dtype, device=errors.device)
    facts[:, 0] = weighted
    facts[:, 1] = torch.where(valid, raw, torch.zeros_like(raw))
    facts[:, 2] = eligible.to(errors.dtype)
    facts[:, 3] = valid.to(errors.dtype)
    facts[:, 4] = (expected & ~valid).to(errors.dtype)
    facts[:, 5] = ready.to(errors.dtype)
    facts[:, 6] = age.to(errors.dtype)
    facts[:, 7:20] = torch.where(valid[:, None], errors, torch.zeros_like(errors))
    facts = torch.where(expected[:, None], facts, torch.zeros_like(facts))
    return eligible, valid, ready, facts


def recovery_status(*, torch, recovering, age, terminated, truncated, ready_seen, end_age):
    failure = recovering & (terminated | truncated)
    complete = recovering & ~failure & (age >= end_age)
    success = complete & ready_seen
    timeout = complete & ~ready_seen
    return failure | success | timeout, success, failure, timeout


__all__ = (
    "PLACEMENT_BROAD_SIGMA_M",
    "OUTCOME_BALL_DEAD",
    "OUTCOME_FLIGHT_EXPIRED",
    "OUTCOME_INVALID",
    "OUTCOME_LEGAL_LANDING",
    "OUTCOME_NONE",
    "OUTCOME_OUT",
    "OUTCOME_OWN_TABLE_LANDING",
    "RECOVERY_COMPONENT_SCALES",
    "RECOVERY_END_AGE_TICK",
    "RECOVERY_READY_TOLERANCES",
    "RECOVERY_REWARD_WEIGHT",
    "RECOVERY_START_AGE_TICK",
    "observe_flight_step",
    "classify_outcome",
    "r06_rows",
    "recovery_errors",
    "r07_rows",
    "recovery_status",
)
