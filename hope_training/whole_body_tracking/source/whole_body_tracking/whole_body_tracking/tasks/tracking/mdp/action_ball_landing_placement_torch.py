"""Torch mirror of the engine-neutral ActionBall landing-placement score.

The function in this module owns no Isaac or MuJoCo type.  It accepts batched
ball-centre landing-plane crossing facts and the canonical standard-library
profile.  Return direction vectors, speed, baseline state, and post-bounce
state are deliberately absent from the ABI.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from action_ball_landing_placement import LandingPlacementProfile, SCORE_REASONS


REASON_TO_CODE = {reason: index for index, reason in enumerate(SCORE_REASONS)}
TASK_IDENTITY_FAULT_CODE = -1
# Reserved for infrastructure ledgers outside this canonical scorer.  Missing
# producer evidence is represented here by the public canonical
# ``crossing_contract_fault`` reason; callers must not inject an infrastructure
# settlement cause into C04's reason field.
PRODUCER_CONTRACT_FAULT_CODE = -2
TASK_IDENTITY_TOKEN_BYTES = 32


@dataclass(frozen=True)
class LandingPlacementTorchScore:
    """Batched score and denominator facts; every field has shape ``[N]``."""

    contact_valid: torch.Tensor
    target_valid: torch.Tensor
    task_target_match: torch.Tensor
    task_identity_match: torch.Tensor
    task_identity_fault: torch.Tensor
    drain_fault: torch.Tensor
    first_plane_crossing_present: torch.Tensor
    first_plane_crossing_valid: torch.Tensor
    first_plane_crossing_nonfinite: torch.Tensor
    first_plane_crossing_contract_fault: torch.Tensor
    ball_center_net_crossed: torch.Tensor
    ball_center_net_clear: torch.Tensor
    opponent_bound: torch.Tensor
    on_opponent_table: torch.Tensor
    reason_code: torch.Tensor
    placement_error_m: torch.Tensor
    broad_kernel: torch.Tensor
    narrow_kernel: torch.Tensor
    blended_kernel: torch.Tensor
    table_gate: torch.Tensor
    total: torch.Tensor


def _bool_vector(value: object, *, name: str, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.bool or value.shape != (batch_size,):
        raise ValueError(f"{name} must have bool shape [N]")
    return value


def _identity_token(
    value: object,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        value.dtype != torch.uint8
        or value.shape != (batch_size, TASK_IDENTITY_TOKEN_BYTES)
        or value.device != device
    ):
        raise ValueError(f"{name} must have device-local uint8 shape [N,32]")
    return value


def score_landing_placement_torch(
    profile: LandingPlacementProfile,
    *,
    frame_id: str,
    profile_sha256: str,
    expected_task_identity_token: torch.Tensor,
    facts_task_identity_token: torch.Tensor,
    expected_target_xy_m: torch.Tensor,
    facts_target_xy_m: torch.Tensor,
    contact_valid: torch.Tensor,
    first_plane_crossing_present: torch.Tensor,
    first_plane_crossing_valid: torch.Tensor,
    first_plane_crossing_nonfinite: torch.Tensor,
    first_plane_crossing_xy_m: torch.Tensor,
    ball_center_net_crossed: torch.Tensor,
    ball_center_net_clear: torch.Tensor,
) -> LandingPlacementTorchScore:
    """Vectorized mirror of ``score_landing_placement``.

    The two identity tokens are the raw 32 canonical-SHA bytes, in digest byte
    order, of the expected and facts-side ``LandingPlacementTaskIdentity``.
    They are compared entirely on device.  Non-finite crossing rows fail
    closed without feeding NaNs through kernel arithmetic.
    ``on_opponent_table`` remains distinct from crossing validity.
    """

    if not isinstance(profile, LandingPlacementProfile):
        raise TypeError("profile must be a LandingPlacementProfile")
    if frame_id != profile.frame_id:
        raise ValueError("landing-placement frame_id differs")
    if profile_sha256 != profile.canonical_sha256:
        raise ValueError("landing-placement profile SHA differs")
    if not isinstance(first_plane_crossing_xy_m, torch.Tensor):
        raise TypeError("first_plane_crossing_xy_m must be a torch.Tensor")
    if (
        first_plane_crossing_xy_m.ndim != 2
        or first_plane_crossing_xy_m.shape[-1] != 2
        or not torch.is_floating_point(first_plane_crossing_xy_m)
    ):
        raise ValueError("first_plane_crossing_xy_m must have floating shape [N,2]")

    batch_size = first_plane_crossing_xy_m.shape[0]
    expected_token = _identity_token(
        expected_task_identity_token,
        name="expected_task_identity_token",
        batch_size=batch_size,
        device=first_plane_crossing_xy_m.device,
    )
    facts_token = _identity_token(
        facts_task_identity_token,
        name="facts_task_identity_token",
        batch_size=batch_size,
        device=first_plane_crossing_xy_m.device,
    )
    for name, tensor in (
        ("expected_target_xy_m", expected_target_xy_m),
        ("facts_target_xy_m", facts_target_xy_m),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != first_plane_crossing_xy_m.shape
            or tensor.dtype != first_plane_crossing_xy_m.dtype
            or tensor.device != first_plane_crossing_xy_m.device
        ):
            raise ValueError(
                f"{name} must match crossing floating shape, dtype, and device"
            )
    contact = _bool_vector(contact_valid, name="contact_valid", batch_size=batch_size)
    present = _bool_vector(
        first_plane_crossing_present,
        name="first_plane_crossing_present",
        batch_size=batch_size,
    )
    claimed_first = _bool_vector(
        first_plane_crossing_valid,
        name="first_plane_crossing_valid",
        batch_size=batch_size,
    )
    claimed_nonfinite = _bool_vector(
        first_plane_crossing_nonfinite,
        name="first_plane_crossing_nonfinite",
        batch_size=batch_size,
    )
    crossed = _bool_vector(
        ball_center_net_crossed,
        name="ball_center_net_crossed",
        batch_size=batch_size,
    )
    clear = _bool_vector(
        ball_center_net_clear,
        name="ball_center_net_clear",
        batch_size=batch_size,
    )
    for name, tensor in (
        ("contact_valid", contact),
        ("first_plane_crossing_present", present),
        ("first_plane_crossing_valid", claimed_first),
        ("first_plane_crossing_nonfinite", claimed_nonfinite),
        ("ball_center_net_crossed", crossed),
        ("ball_center_net_clear", clear),
    ):
        if tensor.device != first_plane_crossing_xy_m.device:
            raise ValueError(f"{name} must share the crossing tensor device")

    finite = torch.isfinite(first_plane_crossing_xy_m).all(dim=-1)
    nonfinite = claimed_nonfinite | ~finite
    crossing_contract_fault = claimed_first & ~present & ~nonfinite
    expected_target_finite = torch.isfinite(expected_target_xy_m).all(dim=-1)
    facts_target_finite = torch.isfinite(facts_target_xy_m).all(dim=-1)
    zero_target = torch.zeros_like(expected_target_xy_m)
    expected_target = torch.where(
        expected_target_finite.unsqueeze(-1),
        expected_target_xy_m,
        zero_target,
    )
    facts_target = torch.where(
        facts_target_finite.unsqueeze(-1),
        facts_target_xy_m,
        zero_target,
    )
    expected_x = expected_target[:, 0]
    expected_y = expected_target[:, 1]
    facts_x = facts_target[:, 0]
    facts_y = facts_target[:, 1]
    expected_target_valid = (
        expected_target_finite
        & (expected_x >= profile.opponent_table_x_min_m)
        & (expected_x <= profile.opponent_table_x_max_m)
        & (expected_y >= profile.table_y_min_m)
        & (expected_y <= profile.table_y_max_m)
    )
    facts_target_valid = (
        facts_target_finite
        & (facts_x >= profile.opponent_table_x_min_m)
        & (facts_x <= profile.opponent_table_x_max_m)
        & (facts_y >= profile.table_y_min_m)
        & (facts_y <= profile.table_y_max_m)
    )
    target_valid = expected_target_valid & facts_target_valid
    task_target_match = torch.eq(expected_target, facts_target).all(dim=-1)
    task_identity_match = torch.eq(expected_token, facts_token).all(dim=-1)
    task_identity_fault = (
        ~target_valid | ~task_target_match | ~task_identity_match
    )
    first_valid = claimed_first & present & ~nonfinite & ~crossing_contract_fault
    drain_fault = task_identity_fault | crossing_contract_fault
    target = expected_target
    safe_xy = torch.where(
        finite.unsqueeze(-1),
        first_plane_crossing_xy_m,
        target,
    )
    x = safe_xy[:, 0]
    y = safe_xy[:, 1]
    opponent_bound = first_valid & ~drain_fault & (x > profile.net_x_m)
    on_table = (
        opponent_bound
        & (x >= profile.opponent_table_x_min_m)
        & (x <= profile.opponent_table_x_max_m)
        & (y >= profile.table_y_min_m)
        & (y <= profile.table_y_max_m)
    )
    net_ok = crossed & clear

    error = torch.linalg.vector_norm(safe_xy - target, dim=-1)
    broad = 1.0 / (
        1.0 + torch.square(error / profile.sigma_broad_m)
    )
    narrow = torch.exp(-torch.square(error / profile.sigma_narrow_m))
    blended = (
        profile.alpha_broad * broad
        + (1.0 - profile.alpha_broad) * narrow
    )
    zero = torch.zeros_like(error)
    kernel_valid = first_valid & ~drain_fault
    broad = torch.where(kernel_valid, broad, zero)
    narrow = torch.where(kernel_valid, narrow, zero)
    blended = torch.where(kernel_valid, blended, zero)
    error = torch.where(kernel_valid, error, zero)

    eligible = contact & first_valid & ~drain_fault & net_ok & opponent_bound
    gate = torch.where(
        eligible,
        torch.where(
            on_table,
            torch.full_like(error, profile.on_table_gate),
            torch.full_like(error, profile.off_table_gate),
        ),
        zero,
    )
    total = gate * blended

    reason_code = torch.full(
        (batch_size,),
        REASON_TO_CODE["scored_off_table"],
        dtype=torch.int64,
        device=first_plane_crossing_xy_m.device,
    )
    reason_code = torch.where(
        on_table,
        torch.full_like(reason_code, REASON_TO_CODE["scored_on_table"]),
        reason_code,
    )
    reason_code = torch.where(
        ~opponent_bound,
        torch.full_like(reason_code, REASON_TO_CODE["not_opponent_bound"]),
        reason_code,
    )
    reason_code = torch.where(
        ~clear,
        torch.full_like(reason_code, REASON_TO_CODE["net_not_clear"]),
        reason_code,
    )
    reason_code = torch.where(
        ~crossed,
        torch.full_like(reason_code, REASON_TO_CODE["net_not_crossed"]),
        reason_code,
    )
    reason_code = torch.where(
        ~first_valid,
        torch.full_like(reason_code, REASON_TO_CODE["no_crossing"]),
        reason_code,
    )
    # Apply primary reasons from lowest to highest precedence so this batched
    # mirror exactly matches LandingPlacementScore/score_landing_placement:
    # no_contact > nonfinite > crossing_contract_fault > no_crossing > ... .
    # The negative producer code is an infrastructure code and is deliberately
    # never emitted by the canonical C04 scorer.
    reason_code = torch.where(
        crossing_contract_fault,
        torch.full_like(
            reason_code, REASON_TO_CODE["crossing_contract_fault"]
        ),
        reason_code,
    )
    reason_code = torch.where(
        nonfinite,
        torch.full_like(reason_code, REASON_TO_CODE["nonfinite"]),
        reason_code,
    )
    reason_code = torch.where(
        ~contact,
        torch.full_like(reason_code, REASON_TO_CODE["no_contact"]),
        reason_code,
    )
    reason_code = torch.where(
        task_identity_fault,
        torch.full_like(reason_code, TASK_IDENTITY_FAULT_CODE),
        reason_code,
    )
    return LandingPlacementTorchScore(
        contact_valid=contact,
        target_valid=target_valid,
        task_target_match=task_target_match,
        task_identity_match=task_identity_match,
        task_identity_fault=task_identity_fault,
        drain_fault=drain_fault,
        first_plane_crossing_present=present,
        first_plane_crossing_valid=first_valid,
        first_plane_crossing_nonfinite=nonfinite,
        first_plane_crossing_contract_fault=crossing_contract_fault,
        ball_center_net_crossed=crossed,
        ball_center_net_clear=clear,
        opponent_bound=opponent_bound,
        on_opponent_table=on_table,
        reason_code=reason_code,
        placement_error_m=error,
        broad_kernel=broad,
        narrow_kernel=narrow,
        blended_kernel=blended,
        table_gate=gate,
        total=total,
    )


__all__ = (
    "LandingPlacementTorchScore",
    "PRODUCER_CONTRACT_FAULT_CODE",
    "REASON_TO_CODE",
    "TASK_IDENTITY_FAULT_CODE",
    "TASK_IDENTITY_TOKEN_BYTES",
    "score_landing_placement_torch",
)
