"""Engine-neutral lifecycle part of the FullMDP Reward20 graph.

The kernel consumes only the four canonical ActionEpoch owner fact rows in
``(physical, R03, R06, R07)`` order.  It owns no Isaac or MuJoCo object and
does not manufacture task identity, contact, landing, or recovery evidence.
Absent or invalid facts therefore pay finite zero while real fact producers
remain responsible for the denominators.
"""

from __future__ import annotations

import math

import torch


OWNER_COUNT = 4
OWNER_FACT_F32_WIDTH = 32
LIFECYCLE_TERM_COUNT = 14

R03_PRESENT = 1 << 0
R03_PHYSICALLY_VALID = 1 << 1
PHYSICAL_PRESENT = 1 << 0
PHYSICAL_SELECTED_CONTACT = 1 << 1
R06_PRESENT = 1 << 0
R06_POLICY_ELIGIBLE = 1 << 1
R06_SOURCE_VALID = 1 << 2
R07_PRESENT = 1 << 0
R07_NUMERICALLY_VALID = 1 << 1

R03_REWARD_SPECS = (
    ("racket_position", 0.2, False),
    ("racket_velocity", 1.0, False),
    ("racket_normal", 0.5, False),
    ("racket_position_coarse", 0.5, True),
    ("racket_velocity_coarse", 2.0, True),
    ("racket_normal_coarse", 1.0, True),
    ("racket_position_precision", 0.1, False),
    ("racket_velocity_precision", 0.5, False),
    ("racket_normal_precision", 0.25, False),
    ("paddle_center_proximity", 0.15, True),
)
LIFECYCLE_WEIGHTS = (1.0,) * 11 + (20.0, 1.0, 1.0)


def _require_inputs(
    valid_bits: torch.Tensor,
    fact_f32: torch.Tensor,
    owner_fault_bits: torch.Tensor,
) -> int:
    if (
        not isinstance(valid_bits, torch.Tensor)
        or valid_bits.dtype != torch.int64
        or valid_bits.ndim != 2
        or valid_bits.shape[1] != OWNER_COUNT
    ):
        raise ValueError("valid_bits must have int64 shape [N,4]")
    batch = int(valid_bits.shape[0])
    if (
        not isinstance(fact_f32, torch.Tensor)
        or fact_f32.shape != (batch, OWNER_COUNT, OWNER_FACT_F32_WIDTH)
        or not torch.is_floating_point(fact_f32)
        or fact_f32.device != valid_bits.device
    ):
        raise ValueError("fact_f32 must have device-local floating shape [N,4,32]")
    if (
        not isinstance(owner_fault_bits, torch.Tensor)
        or owner_fault_bits.dtype != torch.int64
        or owner_fault_bits.shape != valid_bits.shape
        or owner_fault_bits.device != valid_bits.device
    ):
        raise ValueError("owner_fault_bits must match valid_bits")
    return batch


def lifecycle_reward14(
    *,
    valid_bits: torch.Tensor,
    fact_f32: torch.Tensor,
    owner_fault_bits: torch.Tensor,
    step_dt: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return configured Reward20 ordinals 0..13 as ``[N,14]``.

    The packing and math match ``action_ball_full_mdp_lean_rewards``.  This
    function deliberately does not return a success verdict: a row can be a
    valid learning sample with zero payment.  The MuJoCo training lane supplies
    its construction-cached device ``weights`` and attributes any nonfinite
    result in the rollout ledger before the optimizer boundary.  Omitting
    ``weights`` retains the prior finite fail-fast behavior for non-hot
    diagnostic callers.
    """

    batch = _require_inputs(valid_bits, fact_f32, owner_fault_bits)
    dt = float(step_dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    raw = torch.zeros((batch, LIFECYCLE_TERM_COUNT), dtype=fact_f32.dtype, device=fact_f32.device)

    r03_bits = valid_bits[:, 1]
    r03 = fact_f32[:, 1]
    r03_admitted = (
        torch.bitwise_and(r03_bits, R03_PRESENT).ne(0)
        & torch.bitwise_and(r03_bits, R03_PHYSICALLY_VALID).ne(0)
        & owner_fault_bits[:, 1].eq(0)
    )
    target_position, target_velocity, target_normal = r03[:, 0:3], r03[:, 3:6], r03[:, 6:9]
    ball_position = r03[:, 9:12]
    achieved_position, achieved_velocity, achieved_normal = r03[:, 15:18], r03[:, 18:21], r03[:, 21:24]
    position_error = torch.linalg.vector_norm(achieved_position - target_position, dim=-1)
    velocity_error = torch.linalg.vector_norm(achieved_velocity - target_velocity, dim=-1)
    normal_cosine = torch.sum(achieved_normal * target_normal, dim=-1).clamp(-1.0, 1.0)
    normal_error = torch.acos(normal_cosine)
    paddle_center_error = torch.linalg.vector_norm(achieved_position - ball_position, dim=-1)
    for ordinal, (name, scale, reciprocal) in enumerate(R03_REWARD_SPECS):
        if name == "paddle_center_proximity":
            error = paddle_center_error
        elif "position" in name:
            error = position_error
        elif "velocity" in name:
            error = velocity_error
        else:
            error = normal_error
        finite = torch.isfinite(error)
        clean = torch.where(finite, error, torch.zeros_like(error))
        ratio_sq = torch.square(clean / scale)
        value = torch.reciprocal(1.0 + ratio_sq) if reciprocal else torch.exp(-ratio_sq)
        raw[:, ordinal] = torch.where(r03_admitted & finite, value, torch.zeros_like(value))

    physical_bits = valid_bits[:, 0]
    raw[:, 10] = (
        torch.bitwise_and(physical_bits, PHYSICAL_PRESENT).ne(0)
        & torch.bitwise_and(physical_bits, PHYSICAL_SELECTED_CONTACT).ne(0)
        & owner_fault_bits[:, 0].eq(0)
    ).to(fact_f32.dtype)

    r06_bits = valid_bits[:, 2]
    r06 = fact_f32[:, 2]
    r06_eligible = (
        torch.bitwise_and(r06_bits, R06_PRESENT).ne(0)
        & torch.bitwise_and(r06_bits, R06_POLICY_ELIGIBLE).ne(0)
        & torch.bitwise_and(r06_bits, R06_SOURCE_VALID).ne(0)
        & owner_fault_bits[:, 2].eq(0)
    )
    r06_common = r06[:, 0]
    r06_guidance = r06[:, 1] * r06[:, 2]
    raw[:, 11] = torch.where(r06_eligible & torch.isfinite(r06_common), r06_common, torch.zeros_like(r06_common))
    raw[:, 12] = torch.where(r06_eligible & torch.isfinite(r06_guidance), r06_guidance, torch.zeros_like(r06_guidance))

    r07_bits = valid_bits[:, 3]
    r07 = fact_f32[:, 3]
    r07_finite = torch.isfinite(r07[:, 0]) & torch.isfinite(r07[:, 1]) & torch.isfinite(r07[:, 7:20]).all(dim=1)
    r07_eligible = (
        torch.bitwise_and(r07_bits, R07_PRESENT).ne(0)
        & torch.bitwise_and(r07_bits, R07_NUMERICALLY_VALID).ne(0)
        & r07[:, 2].eq(1.0)
        & r07[:, 3].eq(1.0)
        & r07[:, 4].eq(0.0)
        & owner_fault_bits[:, 3].eq(0)
    )
    raw[:, 13] = torch.where(r07_eligible & r07_finite, r07[:, 0], torch.zeros_like(r07[:, 0]))

    cached_weights = weights is not None
    if not cached_weights:
        weights = fact_f32.new_tensor(LIFECYCLE_WEIGHTS)
    elif (
        not isinstance(weights, torch.Tensor)
        or weights.dtype != fact_f32.dtype
        or weights.device != fact_f32.device
        or weights.shape != (LIFECYCLE_TERM_COUNT,)
        or not weights.is_contiguous()
    ):
        raise ValueError("weights must be contiguous [14] matching fact_f32")
    configured = raw * weights * dt
    if not cached_weights and not bool(torch.isfinite(configured).all()):
        raise RuntimeError("portable lifecycle Reward14 is nonfinite")
    return configured


__all__ = (
    "LIFECYCLE_TERM_COUNT",
    "LIFECYCLE_WEIGHTS",
    "OWNER_COUNT",
    "OWNER_FACT_F32_WIDTH",
    "PHYSICAL_PRESENT",
    "PHYSICAL_SELECTED_CONTACT",
    "R03_PHYSICALLY_VALID",
    "R03_PRESENT",
    "R03_REWARD_SPECS",
    "R06_POLICY_ELIGIBLE",
    "R06_PRESENT",
    "R06_SOURCE_VALID",
    "R07_NUMERICALLY_VALID",
    "R07_PRESENT",
    "lifecycle_reward14",
)
