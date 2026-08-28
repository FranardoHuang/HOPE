#!/usr/bin/env python3
"""Pure-Torch Physical question producer for the fresh Device-R05 path.

This module implements only the numerical half of the two-stage contract:

1. discover the largest complete Motion-tick final ballistic segment by
   reverse integrating the candidate contact state with a fixed loop bound;
2. after an *external* Motion owner has chosen exact launch/contact ticks,
   select the retained CUDA prefix state (or use the non-CUDA reference
   recomputation) and form the 13-field Physical state (position, identity
   quaternion, linear velocity, spin).

The test owner below keeps candidate inputs and discovered horizons behind an
opaque one-shot receipt.  Consequently a caller cannot replace
``t_effective`` or attach another candidate's horizon to a launch.  It is not
production authority: the real Motion exact-contact/launch chronology ABI is
not frozen, so :func:`construct_production_physical_question_producer` always
raises ``PhysicalQuestionProductionHold``.

There is no scene handle in this module.  Reveal retains data; only the later
Physical launch owner may publish the selected 13-state to the scene.

CUDA discovery kill switch: ``HOPE_PHYSICAL_QUESTION_FAST=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import struct
from threading import Lock
from typing import Optional, Tuple
import warnings
import weakref

import torch


INTEGRATION_STATUS = "physical_question_numeric_core_motion_chronology_hold"
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_UNAUTHORIZED = True

PHYSICAL_STATE_F32_FIELDS: Tuple[str, ...] = (
    "position_env_m_x",
    "position_env_m_y",
    "position_env_m_z",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "linear_velocity_world_mps_x",
    "linear_velocity_world_mps_y",
    "linear_velocity_world_mps_z",
    "angular_velocity_world_radps_x",
    "angular_velocity_world_radps_y",
    "angular_velocity_world_radps_z",
)

# These values occupy Device-R05's -1 admitted / 0..13 rejected domain; the
# question-owned full-suffix schedule reason is 13 and is outside this leaf.
CONSTRUCTION_REASON_ADMITTED = -1
CONSTRUCTION_REASON_NO_COMPLETE_FINAL_SEGMENT = 0
CONSTRUCTION_REASON_CANDIDATE_BINDING_DIFFERS = 1
CONSTRUCTION_REASON_EXACT_CHRONOLOGY_INVALID = 2
CONSTRUCTION_REASON_RECOMPUTE_INVALID = 3
CONSTRUCTION_REASON_INVALID_PRODUCER = 12

# Row-level infrastructure facts.  Ordinary physical infeasibility (no whole
# final-segment tick) is a construction rejection and does not set a fault.
PRODUCER_FAULT_NONFINITE_INPUT = 1 << 40
PRODUCER_FAULT_INVALID_CANDIDATE_IDENTITY = 1 << 41
PRODUCER_FAULT_CANDIDATE_BINDING_DIFFERS = 1 << 42
PRODUCER_FAULT_EXACT_CHRONOLOGY_INVALID = 1 << 43
PRODUCER_FAULT_RECOMPUTE_INVALID = 1 << 44

_OWNER_CONSTRUCTION_KEY = object()
_BOUND_DIAGNOSTIC_PHYSICAL_OWNERS: weakref.WeakKeyDictionary[object, object] = (
    weakref.WeakKeyDictionary()
)
_BOUND_DIAGNOSTIC_PHYSICAL_OWNERS_LOCK = Lock()


class PhysicalQuestionError(RuntimeError):
    """Base error for the diagnostic Physical question producer."""


class PhysicalQuestionConflictError(PhysicalQuestionError):
    """A receipt was foreign, replayed, or attached to another candidate."""


class PhysicalQuestionProductionHold(PhysicalQuestionError):
    """Production construction lacks the exact Motion chronology owner."""


def _positive_exact_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise PhysicalQuestionError(f"{label} must be a positive exact int")
    return value


def _tensor_row_count(
    value: object,
    *,
    label: str,
    rank: int,
    trailing_shape: tuple[int, ...],
    device: torch.device,
) -> int:
    if (
        type(value) is not torch.Tensor
        or value.ndim != rank
        or tuple(value.shape[1:]) != trailing_shape
        or value.device != device
    ):
        raise PhysicalQuestionError(f"{label} construction tensor ABI differs")
    return _positive_exact_int(value.shape[0], label=f"{label} row count")


@dataclass(frozen=True)
class PhysicalQuestionFlightParams:
    """Exact flight constants needed by the reverse/forward RK4 mirror."""

    k_d: float
    k_m: float
    g: float
    ball_radius_m: float

    def __post_init__(self) -> None:
        for name in ("k_d", "k_m", "g", "ball_radius_m"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(float(value)):
                raise PhysicalQuestionError(f"{name} must be finite")
        if self.k_d < 0.0 or self.g <= 0.0 or self.ball_radius_m <= 0.0:
            raise PhysicalQuestionError("flight constants are outside their domain")


@dataclass(frozen=True)
class PhysicalQuestionNumericConfig:
    """Static loop/tick geometry; all validation is host-only at construction."""

    motion_tick_s: float
    integration_substeps_per_motion_tick: int
    max_final_segment_motion_ticks: int
    table_surface_z_m: float
    table_clearance_margin_m: float = 5.0e-3
    reverse_speed_cap_mps: float = 40.0

    def __post_init__(self) -> None:
        finite_names = (
            "motion_tick_s",
            "table_surface_z_m",
            "table_clearance_margin_m",
            "reverse_speed_cap_mps",
        )
        for name in finite_names:
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(float(value)):
                raise PhysicalQuestionError(f"{name} must be finite")
        if self.motion_tick_s <= 0.0:
            raise PhysicalQuestionError("motion_tick_s must be positive")
        if (
            type(self.integration_substeps_per_motion_tick) is not int
            or self.integration_substeps_per_motion_tick < 1
        ):
            raise PhysicalQuestionError(
                "integration_substeps_per_motion_tick must be a positive exact int"
            )
        if (
            type(self.max_final_segment_motion_ticks) is not int
            or self.max_final_segment_motion_ticks < 1
        ):
            raise PhysicalQuestionError(
                "max_final_segment_motion_ticks must be a positive exact int"
            )
        if self.table_clearance_margin_m < 0.0:
            raise PhysicalQuestionError("table clearance margin must be nonnegative")
        if self.reverse_speed_cap_mps <= 0.0:
            raise PhysicalQuestionError("reverse speed cap must be positive")


@dataclass(frozen=True)
class PhysicalQuestionCandidateBatch:
    """Candidate-keyed contact states before horizon discovery."""

    candidate_identity: torch.Tensor
    contact_position_env_m: torch.Tensor
    incoming_linear_velocity_world_mps: torch.Tensor
    incoming_angular_velocity_world_radps: torch.Tensor


class PhysicalQuestionHorizonReceipt:
    """Empty one-shot identity binding candidate bytes to discovered horizons."""

    __slots__ = ("__weakref__",)

    def __new__(cls):
        del cls
        raise TypeError("Physical question horizon receipts are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Physical question horizon receipts are immutable")

    def __copy__(self):
        raise TypeError("Physical question horizon receipts cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("Physical question horizon receipts cannot be copied")

    def __reduce__(self):
        raise TypeError("Physical question horizon receipts cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("Physical question horizon receipts cannot be serialized")


@dataclass(frozen=True)
class PhysicalQuestionHorizonView:
    """Clone-only discovery view for the eventual exact Motion tick owner."""

    candidate_identity: torch.Tensor
    max_feasible_motion_ticks: torch.Tensor
    max_feasible_horizon_s: torch.Tensor
    construction_reason: torch.Tensor
    producer_fault: torch.Tensor


@dataclass(frozen=True)
class PhysicalQuestionFinalBatch:
    """Exact-tick launch state; construction_reason is the sole admission fact."""

    candidate_identity: torch.Tensor
    contact_tick: torch.Tensor
    launch_tick: torch.Tensor
    construction_reason: torch.Tensor
    physical_state_f32: torch.Tensor
    effective_contact_horizon_s: torch.Tensor
    producer_fault: torch.Tensor


@dataclass(frozen=True)
class _DiscoveryRecord:
    candidate_identity: torch.Tensor
    contact_position_env_m: torch.Tensor
    incoming_linear_velocity_world_mps: torch.Tensor
    incoming_angular_velocity_world_radps: torch.Tensor
    max_feasible_motion_ticks: torch.Tensor
    construction_reason: torch.Tensor
    producer_fault: torch.Tensor
    motion_tick_state_f32: Optional[torch.Tensor] = None


def _require_candidate_batch(batch: object) -> tuple[torch.device, tuple[int, ...]]:
    if type(batch) is not PhysicalQuestionCandidateBatch:
        raise PhysicalQuestionError("candidate batch type differs")
    candidate = batch.candidate_identity
    if (
        type(candidate) is not torch.Tensor
        or candidate.dtype is not torch.int64
        or candidate.ndim < 1
        or candidate.numel() < 1
        or not candidate.is_contiguous()
    ):
        raise PhysicalQuestionError(
            "candidate_identity must be a contiguous rank>=1 int64 tensor"
        )
    shape = tuple(candidate.shape)
    device = candidate.device
    for name in (
        "contact_position_env_m",
        "incoming_linear_velocity_world_mps",
        "incoming_angular_velocity_world_radps",
    ):
        value = getattr(batch, name)
        if (
            type(value) is not torch.Tensor
            or value.device != device
            or value.dtype is not torch.float32
            or tuple(value.shape) != (*shape, 3)
            or not value.is_contiguous()
        ):
            raise PhysicalQuestionError(
                f"{name} must be contiguous float32 on {device} with shape {(*shape, 3)}"
            )
    return device, shape


def _flight_accel(
    velocity: torch.Tensor,
    omega: torch.Tensor,
    params: PhysicalQuestionFlightParams,
) -> torch.Tensor:
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    accel = (
        -float(params.k_d) * speed * velocity
        + float(params.k_m) * torch.cross(omega, velocity, dim=-1)
    )
    gravity = torch.zeros_like(accel)
    gravity[..., 2] = -float(params.g)
    return accel + gravity


def _rk4_step(
    position: torch.Tensor,
    velocity: torch.Tensor,
    omega: torch.Tensor,
    step_s: float,
    params: PhysicalQuestionFlightParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    a1 = _flight_accel(velocity, omega, params)
    a2 = _flight_accel(velocity + 0.5 * step_s * a1, omega, params)
    a3 = _flight_accel(velocity + 0.5 * step_s * a2, omega, params)
    a4 = _flight_accel(velocity + step_s * a3, omega, params)
    velocity_new = velocity + (step_s / 6.0) * (
        a1 + 2.0 * a2 + 2.0 * a3 + a4
    )
    position_new = position + (step_s / 6.0) * (
        velocity
        + 2.0 * (velocity + 0.5 * step_s * a1)
        + 2.0 * (velocity + 0.5 * step_s * a2)
        + (velocity + step_s * a3)
    )
    return position_new, velocity_new


def _prepare_discovery(batch: PhysicalQuestionCandidateBatch):
    _, shape = _require_candidate_batch(batch)
    candidate = batch.candidate_identity.clone()
    contact = batch.contact_position_env_m.clone()
    incoming = batch.incoming_linear_velocity_world_mps.clone()
    omega = batch.incoming_angular_velocity_world_radps.clone()

    finite_input = (
        torch.all(torch.isfinite(contact), dim=-1)
        & torch.all(torch.isfinite(incoming), dim=-1)
        & torch.all(torch.isfinite(omega), dim=-1)
    )
    flat_candidate = candidate.reshape(-1)
    sorted_candidate, permutation = torch.sort(flat_candidate)
    duplicate_sorted = torch.zeros_like(sorted_candidate, dtype=torch.bool)
    adjacent_equal = sorted_candidate[1:].eq(sorted_candidate[:-1])
    duplicate_sorted[1:] |= adjacent_equal
    duplicate_sorted[:-1] |= adjacent_equal
    duplicate_flat = torch.zeros_like(flat_candidate, dtype=torch.bool)
    duplicate_flat.scatter_(0, permutation, duplicate_sorted)
    identity_valid = (candidate > 0) & ~duplicate_flat.reshape(shape)
    input_valid = finite_input & identity_valid
    safe_contact = torch.where(input_valid.unsqueeze(-1), contact, torch.zeros_like(contact))
    safe_incoming = torch.where(input_valid.unsqueeze(-1), incoming, torch.zeros_like(incoming))
    safe_omega = torch.where(input_valid.unsqueeze(-1), omega, torch.zeros_like(omega))
    return (
        shape,
        candidate,
        safe_contact,
        safe_incoming,
        safe_omega,
        finite_input,
        identity_valid,
        input_valid,
    )


def _finish_discovery(
    prepared,
    max_ticks: torch.Tensor,
    tick_states: Optional[torch.Tensor],
) -> _DiscoveryRecord:
    (
        shape,
        candidate,
        safe_contact,
        safe_incoming,
        safe_omega,
        finite_input,
        identity_valid,
        input_valid,
    ) = prepared
    producer_fault = torch.zeros(shape, dtype=torch.int64, device=candidate.device)
    producer_fault = torch.where(
        ~finite_input,
        torch.bitwise_or(
            producer_fault,
            torch.full_like(producer_fault, PRODUCER_FAULT_NONFINITE_INPUT),
        ),
        producer_fault,
    )
    producer_fault = torch.where(
        ~identity_valid,
        torch.bitwise_or(
            producer_fault,
            torch.full_like(
                producer_fault, PRODUCER_FAULT_INVALID_CANDIDATE_IDENTITY
            ),
        ),
        producer_fault,
    ).contiguous()
    construction_reason = torch.full(
        shape,
        CONSTRUCTION_REASON_ADMITTED,
        dtype=torch.int64,
        device=candidate.device,
    )
    construction_reason = torch.where(
        input_valid & max_ticks.eq(0),
        torch.full_like(
            construction_reason, CONSTRUCTION_REASON_NO_COMPLETE_FINAL_SEGMENT
        ),
        construction_reason,
    )
    construction_reason = torch.where(
        ~input_valid,
        torch.full_like(construction_reason, CONSTRUCTION_REASON_INVALID_PRODUCER),
        construction_reason,
    ).contiguous()
    return _DiscoveryRecord(
        candidate_identity=candidate,
        contact_position_env_m=safe_contact.contiguous(),
        incoming_linear_velocity_world_mps=safe_incoming.contiguous(),
        incoming_angular_velocity_world_radps=safe_omega.contiguous(),
        max_feasible_motion_ticks=max_ticks.contiguous(),
        construction_reason=construction_reason,
        producer_fault=producer_fault,
        motion_tick_state_f32=tick_states,
    )


@torch.no_grad()
def _discover_horizon_reference(
    batch: PhysicalQuestionCandidateBatch,
    *,
    params: PhysicalQuestionFlightParams,
    config: PhysicalQuestionNumericConfig,
    _retain_motion_tick_state: bool = False,
) -> _DiscoveryRecord:
    (
        shape,
        candidate,
        safe_contact,
        safe_incoming,
        safe_omega,
        finite_input,
        identity_valid,
        input_valid,
    ) = _prepare_discovery(batch)

    position = safe_contact
    velocity = safe_incoming
    alive = input_valid
    max_ticks = torch.zeros(shape, dtype=torch.int64, device=candidate.device)
    step_s = -float(config.motion_tick_s) / float(
        config.integration_substeps_per_motion_tick
    )
    z_min = (
        float(config.table_surface_z_m)
        + float(params.ball_radius_m)
        + float(config.table_clearance_margin_m)
    )
    motion_tick_states = [] if _retain_motion_tick_state else None

    # A failed partial policy tick is rolled back.  Motion therefore receives
    # a count of complete exact ticks, never a fractional scheduling hint.
    for _ in range(config.max_final_segment_motion_ticks):
        tick_position = position
        tick_velocity = velocity
        work_position = position
        work_velocity = velocity
        tick_valid = alive
        for _ in range(config.integration_substeps_per_motion_tick):
            proposed_position, proposed_velocity = _rk4_step(
                work_position, work_velocity, safe_omega, step_s, params
            )
            finite_state = (
                torch.all(torch.isfinite(proposed_position), dim=-1)
                & torch.all(torch.isfinite(proposed_velocity), dim=-1)
            )
            within_final_segment = proposed_position[..., 2] > z_min
            within_speed = (
                torch.linalg.vector_norm(proposed_velocity, dim=-1)
                < float(config.reverse_speed_cap_mps)
            )
            substep_valid = tick_valid & finite_state & within_final_segment & within_speed
            work_position = torch.where(
                substep_valid.unsqueeze(-1), proposed_position, work_position
            )
            work_velocity = torch.where(
                substep_valid.unsqueeze(-1), proposed_velocity, work_velocity
            )
            tick_valid = substep_valid
        position = torch.where(tick_valid.unsqueeze(-1), work_position, tick_position)
        velocity = torch.where(tick_valid.unsqueeze(-1), work_velocity, tick_velocity)
        max_ticks = max_ticks + tick_valid.to(torch.int64)
        alive = tick_valid
        if motion_tick_states is not None:
            motion_tick_states.append(torch.cat((position, velocity), dim=-1))

    tick_states = (
        torch.stack(motion_tick_states, dim=-2).contiguous()
        if motion_tick_states is not None
        else None
    )
    return _finish_discovery(
        (
            shape,
            candidate,
            safe_contact,
            safe_incoming,
            safe_omega,
            finite_input,
            identity_valid,
            input_valid,
        ),
        max_ticks,
        tick_states,
    )


_FAST_ENV = "HOPE_PHYSICAL_QUESTION_FAST"
_FUSED_BLOCK = 128
_fused_kernel = None
_parity_cache: dict = {}


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _build_fused_kernel():
    global _fused_kernel
    if _fused_kernel is not None:
        return _fused_kernel or None
    try:
        import triton as _triton_mod
        import triton.language as _tl_mod
    except Exception:
        _fused_kernel = False
        return None

    globals()["triton"] = triton = _triton_mod
    globals()["tl"] = tl = _tl_mod

    @triton.jit
    def _pq_accel(vx, vy, vz, wx, wy, wz, kd, km, gravity):
        speed = tl.math.sqrt_rn((vx * vx + vz * vz) + vy * vy)
        drag = (-kd) * speed
        cx = tl.math.fma(wy, vz, -(wz * vy))
        cy = tl.math.fma(wz, vx, -(wx * vz))
        cz = tl.math.fma(wx, vy, -(wy * vx))
        return (
            drag * vx + km * cx,
            drag * vy + km * cy,
            (drag * vz + km * cz) - gravity,
        )

    globals()["_pq_accel"] = _pq_accel

    @triton.jit
    def _pq_rk4(px, py, pz, vx, vy, vz, wx, wy, wz,
                kd, km, gravity, half_step, full_step, sixth_step):
        a1x, a1y, a1z = _pq_accel(vx, vy, vz, wx, wy, wz, kd, km, gravity)
        v2x, v2y, v2z = (
            vx + half_step * a1x,
            vy + half_step * a1y,
            vz + half_step * a1z,
        )
        a2x, a2y, a2z = _pq_accel(v2x, v2y, v2z, wx, wy, wz, kd, km, gravity)
        v3x, v3y, v3z = (
            vx + half_step * a2x,
            vy + half_step * a2y,
            vz + half_step * a2z,
        )
        a3x, a3y, a3z = _pq_accel(v3x, v3y, v3z, wx, wy, wz, kd, km, gravity)
        v4x, v4y, v4z = (
            vx + full_step * a3x,
            vy + full_step * a3y,
            vz + full_step * a3z,
        )
        a4x, a4y, a4z = _pq_accel(v4x, v4y, v4z, wx, wy, wz, kd, km, gravity)
        next_vx = vx + sixth_step * (((a1x + 2.0 * a2x) + 2.0 * a3x) + a4x)
        next_vy = vy + sixth_step * (((a1y + 2.0 * a2y) + 2.0 * a3y) + a4y)
        next_vz = vz + sixth_step * (((a1z + 2.0 * a2z) + 2.0 * a3z) + a4z)
        next_px = px + sixth_step * (((vx + 2.0 * v2x) + 2.0 * v3x) + v4x)
        next_py = py + sixth_step * (((vy + 2.0 * v2y) + 2.0 * v3y) + v4y)
        next_pz = pz + sixth_step * (((vz + 2.0 * v2z) + 2.0 * v3z) + v4z)
        return next_px, next_py, next_pz, next_vx, next_vy, next_vz

    globals()["_pq_rk4"] = _pq_rk4

    @triton.jit
    def _pq_discover(position_ptr, velocity_ptr, omega_ptr, valid_ptr,
                     states_ptr, max_ticks_ptr, n_rows, kd, km, gravity,
                     half_step, full_step, sixth_step, z_min, speed_cap,
                     finite_max, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_rows
        base = offsets * 3
        px = tl.load(position_ptr + base + 0, mask=mask, other=0.0)
        py = tl.load(position_ptr + base + 1, mask=mask, other=0.0)
        pz = tl.load(position_ptr + base + 2, mask=mask, other=0.0)
        vx = tl.load(velocity_ptr + base + 0, mask=mask, other=0.0)
        vy = tl.load(velocity_ptr + base + 1, mask=mask, other=0.0)
        vz = tl.load(velocity_ptr + base + 2, mask=mask, other=0.0)
        wx = tl.load(omega_ptr + base + 0, mask=mask, other=0.0)
        wy = tl.load(omega_ptr + base + 1, mask=mask, other=0.0)
        wz = tl.load(omega_ptr + base + 2, mask=mask, other=0.0)
        alive = tl.load(valid_ptr + offsets, mask=mask, other=0) != 0
        max_ticks = tl.zeros((BLOCK,), tl.int32)

        for tick in range(30):
            tick_px, tick_py, tick_pz = px, py, pz
            tick_vx, tick_vy, tick_vz = vx, vy, vz
            work_px, work_py, work_pz = px, py, pz
            work_vx, work_vy, work_vz = vx, vy, vz
            tick_valid = alive
            for _ in range(4):
                next_px, next_py, next_pz, next_vx, next_vy, next_vz = _pq_rk4(
                    work_px, work_py, work_pz,
                    work_vx, work_vy, work_vz,
                    wx, wy, wz, kd, km, gravity,
                    half_step, full_step, sixth_step,
                )
                finite = (
                    (tl.abs(next_px) <= finite_max)
                    & (tl.abs(next_py) <= finite_max)
                    & (tl.abs(next_pz) <= finite_max)
                    & (tl.abs(next_vx) <= finite_max)
                    & (tl.abs(next_vy) <= finite_max)
                    & (tl.abs(next_vz) <= finite_max)
                )
                speed = tl.math.sqrt_rn(
                    (next_vx * next_vx + next_vz * next_vz) + next_vy * next_vy
                )
                valid = tick_valid & finite & (next_pz > z_min) & (speed < speed_cap)
                work_px, work_py, work_pz = (
                    tl.where(valid, next_px, work_px),
                    tl.where(valid, next_py, work_py),
                    tl.where(valid, next_pz, work_pz),
                )
                work_vx, work_vy, work_vz = (
                    tl.where(valid, next_vx, work_vx),
                    tl.where(valid, next_vy, work_vy),
                    tl.where(valid, next_vz, work_vz),
                )
                tick_valid = valid

            px, py, pz = (
                tl.where(tick_valid, work_px, tick_px),
                tl.where(tick_valid, work_py, tick_py),
                tl.where(tick_valid, work_pz, tick_pz),
            )
            vx, vy, vz = (
                tl.where(tick_valid, work_vx, tick_vx),
                tl.where(tick_valid, work_vy, tick_vy),
                tl.where(tick_valid, work_vz, tick_vz),
            )
            max_ticks += tick_valid.to(tl.int32)
            alive = tick_valid
            state_base = (offsets * 30 + tick) * 6
            tl.store(states_ptr + state_base + 0, px, mask=mask)
            tl.store(states_ptr + state_base + 1, py, mask=mask)
            tl.store(states_ptr + state_base + 2, pz, mask=mask)
            tl.store(states_ptr + state_base + 3, vx, mask=mask)
            tl.store(states_ptr + state_base + 4, vy, mask=mask)
            tl.store(states_ptr + state_base + 5, vz, mask=mask)

        tl.store(max_ticks_ptr + offsets, max_ticks.to(tl.int64), mask=mask)

    _fused_kernel = (triton, _pq_discover)
    return _fused_kernel


@torch.no_grad()
def _discover_horizon_fused(
    batch: PhysicalQuestionCandidateBatch,
    *,
    params: PhysicalQuestionFlightParams,
    config: PhysicalQuestionNumericConfig,
) -> _DiscoveryRecord:
    prepared = _prepare_discovery(batch)
    shape, candidate, contact, incoming, omega, _, _, input_valid = prepared
    triton, kernel = _build_fused_kernel()
    n_rows = candidate.numel()
    tick_states = torch.empty((*shape, 30, 6), dtype=torch.float32, device=candidate.device)
    max_ticks = torch.empty(shape, dtype=torch.int64, device=candidate.device)
    step_s = -float(config.motion_tick_s) / 4.0
    z_min = (
        float(config.table_surface_z_m)
        + float(params.ball_radius_m)
        + float(config.table_clearance_margin_m)
    )
    kernel[(triton.cdiv(n_rows, _FUSED_BLOCK),)](
        contact.reshape(-1, 3), incoming.reshape(-1, 3), omega.reshape(-1, 3),
        input_valid.reshape(-1).view(torch.uint8), tick_states, max_ticks, n_rows,
        _f32(params.k_d), _f32(params.k_m), _f32(params.g),
        _f32(0.5 * step_s), _f32(step_s), _f32(step_s / 6.0),
        _f32(z_min), _f32(config.reverse_speed_cap_mps),
        _f32(torch.finfo(torch.float32).max),
        BLOCK=_FUSED_BLOCK, num_warps=4, enable_fp_fusion=False,
    )
    return _finish_discovery(prepared, max_ticks, tick_states)


def _parity_probe(
    device: torch.device,
    params: PhysicalQuestionFlightParams,
    config: PhysicalQuestionNumericConfig,
) -> PhysicalQuestionCandidateBatch:
    z_min = (
        float(config.table_surface_z_m)
        + float(params.ball_radius_m)
        + float(config.table_clearance_margin_m)
    )
    contact = torch.tensor(
        [
            [0.10, -0.10, z_min + 0.50],
            [0.00, 0.00, z_min + 1.0e-4],
            [-0.10, 0.10, z_min + 0.08],
            [0.20, -0.20, z_min + 0.30],
            [float("nan"), 0.00, z_min + 0.20],
            [0.00, 0.00, z_min + 0.20],
            [0.00, 0.00, z_min + 0.20],
            [0.00, 0.00, z_min + 0.20],
        ],
        dtype=torch.float32,
    ).repeat(192, 1)
    incoming = torch.tensor(
        [
            [-0.8, 0.1, -0.2],
            [0.0, 0.0, 2.0],
            [0.1, 0.0, 1.0],
            [float(config.reverse_speed_cap_mps) * 1.25, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, -0.1],
            [-0.5, 0.0, -0.1],
            [-0.5, 0.0, -0.1],
        ],
        dtype=torch.float32,
    ).repeat(192, 1)
    omega = torch.tensor(
        [
            [20.0, -10.0, 15.0],
            [0.0, 0.0, 0.0],
            [5.0, -3.0, 2.0],
            [0.0, 0.0, 0.0],
            [float("inf"), 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    ).repeat(192, 1)
    candidate = torch.arange(1, 1537, dtype=torch.int64)
    candidate[-3] = 0
    candidate[-2:] = 9001
    return PhysicalQuestionCandidateBatch(
        candidate_identity=candidate.reshape(512, 3).to(device).contiguous(),
        contact_position_env_m=contact.reshape(512, 3, 3).to(device).contiguous(),
        incoming_linear_velocity_world_mps=(
            incoming.reshape(512, 3, 3).to(device).contiguous()
        ),
        incoming_angular_velocity_world_radps=(
            omega.reshape(512, 3, 3).to(device).contiguous()
        ),
    )


def _records_bitwise_equal(reference: _DiscoveryRecord, fused: _DiscoveryRecord) -> bool:
    for name in (
        "candidate_identity",
        "max_feasible_motion_ticks",
        "construction_reason",
        "producer_fault",
    ):
        if not bool(torch.equal(getattr(reference, name), getattr(fused, name))):
            return False
    for name in (
        "contact_position_env_m",
        "incoming_linear_velocity_world_mps",
        "incoming_angular_velocity_world_radps",
        "motion_tick_state_f32",
    ):
        left = getattr(reference, name)
        right = getattr(fused, name)
        if left is None or right is None or tuple(left.shape) != tuple(right.shape):
            return False
        if not bool(torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))):
            return False
    return True


@torch.no_grad()
def _fast_path_admitted(
    batch: PhysicalQuestionCandidateBatch,
    params: PhysicalQuestionFlightParams,
    config: PhysicalQuestionNumericConfig,
) -> bool:
    mode = os.environ.get(_FAST_ENV, "1").strip().lower()
    if mode in ("0", "off", "false", "no", "eager"):
        return False
    device, _ = _require_candidate_batch(batch)
    if (
        device.type != "cuda"
        or config.max_final_segment_motion_ticks != 30
        or config.integration_substeps_per_motion_tick != 4
        or _build_fused_kernel() is None
    ):
        return False
    key = (device.type, device.index, params, config)
    admitted = _parity_cache.get(key)
    if admitted is not None:
        return admitted
    try:
        probe = _parity_probe(device, params, config)
        reference = _discover_horizon_reference(
            probe,
            params=params,
            config=config,
            _retain_motion_tick_state=True,
        )
        fused = _discover_horizon_fused(probe, params=params, config=config)
        admitted = _records_bitwise_equal(reference, fused)
    except Exception as exc:  # pragma: no cover - CUDA/runtime safety net
        warnings.warn(
            f"physical question fused discovery disabled ({exc!r})",
            RuntimeWarning,
            stacklevel=2,
        )
        _parity_cache[key] = False
        return False
    if not admitted:
        warnings.warn(
            "physical question fused discovery is not bit-identical; using reference",
            RuntimeWarning,
            stacklevel=2,
        )
    _parity_cache[key] = admitted
    return admitted


@torch.no_grad()
def _discover_horizon(
    batch: PhysicalQuestionCandidateBatch,
    *,
    params: PhysicalQuestionFlightParams,
    config: PhysicalQuestionNumericConfig,
) -> _DiscoveryRecord:
    device, _ = _require_candidate_batch(batch)
    if _fast_path_admitted(batch, params, config):
        return _discover_horizon_fused(batch, params=params, config=config)
    return _discover_horizon_reference(
        batch,
        params=params,
        config=config,
        # Discovery already computes every prefix. Six float32 values per
        # policy tick replace the entire exact-tick RK4 replay on CUDA.
        _retain_motion_tick_state=device.type == "cuda",
    )


@torch.no_grad()
def _finalize_exact_ticks(
    record: _DiscoveryRecord,
    *,
    candidate_identity: torch.Tensor,
    contact_tick: torch.Tensor,
    launch_tick: torch.Tensor,
    params: PhysicalQuestionFlightParams,
    config: PhysicalQuestionNumericConfig,
) -> PhysicalQuestionFinalBatch:
    shape = tuple(record.candidate_identity.shape)
    device = record.candidate_identity.device
    for name, value in (
        ("candidate_identity", candidate_identity),
        ("contact_tick", contact_tick),
        ("launch_tick", launch_tick),
    ):
        if (
            type(value) is not torch.Tensor
            or value.device != device
            or value.dtype is not torch.int64
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise PhysicalQuestionError(
                f"{name} must be contiguous int64 on {device} with shape {shape}"
            )

    same_candidate = candidate_identity.eq(record.candidate_identity)
    exact_ticks = contact_tick - launch_tick
    chronology_valid = (
        contact_tick >= 0
    ) & (launch_tick >= 0) & (exact_ticks > 0) & (
        exact_ticks <= record.max_feasible_motion_ticks
    )
    discovery_admitted = record.construction_reason.eq(CONSTRUCTION_REASON_ADMITTED)
    recompute_requested = same_candidate & chronology_valid & discovery_admitted

    position = record.contact_position_env_m
    velocity = record.incoming_linear_velocity_world_mps
    omega = record.incoming_angular_velocity_world_radps
    tick_states = record.motion_tick_state_f32
    if tick_states is not None:
        state_index = (
            exact_ticks.clamp(min=1, max=config.max_final_segment_motion_ticks)
            - 1
        )
        state_index = state_index.unsqueeze(-1).unsqueeze(-1).expand(*shape, 1, 6)
        selected_state = torch.gather(
            tick_states, dim=-2, index=state_index
        ).squeeze(-2)
        position = selected_state[..., 0:3]
        velocity = selected_state[..., 3:6]

    path_valid = recompute_requested
    if tick_states is None:
        step_s = -float(config.motion_tick_s) / float(
            config.integration_substeps_per_motion_tick
        )
        requested_substeps = exact_ticks.clamp(min=0) * int(
            config.integration_substeps_per_motion_tick
        )
        total_substeps = (
            config.max_final_segment_motion_ticks
            * config.integration_substeps_per_motion_tick
        )
        z_min = (
            float(config.table_surface_z_m)
            + float(params.ball_radius_m)
            + float(config.table_clearance_margin_m)
        )
        for substep in range(total_substeps):
            active = recompute_requested & (requested_substeps > substep)
            proposed_position, proposed_velocity = _rk4_step(
                position, velocity, omega, step_s, params
            )
            finite_state = (
                torch.all(torch.isfinite(proposed_position), dim=-1)
                & torch.all(torch.isfinite(proposed_velocity), dim=-1)
            )
            step_valid = (
                finite_state
                & (proposed_position[..., 2] > z_min)
                & (
                    torch.linalg.vector_norm(proposed_velocity, dim=-1)
                    < float(config.reverse_speed_cap_mps)
                )
            )
            path_valid = path_valid & (~active | step_valid)
            commit = active & step_valid
            position = torch.where(
                commit.unsqueeze(-1), proposed_position, position
            )
            velocity = torch.where(
                commit.unsqueeze(-1), proposed_velocity, velocity
            )

    recompute_valid = recompute_requested & path_valid
    identity_quaternion = torch.zeros((*shape, 4), dtype=torch.float32, device=device)
    identity_quaternion[..., 0] = 1.0
    physical = torch.cat((position, identity_quaternion, velocity, omega), dim=-1)
    physical = torch.where(
        recompute_valid.unsqueeze(-1), physical, torch.zeros_like(physical)
    ).contiguous()

    reason = record.construction_reason.clone()
    reason = torch.where(
        discovery_admitted & ~same_candidate,
        torch.full_like(reason, CONSTRUCTION_REASON_CANDIDATE_BINDING_DIFFERS),
        reason,
    )
    reason = torch.where(
        discovery_admitted & same_candidate & ~chronology_valid,
        torch.full_like(reason, CONSTRUCTION_REASON_EXACT_CHRONOLOGY_INVALID),
        reason,
    )
    reason = torch.where(
        recompute_requested & ~path_valid,
        torch.full_like(reason, CONSTRUCTION_REASON_RECOMPUTE_INVALID),
        reason,
    ).contiguous()

    fault = record.producer_fault.clone()
    fault = torch.where(
        discovery_admitted & ~same_candidate,
        torch.bitwise_or(
            fault,
            torch.full_like(fault, PRODUCER_FAULT_CANDIDATE_BINDING_DIFFERS),
        ),
        fault,
    )
    fault = torch.where(
        discovery_admitted & same_candidate & ~chronology_valid,
        torch.bitwise_or(
            fault,
            torch.full_like(fault, PRODUCER_FAULT_EXACT_CHRONOLOGY_INVALID),
        ),
        fault,
    )
    fault = torch.where(
        recompute_requested & ~path_valid,
        torch.bitwise_or(
            fault,
            torch.full_like(fault, PRODUCER_FAULT_RECOMPUTE_INVALID),
        ),
        fault,
    ).contiguous()

    horizon_s = exact_ticks.to(torch.float32) * float(config.motion_tick_s)
    horizon_s = torch.where(
        recompute_valid, horizon_s, torch.zeros_like(horizon_s)
    ).contiguous()
    return PhysicalQuestionFinalBatch(
        candidate_identity=record.candidate_identity.clone(),
        contact_tick=contact_tick.clone(),
        launch_tick=launch_tick.clone(),
        construction_reason=reason,
        physical_state_f32=physical,
        effective_contact_horizon_s=horizon_s,
        producer_fault=fault,
    )


class PhysicalQuestionNumericCore:
    """Diagnostic-only owner of immutable inputs, horizon and one-shot receipt."""

    __slots__ = ("_params", "_config", "_pending", "_consumed", "__weakref__")

    def __init__(
        self,
        construction_key: object,
        *,
        params: PhysicalQuestionFlightParams,
        config: PhysicalQuestionNumericConfig,
    ) -> None:
        if construction_key is not _OWNER_CONSTRUCTION_KEY:
            raise TypeError("Physical question owners require an exact factory")
        if type(params) is not PhysicalQuestionFlightParams:
            raise PhysicalQuestionError("flight params type differs")
        if type(config) is not PhysicalQuestionNumericConfig:
            raise PhysicalQuestionError("numeric config type differs")
        self._params = params
        self._config = config
        self._pending: dict[PhysicalQuestionHorizonReceipt, _DiscoveryRecord] = {}
        self._consumed: weakref.WeakSet[PhysicalQuestionHorizonReceipt] = weakref.WeakSet()

    def issue_horizon_for_test(
        self, batch: PhysicalQuestionCandidateBatch
    ) -> PhysicalQuestionHorizonReceipt:
        """Discover and privately retain the maximum complete-tick horizon."""

        record = _discover_horizon(batch, params=self._params, config=self._config)
        receipt = object.__new__(PhysicalQuestionHorizonReceipt)
        self._pending[receipt] = record
        return receipt

    def project_horizon_for_test(
        self, receipt: object
    ) -> PhysicalQuestionHorizonView:
        """Return a clone-only scheduling view; it never authorizes finalization."""

        record = self._require_pending(receipt)
        return PhysicalQuestionHorizonView(
            candidate_identity=record.candidate_identity.clone(),
            max_feasible_motion_ticks=record.max_feasible_motion_ticks.clone(),
            max_feasible_horizon_s=(
                record.max_feasible_motion_ticks.to(torch.float32)
                * float(self._config.motion_tick_s)
            ).contiguous(),
            construction_reason=record.construction_reason.clone(),
            producer_fault=record.producer_fault.clone(),
        )

    def finalize_exact_ticks_for_test(
        self,
        receipt: object,
        *,
        candidate_identity: torch.Tensor,
        contact_tick: torch.Tensor,
        launch_tick: torch.Tensor,
    ) -> PhysicalQuestionFinalBatch:
        """Consume one receipt and resolve the candidate's exact launch state."""

        record = self._require_pending(receipt)
        result = _finalize_exact_ticks(
            record,
            candidate_identity=candidate_identity,
            contact_tick=contact_tick,
            launch_tick=launch_tick,
            params=self._params,
            config=self._config,
        )
        del self._pending[receipt]
        self._consumed.add(receipt)
        return result

    def _require_pending(self, receipt: object) -> _DiscoveryRecord:
        if type(receipt) is not PhysicalQuestionHorizonReceipt:
            raise PhysicalQuestionConflictError("horizon receipt type differs")
        if receipt in self._consumed:
            raise PhysicalQuestionConflictError("horizon receipt was already consumed")
        record = self._pending.get(receipt)
        if record is None:
            raise PhysicalQuestionConflictError("horizon receipt is foreign")
        return record


def make_test_physical_question_numeric_core(
    *,
    params: PhysicalQuestionFlightParams,
    config: PhysicalQuestionNumericConfig,
) -> PhysicalQuestionNumericCore:
    """Construct the diagnostic numerical owner; it cannot authorize runtime."""

    return PhysicalQuestionNumericCore(
        _OWNER_CONSTRUCTION_KEY, params=params, config=config
    )


def construct_diagnostic_n2_no_save_physical_question_numeric_core(
    *,
    physical_flight_owner: object,
    motion_owner: object,
    racket_owner: object,
) -> PhysicalQuestionNumericCore:
    """Bind one numeric core to an exact live cardinality-neutral graph.

    The historical function name is temporary naming debt.  Environment
    cardinality ``N`` comes only from exact equality among the Physical owner,
    its scene port, the Motion owner and Motion's environment; ``N`` may be any
    positive exact integer.  Physical flight capacity remains the independent
    exact constant ``K=2``.  ActionEpoch is deliberately not required here:
    the factory constructs this core before the epoch, and Physical's later
    exact ``bind_action_epoch_owner`` join enforces the same ``N``.

    This is a cold, diagnostic-only bridge.  Callers supply owner identities,
    never flight constants, a table plane, a tick duration, a candidate state,
    or a horizon.  The constants come from the venue file consumed by the
    existing Physical implementation; tick geometry comes from Motion's exact
    environment; and the table plane comes from the exact constructed Racket
    config in that same environment.  The Physical scene assets must be the
    very objects installed in Motion's environment, not a second compatible
    scene.

    The returned core still cannot mint candidate identities or authorize a
    launch.  The construction-bound question composer remains their sole
    writer and Motion must still provide exact contact/launch chronology from
    inside D05.  Consequently this function closes only the factory's missing
    numeric-core dependency; it does not make the production constructor GO.
    """

    try:
        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_physical_flight_device as physical,
        )
        from whole_body_tracking.tasks.tracking.mdp import commands
        from whole_body_tracking.tasks.tracking.mdp import hope_commands
        from whole_body_tracking.tasks.tracking.mdp import physical_ball
        from whole_body_tracking.tasks.tracking.mdp import virtual_ball
        from whole_body_tracking.tasks.tracking.config.agibot_a3 import (
            action_ball_full_mdp_ball_scene as scene,
        )
    except ImportError as exc:
        raise PhysicalQuestionProductionHold(
            "diagnostic Physical question dependencies are unavailable"
        ) from exc

    if type(physical_flight_owner) is not physical.ActionBallPhysicalFlightDeviceOwner:
        raise PhysicalQuestionError(
            "diagnostic numeric core requires the exact Physical owner"
        )
    if type(motion_owner) is not commands.MotionCommand:
        raise PhysicalQuestionError(
            "diagnostic numeric core requires the exact Motion owner"
        )
    if type(racket_owner) is not hope_commands.RacketTargetCommand:
        raise PhysicalQuestionError(
            "diagnostic numeric core requires the exact Racket owner"
        )
    physical_num_envs = _positive_exact_int(
        getattr(physical_flight_owner, "num_envs", None),
        label="Physical owner num_envs",
    )
    motion_num_envs = _positive_exact_int(
        getattr(motion_owner, "num_envs", None),
        label="Motion owner num_envs",
    )
    port = getattr(physical_flight_owner, "scene_port", None)
    spec = getattr(port, "spec", None)
    if (
        type(port) is not scene.IsaacLabPhysicalFlightScenePort
        or type(spec) is not scene.ActionBallFullMdpDiagnosticBallSceneSpec
        or getattr(physical_flight_owner, "_diagnostic_n2_no_save", None)
        is not True
        or getattr(physical_flight_owner, "flight_capacity", None) != 2
        or hasattr(physical_flight_owner, "capacity_receipt")
        or getattr(spec, "formal_capacity_receipt_sha256", object()) is not None
    ):
        raise PhysicalQuestionError(
            "Physical owner is not the exact diagnostic no-save K=2 scene owner"
        )
    port_num_envs = _positive_exact_int(
        getattr(port, "num_envs", None), label="Physical scene port num_envs"
    )
    physical_device = torch.device(
        getattr(physical_flight_owner, "device", "cpu")
    )

    env = getattr(motion_owner, "_env", None)
    env_scene = getattr(env, "scene", None)
    env_num_envs = _positive_exact_int(
        getattr(env, "num_envs", None), label="Motion environment num_envs"
    )
    assets = getattr(port, "assets", ())
    if type(assets) is not tuple or len(assets) != 2:
        raise PhysicalQuestionError(
            "Physical scene installed asset inventory differs from K=2"
        )
    asset_row_counts = tuple(
        _tensor_row_count(
            getattr(getattr(asset, "data", None), "root_state_w", None),
            label=f"Physical installed asset {index} root state",
            rank=2,
            trailing_shape=(13,),
            device=physical_device,
        )
        for index, asset in enumerate(assets)
    )
    env_origin_rows = _tensor_row_count(
        getattr(env_scene, "env_origins", None),
        label="Motion environment origins",
        rank=2,
        trailing_shape=(3,),
        device=physical_device,
    )
    motion_construction_rows = _tensor_row_count(
        getattr(motion_owner, "time_steps", None),
        label="Motion construction time_steps",
        rank=1,
        trailing_shape=(),
        device=physical_device,
    )
    if (
        env is None
        or env_scene is None
        or physical_num_envs != port_num_envs
        or physical_num_envs != motion_num_envs
        or physical_num_envs != env_num_envs
        or any(rows != physical_num_envs for rows in asset_row_counts)
        or env_origin_rows != physical_num_envs
        or motion_construction_rows != physical_num_envs
        or torch.device(getattr(env, "device", "cpu"))
        != physical_device
    ):
        raise PhysicalQuestionError(
            "Physical owner and Motion do not share one exact positive-N environment"
        )
    try:
        live_assets = tuple(env_scene[name] for name in spec.scene_entity_names)
    except (KeyError, TypeError) as exc:
        raise PhysicalQuestionError(
            "Motion environment lacks the Physical owner's exact scene assets"
        ) from exc
    if any(
        actual is not expected
        for actual, expected in zip(live_assets, port.assets, strict=True)
    ):
        raise PhysicalQuestionError(
            "Physical owner is bound to a scene different from Motion's environment"
        )

    if getattr(racket_owner, "_env", None) is not env:
        raise PhysicalQuestionError(
            "Physical, Motion and Racket do not share one exact environment"
        )
    racket_cfg = getattr(racket_owner, "cfg", None)
    if type(racket_cfg) is not hope_commands.RacketTargetCommandCfg:
        raise PhysicalQuestionError(
            "diagnostic numeric core requires the exact constructed Racket config"
        )
    step_s = getattr(env, "step_dt", None)
    table_surface_z_m = getattr(racket_cfg, "vb_table_surface_z", None)
    if (
        type(step_s) is not float
        or not math.isfinite(step_s)
        or step_s <= 0.0
        or type(table_surface_z_m) is not float
        or not math.isfinite(table_surface_z_m)
    ):
        raise PhysicalQuestionError(
            "Motion tick or Racket table-plane source is invalid"
        )

    venue = virtual_ball.load_venue_params()
    if type(venue) is not virtual_ball.VirtualBallParams:
        raise PhysicalQuestionError("venue flight parameter owner returned a foreign type")
    venue_values = (
        venue.k_d,
        venue.k_m,
        venue.g,
        venue.ball_radius,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in venue_values
    ):
        raise PhysicalQuestionError("venue flight parameter owner returned invalid values")
    if float(spec.ball_radius_m) != float(venue.ball_radius):
        raise PhysicalQuestionError(
            "Physical scene radius differs from the venue flight model"
        )

    substep_s = float(physical_ball.SERVE_BACKINT_H)
    horizon_s = float(physical_ball.SERVE_HORIZON_S)
    substeps = round(step_s / substep_s)
    max_ticks = round(horizon_s / step_s)
    if (
        substeps < 1
        or max_ticks < 1
        or not math.isclose(
            substeps * substep_s, step_s, rel_tol=0.0, abs_tol=1.0e-12
        )
        or not math.isclose(
            max_ticks * step_s, horizon_s, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise PhysicalQuestionError(
            "Motion tick does not exactly partition the Physical serve geometry"
        )

    params = PhysicalQuestionFlightParams(
        k_d=float(venue.k_d),
        k_m=float(venue.k_m),
        g=float(venue.g),
        ball_radius_m=float(venue.ball_radius),
    )
    config = PhysicalQuestionNumericConfig(
        motion_tick_s=step_s,
        integration_substeps_per_motion_tick=substeps,
        max_final_segment_motion_ticks=max_ticks,
        table_surface_z_m=table_surface_z_m,
        table_clearance_margin_m=float(physical_ball.SERVE_PLANE_MARGIN),
        reverse_speed_cap_mps=float(physical_ball.BACKINT_SPEED_CAP),
    )
    with _BOUND_DIAGNOSTIC_PHYSICAL_OWNERS_LOCK:
        if physical_flight_owner in _BOUND_DIAGNOSTIC_PHYSICAL_OWNERS:
            raise PhysicalQuestionConflictError(
                "Physical owner already has one diagnostic numeric core"
            )
        core = PhysicalQuestionNumericCore(
            _OWNER_CONSTRUCTION_KEY,
            params=params,
            config=config,
        )
        _BOUND_DIAGNOSTIC_PHYSICAL_OWNERS[physical_flight_owner] = core
    return core


def construct_production_physical_question_producer(
    *,
    exact_motion_contact_launch_tick_authority: object,
    candidate_identity_authority: object,
    venue_flight_parameter_authority: object,
) -> PhysicalQuestionNumericCore:
    """Fail closed until exact source owners and their joins are implemented."""

    del (
        exact_motion_contact_launch_tick_authority,
        candidate_identity_authority,
        venue_flight_parameter_authority,
    )
    raise PhysicalQuestionProductionHold(
        "production Physical question producer remains HOLD: the exact Motion "
        "contact/launch tick authority, candidate identity owner and venue "
        "flight-parameter authority are not frozen"
    )


__all__ = (
    "CONSTRUCTION_REASON_ADMITTED",
    "CONSTRUCTION_REASON_CANDIDATE_BINDING_DIFFERS",
    "CONSTRUCTION_REASON_EXACT_CHRONOLOGY_INVALID",
    "CONSTRUCTION_REASON_INVALID_PRODUCER",
    "CONSTRUCTION_REASON_NO_COMPLETE_FINAL_SEGMENT",
    "CONSTRUCTION_REASON_RECOMPUTE_INVALID",
    "DIAGNOSTIC_UNAUTHORIZED",
    "INTEGRATION_STATUS",
    "LAUNCH_AUTHORIZED",
    "PHYSICAL_STATE_F32_FIELDS",
    "PRODUCER_FAULT_CANDIDATE_BINDING_DIFFERS",
    "PRODUCER_FAULT_EXACT_CHRONOLOGY_INVALID",
    "PRODUCER_FAULT_INVALID_CANDIDATE_IDENTITY",
    "PRODUCER_FAULT_NONFINITE_INPUT",
    "PRODUCER_FAULT_RECOMPUTE_INVALID",
    "PhysicalQuestionCandidateBatch",
    "PhysicalQuestionConflictError",
    "PhysicalQuestionError",
    "PhysicalQuestionFinalBatch",
    "PhysicalQuestionFlightParams",
    "PhysicalQuestionHorizonReceipt",
    "PhysicalQuestionHorizonView",
    "PhysicalQuestionNumericConfig",
    "PhysicalQuestionNumericCore",
    "PhysicalQuestionProductionHold",
    "RUNTIME_INTEGRATED",
    "construct_production_physical_question_producer",
    "construct_diagnostic_n2_no_save_physical_question_numeric_core",
    "make_test_physical_question_numeric_core",
)
