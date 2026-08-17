"""Pure-Torch exact-face and teacher-timing construction leaf.

This module mirrors the deterministic construction performed by
``racket_contact_geometry.solve_exact_face_contact`` followed by the canonical
ActionBall teacher clock.  It is not an achieved-contact detector and does not
observe simulator state.  Dynamic row failures remain device-resident as an
ordinary construction reason plus an independent producer-fault bitmask; every
non-admitted floating output is NaN-masked.

No production consumer is bound yet.  A future construction-bound question
authority must retain these exact tensors and consume ``producer_fault_bits``
at its sole packed reveal boundary before any task, ball, or motion mutation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


CONSTRUCTION_REASON_ADMITTED = -1
CONSTRUCTION_REASON_TEACHER_SITE_RATE_GEOMETRY_UNSOLVED = 8
CONSTRUCTION_REASON_TEACHER_RATE_OUT_OF_BOUNDS = 9
CONSTRUCTION_REASON_PRE_SWING_WAIT_OUT_OF_BOUNDS = 10
CONSTRUCTION_REASON_CYCLE_EXCEEDS_EPISODE_HORIZON = 11

PRODUCER_FAULT_NONFINITE_GEOMETRY_INPUT = 1 << 0
PRODUCER_FAULT_INVALID_FACE_SIGN = 1 << 1
PRODUCER_FAULT_INVALID_REFERENCE_QUATERNION = 1 << 2
PRODUCER_FAULT_INVALID_SOLVED_NORMAL = 1 << 3
PRODUCER_FAULT_INVALID_RATE_PROFILE = 1 << 4
PRODUCER_FAULT_INVALID_TIMING_PROFILE = 1 << 5
PRODUCER_FAULT_MASK = (1 << 6) - 1

_FACE_AREA_CENTER_XZ_FROM_SITE_M = (0.000893694377, 0.000893694377)
_RED_OUTER_Y_FROM_SITE_M = 0.0
_BLACK_OUTER_Y_FROM_SITE_M = -0.013207999989
_BALL_RADIUS_M = 0.020000000148
_RATE_BOUNDARY_ABS_TOL = 5.0e-7
_NORMAL_EPS = 1.0e-12
_ANTIPODAL_TOL = 1.0e-12
_QUADRATIC_REL_TOL = 1.0e-14
_ROOT_RESIDUAL_REL_TOL = 2.0e-11
_ORIENTATION_RECONSTRUCTION_ABS_TOL = 2.0e-12
_QUATERNION_UNIT_PRESERVE_ABS_TOL = 2.0e-15
_MAX_PRE_SWING_WAIT_S = 1.0
_EPISODE_HORIZON_ABS_TOL = 1.0e-12


@dataclass(frozen=True)
class DeviceExactFaceTimingResult:
    """One K-row device-resident construction result."""

    racket_command_quat_wxyz: torch.Tensor
    racket_site_target_w_m: torch.Tensor
    racket_face_center_velocity_w_mps: torch.Tensor
    racket_site_velocity_w_mps: torch.Tensor
    racket_command_angular_velocity_w_radps: torch.Tensor
    required_racket_site_speed_mps: torch.Tensor
    teacher_rate: torch.Tensor
    scaled_t_hit_s: torch.Tensor
    scaled_t_cycle_s: torch.Tensor
    pre_swing_wait_s: torch.Tensor
    construction_reason: torch.Tensor
    producer_fault_bits: torch.Tensor
    admitted: torch.Tensor


def _require_batch_inputs(named: tuple[tuple[str, torch.Tensor, int], ...]) -> int:
    first_name, first, first_width = named[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"{first_name} must be a torch.Tensor")
    if first.ndim != 2 or first.shape[1] != first_width:
        raise ValueError(f"{first_name} must have shape (K,{first_width})")
    row_count = first.shape[0]
    if row_count <= 0:
        raise ValueError("exact-face timing batch must contain at least one row")
    if not first.dtype.is_floating_point:
        raise TypeError("exact-face timing tensors must use a floating dtype")
    for name, tensor, width in named[1:]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected = (row_count,) if width == 0 else (row_count, width)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if tensor.dtype != first.dtype or tensor.device != first.device:
            raise ValueError(
                "exact-face timing tensors must share one dtype and device"
            )
    return row_count


def _cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            left[:, 1] * right[:, 2] - left[:, 2] * right[:, 1],
            left[:, 2] * right[:, 0] - left[:, 0] * right[:, 2],
            left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0],
        ),
        dim=-1,
    )


def _canonical_quaternion(raw: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    normalized = torch.where(
        (norm - 1.0).abs() <= _QUATERNION_UNIT_PRESERVE_ABS_TOL,
        raw,
        raw / norm,
    )
    significant = normalized.abs() > 1.0e-15
    first_index = significant.to(dtype=torch.int64).argmax(dim=-1, keepdim=True)
    first_value = normalized.gather(-1, first_index)
    sign = torch.where(first_value < 0.0, -torch.ones_like(first_value), torch.ones_like(first_value))
    return normalized * sign


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    raw = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )
    return _canonical_quaternion(raw)


def _quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    scalar = quaternion[:, :1]
    xyz = quaternion[:, 1:]
    uv = _cross(xyz, vector)
    uuv = _cross(xyz, uv)
    return vector + 2.0 * (scalar * uv + uuv)


def _masked(value: torch.Tensor, admitted: torch.Tensor) -> torch.Tensor:
    mask = admitted
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, value, torch.full_like(value, float("nan")))


@torch.no_grad()
def solve_exact_face_timing_device(
    *,
    ball_contact_w_m: torch.Tensor,
    racket_face_center_velocity_w_mps: torch.Tensor,
    solved_raw_a_normal_w: torch.Tensor,
    mount_normal_sign: torch.Tensor,
    reference_racket_quat_wxyz: torch.Tensor,
    reference_racket_angular_velocity_w_radps: torch.Tensor,
    reference_racket_site_speed_mps: torch.Tensor,
    teacher_rate_min: torch.Tensor,
    teacher_rate_max: torch.Tensor,
    time_to_contact_s: torch.Tensor,
    reference_t_hit_s: torch.Tensor,
    reference_t_cycle_s: torch.Tensor,
    reaction_margin_s: torch.Tensor,
    attempt_close_margin_s: torch.Tensor,
    episode_length_s: torch.Tensor,
) -> DeviceExactFaceTimingResult:
    """Solve K exact-face rows and their immutable teacher timing on device.

    Shape, dtype, and device mismatches are static API errors.  All predicates
    that depend on tensor values are encoded in the returned tensors without a
    host observation.  Computation uses binary64 because the scalar authority
    consumes the same stored float values through Python binary64 arithmetic.
    """

    _require_batch_inputs(
        (
            ("ball_contact_w_m", ball_contact_w_m, 3),
            (
                "racket_face_center_velocity_w_mps",
                racket_face_center_velocity_w_mps,
                3,
            ),
            ("solved_raw_a_normal_w", solved_raw_a_normal_w, 3),
            ("mount_normal_sign", mount_normal_sign, 0),
            (
                "reference_racket_quat_wxyz",
                reference_racket_quat_wxyz,
                4,
            ),
            (
                "reference_racket_angular_velocity_w_radps",
                reference_racket_angular_velocity_w_radps,
                3,
            ),
            (
                "reference_racket_site_speed_mps",
                reference_racket_site_speed_mps,
                0,
            ),
            ("teacher_rate_min", teacher_rate_min, 0),
            ("teacher_rate_max", teacher_rate_max, 0),
            ("time_to_contact_s", time_to_contact_s, 0),
            ("reference_t_hit_s", reference_t_hit_s, 0),
            ("reference_t_cycle_s", reference_t_cycle_s, 0),
            ("reaction_margin_s", reaction_margin_s, 0),
            ("attempt_close_margin_s", attempt_close_margin_s, 0),
            ("episode_length_s", episode_length_s, 0),
        )
    )

    source_dtype = ball_contact_w_m.dtype
    device = ball_contact_w_m.device
    ball = ball_contact_w_m.to(dtype=torch.float64)
    face_velocity = racket_face_center_velocity_w_mps.to(dtype=torch.float64)
    solved_normal = solved_raw_a_normal_w.to(dtype=torch.float64)
    face_sign = mount_normal_sign.to(dtype=torch.float64)
    reference_quat = reference_racket_quat_wxyz.to(dtype=torch.float64)
    reference_omega = reference_racket_angular_velocity_w_radps.to(dtype=torch.float64)
    reference_speed = reference_racket_site_speed_mps.to(dtype=torch.float64)
    rate_min = teacher_rate_min.to(dtype=torch.float64)
    rate_max = teacher_rate_max.to(dtype=torch.float64)
    ttc = time_to_contact_s.to(dtype=torch.float64)
    reference_hit = reference_t_hit_s.to(dtype=torch.float64)
    reference_cycle = reference_t_cycle_s.to(dtype=torch.float64)
    reaction_margin = reaction_margin_s.to(dtype=torch.float64)
    close_margin = attempt_close_margin_s.to(dtype=torch.float64)
    episode_length = episode_length_s.to(dtype=torch.float64)

    geometry_finite = (
        torch.isfinite(ball).all(dim=-1)
        & torch.isfinite(face_velocity).all(dim=-1)
        & torch.isfinite(reference_omega).all(dim=-1)
    )
    sign_valid = torch.isfinite(face_sign) & ((face_sign == 1.0) | (face_sign == -1.0))
    quaternion_norm = torch.linalg.vector_norm(reference_quat, dim=-1)
    quaternion_valid = (
        torch.isfinite(reference_quat).all(dim=-1)
        & torch.isfinite(quaternion_norm)
        & (quaternion_norm > _NORMAL_EPS)
    )
    solved_normal_norm = torch.linalg.vector_norm(solved_normal, dim=-1)
    solved_normal_valid = (
        torch.isfinite(solved_normal).all(dim=-1)
        & torch.isfinite(solved_normal_norm)
        & (solved_normal_norm > _NORMAL_EPS)
    )
    rate_profile_valid = (
        torch.isfinite(reference_speed)
        & torch.isfinite(rate_min)
        & torch.isfinite(rate_max)
        & (reference_speed > 0.0)
        & (rate_min > 0.0)
        & (rate_max >= rate_min)
        & (rate_min <= 1.0)
        & (rate_max >= 1.0)
    )
    timing_profile_valid = (
        torch.isfinite(ttc)
        & torch.isfinite(reference_hit)
        & torch.isfinite(reference_cycle)
        & torch.isfinite(reaction_margin)
        & torch.isfinite(close_margin)
        & torch.isfinite(episode_length)
        & (ttc >= 0.0)
        & (reference_hit > 0.0)
        & (reference_cycle > reference_hit)
        & (reaction_margin >= 0.0)
        & (close_margin >= 0.0)
        & (episode_length > 0.0)
    )

    fault = torch.zeros(ball.shape[0], dtype=torch.int64, device=device)
    fault = torch.where(
        geometry_finite,
        fault,
        fault | PRODUCER_FAULT_NONFINITE_GEOMETRY_INPUT,
    )
    fault = torch.where(sign_valid, fault, fault | PRODUCER_FAULT_INVALID_FACE_SIGN)
    fault = torch.where(
        quaternion_valid,
        fault,
        fault | PRODUCER_FAULT_INVALID_REFERENCE_QUATERNION,
    )
    fault = torch.where(
        solved_normal_valid,
        fault,
        fault | PRODUCER_FAULT_INVALID_SOLVED_NORMAL,
    )
    fault = torch.where(
        rate_profile_valid,
        fault,
        fault | PRODUCER_FAULT_INVALID_RATE_PROFILE,
    )
    fault = torch.where(
        timing_profile_valid,
        fault,
        fault | PRODUCER_FAULT_INVALID_TIMING_PROFILE,
    )
    fault_free = fault == 0

    safe_ball = torch.where(geometry_finite.unsqueeze(-1), ball, torch.zeros_like(ball))
    safe_face_velocity = torch.where(
        geometry_finite.unsqueeze(-1), face_velocity, torch.zeros_like(face_velocity)
    )
    safe_sign = torch.where(sign_valid, face_sign, torch.ones_like(face_sign))
    identity = torch.zeros_like(reference_quat)
    identity[:, 0] = 1.0
    safe_reference_quat = torch.where(
        quaternion_valid.unsqueeze(-1), reference_quat, identity
    )
    safe_reference_quat = _canonical_quaternion(safe_reference_quat)
    fallback_normal = torch.zeros_like(solved_normal)
    fallback_normal[:, 1] = 1.0
    safe_solved_normal = torch.where(
        solved_normal_valid.unsqueeze(-1), solved_normal, fallback_normal
    )
    safe_solved_normal = safe_solved_normal / torch.linalg.vector_norm(
        safe_solved_normal, dim=-1, keepdim=True
    )
    safe_reference_omega = torch.where(
        geometry_finite.unsqueeze(-1), reference_omega, torch.zeros_like(reference_omega)
    )
    safe_reference_speed = torch.where(rate_profile_valid, reference_speed, torch.ones_like(reference_speed))
    safe_rate_min = torch.where(rate_profile_valid, rate_min, torch.full_like(rate_min, 0.5))
    safe_rate_max = torch.where(rate_profile_valid, rate_max, torch.full_like(rate_max, 2.0))
    safe_ttc = torch.where(timing_profile_valid, ttc, torch.ones_like(ttc))
    safe_hit = torch.where(timing_profile_valid, reference_hit, torch.full_like(reference_hit, 0.5))
    safe_cycle = torch.where(timing_profile_valid, reference_cycle, torch.ones_like(reference_cycle))
    safe_reaction = torch.where(timing_profile_valid, reaction_margin, torch.zeros_like(reaction_margin))
    safe_close = torch.where(timing_profile_valid, close_margin, torch.zeros_like(close_margin))
    safe_episode = torch.where(timing_profile_valid, episode_length, torch.full_like(episode_length, 4.0))

    raw_a_local = torch.zeros_like(safe_solved_normal)
    raw_a_local[:, 1] = 1.0
    reference_raw_a = _quat_rotate(safe_reference_quat, raw_a_local)
    orientation_dot = (reference_raw_a * safe_solved_normal).sum(dim=-1).clamp(-1.0, 1.0)
    antipodal = orientation_dot <= (-1.0 + _ANTIPODAL_TOL)
    delta_raw = torch.cat(
        ((1.0 + orientation_dot).unsqueeze(-1), _cross(reference_raw_a, safe_solved_normal)),
        dim=-1,
    )
    delta_fallback = identity
    delta_norm = torch.linalg.vector_norm(delta_raw, dim=-1)
    delta_safe = torch.where(
        (delta_norm > _NORMAL_EPS).unsqueeze(-1), delta_raw, delta_fallback
    )
    delta_quat = _canonical_quaternion(delta_safe)
    command_quat = _quat_multiply(delta_quat, safe_reference_quat)
    reconstructed = _quat_rotate(command_quat, raw_a_local)
    reconstruction_bad = (
        torch.linalg.vector_norm(reconstructed - safe_solved_normal, dim=-1)
        > _ORIENTATION_RECONSTRUCTION_ABS_TOL
    )

    omega_native_command = _quat_rotate(delta_quat, safe_reference_omega)
    face_offset_local = torch.stack(
        (
            torch.full_like(safe_sign, _FACE_AREA_CENTER_XZ_FROM_SITE_M[0]),
            torch.where(
                safe_sign > 0.0,
                torch.full_like(safe_sign, _RED_OUTER_Y_FROM_SITE_M),
                torch.full_like(safe_sign, _BLACK_OUTER_Y_FROM_SITE_M),
            ),
            torch.full_like(safe_sign, _FACE_AREA_CENTER_XZ_FROM_SITE_M[1]),
        ),
        dim=-1,
    )
    face_offset_w = _quat_rotate(command_quat, face_offset_local)
    angular_face_term = _cross(omega_native_command, face_offset_w)

    angular_sq = (angular_face_term * angular_face_term).sum(dim=-1)
    velocity_sq = (safe_face_velocity * safe_face_velocity).sum(dim=-1)
    speed_sq = safe_reference_speed * safe_reference_speed
    coefficient_a = speed_sq - angular_sq
    coefficient_b = 2.0 * (safe_face_velocity * angular_face_term).sum(dim=-1)
    coefficient_c = -velocity_sq
    scale = torch.maximum(
        torch.maximum(speed_sq, angular_sq),
        torch.maximum(velocity_sq, torch.ones_like(velocity_sq)),
    )
    linear = coefficient_a.abs() <= _QUADRATIC_REL_TOL * scale
    degenerate = linear & (coefficient_b.abs() <= _QUADRATIC_REL_TOL * scale)
    safe_a = torch.where(linear, torch.ones_like(coefficient_a), coefficient_a)
    safe_b = torch.where(
        coefficient_b.abs() > _QUADRATIC_REL_TOL * scale,
        coefficient_b,
        torch.ones_like(coefficient_b),
    )
    discriminant = coefficient_b * coefficient_b - 4.0 * coefficient_a * coefficient_c
    discriminant_scale = torch.maximum(
        torch.maximum(coefficient_b * coefficient_b, (4.0 * coefficient_a * coefficient_c).abs()),
        torch.ones_like(discriminant),
    )
    no_real_root = (~linear) & (discriminant < -_QUADRATIC_REL_TOL * discriminant_scale)
    sqrt_discriminant = torch.sqrt(discriminant.clamp_min(0.0))
    q_value = -0.5 * (
        coefficient_b + torch.copysign(sqrt_discriminant, coefficient_b)
    )
    q_nonzero = q_value != 0.0
    safe_q = torch.where(q_nonzero, q_value, torch.ones_like(q_value))
    quadratic_root_1 = torch.where(
        q_nonzero, q_value / safe_a, -coefficient_b / (2.0 * safe_a)
    )
    quadratic_root_2 = coefficient_c / safe_q
    root_1 = torch.where(linear, -coefficient_c / safe_b, quadratic_root_1)
    root_2 = torch.where(linear | ~q_nonzero, torch.full_like(root_1, float("nan")), quadratic_root_2)

    def verified_root(root: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_valid = torch.isfinite(root) & (root > 0.0)
        safe_root = torch.where(candidate_valid, root, torch.ones_like(root))
        omega = omega_native_command * safe_root.unsqueeze(-1)
        site_velocity = safe_face_velocity - _cross(omega, face_offset_w)
        residual = (
            torch.linalg.vector_norm(site_velocity, dim=-1) / safe_reference_speed
            - safe_root
        ).abs()
        verified = candidate_valid & (
            residual <= _ROOT_RESIDUAL_REL_TOL * torch.maximum(torch.ones_like(safe_root), safe_root)
        )
        return verified, site_velocity, omega

    verified_1, site_velocity_1, omega_1 = verified_root(root_1)
    verified_2, site_velocity_2, omega_2 = verified_root(root_2)
    verified_2 = verified_2 & ~(root_2 == root_1)
    verified_count = verified_1.to(dtype=torch.int64) + verified_2.to(dtype=torch.int64)
    unique_root = verified_count == 1
    selected_rate = torch.where(verified_1, root_1, root_2)
    site_velocity = torch.where(verified_1.unsqueeze(-1), site_velocity_1, site_velocity_2)
    command_omega = torch.where(verified_1.unsqueeze(-1), omega_1, omega_2)

    geometry_unsolved = antipodal | reconstruction_bad | degenerate | no_real_root | ~unique_root
    required_speed = torch.linalg.vector_norm(site_velocity, dim=-1)
    # The scalar authority does not retain the quadratic representative as the
    # teacher clock.  After verifying that root it canonicalizes once more as
    # ||v_site|| / reference_speed, checks the two agree within the root
    # residual tolerance, and uses this value for all timing.  Keeping the raw
    # root here differs by one ULP on real fixed tapes and can flip a closed
    # prewait boundary.
    canonical_rate = required_speed / safe_reference_speed
    root_rate_disagrees = (
        (canonical_rate - selected_rate).abs()
        > _ROOT_RESIDUAL_REL_TOL
        * torch.maximum(torch.ones_like(selected_rate), selected_rate)
    )
    geometry_unsolved = geometry_unsolved | root_rate_disagrees
    rate_in_bounds = (
        (canonical_rate >= safe_rate_min - _RATE_BOUNDARY_ABS_TOL)
        & (canonical_rate <= safe_rate_max + _RATE_BOUNDARY_ABS_TOL)
    )
    scaled_hit = safe_hit / canonical_rate
    scaled_cycle = safe_cycle / canonical_rate
    prewait = safe_ttc - scaled_hit
    prewait_in_bounds = (prewait >= safe_reaction) & (prewait <= _MAX_PRE_SWING_WAIT_S)
    cycle_in_bounds = (
        prewait + scaled_cycle + safe_close
        <= safe_episode + _EPISODE_HORIZON_ABS_TOL
    )

    reason = torch.full(
        (ball.shape[0],),
        CONSTRUCTION_REASON_ADMITTED,
        dtype=torch.int64,
        device=device,
    )
    reason = torch.where(
        geometry_unsolved | ~fault_free,
        torch.full_like(reason, CONSTRUCTION_REASON_TEACHER_SITE_RATE_GEOMETRY_UNSOLVED),
        reason,
    )
    reason = torch.where(
        (reason == CONSTRUCTION_REASON_ADMITTED) & ~rate_in_bounds,
        torch.full_like(reason, CONSTRUCTION_REASON_TEACHER_RATE_OUT_OF_BOUNDS),
        reason,
    )
    reason = torch.where(
        (reason == CONSTRUCTION_REASON_ADMITTED) & ~prewait_in_bounds,
        torch.full_like(reason, CONSTRUCTION_REASON_PRE_SWING_WAIT_OUT_OF_BOUNDS),
        reason,
    )
    reason = torch.where(
        (reason == CONSTRUCTION_REASON_ADMITTED) & ~cycle_in_bounds,
        torch.full_like(reason, CONSTRUCTION_REASON_CYCLE_EXCEEDS_EPISODE_HORIZON),
        reason,
    )
    admitted = reason == CONSTRUCTION_REASON_ADMITTED

    ball_offset_local = face_offset_local.clone()
    ball_offset_local[:, 1] = ball_offset_local[:, 1] + safe_sign * _BALL_RADIUS_M
    site_target = safe_ball - _quat_rotate(command_quat, ball_offset_local)

    return DeviceExactFaceTimingResult(
        racket_command_quat_wxyz=_masked(command_quat.to(dtype=source_dtype), admitted),
        racket_site_target_w_m=_masked(site_target.to(dtype=source_dtype), admitted),
        racket_face_center_velocity_w_mps=_masked(safe_face_velocity.to(dtype=source_dtype), admitted),
        racket_site_velocity_w_mps=_masked(site_velocity.to(dtype=source_dtype), admitted),
        racket_command_angular_velocity_w_radps=_masked(command_omega.to(dtype=source_dtype), admitted),
        required_racket_site_speed_mps=_masked(required_speed.to(dtype=source_dtype), admitted),
        teacher_rate=_masked(canonical_rate.to(dtype=source_dtype), admitted),
        scaled_t_hit_s=_masked(scaled_hit.to(dtype=source_dtype), admitted),
        scaled_t_cycle_s=_masked(scaled_cycle.to(dtype=source_dtype), admitted),
        pre_swing_wait_s=_masked(prewait.to(dtype=source_dtype), admitted),
        construction_reason=reason,
        producer_fault_bits=fault,
        admitted=admitted,
    )


__all__ = [
    "CONSTRUCTION_REASON_ADMITTED",
    "CONSTRUCTION_REASON_TEACHER_SITE_RATE_GEOMETRY_UNSOLVED",
    "CONSTRUCTION_REASON_TEACHER_RATE_OUT_OF_BOUNDS",
    "CONSTRUCTION_REASON_PRE_SWING_WAIT_OUT_OF_BOUNDS",
    "CONSTRUCTION_REASON_CYCLE_EXCEEDS_EPISODE_HORIZON",
    "PRODUCER_FAULT_NONFINITE_GEOMETRY_INPUT",
    "PRODUCER_FAULT_INVALID_FACE_SIGN",
    "PRODUCER_FAULT_INVALID_REFERENCE_QUATERNION",
    "PRODUCER_FAULT_INVALID_SOLVED_NORMAL",
    "PRODUCER_FAULT_INVALID_RATE_PROFILE",
    "PRODUCER_FAULT_INVALID_TIMING_PROFILE",
    "PRODUCER_FAULT_MASK",
    "DeviceExactFaceTimingResult",
    "solve_exact_face_timing_device",
]
