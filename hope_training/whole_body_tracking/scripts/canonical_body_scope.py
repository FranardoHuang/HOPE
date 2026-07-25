#!/usr/bin/env python3
"""Pure-math body-scope compiler for canonical A3 motion candidates.

This module deliberately stops before FK, file I/O, and time-law synthesis.  It
only decides which source degrees of freedom survive and expresses the source
root trajectory in the canonical-ready station.

Two scopes are supported:

``upper``
    Freeze the twelve leg joints and two head joints at canonical ready.  Keep
    both arms and fold the source pelvis' complete *relative* SO(3) motion
    (relative to source frame 0) into the Z-X-Y waist chain.  The global source
    station orientation is not copied.  The output root is canonical ready at
    every frame.  Deleted root translation is measured relative to source
    frame 0 and fails closed above an explicit limit.

``full``
    Keep all 31 source joints.  Apply one rigid SE(2) transform to the complete
    root trajectory so source frame-0 XY/yaw aligns with canonical ready while
    preserving the source-local root and leg motion.  Z is unchanged unless a
    bounded, explicit constant grounding correction is requested; that request
    is reported as a new derived-asset operation.

Quaternion convention is ``wxyz`` throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ALGORITHM = "a3-canonical-body-scope-v2"
REPORT_SCHEMA_VERSION = 1

WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
HEAD_JOINT_NAMES = (
    "head_yaw_joint",
    "head_pitch_joint",
)
LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEG_JOINT_NAMES = (
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
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
UPPER_ACTIVE_JOINT_NAMES = WAIST_JOINT_NAMES + ARM_JOINT_NAMES
UPPER_READY_FIXED_JOINT_NAMES = LEG_JOINT_NAMES + HEAD_JOINT_NAMES
A3_JOINT_NAMES = UPPER_ACTIVE_JOINT_NAMES + UPPER_READY_FIXED_JOINT_NAMES

DEFAULT_MAX_REMOVED_ROOT_TRANSLATION_M = 0.10
DEFAULT_WAIST_YAW_RANGE_RAD = (math.radians(-150.0), math.radians(150.0))
DEFAULT_WAIST_ROLL_RANGE_RAD = (math.radians(-20.0), math.radians(20.0))
DEFAULT_WAIST_PITCH_RANGE_RAD = (math.radians(-28.0), math.radians(24.0))
DEFAULT_MAX_GROUNDING_CORRECTION_M = 0.05

_QUAT_NORM_TOL = 1.0e-5
_ALIGNMENT_TOL = 1.0e-10
_ROTATION_RECONSTRUCTION_TOL_RAD = 1.0e-7


class BodyScopeError(ValueError):
    """The requested body-scope projection is ambiguous or violates a hard gate."""


@dataclass(frozen=True)
class BodyScopeResult:
    """Arrays produced by body-scope preprocessing plus a JSON-compatible report."""

    joint_pos: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    report: Mapping[str, Any]


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray:
    """Wrap radians to ``[-pi, pi)`` without changing the input shape."""

    value = np.asarray(angle, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    """Quaternion conjugate for arrays whose final dimension is ``wxyz``."""

    q = np.asarray(quat, dtype=np.float64)
    if q.shape == () or q.shape[-1] != 4:
        raise BodyScopeError(f"quaternion shape must end in 4, got {q.shape}")
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product ``left * right`` with NumPy broadcasting."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape == () or a.shape[-1] != 4:
        raise BodyScopeError(f"left quaternion shape must end in 4, got {a.shape}")
    if b.shape == () or b.shape[-1] != 4:
        raise BodyScopeError(f"right quaternion shape must end in 4, got {b.shape}")
    try:
        a, b = np.broadcast_arrays(a, b)
    except ValueError as exc:
        raise BodyScopeError(
            f"quaternion arrays cannot broadcast: {a.shape} versus {b.shape}"
        ) from exc
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def yaw_quat_wxyz(yaw_rad: np.ndarray | float) -> np.ndarray:
    """Return world-Z yaw quaternions in ``wxyz`` order."""

    yaw = np.asarray(yaw_rad, dtype=np.float64)
    half = 0.5 * yaw
    out = np.zeros(yaw.shape + (4,), dtype=np.float64)
    out[..., 0] = np.cos(half)
    out[..., 3] = np.sin(half)
    return out


def yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Extract ZYX world-Z yaw from finite, unit ``wxyz`` quaternions."""

    q = _unit_quaternions(quat, label="quaternion")
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_from_quat_wxyz(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract ZYX roll and pitch from finite, unit ``wxyz`` quaternions."""

    q = _unit_quaternions(quat, label="quaternion")
    w, x, y, z = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, np.arcsin(sin_pitch)


def quaternion_distance_rad(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Sign-insensitive SO(3) distance between broadcastable quaternions."""

    a = _unit_quaternions(left, label="left quaternion")
    b = _unit_quaternions(right, label="right quaternion")
    relative = quat_multiply_wxyz(quat_conjugate_wxyz(a), b)
    scalar = np.clip(np.abs(relative[..., 0]), 0.0, 1.0)
    vector_norm = np.linalg.norm(relative[..., 1:], axis=-1)
    return 2.0 * np.arctan2(vector_norm, scalar)


def quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert finite unit ``wxyz`` quaternions to rotation matrices."""

    q = _unit_quaternions(quat, label="quaternion")
    w, x, y, z = np.moveaxis(q, -1, 0)
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[..., 0, 1] = 2.0 * (x * y - z * w)
    out[..., 0, 2] = 2.0 * (x * z + y * w)
    out[..., 1, 0] = 2.0 * (x * y + z * w)
    out[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[..., 1, 2] = 2.0 * (y * z - x * w)
    out[..., 2, 0] = 2.0 * (x * z - y * w)
    out[..., 2, 1] = 2.0 * (y * z + x * w)
    out[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


def waist_zxy_matrix(
    yaw_rad: np.ndarray,
    roll_rad: np.ndarray,
    pitch_rad: np.ndarray,
) -> np.ndarray:
    """Compose the A3 waist chain ``Rz(yaw) @ Rx(roll) @ Ry(pitch)``."""

    yaw, roll, pitch = np.broadcast_arrays(
        np.asarray(yaw_rad, dtype=np.float64),
        np.asarray(roll_rad, dtype=np.float64),
        np.asarray(pitch_rad, dtype=np.float64),
    )
    if not (
        np.isfinite(yaw).all()
        and np.isfinite(roll).all()
        and np.isfinite(pitch).all()
    ):
        raise BodyScopeError("waist angles contain NaN/Inf")
    cz, sz = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    out = np.empty(yaw.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = cz * cy - sz * sx * sy
    out[..., 0, 1] = -sz * cx
    out[..., 0, 2] = cz * sy + sz * sx * cy
    out[..., 1, 0] = sz * cy + cz * sx * sy
    out[..., 1, 1] = cz * cx
    out[..., 1, 2] = sz * sy - cz * sx * cy
    out[..., 2, 0] = -cx * sy
    out[..., 2, 1] = sx
    out[..., 2, 2] = cx * cy
    return out


def decompose_waist_zxy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose matrices as A3 waist Z-X-Y angles on the non-gimbal branch."""

    rotation = np.asarray(matrix, dtype=np.float64)
    if (
        rotation.shape == ()
        or rotation.shape[-2:] != (3, 3)
        or not np.isfinite(rotation).all()
    ):
        raise BodyScopeError(
            f"waist rotation must be a finite (...,3,3) array, got {rotation.shape}"
        )
    roll = np.arcsin(np.clip(rotation[..., 2, 1], -1.0, 1.0))
    cos_roll = np.cos(roll)
    if np.any(np.abs(cos_roll) < 1.0e-8):
        raise BodyScopeError("waist Z-X-Y decomposition reached a gimbal singularity")
    yaw = np.arctan2(-rotation[..., 0, 1], rotation[..., 1, 1])
    pitch = np.arctan2(-rotation[..., 2, 0], rotation[..., 2, 2])
    if yaw.ndim == 1:
        yaw = np.unwrap(yaw)
    return yaw, roll, pitch


def rotate_xy(points_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Apply one world-Z rotation to an ``(..., 2)`` point array."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape == () or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise BodyScopeError(f"points_xy must be a finite (...,2) array, got {points.shape}")
    yaw = _finite_scalar(yaw_rad, label="SE2 yaw")
    c = math.cos(yaw)
    s = math.sin(yaw)
    out = np.empty_like(points, dtype=np.float64)
    out[..., 0] = c * points[..., 0] - s * points[..., 1]
    out[..., 1] = s * points[..., 0] + c * points[..., 1]
    return out


def apply_atomic_se2(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    *,
    yaw_rad: float,
    translation_xy_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one rigid world-frame SE(2) transform to a complete root trajectory.

    Z is copied bit-for-bit in value; callers may apply a separately gated
    grounding correction afterwards.
    """

    pos, quat = _validate_root_trajectory(root_pos_w, root_quat_w)
    translation = np.asarray(translation_xy_m, dtype=np.float64)
    if translation.shape != (2,) or not np.isfinite(translation).all():
        raise BodyScopeError(
            f"translation_xy_m must be a finite length-2 vector, got {translation.shape}"
        )
    yaw = _finite_scalar(yaw_rad, label="SE2 yaw")
    out_pos = pos.copy()
    out_pos[:, :2] = rotate_xy(pos[:, :2], yaw) + translation[None, :]
    out_quat = quat_multiply_wxyz(yaw_quat_wxyz(yaw), quat)
    out_quat /= np.linalg.norm(out_quat, axis=1)[:, None]
    return out_pos, out_quat


def project_upper_body_scope(
    *,
    source_joint_pos: np.ndarray,
    source_root_pos_w: np.ndarray,
    source_root_quat_w: np.ndarray,
    joint_names: Sequence[str],
    canonical_ready_joint_pos: np.ndarray,
    canonical_ready_root_pos_w: np.ndarray,
    canonical_ready_root_quat_w: np.ndarray,
    max_removed_root_translation_m: float = DEFAULT_MAX_REMOVED_ROOT_TRANSLATION_M,
    waist_yaw_range_rad: Sequence[float] = DEFAULT_WAIST_YAW_RANGE_RAD,
    waist_roll_range_rad: Sequence[float] = DEFAULT_WAIST_ROLL_RANGE_RAD,
    waist_pitch_range_rad: Sequence[float] = DEFAULT_WAIST_PITCH_RANGE_RAD,
) -> BodyScopeResult:
    """Compile an upper-body source into a complete 31-DOF canonical-root path."""

    names, source_q = _validate_joint_trajectory(source_joint_pos, joint_names)
    source_pos, source_quat = _validate_root_trajectory(
        source_root_pos_w, source_root_quat_w, expected_frames=source_q.shape[0]
    )
    ready_q = _finite_vector(
        canonical_ready_joint_pos, length=len(names), label="canonical ready joint_pos"
    )
    ready_pos = _finite_vector(
        canonical_ready_root_pos_w, length=3, label="canonical ready root_pos_w"
    )
    ready_quat = _unit_quaternions(
        np.asarray(canonical_ready_root_quat_w, dtype=np.float64),
        label="canonical ready root_quat_w",
    )
    if ready_quat.shape != (4,):
        raise BodyScopeError(
            f"canonical ready root_quat_w must have shape (4,), got {ready_quat.shape}"
        )

    translation_limit = _positive_limit(
        max_removed_root_translation_m,
        label="max_removed_root_translation_m",
    )
    waist_ranges = {
        "yaw": _finite_ordered_range(waist_yaw_range_rad, label="waist_yaw_range_rad"),
        "roll": _finite_ordered_range(
            waist_roll_range_rad, label="waist_roll_range_rad"
        ),
        "pitch": _finite_ordered_range(
            waist_pitch_range_rad, label="waist_pitch_range_rad"
        ),
    }

    indices = {name: index for index, name in enumerate(names)}
    fixed_indices = np.asarray(
        [indices[name] for name in UPPER_READY_FIXED_JOINT_NAMES], dtype=np.int64
    )
    waist_indices = np.asarray(
        [indices[name] for name in WAIST_JOINT_NAMES], dtype=np.int64
    )

    source_root_rotation = quat_to_matrix_wxyz(source_quat)
    relative_root_rotation = np.einsum(
        "ji,tjk->tik", source_root_rotation[0], source_root_rotation
    )
    source_waist_rotation = waist_zxy_matrix(
        source_q[:, waist_indices[0]],
        source_q[:, waist_indices[1]],
        source_q[:, waist_indices[2]],
    )
    target_waist_rotation = np.einsum(
        "tij,tjk->tik", relative_root_rotation, source_waist_rotation
    )
    waist_yaw, waist_roll, waist_pitch = decompose_waist_zxy(
        target_waist_rotation
    )
    folded_waist = np.column_stack((waist_yaw, waist_roll, waist_pitch))
    output_q = source_q.copy()
    output_q[:, fixed_indices] = ready_q[fixed_indices][None, :]
    output_q[:, waist_indices] = folded_waist

    range_failures: list[str] = []
    for column, axis in enumerate(("yaw", "roll", "pitch")):
        lower, upper = waist_ranges[axis]
        observed_min = float(np.min(folded_waist[:, column]))
        observed_max = float(np.max(folded_waist[:, column]))
        if observed_min < lower - 1.0e-12 or observed_max > upper + 1.0e-12:
            range_failures.append(
                f"waist_{axis} [{math.degrees(observed_min):.6f}, "
                f"{math.degrees(observed_max):.6f}] deg exceeds "
                f"[{math.degrees(lower):.6f}, {math.degrees(upper):.6f}] deg"
            )
    if range_failures:
        raise BodyScopeError("; ".join(range_failures))

    reconstructed_waist = waist_zxy_matrix(
        folded_waist[:, 0], folded_waist[:, 1], folded_waist[:, 2]
    )
    relative_error = np.einsum(
        "tji,tjk->tik", reconstructed_waist, target_waist_rotation
    )
    trace = np.trace(relative_error, axis1=1, axis2=2)
    reconstruction_error = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    max_reconstruction_error = float(np.max(reconstruction_error))
    if max_reconstruction_error > _ROTATION_RECONSTRUCTION_TOL_RAD:
        raise BodyScopeError(
            "pelvis-to-waist SO(3) fold reconstruction failed: "
            f"{max_reconstruction_error:.3e} rad"
        )

    translation_delta = source_pos - source_pos[0]
    translation_norm = np.linalg.norm(translation_delta, axis=1)
    max_translation = float(np.max(translation_norm))

    failures: list[str] = []
    if max_translation > translation_limit + 1.0e-12:
        failures.append(
            "deleted root translation "
            f"{max_translation:.6f} m exceeds {translation_limit:.6f} m"
        )
    if failures:
        raise BodyScopeError("; ".join(failures))

    frame_count = source_q.shape[0]
    output_pos = np.broadcast_to(ready_pos, (frame_count, 3)).copy()
    output_quat = np.broadcast_to(ready_quat, (frame_count, 4)).copy()

    source_yaw = np.unwrap(yaw_from_quat_wxyz(source_quat))
    source_frame0_yaw = float(source_yaw[0])
    ready_yaw = float(yaw_from_quat_wxyz(ready_quat))
    report = _base_report(scope="upper", frame_count=frame_count, joint_names=names)
    report.update(
        {
            "joint_policy": {
                "fixed_to_canonical_ready": list(UPPER_READY_FIXED_JOINT_NAMES),
                "source_preserved": [
                    name
                    for name in UPPER_ACTIVE_JOINT_NAMES
                    if name not in WAIST_JOINT_NAMES
                ],
                "relative_root_so3_fold_joints": list(WAIST_JOINT_NAMES),
            },
            "root_policy": {
                "mode": "fixed_canonical_ready",
                "canonical_ready_root_pos_w": ready_pos.tolist(),
                "canonical_ready_root_quat_wxyz": ready_quat.tolist(),
            },
            "relative_root_so3_fold": {
                "reference": "source_frame0_full_orientation",
                "source_frame0_yaw_rad": source_frame0_yaw,
                "canonical_ready_yaw_rad": ready_yaw,
                "global_frame0_orientation_folded": False,
                "chain_order": "Rz(yaw) @ Rx(roll) @ Ry(pitch)",
                "waist_output_rad": {
                    axis: _range_report(folded_waist[:, column])
                    for column, axis in enumerate(("yaw", "roll", "pitch"))
                },
                "waist_hard_ranges_rad": {
                    axis: list(waist_ranges[axis])
                    for axis in ("yaw", "roll", "pitch")
                },
                "max_reconstruction_error_rad": max_reconstruction_error,
                "passed": True,
            },
            "deleted_root_motion_gate": {
                "reference": "source_frame0_not_global_station",
                "rotation": {
                    "policy": "full_relative_so3_folded_into_waist",
                    "removed_relative_rotation_deg": 0.0,
                },
                "translation": {
                    "max_norm_m": max_translation,
                    "max_abs_xyz_m": np.max(
                        np.abs(translation_delta), axis=0
                    ).tolist(),
                    "limit_m": translation_limit,
                },
                "source_frame0_to_ready_reanchor": {
                    "translation_m": (ready_pos - source_pos[0]).tolist(),
                    "yaw_rad": float(wrap_to_pi(ready_yaw - source_frame0_yaw)),
                    "gated_as_deleted_motion": False,
                },
                "passed": True,
            },
        }
    )
    return BodyScopeResult(output_q, output_pos, output_quat, report)


def align_full_body_scope(
    *,
    source_joint_pos: np.ndarray,
    source_root_pos_w: np.ndarray,
    source_root_quat_w: np.ndarray,
    joint_names: Sequence[str],
    canonical_ready_root_pos_w: np.ndarray,
    canonical_ready_root_quat_w: np.ndarray,
    grounding_z_offset_m: float | None = None,
    max_grounding_correction_m: float = DEFAULT_MAX_GROUNDING_CORRECTION_M,
) -> BodyScopeResult:
    """Re-anchor a full-body source with one atomic SE(2), preserving local motion."""

    names, source_q = _validate_joint_trajectory(source_joint_pos, joint_names)
    source_pos, source_quat = _validate_root_trajectory(
        source_root_pos_w, source_root_quat_w, expected_frames=source_q.shape[0]
    )
    ready_pos = _finite_vector(
        canonical_ready_root_pos_w, length=3, label="canonical ready root_pos_w"
    )
    ready_quat = _unit_quaternions(
        np.asarray(canonical_ready_root_quat_w, dtype=np.float64),
        label="canonical ready root_quat_w",
    )
    if ready_quat.shape != (4,):
        raise BodyScopeError(
            f"canonical ready root_quat_w must have shape (4,), got {ready_quat.shape}"
        )
    grounding_limit = _positive_limit(
        max_grounding_correction_m,
        label="max_grounding_correction_m",
        allow_zero=True,
    )

    source_yaw = np.unwrap(yaw_from_quat_wxyz(source_quat))
    ready_yaw = float(yaw_from_quat_wxyz(ready_quat))
    alignment_yaw = float(wrap_to_pi(ready_yaw - source_yaw[0]))
    rotated_frame0_xy = rotate_xy(source_pos[0, :2], alignment_yaw)
    translation_xy = ready_pos[:2] - rotated_frame0_xy
    output_pos, output_quat = apply_atomic_se2(
        source_pos,
        source_quat,
        yaw_rad=alignment_yaw,
        translation_xy_m=translation_xy,
    )

    grounding_requested = grounding_z_offset_m is not None
    grounding_offset = (
        0.0
        if grounding_z_offset_m is None
        else _finite_scalar(grounding_z_offset_m, label="grounding_z_offset_m")
    )
    if abs(grounding_offset) > grounding_limit + 1.0e-12:
        raise BodyScopeError(
            "grounding correction exceeds its hard limit: "
            f"|{grounding_offset:.6f}| m > {grounding_limit:.6f} m"
        )
    if grounding_requested:
        output_pos[:, 2] = source_pos[:, 2] + grounding_offset

    output_yaw = np.unwrap(yaw_from_quat_wxyz(output_quat))
    xy_error = float(np.linalg.norm(output_pos[0, :2] - ready_pos[:2]))
    yaw_error = float(abs(wrap_to_pi(output_yaw[0] - ready_yaw)))
    if xy_error > _ALIGNMENT_TOL or yaw_error > _ALIGNMENT_TOL:
        raise BodyScopeError(
            "internal SE2 alignment check failed: "
            f"xy error {xy_error:.3e} m, yaw error {yaw_error:.3e} rad"
        )
    source_relative_xy = source_pos[:, :2] - source_pos[0, :2]
    output_relative_xy = output_pos[:, :2] - output_pos[0, :2]
    expected_relative_xy = rotate_xy(source_relative_xy, alignment_yaw)
    local_xy_error = float(
        np.max(np.linalg.norm(output_relative_xy - expected_relative_xy, axis=1))
    )
    source_relative_yaw = source_yaw - source_yaw[0]
    output_relative_yaw = output_yaw - output_yaw[0]
    local_yaw_error = float(
        np.max(np.abs(wrap_to_pi(output_relative_yaw - source_relative_yaw)))
    )
    z_delta_error = float(
        np.max(
            np.abs(
                (output_pos[:, 2] - output_pos[0, 2])
                - (source_pos[:, 2] - source_pos[0, 2])
            )
        )
    )
    if max(local_xy_error, local_yaw_error, z_delta_error) > _ALIGNMENT_TOL:
        raise BodyScopeError(
            "internal SE2 local-motion preservation check failed: "
            f"xy {local_xy_error:.3e}, yaw {local_yaw_error:.3e}, "
            f"z-delta {z_delta_error:.3e}"
        )

    report = _base_report(
        scope="full", frame_count=source_q.shape[0], joint_names=names
    )
    report.update(
        {
            "joint_policy": {
                "source_preserved": list(names),
                "fixed_to_canonical_ready": [],
            },
            "root_policy": {
                "mode": "atomic_se2_frame0_to_canonical_ready",
                "source_frame0_xy_m": source_pos[0, :2].tolist(),
                "source_frame0_yaw_rad": float(source_yaw[0]),
                "canonical_ready_xy_m": ready_pos[:2].tolist(),
                "canonical_ready_yaw_rad": ready_yaw,
                "se2": {
                    "yaw_rad": alignment_yaw,
                    "translation_xy_m": translation_xy.tolist(),
                    "applied_once_to_all_frames": True,
                },
                "frame0_alignment_error": {
                    "xy_m": xy_error,
                    "yaw_rad": yaw_error,
                },
                "local_motion_preservation_error": {
                    "xy_m": local_xy_error,
                    "yaw_rad": local_yaw_error,
                    "z_delta_m": z_delta_error,
                },
            },
            "grounding": {
                "requested_explicitly": grounding_requested,
                "z_policy": (
                    "constant_bounded_grounding_offset"
                    if grounding_requested
                    else "preserve_source_z"
                ),
                "z_offset_m": grounding_offset,
                "max_abs_correction_m": grounding_limit,
                "requires_new_derived_asset_record": grounding_requested,
                "passed": True,
            },
        }
    )
    return BodyScopeResult(source_q.copy(), output_pos, output_quat, report)


def preprocess_body_scope(scope: str, **kwargs: Any) -> BodyScopeResult:
    """Dispatch to the strict ``upper`` or ``full`` pure-math compiler."""

    if scope == "upper":
        return project_upper_body_scope(**kwargs)
    if scope == "full":
        return align_full_body_scope(**kwargs)
    raise BodyScopeError(f"body scope must be 'upper' or 'full', got {scope!r}")


def _validate_joint_trajectory(
    source_joint_pos: np.ndarray, joint_names: Sequence[str]
) -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(str(name) for name in joint_names)
    if len(names) != 31 or len(set(names)) != 31:
        raise BodyScopeError("joint_names must contain exactly 31 unique A3 names")
    expected = set(A3_JOINT_NAMES)
    actual = set(names)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BodyScopeError(
            f"joint_names do not match the A3 31-DOF domain: "
            f"missing={missing}, extra={extra}"
        )
    q = np.asarray(source_joint_pos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 31 or q.shape[0] < 1:
        raise BodyScopeError(
            f"source joint_pos must have shape (T,31) with T>=1, got {q.shape}"
        )
    if not np.isfinite(q).all():
        raise BodyScopeError("source joint_pos contains NaN/Inf")
    return names, q.copy()


def _validate_root_trajectory(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    *,
    expected_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(root_pos_w, dtype=np.float64)
    quat = np.asarray(root_quat_w, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] < 1:
        raise BodyScopeError(
            f"source root_pos_w must have shape (T,3) with T>=1, got {pos.shape}"
        )
    if quat.shape != (pos.shape[0], 4):
        raise BodyScopeError(
            f"source root_quat_w must have shape {(pos.shape[0], 4)}, got {quat.shape}"
        )
    if expected_frames is not None and pos.shape[0] != expected_frames:
        raise BodyScopeError(
            "source root trajectory length does not match joint_pos: "
            f"{pos.shape[0]} versus {expected_frames}"
        )
    if not np.isfinite(pos).all():
        raise BodyScopeError("source root_pos_w contains NaN/Inf")
    return pos.copy(), _unit_quaternions(quat, label="source root_quat_w")


def _unit_quaternions(quat: np.ndarray, *, label: str) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape == () or q.shape[-1] != 4 or not np.isfinite(q).all():
        raise BodyScopeError(f"{label} must be a finite (...,4) array, got {q.shape}")
    norms = np.linalg.norm(q, axis=-1)
    if np.any(norms <= 0.0):
        raise BodyScopeError(f"{label} contains a zero quaternion")
    max_error = float(np.max(np.abs(norms - 1.0)))
    if max_error > _QUAT_NORM_TOL:
        raise BodyScopeError(
            f"{label} is not unit length (max norm error {max_error:.3e})"
        )
    return q / norms[..., None]


def _finite_vector(value: np.ndarray, *, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise BodyScopeError(
            f"{label} must be a finite vector of shape ({length},), got {array.shape}"
        )
    return array.copy()


def _finite_scalar(value: float, *, label: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise BodyScopeError(f"{label} must be a finite scalar") from exc
    if not math.isfinite(scalar):
        raise BodyScopeError(f"{label} must be a finite scalar")
    return scalar


def _positive_limit(value: float, *, label: str, allow_zero: bool = False) -> float:
    scalar = _finite_scalar(value, label=label)
    valid = scalar >= 0.0 if allow_zero else scalar > 0.0
    if not valid:
        operator = "non-negative" if allow_zero else "positive"
        raise BodyScopeError(f"{label} must be finite and {operator}")
    return scalar


def _finite_ordered_range(
    value: Sequence[float], *, label: str
) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise BodyScopeError(f"{label} must be a finite (lower, upper) pair")
    lower, upper = float(array[0]), float(array[1])
    if lower >= upper:
        raise BodyScopeError(f"{label} must satisfy lower < upper")
    return lower, upper


def _range_report(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "first": float(array[0]),
        "last": float(array[-1]),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _base_report(
    *, scope: str, frame_count: int, joint_names: Sequence[str]
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "scope": scope,
        "frame_count": int(frame_count),
        "joint_count": 31,
        "joint_names": list(joint_names),
        "quaternion_order": "wxyz",
        "pure_math_only": {
            "fk_performed": False,
            "file_io_performed": False,
            "time_law_changed": False,
        },
    }
