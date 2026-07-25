#!/usr/bin/env python3
"""Encode a root-pose trajectory for Euclidean path construction.

The canonical motion compiler optimizes joints, root translation, and root
orientation on one geometric path.  Translation is already Euclidean.  Root
orientation is represented by the rotation vector of

    inverse(canonical_ready_quaternion) * root_quaternion

using ``wxyz`` quaternions.  This is a local chart, not a claim that SO(3) is
globally Euclidean.  The codec therefore fails closed near the pi singularity
or when the requested trajectory jumps between incompatible logarithm
branches.

No timing, FK, dynamics, file I/O, or asset publication is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RootPoseCodecError(ValueError):
    """A pose cannot be represented unambiguously in the canonical chart."""


@dataclass(frozen=True)
class RootPoseEncoding:
    """Euclidean ``[world_position_xyz, relative_rotation_vector_xyz]`` rows."""

    coordinates: np.ndarray
    report: dict


_UNIT_TOL = 1.0e-5
_SMALL_ANGLE = 1.0e-10
_DEFAULT_PI_MARGIN_RAD = 1.0e-3
_DEFAULT_MAX_STEP_RAD = np.pi / 2.0


def _unit_quaternions(value: np.ndarray, *, label: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape == () or quat.shape[-1] != 4 or not np.isfinite(quat).all():
        raise RootPoseCodecError(
            f"{label} must be a finite (...,4) wxyz array, got {quat.shape}"
        )
    norm = np.linalg.norm(quat, axis=-1)
    if np.any(norm <= 0.0):
        raise RootPoseCodecError(f"{label} contains a zero quaternion")
    error = float(np.max(np.abs(norm - 1.0)))
    if error > _UNIT_TOL:
        raise RootPoseCodecError(
            f"{label} is not unit length (max norm error {error:.3e})"
        )
    return quat / norm[..., None]


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    """Return the conjugate of unit or non-unit ``wxyz`` quaternions."""

    value = np.asarray(quat, dtype=np.float64)
    if value.shape == () or value.shape[-1] != 4 or not np.isfinite(value).all():
        raise RootPoseCodecError(
            f"quaternion must be a finite (...,4) array, got {value.shape}"
        )
    out = value.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product with NumPy broadcasting."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if (
        a.shape == ()
        or b.shape == ()
        or a.shape[-1] != 4
        or b.shape[-1] != 4
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise RootPoseCodecError("quaternion operands must be finite (...,4) arrays")
    try:
        a, b = np.broadcast_arrays(a, b)
    except ValueError as exc:
        raise RootPoseCodecError(
            f"quaternion operands cannot broadcast: {a.shape} and {b.shape}"
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


def make_quaternion_sign_continuous(quat: np.ndarray) -> np.ndarray:
    """Choose equivalent signs so adjacent rows have a non-negative dot."""

    out = _unit_quaternions(quat, label="quaternion trajectory").copy()
    if out.ndim != 2:
        raise RootPoseCodecError(
            f"quaternion trajectory must have shape (T,4), got {out.shape}"
        )
    for frame in range(1, len(out)):
        if float(out[frame - 1] @ out[frame]) < 0.0:
            out[frame] *= -1.0
    return out


def rotation_vector_to_quat_wxyz(rotation_vector: np.ndarray) -> np.ndarray:
    """Exponential map from rotation vectors to unit ``wxyz`` quaternions."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if (
        vector.shape == ()
        or vector.shape[-1] != 3
        or not np.isfinite(vector).all()
    ):
        raise RootPoseCodecError(
            f"rotation_vector must be a finite (...,3) array, got {vector.shape}"
        )
    angle = np.linalg.norm(vector, axis=-1)
    half = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < _SMALL_ANGLE
    # sin(theta/2)/theta = 1/2 - theta^2/48 + O(theta^4)
    scale[small] = 0.5 - np.square(angle[small]) / 48.0
    scale[~small] = np.sin(half[~small]) / angle[~small]
    return np.concatenate(
        (np.cos(half)[..., None], vector * scale[..., None]), axis=-1
    )


def quat_wxyz_to_rotation_vector(
    quat: np.ndarray,
    *,
    pi_margin_rad: float = _DEFAULT_PI_MARGIN_RAD,
) -> np.ndarray:
    """Principal logarithm of unit quaternions, failing near the pi cut."""

    margin = float(pi_margin_rad)
    if not np.isfinite(margin) or margin <= 0.0 or margin >= np.pi:
        raise RootPoseCodecError("pi_margin_rad must lie strictly between 0 and pi")
    value = _unit_quaternions(quat, label="relative quaternion")
    # q and -q are one SO(3) state.  Choose the representative with w >= 0 so
    # the principal angle is in [0, pi].
    principal = value.copy()
    negative = principal[..., 0] < 0.0
    principal[negative] *= -1.0
    vector = principal[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, principal[..., 0])
    if np.any(angle >= np.pi - margin):
        maximum = float(np.max(angle))
        raise RootPoseCodecError(
            "root orientation reaches the rotation-vector pi branch cut: "
            f"max angle {maximum:.9g} rad"
        )
    scale = np.empty_like(angle)
    small = vector_norm < _SMALL_ANGLE
    # theta/sin(theta/2) -> 2 at zero.
    scale[small] = 2.0
    scale[~small] = angle[~small] / vector_norm[~small]
    return vector * scale[..., None]


def encode_root_pose(
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    *,
    canonical_ready_root_quat_wxyz: np.ndarray,
    pi_margin_rad: float = _DEFAULT_PI_MARGIN_RAD,
    max_rotation_vector_step_rad: float = _DEFAULT_MAX_STEP_RAD,
) -> RootPoseEncoding:
    """Encode root poses as position plus ready-relative rotation vectors."""

    pos = np.asarray(root_pos_w, dtype=np.float64)
    if (
        pos.ndim != 2
        or pos.shape[1] != 3
        or len(pos) < 1
        or not np.isfinite(pos).all()
    ):
        raise RootPoseCodecError(
            f"root_pos_w must be a finite (T,3) array, got {pos.shape}"
        )
    quat = make_quaternion_sign_continuous(root_quat_wxyz)
    if len(quat) != len(pos):
        raise RootPoseCodecError(
            f"root pose length mismatch: {len(pos)} positions, {len(quat)} quaternions"
        )
    ready = _unit_quaternions(
        np.asarray(canonical_ready_root_quat_wxyz, dtype=np.float64),
        label="canonical ready root quaternion",
    )
    if ready.shape != (4,):
        raise RootPoseCodecError(
            f"canonical ready root quaternion must have shape (4,), got {ready.shape}"
        )
    relative = quat_multiply_wxyz(quat_conjugate_wxyz(ready), quat)
    rotation_vector = quat_wxyz_to_rotation_vector(
        relative, pi_margin_rad=pi_margin_rad
    )
    maximum_step = (
        0.0
        if len(rotation_vector) < 2
        else float(np.max(np.linalg.norm(np.diff(rotation_vector, axis=0), axis=1)))
    )
    step_limit = float(max_rotation_vector_step_rad)
    if not np.isfinite(step_limit) or step_limit <= 0.0:
        raise RootPoseCodecError(
            "max_rotation_vector_step_rad must be finite and positive"
        )
    if maximum_step > step_limit:
        raise RootPoseCodecError(
            "root rotation-vector chart is discontinuous at source sampling: "
            f"max step {maximum_step:.9g} > {step_limit:.9g} rad"
        )
    coordinates = np.concatenate((pos, rotation_vector), axis=1)
    return RootPoseEncoding(
        coordinates=coordinates,
        report={
            "algorithm": "canonical_ready_relative_so3_log_v1",
            "quaternion_order": "wxyz",
            "coordinate_order": [
                "root_x_w",
                "root_y_w",
                "root_z_w",
                "root_rotvec_x_ready",
                "root_rotvec_y_ready",
                "root_rotvec_z_ready",
            ],
            "frames": int(len(pos)),
            "max_relative_angle_rad": float(
                np.max(np.linalg.norm(rotation_vector, axis=1))
            ),
            "max_rotation_vector_step_rad": maximum_step,
            "pi_margin_rad": float(pi_margin_rad),
            "limitations": [
                "local SO3 chart only; trajectories near pi are rejected",
                "Euclidean interpolation is not a global SO3 geodesic optimizer",
            ],
        },
    )


def decode_root_pose(
    coordinates: np.ndarray,
    *,
    canonical_ready_root_quat_wxyz: np.ndarray,
    pi_margin_rad: float = _DEFAULT_PI_MARGIN_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode position/rotation-vector rows to world poses."""

    encoded = np.asarray(coordinates, dtype=np.float64)
    if (
        encoded.ndim != 2
        or encoded.shape[1] != 6
        or len(encoded) < 1
        or not np.isfinite(encoded).all()
    ):
        raise RootPoseCodecError(
            f"coordinates must be a finite (T,6) array, got {encoded.shape}"
        )
    ready = _unit_quaternions(
        np.asarray(canonical_ready_root_quat_wxyz, dtype=np.float64),
        label="canonical ready root quaternion",
    )
    if ready.shape != (4,):
        raise RootPoseCodecError(
            f"canonical ready root quaternion must have shape (4,), got {ready.shape}"
        )
    rotation_vector = encoded[:, 3:]
    maximum = float(np.max(np.linalg.norm(rotation_vector, axis=1)))
    if maximum >= np.pi - float(pi_margin_rad):
        raise RootPoseCodecError(
            "encoded rotation vector reaches the pi branch cut: "
            f"max angle {maximum:.9g} rad"
        )
    relative = rotation_vector_to_quat_wxyz(rotation_vector)
    world = quat_multiply_wxyz(ready, relative)
    world /= np.linalg.norm(world, axis=1)[:, None]
    world = make_quaternion_sign_continuous(world)
    return encoded[:, :3].copy(), world


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.moveaxis(vector, -1, 0)
    zeros = np.zeros_like(x)
    return np.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), axis=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def _quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = _unit_quaternions(quaternion, label="quaternion")
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _left_jacobian_and_directional_derivative(
    rotation_vector: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``J_left(phi)`` and ``dJ_left(phi)/ds`` for ``dphi/ds``.

    The derivative is analytic, including its small-angle series.  It is used
    to preserve the second-order path state needed by grounded inverse
    dynamics; finite-differencing quaternions or sampled root velocities is
    deliberately not an implementation fallback.
    """

    vector = np.asarray(rotation_vector, dtype=np.float64)
    tangent = np.asarray(direction, dtype=np.float64)
    if (
        vector.shape == ()
        or vector.shape[-1] != 3
        or vector.shape != tangent.shape
        or not np.isfinite(vector).all()
        or not np.isfinite(tangent).all()
    ):
        raise RootPoseCodecError(
            "rotation_vector and direction must be finite arrays with "
            "identical (...,3) shape"
        )

    angle = np.linalg.norm(vector, axis=-1)
    angle2 = angle * angle
    angle4 = angle2 * angle2
    angle6 = angle4 * angle2
    cross = _skew(vector)
    cross_rate = _skew(tangent)
    cross2 = np.matmul(cross, cross)
    identity = np.broadcast_to(np.eye(3), cross.shape)

    coefficient_a = np.empty_like(angle)
    coefficient_b = np.empty_like(angle)
    coefficient_a_rate_over_dot = np.empty_like(angle)
    coefficient_b_rate_over_dot = np.empty_like(angle)
    # The derivative formulas contain cancellations one order earlier than the
    # coefficients themselves, so use the series over a modest neighbourhood.
    small = angle < 1.0e-2
    coefficient_a[small] = (
        0.5
        - angle2[small] / 24.0
        + angle4[small] / 720.0
        - angle6[small] / 40320.0
    )
    coefficient_b[small] = (
        1.0 / 6.0
        - angle2[small] / 120.0
        + angle4[small] / 5040.0
        - angle6[small] / 362880.0
    )
    # These are (dA/dtheta)/theta and (dB/dtheta)/theta.  Multiplying by
    # dot(phi, dphi/ds) avoids an undefined theta_dot at the origin.
    coefficient_a_rate_over_dot[small] = (
        -1.0 / 12.0
        + angle2[small] / 180.0
        - angle4[small] / 6720.0
        + angle6[small] / 453600.0
    )
    coefficient_b_rate_over_dot[small] = (
        -1.0 / 60.0
        + angle2[small] / 1260.0
        - angle4[small] / 60480.0
        + angle6[small] / 4989600.0
    )

    large = ~small
    coefficient_a[large] = (
        1.0 - np.cos(angle[large])
    ) / angle2[large]
    coefficient_b[large] = (
        angle[large] - np.sin(angle[large])
    ) / (angle[large] * angle2[large])
    coefficient_a_rate_over_dot[large] = (
        np.sin(angle[large]) / (angle[large] ** 3)
        - 2.0
        * (1.0 - np.cos(angle[large]))
        / (angle[large] ** 4)
    )
    coefficient_b_rate_over_dot[large] = (
        (1.0 - np.cos(angle[large])) / (angle[large] ** 4)
        - 3.0
        * (angle[large] - np.sin(angle[large]))
        / (angle[large] ** 5)
    )

    left_jacobian = (
        identity
        + coefficient_a[..., None, None] * cross
        + coefficient_b[..., None, None] * cross2
    )
    directional_dot = np.sum(vector * tangent, axis=-1)
    coefficient_a_rate = coefficient_a_rate_over_dot * directional_dot
    coefficient_b_rate = coefficient_b_rate_over_dot * directional_dot
    left_jacobian_rate = (
        coefficient_a_rate[..., None, None] * cross
        + coefficient_a[..., None, None] * cross_rate
        + coefficient_b_rate[..., None, None] * cross2
        + coefficient_b[..., None, None]
        * (
            np.matmul(cross_rate, cross)
            + np.matmul(cross, cross_rate)
        )
    )
    return left_jacobian, left_jacobian_rate


def rotation_vector_rate_to_world_angular_velocity(
    rotation_vector: np.ndarray,
    rotation_vector_rate: np.ndarray,
    *,
    canonical_ready_root_quat_wxyz: np.ndarray,
    pi_margin_rad: float = _DEFAULT_PI_MARGIN_RAD,
) -> np.ndarray:
    """Map ready-relative rotation-vector rates to WORLD angular velocity.

    With ``R_world = R_ready Exp(phi^)``, the exact differential is
    ``omega_world = R_ready J_left(phi) phi_dot``.  Treating ``phi_dot`` itself
    as angular velocity would only be valid infinitesimally near zero.
    """

    vector = np.asarray(rotation_vector, dtype=np.float64)
    rate = np.asarray(rotation_vector_rate, dtype=np.float64)
    if (
        vector.shape == ()
        or vector.shape[-1] != 3
        or vector.shape != rate.shape
        or not np.isfinite(vector).all()
        or not np.isfinite(rate).all()
    ):
        raise RootPoseCodecError(
            "rotation_vector and rate must be finite arrays with identical (...,3) shape"
        )
    angle = np.linalg.norm(vector, axis=-1)
    margin = float(pi_margin_rad)
    if not np.isfinite(margin) or margin <= 0.0 or margin >= np.pi:
        raise RootPoseCodecError("pi_margin_rad must be finite and inside (0, pi)")
    if np.any(angle >= np.pi - margin):
        raise RootPoseCodecError("rotation vector rate reaches the pi branch cut")

    left_jacobian, _ = _left_jacobian_and_directional_derivative(
        vector, rate
    )
    omega_ready = np.einsum("...ij,...j->...i", left_jacobian, rate)

    ready = _unit_quaternions(
        np.asarray(canonical_ready_root_quat_wxyz, dtype=np.float64),
        label="canonical ready root quaternion",
    )
    if ready.shape != (4,):
        raise RootPoseCodecError(
            f"canonical ready root quaternion must have shape (4,), got {ready.shape}"
        )
    ready_rotation = _quat_to_matrix_wxyz(ready)
    return np.einsum("ij,...j->...i", ready_rotation, omega_ready)


def rotation_vector_derivatives_to_world_angular_kinematics(
    rotation_vector: np.ndarray,
    first_derivative: np.ndarray,
    second_derivative: np.ndarray,
    *,
    canonical_ready_root_quat_wxyz: np.ndarray,
    pi_margin_rad: float = _DEFAULT_PI_MARGIN_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a rotation-vector 2-jet to world angular velocity/acceleration.

    The derivatives may be with respect to time or any scalar path parameter.
    If they are ``phi_s`` and ``phi_ss``, the returned arrays are the exact
    world angular path tangent and curvature.  For
    ``R_world = R_ready Exp(phi^)``:

    ``omega = R_ready J_left(phi) phi_s``

    ``alpha = R_ready (J_left(phi) phi_ss + dJ_left/ds phi_s)``.

    This is the producer interface for an exact collocation trace.  It never
    derives second-order state from 50 Hz pose or velocity differences.
    """

    vector = np.asarray(rotation_vector, dtype=np.float64)
    first = np.asarray(first_derivative, dtype=np.float64)
    second = np.asarray(second_derivative, dtype=np.float64)
    if (
        vector.shape == ()
        or vector.shape[-1] != 3
        or vector.shape != first.shape
        or vector.shape != second.shape
        or not np.isfinite(vector).all()
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise RootPoseCodecError(
            "rotation_vector and its first/second derivatives must be finite "
            "arrays with identical (...,3) shape"
        )
    angle = np.linalg.norm(vector, axis=-1)
    margin = float(pi_margin_rad)
    if not np.isfinite(margin) or margin <= 0.0 or margin >= np.pi:
        raise RootPoseCodecError("pi_margin_rad must be finite and inside (0, pi)")
    if np.any(angle >= np.pi - margin):
        raise RootPoseCodecError(
            "rotation vector derivatives reach the pi branch cut"
        )

    left_jacobian, left_jacobian_rate = (
        _left_jacobian_and_directional_derivative(vector, first)
    )
    omega_ready = np.einsum("...ij,...j->...i", left_jacobian, first)
    alpha_ready = (
        np.einsum("...ij,...j->...i", left_jacobian, second)
        + np.einsum(
            "...ij,...j->...i", left_jacobian_rate, first
        )
    )
    ready = _unit_quaternions(
        np.asarray(canonical_ready_root_quat_wxyz, dtype=np.float64),
        label="canonical ready root quaternion",
    )
    if ready.shape != (4,):
        raise RootPoseCodecError(
            f"canonical ready root quaternion must have shape (4,), got {ready.shape}"
        )
    ready_rotation = _quat_to_matrix_wxyz(ready)
    return (
        np.einsum("ij,...j->...i", ready_rotation, omega_ready),
        np.einsum("ij,...j->...i", ready_rotation, alpha_ready),
    )


def root_coordinate_velocity_to_world_twist(
    coordinates: np.ndarray,
    coordinate_velocity: np.ndarray,
    *,
    canonical_ready_root_quat_wxyz: np.ndarray,
    pi_margin_rad: float = _DEFAULT_PI_MARGIN_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``[position, rotvec]`` rates to WORLD linear/angular velocity."""

    pose = np.asarray(coordinates, dtype=np.float64)
    velocity = np.asarray(coordinate_velocity, dtype=np.float64)
    if (
        pose.ndim != 2
        or pose.shape[1] != 6
        or pose.shape != velocity.shape
        or not np.isfinite(pose).all()
        or not np.isfinite(velocity).all()
    ):
        raise RootPoseCodecError(
            "coordinates and coordinate_velocity must be finite arrays with "
            "identical (T,6) shape"
        )
    angular = rotation_vector_rate_to_world_angular_velocity(
        pose[:, 3:],
        velocity[:, 3:],
        canonical_ready_root_quat_wxyz=canonical_ready_root_quat_wxyz,
        pi_margin_rad=pi_margin_rad,
    )
    return velocity[:, :3].copy(), angular


__all__ = [
    "RootPoseCodecError",
    "RootPoseEncoding",
    "decode_root_pose",
    "encode_root_pose",
    "make_quaternion_sign_continuous",
    "quat_conjugate_wxyz",
    "quat_multiply_wxyz",
    "quat_wxyz_to_rotation_vector",
    "root_coordinate_velocity_to_world_twist",
    "rotation_vector_derivatives_to_world_angular_kinematics",
    "rotation_vector_rate_to_world_angular_velocity",
    "rotation_vector_to_quat_wxyz",
]
