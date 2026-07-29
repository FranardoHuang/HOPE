"""Versioned exact A3 racket-face contact geometry.

``exact_face_contact_v2`` distinguishes the official racket control site,
the selected rubber face centre, and the ball centre at contact.  The
canonical numeric payload in this module is the single executable source for
both the production ActionBall runtime and
``scripts/racket_geometry_contract.py``; the audit script no longer owns a
second copy of the measured constants.

The scalar helpers intentionally use only the Python standard library.  They
can validate serialized receipts before Torch/Isaac exists.  The small Torch
helpers import Torch lazily and are used by the live scorer/physical-ball
paths on CPU or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence, Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Matrix3 = Tuple[Vec3, Vec3, Vec3]

EXACT_FACE_CONTACT_SCHEMA_VERSION = 2
EXACT_FACE_CONTACT_KIND = "exact_face_contact_v2"

RACKET_SITE_OFFSET_WRIST_M: Vec3 = (0.21021, 0.032078, 0.032036)
LEGACY_ISAAC_SITE_OFFSET_WRIST_M: Vec3 = (
    0.210211399202899,
    0.0320784994676765,
    0.0320358706296689,
)
FACE_AREA_CENTER_XZ_FROM_SITE_M = (0.000893694377, 0.000893694377)
RED_OUTER_Y_FROM_SITE_M = 0.0
BLACK_OUTER_Y_FROM_SITE_M = -0.013207999989
BALL_RADIUS_M = 0.020000000148
RED_SELECTED_FACE_MESH_SHA256 = (
    "94182ec1c7c64db8c5ec7ce5f9aad44d427f433a6aae5cf23aa655e077633842"
)
BLACK_SELECTED_FACE_MESH_SHA256 = (
    "5f0e772ea9ed81e5b70f5dfb4ded49f9d269c54c893249857209f85168361b1b"
)
# Measured from the exact selected outer-face triangles in the two pinned
# binary STLs above.  Both faces have the same minimum distance from the
# canonical face-area centre to their closed outer boundary.
SELECTED_FACE_CENTER_TO_BOUNDARY_MIN_M = 0.06476387610847782
# The formal MuJoCo fitted-ball gate requires strictly more than one ball
# radius plus this additional clearance to every selected-face boundary.
FORMAL_FACE_EDGE_GUARD_M = 0.0005
# A strict tangential-distance check against this derived radius is a
# conservative subset of the exact mesh predicate.  This is not another
# hand-tuned "capture radius": every admitted point is guaranteed to satisfy
# the formal edge guard, although valid hits outside this inscribed circle are
# intentionally rejected.
SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M = (
    SELECTED_FACE_CENTER_TO_BOUNDARY_MIN_M
    - BALL_RADIUS_M
    - FORMAL_FACE_EDGE_GUARD_M
)
SELECTED_FACE_SWEEP_CLEARANCE_TOLERANCE_M = 1.0e-7
SELECTED_FACE_SWEEP_BISECTION_STEPS = 24
# Back-propagating the nominal incoming ball state in no larger steps than
# the finest formal MuJoCo fitted-ball adjudication step avoids making the
# runtime's control-step start state depend on one coarse 20 ms RK4 solve.
SELECTED_FACE_SWEEP_BALL_BACKPROP_MAX_DT_S = 0.0005
POLAR_ROTATION_SINGULAR_TOLERANCE = 1.0e-12
OFFICIAL_RED_BALL_CENTER_FROM_SITE_M: Vec3 = (
    0.000940485576,
    0.020000000164,
    0.000940485600,
)
RED_FACE_SIGN = 1
BLACK_FACE_SIGN = -1
# The ActionBall solver path stores prototype direction/speed and reference
# kinematics in float32 tensors before this dependency-light scalar solve.
# Reconstructing one native-rate vector can therefore land a few float32 ULPs
# outside an exact [min,max] boundary.  This is an explicit, SHA-bound numeric
# seam, not a general relaxation of the teacher-rate support.
TEACHER_RATE_BOUNDARY_ABS_TOL = 5.0e-7

GEOMETRY_SOURCE_PAYLOAD = {
    "schema_version": EXACT_FACE_CONTACT_SCHEMA_VERSION,
    "kind": EXACT_FACE_CONTACT_KIND,
    "local_frame": "official_pingpang_red_Link_origin_MJCF_right_racket_site",
    "official_racket_body_name": "pingpang_red_Link",
    "official_wrist_body_name": "right_wrist_yaw_Link",
    "raw_A_axis_local": [0.0, 1.0, 0.0],
    "racket_site_offset_wrist_m": list(RACKET_SITE_OFFSET_WRIST_M),
    "racket_mount_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    "legacy_isaac_site_offset_wrist_m": list(
        LEGACY_ISAAC_SITE_OFFSET_WRIST_M
    ),
    "face_area_center_xz_from_site_m": list(
        FACE_AREA_CENTER_XZ_FROM_SITE_M
    ),
    "red_outer_y_from_site_m": RED_OUTER_Y_FROM_SITE_M,
    "black_outer_y_from_site_m": BLACK_OUTER_Y_FROM_SITE_M,
    "ball_radius_m": BALL_RADIUS_M,
    "selected_face_mesh_sha256": {
        "red": RED_SELECTED_FACE_MESH_SHA256,
        "black": BLACK_SELECTED_FACE_MESH_SHA256,
    },
    "selected_face_center_to_boundary_min_m": (
        SELECTED_FACE_CENTER_TO_BOUNDARY_MIN_M
    ),
    "formal_face_edge_guard_m": FORMAL_FACE_EDGE_GUARD_M,
    "safe_ball_center_tangential_radius_m": (
        SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M
    ),
    "selected_face_sweep_clearance_tolerance_m": (
        SELECTED_FACE_SWEEP_CLEARANCE_TOLERANCE_M
    ),
    "selected_face_sweep_bisection_steps": (
        SELECTED_FACE_SWEEP_BISECTION_STEPS
    ),
    "selected_face_sweep_ball_backprop_max_dt_s": (
        SELECTED_FACE_SWEEP_BALL_BACKPROP_MAX_DT_S
    ),
    "polar_rotation_singular_tolerance": (
        POLAR_ROTATION_SINGULAR_TOLERANCE
    ),
    "selected_face_sweep_semantics": (
        "control_step_previous_to_nominal_contact;selected_side_clearance_"
        "bracket;full_rotation_matrix_polar_interpolation;site_plus_rotated_"
        "face_offset;cubic_hermite_ball_path_from_substep_backpropagated_"
        "endpoints;fixed_bisection;strict_inscribed_mesh_edge_guard;"
        "off_center_rigid_point_velocity"
    ),
    "official_red_ball_center_from_site_m": list(
        OFFICIAL_RED_BALL_CENTER_FROM_SITE_M
    ),
    "face_signs": {"red": RED_FACE_SIGN, "black": BLACK_FACE_SIGN},
    "teacher_rate_boundary_abs_tolerance": (
        TEACHER_RATE_BOUNDARY_ABS_TOL
    ),
    "teacher_rate_boundary_semantics": (
        "admit_the_raw_verified_rate_within_abs_tolerance_of_declared_"
        "min_or_max_without_clipping_so_rate_equals_site_speed_over_"
        "reference_speed_remains_exact"
    ),
    "position_semantics": (
        "p_ball_center=p_site+R_world_from_racket*"
        "(r_face_center_from_site+ball_radius*n_selected_local)"
    ),
    "velocity_semantics": (
        "v_face_center=v_site+omega_world_cross_"
        "(R_world_from_racket*r_face_center_from_site)"
    ),
    "orientation_semantics": (
        "minimal_world_rotation_maps_reference_raw_A_to_solved_raw_A_"
        "and_left_multiplies_reference_quaternion_to_preserve_twist"
    ),
    "teacher_rate_semantics": (
        "omega_command=rate*minimal_rotation(omega_reference);"
        "v_site=v_face_center-omega_command_cross_r_face_world;"
        "rate=norm(v_site)/reference_site_speed"
    ),
    "teacher_motion_safety_semantics": {
        "scope": (
            "entire_reference_cycle_all_robot_links_and_full_racket_geometry"
        ),
        "forbidden_world_geometry": [
            "table_top",
            "table_edges",
            "table_underside",
            "net_mesh",
            "net_posts",
        ],
        "verification": (
            "continuous_or_conservative_swept_collision_over_full_t_cycle;"
            "strike_frame_only_or_sampled_point_only_checks_are_insufficient"
        ),
        "admission": (
            "required_per_action_before_action_profile_or_task_receipt_is_"
            "training_eligible"
        ),
        "runtime_outcome": (
            "any_table_or_net_contact_is_unsafe_failure_and_never_"
            "difficulty_failure"
        ),
    },
}
GEOMETRY_SOURCE_BYTES = json.dumps(
    GEOMETRY_SOURCE_PAYLOAD,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
GEOMETRY_SOURCE_SHA256 = hashlib.sha256(GEOMETRY_SOURCE_BYTES).hexdigest()


class ExactFaceContactGeometryError(ValueError):
    """Named fail-closed geometry/timing rejection."""

    def __init__(self, reason: str, message: str):
        self.reason = str(reason)
        super().__init__(f"{self.reason}: {message}")


def canonical_teacher_rate_from_site_speed(
    required_site_speed_mps: float,
    reference_site_speed_mps: float,
    rate_min: float,
    rate_max: float,
) -> float:
    """Return the raw verified rate under the one SHA-bound float32 seam.

    The word ``canonical`` means every geometry/timing/receipt caller uses
    this function.  It deliberately does *not* clip to a boundary: clipping
    would make ``rate != required_site_speed/reference_site_speed`` and break
    the coupled rigid-point equation.
    """

    required = _finite(
        required_site_speed_mps,
        name="required_site_speed_mps",
    )
    reference = _finite(
        reference_site_speed_mps,
        name="reference_site_speed_mps",
    )
    lower = _finite(rate_min, name="rate_min")
    upper = _finite(rate_max, name="rate_max")
    if required < 0.0 or reference <= 0.0 or lower <= 0.0 or upper < lower:
        raise ExactFaceContactGeometryError(
            "teacher_site_rate_geometry_unsolved",
            "site speeds/rate bounds are invalid",
        )
    rate = required / reference
    if not (
        lower - TEACHER_RATE_BOUNDARY_ABS_TOL
        <= rate
        <= upper + TEACHER_RATE_BOUNDARY_ABS_TOL
    ):
        raise ExactFaceContactGeometryError(
            "teacher_rate_out_of_bounds",
            f"exact site teacher rate {rate} is outside [{lower}, {upper}]",
        )
    return rate


def _finite(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
    ):
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_input",
            f"{name} must be one plain finite number",
        )
    return float(value)


def _vec3(value: Sequence[float], *, name: str) -> Vec3:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_input",
            f"{name} must be one length-3 sequence",
        )
    return tuple(
        _finite(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _quat(value: Sequence[float], *, name: str) -> Quat:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 4
    ):
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_input",
            f"{name} must be one length-4 wxyz sequence",
        )
    raw = tuple(
        _finite(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )
    norm = math.sqrt(sum(component * component for component in raw))
    if norm <= 1.0e-12:
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_input",
            f"{name} must be non-zero",
        )
    normalized = tuple(component / norm for component in raw)
    # q and -q are the same rotation.  A canonical sign makes receipt bytes
    # independent of a motion exporter choosing the other representative.
    for component in normalized:
        if abs(component) > 1.0e-15:
            if component < 0.0:
                normalized = tuple(-value for value in normalized)
            break
    return normalized  # type: ignore[return-value]


def _validate_face_sign(face_sign: int | float) -> int:
    if (
        isinstance(face_sign, bool)
        or type(face_sign) not in (int, float)
        or float(face_sign) not in (-1.0, 1.0)
    ):
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_face_sign",
            f"face_sign must be +1 (red) or -1 (black), got {face_sign!r}",
        )
    return int(face_sign)


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec3, scale: float) -> Vec3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(value: Sequence[float], *, name: str) -> Vec3:
    vector = _vec3(value, name=name)
    norm = _norm(vector)
    if norm <= 1.0e-12:
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_input",
            f"{name} must be non-zero",
        )
    return _scale(vector, 1.0 / norm)


def face_normal_local(face_sign: int | float) -> Vec3:
    sign = _validate_face_sign(face_sign)
    return (0.0, float(sign), 0.0)


def face_center_from_site_local(face_sign: int | float) -> Vec3:
    sign = _validate_face_sign(face_sign)
    x, z = FACE_AREA_CENTER_XZ_FROM_SITE_M
    y = (
        RED_OUTER_Y_FROM_SITE_M
        if sign == RED_FACE_SIGN
        else BLACK_OUTER_Y_FROM_SITE_M
    )
    return (x, y, z)


def ball_center_from_site_local(face_sign: int | float) -> Vec3:
    return _add(
        face_center_from_site_local(face_sign),
        _scale(face_normal_local(face_sign), BALL_RADIUS_M),
    )


def quat_multiply_wxyz(left: Sequence[float], right: Sequence[float]) -> Quat:
    lw, lx, ly, lz = _quat(left, name="left_quaternion")
    rw, rx, ry, rz = _quat(right, name="right_quaternion")
    return _quat(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        name="quaternion_product",
    )


def canonical_quat_wxyz(quaternion: Sequence[float]) -> Quat:
    """Return the unit, canonical-sign representative of one wxyz rotation."""

    return _quat(quaternion, name="quaternion")


def quat_rotate_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    w, x, y, z = _quat(quaternion, name="quaternion")
    v = _vec3(vector, name="vector")
    qv = (x, y, z)
    uv = _cross(qv, v)
    uuv = _cross(qv, uv)
    return _add(v, _add(_scale(uv, 2.0 * w), _scale(uuv, 2.0)))


def quat_to_rotation_matrix_wxyz(
    quaternion: Sequence[float],
) -> Matrix3:
    """Return the proper world-from-local rotation matrix."""

    w, x, y, z = _quat(quaternion, name="quaternion")
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        (
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ),
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _matrix3(value: Sequence[Sequence[float]], *, name: str) -> Matrix3:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_input",
            f"{name} must be one 3x3 sequence",
        )
    rows = tuple(
        _vec3(row, name=f"{name}[{index}]")
        for index, row in enumerate(value)
    )
    # Reject reflections and non-rotations before extracting a quaternion.
    columns = tuple(
        tuple(rows[row][column] for row in range(3))
        for column in range(3)
    )
    for index, column in enumerate(columns):
        if abs(_norm(column) - 1.0) > 1.0e-8:
            raise ExactFaceContactGeometryError(
                "exact_face_contact_invalid_rotation",
                f"{name} column {index} is not unit",
            )
    if any(
        abs(_dot(columns[first], columns[second])) > 1.0e-8
        for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_rotation",
            f"{name} columns are not orthogonal",
        )
    determinant = _dot(columns[0], _cross(columns[1], columns[2]))
    if abs(determinant - 1.0) > 1.0e-8:
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_rotation",
            f"{name} is not a proper rotation",
        )
    return rows  # type: ignore[return-value]


def rotation_matrix_to_quat_wxyz(
    rotation: Sequence[Sequence[float]],
) -> Quat:
    """Convert one proper 3x3 rotation to a canonical wxyz quaternion."""

    matrix = _matrix3(rotation, name="rotation")
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = 2.0 * math.sqrt(max(0.0, trace + 1.0))
        raw = (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    elif m00 >= m11 and m00 >= m22:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + m00 - m11 - m22))
        raw = (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    elif m11 >= m22:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + m11 - m00 - m22))
        raw = (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    else:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + m22 - m00 - m11))
        raw = (
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        )
    if scale <= POLAR_ROTATION_SINGULAR_TOLERANCE:
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_rotation",
            "rotation-to-quaternion branch is singular",
        )
    return _quat(raw, name="rotation_quaternion")


def polar_interpolate_quat_wxyz(
    start_quat_wxyz: Sequence[float],
    end_quat_wxyz: Sequence[float],
    alpha: float,
) -> Quat:
    """Match ``polar((1-a) R0 + a R1)`` without an SVD.

    This is the shared formal/runtime definition.  The antipodal 180-degree
    midpoint is singular for the matrix polar decomposition and therefore
    fails closed instead of inventing the reflection-fix axis chosen by one
    particular SVD implementation.
    """

    start = _quat(start_quat_wxyz, name="start_quat_wxyz")
    end = _quat(end_quat_wxyz, name="end_quat_wxyz")
    fraction = _finite(alpha, name="alpha")
    if fraction < 0.0 or fraction > 1.0:
        raise ExactFaceContactGeometryError(
            "exact_face_contact_invalid_input",
            "alpha must be in [0, 1]",
        )
    if sum(a * b for a, b in zip(start, end)) < 0.0:
        end = tuple(-value for value in end)  # type: ignore[assignment]
    sw, sx, sy, sz = start
    ew, ex, ey, ez = end
    relative = _quat(
        (
            sw * ew + sx * ex + sy * ey + sz * ez,
            sw * ex - sx * ew - sy * ez + sz * ey,
            sw * ey + sx * ez - sy * ew - sz * ex,
            sw * ez - sx * ey + sy * ex - sz * ew,
        ),
        name="relative_quaternion",
    )
    rw, rx, ry, rz = relative
    vector_norm = math.sqrt(rx * rx + ry * ry + rz * rz)
    theta = 2.0 * math.atan2(vector_norm, max(0.0, rw))
    if vector_norm <= POLAR_ROTATION_SINGULAR_TOLERANCE:
        return start
    x = (1.0 - fraction) + fraction * math.cos(theta)
    y = fraction * math.sin(theta)
    if math.hypot(x, y) <= POLAR_ROTATION_SINGULAR_TOLERANCE:
        raise ExactFaceContactGeometryError(
            "exact_face_contact_polar_interpolation_singular",
            "rotation polar interpolation is singular",
        )
    phi = math.atan2(y, x)
    axis = (rx / vector_norm, ry / vector_norm, rz / vector_norm)
    half = 0.5 * phi
    delta = (
        math.cos(half),
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
    )
    return quat_multiply_wxyz(start, delta)


def polar_interpolate_rotation_matrix(
    start_rotation: Sequence[Sequence[float]],
    end_rotation: Sequence[Sequence[float]],
    alpha: float,
) -> Matrix3:
    """Shared proper rotation for formal MuJoCo and Torch mirrors."""

    return quat_to_rotation_matrix_wxyz(
        polar_interpolate_quat_wxyz(
            rotation_matrix_to_quat_wxyz(start_rotation),
            rotation_matrix_to_quat_wxyz(end_rotation),
            alpha,
        )
    )


def minimal_rotation_quat_wxyz(
    source_unit: Sequence[float],
    target_unit: Sequence[float],
) -> Quat:
    """Deterministic shortest world rotation from ``source`` to ``target``."""

    source = _unit(source_unit, name="source_unit")
    target = _unit(target_unit, name="target_unit")
    dot = max(-1.0, min(1.0, _dot(source, target)))
    if dot <= -1.0 + 1.0e-12:
        # ActionBall's signed-face gate requires the solved raw-A normal to
        # remain in the reference hemisphere.  An antipodal solution has no
        # unique shortest rotation/twist continuation and is therefore not a
        # place to invent an axis.
        raise ExactFaceContactGeometryError(
            "exact_face_contact_antipodal_orientation",
            "reference and solved raw-A normals are antipodal",
        )
    return _quat(
        (1.0 + dot, *_cross(source, target)),
        name="minimal_rotation",
    )


def command_orientation_preserve_reference_twist(
    reference_quat_wxyz: Sequence[float],
    solved_raw_a_normal_w: Sequence[float],
) -> tuple[Quat, Quat]:
    """Map reference raw +Y to the solved normal with the shortest rotation.

    Left-multiplying the full reference quaternion by this world rotation
    carries both in-plane axes with it, so the face-centre X/Z offset keeps
    the reference twist.  Reconstructing a frame from the normal alone is
    intentionally impossible through this API.
    """

    reference = _quat(
        reference_quat_wxyz, name="reference_racket_quat_wxyz"
    )
    target = _unit(
        solved_raw_a_normal_w, name="solved_raw_a_normal_w"
    )
    reference_raw_a = quat_rotate_wxyz(reference, (0.0, 1.0, 0.0))
    delta = minimal_rotation_quat_wxyz(reference_raw_a, target)
    command = quat_multiply_wxyz(delta, reference)
    reconstructed = quat_rotate_wxyz(command, (0.0, 1.0, 0.0))
    if _norm(_sub(reconstructed, target)) > 2.0e-12:
        raise ExactFaceContactGeometryError(
            "exact_face_contact_orientation_reconstruction_failed",
            "minimal-rotation command quaternion does not reproduce raw-A",
        )
    return command, delta


def site_target_from_ball_center(
    ball_center_w_m: Sequence[float],
    racket_quat_wxyz: Sequence[float],
    face_sign: int | float,
) -> Vec3:
    ball = _vec3(ball_center_w_m, name="ball_center_w_m")
    offset = quat_rotate_wxyz(
        racket_quat_wxyz, ball_center_from_site_local(face_sign)
    )
    return _sub(ball, offset)


def face_center_velocity_from_site(
    site_velocity_w_mps: Sequence[float],
    angular_velocity_w_radps: Sequence[float],
    racket_quat_wxyz: Sequence[float],
    face_sign: int | float,
) -> Vec3:
    site_velocity = _vec3(
        site_velocity_w_mps, name="site_velocity_w_mps"
    )
    omega = _vec3(
        angular_velocity_w_radps, name="angular_velocity_w_radps"
    )
    face_offset_w = quat_rotate_wxyz(
        racket_quat_wxyz, face_center_from_site_local(face_sign)
    )
    return _add(site_velocity, _cross(omega, face_offset_w))


def site_velocity_from_face_center(
    face_center_velocity_w_mps: Sequence[float],
    angular_velocity_w_radps: Sequence[float],
    racket_quat_wxyz: Sequence[float],
    face_sign: int | float,
) -> Vec3:
    face_velocity = _vec3(
        face_center_velocity_w_mps,
        name="face_center_velocity_w_mps",
    )
    omega = _vec3(
        angular_velocity_w_radps, name="angular_velocity_w_radps"
    )
    face_offset_w = quat_rotate_wxyz(
        racket_quat_wxyz, face_center_from_site_local(face_sign)
    )
    return _sub(face_velocity, _cross(omega, face_offset_w))


@dataclass(frozen=True)
class ExactFaceContactSolution:
    geometry_source_sha256: str
    mount_normal_sign: int
    racket_command_quat_wxyz: Quat
    racket_site_target_w_m: Vec3
    racket_face_center_velocity_w_mps: Vec3
    racket_site_velocity_w_mps: Vec3
    racket_command_angular_velocity_w_radps: Vec3
    teacher_rate: float


def solve_exact_face_contact(
    *,
    ball_contact_w_m: Sequence[float],
    racket_face_center_velocity_w_mps: Sequence[float],
    solved_raw_a_normal_w: Sequence[float],
    mount_normal_sign: int | float,
    reference_racket_quat_wxyz: Sequence[float],
    reference_racket_angular_velocity_w_radps: Sequence[float],
    reference_racket_site_speed_mps: float,
    teacher_rate_min: float,
    teacher_rate_max: float,
) -> ExactFaceContactSolution:
    """Solve the coupled exact geometry and teacher site-rate equation.

    The solver's contact law demands a *face-centre* velocity.  Time-warping
    the teacher also scales angular velocity, so converting that demand to the
    canonical site velocity is coupled to the rate.  Squaring

    ``rate = ||v_face - rate * (omega_native x r_face)|| / speed_ref``

    gives one scalar quadratic.  A unique positive root is required and then
    substituted back into the unsquared equation.  Ambiguous/degenerate roots
    are a named rejection; no fixed-point clipping is hidden here.
    """

    sign = _validate_face_sign(mount_normal_sign)
    face_velocity = _vec3(
        racket_face_center_velocity_w_mps,
        name="racket_face_center_velocity_w_mps",
    )
    reference_omega = _vec3(
        reference_racket_angular_velocity_w_radps,
        name="reference_racket_angular_velocity_w_radps",
    )
    reference_speed = _finite(
        reference_racket_site_speed_mps,
        name="reference_racket_site_speed_mps",
    )
    rate_min = _finite(teacher_rate_min, name="teacher_rate_min")
    rate_max = _finite(teacher_rate_max, name="teacher_rate_max")
    if reference_speed <= 0.0 or rate_min <= 0.0 or rate_max < rate_min:
        raise ExactFaceContactGeometryError(
            "teacher_site_rate_geometry_unsolved",
            "reference site speed/rate bounds are invalid",
        )
    command_quat, delta_quat = command_orientation_preserve_reference_twist(
        reference_racket_quat_wxyz,
        solved_raw_a_normal_w,
    )
    omega_native_command = quat_rotate_wxyz(delta_quat, reference_omega)
    face_offset_w = quat_rotate_wxyz(
        command_quat, face_center_from_site_local(sign)
    )
    angular_face_term = _cross(omega_native_command, face_offset_w)

    a = reference_speed * reference_speed - _dot(
        angular_face_term, angular_face_term
    )
    b = 2.0 * _dot(face_velocity, angular_face_term)
    c = -_dot(face_velocity, face_velocity)
    scale = max(
        reference_speed * reference_speed,
        _dot(angular_face_term, angular_face_term),
        _dot(face_velocity, face_velocity),
        1.0,
    )
    roots: list[float] = []
    if abs(a) <= 1.0e-14 * scale:
        if abs(b) <= 1.0e-14 * scale:
            raise ExactFaceContactGeometryError(
                "teacher_site_rate_geometry_unsolved",
                "teacher site-rate equation is degenerate",
            )
        roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        disc_scale = max(b * b, abs(4.0 * a * c), 1.0)
        if discriminant < -1.0e-14 * disc_scale:
            raise ExactFaceContactGeometryError(
                "teacher_site_rate_geometry_unsolved",
                "teacher site-rate quadratic has no real root",
            )
        discriminant = max(0.0, discriminant)
        sqrt_disc = math.sqrt(discriminant)
        # The numerically stable q-form avoids losing the small root.
        q = -0.5 * (b + math.copysign(sqrt_disc, b))
        if q != 0.0:
            roots.extend((q / a, c / q))
        else:
            roots.append(-b / (2.0 * a))
    positive = sorted(
        {
            float(root)
            for root in roots
            if math.isfinite(root) and root > 0.0
        }
    )
    verified = []
    for rate in positive:
        omega = _scale(omega_native_command, rate)
        site_velocity = _sub(
            face_velocity, _cross(omega, face_offset_w)
        )
        residual = abs(_norm(site_velocity) / reference_speed - rate)
        if residual <= 2.0e-11 * max(1.0, rate):
            verified.append((rate, site_velocity, omega))
    if len(verified) != 1:
        raise ExactFaceContactGeometryError(
            "teacher_site_rate_geometry_unsolved",
            "teacher site-rate equation did not yield one verified positive root",
        )
    rate, site_velocity, command_omega = verified[0]
    admitted_rate = canonical_teacher_rate_from_site_speed(
        _norm(site_velocity),
        reference_speed,
        rate_min,
        rate_max,
    )
    if abs(admitted_rate - rate) > 2.0e-11 * max(1.0, rate):
        raise ExactFaceContactGeometryError(
            "teacher_site_rate_geometry_unsolved",
            "canonical site-speed rate disagrees with verified quadratic root",
        )
    rate = admitted_rate
    site_target = site_target_from_ball_center(
        ball_contact_w_m,
        command_quat,
        sign,
    )
    return ExactFaceContactSolution(
        geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
        mount_normal_sign=sign,
        racket_command_quat_wxyz=command_quat,
        racket_site_target_w_m=site_target,
        racket_face_center_velocity_w_mps=face_velocity,
        racket_site_velocity_w_mps=site_velocity,
        racket_command_angular_velocity_w_radps=command_omega,
        teacher_rate=rate,
    )


def legacy_colocation_error_m(face_sign: int | float) -> float:
    return _norm(ball_center_from_site_local(face_sign))


def torch_exact_contact_state(
    *,
    racket_site_pos_w,
    racket_quat_wxyz,
    racket_site_velocity_w,
    racket_angular_velocity_w,
    face_sign,
):
    """Torch batch mirror of the exact position/velocity point contract."""

    import torch

    if racket_site_pos_w.shape[-1:] != (3,):
        raise ValueError("racket_site_pos_w must end in 3")
    if racket_quat_wxyz.shape[-1:] != (4,):
        raise ValueError("racket_quat_wxyz must end in 4")
    for name, value in (
        ("racket_site_velocity_w", racket_site_velocity_w),
        ("racket_angular_velocity_w", racket_angular_velocity_w),
    ):
        if value.shape != racket_site_pos_w.shape:
            raise ValueError(
                f"{name} must match racket_site_pos_w shape"
            )
    sign = torch.as_tensor(
        face_sign,
        device=racket_site_pos_w.device,
        dtype=racket_site_pos_w.dtype,
    )
    while sign.ndim < racket_site_pos_w.ndim - 1:
        sign = sign.unsqueeze(0)
    if sign.shape != racket_site_pos_w.shape[:-1]:
        sign = torch.broadcast_to(sign, racket_site_pos_w.shape[:-1])
    if not bool(((sign == 1.0) | (sign == -1.0)).all()):
        raise ValueError("face_sign must contain only +/-1")

    q = racket_quat_wxyz
    q_norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    if not bool(
        (
            torch.isfinite(q).all(dim=-1)
            & torch.isfinite(q_norm.squeeze(-1))
            & (q_norm.squeeze(-1) > 1.0e-12)
        ).all()
    ):
        raise ValueError(
            "racket_quat_wxyz must contain only finite non-zero quaternions"
        )
    q = q / q_norm

    def rotate(vector):
        vector = torch.as_tensor(
            vector, device=q.device, dtype=q.dtype
        )
        vector = torch.broadcast_to(vector, q.shape[:-1] + (3,))
        qv = q[..., 1:]
        uv = torch.cross(qv, vector, dim=-1)
        uuv = torch.cross(qv, uv, dim=-1)
        return vector + 2.0 * (
            q[..., :1] * uv + uuv
        )

    x, z = FACE_AREA_CENTER_XZ_FROM_SITE_M
    face_local = torch.stack(
        (
            torch.full_like(sign, float(x)),
            torch.where(
                sign > 0.0,
                torch.full_like(sign, RED_OUTER_Y_FROM_SITE_M),
                torch.full_like(sign, BLACK_OUTER_Y_FROM_SITE_M),
            ),
            torch.full_like(sign, float(z)),
        ),
        dim=-1,
    )
    selected_normal_local = torch.stack(
        (torch.zeros_like(sign), sign, torch.zeros_like(sign)), dim=-1
    )
    face_offset_w = rotate(face_local)
    physical_normal_w = rotate(selected_normal_local)
    face_center_w = racket_site_pos_w + face_offset_w
    ball_center_w = (
        face_center_w + BALL_RADIUS_M * physical_normal_w
    )
    face_velocity_w = racket_site_velocity_w + torch.cross(
        racket_angular_velocity_w, face_offset_w, dim=-1
    )
    return {
        "face_center_w_m": face_center_w,
        "ball_center_w_m": ball_center_w,
        "face_center_velocity_w_mps": face_velocity_w,
        "physical_face_normal_w": physical_normal_w,
        "raw_a_normal_w": rotate((0.0, 1.0, 0.0)),
        "face_offset_from_site_w_m": face_offset_w,
    }


def torch_swept_selected_face_contact(
    *,
    ball_start_w_m,
    ball_end_w_m,
    ball_velocity_start_w_mps,
    ball_velocity_end_w_mps,
    racket_site_start_w_m,
    racket_site_end_w_m,
    racket_quat_start_wxyz,
    racket_quat_end_wxyz,
    racket_site_velocity_start_w_mps,
    racket_site_velocity_end_w_mps,
    racket_angular_velocity_start_w_radps,
    racket_angular_velocity_end_w_radps,
    face_sign,
    previous_valid,
    segment_duration_s,
):
    """Conservative control-step sweep against the pinned selected face.

    This helper deliberately admits only a strict inscribed-circle subset of
    the exact selected-face STL predicate.  It first requires a one-sided
    selected-face clearance bracket, then solves the rotating-plane crossing
    with a fixed 24-step vectorized bisection.  The full racket rotation uses
    the proper polar factor between control samples; that interpolation and
    its iteration count are part of :data:`GEOMETRY_SOURCE_PAYLOAD`.

    Rotation uses the exact analytic polar factor of
    ``(1-alpha) R_start + alpha R_end``.  The formal fitted-MuJoCo gate calls
    the scalar mirror in this module, so both paths share the full
    roll/pitch/yaw interpolation rather than interpolating only a normal.
    Face centre and surface-point velocity are reconstructed from the
    interpolated site pose/velocity, matching the formal rigid-point law.
    The ball follows the endpoint-position/velocity cubic Hermite curve,
    rather than a control-step chord.  The caller constructs those endpoints
    with the SHA-bound 0.5 ms maximum backward integration step above.
    """

    import torch

    vectors = {
        "ball_start_w_m": ball_start_w_m,
        "ball_end_w_m": ball_end_w_m,
        "ball_velocity_start_w_mps": ball_velocity_start_w_mps,
        "ball_velocity_end_w_mps": ball_velocity_end_w_mps,
        "racket_site_start_w_m": racket_site_start_w_m,
        "racket_site_end_w_m": racket_site_end_w_m,
        "racket_site_velocity_start_w_mps": (
            racket_site_velocity_start_w_mps
        ),
        "racket_site_velocity_end_w_mps": (
            racket_site_velocity_end_w_mps
        ),
        "racket_angular_velocity_start_w_radps": (
            racket_angular_velocity_start_w_radps
        ),
        "racket_angular_velocity_end_w_radps": (
            racket_angular_velocity_end_w_radps
        ),
    }
    reference = ball_start_w_m
    if reference.ndim != 2 or reference.shape[-1:] != (3,):
        raise ValueError(
            "swept selected-face vectors must have shape (N, 3)"
        )
    for name, value in vectors.items():
        if value.shape != reference.shape:
            raise ValueError(
                f"{name} must match ball_start_w_m shape"
            )
    for name, value in (
        ("racket_quat_start_wxyz", racket_quat_start_wxyz),
        ("racket_quat_end_wxyz", racket_quat_end_wxyz),
    ):
        if value.shape != (reference.shape[0], 4):
            raise ValueError(f"{name} must have shape (N, 4)")
    valid = torch.as_tensor(
        previous_valid,
        dtype=torch.bool,
        device=reference.device,
    )
    if valid.shape != reference.shape[:-1]:
        raise ValueError(
            "previous_valid must match the swept batch dimensions"
        )
    sign = torch.as_tensor(
        face_sign,
        dtype=reference.dtype,
        device=reference.device,
    )
    if sign.shape != valid.shape:
        raise ValueError("face_sign must have shape (N,)")
    sign_valid = (sign == 1.0) | (sign == -1.0)
    if (
        isinstance(segment_duration_s, bool)
        or type(segment_duration_s) not in (int, float)
        or not math.isfinite(float(segment_duration_s))
        or float(segment_duration_s) <= 0.0
    ):
        raise ValueError("segment_duration_s must be one positive finite scalar")
    segment_duration = float(segment_duration_s)

    finite = valid.clone()
    for value in vectors.values():
        finite &= torch.isfinite(value).all(dim=-1)
    finite &= (
        torch.isfinite(racket_quat_start_wxyz).all(dim=-1)
        & torch.isfinite(racket_quat_end_wxyz).all(dim=-1)
        & torch.isfinite(sign)
        & sign_valid
    )
    q0_norm = torch.linalg.norm(
        racket_quat_start_wxyz, dim=-1, keepdim=True
    )
    q1_norm = torch.linalg.norm(
        racket_quat_end_wxyz, dim=-1, keepdim=True
    )
    finite &= (
        torch.isfinite(q0_norm.squeeze(-1))
        & torch.isfinite(q1_norm.squeeze(-1))
        & (q0_norm.squeeze(-1) > 1.0e-12)
        & (q1_norm.squeeze(-1) > 1.0e-12)
    )
    q0 = racket_quat_start_wxyz / q0_norm.clamp_min(1.0e-12)
    q1 = racket_quat_end_wxyz / q1_norm.clamp_min(1.0e-12)
    same_hemisphere = torch.sum(q0 * q1, dim=-1) >= 0.0
    q1 = torch.where(same_hemisphere.unsqueeze(-1), q1, -q1)

    def _quat_multiply(left, right):
        lw, lx, ly, lz = left.unbind(dim=-1)
        rw, rx, ry, rz = right.unbind(dim=-1)
        return torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )

    def _quat_rotate(quaternion, vector):
        qv = quaternion[:, 1:]
        uv = torch.cross(qv, vector, dim=-1)
        uuv = torch.cross(qv, uv, dim=-1)
        return vector + 2.0 * (
            quaternion[:, :1] * uv + uuv
        )

    def _lerp(first, second, alpha):
        return first + (second - first) * alpha.unsqueeze(-1)

    def _ball_at(alpha):
        """Cubic Hermite path and its time derivative."""

        alpha2 = alpha * alpha
        alpha3 = alpha2 * alpha
        h00 = 2.0 * alpha3 - 3.0 * alpha2 + 1.0
        h10 = alpha3 - 2.0 * alpha2 + alpha
        h01 = -2.0 * alpha3 + 3.0 * alpha2
        h11 = alpha3 - alpha2
        ball = (
            h00.unsqueeze(-1) * ball_start_w_m
            + (h10 * segment_duration).unsqueeze(-1)
            * ball_velocity_start_w_mps
            + h01.unsqueeze(-1) * ball_end_w_m
            + (h11 * segment_duration).unsqueeze(-1)
            * ball_velocity_end_w_mps
        )
        dh00 = 6.0 * alpha2 - 6.0 * alpha
        dh10 = 3.0 * alpha2 - 4.0 * alpha + 1.0
        dh01 = -dh00
        dh11 = 3.0 * alpha2 - 2.0 * alpha
        velocity = (
            (dh00 / segment_duration).unsqueeze(-1) * ball_start_w_m
            + dh10.unsqueeze(-1) * ball_velocity_start_w_mps
            + (dh01 / segment_duration).unsqueeze(-1) * ball_end_w_m
            + dh11.unsqueeze(-1) * ball_velocity_end_w_mps
        )
        return ball, velocity

    q0_conjugate = torch.cat((q0[:, :1], -q0[:, 1:]), dim=-1)
    relative = _quat_multiply(q0_conjugate, q1)
    relative_norm = torch.linalg.norm(
        relative, dim=-1, keepdim=True
    )
    finite &= (
        torch.isfinite(relative).all(dim=-1)
        & torch.isfinite(relative_norm.squeeze(-1))
        & (relative_norm.squeeze(-1) > 1.0e-12)
    )
    relative = relative / relative_norm.clamp_min(1.0e-12)
    relative_w = relative[:, 0].clamp(0.0, 1.0)
    relative_xyz = relative[:, 1:]
    relative_xyz_norm = torch.linalg.norm(
        relative_xyz, dim=-1
    )
    theta = 2.0 * torch.atan2(
        relative_xyz_norm, relative_w
    )
    axis = relative_xyz / relative_xyz_norm.unsqueeze(-1).clamp_min(
        POLAR_ROTATION_SINGULAR_TOLERANCE
    )
    small_rotation = (
        relative_xyz_norm <= POLAR_ROTATION_SINGULAR_TOLERANCE
    )

    x, z = FACE_AREA_CENTER_XZ_FROM_SITE_M
    face_local = torch.stack(
        (
            torch.full_like(sign, float(x)),
            torch.where(
                sign > 0.0,
                torch.full_like(sign, RED_OUTER_Y_FROM_SITE_M),
                torch.full_like(sign, BLACK_OUTER_Y_FROM_SITE_M),
            ),
            torch.full_like(sign, float(z)),
        ),
        dim=-1,
    )
    normal_local = torch.stack(
        (
            torch.zeros_like(sign),
            sign,
            torch.zeros_like(sign),
        ),
        dim=-1,
    )

    def _polar_quat_at(alpha):
        polar_x = (
            (1.0 - alpha) + alpha * torch.cos(theta)
        )
        polar_y = alpha * torch.sin(theta)
        polar_norm = torch.hypot(polar_x, polar_y)
        interpolation_valid = (
            torch.isfinite(polar_norm)
            & (
                polar_norm
                > POLAR_ROTATION_SINGULAR_TOLERANCE
            )
        )
        phi = torch.atan2(polar_y, polar_x)
        half = 0.5 * phi
        delta = torch.cat(
            (
                torch.cos(half).unsqueeze(-1),
                axis * torch.sin(half).unsqueeze(-1),
            ),
            dim=-1,
        )
        identity = torch.zeros_like(delta)
        identity[:, 0] = 1.0
        delta = torch.where(
            small_rotation.unsqueeze(-1), identity, delta
        )
        quaternion = _quat_multiply(q0, delta)
        quaternion_norm = torch.linalg.norm(
            quaternion, dim=-1, keepdim=True
        )
        interpolation_valid &= (
            torch.isfinite(quaternion).all(dim=-1)
            & torch.isfinite(quaternion_norm.squeeze(-1))
            & (quaternion_norm.squeeze(-1) > 1.0e-12)
        )
        quaternion = quaternion / quaternion_norm.clamp_min(1.0e-12)
        return quaternion, interpolation_valid

    def _state_at(alpha):
        ball, _ball_velocity = _ball_at(alpha)
        site = _lerp(
            racket_site_start_w_m,
            racket_site_end_w_m,
            alpha,
        )
        quaternion, rotation_ok = _polar_quat_at(alpha)
        face_offset = _quat_rotate(quaternion, face_local)
        face = site + face_offset
        normal = _quat_rotate(quaternion, normal_local)
        normal_norm = torch.linalg.norm(
            normal, dim=-1, keepdim=True
        )
        normal_ok = (
            rotation_ok
            & torch.isfinite(face).all(dim=-1)
            & torch.isfinite(normal).all(dim=-1)
            & torch.isfinite(normal_norm.squeeze(-1))
            & (normal_norm.squeeze(-1) > 1.0e-12)
        )
        normal = normal / normal_norm.clamp_min(1.0e-12)
        gap = torch.sum((ball - face) * normal, dim=-1) - BALL_RADIUS_M
        return (
            ball,
            site,
            quaternion,
            face,
            face_offset,
            normal,
            gap,
            normal_ok,
        )

    zero = torch.zeros(
        reference.shape[:-1],
        device=reference.device,
        dtype=reference.dtype,
    )
    one = torch.ones_like(zero)
    (
        _ball0,
        _site0,
        _quat0,
        _face0,
        _face_offset0,
        _normal0,
        gap0,
        normal0_ok,
    ) = _state_at(zero)
    (
        _ball1,
        _site1,
        _quat1,
        _face1,
        _face_offset1,
        _normal1,
        gap1,
        normal1_ok,
    ) = _state_at(one)
    finite &= (
        normal0_ok
        & normal1_ok
        & torch.isfinite(gap0)
        & torch.isfinite(gap1)
    )
    tol = SELECTED_FACE_SWEEP_CLEARANCE_TOLERANCE_M
    bracketed = (
        finite
        & (gap0 >= -tol)
        & (gap1 <= tol)
        & (gap0 * gap1 <= 0.0)
    )

    lo = zero
    hi = one
    bisection_finite = finite.clone()
    for _ in range(SELECTED_FACE_SWEEP_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        (
            _ball_mid,
            _site_mid,
            _quat_mid,
            _face_mid,
            _face_offset_mid,
            _normal_mid,
            gap_mid,
            normal_mid_ok,
        ) = _state_at(mid)
        bisection_finite &= normal_mid_ok & torch.isfinite(gap_mid)
        positive = gap_mid > 0.0
        lo = torch.where(positive, mid, lo)
        hi = torch.where(positive, hi, mid)
    alpha = 0.5 * (lo + hi)
    (
        ball,
        site,
        quaternion,
        face,
        face_offset,
        normal,
        gap,
        normal_ok,
    ) = _state_at(alpha)
    finite &= (
        bisection_finite
        & normal_ok
        & torch.isfinite(alpha)
        & torch.isfinite(gap)
    )

    _ball_again, ball_velocity = _ball_at(alpha)
    site_velocity = _lerp(
        racket_site_velocity_start_w_mps,
        racket_site_velocity_end_w_mps,
        alpha,
    )
    angular_velocity = _lerp(
        racket_angular_velocity_start_w_radps,
        racket_angular_velocity_end_w_radps,
        alpha,
    )
    contact_point = ball - BALL_RADIUS_M * normal
    tangent_offset = contact_point - face
    tangential_distance = torch.linalg.norm(
        tangent_offset, dim=-1
    )
    face_center_velocity = site_velocity + torch.cross(
        angular_velocity, face_offset, dim=-1
    )
    contact_point_velocity = site_velocity + torch.cross(
        angular_velocity, contact_point - site, dim=-1
    )
    relative_normal_speed = -torch.sum(
        (ball_velocity - contact_point_velocity) * normal,
        dim=-1,
    )
    finite &= (
        torch.isfinite(site).all(dim=-1)
        & torch.isfinite(quaternion).all(dim=-1)
        & torch.isfinite(face_offset).all(dim=-1)
        & torch.isfinite(ball_velocity).all(dim=-1)
        & torch.isfinite(site_velocity).all(dim=-1)
        & torch.isfinite(face_center_velocity).all(dim=-1)
        & torch.isfinite(angular_velocity).all(dim=-1)
        & torch.isfinite(tangent_offset).all(dim=-1)
        & torch.isfinite(tangential_distance)
        & torch.isfinite(contact_point_velocity).all(dim=-1)
        & torch.isfinite(relative_normal_speed)
    )
    edge_safe = (
        finite
        & (
            tangential_distance
            < SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M
        )
    )
    contact = bracketed & edge_safe
    edge_clearance_lower_bound = (
        SELECTED_FACE_CENTER_TO_BOUNDARY_MIN_M
        - tangential_distance
    )
    return {
        "contact": contact,
        "finite": finite,
        "bracketed": bracketed,
        "edge_safe": edge_safe,
        "alpha": alpha,
        "ball_center_w_m": ball,
        "ball_velocity_w_mps": ball_velocity,
        "racket_site_w_m": site,
        "racket_quat_wxyz": quaternion,
        "racket_site_velocity_w_mps": site_velocity,
        "face_center_w_m": face,
        "physical_face_normal_w": normal,
        "face_center_velocity_w_mps": face_center_velocity,
        "face_angular_velocity_w_radps": angular_velocity,
        "contact_point_velocity_w_mps": contact_point_velocity,
        "tangent_offset_w_m": tangent_offset,
        "tangential_distance_m": tangential_distance,
        "edge_clearance_lower_bound_m": edge_clearance_lower_bound,
        "relative_normal_speed_mps": relative_normal_speed,
    }


__all__ = [
    "EXACT_FACE_CONTACT_SCHEMA_VERSION",
    "EXACT_FACE_CONTACT_KIND",
    "GEOMETRY_SOURCE_PAYLOAD",
    "GEOMETRY_SOURCE_BYTES",
    "GEOMETRY_SOURCE_SHA256",
    "RACKET_SITE_OFFSET_WRIST_M",
    "LEGACY_ISAAC_SITE_OFFSET_WRIST_M",
    "FACE_AREA_CENTER_XZ_FROM_SITE_M",
    "RED_OUTER_Y_FROM_SITE_M",
    "BLACK_OUTER_Y_FROM_SITE_M",
    "BALL_RADIUS_M",
    "RED_SELECTED_FACE_MESH_SHA256",
    "BLACK_SELECTED_FACE_MESH_SHA256",
    "SELECTED_FACE_CENTER_TO_BOUNDARY_MIN_M",
    "FORMAL_FACE_EDGE_GUARD_M",
    "SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M",
    "SELECTED_FACE_SWEEP_CLEARANCE_TOLERANCE_M",
    "SELECTED_FACE_SWEEP_BISECTION_STEPS",
    "SELECTED_FACE_SWEEP_BALL_BACKPROP_MAX_DT_S",
    "POLAR_ROTATION_SINGULAR_TOLERANCE",
    "OFFICIAL_RED_BALL_CENTER_FROM_SITE_M",
    "RED_FACE_SIGN",
    "BLACK_FACE_SIGN",
    "ExactFaceContactGeometryError",
    "ExactFaceContactSolution",
    "face_normal_local",
    "face_center_from_site_local",
    "ball_center_from_site_local",
    "canonical_quat_wxyz",
    "quat_multiply_wxyz",
    "quat_rotate_wxyz",
    "quat_to_rotation_matrix_wxyz",
    "rotation_matrix_to_quat_wxyz",
    "polar_interpolate_quat_wxyz",
    "polar_interpolate_rotation_matrix",
    "minimal_rotation_quat_wxyz",
    "command_orientation_preserve_reference_twist",
    "site_target_from_ball_center",
    "face_center_velocity_from_site",
    "site_velocity_from_face_center",
    "solve_exact_face_contact",
    "legacy_colocation_error_m",
    "torch_exact_contact_state",
    "torch_swept_selected_face_contact",
]
