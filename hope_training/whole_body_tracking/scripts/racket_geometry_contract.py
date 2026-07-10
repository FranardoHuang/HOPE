"""Dependency-light A3 ping-pong racket control-point geometry.

This module makes three different points explicit instead of calling all of
them "the racket center":

``site``
    The official URDF ``pingpang_red_Link`` origin and the official MJCF
    ``right_racket`` site.  Existing policies, C++ FK and MuJoCo evaluation use
    this as their canonical control point.

``face center``
    The area centroid of the selected rubber's outer surface.  It is almost,
    but not exactly, the site: the red center is 1.264 mm away in the blade
    plane; the black center is also 13.208 mm behind it through the paddle.

``ball center``
    At geometric contact this is one ball radius outward from the selected
    face center.  Co-locating it with the site is only the historical virtual
    point-contact approximation, not exact rigid geometry.

The constants below are measured from the tracked official Agibot assets:

* ``agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf``
* ``.../meshes/pingpang_{red,black}_Link.STL``
* ``.../meshes/pingbang_ball_Link.STL``
* ``agi/A3_MuJoCo_Sim/.../a3_pingpong.xml``

This is intentionally NumPy-only so geometry/velocity gates run before Isaac.
It does *not* silently change the current policy contract.  Promoting exact
face geometry requires a versioned planner/bank/ONNX/C++ migration because old
banks and checkpoints were trained with site/ball-center co-location.
"""

from __future__ import annotations

import numpy as np


# right_wrist_yaw_Link -> pingpang_red_Link origin (metres, wrist local).
# This exact six-decimal transform is also the official MJCF right_racket site
# and the C++ FK contract.
RACKET_SITE_OFFSET_WRIST_M = np.array(
    [0.21021, 0.032078, 0.032036], dtype=np.float64
)

# Historical Isaac wrist-offset fallback / motion-analysis constant.  It was
# copied from pingbang_ball_joint rather than pingpang_red_joint.  The two
# points differ by only 1.49 micrometres, so this is not a practical error, but
# keeping both named prevents a false claim of bit-exact parity.
LEGACY_ISAAC_SITE_OFFSET_WRIST_M = np.array(
    [0.210211399202899, 0.0320784994676765, 0.0320358706296689], dtype=np.float64
)

# Outer rubber surface area centroid relative to the common red/black link
# origin, derived from the STL triangles.  Both faces have the same XZ outline.
FACE_AREA_CENTER_XZ_FROM_SITE_M = np.array(
    [0.000893694377, 0.000893694377], dtype=np.float64
)
RED_OUTER_Y_FROM_SITE_M = 0.0
BLACK_OUTER_Y_FROM_SITE_M = -0.013207999989

# The orange-ball STL is an exact 20 mm sphere.  Its center is
# (0.000940486, 0.020000000, 0.000940486) in the common link frame, so its red
# tangency point validates the red face centroid within 0.066 mm.
BALL_RADIUS_M = 0.020000000148
OFFICIAL_RED_BALL_CENTER_FROM_SITE_M = np.array(
    [0.000940485576, 0.020000000164, 0.000940485600], dtype=np.float64
)

RED_FACE_SIGN = 1
BLACK_FACE_SIGN = -1


def _validate_face_sign(face_sign: int | float) -> int:
    sign = int(face_sign)
    if float(face_sign) not in (-1.0, 1.0):
        raise ValueError(f"face_sign must be +1 (red/+Y) or -1 (black/-Y), got {face_sign!r}")
    return sign


def face_normal_local(face_sign: int | float) -> np.ndarray:
    """Selected outward face normal in the wrist/racket local frame."""

    sign = _validate_face_sign(face_sign)
    return np.array([0.0, float(sign), 0.0], dtype=np.float64)


def face_center_from_site_local(face_sign: int | float) -> np.ndarray:
    """Vector from the legacy red-link site to the selected face center."""

    sign = _validate_face_sign(face_sign)
    y = RED_OUTER_Y_FROM_SITE_M if sign == RED_FACE_SIGN else BLACK_OUTER_Y_FROM_SITE_M
    x, z = FACE_AREA_CENTER_XZ_FROM_SITE_M
    return np.array([x, y, z], dtype=np.float64)


def ball_center_from_site_local(face_sign: int | float) -> np.ndarray:
    """Vector site -> ball center at exact, centered geometric contact."""

    return face_center_from_site_local(face_sign) + BALL_RADIUS_M * face_normal_local(face_sign)


def site_target_from_ball_center(
    ball_center_w: np.ndarray,
    racket_rot_w: np.ndarray,
    face_sign: int | float,
) -> np.ndarray:
    """Map a ball-center intercept to the existing URDF/MJCF site target.

    ``racket_rot_w`` is world<-racket.  The exact position relation is

    ``p_ball = p_site + R @ (r_face + radius * n_face_local)``.
    """

    p_ball = np.asarray(ball_center_w, dtype=np.float64)
    rot = np.asarray(racket_rot_w, dtype=np.float64)
    if p_ball.shape[-1:] != (3,) or rot.shape[-2:] != (3, 3):
        raise ValueError(f"expected (...,3) ball and (...,3,3) rotation, got {p_ball.shape}, {rot.shape}")
    offset_w = np.einsum("...ij,j->...i", rot, ball_center_from_site_local(face_sign))
    return p_ball - offset_w


def rigid_point_velocity(
    origin_lin_vel_w: np.ndarray,
    angular_vel_w: np.ndarray,
    point_from_origin_w: np.ndarray,
) -> np.ndarray:
    """World velocity of a rigid point: ``v_p = v_o + omega x r_op``."""

    v = np.asarray(origin_lin_vel_w, dtype=np.float64)
    omega = np.asarray(angular_vel_w, dtype=np.float64)
    r = np.asarray(point_from_origin_w, dtype=np.float64)
    if v.shape[-1:] != (3,) or omega.shape[-1:] != (3,) or r.shape[-1:] != (3,):
        raise ValueError(f"expected (...,3) vectors, got {v.shape}, {omega.shape}, {r.shape}")
    return v + np.cross(omega, r)


def face_center_velocity_from_site(
    site_lin_vel_w: np.ndarray,
    angular_vel_w: np.ndarray,
    racket_rot_w: np.ndarray,
    face_sign: int | float,
) -> np.ndarray:
    """Velocity of the selected physical face center from the legacy site state."""

    rot = np.asarray(racket_rot_w, dtype=np.float64)
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"expected (...,3,3) rotation, got {rot.shape}")
    r_w = np.einsum("...ij,j->...i", rot, face_center_from_site_local(face_sign))
    return rigid_point_velocity(site_lin_vel_w, angular_vel_w, r_w)


def legacy_colocation_error_m(face_sign: int | float) -> float:
    """Distance hidden by the historical ``p_site == p_ball`` approximation."""

    return float(np.linalg.norm(ball_center_from_site_local(face_sign)))
