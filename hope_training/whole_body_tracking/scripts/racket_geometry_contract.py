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

import importlib.util
from pathlib import Path
import sys

import numpy as np


# The production module is the one canonical numeric/semantic source.  Keep
# this audit script as a NumPy facade over those bytes so mesh/URDF checks and
# the live ActionBall geometry cannot drift into two plausible constants.
_PRODUCTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "racket_contact_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_hope_racket_contact_geometry", _PRODUCTION_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load production racket geometry from {_PRODUCTION_PATH}"
    )
_production = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _production
_SPEC.loader.exec_module(_production)

RACKET_SITE_OFFSET_WRIST_M = np.asarray(
    _production.RACKET_SITE_OFFSET_WRIST_M, dtype=np.float64
)
RACKET_BUTT_TO_BLADE_AXIS_LOCAL = np.asarray(
    _production.RACKET_BUTT_TO_BLADE_AXIS_LOCAL, dtype=np.float64
)
RACKET_RIGID_VISUAL_MESH_SHA256 = _production.RACKET_RIGID_VISUAL_MESH_SHA256
LEGACY_ISAAC_SITE_OFFSET_WRIST_M = np.asarray(
    _production.LEGACY_ISAAC_SITE_OFFSET_WRIST_M, dtype=np.float64
)
FACE_AREA_CENTER_XZ_FROM_SITE_M = np.asarray(
    _production.FACE_AREA_CENTER_XZ_FROM_SITE_M, dtype=np.float64
)
RED_OUTER_Y_FROM_SITE_M = _production.RED_OUTER_Y_FROM_SITE_M
BLACK_OUTER_Y_FROM_SITE_M = _production.BLACK_OUTER_Y_FROM_SITE_M
BALL_RADIUS_M = _production.BALL_RADIUS_M
OFFICIAL_RED_BALL_CENTER_FROM_SITE_M = np.asarray(
    _production.OFFICIAL_RED_BALL_CENTER_FROM_SITE_M, dtype=np.float64
)
RED_FACE_SIGN = _production.RED_FACE_SIGN
BLACK_FACE_SIGN = _production.BLACK_FACE_SIGN
GEOMETRY_SOURCE_SHA256 = _production.GEOMETRY_SOURCE_SHA256


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


def polar_interpolate_rotation_matrix(
    start_rotation_w_from_local: np.ndarray,
    end_rotation_w_from_local: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Use the exact production polar interpolation in the formal gate.

    The production helper is stdlib-only and validates that both inputs are
    proper rotations.  Returning a NumPy array here keeps the fitted-MuJoCo
    adjudicator from owning a second SVD/reflection convention.
    """

    rotation = _production.polar_interpolate_rotation_matrix(
        np.asarray(start_rotation_w_from_local, dtype=np.float64).tolist(),
        np.asarray(end_rotation_w_from_local, dtype=np.float64).tolist(),
        float(alpha),
    )
    return np.asarray(rotation, dtype=np.float64)


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
