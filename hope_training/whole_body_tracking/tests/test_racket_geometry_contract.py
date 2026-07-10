"""Pure-CPU audit gates for the official A3 racket control-point contract."""

from __future__ import annotations

import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "hope_training" / "whole_body_tracking" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_motion_target_alignment as check  # noqa: E402
import racket_geometry_contract as geom  # noqa: E402


URDF = REPO / "agi" / "URDF" / "A3T2.5-URDF-std-pingpang" / "urdf" / "URDF-JOINT-LINK.urdf"
MESHES = URDF.parent.parent / "meshes"
MJCF = (
    REPO
    / "agi"
    / "A3_MuJoCo_Sim"
    / "aimrt_mujoco_sim"
    / "src"
    / "models"
    / "bin"
    / "cfg"
    / "model"
    / "a3_pingpong"
    / "a3_pingpong.xml"
)


def _xyz(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _joint_origin(root: ET.Element, name: str) -> np.ndarray:
    joint = next(j for j in root.findall("joint") if j.attrib.get("name") == name)
    return _xyz(joint.find("origin").attrib["xyz"])


def _binary_stl_triangles(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n_tri = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + 50 * n_tri, f"expected binary STL: {path}"
    return np.stack(
        [
            np.frombuffer(raw, dtype="<f4", count=9, offset=84 + 50 * i + 12).reshape(3, 3)
            for i in range(n_tri)
        ]
    ).astype(np.float64)


def _planar_surface_centroid(triangles: np.ndarray, y: float) -> np.ndarray:
    on_plane = np.all(np.isclose(triangles[:, :, 1], y, atol=2.0e-8), axis=1)
    tri = triangles[on_plane]
    assert len(tri) > 0
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return np.average(tri.mean(axis=1), axis=0, weights=area)


def _fit_sphere(vertices: np.ndarray) -> tuple[np.ndarray, float, float]:
    vertices = np.unique(vertices.reshape(-1, 3), axis=0)
    design = np.column_stack([2.0 * vertices, np.ones(len(vertices))])
    rhs = np.sum(vertices * vertices, axis=1)
    solution = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = solution[:3]
    radius = float(np.sqrt(solution[3] + np.dot(center, center)))
    residual = float(np.max(np.abs(np.linalg.norm(vertices - center, axis=1) - radius)))
    return center, radius, residual


def test_official_urdf_and_mjcf_share_the_red_link_site():
    urdf = ET.parse(URDF).getroot()
    red = _joint_origin(urdf, "pingpang_red_joint")
    black = _joint_origin(urdf, "pingpang_black_joint")
    orange_ball_link = _joint_origin(urdf, "pingbang_ball_joint")

    np.testing.assert_allclose(red, geom.RACKET_SITE_OFFSET_WRIST_M, atol=1.0e-12)
    np.testing.assert_allclose(black, red, atol=1.0e-12)
    np.testing.assert_allclose(orange_ball_link, geom.LEGACY_ISAAC_SITE_OFFSET_WRIST_M, atol=1.0e-12)
    # Current Isaac fallback differs from the red/MJCF/C++ point by only 1.49 um.
    assert np.linalg.norm(orange_ball_link - red) == pytest.approx(1.491e-6, rel=2.0e-3)

    mjcf = ET.parse(MJCF).getroot()
    site = next(e for e in mjcf.iter("site") if e.attrib.get("name") == "right_racket")
    np.testing.assert_allclose(_xyz(site.attrib["pos"]), red, atol=1.0e-12)


def test_stls_locate_red_black_face_centers_and_official_ball_tangency():
    red_tri = _binary_stl_triangles(MESHES / "pingpang_red_Link.STL")
    black_tri = _binary_stl_triangles(MESHES / "pingpang_black_Link.STL")
    ball_tri = _binary_stl_triangles(MESHES / "pingbang_ball_Link.STL")

    red_outer_y = float(red_tri[:, :, 1].max())
    black_outer_y = float(black_tri[:, :, 1].min())
    red_center = _planar_surface_centroid(red_tri, red_outer_y)
    black_center = _planar_surface_centroid(black_tri, black_outer_y)

    np.testing.assert_allclose(red_center, geom.face_center_from_site_local(+1), atol=2.0e-9)
    np.testing.assert_allclose(black_center, geom.face_center_from_site_local(-1), atol=2.0e-9)
    assert np.linalg.norm(red_center) == pytest.approx(0.001264, abs=1.0e-6)
    assert np.linalg.norm(black_center) == pytest.approx(0.013268, abs=1.0e-6)

    ball_center, radius, residual = _fit_sphere(ball_tri)
    np.testing.assert_allclose(ball_center, geom.OFFICIAL_RED_BALL_CENTER_FROM_SITE_M, atol=2.0e-9)
    assert radius == pytest.approx(geom.BALL_RADIUS_M, abs=2.0e-9)
    assert residual < 2.0e-9
    red_tangent = ball_center - radius * geom.face_normal_local(+1)
    # The orange marker's tangency is essentially the red rubber area center.
    assert np.linalg.norm(red_tangent - red_center) < 0.000067


def test_exact_ball_center_to_site_formula_quantifies_legacy_error():
    identity = np.eye(3)
    ball = np.array([1.0, 2.0, 3.0])
    for sign in (+1, -1):
        site = geom.site_target_from_ball_center(ball, identity, sign)
        reconstructed = site + geom.ball_center_from_site_local(sign)
        np.testing.assert_allclose(reconstructed, ball, atol=1.0e-12)

    # p_site == p_ball hides one radius on red; on black it also hides the
    # 13.208 mm distance from the canonical red site through the paddle.
    assert geom.legacy_colocation_error_m(+1) == pytest.approx(0.020040, abs=1.0e-6)
    assert geom.legacy_colocation_error_m(-1) == pytest.approx(0.033232, abs=1.0e-6)


def test_rigid_point_velocity_and_black_face_velocity_use_the_same_point():
    v_origin = np.array([1.0, 2.0, 3.0])
    omega = np.array([0.0, 0.0, 10.0])
    r = np.array([0.2, 0.0, 0.0])
    np.testing.assert_allclose(
        geom.rigid_point_velocity(v_origin, omega, r), np.array([1.0, 4.0, 3.0]), atol=1.0e-12
    )

    black_r = geom.face_center_from_site_local(-1)
    expected = v_origin + np.cross(omega, black_r)
    np.testing.assert_allclose(
        geom.face_center_velocity_from_site(v_origin, omega, np.eye(3), -1), expected, atol=1.0e-12
    )


def test_alignment_gate_reconstructs_site_velocity_not_bare_wrist_velocity():
    T, B = 5, 32
    pos = np.zeros((T, B, 3))
    quat = np.zeros((T, B, 4))
    quat[..., 0] = 1.0
    lin = np.zeros((T, B, 3))
    ang = np.zeros((T, B, 3))

    # Bare wrist is +Y-dominant, but wrist yaw contributes omega x r and makes
    # the actual racket site +X-dominant.  The old gate inspected only `lin`.
    lin[:, check.RACKET_BODY] = [0.0, 2.2, 0.0]
    ang[:, check.RACKET_BODY] = [0.0, 0.0, -10.0]
    point_pos, point_vel = check.racket_control_point_series(pos, quat, lin, ang)
    assert abs(lin[2, check.RACKET_BODY, 0]) < abs(lin[2, check.RACKET_BODY, 1])
    assert point_vel[2, 0] > 0.0
    assert abs(point_vel[2, 0]) > abs(point_vel[2, 1])
    np.testing.assert_allclose(
        point_pos[2], geom.RACKET_SITE_OFFSET_WRIST_M, atol=1.0e-12
    )


def test_clean_velocity_is_differenced_from_the_same_site_path():
    fps = 50.0
    t = np.arange(9) / fps
    path = np.column_stack([2.0 * t, -0.5 * t, 0.25 * t])
    np.testing.assert_allclose(
        check.clean_point_velocity(path, frame=4, fps=fps, window=2), [2.0, -0.5, 0.25], atol=1.0e-12
    )
