from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MDP = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


def _load_geometry():
    path = MDP / "racket_contact_geometry.py"
    spec = importlib.util.spec_from_file_location(
        "_test_exact_face_contact_v2", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = _load_geometry()


def _load_virtual_ball():
    path = MDP / "virtual_ball.py"
    spec = importlib.util.spec_from_file_location(
        "_test_action_ball_contact_fit", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _close_vec(left, right, *, tolerance=1.0e-11):
    assert len(left) == len(right)
    assert all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def test_red_and_black_face_centres_are_not_the_control_site():
    assert math.isclose(
        G.legacy_colocation_error_m(G.RED_FACE_SIGN),
        0.020039894840017572,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )
    assert math.isclose(
        G.legacy_colocation_error_m(G.BLACK_FACE_SIGN),
        0.03323204255500939,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )
    assert (
        G.ball_center_from_site_local(G.RED_FACE_SIGN)
        != G.ball_center_from_site_local(G.BLACK_FACE_SIGN)
    )


def test_quaternion_canonicalization_is_bitwise_idempotent():
    source = (
        0.6867758396936938,
        0.3442809801333191,
        -0.23836926079947673,
        0.6530397713504417,
    )
    canonical = G.canonical_quat_wxyz(source)
    assert G.canonical_quat_wxyz(canonical) == canonical
    assert G.canonical_quat_wxyz(
        tuple(-component for component in canonical)
    ) == canonical
    assert math.isclose(
        math.sqrt(sum(component * component for component in canonical)),
        1.0,
        rel_tol=0.0,
        abs_tol=G.QUATERNION_UNIT_PRESERVE_ABS_TOL,
    )


@pytest.mark.parametrize("face_sign", [G.RED_FACE_SIGN, G.BLACK_FACE_SIGN])
def test_rotated_site_target_reconstructs_exact_ball_centre(face_sign):
    half = math.sqrt(0.5)
    quat_z_90 = (half, 0.0, 0.0, half)
    ball = (0.61, -0.13, 0.94)
    site = G.site_target_from_ball_center(ball, quat_z_90, face_sign)
    offset = G.quat_rotate_wxyz(
        quat_z_90, G.ball_center_from_site_local(face_sign)
    )
    _close_vec(
        tuple(site[index] + offset[index] for index in range(3)),
        ball,
    )
    assert site != ball


def test_minimal_world_rotation_preserves_reference_in_plane_twist():
    yaw = G.canonical_quat_wxyz(
        (math.cos(0.31), 0.0, 0.0, math.sin(0.31))
    )
    local_twist = G.canonical_quat_wxyz(
        (math.cos(0.44), 0.0, math.sin(0.44), 0.0)
    )
    reference = G.quat_multiply_wxyz(yaw, local_twist)
    target_raw_a = (0.72, -0.18, 0.670223843)
    target_norm = math.sqrt(sum(value * value for value in target_raw_a))
    target_raw_a = tuple(value / target_norm for value in target_raw_a)

    command, delta = G.command_orientation_preserve_reference_twist(
        reference, target_raw_a
    )
    _close_vec(
        G.quat_rotate_wxyz(command, (0.0, 1.0, 0.0)),
        target_raw_a,
        tolerance=3.0e-12,
    )
    # The same shortest world rotation must carry the reference in-plane X
    # axis.  Building a new frame from only the normal would not prove this.
    _close_vec(
        G.quat_rotate_wxyz(command, (1.0, 0.0, 0.0)),
        G.quat_rotate_wxyz(
            delta,
            G.quat_rotate_wxyz(reference, (1.0, 0.0, 0.0)),
        ),
        tolerance=3.0e-12,
    )


def test_polar_rotation_matches_numpy_svd_for_full_roll_pitch_yaw():
    np = pytest.importorskip("numpy")
    start = G.canonical_quat_wxyz((0.91, 0.17, -0.31, 0.22))
    end = G.canonical_quat_wxyz((0.33, -0.54, 0.12, 0.76))
    start_rotation = np.asarray(
        G.quat_to_rotation_matrix_wxyz(start), dtype=np.float64
    )
    end_rotation = np.asarray(
        G.quat_to_rotation_matrix_wxyz(end), dtype=np.float64
    )
    for alpha in (0.0, 0.13, 0.5, 0.87, 1.0):
        matrix = (
            (1.0 - alpha) * start_rotation
            + alpha * end_rotation
        )
        u, _singular_values, vh = np.linalg.svd(matrix)
        expected = u @ vh
        if np.linalg.det(expected) < 0.0:
            u[:, -1] *= -1.0
            expected = u @ vh
        actual = np.asarray(
            G.polar_interpolate_rotation_matrix(
                start_rotation.tolist(),
                end_rotation.tolist(),
                alpha,
            ),
            dtype=np.float64,
        )
        assert np.allclose(actual, expected, rtol=0.0, atol=3.0e-12)
        assert np.linalg.det(actual) == pytest.approx(
            1.0, rel=0.0, abs=3.0e-12
        )


def test_polar_rotation_rejects_reflection_and_exact_180_midpoint():
    with pytest.raises(
        G.ExactFaceContactGeometryError,
        match="exact_face_contact_invalid_rotation",
    ):
        G.rotation_matrix_to_quat_wxyz(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
        )

    axis_norm = math.sqrt(14.0)
    axis = tuple(value / axis_norm for value in (1.0, 2.0, 3.0))
    exact_pi = (0.0, axis[0], axis[1], axis[2])
    with pytest.raises(
        G.ExactFaceContactGeometryError,
        match="exact_face_contact_polar_interpolation_singular",
    ):
        G.polar_interpolate_quat_wxyz(
            (1.0, 0.0, 0.0, 0.0),
            exact_pi,
            0.5,
        )

    near_pi = math.pi - 1.0e-6
    near_quat = (
        math.cos(0.5 * near_pi),
        *(value * math.sin(0.5 * near_pi) for value in axis),
    )
    rotation = G.polar_interpolate_rotation_matrix(
        G.quat_to_rotation_matrix_wxyz((1.0, 0.0, 0.0, 0.0)),
        G.quat_to_rotation_matrix_wxyz(near_quat),
        0.5,
    )
    c0 = (rotation[0][0], rotation[1][0], rotation[2][0])
    c1 = (rotation[0][1], rotation[1][1], rotation[2][1])
    c2 = (rotation[0][2], rotation[1][2], rotation[2][2])
    determinant = sum(
        c0[index]
        * (
            c1[(index + 1) % 3] * c2[(index + 2) % 3]
            - c1[(index + 2) % 3] * c2[(index + 1) % 3]
        )
        for index in range(3)
    )
    assert determinant == pytest.approx(1.0, rel=0.0, abs=3.0e-10)


def test_formal_gate_delegates_polar_rotation_to_shared_geometry():
    source = (
        ROOT / "scripts" / "mujoco_teacher_motion_fitted_ball_gate.py"
    ).read_text()
    block = source[
        source.index("def interpolate_face_state("):
        source.index("def swept_selected_face_intersection(")
    ]
    assert "racket_geometry.polar_interpolate_rotation_matrix(" in block
    assert "np.linalg.svd" not in block


@pytest.mark.parametrize("face_sign", [G.RED_FACE_SIGN, G.BLACK_FACE_SIGN])
def test_coupled_teacher_rate_includes_omega_cross_face_offset(face_sign):
    solution = G.solve_exact_face_contact(
        ball_contact_w_m=(0.5, 0.2, 0.9),
        racket_face_center_velocity_w_mps=(3.0, 0.2, 0.1),
        solved_raw_a_normal_w=(0.0, 1.0, 0.0),
        mount_normal_sign=face_sign,
        reference_racket_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_racket_angular_velocity_w_radps=(0.0, 0.0, 8.0),
        reference_racket_site_speed_mps=3.0,
        teacher_rate_min=0.1,
        teacher_rate_max=3.0,
    )
    reconstructed_face_velocity = G.face_center_velocity_from_site(
        solution.racket_site_velocity_w_mps,
        solution.racket_command_angular_velocity_w_radps,
        solution.racket_command_quat_wxyz,
        face_sign,
    )
    _close_vec(
        reconstructed_face_velocity,
        solution.racket_face_center_velocity_w_mps,
    )
    assert math.isclose(
        solution.teacher_rate,
        math.sqrt(
            sum(
                value * value
                for value in solution.racket_site_velocity_w_mps
            )
        )
        / 3.0,
        rel_tol=0.0,
        abs_tol=2.0e-12,
    )
    assert (
        solution.racket_site_velocity_w_mps
        != solution.racket_face_center_velocity_w_mps
    )


def test_float32_native_rate_boundary_is_admitted_raw_but_real_overflow_rejects():
    # A float32 prototype can reconstruct a nominal 1.0 m/s vector a few
    # ULPs high.  The v2 contract admits only this explicitly SHA-bound seam
    # while retaining the raw verified rate/equation (no inconsistent clip).
    solution = G.solve_exact_face_contact(
        ball_contact_w_m=(0.5, 0.0, 0.9),
        racket_face_center_velocity_w_mps=(1.0000001, 0.0, 0.0),
        solved_raw_a_normal_w=(0.0, 1.0, 0.0),
        mount_normal_sign=1,
        reference_racket_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_racket_angular_velocity_w_radps=(0.0, 0.0, 0.0),
        reference_racket_site_speed_mps=1.0,
        teacher_rate_min=0.6,
        teacher_rate_max=1.0,
    )
    assert math.isclose(
        solution.teacher_rate,
        1.0000001,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        math.sqrt(
            sum(
                value * value
                for value in solution.racket_site_velocity_w_mps
            )
        ),
        1.0000001,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )

    with pytest.raises(
        G.ExactFaceContactGeometryError,
        match="teacher_rate_out_of_bounds",
    ):
        G.solve_exact_face_contact(
            ball_contact_w_m=(0.5, 0.0, 0.9),
            racket_face_center_velocity_w_mps=(1.00001, 0.0, 0.0),
            solved_raw_a_normal_w=(0.0, 1.0, 0.0),
            mount_normal_sign=1,
            reference_racket_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            reference_racket_angular_velocity_w_radps=(0.0, 0.0, 0.0),
            reference_racket_site_speed_mps=1.0,
            teacher_rate_min=0.6,
            teacher_rate_max=1.0,
        )


def test_torch_batch_mirror_keeps_site_face_and_ball_points_distinct():
    torch = pytest.importorskip("torch")
    site = torch.zeros(2, 3, dtype=torch.float64)
    quat = torch.zeros(2, 4, dtype=torch.float64)
    quat[:, 0] = 1.0
    velocity = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float64
    )
    omega = torch.tensor(
        [[0.0, 0.0, 7.0], [0.0, 0.0, 7.0]], dtype=torch.float64
    )
    state = G.torch_exact_contact_state(
        racket_site_pos_w=site,
        racket_quat_wxyz=quat,
        racket_site_velocity_w=velocity,
        racket_angular_velocity_w=omega,
        face_sign=torch.tensor([1.0, -1.0], dtype=torch.float64),
    )
    assert not torch.equal(state["face_center_w_m"], site)
    assert not torch.equal(state["ball_center_w_m"], site)
    assert not torch.equal(
        state["face_center_velocity_w_mps"], velocity
    )
    _close_vec(
        state["ball_center_w_m"][0].tolist(),
        G.ball_center_from_site_local(1),
    )
    _close_vec(
        state["ball_center_w_m"][1].tolist(),
        G.ball_center_from_site_local(-1),
    )


def test_torch_exact_contact_rejects_zero_quaternion():
    torch = pytest.importorskip("torch")
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite non-zero quaternions"):
        G.torch_exact_contact_state(
            racket_site_pos_w=zeros,
            racket_quat_wxyz=torch.zeros(1, 4, dtype=torch.float64),
            racket_site_velocity_w=zeros,
            racket_angular_velocity_w=zeros,
            face_sign=torch.ones(1, dtype=torch.float64),
        )


def _swept_contact(
    torch,
    *,
    tangent_x=0.0,
    start_gap=0.01,
    end_gap=-0.01,
    previous_valid=True,
    speed=2.0,
    angular_z=0.0,
    face_sign=1.0,
    quat_start=None,
    quat_end=None,
    segment_duration_s=0.01,
):
    dtype = torch.float64
    radius = G.BALL_RADIUS_M
    face_center = G.face_center_from_site_local(face_sign)
    ball_start = torch.tensor(
        [[
            face_center[0] + tangent_x,
            face_center[1] + face_sign * (radius + start_gap),
            face_center[2],
        ]],
        dtype=dtype,
    )
    ball_end = torch.tensor(
        [[
            face_center[0] + tangent_x,
            face_center[1] + face_sign * (radius + end_gap),
            face_center[2],
        ]],
        dtype=dtype,
    )
    ball_velocity = torch.tensor(
        [[0.0, -face_sign * speed, 0.0]], dtype=dtype
    )
    zeros = torch.zeros(1, 3, dtype=dtype)
    angular = torch.tensor(
        [[0.0, 0.0, angular_z]], dtype=dtype
    )
    if quat_start is None:
        quat_start = (1.0, 0.0, 0.0, 0.0)
    if quat_end is None:
        quat_end = quat_start
    q_start = torch.tensor([quat_start], dtype=dtype)
    q_end = torch.tensor([quat_end], dtype=dtype)
    return G.torch_swept_selected_face_contact(
        ball_start_w_m=ball_start,
        ball_end_w_m=ball_end,
        ball_velocity_start_w_mps=ball_velocity,
        ball_velocity_end_w_mps=ball_velocity,
        racket_site_start_w_m=zeros,
        racket_site_end_w_m=zeros,
        racket_quat_start_wxyz=q_start,
        racket_quat_end_wxyz=q_end,
        racket_site_velocity_start_w_mps=zeros,
        racket_site_velocity_end_w_mps=zeros,
        racket_angular_velocity_start_w_radps=angular,
        racket_angular_velocity_end_w_radps=angular,
        face_sign=torch.tensor([face_sign], dtype=dtype),
        previous_valid=torch.tensor([previous_valid]),
        segment_duration_s=segment_duration_s,
    )


def test_swept_selected_face_is_one_sided_and_requires_history():
    torch = pytest.importorskip("torch")
    hit = _swept_contact(torch)
    assert bool(hit["contact"][0])
    assert bool(hit["bracketed"][0])
    assert float(hit["alpha"][0]) == pytest.approx(
        0.5, abs=2.0e-7
    )

    reverse = _swept_contact(
        torch, start_gap=-0.01, end_gap=0.01
    )
    assert not bool(reverse["contact"][0])
    assert not bool(reverse["bracketed"][0])

    no_history = _swept_contact(torch, previous_valid=False)
    assert not bool(no_history["contact"][0])
    assert not bool(no_history["finite"][0])


@pytest.mark.parametrize("face_sign", [1.0, -1.0])
def test_swept_selected_face_signs_have_symmetric_closing_contact(face_sign):
    torch = pytest.importorskip("torch")
    hit = _swept_contact(torch, face_sign=face_sign)
    assert bool(hit["contact"][0])
    assert float(hit["relative_normal_speed_mps"][0]) == (
        pytest.approx(2.0, abs=2.0e-7)
    )


def test_swept_full_rotation_uses_same_polar_pose_as_formal_gate():
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    q0 = G.canonical_quat_wxyz((0.93, 0.12, -0.19, 0.27))
    q1 = G.canonical_quat_wxyz((0.79, -0.22, 0.31, 0.45))
    site0 = (0.10, -0.04, 0.82)
    site1 = (0.11, -0.03, 0.825)
    local_face = G.face_center_from_site_local(1)
    local_normal = G.face_normal_local(1)
    center0 = tuple(
        site0[index] + G.quat_rotate_wxyz(q0, local_face)[index]
        for index in range(3)
    )
    center1 = tuple(
        site1[index] + G.quat_rotate_wxyz(q1, local_face)[index]
        for index in range(3)
    )
    normal0 = G.quat_rotate_wxyz(q0, local_normal)
    normal1 = G.quat_rotate_wxyz(q1, local_normal)
    ball0 = tuple(
        center0[index]
        + (G.BALL_RADIUS_M + 0.008) * normal0[index]
        for index in range(3)
    )
    ball1 = tuple(
        center1[index]
        + (G.BALL_RADIUS_M - 0.008) * normal1[index]
        for index in range(3)
    )
    duration = 0.02
    chord_velocity = tuple(
        (ball1[index] - ball0[index]) / duration
        for index in range(3)
    )
    site_velocity = tuple(
        (site1[index] - site0[index]) / duration
        for index in range(3)
    )
    result = G.torch_swept_selected_face_contact(
        ball_start_w_m=torch.tensor([ball0], dtype=dtype),
        ball_end_w_m=torch.tensor([ball1], dtype=dtype),
        ball_velocity_start_w_mps=torch.tensor(
            [chord_velocity], dtype=dtype
        ),
        ball_velocity_end_w_mps=torch.tensor(
            [chord_velocity], dtype=dtype
        ),
        racket_site_start_w_m=torch.tensor([site0], dtype=dtype),
        racket_site_end_w_m=torch.tensor([site1], dtype=dtype),
        racket_quat_start_wxyz=torch.tensor([q0], dtype=dtype),
        racket_quat_end_wxyz=torch.tensor([q1], dtype=dtype),
        racket_site_velocity_start_w_mps=torch.tensor(
            [site_velocity], dtype=dtype
        ),
        racket_site_velocity_end_w_mps=torch.tensor(
            [site_velocity], dtype=dtype
        ),
        racket_angular_velocity_start_w_radps=torch.tensor(
            [[40.0, -30.0, 70.0]], dtype=dtype
        ),
        racket_angular_velocity_end_w_radps=torch.tensor(
            [[40.0, -30.0, 70.0]], dtype=dtype
        ),
        face_sign=torch.ones(1, dtype=dtype),
        previous_valid=torch.tensor([True]),
        segment_duration_s=duration,
    )
    assert bool(result["contact"][0])
    alpha = float(result["alpha"][0])
    expected_quat = G.polar_interpolate_quat_wxyz(q0, q1, alpha)
    expected_normal = G.quat_rotate_wxyz(
        expected_quat, local_normal
    )
    _close_vec(
        result["physical_face_normal_w"][0].tolist(),
        expected_normal,
        tolerance=3.0e-10,
    )


def test_cubic_ball_path_removes_control_step_gravity_chord_error():
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    half = math.sqrt(0.5)
    quaternion = (half, half, 0.0, 0.0)  # local +Y -> world +Z
    face = G.quat_rotate_wxyz(
        quaternion, G.face_center_from_site_local(1)
    )
    normal = G.quat_rotate_wxyz(
        quaternion, G.face_normal_local(1)
    )
    duration = 0.02
    gap0 = 0.0015
    velocity0 = -0.02
    acceleration = -9.81
    gap1 = (
        gap0
        + velocity0 * duration
        + 0.5 * acceleration * duration * duration
    )
    velocity1 = velocity0 + acceleration * duration
    ball0 = tuple(
        face[index]
        + (G.BALL_RADIUS_M + gap0) * normal[index]
        for index in range(3)
    )
    ball1 = tuple(
        face[index]
        + (G.BALL_RADIUS_M + gap1) * normal[index]
        for index in range(3)
    )
    velocity_start = tuple(velocity0 * value for value in normal)
    velocity_end = tuple(velocity1 * value for value in normal)
    zeros = torch.zeros(1, 3, dtype=dtype)
    result = G.torch_swept_selected_face_contact(
        ball_start_w_m=torch.tensor([ball0], dtype=dtype),
        ball_end_w_m=torch.tensor([ball1], dtype=dtype),
        ball_velocity_start_w_mps=torch.tensor(
            [velocity_start], dtype=dtype
        ),
        ball_velocity_end_w_mps=torch.tensor(
            [velocity_end], dtype=dtype
        ),
        racket_site_start_w_m=zeros,
        racket_site_end_w_m=zeros,
        racket_quat_start_wxyz=torch.tensor(
            [quaternion], dtype=dtype
        ),
        racket_quat_end_wxyz=torch.tensor(
            [quaternion], dtype=dtype
        ),
        racket_site_velocity_start_w_mps=zeros,
        racket_site_velocity_end_w_mps=zeros,
        racket_angular_velocity_start_w_radps=zeros,
        racket_angular_velocity_end_w_radps=zeros,
        face_sign=torch.ones(1, dtype=dtype),
        previous_valid=torch.tensor([True]),
        segment_duration_s=duration,
    )
    assert bool(result["contact"][0])
    discriminant = velocity0 * velocity0 - 2.0 * acceleration * gap0
    contact_time = (
        -velocity0 - math.sqrt(discriminant)
    ) / acceleration
    expected_alpha = contact_time / duration
    chord_alpha = gap0 / (gap0 - gap1)
    assert float(result["alpha"][0]) == pytest.approx(
        expected_alpha, abs=2.0e-7
    )
    assert abs(expected_alpha - chord_alpha) > 0.1


def test_swept_selected_face_uses_derived_mesh_edge_guard_not_095m():
    torch = pytest.importorskip("torch")
    inside = _swept_contact(
        torch,
        tangent_x=(
            G.SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M - 1.0e-5
        ),
    )
    outside = _swept_contact(
        torch,
        tangent_x=(
            G.SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M + 1.0e-5
        ),
    )
    legacy_phantom = _swept_contact(torch, tangent_x=0.09)
    assert bool(inside["contact"][0])
    assert (
        float(inside["edge_clearance_lower_bound_m"][0])
        > G.BALL_RADIUS_M + G.FORMAL_FACE_EDGE_GUARD_M
    )
    assert not bool(outside["contact"][0])
    assert not bool(legacy_phantom["contact"][0])


def test_swept_off_center_contact_uses_rigid_surface_point_velocity():
    torch = pytest.importorskip("torch")
    result = _swept_contact(
        torch,
        tangent_x=0.02,
        speed=2.0,
        angular_z=10.0,
    )
    assert bool(result["contact"][0])
    _close_vec(
        result["contact_point_velocity_w_mps"][0].tolist(),
        (
            0.0,
            10.0 * (
                G.FACE_AREA_CENTER_XZ_FROM_SITE_M[0] + 0.02
            ),
            0.0,
        ),
        tolerance=2.0e-7,
    )
    assert float(result["relative_normal_speed_mps"][0]) == (
        pytest.approx(
            2.0
            + 10.0
            * (G.FACE_AREA_CENTER_XZ_FROM_SITE_M[0] + 0.02),
            abs=2.0e-7,
        )
    )


def test_selected_face_payload_seals_mesh_and_formal_edge_erosion():
    assert G.GEOMETRY_SOURCE_PAYLOAD["selected_face_mesh_sha256"] == {
        "red": (
            "94182ec1c7c64db8c5ec7ce5f9aad44d427f433a6aae5cf23aa655e077633842"
        ),
        "black": (
            "5f0e772ea9ed81e5b70f5dfb4ded49f9d269c54c893249857209f85168361b1b"
        ),
    }
    assert G.SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M == pytest.approx(
        G.SELECTED_FACE_CENTER_TO_BOUNDARY_MIN_M
        - G.BALL_RADIUS_M
        - G.FORMAL_FACE_EDGE_GUARD_M,
        abs=0.0,
    )
    assert G.GEOMETRY_SOURCE_PAYLOAD[
        "safe_ball_center_tangential_radius_m"
    ] == G.SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M


def test_contact_fit_envelope_is_inclusive_and_drives_same_restitution():
    torch = pytest.importorskip("torch")
    vb = _load_virtual_ball()
    prm = vb.load_venue_params(
        str(ROOT.parents[1] / "configs" / "ball_physics_venue.yaml")
    )
    speeds = torch.tensor(
        [1.399, 1.4, 7.2, 7.201], dtype=torch.float64
    )
    v_minus = torch.zeros(4, 3, dtype=torch.float64)
    v_minus[:, 0] = -speeds
    zeros = torch.zeros_like(v_minus)
    normal = torch.zeros_like(v_minus)
    normal[:, 0] = 1.0
    state = vb.paddle_contact_state(
        v_minus, zeros, normal, zeros, prm
    )
    assert state["fit_valid"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert torch.equal(
        state["restitution_effective"],
        state["restitution_raw"],
    )
    v_plus, _w_plus = vb.predict_paddle_contact(
        v_minus, zeros, normal, zeros, prm
    )
    expected_x = (
        state["restitution_effective"].squeeze(-1) * speeds
    )
    assert torch.allclose(
        v_plus[:, 0],
        expected_x,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_action_ball_contact_reasons_are_exclusive_and_conserve_denominator():
    torch = pytest.importorskip("torch")
    vb = _load_virtual_ball()
    exact = torch.ones(6, dtype=torch.bool)
    exact_before = exact.clone()
    signed = torch.tensor(
        [False, True, True, True, True, True]
    )
    geometry = torch.tensor(
        [True, False, True, True, True, True]
    )
    finite = torch.tensor(
        [True, True, False, True, True, True]
    )
    speed = torch.tensor(
        [2.0, 2.0, 2.0, 1.399, 7.201, 2.0]
    )
    capture, reasons = vb.classify_action_ball_contact(
        exact_strike=exact,
        signed_face_ok=signed,
        geometry_contact=geometry,
        contact_finite=finite,
        normal_speed_mps=speed,
    )
    assert torch.equal(exact, exact_before)
    assert capture.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert [int(mask.sum()) for mask in reasons.values()] == [
        1,
        1,
        1,
        1,
        1,
    ]
    partition = capture.clone()
    for mask in reasons.values():
        assert not bool((partition & mask).any())
        partition |= mask
    assert torch.equal(partition, exact)


def test_receding_high_omega_sweep_cannot_become_capture_by_normal_flip():
    torch = pytest.importorskip("torch")
    vb = _load_virtual_ball()
    prm = vb.load_venue_params(
        str(ROOT.parents[1] / "configs" / "ball_physics_venue.yaml")
    )
    swept = _swept_contact(
        torch,
        tangent_x=0.02,
        speed=2.0,
        angular_z=-200.0,
    )
    assert bool(swept["contact"][0])
    assert float(swept["relative_normal_speed_mps"][0]) < 0.0
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    state = vb.paddle_contact_state(
        swept["ball_velocity_w_mps"],
        swept["contact_point_velocity_w_mps"],
        swept["physical_face_normal_w"],
        zeros,
        prm,
    )
    assert float(
        state["selected_face_closing_speed_mps"][0]
    ) < 0.0
    # orient_normal still makes the impulse's absolute u_n positive; the
    # selected-side signed gate must prevent that sign erasure from scoring.
    assert float(state["normal_speed_mps"][0]) > 0.0
    selected_side_contact = (
        swept["contact"]
        & (
            state["selected_face_closing_speed_mps"]
            > 0.0
        )
    )
    capture, reasons = vb.classify_action_ball_contact(
        exact_strike=torch.tensor([True]),
        signed_face_ok=torch.tensor([True]),
        geometry_contact=selected_side_contact,
        contact_finite=state["finite"] & swept["finite"],
        normal_speed_mps=(
            state["selected_face_closing_speed_mps"]
        ),
    )
    assert not bool(capture[0])
    assert bool(
        reasons["virtual_contact_geometry_reject_count"][0]
    )


def test_nonfinite_rejected_row_is_sanitized_before_contact_and_rollout():
    torch = pytest.importorskip("torch")
    vb = _load_virtual_ball()
    nan = float("nan")
    origin = torch.tensor(
        [[nan, nan, nan], [1.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    velocity = torch.tensor(
        [[nan, nan, nan], [-2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    surface_velocity = torch.tensor(
        [[nan, nan, nan], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    normal = torch.tensor(
        [[nan, nan, nan], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    spin = torch.tensor(
        [[nan, nan, nan], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    fallback = torch.tensor(
        [[0.5, 0.0, 0.88], [0.5, 0.0, 0.88]],
        dtype=torch.float64,
    )
    safe = vb.finite_action_ball_rollout_inputs(
        capture=torch.tensor([False, True]),
        contact_origin_w_m=origin,
        ball_velocity_w_mps=velocity,
        contact_point_velocity_w_mps=surface_velocity,
        physical_face_normal_w=normal,
        ball_spin_w_radps=spin,
        fallback_origin_w_m=fallback,
    )
    assert all(
        bool(torch.isfinite(value).all())
        for value in safe.values()
    )
    assert torch.equal(safe["contact_origin_w_m"][0], fallback[0])
    assert torch.equal(safe["contact_origin_w_m"][1], origin[1])


def test_swept_history_rejects_action_reset_or_swing_identity_swap():
    torch = pytest.importorskip("torch")
    vb = _load_virtual_ball()
    valid = torch.ones(4, dtype=torch.bool)
    previous_action = torch.tensor([1, 1, 1, 1])
    current_action = torch.tensor([1, 2, 1, 1])
    previous_reset = torch.tensor([7, 7, 7, 7])
    current_reset = torch.tensor([7, 7, 8, 7])
    previous_swing = torch.tensor([9, 9, 9, 9])
    current_swing = torch.tensor([9, 9, 9, 10])
    identity = vb.action_ball_sweep_identity_valid(
        previous_valid=valid,
        previous_action=previous_action,
        current_action=current_action,
        previous_reset_generation=previous_reset,
        current_reset_generation=current_reset,
        previous_swing_generation=previous_swing,
        current_swing_generation=current_swing,
        attempt_active=valid,
    )
    assert identity.tolist() == [True, False, False, False]


def test_production_wiring_uses_site_for_command_and_face_for_contact():
    command_source = (MDP / "hope_commands.py").read_text()
    physical_source = (MDP / "physical_ball.py").read_text()
    shadow_source = (MDP / "shadow_ball.py").read_text()

    assert (
        "self.racket_target_pos_w[ids] = origins + site_target_local"
        in command_source
    )
    assert "self.racket_target_vel_w[ids] = site_velocity" in command_source
    assert 'contact_origin_w = swept["ball_center_w_m"]' in command_source
    assert (
        'v_r = swept["contact_point_velocity_w_mps"]'
        in command_source
    )
    assert "torch_swept_selected_face_contact(" in command_source
    assert "geometry_contact=selected_side_contact" in command_source
    assert "selected_face_closing_speed_mps" in command_source
    assert "finite_action_ball_rollout_inputs(" in command_source
    vb_method = command_source[
        command_source.index("    def _vb_evaluate("):
        command_source.index("    def _vb_book_strike_step(")
    ]
    legacy_capture = (
        "pos_err < float(self.cfg.vb_capture_radius)"
    )
    assert vb_method.count(legacy_capture) == 1
    assert vb_method.index(legacy_capture) > vb_method.index(
        "contact_rejections = None"
    )
    assert "def selected_face_disc_contact(" in physical_source
    assert (
        "cmd._action_ball_ball_contact_target_w"
        in physical_source
    )
    assert (
        "achieved_contact[\"ball_center_w_m\"]"
        in shadow_source
    )


def test_action_ball_contact_reasons_feed_exact_sparse_denominator():
    command_source = (MDP / "hope_commands.py").read_text()
    virtual_source = (MDP / "virtual_ball.py").read_text()
    assert "capture/rejection ledger does not conserve " in command_source
    assert '"the exact-strike denominator"' in command_source
    assert (
        "contact_rejections=contact_rejections"
        in command_source
    )
    for name in (
        "virtual_contact_face_reject_count",
        "virtual_contact_geometry_reject_count",
        "virtual_contact_nonfinite_reject_count",
        "virtual_contact_u_n_below_fit_reject_count",
        "virtual_contact_u_n_above_fit_reject_count",
    ):
        assert name in command_source
        assert name in virtual_source


def test_solver_and_resume_contracts_pin_geometry_source_and_v5_receipts():
    command_source = (MDP / "hope_commands.py").read_text()
    motion_command_source = (MDP / "commands.py").read_text()
    runtime_source = (MDP / "action_ball_runtime.py").read_text()
    train_source = (ROOT / "scripts" / "train.py").read_text()

    assert '"racket_contact_geometry.py"' in command_source
    assert "contact_geometry_contract=contact_geometry_contract" in command_source
    assert "TASK_RECEIPT_SCHEMA_VERSION = 5" in runtime_source
    assert "_ACTION_BALL_STATE_SCHEMA_VERSION = 5" in command_source
    assert "_ACTION_BALL_SOLVER_STATE_SCHEMA_VERSION = 5" in command_source
    assert '"frozen_evaluation"' in command_source
    assert "action-ball exact-resume schema/kind mismatch" in command_source
    authority = "TASK_RECEIPT_TIMING_AUTHORITY"
    assert authority in runtime_source
    assert authority in command_source
    assert authority in motion_command_source
    assert authority in train_source


def test_geometry_sha_binds_full_cycle_teacher_no_contact_semantics():
    safety = G.GEOMETRY_SOURCE_PAYLOAD["teacher_motion_safety_semantics"]
    assert safety["scope"] == (
        "entire_reference_cycle_all_robot_links_and_full_racket_geometry"
    )
    assert set(safety["forbidden_world_geometry"]) == {
        "table_top",
        "table_edges",
        "table_underside",
        "net_mesh",
        "net_posts",
    }
    assert "swept_collision" in safety["verification"]
    assert "required_per_action" in safety["admission"]
    assert "unsafe_failure" in safety["runtime_outcome"]
