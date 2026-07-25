"""Pure-NumPy tests for the fail-closed canonical face-manifold solver."""

from __future__ import annotations

import math
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_face_manifold as cfm  # noqa: E402


def rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )


class FakeRightRacketBackend:
    """Seven-DOF toy: shoulder xyz controls point; wrist roll controls face."""

    joint_names = ("left_dummy",) + cfm.RIGHT_STRIKE_CHAIN + ("leg_dummy",)
    root_body_name = "pelvis_link"

    def __init__(self, *, roll_limit: float = 3.3, pose_is_constant: bool = False):
        self.position_lower = np.full(len(self.joint_names), -4.0)
        self.position_upper = np.full(len(self.joint_names), 4.0)
        self.velocity_limit = np.full(len(self.joint_names), 100.0)
        self.effort_limit = np.full(len(self.joint_names), 100.0)
        roll = self.joint_names.index("right_wrist_roll_joint")
        self.position_lower[roll] = -roll_limit
        self.position_upper[roll] = roll_limit
        self.pose_is_constant = pose_is_constant

    def site_pose(self, joint_pos, root_pos_w, root_quat_w):
        del root_pos_w, root_quat_w
        q = np.asarray(joint_pos, dtype=float)
        if self.pose_is_constant:
            return np.zeros(3), np.eye(3)
        position = np.asarray(
            [
                q[self.joint_names.index("right_shoulder_pitch_joint")],
                q[self.joint_names.index("right_shoulder_roll_joint")],
                q[self.joint_names.index("right_shoulder_yaw_joint")],
            ]
        )
        roll = q[self.joint_names.index("right_wrist_roll_joint")]
        return position, rotation_x(roll)

    def diagonal_dynamics(self, joint_pos, root_pos_w, root_quat_w):
        del joint_pos, root_pos_w, root_quat_w
        return (
            np.ones(len(self.joint_names), dtype=float),
            np.zeros(len(self.joint_names), dtype=float),
        )


class CoupledRightRacketBackend(FakeRightRacketBackend):
    """Face rotation also moves the point, unlike the old roll-only toy."""

    def site_pose(self, joint_pos, root_pos_w, root_quat_w):
        position, rotation = super().site_pose(
            joint_pos, root_pos_w, root_quat_w
        )
        roll = np.asarray(joint_pos)[
            self.joint_names.index("right_wrist_roll_joint")
        ]
        position = position + np.asarray(
            [0.10 * math.sin(roll), 0.08 * (1.0 - math.cos(roll)), 0.0]
        )
        return position, rotation


def inputs(frames: int = 4):
    backend = FakeRightRacketBackend()
    joints = len(backend.joint_names)
    source = np.zeros((frames, joints))
    source[:, 0] = 0.37
    source[:, -1] = -0.29
    for frame in range(frames):
        source[
            frame, backend.joint_names.index("right_shoulder_pitch_joint")
        ] = 0.01 * frame
        source[
            frame, backend.joint_names.index("right_shoulder_roll_joint")
        ] = -0.005 * frame
    root_pos = np.zeros((frames, 3))
    root_quat = np.zeros((frames, 4))
    root_quat[:, 0] = 1.0
    ready = np.zeros(joints)
    return backend, source, root_pos, root_quat, ready


def config(**overrides):
    values = dict(
        position_tolerance_m=1.0e-8,
        normal_tolerance_rad=1.0e-7,
        orientation_tolerance_rad=1.0e-7,
        velocity_tolerance_mps=1.0e-6,
        max_step_rad=0.2,
        random_restarts=0,
        max_iterations=80,
    )
    values.update(overrides)
    return cfm.FaceManifoldConfig(**values)


def test_exact_face_flip_and_so3_log_helpers():
    source = rotation_x(0.31)
    target_normal, target_rotation = cfm.exact_face_flip_target(source)
    np.testing.assert_allclose(target_normal, -source[:, 1], atol=1.0e-15)
    np.testing.assert_allclose(
        target_rotation, source @ np.diag([1.0, -1.0, -1.0]), atol=1.0e-15
    )
    np.testing.assert_allclose(cfm.rotation_log(np.eye(3)), 0.0, atol=1.0e-15)
    assert np.linalg.norm(cfm.rotation_log(rotation_x(math.pi))) == pytest.approx(
        math.pi
    )


def test_normal_mode_preserves_point_speed_and_nonactive_joints():
    backend, source, root_pos, root_quat, ready = inputs()
    result = cfm.solve_face_flipped_window(
        source,
        root_pos,
        root_quat,
        ready,
        fps=50.0,
        backend=backend,
        config=config(mode="normal"),
        frame_indices=[34, 35, 36, 37],
    )
    np.testing.assert_allclose(
        result.output_site_pos_w, result.source_site_pos_w, atol=1.0e-8
    )
    np.testing.assert_allclose(
        result.output_segment_velocity_w,
        result.source_segment_velocity_w,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        result.output_raw_plus_y_w, result.target_raw_plus_y_w, atol=1.0e-7
    )
    np.testing.assert_array_equal(result.joint_pos[:, 0], source[:, 0])
    np.testing.assert_array_equal(result.joint_pos[:, -1], source[:, -1])
    assert result.max_velocity_residual_mps <= 1.0e-6
    assert result.summary()["max_step_rad"] <= 0.2
    assert (
        result.summary()["solver_strategy"]
        == "anchor_independent_full_window_candidate_graph"
    )
    assert result.summary()["window_edges"]["entry"]["end"] == "solved_frame_34"
    assert result.summary()["window_edges"]["exit"]["start"] == "solved_frame_37"


def test_anchor_is_annotation_only_and_cannot_change_selected_window():
    backend, source, root_pos, root_quat, ready = inputs(3)
    kwargs = dict(
        fps=50.0,
        backend=backend,
        config=config(mode="normal"),
        frame_indices=[40, 41, 42],
    )
    automatic = cfm.solve_face_flipped_window(
        source, root_pos, root_quat, ready, anchor_index=None, **kwargs
    )
    annotated = cfm.solve_face_flipped_window(
        source, root_pos, root_quat, ready, anchor_index=2, **kwargs
    )
    np.testing.assert_array_equal(annotated.joint_pos, automatic.joint_pos)
    assert automatic.annotation_frame_index is None
    assert annotated.annotation_frame_index == 42


def test_time_lower_bound_reverses_wide_range_wrist_l2_bias():
    backend, source, root_pos, root_quat, ready = inputs(1)
    active = np.asarray(
        [backend.joint_names.index(name) for name in cfm.RIGHT_STRIKE_CHAIN]
    )
    wrist_only = np.zeros(len(active))
    wrist_only[4] = 2.5
    distributed = np.zeros(len(active))
    distributed[:2] = 1.5
    legacy_span = np.asarray([2.0, 2.0, 2.0, 2.0, 10.0, 2.0, 2.0])
    assert np.sum(np.square(wrist_only / legacy_span)) < np.sum(
        np.square(distributed / legacy_span)
    )

    common = dict(
        source_q=source[0],
        ready=ready,
        root_pos=root_pos[0],
        root_quat=root_quat[0],
        backend=backend,
        active=active,
        velocity_limit=backend.velocity_limit[active],
        effort_limit=backend.effort_limit[active],
        config=config(connector_dynamics_samples=2),
        start_label="ready",
        end_label="candidate",
    )
    wrist_time = cfm._connector_time_lower_bound(
        value_active=wrist_only, **common
    ).time_lower_bound_s
    distributed_time = cfm._connector_time_lower_bound(
        value_active=distributed, **common
    ).time_lower_bound_s
    assert distributed_time < wrist_time


def _graph_candidate(value: float, ready_time: float) -> cfm._FrameCandidate:
    connector = cfm.ConnectorDiagnostics(
        start="ready",
        end="frame",
        time_lower_bound_s=ready_time,
        limiting_joint=cfm.RIGHT_STRIKE_CHAIN[0],
        per_joint_time_lower_bound_s=(ready_time,) * 7,
        velocity_limit_rad_s=(100.0,) * 7,
        acceleration_lower_envelope_rad_s2=(100.0,) * 7,
    )
    diagnostics = cfm.FrameDiagnostics(
        frame_index=0,
        position_residual_m=0.0,
        normal_residual_rad=0.0,
        orientation_residual_rad=None,
        max_joint_limit_violation_rad=0.0,
        max_step_from_previous_rad=None,
        ready_max_abs_delta_rad=abs(value),
        previous_max_abs_delta_rad=None,
        ready_connector_time_lower_bound_s=ready_time,
        ready_connector_limiting_joint=cfm.RIGHT_STRIKE_CHAIN[0],
        iterations=1,
        restart_index=0,
    )
    return cfm._FrameCandidate(
        active_joint_pos=np.full(7, value),
        diagnostics=diagnostics,
        direct_ready_connector=connector,
    )


def test_full_window_backtracking_keeps_non_greedy_connectable_branch():
    fast_dead_end = _graph_candidate(0.0, 0.1)
    slower_connected = _graph_candidate(1.0, 0.2)
    path, utilization = cfm._select_full_window_path(
        [
            [fast_dead_end, slower_connected],
            [fast_dead_end, slower_connected],
            [slower_connected],
        ],
        velocity_limit=np.full(7, 100.0),
        fps=1.0,
        max_step_rad=0.2,
        tolerance=1.0e-12,
        frame_indices=[34, 35, 36],
    )
    assert all(np.all(candidate.active_joint_pos == 1.0) for candidate in path)
    assert utilization == 0.0


def test_triangular_and_trapezoidal_rest_to_rest_formula():
    result = cfm._minimum_rest_to_rest_times(
        np.asarray([1.0, 10.0]),
        np.asarray([10.0, 2.0]),
        np.asarray([4.0, 2.0]),
    )
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(6.0)


def test_full_so3_mode_locks_exact_r_source_diag_target():
    backend, source, root_pos, root_quat, ready = inputs(2)
    result = cfm.solve_face_flipped_window(
        source,
        root_pos,
        root_quat,
        ready,
        fps=50.0,
        backend=backend,
        config=config(mode="full-so3"),
        frame_indices=[44, 45],
    )
    assert result.target_rotation_w is not None
    np.testing.assert_allclose(
        result.output_rotation_w, result.target_rotation_w, atol=1.0e-7
    )
    assert result.summary()["max_orientation_residual_rad"] <= 1.0e-7


def test_antipodal_normal_with_roll_point_coupling_uses_multijoint_escape():
    backend, source, root_pos, root_quat, ready = inputs(3)
    coupled = CoupledRightRacketBackend()
    result = cfm.solve_face_flipped_window(
        source,
        root_pos,
        root_quat,
        ready,
        fps=50.0,
        backend=coupled,
        config=config(mode="normal"),
        frame_indices=[43, 44, 45],
    )
    np.testing.assert_allclose(
        result.output_site_pos_w, result.source_site_pos_w, atol=1.0e-8
    )
    np.testing.assert_allclose(
        result.output_raw_plus_y_w, result.target_raw_plus_y_w, atol=1.0e-7
    )
    shoulder_roll = coupled.joint_names.index("right_shoulder_roll_joint")
    assert np.max(np.abs(result.joint_pos[:, shoulder_roll] - source[:, shoulder_roll])) > 0.1


def test_impossible_joint_limits_fail_closed_without_partial_result():
    backend, source, root_pos, root_quat, ready = inputs(1)
    impossible = FakeRightRacketBackend(roll_limit=0.5)
    with pytest.raises(cfm.FaceManifoldError, match="no feasible"):
        cfm.solve_face_flipped_window(
            source,
            root_pos,
            root_quat,
            ready,
            fps=50.0,
            backend=impossible,
            config=config(mode="normal"),
            frame_indices=[44],
        )


def test_impossible_kinematics_fail_closed():
    backend, source, root_pos, root_quat, ready = inputs(1)
    constant = FakeRightRacketBackend(pose_is_constant=True)
    with pytest.raises(cfm.FaceManifoldError, match="no feasible"):
        cfm.solve_face_flipped_window(
            source,
            root_pos,
            root_quat,
            ready,
            fps=50.0,
            backend=constant,
            config=config(mode="normal"),
            frame_indices=[44],
        )


def test_branch_jump_exceeding_continuity_gate_fails_closed():
    backend, source, root_pos, root_quat, ready = inputs(2)
    roll = backend.joint_names.index("right_wrist_roll_joint")
    source[1, roll] = math.pi
    with pytest.raises(cfm.FaceManifoldError, match="continuous"):
        cfm.solve_face_flipped_window(
            source,
            root_pos,
            root_quat,
            ready,
            fps=50.0,
            backend=backend,
            config=config(mode="normal", max_step_rad=0.2),
            frame_indices=[44, 45],
        )


def test_bad_shapes_nonfinite_and_nonconsecutive_frames_rejected():
    backend, source, root_pos, root_quat, ready = inputs(2)
    with pytest.raises(cfm.FaceManifoldError, match="canonical ready"):
        cfm.solve_face_flipped_window(
            source,
            root_pos,
            root_quat,
            ready[:-1],
            fps=50.0,
            backend=backend,
        )
    source[0, 0] = np.nan
    with pytest.raises(cfm.FaceManifoldError, match="finite"):
        cfm.solve_face_flipped_window(
            source,
            root_pos,
            root_quat,
            ready,
            fps=50.0,
            backend=backend,
        )
    clean_source = np.nan_to_num(source)
    with pytest.raises(cfm.FaceManifoldError, match="consecutive"):
        cfm.solve_face_flipped_window(
            clean_source,
            root_pos,
            root_quat,
            ready,
            fps=50.0,
            backend=backend,
            frame_indices=[44, 46],
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_vendor_bh_block_window_when_private_assets_are_present():
    """Conditional integration regression for the real f34..f48 failure."""

    repo = Path(__file__).resolve().parents[3]
    mjcf = (
        repo
        / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model"
        / "a3_pingpong/a3_pingpong.xml"
    )
    source_path = (
        repo
        / "vendor_assets/motion_finalize_20260724/sources"
        / "SHADOW_bh_block_yaw80.npz"
    )
    ready_path = (
        repo
        / "vendor_assets/motion_finalize_20260724/ready/canonical_ready_v1.npz"
    )
    urdf = (
        repo
        / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf"
        / "URDF-JOINT-LINK.urdf"
    )
    if not all(
        path.is_file() for path in (mjcf, urdf, source_path, ready_path)
    ):
        pytest.skip("exact private vendor face-manifold fixture is absent")
    pytest.importorskip("mujoco")
    assert _sha256(mjcf) == (
        "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97"
    )
    assert _sha256(urdf) == (
        "0d83529cf808e2e68036f8168bd8b7a1c9a97d9c536eb9a14981ea4105d6b9ae"
    )
    assert _sha256(source_path) == (
        "55870b981584a458bfd479171046445845cb74171618b71338fd9dc9f66a5fe0"
    )
    assert _sha256(ready_path) == (
        "cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0"
    )
    joint_names = tuple(
        line.strip()
        for line in (
            repo / "configs/a3_runtime_articulation_joint_order.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    backend = cfm.MujocoRightRacketBackend(
        mjcf, joint_names, urdf_path=urdf
    )
    with np.load(source_path, allow_pickle=False) as source:
        body_names = tuple(str(value) for value in source["body_names"].tolist())
        root_column = body_names.index(backend.root_body_name)
        source_q = np.asarray(source["joint_pos"][34:49], dtype=np.float64)
        root_pos = np.asarray(
            source["body_pos_w"][34:49, root_column], dtype=np.float64
        )
        root_quat = np.asarray(
            source["body_quat_w"][34:49, root_column], dtype=np.float64
        )
        fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    with np.load(ready_path, allow_pickle=False) as ready:
        ready_q = np.asarray(ready["joint_pos"], dtype=np.float64)

    # Recorded exact-vendor f44 feasible seed/fixture.  It prevents a future
    # test from mistaking antipodal numerical stagnation for unreachable IK.
    known_f44_active = np.asarray(
        [
            -1.3255545526939592,
            -0.7175121452331961,
            1.7087934969239347,
            0.6336084309147871,
            -2.165229175294329,
            0.772575396913822,
            0.46186385451410755,
        ]
    )
    active = np.asarray(
        [joint_names.index(name) for name in cfm.RIGHT_STRIKE_CHAIN]
    )
    source_position, source_rotation = backend.site_pose(
        source_q[10], root_pos[10], root_quat[10]
    )
    target_normal, _ = cfm.exact_face_flip_target(source_rotation)
    fixture_q = source_q[10].copy()
    fixture_q[active] = known_f44_active
    fixture_position, fixture_rotation = backend.site_pose(
        fixture_q, root_pos[10], root_quat[10]
    )
    assert np.linalg.norm(fixture_position - source_position) < 2.0e-6
    assert (
        np.arccos(
            np.clip(float(fixture_rotation[:, 1] @ target_normal), -1.0, 1.0)
        )
        < 2.0e-5
    )

    result = cfm.solve_face_flipped_window(
        source_q,
        root_pos,
        root_quat,
        ready_q,
        fps=fps,
        backend=backend,
        config=cfm.FaceManifoldConfig(mode="normal", random_restarts=12),
        frame_indices=range(34, 49),
        # Annotation only: it cannot seed/order/rank the numerical graph.
        anchor_index=10,
        active_candidate_seeds=[
            np.radians(
                [-66.58, -15.83, 90.90, 90.02, -87.57, 72.51, -22.88]
            )
        ],
    )
    assert result.summary()["max_position_residual_m"] <= 2.0e-6
    assert result.summary()["max_normal_residual_rad"] <= 2.0e-5
    assert result.max_velocity_residual_mps <= 5.0e-4
    assert all(
        row.max_joint_limit_violation_rad == 0.0
        for row in result.frame_diagnostics
    )
    assert result.annotation_frame_index == 44
    assert result.entry_connector.end == "solved_frame_34"
    assert result.exit_connector.start == "solved_frame_48"
    assert (
        np.max(np.abs(result.joint_pos[10, active] - ready_q[active]))
        < math.radians(145.0)
    )
