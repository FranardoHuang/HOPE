from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/solve_chingmu_canonical_racket_full_phase.py"
SPEC = importlib.util.spec_from_file_location("canonical_racket_solver", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

GEOM_SPEC = importlib.util.spec_from_file_location(
    "racket_geometry_contract", ROOT / "scripts/racket_geometry_contract.py"
)
GEOM = importlib.util.module_from_spec(GEOM_SPEC)
assert GEOM_SPEC.loader is not None
GEOM_SPEC.loader.exec_module(GEOM)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_signed_dense_face_uses_measured_hit_face_not_family():
    raw = np.tile(np.array([[0.0, 0.0, 1.0]]), (5, 1))
    signed, sign = MODULE.signed_dense_face_normal(
        raw, raw[2], np.array([0.0, 0.0, -1.0])
    )
    assert sign == -1.0
    np.testing.assert_allclose(signed, -raw)


def test_point_velocity_is_same_physical_blade_point():
    t = np.arange(7, dtype=np.float64) / 120.0
    position = np.column_stack((2.0 * t, -0.5 * t, 0.25 * t))
    np.testing.assert_allclose(
        MODULE.point_velocity(position),
        np.tile([2.0, -0.5, 0.25], (7, 1)),
        atol=1.0e-12,
    )


def test_joint_order_contract_binds_gmr_source_and_runtime_target():
    contract = MODULE._load_joint_order_contract(
        ROOT.parents[1] / "configs/a3_joint_order_bijection_v1.json"
    )
    assert contract["contract_id"] == "a3-gmr-dof-pos-to-runtime-articulation-v1"
    assert len(contract["source_names"]) == len(contract["target_names"]) == 31
    assert contract["source_names"] != contract["target_names"]
    MODULE.validate_mjcf_qpos_joint_order(
        contract["source_names"], contract["source_names"]
    )


def test_solver_rejects_runtime_order_as_mjcf_qpos_order():
    contract = MODULE._load_joint_order_contract(
        ROOT.parents[1] / "configs/a3_joint_order_bijection_v1.json"
    )
    try:
        MODULE.validate_mjcf_qpos_joint_order(
            contract["target_names"], contract["source_names"]
        )
    except MODULE.RetargetError as exc:
        assert "GMR source order" in str(exc)
    else:
        raise AssertionError("runtime target order must not be accepted as MJCF qpos order")


def test_exact_73_catalog_joins_manifest_and_selects_double_hit_action():
    manifest = ROOT.parents[1] / "assets/motions/chingmu73_20260728/chingmu_manifest_v1.json"
    catalog = ROOT.parents[1] / "assets/motions/chingmu73_20260728/CLIP_ORDER.json"
    row, clip, binding = MODULE._select_action_contract(
        manifest_path=manifest,
        catalog_path=catalog,
        uid="Take_061_unit09_BH",
    )

    assert row["hit_frame_pkl_120"] == 79
    assert clip["uid"] == "Take_061_unit09_BH"
    assert len(binding["manifest_sha256"]) == len(binding["catalog_sha256"]) == 64


def test_manifest_hit_selection_allows_other_hits_but_rejects_negative_frames():
    selected, frames = MODULE._select_manifest_hit(
        [{"frame_local": 79}, {"frame_local": 249}],
        selected_frame=79,
        frames=280,
    )
    assert selected["frame_local"] == 79
    assert frames == [79, 249]

    try:
        MODULE._select_manifest_hit(
            [{"frame_local": -1}], selected_frame=0, frames=10
        )
    except MODULE.RetargetError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("negative hit frames must fail closed")


def test_long_axis_and_signed_face_define_full_so3():
    long_axis = np.asarray([[1.0, 0.0, 0.0]])
    face = np.asarray([[0.0, -1.0, 0.0]])
    orientation = MODULE._orientation_from_long_face(long_axis, face)

    np.testing.assert_allclose(orientation[0].T @ orientation[0], np.eye(3))
    np.testing.assert_allclose(MODULE._so3_error_deg(orientation, orientation), [0.0])


def test_solver_uses_official_butt_to_blade_axis_not_site_local_x():
    np.testing.assert_array_equal(
        MODULE.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL,
        GEOM.RACKET_BUTT_TO_BLADE_AXIS_LOCAL,
    )
    assert MODULE.ROBOT_RIGID_VISUAL_MESH_SHA256 == (
        GEOM.RACKET_RIGID_VISUAL_MESH_SHA256
    )
    assert not np.array_equal(
        MODULE.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL, np.asarray([1.0, 0.0, 0.0])
    )


def test_solver_loads_exact_urdf_motion_limits_for_optimized_joints():
    urdf = (
        ROOT.parents[1]
        / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"
    )
    assert MODULE._sha256(urdf) == MODULE.EXPECTED_URDF_SHA256
    lower, upper, velocity = MODULE.load_urdf_motion_limits(
        urdf, MODULE.OPTIMIZED_JOINTS
    )

    assert lower.shape == upper.shape == velocity.shape == (10,)
    np.testing.assert_allclose(
        velocity[-3:],
        [15.707963267948966, 12.775810124598491, 12.775810124598491],
    )
    assert np.all(lower < upper)
    assert np.all(velocity > 0.0)

    contract = MODULE._load_joint_order_contract(
        ROOT.parents[1] / "configs/a3_joint_order_bijection_v1.json"
    )
    all_lower, all_upper, all_velocity = MODULE.load_urdf_motion_limits(
        urdf, contract["source_names"]
    )
    assert all_lower.shape == all_upper.shape == all_velocity.shape == (31,)


def test_soft_position_bounds_preserve_symmetric_margin():
    lower, upper = MODULE.soft_position_bounds(
        np.asarray([-2.0, -1.0]), np.asarray([2.0, 3.0]), 0.05
    )
    np.testing.assert_allclose(lower, [-1.8, -0.8])
    np.testing.assert_allclose(upper, [1.8, 2.8])


def test_dynamic_bounds_intersect_velocity_and_acceleration_proxy():
    bounds = MODULE.constrained_frame_bounds(
        position_lower=np.asarray([-2.0, -2.0]),
        position_upper=np.asarray([2.0, 2.0]),
        velocity_rad_s=np.asarray([4.0, 8.0]),
        fps=100.0,
        velocity_fraction=0.5,
        neighbor=np.asarray([0.2, -0.1]),
        second_neighbor=np.asarray([0.19, -0.12]),
        acceleration_proxy_rad_s2=100.0,
    )

    # Velocity permits +/- [0.02, 0.04]; the acceleration proxy permits +/- 0.01
    # around 2*neighbor-second_neighbor = [0.21, -0.08].
    np.testing.assert_allclose(bounds, [(0.2, 0.22), (-0.09, -0.07)])


def test_dynamic_bounds_fail_closed_when_constraints_do_not_intersect():
    try:
        MODULE.constrained_frame_bounds(
            position_lower=np.asarray([-0.1]),
            position_upper=np.asarray([0.1]),
            velocity_rad_s=np.asarray([1.0]),
            fps=100.0,
            velocity_fraction=1.0,
            neighbor=np.asarray([0.09]),
            second_neighbor=np.asarray([-0.09]),
            acceleration_proxy_rad_s2=1.0,
        )
    except MODULE.RetargetError as exc:
        assert "infeasible" in str(exc)
    else:
        raise AssertionError("non-intersecting motion constraints must fail closed")


def test_backward_first_frame_bounds_close_hit_acceleration_seam():
    hit = np.asarray([0.30, -0.20])
    post_hit = np.asarray([0.31, -0.18])
    bounds = MODULE.constrained_frame_bounds(
        position_lower=np.asarray([-2.0, -2.0]),
        position_upper=np.asarray([2.0, 2.0]),
        velocity_rad_s=np.asarray([10.0, 10.0]),
        fps=100.0,
        velocity_fraction=1.0,
        neighbor=hit,
        second_neighbor=post_hit,
        acceleration_proxy_rad_s2=100.0,
    )
    pre_hit_lower = np.asarray([row[0] for row in bounds])
    pre_hit_upper = np.asarray([row[1] for row in bounds])
    for pre_hit in (pre_hit_lower, pre_hit_upper):
        seam_acceleration = (pre_hit - 2.0 * hit + post_hit) * (100.0**2)
        assert np.max(np.abs(seam_acceleration)) <= 100.0 + 1.0e-9


def test_take061u04_v5_candidate_is_content_bound_and_diagnostic_only():
    candidate = (
        ROOT.parents[1]
        / "assets/motions/chingmu_n1_take061u04_mechanical_candidate_v5_20260803"
    )
    pkl_path = candidate / "Take_061_unit04_BH.v70a150.pkl"
    retarget_report_path = candidate / "Take_061_unit04_BH.v70a150.report.json"
    motion_path = candidate / "hope_Take_061_unit04_BH.measured_v5.npz"
    audit_path = candidate / "Take_061_unit04_BH.measured_v5.audit.json"
    receipt = json.loads((candidate / "RECEIPT.json").read_text())

    assert _sha256(pkl_path) == receipt["retarget"]["pkl_sha256"]
    assert _sha256(retarget_report_path) == receipt["retarget"]["report_sha256"]
    assert _sha256(motion_path) == receipt["canonical_motion_npz"]["sha256"]
    assert _sha256(audit_path) == receipt["independent_fk_audit"]["sha256"]
    assert _sha256(candidate / "REPRODUCE.sh") == receipt["materialization"][
        "reproduce_script_sha256"
    ]

    retarget_report = json.loads(retarget_report_path.read_text())
    audit = json.loads(audit_path.read_text())
    with pkl_path.open("rb") as stream:
        retarget = pickle.load(stream)
    with np.load(motion_path, allow_pickle=False) as motion:
        assert motion["joint_pos"].shape == (57, 31)
        assert str(motion["measured_racket_uid"].reshape(-1)[0]) == (
            "Take_061_unit04_BH"
        )
        assert int(motion["measured_racket_mechanical_admission"].reshape(-1)[0]) == 0
        assert int(motion["diagnostic_unauthorized"].reshape(-1)[0]) == 1
        assert int(motion["training_authorized"].reshape(-1)[0]) == 0
        assert float(motion["urdf_soft_limit_margin_fraction"].reshape(-1)[0]) == 0.01
        assert float(motion["urdf_velocity_limit_fraction"].reshape(-1)[0]) == 0.70
        assert float(motion["acceleration_proxy_rad_s2"].reshape(-1)[0]) == 150.0
        assert str(motion["source_retarget_pkl_sha256"].reshape(-1)[0]) == _sha256(
            pkl_path
        )
        assert str(motion["retarget_report_sha256"].reshape(-1)[0]) == _sha256(
            retarget_report_path
        )

    assert retarget_report["admitted"] is True
    assert retarget_report["mechanical_admission"] is False
    assert retarget["measured_racket_mechanical_admission"] is False
    assert retarget["diagnostic_unauthorized"] is True
    assert audit["admitted"] is True
    assert audit["motion_sha256"] == _sha256(motion_path)
    assert receipt["authorization"]["mechanical_admission"] is False
    assert receipt["authorization"]["diagnostic_unauthorized"] is True


def test_atomic_publish_is_no_replace_and_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "receipt.json"
    MODULE._atomic_bytes_no_replace(output, b"first\n")
    assert output.read_bytes() == b"first\n"
    try:
        MODULE._atomic_bytes_no_replace(output, b"second\n")
    except FileExistsError:
        pass
    else:
        raise AssertionError("atomic publication must not replace an existing receipt")
    assert list(tmp_path.iterdir()) == [output]
