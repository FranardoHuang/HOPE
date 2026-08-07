"""Static and local-asset checks for the A3-P1 0803 31-action successor."""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_a3_p1_0803_31d_asset.py"
SPEC = importlib.util.spec_from_file_location("prepare_a3_p1_0803_31d_asset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CORRECTION = MODULE.MIRROR_SYMMETRY_ORIGIN_CORRECTIONS["right_elbow_joint"]


def _mirror_correction_root(
    right_xyz: str = CORRECTION["raw_origin_xyz"],
    right_rpy: str = CORRECTION["raw_origin_rpy"],
    left_xyz: str = CORRECTION["mirror_reference_origin_xyz"],
) -> ET.Element:
    """Minimal two-joint root exercising only the mirror-symmetry premises."""

    root = ET.Element("robot", {"name": "mirror-correction-fixture"})
    for name, xyz, rpy in (
        ("right_elbow_joint", right_xyz, right_rpy),
        ("left_elbow_joint", left_xyz, "0 0 0"),
    ):
        joint = ET.SubElement(root, "joint", {"name": name, "type": "revolute"})
        ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
        ET.SubElement(joint, "axis", {"xyz": "0.0 1.0 0.0"})
    return root


def test_import_is_side_effect_free_and_candidate_path_is_not_current_runtime():
    assert MODULE.DEFAULT_OUTPUT_ROOT.name == "agibot_a3_p1_0803_31d_v2"
    assert MODULE.DEFAULT_OUTPUT_ROOT.name != "agibot_a3"
    assert MODULE.PREDECESSOR_OUTPUT_ROOT.name == "agibot_a3_p1_0803_31d_v1"
    assert MODULE.DEFAULT_OUTPUT_ROOT != MODULE.PREDECESSOR_OUTPUT_ROOT
    assert MODULE.DEFAULT_SUCCESSOR_MANIFEST.is_file()
    assert MODULE.PREDECESSOR_MANIFEST.is_file()


def test_successor_root_hard_rejects_current_runtime_and_descendants():
    with pytest.raises(MODULE.AssetError, match="current runtime asset"):
        MODULE.require_isolated_successor_root(MODULE.ACTIVE_ASSET_ROOT)
    with pytest.raises(MODULE.AssetError, match="current runtime asset"):
        MODULE.require_isolated_successor_root(MODULE.ACTIVE_ASSET_ROOT / "nested")
    with pytest.raises(MODULE.AssetError, match="immutable raw source"):
        MODULE.require_isolated_successor_root(
            MODULE.DEFAULT_SOURCE_ROOT / "derived", MODULE.DEFAULT_SOURCE_ROOT
        )
    with pytest.raises(MODULE.AssetError, match="immutable raw source"):
        MODULE.require_isolated_successor_root(
            MODULE.DEFAULT_SOURCE_ROOT.parent, MODULE.DEFAULT_SOURCE_ROOT
        )
    MODULE.require_isolated_successor_root(MODULE.DEFAULT_OUTPUT_ROOT, MODULE.DEFAULT_SOURCE_ROOT)


def test_successor_root_hard_rejects_the_pod_verified_predecessor():
    with pytest.raises(MODULE.AssetError, match="Pod-verified predecessor"):
        MODULE.require_isolated_successor_root(MODULE.PREDECESSOR_OUTPUT_ROOT)
    with pytest.raises(MODULE.AssetError, match="Pod-verified predecessor"):
        MODULE.require_isolated_successor_root(MODULE.PREDECESSOR_OUTPUT_ROOT / "urdf")


def test_mirror_symmetry_correction_moves_one_component_of_one_joint():
    root = _mirror_correction_root()
    applied = MODULE.apply_mirror_symmetry_corrections(root)
    assert [item["joint"] for item in applied] == ["right_elbow_joint"]
    item = applied[0]
    assert item["delivered_origin_xyz"] == "0.001 0 -0.1325"
    assert item["corrected_origin_xyz"] == "0.01 0 -0.1325"
    assert item["corrected_component"] == "x"
    assert item["correction_m"] == pytest.approx(0.009, abs=1e-15)
    assert item["mirror_reference_joint"] == "left_elbow_joint"
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    assert joints["right_elbow_joint"].find("origin").get("xyz") == "0.01 0 -0.1325"
    # rpy, axis and the mirror reference itself must be untouched.
    assert joints["right_elbow_joint"].find("origin").get("rpy") == "0 0 0"
    assert joints["right_elbow_joint"].find("axis").get("xyz") == "0.0 1.0 0.0"
    assert joints["left_elbow_joint"].find("origin").get("xyz") == "0.01 0 -0.1325"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"right_xyz": "0.002 0 -0.1325"}, "premise drifted"),
        ({"right_xyz": "0.01 0 -0.1325"}, "premise drifted"),
        ({"right_xyz": "0.001 0 -0.133"}, "premise drifted"),
        ({"right_rpy": "0 0.1 0"}, "premise drifted"),
        ({"left_xyz": "0.02 0 -0.1325"}, "mirror reference origin drifted"),
    ],
)
def test_mirror_symmetry_correction_refuses_to_absorb_a_redelivery(kwargs, match):
    """A re-delivery that changes any premise must fail loudly, not be patched silently."""

    with pytest.raises(MODULE.AssetError, match=match):
        MODULE.apply_mirror_symmetry_corrections(_mirror_correction_root(**kwargs))


def test_mirror_symmetry_correction_refuses_a_contract_that_understates_itself(monkeypatch):
    understated = dict(CORRECTION)
    understated["corrected_origin_xyz"] = "0.01 0 -0.14"
    monkeypatch.setitem(
        MODULE.MIRROR_SYMMETRY_ORIGIN_CORRECTIONS, "right_elbow_joint", understated
    )
    with pytest.raises(MODULE.AssetError, match="contract declares only"):
        MODULE.apply_mirror_symmetry_corrections(_mirror_correction_root())

    misreported = dict(CORRECTION)
    misreported["correction_m"] = 0.0
    monkeypatch.setitem(
        MODULE.MIRROR_SYMMETRY_ORIGIN_CORRECTIONS, "right_elbow_joint", misreported
    )
    with pytest.raises(MODULE.AssetError, match="contract declares"):
        MODULE.apply_mirror_symmetry_corrections(_mirror_correction_root())


def test_correction_contract_self_declares_provisional_and_unconfirmed():
    contract = MODULE.MIRROR_SYMMETRY_CORRECTION_CONTRACT
    assert contract["status"] == "provisional_pending_vendor_confirmation"
    assert contract["vendor_question_open"] is True
    assert contract["workbook_agrees_with_raw_urdf"] is True
    manifest = json.loads(MODULE.DEFAULT_SUCCESSOR_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["mirror_symmetry_correction_contract"] == contract
    assert manifest["normalization_diff"]["mirror_symmetry_correction_contract"] == contract
    assert manifest["authorization"]["reproduces_delivered_joint_origins_exactly"] is False
    assert manifest["authorization"]["mirror_symmetry_correction_applied"] is True
    assert manifest["authorization"]["mirror_symmetry_correction_vendor_confirmed"] is False


def test_predecessor_receipt_is_untouched_and_its_pod_evidence_does_not_transfer():
    predecessor = json.loads(MODULE.PREDECESSOR_MANIFEST.read_text(encoding="utf-8"))
    assert predecessor["asset_id"] == "a3_p1_0803_berkeley_pingpang_31action_normalized_v1"
    assert predecessor["status"] == "pod_import_verified_short_step_diagnostic_standing_pending"
    assert predecessor["authorization"]["pod_isaac_import_verified"] is True
    assert predecessor["pod_import_receipt"]["finite_steps_all_state_finite"] is True
    assert predecessor["output"]["closure"]["sha256"] == MODULE.V1_POD_VERIFIED_CLOSURE_SHA256
    assert predecessor["output"]["urdf_sha256"] == MODULE.V1_POD_VERIFIED_URDF_SHA256
    # The predecessor never carried a correction; that is what makes its receipt reusable
    # as evidence about the raw delivery and unusable as evidence about the successor.
    assert "mirror_symmetry_corrections" not in predecessor["normalization_diff"]

    manifest = json.loads(MODULE.DEFAULT_SUCCESSOR_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["predecessor"]["pod_import_evidence_transfers"] is False
    assert manifest["predecessor"]["manifest_path"] == "configs/a3_p1_0803_31d_v1.json"
    assert manifest["predecessor"]["manifest_sha256"] == MODULE.sha256_path(
        MODULE.PREDECESSOR_MANIFEST
    )
    assert manifest["pod_import_receipt"] is None
    assert manifest["status"] == "host_static_candidate_pod_import_pending"
    assert manifest["authorization"]["pod_isaac_import_verified"] is False
    assert manifest["output"]["closure"]["sha256"] != MODULE.V1_POD_VERIFIED_CLOSURE_SHA256
    assert manifest["output"]["urdf_sha256"] != MODULE.V1_POD_VERIFIED_URDF_SHA256


def test_tracked_manifest_binds_project_lock_and_all_order_domains():
    manifest = json.loads(MODULE.DEFAULT_SUCCESSOR_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["asset_id"] == "a3_p1_0803_berkeley_pingpang_31action_normalized_v2"
    assert manifest["candidate_role"] == "future_primary_successor_candidate_not_current_runtime"
    assert manifest["abi"]["policy_action_dim"] == 31
    assert manifest["abi"]["runtime_joint_set_exact"] is True
    assert manifest["abi"]["urdf_movable_document_order_equals_gmr"] is True
    assert manifest["abi"]["runtime_body_names_all_present"] is True
    assert manifest["abi"]["joint_bijection_path"] == "configs/a3_joint_order_bijection_v1.json"
    assert len(manifest["normalization_diff"]["mesh_reference_rewrites"]) == 82
    assert len(manifest["normalization_diff"]["removed_missing_collision_elements"]) == 20
    assert manifest["normalization_diff"]["malformed_fixed_axis_normalizations"] == [
        {
            "joint": name,
            "type": "fixed",
            "raw_axis_xyz": raw_axis,
            "normalized_axis": None,
            "reason": "URDF fixed joints do not use an axis; omit importer-invalid non-3-vector data",
        }
        for name, raw_axis in MODULE.EXPECTED_MALFORMED_FIXED_AXES.items()
    ]
    aliases = manifest["normalization_diff"]["usd_safe_mesh_aliases"]
    assert {
        item["raw_basename"]: item["normalized_basename"] for item in aliases
    } == MODULE.USD_SAFE_MESH_ALIASES
    assert all(item["bytes_unchanged"] is True for item in aliases)
    assert manifest["normalization_diff"]["fixed_gripper_subtree"]["converted_to_fixed_joint_names"] == [
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint6",
        "left_joint9",
        "left_joint10",
        "left_joint13",
        "left_joint8",
        "left_joint15",
    ]
    lock = manifest["normalization_diff"]["fixed_gripper_subtree"]
    assert lock["lock_contract"] == MODULE.PROJECT_GRIPPER_LOCK_CONTRACT
    assert lock["all_q0_within_raw_limits"] is True
    assert lock["retained_subtree_mass_kg"] == pytest.approx(
        MODULE.EXPECTED_GRIPPER_SUBTREE_MASS_KG, abs=1e-12
    )
    assert manifest["normalization_diff"]["normalized_unique_link_mass_kg"] == pytest.approx(
        MODULE.EXPECTED_NORMALIZED_UNIQUE_LINK_MASS_KG, abs=1e-12
    )
    assert manifest["project_gripper_lock_contract"] == MODULE.PROJECT_GRIPPER_LOCK_CONTRACT
    removed = manifest["normalization_diff"]["removed_missing_collision_elements"]
    assert len(removed) == MODULE.EXPECTED_MISSING_GRIPPER_COLLISION_COUNT
    assert all(item["link"].startswith("left_") for item in removed)
    assert tuple((item["link"], item["reference"]) for item in removed) == (
        MODULE.EXPECTED_MISSING_GRIPPER_COLLISIONS
    )
    assert manifest["authorization"] == {
        "current_runtime_pointer_changed": False,
        "canonical_runtime": False,
        "materialization_authorized": True,
        "project_q0_gripper_lock_authorized": True,
        "missing_gripper_collision_elements_explicitly_disabled": True,
        "reproduces_delivered_joint_origins_exactly": False,
        "mirror_symmetry_correction_applied": True,
        "mirror_symmetry_correction_vendor_confirmed": False,
        "pod_isaac_import_verified": False,
        "racket_local_contract_verified": True,
        "standing_pose_verified": False,
        "racket_fk_parity_verified": False,
        "dynamics_parity_verified": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


def test_right_racket_local_center_is_exact_but_successor_world_fk_is_new():
    manifest = json.loads(MODULE.DEFAULT_SUCCESSOR_MANIFEST.read_text(encoding="utf-8"))
    racket = manifest["right_racket_contract"]
    assert tuple(float(value) for value in racket["origin_xyz"].split()) == (
        MODULE.OFFICIAL_RACKET_SITE_XYZ_M
    )
    assert tuple(float(value) for value in racket["origin_rpy"].split()) == (
        MODULE.OFFICIAL_RACKET_SITE_RPY_RAD
    )
    assert racket["mesh_sha256"] == MODULE.EXPECTED_RACKET_MESH_SHA256
    assert racket["local_contract_exact_current_and_raw"] is True
    assert racket["official_paddle_center_control_point"] is True
    lineage = racket["fk_lineage"]
    assert lineage["right_hand_and_paddle_link_inertials_exact_current"] is True
    assert lineage["right_racket_fixed_joint_semantics_exact_current"] is True
    # The successor deliberately no longer reproduces the raw world site.
    assert lineage["normalized_preserves_raw_right_chain_and_site_for_all_common_q"] is False
    assert lineage["normalized_right_chain_deviates_from_raw_only_by_declared_corrections"] is True
    assert lineage["normalized_vs_raw_q0"]["position_norm_m"] == pytest.approx(
        MODULE.EXPECTED_CORRECTED_VS_RAW_RACKET_DELTA_M, abs=1e-15
    )
    assert lineage["normalized_vs_raw_q0"]["rotation_matrix_max_abs"] == 0.0
    assert lineage["raw_vs_current_q0"]["position_norm_m"] == pytest.approx(
        0.009013878161711154, abs=1e-15
    )
    # The correction shrinks the successor-vs-current racket offset by ~18x ...
    assert lineage["successor_vs_current_q0"]["position_norm_m"] == pytest.approx(
        0.0004999999900224823, abs=1e-15
    )
    assert lineage["successor_vs_current_q0"]["rotation_matrix_max_abs"] == 0.0
    # ... but 0.5 mm still exceeds the 1e-4 m racket FK gate, so revalidation still stands.
    assert lineage["successor_vs_current_q0"]["position_norm_m"] > 1.0e-4
    assert lineage["successor_world_site_requires_new_motion_fk_revalidation"] is True
    assert manifest["authorization"]["racket_fk_parity_verified"] is False


def test_raw_intake_records_observed_metadata_defects_and_exact_racket_hashes():
    intake = json.loads(MODULE.RAW_INTAKE_MANIFEST.read_text(encoding="utf-8"))
    assert intake["structure"]["unique_link_name_count"] == 63
    assert intake["structure"]["duplicate_link_names"] == ["imu_in_pelvis_link"]
    defects = intake["known_defects_and_open_questions"]
    assert defects["case_mismatched_mesh_reference_count_on_case_sensitive_filesystem"] == 78
    assert defects["missing_gripper_collision_mesh_reference_count"] == 20
    assert defects["duplicate_imu_in_pelvis_link_definition"] is True
    racket = intake["right_racket_contract"]
    assert racket["red_mesh_sha256"] == "94182ec1c7c64db8c5ec7ce5f9aad44d427f433a6aae5cf23aa655e077633842"
    assert racket["black_mesh_sha256"] == "5f0e772ea9ed81e5b70f5dfb4ded49f9d269c54c893249857209f85168361b1b"


def test_local_ignored_asset_reproduces_and_imports_as_static_31_action_urdf():
    if not MODULE.DEFAULT_SOURCE_ROOT.is_dir() or not MODULE.DEFAULT_OUTPUT_ROOT.is_dir():
        pytest.skip("private raw/generated A3-P1 assets are intentionally absent from a fresh clone")
    report = MODULE.check(
        MODULE.DEFAULT_SOURCE_ROOT,
        MODULE.DEFAULT_OUTPUT_ROOT,
        MODULE.RAW_INTAKE_MANIFEST,
        MODULE.DEFAULT_SUCCESSOR_MANIFEST,
    )
    assert report["status"] == "PASS"
    assert report["movable_joint_count"] == 31
    assert report["link_count"] == 63
    assert report["joint_count"] == 62
    assert report["mesh_reference_count"] == 104
    root = ET.parse(MODULE.DEFAULT_OUTPUT_ROOT / "urdf/model.urdf").getroot()
    assert root.get("name") == "A3-P1-0803-BerkeleyPingpang-31action-normalized-v2"
    assert not MODULE.parse_rgba(root)
    MODULE.validate_importer_safe_axes_and_meshes(root)
    corrected = next(
        joint for joint in root.findall("joint") if joint.get("name") == "right_elbow_joint"
    )
    assert corrected.find("origin").get("xyz") == CORRECTION["corrected_origin_xyz"]
    axes = {
        joint.get("name"): joint.find("axis")
        for joint in root.findall("joint")
        if joint.get("name") in MODULE.EXPECTED_MALFORMED_FIXED_AXES
    }
    assert set(axes) == set(MODULE.EXPECTED_MALFORMED_FIXED_AXES)
    assert all(axis is None for axis in axes.values())
    refs = {Path(ref).name for ref in MODULE.mesh_refs(root)}
    assert set(MODULE.USD_SAFE_MESH_ALIASES.values()).issubset(refs)
    assert set(MODULE.USD_SAFE_MESH_ALIASES).isdisjoint(refs)
    for raw_name, alias_name in MODULE.USD_SAFE_MESH_ALIASES.items():
        raw_path = MODULE.DEFAULT_SOURCE_ROOT / "meshes" / raw_name
        alias_path = MODULE.DEFAULT_OUTPUT_ROOT / "meshes" / alias_name
        assert raw_path.read_bytes() == alias_path.read_bytes()


def test_successor_differs_from_predecessor_only_by_name_and_the_declared_correction():
    if not MODULE.DEFAULT_OUTPUT_ROOT.is_dir() or not MODULE.PREDECESSOR_OUTPUT_ROOT.is_dir():
        pytest.skip("private generated A3-P1 assets are intentionally absent from a fresh clone")
    previous = (MODULE.PREDECESSOR_OUTPUT_ROOT / "urdf/model.urdf").read_text(encoding="utf-8")
    current = (MODULE.DEFAULT_OUTPUT_ROOT / "urdf/model.urdf").read_text(encoding="utf-8")
    changed = [
        (before, after)
        for before, after in zip(previous.splitlines(), current.splitlines())
        if before != after
    ]
    assert len(previous.splitlines()) == len(current.splitlines())
    assert len(changed) == 2
    assert changed[0][0].strip().endswith('normalized-v1">')
    assert changed[0][1].strip().endswith('normalized-v2">')
    assert CORRECTION["raw_origin_xyz"] in changed[1][0]
    assert CORRECTION["corrected_origin_xyz"] in changed[1][1]
    # Geometry bytes are shared verbatim: the correction is a frame edit, not a re-mesh.
    previous_meshes = sorted(
        path.name for path in (MODULE.PREDECESSOR_OUTPUT_ROOT / "meshes").iterdir()
    )
    current_meshes = sorted(path.name for path in (MODULE.DEFAULT_OUTPUT_ROOT / "meshes").iterdir())
    assert previous_meshes == current_meshes
    for name in current_meshes:
        assert MODULE.sha256_path(MODULE.PREDECESSOR_OUTPUT_ROOT / "meshes" / name) == (
            MODULE.sha256_path(MODULE.DEFAULT_OUTPUT_ROOT / "meshes" / name)
        )
