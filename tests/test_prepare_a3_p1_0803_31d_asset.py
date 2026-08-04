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


def test_import_is_side_effect_free_and_candidate_path_is_not_current_runtime():
    assert MODULE.DEFAULT_OUTPUT_ROOT.name == "agibot_a3_p1_0803_31d_v1"
    assert MODULE.DEFAULT_OUTPUT_ROOT.name != "agibot_a3"
    assert MODULE.DEFAULT_SUCCESSOR_MANIFEST.is_file()


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


def test_tracked_manifest_binds_project_lock_and_all_order_domains():
    manifest = json.loads(MODULE.DEFAULT_SUCCESSOR_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["asset_id"] == "a3_p1_0803_berkeley_pingpang_31action_normalized_v1"
    assert manifest["candidate_role"] == "future_primary_successor_candidate_not_current_runtime"
    assert manifest["status"] == "pod_import_verified_short_step_diagnostic_standing_pending"
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
        "pod_isaac_import_verified": True,
        "racket_local_contract_verified": True,
        "standing_pose_verified": False,
        "racket_fk_parity_verified": False,
        "dynamics_parity_verified": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }
    receipt = manifest["pod_import_receipt"]
    assert receipt["diagnostic_unauthorized"] is True
    assert receipt["merge_fixed_joints"] is True
    assert receipt["articulation_joint_count"] == 31
    assert receipt["runtime_joint_order_exact"] is True
    assert receipt["articulation_body_count"] == 32
    assert receipt["runtime_body_order_exact"] is True
    assert receipt["finite_steps_all_state_finite"] is True
    assert receipt["formal_standing_hold_verified"] is False
    assert receipt["table_and_self_collision_verified"] is False


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
    assert lineage["normalized_preserves_raw_right_chain_and_site_for_all_common_q"] is True
    assert lineage["normalized_vs_raw_q0"] == {
        "position_norm_m": 0.0,
        "rotation_matrix_max_abs": 0.0,
    }
    assert lineage["successor_vs_current_q0"]["position_norm_m"] == pytest.approx(
        0.009013878161711154, abs=1e-15
    )
    assert lineage["successor_vs_current_q0"]["rotation_matrix_max_abs"] == 0.0
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
    assert root.get("name") == "A3-P1-0803-BerkeleyPingpang-31action-normalized-v1"
    assert not MODULE.parse_rgba(root)
    MODULE.validate_importer_safe_axes_and_meshes(root)
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
