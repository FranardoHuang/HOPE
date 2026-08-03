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


def test_tracked_manifest_is_fail_closed_and_binds_all_order_domains():
    manifest = json.loads(MODULE.DEFAULT_SUCCESSOR_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["asset_id"] == "a3_p1_0803_berkeley_pingpang_31action_normalized_v1"
    assert manifest["abi"]["policy_action_dim"] == 31
    assert manifest["abi"]["runtime_joint_set_exact"] is True
    assert manifest["abi"]["urdf_movable_document_order_equals_gmr"] is True
    assert manifest["abi"]["runtime_body_names_all_present"] is True
    assert manifest["abi"]["joint_bijection_path"] == "configs/a3_joint_order_bijection_v1.json"
    assert len(manifest["normalization_diff"]["mesh_reference_rewrites"]) == 78
    assert len(manifest["normalization_diff"]["removed_missing_collision_elements"]) == 20
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
    assert manifest["authorization"] == {
        "current_runtime_pointer_changed": False,
        "canonical_runtime": False,
        "pod_isaac_import_verified": False,
        "standing_pose_verified": False,
        "racket_fk_parity_verified": False,
        "dynamics_parity_verified": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


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
