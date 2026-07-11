from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/audit_motion_gmr_frame_contract.py"
PLAN = REPO / "configs/motion_video_gmr_frame_contract_prereg_20260711.json"
REVIEW = REPO / "configs/motion_video_mirror_witness_review_20260711.json"
RESULT = REPO / "configs/motion_video_gmr_frame_contract_results_20260711.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_motion_gmr_frame_contract_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_frame_prereg_review_and_result_are_content_addressed():
    assert _sha(PLAN) == "f625e0a0a403b8908a2e5c575917cea93e9e2a2c88e2db4db385d2b14a07e97e"
    assert _sha(REVIEW) == "729ef33b12ed71b27a20011c17d7a31e7b44e808acc9e6f661e14d518b1d4fb5"
    assert _sha(RESULT) == "e70492becf5a2fae5ee74724d22f9ca9d2e874d535231e3ee6649f01669048f0"
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["tool_contract"]["audit"]["sha256"] == _sha(SCRIPT)
    assert plan["tool_contract"]["audit"]["bytes"] == SCRIPT.stat().st_size


def test_derived_transform_maps_root_and_heading_without_reflection():
    half = math.sqrt(0.5)
    result = mod.derive_transform(
        np.array([[2.0, 3.0, 0.9], [2.0, 3.0, 0.9]]),
        np.array([[0.0, 0.0, half, half], [0.0, 0.0, half, half]]),
        proper_tolerance=1e-9,
        heading_tolerance=1e-10,
        root_xy_tolerance=1e-10,
    )
    matrix = np.asarray(result["matrix_4x4"])
    assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(matrix[:3, :3]), 1.0)
    assert np.allclose(result["mapped_frame0_root_pos_m"], [0.0, 0.0, 0.9])
    assert np.allclose(result["mapped_frame0_pelvis_forward"], [1.0, 0.0, 0.0], atol=1e-12)
    assert result["ground_z_preserved"] is True


def test_accepted_result_proves_per_asset_frame_and_mirror_but_not_capture_table():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "complete_verified_per_asset_hope_frame_and_not_mirrored"
    assert result["contact_phase_truth"] is None
    assert result["frame_contract"]["gmr_world_to_hope_table_transform_verified"] is True
    assert result["frame_contract"]["transform_scope"] == "per_asset"
    assert result["frame_contract"]["capture_table_pose_observed"] is False
    assert result["mirror_contract"]["status"] == "verified_not_mirrored"
    assert result["mirror_contract"]["side_swap_required"] is False
    assert result["eligibility"]["immutable_64_question_returnability_phase_screen"] is True
    assert result["eligibility"]["topp"] is False
    assert result["eligibility"]["real_capture_returnability"] is None
    assert result["returnability_semantics"]["real_capture_table"]["coverage"] is None
    assert result["source_semantics"]["target_table"] == {
        "center_y_m": 0.0,
        "far_edge_x_m": 3.24,
        "half_width_m": 0.7625,
        "length_m": 2.74,
        "near_edge_x_m": 0.5,
        "net_height_m": 0.1525,
        "net_x_m": 1.87,
        "surface_z_m": 0.76,
        "width_m": 1.525,
    }

    assert len(result["assets"]) == 10
    assert min(row["right_left_arm_motion_energy_ratio"] for row in result["assets"]) > 9.9
    for row in result["assets"]:
        matrix = np.asarray(row["transform"]["matrix_4x4"], dtype=np.float64)
        rotation = matrix[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)
        assert np.allclose(rotation @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        assert np.linalg.norm(row["transform"]["mapped_frame0_root_pos_m"][:2]) < 1e-12
        assert abs(row["transform"]["mapped_frame0_heading_yaw_rad"]) < 1e-12
        assert row["mirror_status"] == "verified_not_mirrored"
        assert row["side_swap_required"] is False
        assert len(row["mirror_witness"]["crop_rgb_sha256"]) == 64
