from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "configs/motion_video_gmr_phase_counterfactual_prereg_20260711.json"
EVIDENCE = REPO / "configs/motion_video_gmr_frame_contract_results_20260711.json"
SCREEN = REPO / "scripts/screen_motion_gmr_phase_safety.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_screen():
    spec = importlib.util.spec_from_file_location("screen_phase_counterfactual_contract", SCREEN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_counterfactual_prereg_enables_only_the_bound_per_asset_frame_path():
    expected_sha = "fee1b1f9a68fcc0323c1be5832db1b29bdc5f49421712c6f44506d16dae45529"
    assert _sha(PLAN) == expected_sha
    module = _load_screen()
    plan = module.validate_manifest(
        PLAN, expected_sha, require_ready=True, verify_files=False
    )
    frame = plan["frame_contract"]
    assert frame["returnability_enabled"] is True
    assert frame["gmr_world_to_hope_table_transform_verified"] is True
    assert frame["gmr_world_to_hope_matrix_4x4"] is None
    assert frame["transform_scope"] == "per_asset"
    assert frame["mirror_status"] == "verified_not_mirrored"
    assert frame["side_swap_required"] is False
    assert frame["capture_table_pose_observed"] is False
    assert "question outcomes cannot modify" in frame["anti_tuning_invariant"]
    assert list(frame["per_asset_gmr_world_to_hope_matrix_4x4"]) == plan["expected_asset_ids"]
    for matrix_value in frame["per_asset_gmr_world_to_hope_matrix_4x4"].values():
        matrix = np.asarray(matrix_value, dtype=np.float64)
        assert np.allclose(matrix[3], [0, 0, 0, 1])
        assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-12)


def test_counterfactual_prereg_binds_frame_evidence_tool_and_frozen_paper():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert plan["frame_contract_evidence"]["sha256"] == _sha(EVIDENCE)
    assert plan["frame_contract_evidence"]["bytes"] == EVIDENCE.stat().st_size
    assert plan["tool_contract"]["screen"]["sha256"] == _sha(SCREEN)
    assert plan["tool_contract"]["screen"]["bytes"] == SCREEN.stat().st_size
    assert (
        plan["question_schedule"]["expected_semantic_sha256"]
        == "4dfa0548d898fa6456d09261216e26ff9547c1a9dca2da221835c3cfa25332c7"
    )
    assert plan["question_schedule"]["count_per_side"] == 32
    assert (
        plan["frame_contract"]["per_asset_transform_semantic_sha256"]
        == evidence["frame_contract"]["per_asset_transform_semantic_sha256"]
    )
    evidence_by_id = {row["asset_id"]: row for row in evidence["assets"]}
    for asset_id, matrix in plan["frame_contract"][
        "per_asset_gmr_world_to_hope_matrix_4x4"
    ].items():
        assert matrix == evidence_by_id[asset_id]["transform"]["matrix_4x4"]
    assert plan["accepted_safety_predecessor"]["result"]["sha256"] == (
        "d2518f766720dabf979e2a95e5044fdf7fcc7d85b0622471b476888febf301d8"
    )
    assert plan["output_contract"]["result"].endswith(
        "/phase_safety_v5/phase_safety_result.json"
    )
    assert "not real-capture returnability" in plan["phase_selection_contract"][
        "exact_coverage_semantics"
    ]
