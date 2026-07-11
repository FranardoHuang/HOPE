from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "configs/motion_video_gmr_phase_counterfactual_results_20260711.json"
PLAN = REPO / "configs/motion_video_gmr_phase_counterfactual_prereg_20260711.json"
FRAME = REPO / "configs/motion_video_gmr_frame_contract_results_20260711.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counterfactual_result_is_bound_and_fail_closed():
    assert _sha(RESULT) == "f8e9e3b68bdb36f12383869f82346c36fc4ea7adff8c53e8d4666e5c47d4eb87"
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    assert value["preregistration"]["sha256"] == _sha(PLAN)
    assert value["preregistration"]["bytes"] == PLAN.stat().st_size
    assert value["frame_contract_evidence"]["sha256"] == _sha(FRAME)
    assert value["frame_contract_evidence"]["bytes"] == FRAME.stat().st_size
    assert value["frame_contract_evidence"]["capture_table_pose_observed"] is False
    assert value["question_paper"]["consumed_for_returnability"] is True
    assert value["question_paper"]["questions"] == 64
    assert value["contact_phase_truth"] is None
    assert value["real_capture_returnability"] is None
    assert value["runtime"]["pid"] == value["runtime"]["pgid"] == 1471093
    assert value["runtime"]["fatal_log_token_scan"] == "pass"
    assert value["runtime"]["matching_processes_after"] == []
    assert value["runtime"]["training_checkout"]["clean_after"] is True


def test_zero_exact_coverage_does_not_promote_or_reject_the_library():
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    assert len(value["assets"]) == 10
    assert all(row["exact_question_coverage"] == 0 for row in value["assets"])
    by_id = {row["asset_id"]: row for row in value["assets"]}
    assert by_id["franco_backhand_loop_b"]["top_intrinsic_return_count_of_32"] == 32
    assert by_id["franco_backhand_loop_c"]["top_intrinsic_return_count_of_32"] == 27
    assert by_id["franco_backhand_loop_a"]["top_intrinsic_return_count_of_32"] == 1
    library = value["library_result"]
    assert library["franco_two_vs_four_common_support_count"] == 0
    assert library["decision"] == "inconclusive_zero_common_support_do_not_prefer_two_or_four"
    assert "does not prove" in library["warning"]
    assert value["eligibility"]["topp"].startswith("paused_until_spatial_retarget")
    assert value["eligibility"]["vendor_mujoco_gate3_gate3b"].endswith("no_reset")
