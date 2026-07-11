from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "configs/motion_video_gmr_phase_safety_results_20260711.json"
PREREG = REPO / "configs/motion_video_gmr_phase_safety_prereg_20260711.json"
GROUND = REPO / "configs/motion_video_canonical_gmr_ground_results_20260711.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_safety_result_accepts_only_safety_and_keeps_returnability_null():
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    assert value["status"] == "complete_dense_safety_returnability_blocked"
    assert value["formal_eligible"] is False
    assert value["robot_approved"] is False
    assert value["contact_phase_truth"] is None
    assert value["frame_contract"]["returnability_enabled"] is False
    assert value["frame_contract"]["gmr_world_to_hope_matrix_4x4"] is None
    assert value["question_paper"]["consumed_for_returnability"] is False

    aggregate = value["aggregate"]
    assert aggregate["assets"] == 10
    assert aggregate["source_frames"] == 654
    assert aggregate["dense_samples"] == 5162
    assert aggregate["safe_source_frames"] == 654
    assert aggregate["unsafe_source_frames"] == 0
    assert aggregate["ground_dangerous_dense_samples"] == 0
    assert aggregate["self_collision_dangerous_dense_samples"] == 0
    assert aggregate["racket_body_clearance_dangerous_dense_samples"] == 0
    assert aggregate["racket_body_clearance_warning_dense_samples"] == 0
    assert aggregate["minimum_racket_body_clearance_asset"] == "franco_backhand_loop_a"
    assert aggregate["minimum_racket_body_clearance_m"] > 0.04
    assert aggregate["phase_candidates_reported"] is False
    assert aggregate["question_coverage_reported"] is False
    assert aggregate["library_selector_reported"] is False
    assert aggregate["two_vs_four_decision_reported"] is False
    assert set(value["library_outputs"].values()) == {
        "blocked_unverified_gmr_world_to_hope_table_frame"
    }


def test_phase_safety_result_binds_prereg_grounding_and_revoked_v2():
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    assert value["preregistration"]["sha256"] == _sha(PREREG)
    assert value["preregistration"]["bytes"] == PREREG.stat().st_size
    assert value["canonical_grounding_result"]["sha256"] == _sha(GROUND)
    assert value["canonical_grounding_result"]["bytes"] == GROUND.stat().st_size
    attempts = {item["attempt"]: item for item in value["predecessor_attempts"]}
    assert attempts["v1_validate"]["accepted_evidence"] is False
    assert attempts["v2_run"]["accepted_evidence"].startswith("dense safety")
    assert "virtual-return" in attempts["v2_run"]["revoked_evidence"]
    assert value["invariants"]["v2_v3_v4_safety_subtrees_equal_for_all_assets"] is True
    assert value["invariants"]["returnability_not_executed_in_v3_or_v4"] is True
    assert attempts["v3_run"]["v3_safety_subtrees_equal_v4_for_all_assets"] is True


def test_every_asset_is_safe_but_has_no_phase_or_coverage_claim():
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = value["results"]
    assert len(rows) == 10
    assert sum(row["frames"] for row in rows) == 654
    assert sum(row["dense_frames"] for row in rows) == 5162
    assert all(row["safe_source_frames"] == row["frames"] for row in rows)
    assert all(
        row["phase_and_coverage_status"]
        == "blocked_unverified_gmr_world_to_hope_table_frame"
        for row in rows
    )
    assert max(row["top_safe_speed_mps"] for row in rows) > 6.4
    assert min(row["minimum_racket_body_clearance_m"] for row in rows) > 0.04
