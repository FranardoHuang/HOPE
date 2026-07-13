from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_motion_spatial_retarget_candidates.py"
PLAN = ROOT / "configs" / "motion_backhand_loop_bc_proposal_selection_prereg_20260713.json"
RESULT = ROOT / "configs" / "motion_backhand_loop_bc_proposal_selection_results_20260713.json"
SPEC = importlib.util.spec_from_file_location("select_motion_spatial_retarget_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_bytes(M.canonical_json_bytes(payload))


def tracked_plan() -> tuple[dict, str]:
    plan_sha = M.sha256_file(PLAN)
    plan, actual = M.validate_plan(PLAN, plan_sha)
    assert actual == plan_sha
    return plan, plan_sha


def make_proposal(
    asset_id: str,
    question_id: str,
    tier: str,
    serial: str,
    *,
    frame: int,
    translation_norm: float,
    yaw: float,
    return_margin: float = 0.05,
    body_clearance: float = 0.25,
) -> dict:
    return {
        "asset_id": asset_id,
        "candidate_id": digest(f"candidate-{serial}"),
        "capture_extrinsic_claim": False,
        "frame": frame,
        "landing_xy_m": [2.5, 0.1],
        "mapped_racket_pos_w_m": [0.4, 0.1, 1.1],
        "net_center_z_m": 1.3,
        "phase": 0.5,
        "predecessor_full_result_sha256": "c299b7a04417e855005ad315b40203204bb0cc192398d83980179b212e6bef53",
        "promotion_status": "missing_candidate_certificate",
        "question_id": question_id,
        "return_margin_m": return_margin,
        "side": "backhand",
        "source_dense_racket_body_clearance_m": body_clearance,
        "source_motion_sha256": digest(f"source-{asset_id}"),
        "tier": tier,
        "transform_semantics": M.TRANSFORM_SEMANTICS,
        "translation_norm_m": translation_norm,
        "translation_w_m": [translation_norm, 0.0, 0.0],
        "vertical_position_error_m": 0.02,
        "yaw_deg": yaw,
    }


def synthetic_artifact() -> tuple[dict, list[dict]]:
    side_questions = {
        side: [digest(f"{side}-question-{index}") for index in range(32)]
        for side in ("forehand", "backhand")
    }
    cells: list[dict] = []
    cell_by_key: dict[tuple[str, str, str], dict] = {}
    for asset_id, side in M.EXPECTED_ASSET_SIDES.items():
        for question_id in side_questions[side]:
            for tier in ("R0", "R1"):
                cell = {
                    "accepted_candidate_ids": [],
                    "asset_id": asset_id,
                    "proposal_count_before_certificate": 0,
                    "question_id": question_id,
                    "reported_proposals": [],
                    "side": side,
                    "tier": tier,
                }
                cells.append(cell)
                cell_by_key[(asset_id, question_id, tier)] = cell

    raw_proposals: list[dict] = []
    b_questions = side_questions["backhand"]
    for index in range(16):
        tier = "R0" if index < 3 else "R1"
        yaw = 0.0 if index < 3 else float((index % 4 + 1) * 5)
        row = make_proposal(
            "franco_backhand_loop_b",
            b_questions[index],
            tier,
            f"b-{index}-{tier}",
            frame=48 + (index % 2),
            translation_norm=0.10 + index * 0.01,
            yaw=yaw,
            return_margin=0.08 - index * 0.001,
            body_clearance=0.30 - index * 0.001,
        )
        raw_proposals.append(row)
        cell_by_key[(row["asset_id"], row["question_id"], row["tier"])]["reported_proposals"].append(row)
        if index < 3:
            alias = copy.deepcopy(row)
            alias["candidate_id"] = digest(f"b-{index}-R1-alias")
            alias["tier"] = "R1"
            raw_proposals.append(alias)
            cell_by_key[(alias["asset_id"], alias["question_id"], alias["tier"])]["reported_proposals"].append(alias)

    for index in range(3):
        row = make_proposal(
            "franco_backhand_loop_c",
            b_questions[20 + index],
            "R1",
            f"c-{index}",
            frame=50,
            translation_norm=0.20 + index * 0.02,
            yaw=-10.0 + index * 5.0,
            return_margin=0.03 - index * 0.001,
            body_clearance=0.32,
        )
        raw_proposals.append(row)
        cell_by_key[(row["asset_id"], row["question_id"], row["tier"])]["reported_proposals"].append(row)

    for cell in cells:
        cell["proposal_count_before_certificate"] = len(cell["reported_proposals"])
    assert len(cells) == 640
    assert len(raw_proposals) == 22
    artifact = {
        "accepted_candidates": [],
        "asset_count": 10,
        "capture_table_pose_observed": False,
        "cells": cells,
        "certified_candidate_count": 0,
        "claims": {
            "capture_extrinsic": None,
            "final_arbiter": "agibot_vendor_mujoco_gate3_gate3b_no_reset",
            "real_robot_authorized": False,
            "rl_authorized": False,
            "schema2_L0_L1_table_net": "missing_or_no_matching_pass_certificates",
            "spatial_retarget": "bounded atomic whole-motion SE2 proposal screen",
            "topp_authorized": False,
        },
        "contact_phase_truth": None,
        "formal_eligible": False,
        "generated_utc": "2026-07-13T00:00:00Z",
        "manifest": {
            "bytes": 7319,
            "path": "/ignored/manifest.json",
            "sha256": "0f757c8c4abfc9bf5070b7db79f494fa1d97a45ddb222609898662eff63af66a",
        },
        "predecessor": {
            "bytes": 792241,
            "path": "/ignored/predecessor.json",
            "sha256": "c299b7a04417e855005ad315b40203204bb0cc192398d83980179b212e6bef53",
        },
        "proposal_count_before_certificate": 22,
        "question_count": 64,
        "question_semantic_sha256": digest("synthetic-question-contract"),
        "robot_approved": False,
        "schema_version": 1,
        "status": M.INPUT_STATUS,
    }
    return artifact, raw_proposals


def test_tracked_plan_and_result_are_exact_and_block_promotion() -> None:
    plan, plan_sha = tracked_plan()
    assert plan_sha == "691fd516477a8d7b56aa9fb562a76684e421f6050e447d707467ee267b0b9b8c"
    assert M.sha256_file(RESULT) == "8a80a409ca69e2fa73757b139b8496bb9cdda2e6a66d3fab48412051b408d2be"
    result = M.read_json_bytes(RESULT.read_bytes(), "tracked selection")
    result = M.validate_selection(plan, plan_sha, result)
    assert result["primary_count"] == 2
    assert result["materialization_authorized"] is False
    assert result["training_authorized"] is False
    by_asset = {row["asset_id"]: row for row in result["assets"]}
    assert by_asset["franco_backhand_loop_b"]["selected_primary"]["candidate_id"] == (
        "98e7b883b29d302dc7a24fd3c564648c1f929ff2391e24e58558dcba58af3c14"
    )
    assert by_asset["franco_backhand_loop_c"]["selected_primary"]["candidate_id"] == (
        "aa0c86fd350987bf30e56aebde9789bf9df430b0ec5c3c15cd235410794af299"
    )
    assert len(by_asset["franco_backhand_loop_b"]["yaw_zero_r0_r1_aliases_removed"]) == 3
    assert len(by_asset["franco_backhand_loop_b"]["fallback_ladder"]) == 15
    assert len(by_asset["franco_backhand_loop_c"]["fallback_ladder"]) == 2


def test_synthetic_full_grid_validates_deduplicates_and_sorts() -> None:
    plan, plan_sha = tracked_plan()
    artifact, raw_proposals = synthetic_artifact()
    raw = M.canonical_json_bytes(artifact)
    runtime_plan = copy.deepcopy(plan)
    runtime_plan["proposal_input"]["bytes"] = len(raw)
    runtime_plan["proposal_input"]["sha256"] = M.sha256_bytes(raw)
    _, flattened = M.validate_proposal_artifact(runtime_plan, raw)
    assert len(flattened) == 22
    result = M.build_selection(runtime_plan, plan_sha, flattened)
    by_asset = {row["asset_id"]: row for row in result["assets"]}
    b = by_asset["franco_backhand_loop_b"]
    c = by_asset["franco_backhand_loop_c"]
    assert b["raw_proposal_count"] == 19
    assert b["deduplicated_proposal_count"] == 16
    assert len(b["yaw_zero_r0_r1_aliases_removed"]) == 3
    assert b["selected_primary"]["candidate_id"] == raw_proposals[0]["candidate_id"]
    assert b["selected_primary"]["tier"] == "R0"
    assert c["selected_primary"]["translation_norm_m"] == pytest.approx(0.20)


def test_deduplication_rejects_nonzero_or_same_tier_aliases() -> None:
    question = digest("question")
    first = make_proposal(
        "franco_backhand_loop_b",
        question,
        "R0",
        "first",
        frame=49,
        translation_norm=0.1,
        yaw=5.0,
    )
    second = copy.deepcopy(first)
    second["candidate_id"] = digest("second")
    second["tier"] = "R1"
    with pytest.raises(M.SelectionError, match="yaw-zero"):
        M.deduplicate_candidates([first, second])

    first["yaw_deg"] = 0.0
    second["yaw_deg"] = 0.0
    second["tier"] = "R0"
    with pytest.raises(M.SelectionError, match="R0/R1"):
        M.deduplicate_candidates([first, second])


def test_sort_key_uses_every_frozen_tiebreaker_in_order() -> None:
    question = digest("question")
    base = make_proposal(
        "franco_backhand_loop_b",
        question,
        "R1",
        "base",
        frame=49,
        translation_norm=0.1,
        yaw=10.0,
        return_margin=0.05,
        body_clearance=0.25,
    )
    rows = []
    for serial, changes in (
        ("norm", {"translation_norm_m": 0.09}),
        ("yaw", {"yaw_deg": 5.0}),
        ("margin", {"return_margin_m": 0.06}),
        ("clearance", {"source_dense_racket_body_clearance_m": 0.26}),
        ("frame", {"frame": 48}),
        ("id", {}),
    ):
        row = copy.deepcopy(base)
        row["candidate_id"] = digest(serial)
        row.update(changes)
        rows.append(row)
    assert min(rows, key=M.candidate_sort_key)["candidate_id"] == digest("norm")
    same_norm = [row for row in rows if row["translation_norm_m"] == 0.1]
    assert min(same_norm, key=M.candidate_sort_key)["candidate_id"] == digest("yaw")


@pytest.mark.parametrize("code", M.STOP_CODES)
def test_failure_policy_stops_asset_for_internal_stage_failures(code: str) -> None:
    plan, plan_sha = tracked_plan()
    selection = M.validate_selection(
        plan,
        plan_sha,
        M.read_json_bytes(RESULT.read_bytes(), "selection"),
    )
    b = selection["assets"][0]["selected_primary"]
    decision = M.resolve_outcome(selection, "franco_backhand_loop_b", b["candidate_id"], code)
    assert decision["status"] == "asset_stopped_no_fallback"
    assert decision["next_candidate"] is None
    assert decision["materialization_authorized"] is False


def test_only_external_geometry_failure_advances_frozen_ladder() -> None:
    plan, plan_sha = tracked_plan()
    selection = M.validate_selection(
        plan,
        plan_sha,
        M.read_json_bytes(RESULT.read_bytes(), "selection"),
    )
    b_asset = selection["assets"][0]
    primary = b_asset["selected_primary"]
    decision = M.resolve_outcome(
        selection,
        "franco_backhand_loop_b",
        primary["candidate_id"],
        M.ADVANCE_CODE,
    )
    assert decision["status"] == "next_fallback_selected_certification_blocked"
    assert decision["next_candidate"] == b_asset["fallback_ladder"][0]
    with pytest.raises(M.SelectionError, match="unknown outcome_code"):
        M.resolve_outcome(
            selection,
            "franco_backhand_loop_b",
            primary["candidate_id"],
            "looks_bad_try_another",
        )


def test_last_external_geometry_failure_exhausts_asset() -> None:
    plan, plan_sha = tracked_plan()
    selection = M.validate_selection(
        plan,
        plan_sha,
        M.read_json_bytes(RESULT.read_bytes(), "selection"),
    )
    c_asset = selection["assets"][1]
    last = c_asset["fallback_ladder"][-1]
    decision = M.resolve_outcome(
        selection,
        "franco_backhand_loop_c",
        last["candidate_id"],
        M.ADVANCE_CODE,
    )
    assert decision["status"] == "fallback_exhausted_asset_stopped"
    assert decision["next_candidate"] is None


def test_plan_mutation_and_selection_authorization_fail_closed(tmp_path: Path) -> None:
    plan, plan_sha = tracked_plan()
    mutated = copy.deepcopy(plan)
    mutated["fallback_policy"]["advance_only_on"].append("schema2_materialization_failure")
    bad_plan = tmp_path / "bad-plan.json"
    write_json(bad_plan, mutated)
    with pytest.raises(M.SelectionError, match="fallback policy"):
        M.validate_plan(bad_plan, M.sha256_file(bad_plan))

    result = M.read_json_bytes(RESULT.read_bytes(), "selection")
    result["materialization_authorized"] = True
    with pytest.raises(M.SelectionError, match="must remain false"):
        M.validate_selection(plan, plan_sha, result)


def test_strict_json_and_no_clobber(tmp_path: Path) -> None:
    with pytest.raises(M.SelectionError, match="duplicate JSON key"):
        M.read_json_bytes(b'{"a": 1, "a": 2}\n', "duplicate")
    output = tmp_path / "result.json"
    size, digest_value = M.publish_no_clobber(output, {"finite": 1.0})
    assert size == output.stat().st_size
    assert digest_value == M.sha256_file(output)
    with pytest.raises(M.SelectionError, match="already exists"):
        M.publish_no_clobber(output, {"finite": 2.0})


def test_frame_window_violation_stops_before_selection() -> None:
    plan, plan_sha = tracked_plan()
    artifact, _ = synthetic_artifact()
    raw = M.canonical_json_bytes(artifact)
    runtime_plan = copy.deepcopy(plan)
    runtime_plan["proposal_input"]["bytes"] = len(raw)
    runtime_plan["proposal_input"]["sha256"] = M.sha256_bytes(raw)
    _, flattened = M.validate_proposal_artifact(runtime_plan, raw)
    b = next(row for row in flattened if row["asset_id"] == "franco_backhand_loop_b")
    b["frame"] = 44
    with pytest.raises(M.SelectionError, match="outside frozen frame window"):
        M.build_selection(runtime_plan, plan_sha, flattened)
