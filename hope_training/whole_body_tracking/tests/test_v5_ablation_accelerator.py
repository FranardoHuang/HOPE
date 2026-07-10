"""Pure-Python tests for the deterministic V5/Phase ablation accelerator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "v5_ablation_accelerator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v5_ablation_accelerator_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A = _load_module()
SCHEDULE_SHA = "a" * 64
READY_SHA = "b" * 64
MJCF_SHA = "c" * 64
EXECUTION_SHA = "d" * 64


def _record(teacher, timing, path, *, extension_ok=True, **overrides):
    guards = {key: True for key in A.BASE_HARD_GUARDS}
    if path != "original":
        guards[A.MORPH_HARD_GUARD] = True
    record = {
        "teacher": teacher,
        "timing": timing,
        "path": path,
        "feasible": True,
        "hard_failures": [],
        "hard_guards": guards,
    }
    if path in A.EXTENSION_PATHS:
        record["extension_evidence"] = {
            "original_path_limited": extension_ok,
            "stroke_length_gain_m": 0.08 if extension_ok else 0.0,
            "a_min_reduction_mps2": 3.0 if extension_ok else 0.0,
            "min_torque_margin_gain": 0.05 if extension_ok else 0.0,
        }
    record.update(overrides)
    return record


def _manifest(records, **kwargs):
    return A.build_manifest(
        {"schema_version": 1, "records": list(records)},
        schedule_sha256=SCHEDULE_SHA,
        schedule_seed=17,
        ready_state_sha256=READY_SHA,
        mjcf_sha256=MJCF_SHA,
        execution_contract_sha256=EXECUTION_SHA,
        **kwargs,
    )


def _card(candidate, question_ids, outcomes, **overrides):
    card = {
        "candidate_id": candidate["candidate_id"],
        "schedule_sha256": SCHEDULE_SHA,
        "ready_state_mode": A.FORMAL_READY_STATE_MODE,
        "ready_state_sha256": READY_SHA,
        "mjcf_sha256": MJCF_SHA,
        "execution_contract_sha256": EXECUTION_SHA,
        "contract_exact": True,
        "hard_safety_pass": True,
        "hard_failures": [],
        "attempts": [
            {"question_id": qid, "return_success": bool(outcome)}
            for qid, outcome in zip(question_ids, outcomes)
        ],
    }
    card.update(overrides)
    return card


def _scorecards(candidates, outcomes_by_id, manifest=None):
    n = len(next(iter(outcomes_by_id.values())))
    qids = [f"q{i:03d}" for i in range(n)]
    return {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"] if manifest else None,
        "schedule_sha256": SCHEDULE_SHA,
        "schedule_seed": 17,
        "ready_state_mode": A.FORMAL_READY_STATE_MODE,
        "ready_state_sha256": READY_SHA,
        "mjcf_sha256": MJCF_SHA,
        "execution_contract_sha256": EXECUTION_SHA,
        "question_ids": qids,
        "candidates": [
            _card(candidate, qids, outcomes_by_id[candidate["candidate_id"]])
            for candidate in candidates
        ],
    }


def _find(manifest, teacher, timing, path):
    return next(
        candidate for candidate in manifest["candidates"]
        if candidate["recipe"]["teacher"] == teacher
        and candidate["recipe"]["timing"] == timing
        and candidate["recipe"]["path"] == path
    )


def test_manifest_is_deterministic_semantic_and_budget_bound(tmp_path):
    records = [
        _record("v4", "robot", "original"),
        _record("v4", "human", "followthrough"),
        _record("v5", "robot", "backswing_20"),
        _record("v5", "human", "combined_40"),
    ]
    first = _manifest(records, training_seed_base=100, m4_iterations=30000)
    second = _manifest(list(reversed(records)), training_seed_base=100, m4_iterations=30000)

    # Record order is input presentation, not scientific identity.
    assert first == second
    body = dict(first)
    claimed = body.pop("manifest_sha256")
    assert claimed == A.content_sha256(body)
    assert first["protocol"]["same_question_exam"] == {
        "schedule_sha256": SCHEDULE_SHA,
        "schedule_seed": 17,
        "denominator": "all_attempts",
        "one_question_one_reset": True,
        "ready_state_mode": A.FORMAL_READY_STATE_MODE,
        "ready_state_sha256": READY_SHA,
        "mjcf_sha256": MJCF_SHA,
        "execution_contract_sha256": EXECUTION_SHA,
    }
    assert first["budgets"]["M2"]["num_envs"] == 512
    assert first["budgets"]["M2"]["iterations"] == 25
    assert first["budgets"]["M3"]["num_envs"] == 4096
    assert first["budgets"]["M3"]["iterations"] == 2000
    assert first["budgets"]["M4"]["candidate_limit"] == 2
    assert first["budgets"]["M4"]["training_seeds"] == [100, 101, 102]
    assert first["budgets"]["M4"]["launch_ready"] is True

    task = _find(first, "task_only", "robot", "original")
    assert task["recipe"]["reference_mode"] == "none"
    assert not any(
        row["recipe"]["teacher"] == "task_only"
        and (row["recipe"]["timing"] != "robot" or row["recipe"]["path"] != "original")
        for row in first["candidates"] + first["rejected_candidates"]
    )
    assert len({row["candidate_id"] for row in first["candidates"]}) == len(first["candidates"])

    # The CLI writer produces stable bytes as well as a stable semantic hash.
    feasibility_path = tmp_path / "feasibility.json"
    feasibility_path.write_text(json.dumps({"schema_version": 1, "records": records}))
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    argv = [
        "manifest", "--feasibility", str(feasibility_path),
        "--schedule-sha256", SCHEDULE_SHA, "--schedule-seed", "17",
        "--ready-state-sha256", READY_SHA, "--mjcf-sha256", MJCF_SHA,
        "--execution-contract-sha256", EXECUTION_SHA,
        "--training-seed-base", "100", "--m4-iterations", "30000",
    ]
    assert A.main(argv + ["--output", str(out_a)]) == 0
    assert A.main(argv + ["--output", str(out_b)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()


def test_invalid_combinations_and_extension_evidence_fail_closed():
    with pytest.raises(ValueError, match="no teacher clock"):
        A.validate_candidate_spec(
            {"teacher": "task_only", "timing": "human", "path": "original"}
        )
    with pytest.raises(ValueError, match="reference path"):
        A.validate_candidate_spec(
            {"teacher": "task_only", "timing": "robot", "path": "backswing_20"}
        )

    bad_extension = _record("v5", "robot", "backswing_20", extension_ok=False)
    missing_guard = _record("v5", "human", "followthrough")
    del missing_guard["hard_guards"]["strike_invariant_pass"]
    explicit_failure = _record(
        "v4", "human", "original", hard_failures=["self_collision"]
    )
    manifest = _manifest([
        _record("v4", "robot", "original"),
        bad_extension,
        missing_guard,
        explicit_failure,
    ])

    assert _find(manifest, "v4", "robot", "original")
    rejected = {
        (row["recipe"]["teacher"], row["recipe"]["timing"], row["recipe"]["path"]):
        row["reasons"]
        for row in manifest["rejected_candidates"]
    }
    ext_reasons = rejected[("v5", "robot", "backswing_20")]
    assert "original_path_limit_not_proven" in ext_reasons
    assert "extension_gain_not_positive:stroke_length_gain_m" in ext_reasons
    assert "extension_gain_not_positive:min_torque_margin_gain" in ext_reasons
    assert "guard_not_true:strike_invariant_pass" in rejected[
        ("v5", "human", "followthrough")
    ]
    assert "reported_hard_failure" in rejected[("v4", "human", "original")]
    # A missing report is also a recorded rejection, never an implicit pass.
    assert rejected[("v5", "robot", "original")] == ["missing_feasibility_record"]


def test_small_paired_sample_is_ranked_but_not_falsely_eliminated():
    manifest = _manifest([
        _record("v4", "robot", "original"),
        _record("v5", "robot", "backswing_20"),
        _record("v5", "robot", "followthrough"),
    ])
    candidates = manifest["candidates"]
    outcomes = {
        candidates[0]["candidate_id"]: [False, False, False, False],
        _find(manifest, "v4", "robot", "original")["candidate_id"]:
            [False, False, False, True],
        _find(manifest, "v5", "robot", "backswing_20")["candidate_id"]:
            [True, True, True, True],
        _find(manifest, "v5", "robot", "followthrough")["candidate_id"]:
            [False, False, False, False],
    }
    decision = A.halve_scorecards(manifest, _scorecards(candidates, outcomes, manifest))

    assert decision["best_candidate_id"] == _find(
        manifest, "v5", "robot", "backswing_20"
    )["candidate_id"]
    assert set(decision["survivor_candidate_ids"]) == set(outcomes)
    assert decision["round_summary"]["conservative_hold"] is True
    assert decision["round_summary"]["hold_reason"] == "insufficient_paired_evidence"
    worst = next(
        row for row in decision["ranked_valid_candidates"]
        if row["candidate_id"] == _find(manifest, "v5", "robot", "followthrough")["candidate_id"]
    )
    assert worst["paired_vs_best"]["discordant"] == 4
    assert worst["paired_vs_best"]["stable_dominated"] is False


def test_strong_signal_halves_but_keeps_mechanism_controls():
    manifest = _manifest([
        _record("v4", "robot", "original"),
        _record("v5", "human", "original"),
        _record("v5", "robot", "backswing_20"),
        _record("v5", "robot", "followthrough"),
        _record("v5", "robot", "combined_20"),
    ])
    candidates = manifest["candidates"]
    n = 100
    rates = {
        _find(manifest, "task_only", "robot", "original")["candidate_id"]: 10,
        _find(manifest, "v4", "robot", "original")["candidate_id"]: 20,
        _find(manifest, "v5", "human", "original")["candidate_id"]: 30,
        _find(manifest, "v5", "robot", "backswing_20")["candidate_id"]: 100,
        _find(manifest, "v5", "robot", "followthrough")["candidate_id"]: 0,
        _find(manifest, "v5", "robot", "combined_20")["candidate_id"]: 5,
    }
    outcomes = {cid: [i < successes for i in range(n)] for cid, successes in rates.items()}
    decision = A.halve_scorecards(manifest, _scorecards(candidates, outcomes, manifest))

    best = _find(manifest, "v5", "robot", "backswing_20")["candidate_id"]
    assert decision["best_candidate_id"] == best
    assert decision["round_summary"]["conservative_hold"] is False
    assert len(decision["survivor_candidate_ids"]) == 4  # winner + three distinct controls
    protected = decision["protected_controls"]
    assert protected["task_only_teacher_baseline"] in decision["survivor_candidate_ids"]
    assert protected["v4_nonprofessional_teacher_control"] in decision["survivor_candidate_ids"]
    assert protected["human_timing_control"] in decision["survivor_candidate_ids"]
    assert protected["original_path_control"] == protected["human_timing_control"]
    eliminated_ids = {row["candidate_id"] for row in decision["eliminated"]}
    assert _find(manifest, "v5", "robot", "followthrough")["candidate_id"] in eliminated_ids
    assert _find(manifest, "v5", "robot", "combined_20")["candidate_id"] in eliminated_ids
    ranking = [row["candidate_id"] for row in decision["ranked_valid_candidates"]]
    assert ranking[0] == best
    assert ranking[-1] == _find(manifest, "v5", "robot", "followthrough")["candidate_id"]


def test_hard_safety_precedes_control_protection_and_all_attempts_are_counted():
    manifest = _manifest([_record("v5", "robot", "original")])
    candidates = manifest["candidates"]
    qids = ["q0", "q1", "q2"]
    task = _find(manifest, "task_only", "robot", "original")
    v5 = _find(manifest, "v5", "robot", "original")
    task_card = _card(task, qids, [False, False, False])
    task_card["attempts"][1]["physical_fall"] = True
    v5_card = _card(v5, qids, [True, False, False])
    # eligible=false must not remove q1 from the all-attempt denominator.
    v5_card["attempts"][1]["eligible"] = False
    scorecards = {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": SCHEDULE_SHA,
        "schedule_seed": 17,
        "ready_state_mode": A.FORMAL_READY_STATE_MODE,
        "ready_state_sha256": READY_SHA,
        "mjcf_sha256": MJCF_SHA,
        "execution_contract_sha256": EXECUTION_SHA,
        "question_ids": qids,
        "candidates": [task_card, v5_card],
    }
    decision = A.halve_scorecards(manifest, scorecards)

    assert task["candidate_id"] not in decision["survivor_candidate_ids"]
    hard_row = next(row for row in decision["eliminated"] if row["candidate_id"] == task["candidate_id"])
    assert hard_row["phase"] == "hard_safety_or_contract"
    assert any("physical_fall" in reason for reason in hard_row["reasons"])
    v5_row = next(row for row in decision["ranked_valid_candidates"] if row["candidate_id"] == v5["candidate_id"])
    assert v5_row["attempts"] == 3
    assert v5_row["successes"] == 1
    assert v5_row["return_rate_all_attempts"] == pytest.approx(1.0 / 3.0)


def test_scorecard_contract_is_content_and_schedule_bound():
    manifest = _manifest([_record("v5", "robot", "original")])
    candidates = manifest["candidates"]
    scorecards = _scorecards(
        candidates,
        {candidate["candidate_id"]: [True, False] for candidate in candidates},
        manifest,
    )
    tampered = dict(manifest)
    tampered["budgets"] = dict(tampered["budgets"])
    tampered["budgets"]["M2"] = dict(tampered["budgets"]["M2"])
    tampered["budgets"]["M2"]["iterations"] = 26
    with pytest.raises(ValueError, match="manifest_sha256"):
        A.halve_scorecards(tampered, scorecards)

    wrong_schedule = dict(scorecards)
    wrong_schedule["schedule_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="does not match manifest"):
        A.halve_scorecards(manifest, wrong_schedule)

    wrong_ready = dict(scorecards)
    wrong_ready["ready_state_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="ready_state_sha256 does not match manifest"):
        A.halve_scorecards(manifest, wrong_ready)

    mixed_candidate = dict(scorecards)
    mixed_candidate["candidates"] = [dict(row) for row in scorecards["candidates"]]
    mixed_candidate["candidates"][0]["execution_contract_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="common manifest contract"):
        A.halve_scorecards(manifest, mixed_candidate)
