from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_phase1_timing_exam_0p5.py"
CONFIG = ROOT / "configs" / "phase1_timing_exam_0p5_k100_20260716.json"
SPEC = importlib.util.spec_from_file_location("phase1_timing_exam_0p5_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
T = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = T
SPEC.loader.exec_module(T)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seal(document: dict, field: str) -> dict:
    document[field] = T._content_sha(document, field)
    return document


def _write_pretty(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_schedule(tmp_path: Path) -> tuple[Path, dict]:
    items = []
    for index in range(100):
        clip = index % 2
        side = T.SIDE_ORDER[clip]
        items.append(
            {
                "schedule_index": index,
                "clip": clip,
                "bank_row": index // 2,
                "question_id": f"{side}:{T.sha256_bytes(f'question:{index}'.encode())}",
                "hold_steps": (index * 17) % 101,
                "attempt_seed": index + 1000,
                "repeat": 0,
            }
        )
    schedule = {
        "artifact_type": T.SOURCE_ARTIFACT_TYPE,
        "bank_schema_version": 3,
        "bank_sha256": "b" * 64,
        "clip_order": list(T.SIDE_ORDER),
        "question_counts": [50, 50],
        "per_clip_quota": 50,
        "schedule_seed": 0,
        "hold_range": [0, 100],
        "hold_semantics": "stand-policy-actions-then-raw-frame0-v1",
        "no_wrap": True,
        "items": items,
        "schema_version": 3,
    }
    schedule["schedule_sha256"] = T.canonical_sha256(schedule)
    path = tmp_path / "source.schedule.json"
    path.write_bytes(T.canonical_json_bytes(schedule) + b"\n")
    source = {
        "path": "/workspace/test/source.schedule.json",
        "bytes": path.stat().st_size,
        "file_sha256": T.sha256_file(path),
        "semantic_sha256": schedule["schedule_sha256"],
        "question_id_order_sha256": T.canonical_sha256(
            [item["question_id"] for item in items]
        ),
        "bank_sha256": schedule["bank_sha256"],
        "bank_source_family_sha256": "c" * 64,
        "scheduled_attempts": 100,
        "per_side": {"forehand": 50, "backhand": 50},
    }
    return path, source


def _synthetic_spec(tmp_path: Path) -> tuple[dict, Path, dict]:
    schedule_path, source = _synthetic_schedule(tmp_path)
    spec = _load(CONFIG)
    spec["source_schedule"] = source
    _seal(spec, "contract_content_sha256")
    validated = T.validate_spec_document(
        spec,
        root=ROOT,
        verify_production_source=False,
    )
    schedule = T.load_source_schedule(schedule_path, source_contract=source)
    return validated, schedule_path, schedule


def _paper_fixture(tmp_path: Path) -> tuple[dict, dict, Path, str]:
    spec, _, schedule = _synthetic_spec(tmp_path)
    spec_file_sha = "d" * 64
    paper = T.build_paper(
        spec=spec,
        spec_file_sha256=spec_file_sha,
        source_schedule=schedule,
    )
    T.validate_paper_document(
        paper,
        spec=spec,
        spec_file_sha256=spec_file_sha,
        source_schedule=schedule,
    )
    paper_path = tmp_path / "timing.paper.json"
    T.write_paper_exclusive(paper_path, paper)
    return spec, paper, paper_path, T.sha256_file(paper_path)


def _result(paper: dict, paper_file_sha: str, *, successes_per_side: int = 31) -> dict:
    used = {side: 0 for side in T.SIDE_ORDER}
    attempts = []
    for row in paper["rows"]:
        side = row["side"]
        success = used[side] < successes_per_side
        used[side] += int(success)
        attempts.append(
            {
                "schedule_index": row["schedule_index"],
                "question_id": row["question_id"],
                "side": side,
                "initial_state_id": row["initial_state_id"],
                "tts_ticks": row["tts_ticks"],
                "time_law_id": row["time_law_id"],
                "expected_feasible": row["expected_feasible"],
                "feasibility_status": row["feasibility_status"],
                "observation_valid": True,
                "returned": success,
                "position_error_m": 0.01 if success else 0.2,
                "velocity_error_mps": 0.1 if success else 1.0,
                "signed_normal_error_deg": 2.0 if success else 30.0,
                "planner_infeasible": None,
                "physical_fall": False,
                "self_hit": False,
                "illegal_table_or_net_contact": False,
                "reset_or_teleport": False,
                "deadline_shifted": False,
            }
        )
    result = {
        "schema_version": 2,
        "artifact_type": T.RESULT_ARTIFACT_TYPE,
        "paper_file_sha256": paper_file_sha,
        "paper_semantic_sha256": paper["paper_semantic_sha256"],
        "source_scorecard_file_sha256": "0" * 64,
        "checkpoint_sha256": "1" * 64,
        "checkpoint_hard_contract_sha256": "2" * 64,
        "evaluator_source_sha256": "3" * 64,
        "converter_source_sha256": "5" * 64,
        "evaluation_execution_contract_sha256": "4" * 64,
        "engine": "vendor_mujoco",
        "evaluation_contract_exact": True,
        "attempts": attempts,
    }
    return _seal(result, "result_content_sha256")


def _array(values: list[float]) -> dict:
    return {"shape": [len(values)], "values": values}


def _instrumentation(*, signed_normal: list[float] | None = None) -> dict:
    signed_normal = signed_normal or [0.0, 1.0, 0.0]
    value = {
        "schema": "hope.cross-engine-state-instrumentation.v1",
        "kind": "isaac_question_state",
        "observation_phase": "exact_strike",
        "coordinate_contract": {},
        "base": {},
        "racket": {
            "position_env_m": _array([0.0, 0.0, 0.0]),
            "linear_velocity_world_mps": _array([0.0, 0.0, 0.0]),
            "face_normal_signed_pre_orient_world": _array(signed_normal),
            "face_normal_raw_plus_y_world": _array([0.0, 1.0, 0.0]),
            "analytic_face_normal_oriented_world": _array([0.0, 1.0, 0.0]),
        },
        "target": {
            "racket_position_env_m": _array([0.0, 0.0, 0.0]),
            "racket_linear_velocity_world_mps": _array([0.0, 0.0, 0.0]),
            "face_normal_world": _array([0.0, 1.0, 0.0]),
        },
        "incoming_ball": {},
        "analytic_counterfactual": {
            "available": True,
            "capability": "analytic_counterfactual_contact_and_flight_v1",
            "capture_gate": True,
            "net_clear": True,
            "on_opponent": True,
            "landing_valid": True,
            "landing_xy_env_m": _array([1.0, 0.0]),
        },
        "physical_truth": {"available": False, "capability": "disabled", "reason": "diagnostic"},
    }
    value["sha256"] = T.canonical_sha256(value)
    return value


def _scorecard_source_closure() -> dict:
    paths = {
        "evaluator_sha256": ROOT
        / "hope_training/whole_body_tracking/scripts/isaac_bank_exam.py",
        "adapter_sha256": ROOT
        / "hope_training/whole_body_tracking/scripts/isaac_bank_exam_adapter.py",
        "schedule_module_sha256": ROOT
        / "hope_training/whole_body_tracking/scripts/bank_exam_schedule.py",
        "isaac_scorer_sha256": ROOT
        / (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/virtual_ball.py"
        ),
        "ball_physics_yaml_sha256": ROOT / "configs/ball_physics_venue.yaml",
        "timing_adapter_sha256": ROOT
        / "hope_training/whole_body_tracking/scripts/isaac_timing_exam_adapter.py",
    }
    return {
        "git_head": "a" * 40,
        **{field: T.sha256_file(path) for field, path in paths.items()},
    }


def _isaac_scorecard(
    paper: dict,
    paper_file_sha: str,
    schedule: dict,
    *,
    checkpoint_sha: str,
    hard_sha: str,
) -> dict:
    attempts = []
    for index, row in enumerate(paper["rows"]):
        instrumentation = _instrumentation()
        attempts.append(
            {
                "schedule_index": index,
                "env_id": index,
                "clip": 0 if row["side"] == "forehand" else 1,
                "side": row["side"],
                "bank_row": row["bank_row"],
                "question_id": row["question_id"],
                "repeat": row["repeat"],
                "hold_steps": row["source_hold_steps"],
                "attempt_seed": row["attempt_seed"],
                "ready_state_sha256": "9" * 64,
                "start_step": 0,
                "end_step": 26,
                "finalize_reason": "clip_complete",
                "finalized": True,
                "censored": False,
                "physical_fall": False,
                "guard_reset": False,
                "reached_exact": True,
                "hit": True,
                "returned": True,
                "pos_error_m": 0.01,
                "vel_error_mps": 0.1,
                "normal_error_deg": 0.0,
                "landing_x": 1.0,
                "landing_y": 0.0,
                "net_clear": True,
                "instrumentation": instrumentation,
                "timing_exam_enabled": True,
                "all_attempt_denominator_member": True,
                "eligible": True,
                "planner_infeasible": None,
                "infeasible": None,
                "planner_infeasible_source": "unmeasured_fixed_question_exam_bypasses_planner",
                "deadline_miss": False,
                "deadline_shifted": False,
                "deadline_step": 25,
                "exact_strike_step": 25,
                "initial_state_id": row["initial_state_id"],
                "tts_seconds": 0.5,
                "tts_ticks": 25,
                "time_law_id": row["time_law_id"],
                "expected_feasible": None,
                "feasibility_status": row["feasibility_status"],
                "effective_hold_steps": 0,
                "contact": True,
                "composite": True,
                "safety": {
                    "physical_fall": False,
                    "self_hit": None,
                    "illegal_table_or_net_contact": None,
                    "reset_or_teleport": False,
                    "deadline_shifted": False,
                    "complete": False,
                },
            }
        )
    return {
        "schema": T.ISAAC_SCORECARD_SCHEMA,
        "evaluation_contract_exact": False,
        "inexact_reasons": sorted(T.ISAAC_DIAGNOSTIC_REASONS),
        "simulator": "isaac",
        "protocol": "single",
        "noise_scale": 0.0,
        "schedule": schedule,
        "schedule_sha256": paper["source_schedule"]["semantic_sha256"],
        "hold_semantics": "stand-policy-actions-then-raw-frame0-v1",
        "exam_bank": {
            "path": "/workspace/exam.npz",
            "sha256": paper["source_schedule"]["bank_sha256"],
            "source_family_sha256": paper["source_schedule"]["bank_source_family_sha256"],
            "schema_version": 3,
            "split": "exam",
        },
        "checkpoint": {"path": "/workspace/model.pt", "sha256": checkpoint_sha},
        "training_contract_sha256": hard_sha,
        "termination_contract_id": "termination-v1",
        "ready_state_sha256": "9" * 64,
        "cross_engine_instrumentation": {},
        "nominal_eval_profile": {},
        "sources": _scorecard_source_closure(),
        "timing_exam": {
            "enabled": True,
            "mode": "0.5_second_zero_velocity_frame0_diagnostic",
            "paper_binding": {
                "path": "/workspace/timing.paper.json",
                "file_sha256": paper_file_sha,
                "semantic_sha256": paper["paper_semantic_sha256"],
            },
            "source_schedule_file_sha256": paper["source_schedule"]["file_sha256"],
            "ready_state": {},
            "runtime": {},
            "summary": {"formal_gate_pass": False},
            "all_scheduled_attempts_in_denominator": True,
            "formal_gate_authorized": False,
            "mujoco_retiming_status": "blocked_not_implemented_or_verified_in_this_evaluator",
        },
        "status": "valid",
        "summary": {},
        "attempts": attempts,
    }


def test_tracked_spec_is_exact_paper_only_and_validates_from_cli(capsys):
    file_sha = T.sha256_file(CONFIG)
    spec = T.load_spec(CONFIG, root=ROOT, expected_file_sha256=file_sha)
    assert spec["source_schedule"] == T.EXPECTED_SOURCE_SCHEDULE
    assert spec["paper"]["baseline"] == {
        "human_name": "0.5 second zero-velocity ready-state timing baseline",
        "tts_seconds": 0.5,
        "tts_ticks": 25,
        "initial_state_id": "nominal-frame0-zero-velocity-v1",
        "expected_feasible": None,
        "feasibility_status": "hypothesis_not_certified",
        "scheduled_attempts": 100,
        "per_side": {"forehand": 50, "backhand": 50},
        "source_hold_steps_are_replaced": True,
        "source_attempt_seed_and_question_order_are_preserved": True,
    }
    assert spec["scoring"]["per_side_pass_count"] == 31
    assert spec["execution"]["isaac_diagnostic_evaluator_authorized"] is True
    assert spec["execution"]["isaac_diagnostic_requires_allow_inexact_contract"] is True
    assert spec["execution"]["judge_authorized"] is False
    assert spec["execution"]["real_robot_authorized"] is False
    assert T.main(
        [
            "validate-spec",
            "--spec",
            str(CONFIG),
            "--expected-spec-file-sha256",
            file_sha,
        ]
    ) == 0
    assert '"status": "pass_preregistered_materialization_not_run"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["paper"]["baseline"].__setitem__("tts_ticks", 26), "paper or timing sweep"),
        (lambda value: value["scoring"].__setitem__("per_side_pass_count", 30), "scoring or safety"),
        (
            lambda value: value["scoring"]["denominator"].__setitem__("censoring_allowed", True),
            "scoring or safety",
        ),
        (
            lambda value: value["execution"].__setitem__("judge_authorized", True),
            "execution authorization",
        ),
        (
            lambda value: value["paper"]["time_laws"][0].__setitem__(
                "topp_or_dynamics_certified", True
            ),
            "paper or timing sweep",
        ),
    ],
)
def test_semantic_contract_mutations_fail_even_after_resealing(mutation, message):
    value = _load(CONFIG)
    mutation(value)
    _seal(value, "contract_content_sha256")
    with pytest.raises(T.ContractError, match=message):
        T.validate_spec_document(value, root=ROOT)


def test_duplicate_keys_nonfinite_and_file_sha_fail_closed(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(T.ContractError, match="duplicate JSON key"):
        T.load_json(duplicate, "duplicate")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
    with pytest.raises(T.ContractError, match="non-finite JSON constant"):
        T.load_json(nonfinite, "nonfinite")

    with pytest.raises(T.ContractError, match="spec file SHA mismatch"):
        T.load_spec(CONFIG, root=ROOT, expected_file_sha256="0" * 64)


def test_exact_source_materializes_one_bound_row_per_question_no_replace(tmp_path: Path):
    spec, schedule_path, schedule = _synthetic_spec(tmp_path)
    paper = T.build_paper(
        spec=spec,
        spec_file_sha256="d" * 64,
        source_schedule=schedule,
    )
    assert len(paper["rows"]) == 100
    assert {row["side"] for row in paper["rows"]} == set(T.SIDE_ORDER)
    assert all(row["initial_state_id"] == "nominal-frame0-zero-velocity-v1" for row in paper["rows"])
    assert all(row["tts_seconds"] == 0.5 and row["tts_ticks"] == 25 for row in paper["rows"])
    assert all(row["expected_feasible"] is None for row in paper["rows"])
    assert all(
        row["feasibility_status"] == "hypothesis_not_certified" for row in paper["rows"]
    )
    assert [row["question_id"] for row in paper["rows"]] == [
        item["question_id"] for item in schedule["items"]
    ]
    assert all(row["source_hold_steps_replaced"] is True for row in paper["rows"])
    assert {row["time_law_id"] for row in paper["rows"]} == {
        "v4rg-uniform-phase-forehand-0p5-v1",
        "v4rg-uniform-phase-backhand-0p5-v1",
    }

    output = tmp_path / "paper.json"
    T.write_paper_exclusive(output, paper)
    with pytest.raises(T.ContractError, match="refusing to replace"):
        T.write_paper_exclusive(output, paper)
    assert schedule_path.is_file()


def test_source_schedule_tampering_is_rejected_before_question_materialization(tmp_path: Path):
    spec, schedule_path, schedule = _synthetic_spec(tmp_path)
    tampered = copy.deepcopy(schedule)
    tampered["items"][0]["question_id"] = f"forehand:{'f' * 64}"
    schedule_path.write_bytes(T.canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(T.ContractError, match="byte count mismatch|file SHA mismatch"):
        T.load_source_schedule(schedule_path, source_contract=spec["source_schedule"])

    schedule_path, source = _synthetic_schedule(tmp_path)
    schedule = _load(schedule_path)
    schedule["schedule_sha256"] = "0" * 64
    source["bytes"] = len(T.canonical_json_bytes(schedule) + b"\n")
    path = tmp_path / "bad-semantic.json"
    path.write_bytes(T.canonical_json_bytes(schedule) + b"\n")
    source["file_sha256"] = T.sha256_file(path)
    source["semantic_sha256"] = "0" * 64
    with pytest.raises(T.ContractError, match="declared semantic SHA mismatch"):
        T.load_source_schedule(path, source_contract=source)


def test_result_uses_all_attempts_and_31_per_side_is_the_gate(tmp_path: Path):
    _, paper, paper_path, paper_file_sha = _paper_fixture(tmp_path)
    result = _result(paper, paper_file_sha, successes_per_side=31)
    validated = T.validate_result_document(
        result,
        paper=paper,
        paper_file_sha256=paper_file_sha,
    )
    summary = T.score_result(validated, paper=paper)
    assert summary["denominator_policy"] == "all_scheduled_attempts"
    assert summary["per_side"]["forehand"]["scheduled"] == 50
    assert summary["per_side"]["backhand"]["scheduled"] == 50
    assert summary["per_side"]["forehand"]["composite_successes"] == 31
    assert summary["per_side"]["backhand"]["composite_successes"] == 31
    assert summary["planner_feasibility_observation_complete"] is False
    assert summary["time_laws_dynamics_certified"] is False
    assert summary["formal_gate_pass"] is False
    assert paper_path.is_file()

    below = _result(paper, paper_file_sha, successes_per_side=30)
    below_summary = T.score_result(
        T.validate_result_document(below, paper=paper, paper_file_sha256=paper_file_sha),
        paper=paper,
    )
    assert below_summary["performance_threshold_pass"] is False
    assert below_summary["formal_gate_pass"] is False


def test_safety_is_zero_tolerance_and_inexact_results_stay_diagnostic(tmp_path: Path):
    _, paper, _, paper_file_sha = _paper_fixture(tmp_path)
    unsafe = _result(paper, paper_file_sha)
    unsafe["attempts"][0]["self_hit"] = True
    _seal(unsafe, "result_content_sha256")
    unsafe_summary = T.score_result(
        T.validate_result_document(unsafe, paper=paper, paper_file_sha256=paper_file_sha),
        paper=paper,
    )
    assert unsafe_summary["performance_threshold_pass"] is True
    assert unsafe_summary["safety_pass"] is False
    assert unsafe_summary["formal_gate_pass"] is False

    inexact = _result(paper, paper_file_sha)
    inexact["evaluation_contract_exact"] = False
    _seal(inexact, "result_content_sha256")
    inexact_summary = T.score_result(
        T.validate_result_document(inexact, paper=paper, paper_file_sha256=paper_file_sha),
        paper=paper,
    )
    assert inexact_summary["diagnostic_performance_pass"] is True
    assert inexact_summary["formal_gate_pass"] is False

    unknown = _result(paper, paper_file_sha)
    for attempt in unknown["attempts"]:
        attempt["self_hit"] = None
        attempt["illegal_table_or_net_contact"] = None
    _seal(unknown, "result_content_sha256")
    unknown_summary = T.score_result(
        T.validate_result_document(
            unknown, paper=paper, paper_file_sha256=paper_file_sha
        ),
        paper=paper,
    )
    assert unknown_summary["diagnostic_performance_pass"] is True
    assert unknown_summary["safety_observation_complete"] is False
    assert unknown_summary["safety_pass"] is False
    assert unknown_summary["formal_gate_pass"] is False


def test_missing_rows_wrong_paper_binding_and_invalid_claims_fail_closed(tmp_path: Path):
    _, paper, _, paper_file_sha = _paper_fixture(tmp_path)
    missing = _result(paper, paper_file_sha)
    missing["attempts"].pop()
    _seal(missing, "result_content_sha256")
    with pytest.raises(T.ContractError, match="one row for every scheduled attempt"):
        T.validate_result_document(missing, paper=paper, paper_file_sha256=paper_file_sha)

    wrong_paper = _result(paper, paper_file_sha)
    wrong_paper["paper_file_sha256"] = "9" * 64
    _seal(wrong_paper, "result_content_sha256")
    with pytest.raises(T.ContractError, match="paper file SHA mismatch"):
        T.validate_result_document(wrong_paper, paper=paper, paper_file_sha256=paper_file_sha)

    invalid = _result(paper, paper_file_sha)
    invalid["attempts"][0]["observation_valid"] = False
    _seal(invalid, "result_content_sha256")
    with pytest.raises(T.ContractError, match="invalid observation must not claim"):
        T.validate_result_document(invalid, paper=paper, paper_file_sha256=paper_file_sha)

    wrong_type = _result(paper, paper_file_sha)
    wrong_type["attempts"][1]["schedule_index"] = True
    _seal(wrong_type, "result_content_sha256")
    with pytest.raises(T.ContractError, match="schedule_index must be an integer"):
        T.validate_result_document(
            wrong_type, paper=paper, paper_file_sha256=paper_file_sha
        )

    missing_metric = _result(paper, paper_file_sha)
    missing_metric["attempts"][0]["returned"] = False
    missing_metric["attempts"][0]["position_error_m"] = None
    _seal(missing_metric, "result_content_sha256")
    with pytest.raises(T.ContractError, match="valid observation lacks finite"):
        T.validate_result_document(
            missing_metric, paper=paper, paper_file_sha256=paper_file_sha
        )


def test_isaac_scorecard_converts_to_strict_v2_ledger_without_false_safety(
    tmp_path: Path,
):
    spec, _, schedule = _synthetic_spec(tmp_path)
    paper = T.build_paper(
        spec=spec,
        spec_file_sha256="d" * 64,
        source_schedule=schedule,
    )
    paper_path = tmp_path / "paper.json"
    T.write_paper_exclusive(paper_path, paper)
    paper_file_sha = T.sha256_file(paper_path)
    checkpoint_sha = "1" * 64
    hard_sha = "2" * 64
    raw_scorecard = _isaac_scorecard(
        paper,
        paper_file_sha,
        schedule,
        checkpoint_sha=checkpoint_sha,
        hard_sha=hard_sha,
    )
    scorecard_path = tmp_path / "scorecard.json"
    _write_pretty(scorecard_path, raw_scorecard)
    scorecard_sha = T.sha256_file(scorecard_path)
    validated = T.validate_isaac_timing_scorecard(
        raw_scorecard,
        root=ROOT,
        source_schedule=schedule,
        paper=paper,
        paper_file_sha256=paper_file_sha,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_hard_contract_sha256=hard_sha,
    )
    result = T.build_isaac_result_ledger(
        scorecard=validated,
        scorecard_file_sha256=scorecard_sha,
        paper=paper,
        paper_file_sha256=paper_file_sha,
    )
    assert result["schema_version"] == 2
    assert len(result["attempts"]) == 100
    assert all(attempt["self_hit"] is None for attempt in result["attempts"])
    assert all(
        attempt["illegal_table_or_net_contact"] is None
        for attempt in result["attempts"]
    )
    summary = T.score_result(
        T.validate_result_document(
            result, paper=paper, paper_file_sha256=paper_file_sha
        ),
        paper=paper,
    )
    assert summary["diagnostic_performance_pass"] is True
    assert summary["safety_observation_complete"] is False
    assert summary["formal_gate_pass"] is False

    output = tmp_path / "result.json"

    def validate(path: Path) -> None:
        T.validate_result_document(
            T.load_json(path, "staged result"),
            paper=paper,
            paper_file_sha256=paper_file_sha,
        )

    payload = T.canonical_json_bytes(result) + b"\n"
    T._publish_exclusive_validated(output, payload, validate)
    immutable = output.read_bytes()
    with pytest.raises(T.ContractError, match="refusing to replace"):
        T._publish_exclusive_validated(output, payload, validate)
    assert output.read_bytes() == immutable


def test_isaac_converter_recomputes_raw_signed_face_and_rejects_oriented_false_green(
    tmp_path: Path,
):
    spec, _, schedule = _synthetic_spec(tmp_path)
    paper = T.build_paper(
        spec=spec, spec_file_sha256="d" * 64, source_schedule=schedule
    )
    paper_path = tmp_path / "paper.json"
    T.write_paper_exclusive(paper_path, paper)
    paper_file_sha = T.sha256_file(paper_path)
    scorecard = _isaac_scorecard(
        paper,
        paper_file_sha,
        schedule,
        checkpoint_sha="1" * 64,
        hard_sha="2" * 64,
    )
    # The oriented helper remains +Y, but the physical signed face is flipped to -Y.  Copying the
    # old normal_error_deg=0 would be a false green; the converter must recompute ~180 degrees.
    scorecard["attempts"][0]["instrumentation"] = _instrumentation(
        signed_normal=[0.0, -1.0, 0.0]
    )
    with pytest.raises(T.ContractError, match="raw signed-face error"):
        T.validate_isaac_timing_scorecard(
            scorecard,
            root=ROOT,
            source_schedule=schedule,
            paper=paper,
            paper_file_sha256=paper_file_sha,
            expected_checkpoint_sha256="1" * 64,
            expected_hard_contract_sha256="2" * 64,
        )
