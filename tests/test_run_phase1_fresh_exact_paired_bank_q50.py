from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_phase1_fresh_exact_paired_bank_q50.py"
CAUSAL_RUNNER_PATH = ROOT / "scripts" / "run_phase1_paired_bank_q50.py"
CONFIG_PATH = ROOT / "configs" / "phase1_SZ_seed1_2000_vs_4000_q50_execution_20260711.json"
PREREG_PATH = ROOT / "configs" / "phase1_SZ_seed1_2000_vs_4000_q50_prereg_20260711.json"
SCHEDULE_MODULE_PATH = (
    ROOT / "hope_training" / "whole_body_tracking" / "scripts" / "bank_exam_schedule.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load_module("fresh_exact_paired_q50_under_test", RUNNER_PATH)
S = _load_module("fresh_exact_paired_q50_schedule_under_test", SCHEDULE_MODULE_PATH)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _prereg() -> dict:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _schedule(tmp_path: Path, *, bank_sha256: str):
    ids = tuple(
        tuple(f"{clip + 1:01x}{row:063x}" for row in range(60)) for clip in range(2)
    )
    artifact = S.materialize_balanced_bank_exam_schedule(
        bank_sha256=bank_sha256,
        clip_names=("forehand", "backhand"),
        question_ids=ids,
        per_clip_quota=50,
        schedule_seed=0,
        hold_range=(0, 100),
    )
    path = tmp_path / "shared.schedule.json"
    S.write_schedule_artifact(artifact, path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_frozen_files_are_fresh_exact_formal_and_leave_causal_runner_immutable():
    config = R.load_execution_config(CONFIG_PATH)
    prereg = _prereg()
    assert R.sha256_file(RUNNER_PATH) == config["tools"]["runner_sha256"]
    assert R.sha256_file(CAUSAL_RUNNER_PATH) == config["tools"]["shared_causal_runner"]["sha256"]
    assert R.SHARED_CAUSAL_RUNNER_SHA256 == config["tools"]["shared_causal_runner"]["sha256"]
    assert R.sha256_file(PREREG_PATH) == config["preregistration_sha256"]
    R.validate_preregistration(prereg, config)
    assert config["semantics"] == R.EXPECTED_SEMANTICS
    assert config["schedule"] == R.EXPECTED_SCHEDULE
    assert config["schedule"]["allow_inexact_contract"] is False
    assert prereg["source_trigger"]["observed"] == {
        "model_2000": {
            "attempts": 20,
            "attempts_per_side": 10,
            "aggregate_return_rate": 0.9,
        },
        "model_4000": {
            "attempts": 20,
            "attempts_per_side": 10,
            "aggregate_return_rate": 0.5,
        },
    }
    assert prereg["paper"]["exam_bank"]["sha256"] == (
        "d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096"
    )
    assert prereg["arms"]["model_2000"]["training_contract_sha256"] == (
        prereg["arms"]["model_4000"]["training_contract_sha256"]
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("auto_start", True),
        lambda value: value["semantics"].__setitem__("evaluation_contract_exact", False),
        lambda value: value["semantics"].__setitem__("whole_arm_stop_allowed", True),
        lambda value: value["schedule"].__setitem__("allow_inexact_contract", True),
        lambda value: value["schedule"].__setitem__("same_artifact_for_both_checkpoints", False),
        lambda value: value["selection_policy"].__setitem__("may_promote_whole_arm", True),
    ),
)
def test_execution_config_rejects_relaxation(tmp_path: Path, mutation):
    data = _config()
    mutation(data)
    path = tmp_path / "mutated.json"
    _write_json(path, data)
    with pytest.raises(R.ContractError):
        R.load_execution_config(path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("jobs_started", 1),
        lambda value: value["source_trigger"].__setitem__("trigger_met", False),
        lambda value: value["source_trigger"].__setitem__("stop_or_promote_allowed", True),
        lambda value: value["paper"].__setitem__("allow_inexact_contract_required", True),
        lambda value: value["arms"]["model_2000"].__setitem__("lineage_exact", False),
        lambda value: value["arms"]["model_4000"].__setitem__("zero_joint_friction", False),
        lambda value: value["arms"]["model_4000"].__setitem__(
            "training_contract_sha256", "f" * 64
        ),
        lambda value: value["selection_policy"].__setitem__("may_stop_whole_arm", True),
    ),
)
def test_prereg_rejects_started_inexact_or_repapered_mutation(mutation):
    config = R.load_execution_config(CONFIG_PATH)
    prereg = _prereg()
    mutation(prereg)
    with pytest.raises(R.ContractError):
        R.validate_preregistration(prereg, config)


def test_formal_command_uses_one_shared_json_and_has_no_inexact_escape(tmp_path: Path):
    schedule, _ = _schedule(
        tmp_path, bank_sha256=_prereg()["paper"]["exam_bank"]["sha256"]
    )
    args = R.semantic_judge_args(schedule)
    command = R.build_judge_command(
        judge=Path("/eval/judge.sh"),
        arm=_prereg()["arms"]["model_2000"],
        schedule_path=schedule,
        gpu=2,
    )
    assert args[:8] == [
        "--seed",
        "0",
        "--noise-scales",
        "0.0",
        "--steps",
        "0",
        "--hold-ref",
        "auto",
    ]
    assert "--exam-schedule-json" in args[-1]
    assert all("--allow-inexact-contract" not in value for value in args)
    assert all("--allow-inexact-contract" not in value for value in command)
    assert "--schedule-k" not in command


def _hard_contract() -> dict:
    return {
        "schema_version": 3,
        "motion_kinematics_exact": True,
        "face_command_pairing": "shared_plus_y",
        "joint_friction_coefficients": [0.0] * 31,
        "question_bank": {
            "sha256": "2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700",
            "schema_version": 3,
            "split": "train",
            "source_family_sha256": "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5",
            "exact": True,
        },
    }


def test_hard_contract_requires_exact_motion_zero_plant_and_same_bank_family():
    prereg = _prereg()
    R.validate_hard_contract(_hard_contract(), prereg)
    for mutation in (
        lambda value: value.__setitem__("motion_kinematics_exact", False),
        lambda value: value.__setitem__("face_command_pairing", "legacy_signed_vs_A"),
        lambda value: value["joint_friction_coefficients"].__setitem__(0, 0.01),
        lambda value: value["question_bank"].__setitem__("source_family_sha256", "e" * 64),
    ):
        candidate = _hard_contract()
        mutation(candidate)
        with pytest.raises(R.ContractError):
            R.validate_hard_contract(candidate, prereg)


def test_trigger_sources_are_rehashed_before_prepare(tmp_path: Path):
    artifact = tmp_path / "q10.state.json"
    artifact.write_bytes(b"preserved-q10\n")
    prereg = {
        "source_trigger": {
            "sources": [
                {
                    "role": "state",
                    "path": str(artifact),
                    "sha256": R.sha256_file(artifact),
                },
                {
                    "role": "semantic",
                    "path": "embedded:q10.schedule",
                    "sha256": "a" * 64,
                },
            ]
        }
    }
    R.validate_trigger_sources(prereg)
    artifact.write_bytes(b"changed\n")
    with pytest.raises(R.ContractError, match="changed/missing"):
        R.validate_trigger_sources(prereg)


def _write_result_fixture(
    tmp_path: Path,
    *,
    arm_name: str = "model_2000",
    aggregate_returned: int = 90,
) -> tuple[Path, dict, dict]:
    prereg = _prereg()
    schedule_path, schedule = _schedule(
        tmp_path, bank_sha256=prereg["paper"]["exam_bank"]["sha256"]
    )
    iteration = R.ITERATIONS[arm_name]
    report_parent = tmp_path / "judge"
    report_parent.mkdir(exist_ok=True)
    stamp = "20260711_140500"
    report = report_parent / f"judge_report_model_{iteration}_{stamp}.md"
    report.write_text("# fixture\n", encoding="utf-8")
    exam_dir = report_parent / f"model_{iteration}_{stamp}" / "exam"
    exam_dir.mkdir(parents=True)
    order = [item["question_id"] for item in schedule["items"]]
    ready_sha = "c" * 64
    execution_body = {"schema_version": 1, "contract": "fixture-exact"}
    execution_sha = R.canonical_sha256(execution_body)
    execution_contract = {**execution_body, "sha256": execution_sha}
    runtime_items = []
    for index, item in enumerate(schedule["items"]):
        runtime_items.append(
            {
                **item,
                "question_sequence_index": item["schedule_index"],
                "eligible": True,
                "censored": False,
                "ready_state_mode": "mjcf_named_keyframe:stand:v1",
                "ready_state_sha256": ready_sha,
                "mjcf_sha256": prereg["paper"]["mjcf_sha256"],
                "execution_contract_sha256": execution_sha,
                "physical_fall": False,
                "guard_reset": False,
                "hit": index < aggregate_returned,
                "returned": index < aggregate_returned,
                "finalize_reason": "completed",
            }
        )
    fh_returned = sum(
        item["returned"] for item in runtime_items if item["clip"] == 0
    )
    bh_returned = sum(
        item["returned"] for item in runtime_items if item["clip"] == 1
    )

    def attempt_group(n: int, returned: int) -> dict:
        return {
            "n_attempts": n,
            "n_reached_exact": returned,
            "n_composite": returned,
            "exact_reach_rate": returned / n,
            "composite_rate_per_attempt": returned / n,
            "composite_rate_given_exact": 1.0 if returned else None,
            "finalize_reason_counts": {"completed": n},
        }

    def venue_group(n: int, returned: int) -> dict:
        return {
            "n_attempts": n,
            "n_strikes": returned,
            "contacted": returned,
            "landed_ok": returned,
            "exact_reach_rate_per_attempt": returned / n,
            "contact_rate_per_attempt": returned / n,
            "return_success_rate_per_attempt": returned / n,
        }

    result = {
        "noise_scale": 0.0,
        "evaluation_contract_exact": True,
        "fell": 0,
        "exam_schedule": {
            "question_id_order": order,
            "items": runtime_items,
        },
        "attempts": {
            **attempt_group(100, aggregate_returned),
            "per_clip": {
                "forehand": attempt_group(50, fh_returned),
                "backhand": attempt_group(50, bh_returned),
            },
        },
        "venue": {
            "all": venue_group(100, aggregate_returned),
            "forehand": venue_group(50, fh_returned),
            "backhand": venue_group(50, bh_returned),
        },
    }
    summary = {
        "schema_version": 3,
        "evaluation_contract_exact": True,
        "mjcf_sha256": prereg["paper"]["mjcf_sha256"],
        "execution_contract": execution_contract,
        "execution_contract_sha256": execution_sha,
        "ready_state_sha256": ready_sha,
        "joint_velocity_limit_diagnostics": {
            "hit_count": 0,
            "peak_abs_velocity_over_limit": 0.0,
            "proxy_clamp_applied": False,
        },
        "arguments": {
            "allow_inexact_contract": False,
            "target_source": "bank",
            "seed": 0,
            "noise_scales": [0.0],
            "qdes_clamp": True,
            "hold_ref": "auto",
            "ready_state": "auto",
            "exam_continuity_diagnostic": False,
            "exam_schedule_k": None,
            "exam_schedule_json": str(schedule_path),
        },
        "input_artifacts": {
            "exam_bank": {"sha256": prereg["paper"]["exam_bank"]["sha256"]},
            "exam_schedule_artifact": {
                "sha256": R.sha256_file(schedule_path),
                "schedule_sha256": schedule["schedule_sha256"],
                "schema_version": 3,
            },
        },
        "exam_schedule": {
            "sha256": schedule["schedule_sha256"],
            "bank_sha256": prereg["paper"]["exam_bank"]["sha256"],
            "seed": 0,
            "size": 100,
            "one_question_reset": True,
            "ready_state_mode": "mjcf_named_keyframe:stand:v1",
            "shared_artifact": schedule,
            "items": [
                {**item, "question_sequence_index": item["schedule_index"]}
                for item in schedule["items"]
            ],
        },
        "results": [result],
    }
    summary_path = exam_dir / "mujoco_sim2sim_summary.json"
    _write_json(summary_path, summary)
    attempts_path = exam_dir / "mujoco_sim2sim_attempts.csv"
    fieldnames = (
        "schedule_index",
        "clip_name",
        "bank_row",
        "question_id",
        "repeat",
        "hold_steps",
        "attempt_seed",
        "schedule_sha256",
        "question_sequence_index",
        "ready_state_mode",
        "ready_state_sha256",
        "mjcf_sha256",
        "execution_contract_sha256",
        "eligible",
        "censored",
        "physical_fall",
        "guard_reset",
        "hit",
        "returned",
        "reached_exact",
        "exact_composite",
        "finalize_reason",
    )
    with attempts_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item, runtime in zip(schedule["items"], runtime_items):
            writer.writerow(
                {
                    "schedule_index": item["schedule_index"],
                    "clip_name": "forehand" if item["clip"] == 0 else "backhand",
                    "bank_row": item["bank_row"],
                    "question_id": item["question_id"],
                    "repeat": item["repeat"],
                    "hold_steps": item["hold_steps"],
                    "attempt_seed": item["attempt_seed"],
                    "schedule_sha256": schedule["schedule_sha256"],
                    "question_sequence_index": item["schedule_index"],
                    "ready_state_mode": "mjcf_named_keyframe:stand:v1",
                    "ready_state_sha256": ready_sha,
                    "mjcf_sha256": prereg["paper"]["mjcf_sha256"],
                    "execution_contract_sha256": execution_sha,
                    "eligible": runtime["eligible"],
                    "censored": runtime["censored"],
                    "physical_fall": runtime["physical_fall"],
                    "guard_reset": runtime["guard_reset"],
                    "hit": runtime["hit"],
                    "returned": runtime["returned"],
                    "reached_exact": runtime["hit"],
                    "exact_composite": runtime["hit"],
                    "finalize_reason": runtime["finalize_reason"],
                }
            )
    contract = {
        "shared_schedule": {
            "path": str(schedule_path),
            "file_sha256": R.sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "question_id_order": order,
        }
    }
    return report, contract, summary


def _validate_fixture(report: Path, contract: dict, *, arm_name: str = "model_2000"):
    prereg = _prereg()
    arm = copy.deepcopy(prereg["arms"][arm_name])
    arm["checkpoint_path"] = str(report.parent.parent / f"model_{R.ITERATIONS[arm_name]}.pt")
    return R.validate_exam_result(
        report=report,
        arm_name=arm_name,
        arm=arm,
        prereg=prereg,
        runtime_contract=contract,
    )


def test_result_validator_requires_exact_raw_k100_and_finite_full_denominators(tmp_path: Path):
    report, contract, _ = _write_result_fixture(tmp_path)
    result = _validate_fixture(report, contract)
    assert result["evaluation_contract_exact"] is True
    assert result["formal_target"] is True
    assert result["fresh_lineage"] is True
    assert result["denominators"] == {"aggregate": 100, "forehand": 50, "backhand": 50}
    assert result["returned_counts"] == {
        "aggregate": 90,
        "forehand": 45,
        "backhand": 45,
        "physical_falls": 0,
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda summary: summary.__setitem__("evaluation_contract_exact", False),
        lambda summary: summary["arguments"].__setitem__("allow_inexact_contract", True),
        lambda summary: summary["joint_velocity_limit_diagnostics"].__setitem__("hit_count", 1),
        lambda summary: summary["results"][0]["venue"]["all"].__setitem__("n_attempts", 99),
        lambda summary: summary["results"][0]["venue"]["all"].__setitem__(
            "return_success_rate_per_attempt", None
        ),
        lambda summary: summary["results"][0]["exam_schedule"]["items"][0].__setitem__(
            "censored", True
        ),
    ),
)
def test_result_validator_rejects_inexact_censored_nonfinite_or_denominator_relaxation(
    tmp_path: Path, mutate
):
    report, contract, summary = _write_result_fixture(tmp_path)
    mutate(summary)
    summary_path = report.parent / "model_2000_20260711_140500" / "exam" / (
        "mujoco_sim2sim_summary.json"
    )
    _write_json(summary_path, summary)
    with pytest.raises(R.ContractError):
        _validate_fixture(report, contract)


def test_result_validator_rejects_raw_csv_censor_or_summary_disagreement(tmp_path: Path):
    report, contract, _ = _write_result_fixture(tmp_path)
    attempts_path = report.parent / "model_2000_20260711_140500" / "exam" / (
        "mujoco_sim2sim_attempts.csv"
    )
    rows = list(csv.DictReader(attempts_path.open(newline="", encoding="utf-8")))
    rows[0]["censored"] = "True"
    with attempts_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(R.ContractError, match="censored|disagreement"):
        _validate_fixture(report, contract)


def _selection_result(aggregate: int, fh: int, bh: int, falls: int, sha: str) -> dict:
    return {
        "checkpoint_sha256": sha,
        "returned_counts": {
            "aggregate": aggregate,
            "forehand": fh,
            "backhand": bh,
            "physical_falls": falls,
        },
    }


def test_selection_rule_is_preregistered_deterministic_and_never_stops_or_promotes_arm():
    results = {
        "model_2000": _selection_result(80, 45, 35, 1, "2" * 64),
        "model_4000": _selection_result(80, 40, 40, 8, "4" * 64),
    }
    selected, ledger = R.select_checkpoint(results)
    assert selected == "model_4000"  # min-side tie break precedes falls
    assert ledger["whole_arm_action"] == "continue_unmodified"
    assert ledger["whole_arm_stop_allowed"] is False
    assert ledger["whole_arm_promote_allowed"] is False

    tied = {
        "model_2000": _selection_result(80, 40, 40, 1, "2" * 64),
        "model_4000": _selection_result(80, 40, 40, 1, "4" * 64),
    }
    assert R.select_checkpoint(tied)[0] == "model_2000"


def test_prepare_is_no_auto_start_and_no_clobber(tmp_path: Path, monkeypatch):
    config = _config()
    prereg = _prereg()
    config_path = tmp_path / "execution.json"
    prereg_path = tmp_path / "prereg.json"
    state_dir = tmp_path / "state"
    config["runtime"]["state_dir"] = str(state_dir)
    config["runtime"]["checkpoint_python"] = sys.executable
    _write_json(config_path, config)
    _write_json(prereg_path, prereg)
    schedule_module = ROOT / "hope_training" / "whole_body_tracking" / "scripts" / (
        "materialize_bank_exam_schedule.py"
    )
    monkeypatch.setattr(
        R,
        "validate_runtime_inputs",
        lambda *_: (
            ROOT,
            ROOT,
            {"materialize_schedule": schedule_module},
            {"model_2000": {"iter": 2000}, "model_4000": {"iter": 4000}},
        ),
    )

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        _, source = _schedule(tmp_path, bank_sha256=prereg["paper"]["exam_bank"]["sha256"])
        output.write_bytes(R.canonical_bytes(source) + b"\n")

        class Completed:
            returncode = 0
            stdout = "fixture"

        return Completed()

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    monkeypatch.setattr(
        R.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("prepare must not start a judge"),
    )
    assert R.prepare(config_path, config, prereg_path, prereg) == 0
    runtime = json.loads((state_dir / config["runtime"]["runtime_contract_filename"]).read_text())
    assert runtime["status"] == "prepared_not_started"
    assert runtime["auto_start"] is False
    assert runtime["jobs_started"] == 0
    assert runtime["evaluation_contract_exact"] is True
    assert "--allow-inexact-contract" not in " ".join(runtime["semantic_judge_args"])
    with pytest.raises(R.ContractError, match="no-clobber"):
        R.prepare(config_path, config, prereg_path, prereg)
