from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_phase1_fresh_exact_paired_isaac_q50.py"
CONFIG_PATH = (
    ROOT / "configs" / "phase1_SZ_seed1_2000_vs_4000_q50_isaac_companion_20260711.json"
)
RESULT_PATH = (
    ROOT / "configs" / "phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json"
)
PREREG_PATH = ROOT / "configs" / "phase1_SZ_seed1_2000_vs_4000_q50_prereg_20260711.json"
FRESH_RUNNER_PATH = ROOT / "scripts" / "run_phase1_fresh_exact_paired_bank_q50.py"
SHARED_MUJOCO_PATH = ROOT / "scripts" / "run_phase1_paired_bank_q50.py"
ISAAC_UTILITY_PATH = ROOT / "scripts" / "run_phase1_paired_isaac_q50.py"
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


R = _load_module("fresh_exact_paired_isaac_under_test", RUNNER_PATH)
S = _load_module("fresh_exact_paired_isaac_schedule_under_test", SCHEDULE_MODULE_PATH)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _prereg() -> dict:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _schedule(tmp_path: Path, bank_sha: str):
    ids = tuple(
        tuple(f"{clip + 1:01x}{row:063x}" for row in range(60)) for clip in range(2)
    )
    artifact = S.materialize_balanced_bank_exam_schedule(
        bank_sha256=bank_sha,
        clip_names=("forehand", "backhand"),
        question_ids=ids,
        per_clip_quota=50,
        schedule_seed=0,
        hold_range=(0, 100),
    )
    path = tmp_path / "shared_clean_k100.schedule.json"
    S.write_schedule_artifact(artifact, path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_static_config_binds_exact_helpers_paper_and_non_robot_semantics():
    config = R.load_config(CONFIG_PATH)
    assert R.sha256_file(RUNNER_PATH) == config["tools"]["runner_sha256"]
    assert R.sha256_file(FRESH_RUNNER_PATH) == config["tools"]["fresh_mujoco_runner"]["sha256"]
    assert R.sha256_file(SHARED_MUJOCO_PATH) == config["tools"]["shared_mujoco_runner"]["sha256"]
    assert R.sha256_file(ISAAC_UTILITY_PATH) == config["tools"]["isaac_utility_runner"]["sha256"]
    assert config["paper"]["schedule_file_sha256"] == (
        "66e89986a2b726d529179fcb4c745625ebed0380d59664caceefc55e86071cb3"
    )
    assert config["paper"]["schedule_semantic_sha256"] == (
        "7dc6af822fb4130b8c324843f179d77f882d1326306bb19802b00f94447dff3e"
    )
    assert config["paper"]["allow_inexact_contract"] is False
    assert config["semantics"] == R.EXPECTED_SEMANTICS
    assert config["semantics"]["whole_arm_stop_allowed"] is False
    assert config["semantics"]["whole_arm_promote_allowed"] is False
    assert config["semantics"]["deploy_gate"] is False
    assert config["semantics"]["real_robot_authorized"] is False


def test_accepted_pair_records_exact_cross_engine_tie_without_promotion():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["status"] == "complete_fresh_exact_same_paper_isaac_pair"
    assert result["accepted_execution"]["runner"]["sha256"] == R.sha256_file(RUNNER_PATH)
    assert result["accepted_execution"]["config"]["sha256"] == R.sha256_file(CONFIG_PATH)
    assert result["semantics"] == R.EXPECTED_SEMANTICS
    assert result["immutable_schedule"]["file_sha256"] == (
        "66e89986a2b726d529179fcb4c745625ebed0380d59664caceefc55e86071cb3"
    )
    assert result["immutable_schedule"]["semantic_sha256"] == (
        "7dc6af822fb4130b8c324843f179d77f882d1326306bb19802b00f94447dff3e"
    )
    assert result["immutable_schedule"]["censored_attempts"] == 0
    assert result["mujoco_binding"]["model_2000_minus_model_4000_aggregate_return_rate"] == 0.33
    for name in R.ARM_ORDER:
        arm = result["isaac_arms"][name]
        assert arm["training_contract_sha256"] == (
            "3a3b3d956e19d47f7e6f0a157159dc96c8f09d8345c436a776c8c7e99c0b9972"
        )
        assert arm["summary"] == {
            "forehand_return": 0.98,
            "backhand_return": 1.0,
            "aggregate_return": 0.99,
            "exact_reach": 0.99,
            "hits": 99,
            "guard_resets": 1,
            "physical_falls": 0,
        }
    assert result["paired_delta_model_2000_minus_model_4000"]["aggregate_return"] == 0.0
    judgment = result["cross_engine_judgment"]
    assert judgment["status"] == "ranking_not_reproduced_no_cross_engine_checkpoint_gate"
    assert judgment["whole_arm_action"] == "continue_unmodified"
    assert judgment["formal_or_deployment_promotion"] is False
    assert judgment["deploy_gate"] is False
    assert judgment["real_robot_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("auto_start", True),
        lambda value: value["semantics"].__setitem__("evaluation_contract_exact", False),
        lambda value: value["semantics"].__setitem__("fresh_lineage", False),
        lambda value: value["paper"].__setitem__("allow_inexact_contract", True),
        lambda value: value["paper"].__setitem__("schedule_k", 20),
        lambda value: value["command"].__setitem__("max_parallel", 2),
        lambda value: value["selection_policy"].__setitem__("may_stop_whole_arm", True),
        lambda value: value["tools"]["fresh_mujoco_runner"].__setitem__("sha256", "0" * 64),
    ),
)
def test_config_rejects_inexact_repaper_autostart_or_governance_drift(tmp_path: Path, mutation):
    config = _config()
    mutation(config)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(R.ContractError):
        R.load_config(path)


def _runtime(tmp_path: Path, schedule_path: Path, schedule: dict) -> dict:
    prereg = _prereg()
    state = tmp_path / "isaac_state"
    return {
        "exam_bank": prereg["paper"]["exam_bank"],
        "shared_schedule": {
            "path": str(schedule_path),
            "file_sha256": R.sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "question_id_order": [item["question_id"] for item in schedule["items"]],
        },
        "output_stem": "isaac_clean_k100",
        "arms": {
            name: {
                "run_name": prereg["arms"][name]["run_name"],
                "checkpoint_iteration": R.ITERATIONS[name],
                "run_dir": str(Path(prereg["arms"][name]["checkpoint_path"]).parent),
                "checkpoint_path": prereg["arms"][name]["checkpoint_path"],
                "checkpoint_sha256": prereg["arms"][name]["checkpoint_sha256"],
                "training_contract_sha256": prereg["arms"][name]["training_contract_sha256"],
                "output_dir": str(state / name),
                "output_json": str(state / name / "isaac_clean_k100.json"),
                "output_csv": str(state / name / "isaac_clean_k100.csv"),
            }
            for name in R.ARM_ORDER
        },
    }


def test_command_reuses_mujoco_schedule_and_omits_inexact_escape(tmp_path: Path):
    config = _config()
    prereg = _prereg()
    schedule_path, schedule = _schedule(tmp_path, prereg["paper"]["exam_bank"]["sha256"])
    runtime = _runtime(tmp_path, schedule_path, schedule)
    command = R.build_command(
        config=config,
        runtime=runtime,
        tools={"isaac_evaluator": Path("/eval/isaac_bank_exam.py")},
        arm_name="model_2000",
        gpu=2,
    )
    assert "device=cuda:2" in command
    assert f"+schedule_json={schedule_path}" in command
    assert "+per_clip_quota=50" in command
    assert "+schedule_seed=0" in command
    assert "+noise_scale=0.0" in command
    assert not any("allow_inexact_contract" in value for value in command)
    assert not any("materialize" in value for value in command)


def _scorecard(config: dict, runtime: dict, schedule: dict, arm_name: str) -> dict:
    ready = "d" * 64
    rows = []
    for item in schedule["items"]:
        rows.append(
            {
                "schedule_index": item["schedule_index"],
                "env_id": item["schedule_index"],
                "clip": item["clip"],
                "side": "forehand" if item["clip"] == 0 else "backhand",
                "bank_row": item["bank_row"],
                "question_id": item["question_id"],
                "repeat": item["repeat"],
                "hold_steps": item["hold_steps"],
                "attempt_seed": item["attempt_seed"],
                "ready_state_sha256": ready,
                "start_step": 0,
                "end_step": item["hold_steps"] + 100,
                "finalize_reason": "clip_complete",
                "finalized": True,
                "censored": False,
                "physical_fall": False,
                "guard_reset": False,
                "reached_exact": True,
                "hit": True,
                "returned": True,
                "pos_error_m": 0.01,
                "vel_error_mps": 0.2,
                "normal_error_deg": 5.0,
                "landing_x": 2.5,
                "landing_y": 0.0,
                "net_clear": True,
            }
        )
    tools = config["tools"]["evaluation"]
    return {
        "schema": R.SCORECARD_SCHEMA,
        "status": "valid",
        "evaluation_contract_exact": True,
        "inexact_reasons": [],
        "simulator": "isaac",
        "protocol": "single",
        "noise_scale": 0.0,
        "schedule": schedule,
        "schedule_sha256": schedule["schedule_sha256"],
        "hold_semantics": R.HOLD_SEMANTICS,
        "exam_bank": {
            "path": runtime["exam_bank"]["path"],
            "sha256": runtime["exam_bank"]["sha256"],
            "source_family_sha256": runtime["exam_bank"]["source_family_sha256"],
            "schema_version": 3,
            "split": "exam",
        },
        "checkpoint": {
            "path": runtime["arms"][arm_name]["checkpoint_path"],
            "sha256": runtime["arms"][arm_name]["checkpoint_sha256"],
        },
        "training_contract_sha256": runtime["arms"][arm_name]["training_contract_sha256"],
        "termination_contract_id": "fixture-contract",
        "ready_state_sha256": ready,
        "nominal_eval_profile": {"sha256": "f" * 64},
        "sources": {
            "git_head": config["checkouts"]["evaluation"]["commit"],
            "evaluator_sha256": tools["isaac_evaluator"]["sha256"],
            "adapter_sha256": tools["isaac_adapter"]["sha256"],
            "schedule_module_sha256": tools["schedule_module"]["sha256"],
            "isaac_scorer_sha256": tools["isaac_scorer"]["sha256"],
            "ball_physics_yaml_sha256": tools["ball_physics_yaml"]["sha256"],
        },
        "summary": R.summarize_attempts(rows),
        "attempts": rows,
    }


def _write_scorecard(tmp_path: Path, document: dict) -> tuple[Path, Path]:
    output = tmp_path / "isaac_clean_k100.json"
    ledger = tmp_path / "isaac_clean_k100.csv"
    output.write_text(json.dumps(document), encoding="utf-8")
    columns = (
        "schedule_index",
        "env_id",
        "clip",
        "side",
        "bank_row",
        "question_id",
        "repeat",
        "hold_steps",
        "attempt_seed",
        "ready_state_sha256",
        "start_step",
        "end_step",
        "finalize_reason",
        "finalized",
        "censored",
        "physical_fall",
        "guard_reset",
        "reached_exact",
        "hit",
        "returned",
        "pos_error_m",
        "vel_error_mps",
        "normal_error_deg",
        "landing_x",
        "landing_y",
        "net_clear",
    )
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(document["attempts"])
    return output, ledger


def test_scorecard_requires_exact_fresh_contract_and_uncensored_same_paper(tmp_path: Path):
    config = _config()
    prereg = _prereg()
    schedule_path, schedule = _schedule(tmp_path, prereg["paper"]["exam_bank"]["sha256"])
    runtime = _runtime(tmp_path, schedule_path, schedule)
    document = _scorecard(config, runtime, schedule, "model_2000")
    output, ledger = _write_scorecard(tmp_path, document)
    result = R.validate_scorecard(
        json_path=output,
        csv_path=ledger,
        arm_name="model_2000",
        config=config,
        runtime=runtime,
    )
    assert result["evaluation_contract_exact"] is True
    assert result["fresh_lineage"] is True
    assert result["formal_target"] is True
    assert result["summary"]["return_rate"] == 1.0
    assert result["returned_counts"] == {
        "aggregate": 100,
        "forehand": 50,
        "backhand": 50,
        "physical_falls": 0,
    }

    document["evaluation_contract_exact"] = False
    document["inexact_reasons"] = ["training/checkpoint contract: fixture"]
    document["training_contract_sha256"] = None
    output, ledger = _write_scorecard(tmp_path, document)
    with pytest.raises(R.ContractError, match="exact contract"):
        R.validate_scorecard(
            json_path=output,
            csv_path=ledger,
            arm_name="model_2000",
            config=config,
            runtime=runtime,
        )

    document = _scorecard(config, runtime, schedule, "model_2000")
    document["attempts"][7]["censored"] = True
    document["summary"] = R.summarize_attempts(document["attempts"])
    output, ledger = _write_scorecard(tmp_path, document)
    with pytest.raises(R.ContractError, match="censored|malformed"):
        R.validate_scorecard(
            json_path=output,
            csv_path=ledger,
            arm_name="model_2000",
            config=config,
            runtime=runtime,
        )


def test_isaac_selection_is_pair_only_and_keeps_whole_arm_running():
    base = {
        "checkpoint_sha256": "a" * 64,
        "returned_counts": {
            "aggregate": 83,
            "forehand": 33,
            "backhand": 50,
            "physical_falls": 0,
        },
    }
    later = {
        "checkpoint_sha256": "b" * 64,
        "returned_counts": {
            "aggregate": 50,
            "forehand": 0,
            "backhand": 50,
            "physical_falls": 0,
        },
    }
    selected, selection = R.select_checkpoint({"model_2000": base, "model_4000": later})
    assert selected == "model_2000"
    assert selection["scope"] == "choose_only_between_model_2000_and_model_4000"
    assert selection["whole_arm_action"] == "continue_unmodified"
    assert selection["whole_arm_stop_allowed"] is False
    assert selection["whole_arm_promote_allowed"] is False
    assert selection["deploy_gate"] is False
    assert selection["real_robot_authorized"] is False
