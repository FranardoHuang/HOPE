from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_phase1_paired_isaac_q50.py"
CONFIG_PATH = ROOT / "configs" / "phase1_M3_terminal_q50_isaac_companion_20260711.json"
PREREG_PATH = ROOT / "configs" / "phase1_M3_terminal_q50_prereg_20260711.json"
MUJOCO_RUNNER_PATH = ROOT / "scripts" / "run_phase1_paired_bank_q50.py"
MUJOCO_CONFIG_PATH = ROOT / "configs" / "phase1_M3_terminal_q50_execution_20260711.json"
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


R = _load_module("paired_isaac_q50_under_test", RUNNER_PATH)
M = _load_module("paired_mujoco_q50_for_isaac_test", MUJOCO_RUNNER_PATH)
S = _load_module("paired_isaac_schedule_under_test", SCHEDULE_MODULE_PATH)


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


def test_static_companion_contract_is_bound_to_final_mujoco_runner_and_prereg():
    config = R.load_config(CONFIG_PATH)
    assert R.sha256_file(PREREG_PATH) == config["preregistration_sha256"]
    assert R.sha256_file(MUJOCO_RUNNER_PATH) == config["mujoco_binding"]["runner_sha256"]
    assert R.sha256_file(MUJOCO_CONFIG_PATH) == (
        config["mujoco_binding"]["execution_config_sha256"]
    )
    loaded, prereg, mujoco, _ = R.validate_static_bindings(
        config_path=CONFIG_PATH,
        expected_config_sha=R.sha256_file(CONFIG_PATH),
        prereg_path=PREREG_PATH,
        mujoco_runner_path=MUJOCO_RUNNER_PATH,
        mujoco_config_path=MUJOCO_CONFIG_PATH,
    )
    assert loaded == config
    assert prereg["jobs_started"] == 0
    assert mujoco["contract_id"] == config["mujoco_binding"]["execution_contract_id"]


def test_arm_maps_accept_canonical_sorted_json_but_not_missing_or_extra_keys():
    canonical = json.loads(
        json.dumps({"M3_old": {"value": 1}, "M3_S1": {"value": 2}}, sort_keys=True)
    )
    assert list(canonical) == ["M3_S1", "M3_old"]
    assert R.require_arm_map(canonical, "fixture") is canonical
    with pytest.raises(R.ContractError):
        R.require_arm_map({"M3_old": {}}, "fixture")
    with pytest.raises(R.ContractError):
        R.require_arm_map({**canonical, "M3_extra": {}}, "fixture")


def test_direct_python_environment_uses_pinned_source_first_path_and_drops_ambient(
    tmp_path: Path, monkeypatch,
):
    eval_root = tmp_path / "eval"
    wbt = eval_root / "hope_training" / "whole_body_tracking"
    source = wbt / "source" / "whole_body_tracking"
    scripts = wbt / "scripts"
    source.mkdir(parents=True)
    scripts.mkdir()
    setup = wbt / "setup_train_env.sh"
    setup.write_text(
        f'export HOPE_WBT_PYTHONPATH="{source}:/absolute/isaaclab"\n',
        encoding="utf-8",
    )
    evaluator = scripts / "isaac_bank_exam.py"
    evaluator.write_text("# fixture\n", encoding="utf-8")
    config = _config()
    config["checkouts"]["evaluation"]["path"] = str(eval_root)
    tools = {"isaac_evaluator": evaluator}
    assert R.isaac_workdir(config, tools) == wbt.resolve()

    monkeypatch.setenv("PYTHONPATH", "/ambient/must-not-survive")
    env = R.setup_environment(setup, eval_root=eval_root)
    assert env["HOPE_WBT_PYTHONPATH"] == f"{source}:/absolute/isaaclab"
    assert env["PYTHONPATH"] == env["HOPE_WBT_PYTHONPATH"]
    assert "/ambient/must-not-survive" not in env["PYTHONPATH"]
    assert env["PYTHONPATH"].split(":", 1)[0] == str(source)

    setup.write_text(
        'export HOPE_WBT_PYTHONPATH="/wrong/source:/absolute/isaaclab"\n',
        encoding="utf-8",
    )
    with pytest.raises(R.ContractError, match="source first"):
        R.setup_environment(setup, eval_root=eval_root)


def test_rc_zero_without_evaluator_handshake_remains_a_failure(tmp_path: Path):
    output_json = tmp_path / "score.json"
    output_csv = tmp_path / "score.csv"
    hydra_rc_zero_log = (
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'whole_body_tracking'\n"
    )
    with pytest.raises(R.ContractError, match="handshake"):
        R.require_success_handshake(hydra_rc_zero_log, output_json, output_csv)
    success = (
        f"[isaac-bank-exam] JSON {output_json}\n"
        f"[isaac-bank-exam] CSV  {output_csv}\n"
    )
    R.require_success_handshake(success, output_json, output_csv)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["semantics"].__setitem__("evaluation_contract_exact", True),
        lambda value: value["semantics"].__setitem__("formal_target", True),
        lambda value: value["paper"].__setitem__("schedule_k", 20),
        lambda value: value["paper"].__setitem__("same_schedule_file_as_mujoco", False),
        lambda value: value["command"].__setitem__("max_parallel", 2),
        lambda value: value.__setitem__("auto_start", True),
    ),
)
def test_config_rejects_exact_formal_repaper_or_autostart(tmp_path: Path, mutation):
    config = _config()
    mutation(config)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(R.ContractError):
        R.load_config(path)


def _runtime_for_score(tmp_path: Path, schedule_path: Path, schedule: dict) -> dict:
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
        "arms": {
            name: {
                "run_name": prereg["arms"][name]["run_name"],
                "checkpoint_path": prereg["arms"][name]["checkpoint_path"],
                "checkpoint_sha256": prereg["arms"][name]["checkpoint_sha256"],
                "training_contract_sha256": prereg["arms"][name]["training_contract_sha256"],
                "output_json": str(state / name / "isaac_clean_k100.json"),
                "output_csv": str(state / name / "isaac_clean_k100.csv"),
            }
            for name in R.ARM_ORDER
        },
    }


def _scorecard_document(config: dict, prereg: dict, runtime: dict, schedule: dict, arm: str):
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
    # Mirror the accepted real M3-old shape: one guard reset, no physical fall, 99 exact returns.
    rows[42].update(
        finalize_reason="guard_reset",
        guard_reset=True,
        reached_exact=False,
        hit=False,
        returned=False,
        net_clear=False,
        pos_error_m=None,
        vel_error_mps=None,
        normal_error_deg=None,
        landing_x=None,
        landing_y=None,
    )
    tool = config["tools"]["evaluation"]
    return {
        "schema": R.SCORECARD_SCHEMA,
        "status": "valid",
        "evaluation_contract_exact": False,
        "inexact_reasons": [
            "training/checkpoint contract: checkpoint lineage_exact=0",
            "saved runtime motions use legacy link-origin velocities",
        ],
        "simulator": "isaac",
        "protocol": "single",
        "noise_scale": 0.0,
        "schedule": schedule,
        "schedule_sha256": schedule["schedule_sha256"],
        "hold_semantics": R.HOLD_SEMANTICS,
        "exam_bank": {
            "path": runtime["exam_bank"]["path"],
            "sha256": runtime["exam_bank"]["sha256"],
            "source_family_sha256": "e" * 64,
            "schema_version": 3,
            "split": "exam",
        },
        "checkpoint": {
            "path": runtime["arms"][arm]["checkpoint_path"],
            "sha256": runtime["arms"][arm]["checkpoint_sha256"],
        },
        "training_contract_sha256": None,
        "termination_contract_id": "fixture-contract",
        "ready_state_sha256": ready,
        "nominal_eval_profile": {"sha256": "f" * 64},
        "sources": {
            "git_head": config["checkouts"]["evaluation"]["commit"],
            "evaluator_sha256": tool["isaac_evaluator"]["sha256"],
            "adapter_sha256": tool["isaac_adapter"]["sha256"],
            "schedule_module_sha256": tool["schedule_module"]["sha256"],
            "isaac_scorer_sha256": tool["isaac_scorer"]["sha256"],
            "ball_physics_yaml_sha256": tool["ball_physics_yaml"]["sha256"],
        },
        "summary": R.summarize_scorecard_attempts(rows),
        "attempts": rows,
    }


def _write_scorecard(tmp_path: Path, document: dict) -> tuple[Path, Path]:
    output = tmp_path / "isaac_clean_k100.json"
    ledger = tmp_path / "isaac_clean_k100.csv"
    output.write_text(json.dumps(document), encoding="utf-8")
    fields = (
        "schedule_index",
        "env_id",
        "clip",
        "bank_row",
        "question_id",
        "repeat",
        "hold_steps",
        "attempt_seed",
        "ready_state_sha256",
        "finalized",
        "censored",
    )
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(document["attempts"])
    return output, ledger


def test_scorecard_requires_same_k100_schedule_and_uncensored_all_attempt_ledger(tmp_path: Path):
    config = _config()
    prereg = _prereg()
    schedule_path, schedule = _schedule(tmp_path, prereg["paper"]["exam_bank"]["sha256"])
    runtime = _runtime_for_score(tmp_path, schedule_path, schedule)
    document = _scorecard_document(config, prereg, runtime, schedule, "M3_old")
    output, ledger = _write_scorecard(tmp_path, document)
    result = R.validate_scorecard(
        json_path=output,
        csv_path=ledger,
        arm_name="M3_old",
        config=config,
        prereg=prereg,
        runtime=runtime,
        tools={},
    )
    assert result["evaluation_contract_exact"] is False
    assert result["causal"] is True
    assert result["formal_target"] is False
    assert result["training_contract_sha256"] == (
        prereg["arms"]["M3_old"]["training_contract_sha256"]
    )
    assert result["scorecard_training_contract_sha256"] is None
    assert result["summary"]["return_rate"] == 0.99
    assert result["summary"]["n_guard_reset"] == 1
    assert len(result["question_id_order"]) == 100

    document["training_contract_sha256"] = runtime["arms"]["M3_old"][
        "training_contract_sha256"
    ]
    output, ledger = _write_scorecard(tmp_path, document)
    with pytest.raises(R.ContractError, match="header/summary"):
        R.validate_scorecard(
            json_path=output,
            csv_path=ledger,
            arm_name="M3_old",
            config=config,
            prereg=prereg,
            runtime=runtime,
            tools={},
        )

    document["training_contract_sha256"] = None
    document["inexact_reasons"] = ["legacy motion only"]
    output, ledger = _write_scorecard(tmp_path, document)
    with pytest.raises(R.ContractError, match="header/summary"):
        R.validate_scorecard(
            json_path=output,
            csv_path=ledger,
            arm_name="M3_old",
            config=config,
            prereg=prereg,
            runtime=runtime,
            tools={},
        )

    document["inexact_reasons"] = [
        "training/checkpoint contract: checkpoint lineage_exact=0",
        "saved runtime motions use legacy link-origin velocities",
    ]
    document["attempts"][7]["censored"] = True
    output, ledger = _write_scorecard(tmp_path, document)
    with pytest.raises(R.ContractError, match="header/summary|censored|malformed"):
        R.validate_scorecard(
            json_path=output,
            csv_path=ledger,
            arm_name="M3_old",
            config=config,
            prereg=prereg,
            runtime=runtime,
            tools={},
        )


def test_command_supplies_shared_schedule_and_never_requests_new_materialization(tmp_path: Path):
    config = _config()
    prereg = _prereg()
    schedule_path, schedule = _schedule(tmp_path, prereg["paper"]["exam_bank"]["sha256"])
    runtime = _runtime_for_score(tmp_path, schedule_path, schedule)
    runtime["arms"]["M3_old"].update(
        run_dir=str(Path(runtime["arms"]["M3_old"]["checkpoint_path"]).parent),
        output_dir=str(tmp_path / "M3_old"),
    )
    runtime["output_stem"] = "isaac_clean_k100"
    command = R.build_command(
        config=config,
        runtime=runtime,
        tools={"isaac_evaluator": Path("/eval/isaac_bank_exam.py")},
        arm_name="M3_old",
        gpu=2,
    )
    assert "device=cuda:2" in command
    assert f"+schedule_json={schedule_path}" in command
    assert "+per_clip_quota=50" in command
    assert "+schedule_seed=0" in command
    assert "+noise_scale=0.0" in command
    assert "+allow_inexact_contract=true" in command
    assert not any("materialize" in value for value in command)


def _mujoco_fixture(tmp_path: Path, prereg: dict, schedule_path: Path, schedule: dict):
    runtime_path = tmp_path / "mujoco_runtime.json"
    result_path = tmp_path / "mujoco_result.json"
    runtime = {
        "shared_schedule": {
            "path": str(schedule_path),
            "file_sha256": R.sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "schema_version": 3,
            "schedule_k": 100,
            "attempts_per_side": 50,
            "seed": 0,
            "hold_range": [0, 100],
            "question_id_order": [item["question_id"] for item in schedule["items"]],
        }
    }
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    arms = {}
    observed = {}
    for name in R.ARM_ORDER:
        artifacts = {}
        for kind in ("report", "summary", "attempt_ledger"):
            path = tmp_path / f"{name}.{kind}"
            path.write_text(kind, encoding="utf-8")
            artifacts[kind] = {"path": str(path), "sha256": R.sha256_file(path)}
        arm = {
            "run_name": prereg["arms"][name]["run_name"],
            "checkpoint_sha256": prereg["arms"][name]["checkpoint_sha256"],
            **artifacts,
            "schedule_sha256": schedule["schedule_sha256"],
            "question_id_order": runtime["shared_schedule"]["question_id_order"],
            "mjcf_sha256": prereg["paper"]["mjcf_sha256"],
            "execution_contract_sha256": "1" * 64,
            "ready_state_sha256": "2" * 64,
            "evaluation_contract_exact": False,
            "causal": True,
            "formal_target": False,
        }
        arms[name] = arm
        observed[name] = copy.deepcopy(arm)
    result = {
        "schema_version": 1,
        "pair_id": "phase1-M3-old-vs-S1-terminal-clean-q50-execution-v2",
        "status": "complete",
        "runtime_contract": {
            "path": str(runtime_path),
            "sha256": R.sha256_file(runtime_path),
        },
        "causal": True,
        "evaluation_contract_exact": False,
        "formal_target": False,
        "deploy_gate": False,
        "shared_schedule_sha256": schedule["schedule_sha256"],
        "question_id_order": runtime["shared_schedule"]["question_id_order"],
        "arms": arms,
    }
    # Production pair ledgers are canonical JSON and therefore sort map keys as M3_S1,M3_old.
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    helper = SimpleNamespace(
        validate_runtime_contract=lambda *args: runtime,
        validate_runtime_inputs=lambda *args: None,
        validate_exam_result=lambda **kwargs: observed[
            "M3_old" if "old_pairing" in kwargs["arm"]["run_name"] else "M3_S1"
        ],
    )
    return runtime_path, result_path, helper


def test_completed_mujoco_pair_binding_rejects_formal_or_repapered_result(tmp_path: Path):
    config = _config()
    prereg = _prereg()
    mujoco_config = {"contract_id": config["mujoco_binding"]["execution_contract_id"]}
    schedule_path, schedule = _schedule(tmp_path, prereg["paper"]["exam_bank"]["sha256"])
    runtime_path, result_path, helper = _mujoco_fixture(
        tmp_path, prereg, schedule_path, schedule
    )
    runtime, result, arms = R.validate_mujoco_result(
        result_path=result_path,
        expected_result_sha=R.sha256_file(result_path),
        runtime_path=runtime_path,
        expected_runtime_sha=R.sha256_file(runtime_path),
        config=config,
        prereg=prereg,
        mujoco_config=mujoco_config,
        helper=helper,
    )
    assert runtime["shared_schedule"]["schedule_k"] == 100
    assert result["evaluation_contract_exact"] is False
    assert tuple(arms) == R.ARM_ORDER

    result["formal_target"] = True
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    with pytest.raises(R.ContractError):
        R.validate_mujoco_result(
            result_path=result_path,
            expected_result_sha=R.sha256_file(result_path),
            runtime_path=runtime_path,
            expected_runtime_sha=R.sha256_file(runtime_path),
            config=config,
            prereg=prereg,
            mujoco_config=mujoco_config,
            helper=helper,
        )
