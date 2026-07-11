from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_phase1_paired_bank_q50.py"
CONFIG_PATH = ROOT / "configs" / "phase1_M3_terminal_q50_execution_20260711.json"
PREREG_PATH = ROOT / "configs" / "phase1_M3_terminal_q50_prereg_20260711.json"
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


R = _load_module("paired_q50_runner_under_test", RUNNER_PATH)
S = _load_module("paired_q50_schedule_under_test", SCHEDULE_MODULE_PATH)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _prereg() -> dict:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _schedule(tmp_path: Path, *, bank_sha256: str = "a" * 64):
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
    return path, artifact


def test_frozen_execution_config_and_prereg_are_causal_inexact_and_not_started():
    config = R.load_execution_config(CONFIG_PATH)
    prereg = _prereg()
    assert R.sha256_file(PREREG_PATH) == config["preregistration_sha256"]
    R.validate_preregistration(prereg, config)
    assert config["schedule"]["schedule_k"] == 100
    assert config["schedule"]["attempts_per_side"] == 50
    assert config["semantics"] == {
        "causal": True,
        "evaluation_contract_exact": False,
        "formal_target": False,
        "deploy_gate": False,
    }
    assert prereg["jobs_started"] == 0
    assert prereg["paper"]["schedule_materialization"]["status"] == "not_materialized"


def test_schema3_adjacent_contract_uses_motion_exactness_not_lineage_alias():
    """The real schema-3 sidecar does not duplicate checkpoint lineage_exact."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    validation = source.split("hard = load_json(hard_path)", 1)[1].split(
        "audit = checkpoint_audit", 1
    )[0]
    assert 'hard.get("motion_kinematics_exact") is not False' in validation
    assert 'hard.get("lineage_exact")' not in validation


@pytest.mark.parametrize(
    "mutation",
    (
        lambda c: c["semantics"].__setitem__("formal_target", True),
        lambda c: c["semantics"].__setitem__("evaluation_contract_exact", True),
        lambda c: c["schedule"].__setitem__("schedule_k", 20),
        lambda c: c["schedule"].__setitem__("same_artifact_for_both_arms", False),
        lambda c: c.__setitem__("auto_start", True),
    ),
)
def test_execution_config_rejects_semantic_relaxation(tmp_path: Path, mutation):
    data = _config()
    mutation(data)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(R.ContractError):
        R.load_execution_config(path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda p: p.__setitem__("jobs_started", 1),
        lambda p: p["paper"]["schedule_materialization"].__setitem__("status", "materialized"),
        lambda p: p["paper"].__setitem__("expected_evaluation_contract_exact", True),
        lambda p: p["diagnostic_semantics"].__setitem__("formal_target", True),
        lambda p: p["arms"]["M3_S1"].__setitem__("lineage_exact", True),
    ),
)
def test_preregistration_rejects_started_or_formalized_mutation(mutation):
    config = R.load_execution_config(CONFIG_PATH)
    prereg = _prereg()
    mutation(prereg)
    with pytest.raises(R.ContractError):
        R.validate_preregistration(prereg, config)


def test_shared_q50_schedule_is_canonical_balanced_and_command_uses_json(tmp_path: Path):
    path, artifact = _schedule(tmp_path)
    document = R.validate_schedule_document(path, expected_bank_sha256="a" * 64)
    assert document["schedule_sha256"] == artifact.schedule_sha256
    assert len(document["items"]) == 100
    assert [item["clip"] for item in document["items"]] == [0, 1] * 50
    assert len({(item["clip"], item["bank_row"]) for item in document["items"]}) == 100

    args = R.semantic_judge_args(path)
    assert "--schedule-k" not in args
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
    assert "--allow-inexact-contract" in args[-1]


def test_schedule_rejects_even_canonically_rehashed_duplicate_row(tmp_path: Path):
    path, _ = _schedule(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["items"][2]["bank_row"] = document["items"][0]["bank_row"]
    document["items"][2]["question_id"] = document["items"][0]["question_id"]
    payload = dict(document)
    payload.pop("schedule_sha256")
    document["schedule_sha256"] = R.canonical_sha256(payload)
    path.write_bytes(R.canonical_bytes(document) + b"\n")
    with pytest.raises(R.ContractError, match="duplicate"):
        R.validate_schedule_document(path, expected_bank_sha256="a" * 64)


def _write_result_fixture(
    tmp_path: Path, schedule_path: Path, schedule: dict, prereg: dict
) -> tuple[Path, dict]:
    report_parent = tmp_path / "judge"
    report_parent.mkdir()
    report = report_parent / "judge_report_model_20998_20260711_120000.md"
    report.write_text("# fixture\n", encoding="utf-8")
    exam_dir = report_parent / "model_20998_20260711_120000" / "exam"
    exam_dir.mkdir(parents=True)
    order = [item["question_id"] for item in schedule["items"]]
    execution_sha = "b" * 64
    ready_sha = "c" * 64
    summary = {
        "schema_version": 3,
        "evaluation_contract_exact": False,
        "mjcf_sha256": prereg["paper"]["mjcf_sha256"],
        "execution_contract_sha256": execution_sha,
        "ready_state_sha256": ready_sha,
        "arguments": {
            "allow_inexact_contract": True,
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
        "results": [
            {
                "noise_scale": 0.0,
                "evaluation_contract_exact": False,
                "exam_schedule": {
                    "question_id_order": order,
                    "items": [
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
                            "hit": True,
                            "returned": True,
                        }
                        for item in schedule["items"]
                    ],
                },
            }
        ],
    }
    (exam_dir / "mujoco_sim2sim_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    with (exam_dir / "mujoco_sim2sim_attempts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "schedule_index",
                "clip_name",
                "bank_row",
                "question_id",
                "repeat",
                "hold_steps",
                "attempt_seed",
                "schedule_sha256",
                "censored",
            ),
        )
        writer.writeheader()
        for item in schedule["items"]:
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
                    "censored": "False",
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
    return report, contract


def test_result_validator_requires_full_same_paper_causal_inexact_ledger(tmp_path: Path):
    prereg = _prereg()
    schedule_path, _ = _schedule(
        tmp_path, bank_sha256=prereg["paper"]["exam_bank"]["sha256"]
    )
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    report, contract = _write_result_fixture(tmp_path, schedule_path, schedule, prereg)
    result = R.validate_exam_result(
        report=report,
        arm=prereg["arms"]["M3_old"],
        prereg=prereg,
        runtime_contract=contract,
    )
    assert result["evaluation_contract_exact"] is False
    assert result["causal"] is True
    assert result["formal_target"] is False
    assert result["question_id_order"] == contract["shared_schedule"]["question_id_order"]

    summary_path = Path(result["summary"]["path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evaluation_contract_exact"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(R.ContractError):
        R.validate_exam_result(
            report=report,
            arm=prereg["arms"]["M3_old"],
            prereg=prereg,
            runtime_contract=contract,
        )


def test_result_validator_rejects_runtime_item_relaxation(tmp_path: Path):
    prereg = _prereg()
    schedule_path, _ = _schedule(
        tmp_path, bank_sha256=prereg["paper"]["exam_bank"]["sha256"]
    )
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    report, contract = _write_result_fixture(tmp_path, schedule_path, schedule, prereg)
    summary_path = report.parent / "model_20998_20260711_120000" / "exam" / (
        "mujoco_sim2sim_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["results"][0]["exam_schedule"]["items"][0]["eligible"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(R.ContractError):
        R.validate_exam_result(
            report=report,
            arm=prereg["arms"]["M3_old"],
            prereg=prereg,
            runtime_contract=contract,
        )
