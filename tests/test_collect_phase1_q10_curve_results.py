from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "collect_phase1_q10_curve_results.py"
FIXTURE = ROOT / "tests" / "fixtures" / "phase1_q10_curve_collector"
SPEC = importlib.util.spec_from_file_location("phase1_q10_collector_tested", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _ref(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": COLLECTOR.sha256_file(path)}


def _schedule_items() -> list[dict]:
    result = []
    for index in range(20):
        clip = index % 2
        result.append(
            {
                "schedule_index": index,
                "question_sequence_index": index,
                "clip": clip,
                "bank_row": index // 2,
                "question_id": f"{'forehand' if clip == 0 else 'backhand'}:q{index // 2}",
                "repeat": 0,
                "hold_steps": index,
                "attempt_seed": 1000 + index,
            }
        )
    return result


def _metric(attempts: int, returned: int) -> dict:
    return {
        "n_attempts": attempts,
        "n_strikes": attempts,
        "contacted": attempts,
        "landed_ok": returned,
        "exact_reach_rate_per_attempt": 1.0,
        "contact_rate_per_attempt": 1.0,
        "return_success_rate_per_attempt": returned / attempts,
    }


def _scorecard(*, fh_returned: int, bh_returned: int) -> dict:
    items = _schedule_items()
    question_order = [item["question_id"] for item in items]
    schedule_sha = "7" * 64
    evaluator_sha = "8" * 64
    runtime_items = [
        {
            **item,
            "eligible": True,
            "censored": False,
            "ready_state_mode": "fixture_ready",
            "ready_state_sha256": "9" * 64,
            "mjcf_sha256": "a" * 64,
            "execution_contract_sha256": "b" * 64,
            "physical_fall": False,
            "guard_reset": False,
            "hit": True,
            "returned": index < (fh_returned + bh_returned),
        }
        for index, item in enumerate(items)
    ]
    return {
        "schema_version": 3,
        "evaluation_contract_exact": False,
        "arguments": {
            "target_source": "bank",
            "exam_schedule_k": 20,
            "exam_schedule_json": None,
            "seed": 0,
            "noise_scales": [0.0],
            "qdes_clamp": True,
            "hold_ref": "auto",
            "exam_continuity_diagnostic": False,
            "allow_inexact_contract": True,
        },
        "input_artifacts": {
            "exam_bank": {"path": "/pod/assets/fixture_exam.npz", "sha256": "c" * 64},
            "evaluator_source": {"path": "/eval/mujoco_eval_onnx.py", "sha256": evaluator_sha},
        },
        "exam_schedule": {
            "schema_version": 1,
            "sha256": schedule_sha,
            "bank_sha256": "c" * 64,
            "seed": 0,
            "size": 20,
            "one_question_reset": True,
            "shared_artifact": None,
            "items": [
                {key: value for key, value in item.items() if key != "question_sequence_index"}
                for item in items
            ],
        },
        "results": [
            {
                "noise_scale": 0.0,
                "evaluation_contract_exact": False,
                "exam_schedule": {
                    "sha256": schedule_sha,
                    "size": 20,
                    "one_question_reset": True,
                    "question_id_order": question_order,
                    "items": runtime_items,
                },
                "attempts": {
                    "n_attempts": 20,
                    "per_clip": {
                        "forehand": {"n_attempts": 10},
                        "backhand": {"n_attempts": 10},
                    },
                },
                "venue": {
                    "forehand": _metric(10, fh_returned),
                    "backhand": _metric(10, bh_returned),
                    "all": _metric(20, fh_returned + bh_returned),
                },
            }
        ],
    }


def _report(job: dict, *, fh_returned: int, bh_returned: int) -> str:
    aggregate = (fh_returned + bh_returned) / 20
    return f"""# fixture report

- checkpoint:`{job['checkpoint']}`

```
[mj-sim2sim] evaluation_contract_exact=false
  immutable_schedule: K=20 seed=0 sha256={'7' * 64} hold_range=(0, 100) no_wrap=true
```

| 侧 × 噪声 | 发球/尝试数 | 活到击球帧数 | 接触率(全尝试) | 回球率(全尝试) |
|---|---|---|---|---|
| 正手 ns=0.0 | 10 | 10 | 1.0000 | {fh_returned / 10:.4f} |
| 反手 ns=0.0 | 10 | 10 | 1.0000 | {bh_returned / 10:.4f} |

- 全侧汇总(发球/尝试全分母):尝试数 ns=0.0→20;接触率 ns=0.0→1.0000;回球成功率 ns=0.0→{aggregate:.4f}
"""


def _state(manifest: dict, manifest_sha: str, job: dict, checkpoint_sha: str) -> dict:
    policy = manifest["screen_policy"]
    return {
        "id": job["id"],
        "status": "complete",
        "returncode": 0,
        "command": [
            "bash",
            "/eval/scripts/judge.sh",
            job["run_dir"],
            job["checkpoint"],
            "--gpu",
            str(job["gpu"]),
            "--seed",
            "0",
            "--noise-scales",
            "0.0",
            "--hold-ref",
            "auto",
            *job["extra_args"],
        ],
        "run_dir": job["run_dir"],
        "checkpoint": job["checkpoint"],
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "job_spec_sha256": COLLECTOR.canonical_sha256(job),
        "job_contract_sha256": COLLECTOR.canonical_sha256(
            {"screen_policy": policy, "job": job}
        ),
        "judge_script_sha256": manifest["judge_script_sha256"],
        "eval_commit": "3" * 64,
        "training_commit": manifest["expected_training_commit"],
    }


def _audit(
    *, manifest: dict, manifest_sha: str, job: dict, state_path: Path,
    report_path: Path, scorecard_path: Path, checkpoint_sha: str, contract_sha: str,
) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "audit_kind": "phase1_q10_checkpoint_audit",
        "read_only": True,
        "real_robot_commands": False,
        "job_id": job["id"],
        "manifest_sha256": manifest_sha,
        "state_sha256": COLLECTOR.sha256_file(state_path),
        "job_spec_sha256": state["job_spec_sha256"],
        "job_contract_sha256": state["job_contract_sha256"],
        "judge_report_sha256": COLLECTOR.sha256_file(report_path),
        "scorecard_sha256": COLLECTOR.sha256_file(scorecard_path),
        "checkpoint": {
            "path": job["checkpoint"],
            "sha256": checkpoint_sha,
            "filename_iteration": 19000,
            "embedded_iteration": 19000,
            "floating_tensor_count": 74,
            "nonfinite_floating_elements": 0,
            "all_floating_tensors_finite": True,
            "embedded_training_contract_sha256": contract_sha,
            "embedded_training_contract_lineage_exact": False,
        },
        "training_contract": {
            "path": f"{job['run_dir']}/params/training_contract.json",
            "sha256": contract_sha,
            "schema_version": 3,
            "structure_validator": "pass",
            "binding_validator": "pass",
            "lineage_exact": False,
        },
        "provenance": {
            "training_commit": manifest["expected_training_commit"],
            "eval_commit": "3" * 64,
            "judge_script_sha256": manifest["judge_script_sha256"],
            "evaluator_source_sha256": "8" * 64,
        },
        "evaluation": {
            "immutable_schedule_sha256": "7" * 64,
            "evaluation_contract_exact": False,
        },
    }


def _build_fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"
    shutil.copyfile(FIXTURE / "manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = COLLECTOR.sha256_file(manifest_path)
    evidence = []
    values = ((2, 8), (4, 9))
    for ordinal, (job, returns) in enumerate(zip(manifest["jobs"], values)):
        stem = "old" if ordinal == 0 else "shared"
        checkpoint_sha = ("4" if ordinal == 0 else "5") * 64
        contract_sha = ("6" if ordinal == 0 else "d") * 64
        state_path = tmp_path / f"{stem}.state.json"
        report_path = tmp_path / f"{stem}.report.md"
        scorecard_path = tmp_path / f"{stem}.scorecard.json"
        audit_path = tmp_path / f"{stem}.audit.json"
        _write_json(state_path, _state(manifest, manifest_sha, job, checkpoint_sha))
        report_path.write_text(
            _report(job, fh_returned=returns[0], bh_returned=returns[1]),
            encoding="utf-8",
        )
        _write_json(
            scorecard_path,
            _scorecard(fh_returned=returns[0], bh_returned=returns[1]),
        )
        _write_json(
            audit_path,
            _audit(
                manifest=manifest,
                manifest_sha=manifest_sha,
                job=job,
                state_path=state_path,
                report_path=report_path,
                scorecard_path=scorecard_path,
                checkpoint_sha=checkpoint_sha,
                contract_sha=contract_sha,
            ),
        )
        evidence.append(
            {
                "manifest_id": "fixture_manifest",
                "job_id": job["id"],
                "worker_state": _ref(state_path),
                "judge_report": _ref(report_path),
                "scorecard": _ref(scorecard_path),
                "checkpoint_audit": _ref(audit_path),
            }
        )
    index = {
        "schema_version": 1,
        "archive_id": "fixture_causal_19000_q10",
        "screen_policy": {
            "screen_only": True,
            "stop_or_promote_allowed": False,
            "q50_triggered": False,
            "decision_claim": None,
        },
        "manifests": [
            {
                "id": "fixture_manifest",
                "path": manifest_path.name,
                "sha256": manifest_sha,
                "barrier_ids": ["causal_19000"],
            }
        ],
        "pairs": [
            {
                "id": "fixture_face_pair_19000",
                "kind": "face_pair",
                "members": [
                    f"fixture_manifest:{manifest['jobs'][0]['id']}",
                    f"fixture_manifest:{manifest['jobs'][1]['id']}",
                ],
            }
        ],
        "evidence": evidence,
    }
    index_path = tmp_path / "index.json"
    _write_json(index_path, index)
    return {
        "index": index,
        "index_path": index_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def _refresh_artifact(index: dict, job_id: str, key: str, path: Path) -> None:
    row = next(item for item in index["evidence"] if item["job_id"] == job_id)
    row[key] = _ref(path)


def _refresh_audit(index: dict, job_id: str, audit_path: Path) -> None:
    _refresh_artifact(index, job_id, "checkpoint_audit", audit_path)


def test_bound_pair_emits_deterministic_content_addressed_screen_archive(tmp_path: Path):
    fixture = _build_fixture(tmp_path)
    document, digest = COLLECTOR.validate_index(fixture["index_path"])
    content = document["content"]
    assert document["content_sha256"] == COLLECTOR.canonical_sha256(content) == digest
    assert content["decision_authority"] == {
        "screen_only": True,
        "stop_authorized": False,
        "promote_authorized": False,
        "q50_triggered": False,
        "decision_claim": None,
    }
    assert len(content["records"]) == 2 and len(content["pairs"]) == 1
    old, shared = content["records"]
    assert old["metrics"]["aggregate"]["return_rate"] == 0.5
    assert shared["metrics"]["aggregate"]["return_rate"] == 0.65
    assert all(row["checkpoint_audit"]["checkpoint_all_floating_tensors_finite"] for row in content["records"])

    output = COLLECTOR.write_archive(fixture["index_path"], tmp_path / "archives")
    assert digest in output.name
    assert COLLECTOR.write_archive(fixture["index_path"], tmp_path / "archives") == output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == document


def test_q50_and_decision_claims_are_refused_before_archiving(tmp_path: Path):
    fixture = _build_fixture(tmp_path)
    index = fixture["index"]
    index["screen_policy"]["q50_triggered"] = True
    _write_json(fixture["index_path"], index)
    with pytest.raises(COLLECTOR.ContractError, match="decision claim"):
        COLLECTOR.validate_index(fixture["index_path"])

    fixture = _build_fixture(tmp_path / "q50")
    manifest = fixture["manifest"]
    manifest["screen_policy"]["schedule_k"] = 100
    _write_json(fixture["manifest_path"], manifest)
    fixture["index"]["manifests"][0]["sha256"] = COLLECTOR.sha256_file(fixture["manifest_path"])
    _write_json(fixture["index_path"], fixture["index"])
    with pytest.raises(COLLECTOR.ContractError, match="q50/non-q10"):
        COLLECTOR.validate_index(fixture["index_path"])


def test_legacy_state_and_incomplete_pair_cannot_be_laundered(tmp_path: Path):
    fixture = _build_fixture(tmp_path)
    index = fixture["index"]
    first = index["evidence"][0]
    state_path = tmp_path / first["worker_state"]["path"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state["job_contract_sha256"]
    _write_json(state_path, state)
    first["worker_state"] = _ref(state_path)
    audit_path = tmp_path / first["checkpoint_audit"]["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["state_sha256"] = COLLECTOR.sha256_file(state_path)
    _write_json(audit_path, audit)
    first["checkpoint_audit"] = _ref(audit_path)
    _write_json(fixture["index_path"], index)
    with pytest.raises(COLLECTOR.ContractError, match="job_contract_sha256"):
        COLLECTOR.validate_index(fixture["index_path"])

    fixture = _build_fixture(tmp_path / "incomplete")
    fixture["index"]["evidence"].pop()
    _write_json(fixture["index_path"], fixture["index"])
    with pytest.raises(COLLECTOR.ContractError, match="exactly cover"):
        COLLECTOR.validate_index(fixture["index_path"])


def test_mixed_schedule_nonfinite_checkpoint_and_report_drift_fail(tmp_path: Path):
    fixture = _build_fixture(tmp_path / "mixed")
    index = fixture["index"]
    second = index["evidence"][1]
    score_path = fixture["index_path"].parent / second["scorecard"]["path"]
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["exam_schedule"]["sha256"] = "e" * 64
    score["results"][0]["exam_schedule"]["sha256"] = "e" * 64
    _write_json(score_path, score)
    second["scorecard"] = _ref(score_path)
    report_path = fixture["index_path"].parent / second["judge_report"]["path"]
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace("7" * 64, "e" * 64),
        encoding="utf-8",
    )
    second["judge_report"] = _ref(report_path)
    audit_path = fixture["index_path"].parent / second["checkpoint_audit"]["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["scorecard_sha256"] = COLLECTOR.sha256_file(score_path)
    audit["judge_report_sha256"] = COLLECTOR.sha256_file(report_path)
    audit["evaluation"]["immutable_schedule_sha256"] = "e" * 64
    _write_json(audit_path, audit)
    second["checkpoint_audit"] = _ref(audit_path)
    _write_json(fixture["index_path"], index)
    with pytest.raises(COLLECTOR.ContractError, match="mixed immutable schedules"):
        COLLECTOR.validate_index(fixture["index_path"])

    fixture = _build_fixture(tmp_path / "nonfinite")
    index = fixture["index"]
    first = index["evidence"][0]
    audit_path = fixture["index_path"].parent / first["checkpoint_audit"]["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["checkpoint"]["nonfinite_floating_elements"] = 1
    audit["checkpoint"]["all_floating_tensors_finite"] = False
    _write_json(audit_path, audit)
    first["checkpoint_audit"] = _ref(audit_path)
    _write_json(fixture["index_path"], index)
    with pytest.raises(COLLECTOR.ContractError, match="checkpoint is not finite"):
        COLLECTOR.validate_index(fixture["index_path"])

    fixture = _build_fixture(tmp_path / "report")
    index = fixture["index"]
    first = index["evidence"][0]
    report_path = fixture["index_path"].parent / first["judge_report"]["path"]
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "正手 ns=0.0 | 10 | 10 | 1.0000 | 0.2000",
            "正手 ns=0.0 | 10 | 10 | 1.0000 | 0.9000",
        ),
        encoding="utf-8",
    )
    first["judge_report"] = _ref(report_path)
    audit_path = fixture["index_path"].parent / first["checkpoint_audit"]["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["judge_report_sha256"] = COLLECTOR.sha256_file(report_path)
    _write_json(audit_path, audit)
    first["checkpoint_audit"] = _ref(audit_path)
    _write_json(fixture["index_path"], index)
    with pytest.raises(COLLECTOR.ContractError, match="forehand report metrics differ"):
        COLLECTOR.validate_index(fixture["index_path"])
