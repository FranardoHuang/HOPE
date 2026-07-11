from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "phase1_checkpoint_curve_worker.py"
SPEC = importlib.util.spec_from_file_location("phase1_checkpoint_curve_worker_unit", WORKER)
assert SPEC is not None and SPEC.loader is not None
WORKER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER_MODULE)


def _git_repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    for name, value in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(path), "-c", "user.name=test", "-c",
            "user.email=test@example.com", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_worker_waits_for_a_stable_preregistered_checkpoint(tmp_path):
    judge_body = "#!/usr/bin/env bash\necho '③ mujoco_eval_onnx'\nexit 0\n"
    eval_root = tmp_path / "eval"
    _git_repo(eval_root, {"scripts/judge.sh": judge_body})
    judge = eval_root / "scripts" / "judge.sh"
    judge.chmod(0o755)
    # chmod changes the tracked mode; commit it so the long-lived clean-tree check is meaningful.
    subprocess.run(["git", "-C", str(eval_root), "add", "scripts/judge.sh"], check=True)
    subprocess.run(
        [
            "git", "-C", str(eval_root), "-c", "user.name=test", "-c",
            "user.email=test@example.com", "commit", "-qm", "executable",
        ],
        check=True,
    )

    training = tmp_path / "training"
    training_commit = _git_repo(training, {"README": "frozen\n"})
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "model_1000.pt"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "screen_policy": {
                    "schedule_k": 20,
                    "attempts_per_side": 10,
                    "screen_only": True,
                    "stop_or_promote_allowed": False,
                },
                "judge_script_sha256": hashlib.sha256(judge.read_bytes()).hexdigest(),
                "training_checkout": str(training),
                "expected_training_commit": training_commit,
                "jobs": [
                    {
                        "id": "fresh_1000",
                        "run_dir": str(run_dir),
                        "checkpoint": str(checkpoint),
                        "gpu": 0,
                        "noise_scales": "0.0",
                        "extra_args": ["--schedule-k", "20"],
                        "evaluation_role": "formal_target",
                        "expected_evaluation_contract_exact": True,
                        "formal_target": True,
                        "screen_only": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(WORKER),
            "--manifest", str(manifest),
            "--judge-script", str(judge),
            "--state-dir", str(state_dir),
            "--wait-for-checkpoints",
            "--checkpoint-poll-s", "0.05",
            "--checkpoint-stable-s", "0.05",
            "--poll-s", "0.02",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 5.0
    while not state_dir.is_dir() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert state_dir.is_dir(), "worker did not finish preflight and enter its state directory"
    time.sleep(0.1)
    run_dir.mkdir()
    checkpoint.write_bytes(b"first")
    time.sleep(0.03)
    checkpoint.write_bytes(b"complete-checkpoint")
    output, _ = proc.communicate(timeout=10)

    assert proc.returncode == 0, output
    assert "waiting for checkpoint fresh_1000" in output
    assert "fresh_1000 export complete" in output
    state = json.loads((state_dir / "fresh_1000.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["command"][-2:] == ["--schedule-k", "20"]
    assert state["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert state["job_spec_sha256"] == WORKER_MODULE.canonical_sha256(
        json.loads(manifest.read_text(encoding="utf-8"))["jobs"][0]
    )
    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    assert state["job_contract_sha256"] == WORKER_MODULE.canonical_sha256(
        {"screen_policy": loaded["screen_policy"], "job": loaded["jobs"][0]}
    )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == state["checkpoint_sha256"]


def test_screen_policy_is_fail_closed_per_job(tmp_path):
    base = {
        "schema_version": 1,
        "screen_policy": {
            "schedule_k": 20,
            "attempts_per_side": 10,
            "screen_only": True,
            "stop_or_promote_allowed": False,
        },
        "jobs": [
            {
                "id": "q10",
                "run_dir": "/tmp/run",
                "checkpoint": "/tmp/run/model_1.pt",
                "gpu": 0,
                "extra_args": ["--schedule-k", "20"],
                "evaluation_role": "formal_target",
                "expected_evaluation_contract_exact": True,
                "formal_target": True,
                "screen_only": True,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    assert WORKER_MODULE.load_manifest(path)["jobs"][0]["screen_only"] is True

    base["jobs"][0]["screen_only"] = False
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="requires screen_only"):
        WORKER_MODULE.load_manifest(path)

    base["jobs"][0]["screen_only"] = True
    base["screen_policy"]["stop_or_promote_allowed"] = True
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="stop_or_promote_allowed=false"):
        WORKER_MODULE.load_manifest(path)


def test_screen_policy_cannot_be_omitted_or_disagree_with_job(tmp_path):
    manifest = {
        "schema_version": 1,
        "screen_policy": {
            "schedule_k": 20,
            "attempts_per_side": 10,
            "screen_only": True,
            "stop_or_promote_allowed": False,
        },
        "jobs": [
            {
                "id": "q10",
                "run_dir": "/tmp/run",
                "checkpoint": "/tmp/run/model_1.pt",
                "gpu": 0,
                "extra_args": ["--schedule-k", "20"],
                "evaluation_role": "formal_target",
                "expected_evaluation_contract_exact": True,
                "formal_target": True,
                "screen_only": True,
            }
        ],
    }
    path = tmp_path / "manifest.json"

    without_policy = dict(manifest)
    without_policy.pop("screen_policy")
    path.write_text(json.dumps(without_policy), encoding="utf-8")
    with pytest.raises(ValueError, match="requires screen_policy"):
        WORKER_MODULE.load_manifest(path)

    manifest["jobs"][0]["extra_args"] = ["--schedule-k", "100"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="contradicts"):
        WORKER_MODULE.load_manifest(path)

    manifest["jobs"][0]["extra_args"] = [
        "--schedule-k", "20", "--exam-extra", "--exam-schedule-k 100"
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="arbitrary --exam-extra"):
        WORKER_MODULE.load_manifest(path)
