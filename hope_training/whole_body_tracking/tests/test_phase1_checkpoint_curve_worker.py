from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "phase1_checkpoint_curve_worker.py"


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
    time.sleep(0.15)
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
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == state["checkpoint_sha256"]
