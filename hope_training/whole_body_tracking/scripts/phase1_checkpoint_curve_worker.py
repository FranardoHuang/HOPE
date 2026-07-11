#!/usr/bin/env python3
"""Pipeline Phase-1 checkpoint judges without overlapping Isaac exports.

The worker starts one judge at a time until that judge reaches its CPU-only
MuJoCo phase.  Completed exports may then overlap on the host CPU.  Every judge
owns a new process group; this worker never searches for or signals unrelated
training processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


CPU_MARKER = "③ mujoco_eval_onnx"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("jobs"), list):
        raise ValueError("manifest must be schema_version=1 with a jobs list")
    ids: set[str] = set()
    for raw in data["jobs"]:
        if not isinstance(raw, dict):
            raise ValueError("every manifest job must be an object")
        job_id = raw.get("id")
        if not isinstance(job_id, str) or not SAFE_ID.fullmatch(job_id):
            raise ValueError(f"unsafe or missing job id: {job_id!r}")
        if job_id in ids:
            raise ValueError(f"duplicate job id: {job_id}")
        ids.add(job_id)
        if not isinstance(raw.get("gpu"), int) or raw["gpu"] < 0:
            raise ValueError(f"job {job_id}: gpu must be a non-negative integer")
        for key in ("run_dir", "checkpoint"):
            if not isinstance(raw.get(key), str) or not os.path.isabs(raw[key]):
                raise ValueError(f"job {job_id}: {key} must be an absolute path")
    return data


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def reap(active: list[dict[str, Any]], *, block: bool = False) -> int:
    failures = 0
    while True:
        changed = False
        for item in list(active):
            proc: subprocess.Popen[bytes] = item["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            item["log_handle"].close()
            state = dict(item["state"])
            state.update(status="complete" if rc == 0 else "failed", returncode=rc,
                         finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            atomic_json(item["state_path"], state)
            print(f"[curve-worker] {state['id']} finished rc={rc}", flush=True)
            active.remove(item)
            failures += int(rc != 0)
            changed = True
        if changed or not block or not active:
            return failures
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--judge-script", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--max-active-cpu", type=int, default=12)
    parser.add_argument("--export-timeout-s", type=int, default=1200)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--continue-on-pre-cpu-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_active_cpu < 1 or args.export_timeout_s < 1 or args.poll_s <= 0:
        parser.error("worker limits must be positive")

    manifest = load_manifest(args.manifest.resolve())
    judge_script = args.judge_script.resolve()
    if not judge_script.is_file():
        raise FileNotFoundError(f"judge script not found: {judge_script}")
    judge_sha = sha256(judge_script)
    expected_judge_sha = manifest.get("judge_script_sha256")
    if expected_judge_sha and judge_sha != expected_judge_sha:
        raise RuntimeError("judge.sh SHA-256 does not match the frozen manifest")

    eval_root = Path(git_output(judge_script.parent, "rev-parse", "--show-toplevel"))
    eval_commit = git_output(eval_root, "rev-parse", "HEAD")
    eval_dirty = git_output(eval_root, "status", "--porcelain")
    if eval_dirty:
        raise RuntimeError(f"refusing a dirty evaluation worktree: {eval_root}")

    training_checkout_raw = manifest.get("training_checkout")
    expected_training_commit = manifest.get("expected_training_commit")
    training_commit = None
    if training_checkout_raw is not None or expected_training_commit is not None:
        if not isinstance(training_checkout_raw, str) or not os.path.isabs(training_checkout_raw):
            raise ValueError("training_checkout must be an absolute path when a commit is pinned")
        if not isinstance(expected_training_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", expected_training_commit
        ):
            raise ValueError("expected_training_commit must be a full lowercase Git SHA")
        training_checkout = Path(training_checkout_raw)
        training_commit = git_output(training_checkout, "rev-parse", "HEAD")
        if training_commit != expected_training_commit:
            raise RuntimeError(
                f"training checkout commit mismatch: {training_commit} != {expected_training_commit}"
            )
        if git_output(training_checkout, "status", "--porcelain"):
            raise RuntimeError(f"refusing a dirty training checkout: {training_checkout}")

    state_dir = args.state_dir.resolve()
    if not args.dry_run:
        state_dir.mkdir(parents=True, exist_ok=True)

    active: list[dict[str, Any]] = []
    failures = 0
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )

    for job in manifest["jobs"]:
        job_id = job["id"]
        run_dir = Path(job["run_dir"])
        checkpoint = Path(job["checkpoint"])
        if not run_dir.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError(f"job {job_id}: missing run/checkpoint")
        if checkpoint.parent.resolve() != run_dir.resolve():
            raise ValueError(f"job {job_id}: checkpoint must be directly inside run_dir")
        checkpoint_sha = sha256(checkpoint)

        state_path = state_dir / f"{job_id}.json"
        log_path = state_dir / f"{job_id}.log"
        if state_path.exists():
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            if prior.get("status") == "complete" and prior.get("returncode") == 0:
                if (
                    prior.get("checkpoint_sha256") == checkpoint_sha
                    and prior.get("judge_script_sha256") == judge_sha
                    and prior.get("eval_commit") == eval_commit
                    and prior.get("training_commit") == training_commit
                ):
                    print(f"[curve-worker] skip verified completed {job_id}", flush=True)
                    continue
                raise RuntimeError(f"completed job {job_id} no longer matches its frozen inputs")
            prior_pid = prior.get("pid")
            if isinstance(prior_pid, int) and process_alive(prior_pid):
                raise RuntimeError(f"job {job_id} already has a live judge pid={prior_pid}")
            raise RuntimeError(
                f"job {job_id} has preserved non-success state; use a new/archive state directory"
            )

        command = [
            "bash", str(judge_script), str(run_dir), str(checkpoint),
            "--gpu", str(job["gpu"]),
            "--seed", str(job.get("seed", 0)),
            "--noise-scales", str(job.get("noise_scales", "0.0 0.05")),
            "--hold-ref", str(job.get("hold_ref", "auto")),
        ]
        extra_args = job.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(isinstance(v, str) for v in extra_args):
            raise ValueError(f"job {job_id}: extra_args must be a string list")
        command.extend(extra_args)

        print("[curve-worker] launch " + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        while len(active) >= args.max_active_cpu:
            failures += reap(active, block=True)

        log_handle = log_path.open("wb", buffering=0)
        proc = subprocess.Popen(
            command,
            cwd=judge_script.parent.parent,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state = {
            "id": job_id,
            "status": "exporting",
            "pid": proc.pid,
            "pgid": proc.pid,
            "command": command,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "judge_script_sha256": judge_sha,
            "eval_root": str(eval_root),
            "eval_commit": eval_commit,
            "training_commit": training_commit,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(state_path, state)

        deadline = time.monotonic() + args.export_timeout_s
        while True:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if CPU_MARKER in text:
                state.update(status="cpu_exam",
                             cpu_exam_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                atomic_json(state_path, state)
                active.append({"proc": proc, "log_handle": log_handle, "state": state,
                               "state_path": state_path})
                print(f"[curve-worker] {job_id} export complete; CPU exam active", flush=True)
                break
            rc = proc.poll()
            if rc is not None:
                log_handle.close()
                state.update(status="failed", returncode=rc,
                             finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                atomic_json(state_path, state)
                print(f"[curve-worker] {job_id} failed before CPU phase rc={rc}", flush=True)
                failures += 1
                if not args.continue_on_pre_cpu_failure:
                    while active:
                        failures += reap(active, block=True)
                    atomic_json(
                        state_dir / "summary.json",
                        {
                            "status": "failed",
                            "failures": failures,
                            "stopped_after": job_id,
                            "reason": "judge_failed_before_cpu_phase",
                            "eval_commit": eval_commit,
                            "training_commit": training_commit,
                            "finished_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        },
                    )
                    return 1
                break
            if time.monotonic() >= deadline:
                state.update(status="export_stalled")
                atomic_json(state_path, state)
                log_handle.close()
                raise TimeoutError(
                    f"job {job_id} did not reach CPU phase in time; judge pid={proc.pid} left untouched"
                )
            failures += reap(active)
            time.sleep(args.poll_s)

    if args.dry_run:
        return 0
    while active:
        failures += reap(active, block=True)
    summary = {
        "status": "complete" if failures == 0 else "failed",
        "failures": failures,
        "eval_commit": eval_commit,
        "training_commit": training_commit,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(state_dir / "summary.json", summary)
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
