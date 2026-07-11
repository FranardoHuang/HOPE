#!/usr/bin/env python3
"""Harden the six still-live Phase-1 original/scale-out curve workers.

These workers were launched manually with ``nohup setsid`` and therefore do
not have the launch-contract sidecar used by the later causal-followup
launcher.  This tool deliberately does not invent that sidecar shape.  The
``attest`` phase verifies the exact live process and writes a no-signal Pod
attestation.  ``validate`` and ``replace`` require the explicit SHA-256 of
that attestation and recheck every live fact.

The replacement is Pod-atomic at preflight: every registered worker on the
selected Pod must be the sole member of its PGID and have no child/judge before
any signal.  Replacement sends TERM only to those exact worker PGIDs, never
KILL, preserves the legacy state/log tree, strictly attests completed jobs into
a fresh state directory, and starts the SHA-pinned standalone hardened worker
with the same manifest.  Trainers, judges, Git checkouts, and real hardware are
outside this tool's authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")
HARD_STATE_KEYS = ("manifest_sha256", "job_spec_sha256", "job_contract_sha256")
VALUED_WORKER_OPTIONS = {
    "--manifest",
    "--judge-script",
    "--state-dir",
    "--max-active-cpu",
    "--export-timeout-s",
    "--poll-s",
    "--checkpoint-wait-timeout-s",
    "--checkpoint-poll-s",
    "--checkpoint-stable-s",
}
FLAG_WORKER_OPTIONS = {"--wait-for-checkpoints"}


class ContractError(RuntimeError):
    """A safety, identity, or provenance contract failed."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise ContractError(f"output parent does not exist: {path.parent}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def claim_json(path: Path, value: dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise ContractError(f"claim parent does not exist: {path.parent}")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise ContractError(f"no-clobber claim already exists: {path}") from None
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return value


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_git_checkout(path: Path, expected: str, label: str) -> None:
    if git_output(path, "rev-parse", "HEAD") != expected:
        raise ContractError(f"{label} checkout is not at {expected}")
    if git_output(path, "status", "--porcelain"):
        raise ContractError(f"{label} checkout is dirty: {path}")


def format_runtime_path(template: str, queue_id: str | None = None, pod: str | None = None) -> Path:
    return Path(template.format(queue_id=queue_id, pod=pod)).resolve()


def load_config(path: Path) -> dict[str, Any]:
    data = load_json(path, "global worker hardening config")
    if data.get("schema_version") != 1 or data.get("simulation_only") is not True:
        raise ContractError("config must be schema_version=1 and simulation_only=true")
    if data.get("real_robot_commands_forbidden") is not True:
        raise ContractError("config must explicitly forbid real-robot commands")
    runtime = data.get("runtime")
    pods = data.get("pods")
    queues = data.get("queues")
    if not isinstance(runtime, dict) or not isinstance(pods, dict) or not isinstance(queues, list):
        raise ContractError("runtime/pods/queues have the wrong type")
    if len(queues) != 6:
        raise ContractError("global hardening must bind exactly six live legacy workers")
    for key in ("expected_training_commit", "expected_eval_commit"):
        if not isinstance(runtime.get(key), str) or not GIT_SHA_RE.fullmatch(runtime[key]):
            raise ContractError(f"{key} must be a full lowercase Git SHA")
    for key in (
        "judge_sha256",
        "legacy_worker_sha256",
        "standalone_hardened_worker_sha256",
    ):
        require_sha(runtime.get(key), key)
    if runtime["legacy_worker_sha256"] == runtime["standalone_hardened_worker_sha256"]:
        raise ContractError("legacy and hardened worker SHAs must differ")
    if runtime.get("required_max_active_cpu") != 6:
        raise ContractError("the registered live command requires max_active_cpu=6")
    if not isinstance(runtime.get("worker_python_argv0_allowed"), list) or not all(
        isinstance(item, str) and item for item in runtime["worker_python_argv0_allowed"]
    ):
        raise ContractError("worker_python_argv0_allowed must be a non-empty string list")
    for key in ("training_checkout", "eval_checkout", "external_control_root", "worker_python"):
        if not Path(str(runtime.get(key, ""))).is_absolute():
            raise ContractError(f"runtime {key} must be an absolute path")
    for key in (
        "transaction_sidecar_template",
        "correction_sidecar_template",
        "hardened_launch_sidecar_template",
    ):
        if not Path(str(runtime.get(key, ""))).is_absolute():
            raise ContractError(f"runtime {key} must be an absolute template")

    exclusions = data.get("out_of_live_replacement_scope")
    if not isinstance(exclusions, list) or len(exclusions) != 1:
        raise ContractError("config must record the one naturally completed excluded queue")
    excluded = exclusions[0]
    if (
        not isinstance(excluded, dict)
        or excluded.get("queue_id") != "cadence_causal_pod1"
        or excluded.get("pod") != "pod1"
        or excluded.get("former_legacy_pid") != 1394150
        or excluded.get("disposition") != "naturally_exited_after_M3_terminal_completion"
        or excluded.get("expected_manifest_sha256")
        != "b51ddaa50eba3b06893740c2764e98c96d5fbb8751993e95e3f934602c6a36de"
        or "never" not in str(excluded.get("signal_policy", ""))
    ):
        raise ContractError("cadence_causal_pod1 exclusion is not the frozen natural-exit record")

    expected_pods = {"pod1": 3, "pod2": 3}
    by_id: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for queue in queues:
        if not isinstance(queue, dict):
            raise ContractError("every queue record must be an object")
        queue_id = queue.get("queue_id")
        pod = queue.get("pod")
        if (
            not isinstance(queue_id, str)
            or not SAFE_NAME_RE.fullmatch(queue_id)
            or queue_id in by_id
            or pod not in expected_pods
        ):
            raise ContractError(f"unsafe/duplicate queue or Pod: {queue_id!r}/{pod!r}")
        if not isinstance(queue.get("legacy_pid_hint"), int) or queue["legacy_pid_hint"] <= 1:
            raise ContractError(f"{queue_id}: legacy_pid_hint must be a positive PID")
        source = Path(str(queue.get("source_repo_manifest", "")))
        if source.is_absolute() or ".." in source.parts or source.parts[:1] != ("configs",):
            raise ContractError(f"{queue_id}: unsafe source_repo_manifest")
        require_sha(queue.get("expected_manifest_sha256"), f"{queue_id} manifest")
        for key in (
            "runtime_manifest",
            "legacy_state_dir",
            "legacy_worker_log",
            "hardened_state_dir",
            "hardened_worker_log",
        ):
            value = Path(str(queue.get(key, "")))
            if not value.is_absolute():
                raise ContractError(f"{queue_id}: {key} must be absolute")
            normalized = str(value)
            if normalized in seen_paths:
                raise ContractError(f"{queue_id}: duplicate runtime path {normalized}")
            seen_paths.add(normalized)
        old_state = Path(queue["legacy_state_dir"])
        new_state = Path(queue["hardened_state_dir"])
        if old_state == new_state:
            raise ContractError(f"{queue_id}: new state dir reuses legacy state")
        if Path(queue["legacy_worker_log"]).parent != old_state:
            raise ContractError(f"{queue_id}: legacy worker log must be inside legacy state dir")
        if Path(queue["hardened_worker_log"]).parent != new_state:
            raise ContractError(f"{queue_id}: hardened worker log must be inside new state dir")
        by_id[queue_id] = queue

    for pod, expected_count in expected_pods.items():
        names = pods.get(pod, {}).get("queues")
        if (
            not isinstance(names, list)
            or len(names) != expected_count
            or len(set(names)) != expected_count
            or set(names) != {name for name, queue in by_id.items() if queue["pod"] == pod}
        ):
            raise ContractError(f"{pod} queue list contradicts the six live queue records")
    expected_hints = {1394810, 1380340, 1397266, 194276, 192815, 195085}
    if {queue["legacy_pid_hint"] for queue in queues} != expected_hints:
        raise ContractError("legacy PID hints differ from the reviewed live inventory")
    return data


def checkpoint_iteration(job: dict[str, Any]) -> int:
    checkpoint = Path(str(job.get("checkpoint", "")))
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if not match:
        raise ContractError(f"job {job.get('id')}: checkpoint is not model_<iteration>.pt")
    return int(match.group(1))


def validate_manifest(manifest: dict[str, Any], queue: dict[str, Any]) -> dict[str, Any]:
    queue_id = queue["queue_id"]
    policy = manifest.get("screen_policy")
    jobs = manifest.get("jobs")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(policy, dict)
        or policy.get("screen_only") is not True
        or policy.get("stop_or_promote_allowed") is not False
        or policy.get("schedule_k") != 20
        or policy.get("attempts_per_side") != 10
        or not isinstance(jobs, list)
        or not jobs
    ):
        raise ContractError(f"{queue_id}: manifest lacks the frozen q10 screen policy")
    if manifest.get("judge_script_sha256") is None:
        raise ContractError(f"{queue_id}: manifest does not bind judge_script_sha256")
    ids: set[str] = set()
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for job in jobs:
        if not isinstance(job, dict):
            raise ContractError(f"{queue_id}: every job must be an object")
        job_id = job.get("id")
        if not isinstance(job_id, str) or not SAFE_NAME_RE.fullmatch(job_id) or job_id in ids:
            raise ContractError(f"{queue_id}: unsafe or duplicate job id {job_id!r}")
        ids.add(job_id)
        iteration = checkpoint_iteration(job)
        if not job_id.endswith(f"_{iteration}_clean_q10"):
            raise ContractError(f"{job_id}: id/checkpoint milestone mismatch")
        run_dir = Path(str(job.get("run_dir", "")))
        checkpoint = Path(str(job.get("checkpoint", "")))
        if not run_dir.is_absolute() or not checkpoint.is_absolute() or checkpoint.parent != run_dir:
            raise ContractError(f"{job_id}: run/checkpoint must be absolute and directly paired")
        if not isinstance(job.get("gpu"), int) or job["gpu"] < 0:
            raise ContractError(f"{job_id}: gpu must be a non-negative integer")
        if job.get("screen_only") is not True:
            raise ContractError(f"{job_id}: job is not screen_only")
        exact = job.get("expected_evaluation_contract_exact")
        formal = job.get("formal_target")
        role = job.get("evaluation_role")
        if not isinstance(exact, bool) or not isinstance(formal, bool):
            raise ContractError(f"{job_id}: exact/formal flags must be booleans")
        if formal and not exact:
            raise ContractError(f"{job_id}: inexact job cannot be formal")
        if not isinstance(role, str) or not role:
            raise ContractError(f"{job_id}: evaluation_role is missing")
        expected_args = ["--schedule-k", "20"]
        if not exact:
            expected_args += ["--exam-extra", "--allow-inexact-contract"]
        if job.get("extra_args") != expected_args:
            raise ContractError(f"{job_id}: judge arguments differ from the frozen screen contract")
        if "seed" in policy and job.get("seed") != policy["seed"]:
            raise ContractError(f"{job_id}: seed contradicts screen policy")
        if "noise_scales" in policy and job.get("noise_scales") != policy["noise_scales"]:
            raise ContractError(f"{job_id}: noise_scales contradicts screen policy")

        if current is None or current["iteration"] != iteration:
            if current is not None and iteration <= current["iteration"]:
                raise ContractError(f"{queue_id}: milestone groups are not strictly increasing")
            current = {"iteration": iteration, "jobs": [], "barriers": set(), "runs": set()}
            groups.append(current)
        current["jobs"].append(job_id)
        current["runs"].add(str(run_dir))
        current["barriers"].add(job.get("barrier_id"))

    expected_runs = groups[0]["runs"]
    readiness = manifest.get("checkpoint_readiness_barrier")
    if readiness is not None:
        if not isinstance(readiness, dict) or readiness.get("ordering") != "milestone_major":
            raise ContractError(f"{queue_id}: readiness barrier is not milestone_major")
    seen_barriers: set[str] = set()
    group_summaries = []
    for group in groups:
        if group["runs"] != expected_runs:
            raise ContractError(f"{queue_id}: run membership changes across milestones")
        barriers = group["barriers"]
        if None in barriers and len(barriers) > 1:
            raise ContractError(f"{queue_id}: barrier only covers part of a milestone group")
        if None not in barriers and len(barriers) != 1:
            raise ContractError(f"{queue_id}: barrier/milestone mismatch: contradictory IDs")
        barrier = None if barriers == {None} else next(iter(barriers))
        if barrier is not None:
            if not isinstance(barrier, str) or not SAFE_NAME_RE.fullmatch(barrier):
                raise ContractError(f"{queue_id}: unsafe barrier id {barrier!r}")
            if not barrier.endswith(f"_{group['iteration']}"):
                raise ContractError(f"{queue_id}: barrier/milestone mismatch {barrier}")
            if barrier in seen_barriers:
                raise ContractError(f"{queue_id}: barrier group is discontinuous/reused: {barrier}")
            seen_barriers.add(barrier)
        if readiness is not None and barrier is None:
            raise ContractError(f"{queue_id}: scale-out milestone lacks a barrier id")
        group_summaries.append({
            "iteration": group["iteration"],
            "job_count": len(group["jobs"]),
            "barrier_id": barrier,
        })
    return {
        "job_count": len(jobs),
        "run_count": len(expected_runs),
        "milestone_groups": group_summaries,
        "screen_policy_sha256": canonical_sha256(policy),
    }


def parse_proc_cmdline(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item]


def proc_executable(pid: int) -> Path:
    return Path(f"/proc/{pid}/exe").resolve(strict=True)


def process_alive(pid: int) -> bool:
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = path.read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    return len(fields) > 2 and fields[2] != "Z"


def proc_children(pid: int) -> list[int]:
    raw = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8").strip()
    return [int(value) for value in raw.split() if value.isdigit()]


def process_table() -> list[dict[str, Any]]:
    output = subprocess.check_output(["ps", "-eo", "pid=,pgid=,ppid=,args="], text=True)
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3 or not all(item.isdigit() for item in parts[:3]):
            continue
        rows.append({
            "pid": int(parts[0]),
            "pgid": int(parts[1]),
            "ppid": int(parts[2]),
            "args": parts[3] if len(parts) == 4 else "",
        })
    return rows


def parse_worker_options(command: list[str]) -> dict[str, Any]:
    if len(command) < 3:
        raise ContractError("worker command is too short")
    options: dict[str, Any] = {}
    index = 2
    while index < len(command):
        option = command[index]
        if option in VALUED_WORKER_OPTIONS:
            if option in options or index + 1 >= len(command):
                raise ContractError(f"worker command requires one value for {option}")
            options[option] = command[index + 1]
            index += 2
            continue
        if option in FLAG_WORKER_OPTIONS:
            if option in options:
                raise ContractError(f"duplicate worker flag {option}")
            options[option] = True
            index += 1
            continue
        raise ContractError(f"unregistered worker option/positional argument: {option!r}")
    return options


def validate_worker_command(
    command: list[str], queue: dict[str, Any], config: dict[str, Any], runtime_paths: dict[str, Path]
) -> dict[str, Any]:
    if len(command) < 3:
        raise ContractError(f"{queue['queue_id']}: worker command is too short")
    if command[0] not in config["runtime"]["worker_python_argv0_allowed"]:
        raise ContractError(f"{queue['queue_id']}: unregistered Python argv0 {command[0]!r}")
    if Path(command[1]).resolve() != runtime_paths["legacy_worker"]:
        raise ContractError(f"{queue['queue_id']}: command does not use the legacy worker")
    options = parse_worker_options(command)
    required = {"--manifest", "--judge-script", "--state-dir", "--max-active-cpu", "--wait-for-checkpoints"}
    if not required.issubset(options):
        raise ContractError(f"{queue['queue_id']}: command lacks required wait-worker options")
    exact_paths = {
        "--manifest": Path(queue["runtime_manifest"]).resolve(),
        "--judge-script": runtime_paths["judge"],
        "--state-dir": Path(queue["legacy_state_dir"]).resolve(),
    }
    for option, expected in exact_paths.items():
        if Path(options[option]).resolve() != expected:
            raise ContractError(f"{queue['queue_id']}: {option} changed from its registered path")
    try:
        max_active = int(options["--max-active-cpu"])
    except ValueError:
        raise ContractError(f"{queue['queue_id']}: max-active-cpu is not an integer") from None
    if max_active != config["runtime"]["required_max_active_cpu"]:
        raise ContractError(f"{queue['queue_id']}: max-active-cpu differs from 6")
    numeric_rules = {
        "--export-timeout-s": (1.0, False),
        "--poll-s": (0.0, True),
        "--checkpoint-wait-timeout-s": (0.0, False),
        "--checkpoint-poll-s": (0.0, True),
        "--checkpoint-stable-s": (0.0, False),
    }
    for option, (minimum, strict) in numeric_rules.items():
        if option not in options:
            continue
        try:
            value = float(options[option])
        except ValueError:
            raise ContractError(f"{queue['queue_id']}: {option} is not numeric") from None
        if value < minimum or (strict and value <= minimum):
            raise ContractError(f"{queue['queue_id']}: unsafe {option}={value}")
    return options


def assert_idle_exact_worker(
    queue: dict[str, Any], config: dict[str, Any], runtime_paths: dict[str, Path],
    expected_command: list[str] | None = None,
) -> dict[str, Any]:
    pid = queue["legacy_pid_hint"]
    if not process_alive(pid):
        raise ContractError(f"{queue['queue_id']}: hinted legacy worker pid={pid} is not alive")
    command = parse_proc_cmdline(pid)
    validate_worker_command(command, queue, config, runtime_paths)
    if expected_command is not None and command != expected_command:
        raise ContractError(f"{queue['queue_id']}: /proc command changed from attestation")
    if proc_executable(pid) != Path(config["runtime"]["worker_python"]).resolve():
        raise ContractError(f"{queue['queue_id']}: /proc executable is not the pinned Python")
    rows = process_table()
    row = next((item for item in rows if item["pid"] == pid), None)
    if row is None or row["pgid"] != pid:
        raise ContractError(f"{queue['queue_id']}: worker does not bind pid==pgid")
    children = proc_children(pid)
    ppid_children = sorted(item["pid"] for item in rows if item["ppid"] == pid)
    members = sorted(item["pid"] for item in rows if item["pgid"] == pid)
    if children or ppid_children:
        raise ContractError(
            f"{queue['queue_id']}: worker has child/judge; refusing signal: "
            f"proc={children} ps={ppid_children}"
        )
    if members != [pid]:
        raise ContractError(f"{queue['queue_id']}: PGID {pid} is not single-member: {members}")
    # The PID is only a hint.  Also reject a second exact command for the same queue.
    matches = []
    for item in rows:
        try:
            candidate = parse_proc_cmdline(item["pid"])
            validate_worker_command(candidate, queue, config, runtime_paths)
        except (ContractError, FileNotFoundError, ProcessLookupError, PermissionError, UnicodeError):
            continue
        matches.append(item["pid"])
    if sorted(matches) != [pid]:
        raise ContractError(f"{queue['queue_id']}: expected one exact worker, found {sorted(matches)}")
    return {
        "pid": pid,
        "pgid": pid,
        "command": command,
        "command_sha256": canonical_sha256(command),
        "python_executable": str(proc_executable(pid)),
        "children": [],
        "process_group_members": [pid],
    }


def expected_judge_command(job: dict[str, Any], judge: Path) -> list[str]:
    command = [
        "bash",
        str(judge),
        job["run_dir"],
        job["checkpoint"],
        "--gpu",
        str(job["gpu"]),
        "--seed",
        str(job.get("seed", 0)),
        "--noise-scales",
        str(job.get("noise_scales", "0.0 0.05")),
        "--hold-ref",
        str(job.get("hold_ref", "auto")),
    ]
    command.extend(job["extra_args"])
    return command


def validate_completed_state(
    state_path: Path, log_path: Path, job: dict[str, Any], manifest: dict[str, Any],
    manifest_sha: str, config: dict[str, Any], runtime_paths: dict[str, Path],
) -> dict[str, Any]:
    state = load_json(state_path, f"completed state {job['id']}")
    if state.get("status") != "complete" or state.get("returncode") != 0:
        raise ContractError(f"job {job['id']}: legacy state is not complete rc=0")
    if (
        not isinstance(state.get("pid"), int)
        or state["pid"] <= 1
        or state.get("pgid") != state["pid"]
    ):
        raise ContractError(f"job {job['id']}: legacy judge state does not bind pid==pgid")
    checkpoint = Path(job["checkpoint"])
    if not checkpoint.is_file():
        raise ContractError(f"job {job['id']}: completed checkpoint is missing")
    checkpoint_sha = sha256_file(checkpoint)
    expected = {
        "id": job["id"],
        "run_dir": job["run_dir"],
        "checkpoint": job["checkpoint"],
        "checkpoint_sha256": checkpoint_sha,
        "judge_script_sha256": config["runtime"]["judge_sha256"],
        "eval_commit": config["runtime"]["expected_eval_commit"],
        "training_commit": config["runtime"]["expected_training_commit"],
        "command": expected_judge_command(job, runtime_paths["judge"]),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ContractError(f"job {job['id']}: completed state mismatch at {key}")
    if not log_path.is_file():
        raise ContractError(f"job {job['id']}: completed result log is missing")
    hard_expected = {
        "manifest_sha256": manifest_sha,
        "job_spec_sha256": canonical_sha256(job),
        "job_contract_sha256": canonical_sha256({
            "screen_policy": manifest["screen_policy"], "job": job,
        }),
    }
    present = [key for key in HARD_STATE_KEYS if key in state]
    if present and set(present) != set(HARD_STATE_KEYS):
        raise ContractError(f"job {job['id']}: partially hardened legacy state")
    for key in present:
        if state.get(key) != hard_expected[key]:
            raise ContractError(f"job {job['id']}: existing {key} is wrong")
    state_pid = state.get("pid")
    if isinstance(state_pid, int) and process_alive(state_pid):
        try:
            if parse_proc_cmdline(state_pid) == expected["command"]:
                raise ContractError(f"job {job['id']}: completed state still has a live judge")
        except (FileNotFoundError, ProcessLookupError):
            pass
    return {
        "id": job["id"],
        "job": job,
        "state": state,
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "hard_expected": hard_expected,
        "source_hardening": "already_hardened" if present else "strict_legacy_attestation",
    }


def audit_legacy_states(
    queue: dict[str, Any], manifest: dict[str, Any], manifest_sha: str,
    config: dict[str, Any], runtime_paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    state_dir = Path(queue["legacy_state_dir"])
    worker_log = Path(queue["legacy_worker_log"])
    if not state_dir.is_dir() or not worker_log.is_file():
        raise ContractError(f"{queue['queue_id']}: legacy state dir/worker log is missing")
    job_ids = {job["id"] for job in manifest["jobs"]}
    unexpected = sorted(
        path.name for path in state_dir.glob("*.json")
        if path.name != "summary.json" and path.stem not in job_ids
    )
    if unexpected:
        raise ContractError(f"{queue['queue_id']}: unexpected legacy state files {unexpected}")
    completed = []
    pending = []
    for job in manifest["jobs"]:
        state_path = state_dir / f"{job['id']}.json"
        log_path = state_dir / f"{job['id']}.log"
        if not state_path.exists():
            if log_path.exists():
                raise ContractError(f"job {job['id']}: orphan legacy result log")
            pending.append(job["id"])
            continue
        completed.append(validate_completed_state(
            state_path, log_path, job, manifest, manifest_sha, config, runtime_paths
        ))
    summary_path = state_dir / "summary.json"
    summary = None
    if summary_path.exists():
        summary = {"path": str(summary_path), "sha256": sha256_file(summary_path)}
    evidence = {
        "legacy_state_dir": str(state_dir.resolve()),
        "legacy_worker_log": str(worker_log.resolve()),
        "completed_jobs": [
            {key: item[key] for key in (
                "id", "state_path", "state_sha256", "log_path", "log_sha256",
                "checkpoint_path", "checkpoint_sha256", "source_hardening",
            )}
            for item in completed
        ],
        "pending_job_ids": pending,
        "summary": summary,
    }
    return completed, pending, evidence


def queue_output_paths(config: dict[str, Any], queue: dict[str, Any]) -> dict[str, Path]:
    runtime = config["runtime"]
    queue_id = queue["queue_id"]
    return {
        "new_state_dir": Path(queue["hardened_state_dir"]).resolve(),
        "new_worker_log": Path(queue["hardened_worker_log"]).resolve(),
        "launch_sidecar": format_runtime_path(
            runtime["hardened_launch_sidecar_template"], queue_id=queue_id
        ),
        "correction_sidecar": format_runtime_path(
            runtime["correction_sidecar_template"], queue_id=queue_id
        ),
    }


def audit_queue(
    queue: dict[str, Any], config: dict[str, Any], runtime_paths: dict[str, Path]
) -> dict[str, Any]:
    manifest_path = Path(queue["runtime_manifest"]).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != queue["expected_manifest_sha256"]:
        raise ContractError(f"{queue['queue_id']}: runtime manifest is missing or wrong SHA")
    manifest = load_json(manifest_path, f"{queue['queue_id']} manifest")
    if manifest.get("judge_script_sha256") != config["runtime"]["judge_sha256"]:
        raise ContractError(f"{queue['queue_id']}: manifest judge SHA differs from config")
    if manifest.get("training_checkout") != config["runtime"]["training_checkout"]:
        raise ContractError(f"{queue['queue_id']}: manifest training checkout differs from config")
    if manifest.get("expected_training_commit") != config["runtime"]["expected_training_commit"]:
        raise ContractError(f"{queue['queue_id']}: manifest training commit differs from config")
    manifest_summary = validate_manifest(manifest, queue)
    process = assert_idle_exact_worker(queue, config, runtime_paths)
    completed, pending, state_evidence = audit_legacy_states(
        queue, manifest, queue["expected_manifest_sha256"], config, runtime_paths
    )
    outputs = queue_output_paths(config, queue)
    for label, path in outputs.items():
        if path.exists():
            raise ContractError(f"{queue['queue_id']}: new {label} already exists: {path}")
    evidence = {
        "queue_id": queue["queue_id"],
        "pod": queue["pod"],
        "pid_hint": queue["legacy_pid_hint"],
        "process": process,
        "manifest": {
            "path": str(manifest_path),
            "sha256": queue["expected_manifest_sha256"],
            "summary": manifest_summary,
        },
        "legacy_states": state_evidence,
    }
    return {
        "queue": queue,
        "manifest": manifest,
        "manifest_summary": manifest_summary,
        "process": process,
        "completed": completed,
        "pending": pending,
        "outputs": outputs,
        "evidence": evidence,
    }


def validate_attestation_path(path: Path, config: dict[str, Any]) -> Path:
    path = path.resolve()
    control = Path(config["runtime"]["external_control_root"]).resolve()
    prefix = config["runtime"]["attestation_filename_prefix"]
    if path.parent != control or not path.name.startswith(prefix) or path.suffix != ".json":
        raise ContractError("attestation must be a direct, prefix-bound JSON child of control root")
    return path


def preflight(
    config: dict[str, Any], pod: str, *, config_path: Path, tool_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = config["runtime"]
    training = Path(runtime["training_checkout"]).resolve()
    evaluation = Path(runtime["eval_checkout"]).resolve()
    control = Path(runtime["external_control_root"]).resolve()
    hardened_worker = Path(runtime["standalone_hardened_worker"]).resolve()
    if any(is_within(path, training) or is_within(path, evaluation)
           for path in (config_path, tool_path, hardened_worker)):
        raise ContractError("config/tool/hardened worker must stay outside both Git worktrees")
    if any(not is_within(path, control) for path in (config_path, tool_path, hardened_worker)):
        raise ContractError("config/tool/hardened worker must live under external control root")
    if not control.is_dir():
        raise ContractError(f"external control root is missing: {control}")
    verify_git_checkout(training, runtime["expected_training_commit"], "training")
    verify_git_checkout(evaluation, runtime["expected_eval_commit"], "evaluation")
    legacy_worker = (evaluation / runtime["legacy_worker_relative_path"]).resolve()
    judge = (evaluation / runtime["judge_relative_path"]).resolve()
    fixed = (
        (legacy_worker, runtime["legacy_worker_sha256"], "legacy worker"),
        (judge, runtime["judge_sha256"], "judge"),
        (hardened_worker, runtime["standalone_hardened_worker_sha256"], "hardened worker"),
    )
    for path, expected, label in fixed:
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"{label} is missing or has the wrong SHA: {path}")
    worker_python = Path(runtime["worker_python"]).resolve()
    if not worker_python.is_file():
        raise ContractError(f"worker Python is missing: {worker_python}")
    transaction = format_runtime_path(runtime["transaction_sidecar_template"], pod=pod)
    if transaction.exists():
        raise ContractError(f"Pod transaction already exists: {transaction}")
    if not is_within(transaction, control):
        raise ContractError("transaction path escapes external control root")
    runtime_paths = {
        "training": training,
        "evaluation": evaluation,
        "control": control,
        "legacy_worker": legacy_worker,
        "hardened_worker": hardened_worker,
        "judge": judge,
        "worker_python": worker_python,
    }
    by_id = {queue["queue_id"]: queue for queue in config["queues"]}
    # List comprehension completes the entire read-only Pod audit before callers can signal.
    audits = [audit_queue(by_id[name], config, runtime_paths) for name in config["pods"][pod]["queues"]]
    summary = {
        "pod": pod,
        "training_commit": runtime["expected_training_commit"],
        "eval_commit": runtime["expected_eval_commit"],
        "legacy_worker_sha256": runtime["legacy_worker_sha256"],
        "hardened_worker_sha256": runtime["standalone_hardened_worker_sha256"],
        "judge_sha256": runtime["judge_sha256"],
        "transaction_path": str(transaction),
    }
    return audits, summary


def attestation_document(
    config: dict[str, Any], pod: str, audits: list[dict[str, Any]], summary: dict[str, Any],
    *, config_sha: str, tool_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": config["contract_id"],
        "status": "attested_no_signals",
        "pod": pod,
        "config_sha256": config_sha,
        "tool_sha256": tool_sha,
        "preflight": summary,
        "queues": {
            audit["queue"]["queue_id"]: {
                "evidence": audit["evidence"],
                "evidence_sha256": canonical_sha256(audit["evidence"]),
            }
            for audit in audits
        },
        "attested_utc": utc_now(),
    }


def bind_attestation(
    path: Path, expected_sha: str, config: dict[str, Any], pod: str,
    audits: list[dict[str, Any]], *, config_sha: str, tool_sha: str,
) -> dict[str, Any]:
    require_sha(expected_sha, "expected attestation hash")
    if sha256_file(path) != expected_sha:
        raise ContractError("attestation SHA differs from explicit replacement authorization")
    value = load_json(path, "live worker attestation")
    expected_top = {
        "schema_version": 1,
        "contract_id": config["contract_id"],
        "status": "attested_no_signals",
        "pod": pod,
        "config_sha256": config_sha,
        "tool_sha256": tool_sha,
    }
    for key, expected in expected_top.items():
        if value.get(key) != expected:
            raise ContractError(f"attestation mismatch at {key}")
    queues = value.get("queues")
    if not isinstance(queues, dict) or set(queues) != {
        audit["queue"]["queue_id"] for audit in audits
    }:
        raise ContractError("attestation queue set differs from Pod-atomic preflight")
    for audit in audits:
        queue_id = audit["queue"]["queue_id"]
        entry = queues[queue_id]
        if not isinstance(entry, dict):
            raise ContractError(f"attestation entry is not an object: {queue_id}")
        evidence = entry.get("evidence")
        if entry.get("evidence_sha256") != canonical_sha256(evidence):
            raise ContractError(f"attestation evidence SHA is wrong: {queue_id}")
        if evidence != audit["evidence"]:
            raise ContractError(f"live worker/state evidence changed after attestation: {queue_id}")
    return value


def claim_new_state_dirs(audits: list[dict[str, Any]]) -> None:
    claimed: list[Path] = []
    try:
        for audit in audits:
            path = audit["outputs"]["new_state_dir"]
            path.mkdir(exist_ok=False)
            claimed.append(path)
    except Exception:
        for path in reversed(claimed):
            try:
                path.rmdir()
            except OSError:
                pass
        raise


def exact_term_verified_workers(
    audits: list[dict[str, Any]], config: dict[str, Any], runtime_paths: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    # Recheck the complete Pod set before the first signal.  If any worker grew
    # a child/judge, this comprehension raises and zero workers are signalled.
    snapshots = {
        audit["queue"]["queue_id"]: assert_idle_exact_worker(
            audit["queue"], config, runtime_paths, audit["process"]["command"]
        )
        for audit in audits
    }
    for audit in audits:
        queue_id = audit["queue"]["queue_id"]
        pid = snapshots[queue_id]["pid"]
        if not process_alive(pid) or parse_proc_cmdline(pid) != snapshots[queue_id]["command"]:
            raise ContractError(f"{queue_id}: identity changed before Pod-atomic TERM")
    # Use one final shared process-table snapshot immediately before the tight
    # TERM loop.  This closes the long gap that would otherwise exist between
    # auditing queue 1 and auditing queue N on the same Pod.
    rows = process_table()
    for queue_id, snapshot in snapshots.items():
        pid = snapshot["pid"]
        children = proc_children(pid)
        ppid_children = sorted(item["pid"] for item in rows if item["ppid"] == pid)
        members = sorted(item["pid"] for item in rows if item["pgid"] == pid)
        if children or ppid_children:
            raise ContractError(
                f"{queue_id}: worker gained child/judge before TERM; zero signals sent"
            )
        if members != [pid]:
            raise ContractError(
                f"{queue_id}: worker PGID changed before TERM; zero signals sent"
            )
    stopped: dict[str, dict[str, Any]] = {}
    for audit in audits:
        queue_id = audit["queue"]["queue_id"]
        pid = snapshots[queue_id]["pid"]
        os.killpg(pid, signal.SIGTERM)
        stopped[queue_id] = {
            **snapshots[queue_id],
            "signal": "SIGTERM",
            "signalled_utc": utc_now(),
        }
    wait_seconds = config["runtime"]["old_worker_term_wait_seconds"]
    for audit in audits:
        queue_id = audit["queue"]["queue_id"]
        pid = snapshots[queue_id]["pid"]
        deadline = time.monotonic() + wait_seconds
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.25)
        if process_alive(pid):
            raise ContractError(
                f"{queue_id}: exact worker PGID {pid} did not exit after TERM; no KILL sent"
            )
        stopped[queue_id]["stopped_utc"] = utc_now()
    return stopped


def freeze_legacy_queue(
    audit: dict[str, Any], config: dict[str, Any], runtime_paths: dict[str, Path]
) -> dict[str, Any]:
    queue = audit["queue"]
    completed, pending, evidence = audit_legacy_states(
        queue, audit["manifest"], queue["expected_manifest_sha256"], config, runtime_paths
    )
    worker_log = Path(queue["legacy_worker_log"])
    evidence["legacy_worker_log_sha256"] = sha256_file(worker_log)
    return {"completed": completed, "pending": pending, "evidence": evidence}


def verify_frozen_legacy(frozen: dict[str, Any]) -> None:
    evidence = frozen["evidence"]
    worker_log = Path(evidence["legacy_worker_log"])
    if sha256_file(worker_log) != evidence["legacy_worker_log_sha256"]:
        raise ContractError(f"legacy worker log changed after TERM: {worker_log}")
    for item in evidence["completed_jobs"]:
        for path_key, sha_key in (
            ("state_path", "state_sha256"),
            ("log_path", "log_sha256"),
            ("checkpoint_path", "checkpoint_sha256"),
        ):
            if sha256_file(Path(item[path_key])) != item[sha_key]:
                raise ContractError(f"legacy completed artifact changed: {item[path_key]}")
    summary = evidence.get("summary")
    if summary and sha256_file(Path(summary["path"])) != summary["sha256"]:
        raise ContractError(f"legacy summary changed after TERM: {summary['path']}")


def migrate_completed_states(
    audit: dict[str, Any], frozen: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    output = audit["outputs"]["new_state_dir"]
    records = []
    for source in frozen["completed"]:
        job = source["job"]
        hard = source["hard_expected"]
        state = {
            "id": job["id"],
            "status": "complete",
            "returncode": 0,
            "command": source["state"]["command"],
            "run_dir": job["run_dir"],
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            **hard,
            "judge_script_sha256": config["runtime"]["judge_sha256"],
            "eval_root": str(Path(config["runtime"]["eval_checkout"]).resolve()),
            "eval_commit": config["runtime"]["expected_eval_commit"],
            "training_commit": config["runtime"]["expected_training_commit"],
            "provenance_mode": "strict_legacy_state_attestation",
            "source_state": {
                "path": source["state_path"], "sha256": source["state_sha256"],
            },
            "source_log": {"path": source["log_path"], "sha256": source["log_sha256"]},
            "attested_utc": utc_now(),
        }
        path = output / f"{job['id']}.json"
        if path.exists():
            raise ContractError(f"migrated state no-clobber failure: {path}")
        atomic_json(path, state)
        loaded = load_json(path, "migrated hardened state")
        expected_hard = {
            "manifest_sha256": audit["queue"]["expected_manifest_sha256"],
            "job_spec_sha256": canonical_sha256(job),
            "job_contract_sha256": canonical_sha256({
                "screen_policy": audit["manifest"]["screen_policy"], "job": job,
            }),
        }
        for key, expected in expected_hard.items():
            if loaded.get(key) != expected:
                raise ContractError(f"migrated job {job['id']}: {key} mismatch")
        records.append({"id": job["id"], "path": str(path), "sha256": sha256_file(path)})
    return records


def derive_hardened_command(
    old: list[str], hardened_worker: Path, new_state_dir: Path
) -> list[str]:
    new = list(old)
    new[1] = str(hardened_worker)
    options = parse_worker_options(new)
    old_state = options["--state-dir"]
    positions = [index for index, value in enumerate(new) if value == "--state-dir"]
    if len(positions) != 1:
        raise ContractError("worker command has duplicate --state-dir")
    state_index = positions[0] + 1
    new[state_index] = str(new_state_dir)
    changed = {1, state_index}
    if any(old[index] != new[index] for index in range(len(old)) if index not in changed):
        raise ContractError("hardened command changed an option other than worker/state dir")
    if old_state == str(new_state_dir):
        raise ContractError("hardened command reuses the legacy state dir")
    return new


def start_hardened_worker(
    audit: dict[str, Any], config: dict[str, Any], runtime_paths: dict[str, Path],
    *, attestation_path: Path, attestation_sha: str,
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    queue = audit["queue"]
    outputs = audit["outputs"]
    command = derive_hardened_command(
        audit["process"]["command"], runtime_paths["hardened_worker"], outputs["new_state_dir"]
    )
    if Path(parse_worker_options(command)["--manifest"]).resolve() != Path(queue["runtime_manifest"]).resolve():
        raise ContractError(f"{queue['queue_id']}: hardened command changed manifest")
    log_handle = outputs["new_worker_log"].open("xb", buffering=0)
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    pgid = os.getpgid(proc.pid)
    if pgid != proc.pid:
        raise ContractError(
            f"{queue['queue_id']}: hardened worker is not pid=pgid isolated; left untouched"
        )
    sidecar = {
        "schema_version": 1,
        "contract_id": config["contract_id"],
        "queue_id": queue["queue_id"],
        "pod": queue["pod"],
        "pid": proc.pid,
        "pgid": pgid,
        "command": command,
        "command_sha256": canonical_sha256(command),
        "worker_sha256": config["runtime"]["standalone_hardened_worker_sha256"],
        "manifest": queue["runtime_manifest"],
        "manifest_sha256": queue["expected_manifest_sha256"],
        "state_dir": str(outputs["new_state_dir"]),
        "log": str(outputs["new_worker_log"]),
        "source_attestation": {"path": str(attestation_path), "sha256": attestation_sha},
        "started_utc": utc_now(),
    }
    atomic_json(outputs["launch_sidecar"], sidecar)
    time.sleep(config["runtime"]["new_worker_startup_seconds"])
    returncode = proc.poll()
    if returncode not in (None, 0):
        raise ContractError(f"{queue['queue_id']}: hardened worker exited rc={returncode}")
    sidecar["startup_status"] = "alive" if returncode is None else "complete_rc0"
    atomic_json(outputs["launch_sidecar"], sidecar)
    return proc, sidecar


def replace(
    config: dict[str, Any], pod: str, audits: list[dict[str, Any]], summary: dict[str, Any],
    runtime_paths: dict[str, Path], *, config_sha: str, tool_sha: str,
    attestation_path: Path, attestation_sha: str,
) -> dict[str, Any]:
    transaction_path = Path(summary["transaction_path"])
    transaction: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": config["contract_id"],
        "pod": pod,
        "status": "claimed_after_pod_atomic_preflight",
        "config_sha256": config_sha,
        "tool_sha256": tool_sha,
        "attestation": {"path": str(attestation_path), "sha256": attestation_sha},
        "preflight": summary,
        "started_utc": utc_now(),
        "queues": {},
    }
    claim_json(transaction_path, transaction)
    try:
        claim_new_state_dirs(audits)
        stopped = exact_term_verified_workers(audits, config, runtime_paths)
        frozen = {
            audit["queue"]["queue_id"]: freeze_legacy_queue(audit, config, runtime_paths)
            for audit in audits
        }
        migrated = {
            audit["queue"]["queue_id"]: migrate_completed_states(
                audit, frozen[audit["queue"]["queue_id"]], config
            )
            for audit in audits
        }
        launches: dict[str, dict[str, Any]] = {}
        processes: dict[str, subprocess.Popen[bytes]] = {}
        for audit in audits:
            queue_id = audit["queue"]["queue_id"]
            proc, sidecar = start_hardened_worker(
                audit, config, runtime_paths,
                attestation_path=attestation_path, attestation_sha=attestation_sha,
            )
            processes[queue_id] = proc
            launches[queue_id] = sidecar
        corrections = {}
        for audit in audits:
            queue = audit["queue"]
            queue_id = queue["queue_id"]
            verify_frozen_legacy(frozen[queue_id])
            correction = {
                "schema_version": 1,
                "contract_id": config["contract_id"],
                "status": "complete_worker_replaced_completed_jobs_strictly_attested",
                "pod": pod,
                "queue_id": queue_id,
                "manifest": {
                    "path": queue["runtime_manifest"],
                    "sha256": queue["expected_manifest_sha256"],
                },
                "source_attestation": {
                    "path": str(attestation_path), "sha256": attestation_sha,
                },
                "legacy_worker": {
                    **stopped[queue_id],
                    "worker_sha256": config["runtime"]["legacy_worker_sha256"],
                    "state_tree": frozen[queue_id]["evidence"],
                },
                "hardened_worker": {
                    **launches[queue_id],
                    "sidecar": {
                        "path": str(audit["outputs"]["launch_sidecar"]),
                        "sha256": sha256_file(audit["outputs"]["launch_sidecar"]),
                    },
                    "migrated_completed_states": migrated[queue_id],
                    "pending_job_ids": frozen[queue_id]["pending"],
                },
                "legacy_artifacts_preserved": True,
                "completed_utc": utc_now(),
            }
            atomic_json(audit["outputs"]["correction_sidecar"], correction)
            corrections[queue_id] = {
                "path": str(audit["outputs"]["correction_sidecar"]),
                "sha256": sha256_file(audit["outputs"]["correction_sidecar"]),
            }
        transaction.update(
            status="complete_workers_replaced_completed_jobs_strictly_attested",
            queues=corrections,
            completed_utc=utc_now(),
        )
        atomic_json(transaction_path, transaction)
        return transaction
    except Exception as error:
        transaction.update(status="failed_preserved", failure=repr(error), failed_utc=utc_now())
        try:
            atomic_json(transaction_path, transaction)
        except Exception:
            pass
        # Never broaden a replacement failure into cleanup of a trainer, judge,
        # or correctly started hardened worker.
        raise


def runtime_paths_for(config: dict[str, Any]) -> dict[str, Path]:
    runtime = config["runtime"]
    evaluation = Path(runtime["eval_checkout"]).resolve()
    return {
        "training": Path(runtime["training_checkout"]).resolve(),
        "evaluation": evaluation,
        "control": Path(runtime["external_control_root"]).resolve(),
        "legacy_worker": (evaluation / runtime["legacy_worker_relative_path"]).resolve(),
        "hardened_worker": Path(runtime["standalone_hardened_worker"]).resolve(),
        "judge": (evaluation / runtime["judge_relative_path"]).resolve(),
        "worker_python": Path(runtime["worker_python"]).resolve(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--pod", required=True, choices=("pod1", "pod2"))
    parser.add_argument(
        "--attestation",
        required=True,
        type=Path,
        help="no-clobber output for attest; exact input for validate/replace",
    )
    parser.add_argument("--expected-attestation-sha256")
    parser.add_argument("mode", choices=("attest", "validate", "replace"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    tool_path = Path(__file__).resolve()
    require_sha(args.expected_config_sha256, "expected config hash")
    require_sha(args.expected_tool_sha256, "expected tool hash")
    config_sha = sha256_file(config_path)
    tool_sha = sha256_file(tool_path)
    if config_sha != args.expected_config_sha256:
        raise ContractError("config SHA differs from explicit authorization")
    if tool_sha != args.expected_tool_sha256:
        raise ContractError("tool SHA differs from explicit authorization")
    config = load_config(config_path)
    attestation_path = validate_attestation_path(args.attestation, config)
    if args.mode == "attest" and args.expected_attestation_sha256 is not None:
        raise ContractError("attest creates a new sidecar and must not receive an expected hash")
    if args.mode != "attest" and args.expected_attestation_sha256 is None:
        raise ContractError("validate/replace require --expected-attestation-sha256")
    audits, summary = preflight(
        config, args.pod, config_path=config_path, tool_path=tool_path
    )
    if args.mode == "attest":
        document = attestation_document(
            config, args.pod, audits, summary, config_sha=config_sha, tool_sha=tool_sha
        )
        claim_json(attestation_path, document)
        print(json.dumps({
            "status": "attested_no_signals",
            "pod": args.pod,
            "attestation": str(attestation_path),
            "attestation_sha256": sha256_file(attestation_path),
            "workers": [audit["process"] for audit in audits],
        }, indent=2, sort_keys=True))
        return 0
    attestation_sha = require_sha(
        args.expected_attestation_sha256, "expected attestation hash"
    )
    bind_attestation(
        attestation_path, attestation_sha, config, args.pod, audits,
        config_sha=config_sha, tool_sha=tool_sha,
    )
    if args.mode == "validate":
        print(json.dumps({
            "status": "validated_read_only_attestation_bound",
            "pod": args.pod,
            "attestation_sha256": attestation_sha,
            "workers": [audit["process"] for audit in audits],
            "summary": summary,
        }, indent=2, sort_keys=True))
        return 0
    result = replace(
        config, args.pod, audits, summary, runtime_paths_for(config),
        config_sha=config_sha, tool_sha=tool_sha,
        attestation_path=attestation_path, attestation_sha=attestation_sha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
