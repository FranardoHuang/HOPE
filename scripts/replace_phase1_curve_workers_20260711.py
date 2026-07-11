#!/usr/bin/env python3
"""Replace legacy Phase-1 curve workers without touching trainers or judges.

Production copies of this tool, its config, and the hardened worker live outside
both Git worktrees.  ``validate`` is read-only.  ``replace`` audits both workers
on one Pod before any signal, refuses a worker with any child/judge, sends TERM
only to exact legacy worker PGIDs, starts the SHA-pinned standalone worker with
the same manifest and a fresh state directory, and waits for a hardened 17k
state before recording correction sidecars.
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
HARD_STATE_KEYS = ("manifest_sha256", "job_spec_sha256", "job_contract_sha256")


class ContractError(RuntimeError):
    """A replacement safety or provenance contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise ContractError(f"output parent does not exist: {path.parent}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_git_checkout(path: Path, expected: str, label: str) -> None:
    if git_output(path, "rev-parse", "HEAD") != expected:
        raise ContractError(f"{label} checkout is not at {expected}")
    if git_output(path, "status", "--porcelain"):
        raise ContractError(f"{label} checkout is dirty: {path}")


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("simulation_only") is not True:
        raise ContractError("config must be schema_version=1 and simulation_only=true")
    if data.get("real_robot_commands_forbidden") is not True:
        raise ContractError("config must forbid real-robot commands")
    runtime = data.get("runtime")
    pods = data.get("pods")
    arms = data.get("arms")
    paths = data.get("arm_relative_paths")
    if not all(isinstance(item, dict) for item in (runtime, pods, paths)):
        raise ContractError("runtime/pods/arm_relative_paths must be objects")
    if not isinstance(arms, list) or len(arms) != 4:
        raise ContractError("config must bind exactly four followup workers")
    for key in ("expected_training_commit", "expected_eval_commit"):
        if not isinstance(runtime.get(key), str) or not GIT_SHA_RE.fullmatch(runtime[key]):
            raise ContractError(f"{key} must be a full lowercase Git SHA")
    for key in (
        "judge_sha256", "legacy_worker_sha256", "standalone_hardened_worker_sha256"
    ):
        require_sha(runtime.get(key), key)
    if runtime["legacy_worker_sha256"] == runtime["standalone_hardened_worker_sha256"]:
        raise ContractError("legacy and hardened workers must have different SHAs")
    expected_names = {
        "phase1_M3_S1_only_guidance0_seed1": "pod1",
        "phase1_M3_S1_only_guidance0_seed2": "pod1",
        "phase1_M2_S1_guidance_m095_seed1": "pod2",
        "phase1_M2_S1_guidance_m095_seed2": "pod2",
    }
    by_name: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if not isinstance(arm, dict):
            raise ContractError("every arm must be an object")
        name = arm.get("run_name")
        if not isinstance(name, str) or not SAFE_NAME_RE.fullmatch(name) or name in by_name:
            raise ContractError(f"unsafe or duplicate arm: {name!r}")
        if expected_names.get(name) != arm.get("pod"):
            raise ContractError(f"arm {name} is assigned to the wrong Pod")
        if arm.get("first_job_id") != f"{name}_17000_clean_q10":
            raise ContractError(f"arm {name} must rejudge its exact 17k job")
        run_dir = Path(str(arm.get("artifact_run_dir", "")))
        if not run_dir.is_absolute() or run_dir.name != name:
            raise ContractError(f"arm {name} has an unsafe artifact run directory")
        by_name[name] = arm
    if set(by_name) != set(expected_names):
        raise ContractError("configured arm set differs from the four followups")
    for pod, expected in (("pod1", 2), ("pod2", 2)):
        names = pods.get(pod, {}).get("arms")
        if not isinstance(names, list) or len(names) != expected or len(set(names)) != expected:
            raise ContractError(f"{pod} must contain two unique arms")
        if set(names) != {name for name, owner in expected_names.items() if owner == pod}:
            raise ContractError(f"{pod} arm list contradicts the arm records")
    required_relative = {
        "launch_contract", "manifest", "legacy_worker_sidecar", "legacy_worker_log",
        "legacy_state_dir", "hardened_worker_sidecar", "hardened_worker_log",
        "hardened_state_dir", "correction_sidecar",
    }
    if set(paths) != required_relative:
        raise ContractError("arm_relative_paths keys differ from the frozen contract")
    for key, value in paths.items():
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ContractError(f"unsafe relative path for {key}: {value}")
    if paths["legacy_state_dir"] == paths["hardened_state_dir"]:
        raise ContractError("new worker must not reuse the legacy state directory")
    return data


def arm_paths(config: dict[str, Any], arm: dict[str, Any]) -> dict[str, Path]:
    root = Path(arm["artifact_run_dir"]).resolve()
    return {key: root / relative for key, relative in config["arm_relative_paths"].items()}


def option_value(command: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ContractError(f"worker command requires exactly one {option}")
    return command[positions[0] + 1]


def parse_proc_cmdline(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item]


def process_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat = stat_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    fields = stat.split()
    return len(fields) > 2 and fields[2] != "Z"


def process_table() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,pgid=,ppid=,args="], text=True
    )
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3 or not all(value.isdigit() for value in parts[:3]):
            continue
        rows.append({
            "pid": int(parts[0]), "pgid": int(parts[1]), "ppid": int(parts[2]),
            "args": parts[3] if len(parts) == 4 else "",
        })
    return rows


def proc_children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    raw = path.read_text(encoding="utf-8").strip()
    return [int(value) for value in raw.split() if value.isdigit()]


def assert_idle_exact_worker(pid: int, expected_command: list[str]) -> dict[str, Any]:
    if not process_alive(pid):
        raise ContractError(f"legacy worker pid={pid} is not alive")
    actual_command = parse_proc_cmdline(pid)
    if actual_command != expected_command:
        raise ContractError(
            f"legacy worker /proc command changed: expected={expected_command!r} "
            f"actual={actual_command!r}"
        )
    children = proc_children(pid)
    rows = process_table()
    ppid_children = sorted(row["pid"] for row in rows if row["ppid"] == pid)
    group_members = sorted(row["pid"] for row in rows if row["pgid"] == pid)
    if children or ppid_children:
        raise ContractError(
            f"legacy worker pid={pid} has child/judge; refusing any wait or signal: "
            f"proc={children} ps={ppid_children}"
        )
    if group_members != [pid]:
        raise ContractError(
            f"legacy worker PGID {pid} is not single-member exact: {group_members}"
        )
    return {
        "pid": pid, "pgid": pid, "command": actual_command,
        "children": [], "process_group_members": group_members,
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return value


def validate_manifest(manifest: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
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
        or len(jobs) != 5
    ):
        raise ContractError(f"{arm['run_name']} q10 manifest lacks the hardened screen policy")
    first = jobs[0]
    if first.get("id") != arm["first_job_id"]:
        raise ContractError(f"{arm['run_name']} first job is not its 17k q10")
    if Path(str(first.get("checkpoint", ""))).name != "model_17000.pt":
        raise ContractError(f"{arm['run_name']} first checkpoint is not model_17000.pt")
    for job in jobs:
        if job.get("screen_only") is not True:
            raise ContractError(f"job {job.get('id')} is not screen_only")
        if job.get("extra_args") != [
            "--schedule-k", "20", "--exam-extra", "--allow-inexact-contract"
        ]:
            raise ContractError(f"job {job.get('id')} has an unregistered judge contract")
    return first


def audit_arm(
    config: dict[str, Any], arm: dict[str, Any], runtime_paths: dict[str, Path]
) -> dict[str, Any]:
    paths = arm_paths(config, arm)
    for key in (
        "hardened_worker_sidecar", "hardened_worker_log", "hardened_state_dir",
        "correction_sidecar",
    ):
        if paths[key].exists():
            raise ContractError(f"new replacement path already exists: {paths[key]}")
    launch_contract = load_json(paths["launch_contract"], "launch contract")
    manifest = load_json(paths["manifest"], "q10 manifest")
    old_sidecar = load_json(paths["legacy_worker_sidecar"], "legacy worker sidecar")
    first_job = validate_manifest(manifest, arm)
    manifest_sha = sha256_file(paths["manifest"])
    binding = launch_contract.get("checkpoint_cadence_q10")
    embedded_worker = launch_contract.get("q10_worker")
    if not isinstance(binding, dict) or not isinstance(embedded_worker, dict):
        raise ContractError(f"{arm['run_name']} launch contract lacks cadence/worker binding")
    if Path(str(binding.get("path", ""))).resolve() != paths["manifest"].resolve():
        raise ContractError(f"{arm['run_name']} launch contract points to another manifest")
    if binding.get("sha256") != manifest_sha:
        raise ContractError(f"{arm['run_name']} manifest SHA differs from launch contract")
    for key in ("pid", "pgid", "command", "command_sha256", "state_path"):
        if old_sidecar.get(key) != embedded_worker.get(key):
            raise ContractError(f"{arm['run_name']} worker sidecar differs at {key}")
    pid = old_sidecar.get("pid")
    pgid = old_sidecar.get("pgid")
    command = old_sidecar.get("command")
    if not isinstance(pid, int) or pid <= 0 or pgid != pid or not isinstance(command, list):
        raise ContractError(f"{arm['run_name']} legacy worker does not bind pid==pgid")
    if canonical_sha256(command) != old_sidecar.get("command_sha256"):
        raise ContractError(f"{arm['run_name']} legacy worker command SHA is wrong")
    if command[:2] != [
        config["runtime"]["worker_python"], str(runtime_paths["legacy_worker"])
    ]:
        raise ContractError(f"{arm['run_name']} did not launch the expected legacy worker")
    if Path(option_value(command, "--manifest")).resolve() != paths["manifest"].resolve():
        raise ContractError(f"{arm['run_name']} worker command changed its manifest")
    if Path(option_value(command, "--state-dir")).resolve() != paths["legacy_state_dir"].resolve():
        raise ContractError(f"{arm['run_name']} worker command changed its legacy state dir")
    if Path(str(old_sidecar.get("state_path", ""))).resolve() != paths["legacy_worker_sidecar"].resolve():
        raise ContractError(f"{arm['run_name']} sidecar path self-binding is wrong")
    old_first_state_path = paths["legacy_state_dir"] / f"{arm['first_job_id']}.json"
    old_first_state = load_json(old_first_state_path, "legacy 17k state")
    if old_first_state.get("status") != "complete" or old_first_state.get("returncode") != 0:
        raise ContractError(f"{arm['run_name']} legacy 17k is not complete rc=0")
    present_hard = [key for key in HARD_STATE_KEYS if key in old_first_state]
    if present_hard:
        raise ContractError(
            f"{arm['run_name']} legacy 17k is already hardened; refusing replacement: {present_hard}"
        )
    checkpoint = Path(first_job["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != old_first_state.get("checkpoint_sha256"):
        raise ContractError(f"{arm['run_name']} 17k checkpoint changed after legacy judgment")
    process = assert_idle_exact_worker(pid, command)
    immutable_old = {
        "worker_sidecar": {"path": str(paths["legacy_worker_sidecar"]),
                           "sha256": sha256_file(paths["legacy_worker_sidecar"])},
        "first_state": {"path": str(old_first_state_path),
                        "sha256": sha256_file(old_first_state_path)},
    }
    return {
        "arm": arm, "paths": paths, "manifest": manifest, "manifest_sha256": manifest_sha,
        "first_job": first_job, "old_sidecar": old_sidecar, "old_process": process,
        "old_first_state": old_first_state, "immutable_old": immutable_old,
        "legacy_worker_log_path": str(paths["legacy_worker_log"]),
    }


def preflight(
    config: dict[str, Any], pod: str, *, config_path: Path, tool_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = config["runtime"]
    training = Path(runtime["training_checkout"]).resolve()
    evaluation = Path(runtime["eval_checkout"]).resolve()
    control = Path(runtime["external_control_root"]).resolve()
    hardened_worker = Path(runtime["standalone_hardened_worker"]).resolve()
    if any(is_within(path, training) or is_within(path, evaluation)
           for path in (config_path, tool_path, hardened_worker)):
        raise ContractError("config/tool/standalone worker must live outside both worktrees")
    if any(not is_within(path, control) for path in (config_path, tool_path, hardened_worker)):
        raise ContractError(f"runtime control files must live under {control}")
    verify_git_checkout(training, runtime["expected_training_commit"], "training")
    verify_git_checkout(evaluation, runtime["expected_eval_commit"], "evaluation")
    legacy_worker = evaluation / runtime["legacy_worker_relative_path"]
    judge = evaluation / runtime["judge_relative_path"]
    fixed = (
        (legacy_worker, runtime["legacy_worker_sha256"], "legacy worker"),
        (judge, runtime["judge_sha256"], "judge"),
        (hardened_worker, runtime["standalone_hardened_worker_sha256"], "hardened worker"),
    )
    for path, expected, label in fixed:
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"{label} missing or wrong SHA: {path}")
    worker_python = Path(runtime["worker_python"])
    if not worker_python.is_file():
        raise ContractError(f"worker Python is missing: {worker_python}")
    runtime_paths = {
        "training": training, "evaluation": evaluation, "control": control,
        "legacy_worker": legacy_worker, "judge": judge,
        "hardened_worker": hardened_worker,
    }
    by_name = {arm["run_name"]: arm for arm in config["arms"]}
    audits = [audit_arm(config, by_name[name], runtime_paths)
              for name in config["pods"][pod]["arms"]]
    transaction_path = Path(runtime["transaction_sidecar_template"].format(pod=pod)).resolve()
    if transaction_path.exists():
        raise ContractError(f"replacement transaction already exists: {transaction_path}")
    if not is_within(transaction_path, control):
        raise ContractError("transaction sidecar escapes external control root")
    summary = {
        "pod": pod, "training_commit": runtime["expected_training_commit"],
        "eval_commit": runtime["expected_eval_commit"],
        "legacy_worker_sha256": runtime["legacy_worker_sha256"],
        "hardened_worker_sha256": runtime["standalone_hardened_worker_sha256"],
        "judge_sha256": runtime["judge_sha256"],
        "transaction_path": str(transaction_path),
    }
    return audits, summary


def claim_transaction(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o444)
    except FileExistsError:
        raise ContractError(f"replacement transaction was claimed concurrently: {path}") from None
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def claim_new_state_dirs(audits: list[dict[str, Any]]) -> None:
    claimed: list[Path] = []
    try:
        for audit in audits:
            path = audit["paths"]["hardened_state_dir"]
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
    audits: list[dict[str, Any]], snapshots: dict[str, dict[str, Any]], wait_seconds: int
) -> dict[str, dict[str, Any]]:
    """TERM the already-audited Pod set, never a child/judge/trainer.

    Every worker is re-audited into ``snapshots`` before this function is called.
    We then recheck all command identities once more before sending the first
    signal, emit TERM to the exact worker PGIDs in a tight loop, and never KILL.
    """
    for audit in audits:
        name = audit["arm"]["run_name"]
        pid = audit["old_sidecar"]["pid"]
        if not process_alive(pid) or parse_proc_cmdline(pid) != snapshots[name]["command"]:
            raise ContractError(f"legacy worker identity changed before Pod-atomic TERM: {name}")
    signalled: dict[str, dict[str, Any]] = {}
    for audit in audits:
        name = audit["arm"]["run_name"]
        pid = audit["old_sidecar"]["pid"]
        os.killpg(pid, signal.SIGTERM)
        signalled[name] = {**snapshots[name], "signal": "SIGTERM", "signalled_utc": utc_now()}
    for audit in audits:
        name = audit["arm"]["run_name"]
        pid = audit["old_sidecar"]["pid"]
        deadline = time.monotonic() + wait_seconds
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.25)
        if process_alive(pid):
            raise ContractError(
                f"legacy worker PGID {pid} did not exit after exact TERM; no KILL was sent"
            )
        signalled[name]["stopped_utc"] = utc_now()
    return signalled


def derive_hardened_command(
    audit: dict[str, Any], hardened_worker: Path, new_state_dir: Path
) -> list[str]:
    old = audit["old_sidecar"]["command"]
    new = list(old)
    new[1] = str(hardened_worker)
    state_index = new.index("--state-dir") + 1
    new[state_index] = str(new_state_dir)
    expected_changes = {1, state_index}
    if any(old[index] != new[index] for index in range(len(old)) if index not in expected_changes):
        raise ContractError("hardened command changed an option other than worker/state_dir")
    if option_value(new, "--manifest") != option_value(old, "--manifest"):
        raise ContractError("hardened command changed the immutable manifest")
    return new


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def start_hardened_worker(
    audit: dict[str, Any], config: dict[str, Any], runtime_paths: dict[str, Path]
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    paths = audit["paths"]
    command = derive_hardened_command(
        audit, runtime_paths["hardened_worker"], paths["hardened_state_dir"]
    )
    log_handle = paths["hardened_worker_log"].open("xb", buffering=0)
    try:
        proc = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log_handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    finally:
        log_handle.close()
    pgid = os.getpgid(proc.pid)
    if pgid != proc.pid:
        proc.terminate()
        raise ContractError(f"hardened worker is not isolated: pid={proc.pid} pgid={pgid}")
    sidecar = {
        "schema_version": 1, "pid": proc.pid, "pgid": pgid,
        "command": command, "command_sha256": canonical_sha256(command),
        "worker_sha256": config["runtime"]["standalone_hardened_worker_sha256"],
        "manifest": str(audit["paths"]["manifest"]),
        "manifest_sha256": audit["manifest_sha256"],
        "state_dir": str(paths["hardened_state_dir"]),
        "log": str(paths["hardened_worker_log"]), "started_utc": utc_now(),
    }
    atomic_json(paths["hardened_worker_sidecar"], sidecar)
    time.sleep(config["runtime"]["new_worker_startup_seconds"])
    if proc.poll() is not None:
        raise ContractError(
            f"hardened worker for {audit['arm']['run_name']} exited during startup rc={proc.returncode}"
        )
    return proc, sidecar


def validate_hard_first_state(audit: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    path = audit["paths"]["hardened_state_dir"] / f"{audit['arm']['first_job_id']}.json"
    if not path.is_file():
        return None
    state = load_json(path, "hardened 17k state")
    if state.get("status") != "complete":
        if state.get("status") == "failed" or state.get("returncode") not in (None, 0):
            raise ContractError(
                f"hardened 17k failed for {audit['arm']['run_name']}: {state}"
            )
        return None
    if state.get("returncode") != 0:
        raise ContractError(f"hardened 17k did not complete rc=0: {state}")
    expected = {
        "manifest_sha256": audit["manifest_sha256"],
        "job_spec_sha256": canonical_sha256(audit["first_job"]),
        "job_contract_sha256": canonical_sha256({
            "screen_policy": audit["manifest"]["screen_policy"],
            "job": audit["first_job"],
        }),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ContractError(
                f"hardened 17k {key} mismatch for {audit['arm']['run_name']}"
            )
    if state.get("judge_script_sha256") != config["runtime"]["judge_sha256"]:
        raise ContractError("hardened 17k judge SHA mismatch")
    if state.get("eval_commit") != config["runtime"]["expected_eval_commit"]:
        raise ContractError("hardened 17k eval commit mismatch")
    if state.get("training_commit") != config["runtime"]["expected_training_commit"]:
        raise ContractError("hardened 17k training commit mismatch")
    return {"path": str(path), "sha256": sha256_file(path), "state": state}


def wait_for_hard_states(
    audits: list[dict[str, Any]], processes: dict[str, subprocess.Popen[bytes]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + config["runtime"]["hard_state_timeout_seconds"]
    completed: dict[str, dict[str, Any]] = {}
    while len(completed) < len(audits):
        for audit in audits:
            name = audit["arm"]["run_name"]
            if name in completed:
                continue
            result = validate_hard_first_state(audit, config)
            if result is not None:
                completed[name] = result
                continue
            proc = processes[name]
            if proc.poll() is not None:
                raise ContractError(
                    f"hardened worker {name} exited rc={proc.returncode} before hard 17k state"
                )
        if len(completed) == len(audits):
            return completed
        if time.monotonic() >= deadline:
            raise ContractError("timed out waiting for all hardened 17k states")
        time.sleep(config["runtime"]["hard_state_poll_seconds"])
    return completed


def freeze_final_legacy_log(audit: dict[str, Any]) -> dict[str, str]:
    """Bind the legacy log only after exact TERM has made it immutable.

    A waiting legacy worker may legitimately append status lines between read-only
    preflight and TERM, so a preflight log hash is not an immutability claim.
    """
    path = Path(audit["legacy_worker_log_path"])
    if not path.is_file():
        raise ContractError(f"legacy worker log disappeared after TERM: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def verify_old_artifacts_unchanged(
    audit: dict[str, Any], final_legacy_log: dict[str, str]
) -> None:
    for item in audit["immutable_old"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ContractError(f"legacy result changed during replacement: {item['path']}")
    if sha256_file(Path(final_legacy_log["path"])) != final_legacy_log["sha256"]:
        raise ContractError(
            f"legacy worker log changed after exact TERM: {final_legacy_log['path']}"
        )


def replace(
    config: dict[str, Any], pod: str, audits: list[dict[str, Any]], summary: dict[str, Any],
    *, config_sha: str, tool_sha: str,
) -> dict[str, Any]:
    transaction_path = Path(summary["transaction_path"])
    transaction: dict[str, Any] = {
        "schema_version": 1, "contract_id": config["contract_id"], "pod": pod,
        "status": "claimed_after_read_only_preflight", "config_sha256": config_sha,
        "tool_sha256": tool_sha, "preflight": summary, "started_utc": utc_now(),
        "arms": {},
    }
    claim_transaction(transaction_path, transaction)
    processes: dict[str, subprocess.Popen[bytes]] = {}
    new_sidecars: dict[str, dict[str, Any]] = {}
    try:
        claim_new_state_dirs(audits)
        # The complete Pod set passed read-only preflight. Recheck each exact process
        # immediately before TERM; never wait for or signal a newly appeared judge.
        final_snapshots = {
            audit["arm"]["run_name"]: assert_idle_exact_worker(
                audit["old_sidecar"]["pid"], audit["old_sidecar"]["command"]
            )
            for audit in audits
        }
        stopped = exact_term_verified_workers(
            audits, final_snapshots, config["runtime"]["old_worker_term_wait_seconds"]
        )
        # Only now are the legacy waiting logs immutable.  Freeze their final SHA
        # after both exact worker PGIDs have exited, before any replacement starts.
        for audit in audits:
            name = audit["arm"]["run_name"]
            stopped[name]["final_legacy_log"] = freeze_final_legacy_log(audit)
        runtime_paths = {
            "hardened_worker": Path(config["runtime"]["standalone_hardened_worker"]),
        }
        for audit in audits:
            name = audit["arm"]["run_name"]
            proc, sidecar = start_hardened_worker(audit, config, runtime_paths)
            processes[name] = proc
            new_sidecars[name] = sidecar
        hard_states = wait_for_hard_states(audits, processes, config)
        corrections = {}
        for audit in audits:
            name = audit["arm"]["run_name"]
            verify_old_artifacts_unchanged(audit, stopped[name]["final_legacy_log"])
            correction = {
                "schema_version": 1, "status": "complete_hard_17k_rejudged",
                "contract_id": config["contract_id"], "pod": pod, "run_name": name,
                "manifest": {"path": str(audit["paths"]["manifest"]),
                             "sha256": audit["manifest_sha256"]},
                "legacy_worker": {
                    **stopped[name],
                    "worker_sha256": config["runtime"]["legacy_worker_sha256"],
                    "sidecar": audit["immutable_old"]["worker_sidecar"],
                    "state_dir": str(audit["paths"]["legacy_state_dir"]),
                    "legacy_17k_state": audit["immutable_old"]["first_state"],
                    "legacy_log": stopped[name]["final_legacy_log"],
                    "hardening_fields_missing": list(HARD_STATE_KEYS),
                },
                "hardened_worker": {
                    **new_sidecars[name],
                    "sidecar": {
                        "path": str(audit["paths"]["hardened_worker_sidecar"]),
                        "sha256": sha256_file(audit["paths"]["hardened_worker_sidecar"]),
                    },
                    "hard_17k_state": hard_states[name],
                },
                "old_artifacts_preserved": True, "completed_utc": utc_now(),
            }
            atomic_json(audit["paths"]["correction_sidecar"], correction)
            corrections[name] = {
                "path": str(audit["paths"]["correction_sidecar"]),
                "sha256": sha256_file(audit["paths"]["correction_sidecar"]),
            }
        transaction.update(
            status="complete_hard_17k_rejudged", arms=corrections, completed_utc=utc_now()
        )
        atomic_json(transaction_path, transaction)
        return transaction
    except Exception as error:
        transaction.update(status="failed_preserved", failure=repr(error), failed_utc=utc_now())
        try:
            atomic_json(transaction_path, transaction)
        except Exception:
            pass
        # Deliberately leave any correctly started hardened worker alive.  Never
        # broaden a correction failure into trainer/judge/process cleanup.
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--pod", required=True, choices=("pod1", "pod2"))
    parser.add_argument("mode", choices=("validate", "replace"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    tool_path = Path(__file__).resolve()
    for value, label in ((args.expected_config_sha256, "config"),
                         (args.expected_tool_sha256, "tool")):
        require_sha(value, f"expected {label} hash")
    config_sha = sha256_file(config_path)
    tool_sha = sha256_file(tool_path)
    if config_sha != args.expected_config_sha256:
        raise ContractError("config SHA differs from explicit replacement authorization")
    if tool_sha != args.expected_tool_sha256:
        raise ContractError("tool SHA differs from explicit replacement authorization")
    config = load_config(config_path)
    audits, summary = preflight(
        config, args.pod, config_path=config_path, tool_path=tool_path
    )
    if args.mode == "validate":
        print(json.dumps({
            "status": "validated_read_only", "pod": args.pod,
            "workers": [audit["old_process"] for audit in audits], "summary": summary,
        }, indent=2, sort_keys=True))
        return 0
    result = replace(
        config, args.pod, audits, summary, config_sha=config_sha, tool_sha=tool_sha
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
