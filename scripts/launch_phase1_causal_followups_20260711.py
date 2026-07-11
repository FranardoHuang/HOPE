#!/usr/bin/env python3
"""Fail-closed external launcher for the 2026-07-11 Phase-1 causal followups.

The production copy of this file and its manifest must live outside the frozen
training checkout.  A launch verifies both files, both Git worktrees, every
binary input, GPU capacity, and run-name absence before creating any state.  It
then starts exactly one isolated trainer PGID through the frozen Kit launcher,
checks the emitted hard-contract SHA, materializes the preregistered q10
cadence, and starts one isolated existing checkpoint worker PGID.  It never
searches for or signals an unrelated process and contains no robot command.
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
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
CONTRACT_MARKER = "[train.py] hard training contract:"


class ContractError(RuntimeError):
    """A preregistered launch invariant was violated."""


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def claim_run_directory(path: Path) -> None:
    """Atomically own one arm after read-only preflight, closing the TOCTOU gap."""
    if not path.parent.is_dir():
        raise ContractError(f"artifact runs root must already exist: {path.parent}")
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError:
        raise ContractError(f"run directory was claimed concurrently or already exists: {path}") from None


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("simulation_only") is not True:
        raise ContractError("manifest must be schema_version=1 and simulation_only=true")
    if data.get("real_robot_commands_forbidden") is not True:
        raise ContractError("manifest must explicitly forbid real-robot commands")
    runtime = data.get("runtime")
    cadence = data.get("checkpoint_evaluation")
    continuation = data.get("continuation_contract")
    families = data.get("families")
    arms = data.get("arms")
    if not all(isinstance(v, dict) for v in (runtime, cadence, continuation, families)):
        raise ContractError("manifest runtime/cadence/continuation/families must be objects")
    if not isinstance(arms, list) or len(arms) != 4:
        raise ContractError("manifest must preregister exactly four followup arms")
    expected_commit = runtime.get("expected_training_commit")
    if not isinstance(expected_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ContractError("expected training commit must be a full lowercase Git SHA")
    _require_sha(runtime.get("locked_launcher_sha256"), "locked launcher hash")
    _require_sha(cadence.get("worker_sha256"), "curve worker hash")
    _require_sha(cadence.get("judge_sha256"), "judge hash")
    q10 = cadence.get("q10_screen")
    q50 = cadence.get("q50_decision_paper")
    if not isinstance(q10, dict) or q10.get("milestones") != [17000, 18000, 19000, 20000, 20998]:
        raise ContractError("q10 milestones must be 17k/18k/19k/20k/20998")
    if q10.get("screen_only") is not True or q10.get("stop_or_promote_allowed") is not False:
        raise ContractError("q10 must be screen-only and unable to stop/promote")
    if q10.get("schedule_k") != 20 or q10.get("attempts_per_side") != 10:
        raise ContractError("q10 must use the frozen 10 attempts per side schedule")
    if not isinstance(q50, dict) or q50.get("auto_activate") is not False:
        raise ContractError("q50 must remain an inactive, separately triggered paper")
    if q50.get("schedule_k") != 100 or q50.get("attempts_per_side") != 50:
        raise ContractError("q50 template must freeze 50 attempts per side")

    expected_specs = {
        "phase1_M3_S1_only_guidance0_seed1": ("pod1", 1, "M3_swing", 1, 0.0),
        "phase1_M3_S1_only_guidance0_seed2": ("pod1", 0, "M3_swing", 2, 0.0),
        "phase1_M2_S1_guidance_m095_seed1": ("pod2", 0, "M2_v4rg_legacy", 1, -0.95),
        "phase1_M2_S1_guidance_m095_seed2": ("pod2", 1, "M2_v4rg_legacy", 2, -0.95),
    }
    seen: set[str] = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise ContractError("every arm must be an object")
        name = arm.get("run_name")
        if not isinstance(name, str) or not RUN_NAME_RE.fullmatch(name) or name in seen:
            raise ContractError(f"unsafe or duplicate run name: {name!r}")
        seen.add(name)
        actual = (
            arm.get("pod"), arm.get("gpu"), arm.get("family"),
            arm.get("training_seed"), arm.get("racket_guidance_weight"),
        )
        if expected_specs.get(name) != actual:
            raise ContractError(f"arm {name} contradicts its preregistered causal slot")
        if arm.get("face_command_pairing") != "shared_plus_y":
            raise ContractError(f"arm {name} is not an S1/shared-plus-y arm")
        family = families.get(arm["family"])
        if not isinstance(family, dict):
            raise ContractError(f"arm {name} references an unknown family")
        expected_hard = _require_sha(
            arm.get("expected_training_hard_contract_sha256"), f"{name} hard contract"
        )
        if expected_hard != family.get("shared_plus_y_hard_contract_sha256"):
            raise ContractError(f"arm {name} does not reuse its family hard contract")
    if seen != set(expected_specs):
        raise ContractError("followup arm set differs from the preregistration")
    return data


def select_arm(manifest: dict[str, Any], pod: str, run_name: str) -> dict[str, Any]:
    matches = [a for a in manifest["arms"] if a["run_name"] == run_name]
    if len(matches) != 1:
        raise ContractError(f"unknown arm: {run_name}")
    arm = matches[0]
    if arm["pod"] != pod:
        raise ContractError(f"arm {run_name} belongs to {arm['pod']}, not {pod}")
    return arm


def artifact_input_paths(manifest: dict[str, Any], arm: dict[str, Any]) -> dict[str, tuple[Path, str]]:
    artifact_root = Path(manifest["runtime"]["artifact_root"]).resolve()
    family = manifest["families"][arm["family"]]
    fields = {
        "parent_checkpoint": ("parent_checkpoint", "parent_checkpoint_sha256"),
        "forehand_motion": ("forehand_motion", "forehand_motion_sha256"),
        "backhand_motion": ("backhand_motion", "backhand_motion_sha256"),
        "question_bank": ("question_bank", "question_bank_sha256"),
    }
    result: dict[str, tuple[Path, str]] = {}
    for label, (path_key, sha_key) in fields.items():
        relative = Path(family[path_key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"unsafe artifact-relative path: {relative}")
        path = (artifact_root / relative).resolve()
        if not is_within(path, artifact_root):
            raise ContractError(f"artifact escapes root: {path}")
        result[label] = (path, _require_sha(family[sha_key], f"{label} hash"))
    return result


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return str(value)
    return str(value)


def build_training_command(manifest: dict[str, Any], arm: dict[str, Any]) -> list[str]:
    runtime = manifest["runtime"]
    continuation = manifest["continuation_contract"]
    family = manifest["families"][arm["family"]]
    inputs = artifact_input_paths(manifest, arm)
    phases = json.dumps(family["strike_phase_per_clip"], separators=(",", ":"))
    return [
        "env", f"CUDA_VISIBLE_DEVICES={arm['gpu']}", "PYTHONUNBUFFERED=1",
        runtime["isaac_python"], "scripts/train.py", *manifest["base_recipe"],
        f"seed={arm['training_seed']}", f"num_envs={continuation['num_envs']}",
        f"max_iterations={continuation['max_iterations']}", f"run_name={arm['run_name']}",
        f"checkpoint_path={inputs['parent_checkpoint'][0]}",
        f"checkpoint_tolerant={format_scalar(continuation['checkpoint_tolerant'])}",
        "checkpoint_allow_missing_contract=true", "checkpoint_allow_contract_mismatch=false",
        "task.plant.zero_joint_friction=false",
        "++task.motion.allow_legacy_link_origin_velocity=true",
        f"motion_file={inputs['forehand_motion'][0]}",
        f"motion_file_2={inputs['backhand_motion'][0]}",
        f"task.racket.strike_phase_per_clip={phases}",
        f"++task.racket.question_bank={inputs['question_bank'][0]}",
        f"++task.racket.face_command_pairing={arm['face_command_pairing']}",
        f"++task.rewards.racket_guidance_weight={format_scalar(arm['racket_guidance_weight'])}",
    ]


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_launch_state(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def check_release_gate(arm: dict[str, Any]) -> None:
    gate = arm.get("release_gate")
    if not gate:
        return
    state_path = Path(gate["predecessor_launch_state"])
    log_path = Path(gate["predecessor_log"])
    checkpoint = Path(gate["required_terminal_checkpoint"])
    if not state_path.is_file() or not log_path.is_file() or not checkpoint.is_file():
        raise ContractError(f"{arm['run_name']} remains queue-only: predecessor terminal evidence missing")
    state = parse_launch_state(state_path)
    expected = str(gate["predecessor_recorded_pgid"])
    if state.get("pid") != expected or state.get("pgid") != expected:
        raise ContractError("predecessor launch state no longer binds the preregistered exact PGID")
    if f"run_name={gate['predecessor_run_name']}" not in state.get("command", ""):
        raise ContractError("predecessor launch state no longer binds its run name")
    if process_alive(int(expected)):
        raise ContractError(
            f"{arm['run_name']} remains queue-only while predecessor PGID {expected} is alive"
        )
    before = checkpoint.stat()
    time.sleep(2.0)
    after = checkpoint.stat()
    if before.st_size <= 0 or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError("predecessor terminal checkpoint is absent/unstable")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"Learning iteration\s+20998/20999", text):
        raise ContractError("predecessor log has not reached exact terminal iteration 20998")
    if re.search(
        r"Traceback|CUDA out of memory|OutOfMemory|Segmentation fault|\bNaN\b|\bInf\b",
        text,
        re.I,
    ):
        raise ContractError("predecessor log contains a terminal failure signature")


def gpu_capacity(gpu: int, max_before: int, minimum_free_mib: int) -> dict[str, Any]:
    free_raw = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(free_raw) != 1 or not free_raw[0].strip().isdigit():
        raise ContractError(f"cannot read free memory for GPU {gpu}")
    free_mib = int(free_raw[0].strip())
    pids_raw = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    # Some driver/nvidia-smi combinations repeat one compute PID (observed twice per
    # trainer on the Phase-1 Pods).  Capacity is about unique processes, so preserve
    # first-seen order while deduplicating before classifying trainers or enforcing 4/card.
    compute_pids: list[int] = []
    seen_compute_pids: set[int] = set()
    for line in pids_raw:
        if not line.strip().isdigit():
            continue
        pid = int(line.strip())
        if pid in seen_compute_pids:
            continue
        seen_compute_pids.add(pid)
        compute_pids.append(pid)
    trainer_pids: list[int] = []
    for pid in compute_pids:
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.is_file() and b"scripts/train.py" in cmdline.read_bytes():
            trainer_pids.append(pid)
    if len(trainer_pids) >= 4 or len(compute_pids) > max_before:
        raise ContractError(
            f"GPU {gpu} has no fourth-process slot: trainers={trainer_pids} compute={compute_pids}"
        )
    if free_mib < minimum_free_mib:
        raise ContractError(
            f"GPU {gpu} has only {free_mib} MiB free; requires at least {minimum_free_mib} MiB"
        )
    return {"gpu": gpu, "free_memory_mib": free_mib, "compute_pids": compute_pids,
            "trainer_pids": trainer_pids}


def verify_git_checkout(path: Path, expected_commit: str, label: str) -> None:
    if git_output(path, "rev-parse", "HEAD") != expected_commit:
        raise ContractError(f"{label} checkout is not at {expected_commit}")
    if git_output(path, "status", "--porcelain"):
        raise ContractError(f"{label} checkout is dirty: {path}")


def preflight(
    manifest: dict[str, Any], arm: dict[str, Any], *, config_path: Path, launcher_path: Path
) -> dict[str, Any]:
    runtime = manifest["runtime"]
    cadence = manifest["checkpoint_evaluation"]
    repo = Path(runtime["training_checkout"]).resolve()
    control = Path(runtime["external_control_root"]).resolve()
    if is_within(config_path, repo) or is_within(launcher_path, repo):
        raise ContractError("production manifest and launcher must live outside the training checkout")
    if not is_within(config_path, control) or not is_within(launcher_path, control):
        raise ContractError(f"production manifest and launcher must live under {control}")
    verify_git_checkout(repo, runtime["expected_training_commit"], "training")
    eval_root = Path(cadence["eval_checkout"]).resolve()
    verify_git_checkout(eval_root, cadence["expected_eval_commit"], "evaluation")
    wbt = repo / runtime["wbt_relative_path"]
    locked = repo / runtime["locked_launcher_relative_path"]
    worker = eval_root / cadence["worker_relative_path"]
    judge = eval_root / cadence["judge_relative_path"]
    fixed_files = [
        (locked, runtime["locked_launcher_sha256"], "locked launcher"),
        (worker, cadence["worker_sha256"], "curve worker"),
        (judge, cadence["judge_sha256"], "judge"),
    ]
    for path, expected, label in fixed_files:
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"{label} is missing or has the wrong SHA: {path}")
    for path in (Path(runtime["environment_file"]), Path(runtime["isaac_python"]),
                 Path(cadence["worker_python"])):
        if not path.is_file():
            raise ContractError(f"required runtime file is missing: {path}")
    actual_inputs: dict[str, str] = {}
    for label, (path, expected) in artifact_input_paths(manifest, arm).items():
        if not path.is_file():
            raise ContractError(f"missing {label}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ContractError(f"{label} SHA mismatch: {actual} != {expected}")
        actual_inputs[label] = actual
    check_release_gate(arm)
    artifact_run = Path(runtime["artifact_root"]) / "runs" / arm["run_name"]
    training_matches = list(
        (wbt / "logs/rsl_rl/agibot_a3_hope_virtualball").glob(f"*_{arm['run_name']}")
    )
    if artifact_run.exists() or training_matches:
        raise ContractError(
            f"run name/dir already exists for {arm['run_name']}: "
            f"artifact={artifact_run.exists()} training={training_matches}"
        )
    capacity = gpu_capacity(
        arm["gpu"], runtime["max_compute_processes_before_launch_per_gpu"],
        runtime["minimum_free_gpu_memory_mib"],
    )
    return {
        "training_checkout": str(repo), "eval_checkout": str(eval_root),
        "wbt": str(wbt), "locked_launcher": str(locked), "worker": str(worker),
        "judge": str(judge), "artifact_run": str(artifact_run),
        "verified_inputs": actual_inputs, "gpu_capacity": capacity,
    }


def exact_stop_new_trainer(state_path: Path, run_name: str) -> None:
    state = parse_launch_state(state_path)
    pid_raw, pgid_raw = state.get("pid", ""), state.get("pgid", "")
    if not pid_raw.isdigit() or pid_raw != pgid_raw:
        raise ContractError("refusing cleanup: new launch state does not bind pid==pgid")
    pid = int(pid_raw)
    if not process_alive(pid):
        return
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if not proc_cmdline.is_file() or f"run_name={run_name}".encode() not in proc_cmdline.read_bytes():
        raise ContractError("refusing cleanup: /proc identity does not bind the new run name")
    os.killpg(pid, signal.SIGTERM)
    for _ in range(10):
        if not process_alive(pid):
            return
        time.sleep(1)
    os.killpg(pid, signal.SIGKILL)


def emitted_training_contract(log_path: Path) -> Path:
    matches = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if CONTRACT_MARKER in line:
            matches.append(line.split(CONTRACT_MARKER, 1)[1].strip())
    if len(matches) != 1:
        raise ContractError(f"expected exactly one emitted hard-contract marker, got {matches}")
    path = Path(matches[0]).resolve()
    if path.name != "training_contract.json" or path.parent.name != "params" or not path.is_file():
        raise ContractError(f"invalid emitted training contract path: {path}")
    return path


def verify_training_contract(path: Path, manifest: dict[str, Any], arm: dict[str, Any]) -> str:
    actual_sha = sha256_file(path)
    if actual_sha != arm["expected_training_hard_contract_sha256"]:
        raise ContractError(
            f"hard-contract SHA mismatch for {arm['run_name']}: {actual_sha} != "
            f"{arm['expected_training_hard_contract_sha256']}"
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    family = manifest["families"][arm["family"]]
    if contract.get("face_command_pairing") != "shared_plus_y":
        raise ContractError("emitted contract does not use shared_plus_y")
    if contract.get("motion_allow_legacy_link_origin_velocity") is not True:
        raise ContractError("emitted causal contract lost legacy motion-velocity semantics")
    clips = [item.get("sha256") for item in contract.get("motion_clips", [])]
    if clips != [family["forehand_motion_sha256"], family["backhand_motion_sha256"]]:
        raise ContractError("emitted contract motion hashes disagree with preregistration")
    if contract.get("question_bank", {}).get("sha256") != family["question_bank_sha256"]:
        raise ContractError("emitted contract question-bank hash disagrees with preregistration")
    return actual_sha


def materialize_cadence(
    manifest: dict[str, Any], arm: dict[str, Any], pre: dict[str, Any], training_run: Path,
    external_run: Path,
) -> tuple[Path, Path, list[str]]:
    q10 = manifest["checkpoint_evaluation"]["q10_screen"]
    inputs = artifact_input_paths(manifest, arm)
    parent_source, parent_sha = inputs["parent_checkpoint"]
    jobs = []
    for milestone in q10["milestones"]:
        jobs.append({
            "id": f"{arm['run_name']}_{milestone}_clean_q10",
            "barrier_id": f"{arm['run_name']}_{milestone}",
            "run_dir": str(training_run),
            "checkpoint": str(training_run / f"model_{milestone}.pt"),
            "gpu": arm["gpu"], "seed": q10["seed"], "noise_scales": q10["noise_scales"],
            "extra_args": q10["extra_args"], "training_seed": arm["training_seed"],
            "training_kind": "continuation", "training_family": arm["family"],
            "face_command_pairing": arm["face_command_pairing"],
            "racket_guidance_weight": arm["racket_guidance_weight"],
            "zero_joint_friction": False, "evaluation_role": q10["evaluation_role"],
            "expected_evaluation_contract_exact": False, "formal_target": False,
            "screen_only": True,
        })
    q10_manifest = {
        "schema_version": 1,
        "purpose": "causal followup first-post-parent-to-terminal clean q10 curve; direction only",
        "queue": "causal_followup", "pod": arm["pod"],
        "checkpoint_readiness_barrier": {
            "ordering": "milestone_major", "scope": "this exact preregistered arm",
            "semantics": "17k/18k/19k/20k/20998, one stable checkpoint at a time; parent remains unjudged to preserve lineage",
        },
        "screen_policy": {
            "seed": q10["seed"], "noise_scales": q10["noise_scales"],
            "schedule_k": q10["schedule_k"], "attempts_per_side": q10["attempts_per_side"],
            "screen_only": True, "stop_or_promote_allowed": False,
            "decision_followup": manifest["continuation_contract"]["decision_contract"],
        },
        "judge_script_sha256": manifest["checkpoint_evaluation"]["judge_sha256"],
        "training_checkout": manifest["runtime"]["training_checkout"],
        "expected_training_commit": manifest["runtime"]["expected_training_commit"],
        "source_preregistration_id": manifest["manifest_id"],
        "source_arm_contract_sha256": canonical_sha256(arm),
        "parent_reference": {"path": str(parent_source), "sha256": parent_sha,
                             "judged_under_new_run_contract": False,
                             "reason": "copying the legacy parent beside the new sidecar would launder lineage"},
        "jobs": jobs,
    }
    q10_path = external_run / "checkpoint_cadence_q10.json"
    atomic_json(q10_path, q10_manifest)
    q50_template = {
        "schema_version": 1, "status": "inactive_requires_trigger_evidence",
        "arm": arm["run_name"], "policy": manifest["checkpoint_evaluation"]["q50_decision_paper"],
        "q10_manifest": str(q10_path), "q10_manifest_sha256": sha256_file(q10_path),
        "jobs": [],
    }
    q50_path = external_run / "checkpoint_decision_q50.template.json"
    atomic_json(q50_path, q50_template)
    state_dir = external_run / "checkpoint_cadence_q10_state"
    worker_command = [
        manifest["checkpoint_evaluation"]["worker_python"], pre["worker"],
        "--manifest", str(q10_path), "--judge-script", pre["judge"],
        "--state-dir", str(state_dir), "--max-active-cpu", "6",
        "--export-timeout-s", "1200", "--poll-s", "5",
        "--wait-for-checkpoints", "--checkpoint-wait-timeout-s", "0",
        "--checkpoint-poll-s", "15", "--checkpoint-stable-s", "5",
    ]
    return q10_path, q50_path, worker_command


def start_q10_worker(command: list[str], external_run: Path) -> dict[str, Any]:
    log_path = external_run / "checkpoint_cadence_q10.worker.log"
    state_path = external_run / "checkpoint_cadence_q10.worker.launch.json"
    with log_path.open("wb", buffering=0) as stream:
        proc = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pgid = os.getpgid(proc.pid)
    if pgid != proc.pid:
        raise ContractError(f"curve worker is not isolated: pid={proc.pid} pgid={pgid}")
    state = {
        "pid": proc.pid, "pgid": pgid, "command": command,
        "command_sha256": canonical_sha256(command), "log": str(log_path),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(state_path, state)
    time.sleep(1.0)
    rc = proc.poll()
    if rc is not None:
        raise ContractError(f"curve worker exited during startup with rc={rc}; see {log_path}")
    return {**state, "state_path": str(state_path)}


def exact_stop_new_worker(worker: dict[str, Any]) -> None:
    pid, pgid = worker.get("pid"), worker.get("pgid")
    if not isinstance(pid, int) or pid != pgid:
        raise ContractError("refusing worker cleanup: recorded pid/pgid is not exact")
    if not process_alive(pid):
        return
    state_path = Path(worker.get("state_path", ""))
    if not state_path.is_file():
        raise ContractError("refusing worker cleanup: owned state sidecar is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("pid") != pid or state.get("pgid") != pid:
        raise ContractError("refusing worker cleanup: sidecar no longer binds the exact PGID")
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    command = worker.get("command")
    if (
        not proc_cmdline.is_file()
        or not isinstance(command, list)
        or len(command) < 2
        or str(command[1]).encode() not in proc_cmdline.read_bytes()
    ):
        raise ContractError("refusing worker cleanup: /proc identity does not bind the curve worker")
    os.killpg(pid, signal.SIGTERM)
    for _ in range(10):
        if not process_alive(pid):
            return
        time.sleep(1)
    os.killpg(pid, signal.SIGKILL)


def launch(
    manifest: dict[str, Any], arm: dict[str, Any], pre: dict[str, Any], *,
    manifest_sha: str, launcher_sha: str,
) -> dict[str, Any]:
    runtime = manifest["runtime"]
    external_run = Path(pre["artifact_run"])
    contract_path = external_run / runtime["launch_contract_basename"]
    log_path = external_run / runtime["training_log_basename"]
    state_path = external_run / runtime["launch_state_basename"]
    command = build_training_command(manifest, arm)
    claim_run_directory(external_run)
    record: dict[str, Any] = {
        "schema_version": 1, "status": "preflight_passed", "arm": arm,
        "manifest_id": manifest["manifest_id"], "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha, "training_command": command,
        "training_command_sha256": canonical_sha256(command), "preflight": pre,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(contract_path, record)
    env = os.environ.copy()
    env.update(
        KIT_BOOT_MARKER=runtime["boot_marker"],
        KIT_BOOT_TIMEOUT_S=str(runtime["boot_timeout_seconds"]),
        KIT_BOOT_POLL_S=str(runtime["boot_poll_seconds"]),
    )
    wrapper = [
        "bash", "-c", 'set -euo pipefail; source "$1"; shift; exec "$@"',
        "phase1-causal-followup", runtime["environment_file"], pre["locked_launcher"],
        str(log_path), *command,
    ]
    worker: dict[str, Any] | None = None
    try:
        subprocess.run(wrapper, cwd=pre["wbt"], env=env, check=True)
        state = parse_launch_state(state_path)
        if state.get("pid") != state.get("pgid") or "ready_utc" not in state:
            raise ContractError("frozen launcher did not record a ready isolated pid==pgid")
        hard_path = emitted_training_contract(log_path)
        hard_sha = verify_training_contract(hard_path, manifest, arm)
        training_run = hard_path.parent.parent
        q10_path, q50_path, worker_command = materialize_cadence(
            manifest, arm, pre, training_run, external_run
        )
        worker = start_q10_worker(worker_command, external_run)
        record.update(
            status="ready_verified_with_q10_worker",
            trainer={"pid": int(state["pid"]), "pgid": int(state["pgid"]),
                     "state_path": str(state_path), "log": str(log_path)},
            emitted_training_contract={"path": str(hard_path), "sha256": hard_sha},
            checkpoint_cadence_q10={"path": str(q10_path), "sha256": sha256_file(q10_path)},
            checkpoint_decision_q50_template={"path": str(q50_path), "sha256": sha256_file(q50_path)},
            q10_worker=worker,
            ready_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        atomic_json(contract_path, record)
        return record
    except Exception as error:
        record.update(status="failed_preserved", failure=repr(error),
                      failed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        preserve_error = None
        try:
            atomic_json(contract_path, record)
        except Exception as failure:
            preserve_error = failure
        cleanup_errors = []
        if worker is not None:
            try:
                exact_stop_new_worker(worker)
            except Exception as failure:
                cleanup_errors.append(f"worker cleanup: {failure!r}")
        if state_path.is_file():
            try:
                exact_stop_new_trainer(state_path, arm["run_name"])
            except Exception as failure:
                cleanup_errors.append(f"trainer cleanup: {failure!r}")
        if preserve_error is not None or cleanup_errors:
            raise ContractError(
                f"launch failed with {error!r}; preserve_error={preserve_error!r}; "
                f"cleanup_errors={cleanup_errors!r}"
            ) from error
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--pod", required=True, choices=("pod1", "pod2"))
    parser.add_argument("--arm", required=True)
    parser.add_argument("mode", choices=("validate", "launch"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = args.config.resolve()
    launcher = Path(__file__).resolve()
    for value, label in ((args.expected_config_sha256, "config"),
                         (args.expected_launcher_sha256, "launcher")):
        _require_sha(value, f"expected {label} hash")
    actual_config_sha = sha256_file(config)
    actual_launcher_sha = sha256_file(launcher)
    if actual_config_sha != args.expected_config_sha256:
        raise ContractError("config SHA differs from the explicit launch authorization")
    if actual_launcher_sha != args.expected_launcher_sha256:
        raise ContractError("launcher SHA differs from the explicit launch authorization")
    manifest = load_manifest(config)
    arm = select_arm(manifest, args.pod, args.arm)
    pre = preflight(manifest, arm, config_path=config, launcher_path=launcher)
    if args.mode == "validate":
        print(json.dumps({"status": "validated_no_writes", "arm": args.arm, "preflight": pre},
                         indent=2, sort_keys=True))
        return 0
    result = launch(
        manifest, arm, pre, manifest_sha=actual_config_sha, launcher_sha=actual_launcher_sha
    )
    print(json.dumps({
        "status": result["status"], "arm": args.arm,
        "trainer": result["trainer"], "q10_worker": result["q10_worker"],
        "launch_contract": str(Path(pre["artifact_run"]) / manifest["runtime"]["launch_contract_basename"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
