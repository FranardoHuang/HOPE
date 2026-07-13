#!/usr/bin/env python3
"""Fail-closed A0/A1 non-striking-arm imitation screen.

Default invocation is plan-only.  Simulator launch additionally requires root,
an exact confirmation token, an empty preregistered GPU, and no pre-existing
claim directory.  The script never discovers or signals a process group; the
reviewed per-arm Kit wrapper owns only the newly created arm PGID during boot.
There is no robot command in this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
FAILURE_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemory|Segmentation fault|\bNaN\b|\bInf\b|\bKilled\b|malloc|bad_alloc",
    re.IGNORECASE,
)
MANIFEST_ID = "phase1-non-striking-arm-imitation-a0-a1-single-seed-20260714-v1"
CELL_IDS = ("A0", "A1")
BODY_TERMS = (
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
)
LEFT_ARM = (
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
)


class ContractError(RuntimeError):
    """A preregistered source/runtime/scientific invariant was violated."""


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
    value = str(value)
    if not SHA_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ContractError(
            f"{label} keys changed: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def write_json_exclusive(path: Path, value: Any) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read manifest: {exc}") from exc
    validate_manifest(value)
    return value


def _expected_body_contracts() -> dict[str, dict[str, list[str]]]:
    seven = ["torso_Link", *LEFT_ARM, "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link"]
    six = [name for name in seven if name != "right_wrist_yaw_Link"]
    a0 = {
        "motion_body_pos": seven,
        "motion_body_ori": six,
        "motion_body_lin_vel": seven,
        "motion_body_ang_vel": six,
    }
    a1 = {name: [body for body in bodies if body not in LEFT_ARM] for name, bodies in a0.items()}
    return {"A0": a0, "A1": a1}


def cell_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise ContractError("manifest must contain exactly A0 and A1")
    result = {str(cell.get("cell_id")): cell for cell in cells if isinstance(cell, dict)}
    if tuple(result) != CELL_IDS:
        raise ContractError("cell order and identity must be exactly A0, A1")
    return result


def validate_manifest(data: dict[str, Any]) -> None:
    if data.get("schema_version") != 1 or data.get("manifest_id") != MANIFEST_ID:
        raise ContractError("unexpected manifest schema or identity")
    if data.get("status") != "machine_preregistered_plan_only_root_launch_switch_required":
        raise ContractError("manifest must remain plan-only until the explicit root switch")
    if data.get("simulation_only") is not True or data.get("real_robot_commands_forbidden") is not True:
        raise ContractError("manifest must be simulator-only and forbid robot commands")
    if data.get("automatic_retry_forbidden") is not True:
        raise ContractError("automatic retry must remain forbidden")
    if data.get("second_seed_forbidden_before_paired_checkpoint_decision") is not True:
        raise ContractError("first-round seed replication must remain forbidden")

    source = data.get("source")
    runtime = data.get("runtime")
    inputs = data.get("inputs")
    shared = data.get("shared_training_contract")
    invariants = data.get("invariants")
    evaluation = data.get("evaluation")
    if not all(isinstance(v, dict) for v in (source, runtime, inputs, shared, invariants, evaluation)):
        raise ContractError("source/runtime/inputs/shared/invariants/evaluation must be objects")
    if not COMMIT_RE.fullmatch(str(source.get("expected_training_commit", ""))):
        raise ContractError("training commit must be full lowercase SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("expected_training_tree", ""))):
        raise ContractError("training tree must be full lowercase tree SHA")
    critical = source.get("critical_files")
    if not isinstance(critical, dict) or "scripts/train.py" not in critical:
        raise ContractError("critical source map is incomplete")
    for relative, digest in critical.items():
        rel = Path(str(relative))
        if rel.is_absolute() or ".." in rel.parts:
            raise ContractError(f"unsafe critical source path: {relative}")
        require_sha(digest, f"critical source {relative}")
    ignored = source.get("ignored_runtime_asset")
    if not isinstance(ignored, dict):
        raise ContractError("ignored runtime asset contract is missing")
    for key in ("tree_content_sha256",):
        require_sha(ignored.get(key), f"ignored runtime asset {key}")
    if ignored.get("symlinks_forbidden") is not True or ignored.get("target_must_be_gitignored") is not True:
        raise ContractError("ignored asset must remain copied, gitignored, and symlink-free")

    expected_runtime = {
        "pod": "pod1",
        "gpu": 0,
        "maximum_owned_trainers_on_gpu": 2,
        "initial_gpu_must_have_zero_compute_processes": True,
        "kit_boot_marker": "Learning iteration",
        "root_launch_confirmation": "ROOT_APPROVES_SIM_ONLY_A0_A1_V1",
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise ContractError(f"runtime {key} changed")
    require_sha(runtime.get("training_environment_sha256"), "training environment")
    if int(runtime.get("minimum_free_gpu_memory_mib_before_each_launch", 0)) < 4096:
        raise ContractError("GPU free-memory launch floor is too low")
    if int(runtime.get("minimum_host_available_memory_mib", 0)) < 32768:
        raise ContractError("host-memory launch floor is too low")

    require_exact_keys(inputs, {"forehand_motion", "backhand_motion", "schema3_train_bank"}, "inputs")
    for name in ("forehand_motion", "backhand_motion"):
        item = inputs[name]
        if not isinstance(item, dict) or set(item) != {"relative_path", "sha256"}:
            raise ContractError(f"{name} contract changed")
        require_sha(item.get("sha256"), name)
    bank = inputs["schema3_train_bank"]
    if not isinstance(bank, dict) or set(bank) != {
        "path", "sha256", "physics_contract_sha256", "source_family_sha256"
    }:
        raise ContractError("schema3 train-bank contract changed")
    for key in ("sha256", "physics_contract_sha256", "source_family_sha256"):
        require_sha(bank.get(key), f"train bank {key}")
    if not Path(str(bank.get("path", ""))).is_absolute():
        raise ContractError("train-bank path must be absolute")

    expected_shared = {
        "training_seed": 17,
        "initialization": "fresh",
        "num_envs": 4096,
        "max_iterations": 1001,
        "save_interval": 100,
        "relative_checkpoint_milestones": [200, 500, 1000],
        "expected_terminal_checkpoint_iteration": 1000,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1.0, -1.0],
        "zero_joint_friction": True,
        "motion_kinematics_exact": True,
        "question_bank_schema_version": 3,
        "question_bank_split": "train",
        "actor_observation_contract": "deploy_parity_face179",
        "actor_observation_dim": 179,
        "action_dim": 31,
        "strike_phase_per_clip": [0.471, 0.338],
        "event_timing_mode": "disabled",
    }
    for key, expected in expected_shared.items():
        if shared.get(key) != expected:
            raise ContractError(f"shared training contract {key} changed")
    recipe = shared.get("base_recipe")
    if not isinstance(recipe, list) or not recipe:
        raise ContractError("base recipe must be a non-empty argv list")
    if any("free_non_striking_arm_mimic" in str(item) for item in recipe):
        raise ContractError("causal mask may appear only in the per-cell command")
    required_recipe = {
        "task.actions.qdes_clamp=true",
        "task.plant.zero_joint_friction=true",
        "++task.motion.allow_legacy_link_origin_velocity=false",
        "++task.motion.event_timing_mode=disabled",
        "task.rewards.free_wrist_ori_mimic=true",
        "++task.rewards.free_wrist_vel_mimic=false",
    }
    if not required_recipe.issubset(set(str(v) for v in recipe)):
        raise ContractError("base recipe lost a plant/action/motion/wrist invariant")

    cells = cell_map(data)
    expected_bodies = _expected_body_contracts()
    for cell_id, flag in (("A0", False), ("A1", True)):
        cell = cells[cell_id]
        if cell.get("free_non_striking_arm_mimic") is not flag:
            raise ContractError(f"{cell_id} mask flag changed")
        if cell.get("body_names") != expected_bodies[cell_id]:
            raise ContractError(f"{cell_id} four-term body_names contract changed")
        if not RUN_RE.fullmatch(str(cell.get("run_name", ""))) or "seed17" not in cell["run_name"]:
            raise ContractError(f"{cell_id} run name must bind seed17")
    for term in BODY_TERMS:
        a0 = cells["A0"]["body_names"][term]
        a1 = cells["A1"]["body_names"][term]
        if [name for name in a0 if name not in LEFT_ARM] != a1:
            raise ContractError(f"{term} changes something other than the left non-racket arm")
    if invariants.get("only_causal_command_difference") != (
        "++task.rewards.free_non_striking_arm_mimic=false|true"
    ):
        raise ContractError("causal difference declaration changed")
    for key in (
        "right_striking_arm_and_torso_unchanged",
        "reward_weights_and_stds_unchanged",
        "joint_and_action_limits_unchanged",
        "torque_and_contact_terms_unchanged",
        "self_collision_and_table_net_clearance_unchanged",
        "terminations_and_safety_stop_unchanged",
        "fresh_init_seed_bank_motion_budget_and_checkpoint_cadence_identical",
    ):
        if invariants.get(key) is not True:
            raise ContractError(f"invariant {key} must remain true")
    if evaluation.get("automatic_judge_launch") is not False:
        raise ContractError("training launcher may not start a judge")
    if evaluation.get("checkpoint_milestones") != [200, 500, 1000]:
        raise ContractError("evaluation milestone cadence changed")
    if data.get("a2_fixed_budget_reallocation", {}).get("status") != "blocked_not_materialized":
        raise ContractError("A2 must remain blocked in this direct-mask preregistration")
    design = data.get("launch_design")
    if not isinstance(design, dict) or design.get("separate_25_iteration_training_smoke") is not False:
        raise ContractError("this v1 uses boot-to-learning as smoke, not throwaway 25-update jobs")


def _repo_for_source(manifest: dict[str, Any]) -> Path:
    runtime_checkout = Path(manifest["source"]["training_checkout"])
    if runtime_checkout.exists():
        return runtime_checkout
    return Path(__file__).resolve().parents[1]


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_static_source(manifest: dict[str, Any]) -> dict[str, Any]:
    repo = _repo_for_source(manifest)
    source = manifest["source"]
    commit = source["expected_training_commit"]
    try:
        tree = git_output(repo, "rev-parse", f"{commit}^{{tree}}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"training source commit is unavailable in {repo}") from exc
    if tree != source["expected_training_tree"]:
        raise ContractError("training source tree changed")
    checked = {}
    wbt_rel = Path(source["wbt_relative_path"])
    for relative, expected in source["critical_files"].items():
        git_path = (wbt_rel / relative).as_posix()
        try:
            content = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{git_path}"])
        except subprocess.CalledProcessError as exc:
            raise ContractError(f"critical source is absent from commit: {git_path}") from exc
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ContractError(f"critical source SHA changed: {git_path}")
        checked[relative] = actual
    return {"commit": commit, "tree": tree, "critical_files": checked}


def input_paths(manifest: dict[str, Any]) -> dict[str, tuple[Path, str]]:
    root = Path(manifest["runtime"]["source_asset_root"])
    return {
        "forehand_motion": (root / manifest["inputs"]["forehand_motion"]["relative_path"], manifest["inputs"]["forehand_motion"]["sha256"]),
        "backhand_motion": (root / manifest["inputs"]["backhand_motion"]["relative_path"], manifest["inputs"]["backhand_motion"]["sha256"]),
        "schema3_train_bank": (Path(manifest["inputs"]["schema3_train_bank"]["path"]), manifest["inputs"]["schema3_train_bank"]["sha256"]),
    }


def build_command(manifest: dict[str, Any], cell_id: str) -> list[str]:
    cell = cell_map(manifest)[cell_id]
    shared = manifest["shared_training_contract"]
    paths = input_paths(manifest)
    command = [
        manifest["runtime"]["isaac_python"],
        "scripts/train.py",
        *[str(value) for value in shared["base_recipe"]],
        f"seed={shared['training_seed']}",
        f"num_envs={shared['num_envs']}",
        f"max_iterations={shared['max_iterations']}",
        f"algo.runner.save_interval={shared['save_interval']}",
        f"run_name={cell['run_name']}",
        "checkpoint_path=null",
        "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=false",
        f"motion_file={paths['forehand_motion'][0]}",
        f"motion_file_2={paths['backhand_motion'][0]}",
        "task.racket.strike_phase_per_clip=[0.471,0.338]",
        f"++task.racket.question_bank={paths['schema3_train_bank'][0]}",
        "++task.racket.face_command_pairing=shared_plus_y",
        f"++task.rewards.free_non_striking_arm_mimic={str(cell['free_non_striking_arm_mimic']).lower()}",
    ]
    if sum("free_non_striking_arm_mimic=" in item for item in command) != 1:
        raise ContractError("constructed command must contain exactly one causal mask flag")
    forbidden = ("ros2", "run_deploy", "joint_command", "real_robot", "/dev/")
    if any(any(token in item.lower() for token in forbidden) for item in command):
        raise ContractError("constructed command contains a forbidden robot/runtime token")
    return command


def normalized_paired_command(manifest: dict[str, Any], cell_id: str) -> list[str]:
    return [
        "run_name=<paired>" if item.startswith("run_name=") else
        "++task.rewards.free_non_striking_arm_mimic=<paired>"
        if item.startswith("++task.rewards.free_non_striking_arm_mimic=") else item
        for item in build_command(manifest, cell_id)
    ]


def build_plan(manifest: dict[str, Any], manifest_path: Path, launcher_path: Path) -> dict[str, Any]:
    if normalized_paired_command(manifest, "A0") != normalized_paired_command(manifest, "A1"):
        raise ContractError("A0/A1 commands differ outside run_name and the one mask flag")
    return {
        "artifact_kind": "phase1_non_striking_arm_a01_plan_only",
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "launcher_sha256": sha256_file(launcher_path),
        "source": verify_static_source(manifest),
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "writes_or_launches_performed": False,
        "commands": {cell: shlex.join(build_command(manifest, cell)) for cell in CELL_IDS},
        "runtime_smoke": "locked Kit boot to first Learning iteration marker; no throwaway 25-update pair",
        "mechanism_milestones": [200, 500, 1000],
        "launch_invocation": shlex.join([
            manifest["runtime"]["isaac_python"], str(launcher_path),
            "--manifest", str(manifest_path), "--mode", "launch",
            "--root-confirm", manifest["runtime"]["root_launch_confirmation"],
        ]),
    }


def asset_tree_content(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"ignored asset root is missing or a symlink: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"ignored asset contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"ignored asset contains special entry: {path}")
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "tree_content_sha256": canonical_sha256({"files": rows}),
    }


def build_training_environment(manifest: dict[str, Any], wbt: Path) -> dict[str, str]:
    runtime = manifest["runtime"]
    if (wbt / "setup_train_env.local.sh").exists():
        raise ContractError("untracked setup_train_env.local.sh is forbidden")
    isaaclab = Path(runtime["isaaclab_root"]).resolve()
    paths = [
        (wbt / "source/whole_body_tracking").resolve(),
        isaaclab / "source/isaaclab",
        isaaclab / "source/isaaclab_tasks",
        isaaclab / "source/isaaclab_assets",
        isaaclab / "source/isaaclab_rl",
    ]
    for path in paths:
        if not path.is_dir():
            raise ContractError(f"PYTHONPATH entry is missing: {path}")
    pythonpath = ":".join(str(path) for path in paths)
    env = {
        "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "LOGNAME": "root",
        "USER": "root", "SHELL": "/bin/bash",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
        "HOPE_ISAAC_PYTHON": runtime["isaac_python"], "HOPE_ISAACLAB_ROOT": str(isaaclab),
        "HOPE_WBT_PYTHONPATH": pythonpath, "PYTHONPATH": pythonpath,
        "OMNI_KIT_ACCEPT_EULA": "YES", "TMPDIR": "/workspace/tmp",
        "PIP_CACHE_DIR": "/workspace/.cache/pip", "XDG_CACHE_HOME": "/workspace/.cache",
        "WANDB_DIR": "/workspace/codexschema/.wandb", "WANDB_ENTITY": "BerkeleyPingPong",
        "WANDB_REGISTRY_ORG": "dongc_1-university-of-california-berkeley-org",
        "WANDB_PROJECT": "hope_wbc", "WANDB_MOTION_PROJECT": "csv_to_npz",
        "CUDA_VISIBLE_DEVICES": str(runtime["gpu"]), "PYTHONUNBUFFERED": "1",
    }
    for key in ("TMPDIR", "XDG_CACHE_HOME", "WANDB_DIR"):
        if not Path(env[key]).is_dir():
            raise ContractError(f"runtime directory is missing: {env[key]}")
    if canonical_sha256(env) != runtime["training_environment_sha256"]:
        raise ContractError("deterministic training environment SHA changed")
    return env


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    free = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(free) != 1 or not free[0].strip().isdigit():
        raise ContractError("cannot read exact GPU free memory")
    raw_pids = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    pids = sorted({int(item.strip()) for item in raw_pids if item.strip().isdigit()})
    trainers = []
    for pid in pids:
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.is_file() and b"scripts/train.py" in cmdline.read_bytes():
            trainers.append(pid)
    return {"gpu": gpu, "free_memory_mib": int(free[0]), "compute_pids": pids, "trainer_pids": trainers}


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ContractError("/proc/meminfo lacks MemAvailable")


def verify_runtime(manifest: dict[str, Any], *, require_initial_empty: bool) -> dict[str, Any]:
    source = manifest["source"]
    runtime = manifest["runtime"]
    checkout = Path(source["training_checkout"]).resolve()
    try:
        head = git_output(checkout, "rev-parse", "HEAD")
        tree = git_output(checkout, "rev-parse", "HEAD^{tree}")
        dirty = git_output(checkout, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("training checkout is missing or unreadable") from exc
    if head != source["expected_training_commit"] or tree != source["expected_training_tree"] or dirty:
        raise ContractError("training checkout must be exact and clean")
    wbt = checkout / source["wbt_relative_path"]
    for relative, expected in source["critical_files"].items():
        path = wbt / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"critical training source changed: {path}")
    ignored = source["ignored_runtime_asset"]
    target = (wbt / ignored["relative_path"]).resolve()
    restore_checkout = Path(ignored["restore_source_checkout"]).resolve()
    if git_output(restore_checkout, "rev-parse", "HEAD") != ignored["restore_source_commit"]:
        raise ContractError("ignored-asset restore checkout commit changed")
    if git_output(restore_checkout, "status", "--porcelain"):
        raise ContractError("ignored-asset restore checkout is dirty")
    restore = (restore_checkout / ignored["restore_source_relative_path"]).resolve()
    expected_asset = {
        "file_count": ignored["file_count"],
        "total_file_bytes": ignored["total_file_bytes"],
        "tree_content_sha256": ignored["tree_content_sha256"],
    }
    if asset_tree_content(target) != expected_asset or asset_tree_content(restore) != expected_asset:
        raise ContractError("ignored A3 asset target/restore tree changed")
    if subprocess.run(
        ["git", "-C", str(checkout), "check-ignore", "-q", str(target.relative_to(checkout))],
        check=False,
    ).returncode != 0:
        raise ContractError("restored A3 asset is not gitignored")

    verified_inputs = {}
    for name, (path, expected) in input_paths(manifest).items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"{name} is missing or changed: {path}")
        verified_inputs[name] = {"path": str(path), "sha256": expected}
    python = Path(runtime["isaac_python"])
    locked = wbt / runtime["locked_launcher_relative_path"]
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ContractError("Isaac Python is missing/not executable")
    if not locked.is_file() or not os.access(locked, os.X_OK):
        raise ContractError("reviewed Kit launcher is missing/not executable")
    environment = build_training_environment(manifest, wbt)
    module = subprocess.check_output(
        [str(python), "-c", "import importlib.util,pathlib;s=importlib.util.find_spec('whole_body_tracking');print(pathlib.Path(s.origin).resolve())"],
        cwd=wbt, env=environment, text=True, stderr=subprocess.STDOUT,
    ).strip()
    expected_module_root = (wbt / "source/whole_body_tracking/whole_body_tracking").resolve()
    try:
        Path(module).resolve().relative_to(expected_module_root)
    except ValueError as exc:
        raise ContractError(f"whole_body_tracking resolves outside exact source: {module}") from exc
    if available_memory_mib() < runtime["minimum_host_available_memory_mib"]:
        raise ContractError("host available memory is below preregistered floor")
    gpu = gpu_snapshot(runtime["gpu"])
    if require_initial_empty and gpu["compute_pids"]:
        raise ContractError(f"preregistered GPU is not initially empty: {gpu['compute_pids']}")
    return {
        "checkout": checkout, "wbt": wbt, "python": python, "locked": locked,
        "environment": environment, "training_module_path": module,
        "verified_inputs": verified_inputs, "ignored_asset": {"target": str(target), **expected_asset},
        "gpu_snapshot": gpu,
    }


def parse_launch_state(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def locate_training_run(wbt: Path, run_name: str, timeout_s: int = 60) -> Path:
    root = wbt / "logs/rsl_rl/agibot_a3_hope_virtualball"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hits = sorted(
            path for path in root.glob(f"*_{run_name}")
            if path.is_dir() and (path / "params/training_contract.json").is_file()
        ) if root.is_dir() else []
        if len(hits) == 1:
            return hits[0].resolve()
        if len(hits) > 1:
            raise ContractError(f"run_name maps to multiple training directories: {run_name}")
        time.sleep(2)
    raise ContractError(f"training run directory/contract did not materialize: {run_name}")


def verify_hard_contract(path: Path, manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read emitted training contract: {exc}") from exc
    shared = manifest["shared_training_contract"]
    expected = {
        "schema_version": 3,
        "actor_obs_contract": shared["actor_observation_contract"],
        "actor_obs_total_dim": shared["actor_observation_dim"],
        "face_command_pairing": shared["face_command_pairing"],
        "mount_normal_sign_per_clip": shared["mount_normal_sign_per_clip"],
        "strike_phase_per_clip": shared["strike_phase_per_clip"],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
    }
    for key, wanted in expected.items():
        if contract.get(key) != wanted:
            raise ContractError(f"emitted hard contract {key} changed")
    if contract.get("motion_event_timing") != {"mode": "disabled"}:
        raise ContractError("T1 timing was unexpectedly enabled")
    if len(contract.get("joint_names", [])) != 31 or len(contract.get("action_joint_ids", [])) != 31:
        raise ContractError("hard contract does not bind 31 joints/actions")
    friction = contract.get("joint_friction_coefficients")
    if not isinstance(friction, list) or len(friction) != 31 or any(float(v) != 0.0 for v in friction):
        raise ContractError("hard contract is not 31/31 zero-friction")
    clips = contract.get("motion_clips")
    if not isinstance(clips, list) or [item.get("sha256") for item in clips] != [
        manifest["inputs"]["forehand_motion"]["sha256"],
        manifest["inputs"]["backhand_motion"]["sha256"],
    ]:
        raise ContractError("hard contract motion order/SHA changed")
    bank = contract.get("question_bank")
    expected_bank = manifest["inputs"]["schema3_train_bank"]
    if not isinstance(bank, dict) or any(bank.get(key) != expected_bank[key] for key in (
        "sha256", "physics_contract_sha256", "source_family_sha256"
    )):
        raise ContractError("hard contract train-bank binding changed")
    if bank.get("schema_version") != 3 or bank.get("split") != "train" or bank.get("exact") is not True:
        raise ContractError("hard contract train bank is not exact schema3/train")
    return sha256_file(path), contract


def verify_mask_log(log_path: Path, cell_id: str) -> list[str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    markers = [line.strip() for line in text.splitlines() if "left non-striking arm imitation removed" in line]
    if cell_id == "A0" and markers:
        raise ContractError("A0 log unexpectedly applied the A1 body mask")
    if cell_id == "A1":
        if len(markers) != 4 or not all(any(term in line for line in markers) for term in BODY_TERMS):
            raise ContractError("A1 log does not prove all four body-imitation masks")
    return markers


CHECKPOINT_AUDIT = r"""
import json,sys,torch
p=sys.argv[1]; o=torch.load(p,map_location='cpu',weights_only=False)
stack=[o]; seen=set(); tensors=elements=nonfinite=0
while stack:
 v=stack.pop()
 if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex()):
  tensors+=1; elements+=v.numel(); nonfinite+=int((~torch.isfinite(v)).sum().item())
 elif isinstance(v,dict) and id(v) not in seen:
  seen.add(id(v)); stack.extend(v.values())
 elif isinstance(v,(list,tuple)) and id(v) not in seen:
  seen.add(id(v)); stack.extend(v)
i=o.get('infos') if isinstance(o,dict) else {}; i=i if isinstance(i,dict) else {}
print(json.dumps({'iter':o.get('iter'),'training_contract_schema_version':i.get('training_contract_schema_version'),'training_contract_sha256':i.get('training_contract_sha256'),'training_contract_lineage_exact':i.get('training_contract_lineage_exact'),'floating_tensor_count':tensors,'floating_elements':elements,'nonfinite_floating_elements':nonfinite},sort_keys=True))
"""


def checkpoint_audit(python: Path, path: Path) -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [str(python), "-c", CHECKPOINT_AUDIT, str(path)],
            text=True, stderr=subprocess.STDOUT,
        )
        result = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot audit checkpoint {path}: {exc}") from exc
    if result.get("floating_tensor_count", 0) <= 0 or result.get("nonfinite_floating_elements") != 0:
        raise ContractError(f"checkpoint is empty/non-finite: {path}")
    return result


def launch(manifest: dict[str, Any], manifest_path: Path, launcher_path: Path, root_confirm: str | None) -> None:
    runtime = manifest["runtime"]
    if os.geteuid() != 0:
        raise ContractError("launch requires root on the simulator Pod")
    if root_confirm != runtime["root_launch_confirmation"]:
        raise ContractError("launch requires the exact root simulation-only confirmation token")
    preflight = verify_runtime(manifest, require_initial_empty=True)
    run_root = Path(runtime["run_root"])
    if run_root.exists():
        raise ContractError("run root already exists; preserve it and audit rather than retrying")
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_sha = sha256_file(manifest_path)
    launcher_sha = sha256_file(launcher_path)
    for cell_id in CELL_IDS:
        cell = cell_map(manifest)[cell_id]
        before = gpu_snapshot(runtime["gpu"])
        if len(before["trainer_pids"]) >= runtime["maximum_owned_trainers_on_gpu"]:
            raise ContractError("GPU trainer count reached the two-arm ownership limit")
        if before["free_memory_mib"] < runtime["minimum_free_gpu_memory_mib_before_each_launch"]:
            raise ContractError("GPU free memory fell below the preregistered floor")
        arm_dir = run_root / cell["run_name"]
        arm_dir.mkdir(exist_ok=False)
        log_path = arm_dir / runtime["training_log_basename"]
        state_path = arm_dir / runtime["launch_state_basename"]
        command = build_command(manifest, cell_id)
        launch_contract = {
            "artifact_kind": "phase1_non_striking_arm_a01_launch_contract",
            "schema_version": 1,
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest_sha,
            "launcher_sha256": launcher_sha,
            "training_commit": manifest["source"]["expected_training_commit"],
            "training_tree": manifest["source"]["expected_training_tree"],
            "cell_id": cell_id,
            "run_name": cell["run_name"],
            "seed": manifest["shared_training_contract"]["training_seed"],
            "fresh_initialization": True,
            "expected_body_names": cell["body_names"],
            "command": command,
            "gpu_snapshot_before": before,
            "verified_inputs": preflight["verified_inputs"],
            "ignored_asset": preflight["ignored_asset"],
            "training_environment_sha256": runtime["training_environment_sha256"],
            "training_module_path": preflight["training_module_path"],
            "automatic_judge_launch": False,
            "real_robot_commands_forbidden": True
        }
        write_json_exclusive(arm_dir / runtime["launch_contract_basename"], launch_contract)
        environment = preflight["environment"].copy()
        environment.update({
            "KIT_BOOT_MARKER": runtime["kit_boot_marker"],
            "KIT_BOOT_TIMEOUT_S": str(runtime["kit_boot_timeout_seconds"]),
            "KIT_BOOT_POLL_S": str(runtime["poll_seconds"]),
            "KIT_BOOT_STATE_FILE": str(state_path),
        })
        subprocess.run(
            [str(preflight["locked"]), str(log_path), *command],
            cwd=preflight["wbt"], env=environment, check=True,
        )
        state = parse_launch_state(state_path)
        if not state.get("pid", "").isdigit() or state.get("pid") != state.get("pgid"):
            raise ContractError(f"{cell_id} launcher did not record isolated pid==pgid")
        run_dir = locate_training_run(preflight["wbt"], cell["run_name"])
        hard_path = run_dir / "params/training_contract.json"
        hard_sha, _ = verify_hard_contract(hard_path, manifest)
        markers = verify_mask_log(log_path, cell_id)
        verified = {
            "artifact_kind": "phase1_non_striking_arm_a01_runtime_verified",
            "schema_version": 1,
            "manifest_sha256": manifest_sha,
            "launcher_sha256": launcher_sha,
            "cell_id": cell_id,
            "run_name": cell["run_name"],
            "pid": int(state["pid"]), "pgid": int(state["pgid"]),
            "training_run_dir": str(run_dir),
            "hard_contract_path": str(hard_path),
            "hard_contract_sha256": hard_sha,
            "mask_log_markers": markers,
            "boot_marker_observed": runtime["kit_boot_marker"],
            "judge_started": False,
            "real_robot_commands_executed": False,
        }
        write_json_exclusive(arm_dir / runtime["runtime_verified_basename"], verified)
        print(json.dumps({"status": "launched", "cell": cell_id, "pid": int(state["pid"]), "hard_contract_sha256": hard_sha}, sort_keys=True), flush=True)


def finalize(manifest: dict[str, Any], manifest_path: Path, launcher_path: Path) -> dict[str, Any]:
    runtime = manifest["runtime"]
    preflight = verify_runtime(manifest, require_initial_empty=False)
    manifest_sha = sha256_file(manifest_path)
    launcher_sha = sha256_file(launcher_path)
    run_root = Path(runtime["run_root"])
    results = {}
    hard_shas = set()
    for cell_id in CELL_IDS:
        cell = cell_map(manifest)[cell_id]
        arm_dir = run_root / cell["run_name"]
        launch_path = arm_dir / runtime["launch_contract_basename"]
        verified_path = arm_dir / runtime["runtime_verified_basename"]
        state_path = arm_dir / runtime["launch_state_basename"]
        log_path = arm_dir / runtime["training_log_basename"]
        if not all(path.is_file() for path in (launch_path, verified_path, state_path, log_path)):
            raise ContractError(f"{cell_id} runtime evidence is incomplete")
        launch_contract = json.loads(launch_path.read_text(encoding="utf-8"))
        verified = json.loads(verified_path.read_text(encoding="utf-8"))
        for value in (launch_contract, verified):
            if value.get("manifest_sha256") != manifest_sha or value.get("launcher_sha256") != launcher_sha:
                raise ContractError(f"{cell_id} evidence binds different control bytes")
            if value.get("cell_id") != cell_id or value.get("run_name") != cell["run_name"]:
                raise ContractError(f"{cell_id} evidence identity changed")
        if launch_contract.get("command") != build_command(manifest, cell_id):
            raise ContractError(f"{cell_id} launch command changed")
        state = parse_launch_state(state_path)
        if not state.get("pid", "").isdigit() or state.get("pid") != state.get("pgid"):
            raise ContractError(f"{cell_id} state lost pid==pgid")
        if process_alive(int(state["pid"])):
            raise ContractError(f"{cell_id} is still running; finalize is read-only")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if FAILURE_RE.search(text):
            raise ContractError(f"{cell_id} log contains a hard failure signature")
        verify_mask_log(log_path, cell_id)
        run_dir = Path(verified["training_run_dir"])
        hard_path = run_dir / "params/training_contract.json"
        hard_sha, _ = verify_hard_contract(hard_path, manifest)
        if hard_sha != verified.get("hard_contract_sha256"):
            raise ContractError(f"{cell_id} hard contract changed after launch")
        hard_shas.add(hard_sha)
        milestones = []
        for iteration in manifest["shared_training_contract"]["relative_checkpoint_milestones"]:
            checkpoint = run_dir / f"model_{iteration}.pt"
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise ContractError(f"{cell_id} checkpoint model_{iteration}.pt is missing/empty")
            before = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
            time.sleep(1)
            after = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
            if before != after:
                raise ContractError(f"{cell_id} checkpoint model_{iteration}.pt is unstable")
            audit = checkpoint_audit(preflight["python"], checkpoint)
            expected_audit = {
                "iter": iteration,
                "training_contract_schema_version": 3,
                "training_contract_sha256": hard_sha,
                "training_contract_lineage_exact": 1,
                "nonfinite_floating_elements": 0,
            }
            for key, wanted in expected_audit.items():
                if audit.get(key) != wanted:
                    raise ContractError(f"{cell_id} model_{iteration}.pt {key} changed")
            milestones.append({
                "iteration": iteration, "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint),
                "audit": audit,
            })
        result = {
            "artifact_kind": "phase1_non_striking_arm_a01_checkpoint_result",
            "schema_version": 1,
            "manifest_sha256": manifest_sha,
            "launcher_sha256": launcher_sha,
            "cell_id": cell_id,
            "run_name": cell["run_name"],
            "body_names": cell["body_names"],
            "hard_contract_sha256": hard_sha,
            "checkpoints": milestones,
            "same_immutable_signed_paper_judged": False,
            "stop_or_promote_authorized": False,
            "second_seed_authorized": False,
            "hardware_authorized": False,
        }
        result_path = arm_dir / runtime["final_result_basename"]
        write_json_exclusive(result_path, result)
        results[cell_id] = {"path": str(result_path), "sha256": sha256_file(result_path), **result}
    if len(hard_shas) != 1:
        raise ContractError("paired cells emitted different structural hard contracts")
    return {
        "status": "paired_checkpoints_finite_bound_judging_still_blocked",
        "common_hard_contract_sha256": next(iter(hard_shas)),
        "cells": results,
    }


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--manifest", type=Path,
        default=root / "configs/phase1_non_striking_arm_imitation_a01_prereg_20260714.json",
    )
    value.add_argument("--mode", choices=("plan", "validate-runtime", "launch", "finalize"), default="plan")
    value.add_argument("--plan-output", type=Path)
    value.add_argument("--root-confirm")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    launcher_path = Path(__file__).resolve()
    try:
        manifest = load_manifest(manifest_path)
        if args.mode == "plan":
            plan = build_plan(manifest, manifest_path, launcher_path)
            if args.plan_output is not None:
                write_json_exclusive(args.plan_output.resolve(), plan)
            print(json.dumps(plan, sort_keys=True, indent=2))
        elif args.mode == "validate-runtime":
            verified = verify_runtime(manifest, require_initial_empty=True)
            print(json.dumps({
                "status": "runtime_validated_no_launch",
                "gpu_snapshot": verified["gpu_snapshot"],
                "training_module_path": verified["training_module_path"],
                "verified_inputs": verified["verified_inputs"],
            }, sort_keys=True, indent=2))
        elif args.mode == "launch":
            launch(manifest, manifest_path, launcher_path, args.root_confirm)
        else:
            print(json.dumps(finalize(manifest, manifest_path, launcher_path), sort_keys=True, indent=2))
    except (ContractError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[non-striking-arm-a01] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
