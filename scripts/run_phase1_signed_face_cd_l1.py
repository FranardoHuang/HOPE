#!/usr/bin/env python3
"""Fail-closed, C2/D2-only signed-face L1 launcher and terminal finalizer.

The default is a read-only plan.  A launch creates exactly one new arm claim
on its preregistered empty GPU and returns after the reviewed, host-wide Kit
boot lock observes the hard-contract marker.  C2 then continues on GPU1 while
D2 may boot and train on GPU2.  Each natural exit and terminal ``model_24.pt``
is finalized independently before the pair receipt can be written.

This program contains no judge, activation, L2, second-seed, retry, or robot
path.  The wrapper may signal only the exact PGID that it created when Kit
fails before its marker; this launcher never discovers or signals processes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time
from typing import Any


MANIFEST_ID = "phase1-signed-face-c2-d2-l1-single-seed-20260714-v1"
CELL_IDS = ("C2", "D2")
TRAINING_COMMIT = "4467d79f1ed425a4263f0caaad2f661e1ec737ad"
TRAINING_TREE = "497db1d8f2d7fb1b554337928f098a2951d4cf0d"
SOURCE_CHECKOUT = "/workspace/codexschema/nohope_signed_face_cd_l1_4467d79"
ARTIFACT_ROOT = "/workspace/codexschema/phase1_signed_face_cd_l1_20260714"
RUN_ROOT = f"{ARTIFACT_ROOT}/runs/l1"
CONTROL_ROOT = f"{ARTIFACT_ROOT}/control/v1"
TRAINING_ENV_SHA256_BY_CELL = {
    "C2": "628c5163049a4ea4f4e14b98345bbe168191e2e90efbefbb83be858b5b12869e",
    "D2": "55d332432822e44f3308685f859f9fbdc5d946534104479182437548745920a5",
}
EXPECTED_PYTHONPATH = ":".join((
    f"{SOURCE_CHECKOUT}/hope_training/whole_body_tracking/source/whole_body_tracking",
    "/workspace/IsaacLab/source/isaaclab",
    "/workspace/IsaacLab/source/isaaclab_tasks",
    "/workspace/IsaacLab/source/isaaclab_assets",
    "/workspace/IsaacLab/source/isaaclab_rl",
))
ROOT_CONFIRMATION = "ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_C2_D2_L1_V1"
CLAIM_ARG_PREFIX = "++training_launch_claim_sha256="
CLAIM_PLACEHOLDER = "<computed-after-atomic-claim>"
CARB_THREAD_ARG = "++kit_carb_tasking_thread_count=16"
TBB_THREAD_ARG = "++kit_tbb_thread_count=16"
THREAD_MARKER = (
    "[train.py] KIT_THREAD_CAP_OK: carb.tasking=16 omni.tbb=16 useOmniJob=false"
)
HARD_CONTRACT_MARKER = "[train.py] hard training contract:"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
FAILURE_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemory|Segmentation fault|"
    r"\b(?:NaN|Inf)\b|\bKilled\b|malloc|bad_alloc",
    re.IGNORECASE,
)

# Exact static identity of the recipe list.  This avoids copying a second long
# recipe into Python while still making any manifest edit fail closed.
BASE_RECIPE_SHA256 = "5ed06ee4b0b2977c5a8b24bcaa8aa1998537e4a0d67bf79c08b08016c015c6b7"
CRITICAL_FILES_SHA256 = "f2698ec0508d95680e9f05c6dd35bb46d251d4cff35d0bbabaece0f490636568"

EXPECTED_HISTORICAL = {
    "training_source_commit": "5f691b3400fe3feda1a690675912a97f09e906bb",
    "training_source_tree": "be262d1a43a09ee7713e3a386f53fd50c8f8b076",
    "control_commit": "466f8ea935310407f73b95e812bbd5f0a18705b4",
    "control_tree": "7ea23ec7151ecfc69ad21ea9333449ab7436578e",
    "v9_manifest_sha256": "aba184f568a8598e354c353ec8e7cd4bfed1e9483b05c259e185b2c09409b580",
    "v9_launcher_sha256": "8e8f1bd0e148bf281aa345e34c86fe3d56ec2d5554f1c060e3de796dae30996a",
    "v9_plan_sha256": "afd191e7f766cdf2c01dfa785ecf75b426819f3df800e99da132e078afc66c38",
    "v9_c_failed_before_hard_contract": True,
    "v9_d_absent": True,
    "prior_claims_or_checkpoints_adopted": False,
    "fresh_one_update_diagnostics_adopted": False,
    "reason_for_new_namespace": (
        "v9 did not bind the 0.0 versus -0.4 guidance recipe in the emitted hard "
        "contract; diagnostic one-update runs lacked the formal outer claim"
    ),
}


class ContractError(RuntimeError):
    """One preregistered source, runtime, or scientific invariant changed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if type(value) is not str or not SHA_RE.fullmatch(value):
        raise ContractError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def strictly_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            strictly_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strictly_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def require_exact(actual: Any, expected: Any, label: str) -> None:
    if not strictly_equal(actual, expected):
        raise ContractError(f"{label} changed or has a bool/int type confusion")


def require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ContractError(
            f"{label} keys changed: missing={sorted(keys - actual)} "
            f"extra={sorted(actual - keys)}"
        )
    return value


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} root must be an object")
    return value


def write_json_exclusive(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
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


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def cells(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("cells")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ContractError("manifest must contain exactly two cells")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or type(item.get("cell_id")) is not str:
            raise ContractError("each cell must be an object with a string cell_id")
        if item["cell_id"] in result:
            raise ContractError("duplicate cell_id")
        result[item["cell_id"]] = item
    if tuple(result) != CELL_IDS:
        raise ContractError("cell order and identity must be exactly C2 then D2")
    return result


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("manifest_id") != MANIFEST_ID:
        raise ContractError("unexpected manifest schema or identity")
    expected_top = {
        "status": "machine_preregistered_plan_only_root_launch_switch_required",
        "human_owner": "Franco",
        "executor": "Codex",
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "automatic_retry_forbidden": True,
        "activation_authorized": False,
        "judge_authorized": False,
        "l2_authorized": False,
        "second_seed_authorized": False,
    }
    for key, expected in expected_top.items():
        require_exact(manifest.get(key), expected, f"manifest {key}")

    source = manifest.get("source")
    runtime = manifest.get("runtime")
    inputs = manifest.get("inputs")
    shared = manifest.get("shared_training_contract")
    schedule = manifest.get("execution_schedule")
    evaluation = manifest.get("evaluation")
    if not all(isinstance(v, dict) for v in (source, runtime, inputs, shared, schedule, evaluation)):
        raise ContractError("source/runtime/inputs/shared/schedule/evaluation must be objects")

    require_exact(source.get("expected_training_commit"), TRAINING_COMMIT, "training commit")
    require_exact(source.get("expected_training_tree"), TRAINING_TREE, "training tree")
    require_exact(source.get("training_checkout"), SOURCE_CHECKOUT, "training checkout")
    require_exact(source.get("wbt_relative_path"), "hope_training/whole_body_tracking", "WBT path")
    critical = source.get("critical_files")
    if not isinstance(critical, dict) or canonical_sha256(critical) != CRITICAL_FILES_SHA256:
        raise ContractError("critical source map changed")
    for relative, digest in critical.items():
        rel = Path(str(relative))
        if rel.is_absolute() or ".." in rel.parts:
            raise ContractError(f"unsafe critical source path: {relative}")
        require_sha(digest, f"critical source {relative}")
    ignored = source.get("ignored_runtime_asset")
    if not isinstance(ignored, dict):
        raise ContractError("ignored A3 runtime asset contract is missing")
    require_exact(ignored.get("tree_content_sha256"), "0137f59b1fe45e7d5f8fa731bedca905f5466bc98e8d1354081fe071d60426c6", "A3 asset tree")
    require_exact(ignored.get("file_count"), 46, "A3 asset file count")
    require_exact(ignored.get("total_file_bytes"), 15378264, "A3 asset bytes")
    require_exact(ignored.get("symlinks_forbidden"), True, "A3 symlink rule")
    require_exact(ignored.get("target_must_be_gitignored"), True, "A3 gitignore rule")

    expected_runtime = {
        "pod": "pod1",
        "artifact_root": ARTIFACT_ROOT,
        "external_control_root": CONTROL_ROOT,
        "run_root": RUN_ROOT,
        "source_asset_root": "/workspace/codexschema/phase1_fresh_20260711",
        "isaac_python": "/workspace/hope_isaac_venv/bin/python",
        "isaaclab_root": "/workspace/IsaacLab",
        "training_environment_sha256_by_cell": TRAINING_ENV_SHA256_BY_CELL,
        "locked_launcher_relative_path": "scripts/launch_kit_training_locked.sh",
        "kit_boot_marker": HARD_CONTRACT_MARKER,
        "kit_boot_timeout_seconds": 900,
        "poll_seconds": 5,
        "initial_gpu_must_have_zero_compute_processes": True,
        "maximum_live_trainers_per_assigned_gpu": 1,
        "minimum_free_gpu_memory_mib_before_launch": 4500,
        "minimum_host_available_memory_mib": 65536,
        "root_launch_confirmation": ROOT_CONFIRMATION,
        "launch_state_basename": "run.log.launch",
        "launch_contract_basename": "launch_contract.json",
        "runtime_verified_basename": "runtime_verified.json",
        "training_log_basename": "run.log",
        "failure_basename": "launch_failure.json",
        "terminal_result_basename": "terminal_result.json",
        "pair_result_basename": "paired_l1_result.json",
    }
    for key, expected in expected_runtime.items():
        require_exact(runtime.get(key), expected, f"runtime {key}")
    require_exact(runtime.get("kit_thread_cap_contract"), {
        "carb_tasking_thread_count": 16,
        "omni_tbb_thread_count": 16,
        "use_omni_job": False,
        "runtime_marker": THREAD_MARKER,
    }, "Kit 16/16 thread contract")
    require_exact(runtime.get("runtime_closure"), {
        "isaaclab": {
            "root": "/workspace/IsaacLab",
            "commit": "21f7136325136ca3f6ca4e0a8125edffe5c24f7e",
            "tree": "9ce79305538001b0451d3745f47076c57c5d6006",
            "must_be_clean": True,
        },
        "python": {
            "resolved_executable": "/usr/bin/python3.10",
            "executable_sha256": "06630724486efc9d97db03c62949511584b896c110097153ef970f9294fd3ba0",
            "implementation": "CPython",
            "version": "3.10.18 (main, Jun  4 2025, 08:56:00) [GCC 13.3.0]",
            "pip_freeze_all_sorted_line_count": 193,
            "pip_freeze_all_sorted_sha256": "1429f665d27947c412a454f68f31ecab1ceb6c21a6712931fff0a485fd9e5e0b",
        },
    }, "runtime closure")

    expected_inputs = {
        "forehand_motion": ("f2cb2d9f5d27cefbcee0b790000fcd979abaf02894d4fcad061ebca27f141687", False),
        "backhand_motion": ("1722553375cd28f9b2d567c01b1a5fc6bcd149fa12cadb20e5202a9153367534", False),
        "schema3_train_bank": ("3a9d8851c1c0b13ef82f58228ea1cf83213157c70d72daa514f1bed3a3885b71", True),
        "schema3_train_bank_rebind_report": ("9fffed0308eb0102e3575c3a255e9466c04f45e6c0c303cefb5541a19decbb37", True),
    }
    if set(inputs) != set(expected_inputs):
        raise ContractError("input set changed")
    for name, (digest, absolute) in expected_inputs.items():
        require_exact(inputs[name].get("sha256"), digest, f"{name} SHA")
        path_key = "path" if absolute else "relative_path"
        path = Path(str(inputs[name].get(path_key, "")))
        if path.is_absolute() is not absolute:
            raise ContractError(f"{name} path kind changed")
    bank = inputs["schema3_train_bank"]
    require_exact(bank.get("schema_version"), 3, "train bank schema")
    require_exact(bank.get("split"), "train", "train bank split")
    require_exact(bank.get("physics_contract_sha256"), "09dfe8999c54e36b258fe54b5ec3da5d9816ff3be3675963b919371d7f4afb95", "physics contract")
    require_exact(bank.get("source_family_sha256"), "9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db", "bank family")

    expected_shared = {
        "training_seed": 3,
        "initialization": "fresh",
        "num_envs": 512,
        "max_iterations": 25,
        "save_interval": 100,
        "expected_terminal_checkpoint_iteration": 24,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1, -1],
        "zero_joint_friction": True,
        "motion_kinematics_exact": True,
        "question_bank_schema_version": 3,
        "question_bank_split": "train",
        "actor_observation_contract": "deploy_parity_face179",
        "actor_observation_dim": 179,
        "action_dim": 31,
        "strike_phase_per_clip": [0.471, 0.338],
        "event_timing_mode": "disabled",
        "positional_guidance_weight": 0.0,
        "face_guidance_theta_max": math.pi,
    }
    for key, expected in expected_shared.items():
        require_exact(shared.get(key), expected, f"shared contract {key}")
    recipe = shared.get("base_recipe")
    if not isinstance(recipe, list) or any(type(part) is not str for part in recipe):
        raise ContractError("base recipe must be a string argv list")
    if canonical_sha256(recipe) != BASE_RECIPE_SHA256:
        raise ContractError("base recipe changed")
    forbidden_recipe = (
        "racket_guidance_weight", "racket_face_guidance_weight",
        "racket_face_guidance_theta_max", "training_launch_claim_sha256",
        "kit_carb_tasking_thread_count", "kit_tbb_thread_count",
    )
    if any(any(token in part for token in forbidden_recipe) for part in recipe):
        raise ContractError("per-cell guidance/claim/thread facts leaked into the base recipe")

    expected_cells = {
        "C2": {
            "cell_id": "C2",
            "human_name": "fresh signed-face control with provenance-complete checkpoint",
            "causal_role": "fresh_control",
            "run_name": "phase1_signed_face_l1_c2d2_v1_C2_fresh_control_seed3",
            "gpu": 1,
            "face_guidance_weight": 0.0,
            "expected_lineage_exact": True,
        },
        "D2": {
            "cell_id": "D2",
            "human_name": "fresh signed-face guidance with provenance-complete checkpoint",
            "causal_role": "fresh_guidance",
            "run_name": "phase1_signed_face_l1_c2d2_v1_D2_fresh_guidance_seed3",
            "gpu": 2,
            "face_guidance_weight": -0.4,
            "expected_lineage_exact": True,
        },
    }
    require_exact(cells(manifest), expected_cells, "C2/D2 cells")
    for item in expected_cells.values():
        if not RUN_RE.fullmatch(item["run_name"]):
            raise ContractError("unsafe run name")
    require_exact(schedule, {
        "mode": "serial_kit_boot_then_cross_gpu_concurrent_training",
        "ordered_cells": ["C2", "D2"],
        "same_host": True,
        "distinct_assigned_gpus": True,
        "kit_boot_serialized_by_shared_lock": True,
        "one_new_atomic_claim_per_invocation": True,
        "assigned_gpu_must_be_empty_before_each_claim": True,
        "next_cell_requires_previous_runtime_verified": True,
        "previous_cell_terminal_not_required_for_next_claim": True,
        "stop_on_any_failure": True,
        "automatic_retry_forbidden": True,
    }, "execution schedule")
    require_exact(manifest.get("historical_read_only_evidence"), EXPECTED_HISTORICAL, "historical read-only evidence")
    require_exact(evaluation, {
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "same_immutable_signed_paper_required_before_any_later_decision": True,
        "terminal_result_scope": "finite lineage-exact L1 checkpoint provenance only",
    }, "evaluation boundary")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json_object(path, "C2/D2 manifest")
    validate_manifest(manifest)
    return manifest


def repo_for_static_source(manifest: dict[str, Any]) -> Path:
    runtime_checkout = Path(manifest["source"]["training_checkout"])
    return runtime_checkout if runtime_checkout.is_dir() else Path(__file__).resolve().parents[1]


def verify_static_source(manifest: dict[str, Any]) -> dict[str, Any]:
    repo = repo_for_static_source(manifest)
    source = manifest["source"]
    try:
        tree = git_output(repo, "rev-parse", f"{TRAINING_COMMIT}^{{tree}}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"training source commit is unavailable in {repo}") from exc
    if tree != TRAINING_TREE:
        raise ContractError("training source tree changed")
    checked = {}
    wbt_rel = Path(source["wbt_relative_path"])
    for relative, expected in source["critical_files"].items():
        git_path = (wbt_rel / relative).as_posix()
        try:
            content = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{TRAINING_COMMIT}:{git_path}"]
            )
        except subprocess.CalledProcessError as exc:
            raise ContractError(f"critical source absent from commit: {git_path}") from exc
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ContractError(f"critical source SHA changed: {git_path}")
        checked[relative] = actual
    return {"commit": TRAINING_COMMIT, "tree": tree, "critical_files": checked}


def input_paths(manifest: dict[str, Any]) -> dict[str, tuple[Path, str]]:
    root = Path(manifest["runtime"]["source_asset_root"])
    inputs = manifest["inputs"]
    return {
        "forehand_motion": (root / inputs["forehand_motion"]["relative_path"], inputs["forehand_motion"]["sha256"]),
        "backhand_motion": (root / inputs["backhand_motion"]["relative_path"], inputs["backhand_motion"]["sha256"]),
        "schema3_train_bank": (Path(inputs["schema3_train_bank"]["path"]), inputs["schema3_train_bank"]["sha256"]),
        "schema3_train_bank_rebind_report": (Path(inputs["schema3_train_bank_rebind_report"]["path"]), inputs["schema3_train_bank_rebind_report"]["sha256"]),
    }


def optimization_recipe(manifest: dict[str, Any], cell_id: str) -> dict[str, Any]:
    shared = manifest["shared_training_contract"]
    cell = cells(manifest)[cell_id]
    return {
        "initialization": "fresh",
        "training_seed": shared["training_seed"],
        "num_envs": shared["num_envs"],
        "max_iterations": shared["max_iterations"],
        "save_interval": shared["save_interval"],
        "positional_guidance_weight": shared["positional_guidance_weight"],
        "signed_face_guidance_weight": cell["face_guidance_weight"],
        "face_guidance_theta_max": shared["face_guidance_theta_max"],
    }


def build_command(manifest: dict[str, Any], cell_id: str, claim_sha: str) -> list[str]:
    if claim_sha != CLAIM_PLACEHOLDER:
        require_sha(claim_sha, "training launch claim")
    shared = manifest["shared_training_contract"]
    runtime = manifest["runtime"]
    cell = cells(manifest)[cell_id]
    paths = input_paths(manifest)
    command = [
        runtime["isaac_python"], "scripts/train.py", *shared["base_recipe"],
        f"seed={shared['training_seed']}",
        f"num_envs={shared['num_envs']}",
        f"max_iterations={shared['max_iterations']}",
        f"algo.runner.save_interval={shared['save_interval']}",
        f"run_name={cell['run_name']}",
        "checkpoint_path=null", "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=false",
        f"motion_file={paths['forehand_motion'][0]}",
        f"motion_file_2={paths['backhand_motion'][0]}",
        "task.racket.strike_phase_per_clip=[0.471,0.338]",
        f"++task.racket.question_bank={paths['schema3_train_bank'][0]}",
        "++task.racket.face_command_pairing=shared_plus_y",
        f"++task.rewards.racket_guidance_weight={shared['positional_guidance_weight']}",
        f"++task.rewards.racket_face_guidance_weight={cell['face_guidance_weight']}",
        f"++task.rewards.racket_face_guidance_theta_max={shared['face_guidance_theta_max']}",
        CARB_THREAD_ARG, TBB_THREAD_ARG,
        f"{CLAIM_ARG_PREFIX}{claim_sha}",
    ]
    prefixes = (
        "++task.rewards.racket_guidance_weight=",
        "++task.rewards.racket_face_guidance_weight=",
        "++task.rewards.racket_face_guidance_theta_max=",
        "++kit_carb_tasking_thread_count=", "++kit_tbb_thread_count=",
        CLAIM_ARG_PREFIX,
    )
    for prefix in prefixes:
        if sum(part.startswith(prefix) for part in command) != 1:
            raise ContractError(f"constructed command must contain exactly one {prefix}")
    forbidden = ("ros2", "run_deploy", "joint_command", "real_robot", "/dev/")
    if any(any(token in part.lower() for token in forbidden) for part in command):
        raise ContractError("constructed command contains a robot/runtime token")
    return command


def normalized_command(manifest: dict[str, Any], cell_id: str) -> list[str]:
    result = []
    for part in build_command(manifest, cell_id, CLAIM_PLACEHOLDER):
        if part.startswith("run_name="):
            part = "run_name=<paired>"
        elif part.startswith("++task.rewards.racket_face_guidance_weight="):
            part = "++task.rewards.racket_face_guidance_weight=<causal-axis>"
        elif part.startswith(CLAIM_ARG_PREFIX):
            part = f"{CLAIM_ARG_PREFIX}<paired>"
        result.append(part)
    return result


def expected_source_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest["source"]
    return {
        "commit": source["expected_training_commit"],
        "tree": source["expected_training_tree"],
        "checkout": source["training_checkout"],
        "critical_files": source["critical_files"],
    }


def build_claim(
    manifest: dict[str, Any], *, manifest_sha: str, launcher_sha: str,
    cell_id: str, arm_dir: Path, arm_identity: dict[str, int],
) -> dict[str, Any]:
    require_sha(manifest_sha, "manifest SHA")
    require_sha(launcher_sha, "launcher SHA")
    if set(arm_identity) != {"device", "inode"} or any(
        type(value) is not int or value <= 0 for value in arm_identity.values()
    ):
        raise ContractError("atomic claim directory identity is invalid")
    cell = cells(manifest)[cell_id]
    expected_dir = Path(manifest["runtime"]["run_root"]) / cell["run_name"]
    if arm_dir != expected_dir:
        raise ContractError("atomic claim path changed")
    return {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "training_source": expected_source_identity(manifest),
        "stage": "l1",
        "cell_id": cell_id,
        "run_name": cell["run_name"],
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "execution_lane": {
            "host": manifest["runtime"]["pod"],
            "physical_gpu": cell["gpu"],
            "cuda_visible_devices": str(cell["gpu"]),
            "local_training_device": "cuda:0",
            "training_environment_sha256": TRAINING_ENV_SHA256_BY_CELL[cell_id],
        },
        "expected_terminal_checkpoint_iteration": 24,
        "claim_directory": {
            "path": str(arm_dir),
            "st_dev": arm_identity["device"],
            "st_ino": arm_identity["inode"],
        },
    }


def build_plan(manifest: dict[str, Any], manifest_path: Path, launcher_path: Path) -> dict[str, Any]:
    if normalized_command(manifest, "C2") != normalized_command(manifest, "D2"):
        raise ContractError("C2/D2 commands differ outside run name, claim, and face weight")
    return {
        "artifact_kind": "phase1_signed_face_c2_d2_l1_plan_only",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "launcher_sha256": sha256_file(launcher_path),
        "training_source": verify_static_source(manifest),
        "simulation_only": True,
        "writes_or_launches_performed": False,
        "ordered_cells": list(CELL_IDS),
        "one_new_atomic_claim_per_invocation": True,
        "execution_lanes": {
            cell_id: {
                "host": manifest["runtime"]["pod"],
                "physical_gpu": cells(manifest)[cell_id]["gpu"],
                "local_training_device": "cuda:0",
            }
            for cell_id in CELL_IDS
        },
        "kit_boot_serialized_training_may_overlap": True,
        "commands": {
            cell_id: shlex.join(build_command(manifest, cell_id, CLAIM_PLACEHOLDER))
            for cell_id in CELL_IDS
        },
        "decision_boundary": {
            "activation": False, "judge": False, "l2": False,
            "second_seed": False, "stop_or_promote": False,
        },
    }


def no_symlink_existing_components(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ContractError(f"{label} contains a symlink component: {current}")


def require_regular(path: Path, label: str) -> os.stat_result:
    no_symlink_existing_components(path, label)
    try:
        result = path.stat()
    except OSError as exc:
        raise ContractError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
        raise ContractError(f"{label} must be a single-link regular file: {path}")
    return result


def require_directory(path: Path, label: str) -> os.stat_result:
    no_symlink_existing_components(path, label)
    try:
        result = path.stat()
    except OSError as exc:
        raise ContractError(f"{label} is missing: {path}") from exc
    if not stat.S_ISDIR(result.st_mode):
        raise ContractError(f"{label} must be a directory: {path}")
    return result


def verify_external_control_location(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path
) -> dict[str, Any]:
    """Bind runtime/finalizer invocations to one immutable external control snapshot."""

    root = Path(manifest["runtime"]["external_control_root"])
    require_directory(root, "external control root")
    expected_manifest = root / "phase1_signed_face_cd_l1_prereg_20260714.json"
    expected_launcher = root / "run_phase1_signed_face_cd_l1.py"
    if manifest_path != expected_manifest or launcher_path != expected_launcher:
        raise ContractError("runtime modes require the exact external control paths")
    receipts = {}
    for label, path in (("manifest", manifest_path), ("launcher", launcher_path)):
        result = require_regular(path, f"external control {label}")
        if result.st_mode & 0o222:
            raise ContractError(f"external control {label} must be read-only")
        receipts[label] = {
            "path": str(path), "sha256": sha256_file(path),
            "device": result.st_dev, "inode": result.st_ino,
        }
    return receipts


def identity(result: os.stat_result) -> dict[str, int]:
    return {"device": result.st_dev, "inode": result.st_ino}


def asset_tree(root: Path) -> dict[str, Any]:
    require_directory(root, "A3 runtime asset root")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"A3 runtime asset contains symlink: {path}")
        if path.is_dir():
            continue
        result = require_regular(path, "A3 runtime asset file")
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": result.st_size,
            "sha256": sha256_file(path),
        })
    return {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "tree_content_sha256": canonical_sha256({"files": rows}),
    }


def build_environment(
    manifest: dict[str, Any], wbt: Path, cell_id: str
) -> dict[str, str]:
    runtime = manifest["runtime"]
    gpu = cells(manifest)[cell_id]["gpu"]
    if (wbt / "setup_train_env.local.sh").exists():
        raise ContractError("untracked setup_train_env.local.sh is forbidden")
    isaac = Path(runtime["isaaclab_root"])
    entries = [
        wbt / "source/whole_body_tracking",
        isaac / "source/isaaclab", isaac / "source/isaaclab_tasks",
        isaac / "source/isaaclab_assets", isaac / "source/isaaclab_rl",
    ]
    for entry in entries:
        require_directory(entry, "PYTHONPATH entry")
    pythonpath = ":".join(str(path) for path in entries)
    if pythonpath != EXPECTED_PYTHONPATH:
        raise ContractError("source-first PYTHONPATH entries or order changed")
    environment = {
        "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "LOGNAME": "root", "USER": "root", "SHELL": "/bin/bash",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
        "HOPE_ISAAC_PYTHON": runtime["isaac_python"],
        "HOPE_ISAACLAB_ROOT": str(isaac),
        "HOPE_WBT_PYTHONPATH": pythonpath, "PYTHONPATH": pythonpath,
        "OMNI_KIT_ACCEPT_EULA": "YES", "TMPDIR": "/workspace/tmp",
        "PIP_CACHE_DIR": "/workspace/.cache/pip", "XDG_CACHE_HOME": "/workspace/.cache",
        "WANDB_DIR": "/workspace/codexschema/.wandb",
        "WANDB_ENTITY": "BerkeleyPingPong",
        "WANDB_REGISTRY_ORG": "dongc_1-university-of-california-berkeley-org",
        "WANDB_PROJECT": "hope_wbc", "WANDB_MOTION_PROJECT": "csv_to_npz",
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
    }
    for key in ("TMPDIR", "XDG_CACHE_HOME", "WANDB_DIR"):
        require_directory(Path(environment[key]), f"environment {key}")
    if canonical_sha256(environment) != TRAINING_ENV_SHA256_BY_CELL[cell_id]:
        raise ContractError("explicit source-first training environment changed")
    return environment


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    free = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(free) != 1 or not free[0].strip().isdigit():
        raise ContractError("cannot read exact GPU free memory")
    rows = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    pids = sorted({int(row.strip()) for row in rows if row.strip().isdigit()})
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


def verify_runtime_closure(manifest: dict[str, Any], python: Path) -> dict[str, Any]:
    expected = manifest["runtime"]["runtime_closure"]
    isaac = Path(expected["isaaclab"]["root"])
    if git_output(isaac, "rev-parse", "HEAD") != expected["isaaclab"]["commit"]:
        raise ContractError("IsaacLab commit changed")
    if git_output(isaac, "rev-parse", "HEAD^{tree}") != expected["isaaclab"]["tree"]:
        raise ContractError("IsaacLab tree changed")
    if git_output(isaac, "status", "--porcelain", "--untracked-files=all"):
        raise ContractError("IsaacLab checkout is dirty")
    resolved = Path(os.path.realpath(python))
    py = expected["python"]
    if str(resolved) != py["resolved_executable"] or sha256_file(resolved) != py["executable_sha256"]:
        raise ContractError("Python executable identity changed")
    code = "import json,platform,sys;print(json.dumps({'implementation':platform.python_implementation(),'version':sys.version},sort_keys=True))"
    observed = json.loads(subprocess.check_output([str(python), "-c", code], text=True))
    if observed != {"implementation": py["implementation"], "version": py["version"]}:
        raise ContractError("Python implementation/version changed")
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], text=True)
    lines = sorted(line.strip() for line in freeze.splitlines() if line.strip())
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    if len(lines) != py["pip_freeze_all_sorted_line_count"] or digest != py["pip_freeze_all_sorted_sha256"]:
        raise ContractError("Python package closure changed")
    return expected


def verify_runtime(
    manifest: dict[str, Any], cell_id: str, *, require_empty_gpu: bool
) -> dict[str, Any]:
    source = manifest["source"]
    runtime = manifest["runtime"]
    checkout = Path(source["training_checkout"])
    require_directory(checkout, "training checkout")
    if git_output(checkout, "rev-parse", "HEAD") != TRAINING_COMMIT:
        raise ContractError("training checkout commit changed")
    if git_output(checkout, "rev-parse", "HEAD^{tree}") != TRAINING_TREE:
        raise ContractError("training checkout tree changed")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ContractError("training checkout must be exact and clean")
    wbt = checkout / source["wbt_relative_path"]
    for relative, expected in source["critical_files"].items():
        path = wbt / relative
        require_regular(path, "critical training source")
        if sha256_file(path) != expected:
            raise ContractError(f"critical training source changed: {relative}")

    ignored = source["ignored_runtime_asset"]
    target = wbt / ignored["relative_path"]
    restore_checkout = Path(ignored["restore_source_checkout"])
    if git_output(restore_checkout, "rev-parse", "HEAD") != ignored["restore_source_commit"]:
        raise ContractError("ignored-asset restore checkout changed")
    if git_output(restore_checkout, "status", "--porcelain", "--untracked-files=no"):
        raise ContractError("ignored-asset restore checkout tracked files are dirty")
    restore = restore_checkout / ignored["restore_source_relative_path"]
    expected_asset = {
        "file_count": ignored["file_count"], "total_file_bytes": ignored["total_file_bytes"],
        "tree_content_sha256": ignored["tree_content_sha256"],
    }
    if asset_tree(target) != expected_asset or asset_tree(restore) != expected_asset:
        raise ContractError("ignored A3 asset target/restore tree changed")
    if subprocess.run(
        ["git", "-C", str(checkout), "check-ignore", "-q", str(target.relative_to(checkout))],
        check=False,
    ).returncode != 0:
        raise ContractError("restored A3 asset is not gitignored")

    verified_inputs = {}
    for name, (path, expected) in input_paths(manifest).items():
        require_regular(path, name)
        if sha256_file(path) != expected:
            raise ContractError(f"{name} SHA changed")
        verified_inputs[name] = {"path": str(path), "sha256": expected}
    python = Path(runtime["isaac_python"])
    locked = wbt / runtime["locked_launcher_relative_path"]
    # The frozen venv entrypoint is intentionally a symlink; runtime closure below binds its
    # resolved /usr/bin/python3.10 bytes, version, and package set.
    if not python.is_file():
        raise ContractError("Isaac Python entrypoint is missing")
    require_regular(locked, "locked Kit launcher")
    if not os.access(python, os.X_OK) or not os.access(locked, os.X_OK):
        raise ContractError("Python or locked launcher is not executable")
    environment = build_environment(manifest, wbt, cell_id)
    module_code = "import importlib.util,pathlib;s=importlib.util.find_spec('whole_body_tracking');print(pathlib.Path(s.origin).resolve())"
    module = subprocess.check_output(
        [str(python), "-c", module_code], cwd=wbt, env=environment,
        text=True, stderr=subprocess.STDOUT,
    ).strip()
    expected_module = wbt / "source/whole_body_tracking/whole_body_tracking"
    try:
        Path(module).relative_to(expected_module)
    except ValueError as exc:
        raise ContractError("whole_body_tracking resolves outside exact source") from exc
    closure = verify_runtime_closure(manifest, python)
    if available_memory_mib() < runtime["minimum_host_available_memory_mib"]:
        raise ContractError("host available memory is below the frozen floor")
    assigned_gpu = cells(manifest)[cell_id]["gpu"]
    gpu = gpu_snapshot(assigned_gpu)
    if require_empty_gpu and gpu["compute_pids"]:
        raise ContractError(
            f"{cell_id} assigned GPU{assigned_gpu} must be empty, found {gpu['compute_pids']}"
        )
    if gpu["free_memory_mib"] < runtime["minimum_free_gpu_memory_mib_before_launch"]:
        raise ContractError("GPU free memory is below the frozen floor")
    return {
        "checkout": checkout, "wbt": wbt, "python": python, "locked": locked,
        "environment": environment, "training_module_path": module,
        "runtime_closure": closure, "verified_inputs": verified_inputs,
        "gpu_snapshot": gpu, "assigned_gpu": assigned_gpu,
        "training_environment_sha256": TRAINING_ENV_SHA256_BY_CELL[cell_id],
        "ignored_asset": {"target": str(target), **expected_asset},
    }


def parse_state(path: Path) -> dict[str, str]:
    require_regular(path, "launch state")
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def process_starttime(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return -1
    close = raw.rfind(")")
    if close < 0:
        raise ContractError("cannot parse process stat")
    fields = raw[close + 2:].split()
    if len(fields) <= 19 or not fields[19].isdigit():
        raise ContractError("process stat lacks starttime")
    return int(fields[19])


def locate_training_run(wbt: Path, run_name: str, timeout: int = 90) -> Path:
    root = wbt / "logs/rsl_rl/agibot_a3_hope_virtualball"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hits = sorted(
            path for path in root.glob(f"*_{run_name}")
            if path.is_dir() and (path / "params/training_contract.json").is_file()
        ) if root.is_dir() else []
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ContractError(f"run name maps to multiple training directories: {run_name}")
        time.sleep(2)
    raise ContractError(f"training run/contract did not materialize: {run_name}")


def verify_hard_contract(path: Path, manifest: dict[str, Any], cell_id: str) -> tuple[str, dict[str, Any]]:
    require_regular(path, "adjacent hard contract")
    contract = read_json_object(path, "adjacent hard contract")
    shared = manifest["shared_training_contract"]
    expected = {
        "schema_version": 3,
        "actor_obs_contract": "deploy_parity_face179",
        "actor_obs_total_dim": 179,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1, -1],
        "strike_phase_per_clip": [0.471, 0.338],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
        "motion_event_timing": {"mode": "disabled"},
        "racket_guidance_reward": {
            "position": {"weight": 0.0, "command_name": "racket_target", "d_max": 0.5},
            "signed_face": {
                "weight": cells(manifest)[cell_id]["face_guidance_weight"],
                "command_name": "racket_target",
                "theta_max": math.pi,
            },
        },
    }
    for key, wanted in expected.items():
        require_exact(contract.get(key), wanted, f"hard contract {key}")
    if len(contract.get("joint_names", [])) != 31 or len(contract.get("action_joint_ids", [])) != 31:
        raise ContractError("hard contract does not bind 31 joints/actions")
    friction = contract.get("joint_friction_coefficients")
    if not isinstance(friction, list) or len(friction) != 31 or any(type(v) not in (int, float) or float(v) != 0.0 for v in friction):
        raise ContractError("hard contract is not 31/31 zero-friction")
    clips = contract.get("motion_clips")
    if not isinstance(clips, list) or [item.get("sha256") for item in clips if isinstance(item, dict)] != [
        manifest["inputs"]["forehand_motion"]["sha256"],
        manifest["inputs"]["backhand_motion"]["sha256"],
    ]:
        raise ContractError("hard contract motion order/SHA changed")
    bank = contract.get("question_bank")
    expected_bank = manifest["inputs"]["schema3_train_bank"]
    if not isinstance(bank, dict):
        raise ContractError("hard contract lacks train bank")
    for key in ("sha256", "physics_contract_sha256", "source_family_sha256", "schema_version", "split"):
        require_exact(bank.get(key), expected_bank[key], f"hard contract train bank {key}")
    require_exact(bank.get("exact"), True, "hard contract train bank exactness")
    return sha256_file(path), contract


def verify_pair_contracts(contracts: dict[str, dict[str, Any]]) -> None:
    if set(contracts) != set(CELL_IDS):
        raise ContractError("paired hard-contract audit requires C2 and D2")
    normalized = {}
    weights = {}
    for cell_id in CELL_IDS:
        value = copy.deepcopy(contracts[cell_id])
        try:
            signed = value["racket_guidance_reward"]["signed_face"]
            weights[cell_id] = signed["weight"]
            signed["weight"] = "<causal-axis>"
        except (KeyError, TypeError) as exc:
            raise ContractError("hard contract lacks signed-face causal fact") from exc
        normalized[cell_id] = value
    require_exact(weights, {"C2": 0.0, "D2": -0.4}, "paired hard-contract weights")
    if normalized["C2"] != normalized["D2"]:
        raise ContractError("C2/D2 hard contracts differ outside signed-face guidance weight")


CHECKPOINT_CODE = r"""
import json,sys,torch
obj=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
stack=[obj]; seen=set(); tensors=elements=nonfinite=0
while stack:
 value=stack.pop()
 if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
  tensors+=1; elements+=value.numel(); nonfinite+=int((~torch.isfinite(value)).sum().item())
 elif isinstance(value,dict) and id(value) not in seen:
  seen.add(id(value)); stack.extend(value.values())
 elif isinstance(value,(list,tuple)) and id(value) not in seen:
  seen.add(id(value)); stack.extend(value)
infos=obj.get('infos') if isinstance(obj,dict) else {}; infos=infos if isinstance(infos,dict) else {}
print(json.dumps({'iter':obj.get('iter'),'training_contract_schema_version':infos.get('training_contract_schema_version'),'training_contract_sha256':infos.get('training_contract_sha256'),'training_contract_lineage_exact':infos.get('training_contract_lineage_exact'),'training_launch_claim_sha256':infos.get('training_launch_claim_sha256'),'floating_tensor_count':tensors,'floating_elements':elements,'nonfinite_floating_elements':nonfinite},sort_keys=True))
"""


def checkpoint_audit(python: Path, checkpoint: Path) -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [str(python), "-c", CHECKPOINT_CODE, str(checkpoint)],
            text=True, stderr=subprocess.STDOUT,
        )
        result = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ContractError(f"checkpoint audit failed: {exc}") from exc
    expected_keys = {
        "iter", "training_contract_schema_version", "training_contract_sha256",
        "training_contract_lineage_exact", "training_launch_claim_sha256",
        "floating_tensor_count", "floating_elements", "nonfinite_floating_elements",
    }
    require_exact_keys(result, expected_keys, "checkpoint audit")
    for key in ("iter", "training_contract_schema_version", "training_contract_lineage_exact", "floating_tensor_count", "floating_elements", "nonfinite_floating_elements"):
        if type(result[key]) is not int:
            raise ContractError(f"checkpoint audit {key} has wrong exact type")
    require_sha(result["training_contract_sha256"], "checkpoint hard contract SHA")
    require_sha(result["training_launch_claim_sha256"], "checkpoint launch claim SHA")
    if result["floating_tensor_count"] <= 0 or result["floating_elements"] <= 0:
        raise ContractError("checkpoint contains no floating tensors")
    if result["nonfinite_floating_elements"] != 0:
        raise ContractError("checkpoint contains NaN/Inf")
    return result


def expected_arm_paths(manifest: dict[str, Any], cell_id: str) -> dict[str, Path]:
    runtime = manifest["runtime"]
    arm = Path(runtime["run_root"]) / cells(manifest)[cell_id]["run_name"]
    return {
        "arm": arm,
        "launch": arm / runtime["launch_contract_basename"],
        "runtime": arm / runtime["runtime_verified_basename"],
        "state": arm / runtime["launch_state_basename"],
        "log": arm / runtime["training_log_basename"],
        "failure": arm / runtime["failure_basename"],
        "result": arm / runtime["terminal_result_basename"],
    }


def reconstruct_claim(
    manifest: dict[str, Any], manifest_sha: str, launcher_sha: str,
    cell_id: str, launch: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    claim = launch.get("training_launch_claim")
    digest = require_sha(launch.get("training_launch_claim_sha256"), "recorded launch claim")
    if not isinstance(claim, dict) or canonical_sha256(claim) != digest:
        raise ContractError("recorded launch claim bytes/digest changed")
    directory = claim.get("claim_directory")
    if not isinstance(directory, dict):
        raise ContractError("recorded launch claim lacks directory identity")
    expected = build_claim(
        manifest, manifest_sha=manifest_sha, launcher_sha=launcher_sha,
        cell_id=cell_id, arm_dir=expected_arm_paths(manifest, cell_id)["arm"],
        arm_identity={"device": directory.get("st_dev"), "inode": directory.get("st_ino")},
    )
    require_exact(claim, expected, "reconstructed atomic launch claim")
    if canonical_sha256(expected) != digest:
        raise ContractError("launch claim does not bind reconstructed source/recipe/directory")
    return expected, digest


def verify_launch_and_runtime(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    cell_id: str,
) -> dict[str, Any]:
    paths = expected_arm_paths(manifest, cell_id)
    if paths["failure"].exists():
        raise ContractError(f"{cell_id} has a preserved failure marker; automatic retry is forbidden")
    launch = read_json_object(paths["launch"], f"{cell_id} launch contract")
    runtime = read_json_object(paths["runtime"], f"{cell_id} runtime verification")
    manifest_sha = sha256_file(manifest_path)
    launcher_sha = sha256_file(launcher_path)
    control = verify_external_control_location(manifest, manifest_path, launcher_path)
    claim, claim_sha = reconstruct_claim(manifest, manifest_sha, launcher_sha, cell_id, launch)
    expected_command = build_command(manifest, cell_id, claim_sha)
    common = {
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "training_source": expected_source_identity(manifest),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "command": expected_command,
        "external_control": control,
        "execution_lane": claim["execution_lane"],
        "training_environment_sha256": TRAINING_ENV_SHA256_BY_CELL[cell_id],
    }
    for key, expected in common.items():
        require_exact(launch.get(key), expected, f"{cell_id} launch {key}")
        require_exact(runtime.get(key), expected, f"{cell_id} runtime {key}")
    require_exact(launch.get("explicit_pythonpath"), EXPECTED_PYTHONPATH, f"{cell_id} launch PYTHONPATH")
    require_exact(runtime.get("explicit_pythonpath"), EXPECTED_PYTHONPATH, f"{cell_id} runtime PYTHONPATH")
    snapshot = launch.get("gpu_snapshot_before")
    if not isinstance(snapshot, dict):
        raise ContractError(f"{cell_id} launch lacks assigned-GPU snapshot")
    require_exact(snapshot.get("gpu"), cells(manifest)[cell_id]["gpu"], f"{cell_id} assigned GPU")
    require_exact(snapshot.get("compute_pids"), [], f"{cell_id} pre-claim GPU compute PIDs")
    arm_stat = require_directory(paths["arm"], f"{cell_id} atomic claim directory")
    require_exact(identity(arm_stat), {
        "device": claim["claim_directory"]["st_dev"],
        "inode": claim["claim_directory"]["st_ino"],
    }, f"{cell_id} atomic claim directory identity")
    state = parse_state(paths["state"])
    if not state.get("pid", "").isdigit() or state.get("pgid") != state.get("pid"):
        raise ContractError(f"{cell_id} launch state lost pid==pgid")
    require_exact(runtime.get("pid"), int(state["pid"]), f"{cell_id} runtime pid")
    require_exact(runtime.get("pgid"), int(state["pgid"]), f"{cell_id} runtime pgid")
    if type(runtime.get("process_starttime_ticks")) is not int or runtime["process_starttime_ticks"] <= 0:
        raise ContractError(f"{cell_id} runtime starttime is invalid")
    run_dir = Path(str(runtime.get("training_run_dir", "")))
    expected_logs = Path(manifest["source"]["training_checkout"]) / manifest["source"]["wbt_relative_path"] / "logs/rsl_rl/agibot_a3_hope_virtualball"
    require_directory(run_dir, f"{cell_id} training run directory")
    if run_dir.parent != expected_logs or not run_dir.name.endswith(f"_{cells(manifest)[cell_id]['run_name']}"):
        raise ContractError(f"{cell_id} runtime evidence binds a foreign training run")
    require_exact(identity(run_dir.stat()), runtime.get("training_run_directory_identity"), f"{cell_id} training run identity")
    contract_path = Path(str(runtime.get("hard_contract_path", "")))
    if contract_path != run_dir / "params/training_contract.json":
        raise ContractError(f"{cell_id} adjacent contract path changed")
    contract_stat = require_regular(contract_path, f"{cell_id} adjacent hard contract")
    require_exact(identity(contract_stat), runtime.get("hard_contract_file_identity"), f"{cell_id} hard contract identity")
    hard_sha, hard_contract = verify_hard_contract(contract_path, manifest, cell_id)
    require_exact(runtime.get("hard_contract_sha256"), hard_sha, f"{cell_id} hard contract SHA")
    require_exact(runtime.get("checkpoint_absent_at_runtime_verification"), True, f"{cell_id} runtime checkpoint boundary")
    require_exact(runtime.get("activation_started"), False, f"{cell_id} activation boundary")
    require_exact(runtime.get("judge_started"), False, f"{cell_id} judge boundary")
    return {
        "paths": paths, "launch": launch, "runtime": runtime,
        "claim": claim, "claim_sha": claim_sha, "command": expected_command,
        "run_dir": run_dir, "contract_path": contract_path,
        "hard_sha": hard_sha, "hard_contract": hard_contract,
    }


def audit_terminal(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    preflight: dict[str, Any], cell_id: str, *, stable_delay: float = 1.0,
) -> dict[str, Any]:
    evidence = verify_launch_and_runtime(manifest, manifest_path, launcher_path, cell_id)
    runtime = evidence["runtime"]
    pid = runtime["pid"]
    if process_starttime(pid) == runtime["process_starttime_ticks"]:
        raise ContractError(f"{cell_id} exact trainer is still alive; finalize is read-only")
    assigned_gpu = cells(manifest)[cell_id]["gpu"]
    gpu = gpu_snapshot(assigned_gpu)
    if gpu["compute_pids"]:
        raise ContractError(
            f"{cell_id} terminal barrier requires assigned GPU{assigned_gpu} empty, "
            f"found {gpu['compute_pids']}"
        )
    log_path = evidence["paths"]["log"]
    log_stat = require_regular(log_path, f"{cell_id} training log")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if FAILURE_RE.search(text):
        raise ContractError(f"{cell_id} log contains a hard failure signature")
    if text.count(THREAD_MARKER) != 1:
        raise ContractError(f"{cell_id} log lacks exactly one 16/16 thread marker")
    if HARD_CONTRACT_MARKER not in text:
        raise ContractError(f"{cell_id} log lacks hard-contract marker")
    checkpoint = evidence["run_dir"] / "model_24.pt"
    before = require_regular(checkpoint, f"{cell_id} terminal model_24.pt")
    if before.st_size <= 0:
        raise ContractError(f"{cell_id} terminal checkpoint is empty")
    runtime_stat = require_regular(evidence["paths"]["runtime"], f"{cell_id} runtime verification")
    if before.st_ctime_ns <= runtime_stat.st_ctime_ns:
        raise ContractError(f"{cell_id} model_24.pt does not postdate runtime verification")
    if stable_delay:
        time.sleep(stable_delay)
    after = require_regular(checkpoint, f"{cell_id} stable terminal model_24.pt")
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ContractError(f"{cell_id} model_24.pt is still changing")
    checkpoint_sha = sha256_file(checkpoint)
    audit = checkpoint_audit(preflight["python"], checkpoint)
    expected_audit = {
        "iter": 24,
        "training_contract_schema_version": 3,
        "training_contract_sha256": evidence["hard_sha"],
        "training_contract_lineage_exact": 1,
        "training_launch_claim_sha256": evidence["claim_sha"],
        "nonfinite_floating_elements": 0,
    }
    for key, expected in expected_audit.items():
        require_exact(audit.get(key), expected, f"{cell_id} checkpoint {key}")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise ContractError(f"{cell_id} model_24.pt changed during audit")
    return {
        "artifact_kind": "phase1_signed_face_c2_d2_l1_terminal_result",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "launcher_sha256": sha256_file(launcher_path),
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "execution_lane": evidence["claim"]["execution_lane"],
        "training_source": expected_source_identity(manifest),
        "training_launch_claim": evidence["claim"],
        "training_launch_claim_sha256": evidence["claim_sha"],
        "training_run_dir": str(evidence["run_dir"]),
        "hard_contract_path": str(evidence["contract_path"]),
        "hard_contract_sha256": evidence["hard_sha"],
        "terminal_checkpoint_path": str(checkpoint),
        "terminal_checkpoint_sha256": checkpoint_sha,
        "terminal_checkpoint_file_identity": identity(after),
        "terminal_checkpoint_size_bytes": after.st_size,
        "checkpoint_audit": audit,
        "training_log_sha256": sha256_file(log_path),
        "training_log_file_identity": identity(log_stat),
        "exact_trainer_natural_exit_observed": True,
        "gpu_empty_terminal_barrier_observed": True,
        "kit_thread_cap_marker_occurrences": 1,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "real_robot_commands_executed": False,
    }


def load_and_verify_terminal_result(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    preflight: dict[str, Any], cell_id: str,
) -> dict[str, Any]:
    path = expected_arm_paths(manifest, cell_id)["result"]
    recorded = read_json_object(path, f"{cell_id} terminal result")
    expected = audit_terminal(
        manifest, manifest_path, launcher_path, preflight, cell_id, stable_delay=0.0
    )
    require_exact(recorded, expected, f"{cell_id} terminal result replay")
    return recorded


def choose_next_cell(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
) -> str:
    c2 = expected_arm_paths(manifest, "C2")
    d2 = expected_arm_paths(manifest, "D2")
    for cell_id, paths in (("C2", c2), ("D2", d2)):
        if paths["failure"].exists():
            raise ContractError(f"{cell_id} failure is preserved; no automatic retry is allowed")
    if not c2["arm"].exists():
        if d2["arm"].exists():
            raise ContractError("D2 exists without the required C2 predecessor")
        return "C2"
    if not c2["runtime"].is_file():
        raise ContractError("C2 is already claimed but lacks runtime_verified; D2 boot is blocked")
    # D2 may begin as soon as the shared Kit lock has released after C2's contract marker.  Rebuild
    # the exact C2 claim/source/contract first, but do not require C2 to terminate.
    predecessor = verify_launch_and_runtime(manifest, manifest_path, launcher_path, "C2")
    c2_runtime = predecessor["runtime"]
    if (
        process_starttime(c2_runtime["pid"])
        != c2_runtime["process_starttime_ticks"]
        and not c2["result"].is_file()
    ):
        raise ContractError(
            "C2 exited after runtime verification without an audited terminal result; "
            "preserve evidence before any D2 claim"
        )
    if not d2["arm"].exists():
        return "D2"
    if d2["result"].is_file():
        raise ContractError("C2 and D2 are already complete; use finalize-pair")
    raise ContractError("D2 is already claimed but lacks an audited terminal result")


def launch_next(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    root_confirm: str | None,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise ContractError("launch-next requires root on the simulator Pod")
    if root_confirm != ROOT_CONFIRMATION:
        raise ContractError("launch-next requires the exact simulation-only confirmation token")
    control = verify_external_control_location(manifest, manifest_path, launcher_path)
    cell_id = choose_next_cell(manifest, manifest_path, launcher_path)
    preflight = verify_runtime(manifest, cell_id, require_empty_gpu=True)
    paths = expected_arm_paths(manifest, cell_id)
    run_root = Path(manifest["runtime"]["run_root"])
    no_symlink_existing_components(run_root, "run root")
    run_root.mkdir(parents=True, exist_ok=True)
    require_directory(run_root, "run root")
    # No glob-as-latest: the exact run name must be absent before its one claim.
    logs_root = preflight["wbt"] / "logs/rsl_rl/agibot_a3_hope_virtualball"
    if logs_root.is_dir() and list(logs_root.glob(f"*_{cells(manifest)[cell_id]['run_name']}")):
        raise ContractError(f"{cell_id} exact training run name already exists")
    paths["arm"].mkdir(exist_ok=False)
    arm_identity = identity(paths["arm"].stat())
    manifest_sha = sha256_file(manifest_path)
    launcher_sha = sha256_file(launcher_path)
    claim = build_claim(
        manifest, manifest_sha=manifest_sha, launcher_sha=launcher_sha,
        cell_id=cell_id, arm_dir=paths["arm"], arm_identity=arm_identity,
    )
    claim_sha = canonical_sha256(claim)
    command = build_command(manifest, cell_id, claim_sha)
    launch_contract = {
        "artifact_kind": "phase1_signed_face_c2_d2_l1_launch_contract",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "training_source": expected_source_identity(manifest),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "command": command,
        "external_control": control,
        "atomic_claim_directory_identity": arm_identity,
        "gpu_snapshot_before": preflight["gpu_snapshot"],
        "execution_lane": claim["execution_lane"],
        "training_environment_sha256": preflight["training_environment_sha256"],
        "explicit_pythonpath": preflight["environment"]["PYTHONPATH"],
        "kit_thread_caps": manifest["runtime"]["kit_thread_cap_contract"],
        "verified_inputs": preflight["verified_inputs"],
        "runtime_closure": preflight["runtime_closure"],
        "automatic_retry": False,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "real_robot_commands_forbidden": True,
    }
    write_json_exclusive(paths["launch"], launch_contract)
    environment = preflight["environment"].copy()
    environment.update({
        "KIT_BOOT_MARKER": HARD_CONTRACT_MARKER,
        "KIT_BOOT_TIMEOUT_S": str(manifest["runtime"]["kit_boot_timeout_seconds"]),
        "KIT_BOOT_POLL_S": str(manifest["runtime"]["poll_seconds"]),
        "KIT_BOOT_STATE_FILE": str(paths["state"]),
    })
    completed = subprocess.run(
        [str(preflight["locked"]), str(paths["log"]), *command],
        cwd=preflight["wbt"], env=environment, check=False,
    )
    if completed.returncode != 0:
        failure = {
            "artifact_kind": "phase1_signed_face_c2_d2_l1_launch_failure",
            "schema_version": 1,
            "manifest_sha256": manifest_sha,
            "launcher_sha256": launcher_sha,
            "cell_id": cell_id,
            "wrapper_returncode": completed.returncode,
            "launch_contract_sha256": sha256_file(paths["launch"]),
            "automatic_retry": False,
            "manual_diagnosis_required": True,
        }
        if paths["state"].is_file():
            failure["launch_state_sha256"] = sha256_file(paths["state"])
        if paths["log"].is_file():
            failure["training_log_sha256"] = sha256_file(paths["log"])
        write_json_exclusive(paths["failure"], failure)
        raise ContractError(
            f"{cell_id} reviewed wrapper failed rc={completed.returncode}; evidence preserved, no retry"
        )
    state = parse_state(paths["state"])
    if not state.get("pid", "").isdigit() or state.get("pgid") != state.get("pid"):
        raise ContractError(f"{cell_id} wrapper did not record isolated pid==pgid")
    pid = int(state["pid"])
    starttime = process_starttime(pid)
    if starttime <= 0:
        raise ContractError(f"{cell_id} trainer exited before runtime verification")
    run_dir = locate_training_run(preflight["wbt"], cells(manifest)[cell_id]["run_name"])
    run_stat = require_directory(run_dir, f"{cell_id} training run directory")
    contract_path = run_dir / "params/training_contract.json"
    contract_stat = require_regular(contract_path, f"{cell_id} adjacent hard contract")
    hard_sha, _ = verify_hard_contract(contract_path, manifest, cell_id)
    checkpoint = run_dir / "model_24.pt"
    if checkpoint.exists():
        raise ContractError(f"{cell_id} terminal checkpoint appeared before runtime verification")
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    if THREAD_MARKER not in log_text:
        raise ContractError(f"{cell_id} log lacks verified 16/16 thread marker")
    runtime_verified = {
        "artifact_kind": "phase1_signed_face_c2_d2_l1_runtime_verified",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "training_source": expected_source_identity(manifest),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "command": command,
        "external_control": control,
        "execution_lane": claim["execution_lane"],
        "training_environment_sha256": preflight["training_environment_sha256"],
        "pid": pid,
        "pgid": pid,
        "process_starttime_ticks": starttime,
        "training_run_dir": str(run_dir),
        "training_run_directory_identity": identity(run_stat),
        "hard_contract_path": str(contract_path),
        "hard_contract_file_identity": identity(contract_stat),
        "hard_contract_sha256": hard_sha,
        "checkpoint_absent_at_runtime_verification": True,
        "explicit_pythonpath": preflight["environment"]["PYTHONPATH"],
        "kit_thread_caps_verified": True,
        "activation_started": False,
        "judge_started": False,
        "l2_started": False,
        "second_seed_started": False,
        "real_robot_commands_executed": False,
    }
    write_json_exclusive(paths["runtime"], runtime_verified)
    return {
        "status": "one_arm_claimed_and_runtime_verified",
        "cell_id": cell_id,
        "pid": pid,
        "training_launch_claim_sha256": claim_sha,
        "hard_contract_sha256": hard_sha,
        "next_action": (
            "launch-next D2 on its distinct empty GPU after this C2 runtime verification"
            if cell_id == "C2"
            else "wait for natural exits, then finalize each cell on its assigned empty GPU"
        ),
    }


def finalize_cell(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path, cell_id: str,
) -> dict[str, Any]:
    if cell_id not in CELL_IDS:
        raise ContractError("finalize-cell requires C2 or D2")
    preflight = verify_runtime(manifest, cell_id, require_empty_gpu=True)
    paths = expected_arm_paths(manifest, cell_id)
    if paths["result"].exists():
        raise ContractError(f"{cell_id} terminal result already exists; no-clobber finalization")
    result = audit_terminal(manifest, manifest_path, launcher_path, preflight, cell_id)
    write_json_exclusive(paths["result"], result)
    return {
        "status": "l1_terminal_checkpoint_finite_exactly_bound",
        "cell_id": cell_id,
        "terminal_result_path": str(paths["result"]),
        "terminal_result_sha256": sha256_file(paths["result"]),
        "hard_contract_sha256": result["hard_contract_sha256"],
        "training_launch_claim_sha256": result["training_launch_claim_sha256"],
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
    }


def finalize_pair(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
) -> dict[str, Any]:
    results = {
        cell_id: load_and_verify_terminal_result(
            manifest, manifest_path, launcher_path,
            verify_runtime(manifest, cell_id, require_empty_gpu=True), cell_id
        ) for cell_id in CELL_IDS
    }
    contracts = {
        cell_id: read_json_object(Path(result["hard_contract_path"]), f"{cell_id} hard contract")
        for cell_id, result in results.items()
    }
    verify_pair_contracts(contracts)
    if results["C2"]["hard_contract_sha256"] == results["D2"]["hard_contract_sha256"]:
        raise ContractError("distinct guidance facts must produce distinct hard-contract SHAs")
    output = Path(manifest["runtime"]["run_root"]) / manifest["runtime"]["pair_result_basename"]
    if output.exists():
        raise ContractError("paired L1 result already exists; no-clobber finalization")
    content = {
        "artifact_kind": "phase1_signed_face_c2_d2_l1_paired_provenance_result",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "launcher_sha256": sha256_file(launcher_path),
        "training_source": expected_source_identity(manifest),
        "ordered_cells": list(CELL_IDS),
        "execution_lanes": {
            cell_id: results[cell_id]["execution_lane"] for cell_id in CELL_IDS
        },
        "same_host_source_runtime_distinct_gpus": True,
        "only_hard_contract_difference": "racket_guidance_reward.signed_face.weight",
        "hard_contract_sha256_by_cell": {
            cell_id: results[cell_id]["hard_contract_sha256"] for cell_id in CELL_IDS
        },
        "terminal_checkpoint_sha256_by_cell": {
            cell_id: results[cell_id]["terminal_checkpoint_sha256"] for cell_id in CELL_IDS
        },
        "terminal_result_sha256_by_cell": {
            cell_id: sha256_file(expected_arm_paths(manifest, cell_id)["result"])
            for cell_id in CELL_IDS
        },
        "both_model_24_finite_iter24_lineage1": True,
        "checkpoint_binds_adjacent_hard_contract_and_outer_claim_source": True,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "same_immutable_signed_paper_still_required": True,
        "real_robot_commands_executed": False,
    }
    write_json_exclusive(output, content)
    return {
        "status": "paired_l1_provenance_complete_decision_still_blocked",
        "paired_result_path": str(output),
        "paired_result_sha256": sha256_file(output),
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
    }


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--manifest", type=Path,
        default=root / "configs/phase1_signed_face_cd_l1_prereg_20260714.json",
    )
    value.add_argument(
        "--mode",
        choices=("plan", "static-validate", "validate-runtime", "launch-next", "finalize-cell", "finalize-pair"),
        default="plan",
    )
    value.add_argument("--cell", choices=CELL_IDS)
    value.add_argument("--root-confirm")
    value.add_argument("--plan-output", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    launcher_path = Path(__file__).resolve()
    try:
        manifest = load_manifest(manifest_path)
        if args.mode == "static-validate":
            result = {"status": "static_valid_no_writes", "training_source": verify_static_source(manifest)}
        elif args.mode == "plan":
            result = build_plan(manifest, manifest_path, launcher_path)
            if args.plan_output is not None:
                write_json_exclusive(args.plan_output.resolve(), result)
        elif args.mode == "validate-runtime":
            verify_external_control_location(manifest, manifest_path, launcher_path)
            preflights = {
                cell_id: verify_runtime(manifest, cell_id, require_empty_gpu=True)
                for cell_id in CELL_IDS
            }
            result = {
                "status": "runtime_validated_no_launch",
                "gpu_snapshot_by_cell": {
                    cell_id: preflights[cell_id]["gpu_snapshot"] for cell_id in CELL_IDS
                },
                "training_module_path": preflights["C2"]["training_module_path"],
                "explicit_pythonpath": preflights["C2"]["environment"]["PYTHONPATH"],
                "same_explicit_pythonpath": (
                    preflights["C2"]["environment"]["PYTHONPATH"]
                    == preflights["D2"]["environment"]["PYTHONPATH"]
                ),
            }
        elif args.mode == "launch-next":
            result = launch_next(manifest, manifest_path, launcher_path, args.root_confirm)
        elif args.mode == "finalize-cell":
            if args.cell is None:
                raise ContractError("finalize-cell requires --cell C2 or --cell D2")
            result = finalize_cell(manifest, manifest_path, launcher_path, args.cell)
        else:
            result = finalize_pair(manifest, manifest_path, launcher_path)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (ContractError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[signed-face-c2-d2-l1] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
