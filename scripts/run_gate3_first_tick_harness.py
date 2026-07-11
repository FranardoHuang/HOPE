#!/usr/bin/env python3
"""Content-bound, exact-PID/PGID vendor Gate3 first-tick harness.

The default mode is a read-only plan.  Runtime requires an explicit arming phrase and always
forces the production runner into process-wide no-publish mode.  Runtime is simulation-only:
it never authorizes a robot, never searches for a process to kill, and only signals process
groups created by this invocation after revalidating their ownership token and /proc identity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping, Sequence


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_KEYS = (
    "vendor_sim_binary",
    "vendor_sim_config",
    "vendor_mjcf",
    "planner_binary",
    "planner_config",
    "runner_binary",
    "runner_runtime_config",
    "runner_model",
    "kit_binary",
)
EXECUTABLE_ARTIFACTS = {
    "vendor_sim_binary", "planner_binary", "runner_binary", "kit_binary"
}
COMPONENT_ORDER = ("vendor_sim", "planner", "runner")
CONFLICT_ARTIFACT_KEYS = (
    "kit_binary", "vendor_sim_binary", "planner_binary", "runner_binary"
)
REQUIRED_ENV_KEYS = {
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "AMENT_PREFIX_PATH",
    "HOME",
    "LANG",
    "RMW_IMPLEMENTATION",
    "ROS_DOMAIN_ID",
    "ROS_LOCALHOST_ONLY",
    "A3_SOURCE_ROBOT_ENV",
    "A3_HARDWARE_ALLOWED",
    "A3_TRANSPORT",
    "MUJOCO_GL",
}
ARMING_PHRASE = "I_UNDERSTAND_VENDOR_SIM_FIRST_TICK_NO_PUBLISH"
OWNERSHIP_ENV_KEY = "HOPE_GATE3_HARNESS_TOKEN"
FIRST_TICK_MARKER = "[pp FIRST-TICK DEBUG]"
FIRST_TICK_OUTPUT_FLAG = "--first-tick-json"
FIRST_TICK_OUTPUT_PLACEHOLDER = "{HARNESS_FIRST_TICK_JSON}"
FIRST_TICK_VECTOR_LENGTHS = {
    "qpos": 38,
    "qvel": 37,
    "base_pose": 7,
    "racket_pose": 7,
    "obs": 179,
}
FIRST_TICK_TARGET_FIELDS = (
    "position", "velocity", "normal", "rho", "time_to_strike", "swing_type", "valid"
)
FIRST_TICK_QPOS_LAYOUT = "free_xyz_quat_wxyz_then_31_joint_names"
FIRST_TICK_QVEL_LAYOUT = "free_linear_xyz_angular_xyz_then_31_joint_names"
FIRST_TICK_POSE_QUATERNION_ORDER = "wxyz"
FIRST_TICK_TARGET_FRAME = "world_table"
FIRST_TICK_OBS_CONTRACT = "deploy_parity_face179"
DECISION_POLICY = {
    "behavior_arbiter": "agibot_vendor_mujoco_gate3_gate3b",
    "isaac_role": "training_and_diagnostic_only",
    "first_tick_scope": "runtime_precondition_not_behavior_pass",
    "promotion_requires": "vendor_gate3_gate3b_behavior_evidence",
}
ENGINE_GAP_DIAGNOSTIC_LADDER = [
    {"stage": "kinematic_replay", "status": "not_run", "inference_allowed": False},
    {"stage": "open_loop_action_replay", "status": "not_run", "inference_allowed": False},
    {
        "stage": "external_observation_closed_loop",
        "status": "not_run",
        "inference_allowed": False,
    },
    {"stage": "native_closed_loop", "status": "not_run", "inference_allowed": False},
]
READY_STATE_DIAGNOSTIC = {
    "status": "preregistered_not_run",
    "paper": "same_immutable_k100",
    "formal_result_allowed": False,
    "cells": [
        "vendor_stand",
        "root_only_isaac_match",
        "joints_only_isaac_match",
        "full_isaac_match",
    ],
    "all_cells_role": "inexact_causal_diagnostic",
    "training_reset": {
        "pelvis_xyz": [0.0, 0.0, 1.0684],
        "joint_pose": "training_default_q",
    },
    "vendor_stand_observation": {
        "pelvis_xyz": [-0.0416378, 0.000359, 1.06839],
        "pelvis_rpy_deg_approx": [-0.030, 0.249, 0.042],
        "mapped_joint_l2_rad": 0.171845,
        "head_yaw_delta_rad": -0.169416,
        "mapped_joint_l2_without_head_rad": 0.028789,
    },
    "hypothesis_only": (
        "Stage1 contact_pos is env-origin absolute while 175/179 target position is relative "
        "to current racket FK, so root-x mismatch may contribute and must not be assumed to cancel"
    ),
}


class HarnessError(RuntimeError):
    """The contract or runtime violated a fail-closed harness invariant."""


class HarnessInterrupted(HarnessError):
    """A trapped signal requested exact-owned-process cleanup."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise HarnessError(f"non-finite JSON constant {value!r}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"JSON root must be an object: {path}")
    return value


def exact_keys(value: Mapping[str, Any], expected: set[str], owner: str) -> None:
    actual = set(value)
    if actual != expected:
        raise HarnessError(
            f"{owner} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HarnessError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise HarnessError(f"no-clobber output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def absolute_path(raw: Any, owner: str, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\n" in raw:
        raise HarnessError(f"{owner} must be a non-empty path without NUL/newline")
    path = Path(raw)
    if not path.is_absolute():
        raise HarnessError(f"{owner} must be absolute: {raw!r}")
    if must_exist and not path.exists():
        raise HarnessError(f"{owner} does not exist: {path}")
    return path


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HarnessError(f"git read failed for {repo}: {exc}") from exc
    if completed.returncode != 0:
        raise HarnessError(f"git {' '.join(args)} failed for {repo}: {completed.stdout.strip()}")
    return completed.stdout.strip()


def current_source_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    return _git(root, "rev-parse", "HEAD")


def validate_read_only_checkout(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(spec, {"path", "commit"}, f"read_only_checkouts.{name}")
    path = absolute_path(spec["path"], f"read_only_checkouts.{name}.path")
    commit = spec["commit"]
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise HarnessError(f"read_only_checkouts.{name}.commit must be lowercase SHA-1")
    head = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain=v1", "--untracked-files=normal")
    if head != commit or status:
        raise HarnessError(
            f"read-only {name} checkout changed: head={head} expected={commit} dirty={bool(status)}"
        )
    return {"path": str(path), "commit": head, "clean": True}


def _artifact_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mode": stat.st_mode,
    }


def validate_artifact(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(spec, {"path", "sha256", "executable"}, f"artifacts.{name}")
    path = absolute_path(spec["path"], f"artifacts.{name}.path")
    if path.is_symlink():
        raise HarnessError(f"artifacts.{name}.path must not be a symlink: {path}")
    if not path.is_file():
        raise HarnessError(f"artifacts.{name}.path must be a regular file: {path}")
    expected_sha = spec["sha256"]
    if not isinstance(expected_sha, str) or not HEX64.fullmatch(expected_sha):
        raise HarnessError(f"artifacts.{name}.sha256 must be lowercase SHA-256")
    executable = spec["executable"]
    if not isinstance(executable, bool) or executable != (name in EXECUTABLE_ARTIFACTS):
        raise HarnessError(f"artifacts.{name}.executable changed from the role contract")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise HarnessError(
            f"artifacts.{name} SHA mismatch: observed={observed_sha} expected={expected_sha}"
        )
    if executable and not os.access(path, os.X_OK):
        raise HarnessError(f"artifacts.{name} is not executable: {path}")
    return {
        "path": str(path),
        "sha256": observed_sha,
        "executable": executable,
        "stat": _artifact_stat(path),
    }


def revalidate_artifacts(
    contract: Mapping[str, Any], accepted: Mapping[str, Mapping[str, Any]]
) -> None:
    for name in ARTIFACT_KEYS:
        current = validate_artifact(name, contract["artifacts"][name])
        if current != accepted[name]:
            raise HarnessError(f"artifact changed after plan validation: {name}")


def _validate_env_path_list(name: str, value: str, *, allow_empty: bool) -> None:
    if not value:
        if allow_empty:
            return
        raise HarnessError(f"runtime.environment.{name} must not be empty")
    parts = value.split(":")
    if any(not part for part in parts):
        raise HarnessError(f"runtime.environment.{name} contains an empty path element")
    for index, part in enumerate(parts):
        path = absolute_path(part, f"runtime.environment.{name}[{index}]")
        if not path.is_dir():
            raise HarnessError(f"runtime.environment.{name}[{index}] is not a directory")


def validate_environment(env: Mapping[str, Any], ros_domain_id: int) -> dict[str, str]:
    exact_keys(env, REQUIRED_ENV_KEYS, "runtime.environment")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()):
        raise HarnessError("runtime.environment must contain only string keys and values")
    if env["ROS_DOMAIN_ID"] != str(ros_domain_id):
        raise HarnessError("ROS_DOMAIN_ID disagrees with runtime.ros_domain_id")
    if env["ROS_LOCALHOST_ONLY"] != "1":
        raise HarnessError("ROS_LOCALHOST_ONLY must be 1 for the simulation-only harness")
    if env["A3_SOURCE_ROBOT_ENV"] != "0" or env["A3_HARDWARE_ALLOWED"] != "0":
        raise HarnessError("robot environment/hardware authority must both remain disabled")
    if env["A3_TRANSPORT"] != "iceoryx":
        raise HarnessError("A3_TRANSPORT must be the bound vendor-sim iceoryx transport")
    if not env["RMW_IMPLEMENTATION"].startswith("rmw_"):
        raise HarnessError("RMW_IMPLEMENTATION must be explicit")
    if env["MUJOCO_GL"] != "egl":
        raise HarnessError("MUJOCO_GL must be egl; interactive viewer/Kit is forbidden")
    _validate_env_path_list("PATH", env["PATH"], allow_empty=False)
    for name in ("LD_LIBRARY_PATH", "PYTHONPATH", "AMENT_PREFIX_PATH"):
        _validate_env_path_list(name, env[name], allow_empty=True)
    home = absolute_path(env["HOME"], "runtime.environment.HOME")
    if not home.is_dir():
        raise HarnessError("runtime.environment.HOME must be a directory")
    if not env["LANG"]:
        raise HarnessError("runtime.environment.LANG must be explicit")
    return dict(env)


def _validate_argv(argv: Any, owner: str) -> list[str]:
    if not isinstance(argv, list) or not argv:
        raise HarnessError(f"{owner}.argv must be a non-empty array")
    if any(
        not isinstance(item, str) or not item or "\x00" in item or "\n" in item
        for item in argv
    ):
        raise HarnessError(f"{owner}.argv entries must be non-empty strings without NUL/newline")
    return list(argv)


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) > 1:
        raise HarnessError(f"duplicate command flag {flag}")
    if not positions:
        return None
    index = positions[0]
    if index + 1 >= len(argv):
        raise HarnessError(f"command flag {flag} lacks a value")
    return argv[index + 1]


def validate_commands(
    commands: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]], checkouts: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    exact_keys(commands, set(COMPONENT_ORDER), "runtime.commands")
    result: dict[str, dict[str, Any]] = {}
    expected_binary = {
        "vendor_sim": "vendor_sim_binary",
        "planner": "planner_binary",
        "runner": "runner_binary",
    }
    for role in COMPONENT_ORDER:
        spec = commands[role]
        if not isinstance(spec, dict):
            raise HarnessError(f"runtime.commands.{role} must be an object")
        exact_keys(spec, {"argv", "cwd"}, f"runtime.commands.{role}")
        argv = _validate_argv(spec["argv"], f"runtime.commands.{role}")
        binary = artifacts[expected_binary[role]]["path"]
        if argv[0] != binary:
            raise HarnessError(f"{role} argv[0] must be its SHA-bound executable")
        bound_paths = {artifact["path"] for artifact in artifacts.values()}
        unbound_absolute = [value for value in argv[1:] if value.startswith("/") and value not in bound_paths]
        if unbound_absolute:
            raise HarnessError(
                f"{role} argv contains unbound absolute path tokens: {unbound_absolute}"
            )
        cwd = absolute_path(spec["cwd"], f"runtime.commands.{role}.cwd")
        if cwd.is_symlink() or not cwd.is_dir():
            raise HarnessError(f"runtime.commands.{role}.cwd must be a real directory")
        for checkout in checkouts.values():
            if is_under(cwd, Path(checkout["path"])):
                raise HarnessError(f"{role} cwd must not write inside a train/eval checkout")
        result[role] = {"argv": argv, "cwd": str(cwd)}

    sim_argv = result["vendor_sim"]["argv"]
    if artifacts["vendor_sim_config"]["path"] not in sim_argv:
        raise HarnessError("vendor sim argv must contain the SHA-bound sim config as one exact token")
    planner_argv = result["planner"]["argv"]
    if artifacts["planner_config"]["path"] not in planner_argv:
        raise HarnessError("planner argv must contain the SHA-bound planner config as one exact token")
    runner_argv = result["runner"]["argv"]
    for required in (
        artifacts["runner_runtime_config"]["path"], artifacts["runner_model"]["path"],
        "--planner", "--no-publish",
    ):
        if required not in runner_argv:
            raise HarnessError(f"runner argv is missing exact no-publish binding {required!r}")
    if runner_argv.count("--no-publish") != 1:
        raise HarnessError("runner argv must contain --no-publish exactly once")
    if "--model-preflight-only" in runner_argv:
        raise HarnessError("runtime runner command must reach first tick, not repeat loader-only mode")
    if _flag_value(runner_argv, "--runtime-cfg") != artifacts["runner_runtime_config"]["path"]:
        raise HarnessError("runner --runtime-cfg is not the bound file")
    if _flag_value(runner_argv, "--model-path") != artifacts["runner_model"]["path"]:
        raise HarnessError("runner --model-path is not the bound model")
    if _flag_value(runner_argv, "--start") != "passive":
        raise HarnessError("first-tick harness must start passive; motion/stand/shadow are forbidden")
    if _flag_value(runner_argv, FIRST_TICK_OUTPUT_FLAG) != FIRST_TICK_OUTPUT_PLACEHOLDER:
        raise HarnessError(
            "runner must bind --first-tick-json to the harness-owned output placeholder"
        )
    forbidden = {
        "--dry-run", "--obs-csv", "--trace-csv", "--reference-playback", "--oracle-pelvis"
    }
    if forbidden.intersection(runner_argv):
        raise HarnessError(f"runner argv contains out-of-scope flags: {sorted(forbidden.intersection(runner_argv))}")
    if any(value.startswith("--start=") for value in runner_argv):
        raise HarnessError("runner argv must not contain a second --start= selector")
    return result


def build_formal_loader_argv(artifacts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        artifacts["runner_binary"]["path"],
        "--runtime-cfg", artifacts["runner_runtime_config"]["path"],
        "--model-path", artifacts["runner_model"]["path"],
        "--planner",
        "--no-publish",
        "--model-preflight-only",
    ]


def validate_contract(data: dict[str, Any]) -> dict[str, Any]:
    exact_keys(
        data,
        {
            "schema_version", "contract_id", "created_utc", "status", "scope",
            "source_commit", "harness_sha256", "hardware_authorized", "artifacts",
            "read_only_checkouts", "formal_loader", "first_tick_evidence", "runtime",
            "activation", "decision_policy", "engine_gap_diagnostic_ladder",
            "ready_state_diagnostic",
        },
        "Gate3 first-tick contract",
    )
    if (
        data["schema_version"] != 1
        or data["status"] != "preregistered_not_run"
        or data["scope"] != "vendor_gate3_first_tick_no_publish_only"
        or data["hardware_authorized"] is not False
    ):
        raise HarnessError("contract must remain preregistered, first-tick-only, and hardware-forbidden")
    if not isinstance(data["contract_id"], str) or not data["contract_id"]:
        raise HarnessError("contract_id must be non-empty")
    if not isinstance(data["created_utc"], str) or not data["created_utc"].endswith("Z"):
        raise HarnessError("created_utc must be an explicit UTC string")
    if not isinstance(data["source_commit"], str) or not HEX40.fullmatch(data["source_commit"]):
        raise HarnessError("source_commit must be lowercase SHA-1")
    observed_source_commit = current_source_commit()
    if data["source_commit"] != observed_source_commit:
        raise HarnessError(
            f"source_commit changed: contract={data['source_commit']} checkout={observed_source_commit}"
        )
    if not isinstance(data["harness_sha256"], str) or not HEX64.fullmatch(data["harness_sha256"]):
        raise HarnessError("harness_sha256 must be lowercase SHA-256")
    if data["harness_sha256"] != sha256_file(Path(__file__).resolve()):
        raise HarnessError("contract does not bind these harness source bytes")

    artifacts_raw = data["artifacts"]
    if not isinstance(artifacts_raw, dict):
        raise HarnessError("artifacts must be an object")
    exact_keys(artifacts_raw, set(ARTIFACT_KEYS), "artifacts")
    artifacts = {
        name: validate_artifact(name, artifacts_raw[name]) for name in ARTIFACT_KEYS
    }

    checkouts_raw = data["read_only_checkouts"]
    if not isinstance(checkouts_raw, dict):
        raise HarnessError("read_only_checkouts must be an object")
    exact_keys(checkouts_raw, {"training", "evaluation"}, "read_only_checkouts")
    checkouts = {
        name: validate_read_only_checkout(name, checkouts_raw[name])
        for name in ("training", "evaluation")
    }

    for artifact in artifacts.values():
        for checkout in checkouts.values():
            if is_under(Path(artifact["path"]), Path(checkout["path"])):
                raise HarnessError("runtime artifacts must be external copies, not files in train/eval checkouts")

    formal = data["formal_loader"]
    if formal != {
        "required": True,
        "required_output_substrings": [
            "[pp PREFLIGHT] accepted", "backend_not_initialized=true", "obs_dim=179"
        ],
        "forbidden_output_substrings": [
            "backend cfg", "A3AimrtBackend initialised", "backend started"
        ],
        "requires_no_publish": True,
    }:
        raise HarnessError("formal loader contract changed or no longer requires exact 179/no-publish")

    first_tick = data["first_tick_evidence"]
    if first_tick != {
        "runner_flag": FIRST_TICK_OUTPUT_FLAG,
        "output_placeholder": FIRST_TICK_OUTPUT_PLACEHOLDER,
        "schema_version": 1,
        "required_vector_lengths": FIRST_TICK_VECTOR_LENGTHS,
        "required_joint_count": 31,
        "qpos_layout": FIRST_TICK_QPOS_LAYOUT,
        "qvel_layout": FIRST_TICK_QVEL_LAYOUT,
        "pose_quaternion_order": FIRST_TICK_POSE_QUATERNION_ORDER,
        "target_frame": FIRST_TICK_TARGET_FRAME,
        "obs_contract": FIRST_TICK_OBS_CONTRACT,
        "required_target_fields": list(FIRST_TICK_TARGET_FIELDS),
        "require_all_finite": True,
    }:
        raise HarnessError("first-tick full-state evidence contract changed")

    runtime = data["runtime"]
    if not isinstance(runtime, dict):
        raise HarnessError("runtime must be an object")
    exact_keys(
        runtime,
        {
            "ros_domain_id", "ledger_root", "lock_path", "conflict_locks",
            "conflict_artifact_keys", "environment", "timeouts_s", "readiness_substrings",
            "commands", "transport_scope", "body_command_publish_allowed",
        },
        "runtime",
    )
    domain = runtime["ros_domain_id"]
    if isinstance(domain, bool) or not isinstance(domain, int) or domain < 0 or domain > 232:
        raise HarnessError("runtime.ros_domain_id must be an integer in [0,232]")
    if runtime["transport_scope"] != "vendor_sim_only_no_hardware":
        raise HarnessError("runtime.transport_scope must stay vendor-sim-only")
    if runtime["body_command_publish_allowed"] is not False:
        raise HarnessError("body-command publishing must remain forbidden")
    environment = validate_environment(runtime["environment"], domain)
    ledger_root = absolute_path(runtime["ledger_root"], "runtime.ledger_root")
    if ledger_root.is_symlink() or not ledger_root.is_dir():
        raise HarnessError("runtime.ledger_root must be an existing real directory")
    lock_path = absolute_path(runtime["lock_path"], "runtime.lock_path", must_exist=False)
    if lock_path.parent != ledger_root:
        raise HarnessError("runtime.lock_path must live directly under ledger_root")
    for checkout in checkouts.values():
        if is_under(ledger_root, Path(checkout["path"])):
            raise HarnessError("ledger_root must be outside train/eval checkouts")
    conflict_locks = runtime["conflict_locks"]
    if not isinstance(conflict_locks, dict):
        raise HarnessError("runtime.conflict_locks must map exact Kit/sim/planner/runner roles")
    exact_keys(conflict_locks, {"kit", "vendor_sim", "planner", "runner"}, "runtime.conflict_locks")
    resolved_locks: dict[str, str] = {}
    for role, raw in conflict_locks.items():
        path = absolute_path(raw, f"runtime.conflict_locks.{role}", must_exist=False)
        if path == lock_path:
            raise HarnessError("own harness lock must not be duplicated as a conflict lock")
        resolved_locks[role] = str(path)
    if len(set(resolved_locks.values())) != len(resolved_locks):
        raise HarnessError("runtime.conflict_locks contains duplicates")
    if runtime["conflict_artifact_keys"] != list(CONFLICT_ARTIFACT_KEYS):
        raise HarnessError("exact Kit/sim/planner/runner conflict scan roles changed")

    timeouts = runtime["timeouts_s"]
    expected_timeouts = {
        "formal_loader", "vendor_sim_ready", "planner_ready", "runner_first_tick", "term", "kill"
    }
    if not isinstance(timeouts, dict):
        raise HarnessError("runtime.timeouts_s must be an object")
    exact_keys(timeouts, expected_timeouts, "runtime.timeouts_s")
    for name, value in timeouts.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise HarnessError(f"runtime.timeouts_s.{name} must be finite")
        if value <= 0 or value > 300:
            raise HarnessError(f"runtime.timeouts_s.{name} must be in (0,300]")
    readiness = runtime["readiness_substrings"]
    if not isinstance(readiness, dict):
        raise HarnessError("runtime.readiness_substrings must be an object")
    exact_keys(readiness, set(COMPONENT_ORDER), "runtime.readiness_substrings")
    for name, value in readiness.items():
        if not isinstance(value, str) or not value or len(value) > 256 or "\x00" in value:
            raise HarnessError(f"runtime.readiness_substrings.{name} is invalid")
    if readiness["runner"] != FIRST_TICK_MARKER:
        raise HarnessError("runner readiness must be the production first-tick debug marker")

    commands = validate_commands(runtime["commands"], artifacts, checkouts)
    try:
        config_text = Path(artifacts["vendor_sim_config"]["path"]).read_text(
            encoding="utf-8", errors="strict"
        )
    except (OSError, UnicodeError) as exc:
        raise HarnessError(f"cannot inspect vendor sim config/MJCF binding: {exc}") from exc
    if artifacts["vendor_mjcf"]["path"] not in config_text:
        raise HarnessError("vendor sim config does not contain the exact bound MJCF absolute path")

    activation = data["activation"]
    if activation != {
        "default_mode": "plan",
        "run_cli_arming_phrase": ARMING_PHRASE,
        "no_publish_required": True,
        "real_robot_authorized": False,
    }:
        raise HarnessError("activation fence changed")
    if data["decision_policy"] != DECISION_POLICY:
        raise HarnessError("vendor Gate3 decision authority or Isaac diagnostic role changed")
    if data["engine_gap_diagnostic_ladder"] != ENGINE_GAP_DIAGNOSTIC_LADDER:
        raise HarnessError("engine-gap causal ladder changed or contains an unearned inference")
    if data["ready_state_diagnostic"] != READY_STATE_DIAGNOSTIC:
        raise HarnessError("ready-state inexact four-cell diagnostic changed or was promoted")
    return {
        "contract_id": data["contract_id"],
        "source_commit": data["source_commit"],
        "harness_sha256": data["harness_sha256"],
        "artifacts": artifacts,
        "checkouts": checkouts,
        "formal_loader_argv": build_formal_loader_argv(artifacts),
        "first_tick_evidence": dict(first_tick),
        "decision_policy": dict(DECISION_POLICY),
        "engine_gap_diagnostic_ladder": list(ENGINE_GAP_DIAGNOSTIC_LADDER),
        "ready_state_diagnostic": dict(READY_STATE_DIAGNOSTIC),
        "runtime": {
            "ros_domain_id": domain,
            "ledger_root": str(ledger_root),
            "lock_path": str(lock_path),
            "conflict_locks": resolved_locks,
            "environment": environment,
            "timeouts_s": {key: float(value) for key, value in timeouts.items()},
            "readiness_substrings": dict(readiness),
            "commands": commands,
            "transport_scope": runtime["transport_scope"],
            "body_command_publish_allowed": False,
        },
    }


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    ppid: int
    pgid: int
    session: int
    starttime_ticks: int
    cmdline: tuple[str, ...]
    executable: str
    ownership_token: str


def _parse_proc_stat(text: str, pid: int) -> tuple[str, int, int, int, int]:
    close = text.rfind(")")
    if close < 0:
        raise HarnessError(f"malformed /proc/{pid}/stat")
    rest = text[close + 2 :].split()
    if len(rest) <= 19:
        raise HarnessError(f"short /proc/{pid}/stat")
    try:
        state = rest[0]
        ppid = int(rest[1])
        pgid = int(rest[2])
        session = int(rest[3])
        starttime = int(rest[19])
    except ValueError as exc:
        raise HarnessError(f"non-integer /proc/{pid}/stat identity") from exc
    return state, ppid, pgid, session, starttime


def read_process_identity(
    pid: int, token: str, *, proc_root: Path = Path("/proc")
) -> ProcessIdentity:
    base = proc_root / str(pid)
    try:
        stat_text = (base / "stat").read_text(encoding="utf-8")
        cmdline_raw = (base / "cmdline").read_bytes()
        environ_raw = (base / "environ").read_bytes()
        executable = os.readlink(base / "exe")
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise HarnessError(f"process identity disappeared: pid={pid}") from exc
    except (OSError, UnicodeError) as exc:
        raise HarnessError(f"cannot inspect process identity pid={pid}: {exc}") from exc
    state, ppid, pgid, session, starttime = _parse_proc_stat(stat_text, pid)
    if state == "Z":
        raise HarnessError(f"process pid={pid} is already a zombie")
    cmdline = tuple(
        value.decode("utf-8", errors="surrogateescape")
        for value in cmdline_raw.split(b"\0")
        if value
    )
    if not cmdline:
        raise HarnessError(f"process pid={pid} has empty cmdline")
    expected_env = f"{OWNERSHIP_ENV_KEY}={token}".encode()
    environ = {value for value in environ_raw.split(b"\0") if value}
    if expected_env not in environ:
        raise HarnessError(f"process pid={pid} lacks exact ownership token")
    return ProcessIdentity(
        pid=pid,
        ppid=ppid,
        pgid=pgid,
        session=session,
        starttime_ticks=starttime,
        cmdline=cmdline,
        executable=executable,
        ownership_token=token,
    )


def list_group_identities(
    pgid: int, token: str, *, proc_root: Path = Path("/proc")
) -> dict[int, ProcessIdentity]:
    if not proc_root.is_dir():
        raise HarnessError("Linux /proc is unavailable; runtime is forbidden on this host")
    identities: dict[int, ProcessIdentity] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            state, _, observed_pgid, _, _ = _parse_proc_stat(stat_text, pid)
        except (HarnessError, FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        if observed_pgid != pgid or state == "Z":
            continue
        identities[pid] = read_process_identity(pid, token, proc_root=proc_root)
    return identities


def validate_group_before_signal(
    leader: ProcessIdentity,
    token: str,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[int, ProcessIdentity]:
    first = list_group_identities(leader.pgid, token, proc_root=proc_root)
    if not first:
        return {}
    if leader.pid in first and first[leader.pid] != leader:
        raise HarnessError("leader PID was reused or changed before cleanup")
    if any(identity.starttime_ticks < leader.starttime_ticks for identity in first.values()):
        raise HarnessError("process group contains a member older than the owned leader")
    second = list_group_identities(leader.pgid, token, proc_root=proc_root)
    if first != second:
        raise HarnessError("process group changed during pre-signal identity validation")
    return second


def scan_exact_process_conflicts(
    artifact_paths: Mapping[str, str], *, proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    if not proc_root.is_dir():
        return {"supported": False, "conflicts": []}
    targets = {name: str(Path(path).resolve()) for name, path in artifact_paths.items()}
    conflicts: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            tokens = [
                value.decode("utf-8", errors="surrogateescape")
                for value in raw.split(b"\0")
                if value
            ]
            exe = str(Path(os.readlink(entry / "exe")).resolve())
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        for role, target in targets.items():
            if exe == target or target in tokens:
                conflicts.append(
                    {"role": role, "pid": int(entry.name), "executable": exe, "cmdline": tokens}
                )
    return {"supported": True, "conflicts": conflicts}


def preflight_conflicts(plan: Mapping[str, Any], *, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    runtime = plan["runtime"]
    locks = [runtime["lock_path"], *runtime["conflict_locks"].values()]
    existing_locks = [path for path in locks if Path(path).exists()]
    if existing_locks:
        raise HarnessError(
            f"existing exact Kit/sim/planner/runner/harness lock conflict: {existing_locks}"
        )
    paths = {
        name: plan["artifacts"][name]["path"] for name in CONFLICT_ARTIFACT_KEYS
    }
    process_scan = scan_exact_process_conflicts(paths, proc_root=proc_root)
    if process_scan["conflicts"]:
        raise HarnessError(
            "existing exact Kit/sim/planner/runner process conflict: "
            f"{process_scan['conflicts']}"
        )
    return {
        "lock_paths_checked": locks,
        "existing_locks": [],
        "process_scan": process_scan,
        "runtime_eligible_on_host": bool(process_scan["supported"]),
    }


def build_plan(
    contract_path: Path,
    contract_sha: str,
    contract: Mapping[str, Any],
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    accepted = validate_contract(dict(contract))
    conflicts = preflight_conflicts(accepted, proc_root=proc_root)
    return {
        "schema_version": 1,
        "artifact_kind": "gate3_first_tick_no_publish_plan",
        "status": "validated_plan_no_process_started",
        "contract": {"path": str(contract_path), "sha256": contract_sha},
        "source_commit": accepted["source_commit"],
        "harness_sha256": accepted["harness_sha256"],
        "artifacts": accepted["artifacts"],
        "read_only_checkouts": accepted["checkouts"],
        "formal_loader": {
            "required_before_components": True,
            "argv": accepted["formal_loader_argv"],
            "body_command_publish_allowed": False,
        },
        "first_tick_evidence": accepted["first_tick_evidence"],
        "component_order": list(COMPONENT_ORDER),
        "commands": accepted["runtime"]["commands"],
        "runtime": {
            "ros_domain_id": accepted["runtime"]["ros_domain_id"],
            "transport_scope": accepted["runtime"]["transport_scope"],
            "body_command_publish_allowed": False,
            "ledger_root": accepted["runtime"]["ledger_root"],
            "lock_path": accepted["runtime"]["lock_path"],
            "conflict_locks": accepted["runtime"]["conflict_locks"],
            "environment": accepted["runtime"]["environment"],
            "timeouts_s": accepted["runtime"]["timeouts_s"],
            "readiness_substrings": accepted["runtime"]["readiness_substrings"],
        },
        "conflict_preflight": conflicts,
        "activation": {
            "default_mode": "plan",
            "run_requires_exact_cli_phrase": ARMING_PHRASE,
            "armed": False,
            "hardware_authorized": False,
        },
        "decision_policy": accepted["decision_policy"],
        "engine_gap_diagnostic_ladder": accepted["engine_gap_diagnostic_ladder"],
        "ready_state_diagnostic": accepted["ready_state_diagnostic"],
        "actions": {
            "processes_started": [],
            "signals_sent": [],
            "simulator_started": False,
            "transport_started": False,
            "runner_started": False,
            "real_robot_authorized": False,
        },
    }


def authorize_run(arming_value: str | None, plan: Mapping[str, Any]) -> None:
    if arming_value != ARMING_PHRASE:
        raise HarnessError("run mode requires the exact simulation-only no-publish arming phrase")
    if not plan["conflict_preflight"]["runtime_eligible_on_host"]:
        raise HarnessError("run mode requires Linux /proc exact-identity support")
    if plan["runtime"]["body_command_publish_allowed"] is not False:
        raise HarnessError("run mode cannot authorize body-command publishing")


@dataclass
class ManagedProcess:
    role: str
    popen: subprocess.Popen[bytes]
    identity: ProcessIdentity
    stdout_path: Path
    stderr_path: Path
    started_utc: str
    finished_utc: str | None = None
    returncode: int | None = None
    cleanup: list[dict[str, Any]] | None = None
    known_group_members: dict[int, ProcessIdentity] | None = None

    def ledger_row(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "pid": self.identity.pid,
            "pgid": self.identity.pgid,
            "identity": asdict(self.identity),
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "returncode": self.returncode,
            "stdout": {
                "path": str(self.stdout_path),
                "sha256": sha256_file(self.stdout_path) if self.stdout_path.exists() else None,
            },
            "stderr": {
                "path": str(self.stderr_path),
                "sha256": sha256_file(self.stderr_path) if self.stderr_path.exists() else None,
            },
            "cleanup": self.cleanup or [],
        }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def launch_owned_process(
    role: str,
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    token: str,
    run_dir: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> ManagedProcess:
    stdout_path = run_dir / f"{role}.stdout.log"
    stderr_path = run_dir / f"{role}.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise HarnessError(f"no-clobber logs already exist for {role}")
    child_env = dict(env)
    child_env[OWNERSHIP_ENV_KEY] = token
    with stdout_path.open("xb", buffering=0) as stdout, stderr_path.open("xb", buffering=0) as stderr:
        try:
            proc = subprocess.Popen(
                list(argv),
                cwd=str(cwd),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                shell=False,
            )
        except OSError as exc:
            raise HarnessError(f"failed to start {role}: {exc}") from exc
    identity: ProcessIdentity | None = None
    last_error: Exception | None = None
    for _ in range(100):
        try:
            identity = read_process_identity(proc.pid, token, proc_root=proc_root)
            break
        except HarnessError as exc:
            last_error = exc
            if proc.poll() is not None:
                break
            time.sleep(0.01)
    if identity is None:
        raise HarnessError(f"could not record exact identity for {role} pid={proc.pid}: {last_error}")
    if identity.pid != identity.pgid or identity.session != identity.pid:
        raise HarnessError(f"{role} did not receive its own exact session/process group")
    return ManagedProcess(
        role=role,
        popen=proc,
        identity=identity,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_utc=_utc_now(),
        cleanup=[],
    )


def _read_logs(process: ManagedProcess) -> str:
    output = b""
    for path in (process.stdout_path, process.stderr_path):
        try:
            output += path.read_bytes()
        except FileNotFoundError:
            pass
    return output.decode("utf-8", errors="replace")


def _finite_vector(value: Any, owner: str, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise HarnessError(f"{owner} must contain exactly {expected} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise HarnessError(f"{owner}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise HarnessError(f"{owner}[{index}] is non-finite")
        result.append(number)
    return result


def validate_first_tick_trace(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise HarnessError("runner did not produce a regular first-tick JSON artifact")
    value = load_json(path)
    exact_keys(
        value,
        {
            "schema_version", "source", "tick", "qpos", "qvel", "base_pose",
            "racket_pose", "target", "obs", "joint_names", "qpos_layout", "qvel_layout",
            "pose_quaternion_order", "target_frame", "obs_contract",
        },
        "first-tick JSON",
    )
    if value["schema_version"] != 1 or value["source"] != "production_runner_first_tick":
        raise HarnessError("first-tick JSON schema/source changed")
    if value["qpos_layout"] != FIRST_TICK_QPOS_LAYOUT or value["qvel_layout"] != FIRST_TICK_QVEL_LAYOUT:
        raise HarnessError("first-tick qpos/qvel layout changed")
    if (
        value["pose_quaternion_order"] != FIRST_TICK_POSE_QUATERNION_ORDER
        or value["target_frame"] != FIRST_TICK_TARGET_FRAME
        or value["obs_contract"] != FIRST_TICK_OBS_CONTRACT
    ):
        raise HarnessError("first-tick pose/target/observation frame contract changed")
    joint_names = value["joint_names"]
    if (
        not isinstance(joint_names, list)
        or len(joint_names) != 31
        or any(not isinstance(name, str) or not name for name in joint_names)
        or len(set(joint_names)) != 31
    ):
        raise HarnessError("first-tick joint_names must contain 31 unique non-empty names")
    if isinstance(value["tick"], bool) or not isinstance(value["tick"], int) or value["tick"] < 0:
        raise HarnessError("first-tick JSON tick must be a non-negative integer")
    canonical_fields: dict[str, Any] = {}
    for name, length in FIRST_TICK_VECTOR_LENGTHS.items():
        canonical_fields[name] = _finite_vector(value[name], f"first_tick.{name}", length)
    target = value["target"]
    if not isinstance(target, dict):
        raise HarnessError("first_tick.target must be an object")
    exact_keys(target, set(FIRST_TICK_TARGET_FIELDS), "first_tick.target")
    target_canonical = {
        "position": _finite_vector(target["position"], "first_tick.target.position", 3),
        "velocity": _finite_vector(target["velocity"], "first_tick.target.velocity", 3),
        "normal": _finite_vector(target["normal"], "first_tick.target.normal", 3),
    }
    for name in ("rho", "time_to_strike"):
        item = target[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise HarnessError(f"first_tick.target.{name} must be finite numeric")
        target_canonical[name] = float(item)
    if isinstance(target["swing_type"], bool) or not isinstance(target["swing_type"], int):
        raise HarnessError("first_tick.target.swing_type must be an integer")
    if not isinstance(target["valid"], bool):
        raise HarnessError("first_tick.target.valid must be boolean")
    target_canonical["swing_type"] = target["swing_type"]
    target_canonical["valid"] = target["valid"]
    canonical_fields["target"] = target_canonical
    canonical_fields["joint_names"] = list(joint_names)
    per_field_sha = {
        name: canonical_sha256(canonical_fields[name])
        for name in (
            "joint_names", "qpos", "qvel", "base_pose", "racket_pose", "target", "obs"
        )
    }
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "tick": value["tick"],
        "vector_lengths": {
            name: len(canonical_fields[name]) for name in FIRST_TICK_VECTOR_LENGTHS
        },
        "joint_names_canonical_sha256": per_field_sha["joint_names"],
        "per_field_canonical_sha256": per_field_sha,
        "canonical_trace_sha256": canonical_sha256(
            {
                "tick": value["tick"],
                "qpos_layout": value["qpos_layout"],
                "qvel_layout": value["qvel_layout"],
                "pose_quaternion_order": value["pose_quaternion_order"],
                "target_frame": value["target_frame"],
                "obs_contract": value["obs_contract"],
                **canonical_fields,
            }
        ),
    }


def wait_for_first_tick_trace(path: Path, process: ManagedProcess, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: HarnessError | None = None
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return validate_first_tick_trace(path)
            except HarnessError as exc:
                last_error = exc
        if process.popen.poll() is not None and not path.exists():
            break
        time.sleep(0.05)
    raise HarnessError(f"first-tick full-state JSON missing/invalid: {last_error}")


def wait_for_marker(process: ManagedProcess, marker: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if marker in _read_logs(process):
            return
        rc = process.popen.poll()
        if rc is not None:
            process.returncode = rc
            raise HarnessError(f"{process.role} exited rc={rc} before marker {marker!r}")
        time.sleep(0.05)
    raise HarnessError(f"{process.role} timed out waiting for marker {marker!r}")


def exact_signal_owned_group(
    process: ManagedProcess,
    sig: signal.Signals,
    *,
    proc_root: Path = Path("/proc"),
) -> None:
    members = validate_group_before_signal(
        process.identity, process.identity.ownership_token, proc_root=proc_root
    )
    if not members:
        return
    if process.known_group_members is not None:
        for pid, identity in members.items():
            if process.known_group_members.get(pid) != identity:
                raise HarnessError(
                    "owned process-group member starttime/cmdline/token changed before signal"
                )
    process.known_group_members = dict(members)
    os.killpg(process.identity.pgid, sig)
    assert process.cleanup is not None
    process.cleanup.append(
        {
            "signal": sig.name,
            "pgid": process.identity.pgid,
            "validated_members": [asdict(members[pid]) for pid in sorted(members)],
            "sent_utc": _utc_now(),
        }
    )


def stop_owned_process(
    process: ManagedProcess,
    *,
    term_s: float,
    kill_s: float,
    proc_root: Path = Path("/proc"),
) -> None:
    if process.finished_utc is not None:
        if not list_group_identities(
            process.identity.pgid, process.identity.ownership_token, proc_root=proc_root
        ):
            return
    initial_members = list_group_identities(
        process.identity.pgid, process.identity.ownership_token, proc_root=proc_root
    )
    if process.popen.poll() is None or initial_members:
        exact_signal_owned_group(process, signal.SIGTERM, proc_root=proc_root)
    deadline = time.monotonic() + term_s
    while time.monotonic() < deadline:
        process.popen.poll()
        if not list_group_identities(
            process.identity.pgid, process.identity.ownership_token, proc_root=proc_root
        ):
            break
        time.sleep(0.05)
    remaining = list_group_identities(
        process.identity.pgid, process.identity.ownership_token, proc_root=proc_root
    )
    if remaining:
        exact_signal_owned_group(process, signal.SIGKILL, proc_root=proc_root)
        deadline = time.monotonic() + kill_s
        while time.monotonic() < deadline:
            if not list_group_identities(
                process.identity.pgid, process.identity.ownership_token, proc_root=proc_root
            ):
                break
            time.sleep(0.05)
        if list_group_identities(
            process.identity.pgid, process.identity.ownership_token, proc_root=proc_root
        ):
            raise HarnessError(f"owned {process.role} group did not exit after exact SIGKILL")
    try:
        process.returncode = process.popen.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        process.returncode = process.popen.poll()
    process.finished_utc = _utc_now()


def finish_short_process(
    process: ManagedProcess,
    timeout_s: float,
    *,
    term_s: float,
    kill_s: float,
    proc_root: Path = Path("/proc"),
) -> int:
    try:
        rc = process.popen.wait(timeout=timeout_s)
        process.returncode = rc
        process.finished_utc = _utc_now()
        return rc
    except subprocess.TimeoutExpired:
        stop_owned_process(
            process, term_s=term_s, kill_s=kill_s, proc_root=proc_root
        )
        raise HarnessError(f"{process.role} timed out")


def acquire_lock(path: Path, token: str) -> tuple[int, int]:
    payload = canonical_bytes({"schema_version": 1, "ownership_token": token, "pid": os.getpid()})
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise HarnessError(f"harness lock conflict: {path}") from exc
    try:
        os.write(fd, payload)
        os.fsync(fd)
        inode = os.fstat(fd).st_ino
    finally:
        os.close(fd)
    return inode, len(payload)


def release_owned_lock(path: Path, token: str, inode: int) -> None:
    try:
        stat = path.stat()
        value = load_json(path)
    except FileNotFoundError as exc:
        raise HarnessError("owned harness lock disappeared") from exc
    if stat.st_ino != inode or value != {
        "schema_version": 1, "ownership_token": token, "pid": os.getpid()
    }:
        raise HarnessError("harness lock identity/content changed; refusing unlink")
    path.unlink()


def run_harness(
    contract_path: Path,
    contract_sha: str,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    proc_root: Path = Path("/proc"),
) -> Path:
    accepted = validate_contract(dict(contract))
    runtime_preflight = preflight_conflicts(accepted, proc_root=proc_root)
    revalidate_artifacts(contract, accepted["artifacts"])
    token = uuid.uuid4().hex
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + token[:12]
    run_dir = Path(accepted["runtime"]["ledger_root"]) / run_id
    run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    lock_path = Path(accepted["runtime"]["lock_path"])
    lock_inode, _ = acquire_lock(lock_path, token)
    managed: list[ManagedProcess] = []
    status = "failed"
    error: str | None = None
    interrupted_by: int | None = None
    first_tick_evidence: dict[str, Any] | None = None
    old_handlers: dict[int, Any] = {}

    def trap_handler(signum: int, _frame: Any) -> None:
        nonlocal interrupted_by
        interrupted_by = signum
        raise HarnessInterrupted(f"trapped signal {signal.Signals(signum).name}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, trap_handler)
    timeouts = accepted["runtime"]["timeouts_s"]
    try:
        formal = launch_owned_process(
            "formal_loader",
            accepted["formal_loader_argv"],
            Path(accepted["artifacts"]["runner_binary"]["path"]).parent,
            accepted["runtime"]["environment"],
            token,
            run_dir,
            proc_root=proc_root,
        )
        managed.append(formal)
        rc = finish_short_process(
            formal,
            timeouts["formal_loader"],
            term_s=timeouts["term"],
            kill_s=timeouts["kill"],
            proc_root=proc_root,
        )
        loader_output = _read_logs(formal)
        if rc != 0:
            raise HarnessError(f"formal loader failed rc={rc}")
        for marker in contract["formal_loader"]["required_output_substrings"]:
            if marker not in loader_output:
                raise HarnessError(f"formal loader omitted required marker {marker!r}")
        for marker in contract["formal_loader"]["forbidden_output_substrings"]:
            if marker in loader_output:
                raise HarnessError(f"formal loader crossed backend boundary: {marker!r}")

        for role, timeout_key in (
            ("vendor_sim", "vendor_sim_ready"),
            ("planner", "planner_ready"),
            ("runner", "runner_first_tick"),
        ):
            revalidate_artifacts(contract, accepted["artifacts"])
            spec = accepted["runtime"]["commands"][role]
            argv = list(spec["argv"])
            trace_path = run_dir / "first_tick.full_state.json"
            if role == "runner":
                if trace_path.exists():
                    raise HarnessError(f"no-clobber first-tick output exists: {trace_path}")
                argv = [str(trace_path) if value == FIRST_TICK_OUTPUT_PLACEHOLDER else value for value in argv]
            process = launch_owned_process(
                role,
                argv,
                Path(spec["cwd"]),
                accepted["runtime"]["environment"],
                token,
                run_dir,
                proc_root=proc_root,
            )
            managed.append(process)
            wait_for_marker(
                process,
                accepted["runtime"]["readiness_substrings"][role],
                timeouts[timeout_key],
            )
            if role == "runner":
                first_tick_evidence = wait_for_first_tick_trace(
                    trace_path, process, timeouts["runner_first_tick"]
                )
        status = "first_tick_observed_no_publish"
    except Exception as exc:  # cleanup and ledger are required for every failure
        error = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_errors: list[str] = []
        for process in reversed(managed):
            try:
                stop_owned_process(
                    process,
                    term_s=timeouts["term"],
                    kill_s=timeouts["kill"],
                    proc_root=proc_root,
                )
            except Exception as exc:
                cleanup_errors.append(f"{process.role}: {type(exc).__name__}: {exc}")
        try:
            release_owned_lock(lock_path, token, lock_inode)
        except Exception as exc:
            cleanup_errors.append(f"lock: {type(exc).__name__}: {exc}")
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if cleanup_errors:
            status = "cleanup_identity_failure"
            joined = "; ".join(cleanup_errors)
            error = f"{error}; {joined}" if error else joined

        content = {
            "contract_id": accepted["contract_id"],
            "status": status,
            "started_run_id": run_id,
            "finished_utc": _utc_now(),
            "contract": {"path": str(contract_path), "sha256": contract_sha},
            "source_commit": accepted["source_commit"],
            "harness_sha256": accepted["harness_sha256"],
            "artifacts": accepted["artifacts"],
            "read_only_checkouts": accepted["checkouts"],
            "runtime_binding": {
                "ros_domain_id": accepted["runtime"]["ros_domain_id"],
                "transport_scope": accepted["runtime"]["transport_scope"],
                "body_command_publish_allowed": False,
                "ledger_root": accepted["runtime"]["ledger_root"],
                "lock_path": accepted["runtime"]["lock_path"],
                "conflict_locks": accepted["runtime"]["conflict_locks"],
                "environment": accepted["runtime"]["environment"],
                "timeouts_s": accepted["runtime"]["timeouts_s"],
                "readiness_substrings": accepted["runtime"]["readiness_substrings"],
                "commands": accepted["runtime"]["commands"],
            },
            "runtime_preflight": runtime_preflight,
            "ownership_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "plan_sha256": canonical_sha256(plan),
            "hardware_authorized": False,
            "body_command_publish_allowed": False,
            "interrupted_by": interrupted_by,
            "error": error,
            "first_tick_evidence": first_tick_evidence,
            "decision_policy": accepted["decision_policy"],
            "engine_gap_diagnostic_ladder": accepted["engine_gap_diagnostic_ladder"],
            "ready_state_diagnostic": accepted["ready_state_diagnostic"],
            "processes": [process.ledger_row() for process in managed],
            "actions": {
                "broad_process_search_or_signal": False,
                "real_robot_authorized": False,
                "result_scope": "vendor_sim_first_tick_only_not_gate3_behavior",
            },
        }
        document = {
            "schema_version": 1,
            "artifact_kind": "gate3_first_tick_no_publish_ledger",
            "content_sha256": canonical_sha256(content),
            "content": content,
        }
        ledger_path = run_dir / "ledger.json"
        atomic_json_no_clobber(ledger_path, document)
    if status != "first_tick_observed_no_publish":
        raise HarnessError(f"Gate3 first-tick harness failed; preserved ledger {ledger_path}: {error}")
    return ledger_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--arm-vendor-sim-no-publish")
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args(argv)
    if not args.contract.is_absolute():
        raise HarnessError("--contract must be an absolute path")
    contract_path = args.contract.resolve()
    if not HEX64.fullmatch(args.expected_contract_sha256):
        raise HarnessError("--expected-contract-sha256 must be lowercase SHA-256")
    if not contract_path.is_file() or sha256_file(contract_path) != args.expected_contract_sha256:
        raise HarnessError("contract bytes do not match --expected-contract-sha256")
    contract = load_json(contract_path)
    plan = build_plan(contract_path, args.expected_contract_sha256, contract)
    if args.mode == "plan":
        if args.arm_vendor_sim_no_publish is not None:
            raise HarnessError("arming flag is invalid in default plan mode")
        if args.plan_output is not None:
            if not args.plan_output.is_absolute():
                raise HarnessError("--plan-output must be absolute")
            output = args.plan_output.resolve()
            atomic_json_no_clobber(output, plan)
            print(f"[gate3-first-tick] validated plan written no-clobber: {output}")
        else:
            print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
        print("[gate3-first-tick] PLAN ONLY: no sim/Kit/transport/runner/process/signal")
        return 0
    if args.plan_output is not None:
        raise HarnessError("--plan-output is valid only in plan mode")
    authorize_run(args.arm_vendor_sim_no_publish, plan)
    ledger = run_harness(contract_path, args.expected_contract_sha256, contract, plan)
    print(f"[gate3-first-tick] first tick observed under no-publish; ledger={ledger}")
    print("[gate3-first-tick] this is not Gate3 behavior and never authorizes a robot")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"[gate3-first-tick][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
