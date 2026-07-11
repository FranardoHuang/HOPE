#!/usr/bin/env python3
"""Plan-only, content-bound source gate for a future vendor Gate3 first tick.

This program can validate a static contract and optionally write one atomic no-clobber plan.
It has no runtime mode, process supervisor, signal path, simulator launch, or robot authority.
The only child commands it can start are read-only Git queries with optional locks disabled.
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
import secrets
import stat
import subprocess
import sys
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
    "vendor_sim_binary",
    "planner_binary",
    "runner_binary",
    "kit_binary",
}
COMPONENT_ORDER = ("vendor_sim", "planner", "runner")
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
ENV_DIRECTORY_KEYS = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "AMENT_PREFIX_PATH")
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
    "position",
    "velocity",
    "normal",
    "rho",
    "time_to_strike",
    "swing_type",
    "valid",
)
FIRST_TICK_QPOS_LAYOUT = "free_xyz_quat_wxyz_then_31_joint_names"
FIRST_TICK_QVEL_LAYOUT = "free_linear_xyz_angular_xyz_then_31_joint_names"
FIRST_TICK_POSE_QUATERNION_ORDER = "wxyz"
FIRST_TICK_TARGET_FRAME = "world_table"
FIRST_TICK_OBS_CONTRACT = "deploy_parity_face179"
FORMAL_LOADER_CONTRACT = {
    "required_before_any_future_runtime": True,
    "execution_authorized_in_this_gate": False,
    "required_output_substrings": [
        "[pp PREFLIGHT] accepted",
        "backend_not_initialized=true",
        "obs_dim=179",
    ],
    "forbidden_output_substrings": [
        "backend cfg",
        "A3AimrtBackend initialised",
        "backend started",
    ],
    "requires_no_publish": True,
}
FIRST_TICK_EVIDENCE_CONTRACT = {
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
}
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
RUNTIME_BLOCKERS = {
    "runner_first_tick_json": {
        "status": "blocked",
        "required": "native full qpos38/qvel37/base7/racket7/target/obs179 output",
        "evidence": None,
    },
    "exact_process_supervision": {
        "status": "blocked",
        "required": "pidfd plus cgroup or reviewed supervisor startup handshake",
        "evidence": None,
    },
    "complete_artifact_closure": {
        "status": "blocked",
        "environment_directory_manifests": {
            "PATH": None,
            "LD_LIBRARY_PATH": None,
            "PYTHONPATH": None,
            "AMENT_PREFIX_PATH": None,
        },
        "aimrt_shared_objects": None,
        "transitive_shared_objects": None,
        "plugins": None,
    },
    "vendor_config_semantic_mjcf_binding": {
        "status": "blocked",
        "required": "parser-backed resolved config-to-MJCF proof",
        "evidence": None,
    },
    "atomic_runtime_ledger_and_lock": {
        "status": "blocked",
        "required": "reviewed no-replace ledger and exact-owned runtime lock protocol",
        "evidence": None,
    },
}


class HarnessError(RuntimeError):
    """The static contract violated a fail-closed plan invariant."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise HarnessError(f"non-finite JSON constant {value!r}")


def _parse_json_bytes(payload: bytes, owner: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot parse strict JSON {owner}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"JSON root must be an object: {owner}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise HarnessError(f"cannot read strict JSON {path}: {exc}") from exc
    return _parse_json_bytes(payload, str(path))


def load_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise HarnessError(f"cannot read SHA-bound JSON {path}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if identity_before != identity_after:
        raise HarnessError("contract changed while its bound bytes were read")
    observed_sha = hashlib.sha256(payload).hexdigest()
    if observed_sha != expected_sha256:
        raise HarnessError("contract bytes do not match --expected-contract-sha256")
    return _parse_json_bytes(payload, str(path))


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
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _assert_no_symlink_components(path: Path, owner: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except OSError as exc:
            raise HarnessError(f"cannot inspect {owner} component {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise HarnessError(f"{owner} has a symlink component: {current}")


def canonical_existing_path(raw: Any, owner: str, *, kind: str) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\n" in raw:
        raise HarnessError(f"{owner} must be a non-empty path without NUL/newline")
    if not os.path.isabs(raw):
        raise HarnessError(f"{owner} must be absolute: {raw!r}")
    if os.path.normpath(raw) != raw:
        raise HarnessError(f"{owner} must be a canonical absolute spelling: {raw!r}")
    path = Path(raw)
    _assert_no_symlink_components(path, owner)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise HarnessError(f"{owner} does not resolve to an existing path: {path}: {exc}") from exc
    if resolved != path:
        raise HarnessError(f"{owner} must equal its resolved real path: {path} -> {resolved}")
    if kind == "file" and not path.is_file():
        raise HarnessError(f"{owner} must be a regular file: {path}")
    if kind == "directory" and not path.is_dir():
        raise HarnessError(f"{owner} must be a directory: {path}")
    return resolved


def canonical_output_path(raw: Any, owner: str) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\n" in raw:
        raise HarnessError(f"{owner} must be a non-empty path without NUL/newline")
    if not os.path.isabs(raw) or os.path.normpath(raw) != raw:
        raise HarnessError(f"{owner} must be a canonical absolute path")
    path = Path(raw)
    parent = canonical_existing_path(str(path.parent), f"{owner}.parent", kind="directory")
    canonical = parent / path.name
    if canonical != path:
        raise HarnessError(f"{owner} must be directly represented by its real parent")
    if path.is_symlink():
        raise HarnessError(f"{owner} must not be a symlink")
    return path


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise HarnessError(f"cannot fsync plan-output directory {path}: {exc}") from exc


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    output = canonical_output_path(str(path), "plan_output")
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    random_suffix = f"{secrets.randbits(96):024x}"
    temp = output.parent / f".{output.name}.{os.getpid()}.{random_suffix}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    linked = False
    try:
        fd = os.open(temp, flags, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise HarnessError("short write while creating plan-output temporary file")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temp, output, follow_symlinks=False)
            linked = True
        except FileExistsError as exc:
            raise HarnessError(f"no-clobber plan output already exists: {output}") from exc
        except OSError as exc:
            raise HarnessError(f"cannot atomically link plan output {output}: {exc}") from exc
        _fsync_directory(output.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        if linked:
            _fsync_directory(output.parent)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _git_environment() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(repo),
                *args,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=15,
            env=_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HarnessError(f"read-only git helper failed for {repo}: {exc}") from exc
    if completed.returncode != 0:
        raise HarnessError(
            f"read-only git {' '.join(args)} failed for {repo}: {completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def current_source_commit() -> str:
    root = canonical_existing_path(
        str(Path(__file__).resolve().parents[1]), "source_checkout", kind="directory"
    )
    top = canonical_existing_path(
        _git(root, "rev-parse", "--show-toplevel"),
        "source_checkout.git_toplevel",
        kind="directory",
    )
    if top != root:
        raise HarnessError(f"source checkout path is not its Git top-level: {root} != {top}")
    return _git(root, "rev-parse", "HEAD")


def validate_read_only_checkout(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(spec, {"path", "commit"}, f"read_only_checkouts.{name}")
    path = canonical_existing_path(
        spec["path"], f"read_only_checkouts.{name}.path", kind="directory"
    )
    commit = spec["commit"]
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise HarnessError(f"read_only_checkouts.{name}.commit must be lowercase SHA-1")
    top = canonical_existing_path(
        _git(path, "rev-parse", "--show-toplevel"),
        f"read_only_checkouts.{name}.git_toplevel",
        kind="directory",
    )
    if path != top:
        raise HarnessError(
            f"read_only_checkouts.{name}.path must equal git rev-parse --show-toplevel"
        )
    head = _git(path, "rev-parse", "HEAD")
    status_text = _git(path, "status", "--porcelain=v1", "--untracked-files=all")
    if head != commit or status_text:
        raise HarnessError(
            f"read-only {name} checkout changed: head={head} expected={commit} "
            f"dirty={bool(status_text)}"
        )
    return {"path": str(path), "commit": head, "clean": True, "git_toplevel": str(top)}


def _artifact_stat(path: Path) -> dict[str, Any]:
    observed = path.stat()
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "mode": observed.st_mode,
    }


def validate_artifact(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(spec, {"path", "sha256", "executable"}, f"artifacts.{name}")
    path = canonical_existing_path(spec["path"], f"artifacts.{name}.path", kind="file")
    expected_sha = spec["sha256"]
    if not isinstance(expected_sha, str) or not HEX64.fullmatch(expected_sha):
        raise HarnessError(f"artifacts.{name}.sha256 must be lowercase SHA-256")
    executable = spec["executable"]
    if not isinstance(executable, bool) or executable != (name in EXECUTABLE_ARTIFACTS):
        raise HarnessError(f"artifacts.{name}.executable changed from the role contract")
    before = _artifact_stat(path)
    observed_sha = sha256_file(path)
    after = _artifact_stat(path)
    if before != after:
        raise HarnessError(f"artifacts.{name} changed while it was hashed")
    if observed_sha != expected_sha:
        raise HarnessError(
            f"artifacts.{name} SHA mismatch: observed={observed_sha} expected={expected_sha}"
        )
    if executable and not os.access(path, os.X_OK):
        raise HarnessError(f"artifacts.{name} is not executable: {path}")
    return {
        "path": str(path),
        "resolved_path": str(path),
        "sha256": observed_sha,
        "executable": executable,
        "stat": after,
    }


def _validate_env_path_list(name: str, value: str, *, allow_empty: bool) -> list[str]:
    if not value:
        if allow_empty:
            return []
        raise HarnessError(f"runtime_proposal.environment.{name} must not be empty")
    parts = value.split(":")
    if any(not part for part in parts):
        raise HarnessError(f"runtime_proposal.environment.{name} contains an empty path element")
    return [
        str(
            canonical_existing_path(
                part,
                f"runtime_proposal.environment.{name}[{index}]",
                kind="directory",
            )
        )
        for index, part in enumerate(parts)
    ]


def validate_environment(
    env: Mapping[str, Any], ros_domain_id: int
) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact_keys(env, REQUIRED_ENV_KEYS, "runtime_proposal.environment")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()):
        raise HarnessError("runtime_proposal.environment must contain only string keys and values")
    if env["ROS_DOMAIN_ID"] != str(ros_domain_id):
        raise HarnessError("ROS_DOMAIN_ID disagrees with runtime_proposal.ros_domain_id")
    if env["ROS_LOCALHOST_ONLY"] != "1":
        raise HarnessError("ROS_LOCALHOST_ONLY must be 1 in the static proposal")
    if env["A3_SOURCE_ROBOT_ENV"] != "0" or env["A3_HARDWARE_ALLOWED"] != "0":
        raise HarnessError("robot environment/hardware authority must both remain disabled")
    if env["A3_TRANSPORT"] != "iceoryx":
        raise HarnessError("A3_TRANSPORT must name the proposed vendor-sim transport")
    if not env["RMW_IMPLEMENTATION"].startswith("rmw_"):
        raise HarnessError("RMW_IMPLEMENTATION must be explicit")
    if env["MUJOCO_GL"] != "egl":
        raise HarnessError("MUJOCO_GL must be egl; interactive viewer/Kit launch is out of scope")
    directories = {
        "PATH": _validate_env_path_list("PATH", env["PATH"], allow_empty=False),
        "LD_LIBRARY_PATH": _validate_env_path_list(
            "LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"], allow_empty=True
        ),
        "PYTHONPATH": _validate_env_path_list("PYTHONPATH", env["PYTHONPATH"], allow_empty=True),
        "AMENT_PREFIX_PATH": _validate_env_path_list(
            "AMENT_PREFIX_PATH", env["AMENT_PREFIX_PATH"], allow_empty=True
        ),
    }
    canonical_existing_path(env["HOME"], "runtime_proposal.environment.HOME", kind="directory")
    if not env["LANG"]:
        raise HarnessError("runtime_proposal.environment.LANG must be explicit")
    return dict(env), directories


def _validate_argv(argv: Any, owner: str) -> list[str]:
    if not isinstance(argv, list) or not argv:
        raise HarnessError(f"{owner}.argv must be a non-empty array")
    if any(
        not isinstance(item, str) or not item or "\x00" in item or "\n" in item
        for item in argv
    ):
        raise HarnessError(f"{owner}.argv entries must be non-empty strings without NUL/newline")
    return list(argv)


def _reject_unbound_or_relative_payloads(
    argv: Sequence[str], bound_paths: set[str], owner: str
) -> None:
    allowed_nonpath_values = {"passive", FIRST_TICK_OUTPUT_PLACEHOLDER}
    for token in argv:
        if token.startswith("-"):
            if "=" in token:
                raise HarnessError(f"{owner} forbids --flag=value syntax: {token!r}")
            continue
        if token in bound_paths or token in allowed_nonpath_values:
            continue
        if os.path.isabs(token):
            raise HarnessError(f"{owner} contains an unbound absolute path token: {token!r}")
        raise HarnessError(f"{owner} contains a relative/unclassified payload token: {token!r}")


def validate_commands(
    commands: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    checkouts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    exact_keys(commands, set(COMPONENT_ORDER), "runtime_proposal.commands")
    bound_paths = {artifact["path"] for artifact in artifacts.values()}
    expected_argv = {
        "vendor_sim": [
            artifacts["vendor_sim_binary"]["path"],
            "--config",
            artifacts["vendor_sim_config"]["path"],
        ],
        "planner": [
            artifacts["planner_binary"]["path"],
            "--config",
            artifacts["planner_config"]["path"],
        ],
        "runner": [
            artifacts["runner_binary"]["path"],
            "--runtime-cfg",
            artifacts["runner_runtime_config"]["path"],
            "--model-path",
            artifacts["runner_model"]["path"],
            "--planner",
            "--no-publish",
            "--start",
            "passive",
            FIRST_TICK_OUTPUT_FLAG,
            FIRST_TICK_OUTPUT_PLACEHOLDER,
        ],
    }
    result: dict[str, dict[str, Any]] = {}
    for role in COMPONENT_ORDER:
        spec = commands[role]
        if not isinstance(spec, dict):
            raise HarnessError(f"runtime_proposal.commands.{role} must be an object")
        exact_keys(spec, {"argv", "cwd"}, f"runtime_proposal.commands.{role}")
        argv = _validate_argv(spec["argv"], f"runtime_proposal.commands.{role}")
        _reject_unbound_or_relative_payloads(
            argv, bound_paths, f"runtime_proposal.commands.{role}.argv"
        )
        if argv != expected_argv[role]:
            raise HarnessError(
                f"runtime_proposal.commands.{role}.argv differs from the fixed static proposal"
            )
        cwd = canonical_existing_path(
            spec["cwd"], f"runtime_proposal.commands.{role}.cwd", kind="directory"
        )
        for checkout in checkouts.values():
            if is_under(cwd, Path(checkout["path"])):
                raise HarnessError(f"{role} proposed cwd must be outside train/eval checkouts")
        result[role] = {"argv": argv, "cwd": str(cwd)}
    return result


def build_formal_loader_argv(artifacts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        artifacts["runner_binary"]["path"],
        "--runtime-cfg",
        artifacts["runner_runtime_config"]["path"],
        "--model-path",
        artifacts["runner_model"]["path"],
        "--planner",
        "--no-publish",
        "--model-preflight-only",
    ]


def validate_contract(data: dict[str, Any]) -> dict[str, Any]:
    exact_keys(
        data,
        {
            "schema_version",
            "contract_id",
            "created_utc",
            "status",
            "scope",
            "source_commit",
            "harness_sha256",
            "hardware_authorized",
            "artifacts",
            "read_only_checkouts",
            "formal_loader",
            "first_tick_evidence",
            "runtime_proposal",
            "activation",
            "runtime_blockers",
            "decision_policy",
            "engine_gap_diagnostic_ladder",
            "ready_state_diagnostic",
        },
        "Gate3 first-tick static contract",
    )
    if (
        data["schema_version"] != 2
        or data["status"] != "preregistered_plan_only_not_run"
        or data["scope"] != "vendor_gate3_first_tick_static_plan_only"
        or data["hardware_authorized"] is not False
    ):
        raise HarnessError(
            "contract must remain schema-2, plan-only, not-run, and hardware-forbidden"
        )
    if not isinstance(data["contract_id"], str) or not data["contract_id"]:
        raise HarnessError("contract_id must be non-empty")
    if not isinstance(data["created_utc"], str) or not data["created_utc"].endswith("Z"):
        raise HarnessError("created_utc must be an explicit UTC string")
    if not isinstance(data["source_commit"], str) or not HEX40.fullmatch(data["source_commit"]):
        raise HarnessError("source_commit must be lowercase SHA-1")
    observed_source_commit = current_source_commit()
    if data["source_commit"] != observed_source_commit:
        raise HarnessError(
            "source_commit changed: "
            f"contract={data['source_commit']} checkout={observed_source_commit}"
        )
    if not isinstance(data["harness_sha256"], str) or not HEX64.fullmatch(data["harness_sha256"]):
        raise HarnessError("harness_sha256 must be lowercase SHA-256")
    if data["harness_sha256"] != sha256_file(Path(__file__).resolve()):
        raise HarnessError("contract does not bind these plan-only harness source bytes")

    artifacts_raw = data["artifacts"]
    if not isinstance(artifacts_raw, dict):
        raise HarnessError("artifacts must be an object")
    exact_keys(artifacts_raw, set(ARTIFACT_KEYS), "artifacts")
    artifacts = {name: validate_artifact(name, artifacts_raw[name]) for name in ARTIFACT_KEYS}

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
                raise HarnessError("proposed runtime artifacts must be external immutable copies")

    if data["formal_loader"] != FORMAL_LOADER_CONTRACT:
        raise HarnessError("formal-loader proposal changed or claims execution")
    if data["first_tick_evidence"] != FIRST_TICK_EVIDENCE_CONTRACT:
        raise HarnessError("future first-tick evidence contract changed")

    proposal = data["runtime_proposal"]
    if not isinstance(proposal, dict):
        raise HarnessError("runtime_proposal must be an object")
    exact_keys(
        proposal,
        {
            "ros_domain_id",
            "environment",
            "commands",
            "transport_scope",
            "body_command_publish_allowed",
        },
        "runtime_proposal",
    )
    domain = proposal["ros_domain_id"]
    if isinstance(domain, bool) or not isinstance(domain, int) or not 0 <= domain <= 232:
        raise HarnessError("runtime_proposal.ros_domain_id must be an integer in [0,232]")
    if proposal["transport_scope"] != "vendor_sim_only_no_hardware":
        raise HarnessError("runtime_proposal.transport_scope must stay vendor-sim-only")
    if proposal["body_command_publish_allowed"] is not False:
        raise HarnessError("body-command publishing must remain forbidden")
    environment, environment_directories = validate_environment(proposal["environment"], domain)
    commands = validate_commands(proposal["commands"], artifacts, checkouts)

    if data["activation"] != {
        "mode": "plan_only",
        "runtime_execution_authorized": False,
        "real_robot_authorized": False,
    }:
        raise HarnessError("activation must remain plan-only with no runtime or robot authority")
    if data["runtime_blockers"] != RUNTIME_BLOCKERS:
        raise HarnessError("runtime blockers were edited, filled, or silently removed")
    if data["decision_policy"] != DECISION_POLICY:
        raise HarnessError("vendor Gate3 authority or Isaac diagnostic role changed")
    if data["engine_gap_diagnostic_ladder"] != ENGINE_GAP_DIAGNOSTIC_LADDER:
        raise HarnessError("engine-gap ladder changed or contains an unearned inference")
    if data["ready_state_diagnostic"] != READY_STATE_DIAGNOSTIC:
        raise HarnessError("ready-state diagnostic changed or was promoted")

    return {
        "contract_id": data["contract_id"],
        "source_commit": data["source_commit"],
        "harness_sha256": data["harness_sha256"],
        "artifacts": artifacts,
        "read_only_checkouts": checkouts,
        "formal_loader": {
            "contract": copy.deepcopy(FORMAL_LOADER_CONTRACT),
            "proposed_argv": build_formal_loader_argv(artifacts),
        },
        "first_tick_evidence": copy.deepcopy(FIRST_TICK_EVIDENCE_CONTRACT),
        "runtime_proposal": {
            "ros_domain_id": domain,
            "environment": environment,
            "environment_directories": environment_directories,
            "commands": commands,
            "transport_scope": proposal["transport_scope"],
            "body_command_publish_allowed": False,
        },
        "runtime_blockers": copy.deepcopy(RUNTIME_BLOCKERS),
        "decision_policy": copy.deepcopy(DECISION_POLICY),
        "engine_gap_diagnostic_ladder": copy.deepcopy(ENGINE_GAP_DIAGNOSTIC_LADDER),
        "ready_state_diagnostic": copy.deepcopy(READY_STATE_DIAGNOSTIC),
    }


def build_plan(
    contract_path: Path, contract_sha: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    accepted = validate_contract(dict(contract))
    content = {
        "status": "validated_static_plan_runtime_not_run",
        "contract": {"path": str(contract_path), "sha256": contract_sha},
        "source_commit": accepted["source_commit"],
        "harness_sha256": accepted["harness_sha256"],
        "artifacts": accepted["artifacts"],
        "read_only_checkouts": accepted["read_only_checkouts"],
        "formal_loader": accepted["formal_loader"],
        "first_tick_evidence": accepted["first_tick_evidence"],
        "runtime_proposal": accepted["runtime_proposal"],
        "runtime": {
            "status": "not_run",
            "execution_authorized": False,
            "components_started": [],
            "signals_sent": [],
            "runtime_lock_acquired": False,
            "behavior_result": None,
            "blockers": accepted["runtime_blockers"],
        },
        "decision_policy": accepted["decision_policy"],
        "engine_gap_diagnostic_ladder": accepted["engine_gap_diagnostic_ladder"],
        "ready_state_diagnostic": accepted["ready_state_diagnostic"],
        "actions": {
            "read_only_git_helpers_started": True,
            "git_optional_locks": False,
            "simulator_started": False,
            "kit_started": False,
            "transport_started": False,
            "planner_started": False,
            "runner_started": False,
            "signals_sent": [],
            "real_robot_authorized": False,
        },
    }
    return {
        "schema_version": 2,
        "artifact_kind": "gate3_first_tick_static_plan_ledger",
        "content_sha256": canonical_sha256(content),
        "content": content,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args(argv)

    contract_path = canonical_existing_path(str(args.contract), "contract", kind="file")
    if not HEX64.fullmatch(args.expected_contract_sha256):
        raise HarnessError("--expected-contract-sha256 must be lowercase SHA-256")
    contract = load_bound_json(contract_path, args.expected_contract_sha256)
    plan = build_plan(contract_path, args.expected_contract_sha256, contract)
    if args.plan_output is not None:
        output = canonical_output_path(str(args.plan_output), "plan_output")
        atomic_json_no_clobber(output, plan)
        print(f"[gate3-first-tick] static plan written no-clobber: {output}")
    else:
        print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
    print(
        "[gate3-first-tick] PLAN ONLY: read-only Git helpers ran with "
        "GIT_OPTIONAL_LOCKS=0; no sim/Kit/transport/planner/runner/signal/robot"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"[gate3-first-tick][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
