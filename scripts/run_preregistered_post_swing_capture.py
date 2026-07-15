#!/usr/bin/env python3
"""Fail-closed one-shot launcher for a preregistered post-swing capture.

This program runs *on the simulation host*.  It never opens SSH, sends a
signal, starts a trainer, or retries a spent namespace.  ``plan`` is read-only;
``launch`` performs one Hydra compose before creating the capture directory,
then starts exactly one inference process in a new numeric process group.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


class CaptureContractError(RuntimeError):
    """The frozen plan or current runtime does not authorize a launch."""


class HydraComposeError(CaptureContractError):
    """A read-only Hydra compose failed while retaining its bounded output."""

    def __init__(
        self,
        message: str,
        *,
        output: bytes,
        elapsed_ms: int,
        returncode: int | None,
    ) -> None:
        super().__init__(message)
        self.output = output
        self.elapsed_ms = elapsed_ms
        self.returncode = returncode


ARTIFACT_ROOT = Path("/workspace/codexschema/phase1_post_swing_teacher_20260715")
CAPTURE_PARENT = ARTIFACT_ROOT / "capture"
LAUNCH_PARENT = ARTIFACT_ROOT / "launch"
GPU_LEASE_PATH = Path("/tmp/hope_lean_queue_gpu2.lock")
ISAAC_PYTHON = Path("/workspace/hope_isaac_venv/bin/python")
MACHINE_ID_PATH = Path("/etc/machine-id")
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
RETRY_AUTHORIZATION_RELATIVE = Path(
    "configs/phase1_post_swing_teacher_capture_v3_attestor_retry_authorization_20260715.json"
)
RETRY_AUTHORIZATION_KIND = "hope_post_swing_teacher_attestor_retry_authorization"
UINT32_MAX = 0xFFFFFFFF
NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_REMOVE_KEYS = frozenset(
    {
        "logger",
        "video",
        "checkpoint_path",
        "checkpoint_tolerant",
        "checkpoint_allow_missing_contract",
        "checkpoint_allow_contract_mismatch",
        "max_iterations",
        "algo.runner.save_interval",
        "run_name",
        "training_queue_claim_path",
        "training_run_binding_path",
        "training_launch_claim_sha256",
    }
)
EXPECTED_ADD_KEYS = frozenset(
    {
        "checkpoint",
        "task.motion.post_swing_capture_output_dir",
        "task.motion.post_swing_capture_target_count",
        "post_swing_capture_max_steps",
    }
)
RUNTIME_TREE_LABELS = frozenset(
    {
        "capture_pythonpath",
        "isaaclab",
        "isaaclab_tasks",
        "isaaclab_assets",
        "isaaclab_rl",
    }
)
SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "TERM",
        "TMPDIR",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_regular_bytes(path, str(path)))


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)


def _canonical_absolute_path(value: Any, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value or "\n" in value:
        raise CaptureContractError(f"{label} must be one non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise CaptureContractError(f"{label} must be absolute and contain no dot components")
    normalized = Path(os.path.normpath(value))
    if str(normalized) != value.rstrip("/"):
        raise CaptureContractError(f"{label} must already be normalized")
    return normalized


def _canonical_relative_path(value: Any, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value or "\n" in value:
        raise CaptureContractError(f"{label} must be one non-empty relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CaptureContractError(f"{label} must be normalized, relative, and contain no dot components")
    if str(path) != value:
        raise CaptureContractError(f"{label} must already be normalized")
    return path


@contextmanager
def _open_real_directory(path: Path):
    """Open an existing absolute directory without following any path-component link."""

    canonical = _canonical_absolute_path(str(path), "directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        try:
            for part in canonical.parts[1:]:
                next_descriptor = os.open(
                    part,
                    flags | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError as exc:
            raise CaptureContractError(
                f"directory path is missing, linked, or invalid: {path}"
            ) from exc
        yield descriptor
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _read_regular_bytes(path: Path, label: str) -> bytes:
    canonical = _canonical_absolute_path(str(path), label)
    with _open_real_directory(canonical.parent) as parent_fd:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(canonical.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise CaptureContractError(f"{label} must be a regular non-symlink file: {path}") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise CaptureContractError(f"{label} must be a regular file: {path}")
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(descriptor)
            if _stat_signature(before) != _stat_signature(after):
                raise CaptureContractError(f"{label} changed while reading: {path}")
            path_after = os.stat(canonical.name, dir_fd=parent_fd, follow_symlinks=False)
            if _stat_signature(after) != _stat_signature(path_after):
                raise CaptureContractError(f"{label} path identity changed while reading: {path}")
            return b"".join(chunks)
        finally:
            os.close(descriptor)


def _strict_json_loads(raw: bytes, label: str) -> Mapping[str, Any]:
    def reject_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CaptureContractError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise CaptureContractError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureContractError(f"{label} is not strict JSON: {exc}") from exc
    return _require_mapping(value, label)


def _read_regular_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    raw = _read_regular_bytes(path, label)
    return _strict_json_loads(raw, label), raw


def _require_exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        raise CaptureContractError(
            f"{label} keys differ: missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CaptureContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaptureContractError(f"{label} must be an object")
    return value


def _require_plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CaptureContractError(f"{label} must be an integer >= {minimum}")
    return value


def _normal_key(argument: str) -> str:
    if "=" not in argument:
        raise CaptureContractError(f"non-Hydra argument in frozen recipe: {argument}")
    key = argument.split("=", 1)[0].lstrip("+")
    if not key:
        raise CaptureContractError(f"empty Hydra key in frozen recipe: {argument}")
    return key


def _load_json_document(path: Path, expected_file_sha256: str, label: str) -> Mapping[str, Any]:
    expected = _require_sha256(expected_file_sha256, f"{label}.file_sha256")
    raw = _read_regular_bytes(path, label)
    actual = _sha256_bytes(raw)
    if actual != expected:
        raise CaptureContractError(f"{label} file SHA mismatch: {actual} != {expected}")
    return _strict_json_loads(raw, label)


def _verify_content_document(row: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    path = Path(str(row.get("path", "")))
    document = _load_json_document(path, str(row.get("file_sha256", "")), label)
    content = _require_mapping(document.get("content"), f"{label}.content")
    declared = _require_sha256(row.get("content_sha256"), f"{label}.content_sha256")
    embedded = _require_sha256(document.get("content_sha256"), f"{label}.embedded_content_sha256")
    actual = _sha256_bytes(_canonical_bytes(content))
    if actual != declared or embedded != declared:
        raise CaptureContractError(
            f"{label} content binding mismatch: actual={actual} embedded={embedded} expected={declared}"
        )
    return document


def _inventory(root: Path, *, skip_git: bool = False) -> dict[str, Any]:
    if os.path.lexists(root) is False:
        raise CaptureContractError(f"ignored runtime asset is missing: {root}")
    with _open_real_directory(root):
        pass
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise CaptureContractError(f"ignored runtime asset is not a real directory: {root}")
    rows: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if skip_git and current_path == root:
            dirnames[:] = [name for name in dirnames if name != ".git"]
            filenames = [name for name in filenames if name != ".git"]
        for name in sorted(dirnames):
            path = current_path / name
            result = path.lstat()
            if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
                raise CaptureContractError(f"asset tree contains invalid directory entry: {path}")
        for name in sorted(filenames):
            path = current_path / name
            result = path.lstat()
            if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
                raise CaptureContractError(f"asset tree contains invalid file entry: {path}")
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": result.st_size,
                    "sha256": _sha256_file(path),
                }
            )
    rows.sort(key=lambda row: row["relative_path"])
    return {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "tree_content_sha256": _sha256_bytes(_canonical_bytes({"files": rows})),
    }


def _git_output(executable: Path, checkout: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            [str(executable), "-C", str(checkout), *arguments],
            text=True,
            stderr=subprocess.STDOUT,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise CaptureContractError(f"git check failed for {checkout}: {exc.output}") from exc


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != 2:
        raise CaptureContractError("only post-swing capture plan schema_version=2 is supported")
    plan_id = plan.get("plan_id")
    if type(plan_id) is not str or not NAMESPACE_RE.fullmatch(plan_id):
        raise CaptureContractError("plan_id must be a lowercase filesystem-safe namespace")
    if plan.get("status") != "preregistered_capture_not_started":
        raise CaptureContractError("plan is not in preregistered_capture_not_started state")
    if plan.get("simulation_only") is not True:
        raise CaptureContractError("capture must be simulation_only")
    contract = _require_mapping(plan.get("capture_contract"), "capture_contract")
    _require_exact_keys(
        contract,
        {
            "pod", "gpu", "gpu_uuid", "cuda_visible_devices", "runtime_device",
            "num_envs", "target_count", "max_inference_steps", "seed",
            "wrap_teleport", "post_swing_start_prob",
            "root_linear_velocity_limit_mps", "root_angular_velocity_limit_radps",
            "namespace_id", "output_directory", "launch_root",
            "output_must_be_absent_before_one_shot", "capture_is_inference_only",
            "ppo_updates", "natural_wrap_only", "timeout_or_failure_reset_states_forbidden",
            "launch_handoff",
        },
        "capture_contract",
    )
    if contract.get("pod") != "pod2" or contract.get("gpu") != 2:
        raise CaptureContractError("this plan must remain Pod2 physical GPU2 only")
    gpu_uuid = contract.get("gpu_uuid")
    if type(gpu_uuid) is not str or not gpu_uuid.startswith("GPU-"):
        raise CaptureContractError("capture_contract.gpu_uuid must be an exact NVIDIA GPU UUID")
    if contract.get("cuda_visible_devices") != gpu_uuid or contract.get("runtime_device") != "cuda:0":
        raise CaptureContractError("CUDA remapping contract changed")
    for key in ("num_envs", "target_count", "max_inference_steps", "seed"):
        _require_plain_int(contract.get(key), f"capture_contract.{key}", minimum=1 if key != "seed" else 0)
    if contract["seed"] > UINT32_MAX:
        raise CaptureContractError("capture_contract.seed must fit the play uint32 contract")
    if contract.get("num_envs") != 4096 or contract.get("target_count") != 4096:
        raise CaptureContractError("first formal capture must remain 4096 environments/states")
    if contract.get("max_inference_steps") != 20000:
        raise CaptureContractError("first formal capture must retain the 20000-step ceiling")
    if type(contract.get("post_swing_start_prob")) not in (int, float) or not math.isclose(
        float(contract["post_swing_start_prob"]), 0.25, rel_tol=0.0, abs_tol=0.0
    ):
        raise CaptureContractError("post_swing_start_prob must remain exactly 0.25")
    limits = (
        ("root_linear_velocity_limit_mps", 2.0),
        ("root_angular_velocity_limit_radps", 4.0),
    )
    for key, expected in limits:
        value = contract.get(key)
        if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) != expected:
            raise CaptureContractError(f"capture_contract.{key} must remain {expected}")
    if contract.get("capture_is_inference_only") is not True or contract.get("ppo_updates") != 0:
        raise CaptureContractError("capture may not perform PPO updates")
    if contract.get("natural_wrap_only") is not True or contract.get("wrap_teleport") is not False:
        raise CaptureContractError("capture must remain natural-wrap only")
    if contract.get("timeout_or_failure_reset_states_forbidden") is not True:
        raise CaptureContractError("timeout/failure reset states must remain forbidden")
    if contract.get("launch_handoff") != "execve_same_pid_v1":
        raise CaptureContractError("capture launch must use the no-child execve handoff")
    if contract.get("output_must_be_absent_before_one_shot") is not True:
        raise CaptureContractError("capture output must be no-clobber")
    namespace_id = contract.get("namespace_id")
    if namespace_id != plan_id:
        raise CaptureContractError("capture namespace_id must equal plan_id")
    output = _canonical_absolute_path(contract.get("output_directory"), "output_directory")
    launch_root = _canonical_absolute_path(contract.get("launch_root"), "launch_root")
    if output != CAPTURE_PARENT / plan_id or launch_root != LAUNCH_PARENT / plan_id:
        raise CaptureContractError("capture and launch must be fixed direct namespaces with the plan_id leaf")

    source = _require_mapping(plan.get("capture_source"), "capture_source")
    _require_exact_keys(
        source,
        {"checkout", "commit", "clean_required", "files", "ignored_runtime_asset", "full_tree"},
        "capture_source",
    )
    _canonical_absolute_path(source.get("checkout"), "capture_source.checkout")
    if type(source.get("commit")) is not str or not COMMIT_RE.fullmatch(source["commit"]):
        raise CaptureContractError("capture_source.commit must be 40 lowercase hex characters")
    if source.get("clean_required") is not True:
        raise CaptureContractError("capture source must require a clean checkout")
    files = _require_mapping(source.get("files"), "capture_source.files")
    for label, raw_row in files.items():
        row = _require_mapping(raw_row, f"capture_source.files.{label}")
        _require_exact_keys(row, {"path", "bytes", "sha256"}, f"capture_source.files.{label}")
        _canonical_relative_path(row.get("path"), f"capture_source.files.{label}.path")
        _require_plain_int(row.get("bytes"), f"capture_source.files.{label}.bytes", minimum=1)
        _require_sha256(row.get("sha256"), f"capture_source.files.{label}.sha256")
    for label in ("controller", "inference_runner", "lean_queue_runtime", "producer"):
        if label not in files:
            raise CaptureContractError(f"capture_source.files lacks required {label}")
    if files["producer"]["path"] != (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/commands.py"
    ):
        raise CaptureContractError("capture_source.files.producer is not the MotionCommand source")
    asset = _require_mapping(source.get("ignored_runtime_asset"), "ignored_runtime_asset")
    _require_exact_keys(
        asset,
        {"relative_path", "file_count", "total_file_bytes", "tree_content_sha256", "symlinks_forbidden"},
        "ignored_runtime_asset",
    )
    _canonical_relative_path(asset.get("relative_path"), "ignored_runtime_asset.relative_path")
    if asset.get("symlinks_forbidden") is not True:
        raise CaptureContractError("ignored runtime asset symlinks must be forbidden")
    for label, row in (("capture_source.full_tree", source["full_tree"]), ("ignored_runtime_asset", asset)):
        row = _require_mapping(row, label)
        _require_plain_int(row.get("file_count"), f"{label}.file_count", minimum=1)
        _require_plain_int(row.get("total_file_bytes"), f"{label}.total_file_bytes", minimum=1)
        _require_sha256(row.get("tree_content_sha256"), f"{label}.tree_content_sha256")

    runtime = _require_mapping(plan.get("runtime_environment"), "runtime_environment")
    _require_exact_keys(
        runtime,
        {
            "node", "gpu", "python", "runtime_trees", "tools",
            "environment", "compose_timeout_s",
        },
        "runtime_environment",
    )
    node = _require_mapping(runtime.get("node"), "runtime_environment.node")
    _require_exact_keys(
        node,
        {
            "hostname", "machine_id_path", "machine_id_sha256", "boot_id_path",
            "boot_id_sha256",
        },
        "runtime_environment.node",
    )
    for key in ("hostname",):
        if type(node.get(key)) is not str or not node[key] or "\n" in node[key]:
            raise CaptureContractError(f"runtime_environment.node.{key} must be one non-empty line")
    if _canonical_absolute_path(node.get("machine_id_path"), "machine_id_path") != MACHINE_ID_PATH:
        raise CaptureContractError("runtime machine-id path must be fixed /etc/machine-id")
    if _canonical_absolute_path(node.get("boot_id_path"), "boot_id_path") != BOOT_ID_PATH:
        raise CaptureContractError("runtime boot-id path must be fixed /proc/sys/kernel/random/boot_id")
    _require_sha256(node.get("machine_id_sha256"), "machine_id_sha256")
    _require_sha256(node.get("boot_id_sha256"), "boot_id_sha256")
    gpu = _require_mapping(runtime.get("gpu"), "runtime_environment.gpu")
    _require_exact_keys(gpu, {"physical_index", "uuid", "lease_path"}, "runtime_environment.gpu")
    if gpu.get("physical_index") != 2 or gpu.get("uuid") != gpu_uuid:
        raise CaptureContractError("runtime GPU identity must equal Pod2 physical GPU2 contract")
    if _canonical_absolute_path(gpu.get("lease_path"), "GPU lease path") != GPU_LEASE_PATH:
        raise CaptureContractError("GPU lease path must be the shared lean-queue GPU2 lock")
    python = _require_mapping(runtime.get("python"), "runtime_environment.python")
    _require_exact_keys(
        python,
        {"requested_path", "resolved_path", "symlink_chain", "pyvenv_cfg"},
        "runtime_environment.python",
    )
    if _canonical_absolute_path(python.get("requested_path"), "python.requested_path") != ISAAC_PYTHON:
        raise CaptureContractError("capture must use the frozen Isaac venv Python entry")
    _canonical_absolute_path(python.get("resolved_path"), "python.resolved_path")
    chain = python.get("symlink_chain")
    if not isinstance(chain, list) or len(chain) < 2:
        raise CaptureContractError("python.symlink_chain must bind every link and final regular executable")
    for index, raw_row in enumerate(chain):
        row = _require_mapping(raw_row, f"python.symlink_chain[{index}]")
        kind = row.get("kind")
        expected_keys = {"kind", "path", "target"} if kind == "symlink" else {"kind", "path", "bytes", "sha256"}
        _require_exact_keys(row, expected_keys, f"python.symlink_chain[{index}]")
        _canonical_absolute_path(row.get("path"), f"python.symlink_chain[{index}].path")
        if kind == "symlink":
            if type(row.get("target")) is not str or not row["target"] or "\x00" in row["target"]:
                raise CaptureContractError("python symlink target must be non-empty")
        elif kind == "regular":
            _require_plain_int(row.get("bytes"), "python final bytes", minimum=1)
            _require_sha256(row.get("sha256"), "python final sha256")
        else:
            raise CaptureContractError("python chain kind must be symlink or regular")
    if chain[-1].get("kind") != "regular" or chain[-1].get("path") != str(python["resolved_path"]):
        raise CaptureContractError("python chain must end at the declared resolved regular executable")
    for label, raw_row in (("pyvenv_cfg", python["pyvenv_cfg"]),):
        row = _require_mapping(raw_row, label)
        _require_exact_keys(row, {"path", "bytes", "sha256"}, label)
        _canonical_absolute_path(row.get("path"), f"{label}.path")
        _require_plain_int(row.get("bytes"), f"{label}.bytes", minimum=1)
        _require_sha256(row.get("sha256"), f"{label}.sha256")
    trees = runtime.get("runtime_trees")
    if not isinstance(trees, list) or len(trees) != len(RUNTIME_TREE_LABELS):
        raise CaptureContractError("runtime_trees must bind the exact runtime import closure")
    labels = set()
    for index, raw_row in enumerate(trees):
        row = _require_mapping(raw_row, f"runtime_trees[{index}]")
        _require_exact_keys(
            row,
            {"label", "path", "on_pythonpath", "file_count", "total_file_bytes", "tree_content_sha256"},
            f"runtime_trees[{index}]",
        )
        labels.add(row.get("label"))
        _canonical_absolute_path(row.get("path"), f"runtime_trees[{index}].path")
        if type(row.get("on_pythonpath")) is not bool:
            raise CaptureContractError("runtime tree on_pythonpath must be bool")
        _require_plain_int(row.get("file_count"), "runtime tree file_count", minimum=1)
        _require_plain_int(row.get("total_file_bytes"), "runtime tree bytes", minimum=1)
        _require_sha256(row.get("tree_content_sha256"), "runtime tree sha256")
    if labels != set(RUNTIME_TREE_LABELS):
        raise CaptureContractError("runtime tree labels differ from the required closure")
    tools = _require_mapping(runtime.get("tools"), "runtime_environment.tools")
    _require_exact_keys(tools, {"git", "nvidia_smi"}, "runtime_environment.tools")
    for label, raw_row in tools.items():
        row = _require_mapping(raw_row, f"runtime tool {label}")
        _require_exact_keys(row, {"path", "bytes", "sha256"}, f"runtime tool {label}")
        _canonical_absolute_path(row.get("path"), f"runtime tool {label} path")
        _require_plain_int(row.get("bytes"), f"runtime tool {label} bytes", minimum=1)
        _require_sha256(row.get("sha256"), f"runtime tool {label} SHA")
    environment = _require_mapping(runtime.get("environment"), "runtime_environment.environment")
    _require_exact_keys(environment, {"exact"}, "runtime_environment.environment")
    exact_environment = _require_mapping(environment.get("exact"), "runtime exact environment")
    if (
        "PATH" not in exact_environment
        or not set(exact_environment).issubset(SAFE_ENVIRONMENT_KEYS)
        or any(type(key) is not str or type(value) is not str or "\x00" in value for key, value in exact_environment.items())
    ):
        raise CaptureContractError("runtime exact environment is outside the fixed safe allowlist")
    timeout = _require_plain_int(runtime.get("compose_timeout_s"), "compose_timeout_s", minimum=1)
    if timeout > 300:
        raise CaptureContractError("compose_timeout_s must be <= 300")

    failure = _require_mapping(plan.get("failure_policy"), "failure_policy")
    _require_exact_keys(
        failure,
        {
            "preserve_partial_namespace", "same_namespace_retry_forbidden",
            "automatic_retry_forbidden", "exact_numeric_process_group_only",
            "pod1_and_pod2_gpu0_forbidden",
        },
        "failure_policy",
    )
    for key in failure:
        if failure.get(key) is not True:
            raise CaptureContractError(f"failure_policy.{key} must remain true")
    authorization = _require_mapping(plan.get("authorization"), "authorization")
    _require_exact_keys(
        authorization,
        {
            "capture_authorized", "attestation_authorized_only_after_complete_capture",
            "first_reset_probe_authorized", "scientific_training_authorized",
            "second_seed_authorized", "judge_authorized", "promotion_authorized",
            "hardware_authorized",
        },
        "authorization",
    )
    if authorization.get("capture_authorized") is not True:
        raise CaptureContractError("capture is not authorized")
    for key in ("attestation_authorized_only_after_complete_capture",):
        if authorization.get(key) is not True:
            raise CaptureContractError(f"authorization.{key} must remain true")
    for key in (
        "first_reset_probe_authorized", "scientific_training_authorized",
        "second_seed_authorized", "judge_authorized", "promotion_authorized",
        "hardware_authorized",
    ):
        if authorization.get(key) is not False:
            raise CaptureContractError(f"authorization.{key} must remain false")
    derivation = _require_mapping(plan.get("runtime_recipe_derivation"), "runtime_recipe_derivation")
    _require_exact_keys(
        derivation,
        {
            "source", "keep_all_task_motion_bank_seed_num_env_overrides",
            "deduplicate_identical_hydra_keys", "replace_executable_train_with_play",
            "remove_keys", "add_keys",
            "runtime_hard_contract_must_equal_teacher_checkpoint_hard_contract_before_first_state",
            "seed_must_be_applied_by_play",
        },
        "runtime_recipe_derivation",
    )
    if set(derivation.get("remove_keys", [])) != set(EXPECTED_REMOVE_KEYS):
        raise CaptureContractError("runtime derivation remove_keys must equal the frozen train-only set")
    if set(derivation.get("add_keys", [])) != set(EXPECTED_ADD_KEYS):
        raise CaptureContractError("runtime derivation add_keys must equal the frozen capture set")
    for key in (
        "keep_all_task_motion_bank_seed_num_env_overrides",
        "deduplicate_identical_hydra_keys",
        "replace_executable_train_with_play",
        "runtime_hard_contract_must_equal_teacher_checkpoint_hard_contract_before_first_state",
        "seed_must_be_applied_by_play",
    ):
        if derivation.get(key) is not True:
            raise CaptureContractError(f"runtime derivation {key} must remain true")
    if derivation.get("source") != "teacher_checkpoint.run_binding exact training_argv":
        raise CaptureContractError("runtime derivation source changed")

    teacher = _require_mapping(plan.get("teacher_checkpoint"), "teacher_checkpoint")
    _require_exact_keys(
        teacher,
        {
            "path", "sha256", "embedded_iteration", "floating_elements",
            "nonfinite_floating_elements", "fresh_lineage", "training_source_commit",
            "hard_contract", "launch_claim", "run_binding", "milestone_receipt",
        },
        "teacher_checkpoint",
    )
    _canonical_absolute_path(teacher.get("path"), "teacher_checkpoint.path")
    _require_sha256(teacher.get("sha256"), "teacher_checkpoint.sha256")
    if teacher.get("embedded_iteration") != 500:
        raise CaptureContractError("teacher checkpoint must be exact milestone 500")
    _require_plain_int(teacher.get("floating_elements"), "teacher floating elements", minimum=1)
    if teacher.get("nonfinite_floating_elements") != 0 or teacher.get("fresh_lineage") != 1:
        raise CaptureContractError("teacher checkpoint must be finite fresh lineage 1")
    if type(teacher.get("training_source_commit")) is not str or not COMMIT_RE.fullmatch(
        teacher["training_source_commit"]
    ):
        raise CaptureContractError("teacher training source commit is malformed")
    hard = _require_mapping(teacher.get("hard_contract"), "teacher hard contract")
    _require_exact_keys(hard, {"path", "sha256", "schema_version"}, "teacher hard contract")
    _canonical_absolute_path(hard.get("path"), "teacher hard contract path")
    _require_sha256(hard.get("sha256"), "teacher hard contract SHA")
    if hard.get("schema_version") != 3:
        raise CaptureContractError("teacher hard contract must be schema 3")
    for label in ("launch_claim", "run_binding", "milestone_receipt"):
        row = _require_mapping(teacher.get(label), f"teacher_checkpoint.{label}")
        _require_exact_keys(row, {"path", "file_sha256", "content_sha256"}, label)
        _canonical_absolute_path(row.get("path"), f"{label}.path")
        _require_sha256(row.get("file_sha256"), f"{label}.file_sha256")
        _require_sha256(row.get("content_sha256"), f"{label}.content_sha256")

    motions = plan.get("ordered_motion_inputs")
    if not isinstance(motions, list) or len(motions) != 2:
        raise CaptureContractError("exactly two ordered motion inputs are required")
    for index, raw_row in enumerate(motions):
        row = _require_mapping(raw_row, f"ordered_motion_inputs[{index}]")
        _require_exact_keys(row, {"path", "sha256"}, f"ordered_motion_inputs[{index}]")
        _canonical_absolute_path(row.get("path"), f"ordered_motion_inputs[{index}].path")
        _require_sha256(row.get("sha256"), f"ordered_motion_inputs[{index}].sha256")
    bank = _require_mapping(plan.get("question_bank"), "question_bank")
    _require_exact_keys(bank, {"path", "sha256"}, "question_bank")
    _canonical_absolute_path(bank.get("path"), "question_bank.path")
    _require_sha256(bank.get("sha256"), "question_bank.sha256")


def _derive_argv(plan: Mapping[str, Any], binding: Mapping[str, Any]) -> list[str]:
    _validate_plan(plan)
    content = _require_mapping(binding.get("content"), "run_binding.content")
    base = content.get("training_argv")
    if not isinstance(base, list) or len(base) < 3 or not all(type(value) is str for value in base):
        raise CaptureContractError("run binding lacks a string training_argv")
    runtime = _require_mapping(plan["runtime_environment"], "runtime_environment")
    python = _require_mapping(runtime["python"], "runtime_environment.python")
    if base[0] != python["requested_path"]:
        raise CaptureContractError("training argv executable differs from the exact venv Python entry")
    teacher_source = _require_mapping(content.get("source"), "run_binding.content.source")
    teacher_checkout = _canonical_absolute_path(
        teacher_source.get("checkout"), "run_binding.content.source.checkout"
    )
    expected_teacher_entry = teacher_checkout / "hope_training/whole_body_tracking/scripts/train.py"
    if base[1] != str(expected_teacher_entry):
        raise CaptureContractError("training argv entry differs from the teacher binding source train.py")
    derivation = _require_mapping(plan["runtime_recipe_derivation"], "runtime_recipe_derivation")
    removed = set(derivation["remove_keys"])
    seen: dict[str, tuple[str, str]] = {}
    retained: list[str] = []
    for argument in base[2:]:
        key = _normal_key(argument)
        value = argument.split("=", 1)[1]
        if key in removed:
            continue
        if key in seen:
            if seen[key] != (argument, value):
                raise CaptureContractError(f"conflicting duplicate Hydra key: {key}")
            continue
        seen[key] = (argument, value)
        retained.append(argument)
    teacher = _require_mapping(plan["teacher_checkpoint"], "teacher_checkpoint")
    contract = _require_mapping(plan["capture_contract"], "capture_contract")
    motions = plan.get("ordered_motion_inputs")
    if not isinstance(motions, list) or len(motions) != 2:
        raise CaptureContractError("exactly two ordered motion inputs are required")
    bank = _require_mapping(plan["question_bank"], "question_bank")
    required = {
        "task": "HOPEPingPongVirtualBall",
        "algo": "ppo",
        "headless": "true",
        "device": str(contract["runtime_device"]),
        "num_envs": str(contract["num_envs"]),
        "seed": str(contract["seed"]),
        "task.motion.wrap_teleport": "false",
        "task.motion.post_swing_start_prob": str(contract["post_swing_start_prob"]),
        "motion_file": str(motions[0]["path"]),
        "motion_file_2": str(motions[1]["path"]),
        "task.racket.question_bank": str(bank["path"]),
    }
    for key, expected in required.items():
        if seen.get(key, (None, None))[1] != expected:
            raise CaptureContractError(
                f"training recipe mismatch for {key}: {seen.get(key)!r} != {expected!r}"
            )
    output = Path(str(contract["output_directory"]))
    additions = [
        f"checkpoint={teacher['path']}",
        f"+task.motion.post_swing_capture_output_dir={output}",
        f"+task.motion.post_swing_capture_target_count={contract['target_count']}",
        f"post_swing_capture_max_steps={contract['max_inference_steps']}",
    ]
    for argument in additions:
        key = _normal_key(argument)
        if key in seen:
            raise CaptureContractError(f"capture addition already exists in training recipe: {key}")
        seen[key] = (argument, argument.split("=", 1)[1])
    source = _require_mapping(plan["capture_source"], "capture_source")
    argv = [
        base[0],
        str(Path(str(source["checkout"])) / "hope_training/whole_body_tracking/scripts/play.py"),
        *retained,
        *additions,
    ]
    keys = [_normal_key(argument) for argument in argv[2:]]
    if len(keys) != len(set(keys)):
        raise CaptureContractError("derived capture argv contains duplicate Hydra keys")
    return argv


def _gpu_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _require_mapping(plan["runtime_environment"], "runtime environment")
    tool = _require_mapping(runtime["tools"]["nvidia_smi"], "nvidia-smi tool")
    executable = Path(str(tool["path"]))
    _verify_executable_row(executable, tool, "nvidia-smi executable")
    clean_environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    try:
        gpu_rows = subprocess.check_output(
            [str(executable), "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            env=clean_environment,
        ).strip().splitlines()
        app_rows = subprocess.check_output(
            [
                str(executable), "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            env=clean_environment,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CaptureContractError(f"cannot inventory GPUs: {exc}") from exc
    gpus: list[dict[str, Any]] = []
    uuid_to_index: dict[str, int] = {}
    for raw in gpu_rows:
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise CaptureContractError(f"malformed nvidia-smi GPU row: {raw!r}")
        index = int(fields[0])
        if fields[1] in uuid_to_index or any(row["index"] == index for row in gpus):
            raise CaptureContractError("nvidia-smi reported duplicate GPU identity")
        uuid_to_index[fields[1]] = index
        gpus.append({"index": index, "uuid": fields[1]})
    apps: list[dict[str, Any]] = []
    for raw in app_rows:
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split(",", 2)]
        if len(fields) != 3 or fields[0] not in uuid_to_index or not fields[1].isdigit():
            raise CaptureContractError(f"malformed nvidia-smi compute row: {raw!r}")
        apps.append(
            {
                "gpu": uuid_to_index[fields[0]],
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
            }
        )
    return {"gpus": sorted(gpus, key=lambda row: row["index"]), "compute_apps": apps}


def _verify_file_row(path: Path, row: Mapping[str, Any], label: str) -> dict[str, Any]:
    raw = _read_regular_bytes(path, label)
    if len(raw) != row.get("bytes") or _sha256_bytes(raw) != row.get("sha256"):
        raise CaptureContractError(f"{label} bytes differ from the frozen row: {path}")
    return {"path": str(path), "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _verify_executable_row(path: Path, row: Mapping[str, Any], label: str) -> dict[str, Any]:
    proof = _verify_file_row(path, row, label)
    if path.lstat().st_mode & 0o111 == 0:
        raise CaptureContractError(f"{label} is not executable: {path}")
    return proof


def _expected_inventory(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("file_count", "total_file_bytes", "tree_content_sha256")
    }


def _verify_python_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    python = _require_mapping(runtime["python"], "runtime python")
    current = Path(str(python["requested_path"]))
    visited: set[Path] = set()
    chain_proof: list[dict[str, Any]] = []
    for index, raw_row in enumerate(python["symlink_chain"]):
        row = _require_mapping(raw_row, f"python chain {index}")
        expected_path = Path(str(row["path"]))
        if current != expected_path or current in visited:
            raise CaptureContractError("Python symlink chain path/order is not exact")
        visited.add(current)
        try:
            info = current.lstat()
        except OSError as exc:
            raise CaptureContractError(f"Python chain entry is missing: {current}") from exc
        if row["kind"] == "symlink":
            if not stat.S_ISLNK(info.st_mode):
                raise CaptureContractError(f"Python chain entry is not a symlink: {current}")
            target = os.readlink(current)
            if target != row["target"]:
                raise CaptureContractError(f"Python symlink target drifted: {current}")
            current = Path(os.path.normpath(str(current.parent / target))) if not Path(target).is_absolute() else Path(os.path.normpath(target))
            chain_proof.append({"kind": "symlink", "path": str(expected_path), "target": target})
        else:
            if index != len(python["symlink_chain"]) - 1 or not stat.S_ISREG(info.st_mode):
                raise CaptureContractError("Python chain must terminate in one regular executable")
            proof = _verify_file_row(current, row, "resolved Python executable")
            if info.st_mode & 0o111 == 0:
                raise CaptureContractError("resolved Python executable is not executable")
            chain_proof.append({"kind": "regular", **proof})
    if current != Path(str(python["resolved_path"])):
        raise CaptureContractError("resolved Python path differs from frozen chain")
    pyvenv = _require_mapping(python["pyvenv_cfg"], "pyvenv.cfg")
    pyvenv_proof = _verify_file_row(Path(str(pyvenv["path"])), pyvenv, "pyvenv.cfg")
    return {"symlink_chain": chain_proof, "pyvenv_cfg": pyvenv_proof}


def _load_lean_runtime(path: Path):
    name = f"post_swing_bound_lean_runtime_{_sha256_file(path)[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CaptureContractError("cannot load exact lean queue runtime")
    module = importlib.util.module_from_spec(spec)
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CaptureContractError(f"exact lean queue runtime import failed: {exc}") from exc
    finally:
        sys.dont_write_bytecode = prior
        sys.modules.pop(name, None)
    if not callable(getattr(module, "_load_binding", None)):
        raise CaptureContractError("exact lean queue runtime lacks _load_binding")
    return module


def _verify_teacher_lineage(plan: Mapping[str, Any], checkout: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    teacher = _require_mapping(plan["teacher_checkpoint"], "teacher checkpoint")
    files = _require_mapping(plan["capture_source"]["files"], "capture source files")
    lean_path = checkout / str(files["lean_queue_runtime"]["path"])
    lean = _load_lean_runtime(lean_path)
    binding_row = _require_mapping(teacher["run_binding"], "run binding row")
    binding_path = Path(str(binding_row["path"]))
    try:
        binding, binding_content, claim, claim_content = lean._load_binding(binding_path)
    except Exception as exc:
        raise CaptureContractError(f"lean claim/binding validation failed: {exc}") from exc
    direct_binding = _verify_content_document(binding_row, "run binding")
    if binding != direct_binding or binding.get("content_sha256") != binding_row["content_sha256"]:
        raise CaptureContractError("run binding differs across direct and lean validators")
    claim_row = _require_mapping(teacher["launch_claim"], "launch claim row")
    direct_claim = _verify_content_document(claim_row, "launch claim")
    if claim != direct_claim or claim.get("content_sha256") != claim_row["content_sha256"]:
        raise CaptureContractError("launch claim differs across direct and lean validators")
    receipt_row = _require_mapping(teacher["milestone_receipt"], "milestone receipt row")
    receipt = _verify_content_document(receipt_row, "milestone receipt")
    _require_exact_keys(receipt, {"schema_version", "content", "content_sha256"}, "milestone receipt")
    if receipt.get("schema_version") != 1:
        raise CaptureContractError("milestone receipt schema_version must be 1")
    content = _require_mapping(receipt["content"], "milestone receipt content")
    _require_exact_keys(
        content,
        {
            "schema_version", "job_id", "binding_path", "binding_content_sha256",
            "claim_content_sha256", "milestone", "process_state_at_attestation",
            "checkpoint", "hard_contract",
        },
        "milestone receipt content",
    )
    binding_digest = _require_sha256(binding["content_sha256"], "binding digest")
    claim_digest = _require_sha256(claim["content_sha256"], "claim digest")
    if (
        content.get("schema_version") != 1
        or content.get("job_id") != binding_content.get("job_id")
        or content.get("binding_path") != str(binding_path)
        or content.get("binding_content_sha256") != binding_digest
        or content.get("claim_content_sha256") != claim_digest
        or content.get("milestone") != 500
    ):
        raise CaptureContractError("milestone receipt is rebound from its claim/binding or milestone")
    checkpoint = _require_mapping(content["checkpoint"], "receipt checkpoint")
    hard = _require_mapping(content["hard_contract"], "receipt hard contract")
    checkpoint_path = Path(str(teacher["path"]))
    hard_path = Path(str(teacher["hard_contract"]["path"]))
    if checkpoint_path.name != "model_500.pt" or hard_path != checkpoint_path.parent / "params/training_contract.json":
        raise CaptureContractError("checkpoint and adjacent hard-contract layout is not exact")
    if (
        checkpoint.get("path") != str(checkpoint_path)
        or checkpoint.get("sha256") != teacher["sha256"]
        or checkpoint.get("filename_iteration") != 500
        or checkpoint.get("embedded_iteration") != 500
        or checkpoint.get("floating_elements") != teacher["floating_elements"]
        or checkpoint.get("nonfinite_floating_elements") != 0
        or hard.get("path") != str(hard_path)
        or hard.get("schema_version") != 3
        or hard.get("sha256") != teacher["hard_contract"]["sha256"]
        or hard.get("lineage_exact") != 1
    ):
        raise CaptureContractError("milestone receipt checkpoint/hard-contract lineage differs")
    source = _require_mapping(binding_content.get("source"), "bound training source")
    if source.get("commit") != teacher["training_source_commit"]:
        raise CaptureContractError("teacher source commit differs from immutable binding")
    if binding_content.get("pod") != "pod2" or binding_content.get("gpu") != 1:
        raise CaptureContractError("teacher binding is not the frozen Pod2 GPU1 producer")
    if binding_content.get("claim_content_sha256") != claim_digest:
        raise CaptureContractError("binding does not retain exact claim content digest")
    checkpoint_proof = _verify_file_row(
        checkpoint_path,
        {"bytes": checkpoint_path.lstat().st_size, "sha256": teacher["sha256"]},
        "teacher checkpoint",
    )
    hard_raw = _read_regular_bytes(hard_path, "teacher hard contract")
    hard_document = _strict_json_loads(hard_raw, "teacher hard contract")
    if hard_document.get("schema_version") != 3 or _sha256_bytes(hard_raw) != teacher["hard_contract"]["sha256"]:
        raise CaptureContractError("actual teacher hard-contract bytes differ")
    return binding, {
        "job_id": binding_content["job_id"],
        "binding_content_sha256": binding_digest,
        "claim_content_sha256": claim_digest,
        "milestone_receipt_content_sha256": receipt["content_sha256"],
        "checkpoint": checkpoint_proof,
        "hard_contract_sha256": _sha256_bytes(hard_raw),
    }


def _verify_runtime(plan: Mapping[str, Any], current_script: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    started = time.monotonic_ns()
    _validate_plan(plan)
    source = _require_mapping(plan["capture_source"], "capture_source")
    checkout = Path(str(source["checkout"]))
    runtime = _require_mapping(plan["runtime_environment"], "runtime environment")
    tools = _require_mapping(runtime["tools"], "runtime tools")
    tool_proofs = {
        label: _verify_executable_row(Path(str(row["path"])), row, f"runtime tool {label}")
        for label, row in tools.items()
    }
    git_executable = Path(str(tools["git"]["path"]))
    with _open_real_directory(checkout):
        pass
    if _git_output(git_executable, checkout, "rev-parse", "HEAD") != source["commit"]:
        raise CaptureContractError("capture source commit mismatch")
    if _git_output(git_executable, checkout, "status", "--porcelain=v1", "--untracked-files=no"):
        raise CaptureContractError("capture source has tracked changes")
    source_inventory = _inventory(checkout, skip_git=True)
    if source_inventory != _expected_inventory(_require_mapping(source["full_tree"], "source full tree")):
        raise CaptureContractError("capture source full tree, including untracked/ignored files, drifted")
    files = _require_mapping(source["files"], "capture_source.files")
    verified_files: list[dict[str, Any]] = []
    for label, raw_row in files.items():
        row = _require_mapping(raw_row, f"capture_source.files.{label}")
        verified_files.append(
            {"label": label, **_verify_file_row(checkout / str(row["path"]), row, f"source file {label}")}
        )
    controller = checkout / str(files["controller"]["path"])
    if current_script.absolute() != controller:
        raise CaptureContractError("running launcher is not the exact bound controller path")
    expected_play = checkout / "hope_training/whole_body_tracking/scripts/play.py"
    if expected_play != checkout / str(files["inference_runner"]["path"]):
        raise CaptureContractError("inference runner label is not the exact play.py entry")
    asset = _require_mapping(source["ignored_runtime_asset"], "ignored runtime asset")
    asset_inventory = _inventory(checkout / str(asset["relative_path"]))
    if asset_inventory != _expected_inventory(asset):
        raise CaptureContractError("ignored runtime asset inventory drifted")
    node = _require_mapping(runtime["node"], "runtime node")
    if socket.gethostname() != node["hostname"]:
        raise CaptureContractError("runtime hostname differs from the Pod2 plan")
    if _sha256_file(Path(str(node["machine_id_path"]))) != node["machine_id_sha256"]:
        raise CaptureContractError("runtime machine-id differs from the Pod2 plan")
    if _sha256_file(Path(str(node["boot_id_path"]))) != node["boot_id_sha256"]:
        raise CaptureContractError("runtime boot-id differs from the Pod2 plan")
    gpu_state = _gpu_state(plan)
    gpu = _require_mapping(runtime["gpu"], "runtime GPU")
    expected_gpu = {"index": gpu["physical_index"], "uuid": gpu["uuid"]}
    if expected_gpu not in gpu_state["gpus"]:
        raise CaptureContractError("physical GPU2 UUID differs from the frozen plan")
    occupied = [row for row in gpu_state["compute_apps"] if row["gpu_uuid"] == gpu["uuid"]]
    if occupied:
        raise CaptureContractError(f"frozen Pod2 GPU2 is occupied: {occupied}")
    python_proof = _verify_python_runtime(runtime)
    tree_proofs: list[dict[str, Any]] = []
    for raw_row in runtime["runtime_trees"]:
        row = _require_mapping(raw_row, "runtime tree")
        inventory = _inventory(Path(str(row["path"])), skip_git=True)
        if inventory != _expected_inventory(row):
            raise CaptureContractError(f"runtime import tree drifted: {row['label']}")
        tree_proofs.append({"label": row["label"], "path": row["path"], **inventory})
    binding, lineage = _verify_teacher_lineage(plan, checkout)
    verified_inputs = [lineage["checkpoint"]]
    for label, raw_row in (
        ("question_bank", plan["question_bank"]),
        ("motion_0", plan["ordered_motion_inputs"][0]),
        ("motion_1", plan["ordered_motion_inputs"][1]),
    ):
        row = _require_mapping(raw_row, label)
        verified_inputs.append(
            {"label": label, **_verify_file_row(Path(str(row["path"])), {"bytes": Path(str(row["path"])).lstat().st_size, "sha256": row["sha256"]}, label)}
        )
    elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
    return binding, {
        "source_commit": source["commit"],
        "source_inventory": source_inventory,
        "asset_inventory": asset_inventory,
        "source_files": verified_files,
        "runtime_trees": tree_proofs,
        "tools": tool_proofs,
        "python": python_proof,
        "node": {
            "hostname": node["hostname"],
            "machine_id_sha256": node["machine_id_sha256"],
            "boot_id_sha256": node["boot_id_sha256"],
        },
        "gpu": expected_gpu,
        "gpu_apps": occupied,
        "lineage": lineage,
        "verified_inputs": verified_inputs,
        "verification_elapsed_ms": elapsed_ms,
    }


def _stable_runtime_proof(proof: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in proof.items() if key != "verification_elapsed_ms"}


def _exclusive_write_at(directory_fd: int, name: str, raw: bytes, mode: int = 0o600) -> None:
    if not name or "/" in name or name in {".", ".."}:
        raise CaptureContractError("artifact name must be one direct leaf")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CaptureContractError(f"cannot write no-clobber artifact: {name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _exclusive_write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    with _open_real_directory(path.parent) as parent_fd:
        _exclusive_write_at(parent_fd, path.name, raw, mode)


def _ensure_real_directory(path: Path, mode: int = 0o755) -> int:
    canonical = _canonical_absolute_path(str(path), "directory to create")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for part in canonical.parts[1:]:
            try:
                next_descriptor = os.open(part, flags | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(part, mode, dir_fd=descriptor)
                os.fsync(descriptor)
                next_descriptor = os.open(part, flags | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _leaf_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _create_direct_namespace(parent: Path, leaf: str) -> int:
    if not NAMESPACE_RE.fullmatch(leaf):
        raise CaptureContractError("namespace leaf is invalid")
    parent_fd = _ensure_real_directory(parent)
    try:
        if _leaf_exists(parent_fd, leaf):
            raise CaptureContractError(f"namespace is already spent: {parent / leaf}")
        os.mkdir(leaf, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(leaf, flags, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


@contextmanager
def _gpu_lease():
    with _open_real_directory(GPU_LEASE_PATH.parent) as parent_fd:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(GPU_LEASE_PATH.name, flags, 0o600, dir_fd=parent_fd)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise CaptureContractError("shared GPU2 lease is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CaptureContractError("shared Pod2 GPU2 lease is already held") from exc
            path_info = os.stat(GPU_LEASE_PATH.name, dir_fd=parent_fd, follow_symlinks=False)
            if _stat_signature(info) != _stat_signature(path_info):
                raise CaptureContractError("shared GPU2 lease path was replaced during acquisition")
            os.set_inheritable(descriptor, True)
            yield descriptor
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _lease_is_held() -> bool:
    if not os.path.lexists(GPU_LEASE_PATH):
        return False
    try:
        with _open_real_directory(GPU_LEASE_PATH.parent) as parent_fd:
            descriptor = os.open(GPU_LEASE_PATH.name, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return False
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            return False
        finally:
            os.close(descriptor)
    except OSError:
        return False


def _environment(plan: Mapping[str, Any]) -> dict[str, str]:
    runtime = _require_mapping(plan["runtime_environment"], "runtime environment")
    exact = dict(_require_mapping(runtime["environment"]["exact"], "exact environment"))
    pythonpath = [str(row["path"]) for row in runtime["runtime_trees"] if row["on_pythonpath"]]
    exact.update(
        {
            "CUDA_VISIBLE_DEVICES": str(plan["capture_contract"]["cuda_visible_devices"]),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath),
        }
    )
    return exact


def _compose_recipe(
    plan: Mapping[str, Any], argv: Sequence[str]
) -> tuple[bytes, dict[str, Any]]:
    """Resolve the exact capture recipe without creating a capture/launch namespace."""

    if not argv or not all(type(value) is str for value in argv):
        raise CaptureContractError("Hydra compose argv must be a non-empty string sequence")
    source = _canonical_absolute_path(
        plan["capture_source"]["checkout"], "capture_source.checkout"
    )
    cwd = source / "hope_training/whole_body_tracking"
    with _open_real_directory(cwd):
        pass
    environment = _environment(plan)
    command = [*argv, "--cfg", "job", "--resolve"]
    started = time.monotonic_ns()
    try:
        compose = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=plan["runtime_environment"]["compose_timeout_s"],
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
        output = exc.stdout or b""
        if isinstance(output, str):
            output = output.encode()
        if not isinstance(output, bytes):
            output = bytes(output)
        raise HydraComposeError(
            "Hydra compose exceeded the frozen timeout",
            output=output,
            elapsed_ms=elapsed_ms,
            returncode=None,
        ) from exc
    elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
    output = compose.stdout if compose.stdout is not None else b""
    if isinstance(output, str):
        output = output.encode()
    if not isinstance(output, bytes):
        raise CaptureContractError("Hydra compose returned a non-byte stdout payload")
    if type(compose.returncode) is not int:
        raise CaptureContractError("Hydra compose returned a non-integer status")
    if compose.returncode != 0:
        raise HydraComposeError(
            f"Hydra compose failed with rc={compose.returncode}",
            output=output,
            elapsed_ms=elapsed_ms,
            returncode=compose.returncode,
        )
    return output, {
        "cwd": str(cwd),
        "command_sha256": _sha256_bytes(_canonical_bytes(command)),
        "environment_sha256": _sha256_bytes(_canonical_bytes(environment)),
        "output_bytes": len(output),
        "output_sha256": _sha256_bytes(output),
        "elapsed_ms": elapsed_ms,
        "returncode": compose.returncode,
    }


def _load_plan(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], bytes]:
    raw = _read_regular_bytes(path, "capture plan")
    actual = _sha256_bytes(raw)
    expected = _require_sha256(expected_sha256, "expected plan SHA-256")
    if actual != expected:
        raise CaptureContractError(f"plan SHA mismatch: {actual} != {expected}")
    plan = _strict_json_loads(raw, "capture plan")
    _validate_plan(plan)
    return plan, raw


def _namespace_exists(parent: Path, leaf: str) -> bool:
    try:
        with _open_real_directory(parent) as parent_fd:
            return _leaf_exists(parent_fd, leaf)
    except CaptureContractError:
        return False


def _plan_summary(plan: Mapping[str, Any], plan_raw: bytes, script: Path) -> dict[str, Any]:
    binding, proof = _verify_runtime(plan, script)
    argv = _derive_argv(plan, binding)
    _compose_output, compose_proof = _compose_recipe(plan, argv)
    binding_after, proof_after = _verify_runtime(plan, script)
    if (
        _derive_argv(plan, binding_after) != argv
        or _stable_runtime_proof(proof_after) != _stable_runtime_proof(proof)
    ):
        raise CaptureContractError("source/input/runtime drifted during Hydra compose")
    plan_id = str(plan["plan_id"])
    return {
        "plan_sha256": _sha256_bytes(plan_raw),
        "source_commit": proof["source_commit"],
        "argv_sha256": _sha256_bytes(_canonical_bytes(argv)),
        "launch_root_lexists": _namespace_exists(LAUNCH_PARENT, plan_id),
        "capture_output_lexists": _namespace_exists(CAPTURE_PARENT, plan_id),
        "gpu": proof["gpu"],
        "gpu_lease_held": _lease_is_held(),
        "asset_inventory": proof["asset_inventory"],
        "runtime_verification_elapsed_ms": proof["verification_elapsed_ms"],
        "hydra_compose": compose_proof,
    }


def _record_failure(directory_fd: int, stage: str, exc: BaseException, **extra: Any) -> None:
    failure = {
        "schema_version": 1,
        "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **extra,
    }
    try:
        _exclusive_write_at(directory_fd, "failure.json", _canonical_bytes(failure) + b"\n")
    except (OSError, CaptureContractError):
        pass


def _proc_identity(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    stat_raw = _read_regular_bytes(proc_root / str(pid) / "stat", "process stat").decode("utf-8")
    close = stat_raw.rfind(")")
    if close < 0:
        raise CaptureContractError("process stat has no comm terminator")
    fields = stat_raw[close + 2 :].split()
    if len(fields) < 20:
        raise CaptureContractError("process stat is truncated")
    cmdline = _read_regular_bytes(proc_root / str(pid) / "cmdline", "process cmdline")
    argv = [part.decode("utf-8") for part in cmdline.rstrip(b"\0").split(b"\0") if part]
    return {
        "pid": pid,
        "state": fields[0],
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "starttime_ticks": int(fields[19]),
        "argv": argv,
    }


def _launch(plan: Mapping[str, Any], plan_raw: bytes, script: Path) -> dict[str, Any]:
    plan_id = str(plan["plan_id"])
    plan_sha = _sha256_bytes(plan_raw)
    contract = _require_mapping(plan["capture_contract"], "capture contract")
    with _gpu_lease() as lease_fd:
        capture_parent_fd = _ensure_real_directory(CAPTURE_PARENT)
        try:
            if _leaf_exists(capture_parent_fd, plan_id):
                raise CaptureContractError("capture namespace is already spent")
        finally:
            os.close(capture_parent_fd)
        launch_fd = _create_direct_namespace(LAUNCH_PARENT, plan_id)
        stage = "early_launch_intent"
        try:
            early_intent = {
                "schema_version": 1,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "plan_id": plan_id,
                "plan_sha256": plan_sha,
                "launch_root": str(LAUNCH_PARENT / plan_id),
                "capture_output": str(CAPTURE_PARENT / plan_id),
                "gpu_lease_path": str(GPU_LEASE_PATH),
                "handoff": "execve_same_pid_v1",
            }
            _exclusive_write_at(launch_fd, "launch_intent.json", _canonical_bytes(early_intent) + b"\n")
            _exclusive_write_at(launch_fd, "prereg.json", plan_raw)
            stage = "runtime_verification_before_compose"
            binding, runtime_proof = _verify_runtime(plan, script)
            argv = _derive_argv(plan, binding)
            argv_sha = _sha256_bytes(_canonical_bytes(argv))
            runtime_argv = {
                "schema_version": 1,
                "plan_sha256": plan_sha,
                "argv": argv,
                "argv_sha256": argv_sha,
            }
            _exclusive_write_at(launch_fd, "runtime_argv.json", _canonical_bytes(runtime_argv) + b"\n")
            prelaunch = {
                "schema_version": 1,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "plan_sha256": plan_sha,
                "argv_sha256": argv_sha,
                "runtime_proof": runtime_proof,
            }
            _exclusive_write_at(launch_fd, "prelaunch_receipt.json", _canonical_bytes(prelaunch) + b"\n")
            source = Path(str(plan["capture_source"]["checkout"]))
            cwd = source / "hope_training/whole_body_tracking"
            environment = _environment(plan)
            stage = "hydra_compose"
            try:
                compose_output, compose_proof = _compose_recipe(plan, argv)
            except HydraComposeError as exc:
                _exclusive_write_at(launch_fd, "hydra_compose.log", exc.output)
                raise
            _exclusive_write_at(launch_fd, "hydra_compose.log", compose_output)
            _exclusive_write_at(
                launch_fd,
                "hydra_compose_receipt.json",
                _canonical_bytes({"schema_version": 1, **compose_proof}) + b"\n",
            )
            stage = "runtime_verification_after_compose"
            binding_after, runtime_after = _verify_runtime(plan, script)
            if (
                _derive_argv(plan, binding_after) != argv
                or _stable_runtime_proof(runtime_after) != _stable_runtime_proof(runtime_proof)
            ):
                raise CaptureContractError("source/input/runtime drifted during Hydra compose")
            capture_parent_fd = _ensure_real_directory(CAPTURE_PARENT)
            try:
                if _leaf_exists(capture_parent_fd, plan_id):
                    raise CaptureContractError("capture namespace appeared during compose")
                os.mkdir(plan_id, 0o700, dir_fd=capture_parent_fd)
                os.fsync(capture_parent_fd)
            finally:
                os.close(capture_parent_fd)
            stage = "same_pid_session_handoff"
            identity_before_session = _proc_identity(os.getpid())
            if identity_before_session["pgid"] != identity_before_session["pid"]:
                try:
                    os.setsid()
                except OSError as exc:
                    raise CaptureContractError("controller could not become the same-PID session leader") from exc
            elif identity_before_session["sid"] != identity_before_session["pid"]:
                raise CaptureContractError(
                    "controller is a process-group leader but not a session leader; "
                    "launch it under the reviewed same-PID session contract"
                )
            identity = _proc_identity(os.getpid())
            if (
                identity["pid"] != identity["pgid"]
                or identity["pid"] != identity["sid"]
                or identity["state"] == "Z"
            ):
                raise CaptureContractError("same-PID process-group identity is invalid before exec")
            exec_intent = {
                "schema_version": 1,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "pid": identity["pid"],
                "pgid": identity["pgid"],
                "sid": identity["sid"],
                "leader_starttime_ticks": identity["starttime_ticks"],
                "plan_sha256": plan_sha,
                "argv_sha256": argv_sha,
                "environment_sha256": _sha256_bytes(_canonical_bytes(environment)),
                "source_commit": runtime_proof["source_commit"],
                "capture_output": str(CAPTURE_PARENT / plan_id),
                "run_log": str(LAUNCH_PARENT / plan_id / "run.log"),
                "gpu_lease_path": str(GPU_LEASE_PATH),
                "handoff": "execve_same_pid_v1",
            }
            _exclusive_write_at(launch_fd, "exec_intent.json", _canonical_bytes(exec_intent) + b"\n")
            log_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            log_fd = os.open("run.log", log_flags, 0o600, dir_fd=launch_fd)
            null_fd = os.open("/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            saved_cwd_fd = os.open(".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
            saved_stdio = [os.dup(descriptor) for descriptor in (0, 1, 2)]
            try:
                os.chdir(cwd)
                os.dup2(null_fd, 0, inheritable=True)
                os.dup2(log_fd, 1, inheritable=True)
                os.dup2(log_fd, 2, inheritable=True)
                os.set_inheritable(lease_fd, True)
                stage = "execve_same_pid"
                os.execve(argv[0], argv, environment)
            except BaseException:
                os.fchdir(saved_cwd_fd)
                for target, saved in enumerate(saved_stdio):
                    os.dup2(saved, target, inheritable=True)
                raise
            finally:
                for saved in saved_stdio:
                    os.close(saved)
                os.close(saved_cwd_fd)
                os.close(null_fd)
                os.close(log_fd)
            raise AssertionError("os.execve returned unexpectedly")
        except BaseException as exc:
            _record_failure(launch_fd, stage, exc)
            if isinstance(exc, CaptureContractError):
                raise
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise CaptureContractError(f"capture launch failed at {stage}: {exc}") from exc
        finally:
            os.close(launch_fd)
    raise AssertionError("successful same-PID execve must not return")


def _artifact_status(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"lexists": False, "kind": "absent", "bytes": None, "sha256": None}
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        return {"lexists": True, "kind": "symlink_rejected", "bytes": None, "sha256": None}
    if not stat.S_ISREG(info.st_mode):
        return {"lexists": True, "kind": "nonregular_rejected", "bytes": None, "sha256": None}
    raw = _read_regular_bytes(path, f"status artifact {path.name}")
    return {"lexists": True, "kind": "regular", "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _status_retry_authorization(
    plan: Mapping[str, Any],
    output: Path,
    current_script: Path,
    plan_sha256: str,
) -> dict[str, Any]:
    """Load the committed authorization for one post-fix attestor retry."""

    if not current_script.is_absolute() or current_script.is_symlink():
        raise CaptureContractError("running status controller must be one absolute regular path")
    try:
        current_script = current_script.resolve(strict=True)
    except OSError as exc:
        raise CaptureContractError("running status controller path is unavailable") from exc
    source = current_script.parents[1]
    expected_controller = source / "scripts/run_preregistered_post_swing_capture.py"
    if current_script != expected_controller:
        raise CaptureContractError("running status controller is outside its own source checkout")
    with _open_real_directory(source):
        pass
    runtime = _require_mapping(plan["runtime_environment"], "runtime environment")
    tools = _require_mapping(runtime["tools"], "runtime tools")
    git_row = _require_mapping(tools["git"], "runtime git tool")
    git_executable = Path(str(git_row["path"]))
    _verify_executable_row(git_executable, git_row, "runtime git tool")
    commit = _git_output(git_executable, source, "rev-parse", "HEAD")
    if not COMMIT_RE.fullmatch(commit):
        raise CaptureContractError("attestor source HEAD is not one exact commit")
    if _git_output(
        git_executable, source, "status", "--porcelain=v1", "--untracked-files=no"
    ):
        raise CaptureContractError("status authorization source has tracked changes")
    tracked = _git_output(
        git_executable,
        source,
        "ls-files",
        "--error-unmatch",
        RETRY_AUTHORIZATION_RELATIVE.as_posix(),
    )
    if tracked != RETRY_AUTHORIZATION_RELATIVE.as_posix():
        raise CaptureContractError("retry authorization is not one tracked source file")
    authorization_path = source / RETRY_AUTHORIZATION_RELATIVE
    raw = _read_regular_bytes(authorization_path, "attestor retry authorization")
    value = _strict_json_loads(raw, "attestor retry authorization")
    _require_exact_keys(
        value,
        {
            "schema_version", "artifact_kind", "authorization_id", "v3_plan",
            "capture", "teacher", "capture_source", "attestor_source", "decision",
        },
        "attestor retry authorization",
    )
    if (
        value.get("schema_version") != 1
        or value.get("artifact_kind") != RETRY_AUTHORIZATION_KIND
        or type(value.get("authorization_id")) is not str
        or not value["authorization_id"]
    ):
        raise CaptureContractError("attestor retry authorization header is malformed")
    plan_row = _require_mapping(value["v3_plan"], "retry v3 plan")
    _require_exact_keys(plan_row, {"plan_id", "file_sha256"}, "retry v3 plan")
    capture = _require_mapping(value["capture"], "retry capture")
    _require_exact_keys(
        capture,
        {
            "output_directory", "output_receipt", "capture_claim_sha256",
            "states_sha256", "result_sha256", "state_count",
        },
        "retry capture",
    )
    teacher_row = _require_mapping(value["teacher"], "retry teacher")
    _require_exact_keys(
        teacher_row,
        {"checkpoint_sha256", "hard_contract_sha256", "launch_claim_content_sha256"},
        "retry teacher",
    )
    capture_source = _require_mapping(value["capture_source"], "retry capture source")
    _require_exact_keys(
        capture_source, {"commit", "producer_source_sha256"}, "retry capture source"
    )
    attestor_source = _require_mapping(value["attestor_source"], "retry attestor source")
    _require_exact_keys(
        attestor_source, {"commit", "attestor_source_sha256"}, "retry attestor source"
    )
    decision = _require_mapping(value["decision"], "retry decision")
    _require_exact_keys(
        decision,
        {
            "capture_retry_authorized", "attestor_attempt2_authorized",
            "first_reset_probe_authorized", "scientific_training_authorized",
        },
        "retry decision",
    )
    for label, digest in (
        ("retry plan", plan_row.get("file_sha256")),
        ("retry capture claim", capture.get("capture_claim_sha256")),
        ("retry states", capture.get("states_sha256")),
        ("retry result", capture.get("result_sha256")),
        ("retry checkpoint", teacher_row.get("checkpoint_sha256")),
        ("retry hard contract", teacher_row.get("hard_contract_sha256")),
        ("retry launch claim", teacher_row.get("launch_claim_content_sha256")),
        ("retry producer", capture_source.get("producer_source_sha256")),
        ("retry attestor", attestor_source.get("attestor_source_sha256")),
    ):
        _require_sha256(digest, label)
    for label, commit_value in (
        ("retry capture source", capture_source.get("commit")),
        ("retry attestor source", attestor_source.get("commit")),
    ):
        if type(commit_value) is not str or not COMMIT_RE.fullmatch(commit_value):
            raise CaptureContractError(f"{label} commit must be 40 lowercase hex characters")
    frozen = _require_mapping(plan["teacher_checkpoint"], "frozen teacher")
    producer = _require_mapping(
        plan["capture_source"]["files"]["producer"], "frozen capture producer"
    )
    if (
        plan_row != {"plan_id": plan["plan_id"], "file_sha256": plan_sha256}
        or capture
        != {
            "output_directory": str(output),
            "output_receipt": str(output / "teacher_receipt.json"),
            "capture_claim_sha256": _sha256_file(output / "natural_wrap_capture.claim.json"),
            "states_sha256": _sha256_file(output / "natural_wrap_states.npz"),
            "result_sha256": _sha256_file(output / "natural_wrap_capture.json"),
            "state_count": plan["capture_contract"]["target_count"],
        }
        or teacher_row
        != {
            "checkpoint_sha256": frozen["sha256"],
            "hard_contract_sha256": frozen["hard_contract"]["sha256"],
            "launch_claim_content_sha256": frozen["launch_claim"]["content_sha256"],
        }
        or capture_source
        != {
            "commit": plan["capture_source"]["commit"],
            "producer_source_sha256": producer["sha256"],
        }
        or decision
        != {
            "capture_retry_authorized": False,
            "attestor_attempt2_authorized": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
        }
    ):
        raise CaptureContractError("retry authorization is rebound from immutable v3 evidence")
    return {
        "receipt_binding": {
            "authorization_id": value["authorization_id"],
            "file_sha256": _sha256_bytes(raw),
            "v3_plan_file_sha256": plan_sha256,
        },
        "attestor_source": {
            "commit": attestor_source["commit"],
            "clean": True,
            "attestor_source_sha256": attestor_source["attestor_source_sha256"],
        },
        "status_source_commit": commit,
    }


def _validate_teacher_receipt_binding(
    plan: Mapping[str, Any],
    output: Path,
    raw: bytes,
    *,
    current_script: Path,
    plan_sha256: str,
) -> None:
    receipt = _strict_json_loads(raw, "teacher receipt")
    teacher = _require_mapping(receipt.get("teacher"), "teacher receipt teacher")
    frozen = _require_mapping(plan["teacher_checkpoint"], "frozen teacher")
    if (
        teacher.get("source_commit") != frozen["training_source_commit"]
        or teacher.get("checkpoint_sha256") != frozen["sha256"]
        or teacher.get("training_contract_sha256") != frozen["hard_contract"]["sha256"]
        or teacher.get("training_contract_schema_version") != 3
        or teacher.get("fresh_lineage") is not True
    ):
        raise CaptureContractError("teacher receipt is rebound from the frozen checkpoint")
    receipt_motions = receipt.get("motion_clips")
    expected_motions = plan["ordered_motion_inputs"]
    if not isinstance(receipt_motions, list) or len(receipt_motions) != len(expected_motions):
        raise CaptureContractError("teacher receipt ordered motion count differs")
    for index, (actual, expected) in enumerate(zip(receipt_motions, expected_motions)):
        actual = _require_mapping(actual, f"teacher receipt motion {index}")
        if set(actual) != {"index", "sha256"} or actual.get("index") != index or actual.get("sha256") != expected["sha256"]:
            raise CaptureContractError("teacher receipt is rebound from the ordered motions")
    attestation = _require_mapping(receipt.get("attestation"), "teacher attestation")
    _require_exact_keys(
        attestation,
        {
            "schema_version", "artifact_kind", "capture_result_sha256",
            "capture_result_relative_path", "capture_claim_sha256",
            "capture_claim_relative_path", "checkpoint", "hard_contract",
            "checkpoint_source", "capture_source", "attestor_source",
            "retry_authorization",
        },
        "teacher attestation",
    )
    if (
        attestation.get("schema_version") != 2
        or attestation.get("artifact_kind")
        != "hope_post_swing_teacher_capture_attestation"
    ):
        raise CaptureContractError("teacher attestation schema is not lineage-split v2")
    result_path = output / str(attestation.get("capture_result_relative_path", ""))
    claim_path = output / str(attestation.get("capture_claim_relative_path", ""))
    if result_path.parent != output or claim_path.parent != output:
        raise CaptureContractError("teacher receipt capture paths escape the output namespace")
    if _sha256_file(result_path) != attestation.get("capture_result_sha256"):
        raise CaptureContractError("teacher receipt is rebound from capture result bytes")
    if _sha256_file(claim_path) != attestation.get("capture_claim_sha256"):
        raise CaptureContractError("teacher receipt is rebound from capture claim bytes")
    checkpoint_source = _require_mapping(attestation.get("checkpoint_source"), "checkpoint source")
    if (
        checkpoint_source.get("commit") != frozen["training_source_commit"]
        or checkpoint_source.get("launch_claim_content_sha256") != frozen["launch_claim"]["content_sha256"]
    ):
        raise CaptureContractError("teacher receipt checkpoint lineage is rebound")
    checkpoint = _require_mapping(attestation.get("checkpoint"), "attested checkpoint")
    _require_exact_keys(
        checkpoint,
        {
            "sha256", "training_contract_schema_version",
            "training_contract_sha256", "training_contract_lineage_exact",
            "training_launch_claim_sha256",
        },
        "attested checkpoint",
    )
    hard_contract = _require_mapping(
        attestation.get("hard_contract"), "attested hard contract"
    )
    _require_exact_keys(
        hard_contract, {"sha256", "schema_version"}, "attested hard contract"
    )
    if (
        checkpoint.get("sha256") != frozen["sha256"]
        or checkpoint.get("training_contract_schema_version") != 3
        or checkpoint.get("training_contract_sha256")
        != frozen["hard_contract"]["sha256"]
        or checkpoint.get("training_contract_lineage_exact") is not True
        or checkpoint.get("training_launch_claim_sha256")
        != frozen["launch_claim"]["content_sha256"]
        or hard_contract
        != {"sha256": frozen["hard_contract"]["sha256"], "schema_version": 3}
    ):
        raise CaptureContractError("teacher receipt checkpoint/hard-contract attestation is rebound")
    capture_source = _require_mapping(attestation.get("capture_source"), "capture source")
    _require_exact_keys(
        capture_source,
        {"commit", "clean", "producer_source_sha256"},
        "capture source",
    )
    producer_row = _require_mapping(
        plan["capture_source"]["files"]["producer"], "frozen capture producer"
    )
    if (
        capture_source.get("commit") != plan["capture_source"]["commit"]
        or capture_source.get("clean") is not True
        or capture_source.get("producer_source_sha256") != producer_row["sha256"]
    ):
        raise CaptureContractError("teacher receipt capture source is rebound")
    attestor_source = _require_mapping(attestation.get("attestor_source"), "attestor source")
    _require_exact_keys(
        attestor_source,
        {"commit", "clean", "attestor_source_sha256"},
        "attestor source",
    )
    retry_receipt = _require_mapping(
        attestation.get("retry_authorization"), "retry authorization receipt binding"
    )
    _require_exact_keys(
        retry_receipt,
        {"authorization_id", "file_sha256", "v3_plan_file_sha256"},
        "retry authorization receipt binding",
    )
    authorization = _status_retry_authorization(
        plan, output, current_script, plan_sha256
    )
    if (
        attestor_source != authorization["attestor_source"]
        or retry_receipt != authorization["receipt_binding"]
    ):
        raise CaptureContractError("teacher receipt attestor source is rebound")


def _status(
    plan: Mapping[str, Any],
    plan_raw: bytes,
    proc_root: Path = Path("/proc"),
    *,
    current_script: Path = Path(__file__).resolve(),
) -> dict[str, Any]:
    plan_id = str(plan["plan_id"])
    launch_root = LAUNCH_PARENT / plan_id
    output = CAPTURE_PARENT / plan_id
    plan_sha = _sha256_bytes(plan_raw)
    exec_intent = None
    runtime_argv = None
    alive = False
    identity_exact = False
    process_state = None
    if os.path.lexists(launch_root / "exec_intent.json"):
        exec_intent, _ = _read_regular_json(launch_root / "exec_intent.json", "exec intent")
        runtime_argv, _ = _read_regular_json(launch_root / "runtime_argv.json", "runtime argv")
        _require_exact_keys(
            exec_intent,
            {
                "schema_version", "started_utc", "pid", "pgid", "sid", "leader_starttime_ticks",
                "plan_sha256", "argv_sha256", "environment_sha256", "source_commit",
                "capture_output", "run_log", "gpu_lease_path", "handoff",
            },
            "exec intent",
        )
        _require_exact_keys(runtime_argv, {"schema_version", "plan_sha256", "argv", "argv_sha256"}, "runtime argv")
        expected_argv_sha = _sha256_bytes(_canonical_bytes(runtime_argv["argv"]))
        if (
            exec_intent.get("schema_version") != 1
            or runtime_argv.get("schema_version") != 1
            or exec_intent.get("plan_sha256") != plan_sha
            or runtime_argv.get("plan_sha256") != plan_sha
            or exec_intent.get("argv_sha256") != expected_argv_sha
            or runtime_argv.get("argv_sha256") != expected_argv_sha
            or exec_intent.get("capture_output") != str(output)
            or exec_intent.get("gpu_lease_path") != str(GPU_LEASE_PATH)
            or exec_intent.get("handoff") != "execve_same_pid_v1"
        ):
            raise CaptureContractError("status launch artifacts are rebound")
        pid = exec_intent.get("pid")
        if type(pid) is int and (proc_root / str(pid)).exists():
            try:
                identity = _proc_identity(pid, proc_root)
            except CaptureContractError:
                identity = None
            if identity is not None:
                process_state = identity["state"]
                alive = identity["state"] != "Z"
                identity_exact = alive and (
                    identity["pgid"] == exec_intent["pgid"] == pid
                    and identity["sid"] == exec_intent["sid"] == pid
                    and identity["starttime_ticks"] == exec_intent["leader_starttime_ticks"]
                    and identity["argv"] == runtime_argv["argv"]
                )
    artifacts: dict[str, Any] = {}
    for name in (
        "natural_wrap_capture.claim.json",
        "natural_wrap_states.npz",
        "natural_wrap_capture.json",
        "teacher_receipt.json",
    ):
        path = output / name
        artifacts[name] = _artifact_status(path)
    receipt_status = artifacts["teacher_receipt.json"]
    receipt_binding_exact = None
    if receipt_status["kind"] == "regular":
        try:
            raw = _read_regular_bytes(output / "teacher_receipt.json", "teacher receipt")
            _validate_teacher_receipt_binding(
                plan,
                output,
                raw,
                current_script=current_script,
                plan_sha256=plan_sha,
            )
            receipt_binding_exact = True
        except CaptureContractError:
            receipt_binding_exact = False
    return {
        "launch_root_lexists": os.path.lexists(launch_root),
        "capture_output_lexists": os.path.lexists(output),
        "exec_intent": exec_intent,
        "leader_alive": alive,
        "leader_process_state": process_state,
        "leader_identity_exact": identity_exact,
        "gpu_lease_held": _lease_is_held(),
        "teacher_receipt_binding_exact": receipt_binding_exact,
        "artifacts": artifacts,
        "gpu_state": _gpu_state(plan),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("mode", choices=("plan", "launch", "status"))
    args = parser.parse_args(argv)
    try:
        plan, raw = _load_plan(args.plan, args.expected_plan_sha256)
        if args.mode == "plan":
            result = _plan_summary(plan, raw, Path(__file__))
        elif args.mode == "launch":
            result = _launch(plan, raw, Path(__file__))
        else:
            result = _status(plan, raw)
    except (CaptureContractError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
