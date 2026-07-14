#!/usr/bin/env python3
"""Fail-closed runtime binding and milestone attestation for the lean queue.

The trainer calls :func:`publish_run_binding` immediately after choosing its
RSL-RL log directory.  The standalone ``attest`` command later follows only
that immutable binding; it never scans logs or guesses a checkpoint directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


class LeanQueueRuntimeError(RuntimeError):
    """A queue claim, process binding, or milestone failed closed."""


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
LOG_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
WBT_RELATIVE = Path("hope_training/whole_body_tracking")
TRAIN_ENTRY_RELATIVE = WBT_RELATIVE / "scripts/train.py"
CLAIM_NAME = "queue_claim.json"
BINDING_NAME = "run_binding.json"
MILESTONE_DIR_NAME = "milestones"
TRAINING_CONTRACT_SCHEMA_VERSION = 3


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or not SHA256_RE.fullmatch(value):
        raise LeanQueueRuntimeError(f"{label} must be 64 lowercase hex characters")
    return value


def _require_plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LeanQueueRuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LeanQueueRuntimeError(f"{label} must be a mapping")
    return value


def _require_text(value: Any, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\n" in value:
        raise LeanQueueRuntimeError(f"{label} must be one non-empty line")
    return value


def _canonical_absolute_path(value: Any, label: str) -> Path:
    raw = _require_text(value, label)
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise LeanQueueRuntimeError(f"{label} must be an absolute path without ..")
    normalized = Path(os.path.normpath(raw))
    if str(normalized) != raw.rstrip("/"):
        raise LeanQueueRuntimeError(f"{label} must already be normalized")
    return normalized


def _stat_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _read_regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise LeanQueueRuntimeError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise LeanQueueRuntimeError(f"{label} must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise LeanQueueRuntimeError(f"cannot open {label} without following links: {path}") from exc
    try:
        opened = os.fstat(fd)
        if _stat_signature(opened) != _stat_signature(before):
            raise LeanQueueRuntimeError(f"{label} changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after_path = path.lstat()
    except FileNotFoundError as exc:
        raise LeanQueueRuntimeError(f"{label} vanished while reading: {path}") from exc
    if (
        _stat_signature(before) != _stat_signature(after_fd)
        or _stat_signature(before) != _stat_signature(after_path)
    ):
        raise LeanQueueRuntimeError(f"{label} changed while reading: {path}")
    return b"".join(chunks)


def _read_regular_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeanQueueRuntimeError(f"{label} is not canonical JSON: {path}") from exc
    return _require_mapping(value, label), raw


def _atomic_publish_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    if not path.is_absolute():
        raise LeanQueueRuntimeError(f"{label} path must be absolute")
    parent = path.parent
    try:
        parent_info = parent.lstat()
    except FileNotFoundError as exc:
        raise LeanQueueRuntimeError(f"{label} parent is missing: {parent}") from exc
    if not stat.S_ISDIR(parent_info.st_mode):
        raise LeanQueueRuntimeError(f"{label} parent must be a real directory: {parent}")
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temp = parent / f".{path.name}.tmp.{os.getpid()}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(temp, flags, 0o600)
    except FileExistsError as exc:
        raise LeanQueueRuntimeError(f"{label} temporary path already exists: {temp}") from exc
    published = False
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            os.link(temp, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise LeanQueueRuntimeError(f"{label} already exists; overwrite is forbidden: {path}") from exc
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        if published:
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


def _validate_claim(
    claim_path: Path, *, expected_digest: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], str]:
    claim, _raw = _read_regular_json(claim_path, "queue claim")
    if claim.get("schema_version") != 2:
        raise LeanQueueRuntimeError("queue claim schema_version must be 2")
    content = _require_mapping(claim.get("content"), "queue claim content")
    if content.get("schema_version") != 1:
        raise LeanQueueRuntimeError("queue claim content schema_version must be 1")
    digest = _require_sha256(claim.get("content_sha256"), "queue claim digest")
    if canonical_sha256(content) != digest:
        raise LeanQueueRuntimeError("queue claim canonical digest mismatch")
    if expected_digest is not None and digest != _require_sha256(
        expected_digest, "expected queue claim digest"
    ):
        raise LeanQueueRuntimeError("queue claim does not match trainer launch digest")
    argv_without_claim = content.get("training_argv_without_claim")
    full_argv = claim.get("training_argv")
    if not isinstance(argv_without_claim, list) or not all(
        type(item) is str for item in argv_without_claim
    ):
        raise LeanQueueRuntimeError("queue claim caller argv must be a string list")
    if not isinstance(full_argv, list) or not all(type(item) is str for item in full_argv):
        raise LeanQueueRuntimeError("queue claim full argv must be a string list")
    expected_argv = [*argv_without_claim, f"++training_launch_claim_sha256={digest}"]
    if full_argv != expected_argv:
        raise LeanQueueRuntimeError("queue claim full argv does not self-bind its digest")
    return claim, content, digest


def _claim_layout(
    claim: Mapping[str, Any],
    content: Mapping[str, Any],
    *,
    claim_path: Path,
    binding_path: Path,
    log_dir: Path,
) -> dict[str, Any]:
    run_dir = _canonical_absolute_path(content.get("run_dir"), "queue run_dir")
    source = _require_mapping(content.get("source"), "queue source")
    source_checkout = _canonical_absolute_path(source.get("checkout"), "source checkout")
    commit = _require_text(source.get("commit"), "source commit")
    if not COMMIT_RE.fullmatch(commit):
        raise LeanQueueRuntimeError("source commit must be 40 lowercase hex characters")
    if claim_path != run_dir / CLAIM_NAME:
        raise LeanQueueRuntimeError("claim path does not equal run_dir/queue_claim.json")
    if binding_path != run_dir / BINDING_NAME:
        raise LeanQueueRuntimeError("binding path does not equal run_dir/run_binding.json")

    run_name = _require_text(content.get("run_name"), "run_name")
    log_root = (source_checkout / WBT_RELATIVE / "logs/rsl_rl").resolve(strict=False)
    resolved_log = log_dir.resolve(strict=False)
    try:
        relative_log = resolved_log.relative_to(log_root)
    except ValueError as exc:
        raise LeanQueueRuntimeError("RSL log dir is outside the source-owned log root") from exc
    if len(relative_log.parts) != 2:
        raise LeanQueueRuntimeError("RSL log dir must be exactly experiment/timestamp_run_name")
    suffix = f"_{run_name}"
    if not relative_log.name.endswith(suffix):
        raise LeanQueueRuntimeError("RSL log dir does not end with the claimed run_name")
    stamp = relative_log.name[: -len(suffix)]
    if not LOG_STAMP_RE.fullmatch(stamp):
        raise LeanQueueRuntimeError("RSL log dir lacks the canonical timestamp prefix")

    full_argv = claim["training_argv"]
    expected_entry = str(source_checkout / TRAIN_ENTRY_RELATIVE)
    if len(full_argv) < 2 or full_argv[1] != expected_entry:
        raise LeanQueueRuntimeError("queue claim train.py does not belong to source checkout")
    required_overrides = (
        f"++training_queue_claim_path={claim_path}",
        f"++training_run_binding_path={binding_path}",
    )
    for override in required_overrides:
        if full_argv.count(override) != 1:
            raise LeanQueueRuntimeError(f"queue claim must contain exactly one {override.split('=', 1)[0]}")

    pod = _require_text(content.get("pod"), "claim pod")
    if pod not in {"pod1", "pod2"}:
        raise LeanQueueRuntimeError("claim pod must be pod1 or pod2")
    gpu = _require_plain_int(content.get("gpu"), "claim GPU")
    if gpu not in (0, 1, 2):
        raise LeanQueueRuntimeError("claim GPU must be 0, 1, or 2")
    budget = _require_mapping(content.get("budget"), "claim budget")
    milestones = budget.get("milestones")
    if not isinstance(milestones, list) or not milestones or any(
        type(value) is not int or value <= 0 for value in milestones
    ):
        raise LeanQueueRuntimeError("claim milestones must be positive integers")
    if milestones != sorted(set(milestones)):
        raise LeanQueueRuntimeError("claim milestones must be unique and sorted")
    return {
        "run_dir": run_dir,
        "source": {"checkout": str(source_checkout), "commit": commit},
        "run_name": run_name,
        "rsl_log_dir": resolved_log,
        "pod": pod,
        "gpu": gpu,
        "milestones": milestones,
    }


def _proc_starttime(stat_text: str) -> int:
    close = stat_text.rfind(")")
    if close < 0:
        raise LeanQueueRuntimeError("proc stat lacks a closing command parenthesis")
    fields = stat_text[close + 2 :].split()
    if len(fields) <= 19 or not fields[19].isdigit():
        raise LeanQueueRuntimeError("proc stat lacks a numeric starttime")
    value = int(fields[19])
    if value <= 0:
        raise LeanQueueRuntimeError("proc starttime must be positive")
    return value


def _process_identity(
    pid: int,
    *,
    proc_root: Path,
    getpgid: Callable[[int], int],
) -> dict[str, Any]:
    proc_dir = proc_root / str(pid)
    try:
        stat_before = (proc_dir / "stat").read_text(encoding="utf-8")
        cmdline = (proc_dir / "cmdline").read_bytes()
    except FileNotFoundError as exc:
        raise LeanQueueRuntimeError(f"process {pid} vanished before binding") from exc
    try:
        pgid = getpgid(pid)
    except ProcessLookupError as exc:
        raise LeanQueueRuntimeError(f"process {pid} vanished before PGID binding") from exc
    try:
        stat_after = (proc_dir / "stat").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise LeanQueueRuntimeError(f"process {pid} vanished while binding identity") from exc
    starttime = _proc_starttime(stat_before)
    if _proc_starttime(stat_after) != starttime:
        raise LeanQueueRuntimeError(f"process {pid} changed while reading identity")
    argv = [part.decode("utf-8", "strict") for part in cmdline.split(b"\0") if part]
    if not argv:
        raise LeanQueueRuntimeError(f"process {pid} has an empty cmdline")
    return {"pid": pid, "pgid": pgid, "starttime_ticks": starttime, "argv": argv}


def _verify_git_source(source_checkout: Path, expected_commit: str) -> dict[str, Any]:
    """Verify the trainer's source at binding time, not only in launcher preflight."""

    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"

    def run(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(source_checkout), *arguments],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise LeanQueueRuntimeError(
                f"cannot verify trainer source checkout: {source_checkout}"
            ) from exc
        return completed.stdout.strip()

    observed = run("rev-parse", "HEAD")
    if observed != expected_commit:
        raise LeanQueueRuntimeError(
            f"trainer source HEAD {observed!r} differs from claimed {expected_commit}"
        )
    status = run("status", "--porcelain", "--untracked-files=all")
    if status:
        raise LeanQueueRuntimeError("trainer source checkout is dirty at binding time")
    return {"head": observed, "clean": True}


def publish_run_binding(
    *,
    claim_path: str | Path,
    binding_path: str | Path,
    log_dir: str | Path,
    claim_digest: str,
    actual_argv: list[str] | tuple[str, ...],
    pid: int | None = None,
    proc_root: str | Path = "/proc",
    getpgid: Callable[[int], int] = os.getpgid,
    environ: Mapping[str, str] | None = None,
    source_verifier: Callable[[Path, str], Mapping[str, Any]] = _verify_git_source,
) -> dict[str, Any]:
    """Validate one live trainer and publish its immutable RSL directory binding."""

    claim_path_obj = _canonical_absolute_path(str(claim_path), "claim path")
    binding_path_obj = _canonical_absolute_path(str(binding_path), "binding path")
    log_dir_obj = _canonical_absolute_path(str(log_dir), "RSL log dir")
    claim, content, digest = _validate_claim(
        claim_path_obj, expected_digest=claim_digest
    )
    layout = _claim_layout(
        claim,
        content,
        claim_path=claim_path_obj,
        binding_path=binding_path_obj,
        log_dir=log_dir_obj,
    )
    argv = list(actual_argv)
    if not argv or any(type(item) is not str for item in argv):
        raise LeanQueueRuntimeError("actual trainer argv must be a non-empty string list")
    if argv != claim["training_argv"]:
        raise LeanQueueRuntimeError("actual trainer argv differs from the queue claim")
    process = _process_identity(
        os.getpid() if pid is None else _require_plain_int(pid, "PID", minimum=1),
        proc_root=Path(proc_root),
        getpgid=getpgid,
    )
    if process["pgid"] != process["pid"]:
        raise LeanQueueRuntimeError("queue trainer must be the leader of its isolated PGID")
    if process["argv"] != argv:
        raise LeanQueueRuntimeError("/proc cmdline differs from the claimed trainer argv")
    environment = os.environ if environ is None else environ
    if environment.get("CUDA_VISIBLE_DEVICES") != str(layout["gpu"]):
        raise LeanQueueRuntimeError("CUDA_VISIBLE_DEVICES does not match the claimed physical GPU")
    verified_source = source_verifier(
        Path(layout["source"]["checkout"]), layout["source"]["commit"]
    )
    if not isinstance(verified_source, Mapping):
        raise LeanQueueRuntimeError("source verifier result must be a mapping")
    source_state = dict(verified_source)
    expected_source_state = {"head": layout["source"]["commit"], "clean": True}
    if source_state != expected_source_state:
        raise LeanQueueRuntimeError("source verifier did not prove exact clean claimed source")

    binding_content = {
        "schema_version": 1,
        "job_id": _require_text(content.get("job_id"), "job_id"),
        "claim_path": str(claim_path_obj),
        "claim_content_sha256": digest,
        "binding_path": str(binding_path_obj),
        "rsl_log_dir": str(layout["rsl_log_dir"]),
        "process": process,
        "pod": layout["pod"],
        "gpu": layout["gpu"],
        "source": layout["source"],
        "source_state_at_binding": source_state,
        "run_name": layout["run_name"],
        "run_dir": str(layout["run_dir"]),
        "milestones": list(layout["milestones"]),
        "training_argv": list(claim["training_argv"]),
    }
    binding = {
        "schema_version": 1,
        "content": binding_content,
        "content_sha256": canonical_sha256(binding_content),
    }
    _atomic_publish_json(binding_path_obj, binding, "run binding")
    return binding


def _load_binding(
    binding_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding, _raw = _read_regular_json(binding_path, "run binding")
    if binding.get("schema_version") != 1:
        raise LeanQueueRuntimeError("run binding schema_version must be 1")
    content = _require_mapping(binding.get("content"), "run binding content")
    if content.get("schema_version") != 1:
        raise LeanQueueRuntimeError("run binding content schema_version must be 1")
    digest = _require_sha256(binding.get("content_sha256"), "run binding digest")
    if canonical_sha256(content) != digest:
        raise LeanQueueRuntimeError("run binding canonical digest mismatch")
    if _canonical_absolute_path(content.get("binding_path"), "bound binding path") != binding_path:
        raise LeanQueueRuntimeError("run binding path does not self-bind")
    claim_path = _canonical_absolute_path(content.get("claim_path"), "bound claim path")
    claim, claim_content, claim_digest = _validate_claim(
        claim_path,
        expected_digest=_require_sha256(
            content.get("claim_content_sha256"), "bound claim digest"
        ),
    )
    layout = _claim_layout(
        claim,
        claim_content,
        claim_path=claim_path,
        binding_path=binding_path,
        log_dir=_canonical_absolute_path(content.get("rsl_log_dir"), "bound RSL log dir"),
    )
    expected = {
        "job_id": claim_content.get("job_id"),
        "claim_content_sha256": claim_digest,
        "rsl_log_dir": str(layout["rsl_log_dir"]),
        "pod": layout["pod"],
        "gpu": layout["gpu"],
        "source": layout["source"],
        "source_state_at_binding": {
            "head": layout["source"]["commit"],
            "clean": True,
        },
        "run_name": layout["run_name"],
        "run_dir": str(layout["run_dir"]),
        "milestones": layout["milestones"],
        "training_argv": claim["training_argv"],
    }
    for key, value in expected.items():
        if content.get(key) != value:
            raise LeanQueueRuntimeError(f"run binding {key} differs from its queue claim")
    return binding, content, claim, claim_content


def _verify_bound_process(
    content: Mapping[str, Any],
    *,
    proc_root: Path,
    getpgid: Callable[[int], int],
) -> str:
    process = _require_mapping(content.get("process"), "bound process")
    pid = _require_plain_int(process.get("pid"), "bound PID", minimum=1)
    expected_pgid = _require_plain_int(process.get("pgid"), "bound PGID", minimum=1)
    expected_start = _require_plain_int(
        process.get("starttime_ticks"), "bound process starttime", minimum=1
    )
    expected_argv = process.get("argv")
    if expected_pgid != pid:
        raise LeanQueueRuntimeError("bound trainer PID must equal its isolated PGID")
    if expected_argv != content.get("training_argv"):
        raise LeanQueueRuntimeError("bound process argv differs from bound training argv")
    if not (proc_root / str(pid)).exists():
        return "exited"
    observed = _process_identity(pid, proc_root=proc_root, getpgid=getpgid)
    if observed["starttime_ticks"] != expected_start:
        raise LeanQueueRuntimeError("bound PID was reused with a different proc starttime")
    if observed["pgid"] != expected_pgid or observed["argv"] != expected_argv:
        raise LeanQueueRuntimeError("live process no longer matches the immutable binding")
    return "live"


def _tensor_finiteness(value: Any, torch_module: Any) -> dict[str, int]:
    tensor_count = 0
    floating_tensor_count = 0
    floating_elements = 0
    nonfinite_floating_elements = 0
    seen: set[int] = set()

    def visit(item: Any) -> None:
        nonlocal tensor_count, floating_tensor_count, floating_elements
        nonlocal nonfinite_floating_elements
        if isinstance(item, torch_module.Tensor):
            tensor_count += 1
            is_complex = getattr(torch_module, "is_complex", lambda _value: False)
            if torch_module.is_floating_point(item) or is_complex(item):
                floating_tensor_count += 1
                count = int(item.numel())
                floating_elements += count
                finite = int(torch_module.isfinite(item).sum().item())
                nonfinite_floating_elements += count - finite
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)
            for child in item:
                visit(child)

    visit(value)
    return {
        "tensor_count": tensor_count,
        "floating_tensor_count": floating_tensor_count,
        "floating_elements": floating_elements,
        "nonfinite_floating_elements": nonfinite_floating_elements,
    }


def _checkpoint_stat(path: Path) -> tuple[int, int, int, int, int]:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise LeanQueueRuntimeError(f"checkpoint is missing: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise LeanQueueRuntimeError(f"checkpoint must be a regular non-symlink file: {path}")
    if info.st_size <= 0:
        raise LeanQueueRuntimeError(f"checkpoint is empty: {path}")
    return _stat_signature(info)


def attest_milestone(
    binding_path: str | Path,
    milestone: int,
    *,
    checkpoint_loader: Callable[[Path], Any] | None = None,
    torch_module: Any | None = None,
    proc_root: str | Path = "/proc",
    getpgid: Callable[[int], int] = os.getpgid,
) -> dict[str, Any]:
    """Attest exactly one bound milestone and publish one immutable receipt."""

    binding_path_obj = _canonical_absolute_path(str(binding_path), "binding path")
    binding, content, _claim, _claim_content = _load_binding(binding_path_obj)
    milestone_value = _require_plain_int(milestone, "milestone", minimum=1)
    if milestone_value not in content.get("milestones", []):
        raise LeanQueueRuntimeError("requested iteration is not a preregistered milestone")
    process_state = _verify_bound_process(
        content, proc_root=Path(proc_root), getpgid=getpgid
    )
    log_dir = _canonical_absolute_path(content.get("rsl_log_dir"), "bound RSL log dir")
    checkpoint_path = log_dir / f"model_{milestone_value}.pt"
    before = _checkpoint_stat(checkpoint_path)
    if checkpoint_loader is None:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        checkpoint = torch_module.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    else:
        checkpoint = checkpoint_loader(checkpoint_path)
        if torch_module is None:
            raise LeanQueueRuntimeError("an injected checkpoint loader requires torch_module")
    after_load = _checkpoint_stat(checkpoint_path)
    if after_load != before:
        raise LeanQueueRuntimeError("checkpoint changed while loading")
    checkpoint = _require_mapping(checkpoint, "checkpoint")
    embedded_iteration = _require_plain_int(
        checkpoint.get("iter"), "checkpoint embedded iteration", minimum=1
    )
    if embedded_iteration != milestone_value:
        raise LeanQueueRuntimeError("checkpoint filename iteration differs from embedded iteration")
    infos = _require_mapping(checkpoint.get("infos"), "checkpoint infos")
    hard_path = log_dir / "params/training_contract.json"
    hard_contract, hard_bytes = _read_regular_json(hard_path, "hard training contract")
    hard_schema = _require_plain_int(
        hard_contract.get("schema_version"), "hard contract schema", minimum=1
    )
    if hard_schema != TRAINING_CONTRACT_SCHEMA_VERSION:
        raise LeanQueueRuntimeError("hard contract is not schema 3")
    hard_sha = _sha256_bytes(hard_bytes)
    if infos.get("training_contract_schema_version") != hard_schema:
        raise LeanQueueRuntimeError("checkpoint hard-contract schema binding mismatch")
    if infos.get("training_contract_sha256") != hard_sha:
        raise LeanQueueRuntimeError("checkpoint hard-contract SHA binding mismatch")
    lineage = infos.get("training_contract_lineage_exact")
    if type(lineage) is not int or lineage not in (0, 1):
        raise LeanQueueRuntimeError("checkpoint contract lineage must be exactly 0 or 1")
    claim_digest = _require_sha256(
        content.get("claim_content_sha256"), "bound claim digest"
    )
    if infos.get("training_launch_claim_sha256") != claim_digest:
        raise LeanQueueRuntimeError("checkpoint launch-claim lineage mismatch")
    tensor_audit = _tensor_finiteness(checkpoint, torch_module)
    if tensor_audit["floating_tensor_count"] <= 0:
        raise LeanQueueRuntimeError("checkpoint contains no floating tensors")
    if tensor_audit["nonfinite_floating_elements"] != 0:
        raise LeanQueueRuntimeError("checkpoint contains non-finite floating tensors")
    checkpoint_bytes = _read_regular_bytes(checkpoint_path, "checkpoint")
    after_hash = _checkpoint_stat(checkpoint_path)
    if after_hash != before:
        raise LeanQueueRuntimeError("checkpoint changed while hashing")
    if _read_regular_bytes(hard_path, "hard training contract") != hard_bytes:
        raise LeanQueueRuntimeError("hard training contract changed during attestation")

    binding_digest = _require_sha256(binding.get("content_sha256"), "binding digest")
    receipt_content = {
        "schema_version": 1,
        "job_id": content["job_id"],
        "binding_path": str(binding_path_obj),
        "binding_content_sha256": binding_digest,
        "claim_content_sha256": claim_digest,
        "milestone": milestone_value,
        "process_state_at_attestation": process_state,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_bytes(checkpoint_bytes),
            "filename_iteration": milestone_value,
            "embedded_iteration": embedded_iteration,
            **tensor_audit,
        },
        "hard_contract": {
            "path": str(hard_path),
            "schema_version": hard_schema,
            "sha256": hard_sha,
            "lineage_exact": lineage,
        },
    }
    receipt = {
        "schema_version": 1,
        "content": receipt_content,
        "content_sha256": canonical_sha256(receipt_content),
    }
    receipt_path = (
        _canonical_absolute_path(content.get("run_dir"), "bound run_dir")
        / MILESTONE_DIR_NAME
        / f"model_{milestone_value}.json"
    )
    _atomic_publish_json(receipt_path, receipt, "milestone receipt")
    return {"receipt_path": str(receipt_path), "receipt": receipt}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    attest = sub.add_parser("attest")
    attest.add_argument("--binding", type=Path, required=True)
    attest.add_argument("--milestone", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command != "attest":
            raise LeanQueueRuntimeError(f"unsupported command: {args.command}")
        result = attest_milestone(args.binding, args.milestone)
    except LeanQueueRuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
