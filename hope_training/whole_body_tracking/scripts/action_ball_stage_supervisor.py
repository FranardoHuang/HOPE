#!/usr/bin/env python3
"""Supervise one exact two-GPU ActionBall training stage.

The schema-v3 launcher owns planning, GPU admission, and namespace creation.
This process owns the lifetime of the two children after that point:

* inherit and retain both already-acquired GPU-wide lock descriptors;
* serialize the two Isaac/Kit boot windows with ``/workspace/.kit_boot.lock``;
* start the formal frozen evaluator on physical GPU 1 and wait for its exact
  readiness receipt before starting the trainer on physical GPU 0;
* bind each child to an independent ``setsid`` process group plus Linux
  ``/proc`` start-time identity;
* terminate only a process group whose exact identity was captured by
  ``exact_process_group.py``; and
* publish no-clobber ready, failure, and terminal receipts.

There is deliberately no CPU-fake mode, no GPU fallback, no arbitrary argv
seam, and no PID-only signalling path.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import datetime as _datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple


SCHEMA_VERSION = 3
CLAIM_KIND = "action_ball_no_clobber_launch_claim_v3"
PAYLOAD_KIND = "action_ball_no_clobber_launch_payload_v3"
LAUNCH_PROFILE = "fresh_upper_nomove_n5_v3"  # legacy test compatibility only
SUPERVISOR_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_stage_supervisor.py"
)
EXACT_PROCESS_GROUP_SOURCE = (
    "hope_training/whole_body_tracking/scripts/exact_process_group.py"
)
SETUP_SOURCE = "hope_training/whole_body_tracking/setup_train_env.sh"
SIDECAR_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_frozen_eval_sidecar.py"
)
EXACT_RESUME_VERIFIER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_exact_resume_verifier.py"
)
ACTION_BALL_TASK_ID = "HOPE-PingPong-ActionBall-AgibotA3-v0"
ACTION_BALL_EXPERIMENT_NAME = "agibot_a3_hope_action_ball_fresh_n5"
ACTION_SET_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_action_set_contract.py"
)
NOSITE_BOOTSTRAP_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_python_nosite_bootstrap.py"
)
FIXED_BOOT_LOCK = Path("/workspace/.kit_boot.lock")
FIXED_GPU_LOCKS = {
    "trainer": Path("/tmp/hope_lean_queue_gpu0.lock"),
    "evaluator": Path("/tmp/hope_lean_queue_gpu1.lock"),
}
ROLE_GPU_INDEX = {"trainer": 0, "evaluator": 1, "verifier": 0}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CHECKPOINT_RE = re.compile(r"^model_([0-9]+)\.pt$")
RSL_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
READY_PREFIX = b"ACTION_BALL_SIDECAR_READY "
LEARNING_MARKER_RE = re.compile(
    rb"Learning iteration[ \t]+[0-9]+/[0-9]+"
)
STAGES = frozenset({"smoke", "canary", "long"})
HEARTBEAT_KIND = "whole_body_tracking.action_ball.formal_sidecar_heartbeat"
HEARTBEAT_CONTENT_KEYS = frozenset(
    {
        "owner_id",
        "run_id",
        "pid",
        "sidecar_code_sha256",
        "launch_sha256",
        "backend_contract_sha256",
        "heartbeat_seq",
        "phase",
        "request_seq",
        "request_sha256",
        "attempts_completed",
        "attempts_total",
        "request_started_unix_ns",
        "request_started_monotonic_ns",
        "request_deadline_unix_ns",
        "request_deadline_monotonic_ns",
        "heartbeat_unix_ns",
        "heartbeat_monotonic_ns",
        "error_type",
    }
)
HEARTBEAT_IDLE_PHASES = frozenset(
    {"starting", "ready", "waiting_for_request_or_ack", "stopping"}
)
HEARTBEAT_ACTIVE_PHASES = frozenset(
    {
        "request_accepted",
        "runtime_building",
        "evaluating",
        "validating_evidence",
        "evidence_published",
        "request_failed",
    }
)
SAFE_ERROR_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
SAFE_INBOX_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_SANITIZED_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "TZ",
        "USER",
    }
)


class SupervisorError(RuntimeError):
    """A fail-closed refusal or supervised-stage failure."""


class StopRequest:
    """Signal handler state; handlers never signal children themselves."""

    def __init__(self) -> None:
        self.signum: Optional[int] = None

    def handler(self, signum: int, _frame: Any) -> None:
        if self.signum is None:
            self.signum = signum

    def check(self) -> None:
        if self.signum is not None:
            try:
                name = signal.Signals(self.signum).name
            except ValueError:
                name = str(self.signum)
            raise SupervisorError(
                f"supervisor stop requested by signal {name}"
            )


@dataclass(frozen=True)
class Timing:
    """Pre-registered lifecycle limits; CLI callers cannot override these."""

    poll_seconds: float = 0.10
    boot_lock_timeout_seconds: float = 900.0
    sidecar_ready_timeout_seconds: float = 900.0
    trainer_ready_timeout_seconds: float = 1200.0
    launcher_accept_timeout_seconds: float = 300.0
    publication_grace_seconds: float = 1.0
    exact_resume_timeout_seconds: float = 3600.0
    term_grace_seconds: float = 30.0
    kill_grace_seconds: float = 10.0


@dataclass(frozen=True)
class RuntimePaths:
    """Production constants, injectable only by host-only unit tests."""

    boot_lock: Path = FIXED_BOOT_LOCK
    trainer_lock: Path = FIXED_GPU_LOCKS["trainer"]
    evaluator_lock: Path = FIXED_GPU_LOCKS["evaluator"]
    proc_root: Path = Path("/proc")


@dataclass(frozen=True)
class ValidatedClaim:
    claim_path: Path
    claim_sha256: str
    namespace: Path
    checkout: Path
    source_commit: str
    stage: str
    action_set_contract: Mapping[str, Any]
    trainer_argv: Tuple[str, ...]
    sidecar_argv: Tuple[str, ...]
    runtime_code_sha256: Mapping[str, str]
    gpus: Mapping[str, Mapping[str, Any]]
    setup_path: Path
    exact_process_group_path: Path
    exact_resume_verifier_path: Path
    nosite_bootstrap_path: Path
    nosite_import_roots: Tuple[Mapping[str, Any], ...]
    max_iterations: int
    expected_sidecar_ready: Mapping[str, Any]
    heartbeat_path: Path
    heartbeat_contract: Mapping[str, Any]
    sidecar_code_sha256: str
    sidecar_launch_content_sha256: str
    sidecar_backend_contract_sha256: str


@dataclass(frozen=True)
class HeartbeatObservation:
    document: Mapping[str, Any]
    content_sha256: str
    heartbeat_seq: int
    heartbeat_monotonic_ns: int
    phase: str
    request_seq: Optional[int]
    request_sha256: str
    attempts_completed: int
    attempts_total: int
    request_started_unix_ns: int
    request_started_monotonic_ns: int
    request_deadline_unix_ns: int
    request_deadline_monotonic_ns: int
    max_request_seq: int


@dataclass
class Child:
    role: str
    process: subprocess.Popen
    log: "LogCapture"
    leader_receipt: Path
    identity: Any
    argv_sha256: str
    term_receipt: Optional[Path] = None
    kill_receipt: Optional[Path] = None


def _utc_now() -> str:
    return (
        _datetime.datetime.now(tz=_datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SupervisorError("value is not canonical-JSON encodable") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SupervisorError(f"expected a regular file: {path}")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    ):
        raise SupervisorError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _snapshot_regular_file(
    path: Path, *, label: str, max_bytes: int
) -> Mapping[str, Any]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise SupervisorError(f"{label} has an invalid size limit")
    try:
        before = path.lstat()
    except OSError as exc:
        raise SupervisorError(f"cannot inspect {label}: {path}: {exc}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 1
        or before.st_size > max_bytes
    ):
        raise SupervisorError(
            f"{label} must be a nonempty single-link regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        raw = bytearray()
        while True:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > max_bytes:
                raise SupervisorError(f"{label} exceeds its size limit")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        final = path.lstat()
    except OSError as exc:
        raise SupervisorError(f"{label} vanished while reading") from exc
    identities = [
        (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
        )
        for item in (before, opened, after, final)
    ]
    if len(set(identities)) != 1 or len(raw) != before.st_size:
        raise SupervisorError(f"{label} changed while reading")
    return {
        "raw": bytes(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": before.st_size,
    }


def _strict_json_pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SupervisorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_strict_json_with_raw(
    path: Path, *, label: str, max_bytes: int = 16 * 1024 * 1024
) -> Tuple[Dict[str, Any], bytes]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise SupervisorError(f"{label} has an invalid read-size limit")
    try:
        before = path.lstat()
    except OSError as exc:
        raise SupervisorError(f"cannot inspect {label}: {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SupervisorError(
            f"{label} must be a single-link regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise SupervisorError(
                f"{label} must be a single-link regular non-symlink file"
            )
        raw = bytearray()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > max_bytes:
                raise SupervisorError(
                    f"{label} exceeds its {max_bytes}-byte size limit"
                )
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        final = path.lstat()
    except OSError as exc:
        raise SupervisorError(f"{label} vanished while reading") from exc
    identity = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_nlink,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    ) or identity != (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_nlink,
    ):
        raise SupervisorError(f"{label} changed while reading")
    try:
        value = json.loads(
            bytes(raw).decode("utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SupervisorError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise SupervisorError(f"{label} root must be an object")
    return value, bytes(raw)


def _read_strict_json(path: Path, *, label: str) -> Dict[str, Any]:
    return _read_strict_json_with_raw(path, label=label)[0]


def _read_inflight_published_json(
    path: Path,
    *,
    label: str,
    timing: Timing,
    guard: Callable[[], None],
) -> Dict[str, Any]:
    """Bound the create-before-write window of an O_EXCL publisher.

    The no-clobber publisher owns the final pathname from ``open(O_EXCL)``
    until its fsync completes, so mere pathname existence is not a completion
    signal.  Retry only a real single-link regular file for the preregistered
    publication grace; symlinks/non-regular files still fail immediately.
    """

    deadline = time.monotonic() + timing.publication_grace_seconds
    last_error: Optional[SupervisorError] = None
    while True:
        guard()
        try:
            return _read_strict_json(path, label=label)
        except SupervisorError as exc:
            last_error = exc
            try:
                info = path.lstat()
            except OSError:
                info = None
            if info is not None and (
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            ):
                raise
            if time.monotonic() >= deadline:
                raise last_error
            time.sleep(timing.poll_seconds)


def _exact_keys(value: Any, keys: Iterable[str], *, label: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise SupervisorError(f"{label} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise SupervisorError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise SupervisorError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_string(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise SupervisorError(f"{label} must be a non-empty trimmed string")
    return value


def _absolute_path(value: Any, *, label: str, must_exist: bool) -> Path:
    text = _plain_string(value, label=label)
    path = Path(text)
    if not path.is_absolute() or path != Path(os.path.normpath(text)):
        raise SupervisorError(f"{label} must be an absolute normalized path")
    if must_exist:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise SupervisorError(f"{label} cannot be resolved: {exc}") from exc
        if resolved != path:
            raise SupervisorError(f"{label} must not traverse symlinks")
    return path


def _argv(value: Any, *, label: str) -> Tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item or "\x00" in item for item in value)
    ):
        raise SupervisorError(f"{label} must be a non-empty string array")
    return tuple(value)


def _decode_nosite_argv(
    argv: Tuple[str, ...], *, label: str
) -> Dict[str, Any]:
    """Decode the stdlib-only envelope before committed code is imported."""

    if len(argv) != 10 or argv[1:5] != ("-I", "-B", "-S", "-c"):
        raise SupervisorError(
            f"{label} is not one exact 10-token no-site argv"
        )
    contract_sha = _sha(argv[8], label=f"{label} contract SHA-256")
    try:
        raw = base64.b64decode(argv[9].encode("ascii"), validate=True)
        contract = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_json_pairs
        )
    except (UnicodeEncodeError, UnicodeDecodeError, ValueError) as exc:
        raise SupervisorError(
            f"{label} contract is not strict canonical base64/JSON"
        ) from exc
    if (
        base64.b64encode(raw).decode("ascii") != argv[9]
        or hashlib.sha256(raw).hexdigest() != contract_sha
        or _canonical_bytes(contract) != raw
    ):
        raise SupervisorError(
            f"{label} contract base64/SHA/canonical bytes differ"
        )
    row = _exact_keys(
        contract,
        (
            "schema_version",
            "kind",
            "bootstrap",
            "entrypoint",
            "import_roots",
            "entrypoint_argv",
        ),
        label=f"{label} contract",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != "action_ball_python_nosite_argv_contract_v1"
        or type(row["entrypoint_argv"]) is not list
        or any(
            type(item) is not str or not item or "\x00" in item
            for item in row["entrypoint_argv"]
        )
    ):
        raise SupervisorError(f"{label} contract schema/arguments are invalid")
    return row


def _flag_value(argv: Tuple[str, ...], flag: str) -> str:
    positions = [i for i, token in enumerate(argv) if token == flag]
    if len(positions) != 1:
        raise SupervisorError(f"sidecar argv must contain exactly one {flag}")
    index = positions[0]
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise SupervisorError(f"sidecar argv {flag} lacks a value")
    return argv[index + 1]


def _hydra_overrides(argv: Tuple[str, ...]) -> Dict[str, str]:
    if "--" not in argv:
        raise SupervisorError("trainer argv lacks the isolated-entrypoint -- boundary")
    boundary = argv.index("--")
    result: Dict[str, str] = {}
    pattern = re.compile(
        r"^(?:\+\+|\+)?([A-Za-z_][A-Za-z0-9_.-]*)=(.*)$"
    )
    for token in argv[boundary + 1 :]:
        if token.startswith("~"):
            raise SupervisorError(
                "trainer Hydra deletion overrides are forbidden"
            )
        match = pattern.fullmatch(token)
        if match is None:
            raise SupervisorError(
                f"malformed trainer Hydra override: {token!r}"
            )
        key, value = match.group(1), match.group(2)
        if key in result:
            raise SupervisorError(f"duplicate trainer override: {key}")
        result[key] = value
    return result


def _validate_gpu_roles(
    value: Any, *, runtime_paths: RuntimePaths
) -> Mapping[str, Mapping[str, Any]]:
    roles = _exact_keys(value, ("trainer", "evaluator"), label="claim gpus")
    normalized: Dict[str, Mapping[str, Any]] = {}
    for role in ("trainer", "evaluator"):
        row = _exact_keys(
            roles[role],
            (
                "index",
                "uuid",
                "owner",
                "lock_path",
                "boot_lock_path",
                "require_empty",
                "owner_receipt_sha256",
            ),
            label=f"claim gpus.{role}",
        )
        index = row["index"]
        if type(index) is not int or index != ROLE_GPU_INDEX[role]:
            raise SupervisorError(
                f"{role} must use physical GPU {ROLE_GPU_INDEX[role]}"
            )
        uuid = _plain_string(row["uuid"], label=f"{role} GPU UUID")
        if not uuid.startswith("GPU-"):
            raise SupervisorError(f"{role} GPU UUID must start with GPU-")
        _plain_string(row["owner"], label=f"{role} GPU owner")
        expected_lock = (
            runtime_paths.trainer_lock
            if role == "trainer"
            else runtime_paths.evaluator_lock
        )
        lock_path = _absolute_path(
            row["lock_path"], label=f"{role} lock path", must_exist=False
        )
        if lock_path != expected_lock:
            raise SupervisorError(
                f"{role} lock path must be {expected_lock}, got {lock_path}"
            )
        boot_path = _absolute_path(
            row["boot_lock_path"],
            label=f"{role} boot lock path",
            must_exist=False,
        )
        if boot_path != runtime_paths.boot_lock:
            raise SupervisorError(
                f"{role} boot lock must be {runtime_paths.boot_lock}"
            )
        if row["require_empty"] is not True:
            raise SupervisorError(f"{role} require_empty must be true")
        _sha(
            row["owner_receipt_sha256"],
            label=f"{role} owner receipt SHA-256",
        )
        normalized[role] = dict(row)
    if (
        normalized["trainer"]["uuid"] == normalized["evaluator"]["uuid"]
        or normalized["trainer"]["lock_path"]
        == normalized["evaluator"]["lock_path"]
    ):
        raise SupervisorError("trainer and evaluator must use distinct GPUs/locks")
    return normalized


def _validate_claim_document(
    claim_path: Path,
    expected_claim_sha256: str,
    *,
    runtime_paths: RuntimePaths,
) -> Tuple[Dict[str, Any], Dict[str, Any], Mapping[str, Mapping[str, Any]]]:
    expected_sha = _sha(expected_claim_sha256, label="CLI claim SHA-256")
    claim = _exact_keys(
        _read_strict_json(claim_path, label="launch claim"),
        (
            "schema_version",
            "kind",
            "launch_claim_sha256",
            "canonical_payload",
            "argv",
            "confirmation_claim_sha256",
        ),
        label="launch claim",
    )
    if claim["schema_version"] != SCHEMA_VERSION or claim["kind"] != CLAIM_KIND:
        raise SupervisorError("launch claim is not exact schema v3")
    if (
        claim["launch_claim_sha256"] != expected_sha
        or claim["confirmation_claim_sha256"] != expected_sha
    ):
        raise SupervisorError("launch claim SHA fields differ from CLI pin")
    payload = claim["canonical_payload"]
    if type(payload) is not dict:
        raise SupervisorError("canonical_payload must be an object")
    if canonical_sha256(payload) != expected_sha:
        raise SupervisorError("canonical_payload SHA differs from launch claim")
    required = {
        "schema_version",
        "kind",
        "launch_profile",
        "action_set_contract",
        "ordered_action_ids",
        "manifest",
        "stage",
        "source_checkout",
        "source_commit_sha",
        "runtime_code_sha256",
        "namespace",
        "gpus",
        "argv_without_launch_claim",
        "sidecar_argv",
        "sidecar_launch_receipt",
        "frozen_evaluation_runtime",
        "isaac_python_runtime",
        "stage_budget",
    }
    missing = required - set(payload)
    if missing:
        raise SupervisorError(
            f"canonical_payload lacks supervisor fields: {sorted(missing)}"
        )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["kind"] != PAYLOAD_KIND
    ):
        raise SupervisorError("canonical_payload identity is not launch v3")
    if payload["stage"] not in STAGES:
        raise SupervisorError("canonical_payload stage is invalid")
    gpus = _validate_gpu_roles(payload["gpus"], runtime_paths=runtime_paths)

    trainer_argv = _argv(claim["argv"], label="trainer argv")
    argv_without_claim = _argv(
        payload["argv_without_launch_claim"],
        label="argv_without_launch_claim",
    )
    trainer_contract = _decode_nosite_argv(
        trainer_argv, label="trainer argv"
    )
    base_contract = _decode_nosite_argv(
        argv_without_claim, label="argv_without_launch_claim"
    )
    trainer_args = tuple(trainer_contract["entrypoint_argv"])
    base_args = tuple(base_contract["entrypoint_argv"])
    namespace = _absolute_path(
        payload["namespace"], label="claim namespace", must_exist=True
    )
    action_set_contract = payload["action_set_contract"]
    if type(action_set_contract) is not dict:
        raise SupervisorError("action_set_contract must be an object")
    if payload["launch_profile"] != action_set_contract.get("profile_id"):
        raise SupervisorError(
            "launch_profile differs from action_set_contract profile"
        )
    namespace_identity = action_set_contract.get("namespace_identity")
    if (
        type(namespace_identity) is not str
        or namespace_identity not in namespace.name
    ):
        raise SupervisorError(
            "namespace lacks the contracted N/order identity"
        )
    expected_claim_path = namespace / "launch_claim.json"
    if claim_path != expected_claim_path:
        raise SupervisorError(
            "claim path must be the exact namespace/launch_claim.json"
        )
    expected_path_override = (
        f"++training_launch_claim_path={expected_claim_path}"
    )
    expected_sha_override = (
        f"++training_launch_claim_sha256={expected_sha}"
    )
    path_bindings: List[Tuple[int, str]] = []
    sha_bindings: List[Tuple[int, str]] = []
    for index, token in enumerate(trainer_args):
        if "=" not in token:
            continue
        key = token.split("=", 1)[0].lstrip("+")
        if key == "training_launch_claim_path":
            path_bindings.append((index, token))
        elif key == "training_launch_claim_sha256":
            sha_bindings.append((index, token))
    if (
        {
            key: value
            for key, value in trainer_contract.items()
            if key != "entrypoint_argv"
        }
        != {
            key: value
            for key, value in base_contract.items()
            if key != "entrypoint_argv"
        }
        or trainer_args != (*base_args, expected_sha_override)
        or len(path_bindings) != 1
        or path_bindings[0][1] != expected_path_override
        or sha_bindings
        != [(len(trainer_args) - 1, expected_sha_override)]
        or path_bindings[0][0] >= sha_bindings[0][0]
    ):
        raise SupervisorError("trainer argv is not exactly bound to this claim")
    sidecar_argv = _argv(payload["sidecar_argv"], label="sidecar argv")
    sidecar_contract = _decode_nosite_argv(
        sidecar_argv, label="sidecar argv"
    )
    sidecar_args = tuple(sidecar_contract["entrypoint_argv"])
    if "cpu-fake" in sidecar_args or "--once" in sidecar_args:
        raise SupervisorError("production supervisor refuses CPU-fake/one-shot sidecar")
    if _flag_value(sidecar_args, "--backend") != "formal":
        raise SupervisorError("sidecar backend must be formal")
    if _flag_value(sidecar_args, "--device") != "cuda:0":
        raise SupervisorError("sidecar must target logical cuda:0")

    evaluation = payload["frozen_evaluation_runtime"]
    if type(evaluation) is not dict:
        raise SupervisorError("frozen_evaluation_runtime must be an object")
    owner_id = _plain_string(
        evaluation.get("owner_id"), label="evaluation owner_id"
    )
    run_id = _plain_string(evaluation.get("run_id"), label="evaluation run_id")
    if (
        SAFE_INBOX_ID_RE.fullmatch(owner_id) is None
        or SAFE_INBOX_ID_RE.fullmatch(run_id) is None
    ):
        raise SupervisorError("evaluation owner_id/run_id is not path-safe")
    inbox_root = _absolute_path(
        evaluation.get("inbox_root"),
        label="evaluation inbox_root",
        must_exist=False,
    )
    interval = evaluation.get("interval_updates")
    if type(interval) is not int or interval <= 0:
        raise SupervisorError("evaluation interval_updates must be positive")
    if _flag_value(sidecar_args, "--owner-id") != owner_id:
        raise SupervisorError("sidecar owner-id differs from frozen runtime")
    if _flag_value(sidecar_args, "--run-id") != run_id:
        raise SupervisorError("sidecar run-id differs from frozen runtime")
    if Path(_flag_value(sidecar_args, "--inbox-root")) != inbox_root:
        raise SupervisorError("sidecar inbox-root differs from frozen runtime")

    sidecar_receipt = payload["sidecar_launch_receipt"]
    if type(sidecar_receipt) is not dict:
        raise SupervisorError("sidecar_launch_receipt must be an object")
    launch_content_sha = _sha(
        sidecar_receipt.get("content_sha256"),
        label="sidecar launch receipt content SHA-256",
    )
    _sha(
        sidecar_receipt.get("sidecar_code_sha256"),
        label="sidecar code SHA-256",
    )
    _sha(
        sidecar_receipt.get("backend_contract_sha256"),
        label="sidecar backend contract SHA-256",
    )
    heartbeat_contract = _exact_keys(
        sidecar_receipt.get("heartbeat_contract"),
        (
            "schema_version",
            "heartbeat_interval_seconds",
            "heartbeat_stale_after_seconds",
            "request_deadline_seconds",
        ),
        label="sidecar heartbeat contract",
    )
    expected_heartbeat_contract = {
        "schema_version": 1,
        "heartbeat_interval_seconds": 5.0,
        "heartbeat_stale_after_seconds": 120.0,
        "request_deadline_seconds": 7200.0,
    }
    if (
        type(heartbeat_contract["schema_version"]) is not int
        or any(
            type(heartbeat_contract[field]) is not float
            for field in (
                "heartbeat_interval_seconds",
                "heartbeat_stale_after_seconds",
                "request_deadline_seconds",
            )
        )
        or heartbeat_contract != expected_heartbeat_contract
    ):
        raise SupervisorError(
            "sidecar heartbeat contract is not exact 5s/120s/7200s"
        )
    isaac_runtime = payload["isaac_python_runtime"]
    if type(isaac_runtime) is not dict:
        raise SupervisorError("isaac_python_runtime must be an object")
    isaac_python = isaac_runtime.get("path")
    if (
        type(isaac_python) is not str
        or not isaac_python
        or not Path(isaac_python).is_absolute()
        or Path(isaac_python) != Path(os.path.normpath(isaac_python))
        or trainer_argv[0] != isaac_python
        or sidecar_argv[0] != isaac_python
    ):
        raise SupervisorError(
            "trainer and sidecar must use the exact claimed Isaac Python"
        )
    stage_budget = payload["stage_budget"]
    max_iterations = (
        stage_budget.get("max_iterations")
        if type(stage_budget) is dict
        else None
    )
    if type(max_iterations) is not int or max_iterations < 1:
        raise SupervisorError("stage_budget.max_iterations must be positive")
    overrides = _hydra_overrides(trainer_args)
    expected_overrides = {
        "device": "cuda:0",
        "task.experiment_name": action_set_contract.get("experiment_name"),
        "task.actor_obs_contract": action_set_contract.get(
            "actor_obs_contract"
        ),
        "task.racket.action_ball_manifest_path": action_set_contract.get(
            "manifest_path"
        ),
        "task.racket.action_ball_manifest_sha256": action_set_contract.get(
            "manifest_sha256"
        ),
        "task.racket.action_ball_evaluation_inbox_root": str(inbox_root),
        "task.racket.action_ball_evaluation_owner_id": owner_id,
        "task.racket.action_ball_evaluation_run_id": run_id,
        "task.racket.action_ball_frozen_eval_interval_updates": str(interval),
    }
    for key, expected in expected_overrides.items():
        if type(expected) is not str or overrides.get(key) != expected:
            raise SupervisorError(
                f"trainer override {key} differs from frozen runtime"
            )
    try:
        clip_names = json.loads(overrides["task.racket.clip_names"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SupervisorError("trainer clip_names is not strict JSON") from exc
    if (
        clip_names != action_set_contract.get("ordered_action_ids")
        or payload["ordered_action_ids"] != clip_names
        or type(payload["manifest"]) is not dict
        or payload["manifest"].get("path")
        != action_set_contract.get("manifest_path")
        or payload["manifest"].get("sha256")
        != action_set_contract.get("manifest_sha256")
    ):
        raise SupervisorError(
            "trainer/claim manifest or ordered action identity drifted"
        )
    expected_ready = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.formal_sidecar_ready",
        "owner_id": owner_id,
        "run_id": run_id,
        "backend": "formal",
        "device": "cuda:0",
        "launch_receipt_canonical_sha256": launch_content_sha,
    }
    return claim, payload, gpus


def _git(
    checkout: Path, args: List[str], *, binary: bool = False
) -> Any:
    git_executable = shutil.which("git", path=os.defpath)
    if git_executable is None or not os.path.isabs(git_executable):
        raise SupervisorError("git is absent from the trusted system path")
    completed = subprocess.run(
        [git_executable, "-C", str(checkout), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
        env={
            "PATH": os.defpath,
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    if completed.returncode != 0:
        stderr = (
            completed.stderr.decode("utf-8", "replace")
            if binary
            else completed.stderr
        )
        raise SupervisorError(
            f"git {' '.join(args)} failed: {stderr[-2000:]}"
        )
    return completed.stdout


def _verify_committed_file(
    checkout: Path,
    source_commit: str,
    relative: str,
    expected_sha256: str,
) -> Path:
    if (
        not relative
        or Path(relative).is_absolute()
        or Path(relative).as_posix() != relative
        or any(part in ("", ".", "..") for part in Path(relative).parts)
    ):
        raise SupervisorError(f"invalid committed relative path: {relative}")
    raw = _git(
        checkout,
        ["ls-tree", "-z", source_commit, "--", relative],
        binary=True,
    )
    records = [item for item in raw.split(b"\0") if item]
    if len(records) != 1:
        raise SupervisorError(
            f"{relative} must exist exactly once in source commit"
        )
    try:
        metadata, listed_path = records[0].split(b"\t", 1)
        mode, object_type, _object_id = metadata.split(b" ", 2)
    except ValueError as exc:
        raise SupervisorError(f"malformed git ls-tree row for {relative}") from exc
    if listed_path.decode("utf-8") != relative or object_type != b"blob":
        raise SupervisorError(f"{relative} is not an exact committed blob")
    if mode not in (b"100644", b"100755"):
        raise SupervisorError(f"{relative} has unsupported git mode")
    committed = _git(
        checkout,
        ["cat-file", "blob", f"{source_commit}:{relative}"],
        binary=True,
    )
    actual_committed_sha = hashlib.sha256(committed).hexdigest()
    if actual_committed_sha != _sha(
        expected_sha256, label=f"{relative} runtime pin"
    ):
        raise SupervisorError(f"{relative} committed bytes differ from claim pin")
    path = checkout / relative
    try:
        mode_now = path.lstat().st_mode
    except OSError as exc:
        raise SupervisorError(f"cannot inspect runtime source {relative}") from exc
    if not stat.S_ISREG(mode_now):
        raise SupervisorError(f"runtime source is not a regular file: {relative}")
    if path.resolve(strict=True) != path:
        raise SupervisorError(f"runtime source traverses a symlink: {relative}")
    if _sha256_file(path) != actual_committed_sha:
        raise SupervisorError(f"working-tree bytes differ from commit: {relative}")
    return path


def _verify_source_clean(checkout: Path, source_commit: str) -> None:
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise SupervisorError("source_commit_sha must be a lowercase Git SHA-1")
    root = _git(checkout, ["rev-parse", "--show-toplevel"]).strip()
    head = _git(checkout, ["rev-parse", "--verify", "HEAD"]).strip()
    if root != str(checkout) or head != source_commit:
        raise SupervisorError("checkout root/HEAD differs from claim")
    status = _git(
        checkout,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        binary=True,
    )
    if status:
        preview = status[:1000].decode("utf-8", "replace").replace("\0", " | ")
        raise SupervisorError(f"source checkout is not exact-clean: {preview}")


def _load_exact_process_group(path: Path) -> Any:
    name = "_action_ball_exact_process_group"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SupervisorError("cannot load exact_process_group.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_action_set_contract_module(path: Path) -> Any:
    name = "_action_ball_exact_action_set_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SupervisorError("cannot load exact action-set contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_nosite_bootstrap_module(path: Path) -> Any:
    name = "_action_ball_exact_nosite_bootstrap"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SupervisorError("cannot load exact no-site bootstrap module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SupervisorError(
            "cannot execute exact no-site bootstrap module"
        ) from exc
    return module


def validate_claim_and_source(
    claim_path_value: str,
    expected_claim_sha256: str,
    *,
    runtime_paths: RuntimePaths = RuntimePaths(),
    self_path: Optional[Path] = None,
) -> Tuple[ValidatedClaim, Any]:
    claim_path = _absolute_path(
        claim_path_value, label="claim path", must_exist=True
    )
    claim, payload, gpus = _validate_claim_document(
        claim_path,
        expected_claim_sha256,
        runtime_paths=runtime_paths,
    )
    namespace = _absolute_path(
        payload["namespace"], label="claim namespace", must_exist=True
    )
    if claim_path != namespace / "launch_claim.json":
        raise SupervisorError("claim path must be namespace/launch_claim.json")
    checkout = _absolute_path(
        payload["source_checkout"],
        label="source checkout",
        must_exist=True,
    )
    source_commit = _plain_string(
        payload["source_commit_sha"], label="source commit"
    )
    runtime_pins = payload["runtime_code_sha256"]
    if type(runtime_pins) is not dict:
        raise SupervisorError("runtime_code_sha256 must be an object")
    for required in (
        SUPERVISOR_SOURCE,
        ACTION_SET_CONTRACT_SOURCE,
        NOSITE_BOOTSTRAP_SOURCE,
        EXACT_PROCESS_GROUP_SOURCE,
        EXACT_RESUME_VERIFIER_SOURCE,
        SIDECAR_SOURCE,
        SETUP_SOURCE,
    ):
        if required not in runtime_pins:
            raise SupervisorError(f"claim lacks runtime pin for {required}")
    _verify_source_clean(checkout, source_commit)
    supervisor_path = _verify_committed_file(
        checkout,
        source_commit,
        SUPERVISOR_SOURCE,
        runtime_pins[SUPERVISOR_SOURCE],
    )
    actual_self = (Path(__file__) if self_path is None else self_path).resolve(
        strict=True
    )
    if actual_self != supervisor_path:
        raise SupervisorError("executed supervisor is not the claim-pinned source")
    exact_path = _verify_committed_file(
        checkout,
        source_commit,
        EXACT_PROCESS_GROUP_SOURCE,
        runtime_pins[EXACT_PROCESS_GROUP_SOURCE],
    )
    exact_resume_verifier_path = _verify_committed_file(
        checkout,
        source_commit,
        EXACT_RESUME_VERIFIER_SOURCE,
        runtime_pins[EXACT_RESUME_VERIFIER_SOURCE],
    )
    nosite_bootstrap_path = _verify_committed_file(
        checkout,
        source_commit,
        NOSITE_BOOTSTRAP_SOURCE,
        runtime_pins[NOSITE_BOOTSTRAP_SOURCE],
    )
    action_set_contract_path = _verify_committed_file(
        checkout,
        source_commit,
        ACTION_SET_CONTRACT_SOURCE,
        runtime_pins[ACTION_SET_CONTRACT_SOURCE],
    )
    contract_module = _load_action_set_contract_module(
        action_set_contract_path
    )
    try:
        exact_action_set_contract = contract_module.load_contract_from_source(
            action_set_contract_path.read_bytes(),
            payload["launch_profile"],
        )
    except Exception as exc:
        raise SupervisorError(
            f"code-owned action-set contract validation failed: {exc}"
        ) from exc
    if payload["action_set_contract"] != exact_action_set_contract:
        raise SupervisorError(
            "claim action_set_contract differs from committed registry"
        )
    setup_path = _verify_committed_file(
        checkout,
        source_commit,
        SETUP_SOURCE,
        runtime_pins[SETUP_SOURCE],
    )
    sidecar_pin = payload["sidecar_launch_receipt"].get(
        "sidecar_code_sha256"
    )
    if runtime_pins[SIDECAR_SOURCE] != sidecar_pin:
        raise SupervisorError(
            "claim runtime sidecar pin differs from launch receipt"
        )
    sidecar_path = _verify_committed_file(
        checkout,
        source_commit,
        SIDECAR_SOURCE,
        sidecar_pin,
    )
    sidecar_argv = _argv(payload["sidecar_argv"], label="sidecar argv")
    trainer_argv = _argv(claim["argv"], label="trainer argv")
    base_trainer_argv = _argv(
        payload["argv_without_launch_claim"],
        label="argv_without_launch_claim",
    )
    runtime_inventory = payload["isaac_python_runtime"].get(
        "runtime_inventory"
    )
    import_roots = (
        runtime_inventory.get("import_roots")
        if type(runtime_inventory) is dict
        else None
    )
    if type(import_roots) is not list or not import_roots:
        raise SupervisorError(
            "claim lacks nonempty no-site runtime import roots"
        )
    nosite = _load_nosite_bootstrap_module(nosite_bootstrap_path)
    try:
        base_command = nosite.validate_exact_nosite_argv(
            base_trainer_argv,
            expected_python=Path(trainer_argv[0]),
            expected_import_roots=import_roots,
            verify_live=True,
        )
        final_command = nosite.validate_exact_nosite_argv(
            trainer_argv,
            expected_python=Path(trainer_argv[0]),
            expected_bootstrap=base_command.contract["bootstrap"],
            expected_entrypoint=base_command.contract["entrypoint"],
            expected_import_roots=import_roots,
            expected_entrypoint_argv=[
                *base_command.contract["entrypoint_argv"],
                (
                    "++training_launch_claim_sha256="
                    f"{claim['launch_claim_sha256']}"
                ),
            ],
            verify_live=True,
        )
        sidecar_command = nosite.validate_exact_nosite_argv(
            sidecar_argv,
            expected_python=Path(trainer_argv[0]),
            expected_bootstrap=base_command.contract["bootstrap"],
            expected_entrypoint=nosite.bind_regular_file(sidecar_path),
            expected_import_roots=import_roots,
            verify_live=True,
        )
    except Exception as exc:
        raise SupervisorError(
            f"claim no-site argv validation failed: {exc}"
        ) from exc
    isolated_identity = payload.get("isolated_training_entrypoint")
    sidecar_identity = payload.get("sidecar_nosite_execution")
    if (
        type(isolated_identity) is not dict
        or isolated_identity.get("nosite_argv_contract")
        != base_command.contract
        or isolated_identity.get("nosite_argv_contract_sha256")
        != base_command.contract_sha256
        or type(sidecar_identity) is not dict
        or sidecar_identity
        != {
            "nosite_argv_contract_sha256": (
                sidecar_command.contract_sha256
            ),
            "nosite_argv_contract": dict(sidecar_command.contract),
        }
        or final_command.contract["entrypoint_argv"][-1]
        != (
            "++training_launch_claim_sha256="
            f"{claim['launch_claim_sha256']}"
        )
    ):
        raise SupervisorError(
            "claim no-site identities differ from exact argv contracts"
        )
    evaluation = payload["frozen_evaluation_runtime"]
    sidecar_receipt = payload["sidecar_launch_receipt"]
    expected_ready = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.formal_sidecar_ready",
        "owner_id": evaluation["owner_id"],
        "run_id": evaluation["run_id"],
        "backend": "formal",
        "device": "cuda:0",
        "launch_receipt_canonical_sha256": sidecar_receipt[
            "content_sha256"
        ],
    }
    heartbeat_path = (
        Path(evaluation["inbox_root"])
        / "sidecar_status"
        / evaluation["owner_id"]
        / evaluation["run_id"]
        / "heartbeat.json"
    )
    validated = ValidatedClaim(
        claim_path=claim_path,
        claim_sha256=claim["launch_claim_sha256"],
        namespace=namespace,
        checkout=checkout,
        source_commit=source_commit,
        stage=payload["stage"],
        action_set_contract=dict(exact_action_set_contract),
        trainer_argv=trainer_argv,
        sidecar_argv=sidecar_argv,
        runtime_code_sha256=dict(runtime_pins),
        gpus=gpus,
        setup_path=setup_path,
        exact_process_group_path=exact_path,
        exact_resume_verifier_path=exact_resume_verifier_path,
        nosite_bootstrap_path=nosite_bootstrap_path,
        nosite_import_roots=tuple(dict(row) for row in import_roots),
        max_iterations=payload["stage_budget"]["max_iterations"],
        expected_sidecar_ready=expected_ready,
        heartbeat_path=heartbeat_path,
        heartbeat_contract=dict(sidecar_receipt["heartbeat_contract"]),
        sidecar_code_sha256=sidecar_receipt["sidecar_code_sha256"],
        sidecar_launch_content_sha256=sidecar_receipt["content_sha256"],
        sidecar_backend_contract_sha256=sidecar_receipt[
            "backend_contract_sha256"
        ],
    )
    return validated, _load_exact_process_group(exact_path)


def _publish_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise SupervisorError(f"receipt is not a regular file: {path}")
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


class LogCapture:
    def __init__(self, path: Path):
        self.path = path
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
        )
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.fd = os.open(path, flags, 0o600)
        info = os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(self.fd)
            raise SupervisorError(f"log is not a regular file: {path}")
        self.identity = (info.st_dev, info.st_ino)
        self.offset = 0
        self.buffer = b""

    def lines(self) -> List[bytes]:
        try:
            info = self.path.lstat()
        except OSError as exc:
            raise SupervisorError(f"log path vanished: {self.path}") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != self.identity
        ):
            raise SupervisorError(f"log path identity changed: {self.path}")
        chunks: List[bytes] = []
        while True:
            chunk = os.pread(self.fd, 65536, self.offset)
            if not chunk:
                break
            self.offset += len(chunk)
            chunks.append(chunk)
        self.buffer += b"".join(chunks)
        result: List[bytes] = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            result.append(line)
        return result

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _verify_lock_fd(fd: int, expected_path: Path, *, role: str) -> None:
    if type(fd) is not int or fd < 0:
        raise SupervisorError(f"{role} lock fd must be a non-negative integer")
    try:
        fd_info = os.fstat(fd)
        path_info = expected_path.lstat()
    except OSError as exc:
        raise SupervisorError(f"cannot inspect {role} lock fd/path: {exc}") from exc
    if not stat.S_ISREG(fd_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
        raise SupervisorError(f"{role} lock fd/path must be regular files")
    if (fd_info.st_dev, fd_info.st_ino) != (path_info.st_dev, path_info.st_ino):
        raise SupervisorError(f"{role} lock fd does not bind {expected_path}")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SupervisorError(f"{role} lifetime lock is not owned") from exc


def _open_boot_lock(path: Path) -> int:
    flags = (
        os.O_RDWR
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        fd_info = os.fstat(fd)
        path_info = path.lstat()
        if (
            not stat.S_ISREG(fd_info.st_mode)
            or not stat.S_ISREG(path_info.st_mode)
            or (fd_info.st_dev, fd_info.st_ino)
            != (path_info.st_dev, path_info.st_ino)
        ):
            raise SupervisorError("Kit boot lock must be one stable regular file")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _acquire_boot_lock(
    fd: int,
    *,
    timing: Timing,
    guard: Optional[Callable[[], None]] = None,
) -> None:
    deadline = time.monotonic() + timing.boot_lock_timeout_seconds
    while True:
        if guard is not None:
            guard()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise SupervisorError("timed out waiting for pod Kit boot lock")
            time.sleep(timing.poll_seconds)


def _release_boot_lock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)


def _verify_boot_lock_path_after_flock(fd: int, path: Path) -> None:
    try:
        descriptor_info = os.fstat(fd)
        path_info = path.lstat()
    except OSError as exc:
        raise SupervisorError(
            f"Kit boot lock changed after flock: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(descriptor_info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or descriptor_info.st_nlink != 1
        or path_info.st_nlink != 1
        or (descriptor_info.st_dev, descriptor_info.st_ino)
        != (path_info.st_dev, path_info.st_ino)
    ):
        raise SupervisorError(
            "Kit boot lock pathname identity changed after flock"
        )


def _child_env(role: str) -> Dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _SANITIZED_ENV_ALLOWLIST
    }
    env["CUDA_VISIBLE_DEVICES"] = str(ROLE_GPU_INDEX[role])
    env["KIT_BOOT_LOCK"] = str(FIXED_BOOT_LOCK)
    env["PATH"] = os.defpath
    return env


def _wrapped_command(
    setup_path: Path, argv: Tuple[str, ...], start_gate_fd: int
) -> List[str]:
    # The new setsid leader blocks before sourcing setup or executing the
    # workload.  The parent writes exactly one "G" only after the committed
    # exact-process helper has durably bound PID=PGID plus /proc starttime.
    shell = (
        'gate_fd="$1"; shift; '
        'IFS= read -r -n 1 gate <&"$gate_fd" || exit 125; '
        '[ "$gate" = G ] || exit 125; '
        'eval "exec ${gate_fd}<&-"; '
        'source "$1"; shift; exec "$@"'
    )
    return [
        "/bin/bash",
        "-c",
        shell,
        "action-ball-stage",
        str(start_gate_fd),
        str(setup_path),
        *argv,
    ]


def _start_child(
    *,
    role: str,
    argv: Tuple[str, ...],
    claim: ValidatedClaim,
    lock_fds: Tuple[int, ...],
    exact: Any,
) -> Child:
    log_names = {
        "evaluator": "evaluator.log",
        "trainer": "train.log",
        "verifier": "exact_resume_verifier.log",
    }
    if role not in log_names:
        raise SupervisorError(f"unsupported supervised child role: {role}")
    log_path = claim.namespace / log_names[role]
    log = LogCapture(log_path)
    if hasattr(os, "pipe2"):
        gate_read_fd, gate_write_fd = os.pipe2(
            getattr(os, "O_CLOEXEC", 0)
        )
    else:
        gate_read_fd, gate_write_fd = os.pipe()
    process: Optional[subprocess.Popen] = None
    child: Optional[Child] = None
    pidfd: Optional[int] = None
    gate_released = False
    try:
        process = subprocess.Popen(
            _wrapped_command(claim.setup_path, argv, gate_read_fd),
            cwd=str(claim.checkout / "hope_training/whole_body_tracking"),
            env=_child_env(role),
            stdin=subprocess.DEVNULL,
            stdout=log.fd,
            stderr=log.fd,
            close_fds=True,
            pass_fds=(*lock_fds, gate_read_fd),
            start_new_session=True,
        )
        if hasattr(os, "pidfd_open"):
            pidfd = os.pidfd_open(process.pid, 0)
        os.close(gate_read_fd)
        gate_read_fd = -1
        leader_path = claim.namespace / f"{role}_leader_identity.json"
        identity_doc = exact.bind_leader(
            Path("/proc"), process.pid, process.pid, leader_path
        )
        identity = exact._leader_from(identity_doc)
        child = Child(
            role=role,
            process=process,
            log=log,
            leader_receipt=leader_path,
            identity=identity,
            argv_sha256=canonical_sha256(list(argv)),
        )
        written = os.write(gate_write_fd, b"G")
        if written != 1:
            raise SupervisorError(f"could not release exact {role} start gate")
        gate_released = True
        os.close(gate_write_fd)
        gate_write_fd = -1
        if pidfd is not None:
            os.close(pidfd)
            pidfd = None
        return child
    except BaseException as original:
        if gate_read_fd >= 0:
            os.close(gate_read_fd)
        if gate_write_fd >= 0:
            os.close(gate_write_fd)
            gate_write_fd = -1
        if process is not None and process.poll() is None:
            if gate_released and child is not None:
                try:
                    _stop_exact_child(
                        child,
                        claim=claim,
                        exact=exact,
                        timing=Timing(
                            poll_seconds=0.05,
                            boot_lock_timeout_seconds=5.0,
                            sidecar_ready_timeout_seconds=5.0,
                            trainer_ready_timeout_seconds=5.0,
                            term_grace_seconds=5.0,
                            kill_grace_seconds=5.0,
                        ),
                    )
                except BaseException as cleanup_exc:
                    log.close()
                    if pidfd is not None:
                        os.close(pidfd)
                    raise SupervisorError(
                        f"{role} start-gate failure cleanup refused: "
                        f"{cleanup_exc}"
                    ) from original
            else:
                # Closing the write end delivers EOF to the still-gated shell,
                # so it exits 125 without ever executing workload.  Linux
                # pidfd is the exact, non-reusable fallback if that tiny shell
                # does not reap promptly.
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    if (
                        pidfd is None
                        or not hasattr(signal, "pidfd_send_signal")
                    ):
                        raise SupervisorError(
                            f"still-gated {role} child did not reap after EOF; "
                            "no PID-only signal is permitted"
                        )
                    else:
                        signal.pidfd_send_signal(
                            pidfd, signal.SIGKILL, None, 0
                        )
                        process.wait(timeout=5.0)
        if pidfd is not None:
            os.close(pidfd)
        log.close()
        raise


def _parse_ready_line(
    line: bytes, expected: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not line.startswith(READY_PREFIX):
        raise SupervisorError("internal error: ready parser received other line")
    raw = line[len(READY_PREFIX) :]
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SupervisorError(f"non-finite ready JSON: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisorError("sidecar ready line is not strict UTF-8 JSON") from exc
    if type(value) is not dict or value != dict(expected):
        raise SupervisorError("sidecar ready document differs from exact claim binding")
    if raw != _canonical_bytes(value):
        raise SupervisorError("sidecar ready JSON is not canonical")
    return value


def _check_single_ready(
    lines: Iterable[bytes],
    *,
    expected: Mapping[str, Any],
    already_seen: int,
) -> Tuple[int, Optional[Mapping[str, Any]]]:
    count = already_seen
    found: Optional[Mapping[str, Any]] = None
    for line in lines:
        if line.startswith(READY_PREFIX):
            count += 1
            if count != 1:
                raise SupervisorError("sidecar emitted more than one ready line")
            found = _parse_ready_line(line, expected)
    return count, found


def _observe_sidecar_heartbeat(
    claim: ValidatedClaim,
    evaluator: Child,
    previous: Optional[HeartbeatObservation],
    *,
    now_monotonic_ns: Optional[int] = None,
) -> HeartbeatObservation:
    path = claim.heartbeat_path
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise SupervisorError(
            f"sidecar heartbeat parent cannot be resolved: {exc}"
        ) from exc
    if resolved_parent != path.parent:
        raise SupervisorError("sidecar heartbeat path traverses a symlink")
    heartbeat_value: Dict[str, Any]
    heartbeat_raw: bytes
    for snapshot_attempt in range(5):
        try:
            heartbeat_value, heartbeat_raw = _read_strict_json_with_raw(
                path,
                label="formal sidecar heartbeat",
                max_bytes=64 * 1024,
            )
            break
        except SupervisorError as exc:
            detail = str(exc)
            retryable_atomic_publish_race = any(
                fragment in detail
                for fragment in (
                    "formal sidecar heartbeat changed while reading",
                    "cannot inspect formal sidecar heartbeat",
                    (
                        "formal sidecar heartbeat must be a single-link "
                        "regular non-symlink file"
                    ),
                )
            )
            if not retryable_atomic_publish_race or snapshot_attempt == 4:
                raise
            # The sidecar publishes by atomic rename.  A rename during our
            # read invalidates that snapshot, but a bounded fresh open may
            # accept the next complete single-link inode.
            time.sleep(0)
    document = _exact_keys(
        heartbeat_value,
        ("schema_version", "kind", "content", "content_sha256"),
        label="formal sidecar heartbeat",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != HEARTBEAT_KIND
    ):
        raise SupervisorError("formal sidecar heartbeat schema/kind is invalid")
    canonical_file = _canonical_bytes(document) + b"\n"
    if heartbeat_raw != canonical_file:
        raise SupervisorError(
            "formal sidecar heartbeat is not exact canonical JSON"
        )
    content = _exact_keys(
        document["content"],
        HEARTBEAT_CONTENT_KEYS,
        label="formal sidecar heartbeat content",
    )
    content_sha = _sha(
        document["content_sha256"], label="heartbeat content SHA-256"
    )
    if canonical_sha256(content) != content_sha:
        raise SupervisorError("formal sidecar heartbeat content SHA mismatch")
    expected_identity = {
        "owner_id": claim.expected_sidecar_ready["owner_id"],
        "run_id": claim.expected_sidecar_ready["run_id"],
        "pid": evaluator.identity.pid,
        "sidecar_code_sha256": claim.sidecar_code_sha256,
        "launch_sha256": claim.sidecar_launch_content_sha256,
        "backend_contract_sha256": claim.sidecar_backend_contract_sha256,
    }
    for key, expected in expected_identity.items():
        if content[key] != expected:
            raise SupervisorError(
                f"formal sidecar heartbeat {key} differs from exact claim"
            )

    heartbeat_seq = content["heartbeat_seq"]
    heartbeat_unix_ns = content["heartbeat_unix_ns"]
    heartbeat_monotonic_ns = content["heartbeat_monotonic_ns"]
    phase = content["phase"]
    attempts_completed = content["attempts_completed"]
    attempts_total = content["attempts_total"]
    for label, value, minimum in (
        ("heartbeat_seq", heartbeat_seq, 0),
        ("heartbeat_unix_ns", heartbeat_unix_ns, 1),
        ("heartbeat_monotonic_ns", heartbeat_monotonic_ns, 1),
        ("attempts_completed", attempts_completed, 0),
        ("attempts_total", attempts_total, 0),
    ):
        if type(value) is not int or value < minimum:
            raise SupervisorError(
                f"formal sidecar heartbeat {label} is invalid"
            )
    if type(phase) is not str or phase not in (
        HEARTBEAT_IDLE_PHASES | HEARTBEAT_ACTIVE_PHASES
    ):
        raise SupervisorError("formal sidecar heartbeat phase is invalid")
    now_ns = (
        time.monotonic_ns()
        if now_monotonic_ns is None
        else now_monotonic_ns
    )
    if type(now_ns) is not int or now_ns < heartbeat_monotonic_ns:
        raise SupervisorError(
            "formal sidecar heartbeat monotonic time is from the future"
        )
    stale_ns = int(
        float(
            claim.heartbeat_contract["heartbeat_stale_after_seconds"]
        )
        * 1_000_000_000
    )
    if now_ns - heartbeat_monotonic_ns > stale_ns:
        raise SupervisorError(
            "formal sidecar heartbeat exceeded the 120s stale deadline"
        )

    request_seq = content["request_seq"]
    request_sha = content["request_sha256"]
    request_started_unix_ns = content["request_started_unix_ns"]
    request_started_monotonic_ns = content[
        "request_started_monotonic_ns"
    ]
    request_deadline_unix_ns = content["request_deadline_unix_ns"]
    request_deadline_monotonic_ns = content[
        "request_deadline_monotonic_ns"
    ]
    error_type = content["error_type"]
    if phase in HEARTBEAT_IDLE_PHASES:
        if (
            request_seq is not None
            or request_sha != ""
            or attempts_completed != 0
            or attempts_total != 0
            or request_started_unix_ns != 0
            or request_started_monotonic_ns != 0
            or request_deadline_unix_ns != 0
            or request_deadline_monotonic_ns != 0
            or error_type != ""
        ):
            raise SupervisorError(
                "idle sidecar heartbeat retains active request state"
            )
    else:
        if (
            type(request_seq) is not int
            or request_seq < 0
            or type(request_sha) is not str
            or SHA256_RE.fullmatch(request_sha) is None
            or type(attempts_total) is not int
            or attempts_total < 1
            or type(attempts_completed) is not int
            or not 0 <= attempts_completed <= attempts_total
            or any(
                type(value) is not int or value < 1
                for value in (
                    request_started_unix_ns,
                    request_started_monotonic_ns,
                    request_deadline_unix_ns,
                    request_deadline_monotonic_ns,
                )
            )
            or request_deadline_unix_ns <= request_started_unix_ns
            or request_deadline_monotonic_ns
            <= request_started_monotonic_ns
        ):
            raise SupervisorError(
                "active sidecar heartbeat request state is invalid"
            )
        expected_duration_ns = int(
            float(
                claim.heartbeat_contract["request_deadline_seconds"]
            )
            * 1_000_000_000
        )
        if (
            request_deadline_unix_ns - request_started_unix_ns
            != expected_duration_ns
            or request_deadline_monotonic_ns
            - request_started_monotonic_ns
            != expected_duration_ns
        ):
            raise SupervisorError(
                "active sidecar heartbeat deadline differs from claim"
            )
        if now_ns > request_deadline_monotonic_ns:
            raise SupervisorError(
                "formal sidecar request exceeded its 7200s deadline"
            )
        if phase in ("request_accepted", "runtime_building") and (
            attempts_completed != 0
        ):
            raise SupervisorError(
                f"{phase} heartbeat must have zero completed attempts"
            )
        if phase in ("validating_evidence", "evidence_published") and (
            attempts_completed != attempts_total
        ):
            raise SupervisorError(
                f"{phase} heartbeat must have completed every attempt"
            )
        if phase == "request_failed":
            if (
                type(error_type) is not str
                or SAFE_ERROR_TYPE_RE.fullmatch(error_type) is None
            ):
                raise SupervisorError(
                    "request_failed heartbeat lacks a safe error class"
                )
        elif error_type != "":
            raise SupervisorError(
                "non-failed heartbeat must not carry error_type"
            )

    if previous is not None and content_sha == previous.content_sha256:
        return previous
    max_request_seq = -1 if previous is None else previous.max_request_seq
    if previous is not None:
        if heartbeat_seq <= previous.heartbeat_seq:
            raise SupervisorError(
                "changed sidecar heartbeat sequence did not increase"
            )
        if heartbeat_monotonic_ns <= previous.heartbeat_monotonic_ns:
            raise SupervisorError(
                "changed sidecar heartbeat monotonic time did not increase"
            )
    if phase in HEARTBEAT_ACTIVE_PHASES:
        if type(request_seq) is not int:
            raise SupervisorError(
                "active sidecar heartbeat request sequence is invalid"
            )
        if previous is not None and previous.request_seq == request_seq:
            immutable = (
                request_sha,
                attempts_total,
                request_started_unix_ns,
                request_started_monotonic_ns,
                request_deadline_unix_ns,
                request_deadline_monotonic_ns,
            )
            previous_immutable = (
                previous.request_sha256,
                previous.attempts_total,
                previous.request_started_unix_ns,
                previous.request_started_monotonic_ns,
                previous.request_deadline_unix_ns,
                previous.request_deadline_monotonic_ns,
            )
            if immutable != previous_immutable:
                raise SupervisorError(
                    "sidecar heartbeat changed immutable request identity"
                )
            if attempts_completed < previous.attempts_completed:
                raise SupervisorError(
                    "sidecar heartbeat attempt progress regressed"
                )
        elif request_seq <= max_request_seq:
            raise SupervisorError(
                "sidecar heartbeat request sequence regressed or replayed"
            )
        max_request_seq = max(max_request_seq, request_seq)
    return HeartbeatObservation(
        document=dict(document),
        content_sha256=content_sha,
        heartbeat_seq=heartbeat_seq,
        heartbeat_monotonic_ns=heartbeat_monotonic_ns,
        phase=phase,
        request_seq=request_seq,
        request_sha256=request_sha,
        attempts_completed=attempts_completed,
        attempts_total=attempts_total,
        request_started_unix_ns=request_started_unix_ns,
        request_started_monotonic_ns=request_started_monotonic_ns,
        request_deadline_unix_ns=request_deadline_unix_ns,
        request_deadline_monotonic_ns=request_deadline_monotonic_ns,
        max_request_seq=max_request_seq,
    )


def _process_snapshot(child: Optional[Child]) -> Optional[Mapping[str, Any]]:
    if child is None:
        return None
    return {
        "pid": child.identity.pid,
        "pgid": child.identity.pgid,
        "starttime_ticks": child.identity.starttime_ticks,
        "argv_sha256": child.argv_sha256,
        "returncode": child.process.poll(),
        "leader_receipt": str(child.leader_receipt),
        "leader_receipt_sha256": _sha256_file(child.leader_receipt),
        "term_receipt": (
            "" if child.term_receipt is None else str(child.term_receipt)
        ),
        "term_receipt_sha256": (
            ""
            if child.term_receipt is None
            else _sha256_file(child.term_receipt)
        ),
        "kill_receipt": (
            "" if child.kill_receipt is None else str(child.kill_receipt)
        ),
        "kill_receipt_sha256": (
            ""
            if child.kill_receipt is None
            else _sha256_file(child.kill_receipt)
        ),
    }


def _wait_group_empty(
    exact: Any,
    term_receipt: Path,
    deadline: float,
    poll_seconds: float,
    process: Optional[subprocess.Popen] = None,
) -> bool:
    while True:
        # Reap an exited leader before scanning.  Otherwise its zombie remains
        # visible in /proc as a member of the exact group until Popen.wait/poll.
        if process is not None:
            process.poll()
        residual = exact.verify_residual(Path("/proc"), term_receipt)
        if not residual:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)


def _stop_exact_child(
    child: Child,
    *,
    claim: ValidatedClaim,
    exact: Any,
    timing: Timing,
) -> Mapping[str, Any]:
    if child.term_receipt is not None:
        raise SupervisorError(f"{child.role} termination was attempted twice")
    if child.process.poll() is not None:
        try:
            residual = exact.group_snapshot(Path("/proc"), child.identity.pgid)
        except BaseException as exc:
            raise SupervisorError(
                f"cannot prove exited {child.role} group empty: {exc}"
            ) from exc
        if residual:
            raise SupervisorError(
                f"{child.role} leader exited before TERM with residual group"
            )
        return {
            "role": child.role,
            "already_exited": True,
            "forced_kill": False,
            "returncode": child.process.returncode,
        }
    term_path = claim.namespace / f"{child.role}_pre_term_identity.json"
    try:
        exact.term_group(Path("/proc"), child.leader_receipt, term_path)
    except BaseException as exc:
        # The leader may finish naturally between the pre-TERM poll and the
        # exact identity-bound TERM call.  Reap and accept that race only when
        # the same captured process group is now provably empty.
        child.process.poll()
        try:
            residual = exact.group_snapshot(
                Path("/proc"), child.identity.pgid
            )
        except BaseException:
            residual = [child.identity]
        if child.process.returncode is not None and not residual:
            return {
                "role": child.role,
                "already_exited": True,
                "forced_kill": False,
                "returncode": child.process.returncode,
            }
        raise SupervisorError(
            f"exact {child.role} TERM identity check refused: {exc}"
        ) from exc
    child.term_receipt = term_path
    empty = _wait_group_empty(
        exact,
        term_path,
        time.monotonic() + timing.term_grace_seconds,
        timing.poll_seconds,
        child.process,
    )
    forced = False
    if not empty:
        kill_path = claim.namespace / f"{child.role}_pre_kill_identity.json"
        try:
            exact.kill_residual(Path("/proc"), term_path, kill_path)
        except BaseException as exc:
            raise SupervisorError(
                f"exact {child.role} KILL identity check refused: {exc}"
            ) from exc
        child.kill_receipt = kill_path
        forced = True
        if not _wait_group_empty(
            exact,
            term_path,
            time.monotonic() + timing.kill_grace_seconds,
            timing.poll_seconds,
            child.process,
        ):
            raise SupervisorError(
                f"{child.role} process group survived exact SIGKILL"
            )
    try:
        returncode = child.process.wait(timeout=timing.kill_grace_seconds)
    except subprocess.TimeoutExpired as exc:
        raise SupervisorError(
            f"{child.role} leader did not become waitable after group exit"
        ) from exc
    return {
        "role": child.role,
        "already_exited": False,
        "forced_kill": forced,
        "returncode": returncode,
    }


def _select_terminal_checkpoint(
    claim: ValidatedClaim,
) -> Tuple[Path, Mapping[str, Any]]:
    log_snapshot = _snapshot_regular_file(
        claim.namespace / "train.log",
        label="trainer log",
        max_bytes=64 * 1024 * 1024,
    )
    try:
        text = log_snapshot["raw"].decode("utf-8")
    except UnicodeError as exc:
        raise SupervisorError("trainer log is not UTF-8") from exc
    prefix = (
        f"[INFO] Task: {ACTION_BALL_TASK_ID} | "
        "experiment: "
        f"{claim.action_set_contract['experiment_name']} | log: "
    )
    outputs = [
        line[len(prefix) :]
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(outputs) != 1:
        raise SupervisorError(
            "trainer log must name exactly one claim-bound RSL output"
        )
    output = _absolute_path(
        outputs[0], label="trainer RSL output", must_exist=True
    )
    try:
        output_mode = output.lstat().st_mode
    except OSError as exc:
        raise SupervisorError("cannot inspect trainer RSL output") from exc
    if not stat.S_ISDIR(output_mode):
        raise SupervisorError("trainer RSL output is not a real directory")
    expected_parent = (
        claim.checkout
        / "hope_training/whole_body_tracking/logs/rsl_rl"
        / str(claim.action_set_contract["experiment_name"])
    )
    if output.parent != expected_parent:
        raise SupervisorError(
            "trainer RSL output escaped the dedicated experiment root"
        )
    suffix = "_" + claim.namespace.name
    if not output.name.endswith(suffix):
        raise SupervisorError(
            "trainer RSL output basename does not bind the namespace"
        )
    timestamp = output.name[: -len(suffix)]
    if RSL_TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise SupervisorError("trainer RSL output timestamp is not canonical")
    try:
        parsed = _datetime.datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
    except ValueError as exc:
        raise SupervisorError(
            "trainer RSL output timestamp is not a real date"
        ) from exc
    if parsed.strftime("%Y-%m-%d_%H-%M-%S") != timestamp:
        raise SupervisorError(
            "trainer RSL output timestamp failed round-trip validation"
        )

    candidates: List[Tuple[int, Path]] = []
    try:
        entries = list(output.iterdir())
    except OSError as exc:
        raise SupervisorError("cannot enumerate trainer RSL output") from exc
    for path in entries:
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            continue
        try:
            info = path.lstat()
        except OSError as exc:
            raise SupervisorError(
                f"cannot inspect checkpoint candidate: {path}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise SupervisorError("checkpoint candidate must not be a symlink")
        if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise SupervisorError(
            "trainer RSL output contains no model_<N>.pt checkpoint"
        )
    iteration, checkpoint = max(candidates)
    if (
        iteration < claim.max_iterations - 1
        or iteration > claim.max_iterations
    ):
        raise SupervisorError(
            "terminal checkpoint iteration is outside the stage budget"
        )
    before = checkpoint.lstat()
    checkpoint_sha = _sha256_file(checkpoint)
    after = checkpoint.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    ):
        raise SupervisorError("terminal checkpoint changed while selecting it")
    return checkpoint, {
        "path": str(checkpoint),
        "sha256": checkpoint_sha,
        "size_bytes": before.st_size,
        "embedded_iteration": iteration,
    }


def _validate_exact_resume_receipt(
    *,
    claim: ValidatedClaim,
    checkpoint: Path,
    checkpoint_snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    receipt_path = claim.namespace / "exact_resume_verification.json"
    document, raw = _read_strict_json_with_raw(
        receipt_path,
        label="exact-resume verification receipt",
        max_bytes=16 * 1024 * 1024,
    )
    row = _exact_keys(
        document,
        (
            "schema_version",
            "kind",
            "status",
            "source_commit_sha",
            "launch_claim_sha256",
            "stage",
            "namespace",
            "verifier",
            "source_checkpoint",
            "roundtrip_checkpoint",
            "runtime_bootstrap",
            "restore",
            "state",
            "natural_exit",
            "receipt_payload_sha256",
        ),
        label="exact-resume verification receipt",
    )
    unsigned = {
        key: value
        for key, value in row.items()
        if key != "receipt_payload_sha256"
    }
    payload_sha = _sha(
        row["receipt_payload_sha256"],
        label="exact-resume receipt payload SHA-256",
    )
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != 1
        or row["kind"] != "action_ball_exact_resume_verification_v1"
        or row["status"] != "passed"
        or row["source_commit_sha"] != claim.source_commit
        or row["launch_claim_sha256"] != claim.claim_sha256
        or row["stage"] != claim.stage
        or row["namespace"] != str(claim.namespace)
        or row["natural_exit"] is not True
        or canonical_sha256(unsigned) != payload_sha
        or raw != _canonical_bytes(row) + b"\n"
    ):
        raise SupervisorError(
            "exact-resume verification receipt identity/payload is invalid"
        )

    verifier = _exact_keys(
        row["verifier"],
        (
            "source_path",
            "source_sha256",
            "runtime_factory_source_path",
            "runtime_factory_source_sha256",
        ),
        label="exact-resume verifier identity",
    )
    if (
        verifier["source_path"] != EXACT_RESUME_VERIFIER_SOURCE
        or verifier["source_sha256"]
        != claim.runtime_code_sha256[EXACT_RESUME_VERIFIER_SOURCE]
        or verifier["runtime_factory_source_path"] != SIDECAR_SOURCE
        or verifier["runtime_factory_source_sha256"]
        != claim.runtime_code_sha256[SIDECAR_SOURCE]
    ):
        raise SupervisorError(
            "exact-resume verifier/factory identity differs from claim pins"
        )

    source = _exact_keys(
        row["source_checkpoint"],
        ("path", "sha256", "size_bytes", "embedded_iteration"),
        label="exact-resume source checkpoint",
    )
    if (
        source != dict(checkpoint_snapshot)
        or source["path"] != str(checkpoint)
        or _sha256_file(checkpoint) != source["sha256"]
    ):
        raise SupervisorError(
            "exact-resume receipt source checkpoint differs from terminal bytes"
        )

    roundtrip = _exact_keys(
        row["roundtrip_checkpoint"],
        ("path", "sha256", "size_bytes", "embedded_iteration"),
        label="exact-resume roundtrip checkpoint",
    )
    expected_roundtrip = (
        checkpoint.parent
        / f"exact_resume_roundtrip_{claim.claim_sha256[:16]}"
        / checkpoint.name
    )
    roundtrip_path = _absolute_path(
        roundtrip["path"],
        label="exact-resume roundtrip checkpoint",
        must_exist=True,
    )
    try:
        roundtrip_info = roundtrip_path.lstat()
    except OSError as exc:
        raise SupervisorError(
            "cannot inspect exact-resume roundtrip checkpoint"
        ) from exc
    if (
        roundtrip_path != expected_roundtrip
        or not stat.S_ISREG(roundtrip_info.st_mode)
        or roundtrip_info.st_nlink != 1
        or roundtrip_info.st_size != roundtrip["size_bytes"]
        or roundtrip["embedded_iteration"]
        != checkpoint_snapshot["embedded_iteration"]
        or _sha256_file(roundtrip_path)
        != _sha(roundtrip["sha256"], label="roundtrip checkpoint SHA-256")
    ):
        raise SupervisorError(
            "exact-resume roundtrip checkpoint artifact is invalid"
        )

    bootstrap = _exact_keys(
        row["runtime_bootstrap"],
        ("content_sha256", "lineage_payload_sha256"),
        label="exact-resume runtime bootstrap",
    )
    for key, value in bootstrap.items():
        _sha(value, label=f"exact-resume runtime bootstrap {key}")

    restore = _exact_keys(
        row["restore"],
        (
            "factory_call_count",
            "closed_runtime_count",
            "load_optimizer",
            "fresh_strict_load_token_consumed",
            "roundtrip_save_api",
            "roundtrip_save_receipt_sha256",
            "source_construction_receipt_sha256",
            "roundtrip_construction_receipt_sha256",
            "runtime_inventory_live_verification_sha256",
            "source_live_state_receipt_sha256",
            "roundtrip_live_state_receipt_sha256",
            "live_core_sha256",
            "common_step_counter",
            "common_step_counter_delta",
        ),
        label="exact-resume restore proof",
    )
    if (
        type(restore["factory_call_count"]) is not int
        or restore["factory_call_count"] != 2
        or type(restore["closed_runtime_count"]) is not int
        or restore["closed_runtime_count"] != 2
        or restore["load_optimizer"] is not True
        or restore["fresh_strict_load_token_consumed"] is not True
        or restore["roundtrip_save_api"] != "save_exact_resume_roundtrip"
        or type(restore["common_step_counter"]) is not int
        or restore["common_step_counter"] < 0
        or type(restore["common_step_counter_delta"]) is not int
        or restore["common_step_counter_delta"] != 0
        or restore["source_live_state_receipt_sha256"]
        != restore["roundtrip_live_state_receipt_sha256"]
    ):
        raise SupervisorError(
            "exact-resume restore proof does not prove two closed restores"
        )
    for key in (
        "roundtrip_save_receipt_sha256",
        "source_construction_receipt_sha256",
        "roundtrip_construction_receipt_sha256",
        "runtime_inventory_live_verification_sha256",
        "source_live_state_receipt_sha256",
        "roundtrip_live_state_receipt_sha256",
        "live_core_sha256",
    ):
        _sha(restore[key], label=f"exact-resume restore {key}")

    state = _exact_keys(
        row["state"],
        (
            "source_core_sha256",
            "roundtrip_core_sha256",
            "source_exact_resume_sha256",
            "roundtrip_exact_resume_sha256",
            "model_state_sha256",
            "optimizer_state_sha256",
            "normalizer_state_sha256",
        ),
        label="exact-resume state proof",
    )
    for key, value in state.items():
        _sha(value, label=f"exact-resume state {key}")
    if (
        state["source_core_sha256"] != state["roundtrip_core_sha256"]
        or state["source_exact_resume_sha256"]
        != state["roundtrip_exact_resume_sha256"]
    ):
        raise SupervisorError(
            "exact-resume source and roundtrip state digests differ"
        )
    return {
        "path": str(receipt_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "payload_sha256": payload_sha,
    }


def _wait_for_natural_child_exit(
    child: Child,
    *,
    exact: Any,
    stop: StopRequest,
    timeout_seconds: float,
    poll_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        stop.check()
        returncode = child.process.poll()
        if returncode is not None:
            residual = exact.group_snapshot(
                Path("/proc"), child.identity.pgid
            )
            if residual:
                raise SupervisorError(
                    f"{child.role} leader exited with residual process-group "
                    "members"
                )
            return returncode
        if time.monotonic() >= deadline:
            raise SupervisorError(
                f"timed out waiting for natural {child.role} exit"
            )
        time.sleep(poll_seconds)


def _artifact_paths(namespace: Path) -> Tuple[Path, ...]:
    names = (
        "evaluator.log",
        "train.log",
        "exact_resume_verifier.log",
        "evaluator_leader_identity.json",
        "trainer_leader_identity.json",
        "verifier_leader_identity.json",
        "evaluator_pre_term_identity.json",
        "trainer_pre_term_identity.json",
        "verifier_pre_term_identity.json",
        "evaluator_pre_kill_identity.json",
        "trainer_pre_kill_identity.json",
        "verifier_pre_kill_identity.json",
        "exact_resume_verification.json",
        "supervisor_ready.json",
        "supervisor_failed.json",
        "supervisor_cleanup_blocked.json",
        "supervisor_terminal.json",
        "launch_accept_intent.json",
        "launch_accept_ack.json",
        "launch_accepted.json",
        "launch_commit_ack.json",
    )
    return tuple(namespace / name for name in names)


def _require_no_clobber_namespace(namespace: Path) -> None:
    collisions = [str(path) for path in _artifact_paths(namespace) if os.path.lexists(path)]
    if collisions:
        raise SupervisorError(
            f"supervisor namespace artifacts already exist: {collisions}"
        )


def supervise_stage(
    claim: ValidatedClaim,
    *,
    trainer_lock_fd: int,
    evaluator_lock_fd: int,
    launcher_control_fd: Optional[int] = None,
    exact: Any,
    timing: Timing = Timing(),
    runtime_paths: RuntimePaths = RuntimePaths(),
    stop_request: Optional[StopRequest] = None,
) -> Mapping[str, Any]:
    """Run the complete evaluator-first stage lifecycle."""

    if trainer_lock_fd == evaluator_lock_fd:
        raise SupervisorError("trainer/evaluator lock descriptors must differ")
    if launcher_control_fd is not None:
        if launcher_control_fd in (trainer_lock_fd, evaluator_lock_fd):
            raise SupervisorError(
                "launcher control descriptor must differ from GPU locks"
            )
        control_mode = os.fstat(launcher_control_fd).st_mode
        if not stat.S_ISFIFO(control_mode):
            raise SupervisorError("launcher control descriptor must be a pipe")
        os.set_blocking(launcher_control_fd, False)
    _verify_lock_fd(
        trainer_lock_fd, runtime_paths.trainer_lock, role="trainer"
    )
    _verify_lock_fd(
        evaluator_lock_fd, runtime_paths.evaluator_lock, role="evaluator"
    )
    _require_no_clobber_namespace(claim.namespace)
    boot_fd = _open_boot_lock(runtime_paths.boot_lock)
    children: Dict[str, Child] = {}
    ready_doc: Optional[Mapping[str, Any]] = None
    ready_count = 0
    heartbeat: Optional[HeartbeatObservation] = None
    initial_heartbeat_document: Optional[Mapping[str, Any]] = None
    trainer_learning_line = ""
    state = "preparing_evaluator"
    cleanup: List[Mapping[str, Any]] = []
    lock_fds = (trainer_lock_fd, evaluator_lock_fd)
    stop = StopRequest() if stop_request is None else stop_request
    old_handlers: Dict[int, Any] = {}
    if stop_request is None:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, stop.handler)

    def evaluator_alive() -> None:
        nonlocal heartbeat
        stop.check()
        evaluator = children.get("evaluator")
        if evaluator is not None and evaluator.process.poll() is not None:
            raise SupervisorError(
                "formal evaluator exited before trainer boot completed; "
                f"returncode={evaluator.process.returncode}"
            )
        if evaluator is not None and heartbeat is not None:
            heartbeat = _observe_sidecar_heartbeat(
                claim, evaluator, heartbeat
            )

    try:
        _acquire_boot_lock(boot_fd, timing=timing, guard=stop.check)
        _verify_boot_lock_path_after_flock(
            boot_fd, runtime_paths.boot_lock
        )
        try:
            children["evaluator"] = _start_child(
                role="evaluator",
                argv=claim.sidecar_argv,
                claim=claim,
                lock_fds=lock_fds,
                exact=exact,
            )
            state = "waiting_evaluator_ready"
            deadline = time.monotonic() + timing.sidecar_ready_timeout_seconds
            while ready_doc is None:
                stop.check()
                evaluator = children["evaluator"]
                ready_count, found = _check_single_ready(
                    evaluator.log.lines(),
                    expected=claim.expected_sidecar_ready,
                    already_seen=ready_count,
                )
                if found is not None:
                    ready_doc = found
                    heartbeat = _observe_sidecar_heartbeat(
                        claim, evaluator, None
                    )
                    if heartbeat.phase not in (
                        "ready",
                        "waiting_for_request_or_ack",
                    ):
                        raise SupervisorError(
                            "first ready-bound heartbeat phase must be ready "
                            "or waiting_for_request_or_ack"
                        )
                    initial_heartbeat_document = heartbeat.document
                    break
                returncode = evaluator.process.poll()
                if returncode is not None:
                    raise SupervisorError(
                        "formal evaluator exited before exact ready; "
                        f"returncode={returncode}"
                    )
                if time.monotonic() >= deadline:
                    raise SupervisorError(
                        "timed out waiting for exact formal sidecar ready line"
                    )
                time.sleep(timing.poll_seconds)
        finally:
            _release_boot_lock(boot_fd)

        state = "preparing_trainer"
        _acquire_boot_lock(boot_fd, timing=timing, guard=evaluator_alive)
        _verify_boot_lock_path_after_flock(
            boot_fd, runtime_paths.boot_lock
        )
        try:
            children["trainer"] = _start_child(
                role="trainer",
                argv=claim.trainer_argv,
                claim=claim,
                lock_fds=lock_fds,
                exact=exact,
            )
            state = "waiting_trainer_learning_iteration"
            deadline = time.monotonic() + timing.trainer_ready_timeout_seconds
            while not trainer_learning_line:
                stop.check()
                evaluator = children["evaluator"]
                trainer = children["trainer"]
                ready_count, duplicate = _check_single_ready(
                    evaluator.log.lines(),
                    expected=claim.expected_sidecar_ready,
                    already_seen=ready_count,
                )
                if duplicate is not None:
                    raise SupervisorError("duplicate evaluator ready line")
                evaluator_rc = evaluator.process.poll()
                if evaluator_rc is not None:
                    raise SupervisorError(
                        "formal evaluator exited during trainer boot; "
                        f"returncode={evaluator_rc}"
                    )
                heartbeat = _observe_sidecar_heartbeat(
                    claim, evaluator, heartbeat
                )
                for line in trainer.log.lines():
                    if LEARNING_MARKER_RE.search(line) is not None:
                        try:
                            trainer_learning_line = line.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            raise SupervisorError(
                                "trainer learning marker line is not UTF-8"
                            ) from exc
                        break
                trainer_rc = trainer.process.poll()
                if trainer_rc is not None and not trainer_learning_line:
                    raise SupervisorError(
                        "trainer exited before Learning iteration; "
                        f"returncode={trainer_rc}"
                    )
                if time.monotonic() >= deadline:
                    raise SupervisorError(
                        "timed out waiting for trainer Learning iteration"
                    )
                time.sleep(timing.poll_seconds)
        finally:
            _release_boot_lock(boot_fd)

        state = "running"
        ready_receipt = {
            "schema_version": 1,
            "kind": "action_ball_stage_supervisor_ready",
            "ready_utc": _utc_now(),
            "claim_sha256": claim.claim_sha256,
            "source_commit_sha": claim.source_commit,
            "stage": claim.stage,
            "namespace": str(claim.namespace),
            "gpu_roles": claim.gpus,
            "sidecar_ready": ready_doc,
            "sidecar_heartbeat_initial": initial_heartbeat_document,
            "trainer_learning_line": trainer_learning_line,
            "processes": {
                role: _process_snapshot(children.get(role))
                for role in ("evaluator", "trainer")
            },
            "logs": {
                "evaluator": str(claim.namespace / "evaluator.log"),
                "trainer": str(claim.namespace / "train.log"),
            },
        }
        _publish_exclusive_json(
            claim.namespace / "supervisor_ready.json", ready_receipt
        )

        def acceptance_guard() -> None:
            nonlocal heartbeat
            stop.check()
            evaluator = children["evaluator"]
            trainer = children["trainer"]
            nonlocal ready_count
            ready_count, duplicate = _check_single_ready(
                evaluator.log.lines(),
                expected=claim.expected_sidecar_ready,
                already_seen=ready_count,
            )
            if duplicate is not None:
                raise SupervisorError("duplicate evaluator ready line")
            if evaluator.process.poll() is not None:
                raise SupervisorError(
                    "formal evaluator exited during launcher acceptance"
                )
            heartbeat = _observe_sidecar_heartbeat(
                claim, evaluator, heartbeat
            )
            if trainer.process.poll() is not None:
                raise SupervisorError(
                    "trainer exited during launcher acceptance"
                )

        ready_path = claim.namespace / "supervisor_ready.json"
        ready_sha = _sha256_file(ready_path)
        live_gpu_admission_path = (
            claim.namespace / "live_gpu_admission.json"
        )
        live_gpu_admission_sha = _sha256_file(
            live_gpu_admission_path
        )
        intent_path = claim.namespace / "launch_accept_intent.json"
        ack_path = claim.namespace / "launch_accept_ack.json"
        accepted_path = claim.namespace / "launch_accepted.json"
        commit_ack_path = claim.namespace / "launch_commit_ack.json"
        expected_intent = {
            "schema_version": 1,
            "kind": "action_ball_launcher_accept_intent",
            "launch_claim_sha256": claim.claim_sha256,
            "supervisor_ready_sha256": ready_sha,
            "live_gpu_admission_sha256": live_gpu_admission_sha,
        }
        state = "waiting_launcher_accept_intent"
        deadline = (
            time.monotonic() + timing.launcher_accept_timeout_seconds
        )
        accept_signal = launcher_control_fd is None
        while not (accept_signal and intent_path.is_file()):
            acceptance_guard()
            if launcher_control_fd is not None:
                try:
                    token = os.read(launcher_control_fd, 1)
                except BlockingIOError:
                    token = None
                if token == b"A":
                    accept_signal = True
                elif token in (b"C", b""):
                    raise SupervisorError(
                        "launcher cancelled before acceptance"
                    )
                elif token is not None:
                    raise SupervisorError(
                        "launcher control protocol emitted an invalid token"
                    )
            if time.monotonic() >= deadline:
                raise SupervisorError(
                    "timed out waiting for launcher accept intent"
                )
            time.sleep(timing.poll_seconds)
        intent = _read_inflight_published_json(
            intent_path,
            label="launcher accept intent",
            timing=timing,
            guard=acceptance_guard,
        )
        if intent != expected_intent or intent_path.read_bytes() != (
            _canonical_bytes(expected_intent) + b"\n"
        ):
            raise SupervisorError(
                "launcher accept intent is not exact/canonical"
            )
        intent_sha = _sha256_file(intent_path)
        ack = {
            "schema_version": 1,
            "kind": "action_ball_stage_supervisor_accept_ack",
            "launch_claim_sha256": claim.claim_sha256,
            "supervisor_ready_sha256": ready_sha,
            "accept_intent_sha256": intent_sha,
            "live_gpu_admission_sha256": live_gpu_admission_sha,
        }
        _publish_exclusive_json(ack_path, ack)

        state = "waiting_launcher_accept_commit"
        deadline = (
            time.monotonic() + timing.launcher_accept_timeout_seconds
        )
        while not accepted_path.is_file():
            acceptance_guard()
            if launcher_control_fd is not None:
                try:
                    token = os.read(launcher_control_fd, 1)
                except BlockingIOError:
                    token = None
                if token in (b"C", b""):
                    raise SupervisorError(
                        "launcher cancelled before accepted commit"
                    )
                if token is not None:
                    raise SupervisorError(
                        "launcher control protocol emitted an invalid token"
                    )
            if time.monotonic() >= deadline:
                raise SupervisorError(
                    "timed out waiting for launcher accepted commit"
                )
            time.sleep(timing.poll_seconds)
        accepted_document = _read_inflight_published_json(
            accepted_path,
            label="launcher accepted commit",
            timing=timing,
            guard=acceptance_guard,
        )
        accepted_row = _exact_keys(
            accepted_document,
            (
                "schema_version",
                "kind",
                "accepted_utc",
                "stage",
                "namespace",
                "launch_claim_sha256",
                "supervisor_ready",
                "accept_intent_sha256",
                "supervisor_accept_ack_sha256",
                "live_gpu_admission_sha256",
            ),
            label="launcher accepted commit",
        )
        if (
            accepted_row["schema_version"] != 1
            or accepted_row["kind"] != "action_ball_launch_accepted"
            or not isinstance(accepted_row["accepted_utc"], str)
            or not accepted_row["accepted_utc"]
            or accepted_row["stage"] != claim.stage
            or accepted_row["namespace"] != str(claim.namespace)
            or accepted_row["launch_claim_sha256"] != claim.claim_sha256
            or accepted_row["supervisor_ready"] != ready_receipt
            or accepted_row["accept_intent_sha256"] != intent_sha
            or accepted_row["supervisor_accept_ack_sha256"]
            != _sha256_file(ack_path)
            or accepted_row["live_gpu_admission_sha256"]
            != live_gpu_admission_sha
            or accepted_path.read_bytes()
            != _canonical_bytes(accepted_row) + b"\n"
        ):
            raise SupervisorError(
                "launcher accepted commit is not exact/canonical"
            )
        acceptance_guard()
        accepted_sha = _sha256_file(accepted_path)
        commit_ack = {
            "schema_version": 1,
            "kind": "action_ball_stage_supervisor_launch_commit_ack",
            "launch_claim_sha256": claim.claim_sha256,
            "supervisor_ready_sha256": ready_sha,
            "accept_intent_sha256": intent_sha,
            "supervisor_accept_ack_sha256": _sha256_file(ack_path),
            "launch_accepted_sha256": accepted_sha,
            "live_gpu_admission_sha256": live_gpu_admission_sha,
            "processes": ready_receipt["processes"],
        }
        _publish_exclusive_json(commit_ack_path, commit_ack)

        state = "running"
        while True:
            stop.check()
            evaluator = children["evaluator"]
            trainer = children["trainer"]
            ready_count, duplicate = _check_single_ready(
                evaluator.log.lines(),
                expected=claim.expected_sidecar_ready,
                already_seen=ready_count,
            )
            if duplicate is not None:
                raise SupervisorError("duplicate evaluator ready line")
            evaluator_rc = evaluator.process.poll()
            trainer_rc = trainer.process.poll()
            if evaluator_rc is not None:
                if trainer_rc is None:
                    raise SupervisorError(
                        "formal evaluator exited while trainer was active; "
                        f"returncode={evaluator_rc}"
                    )
                raise SupervisorError(
                    "formal evaluator exited before controlled shutdown; "
                    f"evaluator={evaluator_rc}, trainer={trainer_rc}"
                )
            heartbeat = _observe_sidecar_heartbeat(
                claim, evaluator, heartbeat
            )
            if trainer_rc is not None:
                if trainer_rc != 0:
                    raise SupervisorError(
                        f"trainer exited nonzero; returncode={trainer_rc}"
                    )
                trainer_residual = exact.group_snapshot(
                    Path("/proc"), trainer.identity.pgid
                )
                if trainer_residual:
                    raise SupervisorError(
                        "trainer leader exited zero with residual process-group "
                        "members; refusing stage completion"
                    )
                state = "stopping_evaluator_after_trainer_success"
                stopped = _stop_exact_child(
                    evaluator,
                    claim=claim,
                    exact=exact,
                    timing=timing,
                )
                cleanup.append(stopped)
                if stopped["forced_kill"]:
                    raise SupervisorError(
                        "formal evaluator required SIGKILL after trainer success"
                    )
                state = "selecting_terminal_checkpoint"
                checkpoint, checkpoint_snapshot = (
                    _select_terminal_checkpoint(claim)
                )
                receipt_path = (
                    claim.namespace / "exact_resume_verification.json"
                )
                if os.path.lexists(receipt_path):
                    raise SupervisorError(
                        "exact-resume receipt namespace was spent before verifier"
                    )
                try:
                    nosite = _load_nosite_bootstrap_module(
                        claim.nosite_bootstrap_path
                    )
                    verifier_command = nosite.build_exact_nosite_argv(
                        python=Path(claim.trainer_argv[0]),
                        bootstrap=claim.nosite_bootstrap_path,
                        bootstrap_sha256=claim.runtime_code_sha256[
                            NOSITE_BOOTSTRAP_SOURCE
                        ],
                        entrypoint=claim.exact_resume_verifier_path,
                        entrypoint_sha256=claim.runtime_code_sha256[
                            EXACT_RESUME_VERIFIER_SOURCE
                        ],
                        import_roots=list(claim.nosite_import_roots),
                        entrypoint_argv=[
                            "--claim",
                            str(claim.claim_path),
                            "--checkpoint",
                            str(checkpoint),
                            "--out",
                            str(receipt_path),
                        ],
                    )
                    nosite.validate_exact_nosite_argv(
                        verifier_command.argv,
                        expected_python=Path(claim.trainer_argv[0]),
                        expected_bootstrap=(
                            verifier_command.contract["bootstrap"]
                        ),
                        expected_entrypoint=(
                            verifier_command.contract["entrypoint"]
                        ),
                        expected_import_roots=list(
                            claim.nosite_import_roots
                        ),
                        expected_entrypoint_argv=[
                            "--claim",
                            str(claim.claim_path),
                            "--checkpoint",
                            str(checkpoint),
                            "--out",
                            str(receipt_path),
                        ],
                        expected_contract_sha256=(
                            verifier_command.contract_sha256
                        ),
                        verify_live=True,
                    )
                    verifier_argv = verifier_command.argv
                except Exception as exc:
                    raise SupervisorError(
                        "exact-resume no-site command construction failed"
                    ) from exc
                state = "starting_exact_resume_verifier"
                _acquire_boot_lock(
                    boot_fd, timing=timing, guard=stop.check
                )
                _verify_boot_lock_path_after_flock(
                    boot_fd, runtime_paths.boot_lock
                )
                # Unlike trainer/evaluator, the verifier has no early
                # AppLauncher-ready protocol.  Retain the global Kit boot lock
                # for its entire bounded lifetime (including failure cleanup)
                # so Python pre-import and AppLauncher can never escape the
                # serialized window.
                children["verifier"] = _start_child(
                    role="verifier",
                    argv=verifier_argv,
                    claim=claim,
                    # The inherited descriptor shares the parent's flock open
                    # file description.  If the supervisor is SIGKILLed, the
                    # independent verifier session therefore keeps Kit boot
                    # serialized until that exact child exits.
                    lock_fds=(*lock_fds, boot_fd),
                    exact=exact,
                )
                state = "waiting_exact_resume_verifier"
                verifier_rc = _wait_for_natural_child_exit(
                    children["verifier"],
                    exact=exact,
                    stop=stop,
                    timeout_seconds=(
                        timing.exact_resume_timeout_seconds
                    ),
                    poll_seconds=timing.poll_seconds,
                )
                if verifier_rc != 0:
                    raise SupervisorError(
                        "exact-resume verifier exited nonzero; "
                        f"returncode={verifier_rc}"
                    )
                state = "validating_exact_resume_receipt"
                exact_resume_receipt = _validate_exact_resume_receipt(
                    claim=claim,
                    checkpoint=checkpoint,
                    checkpoint_snapshot=checkpoint_snapshot,
                )
                _release_boot_lock(boot_fd)
                cleanup.append(
                    {
                        "role": "verifier",
                        "already_exited": True,
                        "forced_kill": False,
                        "returncode": verifier_rc,
                        "receipt": exact_resume_receipt,
                    }
                )
                state = "completed"
                terminal = {
                    "schema_version": 1,
                    "kind": "action_ball_stage_supervisor_terminal",
                    "terminal_utc": _utc_now(),
                    "status": "completed",
                    "claim_sha256": claim.claim_sha256,
                    "source_commit_sha": claim.source_commit,
                    "stage": claim.stage,
                    "namespace": str(claim.namespace),
                    "trainer_returncode": trainer_rc,
                    "evaluator_returncode": evaluator.process.returncode,
                    "cleanup": cleanup,
                    "processes": {
                        role: _process_snapshot(children.get(role))
                        for role in ("evaluator", "trainer", "verifier")
                    },
                }
                _publish_exclusive_json(
                    claim.namespace / "supervisor_terminal.json", terminal
                )
                return terminal
            time.sleep(timing.poll_seconds)
    except BaseException as exc:
        original = exc
        failure_state = state
        cleanup_errors: List[str] = []
        closed_roles = {
            item.get("role")
            for item in cleanup
            if type(item) is dict
        }
        for role in ("verifier", "trainer", "evaluator"):
            child = children.get(role)
            if child is None or role in closed_roles:
                continue
            try:
                cleanup.append(
                    _stop_exact_child(
                        child,
                        claim=claim,
                        exact=exact,
                        timing=timing,
                    )
                )
            except BaseException as cleanup_exc:
                cleanup_errors.append(
                    f"{role}: {type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        failure = {
            "schema_version": 2,
            "kind": "action_ball_stage_supervisor_closed_failure",
            "failed_utc": _utc_now(),
            "claim_sha256": claim.claim_sha256,
            "source_commit_sha": claim.source_commit,
            "stage": claim.stage,
            "namespace": str(claim.namespace),
            "state": failure_state,
            "failure_class": type(original).__name__,
            "detail": str(original)[-8000:],
            "cleanup": cleanup,
            "cleanup_errors": cleanup_errors,
            "cleanup_status": (
                "closed" if not cleanup_errors else "blocked"
            ),
            "processes": {
                role: _process_snapshot(children.get(role))
                for role in ("evaluator", "trainer", "verifier")
            },
        }
        failed_path = claim.namespace / "supervisor_failed.json"
        if cleanup_errors:
            blocked_path = (
                claim.namespace / "supervisor_cleanup_blocked.json"
            )
            try:
                _publish_exclusive_json(blocked_path, failure)
            except BaseException as publish_exc:
                raise SupervisorError(
                    "cleanup is blocked and its unresolved receipt could not "
                    f"be published: {publish_exc}"
                ) from publish_exc
            raise SupervisorError(
                f"{original}; exact child cleanup is blocked; unresolved "
                f"identities were published at {blocked_path}"
            ) from original
        if not os.path.lexists(failed_path):
            try:
                _publish_exclusive_json(failed_path, failure)
            except BaseException as publish_exc:
                blocked_path = (
                    claim.namespace / "supervisor_cleanup_blocked.json"
                )
                if not os.path.lexists(blocked_path):
                    try:
                        _publish_exclusive_json(
                            blocked_path,
                            {
                                **failure,
                                "cleanup_status": "blocked",
                                "cleanup_errors": [
                                    "closed_failure_receipt: "
                                    f"{type(publish_exc).__name__}: "
                                    f"{publish_exc}"
                                ],
                            },
                        )
                    except BaseException:
                        pass
                raise SupervisorError(
                    "closed-failure receipt publication failed; the bounded "
                    f"cleanup state was published at {blocked_path}"
                ) from publish_exc
        if isinstance(original, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(original, SupervisorError):
            raise
        raise SupervisorError(
            f"supervisor state {failure_state} failed: "
            f"{type(original).__name__}: {original}"
        ) from original
    finally:
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)
        try:
            _release_boot_lock(boot_fd)
        except OSError:
            pass
        os.close(boot_fd)
        if launcher_control_fd is not None:
            os.close(launcher_control_fd)
        for child in children.values():
            child.log.close()


def run(
    *,
    claim_path: str,
    claim_sha256: str,
    trainer_lock_fd: int,
    evaluator_lock_fd: int,
    launcher_control_fd: int,
) -> Mapping[str, Any]:
    claim, exact = validate_claim_and_source(claim_path, claim_sha256)
    return supervise_stage(
        claim,
        trainer_lock_fd=trainer_lock_fd,
        evaluator_lock_fd=evaluator_lock_fd,
        launcher_control_fd=launcher_control_fd,
        exact=exact,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one exact evaluator-first ActionBall stage"
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    command = sub.add_parser("run")
    command.add_argument("--claim", required=True)
    command.add_argument("--claim-sha256", required=True)
    command.add_argument("--trainer-lock-fd", type=int, required=True)
    command.add_argument("--evaluator-lock-fd", type=int, required=True)
    command.add_argument("--launcher-control-fd", type=int, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(
            claim_path=args.claim,
            claim_sha256=args.claim_sha256,
            trainer_lock_fd=args.trainer_lock_fd,
            evaluator_lock_fd=args.evaluator_lock_fd,
            launcher_control_fd=args.launcher_control_fd,
        )
    except (SupervisorError, OSError, subprocess.SubprocessError) as exc:
        print(
            f"ACTION_BALL_STAGE_SUPERVISOR_REFUSED: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
