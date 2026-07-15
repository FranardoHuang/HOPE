"""Fail-closed loader for exogenous post-swing teacher-state receipts.

The live A8 ring buffer is populated only when the current policy survives to a
natural clip wrap.  That makes a from-scratch buffer endogenous to any reward
being ablated: a treatment that survives longer receives the replay curriculum
earlier.  This module loads a content-addressed set of *previously captured
natural-wrap states* so matched arms can start from the same replay distribution
before their policies diverge.

The loader is intentionally independent of Isaac Lab.  MotionCommand performs
the remaining runtime articulation-order and joint-limit checks before copying
the arrays to the simulator device.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA_VERSION = 2
ARTIFACT_KIND = "hope_post_swing_teacher_state_receipt"
ATTESTATION_KIND = "hope_post_swing_teacher_capture_attestation"
CAPTURE_RESULT_KIND = "hope_post_swing_natural_wrap_capture_result"
CAPTURE_CLAIM_KIND = "hope_post_swing_natural_wrap_exclusive_claim"
CAPTURE_RESULT_NAME = "natural_wrap_capture.json"
CAPTURE_STATE_NAME = "natural_wrap_states.npz"
CAPTURE_CLAIM_NAME = "natural_wrap_capture.claim.json"
CAPTURE_CONTRACT = {
    "event": "natural_clip_wrap",
    "wrap_teleport": False,
    "clip_switch_aborted_states_included": False,
    "root_position_frame": "environment_origin_relative",
    "root_state_layout": "pos3_quat_wxyz4_linear_velocity_com3_angular_velocity3",
    "joint_state_order": "runtime_articulation_joint_names",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "artifact_kind",
    "capture_contract",
    "teacher",
    "motion_clips",
    "states",
    "attestation",
}
_TEACHER_KEYS = {
    "source_commit",
    "checkpoint_sha256",
    "training_contract_sha256",
    "training_contract_schema_version",
    "fresh_lineage",
}
_STATE_KEYS = {
    "relative_path",
    "sha256",
    "count",
    "root_shape",
    "joint_pos_shape",
    "joint_vel_shape",
    "joint_names",
    "velocity_limits",
}
_VELOCITY_LIMIT_KEYS = {
    "root_linear_norm_max_mps",
    "root_angular_norm_max_radps",
    "joint_abs_max_radps",
}
_ATTESTATION_KEYS = {
    "schema_version",
    "artifact_kind",
    "capture_result_sha256",
    "capture_result_relative_path",
    "capture_claim_sha256",
    "capture_claim_relative_path",
    "checkpoint",
    "hard_contract",
    "checkpoint_source",
    "capture_source",
    "attestor_source",
    "retry_authorization",
}
_ATTESTED_CHECKPOINT_KEYS = {
    "sha256",
    "training_contract_schema_version",
    "training_contract_sha256",
    "training_contract_lineage_exact",
    "training_launch_claim_sha256",
}
_ATTESTED_HARD_CONTRACT_KEYS = {"sha256", "schema_version"}
_ATTESTED_CHECKPOINT_SOURCE_KEYS = {"commit", "launch_claim_content_sha256"}
_ATTESTED_CAPTURE_SOURCE_KEYS = {
    "commit",
    "clean",
    "producer_source_sha256",
}
_ATTESTED_ATTESTOR_SOURCE_KEYS = {
    "commit",
    "clean",
    "attestor_source_sha256",
}
_RETRY_AUTHORIZATION_KEYS = {
    "authorization_id",
    "file_sha256",
    "v3_plan_file_sha256",
}
_CAPTURE_EVIDENCE_KEYS = {
    "producer_source_sha256",
    "runtime_hard_contract_sha256",
    "exclusive_claim_sha256",
    "exclusive_claim_relative_path",
    "no_clobber",
}
_CAPTURE_CLAIM_KEYS = {
    "schema_version",
    "artifact_kind",
    "producer_source_sha256",
    "runtime_hard_contract_sha256",
    "target_count",
    "motion_clips",
    "joint_names",
    "exclusive_create",
}
_NPZ_KEYS = {
    "root_state_origin_relative",
    "joint_pos",
    "joint_vel",
}


class PostSwingTeacherError(RuntimeError):
    """The teacher receipt or its state payload violates the frozen contract."""


@dataclass(frozen=True)
class PostSwingTeacherStates:
    root_state_origin_relative: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    hard_contract: dict[str, Any]


def training_contract_extension(post_swing_replay: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only conditional schema-3 extension for receipt-backed replay.

    A default/receipt-free run returns the literal empty mapping.  The training contract builder
    splats this mapping, so historical/default JSON bytes do not gain null/default fields.
    """

    if post_swing_replay.get("teacher_receipt") is None:
        return {}
    return {"motion_post_swing_replay": dict(post_swing_replay)}


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _publish_bytes_no_clobber(path: Path, raw: bytes, label: str) -> None:
    """Durably publish bytes without ever replacing an existing path."""

    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise PostSwingTeacherError(f"{label} parent is not a regular directory: {parent}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temp = Path(temp_name)
    published = False
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise PostSwingTeacherError(f"{label} already exists; overwrite is forbidden: {path}") from exc
        published = True
        dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        if published:
            dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(_read_regular_file_once(Path(path), "content-addressed file")).hexdigest()


def _read_regular_file_once(path: Path, label: str) -> bytes:
    """Read one immutable byte snapshot through one ``O_NOFOLLOW`` descriptor.

    Hashing and parsing must consume the returned buffer; callers must never reopen ``path``.
    The before/after descriptor receipts reject in-place writers racing the read, while the
    immutable buffer prevents a later rename or rewrite from changing what is parsed.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PostSwingTeacherError(f"cannot open {label} without following links: {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PostSwingTeacherError(f"{label} is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise PostSwingTeacherError(f"{label} changed while its immutable byte snapshot was read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise PostSwingTeacherError(f"{label} byte count differs from descriptor size")
    return raw


def _strict_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8", "strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PostSwingTeacherError(f"{label} contains non-finite JSON {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostSwingTeacherError(f"cannot parse {label}: {exc}") from exc


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PostSwingTeacherError(f"teacher receipt has duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PostSwingTeacherError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise PostSwingTeacherError(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _require_sha(value: Any, label: str, pattern=_SHA256) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PostSwingTeacherError(f"{label} is not a lowercase content digest")
    return value


def _regular_non_symlink(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PostSwingTeacherError(f"{label} is not a regular non-symlink file: {path}")


def _payload_path(receipt_path: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise PostSwingTeacherError("states.relative_path must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise PostSwingTeacherError("states.relative_path must not escape the receipt directory")
    current = receipt_path.parent
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise PostSwingTeacherError(f"state payload path contains a symlink: {current}")
    _regular_non_symlink(current, "state payload")
    try:
        current.resolve().relative_to(receipt_path.parent.resolve())
    except ValueError as exc:
        raise PostSwingTeacherError("state payload escapes the receipt directory") from exc
    return current


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PostSwingTeacherError(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise PostSwingTeacherError(f"{label} must be a finite positive JSON float")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise PostSwingTeacherError(f"{label} must be a JSON boolean")
    return value


def _shape(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PostSwingTeacherError(f"{label} must be a non-empty integer list")
    return [_positive_int(item, f"{label} item") for item in value]


def _motion_contract(
    rows: Any, expected_motion_sha256: Sequence[str]
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(expected_motion_sha256):
        raise PostSwingTeacherError(
            "receipt motion_clips must match the runtime clip count exactly"
        )
    normalized = []
    for index, (row, expected_sha) in enumerate(zip(rows, expected_motion_sha256)):
        row = _require_exact_keys(row, {"index", "sha256"}, f"motion_clips[{index}]")
        if type(row["index"]) is not int or row["index"] != index:
            raise PostSwingTeacherError("receipt motion clip indexes must be contiguous and ordered")
        actual_sha = _require_sha(row["sha256"], f"motion_clips[{index}].sha256")
        expected_sha = _require_sha(expected_sha, f"expected motion SHA {index}")
        if actual_sha != expected_sha:
            raise PostSwingTeacherError(
                f"teacher motion clip {index} does not match runtime motion bytes"
            )
        normalized.append({"index": index, "sha256": actual_sha})
    return normalized


def load_post_swing_teacher_states(
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    *,
    expected_motion_sha256: Sequence[str],
    expected_joint_names: Sequence[str],
    expected_joint_velocity_limits: Sequence[float],
    expected_root_linear_velocity_limit_mps: float,
    expected_root_angular_velocity_limit_radps: float,
    min_fill: int,
    buffer_size: int,
) -> PostSwingTeacherStates:
    """Load and validate one immutable natural-wrap teacher-state artifact."""

    receipt_path = Path(receipt_path)
    expected_receipt_sha256 = _require_sha(
        expected_receipt_sha256, "expected teacher receipt SHA-256"
    )
    receipt_bytes = _read_regular_file_once(receipt_path, "teacher receipt")
    actual_receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if actual_receipt_sha256 != expected_receipt_sha256:
        raise PostSwingTeacherError(
            "teacher receipt byte SHA-256 does not match the configured digest"
        )
    document = _strict_json_bytes(receipt_bytes, "teacher receipt")

    document = _require_exact_keys(document, _TOP_LEVEL_KEYS, "teacher receipt")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise PostSwingTeacherError("unsupported post-swing teacher receipt schema")
    if document["artifact_kind"] != ARTIFACT_KIND:
        raise PostSwingTeacherError("wrong post-swing teacher artifact kind")
    if document["capture_contract"] != CAPTURE_CONTRACT:
        raise PostSwingTeacherError(
            "teacher states are not bound to the exact natural-clip-wrap capture contract"
        )

    teacher = _require_exact_keys(document["teacher"], _TEACHER_KEYS, "teacher")
    source_commit = _require_sha(
        teacher["source_commit"], "teacher.source_commit", pattern=_COMMIT
    )
    checkpoint_sha = _require_sha(
        teacher["checkpoint_sha256"], "teacher.checkpoint_sha256"
    )
    training_contract_sha = _require_sha(
        teacher["training_contract_sha256"], "teacher.training_contract_sha256"
    )
    if type(teacher["training_contract_schema_version"]) is not int or teacher["training_contract_schema_version"] != 3:
        raise PostSwingTeacherError("teacher must bind an exact schema-3 training contract")
    if _exact_bool(teacher["fresh_lineage"], "teacher.fresh_lineage") is not True:
        raise PostSwingTeacherError("teacher must have explicit fresh lineage")

    attestation = _require_exact_keys(document["attestation"], _ATTESTATION_KEYS, "attestation")
    if type(attestation["schema_version"]) is not int or attestation["schema_version"] != 2:
        raise PostSwingTeacherError("unsupported teacher attestation schema")
    if attestation["artifact_kind"] != ATTESTATION_KIND:
        raise PostSwingTeacherError("wrong teacher attestation kind")
    capture_result_sha = _require_sha(
        attestation["capture_result_sha256"], "attestation.capture_result_sha256"
    )
    if attestation["capture_result_relative_path"] != CAPTURE_RESULT_NAME:
        raise PostSwingTeacherError("attestation does not bind the fixed no-clobber capture result")
    capture_result_path = _payload_path(
        receipt_path, attestation["capture_result_relative_path"]
    )
    capture_result_bytes = _read_regular_file_once(
        capture_result_path, "natural-wrap capture result"
    )
    if hashlib.sha256(capture_result_bytes).hexdigest() != capture_result_sha:
        raise PostSwingTeacherError("natural-wrap capture result SHA-256 mismatch")
    capture_result = _strict_json_bytes(capture_result_bytes, "natural-wrap capture result")
    capture_claim_sha = _require_sha(
        attestation["capture_claim_sha256"], "attestation.capture_claim_sha256"
    )
    if attestation["capture_claim_relative_path"] != CAPTURE_CLAIM_NAME:
        raise PostSwingTeacherError("attestation does not bind the fixed exclusive capture claim")
    capture_claim_path = _payload_path(
        receipt_path, attestation["capture_claim_relative_path"]
    )
    capture_claim_bytes = _read_regular_file_once(
        capture_claim_path, "natural-wrap exclusive capture claim"
    )
    if hashlib.sha256(capture_claim_bytes).hexdigest() != capture_claim_sha:
        raise PostSwingTeacherError("natural-wrap exclusive claim SHA-256 mismatch")
    capture_claim = _strict_json_bytes(
        capture_claim_bytes, "natural-wrap exclusive capture claim"
    )
    checkpoint_att = _require_exact_keys(
        attestation["checkpoint"], _ATTESTED_CHECKPOINT_KEYS, "attestation.checkpoint"
    )
    hard_att = _require_exact_keys(
        attestation["hard_contract"], _ATTESTED_HARD_CONTRACT_KEYS, "attestation.hard_contract"
    )
    checkpoint_source = _require_exact_keys(
        attestation["checkpoint_source"],
        _ATTESTED_CHECKPOINT_SOURCE_KEYS,
        "attestation.checkpoint_source",
    )
    capture_source = _require_exact_keys(
        attestation["capture_source"],
        _ATTESTED_CAPTURE_SOURCE_KEYS,
        "attestation.capture_source",
    )
    attestor_source = _require_exact_keys(
        attestation["attestor_source"],
        _ATTESTED_ATTESTOR_SOURCE_KEYS,
        "attestation.attestor_source",
    )
    retry_authorization = _require_exact_keys(
        attestation["retry_authorization"],
        _RETRY_AUTHORIZATION_KEYS,
        "attestation.retry_authorization",
    )
    for label, value, pattern in (
        ("attested checkpoint", checkpoint_att["sha256"], _SHA256),
        ("attested checkpoint contract", checkpoint_att["training_contract_sha256"], _SHA256),
        ("attested checkpoint claim", checkpoint_att["training_launch_claim_sha256"], _SHA256),
        ("attested hard contract", hard_att["sha256"], _SHA256),
        ("checkpoint source commit", checkpoint_source["commit"], _COMMIT),
        ("checkpoint source claim", checkpoint_source["launch_claim_content_sha256"], _SHA256),
        ("capture source commit", capture_source["commit"], _COMMIT),
        ("capture producer source", capture_source["producer_source_sha256"], _SHA256),
        ("attestor source commit", attestor_source["commit"], _COMMIT),
        ("attestor source bytes", attestor_source["attestor_source_sha256"], _SHA256),
        ("retry authorization file", retry_authorization["file_sha256"], _SHA256),
        ("retry authorization v3 plan", retry_authorization["v3_plan_file_sha256"], _SHA256),
    ):
        _require_sha(value, label, pattern=pattern)
    if (
        type(retry_authorization["authorization_id"]) is not str
        or not retry_authorization["authorization_id"]
    ):
        raise PostSwingTeacherError("retry authorization id must be non-empty text")
    if (
        type(checkpoint_att["training_contract_schema_version"]) is not int
        or checkpoint_att["training_contract_schema_version"] != 3
        or _exact_bool(
            checkpoint_att["training_contract_lineage_exact"],
            "attestation.checkpoint.training_contract_lineage_exact",
        )
        is not True
        or type(hard_att["schema_version"]) is not int
        or hard_att["schema_version"] != 3
        or _exact_bool(capture_source["clean"], "attestation.capture_source.clean") is not True
        or _exact_bool(attestor_source["clean"], "attestation.attestor_source.clean") is not True
    ):
        raise PostSwingTeacherError("teacher attestation is not exact schema-3 clean lineage")
    if (
        checkpoint_att["sha256"] != checkpoint_sha
        or checkpoint_att["training_contract_sha256"] != training_contract_sha
        or hard_att["sha256"] != training_contract_sha
        or checkpoint_source["commit"] != source_commit
        or checkpoint_source["launch_claim_content_sha256"]
        != checkpoint_att["training_launch_claim_sha256"]
    ):
        raise PostSwingTeacherError("teacher self-summary differs from the one-shot attestation")

    motion_clips = _motion_contract(document["motion_clips"], expected_motion_sha256)
    states = _require_exact_keys(document["states"], _STATE_KEYS, "states")
    count = _positive_int(states["count"], "states.count")
    min_fill = _positive_int(min_fill, "post_swing_min_fill")
    buffer_size = _positive_int(buffer_size, "post_swing_buffer_size")
    if count < min_fill:
        raise PostSwingTeacherError(
            f"teacher state count {count} is below post_swing_min_fill {min_fill}"
        )
    if count > buffer_size:
        raise PostSwingTeacherError(
            f"teacher state count {count} exceeds post_swing_buffer_size {buffer_size}"
        )

    joint_names = states["joint_names"]
    expected_joint_names = [str(name) for name in expected_joint_names]
    if (
        not isinstance(joint_names, list)
        or any(not isinstance(name, str) or not name for name in joint_names)
        or len(set(joint_names)) != len(joint_names)
        or joint_names != expected_joint_names
    ):
        raise PostSwingTeacherError(
            "teacher joint_names/order does not match the runtime articulation"
        )
    expected_shapes = {
        "root_shape": [count, 13],
        "joint_pos_shape": [count, len(joint_names)],
        "joint_vel_shape": [count, len(joint_names)],
    }
    for key, expected in expected_shapes.items():
        if _shape(states[key], f"states.{key}") != expected:
            raise PostSwingTeacherError(
                f"states.{key} does not match the declared count/runtime joint order"
            )

    velocity_limits = _require_exact_keys(
        states["velocity_limits"], _VELOCITY_LIMIT_KEYS, "states.velocity_limits"
    )
    root_lin_limit = _positive_float(
        velocity_limits["root_linear_norm_max_mps"],
        "states.velocity_limits.root_linear_norm_max_mps",
    )
    root_ang_limit = _positive_float(
        velocity_limits["root_angular_norm_max_radps"],
        "states.velocity_limits.root_angular_norm_max_radps",
    )
    expected_root_linear_velocity_limit_mps = _positive_float(
        expected_root_linear_velocity_limit_mps,
        "runtime root linear velocity limit",
    )
    expected_root_angular_velocity_limit_radps = _positive_float(
        expected_root_angular_velocity_limit_radps,
        "runtime root angular velocity limit",
    )
    if (
        root_lin_limit != expected_root_linear_velocity_limit_mps
        or root_ang_limit != expected_root_angular_velocity_limit_radps
    ):
        raise PostSwingTeacherError("teacher root velocity limits differ from runtime limits")
    joint_limits = velocity_limits["joint_abs_max_radps"]
    if (
        not isinstance(joint_limits, list)
        or len(joint_limits) != len(joint_names)
        or any(type(value) is not float or not math.isfinite(value) or value <= 0 for value in joint_limits)
    ):
        raise PostSwingTeacherError("teacher joint velocity limits must be positive JSON floats")
    runtime_joint_limits = [float(value) for value in expected_joint_velocity_limits]
    if joint_limits != runtime_joint_limits:
        raise PostSwingTeacherError("teacher joint velocity limits differ from runtime plant limits")

    capture_claim = _require_exact_keys(
        capture_claim, _CAPTURE_CLAIM_KEYS, "natural-wrap exclusive capture claim"
    )
    if (
        type(capture_claim["schema_version"]) is not int
        or capture_claim["schema_version"] != 1
        or capture_claim["artifact_kind"] != CAPTURE_CLAIM_KIND
        or capture_claim["producer_source_sha256"]
        != capture_source["producer_source_sha256"]
        or capture_claim["runtime_hard_contract_sha256"] != hard_att["sha256"]
        or _positive_int(capture_claim["target_count"], "capture claim target_count") != count
        or capture_claim["motion_clips"] != motion_clips
        or capture_claim["joint_names"] != joint_names
        or _exact_bool(
            capture_claim["exclusive_create"], "capture claim exclusive_create"
        )
        is not True
    ):
        raise PostSwingTeacherError(
            "teacher receipt differs from its source-bound exclusive capture claim"
        )

    capture_result = _require_exact_keys(
        capture_result,
        {
            "schema_version", "artifact_kind", "capture_contract", "evidence",
            "motion_clips", "states",
        },
        "natural-wrap capture result",
    )
    evidence = _require_exact_keys(
        capture_result["evidence"],
        _CAPTURE_EVIDENCE_KEYS,
        "natural-wrap capture evidence",
    )
    if (
        type(capture_result["schema_version"]) is not int
        or capture_result["schema_version"] != 2
        or capture_result["artifact_kind"] != CAPTURE_RESULT_KIND
        or capture_result["capture_contract"] != CAPTURE_CONTRACT
        or evidence["producer_source_sha256"] != capture_source["producer_source_sha256"]
        or evidence["runtime_hard_contract_sha256"] != hard_att["sha256"]
        or evidence["exclusive_claim_sha256"] != capture_claim_sha
        or evidence["exclusive_claim_relative_path"] != CAPTURE_CLAIM_NAME
        or _exact_bool(evidence["no_clobber"], "capture evidence no_clobber") is not True
        or capture_result["motion_clips"] != motion_clips
        or capture_result["states"]
        != {key: value for key, value in states.items() if key != "velocity_limits"}
    ):
        raise PostSwingTeacherError(
            "teacher receipt differs from its reviewed no-clobber natural-wrap capture result"
        )

    payload_path = _payload_path(receipt_path, states["relative_path"])
    payload_sha = _require_sha(states["sha256"], "states.sha256")
    payload_bytes = _read_regular_file_once(payload_path, "teacher state payload")
    if hashlib.sha256(payload_bytes).hexdigest() != payload_sha:
        raise PostSwingTeacherError("teacher state payload byte SHA-256 mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(payload_bytes), "r") as archive:
            members = [item.filename for item in archive.infolist()]
        expected_members = {f"{key}.npy" for key in _NPZ_KEYS}
        if len(members) != len(set(members)):
            raise PostSwingTeacherError("teacher state payload contains duplicate NPZ ZIP keys")
        if set(members) != expected_members:
            raise PostSwingTeacherError("teacher state payload ZIP members differ from numeric schema")
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as payload:
            if set(payload.files) != _NPZ_KEYS:
                raise PostSwingTeacherError(
                    "teacher state payload keys differ from the frozen numeric schema"
                )
            arrays = {
                key: np.asarray(payload[key])
                for key in sorted(_NPZ_KEYS)
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, PostSwingTeacherError):
            raise
        raise PostSwingTeacherError(f"cannot load teacher state payload: {exc}") from exc

    root = arrays["root_state_origin_relative"]
    joint_pos = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    for key, array, expected_shape in (
        ("root_state_origin_relative", root, tuple(expected_shapes["root_shape"])),
        ("joint_pos", joint_pos, tuple(expected_shapes["joint_pos_shape"])),
        ("joint_vel", joint_vel, tuple(expected_shapes["joint_vel_shape"])),
    ):
        if array.dtype != np.float32:
            raise PostSwingTeacherError(f"{key} must use exact float32 storage")
        if array.shape != expected_shape:
            raise PostSwingTeacherError(
                f"{key} shape {array.shape} does not match {expected_shape}"
            )
        if not np.isfinite(array).all():
            raise PostSwingTeacherError(f"{key} contains NaN or Inf")

    quat_norm = np.linalg.norm(root[:, 3:7].astype(np.float64), axis=1)
    if any(not math.isfinite(float(value)) for value in quat_norm) or not np.allclose(
        quat_norm, 1.0, rtol=0.0, atol=1.0e-4
    ):
        raise PostSwingTeacherError("teacher root quaternions are not unit quaternions")
    if np.any(np.linalg.norm(root[:, 7:10].astype(np.float64), axis=1) > root_lin_limit):
        raise PostSwingTeacherError("teacher root linear velocity exceeds the runtime limit")
    if np.any(np.linalg.norm(root[:, 10:13].astype(np.float64), axis=1) > root_ang_limit):
        raise PostSwingTeacherError("teacher root angular velocity exceeds the runtime limit")
    if np.any(np.abs(joint_vel.astype(np.float64)) > np.asarray(joint_limits)[None, :]):
        raise PostSwingTeacherError("teacher joint velocity exceeds runtime plant limits")

    hard_contract = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": actual_receipt_sha256,
        "state_payload_basename": payload_path.name,
        "state_payload_sha256": payload_sha,
        "state_count": count,
        "capture_contract": dict(CAPTURE_CONTRACT),
        "teacher": {
            "source_commit": source_commit,
            "checkpoint_sha256": checkpoint_sha,
            "training_contract_sha256": training_contract_sha,
            "training_contract_schema_version": 3,
            "fresh_lineage": True,
        },
        "attestation": {
            "capture_result_sha256": capture_result_sha,
            "capture_claim_sha256": capture_claim_sha,
            "checkpoint": dict(checkpoint_att),
            "hard_contract": dict(hard_att),
            "checkpoint_source": dict(checkpoint_source),
            "capture_source": dict(capture_source),
            "attestor_source": dict(attestor_source),
            "retry_authorization": dict(retry_authorization),
        },
        "motion_clips": motion_clips,
        "joint_names": list(joint_names),
        "velocity_limits": {
            "root_linear_norm_max_mps": root_lin_limit,
            "root_angular_norm_max_radps": root_ang_limit,
            "joint_abs_max_radps": list(joint_limits),
        },
    }
    return PostSwingTeacherStates(
        root_state_origin_relative=np.array(root, copy=True),
        joint_pos=np.array(joint_pos, copy=True),
        joint_vel=np.array(joint_vel, copy=True),
        hard_contract=hard_contract,
    )
