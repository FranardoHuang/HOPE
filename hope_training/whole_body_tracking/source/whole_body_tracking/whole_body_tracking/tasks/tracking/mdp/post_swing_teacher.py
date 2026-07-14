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
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = 1
ARTIFACT_KIND = "hope_post_swing_teacher_state_receipt"
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if isinstance(row["index"], bool) or row["index"] != index:
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
    min_fill: int,
    buffer_size: int,
) -> PostSwingTeacherStates:
    """Load and validate one immutable natural-wrap teacher-state artifact."""

    receipt_path = Path(receipt_path)
    _regular_non_symlink(receipt_path, "teacher receipt")
    expected_receipt_sha256 = _require_sha(
        expected_receipt_sha256, "expected teacher receipt SHA-256"
    )
    actual_receipt_sha256 = sha256_file(receipt_path)
    if actual_receipt_sha256 != expected_receipt_sha256:
        raise PostSwingTeacherError(
            "teacher receipt byte SHA-256 does not match the configured digest"
        )
    try:
        document = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PostSwingTeacherError(f"teacher receipt contains non-finite JSON {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostSwingTeacherError(f"cannot parse teacher receipt: {exc}") from exc

    document = _require_exact_keys(document, _TOP_LEVEL_KEYS, "teacher receipt")
    if document["schema_version"] != SCHEMA_VERSION:
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
    if teacher["training_contract_schema_version"] != 3:
        raise PostSwingTeacherError("teacher must bind an exact schema-3 training contract")
    if teacher["fresh_lineage"] is not True:
        raise PostSwingTeacherError("teacher must have explicit fresh lineage")

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

    payload_path = _payload_path(receipt_path, states["relative_path"])
    payload_sha = _require_sha(states["sha256"], "states.sha256")
    if sha256_file(payload_path) != payload_sha:
        raise PostSwingTeacherError("teacher state payload byte SHA-256 mismatch")
    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            if set(payload.files) != _NPZ_KEYS:
                raise PostSwingTeacherError(
                    "teacher state payload keys differ from the frozen numeric schema"
                )
            arrays = {
                key: np.asarray(payload[key])
                for key in sorted(_NPZ_KEYS)
            }
    except (OSError, ValueError) as exc:
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
        "motion_clips": motion_clips,
        "joint_names": list(joint_names),
    }
    return PostSwingTeacherStates(
        root_state_origin_relative=np.array(root, copy=True),
        joint_pos=np.array(joint_pos, copy=True),
        joint_vel=np.array(joint_vel, copy=True),
        hard_contract=hard_contract,
    )
