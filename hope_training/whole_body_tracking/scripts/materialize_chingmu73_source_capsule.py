#!/usr/bin/env python3
"""Package the exact ChingMu N=73 source inventory without granting admission.

The historical ChingMu batch manifest binds each converted motion, but it does
not bind the per-unit metadata JSON or ball NPZ and its ball paths are absolute
Pod paths.  This tool closes that source-transport gap only.  It:

* reopens caller-pinned action, build-report, and batch manifest bytes;
* requires the exact ordered 73-action view (batch v1 minus Take_085);
* cross-checks motion, strike timing, base spawn, metadata, and ball samples;
* hashes and copies every motion, metadata sidecar, ball sidecar, and prototype;
* publishes one portable, no-clobber source capsule and inventory receipt.

It deliberately does not compile motions, mint motion admission, or authorize
training/deployment.  The receipt records all of those flags as false.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_COUNT = 73
BATCH_UNIT_COUNT = 74
EXCLUDED_UID = "Take_085_unit00_FH"
EXPECTED_MANIFEST_ID = "action_ball_chingmu73_nomove_f10_20260728"
EXPECTED_BATCH_SCHEMA = "chingmu_manifest_v1"
MOTION_DIRECTORY = PurePosixPath("motions/chingmu73_20260728")
RECEIPT_NAME = "SOURCE_CAPSULE_RECEIPT.json"
RECEIPT_TYPE = "chingmu73_source_capsule_v1"
CONSUMER_INTERFACE = "canonical_arbitrary_n_source_capsule_v1"
MOTION_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "kinematics_schema_version",
    "body_pos_point",
    "body_lin_vel_point",
    "body_names",
)
BALL_KEYS = (
    "ball_real_hope_m",
    "ball_synth_hope_m",
    "unit_offset",
    "n_unit",
    "src_range_120",
    "hit_frames_ext",
    "fps",
)
EXPECTED_BODY_NAMES = (
    "pelvis_link",
    "left_hip_pitch_Link",
    "right_hip_pitch_Link",
    "waist_yaw_Link",
    "left_hip_roll_Link",
    "right_hip_roll_Link",
    "waist_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "torso_Link",
    "left_knee_Link",
    "right_knee_Link",
    "head_yaw_Link",
    "left_shoulder_pitch_Link",
    "right_shoulder_pitch_Link",
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "head_pitch_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_shoulder_yaw_Link",
    "right_shoulder_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
    "left_wrist_roll_Link",
    "right_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "right_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
)
_UID = re.compile(r"[A-Za-z0-9_]+")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_AT_FDCWD = -2
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


class ChingMu73CapsuleError(RuntimeError):
    """The inputs cannot form the exact, portable N=73 source capsule."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ChingMu73CapsuleError(
            f"{label} must be exactly 64 lowercase SHA-256 hex digits"
        )
    return value


def _absolute(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _real_directory(path: os.PathLike[str] | str, label: str) -> Path:
    result = _absolute(path)
    try:
        metadata = result.lstat()
    except OSError as exc:
        raise ChingMu73CapsuleError(
            f"cannot inspect {label} {result}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or result.is_symlink():
        raise ChingMu73CapsuleError(
            f"{label} must be a real non-symlink directory: {result}"
        )
    return result


def _read_regular(path: Path, label: str) -> bytes:
    """Read one stable regular file without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise ChingMu73CapsuleError(
                f"{label} must be a regular non-symlink file: {path}"
            )
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except ChingMu73CapsuleError:
        raise
    except OSError as exc:
        raise ChingMu73CapsuleError(
            f"cannot read {label} {path}: {exc}"
        ) from exc
    identities = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    )
    if identities[0] != identities[1] or identities[1] != identities[2]:
        raise ChingMu73CapsuleError(f"{label} changed during stable read: {path}")
    return b"".join(chunks)


def _strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ChingMu73CapsuleError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ChingMu73CapsuleError(
            f"{label} contains non-finite JSON constant {value}"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except ChingMu73CapsuleError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ChingMu73CapsuleError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ChingMu73CapsuleError(f"{label} must contain one JSON object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ChingMu73CapsuleError(
            f"source capsule receipt is not strict JSON: {exc}"
        ) from exc


def _bound_json(
    path: os.PathLike[str] | str,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes, Mapping[str, Any], str]:
    source = _absolute(path)
    payload = _read_regular(source, label)
    actual = _sha256(payload)
    expected = _digest(expected_sha256, f"expected {label} SHA-256")
    if actual != expected:
        raise ChingMu73CapsuleError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return source, payload, _strict_json(payload, label), actual


def _normalized_relative(value: Any, label: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise ChingMu73CapsuleError(
            f"{label} must be one normalized relative POSIX path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ChingMu73CapsuleError(f"{label} may not contain '.' or '..'")
    return path


def _uid(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UID.fullmatch(value) is None:
        raise ChingMu73CapsuleError(
            f"{label} must contain only ASCII letters, digits, and underscores"
        )
    return value


def _sequence(value: Any, count: int, label: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != count
    ):
        raise ChingMu73CapsuleError(
            f"{label} must contain exactly {count} entries"
        )
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChingMu73CapsuleError(f"{label} must be one finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ChingMu73CapsuleError(f"{label} must be one finite number")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChingMu73CapsuleError(f"{label} must be one integer")
    return value


def _close(actual: Any, expected: Any, label: str, tolerance: float = 1e-8) -> None:
    left = _number(actual, label)
    right = _number(expected, f"{label} reference")
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise ChingMu73CapsuleError(
            f"{label} mismatch: expected {right}, got {left}"
        )


def _vector_close(
    actual: Any,
    expected: Any,
    count: int,
    label: str,
    tolerance: float = 1e-8,
) -> None:
    left = _sequence(actual, count, label)
    right = _sequence(expected, count, f"{label} reference")
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        _close(lhs, rhs, f"{label}[{index}]", tolerance)


def _npz(payload: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise ChingMu73CapsuleError(f"cannot load {label} as safe NPZ: {exc}") from exc


def _scalar(array: np.ndarray, label: str) -> float:
    if array.size != 1:
        raise ChingMu73CapsuleError(f"{label} must contain exactly one scalar")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ChingMu73CapsuleError(f"{label} must be finite")
    return result


def _validate_motion_npz(
    payload: bytes,
    *,
    unit: Mapping[str, Any],
    label: str,
) -> None:
    arrays = _npz(payload, label)
    if tuple(arrays) != MOTION_KEYS:
        raise ChingMu73CapsuleError(
            f"{label} keys/order must be exact schema-2 {MOTION_KEYS}, "
            f"got {tuple(arrays)}"
        )
    total = _integer(unit.get("T"), f"{label} batch T")
    if _scalar(arrays["fps"], f"{label}.fps") != 50.0:
        raise ChingMu73CapsuleError(f"{label}.fps must be exactly 50")
    if _scalar(
        arrays["kinematics_schema_version"],
        f"{label}.kinematics_schema_version",
    ) != 2.0:
        raise ChingMu73CapsuleError(
            f"{label}.kinematics_schema_version must be exactly 2"
        )
    expected_shapes = {
        "joint_pos": (total, 31),
        "joint_vel": (total, 31),
        "body_pos_w": (total, 32, 3),
        "body_quat_w": (total, 32, 4),
        "body_lin_vel_w": (total, 32, 3),
        "body_ang_vel_w": (total, 32, 3),
        "body_names": (32,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ChingMu73CapsuleError(
                f"{label}.{key} shape must be {shape}, got {arrays[key].shape}"
            )
    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        if not np.isfinite(arrays[key]).all():
            raise ChingMu73CapsuleError(f"{label}.{key} contains non-finite values")
    quaternion_norm = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=2e-6):
        raise ChingMu73CapsuleError(
            f"{label}.body_quat_w contains non-unit quaternions"
        )
    if tuple(str(value) for value in arrays["body_names"].tolist()) != (
        EXPECTED_BODY_NAMES
    ):
        raise ChingMu73CapsuleError(
            f"{label}.body_names does not match the exact 32-body contract"
        )
    if str(arrays["body_pos_point"].reshape(-1)[0]) != "link_origin":
        raise ChingMu73CapsuleError(f"{label}.body_pos_point must be link_origin")
    if str(arrays["body_lin_vel_point"].reshape(-1)[0]) != "center_of_mass":
        raise ChingMu73CapsuleError(
            f"{label}.body_lin_vel_point must be center_of_mass"
        )


def _validate_ball_npz(
    payload: bytes,
    *,
    unit: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    arrays = _npz(payload, label)
    if tuple(arrays) != BALL_KEYS:
        raise ChingMu73CapsuleError(
            f"{label} keys/order must be {BALL_KEYS}, got {tuple(arrays)}"
        )
    real = arrays["ball_real_hope_m"]
    synthetic = arrays["ball_synth_hope_m"]
    if real.ndim != 2 or real.shape[1:] != (3,) or synthetic.shape != real.shape:
        raise ChingMu73CapsuleError(
            f"{label} real/synthetic ball arrays must both have shape (N, 3)"
        )
    if _scalar(arrays["fps"], f"{label}.fps") != 120.0:
        raise ChingMu73CapsuleError(f"{label}.fps must be exactly 120")
    offset = int(_scalar(arrays["unit_offset"], f"{label}.unit_offset"))
    count = int(_scalar(arrays["n_unit"], f"{label}.n_unit"))
    hit_frames = arrays["hit_frames_ext"]
    if hit_frames.ndim != 1 or hit_frames.size < 1:
        raise ChingMu73CapsuleError(f"{label}.hit_frames_ext must be non-empty")
    first_hit = int(hit_frames[0])
    batch_hit = _integer(
        unit.get("hit_frame_ball_120"), f"{label} batch hit_frame_ball_120"
    )
    if first_hit - offset != batch_hit:
        raise ChingMu73CapsuleError(
            f"{label} first hit minus unit_offset must equal batch hit frame"
        )
    if not 0 <= offset <= offset + count <= real.shape[0]:
        raise ChingMu73CapsuleError(f"{label} unit slice is outside ball arrays")
    if not 0 <= first_hit < real.shape[0]:
        raise ChingMu73CapsuleError(f"{label} first hit is outside ball arrays")
    if not np.isfinite(real[first_hit]).all():
        raise ChingMu73CapsuleError(f"{label} real ball is missing at first hit")
    _vector_close(
        real[first_hit].tolist(),
        unit.get("ball_pos_hit_hope_m"),
        3,
        f"{label} hit ball",
        tolerance=1e-4,
    )
    finite = np.isfinite(real).all(axis=1)
    coverage_unit = float(finite[offset : offset + count].mean())
    pre_start = max(0, first_hit - 30)
    coverage_pre30 = float(finite[pre_start : first_hit + 1].mean())
    _close(
        coverage_unit,
        unit.get("ball_coverage_unit"),
        f"{label} unit coverage",
        tolerance=5e-4,
    )
    _close(
        coverage_pre30,
        unit.get("ball_coverage_pre30"),
        f"{label} pre30 coverage",
        tolerance=5e-4,
    )
    return {
        "unit_offset": offset,
        "n_unit": count,
        "first_hit_ext": first_hit,
        "coverage_unit": coverage_unit,
        "coverage_pre30": coverage_pre30,
    }


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    uid: str,
    unit: Mapping[str, Any],
    label: str,
) -> None:
    if metadata.get("clip") != uid:
        raise ChingMu73CapsuleError(f"{label}.clip must be {uid!r}")
    _close(metadata.get("source_fps"), 120.0, f"{label}.source_fps")
    _close(
        metadata.get("retime_factor"),
        unit.get("retime_factor"),
        f"{label}.retime_factor",
    )
    _vector_close(
        metadata.get("station_xy_hope_m"),
        unit.get("station_xy_hope_m"),
        2,
        f"{label}.station_xy_hope_m",
    )
    hits = metadata.get("hits")
    if not isinstance(hits, list) or not hits or not isinstance(hits[0], Mapping):
        raise ChingMu73CapsuleError(f"{label}.hits must contain a first hit object")
    hit = hits[0]
    if hit.get("frame_out") != unit.get("hit_frame_50"):
        raise ChingMu73CapsuleError(f"{label} first frame_out mismatches batch")
    if hit.get("frame_src120") != unit.get("hit_frame_src120"):
        raise ChingMu73CapsuleError(f"{label} first frame_src120 mismatches batch")
    _vector_close(
        hit.get("ball_hope_m"),
        unit.get("ball_pos_hit_hope_m"),
        3,
        f"{label} first ball_hope_m",
        tolerance=1e-4,
    )
    _vector_close(
        hit.get("v_in_fit_hope_ms"),
        unit.get("v_in_fit_hope_ms"),
        3,
        f"{label} first v_in_fit_hope_ms",
        tolerance=1e-7,
    )
    _vector_close(
        hit.get("v_out_fit_hope_ms"),
        unit.get("v_out_fit_hope_ms"),
        3,
        f"{label} first v_out_fit_hope_ms",
        tolerance=1e-7,
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as exc:
        raise ChingMu73CapsuleError(
            f"cannot write staged capsule file {path}: {exc}"
        ) from exc


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise OSError(errno.ENOSYS, "renameat2 unavailable", str(destination))
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise OSError(errno.ENOSYS, "renamex_np unavailable", str(destination))
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, _RENAME_EXCL)
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication unsupported",
            str(destination),
        )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def materialize(
    *,
    action_manifest_path: os.PathLike[str] | str,
    expected_action_manifest_sha256: str,
    build_report_path: os.PathLike[str] | str,
    expected_build_report_sha256: str,
    batch_manifest_path: os.PathLike[str] | str,
    expected_batch_manifest_sha256: str,
    profile_root: os.PathLike[str] | str,
    batch_root: os.PathLike[str] | str,
    motion_root: os.PathLike[str] | str,
    ball_root: os.PathLike[str] | str,
    output_directory: os.PathLike[str] | str,
) -> Mapping[str, Any]:
    """Validate and publish one exact N=73 source-only capsule."""

    profile = _real_directory(profile_root, "profile root")
    batch_assets = _real_directory(batch_root, "batch root")
    motion_assets = _real_directory(motion_root, "motion root")
    ball_assets = _real_directory(ball_root, "ball root")
    output = _absolute(output_directory)
    output_parent = _real_directory(output.parent, "output parent")
    if output.exists() or output.is_symlink():
        raise ChingMu73CapsuleError(
            f"output already exists; capsule publication is no-clobber: {output}"
        )

    (
        action_path,
        action_bytes,
        action_manifest,
        action_sha,
    ) = _bound_json(
        action_manifest_path,
        expected_action_manifest_sha256,
        "action manifest",
    )
    (
        report_path,
        report_bytes,
        build_report,
        report_sha,
    ) = _bound_json(
        build_report_path,
        expected_build_report_sha256,
        "build report",
    )
    (
        batch_path,
        batch_bytes,
        batch_manifest,
        batch_sha,
    ) = _bound_json(
        batch_manifest_path,
        expected_batch_manifest_sha256,
        "batch manifest",
    )

    if action_manifest.get("schema_version") != 3:
        raise ChingMu73CapsuleError("action manifest schema_version must be 3")
    if action_manifest.get("manifest_id") != EXPECTED_MANIFEST_ID:
        raise ChingMu73CapsuleError(
            f"action manifest id must be {EXPECTED_MANIFEST_ID!r}"
        )
    if action_manifest.get("mobility_mode") != "no_move":
        raise ChingMu73CapsuleError("ChingMu73 source view must be no_move")
    prototype = action_manifest.get("prototype")
    if not isinstance(prototype, Mapping) or prototype.get("scope") != "full":
        raise ChingMu73CapsuleError(
            "ChingMu73 prototype must declare the full motion scope"
        )
    prototype_relative = _normalized_relative(
        prototype.get("path"), "action manifest prototype.path"
    )
    prototype_path = profile.joinpath(*prototype_relative.parts)
    prototype_bytes = _read_regular(prototype_path, "prototype")
    prototype_sha = _sha256(prototype_bytes)
    if prototype_sha != _digest(
        prototype.get("sha256"), "action manifest prototype.sha256"
    ):
        raise ChingMu73CapsuleError("prototype bytes drifted from action manifest")

    action_order = tuple(
        _uid(value, f"action_order[{index}]")
        for index, value in enumerate(
            _sequence(
                action_manifest.get("action_order"),
                ACTION_COUNT,
                "action_order",
            )
        )
    )
    if len(set(action_order)) != ACTION_COUNT:
        raise ChingMu73CapsuleError("action_order must contain 73 unique ids")
    actions = _sequence(
        action_manifest.get("actions"), ACTION_COUNT, "action manifest actions"
    )
    if any(not isinstance(action, Mapping) for action in actions):
        raise ChingMu73CapsuleError("each action manifest action must be an object")
    if tuple(action.get("action_id") for action in actions) != action_order:
        raise ChingMu73CapsuleError(
            "action rows must exactly preserve declared action_order"
        )

    if batch_manifest.get("schema") != EXPECTED_BATCH_SCHEMA:
        raise ChingMu73CapsuleError(
            f"batch schema must be {EXPECTED_BATCH_SCHEMA!r}"
        )
    if batch_manifest.get("failures") != []:
        raise ChingMu73CapsuleError("batch failures must be exactly empty")
    units = _sequence(
        batch_manifest.get("units"), BATCH_UNIT_COUNT, "batch units"
    )
    if any(not isinstance(unit, Mapping) for unit in units):
        raise ChingMu73CapsuleError("each batch unit must be an object")
    unit_by_uid: dict[str, Mapping[str, Any]] = {}
    for index, unit in enumerate(units):
        uid = _uid(unit.get("uid"), f"batch units[{index}].uid")
        if uid in unit_by_uid:
            raise ChingMu73CapsuleError(f"duplicate batch uid {uid!r}")
        unit_by_uid[uid] = unit
    if EXCLUDED_UID not in unit_by_uid:
        raise ChingMu73CapsuleError(
            f"batch must contain excluded unit {EXCLUDED_UID!r}"
        )
    selected_units = tuple(
        unit for unit in units if unit.get("uid") != EXCLUDED_UID
    )
    expected_order = tuple(str(unit["uid"]).lower() for unit in selected_units)
    if action_order != expected_order:
        raise ChingMu73CapsuleError(
            "action order must be exact batch-v1 order minus Take_085"
        )

    if build_report.get("file_sha256") != action_sha:
        raise ChingMu73CapsuleError(
            "build report does not bind the exact action manifest"
        )
    if build_report.get("batch_manifest_sha256") != batch_sha:
        raise ChingMu73CapsuleError(
            "build report does not bind the exact batch manifest"
        )
    if build_report.get("n_actions") != ACTION_COUNT:
        raise ChingMu73CapsuleError("build report n_actions must be 73")
    if build_report.get("excluded_uids") != [EXCLUDED_UID]:
        raise ChingMu73CapsuleError(
            "build report must exclude exactly Take_085_unit00_FH"
        )

    expected_motion_names = {
        f"hope_{unit['uid']}.npz" for unit in selected_units
    }
    actual_motion_names: set[str] = set()
    for child in motion_assets.iterdir():
        metadata = child.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or child.is_symlink()
            or child.suffix != ".npz"
        ):
            raise ChingMu73CapsuleError(
                f"motion root may contain only regular .npz files: {child}"
            )
        actual_motion_names.add(child.name)
    if actual_motion_names != expected_motion_names:
        missing = sorted(expected_motion_names - actual_motion_names)
        extra = sorted(actual_motion_names - expected_motion_names)
        raise ChingMu73CapsuleError(
            f"motion root is not exact N=73; missing={missing}, extra={extra}"
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=".chingmu73-source-capsule-",
            dir=str(output_parent),
        )
    )
    receipt_rows: list[dict[str, Any]] = []
    try:
        _write(
            staging / "configs" / action_path.name,
            action_bytes,
        )
        _write(
            staging / "configs" / report_path.name,
            report_bytes,
        )
        _write(
            staging / "provenance" / batch_path.name,
            batch_bytes,
        )
        _write(staging.joinpath(*prototype_relative.parts), prototype_bytes)

        for index, (action, unit) in enumerate(zip(actions, selected_units)):
            uid = str(unit["uid"])
            action_id = str(action["action_id"])
            family = {"FH": "forehand", "BH": "backhand"}.get(unit.get("family"))
            if family is None or action.get("family") != family:
                raise ChingMu73CapsuleError(
                    f"{uid}: action family does not match batch family"
                )
            motion_name = f"hope_{uid}.npz"
            motion_relative = _normalized_relative(
                action.get("motion_path"), f"{uid} motion_path"
            )
            if motion_relative != MOTION_DIRECTORY / motion_name:
                raise ChingMu73CapsuleError(
                    f"{uid}: motion_path must be {MOTION_DIRECTORY / motion_name}"
                )
            expected_motion_sha = _digest(
                action.get("motion_sha256"), f"{uid} motion_sha256"
            )
            if unit.get("npz_sha256") != expected_motion_sha:
                raise ChingMu73CapsuleError(
                    f"{uid}: batch and action motion SHA-256 disagree"
                )
            if unit.get("npz") != f"clips/{motion_name}":
                raise ChingMu73CapsuleError(
                    f"{uid}: batch npz path does not match canonical clip name"
                )
            motion_bytes = _read_regular(
                motion_assets / motion_name, f"{uid} WIP motion"
            )
            if _sha256(motion_bytes) != expected_motion_sha:
                raise ChingMu73CapsuleError(
                    f"{uid}: WIP motion bytes drifted from manifests"
                )
            batch_motion_bytes = _read_regular(
                batch_assets / "clips" / motion_name,
                f"{uid} batch motion",
            )
            if batch_motion_bytes != motion_bytes:
                raise ChingMu73CapsuleError(
                    f"{uid}: batch and WIP motion bytes are not identical"
                )
            _validate_motion_npz(motion_bytes, unit=unit, label=f"{uid} motion")

            total = _integer(unit.get("T"), f"{uid} T")
            hit_frame = _integer(unit.get("hit_frame_50"), f"{uid} hit_frame_50")
            if not 0 < hit_frame < total - 1:
                raise ChingMu73CapsuleError(
                    f"{uid}: strike frame must lie strictly inside the clip"
                )
            _close(
                action.get("reference_t_hit_s"),
                hit_frame / 50.0,
                f"{uid} reference_t_hit_s",
            )
            _close(
                action.get("reference_t_cycle_s"),
                (total - 1) / 50.0,
                f"{uid} reference_t_cycle_s",
            )
            _close(
                action.get("strike_phase"),
                unit.get("strike_phase"),
                f"{uid} strike_phase",
                tolerance=5e-5,
            )
            selected_hit_frame = round(
                _number(
                    action.get("strike_phase"),
                    f"{uid} strike_phase",
                )
                * (total - 1)
            )
            if selected_hit_frame != hit_frame:
                raise ChingMu73CapsuleError(
                    f"{uid}: strike_phase selects a different hit frame"
                )
            station = _sequence(
                unit.get("station_xy_hope_m"), 2, f"{uid} station_xy_hope_m"
            )
            expected_spawn = [
                _number(station[0], f"{uid} station x") + 0.5,
                _number(station[1], f"{uid} station y") + 0.7625,
            ]
            ball_profile = action.get("ball_profile")
            if not isinstance(ball_profile, Mapping):
                raise ChingMu73CapsuleError(f"{uid}: ball_profile must be an object")
            _vector_close(
                ball_profile.get("base_spawn_center_w_xy_m"),
                expected_spawn,
                2,
                f"{uid} base_spawn_center_w_xy_m",
            )
            for key in (
                "base_travel_center_b_yaw_xy_m",
                "base_travel_std_lower_initial_m",
                "base_travel_std_lower_max_m",
                "base_travel_std_upper_initial_m",
                "base_travel_std_upper_max_m",
                "base_travel_min_b_yaw_xy_m",
                "base_travel_max_b_yaw_xy_m",
            ):
                _vector_close(
                    ball_profile.get(key),
                    [0.0, 0.0],
                    2,
                    f"{uid} no_move {key}",
                )

            metadata_name = f"hope_{uid}.meta.json"
            metadata_bytes = _read_regular(
                batch_assets / "clips" / metadata_name,
                f"{uid} metadata sidecar",
            )
            metadata = _strict_json(metadata_bytes, f"{uid} metadata sidecar")
            _validate_metadata(
                metadata, uid=uid, unit=unit, label=f"{uid} metadata"
            )

            original_ball_path = unit.get("ball_npz")
            if not isinstance(original_ball_path, str) or not Path(
                original_ball_path
            ).is_absolute():
                raise ChingMu73CapsuleError(
                    f"{uid}: batch ball_npz must preserve its original absolute path"
                )
            ball_name = f"{uid}.ball.npz"
            if Path(original_ball_path).name != ball_name:
                raise ChingMu73CapsuleError(
                    f"{uid}: ball_npz basename must be {ball_name}"
                )
            ball_bytes = _read_regular(
                ball_assets / ball_name, f"{uid} ball sidecar"
            )
            ball_summary = _validate_ball_npz(
                ball_bytes, unit=unit, label=f"{uid} ball"
            )

            metadata_relative = (
                PurePosixPath("metadata/chingmu73_20260728") / metadata_name
            )
            ball_relative = PurePosixPath("balls/chingmu73_20260728") / ball_name
            _write(staging.joinpath(*motion_relative.parts), motion_bytes)
            _write(staging.joinpath(*metadata_relative.parts), metadata_bytes)
            _write(staging.joinpath(*ball_relative.parts), ball_bytes)
            warnings = unit.get("warnings")
            if not isinstance(warnings, list) or any(
                not isinstance(warning, str) for warning in warnings
            ):
                raise ChingMu73CapsuleError(
                    f"{uid}: batch warnings must be a list of strings"
                )
            receipt_rows.append(
                {
                    "index": index,
                    "action_id": action_id,
                    "uid": uid,
                    "family": family,
                    "motion_path": motion_relative.as_posix(),
                    "motion_sha256": expected_motion_sha,
                    "metadata_path": metadata_relative.as_posix(),
                    "metadata_sha256": _sha256(metadata_bytes),
                    "ball_path": ball_relative.as_posix(),
                    "ball_sha256": _sha256(ball_bytes),
                    "original_ball_path": original_ball_path,
                    "T": total,
                    "fps": 50,
                    "hit_frame_50": hit_frame,
                    "reference_t_hit_s": hit_frame / 50.0,
                    "reference_t_cycle_s": (total - 1) / 50.0,
                    "strike_phase": _number(
                        action.get("strike_phase"), f"{uid} strike_phase"
                    ),
                    "base_spawn_center_w_xy_m": expected_spawn,
                    "ball_first_hit_ext_120": ball_summary["first_hit_ext"],
                    "ball_coverage_unit": ball_summary["coverage_unit"],
                    "ball_coverage_pre30": ball_summary["coverage_pre30"],
                    "warnings": list(warnings),
                }
            )

        receipt: dict[str, Any] = {
            "schema_version": 1,
            "receipt_type": RECEIPT_TYPE,
            "consumer_interface": CONSUMER_INTERFACE,
            "verdict": "PASS_SOURCE_INVENTORY_ONLY",
            "authorization": {
                "compiler_candidate_authorized": False,
                "motion_admission_present": False,
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            },
            "contract": {
                "action_count": ACTION_COUNT,
                "batch_unit_count": BATCH_UNIT_COUNT,
                "excluded_uids": [EXCLUDED_UID],
                "motion_scope_declared": "full",
                "ordered_action_identity_bound": True,
                "source_inventory_content_bound": True,
                "motion_bytes_modified": False,
                "metadata_bytes_modified": False,
                "ball_bytes_modified": False,
            },
            "inputs": {
                "action_manifest": {
                    "path": f"configs/{action_path.name}",
                    "sha256": action_sha,
                },
                "build_report": {
                    "path": f"configs/{report_path.name}",
                    "sha256": report_sha,
                },
                "batch_manifest": {
                    "path": f"provenance/{batch_path.name}",
                    "sha256": batch_sha,
                    "contains_original_absolute_paths": True,
                },
                "prototype": {
                    "path": prototype_relative.as_posix(),
                    "sha256": prototype_sha,
                    "scope": "full",
                },
                "solver_profile_sha256": _digest(
                    action_manifest.get("solver_profile_sha256"),
                    "solver_profile_sha256",
                ),
                "physics_profile_sha256": _digest(
                    action_manifest.get("physics_profile_sha256"),
                    "physics_profile_sha256",
                ),
            },
            "limitations": [
                "This capsule is source inventory, not canonical compiler output.",
                "No upper/full compiler matrix or BUILD_MANIFEST is present.",
                "No canonical-ready, bank-gate, registry, alignment, evidence, "
                "or motion-admission certificate is present.",
                "The preserved batch manifest contains historical absolute source "
                "paths; portable consumers must use the receipt paths.",
                "The retarget source PKLs are SHA-bound by the preserved batch "
                "manifest but are not copied into this motion+ball capsule.",
                "The solver/physics values are digest pins only; their source "
                "payloads are not contained in this capsule.",
            ],
            "actions": receipt_rows,
        }
        receipt_bytes = _json_bytes(receipt)
        _write(staging / RECEIPT_NAME, receipt_bytes)
        _rename_directory_noreplace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-manifest", required=True)
    parser.add_argument("--expected-action-manifest-sha256", required=True)
    parser.add_argument("--build-report", required=True)
    parser.add_argument("--expected-build-report-sha256", required=True)
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--expected-batch-manifest-sha256", required=True)
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--motion-root", required=True)
    parser.add_argument("--ball-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = materialize(
            action_manifest_path=args.action_manifest,
            expected_action_manifest_sha256=args.expected_action_manifest_sha256,
            build_report_path=args.build_report,
            expected_build_report_sha256=args.expected_build_report_sha256,
            batch_manifest_path=args.batch_manifest,
            expected_batch_manifest_sha256=args.expected_batch_manifest_sha256,
            profile_root=args.profile_root,
            batch_root=args.batch_root,
            motion_root=args.motion_root,
            ball_root=args.ball_root,
            output_directory=args.output,
        )
    except (ChingMu73CapsuleError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
