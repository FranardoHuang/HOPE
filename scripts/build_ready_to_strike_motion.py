#!/usr/bin/env python3
"""Build a host-only shared-ready -> strike motion candidate.

The tool replaces an early part of a schema-2 motion with the selected motion's
frame-0 pose at explicitly zero velocity, followed by a quintic (C2) join into
the original motion.  It never
claims that interpolated link poses are production FK, and it deliberately keeps
training disabled until the result separately passes TOPP, L0, vendor L1,
table/net clearance, and dynamics gates.

The source pose/body-velocity interval beginning at ``join_frame`` is copied
byte-for-byte.  ``joint_vel`` is instead rebuilt with the schema-2 producer's
exact gradient stencil, so only the protected 0.1 second pre-contact window
(including its joint velocity) is promised bitwise identical to the source.
Publication is a two-file, no-clobber bundle: ``.npz`` plus a JSON contract that
binds the exact input and output SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA2_TIME_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
SCHEMA2_METADATA_KEYS = (
    "fps",
    "kinematics_schema_version",
    "body_pos_point",
    "body_lin_vel_point",
    "body_names",
)
SCHEMA2_MIGRATION_PROVENANCE_KEYS = (
    "kinematics_migration_source_sha256",
    "kinematics_migration_source_point",
    "kinematics_migration_tool",
)
REQUIRED_KEYS = frozenset(SCHEMA2_TIME_KEYS + SCHEMA2_METADATA_KEYS)
ALLOWED_KEYS = REQUIRED_KEYS | frozenset(SCHEMA2_MIGRATION_PROVENANCE_KEYS)
CANONICAL_MIGRATION_TOOL = "migrate_motion_kinematics.py/v2"
CONTRACT_SCHEMA_VERSION = 1
PROTECTED_PRECONTACT_SECONDS = 0.1


class MotionBuildError(ValueError):
    """A source, synthesis, or no-clobber publication contract failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def ensure_no_symlink_components(
    path: Path, label: str, *, leaf_may_be_missing: bool = False
) -> None:
    """Reject symlinks in every existing path component without resolving them."""

    absolute = _absolute(path)
    parts = absolute.parts
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if leaf_may_be_missing and is_leaf:
                return
            raise MotionBuildError(f"{label} path component is missing: {current}") from None
        if stat.S_ISLNK(info.st_mode):
            raise MotionBuildError(f"{label} must not traverse a symlink: {current}")


def read_regular_snapshot(path_like: Path | str, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    path = _absolute(Path(path_like))
    ensure_no_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise MotionBuildError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise MotionBuildError(f"{label} must be a non-empty regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        signature = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if signature(before) != signature(after):
            raise MotionBuildError(f"{label} changed while reading: {path}")
    finally:
        os.close(fd)
    payload = b"".join(chunks)
    evidence = {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
    }
    return path, payload, evidence


def _scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.hasobject:
        raise MotionBuildError(f"{label} must be one non-object scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeError as exc:
            raise MotionBuildError(f"{label} is not UTF-8") from exc
    return str(item)


def _canonical_scalar_text(value: np.ndarray, label: str) -> str:
    """Read one exact scalar string emitted by the canonical migration tool."""

    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U" or array.dtype.hasobject:
        raise MotionBuildError(f"{label} must be one canonical unicode scalar string")
    return _scalar_text(array, label)


def _migration_provenance(arrays: Mapping[str, np.ndarray]) -> dict[str, str] | None:
    present = frozenset(SCHEMA2_MIGRATION_PROVENANCE_KEYS) & frozenset(arrays)
    if not present:
        return None
    if present != frozenset(SCHEMA2_MIGRATION_PROVENANCE_KEYS):
        missing = sorted(frozenset(SCHEMA2_MIGRATION_PROVENANCE_KEYS) - present)
        raise MotionBuildError(f"partial migration provenance; missing={missing}")
    source_sha = _canonical_scalar_text(
        arrays["kinematics_migration_source_sha256"],
        "kinematics_migration_source_sha256",
    )
    if len(source_sha) != 64 or any(character not in "0123456789abcdef" for character in source_sha):
        raise MotionBuildError(
            "kinematics_migration_source_sha256 must be one lowercase SHA-256"
        )
    source_point = _canonical_scalar_text(
        arrays["kinematics_migration_source_point"],
        "kinematics_migration_source_point",
    )
    if source_point not in ("link_origin", "center_of_mass"):
        raise MotionBuildError(
            "kinematics_migration_source_point must be link_origin or center_of_mass"
        )
    tool = _canonical_scalar_text(
        arrays["kinematics_migration_tool"], "kinematics_migration_tool"
    )
    if tool != CANONICAL_MIGRATION_TOOL:
        raise MotionBuildError(
            f"kinematics_migration_tool must be {CANONICAL_MIGRATION_TOOL!r}"
        )
    return {
        "kinematics_migration_source_sha256": source_sha,
        "kinematics_migration_source_point": source_point,
        "kinematics_migration_tool": tool,
    }


def _validate_zip_members(payload: bytes, label: str) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise MotionBuildError(f"{label} is not a valid NPZ/ZIP: {exc}") from exc
    if len(names) != len(set(names)):
        raise MotionBuildError(f"{label} contains duplicate ZIP members")
    if any(not name.endswith(".npy") or "/" in name or "\\" in name for name in names):
        raise MotionBuildError(f"{label} contains a non-canonical NPZ member")


def load_schema2_snapshot(payload: bytes, label: str) -> dict[str, np.ndarray]:
    """Load a schema-2 NPZ from an immutable byte snapshot."""

    _validate_zip_members(payload, label)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            keys = tuple(archive.files)
            key_set = set(keys)
            if len(keys) != len(key_set) or not REQUIRED_KEYS.issubset(key_set):
                missing = sorted(REQUIRED_KEYS - set(keys))
                unexpected = sorted(set(keys) - ALLOWED_KEYS)
                raise MotionBuildError(
                    f"{label} schema-2 field set changed; missing={missing} unexpected={unexpected}"
                )
            unexpected = sorted(key_set - ALLOWED_KEYS)
            if unexpected:
                raise MotionBuildError(
                    f"{label} schema-2 field set changed; missing=[] unexpected={unexpected}"
                )
            arrays = {key: np.asarray(archive[key]).copy() for key in keys}
    except MotionBuildError:
        raise
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile) as exc:
        raise MotionBuildError(f"cannot load {label}: {exc}") from exc

    fps = arrays["fps"]
    schema = arrays["kinematics_schema_version"]
    if fps.shape != (1,) or fps.dtype != np.int64 or int(fps[0]) <= 0:
        raise MotionBuildError(f"{label} fps must be exact positive int64[1]")
    if schema.shape != (1,) or schema.dtype != np.int64 or int(schema[0]) != 2:
        raise MotionBuildError(f"{label} kinematics_schema_version must be int64[2]")
    if _scalar_text(arrays["body_pos_point"], f"{label} body_pos_point") != "link_origin":
        raise MotionBuildError(f"{label} body_pos_point must be link_origin")
    if (
        _scalar_text(arrays["body_lin_vel_point"], f"{label} body_lin_vel_point")
        != "center_of_mass"
    ):
        raise MotionBuildError(f"{label} body_lin_vel_point must be center_of_mass")

    joint_pos = arrays["joint_pos"]
    body_pos = arrays["body_pos_w"]
    if joint_pos.ndim != 2 or body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise MotionBuildError(f"{label} joint/body position shapes are invalid")
    frames, joints = joint_pos.shape
    bodies = body_pos.shape[1]
    expected_shapes = {
        "joint_pos": (frames, joints),
        "joint_vel": (frames, joints),
        "body_pos_w": (frames, bodies, 3),
        "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3),
        "body_ang_vel_w": (frames, bodies, 3),
    }
    if frames < 8 or joints < 1 or bodies < 1:
        raise MotionBuildError(f"{label} is too small to synthesize a protected join")
    for key, shape in expected_shapes.items():
        array = arrays[key]
        if array.shape != shape or array.dtype != np.float32:
            raise MotionBuildError(
                f"{label} {key} must be float32{shape}, got {array.dtype}{array.shape}"
            )
        if not np.isfinite(array).all():
            raise MotionBuildError(f"{label} {key} contains NaN/Inf")
    names = arrays["body_names"]
    if names.shape != (bodies,) or names.dtype.hasobject:
        raise MotionBuildError(f"{label} body_names must be a non-object [{bodies}] array")
    decoded = tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in names.tolist()
    )
    if any(not name for name in decoded) or len(set(decoded)) != len(decoded):
        raise MotionBuildError(f"{label} body_names must be unique and non-empty")
    quat_norm_error = float(
        np.max(
            np.abs(
                np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1) - 1.0
            )
        )
    )
    if quat_norm_error > 2.0e-5:
        raise MotionBuildError(
            f"{label} quaternion norm error {quat_norm_error:.9g} exceeds 2e-5"
        )
    expected_joint_vel = np.gradient(
        joint_pos, 1.0 / float(fps[0]), axis=0
    ).astype(np.float32)
    if not np.array_equal(arrays["joint_vel"], expected_joint_vel):
        raise MotionBuildError(
            f"{label} joint_vel must be producer-exact gradient(joint_pos, 1/fps)"
        )
    try:
        _migration_provenance(arrays)
    except MotionBuildError as exc:
        raise MotionBuildError(f"{label} {exc}") from exc
    return arrays


def _same_schema(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> None:
    fields = ("fps", "kinematics_schema_version", "body_pos_point", "body_lin_vel_point")
    for key in fields:
        if not np.array_equal(left[key], right[key]):
            raise MotionBuildError(f"ready/source schema field differs: {key}")
    if not np.array_equal(left["body_names"], right["body_names"]):
        raise MotionBuildError("ready/source body_names differ")
    for key in SCHEMA2_TIME_KEYS:
        if left[key].shape[1:] != right[key].shape[1:]:
            raise MotionBuildError(f"ready/source channel shape differs: {key}")


def quintic_hermite(
    p0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    duration_s: float,
    sample_times_s: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position, first derivative, and second derivative of a C2 quintic."""

    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise MotionBuildError("quintic duration must be finite and positive")
    arrays = [np.asarray(value, dtype=np.float64) for value in (p0, v0, a0, p1, v1, a1)]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise MotionBuildError("quintic endpoint shapes disagree")
    if not all(np.isfinite(value).all() for value in arrays):
        raise MotionBuildError("quintic endpoints contain NaN/Inf")
    p0d, v0d, a0d, p1d, v1d, a1d = arrays
    c0 = p0d
    c1 = v0d
    c2 = 0.5 * a0d
    t = duration
    rhs = np.stack(
        [
            p1d - (c0 + c1 * t + c2 * t * t),
            v1d - (c1 + 2.0 * c2 * t),
            a1d - 2.0 * c2,
        ],
        axis=0,
    )
    matrix = np.array(
        [
            [t**3, t**4, t**5],
            [3.0 * t**2, 4.0 * t**3, 5.0 * t**4],
            [6.0 * t, 12.0 * t**2, 20.0 * t**3],
        ],
        dtype=np.float64,
    )
    flat = rhs.reshape(3, -1)
    solved = np.linalg.solve(matrix, flat).reshape((3,) + p0d.shape)
    c3, c4, c5 = solved
    samples = np.asarray(sample_times_s, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise MotionBuildError("quintic sample times must be one finite vector")
    if np.any(samples < 0.0) or np.any(samples > duration):
        raise MotionBuildError("quintic sample times fall outside the duration")
    expand = (slice(None),) + (None,) * p0d.ndim
    tt = samples[expand]
    position = c0 + c1 * tt + c2 * tt**2 + c3 * tt**3 + c4 * tt**4 + c5 * tt**5
    velocity = c1 + 2.0 * c2 * tt + 3.0 * c3 * tt**2 + 4.0 * c4 * tt**3 + 5.0 * c5 * tt**4
    acceleration = 2.0 * c2 + 6.0 * c3 * tt + 12.0 * c4 * tt**2 + 20.0 * c5 * tt**3
    return position, velocity, acceleration


def _gradient(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=np.float64), dt, axis=0)


def _quaternion_prefix(
    ready: np.ndarray,
    source: np.ndarray,
    join_frame: int,
    duration: float,
    sample_times: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Component-Hermite quaternion prefix, normalized and hemisphere continuous.

    This is intentionally a host-only candidate construction.  The JSON contract
    requires a production FK rebuild and L0 audit before the result can be trained.
    """

    aligned = np.asarray(source, dtype=np.float64).copy()
    # Quaternion q and -q encode the same rotation, but the output suffix is a
    # byte-exact copy of the original source.  Anchor alignment at the original
    # join row (not frame 0), then propagate only the derivative workspace in
    # both directions.  The synthesized prefix therefore converges to the exact
    # source sign and cannot introduce a component-space jump at the splice.
    for frame in range(join_frame - 1, -1, -1):
        flip = np.sum(aligned[frame] * aligned[frame + 1], axis=-1) < 0.0
        aligned[frame, flip] *= -1.0
    for frame in range(join_frame + 1, aligned.shape[0]):
        flip = np.sum(aligned[frame - 1] * aligned[frame], axis=-1) < 0.0
        aligned[frame, flip] *= -1.0
    q0 = np.asarray(ready, dtype=np.float64).copy()
    q1 = np.asarray(source[join_frame], dtype=np.float64)
    q0[np.sum(q0 * q1, axis=-1) < 0.0] *= -1.0
    qd = _gradient(aligned, dt)
    qdd = _gradient(qd, dt)
    position, _velocity, _acceleration = quintic_hermite(
        q0,
        np.zeros_like(q0),
        np.zeros_like(q0),
        q1,
        qd[join_frame],
        qdd[join_frame],
        duration,
        sample_times,
    )
    norm = np.linalg.norm(position, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-10):
        raise MotionBuildError("quaternion Hermite prefix crossed zero norm")
    return (position / norm).astype(np.float32)


def _quintic_signal_prefix(
    ready_value: np.ndarray,
    source: np.ndarray,
    join_frame: int,
    duration: float,
    sample_times: np.ndarray,
    dt: float,
) -> np.ndarray:
    first = _gradient(source, dt)
    second = _gradient(first, dt)
    values, _first, _second = quintic_hermite(
        ready_value,
        np.zeros_like(ready_value, dtype=np.float64),
        np.zeros_like(ready_value, dtype=np.float64),
        np.asarray(source[join_frame], dtype=np.float64),
        first[join_frame],
        second[join_frame],
        duration,
        sample_times,
    )
    return values.astype(np.float32)


def _array_window_digest(arrays: Mapping[str, np.ndarray], start: int, stop: int) -> str:
    digest = hashlib.sha256()
    for key in SCHEMA2_TIME_KEYS:
        array = np.ascontiguousarray(arrays[key][start:stop])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_candidate(
    source: Mapping[str, np.ndarray],
    ready_source: Mapping[str, np.ndarray],
    *,
    contact_frame: int,
    join_frame: int,
    ready_frame: int,
    hold_frames: int,
    blend_intervals: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _same_schema(source, ready_source)
    frames = int(source["joint_pos"].shape[0])
    ready_frames = int(ready_source["joint_pos"].shape[0])
    fps = int(source["fps"][0])
    if ready_frame != 0 or ready_frame >= ready_frames:
        raise MotionBuildError("shared q_ready must use ready-source frame 0")
    if not 0 < contact_frame < frames - 1:
        raise MotionBuildError("contact_frame must have source frames on both sides")
    protected_frames = int(math.ceil(PROTECTED_PRECONTACT_SECONDS * fps))
    protected_start = contact_frame - protected_frames
    if protected_start < 1:
        raise MotionBuildError("contact_frame is too early for the protected 0.1s window")
    # joint_vel is the exact centered gradient of the output joint positions.
    # Therefore the join must precede the protected window by at least one row;
    # otherwise the first protected velocity would depend on a synthesized row.
    latest_join_frame = protected_start - 1
    if not 2 <= join_frame <= latest_join_frame:
        raise MotionBuildError(
            f"join_frame must be in [2,{latest_join_frame}] so the protected window is untouched"
        )
    if hold_frames < 4:
        raise MotionBuildError(
            "hold_frames must be >=4 so producer-exact gradients retain at least three zero-speed rows"
        )
    if blend_intervals < 5:
        raise MotionBuildError("blend_intervals must be >=5 for a meaningful quintic join")
    if join_frame + 2 >= frames:
        raise MotionBuildError("join_frame needs two following source frames for C2 endpoints")

    dt = 1.0 / float(fps)
    duration = blend_intervals * dt
    # The endpoint is supplied by source[join_frame], so synthesize only interior samples.
    sample_times = np.arange(1, blend_intervals, dtype=np.float64) * dt
    interior_count = int(sample_times.size)
    join_output = hold_frames + interior_count
    contact_output = join_output + (contact_frame - join_frame)
    output_frames = join_output + (frames - join_frame)

    output: dict[str, np.ndarray] = {}
    for key, value in source.items():
        if key not in SCHEMA2_TIME_KEYS:
            output[key] = np.asarray(value).copy()

    ready_joint = ready_source["joint_pos"][ready_frame].astype(np.float64)
    source_joint = source["joint_pos"].astype(np.float64)
    source_joint_vel = source["joint_vel"].astype(np.float64)
    source_joint_acc = _gradient(source_joint_vel, dt)
    joint_pos_interior, _joint_vel_interior, _joint_acc_interior = quintic_hermite(
        ready_joint,
        np.zeros_like(ready_joint),
        np.zeros_like(ready_joint),
        source_joint[join_frame],
        source_joint_vel[join_frame],
        source_joint_acc[join_frame],
        duration,
        sample_times,
    )

    ready_body_pos = ready_source["body_pos_w"][ready_frame].astype(np.float64)
    source_body_pos = source["body_pos_w"].astype(np.float64)
    source_link_vel = _gradient(source_body_pos, dt)
    source_link_acc = _gradient(source_link_vel, dt)
    body_pos_interior, _link_vel, _link_acc = quintic_hermite(
        ready_body_pos,
        np.zeros_like(ready_body_pos),
        np.zeros_like(ready_body_pos),
        source_body_pos[join_frame],
        source_link_vel[join_frame],
        source_link_acc[join_frame],
        duration,
        sample_times,
    )
    quat_interior = _quaternion_prefix(
        ready_source["body_quat_w"][ready_frame],
        source["body_quat_w"],
        join_frame,
        duration,
        sample_times,
        dt,
    )
    body_lin_interior = _quintic_signal_prefix(
        np.zeros_like(ready_source["body_lin_vel_w"][ready_frame]),
        source["body_lin_vel_w"],
        join_frame,
        duration,
        sample_times,
        dt,
    )
    body_ang_interior = _quintic_signal_prefix(
        np.zeros_like(ready_source["body_ang_vel_w"][ready_frame]),
        source["body_ang_vel_w"],
        join_frame,
        duration,
        sample_times,
        dt,
    )

    def assemble(ready_value: np.ndarray, interior: np.ndarray, source_values: np.ndarray) -> np.ndarray:
        hold = np.repeat(ready_value[None, ...], hold_frames, axis=0)
        return np.concatenate([hold, interior, source_values[join_frame:]], axis=0)

    output["joint_pos"] = assemble(
        ready_source["joint_pos"][ready_frame],
        joint_pos_interior.astype(np.float32),
        source["joint_pos"],
    )
    # Schema-2 requires this exact producer stencil.  Recompute it from the new
    # position path instead of publishing the analytic derivative as if it were
    # the discrete runtime channel.  Because the protected window is wholly in
    # the copied source suffix, its velocities still remain bitwise unchanged.
    output["joint_vel"] = np.gradient(output["joint_pos"], dt, axis=0).astype(np.float32)
    output["body_pos_w"] = assemble(
        ready_source["body_pos_w"][ready_frame],
        body_pos_interior.astype(np.float32),
        source["body_pos_w"],
    )
    output["body_quat_w"] = assemble(
        ready_source["body_quat_w"][ready_frame],
        quat_interior,
        source["body_quat_w"],
    )
    output["body_lin_vel_w"] = assemble(
        np.zeros_like(ready_source["body_lin_vel_w"][ready_frame]),
        body_lin_interior,
        source["body_lin_vel_w"],
    )
    output["body_ang_vel_w"] = assemble(
        np.zeros_like(ready_source["body_ang_vel_w"][ready_frame]),
        body_ang_interior,
        source["body_ang_vel_w"],
    )

    for key in SCHEMA2_TIME_KEYS:
        if output[key].shape[0] != output_frames or output[key].dtype != np.float32:
            raise MotionBuildError(f"internal output shape/dtype failure for {key}")
        if not np.isfinite(output[key]).all():
            raise MotionBuildError(f"synthesized {key} contains NaN/Inf")
        if key != "joint_vel" and not np.array_equal(
            output[key][join_output:], source[key][join_frame:]
        ):
            raise MotionBuildError(f"source suffix is not bitwise preserved for {key}")
    if not all(
        np.array_equal(output[key][0], ready_source[key][ready_frame])
        for key in ("joint_pos", "body_pos_w", "body_quat_w")
    ):
        raise MotionBuildError("frame0 does not equal the selected shared q_ready pose")
    initial_zero_frames = hold_frames - 1
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        if not np.array_equal(
            output[key][:initial_zero_frames], np.zeros_like(output[key][:initial_zero_frames])
        ):
            raise MotionBuildError(f"initial hold is not bitwise zero velocity for {key}")

    protected_output_start = join_output + (protected_start - join_frame)
    source_digest = _array_window_digest(source, protected_start, contact_frame + 1)
    output_digest = _array_window_digest(output, protected_output_start, contact_output + 1)
    if source_digest != output_digest:
        raise MotionBuildError("protected pre-contact window changed")
    quat_norm_error = float(
        np.max(
            np.abs(
                np.linalg.norm(output["body_quat_w"].astype(np.float64), axis=-1) - 1.0
            )
        )
    )
    if quat_norm_error > 2.0e-5:
        raise MotionBuildError("synthesized quaternion norm error exceeds schema-2 input tolerance")
    join_joint_velocity_error = float(
        np.max(
            np.abs(
                output["joint_vel"][join_output].astype(np.float64)
                - source["joint_vel"][join_frame].astype(np.float64)
            )
        )
    )

    proof = {
        "source_frames": frames,
        "output_frames": output_frames,
        "fps": fps,
        "source_join_frame": join_frame,
        "output_join_frame": join_output,
        "source_contact_frame": contact_frame,
        "output_contact_frame": contact_output,
        "source_protected_start_frame": protected_start,
        "output_protected_start_frame": protected_output_start,
        "protected_frames_before_contact": protected_frames,
        "protected_window_sha256": source_digest,
        "protected_window_bitwise_equal": True,
        "pose_and_body_velocity_source_suffix_bitwise_equal": True,
        "joint_velocity": "producer_exact_gradient_of_output_joint_position",
        "frame0_shared_ready_pose_bitwise_equal": True,
        "ready_source_velocity_channels_ignored": True,
        "ready_velocity_definition": "explicit_bitwise_zero",
        "initial_zero_velocity_frames": initial_zero_frames,
        "joint_position_continuous_quintic_endpoint_c2": True,
        "producer_gradient_join_velocity_error_rad_s": join_joint_velocity_error,
        "finite": True,
        "quaternion_max_norm_error": quat_norm_error,
        "contact_time_from_frame0_s": contact_output / float(fps),
        "source_migration_provenance": _migration_provenance(source),
        "source_migration_provenance_preserved": all(
            np.array_equal(output[key], source[key])
            for key in SCHEMA2_MIGRATION_PROVENANCE_KEYS
            if key in source
        ),
        "ready_source_migration_provenance": _migration_provenance(ready_source),
        "migration_provenance_validation": {
            "canonical_syntax_and_verbatim_lineage_only": True,
            "legacy_ancestor_bytes_rehashed": False,
        },
    }
    return output, proof


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(fd)


def _validate_persisted_npz(path: Path, expected: Mapping[str, np.ndarray]) -> None:
    payload = path.read_bytes()
    persisted = load_schema2_snapshot(payload, "staged output NPZ")
    if set(persisted) != set(expected):
        raise MotionBuildError("staged output NPZ field set changed")
    for key, value in expected.items():
        if persisted[key].dtype != value.dtype or persisted[key].shape != value.shape:
            raise MotionBuildError(f"staged output {key} dtype/shape changed")
        if not np.array_equal(persisted[key], value):
            raise MotionBuildError(f"staged output {key} values changed")


def _prepare_outputs(npz_like: Path | str, contract_like: Path | str) -> tuple[Path, Path, Path]:
    npz_path = _absolute(Path(npz_like))
    contract_path = _absolute(Path(contract_like))
    if npz_path == contract_path:
        raise MotionBuildError("output NPZ and contract must be different paths")
    if npz_path.suffix != ".npz" or contract_path.suffix != ".json":
        raise MotionBuildError("outputs must use .npz and .json suffixes")
    if npz_path.parent != contract_path.parent:
        raise MotionBuildError("output NPZ and contract must share one existing parent")
    ensure_no_symlink_components(npz_path.parent, "output parent")
    for path in (npz_path, contract_path):
        ensure_no_symlink_components(path, "output", leaf_may_be_missing=True)
        if path.exists() or path.is_symlink():
            raise MotionBuildError(f"no-clobber output already exists: {path}")
    return npz_path, contract_path, npz_path.parent


def publish_bundle(
    output: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    *,
    npz_path: Path,
    contract_path: Path,
    parent: Path,
) -> dict[str, Any]:
    staging = Path(tempfile.mkdtemp(prefix=".ready-to-strike-", dir=parent))
    staged_npz = staging / "candidate.npz"
    staged_contract = staging / "contract.json"
    published: list[Path] = []
    try:
        with staged_npz.open("xb") as stream:
            np.savez(stream, **output)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_persisted_npz(staged_npz, output)
        output_evidence = {
            "path": str(npz_path),
            "bytes": staged_npz.stat().st_size,
            "sha256": sha256_file(staged_npz),
        }
        bound = dict(contract)
        bound["output"] = dict(bound["output"])
        bound["output"]["npz"] = output_evidence
        _write_exclusive(staged_contract, _json_bytes(bound))

        for staged, destination in ((staged_npz, npz_path), (staged_contract, contract_path)):
            os.link(staged, destination)
            published.append(destination)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "npz": output_evidence,
            "contract": {
                "path": str(contract_path),
                "bytes": contract_path.stat().st_size,
                "sha256": sha256_file(contract_path),
            },
        }
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build_contract(
    *,
    source_evidence: Mapping[str, Any],
    ready_evidence: Mapping[str, Any],
    tool_evidence: Mapping[str, Any],
    proof: Mapping[str, Any],
    request: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "artifact_kind": "host_only_ready_to_strike_motion_candidate",
        "status": "candidate_only_all_runtime_and_safety_gates_open",
        "inputs": {
            "source_schema2_npz": dict(source_evidence),
            "shared_ready_schema2_npz": dict(ready_evidence),
            "shared_ready_frame": int(request["ready_frame"]),
        },
        "tool": {
            "path": "scripts/build_ready_to_strike_motion.py",
            **dict(tool_evidence),
            "binding_semantics": "source_file_snapshot_at_main_entry_unchanged_before_publish",
        },
        "request": {
            "source_contact_frame": int(request["contact_frame"]),
            "source_join_frame": int(request["join_frame"]),
            "ready_hold_frames": int(request["hold_frames"]),
            "quintic_blend_intervals": int(request["blend_intervals"]),
            "protected_precontact_seconds": PROTECTED_PRECONTACT_SECONDS,
        },
        "synthesis": {
            "shared_ready_definition": "ready_source_frame0_pose_with_explicit_zero_velocity",
            "joint_and_link_position_prefix": "quintic_Hermite_C2",
            "quaternion_prefix": "hemisphere_aligned_component_quintic_then_unit_normalize",
            "velocity_channels": (
                "joint_velocity_is_exact_output_position_gradient; body_velocity_channels_are_"
                "zero_ready_then_quintic_endpoint_match_then_source_suffix"
            ),
            "source_suffix": (
                "joint_pos_and_body_channels_bitwise_copy_from_source_join_frame; "
                "joint_vel_rebuilt_by_schema2_gradient; all_six_channels_bitwise_in_protected_window"
            ),
            "simple_crop": False,
            "host_only": True,
            "production_fk_rebuild_required": True,
        },
        "proof": dict(proof),
        "output": {
            "npz": None,
            "contract_binding": "JSON binds exact NPZ SHA-256; publication is no-clobber",
        },
        "authorization": {
            "host_candidate_materialized": True,
            "topp_runup_0p5_pass": False,
            "l0_static_pass": False,
            "vendor_l1_pass": False,
            "self_hit_pass": False,
            "table_net_clearance_5mm_pass": False,
            "dynamics_pass": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "required_next_gates": [
            "production_FK_rebuild_with_exact_MJCF_and_runtime_body_order",
            "TOPP_runup_at_most_0.5s_without_relaxing_limits",
            "schema2_L0_static",
            "vendor_L1_full_trajectory_self_collision_and_racket_self_hit",
            "table_and_net_swept_clearance_at_least_5mm",
            "dynamics_CoP_friction_torque_and_stability",
        ],
        "explicit_non_claims": [
            "not_a_TOPP_certificate",
            "not_L0_or_L1",
            "not_table_net_or_dynamics_safe",
            "not_trainable",
            "not_deployable",
            "migration_legacy_ancestor_bytes_not_rehashed",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-clobber host-only shared-ready -> strike motion candidate."
    )
    parser.add_argument("--source", type=Path, required=True, help="Source schema-2 motion NPZ")
    parser.add_argument(
        "--ready-source", type=Path, required=True, help="Schema-2 NPZ containing shared q_ready"
    )
    parser.add_argument(
        "--ready-frame",
        type=int,
        default=0,
        help="Must be 0: waiting is defined as the selected motion's frame-0 pose at zero speed",
    )
    parser.add_argument("--contact-frame", type=int, required=True)
    parser.add_argument("--join-frame", type=int, required=True)
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument(
        "--blend-intervals",
        type=int,
        required=True,
        help="50 Hz intervals from the final held q_ready row to the source join row",
    )
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-contract", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tool_path, tool_payload, tool_evidence = read_regular_snapshot(
            Path(__file__), "ready-to-strike tool source"
        )
        _source_path, source_payload, source_evidence = read_regular_snapshot(
            args.source, "source schema-2 NPZ"
        )
        _ready_path, ready_payload, ready_evidence = read_regular_snapshot(
            args.ready_source, "shared q_ready schema-2 NPZ"
        )
        source = load_schema2_snapshot(source_payload, "source schema-2 NPZ")
        ready = load_schema2_snapshot(ready_payload, "shared q_ready schema-2 NPZ")
        request = {
            "contact_frame": args.contact_frame,
            "join_frame": args.join_frame,
            "ready_frame": args.ready_frame,
            "hold_frames": args.hold_frames,
            "blend_intervals": args.blend_intervals,
        }
        output, proof = build_candidate(source, ready, **request)
        npz_path, contract_path, parent = _prepare_outputs(
            args.output_npz, args.output_contract
        )
        contract = build_contract(
            source_evidence=source_evidence,
            ready_evidence=ready_evidence,
            tool_evidence=tool_evidence,
            proof=proof,
            request=request,
        )
        _tool_path_after, tool_payload_after, tool_evidence_after = read_regular_snapshot(
            tool_path, "ready-to-strike tool source before publish"
        )
        if tool_payload_after != tool_payload or tool_evidence_after != tool_evidence:
            raise MotionBuildError("ready-to-strike tool source changed during the run")
        evidence = publish_bundle(
            output,
            contract,
            npz_path=npz_path,
            contract_path=contract_path,
            parent=parent,
        )
    except MotionBuildError as exc:
        print(f"ready-to-strike contract error: {exc}", file=os.sys.stderr)
        return 2
    except OSError as exc:
        print(f"ready-to-strike I/O error: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "host_candidate_materialized_downstream_gates_open",
                "output_npz_sha256": evidence["npz"]["sha256"],
                "contract_sha256": evidence["contract"]["sha256"],
                "output_contact_frame": proof["output_contact_frame"],
                "contact_time_from_frame0_s": proof["contact_time_from_frame0_s"],
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
