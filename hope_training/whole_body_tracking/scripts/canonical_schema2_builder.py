#!/usr/bin/env python3
"""Build an exact schema-2 motion candidate from a timed canonical trajectory.

This module is deliberately the last *kinematics* stage of the canonical motion
compiler.  Its caller must already have produced a time law:

* ``joint_pos`` and ``joint_vel`` are the same 31-DoF runtime trajectory;
* the free root pose is accompanied by its world-frame link-origin linear and
  angular velocity;
* frame zero and the last frame are the same static canonical-ready pose.

MuJoCo is then the only source of body kinematics.  For every frame this module
sets the named free root and all 31 named joints, calls ``mj_forward``, stores
the 32 named link-origin poses, and evaluates ``mj_jacBodyCom @ qvel`` for the
body-COM linear velocity and body angular velocity.  It never estimates body
velocity by differencing already-sampled body poses.

The result is a compiler candidate only.  It cannot authorize training,
deployment, dynamics, balance, contact, or hardware use.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

import sys

sys.path.insert(0, str(_HERE))
from audit_motion_npz import ISAAC_JOINT_NAMES  # noqa: E402
from motion_kinematics_contract import (  # noqa: E402
    BODY_LIN_VEL_POINT,
    BODY_NAMES_KEY,
    BODY_POS_POINT,
    KINEMATICS_SCHEMA_VERSION,
    LIN_VEL_POINT_KEY,
    MIGRATION_SOURCE_POINT_KEY,
    MIGRATION_SOURCE_SHA256_KEY,
    MIGRATION_TOOL_KEY,
    POS_POINT_KEY,
    SCHEMA_KEY,
)


RUNTIME_JOINT_NAMES = tuple(ISAAC_JOINT_NAMES)
DEFAULT_BODY_ORDER_PATH = _REPO_ROOT / "configs/a3_runtime_body_order.txt"
PUBLICATION_CLASS = "compiler_candidate"
TOOL_ID = "canonical_schema2_builder_v1"

TIME_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
ACTIVE_METADATA_KEYS = (
    SCHEMA_KEY,
    POS_POINT_KEY,
    LIN_VEL_POINT_KEY,
    BODY_NAMES_KEY,
)
MIGRATION_KEYS = (
    MIGRATION_SOURCE_SHA256_KEY,
    MIGRATION_SOURCE_POINT_KEY,
    MIGRATION_TOOL_KEY,
)
BASE_KEYS = ("fps",) + TIME_KEYS + ACTIVE_METADATA_KEYS
ALLOWED_KEYSETS = (
    frozenset(BASE_KEYS),
    frozenset(BASE_KEYS + MIGRATION_KEYS),
)

_QUAT_NORM_TOL = 1.0e-6
_ROOT_TWIST_TOL = 2.0e-10
_ROOT_MATRIX_CONDITION_MAX = 1.0e10
_POSITION_LIMIT_TOL = 1.0e-7


class Schema2BuildError(ValueError):
    """The candidate cannot be built without weakening a data contract."""


@dataclass(frozen=True)
class MigrationProvenance:
    """Optional atomic schema-2 migration tuple (all three fields or none)."""

    source_sha256: str
    source_point: str
    tool: str


@dataclass(frozen=True)
class Schema2Candidate:
    """In-memory candidate, deterministic NPZ payload, manifest, and report."""

    arrays: Mapping[str, np.ndarray]
    npz_bytes: bytes
    manifest: Mapping[str, Any]
    report: Mapping[str, Any]

    @property
    def output_sha256(self) -> str:
        return str(self.manifest["hashes"]["output_npz_sha256"])


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, label: str) -> str:
    try:
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Schema2BuildError(f"cannot read {label} {path}: {exc}") from exc
    return digest.hexdigest()


def _digest(value: str, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise Schema2BuildError(f"{label} must be a 64-digit SHA-256 hex digest")
    return digest


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Schema2BuildError(f"{label} must be numeric") from exc
    if array.shape != shape:
        raise Schema2BuildError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise Schema2BuildError(f"{label} contains NaN/Inf")
    return array


def _read_body_order(path: Path) -> tuple[str, ...]:
    try:
        names = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError as exc:
        raise Schema2BuildError(f"cannot read body-order {path}: {exc}") from exc
    if len(names) != 32:
        raise Schema2BuildError(
            f"body-order must contain exactly 32 names, got {len(names)}"
        )
    if any(not name for name in names) or len(set(names)) != len(names):
        raise Schema2BuildError("body-order names must be non-empty and unique")
    return names


def _continuous_unit_quaternions(value: np.ndarray) -> tuple[np.ndarray, float]:
    norm = np.linalg.norm(value, axis=1)
    if np.any(norm <= 0.0):
        raise Schema2BuildError("root_quat_wxyz contains a zero quaternion")
    maximum_error = float(np.max(np.abs(norm - 1.0)))
    if maximum_error > _QUAT_NORM_TOL:
        raise Schema2BuildError(
            "root_quat_wxyz must be unit length "
            f"(max norm error {maximum_error:.3e})"
        )
    unit = value / norm[:, None]
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    if np.any(dots < 0.0):
        frame = int(np.flatnonzero(dots < 0.0)[0] + 1)
        raise Schema2BuildError(
            f"root_quat_wxyz changes quaternion hemisphere at frame {frame}"
        )
    return unit, maximum_error


def _same_static_ready(
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_vel: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
) -> None:
    if not np.array_equal(joint_pos[0], joint_pos[-1]):
        raise Schema2BuildError(
            "first/last joint_pos must be the exact same canonical-ready pose"
        )
    if not np.array_equal(root_pos[0], root_pos[-1]):
        raise Schema2BuildError(
            "first/last root_pos_w must be the exact same canonical-ready pose"
        )
    if not np.array_equal(root_quat[0], root_quat[-1]):
        raise Schema2BuildError(
            "first/last root_quat_wxyz must be the exact same signed ready quaternion"
        )
    for label, velocity in (
        ("joint_vel", joint_vel),
        ("root_lin_vel_w", root_lin_vel),
        ("root_ang_vel_w", root_ang_vel),
    ):
        if np.count_nonzero(velocity[[0, -1]]) != 0:
            raise Schema2BuildError(
                f"{label} first/last rows must be exactly zero; "
                "the builder will not patch an inconsistent time law"
            )


def _fps_array(fps: float) -> np.ndarray:
    value = float(fps)
    if not np.isfinite(value) or value <= 0.0:
        raise Schema2BuildError(f"fps must be finite and positive, got {fps!r}")
    rounded = int(round(value))
    if abs(value - rounded) <= 1.0e-12:
        return np.array([rounded], dtype=np.int64)
    return np.array([value], dtype=np.float64)


def _load_mujoco(module: Any | None) -> Any:
    if module is not None:
        return module
    try:
        import mujoco
    except (ImportError, OSError) as exc:
        raise Schema2BuildError(
            "MuJoCo Python bindings are required for exact schema-2 FK"
        ) from exc
    return mujoco


@dataclass(frozen=True)
class _ModelBinding:
    root_joint_id: int
    root_body_id: int
    root_qpos_adr: int
    root_dof_adr: int
    joint_ids: np.ndarray
    joint_qpos_adrs: np.ndarray
    joint_dof_adrs: np.ndarray
    body_ids: np.ndarray


def _bind_model(mujoco: Any, model: Any, body_names: Sequence[str]) -> _ModelBinding:
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    free_ids = [
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) == free_type
    ]
    if len(free_ids) != 1:
        raise Schema2BuildError(
            f"MJCF must expose exactly one free root joint, got {len(free_ids)}"
        )
    root_joint_id = int(free_ids[0])
    root_body_id = int(model.jnt_bodyid[root_joint_id])
    root_qpos_adr = int(model.jnt_qposadr[root_joint_id])
    root_dof_adr = int(model.jnt_dofadr[root_joint_id])
    if root_qpos_adr < 0 or root_qpos_adr + 7 > int(model.nq):
        raise Schema2BuildError("free-root qpos address is invalid")
    if root_dof_adr < 0 or root_dof_adr + 6 > int(model.nv):
        raise Schema2BuildError("free-root qvel address is invalid")

    joint_ids: list[int] = []
    joint_qpos_adrs: list[int] = []
    joint_dof_adrs: list[int] = []
    for name in RUNTIME_JOINT_NAMES:
        joint_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        )
        if joint_id < 0:
            raise Schema2BuildError(f"runtime joint {name!r} is absent from MJCF")
        if int(model.jnt_type[joint_id]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise Schema2BuildError(f"runtime joint {name!r} is not scalar")
        joint_ids.append(joint_id)
        joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_adrs.append(int(model.jnt_dofadr[joint_id]))
    if len(set(joint_ids)) != 31:
        raise Schema2BuildError("runtime joint names do not resolve to 31 unique MJCF joints")

    body_ids: list[int] = []
    for name in body_names:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        if body_id < 0:
            raise Schema2BuildError(f"runtime body {name!r} is absent from MJCF")
        body_ids.append(body_id)
    if len(set(body_ids)) != 32:
        raise Schema2BuildError("body-order does not resolve to 32 unique MJCF bodies")
    if body_ids[0] != root_body_id:
        raise Schema2BuildError(
            f"body-order first body {body_names[0]!r} is not the free-root body"
        )

    return _ModelBinding(
        root_joint_id=root_joint_id,
        root_body_id=root_body_id,
        root_qpos_adr=root_qpos_adr,
        root_dof_adr=root_dof_adr,
        joint_ids=np.asarray(joint_ids, dtype=np.int64),
        joint_qpos_adrs=np.asarray(joint_qpos_adrs, dtype=np.int64),
        joint_dof_adrs=np.asarray(joint_dof_adrs, dtype=np.int64),
        body_ids=np.asarray(body_ids, dtype=np.int64),
    )


def _validate_joint_limits(model: Any, binding: _ModelBinding, joint_pos: np.ndarray) -> None:
    for column, joint_id in enumerate(binding.joint_ids.tolist()):
        if not bool(model.jnt_limited[joint_id]):
            continue
        lower, upper = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        minimum = float(np.min(joint_pos[:, column]))
        maximum = float(np.max(joint_pos[:, column]))
        if minimum < lower - _POSITION_LIMIT_TOL or maximum > upper + _POSITION_LIMIT_TOL:
            raise Schema2BuildError(
                f"joint {RUNTIME_JOINT_NAMES[column]!r} exceeds MJCF range "
                f"[{lower:.9g}, {upper:.9g}] with [{minimum:.9g}, {maximum:.9g}]"
            )


def _set_qpos(
    data: Any,
    binding: _ModelBinding,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> None:
    data.qpos[:] = 0.0
    start = binding.root_qpos_adr
    data.qpos[start : start + 3] = root_pos
    data.qpos[start + 3 : start + 7] = root_quat
    data.qpos[binding.joint_qpos_adrs] = joint_pos


def _root_generalized_velocity(
    mujoco: Any,
    model: Any,
    data: Any,
    binding: _ModelBinding,
    joint_velocity: np.ndarray,
    root_lin_velocity_w: np.ndarray,
    root_ang_velocity_w: np.ndarray,
) -> np.ndarray:
    qvel = np.zeros(int(model.nv), dtype=np.float64)
    qvel[binding.joint_dof_adrs] = joint_velocity
    if (
        np.count_nonzero(joint_velocity) == 0
        and np.count_nonzero(root_lin_velocity_w) == 0
        and np.count_nonzero(root_ang_velocity_w) == 0
    ):
        return qvel

    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, binding.root_body_id)
    root_slice = slice(binding.root_dof_adr, binding.root_dof_adr + 6)
    matrix = np.vstack((jacp[:, root_slice], jacr[:, root_slice]))
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > _ROOT_MATRIX_CONDITION_MAX:
        raise Schema2BuildError(
            f"free-root twist Jacobian is singular/ill-conditioned ({condition:.3e})"
        )
    contribution = np.concatenate((jacp @ qvel, jacr @ qvel))
    target = np.concatenate((root_lin_velocity_w, root_ang_velocity_w))
    qvel[root_slice] = np.linalg.solve(matrix, target - contribution)
    reproduced = np.concatenate((jacp @ qvel, jacr @ qvel))
    error = float(np.max(np.abs(reproduced - target)))
    if error > _ROOT_TWIST_TOL:
        raise Schema2BuildError(
            f"free-root world twist reconstruction error {error:.3e} exceeds "
            f"{_ROOT_TWIST_TOL:.3e}"
        )
    return qvel


def _make_quaternion_columns_continuous(quat: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.asarray(quat, dtype=np.float64).copy()
    flips = 0
    for body in range(out.shape[1]):
        for frame in range(1, out.shape[0]):
            if float(out[frame - 1, body] @ out[frame, body]) < 0.0:
                out[frame, body] *= -1.0
                flips += 1
    return out, flips


def _rebuild_bodies(
    mujoco: Any,
    model: Any,
    binding: _ModelBinding,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    data = mujoco.MjData(model)
    frames = int(joint_pos.shape[0])
    body_pos = np.empty((frames, 32, 3), dtype=np.float64)
    body_quat = np.empty((frames, 32, 4), dtype=np.float64)
    body_lin = np.empty((frames, 32, 3), dtype=np.float64)
    body_ang = np.empty((frames, 32, 3), dtype=np.float64)
    maximum_root_twist_error = 0.0

    for frame in range(frames):
        _set_qpos(
            data,
            binding,
            root_pos[frame],
            root_quat[frame],
            joint_pos[frame],
        )
        mujoco.mj_forward(model, data)
        qvel = _root_generalized_velocity(
            mujoco,
            model,
            data,
            binding,
            joint_vel[frame],
            root_lin_vel[frame],
            root_ang_vel[frame],
        )
        data.qvel[:] = qvel

        body_pos[frame] = np.asarray(data.xpos, dtype=np.float64)[binding.body_ids]
        body_quat[frame] = np.asarray(data.xquat, dtype=np.float64)[binding.body_ids]
        for column, body_id in enumerate(binding.body_ids.tolist()):
            jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
            jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
            mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
            body_lin[frame, column] = jacp @ qvel
            body_ang[frame, column] = jacr @ qvel

        root_origin_jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
        root_origin_jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
        mujoco.mj_jacBody(
            model,
            data,
            root_origin_jacp,
            root_origin_jacr,
            binding.root_body_id,
        )
        got = np.concatenate(
            (root_origin_jacp @ qvel, root_origin_jacr @ qvel)
        )
        expected = np.concatenate((root_lin_vel[frame], root_ang_vel[frame]))
        maximum_root_twist_error = max(
            maximum_root_twist_error, float(np.max(np.abs(got - expected)))
        )

    body_quat, sign_flips = _make_quaternion_columns_continuous(body_quat)
    return (
        body_pos,
        body_quat,
        body_lin,
        body_ang,
        maximum_root_twist_error,
        sign_flips,
    )


def _validate_migration(
    migration: MigrationProvenance | None,
) -> dict[str, np.ndarray]:
    if migration is None:
        return {}
    source_sha = _digest(migration.source_sha256, "migration source_sha256")
    source_point = str(migration.source_point)
    tool = str(migration.tool)
    if not source_point or not tool:
        raise Schema2BuildError("migration source_point and tool must be non-empty")
    return {
        MIGRATION_SOURCE_SHA256_KEY: np.array(source_sha),
        MIGRATION_SOURCE_POINT_KEY: np.array(source_point),
        MIGRATION_TOOL_KEY: np.array(tool),
    }


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize an NPZ with stable key order and ZIP metadata."""

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in arrays:
            array_stream = io.BytesIO()
            np.lib.format.write_array(
                array_stream,
                np.asarray(arrays[key]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_stream.getvalue())
    return stream.getvalue()


def _json_safe_copy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    return MappingProxyType(payload)


def build_schema2_candidate(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_lin_vel_w: np.ndarray,
    root_ang_vel_w: np.ndarray,
    fps: float,
    mjcf_path: os.PathLike[str] | str,
    input_sha256: str,
    ready_sha256: str,
    body_order_path: os.PathLike[str] | str = DEFAULT_BODY_ORDER_PATH,
    migration_provenance: MigrationProvenance | None = None,
    _mujoco_module: Any | None = None,
) -> Schema2Candidate:
    """Build one exact 11/14-field schema-2 candidate entirely in memory.

    Velocities are explicit compiler inputs.  This function will not infer them
    from pose samples and will not overwrite non-zero endpoints.
    """

    if len(RUNTIME_JOINT_NAMES) != 31 or len(set(RUNTIME_JOINT_NAMES)) != 31:
        raise Schema2BuildError("runtime joint contract is not 31 unique names")
    q_raw = np.asarray(joint_pos)
    if q_raw.ndim != 2 or q_raw.shape[1] != 31 or q_raw.shape[0] < 2:
        raise Schema2BuildError(
            f"joint_pos must have shape (T>=2,31), got {q_raw.shape}"
        )
    frames = int(q_raw.shape[0])
    q = _finite_array(joint_pos, (frames, 31), "joint_pos")
    qd = _finite_array(joint_vel, (frames, 31), "joint_vel")
    root_pos = _finite_array(root_pos_w, (frames, 3), "root_pos_w")
    root_quat_raw = _finite_array(
        root_quat_wxyz, (frames, 4), "root_quat_wxyz"
    )
    root_lin = _finite_array(root_lin_vel_w, (frames, 3), "root_lin_vel_w")
    root_ang = _finite_array(root_ang_vel_w, (frames, 3), "root_ang_vel_w")
    root_quat, quaternion_norm_error = _continuous_unit_quaternions(root_quat_raw)
    _same_static_ready(q, root_pos, root_quat, qd, root_lin, root_ang)
    fps_field = _fps_array(fps)

    input_digest = _digest(input_sha256, "input_sha256")
    ready_digest = _digest(ready_sha256, "ready_sha256")
    mjcf = Path(mjcf_path).expanduser().resolve()
    body_order = Path(body_order_path).expanduser().resolve()
    if not mjcf.is_file():
        raise Schema2BuildError(f"MJCF not found: {mjcf}")
    body_names = _read_body_order(body_order)
    mjcf_digest = _sha256_file(mjcf, "MJCF")
    body_order_digest = _sha256_file(body_order, "body-order")
    tool_digest = _sha256_file(Path(__file__).resolve(), "builder tool")

    # Quantize the published position/time-law channels first, then derive FK
    # from those exact published values rather than from higher-precision donors.
    q_out = q.astype(np.float32)
    qd_out = qd.astype(np.float32)
    root_pos_fk = root_pos.astype(np.float32).astype(np.float64)
    root_quat_fk = root_quat.astype(np.float32).astype(np.float64)
    root_quat_fk /= np.linalg.norm(root_quat_fk, axis=1, keepdims=True)
    root_lin_fk = root_lin.astype(np.float32).astype(np.float64)
    root_ang_fk = root_ang.astype(np.float32).astype(np.float64)
    _same_static_ready(
        q_out,
        root_pos_fk,
        root_quat_fk,
        qd_out,
        root_lin_fk,
        root_ang_fk,
    )

    mujoco = _load_mujoco(_mujoco_module)
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf))
    except Exception as exc:
        raise Schema2BuildError(f"cannot compile MJCF {mjcf}: {exc}") from exc
    binding = _bind_model(mujoco, model, body_names)
    _validate_joint_limits(model, binding, q_out.astype(np.float64))
    (
        body_pos,
        body_quat,
        body_lin,
        body_ang,
        root_twist_error,
        quaternion_sign_flips,
    ) = _rebuild_bodies(
        mujoco,
        model,
        binding,
        q_out.astype(np.float64),
        qd_out.astype(np.float64),
        root_pos_fk,
        root_quat_fk,
        root_lin_fk,
        root_ang_fk,
    )

    arrays: dict[str, np.ndarray] = {
        "fps": fps_field,
        "joint_pos": q_out,
        "joint_vel": qd_out,
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
        SCHEMA_KEY: np.array([KINEMATICS_SCHEMA_VERSION], dtype=np.int64),
        POS_POINT_KEY: np.array(BODY_POS_POINT),
        LIN_VEL_POINT_KEY: np.array(BODY_LIN_VEL_POINT),
        BODY_NAMES_KEY: np.asarray(body_names),
    }
    arrays.update(_validate_migration(migration_provenance))
    if frozenset(arrays) not in ALLOWED_KEYSETS:
        raise Schema2BuildError(
            f"internal error: output key set is not exact schema-2: {sorted(arrays)}"
        )
    for key in TIME_KEYS:
        if not np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all():
            raise Schema2BuildError(f"internal error: output {key} is not finite")
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        if np.count_nonzero(arrays[key][[0, -1]]) != 0:
            raise Schema2BuildError(
                f"derived {key} endpoints are not exactly zero"
            )

    npz_bytes = _deterministic_npz(arrays)
    output_digest = _sha256_bytes(npz_bytes)
    hashes = {
        "input_sha256": input_digest,
        "ready_sha256": ready_digest,
        "mjcf_sha256": mjcf_digest,
        "body_order_sha256": body_order_digest,
        "tool_sha256": tool_digest,
        "output_npz_sha256": output_digest,
    }
    shared = {
        "publication_class": PUBLICATION_CLASS,
        "training_authorized": False,
        "tool_id": TOOL_ID,
        "hashes": hashes,
        "runtime_contract": {
            "joint_count": 31,
            "joint_names": list(RUNTIME_JOINT_NAMES),
            "body_count": 32,
            "body_names": list(body_names),
            "schema2_field_count": len(arrays),
        },
    }
    manifest = {
        **shared,
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "files": {
            "mjcf_path": str(mjcf),
            "body_order_path": str(body_order),
            "tool_path": str(Path(__file__).resolve()),
        },
        "kinematics": {
            "body_pos_point": BODY_POS_POINT,
            "body_lin_vel_point": BODY_LIN_VEL_POINT,
            "body_velocity_method": "mujoco_mj_jacBodyCom_times_qvel",
            "root_velocity_input": "world_link_origin_twist",
            "joint_velocity_input": "explicit_compiler_time_law",
            "pose_finite_difference_used": False,
            "static_ready_endpoints_required": True,
        },
        "non_claims": [
            "training_authorization",
            "dynamics",
            "balance",
            "contact",
            "deployment",
            "hardware",
        ],
    }
    report = {
        **shared,
        "status": "PASS",
        "frames": frames,
        "fps": float(np.asarray(fps_field).reshape(-1)[0]),
        "checks": {
            "exact_schema2_keyset": True,
            "all_finite": True,
            "same_ready_pose_first_last": True,
            "six_velocity_channels_zero_first_last": True,
            "runtime_joint_names_bound_by_name": True,
            "runtime_body_names_bound_by_name": True,
            "root_world_twist_max_abs_error": root_twist_error,
            "root_quaternion_input_max_norm_error": quaternion_norm_error,
            "body_quaternion_sign_flips": quaternion_sign_flips,
        },
        "maxima": {
            "joint_speed_radps": float(np.max(np.abs(arrays["joint_vel"]))),
            "body_com_speed_mps": float(
                np.max(np.linalg.norm(arrays["body_lin_vel_w"], axis=-1))
            ),
            "body_angular_speed_radps": float(
                np.max(np.linalg.norm(arrays["body_ang_vel_w"], axis=-1))
            ),
        },
    }

    frozen_arrays = MappingProxyType(
        {key: np.array(value, copy=True) for key, value in arrays.items()}
    )
    return Schema2Candidate(
        arrays=frozen_arrays,
        npz_bytes=npz_bytes,
        manifest=_json_safe_copy(manifest),
        report=_json_safe_copy(report),
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_schema2_candidate(
    candidate: Schema2Candidate,
    output_path: os.PathLike[str] | str,
    *,
    manifest_path: os.PathLike[str] | str | None = None,
    report_path: os.PathLike[str] | str | None = None,
) -> tuple[Path, Path, Path]:
    """Write NPZ + manifest + report with no-clobber and rollback on failure."""

    if not isinstance(candidate, Schema2Candidate):
        raise Schema2BuildError("candidate must be a Schema2Candidate")
    # Do not resolve the leaf: resolving a broken symlink erases the very
    # directory entry that a no-clobber check must reject.
    output = Path(os.path.abspath(os.fspath(Path(output_path).expanduser())))
    manifest = (
        Path(os.path.abspath(os.fspath(Path(manifest_path).expanduser())))
        if manifest_path is not None
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    report = (
        Path(os.path.abspath(os.fspath(Path(report_path).expanduser())))
        if report_path is not None
        else output.with_suffix(output.suffix + ".report.json")
    )
    paths = (output, manifest, report)
    if len(set(paths)) != 3:
        raise Schema2BuildError("output, manifest, and report paths must be distinct")
    # ``Path.exists`` is false for a broken symlink.  Treat the directory entry
    # itself as occupied so an attacker or stale staging link cannot redirect a
    # supposedly no-clobber publication.
    existing = [str(path) for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError("refusing to overwrite existing path(s): " + ", ".join(existing))
    parents = {path.parent for path in paths}
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)

    payloads = (
        candidate.npz_bytes,
        _json_bytes(candidate.manifest),
        _json_bytes(candidate.report),
    )
    created: list[Path] = []
    try:
        for path, payload in zip(paths, payloads):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        raise

    if _sha256_file(output, "written output") != candidate.output_sha256:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        raise Schema2BuildError("written output SHA-256 differs from in-memory candidate")
    return paths


__all__ = [
    "ALLOWED_KEYSETS",
    "DEFAULT_BODY_ORDER_PATH",
    "MigrationProvenance",
    "PUBLICATION_CLASS",
    "RUNTIME_JOINT_NAMES",
    "Schema2BuildError",
    "Schema2Candidate",
    "build_schema2_candidate",
    "write_schema2_candidate",
]
