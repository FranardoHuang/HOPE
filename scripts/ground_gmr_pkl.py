#!/usr/bin/env python3
"""Ground exactly one A3 GMR pickle against the canonical MuJoCo collision model.

This tool deliberately does less than the historical GMR postprocessor:

* one explicit input and one explicit, new output (no directory scan or overwrite);
* no schema-2 conversion, resampling, yaw/grip edit, or joint modification;
* one constant translation is added to ``root_pos[:, 2]``;
* the translation comes from the lowest world-z point of every enabled collision
  geom in the floating-root robot subtree, not from body origins or COMs.

The input is the GMR A3 pickle contract: ``root_rot`` is xyzw and ``dof_pos``
uses ``A3_GMR_JOINT_NAMES``.  The pickle format does not normally carry joint
names, so that interpretation is bound here and checked against the exact
MuJoCo joint-id order.  If an input ``joint_names`` field exists it must match.

Passing proves only a structurally valid, collision-geometry-grounded diagnostic
pickle.  It is not a schema-2 motion, a dynamics/safety gate, or robot approval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# GMR's agibot_a3 dof_pos column order.  It is byte-for-order identical to the
# converter's CSV_JOINT_NAMES and the canonical vendor MJCF hinge-joint order.
A3_GMR_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


class GroundingError(ValueError):
    """The single-file grounding contract cannot be satisfied."""


@dataclass(frozen=True)
class ModelBinding:
    model: Any
    data: Any
    root_joint_id: int
    root_body_id: int
    root_qpos_address: int
    joint_ids: tuple[int, ...]
    joint_qpos_addresses: tuple[int, ...]
    collision_geom_ids: tuple[int, ...]
    ground_geom_id: int
    ground_z_m: float
    collision_contract_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if not SHA256_RE.fullmatch(normalized):
        raise GroundingError(f"{label} must be a lowercase 64-hex SHA-256")
    return normalized


def _numeric_ndarray(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise GroundingError(
            f"{name} must be a numpy ndarray in the GMR pickle; refusing a type-changing rewrite "
            f"from {type(value).__name__}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise GroundingError(f"{name} must have floating dtype, got {value.dtype}")
    if value.dtype.itemsize < 4:
        raise GroundingError(f"{name} dtype {value.dtype} is too narrow for grounding")
    if not np.isfinite(value).all():
        bad = int(value.size - np.count_nonzero(np.isfinite(value)))
        raise GroundingError(f"{name} contains {bad} non-finite value(s)")
    return value


def _normalize_joint_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise GroundingError("joint_names must be a list/tuple/ndarray of strings")
    names = tuple(str(item) for item in value)
    if any(not name for name in names):
        raise GroundingError("joint_names contains an empty name")
    return names


def validate_payload(
    payload: Any,
    *,
    expected_frames: int | None,
    expected_fps: float,
    quaternion_norm_tolerance: float,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GroundingError("GMR pickle root must be a mapping")
    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [name for name in required if name not in payload]
    if missing:
        raise GroundingError(f"GMR pickle is missing fields {missing}")

    fps_array = np.asarray(payload["fps"])
    if fps_array.size != 1 or not np.issubdtype(fps_array.dtype, np.number):
        raise GroundingError(f"fps must be one numeric scalar, got shape={fps_array.shape}")
    fps = float(fps_array.reshape(-1)[0])
    if not np.isfinite(fps) or abs(fps - expected_fps) > 1e-9:
        raise GroundingError(f"fps={fps!r}, expected exactly {expected_fps}")

    root_pos = _numeric_ndarray(payload["root_pos"], "root_pos")
    root_rot = _numeric_ndarray(payload["root_rot"], "root_rot")
    dof_pos = _numeric_ndarray(payload["dof_pos"], "dof_pos")
    frames = int(root_pos.shape[0]) if root_pos.ndim == 2 else -1
    if frames < 2:
        raise GroundingError(f"root_pos must have at least two frames, got shape={root_pos.shape}")
    shapes = {
        "root_pos": (frames, 3),
        "root_rot": (frames, 4),
        "dof_pos": (frames, len(A3_GMR_JOINT_NAMES)),
    }
    for name, expected_shape in shapes.items():
        actual = np.asarray(payload[name]).shape
        if actual != expected_shape:
            raise GroundingError(f"{name} shape {actual}, expected {expected_shape}")
    if expected_frames is not None and frames != expected_frames:
        raise GroundingError(f"frames={frames}, expected {expected_frames}")

    if not np.isfinite(quaternion_norm_tolerance) or quaternion_norm_tolerance <= 0:
        raise GroundingError("quaternion_norm_tolerance must be finite and positive")
    quat_norm_error = np.abs(np.linalg.norm(root_rot.astype(np.float64), axis=1) - 1.0)
    max_quat_norm_error = float(quat_norm_error.max())
    if max_quat_norm_error > quaternion_norm_tolerance:
        raise GroundingError(
            f"root_rot xyzw max norm error={max_quat_norm_error:.9g}, required "
            f"<= {quaternion_norm_tolerance}"
        )

    names_present = "joint_names" in payload
    if names_present:
        actual_names = _normalize_joint_names(payload["joint_names"])
        if actual_names != A3_GMR_JOINT_NAMES:
            raise GroundingError("input joint_names do not match the canonical GMR A3 order")

    return {
        "frames": frames,
        "fps": fps,
        "shapes": {name: list(shape) for name, shape in shapes.items()},
        "dtypes": {name: str(np.asarray(payload[name]).dtype) for name in shapes},
        "finite_elements": int(root_pos.size + root_rot.size + dof_pos.size + 1),
        "root_rotation_convention": "xyzw",
        "root_rotation_max_norm_error": max_quat_norm_error,
        "joint_names_present_in_input": names_present,
        "joint_order_interpretation": "A3_GMR_JOINT_NAMES",
    }


def load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        raise GroundingError(f"cannot load GMR pickle {path}: {exc}") from None


def _name(mujoco: Any, model: Any, obj_type: Any, obj_id: int, fallback: str) -> str:
    value = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return value if value is not None else fallback


def _descends_from(model: Any, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    seen: set[int] = set()
    while current not in seen:
        if current == ancestor_id:
            return True
        if current == 0:
            return False
        seen.add(current)
        current = int(model.body_parentid[current])
    raise GroundingError(f"cycle in MJCF body_parentid at body id {body_id}")


def _hash_array(digest: "hashlib._Hash", label: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _compiled_collision_contract_sha256(model: Any, geom_ids: Iterable[int]) -> str:
    """Bind the loaded kinematic/collision arrays, including referenced mesh vertices."""

    digest = hashlib.sha256()
    for label in (
        "body_parentid",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_pos",
        "jnt_axis",
    ):
        _hash_array(digest, label, getattr(model, label))
    gids = np.asarray(tuple(geom_ids), dtype=np.int64)
    _hash_array(digest, "selected_geom_ids", gids)
    for label in (
        "geom_type",
        "geom_bodyid",
        "geom_contype",
        "geom_conaffinity",
        "geom_dataid",
        "geom_size",
        "geom_pos",
        "geom_quat",
    ):
        _hash_array(digest, label, np.asarray(getattr(model, label))[gids])
    mesh_ids = sorted(
        {
            int(model.geom_dataid[gid])
            for gid in gids
            if int(model.geom_dataid[gid]) >= 0
        }
    )
    for mesh_id in mesh_ids:
        address = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        _hash_array(digest, f"mesh_{mesh_id}_vertices", model.mesh_vert[address : address + count])
    return digest.hexdigest()


def bind_model(mujoco: Any, mjcf_path: Path, *, ground_geom_name: str) -> ModelBinding:
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
    except Exception as exc:
        raise GroundingError(f"cannot load MuJoCo model {mjcf_path}: {exc}") from None

    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    free_joint_ids = [jid for jid in range(int(model.njnt)) if int(model.jnt_type[jid]) == free_type]
    if len(free_joint_ids) != 1:
        raise GroundingError(f"MJCF must contain exactly one free joint, got {len(free_joint_ids)}")
    root_joint_id = free_joint_ids[0]
    root_body_id = int(model.jnt_bodyid[root_joint_id])
    if root_body_id == 0:
        raise GroundingError("free joint is attached to the world body")

    subtree_hinges = [
        jid
        for jid in range(int(model.njnt))
        if int(model.jnt_type[jid]) == hinge_type
        and _descends_from(model, int(model.jnt_bodyid[jid]), root_body_id)
    ]
    model_joint_names = tuple(
        _name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, jid, f"joint{jid}")
        for jid in subtree_hinges
    )
    if model_joint_names != A3_GMR_JOINT_NAMES:
        raise GroundingError(
            "MJCF floating-root hinge order does not match A3_GMR_JOINT_NAMES; "
            f"got {model_joint_names}"
        )
    if len(set(model_joint_names)) != len(model_joint_names):
        raise GroundingError("MJCF floating-root hinge names are not unique")

    qpos_addresses = tuple(int(model.jnt_qposadr[jid]) for jid in subtree_hinges)
    if len(set(qpos_addresses)) != len(qpos_addresses):
        raise GroundingError("MJCF hinge qpos addresses are not unique")
    for jid, name in zip(subtree_hinges, model_joint_names):
        if hasattr(model, "jnt_limited") and not bool(model.jnt_limited[jid]):
            raise GroundingError(f"MJCF joint {name!r} is not range-limited")

    ground_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground_geom_name))
    if ground_gid < 0:
        raise GroundingError(f"ground geom {ground_geom_name!r} is missing from MJCF")
    if int(model.geom_type[ground_gid]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
        raise GroundingError(f"ground geom {ground_geom_name!r} is not a plane")
    if int(model.geom_bodyid[ground_gid]) != 0:
        raise GroundingError(f"ground geom {ground_geom_name!r} is not world-fixed")

    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    ground_rotation = np.asarray(data.geom_xmat[ground_gid], dtype=np.float64).reshape(3, 3)
    ground_normal = ground_rotation[:, 2]
    if not np.allclose(ground_normal, np.array([0.0, 0.0, 1.0]), atol=1e-10, rtol=0.0):
        raise GroundingError(
            f"ground geom {ground_geom_name!r} must be horizontal +Z, normal={ground_normal.tolist()}"
        )
    ground_z_m = float(np.asarray(data.geom_xpos[ground_gid], dtype=np.float64)[2])
    if not np.isfinite(ground_z_m):
        raise GroundingError("ground geom world z is non-finite")

    supported = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_MESH),
    }
    collision_geom_ids: list[int] = []
    for gid in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[gid])
        if not _descends_from(model, body_id, root_body_id):
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        geom_type = int(model.geom_type[gid])
        if geom_type not in supported:
            geom_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, gid, f"geom{gid}")
            raise GroundingError(
                f"enabled robot collision geom {geom_name!r} has unsupported type {geom_type}"
            )
        collision_geom_ids.append(gid)
    if not collision_geom_ids:
        raise GroundingError("MJCF floating-root robot subtree has no enabled collision geoms")

    return ModelBinding(
        model=model,
        data=data,
        root_joint_id=root_joint_id,
        root_body_id=root_body_id,
        root_qpos_address=int(model.jnt_qposadr[root_joint_id]),
        joint_ids=tuple(subtree_hinges),
        joint_qpos_addresses=qpos_addresses,
        collision_geom_ids=tuple(collision_geom_ids),
        ground_geom_id=ground_gid,
        ground_z_m=ground_z_m,
        collision_contract_sha256=_compiled_collision_contract_sha256(model, collision_geom_ids),
    )


def validate_joint_ranges(
    payload: dict[str, Any], binding: ModelBinding, *, tolerance_rad: float
) -> dict[str, float]:
    if not np.isfinite(tolerance_rad) or tolerance_rad < 0:
        raise GroundingError("joint_range_tolerance_rad must be finite and non-negative")
    values = np.asarray(payload["dof_pos"], dtype=np.float64)
    ranges = np.asarray(binding.model.jnt_range, dtype=np.float64)[list(binding.joint_ids)]
    lower_excess = ranges[None, :, 0] - values
    upper_excess = values - ranges[None, :, 1]
    excess = np.maximum(np.maximum(lower_excess, upper_excess), 0.0)
    worst_index = np.unravel_index(int(np.argmax(excess)), excess.shape)
    worst = float(excess[worst_index])
    if worst > tolerance_rad:
        frame, column = (int(worst_index[0]), int(worst_index[1]))
        raise GroundingError(
            f"dof_pos exceeds MJCF range by {worst:.9g} rad at frame {frame}, "
            f"joint {A3_GMR_JOINT_NAMES[column]!r}"
        )
    return {"max_joint_range_excess_rad": worst, "tolerance_rad": tolerance_rad}


def geom_world_min_z(mujoco: Any, model: Any, data: Any, geom_id: int) -> float:
    """Exact support point in world -Z for supported MuJoCo geom types."""

    geom_type = int(model.geom_type[geom_id])
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    world_z_in_local = rotation[2]
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        support = float(size[0])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        support = float(size[1] * abs(world_z_in_local[2]) + size[0])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radial = float(np.hypot(world_z_in_local[0], world_z_in_local[1]))
        support = float(size[1] * abs(world_z_in_local[2]) + size[0] * radial)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        support = float(np.linalg.norm(world_z_in_local * size[:3]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        support = float(np.dot(np.abs(world_z_in_local), size[:3]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise GroundingError(f"mesh geom id {geom_id} has no mesh data id")
        address = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[address : address + count], dtype=np.float64)
        if vertices.shape != (count, 3) or count <= 0 or not np.isfinite(vertices).all():
            raise GroundingError(f"mesh geom id {geom_id} has invalid compiled vertices")
        # MuJoCo's compiled mesh_vert is in the referencing geom frame; geom_xmat
        # and geom_xpos therefore map it directly to world (same convention used
        # by MuJoCo's mesh/touch implementation).
        value = float(center[2] + np.min(vertices @ world_z_in_local))
        if not np.isfinite(value):
            raise GroundingError(f"mesh geom id {geom_id} produced non-finite world z")
        return value
    else:
        raise GroundingError(f"unsupported collision geom type {geom_type} at id {geom_id}")

    value = float(center[2] - support)
    if not np.isfinite(value):
        raise GroundingError(f"collision geom id {geom_id} produced non-finite world z")
    return value


def frame_clearances(
    mujoco: Any, binding: ModelBinding, payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    model, data = binding.model, binding.data
    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_xyzw = np.asarray(payload["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
    frames = root_pos.shape[0]
    clearances = np.empty(frames, dtype=np.float64)
    lowest_geom_ids = np.empty(frames, dtype=np.int64)
    root_adr = binding.root_qpos_address

    for frame in range(frames):
        data.qpos[:] = model.qpos0
        data.qpos[root_adr : root_adr + 3] = root_pos[frame]
        xyzw = root_xyzw[frame]
        xyzw = xyzw / np.linalg.norm(xyzw)
        data.qpos[root_adr + 3 : root_adr + 7] = xyzw[[3, 0, 1, 2]]
        data.qpos[list(binding.joint_qpos_addresses)] = dof_pos[frame]
        mujoco.mj_forward(model, data)
        minima = np.asarray(
            [geom_world_min_z(mujoco, model, data, gid) for gid in binding.collision_geom_ids],
            dtype=np.float64,
        )
        local_index = int(np.argmin(minima))
        clearances[frame] = float(minima[local_index] - binding.ground_z_m)
        lowest_geom_ids[frame] = int(binding.collision_geom_ids[local_index])
    if not np.isfinite(clearances).all():
        raise GroundingError("per-frame collision clearance contains non-finite values")
    return clearances, lowest_geom_ids


def _clearance_summary(
    mujoco: Any,
    binding: ModelBinding,
    clearances: np.ndarray,
    lowest_geom_ids: np.ndarray,
) -> dict[str, Any]:
    frame = int(np.argmin(clearances))
    gid = int(lowest_geom_ids[frame])
    body_id = int(binding.model.geom_bodyid[gid])
    return {
        "minimum_clearance_m": float(clearances[frame]),
        "maximum_of_frame_minima_m": float(np.max(clearances)),
        "minimum_frame": frame,
        "minimum_geom_id": gid,
        "minimum_geom_name": _name(
            mujoco, binding.model, mujoco.mjtObj.mjOBJ_GEOM, gid, f"geom{gid}"
        ),
        "minimum_body_id": body_id,
        "minimum_body_name": _name(
            mujoco, binding.model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body{body_id}"
        ),
        "per_frame_minimum_clearance_m": [float(value) for value in clearances],
    }


def _pickle_to_temp(payload: dict[str, Any], output: Path) -> Path:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _json_to_temp(path: Path, value: dict[str, Any]) -> Path:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _install_new_file(temporary: Path, target: Path) -> None:
    """Install without replacement; the temporary is on the same filesystem."""

    try:
        os.link(temporary, target)
    except FileExistsError:
        raise GroundingError(f"refusing to overwrite concurrently-created file: {target}") from None
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        # The installed target is already complete and content-addressed.  A
        # stale hidden temp is cleanup noise, not a reason to report a failed
        # two-file transaction.
        pass


def _rollback_owned_output(path: Path, expected_sha256: str) -> None:
    """Remove only the exact output installed by this invocation."""

    if not path.exists():
        return
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise GroundingError(
            f"transaction rollback refused to delete non-owned output at {path}; manual audit required"
        )
    path.unlink()


def _preflight_paths(input_path: Path, output_path: Path, report_path: Path, mjcf_path: Path) -> None:
    for path, label in ((input_path, "input"), (mjcf_path, "MJCF")):
        if not path.is_file() or path.stat().st_size <= 0:
            raise GroundingError(f"{label} file is missing or empty: {path}")
    resolved = (input_path.resolve(), output_path.resolve(), report_path.resolve())
    if len(set(resolved)) != len(resolved):
        raise GroundingError("input, output, and report paths must be three distinct files")
    for path, label in ((output_path, "output"), (report_path, "report")):
        if path.exists() or path.is_symlink():
            raise GroundingError(f"refusing to overwrite existing {label}: {path}")
        if not path.parent.is_dir():
            raise GroundingError(f"{label} parent directory does not exist: {path.parent}")


def run_grounding(args: argparse.Namespace, *, mujoco_module: Any | None = None) -> dict[str, Any]:
    input_path = Path(args.input).expanduser().resolve()
    # Preserve the leaf path so an existing symlink is rejected rather than
    # silently resolving through it to a different output location.
    output_path = Path(args.output).expanduser().absolute()
    report_path = Path(args.report).expanduser().absolute()
    mjcf_path = Path(args.mjcf).expanduser().resolve()
    _preflight_paths(input_path, output_path, report_path, mjcf_path)

    expected_input_sha = _require_sha256(args.expected_input_sha256, "expected input")
    expected_mjcf_sha = _require_sha256(args.expected_mjcf_sha256, "expected MJCF")
    input_sha = sha256_file(input_path)
    mjcf_sha = sha256_file(mjcf_path)
    if input_sha != expected_input_sha:
        raise GroundingError(
            f"input SHA mismatch: expected {expected_input_sha}, got {input_sha}"
        )
    if mjcf_sha != expected_mjcf_sha:
        raise GroundingError(f"MJCF SHA mismatch: expected {expected_mjcf_sha}, got {mjcf_sha}")

    if mujoco_module is None:
        try:
            import mujoco as mujoco_module  # type: ignore[no-redef]
        except ImportError:
            raise GroundingError("mujoco is not installed; collision grounding cannot run") from None

    target = float(args.target_clearance_m)
    max_grounded = float(args.max_grounded_clearance_m)
    numeric_tolerance = float(args.numerical_tolerance_m)
    max_abs_shift = float(args.max_abs_shift_m)
    if not np.isfinite(target) or target <= 0:
        raise GroundingError("target_clearance_m must be finite and strictly positive")
    if not np.isfinite(max_grounded) or max_grounded < target:
        raise GroundingError("max_grounded_clearance_m must be finite and >= target_clearance_m")
    if not np.isfinite(numeric_tolerance) or numeric_tolerance <= 0:
        raise GroundingError("numerical_tolerance_m must be finite and strictly positive")
    if not np.isfinite(max_abs_shift) or max_abs_shift <= 0:
        raise GroundingError("max_abs_shift_m must be finite and strictly positive")

    payload = load_pickle(input_path)
    structure = validate_payload(
        payload,
        expected_frames=args.expected_frames,
        expected_fps=float(args.expected_fps),
        quaternion_norm_tolerance=float(args.quaternion_norm_tolerance),
    )
    binding = bind_model(mujoco_module, mjcf_path, ground_geom_name=args.ground_geom)
    joint_ranges = validate_joint_ranges(
        payload, binding, tolerance_rad=float(args.joint_range_tolerance_rad)
    )
    before, before_gids = frame_clearances(mujoco_module, binding, payload)
    before_min = float(np.min(before))
    requested_shift = float(target - before_min)
    if abs(requested_shift) > max_abs_shift:
        raise GroundingError(
            f"required root-z shift {requested_shift:.9g} m exceeds max_abs_shift_m={max_abs_shift}"
        )

    grounded = copy.copy(payload)
    old_root = np.asarray(payload["root_pos"])
    new_root = old_root.copy()
    new_root[:, 2] = new_root[:, 2] + np.asarray(requested_shift, dtype=new_root.dtype)
    grounded["root_pos"] = new_root
    applied_shifts = new_root[:, 2].astype(np.float64) - old_root[:, 2].astype(np.float64)
    applied_shift_spread = float(np.ptp(applied_shifts))
    if applied_shift_spread > numeric_tolerance:
        raise GroundingError(
            f"root_pos dtype cannot represent one fixed shift within tolerance: spread="
            f"{applied_shift_spread:.9g} m"
        )

    after, after_gids = frame_clearances(mujoco_module, binding, grounded)
    after_min = float(np.min(after))
    if after_min < target - numeric_tolerance:
        raise GroundingError(
            f"grounded minimum clearance {after_min:.9g} m is below target {target:.9g} m"
        )
    if after_min > max_grounded + numeric_tolerance:
        raise GroundingError(
            f"grounded minimum clearance {after_min:.9g} m exceeds contact ceiling "
            f"{max_grounded:.9g} m (robot would be over-floating)"
        )

    if not np.array_equal(new_root[:, :2], old_root[:, :2]):
        raise GroundingError("internal error: root x/y changed")
    if not np.array_equal(grounded["root_rot"], payload["root_rot"]):
        raise GroundingError("internal error: root_rot changed")
    if not np.array_equal(grounded["dof_pos"], payload["dof_pos"]):
        raise GroundingError("internal error: dof_pos changed")
    relative_error = float(
        np.max(
            np.abs(
                (new_root[:, 2].astype(np.float64) - new_root[0, 2])
                - (old_root[:, 2].astype(np.float64) - old_root[0, 2])
            )
        )
    )
    if relative_error > numeric_tolerance:
        raise GroundingError(
            f"root-z relative trajectory changed by {relative_error:.9g} m, tolerance "
            f"{numeric_tolerance:.9g} m"
        )

    temporary_output = _pickle_to_temp(grounded, output_path)
    try:
        reloaded = load_pickle(temporary_output)
        validate_payload(
            reloaded,
            expected_frames=structure["frames"],
            expected_fps=float(args.expected_fps),
            quaternion_norm_tolerance=float(args.quaternion_norm_tolerance),
        )
        if not np.array_equal(reloaded["root_pos"], grounded["root_pos"]):
            raise GroundingError("serialized output root_pos did not round-trip exactly")
        if not np.array_equal(reloaded["root_rot"], payload["root_rot"]):
            raise GroundingError("serialized output changed root_rot")
        if not np.array_equal(reloaded["dof_pos"], payload["dof_pos"]):
            raise GroundingError("serialized output changed dof_pos")
        verify_after, _ = frame_clearances(mujoco_module, binding, reloaded)
        if not np.allclose(verify_after, after, atol=numeric_tolerance, rtol=0.0):
            raise GroundingError("serialized output clearance verification changed")

        output_sha = sha256_file(temporary_output)
        tool_path = Path(__file__).resolve()
        report = {
            "schema_version": 1,
            "status": "pass",
            "formal_eligible": False,
            "scope": "diagnostic_gmr_root_z_grounding_only",
            "input": {
                "path": str(input_path),
                "bytes": input_path.stat().st_size,
                "sha256": input_sha,
            },
            "output": {
                "path": str(output_path),
                "bytes": temporary_output.stat().st_size,
                "sha256": output_sha,
            },
            "mjcf": {
                "path": str(mjcf_path),
                "bytes": mjcf_path.stat().st_size,
                "sha256": mjcf_sha,
                "compiled_kinematic_collision_sha256": binding.collision_contract_sha256,
                "ground_geom": args.ground_geom,
                "ground_z_m": binding.ground_z_m,
            },
            "tool": {
                "path": str(tool_path),
                "sha256": sha256_file(tool_path),
            },
            "structure": structure,
            "joint_contract": {
                "names": list(A3_GMR_JOINT_NAMES),
                "count": len(A3_GMR_JOINT_NAMES),
                "model_joint_ids": list(binding.joint_ids),
                **joint_ranges,
            },
            "collision_contract": {
                "robot_root_body_id": binding.root_body_id,
                "enabled_robot_geom_count": len(binding.collision_geom_ids),
                "enabled_robot_geom_ids": list(binding.collision_geom_ids),
                "surface_method": "analytic_primitive_support_or_compiled_mesh_vertices",
                "visual_only_geoms_excluded": True,
            },
            "grounding": {
                "target_clearance_m": target,
                "max_grounded_clearance_m": max_grounded,
                "requested_constant_root_z_shift_m": requested_shift,
                "applied_root_z_shift_min_m": float(np.min(applied_shifts)),
                "applied_root_z_shift_max_m": float(np.max(applied_shifts)),
                "applied_root_z_shift_spread_m": applied_shift_spread,
                "before": _clearance_summary(mujoco_module, binding, before, before_gids),
                "after": _clearance_summary(mujoco_module, binding, after, after_gids),
            },
            "invariants": {
                "root_xy_exact": True,
                "root_rotation_exact": True,
                "dof_position_exact": True,
                "root_z_relative_trajectory_max_error_m": relative_error,
                "root_pos_dtype_preserved": str(new_root.dtype) == structure["dtypes"]["root_pos"],
                "all_other_payload_fields_shallow_preserved": True,
            },
            "limitations": [
                "GMR pickles normally omit joint_names; dof_pos is interpreted by the bound GMR A3 schema.",
                "Clearance is evaluated only at original discrete frames; inter-frame/continuous-time ground clearance is not proven.",
                "This does not prove dynamics, balance, self-collision, table/net clearance, racket safety, or returnability.",
                "This output is not schema-2 and must not be used as robot approval.",
            ],
        }
        temporary_report = _json_to_temp(report_path, report)
        output_installed = False
        try:
            _install_new_file(temporary_output, output_path)
            output_installed = True
            _install_new_file(temporary_report, report_path)
        except Exception:
            temporary_report.unlink(missing_ok=True)
            if output_installed:
                _rollback_owned_output(output_path, output_sha)
            raise
        return report
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="one unmodified GMR A3 pickle")
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", required=True, help="explicit new grounded pickle path")
    parser.add_argument("--report", required=True, help="explicit new JSON report path")
    parser.add_argument("--mjcf", required=True, help="canonical vendor A3 MuJoCo XML")
    parser.add_argument("--expected-mjcf-sha256", required=True)
    parser.add_argument("--ground-geom", default="floor")
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--target-clearance-m", type=float, default=1e-5)
    parser.add_argument("--max-grounded-clearance-m", type=float, default=1e-3)
    parser.add_argument("--numerical-tolerance-m", type=float, default=5e-7)
    parser.add_argument("--max-abs-shift-m", type=float, default=0.25)
    parser.add_argument("--quaternion-norm-tolerance", type=float, default=1e-6)
    parser.add_argument("--joint-range-tolerance-rad", type=float, default=1e-5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_grounding(args)
    except (GroundingError, OSError, pickle.PickleError) as exc:
        print(f"[ground-gmr] FAIL: {exc}", file=sys.stderr)
        return 1
    result = report["grounding"]
    print(
        "[ground-gmr] PASS "
        f"before={result['before']['minimum_clearance_m']:+.6f}m "
        f"shift={result['requested_constant_root_z_shift_m']:+.6f}m "
        f"after={result['after']['minimum_clearance_m']:+.6f}m "
        f"sha256={report['output']['sha256'][:12]}..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
