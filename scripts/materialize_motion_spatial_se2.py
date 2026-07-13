#!/usr/bin/env python3
"""Materialize one selected GMR motion with an exact whole-trajectory SE(2).

This is deliberately a narrow Step-B consumer.  It accepts only one tracked
primary selected by the Step-A ledger, verifies the source pickle by SHA-256,
and applies one proper, ground-preserving rigid transform to the floating root
for every frame.  It does not choose a fallback, resample time, edit joints,
run GMR/schema-2/simulation/training, or authorize hardware.

``static`` reads tracked repository contracts only.  ``inspect`` also loads and
verifies the private source without writing.  ``consume`` publishes a new
directory once; the report is linked and fsynced last so its presence is the
completion marker.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
PLAN_STATUS = "preregistered_not_executed"
RESULT_STATUS = "complete_exact_whole_motion_se2_materialization"
SELECTION_STATUS = "complete_primary_pair_selected_certification_blocked"
SELECTION_RESULT = "configs/motion_backhand_loop_bc_proposal_selection_results_20260713.json"
SELECTION_RESULT_SHA256 = "8a80a409ca69e2fa73757b139b8496bb9cdda2e6a66d3fab48412051b408d2be"
SOURCE_REGISTRY = "configs/motion_video_gmr_phase_counterfactual_prereg_20260711.json"
SOURCE_REGISTRY_SHA256 = "fee1b1f9a68fcc0323c1be5832db1b29bdc5f49421712c6f44506d16dae45529"
BASE_PAYLOAD_KEYS = {
    "fps",
    "root_pos",
    "root_rot",
    "dof_pos",
    "local_body_pos",
    "link_body_list",
}
# These names have explicit world-frame semantics in this consumer.  If a
# future content-bound source includes one, it is rotated with the root.  No
# guessed or unknown field is silently copied.
OPTIONAL_WORLD_VECTOR_FIELDS = {
    "root_lin_vel_world",
    "root_ang_vel_world",
}
ALLOWED_ASSETS = {
    "franco_backhand_loop_b": {
        "human_name": "Franco 反手拉候选 B 主选整轨站位实体化",
        "candidate_id": "98e7b883b29d302dc7a24fd3c564648c1f929ff2391e24e58558dcba58af3c14",
        "source_sha256": "90c23a8826397f13c39e5ca023c145c064dd5adfe49feb19043887897c60c17e",
        "source_bytes": 27926,
        "frames": 91,
        "translation_w_m": [0.05035998433, -0.109155849041, 0.0],
        "yaw_deg": -5.0,
        "output_root": "/workspace/codexschema/motion_video_intake_20260711/gmr_spatial_retarget_primary_v1/franco_backhand_loop_b_98e7b883b29d",
    },
    "franco_backhand_loop_c": {
        "human_name": "Franco 反手拉候选 C 主选整轨站位实体化",
        "candidate_id": "aa0c86fd350987bf30e56aebde9789bf9df430b0ec5c3c15cd235410794af299",
        "source_sha256": "4eb40301a51346fd3ad6cae52b13e93ca91b135f9eb9b38e16f7d89e456e9cb6",
        "source_bytes": 30054,
        "frames": 98,
        "translation_w_m": [0.157231187588, -0.157700713465, 0.0],
        "yaw_deg": -10.0,
        "output_root": "/workspace/codexschema/motion_video_intake_20260711/gmr_spatial_retarget_primary_v1/franco_backhand_loop_c_aa0c86fd3509",
    },
}


class MaterializationError(ValueError):
    """The exact SE(2) materialization contract cannot be satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise MaterializationError(f"{label} must be a lowercase SHA-256")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must be an object")
    return value


def require_repo_binding(value: Any, label: str, *, expected: str | None = None) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise MaterializationError(f"{label} must contain exactly path/sha256")
    raw_path = value.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
    ):
        raise MaterializationError(f"{label}.path must be repository-relative")
    if expected is not None and raw_path != expected:
        raise MaterializationError(f"{label}.path must be {expected}")
    path = (REPO_ROOT / raw_path).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise MaterializationError(f"{label}.path escapes the repository") from None
    if not path.is_file():
        raise MaterializationError(f"{label}.path is missing: {path}")
    expected_sha = require_sha(value.get("sha256"), f"{label}.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise MaterializationError(f"{label} sha256 {actual_sha} != {expected_sha}")
    return path


def require_absolute_binding(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise MaterializationError(f"{label} must contain exactly path/bytes/sha256")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise MaterializationError(f"{label}.path must be absolute")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise MaterializationError(f"{label}.bytes must be a positive integer")
    require_sha(value.get("sha256"), f"{label}.sha256")
    return value


def ensure_no_symlink_components(path: Path, label: str) -> None:
    current = path
    missing: list[Path] = []
    while not current.exists():
        missing.append(current)
        if current == current.parent:
            break
        current = current.parent
    if current.exists() and current.is_symlink():
        raise MaterializationError(f"{label} contains symlink component {current}")
    for component in reversed(missing):
        if component.exists() and component.is_symlink():
            raise MaterializationError(f"{label} contains symlink component {component}")
    probe = path
    while probe != probe.parent:
        if probe.exists() and probe.is_symlink():
            raise MaterializationError(f"{label} contains symlink component {probe}")
        probe = probe.parent


def verify_absolute_binding(value: Mapping[str, Any], label: str) -> Path:
    binding = require_absolute_binding(value, label)
    path = Path(binding["path"])
    ensure_no_symlink_components(path, label)
    try:
        info = path.stat()
    except OSError as exc:
        raise MaterializationError(f"cannot stat {label} {path}: {exc}") from None
    if not stat.S_ISREG(info.st_mode):
        raise MaterializationError(f"{label} is not a regular file: {path}")
    if info.st_size != binding["bytes"]:
        raise MaterializationError(f"{label} bytes {info.st_size} != {binding['bytes']}")
    actual_sha = sha256_file(path)
    if actual_sha != binding["sha256"]:
        raise MaterializationError(f"{label} sha256 {actual_sha} != {binding['sha256']}")
    return path


def _numpy_frombuffer(buffer: bytes, dtype: np.dtype, shape: tuple[int, ...], order: str) -> np.ndarray:
    """Narrow replacement for NumPy's private pickle reconstruction helper."""

    return np.frombuffer(buffer, dtype=dtype).reshape(shape, order=order)


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Load only primitive containers plus NumPy numeric array reconstruction."""

    _ALLOWED = {
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.numeric", "_frombuffer"): _numpy_frombuffer,
        ("numpy._core.numeric", "_frombuffer"): _numpy_frombuffer,
    }

    def find_class(self, module: str, name: str) -> Any:
        value = self._ALLOWED.get((module, name))
        if value is None:
            raise MaterializationError(f"pickle global {module}.{name} is not allowlisted")
        return value

    def persistent_load(self, pid: Any) -> Any:
        raise MaterializationError(f"pickle persistent id is forbidden: {pid!r}")


def restricted_pickle_loads(data: bytes) -> Any:
    try:
        return RestrictedNumpyUnpickler(io.BytesIO(data)).load()
    except MaterializationError:
        raise
    except Exception as exc:
        raise MaterializationError(f"cannot decode restricted GMR pickle: {exc}") from None


def load_bound_pickle(path: Path) -> dict[str, Any]:
    value = restricted_pickle_loads(path.read_bytes())
    if not isinstance(value, dict):
        raise MaterializationError("GMR pickle root must be a dict")
    return value


def _numeric_float_array(value: Any, label: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise MaterializationError(f"{label} must be a numpy ndarray")
    if value.shape != shape:
        raise MaterializationError(f"{label} shape {value.shape}, expected {shape}")
    if not np.issubdtype(value.dtype, np.floating) or value.dtype.itemsize < 4:
        raise MaterializationError(f"{label} must use float32 or wider, got {value.dtype}")
    if value.dtype.hasobject or not np.isfinite(value).all():
        raise MaterializationError(f"{label} must be finite and object-free")
    return value


def validate_payload(payload: Any, *, frames: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise MaterializationError("GMR payload must be a dict")
    keys = set(payload)
    unknown = keys - BASE_PAYLOAD_KEYS - OPTIONAL_WORLD_VECTOR_FIELDS
    missing = BASE_PAYLOAD_KEYS - keys
    if missing or unknown:
        raise MaterializationError(
            f"GMR payload key contract failed: missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if payload["local_body_pos"] is not None or payload["link_body_list"] is not None:
        raise MaterializationError(
            "grounded canonical-beta source requires local_body_pos/link_body_list to be null"
        )
    fps = payload["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != 30.0:
        raise MaterializationError("fps must be the exact scalar 30")
    root_pos = _numeric_float_array(payload["root_pos"], "root_pos", (frames, 3))
    root_rot = _numeric_float_array(payload["root_rot"], "root_rot", (frames, 4))
    dof_pos = _numeric_float_array(payload["dof_pos"], "dof_pos", (frames, 31))
    norm_error = np.abs(np.linalg.norm(root_rot.astype(np.float64), axis=1) - 1.0)
    max_norm_error = float(np.max(norm_error))
    if max_norm_error > 1e-6:
        raise MaterializationError(
            f"root_rot xyzw max norm error={max_norm_error:.9g}, required <=1e-6"
        )
    world_fields: list[str] = []
    for field in sorted(keys & OPTIONAL_WORLD_VECTOR_FIELDS):
        _numeric_float_array(payload[field], field, (frames, 3))
        world_fields.append(field)
    return {
        "frames": frames,
        "fps": 30,
        "keys": sorted(keys),
        "dtypes": {
            "root_pos": str(root_pos.dtype),
            "root_rot": str(root_rot.dtype),
            "dof_pos": str(dof_pos.dtype),
            **{field: str(payload[field].dtype) for field in world_fields},
        },
        "root_rotation_convention": "xyzw",
        "root_rotation_max_norm_error": max_norm_error,
        "world_vector_fields_present": world_fields,
    }


def quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(np.asarray(left), -1, 0)
    rx, ry, rz, rw = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        axis=-1,
    )


def yaw_rotation(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    radians = math.radians(yaw_deg)
    c, s = math.cos(radians), math.sin(radians)
    matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    quat = np.array([0.0, 0.0, math.sin(radians / 2.0), math.cos(radians / 2.0)])
    return matrix, quat


def transform_payload(
    payload: Mapping[str, Any], *, translation_w_m: list[float], yaw_deg: float
) -> dict[str, Any]:
    frames = int(np.asarray(payload["root_pos"]).shape[0])
    validate_payload(payload, frames=frames)
    translation = np.asarray(translation_w_m, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all() or translation[2] != 0.0:
        raise MaterializationError("translation must be finite ground-preserving [x,y,0]")
    if not np.isfinite(yaw_deg) or abs(yaw_deg) > 180.0:
        raise MaterializationError("yaw_deg must be finite and within [-180,180]")
    rotation, yaw_quat = yaw_rotation(float(yaw_deg))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-15, rtol=0.0):
        raise MaterializationError("internal yaw rotation is not orthonormal")
    if np.linalg.det(rotation) <= 0.0:
        raise MaterializationError("improper/mirrored transform is forbidden")

    output = copy.copy(payload)
    source_pos = payload["root_pos"]
    new_pos = np.empty_like(source_pos)
    new_pos[:, :2] = (
        source_pos[:, :2].astype(np.float64) @ rotation[:2, :2].T + translation[:2]
    ).astype(source_pos.dtype)
    new_pos[:, 2] = source_pos[:, 2]
    output["root_pos"] = new_pos

    source_rot = payload["root_rot"]
    left = np.broadcast_to(yaw_quat, source_rot.shape)
    new_rot = quat_multiply_xyzw(left, source_rot.astype(np.float64)).astype(source_rot.dtype)
    output["root_rot"] = new_rot

    for field in sorted(set(payload) & OPTIONAL_WORLD_VECTOR_FIELDS):
        source = payload[field]
        rotated = (source.astype(np.float64) @ rotation.T).astype(source.dtype)
        output[field] = rotated
    return output


def _semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    return type(left) is type(right) and left == right


def _max_pairwise_distance_error(source: np.ndarray, output: np.ndarray) -> float:
    source_delta = source[:, None, :].astype(np.float64) - source[None, :, :].astype(np.float64)
    output_delta = output[:, None, :].astype(np.float64) - output[None, :, :].astype(np.float64)
    source_distance = np.linalg.norm(source_delta, axis=-1)
    output_distance = np.linalg.norm(output_delta, axis=-1)
    return float(np.max(np.abs(source_distance - output_distance)))


def verify_transform(
    source: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    translation_w_m: list[float],
    yaw_deg: float,
    tolerance: float,
) -> dict[str, Any]:
    frames = int(np.asarray(source["root_pos"]).shape[0])
    source_structure = validate_payload(source, frames=frames)
    output_structure = validate_payload(output, frames=frames)
    if set(source) != set(output):
        raise MaterializationError("output payload keys changed")
    if source_structure["dtypes"] != output_structure["dtypes"]:
        raise MaterializationError("output payload dtypes changed")
    for field in sorted(set(source) - {"root_pos", "root_rot"} - OPTIONAL_WORLD_VECTOR_FIELDS):
        if not _semantic_equal(source[field], output[field]):
            raise MaterializationError(f"non-spatial field {field} changed")
    if not np.array_equal(source["root_pos"][:, 2], output["root_pos"][:, 2]):
        raise MaterializationError("grounding failed: root z changed")

    rotation, yaw_quat = yaw_rotation(float(yaw_deg))
    inverse_rotation = rotation.T
    translation = np.asarray(translation_w_m, dtype=np.float64)
    recovered_pos = np.empty_like(source["root_pos"], dtype=np.float64)
    recovered_pos[:, :2] = (
        output["root_pos"][:, :2].astype(np.float64) - translation[:2]
    ) @ inverse_rotation[:2, :2].T
    recovered_pos[:, 2] = output["root_pos"][:, 2]
    position_inverse_error = float(
        np.max(np.abs(recovered_pos - source["root_pos"].astype(np.float64)))
    )

    inverse_quat = yaw_quat.copy()
    inverse_quat[:3] *= -1.0
    recovered_rot = quat_multiply_xyzw(
        np.broadcast_to(inverse_quat, output["root_rot"].shape),
        output["root_rot"].astype(np.float64),
    )
    quaternion_inverse_error = float(
        np.max(np.abs(recovered_rot - source["root_rot"].astype(np.float64)))
    )
    pairwise_error = _max_pairwise_distance_error(source["root_pos"], output["root_pos"])
    vector_inverse_error = 0.0
    for field in sorted(set(source) & OPTIONAL_WORLD_VECTOR_FIELDS):
        recovered = output[field].astype(np.float64) @ inverse_rotation.T
        vector_inverse_error = max(
            vector_inverse_error,
            float(np.max(np.abs(recovered - source[field].astype(np.float64)))),
        )
    errors = {
        "root_position_inverse_max_abs_error": position_inverse_error,
        "root_quaternion_inverse_max_abs_error": quaternion_inverse_error,
        "root_pairwise_distance_max_abs_error_m": pairwise_error,
        "world_vector_inverse_max_abs_error": vector_inverse_error,
    }
    if any(error > tolerance for error in errors.values()):
        raise MaterializationError(f"SE(2) inverse/rigidity verification failed: {errors}")
    if output_structure["root_rotation_max_norm_error"] > 1e-6:
        raise MaterializationError("output quaternion norm contract failed")
    return {
        **errors,
        "tolerance": tolerance,
        "proper_rotation_determinant": float(np.linalg.det(rotation)),
        "root_z_bit_exact": True,
        "fps_bit_exact": True,
        "dof_pos_bit_exact": True,
        "local_body_pos_bit_exact": True,
        "link_body_list_bit_exact": True,
        "no_mirror": True,
        "no_resample_or_topp": True,
        "world_vector_fields_rotated": source_structure["world_vector_fields_present"],
    }


def _find_registry_asset(registry: Mapping[str, Any], asset_id: str) -> dict[str, Any]:
    rows = registry.get("inputs")
    if not isinstance(rows, list):
        raise MaterializationError("source registry assets must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("asset_id") == asset_id]
    if len(matches) != 1:
        raise MaterializationError(f"source registry must contain exactly one {asset_id}")
    return matches[0]


def _find_selection_asset(selection: Mapping[str, Any], asset_id: str) -> dict[str, Any]:
    rows = selection.get("assets")
    if not isinstance(rows, list):
        raise MaterializationError("selection assets must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("asset_id") == asset_id]
    if len(matches) != 1:
        raise MaterializationError(f"selection must contain exactly one {asset_id}")
    return matches[0]


def validate_plan(path: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    expected_sha = require_sha(expected_sha, "--expected-prereg-sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise MaterializationError(f"prereg sha256 {actual_sha} != {expected_sha}")
    plan = read_json(path, "SE(2) preregistration")
    exact_plan_keys = {
        "schema_version", "status", "asset_id", "human_name", "scope",
        "formal_eligible", "training_authorized", "hardware_authorized",
        "schema2_materialized", "consumer", "selection_result", "candidate_id",
        "source_registry", "source_motion", "transform", "payload_contract",
        "fallback_policy", "output_contract", "next_gate",
    }
    if set(plan) != exact_plan_keys:
        raise MaterializationError(
            f"plan keys changed: missing={sorted(exact_plan_keys - set(plan))} "
            f"unknown={sorted(set(plan) - exact_plan_keys)}"
        )
    if plan.get("schema_version") != 1 or plan.get("status") != PLAN_STATUS:
        raise MaterializationError("plan must be schema 1 and preregistered_not_executed")
    asset_id = plan.get("asset_id")
    if not isinstance(asset_id, str) or not SAFE_ASSET_ID.fullmatch(asset_id):
        raise MaterializationError("asset_id is unsafe")
    expected = ALLOWED_ASSETS.get(asset_id)
    if expected is None:
        raise MaterializationError("only the frozen Franco backhand-loop B/C primaries are supported")
    if plan.get("human_name") != expected["human_name"]:
        raise MaterializationError("human_name changed")
    expected_scope = (
        f"CPU-only exact whole-motion ground-preserving SE(2) materialization of the frozen "
        f"{asset_id.rsplit('_', 1)[-1].upper()} primary; no fallback selection, schema-2, "
        "simulator, training, TOPP or hardware"
    )
    if plan.get("scope") != expected_scope:
        raise MaterializationError("scope changed or overclaims this stage")
    if plan.get("candidate_id") != expected["candidate_id"]:
        raise MaterializationError("candidate_id is not the frozen primary; fallback is forbidden")
    if any(plan.get(field) is not False for field in (
        "formal_eligible", "training_authorized", "hardware_authorized", "schema2_materialized"
    )):
        raise MaterializationError("formal/training/hardware/schema2 claims must remain false")

    consumer = require_repo_binding(plan.get("consumer"), "consumer")
    if consumer != Path(__file__).resolve():
        raise MaterializationError("consumer path must name this script")
    selection_path = require_repo_binding(
        plan.get("selection_result"), "selection_result", expected=SELECTION_RESULT
    )
    if plan["selection_result"]["sha256"] != SELECTION_RESULT_SHA256:
        raise MaterializationError("selection result SHA is not the frozen main result")
    selection = read_json(selection_path, "selection result")
    if selection.get("status") != SELECTION_STATUS or selection.get("primary_count") != 2:
        raise MaterializationError("selection result is not the complete frozen B/C primary pair")
    selection_row = _find_selection_asset(selection, asset_id)
    primary = selection_row.get("selected_primary")
    if not isinstance(primary, dict) or primary.get("rank") != 0:
        raise MaterializationError("selection row lacks exactly the rank-0 primary")
    if primary.get("candidate_id") != expected["candidate_id"]:
        raise MaterializationError("candidate is not the frozen primary; fallback is forbidden here")
    if primary.get("source_motion_sha256") != expected["source_sha256"]:
        raise MaterializationError("primary source motion SHA mismatch")
    if primary.get("translation_w_m") != expected["translation_w_m"] or primary.get("yaw_deg") != expected["yaw_deg"]:
        raise MaterializationError("primary transform differs from the frozen Step-A row")

    registry_path = require_repo_binding(
        plan.get("source_registry"), "source_registry", expected=SOURCE_REGISTRY
    )
    if plan["source_registry"]["sha256"] != SOURCE_REGISTRY_SHA256:
        raise MaterializationError("source registry SHA mismatch")
    registry_row = _find_registry_asset(read_json(registry_path, "source registry"), asset_id)
    source = require_absolute_binding(plan.get("source_motion"), "source_motion")
    if source != registry_row.get("input"):
        raise MaterializationError("source motion binding differs from exact counterfactual registry")
    if source["sha256"] != expected["source_sha256"] or source["bytes"] != expected["source_bytes"]:
        raise MaterializationError("source motion binding differs from frozen asset contract")
    if registry_row.get("frames") != expected["frames"]:
        raise MaterializationError("source frame count mismatch")

    transform = plan.get("transform")
    exact_transform = {
        "semantics": "proper_atomic_whole_motion_ground_preserving_SE2_left_action_in_GMR_world",
        "translation_w_m": expected["translation_w_m"],
        "yaw_deg": expected["yaw_deg"],
        "root_rotation_rule": "q_out_xyzw=q_yaw_xyzw_left_multiply_q_source_xyzw",
        "root_position_rule": "p_out=Rz(yaw)*p_source+translation",
        "world_vector_rule": "v_out=Rz(yaw)*v_source_no_translation",
        "mirror": False,
        "per_frame_edit": False,
        "joint_edit": False,
        "time_edit_or_topp": False,
    }
    if transform != exact_transform:
        raise MaterializationError("transform contract differs from the frozen primary SE(2)")
    payload_contract = plan.get("payload_contract")
    exact_payload_contract = {
        "required_keys": sorted(BASE_PAYLOAD_KEYS),
        "optional_world_vector_fields": sorted(OPTIONAL_WORLD_VECTOR_FIELDS),
        "unknown_fields": "fail_closed",
        "fps": 30,
        "frames": expected["frames"],
        "root_rotation_convention": "xyzw",
        "local_body_pos": None,
        "link_body_list": None,
        "inverse_and_rigidity_tolerance": 1e-12,
    }
    if payload_contract != exact_payload_contract:
        raise MaterializationError("payload contract changed")
    if plan.get("fallback_policy") != {
        "automatic_fallback": False,
        "materialization_or_internal_failure": "stop_this_asset",
        "external_table_or_net_failure": "future_selector_resolve_only",
    }:
        raise MaterializationError("fallback policy changed")
    output = plan.get("output_contract")
    if not isinstance(output, dict):
        raise MaterializationError("output_contract must be an object")
    output_root = output.get("output_root")
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise MaterializationError("output root must be absolute")
    if output_root != expected["output_root"]:
        raise MaterializationError("output root differs from the frozen no-clobber namespace")
    exact_output = {
        "output_root": output_root,
        "motion_filename": f"{asset_id}.{expected['candidate_id'][:12]}.se2.gmr.pkl",
        "report_filename": "materialization_report.json",
        "output_root_must_not_exist": True,
        "no_clobber": True,
        "report_published_last": True,
    }
    if output != exact_output:
        raise MaterializationError("output contract changed")
    if plan.get("next_gate") != {
        "authorized": "schema2_materialization_preregistration_only",
        "status": "blocked_schema2_L0_L1_table_net_dynamics_not_run",
    }:
        raise MaterializationError("next gate overclaims materialization")
    return plan, actual_sha


def inspect_inputs(plan: Mapping[str, Any]) -> dict[str, Any]:
    source_path = verify_absolute_binding(plan["source_motion"], "source_motion")
    payload = load_bound_pickle(source_path)
    frames = plan["payload_contract"]["frames"]
    structure = validate_payload(payload, frames=frames)
    transformed = transform_payload(
        payload,
        translation_w_m=plan["transform"]["translation_w_m"],
        yaw_deg=plan["transform"]["yaw_deg"],
    )
    invariants = verify_transform(
        payload,
        transformed,
        translation_w_m=plan["transform"]["translation_w_m"],
        yaw_deg=plan["transform"]["yaw_deg"],
        tolerance=plan["payload_contract"]["inverse_and_rigidity_tolerance"],
    )
    return {
        "source_path": source_path,
        "payload": payload,
        "transformed": transformed,
        "structure": structure,
        "invariants": invariants,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def publish_report_last(staging: Path, output_root: Path, motion_name: str, report_name: str) -> None:
    if output_root.exists():
        raise MaterializationError(f"output root already exists: {output_root}")
    ensure_no_symlink_components(output_root.parent, "output root parent")
    try:
        output_root.mkdir()
    except FileExistsError:
        raise MaterializationError(f"output root appeared during publication: {output_root}") from None
    _fsync_directory(output_root.parent)
    try:
        os.link(staging / motion_name, output_root / motion_name)
        _fsync_directory(output_root)
        os.link(staging / report_name, output_root / report_name)
        _fsync_directory(output_root)
    except Exception:
        # Roll back only links that are still the exact inodes we created.  A
        # concurrent foreign file must never be removed by cleanup.
        for name in (report_name, motion_name):
            source = staging / name
            destination = output_root / name
            try:
                if source.exists() and destination.exists() and os.path.samefile(source, destination):
                    destination.unlink()
            except OSError:
                pass
        try:
            output_root.rmdir()
        except OSError:
            pass
        _fsync_directory(output_root.parent)
        raise


def consume(
    plan: Mapping[str, Any], plan_path: Path, plan_sha: str, evidence: Mapping[str, Any]
) -> Path:
    output = plan["output_contract"]
    output_root = Path(output["output_root"])
    if output_root.exists():
        raise MaterializationError(f"output root already exists: {output_root}")
    ensure_no_symlink_components(output_root.parent, "output root parent")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_components(output_root.parent, "output root parent")
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    try:
        staging.mkdir()
    except FileExistsError:
        raise MaterializationError(f"staging path already exists: {staging}") from None
    motion_name = output["motion_filename"]
    report_name = output["report_filename"]
    try:
        motion_path = staging / motion_name
        _write_exclusive(
            motion_path,
            pickle.dumps(evidence["transformed"], protocol=pickle.HIGHEST_PROTOCOL),
        )
        reloaded = load_bound_pickle(motion_path)
        serialized_invariants = verify_transform(
            evidence["payload"],
            reloaded,
            translation_w_m=plan["transform"]["translation_w_m"],
            yaw_deg=plan["transform"]["yaw_deg"],
            tolerance=plan["payload_contract"]["inverse_and_rigidity_tolerance"],
        )
        report = {
            "schema_version": 1,
            "status": RESULT_STATUS,
            "completed_utc": utc_now(),
            "scope": "exact whole-motion root SE(2) only; not schema-2, formal, training, simulator, dynamics, safety, returnability or hardware evidence",
            "asset_id": plan["asset_id"],
            "candidate_id": plan["candidate_id"],
            "preregistration": {"path": str(plan_path), "sha256": plan_sha},
            "consumer": plan["consumer"],
            "selection_result": plan["selection_result"],
            "source_registry": plan["source_registry"],
            "source_motion": plan["source_motion"],
            "output_motion": {
                "path": str(output_root / motion_name),
                "bytes": motion_path.stat().st_size,
                "sha256": sha256_file(motion_path),
            },
            "transform": plan["transform"],
            "structure": evidence["structure"],
            "invariants": serialized_invariants,
            "formal_eligible": False,
            "training_authorized": False,
            "hardware_authorized": False,
            "schema2_materialized": False,
            "fallback_advanced": False,
            "next_gate": plan["next_gate"],
            "limitations": [
                "Only the floating-root whole-motion SE(2) has been materialized.",
                "Schema-2, L0 static, vendor L1 self-collision, table/net swept clearance, dynamics and production-PD replay remain unrun.",
                "A materialization or internal failure stops this asset; only a later external table/net failure may ask the frozen selector to advance.",
            ],
        }
        _write_exclusive(staging / report_name, _json_bytes(report))
        publish_report_last(staging, output_root, motion_name, report_name)
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / report_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("static", help="validate only tracked repository contracts")
    subparsers.add_parser("inspect", help="verify and transform the private source without writing")
    subparsers.add_parser("consume", help="publish the transformed motion and report once")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan_path = args.prereg.resolve()
        plan, plan_sha = validate_plan(plan_path, args.expected_prereg_sha256)
        if args.command == "static":
            print(
                f"[motion-se2] PASS static asset={plan['asset_id']} "
                f"candidate={plan['candidate_id']} prereg_sha256={plan_sha}"
            )
            return 0
        evidence = inspect_inputs(plan)
        if args.command == "inspect":
            print(
                f"[motion-se2] PASS inspect asset={plan['asset_id']} "
                f"frames={evidence['structure']['frames']} no_write=true"
            )
            return 0
        report = consume(plan, plan_path, plan_sha, evidence)
        print(f"[motion-se2] PASS consume report={report}")
        return 0
    except (MaterializationError, OSError, TypeError, ValueError, pickle.PickleError) as exc:
        print(f"[motion-se2] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
