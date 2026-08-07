#!/usr/bin/env python3
"""Audit a materialized motion by recomputing the official MuJoCo racket site.

This is a read-only residual check.  It reconstructs the canonical robot from the NPZ's runtime
joint order, evaluates ``right_racket`` in the supplied MJCF, applies the retarget-selected face
sign, and compares that actual robot FK against the same-clock measured-paddle teacher channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from materialize_measured_racket_motion_npz import (
    MaterializationError,
    _read_joint_order,
    _selected_binding,
)


ROBOT_BUTT_TO_BLADE_AXIS_LOCAL = np.asarray(
    [1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)], dtype=np.float64
)
ROBOT_RIGID_VISUAL_MESH_SHA256 = (
    "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd"
)
# A plant is admissible only if its MJCF appears here.  Keyed rather than a bare set so a
# receipt can say WHICH plant produced a bank instead of only that some known one did.
KNOWN_MJCF_SHA256 = {
    "a3t2p5_0409": "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97",
    "a3p_p1_0807": "7bbda723f339bdf252a20622afa7a7d53a6fca97464252c66c6e1a45199bcae1",
}
# Kept so existing importers keep resolving to the plant the v4 bank was built against.
EXPECTED_MJCF_SHA256 = KNOWN_MJCF_SHA256["a3t2p5_0409"]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _unit(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm <= 1.0e-12):
        raise MaterializationError(f"{label} contains non-finite or zero vectors")
    return values / norm


def _orientation(long_axis: np.ndarray, face: np.ndarray) -> np.ndarray:
    long_axis = _unit(long_axis, "long axis")
    face = _unit(face, "face normal")
    if float(np.max(np.abs(np.sum(long_axis * face, axis=-1)))) > 1.0e-3:
        raise MaterializationError("face and long axes do not define one SO(3) orientation")
    third = _unit(np.cross(long_axis, face), "third orientation axis")
    return np.stack((long_axis, face, third), axis=-1)


def _so3_error_deg(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(actual, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _velocity_errors(
    actual_position: np.ndarray, target_position: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = np.gradient(actual_position, 1.0 / fps, axis=0)
    target = np.gradient(target_position, 1.0 / fps, axis=0)
    actual_speed = np.linalg.norm(actual, axis=-1)
    target_speed = np.linalg.norm(target, axis=-1)
    valid = (actual_speed > 0.10) & (target_speed > 0.10)
    direction = np.zeros(len(actual), dtype=np.float64)
    cosine = np.sum(actual[valid] * target[valid], axis=-1) / (
        actual_speed[valid] * target_speed[valid]
    )
    direction[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    relative = np.abs(actual_speed - target_speed) / np.maximum(target_speed, 0.10)
    return direction, relative, valid


def _atomic_json_no_replace(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit(
    *,
    motion_path: Path,
    model_path: Path,
    joint_order_contract_path: Path,
    manifest_path: Path,
    catalog_path: Path,
    uid: str,
) -> dict:
    actual_model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_model_sha256 not in KNOWN_MJCF_SHA256.values():
        raise MaterializationError(
            "MJCF is not a known plant: "
            f"got {actual_model_sha256}, known are {sorted(KNOWN_MJCF_SHA256.values())}"
        )
    try:
        import mujoco
    except ImportError as exc:
        raise MaterializationError("materialized-racket audit requires MuJoCo") from exc

    with np.load(motion_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = (
        "joint_pos",
        "body_names",
        "body_pos_w",
        "body_quat_w",
        "measured_racket_site_pos_w",
        "measured_racket_normal_w",
        "measured_racket_long_axis_w",
        "measured_racket_robot_mount_normal_sign",
        "measured_racket_robot_butt_to_blade_axis_local",
        "measured_racket_robot_rigid_visual_mesh_sha256",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise MaterializationError(f"materialized motion lacks {missing}")
    row, _, manifest_sha, catalog_sha = _selected_binding(
        manifest_path=manifest_path, catalog_path=catalog_path, uid=uid
    )
    if (
        np.asarray(arrays.get("measured_racket_schema_version")).reshape(-1).tolist()
        != [4]
        or str(np.asarray(arrays.get("measured_racket_uid"))) != uid
        or str(np.asarray(arrays.get("measured_racket_manifest_sha256"))) != manifest_sha
        or str(np.asarray(arrays.get("measured_racket_catalog_sha256"))) != catalog_sha
    ):
        raise MaterializationError("materialized motion identity/schema binding is not v4 exact")
    axis_local = np.asarray(
        arrays["measured_racket_robot_butt_to_blade_axis_local"], dtype=np.float64
    ).reshape(-1)
    mesh_sha = str(
        np.asarray(arrays["measured_racket_robot_rigid_visual_mesh_sha256"])
    )
    if axis_local.shape != (3,) or not np.array_equal(
        axis_local, ROBOT_BUTT_TO_BLADE_AXIS_LOCAL
    ):
        raise MaterializationError(
            "materialized motion does not bind the official butt-to-blade axis"
        )
    if mesh_sha != ROBOT_RIGID_VISUAL_MESH_SHA256:
        raise MaterializationError(
            "materialized motion rigid-racket visual mesh SHA changed"
        )
    hit_frame = row.get("hit_frame_50")

    source_names, target_names, target_from_source = _read_joint_order(
        joint_order_contract_path
    )
    joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(target_names):
        raise MaterializationError(
            f"runtime joint_pos must have {len(target_names)} columns, got {joint_pos.shape}"
        )
    source_joint_pos = np.empty_like(joint_pos)
    source_joint_pos[:, target_from_source] = joint_pos

    body_names = tuple(str(value) for value in arrays["body_names"].tolist())
    if "pelvis_link" not in body_names:
        raise MaterializationError("materialized body order lacks pelvis_link")
    pelvis_index = body_names.index("pelvis_link")
    root_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)[:, pelvis_index]
    root_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)[:, pelvis_index]
    measured_pos = np.asarray(arrays["measured_racket_site_pos_w"], dtype=np.float64)
    measured_normal = np.asarray(arrays["measured_racket_normal_w"], dtype=np.float64)
    measured_long = np.asarray(arrays["measured_racket_long_axis_w"], dtype=np.float64)
    frames = len(joint_pos)
    if any(
        value.shape != (frames, 3)
        for value in (root_pos, measured_pos, measured_normal, measured_long)
    ):
        raise MaterializationError("root/measured arrays do not share the joint time axis")
    sign_raw = np.asarray(
        arrays["measured_racket_robot_mount_normal_sign"]
    ).reshape(-1)
    if sign_raw.size != 1 or float(sign_raw[0]) not in (-1.0, 1.0):
        raise MaterializationError("measured robot mount-normal sign is not scalar +1/-1")
    sign = float(sign_raw[0])
    if type(hit_frame) is not int or hit_frame < 0 or hit_frame >= frames:
        raise MaterializationError(f"hit frame {hit_frame} is outside [0,{frames})")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_racket"))
    if site_id < 0:
        raise MaterializationError("canonical MJCF lacks the right_racket site")
    qpos_adrs = []
    for name in source_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0:
            raise MaterializationError(f"canonical MJCF lacks source joint {name!r}")
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
    if qpos_adrs != list(range(7, 38)):
        raise MaterializationError("canonical MJCF joint order no longer matches the source order")

    fk_pos = np.empty((frames, 3), dtype=np.float64)
    fk_normal = np.empty((frames, 3), dtype=np.float64)
    fk_long = np.empty((frames, 3), dtype=np.float64)
    for frame in range(frames):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[7:38] = source_joint_pos[frame]
        mujoco.mj_forward(model, data)
        fk_pos[frame] = data.site_xpos[site_id]
        rotation = data.site_xmat[site_id].reshape(3, 3)
        fk_normal[frame] = rotation[:, 1] * sign
        fk_long[frame] = rotation @ ROBOT_BUTT_TO_BLADE_AXIS_LOCAL

    pos_error = np.linalg.norm(fk_pos - measured_pos, axis=-1)
    cosine = np.clip(np.sum(fk_normal * measured_normal, axis=-1), -1.0, 1.0)
    face_error_deg = np.degrees(np.arccos(cosine))
    long_cosine = np.clip(np.sum(fk_long * measured_long, axis=-1), -1.0, 1.0)
    long_error_deg = np.degrees(np.arccos(long_cosine))
    so3_error_deg = _so3_error_deg(
        _orientation(fk_long, fk_normal), _orientation(measured_long, measured_normal)
    )
    fps_raw = np.asarray(arrays.get("fps")).reshape(-1)
    if fps_raw.size != 1 or float(fps_raw[0]) <= 0.0:
        raise MaterializationError("motion fps must be one positive scalar")
    velocity_direction_deg, velocity_relative, velocity_valid = _velocity_errors(
        fk_pos, measured_pos, float(fps_raw[0])
    )
    finite = bool(
        np.isfinite(pos_error).all()
        and np.isfinite(face_error_deg).all()
        and np.isfinite(long_error_deg).all()
        and np.isfinite(so3_error_deg).all()
        and np.isfinite(velocity_direction_deg).all()
        and np.isfinite(velocity_relative).all()
    )
    if not finite:
        raise MaterializationError("materialized racket residual contains non-finite values")
    gates = {
        "full_position_p95_le_0p05_m": float(np.percentile(pos_error, 95)) <= 0.05,
        "full_face_p95_le_10_deg": float(np.percentile(face_error_deg, 95)) <= 10.0,
        "full_long_axis_p95_le_10_deg": float(np.percentile(long_error_deg, 95)) <= 10.0,
        "full_so3_p95_le_10_deg": float(np.percentile(so3_error_deg, 95)) <= 10.0,
        "hit_position_le_0p05_m": float(pos_error[hit_frame]) <= 0.05,
        "hit_face_le_5_deg": float(face_error_deg[hit_frame]) <= 5.0,
        "hit_long_axis_le_5_deg": float(long_error_deg[hit_frame]) <= 5.0,
        "hit_so3_le_5_deg": float(so3_error_deg[hit_frame]) <= 5.0,
        "hit_velocity_direction_observable": bool(velocity_valid[hit_frame]),
        "hit_velocity_direction_le_15_deg": bool(velocity_valid[hit_frame])
        and float(velocity_direction_deg[hit_frame]) <= 15.0,
        "hit_velocity_relative_le_0p20": float(velocity_relative[hit_frame]) <= 0.20,
    }
    admitted = all(gates.values())
    return {
        "schema_version": 3,
        "kind": "materialized_measured_racket_fk_audit_v3",
        "uid": uid,
        "motion": str(motion_path.resolve()),
        "motion_sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        "model": str(model_path.resolve()),
        "frames": frames,
        "robot_mount_normal_sign": int(sign),
        "robot_butt_to_blade_axis_local": (
            ROBOT_BUTT_TO_BLADE_AXIS_LOCAL.tolist()
        ),
        "robot_rigid_visual_mesh_sha256": ROBOT_RIGID_VISUAL_MESH_SHA256,
        "position_error_m": _percentiles(pos_error),
        "face_error_deg": _percentiles(face_error_deg),
        "long_axis_error_deg": _percentiles(long_error_deg),
        "so3_error_deg": _percentiles(so3_error_deg),
        "hit": {
            "frame": int(hit_frame),
            "position_error_m": float(pos_error[hit_frame]),
            "face_error_deg": float(face_error_deg[hit_frame]),
            "long_axis_error_deg": float(long_error_deg[hit_frame]),
            "so3_error_deg": float(so3_error_deg[hit_frame]),
            "velocity_direction_deg": float(velocity_direction_deg[hit_frame]),
            "velocity_direction_observable": bool(velocity_valid[hit_frame]),
            "velocity_relative": float(velocity_relative[hit_frame]),
        },
        "gates": gates,
        "admitted": admitted,
        "finite": finite,
        "authorization": {
            "diagnostic_unauthorized": True,
            "training": False,
            "promotion": False,
            "deployment": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--joint-order-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        motion_path=args.motion.resolve(),
        model_path=args.xml.resolve(),
        joint_order_contract_path=args.joint_order_contract.resolve(),
        manifest_path=args.manifest.resolve(),
        catalog_path=args.catalog.resolve(),
        uid=args.uid,
    )
    _atomic_json_no_replace(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
