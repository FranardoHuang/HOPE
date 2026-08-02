#!/usr/bin/env python3
"""Rebuild one admitted canonical-racket PKL as a measured-teacher 50 Hz motion NPZ.

The existing ChingMu converter dropped the paddle channels.  This no-clobber materializer both
rebuilds joint/body kinematics from the repaired PKL and restores the measured paddle teacher.  It
aligns the clocks at the manifest-bound contact frame and applies the same heading normalization to
robot FK and measured paddle.  It never blesses the old v11 bank or copies corrected teacher
channels onto stale robot kinematics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


ROBOT_BUTT_TO_BLADE_AXIS_LOCAL = np.asarray(
    [1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)], dtype=np.float64
)
ROBOT_RIGID_VISUAL_MESH_SHA256 = (
    "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd"
)
EXPECTED_MJCF_SHA256 = (
    "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97"
)


class MaterializationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_savez_no_replace(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a complete NPZ then atomically publish it without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _selected_binding(
    *, manifest_path: Path, catalog_path: Path, uid: str
) -> tuple[dict, dict, str, str]:
    manifest = json.loads(manifest_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    rows = [row for row in manifest.get("units", []) if row.get("uid") == uid]
    clips = catalog.get("clips", [])
    selected = [row for row in clips if row.get("uid") == uid]
    if (
        catalog.get("n_clips") != 73
        or len(clips) != 73
        or len({row.get("uid") for row in clips}) != 73
        or catalog.get("excluded") != ["Take_085_unit00_FH"]
        or len(rows) != 1
        or len(selected) != 1
    ):
        raise MaterializationError("UID is not bound to the exact reviewed 73-action set")
    row, clip = rows[0], selected[0]
    if (
        row.get("npz_sha256") != clip.get("sha256")
        or row.get("T") != clip.get("T")
        or row.get("hit_frame_50") != clip.get("hit_frame_50")
    ):
        raise MaterializationError("manifest/catalog selected rows disagree")
    return row, clip, _sha256(manifest_path), _sha256(catalog_path)


def _resample(values: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise MaterializationError(f"measured array must have shape (T,3), got {values.shape}")
    x = np.arange(len(values), dtype=np.float64)
    result = np.column_stack(
        [np.interp(source_rows, x, values[:, axis]) for axis in range(3)]
    )
    return result


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quat_slerp(left: np.ndarray, right: np.ndarray, blend: np.ndarray) -> np.ndarray:
    dot = np.sum(left * right, axis=-1, keepdims=True)
    right = np.where(dot < 0.0, -right, right)
    dot = np.abs(dot)
    linear = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe_sin = np.where(linear, 1.0, sin_theta)
    weight_left = np.where(linear, 1.0 - blend, np.sin((1.0 - blend) * theta) / safe_sin)
    weight_right = np.where(linear, blend, np.sin(blend * theta) / safe_sin)
    result = weight_left * left + weight_right * right
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _axis_angle_from_quat(value: np.ndarray) -> np.ndarray:
    value = value * np.where(value[..., :1] < 0.0, -1.0, 1.0)
    magnitude = np.linalg.norm(value[..., 1:], axis=-1)
    half_angle = np.arctan2(magnitude, value[..., 0])
    safe = np.where(magnitude > 1.0e-8, magnitude, 1.0)
    axis = value[..., 1:] / safe[..., None]
    return np.where(
        (magnitude > 1.0e-8)[..., None],
        axis * (2.0 * half_angle)[..., None],
        value[..., 1:] * 2.0,
    )


def _so3_derivative(rotation: np.ndarray, dt: float) -> np.ndarray:
    inverse_previous = rotation[:-2].copy()
    inverse_previous[..., 1:] *= -1.0
    relative = _quat_mul(rotation[2:], inverse_previous)
    omega = _axis_angle_from_quat(relative) / (2.0 * float(dt))
    return np.concatenate((omega[:1], omega, omega[-1:]), axis=0)


def _resample_qpos(source_qpos: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    source = np.asarray(source_qpos, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 38 or len(source) < 2:
        raise MaterializationError(
            f"canonical retarget qpos must have shape (T,38), T>=2, got {source.shape}"
        )
    phase = np.clip(np.asarray(source_rows, dtype=np.float64), 0.0, len(source) - 1.0)
    lower = np.floor(phase).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    blend = (phase - lower)[:, None]
    result = source[lower] * (1.0 - blend) + source[upper] * blend
    result[:, 3:7] = _quat_slerp(
        source[lower, 3:7], source[upper, 3:7], blend
    )
    return result


def _read_joint_order(contract_path: Path) -> tuple[list[str], list[str], list[int]]:
    document = json.loads(contract_path.read_text())
    if document.get("expected_joint_count") != 31:
        raise MaterializationError("joint-order contract is not the A3 31-joint contract")
    repo_root = contract_path.parent.parent

    def names(label: str) -> list[str]:
        descriptor = document.get(label)
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
            raise MaterializationError(f"joint-order contract lacks {label}.path")
        path = (repo_root / descriptor["path"]).resolve()
        rows = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(rows) != 31 or len(set(rows)) != 31:
            raise MaterializationError(f"{label} is not one complete 31-joint order")
        return rows

    source_names = names("source_order")
    target_names = names("target_order")
    indices = [int(value) for value in document.get("target_from_source_indices", ())]
    expected = [source_names.index(name) for name in target_names]
    if indices != expected:
        raise MaterializationError("joint-order contract permutation disagrees with its names")
    return source_names, target_names, indices


def _yaw_of_quat(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotate_world(
    position: np.ndarray, normal: np.ndarray, body_quat: np.ndarray, *, theta: float, pivot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cosine, sine = math.cos(theta), math.sin(theta)

    def rotate_point(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).copy()
        x = result[..., 0].copy() - float(pivot[0])
        y = result[..., 1].copy() - float(pivot[1])
        result[..., 0] = float(pivot[0]) + x * cosine - y * sine
        result[..., 1] = float(pivot[1]) + x * sine + y * cosine
        return result

    def rotate_vector(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).copy()
        x = result[..., 0].copy()
        y = result[..., 1].copy()
        result[..., 0] = x * cosine - y * sine
        result[..., 1] = x * sine + y * cosine
        return result

    qz = np.asarray([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])
    rotated_quat = _quat_mul(np.broadcast_to(qz, body_quat.shape), body_quat)
    rotated_quat /= np.linalg.norm(rotated_quat, axis=-1, keepdims=True)
    return rotate_point(position), rotate_vector(normal), rotated_quat


def _rebuild_kinematics(
    *,
    arrays: dict[str, np.ndarray],
    retarget: dict,
    source_rows: np.ndarray,
    manifest_row: dict,
    model_path: Path,
    joint_order_contract_path: Path,
) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
    try:
        import mujoco
    except ImportError as exc:
        raise MaterializationError("canonical motion rebuild requires MuJoCo") from exc

    source_names, target_names, target_from_source = _read_joint_order(
        joint_order_contract_path
    )
    qpos = _resample_qpos(np.asarray(retarget["qpos"]), source_rows)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    root_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint")
    )
    if root_id < 0 or int(model.jnt_qposadr[root_id]) != 0:
        raise MaterializationError("canonical MJCF pelvis_free_joint contract changed")
    qpos_adrs = []
    for name in source_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0:
            raise MaterializationError(f"canonical MJCF lacks source joint {name!r}")
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
    if qpos_adrs != list(range(7, 38)):
        raise MaterializationError(
            f"canonical MJCF no longer uses the GMR source qpos order: {qpos_adrs}"
        )

    body_names = [str(value) for value in np.asarray(arrays["body_names"]).tolist()]
    body_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in body_names
    ]
    if any(value < 0 for value in body_ids):
        missing = [name for name, value in zip(body_names, body_ids) if value < 0]
        raise MaterializationError(f"canonical MJCF lacks motion bodies {missing}")
    frames = len(source_rows)
    body_pos = np.empty((frames, len(body_ids), 3), dtype=np.float64)
    body_quat = np.empty((frames, len(body_ids), 4), dtype=np.float64)
    body_com = np.empty((frames, len(body_ids), 3), dtype=np.float64)
    for frame in range(frames):
        data.qpos[:] = model.qpos0
        data.qpos[:38] = qpos[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]
        body_com[frame] = data.xipos[body_ids]

    pelvis_index = body_names.index("pelvis_link")
    theta = -_yaw_of_quat(body_quat[0, pelvis_index])
    declared = math.radians(float(manifest_row["yaw_norm_deg"]))
    if abs(math.atan2(math.sin(theta - declared), math.cos(theta - declared))) > math.radians(0.02):
        raise MaterializationError(
            "retarget frame-0 heading disagrees with manifest yaw_norm_deg: "
            f"computed={math.degrees(theta):.6f} declared={math.degrees(declared):.6f}"
        )
    pivot = body_pos[0, pelvis_index, :2].copy()
    body_pos, _, body_quat = _rotate_world(
        body_pos,
        np.zeros_like(body_pos),
        body_quat,
        theta=theta,
        pivot=pivot,
    )
    body_com_rotated, _, _ = _rotate_world(
        body_com,
        np.zeros_like(body_com),
        np.broadcast_to([1.0, 0.0, 0.0, 0.0], (*body_com.shape[:-1], 4)).copy(),
        theta=theta,
        pivot=pivot,
    )
    fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    body_lin = np.gradient(body_com_rotated, dt, axis=0)
    body_ang = np.stack(
        [_so3_derivative(body_quat[:, index], dt) for index in range(len(body_ids))],
        axis=1,
    )
    target_joint_pos = qpos[:, 7:38][:, target_from_source]
    rebuilt = dict(arrays)
    rebuilt.update(
        {
            "joint_pos": target_joint_pos.astype(np.float32),
            "joint_vel": np.gradient(target_joint_pos, dt, axis=0).astype(np.float32),
            "body_pos_w": body_pos.astype(np.float32),
            "body_quat_w": body_quat.astype(np.float32),
            "body_lin_vel_w": body_lin.astype(np.float32),
            "body_ang_vel_w": body_ang.astype(np.float32),
        }
    )
    return rebuilt, theta, pivot


def build_arrays(
    *,
    motion_path: Path,
    retarget_path: Path,
    retarget_report_path: Path,
    manifest_path: Path,
    catalog_path: Path,
    uid: str,
    model_path: Path,
    joint_order_contract_path: Path,
) -> dict[str, np.ndarray]:
    actual_model_sha256 = _sha256(model_path)
    if actual_model_sha256 != EXPECTED_MJCF_SHA256:
        raise MaterializationError(
            "canonical MJCF SHA-256 changed: "
            f"expected {EXPECTED_MJCF_SHA256}, got {actual_model_sha256}"
        )
    row, clip, manifest_sha, catalog_sha = _selected_binding(
        manifest_path=manifest_path, catalog_path=catalog_path, uid=uid
    )
    motion_sha = _sha256(motion_path)
    if motion_sha != row.get("npz_sha256") or motion_sha != clip.get("sha256"):
        raise MaterializationError("input motion NPZ SHA-256 differs from manifest/catalog")
    with motion_path.open("rb") as stream:
        with np.load(stream, allow_pickle=False) as motion:
            arrays = {key: np.asarray(motion[key]) for key in motion.files}
    with retarget_path.open("rb") as stream:
        retarget = pickle.load(stream)
    report_sha = _sha256(retarget_report_path)
    report = json.loads(retarget_report_path.read_text())
    if (
        report.get("schema_version") != 4
        or report.get("kind") != "chingmu_canonical_racket_full_phase_retarget_v4"
        or report.get("action_id") != uid
        or report.get("admitted") is not True
    ):
        raise MaterializationError("retarget report is not the admitted selected action")
    if retarget.get("wrist_mode") != "canonical_right_racket_full_phase_v4":
        raise MaterializationError("retarget is not the canonical full-phase racket lineage")
    if retarget.get("measured_racket_retarget_admitted") is not True:
        raise MaterializationError("retarget receipt did not admit this action")
    axis_local = np.asarray(
        retarget.get("measured_racket_robot_butt_to_blade_axis_local"),
        dtype=np.float64,
    ).reshape(-1)
    if axis_local.shape != (3,) or not np.array_equal(
        axis_local, ROBOT_BUTT_TO_BLADE_AXIS_LOCAL
    ):
        raise MaterializationError(
            "retarget does not bind the official butt-to-blade axis"
        )
    if (
        retarget.get("measured_racket_robot_rigid_visual_mesh_sha256")
        != ROBOT_RIGID_VISUAL_MESH_SHA256
    ):
        raise MaterializationError("retarget rigid-racket visual mesh SHA changed")
    joint_order_contract_sha = _sha256(joint_order_contract_path)
    if (
        retarget.get("joint_order_contract_id")
        != "a3-gmr-dof-pos-to-runtime-articulation-v1"
        or retarget.get("joint_order_contract_sha256") != joint_order_contract_sha
    ):
        raise MaterializationError(
            "retarget is not bound to the exact materializer joint-order contract"
        )
    if retarget.get("measured_racket_input_pkl_sha256") != row.get("source_pkl_sha256"):
        raise MaterializationError(
            "retarget input PKL SHA-256 differs from the motion manifest source"
        )
    if (
        retarget.get("measured_racket_uid") != uid
        or retarget.get("measured_racket_manifest_sha256") != manifest_sha
        or retarget.get("measured_racket_catalog_sha256") != catalog_sha
        or retarget.get("measured_racket_selected_hit_frame_120")
        != row.get("hit_frame_pkl_120")
        or retarget.get("measured_racket_retarget_receipt_sha256") != report_sha
    ):
        raise MaterializationError("retarget PKL/report/manifest/catalog binding disagrees")
    position_120 = np.asarray(retarget.get("measured_racket_site_pos_w_120"))
    normal_120 = np.asarray(retarget.get("measured_racket_normal_w_120"))
    long_axis_120 = np.asarray(retarget.get("measured_racket_long_axis_w_120"))
    if (
        position_120.shape != normal_120.shape
        or position_120.shape != long_axis_120.shape
        or position_120.shape[1:] != (3,)
    ):
        raise MaterializationError("retarget measured arrays are absent or malformed")
    target_frames = int(np.asarray(arrays["joint_pos"]).shape[0])
    if target_frames != int(row["T"]):
        raise MaterializationError(
            f"motion frame count {target_frames} differs from manifest {row['T']}"
        )
    target_fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    if target_fps != float(row["fps"]):
        raise MaterializationError("motion fps differs from manifest")
    source_hit = int(row["hit_frame_pkl_120"])
    target_hit = int(row["hit_frame_50"])
    source_rows = source_hit + (
        np.arange(target_frames, dtype=np.float64) - target_hit
    ) * (120.0 / target_fps)
    left_clamped = source_rows < 0.0
    right_clamped = source_rows > (len(position_120) - 1)
    if np.any(left_clamped & right_clamped):
        raise MaterializationError("source row cannot be clamped on both sides")
    position = _resample(position_120, source_rows)
    normal = _resample(normal_120, source_rows)
    long_axis = _resample(long_axis_120, source_rows)
    arrays, theta, pivot = _rebuild_kinematics(
        arrays=arrays,
        retarget=retarget,
        source_rows=source_rows,
        manifest_row=row,
        model_path=model_path,
        joint_order_contract_path=joint_order_contract_path,
    )
    dummy_quat = np.broadcast_to(
        [1.0, 0.0, 0.0, 0.0], (*position.shape[:-1], 4)
    ).copy()
    position, normal, _ = _rotate_world(
        position, normal, dummy_quat, theta=theta, pivot=pivot
    )
    _, long_axis, _ = _rotate_world(
        position, long_axis, dummy_quat, theta=theta, pivot=pivot
    )
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    long_norm = np.linalg.norm(long_axis, axis=-1, keepdims=True)
    if np.any(normal_norm <= 1.0e-12) or np.any(long_norm <= 1.0e-12):
        raise MaterializationError("interpolated measured orientation axis became degenerate")
    normal /= normal_norm
    long_axis /= long_norm
    if float(np.max(np.abs(np.sum(normal * long_axis, axis=-1)))) > 1.0e-3:
        raise MaterializationError(
            "interpolated measured face/long axes no longer define one orientation"
        )
    source_sha = str(retarget.get("measured_racket_source_sha256", ""))
    receipt_sha = str(retarget.get("measured_racket_retarget_receipt_sha256", ""))
    mount_sign = float(retarget.get("mount_normal_sign", 0.0))
    if mount_sign not in (-1.0, 1.0):
        raise MaterializationError(
            "canonical retarget must declare mount_normal_sign as scalar +1/-1"
        )
    for label, digest in (("source", source_sha), ("receipt", receipt_sha)):
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise MaterializationError(f"measured racket {label} SHA-256 is malformed")
    arrays.update(
        {
            "measured_racket_site_pos_w": position.astype(np.float32),
            "measured_racket_normal_w": normal.astype(np.float32),
            "measured_racket_long_axis_w": long_axis.astype(np.float32),
            "measured_racket_schema_version": np.asarray([4], dtype=np.int64),
            "measured_racket_position_semantics": np.asarray("physical_blade_center"),
            "measured_racket_normal_semantics": np.asarray("signed_physical_hitting_face"),
            "measured_racket_long_axis_semantics": np.asarray(
                "measured_paddle_butt_to_blade"
            ),
            "measured_racket_robot_mount_normal_sign": np.asarray(
                [int(mount_sign)], dtype=np.int8
            ),
            "measured_racket_robot_butt_to_blade_axis_local": (
                ROBOT_BUTT_TO_BLADE_AXIS_LOCAL.copy()
            ),
            "measured_racket_robot_rigid_visual_mesh_sha256": np.asarray(
                ROBOT_RIGID_VISUAL_MESH_SHA256
            ),
            "measured_racket_source_sha256": np.asarray(source_sha),
            "measured_racket_retarget_admitted": np.asarray([1], dtype=np.int64),
            "measured_racket_retarget_receipt_sha256": np.asarray(receipt_sha),
            "measured_racket_joint_order_contract_id": np.asarray(
                retarget["joint_order_contract_id"]
            ),
            "measured_racket_joint_order_contract_sha256": np.asarray(
                joint_order_contract_sha
            ),
            "measured_racket_heading_normalization_rad": np.asarray([theta], dtype=np.float64),
            "measured_racket_uid": np.asarray(uid),
            "measured_racket_input_motion_sha256": np.asarray(motion_sha),
            "measured_racket_manifest_sha256": np.asarray(manifest_sha),
            "measured_racket_catalog_sha256": np.asarray(catalog_sha),
            "measured_racket_source_row_min": np.asarray([source_rows.min()]),
            "measured_racket_source_row_max": np.asarray([source_rows.max()]),
            "measured_racket_left_clamped_frames": np.asarray(
                [int(np.count_nonzero(left_clamped))], dtype=np.int64
            ),
            "measured_racket_right_clamped_frames": np.asarray(
                [int(np.count_nonzero(right_clamped))], dtype=np.int64
            ),
        }
    )
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--retarget", type=Path, required=True)
    parser.add_argument("--retarget-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--joint-order-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = build_arrays(
        motion_path=args.motion.resolve(),
        retarget_path=args.retarget.resolve(),
        retarget_report_path=args.retarget_report.resolve(),
        manifest_path=args.manifest.resolve(),
        catalog_path=args.catalog.resolve(),
        uid=args.uid,
        model_path=args.xml.resolve(),
        joint_order_contract_path=args.joint_order_contract.resolve(),
    )
    _atomic_savez_no_replace(args.output, arrays)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": _sha256(args.output.resolve()),
                "frames": int(arrays["measured_racket_site_pos_w"].shape[0]),
                "measured_racket_retarget_admitted": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
