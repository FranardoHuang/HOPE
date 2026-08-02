#!/usr/bin/env python3
"""Retarget ChingMu paddle motion to the canonical A3 ``right_racket`` site.

This is the source repair for the old ``v11_hit_window_anchor`` lineage.  Unlike v11 it:

* resolves the current MJCF site by name and verifies its exact local transform;
* follows measured blade centre, signed physical face, long axis and point velocity over the full
  phase instead of solving only a contact window;
* chooses the red/black face per action from ``hits[].face_normal_hope`` (never by family name);
* emits a per-action residual receipt and publishes retargeted PKL only when all gates pass.

The input PKL is a trusted team artifact and is loaded with pickle.  Outputs are no-clobber.  This
tool does not compile the 50 Hz motion bank or authorize training by itself.
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
from typing import Any, Optional
from xml.etree import ElementTree

import numpy as np


FPS = 120.0
SITE_NAME = "right_racket"
EXPECTED_SITE_POS = np.array([0.21021, 0.032078, 0.032036], dtype=np.float64)
EXPECTED_MJCF_SHA256 = (
    "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97"
)
EXPECTED_URDF_SHA256 = (
    "0d83529cf808e2e68036f8168bd8b7a1c9a97d9c536eb9a14981ea4105d6b9ae"
)
ROBOT_BUTT_TO_BLADE_AXIS_LOCAL = np.asarray(
    [1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)], dtype=np.float64
)
ROBOT_RIGID_VISUAL_MESH_SHA256 = (
    "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd"
)
OPTIMIZED_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
# The original seven-arm-only solver could not satisfy the measured racket pose for every
# canonical body retarget even though the A3 still had waist margin.  Optimizing the three waist
# joints with a much stronger source-pose prior lets the canonical robot absorb human/robot
# proportion mismatch without moving the root or lower body.  These values are part of the
# receipt, not tunable CLI knobs.
# The old 300/20/20/0.1 balance was tuned while site-local +X was incorrectly treated as the
# paddle longitudinal axis.  With the URDF-visual butt-to-blade axis, that recipe sacrificed the
# full rigid-paddle pose (26/73 failures, mainly post-hit) to the finite-difference velocity term.
# Fixed canaries on the corrected geometry use a stronger instantaneous SE(3) target and retain
# velocity only as a light temporal tie-breaker.  A second corrected-axis full-bank pass showed
# four marginal post-hit failures at lower weights (three solver p95 position failures and one
# independent 50 Hz audit failure).  The frozen weights below were the first pre-registered ratio-
# preserving setting that passed those four canaries without loosening any residual or step gate.
# The independent materializer still recomputes point velocity and every FK residual.
W_POSITION = 12000.0
W_FACE = 1400.0
W_LONG_AXIS = 1400.0
W_POINT_VELOCITY = 0.02
W_ORIGINAL = np.array(
    [5.0, 5.0, 5.0, 0.20, 0.20, 0.20, 0.20, 0.02, 0.08, 0.04]
)
W_NEIGHBOR = 0.05
STEP_RAD = 0.12
DEFAULT_SOFT_LIMIT_MARGIN_FRACTION = 0.01
DEFAULT_VELOCITY_LIMIT_FRACTION = 0.90
DEFAULT_ACCELERATION_PROXY_RAD_S2 = 250.0
EXPECTED_CATALOG_SIZE = 73
EXPECTED_EXCLUDED_UIDS = ("Take_085_unit00_FH",)


class RetargetError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names_sha256(names: tuple[str, ...]) -> str:
    payload = json.dumps(list(names), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_urdf_motion_limits(
    path: Path, joint_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load exact position/velocity limits for the optimized joints.

    Effort is deliberately not returned: an effort scalar and a no-load velocity scalar do not
    define a torque-speed curve and therefore cannot authorize mechanical admission.
    """

    root = ElementTree.parse(str(path)).getroot()
    rows: dict[str, tuple[float, float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name not in joint_names:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise RetargetError(f"URDF joint {name!r} lacks a limit element")
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
            velocity = float(limit.attrib["velocity"])
        except (KeyError, ValueError) as exc:
            raise RetargetError(f"URDF joint {name!r} has invalid motion limits") from exc
        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and math.isfinite(velocity)
            and lower < upper
            and velocity > 0.0
        ):
            raise RetargetError(f"URDF joint {name!r} has non-physical motion limits")
        rows[name] = (lower, upper, velocity)
    missing = [name for name in joint_names if name not in rows]
    if missing:
        raise RetargetError(f"URDF lacks optimized-joint limits: {missing}")
    ordered = np.asarray([rows[name] for name in joint_names], dtype=np.float64)
    return ordered[:, 0], ordered[:, 1], ordered[:, 2]


def soft_position_bounds(
    lower: np.ndarray, upper: np.ndarray, margin_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink each URDF range symmetrically without inventing a hardware limit."""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if (
        lower.shape != upper.shape
        or lower.ndim != 1
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower >= upper)
        or not math.isfinite(margin_fraction)
        or margin_fraction < 0.0
        or margin_fraction >= 0.5
    ):
        raise RetargetError("invalid soft-position-limit inputs")
    margin = (upper - lower) * float(margin_fraction)
    return lower + margin, upper - margin


def constrained_frame_bounds(
    *,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
    velocity_rad_s: np.ndarray,
    fps: float,
    velocity_fraction: float,
    neighbor: Optional[np.ndarray],
    second_neighbor: Optional[np.ndarray],
    acceleration_proxy_rad_s2: Optional[float],
) -> list[tuple[float, float]]:
    """Intersect position, one-step velocity and second-difference proxy bounds.

    ``neighbor`` is the already solved frame nearer the hit anchor.  ``second_neighbor`` is the
    next frame inward, so the same expression applies while solving either forward or backward.
    The acceleration value is a diagnostic smoothness proxy, not an actuator authority limit.
    """

    lo = np.asarray(position_lower, dtype=np.float64).copy()
    hi = np.asarray(position_upper, dtype=np.float64).copy()
    velocity = np.asarray(velocity_rad_s, dtype=np.float64)
    if (
        lo.shape != hi.shape
        or lo.shape != velocity.shape
        or lo.ndim != 1
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or not np.isfinite(velocity).all()
        or np.any(lo >= hi)
        or np.any(velocity <= 0.0)
        or not math.isfinite(fps)
        or fps <= 0.0
        or not math.isfinite(velocity_fraction)
        or velocity_fraction <= 0.0
        or velocity_fraction > 1.0
    ):
        raise RetargetError("invalid dynamic-bound inputs")
    if second_neighbor is not None and neighbor is None:
        raise RetargetError("second_neighbor requires neighbor")
    if neighbor is not None:
        neighbor = np.asarray(neighbor, dtype=np.float64)
        if neighbor.shape != lo.shape or not np.isfinite(neighbor).all():
            raise RetargetError("invalid neighbor shape/value")
        max_step = velocity * float(velocity_fraction) / float(fps)
        lo = np.maximum(lo, neighbor - max_step)
        hi = np.minimum(hi, neighbor + max_step)
    if acceleration_proxy_rad_s2 is not None:
        if (
            not math.isfinite(acceleration_proxy_rad_s2)
            or acceleration_proxy_rad_s2 <= 0.0
        ):
            raise RetargetError("acceleration proxy must be finite and positive")
        if second_neighbor is not None:
            second_neighbor = np.asarray(second_neighbor, dtype=np.float64)
            if second_neighbor.shape != lo.shape or not np.isfinite(second_neighbor).all():
                raise RetargetError("invalid second_neighbor shape/value")
            center = 2.0 * neighbor - second_neighbor
            radius = float(acceleration_proxy_rad_s2) / float(fps * fps)
            lo = np.maximum(lo, center - radius)
            hi = np.minimum(hi, center + radius)
    if np.any(lo > hi + 1.0e-12):
        indices = np.flatnonzero(lo > hi + 1.0e-12).tolist()
        raise RetargetError(f"dynamic motion bounds are infeasible for joints {indices}")
    # Remove harmless round-off inversions at an exactly pinned boundary.
    midpoint = 0.5 * (lo + hi)
    lo = np.minimum(lo, midpoint)
    hi = np.maximum(hi, midpoint)
    return [(float(left), float(right)) for left, right in zip(lo, hi)]


def _load_joint_order_contract(path: Path) -> dict[str, Any]:
    """Load the content-bound GMR-source -> runtime joint-order bijection."""

    document = json.loads(path.read_text())
    if (
        document.get("schema_version") != 1
        or document.get("contract_id")
        != "a3-gmr-dof-pos-to-runtime-articulation-v1"
        or document.get("expected_joint_count") != 31
    ):
        raise RetargetError("unsupported A3 joint-order contract")
    repo_root = path.parent.parent

    def side(label: str, expected_name: str) -> tuple[str, ...]:
        descriptor = document.get(label)
        if not isinstance(descriptor, dict) or descriptor.get("name") != expected_name:
            raise RetargetError(f"joint-order contract lacks {label} identity")
        raw_path = descriptor.get("path")
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
            raise RetargetError(f"joint-order contract {label}.path is not repo-relative")
        order_path = (repo_root / raw_path).resolve()
        try:
            order_path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise RetargetError(f"joint-order contract {label}.path escapes repo") from exc
        if not order_path.is_file() or _sha256(order_path) != descriptor.get("file_sha256"):
            raise RetargetError(f"joint-order contract {label} file/SHA mismatch")
        names = tuple(
            line.strip()
            for line in order_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if (
            len(names) != 31
            or len(set(names)) != 31
            or _names_sha256(names) != descriptor.get("names_sha256")
        ):
            raise RetargetError(f"joint-order contract {label} names changed")
        return names

    source = side("source_order", "gmr_dof_pos")
    target = side("target_order", "runtime_articulation_joint_pos")
    if set(source) != set(target) or source == target:
        raise RetargetError("joint-order source/target domains are not the reviewed bijection")
    target_from_source = tuple(source.index(name) for name in target)
    source_from_target = tuple(target.index(name) for name in source)
    if list(target_from_source) != document.get("target_from_source_indices"):
        raise RetargetError("joint-order target_from_source_indices changed")
    if list(source_from_target) != document.get("source_from_target_indices"):
        raise RetargetError("joint-order source_from_target_indices changed")
    return {
        "path": path,
        "sha256": _sha256(path),
        "contract_id": document["contract_id"],
        "source_names": source,
        "target_names": target,
        "target_from_source_indices": target_from_source,
        "source_from_target_indices": source_from_target,
    }


def validate_mjcf_qpos_joint_order(
    model_names_by_qpos: tuple[str, ...], source_names: tuple[str, ...]
) -> None:
    """The retarget PKL is in GMR order; the optimization model must prove the same order."""

    if model_names_by_qpos != source_names:
        raise RetargetError(
            "canonical MJCF qpos joint order is not the content-bound GMR source order"
        )


def _unit(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norm <= 1.0e-12):
        raise RetargetError(f"{name} contains non-finite or zero vectors")
    return array / norm


def signed_dense_face_normal(
    dense_raw_normal: np.ndarray,
    hit_raw_normal: np.ndarray,
    hit_signed_face_normal: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Orient one raw dense normal series using the measured signed contact face."""

    dense = _unit(dense_raw_normal, name="paddle_normal_hope")
    raw_hit = _unit(np.asarray(hit_raw_normal)[None, :], name="hit raw normal")[0]
    signed_hit = _unit(
        np.asarray(hit_signed_face_normal)[None, :], name="hits[].face_normal_hope"
    )[0]
    dot = float(np.dot(raw_hit, signed_hit))
    if abs(dot) < 0.95:
        raise RetargetError(
            "signed hit face is not the same measured paddle plane as the dense raw normal "
            f"(abs dot={abs(dot):.6f})"
        )
    sign = 1.0 if dot >= 0.0 else -1.0
    return dense * sign, sign


def point_velocity(position: np.ndarray, fps: float = FPS) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or len(position) < 2:
        raise RetargetError("point trajectory must have shape (T,3), T>=2")
    return np.gradient(position, 1.0 / float(fps), axis=0)


def _percentiles(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _unit(left, name="angle left")
    right = _unit(right, name="angle right")
    return np.degrees(np.arccos(np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)))


def _velocity_errors(
    actual: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_speed = np.linalg.norm(actual, axis=-1)
    target_speed = np.linalg.norm(target, axis=-1)
    valid = (actual_speed > 0.10) & (target_speed > 0.10)
    direction = np.zeros(len(actual), dtype=np.float64)
    direction[valid] = _angle_deg(actual[valid], target[valid])
    relative = np.abs(actual_speed - target_speed) / np.maximum(target_speed, 0.10)
    return direction, relative, valid


def _atomic_bytes_no_replace(path: Path, payload: bytes) -> None:
    """Publish complete bytes without ever replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link(2) is an atomic no-replace publication on the same filesystem.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _select_action_contract(
    *, manifest_path: Path, catalog_path: Path, uid: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    manifest_rows = manifest.get("units")
    clips = catalog.get("clips")
    if not isinstance(manifest_rows, list) or not isinstance(clips, list):
        raise RetargetError("manifest units and catalog clips must be lists")
    if len({row.get("uid") for row in manifest_rows if isinstance(row, dict)}) != len(
        manifest_rows
    ):
        raise RetargetError("source manifest UIDs are not unique")
    if (
        catalog.get("n_clips") != EXPECTED_CATALOG_SIZE
        or len(clips) != EXPECTED_CATALOG_SIZE
        or len({row.get("uid") for row in clips if isinstance(row, dict)})
        != EXPECTED_CATALOG_SIZE
        or tuple(catalog.get("excluded", ())) != EXPECTED_EXCLUDED_UIDS
    ):
        raise RetargetError("catalog is not the exact reviewed 73-action set")
    rows = [row for row in manifest_rows if isinstance(row, dict) and row.get("uid") == uid]
    selected = [row for row in clips if isinstance(row, dict) and row.get("uid") == uid]
    if len(rows) != 1 or len(selected) != 1:
        raise RetargetError(f"UID {uid!r} is not an exact manifest/catalog join")
    row, clip = rows[0], selected[0]
    if (
        clip.get("sha256") != row.get("npz_sha256")
        or clip.get("T") != row.get("T")
        or clip.get("hit_frame_50") != row.get("hit_frame_50")
    ):
        raise RetargetError(f"UID {uid!r} manifest/catalog content disagrees")
    return row, clip, {
        "manifest_sha256": _sha256(manifest_path),
        "catalog_sha256": _sha256(catalog_path),
    }


def _orientation_from_long_face(long_axis: np.ndarray, face: np.ndarray) -> np.ndarray:
    long_axis = _unit(long_axis, name="orientation long axis")
    face = _unit(face, name="orientation face")
    dot = np.sum(long_axis * face, axis=-1)
    if np.max(np.abs(dot)) > 1.0e-3:
        raise RetargetError(
            "measured blade long axis and face normal do not define an orthogonal orientation"
        )
    third = _unit(np.cross(long_axis, face), name="orientation third axis")
    return np.stack((long_axis, face, third), axis=-1)


def _select_manifest_hit(
    hits: Any, *, selected_frame: Any, frames: int
) -> tuple[dict[str, Any], list[int]]:
    if not isinstance(hits, list) or not hits:
        raise RetargetError("unit JSON must contain at least one measured hit")
    hit_frames = []
    for index, row in enumerate(hits):
        if not isinstance(row, dict) or type(row.get("frame_local")) is not int:
            raise RetargetError(f"hits[{index}].frame_local must be an integer")
        frame = row["frame_local"]
        if frame < 0 or frame >= frames:
            raise RetargetError(
                f"hits[{index}].frame_local {frame} is outside [0,{frames})"
            )
        hit_frames.append(frame)
    if type(selected_frame) is not int or selected_frame < 0 or selected_frame >= frames:
        raise RetargetError("manifest-selected 120 Hz hit frame is invalid")
    matching_hits = [row for row in hits if row["frame_local"] == selected_frame]
    if len(matching_hits) != 1:
        raise RetargetError(
            f"manifest-selected hit frame {selected_frame} has {len(matching_hits)} unit JSON rows"
        )
    return matching_hits[0], hit_frames


def _so3_error_deg(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(actual, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _segment_residuals(
    *, hit_frame: int, position: np.ndarray, face: np.ndarray, long_axis: np.ndarray, so3: np.ndarray
) -> dict[str, Any]:
    segments: dict[str, Any] = {}
    ranges = {
        "pre_hit": slice(0, hit_frame),
        "hit": slice(hit_frame, hit_frame + 1),
        "post_hit": slice(hit_frame + 1, len(position)),
    }
    for label, selected in ranges.items():
        if len(position[selected]) == 0:
            continue
        segments[label] = {
            "frames": int(len(position[selected])),
            "position_m": _percentiles(position[selected]),
            "face_deg": _percentiles(face[selected]),
            "long_axis_deg": _percentiles(long_axis[selected]),
            "so3_deg": _percentiles(so3[selected]),
        }
    return segments


def solve_one(
    *,
    uid: str,
    pkl_path: Path,
    unit_npz_path: Path,
    unit_json_path: Path,
    manifest_path: Path,
    catalog_path: Path,
    model_path: Path,
    urdf_path: Path,
    joint_order_contract_path: Path,
    output_path: Path,
    report_path: Path,
    soft_limit_margin_fraction: float = DEFAULT_SOFT_LIMIT_MARGIN_FRACTION,
    velocity_limit_fraction: float = DEFAULT_VELOCITY_LIMIT_FRACTION,
    acceleration_proxy_rad_s2: Optional[float] = DEFAULT_ACCELERATION_PROXY_RAD_S2,
) -> dict[str, Any]:
    actual_model_sha256 = _sha256(model_path)
    if actual_model_sha256 != EXPECTED_MJCF_SHA256:
        raise RetargetError(
            "canonical MJCF SHA-256 changed: "
            f"expected {EXPECTED_MJCF_SHA256}, got {actual_model_sha256}"
        )
    actual_urdf_sha256 = _sha256(urdf_path)
    if actual_urdf_sha256 != EXPECTED_URDF_SHA256:
        raise RetargetError(
            "canonical URDF SHA-256 changed: "
            f"expected {EXPECTED_URDF_SHA256}, got {actual_urdf_sha256}"
        )
    try:
        import mujoco
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RetargetError("this solver requires MuJoCo and SciPy") from exc

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    manifest_row, catalog_row, action_binding = _select_action_contract(
        manifest_path=manifest_path, catalog_path=catalog_path, uid=uid
    )
    if (
        pkl_path.stem != uid
        or unit_npz_path.stem != uid
        or unit_json_path.stem != uid
    ):
        raise RetargetError("UID does not exactly match PKL/unit NPZ/unit JSON basenames")
    input_pkl_sha256 = _sha256(pkl_path)
    if input_pkl_sha256 != manifest_row.get("source_pkl_sha256"):
        raise RetargetError("input PKL SHA-256 differs from the selected manifest row")
    joint_order = _load_joint_order_contract(joint_order_contract_path)
    all_urdf_lower, all_urdf_upper, all_urdf_velocity = load_urdf_motion_limits(
        urdf_path, joint_order["source_names"]
    )
    all_urdf_soft_lower, all_urdf_soft_upper = soft_position_bounds(
        all_urdf_lower, all_urdf_upper, soft_limit_margin_fraction
    )
    root_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint")
    )
    if root_id < 0 or int(model.jnt_qposadr[root_id]) != 0:
        raise RetargetError("canonical MJCF pelvis_free_joint contract changed")
    model_source_joint_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in joint_order["source_names"]
    ]
    if any(value < 0 for value in model_source_joint_ids):
        raise RetargetError("canonical MJCF is missing a GMR source-order joint")
    model_source_qpos_adrs = tuple(
        int(model.jnt_qposadr[value]) for value in model_source_joint_ids
    )
    if model_source_qpos_adrs != tuple(range(7, 38)):
        raise RetargetError(
            "canonical MJCF GMR source joints do not occupy qpos[7:38] in source order"
        )
    model_names_by_qpos = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in sorted(
            model_source_joint_ids, key=lambda value: int(model.jnt_qposadr[value])
        )
    )
    validate_mjcf_qpos_joint_order(
        model_names_by_qpos, joint_order["source_names"]
    )
    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME))
    if site_id < 0:
        raise RetargetError(f"MJCF lacks site {SITE_NAME!r}")
    if not np.allclose(model.site_pos[site_id], EXPECTED_SITE_POS, rtol=0.0, atol=1.0e-12):
        raise RetargetError(
            f"{SITE_NAME} local position drifted: {model.site_pos[site_id].tolist()}"
        )
    if not np.allclose(
        model.site_quat[site_id], [1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1.0e-12
    ):
        raise RetargetError(f"{SITE_NAME} local orientation is not identity")

    joint_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in OPTIMIZED_JOINTS
    ]
    if any(value < 0 for value in joint_ids):
        raise RetargetError("MJCF is missing one or more canonical right-arm joints")
    qpos_adrs = np.asarray([model.jnt_qposadr[value] for value in joint_ids], dtype=np.int64)
    model_bounds = np.asarray(
        [np.asarray(model.jnt_range[value], dtype=np.float64) for value in joint_ids],
        dtype=np.float64,
    )
    urdf_lower, urdf_upper, urdf_velocity = load_urdf_motion_limits(
        urdf_path, OPTIMIZED_JOINTS
    )
    urdf_soft_lower, urdf_soft_upper = soft_position_bounds(
        urdf_lower, urdf_upper, soft_limit_margin_fraction
    )
    position_lower = np.maximum(model_bounds[:, 0], urdf_soft_lower)
    position_upper = np.minimum(model_bounds[:, 1], urdf_soft_upper)
    if np.any(position_lower >= position_upper):
        raise RetargetError("MJCF and URDF soft position ranges have an empty intersection")

    with pkl_path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "qpos" not in payload:
        raise RetargetError("retarget PKL must contain qpos")
    source_qpos = np.asarray(payload["qpos"], dtype=np.float64)
    if source_qpos.ndim != 2 or source_qpos.shape[1] != 38 or source_qpos.shape[0] < 3:
        raise RetargetError(
            "retarget PKL qpos must have exact GMR robot shape (T,38), T>=3, "
            f"got {source_qpos.shape}"
        )
    source_nq = int(source_qpos.shape[1])
    if source_nq < int(model.nq):
        # The 73-action PKLs contain the 38-D A3 articulation.  The canonical ping-pong MJCF
        # appends one 7-D free ball joint, making nq=45.  Accept that representation only when
        # the source width is an exact joint boundary and every omitted joint starts after it;
        # silently padding through the middle of a joint would corrupt both FK and the output.
        joint_qpos_adrs = np.asarray(model.jnt_qposadr, dtype=np.int64)
        if source_nq not in set(int(value) for value in joint_qpos_adrs):
            raise RetargetError(
                f"source qpos width {source_nq} is not a canonical MJCF joint boundary"
            )
        tail_joint_ids = np.flatnonzero(joint_qpos_adrs >= source_nq)
        if len(tail_joint_ids) == 0 or int(joint_qpos_adrs[tail_joint_ids[0]]) != source_nq:
            raise RetargetError("canonical MJCF tail-joint mapping is ambiguous")
        tail_joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
            for joint_id in tail_joint_ids
        ]
        if tail_joint_names != ["gate3_ball_free_joint"]:
            raise RetargetError(
                "the only supported canonical qpos suffix is gate3_ball_free_joint, got "
                f"{tail_joint_names}"
            )
    else:
        tail_joint_names = []
    qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (len(source_qpos), 1))
    qpos[:, :source_nq] = source_qpos
    frames = len(source_qpos)
    metadata = json.loads(unit_json_path.read_text())
    hit_frame = manifest_row.get("hit_frame_pkl_120")
    hit, hit_frames = _select_manifest_hit(
        metadata.get("hits"), selected_frame=hit_frame, frames=frames
    )

    with np.load(unit_npz_path, allow_pickle=False) as unit:
        required = (
            "paddle_blade_hope_m",
            "paddle_butt_hope_m",
            "paddle_normal_hope",
        )
        missing = [key for key in required if key not in unit.files]
        if missing:
            raise RetargetError(f"unit NPZ lacks measured paddle keys {missing}")
        blade = np.asarray(unit["paddle_blade_hope_m"], dtype=np.float64)
        butt = np.asarray(unit["paddle_butt_hope_m"], dtype=np.float64)
        raw_normal = np.asarray(unit["paddle_normal_hope"], dtype=np.float64)
    if blade.shape != (frames, 3) or butt.shape != blade.shape or raw_normal.shape != blade.shape:
        raise RetargetError(
            f"measured paddle arrays must match qpos ({frames},3), got "
            f"{blade.shape}/{butt.shape}/{raw_normal.shape}"
        )
    station = np.asarray(payload.get("station_xy_hope_m", [0.0, 0.0]), dtype=np.float64)
    if station.shape != (2,) or not np.isfinite(station).all():
        raise RetargetError("station_xy_hope_m must be a finite 2-vector")
    manifest_station = np.asarray(manifest_row.get("station_xy_hope_m"), dtype=np.float64)
    if manifest_station.shape != (2,) or not np.allclose(
        station, manifest_station, rtol=0.0, atol=1.0e-9
    ):
        raise RetargetError("PKL station_xy_hope_m differs from the selected manifest row")
    blade = blade.copy()
    butt = butt.copy()
    blade[:, :2] -= station
    butt[:, :2] -= station
    blade[:, 2] += 0.76
    butt[:, 2] += 0.76
    face_target, measured_face_sign = signed_dense_face_normal(
        raw_normal,
        raw_normal[hit_frame],
        np.asarray(hit["face_normal_hope"], dtype=np.float64),
    )
    long_target = _unit(blade - butt, name="measured blade long axis")
    velocity_target = point_velocity(blade)

    def fk(full_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[:] = full_qpos
        mujoco.mj_forward(model, data)
        return (
            np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
            np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
        )

    def minimize_frame(
        frame: int,
        *,
        robot_face_sign: float,
        initial: np.ndarray,
        neighbor: np.ndarray | None,
        second_neighbor: Optional[np.ndarray],
        neighbor_site: np.ndarray | None,
        direction: int,
        anchor: bool = False,
    ):
        base = qpos[frame].copy()
        original = qpos[frame, qpos_adrs].copy()

        def cost(arm: np.ndarray) -> float:
            base[qpos_adrs] = arm
            site, rotation = fk(base)
            value = W_POSITION * float(np.sum(np.square(site - blade[frame])))
            value += W_FACE * (1.0 - float(np.dot(rotation[:, 1] * robot_face_sign, face_target[frame])))
            robot_butt_to_blade = rotation @ ROBOT_BUTT_TO_BLADE_AXIS_LOCAL
            value += W_LONG_AXIS * (
                1.0 - float(np.dot(robot_butt_to_blade, long_target[frame]))
            )
            value += float(np.sum(W_ORIGINAL * np.square(arm - original)))
            if neighbor is not None:
                value += W_NEIGHBOR * float(np.sum(np.square(arm - neighbor)))
            if neighbor_site is not None:
                if direction > 0:
                    predicted_velocity = (site - neighbor_site) * FPS
                else:
                    predicted_velocity = (neighbor_site - site) * FPS
                value += W_POINT_VELOCITY * float(
                    np.sum(np.square(predicted_velocity - velocity_target[frame]))
                )
            return value

        frame_bounds = constrained_frame_bounds(
            position_lower=position_lower,
            position_upper=position_upper,
            velocity_rad_s=urdf_velocity,
            fps=FPS,
            velocity_fraction=velocity_limit_fraction,
            neighbor=None if anchor else neighbor,
            second_neighbor=None if anchor else second_neighbor,
            acceleration_proxy_rad_s2=acceleration_proxy_rad_s2,
        )
        if not anchor and neighbor is not None:
            # Preserve the reviewed geometric continuation cap when it is tighter than the URDF
            # velocity-derived interval (mainly the high-speed waist-roll joint).
            frame_bounds = [
                (max(lo, value - STEP_RAD), min(hi, value + STEP_RAD))
                for (lo, hi), value in zip(frame_bounds, neighbor)
            ]
            if any(lo > hi + 1.0e-12 for lo, hi in frame_bounds):
                raise RetargetError("reviewed step cap makes dynamic motion bounds infeasible")
        result = minimize(
            cost,
            np.clip(initial, [row[0] for row in frame_bounds], [row[1] for row in frame_bounds]),
            method="L-BFGS-B",
            bounds=frame_bounds,
            options={"maxiter": 300 if anchor else 100},
        )
        if (
            not bool(result.success)
            or not np.isfinite(float(result.fun))
            or not np.isfinite(np.asarray(result.x, dtype=np.float64)).all()
        ):
            # One deterministic retry from SciPy's last finite iterate prevents a transient line
            # search stop from being silently treated as an admitted retarget.
            result = minimize(
                cost,
                np.clip(
                    np.asarray(result.x, dtype=np.float64),
                    [row[0] for row in frame_bounds],
                    [row[1] for row in frame_bounds],
                ),
                method="L-BFGS-B",
                bounds=frame_bounds,
                options={"maxiter": 600 if anchor else 250, "maxls": 40},
            )
        return result

    catalog_face_sign = catalog_row.get("mount_normal_sign")
    catalog_face_sign_source = catalog_row.get("mount_normal_sign_source")
    if catalog_face_sign not in (-1, 1):
        raise RetargetError("catalog mount_normal_sign must be integer +1/-1")
    if not (
        isinstance(catalog_face_sign_source, str)
        and catalog_face_sign_source.startswith("measured-hit-signed-face-discovery-v1")
    ):
        raise RetargetError(
            "catalog mount_normal_sign is not bound to measured-hit sign discovery"
        )
    robot_face_sign = float(catalog_face_sign)
    optimizer_results = []
    original_hit = qpos[hit_frame, qpos_adrs].copy()
    hit_result = minimize_frame(
        hit_frame,
        robot_face_sign=robot_face_sign,
        initial=original_hit,
        neighbor=None,
        second_neighbor=None,
        neighbor_site=None,
        direction=0,
        anchor=True,
    )
    optimizer_results.append(("hit_catalog_sign", hit_frame, hit_result))
    if not bool(hit_result.success):
        raise RetargetError("catalog-signed contact optimization did not converge")
    solved = qpos.copy()
    solved[hit_frame, qpos_adrs] = hit_result.x

    for direction, frame_range in (
        (+1, range(hit_frame + 1, frames)),
        (-1, range(hit_frame - 1, -1, -1)),
    ):
        adjacent = hit_frame
        second_adjacent = hit_frame + 1 if direction < 0 and hit_frame + 1 < frames else None
        adjacent_site, _ = fk(solved[adjacent])
        for frame in frame_range:
            result = minimize_frame(
                frame,
                robot_face_sign=robot_face_sign,
                initial=solved[adjacent, qpos_adrs],
                neighbor=solved[adjacent, qpos_adrs],
                second_neighbor=(
                    None
                    if second_adjacent is None
                    else solved[second_adjacent, qpos_adrs]
                ),
                neighbor_site=adjacent_site,
                direction=direction,
            )
            optimizer_results.append(("forward" if direction > 0 else "backward", frame, result))
            if not bool(result.success):
                raise RetargetError(
                    f"frame optimizer did not converge at frame {frame}: "
                    f"status={result.status} message={result.message}"
                )
            solved[frame, qpos_adrs] = result.x
            second_adjacent = adjacent
            adjacent = frame
            adjacent_site, _ = fk(solved[frame])

    actual_pos = np.empty((frames, 3), dtype=np.float64)
    actual_face = np.empty((frames, 3), dtype=np.float64)
    actual_long = np.empty((frames, 3), dtype=np.float64)
    for frame in range(frames):
        actual_pos[frame], rotation = fk(solved[frame])
        actual_face[frame] = rotation[:, 1] * robot_face_sign
        actual_long[frame] = rotation @ ROBOT_BUTT_TO_BLADE_AXIS_LOCAL
    actual_velocity = point_velocity(actual_pos)
    pos_error = np.linalg.norm(actual_pos - blade, axis=-1)
    face_error = _angle_deg(actual_face, face_target)
    long_error = _angle_deg(actual_long, long_target)
    velocity_direction_error, velocity_relative_error, velocity_direction_valid = _velocity_errors(
        actual_velocity, velocity_target
    )
    actual_orientation = _orientation_from_long_face(actual_long, actual_face)
    target_orientation = _orientation_from_long_face(long_target, face_target)
    so3_error = _so3_error_deg(actual_orientation, target_orientation)
    optimized_qpos = solved[:, qpos_adrs]
    joint_step = np.abs(np.diff(optimized_qpos, axis=0))
    joint_velocity = np.diff(optimized_qpos, axis=0) * FPS
    joint_velocity_ratio = np.abs(joint_velocity) / urdf_velocity[None, :]
    joint_acceleration = np.diff(optimized_qpos, n=2, axis=0) * (FPS * FPS)
    soft_position_margin = np.minimum(
        optimized_qpos - position_lower[None, :],
        position_upper[None, :] - optimized_qpos,
    )
    acceleration_proxy_ok = (
        True
        if acceleration_proxy_rad_s2 is None
        else float(np.max(np.abs(joint_acceleration)))
        <= float(acceleration_proxy_rad_s2) + 1.0e-6
    )
    all_joint_qpos = solved[:, model_source_qpos_adrs]
    all_joint_velocity = np.diff(all_joint_qpos, axis=0) * FPS
    all_joint_velocity_ratio = np.abs(all_joint_velocity) / all_urdf_velocity[None, :]
    all_joint_acceleration = np.diff(all_joint_qpos, n=2, axis=0) * (FPS * FPS)
    all_joint_soft_position_margin = np.minimum(
        all_joint_qpos - all_urdf_soft_lower[None, :],
        all_urdf_soft_upper[None, :] - all_joint_qpos,
    )
    all_joint_acceleration_proxy_ok = (
        True
        if acceleration_proxy_rad_s2 is None
        else float(np.max(np.abs(all_joint_acceleration)))
        <= float(acceleration_proxy_rad_s2) + 1.0e-6
    )
    all_position_worst_flat = int(np.argmin(all_joint_soft_position_margin))
    all_position_worst_frame, all_position_worst_joint = np.unravel_index(
        all_position_worst_flat, all_joint_soft_position_margin.shape
    )
    all_velocity_worst_flat = int(np.argmax(all_joint_velocity_ratio))
    all_velocity_worst_frame, all_velocity_worst_joint = np.unravel_index(
        all_velocity_worst_flat, all_joint_velocity_ratio.shape
    )
    all_acceleration_worst_flat = int(np.argmax(np.abs(all_joint_acceleration)))
    all_acceleration_worst_frame, all_acceleration_worst_joint = np.unravel_index(
        all_acceleration_worst_flat, all_joint_acceleration.shape
    )
    gates = {
        "full_position_p95_le_0p05_m": float(np.percentile(pos_error, 95)) <= 0.05,
        "full_face_p95_le_10_deg": float(np.percentile(face_error, 95)) <= 10.0,
        "full_long_axis_p95_le_10_deg": float(np.percentile(long_error, 95)) <= 10.0,
        "full_so3_p95_le_10_deg": float(np.percentile(so3_error, 95)) <= 10.0,
        "hit_position_le_0p05_m": float(pos_error[hit_frame]) <= 0.05,
        "hit_face_le_5_deg": float(face_error[hit_frame]) <= 5.0,
        "hit_long_axis_le_5_deg": float(long_error[hit_frame]) <= 5.0,
        "hit_so3_le_5_deg": float(so3_error[hit_frame]) <= 5.0,
        "hit_velocity_direction_observable": bool(velocity_direction_valid[hit_frame]),
        "hit_velocity_direction_le_15_deg": bool(velocity_direction_valid[hit_frame])
        and float(velocity_direction_error[hit_frame]) <= 15.0,
        "hit_velocity_relative_le_0p20": float(velocity_relative_error[hit_frame]) <= 0.20,
        "optimized_joint_step_le_0p12_rad": float(np.max(joint_step)) <= STEP_RAD + 1.0e-9,
        "optimized_joint_inside_urdf_soft_position_bounds": float(
            np.min(soft_position_margin)
        )
        >= -1.0e-9,
        "optimized_joint_velocity_le_configured_urdf_fraction": float(
            np.max(joint_velocity_ratio)
        )
        <= float(velocity_limit_fraction) + 1.0e-9,
        "optimized_joint_acceleration_le_diagnostic_proxy": acceleration_proxy_ok,
        "all_source_joints_inside_urdf_soft_position_bounds": float(
            np.min(all_joint_soft_position_margin)
        )
        >= -1.0e-9,
        "all_source_joints_velocity_le_configured_urdf_fraction": float(
            np.max(all_joint_velocity_ratio)
        )
        <= float(velocity_limit_fraction) + 1.0e-9,
        "all_source_joints_acceleration_le_diagnostic_proxy": (
            all_joint_acceleration_proxy_ok
        ),
    }
    admitted = all(gates.values())
    report = {
        "schema_version": 4,
        "kind": "chingmu_canonical_racket_full_phase_retarget_v4",
        "action_id": uid,
        "sources": {
            "input_pkl": {"path": str(pkl_path), "sha256": input_pkl_sha256},
            "unit_npz": {"path": str(unit_npz_path), "sha256": _sha256(unit_npz_path)},
            "unit_json": {"path": str(unit_json_path), "sha256": _sha256(unit_json_path)},
            "mjcf": {"path": str(model_path), "sha256": _sha256(model_path)},
            "urdf": {"path": str(urdf_path), "sha256": actual_urdf_sha256},
            "joint_order_contract": {
                "path": str(joint_order_contract_path),
                "sha256": joint_order["sha256"],
                "contract_id": joint_order["contract_id"],
            },
            "manifest": {
                "path": str(manifest_path),
                "sha256": action_binding["manifest_sha256"],
            },
            "catalog": {
                "path": str(catalog_path),
                "sha256": action_binding["catalog_sha256"],
            },
            "solver": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "action_binding": {
            "uid": uid,
            "catalog_clip_id": catalog_row.get("clip_id"),
            "selected_hit_frame_120": hit_frame,
            "unit_json_hit_frames": hit_frames,
            "selected_by": "manifest.hit_frame_pkl_120",
        },
        "qpos_mapping": {
            "source_nq": source_nq,
            "canonical_model_nq": int(model.nq),
            "preserved_prefix": True,
            "canonical_suffix_joints": tail_joint_names,
            "input_order": "gmr_dof_pos",
            "mjcf_qpos_order": "gmr_dof_pos",
            "materialized_runtime_order": "runtime_articulation_joint_pos",
            "source_names": list(joint_order["source_names"]),
            "target_names": list(joint_order["target_names"]),
            "target_from_source_indices": list(
                joint_order["target_from_source_indices"]
            ),
        },
        "teacher": {
            "position": "measured_physical_blade_center",
            "normal": "hits.face_normal_hope_signed_dense_extension",
            "long_axis": "measured_butt_to_blade",
            "velocity": "finite_difference_measured_blade_center_120hz",
            "measured_dense_normal_sign": measured_face_sign,
            "robot_mount_normal_sign": robot_face_sign,
            "robot_mount_normal_sign_source": catalog_face_sign_source,
            "robot_mount_normal_sign_pinned_by_catalog": True,
            "full_orientation_observable_from": [
                "measured_signed_face_normal",
                "measured_butt_to_blade_long_axis",
            ],
            "robot_butt_to_blade_axis_local": (
                ROBOT_BUTT_TO_BLADE_AXIS_LOCAL.tolist()
            ),
            "robot_butt_to_blade_axis_semantics": (
                "official_rigid_visual_component_longitudinal_axis_"
                "positive_handle_butt_to_blade_center_not_site_local_x"
            ),
            "robot_rigid_visual_mesh_sha256": ROBOT_RIGID_VISUAL_MESH_SHA256,
        },
        "optimization": {
            "optimized_joints": list(OPTIMIZED_JOINTS),
            "weights": {
                "position": W_POSITION,
                "face": W_FACE,
                "long_axis": W_LONG_AXIS,
                "point_velocity": W_POINT_VELOCITY,
                "original": W_ORIGINAL.tolist(),
                "neighbor": W_NEIGHBOR,
            },
            "step_rad": STEP_RAD,
            "motion_constraints": {
                "urdf_soft_limit_margin_fraction": float(soft_limit_margin_fraction),
                "urdf_velocity_limit_fraction": float(velocity_limit_fraction),
                "acceleration_proxy_rad_s2": (
                    None
                    if acceleration_proxy_rad_s2 is None
                    else float(acceleration_proxy_rad_s2)
                ),
                "acceleration_proxy_semantics": (
                    "diagnostic_second_difference_smoothness_cap_not_hardware_authority"
                ),
                "torque_speed_authority": "UNKNOWN",
                "mechanical_admission": False,
            },
            "all_success": True,
            "calls": len(optimizer_results),
            "status_counts": {
                str(status): sum(
                    int(result.status) == status for _, _, result in optimizer_results
                )
                for status in sorted({int(result.status) for _, _, result in optimizer_results})
            },
            "max_nit": max(int(getattr(result, "nit", 0)) for _, _, result in optimizer_results),
            "max_nfev": max(int(getattr(result, "nfev", 0)) for _, _, result in optimizer_results),
            "max_optimized_joint_step_rad": float(np.max(joint_step)),
            "min_optimized_joint_soft_position_margin_rad": float(
                np.min(soft_position_margin)
            ),
            "max_optimized_joint_velocity_rad_s": float(
                np.max(np.abs(joint_velocity))
            ),
            "max_optimized_joint_velocity_to_urdf_limit_ratio": float(
                np.max(joint_velocity_ratio)
            ),
            "max_optimized_joint_acceleration_proxy_rad_s2": float(
                np.max(np.abs(joint_acceleration))
            ),
            "all_source_joint_diagnostics": {
                "joint_count": len(joint_order["source_names"]),
                "min_soft_position_margin_rad": float(
                    all_joint_soft_position_margin[
                        all_position_worst_frame, all_position_worst_joint
                    ]
                ),
                "min_soft_position_margin_joint": joint_order["source_names"][
                    all_position_worst_joint
                ],
                "min_soft_position_margin_frame": int(all_position_worst_frame),
                "max_velocity_to_urdf_limit_ratio": float(
                    all_joint_velocity_ratio[
                        all_velocity_worst_frame, all_velocity_worst_joint
                    ]
                ),
                "max_velocity_ratio_joint": joint_order["source_names"][
                    all_velocity_worst_joint
                ],
                "max_velocity_ratio_frame": int(all_velocity_worst_frame),
                "max_acceleration_proxy_rad_s2": float(
                    abs(
                        all_joint_acceleration[
                            all_acceleration_worst_frame, all_acceleration_worst_joint
                        ]
                    )
                ),
                "max_acceleration_proxy_joint": joint_order["source_names"][
                    all_acceleration_worst_joint
                ],
                "max_acceleration_proxy_frame": int(all_acceleration_worst_frame + 1),
            },
        },
        "residual": {
            "position_m": _percentiles(pos_error),
            "face_deg": _percentiles(face_error),
            "long_axis_deg": _percentiles(long_error),
            "so3_deg": _percentiles(so3_error),
            "velocity_direction_deg": _percentiles(velocity_direction_error),
            "velocity_relative": _percentiles(velocity_relative_error),
            "hit": {
                "frame": hit_frame,
                "position_m": float(pos_error[hit_frame]),
                "face_deg": float(face_error[hit_frame]),
                "long_axis_deg": float(long_error[hit_frame]),
                "so3_deg": float(so3_error[hit_frame]),
                "velocity_direction_deg": float(velocity_direction_error[hit_frame]),
                "velocity_direction_observable": bool(velocity_direction_valid[hit_frame]),
                "velocity_relative": float(velocity_relative_error[hit_frame]),
            },
            "segments": _segment_residuals(
                hit_frame=hit_frame,
                position=pos_error,
                face=face_error,
                long_axis=long_error,
                so3=so3_error,
            ),
        },
        "gates": gates,
        "admitted": admitted,
        "retarget_admission_semantics": (
            "kinematic_plus_configured_diagnostic_motion_constraints_not_mechanical"
        ),
        "mechanical_admission": False,
        "authorization": {
            "training": False,
            "promotion": False,
            "deployment": False,
            "diagnostic_unauthorized": True,
            "mechanical_admission": False,
            "mechanical_unknown_reasons": [
                "authoritative_joint_acceleration_limits_unavailable",
                "authoritative_torque_speed_curves_unavailable",
                "per_frame_inverse_dynamics_joint_torque_unavailable",
            ],
        },
    }
    report_payload = _json_bytes(report)
    if admitted:
        output = dict(payload)
        # Preserve the source artifact's 38-D robot-only schema.  The appended MJCF ball qpos is
        # a kinematics scratch suffix and must never leak into the motion-bank format.
        output["qpos"] = solved[:, :source_nq]
        output["wrist_mode"] = "canonical_right_racket_full_phase_v4"
        output["mount_normal_sign"] = float(robot_face_sign)
        output["measured_racket_site_pos_w_120"] = blade.astype(np.float32)
        output["measured_racket_normal_w_120"] = face_target.astype(np.float32)
        output["measured_racket_long_axis_w_120"] = long_target.astype(np.float32)
        output["measured_racket_robot_butt_to_blade_axis_local"] = (
            ROBOT_BUTT_TO_BLADE_AXIS_LOCAL.copy()
        )
        output["measured_racket_robot_rigid_visual_mesh_sha256"] = (
            ROBOT_RIGID_VISUAL_MESH_SHA256
        )
        output["measured_racket_retarget_admitted"] = True
        output["measured_racket_retarget_admission_semantics"] = (
            "kinematic_plus_configured_diagnostic_motion_constraints_not_mechanical"
        )
        output["measured_racket_mechanical_admission"] = False
        output["diagnostic_unauthorized"] = True
        output["measured_racket_input_pkl_sha256"] = report["sources"]["input_pkl"]["sha256"]
        output["measured_racket_source_sha256"] = report["sources"]["unit_npz"]["sha256"]
        output["measured_racket_retarget_receipt_sha256"] = hashlib.sha256(
            report_payload
        ).hexdigest()
        output["measured_racket_uid"] = uid
        output["measured_racket_manifest_sha256"] = action_binding["manifest_sha256"]
        output["measured_racket_catalog_sha256"] = action_binding["catalog_sha256"]
        output["measured_racket_selected_hit_frame_120"] = hit_frame
        output["joint_order_contract_sha256"] = joint_order["sha256"]
        output["joint_order_contract_id"] = joint_order["contract_id"]
        # The report is the per-action completion marker, so publish the PKL first and the
        # content-bound report last.  Bank-level consumers still require a 73-row completion
        # manifest before promotion.
        _atomic_bytes_no_replace(
            output_path, pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
        )
    _atomic_bytes_no_replace(report_path, report_payload)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--unit-npz", type=Path, required=True)
    parser.add_argument("--unit-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--joint-order-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--soft-limit-margin-fraction",
        type=float,
        default=DEFAULT_SOFT_LIMIT_MARGIN_FRACTION,
        help="symmetric diagnostic margin removed from each URDF joint range",
    )
    parser.add_argument(
        "--velocity-limit-fraction",
        type=float,
        default=DEFAULT_VELOCITY_LIMIT_FRACTION,
        help="fraction of each URDF no-load velocity enforced between 120 Hz frames",
    )
    parser.add_argument(
        "--acceleration-proxy-rad-s2",
        type=float,
        default=DEFAULT_ACCELERATION_PROXY_RAD_S2,
        help="diagnostic second-difference cap; not a hardware acceleration authority",
    )
    args = parser.parse_args()
    report = solve_one(
        uid=args.uid,
        pkl_path=args.pkl.resolve(),
        unit_npz_path=args.unit_npz.resolve(),
        unit_json_path=args.unit_json.resolve(),
        manifest_path=args.manifest.resolve(),
        catalog_path=args.catalog.resolve(),
        model_path=args.xml.resolve(),
        urdf_path=args.urdf.resolve(),
        joint_order_contract_path=args.joint_order_contract.resolve(),
        output_path=args.output.resolve(),
        report_path=args.report.resolve(),
        soft_limit_margin_fraction=args.soft_limit_margin_fraction,
        velocity_limit_fraction=args.velocity_limit_fraction,
        acceleration_proxy_rad_s2=args.acceleration_proxy_rad_s2,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
