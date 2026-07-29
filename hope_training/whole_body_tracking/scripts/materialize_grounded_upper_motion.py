#!/usr/bin/env python3
"""Materialize the narrow grounded-upper q/qd consistency repair.

This tool exists for one specific fast path.  An AgiBot A3 canonical ``upper``
motion may already contain the exact, stationary grounded-ready candidate
(``candidate_id=G1``) leg *positions* in every frame while still carrying
stale, non-zero leg ``joint_vel`` rows inherited
from an earlier compiler trajectory.  Such an archive is internally
inconsistent: its lower-body path is constant but its lower-body velocity is
not.

The materializer does **not** transplant a new strike, retime a motion, change
the root, or change any joint position.  It:

1. pins the input motion, legacy canonical-ready sidecar, published AgiBot A3
   grounded-ready candidate (``candidate_id=G1``), exact A3 MJCF, and runtime
   body order by SHA-256;
2. proves that the input root and all twelve leg positions are constant;
3. directly audits the input motion's own frame-0 pose on the exact A3 model
   (double-foot contact, no unsupported/self collision, and bounded
   penetration), records but does not gate on a zero-velocity static-contact
   LP that is not a dynamic-strike feasibility test, and requires its constant leg
   positions to quantize bitwise to the published grounded-ready candidate
   (``candidate_id=G1``);
4. changes only the twelve leg ``joint_vel`` columns to exact zero;
5. rebuilds every schema-2 body FK/velocity channel with
   ``canonical_schema2_builder``;
6. replays the input and output through the same exact MuJoCo model and
   requires every frame of ``right_racket`` site position, orientation,
   linear velocity, and angular velocity to be bitwise identical;
7. publishes a no-clobber diagnostic bundle and writes ``RECEIPT.json`` last.

The output is deliberately diagnostic.  Every receipt says
``training_authorized=false``, ``deployment_authorized=false``, and
``hardware_authorized=false``.  This tool is not a replacement for compiler,
behavior, table-contact, policy, or deployment gates.

Typical Pod invocation (all hashes are mandatory)::

    python scripts/materialize_grounded_upper_motion.py \
      --input-motion motions/fivebind_n5_20260728/bh_loop_c_upper_fivebind.npz \
      --expected-input-sha256 <sha> \
      --canonical-ready-v1 vendor_assets/.../canonical_ready_v1.npz \
      --expected-canonical-ready-v1-sha256 <sha> \
      --grounded-reference-candidate vendor_assets/.../grounded_ready_candidate_v1.npz \
      --expected-grounded-reference-candidate-sha256 <sha> \
      --grounded-reference-receipt vendor_assets/.../RECEIPT.json \
      --expected-grounded-reference-receipt-sha256 <sha> \
      --body-order configs/a3_runtime_body_order.txt \
      --expected-body-order-sha256 <sha> \
      --strike-frame 31 \
      --output-dir /workspace/.../bh_loop_c_grounded_upper_qvel_fix_v1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import canonical_grounded_ready as grounded  # noqa: E402
import canonical_schema2_builder as schema2  # noqa: E402


TOOL_ID = "grounded_upper_qvel_consistency_materializer_v1"
ARTIFACT_CLASS = "diagnostic_grounded_upper_qvel_consistency_repair"
RECEIPT_FILENAME = "RECEIPT.json"
SCHEMA2_MANIFEST_FILENAME = "SCHEMA2_MANIFEST.json"
SCHEMA2_REPORT_FILENAME = "SCHEMA2_REPORT.json"
SITE_NAME = "right_racket"
A3_XML_MODEL_NAME = "A3T2.5_pingpong_0519"

LEG_JOINT_NAMES = tuple(grounded.LEG_JOINT_NAMES)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
RUNTIME_JOINT_NAMES = tuple(schema2.RUNTIME_JOINT_NAMES)
LEG_JOINT_INDICES = tuple(
    RUNTIME_JOINT_NAMES.index(name) for name in LEG_JOINT_NAMES
)
RIGHT_ARM_JOINT_INDICES = tuple(
    RUNTIME_JOINT_NAMES.index(name) for name in RIGHT_ARM_JOINT_NAMES
)
STABLE_READY_LINEAGE_INDICES = tuple(
    index
    for index in range(len(RUNTIME_JOINT_NAMES))
    if index not in set(LEG_JOINT_INDICES)
    and index not in set(RIGHT_ARM_JOINT_INDICES)
)

_CANONICAL_READY_V1_KEYS = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "source_segment",
        "source_npz",
        "source_frame",
        "striking_joint_ids",
        "note",
    }
)
_GROUNDED_REFERENCE_KEYS = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "candidate_id",
        "receipt_sha256",
        "training_authorized",
        "hardware_authorized",
    }
)

if len(RUNTIME_JOINT_NAMES) != 31 or len(set(RUNTIME_JOINT_NAMES)) != 31:
    raise RuntimeError("runtime joint contract must contain 31 unique names")
if len(LEG_JOINT_INDICES) != 12 or len(set(LEG_JOINT_INDICES)) != 12:
    raise RuntimeError("leg contract must contain 12 unique runtime indices")


class GroundedUpperMaterializationError(RuntimeError):
    """The narrow repair cannot be applied without weakening its contract."""


@dataclass(frozen=True)
class SiteTrace:
    """Exact-model site state sampled at every source frame."""

    position_w: np.ndarray
    rotation_w: np.ndarray
    linear_velocity_w: np.ndarray
    angular_velocity_w: np.ndarray


@dataclass(frozen=True)
class PublishedBundle:
    """Paths and content hashes of one receipt-last diagnostic bundle."""

    directory: Path
    motion: Path
    schema2_manifest: Path
    schema2_report: Path
    receipt: Path
    motion_sha256: str
    receipt_sha256: str


def _require_digest(value: str, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise GroundedUpperMaterializationError(
            f"{label} must be 64 lowercase SHA-256 hex digits"
        )
    return digest


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GroundedUpperMaterializationError(
            f"cannot read {label} {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _pinned_regular_file(
    path_value: str | Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, str]:
    path_input = Path(path_value).expanduser().absolute()
    try:
        path = path_input.resolve(strict=True)
    except OSError as exc:
        raise GroundedUpperMaterializationError(
            f"cannot resolve {label}: {exc}"
        ) from exc
    if path_input != path or path_input.is_symlink() or not path.is_file():
        raise GroundedUpperMaterializationError(
            f"{label} must be one regular file without symlink components: {path_input}"
        )
    expected = _require_digest(expected_sha256, f"expected {label} SHA-256")
    actual = _sha256_file(path, label)
    if actual != expected:
        raise GroundedUpperMaterializationError(
            f"{label} SHA-256 mismatch: {actual} != {expected}"
        )
    return path, actual


def _copy_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except Exception as exc:
        raise GroundedUpperMaterializationError(
            f"cannot load {label} NPZ {path}: {type(exc).__name__}: {exc}"
        ) from exc
    for key, value in arrays.items():
        if value.dtype.hasobject:
            raise GroundedUpperMaterializationError(
                f"{label}.{key} has forbidden object dtype"
            )
    return arrays


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise GroundedUpperMaterializationError(
            f"{label} must be a real array with shape {shape}, got "
            f"{array.shape}/{array.dtype}"
        )
    if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
        raise GroundedUpperMaterializationError(f"{label} contains NaN/Inf")
    return array


def _scalar_string(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise GroundedUpperMaterializationError(
            f"{label} must be one scalar Unicode/bytes string"
        )
    result = str(array.item())
    if not result:
        raise GroundedUpperMaterializationError(f"{label} must be non-empty")
    return result


def _audit_input_grounded_state(
    backend: grounded.MujocoGroundedReadyBackend,
    state: grounded.ReadyState,
) -> dict[str, Any]:
    """Audit the actual upper-motion pose, not a different neutral-arm donor.

    ``candidate_id=G1`` identifies the published source of the twelve constant
    leg coordinates.  It is not a promise that an upper-body strike pose has
    the same root yaw, waist, head, or arm coordinates as that neutral-arm
    candidate.  Re-solving the strike pose as a new G1 candidate therefore
    answers the wrong question and can reject a physically grounded input.
    """

    cfg = grounded.GroundedReadyConfig()
    q = np.asarray(state.joint_pos, dtype=np.float64)
    lower = np.asarray(backend.position_lower, dtype=np.float64)
    upper = np.asarray(backend.position_upper, dtype=np.float64)
    worst_limit_excess = float(
        max(0.0, np.max(np.maximum(lower - q, q - upper)))
    )
    if worst_limit_excess > cfg.joint_limit_tolerance_rad:
        raise GroundedUpperMaterializationError(
            "input upper pose exceeds exact A3 joint limits"
        )

    scene = backend.static_scene(
        state,
        contact_gap_tolerance_m=cfg.floor_gap_tolerance_m,
        penetration_tolerance_m=cfg.penetration_tolerance_m,
    )
    double_support = bool(
        scene.foot_contact_count[0] > 0 and scene.foot_contact_count[1] > 0
    )
    sole_gap = bool(
        np.max(np.abs(scene.sole_minimum_distance_m))
        <= cfg.floor_gap_tolerance_m
    )
    collision_safe = bool(
        not scene.unsupported_contacts
        and not scene.self_collision_pairs
        and scene.maximum_foot_penetration_m <= cfg.penetration_tolerance_m
    )
    dynamics = dict(
        backend.static_ground_dynamics(
            state,
            contact_gap_tolerance_m=cfg.floor_gap_tolerance_m,
            penetration_tolerance_m=cfg.penetration_tolerance_m,
        )
    )
    if not double_support or not sole_gap or not collision_safe:
        raise GroundedUpperMaterializationError(
            "input upper pose is not a collision-safe exact-A3 double-support pose"
        )
    return {
        "joint_limits_passed": True,
        "worst_joint_limit_excess_rad": worst_limit_excess,
        "double_support_passed": True,
        "foot_contact_count": list(scene.foot_contact_count),
        "sole_gap_passed": True,
        "sole_minimum_distance_m": scene.sole_minimum_distance_m.tolist(),
        "collision_safe": True,
        "maximum_foot_penetration_m": scene.maximum_foot_penetration_m,
        "unsupported_contacts": [],
        "self_collision_pairs": [],
        "static_ground_dynamics_gating": False,
        "static_ground_dynamics_passed": dynamics.get("feasible") is True,
        "static_ground_dynamics": _jsonable(dynamics),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GroundedUpperMaterializationError(
            f"cannot read {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise GroundedUpperMaterializationError(f"{label} JSON must be an object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GroundedUpperMaterializationError(
            f"{label} JSON is not finite/serializable: {exc}"
        ) from exc
    return value


def _quat_to_matrix(value: np.ndarray, label: str) -> np.ndarray:
    quat = _finite(value, (4,), label).astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0 or abs(norm - 1.0) > 1.0e-6:
        raise GroundedUpperMaterializationError(
            f"{label} is not a unit quaternion (norm={norm:.12g})"
        )
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _assert_yaw_only_ready_transform(
    motion_root_quat: np.ndarray,
    ready_root_quat: np.ndarray,
) -> float:
    motion_rotation = _quat_to_matrix(motion_root_quat, "motion root quaternion")
    ready_rotation = _quat_to_matrix(ready_root_quat, "ready-v1 root quaternion")
    relative = motion_rotation @ ready_rotation.T
    expected_structure_error = float(
        max(
            abs(relative[0, 2]),
            abs(relative[1, 2]),
            abs(relative[2, 0]),
            abs(relative[2, 1]),
            abs(relative[2, 2] - 1.0),
        )
    )
    if expected_structure_error > 2.0e-6:
        raise GroundedUpperMaterializationError(
            "motion root differs from canonical-ready v1 by more than a world-Z yaw "
            f"(structure error {expected_structure_error:.3e})"
        )
    return math.atan2(float(relative[1, 0]), float(relative[0, 0]))


def _validate_canonical_ready_v1(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if frozenset(arrays) != _CANONICAL_READY_V1_KEYS:
        raise GroundedUpperMaterializationError(
            "canonical-ready v1 keyset mismatch: "
            f"{sorted(arrays)} != {sorted(_CANONICAL_READY_V1_KEYS)}"
        )
    q = _finite(arrays["joint_pos"], (31,), "ready-v1 joint_pos")
    qd = _finite(arrays["joint_vel"], (31,), "ready-v1 joint_vel")
    root_pos = _finite(arrays["root_pos_w"], (3,), "ready-v1 root_pos_w")
    root_quat = _finite(arrays["root_quat_w"], (4,), "ready-v1 root_quat_w")
    _quat_to_matrix(root_quat, "ready-v1 root_quat_w")
    if np.count_nonzero(qd) != 0:
        raise GroundedUpperMaterializationError(
            "canonical-ready v1 joint_vel must be exact zero"
        )
    source_frame = np.asarray(arrays["source_frame"])
    if source_frame.shape != () or source_frame.dtype.kind not in "iu":
        raise GroundedUpperMaterializationError(
            "canonical-ready v1 source_frame must be one integer scalar"
        )
    if int(source_frame) != 0:
        raise GroundedUpperMaterializationError(
            "canonical-ready v1 source_frame must equal zero"
        )
    striking = np.asarray(arrays["striking_joint_ids"])
    if (
        striking.shape != (7,)
        or striking.dtype.kind not in "iu"
        or len(set(map(int, striking.tolist()))) != 7
        or np.any((striking < 0) | (striking >= 31))
    ):
        raise GroundedUpperMaterializationError(
            "canonical-ready v1 striking_joint_ids must be seven unique runtime indices"
        )
    return {
        "joint_pos": q,
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "source_segment": _scalar_string(
            arrays["source_segment"], "ready-v1 source_segment"
        ),
        "source_npz": _scalar_string(arrays["source_npz"], "ready-v1 source_npz"),
        "note": _scalar_string(arrays["note"], "ready-v1 note"),
        "source_frame": int(source_frame),
        "striking_joint_ids": [int(value) for value in striking.tolist()],
    }


def _validate_motion(
    arrays: Mapping[str, np.ndarray],
    *,
    ready_v1: Mapping[str, Any],
    strike_frame: int,
) -> dict[str, Any]:
    if frozenset(arrays) not in schema2.ALLOWED_KEYSETS:
        raise GroundedUpperMaterializationError(
            f"input motion must have exact schema-2 11/14 keys, got {sorted(arrays)}"
        )
    q_raw = np.asarray(arrays["joint_pos"])
    if q_raw.ndim != 2 or q_raw.shape[1] != 31 or q_raw.shape[0] < 2:
        raise GroundedUpperMaterializationError(
            f"joint_pos must have shape (T>=2,31), got {q_raw.shape}"
        )
    frames = int(q_raw.shape[0])
    q = _finite(q_raw, (frames, 31), "motion joint_pos")
    qd = _finite(arrays["joint_vel"], (frames, 31), "motion joint_vel")
    body_pos = _finite(arrays["body_pos_w"], (frames, 32, 3), "motion body_pos_w")
    body_quat = _finite(
        arrays["body_quat_w"], (frames, 32, 4), "motion body_quat_w"
    )
    body_lin = _finite(
        arrays["body_lin_vel_w"], (frames, 32, 3), "motion body_lin_vel_w"
    )
    body_ang = _finite(
        arrays["body_ang_vel_w"], (frames, 32, 3), "motion body_ang_vel_w"
    )
    body_names = tuple(str(value) for value in np.asarray(arrays["body_names"]).tolist())
    if len(body_names) != 32 or len(set(body_names)) != 32:
        raise GroundedUpperMaterializationError(
            "motion body_names must contain 32 unique names"
        )
    fps_array = np.asarray(arrays["fps"])
    if fps_array.size != 1:
        raise GroundedUpperMaterializationError("motion fps must contain one scalar")
    fps = float(fps_array.reshape(-1)[0])
    if not math.isfinite(fps) or fps <= 0.0:
        raise GroundedUpperMaterializationError("motion fps must be finite and positive")
    if isinstance(strike_frame, bool) or not isinstance(strike_frame, int):
        raise GroundedUpperMaterializationError("strike_frame must be an integer")
    if strike_frame < 0 or strike_frame >= frames:
        raise GroundedUpperMaterializationError(
            f"strike_frame {strike_frame} lies outside [0,{frames - 1}]"
        )
    schema_version = np.asarray(arrays["kinematics_schema_version"])
    if schema_version.size != 1 or int(schema_version.reshape(-1)[0]) != 2:
        raise GroundedUpperMaterializationError(
            "motion kinematics_schema_version must equal 2"
        )
    if str(np.asarray(arrays["body_pos_point"]).item()) != "link_origin":
        raise GroundedUpperMaterializationError(
            "motion body_pos_point must equal link_origin"
        )
    if str(np.asarray(arrays["body_lin_vel_point"]).item()) != "center_of_mass":
        raise GroundedUpperMaterializationError(
            "motion body_lin_vel_point must equal center_of_mass"
        )
    if not np.array_equal(q[0], q[-1]):
        raise GroundedUpperMaterializationError(
            "upper motion first/last joint_pos must be bitwise identical"
        )
    if not np.array_equal(body_pos[0, 0], body_pos[-1, 0]) or not np.array_equal(
        body_quat[0, 0], body_quat[-1, 0]
    ):
        raise GroundedUpperMaterializationError(
            "upper motion first/last root pose must be bitwise identical"
        )
    if np.count_nonzero(qd[[0, -1]]) != 0:
        raise GroundedUpperMaterializationError(
            "upper motion joint_vel endpoints must be exact zero"
        )
    for label, velocity in (
        ("root COM linear velocity", body_lin[:, 0]),
        ("root angular velocity", body_ang[:, 0]),
    ):
        if np.count_nonzero(velocity) != 0:
            raise GroundedUpperMaterializationError(
                f"upper motion {label} must be exact zero at every frame"
            )
    if not np.array_equal(
        body_pos[:, 0], np.broadcast_to(body_pos[0, 0], body_pos[:, 0].shape)
    ) or not np.array_equal(
        body_quat[:, 0], np.broadcast_to(body_quat[0, 0], body_quat[:, 0].shape)
    ):
        raise GroundedUpperMaterializationError(
            "upper motion root pose must be bitwise constant at every frame"
        )
    leg = np.asarray(LEG_JOINT_INDICES, dtype=np.int64)
    if not np.array_equal(
        q[:, leg], np.broadcast_to(q[0, leg], q[:, leg].shape)
    ):
        raise GroundedUpperMaterializationError(
            "fast repair only accepts an already-constant 12-leg qpos path"
        )
    stable = np.asarray(STABLE_READY_LINEAGE_INDICES, dtype=np.int64)
    ready_q = np.asarray(ready_v1["joint_pos"])
    if not np.array_equal(q[0, stable], ready_q[stable].astype(q.dtype)):
        raise GroundedUpperMaterializationError(
            "motion ready endpoint differs from canonical-ready v1 outside the "
            "12 legs and seven right-arm neutral overlay coordinates"
        )
    ready_root = np.asarray(ready_v1["root_pos_w"]).astype(body_pos.dtype)
    if not np.array_equal(body_pos[0, 0], ready_root):
        raise GroundedUpperMaterializationError(
            "motion root position differs from canonical-ready v1"
        )
    yaw_rad = _assert_yaw_only_ready_transform(
        body_quat[0, 0], np.asarray(ready_v1["root_quat_w"])
    )
    pre_qd = np.asarray(qd[:, leg], dtype=np.float64)
    per_joint = []
    for column, (index, name) in enumerate(zip(LEG_JOINT_INDICES, LEG_JOINT_NAMES)):
        values = pre_qd[:, column]
        per_joint.append(
            {
                "index": int(index),
                "name": name,
                "nonzero_samples": int(np.count_nonzero(values)),
                "maximum_abs_radps": float(np.max(np.abs(values))),
            }
        )
    if sum(row["nonzero_samples"] for row in per_joint) == 0:
        raise GroundedUpperMaterializationError(
            "input leg joint_vel is already exact zero; refusing a no-op repair"
        )
    return {
        "frames": frames,
        "fps": fps,
        "strike_frame": strike_frame,
        "joint_pos": q,
        "joint_vel": qd,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        "body_names": body_names,
        "ready_yaw_rotation_rad": yaw_rad,
        "leg_velocity_before": per_joint,
    }


def _validate_grounded_reference(
    arrays: Mapping[str, np.ndarray],
    receipt: Mapping[str, Any],
    *,
    candidate_sha256: str,
) -> grounded.ExactModelIdentity:
    if frozenset(arrays) != _GROUNDED_REFERENCE_KEYS:
        raise GroundedUpperMaterializationError(
            "grounded reference candidate keyset mismatch: "
            f"{sorted(arrays)} != {sorted(_GROUNDED_REFERENCE_KEYS)}"
        )
    q = _finite(arrays["joint_pos"], (31,), "grounded reference joint_pos")
    _finite(arrays["joint_vel"], (31,), "grounded reference joint_vel")
    _finite(arrays["root_pos_w"], (3,), "grounded reference root_pos_w")
    _finite(arrays["root_quat_w"], (4,), "grounded reference root_quat_w")
    _finite(arrays["root_lin_vel_w"], (3,), "grounded reference root_lin_vel_w")
    _finite(arrays["root_ang_vel_w"], (3,), "grounded reference root_ang_vel_w")
    for key in (
        "joint_vel",
        "root_lin_vel_w",
        "root_ang_vel_w",
    ):
        if np.count_nonzero(arrays[key]) != 0:
            raise GroundedUpperMaterializationError(
                f"grounded reference {key} must be exact zero"
            )
    if bool(np.asarray(arrays["training_authorized"]).item()) or bool(
        np.asarray(arrays["hardware_authorized"]).item()
    ):
        raise GroundedUpperMaterializationError(
            "grounded reference candidate must remain unauthorized"
        )
    publication = receipt.get("publication")
    authorization = receipt.get("authorization")
    source = receipt.get("source")
    static = receipt.get("static_geometry")
    dynamics = receipt.get("static_ground_dynamics")
    gates = receipt.get("gates")
    exact = receipt.get("exact_model")
    candidate = receipt.get("candidate")
    for label, section in (
        ("publication", publication),
        ("authorization", authorization),
        ("source", source),
        ("static_geometry", static),
        ("static_ground_dynamics", dynamics),
        ("gates", gates),
        ("exact_model", exact),
        ("candidate", candidate),
    ):
        if not isinstance(section, Mapping):
            raise GroundedUpperMaterializationError(
                f"grounded reference receipt lacks mapping {label}"
            )
    if publication.get("candidate_npz_sha256") != candidate_sha256:
        raise GroundedUpperMaterializationError(
            "grounded reference receipt does not bind candidate bytes"
        )
    if (
        _scalar_string(arrays["candidate_id"], "grounded reference candidate_id")
        != "G1"
        or receipt.get("candidate_id") != "G1"
    ):
        raise GroundedUpperMaterializationError(
            "grounded reference must be the AgiBot A3 candidate_id=G1 construction"
        )
    receipt_payload_sha = _require_digest(
        str(receipt.get("receipt_payload_sha256")),
        "grounded reference receipt payload SHA-256",
    )
    if (
        _scalar_string(
            arrays["receipt_sha256"],
            "grounded reference embedded receipt SHA-256",
        )
        != receipt_payload_sha
    ):
        raise GroundedUpperMaterializationError(
            "grounded reference candidate does not bind its construction receipt"
        )
    publication_payload_sha = _require_digest(
        str(receipt.get("publication_payload_sha256")),
        "grounded reference publication payload SHA-256",
    )
    publication_unsigned = dict(receipt)
    publication_unsigned.pop("publication_payload_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(publication_unsigned)) != publication_payload_sha:
        raise GroundedUpperMaterializationError(
            "grounded reference publication payload seal does not verify"
        )
    if (
        authorization.get("training_authorized") is not False
        or authorization.get("deployment_authorized") is not False
        or authorization.get("hardware_authorized") is not False
    ):
        raise GroundedUpperMaterializationError(
            "grounded reference receipt must deny all authorization"
        )
    if (
        static.get("passed") is not True
        or dynamics.get("feasible") is not True
        or exact.get("exact_mujoco_backend") is not True
        or receipt.get("verdict") != "PASS_STATIC_GROUNDED_READY_CANDIDATE"
    ):
        raise GroundedUpperMaterializationError(
            "grounded reference receipt lacks exact static G1 PASS evidence"
        )
    if any(value != "PASS" for value in gates.values()):
        raise GroundedUpperMaterializationError(
            f"grounded reference contains a non-PASS gate: {dict(gates)}"
        )
    if source.get("mode") != "G1_donor_root_flat_feet_leg12_continuation":
        raise GroundedUpperMaterializationError(
            "grounded reference must be the A3 donor-root solve tagged candidate_id=G1"
        )
    if source.get("root_bitwise_preserved") is not True:
        raise GroundedUpperMaterializationError(
            "grounded reference does not prove root preservation"
        )
    embedded_q = np.asarray(candidate.get("joint_pos"), dtype=np.float64)
    if embedded_q.shape != (31,) or not np.array_equal(embedded_q, q.astype(np.float64)):
        raise GroundedUpperMaterializationError(
            "grounded reference candidate joint_pos differs from receipt"
        )
    if exact.get("xml_model_name") != A3_XML_MODEL_NAME:
        raise GroundedUpperMaterializationError(
            "grounded reference is not pinned to the accepted AgiBot A3 exact model"
        )
    return grounded.ExactModelIdentity(
        mjcf_path=str(exact["mjcf_path"]),
        mjcf_sha256=_require_digest(str(exact["mjcf_sha256"]), "exact MJCF SHA-256"),
        compiled_model_sha256=_require_digest(
            str(exact["compiled_model_sha256"]), "compiled-model SHA-256"
        ),
        path_model_binding_sha256=_require_digest(
            str(exact["path_model_binding_sha256"]),
            "path-model binding SHA-256",
        ),
        ground_model_binding_sha256=_require_digest(
            str(exact["ground_model_binding_sha256"]),
            "ground-model binding SHA-256",
        ),
        xml_model_name=A3_XML_MODEL_NAME,
    )


def _migration_provenance(
    arrays: Mapping[str, np.ndarray],
) -> schema2.MigrationProvenance | None:
    if frozenset(arrays) == schema2.ALLOWED_KEYSETS[0]:
        return None
    if frozenset(arrays) != schema2.ALLOWED_KEYSETS[1]:
        raise GroundedUpperMaterializationError(
            "input motion is not an exact 11/14-field schema-2 archive"
        )
    return schema2.MigrationProvenance(
        source_sha256=str(np.asarray(arrays["kinematics_migration_source_sha256"]).item()),
        source_point=str(np.asarray(arrays["kinematics_migration_source_point"]).item()),
        tool=str(np.asarray(arrays["kinematics_migration_tool"]).item()),
    )


def _repair_constant_grounded_leg_velocity(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    solved_grounded_joint_pos: np.ndarray,
    published_grounded_joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a q-identical path with only the twelve leg qd columns zeroed."""

    q = np.asarray(joint_pos)
    qd = np.asarray(joint_vel)
    if q.ndim != 2 or q.shape[1] != 31 or qd.shape != q.shape:
        raise GroundedUpperMaterializationError(
            "joint_pos/joint_vel must have matching (T,31) shapes"
        )
    solved = _finite(
        solved_grounded_joint_pos,
        (31,),
        "current-root solved grounded joint_pos",
    )
    published = _finite(
        published_grounded_joint_pos,
        (31,),
        "published grounded joint_pos",
    )
    leg = np.asarray(LEG_JOINT_INDICES, dtype=np.int64)
    input_leg = np.asarray(q[0, leg], dtype=np.float32)
    if not np.array_equal(np.asarray(solved[leg], dtype=np.float32), input_leg):
        raise GroundedUpperMaterializationError(
            "input leg qpos is not already the exact current-root A3 grounded "
            "solution tagged candidate_id=G1 at "
            "published float32 precision; this narrow qvel-only repair refuses to "
            "change qpos"
        )
    if not np.array_equal(np.asarray(published[leg], dtype=np.float32), input_leg):
        raise GroundedUpperMaterializationError(
            "input leg qpos differs from the pinned published A3 grounded "
            "reference tagged candidate_id=G1"
        )
    q_out = np.array(q, copy=True)
    qd_out = np.array(qd, copy=True)
    qd_out[:, leg] = np.asarray(0, dtype=qd_out.dtype)
    nonleg = np.asarray(
        [index for index in range(31) if index not in set(LEG_JOINT_INDICES)],
        dtype=np.int64,
    )
    if not np.array_equal(q_out, q):
        raise AssertionError("qvel repair changed joint_pos")
    if not np.array_equal(qd_out[:, nonleg], qd[:, nonleg]):
        raise AssertionError("qvel repair changed a non-leg joint velocity")
    if np.count_nonzero(qd_out[:, leg]) != 0:
        raise AssertionError("qvel repair did not zero every leg velocity")
    return q_out, qd_out


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GroundedUpperMaterializationError(
                "receipt cannot contain NaN/Inf"
            )
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise GroundedUpperMaterializationError(
        f"receipt contains unsupported type {type(value).__name__}"
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _hash_array(label: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(label.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _model_site_trace(
    *,
    mujoco: Any,
    model: Any,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
) -> SiteTrace:
    frames = int(joint_pos.shape[0])
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    free_ids = [
        index
        for index in range(int(model.njnt))
        if int(model.jnt_type[index]) == free_type
    ]
    if len(free_ids) != 1:
        raise GroundedUpperMaterializationError(
            f"exact model must contain one free root, got {len(free_ids)}"
        )
    root_joint = int(free_ids[0])
    root_qpos = int(model.jnt_qposadr[root_joint])
    joint_qpos: list[int] = []
    joint_dof: list[int] = []
    for name in RUNTIME_JOINT_NAMES:
        joint_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        )
        if joint_id < 0:
            raise GroundedUpperMaterializationError(
                f"exact model lacks runtime joint {name!r}"
            )
        joint_qpos.append(int(model.jnt_qposadr[joint_id]))
        joint_dof.append(int(model.jnt_dofadr[joint_id]))
    site_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    )
    if site_id < 0:
        raise GroundedUpperMaterializationError(
            f"exact model lacks site {SITE_NAME!r}"
        )
    data = mujoco.MjData(model)
    position = np.empty((frames, 3), dtype=np.float64)
    rotation = np.empty((frames, 9), dtype=np.float64)
    linear = np.empty((frames, 3), dtype=np.float64)
    angular = np.empty((frames, 3), dtype=np.float64)
    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
    qpos_indices = np.asarray(joint_qpos, dtype=np.int64)
    dof_indices = np.asarray(joint_dof, dtype=np.int64)
    for frame in range(frames):
        data.qpos[:] = np.asarray(model.qpos0, dtype=np.float64)
        data.qpos[root_qpos : root_qpos + 3] = root_pos_w[frame]
        data.qpos[root_qpos + 3 : root_qpos + 7] = root_quat_wxyz[frame]
        data.qpos[qpos_indices] = joint_pos[frame]
        data.qvel[:] = 0.0
        data.qvel[dof_indices] = joint_vel[frame]
        mujoco.mj_forward(model, data)
        jacp[:] = 0.0
        jacr[:] = 0.0
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        position[frame] = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        rotation[frame] = np.asarray(
            data.site_xmat[site_id], dtype=np.float64
        ).reshape(9)
        linear[frame] = jacp @ np.asarray(data.qvel, dtype=np.float64)
        angular[frame] = jacr @ np.asarray(data.qvel, dtype=np.float64)
    return SiteTrace(position, rotation, linear, angular)


def _assert_site_trace_bitwise_equal(
    before: SiteTrace,
    after: SiteTrace,
) -> dict[str, Any]:
    rows = (
        ("position_w", before.position_w, after.position_w, "m"),
        ("rotation_w", before.rotation_w, after.rotation_w, "matrix"),
        (
            "linear_velocity_w",
            before.linear_velocity_w,
            after.linear_velocity_w,
            "m/s",
        ),
        (
            "angular_velocity_w",
            before.angular_velocity_w,
            after.angular_velocity_w,
            "rad/s",
        ),
    )
    report: dict[str, Any] = {}
    for label, left, right, unit in rows:
        if left.shape != right.shape:
            raise GroundedUpperMaterializationError(
                f"{SITE_NAME} {label} shape changed: {left.shape} != {right.shape}"
            )
        equal = bool(np.array_equal(left, right))
        maximum = float(np.max(np.abs(left - right))) if left.size else 0.0
        if not equal:
            flat = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
            first = int(flat[0]) if len(flat) else -1
            raise GroundedUpperMaterializationError(
                f"{SITE_NAME} {label} is not bitwise preserved "
                f"(max abs {maximum:.3e}, first flat index {first})"
            )
        report[label] = {
            "bitwise_equal": True,
            "maximum_abs_delta": 0.0,
            "unit": unit,
            "before_sha256": _hash_array(f"{SITE_NAME}.{label}", left),
            "after_sha256": _hash_array(f"{SITE_NAME}.{label}", right),
            "shape": list(left.shape),
            "dtype": str(left.dtype),
        }
    return report


def _exclusive_write_at(directory_fd: int, filename: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags, mode=0o644, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("exclusive write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bundle(
    *,
    output_directory: Path,
    motion_filename: str,
    motion_payload: bytes,
    schema_manifest_payload: bytes,
    schema_report_payload: bytes,
    receipt: Mapping[str, Any],
) -> PublishedBundle:
    authorization = receipt.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(key) is not False
        for key in (
            "training_authorized",
            "deployment_authorized",
            "hardware_authorized",
        )
    ):
        raise GroundedUpperMaterializationError(
            "publication receipt must explicitly deny all authorization"
        )
    output = Path(output_directory).expanduser().absolute()
    parent_input = output.parent
    try:
        parent = parent_input.resolve(strict=True)
    except OSError as exc:
        raise GroundedUpperMaterializationError(
            f"cannot resolve output parent: {exc}"
        ) from exc
    if parent_input != parent or not parent.is_dir():
        raise GroundedUpperMaterializationError(
            "output parent may not contain symlink components"
        )
    if not output.name or output.name in (".", ".."):
        raise GroundedUpperMaterializationError(
            "output directory requires one concrete leaf name"
        )
    if Path(motion_filename).name != motion_filename or not motion_filename.endswith(
        ".npz"
    ):
        raise GroundedUpperMaterializationError("motion filename must be one NPZ leaf")

    unsigned = _jsonable(receipt)
    unsigned.pop("receipt_payload_sha256", None)
    seal = _sha256_bytes(_canonical_json_bytes(unsigned))
    sealed = dict(unsigned)
    sealed["receipt_payload_sha256"] = seal
    receipt_payload = _pretty_json_bytes(sealed)

    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, parent_flags)
    output_fd = -1
    created = False
    filenames = (
        motion_filename,
        SCHEMA2_MANIFEST_FILENAME,
        SCHEMA2_REPORT_FILENAME,
        RECEIPT_FILENAME,
    )
    try:
        try:
            os.mkdir(output.name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite grounded-upper bundle: {output}"
            ) from None
        output_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        output_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(output.name, output_flags, dir_fd=parent_fd)
        _exclusive_write_at(output_fd, motion_filename, motion_payload)
        _exclusive_write_at(
            output_fd, SCHEMA2_MANIFEST_FILENAME, schema_manifest_payload
        )
        _exclusive_write_at(output_fd, SCHEMA2_REPORT_FILENAME, schema_report_payload)
        # Completion marker: always last.
        _exclusive_write_at(output_fd, RECEIPT_FILENAME, receipt_payload)
        os.fsync(output_fd)
        os.fsync(parent_fd)
        entry = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode) or not os.path.samestat(
            entry, os.fstat(output_fd)
        ):
            raise GroundedUpperMaterializationError(
                "published directory identity changed during write"
            )
        published = parent / output.name
        return PublishedBundle(
            directory=published,
            motion=published / motion_filename,
            schema2_manifest=published / SCHEMA2_MANIFEST_FILENAME,
            schema2_report=published / SCHEMA2_REPORT_FILENAME,
            receipt=published / RECEIPT_FILENAME,
            motion_sha256=_sha256_bytes(motion_payload),
            receipt_sha256=_sha256_bytes(receipt_payload),
        )
    except Exception:
        if output_fd >= 0:
            for filename in reversed(filenames):
                try:
                    os.unlink(filename, dir_fd=output_fd)
                except OSError:
                    pass
        if created:
            try:
                os.rmdir(output.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        os.close(parent_fd)


def _body_order_names(path: Path) -> tuple[str, ...]:
    try:
        names = tuple(
            row.strip()
            for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip()
        )
    except OSError as exc:
        raise GroundedUpperMaterializationError(
            f"cannot read body-order {path}: {exc}"
        ) from exc
    if len(names) != 32 or len(set(names)) != 32:
        raise GroundedUpperMaterializationError(
            "body-order must contain 32 unique non-empty names"
        )
    return names


def materialize(args: argparse.Namespace) -> PublishedBundle:
    input_path, input_sha = _pinned_regular_file(
        args.input_motion,
        args.expected_input_sha256,
        "input motion",
    )
    ready_path, ready_sha = _pinned_regular_file(
        args.canonical_ready_v1,
        args.expected_canonical_ready_v1_sha256,
        "canonical-ready v1",
    )
    reference_candidate_path, reference_candidate_sha = _pinned_regular_file(
        args.grounded_reference_candidate,
        args.expected_grounded_reference_candidate_sha256,
        "grounded reference candidate",
    )
    reference_receipt_path, reference_receipt_sha = _pinned_regular_file(
        args.grounded_reference_receipt,
        args.expected_grounded_reference_receipt_sha256,
        "grounded reference receipt",
    )
    body_order_path, body_order_sha = _pinned_regular_file(
        args.body_order,
        args.expected_body_order_sha256,
        "runtime body-order",
    )

    input_arrays = _copy_npz(input_path, "input motion")
    ready_arrays = _copy_npz(ready_path, "canonical-ready v1")
    reference_arrays = _copy_npz(
        reference_candidate_path, "grounded reference candidate"
    )
    reference_receipt = _read_json(
        reference_receipt_path, "grounded reference receipt"
    )
    ready = _validate_canonical_ready_v1(ready_arrays)
    motion = _validate_motion(
        input_arrays,
        ready_v1=ready,
        strike_frame=int(args.strike_frame),
    )
    exact_identity = _validate_grounded_reference(
        reference_arrays,
        reference_receipt,
        candidate_sha256=reference_candidate_sha,
    )
    if tuple(motion["body_names"]) != _body_order_names(body_order_path):
        raise GroundedUpperMaterializationError(
            "input motion body_names differs from explicitly pinned body-order"
        )

    backend = grounded.MujocoGroundedReadyBackend.load(exact_identity)
    root_quat = np.asarray(motion["body_quat_w"][0, 0], dtype=np.float64)
    root_quat /= np.linalg.norm(root_quat)
    donor = grounded.ReadyState(
        np.asarray(motion["joint_pos"][0], dtype=np.float64),
        np.asarray(motion["body_pos_w"][0, 0], dtype=np.float64),
        root_quat,
    )
    input_grounding_audit = _audit_input_grounded_state(backend, donor)
    leg = np.asarray(LEG_JOINT_INDICES, dtype=np.int64)
    input_q = np.asarray(motion["joint_pos"])
    q_out, qd_out = _repair_constant_grounded_leg_velocity(
        input_q,
        np.asarray(motion["joint_vel"]),
        np.asarray(reference_arrays["joint_pos"]),
        np.asarray(reference_arrays["joint_pos"]),
    )
    nonleg = np.asarray(
        [index for index in range(31) if index not in set(LEG_JOINT_INDICES)],
        dtype=np.int64,
    )

    root_pos = np.asarray(motion["body_pos_w"][:, 0], dtype=np.float64)
    root_quat_rows = np.asarray(motion["body_quat_w"][:, 0], dtype=np.float64)
    zeros3 = np.zeros((int(motion["frames"]), 3), dtype=np.float64)
    candidate = schema2.build_schema2_candidate(
        joint_pos=q_out,
        joint_vel=qd_out,
        root_pos_w=root_pos,
        root_quat_wxyz=root_quat_rows,
        root_lin_vel_w=zeros3,
        root_ang_vel_w=zeros3,
        fps=float(motion["fps"]),
        mjcf_path=exact_identity.mjcf_path,
        input_sha256=input_sha,
        # The published A3 grounded candidate is the pinned source of the
        # constant leg coordinates; the actual input strike pose was audited
        # directly above instead of being rewritten into that neutral-arm pose.
        ready_sha256=reference_candidate_sha,
        body_order_path=body_order_path,
        migration_provenance=_migration_provenance(input_arrays),
    )
    output_arrays = candidate.arrays
    if int(output_arrays["joint_pos"].shape[0]) != int(motion["frames"]):
        raise GroundedUpperMaterializationError("schema-2 rebuild changed frame count")
    if not np.array_equal(output_arrays["fps"], input_arrays["fps"]):
        raise GroundedUpperMaterializationError("schema-2 rebuild changed fps bytes")
    if not np.array_equal(output_arrays["joint_pos"], input_arrays["joint_pos"]):
        raise GroundedUpperMaterializationError(
            "schema-2 rebuild changed joint_pos bytes"
        )
    if not np.array_equal(
        output_arrays["joint_vel"][:, nonleg],
        input_arrays["joint_vel"][:, nonleg],
    ):
        raise GroundedUpperMaterializationError(
            "schema-2 rebuild changed non-leg joint_vel bytes"
        )
    if np.count_nonzero(output_arrays["joint_vel"][:, leg]) != 0:
        raise GroundedUpperMaterializationError(
            "schema-2 rebuild emitted non-zero leg joint_vel"
        )
    if not np.array_equal(
        output_arrays["body_pos_w"][:, 0], input_arrays["body_pos_w"][:, 0]
    ) or not np.array_equal(
        output_arrays["body_quat_w"][:, 0], input_arrays["body_quat_w"][:, 0]
    ):
        raise GroundedUpperMaterializationError(
            "schema-2 rebuild changed the stored root pose"
        )

    before_trace = _model_site_trace(
        mujoco=backend._mujoco,
        model=backend.model,
        joint_pos=np.asarray(input_arrays["joint_pos"], dtype=np.float64),
        joint_vel=np.asarray(input_arrays["joint_vel"], dtype=np.float64),
        root_pos_w=np.asarray(input_arrays["body_pos_w"][:, 0], dtype=np.float64),
        root_quat_wxyz=np.asarray(
            input_arrays["body_quat_w"][:, 0], dtype=np.float64
        ),
    )
    after_trace = _model_site_trace(
        mujoco=backend._mujoco,
        model=backend.model,
        joint_pos=np.asarray(output_arrays["joint_pos"], dtype=np.float64),
        joint_vel=np.asarray(output_arrays["joint_vel"], dtype=np.float64),
        root_pos_w=np.asarray(output_arrays["body_pos_w"][:, 0], dtype=np.float64),
        root_quat_wxyz=np.asarray(
            output_arrays["body_quat_w"][:, 0], dtype=np.float64
        ),
    )
    site_proof = _assert_site_trace_bitwise_equal(before_trace, after_trace)
    strike = int(motion["strike_frame"])

    motion_filename = (
        input_path.stem + ".grounded_upper_qvel_fix_v1.npz"
    )
    schema_manifest_payload = _pretty_json_bytes(candidate.manifest)
    schema_report_payload = _pretty_json_bytes(candidate.report)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "tool_id": TOOL_ID,
        "artifact_class": ARTIFACT_CLASS,
        "verdict": "PASS_DIAGNOSTIC_QVEL_CONSISTENCY_REPAIR",
        "robot": {
            "family": "AgiBot A3",
            "exact_xml_model_name": exact_identity.xml_model_name,
            "grounded_ready_candidate_id": "G1",
            "candidate_id_semantics": (
                "G1 names an A3 grounded-ready construction candidate; "
                "it is not a robot model"
            ),
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "inputs": {
            "motion": {
                "path": str(input_path),
                "sha256": input_sha,
            },
            "canonical_ready_v1": {
                "path": str(ready_path),
                "sha256": ready_sha,
            },
            "grounded_reference_candidate": {
                "path": str(reference_candidate_path),
                "sha256": reference_candidate_sha,
            },
            "grounded_reference_receipt": {
                "path": str(reference_receipt_path),
                "sha256": reference_receipt_sha,
            },
            "body_order": {
                "path": str(body_order_path),
                "sha256": body_order_sha,
            },
            "exact_model": _jsonable(reference_receipt["exact_model"]),
        },
        "repair": {
            "semantics": (
                "input already had constant leg qpos from the published AgiBot "
                "A3 grounded-ready candidate (candidate_id=G1); repair only "
                "zeroed stale nonzero leg joint_vel and rebuilt schema-2 body "
                "FK/velocity metadata"
            ),
            "joint_pos_changed": False,
            "root_pose_changed": False,
            "nonleg_joint_velocity_changed": False,
            "leg_joint_velocity_zeroed": True,
            "leg_joint_indices": list(LEG_JOINT_INDICES),
            "leg_joint_names": list(LEG_JOINT_NAMES),
            "leg_velocity_before": motion["leg_velocity_before"],
            "leg_velocity_after_nonzero_samples": 0,
            "ready_yaw_rotation_from_v1_rad": motion["ready_yaw_rotation_rad"],
            "input_pose_exact_a3_grounding_audit": input_grounding_audit,
        },
        "invariants": {
            "frame_count_before": int(motion["frames"]),
            "frame_count_after": int(output_arrays["joint_pos"].shape[0]),
            "fps_before": float(motion["fps"]),
            "fps_after": float(np.asarray(output_arrays["fps"]).reshape(-1)[0]),
            "strike_frame_before": strike,
            "strike_frame_after": strike,
            "joint_pos_all_frames_bitwise_equal": True,
            "root_pose_all_frames_bitwise_equal": True,
            "nonleg_joint_vel_all_frames_bitwise_equal": True,
            "joint_velocity_first_last_exact_zero": bool(
                np.count_nonzero(output_arrays["joint_vel"][[0, -1]]) == 0
            ),
            "body_linear_velocity_first_last_exact_zero": bool(
                np.count_nonzero(output_arrays["body_lin_vel_w"][[0, -1]]) == 0
            ),
            "body_angular_velocity_first_last_exact_zero": bool(
                np.count_nonzero(output_arrays["body_ang_vel_w"][[0, -1]]) == 0
            ),
            "right_racket_site_all_frames": site_proof,
            "right_racket_site_speed_at_strike_before_mps": float(
                np.linalg.norm(before_trace.linear_velocity_w[strike])
            ),
            "right_racket_site_speed_at_strike_after_mps": float(
                np.linalg.norm(after_trace.linear_velocity_w[strike])
            ),
        },
        "outputs": {
            "motion_filename": motion_filename,
            "motion_sha256": candidate.output_sha256,
            "schema2_manifest_filename": SCHEMA2_MANIFEST_FILENAME,
            "schema2_manifest_sha256": _sha256_bytes(schema_manifest_payload),
            "schema2_report_filename": SCHEMA2_REPORT_FILENAME,
            "schema2_report_sha256": _sha256_bytes(schema_report_payload),
            "receipt_filename": RECEIPT_FILENAME,
            "completion_semantics": "exclusive_directory_receipt_written_last",
        },
        "producer": {
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": _sha256_file(Path(__file__).resolve(), "materializer"),
            "schema2_tool_path": str(Path(schema2.__file__).resolve()),
            "schema2_tool_sha256": _sha256_file(
                Path(schema2.__file__).resolve(), "schema-2 builder"
            ),
            "grounded_ready_tool_path": str(Path(grounded.__file__).resolve()),
            "grounded_ready_tool_sha256": _sha256_file(
                Path(grounded.__file__).resolve(), "grounded-ready tool"
            ),
        },
        "non_claims": [
            "no joint-position transplant was needed or performed",
            "not a compiler-bank or behavior certificate",
            "not a table-contact or physical-ball certificate",
            "not a policy-quality or curriculum certificate",
            "not deployment or hardware authorization",
        ],
    }
    return _publish_bundle(
        output_directory=Path(args.output_dir),
        motion_filename=motion_filename,
        motion_payload=candidate.npz_bytes,
        schema_manifest_payload=schema_manifest_payload,
        schema_report_payload=schema_report_payload,
        receipt=receipt,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--canonical-ready-v1", required=True)
    parser.add_argument("--expected-canonical-ready-v1-sha256", required=True)
    parser.add_argument("--grounded-reference-candidate", required=True)
    parser.add_argument(
        "--expected-grounded-reference-candidate-sha256", required=True
    )
    parser.add_argument("--grounded-reference-receipt", required=True)
    parser.add_argument(
        "--expected-grounded-reference-receipt-sha256", required=True
    )
    parser.add_argument("--body-order", required=True)
    parser.add_argument("--expected-body-order-sha256", required=True)
    parser.add_argument("--strike-frame", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = materialize(_parser().parse_args(argv))
    except (
        GroundedUpperMaterializationError,
        grounded.GroundedReadyError,
        schema2.Schema2BuildError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "PASS_DIAGNOSTIC_QVEL_CONSISTENCY_REPAIR",
                "directory": str(result.directory),
                "motion": str(result.motion),
                "motion_sha256": result.motion_sha256,
                "receipt": str(result.receipt),
                "receipt_sha256": result.receipt_sha256,
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
