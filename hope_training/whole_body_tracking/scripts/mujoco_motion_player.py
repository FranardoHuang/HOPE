#!/usr/bin/env python3
"""Kinematic NPZ player and FK smoke check for the AgiBot vendor MuJoCo model.

This tool answers a deliberately narrow question: can one exact schema-2 motion
asset be decoded in the repository's 31-joint/32-body runtime order and posed in
the vendor MJCF without a name/order mismatch?

Two modes share the same fail-closed loader and name-based model binding:

* default (headless): write the stored free-root pose and all 31 joint
  pose/velocity channels for every frame, call ``mj_forward``, compare all 32
  link-origin poses with the NPZ, and report the bound ``right_racket`` site
  position, local +Y face normal, and point twist.  Site velocity comes from
  ``mj_jacSite``/``mj_objectVelocity`` using schema root/joint qvel, never pose
  finite differences;
* ``--viewer``: show the same per-frame kinematic poses in MuJoCo's passive
  viewer.  ``--speed`` and ``--loop`` only control wall-clock playback.

Evidence boundary
-----------------
This is *kinematic playback*, not a dynamics, controller, learnability, balance,
contact, training, deployment, hardware, or real-robot certificate.  It calls
``mj_forward`` and calls ``mj_step`` zero times.  Use the existing
``motion_dynamic_replay.py`` separately for a first dynamics check;
``--suggest-dynamic-replay`` prints (but never runs) that command.

The module intentionally imports MuJoCo only after CLI parsing/model loading so
that schema and helper tests run on machines without MuJoCo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
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
    read_metadata,
)
from racket_geometry_contract import RACKET_SITE_OFFSET_WRIST_M  # noqa: E402


DEFAULT_MJCF = (
    _REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)

# NPZ joint_pos/joint_vel columns have no joint-name metadata in the exact
# 11/14-field schema.  They therefore have one legal interpretation: the
# repository's canonical runtime/Isaac articulation order.
RUNTIME_JOINT_NAMES = tuple(ISAAC_JOINT_NAMES)

# Exact schema-2 body-column truth.  This is the complete 32-link runtime order,
# not the 14-body policy-observation subset in mujoco_eval_onnx.TRACKED_BODIES.
RUNTIME_BODY_ORDER_PATH = _REPO_ROOT / "configs/a3_runtime_body_order.txt"
try:
    RUNTIME_BODY_NAMES = tuple(
        line.strip()
        for line in RUNTIME_BODY_ORDER_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
except OSError as exc:
    raise RuntimeError(
        f"cannot read runtime body-order contract {RUNTIME_BODY_ORDER_PATH}: {exc}"
    ) from exc

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

DEFAULT_POSITION_TOL_M = 1.0e-4
DEFAULT_ORIENTATION_TOL_RAD = 1.0e-4
DEFAULT_RACKET_POSITION_TOL_M = 1.0e-4
DEFAULT_RACKET_NORMAL_TOL_RAD = 1.0e-4
DEFAULT_RACKET_LINEAR_VELOCITY_TOL_M_S = 1.0e-3
DEFAULT_RACKET_ANGULAR_VELOCITY_TOL_RAD_S = 1.0e-3
DEFAULT_RACKET_JACOBIAN_TOL = 1.0e-9
QUATERNION_NORM_TOL = 2.0e-3
RACKET_SITE_NAME = "right_racket"
RACKET_SITE_BODY_NAME = "right_wrist_yaw_Link"
RACKET_NORMAL_LOCAL_AXIS = 1
RACKET_NORMAL_CONVENTION = "right_racket_site_local_plus_y_world_v1"
RACKET_REFERENCE_SCHEMA_VERSION = 1
RACKET_REFERENCE_KEYS = frozenset(
    {
        "racket_reference_schema_version",
        "site_name",
        "normal_convention",
        "site_pos_w",
        "site_normal_w",
        "site_lin_vel_w",
        "site_ang_vel_w",
    }
)
PASS = "PASS"
FAIL = "FAIL"

if len(RUNTIME_JOINT_NAMES) != 31 or len(set(RUNTIME_JOINT_NAMES)) != 31:
    raise RuntimeError("canonical runtime joint contract must contain 31 unique names")
if len(RUNTIME_BODY_NAMES) != 32 or len(set(RUNTIME_BODY_NAMES)) != 32:
    raise RuntimeError("canonical runtime body contract must contain 32 unique names")


@dataclass(frozen=True)
class MotionClip:
    """Validated, exact-schema motion arrays."""

    path: Path
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    has_migration_provenance: bool
    body_lin_vel_point: str

    @property
    def n_frames(self) -> int:
        return int(self.joint_pos.shape[0])


@dataclass(frozen=True)
class ModelBinding:
    """Name-resolved root, joint, body, and racket-site addresses."""

    root_qpos_adr: int
    root_dof_adr: int
    joint_qpos_adrs: np.ndarray
    joint_dof_adrs: np.ndarray
    body_ids: np.ndarray
    racket_site_id: int
    racket_site_body_id: int
    racket_site_body_column: int


@dataclass(frozen=True)
class RacketTrajectoryReference:
    """Optional external expected trajectory for the bound racket site."""

    path: Path
    site_pos_w: np.ndarray
    site_normal_w: np.ndarray
    site_lin_vel_w: np.ndarray
    site_ang_vel_w: np.ndarray


def _scalar_text(value: Any, label: str) -> str:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{label} must be scalar, got shape {np.asarray(value).shape}")
    raw = array[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw)


def _validate_migration_provenance(data: Any) -> bool:
    present = tuple(key in data.files for key in MIGRATION_KEYS)
    if any(present) and not all(present):
        raise ValueError("migration provenance must be all-or-none")
    if not all(present):
        return False
    digest = _scalar_text(data[MIGRATION_SOURCE_SHA256_KEY], MIGRATION_SOURCE_SHA256_KEY)
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError(f"{MIGRATION_SOURCE_SHA256_KEY} must be a 64-digit hex digest")
    if not _scalar_text(data[MIGRATION_SOURCE_POINT_KEY], MIGRATION_SOURCE_POINT_KEY):
        raise ValueError(f"{MIGRATION_SOURCE_POINT_KEY} must be non-empty")
    if not _scalar_text(data[MIGRATION_TOOL_KEY], MIGRATION_TOOL_KEY):
        raise ValueError(f"{MIGRATION_TOOL_KEY} must be non-empty")
    return True


def _finite_array(data: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(data[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a numeric array") from exc
    if array.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains NaN/Inf")
    return array


def load_motion(path: os.PathLike[str] | str) -> MotionClip:
    """Load one exact 11/14-field schema-2 asset and bind its runtime orders.

    ``allow_pickle=False`` is intentional.  A motion asset is numeric/string data,
    never executable Python objects.
    """

    motion_path = Path(path).expanduser().resolve()
    if not motion_path.is_file():
        raise ValueError(f"motion NPZ not found: {motion_path}")

    try:
        archive = np.load(motion_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read motion NPZ {motion_path}: {exc}") from exc

    with archive as data:
        keys = frozenset(data.files)
        if keys not in ALLOWED_KEYSETS:
            expected = [sorted(keyset) for keyset in ALLOWED_KEYSETS]
            raise ValueError(
                f"motion field set changed: got {sorted(keys)}; "
                f"expected exact schema-2 11/14 fields {expected}"
            )
        has_migration = _validate_migration_provenance(data)
        try:
            metadata = read_metadata(data)
        except ValueError as exc:
            raise ValueError(f"motion metadata contract: {exc}") from exc
        if (
            metadata.schema_version != KINEMATICS_SCHEMA_VERSION
            or metadata.body_pos_point != BODY_POS_POINT
            or metadata.body_lin_vel_point not in (BODY_LIN_VEL_POINT, BODY_POS_POINT)
            or metadata.body_names != RUNTIME_BODY_NAMES
        ):
            raise ValueError(
                "motion must bind exact schema-2 metadata: "
                f"{SCHEMA_KEY}=2, {POS_POINT_KEY}={BODY_POS_POINT!r}, "
                f"{LIN_VEL_POINT_KEY} in "
                f"({BODY_LIN_VEL_POINT!r}, {BODY_POS_POINT!r}), and the exact "
                "32-name A3 runtime body order"
            )

        fps_array = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
        if fps_array.size != 1:
            raise ValueError(f"fps must contain one scalar, got {np.asarray(data['fps']).shape}")
        fps = float(fps_array[0])
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {fps!r}")

        joint_pos_raw = np.asarray(data["joint_pos"])
        if joint_pos_raw.ndim != 2:
            raise ValueError(f"joint_pos must be (T,31), got {joint_pos_raw.shape}")
        frames = int(joint_pos_raw.shape[0])
        if frames < 1:
            raise ValueError("motion must contain at least one frame")

        joint_pos = _finite_array(data, "joint_pos", (frames, 31))
        joint_vel = _finite_array(data, "joint_vel", (frames, 31))
        body_pos = _finite_array(data, "body_pos_w", (frames, 32, 3))
        body_quat = _finite_array(data, "body_quat_w", (frames, 32, 4))
        body_lin = _finite_array(data, "body_lin_vel_w", (frames, 32, 3))
        body_ang = _finite_array(data, "body_ang_vel_w", (frames, 32, 3))

        quat_norm = np.linalg.norm(body_quat, axis=-1)
        max_norm_error = float(np.max(np.abs(quat_norm - 1.0)))
        if max_norm_error > QUATERNION_NORM_TOL:
            raise ValueError(
                "body_quat_w contains non-unit quaternions: "
                f"max |norm-1|={max_norm_error:.6g} > {QUATERNION_NORM_TOL:.6g}"
            )

    return MotionClip(
        path=motion_path,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        has_migration_provenance=has_migration,
        body_lin_vel_point=str(metadata.body_lin_vel_point),
    )


def load_racket_reference(
    path: os.PathLike[str] | str, frames: int
) -> RacketTrajectoryReference:
    """Load an optional exact racket-site trajectory sidecar.

    The sidecar is deliberately separate from schema-2: it is validation
    evidence, never an input used to construct or repair the motion.
    """

    reference_path = Path(path).expanduser().resolve()
    if not reference_path.is_file():
        raise ValueError(f"racket reference NPZ not found: {reference_path}")
    try:
        archive = np.load(reference_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"cannot read racket reference NPZ {reference_path}: {exc}"
        ) from exc

    with archive as data:
        keys = frozenset(data.files)
        if keys != RACKET_REFERENCE_KEYS:
            raise ValueError(
                "racket reference field set changed: "
                f"got {sorted(keys)}, expected {sorted(RACKET_REFERENCE_KEYS)}"
            )
        version_raw = np.asarray(
            data["racket_reference_schema_version"]
        ).reshape(-1)
        if version_raw.size != 1 or int(version_raw[0]) != RACKET_REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                "racket_reference_schema_version must be scalar integer 1"
            )
        if _scalar_text(data["site_name"], "site_name") != RACKET_SITE_NAME:
            raise ValueError(
                f"racket reference site_name must be {RACKET_SITE_NAME!r}"
            )
        if (
            _scalar_text(data["normal_convention"], "normal_convention")
            != RACKET_NORMAL_CONVENTION
        ):
            raise ValueError(
                "racket reference normal_convention must be "
                f"{RACKET_NORMAL_CONVENTION!r}"
            )
        site_pos = _finite_array(data, "site_pos_w", (frames, 3))
        site_normal = _finite_array(data, "site_normal_w", (frames, 3))
        site_lin = _finite_array(data, "site_lin_vel_w", (frames, 3))
        site_ang = _finite_array(data, "site_ang_vel_w", (frames, 3))
        normal_error = np.max(
            np.abs(np.linalg.norm(site_normal, axis=1) - 1.0)
        )
        if float(normal_error) > 1.0e-6:
            raise ValueError(
                "racket reference site_normal_w must contain unit vectors; "
                f"max |norm-1|={float(normal_error):.6g}"
            )

    return RacketTrajectoryReference(
        path=reference_path,
        site_pos_w=site_pos,
        site_normal_w=site_normal,
        site_lin_vel_w=site_lin,
        site_ang_vel_w=site_ang,
    )


def load_mujoco() -> Any:
    """Deferred import with a user-facing dependency error."""

    try:
        return importlib.import_module("mujoco")
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Python is required for playback/FK smoke; install it in the "
            "documented MuJoCo environment"
        ) from exc


def _name_id(mujoco: Any, model: Any, obj_type: Any, name: str, label: str) -> int:
    value = int(mujoco.mj_name2id(model, obj_type, name))
    if value < 0:
        raise ValueError(f"MJCF is missing {label} {name!r}")
    return value


def _unique_site_id(mujoco: Any, model: Any, name: str) -> int:
    """Resolve one site and independently prove its name is unique."""

    if not hasattr(model, "nsite") or not hasattr(mujoco, "mj_id2name"):
        raise ValueError(
            "MuJoCo model API cannot prove the right_racket site is unique"
        )
    matching = [
        site_id
        for site_id in range(int(model.nsite))
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) == name
    ]
    if len(matching) != 1:
        raise ValueError(
            f"MJCF must contain exactly one site {name!r}, found {len(matching)}"
        )
    resolved = _name_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_SITE, name, "racket site"
    )
    if resolved != matching[0]:
        raise ValueError(
            f"MuJoCo name lookup for site {name!r} disagrees with the site table"
        )
    return resolved


def bind_model(mujoco: Any, model: Any) -> ModelBinding:
    """Resolve the exact A3 root/joints/bodies/racket by name."""

    joint_ids = np.asarray(
        [
            _name_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
                "runtime joint",
            )
            for name in RUNTIME_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    if len(set(joint_ids.tolist())) != 31:
        raise ValueError("MJCF runtime joint names do not resolve bijectively")

    joint_types = np.asarray(model.jnt_type)[joint_ids]
    if np.any(joint_types != int(mujoco.mjtJoint.mjJNT_HINGE)):
        bad = [
            RUNTIME_JOINT_NAMES[index]
            for index, value in enumerate(joint_types)
            if int(value) != int(mujoco.mjtJoint.mjJNT_HINGE)
        ]
        raise ValueError(f"A3 runtime joints must be scalar hinges, got {bad}")
    joint_qpos_adrs = np.asarray(model.jnt_qposadr, dtype=np.int64)[joint_ids]
    joint_dof_adrs = np.asarray(model.jnt_dofadr, dtype=np.int64)[joint_ids]
    if len(set(joint_qpos_adrs.tolist())) != 31:
        raise ValueError("MJCF runtime joints do not have 31 unique qpos addresses")
    if len(set(joint_dof_adrs.tolist())) != 31:
        raise ValueError("MJCF runtime joints do not have 31 unique dof addresses")

    body_ids = np.asarray(
        [
            _name_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
                "runtime body",
            )
            for name in RUNTIME_BODY_NAMES
        ],
        dtype=np.int64,
    )
    if len(set(body_ids.tolist())) != 32:
        raise ValueError("MJCF runtime body names do not resolve bijectively")

    pelvis_id = int(body_ids[0])
    joint_type_all = np.asarray(model.jnt_type)
    joint_body_all = np.asarray(model.jnt_bodyid)
    root_candidates = np.flatnonzero(
        (joint_type_all == int(mujoco.mjtJoint.mjJNT_FREE))
        & (joint_body_all == pelvis_id)
    )
    if root_candidates.shape != (1,):
        raise ValueError(
            "pelvis_link must own exactly one freejoint, "
            f"found {root_candidates.size}"
        )
    root_qpos_adr = int(np.asarray(model.jnt_qposadr)[int(root_candidates[0])])
    root_dof_adr = int(np.asarray(model.jnt_dofadr)[int(root_candidates[0])])
    if root_qpos_adr != 0:
        raise ValueError(
            f"vendor pelvis freejoint must start at qpos address 0, got {root_qpos_adr}"
        )
    if root_dof_adr != 0:
        raise ValueError(
            f"vendor pelvis freejoint must start at dof address 0, got {root_dof_adr}"
        )
    if np.any(joint_qpos_adrs < 7) or np.any(joint_qpos_adrs >= int(model.nq)):
        raise ValueError("runtime joint qpos address falls outside the scalar-joint region")
    if np.any(joint_dof_adrs < 6) or np.any(joint_dof_adrs >= int(model.nv)):
        raise ValueError("runtime joint dof address falls outside the scalar-joint region")

    racket_site_id = _unique_site_id(mujoco, model, RACKET_SITE_NAME)
    racket_site_body_id = int(np.asarray(model.site_bodyid)[racket_site_id])
    expected_site_body_id = _name_id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        RACKET_SITE_BODY_NAME,
        "racket site body",
    )
    if racket_site_body_id != expected_site_body_id:
        actual = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, racket_site_body_id
        )
        raise ValueError(
            f"{RACKET_SITE_NAME!r} must be attached to {RACKET_SITE_BODY_NAME!r}, "
            f"got {actual!r}"
        )
    matches = np.flatnonzero(body_ids == racket_site_body_id)
    if matches.shape != (1,):
        raise ValueError(
            "right_racket body must occur exactly once in the runtime body order"
        )
    site_position = np.asarray(model.site_pos, dtype=np.float64)[racket_site_id]
    if not np.allclose(
        site_position,
        np.asarray(RACKET_SITE_OFFSET_WRIST_M, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"{RACKET_SITE_NAME!r} local position drifted: "
            f"got {site_position.tolist()}, "
            f"expected {RACKET_SITE_OFFSET_WRIST_M.tolist()}"
        )
    site_quat = np.asarray(model.site_quat, dtype=np.float64)[racket_site_id]
    if not np.allclose(
        site_quat,
        np.array([1.0, 0.0, 0.0, 0.0]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"{RACKET_SITE_NAME!r} local orientation drifted; "
            "the formal face normal is the wrist/site local +Y axis"
        )
    return ModelBinding(
        root_qpos_adr=root_qpos_adr,
        root_dof_adr=root_dof_adr,
        joint_qpos_adrs=joint_qpos_adrs,
        joint_dof_adrs=joint_dof_adrs,
        body_ids=body_ids,
        racket_site_id=racket_site_id,
        racket_site_body_id=racket_site_body_id,
        racket_site_body_column=int(matches[0]),
    )


def reset_kinematic_state(mujoco: Any, model: Any, data: Any) -> None:
    """Reset once; no stepping or control is used."""

    mujoco.mj_resetData(model, data)
    if hasattr(data, "qvel"):
        data.qvel[:] = 0.0
    if hasattr(data, "act"):
        data.act[:] = 0.0
    if hasattr(data, "ctrl"):
        data.ctrl[:] = 0.0


def apply_frame(
    mujoco: Any,
    model: Any,
    data: Any,
    binding: ModelBinding,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
) -> None:
    """Write one free-root + 31-joint pose by resolved address and run FK."""

    root = binding.root_qpos_adr
    data.qpos[root : root + 3] = root_pos
    data.qpos[root + 3 : root + 7] = root_quat_wxyz
    data.qpos[binding.joint_qpos_adrs] = joint_pos
    mujoco.mj_forward(model, data)


def apply_frame_velocity(
    clip: MotionClip,
    frame: int,
    mujoco: Any,
    model: Any,
    data: Any,
    binding: ModelBinding,
) -> float:
    """Install schema root/joint velocity and return root-twist residual.

    Schema-2 stores root position at the link origin but may store linear
    velocity at either the link origin or body CoM.  The free-joint qvel is
    recovered from the exact root-body Jacobian instead of assuming a MuJoCo
    angular-velocity basis.
    """

    data.qvel[:] = 0.0
    data.qvel[binding.joint_dof_adrs] = clip.joint_vel[frame]
    root_angular_w = clip.body_ang_vel_w[frame, 0]
    root_linear_w = clip.body_lin_vel_w[frame, 0]
    if clip.body_lin_vel_point == BODY_LIN_VEL_POINT:
        root_offset_w = (
            np.asarray(data.xipos[binding.body_ids[0]], dtype=np.float64)
            - np.asarray(data.xpos[binding.body_ids[0]], dtype=np.float64)
        )
        root_linear_w = root_linear_w - np.cross(
            root_angular_w, root_offset_w
        )

    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jacBody(
        model, data, jacp, jacr, int(binding.body_ids[0])
    )
    root_slice = slice(binding.root_dof_adr, binding.root_dof_adr + 6)
    matrix = np.vstack((jacp[:, root_slice], jacr[:, root_slice]))
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > 1.0e10:
        raise ValueError(
            "pelvis free-root twist Jacobian is singular or ill-conditioned: "
            f"{condition:.6g}"
        )
    nonroot = np.concatenate((jacp @ data.qvel, jacr @ data.qvel))
    target = np.concatenate((root_linear_w, root_angular_w))
    data.qvel[root_slice] = np.linalg.solve(matrix, target - nonroot)
    reproduced = np.concatenate((jacp @ data.qvel, jacr @ data.qvel))
    residual = float(np.max(np.abs(reproduced - target)))
    if not math.isfinite(residual):
        raise ValueError("root-twist reconstruction produced NaN/Inf")
    mujoco.mj_forward(model, data)
    return residual


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert one finite unit wxyz quaternion to a world-from-body matrix."""

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("quaternion is zero")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def racket_site_state(
    mujoco: Any,
    model: Any,
    data: Any,
    binding: ModelBinding,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Read the bound site and cross-check object velocity against its Jacobian."""

    site_id = binding.racket_site_id
    position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    normal = rotation[:, RACKET_NORMAL_LOCAL_AXIS].copy()

    object_twist = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_SITE,
        site_id,
        object_twist,
        0,
    )
    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac_linear = jacp @ np.asarray(data.qvel, dtype=np.float64)
    jac_angular = jacr @ np.asarray(data.qvel, dtype=np.float64)
    angular = object_twist[:3].copy()
    linear = object_twist[3:].copy()
    values = (position, normal, linear, angular, jac_linear, jac_angular)
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("MuJoCo racket-site state produced NaN/Inf")
    return (
        position,
        normal,
        linear,
        angular,
        float(np.max(np.abs(linear - jac_linear))),
        float(np.max(np.abs(angular - jac_angular))),
    )


def quaternion_angle_error_rad(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Shortest SO(3) angle; q and -q are treated as the same orientation."""

    lhs = np.asarray(actual, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.shape[-1] != 4:
        raise ValueError(f"quaternion shapes must match (...,4), got {lhs.shape}/{rhs.shape}")
    lhs = lhs / np.maximum(np.linalg.norm(lhs, axis=-1, keepdims=True), 1.0e-15)
    rhs = rhs / np.maximum(np.linalg.norm(rhs, axis=-1, keepdims=True), 1.0e-15)
    dot = np.abs(np.sum(lhs * rhs, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def direction_angle_error_rad(
    actual: np.ndarray, expected: np.ndarray
) -> np.ndarray:
    """Directed vector angle; +Y and -Y are intentionally not equivalent."""

    lhs = np.asarray(actual, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.shape[-1] != 3:
        raise ValueError(
            f"direction shapes must match (...,3), got {lhs.shape}/{rhs.shape}"
        )
    lhs_norm = np.linalg.norm(lhs, axis=-1, keepdims=True)
    rhs_norm = np.linalg.norm(rhs, axis=-1, keepdims=True)
    if np.any(lhs_norm <= 0.0) or np.any(rhs_norm <= 0.0):
        raise ValueError("direction contains a zero vector")
    dot = np.sum((lhs / lhs_norm) * (rhs / rhs_norm), axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def array_sha256(value: np.ndarray) -> str:
    """Content-address one numeric trajectory with explicit dtype and shape."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(b"numpy-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _combined_racket_sha256(arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"right-racket-trajectory-v1\0")
    for name, value in arrays:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def smoke_check(
    clip: MotionClip,
    mujoco: Any,
    model: Any,
    data: Any,
    binding: ModelBinding,
    *,
    position_tol_m: float = DEFAULT_POSITION_TOL_M,
    orientation_tol_rad: float = DEFAULT_ORIENTATION_TOL_RAD,
    racket_reference: RacketTrajectoryReference | None = None,
    racket_position_tol_m: float = DEFAULT_RACKET_POSITION_TOL_M,
    racket_normal_tol_rad: float = DEFAULT_RACKET_NORMAL_TOL_RAD,
    racket_linear_velocity_tol_m_s: float = (
        DEFAULT_RACKET_LINEAR_VELOCITY_TOL_M_S
    ),
    racket_angular_velocity_tol_rad_s: float = (
        DEFAULT_RACKET_ANGULAR_VELOCITY_TOL_RAD_S
    ),
    racket_jacobian_tol: float = DEFAULT_RACKET_JACOBIAN_TOL,
) -> dict[str, Any]:
    """Recompute body FK and the complete right-racket site trajectory."""

    for value, label in (
        (position_tol_m, "position_tol_m"),
        (orientation_tol_rad, "orientation_tol_rad"),
        (racket_position_tol_m, "racket_position_tol_m"),
        (racket_normal_tol_rad, "racket_normal_tol_rad"),
        (
            racket_linear_velocity_tol_m_s,
            "racket_linear_velocity_tol_m_s",
        ),
        (
            racket_angular_velocity_tol_rad_s,
            "racket_angular_velocity_tol_rad_s",
        ),
        (racket_jacobian_tol, "racket_jacobian_tol"),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
    if racket_reference is not None:
        for label, values in (
            ("site_pos_w", racket_reference.site_pos_w),
            ("site_normal_w", racket_reference.site_normal_w),
            ("site_lin_vel_w", racket_reference.site_lin_vel_w),
            ("site_ang_vel_w", racket_reference.site_ang_vel_w),
        ):
            array = np.asarray(values, dtype=np.float64)
            expected_shape = (clip.n_frames, 3)
            if array.shape != expected_shape:
                raise ValueError(
                    f"racket reference {label} must have shape "
                    f"{expected_shape}, got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"racket reference {label} contains NaN/Inf"
                )
        normal_norm_error = float(
            np.max(
                np.abs(
                    np.linalg.norm(racket_reference.site_normal_w, axis=1)
                    - 1.0
                )
            )
        )
        if normal_norm_error > 1.0e-6:
            raise ValueError(
                "racket reference site_normal_w must contain unit vectors; "
                f"max |norm-1|={normal_norm_error:.6g}"
            )

    reset_kinematic_state(mujoco, model, data)
    per_body_pos = np.zeros(32, dtype=np.float64)
    per_body_ori = np.zeros(32, dtype=np.float64)
    worst_pos = (-1.0, -1, -1)
    worst_ori = (-1.0, -1, -1)
    site_pos = np.empty((clip.n_frames, 3), dtype=np.float64)
    site_normal = np.empty((clip.n_frames, 3), dtype=np.float64)
    site_lin = np.empty((clip.n_frames, 3), dtype=np.float64)
    site_ang = np.empty((clip.n_frames, 3), dtype=np.float64)
    schema_site_pos = np.empty_like(site_pos)
    schema_site_normal = np.empty_like(site_normal)
    schema_site_lin = np.empty_like(site_lin)
    schema_site_ang = np.empty_like(site_ang)
    jac_lin_error = np.zeros(clip.n_frames, dtype=np.float64)
    jac_ang_error = np.zeros(clip.n_frames, dtype=np.float64)
    root_twist_error = np.zeros(clip.n_frames, dtype=np.float64)

    for frame in range(clip.n_frames):
        apply_frame(
            mujoco,
            model,
            data,
            binding,
            clip.body_pos_w[frame, 0],
            clip.body_quat_w[frame, 0],
            clip.joint_pos[frame],
        )
        root_twist_error[frame] = apply_frame_velocity(
            clip, frame, mujoco, model, data, binding
        )
        actual_pos = np.asarray(data.xpos, dtype=np.float64)[binding.body_ids]
        actual_quat = np.asarray(data.xquat, dtype=np.float64)[binding.body_ids]
        if not (np.isfinite(actual_pos).all() and np.isfinite(actual_quat).all()):
            raise ValueError(f"MuJoCo FK produced NaN/Inf at frame {frame}")
        pos_error = np.linalg.norm(actual_pos - clip.body_pos_w[frame], axis=-1)
        ori_error = quaternion_angle_error_rad(actual_quat, clip.body_quat_w[frame])
        per_body_pos = np.maximum(per_body_pos, pos_error)
        per_body_ori = np.maximum(per_body_ori, ori_error)
        pos_body = int(np.argmax(pos_error))
        ori_body = int(np.argmax(ori_error))
        if float(pos_error[pos_body]) > worst_pos[0]:
            worst_pos = (float(pos_error[pos_body]), frame, pos_body)
        if float(ori_error[ori_body]) > worst_ori[0]:
            worst_ori = (float(ori_error[ori_body]), frame, ori_body)

        (
            site_pos[frame],
            site_normal[frame],
            site_lin[frame],
            site_ang[frame],
            jac_lin_error[frame],
            jac_ang_error[frame],
        ) = racket_site_state(mujoco, model, data, binding)

        body_column = binding.racket_site_body_column
        body_rotation = quaternion_wxyz_to_matrix(
            clip.body_quat_w[frame, body_column]
        )
        schema_site_pos[frame] = (
            clip.body_pos_w[frame, body_column]
            + body_rotation @ RACKET_SITE_OFFSET_WRIST_M
        )
        schema_site_normal[frame] = body_rotation[
            :, RACKET_NORMAL_LOCAL_AXIS
        ]
        point_origin = (
            np.asarray(
                data.xipos[binding.racket_site_body_id], dtype=np.float64
            )
            if clip.body_lin_vel_point == BODY_LIN_VEL_POINT
            else np.asarray(
                data.xpos[binding.racket_site_body_id], dtype=np.float64
            )
        )
        body_angular = clip.body_ang_vel_w[frame, body_column]
        schema_site_lin[frame] = (
            clip.body_lin_vel_w[frame, body_column]
            + np.cross(body_angular, site_pos[frame] - point_origin)
        )
        schema_site_ang[frame] = body_angular

    pos_pass = worst_pos[0] <= float(position_tol_m)
    ori_pass = worst_ori[0] <= float(orientation_tol_rad)
    site_pos_error = np.linalg.norm(site_pos - schema_site_pos, axis=1)
    site_normal_error = direction_angle_error_rad(
        site_normal, schema_site_normal
    )
    site_lin_error = np.linalg.norm(site_lin - schema_site_lin, axis=1)
    site_ang_error = np.linalg.norm(site_ang - schema_site_ang, axis=1)
    site_pos_pass = float(np.max(site_pos_error)) <= float(
        racket_position_tol_m
    )
    site_normal_pass = float(np.max(site_normal_error)) <= float(
        racket_normal_tol_rad
    )
    site_lin_pass = float(np.max(site_lin_error)) <= float(
        racket_linear_velocity_tol_m_s
    )
    site_ang_pass = float(np.max(site_ang_error)) <= float(
        racket_angular_velocity_tol_rad_s
    )
    jacobian_pass = bool(
        float(np.max(jac_lin_error)) <= float(racket_jacobian_tol)
        and float(np.max(jac_ang_error)) <= float(racket_jacobian_tol)
        and float(np.max(root_twist_error)) <= float(racket_jacobian_tol)
    )

    external_errors: dict[str, np.ndarray] | None = None
    external_pass = True
    if racket_reference is not None:
        external_errors = {
            "position_m": np.linalg.norm(
                site_pos - racket_reference.site_pos_w, axis=1
            ),
            "normal_rad": direction_angle_error_rad(
                site_normal, racket_reference.site_normal_w
            ),
            "linear_velocity_m_s": np.linalg.norm(
                site_lin - racket_reference.site_lin_vel_w, axis=1
            ),
            "angular_velocity_rad_s": np.linalg.norm(
                site_ang - racket_reference.site_ang_vel_w, axis=1
            ),
        }
        external_pass = bool(
            float(np.max(external_errors["position_m"]))
            <= float(racket_position_tol_m)
            and float(np.max(external_errors["normal_rad"]))
            <= float(racket_normal_tol_rad)
            and float(np.max(external_errors["linear_velocity_m_s"]))
            <= float(racket_linear_velocity_tol_m_s)
            and float(np.max(external_errors["angular_velocity_rad_s"]))
            <= float(racket_angular_velocity_tol_rad_s)
        )

    overall_pass = bool(
        pos_pass
        and ori_pass
        and site_pos_pass
        and site_normal_pass
        and site_lin_pass
        and site_ang_pass
        and jacobian_pass
        and external_pass
    )
    racket_arrays = (
        ("site_pos_w", site_pos),
        ("site_normal_w", site_normal),
        ("site_lin_vel_w", site_lin),
        ("site_ang_vel_w", site_ang),
    )
    lin_peak_frame = int(np.argmax(np.linalg.norm(site_lin, axis=1)))
    ang_peak_frame = int(np.argmax(np.linalg.norm(site_ang, axis=1)))

    def maximum_row(
        values: np.ndarray, label: str, threshold: float
    ) -> dict[str, Any]:
        frame = int(np.argmax(values))
        maximum = float(values[frame])
        return {
            "pass": bool(maximum <= float(threshold)),
            "threshold": float(threshold),
            f"max_error_{label}": maximum,
            "worst_frame": frame,
        }

    return {
        "verdict": PASS if overall_pass else FAIL,
        "evidence_boundary": {
            "level": "kinematic_playback_only",
            "mj_forward_calls": 2 * clip.n_frames,
            "mj_step_calls": 0,
            "dynamic_certificate": False,
            "training_certificate": False,
            "deployment_certificate": False,
            "hardware_certificate": False,
            "real_robot_certificate": False,
            "racket_velocity_source": (
                "schema root/joint qvel -> MuJoCo root Jacobian -> "
                "mj_jacSite and mj_objectVelocity; no pose finite differences"
            ),
            "statement": (
                "Checks schema/order/name binding, FK pose agreement, and the "
                "kinematic right-racket site trajectory only; it does not test "
                "dynamics, balance, contact, control, or learnability."
            ),
        },
        "authorization": {
            "training": False,
            "deployment": False,
            "hardware": False,
        },
        "contract": {
            "schema": "exact schema-2 11/14 fields",
            "joint_columns": 31,
            "body_columns": 32,
            "joint_order": list(RUNTIME_JOINT_NAMES),
            "body_order": list(RUNTIME_BODY_NAMES),
            "joint_mapping": "name_to_mjcf_qpos_address",
            "body_mapping": "name_to_mjcf_body_id",
            "racket_site": RACKET_SITE_NAME,
            "racket_site_body": RACKET_SITE_BODY_NAME,
            "racket_site_local_position_m": RACKET_SITE_OFFSET_WRIST_M.tolist(),
            "racket_normal_convention": RACKET_NORMAL_CONVENTION,
        },
        "motion": {
            "path": str(clip.path),
            "frames": clip.n_frames,
            "fps": clip.fps,
            "duration_s": (clip.n_frames - 1) / clip.fps,
            "migration_provenance": clip.has_migration_provenance,
            "body_lin_vel_point": clip.body_lin_vel_point,
        },
        "gates": {
            "position": {
                "pass": bool(pos_pass),
                "threshold_m": float(position_tol_m),
                "max_error_m": worst_pos[0],
                "worst_frame": worst_pos[1],
                "worst_body": RUNTIME_BODY_NAMES[worst_pos[2]],
            },
            "orientation": {
                "pass": bool(ori_pass),
                "threshold_rad": float(orientation_tol_rad),
                "max_error_rad": worst_ori[0],
                "worst_frame": worst_ori[1],
                "worst_body": RUNTIME_BODY_NAMES[worst_ori[2]],
            },
            "racket_site_position_vs_schema": maximum_row(
                site_pos_error, "m", racket_position_tol_m
            ),
            "racket_site_normal_vs_schema": maximum_row(
                site_normal_error, "rad", racket_normal_tol_rad
            ),
            "racket_site_linear_velocity_vs_schema": maximum_row(
                site_lin_error,
                "m_s",
                racket_linear_velocity_tol_m_s,
            ),
            "racket_site_angular_velocity_vs_schema": maximum_row(
                site_ang_error,
                "rad_s",
                racket_angular_velocity_tol_rad_s,
            ),
            "racket_site_jacobian_vs_object_velocity": {
                "pass": jacobian_pass,
                "threshold_max_abs": float(racket_jacobian_tol),
                "linear_max_abs_error": float(np.max(jac_lin_error)),
                "linear_worst_frame": int(np.argmax(jac_lin_error)),
                "angular_max_abs_error": float(np.max(jac_ang_error)),
                "angular_worst_frame": int(np.argmax(jac_ang_error)),
                "root_twist_max_abs_error": float(
                    np.max(root_twist_error)
                ),
            },
            "racket_external_reference": {
                "enabled": racket_reference is not None,
                "pass": external_pass,
                "path": (
                    None
                    if racket_reference is None
                    else str(racket_reference.path)
                ),
                "max_errors": (
                    None
                    if external_errors is None
                    else {
                        key: float(np.max(value))
                        for key, value in external_errors.items()
                    }
                ),
            },
        },
        "racket": {
            "array_receipts": {
                name: {
                    "sha256": array_sha256(value),
                    "dtype": "<f8",
                    "shape": list(value.shape),
                }
                for name, value in racket_arrays
            },
            "trajectory_sha256": _combined_racket_sha256(racket_arrays),
            "peaks": {
                "site_position_norm_max_m": float(
                    np.max(np.linalg.norm(site_pos, axis=1))
                ),
                "site_normal_norm_error_max": float(
                    np.max(
                        np.abs(np.linalg.norm(site_normal, axis=1) - 1.0)
                    )
                ),
                "site_linear_speed_max_m_s": float(
                    np.linalg.norm(site_lin[lin_peak_frame])
                ),
                "site_linear_speed_peak_frame": lin_peak_frame,
                "site_angular_speed_max_rad_s": float(
                    np.linalg.norm(site_ang[ang_peak_frame])
                ),
                "site_angular_speed_peak_frame": ang_peak_frame,
            },
            "per_frame": [
                {
                    "frame": frame,
                    "time_s": frame / clip.fps,
                    "site_pos_w_m": site_pos[frame].tolist(),
                    "site_local_plus_y_normal_w": site_normal[frame].tolist(),
                    "site_lin_vel_w_m_s": site_lin[frame].tolist(),
                    "site_ang_vel_w_rad_s": site_ang[frame].tolist(),
                }
                for frame in range(clip.n_frames)
            ],
        },
        "per_body_max": {
            name: {
                "position_m": float(per_body_pos[index]),
                "orientation_rad": float(per_body_ori[index]),
            }
            for index, name in enumerate(RUNTIME_BODY_NAMES)
        },
    }


def play_viewer(
    clip: MotionClip,
    mujoco: Any,
    model: Any,
    data: Any,
    binding: ModelBinding,
    *,
    speed: float,
    loop: bool,
) -> None:
    """Passive kinematic viewer; the robot is teleported and forwarded per frame."""

    if not math.isfinite(float(speed)) or float(speed) <= 0.0:
        raise ValueError("viewer speed must be finite and positive")
    try:
        viewer_module = importlib.import_module("mujoco.viewer")
    except ImportError as exc:
        raise RuntimeError("mujoco.viewer is unavailable in this environment") from exc

    reset_kinematic_state(mujoco, model, data)
    frame_period = 1.0 / (clip.fps * float(speed))
    with viewer_module.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start_cycle = time.perf_counter()
            for frame in range(clip.n_frames):
                if not viewer.is_running():
                    return
                target = start_cycle + frame * frame_period
                remaining = target - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                apply_frame(
                    mujoco,
                    model,
                    data,
                    binding,
                    clip.body_pos_w[frame, 0],
                    clip.body_quat_w[frame, 0],
                    clip.joint_pos[frame],
                )
                apply_frame_velocity(
                    clip, frame, mujoco, model, data, binding
                )
                viewer.sync()
            if not loop:
                return


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dynamic_replay_command(
    motion_path: os.PathLike[str] | str,
    mjcf_path: os.PathLike[str] | str,
    report_path: os.PathLike[str] | str | None = None,
) -> str:
    """Suggest the existing L2 tool; never import, copy, or execute its logic."""

    motion = Path(motion_path).expanduser().resolve()
    mjcf = Path(mjcf_path).expanduser().resolve()
    output = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else motion.with_suffix(".dynamic_replay.json")
    )
    command: Sequence[str] = (
        sys.executable,
        str(_HERE / "motion_dynamic_replay.py"),
        "--motion",
        str(motion),
        "--mjcf",
        str(mjcf),
        "--out",
        str(output),
    )
    return shlex.join(command)


def write_report_no_clobber(path: os.PathLike[str] | str, report: dict[str, Any]) -> None:
    """Create one JSON report exclusively; an existing target is never overwritten."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Play an exact schema-2 A3 motion kinematically and verify stored "
            "32-body poses plus the right_racket site pose/twist against the "
            "vendor MJCF. This remains mj_forward-only kinematic evidence."
        )
    )
    parser.add_argument("--motion", required=True, help="exact schema-2 11/14-field motion NPZ")
    parser.add_argument(
        "--mjcf",
        default=str(DEFAULT_MJCF),
        help="AgiBot vendor a3_pingpong.xml (default: repository vendor model)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open passive kinematic viewer after the headless FK smoke check",
    )
    parser.add_argument("--loop", action="store_true", help="loop viewer playback")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="viewer wall-clock speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--position-tol-m",
        type=float,
        default=DEFAULT_POSITION_TOL_M,
        help="maximum allowed link-origin FK position error",
    )
    parser.add_argument(
        "--orientation-tol-rad",
        type=float,
        default=DEFAULT_ORIENTATION_TOL_RAD,
        help="maximum allowed shortest quaternion FK angle error",
    )
    parser.add_argument(
        "--racket-reference",
        help=(
            "optional exact racket-reference NPZ; validates site position, "
            "local +Y normal, and site point linear/angular velocity"
        ),
    )
    parser.add_argument(
        "--racket-position-tol-m",
        type=float,
        default=DEFAULT_RACKET_POSITION_TOL_M,
    )
    parser.add_argument(
        "--racket-normal-tol-rad",
        type=float,
        default=DEFAULT_RACKET_NORMAL_TOL_RAD,
    )
    parser.add_argument(
        "--racket-linear-velocity-tol-m-s",
        type=float,
        default=DEFAULT_RACKET_LINEAR_VELOCITY_TOL_M_S,
    )
    parser.add_argument(
        "--racket-angular-velocity-tol-rad-s",
        type=float,
        default=DEFAULT_RACKET_ANGULAR_VELOCITY_TOL_RAD_S,
    )
    parser.add_argument(
        "--racket-jacobian-tol",
        type=float,
        default=DEFAULT_RACKET_JACOBIAN_TOL,
        help="max component error between mj_jacSite and mj_objectVelocity",
    )
    parser.add_argument(
        "--report",
        help="optional JSON report; created no-clobber (the only file this tool writes)",
    )
    parser.add_argument(
        "--suggest-dynamic-replay",
        action="store_true",
        help="print, but do not execute, the existing motion_dynamic_replay.py command",
    )
    parser.add_argument(
        "--dynamic-report",
        help="report path used only in the printed dynamic-replay suggestion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        clip = load_motion(args.motion)
        racket_reference = (
            None
            if args.racket_reference is None
            else load_racket_reference(args.racket_reference, clip.n_frames)
        )
        mjcf_path = Path(args.mjcf).expanduser().resolve()
        if not mjcf_path.is_file():
            raise ValueError(f"MJCF not found: {mjcf_path}")
        mujoco = load_mujoco()
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        binding = bind_model(mujoco, model)
        report = smoke_check(
            clip,
            mujoco,
            model,
            data,
            binding,
            position_tol_m=args.position_tol_m,
            orientation_tol_rad=args.orientation_tol_rad,
            racket_reference=racket_reference,
            racket_position_tol_m=args.racket_position_tol_m,
            racket_normal_tol_rad=args.racket_normal_tol_rad,
            racket_linear_velocity_tol_m_s=(
                args.racket_linear_velocity_tol_m_s
            ),
            racket_angular_velocity_tol_rad_s=(
                args.racket_angular_velocity_tol_rad_s
            ),
            racket_jacobian_tol=args.racket_jacobian_tol,
        )
        report["artifacts"] = {
            "motion_sha256": sha256_file(clip.path),
            "mjcf_path": str(mjcf_path),
            "mjcf_sha256": sha256_file(mjcf_path),
            "racket_reference_path": (
                None
                if racket_reference is None
                else str(racket_reference.path)
            ),
            "racket_reference_sha256": (
                None
                if racket_reference is None
                else sha256_file(racket_reference.path)
            ),
        }
        if args.suggest_dynamic_replay:
            suggestion = dynamic_replay_command(
                clip.path,
                mjcf_path,
                report_path=args.dynamic_report,
            )
            report["next_step_dynamic_replay_command"] = suggestion
            print(
                "[kinematic-player] optional separate dynamics check "
                "(not executed):\n" + suggestion
            )
        if args.report:
            write_report_no_clobber(args.report, report)

        pos = report["gates"]["position"]
        ori = report["gates"]["orientation"]
        racket_peaks = report["racket"]["peaks"]
        print(
            "[kinematic-player] "
            f"{report['verdict']}: {clip.n_frames} frames @ {clip.fps:g} Hz; "
            f"FK max pos={pos['max_error_m']:.6g} m "
            f"({pos['worst_body']} f{pos['worst_frame']}), "
            f"ori={ori['max_error_rad']:.6g} rad "
            f"({ori['worst_body']} f{ori['worst_frame']}); "
            f"racket |v|max={racket_peaks['site_linear_speed_max_m_s']:.6g} m/s, "
            f"|w|max={racket_peaks['site_angular_speed_max_rad_s']:.6g} rad/s; "
            "mj_step_calls=0 (kinematic evidence only)"
        )
        if args.viewer:
            play_viewer(
                clip,
                mujoco,
                model,
                data,
                binding,
                speed=args.speed,
                loop=args.loop,
            )
        return 0 if report["verdict"] == PASS else 2
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"[kinematic-player] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
