#!/usr/bin/env python3
"""Plant-bound MuJoCo inverse-dynamics and geometry screen for canonical motions.

This module is deliberately a *screen*, not a controller or balance certificate.
It evaluates a prescribed trajectory in one explicitly named MuJoCo plant and
reports:

* runtime-order joint position and velocity limit utilization;
* contact-free ``mj_inverse`` generalized joint effort and direct-motor force
  utilization, frame by frame;
* the unactuated floating-root residual left by that inverse-dynamics query;
* floor penetration, non-foot floor contact, and robot self-collision;
* root and whole-robot centre-of-mass (CoM) motion diagnostics.

The important fail-closed boundary is the floating base.  For a grounded robot,

    M(q) qdd + h(q, qd) = S^T tau + J_contact^T lambda

does not determine ``tau`` until a contact-wrench distribution ``lambda`` is
specified.  Reading the 31 joint rows of contact-disabled ``qfrc_inverse`` and
calling them "the actuator torque" would silently choose a fictitious external
wrench at the free root.  This tool therefore labels those rows as a
``joint_effort_proxy`` and certifies them as actuator torque only when:

1. the exact plant has a one-to-one direct motor for every runtime joint;
2. both the source-XML hash and compiled-model signature are explicitly bound;
3. no contact is geometrically active on any frame; and
4. every unactuated root residual is numerically zero.

Grounded clips consequently return ``INCOMPLETE_FAIL_CLOSED`` for the torque
certificate while still returning useful plant-specific limit, collision,
root-residual, and effort-proxy diagnostics.  A downstream constrained-contact
inverse dynamics solver, PD replay, policy replay, and behaviour gate remain
mandatory.  This module never steps training, writes a motion asset, or commands
hardware.

Array API
---------
``analyze_trajectory`` consumes:

* ``joint_pos`` / ``joint_vel``: ``(T, 31)`` in
  ``audit_motion_npz.ISAAC_JOINT_NAMES`` order;
* ``root_pos_w`` / ``root_quat_wxyz``: free-joint origin pose;
* ``root_lin_vel_w``: free-joint origin linear velocity in world coordinates;
* ``root_ang_vel_w``: root angular velocity in world coordinates;
* scalar ``fps``.

The NPZ adapter accepts exact schema-2 motion files.  Schema-2 stores link-origin
positions but rigid-body-CoM linear velocities.  For the pelvis row the adapter
converts the stored pelvis-CoM velocity back to free-joint-origin velocity using
the exact MJCF inertial offset before calling the array API.

CLI
---
The report is JSON on stdout unless ``--out`` is supplied.  The latter writes
only a report, never a motion asset.

    python canonical_mujoco_dynamics_gate.py \
      --motion candidate.npz --mjcf /exact/a3_pingpong.xml \
      --urdf /exact/URDF-JOINT-LINK.urdf \
      [--expected-mjcf-sha256 HEX] [--expected-compiled-signature HEX] \
      [--out report.json]

Exit status is 0 only for a complete PASS, and 2 for input/model rejection,
limit/collision failure, or an intentionally incomplete floating-base torque
interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# After the sibling directory, never before it, so sibling contracts win any name clash.
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))
from audit_motion_npz import ISAAC_JOINT_NAMES, parse_urdf_limits  # noqa: E402

import mujoco_table_scene  # noqa: E402  (pure Python; imports MuJoCo nowhere)

try:  # pure helpers and tests remain importable without MuJoCo
    import mujoco
except ImportError:  # pragma: no cover - environment-dependent
    mujoco = None


RUNTIME_JOINT_NAMES: Tuple[str, ...] = tuple(ISAAC_JOINT_NAMES)
ROOT_BODY = "pelvis_link"
FLOOR_GEOM = "floor"
FOOT_BODIES: Tuple[str, str] = ("left_ankle_roll_Link", "right_ankle_roll_Link")
DEFAULT_MJCF_REL = (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
DEFAULT_URDF_REL = (
    "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"
)

REQUIRED_SCHEMA2_FIELDS = frozenset(
    {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "kinematics_schema_version",
        "body_pos_point",
        "body_lin_vel_point",
    }
)
MIGRATION_SCHEMA2_FIELDS = frozenset(
    {
        "kinematics_migration_source_sha256",
        "kinematics_migration_source_point",
        "kinematics_migration_tool",
    }
)
ALLOWED_SCHEMA2_FIELDSETS = (
    REQUIRED_SCHEMA2_FIELDS,
    REQUIRED_SCHEMA2_FIELDS | MIGRATION_SCHEMA2_FIELDS,
)
SCHEMA_FK_POSITION_TOL_M = 1.0e-4
SCHEMA_FK_ORIENTATION_TOL_RAD = 1.0e-4
SCHEMA_BODY_LINEAR_VELOCITY_TOL_M_S = 1.0e-3
SCHEMA_BODY_ANGULAR_VELOCITY_TOL_RAD_S = 1.0e-3
SCHEMA_JOINT_VELOCITY_TOL_RAD_S = 1.0e-3
SCHEMA_ROOT_LINEAR_VELOCITY_TOL_M_S = 1.0e-3
SCHEMA_ROOT_ANGULAR_VELOCITY_TOL_RAD_S = 1.0e-3
SCHEMA_VELOCITY_JACOBIAN_CONDITION_MAX = 1.0e8


class DynamicsGateError(ValueError):
    """Fail-closed input or plant-contract error."""


@dataclass(frozen=True)
class GateThresholds:
    """Numerical thresholds for a diagnostic screen, not learned acceptance bars."""

    position_tolerance_rad: float = 1.0e-6
    velocity_tolerance_fraction: float = 1.0e-9
    stored_joint_velocity_tolerance_rad_s: float = 5.0e-2
    stored_root_linear_velocity_tolerance_m_s: float = 5.0e-2
    stored_root_angular_velocity_tolerance_rad_s: float = 5.0e-2
    self_penetration_tolerance_m: float = 1.0e-4
    foot_floor_penetration_tolerance_m: float = 2.0e-3
    nonfoot_floor_penetration_tolerance_m: float = 1.0e-4
    root_force_zero_tolerance_n: float = 1.0e-4
    root_torque_zero_tolerance_nm: float = 1.0e-4

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise DynamicsGateError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class MotionTrajectory:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    fps: float
    source: str = "array-api"
    schema2_body_validation: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        arrays = {
            "joint_pos": (self.joint_pos, 2, len(RUNTIME_JOINT_NAMES)),
            "joint_vel": (self.joint_vel, 2, len(RUNTIME_JOINT_NAMES)),
            "root_pos_w": (self.root_pos_w, 2, 3),
            "root_quat_wxyz": (self.root_quat_wxyz, 2, 4),
            "root_lin_vel_w": (self.root_lin_vel_w, 2, 3),
            "root_ang_vel_w": (self.root_ang_vel_w, 2, 3),
        }
        lengths: List[int] = []
        for name, (raw, ndim, width) in arrays.items():
            arr = np.asarray(raw)
            if arr.ndim != ndim or arr.shape[1] != width:
                raise DynamicsGateError(
                    f"{name} shape {arr.shape}, expected (T, {width})"
                )
            if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
                raise DynamicsGateError(f"{name} must contain only finite numbers")
            lengths.append(int(arr.shape[0]))
        if len(set(lengths)) != 1:
            raise DynamicsGateError(f"trajectory arrays disagree on T: {lengths}")
        if lengths[0] < 3:
            raise DynamicsGateError("at least 3 frames are required for acceleration")
        if not math.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
            raise DynamicsGateError("fps must be finite and strictly positive")
        qnorm = np.linalg.norm(np.asarray(self.root_quat_wxyz, float), axis=1)
        if np.any(qnorm < 1.0e-12):
            raise DynamicsGateError("root quaternion contains a zero-norm row")
        if np.max(np.abs(qnorm - 1.0)) > 1.0e-3:
            raise DynamicsGateError(
                "root quaternions must already be normalized within 1e-3"
            )

    @property
    def frames(self) -> int:
        return int(np.asarray(self.joint_pos).shape[0])


@dataclass(frozen=True)
class PlantModel:
    contact_model: Any
    inverse_model: Any
    mjcf_path: str
    mjcf_sha256: str
    urdf_path: Optional[str]
    urdf_sha256: Optional[str]
    compiled_signature_sha256: str
    root_body_id: int
    floor_geom_id: int
    foot_body_ids: Tuple[int, int]
    runtime_body_names: Tuple[str, ...]
    runtime_body_ids: np.ndarray
    joint_ids: np.ndarray
    joint_qposadr: np.ndarray
    joint_dofadr: np.ndarray
    actuator_ids: np.ndarray
    motor_gear: np.ndarray
    joint_effort_limits: np.ndarray
    actuator_force_limits: np.ndarray
    velocity_limits: np.ndarray
    position_lower: np.ndarray
    position_upper: np.ndarray
    root_body_ipos: np.ndarray
    weight_n: float
    direct_actuation_verified: bool
    actuation_mapping_verified: bool
    identity_bound: bool
    identity_expectations: Mapping[str, Optional[str]]
    #: Set only by ``--with-table``.  When present, ``contact_model`` is the in-memory
    #: table-augmented model and the geometry check's existing ``other_world`` bucket picks up
    #: robot-vs-table overlap on its own.  ``inverse_model`` is always the canonical model, and
    #: ``compiled_signature_sha256`` is always the canonical signature, so the identity pin keeps
    #: meaning exactly what it meant before.
    table_scene: Any = None

    @property
    def nq(self) -> int:
        return int(self.contact_model.nq)

    @property
    def nv(self) -> int:
        return int(self.contact_model.nv)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_array(h: "hashlib._Hash", label: str, value: Any) -> None:
    arr = np.ascontiguousarray(np.asarray(value))
    h.update(label.encode("utf-8"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(np.asarray(arr.shape, np.int64).tobytes())
    h.update(arr.tobytes())


def compiled_model_signature(model: Any) -> str:
    """Content signature of the compiled dynamics/collision arrays used here.

    This is not a portable MJB byte hash.  It is a deterministic receipt over
    the compiled arrays that materially affect this screen, including loaded
    mesh vertices/faces.  Production manifests may bind this receipt in
    addition to the source XML hash.
    """

    h = hashlib.sha256()
    for name in (
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_limited",
        "jnt_range",
        "jnt_actfrclimited",
        "jnt_actfrcrange",
        "dof_damping",
        "dof_frictionloss",
        "dof_armature",
        "body_parentid",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_inertia",
        "geom_type",
        "geom_bodyid",
        "geom_contype",
        "geom_conaffinity",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "mesh_vert",
        "mesh_face",
        "actuator_trntype",
        "actuator_trnid",
        "actuator_gear",
        "actuator_ctrllimited",
        "actuator_ctrlrange",
        "actuator_forcelimited",
        "actuator_forcerange",
    ):
        if hasattr(model, name):
            _hash_array(h, name, getattr(model, name))
    _hash_array(h, "gravity", model.opt.gravity)
    _hash_array(h, "timestep", [model.opt.timestep])
    for obj, count, enum in (
        ("body", int(model.nbody), mujoco.mjtObj.mjOBJ_BODY),
        ("joint", int(model.njnt), mujoco.mjtObj.mjOBJ_JOINT),
        ("geom", int(model.ngeom), mujoco.mjtObj.mjOBJ_GEOM),
        ("actuator", int(model.nu), mujoco.mjtObj.mjOBJ_ACTUATOR),
    ):
        h.update(obj.encode("ascii"))
        for index in range(count):
            name = mujoco.mj_id2name(model, enum, index) or ""
            h.update(name.encode("utf-8"))
            h.update(b"\0")
    return h.hexdigest()


def _urdf_contract(urdf_path: os.PathLike[str] | str) -> Tuple[np.ndarray, np.ndarray]:
    """Return effort and velocity vectors from the exact A3 URDF."""

    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise DynamicsGateError(f"URDF does not exist: {path}")
    try:
        limits = parse_urdf_limits(str(path))
    except Exception as exc:
        raise DynamicsGateError(f"cannot parse A3 URDF limits: {exc}") from exc
    missing = [name for name in RUNTIME_JOINT_NAMES if name not in limits]
    if missing:
        raise DynamicsGateError(f"URDF misses runtime joints: {missing}")
    effort = np.asarray([limits[name].effort for name in RUNTIME_JOINT_NAMES], np.float64)
    velocity = np.asarray([limits[name].velocity for name in RUNTIME_JOINT_NAMES], np.float64)
    if (
        effort.shape != (len(RUNTIME_JOINT_NAMES),)
        or velocity.shape != effort.shape
        or not np.isfinite(effort).all()
        or not np.isfinite(velocity).all()
        or np.any(effort <= 0.0)
        or np.any(velocity <= 0.0)
    ):
        raise DynamicsGateError("vendor effort/velocity limit vectors are invalid")
    return effort, velocity


def _name2id(model: Any, obj: Any, name: str, label: str) -> int:
    index = int(mujoco.mj_name2id(model, obj, name))
    if index < 0:
        raise DynamicsGateError(f"{label} {name!r} is missing from MJCF")
    return index


def _descends_from(model: Any, body_id: int, ancestor_id: int) -> bool:
    cur = int(body_id)
    while cur > 0:
        if cur == ancestor_id:
            return True
        cur = int(model.body_parentid[cur])
    return ancestor_id == 0 and cur == 0


def _runtime_body_contract() -> Tuple[str, ...]:
    path = _repo_root() / "configs/a3_runtime_body_order.txt"
    if not path.is_file():
        raise DynamicsGateError(f"runtime body-order contract does not exist: {path}")
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(names) != 32 or len(set(names)) != 32 or names[0] != ROOT_BODY:
        raise DynamicsGateError(
            "runtime body-order contract must contain 32 unique bodies with pelvis_link first"
        )
    return names


def load_plant(
    mjcf_path: os.PathLike[str] | str,
    *,
    expected_mjcf_sha256: Optional[str] = None,
    expected_compiled_signature: Optional[str] = None,
    urdf_path: Optional[os.PathLike[str] | str] = None,
    velocity_limits: Optional[Sequence[float]] = None,
    expected_effort_limits: Optional[Sequence[float]] = None,
    with_table: bool = False,
) -> PlantModel:
    """Load and validate the exact floating-base, direct-motor A3 plant.

    ``with_table`` (``--with-table``, OFF by default) appends the ping-pong table to an IN-MEMORY
    copy of the MJCF as a colliding obstacle; the file on disk is byte-pinned and never written.
    Only ``contact_model`` gains the table.  ``inverse_model`` has contacts disabled anyway, and
    the compiled-signature identity receipt is taken from the CANONICAL model before the swap, so
    a pinned ``--expected-compiled-signature`` still verifies the same thing it always verified.
    """

    if mujoco is None:
        raise DynamicsGateError("mujoco is not installed")
    path = Path(mjcf_path).expanduser().resolve()
    if not path.is_file():
        raise DynamicsGateError(f"MJCF does not exist: {path}")
    source_sha = _sha256_file(path)
    if expected_mjcf_sha256 is not None:
        expected = str(expected_mjcf_sha256).lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise DynamicsGateError("expected_mjcf_sha256 must be 64 lowercase hex digits")
        if source_sha != expected:
            raise DynamicsGateError(
                f"MJCF SHA mismatch: expected {expected}, got {source_sha}"
            )

    contact_model = mujoco.MjModel.from_xml_path(str(path))
    inverse_model = mujoco.MjModel.from_xml_path(str(path))
    # Identity receipt is taken from the canonical model, BEFORE any table is attached.
    signature = compiled_model_signature(contact_model)
    table_scene = None
    if with_table:
        table_scene = mujoco_table_scene.load_table_scene(mujoco, path, collidable=True)
        contact_model = table_scene.model
    if expected_compiled_signature is not None:
        expected_sig = str(expected_compiled_signature).lower()
        if len(expected_sig) != 64 or any(c not in "0123456789abcdef" for c in expected_sig):
            raise DynamicsGateError(
                "expected_compiled_signature must be 64 lowercase hex digits"
            )
        if signature != expected_sig:
            raise DynamicsGateError(
                "compiled model signature mismatch: "
                f"expected {expected_sig}, got {signature}"
            )

    if int(contact_model.njnt) != len(RUNTIME_JOINT_NAMES) + 1:
        raise DynamicsGateError(
            f"expected one free joint plus 31 runtime joints, got njnt={contact_model.njnt}"
        )
    if int(contact_model.nv) != 6 + len(RUNTIME_JOINT_NAMES):
        raise DynamicsGateError(
            f"expected nv=37 (free root + 31 hinges), got {contact_model.nv}"
        )
    if int(contact_model.nu) != len(RUNTIME_JOINT_NAMES):
        raise DynamicsGateError(f"expected 31 motors, got nu={contact_model.nu}")
    free_ids = np.flatnonzero(
        np.asarray(contact_model.jnt_type)
        == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if free_ids.tolist() != [0]:
        raise DynamicsGateError(f"expected exactly joint 0 to be FREE, got {free_ids.tolist()}")
    if int(contact_model.jnt_qposadr[0]) != 0 or int(contact_model.jnt_dofadr[0]) != 0:
        raise DynamicsGateError("free root must occupy qpos[0:7] and qvel[0:6]")

    root_body_id = _name2id(
        contact_model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY, "root body"
    )
    if int(contact_model.jnt_bodyid[0]) != root_body_id:
        raise DynamicsGateError("free joint is not attached to pelvis_link")
    floor_geom_id = _name2id(
        contact_model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM, "floor geom"
    )
    if int(contact_model.geom_bodyid[floor_geom_id]) != 0:
        raise DynamicsGateError("floor geom is not attached to the world")
    foot_body_ids = tuple(
        _name2id(contact_model, mujoco.mjtObj.mjOBJ_BODY, n, "foot body")
        for n in FOOT_BODIES
    )
    runtime_body_names = _runtime_body_contract()
    runtime_body_ids = np.asarray(
        [
            _name2id(
                contact_model, mujoco.mjtObj.mjOBJ_BODY, name, "runtime body"
            )
            for name in runtime_body_names
        ],
        np.int64,
    )
    if len(set(runtime_body_ids.tolist())) != 32:
        raise DynamicsGateError("runtime body names do not resolve bijectively")
    if any(
        not _descends_from(contact_model, int(body_id), root_body_id)
        for body_id in runtime_body_ids
    ):
        raise DynamicsGateError("runtime body contract escapes the pelvis subtree")

    joint_ids: List[int] = []
    qposadr: List[int] = []
    dofadr: List[int] = []
    lower: List[float] = []
    upper: List[float] = []
    joint_effort: List[float] = []
    actuator_ids: List[int] = []
    gears: List[float] = []
    actuator_force_limits: List[float] = []

    for name in RUNTIME_JOINT_NAMES:
        jid = _name2id(contact_model, mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
        if int(contact_model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise DynamicsGateError(f"joint {name!r} is not a hinge")
        if not bool(contact_model.jnt_limited[jid]):
            raise DynamicsGateError(f"joint {name!r} has no position limit")
        lo, hi = map(float, contact_model.jnt_range[jid])
        if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
            raise DynamicsGateError(f"joint {name!r} has invalid range [{lo}, {hi}]")
        if not bool(contact_model.jnt_actfrclimited[jid]):
            raise DynamicsGateError(f"joint {name!r} has no generalized effort limit")
        elo, ehi = map(float, contact_model.jnt_actfrcrange[jid])
        if not math.isclose(abs(elo), abs(ehi), rel_tol=0.0, abs_tol=1.0e-9):
            raise DynamicsGateError(f"joint {name!r} effort range is not symmetric")

        matches = [
            aid
            for aid in range(int(contact_model.nu))
            if int(contact_model.actuator_trnid[aid, 0]) == jid
        ]
        if len(matches) != 1:
            raise DynamicsGateError(
                f"joint {name!r} must have exactly one actuator, got {matches}"
            )
        aid = matches[0]
        if int(contact_model.actuator_trntype[aid]) != int(
            mujoco.mjtTrn.mjTRN_JOINT
        ):
            raise DynamicsGateError(f"actuator for {name!r} is not a joint transmission")
        if int(contact_model.actuator_dyntype[aid]) != int(
            mujoco.mjtDyn.mjDYN_NONE
        ):
            raise DynamicsGateError(f"actuator for {name!r} has non-trivial dynamics")
        if int(contact_model.actuator_gaintype[aid]) != int(
            mujoco.mjtGain.mjGAIN_FIXED
        ):
            raise DynamicsGateError(f"actuator for {name!r} has non-fixed gain")
        if int(contact_model.actuator_biastype[aid]) != int(
            mujoco.mjtBias.mjBIAS_NONE
        ):
            raise DynamicsGateError(f"actuator for {name!r} has non-zero bias model")
        gain = np.asarray(contact_model.actuator_gainprm[aid], np.float64)
        bias = np.asarray(contact_model.actuator_biasprm[aid], np.float64)
        if not math.isclose(float(gain[0]), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise DynamicsGateError(f"actuator for {name!r} does not have unit gain")
        if np.max(np.abs(gain[1:])) > 1.0e-12 or np.max(np.abs(bias)) > 1.0e-12:
            raise DynamicsGateError(
                f"actuator for {name!r} has hidden gain/bias parameters"
            )
        gear_row = np.asarray(contact_model.actuator_gear[aid], float)
        if np.max(np.abs(gear_row[1:])) > 1.0e-12 or abs(gear_row[0]) < 1.0e-12:
            raise DynamicsGateError(
                f"actuator for {name!r} is not a scalar direct joint motor"
            )
        gear = float(gear_row[0])

        force_cap: Optional[float] = None
        if bool(contact_model.actuator_forcelimited[aid]):
            flo, fhi = map(float, contact_model.actuator_forcerange[aid])
            if not math.isclose(abs(flo), abs(fhi), rel_tol=0.0, abs_tol=1.0e-9):
                raise DynamicsGateError(f"actuator {aid} force range is not symmetric")
            force_cap = abs(fhi)
        if bool(contact_model.actuator_ctrllimited[aid]):
            clo, chi = map(float, contact_model.actuator_ctrlrange[aid])
            if not math.isclose(abs(clo), abs(chi), rel_tol=0.0, abs_tol=1.0e-9):
                raise DynamicsGateError(f"actuator {aid} control range is not symmetric")
            ctrl_force_cap = abs(chi)
            force_cap = ctrl_force_cap if force_cap is None else min(force_cap, ctrl_force_cap)
        if force_cap is None:
            # Joint-level actuator-force clipping still supplies a valid
            # generalized limit.  Convert it through the scalar gear.
            force_cap = abs(ehi) / abs(gear)
        if not math.isclose(
            force_cap * abs(gear), abs(ehi), rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise DynamicsGateError(
                f"actuator/joint effort limits disagree for {name!r}: "
                f"|gear|*force={force_cap * abs(gear):.9g}, joint={abs(ehi):.9g}"
            )

        joint_ids.append(jid)
        qposadr.append(int(contact_model.jnt_qposadr[jid]))
        dofadr.append(int(contact_model.jnt_dofadr[jid]))
        lower.append(lo)
        upper.append(hi)
        joint_effort.append(abs(ehi))
        actuator_ids.append(aid)
        gears.append(gear)
        actuator_force_limits.append(force_cap)

    if sorted(qposadr) != list(range(7, 7 + len(RUNTIME_JOINT_NAMES))):
        raise DynamicsGateError("runtime joints do not cover every post-root qpos column once")
    if sorted(dofadr) != list(range(6, 6 + len(RUNTIME_JOINT_NAMES))):
        raise DynamicsGateError("runtime joints do not cover every actuated DoF once")
    if len(set(actuator_ids)) != len(RUNTIME_JOINT_NAMES):
        raise DynamicsGateError("actuator mapping is not one-to-one")

    resolved_urdf: Optional[Path] = None
    urdf_sha: Optional[str] = None
    if expected_effort_limits is None or velocity_limits is None:
        resolved_urdf = (
            Path(urdf_path).expanduser().resolve()
            if urdf_path is not None
            else resolve_default_urdf().resolve()
        )
        urdf_effort, urdf_velocity = _urdf_contract(resolved_urdf)
        urdf_sha = _sha256_file(resolved_urdf)
    else:
        urdf_effort = np.asarray(expected_effort_limits, np.float64)
        urdf_velocity = np.asarray(velocity_limits, np.float64)
    effort_expected = (
        urdf_effort
        if expected_effort_limits is None
        else np.asarray(expected_effort_limits, np.float64)
    )
    velocity = (
        urdf_velocity
        if velocity_limits is None
        else np.asarray(velocity_limits, np.float64)
    )
    if effort_expected.shape != (len(RUNTIME_JOINT_NAMES),):
        raise DynamicsGateError("expected_effort_limits must contain 31 values")
    if velocity.shape != (len(RUNTIME_JOINT_NAMES),):
        raise DynamicsGateError("velocity_limits must contain 31 values")
    if (
        not np.isfinite(effort_expected).all()
        or not np.isfinite(velocity).all()
        or np.any(effort_expected <= 0)
        or np.any(velocity <= 0)
    ):
        raise DynamicsGateError("effort and velocity limits must be finite and positive")
    if not np.allclose(
        np.asarray(joint_effort), effort_expected, rtol=0.0, atol=1.0e-6
    ):
        bad = int(
            np.argmax(np.abs(np.asarray(joint_effort) - effort_expected))
        )
        raise DynamicsGateError(
            "MJCF effort limit disagrees with the expected vendor contract at "
            f"{RUNTIME_JOINT_NAMES[bad]!r}"
        )

    # The inverse pass must not hide force in contact or position-limit
    # constraints.  Passive damping/friction/armature intentionally remain on.
    inverse_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    inverse_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)

    weight_n = float(contact_model.body_subtreemass[root_body_id]) * float(
        np.linalg.norm(contact_model.opt.gravity)
    )
    if not math.isfinite(weight_n) or weight_n <= 0.0:
        raise DynamicsGateError("robot subtree has invalid weight")

    identity_bound = bool(
        expected_mjcf_sha256 is not None
        and expected_compiled_signature is not None
    )
    return PlantModel(
        contact_model=contact_model,
        inverse_model=inverse_model,
        mjcf_path=str(path),
        mjcf_sha256=source_sha,
        urdf_path=None if resolved_urdf is None else str(resolved_urdf),
        urdf_sha256=urdf_sha,
        compiled_signature_sha256=signature,
        root_body_id=root_body_id,
        floor_geom_id=floor_geom_id,
        foot_body_ids=(int(foot_body_ids[0]), int(foot_body_ids[1])),
        runtime_body_names=runtime_body_names,
        runtime_body_ids=runtime_body_ids,
        joint_ids=np.asarray(joint_ids, np.int64),
        joint_qposadr=np.asarray(qposadr, np.int64),
        joint_dofadr=np.asarray(dofadr, np.int64),
        actuator_ids=np.asarray(actuator_ids, np.int64),
        motor_gear=np.asarray(gears, np.float64),
        joint_effort_limits=np.asarray(joint_effort, np.float64),
        actuator_force_limits=np.asarray(actuator_force_limits, np.float64),
        velocity_limits=np.asarray(velocity, np.float64),
        position_lower=np.asarray(lower, np.float64),
        position_upper=np.asarray(upper, np.float64),
        root_body_ipos=np.asarray(contact_model.body_ipos[root_body_id], np.float64),
        weight_n=weight_n,
        direct_actuation_verified=identity_bound,
        actuation_mapping_verified=True,
        identity_bound=identity_bound,
        identity_expectations={
            "source_xml": expected_mjcf_sha256,
            "compiled_signature": expected_compiled_signature,
        },
        table_scene=table_scene,
    )


def _text_scalar(value: Any, label: str) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise DynamicsGateError(f"{label} must be a scalar string")
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _float_scalar(value: Any, label: str) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise DynamicsGateError(f"{label} must be scalar")
    out = float(arr.reshape(-1)[0])
    if not math.isfinite(out):
        raise DynamicsGateError(f"{label} must be finite")
    return out


def _quat_to_mat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Pure NumPy wxyz quaternion to rotation matrix."""

    q = np.asarray(quat, np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = map(float, q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def com_velocity_to_origin_velocity(
    com_velocity_w: np.ndarray,
    angular_velocity_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    body_ipos: np.ndarray,
) -> np.ndarray:
    """Convert rigid-body CoM velocity to body/free-joint-origin velocity."""

    v_com = np.asarray(com_velocity_w, np.float64)
    omega = np.asarray(angular_velocity_w, np.float64)
    quat = np.asarray(root_quat_wxyz, np.float64)
    ipos = np.asarray(body_ipos, np.float64)
    if v_com.shape != omega.shape or v_com.ndim != 2 or v_com.shape[1] != 3:
        raise DynamicsGateError("CoM and angular velocity must both have shape (T, 3)")
    if quat.shape != (v_com.shape[0], 4) or ipos.shape != (3,):
        raise DynamicsGateError("quaternion/body_ipos shape mismatch")
    result = np.empty_like(v_com)
    for frame in range(v_com.shape[0]):
        offset_w = _quat_to_mat_wxyz(quat[frame]) @ ipos
        result[frame] = v_com[frame] - np.cross(omega[frame], offset_w)
    return result


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a = np.asarray(lhs, np.float64)
    b = np.asarray(rhs, np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0.0, -q, q)
    vector_norm = np.linalg.norm(q[..., 1:], axis=-1)
    half_angle = np.arctan2(vector_norm, q[..., 0])
    angle = 2.0 * half_angle
    scale = np.where(vector_norm > 1.0e-10, angle / vector_norm, 2.0)
    return q[..., 1:] * scale[..., None]


def _so3_derivative_w(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Central angular velocity in world coordinates, matching schema producers."""

    quat = np.asarray(quat_wxyz, np.float64)
    if quat.ndim != 3 or quat.shape[2] != 4 or quat.shape[0] < 3:
        raise DynamicsGateError("body quaternion path must have shape (T>=3, B, 4)")
    normalized = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    rel = _quat_multiply(normalized[2:], _quat_conjugate(normalized[:-2]))
    middle = _axis_angle_from_quaternion(rel) / (2.0 * dt)
    return np.concatenate((middle[:1], middle, middle[-1:]), axis=0)


def _quaternion_angle_error(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a = np.asarray(lhs, np.float64).copy()
    b = np.asarray(rhs, np.float64).copy()
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _validate_schema2_body_channels(
    plant: PlantModel,
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    stored_body_pos: np.ndarray,
    stored_body_quat: np.ndarray,
    stored_body_lin: np.ndarray,
    stored_body_ang: np.ndarray,
    fps: float,
) -> Dict[str, float]:
    """Rebuild all 32 body pose/velocity channels from the exact plant.

    Schema-2 velocities are instantaneous analytic channels.  They are not
    finite differences of the 50 Hz pose samples.  The strict check therefore
    uses two independent MuJoCo identities at every frame:

    * the stored joint/root velocity defines one generalized velocity, whose
      32 body velocities are recomputed with ``mj_objectVelocity`` in world
      orientation; and
    * all 32 stored body COM/angular velocities are stacked into an
      overdetermined Jacobian system and solved back to generalized velocity.

    The second direction catches a forged ``joint_vel`` even when pose/FK bytes
    remain valid.  A coarse pose finite difference is retained later as a
    diagnostic only; treating it as the instantaneous truth rejects valid
    high-acceleration samples at bang-bang switches.
    """

    arrays = (stored_body_pos, stored_body_quat, stored_body_lin, stored_body_ang)
    if not all(np.isfinite(np.asarray(value)).all() for value in arrays):
        raise DynamicsGateError("schema-2 body channels contain NaN/Inf")
    if joint_vel.shape != joint_pos.shape or not np.isfinite(joint_vel).all():
        raise DynamicsGateError("schema-2 joint_vel must be finite and match joint_pos")
    quat_norm_error = float(
        np.max(np.abs(np.linalg.norm(stored_body_quat, axis=-1) - 1.0))
    )
    if quat_norm_error > 2.0e-3:
        raise DynamicsGateError(
            f"body_quat_w max norm error {quat_norm_error:.6g} exceeds 2e-3"
        )

    frames = int(joint_pos.shape[0])
    model = plant.contact_model
    data = mujoco.MjData(model)
    fk_pos = np.zeros_like(stored_body_pos, dtype=np.float64)
    fk_quat = np.zeros_like(stored_body_quat, dtype=np.float64)
    analytic_body_lin = np.zeros_like(stored_body_lin, dtype=np.float64)
    analytic_body_ang = np.zeros_like(stored_body_ang, dtype=np.float64)
    normalized_root = _normalize_quaternions(root_quat)
    joint_velocity_mismatch = 0.0
    root_linear_velocity_mismatch = 0.0
    root_angular_velocity_mismatch = 0.0
    velocity_jacobian_min_rank = int(model.nv)
    velocity_jacobian_max_condition = 0.0
    inverse_body_velocity_residual = 0.0
    for frame in range(frames):
        data.qpos[:] = 0.0
        data.qpos[0:3] = root_pos[frame]
        data.qpos[3:7] = normalized_root[frame]
        data.qpos[plant.joint_qposadr] = joint_pos[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        fk_pos[frame] = np.asarray(data.xpos, np.float64)[plant.runtime_body_ids]
        fk_quat[frame] = np.asarray(data.xquat, np.float64)[plant.runtime_body_ids]

        # Recover the root link-origin world twist from the root COM channel.
        # xipos-xpos is the exact compiled-model COM offset in world axes; this
        # avoids assuming that the free-joint quaternion equals a body's local
        # inertial-frame orientation.
        root_offset_w = (
            np.asarray(data.xipos[plant.root_body_id], np.float64)
            - np.asarray(data.xpos[plant.root_body_id], np.float64)
        )
        root_origin_linear_w = stored_body_lin[frame, 0] - np.cross(
            stored_body_ang[frame, 0], root_offset_w
        )

        expected_qvel = np.zeros(int(model.nv), dtype=np.float64)
        expected_qvel[plant.joint_dofadr] = joint_vel[frame]
        root_jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
        root_jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
        mujoco.mj_jacBody(
            model,
            data,
            root_jacp,
            root_jacr,
            int(plant.root_body_id),
        )
        root_slice = slice(0, 6)
        root_matrix = np.vstack(
            (root_jacp[:, root_slice], root_jacr[:, root_slice])
        )
        root_condition = float(np.linalg.cond(root_matrix))
        if (
            not np.isfinite(root_condition)
            or root_condition > SCHEMA_VELOCITY_JACOBIAN_CONDITION_MAX
        ):
            raise DynamicsGateError(
                "schema-2 root twist Jacobian is singular or ill-conditioned: "
                f"{root_condition:.9g}"
            )
        nonroot_contribution = np.concatenate(
            (root_jacp @ expected_qvel, root_jacr @ expected_qvel)
        )
        expected_qvel[root_slice] = np.linalg.solve(
            root_matrix,
            np.concatenate(
                (root_origin_linear_w, stored_body_ang[frame, 0])
            )
            - nonroot_contribution,
        )

        # Forward direction: mj_objectVelocity returns [angular, linear] at the
        # body COM.  flg_local=0 keeps both vectors in world orientation.
        data.qvel[:] = expected_qvel
        mujoco.mj_forward(model, data)
        velocity_jacobian = np.empty(
            (6 * len(plant.runtime_body_ids), int(model.nv)), dtype=np.float64
        )
        velocity_rhs = np.empty(
            6 * len(plant.runtime_body_ids), dtype=np.float64
        )
        for column, body_id in enumerate(plant.runtime_body_ids.tolist()):
            object_velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                object_velocity,
                0,
            )
            analytic_body_ang[frame, column] = object_velocity[:3]
            analytic_body_lin[frame, column] = object_velocity[3:]

            jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
            jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
            mujoco.mj_jacBodyCom(model, data, jacp, jacr, int(body_id))
            row = 6 * column
            velocity_jacobian[row : row + 3] = jacp
            velocity_jacobian[row + 3 : row + 6] = jacr
            velocity_rhs[row : row + 3] = stored_body_lin[frame, column]
            velocity_rhs[row + 3 : row + 6] = stored_body_ang[frame, column]

        # Inverse direction: the complete named body set must observe every
        # generalized velocity DoF.  Solving all body channels back to qvel is
        # independent of the stored joint_vel values being checked.
        recovered_qvel, _, rank, singular_values = np.linalg.lstsq(
            velocity_jacobian,
            velocity_rhs,
            rcond=None,
        )
        rank = int(rank)
        velocity_jacobian_min_rank = min(velocity_jacobian_min_rank, rank)
        if rank != int(model.nv):
            raise DynamicsGateError(
                "schema-2 body velocity Jacobian is not full column rank: "
                f"{rank} != {int(model.nv)}"
            )
        condition = float(singular_values[0] / singular_values[-1])
        velocity_jacobian_max_condition = max(
            velocity_jacobian_max_condition, condition
        )
        if (
            not np.isfinite(condition)
            or condition > SCHEMA_VELOCITY_JACOBIAN_CONDITION_MAX
        ):
            raise DynamicsGateError(
                "schema-2 body velocity Jacobian is ill-conditioned: "
                f"{condition:.9g}"
            )
        inverse_body_velocity_residual = max(
            inverse_body_velocity_residual,
            float(
                np.max(
                    np.abs(velocity_jacobian @ recovered_qvel - velocity_rhs)
                )
            ),
        )
        joint_velocity_mismatch = max(
            joint_velocity_mismatch,
            float(
                np.max(
                    np.abs(
                        recovered_qvel[plant.joint_dofadr] - joint_vel[frame]
                    )
                )
            ),
        )
        root_linear_velocity_mismatch = max(
            root_linear_velocity_mismatch,
            float(
                np.linalg.norm(
                    recovered_qvel[0:3] - expected_qvel[0:3]
                )
            ),
        )
        root_angular_velocity_mismatch = max(
            root_angular_velocity_mismatch,
            float(
                np.linalg.norm(
                    recovered_qvel[3:6] - expected_qvel[3:6]
                )
            ),
        )

    position_error = np.linalg.norm(fk_pos - stored_body_pos, axis=-1)
    orientation_error = _quaternion_angle_error(fk_quat, stored_body_quat)
    lin_delta = analytic_body_lin - stored_body_lin
    ang_delta = analytic_body_ang - stored_body_ang
    lin_error = np.linalg.norm(lin_delta, axis=-1)
    ang_error = np.linalg.norm(ang_delta, axis=-1)
    receipt = {
        "body_count": float(len(plant.runtime_body_ids)),
        "position_peak_error_m": float(position_error.max()),
        "orientation_peak_error_rad": float(orientation_error.max()),
        "linear_velocity_peak_error_m_s": float(lin_error.max()),
        "angular_velocity_peak_error_rad_s": float(ang_error.max()),
        "linear_velocity_peak_abs_error_m_s": float(np.max(np.abs(lin_delta))),
        "angular_velocity_peak_abs_error_rad_s": float(np.max(np.abs(ang_delta))),
        "joint_velocity_peak_abs_mismatch_rad_s": joint_velocity_mismatch,
        "root_linear_velocity_peak_mismatch_m_s": (
            root_linear_velocity_mismatch
        ),
        "root_angular_velocity_peak_mismatch_rad_s": (
            root_angular_velocity_mismatch
        ),
        "inverse_body_velocity_peak_abs_residual": (
            inverse_body_velocity_residual
        ),
        "velocity_jacobian_min_rank": float(velocity_jacobian_min_rank),
        "velocity_jacobian_max_condition": velocity_jacobian_max_condition,
        "absolute_tolerance": SCHEMA_BODY_LINEAR_VELOCITY_TOL_M_S,
    }
    limits = {
        "position_peak_error_m": SCHEMA_FK_POSITION_TOL_M,
        "orientation_peak_error_rad": SCHEMA_FK_ORIENTATION_TOL_RAD,
        "linear_velocity_peak_error_m_s": SCHEMA_BODY_LINEAR_VELOCITY_TOL_M_S,
        "angular_velocity_peak_error_rad_s": SCHEMA_BODY_ANGULAR_VELOCITY_TOL_RAD_S,
        "joint_velocity_peak_abs_mismatch_rad_s": (
            SCHEMA_JOINT_VELOCITY_TOL_RAD_S
        ),
        "root_linear_velocity_peak_mismatch_m_s": (
            SCHEMA_ROOT_LINEAR_VELOCITY_TOL_M_S
        ),
        "root_angular_velocity_peak_mismatch_rad_s": (
            SCHEMA_ROOT_ANGULAR_VELOCITY_TOL_RAD_S
        ),
    }
    violations = [
        f"{name}={receipt[name]:.9g}>{limit:.9g}"
        for name, limit in limits.items()
        if receipt[name] > limit
    ]
    if violations:
        raise DynamicsGateError(
            "schema-2 body channels disagree with exact MJCF reconstruction: "
            + ", ".join(violations)
        )
    return receipt


def trajectory_from_schema2_npz(
    path: os.PathLike[str] | str, plant: PlantModel
) -> MotionTrajectory:
    """Load an exact schema-2 motion and recover free-joint-origin velocity."""

    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        raise DynamicsGateError(f"motion NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        if frozenset(data.files) not in ALLOWED_SCHEMA2_FIELDSETS:
            raise DynamicsGateError(
                "schema-2 accepts exactly the 11 base fields or all 3 migration "
                f"fields; got {sorted(data.files)}"
            )
        missing = REQUIRED_SCHEMA2_FIELDS - set(data.files)
        if missing:
            raise DynamicsGateError(
                f"{npz_path.name} is missing schema-2 fields: {sorted(missing)}"
            )
        version = _float_scalar(data["kinematics_schema_version"], "schema version")
        if version != 2.0:
            raise DynamicsGateError(f"expected kinematics_schema_version=2, got {version}")
        if _text_scalar(data["body_pos_point"], "body_pos_point") != "link_origin":
            raise DynamicsGateError("body_pos_point must be 'link_origin'")
        if (
            _text_scalar(data["body_lin_vel_point"], "body_lin_vel_point")
            != "center_of_mass"
        ):
            raise DynamicsGateError("body_lin_vel_point must be 'center_of_mass'")
        body_names_raw = np.asarray(data["body_names"]).reshape(-1).tolist()
        body_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in body_names_raw
        ]
        if tuple(body_names) != plant.runtime_body_names:
            raise DynamicsGateError(
                "body_names must exactly equal the 32-body runtime order"
            )
        root_col = 0

        body_pos = np.asarray(data["body_pos_w"], np.float64)
        body_quat = np.asarray(data["body_quat_w"], np.float64)
        body_lin = np.asarray(data["body_lin_vel_w"], np.float64)
        body_ang = np.asarray(data["body_ang_vel_w"], np.float64)
        if (
            body_pos.ndim != 3
            or body_pos.shape[2] != 3
            or body_quat.shape != (*body_pos.shape[:2], 4)
            or body_lin.shape != body_pos.shape
            or body_ang.shape != body_pos.shape
            or body_pos.shape[1] != len(body_names)
        ):
            raise DynamicsGateError("schema-2 body array shapes/body_names disagree")

        joint_pos = np.asarray(data["joint_pos"], np.float64)
        joint_vel = np.asarray(data["joint_vel"], np.float64)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != len(RUNTIME_JOINT_NAMES):
            raise DynamicsGateError("joint_pos must have shape (T, 31)")
        if joint_vel.shape != joint_pos.shape:
            raise DynamicsGateError("joint_vel shape must equal joint_pos")
        if body_pos.shape[0] != joint_pos.shape[0]:
            raise DynamicsGateError("joint and body channels disagree on T")
        root_quat = body_quat[:, root_col]
        root_ang = body_ang[:, root_col]
        fps = _float_scalar(data["fps"], "fps")
        body_receipt = _validate_schema2_body_channels(
            plant,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_pos=body_pos[:, root_col],
            root_quat=root_quat,
            stored_body_pos=body_pos,
            stored_body_quat=body_quat,
            stored_body_lin=body_lin,
            stored_body_ang=body_ang,
            fps=fps,
        )
        root_lin_origin = com_velocity_to_origin_velocity(
            body_lin[:, root_col],
            root_ang,
            root_quat,
            plant.root_body_ipos,
        )
        return MotionTrajectory(
            joint_pos=joint_pos.copy(),
            joint_vel=joint_vel.copy(),
            root_pos_w=body_pos[:, root_col].copy(),
            root_quat_wxyz=root_quat.copy(),
            root_lin_vel_w=root_lin_origin,
            root_ang_vel_w=root_ang.copy(),
            fps=fps,
            source=str(npz_path),
            schema2_body_validation=body_receipt,
        )


def _normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, np.float64).copy()
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    for frame in range(1, len(out)):
        if float(out[frame] @ out[frame - 1]) < 0.0:
            out[frame] *= -1.0
    return out


def build_generalized_state(
    plant: PlantModel, trajectory: MotionTrajectory
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build MuJoCo qpos/qvel/qacc using the provided velocities."""

    frames = trajectory.frames
    quat = _normalize_quaternions(trajectory.root_quat_wxyz)
    qpos = np.zeros((frames, plant.nq), np.float64)
    qvel = np.zeros((frames, plant.nv), np.float64)
    qpos[:, 0:3] = np.asarray(trajectory.root_pos_w, np.float64)
    qpos[:, 3:7] = quat
    qpos[:, plant.joint_qposadr] = np.asarray(trajectory.joint_pos, np.float64)
    qvel[:, 0:3] = np.asarray(trajectory.root_lin_vel_w, np.float64)
    for frame in range(frames):
        rotation = _quat_to_mat_wxyz(quat[frame])
        qvel[frame, 3:6] = rotation.T @ np.asarray(
            trajectory.root_ang_vel_w[frame], np.float64
        )
    qvel[:, plant.joint_dofadr] = np.asarray(trajectory.joint_vel, np.float64)
    qacc = np.gradient(qvel, 1.0 / float(trajectory.fps), axis=0, edge_order=2)
    if not np.isfinite(qacc).all():
        raise DynamicsGateError("derived generalized acceleration is non-finite")
    return qpos, qvel, qacc


def finite_difference_velocity(
    plant: PlantModel, qpos: np.ndarray, fps: float
) -> np.ndarray:
    """Quaternion-correct frame-centred velocity reconstructed from positions."""

    frames = qpos.shape[0]
    dt = 1.0 / float(fps)
    half = np.zeros((frames - 1, plant.nv), np.float64)
    buffer = np.zeros(plant.nv, np.float64)
    for frame in range(frames - 1):
        mujoco.mj_differentiatePos(
            plant.contact_model, buffer, dt, qpos[frame], qpos[frame + 1]
        )
        half[frame] = buffer
    result = np.zeros((frames, plant.nv), np.float64)
    result[0] = half[0]
    result[-1] = half[-1]
    result[1:-1] = 0.5 * (half[:-1] + half[1:])
    return result


def velocity_consistency_report(
    plant: PlantModel,
    trajectory: MotionTrajectory,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> Dict[str, Any]:
    """Return the correct velocity receipt for schema-2 or an array-only proxy.

    A schema-2 loader has all 32 analytic body channels, so it can recover the
    complete generalized velocity independently and gate that exact result.
    Bare array callers do not have those redundant channels; for them the old
    centred finite-difference comparison remains explicitly labelled a proxy.
    In both cases the 50 Hz pose difference is reported, but it is not allowed
    to replace analytic instantaneous truth for schema-2.
    """

    fd_qvel = finite_difference_velocity(plant, qpos, trajectory.fps)
    eval_mask = np.zeros(trajectory.frames, dtype=bool)
    eval_mask[1:-1] = True
    joint_fd_mismatch = np.abs(
        fd_qvel[:, plant.joint_dofadr] - qvel[:, plant.joint_dofadr]
    )
    root_lin_fd_mismatch = np.linalg.norm(
        fd_qvel[:, 0:3] - qvel[:, 0:3], axis=1
    )
    root_ang_fd_mismatch = np.linalg.norm(
        fd_qvel[:, 3:6] - qvel[:, 3:6], axis=1
    )
    finite_difference_diagnostic = {
        "method": "50hz_adjacent_pose_average_not_instantaneous_truth",
        "gating": False,
        "joint_velocity_peak_abs_mismatch_rad_s": float(
            joint_fd_mismatch[eval_mask].max()
        ),
        "root_linear_velocity_peak_mismatch_m_s": float(
            root_lin_fd_mismatch[eval_mask].max()
        ),
        "root_angular_velocity_peak_mismatch_rad_s": float(
            root_ang_fd_mismatch[eval_mask].max()
        ),
        "evaluated_frames": np.flatnonzero(eval_mask).astype(int).tolist(),
    }

    receipt = trajectory.schema2_body_validation
    if receipt is None:
        joint_peak = finite_difference_diagnostic[
            "joint_velocity_peak_abs_mismatch_rad_s"
        ]
        root_linear_peak = finite_difference_diagnostic[
            "root_linear_velocity_peak_mismatch_m_s"
        ]
        root_angular_peak = finite_difference_diagnostic[
            "root_angular_velocity_peak_mismatch_rad_s"
        ]
        passed = bool(
            joint_peak <= thresholds.stored_joint_velocity_tolerance_rad_s
            and root_linear_peak
            <= thresholds.stored_root_linear_velocity_tolerance_m_s
            and root_angular_peak
            <= thresholds.stored_root_angular_velocity_tolerance_rad_s
        )
        return {
            "method": "array_api_50hz_finite_difference_proxy",
            "pass": passed,
            "joint_velocity_peak_abs_mismatch_rad_s": joint_peak,
            "root_linear_velocity_peak_mismatch_m_s": root_linear_peak,
            "root_angular_velocity_peak_mismatch_rad_s": root_angular_peak,
            "evaluated_frames": finite_difference_diagnostic[
                "evaluated_frames"
            ],
            "sampled_pose_finite_difference_diagnostic": (
                finite_difference_diagnostic
            ),
        }

    required = {
        "joint_velocity_peak_abs_mismatch_rad_s",
        "root_linear_velocity_peak_mismatch_m_s",
        "root_angular_velocity_peak_mismatch_rad_s",
    }
    missing = required - set(receipt)
    if missing:
        raise DynamicsGateError(
            "schema2_body_validation lacks analytic qvel recovery fields: "
            f"{sorted(missing)}"
        )
    joint_peak = float(receipt["joint_velocity_peak_abs_mismatch_rad_s"])
    root_linear_peak = float(
        receipt["root_linear_velocity_peak_mismatch_m_s"]
    )
    root_angular_peak = float(
        receipt["root_angular_velocity_peak_mismatch_rad_s"]
    )
    if not np.isfinite([joint_peak, root_linear_peak, root_angular_peak]).all():
        raise DynamicsGateError(
            "schema2 analytic velocity consistency receipt is non-finite"
        )
    passed = bool(
        joint_peak <= SCHEMA_JOINT_VELOCITY_TOL_RAD_S
        and root_linear_peak <= SCHEMA_ROOT_LINEAR_VELOCITY_TOL_M_S
        and root_angular_peak <= SCHEMA_ROOT_ANGULAR_VELOCITY_TOL_RAD_S
    )
    return {
        "method": (
            "schema2_all_32_body_com_velocity_jacobian_inverse_and_"
            "mujoco_objectVelocity_world_forward"
        ),
        "pass": passed,
        "joint_velocity_peak_abs_mismatch_rad_s": joint_peak,
        "root_linear_velocity_peak_mismatch_m_s": root_linear_peak,
        "root_angular_velocity_peak_mismatch_rad_s": root_angular_peak,
        "evaluated_frames": list(range(trajectory.frames)),
        "sampled_pose_finite_difference_diagnostic": (
            finite_difference_diagnostic
        ),
    }


def evaluate_limits(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_limits: np.ndarray,
    thresholds: GateThresholds,
) -> Dict[str, Any]:
    """Pure NumPy joint position/qdot diagnostics."""

    q = np.asarray(joint_pos, np.float64)
    qd = np.asarray(joint_vel, np.float64)
    lo = np.asarray(lower, np.float64)
    hi = np.asarray(upper, np.float64)
    vl = np.asarray(velocity_limits, np.float64)
    if q.shape != qd.shape or q.ndim != 2 or q.shape[1] != len(RUNTIME_JOINT_NAMES):
        raise DynamicsGateError("joint arrays must both have shape (T, 31)")
    for label, vector in (("lower", lo), ("upper", hi), ("velocity_limits", vl)):
        if vector.shape != (len(RUNTIME_JOINT_NAMES),) or not np.isfinite(vector).all():
            raise DynamicsGateError(f"{label} must contain 31 finite values")
    if np.any(vl <= 0) or np.any(lo >= hi):
        raise DynamicsGateError("invalid limit vectors")

    below = np.maximum(lo[None, :] - q, 0.0)
    above = np.maximum(q - hi[None, :], 0.0)
    position_excess = np.maximum(below, above)
    velocity_util = np.abs(qd) / vl[None, :]
    pos_bad = position_excess > thresholds.position_tolerance_rad
    vel_bad = velocity_util > 1.0 + thresholds.velocity_tolerance_fraction

    pos_flat = int(np.argmax(position_excess))
    pos_frame, pos_joint = np.unravel_index(pos_flat, position_excess.shape)
    vel_flat = int(np.argmax(velocity_util))
    vel_frame, vel_joint = np.unravel_index(vel_flat, velocity_util.shape)
    return {
        "pass": bool(not pos_bad.any() and not vel_bad.any()),
        "position": {
            "pass": bool(not pos_bad.any()),
            "violation_samples": int(pos_bad.sum()),
            "violation_frames": np.flatnonzero(pos_bad.any(axis=1)).astype(int).tolist(),
            "peak_excess_rad": float(position_excess[pos_frame, pos_joint]),
            "peak_frame": int(pos_frame),
            "peak_joint": RUNTIME_JOINT_NAMES[int(pos_joint)],
        },
        "velocity": {
            "pass": bool(not vel_bad.any()),
            "violation_samples": int(vel_bad.sum()),
            "violation_frames": np.flatnonzero(vel_bad.any(axis=1)).astype(int).tolist(),
            "peak_utilization": float(velocity_util[vel_frame, vel_joint]),
            "peak_frame": int(vel_frame),
            "peak_joint": RUNTIME_JOINT_NAMES[int(vel_joint)],
            "per_frame_peak_utilization": velocity_util.max(axis=1).tolist(),
            "per_joint_peak_utilization": velocity_util.max(axis=0).tolist(),
        },
    }


def assess_torque_interpretation(
    *,
    direct_actuation_verified: bool,
    plant_identity_bound: bool = True,
    root_force_w: np.ndarray,
    root_torque_local: np.ndarray,
    contact_count: np.ndarray,
    thresholds: GateThresholds,
    evaluated_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Decide whether inverse joint rows are uniquely actuator torque."""

    force = np.asarray(root_force_w, np.float64)
    torque = np.asarray(root_torque_local, np.float64)
    contacts = np.asarray(contact_count, np.int64)
    if (
        force.ndim != 2
        or force.shape[1] != 3
        or torque.shape != force.shape
        or contacts.shape != (force.shape[0],)
    ):
        raise DynamicsGateError("root residual/contact shapes disagree")
    mask = (
        np.ones(force.shape[0], dtype=bool)
        if evaluated_mask is None
        else np.asarray(evaluated_mask, bool)
    )
    if mask.shape != contacts.shape or not mask.any():
        raise DynamicsGateError("evaluated_mask must select at least one frame")
    force_norm = np.linalg.norm(force, axis=1)
    torque_norm = np.linalg.norm(torque, axis=1)
    reasons: List[str] = []
    if not direct_actuation_verified:
        reasons.append("plant actuation is not a verified one-to-one direct motor map")
    if not plant_identity_bound:
        reasons.append(
            "exact MJCF source hash and compiled-model signature were not both bound"
        )
    if np.any(contacts[mask] > 0):
        reasons.append(
            "geometric contact is active; contact wrench distribution is unspecified"
        )
    if np.any(force_norm[mask] > thresholds.root_force_zero_tolerance_n):
        reasons.append(
            "floating-root force residual is non-zero; actuators alone cannot realize the path"
        )
    if np.any(torque_norm[mask] > thresholds.root_torque_zero_tolerance_nm):
        reasons.append(
            "floating-root torque residual is non-zero; actuators alone cannot realize the path"
        )
    valid = not reasons
    return {
        "valid": valid,
        "label": (
            "certified_contact_free_direct_motor_torque"
            if valid
            else "uncertified_contact_disabled_joint_effort_proxy"
        ),
        "reasons": reasons,
        "root_force_peak_n": float(force_norm[mask].max()),
        "root_force_peak_frame": int(np.flatnonzero(mask)[np.argmax(force_norm[mask])]),
        "root_torque_peak_nm": float(torque_norm[mask].max()),
        "root_torque_peak_frame": int(
            np.flatnonzero(mask)[np.argmax(torque_norm[mask])]
        ),
        "contact_frames": np.flatnonzero((contacts > 0) & mask).astype(int).tolist(),
    }


def _safe_obj_name(model: Any, obj: Any, index: int, fallback: str) -> str:
    name = mujoco.mj_id2name(model, obj, int(index))
    return name if name else f"{fallback}_{int(index)}"


def _scan_pose_geometry(
    plant: PlantModel, qpos: np.ndarray, qvel: np.ndarray
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model = plant.contact_model
    data = mujoco.MjData(model)
    frames: List[Dict[str, Any]] = []
    com_pos = np.zeros((len(qpos), 3), np.float64)
    root_tilt = np.zeros(len(qpos), np.float64)
    contact_count = np.zeros(len(qpos), np.int64)
    self_rows: List[Dict[str, Any]] = []
    floor_rows: List[Dict[str, Any]] = []
    world_rows: List[Dict[str, Any]] = []

    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        com_pos[frame] = np.asarray(data.subtree_com[plant.root_body_id], float)
        root_z_axis = np.asarray(data.xmat[plant.root_body_id], float).reshape(3, 3)[:, 2]
        root_tilt[frame] = math.acos(float(np.clip(root_z_axis[2], -1.0, 1.0)))

        frame_self_depth = 0.0
        frame_floor_depth = 0.0
        frame_nonfoot_floor_depth = 0.0
        frame_world_depth = 0.0
        active = 0
        for ci in range(int(data.ncon)):
            contact = data.contact[ci]
            if float(contact.dist) > 0.0:
                continue
            active += 1
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            depth = max(0.0, -float(contact.dist))
            geom1 = _safe_obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, g1, "geom")
            geom2 = _safe_obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, g2, "geom")
            body1 = _safe_obj_name(model, mujoco.mjtObj.mjOBJ_BODY, b1, "body")
            body2 = _safe_obj_name(model, mujoco.mjtObj.mjOBJ_BODY, b2, "body")
            row = {
                "frame": frame,
                "depth_m": depth,
                "geom_pair": [geom1, geom2],
                "body_pair": [body1, body2],
            }
            if b1 == 0 or b2 == 0:
                robot_body = b2 if b1 == 0 else b1
                world_geom = g1 if b1 == 0 else g2
                if world_geom == plant.floor_geom_id:
                    is_foot = robot_body in plant.foot_body_ids
                    floor_row = dict(row)
                    floor_row["robot_body_is_foot"] = bool(is_foot)
                    floor_rows.append(floor_row)
                    frame_floor_depth = max(frame_floor_depth, depth)
                    if not is_foot:
                        frame_nonfoot_floor_depth = max(
                            frame_nonfoot_floor_depth, depth
                        )
                else:
                    world_rows.append(row)
                    frame_world_depth = max(frame_world_depth, depth)
            elif (
                _descends_from(model, b1, plant.root_body_id)
                and _descends_from(model, b2, plant.root_body_id)
                and b1 != b2
            ):
                self_rows.append(row)
                frame_self_depth = max(frame_self_depth, depth)

        contact_count[frame] = active
        frames.append(
            {
                "contact_count": active,
                "self_penetration_peak_m": frame_self_depth,
                "floor_penetration_peak_m": frame_floor_depth,
                "nonfoot_floor_penetration_peak_m": frame_nonfoot_floor_depth,
                "other_world_penetration_peak_m": frame_world_depth,
            }
        )

    return (
        frames,
        {
            "self_contacts": self_rows,
            "floor_contacts": floor_rows,
            "other_world_contacts": world_rows,
        },
        com_pos,
        root_tilt,
        contact_count,
    )


def _peak_matrix(matrix: np.ndarray) -> Tuple[int, int, float]:
    flat = int(np.argmax(matrix))
    frame, joint = np.unravel_index(flat, matrix.shape)
    return int(frame), int(joint), float(matrix[frame, joint])


def analyze_trajectory(
    plant: PlantModel,
    trajectory: MotionTrajectory,
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> Dict[str, Any]:
    """Run the CPU-only plant-specific screen and return JSON-safe evidence."""

    qpos, qvel, qacc = build_generalized_state(plant, trajectory)
    consistency = velocity_consistency_report(
        plant,
        trajectory,
        qpos,
        qvel,
        thresholds=thresholds,
    )

    limits = evaluate_limits(
        trajectory.joint_pos,
        trajectory.joint_vel,
        plant.position_lower,
        plant.position_upper,
        plant.velocity_limits,
        thresholds,
    )

    inverse_data = mujoco.MjData(plant.inverse_model)
    generalized_effort = np.zeros((trajectory.frames, plant.nv), np.float64)
    for frame in range(trajectory.frames):
        inverse_data.qpos[:] = qpos[frame]
        inverse_data.qvel[:] = qvel[frame]
        inverse_data.qacc[:] = qacc[frame]
        mujoco.mj_inverse(plant.inverse_model, inverse_data)
        generalized_effort[frame] = inverse_data.qfrc_inverse
    if not np.isfinite(generalized_effort).all():
        raise DynamicsGateError("mj_inverse produced NaN/Inf")

    joint_effort_proxy = generalized_effort[:, plant.joint_dofadr]
    actuator_force_proxy = joint_effort_proxy / plant.motor_gear[None, :]
    joint_util_proxy = np.abs(joint_effort_proxy) / plant.joint_effort_limits[None, :]
    actuator_util_proxy = (
        np.abs(actuator_force_proxy) / plant.actuator_force_limits[None, :]
    )
    root_force_w = generalized_effort[:, 0:3]
    root_torque_local = generalized_effort[:, 3:6]

    geometry_frames, contacts, com_pos, root_tilt, contact_count = _scan_pose_geometry(
        plant, qpos, qvel
    )
    torque_interpretation = assess_torque_interpretation(
        direct_actuation_verified=plant.actuation_mapping_verified,
        plant_identity_bound=plant.identity_bound,
        root_force_w=root_force_w,
        root_torque_local=root_torque_local,
        contact_count=contact_count,
        thresholds=thresholds,
        evaluated_mask=np.ones(trajectory.frames, dtype=bool),
    )

    self_bad = [
        row
        for row in contacts["self_contacts"]
        if row["depth_m"] > thresholds.self_penetration_tolerance_m
    ]
    foot_floor_bad = [
        row
        for row in contacts["floor_contacts"]
        if row["robot_body_is_foot"]
        and row["depth_m"] > thresholds.foot_floor_penetration_tolerance_m
    ]
    nonfoot_floor_bad = [
        row
        for row in contacts["floor_contacts"]
        if not row["robot_body_is_foot"]
        and row["depth_m"] > thresholds.nonfoot_floor_penetration_tolerance_m
    ]
    other_world_bad = [
        row
        for row in contacts["other_world_contacts"]
        if row["depth_m"] > thresholds.nonfoot_floor_penetration_tolerance_m
    ]
    geometry_pass = not (
        self_bad or foot_floor_bad or nonfoot_floor_bad or other_world_bad
    )

    jf, jj, jp = _peak_matrix(joint_util_proxy)
    af, aj, ap = _peak_matrix(actuator_util_proxy)
    effort_pass = bool(
        torque_interpretation["valid"]
        and jp <= 1.0 + thresholds.velocity_tolerance_fraction
        and ap <= 1.0 + thresholds.velocity_tolerance_fraction
    )

    dt = 1.0 / float(trajectory.fps)
    com_velocity = np.gradient(com_pos, dt, axis=0, edge_order=2)
    root_pos = np.asarray(trajectory.root_pos_w, np.float64)
    root_xy_displacement = np.linalg.norm(root_pos[:, :2] - root_pos[0, :2], axis=1)
    com_xy_displacement = np.linalg.norm(com_pos[:, :2] - com_pos[0, :2], axis=1)
    root_com_xy = np.linalg.norm(com_pos[:, :2] - root_pos[:, :2], axis=1)

    per_frame: List[Dict[str, Any]] = []
    for frame in range(trajectory.frames):
        per_frame.append(
            {
                "frame": frame,
                "time_s": frame / float(trajectory.fps),
                "joint_position_limit_excess_peak_rad": float(
                    np.maximum(
                        np.maximum(
                            plant.position_lower - trajectory.joint_pos[frame], 0.0
                        ),
                        np.maximum(
                            trajectory.joint_pos[frame] - plant.position_upper, 0.0
                        ),
                    ).max()
                ),
                "joint_velocity_utilization_peak": float(
                    np.max(np.abs(trajectory.joint_vel[frame]) / plant.velocity_limits)
                ),
                "joint_effort_proxy_utilization_peak": float(
                    joint_util_proxy[frame].max()
                ),
                "actuator_force_proxy_utilization_peak": float(
                    actuator_util_proxy[frame].max()
                ),
                "root_force_residual_norm_n": float(
                    np.linalg.norm(root_force_w[frame])
                ),
                "root_torque_residual_norm_nm": float(
                    np.linalg.norm(root_torque_local[frame])
                ),
                "root_height_m": float(root_pos[frame, 2]),
                "root_tilt_rad": float(root_tilt[frame]),
                "root_linear_speed_m_s": float(
                    np.linalg.norm(trajectory.root_lin_vel_w[frame])
                ),
                "root_angular_speed_rad_s": float(
                    np.linalg.norm(trajectory.root_ang_vel_w[frame])
                ),
                "com_position_w_m": com_pos[frame].tolist(),
                "com_speed_m_s": float(np.linalg.norm(com_velocity[frame])),
                **geometry_frames[frame],
            }
        )

    complete_pass = bool(
        consistency["pass"]
        and limits["pass"]
        and geometry_pass
        and effort_pass
    )
    if complete_pass:
        verdict = "PASS"
    elif not torque_interpretation["valid"] and (
        consistency["pass"] and limits["pass"] and geometry_pass
    ):
        verdict = "INCOMPLETE_FAIL_CLOSED"
    else:
        verdict = "FAIL"

    return {
        "schema_version": 1,
        "verdict": verdict,
        "screen_pass": complete_pass,
        "plant_specific_screen": True,
        "source": trajectory.source,
        "non_claims": [
            "not a balance certificate",
            "not a controller tracking or policy learnability certificate",
            "not a legal table return or strike behaviour certificate",
            "not a hardware/deployment authorization",
            "contact-disabled joint effort is not actuator torque unless torque_interpretation.valid=true",
        ],
        "plant": {
            "mjcf_path": plant.mjcf_path,
            "mjcf_sha256": plant.mjcf_sha256,
            "urdf_path": plant.urdf_path,
            "urdf_sha256": plant.urdf_sha256,
            "compiled_model_signature_sha256": plant.compiled_signature_sha256,
            "table_obstacle": (
                {"enabled": False}
                if plant.table_scene is None
                else {
                    "enabled": True,
                    "source": "in_memory_mjcf_augmentation_disk_file_untouched",
                    "obstacle_names": list(mujoco_table_scene.OBSTACLE_NAMES),
                    "isaac_equivalent_obstacles": list(
                        mujoco_table_scene.ISAAC_EQUIVALENT_OBSTACLES
                    ),
                    "near_x_m": plant.table_scene.near_x,
                    "surface_z_m": plant.table_scene.surface_z,
                    "augmented_mjcf_sha256": plant.table_scene.augmented_xml_sha256,
                    "note": (
                        "table overlap is reported under checks.geometry."
                        "other_world_penetration_violations"
                    ),
                }
            ),
            "expected_identity": dict(plant.identity_expectations),
            "identity_bound": plant.identity_bound,
            "actuation_mapping_verified": plant.actuation_mapping_verified,
            "runtime_joint_names": list(RUNTIME_JOINT_NAMES),
            "weight_n": plant.weight_n,
            "joint_effort_limits_nm": plant.joint_effort_limits.tolist(),
            "actuator_force_limits": plant.actuator_force_limits.tolist(),
            "joint_velocity_limits_rad_s": plant.velocity_limits.tolist(),
        },
        "input": {
            "frames": trajectory.frames,
            "fps": float(trajectory.fps),
            "duration_s": (trajectory.frames - 1) / float(trajectory.fps),
            "velocity_position_consistency": consistency,
            "schema2_body_validation": trajectory.schema2_body_validation,
        },
        "checks": {
            "joint_limits": limits,
            "geometry": {
                "pass": geometry_pass,
                "self_collision_violation_count": len(self_bad),
                "foot_floor_penetration_violation_count": len(foot_floor_bad),
                "nonfoot_floor_penetration_violation_count": len(nonfoot_floor_bad),
                "other_world_penetration_violation_count": len(other_world_bad),
                "self_collision_violations": self_bad,
                "foot_floor_penetration_violations": foot_floor_bad,
                "nonfoot_floor_penetration_violations": nonfoot_floor_bad,
                "other_world_penetration_violations": other_world_bad,
                "all_contacts": contacts,
            },
            "inverse_dynamics": {
                "method": "contact_and_limit_disabled_mj_inverse",
                "torque_interpretation": torque_interpretation,
                "joint_effort_proxy_peak_utilization": jp,
                "joint_effort_proxy_peak_frame": jf,
                "joint_effort_proxy_peak_joint": RUNTIME_JOINT_NAMES[jj],
                "actuator_force_proxy_peak_utilization": ap,
                "actuator_force_proxy_peak_frame": af,
                "actuator_force_proxy_peak_joint": RUNTIME_JOINT_NAMES[aj],
                "effort_limit_pass_only_if_interpretation_valid": effort_pass,
            },
            "root_and_com": {
                "root_height_min_m": float(root_pos[:, 2].min()),
                "root_height_max_m": float(root_pos[:, 2].max()),
                "root_tilt_peak_rad": float(root_tilt.max()),
                "root_xy_displacement_peak_m": float(root_xy_displacement.max()),
                "root_linear_speed_peak_m_s": float(
                    np.linalg.norm(trajectory.root_lin_vel_w, axis=1).max()
                ),
                "root_angular_speed_peak_rad_s": float(
                    np.linalg.norm(trajectory.root_ang_vel_w, axis=1).max()
                ),
                "com_height_min_m": float(com_pos[:, 2].min()),
                "com_height_max_m": float(com_pos[:, 2].max()),
                "com_xy_displacement_peak_m": float(com_xy_displacement.max()),
                "com_speed_peak_m_s": float(np.linalg.norm(com_velocity, axis=1).max()),
                "root_to_com_xy_offset_peak_m": float(root_com_xy.max()),
                "note": (
                    "CoM path is kinematic only; support polygon, CoP, friction, "
                    "controller response, and balance remain downstream gates"
                ),
            },
        },
        "series": {
            "joint_names": list(RUNTIME_JOINT_NAMES),
            "joint_effort_proxy_nm": joint_effort_proxy.tolist(),
            "joint_effort_proxy_utilization": joint_util_proxy.tolist(),
            "actuator_force_proxy": actuator_force_proxy.tolist(),
            "actuator_force_proxy_utilization": actuator_util_proxy.tolist(),
            "root_force_residual_w_n": root_force_w.tolist(),
            "root_torque_residual_local_nm": root_torque_local.tolist(),
        },
        "frames": per_frame,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_default_mjcf() -> Path:
    return _repo_root() / DEFAULT_MJCF_REL


def resolve_default_urdf() -> Path:
    return _repo_root() / DEFAULT_URDF_REL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="exact schema-2 motion NPZ")
    parser.add_argument(
        "--mjcf",
        default=str(resolve_default_mjcf()),
        help="exact vendor floating-base MJCF",
    )
    parser.add_argument(
        "--urdf",
        default=str(resolve_default_urdf()),
        help="exact A3 URDF supplying joint velocity limits",
    )
    parser.add_argument("--expected-mjcf-sha256")
    parser.add_argument("--expected-compiled-signature")
    parser.add_argument(
        "--with-table",
        action="store_true",
        help=(
            "append the colliding ping-pong table to an IN-MEMORY copy of the MJCF (the file on "
            "disk is never touched). Table overlap then lands in the existing geometry check's "
            "other_world bucket and can fail the clip. OFF by default so no existing verdict "
            "changes; see scripts/mujoco_table_scene.py"
        ),
    )
    parser.add_argument("--out", help="optional JSON report path; stdout otherwise")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plant = load_plant(
            args.mjcf,
            expected_mjcf_sha256=args.expected_mjcf_sha256,
            expected_compiled_signature=args.expected_compiled_signature,
            urdf_path=args.urdf,
            with_table=args.with_table,
        )
        trajectory = trajectory_from_schema2_npz(args.motion, plant)
        report = analyze_trajectory(plant, trajectory)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "verdict": "FAIL",
            "screen_pass": False,
            "plant_specific_screen": True,
            "error": f"{type(exc).__name__}: {exc}",
            "non_claims": [
                "input/model failure is fail-closed",
                "no motion asset, training process, simulator step, or hardware was modified",
            ],
        }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if report.get("screen_pass") is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
