#!/usr/bin/env python3
"""Adapt canonical path 2-jets to the exact AgiBot MuJoCo tangent space.

The canonical compiler represents an upper-body candidate as 31 runtime-order
joint coordinates.  A full-body candidate appends

``[root_xyz_world, root_rotation_vector_ready_relative]``.

MuJoCo instead stores a floating root as seven position coordinates
(``xyz + wxyz quaternion``) and six tangent coordinates.  In particular, the
rotational part of a free-joint ``qvel``/``qacc`` is expressed in the current
body frame.  This module performs that change of coordinates analytically:

``omega_body = R_world.T @ omega_world``

``alpha_body = R_world.T @ alpha_world``.

The world angular 2-jet is supplied by
``canonical_root_pose_codec.rotation_vector_derivatives_to_world_angular_kinematics``.
No finite difference is used to generate production values.  A separate
``validate_with_mj_differentiate_pos`` helper is intentionally only a sampled,
independent consistency check.

Every adaptation rechecks the source-MJCF digest, XML model name, compiled-model
signature, exact root free-joint, and the name-resolved 31-joint address map.
The result is candidate evidence only: it authorizes neither training,
deployment, nor hardware use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from audit_motion_npz import ISAAC_JOINT_NAMES
from canonical_root_pose_codec import (
    RootPoseCodecError,
    decode_root_pose,
    rotation_vector_derivatives_to_world_angular_kinematics,
)


RUNTIME_JOINT_NAMES = tuple(ISAAC_JOINT_NAMES)
ROOT_COORDINATE_NAMES = (
    "root_x_w",
    "root_y_w",
    "root_z_w",
    "root_rotvec_x_ready",
    "root_rotvec_y_ready",
    "root_rotvec_z_ready",
)
UPPER_COORDINATE_NAMES = RUNTIME_JOINT_NAMES
FULL_COORDINATE_NAMES = RUNTIME_JOINT_NAMES + ROOT_COORDINATE_NAMES

EXPECTED_NQ = 38
EXPECTED_NV = 37
EXPECTED_NJNT = 32
EXPECTED_NU = 31
EXPECTED_MJCF_MODEL_NAME = "A3T2.5_pingpong_0519"
ROOT_BODY_NAME = "pelvis_link"
ROOT_JOINT_NAME = "pelvis_free_joint"

PUBLICATION_CLASS = "canonical_mujoco_path_candidate"
ADAPTER_SEMANTICS = "analytic_ready_rotvec_to_mujoco_freejoint_2jet_v1"
PI_MARGIN_RAD = 1.0e-3
_QUATERNION_NORM_TOL = 1.0e-8
_DIGEST_KEYS = (
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
)


if len(RUNTIME_JOINT_NAMES) != 31 or len(set(RUNTIME_JOINT_NAMES)) != 31:
    raise RuntimeError("canonical runtime joint contract must contain 31 unique names")


class MujocoPathAdapterError(ValueError):
    """A path or model cannot satisfy the fail-closed adapter contract."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ExactMujocoModelBinding:
    """Content and name binding for one exact compiled A3 MuJoCo model."""

    mjcf_path: str
    mjcf_sha256: str
    compiled_model_sha256: str
    xml_model_name: str
    root_body_name: str
    root_joint_name: str
    root_body_id: int
    root_joint_id: int
    root_qpos_adr: int
    root_dof_adr: int
    runtime_joint_names: tuple[str, ...]
    joint_ids: np.ndarray
    joint_qpos_adrs: np.ndarray
    joint_dof_adrs: np.ndarray
    nq: int
    nv: int
    model_binding_sha256: str


@dataclass(frozen=True)
class MujocoPathJet:
    """One sampled candidate path expressed in MuJoCo ``nq``/``nv`` space."""

    qpos: np.ndarray
    q_s: np.ndarray
    q_ss: np.ndarray
    report: Mapping[str, Any]

    @property
    def samples(self) -> int:
        return int(self.qpos.shape[0])


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise MujocoPathAdapterError(
            f"{label} must be exactly 64 lowercase hexadecimal SHA-256 digits"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(digest: "hashlib._Hash", label: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())


def compiled_model_signature(model: Any, mujoco_module: Any) -> str:
    """Match ``canonical_mujoco_dynamics_gate.compiled_model_signature``.

    The function is kept dependency-injected so pure tests can exercise the
    exact name/address contract without importing MuJoCo.  On a real model its
    bytes intentionally match the repository's dynamics-gate receipt.  That
    shared receipt is an allowlist signature, not a hash of complete MJB bytes;
    this adapter therefore uses it only for the coordinate/model mapping and
    makes no dynamics or contact identity claim.
    """

    digest = hashlib.sha256()
    for name in _DIGEST_KEYS:
        if hasattr(model, name):
            _hash_array(digest, name, getattr(model, name))
    if not hasattr(model, "opt"):
        raise MujocoPathAdapterError("compiled MuJoCo model has no option block")
    _hash_array(digest, "gravity", model.opt.gravity)
    _hash_array(digest, "timestep", [model.opt.timestep])
    object_rows = (
        ("body", int(model.nbody), mujoco_module.mjtObj.mjOBJ_BODY),
        ("joint", int(model.njnt), mujoco_module.mjtObj.mjOBJ_JOINT),
        ("geom", int(model.ngeom), mujoco_module.mjtObj.mjOBJ_GEOM),
        ("actuator", int(model.nu), mujoco_module.mjtObj.mjOBJ_ACTUATOR),
    )
    for label, count, object_type in object_rows:
        digest.update(label.encode("ascii"))
        for index in range(count):
            name = mujoco_module.mj_id2name(model, object_type, index) or ""
            digest.update(str(name).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _xml_model_name(path: Path) -> str:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise MujocoPathAdapterError(f"cannot parse MJCF {path}: {exc}") from exc
    if root.tag != "mujoco":
        raise MujocoPathAdapterError(
            f"MJCF root element must be 'mujoco', got {root.tag!r}"
        )
    value = root.attrib.get("model")
    if not isinstance(value, str) or not value:
        raise MujocoPathAdapterError("MJCF root is missing a non-empty model name")
    return value


def _name_table(
    mujoco_module: Any,
    model: Any,
    object_type: Any,
    count: int,
    label: str,
) -> tuple[str | None, ...]:
    rows = tuple(
        mujoco_module.mj_id2name(model, object_type, index)
        for index in range(count)
    )
    duplicates = sorted(
        {
            str(name)
            for name in rows
            if name is not None and rows.count(name) > 1
        }
    )
    if duplicates:
        raise MujocoPathAdapterError(
            f"compiled model has duplicate {label} names: {duplicates}"
        )
    return rows


def _unique_name_id(
    mujoco_module: Any,
    model: Any,
    object_type: Any,
    name_table: tuple[str | None, ...],
    name: str,
    label: str,
) -> int:
    matches = [index for index, value in enumerate(name_table) if value == name]
    if len(matches) != 1:
        raise MujocoPathAdapterError(
            f"compiled model must contain exactly one {label} {name!r}, "
            f"found {len(matches)}"
        )
    resolved = int(mujoco_module.mj_name2id(model, object_type, name))
    if resolved != matches[0]:
        raise MujocoPathAdapterError(
            f"name lookup for {label} {name!r} disagrees with compiled name table"
        )
    return resolved


def _readonly_int64(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int64)
    result.setflags(write=False)
    return result


def _binding_payload(
    *,
    mjcf_sha256: str,
    compiled_model_sha256: str,
    xml_model_name: str,
    root_body_id: int,
    root_joint_id: int,
    joint_ids: np.ndarray,
    joint_qpos_adrs: np.ndarray,
    joint_dof_adrs: np.ndarray,
) -> dict[str, Any]:
    return {
        "contract": "exact_a3_mujoco_name_address_binding_v1",
        "mjcf_sha256": mjcf_sha256,
        "compiled_model_sha256": compiled_model_sha256,
        "xml_model_name": xml_model_name,
        "root_body_name": ROOT_BODY_NAME,
        "root_joint_name": ROOT_JOINT_NAME,
        "root_body_id": int(root_body_id),
        "root_joint_id": int(root_joint_id),
        "root_qpos_adr": 0,
        "root_dof_adr": 0,
        "runtime_joint_names": list(RUNTIME_JOINT_NAMES),
        "joint_ids": np.asarray(joint_ids, np.int64).tolist(),
        "joint_qpos_adrs": np.asarray(joint_qpos_adrs, np.int64).tolist(),
        "joint_dof_adrs": np.asarray(joint_dof_adrs, np.int64).tolist(),
        "nq": EXPECTED_NQ,
        "nv": EXPECTED_NV,
    }


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def bind_exact_mujoco_model(
    mujoco_module: Any,
    model: Any,
    *,
    mjcf_path: os.PathLike[str] | str,
    expected_mjcf_sha256: str,
    expected_compiled_model_sha256: str,
    expected_xml_model_name: str = EXPECTED_MJCF_MODEL_NAME,
) -> ExactMujocoModelBinding:
    """Bind one already-loaded model to exact source and compiled identities."""

    expected_source = _digest(expected_mjcf_sha256, "expected_mjcf_sha256")
    expected_compiled = _digest(
        expected_compiled_model_sha256, "expected_compiled_model_sha256"
    )
    if (
        not isinstance(expected_xml_model_name, str)
        or not expected_xml_model_name
        or expected_xml_model_name.strip() != expected_xml_model_name
    ):
        raise MujocoPathAdapterError(
            "expected_xml_model_name must be one non-empty trimmed string"
        )
    source_path = Path(mjcf_path).expanduser().resolve()
    if not source_path.is_file():
        raise MujocoPathAdapterError(f"MJCF does not exist: {source_path}")
    source_sha = _sha256_file(source_path)
    if source_sha != expected_source:
        raise MujocoPathAdapterError(
            f"MJCF SHA mismatch: expected {expected_source}, got {source_sha}"
        )
    xml_name = _xml_model_name(source_path)
    if xml_name != expected_xml_model_name:
        raise MujocoPathAdapterError(
            f"MJCF model name mismatch: expected {expected_xml_model_name!r}, "
            f"got {xml_name!r}"
        )

    dimensions = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
    }
    expected_dimensions = {
        "nq": EXPECTED_NQ,
        "nv": EXPECTED_NV,
        "njnt": EXPECTED_NJNT,
        "nu": EXPECTED_NU,
    }
    if dimensions != expected_dimensions:
        raise MujocoPathAdapterError(
            f"compiled A3 dimensions drifted: expected {expected_dimensions}, "
            f"got {dimensions}"
        )

    compiled_sha = compiled_model_signature(model, mujoco_module)
    if compiled_sha != expected_compiled:
        raise MujocoPathAdapterError(
            "compiled model signature mismatch: "
            f"expected {expected_compiled}, got {compiled_sha}"
        )

    joint_names = _name_table(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_JOINT,
        int(model.njnt),
        "joint",
    )
    body_names = _name_table(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_BODY,
        int(model.nbody),
        "body",
    )
    root_joint_id = _unique_name_id(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_JOINT,
        joint_names,
        ROOT_JOINT_NAME,
        "root joint",
    )
    root_body_id = _unique_name_id(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_BODY,
        body_names,
        ROOT_BODY_NAME,
        "root body",
    )

    joint_type = np.asarray(model.jnt_type)
    free_type = int(mujoco_module.mjtJoint.mjJNT_FREE)
    hinge_type = int(mujoco_module.mjtJoint.mjJNT_HINGE)
    free_ids = np.flatnonzero(joint_type == free_type)
    if free_ids.tolist() != [root_joint_id]:
        raise MujocoPathAdapterError(
            f"compiled model must have exactly the named root free-joint, "
            f"got free joint ids {free_ids.tolist()}"
        )
    if (
        int(np.asarray(model.jnt_qposadr)[root_joint_id]) != 0
        or int(np.asarray(model.jnt_dofadr)[root_joint_id]) != 0
    ):
        raise MujocoPathAdapterError(
            "root free-joint must occupy qpos[0:7] and tangent[0:6]"
        )
    if int(np.asarray(model.jnt_bodyid)[root_joint_id]) != root_body_id:
        raise MujocoPathAdapterError(
            f"{ROOT_JOINT_NAME!r} is not attached to {ROOT_BODY_NAME!r}"
        )

    joint_ids = np.asarray(
        [
            _unique_name_id(
                mujoco_module,
                model,
                mujoco_module.mjtObj.mjOBJ_JOINT,
                joint_names,
                name,
                "runtime joint",
            )
            for name in RUNTIME_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    if len(set(joint_ids.tolist())) != 31 or root_joint_id in set(
        joint_ids.tolist()
    ):
        raise MujocoPathAdapterError(
            "runtime joint names do not resolve bijectively away from the root"
        )
    if set(name for name in joint_names if name is not None) != {
        ROOT_JOINT_NAME,
        *RUNTIME_JOINT_NAMES,
    }:
        raise MujocoPathAdapterError(
            "compiled joint name set differs from root plus 31 runtime joints"
        )
    if np.any(joint_type[joint_ids] != hinge_type):
        bad = [
            RUNTIME_JOINT_NAMES[index]
            for index, value in enumerate(joint_type[joint_ids])
            if int(value) != hinge_type
        ]
        raise MujocoPathAdapterError(
            f"runtime joints must all be scalar hinges, got {bad}"
        )

    joint_qpos_adrs = np.asarray(model.jnt_qposadr, np.int64)[joint_ids]
    joint_dof_adrs = np.asarray(model.jnt_dofadr, np.int64)[joint_ids]
    if sorted(joint_qpos_adrs.tolist()) != list(range(7, EXPECTED_NQ)):
        raise MujocoPathAdapterError(
            "runtime joint names must cover every post-root qpos address exactly once"
        )
    if sorted(joint_dof_adrs.tolist()) != list(range(6, EXPECTED_NV)):
        raise MujocoPathAdapterError(
            "runtime joint names must cover every actuated tangent address exactly once"
        )

    payload = _binding_payload(
        mjcf_sha256=source_sha,
        compiled_model_sha256=compiled_sha,
        xml_model_name=xml_name,
        root_body_id=root_body_id,
        root_joint_id=root_joint_id,
        joint_ids=joint_ids,
        joint_qpos_adrs=joint_qpos_adrs,
        joint_dof_adrs=joint_dof_adrs,
    )
    return ExactMujocoModelBinding(
        mjcf_path=str(source_path),
        mjcf_sha256=source_sha,
        compiled_model_sha256=compiled_sha,
        xml_model_name=xml_name,
        root_body_name=ROOT_BODY_NAME,
        root_joint_name=ROOT_JOINT_NAME,
        root_body_id=int(root_body_id),
        root_joint_id=int(root_joint_id),
        root_qpos_adr=0,
        root_dof_adr=0,
        runtime_joint_names=RUNTIME_JOINT_NAMES,
        joint_ids=_readonly_int64(joint_ids),
        joint_qpos_adrs=_readonly_int64(joint_qpos_adrs),
        joint_dof_adrs=_readonly_int64(joint_dof_adrs),
        nq=EXPECTED_NQ,
        nv=EXPECTED_NV,
        model_binding_sha256=_payload_sha256(payload),
    )


def load_exact_mujoco_model(
    *,
    mjcf_path: os.PathLike[str] | str,
    expected_mjcf_sha256: str,
    expected_compiled_model_sha256: str,
    expected_xml_model_name: str = EXPECTED_MJCF_MODEL_NAME,
    mujoco_module: Any | None = None,
) -> tuple[Any, ExactMujocoModelBinding]:
    """Load and bind one exact model without importing MuJoCo at module import."""

    if mujoco_module is None:
        try:
            import mujoco as mujoco_module  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MujocoPathAdapterError(
                "MuJoCo Python is required to load the exact A3 model"
            ) from exc
    source_path = Path(mjcf_path).expanduser().resolve()
    try:
        model = mujoco_module.MjModel.from_xml_path(str(source_path))
    except Exception as exc:
        raise MujocoPathAdapterError(
            f"MuJoCo cannot compile MJCF {source_path}: {exc}"
        ) from exc
    binding = bind_exact_mujoco_model(
        mujoco_module,
        model,
        mjcf_path=source_path,
        expected_mjcf_sha256=expected_mjcf_sha256,
        expected_compiled_model_sha256=expected_compiled_model_sha256,
        expected_xml_model_name=expected_xml_model_name,
    )
    return model, binding


def _assert_current_binding(
    mujoco_module: Any,
    model: Any,
    binding: ExactMujocoModelBinding,
) -> None:
    if not isinstance(binding, ExactMujocoModelBinding):
        raise MujocoPathAdapterError(
            "binding must be an ExactMujocoModelBinding produced by this module"
        )
    current = bind_exact_mujoco_model(
        mujoco_module,
        model,
        mjcf_path=binding.mjcf_path,
        expected_mjcf_sha256=binding.mjcf_sha256,
        expected_compiled_model_sha256=binding.compiled_model_sha256,
        expected_xml_model_name=binding.xml_model_name,
    )
    scalar_fields = (
        "mjcf_path",
        "mjcf_sha256",
        "compiled_model_sha256",
        "xml_model_name",
        "root_body_name",
        "root_joint_name",
        "root_body_id",
        "root_joint_id",
        "root_qpos_adr",
        "root_dof_adr",
        "runtime_joint_names",
        "nq",
        "nv",
        "model_binding_sha256",
    )
    if any(getattr(current, name) != getattr(binding, name) for name in scalar_fields):
        raise MujocoPathAdapterError("model binding record drifted after construction")
    for name in ("joint_ids", "joint_qpos_adrs", "joint_dof_adrs"):
        if not np.array_equal(getattr(current, name), getattr(binding, name)):
            raise MujocoPathAdapterError(
                f"model binding address vector {name} drifted after construction"
            )


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.dtype.kind not in "iuf"
        or raw.dtype.kind == "b"
        or np.iscomplexobj(raw)
    ):
        raise MujocoPathAdapterError(
            f"{label} must be a real numeric two-dimensional array, got "
            f"shape={raw.shape}, dtype={raw.dtype}"
        )
    try:
        result = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MujocoPathAdapterError(
            f"{label} cannot be represented as float64"
        ) from exc
    if not np.isfinite(result).all():
        raise MujocoPathAdapterError(f"{label} contains NaN/Inf")
    return result


def _ready_root(
    root_pos_w: Any, root_quat_wxyz: Any
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(root_pos_w)
    quat = np.asarray(root_quat_wxyz)
    if (
        pos.shape != (3,)
        or pos.dtype.kind not in "iuf"
        or pos.dtype.kind == "b"
        or np.iscomplexobj(pos)
        or not np.isfinite(pos).all()
    ):
        raise MujocoPathAdapterError(
            "canonical_ready_root_pos_w must be one finite real (3,) vector"
        )
    if (
        quat.shape != (4,)
        or quat.dtype.kind not in "iuf"
        or quat.dtype.kind == "b"
        or np.iscomplexobj(quat)
        or not np.isfinite(quat).all()
    ):
        raise MujocoPathAdapterError(
            "canonical_ready_root_quat_wxyz must be one finite real (4,) vector"
        )
    pos64 = np.ascontiguousarray(pos, dtype=np.float64)
    quat64 = np.ascontiguousarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat64))
    if norm <= 0.0 or abs(norm - 1.0) > _QUATERNION_NORM_TOL:
        raise MujocoPathAdapterError(
            "canonical ready quaternion is not unit length within "
            f"{_QUATERNION_NORM_TOL:.1e}"
        )
    return pos64, quat64 / norm


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.ndim != 2 or quat.shape[1] != 4 or not np.isfinite(quat).all():
        raise MujocoPathAdapterError(
            f"world quaternion must be finite (N,4), got {quat.shape}"
        )
    norm = np.linalg.norm(quat, axis=1)
    if np.any(norm <= 0.0) or np.max(np.abs(norm - 1.0)) > 1.0e-10:
        raise MujocoPathAdapterError("decoded world quaternion is not unit length")
    value = quat / norm[:, None]
    w, x, y, z = np.moveaxis(value, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


def _array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    _hash_array(digest, "array", value)
    return digest.hexdigest()


def _readonly_float64(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def adapt_canonical_path_jet(
    mujoco_module: Any,
    model: Any,
    binding: ExactMujocoModelBinding,
    *,
    scope: str,
    coordinate_names: Sequence[str],
    q: Any,
    q_s: Any,
    q_ss: Any,
    canonical_ready_root_pos_w: Any,
    canonical_ready_root_quat_wxyz: Any,
    pi_margin_rad: float = PI_MARGIN_RAD,
) -> MujocoPathJet:
    """Convert a sampled canonical 2-jet to exact MuJoCo ``nq``/``nv`` arrays."""

    _assert_current_binding(mujoco_module, model, binding)
    if scope not in ("upper", "full"):
        raise MujocoPathAdapterError(
            f"scope must be exactly 'upper' or 'full', got {scope!r}"
        )
    if isinstance(coordinate_names, (str, bytes)):
        raise MujocoPathAdapterError("coordinate_names must be a sequence of names")
    names = tuple(coordinate_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise MujocoPathAdapterError(
            "coordinate_names must contain only non-empty strings"
        )
    if len(set(names)) != len(names):
        raise MujocoPathAdapterError("coordinate_names contains duplicates")
    expected_names = (
        UPPER_COORDINATE_NAMES if scope == "upper" else FULL_COORDINATE_NAMES
    )
    if names != expected_names:
        missing = [name for name in expected_names if name not in names]
        unexpected = [name for name in names if name not in expected_names]
        raise MujocoPathAdapterError(
            "canonical coordinate order drifted: "
            f"missing={missing}, unexpected={unexpected}, "
            f"expected_order={list(expected_names)}"
        )

    coordinates = _finite_matrix(q, "q")
    tangent = _finite_matrix(q_s, "q_s")
    curvature = _finite_matrix(q_ss, "q_ss")
    if coordinates.shape != tangent.shape or coordinates.shape != curvature.shape:
        raise MujocoPathAdapterError(
            "q, q_s, and q_ss must have identical sampled shapes"
        )
    if coordinates.shape[0] < 1 or coordinates.shape[1] != len(expected_names):
        raise MujocoPathAdapterError(
            f"{scope} canonical path must have shape (N,{len(expected_names)}), "
            f"got {coordinates.shape}"
        )
    ready_pos, ready_quat = _ready_root(
        canonical_ready_root_pos_w, canonical_ready_root_quat_wxyz
    )
    margin = float(pi_margin_rad)
    if not math.isfinite(margin) or margin <= 0.0 or margin >= math.pi:
        raise MujocoPathAdapterError("pi_margin_rad must lie strictly inside (0,pi)")

    samples = len(coordinates)
    qpos = np.empty((samples, binding.nq), dtype=np.float64)
    mujoco_q_s = np.empty((samples, binding.nv), dtype=np.float64)
    mujoco_q_ss = np.empty((samples, binding.nv), dtype=np.float64)
    qpos[:, binding.joint_qpos_adrs] = coordinates[:, :31]
    mujoco_q_s[:, binding.joint_dof_adrs] = tangent[:, :31]
    mujoco_q_ss[:, binding.joint_dof_adrs] = curvature[:, :31]

    if scope == "upper":
        root_pos = np.broadcast_to(ready_pos, (samples, 3))
        root_quat = np.broadcast_to(ready_quat, (samples, 4))
        root_tangent = np.zeros((samples, 6), dtype=np.float64)
        root_curvature = np.zeros((samples, 6), dtype=np.float64)
        max_relative_angle = 0.0
    else:
        root_coordinates = coordinates[:, 31:]
        root_coordinate_tangent = tangent[:, 31:]
        root_coordinate_curvature = curvature[:, 31:]
        try:
            root_pos, root_quat = decode_root_pose(
                root_coordinates,
                canonical_ready_root_quat_wxyz=ready_quat,
                pi_margin_rad=margin,
            )
            omega_world_s, alpha_world_ss = (
                rotation_vector_derivatives_to_world_angular_kinematics(
                    root_coordinates[:, 3:],
                    root_coordinate_tangent[:, 3:],
                    root_coordinate_curvature[:, 3:],
                    canonical_ready_root_quat_wxyz=ready_quat,
                    pi_margin_rad=margin,
                )
            )
        except RootPoseCodecError as exc:
            raise MujocoPathAdapterError(
                f"full root chart cannot be adapted: {exc}"
            ) from exc
        rotation_world = _rotation_matrix_wxyz(root_quat)
        omega_body_s = np.einsum(
            "nji,nj->ni", rotation_world, omega_world_s
        )
        alpha_body_ss = np.einsum(
            "nji,nj->ni", rotation_world, alpha_world_ss
        )
        root_tangent = np.concatenate(
            (root_coordinate_tangent[:, :3], omega_body_s), axis=1
        )
        root_curvature = np.concatenate(
            (root_coordinate_curvature[:, :3], alpha_body_ss), axis=1
        )
        max_relative_angle = float(
            np.max(np.linalg.norm(root_coordinates[:, 3:], axis=1))
        )

    qpos[:, binding.root_qpos_adr : binding.root_qpos_adr + 3] = root_pos
    qpos[:, binding.root_qpos_adr + 3 : binding.root_qpos_adr + 7] = root_quat
    mujoco_q_s[
        :, binding.root_dof_adr : binding.root_dof_adr + 6
    ] = root_tangent
    mujoco_q_ss[
        :, binding.root_dof_adr : binding.root_dof_adr + 6
    ] = root_curvature
    if (
        not np.isfinite(qpos).all()
        or not np.isfinite(mujoco_q_s).all()
        or not np.isfinite(mujoco_q_ss).all()
    ):
        raise MujocoPathAdapterError("adapted MuJoCo path contains NaN/Inf")
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0))
    )
    if quaternion_norm_error > 1.0e-10:
        raise MujocoPathAdapterError(
            "adapted free-joint quaternion lost unit norm: "
            f"{quaternion_norm_error:.9g}"
        )

    qpos_out = _readonly_float64(qpos)
    q_s_out = _readonly_float64(mujoco_q_s)
    q_ss_out = _readonly_float64(mujoco_q_ss)
    report = MappingProxyType(
        {
            "status": "PASS_SAMPLED_MUJOCO_PATH_CANDIDATE_ONLY",
            "publication_class": PUBLICATION_CLASS,
            "adapter_semantics": ADAPTER_SEMANTICS,
            "scope": scope,
            "samples": samples,
            "canonical_dimension": int(coordinates.shape[1]),
            "mujoco_nq": binding.nq,
            "mujoco_nv": binding.nv,
            "model_binding_sha256": binding.model_binding_sha256,
            "mjcf_sha256": binding.mjcf_sha256,
            "compiled_model_sha256": binding.compiled_model_sha256,
            "compiled_model_signature_semantics": (
                "canonical_mujoco_dynamics_gate_allowlist_v1; "
                "not_a_complete_compiled_mjb_hash"
            ),
            "complete_compiled_mjb_bound": False,
            "joint_mapping": "canonical_name_to_compiled_qpos_and_dof_address",
            "root_pose": "world_xyz_plus_wxyz_quaternion",
            "root_tangent": (
                "world_linear_plus_current_body_angular; "
                "omega_body=R_world.T@omega_world"
            ),
            "root_curvature": (
                "world_linear_plus_current_body_angular; "
                "alpha_body=R_world.T@alpha_world"
            ),
            "analytic_root_second_order": True,
            "finite_difference_generation_used": False,
            "maximum_ready_relative_root_angle_rad": max_relative_angle,
            "quaternion_norm_error_max": quaternion_norm_error,
            "qpos_sha256": _array_sha256(qpos_out),
            "q_s_sha256": _array_sha256(q_s_out),
            "q_ss_sha256": _array_sha256(q_ss_out),
            "sampled_candidate_only": True,
            "continuous_path_certificate": False,
            "dynamics_certificate": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        }
    )
    return MujocoPathJet(
        qpos=qpos_out,
        q_s=q_s_out,
        q_ss=q_ss_out,
        report=report,
    )


def _tolerance(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MujocoPathAdapterError(f"{label} must be a number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise MujocoPathAdapterError(
            f"{label} must be finite and non-negative"
        )
    return number


def validate_with_mj_differentiate_pos(
    mujoco_module: Any,
    model: Any,
    binding: ExactMujocoModelBinding,
    path: MujocoPathJet,
    *,
    path_parameter: Any,
    tangent_absolute_tolerance: float = 1.0e-4,
    tangent_relative_tolerance: float = 1.0e-3,
    curvature_absolute_tolerance: float = 2.0e-3,
    curvature_relative_tolerance: float = 2.0e-2,
) -> Mapping[str, Any]:
    """Sample-check an adapted 2-jet with ``mj_differentiatePos``.

    This helper is deliberately not called by ``adapt_canonical_path_jet``.
    It excludes endpoints and provides only a finite-spacing consistency check;
    its output is never substituted back into the path.
    """

    _assert_current_binding(mujoco_module, model, binding)
    if not isinstance(path, MujocoPathJet):
        raise MujocoPathAdapterError("path must be a MujocoPathJet")
    if path.report.get("model_binding_sha256") != binding.model_binding_sha256:
        raise MujocoPathAdapterError("path belongs to a different model binding")
    qpos = _finite_matrix(path.qpos, "path.qpos")
    q_s = _finite_matrix(path.q_s, "path.q_s")
    q_ss = _finite_matrix(path.q_ss, "path.q_ss")
    if (
        qpos.shape[1] != binding.nq
        or q_s.shape != (len(qpos), binding.nv)
        or q_ss.shape != q_s.shape
    ):
        raise MujocoPathAdapterError(
            "path qpos/q_s/q_ss dimensions do not match the bound model"
        )
    parameter = np.asarray(path_parameter)
    if (
        parameter.ndim != 1
        or parameter.dtype.kind not in "iuf"
        or parameter.dtype.kind == "b"
        or np.iscomplexobj(parameter)
    ):
        raise MujocoPathAdapterError(
            "path_parameter must be one real numeric vector"
        )
    parameter = np.ascontiguousarray(parameter, dtype=np.float64)
    if (
        len(parameter) != len(qpos)
        or len(parameter) < 3
        or not np.isfinite(parameter).all()
        or np.any(np.diff(parameter) <= 0.0)
    ):
        raise MujocoPathAdapterError(
            "path_parameter must be finite, strictly increasing, and match "
            "at least three path samples"
        )
    if not hasattr(mujoco_module, "mj_differentiatePos"):
        raise MujocoPathAdapterError(
            "MuJoCo module does not expose mj_differentiatePos"
        )

    tangent_atol = _tolerance(
        tangent_absolute_tolerance, "tangent_absolute_tolerance"
    )
    tangent_rtol = _tolerance(
        tangent_relative_tolerance, "tangent_relative_tolerance"
    )
    curvature_atol = _tolerance(
        curvature_absolute_tolerance, "curvature_absolute_tolerance"
    )
    curvature_rtol = _tolerance(
        curvature_relative_tolerance, "curvature_relative_tolerance"
    )
    max_tangent = 0.0
    max_curvature = 0.0
    failures: list[dict[str, Any]] = []
    worst_tangent: dict[str, Any] | None = None
    worst_curvature: dict[str, Any] | None = None
    for sample in range(1, len(qpos) - 1):
        h_backward = float(parameter[sample] - parameter[sample - 1])
        h_forward = float(parameter[sample + 1] - parameter[sample])
        backward_raw = np.empty(binding.nv, dtype=np.float64)
        forward = np.empty(binding.nv, dtype=np.float64)
        mujoco_module.mj_differentiatePos(
            model,
            backward_raw,
            h_backward,
            qpos[sample],
            qpos[sample - 1],
        )
        mujoco_module.mj_differentiatePos(
            model,
            forward,
            h_forward,
            qpos[sample],
            qpos[sample + 1],
        )
        if not np.isfinite(backward_raw).all() or not np.isfinite(forward).all():
            raise MujocoPathAdapterError(
                "mj_differentiatePos returned NaN/Inf"
            )
        backward = -backward_raw
        numeric_tangent = (
            h_forward * backward + h_backward * forward
        ) / (h_backward + h_forward)
        numeric_curvature = 2.0 * (forward - backward) / (
            h_backward + h_forward
        )

        tangent_error = np.abs(q_s[sample] - numeric_tangent)
        tangent_allowed = tangent_atol + tangent_rtol * np.maximum(
            np.abs(q_s[sample]), np.abs(numeric_tangent)
        )
        curvature_error = np.abs(q_ss[sample] - numeric_curvature)
        curvature_allowed = curvature_atol + curvature_rtol * np.maximum(
            np.abs(q_ss[sample]), np.abs(numeric_curvature)
        )
        tangent_dof = int(np.argmax(tangent_error))
        curvature_dof = int(np.argmax(curvature_error))
        tangent_peak = float(tangent_error[tangent_dof])
        curvature_peak = float(curvature_error[curvature_dof])
        if tangent_peak >= max_tangent:
            max_tangent = tangent_peak
            worst_tangent = {
                "sample": sample,
                "path_parameter": float(parameter[sample]),
                "dof": tangent_dof,
                "provided": float(q_s[sample, tangent_dof]),
                "finite_difference": float(numeric_tangent[tangent_dof]),
                "residual": tangent_peak,
                "allowed": float(tangent_allowed[tangent_dof]),
            }
        if curvature_peak >= max_curvature:
            max_curvature = curvature_peak
            worst_curvature = {
                "sample": sample,
                "path_parameter": float(parameter[sample]),
                "dof": curvature_dof,
                "provided": float(q_ss[sample, curvature_dof]),
                "finite_difference": float(numeric_curvature[curvature_dof]),
                "residual": curvature_peak,
                "allowed": float(curvature_allowed[curvature_dof]),
            }
        bad_tangent = np.flatnonzero(tangent_error > tangent_allowed)
        bad_curvature = np.flatnonzero(curvature_error > curvature_allowed)
        if len(bad_tangent) or len(bad_curvature):
            failures.append(
                {
                    "sample": sample,
                    "path_parameter": float(parameter[sample]),
                    "tangent_bad_dofs": bad_tangent.tolist(),
                    "curvature_bad_dofs": bad_curvature.tolist(),
                }
            )

    report = MappingProxyType(
        {
            "status": (
                "PASS_SAMPLED_MUJOCO_DIFFERENTIATE_POS"
                if not failures
                else "INCOMPLETE_FAIL_CLOSED"
            ),
            "method": (
                "interior_nonuniform_central_mj_differentiatePos_"
                "current_tangent_frame"
            ),
            "samples_total": int(len(qpos)),
            "interior_samples_checked": int(len(qpos) - 2),
            "endpoints_checked": False,
            "model_binding_sha256": binding.model_binding_sha256,
            "max_tangent_residual": max_tangent,
            "max_curvature_residual": max_curvature,
            "worst_tangent": worst_tangent,
            "worst_curvature": worst_curvature,
            "failures": tuple(failures),
            "finite_difference_used_for_generation": False,
            "sampled_consistency_only": True,
            "continuous_derivative_certificate": False,
            "dynamics_certificate": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        }
    )
    if failures:
        raise MujocoPathAdapterError(
            "analytic q_s/q_ss disagree with sampled mj_differentiatePos check",
            report=report,
        )
    return report


__all__ = [
    "ADAPTER_SEMANTICS",
    "EXPECTED_MJCF_MODEL_NAME",
    "FULL_COORDINATE_NAMES",
    "MujocoPathAdapterError",
    "MujocoPathJet",
    "ExactMujocoModelBinding",
    "PI_MARGIN_RAD",
    "ROOT_BODY_NAME",
    "ROOT_COORDINATE_NAMES",
    "ROOT_JOINT_NAME",
    "RUNTIME_JOINT_NAMES",
    "UPPER_COORDINATE_NAMES",
    "adapt_canonical_path_jet",
    "bind_exact_mujoco_model",
    "compiled_model_signature",
    "load_exact_mujoco_model",
    "validate_with_mj_differentiate_pos",
]
