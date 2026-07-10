"""Versioned point and body-column semantics for BeyondMimic motion NPZ kinematics.

Isaac Lab 2.1's historical array names are easy to misread:

* ``body_pos_w`` is the body LINK/actor origin position;
* ``body_lin_vel_w`` is the body CENTER-OF-MASS point velocity;
* ``body_quat_w`` and ``body_ang_vel_w`` describe the common link orientation
  and angular velocity (angular velocity is point-independent).

MotionCommand compares the stored linear velocity to Isaac's COM velocity, so
new files must preserve this convention even when MuJoCo supplies link poses.
Schema 2 additionally binds every body-array column to its articulation body
name.  Shape alone cannot distinguish two assets with different body orders.
This NumPy-only module is shared by converters, migration and tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


KINEMATICS_SCHEMA_VERSION = 2
BODY_POS_POINT = "link_origin"
BODY_LIN_VEL_POINT = "center_of_mass"
SCHEMA_KEY = "kinematics_schema_version"
POS_POINT_KEY = "body_pos_point"
LIN_VEL_POINT_KEY = "body_lin_vel_point"
BODY_NAMES_KEY = "body_names"
MIGRATION_SOURCE_SHA256_KEY = "kinematics_migration_source_sha256"
MIGRATION_SOURCE_POINT_KEY = "kinematics_migration_source_point"
MIGRATION_TOOL_KEY = "kinematics_migration_tool"

# Scalar provenance fields are not time-series channels.  Motion-rewrite tools
# may preserve them verbatim while replacing the four active schema fields to
# describe their newly generated arrays.
KINEMATICS_METADATA_KEYS = (
    SCHEMA_KEY,
    POS_POINT_KEY,
    LIN_VEL_POINT_KEY,
    BODY_NAMES_KEY,
    MIGRATION_SOURCE_SHA256_KEY,
    MIGRATION_SOURCE_POINT_KEY,
    MIGRATION_TOOL_KEY,
)


def _scalar_text(value) -> str:
    item = np.asarray(value).reshape(-1)
    if item.size != 1:
        raise ValueError(f"metadata value must be scalar, got shape {np.asarray(value).shape}")
    raw = item[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw)


@dataclass(frozen=True)
class KinematicsMetadata:
    schema_version: int | None
    body_pos_point: str | None
    body_lin_vel_point: str | None
    body_names: tuple[str, ...] | None

    @property
    def exact_motion_command_v2(self) -> bool:
        return (
            self.schema_version == KINEMATICS_SCHEMA_VERSION
            and self.body_pos_point == BODY_POS_POINT
            and self.body_lin_vel_point == BODY_LIN_VEL_POINT
            and self.body_names is not None
            and len(self.body_names) > 0
        )

def normalize_body_names(
    body_names: Sequence[str], *, expected_count: int | None = None
) -> tuple[str, ...]:
    """Return a validated immutable body-column order."""

    names = tuple(str(name) for name in body_names)
    if not names or any(not name for name in names):
        raise ValueError("body_names must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("body_names contains duplicates")
    if expected_count is not None and len(names) != int(expected_count):
        raise ValueError(
            f"body_names has {len(names)} entries, expected {int(expected_count)} body columns"
        )
    return names


def _read_body_names(value) -> tuple[str, ...]:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"body_names must be one-dimensional, got {raw.shape}")
    names = []
    for item in raw.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        names.append(str(item))
    return normalize_body_names(names)


def read_metadata(data) -> KinematicsMetadata:
    files = set(getattr(data, "files", data.keys()))
    core = {SCHEMA_KEY, POS_POINT_KEY, LIN_VEL_POINT_KEY}
    present = core & files
    if not present:
        if BODY_NAMES_KEY in files:
            raise ValueError("body_names exists without a kinematics schema declaration")
        return KinematicsMetadata(None, None, None, None)
    if present != core:
        raise ValueError(
            "partial motion kinematics metadata: "
            + ", ".join(sorted(core - present))
            + " missing"
        )
    try:
        schema_raw = np.asarray(data[SCHEMA_KEY]).reshape(-1)
        if schema_raw.size != 1:
            raise ValueError(
                f"{SCHEMA_KEY} must be scalar, got {np.asarray(data[SCHEMA_KEY]).shape}"
            )
        schema = int(schema_raw[0])
        pos = _scalar_text(data[POS_POINT_KEY])
        vel = _scalar_text(data[LIN_VEL_POINT_KEY])
        names = _read_body_names(data[BODY_NAMES_KEY]) if BODY_NAMES_KEY in files else None
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("malformed motion kinematics metadata") from exc
    if schema not in (1, KINEMATICS_SCHEMA_VERSION):
        raise ValueError(f"unsupported motion kinematics schema {schema}")
    if schema >= 2 and names is None:
        raise ValueError(f"kinematics schema {schema} requires body_names")
    return KinematicsMetadata(schema, pos, vel, names)


def body_names_from_data(data, *, expected_count: int | None = None) -> tuple[str, ...]:
    """Read the schema-bound body order, failing on legacy/unbound inputs."""

    metadata = read_metadata(data)
    if metadata.body_names is None:
        raise ValueError(
            "motion has no schema-bound body_names; migrate it with an explicit body-order file"
        )
    return normalize_body_names(metadata.body_names, expected_count=expected_count)


def resolve_body_names(
    data,
    *,
    explicit_body_names: Sequence[str] | None = None,
    expected_count: int | None = None,
) -> tuple[str, ...]:
    """Resolve an output body order and reject disagreement with source metadata."""

    metadata = read_metadata(data)
    embedded = (
        None
        if metadata.body_names is None
        else normalize_body_names(metadata.body_names, expected_count=expected_count)
    )
    explicit = (
        None
        if explicit_body_names is None
        else normalize_body_names(explicit_body_names, expected_count=expected_count)
    )
    if embedded is not None and explicit is not None and embedded != explicit:
        raise ValueError("explicit body order disagrees with source schema body_names")
    if explicit is not None:
        return explicit
    if embedded is not None:
        return embedded
    raise ValueError(
        "motion has no schema-bound body_names and no explicit body order was supplied"
    )


def metadata_arrays(
    *, body_names: Sequence[str], body_lin_vel_point: str = BODY_LIN_VEL_POINT
) -> dict[str, np.ndarray]:
    if body_lin_vel_point not in ("center_of_mass", "link_origin"):
        raise ValueError(f"unsupported body linear-velocity point {body_lin_vel_point!r}")
    names = normalize_body_names(body_names)
    return {
        SCHEMA_KEY: np.array([KINEMATICS_SCHEMA_VERSION], dtype=np.int64),
        POS_POINT_KEY: np.array(BODY_POS_POINT),
        LIN_VEL_POINT_KEY: np.array(body_lin_vel_point),
        BODY_NAMES_KEY: np.asarray(names),
    }


def quat_to_rot_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)
    w, x, y, z = (q[..., i] for i in range(4))
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[..., 0, 1] = 2.0 * (x * y - w * z)
    out[..., 0, 2] = 2.0 * (x * z + w * y)
    out[..., 1, 0] = 2.0 * (x * y + w * z)
    out[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[..., 1, 2] = 2.0 * (y * z - w * x)
    out[..., 2, 0] = 2.0 * (x * z - w * y)
    out[..., 2, 1] = 2.0 * (y * z + w * x)
    out[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


def link_velocity_to_com(
    body_link_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
    body_quat_w: np.ndarray,
    com_pos_b: np.ndarray,
) -> np.ndarray:
    """Convert link-origin velocity to COM-point velocity.

    ``com_pos_b`` is body-local link-origin -> COM, shape ``(B,3)``.  Other
    arrays have shape ``(T,B,*)``.
    """

    lin = np.asarray(body_link_lin_vel_w, dtype=np.float64)
    ang = np.asarray(body_ang_vel_w, dtype=np.float64)
    quat = np.asarray(body_quat_w, dtype=np.float64)
    com = np.asarray(com_pos_b, dtype=np.float64)
    if lin.ndim != 3 or lin.shape[-1] != 3:
        raise ValueError(f"body linear velocity must be (T,B,3), got {lin.shape}")
    if ang.shape != lin.shape or quat.shape != lin.shape[:2] + (4,) or com.shape != lin.shape[1:]:
        raise ValueError(
            f"kinematic shapes disagree: lin={lin.shape}, ang={ang.shape}, quat={quat.shape}, com={com.shape}"
        )
    offset_w = np.einsum("tbij,bj->tbi", quat_to_rot_wxyz(quat), com)
    return lin + np.cross(ang, offset_w)


def link_fd_signature(data, *, atol: float = 1.0e-4) -> dict:
    """Content-only detector for untagged files written as d(link_pos)/dt.

    It returns ``suspected_link_origin=True`` only when the stored field is
    numerically the finite difference of link positions *and* the clip has
    meaningful angular motion.  Static clips stay ``ambiguous`` rather than
    being falsely classified.
    """

    pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    lin = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
    ang = np.asarray(data["body_ang_vel_w"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if pos.shape != lin.shape or ang.shape != lin.shape or len(pos) < 2 or fps <= 0.0:
        raise ValueError("invalid motion arrays for velocity-point signature")
    link_fd = np.gradient(pos, 1.0 / fps, axis=0)
    max_abs = float(np.max(np.abs(lin - link_fd)))
    rms = float(np.sqrt(np.mean((lin - link_fd) ** 2)))
    max_ang = float(np.max(np.linalg.norm(ang, axis=-1)))
    moving = max_ang > 0.2
    return {
        "max_abs_mps": max_abs,
        "rms_mps": rms,
        "max_ang_radps": max_ang,
        "suspected_link_origin": bool(moving and max_abs <= float(atol)),
        "ambiguous": bool(not moving),
    }
