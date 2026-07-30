#!/usr/bin/env python3
"""Content-addressed continuous swept-clearance producer for ActionBall motions.

This producer is intentionally independent of ``canonical_motion_bank_gate.py``.
It proves a geometric statement only:

* every byte-pinned ``upper`` and ``full`` motion is interpreted from its first
  frame through its last frame;
* root translation and joint position are piecewise linear, while the root
  quaternion follows shortest-arc SLERP;
* every enabled collision geom in the pinned floating-root robot subtree,
  including the exact racket face and handle geoms, stays at least 5 mm from
  all five ActionBall world solids;
* the five solids are derived from the pinned runtime table sources and include
  the robot-only floor-to-slab-underside keepout.

The proof is not a dense-sampling claim.  For each source frame interval and
robot/obstacle pair, it recursively partitions continuous time.  At each leaf,
MuJoCo's ``mj_geomDistance`` saturation predicate proves a midpoint lower
bound.  A conservative Hausdorff displacement bound covers every time in the
leaf:

    root translation
    + root shortest-arc rotation * subtree reach
    + sum(ancestor hinge angle change * descendant reach).

Only a complete, gap-free proof ledger with zero UNKNOWN, NONFINITE, or UNSAFE
base pair-intervals can emit PASS.  Unsupported topology, missing geometry,
distance-oracle ambiguity, or subdivision exhaustion remains fail-closed.

No simulator is stepped and no input is modified.  The JSON receipt is written
outside the bank with atomic no-clobber publication.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
import math
import os
import platform
import stat
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
TABLE_TENNIS_DIR = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis"
)
TRACKING_MDP_DIR = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
TRACKING_CFG_DIR = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3"
)

DEFAULT_GEOMETRY_SOURCE = TABLE_TENNIS_DIR / "geometry.py"
DEFAULT_TABLE_FRAME_SOURCE = TABLE_TENNIS_DIR / "table_frame.py"
DEFAULT_HOPE_COMMANDS_SOURCE = TRACKING_MDP_DIR / "hope_commands.py"
DEFAULT_SCENE_BUILDER_SOURCE = TRACKING_CFG_DIR / "hope_env_cfg.py"
DEFAULT_JOINT_ORDER_SOURCE = SCRIPTS_DIR / "audit_motion_npz.py"

REQUESTED_SCOPES = ("upper", "full")
ACTION_BALL_ROLES = ("top", "keepout", "net", "post_left", "post_right")
OBSTACLE_GEOM_NAMES = {
    "top": "action_ball_table_top",
    "keepout": "action_ball_robot_keepout",
    "net": "action_ball_net",
    "post_left": "action_ball_net_post_left",
    "post_right": "action_ball_net_post_right",
}
RACKET_AND_HANDLE_GEOMS = (
    "right_racket_collision",
    "right_racket_handle_collision",
)
EXPECTED_ASSEMBLY_BOUNDARY_CONTACTS = frozenset(
    {
        frozenset(("top", "keepout")),
        frozenset(("top", "net")),
    }
)
EXPECTED_ASSEMBLY_INTERIOR_OVERLAPS = frozenset(
    {
        frozenset(("net", "post_left")),
        frozenset(("net", "post_right")),
    }
)

SCHEMA_VERSION = 1
REQUEST_CLASS = "action_ball_continuous_swept_clearance_request_v1"
RECEIPT_CLASS = "independent_continuous_swept_clearance_v1"
INTERNAL_RECEIPT_CLASS = (
    "trusted_action_ball_continuous_swept_clearance_evidence_v1"
)
PRODUCER_ID = "certify_action_ball_swept_clearance"
ALGORITHM_ID = "recursive_midpoint_hinged_chain_hausdorff_enclosure_v1"
SCENE_PROFILE = "action_ball_robot_keepout_v1"
INTERPOLATION_ID = (
    "root_translation_piecewise_linear__root_quaternion_shortest_arc_slerp__"
    "joint_position_piecewise_linear_v1"
)
DISTANCE_ORACLE_ID = (
    "mujoco_mj_geomDistance_distmax_saturation_predicate_no_epsilon_relaxation_v1"
)
ENCLOSURE_ID = (
    "midpoint_distance_minus_root_translation_root_rotation_and_ancestor_hinge_"
    "reach_hausdorff_bound_v1"
)
HARD_CLEARANCE_M = 0.005
KEEP_OUT_FLOOR_Z_M = 0.0
MAX_SUBDIVISION_DEPTH = 18
DISTANCE_QUERY_CAP_M = 10.0
QUATERNION_NORM_TOL = 2.0e-3
FK_POSITION_TOL_M = 1.0e-4
FK_ORIENTATION_TOL_RAD = 1.0e-4

# Exact public contract consumed by canonical_motion_bank_gate.py and reopened
# by canonical_motion_admission.py.  The producer keeps its larger proof ledger
# internally, validates it first, and projects only these normalized claims.
BANK_GATE_COVERAGE = "entire_prep_hit_recovery_continuous_time"
BANK_GATE_SUBJECTS = (
    "robot_collision_geoms",
    "racket_and_handle_geoms",
)
BANK_GATE_OBSTACLES = (
    "table_top",
    "table_edges",
    "table_underside",
    "action_ball_under_table_keepout",
    "net",
    "net_posts",
)

BASE_NPZ_FIELDS = frozenset(
    {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "kinematics_schema_version",
        "body_pos_point",
        "body_lin_vel_point",
        "body_names",
    }
)
MIGRATION_NPZ_FIELDS = frozenset(
    {
        "migration_source_sha256",
        "migration_source_body_pos_point",
        "migration_tool",
    }
)


class ClearanceError(ValueError):
    """Fail-closed input, identity, proof, or publication error."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    size: int
    device: int
    inode: int

    def binding(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class MotionClip:
    motion_id: str
    scope: str
    snapshot: FileSnapshot
    fps: float
    joint_pos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    contact_window_start_s: float
    contact_window_end_s: float

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.frames - 1) / self.fps


@dataclass(frozen=True)
class GeomEnvelope:
    name: str
    body_name: str
    root_rotation_reach_m: float
    joint_rotation_reach_m: tuple[float, ...]
    geom_rbound_m: float


@dataclass(frozen=True)
class KinematicBinding:
    root_body_id: int
    root_joint_id: int
    root_qpos_address: int
    joint_ids: tuple[int, ...]
    joint_qpos_addresses: tuple[int, ...]
    body_ids: tuple[int, ...]
    collision_geom_ids: tuple[int, ...]
    collision_geom_names: tuple[str, ...]
    racket_geom_names: tuple[str, ...]
    envelopes: tuple[GeomEnvelope, ...]


class ClearanceBackend(Protocol):
    """The small real/fake backend boundary used by the continuous proof."""

    robot_geometries: tuple[GeomEnvelope, ...]
    obstacle_roles: tuple[str, ...]

    def apply_pose(
        self,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        joint_position: np.ndarray,
    ) -> None: ...

    def distance_saturation_query(
        self, robot_geom_name: str, obstacle_role: str, distmax_m: float
    ) -> tuple[float, bool]: ...


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ClearanceError(f"value is not strict canonical JSON: {exc}") from exc


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ClearanceError(f"{label} must be a lowercase SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ClearanceError(f"{label} must be a lowercase 64-digit SHA-256")
    return normalized


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ClearanceError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ClearanceError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ClearanceError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ClearanceError(f"{label} must be >= {minimum}")
    return result


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ClearanceError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ClearanceError(f"{label} must be >= {minimum}")
    return result


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ClearanceError(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClearanceError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ClearanceError(f"{label} must be an array")
    return value


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ClearanceError(f"{label} contains forbidden JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ClearanceError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClearanceError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def read_snapshot(
    raw_path: os.PathLike[str] | str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> FileSnapshot:
    """Read one regular file through one O_NOFOLLOW descriptor."""

    path = Path(os.path.abspath(os.fspath(Path(raw_path).expanduser())))
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ClearanceError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ClearanceError(f"{label} must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ClearanceError(f"cannot open {label} {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ClearanceError(f"{label} descriptor is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ClearanceError(f"{label} changed while being read")
    payload = b"".join(chunks)
    if len(payload) != int(opened.st_size):
        raise ClearanceError(f"{label} short read")
    digest = _sha256_bytes(payload)
    if expected_sha256 is not None and digest != _digest(expected_sha256, f"{label} SHA-256"):
        raise ClearanceError(
            f"{label} SHA-256 mismatch: expected={expected_sha256} actual={digest}"
        )
    return FileSnapshot(
        path=path,
        data=payload,
        sha256=digest,
        size=len(payload),
        device=int(opened.st_dev),
        inode=int(opened.st_ino),
    )


def _resolve_repo_path(raw: Any, label: str) -> Path:
    value = Path(_nonempty_string(raw, label)).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return Path(os.path.abspath(os.fspath(value)))


def _exact_literal_assignment(
    source: FileSnapshot, *, class_name: str | None, names: Sequence[str]
) -> dict[str, Any]:
    """Read literal assignments from pinned Python source without importing it."""

    try:
        tree = ast.parse(source.data.decode("utf-8"), filename=str(source.path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ClearanceError(f"cannot parse pinned source {source.path}: {exc}") from exc
    scope: Sequence[ast.stmt] = tree.body
    if class_name is not None:
        classes = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise ClearanceError(
                f"pinned source must define exactly one class {class_name!r}"
            )
        scope = classes[0].body
    wanted = set(names)
    result: dict[str, Any] = {}
    for statement in scope:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        if (
            not isinstance(target, ast.Name)
            or target.id not in wanted
            or value is None
        ):
            continue
        if target.id in result:
            raise ClearanceError(f"duplicate literal assignment for {target.id}")
        try:
            result[target.id] = ast.literal_eval(value)
        except (TypeError, ValueError) as exc:
            raise ClearanceError(
                f"{target.id} in pinned source is not a literal"
            ) from exc
    missing = wanted - set(result)
    if missing:
        raise ClearanceError(f"pinned source is missing literal assignments {sorted(missing)}")
    return result


def load_runtime_joint_names(source: FileSnapshot) -> tuple[str, ...]:
    values = _exact_literal_assignment(
        source, class_name=None, names=("ISAAC_JOINT_NAMES",)
    )["ISAAC_JOINT_NAMES"]
    names = tuple(
        _nonempty_string(value, f"ISAAC_JOINT_NAMES[{index}]")
        for index, value in enumerate(_sequence(values, "ISAAC_JOINT_NAMES"))
    )
    if len(names) != 31 or len(set(names)) != 31:
        raise ClearanceError("runtime joint order must contain 31 unique names")
    return names


def read_body_order(snapshot: FileSnapshot) -> tuple[str, ...]:
    try:
        names = tuple(
            line.strip()
            for line in snapshot.data.decode("utf-8").splitlines()
            if line.strip()
        )
    except UnicodeDecodeError as exc:
        raise ClearanceError("runtime body order is not UTF-8") from exc
    if len(names) != 32 or len(set(names)) != 32:
        raise ClearanceError("runtime body order must contain 32 unique names")
    return names


def _exec_geometry_and_frame(
    geometry_source: FileSnapshot, table_frame_source: FileSnapshot
) -> tuple[types.ModuleType, types.ModuleType]:
    """Execute exactly snapshotted pure sources without importing Isaac."""

    geometry_name = "whole_body_tracking.tasks.table_tennis.geometry"
    frame_name = "whole_body_tracking.tasks.table_tennis.table_frame"
    geometry = types.ModuleType(geometry_name)
    geometry.__file__ = str(geometry_source.path)
    geometry.__package__ = "whole_body_tracking.tasks.table_tennis"
    saved = {
        name: sys.modules.get(name)
        for name in (
            "whole_body_tracking",
            "whole_body_tracking.tasks",
            "whole_body_tracking.tasks.table_tennis",
            geometry_name,
            frame_name,
        )
    }
    try:
        for package in (
            "whole_body_tracking",
            "whole_body_tracking.tasks",
            "whole_body_tracking.tasks.table_tennis",
        ):
            module = sys.modules.get(package)
            if module is None:
                module = types.ModuleType(package)
                module.__path__ = []  # type: ignore[attr-defined]
                sys.modules[package] = module
        sys.modules[geometry_name] = geometry
        setattr(sys.modules["whole_body_tracking.tasks.table_tennis"], "geometry", geometry)
        exec(
            compile(
                geometry_source.data.decode("utf-8"),
                str(geometry_source.path),
                "exec",
            ),
            geometry.__dict__,
        )
        frame = types.ModuleType(frame_name)
        frame.__file__ = str(table_frame_source.path)
        frame.__package__ = "whole_body_tracking.tasks.table_tennis"
        sys.modules[frame_name] = frame
        exec(
            compile(
                table_frame_source.data.decode("utf-8"),
                str(table_frame_source.path),
                "exec",
            ),
            frame.__dict__,
        )
    except Exception as exc:
        raise ClearanceError(
            f"cannot execute pinned geometry/table-frame source: {exc}"
        ) from exc
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return geometry, frame


def validate_action_ball_assembly_contacts(
    components: Sequence[Mapping[str, Any]],
) -> None:
    """Validate only the intentional contacts inside the five-piece assembly.

    The runtime boxes are not pairwise disjoint by design.  The keepout ends on
    the slab underside, the net starts on the slab top, and each post straddles
    one lateral end of the net.  Treating the two net/post joints as forbidden
    interior overlap makes the real runtime scene impossible to certify.

    This validator is deliberately stricter than a generic "allow these pair
    names" list.  It checks the exact seam geometry and rejects a shifted post,
    duplicated box, net intrusion into the slab, or any other new contact.
    """

    if len(components) != len(ACTION_BALL_ROLES):
        raise ClearanceError("ActionBall assembly must contain exactly five components")
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    geom_names: set[str] = set()
    for index, expected_role in enumerate(ACTION_BALL_ROLES):
        component = _mapping(components[index], f"assembly component {index}")
        if component.get("role") != expected_role:
            raise ClearanceError("ActionBall assembly component order/role changed")
        expected_geom = OBSTACLE_GEOM_NAMES[expected_role]
        if component.get("geom_name") != expected_geom or expected_geom in geom_names:
            raise ClearanceError("ActionBall assembly geom identity is duplicated or changed")
        geom_names.add(expected_geom)
        lo = np.asarray(component.get("aabb_lo_m"), dtype=np.float64)
        hi = np.asarray(component.get("aabb_hi_m"), dtype=np.float64)
        if (
            lo.shape != (3,)
            or hi.shape != (3,)
            or not np.isfinite(lo).all()
            or not np.isfinite(hi).all()
            or not np.all(hi > lo)
        ):
            raise ClearanceError(f"ActionBall {expected_role} AABB is invalid")
        bounds[expected_role] = (lo, hi)

    top_lo, top_hi = bounds["top"]
    keepout_lo, keepout_hi = bounds["keepout"]
    net_lo, net_hi = bounds["net"]
    left_lo, left_hi = bounds["post_left"]
    right_lo, right_hi = bounds["post_right"]

    if not (
        np.array_equal(top_lo[:2], keepout_lo[:2])
        and np.array_equal(top_hi[:2], keepout_hi[:2])
        and keepout_hi[2] == top_lo[2]
    ):
        raise ClearanceError(
            "ActionBall keepout is not the exact floor-to-slab-underside volume "
            "with the top's shared horizontal footprint"
        )
    if not (
        net_lo[2] == top_hi[2]
        and np.all(np.minimum(top_hi[:2], net_hi[:2]) > np.maximum(top_lo[:2], net_lo[:2]))
    ):
        raise ClearanceError("ActionBall net must meet, not intrude into, the slab top")

    # The two post boxes conservatively straddle the net endpoints.  The net is
    # fully nested inside each post in x and z over the joint, while exactly
    # half of each post's y width lies inside the net.
    if not (
        np.array_equal(left_lo[[0, 2]], right_lo[[0, 2]])
        and np.array_equal(left_hi[[0, 2]], right_hi[[0, 2]])
        and left_lo[1] == -right_hi[1]
        and left_hi[1] == -right_lo[1]
        and net_lo[1] == -net_hi[1]
    ):
        raise ClearanceError("ActionBall net posts lost their exact mirrored geometry")
    for role, post_lo, post_hi, endpoint in (
        ("post_left", left_lo, left_hi, net_hi[1]),
        ("post_right", right_lo, right_hi, net_lo[1]),
    ):
        if not (
            post_lo[0] < net_lo[0] < net_hi[0] < post_hi[0]
            and post_lo[2] == net_lo[2]
            and post_hi[2] > net_hi[2]
            and post_lo[1] < endpoint < post_hi[1]
            and endpoint - post_lo[1] == post_hi[1] - endpoint
        ):
            raise ClearanceError(
                f"ActionBall {role} does not form the exact conservative net joint"
            )

    roles_and_bounds = tuple(
        (role, *bounds[role]) for role in ACTION_BALL_ROLES
    )
    for index, (left_role, left_lo, left_hi) in enumerate(roles_and_bounds):
        for right_role, right_lo, right_hi in roles_and_bounds[index + 1 :]:
            pair = frozenset((left_role, right_role))
            signed_overlap = np.minimum(left_hi, right_hi) - np.maximum(
                left_lo, right_lo
            )
            interior = bool(np.all(signed_overlap > 0.0))
            boundary = bool(
                np.all(signed_overlap >= 0.0)
                and np.any(signed_overlap == 0.0)
            )
            if interior:
                if pair not in EXPECTED_ASSEMBLY_INTERIOR_OVERLAPS:
                    raise ClearanceError(
                        "ActionBall assembly has an unapproved interior overlap: "
                        f"{left_role}/{right_role}"
                    )
            elif boundary:
                if pair not in EXPECTED_ASSEMBLY_BOUNDARY_CONTACTS:
                    raise ClearanceError(
                        "ActionBall assembly has an unapproved boundary contact: "
                        f"{left_role}/{right_role}"
                    )
            elif (
                pair in EXPECTED_ASSEMBLY_BOUNDARY_CONTACTS
                or pair in EXPECTED_ASSEMBLY_INTERIOR_OVERLAPS
            ):
                raise ClearanceError(
                    "ActionBall assembly lost an expected structural contact: "
                    f"{left_role}/{right_role}"
                )


def derive_action_ball_assembly(
    *,
    geometry_source: FileSnapshot,
    table_frame_source: FileSnapshot,
    hope_commands_source: FileSnapshot,
    scene_builder_source: FileSnapshot,
) -> dict[str, Any]:
    """Derive the exact runtime five-piece AABB assembly from pinned sources."""

    del scene_builder_source  # Its bytes are a mandatory pin recorded by the caller.
    defaults = _exact_literal_assignment(
        hope_commands_source,
        class_name="RacketTargetCommandCfg",
        names=("vb_table_near_x", "vb_table_surface_z"),
    )
    near_x = _finite(defaults["vb_table_near_x"], "vb_table_near_x")
    surface_z = _finite(defaults["vb_table_surface_z"], "vb_table_surface_z")
    geometry, table_frame = _exec_geometry_and_frame(
        geometry_source, table_frame_source
    )
    roles = tuple(getattr(table_frame, "TABLE_ASSEMBLY_ROLES", ()))
    if roles != ACTION_BALL_ROLES:
        raise ClearanceError(
            f"runtime table assembly roles changed: expected={ACTION_BALL_ROLES} actual={roles}"
        )
    try:
        raw_aabbs = table_frame.table_assembly_aabbs_env(
            near_x,
            surface_z,
            keepout_floor_z=KEEP_OUT_FLOOR_Z_M,
            margin=0.0,
        )
    except Exception as exc:
        raise ClearanceError(f"cannot derive ActionBall table assembly: {exc}") from exc
    if len(raw_aabbs) != len(ACTION_BALL_ROLES):
        raise ClearanceError("runtime table assembly did not produce exactly five AABBs")

    components: list[dict[str, Any]] = []
    for role, raw in zip(ACTION_BALL_ROLES, raw_aabbs):
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise ClearanceError(f"ActionBall {role} AABB is malformed")
        lo = np.asarray(raw[0], dtype=np.float64)
        hi = np.asarray(raw[1], dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,) or not (
            np.isfinite(lo).all() and np.isfinite(hi).all()
        ):
            raise ClearanceError(f"ActionBall {role} AABB is non-finite")
        if not np.all(hi > lo):
            raise ClearanceError(f"ActionBall {role} AABB has non-positive extent")
        center = (lo + hi) / 2.0
        extents = hi - lo
        components.append(
            {
                "role": role,
                "geom_name": OBSTACLE_GEOM_NAMES[role],
                "center_m": center.tolist(),
                "full_extents_m": extents.tolist(),
                "aabb_lo_m": lo.tolist(),
                "aabb_hi_m": hi.tolist(),
            }
        )

    keepout = components[1]
    top = components[0]
    thickness = _finite(
        getattr(geometry, "TABLE_THICKNESS", None), "geometry.TABLE_THICKNESS"
    )
    keepout_lo = np.asarray(keepout["aabb_lo_m"], dtype=np.float64)
    keepout_hi = np.asarray(keepout["aabb_hi_m"], dtype=np.float64)
    top_lo = np.asarray(top["aabb_lo_m"], dtype=np.float64)
    if (
        keepout_lo[2] != KEEP_OUT_FLOOR_Z_M
        or keepout_hi[2] != surface_z - thickness
        or keepout_hi[2] != top_lo[2]
    ):
        raise ClearanceError(
            "ActionBall keepout must fill exactly floor-to-slab-underside"
        )
    validate_action_ball_assembly_contacts(components)
    return {
        "scene_profile": SCENE_PROFILE,
        "with_table": True,
        "near_x_m": near_x,
        "surface_z_m": surface_z,
        "keepout_floor_z_m": KEEP_OUT_FLOOR_Z_M,
        "action_ball_keepout_semantics": "robot_only_keepout_ball_excluded",
        "roles": list(ACTION_BALL_ROLES),
        "components": components,
        "components_sha256": _canonical_json_sha256(components),
    }


def _scalar_text(value: Any, label: str) -> str:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ClearanceError(f"{label} must be scalar")
    item = array[0]
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ClearanceError(f"{label} is not UTF-8") from exc
    return str(item)


def load_motion_snapshot(
    *,
    snapshot: FileSnapshot,
    motion_id: str,
    scope: str,
    body_names: tuple[str, ...],
    contact_window_start_s: float,
    contact_window_end_s: float,
) -> MotionClip:
    try:
        archive = np.load(io.BytesIO(snapshot.data), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ClearanceError(f"cannot parse motion NPZ {snapshot.path}: {exc}") from exc
    with archive as payload:
        fields = frozenset(payload.files)
        if fields not in (BASE_NPZ_FIELDS, BASE_NPZ_FIELDS | MIGRATION_NPZ_FIELDS):
            raise ClearanceError(
                f"motion {motion_id}/{scope} field set is not exact schema-2"
            )
        if _integer(
            int(np.asarray(payload["kinematics_schema_version"]).reshape(-1)[0]),
            "kinematics_schema_version",
        ) != 2:
            raise ClearanceError("motion kinematics_schema_version must be 2")
        if _scalar_text(payload["body_pos_point"], "body_pos_point") != "link_origin":
            raise ClearanceError("motion body_pos_point must be link_origin")
        stored_body_names = tuple(
            str(value)
            for value in np.asarray(payload["body_names"]).reshape(-1).tolist()
        )
        if stored_body_names != body_names:
            raise ClearanceError("motion body_names do not match pinned runtime order")
        fps_values = np.asarray(payload["fps"], dtype=np.float64).reshape(-1)
        if fps_values.size != 1:
            raise ClearanceError("motion fps must contain one scalar")
        fps = _finite(fps_values[0], "motion fps", minimum=0.0)
        if fps == 0.0:
            raise ClearanceError("motion fps must be positive")
        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 31 or joint_pos.shape[0] < 2:
            raise ClearanceError("motion joint_pos must be (T>=2,31)")
        frames = int(joint_pos.shape[0])
        body_pos = np.asarray(payload["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(payload["body_quat_w"], dtype=np.float64)
        expected_shapes = {
            "joint_vel": (frames, 31),
            "body_pos_w": (frames, 32, 3),
            "body_quat_w": (frames, 32, 4),
            "body_lin_vel_w": (frames, 32, 3),
            "body_ang_vel_w": (frames, 32, 3),
        }
        for key, shape in expected_shapes.items():
            array = np.asarray(payload[key], dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ClearanceError(f"motion {key} must be finite with shape {shape}")
        if not np.isfinite(joint_pos).all():
            raise ClearanceError("motion joint_pos contains NaN/Inf")
        norms = np.linalg.norm(body_quat, axis=-1)
        if float(np.max(np.abs(norms - 1.0))) > QUATERNION_NORM_TOL:
            raise ClearanceError("motion body_quat_w contains non-unit quaternions")
    contact_start = _finite(
        contact_window_start_s, "contact_window_start_s", minimum=0.0
    )
    contact_end = _finite(
        contact_window_end_s, "contact_window_end_s", minimum=0.0
    )
    duration = (frames - 1) / fps
    one_frame = 1.0 / fps
    if not (
        one_frame <= contact_start <= contact_end <= duration - one_frame
    ):
        raise ClearanceError(
            "contact window must leave at least one complete preparation frame "
            "and one complete recovery frame"
        )
    return MotionClip(
        motion_id=_nonempty_string(motion_id, "motion_id"),
        scope=_nonempty_string(scope, "scope"),
        snapshot=snapshot,
        fps=fps,
        joint_pos=joint_pos,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        contact_window_start_s=contact_start,
        contact_window_end_s=contact_end,
    )


def _quaternion_normalize_wxyz(value: np.ndarray, label: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.isfinite(quaternion).all():
        raise ClearanceError(f"{label} quaternion is non-finite")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0 or not math.isfinite(norm):
        raise ClearanceError(f"{label} quaternion has invalid norm")
    return quaternion / norm


def shortest_arc_angle_rad(q0: np.ndarray, q1: np.ndarray) -> float:
    a = _quaternion_normalize_wxyz(q0, "q0")
    b = _quaternion_normalize_wxyz(q1, "q1")
    dot = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Constant-angular-speed shortest-arc SLERP."""

    u = _finite(fraction, "SLERP fraction")
    if not 0.0 <= u <= 1.0:
        raise ClearanceError("SLERP fraction must be in [0,1]")
    a = _quaternion_normalize_wxyz(q0, "q0")
    b = _quaternion_normalize_wxyz(q1, "q1")
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 1.0 - 1.0e-12:
        value = (1.0 - u) * a + u * b
        return _quaternion_normalize_wxyz(value, "SLERP near-identity")
    angle = math.acos(dot)
    denominator = math.sin(angle)
    value = (
        math.sin((1.0 - u) * angle) * a + math.sin(u * angle) * b
    ) / denominator
    return _quaternion_normalize_wxyz(value, "SLERP result")


def interpolate_pose(
    clip: MotionClip, frame: int, fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame < 0 or frame >= clip.frames - 1:
        raise ClearanceError("interpolation frame interval is out of range")
    u = _finite(fraction, "interpolation fraction")
    if not 0.0 <= u <= 1.0:
        raise ClearanceError("interpolation fraction must be in [0,1]")
    root0 = clip.body_pos_w[frame, 0]
    root1 = clip.body_pos_w[frame + 1, 0]
    joint0 = clip.joint_pos[frame]
    joint1 = clip.joint_pos[frame + 1]
    return (
        (1.0 - u) * root0 + u * root1,
        slerp_wxyz(
            clip.body_quat_w[frame, 0],
            clip.body_quat_w[frame + 1, 0],
            u,
        ),
        (1.0 - u) * joint0 + u * joint1,
    )


def interval_motion_bound_m(
    clip: MotionClip,
    frame: int,
    u_lo: float,
    u_hi: float,
    envelope: GeomEnvelope,
) -> float:
    """Hausdorff displacement from the leaf midpoint to any leaf time."""

    lo = _finite(u_lo, "u_lo")
    hi = _finite(u_hi, "u_hi")
    if not 0.0 <= lo < hi <= 1.0:
        raise ClearanceError("leaf interval must satisfy 0 <= u_lo < u_hi <= 1")
    joint_reaches = np.asarray(
        envelope.joint_rotation_reach_m, dtype=np.float64
    )
    if joint_reaches.shape != (31,) or not (
        np.isfinite(joint_reaches).all() and np.all(joint_reaches >= 0.0)
    ):
        raise ClearanceError("geom joint reach vector is invalid")
    root_delta = float(
        np.linalg.norm(
            clip.body_pos_w[frame + 1, 0] - clip.body_pos_w[frame, 0]
        )
    )
    root_angle = shortest_arc_angle_rad(
        clip.body_quat_w[frame, 0],
        clip.body_quat_w[frame + 1, 0],
    )
    joint_delta = np.abs(clip.joint_pos[frame + 1] - clip.joint_pos[frame])
    full_parameter_path_bound = (
        root_delta
        + envelope.root_rotation_reach_m * root_angle
        + float(np.dot(joint_reaches, joint_delta))
    )
    bound = 0.5 * (hi - lo) * full_parameter_path_bound
    if not math.isfinite(bound) or bound < 0.0:
        raise ClearanceError("computed interval motion bound is invalid")
    return bound


def clearance_threshold_passes(lower_bound_m: float) -> bool:
    """The exact no-epsilon hard predicate used by proof and tests."""

    return math.isfinite(lower_bound_m) and lower_bound_m >= HARD_CLEARANCE_M


def distance_query_is_saturated(observed_m: float, distmax_m: float) -> bool:
    """Return only MuJoCo's capped-distance lower-bound predicate.

    ``mj_geomDistance`` is known to return unreliable *unsaturated* mesh
    distances for some pairs.  The only value used by this proof is the binary
    fact that a query reached its exact ``distmax`` cap, which proves
    ``true_distance >= distmax``.  No epsilon relaxation and no unsaturated
    distance estimate enters a certificate.
    """

    cap = _finite(distmax_m, "distance query distmax", minimum=0.0)
    try:
        observed = float(observed_m)
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and observed >= cap


def _validate_backend_saturation(
    observed_m: float, distmax_m: float, declared_saturated: Any
) -> bool:
    """Reject a backend whose flag contradicts the capped return value."""

    if not isinstance(declared_saturated, (bool, np.bool_)):
        raise ClearanceError("distance backend saturation flag is not boolean")
    predicate = distance_query_is_saturated(observed_m, distmax_m)
    if bool(declared_saturated) != predicate:
        raise ClearanceError(
            "distance backend saturation flag contradicts mj_geomDistance cap"
        )
    return predicate


def _leaf_record(
    *,
    frame: int,
    u_lo: float,
    u_hi: float,
    depth: int,
    geom: GeomEnvelope,
    obstacle_role: str,
    motion_bound_m: float | None,
    query_distmax_m: float | None,
    query_observed_m: float | None,
    query_saturated: bool,
    status: str,
) -> dict[str, Any]:
    interval_lower = (
        HARD_CLEARANCE_M
        if status == "CERTIFIED"
        else None
    )
    return {
        "source_interval": frame,
        "u_lo": u_lo,
        "u_hi": u_hi,
        "depth": depth,
        "robot_geom": geom.name,
        "obstacle_role": obstacle_role,
        "subject_groups": (
            ["robot_collision_geoms", "racket_and_handle_geoms"]
            if geom.name in RACKET_AND_HANDLE_GEOMS
            else ["robot_collision_geoms"]
        ),
        "motion_displacement_upper_bound_m": motion_bound_m,
        "distance_query_distmax_m": query_distmax_m,
        "distance_query_observed_m": query_observed_m,
        "distance_query_saturated": query_saturated,
        "interval_clearance_certified_lower_bound_m": interval_lower,
        "status": status,
    }


def certify_motion_continuous(
    clip: MotionClip,
    backend: ClearanceBackend,
    *,
    max_subdivision_depth: int = MAX_SUBDIVISION_DEPTH,
) -> dict[str, Any]:
    """Produce a complete continuous-time proof ledger for one motion output."""

    max_depth = _integer(
        max_subdivision_depth, "max_subdivision_depth", minimum=0
    )
    geometries = tuple(backend.robot_geometries)
    obstacles = tuple(backend.obstacle_roles)
    if not geometries or len({geom.name for geom in geometries}) != len(geometries):
        raise ClearanceError("backend robot geometry set must be non-empty and unique")
    if tuple(obstacles) != ACTION_BALL_ROLES:
        raise ClearanceError("backend must expose the exact five ActionBall obstacles")
    if not set(RACKET_AND_HANDLE_GEOMS).issubset({geom.name for geom in geometries}):
        raise ClearanceError("backend robot geometry set lost racket face or handle")

    ledger: list[dict[str, Any]] = []
    pair_order = tuple(
        (geom, obstacle) for geom in geometries for obstacle in obstacles
    )

    for frame in range(clip.frames - 1):
        def walk(
            u_lo: float,
            u_hi: float,
            depth: int,
            unresolved: tuple[tuple[GeomEnvelope, str], ...],
        ) -> None:
            midpoint = 0.5 * (u_lo + u_hi)
            root_pos, root_quat, joint_pos = interpolate_pose(
                clip, frame, midpoint
            )
            backend.apply_pose(root_pos, root_quat, joint_pos)
            split: list[tuple[GeomEnvelope, str]] = []
            pending_rows: list[
                tuple[GeomEnvelope, str, float, float, float | None, bool]
            ] = []
            for geom, obstacle in unresolved:
                try:
                    motion_bound = interval_motion_bound_m(
                        clip, frame, u_lo, u_hi, geom
                    )
                    required_midpoint_clearance = (
                        HARD_CLEARANCE_M + motion_bound
                    )
                    if not math.isfinite(required_midpoint_clearance):
                        ledger.append(
                            _leaf_record(
                                frame=frame,
                                u_lo=u_lo,
                                u_hi=u_hi,
                                depth=depth,
                                geom=geom,
                                obstacle_role=obstacle,
                                motion_bound_m=motion_bound,
                                query_distmax_m=required_midpoint_clearance,
                                query_observed_m=None,
                                query_saturated=False,
                                status="NONFINITE",
                            )
                        )
                        continue
                    if required_midpoint_clearance > DISTANCE_QUERY_CAP_M:
                        pending_rows.append(
                            (
                                geom,
                                obstacle,
                                motion_bound,
                                required_midpoint_clearance,
                                None,
                                False,
                            )
                        )
                        split.append((geom, obstacle))
                        continue
                    observed, saturated = backend.distance_saturation_query(
                        geom.name,
                        obstacle,
                        required_midpoint_clearance,
                    )
                    saturated = _validate_backend_saturation(
                        observed, required_midpoint_clearance, saturated
                    )
                except Exception:
                    ledger.append(
                        _leaf_record(
                            frame=frame,
                            u_lo=u_lo,
                            u_hi=u_hi,
                            depth=depth,
                            geom=geom,
                            obstacle_role=obstacle,
                            motion_bound_m=None,
                            query_distmax_m=None,
                            query_observed_m=None,
                            query_saturated=False,
                            status="NONFINITE",
                        )
                    )
                    continue
                if not math.isfinite(observed):
                    ledger.append(
                        _leaf_record(
                            frame=frame,
                            u_lo=u_lo,
                            u_hi=u_hi,
                            depth=depth,
                            geom=geom,
                            obstacle_role=obstacle,
                            motion_bound_m=motion_bound,
                            query_distmax_m=required_midpoint_clearance,
                            query_observed_m=None,
                            query_saturated=False,
                            status="NONFINITE",
                        )
                    )
                elif saturated:
                    ledger.append(
                        _leaf_record(
                            frame=frame,
                            u_lo=u_lo,
                            u_hi=u_hi,
                            depth=depth,
                            geom=geom,
                            obstacle_role=obstacle,
                            motion_bound_m=motion_bound,
                            query_distmax_m=required_midpoint_clearance,
                            query_observed_m=observed,
                            query_saturated=True,
                            status="CERTIFIED",
                        )
                    )
                else:
                    pending_rows.append(
                        (
                            geom,
                            obstacle,
                            motion_bound,
                            required_midpoint_clearance,
                            observed,
                            False,
                        )
                    )
                    split.append((geom, obstacle))
            if not split:
                return
            if depth < max_depth:
                middle = 0.5 * (u_lo + u_hi)
                next_pairs = tuple(split)
                walk(u_lo, middle, depth + 1, next_pairs)
                walk(middle, u_hi, depth + 1, next_pairs)
                return

            # A failed continuous enclosure at maximum depth is never promoted.
            # Distinguish an observed midpoint violation from unresolved motion.
            for (
                geom,
                obstacle,
                motion_bound,
                required_midpoint_clearance,
                observed,
                _saturated,
            ) in pending_rows:
                try:
                    threshold_observed, threshold_saturated = (
                        backend.distance_saturation_query(
                            geom.name, obstacle, HARD_CLEARANCE_M
                        )
                    )
                    threshold_saturated = _validate_backend_saturation(
                        threshold_observed,
                        HARD_CLEARANCE_M,
                        threshold_saturated,
                    )
                except Exception:
                    threshold_observed = math.nan
                    threshold_saturated = False
                if not math.isfinite(threshold_observed):
                    status = "NONFINITE"
                    threshold_value: float | None = None
                elif threshold_saturated:
                    status = "UNKNOWN"
                    threshold_value = threshold_observed
                else:
                    status = "UNSAFE"
                    threshold_value = threshold_observed
                ledger.append(
                    _leaf_record(
                        frame=frame,
                        u_lo=u_lo,
                        u_hi=u_hi,
                        depth=depth,
                        geom=geom,
                        obstacle_role=obstacle,
                        motion_bound_m=motion_bound,
                        query_distmax_m=(
                            HARD_CLEARANCE_M
                            if status != "UNKNOWN"
                            else required_midpoint_clearance
                        ),
                        query_observed_m=(
                            threshold_value if status != "UNKNOWN" else observed
                        ),
                        query_saturated=False,
                        status=status,
                    )
                )

        walk(0.0, 1.0, 0, pair_order)

    ledger.sort(
        key=lambda row: (
            int(row["source_interval"]),
            str(row["robot_geom"]),
            str(row["obstacle_role"]),
            float(row["u_lo"]),
            float(row["u_hi"]),
        )
    )
    result = {
        "motion_id": clip.motion_id,
        "scope": clip.scope,
        "filename": clip.snapshot.path.name,
        "sha256": clip.snapshot.sha256,
        "frames": clip.frames,
        "fps": clip.fps,
        "duration_s": clip.duration_s,
        "start_frame": 0,
        "end_frame": clip.frames - 1,
        "interval_count": clip.frames - 1,
        "contact_window_start_s": clip.contact_window_start_s,
        "contact_window_end_s": clip.contact_window_end_s,
        "complete_cycle": True,
        "with_table": True,
        "interpolation": INTERPOLATION_ID,
        "robot_geom_names": [geom.name for geom in geometries],
        "obstacle_roles": list(obstacles),
        "proof_ledger": ledger,
        "proof_ledger_sha256": _canonical_json_sha256(ledger),
    }
    summary = validate_motion_proof_result(result)
    result["summary"] = summary
    result["verdict"] = (
        "PASS" if summary["all_base_pair_intervals_certified"] else "FAIL_CLOSED"
    )
    return result


def _group_leaf_coverage(
    result: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, str, str], list[Mapping[str, Any]]],
    tuple[str, ...],
    tuple[str, ...],
    int,
]:
    frames = _integer(result.get("frames"), "result.frames", minimum=2)
    geom_names = tuple(
        _nonempty_string(value, f"robot_geom_names[{index}]")
        for index, value in enumerate(
            _sequence(result.get("robot_geom_names"), "robot_geom_names")
        )
    )
    obstacle_roles = tuple(
        _nonempty_string(value, f"obstacle_roles[{index}]")
        for index, value in enumerate(
            _sequence(result.get("obstacle_roles"), "obstacle_roles")
        )
    )
    if not geom_names or len(set(geom_names)) != len(geom_names):
        raise ClearanceError("result robot geom set is empty or duplicated")
    if obstacle_roles != ACTION_BALL_ROLES:
        raise ClearanceError("result obstacle set is not the exact five-piece assembly")
    if not set(RACKET_AND_HANDLE_GEOMS).issubset(geom_names):
        raise ClearanceError("result lost racket face or handle coverage")
    ledger = _sequence(result.get("proof_ledger"), "proof_ledger")
    if _canonical_json_sha256(ledger) != _digest(
        result.get("proof_ledger_sha256"), "proof_ledger_sha256"
    ):
        raise ClearanceError("proof ledger digest mismatch")
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for index, raw in enumerate(ledger):
        row = _mapping(raw, f"proof_ledger[{index}]")
        key = (
            _integer(
                row.get("source_interval"),
                f"proof_ledger[{index}].source_interval",
            ),
            _nonempty_string(
                row.get("robot_geom"), f"proof_ledger[{index}].robot_geom"
            ),
            _nonempty_string(
                row.get("obstacle_role"), f"proof_ledger[{index}].obstacle_role"
            ),
        )
        if not (
            0 <= key[0] < frames - 1
            and key[1] in geom_names
            and key[2] in obstacle_roles
        ):
            raise ClearanceError(f"proof ledger row {index} leaves declared coverage")
        grouped.setdefault(key, []).append(row)
    return grouped, geom_names, obstacle_roles, frames


def validate_motion_proof_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute completeness/status summaries from the full proof ledger."""

    grouped, geom_names, obstacle_roles, frames = _group_leaf_coverage(result)
    expected_keys = {
        (frame, geom, obstacle)
        for frame in range(frames - 1)
        for geom in geom_names
        for obstacle in obstacle_roles
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise ClearanceError(
            f"proof ledger base coverage mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    status_counts = {
        "CERTIFIED": 0,
        "UNKNOWN": 0,
        "UNSAFE": 0,
        "NONFINITE": 0,
    }
    leaf_counts = {key: 0 for key in status_counts}
    per_obstacle = {
        role: {
            "required": 0,
            "certified": 0,
            "unknown": 0,
            "unsafe": 0,
            "nonfinite": 0,
        }
        for role in obstacle_roles
    }
    per_subject = {
        "robot_collision_geoms": {
            "required": 0,
            "certified": 0,
            "unknown": 0,
            "unsafe": 0,
            "nonfinite": 0,
        },
        "racket_and_handle_geoms": {
            "required": 0,
            "certified": 0,
            "unknown": 0,
            "unsafe": 0,
            "nonfinite": 0,
        },
    }
    minimum_lower = math.inf
    minimum_witness: dict[str, Any] | None = None

    for key in sorted(grouped):
        leaves = sorted(
            grouped[key],
            key=lambda row: (float(row["u_lo"]), float(row["u_hi"])),
        )
        cursor = 0.0
        base_statuses: set[str] = set()
        for row in leaves:
            lo = _finite(row.get("u_lo"), "leaf.u_lo")
            hi = _finite(row.get("u_hi"), "leaf.u_hi")
            _integer(row.get("depth"), "leaf.depth", minimum=0)
            if lo != cursor or not cursor <= lo < hi <= 1.0:
                raise ClearanceError(f"proof ledger has a gap/overlap for {key}")
            cursor = hi
            expected_subject_groups = (
                ["robot_collision_geoms", "racket_and_handle_geoms"]
                if key[1] in RACKET_AND_HANDLE_GEOMS
                else ["robot_collision_geoms"]
            )
            if row.get("subject_groups") != expected_subject_groups:
                raise ClearanceError("proof leaf subject-group coverage changed")
            status = _nonempty_string(row.get("status"), "leaf.status")
            if status not in status_counts:
                raise ClearanceError(f"unknown leaf status {status!r}")
            base_statuses.add(status)
            leaf_counts[status] += 1
            motion_bound = row.get("motion_displacement_upper_bound_m")
            distmax = row.get("distance_query_distmax_m")
            if status != "NONFINITE":
                _finite(motion_bound, "leaf motion bound", minimum=0.0)
                _finite(distmax, "leaf distance query", minimum=0.0)
            if status == "CERTIFIED":
                if row.get("distance_query_saturated") is not True:
                    raise ClearanceError("CERTIFIED leaf lacks a saturated distance query")
                observed = _finite(
                    row.get("distance_query_observed_m"),
                    "CERTIFIED leaf observed distance",
                )
                required = _finite(distmax, "CERTIFIED leaf distmax")
                if observed < required:
                    raise ClearanceError("CERTIFIED leaf contradicts distance saturation")
                lower = _finite(
                    row.get("interval_clearance_certified_lower_bound_m"),
                    "CERTIFIED leaf lower bound",
                )
                if not clearance_threshold_passes(lower):
                    raise ClearanceError("CERTIFIED leaf weakens the exact 5 mm threshold")
                if lower < minimum_lower:
                    minimum_lower = lower
                    minimum_witness = {
                        "source_interval": key[0],
                        "robot_geom": key[1],
                        "obstacle_role": key[2],
                        "u_lo": lo,
                        "u_hi": hi,
                        "lower_bound_m": lower,
                    }
            elif row.get("interval_clearance_certified_lower_bound_m") is not None:
                raise ClearanceError("non-certified leaf may not publish a clearance bound")
        if cursor != 1.0:
            raise ClearanceError(f"proof ledger does not end at u=1 for {key}")
        if base_statuses == {"CERTIFIED"}:
            base_status = "CERTIFIED"
        elif "NONFINITE" in base_statuses:
            base_status = "NONFINITE"
        elif "UNSAFE" in base_statuses:
            base_status = "UNSAFE"
        else:
            base_status = "UNKNOWN"
        status_counts[base_status] += 1
        obstacle_row = per_obstacle[key[2]]
        obstacle_row["required"] += 1
        obstacle_row[base_status.lower()] += 1
        subject_names = ["robot_collision_geoms"]
        if key[1] in RACKET_AND_HANDLE_GEOMS:
            subject_names.append("racket_and_handle_geoms")
        for subject in subject_names:
            subject_row = per_subject[subject]
            subject_row["required"] += 1
            subject_row[base_status.lower()] += 1

    required = len(expected_keys)
    all_certified = (
        status_counts["CERTIFIED"] == required
        and status_counts["UNKNOWN"] == 0
        and status_counts["UNSAFE"] == 0
        and status_counts["NONFINITE"] == 0
    )
    return {
        "required_base_pair_interval_count": required,
        "certified_base_pair_interval_count": status_counts["CERTIFIED"],
        "unknown_base_pair_interval_count": status_counts["UNKNOWN"],
        "unsafe_base_pair_interval_count": status_counts["UNSAFE"],
        "nonfinite_base_pair_interval_count": status_counts["NONFINITE"],
        "proof_leaf_count": sum(leaf_counts.values()),
        "proof_leaf_status_counts": {
            key.lower(): value for key, value in leaf_counts.items()
        },
        "per_obstacle": per_obstacle,
        "per_subject_group": per_subject,
        "minimum_clearance_certified_lower_bound_m": (
            minimum_lower if minimum_witness is not None else None
        ),
        "minimum_clearance_witness": minimum_witness,
        "all_base_pair_intervals_certified": all_certified,
    }


def validate_receipt_self_consistency(receipt: Mapping[str, Any]) -> None:
    """Reject omissions/tampering before no-clobber publication."""

    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("receipt_class") != INTERNAL_RECEIPT_CLASS
        or receipt.get("with_table") is not True
    ):
        raise ClearanceError("receipt identity/with_table contract changed")
    producer = _mapping(receipt.get("trusted_producer"), "trusted_producer")
    if producer.get("producer_id") != PRODUCER_ID:
        raise ClearanceError("receipt producer_id changed")
    algorithm = _mapping(producer.get("algorithm_contract"), "algorithm_contract")
    if algorithm.get("algorithm_id") != ALGORITHM_ID:
        raise ClearanceError("receipt algorithm_id changed")
    if _canonical_json_sha256(algorithm) != _digest(
        producer.get("algorithm_contract_sha256"), "algorithm_contract_sha256"
    ):
        raise ClearanceError("algorithm contract digest mismatch")
    source_rows = _sequence(
        producer.get("runtime_source_pins"), "runtime_source_pins"
    )
    expected_source_roles = {
        "geometry",
        "table_frame",
        "hope_commands",
        "scene_builder",
        "joint_order",
    }
    observed_source_roles: set[str] = set()
    for index, raw in enumerate(source_rows):
        row = _mapping(raw, f"runtime_source_pins[{index}]")
        role = _nonempty_string(row.get("role"), "runtime source role")
        if role in observed_source_roles:
            raise ClearanceError("runtime source pin roles are duplicated")
        observed_source_roles.add(role)
        _nonempty_string(row.get("path"), "runtime source path")
        _integer(row.get("bytes"), "runtime source bytes", minimum=1)
        _digest(row.get("sha256"), "runtime source sha256")
    if observed_source_roles != expected_source_roles:
        raise ClearanceError("runtime source pin set is incomplete")

    scene = _mapping(receipt.get("scene_contract"), "scene_contract")
    if (
        scene.get("scene_profile") != SCENE_PROFILE
        or scene.get("roles") != list(ACTION_BALL_ROLES)
        or scene.get("action_ball_keepout_semantics")
        != "robot_only_keepout_ball_excluded"
    ):
        raise ClearanceError("receipt scene profile/roles/keepout semantics changed")
    components = _sequence(scene.get("components"), "scene components")
    if len(components) != 5 or _canonical_json_sha256(components) != _digest(
        scene.get("components_sha256"), "components_sha256"
    ):
        raise ClearanceError("receipt five-piece component digest mismatch")
    if [row.get("role") for row in components if isinstance(row, Mapping)] != list(
        ACTION_BALL_ROLES
    ):
        raise ClearanceError("receipt component order/roles changed")
    normalized_components: list[dict[str, Any]] = []
    for expected_role, raw in zip(ACTION_BALL_ROLES, components):
        component = _mapping(raw, f"scene component {expected_role}")
        if component.get("geom_name") != OBSTACLE_GEOM_NAMES[expected_role]:
            raise ClearanceError("receipt obstacle geom name changed")
        center = np.asarray(component.get("center_m"), dtype=np.float64)
        extents = np.asarray(component.get("full_extents_m"), dtype=np.float64)
        lo = np.asarray(component.get("aabb_lo_m"), dtype=np.float64)
        hi = np.asarray(component.get("aabb_hi_m"), dtype=np.float64)
        if any(value.shape != (3,) for value in (center, extents, lo, hi)):
            raise ClearanceError("receipt component vector shape changed")
        if not all(np.isfinite(value).all() for value in (center, extents, lo, hi)):
            raise ClearanceError("receipt component contains NaN/Inf")
        if not (
            np.all(extents > 0.0)
            and np.allclose(
                center, (lo + hi) / 2.0, rtol=0.0, atol=1.0e-15
            )
            and np.allclose(
                extents, hi - lo, rtol=0.0, atol=1.0e-15
            )
        ):
            raise ClearanceError("receipt component center/extents contradict AABB")
        normalized_components.append(dict(component))
    validate_action_ball_assembly_contacts(normalized_components)
    keepout_lo = np.asarray(normalized_components[1]["aabb_lo_m"])
    keepout_hi = np.asarray(normalized_components[1]["aabb_hi_m"])
    top_lo = np.asarray(normalized_components[0]["aabb_lo_m"])
    if (
        keepout_lo[2] != KEEP_OUT_FLOOR_Z_M
        or keepout_hi[2] != top_lo[2]
    ):
        raise ClearanceError("receipt keepout is not floor-to-slab-underside")
    robot_geometry = _mapping(
        scene.get("robot_geometry"), "scene robot_geometry"
    )
    robot_names = tuple(
        _nonempty_string(value, f"collision_geom_names[{index}]")
        for index, value in enumerate(
            _sequence(
                robot_geometry.get("collision_geom_names"),
                "collision_geom_names",
            )
        )
    )
    if (
        not robot_names
        or len(set(robot_names)) != len(robot_names)
        or robot_geometry.get("collision_geom_count") != len(robot_names)
        or robot_geometry.get("all_enabled_collision_geoms") is not True
        or robot_geometry.get("racket_and_handle_geom_names")
        != list(RACKET_AND_HANDLE_GEOMS)
        or not set(RACKET_AND_HANDLE_GEOMS).issubset(robot_names)
    ):
        raise ClearanceError("receipt enabled robot geometry contract is incomplete")
    collision_rows = _sequence(
        robot_geometry.get("collision_geometry_rows"),
        "collision_geometry_rows",
    )
    if (
        [row.get("name") for row in collision_rows if isinstance(row, Mapping)]
        != list(robot_names)
        or _canonical_json_sha256(collision_rows)
        != _digest(
            robot_geometry.get("collision_geometry_sha256"),
            "collision_geometry_sha256",
        )
    ):
        raise ClearanceError("receipt robot collision geometry digest is false")

    bank = _mapping(receipt.get("bank_binding"), "bank_binding")
    matrix = _mapping(bank.get("output_matrix"), "bank output_matrix")
    if matrix.get("scopes") != list(REQUESTED_SCOPES):
        raise ClearanceError("bank matrix must contain exact upper/full scopes")
    motion_ids = tuple(
        _nonempty_string(value, f"motion_ids[{index}]")
        for index, value in enumerate(
            _sequence(matrix.get("motion_ids"), "motion_ids")
        )
    )
    if not motion_ids or len(set(motion_ids)) != len(motion_ids):
        raise ClearanceError("bank motion_ids must be non-empty and unique")
    expected_pairs = tuple(
        (motion_id, scope)
        for motion_id in motion_ids
        for scope in REQUESTED_SCOPES
    )
    if matrix.get("candidate_count") != len(expected_pairs):
        raise ClearanceError("bank candidate_count does not match full matrix")
    outputs = _sequence(bank.get("outputs"), "bank outputs")
    observed_pairs = tuple(
        (
            _nonempty_string(_mapping(row, "bank output").get("motion_id"), "motion_id"),
            _nonempty_string(_mapping(row, "bank output").get("scope"), "scope"),
        )
        for row in outputs
    )
    if observed_pairs != expected_pairs:
        raise ClearanceError("bank output order is not the exact upper/full matrix")

    result_rows = _sequence(receipt.get("results"), "receipt results")
    result_pairs = tuple(
        (
            _nonempty_string(_mapping(row, "result").get("motion_id"), "motion_id"),
            _nonempty_string(_mapping(row, "result").get("scope"), "scope"),
        )
        for row in result_rows
    )
    if result_pairs != expected_pairs:
        raise ClearanceError("receipt results do not cover the exact bank matrix")
    summaries = []
    for output, raw_result in zip(outputs, result_rows):
        result = _mapping(raw_result, "result")
        if (
            result.get("sha256") != output.get("sha256")
            or result.get("filename") != output.get("filename")
        ):
            raise ClearanceError("result does not bind exact output bytes")
        if result.get("robot_geom_names") != list(robot_names):
            raise ClearanceError("result omits or reorders an enabled robot geom")
        if result.get("obstacle_roles") != list(ACTION_BALL_ROLES):
            raise ClearanceError("result omits or reorders an ActionBall obstacle")
        recomputed = validate_motion_proof_result(result)
        if result.get("summary") != recomputed:
            raise ClearanceError("result summary does not match its proof ledger")
        expected_verdict = (
            "PASS"
            if recomputed["all_base_pair_intervals_certified"]
            else "FAIL_CLOSED"
        )
        if result.get("verdict") != expected_verdict:
            raise ClearanceError("result verdict contradicts proof ledger")
        summaries.append(recomputed)

    aggregate = _mapping(receipt.get("aggregate"), "aggregate")
    required = sum(
        row["required_base_pair_interval_count"] for row in summaries
    )
    certified = sum(
        row["certified_base_pair_interval_count"] for row in summaries
    )
    unknown = sum(
        row["unknown_base_pair_interval_count"] for row in summaries
    )
    unsafe = sum(
        row["unsafe_base_pair_interval_count"] for row in summaries
    )
    nonfinite = sum(
        row["nonfinite_base_pair_interval_count"] for row in summaries
    )
    expected_aggregate = {
        "output_count": len(result_rows),
        "required_base_pair_interval_count": required,
        "certified_base_pair_interval_count": certified,
        "unknown_base_pair_interval_count": unknown,
        "unsafe_base_pair_interval_count": unsafe,
        "nonfinite_base_pair_interval_count": nonfinite,
        "all_outputs_complete": (
            certified == required and unknown == unsafe == nonfinite == 0
        ),
    }
    if dict(aggregate) != expected_aggregate:
        raise ClearanceError("receipt aggregate contradicts proof ledgers")
    expected_verdict = (
        "PASS" if expected_aggregate["all_outputs_complete"] else "FAIL_CLOSED"
    )
    if receipt.get("verdict") != expected_verdict:
        raise ClearanceError("receipt verdict contradicts aggregate")
    authorization = _mapping(receipt.get("authorization"), "authorization")
    if dict(authorization) != {
        "swept_clearance_complete": expected_verdict == "PASS",
        "training_authorized": False,
        "hardware_authorized": False,
    }:
        raise ClearanceError("receipt authorization boundary changed")


def _source_pin_rows(
    snapshots: Mapping[str, FileSnapshot]
) -> list[dict[str, Any]]:
    return [
        {"role": role, **snapshot.binding()}
        for role, snapshot in sorted(snapshots.items())
    ]


def build_receipt(
    *,
    bank_binding: Mapping[str, Any],
    scene_contract: Mapping[str, Any],
    source_pins: Mapping[str, FileSnapshot],
    dependency_pins: Mapping[str, Any],
    robot_geometry: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    algorithm_contract = {
        "algorithm_id": ALGORITHM_ID,
        "certificate_kind": "conservative_continuous_time_swept_volume",
        "interpolation": INTERPOLATION_ID,
        "distance_oracle": DISTANCE_ORACLE_ID,
        "continuous_enclosure": ENCLOSURE_ID,
        "hard_clearance_m": HARD_CLEARANCE_M,
        "hard_threshold_predicate": "certified_lower_bound_m >= 0.005 exactly",
        "max_subdivision_depth": MAX_SUBDIVISION_DEPTH,
        "distance_query_cap_m": DISTANCE_QUERY_CAP_M,
        "sampled_only_claim": False,
        "geometry_only_claim": False,
        "leaf_partition_required": True,
        "unknown_nonfinite_unsafe_allowed_for_pass": False,
    }
    normalized_results = [dict(row) for row in results]
    required = sum(
        int(row["summary"]["required_base_pair_interval_count"])
        for row in normalized_results
    )
    certified = sum(
        int(row["summary"]["certified_base_pair_interval_count"])
        for row in normalized_results
    )
    unknown = sum(
        int(row["summary"]["unknown_base_pair_interval_count"])
        for row in normalized_results
    )
    unsafe = sum(
        int(row["summary"]["unsafe_base_pair_interval_count"])
        for row in normalized_results
    )
    nonfinite = sum(
        int(row["summary"]["nonfinite_base_pair_interval_count"])
        for row in normalized_results
    )
    complete = certified == required and unknown == unsafe == nonfinite == 0
    scene = dict(scene_contract)
    scene["robot_geometry"] = dict(robot_geometry)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_class": INTERNAL_RECEIPT_CLASS,
        "verdict": "PASS" if complete else "FAIL_CLOSED",
        "with_table": True,
        "trusted_producer": {
            "producer_id": PRODUCER_ID,
            "code": read_snapshot(SCRIPT_PATH, label="producer code").binding(),
            "algorithm_contract": algorithm_contract,
            "algorithm_contract_sha256": _canonical_json_sha256(
                algorithm_contract
            ),
            "dependency_pins": dict(dependency_pins),
            "runtime_source_pins": _source_pin_rows(source_pins),
        },
        "bank_binding": dict(bank_binding),
        "trajectory_contract": {
            "coverage": "entire_prepare_hit_recovery_continuous_time",
            "start": "first_frame",
            "includes_contact_opportunity": True,
            "end": "last_recovery_ready_frame",
            "scopes": list(REQUESTED_SCOPES),
            "interpolation": INTERPOLATION_ID,
            "time_scaling_invariance": (
                "positive teacher-rate scaling changes time only, not the certified path"
            ),
        },
        "scene_contract": scene,
        "method": {
            "certificate_kind": "conservative_continuous_time_swept_volume",
            "continuous_time_swept_volume": True,
            "sampled_or_geometry_only": False,
            "proof_ledger_complete": True,
        },
        "results": normalized_results,
        "aggregate": {
            "output_count": len(normalized_results),
            "required_base_pair_interval_count": required,
            "certified_base_pair_interval_count": certified,
            "unknown_base_pair_interval_count": unknown,
            "unsafe_base_pair_interval_count": unsafe,
            "nonfinite_base_pair_interval_count": nonfinite,
            "all_outputs_complete": complete,
        },
        "authorization": {
            "swept_clearance_complete": complete,
            "training_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "training_authorization",
            "hardware_authorization",
            "dynamics_or_balance",
            "ball_collision_with_robot_only_keepout",
            "sampled_only_continuous_time_proof",
        ],
    }
    validate_receipt_self_consistency(receipt)
    return receipt


def project_bank_gate_receipt(
    internal_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Project validated proof evidence to the exact bank/admission contract.

    The public receipt is deliberately not a second proof implementation.  It
    is a lossless identity/claim projection of the producer's validated proof
    ledger.  In particular, a caller cannot use this function to turn a
    sampled-only or incomplete result into the bank's continuous-time schema.
    """

    validate_receipt_self_consistency(internal_receipt)
    if internal_receipt["verdict"] != "PASS":
        raise ClearanceError(
            "only a complete internal swept-clearance PASS may be projected"
        )

    producer = _mapping(
        internal_receipt["trusted_producer"], "trusted_producer"
    )
    verifier = _mapping(producer["code"], "trusted producer code")
    source_rows = {
        _nonempty_string(row.get("role"), "runtime source role"): _mapping(
            row, "runtime source pin"
        )
        for row in _sequence(
            producer["runtime_source_pins"], "runtime_source_pins"
        )
    }
    geometry_sources = []
    for public_role, source_role in (
        ("table_dimensions", "geometry"),
        ("table_frame", "table_frame"),
        ("scene_builder", "scene_builder"),
    ):
        source = source_rows[source_role]
        geometry_sources.append(
            {
                "role": public_role,
                "path": _nonempty_string(
                    source.get("path"), f"{source_role} source path"
                ),
                "sha256": _digest(
                    source.get("sha256"), f"{source_role} source sha256"
                ),
            }
        )

    internal_bank = _mapping(
        internal_receipt["bank_binding"], "bank_binding"
    )
    public_bank: dict[str, Any] = {}
    for role in ("manifest", "recipe", "ready", "mjcf", "urdf", "body_order"):
        binding = _mapping(internal_bank[role], f"bank_binding.{role}")
        public_bank[f"{role}_sha256"] = _digest(
            binding.get("sha256"), f"bank_binding.{role}.sha256"
        )
    public_bank["station_center_shift_xy_m"] = internal_bank[
        "station_center_shift_xy_m"
    ]
    public_bank["output_matrix"] = dict(
        _mapping(internal_bank["output_matrix"], "bank output_matrix")
    )

    internal_outputs = _sequence(
        internal_bank["outputs"], "bank outputs"
    )
    public_outputs = [
        {
            "motion_id": _nonempty_string(
                row.get("motion_id"), f"bank output[{index}].motion_id"
            ),
            "scope": _nonempty_string(
                row.get("scope"), f"bank output[{index}].scope"
            ),
            "filename": _nonempty_string(
                row.get("filename"), f"bank output[{index}].filename"
            ),
            "sha256": _digest(
                row.get("sha256"), f"bank output[{index}].sha256"
            ),
        }
        for index, row in enumerate(internal_outputs)
    ]
    public_bank["outputs"] = public_outputs

    internal_scene = _mapping(
        internal_receipt["scene_contract"], "scene_contract"
    )
    public_components = []
    for index, raw in enumerate(
        _sequence(internal_scene["components"], "scene components")
    ):
        component = _mapping(raw, f"scene component[{index}]")
        public_components.append(
            {
                "role": _nonempty_string(
                    component.get("role"), f"scene component[{index}].role"
                ),
                "center_m": [
                    float(value)
                    for value in np.asarray(
                        component.get("center_m"), dtype=np.float64
                    ).reshape(3)
                ],
                "full_extents_m": [
                    float(value)
                    for value in np.asarray(
                        component.get("full_extents_m"), dtype=np.float64
                    ).reshape(3)
                ],
            }
        )
    robot_geometry = _mapping(
        internal_scene["robot_geometry"], "robot_geometry"
    )
    collision_names = sorted(
        _nonempty_string(value, "collision geom name")
        for value in _sequence(
            robot_geometry["collision_geom_names"],
            "collision_geom_names",
        )
    )

    internal_results = _sequence(
        internal_receipt["results"], "receipt results"
    )
    if len(internal_results) != len(public_outputs):
        raise ClearanceError(
            "internal result/output counts differ during public projection"
        )
    public_results = []
    for index, (raw_result, output) in enumerate(
        zip(internal_results, public_outputs)
    ):
        result = _mapping(raw_result, f"result[{index}]")
        summary = _mapping(result["summary"], f"result[{index}].summary")
        endpoint = _mapping(
            result.get("endpoint_contract"),
            f"result[{index}].endpoint_contract",
        )
        stored_fk = _mapping(
            result.get("stored_frame_fk_contract"),
            f"result[{index}].stored_frame_fk_contract",
        )
        frames = _integer(
            result.get("frames"), f"result[{index}].frames", minimum=2
        )
        if (
            endpoint.get("start_frame") != 0
            or endpoint.get("end_frame") != frames - 1
            or endpoint.get("shared_ready_joint_exact") is not True
            or endpoint.get("endpoint_velocity_channels_exact_zero") is not True
            or _integer(
                endpoint.get("prepare_frame_count_minimum"),
                f"result[{index}] prepare frames",
                minimum=1,
            )
            < 1
            or _integer(
                endpoint.get("recovery_frame_count_minimum"),
                f"result[{index}] recovery frames",
                minimum=1,
            )
            < 1
            or stored_fk.get("pass") is not True
            or stored_fk.get("frame_count") != frames
        ):
            raise ClearanceError(
                "public swept receipt requires the producer's exact "
                "shared-ready endpoints and stored-frame FK PASS"
            )
        interval_count = frames - 1
        minimum_clearance = _finite(
            summary.get("minimum_clearance_certified_lower_bound_m"),
            f"result[{index}] minimum clearance",
            minimum=HARD_CLEARANCE_M,
        )
        public_results.append(
            {
                **output,
                "frames": frames,
                "fps": _finite(
                    result.get("fps"), f"result[{index}].fps", minimum=0.0
                ),
                "duration_s": _finite(
                    result.get("duration_s"),
                    f"result[{index}].duration_s",
                    minimum=0.0,
                ),
                "start_frame": 0,
                "end_frame": frames - 1,
                "interval_count": interval_count,
                "certified_interval_count": interval_count,
                "unknown_interval_count": 0,
                "unsafe_interval_count": 0,
                "nonfinite_interval_count": 0,
                "all_intervals_conservatively_bounded": True,
                "contact_window_start_s": _finite(
                    result.get("contact_window_start_s"),
                    f"result[{index}].contact_window_start_s",
                    minimum=0.0,
                ),
                "contact_window_end_s": _finite(
                    result.get("contact_window_end_s"),
                    f"result[{index}].contact_window_end_s",
                    minimum=0.0,
                ),
                "coverage_start": "first_frame",
                "contact_opportunity_covered": True,
                "coverage_end": "last_frame",
                "complete_cycle": True,
                "with_table": True,
                "subjects": list(BANK_GATE_SUBJECTS),
                "obstacles": list(BANK_GATE_OBSTACLES),
                "verdict": "PASS",
                "hard_collision_count": 0,
                "minimum_clearance_certified_lower_bound_m": (
                    minimum_clearance
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_class": RECEIPT_CLASS,
        "verdict": "PASS",
        "with_table": True,
        "independent_verifier": {
            "path": _nonempty_string(
                verifier.get("path"), "producer code path"
            ),
            "sha256": _digest(
                verifier.get("sha256"), "producer code sha256"
            ),
        },
        "bank_binding": public_bank,
        "trajectory_contract": {
            "coverage": BANK_GATE_COVERAGE,
            "complete_cycle": True,
            "start": "first_canonical_ready_frame",
            "includes_contact_opportunity": True,
            "end": "final_canonical_recovery_ready_frame",
            "scopes": list(REQUESTED_SCOPES),
        },
        "scene_contract": {
            "subjects": list(BANK_GATE_SUBJECTS),
            "forbidden_world_geometry": list(BANK_GATE_OBSTACLES),
            "action_ball_keepout_semantics": (
                "robot_only_keepout_ball_excluded"
            ),
            "action_ball_assembly": {
                "roles": list(ACTION_BALL_ROLES),
                "geometry_sources": geometry_sources,
                "components": public_components,
                "components_sha256": _canonical_json_sha256(
                    public_components
                ),
            },
            "robot_geometry": {
                "all_enabled_collision_geoms": True,
                "collision_geom_names": collision_names,
                "collision_geom_names_sha256": _canonical_json_sha256(
                    collision_names
                ),
                "racket_and_handle_geom_names": list(
                    RACKET_AND_HANDLE_GEOMS
                ),
            },
        },
        "method": {
            "certificate_kind": (
                "conservative_continuous_time_swept_volume"
            ),
            "continuous_time_swept_volume": True,
            "sampled_or_geometry_only": False,
            "inter_sample_conservative_bound": True,
        },
        "results": public_results,
        "authorization": {
            "swept_clearance_complete": True,
            "training_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "dynamics_or_balance",
            "training_authorization",
            "hardware_authorization",
        ],
    }


def write_json_no_clobber(
    value: Mapping[str, Any],
    output_path: os.PathLike[str] | str,
    *,
    forbidden_tree: Path | None = None,
) -> Path:
    """Atomically publish strict JSON without overwriting an existing leaf."""

    destination = Path(
        os.path.abspath(os.fspath(Path(output_path).expanduser()))
    )
    if forbidden_tree is not None:
        tree = Path(os.path.abspath(os.fspath(forbidden_tree)))
        try:
            destination.relative_to(tree)
        except ValueError:
            pass
        else:
            raise ClearanceError(
                "swept-clearance receipt must be external to the motion bank"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ClearanceError("receipt parent must be a real directory")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing receipt {destination}")
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite existing receipt {destination}"
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _asset_relative_path(
    mjcf_path: Path,
    *,
    compiler: ET.Element | None,
    node: ET.Element,
) -> tuple[Path, str]:
    raw = node.get("file")
    if not raw:
        raise ClearanceError(f"MJCF <{node.tag}> asset lacks file")
    assetdir = compiler.get("assetdir", "") if compiler is not None else ""
    if node.tag == "mesh":
        typed_dir = compiler.get("meshdir", "") if compiler is not None else ""
    elif node.tag == "texture":
        typed_dir = compiler.get("texturedir", "") if compiler is not None else ""
    else:
        typed_dir = ""
    relative = Path(assetdir) / Path(typed_dir) / Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ClearanceError(f"MJCF asset escapes model root: {relative}")
    normalized = Path(os.path.normpath(os.fspath(relative)))
    absolute = Path(
        os.path.abspath(os.fspath(mjcf_path.parent / normalized))
    )
    try:
        absolute.relative_to(mjcf_path.parent)
    except ValueError as exc:
        raise ClearanceError(f"MJCF asset escapes model root: {absolute}") from exc
    return absolute, normalized.as_posix()


def snapshot_mjcf_closure(
    mjcf_source: FileSnapshot,
) -> tuple[dict[str, bytes], dict[str, Any], ET.Element]:
    """Snapshot every file-backed asset named by one pinned, include-free MJCF."""

    if b"<!DOCTYPE" in mjcf_source.data or b"<!ENTITY" in mjcf_source.data:
        raise ClearanceError("MJCF may not contain a DTD or entity declaration")
    try:
        root = ET.fromstring(mjcf_source.data)
    except ET.ParseError as exc:
        raise ClearanceError(f"cannot parse pinned MJCF: {exc}") from exc
    if list(root.iter("include")):
        raise ClearanceError("MJCF include files are unsupported and fail closed")
    compiler = root.find("./compiler")
    assets: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = [mjcf_source.binding()]
    supported_file_nodes = {"mesh", "texture", "hfield", "skin"}
    for node in root.iter():
        if node.get("file") is None:
            continue
        if node.tag not in supported_file_nodes:
            raise ClearanceError(
                f"unsupported file-backed MJCF element <{node.tag}>"
            )
        absolute, key = _asset_relative_path(
            mjcf_source.path, compiler=compiler, node=node
        )
        if key in assets:
            raise ClearanceError(f"duplicate MJCF asset key {key!r}")
        snapshot = read_snapshot(absolute, label=f"MJCF asset {key}")
        assets[key] = snapshot.data
        rows.append(
            {
                "path": str(snapshot.path),
                "asset_key": key,
                "bytes": snapshot.size,
                "sha256": snapshot.sha256,
            }
        )
    rows[1:] = sorted(rows[1:], key=lambda row: str(row["asset_key"]))
    closure = {
        "algorithm": "sha256(canonical-json(ordered-file-bindings))-v1",
        "file_count": len(rows),
        "asset_file_count": len(rows) - 1,
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
        "manifest_sha256": _canonical_json_sha256(rows),
    }
    return assets, closure, root


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def augment_mjcf_with_action_ball(
    mjcf_source: FileSnapshot, scene_contract: Mapping[str, Any]
) -> bytes:
    """Append the exact five inert measurement boxes to a copy of pinned XML."""

    if b"<!DOCTYPE" in mjcf_source.data or b"<!ENTITY" in mjcf_source.data:
        raise ClearanceError("MJCF may not contain a DTD or entity declaration")
    try:
        root = ET.fromstring(mjcf_source.data)
    except ET.ParseError as exc:
        raise ClearanceError(f"cannot parse pinned MJCF for augmentation: {exc}") from exc
    worldbodies = root.findall("./worldbody")
    if len(worldbodies) != 1:
        raise ClearanceError("MJCF must contain exactly one worldbody")
    existing_names = {
        node.get("name") for node in root.iter("geom") if node.get("name")
    }
    components = _sequence(scene_contract.get("components"), "scene components")
    if len(components) != 5:
        raise ClearanceError("scene must contain exactly five components")
    for expected_role, raw in zip(ACTION_BALL_ROLES, components):
        component = _mapping(raw, f"scene component {expected_role}")
        if component.get("role") != expected_role:
            raise ClearanceError("scene component order/role changed")
        name = OBSTACLE_GEOM_NAMES[expected_role]
        if component.get("geom_name") != name or name in existing_names:
            raise ClearanceError(f"ActionBall obstacle geom identity collision: {name}")
        center = np.asarray(component.get("center_m"), dtype=np.float64)
        extents = np.asarray(component.get("full_extents_m"), dtype=np.float64)
        if (
            center.shape != (3,)
            or extents.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(extents).all()
            or not np.all(extents > 0.0)
        ):
            raise ClearanceError(f"ActionBall component {expected_role} is invalid")
        ET.SubElement(
            worldbodies[0],
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": _format_vector(center),
                "size": _format_vector(extents / 2.0),
                "contype": "0",
                "conaffinity": "0",
                "group": "6",
            },
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _mj_name(mujoco: Any, model: Any, object_type: Any, index: int, label: str) -> str:
    value = mujoco.mj_id2name(model, object_type, int(index))
    if not isinstance(value, str) or not value:
        raise ClearanceError(f"{label} {index} must have a unique non-empty name")
    return value


def _mj_id(mujoco: Any, model: Any, object_type: Any, name: str, label: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ClearanceError(f"compiled MJCF is missing {label} {name!r}")
    return value


def _body_descends_from(model: Any, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    ancestor = int(ancestor_id)
    visited: set[int] = set()
    while current > 0 and current not in visited:
        if current == ancestor:
            return True
        visited.add(current)
        current = int(model.body_parentid[current])
    return current == ancestor


def _body_path_edge_bound_m(
    model: Any,
    *,
    ancestor_body_id: int,
    descendant_body_id: int,
) -> float:
    """All-configuration path-length bound from ancestor to descendant origins."""

    ancestor = int(ancestor_body_id)
    current = int(descendant_body_id)
    total = 0.0
    visited: set[int] = set()
    while current != ancestor:
        if current <= 0 or current in visited:
            raise ClearanceError("robot body tree is not a rooted acyclic subtree")
        visited.add(current)
        offset = np.asarray(model.body_pos[current], dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ClearanceError("compiled body offset is invalid")
        total += float(np.linalg.norm(offset))
        joint_ids = np.flatnonzero(
            np.asarray(model.jnt_bodyid, dtype=np.int64) == current
        )
        # A joint anchor can move the body origin by at most twice its radius
        # about that anchor.  Adding this term is deliberately conservative.
        for joint_id in joint_ids.tolist():
            joint_pos = np.asarray(model.jnt_pos[int(joint_id)], dtype=np.float64)
            if joint_pos.shape != (3,) or not np.isfinite(joint_pos).all():
                raise ClearanceError("compiled joint anchor is invalid")
            total += 2.0 * float(np.linalg.norm(joint_pos))
        current = int(model.body_parentid[current])
    return total


def _geom_identity_row(
    mujoco: Any, model: Any, geom_id: int
) -> dict[str, Any]:
    gid = int(geom_id)
    body_id = int(model.geom_bodyid[gid])
    name = _mj_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, gid, "geom")
    body_name = _mj_name(
        mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body"
    )
    row: dict[str, Any] = {
        "name": name,
        "body_name": body_name,
        "type": int(model.geom_type[gid]),
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
        "dataid": int(model.geom_dataid[gid]),
        "size": np.asarray(model.geom_size[gid], dtype=np.float64).tolist(),
        "pos": np.asarray(model.geom_pos[gid], dtype=np.float64).tolist(),
        "quat": np.asarray(model.geom_quat[gid], dtype=np.float64).tolist(),
        "rbound_m": float(model.geom_rbound[gid]),
    }
    data_id = int(model.geom_dataid[gid])
    if (
        int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        and data_id >= 0
    ):
        address = int(model.mesh_vertadr[data_id])
        count = int(model.mesh_vertnum[data_id])
        vertices = np.asarray(
            model.mesh_vert[address : address + count], dtype=np.float64
        )
        row["mesh_vertex_count"] = count
        row["mesh_vertices_sha256"] = _sha256_bytes(
            np.ascontiguousarray(vertices, dtype="<f8").tobytes(order="C")
        )
    return row


def bind_robot_kinematics(
    mujoco: Any,
    model: Any,
    *,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
) -> KinematicBinding:
    """Bind the exact floating-root robot and derive conservative reach coefficients."""

    joint_ids = tuple(
        _mj_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
            "runtime joint",
        )
        for name in joint_names
    )
    if len(set(joint_ids)) != 31:
        raise ClearanceError("runtime joint names do not resolve bijectively")
    if any(
        int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE)
        for joint_id in joint_ids
    ):
        raise ClearanceError("all 31 runtime joints must be scalar hinges")
    joint_qpos_addresses = tuple(
        int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids
    )
    if len(set(joint_qpos_addresses)) != 31:
        raise ClearanceError("runtime hinge qpos addresses are not unique")

    body_ids = tuple(
        _mj_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
            "runtime body",
        )
        for name in body_names
    )
    if len(set(body_ids)) != 32:
        raise ClearanceError("runtime body names do not resolve bijectively")
    root_body_id = body_ids[0]
    root_candidates = [
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_bodyid[joint_id]) == root_body_id
        and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(root_candidates) != 1:
        raise ClearanceError("runtime root body must own exactly one free joint")
    root_joint_id = root_candidates[0]
    root_qpos_address = int(model.jnt_qposadr[root_joint_id])
    if root_qpos_address < 0 or root_qpos_address + 7 > int(model.nq):
        raise ClearanceError("free-root qpos range is invalid")

    subtree_joint_ids = {
        joint_id
        for joint_id in range(int(model.njnt))
        if _body_descends_from(
            model, int(model.jnt_bodyid[joint_id]), root_body_id
        )
    }
    expected_subtree_joints = {root_joint_id, *joint_ids}
    if subtree_joint_ids != expected_subtree_joints:
        raise ClearanceError(
            "robot subtree contains an unbound or missing joint; continuous proof "
            "cannot cover unknown DOFs"
        )

    supported_types = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_MESH),
    }
    collision_ids: list[int] = []
    collision_names: list[str] = []
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if not _body_descends_from(model, body_id, root_body_id):
            continue
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        if int(model.geom_type[geom_id]) not in supported_types:
            raise ClearanceError(
                f"enabled robot geom {geom_id} has unsupported type"
            )
        name = _mj_name(
            mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, "robot geom"
        )
        collision_ids.append(geom_id)
        collision_names.append(name)
    if not collision_ids or len(set(collision_names)) != len(collision_names):
        raise ClearanceError("enabled robot collision geom names are empty/duplicated")
    if not set(RACKET_AND_HANDLE_GEOMS).issubset(collision_names):
        raise ClearanceError("pinned MJCF lost racket collision face or handle")

    envelopes: list[GeomEnvelope] = []
    for geom_id, geom_name in zip(collision_ids, collision_names):
        body_id = int(model.geom_bodyid[geom_id])
        geom_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
        rbound = _finite(
            model.geom_rbound[geom_id],
            f"{geom_name} geom_rbound",
            minimum=0.0,
        )
        if geom_pos.shape != (3,) or not np.isfinite(geom_pos).all():
            raise ClearanceError(f"{geom_name} local position is invalid")
        local_shape_reach = float(np.linalg.norm(geom_pos)) + rbound
        root_path = _body_path_edge_bound_m(
            model,
            ancestor_body_id=root_body_id,
            descendant_body_id=body_id,
        )
        root_anchor = float(
            np.linalg.norm(
                np.asarray(model.jnt_pos[root_joint_id], dtype=np.float64)
            )
        )
        root_reach = root_anchor + root_path + local_shape_reach
        joint_reaches: list[float] = []
        for joint_id in joint_ids:
            joint_body = int(model.jnt_bodyid[joint_id])
            if not _body_descends_from(model, body_id, joint_body):
                joint_reaches.append(0.0)
                continue
            path = _body_path_edge_bound_m(
                model,
                ancestor_body_id=joint_body,
                descendant_body_id=body_id,
            )
            anchor = float(
                np.linalg.norm(
                    np.asarray(model.jnt_pos[joint_id], dtype=np.float64)
                )
            )
            joint_reaches.append(anchor + path + local_shape_reach)
        if not (
            math.isfinite(root_reach)
            and root_reach >= 0.0
            and all(math.isfinite(value) and value >= 0.0 for value in joint_reaches)
        ):
            raise ClearanceError(f"{geom_name} reach envelope is invalid")
        envelopes.append(
            GeomEnvelope(
                name=geom_name,
                body_name=_mj_name(
                    mujoco,
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                    "robot geom body",
                ),
                root_rotation_reach_m=root_reach,
                joint_rotation_reach_m=tuple(joint_reaches),
                geom_rbound_m=rbound,
            )
        )
    return KinematicBinding(
        root_body_id=root_body_id,
        root_joint_id=root_joint_id,
        root_qpos_address=root_qpos_address,
        joint_ids=joint_ids,
        joint_qpos_addresses=joint_qpos_addresses,
        body_ids=body_ids,
        collision_geom_ids=tuple(collision_ids),
        collision_geom_names=tuple(collision_names),
        racket_geom_names=RACKET_AND_HANDLE_GEOMS,
        envelopes=tuple(envelopes),
    )


class MujocoClearanceBackend:
    def __init__(
        self,
        *,
        mujoco: Any,
        model: Any,
        binding: KinematicBinding,
        obstacle_geom_ids: Mapping[str, int],
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.binding = binding
        self.robot_geometries = binding.envelopes
        self.obstacle_roles = ACTION_BALL_ROLES
        self.robot_geom_ids = dict(
            zip(binding.collision_geom_names, binding.collision_geom_ids)
        )
        self.obstacle_geom_ids = {
            role: int(obstacle_geom_ids[role]) for role in ACTION_BALL_ROLES
        }
        if len(set(self.obstacle_geom_ids.values())) != 5:
            raise ClearanceError("ActionBall obstacle geom IDs are not unique")

    def apply_pose(
        self,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        joint_position: np.ndarray,
    ) -> None:
        root = np.asarray(root_position, dtype=np.float64)
        quaternion = _quaternion_normalize_wxyz(
            root_quaternion_wxyz, "applied root"
        )
        joints = np.asarray(joint_position, dtype=np.float64)
        if (
            root.shape != (3,)
            or joints.shape != (31,)
            or not np.isfinite(root).all()
            or not np.isfinite(joints).all()
        ):
            raise ClearanceError("applied kinematic pose is invalid")
        self.data.qpos[:] = self.model.qpos0
        address = self.binding.root_qpos_address
        self.data.qpos[address : address + 3] = root
        self.data.qpos[address + 3 : address + 7] = quaternion
        self.data.qpos[list(self.binding.joint_qpos_addresses)] = joints
        self.mujoco.mj_forward(self.model, self.data)
        if not (
            np.isfinite(np.asarray(self.data.geom_xpos)).all()
            and np.isfinite(np.asarray(self.data.geom_xmat)).all()
        ):
            raise ClearanceError("MuJoCo FK produced non-finite geom poses")

    def distance_saturation_query(
        self, robot_geom_name: str, obstacle_role: str, distmax_m: float
    ) -> tuple[float, bool]:
        cap = _finite(distmax_m, "distance query distmax", minimum=0.0)
        if cap > DISTANCE_QUERY_CAP_M:
            raise ClearanceError("distance query exceeds algorithm cap")
        if robot_geom_name not in self.robot_geom_ids:
            raise ClearanceError(f"unknown robot geom {robot_geom_name!r}")
        if obstacle_role not in self.obstacle_geom_ids:
            raise ClearanceError(f"unknown ActionBall obstacle {obstacle_role!r}")
        observed = float(
            self.mujoco.mj_geomDistance(
                self.model,
                self.data,
                self.robot_geom_ids[robot_geom_name],
                self.obstacle_geom_ids[obstacle_role],
                cap,
                None,
            )
        )
        return observed, distance_query_is_saturated(observed, cap)

    def validate_stored_frame_fk(self, clip: MotionClip) -> dict[str, Any]:
        max_position_error = 0.0
        max_orientation_error = 0.0
        for frame in range(clip.frames):
            self.apply_pose(
                clip.body_pos_w[frame, 0],
                clip.body_quat_w[frame, 0],
                clip.joint_pos[frame],
            )
            positions = np.asarray(self.data.xpos, dtype=np.float64)[
                list(self.binding.body_ids)
            ]
            quaternions = np.asarray(self.data.xquat, dtype=np.float64)[
                list(self.binding.body_ids)
            ]
            expected_positions = clip.body_pos_w[frame]
            expected_quaternions = clip.body_quat_w[frame]
            position_error = float(
                np.max(np.linalg.norm(positions - expected_positions, axis=1))
            )
            dots = np.abs(
                np.sum(
                    quaternions
                    * expected_quaternions
                    / np.linalg.norm(expected_quaternions, axis=1, keepdims=True),
                    axis=1,
                )
            )
            dots = np.clip(dots, -1.0, 1.0)
            orientation_error = float(np.max(2.0 * np.arccos(dots)))
            if not (
                math.isfinite(position_error)
                and math.isfinite(orientation_error)
            ):
                raise ClearanceError("stored-frame FK comparison is non-finite")
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(
                max_orientation_error, orientation_error
            )
        if (
            max_position_error > FK_POSITION_TOL_M
            or max_orientation_error > FK_ORIENTATION_TOL_RAD
        ):
            raise ClearanceError(
                "stored-frame schema-2 FK does not match pinned MJCF: "
                f"position={max_position_error:.6g} "
                f"orientation={max_orientation_error:.6g}"
            )
        return {
            "frame_count": clip.frames,
            "position_tolerance_m": FK_POSITION_TOL_M,
            "orientation_tolerance_rad": FK_ORIENTATION_TOL_RAD,
            "maximum_position_error_m": max_position_error,
            "maximum_orientation_error_rad": max_orientation_error,
            "pass": True,
        }


def compile_action_ball_backend(
    *,
    mujoco: Any,
    mjcf_source: FileSnapshot,
    scene_contract: Mapping[str, Any],
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
) -> tuple[MujocoClearanceBackend, dict[str, Any], dict[str, Any]]:
    assets, closure, _root = snapshot_mjcf_closure(mjcf_source)
    try:
        canonical_model = mujoco.MjModel.from_xml_string(
            mjcf_source.data.decode("utf-8"), assets=assets
        )
    except Exception as exc:
        raise ClearanceError(f"cannot compile pinned canonical MJCF closure: {exc}") from exc
    canonical_binding = bind_robot_kinematics(
        mujoco,
        canonical_model,
        joint_names=joint_names,
        body_names=body_names,
    )
    augmented_xml = augment_mjcf_with_action_ball(
        mjcf_source, scene_contract
    )
    try:
        augmented_model = mujoco.MjModel.from_xml_string(
            augmented_xml.decode("utf-8"), assets=assets
        )
    except Exception as exc:
        raise ClearanceError(f"cannot compile five-piece ActionBall MJCF: {exc}") from exc
    if int(augmented_model.ngeom) != int(canonical_model.ngeom) + 5:
        raise ClearanceError("ActionBall augmentation did not add exactly five geoms")
    augmented_binding = bind_robot_kinematics(
        mujoco,
        augmented_model,
        joint_names=joint_names,
        body_names=body_names,
    )
    if augmented_binding.collision_geom_names != canonical_binding.collision_geom_names:
        raise ClearanceError("ActionBall augmentation changed robot collision geom order")
    canonical_rows = [
        _geom_identity_row(mujoco, canonical_model, geom_id)
        for geom_id in canonical_binding.collision_geom_ids
    ]
    augmented_rows = [
        _geom_identity_row(mujoco, augmented_model, geom_id)
        for geom_id in augmented_binding.collision_geom_ids
    ]
    if canonical_rows != augmented_rows:
        raise ClearanceError("ActionBall augmentation changed robot collision geometry")
    obstacle_ids = {
        role: _mj_id(
            mujoco,
            augmented_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            OBSTACLE_GEOM_NAMES[role],
            "ActionBall obstacle geom",
        )
        for role in ACTION_BALL_ROLES
    }
    if any(
        int(augmented_model.geom_bodyid[geom_id]) != 0
        or int(augmented_model.geom_type[geom_id])
        != int(mujoco.mjtGeom.mjGEOM_BOX)
        for geom_id in obstacle_ids.values()
    ):
        raise ClearanceError("ActionBall obstacles must compile as world-fixed boxes")
    robot_geometry = {
        "all_enabled_collision_geoms": True,
        "collision_geom_count": len(augmented_binding.collision_geom_names),
        "collision_geom_names": list(augmented_binding.collision_geom_names),
        "collision_geometry_rows": augmented_rows,
        "collision_geometry_sha256": _canonical_json_sha256(augmented_rows),
        "racket_and_handle_geom_names": list(RACKET_AND_HANDLE_GEOMS),
        "reach_envelopes": [
            {
                "name": envelope.name,
                "body_name": envelope.body_name,
                "root_rotation_reach_m": envelope.root_rotation_reach_m,
                "joint_rotation_reach_m": list(
                    envelope.joint_rotation_reach_m
                ),
                "geom_rbound_m": envelope.geom_rbound_m,
            }
            for envelope in augmented_binding.envelopes
        ],
    }
    compiled_contract = {
        "canonical_counts": {
            "nbody": int(canonical_model.nbody),
            "njnt": int(canonical_model.njnt),
            "nq": int(canonical_model.nq),
            "nv": int(canonical_model.nv),
            "ngeom": int(canonical_model.ngeom),
        },
        "augmented_counts": {
            "nbody": int(augmented_model.nbody),
            "njnt": int(augmented_model.njnt),
            "nq": int(augmented_model.nq),
            "nv": int(augmented_model.nv),
            "ngeom": int(augmented_model.ngeom),
        },
        "obstacle_geom_ids": obstacle_ids,
        "robot_geometry_unchanged_after_five_world_boxes": True,
        "mjcf_asset_closure": closure,
    }
    backend = MujocoClearanceBackend(
        mujoco=mujoco,
        model=augmented_model,
        binding=augmented_binding,
        obstacle_geom_ids=obstacle_ids,
    )
    return backend, robot_geometry, compiled_contract


def _binding_from_json(
    raw: Any,
    *,
    owner_path: Path,
    label: str,
) -> tuple[Path, str]:
    value = _mapping(raw, label)
    path_value = Path(_nonempty_string(value.get("path"), f"{label}.path")).expanduser()
    if not path_value.is_absolute():
        # Repository contracts use repo-relative paths.  Test/local receipts may
        # use paths relative to their owner when no repo-relative file exists.
        repo_candidate = REPO_ROOT / path_value
        owner_candidate = owner_path.parent / path_value
        path_value = repo_candidate if repo_candidate.exists() else owner_candidate
    return (
        Path(os.path.abspath(os.fspath(path_value))),
        _digest(value.get("sha256"), f"{label}.sha256"),
    )


def _manifest_output_rows(
    manifest: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> tuple[tuple[str, ...], Sequence[Mapping[str, Any]]]:
    matrix = _mapping(manifest.get("output_matrix"), "manifest output_matrix")
    recipe_matrix = _mapping(
        recipe.get("required_output_matrix"), "recipe required_output_matrix"
    )
    if dict(matrix) != dict(recipe_matrix):
        raise ClearanceError(
            "manifest matrix must equal the recipe's final composed matrix; "
            "append-suffix-only manifests are not admissible"
        )
    motion_ids = tuple(
        _nonempty_string(value, f"output_matrix.motion_ids[{index}]")
        for index, value in enumerate(
            _sequence(matrix.get("motion_ids"), "output_matrix.motion_ids")
        )
    )
    scopes = tuple(
        _nonempty_string(value, f"output_matrix.scopes[{index}]")
        for index, value in enumerate(
            _sequence(matrix.get("scopes"), "output_matrix.scopes")
        )
    )
    if (
        not motion_ids
        or len(set(motion_ids)) != len(motion_ids)
        or scopes != REQUESTED_SCOPES
        or matrix.get("candidate_count") != len(motion_ids) * 2
    ):
        raise ClearanceError(
            "final motion matrix must be non-empty with exact upper/full scopes"
        )
    outputs = tuple(
        _mapping(value, f"manifest outputs[{index}]")
        for index, value in enumerate(
            _sequence(manifest.get("outputs"), "manifest outputs")
        )
    )
    expected_pairs = tuple(
        (motion_id, scope)
        for motion_id in motion_ids
        for scope in REQUESTED_SCOPES
    )
    observed_pairs = tuple(
        (
            _nonempty_string(row.get("motion_id"), "output motion_id"),
            _nonempty_string(row.get("scope"), "output scope"),
        )
        for row in outputs
    )
    if observed_pairs != expected_pairs:
        raise ClearanceError(
            "manifest outputs must cover the exact ordered upper/full final matrix"
        )
    return motion_ids, outputs


def _clearance_recipe_contract(
    recipe_snapshot: FileSnapshot,
    recipe: Mapping[str, Any],
    *,
    recipe_repo_root: Path,
) -> tuple[Mapping[str, Any], Any | None, Any | None]:
    """Return the canonical fields needed by clearance without schema guessing.

    Historical canonical recipes expose the fields directly.  Arbitrary-N
    recipes are accepted only through their strict loader and its explicit
    projection of the embedded canonical template; copying
    ``compiler_template`` fields ad hoc here would bypass source/order/ready
    validation.
    """

    try:
        arbitrary = importlib.import_module("canonical_motion_arbitrary_bank")
    except (ImportError, OSError) as exc:
        if recipe.get("recipe_type") == "canonical_arbitrary_n_recipe_v1":
            raise ClearanceError(
                f"cannot import strict arbitrary-N recipe loader: {exc}"
            ) from exc
        return recipe, None, None
    if recipe.get("recipe_type") != arbitrary.RECIPE_TYPE:
        return recipe, None, None
    try:
        loaded = arbitrary.load_arbitrary_bank_recipe(
            recipe_snapshot.path,
            repo_root=recipe_repo_root,
        )
        if (
            Path(loaded.path).resolve() != recipe_snapshot.path
            or loaded.sha256 != recipe_snapshot.sha256
        ):
            raise ClearanceError(
                "strict arbitrary-N loader reopened different recipe identity/bytes"
            )
        view = arbitrary.swept_clearance_recipe_contract(loaded)
    except ClearanceError:
        raise
    except Exception as exc:
        raise ClearanceError(
            f"strict arbitrary-N recipe validation failed: {exc}"
        ) from exc
    return _mapping(view, "arbitrary-N swept-clearance recipe view"), loaded, arbitrary


def _ready_arrays(snapshot: FileSnapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        archive = np.load(io.BytesIO(snapshot.data), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ClearanceError(f"cannot parse canonical ready NPZ: {exc}") from exc
    with archive as payload:
        for key, shape in (
            ("joint_pos", (31,)),
            ("root_pos_w", (3,)),
            ("root_quat_w", (4,)),
        ):
            if key not in payload.files:
                raise ClearanceError(f"canonical ready NPZ is missing {key}")
            value = np.asarray(payload[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ClearanceError(
                    f"canonical ready {key} must be finite with shape {shape}"
                )
        joint = np.asarray(payload["joint_pos"], dtype=np.float64)
        root = np.asarray(payload["root_pos_w"], dtype=np.float64)
        quaternion = _quaternion_normalize_wxyz(
            np.asarray(payload["root_quat_w"], dtype=np.float64),
            "canonical ready",
        )
    return joint, root, quaternion


def validate_complete_ready_cycle(
    clip: MotionClip,
    *,
    ready_joint: np.ndarray,
    ready_root: np.ndarray,
    ready_quaternion: np.ndarray,
) -> dict[str, Any]:
    if not (
        np.array_equal(clip.joint_pos[0], ready_joint)
        and np.array_equal(clip.joint_pos[-1], ready_joint)
    ):
        raise ClearanceError(
            f"{clip.motion_id}/{clip.scope} does not start and finish at exact shared-ready joints"
        )
    root_errors = [
        float(np.max(np.abs(clip.body_pos_w[index, 0] - ready_root)))
        for index in (0, -1)
    ]
    orientation_errors = [
        shortest_arc_angle_rad(
            clip.body_quat_w[index, 0], ready_quaternion
        )
        for index in (0, -1)
    ]
    ready_tolerance = 2.0e-6
    if max(root_errors) > ready_tolerance or max(orientation_errors) > ready_tolerance:
        raise ClearanceError(
            f"{clip.motion_id}/{clip.scope} root endpoints leave shared ready"
        )
    try:
        archive = np.load(io.BytesIO(clip.snapshot.data), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ClearanceError("cannot reopen exact motion bytes for endpoint velocity") from exc
    with archive as payload:
        for key in (
            "joint_vel",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            array = np.asarray(payload[key])
            if not (
                np.array_equal(array[0], np.zeros_like(array[0]))
                and np.array_equal(array[-1], np.zeros_like(array[-1]))
            ):
                raise ClearanceError(
                    f"{clip.motion_id}/{clip.scope} {key} endpoints are not exact zero"
                )
    return {
        "start_frame": 0,
        "end_frame": clip.frames - 1,
        "shared_ready_joint_exact": True,
        "shared_ready_root_tolerance_m_rad": ready_tolerance,
        "start_root_position_error_m": root_errors[0],
        "end_root_position_error_m": root_errors[1],
        "start_root_orientation_error_rad": orientation_errors[0],
        "end_root_orientation_error_rad": orientation_errors[1],
        "endpoint_velocity_channels_exact_zero": True,
        "prepare_frame_count_minimum": 1,
        "recovery_frame_count_minimum": 1,
    }


def load_pinned_bank(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    recipe_path: Path,
    expected_recipe_sha256: str,
    bank_dir: Path,
    mjcf_path: Path,
    expected_mjcf_sha256: str,
    body_names: tuple[str, ...] | None = None,
    recipe_repo_root: Path | None = None,
) -> tuple[
    dict[str, Any],
    list[MotionClip],
    FileSnapshot,
    FileSnapshot,
    FileSnapshot,
    FileSnapshot,
    FileSnapshot,
    tuple[str, ...],
]:
    """Bind the final composed bank and parse exact motion bytes."""

    manifest_snapshot = read_snapshot(
        manifest_path,
        label="bank manifest",
        expected_sha256=expected_manifest_sha256,
    )
    manifest = _mapping(
        _strict_json_bytes(manifest_snapshot.data, "bank manifest"),
        "bank manifest",
    )
    recipe_snapshot = read_snapshot(
        recipe_path,
        label="motion recipe",
        expected_sha256=expected_recipe_sha256,
    )
    recipe = _mapping(
        _strict_json_bytes(recipe_snapshot.data, "motion recipe"),
        "motion recipe",
    )
    declared_recipe_path, declared_recipe_sha = _binding_from_json(
        manifest.get("recipe"),
        owner_path=manifest_snapshot.path,
        label="manifest recipe",
    )
    if (
        declared_recipe_path != recipe_snapshot.path
        or declared_recipe_sha != recipe_snapshot.sha256
    ):
        raise ClearanceError("manifest does not bind the exact recipe bytes")

    strict_recipe_root = (
        REPO_ROOT
        if recipe_repo_root is None
        else Path(recipe_repo_root).expanduser().resolve(strict=True)
    )
    if not strict_recipe_root.is_dir() or strict_recipe_root.is_symlink():
        raise ClearanceError("recipe_repo_root must be one real directory")
    recipe_contract, arbitrary_loaded, arbitrary_module = (
        _clearance_recipe_contract(
            recipe_snapshot,
            recipe,
            recipe_repo_root=strict_recipe_root,
        )
    )
    if arbitrary_loaded is not None:
        try:
            arbitrary_module.validate_arbitrary_build_manifest(
                manifest,
                arbitrary_loaded,
            )
        except Exception as exc:
            raise ClearanceError(
                f"arbitrary-N compiler manifest validation failed: {exc}"
            ) from exc

    model_contract = _mapping(
        recipe_contract.get("model_contract"),
        "recipe model_contract",
    )
    declared_mjcf = _resolve_repo_path(
        model_contract.get("mjcf_path"), "model_contract.mjcf_path"
    )
    declared_mjcf_sha = _digest(
        model_contract.get("mjcf_sha256"), "model_contract.mjcf_sha256"
    )
    mjcf_snapshot = read_snapshot(
        mjcf_path,
        label="canonical MJCF",
        expected_sha256=expected_mjcf_sha256,
    )
    if (
        declared_mjcf != mjcf_snapshot.path
        or declared_mjcf_sha != mjcf_snapshot.sha256
    ):
        raise ClearanceError("recipe does not bind the exact canonical MJCF")

    urdf_path = _resolve_repo_path(
        model_contract.get("urdf_path"), "model_contract.urdf_path"
    )
    urdf_snapshot = read_snapshot(
        urdf_path,
        label="canonical URDF",
        expected_sha256=_digest(
            model_contract.get("urdf_sha256"), "model_contract.urdf_sha256"
        ),
    )
    body_order_path = _resolve_repo_path(
        model_contract.get("body_order_path"), "model_contract.body_order_path"
    )
    body_order_snapshot = read_snapshot(
        body_order_path,
        label="runtime body order",
        expected_sha256=_digest(
            model_contract.get("body_order_sha256"),
            "model_contract.body_order_sha256",
        ),
    )
    resolved_body_names = read_body_order(body_order_snapshot)
    if body_names is not None and resolved_body_names != body_names:
        raise ClearanceError("injected body names do not match recipe body order")

    ready_contract = _mapping(
        recipe_contract.get("canonical_ready"), "recipe canonical_ready"
    )
    ready_path = _resolve_repo_path(
        ready_contract.get("path"), "canonical_ready.path"
    )
    ready_snapshot = read_snapshot(
        ready_path,
        label="canonical ready",
        expected_sha256=_digest(
            ready_contract.get("sha256"), "canonical_ready.sha256"
        ),
    )
    manifest_ready_path, manifest_ready_sha = _binding_from_json(
        manifest.get("ready"),
        owner_path=manifest_snapshot.path,
        label="manifest ready",
    )
    if (
        manifest_ready_path != ready_snapshot.path
        or manifest_ready_sha != ready_snapshot.sha256
    ):
        raise ClearanceError("manifest does not bind the exact canonical ready")

    motion_ids, outputs = _manifest_output_rows(manifest, recipe_contract)
    bank = Path(os.path.abspath(os.fspath(bank_dir.expanduser())))
    if not bank.is_dir() or bank.is_symlink():
        raise ClearanceError("bank directory must be a real directory")
    expected_filenames: list[str] = []
    clips: list[MotionClip] = []
    normalized_outputs: list[dict[str, Any]] = []
    for index, row in enumerate(outputs):
        motion_id = str(row["motion_id"])
        scope = str(row["scope"])
        filename = _nonempty_string(row.get("filename"), f"outputs[{index}].filename")
        if Path(filename).name != filename:
            raise ClearanceError("motion output filename must be one safe basename")
        expected_filename = f"{motion_id}_{scope}_canonical_v2.npz"
        if filename != expected_filename:
            raise ClearanceError(
                f"motion output filename must be {expected_filename!r}"
            )
        expected_filenames.append(filename)
        snapshot = read_snapshot(
            bank / filename,
            label=f"motion output {motion_id}/{scope}",
            expected_sha256=_digest(
                row.get("output_npz_sha256"),
                f"outputs[{index}].output_npz_sha256",
            ),
        )
        clip = load_motion_snapshot(
            snapshot=snapshot,
            motion_id=motion_id,
            scope=scope,
            body_names=resolved_body_names,
            contact_window_start_s=row.get("contact_window_start_s"),
            contact_window_end_s=row.get("contact_window_end_s"),
        )
        duration_matches = (
            abs(
                _finite(row.get("duration_s"), f"outputs[{index}].duration_s")
                - clip.duration_s
            )
            <= 1.0e-9
        )
        # Canonical-five composed manifests historically used output-relative
        # entry/exit.  The arbitrary compiler's identically named fields bind
        # source frames and are validated against the exact source clip by its
        # strict loader.  In both cases this producer reads and certifies every
        # frame of the exact output NPZ; no output interval comes from either
        # manifest field.
        output_track_bound = duration_matches
        if arbitrary_loaded is None:
            output_track_bound = output_track_bound and (
                _integer(
                    row.get("entry_frame"),
                    f"outputs[{index}].entry_frame",
                )
                == 0
                and _integer(
                    row.get("exit_frame"),
                    f"outputs[{index}].exit_frame",
                )
                == clip.frames - 1
            )
        if not output_track_bound:
            raise ClearanceError(
                f"{motion_id}/{scope} manifest does not bind its full first-to-last track"
            )
        clips.append(clip)
        normalized_outputs.append(
            {
                "motion_id": motion_id,
                "scope": scope,
                "filename": filename,
                "path": str(snapshot.path),
                "bytes": snapshot.size,
                "sha256": snapshot.sha256,
                "frames": clip.frames,
                "fps": clip.fps,
                "duration_s": clip.duration_s,
                "contact_window_start_s": clip.contact_window_start_s,
                "contact_window_end_s": clip.contact_window_end_s,
            }
        )
    actual_npz = sorted(path.name for path in bank.glob("*.npz"))
    if actual_npz != sorted(expected_filenames):
        raise ClearanceError(
            "bank NPZ set is not the exact final composed output matrix"
        )
    bank_binding = {
        "manifest": manifest_snapshot.binding(),
        "recipe": recipe_snapshot.binding(),
        "ready": ready_snapshot.binding(),
        "mjcf": mjcf_snapshot.binding(),
        "urdf": urdf_snapshot.binding(),
        "body_order": body_order_snapshot.binding(),
        "station_center_shift_xy_m": manifest.get(
            "station_center_shift_xy_m"
        ),
        "output_matrix": {
            "motion_ids": list(motion_ids),
            "scopes": list(REQUESTED_SCOPES),
            "candidate_count": len(clips),
        },
        "outputs": normalized_outputs,
    }
    return (
        bank_binding,
        clips,
        manifest_snapshot,
        recipe_snapshot,
        ready_snapshot,
        mjcf_snapshot,
        body_order_snapshot,
        resolved_body_names,
    )


def _module_dependency_pin(module: Any, label: str) -> dict[str, Any]:
    path_raw = getattr(module, "__file__", None)
    if not isinstance(path_raw, str):
        raise ClearanceError(f"{label} module has no file origin")
    snapshot = read_snapshot(path_raw, label=f"{label} module origin")
    return {
        "version": _nonempty_string(
            getattr(module, "__version__", "unknown"), f"{label} version"
        ),
        "origin": snapshot.binding(),
    }


def produce_receipt(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    recipe_path: Path,
    expected_recipe_sha256: str,
    bank_dir: Path,
    mjcf_path: Path,
    expected_mjcf_sha256: str,
    geometry_source_path: Path,
    expected_geometry_sha256: str,
    table_frame_source_path: Path,
    expected_table_frame_sha256: str,
    hope_commands_source_path: Path,
    expected_hope_commands_sha256: str,
    scene_builder_source_path: Path,
    expected_scene_builder_sha256: str,
    joint_order_source_path: Path,
    expected_joint_order_source_sha256: str,
) -> dict[str, Any]:
    """Run the complete trusted producer in memory and return one strict receipt."""

    source_pins = {
        "geometry": read_snapshot(
            geometry_source_path,
            label="table geometry source",
            expected_sha256=expected_geometry_sha256,
        ),
        "table_frame": read_snapshot(
            table_frame_source_path,
            label="table frame source",
            expected_sha256=expected_table_frame_sha256,
        ),
        "hope_commands": read_snapshot(
            hope_commands_source_path,
            label="table pose command source",
            expected_sha256=expected_hope_commands_sha256,
        ),
        "scene_builder": read_snapshot(
            scene_builder_source_path,
            label="ActionBall scene builder source",
            expected_sha256=expected_scene_builder_sha256,
        ),
        "joint_order": read_snapshot(
            joint_order_source_path,
            label="runtime joint-order source",
            expected_sha256=expected_joint_order_source_sha256,
        ),
    }
    joint_names = load_runtime_joint_names(source_pins["joint_order"])
    scene_contract = derive_action_ball_assembly(
        geometry_source=source_pins["geometry"],
        table_frame_source=source_pins["table_frame"],
        hope_commands_source=source_pins["hope_commands"],
        scene_builder_source=source_pins["scene_builder"],
    )
    (
        bank_binding,
        clips,
        _manifest_snapshot,
        _recipe_snapshot,
        ready_snapshot,
        mjcf_snapshot,
        _body_order_snapshot,
        body_names,
    ) = load_pinned_bank(
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        recipe_path=recipe_path,
        expected_recipe_sha256=expected_recipe_sha256,
        bank_dir=bank_dir,
        mjcf_path=mjcf_path,
        expected_mjcf_sha256=expected_mjcf_sha256,
    )

    try:
        mujoco = importlib.import_module("mujoco")
    except ImportError as exc:
        raise ClearanceError(
            "mathematical producer blocked: pinned MuJoCo dependency is unavailable"
        ) from exc
    backend, robot_geometry, compiled_contract = compile_action_ball_backend(
        mujoco=mujoco,
        mjcf_source=mjcf_snapshot,
        scene_contract=scene_contract,
        joint_names=joint_names,
        body_names=body_names,
    )
    scene_contract = dict(scene_contract)
    scene_contract["compiled_model_contract"] = compiled_contract
    ready_joint, ready_root, ready_quaternion = _ready_arrays(ready_snapshot)

    results: list[dict[str, Any]] = []
    for clip in clips:
        endpoint = validate_complete_ready_cycle(
            clip,
            ready_joint=ready_joint,
            ready_root=ready_root,
            ready_quaternion=ready_quaternion,
        )
        stored_fk = backend.validate_stored_frame_fk(clip)
        result = certify_motion_continuous(clip, backend)
        result["endpoint_contract"] = endpoint
        result["stored_frame_fk_contract"] = stored_fk
        results.append(result)

    dependency_pins = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "numpy": _module_dependency_pin(np, "numpy"),
        "mujoco": _module_dependency_pin(mujoco, "mujoco"),
    }
    return project_bank_gate_receipt(
        build_receipt(
            bank_binding=bank_binding,
            scene_contract=scene_contract,
            source_pins=source_pins,
            dependency_pins=dependency_pins,
            robot_geometry=robot_geometry,
            results=results,
        )
    )


def _failure_receipt(exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_class": RECEIPT_CLASS,
        "verdict": "FAIL_CLOSED",
        "with_table": True,
        "trusted_producer": {
            "producer_id": PRODUCER_ID,
            "code": read_snapshot(SCRIPT_PATH, label="producer code").binding(),
            "algorithm_id": ALGORITHM_ID,
        },
        "error": f"{type(exc).__name__}: {exc}",
        "authorization": {
            "swept_clearance_complete": False,
            "training_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "continuous_swept_clearance_pass",
            "training_authorization",
            "hardware_authorization",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--expected-mjcf-sha256", required=True)
    parser.add_argument(
        "--geometry-source",
        type=Path,
        default=DEFAULT_GEOMETRY_SOURCE,
    )
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument(
        "--table-frame-source",
        type=Path,
        default=DEFAULT_TABLE_FRAME_SOURCE,
    )
    parser.add_argument("--expected-table-frame-sha256", required=True)
    parser.add_argument(
        "--hope-commands-source",
        type=Path,
        default=DEFAULT_HOPE_COMMANDS_SOURCE,
    )
    parser.add_argument("--expected-hope-commands-sha256", required=True)
    parser.add_argument(
        "--scene-builder-source",
        type=Path,
        default=DEFAULT_SCENE_BUILDER_SOURCE,
    )
    parser.add_argument("--expected-scene-builder-sha256", required=True)
    parser.add_argument(
        "--joint-order-source",
        type=Path,
        default=DEFAULT_JOINT_ORDER_SOURCE,
    )
    parser.add_argument("--expected-joint-order-source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = produce_receipt(
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            recipe_path=args.recipe,
            expected_recipe_sha256=args.expected_recipe_sha256,
            bank_dir=args.bank_dir,
            mjcf_path=args.mjcf,
            expected_mjcf_sha256=args.expected_mjcf_sha256,
            geometry_source_path=args.geometry_source,
            expected_geometry_sha256=args.expected_geometry_sha256,
            table_frame_source_path=args.table_frame_source,
            expected_table_frame_sha256=args.expected_table_frame_sha256,
            hope_commands_source_path=args.hope_commands_source,
            expected_hope_commands_sha256=args.expected_hope_commands_sha256,
            scene_builder_source_path=args.scene_builder_source,
            expected_scene_builder_sha256=args.expected_scene_builder_sha256,
            joint_order_source_path=args.joint_order_source,
            expected_joint_order_source_sha256=(
                args.expected_joint_order_source_sha256
            ),
        )
        write_json_no_clobber(
            receipt,
            args.output,
            forbidden_tree=Path(
                os.path.abspath(os.fspath(args.bank_dir.expanduser()))
            ),
        )
    except Exception as exc:
        try:
            write_json_no_clobber(
                _failure_receipt(exc),
                args.output,
                forbidden_tree=Path(
                    os.path.abspath(os.fspath(args.bank_dir.expanduser()))
                ),
            )
        except Exception as publication_exc:
            print(
                f"{type(exc).__name__}: {exc}; "
                f"failure receipt publication also failed: {publication_exc}",
                file=sys.stderr,
            )
            return 2
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "receipt": str(Path(args.output).resolve()),
                "receipt_sha256": _sha256_bytes(
                    Path(args.output).read_bytes()
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
