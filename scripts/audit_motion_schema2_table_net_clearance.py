#!/usr/bin/env python3
"""Fail-closed dense table/net clearance gate for Franco backhand-loop B.

The gate consumes the exact vendor-L1 certificate and the same exact schema-2
motion/MJCF closure.  It appends the frozen tracking-task table top, net and two
net posts to an in-memory copy of the vendor MJCF, then checks every enabled
robot collision geom against every obstacle at 1201 finite 400 Hz samples.

This is a finite dense geometric screen, not a mathematical continuous-time
certificate.  The tool never calls ``mj_step``, trains a policy, deploys, or
issues hardware commands.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import stat
import sys
import types
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "motion-franco-backhand-loop-b-table-net-clearance-20260715-v1"
PLAN_STATUS = "preregistered_source_gate_pass_runtime_audit_not_run"
ASSET_ID = "franco_backhand_loop_b"
CERTIFICATE_STATUS = "complete_cpu_dense_table_net_clearance_pass_dynamics_blocked"
L1_CERTIFICATE_STATUS = "complete_cpu_vendor_l1_safety_pass_downstream_blocked"
L1_PLAN_PATH = REPO_ROOT / "configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json"
L1_VALIDATOR_PATH = REPO_ROOT / "scripts/audit_motion_schema2_vendor_l1_safety.py"
RACKET_GEOMS = ("right_racket_collision", "right_racket_handle_collision")
NPZ_FIELDS = {
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
OBSTACLE_NAMES = (
    "motion_table_top",
    "motion_net",
    "motion_net_post_left",
    "motion_net_post_right",
)


class TableNetError(ValueError):
    """Fail-closed source, lineage, frame, runtime, clearance or publication error."""


@dataclass(frozen=True)
class FileSnapshot:
    """One immutable read from one O_NOFOLLOW file descriptor."""

    path: Path
    data: bytes
    device: int
    inode: int
    mode: int
    size: int
    sha256: str

    def binding(self) -> dict[str, Any]:
        return {"path": str(self.path), "bytes": self.size, "sha256": self.sha256}


@dataclass
class OutputTarget:
    """Pinned output parent used for all no-clobber publication operations."""

    path: Path
    parent_path: Path
    name: str
    dir_fd: int
    parent_device: int
    parent_inode: int
    closed: bool = False

    def close(self) -> None:
        if not self.closed:
            os.close(self.dir_fd)
            self.closed = True


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_file_snapshot(
    path: Path,
    label: str,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> FileSnapshot:
    """Read exactly one regular inode once; all consumers use returned bytes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise TableNetError(f"cannot open exact {label}: {exc}") from exc
    try:
        return _read_open_fd_snapshot(
            fd,
            path,
            label,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
    finally:
        os.close(fd)


def _read_open_fd_snapshot(
    fd: int,
    path: Path,
    label: str,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> FileSnapshot:
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise TableNetError(f"{label} must be a regular non-symlink file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise TableNetError(f"{label} inode/metadata changed during immutable read")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise TableNetError(f"{label} short/long read {len(data)} != {before.st_size}")
        digest = _sha256_bytes(data)
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise TableNetError(f"{label} bytes {before.st_size} != {expected_bytes}")
        if expected_sha256 is not None and digest != expected_sha256:
            raise TableNetError(f"{label} SHA-256 {digest} != {expected_sha256}")
        return FileSnapshot(
            path=path,
            data=data,
            device=int(before.st_dev),
            inode=int(before.st_ino),
            mode=int(before.st_mode),
            size=int(before.st_size),
            sha256=digest,
        )
    except OSError as exc:
        raise TableNetError(f"cannot read exact {label}: {exc}") from exc


def _read_relative_snapshot(
    root_fd: int,
    root_path: Path,
    relative: Path,
    label: str,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> FileSnapshot:
    if relative.is_absolute() or not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise TableNetError(f"unsafe relative path for {label}: {relative}")
    held: list[int] = []
    current = root_fd
    try:
        for component in relative.parts[:-1]:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            current = os.open(component, flags, dir_fd=current)
            held.append(current)
            st = os.fstat(current)
            if not stat.S_ISDIR(st.st_mode):
                raise TableNetError(f"{label} parent component is not a directory: {component}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(relative.parts[-1], flags, dir_fd=current)
        try:
            return _read_open_fd_snapshot(
                fd,
                root_path / relative,
                label,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )
        finally:
            os.close(fd)
    except OSError as exc:
        raise TableNetError(f"cannot open exact {label} beneath bound root: {exc}") from exc
    finally:
        for held_fd in reversed(held):
            os.close(held_fd)


def sha256_file(path: Path) -> str:
    """Compatibility helper for source-only callers; runtime uses snapshots."""

    return read_file_snapshot(path, str(path)).sha256


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TableNetError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def parse_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                TableNetError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TableNetError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TableNetError(f"{label} must be a JSON object")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    return parse_json_bytes(read_file_snapshot(path, label).data, label)


def exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise TableNetError(f"{label} keys changed: actual={actual}")
    return value


def _binding(snapshot: FileSnapshot) -> dict[str, Any]:
    return snapshot.binding()


def _snapshot_repo_binding(row: Any, label: str, relative: str) -> FileSnapshot:
    binding = exact_keys(row, {"path", "bytes", "sha256"}, label)
    if binding["path"] != relative:
        raise TableNetError(f"{label} path changed")
    return read_file_snapshot(
        REPO_ROOT / relative,
        label,
        expected_bytes=binding["bytes"],
        expected_sha256=binding["sha256"],
    )


def _verify_repo_binding(row: Any, label: str, relative: str) -> Path:
    return _snapshot_repo_binding(row, label, relative).path


def _snapshot_absolute_sha(row: Any, label: str) -> FileSnapshot:
    binding = exact_keys(row, {"path", "sha256"}, label)
    return read_file_snapshot(
        Path(binding["path"]), label, expected_sha256=binding["sha256"]
    )


def _verify_absolute_sha(row: Any, label: str) -> Path:
    return _snapshot_absolute_sha(row, label).path


def _load_exact_module(
    name: str, path: Path, *, expected_bytes: int, expected_sha256: str, label: str
) -> Any:
    """Load one exact source by path without accepting a stale module."""

    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise TableNetError(f"{label} expected_bytes must be a positive integer")
    snapshot = read_file_snapshot(
        path,
        label,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        code = compile(snapshot.data, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        if sys.modules.get(name) is not module:
            raise TableNetError(f"{label} replaced its module entry")
    except BaseException as exc:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, TableNetError):
            raise
        raise TableNetError(f"cannot import exact {label} from {path}: {exc}") from exc
    return module


def _ast_number(source: FileSnapshot, class_name: str | None, variable: str) -> float:
    try:
        tree = ast.parse(source.data.decode("utf-8"))
    except (UnicodeError, SyntaxError) as exc:
        raise TableNetError(f"cannot parse source literal {source.path}: {exc}") from exc
    nodes: Sequence[ast.stmt] = tree.body
    if class_name is not None:
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(classes) != 1:
            raise TableNetError(f"cannot find exactly one class {class_name}")
        nodes = classes[0].body
    values: list[float] = []
    for node in nodes:
        target: ast.expr | None = None
        expression: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, expression = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, expression = node.target, node.value
        if isinstance(target, ast.Name) and target.id == variable and expression is not None:
            try:
                raw = ast.literal_eval(expression)
            except (ValueError, TypeError):
                continue
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                values.append(float(raw))
    if len(values) != 1:
        raise TableNetError(
            f"cannot extract exactly one literal {class_name + '.' if class_name else ''}{variable}"
        )
    return values[0]


def _close(actual: Any, expected: float, label: str) -> float:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise TableNetError(f"{label} must be numeric")
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(value, expected, abs_tol=1e-12, rel_tol=0.0):
        raise TableNetError(f"{label}={value} != {expected}")
    return value


def _expected_frame_contract() -> dict[str, Any]:
    return {
        "schema2_mjcf_world": {
            "origin": "robot_environment_origin_on_floor_plane",
            "axes": {"x": "forward", "y": "robot_anatomical_left", "z": "up"},
            "floor_z_m": 0.0,
        },
        "hope_table_frame": {
            "origin": "near_side_left_corner_of_table_surface_from_P1",
            "axes": {"x": "toward_P2", "y": "left_from_P1", "z": "up"},
            "table_x_range_m": [0.0, 2.74],
            "table_y_range_m": [-1.525, 0.0],
            "table_surface_z_m": 0.0,
        },
        "hope_to_schema2_mjcf": {
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_m": [0.5, 0.7625, 0.76],
            "formula": (
                "p_schema2_mjcf=p_HOPE+[vb_table_near_x,TABLE_WIDTH/2,"
                "vb_table_surface_z]"
            ),
            "capture_table_pose_observed": False,
            "pose_semantics": (
                "frozen_counterfactual_tracking_task_virtual_table_not_capture_extrinsic"
            ),
        },
    }


def _expected_obstacle_geometry() -> dict[str, Any]:
    return {
        "primitive": "axis_aligned_box_full_extents_m",
        "table_top": {
            "name": "motion_table_top",
            "center_mjcf_world_m": [1.87, 0.0, 0.735],
            "full_extents_m": [2.74, 1.525, 0.05],
        },
        "net": {
            "name": "motion_net",
            "center_mjcf_world_m": [1.87, 0.0, 0.83625],
            "full_extents_m": [0.01, 1.825, 0.1525],
        },
        "net_posts": [
            {
                "name": "motion_net_post_left",
                "center_mjcf_world_m": [1.87, 0.9125, 0.84625],
                "full_extents_m": [0.02, 0.02, 0.1725],
            },
            {
                "name": "motion_net_post_right",
                "center_mjcf_world_m": [1.87, -0.9125, 0.84625],
                "full_extents_m": [0.02, 0.02, 0.1725],
            },
        ],
        "source_semantics": (
            "table slab and net from canonical HOPE geometry; posts match build_net_post_cfg "
            "and are conservatively treated as clearance obstacles even though current Isaac "
            "posts are visual-only"
        ),
    }


def _obstacle_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    posts = value.get("net_posts")
    if not isinstance(posts, list):
        raise TableNetError("obstacle geometry changed: net_posts must be a list")
    rows = [value.get("table_top"), value.get("net"), *posts]
    if len(rows) != 4 or any(not isinstance(row, dict) for row in rows):
        raise TableNetError("obstacle geometry changed: exactly four boxes are required")
    return [dict(row) for row in rows]


def validate_frame_and_geometry_sources(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the one accepted HOPE -> schema2/MJCF table placement from source."""

    sources = exact_keys(
        plan["frame_sources"],
        {"table_geometry", "tracking_command", "tracking_scene_adapter"},
        "frame_sources",
    )
    geometry_source = _snapshot_repo_binding(
        sources["table_geometry"],
        "table geometry source",
        (
            "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
            "tasks/table_tennis/geometry.py"
        ),
    )
    command_source = _snapshot_repo_binding(
        sources["tracking_command"],
        "tracking command source",
        (
            "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
            "tasks/tracking/mdp/hope_commands.py"
        ),
    )
    _snapshot_repo_binding(
        sources["tracking_scene_adapter"],
        "tracking scene adapter",
        (
            "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
            "tasks/tracking/config/agibot_a3/hope_env_cfg.py"
        ),
    )
    constants = {
        name: _ast_number(geometry_source, None, name)
        for name in (
            "TABLE_LENGTH", "TABLE_WIDTH", "TABLE_HEIGHT", "TABLE_THICKNESS",
            "NET_X", "NET_HEIGHT", "NET_OVERHANG", "NET_THICKNESS",
        )
    }
    expected_constants = {
        "TABLE_LENGTH": 2.74,
        "TABLE_WIDTH": 1.525,
        "TABLE_HEIGHT": 0.76,
        "TABLE_THICKNESS": 0.05,
        "NET_X": 1.37,
        "NET_HEIGHT": 0.1525,
        "NET_OVERHANG": 0.15,
        "NET_THICKNESS": 0.01,
    }
    for key, expected in expected_constants.items():
        _close(constants[key], expected, f"table source {key}")
    near_x = _ast_number(command_source, "RacketTargetCommandCfg", "vb_table_near_x")
    surface_z = _ast_number(command_source, "RacketTargetCommandCfg", "vb_table_surface_z")
    _close(near_x, 0.5, "tracking vb_table_near_x")
    _close(surface_z, 0.76, "tracking vb_table_surface_z")

    expected_frame = _expected_frame_contract()
    if plan["frame_contract"] != expected_frame:
        raise TableNetError("frame contract changed")
    expected_obstacles = _expected_obstacle_geometry()
    if plan["obstacle_geometry"] != expected_obstacles:
        raise TableNetError("obstacle geometry changed")

    # Independently derive every center/extent from the bound source constants.
    dx, dy, dz = (near_x, constants["TABLE_WIDTH"] / 2.0, surface_z)
    post_h = constants["NET_HEIGHT"] + 0.02
    derived = [
        {
            "name": "motion_table_top",
            "center_mjcf_world_m": [
                dx + constants["TABLE_LENGTH"] / 2.0,
                dy - constants["TABLE_WIDTH"] / 2.0,
                dz - constants["TABLE_THICKNESS"] / 2.0,
            ],
            "full_extents_m": [
                constants["TABLE_LENGTH"], constants["TABLE_WIDTH"], constants["TABLE_THICKNESS"]
            ],
        },
        {
            "name": "motion_net",
            "center_mjcf_world_m": [
                dx + constants["NET_X"],
                dy - constants["TABLE_WIDTH"] / 2.0,
                dz + constants["NET_HEIGHT"] / 2.0,
            ],
            "full_extents_m": [
                constants["NET_THICKNESS"],
                constants["TABLE_WIDTH"] + 2.0 * constants["NET_OVERHANG"],
                constants["NET_HEIGHT"],
            ],
        },
        {
            "name": "motion_net_post_left",
            "center_mjcf_world_m": [
                dx + constants["NET_X"], dy + constants["NET_OVERHANG"], dz + post_h / 2.0
            ],
            "full_extents_m": [0.02, 0.02, post_h],
        },
        {
            "name": "motion_net_post_right",
            "center_mjcf_world_m": [
                dx + constants["NET_X"],
                dy - constants["TABLE_WIDTH"] - constants["NET_OVERHANG"],
                dz + post_h / 2.0,
            ],
            "full_extents_m": [0.02, 0.02, post_h],
        },
    ]
    frozen = _obstacle_rows(plan["obstacle_geometry"])
    for index, (actual, wanted) in enumerate(zip(frozen, derived)):
        if actual["name"] != wanted["name"]:
            raise TableNetError(f"obstacle geometry changed at row {index}")
        for field in ("center_mjcf_world_m", "full_extents_m"):
            if len(actual[field]) != 3:
                raise TableNetError(f"obstacle {actual['name']} {field} must have three values")
            for axis, expected in zip(actual[field], wanted[field]):
                _close(axis, float(expected), f"obstacle {actual['name']} {field}")

    # Static frame evidence from the exact vendor XML: floor z=0 and +Y is anatomical left.
    canonical_binding = plan["a3_model"]["canonical_mjcf"]
    canonical_source = _snapshot_repo_binding(
        canonical_binding,
        "canonical vendor MJCF frame source",
        canonical_binding["path"],
    )
    try:
        root = ET.fromstring(canonical_source.data)
    except ET.ParseError as exc:
        raise TableNetError(f"cannot parse canonical vendor MJCF frame source: {exc}") from exc
    worldbodies = root.findall("./worldbody")
    if len(worldbodies) != 1:
        raise TableNetError("vendor MJCF must contain exactly one worldbody")
    floor = worldbodies[0].find("./geom[@name='floor']")
    if floor is None or floor.get("type") != "plane":
        raise TableNetError("vendor MJCF floor frame source changed")
    floor_pos = [float(value) for value in floor.get("pos", "0 0 0").split()]
    if floor_pos != [0.0, 0.0, 0.0]:
        raise TableNetError("vendor MJCF floor is not at schema2/MJCF z=0")
    left = root.find(".//body[@name='left_shoulder_pitch_Link']")
    right = root.find(".//body[@name='right_shoulder_pitch_Link']")
    if left is None or right is None:
        raise TableNetError("vendor MJCF shoulder frame witnesses are missing")
    if not (float(left.get("pos", "0 0 0").split()[1]) > 0.0 > float(right.get("pos", "0 0 0").split()[1])):
        raise TableNetError("vendor MJCF +Y is no longer robot anatomical left")
    return {
        "hope_to_schema2_mjcf": expected_frame["hope_to_schema2_mjcf"],
        "obstacles": derived,
        "source_constants": constants,
        "vendor_floor_z_m": 0.0,
        "capture_table_pose_observed": False,
    }


def _expected_audit_contract() -> dict[str, Any]:
    return {
        "source_frames": 151,
        "source_fps": 50,
        "substeps_per_source_interval": 8,
        "dense_frames": 1201,
        "effective_sampling_hz": 400,
        "interpolation": (
            "inherit_exact_vendor_L1_root_xyz_linear_root_quaternion_shortest_arc_"
            "slerp_joint_position_linear"
        ),
        "dense_sampling_is_continuous_time_certificate": False,
        "robot_geometry_scope": (
            "all_37_enabled_robot_collision_geoms_with_separate_racket_and_handle_rollup"
        ),
        "racket_collision_geoms": list(RACKET_GEOMS),
        "obstacle_names": list(OBSTACLE_NAMES),
        "hard_clearance_m": 0.005,
        "warning_clearance_m": 0.02,
        "reporting_clearance_bisection_tolerance_m": 0.000001,
        "reporting_clearance_cap_m": 0.1,
        "hard_threshold_predicate": (
            "fail iff audit_self_collision._far(model,data,robot_geom,obstacle_geom,0.005) "
            "is false"
        ),
        "danger_propagation": (
            "any robot-obstacle dense sample below 5mm fails the whole asset and marks both "
            "adjacent source frames"
        ),
        "hard_fail_is_noncompensable": True,
        "model_augmentation": (
            "append four exact world-fixed box geoms after the canonical worldbody children and "
            "compile from in-memory XML plus exact 74-file asset map"
        ),
        "mj_step_calls": 0,
    }


def validate_plan(plan_path: Path, expected_sha256: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    plan_snapshot = read_file_snapshot(plan_path, "table/net preregistration")
    actual_sha = plan_snapshot.sha256
    if actual_sha != expected_sha256 or len(expected_sha256) != 64:
        raise TableNetError(
            f"table/net preregistration SHA mismatch: expected={expected_sha256} actual={actual_sha}"
        )
    plan = parse_json_bytes(plan_snapshot.data, "table/net preregistration")
    exact_keys(
        plan,
        {
            "schema_version", "plan_id", "status", "human_owner", "executor", "scope",
            "asset_id", "validator", "frozen_vendor_l1", "exact_runtime_input", "a3_model",
            "frame_sources", "frame_contract", "obstacle_geometry", "audit_contract",
            "output_contract", "authorization", "explicit_non_claims", "next_gate",
        },
        "table/net preregistration",
    )
    expected_scope = (
        "CPU-only finite 400 Hz whole-trajectory table-top, net and net-post clearance audit "
        "for exact Franco backhand-loop B in the frozen tracking-task table pose; no dynamics, "
        "training, deployment or hardware"
    )
    if (
        plan["schema_version"] != 1
        or plan["plan_id"] != PLAN_ID
        or plan["status"] != PLAN_STATUS
        or plan["human_owner"] != "Franco"
        or plan["executor"] != "Codex"
        or plan["scope"] != expected_scope
        or plan["asset_id"] != ASSET_ID
    ):
        raise TableNetError("table/net identity, status, attribution or scope changed")
    _verify_repo_binding(
        plan["validator"], "table/net validator", "scripts/audit_motion_schema2_table_net_clearance.py"
    )

    frozen = exact_keys(
        plan["frozen_vendor_l1"],
        {
            "certificate", "preregistration", "validator", "required_certificate_status",
            "required_authorization",
        },
        "frozen_vendor_l1",
    )
    if frozen["certificate"] != {
        "path": (
            "/workspace/codexschema/motion_video_intake_20260711/vendor_l1_primary_v1/"
            "franco_backhand_loop_b_98e7b883b29d.vendor_l1_safety_certificate.json"
        ),
        "sha256": "6840df34a6aa6e5636192c705a8ecaa563f751658fe538df428bc317c858db60",
    }:
        raise TableNetError("vendor L1 certificate binding changed")
    l1_plan_snapshot = _snapshot_repo_binding(
        frozen["preregistration"],
        "vendor L1 preregistration",
        "configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json",
    )
    l1_validator_snapshot = _snapshot_repo_binding(
        frozen["validator"],
        "vendor L1 validator",
        "scripts/audit_motion_schema2_vendor_l1_safety.py",
    )
    l1 = _load_exact_module(
        "motion_vendor_l1_for_table_net",
        l1_validator_snapshot.path,
        expected_bytes=frozen["validator"]["bytes"],
        expected_sha256=frozen["validator"]["sha256"],
        label="vendor L1 validator",
    )
    secure_l1_plan = parse_json_bytes(l1_plan_snapshot.data, "vendor L1 preregistration")
    try:
        l1_plan, _, l0_v1_plan = l1.validate_plan(
            l1_plan_snapshot.path, frozen["preregistration"]["sha256"]
        )
    except (OSError, TypeError, ValueError) as exc:
        raise TableNetError(f"vendor L1 source closure changed: {exc}") from exc
    if l1_plan != secure_l1_plan:
        raise TableNetError("vendor L1 preregistration changed between bound bytes and validation")
    l0_binding = secure_l1_plan["frozen_l0"]["preregistration"]
    secure_l0_snapshot = _snapshot_repo_binding(
        l0_binding,
        "vendor L0 preregistration",
        l0_binding["path"],
    )
    secure_l0_v2_plan = parse_json_bytes(
        secure_l0_snapshot.data, "vendor L0 V2 preregistration"
    )
    l0_v1_binding = secure_l0_v2_plan["frozen_v1"]["preregistration"]
    secure_l0_v1_snapshot = _snapshot_repo_binding(
        l0_v1_binding,
        "vendor L0 V1 preregistration",
        l0_v1_binding["path"],
    )
    if l0_v1_plan != parse_json_bytes(
        secure_l0_v1_snapshot.data, "vendor L0 V1 preregistration"
    ):
        raise TableNetError("vendor L0 preregistration changed between bound bytes and validation")
    if frozen["required_certificate_status"] != L1_CERTIFICATE_STATUS:
        raise TableNetError("required vendor L1 certificate status changed")
    required_auth = {
        "vendor_l1_complete": True,
        "table_net_authorized": True,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }
    if frozen["required_authorization"] != required_auth:
        raise TableNetError("required vendor L1 authorization changed")
    if plan["exact_runtime_input"] != l1_plan["exact_runtime_input"]:
        raise TableNetError("schema-2 NPZ differs from exact vendor L1 plan")
    model = exact_keys(
        plan["a3_model"],
        {
            "canonical_mjcf", "derived_closure", "compiled_collision_contract_sha256",
            "enabled_robot_geom_count",
        },
        "a3_model",
    )
    if (
        model["canonical_mjcf"] != l1_plan["a3_model"]["canonical_mjcf"]
        or model["derived_closure"] != l1_plan["a3_model"]["derived_closure"]
        or model["compiled_collision_contract_sha256"]
        != l1_plan["a3_model"]["compiled_collision_contract"]["sha256"]
        or model["enabled_robot_geom_count"]
        != l1_plan["a3_model"]["compiled_collision_contract"]["enabled_robot_geom_count"]
        or model["enabled_robot_geom_count"] != 37
    ):
        raise TableNetError("MJCF or derived closure differs from exact vendor L1 plan")
    validate_frame_and_geometry_sources(plan)
    _snapshot_exact_mjcf_closure(plan)
    if plan["audit_contract"] != _expected_audit_contract():
        raise TableNetError("audit contract changed or weakened")
    expected_output = {
        "certificate_path": (
            "/workspace/codexschema/motion_video_intake_20260711/table_net_primary_v1/"
            "franco_backhand_loop_b_98e7b883b29d.table_net_clearance_certificate.json"
        ),
        "must_be_absent": True,
        "parent_must_exist": True,
        "no_clobber": True,
    }
    if plan["output_contract"] != expected_output:
        raise TableNetError("table/net output contract changed")
    expected_auth = {
        "source_gate_pass": True,
        "cpu_table_net_audit_authorized_after_review": True,
        "vendor_l1_certificate_required": True,
        "table_net_complete": False,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }
    if plan["authorization"] != expected_auth:
        raise TableNetError("table/net source authorization changed")
    expected_nonclaims = [
        "mathematical_continuous_time_table_net_clearance",
        "dynamics_balance_or_contact_stability",
        "TOPP_or_time_warp",
        "strike_or_returnability",
        "RL_training_or_checkpoint_quality",
        "Gate3_or_hardware_safety",
    ]
    if plan["explicit_non_claims"] != expected_nonclaims:
        raise TableNetError("table/net explicit non-claims changed")
    if plan["next_gate"] != "only_after_exact_table_net_certificate_vendor_dynamics_and_balance_gate":
        raise TableNetError("table/net next gate changed")
    return plan, actual_sha, l1_plan


def validate_vendor_l1_certificate(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot = _snapshot_absolute_sha(
        plan["frozen_vendor_l1"]["certificate"], "vendor L1 certificate"
    )
    cert = parse_json_bytes(snapshot.data, "vendor L1 certificate")
    exact_keys(
        cert,
        {
            "schema_version", "status", "completed_utc", "scope", "asset_id",
            "preregistration", "validator", "runtime", "lineage", "audit", "authorization",
            "explicit_non_claims", "next_gate",
        },
        "vendor L1 certificate",
    )
    if (
        cert["schema_version"] != 1
        or cert["status"] != plan["frozen_vendor_l1"]["required_certificate_status"]
        or cert["asset_id"] != ASSET_ID
    ):
        raise TableNetError("vendor L1 certificate identity/status changed")
    prereg = exact_keys(
        cert["preregistration"], {"path", "sha256"}, "vendor L1 certificate preregistration"
    )
    if (
        prereg["sha256"] != plan["frozen_vendor_l1"]["preregistration"]["sha256"]
        or Path(str(prereg["path"])).name
        != Path(plan["frozen_vendor_l1"]["preregistration"]["path"]).name
    ):
        raise TableNetError("vendor L1 certificate preregistration binding changed")
    if cert["validator"] != plan["frozen_vendor_l1"]["validator"]:
        raise TableNetError("vendor L1 certificate validator binding changed")
    lineage = cert["lineage"]
    if not isinstance(lineage, dict):
        raise TableNetError("vendor L1 certificate lineage must be an object")
    motion = lineage.get("motion_npz")
    if not isinstance(motion, dict) or {
        "path": motion.get("path"), "sha256": motion.get("sha256")
    } != plan["exact_runtime_input"]:
        raise TableNetError("vendor L1 certificate schema-2 lineage changed")
    canonical = lineage.get("canonical_mjcf")
    expected_canonical = plan["a3_model"]["canonical_mjcf"]
    if (
        not isinstance(canonical, dict)
        or set(canonical) != {"path", "bytes", "sha256"}
        or canonical.get("bytes") != expected_canonical["bytes"]
        or canonical.get("sha256") != expected_canonical["sha256"]
        or not str(canonical.get("path", "")).endswith("/" + expected_canonical["path"])
    ):
        raise TableNetError("vendor L1 certificate canonical MJCF lineage changed")
    if lineage.get("derived_mjcf_closure") != plan["a3_model"]["derived_closure"]:
        raise TableNetError("vendor L1 certificate derived MJCF closure changed")
    if lineage.get("compiled_collision_sha256") != plan["a3_model"]["compiled_collision_contract_sha256"]:
        raise TableNetError("vendor L1 certificate compiled collision lineage changed")
    audit = cert["audit"]
    if not isinstance(audit, dict) or audit.get("mj_step_calls") != 0:
        raise TableNetError("vendor L1 certificate audit or mj_step boundary changed")
    sampling = audit.get("sampling")
    if not isinstance(sampling, dict) or {
        "dense_frames": sampling.get("dense_frames"),
        "effective_sampling_hz": sampling.get("effective_sampling_hz"),
        "continuous_time_certificate": sampling.get("continuous_time_certificate"),
    } != {"dense_frames": 1201, "effective_sampling_hz": 400, "continuous_time_certificate": False}:
        raise TableNetError("vendor L1 certificate sampling boundary changed")
    for field in ("self_collision", "racket_body_clearance", "hard_gate"):
        value = audit.get(field)
        if not isinstance(value, dict) or int(value.get("dangerous_dense_samples", -1)) != 0:
            raise TableNetError(f"vendor L1 certificate {field} is not a zero-hard-event pass")
    expected_auth = {
        "l0_static_complete": True,
        **plan["frozen_vendor_l1"]["required_authorization"],
    }
    if cert["authorization"] != expected_auth:
        raise TableNetError("vendor L1 certificate authorization changed")
    return cert, snapshot


def _scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise TableNetError(f"{label} must be one non-object scalar string")
    item = array.item()
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TableNetError(f"{label} is not UTF-8") from exc
    if not isinstance(item, str):
        raise TableNetError(f"{label} is not text")
    return item


def _read_names_snapshot(snapshot: FileSnapshot, count: int, label: str) -> tuple[str, ...]:
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TableNetError(f"{label} is not UTF-8") from exc
    names = tuple(line.strip() for line in text.splitlines() if line.strip())
    if len(names) != count or len(set(names)) != count:
        raise TableNetError(f"{label} must contain exactly {count} unique names")
    return names


def load_schema2_npz_snapshot(
    snapshot: FileSnapshot,
    plan: Mapping[str, Any],
    body_names_expected: Sequence[str],
) -> dict[str, np.ndarray]:
    """Load the schema-2 archive exclusively from its already-bound bytes."""

    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.data)) as archive:
            members = archive.namelist()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise TableNetError(f"schema2 NPZ is not a valid ZIP: {exc}") from exc
    expected_members = {f"{name}.npy" for name in NPZ_FIELDS}
    if len(members) != len(set(members)) or set(members) != expected_members:
        raise TableNetError("schema2 NPZ members are missing, duplicated or unexpected")

    contract = plan["l0_contract"]
    expected_shapes = {
        "joint_pos": (151, 31),
        "joint_vel": (151, 31),
        "body_pos_w": (151, 32, 3),
        "body_quat_w": (151, 32, 4),
        "body_lin_vel_w": (151, 32, 3),
        "body_ang_vel_w": (151, 32, 3),
    }
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(io.BytesIO(snapshot.data), allow_pickle=False) as data:
            if set(data.files) != NPZ_FIELDS or len(data.files) != len(NPZ_FIELDS):
                raise TableNetError("schema2 NPZ field set changed")
            fps = np.asarray(data["fps"])
            schema = np.asarray(data["kinematics_schema_version"])
            if fps.shape != (1,) or fps.dtype != np.int64 or int(fps[0]) != contract["fps"]:
                raise TableNetError("schema2 fps must be exact int64 [50]")
            if schema.shape != (1,) or schema.dtype != np.int64 or int(schema[0]) != 2:
                raise TableNetError("schema2 version must be exact int64 [2]")
            if _scalar_text(data["body_pos_point"], "body_pos_point") != "link_origin":
                raise TableNetError("schema2 body_pos_point must be link_origin")
            if (
                _scalar_text(data["body_lin_vel_point"], "body_lin_vel_point")
                != "center_of_mass"
            ):
                raise TableNetError("schema2 body_lin_vel_point must be center_of_mass")
            names_raw = np.asarray(data["body_names"])
            if names_raw.shape != (32,) or names_raw.dtype.hasobject:
                raise TableNetError("schema2 body_names shape/dtype changed")
            names = tuple(
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in names_raw.tolist()
            )
            if names != tuple(body_names_expected):
                raise TableNetError("schema2 body_names differ from exact runtime order")
            for name, shape in expected_shapes.items():
                array = np.asarray(data[name])
                if array.shape != shape or array.dtype != np.float32:
                    raise TableNetError(
                        f"schema2 {name} shape/dtype {array.shape}/{array.dtype} "
                        f"!= {shape}/float32"
                    )
                if not np.isfinite(array).all():
                    raise TableNetError(f"schema2 {name} contains NaN/Inf")
                arrays[name] = array.copy()
    except TableNetError:
        raise
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile) as exc:
        raise TableNetError(f"cannot load exact schema2 NPZ bytes: {exc}") from exc

    quat_error = float(
        np.max(np.abs(np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1) - 1.0))
    )
    if quat_error > contract["quaternion_norm_tolerance"]:
        raise TableNetError(
            f"body quaternion max norm error {quat_error:.9g} exceeds frozen tolerance"
        )
    dt = 1.0 / float(contract["fps"])
    expected_joint_vel = np.gradient(arrays["joint_pos"], dt, axis=0).astype(np.float32)
    if not np.array_equal(arrays["joint_vel"], expected_joint_vel):
        raise TableNetError("joint_vel is not producer-exact gradient(joint_pos, 1/50)")
    arrays["_quaternion_max_norm_error"] = np.asarray(quat_error)
    return arrays


def _format_vec(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def augment_mjcf_xml(canonical_xml: bytes, obstacle_geometry: Mapping[str, Any]) -> bytes:
    """Append four world-fixed boxes in memory, preserving all canonical geom IDs."""

    if b"<!DOCTYPE" in canonical_xml or b"<!ENTITY" in canonical_xml:
        raise TableNetError("canonical MJCF contains forbidden DTD/entity declarations")
    try:
        root = ET.fromstring(canonical_xml)
    except ET.ParseError as exc:
        raise TableNetError(f"cannot parse canonical MJCF for augmentation: {exc}") from exc
    worldbodies = root.findall("./worldbody")
    if len(worldbodies) != 1:
        raise TableNetError("canonical MJCF must contain exactly one worldbody")
    existing = {node.get("name") for node in root.iter("geom") if node.get("name")}
    rows = _obstacle_rows(obstacle_geometry)
    if [row["name"] for row in rows] != list(OBSTACLE_NAMES):
        raise TableNetError("obstacle append order changed")
    for row in rows:
        if row["name"] in existing:
            raise TableNetError(f"obstacle geom name already exists: {row['name']}")
        center = row["center_mjcf_world_m"]
        extents = row["full_extents_m"]
        if len(center) != 3 or len(extents) != 3 or any(
            not math.isfinite(float(value)) for value in [*center, *extents]
        ) or any(float(value) <= 0.0 for value in extents):
            raise TableNetError(f"obstacle {row['name']} has invalid center/extents")
        ET.SubElement(
            worldbodies[0],
            "geom",
            {
                "name": row["name"],
                "type": "box",
                "pos": _format_vec(center),
                "size": _format_vec([float(value) / 2.0 for value in extents]),
                "contype": "0",
                "conaffinity": "0",
                "group": "6",
            },
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _snapshot_exact_mjcf_closure(
    plan: Mapping[str, Any],
) -> tuple[FileSnapshot, dict[str, FileSnapshot]]:
    """Snapshot canonical XML and all 74 meshes beneath one pinned model-root dirfd."""

    canonical_binding = plan["a3_model"]["canonical_mjcf"]
    canonical_relative = Path(canonical_binding["path"])
    model_root = (REPO_ROOT / canonical_relative).parent
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(model_root, flags)
    except OSError as exc:
        raise TableNetError(f"cannot bind vendor MJCF model root: {exc}") from exc
    try:
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise TableNetError("vendor MJCF model root is not a directory")
        canonical = _read_relative_snapshot(
            root_fd,
            model_root,
            Path(canonical_relative.name),
            "canonical vendor MJCF",
            expected_bytes=canonical_binding["bytes"],
            expected_sha256=canonical_binding["sha256"],
        )
        if b"<!DOCTYPE" in canonical.data or b"<!ENTITY" in canonical.data:
            raise TableNetError("canonical vendor MJCF contains forbidden DTD/entity declarations")
        try:
            root = ET.fromstring(canonical.data)
        except ET.ParseError as exc:
            raise TableNetError(f"cannot parse canonical vendor MJCF: {exc}") from exc
        if list(root.iter("include")):
            raise TableNetError("canonical vendor MJCF unexpectedly gained include files")
        compiler = root.find("./compiler")
        if compiler is None or compiler.get("meshdir") != "meshes":
            raise TableNetError("canonical MJCF meshdir changed")
        mesh_nodes = list(root.findall("./asset/mesh"))
        meshes: dict[str, FileSnapshot] = {}
        for node in mesh_nodes:
            raw = node.get("file")
            if not raw:
                raise TableNetError("canonical MJCF mesh lacks file")
            relative = Path("meshes") / Path(raw)
            key = relative.as_posix()
            if key in meshes:
                raise TableNetError(f"duplicate canonical MJCF mesh asset {key}")
            meshes[key] = _read_relative_snapshot(
                root_fd, model_root, relative, f"MJCF mesh {key}"
            )
        if len(meshes) != 74 or len(mesh_nodes) != 74:
            raise TableNetError("in-memory MJCF asset map is not the exact 74-file closure")
        rows = [
            {
                "path": canonical.path.relative_to(model_root).as_posix(),
                "bytes": canonical.size,
                "sha256": canonical.sha256,
            },
            *[
                {
                    "path": key,
                    "bytes": snapshot.size,
                    "sha256": snapshot.sha256,
                }
                for key, snapshot in meshes.items()
            ],
        ]
        rows.sort(key=lambda row: row["path"])
        manifest = _sha256_bytes(
            json.dumps(
                rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        observed = {
            "algorithm": "sha256(canonical-json(sorted[{path,bytes,sha256}]))-v1",
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "manifest_sha256": manifest,
            "xml_file_count": 1,
            "include_reference_count": 0,
            "external_file_reference_count": len(mesh_nodes),
            "unique_external_file_count": len(meshes),
            "mesh_reference_count": len(mesh_nodes),
        }
        if observed != plan["a3_model"]["derived_closure"]:
            raise TableNetError(f"immutable MJCF closure changed: actual={observed}")
        # The path itself may be renamed after snapshot, but no future consumer reopens it.
        return canonical, meshes
    finally:
        os.close(root_fd)


def _compile_canonical_model(
    mujoco: Any,
    canonical_source: FileSnapshot,
    mesh_sources: Mapping[str, FileSnapshot],
) -> tuple[Any, Any]:
    assets = {name: snapshot.data for name, snapshot in mesh_sources.items()}
    if len(assets) != 74:
        raise TableNetError("in-memory MJCF asset map is not the exact 74-file closure")
    try:
        model = mujoco.MjModel.from_xml_string(
            canonical_source.data.decode("utf-8"), assets=assets
        )
        return model, mujoco.MjData(model)
    except Exception as exc:
        raise TableNetError(f"cannot compile immutable canonical vendor MJCF bytes: {exc}") from exc


def _bind_compiled_model(
    mujoco: Any, ground: Any, model: Any, data: Any, *, ground_geom_name: str
) -> Any:
    """Apply the frozen grounding helper's binding checks to an in-memory model."""

    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    free_joint_ids = [
        jid for jid in range(int(model.njnt)) if int(model.jnt_type[jid]) == free_type
    ]
    if len(free_joint_ids) != 1:
        raise TableNetError(
            f"MJCF must contain exactly one free joint, got {len(free_joint_ids)}"
        )
    root_joint_id = free_joint_ids[0]
    root_body_id = int(model.jnt_bodyid[root_joint_id])
    if root_body_id == 0:
        raise TableNetError("free joint is attached to the world body")
    subtree_hinges = [
        jid
        for jid in range(int(model.njnt))
        if int(model.jnt_type[jid]) == hinge_type
        and ground._descends_from(model, int(model.jnt_bodyid[jid]), root_body_id)
    ]
    names = tuple(
        ground._name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, jid, f"joint{jid}")
        for jid in subtree_hinges
    )
    if names != tuple(ground.A3_GMR_JOINT_NAMES) or len(set(names)) != len(names):
        raise TableNetError("compiled MJCF hinge order differs from A3_GMR_JOINT_NAMES")
    qpos_addresses = tuple(int(model.jnt_qposadr[jid]) for jid in subtree_hinges)
    if len(set(qpos_addresses)) != len(qpos_addresses):
        raise TableNetError("compiled MJCF hinge qpos addresses are not unique")
    for jid, name in zip(subtree_hinges, names):
        if hasattr(model, "jnt_limited") and not bool(model.jnt_limited[jid]):
            raise TableNetError(f"compiled MJCF joint {name!r} is not range-limited")

    ground_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground_geom_name))
    if (
        ground_gid < 0
        or int(model.geom_type[ground_gid]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
        or int(model.geom_bodyid[ground_gid]) != 0
    ):
        raise TableNetError("compiled MJCF floor is missing or not a world-fixed plane")
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    rotation = np.asarray(data.geom_xmat[ground_gid], dtype=np.float64).reshape(3, 3)
    if not np.allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-10, rtol=0.0):
        raise TableNetError("compiled MJCF floor normal is not +Z")
    ground_z_m = float(np.asarray(data.geom_xpos[ground_gid], dtype=np.float64)[2])
    if not math.isfinite(ground_z_m):
        raise TableNetError("compiled MJCF floor z is non-finite")

    supported = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_MESH),
    }
    collision_geom_ids: list[int] = []
    for gid in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[gid])
        if not ground._descends_from(model, body_id, root_body_id):
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        if int(model.geom_type[gid]) not in supported:
            raise TableNetError(f"enabled robot collision geom {gid} has unsupported type")
        collision_geom_ids.append(gid)
    if not collision_geom_ids:
        raise TableNetError("compiled MJCF robot subtree has no enabled collision geoms")
    return ground.ModelBinding(
        model=model,
        data=data,
        root_joint_id=root_joint_id,
        root_body_id=root_body_id,
        root_qpos_address=int(model.jnt_qposadr[root_joint_id]),
        joint_ids=tuple(subtree_hinges),
        joint_qpos_addresses=qpos_addresses,
        collision_geom_ids=tuple(collision_geom_ids),
        ground_geom_id=ground_gid,
        ground_z_m=ground_z_m,
        collision_contract_sha256=ground._compiled_collision_contract_sha256(
            model, collision_geom_ids
        ),
    )


def _compile_augmented_model(
    mujoco: Any,
    ground: Any,
    canonical_binding: Any,
    canonical_source: FileSnapshot,
    mesh_sources: Mapping[str, FileSnapshot],
    plan: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, int], dict[str, Any]]:
    canonical_xml = canonical_source.data
    augmented_xml = augment_mjcf_xml(canonical_xml, plan["obstacle_geometry"])
    assets = {name: snapshot.data for name, snapshot in mesh_sources.items()}
    if len(assets) != 74:
        raise TableNetError("in-memory MJCF asset map is not the exact 74-file closure")
    try:
        model = mujoco.MjModel.from_xml_string(augmented_xml.decode("utf-8"), assets=assets)
        data = mujoco.MjData(model)
    except Exception as exc:
        raise TableNetError(f"cannot compile in-memory table/net augmented MJCF: {exc}") from exc
    canonical_model = canonical_binding.model
    if (
        int(model.ngeom) != int(canonical_model.ngeom) + 4
        or int(model.nbody) != int(canonical_model.nbody)
        or int(model.nq) != int(canonical_model.nq)
        or int(model.nv) != int(canonical_model.nv)
        or not np.array_equal(model.qpos0, canonical_model.qpos0)
    ):
        raise TableNetError("in-memory augmentation changed robot model topology or qpos0")
    robot_ids = tuple(int(value) for value in canonical_binding.collision_geom_ids)
    for geom_id in robot_ids:
        old = mujoco.mj_id2name(canonical_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        new = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if old != new:
            raise TableNetError("in-memory augmentation reordered canonical robot geoms")
    collision_sha = ground._compiled_collision_contract_sha256(model, robot_ids)
    if collision_sha != plan["a3_model"]["compiled_collision_contract_sha256"]:
        raise TableNetError("augmented model changed compiled robot collision contract")
    obstacle_ids: dict[str, int] = {}
    mujoco.mj_forward(model, data)
    expected_rows = {row["name"]: row for row in _obstacle_rows(plan["obstacle_geometry"])}
    for name in OBSTACLE_NAMES:
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if geom_id < 0 or geom_id in robot_ids:
            raise TableNetError(f"augmented obstacle geom missing or aliases robot: {name}")
        if (
            int(model.geom_bodyid[geom_id]) != 0
            or int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX)
            or int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        ):
            raise TableNetError(f"augmented obstacle {name} is not an inert world-fixed box")
        row = expected_rows[name]
        if not np.allclose(data.geom_xpos[geom_id], row["center_mjcf_world_m"], atol=1e-12, rtol=0.0):
            raise TableNetError(f"augmented obstacle {name} center changed")
        if not np.allclose(
            model.geom_size[geom_id, :3],
            np.asarray(row["full_extents_m"], dtype=np.float64) / 2.0,
            atol=1e-12,
            rtol=0.0,
        ):
            raise TableNetError(f"augmented obstacle {name} extents changed")
        obstacle_ids[name] = geom_id
    evidence = {
        "assembly": plan["audit_contract"]["model_augmentation"],
        "canonical_ngeom": int(canonical_model.ngeom),
        "augmented_ngeom": int(model.ngeom),
        "robot_geom_ids_preserved": True,
        "robot_collision_contract_sha256": collision_sha,
        "asset_map_file_count": len(assets),
        "obstacle_geom_ids": obstacle_ids,
        "mj_step_calls": 0,
    }
    return model, data, obstacle_ids, evidence


def evaluate_robot_obstacle_pairs(
    helper: Any,
    model: Any,
    data: Any,
    *,
    robot_ids: Sequence[int],
    racket_ids: Sequence[int],
    obstacle_ids: Mapping[str, int],
    hard_threshold_m: float,
    warning_threshold_m: float,
    reporting_tolerance_m: float,
    geom_name: Any,
    reporting_cap_m: float = 0.1,
) -> dict[str, Any]:
    """Check every robot/obstacle pair with the exact saturation predicate."""

    robots = tuple(int(value) for value in robot_ids)
    rackets = set(int(value) for value in racket_ids)
    obstacles = {str(name): int(value) for name, value in obstacle_ids.items()}
    if (
        len(robots) == 0
        or len(set(robots)) != len(robots)
        or not rackets.issubset(set(robots))
        or not obstacles
        or any(name not in OBSTACLE_NAMES for name in obstacles)
        or len(set(obstacles.values())) != len(obstacles)
    ):
        raise TableNetError("robot/racket/obstacle geom sets are malformed")
    if not (
        math.isfinite(hard_threshold_m)
        and math.isfinite(warning_threshold_m)
        and math.isfinite(reporting_tolerance_m)
        and math.isfinite(reporting_cap_m)
        and 0.0 < reporting_tolerance_m < hard_threshold_m < warning_threshold_m < reporting_cap_m
    ):
        raise TableNetError("clearance thresholds/tolerances are invalid")
    hard_pairs: list[list[str]] = []
    warning_pairs: list[list[str]] = []
    racket_hard_pairs: list[list[str]] = []
    minimum: float | None = None
    minimum_pair: list[str] | None = None
    for robot in robots:
        for obstacle_name, obstacle in obstacles.items():
            pair = [geom_name(robot), obstacle_name]
            if not bool(helper._far(model, data, robot, obstacle, hard_threshold_m)):
                hard_pairs.append(pair)
                if robot in rackets:
                    racket_hard_pairs.append(pair)
            if not bool(helper._far(model, data, robot, obstacle, warning_threshold_m)):
                warning_pairs.append(pair)
            if not bool(helper._far(model, data, robot, obstacle, reporting_cap_m)):
                distance, _ = helper.geom_clearance(
                    model,
                    data,
                    robot,
                    obstacle,
                    distmax=reporting_cap_m,
                    tol=reporting_tolerance_m,
                )
                distance = float(distance)
                if not math.isfinite(distance):
                    raise TableNetError("robot/obstacle clearance produced a non-finite distance")
                if minimum is None or distance < minimum:
                    minimum = distance
                    minimum_pair = pair
    return {
        "pair_count": len(robots) * len(obstacles),
        "hard_failure": bool(hard_pairs),
        "warning": bool(warning_pairs),
        "racket_or_handle_hard_failure": bool(racket_hard_pairs),
        "hard_pairs": hard_pairs,
        "warning_pairs": warning_pairs,
        "racket_or_handle_hard_pairs": racket_hard_pairs,
        "minimum_clearance_m": minimum,
        "minimum_clearance_lower_bound_m": minimum if minimum is not None else reporting_cap_m,
        "minimum_pair": minimum_pair,
        "reporting_cap_m": reporting_cap_m,
    }


def summarize_dense_failures(
    all_robot_bad: np.ndarray,
    racket_bad: np.ndarray,
    source_time: np.ndarray,
    *,
    source_frames: int,
    unsafe_source_mask_fn: Any,
    first_hard_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    all_bad = np.asarray(all_robot_bad, dtype=bool)
    racket = np.asarray(racket_bad, dtype=bool)
    times = np.asarray(source_time, dtype=np.float64)
    if all_bad.shape != racket.shape or all_bad.shape != times.shape or np.any(racket & ~all_bad):
        raise TableNetError("dense table/net masks/time are inconsistent")
    unsafe = np.asarray(unsafe_source_mask_fn(source_frames, times, all_bad), dtype=bool)
    if unsafe.shape != (source_frames,):
        raise TableNetError("unsafe source mask shape changed")
    result = {
        "dangerous_dense_samples": int(np.count_nonzero(all_bad)),
        "racket_or_handle_dangerous_dense_samples": int(np.count_nonzero(racket)),
        "unsafe_source_frames": int(np.count_nonzero(unsafe)),
        "unsafe_source_indices": np.flatnonzero(unsafe).astype(int).tolist(),
        "hard_fail_is_noncompensable": True,
    }
    if result["dangerous_dense_samples"]:
        raise TableNetError(
            "table/net clearance hard failure: "
            f"all_robot={result['dangerous_dense_samples']} "
            f"racket_or_handle={result['racket_or_handle_dangerous_dense_samples']} "
            f"first_event={dict(first_hard_event) if first_hard_event is not None else None}"
        )
    return result


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return value if value is not None else f"geom{geom_id}"


def _verify_output_parent_identity(target: OutputTarget) -> None:
    try:
        observed = os.stat(target.parent_path, follow_symlinks=False)
    except OSError as exc:
        raise TableNetError(f"certificate parent path changed after binding: {exc}") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or int(observed.st_dev) != target.parent_device
        or int(observed.st_ino) != target.parent_inode
    ):
        raise TableNetError("certificate parent path no longer names the bound directory inode")


def validate_output_preconditions(plan: Mapping[str, Any]) -> OutputTarget:
    output = Path(plan["output_contract"]["certificate_path"])
    if not output.is_absolute() or output.name in ("", ".", ".."):
        raise TableNetError("certificate path must be an absolute regular filename")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        dir_fd = os.open(output.parent, flags)
    except OSError as exc:
        raise TableNetError(
            f"certificate parent must pre-exist and be a real directory: {output.parent}: {exc}"
        ) from exc
    try:
        parent = os.fstat(dir_fd)
        if not stat.S_ISDIR(parent.st_mode):
            raise TableNetError("certificate parent is not a directory")
        try:
            os.stat(output.name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise TableNetError(f"cannot inspect certificate output slot: {exc}") from exc
        else:
            raise TableNetError(
                f"certificate path already exists or is a symlink; no-clobber: {output}"
            )
        target = OutputTarget(
            path=output,
            parent_path=output.parent,
            name=output.name,
            dir_fd=dir_fd,
            parent_device=int(parent.st_dev),
            parent_inode=int(parent.st_ino),
        )
        _verify_output_parent_identity(target)
        return target
    except BaseException:
        os.close(dir_fd)
        raise


def audit_runtime(plan: Mapping[str, Any]) -> dict[str, Any]:
    frozen = plan["frozen_vendor_l1"]
    l1_plan_snapshot = _snapshot_repo_binding(
        frozen["preregistration"],
        "vendor L1 preregistration",
        "configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json",
    )
    secure_l1_plan = parse_json_bytes(l1_plan_snapshot.data, "vendor L1 preregistration")
    l1 = _load_exact_module(
        "motion_vendor_l1_for_table_net_runtime",
        L1_VALIDATOR_PATH,
        expected_bytes=frozen["validator"]["bytes"],
        expected_sha256=frozen["validator"]["sha256"],
        label="vendor L1 validator",
    )
    try:
        l1_plan, _, l0_v1_plan = l1.validate_plan(
            l1_plan_snapshot.path, frozen["preregistration"]["sha256"]
        )
        if l1_plan != secure_l1_plan:
            raise TableNetError(
                "vendor L1 preregistration changed between bound bytes and runtime validation"
            )
        l0_v2_binding = secure_l1_plan["frozen_l0"]["preregistration"]
        l0_v2_snapshot = _snapshot_repo_binding(
            l0_v2_binding,
            "vendor L0 V2 preregistration",
            l0_v2_binding["path"],
        )
        secure_l0_v2 = parse_json_bytes(l0_v2_snapshot.data, "vendor L0 V2 preregistration")
        l0_v1_binding = secure_l0_v2["frozen_v1"]["preregistration"]
        l0_v1_snapshot = _snapshot_repo_binding(
            l0_v1_binding,
            "vendor L0 V1 preregistration",
            l0_v1_binding["path"],
        )
        if l0_v1_plan != parse_json_bytes(
            l0_v1_snapshot.data, "vendor L0 V1 preregistration"
        ):
            raise TableNetError(
                "vendor L0 preregistration changed between bound bytes and runtime validation"
            )
        l0_v1_validator_binding = secure_l0_v2["frozen_v1"]["validator"]
        l0_v1_validator_snapshot = _snapshot_repo_binding(
            l0_v1_validator_binding,
            "vendor L0 V1 validator",
            l0_v1_validator_binding["path"],
        )
        l0_v1 = _load_exact_module(
            "motion_schema2_l0_v1_for_table_net_runtime",
            l0_v1_validator_snapshot.path,
            expected_bytes=l0_v1_validator_snapshot.size,
            expected_sha256=l0_v1_validator_snapshot.sha256,
            label="vendor L0 V1 validator",
        )
        mujoco, runtime = l0_v1.validate_runtime_environment(l0_v1_plan)
    except (OSError, TypeError, ValueError) as exc:
        raise TableNetError(f"exact vendor L1 CPU runtime validation failed: {exc}") from exc
    l1_certificate, l1_certificate_snapshot = validate_vendor_l1_certificate(plan)
    npz_snapshot = _snapshot_absolute_sha(
        plan["exact_runtime_input"], "exact B schema-2 NPZ"
    )
    body_order_binding = l0_v1_plan["upstream_contracts"]["runtime_body_order"]
    body_order_snapshot = _snapshot_repo_binding(
        body_order_binding,
        "schema-2 runtime body order",
        body_order_binding["path"],
    )
    body_names = _read_names_snapshot(body_order_snapshot, 32, "schema-2 runtime body order")
    arrays = load_schema2_npz_snapshot(npz_snapshot, l0_v1_plan, body_names)

    phase_binding = l1_plan["dependencies"]["dense_safety_tool"]
    phase = _load_exact_module(
        "motion_phase_safety_for_table_net",
        REPO_ROOT / phase_binding["path"],
        expected_bytes=phase_binding["bytes"],
        expected_sha256=phase_binding["sha256"],
        label="dense safety tool",
    )
    helper_binding = l1_plan["dependencies"]["self_collision_helper"]
    helper = _load_exact_module(
        "motion_self_collision_for_table_net",
        REPO_ROOT / helper_binding["path"],
        expected_bytes=helper_binding["bytes"],
        expected_sha256=helper_binding["sha256"],
        label="self-collision distance helper",
    )
    ground_binding = l1_plan["dependencies"]["grounding_helper"]
    ground = _load_exact_module(
        "ground_gmr_pkl_for_table_net",
        REPO_ROOT / ground_binding["path"],
        expected_bytes=ground_binding["bytes"],
        expected_sha256=ground_binding["sha256"],
        label="grounding helper",
    )
    canonical_source, mesh_sources = _snapshot_exact_mjcf_closure(plan)
    canonical_model, canonical_data = _compile_canonical_model(
        mujoco, canonical_source, mesh_sources
    )
    canonical_binding = _bind_compiled_model(
        mujoco, ground, canonical_model, canonical_data, ground_geom_name="floor"
    )
    if (
        canonical_binding.collision_contract_sha256
        != plan["a3_model"]["compiled_collision_contract_sha256"]
        or len(canonical_binding.collision_geom_ids) != 37
    ):
        raise TableNetError("canonical compiled robot collision contract changed at runtime")
    model, data, obstacle_ids, assembly = _compile_augmented_model(
        mujoco, ground, canonical_binding, canonical_source, mesh_sources, plan
    )
    augmented_binding = ground.ModelBinding(
        model=model,
        data=data,
        root_joint_id=canonical_binding.root_joint_id,
        root_body_id=canonical_binding.root_body_id,
        root_qpos_address=canonical_binding.root_qpos_address,
        joint_ids=canonical_binding.joint_ids,
        joint_qpos_addresses=canonical_binding.joint_qpos_addresses,
        collision_geom_ids=canonical_binding.collision_geom_ids,
        ground_geom_id=canonical_binding.ground_geom_id,
        ground_z_m=canonical_binding.ground_z_m,
        collision_contract_sha256=canonical_binding.collision_contract_sha256,
    )

    joint_order_binding = l0_v1_plan["upstream_contracts"]["runtime_joint_order"]
    joint_order_snapshot = _snapshot_repo_binding(
        joint_order_binding,
        "schema-2 runtime joint order",
        joint_order_binding["path"],
    )
    runtime_joint_names = _read_names_snapshot(
        joint_order_snapshot, 31, "schema-2 runtime joint order"
    )
    reordered, joint_adapter = l1.reorder_runtime_joint_pos_for_ground(
        arrays["joint_pos"], runtime_joint_names, tuple(ground.A3_GMR_JOINT_NAMES)
    )
    payload = {
        "root_pos": np.asarray(arrays["body_pos_w"][:, 0], dtype=np.float64),
        "root_rot": np.asarray(arrays["body_quat_w"][:, 0], dtype=np.float64)[:, [1, 2, 3, 0]],
        "dof_pos": np.asarray(reordered, dtype=np.float64),
        "fps": np.array([50.0], dtype=np.float64),
    }
    audit_contract = plan["audit_contract"]
    dense, source_time = phase.densify_payload(
        payload, audit_contract["substeps_per_source_interval"]
    )
    if (
        dense["root_pos"].shape != (1201, 3)
        or dense["dof_pos"].shape != (1201, 31)
        or float(np.asarray(dense["fps"]).reshape(-1)[0]) != 400.0
        or not all(
            np.isfinite(np.asarray(dense[key])).all()
            for key in ("root_pos", "root_rot", "dof_pos")
        )
    ):
        raise TableNetError("dense table/net trajectory structure/rate/finite contract changed")
    ground.validate_joint_ranges(
        dense,
        augmented_binding,
        tolerance_rad=l1_plan["safety_contract"]["joint_range_tolerance_rad"],
    )
    qpos = phase._qpos_from_payload(augmented_binding, dense)
    robot_ids = tuple(int(value) for value in augmented_binding.collision_geom_ids)
    geom_by_name = {_geom_name(mujoco, model, geom_id): geom_id for geom_id in robot_ids}
    if any(name not in geom_by_name for name in RACKET_GEOMS):
        raise TableNetError("augmented vendor MJCF lacks racket/handle collision geoms")
    racket_ids = tuple(geom_by_name[name] for name in RACKET_GEOMS)

    all_bad = np.zeros(1201, dtype=bool)
    racket_bad = np.zeros(1201, dtype=bool)
    warnings = np.zeros(1201, dtype=bool)
    minimum_value: float | None = None
    minimum_pair: list[str] | None = None
    minimum_source_time: float | None = None
    warning_events: list[dict[str, Any]] = []
    hard_events: list[dict[str, Any]] = []
    for dense_frame in range(1201):
        data.qpos[:] = qpos[dense_frame]
        mujoco.mj_forward(model, data)
        result = evaluate_robot_obstacle_pairs(
            helper,
            model,
            data,
            robot_ids=robot_ids,
            racket_ids=racket_ids,
            obstacle_ids=obstacle_ids,
            hard_threshold_m=audit_contract["hard_clearance_m"],
            warning_threshold_m=audit_contract["warning_clearance_m"],
            reporting_tolerance_m=audit_contract["reporting_clearance_bisection_tolerance_m"],
            reporting_cap_m=audit_contract["reporting_clearance_cap_m"],
            geom_name=lambda geom_id: _geom_name(mujoco, model, geom_id),
        )
        all_bad[dense_frame] = result["hard_failure"]
        racket_bad[dense_frame] = result["racket_or_handle_hard_failure"]
        warnings[dense_frame] = result["warning"]
        value = result["minimum_clearance_m"]
        if value is not None and (minimum_value is None or value < minimum_value):
            minimum_value = float(value)
            minimum_pair = result["minimum_pair"]
            minimum_source_time = float(source_time[dense_frame])
        if result["hard_failure"] and len(hard_events) < 512:
            hard_events.append(
                {
                    "dense_frame": dense_frame,
                    "source_time_frames": float(source_time[dense_frame]),
                    "pairs": result["hard_pairs"],
                }
            )
        if result["warning"] and len(warning_events) < 512:
            warning_events.append(
                {
                    "dense_frame": dense_frame,
                    "source_time_frames": float(source_time[dense_frame]),
                    "pairs": result["warning_pairs"],
                }
            )
    hard_gate = summarize_dense_failures(
        all_bad,
        racket_bad,
        source_time,
        source_frames=151,
        unsafe_source_mask_fn=phase.unsafe_source_mask,
        first_hard_event=hard_events[0] if hard_events else None,
    )
    frame_evidence = validate_frame_and_geometry_sources(plan)
    return {
        "runtime": runtime,
        "lineage": {
            "vendor_l1_certificate": _binding(l1_certificate_snapshot),
            "vendor_l1_certificate_authorization": l1_certificate["authorization"],
            "vendor_l1_preregistration": frozen["preregistration"],
            "vendor_l1_validator": frozen["validator"],
            "motion_npz": _binding(npz_snapshot),
            "canonical_mjcf": _binding(canonical_source),
            "derived_mjcf_closure": plan["a3_model"]["derived_closure"],
            "compiled_robot_collision_sha256": augmented_binding.collision_contract_sha256,
        },
        "frame_and_obstacles": frame_evidence,
        "model_assembly": assembly,
        "audit": {
            "joint_order_adapter": joint_adapter,
            "sampling": {
                "source_frames": 151,
                "source_fps": 50,
                "dense_frames": 1201,
                "substeps_per_source_interval": 8,
                "effective_sampling_hz": 400,
                "interpolation": audit_contract["interpolation"],
                "continuous_time_certificate": False,
            },
            "all_enabled_robot_to_table_net": {
                "enabled_robot_geom_count": 37,
                "obstacle_geom_count": 4,
                "pairs_per_dense_sample": 148,
                "hard_threshold_m": audit_contract["hard_clearance_m"],
                "warning_threshold_m": audit_contract["warning_clearance_m"],
                "hard_threshold_predicate": audit_contract["hard_threshold_predicate"],
                "dangerous_dense_samples": int(np.count_nonzero(all_bad)),
                "warning_dense_samples": int(np.count_nonzero(warnings)),
                "minimum_clearance_m": minimum_value,
                "minimum_clearance_lower_bound_m": (
                    minimum_value
                    if minimum_value is not None
                    else audit_contract["reporting_clearance_cap_m"]
                ),
                "minimum_pair": minimum_pair,
                "minimum_source_time_frames": minimum_source_time,
                "hard_events_truncated": len(hard_events) >= 512,
                "hard_events": hard_events,
                "warning_events_truncated": len(warning_events) >= 512,
                "warning_events": warning_events,
            },
            "racket_and_handle_rollup": {
                "geom_names": list(RACKET_GEOMS),
                "dangerous_dense_samples": int(np.count_nonzero(racket_bad)),
            },
            "hard_gate": hard_gate,
            "mj_step_calls": 0,
        },
    }


def build_certificate(
    plan: Mapping[str, Any], plan_path: Path, plan_sha: str, _l1_plan: Mapping[str, Any]
) -> dict[str, Any]:
    result = audit_runtime(plan)
    return {
        "schema_version": 1,
        "status": CERTIFICATE_STATUS,
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": plan["scope"],
        "asset_id": ASSET_ID,
        "preregistration": {"path": str(plan_path), "sha256": plan_sha},
        "validator": plan["validator"],
        **result,
        "authorization": {
            "vendor_l1_complete": True,
            "table_net_complete": True,
            "dynamics_authorized": True,
            "simulator_authorized": False,
            "training_authorized": False,
            "formal_motion_authorized": False,
            "hardware_authorized": False,
        },
        "explicit_non_claims": plan["explicit_non_claims"],
        "next_gate": plan["next_gate"],
    }


def write_exclusive(target: OutputTarget, value: Mapping[str, Any]) -> FileSnapshot:
    if target.closed:
        raise TableNetError("certificate output target is already closed")
    _verify_output_parent_identity(target)
    try:
        os.stat(target.name, dir_fd=target.dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise TableNetError(f"cannot recheck certificate output slot: {exc}") from exc
    else:
        raise TableNetError(f"certificate path already exists; no-clobber: {target.path}")
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    created: tuple[int, int] | None = None
    try:
        fd = os.open(target.name, flags, 0o444, dir_fd=target.dir_fd)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise TableNetError("new certificate output is not a regular file")
            created = (int(opened.st_dev), int(opened.st_ino))
            view = memoryview(payload)
            offset = 0
            while offset < len(view):
                written = os.write(fd, view[offset:])
                if written <= 0:
                    raise TableNetError("short certificate write")
                offset += written
            os.fsync(fd)
            after = os.fstat(fd)
            if (
                (int(after.st_dev), int(after.st_ino)) != created
                or int(after.st_size) != len(payload)
            ):
                raise TableNetError("certificate inode or size changed during publication")
        finally:
            os.close(fd)
        os.fsync(target.dir_fd)
        _verify_output_parent_identity(target)
        read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            read_flags |= os.O_NOFOLLOW
        verify_fd = os.open(target.name, read_flags, dir_fd=target.dir_fd)
        try:
            snapshot = _read_open_fd_snapshot(
                verify_fd,
                target.path,
                "published table/net certificate",
                expected_bytes=len(payload),
                expected_sha256=_sha256_bytes(payload),
            )
        finally:
            os.close(verify_fd)
        if (snapshot.device, snapshot.inode) != created or snapshot.data != payload:
            raise TableNetError("published certificate is not the exact created inode/bytes")
        return snapshot
    except BaseException:
        if created is not None:
            try:
                observed = os.stat(target.name, dir_fd=target.dir_fd, follow_symlinks=False)
                if (int(observed.st_dev), int(observed.st_ino)) == created:
                    os.unlink(target.name, dir_fd=target.dir_fd)
                    os.fsync(target.dir_fd)
            except OSError:
                pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("command", choices=("static", "dry-run", "audit"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan, plan_sha, l1_plan = validate_plan(
            args.prereg.resolve(), args.expected_prereg_sha256
        )
        if args.command == "static":
            print(
                f"[motion-table-net] PASS static asset={ASSET_ID} source_exact=true "
                "runtime_audit=false no_write=true continuous_time_claim=false"
            )
            return 0
        output = validate_output_preconditions(plan)
        try:
            certificate = build_certificate(plan, args.prereg.resolve(), plan_sha, l1_plan)
            if args.command == "dry-run":
                print(
                    f"[motion-table-net] PASS dry-run asset={ASSET_ID} runtime_audit=true "
                    "certificate_written=false table_net_complete=false downstream_blocked=true"
                )
                return 0
            published = write_exclusive(output, certificate)
            print(
                f"[motion-table-net] PASS audit asset={ASSET_ID} table_net=true "
                f"certificate_sha256={published.sha256} dynamics_next=true"
            )
            return 0
        finally:
            output.close()
    except (TableNetError, OSError, TypeError, ValueError) as exc:
        print(f"[motion-table-net] FAIL {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
