#!/usr/bin/env python3
"""Fail-closed historical attestation for the ready-to-strike stage-1 ladder.

This program only reads an already completed local stage-1 result tree.  It has
no SSH, process-control, simulator, trainer, deployment, or robot command path.
Dry-run is the default; ``--execute`` publishes exactly one O_EXCL receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA_VERSION = 1
CONFIRM_TOKEN = "ATTEST_READY_TO_STRIKE_STAGE1_ONCE"
EXPECTED_EXPERIMENT = "ready_to_strike_join_ladder_20260717"
EXPECTED_PREREG_COMMIT = "8d74025e88fee832fae0ac2f672ec0eb9b2d3d5a"
EXPECTED_CELLS = {
    "fh_rf_d17": ("forehand", "forehand", 17),
    "fh_rb_d06": ("forehand", "backhand", 6),
    "fh_rb_d17": ("forehand", "backhand", 17),
    "bh_rf_d17": ("backhand", "forehand", 17),
    "bh_rb_d06": ("backhand", "backhand", 6),
    "bh_rb_d17": ("backhand", "backhand", 17),
}
RUNTIME_RELATIVE_PATHS = {
    "generator_sha256": Path("scripts/build_ready_to_strike_motion.py"),
    "topp_sha256": Path("hope_training/whole_body_tracking/scripts/topp_mintime.py"),
    "mjcf_sha256": Path(
        "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/a3_pingpong.xml"
    ),
    "urdf_sha256": Path(
        "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"
    ),
    "body_order_sha256": Path("configs/a3_runtime_body_order.txt"),
}
TOPP_DEPENDENCIES = {
    "synthesize_timing": Path(
        "hope_training/whole_body_tracking/scripts/synthesize_timing.py"
    ),
    "synthesize_timing_v2": Path(
        "hope_training/whole_body_tracking/scripts/synthesize_timing_v2.py"
    ),
    "audit_motion_npz": Path(
        "hope_training/whole_body_tracking/scripts/audit_motion_npz.py"
    ),
}
TOPP_CERTIFICATE_KEYS = {
    "tool",
    "algorithm_scope",
    "search_objective",
    "generated_utc",
    "verdict",
    "direction",
    "chosen_scale",
    "feasible_reason",
    "budget",
    "acceptance",
    "durations",
    "oracle_before",
    "oracle_after",
    "kin",
    "source",
    "answer",
    "baseline_law",
    "stretch",
    "output",
    "timing_bound",
    "fidelity",
    "outer_trace",
    "inner_trace_best",
    "files",
    "budget_provenance",
    "runtime_provenance",
}
TOPP_TOOL = "topp_mintime.py v3 (unified-budget min-time bidirectional retiming)"
TOPP_ALGORITHM_SCOPE = (
    "heuristic upper bound within the sampled gamma ladder plus greedy local repair; "
    "not strict TOPP and not a global minimum proof"
)
SCHEMA2_TIME_KEYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}
SCHEMA2_METADATA_KEYS = {
    "fps",
    "kinematics_schema_version",
    "body_pos_point",
    "body_lin_vel_point",
    "body_names",
}
SCHEMA2_MIGRATION_KEYS = {
    "kinematics_migration_source_sha256",
    "kinematics_migration_source_point",
    "kinematics_migration_tool",
}


class AttestationError(ValueError):
    """Historical evidence is incomplete, malformed, mutable, or misbound."""


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttestationError(message)


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    actual = set(value)
    _require(
        actual == expected,
        f"{label} keys changed: missing={sorted(expected - actual)} "
        f"unexpected={sorted(actual - expected)}",
    )
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    _require(type(value) in (int, float), f"{label} must be a number, not bool")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{label} must be >= {minimum}")
    return result


def _reject_constant(value: str) -> None:
    raise AttestationError(f"JSON contains non-finite constant {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AttestationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs_no_duplicates,
        )
    except AttestationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"cannot parse {label}: {exc}") from exc


def _ensure_no_symlink_components(
    path: Path, label: str, *, leaf_may_be_missing: bool = False
) -> None:
    absolute = _absolute(path)
    current = Path(absolute.parts[0])
    for index, part in enumerate(absolute.parts[1:], start=1):
        current /= part
        leaf = index == len(absolute.parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if leaf and leaf_may_be_missing:
                return
            raise AttestationError(f"{label} path component is missing: {current}") from None
        _require(
            not stat.S_ISLNK(info.st_mode),
            f"{label} must not traverse a symlink: {current}",
        )


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


class _Snapshot:
    def __init__(self, path: Path, payload: bytes, info: os.stat_result):
        self.path = path
        self.payload = payload
        self.signature = _signature(info)
        self.evidence = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": _sha256(payload),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "mode": int(info.st_mode),
            "mtime_ns": int(info.st_mtime_ns),
            "ctime_ns": int(info.st_ctime_ns),
        }


class _AuditContext:
    def __init__(self) -> None:
        self.snapshots: dict[Path, _Snapshot] = {}

    def read(self, path_like: Path | str, label: str) -> _Snapshot:
        path = _absolute(path_like)
        _ensure_no_symlink_components(path, label)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise AttestationError(f"cannot open {label}: {path}: {exc}") from exc
        try:
            before = os.fstat(fd)
            _require(
                stat.S_ISREG(before.st_mode) and before.st_size > 0,
                f"{label} must be a non-empty regular file: {path}",
            )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
            _require(
                _signature(before) == _signature(after),
                f"{label} changed while reading: {path}",
            )
        finally:
            os.close(fd)
        outside = path.lstat()
        _require(
            _signature(after) == _signature(outside),
            f"{label} pathname changed while reading: {path}",
        )
        snapshot = _Snapshot(path, b"".join(chunks), after)
        previous = self.snapshots.get(path)
        if previous is not None:
            _require(
                previous.signature == snapshot.signature
                and previous.payload == snapshot.payload,
                f"{label} changed between reads: {path}",
            )
            return previous
        self.snapshots[path] = snapshot
        return snapshot

    def verify_all_unchanged(self) -> None:
        for path, expected in tuple(self.snapshots.items()):
            current = self.read(path, f"pre-publication snapshot {path}")
            _require(
                current.signature == expected.signature
                and current.payload == expected.payload,
                f"input changed before receipt publication: {path}",
            )


def _validate_file_evidence(
    context: _AuditContext,
    evidence: Any,
    expected_path: Path,
    label: str,
    *,
    generator_style: bool = False,
) -> _Snapshot:
    expected_keys = {"path", "bytes", "sha256"}
    if generator_style:
        expected_keys |= {"device", "inode", "mtime_ns", "ctime_ns"}
    evidence = _exact_keys(evidence, expected_keys, label)
    _require(evidence["path"] == str(expected_path), f"{label}.path is misbound")
    snapshot = context.read(expected_path, label)
    _require(evidence["bytes"] == len(snapshot.payload), f"{label}.bytes is misbound")
    _require(evidence["sha256"] == _sha256(snapshot.payload), f"{label}.sha256 is misbound")
    if generator_style:
        fields = {
            "device": snapshot.evidence["device"],
            "inode": snapshot.evidence["inode"],
            "mtime_ns": snapshot.evidence["mtime_ns"],
            "ctime_ns": snapshot.evidence["ctime_ns"],
        }
        for key, value in fields.items():
            _require(evidence[key] == value, f"{label}.{key} is misbound")
    return snapshot


def _load_npz(snapshot: _Snapshot, label: str) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.payload)) as archive:
            members = archive.namelist()
            _require(len(members) == len(set(members)), f"{label} has duplicate ZIP members")
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as archive:
            names = tuple(archive.files)
            _require(names and len(names) == len(set(names)), f"{label} fields are invalid")
            arrays = {name: np.asarray(archive[name]).copy() for name in names}
    except AttestationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise AttestationError(f"cannot load {label}: {exc}") from exc
    for name, array in arrays.items():
        _require(not array.dtype.hasobject, f"{label}.{name} must not use object dtype")
        if array.dtype.kind in "fcui":
            _require(np.isfinite(array).all(), f"{label}.{name} contains NaN/Inf")
    return arrays


def _scalar_unicode(array: np.ndarray, label: str) -> str:
    value = np.asarray(array)
    _require(
        value.shape == () and value.dtype.kind == "U" and not value.dtype.hasobject,
        f"{label} must be one canonical Unicode scalar",
    )
    return str(value.item())


def _validate_schema2_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    body_order: Sequence[str],
    label: str,
    allow_migration_provenance: bool,
) -> int:
    """Validate the exact production motion schema used by generator and TOPP."""

    required = SCHEMA2_TIME_KEYS | SCHEMA2_METADATA_KEYS
    actual = set(arrays)
    optional = actual & SCHEMA2_MIGRATION_KEYS
    if optional:
        _require(
            allow_migration_provenance and optional == SCHEMA2_MIGRATION_KEYS,
            f"{label} has partial or forbidden migration provenance",
        )
    expected = required | optional
    _require(
        actual == expected,
        f"{label} schema keys changed: missing={sorted(expected - actual)} "
        f"unexpected={sorted(actual - expected)}",
    )
    fps = np.asarray(arrays["fps"])
    schema = np.asarray(arrays["kinematics_schema_version"])
    _require(fps.dtype == np.int64 and fps.shape == (1,) and int(fps[0]) == 50,
             f"{label}.fps must be exact int64[1] value 50")
    _require(schema.dtype == np.int64 and schema.shape == (1,) and int(schema[0]) == 2,
             f"{label} must use kinematics schema 2")
    _require(_scalar_unicode(arrays["body_pos_point"], f"{label}.body_pos_point") == "link_origin",
             f"{label}.body_pos_point changed")
    _require(
        _scalar_unicode(arrays["body_lin_vel_point"], f"{label}.body_lin_vel_point")
        == "center_of_mass",
        f"{label}.body_lin_vel_point changed",
    )
    names = np.asarray(arrays["body_names"])
    _require(names.ndim == 1 and names.dtype.kind == "U" and not names.dtype.hasobject,
             f"{label}.body_names must be a one-dimensional Unicode array")
    decoded_names = tuple(str(value) for value in names.tolist())
    _require(decoded_names == tuple(body_order), f"{label}.body_names disagrees with body-order")
    _require(len(decoded_names) == len(set(decoded_names)) and all(decoded_names),
             f"{label}.body_names must be unique and non-empty")

    joint_pos = np.asarray(arrays["joint_pos"])
    joint_vel = np.asarray(arrays["joint_vel"])
    _require(joint_pos.dtype == np.float32 and joint_pos.ndim == 2,
             f"{label}.joint_pos must be float32 [T,J]")
    frames, joints = joint_pos.shape
    _require(frames > 26 and joints == 31, f"{label} must contain T>26 and exactly 31 joints")
    _require(joint_vel.dtype == np.float32 and joint_vel.shape == joint_pos.shape,
             f"{label}.joint_vel shape/dtype changed")
    bodies = len(decoded_names)
    expected_shapes = {
        "body_pos_w": (frames, bodies, 3),
        "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3),
        "body_ang_vel_w": (frames, bodies, 3),
    }
    for key, shape in expected_shapes.items():
        value = np.asarray(arrays[key])
        _require(value.dtype == np.float32 and value.shape == shape,
                 f"{label}.{key} shape/dtype changed")
    recomputed_velocity = np.gradient(
        joint_pos.astype(np.float64), 1.0 / 50.0, axis=0
    ).astype(np.float32)
    _require(np.array_equal(joint_vel, recomputed_velocity),
             f"{label}.joint_vel is not the canonical position gradient")
    quaternion_norm = np.linalg.norm(np.asarray(arrays["body_quat_w"], dtype=np.float64), axis=-1)
    _require(np.max(np.abs(quaternion_norm - 1.0)) <= 2.0e-5,
             f"{label}.body_quat_w is not normalized")

    if optional:
        source_sha = _scalar_unicode(
            arrays["kinematics_migration_source_sha256"],
            f"{label}.kinematics_migration_source_sha256",
        )
        _require(_is_sha256(source_sha), f"{label} migration source SHA is malformed")
        source_point = _scalar_unicode(
            arrays["kinematics_migration_source_point"],
            f"{label}.kinematics_migration_source_point",
        )
        _require(source_point in {"link_origin", "center_of_mass"},
                 f"{label} migration source point changed")
        migration_tool = _scalar_unicode(
            arrays["kinematics_migration_tool"], f"{label}.kinematics_migration_tool"
        )
        _require(migration_tool == "migrate_motion_kinematics.py/v2",
                 f"{label} migration tool changed")
    return frames


def _validate_queue(queue: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    queue = _exact_keys(
        queue,
        {
            "schema_version",
            "experiment_id",
            "created_utc",
            "human_owner",
            "executor",
            "purpose",
            "runtime",
            "assets",
            "fixed_contract",
            "observed_baseline_not_to_rerun",
            "staged_cells",
            "derived_cell_rule",
            "acceptance",
            "selection",
        },
        "queue",
    )
    _require(queue["schema_version"] == 1, "queue schema_version changed")
    _require(queue["experiment_id"] == EXPECTED_EXPERIMENT, "queue experiment changed")
    runtime = _exact_keys(
        queue["runtime"],
        {
            "checkout_path",
            "checkout_commit",
            "generator_source_commit",
            "generator_sha256",
            "topp_sha256",
            "mjcf_sha256",
            "urdf_sha256",
            "body_order_sha256",
        },
        "queue.runtime",
    )
    _require(Path(runtime["checkout_path"]).is_absolute(), "runtime checkout must be absolute")
    _require(_is_commit(runtime["checkout_commit"]), "runtime checkout commit is malformed")
    _require(
        _is_commit(runtime["generator_source_commit"]),
        "generator source commit is malformed",
    )
    for key in RUNTIME_RELATIVE_PATHS:
        _require(_is_sha256(runtime[key]), f"queue.runtime.{key} is malformed")
    assets = _exact_keys(queue["assets"], {"forehand", "backhand"}, "queue.assets")
    for name, asset in assets.items():
        asset = _exact_keys(asset, {"path", "sha256", "contact_frame"}, f"asset {name}")
        _require(Path(asset["path"]).is_absolute(), f"asset {name} path must be absolute")
        _require(_is_sha256(asset["sha256"]), f"asset {name} SHA is malformed")
        _require(type(asset["contact_frame"]) is int and asset["contact_frame"] > 5,
                 f"asset {name} contact frame is invalid")
    fixed = queue["fixed_contract"]
    required_fixed = {
        "fps": 50,
        "ready_frame": 0,
        "ready_velocity": "bitwise_zero",
        "hold_frames": 4,
        "output_contact_frame": 25,
        "protected_precontact_seconds": 0.1,
        "delta_plus_blend_intervals": 22,
        "topp_objective": "runup",
        "body_mode": "fk",
        "automatic_retry": False,
        "gpu_or_trainer_signals": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }
    _exact_keys(fixed, set(required_fixed), "queue.fixed_contract")
    _require(dict(fixed) == required_fixed, "queue fixed contract changed")
    _require(
        queue["derived_cell_rule"]
        == {
            "join_frame": "contact_frame - delta",
            "blend_intervals": "22 - delta",
            "output_contact_frame": 25,
        },
        "queue derived-cell rule changed",
    )
    staged = _exact_keys(
        queue["staged_cells"],
        {"stage1_endpoint_factorial", "stage2_midpoint_rule", "stage3_refinement_rule"},
        "queue.staged_cells",
    )
    cells = staged["stage1_endpoint_factorial"]
    _require(isinstance(cells, list) and len(cells) == 6, "stage1 must contain six cells")
    seen_ids: set[str] = set()
    seen_tuples: set[tuple[str, str, int]] = set()
    for cell in cells:
        cell = _exact_keys(cell, {"cell_id", "action", "ready_source", "delta"}, "cell")
        cell_id = cell["cell_id"]
        _require(isinstance(cell_id, str) and cell_id in EXPECTED_CELLS,
                 f"unexpected stage1 cell id {cell_id!r}")
        _require(cell_id not in seen_ids, f"duplicate cell id {cell_id}")
        seen_ids.add(cell_id)
        identity = (cell["action"], cell["ready_source"], cell["delta"])
        _require(identity == EXPECTED_CELLS[cell_id], f"cell {cell_id} tuple changed")
        _require(identity not in seen_tuples, f"duplicate cell tuple {identity}")
        seen_tuples.add(identity)
        contact = assets[cell["action"]]["contact_frame"]
        join = contact - cell["delta"]
        blend = 22 - cell["delta"]
        _require(join >= 0 and blend >= 5, f"cell {cell_id} derived frames are invalid")
        _require(4 + (blend - 1) + cell["delta"] == 25,
                 f"cell {cell_id} does not reconstruct output contact 25")
    _require(seen_ids == set(EXPECTED_CELLS), "stage1 cell set is incomplete")
    acceptance = queue["acceptance"]
    _require(
        acceptance
        == {
            "candidate_contract_required": True,
            "protected_window_bitwise_equal": True,
            "production_fk_required": True,
            "topp_within_budget": True,
            "kinematic_hard_limits_clean": True,
            "candidate_start_to_contact_s_max": 0.5,
            "strict_global_minimum_not_claimed": True,
        },
        "queue acceptance contract changed",
    )
    return queue, cells


def _validate_candidate_contract(
    context: _AuditContext,
    contract: Any,
    *,
    cell: Mapping[str, Any],
    queue: Mapping[str, Any],
    candidate_path: Path,
    candidate_snapshot: _Snapshot,
    runtime_generator: Path,
) -> Mapping[str, Any]:
    contract = _exact_keys(
        contract,
        {
            "schema_version", "artifact_kind", "status", "inputs", "tool", "request",
            "synthesis", "proof", "output", "authorization", "required_next_gates",
            "explicit_non_claims",
        },
        f"{cell['cell_id']} candidate contract",
    )
    _require(contract["schema_version"] == 1, "candidate contract schema changed")
    _require(contract["artifact_kind"] == "host_only_ready_to_strike_motion_candidate",
             "candidate artifact kind changed")
    _require(contract["status"] == "candidate_only_all_runtime_and_safety_gates_open",
             "candidate status changed")
    inputs = _exact_keys(
        contract["inputs"],
        {"source_schema2_npz", "shared_ready_schema2_npz", "shared_ready_frame"},
        "candidate inputs",
    )
    action_asset = queue["assets"][cell["action"]]
    ready_asset = queue["assets"][cell["ready_source"]]
    _validate_file_evidence(
        context, inputs["source_schema2_npz"], _absolute(action_asset["path"]),
        "candidate source evidence", generator_style=True,
    )
    _validate_file_evidence(
        context, inputs["shared_ready_schema2_npz"], _absolute(ready_asset["path"]),
        "candidate ready evidence", generator_style=True,
    )
    _require(inputs["shared_ready_frame"] == 0, "candidate ready frame changed")
    tool = _exact_keys(
        contract["tool"],
        {"path", "bytes", "sha256", "device", "inode", "mtime_ns", "ctime_ns",
         "binding_semantics"},
        "candidate generator evidence",
    )
    _require(
        tool["binding_semantics"]
        == "source_file_snapshot_at_main_entry_unchanged_before_publish",
        "candidate generator binding semantics changed",
    )
    evidence_without_semantics = dict(tool)
    evidence_without_semantics.pop("binding_semantics")
    _validate_file_evidence(
        context, evidence_without_semantics, runtime_generator,
        "candidate generator evidence", generator_style=True,
    )
    delta = cell["delta"]
    contact = action_asset["contact_frame"]
    join = contact - delta
    blend = 22 - delta
    _require(
        contract["request"]
        == {
            "source_contact_frame": contact,
            "source_join_frame": join,
            "ready_hold_frames": 4,
            "quintic_blend_intervals": blend,
            "protected_precontact_seconds": 0.1,
        },
        f"{cell['cell_id']} generator request changed",
    )
    proof = contract["proof"]
    _require(isinstance(proof, Mapping), "candidate proof must be an object")
    required_proof = {
        "fps": 50,
        "source_join_frame": join,
        "source_contact_frame": contact,
        "output_contact_frame": 25,
        "protected_frames_before_contact": 5,
        "protected_window_bitwise_equal": True,
        "pose_and_body_velocity_source_suffix_bitwise_equal": True,
        "frame0_shared_ready_pose_bitwise_equal": True,
        "ready_source_velocity_channels_ignored": True,
        "ready_velocity_definition": "explicit_bitwise_zero",
        "initial_zero_velocity_frames": 3,
        "joint_position_continuous_quintic_endpoint_c2": True,
        "finite": True,
        "contact_time_from_frame0_s": 0.5,
    }
    for key, expected in required_proof.items():
        _require(proof.get(key) == expected, f"candidate proof field {key} changed")
    _require(_is_sha256(proof.get("protected_window_sha256")),
             "candidate protected-window SHA is malformed")
    for key in ("quaternion_max_norm_error", "producer_gradient_join_velocity_error_rad_s"):
        _finite_number(proof.get(key), f"candidate proof {key}", minimum=0.0)
    output = _exact_keys(contract["output"], {"npz", "contract_binding"}, "candidate output")
    _require(
        output["contract_binding"]
        == "JSON binds exact NPZ SHA-256; publication is no-clobber",
        "candidate output binding changed",
    )
    output_evidence = output["npz"]
    _validate_file_evidence(context, output_evidence, candidate_path, "candidate NPZ evidence")
    _require(output_evidence["sha256"] == _sha256(candidate_snapshot.payload),
             "candidate contract does not bind candidate bytes")
    authorization = contract["authorization"]
    _require(
        authorization
        == {
            "host_candidate_materialized": True,
            "topp_runup_0p5_pass": False,
            "l0_static_pass": False,
            "vendor_l1_pass": False,
            "self_hit_pass": False,
            "table_net_clearance_5mm_pass": False,
            "dynamics_pass": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "candidate authorization changed",
    )
    return contract


def _validate_topp_certificate(
    context: _AuditContext,
    certificate: Any,
    *,
    cell: Mapping[str, Any],
    queue: Mapping[str, Any],
    candidate_path: Path,
    candidate_arrays: Mapping[str, np.ndarray],
    candidate_frames: int,
    phase: float,
    output_path: Path,
    certificate_path: Path,
    markdown_path: Path,
    runtime_paths: Mapping[str, Path],
    body_order: Sequence[str],
) -> dict[str, Any]:
    certificate = _exact_keys(certificate, TOPP_CERTIFICATE_KEYS, "TOPP certificate")
    _require(certificate["tool"] == TOPP_TOOL, "TOPP tool/schema identity changed")
    _require(certificate["algorithm_scope"] == TOPP_ALGORITHM_SCOPE,
             "TOPP algorithm-scope honesty changed")
    _require(certificate["search_objective"] == "runup", "TOPP objective is not runup")
    _require(isinstance(certificate["generated_utc"], str) and certificate["generated_utc"],
             "TOPP generated time is missing")
    _require(isinstance(certificate["verdict"], str) and certificate["verdict"],
             "TOPP verdict is missing")
    _require(certificate["direction"] in {"accelerated", "slowed", "unchanged"},
             "TOPP direction changed")
    _finite_number(certificate["chosen_scale"], "TOPP chosen scale", minimum=0.0)
    _require(isinstance(certificate["feasible_reason"], str) and certificate["feasible_reason"],
             "TOPP feasible reason is missing")
    files = _exact_keys(
        certificate.get("files"),
        {"input", "output", "report_path", "markdown_path"},
        "TOPP files",
    )
    _validate_file_evidence(context, files["input"], candidate_path, "TOPP input evidence")
    output_snapshot = _validate_file_evidence(
        context, files["output"], output_path, "TOPP output evidence"
    )
    _require(files["report_path"] == str(certificate_path), "TOPP report path is misbound")
    _require(files["markdown_path"] == str(markdown_path), "TOPP markdown path is misbound")
    context.read(markdown_path, "TOPP markdown certificate")
    output_arrays = _load_npz(output_snapshot, "TOPP output NPZ")
    output_frames_actual = _validate_schema2_arrays(
        output_arrays,
        body_order=body_order,
        label="TOPP output NPZ",
        allow_migration_provenance=False,
    )

    clips = certificate.get("budget_provenance")
    clips = _exact_keys(clips, {"clips", "scale", "envelope"}, "TOPP budget provenance")
    _require(_finite_number(clips["scale"], "TOPP budget scale", minimum=0.0) == 1.0,
             "TOPP budget scale changed")
    _require(isinstance(clips["clips"], list) and len(clips["clips"]) == 2,
             "TOPP must bind exactly two budget clips")
    for index, name in enumerate(("forehand", "backhand")):
        asset = queue["assets"][name]
        _validate_file_evidence(
            context, clips["clips"][index], _absolute(asset["path"]),
            f"TOPP budget clip {index}",
        )
    _require(isinstance(clips["envelope"], list) and clips["envelope"],
             "TOPP budget envelope must be non-empty")
    for index, value in enumerate(clips["envelope"]):
        _finite_number(value, f"TOPP budget envelope {index}", minimum=0.0)

    provenance = _exact_keys(
        certificate.get("runtime_provenance"),
        {"mjcf", "urdf", "body_order", "tool"},
        "TOPP runtime provenance",
    )
    for key in ("mjcf", "urdf", "body_order"):
        _validate_file_evidence(
            context, provenance[key], runtime_paths[f"{key}_sha256"], f"TOPP {key} evidence"
        )
    tool = _exact_keys(
        provenance["tool"], {"topp_mintime", "dependencies"}, "TOPP tool provenance"
    )
    _validate_file_evidence(
        context, tool["topp_mintime"], runtime_paths["topp_sha256"], "TOPP tool evidence"
    )
    dependencies = _exact_keys(
        tool["dependencies"], set(TOPP_DEPENDENCIES), "TOPP dependencies"
    )
    runtime_root = _absolute(queue["runtime"]["checkout_path"])
    for name, relative in TOPP_DEPENDENCIES.items():
        _validate_file_evidence(
            context, dependencies[name], runtime_root / relative, f"TOPP dependency {name}"
        )

    source = _exact_keys(
        certificate["source"],
        {"frames", "fps", "contact_frame", "phase", "runup_s", "duration_s",
         "clean_blade_speed_mps", "mean_abs_acc"},
        "TOPP source",
    )
    _require(source.get("frames") == candidate_frames, "TOPP source frame count is misbound")
    _require(source.get("fps") == 50, "TOPP source fps changed")
    _require(source.get("contact_frame") == 25, "TOPP source contact frame changed")
    _require(math.isclose(_finite_number(source.get("phase"), "TOPP source phase"), phase,
                          rel_tol=0.0, abs_tol=1e-12), "TOPP source phase is misbound")
    _require(_finite_number(source.get("runup_s"), "TOPP source runup") == 0.5,
             "TOPP source runup changed")
    source_duration = _finite_number(source["duration_s"], "TOPP source duration", minimum=0.0)
    _require(math.isclose(source_duration, (candidate_frames - 1) / 50.0,
                          rel_tol=0.0, abs_tol=5.1e-5),
             "TOPP source duration is misbound")
    _finite_number(source["clean_blade_speed_mps"], "TOPP source blade speed", minimum=0.0)
    _finite_number(source["mean_abs_acc"], "TOPP source mean acceleration", minimum=0.0)

    output = _exact_keys(
        certificate["output"],
        {"frames", "fps", "contact_frame", "phase_out", "runup_s", "duration_s",
         "runup_change_x", "duration_change_x", "wait_s", "body_mode", "mean_abs_acc"},
        "TOPP output",
    )
    _require(output.get("frames") == output_frames_actual, "TOPP output frames are misbound")
    _require(output.get("fps") == 50.0, "TOPP output fps changed")
    output_contact = output.get("contact_frame")
    _require(type(output_contact) is int and 0 <= output_contact < output_frames_actual,
             "TOPP output contact frame is invalid")
    expected_phase_out = output_contact / float(output_frames_actual - 1)
    _require(math.isclose(_finite_number(output.get("phase_out"), "TOPP output phase"),
                          expected_phase_out, rel_tol=0.0, abs_tol=5.1e-7),
             "TOPP output phase is misbound")
    _require(output.get("body_mode") == "fk", "TOPP output is not production FK")
    _require(
        np.array_equal(
            np.asarray(output_arrays["joint_pos"])[output_contact],
            np.asarray(candidate_arrays["joint_pos"])[25],
        ),
        "TOPP output contact row differs from candidate contact row",
    )
    output_runup = _finite_number(output.get("runup_s"), "TOPP output runup", minimum=0.0)
    output_duration = _finite_number(output["duration_s"], "TOPP output duration", minimum=0.0)
    _require(math.isclose(output_duration, (output_frames_actual - 1) / 50.0,
                          rel_tol=0.0, abs_tol=5.1e-5),
             "TOPP output duration is misbound")
    _finite_number(output["runup_change_x"], "TOPP output runup ratio", minimum=0.0)
    _finite_number(output["duration_change_x"], "TOPP output duration ratio", minimum=0.0)
    _finite_number(output["wait_s"], "TOPP output wait", minimum=0.0)
    _finite_number(output["mean_abs_acc"], "TOPP output mean acceleration", minimum=0.0)

    acceptance = _exact_keys(
        certificate.get("acceptance"),
        {"cop_dose_final", "fric_dose_final", "tau_dose_final", "within_budget",
         "kin_out_window_clean", "kin_lock_window_clean", "kinematic_hard_limits_clean"},
        "TOPP acceptance",
    )
    doses = {
        "cop": _finite_number(acceptance["cop_dose_final"], "TOPP CoP dose", minimum=0.0),
        "friction": _finite_number(
            acceptance["fric_dose_final"], "TOPP friction dose", minimum=0.0
        ),
        "torque": _finite_number(
            acceptance["tau_dose_final"], "TOPP torque dose", minimum=0.0
        ),
    }
    for key in ("within_budget", "kin_out_window_clean", "kin_lock_window_clean",
                "kinematic_hard_limits_clean"):
        _require(type(acceptance[key]) is bool, f"TOPP acceptance {key} must be boolean")
        _require(acceptance[key] is True, f"TOPP acceptance {key} is false")
    budget = _exact_keys(
        certificate["budget"],
        {"cop_gate", "fric_gate", "tau_gate", "vel_limit_frac", "kin_vel_target",
         "kin_acc_target", "note"},
        "TOPP budget",
    )
    gates = {
        "cop": _finite_number(budget.get("cop_gate"), "TOPP CoP gate", minimum=0.0),
        "friction": _finite_number(budget.get("fric_gate"), "TOPP friction gate", minimum=0.0),
        "torque": _finite_number(budget.get("tau_gate"), "TOPP torque gate", minimum=0.0),
    }
    for name in doses:
        _require(doses[name] <= gates[name] + 5e-5,
                 f"TOPP {name} dose exceeds its recorded gate")

    fidelity = _exact_keys(
        certificate.get("fidelity"),
        {"contact_row_bitwise", "blade_speed_clean_out_mps", "blade_speed_dev_frac",
         "face_normal_diff_deg", "first_frame_max_joint_vel"},
        "TOPP fidelity",
    )
    _require(fidelity["contact_row_bitwise"] is True, "TOPP contact row is not bitwise")
    _finite_number(fidelity["blade_speed_clean_out_mps"], "TOPP blade speed", minimum=0.0)
    speed_dev = _finite_number(
        fidelity["blade_speed_dev_frac"], "TOPP blade speed deviation", minimum=0.0
    )
    _require(speed_dev <= 0.02, "TOPP blade speed deviation exceeds 2 percent")
    _finite_number(fidelity["face_normal_diff_deg"], "TOPP face difference", minimum=0.0)
    actual_first_velocity = float(np.max(np.abs(np.asarray(output_arrays["joint_vel"])[0])))
    recorded_first_velocity = _finite_number(
        fidelity["first_frame_max_joint_vel"], "TOPP first velocity", minimum=0.0
    )
    _require(actual_first_velocity == 0.0 and recorded_first_velocity == actual_first_velocity,
             "TOPP first-frame velocity is not exactly zero or is misreported")

    timing = _exact_keys(
        certificate.get("timing_bound"),
        {"candidate_start_to_contact_s", "bound_semantics", "strict_global_minimum_proven"},
        "TOPP timing bound",
    )
    timing_s = _finite_number(
        timing["candidate_start_to_contact_s"], "TOPP timing bound", minimum=0.0
    )
    _require(timing_s > 0.0, "TOPP timing bound must be positive")
    _require(timing["bound_semantics"] == "feasible upper bound within this searched family",
             "TOPP timing-bound semantics changed")
    _require(timing["strict_global_minimum_proven"] is False,
             "TOPP must not claim a strict global minimum")
    _require(math.isclose(timing_s, output_runup, rel_tol=0.0, abs_tol=1e-12),
             "TOPP timing bound disagrees with output runup")
    outer_trace = certificate["outer_trace"]
    _require(isinstance(outer_trace, list) and outer_trace, "TOPP outer trace is empty")
    feasible_rows: list[tuple[Mapping[str, Any], float]] = []
    for index, row in enumerate(outer_trace):
        row = _exact_keys(
            row,
            {"gamma", "feasible", "reason", "iters", "T_out", "duration_s", "runup_s",
             "cop", "fric", "tau"},
            f"TOPP outer trace row {index}",
        )
        _finite_number(row["gamma"], f"TOPP outer trace {index} gamma", minimum=0.0)
        _require(type(row["feasible"]) is bool, f"TOPP outer trace {index} feasible is not bool")
        _require(isinstance(row["reason"], str), f"TOPP outer trace {index} reason is invalid")
        _require(type(row["iters"]) is int and row["iters"] >= 0,
                 f"TOPP outer trace {index} iterations are invalid")
        _require(type(row["T_out"]) is int and row["T_out"] > 1,
                 f"TOPP outer trace {index} output frames are invalid")
        row_runup = _finite_number(row["runup_s"], f"TOPP outer trace {index} runup", minimum=0.0)
        for key in ("duration_s", "cop", "fric", "tau"):
            _finite_number(row[key], f"TOPP outer trace {index} {key}", minimum=0.0)
        if row["feasible"]:
            feasible_rows.append((row, row_runup))
    _require(feasible_rows, "TOPP outer trace contains no feasible point")
    minimum_runup = min(value for _, value in feasible_rows)
    _require(math.isclose(minimum_runup, timing_s, rel_tol=0.0, abs_tol=5.1e-5),
             "TOPP chosen runup is not the best recorded feasible point")
    selected_row = next(
        row for row, value in feasible_rows
        if math.isclose(value, minimum_runup, rel_tol=0.0, abs_tol=5.1e-5)
    )
    selected_bindings = {
        "gamma": (_finite_number(selected_row["gamma"], "TOPP selected gamma"),
                  _finite_number(certificate["chosen_scale"], "TOPP chosen scale")),
        "duration_s": (_finite_number(selected_row["duration_s"], "TOPP selected duration"),
                       output_duration),
        "runup_s": (_finite_number(selected_row["runup_s"], "TOPP selected runup"), timing_s),
        "cop": (_finite_number(selected_row["cop"], "TOPP selected CoP"), doses["cop"]),
        "fric": (_finite_number(selected_row["fric"], "TOPP selected friction"), doses["friction"]),
        "tau": (_finite_number(selected_row["tau"], "TOPP selected torque"), doses["torque"]),
    }
    for name, (recorded, chosen) in selected_bindings.items():
        _require(math.isclose(recorded, chosen, rel_tol=0.0, abs_tol=5.1e-5),
                 f"TOPP selected trace row {name} is not bound to the published answer")
    _require(selected_row["T_out"] == output_frames_actual,
             "TOPP selected trace output frames are not bound to the published answer")
    _require(selected_row["reason"] == certificate["feasible_reason"],
             "TOPP selected trace reason is not bound to the published answer")
    _require(isinstance(certificate["inner_trace_best"], list) and certificate["inner_trace_best"],
             "TOPP inner trace is empty")
    return {
        "output_sha256": _sha256(output_snapshot.payload),
        "certificate_sha256": _sha256(context.read(certificate_path, "TOPP certificate").payload),
        "candidate_start_to_contact_s": timing_s,
        "within_0p5_s": timing_s <= 0.5,
        "doses": doses,
    }


def _publish_exclusive(path: Path, payload: bytes) -> None:
    _ensure_no_symlink_components(path.parent, "receipt parent")
    _ensure_no_symlink_components(path, "receipt", leaf_may_be_missing=True)
    _require(not path.exists() and not path.is_symlink(), f"receipt already exists: {path}")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags, 0o444)
    except OSError as exc:
        raise AttestationError(f"cannot publish no-clobber receipt {path}: {exc}") from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def attest_stage1(
    *,
    root: Path | str,
    queue_path: Path | str,
    receipt_path: Path | str | None = None,
    execute: bool = False,
    confirm: str | None = None,
    attestor_source: Path | str | None = None,
    launch_argv: Sequence[str] | None = None,
    before_publish_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    root = _absolute(root)
    queue_path = _absolute(queue_path)
    receipt = _absolute(receipt_path or (root / "stage1_historical_attestation.json"))
    source_path = _absolute(attestor_source or Path(__file__))
    _ensure_no_symlink_components(root, "stage1 root")
    _require(root.is_dir(), f"stage1 root must be a directory: {root}")
    _require(receipt.parent == root, "receipt must be published directly inside stage1 root")
    _ensure_no_symlink_components(receipt, "receipt", leaf_may_be_missing=True)
    _require(not receipt.exists() and not receipt.is_symlink(), f"receipt already exists: {receipt}")
    if execute:
        _require(confirm == CONFIRM_TOKEN, f"--execute requires --confirm {CONFIRM_TOKEN}")
    else:
        _require(confirm is None, "--confirm is only valid with --execute")

    context = _AuditContext()
    source_snapshot = context.read(source_path, "attestor source")
    queue_snapshot = context.read(queue_path, "registered queue")
    queue, cells = _validate_queue(_load_json(queue_snapshot.payload, "registered queue"))
    queue_copy = context.read(root / "queue.yaml", "stage1 queue copy")
    _require(queue_copy.payload == queue_snapshot.payload, "stage1 queue copy is not exact")
    queue_sha = _sha256(queue_snapshot.payload)

    runtime_root = _absolute(queue["runtime"]["checkout_path"])
    _ensure_no_symlink_components(runtime_root, "runtime checkout")
    _require(runtime_root.is_dir(), "runtime checkout is not a directory")
    runtime_paths = {
        key: runtime_root / relative
        for key, relative in RUNTIME_RELATIVE_PATHS.items()
        if key != "generator_sha256"
    }
    runtime_snapshots: dict[str, _Snapshot] = {}
    for key, path in runtime_paths.items():
        snapshot = context.read(path, f"runtime {key}")
        _require(_sha256(snapshot.payload) == queue["runtime"][key],
                 f"runtime {key} SHA changed")
        runtime_snapshots[key] = snapshot
    generator_copy = context.read(root / "build_ready_to_strike_motion.py", "stage1 generator copy")
    _require(
        _sha256(generator_copy.payload) == queue["runtime"]["generator_sha256"],
        "stage1 executed generator copy does not match its registered SHA",
    )
    try:
        body_order = tuple(
            line.strip()
            for line in runtime_snapshots["body_order_sha256"].payload.decode("utf-8").splitlines()
            if line.strip()
        )
    except UnicodeDecodeError as exc:
        raise AttestationError("body-order is not UTF-8") from exc
    _require(body_order and len(body_order) == len(set(body_order)) and all(body_order),
             "body-order must contain unique non-empty names")
    asset_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, asset in queue["assets"].items():
        snapshot = context.read(asset["path"], f"{name} source asset")
        _require(_sha256(snapshot.payload) == asset["sha256"], f"{name} source asset SHA changed")
        arrays = _load_npz(snapshot, f"{name} source asset")
        _validate_schema2_arrays(
            arrays,
            body_order=body_order,
            label=f"{name} source asset",
            allow_migration_provenance=True,
        )
        asset_arrays[name] = arrays

    summary_snapshot = context.read(root / "stage1_summary.json", "stage1 summary")
    summary = _exact_keys(
        _load_json(summary_snapshot.payload, "stage1 summary"),
        {"schema_version", "status", "queue_sha256", "generator_sha256",
         "main_prereg_commit", "runtime_source_commit", "rows",
         "trainer_or_robot_signals", "automatic_retry"},
        "stage1 summary",
    )
    _require(summary["schema_version"] == 1, "summary schema changed")
    _require(summary["status"] == "stage1_complete_no_retry", "summary status changed")
    _require(summary["queue_sha256"] == queue_sha, "summary queue SHA is misbound")
    _require(summary["generator_sha256"] == queue["runtime"]["generator_sha256"],
             "summary generator SHA is misbound")
    _require(summary["main_prereg_commit"] == EXPECTED_PREREG_COMMIT,
             "summary prereg commit changed")
    _require(summary["runtime_source_commit"] == queue["runtime"]["checkout_commit"],
             "summary runtime commit is misbound")
    _require(summary["trainer_or_robot_signals"] == [], "summary records trainer/robot signals")
    _require(summary["automatic_retry"] is False, "summary records automatic retry")
    _require(isinstance(summary["rows"], list) and len(summary["rows"]) == 6,
             "summary must contain six rows")

    receipt_cells: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    for expected_cell, row in zip(cells, summary["rows"]):
        row = _exact_keys(
            row,
            {"cell_id", "action", "ready_source", "delta", "join_frame", "blend_intervals",
             "generator_rc", "candidate", "candidate_sha256", "contract_sha256", "frames",
             "phase", "joint_path_l2", "joint_curvature_l2", "max_joint_step_rad", "topp_rc",
             "topp_certificate_sha256", "topp_acceptance", "topp_timing_bound", "topp_fidelity"},
            "stage1 summary row",
        )
        cell_id = expected_cell["cell_id"]
        _require(row["cell_id"] == cell_id, "summary row order/cell changed")
        _require(cell_id not in seen_rows, f"duplicate summary row {cell_id}")
        seen_rows.add(cell_id)
        for key in ("action", "ready_source", "delta"):
            _require(row[key] == expected_cell[key], f"summary {cell_id}.{key} changed")
        contact = queue["assets"][row["action"]]["contact_frame"]
        join = contact - row["delta"]
        blend = 22 - row["delta"]
        _require(row["join_frame"] == join and row["blend_intervals"] == blend,
                 f"summary {cell_id} derived join/blend changed")
        _require(row["generator_rc"] == 0 and row["topp_rc"] == 0,
                 f"{cell_id} did not complete both historical stages")
        cell_root = root / cell_id
        _ensure_no_symlink_components(cell_root, f"{cell_id} directory")
        _require(cell_root.is_dir(), f"{cell_id} directory is missing")
        candidate_path = cell_root / "candidate.npz"
        contract_path = cell_root / "candidate.contract.json"
        expected_candidate_path = str(candidate_path)
        _require(row["candidate"] == expected_candidate_path,
                 f"summary {cell_id} candidate path is misbound")
        candidate_snapshot = context.read(candidate_path, f"{cell_id} candidate")
        contract_snapshot = context.read(contract_path, f"{cell_id} generator contract")
        _require(row["candidate_sha256"] == _sha256(candidate_snapshot.payload),
                 f"summary {cell_id} candidate SHA is misbound")
        _require(row["contract_sha256"] == _sha256(contract_snapshot.payload),
                 f"summary {cell_id} contract SHA is misbound")
        candidate_arrays = _load_npz(candidate_snapshot, f"{cell_id} candidate")
        _validate_schema2_arrays(
            candidate_arrays,
            body_order=body_order,
            label=f"{cell_id} candidate",
            allow_migration_provenance=True,
        )
        ready_arrays = asset_arrays[row["ready_source"]]
        action_arrays = asset_arrays[row["action"]]
        for key in ("joint_pos", "body_pos_w", "body_quat_w"):
            _require(
                np.array_equal(np.asarray(candidate_arrays[key])[0], np.asarray(ready_arrays[key])[0]),
                f"{cell_id} candidate frame0 {key} differs from selected ready frame0",
            )
        protected_start = contact - 5
        _require(protected_start >= 0, f"{cell_id} protected source window is invalid")
        for key in SCHEMA2_TIME_KEYS:
            _require(
                np.array_equal(
                    np.asarray(candidate_arrays[key])[20:26],
                    np.asarray(action_arrays[key])[protected_start:contact + 1],
                ),
                f"{cell_id} candidate protected {key} window differs from source strike window",
            )
        joint_pos = np.asarray(candidate_arrays["joint_pos"], dtype=np.float64)
        _require(joint_pos.ndim == 2 and joint_pos.shape[0] > 26,
                 f"{cell_id} candidate joint_pos shape is invalid")
        frames = int(joint_pos.shape[0])
        phase = 25.0 / float(frames - 1)
        _require(row["frames"] == frames, f"summary {cell_id} frame count is misbound")
        _require(math.isclose(_finite_number(row["phase"], f"{cell_id} phase"), phase,
                              rel_tol=0.0, abs_tol=1e-15),
                 f"summary {cell_id} phase is misbound")
        segment = np.diff(joint_pos[:26], axis=0)
        second = np.diff(joint_pos[:26], n=2, axis=0)
        recomputed = {
            "joint_path_l2": float(np.linalg.norm(segment, axis=1).sum()),
            "joint_curvature_l2": float(np.linalg.norm(second, axis=1).sum()),
            "max_joint_step_rad": float(np.abs(segment).max()),
        }
        for key, expected in recomputed.items():
            _require(math.isclose(_finite_number(row[key], f"{cell_id} {key}"), expected,
                                  rel_tol=0.0, abs_tol=1e-12),
                     f"summary {cell_id} {key} is misbound")
        for velocity_key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            velocity = candidate_arrays[velocity_key]
            _require(np.array_equal(velocity[:3], np.zeros_like(velocity[:3])),
                     f"{cell_id} candidate initial velocity is not bitwise zero")
        _validate_candidate_contract(
            context,
            _load_json(contract_snapshot.payload, f"{cell_id} generator contract"),
            cell=expected_cell,
            queue=queue,
            candidate_path=candidate_path,
            candidate_snapshot=candidate_snapshot,
            runtime_generator=generator_copy.path,
        )
        topp_root = cell_root / "topp"
        certificate_path = topp_root / "certificate.json"
        certificate_snapshot = context.read(certificate_path, f"{cell_id} TOPP certificate")
        _require(row["topp_certificate_sha256"] == _sha256(certificate_snapshot.payload),
                 f"summary {cell_id} TOPP certificate SHA is misbound")
        certificate = _load_json(certificate_snapshot.payload, f"{cell_id} TOPP certificate")
        _require(row["topp_acceptance"] == certificate.get("acceptance"),
                 f"summary {cell_id} TOPP acceptance differs from certificate")
        _require(row["topp_timing_bound"] == certificate.get("timing_bound"),
                 f"summary {cell_id} TOPP timing differs from certificate")
        _require(row["topp_fidelity"] == certificate.get("fidelity"),
                 f"summary {cell_id} TOPP fidelity differs from certificate")
        topp_result = _validate_topp_certificate(
            context,
            certificate,
            cell=expected_cell,
            queue=queue,
            candidate_path=candidate_path,
            candidate_arrays=candidate_arrays,
            candidate_frames=frames,
            phase=phase,
            output_path=topp_root / "motion.npz",
            certificate_path=certificate_path,
            markdown_path=topp_root / "certificate.md",
            runtime_paths=runtime_paths,
            body_order=body_order,
        )
        receipt_cells.append(
            {
                "cell_id": cell_id,
                "action": row["action"],
                "ready_source": row["ready_source"],
                "delta": row["delta"],
                "join_frame": join,
                "blend_intervals": blend,
                "candidate_sha256": _sha256(candidate_snapshot.payload),
                "generator_contract_sha256": _sha256(contract_snapshot.payload),
                **topp_result,
            }
        )
    _require(seen_rows == set(EXPECTED_CELLS), "summary cell set is incomplete")

    launch_snapshot = {
        "argv": list(launch_argv or ()),
        "root": str(root),
        "queue": str(queue_path),
        "receipt": str(receipt),
        "mode": "execute" if execute else "dry_run",
        "attestor_source": {
            "path": str(source_path),
            "bytes": len(source_snapshot.payload),
            "sha256": _sha256(source_snapshot.payload),
            "device": source_snapshot.evidence["device"],
            "inode": source_snapshot.evidence["inode"],
            "mtime_ns": source_snapshot.evidence["mtime_ns"],
            "ctime_ns": source_snapshot.evidence["ctime_ns"],
        },
    }
    receipt_document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ready_to_strike_stage1_historical_attestation",
        "status": "historical_evidence_attested_no_runtime_authority",
        "experiment_id": EXPECTED_EXPERIMENT,
        "inputs": {
            "root": str(root),
            "queue_path": str(queue_path),
            "queue_sha256": queue_sha,
            "generator_sha256": _sha256(generator_copy.payload),
            "summary_sha256": _sha256(summary_snapshot.payload),
            "runtime_source_commit": queue["runtime"]["checkout_commit"],
        },
        "cells": receipt_cells,
        "attestor": {
            "source_sha256": _sha256(source_snapshot.payload),
            "launch_snapshot": launch_snapshot,
            "launch_snapshot_sha256": _sha256(_canonical_json(launch_snapshot)),
            "source_and_inputs_unchanged_before_publish": True,
        },
        "formal_claims": {
            "physics_replay_exact": False,
            "source_closure_exact": False,
            "mjcf_closure_exact": False,
            "screening_activation_evidence_only": True,
        },
        "runtime_authority": {
            "read_only_historical": True,
            "ssh": False,
            "process_signal": False,
            "automatic_retry": False,
            "simulator": False,
            "trainer": False,
            "deployment": False,
            "robot_command": False,
        },
    }
    if before_publish_hook is not None:
        before_publish_hook()
    context.verify_all_unchanged()
    if execute:
        _publish_exclusive(receipt, _canonical_json(receipt_document))
    return receipt_document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="Existing absolute stage-1 root")
    parser.add_argument("--queue", type=Path, required=True, help="Registered queue JSON/YAML path")
    parser.add_argument("--receipt", type=Path, help="O_EXCL receipt path (default: ROOT/stage1_historical_attestation.json)")
    parser.add_argument("--execute", action="store_true", help="Publish one O_EXCL receipt")
    parser.add_argument("--confirm", help=f"Required with --execute: {CONFIRM_TOKEN}")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    launch_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        result = attest_stage1(
            root=args.root,
            queue_path=args.queue,
            receipt_path=args.receipt,
            execute=args.execute,
            confirm=args.confirm,
            attestor_source=Path(__file__),
            launch_argv=launch_argv,
        )
    except (AttestationError, OSError) as exc:
        print(f"ready-to-strike stage1 attestation error: {exc}", file=sys.stderr)
        return 2
    print(_canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
