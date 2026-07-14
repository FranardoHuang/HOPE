#!/usr/bin/env python3
"""Exact CPU-only L0 audit for one runtime-order schema-2 A3 motion.

The audit is intentionally narrower than a simulator or safety gate.  It first
replays the already-published schema-2 runner's formal-result validator, then
checks the exact NPZ against the exact vendor MJCF with kinematic
``mj_forward`` calls only (no ``mj_step``):

* exact schema, body/joint order, shapes, dtypes, finite values and 50 Hz time;
* body quaternion normalization and producer-exact velocity reconstruction;
* vendor-MJCF joint ranges in the bound runtime joint order;
* producer-exact link-pose FK reconstruction from the stored pelvis pose and
  joint positions;
* discrete-frame collision-ground clearance using the already frozen grounding
  tolerances and compiled collision contract.

It does not check self-collision, racket/body contact, table/net clearance,
dynamics, balance, strike feasibility, training, Gate3 or hardware.  Passing
only publishes one no-clobber JSON certificate for the exact input bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
HELPER_ROOT = REPO_ROOT / "hope_training/whole_body_tracking/scripts"
for _entry in (str(SCRIPT_ROOT), str(HELPER_ROOT)):
    while _entry in sys.path:
        sys.path.remove(_entry)
    sys.path.insert(0, _entry)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_ID = "franco_backhand_loop_b"
PLAN_ID = "motion-franco-backhand-loop-b-l0-static-20260714-v1"
PLAN_STATUS = "preregistered_source_gate_pass_runtime_audit_not_run"
CERTIFICATE_STATUS = "complete_exact_cpu_l0_static_pass_downstream_blocked"
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


class L0ContractError(ValueError):
    """The L0 source, lineage, model, motion or publication contract failed."""


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise L0ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def json_loads_exact(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                L0ContractError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise L0ContractError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise L0ContractError(f"{label} must be a JSON object")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json_loads_exact(path.read_bytes(), label)
    except OSError as exc:
        raise L0ContractError(f"cannot read {label} {path}: {exc}") from exc


def json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise L0ContractError(f"certificate is not finite canonical JSON: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise L0ContractError(f"{label} must be one lowercase SHA-256")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise L0ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise L0ContractError(
            f"{label} keys changed: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def ensure_no_symlink_components(path: Path, label: str) -> None:
    probe = Path(os.path.abspath(path))
    while True:
        try:
            info = probe.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise L0ContractError(f"cannot inspect {label} component {probe}: {exc}") from exc
        else:
            if stat.S_ISLNK(info.st_mode):
                raise L0ContractError(f"{label} contains symlink component {probe}")
        if probe == probe.parent:
            break
        probe = probe.parent


def ensure_regular_no_symlink(path: Path, label: str) -> None:
    ensure_no_symlink_components(path, label)
    try:
        info = path.stat()
    except OSError as exc:
        raise L0ContractError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise L0ContractError(f"{label} is not a regular file: {path}")


def binding(path: Path) -> dict[str, Any]:
    ensure_regular_no_symlink(path, "bound file")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_binding(
    value: Any,
    label: str,
    *,
    repo_root: Path | None = None,
    expected_path: str | None = None,
) -> Path:
    row = exact_keys(value, {"path", "bytes", "sha256"}, label)
    raw = row["path"]
    if not isinstance(raw, str) or not raw:
        raise L0ContractError(f"{label}.path must be non-empty text")
    if repo_root is None:
        path = Path(raw)
        if not path.is_absolute():
            raise L0ContractError(f"{label}.path must be absolute")
    else:
        if Path(raw).is_absolute():
            raise L0ContractError(f"{label}.path must be repository-relative")
        path = repo_root / raw
        try:
            path.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise L0ContractError(f"{label}.path escapes repository") from exc
    if expected_path is not None and raw != expected_path:
        raise L0ContractError(f"{label}.path changed: {raw!r}")
    ensure_regular_no_symlink(path, label)
    actual = binding(path)
    expected_bytes = row["bytes"]
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise L0ContractError(f"{label}.bytes must be a positive integer")
    if actual["bytes"] != expected_bytes or actual["sha256"] != require_sha(
        row["sha256"], f"{label}.sha256"
    ):
        raise L0ContractError(f"{label} content binding changed: actual={actual}")
    return path


def _read_names(path: Path, expected_count: int, label: str) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise L0ContractError(f"{label} is not {expected_count} unique names")
    return names


def _import_exact(module_name: str, expected_path: Path):
    module = importlib.import_module(module_name)
    actual = Path(module.__file__).resolve()
    if actual != expected_path.resolve():
        raise L0ContractError(
            f"imported {module_name} from {actual}, expected {expected_path.resolve()}"
        )
    return module


def validate_plan(plan_path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    ensure_regular_no_symlink(plan_path, "L0 preregistration")
    actual_sha = sha256_file(plan_path)
    if actual_sha != require_sha(expected_sha256, "expected preregistration SHA-256"):
        raise L0ContractError(
            f"L0 preregistration SHA mismatch: expected={expected_sha256} actual={actual_sha}"
        )
    plan = read_json(plan_path, "L0 preregistration")
    exact_keys(
        plan,
        {
            "schema_version",
            "plan_id",
            "status",
            "human_owner",
            "executor",
            "scope",
            "asset_id",
            "validator",
            "exact_runtime_inputs",
            "upstream_contracts",
            "a3_model",
            "runtime",
            "l0_contract",
            "output_contract",
            "authorization",
            "explicit_non_claims",
            "next_gate",
        },
        "L0 preregistration",
    )
    if (
        plan["schema_version"] != 1
        or plan["plan_id"] != PLAN_ID
        or plan["status"] != PLAN_STATUS
        or plan["human_owner"] != "Franco"
        or plan["executor"] != "Codex"
        or plan["asset_id"] != ASSET_ID
    ):
        raise L0ContractError("L0 preregistration identity/status/attribution changed")
    if plan["scope"] != (
        "CPU-only exact runtime-order schema-2 L0 static audit; kinematic mj_forward only, "
        "no simulator step, dynamics, training, deployment or hardware"
    ):
        raise L0ContractError("L0 scope changed or overclaims")

    verify_binding(
        plan["validator"],
        "validator",
        repo_root=REPO_ROOT,
        expected_path="scripts/audit_motion_schema2_l0_static.py",
    )
    inputs = exact_keys(
        plan["exact_runtime_inputs"],
        {"motion_npz", "materialization_report", "consume_claim", "consume_success"},
        "exact_runtime_inputs",
    )
    expected_inputs = {
        "motion_npz": (
            "/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/"
            "franco_backhand_loop_b_98e7b883b29d/"
            "franco_backhand_loop_b.98e7b883b29d.schema2_fk.npz",
            "e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc",
        ),
        "materialization_report": (
            "/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/"
            "franco_backhand_loop_b_98e7b883b29d/schema2_fk_report.json",
            "4f5245937956290b3f623acbb588d99b346e5a1d874e55ee9caf010f2d75bc38",
        ),
        "consume_claim": (
            "/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/"
            ".bc_schema2_fk_consume_control_v2/franco_backhand_loop_b.claim.json",
            "76e7ff88fea39c13b45096edaad504b2570b3ce079acc96366b820a9c1295fb0",
        ),
        "consume_success": (
            "/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/"
            ".bc_schema2_fk_consume_control_v2/franco_backhand_loop_b.success.json",
            "c0a25f2cba0e61bf0df7f63e6493948e16c5a3d3074f65091430f29e417f4f8b",
        ),
    }
    for label, (path, sha) in expected_inputs.items():
        row = exact_keys(inputs[label], {"path", "sha256"}, label)
        if row != {"path": path, "sha256": sha}:
            raise L0ContractError(f"{label} exact runtime binding changed")

    upstream = exact_keys(
        plan["upstream_contracts"],
        {
            "consume_activation",
            "consume_runner",
            "consume_source_gate_validator",
            "schema2_materializer",
            "schema2_preregistration",
            "shared_schema2_runtime",
            "donor_metadata",
            "joint_order_contract",
            "joint_order_validator",
            "runtime_joint_order",
            "runtime_body_order",
            "kinematics_contract",
            "converter_helper",
            "grounding_helper",
            "grounding_preregistration",
        },
        "upstream_contracts",
    )
    expected_repo_paths = {
        "consume_activation": "configs/motion_backhand_loop_bc_schema2_fk_consume_activation_20260714.json",
        "consume_runner": "scripts/run_motion_schema2_fk_consume_once.py",
        "consume_source_gate_validator": (
            "scripts/validate_motion_schema2_fk_consume_activation.py"
        ),
        "schema2_materializer": "scripts/materialize_motion_schema2_fk.py",
        "schema2_preregistration": "configs/motion_backhand_loop_b_schema2_fk_prereg_20260714.json",
        "shared_schema2_runtime": "configs/motion_backhand_loop_bc_schema2_fk_runtime_v1.json",
        "donor_metadata": "configs/a3_schema2_fk_donor_metadata_v1.json",
        "joint_order_contract": "configs/a3_joint_order_bijection_v1.json",
        "joint_order_validator": "hope_training/whole_body_tracking/scripts/a3_joint_order_contract.py",
        "runtime_joint_order": "configs/a3_runtime_articulation_joint_order.txt",
        "runtime_body_order": "configs/a3_runtime_body_order.txt",
        "kinematics_contract": "hope_training/whole_body_tracking/scripts/motion_kinematics_contract.py",
        "converter_helper": "hope_training/whole_body_tracking/scripts/csv_to_npz_mujoco.py",
        "grounding_helper": "scripts/ground_gmr_pkl.py",
        "grounding_preregistration": "configs/motion_video_canonical_gmr_ground_prereg_v2_20260711.json",
    }
    paths: dict[str, Path] = {}
    for label, relative in expected_repo_paths.items():
        paths[label] = verify_binding(
            upstream[label], label, repo_root=REPO_ROOT, expected_path=relative
        )

    grounding_source = read_json(
        paths["grounding_preregistration"], "grounding preregistration"
    )
    source_grounding = exact_keys(
        grounding_source.get("grounding_contract"),
        {
            "expected_fps",
            "target_clearance_m",
            "max_grounded_clearance_m",
            "numerical_tolerance_m",
            "max_abs_shift_m",
            "quaternion_norm_tolerance",
            "joint_range_tolerance_rad",
            "translation",
            "clearance_sampling",
            "continuous_time_clearance_proven",
        },
        "source grounding contract",
    )
    if source_grounding != {
        "expected_fps": 30.0,
        "target_clearance_m": 0.00001,
        "max_grounded_clearance_m": 0.001,
        "numerical_tolerance_m": 0.0000005,
        "max_abs_shift_m": 0.25,
        "quaternion_norm_tolerance": 0.000001,
        "joint_range_tolerance_rad": 0.00001,
        "translation": "one_constant_value_added_only_to_root_pos[:,2]",
        "clearance_sampling": "original_discrete_frames_only",
        "continuous_time_clearance_proven": False,
    }:
        raise L0ContractError("frozen source grounding thresholds changed")
    source_collision = exact_keys(
        grounding_source.get("compiled_collision_contract"),
        {
            "expected_sha256",
            "robot_root_body_id",
            "enabled_robot_geom_count",
            "enabled_robot_geom_ids",
            "surface_method",
            "visual_only_geoms_excluded",
            "ground_geom",
            "ground_z_m",
        },
        "source compiled collision contract",
    )

    materializer = _import_exact("materialize_motion_schema2_fk", paths["schema2_materializer"])
    shared = materializer.read_json(paths["shared_schema2_runtime"], "shared schema2 runtime")
    materializer.validate_shared_document(shared, repo_root=REPO_ROOT)
    joint_module = _import_exact("a3_joint_order_contract", paths["joint_order_validator"])
    joint_contract = joint_module.load_contract(
        upstream["joint_order_contract"]["path"], repo_root=REPO_ROOT
    )
    runtime_joint_names = _read_names(paths["runtime_joint_order"], 31, "runtime joint order")
    if runtime_joint_names != tuple(joint_contract.target_names):
        raise L0ContractError("runtime joint order differs from bound target order")
    runtime_body_names = _read_names(paths["runtime_body_order"], 32, "runtime body order")
    if runtime_body_names[0] != "pelvis_link":
        raise L0ContractError("runtime body order no longer starts with pelvis_link")

    model = exact_keys(
        plan["a3_model"],
        {"model_root", "canonical_mjcf", "derived_closure", "compiled_collision_contract"},
        "a3_model",
    )
    if model["model_root"] != (
        "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong"
    ):
        raise L0ContractError("A3 model root changed")
    mjcf_path = verify_binding(
        model["canonical_mjcf"],
        "canonical_mjcf",
        repo_root=REPO_ROOT,
        expected_path=(
            "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
            "a3_pingpong/a3_pingpong.xml"
        ),
    )
    actual_closure = materializer.derive_mjcf_closure(
        mjcf_path, REPO_ROOT / model["model_root"]
    )
    if model["derived_closure"] != actual_closure:
        raise L0ContractError(f"A3 model closure changed: actual={actual_closure}")
    collision = exact_keys(
        model["compiled_collision_contract"],
        {
            "sha256",
            "ground_geom",
            "ground_z_m",
            "enabled_robot_geom_count",
            "enabled_robot_geom_ids",
            "surface_method",
        },
        "compiled_collision_contract",
    )
    if collision != {
        "sha256": "18e7f6ffbefba9dbd988f7c3cb9fb92b250777862fc25fa3d4a0b2ca0f8386e5",
        "ground_geom": "floor",
        "ground_z_m": 0.0,
        "enabled_robot_geom_count": 37,
        "enabled_robot_geom_ids": [
            3, 9, 11, 13, 15, 17, 19, 21, 23, 24, 26, 29, 30, 32, 34, 36,
            38, 40, 42, 47, 48, 49, 50, 51, 52, 54, 56, 58, 61, 63, 65, 67,
            69, 71, 74, 76, 78,
        ],
        "surface_method": "analytic_primitive_support_or_compiled_mesh_vertices",
    }:
        raise L0ContractError("compiled collision contract changed")
    if (
        source_collision["expected_sha256"] != collision["sha256"]
        or source_collision["enabled_robot_geom_count"]
        != collision["enabled_robot_geom_count"]
        or source_collision["enabled_robot_geom_ids"] != collision["enabled_robot_geom_ids"]
        or source_collision["surface_method"] != collision["surface_method"]
        or source_collision["ground_geom"] != collision["ground_geom"]
        or source_collision["ground_z_m"] != collision["ground_z_m"]
        or source_collision["robot_root_body_id"] != 1
        or source_collision["visual_only_geoms_excluded"] is not True
    ):
        raise L0ContractError("L0 collision contract differs from frozen grounding source")

    runtime = exact_keys(
        plan["runtime"],
        {
            "launcher",
            "resolved_executable",
            "resolved_executable_bytes",
            "resolved_executable_sha256",
            "python_version",
            "packages",
            "module_origins",
            "environment",
        },
        "runtime",
    )
    if runtime != {
        "launcher": "/workspace/hope_mjeval_venv/bin/python",
        "resolved_executable": "/usr/bin/python3.12",
        "resolved_executable_bytes": 8021824,
        "resolved_executable_sha256": "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5",
        "python_version": "3.12.3",
        "packages": {"numpy": "2.5.0", "mujoco": "3.10.0"},
        "module_origins": {
            "numpy": "/workspace/hope_mjeval_venv/lib/python3.12/site-packages/numpy/__init__.py",
            "mujoco": "/workspace/hope_mjeval_venv/lib/python3.12/site-packages/mujoco/__init__.py",
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    }:
        raise L0ContractError("CPU audit runtime contract changed")

    contract = exact_keys(
        plan["l0_contract"],
        {
            "frames",
            "fps",
            "joint_count",
            "body_count",
            "npz_fields",
            "time_series_dtype",
            "quaternion_norm_tolerance",
            "joint_range_tolerance_rad",
            "kinematic_replay",
            "grounding",
            "support_bodies",
        },
        "l0_contract",
    )
    if {
        key: contract[key]
        for key in (
            "frames", "fps", "joint_count", "body_count", "npz_fields",
            "time_series_dtype", "quaternion_norm_tolerance",
            "joint_range_tolerance_rad", "kinematic_replay", "support_bodies",
        )
    } != {
        "frames": 151,
        "fps": 50,
        "joint_count": 31,
        "body_count": 32,
        "npz_fields": sorted(NPZ_FIELDS),
        "time_series_dtype": "float32",
        "quaternion_norm_tolerance": 0.00001,
        "joint_range_tolerance_rad": 0.00001,
        "kinematic_replay": (
            "stored_pelvis_pose_plus_runtime_joint_pos_to_exact_MJCF_mj_forward; "
            "recomputed_float32_link_pose_and_velocity_arrays_must_be_byte_equal"
        ),
        "support_bodies": ["left_ankle_roll_Link", "right_ankle_roll_Link"],
    }:
        raise L0ContractError("L0 structural/kinematic contract changed")
    grounding = exact_keys(
        contract["grounding"],
        {
            "target_clearance_m",
            "max_grounded_clearance_m",
            "numerical_tolerance_m",
            "sampling",
            "lowest_collision_body_must_descend_from_support_body",
        },
        "L0 grounding contract",
    )
    if grounding != {
        "target_clearance_m": 0.00001,
        "max_grounded_clearance_m": 0.001,
        "numerical_tolerance_m": 0.0000005,
        "sampling": "all_151_discrete_50Hz_frames_no_continuous_time_claim",
        "lowest_collision_body_must_descend_from_support_body": True,
    }:
        raise L0ContractError("L0 grounding contract changed")

    output = exact_keys(
        plan["output_contract"],
        {"certificate_path", "must_be_absent", "parent_must_exist", "no_clobber"},
        "output_contract",
    )
    if output != {
        "certificate_path": (
            "/workspace/codexschema/motion_video_intake_20260711/l0_static_primary_v1/"
            "franco_backhand_loop_b_98e7b883b29d.l0_static_certificate.json"
        ),
        "must_be_absent": True,
        "parent_must_exist": True,
        "no_clobber": True,
    }:
        raise L0ContractError("L0 output contract changed")
    if plan["authorization"] != {
        "source_gate_pass": True,
        "cpu_l0_audit_authorized_after_review": True,
        "l0_static_complete": False,
        "vendor_l1_authorized": False,
        "table_net_authorized": False,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }:
        raise L0ContractError("L0 source authorization changed")
    expected_non_claims = [
        "vendor_self_collision_or_racket_self_hit",
        "table_or_net_swept_clearance",
        "continuous_time_ground_clearance",
        "dynamics_balance_or_contact_stability",
        "TOPP_or_time_warp",
        "strike_or_returnability",
        "RL_training_or_checkpoint_quality",
        "Gate3_or_hardware_safety",
    ]
    if plan["explicit_non_claims"] != expected_non_claims:
        raise L0ContractError("explicit L0 boundaries changed")
    if plan["next_gate"] != (
        "only_after_exact_L0_certificate_vendor_L1_self_collision_and_racket_self_hit"
    ):
        raise L0ContractError("L0 next gate changed")
    return plan, actual_sha


def _scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise L0ContractError(f"{label} must be one non-object scalar string")
    item = array.item()
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise L0ContractError(f"{label} is not UTF-8") from exc
    if not isinstance(item, str):
        raise L0ContractError(f"{label} is not text")
    return item


def load_npz_exact(path: Path, plan: Mapping[str, Any]) -> dict[str, np.ndarray]:
    ensure_regular_no_symlink(path, "schema2 NPZ")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise L0ContractError(f"schema2 NPZ is not a valid ZIP: {exc}") from exc
    expected_members = {f"{name}.npy" for name in NPZ_FIELDS}
    if len(members) != len(set(members)) or set(members) != expected_members:
        raise L0ContractError("schema2 NPZ members are missing, duplicated or unexpected")
    contract = plan["l0_contract"]
    body_names_expected = _read_names(
        REPO_ROOT / plan["upstream_contracts"]["runtime_body_order"]["path"],
        32,
        "runtime body order",
    )
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
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != NPZ_FIELDS or len(data.files) != len(NPZ_FIELDS):
                raise L0ContractError("schema2 NPZ field set changed")
            fps = np.asarray(data["fps"])
            schema = np.asarray(data["kinematics_schema_version"])
            if fps.shape != (1,) or fps.dtype != np.int64 or int(fps[0]) != contract["fps"]:
                raise L0ContractError("schema2 fps must be exact int64 [50]")
            if schema.shape != (1,) or schema.dtype != np.int64 or int(schema[0]) != 2:
                raise L0ContractError("schema2 version must be exact int64 [2]")
            if _scalar_text(data["body_pos_point"], "body_pos_point") != "link_origin":
                raise L0ContractError("schema2 body_pos_point must be link_origin")
            if _scalar_text(data["body_lin_vel_point"], "body_lin_vel_point") != "center_of_mass":
                raise L0ContractError("schema2 body_lin_vel_point must be center_of_mass")
            names_raw = np.asarray(data["body_names"])
            if names_raw.shape != (32,) or names_raw.dtype.hasobject:
                raise L0ContractError("schema2 body_names shape/dtype changed")
            names = tuple(
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in names_raw.tolist()
            )
            if names != body_names_expected:
                raise L0ContractError("schema2 body_names differ from exact runtime order")
            for name, shape in expected_shapes.items():
                array = np.asarray(data[name])
                if array.shape != shape or array.dtype != np.float32:
                    raise L0ContractError(
                        f"schema2 {name} shape/dtype {array.shape}/{array.dtype} != {shape}/float32"
                    )
                if not np.isfinite(array).all():
                    raise L0ContractError(f"schema2 {name} contains NaN/Inf")
                arrays[name] = array.copy()
    except L0ContractError:
        raise
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile) as exc:
        raise L0ContractError(f"cannot load exact schema2 NPZ: {exc}") from exc
    quat_error = float(
        np.max(np.abs(np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1) - 1.0))
    )
    if quat_error > contract["quaternion_norm_tolerance"]:
        raise L0ContractError(
            f"body quaternion max norm error {quat_error:.9g} exceeds frozen tolerance"
        )
    dt = 1.0 / float(contract["fps"])
    expected_joint_vel = np.gradient(arrays["joint_pos"], dt, axis=0).astype(np.float32)
    if not np.array_equal(arrays["joint_vel"], expected_joint_vel):
        raise L0ContractError("joint_vel is not producer-exact gradient(joint_pos, 1/50)")
    arrays["_quaternion_max_norm_error"] = np.asarray(quat_error)
    return arrays


def validate_runtime_environment(plan: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    runtime = plan["runtime"]
    for key, expected in runtime["environment"].items():
        if os.environ.get(key) != expected:
            raise L0ContractError(f"runtime environment {key} must equal {expected!r}")
    resolved = Path(sys.executable).resolve()
    if str(resolved) != runtime["resolved_executable"]:
        raise L0ContractError(f"resolved Python executable changed: {resolved}")
    ensure_regular_no_symlink(resolved, "resolved Python executable")
    if (
        resolved.stat().st_size != runtime["resolved_executable_bytes"]
        or sha256_file(resolved) != runtime["resolved_executable_sha256"]
    ):
        raise L0ContractError("resolved Python executable content changed")
    version = ".".join(str(value) for value in sys.version_info[:3])
    if version != runtime["python_version"]:
        raise L0ContractError(f"Python version changed: {version}")
    if np.__version__ != runtime["packages"]["numpy"] or str(Path(np.__file__).resolve()) != runtime[
        "module_origins"
    ]["numpy"]:
        raise L0ContractError("NumPy version/origin changed")
    try:
        mujoco = importlib.import_module("mujoco")
    except ImportError as exc:
        raise L0ContractError("mujoco is missing from exact CPU audit runtime") from exc
    if (
        mujoco.__version__ != runtime["packages"]["mujoco"]
        or str(Path(mujoco.__file__).resolve()) != runtime["module_origins"]["mujoco"]
    ):
        raise L0ContractError("MuJoCo version/origin changed")
    return mujoco, {
        "launcher": runtime["launcher"],
        "resolved_executable": binding(resolved),
        "python_version": version,
        "numpy_version": np.__version__,
        "numpy_origin": str(Path(np.__file__).resolve()),
        "mujoco_version": mujoco.__version__,
        "mujoco_origin": str(Path(mujoco.__file__).resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def evaluate_joint_ranges(
    joint_pos: np.ndarray,
    ranges: np.ndarray,
    joint_names: Sequence[str],
    tolerance_rad: float,
) -> dict[str, Any]:
    values = np.asarray(joint_pos, dtype=np.float64)
    bounds = np.asarray(ranges, dtype=np.float64)
    if (
        values.ndim != 2
        or bounds.shape != (values.shape[1], 2)
        or len(joint_names) != values.shape[1]
        or not np.isfinite(values).all()
        or not np.isfinite(bounds).all()
        or not np.isfinite(tolerance_rad)
        or tolerance_rad < 0.0
    ):
        raise L0ContractError("joint range inputs are malformed or non-finite")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise L0ContractError("A3 model has an inverted joint range")
    excess = np.maximum(
        np.maximum(bounds[None, :, 0] - values, values - bounds[None, :, 1]), 0.0
    )
    worst_flat = int(np.argmax(excess))
    worst_frame, worst_col = (
        int(value) for value in np.unravel_index(worst_flat, excess.shape)
    )
    worst_excess = float(excess[worst_frame, worst_col])
    if worst_excess > tolerance_rad:
        raise L0ContractError(
            f"joint range excess {worst_excess:.9g} rad at frame {worst_frame}, "
            f"joint {joint_names[worst_col]!r}"
        )
    return {
        "max_excess_rad": worst_excess,
        "tolerance_rad": float(tolerance_rad),
        "worst_frame": worst_frame,
        "worst_joint": joint_names[worst_col],
    }


def evaluate_ground_clearance(
    clearances: np.ndarray, *, target_m: float, maximum_m: float, tolerance_m: float
) -> dict[str, Any]:
    values = np.asarray(clearances, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not all(np.isfinite(value) for value in (target_m, maximum_m, tolerance_m))
        or target_m <= 0.0
        or maximum_m < target_m
        or tolerance_m < 0.0
    ):
        raise L0ContractError("ground clearance inputs are malformed or non-finite")
    minimum = float(np.min(values))
    low = target_m - tolerance_m
    high = maximum_m + tolerance_m
    if minimum < low or minimum > high:
        raise L0ContractError(
            f"discrete-frame minimum ground clearance {minimum:.9g} m not in "
            f"[{low:.9g}, {high:.9g}]"
        )
    return {
        "minimum_clearance_m": minimum,
        "minimum_frame": int(np.argmin(values)),
        "maximum_of_frame_minima_m": float(np.max(values)),
        "sample_count": int(values.size),
    }


def _activation_content_identity(value: Any, label: str) -> dict[str, Any]:
    row = exact_keys(value, {"path", "bytes", "sha256"}, label)
    path = row["path"]
    size = row["bytes"]
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise L0ContractError(f"{label}.path must be a non-empty absolute provenance path")
    if ".." in Path(path).parts:
        raise L0ContractError(f"{label}.path may not contain parent traversal")
    if type(size) is not int or size <= 0:
        raise L0ContractError(f"{label}.bytes must be a positive integer")
    return {"bytes": size, "sha256": require_sha(row["sha256"], f"{label}.sha256")}


def _recorded_checkout_root(recorded_path: str, canonical_relative_path: str) -> Path:
    canonical = Path(canonical_relative_path)
    if canonical.is_absolute() or not canonical.parts or ".." in canonical.parts:
        raise L0ContractError("consume activation canonical path is unsafe")
    path = Path(recorded_path)
    if not path.is_absolute() or ".." in path.parts:
        raise L0ContractError("recorded consume activation path is unsafe")
    root = path
    for _part in canonical.parts:
        root = root.parent
    if root / canonical != path:
        raise L0ContractError(
            "recorded consume activation path does not end in the canonical repository path"
        )
    return root


def _current_checkout_commit(repo_root: Path) -> str:
    ensure_no_symlink_components(repo_root, "portable current checkout")
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    run = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo_root), "rev-parse", "HEAD"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    commit = run.stdout.strip()
    if run.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise L0ContractError("cannot bind portable current checkout commit")
    return commit


def build_portable_source_context(
    runner: Any,
    plan: Mapping[str, Any],
    recorded_source_checkout: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build the explicit current-source tuple; never fall back to the old absolute root."""

    recorded = exact_keys(
        recorded_source_checkout,
        {
            "path", "commit", "must_be_detached", "must_be_clean_before_and_after",
            "may_not_be_archive_or_live_a0",
        },
        "recorded source checkout",
    )
    commit = _current_checkout_commit(repo_root)
    try:
        checkout = runner.validate_detached_clean_checkout(repo_root, commit)
    except Exception as exc:
        raise L0ContractError(f"portable current checkout is not exact: {exc}") from exc
    upstream = plan["upstream_contracts"]
    context = {
        "current_checkout": checkout,
        "current_runner": dict(upstream["consume_runner"]),
        "current_source_gate_validator": dict(
            upstream["consume_source_gate_validator"]
        ),
        "runtime_body_order": dict(upstream["runtime_body_order"]),
        "recorded_source_checkout": dict(recorded),
    }
    try:
        runner._validate_portable_source_context(
            {
                "source_checkout": recorded,
                "runner": runner.HISTORICAL_RUNNER_BINDING,
                "source_gate_validator": (
                    runner.HISTORICAL_SOURCE_GATE_VALIDATOR_BINDING
                ),
            },
            context,
        )
    except Exception as exc:
        raise L0ContractError(f"portable current source context is not exact: {exc}") from exc
    return context


def portable_activation_context(
    plan: Mapping[str, Any],
    activation: Mapping[str, Any],
    receipt: Mapping[str, Any],
    current_meta: Mapping[str, Any],
    claim: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[Mapping[str, Any], Path]:
    """Bind old claim bytes without treating its checkout location as semantic identity."""

    canonical_binding = plan["upstream_contracts"]["consume_activation"]
    canonical_relative_path = canonical_binding["path"]
    expected_current_path = (repo_root / canonical_relative_path).resolve()
    current_identity = _activation_content_identity(current_meta, "current consume activation")
    if Path(current_meta["path"]) != expected_current_path:
        raise L0ContractError("current consume activation is not at its canonical repository path")
    expected_identity = {
        "bytes": canonical_binding["bytes"],
        "sha256": canonical_binding["sha256"],
    }
    if current_identity != expected_identity:
        raise L0ContractError("current consume activation differs from the preregistered content")

    claim_activation = _activation_content_identity(
        claim.get("activation"), "claim consume activation"
    )
    if claim_activation != current_identity:
        raise L0ContractError("claim consume activation content identity changed")

    source = exact_keys(
        activation.get("source_checkout"),
        {
            "path", "commit", "must_be_detached", "must_be_clean_before_and_after",
            "may_not_be_archive_or_live_a0",
        },
        "activation source checkout",
    )
    receipt_checkout = exact_keys(
        receipt.get("inspection_checkout"),
        {"path", "commit", "detached", "clean_before", "clean_after", "tracked_files"},
        "inspection receipt checkout",
    )
    claim_source = exact_keys(
        claim.get("source_checkout"), set(source), "claim source checkout"
    )
    if (
        claim_source != source
        or source["commit"] != receipt_checkout["commit"]
        or source["path"] != receipt_checkout["path"]
    ):
        raise L0ContractError("claim activation is not bound to the inspected source commit")

    recorded_meta = exact_keys(
        claim["activation"], {"path", "bytes", "sha256"}, "claim activation binding"
    )
    recorded_root = _recorded_checkout_root(
        recorded_meta["path"], canonical_relative_path
    )
    return recorded_meta, recorded_root


def validate_formal_result_portably(
    runner: Any,
    activation: Mapping[str, Any],
    receipt: Mapping[str, Any],
    asset: str,
    current_meta: Mapping[str, Any],
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    recorded_meta, recorded_root = portable_activation_context(
        plan, activation, receipt, current_meta, claim, repo_root=repo_root
    )
    portable_source_context = build_portable_source_context(
        runner,
        plan,
        activation["source_checkout"],
        repo_root=repo_root,
    )
    original_root = runner.gate.REPO_ROOT
    try:
        # The frozen historical runner recorded absolute checkout paths in three redundant lineage
        # fields.  Replaying its exact validator against the recorded lexical root preserves those
        # internal equalities without requiring that old root to exist or equal this checkout.
        runner.gate.REPO_ROOT = recorded_root
        return runner.validate_formal_result(
            activation,
            receipt,
            asset,
            recorded_meta,
            portable_source_context=portable_source_context,
        )
    finally:
        runner.gate.REPO_ROOT = original_root


def validate_upstream_result(plan: Mapping[str, Any]) -> dict[str, Any]:
    upstream = plan["upstream_contracts"]
    runner_path = REPO_ROOT / upstream["consume_runner"]["path"]
    runner = _import_exact("run_motion_schema2_fk_consume_once", runner_path)
    activation_path = REPO_ROOT / upstream["consume_activation"]["path"]
    ensure_regular_no_symlink(activation_path, "consume activation")
    if (
        activation_path.stat().st_size != upstream["consume_activation"]["bytes"]
        or sha256_file(activation_path) != upstream["consume_activation"]["sha256"]
    ):
        raise L0ContractError("current consume activation binding changed")
    activation_preview = read_json(activation_path, "consume activation")
    preview_source = exact_keys(
        activation_preview.get("source_checkout"),
        {
            "path", "commit", "must_be_detached", "must_be_clean_before_and_after",
            "may_not_be_archive_or_live_a0",
        },
        "activation source checkout",
    )
    portable_source_context = build_portable_source_context(
        runner, plan, preview_source
    )
    activation, receipt, activation_meta = runner.load_validated_contract_portably(
        activation_path,
        upstream["consume_activation"]["sha256"],
        portable_source_context,
    )
    if activation != activation_preview:
        raise L0ContractError("portable activation loader changed frozen activation bytes")
    claim_path = Path(plan["exact_runtime_inputs"]["consume_claim"]["path"])
    ensure_regular_no_symlink(claim_path, "consume claim")
    if sha256_file(claim_path) != plan["exact_runtime_inputs"]["consume_claim"]["sha256"]:
        raise L0ContractError("consume claim SHA changed before portable lineage validation")
    claim = read_json(claim_path, "consume claim")
    formal = validate_formal_result_portably(
        runner, activation, receipt, ASSET_ID, activation_meta, plan, claim
    )
    expected = plan["exact_runtime_inputs"]
    actual = {
        "motion_npz": formal["output"]["motion"],
        "materialization_report": formal["output"]["report"],
        "consume_claim": formal["claim"],
        "consume_success": formal["success"],
    }
    for label, record in actual.items():
        if record["path"] != expected[label]["path"] or record["sha256"] != expected[label][
            "sha256"
        ]:
            raise L0ContractError(f"upstream formal result {label} differs from frozen input")
    # Re-read JSON inputs through this audit's duplicate-key rejecting parser as
    # an independent guard, even though the exact upstream runner already did so.
    for label in ("materialization_report", "consume_claim", "consume_success"):
        path = Path(expected[label]["path"])
        verify_binding({**expected[label], "bytes": path.stat().st_size}, label)
        read_json(path, label)
    motion_path = Path(expected["motion_npz"]["path"])
    ensure_regular_no_symlink(motion_path, "motion_npz")
    if sha256_file(motion_path) != expected["motion_npz"]["sha256"]:
        raise L0ContractError("motion_npz SHA changed after upstream validation")
    return {
        "activation": activation_meta,
        "motion_npz": binding(motion_path),
        "materialization_report": binding(Path(expected["materialization_report"]["path"])),
        "consume_claim": binding(Path(expected["consume_claim"]["path"])),
        "consume_success": binding(Path(expected["consume_success"]["path"])),
        "runner_lineage": True,
        "npz_bound": True,
    }


def audit_kinematics(
    plan: Mapping[str, Any], arrays: Mapping[str, np.ndarray], mujoco: Any
) -> dict[str, Any]:
    upstream = plan["upstream_contracts"]
    ground = _import_exact("ground_gmr_pkl", REPO_ROOT / upstream["grounding_helper"]["path"])
    converter = _import_exact(
        "csv_to_npz_mujoco", REPO_ROOT / upstream["converter_helper"]["path"]
    )
    mjcf_path = REPO_ROOT / plan["a3_model"]["canonical_mjcf"]["path"]
    try:
        model_binding = ground.bind_model(
            mujoco, mjcf_path, ground_geom_name=plan["a3_model"]["compiled_collision_contract"]["ground_geom"]
        )
    except Exception as exc:
        raise L0ContractError(f"cannot bind exact compiled A3 model: {exc}") from exc
    collision = plan["a3_model"]["compiled_collision_contract"]
    if (
        model_binding.collision_contract_sha256 != collision["sha256"]
        or list(model_binding.collision_geom_ids) != collision["enabled_robot_geom_ids"]
        or len(model_binding.collision_geom_ids) != collision["enabled_robot_geom_count"]
        or model_binding.ground_z_m != collision["ground_z_m"]
    ):
        raise L0ContractError("compiled MuJoCo collision contract changed")

    joint_names = _read_names(
        REPO_ROOT / upstream["runtime_joint_order"]["path"], 31, "runtime joint order"
    )
    body_names = _read_names(
        REPO_ROOT / upstream["runtime_body_order"]["path"], 32, "runtime body order"
    )
    model = model_binding.model
    data = model_binding.data
    joint_ids = []
    qpos_addresses = []
    for name in joint_names:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise L0ContractError(f"runtime joint {name!r} missing from A3 model")
        joint_ids.append(jid)
        qpos_addresses.append(int(model.jnt_qposadr[jid]))
    if len(set(joint_ids)) != 31 or len(set(qpos_addresses)) != 31:
        raise L0ContractError("runtime joint mapping is not a 31-joint bijection")
    body_ids = []
    for name in body_names:
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        if bid < 0:
            raise L0ContractError(f"runtime body {name!r} missing from A3 model")
        body_ids.append(bid)
    if len(set(body_ids)) != 32 or body_ids[0] != model_binding.root_body_id:
        raise L0ContractError("runtime body mapping is not the exact root-first 32-body bijection")

    q = arrays["joint_pos"].astype(np.float64)
    ranges = np.asarray(model.jnt_range, dtype=np.float64)[joint_ids]
    tolerance_rad = float(plan["l0_contract"]["joint_range_tolerance_rad"])
    joint_range_result = evaluate_joint_ranges(q, ranges, joint_names, tolerance_rad)

    frames = int(plan["l0_contract"]["frames"])
    recomputed_pos = np.empty((frames, 32, 3), dtype=np.float32)
    recomputed_quat = np.empty((frames, 32, 4), dtype=np.float32)
    recomputed_com = np.empty((frames, 32, 3), dtype=np.float32)
    clearances = np.empty(frames, dtype=np.float64)
    lowest_body_ids = np.empty(frames, dtype=np.int64)
    root_adr = model_binding.root_qpos_address
    for frame in range(frames):
        data.qpos[:] = model.qpos0
        data.qpos[root_adr : root_adr + 3] = arrays["body_pos_w"][frame, 0]
        data.qpos[root_adr + 3 : root_adr + 7] = arrays["body_quat_w"][frame, 0]
        data.qpos[qpos_addresses] = arrays["joint_pos"][frame]
        mujoco.mj_forward(model, data)
        recomputed_pos[frame] = np.asarray(data.xpos, dtype=np.float64)[body_ids].astype(np.float32)
        recomputed_quat[frame] = np.asarray(data.xquat, dtype=np.float64)[body_ids].astype(np.float32)
        recomputed_com[frame] = np.asarray(data.xipos, dtype=np.float64)[body_ids].astype(np.float32)
        minima = np.asarray(
            [
                ground.geom_world_min_z(mujoco, model, data, gid)
                for gid in model_binding.collision_geom_ids
            ],
            dtype=np.float64,
        )
        index = int(np.argmin(minima))
        gid = int(model_binding.collision_geom_ids[index])
        clearances[frame] = float(minima[index] - model_binding.ground_z_m)
        lowest_body_ids[frame] = int(model.geom_bodyid[gid])

    if not np.array_equal(recomputed_pos, arrays["body_pos_w"]):
        delta = float(np.max(np.abs(recomputed_pos - arrays["body_pos_w"])))
        raise L0ContractError(f"stored link positions differ from exact FK; max_abs={delta:.9g}")
    if not np.array_equal(recomputed_quat, arrays["body_quat_w"]):
        delta = float(np.max(np.abs(recomputed_quat - arrays["body_quat_w"])))
        raise L0ContractError(f"stored link quaternions differ from exact FK; max_abs={delta:.9g}")
    dt = 1.0 / float(plan["l0_contract"]["fps"])
    recomputed_lin = np.gradient(recomputed_com, dt, axis=0).astype(np.float32)
    if not np.array_equal(recomputed_lin, arrays["body_lin_vel_w"]):
        delta = float(np.max(np.abs(recomputed_lin - arrays["body_lin_vel_w"])))
        raise L0ContractError(f"stored COM velocities differ from exact producer; max_abs={delta:.9g}")
    recomputed_ang = np.stack(
        [converter.so3_derivative(recomputed_quat[:, col], dt) for col in range(32)],
        axis=1,
    ).astype(np.float32)
    if not np.array_equal(recomputed_ang, arrays["body_ang_vel_w"]):
        delta = float(np.max(np.abs(recomputed_ang - arrays["body_ang_vel_w"])))
        raise L0ContractError(f"stored angular velocities differ from exact producer; max_abs={delta:.9g}")

    support_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in plan["l0_contract"]["support_bodies"]
    ]
    if any(body_id < 0 for body_id in support_ids):
        raise L0ContractError("support body missing from exact A3 model")
    bad_frames = [
        frame
        for frame, body_id in enumerate(lowest_body_ids.tolist())
        if not any(ground._descends_from(model, body_id, support_id) for support_id in support_ids)
    ]
    if bad_frames:
        raise L0ContractError(
            f"lowest collision body is not under either support foot at frames {bad_frames[:12]}"
        )
    if not np.isfinite(clearances).all():
        raise L0ContractError("ground clearance contains NaN/Inf")
    ground_contract = plan["l0_contract"]["grounding"]
    clearance_result = evaluate_ground_clearance(
        clearances,
        target_m=ground_contract["target_clearance_m"],
        maximum_m=ground_contract["max_grounded_clearance_m"],
        tolerance_m=ground_contract["numerical_tolerance_m"],
    )
    return {
        "model": {
            "canonical_mjcf": binding(mjcf_path),
            "compiled_collision_sha256": model_binding.collision_contract_sha256,
            "enabled_robot_geom_ids": list(model_binding.collision_geom_ids),
            "ground_geom": collision["ground_geom"],
            "ground_z_m": model_binding.ground_z_m,
        },
        "joint_ranges": joint_range_result,
        "kinematic_replay": {
            "frames": frames,
            "link_position_float32_byte_equal": True,
            "link_quaternion_float32_byte_equal": True,
            "joint_velocity_float32_byte_equal": True,
            "com_linear_velocity_float32_byte_equal": True,
            "body_angular_velocity_float32_byte_equal": True,
            "mj_step_calls": 0,
        },
        "grounding": {
            **clearance_result,
            "all_lowest_collision_bodies_under_support_feet": True,
            "continuous_time_clearance_proven": False,
        },
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_certificate(
    plan: Mapping[str, Any], plan_path: Path, plan_sha: str
) -> dict[str, Any]:
    mujoco, runtime = validate_runtime_environment(plan)
    lineage = validate_upstream_result(plan)
    arrays = load_npz_exact(Path(plan["exact_runtime_inputs"]["motion_npz"]["path"]), plan)
    audit = audit_kinematics(plan, arrays, mujoco)
    return {
        "schema_version": 1,
        "status": CERTIFICATE_STATUS,
        "completed_utc": utc_now(),
        "scope": (
            "Exact CPU-only discrete-frame L0 static certificate for one B schema-2 NPZ; "
            "no simulator step or downstream safety/behavior claim"
        ),
        "asset_id": ASSET_ID,
        "preregistration": {"path": str(plan_path), "sha256": plan_sha},
        "validator": plan["validator"],
        "runtime": runtime,
        "lineage": lineage,
        "structure": {
            "frames": 151,
            "fps": 50,
            "joint_count": 31,
            "body_count": 32,
            "kinematics_schema_version": 2,
            "body_pos_point": "link_origin",
            "body_lin_vel_point": "center_of_mass",
            "time_series_dtype": "float32",
            "finite": True,
            "body_quaternion_max_norm_error": float(arrays["_quaternion_max_norm_error"]),
            "body_quaternion_norm_tolerance": plan["l0_contract"]["quaternion_norm_tolerance"],
        },
        "audit": audit,
        "authorization": {
            "l0_static_complete": True,
            "vendor_l1_authorized": True,
            "table_net_authorized": False,
            "dynamics_authorized": False,
            "simulator_authorized": False,
            "training_authorized": False,
            "formal_motion_authorized": False,
            "hardware_authorized": False,
        },
        "explicit_non_claims": plan["explicit_non_claims"],
        "next_gate": plan["next_gate"],
    }


def write_certificate_exclusive(path: Path, certificate: Mapping[str, Any]) -> None:
    ensure_no_symlink_components(path, "certificate path")
    parent = path.parent
    try:
        info = parent.stat()
    except OSError as exc:
        raise L0ContractError(f"certificate parent must already exist: {parent}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or parent.is_symlink():
        raise L0ContractError("certificate parent must be one real directory")
    if _lexists(path):
        raise L0ContractError(f"certificate path already exists; no-clobber: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json_bytes(certificate))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
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
        plan, plan_sha = validate_plan(args.prereg.resolve(), args.expected_prereg_sha256)
        if args.command == "static":
            print(
                f"[motion-l0] PASS static asset={ASSET_ID} source_exact=true "
                "runtime_audit=false no_write=true"
            )
            return 0
        output = Path(plan["output_contract"]["certificate_path"])
        if _lexists(output):
            raise L0ContractError(f"certificate path already exists; no-clobber: {output}")
        certificate = build_certificate(plan, args.prereg.resolve(), plan_sha)
        if args.command == "dry-run":
            print(
                f"[motion-l0] PASS dry-run asset={ASSET_ID} runtime_audit=true "
                "certificate_written=false l0_static_complete=false downstream_blocked=true"
            )
            return 0
        write_certificate_exclusive(output, certificate)
        print(
            f"[motion-l0] PASS audit asset={ASSET_ID} l0_static=true "
            f"certificate_sha256={sha256_file(output)} downstream_blocked=true"
        )
        return 0
    except (L0ContractError, OSError, TypeError, ValueError) as exc:
        print(f"[motion-l0] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
