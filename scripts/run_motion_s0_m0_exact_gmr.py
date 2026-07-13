#!/usr/bin/env python3
"""Inspect or consume the exact S0/M0 canonical-beta -> A3 GMR plans.

The two batches are deliberately independent.  ``static`` validates only the
committed machine contract, ``inspect`` verifies every private/runtime binding
without creating the output root, and ``consume`` creates a new no-clobber root
and publishes its completion manifest last.  Nothing in this tool authorizes
schema-2, simulation, training, TOPP, Gate3, hardware, or a strike claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^[0-9a-f]{40}$")
SAFE_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
BODY_SHAPE_CONTRACT = "diagnostic_same_performer_coordinatewise_median_betas_v1"
PLAN_STATUS = "preregistered_not_executed"
BLOCKED_RUNTIME_STATUS = "blocked_pending_exact_ignored_gmr_source_closure"
MATERIALIZATION_STATUS = "complete_exact_donor_beta_materialization"
EXPECTED_BATCHES = {
    "s0_static_high_press": ["static_backhand_high_press"],
    "m0_lateral_teachers": [
        "lateral_step_left_1",
        "lateral_step_left_2",
        "lateral_step_right_1",
        "lateral_step_right_2",
    ],
}
REQUIRED_GMR_RUNTIME_FILES = (
    "converter",
    "package_init",
    "smpl_loader",
    "motion_retarget",
    "params",
    "kinematics_model",
    "robot_motion_viewer",
    "data_loader",
    "neck_retarget",
    "neutral_smplx_model",
    "a3_retarget_mjcf",
    "smplx_to_a3_mapping",
)


class ContractError(ValueError):
    """The preregistered exact-GMR contract cannot be satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def normalized_pip_freeze_bytes(text: str) -> bytes:
    lines = sorted(line.strip().encode("utf-8") for line in text.splitlines() if line.strip())
    return b"\n".join(lines) + b"\n"


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON mapping: {path}")
    return value


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def require_git_oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or not GIT_OID.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase 40-character git object id")
    return value


def require_binding(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a mapping")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise ContractError(f"{label}.path must be a non-empty string")
    if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        raise ContractError(f"{label}.bytes must be a positive integer")
    require_sha(value.get("sha256"), f"{label}.sha256")
    return value


def verify_regular_file(binding: dict[str, Any], label: str, *, root: Path | None = None) -> Path:
    raw = Path(binding["path"])
    candidate = root / raw if root is not None and not raw.is_absolute() else raw
    absolute = Path(os.path.abspath(candidate))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{label} path must not traverse a symlink: {current}")
    path = absolute.resolve()
    if not path.is_file():
        raise ContractError(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size != binding["bytes"]:
        raise ContractError(f"{label} bytes {path.stat().st_size} != {binding['bytes']}: {path}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise ContractError(f"{label} sha256 {actual} != {binding['sha256']}: {path}")
    return path


def verify_real_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{label} path must not traverse a symlink: {current}")
    resolved = absolute.resolve()
    if not resolved.is_dir():
        raise ContractError(f"{label} must be a real directory: {resolved}")
    return resolved


def tree_fingerprint(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"dependency tree must be a real directory: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"dependency tree contains a symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not rows:
        raise ContractError(f"dependency tree is empty: {root}")
    return {
        "algorithm": "sha256(canonical-json(sorted[{path,bytes,sha256}]))-v1",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
    }


def closed_window_sample_mapping(window: Any, fps: int, frames: int, label: str) -> dict[str, Any]:
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in window)
    ):
        raise ContractError(f"{label} must be two finite seconds")
    start, end = (float(window[0]), float(window[1]))
    if start < 0.0 or end < start:
        raise ContractError(f"{label} must satisfy 0 <= start <= end")
    # Decimal prereg values such as 0.833333 intentionally mean the samples
    # whose exact t=i/fps lies in the closed interval, not a rounded endpoint.
    indices = [index for index in range(frames) if start <= index / fps <= end]
    if not indices:
        raise ContractError(f"{label} selects no GMR samples")
    return {
        "time_rule": "sample_i_time_seconds=i/30; closed interval start<=t_i<=end",
        "indices": indices,
        "count": len(indices),
        "first_index": indices[0],
        "last_index": indices[-1],
    }


def _require_string_list(value: Any, label: str, *, exact_count: int | None = None) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ContractError(f"{label} must be a non-empty string list")
    if exact_count is not None and len(value) != exact_count:
        raise ContractError(f"{label} must contain exactly {exact_count} values")
    if len(value) != len(set(value)):
        raise ContractError(f"{label} must not contain duplicates")
    return value


def _validate_source_contract(plan: dict[str, Any]) -> None:
    source = plan.get("ignored_gmr_source")
    if not isinstance(source, dict):
        raise ContractError("ignored_gmr_source must be a mapping")
    expected_source_keys = {
        "root",
        "commit",
        "tree_oid",
        "worktree_must_be_clean",
        "recovery_bundle",
        "runtime_files",
        "checkpoint_contract",
        "retarget_joint_order",
        "retarget_body_order",
        "retarget_foot_site_mapping",
        "joint_bijection_to_canonical",
    }
    if set(source) != expected_source_keys:
        raise ContractError("ignored_gmr_source field closure changed")
    if not isinstance(source.get("root"), str) or not Path(source["root"]).is_absolute():
        raise ContractError("ignored_gmr_source.root must be an absolute path")
    require_git_oid(source.get("commit"), "ignored_gmr_source.commit")
    require_git_oid(source.get("tree_oid"), "ignored_gmr_source.tree_oid")
    if source.get("worktree_must_be_clean") is not True:
        raise ContractError("ignored GMR worktree must be required clean")
    require_binding(source.get("recovery_bundle"), "ignored_gmr_source.recovery_bundle")
    files = source.get("runtime_files")
    if not isinstance(files, dict):
        raise ContractError("ignored_gmr_source.runtime_files must be a mapping")
    if set(files) != set(REQUIRED_GMR_RUNTIME_FILES):
        raise ContractError("ignored GMR runtime file closure changed")
    for name in REQUIRED_GMR_RUNTIME_FILES:
        require_binding(files.get(name), f"ignored_gmr_source.runtime_files.{name}")
    checkpoints = source.get("checkpoint_contract")
    if checkpoints != {
        "runtime_checkpoint_files": [],
        "semantic": "checkpoint_free_optimization_converter; GVHMR inference is already frozen in each input PT",
        "evidence": "bound converter/import closure has no checkpoint CLI argument or runtime checkpoint read",
    }:
        raise ContractError("GMR checkpoint-free runtime contract changed")
    retarget_joints = _require_string_list(
        source.get("retarget_joint_order"), "ignored_gmr_source.retarget_joint_order", exact_count=31
    )
    canonical_joints = plan["a3_robot_contract"]["joint_order"]
    if set(retarget_joints) != set(canonical_joints):
        raise ContractError("GMR retarget and canonical A3 joint-name sets differ")
    _require_string_list(source.get("retarget_body_order"), "ignored_gmr_source.retarget_body_order")
    retarget_sites = source.get("retarget_foot_site_mapping")
    if not isinstance(retarget_sites, dict) or set(retarget_sites) != {"left", "right"}:
        raise ContractError("GMR retarget foot-site mapping must bind left and right")
    for side, row in retarget_sites.items():
        if (
            not isinstance(row, dict)
            or set(row) != {"site", "parent_body", "local_pos_m"}
            or not isinstance(row["site"], str)
            or not row["site"]
            or not isinstance(row["parent_body"], str)
            or not row["parent_body"]
            or not isinstance(row["local_pos_m"], list)
            or len(row["local_pos_m"]) != 3
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in row["local_pos_m"]
            )
        ):
            raise ContractError(f"GMR retarget {side} foot-site mapping is malformed")
    bijection = source.get("joint_bijection_to_canonical")
    if not isinstance(bijection, list) or len(bijection) != 31:
        raise ContractError("ignored GMR source must bind an explicit 31-joint bijection")
    for index, row in enumerate(bijection):
        joint = retarget_joints[index]
        canonical_index = canonical_joints.index(joint)
        expected = {
            "gmr_dof_index": index,
            "gmr_joint": joint,
            "canonical_qpos_index": canonical_index + 7,
            "canonical_joint": canonical_joints[canonical_index],
        }
        if row != expected:
            raise ContractError(f"GMR/canonical joint bijection mismatch at index {index}")


def _validate_robot_contract(plan: dict[str, Any]) -> None:
    robot = plan.get("a3_robot_contract")
    if not isinstance(robot, dict):
        raise ContractError("a3_robot_contract must be a mapping")
    if set(robot) != {
        "canonical_mjcf",
        "canonical_model_tree",
        "joint_order_source",
        "joint_order",
        "body_order",
        "gmr_output_to_mjcf_qpos",
        "foot_site_mapping",
    }:
        raise ContractError("a3_robot_contract field closure changed")
    require_binding(robot.get("canonical_mjcf"), "a3_robot_contract.canonical_mjcf")
    tree = robot.get("canonical_model_tree")
    if not isinstance(tree, dict):
        raise ContractError("a3_robot_contract.canonical_model_tree must be a mapping")
    expected_algorithm = "sha256(canonical-json(sorted[{path,bytes,sha256}]))-v1"
    if tree.get("algorithm") != expected_algorithm:
        raise ContractError("canonical model tree algorithm changed")
    if not isinstance(tree.get("root"), str):
        raise ContractError("canonical model tree root must be a path")
    if not isinstance(tree.get("file_count"), int) or tree["file_count"] <= 0:
        raise ContractError("canonical model tree file_count must be positive")
    if not isinstance(tree.get("total_bytes"), int) or tree["total_bytes"] <= 0:
        raise ContractError("canonical model tree total_bytes must be positive")
    require_sha(tree.get("manifest_sha256"), "canonical model tree manifest_sha256")
    require_binding(robot.get("joint_order_source"), "a3_robot_contract.joint_order_source")
    joints = _require_string_list(robot.get("joint_order"), "a3_robot_contract.joint_order", exact_count=31)
    bodies = _require_string_list(robot.get("body_order"), "a3_robot_contract.body_order")
    if len(bodies) != 32 or bodies[0] != "pelvis_link":
        raise ContractError("A3 body order must bind pelvis plus 31 ordered link bodies")
    qpos = robot.get("gmr_output_to_mjcf_qpos")
    if qpos != {
        "root_pos": "qpos[0:3]",
        "root_rot_input": "xyzw",
        "root_rot_mjcf": "qpos[3:7]=[w,x,y,z]",
        "dof_pos": "qpos[7:38] via explicit 31-name/index bijection",
    }:
        raise ContractError("GMR output to A3 qpos mapping changed")
    sites = robot.get("foot_site_mapping")
    expected_sites = {
        "left": {
            "site": "left_foot",
            "parent_body": "left_ankle_roll_Link",
            "local_pos_m": [0.04, 0.0, -0.067],
        },
        "right": {
            "site": "right_foot",
            "parent_body": "right_ankle_roll_Link",
            "local_pos_m": [0.04, 0.0, -0.067],
        },
    }
    if sites != expected_sites:
        raise ContractError("A3 foot-site mapping changed")
    # Make unused-variable intent explicit: exact order is validated above and
    # again against both XMLs during inspect.
    del joints


def _validate_execution_contract(plan: dict[str, Any]) -> None:
    execution = plan.get("execution_contract")
    if not isinstance(execution, dict):
        raise ContractError("execution_contract must be a mapping")
    if set(execution) != {
        "cpu_only",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "PYTHONDONTWRITEBYTECODE",
        "robot",
        "target_fps",
        "warmup_threshold_strict_lt",
        "warmup_max_rounds",
        "timeout_seconds_per_asset",
        "python_environment",
        "required_imports",
        "converter_argv_template",
    }:
        raise ContractError("execution_contract field closure changed")
    if execution.get("cpu_only") is not True or execution.get("CUDA_VISIBLE_DEVICES") != "":
        raise ContractError("exact GMR must remain CPU-only")
    if execution.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
        raise ContractError("Torch compatibility environment changed")
    if execution.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ContractError("inspect must suppress Python bytecode writes")
    for field in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if not isinstance(execution.get(field), int) or execution[field] <= 0:
            raise ContractError(f"execution_contract.{field} must be positive")
    python = execution.get("python_environment")
    if not isinstance(python, dict) or not isinstance(python.get("executable"), str):
        raise ContractError("python_environment must bind an executable")
    if set(python) != {
        "executable",
        "executable_bytes",
        "executable_sha256",
        "version",
        "pip_version",
        "pip_freeze_command",
        "pip_freeze_normalization",
        "pip_freeze_sha256",
        "xrobot_utils_resolution",
    }:
        raise ContractError("python_environment field closure changed")
    if not isinstance(python.get("executable_bytes"), int) or python["executable_bytes"] <= 0:
        raise ContractError("python executable bytes must be positive")
    require_sha(python.get("executable_sha256"), "python_environment.executable_sha256")
    if not isinstance(python.get("version"), str) or not python["version"]:
        raise ContractError("python_environment.version must be non-empty")
    if not isinstance(python.get("pip_version"), str) or not python["pip_version"]:
        raise ContractError("python_environment.pip_version must be non-empty")
    if python.get("pip_freeze_command") != "python -m pip freeze --all":
        raise ContractError("pip freeze command changed")
    if python.get("pip_freeze_normalization") != "strip nonempty lines; bytewise sort; join LF; append one LF; sha256":
        raise ContractError("pip freeze normalization changed")
    require_sha(python.get("pip_freeze_sha256"), "python_environment.pip_freeze_sha256")
    resolution = python.get("xrobot_utils_resolution")
    if not isinstance(resolution, dict) or set(resolution) != {
        "status",
        "origin",
        "bytes",
        "sha256",
    }:
        raise ContractError("xrobot_utils resolution contract changed")
    if resolution["status"] == "absent":
        if any(resolution[field] is not None for field in ("origin", "bytes", "sha256")):
            raise ContractError("absent xrobot_utils must have null file binding")
    elif resolution["status"] == "regular_file":
        require_binding(
            {
                "path": resolution["origin"],
                "bytes": resolution["bytes"],
                "sha256": resolution["sha256"],
            },
            "python_environment.xrobot_utils_resolution",
        )
    else:
        raise ContractError("xrobot_utils status must be absent or regular_file")
    imports = _require_string_list(execution.get("required_imports"), "required_imports")
    if "mujoco" not in imports or "torch" not in imports:
        raise ContractError("required imports must include mujoco and torch")
    if execution.get("robot") != "agibot_a3" or execution.get("target_fps") != 30:
        raise ContractError("execution must bind agibot_a3 at 30 Hz")
    if execution.get("warmup_threshold_strict_lt") != 0.0001:
        raise ContractError("warmup threshold must remain 1e-4")
    if execution.get("warmup_max_rounds") != 200:
        raise ContractError("warmup max rounds must remain 200")
    timeout = execution.get("timeout_seconds_per_asset")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ContractError("per-asset timeout must be finite and positive")
    argv = execution.get("converter_argv_template")
    if argv != [
        "{python}",
        "{converter}",
        "--gvhmr_pred_file",
        "{input}",
        "--robot",
        "agibot_a3",
        "--save_path",
        "{output}",
    ]:
        raise ContractError("converter argv template changed")


def _validate_m0_stance(plan: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    stance = plan.get("m0_stance_contract")
    if not isinstance(stance, dict):
        raise ContractError("M0 requires m0_stance_contract")
    required = {
        "measurement_stage": "exact_robot_coordinate_gmr_before_schema2_promotion",
        "normalization": "subtract_frame0_common_root_xy_then_rotate_frame0_pelvis_heading_to_plus_x",
        "vector_definition": "d_xy=right_foot_xy-left_foot_xy",
        "coordinate_axes": {"x": "forward", "y": "left"},
        "window_estimator": "coordinatewise_median_over_preregistered_closed_sample_indices",
        "must_preserve_components": ["lateral_separation", "fore_aft_stagger"],
        "feet_together_or_narrower_substitute_allowed": False,
    }
    for field, expected in required.items():
        if stance.get(field) != expected:
            raise ContractError(f"m0_stance_contract.{field} changed")
    tolerances = stance.get("preregistered_tolerances_m")
    if tolerances != {
        "fore_aft_component_abs_error_max": 0.03,
        "lateral_component_abs_error_max": 0.03,
        "lateral_narrowing_max": 0.005,
        "minimum_initial_abs_lateral_separation": 0.05,
    }:
        raise ContractError("M0 preregistered stance tolerances changed")
    placeholders = stance.get("result_placeholders_before_execution")
    if placeholders != {
        "initial_d_xy_m": None,
        "terminal_d_xy_m": None,
        "fore_aft_stagger_initial_m": None,
        "fore_aft_stagger_terminal_m": None,
        "lateral_separation_initial_signed_m": None,
        "lateral_separation_terminal_signed_m": None,
        "lateral_separation_initial_abs_m": None,
        "lateral_separation_terminal_abs_m": None,
        "stance_passed": None,
    }:
        raise ContractError("M0 result placeholders must remain null before execution")
    mappings = stance.get("ready_window_sample_mappings")
    if not isinstance(mappings, list) or len(mappings) != len(rows):
        raise ContractError("M0 must freeze one sample mapping per input")
    by_id = {mapping.get("asset_id"): mapping for mapping in mappings if isinstance(mapping, dict)}
    if len(by_id) != len(mappings):
        raise ContractError("M0 sample mappings contain malformed or duplicate ids")
    for row in rows:
        mapping = by_id.get(row["asset_id"])
        if not isinstance(mapping, dict):
            raise ContractError(f"M0 lacks sample mapping for {row['asset_id']}")
        before = closed_window_sample_mapping(
            row["ready_before_window_s"], 30, row["frames"], f"{row['asset_id']} ready_before"
        )
        after = closed_window_sample_mapping(
            row["ready_after_window_s"], 30, row["frames"], f"{row['asset_id']} ready_after"
        )
        if mapping != {"asset_id": row["asset_id"], "ready_before": before, "ready_after": after}:
            raise ContractError(f"M0 frozen sample mapping mismatch for {row['asset_id']}")


def validate_plan(plan_path: Path, expected_plan_sha256: str, repo_root: Path) -> dict[str, Any]:
    require_sha(expected_plan_sha256, "--expected-plan-sha256")
    actual = sha256_file(plan_path)
    if actual != expected_plan_sha256:
        raise ContractError(f"plan sha256 {actual} != expected {expected_plan_sha256}")
    plan = read_json(plan_path, "exact GMR preregistration")
    if plan.get("schema_version") != 1 or plan.get("status") != PLAN_STATUS:
        raise ContractError(f"plan must be schema 1 and status {PLAN_STATUS}")
    batch = plan.get("batch_kind")
    if batch not in EXPECTED_BATCHES:
        raise ContractError(f"unsupported batch_kind: {batch!r}")
    if plan.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise ContractError("body-shape contract changed")
    allowed_plan_keys = {
        "schema_version",
        "plan_id",
        "status",
        "batch_kind",
        "scope",
        "human_owner",
        "executor",
        "body_shape_contract",
        "formal_eligible",
        "schema2_authorized",
        "training_authorized",
        "hardware_authorized",
        "runtime_contract",
        "source_materialization",
        "processing_order",
        "inputs",
        "s0_semantic_guard",
        "m0_stance_contract",
        "output_contract",
        "formal_blockers",
    }
    if set(plan) != allowed_plan_keys:
        raise ContractError("batch plan field closure changed")
    for flag in ("formal_eligible", "schema2_authorized", "training_authorized", "hardware_authorized"):
        if plan.get(flag) is not False:
            raise ContractError(f"{flag} must remain false")
    if plan.get("human_owner") != "Franco" or plan.get("executor") != "Codex":
        raise ContractError("exact-GMR owner/executor contract changed")
    _require_string_list(plan.get("formal_blockers"), "formal_blockers")
    runtime_binding = require_binding(plan.get("runtime_contract"), "runtime_contract")
    runtime_path = verify_regular_file(runtime_binding, "shared exact-GMR runtime contract", root=repo_root)
    runtime = read_json(runtime_path, "shared exact-GMR runtime contract")
    if runtime.get("schema_version") == 1 and runtime.get("status") == BLOCKED_RUNTIME_STATUS:
        _validate_blocked_runtime(runtime, repo_root)
        unresolved = runtime["required_unresolved_evidence"]
        raise ContractError(
            "shared exact-GMR runtime is intentionally blocked; unresolved exact evidence: "
            + ", ".join(item["json_pointer"] for item in unresolved)
        )
    if runtime.get("schema_version") != 1 or runtime.get("status") != PLAN_STATUS:
        raise ContractError("shared exact-GMR runtime contract is not executable schema 1")
    if set(runtime) != {
        "schema_version",
        "status",
        "scope",
        "tool_contract",
        "ignored_gmr_source",
        "a3_robot_contract",
        "execution_contract",
    }:
        raise ContractError("shared exact-GMR runtime field closure changed")
    runtime_keys = (
        "tool_contract",
        "ignored_gmr_source",
        "a3_robot_contract",
        "execution_contract",
    )
    for key in runtime_keys:
        if key in plan:
            raise ContractError(f"batch plan must not override shared runtime field {key}")
        if key not in runtime:
            raise ContractError(f"shared runtime contract lacks {key}")
        plan[key] = runtime[key]
    plan["_runtime_contract_binding"] = runtime_binding
    tool = plan.get("tool_contract")
    if not isinstance(tool, dict):
        raise ContractError("tool_contract must be a mapping")
    if set(tool) != {"consumer", "result_auditor"}:
        raise ContractError("tool_contract field closure changed")
    require_binding(tool.get("consumer"), "tool_contract.consumer")
    require_binding(tool.get("result_auditor"), "tool_contract.result_auditor")
    predecessor = plan.get("source_materialization")
    if not isinstance(predecessor, dict):
        raise ContractError("source_materialization must be a mapping")
    if set(predecessor) != {
        "body_shape_contract",
        "preregistration",
        "completion_manifest",
        "canonical_betas_artifact",
        "canonical_vector_sha256",
    }:
        raise ContractError("source_materialization field closure changed")
    require_binding(predecessor.get("preregistration"), "source_materialization.preregistration")
    require_binding(predecessor.get("completion_manifest"), "source_materialization.completion_manifest")
    require_binding(predecessor.get("canonical_betas_artifact"), "source_materialization.canonical_betas_artifact")
    require_sha(predecessor.get("canonical_vector_sha256"), "canonical_vector_sha256")
    if predecessor.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise ContractError("source materialization body-shape contract changed")
    rows = plan.get("inputs")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_BATCHES[batch]):
        raise ContractError(f"{batch} input count changed")
    ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"inputs[{index}] must be a mapping")
        if set(row) != {
            "asset_id",
            "frames",
            "ready_before_window_s",
            "ready_after_window_s",
            "canonical_vector_sha256",
            "input",
        }:
            raise ContractError(f"inputs[{index}] field closure changed")
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not SAFE_ASSET_ID.fullmatch(asset_id):
            raise ContractError(f"inputs[{index}].asset_id is unsafe")
        ids.append(asset_id)
        if not isinstance(row.get("frames"), int) or row["frames"] <= 1:
            raise ContractError(f"{asset_id}.frames must be > 1")
        require_binding(row.get("input"), f"{asset_id}.input")
        for window in ("ready_before_window_s", "ready_after_window_s"):
            closed_window_sample_mapping(row.get(window), 30, row["frames"], f"{asset_id}.{window}")
        if row.get("canonical_vector_sha256") != predecessor.get("canonical_vector_sha256"):
            raise ContractError(f"{asset_id} canonical vector SHA mismatch")
    if ids != EXPECTED_BATCHES[batch] or plan.get("processing_order") != ids:
        raise ContractError(f"{batch} asset order changed")
    _validate_robot_contract(plan)
    _validate_source_contract(plan)
    _validate_execution_contract(plan)
    output = plan.get("output_contract")
    if not isinstance(output, dict):
        raise ContractError("output_contract must be a mapping")
    required_output = {
        "result_suffix": ".exact_franco_donor_betas.gmr.pkl",
        "output_root_must_not_exist": True,
        "no_clobber": True,
        "stop_on_first_failure": True,
        "completion_manifest_filename": "completion_manifest.json",
        "completion_manifest_published_last": True,
    }
    if set(output) != {"output_root", *required_output.keys()}:
        raise ContractError("output_contract field closure changed")
    for field, expected in required_output.items():
        if output.get(field) != expected:
            raise ContractError(f"output_contract.{field} changed")
    if not isinstance(output.get("output_root"), str) or not Path(output["output_root"]).is_absolute():
        raise ContractError("output root must be absolute")
    if batch == "m0_lateral_teachers":
        _validate_m0_stance(plan, rows)
    else:
        if plan.get("m0_stance_contract") is not None:
            raise ContractError("S0 must not carry an M0 stance result")
        semantic = plan.get("s0_semantic_guard")
        if semantic != {
            "motion_role": "fifth_action_backhand_high_ball_forward_downward_press",
            "question_family": "separate_high_ball_high_press_paper_required_not_yet_preregistered",
            "pull_or_loop_question_paper_allowed": False,
            "observed_ball_contact": None,
            "strike_effectiveness": None,
        }:
            raise ContractError("S0 contact/effect semantic guard changed")
    verify_tool_contract(plan, repo_root)
    return plan


def _validate_blocked_runtime(runtime: dict[str, Any], repo_root: Path | None = None) -> None:
    """Validate the machine-readable negative space of an incomplete closure.

    This deliberately does not make a blocked runtime executable.  It prevents
    a partial network receipt from being mistaken for a ready plan while still
    making the exact missing reads testable and reviewable.
    """

    expected_keys = {
        "schema_version",
        "status",
        "scope",
        "tool_contract",
        "ignored_gmr_source",
        "a3_robot_contract",
        "execution_contract",
        "closure_evidence_receipt",
        "required_unresolved_evidence",
    }
    if set(runtime) != expected_keys:
        raise ContractError("blocked shared runtime field closure changed")
    tool = runtime.get("tool_contract")
    if not isinstance(tool, dict) or set(tool) != {"consumer", "result_auditor"}:
        raise ContractError("blocked runtime tool closure changed")
    require_binding(tool.get("consumer"), "blocked tool_contract.consumer")
    require_binding(tool.get("result_auditor"), "blocked tool_contract.result_auditor")
    if repo_root is not None:
        for name in ("consumer", "result_auditor"):
            verify_regular_file(
                tool[name], f"blocked tool_contract.{name}", root=repo_root
            )
    source = runtime.get("ignored_gmr_source")
    if not isinstance(source, dict):
        raise ContractError("blocked runtime ignored_gmr_source must be a mapping")
    if not isinstance(source.get("root"), str) or not Path(source["root"]).is_absolute():
        raise ContractError("blocked runtime GMR root must be absolute")
    require_git_oid(source.get("commit"), "blocked runtime GMR commit")
    require_git_oid(source.get("tree_oid"), "blocked runtime GMR tree")
    require_binding(source.get("recovery_bundle"), "blocked runtime GMR recovery bundle")
    files = source.get("runtime_files")
    if not isinstance(files, dict) or set(files) != set(REQUIRED_GMR_RUNTIME_FILES):
        raise ContractError("blocked runtime GMR file closure changed")
    for name, binding in files.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
            raise ContractError(f"blocked GMR runtime file {name} binding shape changed")
        if binding["path"] is not None and not isinstance(binding["path"], str):
            raise ContractError(f"blocked GMR runtime file {name} path is malformed")
        if binding["bytes"] is not None and (
            not isinstance(binding["bytes"], int) or binding["bytes"] <= 0
        ):
            raise ContractError(f"blocked GMR runtime file {name} bytes are malformed")
        if binding["sha256"] is not None:
            require_sha(binding["sha256"], f"blocked GMR runtime file {name} sha256")
    if source.get("retarget_joint_order") != [] or source.get("retarget_body_order") != []:
        raise ContractError("truncated retarget XML orders must remain empty while blocked")
    if source.get("retarget_foot_site_mapping") is not None:
        raise ContractError("truncated retarget XML foot sites must remain null while blocked")
    if source.get("joint_bijection_to_canonical") != []:
        raise ContractError("retarget/canonical bijection must remain empty while blocked")
    _validate_robot_contract(runtime)
    execution = runtime.get("execution_contract")
    if not isinstance(execution, dict):
        raise ContractError("blocked runtime execution_contract must be a mapping")
    python = execution.get("python_environment")
    if not isinstance(python, dict):
        raise ContractError("blocked runtime python_environment must be a mapping")
    if python.get("executable") is not None or python.get("executable_bytes") is not None:
        raise ContractError("unobserved Python path/bytes must remain null while blocked")
    require_sha(python.get("executable_sha256"), "blocked Python executable SHA")
    require_sha(python.get("pip_freeze_sha256"), "blocked Python pip-freeze SHA")
    if python.get("pip_version") is not None or python.get("xrobot_utils_resolution") is not None:
        raise ContractError("unobserved Python pip/module resolution must remain null while blocked")
    receipt = runtime.get("closure_evidence_receipt")
    if receipt != {
        "capture_path": "/private/tmp/pod_network_exam_gmr_evidence_20260714.md",
        "bytes": 5621,
        "sha256": "32c90a8882be02e5bd7260a8531f1cc0c5b212e88663c3f5e3a7a8aec13c8236",
        "audit_utc": "2026-07-13T16:08:48Z/2026-07-13T16:28:00Z",
        "read_only": True,
    }:
        raise ContractError("blocked runtime closure evidence receipt changed")
    unresolved = runtime.get("required_unresolved_evidence")
    if not isinstance(unresolved, list) or not unresolved:
        raise ContractError("blocked runtime must enumerate unresolved exact evidence")
    pointers: list[str] = []
    for index, item in enumerate(unresolved):
        if (
            not isinstance(item, dict)
            or set(item) != {"json_pointer", "reason", "next_read_only_probe"}
            or not all(isinstance(item[field], str) and item[field] for field in item)
        ):
            raise ContractError(f"malformed required_unresolved_evidence[{index}]")
        pointers.append(item["json_pointer"])
    expected_pointers = [
        "/ignored_gmr_source/runtime_files/package_init/path",
        "/ignored_gmr_source/runtime_files/motion_retarget/path",
        "/ignored_gmr_source/runtime_files/params/path",
        "/ignored_gmr_source/runtime_files/kinematics_model/path",
        "/ignored_gmr_source/runtime_files/robot_motion_viewer",
        "/ignored_gmr_source/runtime_files/data_loader",
        "/ignored_gmr_source/runtime_files/neck_retarget",
        "/ignored_gmr_source/runtime_files/smplx_to_a3_mapping/path",
        "/ignored_gmr_source/retarget_joint_order",
        "/ignored_gmr_source/retarget_body_order",
        "/ignored_gmr_source/retarget_foot_site_mapping",
        "/ignored_gmr_source/joint_bijection_to_canonical",
        "/execution_contract/python_environment/executable",
        "/execution_contract/python_environment/executable_bytes",
        "/execution_contract/python_environment/pip_version",
        "/execution_contract/python_environment/xrobot_utils_resolution",
    ]
    if pointers != expected_pointers or len(pointers) != len(set(pointers)):
        raise ContractError("blocked runtime unresolved evidence list changed")


def verify_tool_contract(plan: dict[str, Any], repo_root: Path) -> None:
    paths = {
        "consumer": Path(__file__).resolve(),
        "result_auditor": (repo_root / "scripts" / "audit_gmr_result.py").resolve(),
    }
    for name, path in paths.items():
        binding = plan["tool_contract"][name]
        if Path(binding["path"]).name != path.name:
            raise ContractError(f"tool_contract.{name} basename mismatch")
        verify_regular_file(binding, f"tool_contract.{name}", root=repo_root)


def verify_tree_contract(plan: dict[str, Any], repo_root: Path) -> Path:
    robot = plan["a3_robot_contract"]
    mjcf = verify_regular_file(robot["canonical_mjcf"], "canonical A3 MJCF", root=repo_root)
    tree = robot["canonical_model_tree"]
    root = verify_real_directory(repo_root / tree["root"], "canonical A3 model tree")
    try:
        root.relative_to(repo_root.resolve())
    except ValueError:
        raise ContractError("canonical A3 model tree must remain inside the repository") from None
    actual = tree_fingerprint(root)
    expected = {key: tree[key] for key in ("algorithm", "file_count", "total_bytes", "manifest_sha256")}
    if actual != expected:
        raise ContractError(f"canonical A3 model tree mismatch: actual={actual}, expected={expected}")
    if mjcf.parent != root:
        raise ContractError("canonical A3 MJCF must be inside the bound model tree root")
    return mjcf


def _xml_names_and_sites(path: Path) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ContractError(f"cannot parse A3 MJCF {path}: {exc}") from None
    joints = [
        node.get("name")
        for node in root.iter("joint")
        if node.get("type", "hinge") == "hinge" and node.get("name")
    ]
    bodies = [node.get("name") for node in root.iter("body") if node.get("name")]
    sites: dict[str, dict[str, Any]] = {}
    for body in root.iter("body"):
        body_name = body.get("name")
        for site in body.findall("site"):
            name = site.get("name")
            if name:
                sites[name] = {
                    "parent_body": body_name,
                    "local_pos_m": [float(value) for value in (site.get("pos") or "0 0 0").split()],
                }
    return joints, bodies, sites


def verify_a3_orders_and_sites(plan: dict[str, Any], repo_root: Path, canonical_mjcf: Path) -> None:
    robot = plan["a3_robot_contract"]
    order_source = verify_regular_file(robot["joint_order_source"], "joint order source", root=repo_root)
    text_names = [
        line.strip()[2:]
        for line in order_source.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- ")
    ]
    if text_names != robot["joint_order"]:
        raise ContractError("tracked YAML joint order does not match preregistration")
    joints, bodies, sites = _xml_names_and_sites(canonical_mjcf)
    if joints != robot["joint_order"] or bodies != robot["body_order"]:
        raise ContractError("canonical A3 MJCF joint/body order mismatch")
    for side in ("left", "right"):
        expected = robot["foot_site_mapping"][side]
        actual = sites.get(expected["site"])
        if actual != {"parent_body": expected["parent_body"], "local_pos_m": expected["local_pos_m"]}:
            raise ContractError(f"canonical A3 {side} foot site mismatch")


def _run_git(root: Path, args: list[str], label: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ContractError(f"cannot {label}: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_gmr_source(plan: dict[str, Any]) -> dict[str, Path]:
    source = plan["ignored_gmr_source"]
    root = verify_real_directory(Path(source["root"]), "ignored GMR root")
    if _run_git(root, ["rev-parse", "HEAD"], "resolve GMR HEAD") != source["commit"]:
        raise ContractError("ignored GMR HEAD mismatch")
    if _run_git(root, ["rev-parse", "HEAD^{tree}"], "resolve GMR tree") != source["tree_oid"]:
        raise ContractError("ignored GMR source tree mismatch")
    status = _run_git(root, ["status", "--porcelain", "--untracked-files=all"], "inspect GMR status")
    if status:
        raise ContractError(f"ignored GMR worktree is not clean: {status.splitlines()[:10]}")
    bundle = verify_regular_file(source["recovery_bundle"], "GMR recovery bundle")
    verify = subprocess.run(
        ["git", "-C", str(root), "bundle", "verify", str(bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0:
        raise ContractError(f"GMR recovery bundle verification failed: {verify.stderr.strip()}")
    heads = _run_git(root, ["bundle", "list-heads", str(bundle)], "list GMR bundle heads")
    if not any(line.split(maxsplit=1)[0] == source["commit"] for line in heads.splitlines() if line):
        raise ContractError("GMR recovery bundle does not advertise the bound commit")
    paths: dict[str, Path] = {"root": root}
    for name, binding in source["runtime_files"].items():
        path = verify_regular_file(binding, f"GMR runtime {name}")
        try:
            path.relative_to(root)
        except ValueError:
            raise ContractError(f"GMR runtime {name} is outside clean source root") from None
        paths[name] = path
    converter_expected = root / "scripts" / "gvhmr_to_robot.py"
    if paths["converter"] != converter_expected:
        raise ContractError("bound converter is not clean-root scripts/gvhmr_to_robot.py")
    retarget_joints, retarget_bodies, retarget_sites = _xml_names_and_sites(
        paths["a3_retarget_mjcf"]
    )
    if retarget_joints != source["retarget_joint_order"]:
        raise ContractError("GMR A3 retarget MJCF joint order mismatch")
    if retarget_bodies != source["retarget_body_order"]:
        raise ContractError("GMR A3 retarget MJCF body order mismatch")
    for side in ("left", "right"):
        expected = source["retarget_foot_site_mapping"][side]
        actual = retarget_sites.get(expected["site"])
        if actual != {
            "parent_body": expected["parent_body"],
            "local_pos_m": expected["local_pos_m"],
        }:
            raise ContractError(f"GMR A3 retarget {side} foot site mismatch")
    return paths


def python_fingerprint(python: Path) -> dict[str, Any]:
    version = subprocess.run([str(python), "--version"], capture_output=True, text=True, check=False)
    if version.returncode != 0:
        raise ContractError(f"motion Python --version failed: {version.stderr.strip()}")
    pip_version = subprocess.run(
        [str(python), "-m", "pip", "--version"], capture_output=True, text=True, check=False
    )
    if pip_version.returncode != 0:
        raise ContractError(f"motion Python pip --version failed: {pip_version.stderr.strip()}")
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if freeze.returncode != 0:
        raise ContractError(f"motion Python pip freeze failed: {freeze.stderr.strip()}")
    spec_probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.util,json; s=importlib.util.find_spec('xrobot_utils'); "
                "print(json.dumps({'found':s is not None,'origin':None if s is None else s.origin}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if spec_probe.returncode != 0:
        raise ContractError(f"xrobot_utils resolution probe failed: {spec_probe.stderr.strip()}")
    try:
        spec = json.loads(spec_probe.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"xrobot_utils resolution probe returned invalid JSON: {exc}") from None
    if not isinstance(spec, dict):
        raise ContractError("xrobot_utils resolution probe JSON must be a mapping")
    if spec == {"found": False, "origin": None}:
        xrobot = {"status": "absent", "origin": None, "bytes": None, "sha256": None}
    elif spec.get("found") is True and isinstance(spec.get("origin"), str):
        origin = Path(spec["origin"])
        if origin.is_symlink() or not origin.is_file():
            raise ContractError(f"xrobot_utils origin must be a regular non-symlink file: {origin}")
        xrobot = {
            "status": "regular_file",
            "origin": str(origin),
            "bytes": origin.stat().st_size,
            "sha256": sha256_file(origin),
        }
    else:
        raise ContractError(f"xrobot_utils namespace/unknown resolution is unsupported: {spec}")
    normalized = normalized_pip_freeze_bytes(freeze.stdout)
    return {
        "executable": str(python),
        "executable_bytes": python.stat().st_size,
        "executable_sha256": sha256_file(python),
        "version": (version.stdout or version.stderr).strip(),
        "pip_version": (pip_version.stdout or pip_version.stderr).strip(),
        "pip_freeze_command": "python -m pip freeze --all",
        "pip_freeze_normalization": "strip nonempty lines; bytewise sort; join LF; append one LF; sha256",
        "pip_freeze_sha256": hashlib.sha256(normalized).hexdigest(),
        "xrobot_utils_resolution": xrobot,
    }


def verify_python(plan: dict[str, Any], gmr_root: Path) -> Path:
    execution = plan["execution_contract"]
    expected = execution["python_environment"]
    python = Path(expected["executable"])
    if python.is_symlink() or not python.is_file():
        raise ContractError(f"motion Python must be an exact non-symlink executable: {python}")
    actual = python_fingerprint(python)
    if actual != expected:
        raise ContractError(f"motion Python fingerprint mismatch: actual={actual}, expected={expected}")
    code = "; ".join(f"import {name}" for name in execution["required_imports"])
    env = build_environment(plan, gmr_root)
    probe = subprocess.run([str(python), "-c", code], env=env, capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        raise ContractError(f"motion Python import closure failed: {probe.stderr.strip()}")
    return python


def verify_materialization(plan: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    source = plan["source_materialization"]
    verify_regular_file(source["preregistration"], "canonical-beta preregistration", root=repo_root)
    completion_path = verify_regular_file(source["completion_manifest"], "canonical-beta completion")
    verify_regular_file(source["canonical_betas_artifact"], "canonical-beta artifact")
    completion = read_json(completion_path, "canonical-beta completion")
    expected_flags = {
        "status": MATERIALIZATION_STATUS,
        "body_shape_contract": BODY_SHAPE_CONTRACT,
        "formal_eligible": False,
        "training_authorized": False,
        "hardware_authorized": False,
    }
    for field, expected in expected_flags.items():
        if completion.get(field) != expected:
            raise ContractError(f"canonical-beta completion {field} mismatch")
    if completion.get("next_gate") != {
        "authorized": "separate_exact_gmr_preregistration_only",
        "status": "blocked_until_exact_gmr_plan_and_runtime_are_bound",
    }:
        raise ContractError("canonical-beta completion next-gate contract mismatch")
    plan_binding = completion.get("plan")
    if not isinstance(plan_binding, dict) or plan_binding.get("sha256") != source["preregistration"]["sha256"]:
        raise ContractError("canonical-beta completion preregistration SHA mismatch")
    donor = completion.get("canonical_beta_donor")
    if not isinstance(donor, dict) or donor.get("vector_sha256") != source["canonical_vector_sha256"]:
        raise ContractError("canonical-beta donor vector mismatch")
    rows = completion.get("results")
    if not isinstance(rows, list):
        raise ContractError("canonical-beta completion results must be a list")
    by_id = {row.get("asset_id"): row for row in rows if isinstance(row, dict)}
    if len(by_id) != len(rows):
        raise ContractError("canonical-beta completion has malformed/duplicate asset ids")
    verified: list[dict[str, Any]] = []
    for expected in plan["inputs"]:
        asset_id = expected["asset_id"]
        row = by_id.get(asset_id)
        if not isinstance(row, dict):
            raise ContractError(f"canonical-beta completion lacks {asset_id}")
        binding = expected["input"]
        exact = {
            "frames": expected["frames"],
            "output_path": binding["path"],
            "output_bytes": binding["bytes"],
            "output_sha256": binding["sha256"],
            "output_canonical_vector_sha256": expected["canonical_vector_sha256"],
            "non_beta_bit_exact": True,
            "ready_before_window_s": expected["ready_before_window_s"],
            "ready_after_window_s": expected["ready_after_window_s"],
        }
        for field, value in exact.items():
            if row.get(field) != value:
                raise ContractError(f"canonical-beta {asset_id}.{field} mismatch")
        input_path = verify_regular_file(binding, f"canonical-beta input {asset_id}")
        verified.append({**expected, "input_path": str(input_path)})
    if set(by_id) != set(EXPECTED_BATCHES[plan["batch_kind"]]):
        raise ContractError("canonical-beta completion contains unexpected assets")
    return verified


def build_environment(plan: dict[str, Any], gmr_root: Path) -> dict[str, str]:
    execution = plan["execution_contract"]
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(gmr_root),
            "OMP_NUM_THREADS": str(execution["OMP_NUM_THREADS"]),
            "MKL_NUM_THREADS": str(execution["MKL_NUM_THREADS"]),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def inspect_plan(plan: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    canonical_mjcf = verify_tree_contract(plan, repo_root)
    verify_a3_orders_and_sites(plan, repo_root, canonical_mjcf)
    gmr = verify_gmr_source(plan)
    python = verify_python(plan, gmr["root"])
    rows = verify_materialization(plan, repo_root)
    output_root = Path(plan["output_contract"]["output_root"])
    if output_root.exists():
        raise ContractError(f"no-clobber output root already exists: {output_root}")
    return {
        "canonical_mjcf": canonical_mjcf,
        "gmr": gmr,
        "python": python,
        "rows": rows,
        "output_root": output_root,
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build_converter_command(plan: dict[str, Any], python: Path, converter: Path, source: Path, output: Path) -> list[str]:
    replacements = {
        "{python}": str(python),
        "{converter}": str(converter),
        "{input}": str(source),
        "{output}": str(output),
    }
    return [replacements.get(token, token) for token in plan["execution_contract"]["converter_argv_template"]]


def load_gmr_payload(path: Path, frames: int) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ContractError(f"cannot load GMR result {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise ContractError("GMR output root must be a mapping")
    try:
        import numpy as np
    except ImportError as exc:
        raise ContractError(f"NumPy unavailable for GMR verification: {exc}") from None
    expected = {"root_pos": (frames, 3), "root_rot": (frames, 4), "dof_pos": (frames, 31)}
    arrays: dict[str, Any] = {}
    for name, shape in expected.items():
        if name not in payload:
            raise ContractError(f"GMR output lacks {name}")
        value = payload[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ContractError(f"GMR {name} must be finite shape {shape}, got {array.shape}")
        arrays[name] = array
    fps = float(np.asarray(payload.get("fps")).reshape(-1)[0])
    if fps != 30.0:
        raise ContractError(f"GMR fps must be exactly 30, got {fps}")
    return {"fps": fps, **arrays}


def evaluate_stance_vectors(initial: list[float], terminal: list[float], tolerances: dict[str, float]) -> dict[str, Any]:
    if len(initial) != 2 or len(terminal) != 2:
        raise ContractError("stance vectors must be two-dimensional")
    if not all(math.isfinite(float(value)) for value in [*initial, *terminal]):
        raise ContractError("stance vectors must be finite")
    initial_x, initial_y = (float(initial[0]), float(initial[1]))
    terminal_x, terminal_y = (float(terminal[0]), float(terminal[1]))
    error_x = terminal_x - initial_x
    error_y = terminal_y - initial_y
    initial_abs_y = abs(initial_y)
    terminal_abs_y = abs(terminal_y)
    checks = {
        "minimum_initial_lateral_separation": (
            initial_abs_y >= tolerances["minimum_initial_abs_lateral_separation"]
        ),
        "fore_aft_component_preserved": (
            abs(error_x) <= tolerances["fore_aft_component_abs_error_max"]
        ),
        "lateral_component_preserved": (
            abs(error_y) <= tolerances["lateral_component_abs_error_max"]
        ),
        "lateral_sign_preserved": initial_y * terminal_y > 0.0,
        "not_narrowed": (
            terminal_abs_y + tolerances["lateral_narrowing_max"] >= initial_abs_y
        ),
    }
    return {
        "initial_d_xy_m": [initial_x, initial_y],
        "terminal_d_xy_m": [terminal_x, terminal_y],
        "fore_aft_stagger_initial_m": initial_x,
        "fore_aft_stagger_terminal_m": terminal_x,
        "lateral_separation_initial_signed_m": initial_y,
        "lateral_separation_terminal_signed_m": terminal_y,
        "lateral_separation_initial_abs_m": initial_abs_y,
        "lateral_separation_terminal_abs_m": terminal_abs_y,
        "component_error_terminal_minus_initial_m": [error_x, error_y],
        "checks": checks,
        "stance_passed": all(checks.values()),
    }


def reorder_dof_row_to_canonical(row: Any, bijection: list[dict[str, Any]]) -> list[float]:
    if len(row) != 31 or len(bijection) != 31:
        raise ContractError("GMR dof row and bijection must each contain 31 entries")
    output: list[float | None] = [None] * 31
    for mapping in bijection:
        gmr_index = mapping["gmr_dof_index"]
        canonical_index = mapping["canonical_qpos_index"] - 7
        if not 0 <= gmr_index < 31 or not 0 <= canonical_index < 31:
            raise ContractError("GMR/canonical bijection index is out of range")
        if output[canonical_index] is not None:
            raise ContractError("GMR/canonical bijection repeats a canonical index")
        value = float(row[gmr_index])
        if not math.isfinite(value):
            raise ContractError("GMR dof row contains a non-finite value")
        output[canonical_index] = value
    if any(value is None for value in output):
        raise ContractError("GMR/canonical bijection does not cover every canonical index")
    return [float(value) for value in output]


def compute_m0_stance(plan: dict[str, Any], row: dict[str, Any], payload: dict[str, Any], mjcf: Path) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise ContractError(f"MuJoCo/NumPy unavailable for M0 foot FK: {exc}") from None
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    robot = plan["a3_robot_contract"]
    site_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot["foot_site_mapping"][side]["site"])
        for side in ("left", "right")
    }
    if any(site_id < 0 for site_id in site_ids.values()):
        raise ContractError("bound A3 foot site is absent from compiled MJCF")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    if pelvis_id < 0 or model.nq != 38:
        raise ContractError(f"compiled A3 model must have pelvis and nq=38, got nq={model.nq}")
    left_xy: list[Any] = []
    right_xy: list[Any] = []
    source_heading = None
    frame0_root_xy = payload["root_pos"][0, :2].copy()
    for index in range(row["frames"]):
        data.qpos[:3] = payload["root_pos"][index]
        x, y, z, w = payload["root_rot"][index]
        data.qpos[3:7] = [w, x, y, z]
        data.qpos[7:38] = reorder_dof_row_to_canonical(
            payload["dof_pos"][index], plan["ignored_gmr_source"]["joint_bijection_to_canonical"]
        )
        mujoco.mj_forward(model, data)
        if index == 0:
            forward = data.xmat[pelvis_id].reshape(3, 3)[:, 0]
            norm = float(np.linalg.norm(forward[:2]))
            if norm <= 1e-9:
                raise ContractError("frame-0 pelvis heading projection is degenerate")
            source_heading = math.atan2(float(forward[1]), float(forward[0]))
        left_xy.append(data.site_xpos[site_ids["left"]][:2].copy())
        right_xy.append(data.site_xpos[site_ids["right"]][:2].copy())
    assert source_heading is not None
    c, s = math.cos(-source_heading), math.sin(-source_heading)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    left_norm = (np.asarray(left_xy) - frame0_root_xy) @ rotation.T
    right_norm = (np.asarray(right_xy) - frame0_root_xy) @ rotation.T
    d_xy = right_norm - left_norm
    mapping = next(
        item
        for item in plan["m0_stance_contract"]["ready_window_sample_mappings"]
        if item["asset_id"] == row["asset_id"]
    )
    before = mapping["ready_before"]["indices"]
    after = mapping["ready_after"]["indices"]
    initial = np.median(d_xy[before], axis=0).tolist()
    terminal = np.median(d_xy[after], axis=0).tolist()
    result = evaluate_stance_vectors(
        initial, terminal, plan["m0_stance_contract"]["preregistered_tolerances_m"]
    )
    result.update(
        {
            "foot_site_mapping": robot["foot_site_mapping"],
            "normalization": plan["m0_stance_contract"]["normalization"],
            "source_frame0_heading_yaw_rad": source_heading,
            "frame0_common_root_xy_m": frame0_root_xy.tolist(),
            "ready_window_sample_mapping": mapping,
            "vector_definition": "right_foot_xy-left_foot_xy in heading-aligned +X forward/+Y left",
            "preregistered_tolerances_m": plan["m0_stance_contract"]["preregistered_tolerances_m"],
        }
    )
    return result


def run_auditor(
    plan: dict[str, Any], python: Path, auditor: Path, output: Path, log: Path, audit: Path, frames: int, env: dict[str, str]
) -> dict[str, Any]:
    execution = plan["execution_contract"]
    command = [
        str(python),
        str(auditor),
        "--result",
        str(output),
        "--expected-frames",
        str(frames),
        "--run-log",
        str(log),
        "--warmup-threshold",
        str(execution["warmup_threshold_strict_lt"]),
        "--warmup-max-rounds",
        str(execution["warmup_max_rounds"]),
        "--body-shape-contract",
        BODY_SHAPE_CONTRACT,
        "--json-out",
        str(audit),
    ]
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=min(float(execution["timeout_seconds_per_asset"]), 120.0),
    )
    if result.returncode != 0 or not audit.is_file():
        raise ContractError(
            f"GMR structural auditor failed rc={result.returncode}: "
            f"{(result.stdout + result.stderr)[-2000:]}"
        )
    report = read_json(audit, "GMR structural audit")
    if (
        report.get("status") != "pass"
        or report.get("result_sha256") != sha256_file(output)
        or report.get("run_log_sha256") != sha256_file(log)
        or report.get("actual_frames") != frames
        or report.get("body_shape_contract") != BODY_SHAPE_CONTRACT
        or report.get("formal_eligible") is not False
    ):
        raise ContractError("GMR structural audit content/SHA/lineage mismatch")
    return report


def consume(plan: dict[str, Any], plan_path: Path, expected_sha: str, repo_root: Path) -> Path:
    inspected = inspect_plan(plan, repo_root)
    root: Path = inspected["output_root"]
    root.mkdir(parents=True, exist_ok=False)
    for name in ("outputs", "logs", "audits", "bindings"):
        (root / name).mkdir()
    fsync_dir(root)
    gmr = inspected["gmr"]
    python: Path = inspected["python"]
    auditor = (repo_root / "scripts" / "audit_gmr_result.py").resolve()
    env = build_environment(plan, gmr["root"])
    rows_out: list[dict[str, Any]] = []
    for row in inspected["rows"]:
        asset_id = row["asset_id"]
        source = Path(row["input_path"])
        output = root / "outputs" / f"{asset_id}{plan['output_contract']['result_suffix']}"
        log = root / "logs" / f"{asset_id}.log"
        audit = root / "audits" / f"{asset_id}.json"
        binding_path = root / "bindings" / f"{asset_id}.json"
        command = build_converter_command(plan, python, gmr["converter"], source, output)
        try:
            with log.open("x", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    cwd=gmr["root"],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=float(plan["execution_contract"]["timeout_seconds_per_asset"]),
                )
            if result.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
                raise ContractError(
                    f"converter failed rc={result.returncode}, output_exists={output.is_file()}"
                )
            structural = run_auditor(
                plan, python, auditor, output, log, audit, row["frames"], env
            )
            payload = load_gmr_payload(output, row["frames"])
            stance = None
            if plan["batch_kind"] == "m0_lateral_teachers":
                stance = compute_m0_stance(plan, row, payload, inspected["canonical_mjcf"])
            result_row = {
                "asset_id": asset_id,
                "status": "complete_exact_gmr_diagnostic",
                "input": row["input"],
                "frames": row["frames"],
                "output": {
                    "path": str(output),
                    "bytes": output.stat().st_size,
                    "sha256": sha256_file(output),
                },
                "run_log": {
                    "path": str(log),
                    "bytes": log.stat().st_size,
                    "sha256": sha256_file(log),
                },
                "structural_audit": {
                    "path": str(audit),
                    "bytes": audit.stat().st_size,
                    "sha256": sha256_file(audit),
                    "warmup": structural["warmup"],
                },
                "m0_stance": stance,
                "observed_ball_contact": None,
                "strike_effectiveness": None,
                "formal_eligible": False,
                "schema2_authorized": False,
                "training_authorized": False,
                "hardware_authorized": False,
                "command": command,
            }
            write_json_exclusive(binding_path, result_row)
            for durable in (output, log, audit, binding_path):
                fsync_file(durable)
            rows_out.append(result_row)
        except (OSError, subprocess.SubprocessError, ContractError) as exc:
            if not binding_path.exists():
                write_json_exclusive(
                    binding_path,
                    {
                        "schema_version": 1,
                        "asset_id": asset_id,
                        "status": "failed_preserved_no_completion_manifest",
                        "error": str(exc),
                        "command": command,
                        "formal_eligible": False,
                        "training_authorized": False,
                        "hardware_authorized": False,
                    },
                )
            fsync_dir(root / "bindings")
            raise
    for name in ("outputs", "logs", "audits", "bindings"):
        fsync_dir(root / name)
    fsync_dir(root)
    # Revalidate every mutable/private input and the ignored source after all
    # converter children exit.  A post-start mutation leaves a partial root and
    # never receives the report-last completion marker.
    verify_gmr_source(plan)
    verify_materialization(plan, repo_root)
    verify_tree_contract(plan, repo_root)
    expected_files = {
        f"outputs/{row['asset_id']}{plan['output_contract']['result_suffix']}"
        for row in inspected["rows"]
    }
    expected_files |= {f"logs/{row['asset_id']}.log" for row in inspected["rows"]}
    expected_files |= {f"audits/{row['asset_id']}.json" for row in inspected["rows"]}
    expected_files |= {f"bindings/{row['asset_id']}.json" for row in inspected["rows"]}
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ContractError(
            f"unexpected pre-completion output file set: actual={sorted(actual_files)}, "
            f"expected={sorted(expected_files)}"
        )
    completion = {
        "schema_version": 1,
        "status": "complete_exact_gmr_diagnostic",
        "batch_kind": plan["batch_kind"],
        "scope": "exact canonical-beta to A3 GMR plus structural audit; M0 includes preregistered foot-stance measurement only",
        "completed_utc": utc_now(),
        "plan": {"path": str(plan_path), "sha256": expected_sha},
        "source_materialization": plan["source_materialization"],
        "runtime_contract": plan["_runtime_contract_binding"],
        "ignored_gmr_source": plan["ignored_gmr_source"],
        "execution_contract": plan["execution_contract"],
        "a3_robot_contract": plan["a3_robot_contract"],
        "results": rows_out,
        "s0_semantic_guard": plan.get("s0_semantic_guard"),
        "m0_stance_contract": plan.get("m0_stance_contract"),
        "formal_eligible": False,
        "schema2_authorized": False,
        "training_authorized": False,
        "hardware_authorized": False,
        "next_gate": "separate_schema2_L0_L1_table_net_and_dynamics_preregistration_required",
    }
    completion_path = root / plan["output_contract"]["completion_manifest_filename"]
    if any(path.name == completion_path.name for path in root.iterdir()):
        raise ContractError("completion manifest target unexpectedly exists before report-last publish")
    write_json_exclusive(completion_path, completion)
    fsync_dir(root)
    return completion_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("command", choices=("static", "inspect", "consume"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        plan_path = args.plan.resolve()
        plan = validate_plan(plan_path, args.expected_plan_sha256, repo_root)
        if args.command == "static":
            print(f"PASS static {plan['batch_kind']} plan_sha256={args.expected_plan_sha256}")
            return 0
        if args.command == "inspect":
            inspected = inspect_plan(plan, repo_root)
            print(
                f"PASS inspect {plan['batch_kind']} inputs={len(inspected['rows'])} "
                f"output_root_absent={not inspected['output_root'].exists()}"
            )
            return 0
        completion = consume(plan, plan_path, args.expected_plan_sha256, repo_root)
        print(f"PASS consume {plan['batch_kind']} completion={completion}")
        return 0
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL exact-gmr: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
