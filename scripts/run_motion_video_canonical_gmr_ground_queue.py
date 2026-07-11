#!/usr/bin/env python3
"""Ground the ten canonical-beta GMR pickles through one bound CPU-only queue.

The queue is intentionally narrower than a generic directory launcher.  It
binds one preregistration, the tracked canonical-beta GMR completion manifest,
every physical input pickle, the exact single-file grounding tool, and the
canonical A3 MuJoCo model.  Output and state roots must both be absent before
launch; a failure is preserved in place and requires a new preregistered root.

Each accepted output is produced by ``ground_gmr_pkl.py`` and therefore changes
only ``root_pos[:, 2]`` by one fixed translation.  This is a discrete-frame
grounding diagnostic, not a continuous-time, dynamics, training, or robot gate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BODY_SHAPE_CONTRACT = "diagnostic_same_performer_coordinatewise_median_betas_v1"
SAFE_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class QueueError(ValueError):
    """The canonical grounding queue contract cannot be satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a JSON mapping: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise QueueError(f"{label} must be a lowercase SHA-256")
    return value


def require_binding(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a mapping")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise QueueError(f"{label}.path must be a non-empty string")
    require_sha(value.get("sha256"), f"{label}.sha256")
    if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        raise QueueError(f"{label}.bytes must be a positive integer")
    return value


def verify_bound_file(binding: dict[str, Any], label: str) -> Path:
    path = Path(binding["path"]).expanduser().resolve()
    if not path.is_file():
        raise QueueError(f"{label} is missing: {path}")
    size = path.stat().st_size
    if size != binding["bytes"]:
        raise QueueError(f"{label} bytes {size} != {binding['bytes']}: {path}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise QueueError(f"{label} sha256 {actual} != {binding['sha256']}: {path}")
    return path


def validate_plan(plan_path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    require_sha(expected_plan_sha256, "--expected-plan-sha256")
    actual = sha256_file(plan_path)
    if actual != expected_plan_sha256:
        raise QueueError(f"plan sha256 {actual} != expected {expected_plan_sha256}")
    plan = read_json(plan_path, "canonical grounding plan")
    if plan.get("schema_version") != 1 or plan.get("status") != "preregistered_not_executed":
        raise QueueError("plan must be schema 1 and preregistered_not_executed")
    if plan.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise QueueError(f"body_shape_contract must be {BODY_SHAPE_CONTRACT}")
    if plan.get("formal_eligible") is not False:
        raise QueueError("plan must explicitly remain formal_eligible=false")
    if plan.get("cpu_only") is not True or plan.get("CUDA_VISIBLE_DEVICES") != "":
        raise QueueError("plan must be CPU-only with CUDA_VISIBLE_DEVICES empty")
    for field in ("canonical_gmr_result", "grounding_tool", "queue_tool", "mjcf"):
        require_binding(plan.get(field), field)

    collision = plan.get("compiled_collision_contract")
    if not isinstance(collision, dict):
        raise QueueError("compiled_collision_contract must be a mapping")
    require_sha(collision.get("expected_sha256"), "compiled_collision_contract.expected_sha256")
    if collision.get("ground_geom") != "floor" or collision.get("ground_z_m") != 0.0:
        raise QueueError("compiled collision contract must bind the horizontal floor at z=0")
    if collision.get("enabled_robot_geom_count") != 37:
        raise QueueError("compiled collision contract must bind 37 enabled robot geoms")
    geom_ids = collision.get("enabled_robot_geom_ids")
    if (
        not isinstance(geom_ids, list)
        or len(geom_ids) != collision["enabled_robot_geom_count"]
        or not all(isinstance(value, int) and value >= 0 for value in geom_ids)
        or len(set(geom_ids)) != len(geom_ids)
        or geom_ids != sorted(geom_ids)
    ):
        raise QueueError("compiled collision contract must bind 37 unique sorted geom ids")
    if collision.get("robot_root_body_id") != 1:
        raise QueueError("compiled collision contract must bind robot root body id 1")
    if collision.get("surface_method") != "analytic_primitive_support_or_compiled_mesh_vertices":
        raise QueueError("compiled collision surface method mismatch")
    if collision.get("visual_only_geoms_excluded") is not True:
        raise QueueError("visual-only geoms must be excluded")

    grounding = plan.get("grounding_contract")
    expected_grounding = {
        "expected_fps": 30.0,
        "target_clearance_m": 1e-5,
        "max_grounded_clearance_m": 1e-3,
        "numerical_tolerance_m": 5e-7,
        "max_abs_shift_m": 0.25,
        "quaternion_norm_tolerance": 1e-6,
        "joint_range_tolerance_rad": 1e-5,
        "translation": "one_constant_value_added_only_to_root_pos[:,2]",
        "clearance_sampling": "original_discrete_frames_only",
        "continuous_time_clearance_proven": False,
    }
    if grounding != expected_grounding:
        raise QueueError("grounding_contract does not match the frozen diagnostic contract")

    output = plan.get("output_contract")
    if not isinstance(output, dict):
        raise QueueError("output_contract must be a mapping")
    required_output = {
        "output_suffix": ".diagnostic_cohort_median_betas.grounded.pkl",
        "report_suffix": ".grounding.json",
        "output_root_must_not_exist": True,
        "state_root_must_not_exist": True,
        "no_clobber": True,
        "stop_on_first_failure": True,
    }
    if not isinstance(output.get("output_root"), str) or not isinstance(output.get("state_root"), str):
        raise QueueError("output_contract roots must be strings")
    for field, expected in required_output.items():
        if output.get(field) != expected:
            raise QueueError(f"output_contract.{field} must be {expected!r}")

    python = plan.get("python_environment")
    if not isinstance(python, dict) or not isinstance(python.get("executable"), str):
        raise QueueError("python_environment.executable must be a string")
    if not isinstance(python.get("version"), str) or not python["version"].startswith("Python 3."):
        raise QueueError("python_environment.version must bind Python 3")

    checkouts = plan.get("read_only_checkout_contracts")
    if not isinstance(checkouts, list) or len(checkouts) != 2:
        raise QueueError("read_only_checkout_contracts must bind exactly training and GMR")
    labels: set[str] = set()
    for index, checkout in enumerate(checkouts):
        if not isinstance(checkout, dict):
            raise QueueError(f"read_only_checkout_contracts[{index}] must be a mapping")
        label = checkout.get("label")
        if label not in {"training", "gmr"} or label in labels:
            raise QueueError("read-only checkout labels must be unique training/gmr")
        labels.add(label)
        if not isinstance(checkout.get("root"), str):
            raise QueueError(f"read-only checkout {label} root must be a string")
        if not isinstance(checkout.get("head"), str) or not re.fullmatch(r"[0-9a-f]{40}", checkout["head"]):
            raise QueueError(f"read-only checkout {label} head must be a git commit")
        if checkout.get("must_be_clean_before_and_after") is not True:
            raise QueueError(f"read-only checkout {label} must remain clean before and after")

    rows = plan.get("inputs")
    order = plan.get("processing_order")
    if not isinstance(rows, list) or len(rows) != 10 or not isinstance(order, list):
        raise QueueError("plan must contain exactly ten ordered inputs")
    ids: list[str] = []
    paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise QueueError(f"inputs[{index}] must be a mapping")
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not SAFE_ASSET_ID.fullmatch(asset_id):
            raise QueueError(f"inputs[{index}].asset_id is unsafe: {asset_id!r}")
        if asset_id in ids:
            raise QueueError(f"duplicate asset_id {asset_id}")
        ids.append(asset_id)
        binding = require_binding(row.get("input"), f"inputs[{index}].input")
        if binding["path"] in paths:
            raise QueueError(f"duplicate input path {binding['path']}")
        paths.add(binding["path"])
        if not isinstance(row.get("frames"), int) or row["frames"] <= 1:
            raise QueueError(f"inputs[{index}].frames must be an integer > 1")
    if order != ids:
        raise QueueError("processing_order must exactly match input row order")
    return plan


def verify_source_manifest(plan: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = verify_bound_file(plan["canonical_gmr_result"], "canonical GMR result")
    manifest = read_json(manifest_path, "canonical GMR result")
    if manifest.get("status") != "complete_diagnostic_canonical_gmr":
        raise QueueError("canonical GMR result is not complete")
    if manifest.get("body_shape_contract") != BODY_SHAPE_CONTRACT:
        raise QueueError("canonical GMR result body-shape contract mismatch")
    if manifest.get("formal_eligible") is not False:
        raise QueueError("canonical GMR result must remain formal-ineligible")
    result_rows = manifest.get("results")
    if not isinstance(result_rows, list):
        raise QueueError("canonical GMR result rows are missing")
    by_id = {row.get("asset_id"): row for row in result_rows if isinstance(row, dict)}
    if len(by_id) != 10 or len(by_id) != len(result_rows):
        raise QueueError("canonical GMR result asset set must contain ten unique rows")

    verified: list[dict[str, Any]] = []
    for row in plan["inputs"]:
        source = by_id.get(row["asset_id"])
        if not isinstance(source, dict):
            raise QueueError(f"canonical GMR result lacks {row['asset_id']}")
        expected = {
            "output_path": row["input"]["path"],
            "output_bytes": row["input"]["bytes"],
            "output_sha256": row["input"]["sha256"],
            "frames": row["frames"],
        }
        for field, value in expected.items():
            if source.get(field) != value:
                raise QueueError(f"canonical GMR {row['asset_id']}.{field} mismatch")
        physical = verify_bound_file(row["input"], f"canonical GMR input {row['asset_id']}")
        verified.append({**row, "input_path": str(physical)})
    if set(by_id) != {row["asset_id"] for row in verified}:
        raise QueueError("canonical GMR result contains an unexpected asset set")
    return verified


def verify_tools_and_runtime(plan: dict[str, Any]) -> tuple[Path, Path, Path]:
    queue_path = Path(__file__).resolve()
    ground_path = queue_path.with_name("ground_gmr_pkl.py")
    if Path(plan["queue_tool"]["path"]).name != queue_path.name:
        raise QueueError("queue_tool basename mismatch")
    if Path(plan["grounding_tool"]["path"]).name != ground_path.name:
        raise QueueError("grounding_tool basename mismatch")
    for path, binding, label in (
        (queue_path, plan["queue_tool"], "queue tool"),
        (ground_path, plan["grounding_tool"], "grounding tool"),
    ):
        if not path.is_file() or path.stat().st_size != binding["bytes"]:
            raise QueueError(f"{label} byte mismatch: {path}")
        if sha256_file(path) != binding["sha256"]:
            raise QueueError(f"{label} SHA mismatch: {path}")
    mjcf = verify_bound_file(plan["mjcf"], "canonical MJCF")
    # Preserve the venv launcher path.  Resolving its symlink to /usr/bin/python
    # drops pyvenv.cfg discovery and silently executes outside the bound venv.
    python = Path(plan["python_environment"]["executable"]).expanduser().absolute()
    if not python.is_file():
        raise QueueError(f"bound Python executable is missing: {python}")
    version = subprocess.run(
        [str(python), "--version"], capture_output=True, text=True, check=False, timeout=10
    )
    actual_version = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or actual_version != plan["python_environment"]["version"]:
        raise QueueError(
            f"Python version {actual_version!r} != {plan['python_environment']['version']!r}"
        )
    return ground_path, mjcf, python


def verify_read_only_checkouts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for contract in plan["read_only_checkout_contracts"]:
        root = Path(contract["root"]).resolve()
        if not (root / ".git").exists():
            raise QueueError(f"bound {contract['label']} checkout is missing: {root}")
        head_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        head = head_result.stdout.strip()
        dirty_lines = [line for line in status_result.stdout.splitlines() if line.strip()]
        if head_result.returncode != 0 or head != contract["head"]:
            raise QueueError(
                f"{contract['label']} HEAD {head!r} != bound {contract['head']}"
            )
        if status_result.returncode != 0 or dirty_lines:
            raise QueueError(f"{contract['label']} checkout is not clean: {dirty_lines[:5]}")
        observations.append(
            {
                "label": contract["label"],
                "root": str(root),
                "head": head,
                "clean": True,
            }
        )
    return observations


def validate_report(
    report: dict[str, Any],
    *,
    row: dict[str, Any],
    output: Path,
    ground_path: Path,
    mjcf: Path,
    plan: dict[str, Any],
) -> None:
    collision = plan["compiled_collision_contract"]
    if report.get("status") != "pass" or report.get("formal_eligible") is not False:
        raise QueueError(f"{row['asset_id']} grounding report is not a diagnostic pass")
    required = {
        "input": (row["input"]["sha256"], row["input"]["bytes"]),
        "output": (sha256_file(output), output.stat().st_size),
        "mjcf": (plan["mjcf"]["sha256"], plan["mjcf"]["bytes"]),
        "tool": (plan["grounding_tool"]["sha256"], plan["grounding_tool"]["bytes"]),
    }
    for field, (expected_sha, expected_bytes) in required.items():
        binding = report.get(field)
        if not isinstance(binding, dict):
            raise QueueError(f"{row['asset_id']} report lacks {field} binding")
        if binding.get("sha256") != expected_sha:
            raise QueueError(f"{row['asset_id']} report {field} SHA mismatch")
        if field != "tool" and binding.get("bytes") != expected_bytes:
            raise QueueError(f"{row['asset_id']} report {field} byte mismatch")
    if Path(report["tool"]["path"]).resolve() != ground_path:
        raise QueueError(f"{row['asset_id']} report tool path mismatch")
    if Path(report["mjcf"]["path"]).resolve() != mjcf:
        raise QueueError(f"{row['asset_id']} report MJCF path mismatch")
    if report["mjcf"].get("compiled_kinematic_collision_sha256") != collision["expected_sha256"]:
        raise QueueError(f"{row['asset_id']} compiled collision SHA mismatch")
    collision_report = report.get("collision_contract")
    if not isinstance(collision_report, dict):
        raise QueueError(f"{row['asset_id']} collision contract missing")
    for field in (
        "robot_root_body_id",
        "enabled_robot_geom_count",
        "enabled_robot_geom_ids",
        "surface_method",
        "visual_only_geoms_excluded",
    ):
        expected = collision[field]
        if collision_report.get(field) != expected:
            raise QueueError(f"{row['asset_id']} collision contract {field} mismatch")
    structure = report.get("structure")
    if not isinstance(structure, dict) or structure.get("frames") != row["frames"]:
        raise QueueError(f"{row['asset_id']} frame count mismatch")
    if structure.get("fps") != plan["grounding_contract"]["expected_fps"]:
        raise QueueError(f"{row['asset_id']} fps mismatch")
    invariants = report.get("invariants")
    if not isinstance(invariants, dict):
        raise QueueError(f"{row['asset_id']} invariants missing")
    for field in (
        "root_xy_exact",
        "root_rotation_exact",
        "dof_position_exact",
        "root_pos_dtype_preserved",
        "all_other_payload_fields_shallow_preserved",
    ):
        if invariants.get(field) is not True:
            raise QueueError(f"{row['asset_id']} invariant {field} did not pass")
    if invariants.get("root_z_relative_trajectory_max_error_m", math.inf) > plan["grounding_contract"]["numerical_tolerance_m"]:
        raise QueueError(f"{row['asset_id']} root-z relative trajectory changed")
    grounding = report.get("grounding")
    if not isinstance(grounding, dict):
        raise QueueError(f"{row['asset_id']} grounding values missing")
    applied_min = grounding.get("applied_root_z_shift_min_m")
    applied_max = grounding.get("applied_root_z_shift_max_m")
    if not isinstance(applied_min, (int, float)) or not isinstance(applied_max, (int, float)):
        raise QueueError(f"{row['asset_id']} applied shift is non-numeric")
    if not math.isfinite(applied_min) or not math.isfinite(applied_max):
        raise QueueError(f"{row['asset_id']} applied shift is non-finite")
    if abs(applied_max - applied_min) > plan["grounding_contract"]["numerical_tolerance_m"]:
        raise QueueError(f"{row['asset_id']} shift is not constant")
    after_min = grounding.get("after", {}).get("minimum_clearance_m")
    target = plan["grounding_contract"]["target_clearance_m"]
    ceiling = plan["grounding_contract"]["max_grounded_clearance_m"]
    tolerance = plan["grounding_contract"]["numerical_tolerance_m"]
    if not isinstance(after_min, (int, float)) or not math.isfinite(after_min):
        raise QueueError(f"{row['asset_id']} grounded clearance is non-finite")
    if after_min < target - tolerance or after_min > ceiling + tolerance:
        raise QueueError(f"{row['asset_id']} grounded clearance is outside the accepted range")


def run_one(
    row: dict[str, Any],
    *,
    output_root: Path,
    state_root: Path,
    ground_path: Path,
    mjcf: Path,
    python: Path,
    plan: dict[str, Any],
    timeout_seconds: float,
) -> bool:
    asset_id = row["asset_id"]
    output = output_root / f"{asset_id}{plan['output_contract']['output_suffix']}"
    report_path = output_root / f"{asset_id}{plan['output_contract']['report_suffix']}"
    log_path = state_root / "logs" / f"{asset_id}.log"
    binding_path = state_root / "bindings" / f"{asset_id}.json"
    for path in (output, report_path, log_path, binding_path):
        if path.exists() or path.is_symlink():
            raise QueueError(f"no-clobber target already exists for {asset_id}: {path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    grounding = plan["grounding_contract"]
    command = [
        str(python),
        str(ground_path),
        "--input", str(row["input_path"]),
        "--expected-input-sha256", row["input"]["sha256"],
        "--output", str(output),
        "--report", str(report_path),
        "--mjcf", str(mjcf),
        "--expected-mjcf-sha256", plan["mjcf"]["sha256"],
        "--ground-geom", plan["compiled_collision_contract"]["ground_geom"],
        "--expected-frames", str(row["frames"]),
        "--expected-fps", str(grounding["expected_fps"]),
        "--target-clearance-m", str(grounding["target_clearance_m"]),
        "--max-grounded-clearance-m", str(grounding["max_grounded_clearance_m"]),
        "--numerical-tolerance-m", str(grounding["numerical_tolerance_m"]),
        "--max-abs-shift-m", str(grounding["max_abs_shift_m"]),
        "--quaternion-norm-tolerance", str(grounding["quaternion_norm_tolerance"]),
        "--joint-range-tolerance-rad", str(grounding["joint_range_tolerance_rad"]),
    ]
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "asset_id": asset_id,
        "body_shape_contract": BODY_SHAPE_CONTRACT,
        "formal_eligible": False,
        "input": row["input"],
        "frames": row["frames"],
        "output_path": str(output),
        "report_path": str(report_path),
        "log_path": str(log_path),
        "command": command,
        "started_utc": utc_now(),
    }
    with binding_path.open("x", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"[canonical-ground] START {asset_id}", flush=True)
    try:
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        state.update(status="failed", completed_utc=utc_now(), failure=str(exc))
        atomic_json(binding_path, state)
        return False
    state.update(returncode=result.returncode, completed_utc=utc_now())
    if result.returncode != 0 or not output.is_file() or not report_path.is_file():
        state.update(
            status="failed",
            failure=(
                f"returncode={result.returncode}, output={output.is_file()}, "
                f"report={report_path.is_file()}"
            ),
        )
        atomic_json(binding_path, state)
        return False
    try:
        report = read_json(report_path, f"grounding report for {asset_id}")
        validate_report(
            report,
            row=row,
            output=output,
            ground_path=ground_path,
            mjcf=mjcf,
            plan=plan,
        )
    except QueueError as exc:
        state.update(status="failed", failure=str(exc))
        atomic_json(binding_path, state)
        return False
    state.update(
        status="complete",
        output={"path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        report={"path": str(report_path), "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
        log={"path": str(log_path), "bytes": log_path.stat().st_size, "sha256": sha256_file(log_path)},
        before_minimum_clearance_m=report["grounding"]["before"]["minimum_clearance_m"],
        applied_constant_root_z_shift_m=report["grounding"]["applied_root_z_shift_min_m"],
        after_minimum_clearance_m=report["grounding"]["after"]["minimum_clearance_m"],
        compiled_collision_sha256=report["mjcf"]["compiled_kinematic_collision_sha256"],
    )
    atomic_json(binding_path, state)
    print(
        f"[canonical-ground] COMPLETE {asset_id} "
        f"shift={state['applied_constant_root_z_shift_m']:+.6f}m "
        f"sha256={state['output']['sha256'][:12]}...",
        flush=True,
    )
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_state: dict[str, Any] | None = None
    queue_state_path: Path | None = None
    try:
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            raise QueueError("timeout-seconds must be finite and positive")
        plan_path = args.plan.resolve()
        plan = validate_plan(plan_path, args.expected_plan_sha256)
        rows = verify_source_manifest(plan)
        ground_path, mjcf, python = verify_tools_and_runtime(plan)
        checkout_observations_before = verify_read_only_checkouts(plan)
        output_root = Path(plan["output_contract"]["output_root"]).resolve()
        state_root = Path(plan["output_contract"]["state_root"]).resolve()
        if output_root.exists() or state_root.exists():
            raise QueueError(
                f"no-clobber roots must not exist: output={output_root.exists()} "
                f"state={state_root.exists()}"
            )
        output_root.mkdir(parents=True)
        state_root.mkdir(parents=True)
        lock_handle = (state_root / "queue.lock").open("x+")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue_state = {
            "schema_version": 1,
            "status": "running",
            "plan": {"path": str(plan_path), "sha256": args.expected_plan_sha256},
            "canonical_gmr_result": plan["canonical_gmr_result"],
            "grounding_tool": plan["grounding_tool"],
            "queue_tool": plan["queue_tool"],
            "mjcf": plan["mjcf"],
            "compiled_collision_contract": plan["compiled_collision_contract"],
            "body_shape_contract": BODY_SHAPE_CONTRACT,
            "cpu_only": True,
            "CUDA_VISIBLE_DEVICES": "",
            "formal_eligible": False,
            "asset_ids": [row["asset_id"] for row in rows],
            "started_utc": utc_now(),
            "read_only_checkouts_before": checkout_observations_before,
        }
        queue_state_path = state_root / "queue_state.json"
        atomic_json(queue_state_path, queue_state)
        for row in rows:
            if not run_one(
                row,
                output_root=output_root,
                state_root=state_root,
                ground_path=ground_path,
                mjcf=mjcf,
                python=python,
                plan=plan,
                timeout_seconds=args.timeout_seconds,
            ):
                queue_state.update(
                    status="failed", failed_asset_id=row["asset_id"], completed_utc=utc_now()
                )
                atomic_json(queue_state_path, queue_state)
                return 1
        queue_state.update(
            status="complete",
            completed_utc=utc_now(),
            read_only_checkouts_after=verify_read_only_checkouts(plan),
        )
        atomic_json(queue_state_path, queue_state)
        print(f"[canonical-ground] PASS: {len(rows)} grounded diagnostic assets", flush=True)
        return 0
    except (QueueError, OSError, subprocess.SubprocessError) as exc:
        if queue_state is not None and queue_state_path is not None:
            queue_state.update(status="fatal", failure=str(exc), completed_utc=utc_now())
            atomic_json(queue_state_path, queue_state)
        print(f"[canonical-ground] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
