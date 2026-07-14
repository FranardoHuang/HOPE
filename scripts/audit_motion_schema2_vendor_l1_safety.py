#!/usr/bin/env python3
"""Fail-closed vendor-MuJoCo L1 safety audit for Franco backhand-loop B.

This gate consumes the already-published exact L0 certificate and the exact
runtime-order schema-2 NPZ.  It replays the complete 151-frame path and eight
finite interpolation substeps per source interval (1201 samples, 400 Hz) in the
bound vendor MJCF.  Any robot self-interpenetration or any racket/handle
clearance below 5 mm to the frozen critical body groups fails the entire asset.

The dense interpolation is a finite conservative screen, not a mathematical
continuous-time swept-volume certificate.  This tool does not inspect the table
or net, call ``mj_step``, test dynamics/balance, train a policy, deploy, or issue
hardware commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "motion-franco-backhand-loop-b-vendor-l1-safety-20260715-v1"
PLAN_STATUS = "preregistered_source_gate_pass_runtime_audit_not_run"
ASSET_ID = "franco_backhand_loop_b"
CERTIFICATE_STATUS = "complete_cpu_vendor_l1_safety_pass_downstream_blocked"
L0_CERTIFICATE_STATUS = "complete_numeric_cpu_l0_static_pass_downstream_blocked"
L0_V2_PATH = REPO_ROOT / "scripts/audit_motion_schema2_l0_static_v2.py"
PHASE_SAFETY_PATH = REPO_ROOT / "scripts/screen_motion_gmr_phase_safety.py"
SELF_COLLISION_PATH = (
    REPO_ROOT / "hope_training/whole_body_tracking/scripts/audit_self_collision.py"
)
RACKET_GEOMS = ("right_racket_collision", "right_racket_handle_collision")


class VendorL1Error(ValueError):
    """Fail-closed source, lineage, runtime, safety or publication error."""


def _load_module(
    name: str,
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> Any:
    """Load one exact source file under ``name`` without stale-module reuse.

    ``import_module`` cannot resolve the private names used by this audit (for
    example ``ground_gmr_pkl_for_vendor_l1``), because those names deliberately
    do not exist on ``sys.path``.  Loading by path also needs to be
    transactional: a failed module body must not leave a half-initialized
    object in ``sys.modules`` or erase a pre-existing entry owned by the caller.
    """

    ensure_regular_no_symlink(path, label)
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise VendorL1Error(f"{label} expected_bytes must be a positive integer")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
    ):
        raise VendorL1Error(f"{label} expected_sha256 must be one lowercase SHA-256")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise VendorL1Error(f"{label} content binding changed before import")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VendorL1Error(f"cannot import exact {label} {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        actual_file = getattr(module, "__file__", None)
        if actual_file is None or Path(actual_file).resolve() != path.resolve():
            raise VendorL1Error(f"{label} module origin changed during import")
        if sys.modules.get(name) is not module:
            raise VendorL1Error(f"{label} replaced its exact sys.modules entry during import")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
            raise VendorL1Error(f"{label} content binding changed during import")
    except BaseException as exc:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, VendorL1Error):
            raise
        raise VendorL1Error(f"cannot import exact {label} {name} from {path}: {exc}") from exc
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VendorL1Error(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    ensure_regular_no_symlink(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VendorL1Error(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise VendorL1Error(f"{label} must be a JSON object")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise VendorL1Error(f"{label} keys changed: actual={actual}")
    return value


def ensure_regular_no_symlink(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise VendorL1Error(f"{label} must be a regular non-symlink file: {path}")


def _binding(path: Path) -> dict[str, Any]:
    ensure_regular_no_symlink(path, str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_repo_binding(row: Any, label: str, relative: str) -> Path:
    record = exact_keys(row, {"path", "bytes", "sha256"}, label)
    if record["path"] != relative:
        raise VendorL1Error(f"{label} path changed")
    path = REPO_ROOT / relative
    ensure_regular_no_symlink(path, label)
    if record["bytes"] != path.stat().st_size or record["sha256"] != sha256_file(path):
        raise VendorL1Error(f"{label} content binding changed")
    return path


def _verify_absolute_sha(row: Any, label: str) -> Path:
    record = exact_keys(row, {"path", "sha256"}, label)
    path = Path(record["path"])
    ensure_regular_no_symlink(path, label)
    if sha256_file(path) != record["sha256"]:
        raise VendorL1Error(f"{label} SHA-256 changed")
    return path


def _load_l0_v2(binding: Mapping[str, Any]) -> Any:
    path = _verify_repo_binding(
        binding, "frozen L0 validator", "scripts/audit_motion_schema2_l0_static_v2.py"
    )
    return _load_module(
        "motion_schema2_l0_v2_for_vendor_l1",
        path,
        expected_bytes=binding["bytes"],
        expected_sha256=binding["sha256"],
        label="frozen L0 validator",
    )


def validate_plan(plan_path: Path, expected_sha256: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Validate the L1 preregistration and its frozen L0/model/runtime closure."""

    ensure_regular_no_symlink(plan_path, "vendor L1 preregistration")
    actual_sha = sha256_file(plan_path)
    if actual_sha != expected_sha256 or len(expected_sha256) != 64:
        raise VendorL1Error(
            f"vendor L1 preregistration SHA mismatch: expected={expected_sha256} actual={actual_sha}"
        )
    plan = read_json(plan_path, "vendor L1 preregistration")
    exact_keys(
        plan,
        {
            "schema_version", "plan_id", "status", "human_owner", "executor", "scope",
            "asset_id", "validator", "frozen_l0", "exact_runtime_input", "a3_model",
            "runtime", "dependencies", "safety_contract", "output_contract",
            "authorization", "explicit_non_claims", "next_gate",
        },
        "vendor L1 preregistration",
    )
    if (
        plan["schema_version"] != 1
        or plan["plan_id"] != PLAN_ID
        or plan["status"] != PLAN_STATUS
        or plan["human_owner"] != "Franco"
        or plan["executor"] != "Codex"
        or plan["asset_id"] != ASSET_ID
    ):
        raise VendorL1Error("vendor L1 identity/status/attribution changed")
    if plan["scope"] != (
        "CPU-only vendor-MuJoCo whole-trajectory self-collision and racket/handle-to-robot "
        "clearance audit for exact Franco backhand-loop B schema-2; no table/net, dynamics, "
        "training, deployment or hardware"
    ):
        raise VendorL1Error("vendor L1 scope changed or overclaims")
    _verify_repo_binding(plan["validator"], "validator", "scripts/audit_motion_schema2_vendor_l1_safety.py")

    frozen = exact_keys(
        plan["frozen_l0"], {"certificate", "preregistration", "validator"}, "frozen_l0"
    )
    expected_cert = {
        "path": (
            "/workspace/codexschema/motion_video_intake_20260711/l0_static_primary_v2/"
            "franco_backhand_loop_b_98e7b883b29d.l0_static_certificate.json"
        ),
        "sha256": "60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6",
    }
    if frozen["certificate"] != expected_cert:
        raise VendorL1Error("frozen L0 certificate binding changed")
    l0_plan_path = _verify_repo_binding(
        frozen["preregistration"],
        "frozen L0 preregistration",
        "configs/motion_backhand_loop_b_l0_static_prereg_20260715_v2.json",
    )
    l0_v2 = _load_l0_v2(frozen["validator"])
    try:
        _, _, l0_v1_plan = l0_v2.validate_plan(
            l0_plan_path, frozen["preregistration"]["sha256"]
        )
    except (OSError, TypeError, ValueError) as exc:
        raise VendorL1Error(f"frozen L0 source closure changed: {exc}") from exc

    expected_npz = l0_v1_plan["exact_runtime_inputs"]["motion_npz"]
    if plan["exact_runtime_input"] != expected_npz:
        raise VendorL1Error("exact B NPZ binding differs from frozen L0")
    model = exact_keys(
        plan["a3_model"],
        {"canonical_mjcf", "derived_closure", "compiled_collision_contract"},
        "a3_model",
    )
    if model != {
        "canonical_mjcf": l0_v1_plan["a3_model"]["canonical_mjcf"],
        "derived_closure": l0_v1_plan["a3_model"]["derived_closure"],
        "compiled_collision_contract": l0_v1_plan["a3_model"]["compiled_collision_contract"],
    }:
        raise VendorL1Error("MJCF/closure/compiled collision binding differs from frozen L0")
    if plan["runtime"] != l0_v1_plan["runtime"]:
        raise VendorL1Error("vendor L1 runtime differs from frozen exact L0 runtime")

    deps = exact_keys(
        plan["dependencies"],
        {"dense_safety_tool", "self_collision_helper", "grounding_helper"},
        "dependencies",
    )
    _verify_repo_binding(
        deps["dense_safety_tool"], "dense safety tool", "scripts/screen_motion_gmr_phase_safety.py"
    )
    _verify_repo_binding(
        deps["self_collision_helper"],
        "self collision helper",
        "hope_training/whole_body_tracking/scripts/audit_self_collision.py",
    )
    if deps["grounding_helper"] != l0_v1_plan["upstream_contracts"]["grounding_helper"]:
        raise VendorL1Error("grounding helper differs from frozen L0")

    groups = {
        "head_neck": ["head_yaw_collision", "head_pitch_collision"],
        "trunk": ["torso_collision", "pelvis_collision"],
        "contralateral_arm": [
            "left_shoulder_pitch_collision", "left_shoulder_roll_collision",
            "left_shoulder_yaw_collision", "left_elbow_collision",
            "left_wrist_roll_collision_0", "left_wrist_roll_collision_1",
            "left_wrist_pitch_collision", "left_wrist_yaw_collision", "left_hand_collision",
        ],
        "striking_proximal_arm": [
            "right_shoulder_pitch_collision", "right_shoulder_roll_collision",
            "right_shoulder_yaw_collision", "right_elbow_collision",
        ],
        "lower_body": [
            "left_hip_pitch_collision", "left_hip_roll_collision", "left_hip_yaw_collision",
            "left_knee_collision", "left_ankle_pitch_collision", "left_ankle_roll_collision",
            "right_hip_pitch_collision", "right_hip_roll_collision", "right_hip_yaw_collision",
            "right_knee_collision", "right_ankle_pitch_collision", "right_ankle_roll_collision",
        ],
    }
    expected_safety = {
        "source_frames": 151,
        "source_fps": 50,
        "substeps_per_source_interval": 8,
        "dense_frames": 1201,
        "effective_sampling_hz": 400,
        "interpolation": "root_xyz_linear_root_quaternion_shortest_arc_slerp_joint_position_linear",
        "dense_sampling_is_continuous_time_certificate": False,
        "danger_propagation": "any dangerous dense sample fails the whole asset and marks both adjacent source frames",
        "joint_range_tolerance_rad": 0.00001,
        "self_collision_penetration_tolerance_m": 0.000001,
        "hard_racket_body_clearance_m": 0.005,
        "warning_racket_body_clearance_m": 0.02,
        "hard_threshold_predicate": (
            "fail iff audit_self_collision._far(model,data,racket,body,0.005) is false"
        ),
        "reporting_clearance_bisection_tolerance_m": 0.000001,
        "racket_collision_geoms": list(RACKET_GEOMS),
        "racket_body_clearance_groups": groups,
        "striking_mount_chain_clearance_exclusion": (
            "right wrist, hand and racket mounting chain excluded only from the 5 mm proximity "
            "pairs; enabled-robot contact penetration remains a hard failure"
        ),
        "floor_contact_in_scope": False,
        "hard_fail_is_noncompensable": True,
    }
    if plan["safety_contract"] != expected_safety:
        raise VendorL1Error("vendor L1 safety contract changed or weakened")
    expected_output = {
        "certificate_path": (
            "/workspace/codexschema/motion_video_intake_20260711/vendor_l1_primary_v1/"
            "franco_backhand_loop_b_98e7b883b29d.vendor_l1_safety_certificate.json"
        ),
        "must_be_absent": True,
        "parent_must_exist": True,
        "no_clobber": True,
    }
    if plan["output_contract"] != expected_output:
        raise VendorL1Error("vendor L1 output contract changed")
    expected_auth = {
        "source_gate_pass": True,
        "cpu_vendor_l1_audit_authorized_after_review": True,
        "l0_static_complete": True,
        "vendor_l1_complete": False,
        "table_net_authorized": False,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }
    if plan["authorization"] != expected_auth:
        raise VendorL1Error("vendor L1 source authorization changed")
    expected_non_claims = [
        "mathematical_continuous_time_collision_clearance",
        "table_or_net_swept_clearance",
        "ground_clearance_beyond_the_bound_L0_certificate",
        "dynamics_balance_or_contact_stability",
        "TOPP_or_time_warp",
        "strike_or_returnability",
        "RL_training_or_checkpoint_quality",
        "Gate3_or_hardware_safety",
    ]
    if plan["explicit_non_claims"] != expected_non_claims:
        raise VendorL1Error("vendor L1 explicit non-claims changed")
    if plan["next_gate"] != "only_after_exact_vendor_L1_certificate_full_trajectory_table_net_clearance":
        raise VendorL1Error("vendor L1 next gate changed")
    return plan, actual_sha, l0_v1_plan


def validate_l0_certificate(plan: Mapping[str, Any], l0_v1_plan: Mapping[str, Any]) -> dict[str, Any]:
    cert_path = _verify_absolute_sha(plan["frozen_l0"]["certificate"], "L0 certificate")
    cert = read_json(cert_path, "L0 certificate")
    exact_keys(
        cert,
        {
            "schema_version", "status", "completed_utc", "scope", "asset_id",
            "preregistration", "validator", "frozen_v1", "runtime", "lineage",
            "structure", "audit", "authorization", "explicit_non_claims", "next_gate",
        },
        "L0 certificate",
    )
    if cert["schema_version"] != 2 or cert["status"] != L0_CERTIFICATE_STATUS or cert["asset_id"] != ASSET_ID:
        raise VendorL1Error("L0 certificate identity/status changed")
    prereg = exact_keys(cert["preregistration"], {"path", "sha256"}, "L0 certificate preregistration")
    if (
        prereg["sha256"] != plan["frozen_l0"]["preregistration"]["sha256"]
        or Path(str(prereg["path"])).name
        != Path(plan["frozen_l0"]["preregistration"]["path"]).name
    ):
        raise VendorL1Error("L0 certificate preregistration binding changed")
    lineage = cert.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("motion_npz") != {
        "path": plan["exact_runtime_input"]["path"],
        "bytes": lineage.get("motion_npz", {}).get("bytes") if isinstance(lineage.get("motion_npz"), dict) else None,
        "sha256": plan["exact_runtime_input"]["sha256"],
    }:
        raise VendorL1Error("L0 certificate exact NPZ lineage changed")
    if lineage.get("runner_lineage") is not True or lineage.get("npz_bound") is not True:
        raise VendorL1Error("L0 certificate upstream lineage is not complete")
    structure = cert.get("structure")
    if not isinstance(structure, dict) or {
        key: structure.get(key) for key in ("frames", "fps", "joint_count", "body_count", "kinematics_schema_version", "finite")
    } != {
        "frames": 151, "fps": 50, "joint_count": 31, "body_count": 32,
        "kinematics_schema_version": 2, "finite": True,
    }:
        raise VendorL1Error("L0 certificate structure changed")
    expected_auth = {
        "l0_static_complete": True, "vendor_l1_authorized": True,
        "table_net_authorized": False, "dynamics_authorized": False,
        "simulator_authorized": False, "training_authorized": False,
        "formal_motion_authorized": False, "hardware_authorized": False,
    }
    if cert.get("authorization") != expected_auth:
        raise VendorL1Error("L0 certificate authorization changed")
    audit = cert.get("audit")
    if not isinstance(audit, dict) or audit.get("model", {}).get("compiled_collision_sha256") != (
        l0_v1_plan["a3_model"]["compiled_collision_contract"]["sha256"]
    ):
        raise VendorL1Error("L0 certificate compiled collision lineage changed")
    if audit.get("model", {}).get("canonical_mjcf", {}).get("sha256") != (
        l0_v1_plan["a3_model"]["canonical_mjcf"]["sha256"]
    ):
        raise VendorL1Error("L0 certificate MJCF lineage changed")
    return _binding(cert_path)


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return value if value is not None else f"geom{geom_id}"


def _body_name(mujoco: Any, model: Any, body_id: int) -> str:
    value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
    return value if value is not None else f"body{body_id}"


def summarize_hard_failures(
    collision_bad: np.ndarray,
    clearance_bad: np.ndarray,
    source_time: np.ndarray,
    source_frames: int,
    unsafe_source_mask_fn: Any,
) -> dict[str, Any]:
    """Enforce non-compensable hard failures and expose conservative source marking."""

    collision = np.asarray(collision_bad, dtype=bool)
    clearance = np.asarray(clearance_bad, dtype=bool)
    times = np.asarray(source_time, dtype=np.float64)
    if collision.shape != clearance.shape or collision.shape != times.shape:
        raise VendorL1Error("dense safety masks/time have inconsistent shapes")
    dangerous = collision | clearance
    unsafe = np.asarray(unsafe_source_mask_fn(source_frames, times, dangerous), dtype=bool)
    result = {
        "dangerous_dense_samples": int(np.count_nonzero(dangerous)),
        "self_collision_dense_samples": int(np.count_nonzero(collision)),
        "racket_clearance_dense_samples": int(np.count_nonzero(clearance)),
        "unsafe_source_frames": int(np.count_nonzero(unsafe)),
        "unsafe_source_indices": np.flatnonzero(unsafe).astype(int).tolist(),
        "hard_fail_is_noncompensable": True,
    }
    if result["dangerous_dense_samples"]:
        raise VendorL1Error(
            "vendor L1 safety hard failure: "
            f"self_collision={result['self_collision_dense_samples']} "
            f"racket_clearance={result['racket_clearance_dense_samples']}"
        )
    return result


def evaluate_racket_clearance_pairs(
    helper: Any,
    model: Any,
    data: Any,
    racket_ids: Sequence[int],
    group_ids: Mapping[str, Sequence[int]],
    *,
    hard_threshold_m: float,
    warning_threshold_m: float,
    reporting_tolerance_m: float,
    geom_name: Any,
) -> dict[str, Any]:
    """Evaluate one pose using exact saturation predicates for hard decisions.

    ``geom_clearance`` remains useful for a human-readable minimum, but its
    bisection midpoint must never decide the 5 mm gate.  ``_far(dm)`` is the
    helper's exact/saturating predicate for true distance >= dm, so the strict
    ``distance < threshold`` contract is exactly ``not _far(threshold)``.
    """

    if not (
        math.isfinite(hard_threshold_m)
        and math.isfinite(warning_threshold_m)
        and math.isfinite(reporting_tolerance_m)
        and 0.0 < reporting_tolerance_m < hard_threshold_m < warning_threshold_m
    ):
        raise VendorL1Error("racket clearance thresholds/tolerance are invalid")
    minimum = float("inf")
    minimum_pair: list[str] | None = None
    hard_failure = False
    warning = False
    for group_name, bodies in group_ids.items():
        for racket in racket_ids:
            for body in bodies:
                if not bool(helper._far(model, data, racket, body, hard_threshold_m)):
                    hard_failure = True
                if not bool(helper._far(model, data, racket, body, warning_threshold_m)):
                    warning = True
                distance, _ = helper.geom_clearance(
                    model, data, racket, body, tol=reporting_tolerance_m
                )
                distance = float(distance)
                if distance < minimum:
                    minimum = distance
                    minimum_pair = [geom_name(racket), geom_name(body), str(group_name)]
    if minimum_pair is None or not math.isfinite(minimum):
        raise VendorL1Error("racket clearance group evaluation produced no finite pair")
    return {
        "hard_failure": hard_failure,
        "warning": warning,
        "minimum_clearance_m": minimum,
        "minimum_pair_and_group": minimum_pair,
    }


def validate_output_preconditions(plan: Mapping[str, Any]) -> Path:
    """Require the preregistered absent target and real pre-existing parent."""

    output = Path(plan["output_contract"]["certificate_path"])
    if os.path.lexists(output):
        raise VendorL1Error(f"certificate path already exists or is a symlink; no-clobber: {output}")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise VendorL1Error(
            f"certificate parent must pre-exist and be a real directory: {parent}"
        )
    return output


def audit_runtime(plan: Mapping[str, Any], l0_v1_plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    l0_v2 = _load_l0_v2(plan["frozen_l0"]["validator"])
    try:
        mujoco, runtime = l0_v2.V1.validate_runtime_environment(l0_v1_plan)
    except (OSError, TypeError, ValueError) as exc:
        raise VendorL1Error(f"exact CPU runtime validation failed: {exc}") from exc
    l0_binding = validate_l0_certificate(plan, l0_v1_plan)
    npz_path = _verify_absolute_sha(plan["exact_runtime_input"], "exact B schema-2 NPZ")
    arrays = l0_v2.V1.load_npz_exact(npz_path, l0_v1_plan)

    phase_binding = plan["dependencies"]["dense_safety_tool"]
    phase = _load_module(
        "motion_phase_safety_for_vendor_l1",
        PHASE_SAFETY_PATH,
        expected_bytes=phase_binding["bytes"],
        expected_sha256=phase_binding["sha256"],
        label="dense safety tool",
    )
    collision_binding = plan["dependencies"]["self_collision_helper"]
    self_collision = _load_module(
        "motion_self_collision_for_vendor_l1",
        SELF_COLLISION_PATH,
        expected_bytes=collision_binding["bytes"],
        expected_sha256=collision_binding["sha256"],
        label="self-collision helper",
    )
    ground_binding = plan["dependencies"]["grounding_helper"]
    ground_path = REPO_ROOT / plan["dependencies"]["grounding_helper"]["path"]
    ground = _load_module(
        "ground_gmr_pkl_for_vendor_l1",
        ground_path,
        expected_bytes=ground_binding["bytes"],
        expected_sha256=ground_binding["sha256"],
        label="grounding helper",
    )
    mjcf_path = REPO_ROOT / plan["a3_model"]["canonical_mjcf"]["path"]
    binding = ground.bind_model(mujoco, mjcf_path, ground_geom_name="floor")
    expected_collision = plan["a3_model"]["compiled_collision_contract"]
    if (
        binding.collision_contract_sha256 != expected_collision["sha256"]
        or list(binding.collision_geom_ids) != expected_collision["enabled_robot_geom_ids"]
    ):
        raise VendorL1Error("compiled vendor collision contract changed at runtime")

    root_wxyz = np.asarray(arrays["body_quat_w"][:, 0], dtype=np.float64)
    payload = {
        "root_pos": np.asarray(arrays["body_pos_w"][:, 0], dtype=np.float64),
        "root_rot": root_wxyz[:, [1, 2, 3, 0]],
        "dof_pos": np.asarray(arrays["joint_pos"], dtype=np.float64),
        "fps": np.array([50.0], dtype=np.float64),
    }
    safety = plan["safety_contract"]
    dense, source_time = phase.densify_payload(payload, safety["substeps_per_source_interval"])
    if (
        dense["root_pos"].shape != (safety["dense_frames"], 3)
        or dense["dof_pos"].shape != (safety["dense_frames"], 31)
        or float(np.asarray(dense["fps"]).reshape(-1)[0]) != safety["effective_sampling_hz"]
        or not all(np.isfinite(np.asarray(dense[key])).all() for key in ("root_pos", "root_rot", "dof_pos"))
    ):
        raise VendorL1Error("dense trajectory structure/rate/finite contract changed")
    ground.validate_joint_ranges(
        dense, binding, tolerance_rad=safety["joint_range_tolerance_rad"]
    )
    qpos = phase._qpos_from_payload(binding, dense)
    model, data = binding.model, binding.data
    model.geom_contype[binding.ground_geom_id] = 0
    model.geom_conaffinity[binding.ground_geom_id] = 0
    robot_geoms = set(int(value) for value in binding.collision_geom_ids)
    geom_by_name = {_geom_name(mujoco, model, geom_id): int(geom_id) for geom_id in robot_geoms}
    missing_racket = [name for name in RACKET_GEOMS if name not in geom_by_name]
    if missing_racket:
        raise VendorL1Error(f"vendor MJCF lacks racket collision geoms {missing_racket}")
    group_ids: dict[str, tuple[int, ...]] = {}
    for group, names in safety["racket_body_clearance_groups"].items():
        missing = [name for name in names if name not in geom_by_name]
        if missing:
            raise VendorL1Error(f"vendor MJCF lacks {group} clearance geoms {missing}")
        group_ids[group] = tuple(geom_by_name[name] for name in names)
    racket_ids = tuple(geom_by_name[name] for name in RACKET_GEOMS)

    count = qpos.shape[0]
    collision_bad = np.zeros(count, dtype=bool)
    clearance_bad = np.zeros(count, dtype=bool)
    clearance_warn = np.zeros(count, dtype=bool)
    minimum = np.full(count, np.inf, dtype=np.float64)
    minimum_pair: list[list[str] | None] = [None] * count
    events: list[dict[str, Any]] = []
    tolerance = float(safety["self_collision_penetration_tolerance_m"])
    hard = float(safety["hard_racket_body_clearance_m"])
    warning = float(safety["warning_racket_body_clearance_m"])
    for dense_frame in range(count):
        data.qpos[:] = qpos[dense_frame]
        mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if (
                float(contact.dist) >= -tolerance
                or int(contact.geom1) not in robot_geoms
                or int(contact.geom2) not in robot_geoms
            ):
                continue
            collision_bad[dense_frame] = True
            if len(events) < 512:
                events.append({
                    "dense_frame": dense_frame,
                    "source_time_frames": float(source_time[dense_frame]),
                    "penetration_m": float(-contact.dist),
                    "geom_pair": [
                        _geom_name(mujoco, model, contact.geom1),
                        _geom_name(mujoco, model, contact.geom2),
                    ],
                    "body_pair": [
                        _body_name(mujoco, model, int(model.geom_bodyid[contact.geom1])),
                        _body_name(mujoco, model, int(model.geom_bodyid[contact.geom2])),
                    ],
                })
        clearance = evaluate_racket_clearance_pairs(
            self_collision,
            model,
            data,
            racket_ids,
            group_ids,
            hard_threshold_m=hard,
            warning_threshold_m=warning,
            reporting_tolerance_m=float(safety["reporting_clearance_bisection_tolerance_m"]),
            geom_name=lambda geom_id: _geom_name(mujoco, model, geom_id),
        )
        minimum[dense_frame] = clearance["minimum_clearance_m"]
        minimum_pair[dense_frame] = clearance["minimum_pair_and_group"]
        clearance_bad[dense_frame] = clearance["hard_failure"]
        clearance_warn[dense_frame] = clearance["warning"]

    hard_summary = summarize_hard_failures(
        collision_bad, clearance_bad, source_time, 151, phase.unsafe_source_mask
    )
    closest = int(np.argmin(minimum))
    audit = {
        "sampling": {
            "source_frames": 151,
            "source_fps": 50,
            "dense_frames": count,
            "substeps_per_source_interval": safety["substeps_per_source_interval"],
            "effective_sampling_hz": safety["effective_sampling_hz"],
            "interpolation": safety["interpolation"],
            "continuous_time_certificate": False,
        },
        "self_collision": {
            "penetration_tolerance_m": tolerance,
            "dangerous_dense_samples": int(np.count_nonzero(collision_bad)),
            "events_truncated": len(events) >= 512,
            "events": events,
        },
        "racket_body_clearance": {
            "hard_threshold_m": hard,
            "warning_threshold_m": warning,
            "hard_threshold_predicate": safety["hard_threshold_predicate"],
            "reporting_bisection_tolerance_m": safety[
                "reporting_clearance_bisection_tolerance_m"
            ],
            "dangerous_dense_samples": int(np.count_nonzero(clearance_bad)),
            "warning_dense_samples": int(np.count_nonzero(clearance_warn)),
            "minimum_clearance_m": float(minimum[closest]),
            "minimum_source_time_frames": float(source_time[closest]),
            "minimum_pair_and_group": minimum_pair[closest],
        },
        "hard_gate": hard_summary,
        "mj_step_calls": 0,
    }
    lineage = {
        "l0_certificate": l0_binding,
        "motion_npz": _binding(npz_path),
        "canonical_mjcf": _binding(mjcf_path),
        "derived_mjcf_closure": plan["a3_model"]["derived_closure"],
        "compiled_collision_sha256": binding.collision_contract_sha256,
    }
    return {"runtime": runtime, "lineage": lineage, "audit": audit}, arrays


def build_certificate(
    plan: Mapping[str, Any], plan_path: Path, plan_sha: str, l0_v1_plan: Mapping[str, Any]
) -> dict[str, Any]:
    result, _ = audit_runtime(plan, l0_v1_plan)
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
            "l0_static_complete": True,
            "vendor_l1_complete": True,
            "table_net_authorized": True,
            "dynamics_authorized": False,
            "simulator_authorized": False,
            "training_authorized": False,
            "formal_motion_authorized": False,
            "hardware_authorized": False,
        },
        "explicit_non_claims": plan["explicit_non_claims"],
        "next_gate": plan["next_gate"],
    }


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise VendorL1Error(f"certificate path already exists; no-clobber: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise VendorL1Error(f"certificate parent must pre-exist and be a real directory: {path.parent}")
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
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
        plan, plan_sha, l0_v1_plan = validate_plan(
            args.prereg.resolve(), args.expected_prereg_sha256
        )
        if args.command == "static":
            print(
                f"[motion-vendor-l1] PASS static asset={ASSET_ID} source_exact=true "
                "runtime_audit=false no_write=true continuous_time_claim=false"
            )
            return 0
        output = validate_output_preconditions(plan)
        certificate = build_certificate(plan, args.prereg.resolve(), plan_sha, l0_v1_plan)
        if args.command == "dry-run":
            print(
                f"[motion-vendor-l1] PASS dry-run asset={ASSET_ID} runtime_audit=true "
                "certificate_written=false vendor_l1_complete=false downstream_blocked=true"
            )
            return 0
        write_exclusive(output, certificate)
        print(
            f"[motion-vendor-l1] PASS audit asset={ASSET_ID} vendor_l1=true "
            f"certificate_sha256={sha256_file(output)} table_net_next=true"
        )
        return 0
    except (VendorL1Error, OSError, TypeError, ValueError) as exc:
        print(f"[motion-vendor-l1] FAIL {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
