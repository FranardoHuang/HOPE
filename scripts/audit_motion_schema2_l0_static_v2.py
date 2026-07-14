#!/usr/bin/env python3
"""Numerically honest CPU-only L0 audit for the frozen backhand-loop B schema-2 asset.

Version 1 required byte equality after feeding the stored float32 pelvis body pose back
into MuJoCo as a free-joint state.  That state is not the producer's original free-joint
qpos: the producer stored MuJoCo's normalized body pose after FK and rounded it to
float32.  A second normalize/FK/float32 projection is therefore not bit-idempotent.

V2 keeps every lineage, model, joint-range, grounding, support-foot and publication gate
from the frozen V1 plan.  Only the non-reconstructable replay comparison changes:

* link position/quaternion components get a two-bin float32-ULP envelope, with one-unit
  absolute scaling near zero and explicit physical caps;
* COM velocity is reconstructed from the stored link pose plus the exact MJCF inertial
  offsets, with a bound derived from float32 input projection and the frozen 50 Hz
  finite-difference stencil;
* angular velocity remains byte-exact because the producer computed it directly from
  the stored body quaternion array.

No simulator step, dynamics, training, deployment or hardware command is performed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = REPO_ROOT / "scripts/audit_motion_schema2_l0_static.py"
LEGACY_PLAN_PATH = REPO_ROOT / "configs/motion_backhand_loop_b_l0_static_prereg_20260714.json"
PLAN_ID = "motion-franco-backhand-loop-b-l0-static-20260715-v2"
PLAN_STATUS = "preregistered_v1_numeric_fail_v2_source_gate_pass_runtime_audit_not_run"
ASSET_ID = "franco_backhand_loop_b"
CERTIFICATE_STATUS = "complete_numeric_cpu_l0_static_pass_downstream_blocked"
FLOAT32_EPSILON = float(np.finfo(np.float32).eps)


def _load_legacy():
    spec = importlib.util.spec_from_file_location("motion_schema2_l0_static_v1_frozen", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen V1 L0 validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V1 = _load_legacy()
L0ContractError = V1.L0ContractError


def _binding_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def validate_plan(plan_path: Path, expected_sha256: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Validate V2 plus the complete frozen V1 contract it inherits."""

    V1.ensure_regular_no_symlink(plan_path, "V2 L0 preregistration")
    actual_sha = V1.sha256_file(plan_path)
    if actual_sha != V1.require_sha(expected_sha256, "expected V2 preregistration SHA-256"):
        raise L0ContractError(
            f"V2 L0 preregistration SHA mismatch: expected={expected_sha256} actual={actual_sha}"
        )
    plan = V1.read_json(plan_path, "V2 L0 preregistration")
    V1.exact_keys(
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
            "frozen_v1",
            "v1_failure_evidence",
            "numerical_replay_contract",
            "inherited_hard_gates",
            "output_contract",
            "authorization",
            "explicit_non_claims",
            "next_gate",
        },
        "V2 L0 preregistration",
    )
    if (
        plan["schema_version"] != 2
        or plan["plan_id"] != PLAN_ID
        or plan["status"] != PLAN_STATUS
        or plan["human_owner"] != "Franco"
        or plan["executor"] != "Codex"
        or plan["asset_id"] != ASSET_ID
    ):
        raise L0ContractError("V2 L0 identity/status/attribution changed")
    if plan["scope"] != (
        "CPU-only runtime-order schema-2 L0 static audit with field-specific numerical "
        "replay contract; kinematic mj_forward only, no simulator step, dynamics, training, "
        "deployment or hardware"
    ):
        raise L0ContractError("V2 L0 scope changed or overclaims")
    V1.verify_binding(
        plan["validator"],
        "V2 validator",
        repo_root=REPO_ROOT,
        expected_path="scripts/audit_motion_schema2_l0_static_v2.py",
    )

    frozen = V1.exact_keys(
        plan["frozen_v1"], {"preregistration", "validator"}, "frozen V1 binding"
    )
    v1_plan_path = V1.verify_binding(
        frozen["preregistration"],
        "frozen V1 preregistration",
        repo_root=REPO_ROOT,
        expected_path="configs/motion_backhand_loop_b_l0_static_prereg_20260714.json",
    )
    V1.verify_binding(
        frozen["validator"],
        "frozen V1 validator",
        repo_root=REPO_ROOT,
        expected_path="scripts/audit_motion_schema2_l0_static.py",
    )
    v1_plan, _v1_sha = V1.validate_plan(
        v1_plan_path, frozen["preregistration"]["sha256"]
    )
    if not _binding_matches(v1_plan["validator"], frozen["validator"]):
        raise L0ContractError("V2 frozen V1 validator binding differs from V1 plan")

    failure = V1.exact_keys(
        plan["v1_failure_evidence"],
        {
            "evidence_level",
            "producer_host",
            "audit_host",
            "command",
            "outcome",
            "certificate_written",
            "position",
            "quaternion",
            "com_linear_velocity",
            "body_angular_velocity",
            "source_root_cause",
        },
        "V1 failure evidence",
    )
    expected_failure = {
        "evidence_level": "operator_preserved_summary_plus_source_proof_no_result_artifact",
        "producer_host": "pod1",
        "audit_host": "pod2",
        "command": "dry-run",
        "outcome": "fail_closed_before_certificate",
        "certificate_written": False,
        "position": {"not_equal_components": 537, "max_abs": 1.1920929e-7},
        "quaternion": {"not_equal_components": 917, "max_abs": 5.9604645e-8},
        "com_linear_velocity": {"not_equal_components": 1261, "max_abs": 2.9802322e-6},
        "body_angular_velocity": {"not_equal_components": 2320, "max_abs": 5.9679151e-6},
        "source_root_cause": (
            "schema2_stores_post_FK_normalized_float32_root_body_pose_not_original_free_joint_qpos; "
            "V1_reinjects_that_lossy_pose_and_demands_non_idempotent_byte_equality"
        ),
    }
    if failure != expected_failure:
        raise L0ContractError("V1 failure evidence changed")

    numeric = V1.exact_keys(
        plan["numerical_replay_contract"],
        {
            "float32_epsilon",
            "link_position",
            "link_quaternion",
            "com_linear_velocity",
            "body_angular_velocity",
        },
        "V2 numerical replay contract",
    )
    if numeric != {
        "float32_epsilon": FLOAT32_EPSILON,
        "link_position": {
            "comparison": "componentwise_ulp_scaled_absolute",
            "max_ulp_bins": 2,
            "one_unit_floor": True,
            "max_abs_tolerance_m": 5.0e-7,
        },
        "link_quaternion": {
            "comparison": "componentwise_ulp_scaled_absolute_same_hemisphere",
            "max_ulp_bins": 2,
            "one_unit_floor": True,
            "max_abs_tolerance": 5.0e-7,
        },
        "com_linear_velocity": {
            "reconstruction": (
                "stored_link_pose_plus_exact_MJCF_body_ipos_then_numpy_gradient_dt_1_over_50"
            ),
            "rotation_component_bound": "8*q_component_bound+4*q_component_bound_squared",
            "finite_difference_bound": "two_com_position_bounds_divided_by_dt_plus_output_roundoff",
            "output_roundoff_ulp_bins": 2,
            "max_abs_tolerance_mps": 2.0e-4,
        },
        "body_angular_velocity": {
            "reconstruction": "producer_exact_so3_derivative_of_stored_body_quat_w_dt_1_over_50",
            "comparison": "byte_equal",
        },
    }:
        raise L0ContractError("V2 numerical replay contract changed")

    inherited = V1.exact_keys(
        plan["inherited_hard_gates"],
        {
            "joint_range_tolerance_rad",
            "grounding",
            "support_bodies",
            "unchanged_checks",
        },
        "inherited hard gates",
    )
    if (
        inherited["joint_range_tolerance_rad"]
        != v1_plan["l0_contract"]["joint_range_tolerance_rad"]
        or inherited["grounding"] != v1_plan["l0_contract"]["grounding"]
        or inherited["support_bodies"] != v1_plan["l0_contract"]["support_bodies"]
        or inherited["unchanged_checks"]
        != [
            "exact_input_lineage_and_SHA",
            "schema_shape_dtype_finite_and_order",
            "joint_velocity_byte_equality",
            "joint_range",
            "ground_clearance",
            "support_foot_ancestry",
            "certificate_no_clobber",
        ]
    ):
        raise L0ContractError("V2 weakened or changed a frozen V1 hard gate")

    output = V1.exact_keys(
        plan["output_contract"],
        {"certificate_path", "must_be_absent", "parent_must_exist", "no_clobber"},
        "V2 output contract",
    )
    if output != {
        "certificate_path": (
            "/workspace/codexschema/motion_video_intake_20260711/l0_static_primary_v2/"
            "franco_backhand_loop_b_98e7b883b29d.l0_static_certificate.json"
        ),
        "must_be_absent": True,
        "parent_must_exist": True,
        "no_clobber": True,
    }:
        raise L0ContractError("V2 output contract changed")
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
        raise L0ContractError("V2 source authorization changed")
    if plan["explicit_non_claims"] != v1_plan["explicit_non_claims"]:
        raise L0ContractError("V2 explicit non-claims differ from frozen V1")
    if plan["next_gate"] != v1_plan["next_gate"]:
        raise L0ContractError("V2 next gate differs from frozen V1")
    return plan, actual_sha, v1_plan


def _require_float32_pair(
    reference: np.ndarray, candidate: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    if left.shape != right.shape or left.dtype != np.float32 or right.dtype != np.float32:
        raise L0ContractError(f"{label} must be same-shape float32 arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise L0ContractError(f"{label} contains NaN/Inf")
    return left, right


def float32_ulp_width(values: np.ndarray, *, one_unit_floor: bool) -> np.ndarray:
    """Return local float32 bin width, optionally floored at spacing(1.0).

    The floor is deliberate: downstream FK components can be near zero because of
    cancellation even though their inputs have unit-scale float32 quantization.
    """

    array = np.asarray(values)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise L0ContractError("ULP width input must be finite float32")
    positive = np.nextafter(array, np.float32(np.inf))
    negative = np.nextafter(array, np.float32(-np.inf))
    base = array.astype(np.float64)
    width = np.maximum(
        np.abs(positive.astype(np.float64) - base),
        np.abs(base - negative.astype(np.float64)),
    )
    if one_unit_floor:
        width = np.maximum(width, FLOAT32_EPSILON)
    if not np.isfinite(width).all() or np.any(width <= 0.0):
        raise L0ContractError("cannot derive finite positive float32 ULP widths")
    return width


def evaluate_ulp_scaled_replay(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    label: str,
    max_ulp_bins: int,
    one_unit_floor: bool,
    max_abs_tolerance: float,
) -> dict[str, Any]:
    left, right = _require_float32_pair(reference, candidate, label)
    if type(max_ulp_bins) is not int or max_ulp_bins <= 0:
        raise L0ContractError(f"{label} ULP budget is malformed")
    if not np.isfinite(max_abs_tolerance) or max_abs_tolerance <= 0.0:
        raise L0ContractError(f"{label} physical tolerance cap is malformed")
    widths = np.maximum(
        float32_ulp_width(left, one_unit_floor=one_unit_floor),
        float32_ulp_width(right, one_unit_floor=one_unit_floor),
    )
    tolerance = float(max_ulp_bins) * widths
    if float(np.max(tolerance)) > max_abs_tolerance:
        raise L0ContractError(
            f"{label} derived numerical tolerance exceeds physical cap: "
            f"derived={float(np.max(tolerance)):.9g} cap={max_abs_tolerance:.9g}"
        )
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    bad = delta > tolerance
    if np.any(bad):
        flat = int(np.argmax(delta - tolerance))
        index = tuple(int(v) for v in np.unravel_index(flat, delta.shape))
        raise L0ContractError(
            f"{label} exceeds {max_ulp_bins}-bin float32 replay budget at {index}: "
            f"delta={float(delta[index]):.9g} tolerance={float(tolerance[index]):.9g}"
        )
    ratio = np.divide(delta, tolerance, out=np.zeros_like(delta), where=tolerance > 0.0)
    return {
        "comparison": "componentwise_ulp_scaled_absolute",
        "max_ulp_bins": max_ulp_bins,
        "one_unit_floor": bool(one_unit_floor),
        "max_abs_delta": float(np.max(delta)),
        "max_abs_tolerance": float(np.max(tolerance)),
        "max_budget_fraction": float(np.max(ratio)),
        "not_byte_equal_components": int(np.count_nonzero(left != right)),
    }


def evaluate_pose_replay(
    stored_pos: np.ndarray,
    recomputed_pos: np.ndarray,
    stored_quat: np.ndarray,
    recomputed_quat: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    pos_contract = contract["link_position"]
    quat_contract = contract["link_quaternion"]
    left_quat, right_quat = _require_float32_pair(
        stored_quat, recomputed_quat, "link quaternion replay"
    )
    dots = np.sum(left_quat.astype(np.float64) * right_quat.astype(np.float64), axis=-1)
    if np.any(dots <= 0.0):
        index = tuple(int(v) for v in np.argwhere(dots <= 0.0)[0])
        raise L0ContractError(f"link quaternion replay changed hemisphere at {index}")
    return {
        "link_position": evaluate_ulp_scaled_replay(
            stored_pos,
            recomputed_pos,
            label="link position replay",
            max_ulp_bins=pos_contract["max_ulp_bins"],
            one_unit_floor=pos_contract["one_unit_floor"],
            max_abs_tolerance=pos_contract["max_abs_tolerance_m"],
        ),
        "link_quaternion": evaluate_ulp_scaled_replay(
            left_quat,
            right_quat,
            label="link quaternion replay",
            max_ulp_bins=quat_contract["max_ulp_bins"],
            one_unit_floor=quat_contract["one_unit_floor"],
            max_abs_tolerance=quat_contract["max_abs_tolerance"],
        ),
    }


def quaternion_rotation_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise L0ContractError("COM reconstruction quaternion input is malformed")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 0.0) or not np.isfinite(norm).all():
        raise L0ContractError("COM reconstruction quaternion norm is invalid")
    q = q / norm
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - w * z)
    matrix[..., 0, 2] = 2.0 * (x * z + w * y)
    matrix[..., 1, 0] = 2.0 * (x * y + w * z)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - w * x)
    matrix[..., 2, 0] = 2.0 * (x * z - w * y)
    matrix[..., 2, 1] = 2.0 * (y * z + w * x)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def reconstruct_com_from_stored_pose(
    body_pos: np.ndarray, body_quat: np.ndarray, body_ipos: np.ndarray
) -> np.ndarray:
    pos = np.asarray(body_pos)
    quat = np.asarray(body_quat)
    offsets = np.asarray(body_ipos, dtype=np.float64)
    if (
        pos.dtype != np.float32
        or quat.dtype != np.float32
        or pos.shape[:-1] != quat.shape[:-1]
        or pos.shape[-1] != 3
        or quat.shape[-1] != 4
        or offsets.shape != (pos.shape[1], 3)
        or not np.isfinite(pos).all()
        or not np.isfinite(quat).all()
        or not np.isfinite(offsets).all()
    ):
        raise L0ContractError("stored-pose COM reconstruction inputs are malformed")
    matrices = quaternion_rotation_matrices_wxyz(quat)
    rotated = np.einsum("tbij,bj->tbi", matrices, offsets, optimize=False)
    return (pos.astype(np.float64) + rotated).astype(np.float32)


def evaluate_com_linear_velocity(
    stored_velocity: np.ndarray,
    reconstructed_com: np.ndarray,
    body_ipos: np.ndarray,
    *,
    dt: float,
    pose_ulp_bins: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    velocity = np.asarray(stored_velocity)
    com = np.asarray(reconstructed_com)
    offsets = np.asarray(body_ipos, dtype=np.float64)
    if (
        velocity.dtype != np.float32
        or com.dtype != np.float32
        or velocity.shape != com.shape
        or not np.isfinite(velocity).all()
        or not np.isfinite(com).all()
        or not np.isfinite(offsets).all()
        or not np.isfinite(dt)
        or dt <= 0.0
    ):
        raise L0ContractError("COM linear velocity replay inputs are malformed")
    expected = np.gradient(com, dt, axis=0).astype(np.float32)
    q_component_bound = float(pose_ulp_bins) * FLOAT32_EPSILON
    position_scale = max(1.0, float(np.max(np.abs(com.astype(np.float64)))))
    position_component_bound = float(pose_ulp_bins) * FLOAT32_EPSILON * position_scale
    rotation_component_bound = 8.0 * q_component_bound + 4.0 * q_component_bound**2
    inertial_radius_l1 = float(np.max(np.sum(np.abs(offsets), axis=-1)))
    com_scale = max(1.0, float(np.max(np.abs(com.astype(np.float64)))))
    com_position_bound = (
        position_component_bound
        + inertial_radius_l1 * rotation_component_bound
        + 2.0 * FLOAT32_EPSILON * com_scale
    )
    velocity_scale = max(
        1.0,
        float(np.max(np.abs(velocity.astype(np.float64)))),
        float(np.max(np.abs(expected.astype(np.float64)))),
    )
    derived_tolerance = (
        2.0 * com_position_bound / dt
        + float(contract["output_roundoff_ulp_bins"]) * FLOAT32_EPSILON * velocity_scale
    )
    physical_cap = float(contract["max_abs_tolerance_mps"])
    if not np.isfinite(derived_tolerance) or derived_tolerance > physical_cap:
        raise L0ContractError(
            "COM linear velocity derived tolerance exceeds physical cap: "
            f"derived={derived_tolerance:.9g} cap={physical_cap:.9g}"
        )
    delta = np.abs(velocity.astype(np.float64) - expected.astype(np.float64))
    if float(np.max(delta)) > derived_tolerance:
        flat = int(np.argmax(delta))
        index = tuple(int(v) for v in np.unravel_index(flat, delta.shape))
        raise L0ContractError(
            f"COM linear velocity exceeds 50 Hz roundoff bound at {index}: "
            f"delta={float(delta[index]):.9g} tolerance={derived_tolerance:.9g}"
        )
    return {
        "comparison": "derived_absolute_bound",
        "reconstruction": contract["reconstruction"],
        "max_abs_delta_mps": float(np.max(delta)),
        "derived_abs_tolerance_mps": float(derived_tolerance),
        "physical_cap_mps": physical_cap,
        "inertial_radius_l1_m": inertial_radius_l1,
        "not_byte_equal_components": int(np.count_nonzero(velocity != expected)),
        "dt_s": float(dt),
    }


def evaluate_body_angular_velocity_exact(
    stored_velocity: np.ndarray,
    stored_quat: np.ndarray,
    *,
    dt: float,
    converter: Any,
) -> dict[str, Any]:
    velocity = np.asarray(stored_velocity)
    quat = np.asarray(stored_quat)
    if velocity.dtype != np.float32 or quat.dtype != np.float32:
        raise L0ContractError("body angular velocity replay requires float32 arrays")
    expected = np.stack(
        [converter.so3_derivative(quat[:, col], dt) for col in range(quat.shape[1])],
        axis=1,
    ).astype(np.float32)
    if not np.array_equal(velocity, expected):
        delta = float(np.max(np.abs(velocity.astype(np.float64) - expected.astype(np.float64))))
        raise L0ContractError(
            f"body angular velocity is not producer-exact from stored quaternion; max_abs={delta:.9g}"
        )
    return {
        "comparison": "byte_equal",
        "producer_exact_from_stored_quaternion": True,
        "dt_s": float(dt),
    }


def audit_kinematics_v2(
    v2_plan: Mapping[str, Any],
    v1_plan: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    mujoco: Any,
) -> dict[str, Any]:
    upstream = v1_plan["upstream_contracts"]
    ground = V1._import_exact(
        "ground_gmr_pkl", REPO_ROOT / upstream["grounding_helper"]["path"]
    )
    converter = V1._import_exact(
        "csv_to_npz_mujoco", REPO_ROOT / upstream["converter_helper"]["path"]
    )
    mjcf_path = REPO_ROOT / v1_plan["a3_model"]["canonical_mjcf"]["path"]
    try:
        model_binding = ground.bind_model(
            mujoco,
            mjcf_path,
            ground_geom_name=v1_plan["a3_model"]["compiled_collision_contract"]["ground_geom"],
        )
    except Exception as exc:
        raise L0ContractError(f"cannot bind exact compiled A3 model: {exc}") from exc
    collision = v1_plan["a3_model"]["compiled_collision_contract"]
    if (
        model_binding.collision_contract_sha256 != collision["sha256"]
        or list(model_binding.collision_geom_ids) != collision["enabled_robot_geom_ids"]
        or len(model_binding.collision_geom_ids) != collision["enabled_robot_geom_count"]
        or model_binding.ground_z_m != collision["ground_z_m"]
    ):
        raise L0ContractError("compiled MuJoCo collision contract changed")

    joint_names = V1._read_names(
        REPO_ROOT / upstream["runtime_joint_order"]["path"], 31, "runtime joint order"
    )
    body_names = V1._read_names(
        REPO_ROOT / upstream["runtime_body_order"]["path"], 32, "runtime body order"
    )
    model = model_binding.model
    data = model_binding.data
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    for name in joint_names:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise L0ContractError(f"runtime joint {name!r} missing from A3 model")
        joint_ids.append(jid)
        qpos_addresses.append(int(model.jnt_qposadr[jid]))
    if len(set(joint_ids)) != 31 or len(set(qpos_addresses)) != 31:
        raise L0ContractError("runtime joint mapping is not a 31-joint bijection")
    body_ids: list[int] = []
    for name in body_names:
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        if bid < 0:
            raise L0ContractError(f"runtime body {name!r} missing from A3 model")
        body_ids.append(bid)
    if len(set(body_ids)) != 32 or body_ids[0] != model_binding.root_body_id:
        raise L0ContractError("runtime body mapping is not the exact root-first 32-body bijection")

    q = arrays["joint_pos"].astype(np.float64)
    ranges = np.asarray(model.jnt_range, dtype=np.float64)[joint_ids]
    tolerance_rad = float(v1_plan["l0_contract"]["joint_range_tolerance_rad"])
    joint_range_result = V1.evaluate_joint_ranges(q, ranges, joint_names, tolerance_rad)

    frames = int(v1_plan["l0_contract"]["frames"])
    recomputed_pos = np.empty((frames, 32, 3), dtype=np.float32)
    recomputed_quat = np.empty((frames, 32, 4), dtype=np.float32)
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

    numeric = v2_plan["numerical_replay_contract"]
    pose_result = evaluate_pose_replay(
        arrays["body_pos_w"],
        recomputed_pos,
        arrays["body_quat_w"],
        recomputed_quat,
        numeric,
    )
    dt = 1.0 / float(v1_plan["l0_contract"]["fps"])
    body_ipos = np.asarray(model.body_ipos, dtype=np.float64)[body_ids]
    reconstructed_com = reconstruct_com_from_stored_pose(
        arrays["body_pos_w"], arrays["body_quat_w"], body_ipos
    )
    lin_result = evaluate_com_linear_velocity(
        arrays["body_lin_vel_w"],
        reconstructed_com,
        body_ipos,
        dt=dt,
        pose_ulp_bins=numeric["link_position"]["max_ulp_bins"],
        contract=numeric["com_linear_velocity"],
    )
    ang_result = evaluate_body_angular_velocity_exact(
        arrays["body_ang_vel_w"],
        arrays["body_quat_w"],
        dt=dt,
        converter=converter,
    )

    support_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in v1_plan["l0_contract"]["support_bodies"]
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
    ground_contract = v1_plan["l0_contract"]["grounding"]
    clearance_result = V1.evaluate_ground_clearance(
        clearances,
        target_m=ground_contract["target_clearance_m"],
        maximum_m=ground_contract["max_grounded_clearance_m"],
        tolerance_m=ground_contract["numerical_tolerance_m"],
    )
    return {
        "model": {
            "canonical_mjcf": V1.binding(mjcf_path),
            "compiled_collision_sha256": model_binding.collision_contract_sha256,
            "enabled_robot_geom_ids": list(model_binding.collision_geom_ids),
            "ground_geom": collision["ground_geom"],
            "ground_z_m": model_binding.ground_z_m,
        },
        "joint_ranges": joint_range_result,
        "kinematic_replay": {
            "frames": frames,
            **pose_result,
            "joint_velocity_float32_byte_equal": True,
            "com_linear_velocity": lin_result,
            "body_angular_velocity": ang_result,
            "mj_step_calls": 0,
        },
        "grounding": {
            **clearance_result,
            "all_lowest_collision_bodies_under_support_feet": True,
            "continuous_time_clearance_proven": False,
        },
    }


def build_certificate(
    v2_plan: Mapping[str, Any], v1_plan: Mapping[str, Any], plan_path: Path, plan_sha: str
) -> dict[str, Any]:
    mujoco, runtime = V1.validate_runtime_environment(v1_plan)
    lineage = V1.validate_upstream_result(v1_plan)
    arrays = V1.load_npz_exact(Path(v1_plan["exact_runtime_inputs"]["motion_npz"]["path"]), v1_plan)
    audit = audit_kinematics_v2(v2_plan, v1_plan, arrays, mujoco)
    return {
        "schema_version": 2,
        "status": CERTIFICATE_STATUS,
        "completed_utc": V1.utc_now(),
        "scope": (
            "CPU-only discrete-frame L0 static certificate for one B schema-2 NPZ under "
            "the V2 numerical replay contract; no simulator step or downstream claim"
        ),
        "asset_id": ASSET_ID,
        "preregistration": {"path": str(plan_path), "sha256": plan_sha},
        "validator": v2_plan["validator"],
        "frozen_v1": v2_plan["frozen_v1"],
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
            "body_quaternion_norm_tolerance": v1_plan["l0_contract"]["quaternion_norm_tolerance"],
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
        "explicit_non_claims": v2_plan["explicit_non_claims"],
        "next_gate": v2_plan["next_gate"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("command", choices=("static", "dry-run", "audit"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        v2_plan, plan_sha, v1_plan = validate_plan(
            args.prereg.resolve(), args.expected_prereg_sha256
        )
        if args.command == "static":
            print(
                f"[motion-l0-v2] PASS static asset={ASSET_ID} source_exact=true "
                "runtime_audit=false no_write=true v1_unchanged=true"
            )
            return 0
        output = Path(v2_plan["output_contract"]["certificate_path"])
        if V1._lexists(output):
            raise L0ContractError(f"certificate path already exists; no-clobber: {output}")
        certificate = build_certificate(v2_plan, v1_plan, args.prereg.resolve(), plan_sha)
        if args.command == "dry-run":
            print(
                f"[motion-l0-v2] PASS dry-run asset={ASSET_ID} runtime_audit=true "
                "certificate_written=false l0_static_complete=false downstream_blocked=true"
            )
            return 0
        V1.write_certificate_exclusive(output, certificate)
        print(
            f"[motion-l0-v2] PASS audit asset={ASSET_ID} l0_static=true "
            f"certificate_sha256={V1.sha256_file(output)} downstream_blocked=true"
        )
        return 0
    except (L0ContractError, OSError, TypeError, ValueError) as exc:
        print(f"[motion-l0-v2] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
