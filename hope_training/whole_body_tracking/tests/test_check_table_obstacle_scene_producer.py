"""Host-only fail-closed tests for the formal Isaac table-smoke producer.

These tests deliberately do not import Isaac Lab or launch Kit.  They cover the
strict input/receipt closure and prove that a host process with no live runtime
origin cannot mint ``PASS``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_table_obstacle_scene.py"
)
SPEC = importlib.util.spec_from_file_location(
    "check_table_obstacle_scene_producer_under_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
P = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P
SPEC.loader.exec_module(P)

ADMISSION_SCRIPT = SCRIPT.with_name("canonical_motion_admission.py")
ADMISSION_SPEC = importlib.util.spec_from_file_location(
    "canonical_motion_admission_isaac_producer_roundtrip_test",
    ADMISSION_SCRIPT,
)
assert ADMISSION_SPEC is not None and ADMISSION_SPEC.loader is not None
ADMISSION = importlib.util.module_from_spec(ADMISSION_SPEC)
sys.modules[ADMISSION_SPEC.name] = ADMISSION
ADMISSION_SPEC.loader.exec_module(ADMISSION)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


A3_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


def _nominal_hold_fixture(tmp_path: Path, *, delay_max_steps: int = 2):
    joint_names = list(A3_JOINT_NAMES)
    source_rows = {}
    for key, payload in (
        ("stable_motion", b"stable-motion-npz-fixture"),
        ("stable_receipt", b'{"stable":true}'),
        ("mujoco_model", b"<mujoco model='a3'/>"),
    ):
        path = tmp_path / f"{key}.bin"
        path.write_bytes(payload)
        source_rows[key] = {
            "path": str(path.resolve()),
            "sha256": _sha(payload),
        }
    runtime_contract = {
        "target_mode": "action_ball",
        "joint_names": joint_names,
        "joint_stiffness": [100.0] * 31,
        "joint_damping": [4.0] * 31,
        "joint_effort_limits": [40.0] * 31,
        "joint_velocity_limits": [12.0] * 31,
        "qdes_joint_pos_limits": [[-1.0, 1.0] for _ in range(31)],
        "default_joint_pos": [0.0] * 31,
        "action_scale": [0.25] * 31,
        "joint_armature": [0.01] * 31,
        "joint_friction_coefficients": [0.02] * 31,
        "finite_projection_soft_envelope_inset_fraction": 0.05,
        "physics_step_dt_s": 0.005,
        "policy_step_dt_s": 0.02,
        "control_decimation": 4,
        "control_step_action_delay": {
            "schema_version": 1,
            "enabled": delay_max_steps > 0,
            "semantic_unit": "policy_control_step",
            "sample_timing": "once_per_episode_reset",
            "distribution": "discrete_uniform_inclusive",
            "min_steps": 0,
            "max_steps": delay_max_steps,
            "shared_across_all_31_joints": True,
            "history_fill": "safe_default_or_action_specific_hold",
        },
    }
    runtime_payload = P._canonical_json_bytes(runtime_contract)
    runtime_path = tmp_path / "runtime_training_contract.json"
    runtime_path.write_bytes(runtime_payload)
    source_rows["runtime_training_contract"] = {
        "path": str(runtime_path.resolve()),
        "sha256": _sha(runtime_payload),
    }
    document = {
        "schema_version": 2,
        "kind": P.NOMINAL_HOLD_ARTIFACT_KIND,
        "action_id": "bh_block",
        "robot": {
            "family": "AgiBot A3",
            "joint_names": joint_names,
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "isaac_nominal_hold_validated": False,
        },
        "sources": source_rows,
        "physical_ready": {
            "root_pos_w_m": [0.0, 0.0, 1.0],
            "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_pos_rad": [0.0] * 31,
            "joint_vel_radps": [0.0] * 31,
        },
        "runtime_plant": {
            "joint_names": joint_names,
            "articulation_joint_names": joint_names,
            "action_joint_ids": list(range(31)),
            "joint_stiffness": [100.0] * 31,
            "joint_damping": [4.0] * 31,
            "joint_effort_limits": [40.0] * 31,
            "joint_velocity_limits": [12.0] * 31,
            "joint_armature": [0.01] * 31,
            "joint_friction_coefficients": [0.02] * 31,
            "qdes_joint_pos_limits": [[-1.0, 1.0] for _ in range(31)],
            "finite_projection_soft_envelope_inset_fraction": 0.05,
            "executed_qdes_lower_rad": [-0.9] * 31,
            "executed_qdes_upper_rad": [0.9] * 31,
            "default_joint_pos_rad": [0.0] * 31,
            "action_scale_rad": [0.25] * 31,
            "physics_step_dt_s": 0.005,
            "policy_step_dt_s": 0.02,
            "control_decimation": 4,
            "control_step_action_delay": runtime_contract[
                "control_step_action_delay"
            ],
        },
        "hold_candidate": {
            "hold_qdes_joint_pos_rad": [0.0] * 31,
            "normalized_actor_action": [0.0] * 31,
        },
        "required_next_gate": {
            "kind": P.NOMINAL_HOLD_RECEIPT_KIND,
        },
    }
    document["content_sha256"] = _sha(P._canonical_json_bytes(document))
    artifact_path = tmp_path / "dynamic_ready.json"
    artifact_path.write_bytes(P._canonical_json_bytes(document))
    return artifact_path, document, runtime_contract


def _split_nominal_hold_fixture(tmp_path: Path):
    artifact_path, document, runtime_contract = _nominal_hold_fixture(tmp_path)
    leg_indices = tuple(
        index
        for index, name in enumerate(A3_JOINT_NAMES)
        if name in P._A3_LEG_JOINT_NAMES
    )
    nonleg_indices = tuple(
        index for index in range(31) if index not in frozenset(leg_indices)
    )
    teacher_q = [0.0] * 31
    physical_q = teacher_q.copy()
    for offset, index in enumerate(leg_indices, start=1):
        physical_q[index] = 0.01 * offset
    teacher_root = [0.0, 0.0, 1.0]
    physical_root = [0.15, -0.18, 1.0684]
    teacher_quat = [1.0, 0.0, 0.0, 0.0]
    physical_quat = [0.9999500004166653, 0.009999833334166664, 0.0, 0.0]
    document["physical_ready"]["joint_pos_rad"] = physical_q
    document["physical_ready"]["root_pos_w_m"] = physical_root
    document["physical_ready"]["root_quat_wxyz"] = physical_quat
    seed = {
        "schema_version": 2,
        "kind": P.NOMINAL_HOLD_ARTIFACT_KIND,
        "content_sha256": "1" * 64,
    }
    seed_path = tmp_path / "physical_birth_seed.json"
    seed_path.write_bytes(P._canonical_json_bytes(seed))
    document["sources"]["physical_birth_seed"] = {
        "path": str(seed_path.resolve()),
        "sha256": _sha(seed_path.read_bytes()),
        "content_sha256": seed["content_sha256"],
        "source_action_id": "bh_loop_c",
        "source_role": "numerical_seed_only",
        "consumed_fields": [
            "physical_ready.root_pos_w_m",
            "physical_ready.root_quat_wxyz",
            "physical_ready.12_leg_joint_pos_rad",
        ],
        "inherited_model_identity": False,
        "inherited_hold_claim": False,
        "inherited_nominal_hold_claim": False,
    }
    document["ready_source"] = {
        "kind": "measured_retarget_l0_diagnostic",
        "frame_index": 0,
        "teacher_reference_unchanged": True,
        "teacher_and_physical_birth_same": False,
        "physical_birth_semantics": (
            "shared_seed_root_leg12_plus_teacher_frame0_nonleg19"
        ),
    }
    document["teacher_reference"] = {
        "semantics": "exact_motion_bytes_frame0_reference",
        "motion_sha256": document["sources"]["stable_motion"]["sha256"],
        "frame_index": 0,
        "root_pos_w_m": teacher_root,
        "root_quat_wxyz": teacher_quat,
        "joint_pos_rad": teacher_q,
    }
    document["physical_birth_composition"] = {
        "semantics": "shared_seed_root_leg12_plus_teacher_frame0_nonleg19",
        "leg_joint_indices": list(leg_indices),
        "leg_joint_names": [A3_JOINT_NAMES[index] for index in leg_indices],
        "nonleg_joint_indices": list(nonleg_indices),
        "nonleg_joint_names": [
            A3_JOINT_NAMES[index] for index in nonleg_indices
        ],
        "teacher_nonleg_exactly_preserved": True,
        "physical_minus_teacher_joint_pos_rad": physical_q,
        "physical_minus_teacher_root_pos_m": [0.15, -0.18, 0.0684],
        "physical_root_quat_wxyz": physical_quat,
        "teacher_root_quat_wxyz": teacher_quat,
        "teacher_and_physical_birth_differ": True,
        "seed_world_yaw_alignment": {
            "schema_version": 1,
            "semantics": P.MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS,
            "seed_root_yaw_rad": 1.476548547,
            "teacher_root_yaw_rad": 0.0,
            "applied_world_z_rotation_rad": -1.476548547,
            "aligned_root_yaw_rad": 0.0,
            "aligned_minus_teacher_yaw_rad": 0.0,
            "support_pivot_xy_w_m": [0.0, -0.18],
            "seed_root_pos_w_m": [0.15, -0.18, 1.0684],
            "aligned_root_pos_w_m": physical_root,
            "seed_root_quat_wxyz": [
                0.731419,
                0.007394,
                -0.006732,
                0.666534,
            ],
            "aligned_root_quat_wxyz": physical_quat,
            "seed_root_tilt_rad": 0.02,
            "aligned_root_tilt_rad": 0.02,
            "expected_aligned_seed_foot_positions_w_m": [
                [-0.12, -0.18, 0.0],
                [0.12, -0.18, 0.0],
            ],
            "support_centroid_preserved": True,
            "seed_tilt_preserved": True,
            "teacher_yaw_exact": True,
            "realized_current_mjcf_fk": {
                "authority": "current_exact_mjcf_fk",
                "semantics": P.MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS,
                "passed": True,
                "absolute_tolerance": 2.0e-10,
                "maximum_foot_position_error_m": 0.0,
                "maximum_foot_rotation_matrix_error": 0.0,
                "support_centroid_xy_error_m": 0.0,
                "maximum_foot_height_error_m": 0.0,
                "expected_foot_positions_w_m": [
                    [-0.12, -0.18, 0.0],
                    [0.12, -0.18, 0.0],
                ],
                "realized_foot_positions_w_m": [
                    [-0.12, -0.18, 0.0],
                    [0.12, -0.18, 0.0],
                ],
            },
        },
    }
    document["physical_birth_static_evidence"] = {
        "authority": "fresh_current_exact_mjcf_reaudit",
        "geometry_passed": True,
        "ground_dynamics_passed": True,
        "gates": {"static_geometry": "PASS", "ground": "PASS"},
        "grounded_ready_receipt_sha256": "2" * 64,
    }
    document.pop("content_sha256")
    document["content_sha256"] = _sha(
        P._canonical_ascii_json_bytes(document)
    )
    artifact_path.write_bytes(P._canonical_json_bytes(document))
    return artifact_path, document, runtime_contract


def _direct_frame0_nominal_hold_fixture(tmp_path: Path):
    artifact_path, document, runtime_contract = _split_nominal_hold_fixture(
        tmp_path
    )
    teacher = document["teacher_reference"]
    document["physical_ready"]["joint_pos_rad"] = list(
        teacher["joint_pos_rad"]
    )
    document["physical_ready"]["root_pos_w_m"] = list(
        teacher["root_pos_w_m"]
    )
    document["physical_ready"]["root_quat_wxyz"] = list(
        teacher["root_quat_wxyz"]
    )
    document["ready_source"]["teacher_and_physical_birth_same"] = True
    document["ready_source"]["physical_birth_semantics"] = (
        P.MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS
    )
    document["physical_birth_composition"] = {
        "semantics": P.MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS,
        "teacher_root_exactly_preserved": True,
        "teacher_all_joints_exactly_preserved": True,
        "physical_minus_teacher_joint_pos_rad": [0.0] * 31,
        "physical_minus_teacher_root_pos_m": [0.0, 0.0, 0.0],
        "physical_root_quat_wxyz": list(teacher["root_quat_wxyz"]),
        "teacher_root_quat_wxyz": list(teacher["root_quat_wxyz"]),
        "teacher_and_physical_birth_differ": False,
        "historical_physical_birth_seed_consumed": False,
        "required_live_table_gate": P.NOMINAL_HOLD_RECEIPT_KIND,
        "current_mjcf_audit_quaternion": {
            "semantics": "unit_normalization_for_numerical_backend_only",
            "stored_teacher_and_physical_quaternion_unchanged": True,
            "stored_quaternion_norm": 1.0,
            "backend_root_quat_wxyz": list(teacher["root_quat_wxyz"]),
        },
    }
    document["sources"].pop("physical_birth_seed")
    selected = {
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    }
    mechanical = [
        [-1.1, 1.1] if name in selected else [-1.0, 1.0]
        for name in A3_JOINT_NAMES
    ]
    control = [
        [-1.056, 1.056] if name in selected else [-1.0, 1.0]
        for name in A3_JOINT_NAMES
    ]
    hctrl = {
        "schema_version": 1,
        "backend": "physx_root_view_dof_limits",
        "inset_fraction_per_side_hard_span": 0.02,
        "selected_joint_names": [
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        ],
        "mechanical_joint_pos_limits": mechanical,
        "control_joint_pos_limits": control,
        "unselected_joint_count": 27,
        "unselected_limits_equal_mechanical": True,
        "articulation_mechanical_ledger_unchanged": True,
        "soft_qdes_ledger_unchanged": True,
    }
    document["runtime_plant"]["physx_control_position_limits"] = hctrl
    runtime_contract["physx_control_position_limits"] = hctrl
    document.pop("content_sha256")
    document["content_sha256"] = _sha(P._canonical_json_bytes(document))
    artifact_path.write_bytes(P._canonical_json_bytes(document))
    return artifact_path, document, runtime_contract


def _whole_body_threshold_frame0_nominal_hold_fixture(tmp_path: Path):
    artifact_path, document, runtime_contract = (
        _direct_frame0_nominal_hold_fixture(tmp_path)
    )
    semantics = P.MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS
    teacher = document["teacher_reference"]
    physical = document["physical_ready"]
    # Real Take061 has this same important shape: stored source bytes are close
    # to unit length, but not bitwise equal to their normalized MuJoCo copy.
    raw_quat = [1.0000000259, 0.0, 0.0, 0.0]
    quat_norm = float(np.linalg.norm(np.asarray(raw_quat, np.float64)))
    audit_quat = [value / quat_norm for value in raw_quat]
    teacher["root_quat_wxyz"] = raw_quat
    teacher["static_handoff_joint_vel_radps"] = [0.0] * 31
    teacher["static_handoff_velocity_semantics"] = (
        "constructed_zero_joint_velocity_endpoint_not_measured_motion_velocity"
    )
    physical["root_quat_wxyz"] = raw_quat
    state_sha = P._whole_body_state_sha256(
        physical["joint_pos_rad"],
        physical["root_pos_w_m"],
        physical["root_quat_wxyz"],
    )
    audit_state_sha = P._whole_body_state_sha256(
        physical["joint_pos_rad"],
        physical["root_pos_w_m"],
        audit_quat,
    )
    handoff = {
        "schema_version": 1,
        "kind": "exact_frame0_zero_duration_handoff_v1",
        "selection_semantics": "threshold_first_exact_frame0_direct",
        "state_sha256_semantics": (
            "float64_array_bytes_without_quaternion_normalization_v1"
        ),
        "physical_ready_state_sha256": state_sha,
        "teacher_frame0_state_sha256": state_sha,
        "mjcf_audit_state_sha256": audit_state_sha,
        "stored_root_quaternion_norm": quat_norm,
        "mjcf_audit_root_quat_wxyz": audit_quat,
        "mjcf_audit_quaternion_semantics": (
            "stored_root_quat_unit_normalized_for_numerical_backend_only"
        ),
        "stored_teacher_and_physical_quaternion_unchanged": True,
        "endpoints_bitwise_equal": True,
        "physical_ready_joint_velocity_exact_zero": True,
        "teacher_static_endpoint_joint_velocity_exact_zero": True,
        "measured_motion_velocity_channels_consumed": False,
        "not_a_motion_velocity_continuity_claim": True,
        "certified_transition_s": 0.0,
        "required_min_wait_s": 0.0,
        "torque_speed_curve_required": False,
        "torque_speed_non_requirement_reason": (
            "identical_stored_configuration_and_constructed_zero_joint_"
            "velocity_endpoints"
        ),
        "runtime_transition_reference_required": False,
        "required_followup_hold_gate": P.NOMINAL_HOLD_RECEIPT_KIND,
        "required_followup_policy_steps": 200,
        "required_followup_physics_steps": 800,
        "diagnostic_unauthorized": True,
        "training_authorized": False,
    }
    safety_slacks = {
        "left_sole_floor_slack_m": 1.0e-3,
        "right_sole_floor_slack_m": 1.0e-3,
        "left_contact_load_slack_n": 9.9,
        "right_contact_load_slack_n": 9.9,
        "support_margin_slack_m": 1.95e-2,
        "joint_position_slack_rad": 1.0,
        "qdes_slack_rad": 0.9,
        "torque_slack_nm": 40.0,
        "table_clearance_slack_m": 2.0e-2,
        "root_height_slack_m": 0.5,
        "root_tilt_slack_rad": 0.7,
        "collision_slack_m": 1.0e-2,
        "ground_lp_residual_slack": 2.0e-7,
    }
    normalized = {
        name: safety_slacks[name] / scale
        for name, scale in P._WHOLE_BODY_SAFETY_SLACK_SCALES.items()
    }
    thresholds = dict(P._WHOLE_BODY_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS)
    threshold_sha = _sha(P._canonical_json_bytes(thresholds))
    optimizer = {
        "algorithm": "exact_measured_frame0_safety_short_circuit",
        "global_optimum_claimed": False,
        "stage1_objective": (
            "prefer_exact_measured_frame0_when_all_safety_gates_pass"
        ),
        "stage2_objective": "not_run_exact_frame0_already_safe",
        "safety_weighted_against_tracking": False,
        "exact_measured_frame0_selected": True,
        "direct_frame0_robust_minimum_slacks": thresholds,
        "stage1_runs": [],
        "stage1_worst_normalized_slack": min(normalized.values()),
        "stage1_lock_tolerance_normalized": 5.0e-5,
        "stage1_locked_worst_normalized_slack": min(normalized.values()),
        "stage2_success": True,
        "stage2_status": 0,
        "stage2_message": "not run; exact measured frame0 is safe",
        "stage2_iterations": 0,
        "stage2_accepted_steps": 0,
        "stage2_objective_value": 0.0,
        "evaluation_count": 4,
        "movable_joint_names": list(A3_JOINT_NAMES),
        "root_degrees_of_freedom": ["z", "roll", "pitch"],
        "slack_scales": dict(P._WHOLE_BODY_SAFETY_SLACK_SCALES),
        "racket_reference_authority": (
            "caller_supplied_independent_measurement"
        ),
    }
    axis = list(P._WHOLE_BODY_MEASURED_RACKET_AXIS_LOCAL)
    racket_reference = {
        "authority": "independent_schema_v4_measured_racket_channel",
        "motion_sha256": document["sources"]["stable_motion"]["sha256"],
        "frame_index": 0,
        "site_pos_w_m": [0.5, 0.0, 1.1],
        "signed_face_normal_w": [0.0, 1.0, 0.0],
        "long_axis_w": axis,
        "position_semantics": "physical_blade_center",
        "normal_semantics": "signed_physical_hitting_face",
        "long_axis_semantics": "measured_paddle_butt_to_blade",
        "robot_mount_normal_sign": 1,
        "robot_butt_to_blade_axis_local": axis,
        "robot_rigid_visual_mesh_sha256": (
            P._WHOLE_BODY_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256
        ),
        "source_sha256": "4" * 64,
        "retarget_receipt_sha256": "5" * 64,
        "input_motion_sha256": "6" * 64,
        "manifest_sha256": "7" * 64,
        "catalog_sha256": "8" * 64,
        "source_and_retarget_receipt_sha_semantics": (
            "opaque_labels_content_bound_by_exact_materialized_motion_sha"
        ),
        "signed_face_normal_w_unit": [0.0, 1.0, 0.0],
        "long_axis_w_unit": axis,
        "official_site_rotation_w": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    racket_fidelity = {
        "site_name": "right_racket",
        "site_semantics": (
            "official_mjcf_site_against_independent_schema_v4_measured_blade"
        ),
        "reference_authority": racket_reference,
        "physical_site_pos_w_m": [0.5, 0.0, 1.1],
        "physical_signed_face_normal_w": [0.0, 1.0, 0.0],
        "physical_long_axis_w": axis,
        "physical_minus_measured_position_w_m": [0.0, 0.0, 0.0],
        "position_error_m": 0.0,
        "physical_minus_measured_rotation_vector_rad": [0.0, 0.0, 0.0],
        "orientation_error_rad": 0.0,
        "signed_face_error_rad": 0.0,
        "long_axis_error_rad": 0.0,
        "independent_measured_frame0_required": True,
    }
    document["ready_source"].update(
        {
            "kind": "measured_retarget_l0_diagnostic",
            "frame_index": 0,
            "teacher_reference_unchanged": True,
            "teacher_and_physical_birth_same": True,
            "physical_birth_semantics": semantics,
            "plant_template_action_binding_consumed": False,
            "plant_template_delay_overridden_to_zero": True,
            "isaac_live_plant_match_required": True,
            "diagnostic_unauthorized": True,
            "training_authorized": False,
        }
    )
    document["physical_birth_composition"] = {
        "semantics": semantics,
        "teacher_reference_unchanged": True,
        "historical_physical_birth_seed_consumed": False,
        "vendor_key_used_as_optimizer_start_only": True,
        "selection_priority": [
            "exact_measured_frame0_if_all_safety_gates_pass",
            "lexicographic_whole_body_safe_ready_only_if_frame0_unsafe",
        ],
        "exact_measured_frame0_selected": True,
        "released_root_degrees_of_freedom": ["z", "roll", "pitch"],
        "released_joint_indices": list(range(31)),
        "released_joint_names": list(A3_JOINT_NAMES),
        "changed_joint_mask": [False] * 31,
        "changed_joint_indices": [],
        "changed_joint_names": [],
        "physical_minus_teacher_joint_pos_rad": [0.0] * 31,
        "physical_minus_teacher_joint_pos_by_name_rad": {
            name: 0.0 for name in A3_JOINT_NAMES
        },
        "physical_minus_teacher_root_pos_m": [0.0, 0.0, 0.0],
        "physical_minus_teacher_root_rotation_vector_rad": [0.0, 0.0, 0.0],
        "physical_root_quat_wxyz": list(teacher["root_quat_wxyz"]),
        "stored_physical_root_quat_wxyz": list(teacher["root_quat_wxyz"]),
        "mjcf_audit_root_quat_wxyz": audit_quat,
        "teacher_root_quat_wxyz": list(teacher["root_quat_wxyz"]),
        "teacher_and_physical_birth_differ": False,
        "racket_site_fidelity": racket_fidelity,
        "safety_slacks": safety_slacks,
        "normalized_safety_slacks": normalized,
        "worst_normalized_safety_slack": min(normalized.values()),
        "stage1_locked_worst_normalized_safety_slack": min(
            normalized.values()
        ),
        "optimizer_report": optimizer,
        "evaluator_contract": {},
        "safety_weighted_against_tracking": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "required_live_table_gate": P.NOMINAL_HOLD_RECEIPT_KIND,
        "frame0_handoff": handoff,
    }
    contact_normals = [10.0] * 6
    solver_report = {
        "model_binding": "3" * 64,
        "exact_state_lp_cache_hit": False,
        "normal_force_per_foot_n": [30.0, 30.0],
        "normal_force_per_contact_n": contact_normals,
        "cop_interior_margin_per_foot_m": [0.02, 0.02],
        "contact_geometry": {
            "feet": [
                {"support_point_range": [0, 3]},
                {"support_point_range": [3, 6]},
            ]
        },
    }
    collision_authority = {
        "self_collision_geom_id_pairs": [[1, 2]],
        "unsupported_floor_robot_geom_ids": [],
        "expected_foot_floor_geom_ids": [3, 4],
        "floor_geom_id": 0,
    }
    collision = {
        **collision_authority,
        "pair_authority_sha256": _sha(
            P._canonical_json_bytes(collision_authority)
        ),
        "enabled_self_pair_count": 1,
        "unsupported_floor_pair_count": 0,
        "required_clearance_m": 2.0e-3,
        "capped_clearance_m": 2.0e-2,
        "bisection_tolerance_m": 1.0e-4,
        "distance_semantics": (
            "mujoco_geomDistance_saturation_bisection_with_robot_pair_"
            "sphere_lower_bound_pruning"
        ),
        "realized_capped_minimum_clearance_m": 1.2e-2,
        "raw_bisection_midpoint_or_saturated_cap_m": 1.21e-2,
        "positive_unsaturated_conservative_deduction_m": 1.0e-4,
        "realized_slack_m": 1.0e-2,
        "unsupported_contacts": [],
        "self_collision_pairs": [],
    }
    tau_lower = [-40.0] * 31
    tau_upper = [40.0] * 31
    witness = {
        "exact_contact_lp_reused": True,
        "lp_feasible": True,
        "lp_error": None,
        "lp_objective": "hold_minimax_normalized_available_torque",
        "equality_residual": 0.0,
        "root_residual": 0.0,
        "normal_force_per_foot_n": [30.0, 30.0],
        "normal_force_per_contact_n": contact_normals,
        "minimum_normal_force_per_contact_per_foot_n": [10.0, 10.0],
        "required_minimum_normal_force_per_contact_n": 0.1,
        "required_minimum_normal_force_per_foot_n": 1.0,
        "cop_interior_margin_per_foot_m": [0.02, 0.02],
        "global_support_margin_m": 0.02,
        "support_hull_floor_xy_m": [
            [-0.1, -0.1],
            [0.1, -0.1],
            [0.1, 0.1],
            [-0.1, 0.1],
        ],
        "hold_qdes_joint_pos_rad": [0.0] * 31,
        "actuator_generalized_force_runtime_order_nm": [0.0] * 31,
        "actuator_generalized_force_mujoco_row_order_nm": [0.0] * 31,
        "mujoco_row_for_runtime_joint": list(range(31)),
        "mujoco_actuated_dof_indices": list(range(6, 37)),
        "executed_qdes_lower_rad": [-0.9] * 31,
        "executed_qdes_upper_rad": [0.9] * 31,
        "model_tau_lower_mujoco_row_order_nm": tau_lower,
        "model_tau_upper_mujoco_row_order_nm": tau_upper,
        "runtime_tau_lower_runtime_order_nm": tau_lower,
        "runtime_tau_upper_runtime_order_nm": tau_upper,
        "runtime_tau_lower_mujoco_row_order_nm": tau_lower,
        "runtime_tau_upper_mujoco_row_order_nm": tau_upper,
        "effective_tau_lower_mujoco_row_order_nm": tau_lower,
        "effective_tau_upper_mujoco_row_order_nm": tau_upper,
        "actuator_limit_contract": {},
        "solver_report": solver_report,
        "exact_state_lp_cache_hit": False,
        "evaluated_state_sha256": audit_state_sha,
        "evaluated_joint_pos_rad": [0.0] * 31,
        "evaluated_root_pos_w_m": [0.0, 0.0, 1.0],
        "evaluated_root_quat_wxyz": audit_quat,
        "sole_minimum_distance_m": [1.0e-3, -1.0e-3],
        "exact_joint_position_lower_rad": [-1.0] * 31,
        "exact_joint_position_upper_rad": [1.0] * 31,
        "conservative_table_clearance_m": 3.0e-2,
        "table_geometry": {
            "near_x_m": 0.8,
            "half_width_m": 0.7625,
            "surface_z_m": 0.76,
            "required_clearance_m": 1.0e-2,
            "semantics": (
                "collision_sphere_separation_from_overapproximated_near_side_table_prism"
            ),
        },
        "root_limits": {
            "minimum_height_m": 0.5,
            "maximum_tilt_rad": 0.7,
        },
        "collision_clearance": collision,
    }
    document["physical_birth_composition"]["evaluator_contract"] = {
        "executed_qdes_lower_rad": [-0.9] * 31,
        "executed_qdes_upper_rad": [0.9] * 31,
        "exact_joint_position_lower_rad": [-1.0] * 31,
        "exact_joint_position_upper_rad": [1.0] * 31,
        "table_near_x_m": 0.8,
        "table_half_width_m": 0.7625,
        "table_surface_z_m": 0.76,
        "minimum_table_clearance_m": 1.0e-2,
        "minimum_root_height_m": 0.5,
        "maximum_root_tilt_rad": 0.7,
        "collision_pair_authority": {
            name: collision[name]
            for name in (
                "self_collision_geom_id_pairs",
                "unsupported_floor_robot_geom_ids",
                "expected_foot_floor_geom_ids",
                "floor_geom_id",
                "pair_authority_sha256",
                "enabled_self_pair_count",
                "unsupported_floor_pair_count",
                "required_clearance_m",
                "capped_clearance_m",
                "bisection_tolerance_m",
                "distance_semantics",
            )
        },
    }
    document["physical_birth_static_evidence"] = {
        "authority": "fresh_current_exact_mjcf_whole_body_lexicographic_search",
        "selected_hold_witness_authority": (
            "new_backend_new_solver_final_state_cache_miss"
        ),
        "exact_contact_lp_reused": False,
        "all_safety_slacks_meet_original_and_locked_gate": True,
        "required_final_normalized_safety_gate": min(normalized.values()),
        "direct_frame0_robust_minimum_slacks": thresholds,
        "direct_frame0_robust_gate_sha256": threshold_sha,
        "fresh_direct_robust_gate_passed": True,
        "safety_slacks": safety_slacks,
        "normalized_safety_slacks": normalized,
        "evaluator_evidence": witness,
        "stored_endpoint_state_sha256": state_sha,
        "mjcf_audit_state_sha256": audit_state_sha,
        "stored_root_quat_wxyz": raw_quat,
        "mjcf_audit_root_quat_wxyz": audit_quat,
        "stored_root_quaternion_norm": quat_norm,
        "independent_measured_racket_frame0": racket_reference,
        "racket_site_fidelity": racket_fidelity,
        "frame0_handoff": handoff,
        "optimizer_report": optimizer,
        "geometry_passed": True,
        "ground_dynamics_passed": True,
    }
    document["frame0_handoff"] = handoff
    document["sources"]["mujoco_model"][
        "ground_model_binding_sha256"
    ] = "3" * 64
    document["hold_candidate"].update(
        {
            "semantics": (
                "tau_pd=kp*(qdes-physical_q) at zero joint velocity; "
                "the new-backend cache-miss whole-body final-state LP is the "
                "single selected witness; Isaac must validate it"
            ),
            "hold_qdes_mode": "fresh_static_lp",
            "selected_hold_authority": {
                "semantics": (
                    "fresh_new_backend_whole_body_final_state_0p1n_static_lp"
                ),
                "source_physical_birth_seed_sha256": None,
                "inherited_hold_claim": False,
            },
            "lp_objective": "hold_minimax_normalized_available_torque",
            "actuator_generalized_force_runtime_order_nm": [0.0] * 31,
            "actuator_generalized_force_mujoco_row_order_nm": [0.0] * 31,
            "mujoco_row_for_runtime_joint": list(range(31)),
            "mujoco_actuated_dof_indices": list(range(6, 37)),
            "model_tau_lower_mujoco_row_order_nm": tau_lower,
            "model_tau_upper_mujoco_row_order_nm": tau_upper,
            "runtime_tau_lower_runtime_order_nm": tau_lower,
            "runtime_tau_upper_runtime_order_nm": tau_upper,
            "runtime_tau_lower_mujoco_row_order_nm": tau_lower,
            "runtime_tau_upper_mujoco_row_order_nm": tau_upper,
            "effective_tau_lower_mujoco_row_order_nm": tau_lower,
            "effective_tau_upper_mujoco_row_order_nm": tau_upper,
            "actuator_limit_contract": {},
            "solver_report": solver_report,
            "solver_report_role": (
                "selected_whole_body_final_state_single_witness"
            ),
        }
    )
    document["required_next_gate"] = {
        "kind": P.NOMINAL_HOLD_RECEIPT_KIND,
        "required_policy_steps": 200,
        "required_physics_steps": 800,
        "required_min_wait_s": 0.0,
        "minimum_horizon_semantics": "validated_t_hit_plus_reaction_margin",
        "zero_terminal_required": [
            "joint_qdes_forbidden",
            "joint_actual_forbidden",
            "robot_hit_table",
            "base_fell_tilt",
            "base_too_low",
        ],
    }
    document.pop("content_sha256")
    document["content_sha256"] = _sha(
        P._canonical_ascii_json_bytes(document)
    )
    artifact_path.write_bytes(P._canonical_json_bytes(document))
    return artifact_path, document, runtime_contract


def _rewrite_dynamic_ready(path: Path, document: dict) -> None:
    document.pop("content_sha256", None)
    document["content_sha256"] = _sha(
        P._canonical_ascii_json_bytes(document)
    )
    path.write_bytes(P._canonical_json_bytes(document))


def _fixture_tree(tmp_path: Path, action_count: int = 5):
    source = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "check_table_obstacle_scene.py"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# committed producer fixture\n")
    solver_source_dir = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
    )
    solver_source_dir.mkdir(parents=True)
    solver_source_sha = {}
    for name in P._ACTION_BALL_SOLVER_SOURCE_NAMES:
        path = solver_source_dir / name
        path.write_bytes(f"# exact solver fixture: {name}\n".encode("ascii"))
        solver_source_sha[name] = _sha(path.read_bytes())
    contact_geometry_payload = {
        "schema_version": 2,
        "semantics": "fixture canonical exact-face geometry",
    }
    contact_geometry = {
        "payload": contact_geometry_payload,
        "sha256": _sha(P._canonical_json_bytes(contact_geometry_payload)),
    }
    # Solver profile v3: the sealed payload binds the per-symbol semantic
    # surface; the byte map above stays in the document as provenance.
    semantic_surface = {
        "kind": "whole_body_tracking.action_ball.solver_semantic_surface",
        "schema_version": 1,
        "sha256": _sha(b"fixture semantic surface"),
    }
    solver_payload = {
        "kind": "fixture.frozen_ball_to_task_solver",
        "semantic_surface": semantic_surface,
        "contact_geometry": contact_geometry,
    }
    physics_payload = {
        "kind": "fixture.action_ball_physics",
        "geometry_and_grading": {
            "table_surface_z_m": 0.76,
            "ball_center_net_top_z_m": 0.9325,
            "net_x_m": 1.87,
            "opponent_near_x_m": 0.5,
            "opponent_far_x_m": 3.24,
            "minimum_landing_depth_m": 0.3,
            "table_half_width_m": 0.7625,
        },
    }
    solver_profile_sha = _sha(P._canonical_json_bytes(solver_payload))
    physics_profile_sha = _sha(P._canonical_json_bytes(physics_payload))
    profile_pins = {
        "solver_payload": solver_payload,
        "physics_payload": physics_payload,
        "solver_profile_sha256": solver_profile_sha,
        "physics_profile_sha256": physics_profile_sha,
        "solver_implementation_source_sha256": solver_source_sha,
        "solver_semantic_surface": {"sha256": semantic_surface["sha256"]},
        "contact_geometry": contact_geometry,
    }
    profile_path = tmp_path / "profile_pins.json"
    profile_path.write_text(
        json.dumps(
            profile_pins,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    geometry_source = solver_source_dir / "racket_contact_geometry.py"
    geometry_contract = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": geometry_source.relative_to(tmp_path).as_posix(),
        "source_sha256": _sha(geometry_source.read_bytes()),
        "geometry_source_sha256": contact_geometry["sha256"],
    }
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    action_ids = (
        P.FRESH_N5_ACTION_IDS
        if action_count == 5
        else tuple(f"fixture_action_{index:03d}" for index in range(action_count))
    )
    families = (
        (
            "backhand",
            "forehand",
            "backhand",
            "backhand",
            "forehand",
        )
        if action_count == 5
        else tuple(
            "backhand" if index % 2 == 0 else "forehand"
            for index in range(action_count)
        )
    )
    signs = tuple(
        -1 if family == "backhand" else 1 for family in families
    )
    actions = []
    for index, (motion_id, family, sign) in enumerate(
        zip(action_ids, families, signs)
    ):
        path = motion_dir / f"{motion_id}.npz"
        payload = f"exact-motion-{index}-{motion_id}".encode("ascii")
        path.write_bytes(payload)
        actions.append(
            {
                "action_id": motion_id,
                "action_uid": P._derive_action_uid(
                    motion_id, family, _sha(payload)
                ),
                "family": family,
                "motion_path": path.relative_to(tmp_path).as_posix(),
                "motion_sha256": _sha(payload),
                "strike_phase": 0.5,
                "reference_t_hit_s": 0.5,
                "reference_t_cycle_s": 1.0,
                "reference_racket_site_speed_mps": 2.0,
                "reaction_margin_s": 0.1,
                "teacher_rate_min": 0.5,
                "teacher_rate_max": 1.0,
                "mount_normal_sign": sign,
                "ball_profile": {
                    "time_to_contact_center_s": 1.2,
                },
            }
        )
    manifest = {
        "schema_version": 3,
        "manifest_id": f"producer-fixture-n{action_count}",
        "mobility_mode": "no_move",
        "action_order": list(action_ids),
        "prototype": {
            "scope": "full" if action_count == 73 else "upper"
        },
        "solver_profile_sha256": solver_profile_sha,
        "physics_profile_sha256": physics_profile_sha,
        "actions": actions,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, manifest_path, manifest


def _load_fixture_inputs(
    source: Path,
    manifest_path: Path,
    *,
    repo_root: Path,
):
    profile_path = manifest_path.with_name("profile_pins.json")
    return _load_formal_fixture(
        manifest_path.relative_to(repo_root).as_posix(),
        profile_pins_value=profile_path.relative_to(repo_root).as_posix(),
        expected_profile_pins_sha256=_sha(profile_path.read_bytes()),
        repo_root=repo_root,
        source_path=source,
    )


def _fixture_action_set(repo_root: Path, manifest_path: Path) -> dict:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {
        row["action_id"]: row
        for row in document["actions"]
    }
    if len(document["actions"]) == 5:
        ids = list(P.FRESH_N5_ACTION_IDS)
    else:
        ids = [row["action_id"] for row in document["actions"]]
    uids = [by_id[action_id]["action_uid"] for action_id in ids]
    action_count = len(ids)
    profile_id = (
        "fresh_upper_nomove_n5_v3"
        if action_count == 5
        else f"fixture_n{action_count}"
    )
    row = {
        "profile_id": profile_id,
        "expected_n": action_count,
        "scope": "full" if action_count == 73 else "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": ids,
        "ordered_action_uids": uids,
        "order_uid_digest_sha256": P.action_set_contract.order_uid_digest(
            ids, uids
        ),
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "manifest_sha256": _sha(manifest_path.read_bytes()),
        "experiment_name": f"fixture_n{action_count}",
    }
    # These receipt-schema fixtures deliberately exercise future N5/N73
    # identities.  The production v2 actor contract is correctly N=1-only, so
    # give only this fixture validation a non-v2 future-contract placeholder.
    original_actor_obs_contract = P.action_set_contract.ACTOR_OBS_CONTRACT
    P.action_set_contract.ACTOR_OBS_CONTRACT = (
        "fixture_content_derived_future_motion_intent_v1"
    )
    try:
        return P.action_set_contract.validate_contract(
            row, profile_id=profile_id, profile_policies={}
        )
    finally:
        P.action_set_contract.ACTOR_OBS_CONTRACT = original_actor_obs_contract


def _load_formal_fixture(
    manifest_value,
    *,
    profile_pins_value,
    expected_profile_pins_sha256,
    repo_root,
    source_path,
):
    manifest_path = repo_root / manifest_value
    trusted = _fixture_action_set(repo_root, manifest_path)
    original = P._load_trusted_action_set
    P._load_trusted_action_set = lambda profile_id: trusted
    try:
        return P._load_formal_inputs(
            manifest_value,
            action_set_profile=trusted["profile_id"],
            profile_pins_value=profile_pins_value,
            expected_profile_pins_sha256=expected_profile_pins_sha256,
            repo_root=repo_root,
            source_path=source_path,
        )
    finally:
        P._load_trusted_action_set = original


def _commit_fixture(repo_root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "producer-test@example.invalid"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Producer Test"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"], cwd=repo_root, check=True
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _valid_evidence(inputs):
    token = object()
    P._ISAAC_RUNTIME_ORIGIN = token
    action_rows = tuple(
        P._RuntimeActionEvidence(
            motion_id=row.motion_id,
            action_uid=row.action_uid,
            motion_sha256=row.file.sha256,
            frame_count=3,
            physics_steps=12,
            complete_cycle=True,
            table_contact_count=0,
            fall_count=0,
            hard_limit_count=0,
            unsafe_count=0,
        )
        for row in inputs.motions
    )
    return P._RuntimeEvidence(
        origin=token,
        source_commit_sha="1" * 40,
        isaac_version="isaaclab=fixture",
        python_executable=sys.executable,
        gpu_identity={
            "physical_index": 2,
            "logical_index": 0,
            "cuda_visible_devices": "2",
            "gpu_uuid": "GPU-fixture",
            "gpu_name": "Fixture GPU",
            "driver_version": "fixture-driver",
            "nvml_verified": True,
        },
        physics_steps=sum(row.physics_steps for row in action_rows),
        actions=action_rows,
        pose_obb_guard_pass=True,
        full_action_ball_assembly=True,
        all_five_table_components_with_pose_obb=True,
        all_five_obstacles=True,
        all_four_substeps=True,
        positive_control_pass=True,
        negative_control_pass=True,
        zero_reset_leakage=True,
    )


def test_host_import_does_not_launch_kit_or_create_runtime_origin():
    assert P._app is None
    assert P._ISAAC_RUNTIME_ORIGIN is None
    assert P.gym is None
    assert P.torch is None


def test_real_task_id_is_default_and_retired_fake_id_is_rejected():
    args = P._parse([])
    assert args.task == "HOPE-PingPong-ActionBall-AgibotA3-v0"
    P._validate_cli_mode(args)

    fake = P._parse(
        ["--task", "Tracking-Flat-AgibotA3-Hope-ActionBall-v0"]
    )
    with pytest.raises(P.TableSmokeReceiptError, match="retired fake task id"):
        P._validate_cli_mode(fake)


def test_nominal_hold_cli_is_opt_in_one_env_and_pinned():
    args = P._parse(
        [
            "--num-envs",
            "1",
            "--device",
            "cuda:1",
            "--nominal-hold",
            "/tmp/ready.json",
            "--nominal-hold-sha256",
            "1" * 64,
            "--nominal-hold-receipt-out",
            "/tmp/hold.json",
        ]
    )
    P._validate_cli_mode(args)
    with pytest.raises(P.TableSmokeReceiptError, match="requires one explicit cuda:N"):
        P._validate_cli_mode(
            P._parse(
                [
                    "--num-envs",
                    "2",
                    "--nominal-hold",
                    "/tmp/ready.json",
                    "--nominal-hold-sha256",
                    "1" * 64,
                    "--nominal-hold-receipt-out",
                    "/tmp/hold.json",
                ]
            )
        )


def test_nominal_hold_artifact_pins_a3_motion_and_core_plant(tmp_path):
    path, document, _contract = _nominal_hold_fixture(tmp_path)
    loaded = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )
    assert loaded.action_id == "bh_block"
    assert loaded.joint_names == tuple(document["robot"]["joint_names"])
    assert loaded.motion_sha256 == document["sources"]["stable_motion"]["sha256"]
    assert loaded.expected_plant["control_decimation"] == 4
    assert loaded.expected_plant["control_step_action_delay"]["max_steps"] == 2

    document["hold_candidate"]["hold_qdes_joint_pos_rad"][0] = "nan"
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    document["content_sha256"] = _sha(P._canonical_json_bytes(unsigned))
    path.write_bytes(P._canonical_json_bytes(document))
    with pytest.raises(P.TableSmokeReceiptError, match="hold q_des"):
        P._load_nominal_hold_input(
            path, expected_sha256=_sha(path.read_bytes())
        )


def test_nominal_hold_separates_exact_teacher_from_composed_physical_birth(
    tmp_path,
):
    path, document, _contract = _split_nominal_hold_fixture(tmp_path)
    loaded = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )
    assert loaded.teacher_physical_separated is True
    assert loaded.teacher_joint_pos == (0.0,) * 31
    assert loaded.teacher_root_pos == (0.0, 0.0, 1.0)
    assert loaded.physical_joint_pos != loaded.teacher_joint_pos
    assert loaded.physical_root_pos != loaded.teacher_root_pos
    assert document["ready_source"].get(
        "original_motion_frame0_preserved"
    ) is None

    stolen = json.loads(json.dumps(document))
    nonleg_index = stolen["physical_birth_composition"][
        "nonleg_joint_indices"
    ][0]
    stolen["physical_ready"]["joint_pos_rad"][nonleg_index] = 0.25
    stolen.pop("content_sha256")
    stolen["content_sha256"] = _sha(P._canonical_json_bytes(stolen))
    path.write_bytes(P._canonical_json_bytes(stolen))
    with pytest.raises(
        P.TableSmokeReceiptError,
        match="differs from recorded teacher delta",
    ):
        P._load_nominal_hold_input(
            path, expected_sha256=_sha(path.read_bytes())
        )


def test_nominal_hold_rejects_physical_birth_yaw_drift(tmp_path):
    path, document, _contract = _split_nominal_hold_fixture(tmp_path)
    drifted = json.loads(json.dumps(document))
    yaw = 0.1
    drifted_quat = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    drifted["physical_ready"]["root_quat_wxyz"] = drifted_quat
    drifted["physical_birth_composition"][
        "physical_root_quat_wxyz"
    ] = drifted_quat
    drifted["physical_birth_composition"]["seed_world_yaw_alignment"][
        "aligned_root_quat_wxyz"
    ] = drifted_quat
    drifted.pop("content_sha256")
    drifted["content_sha256"] = _sha(P._canonical_json_bytes(drifted))
    path.write_bytes(P._canonical_json_bytes(drifted))

    with pytest.raises(
        P.TableSmokeReceiptError,
        match="not an exact teacher-yaw-aligned seed",
    ):
        P._load_nominal_hold_input(
            path, expected_sha256=_sha(path.read_bytes())
        )


def test_nominal_hold_accepts_full_seed_birth_with_exact_teacher(tmp_path):
    path, document, _contract = _split_nominal_hold_fixture(tmp_path)
    full_seed = json.loads(json.dumps(document))
    semantics = P.MEASURED_BIRTH_FULL_SEED_SEMANTICS
    composition = full_seed["physical_birth_composition"]
    composition["semantics"] = semantics
    composition["teacher_nonleg_exactly_preserved"] = False
    composition["seed_all_joints_exactly_preserved"] = True
    composition["seed_joint_indices"] = list(range(31))
    composition["seed_joint_names"] = list(A3_JOINT_NAMES)
    nonleg_index = composition["nonleg_joint_indices"][0]
    full_seed["physical_ready"]["joint_pos_rad"][nonleg_index] = 0.2
    composition["physical_minus_teacher_joint_pos_rad"][nonleg_index] = 0.2
    full_seed["ready_source"]["physical_birth_semantics"] = semantics
    full_seed["sources"]["physical_birth_seed"]["consumed_fields"][-1] = (
        "physical_ready.31_joint_pos_rad"
    )
    full_seed.pop("content_sha256")
    full_seed["content_sha256"] = _sha(P._canonical_json_bytes(full_seed))
    path.write_bytes(P._canonical_json_bytes(full_seed))

    loaded = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )
    assert loaded.teacher_physical_separated is True
    assert loaded.teacher_joint_pos[nonleg_index] == 0.0
    assert loaded.physical_joint_pos[nonleg_index] == 0.2


def test_nominal_hold_accepts_direct_frame0_and_requires_vendor_hctrl(tmp_path):
    path, document, _contract = _direct_frame0_nominal_hold_fixture(tmp_path)
    loaded = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )
    assert loaded.teacher_physical_separated is False
    assert loaded.physical_joint_pos == loaded.teacher_joint_pos
    assert loaded.physical_root_pos == loaded.teacher_root_pos
    assert loaded.physical_root_quat == loaded.teacher_root_quat
    assert loaded.expected_plant["physx_control_position_limits"][
        "inset_fraction_per_side_hard_span"
    ] == 0.02

    missing = json.loads(json.dumps(document))
    missing["runtime_plant"].pop("physx_control_position_limits")
    missing.pop("content_sha256")
    missing["content_sha256"] = _sha(P._canonical_json_bytes(missing))
    path.write_bytes(P._canonical_json_bytes(missing))
    with pytest.raises(
        P.TableSmokeReceiptError,
        match="requires exact Vendor PhysX H_ctrl",
    ):
        P._load_nominal_hold_input(
            path, expected_sha256=_sha(path.read_bytes())
        )


def test_direct_frame0_rejects_any_teacher_physical_delta(tmp_path):
    path, document, _contract = _direct_frame0_nominal_hold_fixture(tmp_path)
    drifted = json.loads(json.dumps(document))
    drifted["physical_ready"]["joint_pos_rad"][0] = 1.0e-4
    drifted.pop("content_sha256")
    drifted["content_sha256"] = _sha(P._canonical_json_bytes(drifted))
    path.write_bytes(P._canonical_json_bytes(drifted))
    with pytest.raises(
        P.TableSmokeReceiptError,
        match="direct measured frame0 physical-birth authority is invalid",
    ):
        P._load_nominal_hold_input(
            path, expected_sha256=_sha(path.read_bytes())
        )


def test_nominal_hold_accepts_only_threshold_first_whole_body_frame0(
    tmp_path,
):
    path, document, _contract = (
        _whole_body_threshold_frame0_nominal_hold_fixture(tmp_path)
    )
    loaded = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )
    assert loaded.teacher_physical_separated is False
    assert loaded.physical_joint_pos == loaded.teacher_joint_pos
    assert loaded.physical_root_pos == loaded.teacher_root_pos
    assert loaded.physical_root_quat == loaded.teacher_root_quat
    assert document["frame0_handoff"]["certified_transition_s"] == 0.0
    assert document["required_next_gate"]["required_policy_steps"] == 200
    assert document["required_next_gate"]["required_physics_steps"] == 800
    assert "physical_birth_seed" not in document["sources"]

    unicode_root = tmp_path / "乒乓"
    unicode_root.mkdir()
    unicode_path, _unicode_document, _unicode_contract = (
        _whole_body_threshold_frame0_nominal_hold_fixture(unicode_root)
    )
    unicode_loaded = P._load_nominal_hold_input(
        unicode_path, expected_sha256=_sha(unicode_path.read_bytes())
    )
    assert unicode_loaded.teacher_physical_separated is False

    missing_hctrl = json.loads(json.dumps(document))
    missing_hctrl["runtime_plant"].pop("physx_control_position_limits")
    _rewrite_dynamic_ready(path, missing_hctrl)
    with pytest.raises(
        P.TableSmokeReceiptError,
        match="whole-body frame0 hold requires exact Vendor PhysX H_ctrl",
    ):
        P._load_nominal_hold_input(
            path, expected_sha256=_sha(path.read_bytes())
        )

    qdes_outside_hctrl = json.loads(json.dumps(document))
    qdes_outside_hctrl["runtime_plant"]["qdes_joint_pos_limits"][0] = [
        -1.2,
        1.2,
    ]
    _rewrite_dynamic_ready(path, qdes_outside_hctrl)
    with pytest.raises(
        P.TableSmokeReceiptError,
        match="qdes envelope must remain inside exact Vendor PhysX H_ctrl",
    ):
        P._load_nominal_hold_input(
            path, expected_sha256=_sha(path.read_bytes())
        )


def test_whole_body_frame0_rejects_fallback_or_tampered_handoff(tmp_path):
    path, document, _contract = (
        _whole_body_threshold_frame0_nominal_hold_fixture(tmp_path)
    )
    variants = []

    fallback = json.loads(json.dumps(document))
    fallback["physical_birth_composition"][
        "exact_measured_frame0_selected"
    ] = False
    fallback["physical_birth_composition"]["optimizer_report"][
        "algorithm"
    ] = "two_stage_deterministic_coordinate_local_lexicographic"
    fallback["physical_birth_static_evidence"]["optimizer_report"] = fallback[
        "physical_birth_composition"
    ]["optimizer_report"]
    variants.append(fallback)

    missing = json.loads(json.dumps(document))
    missing.pop("frame0_handoff")
    variants.append(missing)

    transition = json.loads(json.dumps(document))
    transition["frame0_handoff"]["certified_transition_s"] = 0.01
    variants.append(transition)

    wrong_state_sha = json.loads(json.dumps(document))
    for location in (
        wrong_state_sha,
        wrong_state_sha["physical_birth_composition"],
        wrong_state_sha["physical_birth_static_evidence"],
    ):
        location["frame0_handoff"]["physical_ready_state_sha256"] = "9" * 64
        location["frame0_handoff"]["teacher_frame0_state_sha256"] = "9" * 64
    variants.append(wrong_state_sha)

    moved_physical = json.loads(json.dumps(document))
    moved_physical["physical_ready"]["joint_pos_rad"][0] = 1.0e-5
    variants.append(moved_physical)

    moving_teacher_endpoint = json.loads(json.dumps(document))
    moving_teacher_endpoint["teacher_reference"][
        "static_handoff_joint_vel_radps"
    ][0] = 1.0e-5
    variants.append(moving_teacher_endpoint)

    wrong_gate = json.loads(json.dumps(document))
    wrong_gate["required_next_gate"]["required_policy_steps"] = 199
    variants.append(wrong_gate)

    for variant in variants:
        _rewrite_dynamic_ready(path, variant)
        with pytest.raises(P.TableSmokeReceiptError, match="threshold-first"):
            P._load_nominal_hold_input(
                path, expected_sha256=_sha(path.read_bytes())
            )


def test_whole_body_frame0_rejects_tampered_fresh_evidence_or_seed(tmp_path):
    path, document, _contract = (
        _whole_body_threshold_frame0_nominal_hold_fixture(tmp_path)
    )
    variants = []

    robust_hash = json.loads(json.dumps(document))
    robust_hash["physical_birth_static_evidence"][
        "direct_frame0_robust_gate_sha256"
    ] = "0" * 64
    variants.append(robust_hash)

    robust_pass = json.loads(json.dumps(document))
    robust_pass["physical_birth_static_evidence"][
        "fresh_direct_robust_gate_passed"
    ] = False
    variants.append(robust_pass)

    reused_lp = json.loads(json.dumps(document))
    reused_lp["physical_birth_static_evidence"]["evaluator_evidence"][
        "exact_state_lp_cache_hit"
    ] = True
    variants.append(reused_lp)

    residual = json.loads(json.dumps(document))
    residual["physical_birth_static_evidence"]["evaluator_evidence"][
        "equality_residual"
    ] = 1.0
    variants.append(residual)

    sole_distance = json.loads(json.dumps(document))
    sole_distance["physical_birth_static_evidence"]["evaluator_evidence"][
        "sole_minimum_distance_m"
    ][0] = 1.0e-2
    variants.append(sole_distance)

    joint_limits = json.loads(json.dumps(document))
    joint_limits["physical_birth_static_evidence"]["evaluator_evidence"][
        "exact_joint_position_lower_rad"
    ][0] = -0.5
    variants.append(joint_limits)

    contact = json.loads(json.dumps(document))
    contact["physical_birth_static_evidence"]["evaluator_evidence"][
        "normal_force_per_contact_n"
    ][0] = 0.0
    variants.append(contact)

    cop = json.loads(json.dumps(document))
    cop["physical_birth_static_evidence"]["evaluator_evidence"][
        "cop_interior_margin_per_foot_m"
    ][0] = -0.01
    variants.append(cop)

    qdes = json.loads(json.dumps(document))
    qdes["physical_birth_static_evidence"]["evaluator_evidence"][
        "hold_qdes_joint_pos_rad"
    ][0] = 1.0e-3
    variants.append(qdes)

    racket = json.loads(json.dumps(document))
    racket["physical_birth_composition"]["racket_site_fidelity"][
        "reference_authority"
    ]["position_semantics"] = "robot_site_origin"
    variants.append(racket)

    collision_authority = json.loads(json.dumps(document))
    collision_authority["physical_birth_static_evidence"][
        "evaluator_evidence"
    ]["collision_clearance"]["enabled_self_pair_count"] = 0
    variants.append(collision_authority)

    seeded = json.loads(json.dumps(document))
    seed_path = tmp_path / "physical_birth_seed.json"
    seeded["sources"]["physical_birth_seed"] = {
        "path": str(seed_path.resolve()),
        "sha256": _sha(seed_path.read_bytes()),
        "content_sha256": "1" * 64,
        "source_role": "numerical_seed_only",
        "inherited_model_identity": False,
        "inherited_hold_claim": False,
        "inherited_nominal_hold_claim": False,
    }
    variants.append(seeded)

    for variant in variants:
        _rewrite_dynamic_ready(path, variant)
        with pytest.raises(P.TableSmokeReceiptError, match="threshold-first"):
            P._load_nominal_hold_input(
                path, expected_sha256=_sha(path.read_bytes())
            )


def test_nominal_hold_live_motion_must_remain_teacher_not_physical(
    tmp_path, monkeypatch
):
    path, _document, _contract = _split_nominal_hold_fixture(tmp_path)
    inputs = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float64)
            self.dtype = self.value.dtype
            self.device = "cpu"

        @property
        def shape(self):
            return self.value.shape

        def __getitem__(self, item):
            return FakeTensor(self.value[item])

        def __array__(self, dtype=None):
            return np.asarray(self.value, dtype=dtype)

    class FakeTorch:
        @staticmethod
        def tensor(value, *, dtype=None, device=None):
            del dtype, device
            return FakeTensor(value)

        @staticmethod
        def allclose(left, right, *, rtol, atol):
            return np.allclose(
                np.asarray(left), np.asarray(right), rtol=rtol, atol=atol
            )

    motion = SimpleNamespace(
        num_segments=1,
        joint_pos=FakeTensor([inputs.teacher_joint_pos]),
        body_pos_w=FakeTensor([[inputs.teacher_root_pos]]),
        body_quat_w=FakeTensor([[inputs.teacher_root_quat]]),
    )
    command = SimpleNamespace(
        motion=motion,
        _motion_files=(str(inputs.motion_path),),
        _motion_file_sha256=(inputs.motion_sha256,),
    )
    unwrapped = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command)
    )
    monkeypatch.setattr(P, "torch", FakeTorch)
    P._assert_nominal_hold_motion(unwrapped, inputs)

    motion.joint_pos.value[0, 0] = inputs.physical_joint_pos[0]
    with pytest.raises(SystemExit):
        P._assert_nominal_hold_motion(unwrapped, inputs)


@pytest.mark.parametrize("delay_max_steps", (0, 2))
def test_nominal_hold_cfg_replays_artifact_control_step_delay(
    tmp_path, delay_max_steps
):
    path, _document, _contract = _nominal_hold_fixture(
        tmp_path, delay_max_steps=delay_max_steps
    )
    inputs = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )
    action_cfg = SimpleNamespace(
        control_step_action_delay_min=0,
        control_step_action_delay_max=0,
    )
    events = SimpleNamespace(
        add_joint_default_pos=SimpleNamespace(
            params={"pos_distribution_params": (-0.1, 0.1)}
        ),
        physics_material=object(),
        base_com=object(),
        randomize_link_mass=object(),
        randomize_pd_gains=object(),
    )
    cfg = SimpleNamespace(
        commands=SimpleNamespace(
            motion=SimpleNamespace(),
            racket_target=SimpleNamespace(),
        ),
        terminations=SimpleNamespace(
            anchor_pos=object(), anchor_ori=object(), ee_body_pos=object()
        ),
        events=events,
        actions=SimpleNamespace(joint_pos=action_cfg),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
        episode_length_s=1.0,
    )
    P._configure_nominal_hold_cfg(cfg, inputs, duration_s=0.8)
    assert action_cfg.control_step_action_delay_min == 0
    assert action_cfg.control_step_action_delay_max == delay_max_steps
    assert events.randomize_pd_gains is None


def test_direct_frame0_cfg_installs_exact_vendor_guard_and_hctrl(tmp_path):
    path, _document, _contract = _direct_frame0_nominal_hold_fixture(tmp_path)
    inputs = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )
    action_cfg = SimpleNamespace(
        control_step_action_delay_min=2,
        control_step_action_delay_max=2,
    )
    cfg = SimpleNamespace(
        commands=SimpleNamespace(
            motion=SimpleNamespace(), racket_target=SimpleNamespace()
        ),
        terminations=SimpleNamespace(
            anchor_pos=object(), anchor_ori=object(), ee_body_pos=object()
        ),
        events=SimpleNamespace(
            add_joint_default_pos=SimpleNamespace(
                params={"pos_distribution_params": (-0.1, 0.1)}
            ),
            physics_material=object(),
            base_com=object(),
            randomize_link_mass=object(),
            randomize_pd_gains=object(),
        ),
        actions=SimpleNamespace(joint_pos=action_cfg),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
        episode_length_s=1.0,
    )
    P._configure_nominal_hold_cfg(cfg, inputs, duration_s=0.4)
    assert action_cfg.pre_apply_guard_brake_mode == (
        "max_inward_until_nonoutward_v1"
    )
    assert action_cfg.pre_apply_guard_margin_rad == 0.0
    assert action_cfg.pre_apply_guard_margin_fraction == 0.06
    assert action_cfg.physx_control_position_limit_inset_fraction == 0.02


def test_nominal_hold_delay_match_accepts_only_runtime_disabled_omission():
    enabled = {
        "schema_version": 1,
        "enabled": True,
        "semantic_unit": "policy_control_step",
        "sample_timing": "once_per_episode_reset",
        "distribution": "discrete_uniform_inclusive",
        "min_steps": 0,
        "max_steps": 2,
        "shared_across_all_31_joints": True,
        "history_fill": "safe_default_or_action_specific_hold",
    }
    disabled = {**enabled, "enabled": False, "max_steps": 0}
    inherited_one_step = {**enabled, "min_steps": 1, "max_steps": 1}

    assert P._nominal_hold_delay_contract_matches(
        present=False, actual=None, expected=disabled
    )
    assert not P._nominal_hold_delay_contract_matches(
        present=True, actual=None, expected=disabled
    )
    assert not P._nominal_hold_delay_contract_matches(
        present=True, actual=disabled, expected=disabled
    )
    assert not P._nominal_hold_delay_contract_matches(
        present=True, actual=inherited_one_step, expected=disabled
    )
    assert P._nominal_hold_delay_contract_matches(
        present=True, actual=enabled, expected=enabled
    )
    assert not P._nominal_hold_delay_contract_matches(
        present=False, actual=None, expected=enabled
    )


def test_nominal_hold_joint_safety_summary_names_exact_current_and_substep():
    lower = [-1.0] * 31
    upper = [1.0] * 31
    pre_q = [0.0] * 31
    pre_qdot = [0.0] * 31
    final_q = [0.0] * 31
    final_qdot = [0.0] * 31
    final_q[0] = 1.001
    final_qdot[0] = 0.4
    final_q[1] = 0.95
    final_qdot[1] = -0.2
    current = [False] * 31
    current[0] = True
    substep = [False] * 31
    substep[1] = True

    summary = P._nominal_hold_joint_safety_summary(
        joint_names=A3_JOINT_NAMES,
        hard_lower=lower,
        hard_upper=upper,
        preterminal_q=pre_q,
        preterminal_qdot=pre_qdot,
        final_q=final_q,
        final_qdot=final_qdot,
        current_hard_edge=current,
        substep_actual_hard_edge=substep,
        final_source={"kind": "fixture"},
    )

    assert summary["complete"] is True
    assert summary["current_actual_hard_edge_joint_names"] == [
        A3_JOINT_NAMES[0]
    ]
    assert summary["substep_actual_hard_edge_joint_names"] == [
        A3_JOINT_NAMES[1]
    ]
    assert summary["current_actual_hard_edge_joint_count"] == 1
    assert summary["substep_actual_hard_edge_joint_count"] == 1
    assert summary["final_minimum_hard_gap_joint_name"] == A3_JOINT_NAMES[0]
    assert summary["flagged_joint_rows"] == [
        {
            "joint_index": 0,
            "joint_name": A3_JOINT_NAMES[0],
            "current_actual_hard_edge": True,
            "substep_actual_hard_edge": False,
            "preterminal_joint_pos_rad": 0.0,
            "preterminal_joint_vel_radps": 0.0,
            "final_joint_pos_rad": 1.001,
            "final_joint_vel_radps": 0.4,
            "hard_lower_rad": -1.0,
            "hard_upper_rad": 1.0,
            "final_minimum_hard_gap_rad": pytest.approx(-0.001),
        },
        {
            "joint_index": 1,
            "joint_name": A3_JOINT_NAMES[1],
            "current_actual_hard_edge": False,
            "substep_actual_hard_edge": True,
            "preterminal_joint_pos_rad": 0.0,
            "preterminal_joint_vel_radps": 0.0,
            "final_joint_pos_rad": 0.95,
            "final_joint_vel_radps": -0.2,
            "hard_lower_rad": -1.0,
            "hard_upper_rad": 1.0,
            "final_minimum_hard_gap_rad": pytest.approx(0.05),
        },
    ]


def test_nominal_hold_frame0_fidelity_summary_quantifies_all_three_levels():
    zero = {
        "joint_error_rad": [0.0] * 31,
        "root_position_error_m": [0.0, 0.0, 0.0],
        "root_orientation_error_rad": 0.0,
        "paddle_center_error_m": [0.0, 0.0, 0.0],
    }
    drift = {
        "joint_error_rad": [0.01] + [0.0] * 30,
        "root_position_error_m": [0.003, 0.004, 0.0],
        "root_orientation_error_rad": 0.02,
        "paddle_center_error_m": [0.0, 0.0, 0.012],
    }
    summary = P._nominal_hold_frame0_fidelity_summary(
        (zero, drift),
        joint_names=A3_JOINT_NAMES,
        paddle_reference_source="motion_npz.measured_racket_site_pos_w[0]",
    )
    assert summary["sample_count"] == 2
    assert summary["maximum_absolute_joint_error_rad"] == 0.01
    assert summary["maximum_root_position_error_m"] == 0.005
    assert summary["maximum_root_orientation_error_rad"] == 0.02
    assert summary["maximum_paddle_center_error_m"] == 0.012
    assert summary["formal_thresholds_adopted"] is False


def test_nominal_hold_terminal_joint_safety_uses_reset_surviving_archive():
    q = np.zeros((2, 31), dtype=np.float64)
    qdot = np.zeros((2, 31), dtype=np.float64)
    q[1, 4] = -1.01
    qdot[1, 4] = -0.3
    actual = np.zeros((2, 31), dtype=np.bool_)
    actual[1, 4] = True
    substep = np.zeros(31, dtype=np.bool_)
    substep[4] = True
    transcript = {
        "complete": True,
        "record_count": 2,
        "record_kind": ("apply", "post_step"),
        "timestamp_s": (0.0, 0.02),
        "q": q,
        "qdot": qdot,
        "actual_hard_edge": actual,
        "substep_actual_joint_latch": substep,
    }
    action = SimpleNamespace(
        joint_safety_ledger_snapshot=lambda: {
            "terminal_archives": (
                {
                    "archive_sequence": 7,
                    "policy_step_sequence": 53,
                    "transcript": transcript,
                },
            )
        }
    )
    preterminal = {
        "joint_pos_rad": [0.0] * 31,
        "joint_vel_radps": [0.0] * 31,
    }
    summary = P._nominal_hold_terminal_joint_safety(
        action,
        joint_names=A3_JOINT_NAMES,
        hard_lower=[-1.0] * 31,
        hard_upper=[1.0] * 31,
        preterminal=preterminal,
    )
    assert summary["final_source"] == {
        "kind": "joint_safety_terminal_archive",
        "archive_sequence": 7,
        "policy_step_sequence": 53,
        "transcript_complete": True,
        "record_count": 2,
        "record_kind": "post_step",
        "timestamp_s": 0.02,
    }
    assert summary["current_actual_hard_edge_joint_names"] == [
        A3_JOINT_NAMES[4]
    ]
    assert summary["flagged_joint_rows"][0]["final_joint_pos_rad"] == -1.01
    assert summary["substep_trigger_joint_rows"] == [
        {
            "joint_index": 4,
            "joint_name": A3_JOINT_NAMES[4],
            "side": "lower",
            "record_index": 1,
            "record_kind": "post_step",
            "timestamp_s": 0.02,
            "joint_pos_rad": -1.01,
            "joint_vel_radps": -0.3,
            "hard_lower_rad": -1.0,
            "hard_upper_rad": 1.0,
            "signed_hard_gap_rad": pytest.approx(-0.01),
        }
    ]


def test_nominal_hold_cfg_is_seeded_and_receipt_records_sampled_delay():
    cfg_source = inspect.getsource(P._cfg)
    probe_source = inspect.getsource(P.nominal_hold_probe)
    assert "or nominal_hold_inputs is not None" in cfg_source
    assert '"control_step_action_delay_runtime"' in probe_source
    assert "action.control_step_action_delay_runtime_receipt()" in probe_source
    assert "_nominal_hold_delay_contract_matches" in probe_source
    assert '"joint_safety_telemetry"' in probe_source
    assert "_nominal_hold_terminal_joint_safety" in probe_source
    assert "action.install_action_ball_dynamic_ready_state(" in probe_source
    assert "candidate_hold_qdes_and_delay_history_installed" in probe_source


def test_nominal_hold_outputs_are_no_clobber(tmp_path):
    output = P._fresh_nominal_path(tmp_path / "hold.json", "receipt")
    P._exclusive_publish_nominal_hold_receipt(
        output,
        {"kind": P.NOMINAL_HOLD_RECEIPT_KIND, "verdict": "FAIL"},
    )
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        P._fresh_nominal_path(output, "receipt")


def test_nominal_hold_captures_raw_reset_before_dynamic_ready_write():
    events = []
    unwrapped = SimpleNamespace(
        sim=SimpleNamespace(forward=lambda: events.append("forward")),
        scene=SimpleNamespace(
            update=lambda dt: events.append(("scene.update", dt))
        ),
    )
    P._refresh_nominal_hold_derived_state(unwrapped)
    assert events == ["forward", ("scene.update", 0.0)]

    source = inspect.getsource(P.nominal_hold_probe)
    reset = source.index("env.reset()")
    reset_refresh = source.index(
        "_refresh_nominal_hold_derived_state(unwrapped)", reset
    )
    raw_frame = source.index('save_frame("raw_env_reset", 0, last_png)')
    raw_render = source.index(
        "_nominal_hold_render_png(env)", reset_refresh
    )
    raw_saved_render = source.index(
        "last_png = _nominal_hold_render_png(env)", raw_render
    )
    artifact_write = source.index("motion_command.clip_id[env_ids] = 0")
    simulator_write = source.index("robot.write_root_state_to_sim(")
    ready_refresh = source.index(
        "_refresh_nominal_hold_derived_state(unwrapped)", simulator_write
    )
    ready_frame = source.index(
        'save_frame("physical_ready_after_reset_write", 0, last_png)'
    )
    assert (
        reset
        < reset_refresh
        < raw_render
        < raw_saved_render
        < raw_frame
        < artifact_write
        < simulator_write
        < ready_refresh
        < ready_frame
    )
    assert raw_render < raw_saved_render


def test_runtime_launcher_lifetime_and_stage_markers_are_explicit():
    init_source = inspect.getsource(P._initialize_isaac_runtime)
    assert "_app_launcher = AppLauncher(launcher_args)" in init_source
    assert "_app = _app_launcher.app" in init_source

    main_source = inspect.getsource(P.main)
    stages = [
        "gym_make_begin",
        "gym_make_done",
        "initial_reset_done",
        "nominal_hold_begin",
        "nominal_hold_done",
    ]
    positions = [main_source.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "and nominal_hold_inputs is None" in main_source
    assert "spawn_check_skipped_for_nominal_hold" in main_source
    assert (
        "failure_active = sys.exc_info()[0] is not None or exit_code != 0"
        in main_source
    )
    assert main_source.index("if not failure_active:") < main_source.index(
        "_app.close()"
    )
    assert main_source.index("_flush_process_streams()") < main_source.index(
        "_app.close()"
    )

    entrypoint_source = inspect.getsource(P._entrypoint)
    assert "except BaseException as exc:" in entrypoint_source
    assert "traceback.print_exc()" in entrypoint_source
    assert "os._exit(failure_code)" in entrypoint_source
    assert "os._exit(int(exit_code))" in entrypoint_source


@pytest.mark.parametrize(
    ("raised", "expected_exit"),
    (
        (SystemExit(7), 7),
        (SystemExit(0), 1),
        (RuntimeError("boom"), 1),
        (KeyboardInterrupt(), 130),
    ),
)
def test_entrypoint_preserves_nonzero_failure_at_process_boundary(
    monkeypatch, raised, expected_exit
):
    class _ExitCaptured(BaseException):
        def __init__(self, code):
            self.code = code

    def fail_main():
        raise raised

    def capture_exit(code):
        raise _ExitCaptured(code)

    monkeypatch.setattr(P, "main", fail_main)
    monkeypatch.setattr(P.os, "_exit", capture_exit)
    monkeypatch.setattr(P.traceback, "print_exc", lambda: None)
    with pytest.raises(_ExitCaptured) as captured:
        P._entrypoint()
    assert captured.value.code == expected_exit


def test_entrypoint_exits_zero_only_after_main_returns(monkeypatch):
    class _ExitCaptured(BaseException):
        def __init__(self, code):
            self.code = code

    monkeypatch.setattr(P, "main", lambda: 0)
    monkeypatch.setattr(
        P.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitCaptured(code)),
    )
    with pytest.raises(_ExitCaptured) as captured:
        P._entrypoint()
    assert captured.value.code == 0


def test_contact_smoke_rejects_nonfinite_live_pose():
    source = inspect.getsource(P.contact_smoke)
    assert "selected_body_pos[0, 0, 0] = float(\"nan\")" in source
    assert "geometric_table_contact_hit_mask(" in source
    assert "if not bool(nonfinite_hit[0].item())" in source
    assert "did not fail safe on non-finite live pose" in source
    assert "move_root(contact_root_pose)" in source
    assert "move_root(safe_root_pose)" in source
    assert source.index("original_apply()") < source.index(
        "move_root(contact_root_pose)"
    )


def test_contact_smoke_uses_deterministic_table_clear_stand_reset():
    source = inspect.getsource(P._cfg)
    assert "formal_inputs is not None or ARGS.contact_smoke" in source
    assert "cfg.seed = 0" in source
    assert "cfg.commands.motion.stand_start_prob = 1.0" in source
    assert "cfg.commands.motion.hold_steps_range = (100, 100)" in source
    assert "cfg.commands.motion.stand_start_min_hold = 100" in source


def test_formal_cli_has_no_boolean_pass_claims_and_requires_pod_shape():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "--real-physx-contacts",
        "--all-32-body-pair-filters",
        "--positive-control-pass",
        "--negative-control-pass",
        "--zero-reset-leakage",
    ):
        assert forbidden not in source

    args = P._parse(
        [
            "--receipt-out",
            "receipt.json",
            "--action-set-profile",
            "fixture_n5",
            "--manifest",
            "manifest.json",
        ]
    )
    with pytest.raises(P.TableSmokeReceiptError, match="num-envs"):
        P._validate_cli_mode(args)


def test_manifest_snapshots_exact_fresh_n5_order_and_motion_bytes(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    assert tuple(row.motion_id for row in inputs.motions) == P.FRESH_N5_ACTION_IDS
    assert len({row.action_uid for row in inputs.motions}) == 5
    assert inputs.manifest.sha256 == _sha(manifest_path.read_bytes())
    assert len({row.file.sha256 for row in inputs.motions}) == 5
    P._assert_formal_inputs_unchanged(inputs)

    inputs.motions[0].file.path.write_bytes(b"changed-after-snapshot")
    with pytest.raises(P.TableSmokeReceiptError, match="inode or bytes changed"):
        P._assert_formal_inputs_unchanged(inputs)


@pytest.mark.parametrize("action_count", (1, 5, 73))
def test_formal_table_inputs_and_receipt_keep_every_action_and_32xn_body_contract_rows(
    tmp_path, action_count: int
):
    root = tmp_path / f"n{action_count}"
    source, manifest_path, _ = _fixture_tree(
        root, action_count=action_count
    )
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=root
    )
    assert len(inputs.motions) == action_count
    assert inputs.action_set_contract["expected_n"] == action_count
    receipt = P._build_formal_receipt(
        inputs, _valid_evidence(inputs)
    )
    assert len(receipt["actions"]) == action_count
    assert all(
        row["robot_body_contract_count"] == 32
        for row in receipt["actions"]
    )
    assert (
        receipt["runtime_contract"]["action_robot_body_contract_rows"]
        == 32 * action_count
    )
    assert (
        receipt["runtime_contract"][
            "all_five_table_components_with_pose_obb"
        ]
        is True
    )
    assert receipt["schema_version"] == 4


@pytest.mark.parametrize("action_count", (1, 5, 73))
@pytest.mark.parametrize(
    "tamper",
    ("count", "order", "uid", "scope", "mobility", "manifest_sha"),
)
def test_formal_table_rejects_action_set_or_manifest_tamper(
    tmp_path, action_count: int, tamper: str
):
    root = tmp_path / f"n{action_count}_{tamper}"
    source, manifest_path, manifest = _fixture_tree(
        root, action_count=action_count
    )
    trusted = _fixture_action_set(root, manifest_path)
    if tamper == "count":
        trusted = dict(trusted)
        trusted["expected_n"] += 1
    elif tamper == "order":
        manifest["action_order"] = (
            list(reversed(manifest["action_order"]))
            if action_count > 1
            else ["wrong_action"]
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "uid":
        manifest["actions"][0]["action_uid"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "scope":
        manifest["prototype"]["scope"] = (
            "upper" if trusted["scope"] == "full" else "full"
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "mobility":
        manifest["mobility_mode"] = "move"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        trusted = dict(trusted)
        trusted["manifest_sha256"] = "0" * 64
    original = P._load_trusted_action_set
    P._load_trusted_action_set = lambda profile_id: trusted
    try:
        with pytest.raises(P.TableSmokeReceiptError):
            P._load_formal_inputs(
                manifest_path.relative_to(root).as_posix(),
                action_set_profile=trusted["profile_id"],
                profile_pins_value="profile_pins.json",
                expected_profile_pins_sha256=_sha(
                    (root / "profile_pins.json").read_bytes()
                ),
                repo_root=root,
                source_path=source,
            )
    finally:
        P._load_trusted_action_set = original


def test_profile_pins_and_solver_geometry_bytes_are_fail_closed(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    profile_path = tmp_path / "profile_pins.json"
    expected_profile_sha = _sha(profile_path.read_bytes())
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["physics_payload"]["kind"] = "tampered"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(
        P.TableSmokeReceiptError, match="preregistered SHA-256"
    ):
        _load_formal_fixture(
            manifest_path.name,
            profile_pins_value=profile_path.name,
            expected_profile_pins_sha256=expected_profile_sha,
            repo_root=tmp_path,
            source_path=source,
        )

    source, manifest_path, _ = _fixture_tree(tmp_path / "source")
    source_root = tmp_path / "source"
    solver_path = (
        source_root
        / "hope_training/whole_body_tracking/source/"
        "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
        "racket_contact_geometry.py"
    )
    solver_path.write_text("# drifted geometry source\n", encoding="utf-8")
    profile_path = source_root / "profile_pins.json"
    with pytest.raises(
        P.TableSmokeReceiptError, match="bytes differ from profile pins"
    ):
        _load_formal_fixture(
            manifest_path.name,
            profile_pins_value=profile_path.name,
            expected_profile_pins_sha256=_sha(
                profile_path.read_bytes()
            ),
            repo_root=source_root,
            source_path=source,
        )

    semantic_root = tmp_path / "semantic"
    source, manifest_path, manifest = _fixture_tree(semantic_root)
    manifest["racket_geometry_contract"] = geometry_contract = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": (
            "hope_training/whole_body_tracking/source/"
            "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
            "racket_contact_geometry.py"
        ),
        "source_sha256": "1" * 64,
        "geometry_source_sha256": "2" * 64,
    }
    assert geometry_contract["schema_version"] == 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    profile_path = semantic_root / "profile_pins.json"
    with pytest.raises(
        P.TableSmokeReceiptError,
        match="strict training manifest contains gate-only",
    ):
        _load_formal_fixture(
            manifest_path.name,
            profile_pins_value=profile_path.name,
            expected_profile_pins_sha256=_sha(profile_path.read_bytes()),
            repo_root=semantic_root,
            source_path=source,
        )


def test_source_identity_rejects_nonignored_untracked_checkout_files(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    expected_commit = _commit_fixture(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    assert P._committed_source_identity(inputs) == expected_commit

    (tmp_path / "untracked_override.py").write_text(
        "raise RuntimeError('must not be ignored')\n", encoding="utf-8"
    )
    with pytest.raises(P.TableSmokeReceiptError, match="exact clean checkout"):
        P._committed_source_identity(inputs)


def test_runtime_module_closure_rejects_other_checkout_and_byte_drift(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    module_path = tmp_path / "package" / "runtime_fixture.py"
    module_path.parent.mkdir()
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    expected_commit = _commit_fixture(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    module_name = "whole_body_tracking.runtime_fixture"
    module = ModuleType(module_name)
    module.__file__ = str(module_path)
    # Other host-only tests install synthetic ``whole_body_tracking`` namespace
    # parents during collection.  They intentionally have no source file and
    # are not part of this fixture checkout.  Isolate the closure fixture
    # instead of weakening the formal producer to accept source-less runtime
    # modules.
    displaced = {
        name: loaded
        for name, loaded in tuple(sys.modules.items())
        if name == "whole_body_tracking"
        or name.startswith("whole_body_tracking.")
    }
    for name in displaced:
        sys.modules.pop(name, None)
    sys.modules[module_name] = module
    try:
        baseline = P._assert_runtime_source_closure(
            inputs,
            expected_commit,
            required_modules=(module_name,),
        )
        assert baseline[module_name].sha256 == _sha(b"VALUE = 1\n")

        module_path.write_text("VALUE = 2\n", encoding="utf-8")
        with pytest.raises(
            P.TableSmokeReceiptError, match="bytes differ from source commit"
        ):
            P._assert_runtime_source_closure(
                inputs,
                expected_commit,
                baseline=baseline,
                required_modules=(module_name,),
            )

        outside = tmp_path.parent / f"{tmp_path.name}-other-checkout.py"
        outside.write_text("VALUE = 1\n", encoding="utf-8")
        module.__file__ = str(outside)
        with pytest.raises(
            P.TableSmokeReceiptError, match="must resolve inside repository root"
        ):
            P._assert_runtime_source_closure(
                inputs,
                expected_commit,
                required_modules=(module_name,),
            )
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.update(displaced)


def test_formal_repo_root_is_derived_from_exact_tracked_producer_path(
    tmp_path,
):
    source = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "check_table_obstacle_scene.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("# producer\n", encoding="utf-8")
    assert P._repository_root_from_producer(source) == tmp_path

    relocated = tmp_path / "scripts" / "check_table_obstacle_scene.py"
    relocated.parent.mkdir()
    relocated.write_text("# producer\n", encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="exact tracked"):
        P._repository_root_from_producer(relocated)


def test_manifest_rejects_reorder_retired_id_hash_drift_and_traversal(tmp_path):
    source, manifest_path, manifest = _fixture_tree(tmp_path)

    manifest["action_order"] = list(reversed(P.FRESH_N5_ACTION_IDS))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="action order"):
        _load_fixture_inputs(
            source, manifest_path, repo_root=tmp_path
        )

    hash_root = tmp_path / "hash"
    hash_source, manifest_path, manifest = _fixture_tree(hash_root)
    manifest["actions"][0]["motion_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="differ from manifest"):
        _load_fixture_inputs(
            hash_source, manifest_path, repo_root=hash_root
        )

    traversal_root = tmp_path / "traversal"
    traversal_source, manifest_path, manifest = _fixture_tree(
        traversal_root
    )
    manifest["actions"][0]["motion_path"] = "../escape.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="path traversal"):
        _load_fixture_inputs(
            traversal_source,
            manifest_path,
            repo_root=traversal_root,
        )

    uid_root = tmp_path / "uid"
    uid_source, manifest_path, manifest = _fixture_tree(uid_root)
    manifest["actions"][0]["action_uid"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="canonical action"):
        _load_fixture_inputs(
            uid_source, manifest_path, repo_root=uid_root
        )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers():
    with pytest.raises(P.TableSmokeReceiptError, match="duplicate JSON key"):
        P._strict_json_object(b'{"schema_version":3,"schema_version":3}', "x")
    with pytest.raises(P.TableSmokeReceiptError, match="forbidden JSON constant"):
        P._strict_json_object(b'{"value":NaN}', "x")


def test_symlinked_motion_and_symlinked_output_parent_are_rejected(tmp_path):
    source, manifest_path, manifest = _fixture_tree(tmp_path)
    real = tmp_path / "motions" / "bh_loop_c.npz"
    target = tmp_path / "real-motion.npz"
    target.write_bytes(real.read_bytes())
    real.unlink()
    real.symlink_to(target)
    with pytest.raises(P.TableSmokeReceiptError, match="symlink component"):
        _load_fixture_inputs(
            source, manifest_path, repo_root=tmp_path
        )

    output_real = tmp_path / "output-real"
    output_real.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output_real, target_is_directory=True)
    with pytest.raises(P.TableSmokeReceiptError, match="symlink component"):
        P._prepare_output_path(
            "output-link/receipt.json", repo_root=tmp_path
        )


def test_without_live_isaac_origin_no_pass_receipt_can_be_built(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    evidence = _valid_evidence(inputs)
    P._ISAAC_RUNTIME_ORIGIN = None
    with pytest.raises(P.TableSmokeReceiptError, match="live Isaac runtime"):
        P._build_formal_receipt(inputs, evidence)


def test_exact_receipt_schema_seal_and_exclusive_readback(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    receipt = P._build_formal_receipt(inputs, _valid_evidence(inputs))
    assert receipt["task_id"] == P.ACTION_BALL_TASK_ID
    assert receipt["manifest"] == {
        "path": "manifest.json",
        "sha256": _sha(manifest_path.read_bytes()),
    }
    assert receipt["runtime_contract"]["runtime_source"] == {
        "path": (
            "hope_training/whole_body_tracking/scripts/"
            "check_table_obstacle_scene.py"
        ),
        "sha256": _sha(source.read_bytes()),
    }
    assert receipt["runtime_contract"]["gpu_identity"]["nvml_verified"] is True
    assert [row["action_uid"] for row in receipt["actions"]] == [
        row.action_uid for row in inputs.motions
    ]
    P._validate_formal_receipt_document(receipt, inputs=inputs)

    output, _ = P._prepare_output_path(
        "receipt.json", repo_root=tmp_path
    )
    previous_umask = os.umask(0o077)
    try:
        file_sha = P._exclusive_publish_receipt(output, receipt)
    finally:
        os.umask(previous_umask)
    payload = output.read_bytes()
    assert file_sha == _sha(payload)
    assert payload == P._canonical_json_bytes(json.loads(payload))
    assert (os.stat(output).st_mode & 0o777) == 0o444
    with pytest.raises(FileExistsError):
        P._exclusive_publish_receipt(output, receipt)

    forged = dict(receipt)
    forged["task_id"] = "Tracking-Flat-AgibotA3-Hope-ActionBall-v0"
    with pytest.raises(P.TableSmokeReceiptError, match="identity is not exact"):
        P._validate_formal_receipt_document(forged)


def test_schema4_stale_filtered_field_is_rejected(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    receipt = P._build_formal_receipt(inputs, _valid_evidence(inputs))
    runtime = receipt["runtime_contract"]
    runtime["all_five_table_sources_with_explicit_robot_body_filters"] = (
        runtime.pop("all_five_table_components_with_pose_obb")
    )
    receipt["receipt_payload_sha256"] = _sha(
        P._canonical_json_bytes(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_payload_sha256"
            }
        )
    )
    with pytest.raises(
        P.TableSmokeReceiptError,
        match="runtime_contract keys are not exact",
    ):
        P._validate_formal_receipt_document(receipt, inputs=inputs)


def test_pose_obb_v4_roundtrips_into_canonical_admission(
    tmp_path,
):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    receipt = P._build_formal_receipt(
        inputs, _valid_evidence(inputs)
    )
    output, relative = P._prepare_output_path(
        "isaac_table_smoke.json", repo_root=tmp_path
    )
    receipt_sha = P._exclusive_publish_receipt(output, receipt)
    binding = SimpleNamespace(
        isaac_table_filtered_smoke_receipt_sha256=receipt_sha,
        motion_ids=tuple(P.FRESH_N5_ACTION_IDS),
        npz_sha256=tuple(row.file.sha256 for row in inputs.motions),
    )

    ADMISSION._validate_fresh_n5_isaac_table_smoke_receipt(
        {"path": relative, "sha256": receipt_sha},
        binding=binding,
        repo_root=tmp_path,
    )


def test_false_runtime_boolean_or_unsafe_action_cannot_seal_pass(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    good = _valid_evidence(inputs)
    bad_boolean = P._RuntimeEvidence(
        **{**good.__dict__, "positive_control_pass": False}
    )
    with pytest.raises(P.TableSmokeReceiptError, match="required table-smoke"):
        P._build_formal_receipt(inputs, bad_boolean)

    first = good.actions[0]
    unsafe = P._RuntimeActionEvidence(
        **{**first.__dict__, "table_contact_count": 1, "unsafe_count": 1}
    )
    bad_action = P._RuntimeEvidence(
        **{**good.__dict__, "actions": (unsafe, *good.actions[1:])}
    )
    with pytest.raises(P.TableSmokeReceiptError, match="not zero"):
        P._build_formal_receipt(inputs, bad_action)
