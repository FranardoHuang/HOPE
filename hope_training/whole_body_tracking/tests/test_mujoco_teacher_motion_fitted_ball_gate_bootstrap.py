from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
BOOTSTRAP_PATH = (
    REPO
    / "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate_bootstrap.py"
)
CORE_PATH = (
    REPO
    / "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate.py"
)
ADMISSION_PATH = (
    REPO
    / "hope_training/whole_body_tracking/scripts/"
    "canonical_motion_admission.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BOOTSTRAP = _load("fitted_gate_bootstrap_test", BOOTSTRAP_PATH)
CORE = _load("fitted_gate_core_bootstrap_test", CORE_PATH)
ADMISSION = _load("fitted_gate_admission_handoff_test", ADMISSION_PATH)


def _snapshot(path: Path, repo_path: str, raw: bytes) -> object:
    metadata = path.stat()
    return BOOTSTRAP.PinnedBytes(
        repo_path=repo_path,
        path=path.resolve(),
        raw=raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        stat_device=metadata.st_dev,
        stat_inode=metadata.st_ino,
        stat_size=metadata.st_size,
        stat_mtime_ns=metadata.st_mtime_ns,
    )


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        BOOTSTRAP._canonical_json_bytes(value)
    ).hexdigest()


def _write_canonical_json(path: Path, value: object) -> str:
    raw = BOOTSTRAP._canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _dual_manifest_fixture(tmp_path: Path) -> dict:
    action_order = list(BOOTSTRAP.LEGACY_FRESH_N5_ORDER)
    training = {
        "schema_version": 3,
        "manifest_id": "dual-manifest-fixture",
        "action_order": action_order,
        "actions": [
            {"action_id": action_id, "motion_sha256": str(index) * 64}
            for index, action_id in enumerate(action_order, start=1)
        ],
    }
    physical = {
        **training,
        "actions": [
            {
                **row,
                "physical_ball_launch": {"source": "fixture"},
                "physical_task_binding": {"cases": ["fixture"]},
                "admission": {"authorization_granted": False},
            }
            for row in training["actions"]
        ],
        "racket_geometry_contract": {"schema_version": 2},
        "physical_contact_contract": {"schema_version": 2},
    }
    training_path = tmp_path / "strict.json"
    physical_path = tmp_path / "physical.json"
    receipt_path = tmp_path / "materialization.json"
    training_sha = _write_canonical_json(training_path, training)
    physical_sha = _write_canonical_json(physical_path, physical)
    receipt = {
        "schema_version": 1,
        "kind": BOOTSTRAP.PHYSICAL_GATE_MATERIALIZATION_KIND,
        "strict_training_manifest": {
            "path": "artifacts/strict.json",
            "sha256": training_sha,
        },
        "physical_task_bundle": {
            "path": "artifacts/bundle.json",
            "sha256": "3" * 64,
        },
        "physical_gate_manifest": {
            "path": "artifacts/physical.json",
            "sha256": physical_sha,
        },
        "candidate_entries": [
            {
                "action_id": action_id,
                "path": f"artifacts/candidates/{action_id}.json",
                "sha256": str(index + 3) * 64,
            }
            for index, action_id in enumerate(action_order)
        ],
        "compiler_manifests": {
            "base": {"path": "artifacts/compiler/base.json", "sha256": "6" * 64},
            "append": {
                "path": "artifacts/compiler/append.json",
                "sha256": "7" * 64,
            },
        },
        "bank_gate_reports": {
            "base": {"path": "artifacts/bank/base.json", "sha256": "8" * 64},
            "append": {
                "path": "artifacts/bank/append.json",
                "sha256": "9" * 64,
            },
        },
        "action_order": action_order,
        "strict_training_manifest_preserved": True,
        "inline_manifest_gate_only": True,
        "selector_executed": False,
        "authorization_granted": False,
    }
    receipt_sha = _write_canonical_json(receipt_path, receipt)
    return {
        "training": training,
        "physical": physical,
        "receipt": receipt,
        "training_path": training_path,
        "physical_path": physical_path,
        "receipt_path": receipt_path,
        "training_sha": training_sha,
        "physical_sha": physical_sha,
        "receipt_sha": receipt_sha,
    }


def _schema2_dual_manifest_fixture(
    tmp_path: Path, action_count: int
) -> dict:
    action_order = [
        f"schema2_action_{index:03d}" for index in range(action_count)
    ]
    action_uids = list(range(1000, 1000 + action_count))
    actions = []
    for index, (action_id, action_uid) in enumerate(
        zip(action_order, action_uids)
    ):
        actions.append(
            {
                "action_id": action_id,
                "action_uid": action_uid,
                "family": (
                    "backhand" if index % 2 == 0 else "forehand"
                ),
                "motion_path": f"motions/{action_id}.npz",
                "motion_sha256": f"{index + 1:064x}",
                "ball_profile": {
                    "contact_offset_center_b_yaw_m": [
                        0.6,
                        0.001 * index,
                        1.0,
                    ],
                    "time_to_contact_center_s": 1.2 + 0.001 * index,
                    "incoming_direction_center_b_yaw": [
                        -1.0,
                        0.0,
                        0.0,
                    ],
                    "incoming_speed_center_mps": 2.0 + 0.01 * index,
                    "spin_direction_center_b_yaw": [0.0, 1.0, 0.0],
                    "spin_magnitude_center_radps": 0.0,
                    "base_spawn_center_w_xy_m": [0.0, 0.0],
                    "base_travel_center_b_yaw_xy_m": [0.0, 0.0],
                },
            }
        )
    training = {
        "schema_version": 3,
        "manifest_id": f"schema2-n{action_count}",
        "action_order": action_order,
        "actions": actions,
    }
    training_path = tmp_path / f"strict_n{action_count}.json"
    physical_path = tmp_path / f"physical_n{action_count}.json"
    receipt_path = tmp_path / f"materialization_n{action_count}.json"
    training_sha = _write_canonical_json(training_path, training)
    profile_id = f"fixture_schema2_n{action_count}"
    contract = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.action_set_contract",
        "profile_id": profile_id,
        "expected_n": action_count,
        "scope": "full" if action_count == 73 else "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": action_order,
        "ordered_action_uids": action_uids,
        "order_uid_digest_sha256": hashlib.sha256(
            BOOTSTRAP._canonical_json_bytes(
                list(zip(action_order, action_uids))
            )
        ).hexdigest(),
        "manifest_path": "artifacts/strict.json",
        "manifest_sha256": training_sha,
        "experiment_name": f"schema2-n{action_count}",
        "actor_obs_contract": "fixture",
        "actor_obs_width": 1,
        "namespace_identity": f"fixture-schema2-n{action_count}",
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    physical = {
        **training,
        "actions": [
            {
                **row,
                "physical_ball_launch": {"source": "fixture"},
                "physical_task_binding": {"cases": ["fixture"]},
                "admission": {"authorization_granted": False},
            }
            for row in training["actions"]
        ],
        "racket_geometry_contract": {"schema_version": 2},
        "physical_contact_contract": {"schema_version": 2},
    }
    compiler_manifests = []
    bank_gate_reports = []
    for index, row in enumerate(physical["actions"]):
        compiler = {
            "action_id": row["action_id"],
            "path": f"artifacts/compiler/{row['action_id']}.json",
            "sha256": f"{index + 200:064x}",
        }
        bank = {
            "action_id": row["action_id"],
            "path": f"artifacts/bank/{row['action_id']}.json",
            "sha256": f"{index + 300:064x}",
        }
        compiler_manifests.append(compiler)
        bank_gate_reports.append(bank)
        row["admission"].update(
            {
                "compiler_manifest_path": compiler["path"],
                "compiler_manifest_sha256": compiler["sha256"],
                "bank_gate_report_path": bank["path"],
                "bank_gate_report_sha256": bank["sha256"],
            }
        )
    physical_sha = _write_canonical_json(physical_path, physical)
    receipt = {
        "schema_version": 2,
        "kind": BOOTSTRAP.GENERIC_PHYSICAL_GATE_MATERIALIZATION_KIND,
        "action_set_contract": contract,
        "action_identity_matrix": (
            BOOTSTRAP._materialization_action_identity_matrix(
                training, contract
            )
        ),
        "strict_training_manifest": {
            "path": "artifacts/strict.json",
            "sha256": training_sha,
        },
        "physical_task_bundle": {
            "path": "artifacts/bundle.json",
            "sha256": "3" * 64,
        },
        "physical_gate_manifest": {
            "path": "artifacts/physical.json",
            "sha256": physical_sha,
        },
        "candidate_entries": [
            {
                "action_id": action_id,
                "path": f"artifacts/candidates/{action_id}.json",
                "sha256": f"{index + 100:064x}",
            }
            for index, action_id in enumerate(action_order)
        ],
        "compiler_manifests": compiler_manifests,
        "bank_gate_reports": bank_gate_reports,
        "action_order": action_order,
        "strict_training_manifest_preserved": True,
        "inline_manifest_gate_only": True,
        "selector_executed": False,
        "authorization_granted": False,
    }
    receipt_sha = _write_canonical_json(receipt_path, receipt)
    return {
        "profile_id": profile_id,
        "training": training,
        "physical": physical,
        "receipt": receipt,
        "training_path": training_path,
        "physical_path": physical_path,
        "receipt_path": receipt_path,
        "training_sha": training_sha,
        "physical_sha": physical_sha,
        "receipt_sha": receipt_sha,
    }


def _schema2_formal_identity_fixture(
    tmp_path: Path, action_count: int
) -> tuple[dict, dict]:
    fixture = _schema2_dual_manifest_fixture(tmp_path, action_count)
    contract = fixture["receipt"]["action_set_contract"]
    matrix = fixture["receipt"]["action_identity_matrix"]
    ground_policy = {
        "floor_geom_name": BOOTSTRAP.FLOOR_GEOM_NAME,
        "legal_foot_body_names": list(
            BOOTSTRAP.LEGAL_FOOT_BODY_NAMES
        ),
        "all_collision_enabled_robot_geoms_floor_pair_enabled": True,
        "foot_floor_penetration_tolerance_m": (
            BOOTSTRAP.FOOT_FLOOR_PENETRATION_TOLERANCE_M
        ),
        "nonfoot_floor_penetration_tolerance_m": (
            BOOTSTRAP.NONFOOT_FLOOR_PENETRATION_TOLERANCE_M
        ),
        "nonfoot_force_threshold_n": (
            BOOTSTRAP.GROUND_CONTACT_FORCE_THRESHOLD_N
        ),
        "continuous_nonfoot_clearance_guard_m": (
            BOOTSTRAP.NONFOOT_GROUND_CLEARANCE_GUARD_M
        ),
        "continuous_distance_query_cap_m": (
            BOOTSTRAP.GROUND_DISTANCE_QUERY_CAP_M
        ),
    }
    ground_contract = {
        "floor_geom_name": BOOTSTRAP.FLOOR_GEOM_NAME,
        "floor_geom_id": 0,
        "floor_geom_type": "plane",
        "legal_foot_body_names": list(
            BOOTSTRAP.LEGAL_FOOT_BODY_NAMES
        ),
        "legal_foot_body_ids": [10, 11],
        "legal_foot_geom_names": [
            "left_ankle_roll_collision",
            "right_ankle_roll_collision",
        ],
        "legal_foot_geom_ids": [20, 21],
        "nonfoot_floor_pair_enabled_robot_geom_count": 30,
        "all_collision_enabled_robot_geoms_floor_pair_enabled": True,
        "foot_floor_penetration_tolerance_m": (
            BOOTSTRAP.FOOT_FLOOR_PENETRATION_TOLERANCE_M
        ),
        "nonfoot_floor_penetration_tolerance_m": (
            BOOTSTRAP.NONFOOT_FLOOR_PENETRATION_TOLERANCE_M
        ),
        "nonfoot_force_threshold_n": (
            BOOTSTRAP.GROUND_CONTACT_FORCE_THRESHOLD_N
        ),
        "continuous_nonfoot_clearance_guard_m": (
            BOOTSTRAP.NONFOOT_GROUND_CLEARANCE_GUARD_M
        ),
        "continuous_distance_query_cap_m": (
            BOOTSTRAP.GROUND_DISTANCE_QUERY_CAP_M
        ),
        "policy": "fixture exact-feet-only continuous ground policy",
    }
    geometry_payload = {"fixture": "five-solid"}
    geometry_sha = _canonical_sha256(geometry_payload)
    scene = {
        "five_solid_geometry_sha256": geometry_sha,
        "geometry_payload": geometry_payload,
        "obstacle_order": list(
            BOOTSTRAP.FIVE_SOLID_OBSTACLE_ORDER
        ),
        "under_table_keepout_role": "robot_only",
        "ball_keepout_native_pair_enabled": False,
        "ball_keepout_analytic_surface_enabled": False,
        "contact_force_threshold_n": 1.0e-6,
        "continuous_sweep_method": "fixture_continuous_sweep",
        "ground_contact_policy": ground_policy,
        "compiled_by_dt": {
            timestep: {
                "five_solid_geometry_sha256": geometry_sha,
                "assembled_xml_sha256": hashlib.sha256(
                    f"assembled:{timestep}".encode()
                ).hexdigest(),
                "ball_keepout_native_pair_enabled": False,
                "ball_keepout_analytic_surface_enabled": False,
                "ground_contact_safety_contract": ground_contract,
            }
            for timestep in ("0.0010", "0.0005")
        },
    }
    bindings = {
        "training_manifest": {
            "repo_path": contract["manifest_path"],
            "sha256": contract["manifest_sha256"],
        },
        "physical_gate_manifest": {
            "repo_path": "artifacts/physical.json",
            "sha256": "2" * 64,
        },
        "physical_gate_materialization_receipt": {
            "repo_path": "artifacts/materialization.json",
            "sha256": "3" * 64,
        },
        "profile_pins": {
            "repo_path": "artifacts/profile.json",
            "sha256": "4" * 64,
        },
        "launch_evidence_trust_root": {
            "repo_path": "artifacts/launch_root.json",
            "sha256": "5" * 64,
        },
    }
    formal = {
        "schema_version": 2,
        "materialization_receipt_schema_version": 2,
        "materialization_receipt_kind": (
            BOOTSTRAP.GENERIC_PHYSICAL_GATE_MATERIALIZATION_KIND
        ),
        "expected_actions": action_count,
        "expected_action_order": list(contract["ordered_action_ids"]),
        "action_order": list(contract["ordered_action_ids"]),
        "action_set_contract": contract,
        "action_identity_matrix": matrix,
        "action_identity_matrix_sha256": _canonical_sha256(matrix),
        "actions": [
            {
                "action_id": row["action_id"],
                "action_uid": row["action_uid"],
                "family": row["family"],
                "scope": row["scope"],
                "motion_path": row["motion_path"],
                "motion_sha256": row["motion_sha256"],
                "profile_center": row["profile_center"],
                "profile_center_sha256": (
                    row["profile_center_sha256"]
                ),
            }
            for row in matrix
        ],
        "five_solid_safety_scene": scene,
        "runtime_code_identity": {
            "code_commit": "a" * 40,
            "committed_trust_spec": {"bindings": bindings},
        },
        "runtime_input_snapshot": {"files": []},
        "receipt_payload_sha256": "6" * 64,
    }
    artifact_tree = {
        "tree_sha256": "7" * 64,
        "file_count": 1,
        "total_size_bytes": 1,
    }
    return formal, artifact_tree


def _ready_recovery_fixture() -> dict:
    metric_names = (
        "joint_linf_rad",
        "root_position_l2_m",
        "root_orientation_angle_rad",
        "endpoint_joint_velocity_peak_radps",
        "endpoint_root_linear_velocity_peak_mps",
        "endpoint_root_angular_velocity_peak_radps",
    )
    metrics = {name: 0.0 for name in metric_names}
    thresholds = {
        "joint_linf_rad": 0.1,
        "root_position_l2_m": 0.1,
        "root_orientation_angle_rad": 0.1,
        "endpoint_velocity_peak": 0.1,
    }
    return {
        "shared_ready": {
            **metrics,
            "thresholds": thresholds,
        },
        "action_recovery": metrics,
        "recovery_thresholds": thresholds,
        "grounded_bank_evidence": {
            "bank_gate_pass": True,
            "publication_class": "post_build_diagnostic_only",
            "training_authorized": False,
            "scope": "upper",
            "grounded_trace_status": (
                "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
            ),
            "shared_ready": True,
            "six_endpoint_velocity_classes_exact_zero": True,
            "time_law": {
                "schema_version": 2,
                "artifact_type": "canonical_time_law_collocation_v2",
                "artifact_npz_sha256": "1" * 64,
                "artifact_manifest_sha256": "2" * 64,
                "artifact_bundle_sha256": "3" * 64,
            },
            "grounded_lmr": {
                "cell_count": 1,
                "sample_count": 3,
                "finite_difference_qacc_used": False,
            },
            "safety_counts": {
                "joint_limit_violation_count": 0,
                "self_collision_violation_count": 0,
                "world_collision_violation_count": 0,
            },
        },
    }


def _physical_dt_result_fixture(
    *,
    case_role: str,
    timestep_s: float,
    task_timing: dict,
    task_geometry: dict,
    physics_bounds: dict,
    contact_model_sha256: str,
    face_mesh_sha256: str,
) -> dict:
    positive = (
        case_role
        in ADMISSION._FRESH_N5_PHYSICAL_TASK_POSITIVE_ROLES
    )
    negative_reason = {
        "negative_t_hit_offset": "physical_contact_time_mismatch",
        "negative_face_sign": "teacher_task_face_normal_mismatch",
        "negative_ball_state_mismatch": (
            "physical_incoming_velocity_mismatch"
        ),
    }.get(case_role)
    pre_swing_wait = float(task_timing["pre_swing_wait_s"])
    scaled_t_hit = float(task_timing["scaled_t_hit_s"])
    scaled_t_cycle = float(task_timing["scaled_t_cycle_s"])
    teacher_rate = float(task_timing["teacher_rate"])
    contact_time = pre_swing_wait + scaled_t_hit
    required_end = pre_swing_wait + scaled_t_cycle
    physics_steps = 2
    ball_contact = list(task_geometry["ball_contact_w_m"])
    landing_aim = list(task_geometry["landing_aim_w_xy_m"])
    if positive:
        paddle_contact = {
            "time_s": contact_time,
            "ball_center_m": ball_contact,
            "face_mesh_sha256": face_mesh_sha256,
            "selected_face_sign": task_geometry["mount_normal_sign"],
            "contact_model_sha256": contact_model_sha256,
            "face_edge_clearance_m": 0.02,
            "required_face_edge_clearance_m": 0.01,
            "selected_face_return_normal_x_margin": 1.0,
            "relative_normal_speed_mps": -1.0,
            "face_point_m": ball_contact,
            "face_point_local_m": [0.0, 0.0, 0.0],
            "face_triangle_index": 0,
            "selected_face_return_normal_w": [1.0, 0.0, 0.0],
            "face_point_velocity_mps": [3.0, 0.0, 0.0],
            "velocity_minus_mps": [-2.0, 0.0, 0.0],
            "velocity_plus_mps": [3.0, 0.0, 1.0],
            "spin_minus_radps": [0.0, 0.0, 0.0],
            "spin_plus_radps": [0.0, 0.0, 0.0],
            "e_eff": 0.8,
        }
        net_crossing = {
            "time_s": contact_time + 0.1,
            "ball_center_z_m": (
                physics_bounds["ball_center_net_top_z_m"] + 0.1
            ),
            "required_center_z_m": (
                physics_bounds["ball_center_net_top_z_m"]
            ),
            "ball_center_y_m": 0.0,
            "clearance_m": 0.1,
            "cleared": True,
        }
        first_landing = {
            "time_s": contact_time + 0.2,
            "authority": "venue_fitted_table_impulse",
            "ball_center_xy_m": landing_aim,
            "ball_center_z_m": (
                physics_bounds["ball_center_surface_z_m"]
            ),
            "velocity_minus_mps": [3.0, 0.0, -1.0],
        }
    else:
        paddle_contact = None
        net_crossing = None
        first_landing = None
    return {
        "timestep_s": timestep_s,
        "verdict": "PASS" if positive else "FAIL",
        "failure_reasons": [] if positive else [negative_reason],
        "paddle_impulse_count": 1 if positive else 0,
        "teacher_reference_hit": {
            "physical_ball_center_m": ball_contact,
            "manifest_contact_center_m": ball_contact,
            "center_error_m": 0.0,
            "site_target_task_m": list(
                task_geometry["racket_site_target_w_m"]
            ),
            "site_target_error_m": 0.0,
            "site_speed_actual_mps": 3.0,
            "site_speed_task_mps": 3.0,
            "reference_site_speed_manifest_mps": 3.0,
            "site_speed_error_mps": 0.0,
            "face_center_velocity_actual_mps": list(
                task_geometry["racket_face_center_velocity_w_mps"]
            ),
            "face_center_velocity_task_mps": list(
                task_geometry["racket_face_center_velocity_w_mps"]
            ),
            "face_center_velocity_error_mps": 0.0,
            "site_velocity_task_mps": list(
                task_geometry["racket_site_velocity_w_mps"]
            ),
            "site_velocity_error_mps": 0.0,
            "angular_velocity_task_radps": list(
                task_geometry[
                    "racket_command_angular_velocity_w_radps"
                ]
            ),
            "angular_velocity_error_radps": 0.0,
            "selected_face_normal_w": list(
                task_geometry["racket_normal_w"]
            ),
            "task_face_normal_w": list(
                task_geometry["racket_normal_w"]
            ),
            "face_normal_angle_error_rad": 0.0,
            "selected_face_return_normal_x_margin": 1.0,
            "fixed_center_tolerance_m": 0.005,
            "fixed_site_speed_tolerance_mps": 0.10,
        },
        "paddle_contact": paddle_contact,
        "net_crossing": net_crossing,
        "first_landing": first_landing,
        "first_landing_task_aim_w_xy_m": landing_aim,
        "first_landing_task_error_m": 0.0,
        "incoming_task_state_error": {
            "velocity_mps": 0.0,
            "velocity_tolerance_mps": 0.1,
            "spin_radps": 0.0,
            "spin_tolerance_radps": 0.1,
        },
        "ball_net_collision": None,
        "activation_time_s": 0.0,
        "incoming_table_bounces": 1 if positive else 0,
        "return_table_bounces": 1 if positive else 0,
        "incoming_table_bounce_times_s": [0.5] if positive else [],
        "return_table_bounce_times_s": (
            [contact_time + 0.2] if positive else []
        ),
        "table_contacts": [],
        "event_order_violations": [],
        "ball_forbidden_contacts": [],
        "shadow_probe_samples": physics_steps,
        "shadow_robot_obstacle_near_contacts": [],
        "shadow_self_near_contacts": [],
        "shadow_relative_motion_certificate": {
            "intervals": 1,
            "covered_duration_s": required_end,
            "required_duration_s": required_end,
            "max_ball_path_bound_m": 0.001,
            "max_robot_surface_path_bound_m": 0.001,
            "ball_plus_robot_guard_margin_m": 0.01,
            "two_robot_surface_guard_margin_m": 0.01,
            "self_collision_pair_count": 1,
            "robot_geom_count": 1,
            "obstacle_names": list(CORE.table_scene.OBSTACLE_NAMES),
            "motion_frame_knots_are_interval_boundaries": True,
            "whole_prep_hit_recovery_required": True,
        },
        "native_ball_contact_count": 0,
        "robot_obstacle_contacts": [],
        "self_contacts": [],
        "joint_limit_violation": None,
        "fall": None,
        "simulation_window": {
            "start_time_s": 0.0,
            "executed_end_time_s": required_end,
            "required_ready_to_recovery_end_time_s": required_end,
            "task_pre_swing_wait_s": pre_swing_wait,
            "executed_pre_swing_wait_s": pre_swing_wait,
            "teacher_rate": teacher_rate,
            "scaled_t_hit_s": scaled_t_hit,
            "scaled_t_cycle_s": scaled_t_cycle,
            "physics_steps": physics_steps,
            "exact_teacher_pose_safety_scans": 2 * physics_steps,
            "post_dynamics_safety_scans": physics_steps,
            "expected_render_frames": 10,
        },
        "mandatory_gates": {
            "physical_ball_selected_face_return_and_first_landing": positive,
            "teacher_matches_frozen_solver_task": positive,
            "teacher_robot_and_racket_table_net_post_clearance": True,
        },
        "frame_metrics": [
            {
                "physics_step": index,
                "time_s": (index + 1) * timestep_s,
                "ball_active": True,
                "ball_position_m": [3.0, 0.0, 1.3],
                "ball_velocity_mps": [-2.0, 0.0, 0.0],
                "ball_spin_radps": [0.0, 0.0, 0.0],
                "racket_face_center_m": ball_contact,
                "racket_face_normal_w": list(
                    task_geometry["racket_normal_w"]
                ),
                "paddle_impulse_count": 1 if positive else 0,
                "robot_obstacle_contact_records": 0,
                "self_contact_records": 0,
                "ball_forbidden_contact_records": 0,
                "shadow_robot_obstacle_near_contact_records": 0,
                "shadow_self_near_contact_records": 0,
                "root_z_m": 1.0,
                "root_tilt_rad": 0.0,
            }
            for index in range(physics_steps)
        ],
        "ready_recovery": _ready_recovery_fixture(),
    }


def _retained_capsule_fixture(tmp_path: Path, *, pre_finalize=None):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "fixture@example.com")
    _git(source, "config", "user.name", "Fixture")
    (source / ".gitignore").write_text(
        "hope_training/whole_body_tracking/artifacts/\n",
        encoding="utf-8",
    )
    (source / "fixture.txt").write_text("fixture\n", encoding="utf-8")
    motion_dir = source / "motions"
    motion_dir.mkdir()
    motion_bytes = {}
    for index, action_id in enumerate(
        ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
    ):
        path = motion_dir / f"{action_id}.npz"
        path.write_bytes(f"motion:{index}:{action_id}\n".encode())
        motion_bytes[action_id] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    action_uids = {
        action_id: ADMISSION._derive_action_uid(
            action_id,
            ADMISSION.FRESH_N5_ACTION_FAMILY[action_id],
            motion_bytes[action_id],
        )
        for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
    }
    solver_source_names = (
        "continuous_questions.py",
        "hope_commands.py",
        "racket_contact_geometry.py",
        "stroke_adapt_torch.py",
        "virtual_ball.py",
    )
    solver_source_dir = (
        source
        / "hope_training/whole_body_tracking/source/"
        "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    )
    solver_source_dir.mkdir(parents=True)
    solver_source_sha = {}
    for name in solver_source_names:
        path = solver_source_dir / name
        path.write_text(f"SOURCE = {name!r}\n", encoding="utf-8")
        solver_source_sha[name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    contact_geometry_payload = (
        CORE.racket_geometry._production.GEOMETRY_SOURCE_PAYLOAD
    )
    geometry_payload_sha = _canonical_sha256(contact_geometry_payload)
    contact_geometry_contract = {
        "payload": contact_geometry_payload,
        "sha256": geometry_payload_sha,
    }
    solver_payload = {
        "kind": "fixture_solver",
        "implementation_source_sha256": solver_source_sha,
        "contact_geometry": contact_geometry_contract,
    }
    physics_bounds = {
        "table_surface_z_m": 0.76,
        "ball_center_net_top_z_m": 0.90,
        "ball_center_surface_z_m": 0.78,
        "opponent_near_x_m": 1.37,
        "net_x_m": 0.0,
        "opponent_far_x_m": 2.74,
        "minimum_landing_depth_m": 0.10,
        "table_half_width_m": 0.7625,
        "capture_radius_m": 0.04,
        "minimum_approach_speed_mps": 0.10,
    }
    physics_payload = {
        "kind": "fixture_physics",
        "geometry_and_grading": physics_bounds,
    }
    solver_profile_sha = hashlib.sha256(
        BOOTSTRAP._canonical_json_bytes(solver_payload)
    ).hexdigest()
    physics_profile_sha = hashlib.sha256(
        BOOTSTRAP._canonical_json_bytes(physics_payload)
    ).hexdigest()
    profile_payload = {
        "solver_payload": solver_payload,
        "physics_payload": physics_payload,
        "solver_profile_sha256": solver_profile_sha,
        "physics_profile_sha256": physics_profile_sha,
        "solver_implementation_source_sha256": solver_source_sha,
        "contact_geometry": contact_geometry_contract,
    }
    (source / "profile.json").write_text(
        json.dumps(
            profile_payload, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    assert geometry_payload_sha == (
        CORE.racket_geometry.GEOMETRY_SOURCE_SHA256
    )
    mesh_dir = source / "meshes"
    mesh_dir.mkdir()
    face_mesh_path = mesh_dir / "selected_racket_face.obj"
    face_mesh_path.write_text(
        "v 0 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n",
        encoding="utf-8",
    )
    face_mesh_sha = hashlib.sha256(
        face_mesh_path.read_bytes()
    ).hexdigest()
    ball_profile = {
        "time_to_contact_center_s": 1.2,
        "time_to_contact_min_s": 1.1,
        "time_to_contact_max_s": 1.3,
        "incoming_speed_min_mps": 1.0,
        "incoming_speed_max_mps": 3.0,
        "spin_magnitude_min_radps": 0.0,
        "spin_magnitude_max_radps": 20.0,
    }
    ball_profile_sha = _canonical_sha256(ball_profile)
    solver_receipt_dir = source / "solver_receipts"
    solver_receipt_dir.mkdir()
    launch_upstream_dir = source / "launch_upstream"
    launch_upstream_dir.mkdir()
    launch_artifact_dir = source / "launch_artifacts"
    launch_artifact_dir.mkdir()
    launch_fixture_metadata = {}
    raw_bindings = {}

    def build_raw_case(
        *,
        action_id: str,
        role: str,
        index: int,
    ) -> dict:
        support = role == "support_positive"
        contact = [0.51, 0.0, 1.2] if support else [0.5, 0.0, 1.2]
        time_to_contact = 1.25 if support else 1.2
        incoming_velocity = (
            [-2.1, 0.0, 0.0] if support else [-2.0, 0.0, 0.0]
        )
        launch_payload = {
            "activation_time_s": 0.0,
            "position_w_m": [3.0, 0.0, 1.3],
            "velocity_w_mps": incoming_velocity,
            "spin_w_radps": [0.0, 0.0, 0.0],
            "required_incoming_table_bounces": 1,
        }
        launch = {
            **launch_payload,
            "state_sha256": _canonical_sha256(launch_payload),
        }
        sample_seed = index + 1
        proposal = {
            "action_id": action_id,
            "action_uid": action_uids[action_id],
            "motion_sha256": motion_bytes[action_id],
            "sample_seed": sample_seed,
            "sample_index": sample_seed,
            "ball_contact_w_m": contact,
            "time_to_contact_s": time_to_contact,
            "incoming_velocity_w_mps": incoming_velocity,
            "incoming_spin_w_radps": [0.0, 0.0, 0.0],
            "base_spawn_w_m": [0.0, 0.0, 1.0],
            "base_goal_w_m": [0.0, 0.0, 1.0],
            "landing_aim_w_xy_m": [2.5, 0.0],
            "launch": launch,
        }
        proposal_sha = _canonical_sha256(proposal)
        exact_geometry = (
            CORE.racket_geometry._production.solve_exact_face_contact(
                ball_contact_w_m=contact,
                racket_face_center_velocity_w_mps=[3.0, 0.0, 0.0],
                solved_raw_a_normal_w=[1.0, 0.0, 0.0],
                mount_normal_sign=1,
                reference_racket_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
                reference_racket_angular_velocity_w_radps=[
                    0.0,
                    0.0,
                    0.0,
                ],
                reference_racket_site_speed_mps=3.0,
                teacher_rate_min=1.0,
                teacher_rate_max=1.0,
            )
        )
        task = {
            "action_id": action_id,
            "action_uid": action_uids[action_id],
            "motion_sha256": motion_bytes[action_id],
            "ball_proposal_sha256": proposal_sha,
            "mount_normal_sign": 1,
            "ball_contact_w_m": contact,
            "racket_site_target_w_m": list(
                exact_geometry.racket_site_target_w_m
            ),
            "racket_normal_w": [1.0, 0.0, 0.0],
            "reference_racket_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "reference_racket_angular_velocity_w_radps": [
                0.0,
                0.0,
                0.0,
            ],
            "racket_command_quat_wxyz": list(
                exact_geometry.racket_command_quat_wxyz
            ),
            "racket_face_center_velocity_w_mps": list(
                exact_geometry.racket_face_center_velocity_w_mps
            ),
            "racket_site_velocity_w_mps": list(
                exact_geometry.racket_site_velocity_w_mps
            ),
            "racket_command_angular_velocity_w_radps": list(
                exact_geometry.racket_command_angular_velocity_w_radps
            ),
            "geometry_source_sha256": geometry_payload_sha,
            "reference_t_hit_s": 1.0,
            "reference_t_cycle_s": 2.0,
            "reference_racket_site_speed_mps": 3.0,
            "required_racket_site_speed_mps": 3.0,
            "reaction_margin_s": 0.1,
            "teacher_rate_min": 1.0,
            "teacher_rate_max": 1.0,
            "teacher_rate": 1.0,
            "scaled_t_hit_s": 1.0,
            "scaled_t_cycle_s": 2.0,
            "pre_swing_wait_s": time_to_contact - 1.0,
            "solver_residual_m": 0.01,
            "landing_aim_w_xy_m": [2.5, 0.0],
            "solver_profile_sha256": solver_profile_sha,
            "physics_profile_sha256": physics_profile_sha,
        }
        task_sha = _canonical_sha256(task)
        if role in ADMISSION._FRESH_N5_PHYSICAL_TASK_POSITIVE_ROLES:
            fault = {"kind": "none"}
            expected_verdict = "PASS"
            expected_reason = None
        elif role == "negative_t_hit_offset":
            fault = {
                "kind": "teacher_t_hit_offset",
                "offset_s": 0.05,
            }
            expected_verdict = "FAIL"
            expected_reason = (
                ADMISSION._FRESH_N5_PHYSICAL_TASK_NEGATIVE_REASON[role]
            )
        elif role == "negative_face_sign":
            fault = {"kind": "selected_face_sign_flip"}
            expected_verdict = "FAIL"
            expected_reason = (
                ADMISSION._FRESH_N5_PHYSICAL_TASK_NEGATIVE_REASON[role]
            )
        else:
            fault = {
                "kind": "launch_velocity_delta",
                "launch_velocity_delta_w_mps": [0.0, 0.3, 0.0],
            }
            expected_verdict = "FAIL"
            expected_reason = (
                ADMISSION._FRESH_N5_PHYSICAL_TASK_NEGATIVE_REASON[role]
            )
        solver_identity = {
            "artifact_type": "frozen_ball_to_task_solver_execution_v1",
            "execution_id": f"fixture:{action_id}",
            "executed_before_gate": True,
            "solver_replayed_exact": True,
            "selector_executed": False,
            "action_identity_frozen": True,
            "action_switching_allowed": False,
            "hardware_authorized": False,
        }
        case_id = f"{action_id}:{role}"
        binding_payload = {
            "action_id": action_id,
            "action_uid": action_uids[action_id],
            "motion_sha256": motion_bytes[action_id],
            "case_id": case_id,
            "case_role": role,
            "sample_seed": sample_seed,
            "ball_proposal_sha256": proposal_sha,
            "task_payload_sha256": task_sha,
            "solver_execution_identity_sha256": _canonical_sha256(
                solver_identity
            ),
            "fault_injection": fault,
            "expected_physical_verdict": expected_verdict,
            "expected_failure_reason": expected_reason,
        }
        return {
            "case_id": case_id,
            "case_role": role,
            "sample_seed": sample_seed,
            "expected_physical_verdict": expected_verdict,
            "expected_failure_reason": expected_reason,
            "ball_proposal": proposal,
            "ball_proposal_sha256": proposal_sha,
            "task_payload": task,
            "task_payload_sha256": task_sha,
            "fault_injection": fault,
            "case_binding_sha256": _canonical_sha256(binding_payload),
        }

    manifest_actions = []
    for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS:
        cases = [
            build_raw_case(action_id=action_id, role=role, index=index)
            for index, role in enumerate(
                ADMISSION._FRESH_N5_PHYSICAL_TASK_CASE_ROLES
            )
        ]
        center_launch = cases[0]["ball_proposal"]["launch"]
        units = {
            "position": "m",
            "velocity": "m/s",
            "spin": "rad/s",
            "time": "s",
        }
        launch_state = {
            "source": "recorded_pre_hit_state_v1",
            "activation_time_s": center_launch["activation_time_s"],
            "position_w_m": center_launch["position_w_m"],
            "velocity_w_mps": center_launch["velocity_w_mps"],
            "spin_w_radps": center_launch["spin_w_radps"],
            "required_incoming_table_bounces": center_launch[
                "required_incoming_table_bounces"
            ],
        }
        upstream_receipt = {
            "schema_version": 1,
            "artifact_type": "recorded_ball_state_series_v1",
            "action_id": action_id,
            "action_uid": action_uids[action_id],
            "motion_sha256": motion_bytes[action_id],
            "coordinate_frame": "mujoco_world",
            "units": units,
            "samples": [
                {
                    "sample_time_s": center_launch[
                        "activation_time_s"
                    ],
                    "position_w_m": center_launch["position_w_m"],
                    "velocity_w_mps": center_launch["velocity_w_mps"],
                    "spin_w_radps": center_launch["spin_w_radps"],
                }
            ],
        }
        upstream_receipt["receipt_payload_sha256"] = _canonical_sha256(
            upstream_receipt
        )
        upstream_path = launch_upstream_dir / f"{action_id}.json"
        upstream_path.write_text(
            json.dumps(
                upstream_receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        upstream_sha = hashlib.sha256(
            upstream_path.read_bytes()
        ).hexdigest()
        source_artifact = {
            "schema_version": 1,
            "artifact_type": "recorded_pre_hit_state_v1",
            "action_id": action_id,
            "action_uid": action_uids[action_id],
            "motion_sha256": motion_bytes[action_id],
            "coordinate_frame": "mujoco_world",
            "units": units,
            "authorization": {
                "physical_gate_input_authorized": True,
                "hardware_authorized": False,
            },
            "launch_state": launch_state,
            "upstream_evidence_path": (
                f"launch_upstream/{action_id}.json"
            ),
            "upstream_evidence_sha256": upstream_sha,
            "recording_sample_index": 0,
            "recording_sample_time_s": center_launch[
                "activation_time_s"
            ],
        }
        source_artifact_path = (
            launch_artifact_dir / f"{action_id}.json"
        )
        source_artifact_path.write_text(
            json.dumps(
                source_artifact,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        source_artifact_sha = hashlib.sha256(
            source_artifact_path.read_bytes()
        ).hexdigest()
        physical_launch_payload = {
            **launch_state,
            "source_artifact_path": (
                f"launch_artifacts/{action_id}.json"
            ),
            "source_artifact_sha256": source_artifact_sha,
        }
        physical_launch = {
            **physical_launch_payload,
            "state_sha256": _canonical_sha256(
                physical_launch_payload
            ),
        }
        launch_fixture_metadata[action_id] = {
            "upstream_receipt_payload_sha256": upstream_receipt[
                "receipt_payload_sha256"
            ],
            "upstream_sha256": upstream_sha,
            "source_artifact_sha256": source_artifact_sha,
        }
        solver_identity = {
            "artifact_type": "frozen_ball_to_task_solver_execution_v1",
            "execution_id": f"fixture:{action_id}",
            "executed_before_gate": True,
            "solver_replayed_exact": True,
            "selector_executed": False,
            "action_identity_frozen": True,
            "action_switching_allowed": False,
            "hardware_authorized": False,
        }
        external_receipt = {
            "schema_version": 1,
            "artifact_type": (
                "frozen_action_ball_solver_execution_receipt_v1"
            ),
            "producer": {
                "source_path": (
                    "hope_training/whole_body_tracking/source/"
                    "whole_body_tracking/whole_body_tracking/tasks/"
                    "tracking/mdp/hope_commands.py"
                ),
                "source_sha256": solver_source_sha["hope_commands.py"],
                "runtime_receipt_type": "ActionBallTaskReceipt",
                "exact_solver_replay_required": True,
                "selector_executed": False,
                "hardware_authorized": False,
            },
            "action_identity": {
                "action_id": action_id,
                "action_uid": action_uids[action_id],
                "motion_sha256": motion_bytes[action_id],
            },
            "profile_identity": {
                "ball_profile_sha256": ball_profile_sha,
                "solver_profile_sha256": solver_profile_sha,
                "physics_profile_sha256": physics_profile_sha,
                "solver_implementation_source_sha256": (
                    solver_source_sha
                ),
                "geometry_source_sha256": geometry_payload_sha,
            },
            "solver_execution_identity": solver_identity,
            "cases": cases,
        }
        external_receipt["receipt_payload_sha256"] = _canonical_sha256(
            external_receipt
        )
        receipt_path = solver_receipt_dir / f"{action_id}.json"
        receipt_path.write_text(
            json.dumps(
                external_receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        raw_binding = {
            "schema_version": 1,
            "authority": (
                "pre_registered_frozen_action_ball_solver_receipt_v1"
            ),
            "action_id": action_id,
            "action_uid": action_uids[action_id],
            "motion_sha256": motion_bytes[action_id],
            "ball_profile_sha256": ball_profile_sha,
            "solver_profile_sha256": solver_profile_sha,
            "physics_profile_sha256": physics_profile_sha,
            "solver_implementation_source_sha256": solver_source_sha,
            "solver_execution_receipt_path": (
                f"solver_receipts/{action_id}.json"
            ),
            "solver_execution_receipt_sha256": hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            "solver_execution_identity": solver_identity,
            "solver_execution_identity_sha256": _canonical_sha256(
                solver_identity
            ),
            "selector_executed": False,
            "action_identity_frozen": True,
            "cases": cases,
            "cases_sha256": _canonical_sha256(cases),
        }
        raw_bindings[action_id] = raw_binding
        manifest_actions.append(
            {
                "action_id": action_id,
                "family": ADMISSION.FRESH_N5_ACTION_FAMILY[action_id],
                "action_uid": action_uids[action_id],
                "motion_path": f"motions/{action_id}.npz",
                "motion_sha256": motion_bytes[action_id],
                "strike_phase": 0.5,
                "reference_t_hit_s": 1.0,
                "reference_t_cycle_s": 2.0,
                "reference_racket_site_speed_mps": 3.0,
                "reaction_margin_s": 0.1,
                "teacher_rate_min": 1.0,
                "teacher_rate_max": 1.0,
                "mount_normal_sign": 1,
                "ball_profile": ball_profile,
                "physical_ball_launch": physical_launch,
                "physical_task_binding": raw_binding,
            }
        )
    manifest_payload = {
        "schema_version": 3,
        "manifest_id": "fixture-fresh-n5",
        "mobility_mode": "no_move",
        "prototype": {"scope": "upper"},
        "solver_profile_sha256": solver_profile_sha,
        "physics_profile_sha256": physics_profile_sha,
        "racket_geometry_contract": {
            "schema_version": 2,
            "semantics": "exact_face_contact_v2",
            "ball_target_point": (
                "physical_ball_center_at_native_contact"
            ),
            "site_target_mapping": "site_target_from_ball_center",
            "face_velocity_mapping": (
                "site_linear_plus_omega_cross_face_center_offset"
            ),
            "source_path": (
                "hope_training/whole_body_tracking/source/"
                "whole_body_tracking/whole_body_tracking/tasks/"
                "tracking/mdp/racket_contact_geometry.py"
            ),
            "source_sha256": solver_source_sha[
                "racket_contact_geometry.py"
            ],
            "geometry_source_sha256": geometry_payload_sha,
        },
        "action_order": list(
            ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
        ),
        "actions": manifest_actions,
    }
    (source / "manifest.json").write_text(
        json.dumps(
            manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    strict_manifest_payload = {
        key: value
        for key, value in manifest_payload.items()
        if key != "racket_geometry_contract"
    }
    strict_manifest_payload["actions"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"physical_ball_launch", "physical_task_binding"}
        }
        for row in manifest_actions
    ]
    (source / "strict_manifest.json").write_text(
        json.dumps(
            strict_manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "physical_gate_materialization_receipt.json").write_text(
        '{"fixture":"physical-gate-materialization"}\n',
        encoding="utf-8",
    )
    (source / "venue.yaml").write_text("venue: fixture\n", encoding="utf-8")
    (source / "contact_model.py").write_text(
        "MODEL = 'fixture'\n", encoding="utf-8"
    )
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "fixture")
    commit = _git(source, "rev-parse", "HEAD").stdout.decode().strip()
    store = source / BOOTSTRAP.CAPSULE_STORE_REPO_PATH
    store.mkdir(parents=True)
    staging = store / "fixture-run"
    artifacts = staging / BOOTSTRAP.CAPSULE_ARTIFACTS_DIRNAME
    artifacts.mkdir(parents=True)
    checkout = staging / BOOTSTRAP.CAPSULE_CHECKOUT_DIRNAME
    _git(
        source,
        "worktree",
        "add",
        "--detach",
        str(checkout),
        commit,
    )
    BOOTSTRAP._make_tree_read_only(checkout)
    manifest_sha = hashlib.sha256(
        (checkout / "manifest.json").read_bytes()
    ).hexdigest()
    strict_manifest_sha = hashlib.sha256(
        (checkout / "strict_manifest.json").read_bytes()
    ).hexdigest()
    materialization_sha = hashlib.sha256(
        (
            checkout / "physical_gate_materialization_receipt.json"
        ).read_bytes()
    ).hexdigest()
    profile_sha = hashlib.sha256(
        (checkout / "profile.json").read_bytes()
    ).hexdigest()
    trust_sha = hashlib.sha256(b"trust").hexdigest()
    checkout_solver_source_dir = (
        checkout
        / "hope_training/whole_body_tracking/source/"
        "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    )
    checkout_solver_source_sha = {
        name: hashlib.sha256(
            (checkout_solver_source_dir / name).read_bytes()
        ).hexdigest()
        for name in solver_source_names
    }
    solver_execution_receipts = {}
    for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS:
        path = checkout / "solver_receipts" / f"{action_id}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        solver_execution_receipts[action_id] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "receipt_payload_sha256": raw[
                "receipt_payload_sha256"
            ],
        }
    launch_source_receipts = {}
    for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS:
        source_artifact_path = (
            checkout / "launch_artifacts" / f"{action_id}.json"
        )
        upstream_path = (
            checkout / "launch_upstream" / f"{action_id}.json"
        )
        metadata = launch_fixture_metadata[action_id]
        launch_source_receipts[action_id] = {
            "artifact": {
                "path": str(source_artifact_path),
                "sha256": metadata["source_artifact_sha256"],
                "size_bytes": source_artifact_path.stat().st_size,
            },
            "artifact_type": "recorded_pre_hit_state_v1",
            "action_id": action_id,
            "action_uid": action_uids[action_id],
            "motion_sha256": motion_bytes[action_id],
            "coordinate_frame": "mujoco_world",
            "units": {
                "position": "m",
                "velocity": "m/s",
                "spin": "rad/s",
                "time": "s",
            },
            "upstream_evidence": {
                "path": str(upstream_path),
                "sha256": metadata["upstream_sha256"],
                "size_bytes": upstream_path.stat().st_size,
            },
            "upstream_receipt_payload_sha256": metadata[
                "upstream_receipt_payload_sha256"
            ],
            "recording_sample_index": 0,
            "recording_sample_time_s": 0.0,
        }
    venue_sha = hashlib.sha256(
        (checkout / "venue.yaml").read_bytes()
    ).hexdigest()
    contact_sha = hashlib.sha256(
        (checkout / "contact_model.py").read_bytes()
    ).hexdigest()
    execution_marker = {
        "capsule_layout": BOOTSTRAP.CAPSULE_LAYOUT,
        "code_commit": commit,
        "source_repo": str(source),
        "capsule_staging_root": str(staging),
        "checkout_root": str(checkout),
        "artifacts_root": str(artifacts),
    }
    runtime_files = [
        {
            "path": str(checkout / "strict_manifest.json"),
            "sha256": strict_manifest_sha,
            "size_bytes": (
                checkout / "strict_manifest.json"
            ).stat().st_size,
            "roles": ["strict_training_manifest"],
        },
        {
            "path": str(checkout / "manifest.json"),
            "sha256": manifest_sha,
            "size_bytes": (checkout / "manifest.json").stat().st_size,
            "roles": ["physical_gate_manifest"],
        },
        {
            "path": str(
                checkout / "physical_gate_materialization_receipt.json"
            ),
            "sha256": materialization_sha,
            "size_bytes": (
                checkout / "physical_gate_materialization_receipt.json"
            ).stat().st_size,
            "roles": ["physical_gate_materialization_receipt"],
        },
        {
            "path": str(checkout / "profile.json"),
            "sha256": profile_sha,
            "size_bytes": (checkout / "profile.json").stat().st_size,
            "roles": ["profile_pins"],
        },
        *[
        {
            "path": str(
                checkout / "motions" / f"{action_id}.npz"
            ),
            "sha256": motion_bytes[action_id],
            "size_bytes": (
                checkout / "motions" / f"{action_id}.npz"
            ).stat().st_size,
            "roles": [f"motion:{action_id}"],
        }
        for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
        ],
        *[
        {
            "path": solver_execution_receipts[action_id]["path"],
            "sha256": solver_execution_receipts[action_id]["sha256"],
            "size_bytes": (
                checkout / "solver_receipts" / f"{action_id}.json"
            ).stat().st_size,
            "roles": [f"solver_execution_receipt:{action_id}"],
        }
        for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
        ],
        *[
        {
            "path": launch_source_receipts[action_id]["artifact"]["path"],
            "sha256": launch_source_receipts[action_id]["artifact"][
                "sha256"
            ],
            "size_bytes": launch_source_receipts[action_id][
                "artifact"
            ]["size_bytes"],
            "roles": [f"launch_source_artifact:{action_id}"],
        }
        for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
        ],
        *[
        {
            "path": launch_source_receipts[action_id][
                "upstream_evidence"
            ]["path"],
            "sha256": launch_source_receipts[action_id][
                "upstream_evidence"
            ]["sha256"],
            "size_bytes": launch_source_receipts[action_id][
                "upstream_evidence"
            ]["size_bytes"],
            "roles": [f"launch_upstream_evidence:{action_id}"],
        }
        for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
        ],
    ]
    runtime_files.extend(
        [
            *[
                {
                    "path": str(
                        checkout_solver_source_dir / name
                    ),
                    "sha256": checkout_solver_source_sha[name],
                    "size_bytes": (
                        checkout_solver_source_dir / name
                    ).stat().st_size,
                    "roles": sorted(
                        [
                            f"solver_source:{name}",
                            *(
                                ["racket_geometry_production_source"]
                                if name == "racket_contact_geometry.py"
                                else []
                            ),
                        ]
                    ),
                }
                for name in solver_source_names
            ],
            {
                "path": str(checkout / "venue.yaml"),
                "sha256": venue_sha,
                "size_bytes": (checkout / "venue.yaml").stat().st_size,
                "roles": ["venue_yaml"],
            },
            {
                "path": str(checkout / "contact_model.py"),
                "sha256": contact_sha,
                "size_bytes": (
                    checkout / "contact_model.py"
                ).stat().st_size,
                "roles": ["fitted_contact_model"],
            },
        ]
    )
    video_dir = artifacts / "videos"
    video_dir.mkdir()
    video_rows = {}
    for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS:
        video_path = (
            video_dir / f"{action_id}_fitted_teacher_ball.mp4"
        )
        video_path.write_bytes(
            f"fixture-mp4:{action_id}\n".encode()
        )
        video_rows[action_id] = {
            "status": "WRITTEN",
            "path": str(video_path),
            "capsule_relative_path": (
                "artifacts/videos/"
                f"{action_id}_fitted_teacher_ball.mp4"
            ),
            "sha256": hashlib.sha256(
                video_path.read_bytes()
            ).hexdigest(),
            "size_bytes": video_path.stat().st_size,
            "frames": 10,
            "fps": 30,
            "camera": "torso_follow",
            "evidence_role": (
                "human_visualization_only_not_physical_or_analytic_grader"
            ),
        }
    convergence_metrics = {
        name: 0.0
        for name in ADMISSION._FRESH_N5_CONVERGENCE_METRICS
    }
    convergence_tolerances = {
        name: 0.1
        for name in ADMISSION._FRESH_N5_CONVERGENCE_METRICS
    }
    passing_convergence = {
        "pass": True,
        "metrics": convergence_metrics,
        "failure_reasons": [],
        "tolerances": convergence_tolerances,
    }

    def physical_task_binding(action_id: str) -> dict:
        raw_binding = raw_bindings[action_id]
        replay_cases = []
        for raw_case in raw_binding["cases"]:
            role = raw_case["case_role"]
            positive = (
                role
                in ADMISSION._FRESH_N5_PHYSICAL_TASK_POSITIVE_ROLES
            )
            expected_verdict = raw_case["expected_physical_verdict"]
            expected_reason = raw_case["expected_failure_reason"]
            proposal = raw_case["ball_proposal"]
            task = raw_case["task_payload"]
            task_geometry = {
                key: task[key]
                for key in ADMISSION._FRESH_N5_TASK_GEOMETRY_KEYS
            }
            task_timing = {
                key: task[key]
                for key in ADMISSION._FRESH_N5_TASK_TIMING_KEYS
            }
            case_dt = {
                "0.0010": _physical_dt_result_fixture(
                    case_role=role,
                    timestep_s=0.001,
                    task_timing=task_timing,
                    task_geometry=task_geometry,
                    physics_bounds=physics_bounds,
                    contact_model_sha256=contact_sha,
                    face_mesh_sha256=face_mesh_sha,
                ),
                "0.0005": _physical_dt_result_fixture(
                    case_role=role,
                    timestep_s=0.0005,
                    task_timing=task_timing,
                    task_geometry=task_geometry,
                    physics_bounds=physics_bounds,
                    contact_model_sha256=contact_sha,
                    face_mesh_sha256=face_mesh_sha,
                ),
            }
            fault = raw_case["fault_injection"]
            nominal_sign = task["mount_normal_sign"]
            nominal_wait = task["pre_swing_wait_s"]
            nominal_velocity = proposal["launch"]["velocity_w_mps"]
            fault_application = {
                "kind": fault["kind"],
                "applied": True,
                "nominal_mount_normal_sign": nominal_sign,
                "executed_mount_normal_sign": nominal_sign,
                "nominal_pre_swing_wait_s": nominal_wait,
                "executed_pre_swing_wait_s": nominal_wait,
                "nominal_launch_velocity_w_mps": nominal_velocity,
                "executed_launch_velocity_w_mps": nominal_velocity,
            }
            if role == "negative_t_hit_offset":
                fault_application["offset_s"] = fault["offset_s"]
                fault_application["executed_pre_swing_wait_s"] = (
                    nominal_wait + fault["offset_s"]
                )
                for result in case_dt.values():
                    result["simulation_window"][
                        "executed_pre_swing_wait_s"
                    ] = fault_application["executed_pre_swing_wait_s"]
            elif role == "negative_face_sign":
                fault_application["executed_mount_normal_sign"] = (
                    -nominal_sign
                )
            elif role == "negative_ball_state_mismatch":
                delta = fault["launch_velocity_delta_w_mps"]
                fault_application[
                    "launch_velocity_delta_w_mps"
                ] = delta
                fault_application[
                    "executed_launch_velocity_w_mps"
                ] = [
                    nominal + change
                    for nominal, change in zip(nominal_velocity, delta)
                ]
            control = {
                "expected_physical_verdict": expected_verdict,
                "expected_failure_reason": expected_reason,
                "observed_physical_verdict": expected_verdict,
                "observed_failure_reason": expected_reason,
                "observed_dt_verdicts": {
                    "0.0010": expected_verdict,
                    "0.0005": expected_verdict,
                },
                "fault_application": fault_application,
                "convergence_required": positive,
                "convergence_pass": True if positive else None,
                "control_verdict": "PASS",
                "failure_reasons": [],
            }
            convergence = (
                passing_convergence
                if positive
                else {
                    "pass": False,
                    "metrics": convergence_metrics,
                    "failure_reasons": ["expected_negative_control"],
                    "tolerances": convergence_tolerances,
                }
            )
            replay_cases.append(
                {
                    "case_id": raw_case["case_id"],
                    "case_role": role,
                    "sample_seed": raw_case["sample_seed"],
                    "expected_physical_verdict": expected_verdict,
                    "expected_failure_reason": expected_reason,
                    "ball_proposal_sha256": raw_case[
                        "ball_proposal_sha256"
                    ],
                    "task_payload_sha256": raw_case[
                        "task_payload_sha256"
                    ],
                    "solved_task_geometry_sha256": _canonical_sha256(
                        task_geometry
                    ),
                    "case_binding_sha256": raw_case[
                        "case_binding_sha256"
                    ],
                    "solver_execution_identity": raw_binding[
                        "solver_execution_identity"
                    ],
                    "task_timing": task_timing,
                    "task_geometry": task_geometry,
                    "dt_results": case_dt,
                    "convergence": convergence,
                    "control": control,
                    "observed_physical_verdict": expected_verdict,
                    "control_verdict": "PASS",
                    "failure_reasons": [],
                }
            )
        return {
            "ball_profile_sha256": raw_binding[
                "ball_profile_sha256"
            ],
            "solver_profile_sha256": solver_profile_sha,
            "physics_profile_sha256": physics_profile_sha,
            "solver_source_sha256": checkout_solver_source_sha,
            "solver_execution_receipt": (
                solver_execution_receipts[action_id]
            ),
            "cases_sha256": raw_binding["cases_sha256"],
            "case_order": list(
                ADMISSION._FRESH_N5_PHYSICAL_TASK_CASE_ROLES
            ),
            "cases": replay_cases,
        }

    formal_actions = []
    for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS:
        summary_binding = physical_task_binding(action_id)
        manifest_action = next(
            action
            for action in manifest_actions
            if action["action_id"] == action_id
        )
        formal_actions.append(
            {
                "action_id": action_id,
                "action_uid": action_uids[action_id],
                "motion_path": str(
                    checkout / "motions" / f"{action_id}.npz"
                ),
                "motion_sha256": motion_bytes[action_id],
                "launch": {
                    "source": manifest_action[
                        "physical_ball_launch"
                    ]["source"],
                    "state_sha256": manifest_action[
                        "physical_ball_launch"
                    ]["state_sha256"],
                    "source_receipt": launch_source_receipts[
                        action_id
                    ],
                },
                "face_geometry": {
                    "sign": 1,
                    "mesh_path": str(
                        checkout / "meshes/selected_racket_face.obj"
                    ),
                    "mesh_sha256": face_mesh_sha,
                    "outer_triangle_count": 1,
                    "geometry_contract_sha256": (
                        solver_source_sha[
                            "racket_contact_geometry.py"
                        ]
                    ),
                },
                "t_hit_s": 1.0,
                "t_cycle_s": 2.0,
                "reference_racket_site_speed_mps": 3.0,
                "dt_results": summary_binding["cases"][0][
                    "dt_results"
                ],
                "convergence": summary_binding["cases"][0][
                    "convergence"
                ],
                "physical_task_binding": summary_binding,
                "shared_ready_joint_linf_rad": 0.0,
                "recovery_joint_linf_rad": 0.0,
                "video": video_rows[action_id],
                "verdict": "PASS",
                "failure_reasons": [],
            }
        )
    ordered_action_ids = list(ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS)
    ordered_action_uids = [
        action_uids[action_id] for action_id in ordered_action_ids
    ]
    order_uid_digest = CORE.action_set_contract.order_uid_digest(
        ordered_action_ids, ordered_action_uids
    )
    action_set_contract = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.action_set_contract",
        "profile_id": "fresh_upper_nomove_n5_v3",
        "expected_n": len(ordered_action_ids),
        "scope": "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": ordered_action_ids,
        "ordered_action_uids": ordered_action_uids,
        "order_uid_digest_sha256": order_uid_digest,
        "manifest_path": "strict_manifest.json",
        "manifest_sha256": strict_manifest_sha,
        "experiment_name": "fixture-fresh-n5",
        "actor_obs_contract": "action_ball_n5",
        "actor_obs_width": 186,
        "namespace_identity": f"n5-{order_uid_digest[:12]}",
    }
    action_set_contract["contract_sha256"] = _canonical_sha256(
        action_set_contract
    )
    five_solid_payload = {
        "schema_version": 1,
        "obstacle_order": list(
            BOOTSTRAP.FIVE_SOLID_OBSTACLE_ORDER
        ),
        "fixture": True,
    }
    five_solid_sha = _canonical_sha256(five_solid_payload)
    five_solid_scene = {
        "five_solid_geometry_sha256": five_solid_sha,
        "geometry_payload": five_solid_payload,
        "obstacle_order": list(
            BOOTSTRAP.FIVE_SOLID_OBSTACLE_ORDER
        ),
        "under_table_keepout_role": "robot_only",
        "ball_keepout_native_pair_enabled": False,
        "ball_keepout_analytic_surface_enabled": False,
        "contact_force_threshold_n": 1.0e-6,
        "continuous_sweep_method": "fixture_sweep_v1",
        "compiled_by_dt": {
            timestep: {
                "five_solid_geometry_sha256": five_solid_sha,
                "assembled_xml_sha256": hashlib.sha256(
                    f"five-solid-{timestep}".encode()
                ).hexdigest(),
                "ball_keepout_native_pair_enabled": False,
                "ball_keepout_analytic_surface_enabled": False,
            }
            for timestep in ("0.0010", "0.0005")
        },
    }
    formal = {
        "schema_version": 1,
        "gate": "mujoco_teacher_motion_fitted_ball_gate",
        "contact_authority": "venue_fitted_swept_selected_face_v2",
        "native_ball_contact_enabled": False,
        "selector_executed": False,
        "ball_to_task_solver_executed": False,
        "ball_to_task_solver_executed_by_gate": False,
        "pre_registered_ball_to_task_solver_receipt_consumed": True,
        "solver_execution_receipt_authority": (
            "pre_registered_frozen_action_ball_solver_receipt_v1"
        ),
        "analytic_return_scorer_executed": False,
        "five_solid_safety_scene": five_solid_scene,
        "teacher_return_safety_rows": [],
        "expected_actions": 5,
        "expected_action_order": list(
            ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
        ),
        "action_set_contract": action_set_contract,
        "preflight": {
            "status": "PASS",
            "blockers": [],
            "evidence": {"fixture": True},
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "runtime_code_identity": {
            "code_commit": commit,
            "external_preexec": execution_marker,
            "committed_trust_spec": {
                "bindings": {
                    "training_manifest": {
                        "repo_path": "strict_manifest.json",
                        "sha256": strict_manifest_sha,
                    },
                    "physical_gate_manifest": {
                        "repo_path": "manifest.json",
                        "sha256": manifest_sha,
                    },
                    "physical_gate_materialization_receipt": {
                        "repo_path": (
                            "physical_gate_materialization_receipt.json"
                        ),
                        "sha256": materialization_sha,
                    },
                    "profile_pins": {
                        "repo_path": "profile.json",
                        "sha256": profile_sha,
                    },
                    "launch_evidence_trust_root": {
                        "repo_path": "launch_evidence_trust_root.json",
                        "sha256": trust_sha,
                    },
                }
            },
        },
        "formal_gate_executed": True,
        "runtime_environment": {"fixture": True},
        "runtime_input_snapshot": {
            "phase": "captured_before_runtime",
            "files": runtime_files,
            "post_runtime": {
                "stable": True,
                "checked_files": len(runtime_files),
                "check": "pinned_sha256_before_and_after_runtime",
            },
            "checkout_post_runtime": {
                "commit": commit,
                "clean": True,
            },
        },
        "runtime_code_identity_post_runtime": {"fixture": True},
        "runtime_code_identity_final": {"fixture": True},
        "status": "PASS",
        "verdict": "PASS",
        "manifest_id": "fixture-fresh-n5",
        "action_order": list(ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS),
        "base_mujoco_portable_identity_sha256": hashlib.sha256(
            b"portable"
        ).hexdigest(),
        "base_mujoco_verification_receipt_sha256": hashlib.sha256(
            b"verification"
        ).hexdigest(),
        "compiler_mesh_assets": [{"fixture": True}],
        "scene_contracts": {"fixture": True},
        "venue": {
            "path": str(checkout / "venue.yaml"),
            "sha256": venue_sha,
        },
        "contact_model": {
            "path": str(checkout / "contact_model.py"),
            "sha256": contact_sha,
        },
        "actions": formal_actions,
    }
    formal["receipt_payload_sha256"] = hashlib.sha256(
        BOOTSTRAP._canonical_json_bytes(formal)
    ).hexdigest()
    formal_path = staging / BOOTSTRAP.CAPSULE_FORMAL_RECEIPT_RELPATH
    formal_path.write_text(
        json.dumps(formal, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    if pre_finalize is not None:
        pre_finalize(
            source=source,
            staging=staging,
            checkout=checkout,
            artifacts=artifacts,
            formal_path=formal_path,
            commit=commit,
        )
    result = BOOTSTRAP._finalize_retained_capsule(
        source_repo=source,
        staging_root=staging,
        checkout_root=checkout,
        artifacts_root=artifacts,
        formal_receipt_path=formal_path,
        code_commit=commit,
        git=Path(subprocess.check_output(["which", "git"]).decode().strip()),
        core_return_code=0,
    )
    final_root = Path(result["capsule_root"])
    formal_final = Path(result["formal_receipt"]["path"])
    retained_final = Path(result["retained_capsule_receipt"]["path"])
    row = {
        "path": formal_final.relative_to(source).as_posix(),
        "sha256": result["formal_receipt"]["sha256"],
        "retained_capsule_receipt": {
            "path": retained_final.relative_to(source).as_posix(),
            "sha256": result["retained_capsule_receipt"]["sha256"],
        },
    }
    return source, final_root, formal_final, retained_final, row, result


def _fresh_binding_for_capsule(
    formal_path: Path, result: dict
) -> object:
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    selected = {
        row["action_id"]: row["motion_sha256"]
        for row in formal["actions"]
    }

    def digest(label: str) -> str:
        return hashlib.sha256(label.encode()).hexdigest()

    bank_shas = []
    for motion_id in ADMISSION.FRESH_N5_BANK_MOTION_IDS:
        for scope in ("upper", "full"):
            bank_shas.append(
                selected[motion_id]
                if scope == "upper" and motion_id in selected
                else digest(f"bank:{motion_id}:{scope}")
            )
    base_manifest = digest("base-manifest")
    append_manifest = digest("append-manifest")
    action_count = len(ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS)
    return ADMISSION.FreshN5BankPromotionBinding(
        purpose="training",
        bank_id="fresh_n5_fixture",
        scope="upper",
        registry_sha256=digest("registry"),
        alignment_sha256=digest("alignment"),
        motion_ids=ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS,
        npz_sha256=tuple(
            selected[action_id]
            for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
        ),
        canonical_ready_sha256=digest("ready"),
        canonical_ready_fk_sha256=digest("ready-fk"),
        build_manifest_sha256=tuple(
            (
                append_manifest
                if action_id in ADMISSION.FRESH_N5_APPEND_MOTION_IDS
                else base_manifest
            )
            for action_id in ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
        ),
        evidence_levels=("E0",) * action_count,
        evidence_manifest_sha256=tuple(
            digest(f"evidence:{index}") for index in range(action_count)
        ),
        evidence_certificate_sha256=((),) * action_count,
        question_bank_sha256=tuple(
            digest(f"questions:{index}") for index in range(action_count)
        ),
        training_config_sha256=tuple(
            digest(f"training:{index}") for index in range(action_count)
        ),
        onnx_model_sha256=(None,) * action_count,
        onnx_metadata_sha256=(None,) * action_count,
        adoption_manifest_sha256=tuple(
            digest(f"adoption:{index}") for index in range(action_count)
        ),
        base_bank_id="base_five_fixture",
        bank_motion_ids=ADMISSION.FRESH_N5_BANK_MOTION_IDS,
        bank_npz_sha256=tuple(bank_shas),
        base_build_manifest_sha256=base_manifest,
        append_build_manifest_sha256=append_manifest,
        base_bank_gate_report_sha256=digest("base-bank"),
        append_bank_gate_report_sha256=digest("append-bank"),
        base_swept_clearance_receipt_sha256=digest("base-swept"),
        append_swept_clearance_receipt_sha256=digest("append-swept"),
        mujoco_fitted_ball_receipt_sha256=(
            result["formal_receipt"]["sha256"]
        ),
        mujoco_fitted_ball_capsule_receipt_sha256=(
            result["retained_capsule_receipt"]["sha256"]
        ),
        isaac_table_filtered_smoke_receipt_sha256=digest("isaac"),
    )


def test_pinned_loader_executes_captured_source_and_never_uses_forged_pyc(
    tmp_path: Path,
):
    module_path = tmp_path / "fixture.py"
    trusted = b"VALUE = 'trusted'\n"
    malicious = b"VALUE = 'malicious'\n"
    module_path.write_bytes(malicious)
    cache_path = Path(
        py_compile.compile(
            str(module_path),
            cfile=str(tmp_path / "forged.pyc"),
            doraise=True,
        )
    )
    assert cache_path.is_file()
    module_path.write_bytes(trusted)
    snapshot = _snapshot(module_path, "fixture.py", trusted)
    loader = BOOTSTRAP.PinnedBytesLoader(snapshot, "a" * 64)
    spec = importlib.util.spec_from_loader(
        "pinned_loader_fixture",
        loader,
        origin=str(module_path.resolve()),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VALUE == "trusted"
    assert module.__cached__ is None
    assert module.__pinned_capsule_id__ == "a" * 64
    assert module.__pinned_executed_sha256__ == hashlib.sha256(
        trusted
    ).hexdigest()


def test_pinned_execution_rejects_any_preloaded_repo_module(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "contact_model", types.ModuleType("x"))
    with pytest.raises(
        BOOTSTRAP.BootstrapError, match="were preloaded"
    ):
        BOOTSTRAP._install_pinned_execution(
            sources={},
            data={},
            capsule_sha256="a" * 64,
            consumed_data=set(),
        )


def test_post_runtime_stability_rejects_inode_or_byte_replacement(
    tmp_path: Path,
):
    source_path = tmp_path / "source.py"
    data_path = tmp_path / "data.txt"
    source_path.write_bytes(b"VALUE = 1\n")
    data_path.write_bytes(b"one\n")
    source = _snapshot(source_path, "source.py", source_path.read_bytes())
    data = _snapshot(data_path, "data.txt", data_path.read_bytes())
    replacement = tmp_path / "replacement.txt"
    replacement.write_bytes(b"one\n")
    os.replace(replacement, data_path)
    with pytest.raises(
        BOOTSTRAP.BootstrapError, match="inode or bytes changed"
    ):
        BOOTSTRAP._post_runtime_stability(
            sources={"source.py": source},
            data={"data.txt": data},
            consumed_data={"data.txt"},
            git=Path("/unused/git"),
            commit="1" * 40,
        )


def test_core_and_bootstrap_freeze_the_same_execution_closure():
    assert CORE.RUNTIME_EXECUTION_SOURCE_PATHS == BOOTSTRAP.SOURCE_PATHS
    assert CORE.RUNTIME_EXECUTION_DATA_PATHS == BOOTSTRAP.DATA_PATHS
    assert BOOTSTRAP.MODULE_BINDINGS[
        BOOTSTRAP.CORE_MODULE_NAME
    ] == BOOTSTRAP.CORE_REPO_PATH
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    assert "\nimport numpy" not in source
    assert "\nimport mujoco" not in source
    assert "external_git_show_stdin" in source
    assert "fresh_detached_worktree" in source
    assert "mujoco_fitted_ball_pre_registered_dual_manifest_launch_v2" in source


def test_direct_core_execution_cannot_run_formal_gate(tmp_path: Path):
    completed = subprocess.run(
        [sys.executable, str(CORE_PATH), "--out", str(tmp_path / "x.json")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "formal execution must use" in completed.stderr
    assert not (tmp_path / "x.json").exists()


def test_bootstrap_requires_isolated_python_before_parsing_formal_inputs():
    completed = subprocess.run(
        [sys.executable, str(BOOTSTRAP_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "requires `python -I -S -B`" in completed.stderr


def test_direct_bootstrap_path_cannot_materialize_or_run_formal_gate():
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(BOOTSTRAP_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "external `git show" in completed.stderr


def test_committed_trust_spec_is_exact_and_security_inputs_are_not_forwarded():
    payload = {
        "schema_version": BOOTSTRAP.TRUST_SPEC_SCHEMA_VERSION,
        "artifact_type": BOOTSTRAP.TRUST_SPEC_ARTIFACT_TYPE,
        "authorization": {
            "formal_simulation_authorized": True,
            "hardware_authorized": False,
            "registered_before_gate_run": True,
            "decision_id": "fixture-decision",
            "human_dri": "Fixture Human",
        },
        "bootstrap": {
            "repo_path": BOOTSTRAP.BOOTSTRAP_REPO_PATH,
            "sha256": "0" * 64,
        },
        "training_manifest": {
            "repo_path": "configs/training_manifest.json",
            "sha256": "1" * 64,
        },
        "physical_gate_manifest": {
            "repo_path": "configs/physical_gate_manifest.json",
            "sha256": "9" * 64,
        },
        "physical_gate_materialization_receipt": {
            "repo_path": "configs/physical_gate_materialization.json",
            "sha256": "a" * 64,
        },
        "profile_pins": {
            "repo_path": "configs/profile.json",
            "sha256": "2" * 64,
        },
        "launch_evidence_trust_root": {
            "repo_path": "configs/launch-root.json",
            "sha256": "3" * 64,
        },
        "runtime_environment": {
            "python_executable_sha256": "4" * 64,
            "git_executable_sha256": "5" * 64,
            "python_version": "fixture-python",
            "python_cache_tag": "fixture-cache",
            "python_import_roots": [
                {
                    "path": "/exact/site-packages",
                    "tree_sha256": "6" * 64,
                }
            ],
            "required_distributions": {
                "mujoco": {
                    "import_name": "mujoco",
                    "version": "fixture-mujoco",
                    "import_root": "/exact/site-packages",
                    "package_subpath": "mujoco",
                    "tree_sha256": "7" * 64,
                },
                "numpy": {
                    "import_name": "numpy",
                    "version": "fixture-numpy",
                    "import_root": "/exact/site-packages",
                    "package_subpath": "numpy",
                    "tree_sha256": "8" * 64,
                },
            },
        },
    }
    payload["receipt_payload_sha256"] = hashlib.sha256(
        BOOTSTRAP._canonical_json_bytes(payload)
    ).hexdigest()
    parsed = BOOTSTRAP._parse_trust_spec(
        json.dumps(payload).encode("utf-8")
    )
    assert parsed["training_manifest"]["sha256"] == "1" * 64
    assert parsed["physical_gate_manifest"]["sha256"] == "9" * 64
    assert (
        parsed["physical_gate_materialization_receipt"]["sha256"]
        == "a" * 64
    )
    arguments = [
        "--source-repo",
        "/repo",
        "--code-commit",
        "a" * 40,
        "--capsule-dir",
        "/capsule",
        "--out",
        "/out.json",
        "--physical-gate-manifest",
        "/attacker.json",
    ]
    with pytest.raises(
        BOOTSTRAP.BootstrapError,
        match="security-critical argument",
    ):
        BOOTSTRAP._materializer_forward_args(arguments)


def test_dual_manifest_closure_accepts_only_receipt_bound_gate_overlay(
    tmp_path: Path,
):
    fixture = _dual_manifest_fixture(tmp_path)
    receipt = BOOTSTRAP._validate_dual_manifest_closure(
        training_manifest_path=fixture["training_path"],
        training_manifest_sha256=fixture["training_sha"],
        training_manifest_repo_path="artifacts/strict.json",
        physical_gate_manifest_path=fixture["physical_path"],
        physical_gate_manifest_sha256=fixture["physical_sha"],
        physical_gate_manifest_repo_path="artifacts/physical.json",
        materialization_receipt_path=fixture["receipt_path"],
        materialization_receipt_sha256=fixture["receipt_sha"],
        expected_action_set_profile=BOOTSTRAP.ACTION_SET_PROFILE,
    )
    assert receipt["strict_training_manifest_preserved"] is True
    assert receipt["inline_manifest_gate_only"] is True


@pytest.mark.parametrize("action_count", (1, 5, 73))
def test_schema2_bootstrap_closure_accepts_exact_n1_n5_n73(
    tmp_path: Path, action_count: int
):
    fixture = _schema2_dual_manifest_fixture(tmp_path, action_count)
    receipt = BOOTSTRAP._validate_dual_manifest_closure(
        training_manifest_path=fixture["training_path"],
        training_manifest_sha256=fixture["training_sha"],
        training_manifest_repo_path="artifacts/strict.json",
        physical_gate_manifest_path=fixture["physical_path"],
        physical_gate_manifest_sha256=fixture["physical_sha"],
        physical_gate_manifest_repo_path="artifacts/physical.json",
        materialization_receipt_path=fixture["receipt_path"],
        materialization_receipt_sha256=fixture["receipt_sha"],
        expected_action_set_profile=fixture["profile_id"],
    )
    assert len(receipt["action_identity_matrix"]) == action_count
    assert (
        BOOTSTRAP._materialization_action_set_profile(receipt)
        == fixture["profile_id"]
    )


@pytest.mark.parametrize("action_count", (1, 5, 73))
def test_schema2_retained_identity_binds_ordered_action_matrix_and_ground_gate(
    tmp_path: Path, action_count: int
):
    formal, artifact_tree = _schema2_formal_identity_fixture(
        tmp_path, action_count
    )
    retained = BOOTSTRAP._retained_capsule_identity(
        formal_receipt=formal,
        formal_raw=BOOTSTRAP._canonical_json_bytes(formal),
        artifact_tree=artifact_tree,
    )
    assert retained["schema_version"] == 3
    assert retained["ordered_action_ids"] == formal["action_order"]
    assert (
        retained["action_identity_matrix"]
        == formal["action_identity_matrix"]
    )
    assert (
        retained["action_identity_matrix_sha256"]
        == formal["action_identity_matrix_sha256"]
    )
    assert retained["five_solid_safety"][
        "ground_contact_policy"
    ] == formal["five_solid_safety_scene"]["ground_contact_policy"]
    assert all(
        len(row["ground_contact_safety_contract_sha256"]) == 64
        for row in retained["five_solid_safety"][
            "assembled_xml_by_dt"
        ]
    )


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            "formal_action_family",
            "disconnected from the schema-2 identity matrix",
        ),
        (
            "matrix_seal",
            "action identity matrix/key set/seal drifted",
        ),
        (
            "materialization_kind",
            "not bound to the exact generic schema-2 materialization",
        ),
        (
            "ground_policy",
            "ground-contact policy drifted",
        ),
    ),
)
def test_schema2_retained_identity_rejects_identity_or_ground_drift(
    tmp_path: Path, mutation: str, match: str
):
    formal, artifact_tree = _schema2_formal_identity_fixture(
        tmp_path, 1
    )
    if mutation == "formal_action_family":
        formal["actions"][0]["family"] = "tampered"
    elif mutation == "matrix_seal":
        formal["action_identity_matrix_sha256"] = "f" * 64
    elif mutation == "materialization_kind":
        formal["materialization_receipt_kind"] = (
            BOOTSTRAP.PHYSICAL_GATE_MATERIALIZATION_KIND
        )
    else:
        formal["five_solid_safety_scene"]["ground_contact_policy"][
            "continuous_nonfoot_clearance_guard_m"
        ] = 0.0
    with pytest.raises(BOOTSTRAP.BootstrapError, match=match):
        BOOTSTRAP._retained_capsule_identity(
            formal_receipt=formal,
            formal_raw=BOOTSTRAP._canonical_json_bytes(formal),
            artifact_tree=artifact_tree,
        )


@pytest.mark.parametrize("field", ("family", "profile_center"))
def test_schema2_bootstrap_rejects_review_matrix_tamper(
    tmp_path: Path, field: str
):
    fixture = _schema2_dual_manifest_fixture(tmp_path, 1)
    row = fixture["receipt"]["action_identity_matrix"][0]
    if field == "family":
        row["family"] = "tampered"
    else:
        row["profile_center"]["incoming_speed_center_mps"] += 1.0
    fixture["receipt_sha"] = _write_canonical_json(
        fixture["receipt_path"], fixture["receipt"]
    )
    with pytest.raises(
        BOOTSTRAP.BootstrapError,
        match="family/motion/profile-center matrix drifted",
    ):
        BOOTSTRAP._validate_dual_manifest_closure(
            training_manifest_path=fixture["training_path"],
            training_manifest_sha256=fixture["training_sha"],
            training_manifest_repo_path="artifacts/strict.json",
            physical_gate_manifest_path=fixture["physical_path"],
            physical_gate_manifest_sha256=fixture["physical_sha"],
            physical_gate_manifest_repo_path="artifacts/physical.json",
            materialization_receipt_path=fixture["receipt_path"],
            materialization_receipt_sha256=fixture["receipt_sha"],
            expected_action_set_profile=fixture["profile_id"],
        )


def test_dual_manifest_closure_rejects_swapped_manifest_roles(
    tmp_path: Path,
):
    fixture = _dual_manifest_fixture(tmp_path)
    with pytest.raises(BOOTSTRAP.BootstrapError):
        BOOTSTRAP._validate_dual_manifest_closure(
            training_manifest_path=fixture["physical_path"],
            training_manifest_sha256=fixture["physical_sha"],
            training_manifest_repo_path="artifacts/strict.json",
            physical_gate_manifest_path=fixture["training_path"],
            physical_gate_manifest_sha256=fixture["training_sha"],
            physical_gate_manifest_repo_path="artifacts/physical.json",
            materialization_receipt_path=fixture["receipt_path"],
            materialization_receipt_sha256=fixture["receipt_sha"],
            expected_action_set_profile=BOOTSTRAP.ACTION_SET_PROFILE,
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing_action_overlay", "receipt_path_drift"),
)
def test_dual_manifest_closure_rejects_overlay_or_receipt_drift(
    tmp_path: Path,
    mutation: str,
):
    fixture = _dual_manifest_fixture(tmp_path)
    if mutation == "missing_action_overlay":
        del fixture["physical"]["actions"][0]["physical_task_binding"]
        fixture["physical_sha"] = _write_canonical_json(
            fixture["physical_path"], fixture["physical"]
        )
        fixture["receipt"]["physical_gate_manifest"]["sha256"] = fixture[
            "physical_sha"
        ]
    else:
        fixture["receipt"]["strict_training_manifest"]["path"] = (
            "artifacts/wrong.json"
        )
    fixture["receipt_sha"] = _write_canonical_json(
        fixture["receipt_path"], fixture["receipt"]
    )
    with pytest.raises(BOOTSTRAP.BootstrapError):
        BOOTSTRAP._validate_dual_manifest_closure(
            training_manifest_path=fixture["training_path"],
            training_manifest_sha256=fixture["training_sha"],
            training_manifest_repo_path="artifacts/strict.json",
            physical_gate_manifest_path=fixture["physical_path"],
            physical_gate_manifest_sha256=fixture["physical_sha"],
            physical_gate_manifest_repo_path="artifacts/physical.json",
            materialization_receipt_path=fixture["receipt_path"],
            materialization_receipt_sha256=fixture["receipt_sha"],
            expected_action_set_profile=BOOTSTRAP.ACTION_SET_PROFILE,
        )


def test_retained_identity_rejects_legacy_single_manifest_binding():
    legacy_formal = {
        "runtime_code_identity": {
            "committed_trust_spec": {
                "bindings": {
                    "manifest": {
                        "repo_path": "configs/manifest.json",
                        "sha256": "1" * 64,
                    }
                }
            }
        }
    }
    artifact_tree = {
        "tree_sha256": "2" * 64,
        "file_count": 1,
        "total_size_bytes": 1,
    }
    with pytest.raises(
        BOOTSTRAP.BootstrapError,
        match="training_manifest key set is not exact",
    ):
        BOOTSTRAP._retained_capsule_identity(
            formal_receipt=legacy_formal,
            formal_raw=b"{}",
            artifact_tree=artifact_tree,
        )
    with pytest.raises(
        ADMISSION.MotionAdmissionError,
        match="training_manifest key set is not exact",
    ):
        ADMISSION._retained_capsule_identity(
            formal_receipt=legacy_formal,
            formal_raw=b"{}",
            artifact_tree=artifact_tree,
        )


def test_dependency_import_roots_are_full_tree_pinned_and_directly_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import_root = tmp_path / "site-packages"
    numpy_root = import_root / "numpy"
    mujoco_root = import_root / "mujoco"
    numpy_root.mkdir(parents=True)
    mujoco_root.mkdir()
    (numpy_root / "__init__.py").write_text("__version__ = '2.0.0'\n")
    (mujoco_root / "__init__.py").write_text("__version__ = '3.3.5'\n")
    root_receipt = BOOTSTRAP._hash_regular_tree(import_root, "fixture root")
    numpy_receipt = BOOTSTRAP._hash_regular_tree(
        numpy_root, "fixture numpy"
    )
    mujoco_receipt = BOOTSTRAP._hash_regular_tree(
        mujoco_root, "fixture mujoco"
    )
    trust_spec = {
        "runtime_environment": {
            "python_import_roots": [
                {
                    "path": str(import_root.resolve()),
                    "tree_sha256": root_receipt["tree_sha256"],
                }
            ],
            "required_distributions": {
                "mujoco": {
                    "import_name": "mujoco",
                    "version": "3.3.5",
                    "import_root": str(import_root.resolve()),
                    "package_subpath": "mujoco",
                    "tree_sha256": mujoco_receipt["tree_sha256"],
                },
                "numpy": {
                    "import_name": "numpy",
                    "version": "2.0.0",
                    "import_root": str(import_root.resolve()),
                    "package_subpath": "numpy",
                    "tree_sha256": numpy_receipt["tree_sha256"],
                },
            },
        }
    }
    monkeypatch.setattr(BOOTSTRAP.sys, "path", ["/stdlib"])
    receipt = BOOTSTRAP._validate_external_dependency_roots(
        trust_spec, install=True
    )
    assert BOOTSTRAP.sys.path == [
        "/stdlib",
        str(import_root.resolve()),
    ]
    assert receipt["site_module_executed"] is False
    assert receipt["pth_files_executed"] is False

    (numpy_root / "__init__.py").write_text("__version__ = 'forged'\n")
    with pytest.raises(
        BOOTSTRAP.BootstrapError, match="committed dependency tree"
    ):
        BOOTSTRAP._validate_external_dependency_roots(
            trust_spec, install=False
        )


def test_retained_capsule_receipt_is_reopened_by_motion_admission(
    tmp_path: Path,
):
    source, final_root, formal, retained, row, result = (
        _retained_capsule_fixture(tmp_path)
    )
    reopened = ADMISSION._reopen_retained_fitted_capsule(
        row,
        repo_root=source,
        expected_formal_sha256=result["formal_receipt"]["sha256"],
        expected_retained_sha256=(
            result["retained_capsule_receipt"]["sha256"]
        ),
    )
    assert reopened.root == final_root
    assert reopened.formal_path == formal
    assert reopened.retained_path == retained
    assert reopened.root.name == result["capsule_id"]
    assert len(result["capsule_id"]) == 64
    retained_payload = json.loads(retained.read_text(encoding="utf-8"))
    identity = retained_payload["identity"]
    assert identity["schema_version"] == 2
    assert identity["code_commit"]
    assert identity["strict_training_manifest_repo_path"] == (
        "strict_manifest.json"
    )
    assert identity["strict_training_manifest_sha256"]
    assert identity["physical_gate_manifest_repo_path"] == "manifest.json"
    assert identity["physical_gate_materialization_receipt_repo_path"] == (
        "physical_gate_materialization_receipt.json"
    )
    assert identity["action_set_contract_sha256"]
    assert identity["ordered_action_ids"] == list(
        ADMISSION.FRESH_N5_DOWNSTREAM_MOTION_IDS
    )
    assert identity["ordered_action_uids"] == [
        row["action_uid"]
        for row in json.loads(formal.read_text(encoding="utf-8"))["actions"]
    ]
    assert identity["motion_sha256"][0]["action_id"] == "bh_loop_c"
    assert identity["solver_source_sha256"][0]["role"].startswith(
        "solver_source:"
    )
    assert {row["role"] for row in identity["physics_sha256"]} == {
        "fitted_contact_model",
        "venue_yaml",
    }
    assert identity["geometry_sha256"][0]["role"].startswith(
        "racket_geometry_"
    )
    assert identity["formal_receipt_sha256"] == row["sha256"]
    binding = _fresh_binding_for_capsule(formal, result)
    ADMISSION._validate_fresh_n5_fitted_ball_receipt(
        row,
        binding=binding,
        repo_root=source,
    )


def test_retained_capsule_existing_content_id_is_never_overwritten(
    tmp_path: Path, monkeypatch
):
    fixed_identity = {"fixture": "preexisting-content-addressed-id"}
    capsule_id = hashlib.sha256(
        BOOTSTRAP._canonical_json_bytes(fixed_identity)
    ).hexdigest()
    sentinel = b"must-not-be-overwritten\n"

    def reserve_existing(**values):
        final_root = (
            values["source"]
            / BOOTSTRAP.CAPSULE_STORE_REPO_PATH
            / capsule_id
        )
        final_root.mkdir()
        (final_root / "sentinel.bin").write_bytes(sentinel)
        monkeypatch.setattr(
            BOOTSTRAP,
            "_retained_capsule_identity",
            lambda **_kwargs: fixed_identity,
        )

    with pytest.raises(
        BOOTSTRAP.BootstrapError,
        match="content-addressed capsule already exists",
    ):
        _retained_capsule_fixture(
            tmp_path, pre_finalize=reserve_existing
        )

    final_root = (
        tmp_path
        / "source"
        / BOOTSTRAP.CAPSULE_STORE_REPO_PATH
        / capsule_id
    )
    assert (final_root / "sentinel.bin").read_bytes() == sentinel
    assert sorted(path.name for path in final_root.iterdir()) == [
        "sentinel.bin"
    ]


def test_retained_capsule_admission_rejects_path_escape(
    tmp_path: Path,
):
    source, _root, _formal, _retained, row, result = (
        _retained_capsule_fixture(tmp_path)
    )
    escaped = dict(row)
    escaped["path"] = "../fitted_ball_receipt.json"
    with pytest.raises(
        ADMISSION.MotionAdmissionError,
        match="relative path|traversal|layout",
    ):
        ADMISSION._reopen_retained_fitted_capsule(
            escaped,
            repo_root=source,
            expected_formal_sha256=result["formal_receipt"]["sha256"],
            expected_retained_sha256=(
                result["retained_capsule_receipt"]["sha256"]
            ),
        )


def test_retained_capsule_admission_rejects_replaced_formal_receipt(
    tmp_path: Path,
):
    source, root, formal, _retained, row, result = (
        _retained_capsule_fixture(tmp_path)
    )
    artifacts = root / "artifacts"
    os.chmod(root, 0o755)
    os.chmod(artifacts, 0o755)
    os.chmod(formal, 0o644)
    replacement = artifacts / "replacement.json"
    replacement.write_text("{}\n", encoding="utf-8")
    os.chmod(replacement, 0o444)
    os.replace(replacement, formal)
    os.chmod(artifacts, 0o555)
    os.chmod(root, 0o555)
    with pytest.raises(
        ADMISSION.MotionAdmissionError,
        match="bytes changed|artifact tree|identity",
    ):
        ADMISSION._reopen_retained_fitted_capsule(
            row,
            repo_root=source,
            expected_formal_sha256=result["formal_receipt"]["sha256"],
            expected_retained_sha256=(
                result["retained_capsule_receipt"]["sha256"]
            ),
        )


def test_retained_capsule_admission_rejects_symlinked_formal_receipt(
    tmp_path: Path,
):
    source, root, formal, _retained, row, result = (
        _retained_capsule_fixture(tmp_path)
    )
    artifacts = root / "artifacts"
    os.chmod(root, 0o755)
    os.chmod(artifacts, 0o755)
    target = artifacts / "target.json"
    target.write_bytes(formal.read_bytes())
    os.chmod(target, 0o444)
    formal.unlink()
    formal.symlink_to(target.name)
    os.chmod(artifacts, 0o555)
    os.chmod(root, 0o555)
    with pytest.raises(
        ADMISSION.MotionAdmissionError,
        match="symlink",
    ):
        ADMISSION._reopen_retained_fitted_capsule(
            row,
            repo_root=source,
            expected_formal_sha256=result["formal_receipt"]["sha256"],
            expected_retained_sha256=(
                result["retained_capsule_receipt"]["sha256"]
            ),
        )
