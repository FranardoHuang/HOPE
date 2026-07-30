#!/usr/bin/env python3
"""Code-rooted admission capability for the canonical motion-bank consumer.

Registry JSON, evidence labels, and adoption manifests are provenance records,
not authority.  This module is the only place that turns a bank-promotion
certificate into an opaque runtime capability.  A certificate is trusted only
when the SHA-256 of the exact bytes parsed by this verifier is present in the
code-owned trust set below.  The trust set is not configurable through Hydra.

It intentionally ships empty.  Adding a digest is a reviewed source-code
change.  Until that happens, new canonical-bank launches fail closed.  The
legacy/default ``motion_file`` training path is not gated here: it loads raw
NPZ bytes directly, as it did before the canonical consumer existed.

This is an operational configuration/provenance boundary, not a sandbox
against arbitrary Python already executing in the process.  In-process code is
part of the trusted computing base and can rebind module globals; an adversarial
plugin threat model would require an external signed ledger or launcher.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256: frozenset[str] = frozenset()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CAPSULE_ID = _SHA256
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ACTION_UID_MAX = (1 << 53) - 1
_FITTED_CAPSULE_STORE = PurePosixPath(
    "hope_training/whole_body_tracking/artifacts/"
    "formal_fitted_ball_capsules_v1"
)
_FITTED_CAPSULE_LAYOUT = "formal_fitted_ball_retained_capsule_v1"
_FITTED_CAPSULE_FORMAL_RELPATH = PurePosixPath(
    "artifacts/fitted_ball_receipt.json"
)
_FITTED_CAPSULE_RETAINED_BASENAME = "retained_capsule_receipt.json"
_MINT_TOKEN = object()
_PURPOSES = ("training", "deployment", "hardware")
_SCOPES = ("upper", "full")
_EVIDENCE_LEVELS = ("E0", "E1", "E2", "E3", "E4", "E5")
FRESH_N5_DOWNSTREAM_MOTION_IDS = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
FRESH_N5_FORBIDDEN_MOTION_IDS = frozenset({"fh_loop", "fh_block_syn"})
FRESH_N5_ACTION_FAMILY = {
    "bh_loop_c": "backhand",
    "v12_forehand_block": "forehand",
    "bh_block": "backhand",
    "s0_highpress": "backhand",
    "fh_loop_high": "forehand",
}
_ACTION_BALL_SOLVER_SOURCE_NAMES = frozenset(
    {
        "continuous_questions.py",
        "hope_commands.py",
        "racket_contact_geometry.py",
        "stroke_adapt_torch.py",
        "virtual_ball.py",
    }
)
FRESH_N5_BASE_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)
FRESH_N5_APPEND_MOTION_IDS = (
    "fh_loop_high",
    "v12_forehand_block",
)
FRESH_N5_BANK_MOTION_IDS = (
    *FRESH_N5_BASE_MOTION_IDS,
    *FRESH_N5_APPEND_MOTION_IDS,
)
ACTION_BALL_ISAAC_TASK_ID = "HOPE-PingPong-ActionBall-AgibotA3-v0"
_TIME_LAW_ARTIFACT_SCHEMA_VERSION = 2
_TIME_LAW_ARTIFACT_TYPE = "canonical_time_law_collocation_v2"
_TIME_LAW_MARKER_NAMES = (
    "window_start",
    "source_anchor",
    "window_end",
)
_TIME_LAW_MARKER_CONTRACT_KEYS = frozenset(
    {
        "marker_names",
        "path_s",
        "time_s",
        "source_anchor_within_solved_path",
        "source_anchor_independent_of_protected_window",
        "protected_window_order_valid",
        "no_early_brake_from_path_start_through_window_end",
        "inclusive_tick_nonempty",
    }
)
_CERTIFICATE_KEYS = frozenset(
    {
        "schema_version",
        "certificate_type",
        "purpose",
        "bank_id",
        "scope",
        "registry_sha256",
        "alignment_sha256",
        "motion_ids",
        "npz_sha256",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "build_manifest_sha256",
        "bank_gate_report",
        "evidence_receipts",
        "question_bank_sha256",
        "training_config_sha256",
        "onnx_model_sha256",
        "onnx_metadata_sha256",
        "adoption_manifest_sha256",
    }
)
_FRESH_N5_CERTIFICATE_KEYS = frozenset(
    {
        "schema_version",
        "certificate_type",
        "purpose",
        "bank_id",
        "base_bank_id",
        "scope",
        "registry_sha256",
        "alignment_sha256",
        "motion_ids",
        "npz_sha256",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "build_manifest_sha256",
        "bank_motion_ids",
        "bank_npz_sha256",
        "base_build_manifest_sha256",
        "append_build_manifest_sha256",
        "evidence_receipts",
        "question_bank_sha256",
        "training_config_sha256",
        "onnx_model_sha256",
        "onnx_metadata_sha256",
        "adoption_manifest_sha256",
        "base_bank_gate_report_sha256",
        "append_bank_gate_report_sha256",
        "base_swept_clearance_receipt_sha256",
        "append_swept_clearance_receipt_sha256",
        "mujoco_fitted_ball_receipt_sha256",
        "mujoco_fitted_ball_capsule_receipt_sha256",
        "isaac_table_filtered_smoke_receipt_sha256",
        "bank_gate_reports",
        "continuous_swept_clearance_receipts",
        "mujoco_fitted_ball_receipt",
        "isaac_table_filtered_smoke_receipt",
    }
)
_BANK_GATE_BINDING_KEYS = frozenset({"path", "sha256"})
_GENERIC_BANK_GATE_REPORT_SCHEMA_VERSION = 2
_LEGACY_BANK_GATE_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/"
    "canonical_motion_bank_gate.py"
)
_GENERIC_BANK_GATE_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/"
    "canonical_motion_generic_bank_gate.py"
)
_FITTED_CAPSULE_BINDING_KEYS = frozenset(
    {"path", "sha256", "retained_capsule_receipt"}
)
_BANK_GATE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "verdict",
        "bank_gate_pass",
        "candidate_integrity_pass",
        "grounded_trace_status",
        "publication_class",
        "training_authorized",
        "hardware_authorized",
        "library_id",
        "manifest",
        "bank_dir",
        "bound_inputs",
        "contracts",
        "aggregate",
        "clips",
        "non_claims",
    }
)
_BANK_GATE_REPORT_KEYS_V2 = _BANK_GATE_REPORT_KEYS | frozenset(
    {"selected_registry_binding"}
)
_BANK_GATE_REPORT_KEYS_APPEND = _BANK_GATE_REPORT_KEYS | frozenset(
    {
        "append_only_composition",
        "append_only_base_validation_scope",
        "station_center_shift_xy_m",
    }
)
_SELECTED_REGISTRY_BINDING_KEYS = frozenset(
    {
        "scope",
        "registry_sha256",
        "alignment_sha256",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "motion_ids",
        "npz_sha256",
        "build_manifest_sha256",
    }
)
_BANK_GATE_BOUND_INPUT_KEYS = frozenset(
    {
        "recipe",
        "compiler",
        "geometry_tool",
        "compiler_options_sha256",
        "ready",
        "mjcf",
        "urdf",
        "body_order",
        "plant",
        "verifier_tools",
    }
)
_BANK_GATE_BOUND_INPUT_KEYS_SWEPT = _BANK_GATE_BOUND_INPUT_KEYS | frozenset(
    {"swept_clearance_receipt"}
)
_BANK_GATE_CONTRACT_KEYS = frozenset(
    {
        "matrix",
        "shared_ready",
        "six_endpoint_velocity_classes_exact_zero",
        "contact_opportunity_is_marker_only",
        "acceleration_allowed_through_window_end",
        "nonnegative_scalar_acceleration_through_window_end",
        "adv2c3_role",
        "grounded_inverse_dynamics",
        "grounded_trace_status",
    }
)
_BANK_GATE_CONTRACT_KEYS_SWEPT = _BANK_GATE_CONTRACT_KEYS | frozenset(
    {"swept_clearance"}
)
_BANK_GATE_CONTRACT_KEYS_APPEND_SWEPT = (
    _BANK_GATE_CONTRACT_KEYS_SWEPT | frozenset({"verification_scope"})
)
_BANK_GATE_AGGREGATE_KEYS = frozenset(
    {
        "clip_count",
        "fk_pass_count",
        "velocity_consistency_pass_count",
        "joint_limit_pass_count",
        "geometry_pass_count",
        "non_torque_dynamics_pass_count",
        "complete_dynamics_pass_count",
        "incomplete_fail_closed_count",
        "failed_count",
        "torque_interpretation_valid_count",
        "clips_with_contact_count",
        "contact_frame_count",
        "self_collision_violation_count",
        "foot_floor_penetration_violation_count",
        "nonfoot_floor_penetration_violation_count",
        "other_world_penetration_violation_count",
        "joint_effort_proxy_peak_utilization",
        "actuator_force_proxy_peak_utilization",
        "root_height_min_m",
        "root_height_max_m",
        "root_tilt_peak_rad",
        "root_xy_displacement_peak_m",
        "com_height_min_m",
        "com_height_max_m",
    }
)
_BANK_GATE_AGGREGATE_KEYS_SWEPT = _BANK_GATE_AGGREGATE_KEYS | frozenset(
    {
        "swept_clearance_pass_count",
        "swept_clearance_minimum_certified_lower_bound_m",
    }
)
_BANK_GATE_AGGREGATE_KEYS_FRESH = _BANK_GATE_AGGREGATE_KEYS_SWEPT | frozenset(
    {
        "time_law_artifact_count",
        "grounded_lmr_pass_count",
        "grounded_lmr_incomplete_count",
    }
)
_BANK_GATE_CLIP_KEYS = frozenset(
    {
        "motion_id",
        "scope",
        "filename",
        "sha256",
        "frames",
        "fps",
        "duration_s",
        "schema2_receipts",
        "strict_schema2_and_ready",
        "contact_opportunity",
        "mujoco_fk",
        "plant_specific_dynamics",
    }
)
_BANK_GATE_CLIP_KEYS_FRESH = _BANK_GATE_CLIP_KEYS | frozenset(
    {
        "canonical_time_law",
        "grounded_left_midpoint_right",
    }
)
_EVIDENCE_RECEIPT_KEYS = frozenset(
    {
        "motion_id",
        "evidence_level",
        "evidence_manifest_sha256",
        "certificate_sha256",
    }
)
_DISCRIMINATED_RECEIPT_SET_KEYS = frozenset({"base", "append"})
_DISCRIMINATED_RECEIPT_KEYS = frozenset({"kind", "path", "sha256"})
_SWEPT_CLEARANCE_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "receipt_class",
        "verdict",
        "with_table",
        "independent_verifier",
        "bank_binding",
        "trajectory_contract",
        "scene_contract",
        "method",
        "results",
        "authorization",
        "non_claims",
    }
)
_SWEPT_BANK_BINDING_KEYS = frozenset(
    {
        "manifest_sha256",
        "recipe_sha256",
        "ready_sha256",
        "mjcf_sha256",
        "urdf_sha256",
        "body_order_sha256",
        "station_center_shift_xy_m",
        "output_matrix",
        "outputs",
    }
)
_SWEPT_RESULT_KEYS = frozenset(
    {
        "motion_id",
        "scope",
        "filename",
        "sha256",
        "frames",
        "fps",
        "duration_s",
        "start_frame",
        "end_frame",
        "interval_count",
        "certified_interval_count",
        "unknown_interval_count",
        "unsafe_interval_count",
        "nonfinite_interval_count",
        "all_intervals_conservatively_bounded",
        "contact_window_start_s",
        "contact_window_end_s",
        "coverage_start",
        "contact_opportunity_covered",
        "coverage_end",
        "complete_cycle",
        "with_table",
        "subjects",
        "obstacles",
        "verdict",
        "hard_collision_count",
        "minimum_clearance_certified_lower_bound_m",
    }
)
_SWEPT_RECEIPT_CLASS = "independent_continuous_swept_clearance_v1"
_SWEPT_COVERAGE = "entire_prep_hit_recovery_continuous_time"
_SWEPT_SUBJECTS = (
    "robot_collision_geoms",
    "racket_and_handle_geoms",
)
_SWEPT_OBSTACLES = (
    "table_top",
    "table_edges",
    "table_underside",
    "action_ball_under_table_keepout",
    "net",
    "net_posts",
)
_SWEPT_ACTION_BALL_ASSEMBLY_ROLES = (
    "top",
    "keepout",
    "net",
    "post_left",
    "post_right",
)
_SWEPT_KEEPOUT = "robot_only_keepout_ball_excluded"
_ACTION_SET_CONTRACT_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "expected_n",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "ordered_action_uids",
        "order_uid_digest_sha256",
        "manifest_path",
        "manifest_sha256",
        "experiment_name",
        "actor_obs_contract",
        "actor_obs_width",
        "namespace_identity",
        "contract_sha256",
    }
)
_FRESH_N5_FITTED_BALL_KEYS = frozenset(
    {
        "schema_version",
        "gate",
        "contact_authority",
        "native_ball_contact_enabled",
        "selector_executed",
        "ball_to_task_solver_executed",
        "ball_to_task_solver_executed_by_gate",
        "pre_registered_ball_to_task_solver_receipt_consumed",
        "solver_execution_receipt_authority",
        "analytic_return_scorer_executed",
        "expected_actions",
        "expected_action_order",
        "action_set_contract",
        "preflight",
        "authorization",
        "runtime_code_identity",
        "formal_gate_executed",
        "runtime_environment",
        "runtime_input_snapshot",
        "runtime_code_identity_post_runtime",
        "runtime_code_identity_final",
        "status",
        "verdict",
        "manifest_id",
        "action_order",
        "base_mujoco_portable_identity_sha256",
        "base_mujoco_verification_receipt_sha256",
        "compiler_mesh_assets",
        "scene_contracts",
        "venue",
        "contact_model",
        "actions",
        "receipt_payload_sha256",
    }
)
_FRESH_N5_FITTED_ACTION_KEYS = frozenset(
    {
        "action_id",
        "action_uid",
        "motion_path",
        "motion_sha256",
        "launch",
        "face_geometry",
        "t_hit_s",
        "t_cycle_s",
        "reference_racket_site_speed_mps",
        "dt_results",
        "convergence",
        "physical_task_binding",
        "shared_ready_joint_linf_rad",
        "recovery_joint_linf_rad",
        "video",
        "verdict",
        "failure_reasons",
    }
)
_FRESH_N5_PHYSICAL_TASK_BINDING_KEYS = frozenset(
    {
        "ball_profile_sha256",
        "solver_profile_sha256",
        "physics_profile_sha256",
        "solver_source_sha256",
        "solver_execution_receipt",
        "cases_sha256",
        "case_order",
        "cases",
    }
)
_FRESH_N5_SOLVER_EXECUTION_RECEIPT_BINDING_KEYS = frozenset(
    {"path", "sha256", "receipt_payload_sha256"}
)
_FRESH_N5_PHYSICAL_TASK_CASE_ROLES = (
    "center_positive_seed_0",
    "center_positive_seed_1",
    "support_positive",
    "negative_t_hit_offset",
    "negative_face_sign",
    "negative_ball_state_mismatch",
)
_FRESH_N5_PHYSICAL_TASK_POSITIVE_ROLES = frozenset(
    _FRESH_N5_PHYSICAL_TASK_CASE_ROLES[:3]
)
_FRESH_N5_PHYSICAL_TASK_NEGATIVE_REASON = {
    "negative_t_hit_offset": "teacher_task_contact_time_mismatch",
    "negative_face_sign": "teacher_task_face_sign_mismatch",
    "negative_ball_state_mismatch": "teacher_task_ball_state_mismatch",
}
_FRESH_N5_PHYSICAL_TASK_CASE_KEYS = frozenset(
    {
        "case_id",
        "case_role",
        "sample_seed",
        "expected_physical_verdict",
        "expected_failure_reason",
        "ball_proposal_sha256",
        "task_payload_sha256",
        "solved_task_geometry_sha256",
        "case_binding_sha256",
        "solver_execution_identity",
        "task_timing",
        "task_geometry",
        "dt_results",
        "convergence",
        "control",
        "observed_physical_verdict",
        "control_verdict",
        "failure_reasons",
    }
)
_FRESH_N5_TASK_TIMING_KEYS = frozenset(
    {
        "teacher_rate",
        "scaled_t_hit_s",
        "scaled_t_cycle_s",
        "pre_swing_wait_s",
    }
)
_FRESH_N5_TASK_GEOMETRY_KEYS = frozenset(
    {
        "mount_normal_sign",
        "ball_contact_w_m",
        "racket_site_target_w_m",
        "racket_normal_w",
        "reference_racket_quat_wxyz",
        "reference_racket_angular_velocity_w_radps",
        "racket_command_quat_wxyz",
        "racket_face_center_velocity_w_mps",
        "racket_site_velocity_w_mps",
        "racket_command_angular_velocity_w_radps",
        "geometry_source_sha256",
        "landing_aim_w_xy_m",
    }
)
_FRESH_N5_PHYSICAL_CONTROL_KEYS = frozenset(
    {
        "expected_physical_verdict",
        "expected_failure_reason",
        "observed_physical_verdict",
        "observed_failure_reason",
        "observed_dt_verdicts",
        "fault_application",
        "convergence_required",
        "convergence_pass",
        "control_verdict",
        "failure_reasons",
    }
)
_FRESH_N5_CONVERGENCE_METRICS = (
    "contact_time_s",
    "contact_position_m",
    "outgoing_velocity_mps",
    "net_height_m",
    "landing_xy_m",
    "landing_time_s",
)
_FRESH_N5_FITTED_VIDEO_KEYS = frozenset(
    {
        "status",
        "path",
        "capsule_relative_path",
        "sha256",
        "size_bytes",
        "frames",
        "fps",
        "camera",
        "evidence_role",
    }
)
_RETAINED_FITTED_CAPSULE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "layout",
        "capsule_id",
        "code_commit",
        "paths",
        "identity",
        "artifact_tree",
        "filesystem",
        "checkout",
        "gate_return_code",
        "authorization",
        "receipt_payload_sha256",
    }
)
_RETAINED_FITTED_CAPSULE_PATH_KEYS = frozenset(
    {"checkout", "artifacts", "formal_receipt"}
)
_RETAINED_FITTED_CAPSULE_TREE_KEYS = frozenset(
    {"tree_sha256", "file_count", "total_size_bytes", "symlink_free"}
)
_RETAINED_FITTED_CAPSULE_FILESYSTEM_KEYS = frozenset(
    {"capsule_root", "checkout", "artifacts"}
)
_RETAINED_FITTED_CAPSULE_STAT_KEYS = frozenset(
    {"device", "inode", "mode"}
)
_RETAINED_FITTED_CAPSULE_CHECKOUT_KEYS = frozenset(
    {"commit", "clean", "detached", "read_only"}
)
_RETAINED_FITTED_CAPSULE_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "code_commit",
        "strict_training_manifest_repo_path",
        "strict_training_manifest_sha256",
        "physical_gate_manifest_repo_path",
        "physical_gate_manifest_sha256",
        "physical_gate_materialization_receipt_repo_path",
        "physical_gate_materialization_receipt_sha256",
        "action_set_contract_sha256",
        "ordered_action_ids",
        "ordered_action_uids",
        "profile_pins_sha256",
        "launch_evidence_trust_root_sha256",
        "motion_sha256",
        "solver_source_sha256",
        "physics_sha256",
        "geometry_sha256",
        "formal_receipt_sha256",
        "formal_receipt_payload_sha256",
        "artifact_tree_sha256",
        "artifact_file_count",
        "artifact_total_size_bytes",
    }
)
_ISAAC_TABLE_SMOKE_KEYS = frozenset(
    {
        "schema_version",
        "receipt_class",
        "verdict",
        "task_id",
        "with_table",
        "scope",
        "mobility_mode",
        "manifest",
        "profile_contract",
        "ordered_action_ids",
        "action_set_contract",
        "motion_sha256",
        "runtime_contract",
        "actions",
        "authorization",
        "non_claims",
        "receipt_payload_sha256",
    }
)
_ISAAC_TABLE_SMOKE_PROFILE_KEYS = frozenset(
    {
        "profile_pins",
        "solver_profile_sha256",
        "physics_profile_sha256",
        "solver_implementation_sources",
        "racket_geometry_contract",
    }
)
_ISAAC_TABLE_SMOKE_SOLVER_SOURCE_KEYS = frozenset(
    {"name", "path", "sha256"}
)
_RACKET_GEOMETRY_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "semantics",
        "ball_target_point",
        "site_target_mapping",
        "face_velocity_mapping",
        "source_path",
        "source_sha256",
        "geometry_source_sha256",
    }
)
_ISAAC_TABLE_SMOKE_ACTION_KEYS = frozenset(
    {
        "motion_id",
        "action_uid",
        "scope",
        "body_pair_filter_count",
        "motion_sha256",
        "complete_cycle",
        "isaac_filtered_contact_pass",
        "table_contact_count",
        "fall_count",
        "hard_limit_count",
        "unsafe_count",
        "verdict",
    }
)
_ISAAC_TABLE_SMOKE_RUNTIME_KEYS = frozenset(
    {
        "source_commit_sha",
        "isaac_version",
        "python_executable",
        "runtime_source",
        "gpu_identity",
        "physics_steps",
        "real_physx_contacts",
        "full_action_ball_assembly",
        "all_32_body_pair_filters",
        "action_body_pair_filter_rows",
        "all_five_obstacles",
        "all_four_substeps",
        "positive_control_pass",
        "negative_control_pass",
        "zero_reset_leakage",
    }
)
_ISAAC_TABLE_SMOKE_GPU_KEYS = frozenset(
    {
        "physical_index",
        "logical_index",
        "cuda_visible_devices",
        "gpu_uuid",
        "gpu_name",
        "driver_version",
        "nvml_verified",
    }
)


class MotionAdmissionError(ValueError):
    """No trusted capability can be minted for the requested motion bytes."""


def _validate_canonical_time_law_identity(
    value: Any, label: str
) -> Mapping[str, Any]:
    """Reject legacy/mixed summaries and ambiguous marker-window semantics."""

    if not isinstance(value, Mapping):
        raise MotionAdmissionError(f"{label} must be one mapping")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version")
        != _TIME_LAW_ARTIFACT_SCHEMA_VERSION
        or value.get("artifact_type") != _TIME_LAW_ARTIFACT_TYPE
    ):
        raise MotionAdmissionError(
            f"{label} is not exact schema-v2 "
            f"{_TIME_LAW_ARTIFACT_TYPE!r}"
        )
    marker_contract = _exact_keys(
        value.get("marker_contract"),
        _TIME_LAW_MARKER_CONTRACT_KEYS,
        f"{label}.marker_contract",
    )
    paths = _exact_keys(
        marker_contract["path_s"],
        frozenset(_TIME_LAW_MARKER_NAMES),
        f"{label}.marker_contract.path_s",
    )
    times = _exact_keys(
        marker_contract["time_s"],
        frozenset(_TIME_LAW_MARKER_NAMES),
        f"{label}.marker_contract.time_s",
    )
    checked_paths = {
        name: _finite(
            paths[name],
            f"{label}.marker_contract.path_s.{name}",
            minimum=0.0,
        )
        for name in _TIME_LAW_MARKER_NAMES
    }
    checked_times = {
        name: _finite(
            times[name],
            f"{label}.marker_contract.time_s.{name}",
            minimum=0.0,
        )
        for name in _TIME_LAW_MARKER_NAMES
    }
    if (
        marker_contract["marker_names"]
        != list(_TIME_LAW_MARKER_NAMES)
        or marker_contract["source_anchor_within_solved_path"] is not True
        or marker_contract[
            "source_anchor_independent_of_protected_window"
        ]
        is not True
        or marker_contract["protected_window_order_valid"] is not True
        or marker_contract[
            "no_early_brake_from_path_start_through_window_end"
        ]
        is not True
        or marker_contract["inclusive_tick_nonempty"] is not True
        or checked_paths["window_start"] > checked_paths["window_end"]
        or checked_times["window_start"] > checked_times["window_end"]
    ):
        raise MotionAdmissionError(
            f"{label} marker/window contract is incomplete or contradictory"
        )
    return value


@dataclass(frozen=True)
class BankPromotionBinding:
    """Exact registry-derived values a trusted certificate must authorize."""

    purpose: str
    bank_id: str
    scope: str
    registry_sha256: str
    alignment_sha256: str
    motion_ids: tuple[str, ...]
    npz_sha256: tuple[str, ...]
    canonical_ready_sha256: str
    canonical_ready_fk_sha256: str
    build_manifest_sha256: tuple[str, ...]
    evidence_levels: tuple[str, ...]
    evidence_manifest_sha256: tuple[str, ...]
    evidence_certificate_sha256: tuple[tuple[str, ...], ...]
    question_bank_sha256: tuple[str | None, ...]
    training_config_sha256: tuple[str | None, ...]
    onnx_model_sha256: tuple[str | None, ...]
    onnx_metadata_sha256: tuple[str | None, ...]
    adoption_manifest_sha256: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if self.purpose not in _PURPOSES:
            raise MotionAdmissionError(
                f"purpose must be one of {_PURPOSES}, got {self.purpose!r}"
            )
        count = len(self.motion_ids)
        if count != 5 or len(set(self.motion_ids)) != count:
            raise MotionAdmissionError(
                "bank promotion binding requires five unique ordered motion ids"
            )
        columns = (
            self.npz_sha256,
            self.build_manifest_sha256,
            self.evidence_levels,
            self.evidence_manifest_sha256,
            self.evidence_certificate_sha256,
            self.question_bank_sha256,
            self.training_config_sha256,
            self.onnx_model_sha256,
            self.onnx_metadata_sha256,
            self.adoption_manifest_sha256,
        )
        if any(len(column) != count for column in columns):
            raise MotionAdmissionError(
                "bank promotion binding columns must all have length five"
            )
        for label, digest in (
            ("registry_sha256", self.registry_sha256),
            ("alignment_sha256", self.alignment_sha256),
            ("canonical_ready_sha256", self.canonical_ready_sha256),
            ("canonical_ready_fk_sha256", self.canonical_ready_fk_sha256),
        ):
            _digest(digest, label)
        for label, values in (
            ("npz_sha256", self.npz_sha256),
            ("build_manifest_sha256", self.build_manifest_sha256),
            ("evidence_manifest_sha256", self.evidence_manifest_sha256),
        ):
            for index, digest in enumerate(values):
                _digest(digest, f"{label}[{index}]")
        for row, values in enumerate(self.evidence_certificate_sha256):
            for column, digest in enumerate(values):
                _digest(
                    digest,
                    f"evidence_certificate_sha256[{row}][{column}]",
                )
        for label, values in (
            ("question_bank_sha256", self.question_bank_sha256),
            ("training_config_sha256", self.training_config_sha256),
            ("onnx_model_sha256", self.onnx_model_sha256),
            ("onnx_metadata_sha256", self.onnx_metadata_sha256),
            ("adoption_manifest_sha256", self.adoption_manifest_sha256),
        ):
            for index, digest in enumerate(values):
                if digest is not None:
                    _digest(digest, f"{label}[{index}]")


@dataclass(frozen=True)
class GenericBankPromotionBinding:
    """Exact arbitrary-N registry values authorized by a v2 certificate."""

    purpose: str
    bank_id: str
    scope: str
    registry_sha256: str
    alignment_sha256: str
    motion_ids: tuple[str, ...]
    npz_sha256: tuple[str, ...]
    canonical_ready_sha256: str
    canonical_ready_fk_sha256: str
    build_manifest_sha256: tuple[str, ...]
    evidence_levels: tuple[str, ...]
    evidence_manifest_sha256: tuple[str, ...]
    evidence_certificate_sha256: tuple[tuple[str, ...], ...]
    question_bank_sha256: tuple[str | None, ...]
    training_config_sha256: tuple[str | None, ...]
    onnx_model_sha256: tuple[str | None, ...]
    onnx_metadata_sha256: tuple[str | None, ...]
    adoption_manifest_sha256: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if self.purpose not in _PURPOSES:
            raise MotionAdmissionError(
                f"purpose must be one of {_PURPOSES}, got {self.purpose!r}"
            )
        if (
            not isinstance(self.bank_id, str)
            or _SLUG.fullmatch(self.bank_id) is None
        ):
            raise MotionAdmissionError(
                "bank_id must be one lowercase normalized slug"
            )
        if self.scope not in _SCOPES:
            raise MotionAdmissionError(
                f"scope must select exactly one of {_SCOPES}, got {self.scope!r}"
            )
        if type(self.motion_ids) is not tuple:
            raise MotionAdmissionError(
                "generic bank promotion motion_ids must be an exact tuple"
            )
        count = len(self.motion_ids)
        if count < 1:
            raise MotionAdmissionError(
                "generic bank promotion binding requires a non-empty ordered "
                "motion id list"
            )
        for index, motion_id in enumerate(self.motion_ids):
            if (
                not isinstance(motion_id, str)
                or _SLUG.fullmatch(motion_id) is None
            ):
                raise MotionAdmissionError(
                    f"motion_ids[{index}] must be one lowercase normalized slug"
                )
        if len(set(self.motion_ids)) != count:
            raise MotionAdmissionError(
                "generic bank promotion binding requires unique ordered motion ids"
            )
        columns = (
            self.npz_sha256,
            self.build_manifest_sha256,
            self.evidence_levels,
            self.evidence_manifest_sha256,
            self.evidence_certificate_sha256,
            self.question_bank_sha256,
            self.training_config_sha256,
            self.onnx_model_sha256,
            self.onnx_metadata_sha256,
            self.adoption_manifest_sha256,
        )
        if any(type(column) is not tuple for column in columns):
            raise MotionAdmissionError(
                "generic bank promotion binding columns must be exact tuples"
            )
        if any(len(column) != count for column in columns):
            raise MotionAdmissionError(
                "generic bank promotion binding columns must all match "
                "motion_ids length"
            )
        for label, digest in (
            ("registry_sha256", self.registry_sha256),
            ("alignment_sha256", self.alignment_sha256),
            ("canonical_ready_sha256", self.canonical_ready_sha256),
            ("canonical_ready_fk_sha256", self.canonical_ready_fk_sha256),
        ):
            _digest(digest, label)
        for label, values in (
            ("npz_sha256", self.npz_sha256),
            ("build_manifest_sha256", self.build_manifest_sha256),
            ("evidence_manifest_sha256", self.evidence_manifest_sha256),
        ):
            for index, digest in enumerate(values):
                _digest(digest, f"{label}[{index}]")
        for index, level in enumerate(self.evidence_levels):
            if level not in _EVIDENCE_LEVELS:
                raise MotionAdmissionError(
                    f"evidence_levels[{index}] must be one of "
                    f"{_EVIDENCE_LEVELS}"
                )
        for row, values in enumerate(self.evidence_certificate_sha256):
            if type(values) is not tuple:
                raise MotionAdmissionError(
                    "generic bank promotion evidence certificate rows must "
                    "be exact tuples"
                )
            expected_receipts = _EVIDENCE_LEVELS.index(
                self.evidence_levels[row]
            )
            if len(values) != expected_receipts:
                raise MotionAdmissionError(
                    "generic bank promotion evidence certificate row "
                    f"{row} must contain exactly {expected_receipts} receipts "
                    f"for {self.evidence_levels[row]}"
                )
            for column, digest in enumerate(values):
                _digest(
                    digest,
                    f"evidence_certificate_sha256[{row}][{column}]",
                )
        for label, values in (
            ("question_bank_sha256", self.question_bank_sha256),
            ("training_config_sha256", self.training_config_sha256),
            ("onnx_model_sha256", self.onnx_model_sha256),
            ("onnx_metadata_sha256", self.onnx_metadata_sha256),
            ("adoption_manifest_sha256", self.adoption_manifest_sha256),
        ):
            for index, digest in enumerate(values):
                if digest is not None:
                    _digest(digest, f"{label}[{index}]")


@dataclass(frozen=True)
class FreshN5BankPromotionBinding(GenericBankPromotionBinding):
    """Exact upper-only fresh-N5 projection over a base-five plus append bank.

    The compiler/bank identity remains the immutable canonical five followed
    by ``fh_loop_high`` and ``v12_forehand_block``.  Runtime admission is a
    separate projection with a fixed order.  Keeping both identities in the
    same immutable value prevents the two retired rows from acquiring runtime
    authority merely because they remain in the compiler provenance prefix.
    """

    base_bank_id: str
    bank_motion_ids: tuple[str, ...]
    bank_npz_sha256: tuple[str, ...]
    base_build_manifest_sha256: str
    append_build_manifest_sha256: str
    base_bank_gate_report_sha256: str
    append_bank_gate_report_sha256: str
    base_swept_clearance_receipt_sha256: str
    append_swept_clearance_receipt_sha256: str
    mujoco_fitted_ball_receipt_sha256: str
    mujoco_fitted_ball_capsule_receipt_sha256: str
    isaac_table_filtered_smoke_receipt_sha256: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.purpose != "training":
            raise MotionAdmissionError(
                "fresh N5 promotion is a training-only admission profile"
            )
        if self.scope != "upper":
            raise MotionAdmissionError(
                "fresh N5 promotion requires the single upper runtime scope"
            )
        if self.motion_ids != FRESH_N5_DOWNSTREAM_MOTION_IDS:
            raise MotionAdmissionError(
                "fresh N5 downstream order must be exactly "
                f"{FRESH_N5_DOWNSTREAM_MOTION_IDS}"
            )
        if set(self.motion_ids).intersection(FRESH_N5_FORBIDDEN_MOTION_IDS):
            raise MotionAdmissionError(
                "fresh N5 downstream view contains a retired legacy motion"
            )
        if self.bank_motion_ids != FRESH_N5_BANK_MOTION_IDS:
            raise MotionAdmissionError(
                "fresh N5 bank identity must retain the exact canonical-five "
                "prefix followed by fh_loop_high and v12_forehand_block"
            )
        if (
            not isinstance(self.base_bank_id, str)
            or _SLUG.fullmatch(self.base_bank_id) is None
            or self.base_bank_id == self.bank_id
        ):
            raise MotionAdmissionError(
                "fresh N5 base_bank_id must be a distinct normalized slug"
            )
        expected_bank_count = 2 * len(FRESH_N5_BANK_MOTION_IDS)
        if (
            type(self.bank_npz_sha256) is not tuple
            or len(self.bank_npz_sha256) != expected_bank_count
        ):
            raise MotionAdmissionError(
                "fresh N5 bank_npz_sha256 must close the exact ordered 7x2 "
                "upper/full matrix"
            )
        for index, digest in enumerate(self.bank_npz_sha256):
            _digest(digest, f"bank_npz_sha256[{index}]")
        if len(set(self.bank_npz_sha256)) != expected_bank_count:
            raise MotionAdmissionError(
                "fresh N5 bank outputs must be fourteen distinct exact byte sets"
            )
        matrix = {
            (motion_id, scope): self.bank_npz_sha256[index]
            for index, (motion_id, scope) in enumerate(
                (pair for motion_id in self.bank_motion_ids for pair in (
                    (motion_id, "upper"),
                    (motion_id, "full"),
                ))
            )
        }
        projected = tuple(
            matrix[(motion_id, "upper")]
            for motion_id in FRESH_N5_DOWNSTREAM_MOTION_IDS
        )
        if projected != self.npz_sha256:
            raise MotionAdmissionError(
                "fresh N5 selected upper hashes do not equal the fixed "
                "downstream projection of the exact 7x2 bank"
            )
        if len(set(self.npz_sha256)) != len(self.npz_sha256):
            raise MotionAdmissionError(
                "fresh N5 downstream actions must bind five distinct motion bytes"
            )
        for label, digest in (
            (
                "base_build_manifest_sha256",
                self.base_build_manifest_sha256,
            ),
            (
                "append_build_manifest_sha256",
                self.append_build_manifest_sha256,
            ),
            (
                "base_bank_gate_report_sha256",
                self.base_bank_gate_report_sha256,
            ),
            (
                "append_bank_gate_report_sha256",
                self.append_bank_gate_report_sha256,
            ),
            (
                "base_swept_clearance_receipt_sha256",
                self.base_swept_clearance_receipt_sha256,
            ),
            (
                "append_swept_clearance_receipt_sha256",
                self.append_swept_clearance_receipt_sha256,
            ),
            (
                "mujoco_fitted_ball_receipt_sha256",
                self.mujoco_fitted_ball_receipt_sha256,
            ),
            (
                "mujoco_fitted_ball_capsule_receipt_sha256",
                self.mujoco_fitted_ball_capsule_receipt_sha256,
            ),
            (
                "isaac_table_filtered_smoke_receipt_sha256",
                self.isaac_table_filtered_smoke_receipt_sha256,
            ),
        ):
            _digest(digest, label)
        expected_manifests = tuple(
            (
                self.append_build_manifest_sha256
                if motion_id in FRESH_N5_APPEND_MOTION_IDS
                else self.base_build_manifest_sha256
            )
            for motion_id in self.motion_ids
        )
        if self.build_manifest_sha256 != expected_manifests:
            raise MotionAdmissionError(
                "fresh N5 per-action build manifests do not discriminate "
                "base outputs from appended outputs"
            )
        for label, values in (
            ("question_bank_sha256", self.question_bank_sha256),
            ("training_config_sha256", self.training_config_sha256),
            ("adoption_manifest_sha256", self.adoption_manifest_sha256),
        ):
            if any(value is None for value in values):
                raise MotionAdmissionError(
                    f"fresh N5 training promotion requires all five {label} pins"
                )


@dataclass(frozen=True)
class _BankReportView:
    """Minimal selected-scope identity consumed by the bank-report verifier."""

    bank_id: str
    scope: str
    motion_ids: tuple[str, ...]
    npz_sha256: tuple[str, ...]
    canonical_ready_sha256: str
    registry_sha256: str = ""
    alignment_sha256: str = ""
    canonical_ready_fk_sha256: str = ""
    build_manifest_sha256: tuple[str, ...] = ()


class TrustedMotionAdmission:
    """Config-opaque token minted after code-rooted certificate verification."""

    __slots__ = (
        "_certificate_sha256",
        "_binding_sha256",
        "_purpose",
        "_bank_id",
        "_scope",
        "_certificate_path",
        "_repo_root",
        "_sealed",
    )

    def __init__(
        self,
        *,
        _token: object | None = None,
        certificate_sha256: str = "",
        binding_sha256: str = "",
        purpose: str = "",
        bank_id: str = "",
        scope: str = "",
        certificate_path: str = "",
        repo_root: str = "",
    ) -> None:
        if _token is not _MINT_TOKEN:
            raise TypeError(
                "TrustedMotionAdmission is opaque; use "
                "verify_bank_promotion_certificate"
            )
        object.__setattr__(self, "_certificate_sha256", certificate_sha256)
        object.__setattr__(self, "_binding_sha256", binding_sha256)
        object.__setattr__(self, "_purpose", purpose)
        object.__setattr__(self, "_bank_id", bank_id)
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_certificate_path", certificate_path)
        object.__setattr__(self, "_repo_root", repo_root)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TrustedMotionAdmission is immutable")
        object.__setattr__(self, name, value)

    @property
    def certificate_sha256(self) -> str:
        return self._certificate_sha256

    @property
    def purpose(self) -> str:
        return self._purpose


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MotionAdmissionError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise MotionAdmissionError(f"JSON contains forbidden constant {value}")


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MotionAdmissionError(
            f"{label} is not strict UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise MotionAdmissionError(f"{label} must contain one JSON object")
    return value


def _exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MotionAdmissionError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise MotionAdmissionError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MotionAdmissionError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MotionAdmissionError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise MotionAdmissionError(f"{label} must be one finite number")
    result = float(value)
    if not math.isfinite(result) or (
        minimum is not None and result < minimum
    ):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise MotionAdmissionError(f"{label} must be finite{suffix}")
    return result


def _assert_no_runtime_self_authorization(value: Any, label: str) -> None:
    """Evidence may prove a gate but may never grant a runtime purpose."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in (
                "training_authorized",
                "deployment_authorized",
                "hardware_authorized",
            ) and item is not False:
                raise MotionAdmissionError(
                    f"{label}.{key} is forbidden evidence self-authorization"
                )
            if key == "authorization" and isinstance(item, Mapping):
                for purpose in _PURPOSES:
                    if purpose in item and item[purpose] is not False:
                        raise MotionAdmissionError(
                            f"{label}.authorization.{purpose} is forbidden "
                            "evidence self-authorization"
                        )
            _assert_no_runtime_self_authorization(item, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_no_runtime_self_authorization(
                item, f"{label}[{index}]"
            )


def _snapshot(path: Path, label: str) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MotionAdmissionError(f"cannot read {label} {path}: {exc}") from exc
    return payload, hashlib.sha256(payload).hexdigest()


def _repo_file(value: Any, repo_root: Path, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise MotionAdmissionError(
            f"{label} must be one normalized repository-relative path"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise MotionAdmissionError(f"{label} may not contain '.' or '..'")
    try:
        root = repo_root.resolve(strict=True)
        path = root.joinpath(*pure.parts).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise MotionAdmissionError(
            f"{label} does not resolve inside repository root"
        ) from exc
    if not path.is_file():
        raise MotionAdmissionError(f"{label} is not a regular file")
    return path


def _repo_directory(value: Any, repo_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MotionAdmissionError(f"{label} must be one non-empty path")
    candidate = Path(value).expanduser()
    try:
        root = repo_root.resolve(strict=True)
        if candidate.is_absolute():
            path = candidate.resolve(strict=True)
        else:
            if "\\" in value or "//" in value:
                raise ValueError("path is not normalized")
            pure = PurePosixPath(value)
            if any(part in ("", ".", "..") for part in pure.parts):
                raise ValueError("path contains dot traversal")
            path = root.joinpath(*pure.parts).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise MotionAdmissionError(
            f"{label} does not resolve inside repository root"
        ) from exc
    if not path.is_dir():
        raise MotionAdmissionError(f"{label} is not a directory")
    return path


def _receipt_file(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
    expected_repo_path: str | None = None,
) -> Path:
    """Reopen one content-addressed receipt inside the repository boundary."""

    path_value = receipt["path"]
    if not isinstance(path_value, str) or not path_value:
        raise MotionAdmissionError(f"{label}.path must be non-empty")
    candidate = Path(path_value).expanduser()
    try:
        root = repo_root.resolve(strict=True)
        if candidate.is_absolute():
            path = candidate.resolve(strict=True)
            path.relative_to(root)
        else:
            path = _repo_file(path_value, root, f"{label}.path")
    except (OSError, ValueError) as exc:
        raise MotionAdmissionError(
            f"{label}.path does not resolve inside repository root"
        ) from exc
    if not path.is_file():
        raise MotionAdmissionError(f"{label}.path is not a regular file")
    if expected_repo_path is not None:
        expected = root.joinpath(*PurePosixPath(expected_repo_path).parts).resolve(
            strict=True
        )
        if path != expected:
            raise MotionAdmissionError(
                f"{label}.path is not the repository verifier {expected_repo_path}"
            )
    expected_sha = _digest(receipt["sha256"], f"{label}.sha256")
    _, actual_sha = _snapshot(path, label)
    if actual_sha != expected_sha:
        raise MotionAdmissionError(f"{label} bytes differ from its receipt")
    return path


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _derive_action_uid(
    action_id: str, family: str, motion_sha256: str
) -> int:
    """Recompute the planner-compatible, float64-exact action identity."""

    if (
        not isinstance(action_id, str)
        or _SLUG.fullmatch(action_id) is None
        or family not in ("forehand", "backhand")
    ):
        raise MotionAdmissionError(
            "action UID inputs must be one normalized action and family"
        )
    motion_sha = _digest(motion_sha256, "action UID motion_sha256")
    identity = {
        "action_id": action_id,
        "content_sha256": motion_sha,
        "family": family,
    }
    digest = hashlib.sha256(_canonical_json_bytes(identity)).digest()
    return 1 + (
        int.from_bytes(digest, byteorder="big") % _ACTION_UID_MAX
    )


@dataclass(frozen=True)
class _FreshN5EvidenceIdentity:
    """Identity shared by the fitted-MuJoCo and filtered-Isaac receipts."""

    manifest_sha256: str
    manifest_id: str
    action_set_contract_sha256: str
    action_ids: tuple[str, ...]
    action_uids: tuple[int, ...]
    profile_pins_sha256: str
    solver_profile_sha256: str
    physics_profile_sha256: str
    geometry_source_sha256: str
    code_commit: str


def _validate_fresh_n5_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
    binding: Any,
    repo_root: Path,
    label: str,
    reopen_motion_files: bool,
    require_gate_geometry: bool = True,
) -> tuple[
    str,
    tuple[int, ...],
    tuple[tuple[float, float, float], ...],
]:
    """Validate the exact fresh-N5 order, bytes, family, and wire UIDs."""

    manifest_sha = _digest(manifest_sha256, f"{label} SHA-256")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 3
        or manifest.get("mobility_mode") != "no_move"
        or manifest.get("action_order")
        != list(FRESH_N5_DOWNSTREAM_MOTION_IDS)
    ):
        raise MotionAdmissionError(
            f"{label} is not the exact schema-v3 no-move fresh-N5 manifest"
        )
    _assert_no_runtime_self_authorization(manifest, label)
    manifest_id = manifest.get("manifest_id")
    if (
        not isinstance(manifest_id, str)
        or _SLUG.fullmatch(manifest_id) is None
    ):
        raise MotionAdmissionError(f"{label} manifest_id is invalid")
    prototype = manifest.get("prototype")
    if (
        not isinstance(prototype, Mapping)
        or prototype.get("scope") != "upper"
    ):
        raise MotionAdmissionError(
            f"{label} prototype scope must be exactly upper"
        )
    _digest(
        manifest.get("solver_profile_sha256"),
        f"{label}.solver_profile_sha256",
    )
    _digest(
        manifest.get("physics_profile_sha256"),
        f"{label}.physics_profile_sha256",
    )
    if require_gate_geometry:
        geometry = _exact_keys(
            manifest.get("racket_geometry_contract"),
            _RACKET_GEOMETRY_CONTRACT_KEYS,
            f"{label}.racket_geometry_contract",
        )
        if (
            geometry["schema_version"] != 2
            or geometry["semantics"] != "exact_face_contact_v2"
            or geometry["ball_target_point"]
            != "physical_ball_center_at_native_contact"
            or geometry["site_target_mapping"]
            != "site_target_from_ball_center"
            or geometry["face_velocity_mapping"]
            != "site_linear_plus_omega_cross_face_center_offset"
        ):
            raise MotionAdmissionError(
                f"{label} does not bind exact physical racket geometry v2"
            )
        _digest(
            geometry["source_sha256"],
            f"{label}.racket_geometry_contract.source_sha256",
        )
        _digest(
            geometry["geometry_source_sha256"],
            f"{label}.racket_geometry_contract.geometry_source_sha256",
        )
    elif "racket_geometry_contract" in manifest:
        raise MotionAdmissionError(
            f"{label} strict training manifest contains gate-only geometry"
        )
    raw_actions = manifest.get("actions")
    if not isinstance(raw_actions, list) or len(raw_actions) != 5:
        raise MotionAdmissionError(
            f"{label} must contain exactly five ordered actions"
        )
    action_uids: list[int] = []
    action_kinematics: list[tuple[float, float, float]] = []
    for index, (raw, motion_id, motion_sha) in enumerate(
        zip(raw_actions, binding.motion_ids, binding.npz_sha256)
    ):
        if not isinstance(raw, Mapping):
            raise MotionAdmissionError(
                f"{label} actions[{index}] must be an object"
            )
        family = FRESH_N5_ACTION_FAMILY[motion_id]
        expected_uid = _derive_action_uid(
            motion_id, family, motion_sha
        )
        action_uid = raw.get("action_uid")
        if (
            raw.get("action_id") != motion_id
            or raw.get("family") != family
            or raw.get("motion_sha256") != motion_sha
            or type(action_uid) is not int
            or action_uid != expected_uid
        ):
            raise MotionAdmissionError(
                f"{label} actions[{index}] does not bind the exact "
                "action/family/motion/action_uid identity"
            )
        t_hit = _finite(
            raw.get("reference_t_hit_s"),
            f"{label} actions[{index}].reference_t_hit_s",
            minimum=0.0,
        )
        t_cycle = _finite(
            raw.get("reference_t_cycle_s"),
            f"{label} actions[{index}].reference_t_cycle_s",
            minimum=0.0,
        )
        racket_speed = _finite(
            raw.get("reference_racket_site_speed_mps"),
            (
                f"{label} actions[{index}]"
                ".reference_racket_site_speed_mps"
            ),
            minimum=0.0,
        )
        if t_hit <= 0.0 or t_hit >= t_cycle or racket_speed <= 0.0:
            raise MotionAdmissionError(
                f"{label} actions[{index}] timing/speed is not physical"
            )
        strike_phase = _finite(
            raw.get("strike_phase"),
            f"{label} actions[{index}].strike_phase",
            minimum=0.0,
        )
        reaction_margin = _finite(
            raw.get("reaction_margin_s"),
            f"{label} actions[{index}].reaction_margin_s",
            minimum=0.0,
        )
        teacher_rate_min = _finite(
            raw.get("teacher_rate_min"),
            f"{label} actions[{index}].teacher_rate_min",
            minimum=0.0,
        )
        teacher_rate_max = _finite(
            raw.get("teacher_rate_max"),
            f"{label} actions[{index}].teacher_rate_max",
            minimum=0.0,
        )
        ball_profile = raw.get("ball_profile")
        if not isinstance(ball_profile, Mapping):
            raise MotionAdmissionError(
                f"{label} actions[{index}].ball_profile must be an object"
            )
        time_to_contact = _finite(
            ball_profile.get("time_to_contact_center_s"),
            (
                f"{label} actions[{index}].ball_profile"
                ".time_to_contact_center_s"
            ),
            minimum=0.0,
        )
        if (
            not 0.0 < strike_phase < 1.0
            or abs(strike_phase * t_cycle - t_hit) > 1.0e-5
            or teacher_rate_min <= 0.0
            or not teacher_rate_min <= 1.0 <= teacher_rate_max
            or time_to_contact
            < t_hit / teacher_rate_min + reaction_margin
            or time_to_contact - t_hit / teacher_rate_max
            > 1.0 + 1.0e-12
        ):
            raise MotionAdmissionError(
                f"{label} actions[{index}] teacher-rate/wait contract "
                "is invalid"
            )
        if reopen_motion_files:
            motion_path = _repo_file(
                raw.get("motion_path"),
                repo_root,
                f"{label} actions[{index}].motion_path",
            )
            _, actual_sha = _snapshot(
                motion_path, f"{label} actions[{index}] motion"
            )
            if actual_sha != motion_sha:
                raise MotionAdmissionError(
                    f"{label} actions[{index}] motion bytes changed"
                )
        action_uids.append(action_uid)
        action_kinematics.append((t_hit, t_cycle, racket_speed))
    if len(set(action_uids)) != len(action_uids):
        raise MotionAdmissionError(
            f"{label} contains duplicate action_uid values"
        )
    return manifest_id, tuple(action_uids), tuple(action_kinematics)


def _assert_plain_path_components(path: Path, label: str) -> Path:
    """Resolve an existing path while rejecting every symlink component."""

    lexical = path.expanduser()
    if not lexical.is_absolute():
        raise MotionAdmissionError(f"{label} must be an absolute path")
    current = Path(lexical.parts[0])
    for part in lexical.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise MotionAdmissionError(
                f"cannot lstat {label} component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise MotionAdmissionError(
                f"{label} contains a symlink component: {current}"
            )
    return lexical.resolve(strict=True)


def _normalized_repo_member(
    value: Any, *, repo_root: Path, label: str
) -> tuple[PurePosixPath, Path]:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or "//" in value
    ):
        raise MotionAdmissionError(
            f"{label} must be a normalized repository-relative path"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(
        part in ("", ".", "..") for part in pure.parts
    ):
        raise MotionAdmissionError(f"{label} contains path traversal")
    root_input = repo_root.expanduser()
    if not root_input.is_absolute():
        root_input = Path.cwd() / root_input
    root = _assert_plain_path_components(root_input, "repository root")
    lexical = root.joinpath(*pure.parts)
    resolved = _assert_plain_path_components(lexical, label)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MotionAdmissionError(f"{label} escaped repository root") from exc
    return pure, resolved


def _read_plain_regular(
    path: Path, label: str
) -> tuple[bytes, os.stat_result]:
    resolved = _assert_plain_path_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(resolved), flags)
    except OSError as exc:
        raise MotionAdmissionError(
            f"cannot open {label} {resolved}: {exc}"
        ) from exc
    chunks: list[bytes] = []
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise MotionAdmissionError(
                f"{label} is not a regular file: {resolved}"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    path_stat = os.stat(resolved, follow_symlinks=False)
    raw = b"".join(chunks)
    if (
        descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
        or descriptor_stat.st_size != path_stat.st_size
        or len(raw) != descriptor_stat.st_size
    ):
        raise MotionAdmissionError(
            f"{label} identity changed during descriptor read"
        )
    return raw, descriptor_stat


def _plain_directory_identity(path: Path, label: str) -> dict[str, int]:
    resolved = _assert_plain_path_components(path, label)
    metadata = os.lstat(resolved)
    if not stat.S_ISDIR(metadata.st_mode):
        raise MotionAdmissionError(f"{label} is not a directory")
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(stat.S_IMODE(metadata.st_mode)),
    }


def _hash_plain_artifact_tree(
    path: Path, label: str
) -> dict[str, Any]:
    root = _assert_plain_path_components(path, label)
    initial = os.lstat(root)
    if not stat.S_ISDIR(initial.st_mode):
        raise MotionAdmissionError(f"{label} is not a directory")
    pending = [root]
    rows: list[dict[str, Any]] = []
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda row: row.name)
        except OSError as exc:
            raise MotionAdmissionError(
                f"cannot scan {label} directory {current}: {exc}"
            ) from exc
        for entry in entries:
            member = Path(entry.path)
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise MotionAdmissionError(
                    f"{label} contains symlink {member}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(member)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise MotionAdmissionError(
                    f"{label} contains special file {member}"
                )
            raw, descriptor_stat = _read_plain_regular(
                member, f"{label} member"
            )
            if (
                descriptor_stat.st_dev != metadata.st_dev
                or descriptor_stat.st_ino != metadata.st_ino
            ):
                raise MotionAdmissionError(
                    f"{label} member was replaced during tree hash"
                )
            rows.append(
                {
                    "path": member.relative_to(root).as_posix(),
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    final = os.lstat(root)
    if (
        initial.st_dev != final.st_dev
        or initial.st_ino != final.st_ino
        or initial.st_mtime_ns != final.st_mtime_ns
    ):
        raise MotionAdmissionError(
            f"{label} root changed during tree hash"
        )
    if not rows:
        raise MotionAdmissionError(f"{label} contains no regular files")
    rows.sort(key=lambda row: row["path"])
    return {
        "tree_sha256": hashlib.sha256(
            _canonical_json_bytes(
                {
                    "schema_version": 1,
                    "manifest_type": (
                        "symlink_free_regular_file_tree_v1"
                    ),
                    "files": rows,
                }
            )
        ).hexdigest(),
        "file_count": len(rows),
        "total_size_bytes": sum(
            int(row["size_bytes"]) for row in rows
        ),
        "symlink_free": True,
    }


def _runtime_input_digest_rows(
    formal_receipt: Mapping[str, Any],
    *,
    role_prefixes: Sequence[str],
) -> list[dict[str, str]]:
    snapshot = formal_receipt.get("runtime_input_snapshot")
    files = snapshot.get("files") if isinstance(snapshot, Mapping) else None
    if not isinstance(files, list):
        return []
    output: list[dict[str, str]] = []
    for raw_row in files:
        if not isinstance(raw_row, Mapping):
            continue
        digest = raw_row.get("sha256")
        roles = raw_row.get("roles")
        if (
            not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or not isinstance(roles, list)
        ):
            continue
        for role in sorted(
            role
            for role in roles
            if isinstance(role, str)
            and any(role.startswith(prefix) for prefix in role_prefixes)
        ):
            output.append({"role": role, "sha256": digest})
    output.sort(key=lambda row: (row["role"], row["sha256"]))
    return output


def _retained_action_set_identity(
    *,
    formal_receipt: Mapping[str, Any],
    strict_manifest_repo_path: str,
    strict_manifest_sha256: str,
) -> tuple[str, list[str], list[int]]:
    value = _exact_keys(
        formal_receipt.get("action_set_contract"),
        _ACTION_SET_CONTRACT_IDENTITY_KEYS,
        "formal receipt action_set_contract",
    )
    contract = dict(value)
    declared_sha = _digest(
        contract.pop("contract_sha256"),
        "formal receipt action_set_contract.contract_sha256",
    )
    if hashlib.sha256(_canonical_json_bytes(contract)).hexdigest() != declared_sha:
        raise MotionAdmissionError(
            "formal receipt action_set_contract payload seal is false"
        )
    expected_n = contract.get("expected_n")
    action_ids = contract.get("ordered_action_ids")
    action_uids = contract.get("ordered_action_uids")
    formal_actions = formal_receipt.get("actions")
    formal_action_ids = (
        [row.get("action_id") for row in formal_actions]
        if isinstance(formal_actions, list)
        and all(isinstance(row, Mapping) for row in formal_actions)
        else None
    )
    formal_action_uids = (
        [row.get("action_uid") for row in formal_actions]
        if isinstance(formal_actions, list)
        and all(isinstance(row, Mapping) for row in formal_actions)
        else None
    )
    if (
        contract.get("schema_version") != 1
        or contract.get("kind")
        != "whole_body_tracking.action_ball.action_set_contract"
        or contract.get("profile_id") != "fresh_upper_nomove_n5_v3"
        or type(expected_n) is not int
        or expected_n <= 0
        or not isinstance(action_ids, list)
        or len(action_ids) != expected_n
        or any(not isinstance(item, str) or not item for item in action_ids)
        or len(set(action_ids)) != expected_n
        or not isinstance(action_uids, list)
        or len(action_uids) != expected_n
        or any(type(item) is not int or item <= 0 for item in action_uids)
        or len(set(action_uids)) != expected_n
        or contract.get("manifest_path") != strict_manifest_repo_path
        or contract.get("manifest_sha256") != strict_manifest_sha256
        or formal_receipt.get("expected_actions") != expected_n
        or formal_receipt.get("expected_action_order") != action_ids
        or formal_receipt.get("action_order") != action_ids
        or formal_action_ids != action_ids
        or formal_action_uids != action_uids
    ):
        raise MotionAdmissionError(
            "formal receipt action-set/strict-manifest/action identity "
            "does not close"
        )
    return declared_sha, list(action_ids), list(action_uids)


def _retained_capsule_identity(
    *,
    formal_receipt: Mapping[str, Any],
    formal_raw: bytes,
    artifact_tree: Mapping[str, Any],
) -> dict[str, Any]:
    attestation = formal_receipt.get("runtime_code_identity")
    trust = (
        attestation.get("committed_trust_spec")
        if isinstance(attestation, Mapping)
        else None
    )
    bindings = trust.get("bindings") if isinstance(trust, Mapping) else None

    def binding(name: str) -> tuple[str, str]:
        row = bindings.get(name) if isinstance(bindings, Mapping) else None
        if not isinstance(row, Mapping) or set(row) != {
            "repo_path",
            "sha256",
        }:
            raise MotionAdmissionError(
                f"formal receipt trust binding {name} key set is not exact"
            )
        repo_path = row["repo_path"]
        if (
            not isinstance(repo_path, str)
            or not repo_path
            or repo_path.startswith("/")
            or repo_path.endswith("/")
            or "\\" in repo_path
            or "//" in repo_path
        ):
            raise MotionAdmissionError(
                f"formal receipt trust binding {name}.repo_path must be a "
                "normalized repository-relative path"
            )
        pure = PurePosixPath(repo_path)
        if pure.is_absolute() or any(
            part in ("", ".", "..") for part in pure.parts
        ):
            raise MotionAdmissionError(
                f"formal receipt trust binding {name}.repo_path contains "
                "path traversal"
            )
        return (
            repo_path,
            _digest(
                row["sha256"],
                f"formal receipt trust binding {name}.sha256",
            ),
        )

    training_path, training_sha = binding("training_manifest")
    physical_path, physical_sha = binding("physical_gate_manifest")
    materialization_path, materialization_sha = binding(
        "physical_gate_materialization_receipt"
    )
    (
        action_set_contract_sha,
        ordered_action_ids,
        ordered_action_uids,
    ) = _retained_action_set_identity(
        formal_receipt=formal_receipt,
        strict_manifest_repo_path=training_path,
        strict_manifest_sha256=training_sha,
    )

    motion_rows: list[dict[str, str]] = []
    actions = formal_receipt.get("actions")
    if isinstance(actions, list):
        for row in actions:
            if not isinstance(row, Mapping):
                continue
            action_id = row.get("action_id")
            digest = row.get("motion_sha256")
            if isinstance(action_id, str) and isinstance(digest, str):
                motion_rows.append(
                    {"action_id": action_id, "sha256": digest}
                )
    return {
        "schema_version": 2,
        "code_commit": (
            attestation.get("code_commit", "")
            if isinstance(attestation, Mapping)
            else ""
        ),
        "strict_training_manifest_repo_path": training_path,
        "strict_training_manifest_sha256": training_sha,
        "physical_gate_manifest_repo_path": physical_path,
        "physical_gate_manifest_sha256": physical_sha,
        "physical_gate_materialization_receipt_repo_path": (
            materialization_path
        ),
        "physical_gate_materialization_receipt_sha256": (
            materialization_sha
        ),
        "action_set_contract_sha256": action_set_contract_sha,
        "ordered_action_ids": ordered_action_ids,
        "ordered_action_uids": ordered_action_uids,
        "profile_pins_sha256": binding("profile_pins")[1],
        "launch_evidence_trust_root_sha256": binding(
            "launch_evidence_trust_root"
        )[1],
        "motion_sha256": motion_rows,
        "solver_source_sha256": _runtime_input_digest_rows(
            formal_receipt, role_prefixes=("solver_source:",)
        ),
        "physics_sha256": _runtime_input_digest_rows(
            formal_receipt,
            role_prefixes=("venue_yaml", "fitted_contact_model"),
        ),
        "geometry_sha256": _runtime_input_digest_rows(
            formal_receipt,
            role_prefixes=(
                "racket_geometry_",
                "scene_source:",
                "selected_face_mesh:",
                "mujoco_identity_manifest",
                "vendor_root_mjcf",
            ),
        ),
        "formal_receipt_sha256": hashlib.sha256(formal_raw).hexdigest(),
        "formal_receipt_payload_sha256": formal_receipt.get(
            "receipt_payload_sha256", ""
        ),
        "artifact_tree_sha256": artifact_tree.get("tree_sha256", ""),
        "artifact_file_count": artifact_tree.get("file_count", 0),
        "artifact_total_size_bytes": artifact_tree.get(
            "total_size_bytes", 0
        ),
    }


def _validate_detached_clean_checkout(
    checkout: Path, expected_commit: str
) -> None:
    git_name = shutil.which("git")
    if git_name is None:
        raise MotionAdmissionError(
            "cannot verify retained capsule: git executable is unavailable"
        )
    git = _assert_plain_path_components(
        Path(git_name).resolve(strict=True), "git executable"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )

    def run(arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [str(git), "-C", str(checkout), *arguments],
            check=False,
            capture_output=True,
            env=environment,
        )

    head = run(("rev-parse", "--verify", "HEAD"))
    detached = run(("symbolic-ref", "--quiet", "HEAD"))
    status = run(("status", "--porcelain", "--untracked-files=all"))
    if (
        head.returncode != 0
        or head.stderr
        or head.stdout.decode("ascii", errors="strict").strip()
        != expected_commit
        or detached.returncode != 1
        or detached.stdout
        or detached.stderr
        or status.returncode != 0
        or status.stdout
        or status.stderr
    ):
        raise MotionAdmissionError(
            "retained fitted-ball checkout is not the exact clean detached "
            "commit"
        )


@dataclass(frozen=True)
class _RetainedFittedCapsule:
    root: Path
    checkout: Path
    artifacts: Path
    formal_path: Path
    formal_raw: bytes
    formal_receipt: Mapping[str, Any]
    retained_path: Path
    retained_sha256: str


def _reopen_retained_fitted_capsule(
    binding_row: Any,
    *,
    repo_root: Path,
    expected_formal_sha256: str,
    expected_retained_sha256: str,
) -> _RetainedFittedCapsule:
    """Reopen the one fixed content-addressed fitted-ball capsule layout."""

    row = _exact_keys(
        binding_row,
        _FITTED_CAPSULE_BINDING_KEYS,
        "mujoco_fitted_ball_receipt",
    )
    formal_sha = _digest(
        row["sha256"], "mujoco_fitted_ball_receipt.sha256"
    )
    if formal_sha != expected_formal_sha256:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball formal receipt SHA is not crossbound"
        )
    retained_binding = _exact_keys(
        row["retained_capsule_receipt"],
        _BANK_GATE_BINDING_KEYS,
        "mujoco_fitted_ball_receipt.retained_capsule_receipt",
    )
    retained_sha = _digest(
        retained_binding["sha256"],
        "retained fitted-ball capsule receipt SHA",
    )
    if retained_sha != expected_retained_sha256:
        raise MotionAdmissionError(
            "retained fitted-ball capsule receipt SHA is not crossbound"
        )
    formal_pure, formal_path = _normalized_repo_member(
        row["path"],
        repo_root=repo_root,
        label="MuJoCo fitted-ball receipt path",
    )
    retained_pure, retained_path = _normalized_repo_member(
        retained_binding["path"],
        repo_root=repo_root,
        label="retained fitted-ball capsule receipt path",
    )
    store_parts = _FITTED_CAPSULE_STORE.parts
    expected_formal_tail = _FITTED_CAPSULE_FORMAL_RELPATH.parts
    if (
        len(formal_pure.parts)
        != len(store_parts) + 1 + len(expected_formal_tail)
        or formal_pure.parts[: len(store_parts)] != store_parts
        or formal_pure.parts[-len(expected_formal_tail) :]
        != expected_formal_tail
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball receipt is outside the fixed capsule layout"
        )
    capsule_id = formal_pure.parts[len(store_parts)]
    if _CAPSULE_ID.fullmatch(capsule_id) is None:
        raise MotionAdmissionError(
            "retained fitted-ball capsule directory is not content-addressed"
        )
    expected_retained = PurePosixPath(
        *store_parts, capsule_id, _FITTED_CAPSULE_RETAINED_BASENAME
    )
    if retained_pure != expected_retained:
        raise MotionAdmissionError(
            "retained fitted-ball capsule receipt path is not canonical"
        )
    repo_input = repo_root.expanduser()
    if not repo_input.is_absolute():
        repo_input = Path.cwd() / repo_input
    resolved_repo = _assert_plain_path_components(
        repo_input, "repository root"
    )
    capsule_root = resolved_repo.joinpath(*store_parts, capsule_id)
    checkout = capsule_root / "checkout"
    artifacts = capsule_root / "artifacts"
    filesystem = {
        "capsule_root": _plain_directory_identity(
            capsule_root, "retained capsule root"
        ),
        "checkout": _plain_directory_identity(
            checkout, "retained capsule checkout"
        ),
        "artifacts": _plain_directory_identity(
            artifacts, "retained capsule artifacts"
        ),
    }
    if any(row["mode"] & 0o222 for row in filesystem.values()):
        raise MotionAdmissionError(
            "retained fitted-ball capsule directories are writable"
        )
    formal_raw, formal_stat = _read_plain_regular(
        formal_path, "MuJoCo fitted-ball formal receipt"
    )
    actual_formal_sha = hashlib.sha256(formal_raw).hexdigest()
    if actual_formal_sha != formal_sha:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball formal receipt bytes changed"
        )
    if stat.S_IMODE(formal_stat.st_mode) & 0o222:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball formal receipt is writable"
        )
    retained_raw, retained_stat = _read_plain_regular(
        retained_path, "retained fitted-ball capsule receipt"
    )
    actual_retained_sha = hashlib.sha256(retained_raw).hexdigest()
    if actual_retained_sha != retained_sha:
        raise MotionAdmissionError(
            "retained fitted-ball capsule receipt bytes changed"
        )
    if stat.S_IMODE(retained_stat.st_mode) & 0o222:
        raise MotionAdmissionError(
            "retained fitted-ball capsule receipt is writable"
        )
    formal_receipt = _strict_json_bytes(
        formal_raw, "MuJoCo fitted-ball formal receipt"
    )
    retained = _exact_keys(
        _strict_json_bytes(
            retained_raw, "retained fitted-ball capsule receipt"
        ),
        _RETAINED_FITTED_CAPSULE_KEYS,
        "retained fitted-ball capsule receipt",
    )
    sealed = dict(retained)
    observed_seal = sealed.pop("receipt_payload_sha256")
    if (
        _digest(
            observed_seal,
            "retained fitted-ball capsule receipt payload SHA",
        )
        != hashlib.sha256(_canonical_json_bytes(sealed)).hexdigest()
    ):
        raise MotionAdmissionError(
            "retained fitted-ball capsule receipt seal is false"
        )
    paths = _exact_keys(
        retained["paths"],
        _RETAINED_FITTED_CAPSULE_PATH_KEYS,
        "retained fitted-ball capsule paths",
    )
    tree = _exact_keys(
        retained["artifact_tree"],
        _RETAINED_FITTED_CAPSULE_TREE_KEYS,
        "retained fitted-ball artifact tree",
    )
    recorded_filesystem = _exact_keys(
        retained["filesystem"],
        _RETAINED_FITTED_CAPSULE_FILESYSTEM_KEYS,
        "retained fitted-ball filesystem",
    )
    for name, actual in filesystem.items():
        recorded = _exact_keys(
            recorded_filesystem[name],
            _RETAINED_FITTED_CAPSULE_STAT_KEYS,
            f"retained fitted-ball filesystem.{name}",
        )
        if dict(recorded) != actual:
            raise MotionAdmissionError(
                f"retained fitted-ball {name} inode/device/mode changed"
            )
    checkout_receipt = _exact_keys(
        retained["checkout"],
        _RETAINED_FITTED_CAPSULE_CHECKOUT_KEYS,
        "retained fitted-ball checkout receipt",
    )
    authorization = _exact_keys(
        retained["authorization"],
        frozenset(
            {
                "training_authorized",
                "deployment_authorized",
                "hardware_authorized",
            }
        ),
        "retained fitted-ball authorization",
    )
    if (
        type(retained["schema_version"]) is not int
        or retained["schema_version"] != 1
        or retained["artifact_type"]
        != "retained_formal_fitted_ball_capsule_v1"
        or retained["layout"] != _FITTED_CAPSULE_LAYOUT
        or retained["capsule_id"] != capsule_id
        or _GIT_SHA.fullmatch(str(retained["code_commit"])) is None
        or dict(paths)
        != {
            "checkout": "checkout",
            "artifacts": "artifacts",
            "formal_receipt": _FITTED_CAPSULE_FORMAL_RELPATH.as_posix(),
        }
        or type(retained["gate_return_code"]) is not int
        or dict(checkout_receipt)
        != {
            "commit": retained["code_commit"],
            "clean": True,
            "detached": True,
            "read_only": True,
        }
        or dict(authorization)
        != {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        }
    ):
        raise MotionAdmissionError(
            "retained fitted-ball capsule header/authorization is invalid"
        )
    artifact_tree = _hash_plain_artifact_tree(
        artifacts, "retained fitted-ball artifacts"
    )
    if (
        dict(tree) != artifact_tree
        or tree["symlink_free"] is not True
    ):
        raise MotionAdmissionError(
            "retained fitted-ball artifact tree bytes changed"
        )
    expected_identity = _retained_capsule_identity(
        formal_receipt=formal_receipt,
        formal_raw=formal_raw,
        artifact_tree=artifact_tree,
    )
    identity = _exact_keys(
        retained["identity"],
        _RETAINED_FITTED_CAPSULE_IDENTITY_KEYS,
        "retained fitted-ball capsule identity",
    )
    if dict(identity) != expected_identity:
        raise MotionAdmissionError(
            "retained fitted-ball capsule identity does not match stable bytes"
        )
    expected_capsule_id = hashlib.sha256(
        _canonical_json_bytes(expected_identity)
    ).hexdigest()
    if expected_capsule_id != capsule_id:
        raise MotionAdmissionError(
            "retained fitted-ball capsule directory does not match its "
            "commit/motion/solver/physics/geometry/output identity"
        )
    if retained["code_commit"] != expected_identity["code_commit"]:
        raise MotionAdmissionError(
            "retained fitted-ball checkout commit is not the executed commit"
        )
    _validate_detached_clean_checkout(
        checkout, str(retained["code_commit"])
    )
    # Re-open after the relatively expensive Git/tree checks so replacement
    # during validation is still caught before admission can be minted.
    final_formal, _ = _read_plain_regular(
        formal_path, "MuJoCo fitted-ball formal receipt after validation"
    )
    final_retained, _ = _read_plain_regular(
        retained_path,
        "retained fitted-ball capsule receipt after validation",
    )
    if final_formal != formal_raw or final_retained != retained_raw:
        raise MotionAdmissionError(
            "retained fitted-ball capsule receipts changed during validation"
        )
    return _RetainedFittedCapsule(
        root=capsule_root,
        checkout=checkout,
        artifacts=artifacts,
        formal_path=formal_path,
        formal_raw=formal_raw,
        formal_receipt=formal_receipt,
        retained_path=retained_path,
        retained_sha256=retained_sha,
    )


def _certificate_profile(binding: Any) -> tuple[int, str, int]:
    """Return certificate/report schema and paired clip count for one binding."""

    if type(binding) is BankPromotionBinding:
        return 1, "canonical-motion-bank-promotion-v1", 10
    if type(binding) is GenericBankPromotionBinding:
        return (
            2,
            "canonical-motion-bank-promotion-v2",
            2 * len(binding.motion_ids),
        )
    if type(binding) is FreshN5BankPromotionBinding:
        return (
            3,
            "canonical-motion-fresh-n5-append-swept-promotion-v3",
            2 * len(FRESH_N5_BANK_MOTION_IDS),
        )
    raise MotionAdmissionError(
        "binding must be an exact BankPromotionBinding or "
        "GenericBankPromotionBinding or FreshN5BankPromotionBinding"
    )


def _binding_document(binding: Any) -> Mapping[str, Any]:
    _certificate_profile(binding)
    document = {
        "purpose": binding.purpose,
        "bank_id": binding.bank_id,
        "scope": binding.scope,
        "registry_sha256": binding.registry_sha256,
        "alignment_sha256": binding.alignment_sha256,
        "motion_ids": list(binding.motion_ids),
        "npz_sha256": list(binding.npz_sha256),
        "canonical_ready_sha256": binding.canonical_ready_sha256,
        "canonical_ready_fk_sha256": binding.canonical_ready_fk_sha256,
        "build_manifest_sha256": list(binding.build_manifest_sha256),
        "evidence_receipts": [
            {
                "motion_id": motion_id,
                "evidence_level": level,
                "evidence_manifest_sha256": manifest_sha,
                "certificate_sha256": list(certificate_sha),
            }
            for motion_id, level, manifest_sha, certificate_sha in zip(
                binding.motion_ids,
                binding.evidence_levels,
                binding.evidence_manifest_sha256,
                binding.evidence_certificate_sha256,
            )
        ],
        "question_bank_sha256": list(binding.question_bank_sha256),
        "training_config_sha256": list(binding.training_config_sha256),
        "onnx_model_sha256": list(binding.onnx_model_sha256),
        "onnx_metadata_sha256": list(binding.onnx_metadata_sha256),
        "adoption_manifest_sha256": list(binding.adoption_manifest_sha256),
    }
    if type(binding) is FreshN5BankPromotionBinding:
        document.update(
            {
                "base_bank_id": binding.base_bank_id,
                "bank_motion_ids": list(binding.bank_motion_ids),
                "bank_npz_sha256": list(binding.bank_npz_sha256),
                "base_build_manifest_sha256": (
                    binding.base_build_manifest_sha256
                ),
                "append_build_manifest_sha256": (
                    binding.append_build_manifest_sha256
                ),
                "base_bank_gate_report_sha256": (
                    binding.base_bank_gate_report_sha256
                ),
                "append_bank_gate_report_sha256": (
                    binding.append_bank_gate_report_sha256
                ),
                "base_swept_clearance_receipt_sha256": (
                    binding.base_swept_clearance_receipt_sha256
                ),
                "append_swept_clearance_receipt_sha256": (
                    binding.append_swept_clearance_receipt_sha256
                ),
                "mujoco_fitted_ball_receipt_sha256": (
                    binding.mujoco_fitted_ball_receipt_sha256
                ),
                "mujoco_fitted_ball_capsule_receipt_sha256": (
                    binding.mujoco_fitted_ball_capsule_receipt_sha256
                ),
                "isaac_table_filtered_smoke_receipt_sha256": (
                    binding.isaac_table_filtered_smoke_receipt_sha256
                ),
            }
        )
    return document


def _binding_sha256(binding: Any) -> str:
    payload = json.dumps(
        _binding_document(binding),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_continuous_swept_receipt(
    binding_row: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    rows: tuple[Mapping[str, Any], ...],
    repo_root: Path,
) -> None:
    """Reopen the independent whole-cycle table proof bound by one bank PASS."""

    discriminated = _exact_keys(
        binding_row,
        _DISCRIMINATED_RECEIPT_KEYS,
        "continuous_swept_clearance_receipt",
    )
    if discriminated["kind"] not in (
        "canonical_base_five",
        "fresh_n5_append_suffix",
    ):
        raise MotionAdmissionError(
            "continuous swept receipt discriminator is unsupported"
        )
    certificate_receipt_path = _receipt_file(
        discriminated,
        repo_root=repo_root,
        label="continuous swept receipt",
    )
    report_binding = _exact_keys(
        report["bound_inputs"]["swept_clearance_receipt"],
        frozenset(
            {
                "path",
                "sha256",
                "independent_verifier",
                "action_ball_assembly_components_sha256",
                "robot_collision_geometry_sha256",
            }
        ),
        "bank gate swept_clearance_receipt",
    )
    report_receipt_path = _receipt_file(
        report_binding,
        repo_root=repo_root,
        label="bank gate swept-clearance receipt",
    )
    if (
        certificate_receipt_path != report_receipt_path
        or discriminated["sha256"] != report_binding["sha256"]
    ):
        raise MotionAdmissionError(
            "promotion and bank gate bind different swept-clearance bytes"
        )
    payload, receipt_sha = _snapshot(
        certificate_receipt_path, "continuous swept receipt"
    )
    receipt = _exact_keys(
        _strict_json_bytes(payload, "continuous swept receipt"),
        _SWEPT_CLEARANCE_RECEIPT_KEYS,
        "continuous swept receipt",
    )
    _assert_no_runtime_self_authorization(
        receipt, "continuous swept receipt"
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["receipt_class"] != _SWEPT_RECEIPT_CLASS
        or receipt["verdict"] != "PASS"
        or receipt["with_table"] is not True
    ):
        raise MotionAdmissionError(
            "continuous swept receipt is not an independent with-table PASS"
        )
    verifier = _exact_keys(
        receipt["independent_verifier"],
        _BANK_GATE_BINDING_KEYS,
        "continuous swept independent_verifier",
    )
    verifier_path = _receipt_file(
        verifier,
        repo_root=repo_root,
        label="continuous swept independent verifier",
    )
    bank_gate_path = repo_root.joinpath(
        "hope_training",
        "whole_body_tracking",
        "scripts",
        "canonical_motion_bank_gate.py",
    ).resolve(strict=True)
    if verifier_path == bank_gate_path:
        raise MotionAdmissionError(
            "continuous swept verifier is not independent of the bank gate"
        )
    report_verifier = _exact_keys(
        report_binding["independent_verifier"],
        _BANK_GATE_BINDING_KEYS,
        "bank gate swept independent_verifier",
    )
    if (
        report_verifier["sha256"] != verifier["sha256"]
        or _receipt_file(
            report_verifier,
            repo_root=repo_root,
            label="bank gate swept independent verifier",
        )
        != verifier_path
    ):
        raise MotionAdmissionError(
            "bank gate and swept receipt bind different independent verifiers"
        )

    bound_inputs = report["bound_inputs"]
    bank = _exact_keys(
        receipt["bank_binding"],
        _SWEPT_BANK_BINDING_KEYS,
        "continuous swept bank_binding",
    )
    expected_hashes = {
        "manifest_sha256": report["manifest"]["sha256"],
        "recipe_sha256": bound_inputs["recipe"]["sha256"],
        "ready_sha256": bound_inputs["ready"]["sha256"],
        "mjcf_sha256": bound_inputs["mjcf"]["sha256"],
        "urdf_sha256": bound_inputs["urdf"]["sha256"],
        "body_order_sha256": bound_inputs["body_order"]["sha256"],
    }
    for key, expected in expected_hashes.items():
        if _digest(bank[key], f"continuous swept {key}") != expected:
            raise MotionAdmissionError(
                f"continuous swept receipt does not bind exact {key}"
            )
    expected_station = report.get("station_center_shift_xy_m")
    if bank["station_center_shift_xy_m"] != expected_station:
        raise MotionAdmissionError(
            "continuous swept station-center binding differs from bank report"
        )
    expected_motion_ids = []
    for row in rows:
        if row["motion_id"] not in expected_motion_ids:
            expected_motion_ids.append(row["motion_id"])
    expected_matrix = {
        "motion_ids": expected_motion_ids,
        "scopes": list(_SCOPES),
        "candidate_count": len(rows),
    }
    matrix = _exact_keys(
        bank["output_matrix"],
        frozenset({"motion_ids", "scopes", "candidate_count"}),
        "continuous swept output_matrix",
    )
    if dict(matrix) != expected_matrix:
        raise MotionAdmissionError(
            "continuous swept receipt matrix is not the exact upper/full bank"
        )
    outputs = bank["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(rows):
        raise MotionAdmissionError(
            "continuous swept outputs do not cover the exact bank matrix"
        )
    for index, (raw, clip) in enumerate(zip(outputs, rows)):
        output = _exact_keys(
            raw,
            frozenset({"motion_id", "scope", "filename", "sha256"}),
            f"continuous swept outputs[{index}]",
        )
        if dict(output) != {
            "motion_id": clip["motion_id"],
            "scope": clip["scope"],
            "filename": clip["filename"],
            "sha256": clip["sha256"],
        }:
            raise MotionAdmissionError(
                "continuous swept output row differs from exact bank bytes"
            )

    trajectory = _exact_keys(
        receipt["trajectory_contract"],
        frozenset(
            {
                "coverage",
                "complete_cycle",
                "start",
                "includes_contact_opportunity",
                "end",
                "scopes",
            }
        ),
        "continuous swept trajectory_contract",
    )
    if dict(trajectory) != {
        "coverage": _SWEPT_COVERAGE,
        "complete_cycle": True,
        "start": "first_canonical_ready_frame",
        "includes_contact_opportunity": True,
        "end": "final_canonical_recovery_ready_frame",
        "scopes": list(_SCOPES),
    }:
        raise MotionAdmissionError(
            "continuous swept receipt does not cover the entire upper/full "
            "prepare-hit-recovery cycle"
        )
    method = _exact_keys(
        receipt["method"],
        frozenset(
            {
                "certificate_kind",
                "continuous_time_swept_volume",
                "sampled_or_geometry_only",
                "inter_sample_conservative_bound",
            }
        ),
        "continuous swept method",
    )
    if dict(method) != {
        "certificate_kind": "conservative_continuous_time_swept_volume",
        "continuous_time_swept_volume": True,
        "sampled_or_geometry_only": False,
        "inter_sample_conservative_bound": True,
    }:
        raise MotionAdmissionError(
            "sampled or geometry-only evidence cannot replace a continuous "
            "swept-volume proof"
        )
    scene = _exact_keys(
        receipt["scene_contract"],
        frozenset(
            {
                "subjects",
                "forbidden_world_geometry",
                "action_ball_keepout_semantics",
                "action_ball_assembly",
                "robot_geometry",
            }
        ),
        "continuous swept scene_contract",
    )
    if (
        scene["subjects"] != list(_SWEPT_SUBJECTS)
        or scene["forbidden_world_geometry"] != list(_SWEPT_OBSTACLES)
        or scene["action_ball_keepout_semantics"] != _SWEPT_KEEPOUT
    ):
        raise MotionAdmissionError(
            "continuous swept scene lost robot/racket or table/net coverage"
        )
    assembly = _exact_keys(
        scene["action_ball_assembly"],
        frozenset(
            {
                "roles",
                "geometry_sources",
                "components",
                "components_sha256",
            }
        ),
        "continuous swept action_ball_assembly",
    )
    if assembly["roles"] != list(_SWEPT_ACTION_BALL_ASSEMBLY_ROLES):
        raise MotionAdmissionError(
            "continuous swept ActionBall assembly is incomplete"
        )
    _digest(
        assembly["components_sha256"],
        "continuous swept assembly components_sha256",
    )
    geometry_sources = assembly["geometry_sources"]
    if not isinstance(geometry_sources, list) or len(geometry_sources) != 3:
        raise MotionAdmissionError(
            "continuous swept geometry-source closure is incomplete"
        )
    for index, raw in enumerate(geometry_sources):
        source = _exact_keys(
            raw,
            frozenset({"role", "path", "sha256"}),
            f"continuous swept geometry_sources[{index}]",
        )
        if source["role"] not in (
            "table_dimensions",
            "table_frame",
            "scene_builder",
        ):
            raise MotionAdmissionError(
                "continuous swept geometry source role changed"
            )
        _receipt_file(
            source,
            repo_root=repo_root,
            label=f"continuous swept geometry source {source['role']}",
        )
    robot = _exact_keys(
        scene["robot_geometry"],
        frozenset(
            {
                "all_enabled_collision_geoms",
                "collision_geom_names",
                "collision_geom_names_sha256",
                "racket_and_handle_geom_names",
            }
        ),
        "continuous swept robot_geometry",
    )
    collision_names = robot["collision_geom_names"]
    if (
        robot["all_enabled_collision_geoms"] is not True
        or not isinstance(collision_names, list)
        or not collision_names
        or collision_names != sorted(set(collision_names))
        or robot["racket_and_handle_geom_names"]
        != ["right_racket_collision", "right_racket_handle_collision"]
        or not set(robot["racket_and_handle_geom_names"]).issubset(
            collision_names
        )
    ):
        raise MotionAdmissionError(
            "continuous swept robot geometry is incomplete"
        )
    _digest(
        robot["collision_geom_names_sha256"],
        "continuous swept robot geometry SHA-256",
    )

    results = receipt["results"]
    if not isinstance(results, list) or len(results) != len(rows):
        raise MotionAdmissionError(
            "continuous swept results do not cover the exact matrix"
        )
    minimum_clearance = math.inf
    for index, (raw, clip) in enumerate(zip(results, rows)):
        result = _exact_keys(
            raw,
            _SWEPT_RESULT_KEYS,
            f"continuous swept results[{index}]",
        )
        frames = _integer(
            result["frames"],
            f"continuous swept results[{index}].frames",
            minimum=2,
        )
        interval_count = _integer(
            result["interval_count"],
            f"continuous swept results[{index}].interval_count",
        )
        clearance = _finite(
            result["minimum_clearance_certified_lower_bound_m"],
            f"continuous swept results[{index}] clearance",
            minimum=0.005,
        )
        contact_start = _finite(
            result["contact_window_start_s"],
            f"continuous swept results[{index}].contact_window_start_s",
            minimum=0.0,
        )
        contact_end = _finite(
            result["contact_window_end_s"],
            f"continuous swept results[{index}].contact_window_end_s",
            minimum=0.0,
        )
        time_law = _validate_canonical_time_law_identity(
            clip["canonical_time_law"],
            f"continuous swept clips[{index}].canonical_time_law",
        )
        marker_times = time_law["marker_contract"]["time_s"]
        duration = _finite(
            result["duration_s"],
            f"continuous swept results[{index}].duration_s",
            minimum=0.0,
        )
        if (
            result["motion_id"] != clip["motion_id"]
            or result["scope"] != clip["scope"]
            or result["filename"] != clip["filename"]
            or _digest(
                result["sha256"],
                f"continuous swept results[{index}].sha256",
            )
            != clip["sha256"]
            or frames != clip["frames"]
            or _finite(
                result["fps"],
                f"continuous swept results[{index}].fps",
                minimum=0.0,
            )
            != float(clip["fps"])
            or duration != float(clip["duration_s"])
            or not (0.0 <= contact_start <= contact_end <= duration)
            or contact_start != float(marker_times["window_start"])
            or contact_end != float(marker_times["window_end"])
            or _integer(
                result["start_frame"],
                f"continuous swept results[{index}].start_frame",
            )
            != 0
            or _integer(
                result["end_frame"],
                f"continuous swept results[{index}].end_frame",
            )
            != frames - 1
            or interval_count != frames - 1
            or _integer(
                result["certified_interval_count"],
                f"continuous swept results[{index}].certified_interval_count",
            )
            != interval_count
            or any(
                _integer(
                    result[key],
                    f"continuous swept results[{index}].{key}",
                )
                != 0
                for key in (
                    "unknown_interval_count",
                    "unsafe_interval_count",
                    "nonfinite_interval_count",
                    "hard_collision_count",
                )
            )
            or result["all_intervals_conservatively_bounded"] is not True
            or result["coverage_start"] != "first_frame"
            or result["contact_opportunity_covered"] is not True
            or result["coverage_end"] != "last_frame"
            or result["complete_cycle"] is not True
            or result["with_table"] is not True
            or result["subjects"] != list(_SWEPT_SUBJECTS)
            or result["obstacles"] != list(_SWEPT_OBSTACLES)
            or result["verdict"] != "PASS"
        ):
            raise MotionAdmissionError(
                "continuous swept result is partial, sampled, unsafe, or "
                "does not bind the exact output"
            )
        minimum_clearance = min(minimum_clearance, clearance)
    authorization = _exact_keys(
        receipt["authorization"],
        frozenset(
            {
                "swept_clearance_complete",
                "training_authorized",
                "hardware_authorized",
            }
        ),
        "continuous swept authorization",
    )
    if dict(authorization) != {
        "swept_clearance_complete": True,
        "training_authorized": False,
        "hardware_authorized": False,
    }:
        raise MotionAdmissionError(
            "continuous swept receipt authorization boundary changed"
        )
    non_claims = receipt["non_claims"]
    if (
        not isinstance(non_claims, list)
        or any(not isinstance(item, str) or not item for item in non_claims)
        or not {
            "dynamics_or_balance",
            "training_authorization",
            "hardware_authorization",
        }.issubset(set(non_claims))
    ):
        raise MotionAdmissionError(
            "continuous swept receipt non-claims are incomplete"
        )
    swept_contract = _exact_keys(
        report["contracts"]["swept_clearance"],
        frozenset(
            {
                "receipt_class",
                "with_table",
                "coverage",
                "subjects",
                "obstacles",
                "action_ball_assembly_roles",
                "action_ball_keepout_semantics",
                "continuous_time_swept_volume",
                "sampled_or_geometry_only",
                "all_exact_output_intervals_conservatively_bounded",
                "minimum_required_clearance_m",
            }
        ),
        "bank gate swept_clearance contract",
    )
    if (
        swept_contract["receipt_class"] != _SWEPT_RECEIPT_CLASS
        or swept_contract["with_table"] is not True
        or swept_contract["coverage"] != _SWEPT_COVERAGE
        or swept_contract["subjects"] != list(_SWEPT_SUBJECTS)
        or swept_contract["obstacles"] != list(_SWEPT_OBSTACLES)
        or swept_contract["action_ball_assembly_roles"]
        != list(_SWEPT_ACTION_BALL_ASSEMBLY_ROLES)
        or swept_contract["action_ball_keepout_semantics"] != _SWEPT_KEEPOUT
        or swept_contract["continuous_time_swept_volume"] is not True
        or swept_contract["sampled_or_geometry_only"] is not False
        or swept_contract[
            "all_exact_output_intervals_conservatively_bounded"
        ]
        is not True
        or _finite(
            swept_contract["minimum_required_clearance_m"],
            "bank gate swept minimum required clearance",
            minimum=0.005,
        )
        != 0.005
    ):
        raise MotionAdmissionError(
            "bank gate swept-clearance contract is incomplete"
        )
    if (
        _digest(
            report_binding["action_ball_assembly_components_sha256"],
            "bank gate ActionBall assembly digest",
        )
        != assembly["components_sha256"]
        or _digest(
            report_binding["robot_collision_geometry_sha256"],
            "bank gate robot collision geometry digest",
        )
        != robot["collision_geom_names_sha256"]
        or _finite(
            report["aggregate"][
                "swept_clearance_minimum_certified_lower_bound_m"
            ],
            "bank gate swept minimum clearance",
            minimum=0.005,
        )
        != minimum_clearance
    ):
        raise MotionAdmissionError(
            "bank gate swept-clearance summary differs from the exact receipt"
        )
    _, final_sha = _snapshot(
        certificate_receipt_path, "continuous swept receipt after validation"
    )
    if final_sha != receipt_sha:
        raise MotionAdmissionError(
            "continuous swept receipt changed during validation"
        )


def _validate_bank_gate_report(
    binding_row: Mapping[str, Any],
    *,
    binding: Any,
    repo_root: Path,
    expected_report_schema_version: int = 1,
    expected_clip_count: int = 10,
    report_profile: str = "legacy",
    expected_manifest_sha256: str | None = None,
    expected_all_npz_sha256: tuple[str, ...] | None = None,
    swept_receipt_row: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if report_profile not in (
        "legacy",
        "generic_v2",
        "fresh_base",
        "fresh_append",
    ):
        raise MotionAdmissionError("unknown bank report validation profile")
    row = _exact_keys(
        binding_row, _BANK_GATE_BINDING_KEYS, "bank_gate_report"
    )
    path = _repo_file(row["path"], repo_root, "bank_gate_report.path")
    expected_sha = _digest(row["sha256"], "bank_gate_report.sha256")
    payload, actual_sha = _snapshot(path, "bank gate report")
    if actual_sha != expected_sha:
        raise MotionAdmissionError(
            "bank gate report SHA-256 differs from the trusted certificate"
        )
    report = _exact_keys(
        _strict_json_bytes(payload, "bank gate report"),
        {
            "legacy": _BANK_GATE_REPORT_KEYS,
            "generic_v2": _BANK_GATE_REPORT_KEYS_V2,
            "fresh_base": _BANK_GATE_REPORT_KEYS,
            "fresh_append": _BANK_GATE_REPORT_KEYS_APPEND,
        }[report_profile],
        "bank gate report",
    )
    _assert_no_runtime_self_authorization(report, "bank gate report")
    expected_grounded_statuses = (
        ("PASS_GROUNDED_LEFT_MIDPOINT_RIGHT",)
        if report_profile in ("fresh_base", "fresh_append")
        else ("PASS", "COMPLETE_PASS")
    )
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != expected_report_schema_version
        or report["verdict"] != "PASS"
        or report["bank_gate_pass"] is not True
        or report["candidate_integrity_pass"] is not True
        or report["grounded_trace_status"] not in expected_grounded_statuses
        # The bank gate is an evidence producer, never its own adopter.  A
        # trusted promotion certificate supplies the separate purpose-specific
        # authorization after this diagnostic-only PASS is reviewed.
        or report["publication_class"] != "post_build_diagnostic_only"
        or report["training_authorized"] is not False
        or report["hardware_authorized"] is not False
    ):
        raise MotionAdmissionError(
            "bank gate report is not a complete, promotable exact PASS"
        )
    if report["library_id"] != binding.bank_id:
        raise MotionAdmissionError(
            "bank gate library_id differs from the promoted bank_id"
        )
    if report_profile == "generic_v2":
        selected = _exact_keys(
            report["selected_registry_binding"],
            _SELECTED_REGISTRY_BINDING_KEYS,
            "bank gate selected_registry_binding",
        )
        expected_selected = {
            "scope": binding.scope,
            "registry_sha256": binding.registry_sha256,
            "alignment_sha256": binding.alignment_sha256,
            "canonical_ready_sha256": binding.canonical_ready_sha256,
            "canonical_ready_fk_sha256": (
                binding.canonical_ready_fk_sha256
            ),
            "motion_ids": list(binding.motion_ids),
            "npz_sha256": list(binding.npz_sha256),
            "build_manifest_sha256": list(
                binding.build_manifest_sha256
            ),
        }
        if selected != expected_selected:
            raise MotionAdmissionError(
                "bank gate selected registry lineage differs from the "
                "promoted binding"
            )
    grounded_claim = report.get("contracts", {}).get(
        "grounded_inverse_dynamics"
    )
    if (
        not isinstance(grounded_claim, str)
        or not grounded_claim
        or "incomplete" in grounded_claim.lower()
        or "missing" in grounded_claim.lower()
    ):
        raise MotionAdmissionError(
            "bank gate grounded inverse-dynamics claim is not complete"
        )

    manifest = _exact_keys(
        report["manifest"], frozenset({"path", "sha256"}), "bank gate manifest"
    )
    _receipt_file(
        manifest,
        repo_root=repo_root,
        label="bank gate manifest",
    )
    if (
        expected_manifest_sha256 is not None
        and manifest["sha256"] != expected_manifest_sha256
    ):
        raise MotionAdmissionError(
            "bank gate manifest differs from the promoted build manifest"
        )
    if not isinstance(report["bank_dir"], str) or not report["bank_dir"]:
        raise MotionAdmissionError("bank gate bank_dir must be non-empty")

    bound = _exact_keys(
        report["bound_inputs"],
        (
            _BANK_GATE_BOUND_INPUT_KEYS_SWEPT
            if report_profile in ("fresh_base", "fresh_append")
            else _BANK_GATE_BOUND_INPUT_KEYS
        ),
        "bank gate bound_inputs",
    )
    bound_paths: dict[str, Path] = {}
    for name in (
        "recipe",
        "compiler",
        "geometry_tool",
        "ready",
        "mjcf",
        "urdf",
        "body_order",
    ):
        receipt = _exact_keys(
            bound[name],
            frozenset({"path", "sha256"}),
            f"bank gate bound_inputs.{name}",
        )
        expected_path = {
            "compiler": (
                "hope_training/whole_body_tracking/scripts/"
                "canonical_motion_compiler.py"
            ),
            "geometry_tool": (
                "hope_training/whole_body_tracking/scripts/"
                "canonical_motion_geometry.py"
            ),
        }.get(name)
        bound_paths[name] = _receipt_file(
            receipt,
            repo_root=repo_root,
            label=f"bank gate bound_inputs.{name}",
            expected_repo_path=expected_path,
        )
    if bound["ready"]["sha256"] != binding.canonical_ready_sha256:
        raise MotionAdmissionError(
            "bank gate ready digest differs from the promoted canonical ready"
        )
    _digest(
        bound["compiler_options_sha256"],
        "bank gate bound_inputs.compiler_options_sha256",
    )
    plant = _exact_keys(
        bound["plant"],
        frozenset(
            {
                "mjcf_sha256",
                "urdf_sha256",
                "compiled_signature_sha256",
                "identity_bound",
                "runtime_body_order",
            }
        ),
        "bank gate bound_inputs.plant",
    )
    for key in ("mjcf_sha256", "urdf_sha256", "compiled_signature_sha256"):
        _digest(plant[key], f"bank gate bound_inputs.plant.{key}")
    if (
        plant["mjcf_sha256"] != bound["mjcf"]["sha256"]
        or plant["urdf_sha256"] != bound["urdf"]["sha256"]
        or plant["identity_bound"] is not True
        or not isinstance(plant["runtime_body_order"], list)
        or not plant["runtime_body_order"]
    ):
        raise MotionAdmissionError("bank gate plant identity is not bound")
    tools = _exact_keys(
        bound["verifier_tools"],
        frozenset(
            {
                "bank_gate",
                "mujoco_motion_player",
                "canonical_mujoco_dynamics_gate",
            }
        ),
        "bank gate bound_inputs.verifier_tools",
    )
    expected_bank_gate_repo_path = (
        _GENERIC_BANK_GATE_REPO_PATH
        if report_profile == "generic_v2"
        else _LEGACY_BANK_GATE_REPO_PATH
    )
    for name, receipt_keys in (
        ("bank_gate", frozenset({"path", "sha256"})),
        ("mujoco_motion_player", frozenset({"path", "sha256"})),
        (
            "canonical_mujoco_dynamics_gate",
            frozenset({"path", "sha256", "report_schema_version"}),
        ),
    ):
        receipt = _exact_keys(
            tools[name],
            receipt_keys,
            f"bank gate verifier_tools.{name}",
        )
        _receipt_file(
            receipt,
            repo_root=repo_root,
            label=f"bank gate verifier_tools.{name}",
            expected_repo_path={
                "bank_gate": expected_bank_gate_repo_path,
                "mujoco_motion_player": (
                    "hope_training/whole_body_tracking/scripts/"
                    "mujoco_motion_player.py"
                ),
                "canonical_mujoco_dynamics_gate": (
                    "hope_training/whole_body_tracking/scripts/"
                    "canonical_mujoco_dynamics_gate.py"
                ),
            }[name],
        )
    dynamics_report_schema = tools["canonical_mujoco_dynamics_gate"][
        "report_schema_version"
    ]
    if type(dynamics_report_schema) is not int or dynamics_report_schema != 1:
        raise MotionAdmissionError(
            "bank gate dynamics verifier report schema changed"
        )

    contracts = _exact_keys(
        report["contracts"],
        {
            "legacy": _BANK_GATE_CONTRACT_KEYS,
            "generic_v2": _BANK_GATE_CONTRACT_KEYS,
            "fresh_base": _BANK_GATE_CONTRACT_KEYS_SWEPT,
            "fresh_append": _BANK_GATE_CONTRACT_KEYS_APPEND_SWEPT,
        }[report_profile],
        "bank gate contracts",
    )
    matrix = _exact_keys(
        contracts["matrix"],
        frozenset({"motion_ids", "scopes", "count"}),
        "bank gate contracts.matrix",
    )
    if (
        matrix["motion_ids"] != list(binding.motion_ids)
        or matrix["scopes"] != ["upper", "full"]
        or type(matrix["count"]) is not int
        or matrix["count"] != expected_clip_count
        or contracts["shared_ready"] is not True
        or contracts["six_endpoint_velocity_classes_exact_zero"] is not True
        or contracts["contact_opportunity_is_marker_only"] is not True
        or contracts["acceleration_allowed_through_window_end"] is not True
        or contracts["nonnegative_scalar_acceleration_through_window_end"]
        is not True
        or contracts["adv2c3_role"] != "comparator_only_not_default"
        or contracts["grounded_trace_status"]
        != report["grounded_trace_status"]
    ):
        raise MotionAdmissionError(
            "bank gate contracts do not describe the complete canonical matrix"
        )

    aggregate = _exact_keys(
        report["aggregate"],
        (
            _BANK_GATE_AGGREGATE_KEYS_FRESH
            if report_profile in ("fresh_base", "fresh_append")
            else _BANK_GATE_AGGREGATE_KEYS
        ),
        "bank gate aggregate",
    )
    exact_complete_counts = (
        "clip_count",
        "fk_pass_count",
        "velocity_consistency_pass_count",
        "joint_limit_pass_count",
        "geometry_pass_count",
        "non_torque_dynamics_pass_count",
        "complete_dynamics_pass_count",
        "torque_interpretation_valid_count",
    )
    count_keys = tuple(
        key
        for key in _BANK_GATE_AGGREGATE_KEYS
        if key == "clip_count" or key.endswith("_count")
    )
    if any(
        type(aggregate[key]) is not int or aggregate[key] < 0
        for key in count_keys
    ) or any(
        aggregate[key] != expected_clip_count
        for key in exact_complete_counts
    ) or (
        aggregate["incomplete_fail_closed_count"] != 0
        or aggregate["failed_count"] != 0
    ):
        raise MotionAdmissionError(
            "bank gate aggregate is not a complete paired-scope dynamics PASS"
        )
    if report_profile in ("fresh_base", "fresh_append"):
        if (
            type(aggregate["swept_clearance_pass_count"]) is not int
            or aggregate["swept_clearance_pass_count"]
            != expected_clip_count
            or type(aggregate["time_law_artifact_count"]) is not int
            or aggregate["time_law_artifact_count"] != expected_clip_count
            or type(aggregate["grounded_lmr_pass_count"]) is not int
            or aggregate["grounded_lmr_pass_count"] != expected_clip_count
            or type(aggregate["grounded_lmr_incomplete_count"]) is not int
            or aggregate["grounded_lmr_incomplete_count"] != 0
            or _finite(
                aggregate[
                    "swept_clearance_minimum_certified_lower_bound_m"
                ],
                "bank gate swept minimum clearance",
                minimum=0.005,
            )
            < 0.005
        ):
            raise MotionAdmissionError(
                "bank gate swept-clearance aggregate is incomplete"
            )

    clips = report.get("clips")
    if (
        not isinstance(clips, list)
        or len(clips) != expected_clip_count
    ):
        raise MotionAdmissionError(
            "bank gate report must contain the complete paired-scope matrix"
        )
    expected_matrix = tuple(
        (motion_id, scope)
        for motion_id in binding.motion_ids
        for scope in ("upper", "full")
    )
    rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(clips):
        clip = _exact_keys(
            raw,
            (
                _BANK_GATE_CLIP_KEYS_FRESH
                if report_profile in ("fresh_base", "fresh_append")
                else _BANK_GATE_CLIP_KEYS
            ),
            f"bank gate clips[{index}]",
        )
        _digest(clip["sha256"], f"bank gate clips[{index}].sha256")
        schema2 = _exact_keys(
            clip["schema2_receipts"],
            frozenset(
                {
                    "input_sha256",
                    "builder_tool_sha256",
                    "manifest_sidecar",
                    "report_sidecar",
                }
            ),
            f"bank gate clips[{index}].schema2_receipts",
        )
        _digest(
            schema2["input_sha256"],
            f"bank gate clips[{index}].schema2_receipts.input_sha256",
        )
        _digest(
            schema2["builder_tool_sha256"],
            f"bank gate clips[{index}].schema2_receipts.builder_tool_sha256",
        )
        for sidecar in ("manifest_sidecar", "report_sidecar"):
            receipt = _exact_keys(
                schema2[sidecar],
                frozenset({"path", "sha256"}),
                f"bank gate clips[{index}].schema2_receipts.{sidecar}",
            )
            _receipt_file(
                receipt,
                repo_root=repo_root,
                label=(
                    f"bank gate clips[{index}]."
                    f"schema2_receipts.{sidecar}"
                ),
            )
        builder_path = repo_root.joinpath(
            "hope_training",
            "whole_body_tracking",
            "scripts",
            "canonical_schema2_builder.py",
        )
        _, builder_sha = _snapshot(builder_path, "canonical schema2 builder")
        if schema2["builder_tool_sha256"] != builder_sha:
            raise MotionAdmissionError(
                f"bank gate clips[{index}] schema2 builder digest changed"
            )
        ready = clip["strict_schema2_and_ready"]
        dynamics = clip["plant_specific_dynamics"]
        inverse = (
            dynamics.get("inverse_dynamics", {})
            if isinstance(dynamics, Mapping)
            else {}
        )
        torque = (
            inverse.get("torque_interpretation", {})
            if isinstance(inverse, Mapping)
            else {}
        )
        endpoint_zero = (
            ready.get("six_velocity_classes_exact_zero", {})
            if isinstance(ready, Mapping)
            else {}
        )
        if (
            not isinstance(ready, Mapping)
            or ready.get("shared_joint_ready_exact") is not True
            or ready.get("shared_32_body_ready_exact") is not True
            or not isinstance(endpoint_zero, Mapping)
            or frozenset(endpoint_zero)
            != frozenset(
                {
                    "joint_start",
                    "joint_end",
                    "body_linear_start",
                    "body_linear_end",
                    "body_angular_start",
                    "body_angular_end",
                }
            )
            or not all(value is True for value in endpoint_zero.values())
            or not isinstance(clip["contact_opportunity"], Mapping)
            or clip["contact_opportunity"].get(
                "acceleration_allowed_through_window_end"
            )
            is not True
            or not isinstance(clip["mujoco_fk"], Mapping)
            or clip["mujoco_fk"].get("pass") is not True
            or not isinstance(dynamics, Mapping)
            or dynamics.get("verdict") != "PASS"
            or dynamics.get("screen_pass") is not True
            or dynamics.get("non_torque_screens_pass") is not True
            or not isinstance(torque, Mapping)
            or torque.get("valid") is not True
        ):
            raise MotionAdmissionError(
                f"bank gate clips[{index}] is not an exact ready/FK/dynamics PASS"
            )
        if report_profile in ("fresh_base", "fresh_append"):
            time_law = _validate_canonical_time_law_identity(
                clip["canonical_time_law"],
                f"bank gate clips[{index}].canonical_time_law",
            )
            grounded_lmr = clip["grounded_left_midpoint_right"]
            if (
                time_law.get(
                    "schema2_joint_tick_q_exact_after_published_dtype_cast"
                )
                is not True
                or time_law.get(
                    "schema2_joint_tick_qdot_exact_after_published_dtype_cast"
                )
                is not True
                or time_law.get(
                    "solver_input_output_array_binding_recomputed"
                )
                is not True
                or time_law.get("finite_difference_reconstruction_used")
                is not False
                or time_law.get("soft_safety_envelope_pass") is not True
                or not isinstance(grounded_lmr, Mapping)
                or grounded_lmr.get("status")
                != "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
                or grounded_lmr.get("all_feasible") is not True
                or grounded_lmr.get("finite_difference_qacc_used") is not False
                or grounded_lmr.get("qacc_contract")
                != "q_s*u+q_ss*x_from_persisted_compiler_trace"
                or grounded_lmr.get("roles")
                != ["left", "midpoint", "right"]
                or _integer(
                    grounded_lmr.get("cell_count"),
                    f"bank gate clips[{index}] grounded cell_count",
                    minimum=1,
                )
                * 3
                != _integer(
                    grounded_lmr.get("sample_count"),
                    f"bank gate clips[{index}] grounded sample_count",
                    minimum=3,
                )
            ):
                raise MotionAdmissionError(
                    f"bank gate clips[{index}] lacks the persisted time-law "
                    "and grounded left/midpoint/right PASS"
                )
        rows.append(clip)
    observed_matrix = tuple(
        (row_["motion_id"], row_["scope"]) for row_ in rows
    )
    if observed_matrix != expected_matrix:
        raise MotionAdmissionError(
            "bank gate report clip matrix order or identity changed"
        )
    scoped = [row_ for row_ in rows if row_["scope"] == binding.scope]
    observed = tuple(
        (row_["motion_id"], row_["sha256"]) for row_ in scoped
    )
    expected = tuple(zip(binding.motion_ids, binding.npz_sha256))
    if observed != expected:
        raise MotionAdmissionError(
            "bank gate report does not bind the ordered selected-scope NPZ hashes"
        )
    if expected_all_npz_sha256 is not None:
        if (
            type(expected_all_npz_sha256) is not tuple
            or tuple(row_["sha256"] for row_ in rows)
            != expected_all_npz_sha256
        ):
            raise MotionAdmissionError(
                "bank gate report does not close every ordered upper/full "
                "output hash"
            )
    if report_profile in ("fresh_base", "fresh_append"):
        bank_dir = _repo_directory(
            report["bank_dir"], repo_root, "bank gate bank_dir"
        )
        for index, clip in enumerate(rows):
            filename = clip["filename"]
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
            ):
                raise MotionAdmissionError(
                    f"bank gate clips[{index}].filename is not one leaf"
                )
            output = bank_dir / filename
            payload, actual = _snapshot(
                output, f"bank gate clips[{index}] output"
            )
            del payload
            if actual != clip["sha256"]:
                raise MotionAdmissionError(
                    f"bank gate clips[{index}] output bytes drifted"
                )
        if swept_receipt_row is None:
            raise MotionAdmissionError(
                "fresh bank report requires a discriminated swept receipt"
            )
        _validate_continuous_swept_receipt(
            swept_receipt_row,
            report=report,
            rows=tuple(rows),
            repo_root=repo_root,
        )
    return report


def _discriminated_receipt_rows(
    value: Any,
    *,
    label: str,
    expected_kinds: Mapping[str, str],
) -> Mapping[str, Mapping[str, Any]]:
    rows = _exact_keys(
        value, frozenset(expected_kinds), label
    )
    result: dict[str, Mapping[str, Any]] = {}
    for discriminator, expected_kind in expected_kinds.items():
        row = _exact_keys(
            rows[discriminator],
            _DISCRIMINATED_RECEIPT_KEYS,
            f"{label}.{discriminator}",
        )
        if row["kind"] != expected_kind:
            raise MotionAdmissionError(
                f"{label}.{discriminator}.kind changed"
            )
        _digest(row["sha256"], f"{label}.{discriminator}.sha256")
        result[discriminator] = row
    return result


def _validate_fresh_append_composition(
    append_report: Mapping[str, Any],
    base_report: Mapping[str, Any],
    *,
    binding: FreshN5BankPromotionBinding,
    repo_root: Path,
) -> None:
    if (
        append_report["append_only_base_validation_scope"]
        != "base_recipe_bytes_manifest_bytes_and_ten_output_npz_sha256_only"
        or append_report["contracts"]["verification_scope"]
        != "appended_outputs_plus_content_bound_base_identity"
        or append_report["station_center_shift_xy_m"] != [0.0, 0.0]
    ):
        raise MotionAdmissionError(
            "fresh N5 append report is not the exact no-move append contract"
        )
    composition = _exact_keys(
        append_report["append_only_composition"],
        frozenset(
            {
                "mode",
                "base_outputs_rebuilt",
                "base_recipe",
                "base_build_manifest",
                "base_output_matrix",
                "base_outputs",
                "appended_motion_ids",
                "appended_scopes",
                "station_center_shift_xy_m",
                "composed_candidate_count",
            }
        ),
        "fresh N5 append_only_composition",
    )
    if (
        composition["mode"]
        != "reuse_exact_base_outputs_compile_appended_only"
        or composition["base_outputs_rebuilt"] is not False
        or composition["appended_motion_ids"]
        != list(FRESH_N5_APPEND_MOTION_IDS)
        or composition["appended_scopes"] != list(_SCOPES)
        or composition["station_center_shift_xy_m"] != [0.0, 0.0]
        or type(composition["composed_candidate_count"]) is not int
        or composition["composed_candidate_count"] != 14
    ):
        raise MotionAdmissionError(
            "fresh N5 append composition may not rebuild, replace, reorder, "
            "or station-shift the base bank"
        )
    base_manifest = _exact_keys(
        composition["base_build_manifest"],
        _BANK_GATE_BINDING_KEYS,
        "fresh N5 append base_build_manifest",
    )
    if (
        base_manifest["sha256"] != binding.base_build_manifest_sha256
        or _receipt_file(
            base_manifest,
            repo_root=repo_root,
            label="fresh N5 append base build manifest",
        )
        != _receipt_file(
            base_report["manifest"],
            repo_root=repo_root,
            label="fresh N5 base bank manifest",
        )
    ):
        raise MotionAdmissionError(
            "fresh N5 append composition binds a different base manifest"
        )
    base_recipe = _exact_keys(
        composition["base_recipe"],
        _BANK_GATE_BINDING_KEYS,
        "fresh N5 append base_recipe",
    )
    if (
        base_recipe["sha256"]
        != base_report["bound_inputs"]["recipe"]["sha256"]
        or _receipt_file(
            base_recipe,
            repo_root=repo_root,
            label="fresh N5 append base recipe",
        )
        != _receipt_file(
            base_report["bound_inputs"]["recipe"],
            repo_root=repo_root,
            label="fresh N5 base bank recipe",
        )
    ):
        raise MotionAdmissionError(
            "fresh N5 append composition binds a different base recipe"
        )
    base_matrix = _exact_keys(
        composition["base_output_matrix"],
        frozenset({"motion_ids", "scopes", "candidate_count"}),
        "fresh N5 append base_output_matrix",
    )
    if dict(base_matrix) != {
        "motion_ids": list(FRESH_N5_BASE_MOTION_IDS),
        "scopes": list(_SCOPES),
        "candidate_count": 10,
    }:
        raise MotionAdmissionError(
            "fresh N5 append composition lost the exact canonical-five 5x2 base"
        )
    base_outputs = composition["base_outputs"]
    if not isinstance(base_outputs, list) or len(base_outputs) != 10:
        raise MotionAdmissionError(
            "fresh N5 append composition must bind all ten base outputs"
        )
    base_rows = tuple(base_report["clips"])
    for index, (raw, clip) in enumerate(zip(base_outputs, base_rows)):
        row = _exact_keys(
            raw,
            frozenset({"motion_id", "scope", "path", "sha256"}),
            f"fresh N5 append base_outputs[{index}]",
        )
        if (
            row["motion_id"] != clip["motion_id"]
            or row["scope"] != clip["scope"]
            or row["sha256"] != clip["sha256"]
        ):
            raise MotionAdmissionError(
                "fresh N5 append base output identity differs from base PASS"
            )
        output_path = _receipt_file(
            row,
            repo_root=repo_root,
            label=f"fresh N5 append base output[{index}]",
        )
        if output_path.name != clip["filename"]:
            raise MotionAdmissionError(
                "fresh N5 append base output filename differs from base PASS"
            )
    for name in ("ready", "mjcf", "urdf", "body_order"):
        base_row = base_report["bound_inputs"][name]
        append_row = append_report["bound_inputs"][name]
        if (
            base_row["sha256"] != append_row["sha256"]
            or _receipt_file(
                base_row,
                repo_root=repo_root,
                label=f"fresh N5 base {name}",
            )
            != _receipt_file(
                append_row,
                repo_root=repo_root,
                label=f"fresh N5 append {name}",
            )
        ):
            raise MotionAdmissionError(
                f"fresh N5 base/append {name} identity differs"
            )


def _validate_fresh_n5_bank_closure(
    certificate: Mapping[str, Any],
    *,
    binding: FreshN5BankPromotionBinding,
    repo_root: Path,
) -> None:
    gate_rows = _discriminated_receipt_rows(
        certificate["bank_gate_reports"],
        label="fresh N5 bank_gate_reports",
        expected_kinds={
            "base": "canonical_base_five_full_replay",
            "append": "fresh_n5_append_suffix",
        },
    )
    swept_rows = _discriminated_receipt_rows(
        certificate["continuous_swept_clearance_receipts"],
        label="fresh N5 continuous_swept_clearance_receipts",
        expected_kinds={
            "base": "canonical_base_five",
            "append": "fresh_n5_append_suffix",
        },
    )
    expected_gate_shas = {
        "base": binding.base_bank_gate_report_sha256,
        "append": binding.append_bank_gate_report_sha256,
    }
    expected_swept_shas = {
        "base": binding.base_swept_clearance_receipt_sha256,
        "append": binding.append_swept_clearance_receipt_sha256,
    }
    for discriminator in ("base", "append"):
        if gate_rows[discriminator]["sha256"] != expected_gate_shas[
            discriminator
        ]:
            raise MotionAdmissionError(
                f"fresh N5 {discriminator} bank report SHA is not crossbound"
            )
        if swept_rows[discriminator]["sha256"] != expected_swept_shas[
            discriminator
        ]:
            raise MotionAdmissionError(
                f"fresh N5 {discriminator} swept receipt SHA is not crossbound"
            )

    base_all = binding.bank_npz_sha256[:10]
    append_all = binding.bank_npz_sha256[10:]
    base_view = _BankReportView(
        bank_id=binding.base_bank_id,
        scope="upper",
        motion_ids=FRESH_N5_BASE_MOTION_IDS,
        npz_sha256=tuple(base_all[index] for index in range(0, 10, 2)),
        canonical_ready_sha256=binding.canonical_ready_sha256,
    )
    append_view = _BankReportView(
        bank_id=binding.bank_id,
        scope="upper",
        motion_ids=FRESH_N5_APPEND_MOTION_IDS,
        npz_sha256=tuple(append_all[index] for index in range(0, 4, 2)),
        canonical_ready_sha256=binding.canonical_ready_sha256,
    )
    base_report = _validate_bank_gate_report(
        {
            "path": gate_rows["base"]["path"],
            "sha256": gate_rows["base"]["sha256"],
        },
        binding=base_view,
        repo_root=repo_root,
        expected_report_schema_version=1,
        expected_clip_count=10,
        report_profile="fresh_base",
        expected_manifest_sha256=binding.base_build_manifest_sha256,
        expected_all_npz_sha256=base_all,
        swept_receipt_row=swept_rows["base"],
    )
    append_report = _validate_bank_gate_report(
        {
            "path": gate_rows["append"]["path"],
            "sha256": gate_rows["append"]["sha256"],
        },
        binding=append_view,
        repo_root=repo_root,
        expected_report_schema_version=1,
        expected_clip_count=4,
        report_profile="fresh_append",
        expected_manifest_sha256=binding.append_build_manifest_sha256,
        expected_all_npz_sha256=append_all,
        swept_receipt_row=swept_rows["append"],
    )
    _validate_fresh_append_composition(
        append_report,
        base_report,
        binding=binding,
        repo_root=repo_root,
    )


def _finite_vector(
    value: Any, size: int, label: str
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise MotionAdmissionError(
            f"{label} must be one finite length-{size} vector"
        )
    return tuple(
        _finite(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


def _vector_distance(
    first: Sequence[float], second: Sequence[float]
) -> float:
    if len(first) != len(second):
        raise MotionAdmissionError("cannot compare vectors of different size")
    return math.sqrt(
        sum(
            (float(left) - float(right)) ** 2
            for left, right in zip(first, second)
        )
    )


def _validate_fitted_convergence(
    value: Any, label: str, *, require_pass: bool
) -> None:
    row = _exact_keys(
        value,
        frozenset({"pass", "metrics", "failure_reasons", "tolerances"}),
        label,
    )
    metrics = _exact_keys(
        row["metrics"], frozenset(_FRESH_N5_CONVERGENCE_METRICS),
        f"{label}.metrics",
    )
    tolerances = _exact_keys(
        row["tolerances"],
        frozenset(_FRESH_N5_CONVERGENCE_METRICS),
        f"{label}.tolerances",
    )
    reasons = row["failure_reasons"]
    if (
        type(row["pass"]) is not bool
        or not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or (row["pass"] is True) != (reasons == [])
    ):
        raise MotionAdmissionError(
            f"{label} pass/failure-reason closure is malformed"
        )
    for name in _FRESH_N5_CONVERGENCE_METRICS:
        tolerance = _finite(
            tolerances[name], f"{label}.tolerances.{name}", minimum=0.0
        )
        if tolerance <= 0.0:
            raise MotionAdmissionError(
                f"{label}.tolerances.{name} must be positive"
            )
        metric = metrics[name]
        if metric is None:
            if row["pass"] is True:
                raise MotionAdmissionError(
                    f"{label}.metrics.{name} is absent on a PASS"
                )
            continue
        observed = _finite(
            metric, f"{label}.metrics.{name}", minimum=0.0
        )
        if (observed <= tolerance) != (
            f"nonconverged_{name}" not in reasons
        ):
            raise MotionAdmissionError(
                f"{label}.metrics.{name} disagrees with convergence verdict"
            )
    if require_pass and (row["pass"] is not True or reasons != []):
        raise MotionAdmissionError(
            f"{label} is not an exact positive-control convergence PASS"
        )


_READY_METRIC_NAMES = (
    "joint_linf_rad",
    "root_position_l2_m",
    "root_orientation_angle_rad",
    "endpoint_joint_velocity_peak_radps",
    "endpoint_root_linear_velocity_peak_mps",
    "endpoint_root_angular_velocity_peak_radps",
)
_READY_THRESHOLD_NAMES = (
    "joint_linf_rad",
    "root_position_l2_m",
    "root_orientation_angle_rad",
    "endpoint_velocity_peak",
)


def _validate_fitted_ready_recovery(value: Any, label: str) -> None:
    row = _exact_keys(
        value,
        frozenset(
            {
                "shared_ready",
                "action_recovery",
                "recovery_thresholds",
                "grounded_bank_evidence",
            }
        ),
        label,
    )
    shared = _exact_keys(
        row["shared_ready"],
        frozenset((*_READY_METRIC_NAMES, "thresholds")),
        f"{label}.shared_ready",
    )
    recovery = _exact_keys(
        row["action_recovery"],
        frozenset(_READY_METRIC_NAMES),
        f"{label}.action_recovery",
    )
    shared_thresholds = _exact_keys(
        shared["thresholds"],
        frozenset(_READY_THRESHOLD_NAMES),
        f"{label}.shared_ready.thresholds",
    )
    recovery_thresholds = _exact_keys(
        row["recovery_thresholds"],
        frozenset(_READY_THRESHOLD_NAMES),
        f"{label}.recovery_thresholds",
    )
    for metric_row, thresholds, metric_label in (
        (shared, shared_thresholds, "shared_ready"),
        (recovery, recovery_thresholds, "action_recovery"),
    ):
        for name in _READY_METRIC_NAMES:
            observed = _finite(
                metric_row[name],
                f"{label}.{metric_label}.{name}",
                minimum=0.0,
            )
            threshold_name = (
                "endpoint_velocity_peak"
                if name.startswith("endpoint_")
                else name
            )
            threshold = _finite(
                thresholds[threshold_name],
                (
                    f"{label}.{metric_label}.thresholds."
                    f"{threshold_name}"
                ),
                minimum=0.0,
            )
            if observed > threshold:
                raise MotionAdmissionError(
                    f"{label}.{metric_label}.{name} exceeds its frozen gate"
                )
    grounded = row["grounded_bank_evidence"]
    if not isinstance(grounded, Mapping):
        raise MotionAdmissionError(
            f"{label}.grounded_bank_evidence must be an object"
        )
    _assert_no_runtime_self_authorization(
        grounded, f"{label}.grounded_bank_evidence"
    )
    time_law = grounded.get("time_law")
    lmr = grounded.get("grounded_lmr")
    safety_counts = grounded.get("safety_counts")
    if (
        grounded.get("bank_gate_pass") is not True
        or grounded.get("publication_class")
        != "post_build_diagnostic_only"
        or grounded.get("training_authorized") is not False
        or grounded.get("scope") not in _SCOPES
        or grounded.get("grounded_trace_status")
        != "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
        or grounded.get("shared_ready") is not True
        or grounded.get(
            "six_endpoint_velocity_classes_exact_zero"
        )
        is not True
        or not isinstance(time_law, Mapping)
        or time_law.get("schema_version") != 2
        or time_law.get("artifact_type")
        != "canonical_time_law_collocation_v2"
        or any(
            _SHA256.fullmatch(str(time_law.get(name))) is None
            for name in (
                "artifact_npz_sha256",
                "artifact_manifest_sha256",
                "artifact_bundle_sha256",
            )
        )
        or not isinstance(lmr, Mapping)
        or type(lmr.get("cell_count")) is not int
        or type(lmr.get("sample_count")) is not int
        or lmr["cell_count"] <= 0
        or lmr["sample_count"] != 3 * lmr["cell_count"]
        or lmr.get("finite_difference_qacc_used") is not False
        or not isinstance(safety_counts, Mapping)
        or not safety_counts
        or any(type(count) is not int or count != 0 for count in safety_counts.values())
    ):
        raise MotionAdmissionError(
            f"{label} lacks exact non-authorizing grounded ready evidence"
        )


def _solver_contact_geometry_sha256(
    profile_pins: Mapping[str, Any],
    solver_payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    """Close semantic geometry bytes over profile, solver, and manifest.

    The Python implementation SHA and canonical geometry-payload SHA are
    deliberately different identities.  Both must be real: accepting any
    merely-distinct 64-hex value for the latter would let a manifest invent its
    own contact convention while still pointing at the correct source file.
    """

    solver_geometry = _exact_keys(
        solver_payload.get("contact_geometry"),
        frozenset({"payload", "sha256"}),
        f"{label}.solver_payload.contact_geometry",
    )
    profile_geometry = _exact_keys(
        profile_pins.get("contact_geometry"),
        frozenset({"payload", "sha256"}),
        f"{label}.contact_geometry",
    )
    payload = solver_geometry["payload"]
    geometry_sha = _digest(
        solver_geometry["sha256"],
        f"{label}.solver_payload.contact_geometry.sha256",
    )
    if (
        not isinstance(payload, Mapping)
        or dict(profile_geometry) != dict(solver_geometry)
        or hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        != geometry_sha
    ):
        raise MotionAdmissionError(
            f"{label} canonical contact geometry payload seal is false"
        )
    return geometry_sha


def _fitted_physics_bounds(physics_payload: Mapping[str, Any]) -> Mapping[str, float]:
    geometry = physics_payload.get("geometry_and_grading")
    if not isinstance(geometry, Mapping):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball physics profile omits geometry_and_grading"
        )
    values = {
        name: _finite(
            geometry.get(name),
            f"MuJoCo fitted-ball physics.{name}",
        )
        for name in (
            "ball_center_net_top_z_m",
            "ball_center_surface_z_m",
            "opponent_near_x_m",
            "opponent_far_x_m",
            "minimum_landing_depth_m",
            "table_half_width_m",
        )
    }
    if (
        values["opponent_far_x_m"]
        <= values["opponent_near_x_m"]
        + values["minimum_landing_depth_m"]
        or values["minimum_landing_depth_m"] < 0.0
        or values["table_half_width_m"] <= 0.0
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball physics grading bounds are invalid"
        )
    return values


def _validate_fitted_dt_result(
    value: Any,
    *,
    label: str,
    timestep_s: float,
    case_role: str,
    task_timing: Mapping[str, Any],
    task_geometry: Mapping[str, Any],
    physics_bounds: Mapping[str, float],
    contact_model_sha256: str,
    nominal_face_mesh_sha256: str,
) -> None:
    result = _exact_keys(
        value,
        frozenset(
            {
                "timestep_s",
                "verdict",
                "failure_reasons",
                "paddle_impulse_count",
                "teacher_reference_hit",
                "paddle_contact",
                "net_crossing",
                "first_landing",
                "first_landing_task_aim_w_xy_m",
                "first_landing_task_error_m",
                "incoming_task_state_error",
                "ball_net_collision",
                "activation_time_s",
                "incoming_table_bounces",
                "return_table_bounces",
                "incoming_table_bounce_times_s",
                "return_table_bounce_times_s",
                "table_contacts",
                "event_order_violations",
                "ball_forbidden_contacts",
                "shadow_probe_samples",
                "shadow_robot_obstacle_near_contacts",
                "shadow_self_near_contacts",
                "shadow_relative_motion_certificate",
                "native_ball_contact_count",
                "robot_obstacle_contacts",
                "self_contacts",
                "joint_limit_violation",
                "fall",
                "simulation_window",
                "mandatory_gates",
                "frame_metrics",
                "ready_recovery",
            }
        ),
        label,
    )
    positive = case_role in _FRESH_N5_PHYSICAL_TASK_POSITIVE_ROLES
    failure_reasons = result["failure_reasons"]
    expected_verdict = "PASS" if positive else "FAIL"
    if (
        abs(
            _finite(result["timestep_s"], f"{label}.timestep_s")
            - timestep_s
        )
        > 1.0e-12
        or result["verdict"] != expected_verdict
        or not isinstance(failure_reasons, list)
        or any(
            not isinstance(reason, str) or not reason
            for reason in failure_reasons
        )
        or (positive and failure_reasons != [])
        or (not positive and not failure_reasons)
        or result["robot_obstacle_contacts"] != []
        or result["self_contacts"] != []
        or result["shadow_robot_obstacle_near_contacts"] != []
        or result["shadow_self_near_contacts"] != []
        or result["joint_limit_violation"] is not None
        or result["fall"] is not None
        or result["event_order_violations"] != []
        or result["ball_forbidden_contacts"] != []
        or _integer(
            result["native_ball_contact_count"],
            f"{label}.native_ball_contact_count",
        )
        != 0
    ):
        raise MotionAdmissionError(
            f"{label} is not the expected physical-control verdict with "
            "zero robot/table/self/joint/fall unsafe evidence"
        )
    mandatory = _exact_keys(
        result["mandatory_gates"],
        frozenset(
            {
                "physical_ball_selected_face_return_and_first_landing",
                "teacher_matches_frozen_solver_task",
                "teacher_robot_and_racket_table_net_post_clearance",
            }
        ),
        f"{label}.mandatory_gates",
    )
    if (
        mandatory[
            "physical_ball_selected_face_return_and_first_landing"
        ]
        is not positive
        or mandatory[
            "teacher_robot_and_racket_table_net_post_clearance"
        ]
        is not True
        or (
            positive
            and mandatory["teacher_matches_frozen_solver_task"] is not True
        )
    ):
        raise MotionAdmissionError(
            f"{label} physical-return/task/safety gates disagree with its "
            "pre-registered control role"
        )
    if not positive:
        negative_signatures = {
            "negative_t_hit_offset": {
                "teacher_physical_face_center_target_mismatch",
                "teacher_task_site_target_mismatch",
                "teacher_task_face_velocity_mismatch",
                "teacher_task_site_velocity_mismatch",
                "physical_contact_time_mismatch",
                "fitted_paddle_impulse_count_not_exactly_one",
            },
            "negative_face_sign": {
                "teacher_physical_face_center_target_mismatch",
                "teacher_task_site_target_mismatch",
                "teacher_task_face_normal_mismatch",
                "teacher_selected_face_not_oriented_toward_opponent",
                "fitted_paddle_impulse_count_not_exactly_one",
            },
            "negative_ball_state_mismatch": {
                "physical_incoming_velocity_mismatch",
                "physical_contact_time_mismatch",
                "physical_contact_position_mismatch",
                "no_post_hit_net_crossing",
                "no_fitted_first_table_landing",
                "first_landing_misses_frozen_task_aim",
                "fitted_paddle_impulse_count_not_exactly_one",
            },
        }[case_role]
        if not negative_signatures.intersection(failure_reasons):
            raise MotionAdmissionError(
                f"{label} did not fail for its pre-registered negative-control "
                "signature"
            )

    window = _exact_keys(
        result["simulation_window"],
        frozenset(
            {
                "start_time_s",
                "executed_end_time_s",
                "required_ready_to_recovery_end_time_s",
                "task_pre_swing_wait_s",
                "executed_pre_swing_wait_s",
                "teacher_rate",
                "scaled_t_hit_s",
                "scaled_t_cycle_s",
                "physics_steps",
                "exact_teacher_pose_safety_scans",
                "post_dynamics_safety_scans",
                "expected_render_frames",
            }
        ),
        f"{label}.simulation_window",
    )
    physics_steps = _integer(
        window["physics_steps"], f"{label}.physics_steps", minimum=1
    )
    executed_end = _finite(
        window["executed_end_time_s"],
        f"{label}.executed_end_time_s",
        minimum=0.0,
    )
    required_end = _finite(
        window["required_ready_to_recovery_end_time_s"],
        f"{label}.required_end_time_s",
        minimum=0.0,
    )
    if (
        _finite(window["start_time_s"], f"{label}.start_time_s")
        != 0.0
        or executed_end < required_end
        or _integer(
            window["exact_teacher_pose_safety_scans"],
            f"{label}.exact_teacher_pose_safety_scans",
        )
        != 2 * physics_steps
        or _integer(
            window["post_dynamics_safety_scans"],
            f"{label}.post_dynamics_safety_scans",
        )
        != physics_steps
        or abs(
            _finite(
                window["task_pre_swing_wait_s"],
                f"{label}.task_pre_swing_wait_s",
                minimum=0.0,
            )
            - float(task_timing["pre_swing_wait_s"])
        )
        > 1.0e-9
        or abs(
            _finite(
                window["teacher_rate"],
                f"{label}.teacher_rate",
                minimum=0.0,
            )
            - float(task_timing["teacher_rate"])
        )
        > 1.0e-9
        or abs(
            _finite(
                window["scaled_t_hit_s"],
                f"{label}.scaled_t_hit_s",
                minimum=0.0,
            )
            - float(task_timing["scaled_t_hit_s"])
        )
        > 1.0e-9
        or abs(
            _finite(
                window["scaled_t_cycle_s"],
                f"{label}.scaled_t_cycle_s",
                minimum=0.0,
            )
            - float(task_timing["scaled_t_cycle_s"])
        )
        > 1.0e-9
    ):
        raise MotionAdmissionError(
            f"{label} timing/recovery simulation window is incomplete or "
            "not the frozen task"
        )
    frame_metrics = result["frame_metrics"]
    if not isinstance(frame_metrics, list) or len(frame_metrics) != physics_steps:
        raise MotionAdmissionError(
            f"{label} does not retain one metric row per physics step"
        )
    shadow = result["shadow_relative_motion_certificate"]
    if (
        not isinstance(shadow, Mapping)
        or _integer(
            shadow.get("intervals"), f"{label}.shadow.intervals", minimum=1
        )
        < 1
        or abs(
            _finite(
                shadow.get("covered_duration_s"),
                f"{label}.shadow.covered_duration_s",
                minimum=0.0,
            )
            - _finite(
                shadow.get("required_duration_s"),
                f"{label}.shadow.required_duration_s",
                minimum=0.0,
            )
        )
        > 1.0e-9
        or _finite(
            shadow.get("ball_plus_robot_guard_margin_m"),
            f"{label}.shadow.ball_plus_robot_guard_margin_m",
            minimum=0.0,
        )
        < 0.0
        or _finite(
            shadow.get("two_robot_surface_guard_margin_m"),
            f"{label}.shadow.two_robot_surface_guard_margin_m",
            minimum=0.0,
        )
        < 0.0
        or shadow.get("motion_frame_knots_are_interval_boundaries")
        is not True
        or shadow.get("whole_prep_hit_recovery_required") is not True
    ):
        raise MotionAdmissionError(
            f"{label} whole-cycle shadow clearance certificate is incomplete"
        )
    _validate_fitted_ready_recovery(
        result["ready_recovery"], f"{label}.ready_recovery"
    )
    if not positive:
        return

    if (
        _integer(
            result["paddle_impulse_count"],
            f"{label}.paddle_impulse_count",
        )
        != 1
        or result["ball_net_collision"] is not None
        or _integer(
            result["incoming_table_bounces"],
            f"{label}.incoming_table_bounces",
        )
        != 1
        or _integer(
            result["return_table_bounces"],
            f"{label}.return_table_bounces",
            minimum=1,
        )
        < 1
    ):
        raise MotionAdmissionError(
            f"{label} lacks the exact one-paddle, one-incoming-bounce legal "
            "physical return"
        )
    incoming_times = result["incoming_table_bounce_times_s"]
    return_times = result["return_table_bounce_times_s"]
    if (
        not isinstance(incoming_times, list)
        or len(incoming_times) != 1
        or not isinstance(return_times, list)
        or len(return_times) != result["return_table_bounces"]
    ):
        raise MotionAdmissionError(
            f"{label} table-bounce event ledger is incomplete"
        )
    activation = _finite(
        result["activation_time_s"], f"{label}.activation_time_s", minimum=0.0
    )
    contact = result["paddle_contact"]
    net = result["net_crossing"]
    landing = result["first_landing"]
    if not all(isinstance(item, Mapping) for item in (contact, net, landing)):
        raise MotionAdmissionError(
            f"{label} omits contact/net/landing physical events"
        )
    contact_time = _finite(
        contact.get("time_s"), f"{label}.paddle_contact.time_s", minimum=0.0
    )
    net_time = _finite(
        net.get("time_s"), f"{label}.net_crossing.time_s", minimum=0.0
    )
    landing_time = _finite(
        landing.get("time_s"), f"{label}.first_landing.time_s", minimum=0.0
    )
    if not activation < contact_time < net_time < landing_time:
        raise MotionAdmissionError(
            f"{label} physical event ordering is not launch-contact-net-land"
        )
    contact_center = _finite_vector(
        contact.get("ball_center_m"), 3, f"{label}.paddle_contact.ball_center_m"
    )
    task_contact = _finite_vector(
        task_geometry["ball_contact_w_m"],
        3,
        f"{label}.task_geometry.ball_contact_w_m",
    )
    if (
        _vector_distance(contact_center, task_contact) > 0.005
        or contact.get("face_mesh_sha256") != nominal_face_mesh_sha256
        or contact.get("selected_face_sign")
        != task_geometry["mount_normal_sign"]
        or contact.get("contact_model_sha256")
        != contact_model_sha256
        or _finite(
            contact.get("face_edge_clearance_m"),
            f"{label}.paddle_contact.face_edge_clearance_m",
            minimum=0.0,
        )
        < _finite(
            contact.get("required_face_edge_clearance_m"),
            f"{label}.paddle_contact.required_face_edge_clearance_m",
            minimum=0.0,
        )
        or _finite(
            contact.get("selected_face_return_normal_x_margin"),
            f"{label}.paddle_contact.return_normal_margin",
        )
        <= 0.0
        or _finite(
            contact.get("relative_normal_speed_mps"),
            f"{label}.paddle_contact.relative_normal_speed_mps",
        )
        >= 0.0
    ):
        raise MotionAdmissionError(
            f"{label} selected-face fitted contact is stale or not physical"
        )
    for name in (
        "face_point_m",
        "face_point_local_m",
        "selected_face_return_normal_w",
        "face_point_velocity_mps",
        "velocity_minus_mps",
        "velocity_plus_mps",
        "spin_minus_radps",
        "spin_plus_radps",
    ):
        _finite_vector(
            contact.get(name), 3, f"{label}.paddle_contact.{name}"
        )
    net_z = _finite(
        net.get("ball_center_z_m"),
        f"{label}.net_crossing.ball_center_z_m",
    )
    required_net_z = _finite(
        net.get("required_center_z_m"),
        f"{label}.net_crossing.required_center_z_m",
    )
    if (
        net.get("cleared") is not True
        or abs(
            required_net_z
            - physics_bounds["ball_center_net_top_z_m"]
        )
        > 1.0e-9
        or net_z <= required_net_z
        or abs(
            _finite(
                net.get("ball_center_y_m"),
                f"{label}.net_crossing.ball_center_y_m",
            )
        )
        > physics_bounds["table_half_width_m"]
        or abs(
            _finite(
                net.get("clearance_m"),
                f"{label}.net_crossing.clearance_m",
            )
            - (net_z - required_net_z)
        )
        > 1.0e-9
    ):
        raise MotionAdmissionError(
            f"{label} did not physically clear the exact net geometry"
        )
    landing_xy = _finite_vector(
        landing.get("ball_center_xy_m"),
        2,
        f"{label}.first_landing.ball_center_xy_m",
    )
    landing_aim = _finite_vector(
        task_geometry["landing_aim_w_xy_m"],
        2,
        f"{label}.task_geometry.landing_aim_w_xy_m",
    )
    landing_error = _finite(
        result["first_landing_task_error_m"],
        f"{label}.first_landing_task_error_m",
        minimum=0.0,
    )
    if (
        landing.get("authority") != "venue_fitted_table_impulse"
        or landing_xy[0]
        < physics_bounds["opponent_near_x_m"]
        + physics_bounds["minimum_landing_depth_m"]
        or landing_xy[0] > physics_bounds["opponent_far_x_m"]
        or abs(landing_xy[1]) > physics_bounds["table_half_width_m"]
        or abs(
            _finite(
                landing.get("ball_center_z_m"),
                f"{label}.first_landing.ball_center_z_m",
            )
            - physics_bounds["ball_center_surface_z_m"]
        )
        > 0.002
        or _vector_distance(landing_xy, landing_aim) > 0.10
        or abs(
            landing_error - _vector_distance(landing_xy, landing_aim)
        )
        > 1.0e-9
        or _finite_vector(
            result["first_landing_task_aim_w_xy_m"],
            2,
            f"{label}.first_landing_task_aim_w_xy_m",
        )
        != landing_aim
    ):
        raise MotionAdmissionError(
            f"{label} first physical landing misses the frozen legal target"
        )
    incoming_error = result["incoming_task_state_error"]
    if not isinstance(incoming_error, Mapping):
        raise MotionAdmissionError(
            f"{label}.incoming_task_state_error must be an object"
        )
    for metric, tolerance_name in (
        ("velocity_mps", "velocity_tolerance_mps"),
        ("spin_radps", "spin_tolerance_radps"),
    ):
        if _finite(
            incoming_error.get(metric),
            f"{label}.incoming_task_state_error.{metric}",
            minimum=0.0,
        ) > _finite(
            incoming_error.get(tolerance_name),
            f"{label}.incoming_task_state_error.{tolerance_name}",
            minimum=0.0,
        ):
            raise MotionAdmissionError(
                f"{label} incoming physical ball differs from frozen proposal"
            )


def _validate_fitted_physical_task_binding(
    summary_value: Any,
    *,
    manifest_action: Mapping[str, Any],
    action_row: Mapping[str, Any],
    action_index: int,
    solver_profile_sha256: str,
    physics_profile_sha256: str,
    solver_source_sha256: Mapping[str, Any],
    physics_bounds: Mapping[str, float],
    contact_model_sha256: str,
    snapshot_rows_by_role: Mapping[str, list[Mapping[str, Any]]],
    reopen_checkout_artifact: Any,
) -> None:
    action_id = str(action_row["action_id"])
    label = f"MuJoCo fitted-ball actions[{action_index}].physical_task_binding"
    summary = _exact_keys(
        summary_value, _FRESH_N5_PHYSICAL_TASK_BINDING_KEYS, label
    )
    raw_binding = _exact_keys(
        manifest_action.get("physical_task_binding"),
        frozenset(
            {
                "schema_version",
                "authority",
                "action_id",
                "action_uid",
                "motion_sha256",
                "ball_profile_sha256",
                "solver_profile_sha256",
                "physics_profile_sha256",
                "solver_implementation_source_sha256",
                "solver_execution_receipt_path",
                "solver_execution_receipt_sha256",
                "solver_execution_identity",
                "solver_execution_identity_sha256",
                "selector_executed",
                "action_identity_frozen",
                "cases",
                "cases_sha256",
            }
        ),
        f"{label}.manifest_binding",
    )
    ball_profile_sha = hashlib.sha256(
        _canonical_json_bytes(manifest_action["ball_profile"])
    ).hexdigest()
    normalized_source_map = {
        name: _digest(
            digest, f"{label}.solver_source_sha256.{name}"
        )
        for name, digest in solver_source_sha256.items()
    }
    if (
        raw_binding["schema_version"] != 1
        or raw_binding["authority"]
        != "pre_registered_frozen_action_ball_solver_receipt_v1"
        or raw_binding["action_id"] != action_id
        or raw_binding["action_uid"] != action_row["action_uid"]
        or raw_binding["motion_sha256"] != action_row["motion_sha256"]
        or raw_binding["ball_profile_sha256"] != ball_profile_sha
        or raw_binding["solver_profile_sha256"]
        != solver_profile_sha256
        or raw_binding["physics_profile_sha256"]
        != physics_profile_sha256
        or dict(
            _exact_keys(
                raw_binding["solver_implementation_source_sha256"],
                _ACTION_BALL_SOLVER_SOURCE_NAMES,
                f"{label}.manifest_solver_sources",
            )
        )
        != normalized_source_map
        or raw_binding["selector_executed"] is not False
        or raw_binding["action_identity_frozen"] is not True
        or summary["ball_profile_sha256"] != ball_profile_sha
        or summary["solver_profile_sha256"] != solver_profile_sha256
        or summary["physics_profile_sha256"] != physics_profile_sha256
        or dict(
            _exact_keys(
                summary["solver_source_sha256"],
                _ACTION_BALL_SOLVER_SOURCE_NAMES,
                f"{label}.solver_source_sha256",
            )
        )
        != normalized_source_map
    ):
        raise MotionAdmissionError(
            f"{label} action/profile/solver identity does not close"
        )
    execution_identity = _exact_keys(
        raw_binding["solver_execution_identity"],
        frozenset(
            {
                "artifact_type",
                "execution_id",
                "executed_before_gate",
                "solver_replayed_exact",
                "selector_executed",
                "action_identity_frozen",
                "action_switching_allowed",
                "hardware_authorized",
            }
        ),
        f"{label}.solver_execution_identity",
    )
    if (
        execution_identity["artifact_type"]
        != "frozen_ball_to_task_solver_execution_v1"
        or not isinstance(execution_identity["execution_id"], str)
        or not execution_identity["execution_id"]
        or execution_identity["executed_before_gate"] is not True
        or execution_identity["solver_replayed_exact"] is not True
        or execution_identity["selector_executed"] is not False
        or execution_identity["action_identity_frozen"] is not True
        or execution_identity["action_switching_allowed"] is not False
        or execution_identity["hardware_authorized"] is not False
    ):
        raise MotionAdmissionError(
            f"{label} solver execution identity is not a frozen "
            "simulation-only receipt"
        )
    execution_identity_sha = _digest(
        raw_binding["solver_execution_identity_sha256"],
        f"{label}.solver_execution_identity_sha256",
    )
    if (
        hashlib.sha256(
            _canonical_json_bytes(execution_identity)
        ).hexdigest()
        != execution_identity_sha
    ):
        raise MotionAdmissionError(
            f"{label} solver execution identity seal is false"
        )

    raw_cases = raw_binding["cases"]
    if (
        not isinstance(raw_cases, list)
        or len(raw_cases) != len(_FRESH_N5_PHYSICAL_TASK_CASE_ROLES)
        or [
            case.get("case_role")
            for case in raw_cases
            if isinstance(case, Mapping)
        ]
        != list(_FRESH_N5_PHYSICAL_TASK_CASE_ROLES)
    ):
        raise MotionAdmissionError(
            f"{label} manifest case order is not exact"
        )
    cases_sha = _digest(
        raw_binding["cases_sha256"], f"{label}.manifest_cases_sha256"
    )
    if (
        hashlib.sha256(_canonical_json_bytes(raw_cases)).hexdigest()
        != cases_sha
        or summary["cases_sha256"] != cases_sha
        or summary["case_order"]
        != list(_FRESH_N5_PHYSICAL_TASK_CASE_ROLES)
    ):
        raise MotionAdmissionError(
            f"{label} case list seal/order does not close"
        )

    solver_receipt_binding = _exact_keys(
        summary["solver_execution_receipt"],
        _FRESH_N5_SOLVER_EXECUTION_RECEIPT_BINDING_KEYS,
        f"{label}.solver_execution_receipt",
    )
    solver_receipt_sha = _digest(
        raw_binding["solver_execution_receipt_sha256"],
        f"{label}.manifest_solver_execution_receipt_sha256",
    )
    if solver_receipt_binding["sha256"] != solver_receipt_sha:
        raise MotionAdmissionError(
            f"{label} external solver receipt SHA differs from manifest"
        )
    solver_receipt_raw = reopen_checkout_artifact(
        solver_receipt_binding["path"],
        solver_receipt_sha,
        f"{label} external solver execution receipt",
    )
    snapshot_rows = snapshot_rows_by_role.get(
        f"solver_execution_receipt:{action_id}", []
    )
    if len(snapshot_rows) != 1 or snapshot_rows[0]["sha256"] != solver_receipt_sha:
        raise MotionAdmissionError(
            f"{label} external solver receipt is not uniquely snapshotted"
        )
    solver_receipt = _exact_keys(
        _strict_json_bytes(
            solver_receipt_raw, f"{label} external solver execution receipt"
        ),
        frozenset(
            {
                "schema_version",
                "artifact_type",
                "producer",
                "action_identity",
                "profile_identity",
                "solver_execution_identity",
                "cases",
                "receipt_payload_sha256",
            }
        ),
        f"{label}.external_solver_receipt",
    )
    solver_receipt_unsigned = dict(solver_receipt)
    solver_receipt_payload_sha = _digest(
        solver_receipt_unsigned.pop("receipt_payload_sha256"),
        f"{label}.external_solver_receipt.receipt_payload_sha256",
    )
    producer = _exact_keys(
        solver_receipt["producer"],
        frozenset(
            {
                "source_path",
                "source_sha256",
                "runtime_receipt_type",
                "exact_solver_replay_required",
                "selector_executed",
                "hardware_authorized",
            }
        ),
        f"{label}.external_solver_receipt.producer",
    )
    expected_hope_commands_path = (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    )
    if (
        solver_receipt["schema_version"] != 1
        or solver_receipt["artifact_type"]
        != "frozen_action_ball_solver_execution_receipt_v1"
        or hashlib.sha256(
            _canonical_json_bytes(solver_receipt_unsigned)
        ).hexdigest()
        != solver_receipt_payload_sha
        or solver_receipt_binding["receipt_payload_sha256"]
        != solver_receipt_payload_sha
        or producer["source_path"] != expected_hope_commands_path
        or producer["source_sha256"]
        != normalized_source_map["hope_commands.py"]
        or producer["runtime_receipt_type"] != "ActionBallTaskReceipt"
        or producer["exact_solver_replay_required"] is not True
        or producer["selector_executed"] is not False
        or producer["hardware_authorized"] is not False
        or dict(
            _exact_keys(
                solver_receipt["action_identity"],
                frozenset({"action_id", "action_uid", "motion_sha256"}),
                f"{label}.external_solver_receipt.action_identity",
            )
        )
        != {
            "action_id": action_id,
            "action_uid": action_row["action_uid"],
            "motion_sha256": action_row["motion_sha256"],
        }
        or dict(
            _exact_keys(
                solver_receipt["profile_identity"],
                frozenset(
                    {
                        "ball_profile_sha256",
                        "solver_profile_sha256",
                        "physics_profile_sha256",
                        "solver_implementation_source_sha256",
                        "geometry_source_sha256",
                    }
                ),
                f"{label}.external_solver_receipt.profile_identity",
            )
        )
        != {
            "ball_profile_sha256": ball_profile_sha,
            "solver_profile_sha256": solver_profile_sha256,
            "physics_profile_sha256": physics_profile_sha256,
            "solver_implementation_source_sha256": normalized_source_map,
            "geometry_source_sha256": manifest_action[
                "physical_task_binding"
            ]["cases"][0]["task_payload"]["geometry_source_sha256"],
        }
        or dict(
            _exact_keys(
                solver_receipt["solver_execution_identity"],
                frozenset(execution_identity),
                f"{label}.external_solver_receipt.execution_identity",
            )
        )
        != dict(execution_identity)
        or solver_receipt["cases"] != raw_cases
    ):
        raise MotionAdmissionError(
            f"{label} external solver receipt provenance/action/profile/cases "
            "do not close"
        )

    summary_cases = summary["cases"]
    if (
        not isinstance(summary_cases, list)
        or len(summary_cases) != len(raw_cases)
        or action_row["dt_results"] != summary_cases[0].get("dt_results")
        or action_row["convergence"] != summary_cases[0].get("convergence")
    ):
        raise MotionAdmissionError(
            f"{label} physical replay case matrix is incomplete or its "
            "top-level center alias drifted"
        )
    manifest_t_hit = _finite(
        manifest_action["reference_t_hit_s"],
        f"{label}.manifest_reference_t_hit_s",
        minimum=0.0,
    )
    manifest_t_cycle = _finite(
        manifest_action["reference_t_cycle_s"],
        f"{label}.manifest_reference_t_cycle_s",
        minimum=0.0,
    )
    manifest_speed = _finite(
        manifest_action["reference_racket_site_speed_mps"],
        f"{label}.manifest_reference_racket_site_speed_mps",
        minimum=0.0,
    )
    reaction_margin = _finite(
        manifest_action["reaction_margin_s"],
        f"{label}.manifest_reaction_margin_s",
        minimum=0.0,
    )
    rate_min = _finite(
        manifest_action["teacher_rate_min"],
        f"{label}.manifest_teacher_rate_min",
        minimum=0.0,
    )
    rate_max = _finite(
        manifest_action["teacher_rate_max"],
        f"{label}.manifest_teacher_rate_max",
        minimum=0.0,
    )
    ball_profile = manifest_action["ball_profile"]
    case_ids: set[str] = set()
    center_case_payloads: list[Mapping[str, Any]] = []
    for case_index, (case_role, raw_case_value, replay_value) in enumerate(
        zip(
            _FRESH_N5_PHYSICAL_TASK_CASE_ROLES,
            raw_cases,
            summary_cases,
        )
    ):
        case_label = f"{label}.cases[{case_index}]"
        raw_case = _exact_keys(
            raw_case_value,
            frozenset(
                {
                    "case_id",
                    "case_role",
                    "sample_seed",
                    "expected_physical_verdict",
                    "expected_failure_reason",
                    "ball_proposal",
                    "ball_proposal_sha256",
                    "task_payload",
                    "task_payload_sha256",
                    "fault_injection",
                    "case_binding_sha256",
                }
            ),
            f"{case_label}.manifest_case",
        )
        replay = _exact_keys(
            replay_value, _FRESH_N5_PHYSICAL_TASK_CASE_KEYS, case_label
        )
        case_id = raw_case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise MotionAdmissionError(
                f"{case_label}.case_id is empty or duplicate"
            )
        case_ids.add(case_id)
        sample_seed = _integer(
            raw_case["sample_seed"],
            f"{case_label}.sample_seed",
        )
        positive = case_role in _FRESH_N5_PHYSICAL_TASK_POSITIVE_ROLES
        expected_verdict = "PASS" if positive else "FAIL"
        expected_reason = (
            None
            if positive
            else _FRESH_N5_PHYSICAL_TASK_NEGATIVE_REASON[case_role]
        )
        if (
            raw_case["case_role"] != case_role
            or raw_case["expected_physical_verdict"]
            != expected_verdict
            or raw_case["expected_failure_reason"] != expected_reason
            or replay["case_id"] != case_id
            or replay["case_role"] != case_role
            or replay["sample_seed"] != sample_seed
            or replay["expected_physical_verdict"]
            != expected_verdict
            or replay["expected_failure_reason"] != expected_reason
        ):
            raise MotionAdmissionError(
                f"{case_label} role/seed/expectation changed"
            )
        proposal = _exact_keys(
            raw_case["ball_proposal"],
            frozenset(
                {
                    "action_id",
                    "action_uid",
                    "motion_sha256",
                    "sample_seed",
                    "sample_index",
                    "ball_contact_w_m",
                    "time_to_contact_s",
                    "incoming_velocity_w_mps",
                    "incoming_spin_w_radps",
                    "base_spawn_w_m",
                    "base_goal_w_m",
                    "landing_aim_w_xy_m",
                    "launch",
                }
            ),
            f"{case_label}.ball_proposal",
        )
        proposal_sha = _digest(
            raw_case["ball_proposal_sha256"],
            f"{case_label}.ball_proposal_sha256",
        )
        contact_w = _finite_vector(
            proposal["ball_contact_w_m"],
            3,
            f"{case_label}.ball_contact_w_m",
        )
        incoming_velocity = _finite_vector(
            proposal["incoming_velocity_w_mps"],
            3,
            f"{case_label}.incoming_velocity_w_mps",
        )
        incoming_spin = _finite_vector(
            proposal["incoming_spin_w_radps"],
            3,
            f"{case_label}.incoming_spin_w_radps",
        )
        base_spawn = _finite_vector(
            proposal["base_spawn_w_m"],
            3,
            f"{case_label}.base_spawn_w_m",
        )
        base_goal = _finite_vector(
            proposal["base_goal_w_m"],
            3,
            f"{case_label}.base_goal_w_m",
        )
        landing_aim = _finite_vector(
            proposal["landing_aim_w_xy_m"],
            2,
            f"{case_label}.landing_aim_w_xy_m",
        )
        time_to_contact = _finite(
            proposal["time_to_contact_s"],
            f"{case_label}.time_to_contact_s",
            minimum=0.0,
        )
        if (
            proposal["action_id"] != action_id
            or proposal["action_uid"] != action_row["action_uid"]
            or proposal["motion_sha256"] != action_row["motion_sha256"]
            or proposal["sample_seed"] != sample_seed
            or _integer(
                proposal["sample_index"], f"{case_label}.sample_index"
            )
            < 0
            or incoming_velocity[0] >= -1.0e-6
            or base_spawn != base_goal
            or hashlib.sha256(
                _canonical_json_bytes(proposal)
            ).hexdigest()
            != proposal_sha
            or replay["ball_proposal_sha256"] != proposal_sha
        ):
            raise MotionAdmissionError(
                f"{case_label} proposal/action/no-move/hash identity is false"
            )
        launch = _exact_keys(
            proposal["launch"],
            frozenset(
                {
                    "activation_time_s",
                    "position_w_m",
                    "velocity_w_mps",
                    "spin_w_radps",
                    "required_incoming_table_bounces",
                    "state_sha256",
                }
            ),
            f"{case_label}.launch",
        )
        launch_payload = {
            key: launch[key]
            for key in (
                "activation_time_s",
                "position_w_m",
                "velocity_w_mps",
                "spin_w_radps",
                "required_incoming_table_bounces",
            )
        }
        activation = _finite(
            launch["activation_time_s"],
            f"{case_label}.launch.activation_time_s",
            minimum=0.0,
        )
        _finite_vector(
            launch["position_w_m"], 3, f"{case_label}.launch.position_w_m"
        )
        launch_velocity = _finite_vector(
            launch["velocity_w_mps"],
            3,
            f"{case_label}.launch.velocity_w_mps",
        )
        _finite_vector(
            launch["spin_w_radps"], 3, f"{case_label}.launch.spin_w_radps"
        )
        if (
            activation >= time_to_contact
            or launch_velocity[0] >= -1.0e-6
            or launch["required_incoming_table_bounces"] != 1
            or hashlib.sha256(
                _canonical_json_bytes(launch_payload)
            ).hexdigest()
            != launch["state_sha256"]
        ):
            raise MotionAdmissionError(
                f"{case_label} launch state is not a sealed incoming ball"
            )
        task = _exact_keys(
            raw_case["task_payload"],
            frozenset(
                {
                    "action_id",
                    "action_uid",
                    "motion_sha256",
                    "ball_proposal_sha256",
                    "mount_normal_sign",
                    "ball_contact_w_m",
                    "racket_site_target_w_m",
                    "racket_normal_w",
                    "reference_racket_quat_wxyz",
                    "reference_racket_angular_velocity_w_radps",
                    "racket_command_quat_wxyz",
                    "racket_face_center_velocity_w_mps",
                    "racket_site_velocity_w_mps",
                    "racket_command_angular_velocity_w_radps",
                    "geometry_source_sha256",
                    "reference_t_hit_s",
                    "reference_t_cycle_s",
                    "reference_racket_site_speed_mps",
                    "required_racket_site_speed_mps",
                    "reaction_margin_s",
                    "teacher_rate_min",
                    "teacher_rate_max",
                    "teacher_rate",
                    "scaled_t_hit_s",
                    "scaled_t_cycle_s",
                    "pre_swing_wait_s",
                    "solver_residual_m",
                    "landing_aim_w_xy_m",
                    "solver_profile_sha256",
                    "physics_profile_sha256",
                }
            ),
            f"{case_label}.task_payload",
        )
        task_sha = _digest(
            raw_case["task_payload_sha256"],
            f"{case_label}.task_payload_sha256",
        )
        required_speed = _finite(
            task["required_racket_site_speed_mps"],
            f"{case_label}.required_racket_site_speed_mps",
            minimum=0.0,
        )
        teacher_rate = _finite(
            task["teacher_rate"],
            f"{case_label}.teacher_rate",
            minimum=0.0,
        )
        scaled_t_hit = _finite(
            task["scaled_t_hit_s"],
            f"{case_label}.scaled_t_hit_s",
            minimum=0.0,
        )
        scaled_t_cycle = _finite(
            task["scaled_t_cycle_s"],
            f"{case_label}.scaled_t_cycle_s",
            minimum=0.0,
        )
        pre_swing_wait = _finite(
            task["pre_swing_wait_s"],
            f"{case_label}.pre_swing_wait_s",
            minimum=0.0,
        )
        task_geometry = _exact_keys(
            replay["task_geometry"],
            _FRESH_N5_TASK_GEOMETRY_KEYS,
            f"{case_label}.task_geometry",
        )
        expected_task_geometry = {
            key: task[key] for key in _FRESH_N5_TASK_GEOMETRY_KEYS
        }
        task_timing = _exact_keys(
            replay["task_timing"],
            _FRESH_N5_TASK_TIMING_KEYS,
            f"{case_label}.task_timing",
        )
        expected_task_timing = {
            key: task[key] for key in _FRESH_N5_TASK_TIMING_KEYS
        }
        racket_normal = _finite_vector(
            task["racket_normal_w"], 3, f"{case_label}.racket_normal_w"
        )
        command_quat = _finite_vector(
            task["racket_command_quat_wxyz"],
            4,
            f"{case_label}.racket_command_quat_wxyz",
        )
        if (
            task["action_id"] != action_id
            or task["action_uid"] != action_row["action_uid"]
            or task["motion_sha256"] != action_row["motion_sha256"]
            or task["ball_proposal_sha256"] != proposal_sha
            or task["mount_normal_sign"]
            != manifest_action["mount_normal_sign"]
            or _finite_vector(
                task["ball_contact_w_m"],
                3,
                f"{case_label}.task_ball_contact_w_m",
            )
            != contact_w
            or _finite_vector(
                task["landing_aim_w_xy_m"],
                2,
                f"{case_label}.task_landing_aim_w_xy_m",
            )
            != landing_aim
            or abs(sum(value * value for value in racket_normal) - 1.0)
            > 4.0e-5
            or abs(sum(value * value for value in command_quat) - 1.0)
            > 4.0e-5
            or task["geometry_source_sha256"]
            != manifest_action.get("physical_task_binding", {}).get(
                "cases", [{}]
            )[0].get("task_payload", {}).get(
                "geometry_source_sha256"
            )
            or task["solver_profile_sha256"] != solver_profile_sha256
            or task["physics_profile_sha256"] != physics_profile_sha256
            or abs(float(task["reference_t_hit_s"]) - manifest_t_hit)
            > 1.0e-9
            or abs(float(task["reference_t_cycle_s"]) - manifest_t_cycle)
            > 1.0e-9
            or abs(
                float(task["reference_racket_site_speed_mps"])
                - manifest_speed
            )
            > 1.0e-9
            or abs(float(task["reaction_margin_s"]) - reaction_margin)
            > 1.0e-9
            or abs(float(task["teacher_rate_min"]) - rate_min) > 1.0e-9
            or abs(float(task["teacher_rate_max"]) - rate_max) > 1.0e-9
            or required_speed <= 0.0
            or not rate_min <= teacher_rate <= rate_max
            or abs(teacher_rate - required_speed / manifest_speed) > 1.0e-9
            or abs(scaled_t_hit - manifest_t_hit / teacher_rate) > 1.0e-9
            or abs(scaled_t_cycle - manifest_t_cycle / teacher_rate)
            > 1.0e-9
            or abs(pre_swing_wait - (time_to_contact - scaled_t_hit))
            > 1.0e-9
            or pre_swing_wait < reaction_margin
            or pre_swing_wait > 1.0
            or _finite(
                task["solver_residual_m"],
                f"{case_label}.solver_residual_m",
                minimum=0.0,
            )
            < 0.0
            or hashlib.sha256(
                _canonical_json_bytes(task)
            ).hexdigest()
            != task_sha
            or replay["task_payload_sha256"] != task_sha
            or dict(task_geometry) != expected_task_geometry
            or dict(task_timing) != expected_task_timing
            or replay["solved_task_geometry_sha256"]
            != hashlib.sha256(
                _canonical_json_bytes(expected_task_geometry)
            ).hexdigest()
            or replay["solver_execution_identity"]
            != dict(execution_identity)
        ):
            raise MotionAdmissionError(
                f"{case_label} frozen ball-to-task formula/geometry/hash "
                "does not close"
            )
        case_binding_payload = {
            "action_id": action_id,
            "action_uid": action_row["action_uid"],
            "motion_sha256": action_row["motion_sha256"],
            "case_id": case_id,
            "case_role": case_role,
            "sample_seed": sample_seed,
            "ball_proposal_sha256": proposal_sha,
            "task_payload_sha256": task_sha,
            "solver_execution_identity_sha256": execution_identity_sha,
            "fault_injection": raw_case["fault_injection"],
            "expected_physical_verdict": expected_verdict,
            "expected_failure_reason": expected_reason,
        }
        case_binding_sha = _digest(
            raw_case["case_binding_sha256"],
            f"{case_label}.case_binding_sha256",
        )
        if (
            hashlib.sha256(
                _canonical_json_bytes(case_binding_payload)
            ).hexdigest()
            != case_binding_sha
            or replay["case_binding_sha256"] != case_binding_sha
        ):
            raise MotionAdmissionError(
                f"{case_label} case binding seal is false"
            )
        control = _exact_keys(
            replay["control"],
            _FRESH_N5_PHYSICAL_CONTROL_KEYS,
            f"{case_label}.control",
        )
        if (
            control["expected_physical_verdict"] != expected_verdict
            or control["expected_failure_reason"] != expected_reason
            or control["observed_physical_verdict"] != expected_verdict
            or control["observed_failure_reason"] != expected_reason
            or control["observed_dt_verdicts"]
            != {"0.0010": expected_verdict, "0.0005": expected_verdict}
            or control["convergence_required"] is not positive
            or (
                positive
                and control["convergence_pass"] is not True
            )
            or (
                not positive
                and control["convergence_pass"] is not None
            )
            or control["control_verdict"] != "PASS"
            or control["failure_reasons"] != []
            or replay["observed_physical_verdict"] != expected_verdict
            or replay["control_verdict"] != "PASS"
            or replay["failure_reasons"] != []
        ):
            raise MotionAdmissionError(
                f"{case_label} positive/negative control ledger is false"
            )
        fault = raw_case["fault_injection"]
        fault_application = control["fault_application"]
        if not isinstance(fault, Mapping) or not isinstance(
            fault_application, Mapping
        ):
            raise MotionAdmissionError(
                f"{case_label} fault injection ledger is malformed"
            )
        if positive:
            exact_fault = _exact_keys(
                fault,
                frozenset({"kind"}),
                f"{case_label}.fault_injection",
            )
            if exact_fault["kind"] != "none":
                raise MotionAdmissionError(
                    f"{case_label} positive control contains a fault"
                )
        elif case_role == "negative_t_hit_offset":
            exact_fault = _exact_keys(
                fault,
                frozenset({"kind", "offset_s"}),
                f"{case_label}.fault_injection",
            )
            offset = _finite(
                exact_fault["offset_s"],
                f"{case_label}.fault_injection.offset_s",
            )
            if (
                exact_fault["kind"] != "teacher_t_hit_offset"
                or not 0.02 <= abs(offset) <= 0.25
            ):
                raise MotionAdmissionError(
                    f"{case_label} t_hit fault is not the preregistered "
                    "bounded offset"
                )
        elif case_role == "negative_face_sign":
            exact_fault = _exact_keys(
                fault,
                frozenset({"kind"}),
                f"{case_label}.fault_injection",
            )
            if exact_fault["kind"] != "selected_face_sign_flip":
                raise MotionAdmissionError(
                    f"{case_label} face-sign fault kind changed"
                )
        elif case_role == "negative_ball_state_mismatch":
            exact_fault = _exact_keys(
                fault,
                frozenset({"kind", "launch_velocity_delta_w_mps"}),
                f"{case_label}.fault_injection",
            )
            delta = _finite_vector(
                exact_fault["launch_velocity_delta_w_mps"],
                3,
                (
                    f"{case_label}.fault_injection"
                    ".launch_velocity_delta_w_mps"
                ),
            )
            delta_norm = math.sqrt(
                sum(component * component for component in delta)
            )
            if (
                exact_fault["kind"] != "launch_velocity_delta"
                or not 0.25 <= delta_norm <= 1.0
            ):
                raise MotionAdmissionError(
                    f"{case_label} ball-state fault is not the "
                    "preregistered bounded velocity delta"
                )
        else:  # pragma: no cover - exact role tuple above makes this defensive.
            raise MotionAdmissionError(
                f"{case_label} has an unsupported fault-control role"
            )

        common_application_keys = frozenset(
            {
                "kind",
                "applied",
                "nominal_mount_normal_sign",
                "executed_mount_normal_sign",
                "nominal_pre_swing_wait_s",
                "executed_pre_swing_wait_s",
                "nominal_launch_velocity_w_mps",
                "executed_launch_velocity_w_mps",
            }
        )
        extra_application_keys = {
            "negative_t_hit_offset": frozenset({"offset_s"}),
            "negative_ball_state_mismatch": frozenset(
                {"launch_velocity_delta_w_mps"}
            ),
        }.get(case_role, frozenset())
        applied = _exact_keys(
            fault_application,
            common_application_keys | extra_application_keys,
            f"{case_label}.fault_application",
        )
        nominal_sign = _finite(
            applied["nominal_mount_normal_sign"],
            f"{case_label}.fault_application.nominal_mount_normal_sign",
        )
        executed_sign = _finite(
            applied["executed_mount_normal_sign"],
            f"{case_label}.fault_application.executed_mount_normal_sign",
        )
        nominal_wait = _finite(
            applied["nominal_pre_swing_wait_s"],
            f"{case_label}.fault_application.nominal_pre_swing_wait_s",
            minimum=0.0,
        )
        executed_wait = _finite(
            applied["executed_pre_swing_wait_s"],
            f"{case_label}.fault_application.executed_pre_swing_wait_s",
            minimum=0.0,
        )
        nominal_launch_velocity = _finite_vector(
            applied["nominal_launch_velocity_w_mps"],
            3,
            f"{case_label}.fault_application.nominal_launch_velocity_w_mps",
        )
        executed_launch_velocity = _finite_vector(
            applied["executed_launch_velocity_w_mps"],
            3,
            f"{case_label}.fault_application.executed_launch_velocity_w_mps",
        )
        mount_sign = _finite(
            task["mount_normal_sign"],
            f"{case_label}.task_payload.mount_normal_sign",
        )
        if (
            applied["kind"] != exact_fault["kind"]
            or applied["applied"] is not True
            or nominal_sign != mount_sign
            or abs(nominal_wait - pre_swing_wait) > 1.0e-12
            or nominal_launch_velocity != launch_velocity
        ):
            raise MotionAdmissionError(
                f"{case_label} fault application does not start from the "
                "frozen nominal task"
            )
        if positive:
            application_valid = (
                executed_sign == mount_sign
                and abs(executed_wait - pre_swing_wait) <= 1.0e-12
                and executed_launch_velocity == launch_velocity
            )
        elif case_role == "negative_t_hit_offset":
            applied_offset = _finite(
                applied["offset_s"],
                f"{case_label}.fault_application.offset_s",
            )
            expected_wait = pre_swing_wait + offset
            application_valid = (
                abs(applied_offset - offset) <= 1.0e-12
                and 0.0 <= expected_wait <= 1.0
                and executed_sign == mount_sign
                and abs(executed_wait - expected_wait) <= 1.0e-12
                and executed_launch_velocity == launch_velocity
            )
        elif case_role == "negative_face_sign":
            application_valid = (
                executed_sign == -mount_sign
                and abs(executed_wait - pre_swing_wait) <= 1.0e-12
                and executed_launch_velocity == launch_velocity
            )
        else:
            applied_delta = _finite_vector(
                applied["launch_velocity_delta_w_mps"],
                3,
                (
                    f"{case_label}.fault_application"
                    ".launch_velocity_delta_w_mps"
                ),
            )
            expected_velocity = tuple(
                nominal + change
                for nominal, change in zip(launch_velocity, delta)
            )
            application_valid = (
                applied_delta == delta
                and executed_sign == mount_sign
                and abs(executed_wait - pre_swing_wait) <= 1.0e-12
                and _vector_distance(
                    executed_launch_velocity, expected_velocity
                )
                <= 1.0e-12
                and executed_launch_velocity[0] < -1.0e-6
            )
        if not application_valid:
            raise MotionAdmissionError(
                f"{case_label} injected fault was not executed exactly once "
                "as preregistered"
            )
        convergence = replay["convergence"]
        _validate_fitted_convergence(
            convergence,
            f"{case_label}.convergence",
            require_pass=positive,
        )
        dt_results = _exact_keys(
            replay["dt_results"],
            frozenset({"0.0010", "0.0005"}),
            f"{case_label}.dt_results",
        )
        for dt_name, expected_dt in (
            ("0.0010", 0.001),
            ("0.0005", 0.0005),
        ):
            _validate_fitted_dt_result(
                dt_results[dt_name],
                label=f"{case_label}.dt_results[{dt_name}]",
                timestep_s=expected_dt,
                case_role=case_role,
                task_timing=task_timing,
                task_geometry=task_geometry,
                physics_bounds=physics_bounds,
                contact_model_sha256=contact_model_sha256,
                nominal_face_mesh_sha256=action_row["face_geometry"][
                    "mesh_sha256"
                ],
            )
        if case_index < 2:
            center_case_payloads.append(
                {
                    "proposal": proposal,
                    "task": task,
                }
            )
        if case_role == "support_positive":
            speed = math.sqrt(
                sum(component * component for component in incoming_velocity)
            )
            spin = math.sqrt(
                sum(component * component for component in incoming_spin)
            )
            for observed, low_name, high_name in (
                (
                    time_to_contact,
                    "time_to_contact_min_s",
                    "time_to_contact_max_s",
                ),
                (
                    speed,
                    "incoming_speed_min_mps",
                    "incoming_speed_max_mps",
                ),
                (
                    spin,
                    "spin_magnitude_min_radps",
                    "spin_magnitude_max_radps",
                ),
            ):
                if not (
                    _finite(
                        ball_profile[low_name],
                        f"{case_label}.{low_name}",
                    )
                    <= observed
                    <= _finite(
                        ball_profile[high_name],
                        f"{case_label}.{high_name}",
                    )
                ):
                    raise MotionAdmissionError(
                        f"{case_label} support control escaped action support"
                    )
    if len(center_case_payloads) != 2:
        raise MotionAdmissionError(
            f"{label} lacks two independent center controls"
        )
    first_proposal = dict(center_case_payloads[0]["proposal"])
    second_proposal = dict(center_case_payloads[1]["proposal"])
    first_proposal.pop("sample_seed")
    second_proposal.pop("sample_seed")
    first_proposal.pop("sample_index")
    second_proposal.pop("sample_index")
    first_task = dict(center_case_payloads[0]["task"])
    second_task = dict(center_case_payloads[1]["task"])
    first_task.pop("ball_proposal_sha256")
    second_task.pop("ball_proposal_sha256")
    if (
        first_proposal != second_proposal
        or first_task != second_task
        or raw_cases[0]["sample_seed"] == raw_cases[1]["sample_seed"]
    ):
        raise MotionAdmissionError(
            f"{label} center multi-seed controls are not the same frozen task"
        )


def _validate_fresh_n5_fitted_ball_receipt(
    binding_row: Any,
    *,
    binding: FreshN5BankPromotionBinding,
    repo_root: Path,
) -> _FreshN5EvidenceIdentity:
    capsule = _reopen_retained_fitted_capsule(
        binding_row,
        repo_root=repo_root,
        expected_formal_sha256=binding.mujoco_fitted_ball_receipt_sha256,
        expected_retained_sha256=(
            binding.mujoco_fitted_ball_capsule_receipt_sha256
        ),
    )
    path = capsule.formal_path
    payload = capsule.formal_raw
    receipt_sha = hashlib.sha256(payload).hexdigest()
    receipt = _exact_keys(
        capsule.formal_receipt,
        _FRESH_N5_FITTED_BALL_KEYS,
        "MuJoCo fitted-ball receipt",
    )
    _assert_no_runtime_self_authorization(
        receipt, "MuJoCo fitted-ball receipt"
    )
    preflight = _exact_keys(
        receipt["preflight"],
        frozenset({"status", "blockers", "evidence"}),
        "MuJoCo fitted-ball preflight",
    )
    authorization = _exact_keys(
        receipt["authorization"],
        frozenset(
            {
                "training_authorized",
                "deployment_authorized",
                "hardware_authorized",
            }
        ),
        "MuJoCo fitted-ball authorization",
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["gate"] != "mujoco_teacher_motion_fitted_ball_gate"
        or receipt["contact_authority"]
        != "venue_fitted_swept_selected_face_v2"
        or receipt["native_ball_contact_enabled"] is not False
        or receipt["selector_executed"] is not False
        or receipt["ball_to_task_solver_executed"] is not False
        or receipt["ball_to_task_solver_executed_by_gate"] is not False
        or receipt[
            "pre_registered_ball_to_task_solver_receipt_consumed"
        ]
        is not True
        or receipt["solver_execution_receipt_authority"]
        != "pre_registered_frozen_action_ball_solver_receipt_v1"
        or receipt["analytic_return_scorer_executed"] is not False
        or receipt["expected_actions"] != 5
        or receipt["expected_action_order"]
        != list(FRESH_N5_DOWNSTREAM_MOTION_IDS)
        or preflight["status"] != "PASS"
        or preflight["blockers"] != []
        or not isinstance(preflight["evidence"], Mapping)
        or not preflight["evidence"]
        or receipt["formal_gate_executed"] is not True
        or receipt["status"] != "PASS"
        or receipt["verdict"] != "PASS"
        or receipt["action_order"]
        != list(FRESH_N5_DOWNSTREAM_MOTION_IDS)
        or dict(authorization)
        != {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        }
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball receipt is not the exact formal fresh-N5 PASS"
        )
    for label in (
        "runtime_code_identity",
        "runtime_environment",
        "runtime_input_snapshot",
        "runtime_code_identity_post_runtime",
        "runtime_code_identity_final",
        "compiler_mesh_assets",
        "scene_contracts",
    ):
        value = receipt[label]
        if not isinstance(value, (Mapping, list)) or not value:
            raise MotionAdmissionError(
                f"MuJoCo fitted-ball {label} closure is empty"
            )
    for label in (
        "base_mujoco_portable_identity_sha256",
        "base_mujoco_verification_receipt_sha256",
    ):
        _digest(receipt[label], f"MuJoCo fitted-ball {label}")
    runtime_attestation = receipt["runtime_code_identity"]
    external_preexec = (
        runtime_attestation.get("external_preexec")
        if isinstance(runtime_attestation, Mapping)
        else None
    )
    if (
        not isinstance(external_preexec, Mapping)
        or external_preexec.get("capsule_layout")
        != _FITTED_CAPSULE_LAYOUT
        or external_preexec.get("code_commit")
        != capsule.formal_receipt["runtime_code_identity"].get(
            "code_commit"
        )
        or not isinstance(external_preexec.get("checkout_root"), str)
        or not isinstance(
            external_preexec.get("capsule_staging_root"), str
        )
        or not isinstance(external_preexec.get("artifacts_root"), str)
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball execution capsule layout is invalid"
        )
    execution_checkout = Path(
        str(external_preexec["checkout_root"])
    )
    execution_staging = Path(
        str(external_preexec["capsule_staging_root"])
    )
    execution_artifacts = Path(
        str(external_preexec["artifacts_root"])
    )
    if (
        not execution_checkout.is_absolute()
        or execution_checkout.parent != execution_staging
        or execution_checkout.name != "checkout"
        or execution_artifacts.parent != execution_staging
        or execution_artifacts.name != "artifacts"
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball execution capsule paths are inconsistent"
        )

    def reopen_checkout_artifact(
        raw_path: Any, digest: Any, label: str
    ) -> bytes:
        if not isinstance(raw_path, str) or not raw_path:
            raise MotionAdmissionError(f"{label}.path is invalid")
        execution_path = Path(raw_path)
        if not execution_path.is_absolute():
            raise MotionAdmissionError(
                f"{label}.path was not an execution-checkout absolute path"
            )
        try:
            relative = execution_path.relative_to(execution_checkout)
        except ValueError as exc:
            raise MotionAdmissionError(
                f"{label}.path escaped the executed capsule checkout"
            ) from exc
        if any(part in ("", ".", "..") for part in relative.parts):
            raise MotionAdmissionError(
                f"{label}.path contains traversal"
            )
        retained_path = capsule.checkout.joinpath(*relative.parts)
        raw, _metadata = _read_plain_regular(retained_path, label)
        expected = _digest(digest, f"{label}.sha256")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise MotionAdmissionError(
                f"{label} bytes differ in retained checkout"
            )
        return raw

    fitted_artifacts: dict[str, Mapping[str, Any]] = {}
    for label in ("venue", "contact_model"):
        artifact = _exact_keys(
            receipt[label],
            _BANK_GATE_BINDING_KEYS,
            f"MuJoCo fitted-ball {label}",
        )
        reopen_checkout_artifact(
            artifact["path"],
            artifact["sha256"],
            f"MuJoCo fitted-ball {label}",
        )
        fitted_artifacts[label] = artifact

    runtime_snapshot = _exact_keys(
        receipt["runtime_input_snapshot"],
        frozenset(
            {
                "phase",
                "files",
                "post_runtime",
                "checkout_post_runtime",
            }
        ),
        "MuJoCo fitted-ball runtime_input_snapshot",
    )
    snapshot_files = runtime_snapshot["files"]
    if (
        runtime_snapshot["phase"] != "captured_before_runtime"
        or not isinstance(snapshot_files, list)
        or not snapshot_files
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball runtime input snapshot is incomplete"
        )
    training_manifest_rows: list[Mapping[str, Any]] = []
    physical_manifest_rows: list[Mapping[str, Any]] = []
    materialization_receipt_rows: list[Mapping[str, Any]] = []
    profile_rows: list[Mapping[str, Any]] = []
    snapshot_rows_by_role: dict[str, list[Mapping[str, Any]]] = {}
    for index, raw in enumerate(snapshot_files):
        row = _exact_keys(
            raw,
            frozenset({"path", "sha256", "size_bytes", "roles"}),
            f"MuJoCo fitted-ball runtime input files[{index}]",
        )
        _digest(
            row["sha256"],
            f"MuJoCo fitted-ball runtime input files[{index}].sha256",
        )
        roles = row["roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or roles != sorted(set(roles))
            or any(not isinstance(role, str) or not role for role in roles)
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] <= 0
        ):
            raise MotionAdmissionError(
                "MuJoCo fitted-ball runtime input row is malformed"
            )
        if "strict_training_manifest" in roles:
            training_manifest_rows.append(row)
        if "physical_gate_manifest" in roles:
            physical_manifest_rows.append(row)
        if "physical_gate_materialization_receipt" in roles:
            materialization_receipt_rows.append(row)
        if "profile_pins" in roles:
            profile_rows.append(row)
        for role in roles:
            snapshot_rows_by_role.setdefault(role, []).append(row)
    if (
        len(training_manifest_rows) != 1
        or len(physical_manifest_rows) != 1
        or len(materialization_receipt_rows) != 1
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball snapshot must bind exactly one strict "
            "manifest, physical overlay, and materialization receipt"
        )
    if len(profile_rows) != 1:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball snapshot must bind exactly one profile-pins file"
        )
    manifest_row = training_manifest_rows[0]
    manifest_raw = reopen_checkout_artifact(
        manifest_row["path"],
        manifest_row["sha256"],
        "MuJoCo fitted-ball strict ActionBall manifest",
    )
    if len(manifest_raw) != manifest_row["size_bytes"]:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball strict manifest size receipt is false"
        )
    manifest = _strict_json_bytes(
        manifest_raw, "MuJoCo fitted-ball strict ActionBall manifest"
    )
    manifest_id, manifest_action_uids, manifest_action_kinematics = (
        _validate_fresh_n5_manifest_identity(
            manifest,
            manifest_sha256=manifest_row["sha256"],
            binding=binding,
            repo_root=capsule.checkout,
            label="MuJoCo fitted-ball strict ActionBall manifest",
            reopen_motion_files=True,
            require_gate_geometry=False,
        )
    )
    physical_manifest_row = physical_manifest_rows[0]
    physical_manifest_raw = reopen_checkout_artifact(
        physical_manifest_row["path"],
        physical_manifest_row["sha256"],
        "MuJoCo fitted-ball physical-gate manifest",
    )
    if len(physical_manifest_raw) != physical_manifest_row["size_bytes"]:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball physical manifest size receipt is false"
        )
    physical_manifest = _strict_json_bytes(
        physical_manifest_raw,
        "MuJoCo fitted-ball physical-gate manifest",
    )
    (
        physical_manifest_id,
        physical_action_uids,
        physical_action_kinematics,
    ) = _validate_fresh_n5_manifest_identity(
        physical_manifest,
        manifest_sha256=physical_manifest_row["sha256"],
        binding=binding,
        repo_root=capsule.checkout,
        label="MuJoCo fitted-ball physical-gate manifest",
        reopen_motion_files=True,
    )
    if (
        physical_manifest_id != manifest_id
        or physical_action_uids != manifest_action_uids
        or physical_action_kinematics != manifest_action_kinematics
    ):
        raise MotionAdmissionError(
            "strict and physical fitted-ball manifests bind different "
            "action identities"
        )
    materialization_row = materialization_receipt_rows[0]
    materialization_raw = reopen_checkout_artifact(
        materialization_row["path"],
        materialization_row["sha256"],
        "MuJoCo fitted-ball physical materialization receipt",
    )
    if len(materialization_raw) != materialization_row["size_bytes"]:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball materialization receipt size is false"
        )
    profile_row = profile_rows[0]
    profile_raw = reopen_checkout_artifact(
        profile_row["path"],
        profile_row["sha256"],
        "MuJoCo fitted-ball ActionBall profile pins",
    )
    if len(profile_raw) != profile_row["size_bytes"]:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball profile-pins size receipt is false"
        )
    profile_pins = _strict_json_bytes(
        profile_raw, "MuJoCo fitted-ball ActionBall profile pins"
    )
    solver_payload = profile_pins.get("solver_payload")
    physics_payload = profile_pins.get("physics_payload")
    if not isinstance(solver_payload, Mapping) or not isinstance(
        physics_payload, Mapping
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball profile pins omit solver/physics payloads"
        )
    solver_profile_sha = hashlib.sha256(
        _canonical_json_bytes(solver_payload)
    ).hexdigest()
    physics_profile_sha = hashlib.sha256(
        _canonical_json_bytes(physics_payload)
    ).hexdigest()
    contact_geometry_sha = _solver_contact_geometry_sha256(
        profile_pins,
        solver_payload,
        label="MuJoCo fitted-ball profile pins",
    )
    physics_bounds = _fitted_physics_bounds(physics_payload)
    source_map = profile_pins.get(
        "solver_implementation_source_sha256"
    )
    payload_source_map = solver_payload.get(
        "implementation_source_sha256"
    )
    if (
        solver_profile_sha != manifest.get("solver_profile_sha256")
        or solver_profile_sha
        != physical_manifest.get("solver_profile_sha256")
        or solver_profile_sha
        != profile_pins.get("solver_profile_sha256")
        or physics_profile_sha != manifest.get("physics_profile_sha256")
        or physics_profile_sha
        != physical_manifest.get("physics_profile_sha256")
        or physics_profile_sha
        != profile_pins.get("physics_profile_sha256")
        or not isinstance(source_map, Mapping)
        or dict(source_map)
        != (
            dict(payload_source_map)
            if isinstance(payload_source_map, Mapping)
            else None
        )
        or frozenset(source_map) != _ACTION_BALL_SOLVER_SOURCE_NAMES
        or physical_manifest["racket_geometry_contract"][
            "geometry_source_sha256"
        ]
        != contact_geometry_sha
        or physical_manifest["racket_geometry_contract"]["source_sha256"]
        != source_map["racket_contact_geometry.py"]
        or contact_geometry_sha
        == source_map["racket_contact_geometry.py"]
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball frozen solver/physics profile identity "
            "does not close"
        )
    for name in sorted(_ACTION_BALL_SOLVER_SOURCE_NAMES):
        digest = _digest(
            source_map[name],
            f"MuJoCo fitted-ball solver source {name}",
        )
        rows = snapshot_rows_by_role.get(f"solver_source:{name}", [])
        expected_solver_path = (
            execution_checkout
            / "hope_training/whole_body_tracking/source/"
            "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
            / name
        )
        if (
            len(rows) != 1
            or rows[0]["sha256"] != digest
            or rows[0]["path"] != str(expected_solver_path)
        ):
            raise MotionAdmissionError(
                f"MuJoCo fitted-ball solver source {name} is not "
                "uniquely byte-bound"
            )
        reopen_checkout_artifact(
            rows[0]["path"],
            digest,
            f"MuJoCo fitted-ball solver source {name}",
        )
    post_runtime = _exact_keys(
        runtime_snapshot["post_runtime"],
        frozenset({"stable", "checked_files", "check"}),
        "MuJoCo fitted-ball post-runtime snapshot",
    )
    checkout_post_runtime = _exact_keys(
        runtime_snapshot["checkout_post_runtime"],
        frozenset({"commit", "clean"}),
        "MuJoCo fitted-ball post-runtime checkout",
    )
    code_commit = capsule.formal_receipt["runtime_code_identity"].get(
        "code_commit"
    )
    if (
        post_runtime
        != {
            "stable": True,
            "checked_files": len(snapshot_files),
            "check": "pinned_sha256_before_and_after_runtime",
        }
        or checkout_post_runtime
        != {"commit": code_commit, "clean": True}
        or not isinstance(code_commit, str)
        or _GIT_SHA.fullmatch(code_commit) is None
        or receipt["manifest_id"] != manifest_id
    ):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball manifest/checkout snapshot is stale"
        )

    actions = receipt["actions"]
    if not isinstance(actions, list) or len(actions) != 5:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball receipt must contain exactly five actions"
        )
    expected_video_names: set[str] = set()
    for index, (
        raw,
        motion_id,
        motion_sha,
        action_uid,
        manifest_kinematics,
    ) in enumerate(
        zip(
            actions,
            binding.motion_ids,
            binding.npz_sha256,
            manifest_action_uids,
            manifest_action_kinematics,
        )
    ):
        action = _exact_keys(
            raw,
            _FRESH_N5_FITTED_ACTION_KEYS,
            f"MuJoCo fitted-ball actions[{index}]",
        )
        action_t_hit = _finite(
            action["t_hit_s"],
            f"MuJoCo fitted-ball actions[{index}].t_hit_s",
            minimum=0.0,
        )
        action_t_cycle = _finite(
            action["t_cycle_s"],
            f"MuJoCo fitted-ball actions[{index}].t_cycle_s",
            minimum=0.0,
        )
        action_racket_speed = _finite(
            action["reference_racket_site_speed_mps"],
            (
                f"MuJoCo fitted-ball actions[{index}]"
                ".reference_racket_site_speed_mps"
            ),
            minimum=0.0,
        )
        if (
            action["action_id"] != motion_id
            or action["motion_sha256"] != motion_sha
            or type(action["action_uid"]) is not int
            or action["action_uid"] != action_uid
            or (
                action_t_hit,
                action_t_cycle,
                action_racket_speed,
            )
            != manifest_kinematics
            or action["verdict"] != "PASS"
            or action["failure_reasons"] != []
            or action_t_hit > action_t_cycle
            or action_racket_speed <= 0.0
            or _finite(
                action["shared_ready_joint_linf_rad"],
                f"MuJoCo fitted-ball actions[{index}] shared-ready error",
                minimum=0.0,
            )
            > 1.0e-6
            or _finite(
                action["recovery_joint_linf_rad"],
                f"MuJoCo fitted-ball actions[{index}] recovery error",
                minimum=0.0,
            )
            > 1.0e-6
        ):
            raise MotionAdmissionError(
                "MuJoCo fitted-ball action is incomplete, failed, stale, "
                "or does not recover to the shared ready"
            )
        reopen_checkout_artifact(
            action["motion_path"],
            motion_sha,
            f"MuJoCo fitted-ball actions[{index}] motion",
        )
        manifest_action = physical_manifest["actions"][index]
        launch = _exact_keys(
            action["launch"],
            frozenset({"source", "state_sha256", "source_receipt"}),
            f"MuJoCo fitted-ball actions[{index}].launch",
        )
        manifest_launch = manifest_action.get("physical_ball_launch")
        if (
            not isinstance(manifest_launch, Mapping)
            or launch["source"] != manifest_launch.get("source")
            or launch["state_sha256"]
            != manifest_launch.get("state_sha256")
            or not isinstance(launch["source_receipt"], Mapping)
            or not launch["source_receipt"]
        ):
            raise MotionAdmissionError(
                "MuJoCo fitted-ball physical launch does not bind the "
                "manifest center-ball state/source receipt"
            )
        _assert_no_runtime_self_authorization(
            launch["source_receipt"],
            f"MuJoCo fitted-ball actions[{index}].launch.source_receipt",
        )
        face_geometry = _exact_keys(
            action["face_geometry"],
            frozenset(
                {
                    "sign",
                    "mesh_path",
                    "mesh_sha256",
                    "outer_triangle_count",
                    "geometry_contract_sha256",
                }
            ),
            f"MuJoCo fitted-ball actions[{index}].face_geometry",
        )
        if (
            face_geometry["sign"]
            != manifest_action.get("mount_normal_sign")
            or face_geometry["geometry_contract_sha256"]
            != physical_manifest["racket_geometry_contract"]["source_sha256"]
            or _integer(
                face_geometry["outer_triangle_count"],
                (
                    f"MuJoCo fitted-ball actions[{index}]"
                    ".face_geometry.outer_triangle_count"
                ),
                minimum=1,
            )
            < 1
        ):
            raise MotionAdmissionError(
                "MuJoCo fitted-ball selected face/geometry identity drifted"
            )
        reopen_checkout_artifact(
            face_geometry["mesh_path"],
            face_geometry["mesh_sha256"],
            f"MuJoCo fitted-ball actions[{index}] selected face mesh",
        )
        _validate_fitted_physical_task_binding(
            action["physical_task_binding"],
            manifest_action=manifest_action,
            action_row=action,
            action_index=index,
            solver_profile_sha256=solver_profile_sha,
            physics_profile_sha256=physics_profile_sha,
            solver_source_sha256=source_map,
            physics_bounds=physics_bounds,
            contact_model_sha256=fitted_artifacts["contact_model"][
                "sha256"
            ],
            snapshot_rows_by_role=snapshot_rows_by_role,
            reopen_checkout_artifact=reopen_checkout_artifact,
        )
        dt_results = _exact_keys(
            action["dt_results"],
            frozenset({"0.0010", "0.0005"}),
            f"MuJoCo fitted-ball actions[{index}].dt_results",
        )
        for dt_name, result_raw in dt_results.items():
            result = result_raw
            if not isinstance(result, Mapping):
                raise MotionAdmissionError(
                    "MuJoCo fitted-ball dt result must be an object"
                )
            window = result.get("simulation_window")
            mandatory = result.get("mandatory_gates")
            if (
                result.get("verdict") != "PASS"
                or result.get("failure_reasons") != []
                or result.get("joint_limit_violation") is not None
                or result.get("fall") is not None
                or result.get("robot_obstacle_contacts") != []
                or result.get("self_contacts") != []
                or result.get("shadow_robot_obstacle_near_contacts") != []
                or result.get("shadow_self_near_contacts") != []
                or not isinstance(window, Mapping)
                or _integer(
                    window.get("physics_steps"),
                    (
                        f"MuJoCo fitted-ball actions[{index}]"
                        f".dt_results[{dt_name}].physics_steps"
                    ),
                    minimum=1,
                )
                < 1
                or _finite(
                    window.get("executed_end_time_s"),
                    "MuJoCo fitted-ball executed end time",
                    minimum=0.0,
                )
                < _finite(
                    window.get("required_ready_to_recovery_end_time_s"),
                    "MuJoCo fitted-ball required recovery time",
                    minimum=0.0,
                )
                or not isinstance(mandatory, Mapping)
                or mandatory.get(
                    "physical_ball_selected_face_return_and_first_landing"
                )
                is not True
                or mandatory.get(
                    "teacher_robot_and_racket_table_net_post_clearance"
                )
                is not True
            ):
                raise MotionAdmissionError(
                    "MuJoCo fitted-ball dt replay is not a complete physical "
                    "return and whole-cycle safety PASS"
                )
        expected_video_name = (
            f"{motion_id}_fitted_teacher_ball.mp4"
        )
        expected_video_names.add(expected_video_name)
        video = _exact_keys(
            action["video"],
            _FRESH_N5_FITTED_VIDEO_KEYS,
            f"MuJoCo fitted-ball actions[{index}].video",
        )
        expected_capsule_relative = (
            f"artifacts/videos/{expected_video_name}"
        )
        execution_video_path = execution_artifacts / "videos" / (
            expected_video_name
        )
        retained_video_path = capsule.root / "artifacts" / "videos" / (
            expected_video_name
        )
        video_raw, _video_metadata = _read_plain_regular(
            retained_video_path,
            f"MuJoCo fitted-ball actions[{index}] retained video",
        )
        if (
            video["status"] != "WRITTEN"
            or video["path"] != str(execution_video_path)
            or video["capsule_relative_path"]
            != expected_capsule_relative
            or _digest(
                video["sha256"],
                f"MuJoCo fitted-ball actions[{index}].video.sha256",
            )
            != hashlib.sha256(video_raw).hexdigest()
            or _integer(
                video["size_bytes"],
                f"MuJoCo fitted-ball actions[{index}].video.size_bytes",
                minimum=1,
            )
            != len(video_raw)
            or _integer(
                video["frames"],
                f"MuJoCo fitted-ball actions[{index}].video.frames",
                minimum=1,
            )
            != _integer(
                dt_results["0.0010"]["simulation_window"].get(
                    "expected_render_frames"
                ),
                (
                    f"MuJoCo fitted-ball actions[{index}]"
                    ".expected_render_frames"
                ),
                minimum=1,
            )
            or _finite(
                video["fps"],
                f"MuJoCo fitted-ball actions[{index}].video.fps",
                minimum=0.0,
            )
            <= 0.0
            or video["camera"] != "torso_follow"
            or video["evidence_role"]
            != "human_visualization_only_not_physical_or_analytic_grader"
        ):
            raise MotionAdmissionError(
                "MuJoCo fitted-ball video is missing, stale, partial, or "
                "not the exact retained torso-follow visualization"
            )
    retained_video_dir = _assert_plain_path_components(
        capsule.artifacts / "videos",
        "MuJoCo fitted-ball retained video directory",
    )
    observed_videos = tuple(
        sorted(
            path.name
            for path in retained_video_dir.iterdir()
            if path.suffix.lower() == ".mp4"
        )
    )
    if observed_videos != tuple(sorted(expected_video_names)):
        raise MotionAdmissionError(
            "MuJoCo fitted-ball retained capsule video set is not exact"
        )
    declared_payload_sha = _digest(
        receipt["receipt_payload_sha256"],
        "MuJoCo fitted-ball receipt_payload_sha256",
    )
    unsigned = dict(receipt)
    del unsigned["receipt_payload_sha256"]
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != declared_payload_sha:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball receipt payload seal is false"
        )
    final_payload, _ = _read_plain_regular(
        path, "MuJoCo fitted-ball receipt after validation"
    )
    if hashlib.sha256(final_payload).hexdigest() != receipt_sha:
        raise MotionAdmissionError(
            "MuJoCo fitted-ball receipt changed during validation"
        )
    return _FreshN5EvidenceIdentity(
        manifest_sha256=manifest_row["sha256"],
        manifest_id=manifest_id,
        action_uids=manifest_action_uids,
        profile_pins_sha256=profile_row["sha256"],
        solver_profile_sha256=solver_profile_sha,
        physics_profile_sha256=physics_profile_sha,
        action_set_contract_sha256=_digest(
            receipt["action_set_contract"].get("contract_sha256"),
            "MuJoCo fitted-ball action_set_contract.contract_sha256",
        ),
        action_ids=tuple(FRESH_N5_DOWNSTREAM_MOTION_IDS),
        geometry_source_sha256=physical_manifest[
            "racket_geometry_contract"
        ]["geometry_source_sha256"],
        code_commit=code_commit,
    )


def _validate_fresh_n5_isaac_table_smoke_receipt(
    binding_row: Any,
    *,
    binding: FreshN5BankPromotionBinding,
    repo_root: Path,
    expected_identity: _FreshN5EvidenceIdentity | None = None,
) -> None:
    row = _exact_keys(
        binding_row,
        _BANK_GATE_BINDING_KEYS,
        "isaac_table_filtered_smoke_receipt",
    )
    if row["sha256"] != binding.isaac_table_filtered_smoke_receipt_sha256:
        raise MotionAdmissionError(
            "Isaac table-filtered smoke receipt SHA is not crossbound"
        )
    path = _receipt_file(
        row,
        repo_root=repo_root,
        label="Isaac table-filtered smoke receipt",
    )
    payload, receipt_sha = _snapshot(
        path, "Isaac table-filtered smoke receipt"
    )
    receipt = _exact_keys(
        _strict_json_bytes(payload, "Isaac table-filtered smoke receipt"),
        _ISAAC_TABLE_SMOKE_KEYS,
        "Isaac table-filtered smoke receipt",
    )
    _assert_no_runtime_self_authorization(
        receipt, "Isaac table-filtered smoke receipt"
    )
    authorization = _exact_keys(
        receipt["authorization"],
        frozenset(
            {
                "training_authorized",
                "deployment_authorized",
                "hardware_authorized",
            }
        ),
        "Isaac table-filtered smoke authorization",
    )
    runtime = _exact_keys(
        receipt["runtime_contract"],
        _ISAAC_TABLE_SMOKE_RUNTIME_KEYS,
        "Isaac table-filtered smoke runtime_contract",
    )
    runtime_source = _exact_keys(
        runtime["runtime_source"],
        _BANK_GATE_BINDING_KEYS,
        "Isaac table-filtered runtime source",
    )
    _receipt_file(
        runtime_source,
        repo_root=repo_root,
        label="Isaac table-filtered runtime source",
        expected_repo_path=(
            "hope_training/whole_body_tracking/scripts/"
            "check_table_obstacle_scene.py"
        ),
    )
    manifest_binding = _exact_keys(
        receipt["manifest"],
        _BANK_GATE_BINDING_KEYS,
        "Isaac table-filtered manifest",
    )
    manifest_path = _receipt_file(
        manifest_binding,
        repo_root=repo_root,
        label="Isaac table-filtered manifest",
    )
    manifest_payload, _manifest_sha = _snapshot(
        manifest_path, "Isaac table-filtered manifest"
    )
    manifest = _strict_json_bytes(
        manifest_payload, "Isaac table-filtered manifest"
    )
    manifest_id, manifest_action_uids, _manifest_action_kinematics = (
        _validate_fresh_n5_manifest_identity(
            manifest,
            manifest_sha256=manifest_binding["sha256"],
            binding=binding,
            repo_root=repo_root,
            label="Isaac table-filtered manifest",
            reopen_motion_files=True,
            require_gate_geometry=False,
        )
    )
    action_set = _exact_keys(
        receipt["action_set_contract"],
        _ACTION_SET_CONTRACT_IDENTITY_KEYS,
        "Isaac table-filtered action_set_contract",
    )
    action_set_unsigned = dict(action_set)
    action_set_sha = _digest(
        action_set_unsigned.pop("contract_sha256"),
        "Isaac table-filtered action_set_contract.contract_sha256",
    )
    if (
        hashlib.sha256(
            _canonical_json_bytes(action_set_unsigned)
        ).hexdigest()
        != action_set_sha
        or action_set["schema_version"] != 1
        or action_set["kind"]
        != "whole_body_tracking.action_ball.action_set_contract"
        or action_set["profile_id"] != "fresh_upper_nomove_n5_v3"
        or action_set["expected_n"] != len(FRESH_N5_DOWNSTREAM_MOTION_IDS)
        or action_set["scope"] != "upper"
        or action_set["mobility_mode"] != "no_move"
        or action_set["ordered_action_ids"]
        != list(FRESH_N5_DOWNSTREAM_MOTION_IDS)
        or action_set["ordered_action_uids"]
        != list(manifest_action_uids)
        or action_set["manifest_path"] != manifest_binding["path"]
        or action_set["manifest_sha256"] != manifest_binding["sha256"]
    ):
        raise MotionAdmissionError(
            "Isaac table-filtered action-set/manifest identity does not close"
        )
    profile_contract = _exact_keys(
        receipt["profile_contract"],
        _ISAAC_TABLE_SMOKE_PROFILE_KEYS,
        "Isaac table-filtered profile_contract",
    )
    profile_pins_binding = _exact_keys(
        profile_contract["profile_pins"],
        _BANK_GATE_BINDING_KEYS,
        "Isaac table-filtered profile pins",
    )
    profile_pins_path = _receipt_file(
        profile_pins_binding,
        repo_root=repo_root,
        label="Isaac table-filtered profile pins",
    )
    profile_pins_raw, _profile_pins_sha = _snapshot(
        profile_pins_path, "Isaac table-filtered profile pins"
    )
    profile_pins = _strict_json_bytes(
        profile_pins_raw, "Isaac table-filtered profile pins"
    )
    solver_payload = profile_pins.get("solver_payload")
    physics_payload = profile_pins.get("physics_payload")
    if not isinstance(solver_payload, Mapping) or not isinstance(
        physics_payload, Mapping
    ):
        raise MotionAdmissionError(
            "Isaac table-filtered profile pins omit solver/physics payloads"
        )
    solver_profile_sha = hashlib.sha256(
        _canonical_json_bytes(solver_payload)
    ).hexdigest()
    physics_profile_sha = hashlib.sha256(
        _canonical_json_bytes(physics_payload)
    ).hexdigest()
    contact_geometry_sha = _solver_contact_geometry_sha256(
        profile_pins,
        solver_payload,
        label="Isaac table-filtered profile pins",
    )
    source_map = profile_pins.get(
        "solver_implementation_source_sha256"
    )
    payload_source_map = solver_payload.get(
        "implementation_source_sha256"
    )
    if (
        profile_contract["solver_profile_sha256"]
        != solver_profile_sha
        or profile_contract["physics_profile_sha256"]
        != physics_profile_sha
        or profile_pins.get("solver_profile_sha256")
        != solver_profile_sha
        or profile_pins.get("physics_profile_sha256")
        != physics_profile_sha
        or manifest.get("solver_profile_sha256")
        != solver_profile_sha
        or manifest.get("physics_profile_sha256")
        != physics_profile_sha
        or not isinstance(source_map, Mapping)
        or not isinstance(payload_source_map, Mapping)
        or dict(source_map) != dict(payload_source_map)
        or frozenset(source_map) != _ACTION_BALL_SOLVER_SOURCE_NAMES
        or contact_geometry_sha
        == source_map["racket_contact_geometry.py"]
    ):
        raise MotionAdmissionError(
            "Isaac table-filtered solver/physics profile identity does not "
            "close"
        )
    source_rows = profile_contract["solver_implementation_sources"]
    expected_source_names = tuple(
        sorted(_ACTION_BALL_SOLVER_SOURCE_NAMES)
    )
    if (
        not isinstance(source_rows, list)
        or len(source_rows) != len(expected_source_names)
    ):
        raise MotionAdmissionError(
            "Isaac table-filtered solver source closure is incomplete"
        )
    observed_source_rows: dict[str, Mapping[str, Any]] = {}
    solver_source_base = (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp"
    )
    for index, (raw_source, expected_name) in enumerate(
        zip(source_rows, expected_source_names)
    ):
        source = _exact_keys(
            raw_source,
            _ISAAC_TABLE_SMOKE_SOLVER_SOURCE_KEYS,
            (
                "Isaac table-filtered solver_implementation_sources"
                f"[{index}]"
            ),
        )
        if (
            source["name"] != expected_name
            or source["sha256"] != source_map[expected_name]
        ):
            raise MotionAdmissionError(
                "Isaac table-filtered solver source order/hash changed"
            )
        _receipt_file(
            source,
            repo_root=repo_root,
            label=f"Isaac table-filtered solver source {expected_name}",
            expected_repo_path=f"{solver_source_base}/{expected_name}",
        )
        observed_source_rows[expected_name] = source
    geometry = _exact_keys(
        profile_contract["racket_geometry_contract"],
        _RACKET_GEOMETRY_CONTRACT_KEYS,
        "Isaac table-filtered racket_geometry_contract",
    )
    geometry_source = observed_source_rows[
        "racket_contact_geometry.py"
    ]
    if (
        geometry["source_path"] != geometry_source["path"]
        or geometry["source_sha256"] != geometry_source["sha256"]
        or geometry["geometry_source_sha256"] != contact_geometry_sha
        or geometry["geometry_source_sha256"]
        == geometry["source_sha256"]
    ):
        raise MotionAdmissionError(
            "Isaac table-filtered physical racket geometry bytes/semantics "
            "do not close"
        )
    if expected_identity is not None and (
        manifest_binding["sha256"] != expected_identity.manifest_sha256
        or manifest_id != expected_identity.manifest_id
        or tuple(action_set["ordered_action_ids"])
        != expected_identity.action_ids
        or manifest_action_uids != expected_identity.action_uids
        or action_set_sha
        != expected_identity.action_set_contract_sha256
        or profile_pins_binding["sha256"]
        != expected_identity.profile_pins_sha256
        or solver_profile_sha
        != expected_identity.solver_profile_sha256
        or physics_profile_sha
        != expected_identity.physics_profile_sha256
        or geometry["geometry_source_sha256"]
        != expected_identity.geometry_source_sha256
        or runtime["source_commit_sha"] != expected_identity.code_commit
    ):
        raise MotionAdmissionError(
            "Isaac and MuJoCo evidence do not bind the same manifest, "
            "action UIDs, and code commit"
        )
    gpu = _exact_keys(
        runtime["gpu_identity"],
        _ISAAC_TABLE_SMOKE_GPU_KEYS,
        "Isaac table-filtered GPU identity",
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 2
        or receipt["receipt_class"]
        != "isaac_action_ball_table_filtered_smoke_v2"
        or receipt["verdict"] != "PASS"
        or receipt["task_id"]
        != ACTION_BALL_ISAAC_TASK_ID
        or receipt["with_table"] is not True
        or receipt["scope"] != "upper"
        or receipt["mobility_mode"] != "no_move"
        or receipt["ordered_action_ids"]
        != list(FRESH_N5_DOWNSTREAM_MOTION_IDS)
        or dict(receipt["action_set_contract"]) != dict(action_set)
        or receipt["motion_sha256"] != list(binding.npz_sha256)
        or dict(authorization)
        != {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        }
        or not isinstance(runtime["source_commit_sha"], str)
        or _GIT_SHA.fullmatch(runtime["source_commit_sha"]) is None
        or not isinstance(runtime["isaac_version"], str)
        or not runtime["isaac_version"]
        or not isinstance(runtime["python_executable"], str)
        or not runtime["python_executable"]
        or _integer(
            gpu["physical_index"],
            "Isaac table-filtered physical GPU index",
            minimum=0,
        )
        < 0
        or type(gpu["logical_index"]) is not int
        or gpu["logical_index"] != 0
        or gpu["cuda_visible_devices"] != str(gpu["physical_index"])
        or not isinstance(gpu["gpu_uuid"], str)
        or not gpu["gpu_uuid"].startswith("GPU-")
        or len(gpu["gpu_uuid"]) <= 4
        or not isinstance(gpu["gpu_name"], str)
        or not gpu["gpu_name"]
        or not isinstance(gpu["driver_version"], str)
        or not gpu["driver_version"]
        or gpu["nvml_verified"] is not True
        or _integer(
            runtime["physics_steps"],
            "Isaac table-filtered physics_steps",
            minimum=1,
        )
        < 1
        or _integer(
            runtime["action_body_pair_filter_rows"],
            "Isaac table-filtered action body-pair filter rows",
            minimum=1,
        )
        != 32 * len(FRESH_N5_DOWNSTREAM_MOTION_IDS)
        or any(
            runtime[key] is not True
            for key in (
                "real_physx_contacts",
                "full_action_ball_assembly",
                "all_32_body_pair_filters",
                "all_five_obstacles",
                "all_four_substeps",
                "positive_control_pass",
                "negative_control_pass",
                "zero_reset_leakage",
            )
        )
    ):
        raise MotionAdmissionError(
            "Isaac table-filtered receipt is not an exact stepped fresh-N5 "
            "ActionBall PASS"
        )
    actions = receipt["actions"]
    if not isinstance(actions, list) or len(actions) != 5:
        raise MotionAdmissionError(
            "Isaac table-filtered receipt must cover exactly five actions"
        )
    for index, (raw, motion_id, motion_sha, action_uid) in enumerate(
        zip(
            actions,
            binding.motion_ids,
            binding.npz_sha256,
            manifest_action_uids,
        )
    ):
        action = _exact_keys(
            raw,
            _ISAAC_TABLE_SMOKE_ACTION_KEYS,
            f"Isaac table-filtered actions[{index}]",
        )
        if dict(action) != {
            "motion_id": motion_id,
            "action_uid": action_uid,
            "scope": "upper",
            "body_pair_filter_count": 32,
            "motion_sha256": motion_sha,
            "complete_cycle": True,
            "isaac_filtered_contact_pass": True,
            "table_contact_count": 0,
            "fall_count": 0,
            "hard_limit_count": 0,
            "unsafe_count": 0,
            "verdict": "PASS",
        }:
            raise MotionAdmissionError(
                "Isaac table-filtered action is partial, unsafe, or binds "
                "different upper motion bytes"
            )
    non_claims = receipt["non_claims"]
    if (
        not isinstance(non_claims, list)
        or any(not isinstance(item, str) or not item for item in non_claims)
        or not {
            "training_authorization",
            "deployment_authorization",
            "hardware_authorization",
        }.issubset(set(non_claims))
    ):
        raise MotionAdmissionError(
            "Isaac table-filtered smoke non-claims are incomplete"
        )
    declared_payload_sha = _digest(
        receipt["receipt_payload_sha256"],
        "Isaac table-filtered smoke receipt_payload_sha256",
    )
    unsigned = dict(receipt)
    del unsigned["receipt_payload_sha256"]
    if (
        hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        != declared_payload_sha
    ):
        raise MotionAdmissionError(
            "Isaac table-filtered smoke receipt payload seal is false"
        )
    _, final_sha = _snapshot(
        path, "Isaac table-filtered smoke receipt after validation"
    )
    if final_sha != receipt_sha:
        raise MotionAdmissionError(
            "Isaac table-filtered smoke receipt changed during validation"
        )


def _verify_bank_promotion_certificate_snapshot(
    certificate_path: os.PathLike[str] | str,
    *,
    binding: Any,
    repo_root: os.PathLike[str] | str,
) -> tuple[Path, Path, str]:
    """Reopen and verify the complete certificate/report closure."""

    path = Path(certificate_path).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise MotionAdmissionError(
            f"cannot resolve promotion certificate {path}: {exc}"
        ) from exc
    if not path.is_file():
        raise MotionAdmissionError("promotion certificate is not a regular file")
    payload, certificate_sha = _snapshot(path, "promotion certificate")
    if certificate_sha not in TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256:
        raise MotionAdmissionError(
            "promotion certificate SHA-256 is absent from the code trust set"
        )
    certificate = _exact_keys(
        _strict_json_bytes(payload, "promotion certificate"),
        (
            _FRESH_N5_CERTIFICATE_KEYS
            if type(binding) is FreshN5BankPromotionBinding
            else _CERTIFICATE_KEYS
        ),
        "promotion certificate",
    )
    _assert_no_runtime_self_authorization(
        certificate, "promotion certificate evidence"
    )
    (
        expected_schema_version,
        expected_certificate_type,
        expected_clip_count,
    ) = _certificate_profile(binding)
    if (
        type(certificate["schema_version"]) is not int
        or certificate["schema_version"] != expected_schema_version
        or certificate["certificate_type"] != expected_certificate_type
    ):
        raise MotionAdmissionError(
            "promotion certificate schema/type is unsupported"
        )
    expected = _binding_document(binding)
    for key, value in expected.items():
        if certificate[key] != value:
            raise MotionAdmissionError(
                f"promotion certificate does not crossbind {key}"
            )
    evidence = certificate["evidence_receipts"]
    if not isinstance(evidence, list):
        raise MotionAdmissionError(
            "promotion certificate evidence_receipts must be a list"
        )
    for index, row in enumerate(evidence):
        _exact_keys(
            row,
            _EVIDENCE_RECEIPT_KEYS,
            f"promotion certificate evidence_receipts[{index}]",
        )
    if type(binding) is FreshN5BankPromotionBinding:
        _validate_fresh_n5_bank_closure(
            certificate,
            binding=binding,
            repo_root=Path(repo_root),
        )
        fitted_identity = _validate_fresh_n5_fitted_ball_receipt(
            certificate["mujoco_fitted_ball_receipt"],
            binding=binding,
            repo_root=Path(repo_root),
        )
        _validate_fresh_n5_isaac_table_smoke_receipt(
            certificate["isaac_table_filtered_smoke_receipt"],
            binding=binding,
            repo_root=Path(repo_root),
            expected_identity=fitted_identity,
        )
    else:
        _validate_bank_gate_report(
            certificate["bank_gate_report"],
            binding=binding,
            repo_root=Path(repo_root),
            expected_report_schema_version=(
                _GENERIC_BANK_GATE_REPORT_SCHEMA_VERSION
                if type(binding) is GenericBankPromotionBinding
                else expected_schema_version
            ),
            expected_clip_count=expected_clip_count,
            report_profile=(
                "generic_v2"
                if type(binding) is GenericBankPromotionBinding
                else "legacy"
            ),
        )
    try:
        resolved_root = Path(repo_root).resolve(strict=True)
    except OSError as exc:
        raise MotionAdmissionError(
            f"cannot resolve repository root {repo_root}: {exc}"
        ) from exc
    return path, resolved_root, certificate_sha


def verify_bank_promotion_certificate(
    certificate_path: os.PathLike[str] | str,
    *,
    binding: Any,
    repo_root: os.PathLike[str] | str,
) -> TrustedMotionAdmission:
    """Verify exact trusted bytes and mint one opaque bank capability."""

    path, resolved_root, certificate_sha = (
        _verify_bank_promotion_certificate_snapshot(
            certificate_path,
            binding=binding,
            repo_root=repo_root,
        )
    )
    return TrustedMotionAdmission(
        _token=_MINT_TOKEN,
        certificate_sha256=certificate_sha,
        binding_sha256=_binding_sha256(binding),
        purpose=binding.purpose,
        bank_id=binding.bank_id,
        scope=binding.scope,
        certificate_path=str(path),
        repo_root=str(resolved_root),
    )


def require_matching_admission(
    admission: Any, binding: Any
) -> None:
    """Reject missing, fabricated, stale, or wrong-purpose capabilities."""

    if type(admission) is not TrustedMotionAdmission:
        raise MotionAdmissionError(
            "runtime purpose requires one opaque TrustedMotionAdmission"
        )
    try:
        path, resolved_root, certificate_sha = (
            _verify_bank_promotion_certificate_snapshot(
                admission._certificate_path,
                binding=binding,
                repo_root=admission._repo_root,
            )
        )
        required = (
            admission._binding_sha256 == _binding_sha256(binding)
            and admission._purpose == binding.purpose
            and admission._bank_id == binding.bank_id
            and admission._scope == binding.scope
            and admission._certificate_sha256 == certificate_sha
            and admission._certificate_path == str(path)
            and admission._repo_root == str(resolved_root)
        )
    except (AttributeError, MotionAdmissionError) as exc:
        raise MotionAdmissionError(
            "TrustedMotionAdmission cannot revalidate its trusted certificate"
        ) from exc
    if not required:
        raise MotionAdmissionError(
            "TrustedMotionAdmission does not match this exact registry binding"
        )


__all__ = [
    "ACTION_BALL_ISAAC_TASK_ID",
    "BankPromotionBinding",
    "FreshN5BankPromotionBinding",
    "FRESH_N5_APPEND_MOTION_IDS",
    "FRESH_N5_BANK_MOTION_IDS",
    "FRESH_N5_BASE_MOTION_IDS",
    "FRESH_N5_DOWNSTREAM_MOTION_IDS",
    "FRESH_N5_FORBIDDEN_MOTION_IDS",
    "GenericBankPromotionBinding",
    "MotionAdmissionError",
    "TrustedMotionAdmission",
    "require_matching_admission",
    "verify_bank_promotion_certificate",
]
