"""Fail-closed tests for the commit-required A211 lineage producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_action_ball_a211_lineage.py"
SPEC = importlib.util.spec_from_file_location("materialize_a211_lineage", SCRIPT)
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    raw = materializer.canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha(raw)


def _sealed(value: dict) -> dict:
    return {**value, "content_sha256": materializer.canonical_sha256(value)}


def _zero_handoff_artifact(motion_sha: str = "a" * 64) -> dict:
    joint_pos = [float(index) / 100.0 for index in range(31)]
    root_pos = [0.0, 0.0, 0.9]
    # The tracked motion endpoint is preserved byte-for-byte even when its
    # stored quaternion needs normalization for the MuJoCo numerical audit.
    root_quat = [1.000001, 0.0, 0.0, 0.0]
    state_sha = materializer._whole_body_state_sha256(
        joint_pos, root_pos, root_quat
    )
    root_norm = float(np.linalg.norm(np.asarray(root_quat, np.float64)))
    audit_quat = (np.asarray(root_quat, np.float64) / root_norm).tolist()
    audit_state_sha = materializer._whole_body_state_sha256(
        joint_pos, root_pos, audit_quat
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
        "stored_root_quaternion_norm": root_norm,
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
        "required_followup_hold_gate": materializer._L.FRAME0_LIVE_RECEIPT_KIND,
        "required_followup_policy_steps": (
            materializer._L.PHYSICAL_READY_HOLD_POLICY_STEPS
        ),
        "required_followup_physics_steps": (
            materializer._L.PHYSICAL_READY_HOLD_PHYSICS_STEPS
        ),
        "diagnostic_unauthorized": True,
        "training_authorized": False,
    }
    robust = {"table_clearance_m": 0.01, "ground_lp_margin": 0.1}
    racket = {
        "authority": "independent_schema_v4_measured_racket_channel",
        "motion_sha256": motion_sha,
        "frame_index": 0,
    }
    return {
        "teacher_reference": {
            "semantics": "exact_motion_bytes_frame0_reference",
            "motion_sha256": motion_sha,
            "frame_index": 0,
            "joint_pos_rad": list(joint_pos),
            "root_pos_w_m": list(root_pos),
            "root_quat_wxyz": list(root_quat),
            "static_handoff_joint_vel_radps": [0.0] * 31,
            "static_handoff_velocity_semantics": (
                "constructed_zero_joint_velocity_endpoint_not_measured_motion_velocity"
            ),
        },
        "physical_ready": {
            "joint_pos_rad": list(joint_pos),
            "joint_vel_radps": [0.0] * 31,
            "root_pos_w_m": list(root_pos),
            "root_quat_wxyz": list(root_quat),
        },
        "frame0_handoff": dict(handoff),
        "physical_birth_composition": {
            "semantics": (
                "measured_frame0_direct_if_safe_else_lexicographic_whole_body_safe_ready"
            ),
            "teacher_reference_unchanged": True,
            "historical_physical_birth_seed_consumed": False,
            "selection_priority": [
                "exact_measured_frame0_if_all_safety_gates_pass",
                "lexicographic_whole_body_safe_ready_only_if_frame0_unsafe",
            ],
            "exact_measured_frame0_selected": True,
            "teacher_and_physical_birth_differ": False,
            "changed_joint_mask": [False] * 31,
            "changed_joint_indices": [],
            "changed_joint_names": [],
            "physical_minus_teacher_joint_pos_rad": [0.0] * 31,
            "physical_minus_teacher_root_pos_m": [0.0] * 3,
            "physical_minus_teacher_root_rotation_vector_rad": [0.0] * 3,
            "teacher_root_quat_wxyz": list(root_quat),
            "physical_root_quat_wxyz": list(root_quat),
            "stored_physical_root_quat_wxyz": list(root_quat),
            "mjcf_audit_root_quat_wxyz": list(audit_quat),
            "frame0_handoff": dict(handoff),
        },
        "physical_birth_static_evidence": {
            "authority": "fresh_current_exact_mjcf_whole_body_lexicographic_search",
            "selected_hold_witness_authority": (
                "new_backend_new_solver_final_state_cache_miss"
            ),
            "exact_contact_lp_reused": False,
            "fresh_direct_robust_gate_passed": True,
            "all_safety_slacks_meet_original_and_locked_gate": True,
            "geometry_passed": True,
            "ground_dynamics_passed": True,
            "stored_endpoint_state_sha256": state_sha,
            "mjcf_audit_state_sha256": audit_state_sha,
            "stored_root_quat_wxyz": list(root_quat),
            "mjcf_audit_root_quat_wxyz": list(audit_quat),
            "stored_root_quaternion_norm": root_norm,
            "direct_frame0_robust_minimum_slacks": dict(robust),
            "direct_frame0_robust_gate_sha256": materializer.canonical_sha256(
                robust
            ),
            "evaluator_evidence": {
                "lp_feasible": True,
                "exact_state_lp_cache_hit": False,
                "evaluated_state_sha256": audit_state_sha,
                "required_minimum_normal_force_per_contact_n": 0.1,
                "required_minimum_normal_force_per_foot_n": 1.0,
            },
            "independent_measured_racket_frame0": dict(racket),
            "frame0_handoff": dict(handoff),
        },
    }


def test_exact_zero_handoff_accepts_only_the_fully_sealed_equal_endpoint() -> None:
    artifact = _zero_handoff_artifact()
    assert materializer._exact_zero_handoff_semantics(
        artifact, motion_sha256="a" * 64
    ) is True

    tampered = json.loads(json.dumps(artifact))
    tampered["physical_ready"]["joint_pos_rad"][0] += 0.01
    with pytest.raises(materializer.MaterializationError, match="zero-handoff"):
        materializer._exact_zero_handoff_semantics(
            tampered, motion_sha256="a" * 64
        )

    legacy = json.loads(json.dumps(artifact))
    legacy.pop("frame0_handoff")
    assert materializer._exact_zero_handoff_semantics(
        legacy, motion_sha256="a" * 64
    ) is False


def _live_safety(action_id: str, motion_sha: str, ticks: int) -> dict:
    names = ["joint_%02d" % index for index in range(31)]
    joint = {
        "schema_version": 1, "complete": True, "joint_order": names,
        "current_actual_hard_edge_joint_count": 0,
        "current_actual_hard_edge_joint_names": [],
        "substep_actual_hard_edge_joint_count": 0,
        "substep_actual_hard_edge_joint_names": [],
        "final_minimum_hard_gap_rad": 0.05,
        "preterminal_joint_pos_rad": [0.0] * 31,
        "preterminal_joint_vel_radps": [0.0] * 31,
        "final_joint_pos_rad": [0.0] * 31,
        "final_joint_vel_radps": [0.0] * 31,
        "hard_lower_rad": [-1.0] * 31,
        "hard_upper_rad": [1.0] * 31,
    }
    unsigned = {
        "schema_version": 1,
        "kind": materializer._L.FRAME0_LIVE_RECEIPT_KIND,
        "verdict": "PASS", "action_id": action_id,
        "motion_sha256": motion_sha,
        "teacher_reference_unchanged": True,
        "teacher_physical_birth_separated": False,
        "candidate_physical_birth_written": True,
        "candidate_hold_qdes_and_delay_history_installed": True,
        "plant_contract_match": True,
        "active_terminations": list(materializer._L.HARD_TERMINATION_UNION),
        "requested_duration_s": ticks * materializer._L.POLICY_DT_S,
        "completed_duration_s": ticks * materializer._L.POLICY_DT_S,
        "completed_policy_steps": ticks, "completed_physics_steps": ticks * 4,
        "terminal_reasons": [], "generic_terminated": False,
        "generic_truncated": False, "minimum_root_z_m": 0.9,
        "maximum_root_tilt_rad": 0.1, "both_feet_contact_fraction": 1.0,
        "joint_safety_telemetry": joint,
        "screenshots": [
            {"label": label, "sha256": ("%x" % (index + 1)) * 64}
            for index, label in enumerate((
                "raw_env_reset", "physical_ready_after_reset_write",
                "after_step_1", "after_step_10", "final",
            ))
        ],
    }
    return _sealed(unsigned)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _write_motion(path: Path) -> dict[str, list[float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_pos = np.asarray(
        [
            [[-0.125, 0.375, 0.8125], [1.0, 2.0, 3.0]],
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        ],
        dtype=np.float32,
    )
    root_quat = np.asarray(
        [
            [[0.5, 0.5, -0.5, 0.5], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    joint_pos = (
        np.arange(62, dtype=np.float32).reshape(2, 31) / np.float32(17.0)
    )
    np.savez(
        path,
        body_names=np.asarray(["pelvis_link", "torso_link"]),
        body_pos_w=root_pos,
        body_quat_w=root_quat,
        joint_pos=joint_pos,
    )
    return {
        "root_pos_w_m": root_pos[0, 0].tolist(),
        "root_quat_wxyz": root_quat[0, 0].tolist(),
        "root_lin_vel_w_mps": [0.0, 0.0, 0.0],
        "root_ang_vel_w_radps": [0.0, 0.0, 0.0],
        "joint_pos_rad": joint_pos[0].tolist(),
        "joint_vel_radps": [0.0] * 31,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / ".gitignore").write_text("vendor_assets/\n", encoding="utf-8")
    authority_root = Path(__file__).resolve().parents[3]
    authority_sources = {
        "motion": (
            "assets/motions/chingmu73_measured_v4_20260803/"
            "hope_Take_061_unit04_BH.npz"
        ),
        "dynamic": (
            "configs/action_ball_n1_measured_20260803/"
            "evidence_holdpass_robust20n_20260803/"
            "take061.measured_teacher.yaw_aligned_full_seed.robust20n."
            "dynamic_ready.v2.json"
        ),
        "hold": (
            "configs/action_ball_n1_measured_20260803/"
            "evidence_holdpass_robust20n_20260803/"
            "take061.robust20n.nominal_hold.v1.json"
        ),
        "teacher_frame0": (
            "configs/action_ball_n1_measured_20260803/"
            "a211_frame0_exact_20260803/"
            "take_061_unit04_bh.frame0_exact.v1.json"
        ),
    }
    tracked = [".gitignore"]
    pins: dict[str, str] = {}
    for key, relative in authority_sources.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(authority_root / relative, destination)
        pins[key] = _sha(destination.read_bytes())
        tracked.append(relative)

    manifest_relative = "configs/action_manifest.json"
    manifest_source = authority_root / (
        "configs/action_ball_n1_measured_20260803/"
        "fresh_core_seed0_20260803_take061_robust20n_r8_splitready/"
        "take_061_unit04_bh.full.manifest.v3.7d2139028427.json"
    )
    shutil.copyfile(manifest_source, root / manifest_relative)
    pins["manifest"] = _sha((root / manifest_relative).read_bytes())
    tracked.append(manifest_relative)

    receipt_source = authority_root / (
        "configs/action_ball_n1_measured_20260803/"
        "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/"
        "current_lm.target.task_receipt.v5.f64f52137ad8.json"
    )
    receipt = json.loads(receipt_source.read_text(encoding="utf-8"))
    receipt.pop("canonical_sha256")
    receipt.update(
        {
            "sampling_stratum": "center",
            "birth_sampling_stratum": "center",
            "frontier_arm": None,
            "birth_frontier_arm": None,
            "time_to_contact_tick": 91,
            "time_to_contact_s": 1.82,
            "pre_swing_wait_s": 1.82 - receipt["scaled_t_hit_s"],
            "manifest_sha256": pins["manifest"],
        }
    )
    receipt["canonical_sha256"] = materializer.canonical_sha256(receipt)
    receipt_relative = "configs/initial_center_a.task_receipt.v5.json"
    pins["initial_center_receipt"] = _write_json(
        root / receipt_relative, receipt
    )
    tracked.append(receipt_relative)

    for relative in (
        materializer._L.TRAINING_CONTRACT_SOURCE,
        materializer._L.TASK_PROFILE_SOURCE,
        materializer._L.RETAINED_TASK_PROFILE_PARENT_SOURCE,
        materializer._L.DR_L0_MANIFEST_SOURCE,
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(authority_root / relative, destination)
        tracked.append(relative)
    _git(root, "add", *tracked)
    _git(root, "commit", "-m", "track split-ready DR-L0 authorities")
    return root, pins, _git(root, "rev-parse", "HEAD")

    motion_path = root / "assets/motion.npz"
    frame0 = _write_motion(motion_path)
    motion_sha = _sha(motion_path.read_bytes())
    action_id = "take_061_unit04_bh"
    joint_names = ["joint_%02d" % index for index in range(31)]
    timing_unsigned = {
        "schema_version": 5,
        "kind": "action_ball_n1_task_receipt_v5",
        "contact_time_step_s": materializer._L.POLICY_DT_S,
        "pre_swing_wait_s": 0.7123799138976297,
    }
    timing = {
        **timing_unsigned,
        "canonical_sha256": materializer.canonical_sha256(timing_unsigned),
    }
    timing_path = root / "configs/timing_receipt.json"
    timing_sha = _write_json(timing_path, timing)
    birth_horizon = {
        "schema_version": 1,
        "kind": "action_ball_frame0_dynamic_birth_horizon_v1",
        "derivation": (
            "post_reset_coverage_plus_max_reset_wait_plus_ceil_pre_swing_wait"
        ),
        "timing_receipt_canonical_sha256": timing["canonical_sha256"],
        "policy_dt_s": materializer._L.POLICY_DT_S,
        "post_reset_coverage_policy_ticks": 1,
        "max_reset_wait_policy_ticks": 25,
        "pre_swing_wait_s": timing["pre_swing_wait_s"],
        "pre_swing_wait_policy_ticks_ceil": 36,
        "required_policy_ticks": 62,
    }
    physical_joint = [value + 0.25 for value in frame0["joint_pos_rad"]]
    dynamic = _sealed(
        {
            "schema_version": 1,
            "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
            "action_id": action_id,
            "robot": {"joint_names": joint_names},
            "runtime_plant": {"control_decimation": 4},
            "teacher_reference": {
                "motion_sha256": motion_sha,
                "root_pos_w_m": frame0["root_pos_w_m"],
                "root_quat_wxyz": frame0["root_quat_wxyz"],
                "joint_pos_rad": frame0["joint_pos_rad"],
            },
            "physical_ready": {
                "root_pos_w_m": [0.0, 0.0, 1.0],
                "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "joint_pos_rad": physical_joint,
                "joint_vel_radps": [0.0] * 31,
            },
        }
    )
    hold = _sealed(
        {
            "schema_version": 1,
            "kind": materializer._L.FRAME0_LIVE_RECEIPT_KIND,
            "verdict": "PASS",
            "action_id": action_id,
            "motion_sha256": motion_sha,
            "artifact": {
                "sha256": "0" * 64,
                "content_sha256": dynamic["content_sha256"],
            },
            "teacher_reference_unchanged": True,
            "teacher_physical_birth_separated": True,
            "candidate_physical_birth_written": True,
            "candidate_hold_qdes_and_delay_history_installed": True,
            "plant_contract_match": True,
            "active_terminations": list(materializer._L.HARD_TERMINATION_UNION),
            "requested_duration_s": 4.0,
            "completed_duration_s": 4.0,
            "completed_policy_steps": 200,
            "completed_physics_steps": 800,
            "terminal_reasons": [],
            "generic_terminated": False,
            "generic_truncated": False,
            "minimum_root_z_m": 1.0,
            "maximum_root_tilt_rad": 0.02,
            "both_feet_contact_fraction": 1.0,
            "joint_safety_telemetry": {
                "schema_version": 1,
                "complete": True,
                "joint_order": joint_names,
                "current_actual_hard_edge_joint_count": 0,
                "current_actual_hard_edge_joint_names": [],
                "substep_actual_hard_edge_joint_count": 0,
                "substep_actual_hard_edge_joint_names": [],
                "final_minimum_hard_gap_rad": 0.1,
                "preterminal_joint_pos_rad": [0.0] * 31,
                "preterminal_joint_vel_radps": [0.0] * 31,
                "final_joint_pos_rad": [0.0] * 31,
                "final_joint_vel_radps": [0.0] * 31,
                "hard_lower_rad": [-1.0] * 31,
                "hard_upper_rad": [1.0] * 31,
            },
        }
    )
    dynamic_sha = _write_json(root / "configs/dynamic.json", dynamic)
    hold_unsigned = dict(hold)
    hold_unsigned.pop("content_sha256")
    hold_unsigned["artifact"]["sha256"] = dynamic_sha
    hold = _sealed(hold_unsigned)
    hold_sha = _write_json(root / "configs/hold.json", hold)
    frame0_artifact = _sealed(
        {
            "schema_version": 2,
            "kind": materializer._L.FRAME0_EXACT_ARTIFACT_KIND,
            "diagnostic_unauthorized": True,
            "source_kind": materializer._L.FRAME0_EXACT_SOURCE_KIND,
            "action_id": action_id,
            "motion_sha256": motion_sha,
            "policy_dt_s": materializer._L.POLICY_DT_S,
            "wait_schedule_canonical_sha256": materializer._L.WAIT_SCHEDULE[
                "canonical_sha256"
            ],
            "timing_receipt": {
                "path": "configs/timing_receipt.json",
                "sha256": timing_sha,
            },
            "birth_horizon": birth_horizon,
            "frame0": frame0,
        }
    )
    frame0_artifact_sha = _write_json(
        root / "configs/frame0_exact_artifact.json", frame0_artifact
    )
    for source_path in materializer._L.FRAME0_RECEIPT_PROBE_SOURCE_PATHS:
        path = root / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# exact probe fixture\n", encoding="utf-8")
    _git(
        root, "add", ".gitignore", "assets/motion.npz", "configs/dynamic.json",
        "configs/hold.json", "configs/frame0_exact_artifact.json",
        "configs/timing_receipt.json",
        *materializer._L.FRAME0_RECEIPT_PROBE_SOURCE_PATHS,
    )
    _git(root, "commit", "-m", "tracked frame0 artifact inputs")
    artifact_source_commit = _git(root, "rev-parse", "HEAD")
    live = _live_safety(action_id, motion_sha, 62)
    birth_execution_horizon = {
        "schema_version": 1,
        "kind": "action_ball_frame0_dynamic_birth_execution_horizon_v1",
        "required_policy_ticks": 62,
        "control_decimation": 4,
        "required_physics_substeps": 248,
        "plant_template_file_sha256": dynamic_sha,
        "plant_template_content_sha256": dynamic["content_sha256"],
    }
    dynamic_birth_gate_evidence = {
        "schema_version": 1,
        "kind": "action_ball_frame0_dynamic_birth_gate_evidence_v1",
        "thresholds_preregistered": {
            "table_contact_count_max": 0,
            "nonfinite_count_max": 0,
            "actual_hard_edge_joint_count_max": 0,
            "minimum_forward_hard_gap_rad_exclusive_min": 0.0,
        },
        "observed": {
            "table_contact_count": 0,
            "nonfinite_count": 0,
            "current_actual_hard_edge_joint_count": 0,
            "substep_actual_hard_edge_joint_count": 0,
            "forward_headroom": [
                {
                    "state": state,
                    "minimum_forward_hard_gap_rad": 1.0,
                    "minimum_forward_hard_gap_joint_name": "joint_00",
                }
                for state in ("preterminal", "final")
            ],
        },
        "nominal_scope": {
            "actor_bias": "exact_frame0_normalized_action",
            "per_env_joint_default_offset_dr_preserved": True,
            "per_env_joint_default_offset_range_rad": [-0.01, 0.01],
            "full_dr_distribution_hold_pass_claimed": False,
        },
    }
    frame0_receipt = _sealed(
        {
            "schema_version": 2,
            "kind": materializer._L.FRAME0_EXACT_RECEIPT_KIND,
            "diagnostic_unauthorized": True,
            "source_kind": materializer._L.FRAME0_EXACT_SOURCE_KIND,
            "verdict": "PASS",
            "action_id": action_id,
            "motion_sha256": motion_sha,
            "artifact_file_sha256": frame0_artifact_sha,
            "artifact_content_sha256": frame0_artifact["content_sha256"],
            "artifact_source_commit": artifact_source_commit,
            "probe_source_commit": artifact_source_commit,
            "plant_template_file_sha256": dynamic_sha,
            "plant_template_content_sha256": dynamic["content_sha256"],
            "probe_input_file_sha256": "3" * 64,
            "probe_input_content_sha256": "4" * 64,
            "live_safety_evidence_file_sha256": _sha(materializer.canonical_bytes(live)),
            "live_safety_evidence_content_sha256": live["content_sha256"],
            "live_safety_evidence": live,
            "policy_dt_s": materializer._L.POLICY_DT_S,
            "wait_schedule_canonical_sha256": materializer._L.WAIT_SCHEDULE[
                "canonical_sha256"
            ],
            "timing_receipt": frame0_artifact["timing_receipt"],
            "timing_receipt_canonical_sha256": timing["canonical_sha256"],
            "birth_horizon": birth_horizon,
            "birth_execution_horizon": birth_execution_horizon,
            "dynamic_birth_gate_evidence": dynamic_birth_gate_evidence,
        }
    )
    frame0_receipt_sha = _write_json(
        root / "configs/frame0_exact_receipt.json", frame0_receipt
    )
    _git(root, "add", "configs/frame0_exact_receipt.json")
    _git(root, "commit", "-m", "track frame0 exact receipt")
    source_commit = _git(root, "rev-parse", "HEAD")

    manifest = {
        "schema_version": 3,
        "action_order": [action_id],
        "mobility_mode": "no_move",
        "actions": [{
            "action_id": action_id, "action_uid": materializer._L.ACTION_UID,
            "motion_path": "assets/motion.npz", "motion_sha256": motion_sha,
        }],
        "solver_profile_sha256": "1" * 64,
        "physics_profile_sha256": "2" * 64,
    }
    manifest_sha = _write_json(root / "vendor_assets/manifest.json", manifest)
    tape_unsigned = {
        "schema_version": 1,
        "kind": "action_ball_n1_immutable_single_question_tape",
        "diagnostic_unauthorized": True,
        "row_count": 1,
        "reset_semantics": {
            "online_lm_calls": 0,
            "physical_rng_draws": 0,
        },
        "question": {
            "action_uid": materializer._L.ACTION_UID,
            "motion_sha256": motion_sha,
            "physics_sha256": "2" * 64,
            "ball_contact_w_m": [0.5, 0.0, 1.0],
            "incoming_velocity_w_mps": [-3.0, 0.1, 0.2],
            "incoming_spin_w_radps": [0.0, 10.0, 0.0],
            "base_goal_w_m": [-0.2, 0.3, 1.0],
        },
        "targets": {
            materializer._L.A_SELECTED_TAPE_VARIANT: {
                "recipe": materializer._L.A_SELECTED_TAPE_VARIANT,
                "producer_sha256": "a" * 64,
                "column_sha256": "b" * 64,
                "runtime_target": {"pre_swing_wait_s": 0.7},
            }
        },
        "source_task_receipt": timing,
    }
    tape = {**tape_unsigned, "canonical_sha256": materializer.canonical_sha256(tape_unsigned)}
    tape_sha = _write_json(root / "vendor_assets/tape.json", tape)
    bundle = {
        "schema_version": 1,
        "artifact_type": "measured_action_ball_n1_diagnostic_bundle_v1",
        "action_id": action_id, "action_uid": materializer._L.ACTION_UID, "measured_uid": "Take_061_unit04_BH",
        "target_recipe": "current_lm",
        "target_validity": {"order": ["position", "velocity", "face"], "mask": [True, True, True]},
        "immutable_tape": {"path": "vendor_assets/tape.json", "sha256": tape_sha},
        "motion": {"path": "assets/motion.npz", "sha256": motion_sha},
        "claims": {"diagnostic_unauthorized": True},
        "runtime_contract": {
            "physical_ball_semantics": materializer._L.PHYSICAL_BALL_SEMANTICS,
            "reset_inverse_solve": False, "target_source": "immutable_tape",
        },
    }
    bundle_sha = _write_json(root / "vendor_assets/bundle.json", bundle)
    pins = {
        "bundle": bundle_sha, "tape": tape_sha, "manifest": manifest_sha,
        "motion": motion_sha, "dynamic": dynamic_sha, "hold": hold_sha,
        "frame0_artifact": frame0_artifact_sha,
        "frame0_receipt": frame0_receipt_sha,
    }
    return root, pins, source_commit


def _argv(root: Path, pins: dict[str, str], commit: str, output: str) -> list[str]:
    return [
        "--repo-root", str(root), "--source-commit", commit,
        "--action-manifest-path", "configs/action_manifest.json", "--expected-action-manifest-sha256", pins["manifest"], "--action-manifest-explicit",
        "--motion-path", "assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz", "--expected-motion-sha256", pins["motion"],
        "--dynamic-ready-artifact-path", "configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json", "--expected-dynamic-ready-artifact-sha256", pins["dynamic"],
        "--dynamic-ready-nominal-receipt-path", "configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/take061.robust20n.nominal_hold.v1.json", "--expected-dynamic-ready-nominal-receipt-sha256", pins["hold"],
        "--teacher-frame0-artifact-path", "configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/take_061_unit04_bh.frame0_exact.v1.json", "--expected-teacher-frame0-artifact-sha256", pins["teacher_frame0"],
        "--initial-center-task-receipt-path", "configs/initial_center_a.task_receipt.v5.json", "--expected-initial-center-task-receipt-sha256", pins["initial_center_receipt"],
        "--output", output,
    ]


def _recommit_frame0_artifact(
    root: Path, pins: dict[str, str], mutate
) -> str:
    artifact_path = root / (
        "configs/action_ball_n1_measured_20260803/"
        "a211_frame0_exact_20260803/"
        "take_061_unit04_bh.frame0_exact.v1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact.pop("content_sha256")
    mutate(artifact)
    artifact = _sealed(artifact)
    pins["teacher_frame0"] = _write_json(artifact_path, artifact)
    _git(root, "add", str(artifact_path.relative_to(root)))
    _git(root, "commit", "-m", "replace resealed frame0 artifact")
    return _git(root, "rev-parse", "HEAD")


def test_canonical_split_ready_chain_materializes_then_active_validator_accepts(
    tmp_path,
):
    root, pins, commit = _fixture(tmp_path)
    output = "configs/a211/fresh_lineage.json"
    assert materializer.main(_argv(root, pins, commit, output)) == 0
    lineage_path = root / output
    raw = lineage_path.read_bytes()
    lineage = json.loads(raw)
    assert raw == materializer.canonical_bytes(lineage) + b"\n"
    assert lineage["schema_version"] == 5
    assert lineage["actor_layout_identity"] == materializer._L._actor_layout_identity()
    assert "bundle" not in lineage
    assert "immutable_tape" not in lineage
    assert lineage["runtime_target_contract"]["source"] == "online_solver"
    assert lineage["runtime_target_contract"]["immutable_tape_consumed_by_runtime"] is False
    assert lineage["dr_l0_manifest"]["hard_contract_identity"] == (
        "action_ball_dr_l0_exact_all_off_v1"
    )
    pin = {"path": output, "sha256": _sha(raw)}
    with pytest.raises(materializer._L.LaunchRefused, match="not tracked"):
        materializer._L._validate_lineage(root, commit, pin)
    _git(root, "add", output)
    _git(root, "commit", "-m", "commit fresh lineage closure")
    committed = _git(root, "rev-parse", "HEAD")
    accepted = materializer._L._validate_lineage(root, committed, pin)
    assert accepted["lineage_sha256"] == pin["sha256"]
    assert accepted["dr_l0_manifest"] == lineage["dr_l0_manifest"]


def test_formal_parser_has_no_bundle_or_tape_surface(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    argv = _argv(root, pins, commit, "configs/a211/no_tape.json")
    parsed = materializer._parser().parse_args(argv)
    assert not hasattr(parsed, "bundle_path")
    assert not hasattr(parsed, "immutable_tape_path")


@pytest.mark.parametrize(
    ("field", "index"),
    (
        ("root_pos_w_m", 1),
        ("root_quat_wxyz", 2),
        ("joint_pos_rad", 7),
    ),
)
def test_rejects_resealed_frame0_state_that_differs_from_pinned_motion(
    tmp_path, field, index
):
    root, pins, _commit = _fixture(tmp_path)

    def mutate(artifact):
        artifact["frame0"][field][index] += 0.125

    commit = _recommit_frame0_artifact(root, pins, mutate)
    output = "configs/a211/wrong-resealed-state.json"
    assert materializer.main(_argv(root, pins, commit, output)) == 2
    assert not (root / output).exists()


@pytest.mark.parametrize("mutation", ("extra", "missing", "integer_zero"))
def test_rejects_non_exact_frame0_payload_contract(tmp_path, mutation):
    root, pins, _commit = _fixture(tmp_path)

    def mutate(artifact):
        if mutation == "extra":
            artifact["frame0"]["units"] = "SI"
        elif mutation == "missing":
            artifact["frame0"].pop("root_quat_wxyz")
        else:
            artifact["frame0"]["root_lin_vel_w_mps"][0] = 0

    commit = _recommit_frame0_artifact(root, pins, mutate)
    output = "configs/a211/bad-payload-%s.json" % mutation
    assert materializer.main(_argv(root, pins, commit, output)) == 2
    assert not (root / output).exists()


def test_accepts_same_chain_after_inputs_are_tracked_at_source_commit(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    argv = _argv(root, pins, commit, "configs/a211/tracked_lineage.json")
    argv = [value for value in argv if value != "--action-manifest-explicit"]
    assert materializer.main(argv) == 0


def test_rejects_reformatted_split_ready_authority_bytes(tmp_path):
    root, pins, _commit = _fixture(tmp_path)
    relative = (
        "configs/action_ball_n1_measured_20260803/"
        "evidence_holdpass_robust20n_20260803/"
        "take061.measured_teacher.yaw_aligned_full_seed.robust20n."
        "dynamic_ready.v2.json"
    )
    path = root / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = materializer.canonical_bytes(document) + b"\n"
    path.write_bytes(raw)
    pins["dynamic"] = _sha(raw)
    _git(root, "add", relative)
    _git(root, "commit", "-m", "reformat authority bytes")
    commit = _git(root, "rev-parse", "HEAD")
    assert materializer.main(
        _argv(root, pins, commit, "configs/a211/pretty_receipt_lineage.json")
    ) == 2


def test_refuses_ignored_lineage_destination(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    assert materializer.main(_argv(root, pins, commit, "vendor_assets/lineage.json")) == 2
