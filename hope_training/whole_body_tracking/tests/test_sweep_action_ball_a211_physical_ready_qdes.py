from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/sweep_action_ball_a211_physical_ready_qdes.py"
)
SPEC = importlib.util.spec_from_file_location("_a211_ready_sweep_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


def _artifact() -> dict:
    names = ["joint_%02d" % index for index in range(31)]
    names[5] = sweep.JOINT_NAME
    unsigned = {
        "schema_version": 2,
        "kind": sweep.ARTIFACT_KIND,
        "action_id": "take061",
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "isaac_nominal_hold_validated": False,
        },
        "robot": {"family": "AgiBot A3", "joint_names": names},
        "physical_ready": {
            "root_pos_w_m": [0.0, 0.0, 1.0],
            "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_pos_rad": [0.0] * 31,
            "joint_vel_radps": [0.0] * 31,
        },
        "teacher_reference": {"motion_sha256": "1" * 64},
        "runtime_plant": {
            "default_joint_pos_rad": [0.0] * 31,
            "action_scale_rad": [1.0] * 31,
            "joint_stiffness": [50.0] * 31,
            "joint_effort_limits": [100.0] * 31,
            "qdes_joint_pos_limits": [[-0.5, 0.5] for _ in range(31)],
            "finite_projection_soft_envelope_inset_fraction": 0.1,
        },
        "hold_candidate": {
            "hold_qdes_joint_pos_rad": [0.0] * 31,
            "normalized_actor_action": [0.0] * 31,
            "hold_qdes_mode": "fresh_static_lp",
            "selected_hold_authority": {},
            "semantics": "old",
            "solver_report_role": "selected_hold_solution",
        },
        "sources": {"stable_motion": {"sha256": "1" * 64}},
    }
    return {**unsigned, "content_sha256": sweep.canonical_sha256(unsigned)}


def test_candidate_changes_only_hold_candidate_and_content_seal():
    base = _artifact()
    frozen = copy.deepcopy(base)
    candidate, metadata = sweep.derive_candidate(base, 0.12)
    assert base == frozen
    for key in base:
        if key not in {"hold_candidate", "content_sha256"}:
            assert candidate[key] == base[key]
    index = candidate["robot"]["joint_names"].index(sweep.JOINT_NAME)
    assert candidate["hold_candidate"]["hold_qdes_joint_pos_rad"][index] == 0.12
    assert candidate["hold_candidate"]["normalized_actor_action"][index] == 0.12
    assert metadata == {
        "candidate_id": "waist_roll_+0.12",
        "waist_roll_offset_rad": 0.12,
        "waist_roll_qdes_rad": 0.12,
        "maximum_initial_pd_effort_ratio": 0.06,
    }
    assert candidate["content_sha256"] != base["content_sha256"]
    sweep._verify_seal(candidate, name="candidate")


def test_candidate_refuses_qdes_outside_soft_envelope():
    with pytest.raises(sweep.SweepError, match="soft envelope"):
        sweep.derive_candidate(_artifact(), 0.41)


def _receipt(*, artifact_sha: str, content_sha: str, steps: int) -> dict:
    unsigned = {
        "schema_version": 1,
        "kind": sweep.RECEIPT_KIND,
        "verdict": "PASS",
        "artifact": {"sha256": artifact_sha, "content_sha256": content_sha},
        "candidate_physical_birth_written": True,
        "candidate_hold_qdes_and_delay_history_installed": True,
        "teacher_reference_unchanged": True,
        "teacher_physical_birth_separated": True,
        "plant_contract_match": True,
        "active_terminations": list(sweep.HARD_TERMINATIONS),
        "completed_policy_steps": steps,
        "completed_physics_steps": steps * 4,
        "terminal_reasons": [],
        "generic_terminated": False,
        "generic_truncated": False,
        "minimum_root_z_m": 1.0,
        "maximum_root_tilt_rad": 0.02,
        "joint_safety_telemetry": {
            "schema_version": 1,
            "complete": True,
            "current_actual_hard_edge_joint_count": 0,
            "substep_actual_hard_edge_joint_count": 0,
            "final_minimum_hard_gap_rad": 0.1,
        },
    }
    return {**unsigned, "content_sha256": sweep.canonical_sha256(unsigned)}


def test_receipt_requires_exact_four_substeps_per_policy_step():
    receipt = _receipt(artifact_sha="2" * 64, content_sha="3" * 64, steps=200)
    assert sweep.validate_receipt(
        receipt,
        artifact_sha="2" * 64,
        artifact_content_sha="3" * 64,
        policy_steps=200,
    )
    receipt["completed_physics_steps"] = 799
    receipt.pop("content_sha256")
    receipt["content_sha256"] = sweep.canonical_sha256(receipt)
    with pytest.raises(sweep.SweepError, match="structural"):
        sweep.validate_receipt(
            receipt,
            artifact_sha="2" * 64,
            artifact_content_sha="3" * 64,
            policy_steps=200,
        )


def test_selection_rule_prioritizes_hard_gap_before_tilt():
    better_gap = {
        "candidate_id": "b",
        "maximum_initial_pd_effort_ratio": 0.5,
        "full_receipt": {
            "joint_safety_telemetry": {"final_minimum_hard_gap_rad": 0.2},
            "maximum_root_tilt_rad": 0.5,
            "minimum_root_z_m": 0.9,
        },
    }
    prettier_pose = {
        "candidate_id": "a",
        "maximum_initial_pd_effort_ratio": 0.1,
        "full_receipt": {
            "joint_safety_telemetry": {"final_minimum_hard_gap_rad": 0.1},
            "maximum_root_tilt_rad": 0.01,
            "minimum_root_z_m": 1.1,
        },
    }
    assert min((better_gap, prettier_pose), key=sweep._rank) is better_gap
