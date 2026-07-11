#!/usr/bin/env python3
"""Fail-closed validator for the Phase-1 recovery-tuple A/B/C preregistration.

This tool is deliberately CPU-only and side-effect free.  ``design-check`` verifies the
content-addressed source audit, the rejected deploy hybrid, the ordered structural arms,
checkpoint compatibility boundaries, the ready-set definition, and the conditional reward
follow-up.  ``launch-check`` additionally requires every execution binding and therefore fails
until a later, newly hashed launch manifest supplies the missing implementations and artifacts.

It never imports Isaac, opens a simulator, contacts a pod, or issues a robot command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_TRAINING_COMMIT = "d3cdbdfc2f6e30726aa197b7bca66496ee0d39e5"
EXPECTED_GATE3_COMMIT = "1d46ef2cbb915efc135251f9b32f4ec25d0342ab"
EXPECTED_GATE3_PATH = (
    "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
    "a3_pingpong/pp_policy.hpp"
)
EXPECTED_GATE3_BYTES = 120413
EXPECTED_GATE3_SHA256 = "8c9814c42ebea404f95ef74821a924776860d4aef8b8ba726ad3364a5f00eba4"
EXPECTED_GATE3_STAND_PATH = (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
EXPECTED_GATE3_STAND_BYTES = 49107
EXPECTED_GATE3_STAND_SHA256 = "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97"

EXPECTED_PREREQUISITES = {
    "event_timing_prereg": (
        "configs/phase1_event_timing_t0_t1_prereg_20260711.json",
        16231,
        "2e7c4a344c0f2f81f67fd1246e5a724eaef92570c45e830c85f298377f52289c",
    ),
    "event_timing_schedule_spec": (
        "configs/phase1_event_timing_schedule_spec_20260711.json",
        5813,
        "8477e3e60c9e9fe193fd82e2b79e94e2a3037e7d4b7a969b34b36ad7f1f96396",
    ),
}

EXPECTED_TRAINING_BLOBS = {
    "robot_config": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "robots/agibot_a3.py",
        17640,
        "885626271940ae75c9b6c05bce347168fcb7fdb0d8ea7b1be0c5144a18169053",
    ),
    "motion_command": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "tasks/tracking/mdp/commands.py",
        83784,
        "648378535d3cc3d32ab3be14ce41007d377ea394baf209e0872545b26834a7ee",
    ),
    "racket_target_command": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "tasks/tracking/mdp/hope_commands.py",
        255691,
        "5c6db5bd63c6bc36b48b428ff809312cceebdc1f968e9685c54bc32a4a73bd93",
    ),
    "event_timing": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "tasks/tracking/mdp/event_timing.py",
        25801,
        "07adabda09d04c8c8b7ed02ea0255f3b4880201836e7bb3a0e45e234cfd06baa",
    ),
    "observations": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "tasks/tracking/mdp/hope_observations.py",
        12600,
        "edb4a11426b77c47a6878b042597520a35284221a3a2127c959d4f3a710e66e2",
    ),
    "rewards": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "tasks/tracking/mdp/hope_rewards.py",
        47563,
        "dfe8cec3764f62f5fc13b312805ea28e21d73a1038aa7d9f3e8cdd649f211f7f",
    ),
}

EXPECTED_ARMS = ("A_explicit_bridge", "B_canonical_tuple", "C_previous_tuple")
EXPECTED_TRANSITIONS = (
    "forehand_to_forehand",
    "forehand_to_backhand",
    "backhand_to_forehand",
    "backhand_to_backhand",
)
EXPECTED_READY_CONJUNCTS = {
    "station_xy_yaw_tolerance",
    "upright_height_and_gravity_projection",
    "low_base_joint_and_racket_velocity",
    "bilateral_support_contact_and_slip",
    "joint_limit_torque_qdes_and_thermal_margin",
    "self_table_net_and_ground_clearance",
    "next_family_and_deadline_reachability",
}
REQUIRED_IMPLEMENTATION_BINDINGS = (
    "immutable_random_arrival_screen_schedule",
    "immutable_random_arrival_decision_schedule",
    "immutable_question_schedule",
    "current_179_checkpoint_inventory",
    "A_bridge_source",
    "A_bridge_trajectory_certificate",
    "A_actor_handoff_contract",
    "B_canonical_tuple_source",
    "B_ready_set_selector",
    "B_fresh_checkpoint_set",
    "C_complete_tuple_source",
    "C_fresh_checkpoint_set",
    "shared_training_hard_contract",
    "isaac_continuous_no_reset_judge",
    "vendor_gate3_continuous_no_reset_judge",
    "canonical_ready_base_racket_target_numeric_contract",
    "ready_set_measurement",
    "racket_handle_self_hit_instrumentation",
    "semantics_correct_calibrated_plant",
)


class ContractError(ValueError):
    """The preregistered design is malformed or has been changed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON mapping")
    return value


def git_bytes(root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if result.returncode:
        raise ContractError(
            f"cannot read git blob {commit}:{path}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def verify_repo_binding(binding: Any, *, label: str, root: Path) -> Path:
    if not isinstance(binding, dict):
        raise ContractError(f"{label} must be a mapping")
    rel = binding.get("repo_path")
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise ContractError(f"{label}.repo_path must be a non-empty relative path")
    expected_sha = require_sha(binding.get("sha256"), f"{label}.sha256")
    expected_bytes = binding.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ContractError(f"{label}.bytes must be positive")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ContractError(f"{label} escapes repository") from None
    if not path.is_file():
        raise ContractError(f"{label} missing: {path}")
    data = path.read_bytes()
    if len(data) != expected_bytes or sha256_bytes(data) != expected_sha:
        raise ContractError(f"{label} content binding mismatch")
    return path


def verify_git_binding(binding: Any, *, label: str, root: Path) -> None:
    if not isinstance(binding, dict):
        raise ContractError(f"{label} must be a mapping")
    commit = binding.get("commit")
    path = binding.get("repo_path")
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise ContractError(f"{label}.commit must be a full lowercase commit")
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ContractError(f"{label}.repo_path must be relative")
    expected_sha = require_sha(binding.get("sha256"), f"{label}.sha256")
    expected_bytes = binding.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ContractError(f"{label}.bytes must be positive")
    data = git_bytes(root, commit, path)
    if len(data) != expected_bytes or sha256_bytes(data) != expected_sha:
        raise ContractError(f"{label} git blob content mismatch")


def _require_map(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a mapping")
    return value


def validate_sources(prereg: dict[str, Any], root: Path) -> None:
    prereqs = _require_map(prereg.get("prerequisites"), "prerequisites")
    if set(prereqs) != set(EXPECTED_PREREQUISITES):
        raise ContractError("prerequisite set changed")
    for name, (path, size, digest) in EXPECTED_PREREQUISITES.items():
        if prereqs[name] != {"repo_path": path, "bytes": size, "sha256": digest}:
            raise ContractError(f"{name} prerequisite binding changed")
        verify_repo_binding(prereqs[name], label=name, root=root)

    source = _require_map(prereg.get("audited_training_source"), "audited_training_source")
    if source.get("commit") != EXPECTED_TRAINING_COMMIT:
        raise ContractError("audited training commit changed")
    blobs = _require_map(source.get("git_blobs"), "audited_training_source.git_blobs")
    if set(blobs) != set(EXPECTED_TRAINING_BLOBS):
        raise ContractError("audited training blob set changed")
    for name, (path, size, digest) in EXPECTED_TRAINING_BLOBS.items():
        expected = {
            "commit": EXPECTED_TRAINING_COMMIT,
            "repo_path": path,
            "bytes": size,
            "sha256": digest,
        }
        if blobs[name] != expected:
            raise ContractError(f"training source binding changed for {name}")
        verify_git_binding(blobs[name], label=f"training {name}", root=root)

    gate3 = _require_map(prereg.get("gate3_readonly_evidence"), "gate3_readonly_evidence")
    expected_gate3 = {
        "commit": EXPECTED_GATE3_COMMIT,
        "repo_path": EXPECTED_GATE3_PATH,
        "bytes": EXPECTED_GATE3_BYTES,
        "sha256": EXPECTED_GATE3_SHA256,
        "worktree_was_read_only": True,
        "modified_by_this_work": False,
    }
    if gate3 != expected_gate3:
        raise ContractError("Gate3 read-only evidence binding changed")
    verify_git_binding(gate3, label="Gate3 pp_policy", root=root)


def validate_semantics(prereg: dict[str, Any]) -> None:
    observed = _require_map(prereg.get("observed_training_semantics"), "observed_training_semantics")
    required = {
        "natural_hold_reference": "default_stand_position_plus_zero_reference_velocity",
        "natural_wrap_physical_state": "carry_no_teleport_when_wrap_teleport_false",
        "natural_wrap_question_install": "complete_new_pos_vel_normal_side_tuple_atomically",
        "post_strike_before_wrap": "complete_previous_swing_tuple_unchanged",
        "T1_before_reveal": "complete_previous_swing_tuple_unchanged_while_final_clip_frame_clamped",
        "T1_at_reveal": "complete_new_pos_vel_normal_side_incoming_base_anchor_tuple_atomically",
        "T1_state_reset": "none_robot_last_action_history_noise_all_carry",
        "actor_target_latency": "pos_vel_normal_side_share_one_atomic_delay_dropout_generation",
        "time_to_strike_latency": "not_delayed",
        "mixed_generation_tuple_seen_in_training": False,
    }
    if observed != required:
        raise ContractError("observed training tuple semantics changed")

    hybrid = _require_map(prereg.get("rejected_current_hybrid"), "rejected_current_hybrid")
    expected = {
        "idle_position_generation": "new_live_base_anchored_position",
        "idle_velocity_generation": "previous_strike_velocity",
        "idle_normal_generation": "previous_strike_normal_and_rho",
        "position_motion": "moves_with_live_base_each_tick",
        "training_equivalent": False,
        "formal_arm_allowed": False,
        "classification": "mixed_generation_tuple_and_moving_target_OOD",
        "parameter_tuning_can_make_formal": False,
        "separate_handoff_mismatch": "static_stand_handoff_zeroes_last_action_before_reengage",
    }
    if hybrid != expected:
        raise ContractError("rejected hybrid finding changed")


def validate_structural_axis(prereg: dict[str, Any]) -> None:
    axis = _require_map(prereg.get("structural_axis"), "structural_axis")
    if axis.get("stage_order") != [
        "D0_current_checkpoint_A_vs_C_diagnostic_only",
        "S1_fresh_exact_A_B_C_paired_structure",
        "R2_conditional_reward_followup_only_after_structure",
    ]:
        raise ContractError("structural stage order changed")
    arms = axis.get("arms")
    if not isinstance(arms, list) or tuple(arm.get("id") for arm in arms if isinstance(arm, dict)) != EXPECTED_ARMS:
        raise ContractError("ordered A/B/C arms changed")

    a, b, c = arms
    if a != {
        "id": "A_explicit_bridge",
        "controller": "content_bound_interruptible_safe_PD_or_trajectory_bridge",
        "recovery_target": "deterministic_projection_into_ready_set",
        "actor_role_during_bridge": "not_in_control_but_state_history_must_remain_continuous",
        "action_history_semantics": "content_bound_executed_vs_policy_action_mapping_required_no_assumed_equivalence",
        "reveal_policy": "interrupt_or_replan_without_deadline_shift_else_count_infeasible",
        "physical_or_policy_state_reset": False,
        "last_action_or_history_zeroing_allowed": False,
        "claim": "explicit_controller_baseline_not_learned_recovery",
    }:
        raise ContractError("A explicit bridge contract changed")
    if b != {
        "id": "B_canonical_tuple",
        "controller": "same_179_actor_through_recovery_and_next_swing",
        "recovery_tuple": [
            "ready_set_selected_racket_position",
            "zero_desired_racket_velocity",
            "neutral_ready_normal",
            "rho_zero",
            "ready_phase_semantics",
        ],
        "tuple_install": "atomic_at_recovery_entry_then_atomic_next_question_at_reveal",
        "ready_position_is_exact_frame0": False,
        "new_actor_observation_dimension_in_S1": False,
        "fresh_training_required": True,
    }:
        raise ContractError("B canonical tuple contract changed")
    if c != {
        "id": "C_previous_tuple",
        "controller": "same_179_actor_through_recovery_and_next_swing",
        "recovery_tuple": "complete_previous_pos_vel_normal_rho_side_tuple",
        "tuple_install": "no_field_change_before_reveal_then_atomic_complete_next_question",
        "base_anchored_position_rewrite": False,
        "mixed_generation_fields": False,
        "fresh_training_required_for_learned_random_arrival_claim": True,
    }:
        raise ContractError("C previous tuple contract changed")

    compat = _require_map(axis.get("current_179_checkpoint_compatibility"), "current checkpoint compatibility")
    expected_compat = {
        "scope": "only_individually_finite_lineage_bound_179_atomic_question_checkpoints",
        "A": {
            "usable_now": "frozen_swing_subpolicy_diagnostic_after_bridge_and_handoff_cert",
            "formal_ABC_causal_comparison_requires_fresh": True,
            "learned_recovery_claim_allowed": False,
        },
        "B": {
            "usable_now": False,
            "reason": "zero_velocity_neutral_normal_canonical_tuple_absent_from_current_training",
            "fresh_retrain_required": True,
        },
        "C": {
            "usable_now": "zero_shot_coherent_tuple_diagnostic_only",
            "reason_not_formal": "extended_post_clip_dwell_and_random_reveal_not_trained",
            "fresh_retrain_required_for_learned_recovery_claim": True,
        },
        "current_checkpoint_may_be_relabeled_T1_trained": False,
        "unbound_checkpoint_path_or_SHA_may_run": False,
    }
    if compat != expected_compat:
        raise ContractError("current 179 checkpoint compatibility boundary changed")


def validate_ready_set(prereg: dict[str, Any]) -> None:
    ready = _require_map(prereg.get("ready_set_contract"), "ready_set_contract")
    if ready.get("ready_is_exact_motion_frame0") is not False:
        raise ContractError("ready must be a set, never exact frame 0")
    if ready.get("definition") != "conjunction_not_weighted_score":
        raise ContractError("ready-set definition must be a conjunction")
    if set(ready.get("required_conjuncts", [])) != EXPECTED_READY_CONJUNCTS:
        raise ContractError("ready-set safety/reachability conjuncts changed")
    if ready.get("selector") != "deterministic_content_bound_projection_inside_set":
        raise ContractError("canonical ready selector changed")
    if ready.get("reachability_quantifier") != "every_enabled_next_motion_question_family_and_random_arrival_deadline_cell":
        raise ContractError("ready set must quantify over every enabled next task/deadline")
    if ready.get("empty_global_intersection_policy") != "declare_family_ready_sets_and_explicit_transition_graph_or_fail_closed":
        raise ContractError("empty ready intersection cannot silently shrink the task")
    if tuple(ready.get("transition_cells", [])) != EXPECTED_TRANSITIONS:
        raise ContractError("ready-set transition coverage changed")


def validate_ready_static_evidence(prereg: dict[str, Any], root: Path) -> None:
    evidence = _require_map(prereg.get("ready_state_static_evidence"), "ready_state_static_evidence")
    expected = {
        "isaac_robot_source": {
            "commit": EXPECTED_TRAINING_COMMIT,
            "repo_path": EXPECTED_TRAINING_BLOBS["robot_config"][0],
            "bytes": EXPECTED_TRAINING_BLOBS["robot_config"][1],
            "sha256": EXPECTED_TRAINING_BLOBS["robot_config"][2],
        },
        "vendor_stand_source": {
            "commit": EXPECTED_GATE3_COMMIT,
            "repo_path": EXPECTED_GATE3_STAND_PATH,
            "bytes": EXPECTED_GATE3_STAND_BYTES,
            "sha256": EXPECTED_GATE3_STAND_SHA256,
        },
        "isaac_reset_pelvis_xyz_m": [0.0, 0.0, 1.0684],
        "vendor_stand_pelvis_xyz_m": [-0.0416378, 0.000359049, 1.06839],
        "vendor_stand_rpy_deg_approx": [-0.030, 0.249, 0.042],
        "joint_31_l2_difference_rad": 0.171845,
        "head_yaw_difference_rad": -0.169416,
        "joint_l2_without_head_rad": 0.028789,
        "root_x_difference_m": -0.0416378,
        "stage1_contact_position_frame": "env_origin_absolute_world",
        "actor_179_position_observation": "target_minus_current_racket_FK_then_yaw_frame",
        "root_x_difference_automatically_cancels": False,
        "causal_status": "hypothesis_to_isolate_not_proven_root_cause",
        "complete_numeric_ready_base_racket_target_contract_bound": False,
    }
    if evidence != expected:
        raise ContractError("ready-state static evidence or causal boundary changed")
    verify_git_binding(evidence["isaac_robot_source"], label="Isaac ready source", root=root)
    verify_git_binding(evidence["vendor_stand_source"], label="vendor stand source", root=root)


def validate_frozen_and_reward(prereg: dict[str, Any]) -> None:
    frozen = _require_map(prereg.get("frozen_structural_comparison"), "frozen_structural_comparison")
    expected_frozen = {
        "immutable_random_arrival_schedule": "same_rows_order_reveal_ticks_deadlines_for_every_arm",
        "question_bank_motion_face_plant": "same_content_SHAs",
        "network_observation_action": "same_179_schema_no_new_phase_dimension_in_S1",
        "reward_source_weights_and_total_budget": "same_bytes_and_values_no_new_recovery_income",
        "seeds_optimizer_updates_batching_checkpoint_cadence": "paired_and_identical",
        "episode_horizon": "same_30_seconds",
        "mid_sequence_reset_teleport_last_action_history_noise_reset": False,
        "deadline_shift_or_replacement": False,
        "safety_is_noncompensable_gate": True,
        "all_scheduled_opportunities_denominator": True,
    }
    if frozen != expected_frozen:
        raise ContractError("non-treatment structural axes are not frozen")

    reward = _require_map(prereg.get("conditional_reward_followup"), "conditional_reward_followup")
    if reward.get("status") != "not_authorized_until_structural_gate":
        raise ContractError("reward follow-up was prematurely authorized")
    if reward.get("trigger") != "B_or_C_fails_ready_set_acquisition_without_single_strike_regression":
        raise ContractError("reward follow-up trigger changed")
    if reward.get("components") != [
        "post_swing_balance_absorption_debt",
        "ready_set_potential_progress",
        "random_arrival_task_readiness",
    ]:
        raise ContractError("recovery reward components changed")
    if reward.get("first_step") != "scale_normalize_each_component_on_frozen_rollouts":
        raise ContractError("reward scale normalization must precede ablation")
    if reward.get("interaction_design") != "full_2_pow_3_on_off_factorial_paired_seeds":
        raise ContractError("reward interactions require a full 2^3 design")
    if reward.get("mixture_design_after_interactions_only") != "constant_total_budget_simplex_or_D_optimal":
        raise ContractError("reward mixture must be conditional and constant-budget")
    if reward.get("positive_brake_hold_survival_income_allowed") is not False:
        raise ContractError("positive hold income may leak through GAE and is prohibited")
    if reward.get("safety_as_reward_weight_allowed") is not False:
        raise ContractError("safety cannot be traded as a reward weight")
    if reward.get("single_strike_nonregression_gate") is not True:
        raise ContractError("reward follow-up must retain the single-strike gate")


def validate_evaluation(prereg: dict[str, Any]) -> None:
    evaluation = _require_map(prereg.get("evaluation_contract"), "evaluation_contract")
    required = {
        "same_immutable_random_arrival_exam_for_all_arms": True,
        "engines_in_order": ["Isaac", "Agibot_vendor_MuJoCo_Gate3"],
        "vendor_gate3_is_final_arbiter": True,
        "mid_sequence_reset_or_teleport": False,
        "q10_role": "screen_only_directional_no_stop_no_promote_no_checkpoint_selection",
        "q50_role": "decision_same_family_immutable",
        "opportunity_denominator": "all_scheduled_including_miss_infeasible_and_fall",
        "transition_cells": list(EXPECTED_TRANSITIONS),
        "report_peak_and_terminal_checkpoints": True,
        "self_hit_fall_table_net_contact": "hard_fail_not_reward_tradeoff",
        "first_strike_nonregression_required": True,
        "isaac_vendor_disagreement_policy": "block_and_root_cause_never_average",
        "real_robot_authorized": False,
    }
    if evaluation != required:
        raise ContractError("immutable no-reset evaluation contract changed")


def validate_prereg(path: Path, expected_sha256: str) -> dict[str, Any]:
    root = repo_root()
    require_sha(expected_sha256, "--expected-prereg-sha256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ContractError(f"prereg SHA mismatch: {actual} != {expected_sha256}")
    prereg = read_json(path, "recovery tuple preregistration")
    if prereg.get("schema_version") != 1:
        raise ContractError("recovery tuple prereg schema must remain 1")
    if prereg.get("status") != "preregistered_structure_only_launch_blocked":
        raise ContractError("recovery tuple prereg status changed")
    if prereg.get("launch_authorized") is not False or prereg.get("real_robot_authorized") is not False:
        raise ContractError("this design cannot authorize launch or real robot")

    validator = _require_map(prereg.get("validator"), "validator")
    validator_path = verify_repo_binding(validator, label="validator", root=root)
    if validator_path != Path(__file__).resolve():
        raise ContractError("validator binding does not identify this script")

    validate_sources(prereg, root)
    validate_semantics(prereg)
    validate_structural_axis(prereg)
    validate_ready_set(prereg)
    validate_ready_static_evidence(prereg, root)
    validate_frozen_and_reward(prereg)
    validate_evaluation(prereg)

    bindings = _require_map(prereg.get("implementation_bindings"), "implementation_bindings")
    if tuple(bindings) != REQUIRED_IMPLEMENTATION_BINDINGS:
        raise ContractError("implementation binding names/order changed")
    if any(bindings[name] is not None for name in REQUIRED_IMPLEMENTATION_BINDINGS):
        raise ContractError("blocked prereg must keep every execution binding null")
    return prereg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("--mode", choices=("design-check", "launch-check"), default="design-check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prereg = validate_prereg(args.prereg, args.expected_prereg_sha256)
        if args.mode == "launch-check":
            missing = [name for name in REQUIRED_IMPLEMENTATION_BINDINGS if prereg["implementation_bindings"][name] is None]
            raise ContractError("LAUNCH BLOCKED: missing " + ", ".join(missing))
    except ContractError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "pass_design_only",
                "launch_authorized": False,
                "formal_arms": list(EXPECTED_ARMS),
                "current_hybrid_formal": False,
                "current_179_B_usable": False,
                "vendor_gate3_final": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
