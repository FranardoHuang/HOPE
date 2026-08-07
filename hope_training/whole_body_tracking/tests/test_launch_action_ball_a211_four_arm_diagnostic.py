"""CPU-only fail-closed tests for the executable A211 four-arm launcher."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch_action_ball_a211_four_arm_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("launch_a211_four_arm", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

_TRAINING_CONTRACT = launcher._OLD._load_training_contract_module(
    Path(__file__).resolve().parents[3]
)
_DR_L0_PAYLOAD = _TRAINING_CONTRACT.action_ball_dr_l0_contract_payload()
_DR_L0N_PAYLOAD = _TRAINING_CONTRACT.action_ball_dr_l0n_contract_payload()
_DR_L0_CONTRACT_SHA256 = (
    _TRAINING_CONTRACT.action_ball_dr_l0_contract_sha256()
)

_DR_L0_BINDING = {
    "path": launcher.DR_L0_MANIFEST_SOURCE,
    "file_sha256": "d" * 64,
    "contract_sha256": _DR_L0_CONTRACT_SHA256,
    "hard_contract_identity": "action_ball_dr_l0_exact_all_off_v1",
    "task_profile": launcher.TASK_PROFILE_ID,
}


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)


class _FakeFinite:
    def __init__(self, tensor):
        self.tensor = tensor

    def all(self):
        return self

    def item(self):
        return all(math.isfinite(value) for value in self.tensor.values)


class _FakeTorch:
    checkpoint = None
    load_error = None

    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)

    @staticmethod
    def isfinite(value):
        return _FakeFinite(value)

    @classmethod
    def load(cls, stream, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        assert stream.read()
        if cls.load_error is not None:
            raise cls.load_error
        return cls.checkpoint


def _write(path: Path, value) -> str:
    raw = launcher._B._canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _sealed(value):
    return {**value, "content_sha256": launcher.canonical_sha256(value)}


def _exact_zero_handoff_fields(
    *, motion_sha: str, joint_pos, root_pos, root_quat
):
    joint_pos = [float(value) for value in joint_pos]
    root_pos = [float(value) for value in root_pos]
    root_quat = [float(value) for value in root_quat]
    root_norm = float(np.linalg.norm(np.asarray(root_quat, np.float64)))
    audit_quat = (np.asarray(root_quat, np.float64) / root_norm).tolist()
    raw_sha = launcher._whole_body_state_sha256(joint_pos, root_pos, root_quat)
    audit_sha = launcher._whole_body_state_sha256(
        joint_pos, root_pos, audit_quat
    )
    handoff = {
        "schema_version": 1,
        "kind": "exact_frame0_zero_duration_handoff_v1",
        "selection_semantics": "threshold_first_exact_frame0_direct",
        "state_sha256_semantics": (
            "float64_array_bytes_without_quaternion_normalization_v1"
        ),
        "physical_ready_state_sha256": raw_sha,
        "teacher_frame0_state_sha256": raw_sha,
        "mjcf_audit_state_sha256": audit_sha,
        "stored_root_quaternion_norm": root_norm,
        "mjcf_audit_root_quat_wxyz": list(audit_quat),
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
        "required_followup_hold_gate": launcher.FRAME0_LIVE_RECEIPT_KIND,
        "required_followup_policy_steps": launcher.PHYSICAL_READY_HOLD_POLICY_STEPS,
        "required_followup_physics_steps": launcher.PHYSICAL_READY_HOLD_PHYSICS_STEPS,
        "diagnostic_unauthorized": True,
        "training_authorized": False,
    }
    robust = dict(launcher._DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS)
    return {
        "teacher_reference": {
            "semantics": "exact_motion_bytes_frame0_reference",
            "motion_sha256": motion_sha,
            "frame_index": 0,
            "root_pos_w_m": list(root_pos),
            "root_quat_wxyz": list(root_quat),
            "joint_pos_rad": list(joint_pos),
            "static_handoff_joint_vel_radps": [0.0] * 31,
            "static_handoff_velocity_semantics": (
                "constructed_zero_joint_velocity_endpoint_not_measured_motion_"
                "velocity"
            ),
        },
        "physical_ready": {
            "root_pos_w_m": list(root_pos),
            "root_quat_wxyz": list(root_quat),
            "joint_pos_rad": list(joint_pos),
            "joint_vel_radps": [0.0] * 31,
        },
        "frame0_handoff": dict(handoff),
        "physical_birth_composition": {
            "semantics": (
                "measured_frame0_direct_if_safe_else_lexicographic_whole_body_"
                "safe_ready"
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
            "stored_endpoint_state_sha256": raw_sha,
            "mjcf_audit_state_sha256": audit_sha,
            "stored_root_quat_wxyz": list(root_quat),
            "mjcf_audit_root_quat_wxyz": list(audit_quat),
            "stored_root_quaternion_norm": root_norm,
            "direct_frame0_robust_minimum_slacks": robust,
            "direct_frame0_robust_gate_sha256": launcher.canonical_sha256(robust),
            "safety_slacks": dict(robust),
            "evaluator_evidence": {
                "lp_feasible": True,
                "exact_state_lp_cache_hit": False,
                "evaluated_state_sha256": audit_sha,
                "required_minimum_normal_force_per_contact_n": 0.1,
                "required_minimum_normal_force_per_foot_n": 1.0,
            },
            "independent_measured_racket_frame0": {
                "authority": "independent_schema_v4_measured_racket_channel",
                "motion_sha256": motion_sha,
                "frame_index": 0,
            },
            "frame0_handoff": dict(handoff),
        },
    }


def _prelong_marker_lines(update: int) -> list[str]:
    invalid_samples = 7 if update == 0 else 0
    counters = {name: 0 for name in launcher._S.required_prelong_counter_names()}
    counters.update({
        launcher._S.TASK_INVALID_OBSERVED_COUNTER: invalid_samples,
        launcher._S.TASK_INVALID_REWARD_SUM_COUNTER: 0.0,
        launcher._S.TASK_INVALID_REWARD_ELIGIBLE_COUNTER: 0,
        launcher._S.READY_MIMIC_REWARD_SUM_COUNTER: 0.0,
        launcher._S.READY_MIMIC_ELIGIBLE_COUNTER: invalid_samples,
        launcher._S.SWING_MIMIC_REWARD_SUM_COUNTER: 3.0,
        launcher._S.SWING_MIMIC_ELIGIBLE_COUNTER: 98304 - invalid_samples,
        launcher._S.EXACT_STRIKE_TIMING_COUNTER: 9,
        launcher._S.ELIGIBLE_CLOSED_SWING_COUNTER: 9,
        launcher._S.ACTUAL_CONTACT_COUNTER: 0,
        launcher._S.ACHIEVED_FLIGHT_COUNTER: 0,
        launcher._S.UNKNOWN_ATTRIBUTION_COUNTER: 0,
    })
    group_values = {
        "balance": (4.0, 98304),
        "mimic": (3.0, 98304),
        "strike": (0.0, 9),
        "target": (1.0, 1),
        "outcome": (0.0, 0),
    }
    for group, (income, denominator) in group_values.items():
        counters[launcher._S.reward_group_sum_counter(group)] = income
        counters[launcher._S.reward_group_eligible_counter(group)] = denominator
    economy = {
        "event": "hope_action_ball_reward_ppo_economy_update",
        "schema_version": 1,
        "status": "PASS",
        "ppo_update": update,
        "gate": {
            "num_envs": 4096,
            "steps_per_env_per_update": 24,
            "rollout_samples_per_update": 98304,
        },
        "reward": {
            "explained_variance": 0.1,
            # ``motion`` 逐 update 变化是**故意**的:2026-08-07 起 pre-long gate 会拒收
            # "整窗逐位相同且非零"的未申报奖励项(见 DECLARED_CONSTANT_REWARD_TERMS)。
            # 本夹具原本写死 1.0,那正好撞在那道门上。
            "per_term_weighted_dt_sum": {
                "motion": 1.0 + 0.25 * update,
                "task": 0.0,
            },
            "per_term_eligible_denominator": {
                "motion": 98304,
                "task": 98304,
            },
        },
        "ppo": {
            "learning_rate": 1.0e-4,
            "approx_kl": 0.01,
            "clip_fraction": 0.2,
        },
        "gradient": {"pre_clip_total_grad_norm": 0.5},
        "policy": {
            "policy_std_min": 0.01,
            "policy_std_mean": 0.02,
            "policy_std_max": 0.03,
        },
    }
    groups = {
        "event": "hope_effective_reward_activation_by_action_update",
        "schema_version": 2,
        "ppo_update": update,
        "actions": [
            {
                "action_id": launcher.TEACHER_ID,
                "reward_groups": [
                    {
                        "group": "motion",
                        "eligibility": (
                            "reward_manager_evaluated_active_group_terms"
                        ),
                        "eligible_sample_count": 98304,
                        "weighted_sum": 4.0,
                    },
                    {
                        "group": "task",
                        "eligibility": (
                            "reward_manager_evaluated_active_group_terms"
                        ),
                        "eligible_sample_count": 98304,
                        "weighted_sum": 0.0,
                    },
                ],
            }
        ],
    }
    return [
        launcher._P.ECONOMY_PREFIX
        + json.dumps(economy, sort_keys=True, separators=(",", ":")),
        launcher._P.GROUP_PREFIX
        + json.dumps(groups, sort_keys=True, separators=(",", ":")),
        launcher._S.prelong_semantics_marker_line(
            ppo_update=update,
            counters=counters,
            profile=launcher._S.PRELONG_PROFILE_A211,
        ),
    ]


def _rewrite_prelong_marker(log_path: Path, update: int, mutate, *, allow_nan=False):
    prefix = launcher._S.PRELONG_SEMANTICS_MARKER_PREFIX
    rewritten = []
    matched = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            row = json.loads(line[len(prefix) :])
            if row["ppo_update"] == update:
                mutate(row)
                line = prefix + json.dumps(
                    row,
                    allow_nan=allow_nan,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                matched += 1
        rewritten.append(line)
    assert matched == 1
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _terminal_acceptance_fixture():
    gate = {
        "schema_version": 1,
        "kind": "action_ball_4096x5_prelong_terminal_gate",
        "status": "PASS",
        "diagnostic_unauthorized": True,
        "num_envs": 4096,
        "ppo_updates": 5,
    }
    prelong_gate = _sealed(
        {
            "schema_version": 1,
            "kind": "action_ball_a211_4096x5_prelong_gate_binding_v1",
            "diagnostic_unauthorized": True,
            "launch_claim_sha256": "1" * 64,
            "run_log_sha256": "2" * 64,
            "finite_model_sha256": "3" * 64,
            "semantic_marker_prefix": (
                launcher._S.PRELONG_SEMANTICS_MARKER_PREFIX
            ),
            "semantic_update_count": 5,
            "gate": gate,
            "gate_sha256": launcher.canonical_sha256(gate),
        }
    )
    return _sealed(
        {
            "schema_version": 1,
            "kind": launcher.SCALE4096_TERMINAL_ACCEPTANCE_KIND,
            "diagnostic_unauthorized": True,
            "launch_claim_sha256": "1" * 64,
            "run_log": {"path": "/fixture/run.log", "size_bytes": 1, "sha256": "2" * 64},
            "checkpoint": {
                "path": "/fixture/model_4.pt",
                "size_bytes": 1,
                "sha256": "3" * 64,
                # 跑满 5 个 update 的末位是 model_4.pt / iter=4。
                "filename_iteration": 4,
                "embedded_iteration": 4,
                "map_location": "cpu",
                "load_mode": "torch_weights_only",
                "tensor_groups": {},
                "all_tensors_finite": True,
            },
            "safety_counters": {
                "observed_ppo_updates": 5,
                "actual_hard_edge_event_count": 0,
                "actual_hard_terminal_count": 0,
                "joint_qdes_forbidden_terminal_count": 0,
                "joint_actual_forbidden_terminal_count": 0,
                "strict_hard_termination_count": 0,
                "table_contact_count": 0,
                "nonfinite_count": 0,
                "base_fell_tilt_terminal_count": 0,
                "base_too_low_terminal_count": 0,
                "physical_fall_by_reason_phase": {
                    reason: {phase: 0 for phase in launcher.PHYSICAL_FALL_PHASES}
                    for reason in launcher.PHYSICAL_FALL_REASONS
                },
                "table_contact_by_phase": {
                    phase: 0 for phase in launcher.PHYSICAL_FALL_PHASES
                },
                "task_wait_started_by_update": [12] * 5,
                "task_wait_started_count": 60,
                "task_reveal_reached_by_update": [10] * 5,
                "task_reveal_reached_count": 50,
            },
            "prelong_gate": prelong_gate,
        }
    )


def _live_safety(action_id: str, motion_sha: str, ticks: int) -> dict:
    names = ["joint_%02d" % index for index in range(31)]
    joint = {
        "schema_version": 1,
        "complete": True,
        "joint_order": names,
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
        "kind": launcher.FRAME0_LIVE_RECEIPT_KIND,
        "verdict": "PASS",
        "action_id": action_id,
        "motion_sha256": motion_sha,
        "teacher_reference_unchanged": True,
        "teacher_physical_birth_separated": False,
        "candidate_physical_birth_written": True,
        "candidate_hold_qdes_and_delay_history_installed": True,
        "plant_contract_match": True,
        "active_terminations": list(launcher.HARD_TERMINATION_UNION),
        "requested_duration_s": ticks * launcher.POLICY_DT_S,
        "completed_duration_s": ticks * launcher.POLICY_DT_S,
        "completed_policy_steps": ticks,
        "completed_physics_steps": ticks * 4,
        "terminal_reasons": [],
        "generic_terminated": False,
        "generic_truncated": False,
        "minimum_root_z_m": 0.9,
        "maximum_root_tilt_rad": 0.1,
        "both_feet_contact_fraction": 1.0,
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


def _required_effective_terms():
    return [
        {
            "name": name,
            "callable": requirement["callable"],
            "weight": 1.0,
            "params": dict(requirement["params"]),
        }
        for name, requirement in sorted(launcher.REQUIRED_EFFECTIVE_TERMS.items())
    ]


def _lineage(checkout: Path) -> dict:
    repo = Path(__file__).resolve().parents[3]
    pins = {}
    sources = {
        "motion": repo / "assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz",
        "action_manifest": repo / "configs/action_ball_n1_measured_20260803/fresh_core_seed0_20260803_take061_robust20n_r8_splitready/take_061_unit04_bh.full.manifest.v3.7d2139028427.json",
        "dynamic_ready_artifact": repo / "configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json",
        "dynamic_ready_nominal_receipt": repo / "configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/take061.robust20n.nominal_hold.v1.json",
        "teacher_frame0_artifact": repo / "configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/take_061_unit04_bh.frame0_exact.v1.json",
    }
    for key, source in sources.items():
        destination = checkout / source.name
        destination.write_bytes(source.read_bytes())
        pins[key] = {
            "path": destination.name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    source_receipt = json.loads(
        (
            repo
            / "configs/action_ball_n1_measured_20260803/"
            "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/"
            "current_lm.target.task_receipt.v5.f64f52137ad8.json"
        ).read_text(encoding="utf-8")
    )
    source_receipt.pop("canonical_sha256")
    source_receipt.update(
        {
            "sampling_stratum": "center",
            "birth_sampling_stratum": "center",
            "frontier_arm": None,
            "birth_frontier_arm": None,
            "time_to_contact_tick": 91,
            "time_to_contact_s": 1.82,
            "pre_swing_wait_s": 1.82 - source_receipt["scaled_t_hit_s"],
            "manifest_sha256": pins["action_manifest"]["sha256"],
        }
    )
    source_receipt["canonical_sha256"] = launcher.canonical_sha256(source_receipt)
    receipt_path = checkout / "initial_center_a.task_receipt.v5.json"
    receipt_sha = _write(receipt_path, source_receipt)
    pins["initial_center_task_receipt"] = {
        "path": receipt_path.name,
        "sha256": receipt_sha,
    }
    assert pins["dynamic_ready_artifact"]["sha256"] == launcher.SPLIT_READY_DYNAMIC_ARTIFACT_SHA256
    assert pins["dynamic_ready_nominal_receipt"]["sha256"] == launcher.SPLIT_READY_NOMINAL_HOLD_SHA256
    assert pins["teacher_frame0_artifact"]["sha256"] == launcher.SPLIT_READY_TEACHER_FRAME0_ARTIFACT_SHA256
    return {
        "schema_version": 5,
        "kind": launcher.LINEAGE_KIND,
        "actor_contract": launcher.ACTOR_CONTRACT,
        "actor_width": 211,
        "critic_contract": launcher.CRITIC_CONTRACT,
        "critic_width": 319,
        "trainability_contract": launcher.TRAINABILITY_CONTRACT,
        "actor_layout_identity": launcher._actor_layout_identity(),
        "task_profile": launcher.TASK_PROFILE_ID,
        "gym_task": launcher.GYM_TASK_ID,
        "target_semantics": launcher.TARGET_SEMANTICS,
        "runtime_target_contract": launcher._runtime_target_contract(),
        "curriculum_scope": launcher._curriculum_scope_contract(),
        "action_id": launcher.ACTION_ID,
        "teacher_id": launcher.TEACHER_ID,
        "seed": 0,
        "dr_l0_manifest": dict(_DR_L0_BINDING),
        **pins,
    }


def _result(
    path: Path,
    *,
    stage: str,
    materialization,
    policy=None,
    oracle=None,
    predecessor=None,
    completion=None,
    output_contract=None,
    terminal_acceptance=None,
) -> dict:
    result_namespace = path.parent / (path.stem + ".namespace")
    result_namespace.mkdir(parents=True, exist_ok=True)
    unsigned = {
        "schema_version": 1,
        "kind": launcher.RESULT_KIND,
        "diagnostic_unauthorized": True,
        "accepted": True,
        "launch_claim_sha256": "1" * 64,
        "stage": stage,
        "namespace": str(result_namespace),
        "completion": (
            {"terminal_kind": "clean_completion"}
            if completion is None
            else completion
        ),
        "gpu_admission": {"phase": "post_completion"},
        "output_contract": (
            {"fixture": True} if output_contract is None else output_contract
        ),
        "arm_materialization": materialization,
        "policy_recipe_materialization": policy,
        "oracle32_receipt": oracle,
        "predecessor_result": predecessor,
    }
    if terminal_acceptance is not None:
        unsigned["terminal_acceptance"] = terminal_acceptance
    digest = _write(path, _sealed(unsigned))
    return {"path": str(path), "sha256": digest}


def _generated_chain(tmp_path: Path, arm_id: str, lineage_sha: str):
    arm = launcher._arm_contract(arm_id)
    planned = launcher._planned_materialization(
        arm=arm,
        lineage={
            "lineage_sha256": lineage_sha,
            "dr_l0_manifest": dict(_DR_L0_BINDING),
        },
    )
    reward_artifact = tmp_path / (arm_id + ".effective_reward.json")
    reward_artifact.write_text("fixture\n", encoding="utf-8")
    materialization_unsigned = {
        key: value for key, value in planned.items() if key != "content_sha256"
    }
    materialization_unsigned.update(
        {
            "runtime_effective_reward_artifact": {
                "path": str(reward_artifact),
                "sha256": hashlib.sha256(reward_artifact.read_bytes()).hexdigest(),
            },
            "runtime_effective_reward_sha256": "3" * 64,
            "runtime_effective_reward_term_count": 10,
            "runtime_soft_weights": {
                "death_penalty": arm["soft_weights"]["death_penalty"],
                "joint_limit": arm["soft_weights"]["joint_limit"],
                "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
                "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
            },
        }
    )
    materialization = _sealed(materialization_unsigned)
    materialize = _result(
        tmp_path / (arm_id + ".materialize.json"),
        stage="materialize",
        materialization=materialization,
    )
    policy_artifact = tmp_path / (arm_id + ".policy_recipe.json")
    policy_artifact.write_text("fixture-policy\n", encoding="utf-8")
    policy = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.POLICY_MATERIALIZATION_KIND,
            "diagnostic_unauthorized": True,
            "arm_id": arm_id,
            "lineage_sha256": lineage_sha,
            "arm_contract_sha256": arm["arm_contract_sha256"],
            "runtime_policy_recipe_artifact": {
                "path": str(policy_artifact),
                "sha256": hashlib.sha256(policy_artifact.read_bytes()).hexdigest(),
            },
            "runtime_policy_recipe_sha256": "4" * 64,
            "dynamic_ready_binding_sha256": "5" * 64,
            "noise_std_type": "log",
            "configured_and_realized_init_noise_std": 0.02,
        }
    )
    recipe = _result(
        tmp_path / (arm_id + ".recipe.json"),
        stage="recipe",
        materialization=materialization,
        policy=policy,
    )
    oracle = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.ORACLE32_KIND,
            "diagnostic_unauthorized": True,
            "verdict": "PASS",
            "episodes": 32,
            "arm_id": arm_id,
            "lineage_sha256": lineage_sha,
            "arm_contract_sha256": arm["arm_contract_sha256"],
            "reward_contract_sha256": materialization["reward_contract_sha256"],
            "runtime_effective_reward_sha256": "3" * 64,
            "policy_contract_sha256": "4" * 64,
            "runtime_policy_recipe_sha256": "4" * 64,
            "actor_contract": launcher.ACTOR_CONTRACT,
            "actor_width": 211,
            "critic_contract": launcher.CRITIC_CONTRACT,
            "critic_width": 319,
            "trainability_contract": launcher.TRAINABILITY_CONTRACT,
            "seed": 0,
            "raw_oracle_sha256": "2" * 64,
        }
    )
    oracle_result = _result(
        tmp_path / (arm_id + ".oracle32.json"),
        stage="oracle32",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
    )
    smoke_result = _result(
        tmp_path / (arm_id + ".smoke.json"),
        stage="smoke",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
    )
    probe_result = _result(
        tmp_path / (arm_id + ".probe512.json"),
        stage="probe512",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
        predecessor={"stage": "smoke"},
    )
    scale_result = _result(
        tmp_path / (arm_id + ".scale4096.json"),
        stage="scale4096",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
        completion={
            "completion_exit_code": "0",
            "terminal_kind": "clean_completion",
            "terminal_exit_code": "0",
        },
        output_contract={
            "ppo_update_count": 5,
            "finite_model_save_interval": 1,
        },
        terminal_acceptance=_terminal_acceptance_fixture(),
    )
    return (
        materialize,
        recipe,
        oracle_result,
        smoke_result,
        probe_result,
        scale_result,
    )


def _case(tmp_path: Path, *, arm_id: str, stage: str, allow_colocation: bool = False):
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    lineage = _lineage(checkout)
    lineage_path = checkout / "a211_lineage.json"
    lineage_sha = _write(lineage_path, lineage)
    generated = _generated_chain(tmp_path, arm_id, lineage_sha)
    (
        materialize,
        recipe_result,
        oracle_result,
        smoke_result,
        probe_result,
        scale_result,
    ) = generated
    root = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / launcher.EXPERIMENT_NAME
    )
    root.mkdir(parents=True)
    namespace = root / (arm_id + "-" + stage)
    budget = launcher.BUDGETS[stage]
    four_grid_receipt_path = tmp_path / (arm_id + ".four-grid-scale4096.json")
    four_grid_receipt_sha = _write(
        four_grid_receipt_path, {"fixture": "four-grid-scale4096"}
    )
    spec = {
        "schema_version": launcher.SCHEMA_VERSION,
        "kind": launcher.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "isaac_python": str(python),
        },
        "arm_id": arm_id,
        "lineage": {"path": lineage_path.name, "sha256": lineage_sha},
        "arm_materialization": None if stage == "materialize" else materialize,
        "policy_recipe_materialization": (
            None if stage in ("materialize", "recipe") else recipe_result
        ),
        "oracle32_receipt": (
            oracle_result
            if stage
            in ("smoke", "probe512", "long512", "scale4096", "long4096")
            else None
        ),
        "predecessor_result": (
            smoke_result
            if stage == "probe512"
            else probe_result
            if stage == "long512"
            else scale_result
            if stage == "long4096"
            else None
        ),
        "four_grid_scale4096_receipt": (
            {
                "path": str(four_grid_receipt_path),
                "sha256": four_grid_receipt_sha,
            }
            if stage == "long4096"
            else None
        ),
        "stage": stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "wait_contract": launcher._wait_contract(),
        "gpu": {
            "index": 2,
            "uuid": "GPU-12345678",
            "owner": "Franco",
            "lock_path": "/tmp/hope_lean_queue_gpu2.lock",
            "require_empty": not allow_colocation,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }
    if allow_colocation:
        spec[launcher.COLOCATION_SPEC_KEY] = True
    spec_path = tmp_path / (arm_id + "-" + stage + ".spec.json")
    _write(spec_path, spec)
    return spec_path, spec, lineage


def _rewrite_physical_ready_hold(
    spec_path: Path,
    spec: dict,
    lineage: dict,
    mutate,
) -> None:
    """Rewrite the tracked hold and enclosing pins for semantic negatives."""

    checkout = Path(spec["source"]["checkout"])
    receipt_path = checkout / lineage["dynamic_ready_nominal_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("content_sha256")
    mutate(receipt)
    receipt["content_sha256"] = launcher.canonical_sha256(receipt)
    lineage["dynamic_ready_nominal_receipt"]["sha256"] = _write(
        receipt_path, receipt
    )
    lineage_path = checkout / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)


def _patch_plan_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda checkout, commit: {"checkout": str(checkout), "commit_sha": commit, "clean": True},
    )
    monkeypatch.setattr(launcher, "_runtime_sources", lambda checkout, commit: {})
    monkeypatch.setattr(
        launcher,
        "_dr_l0_manifest_binding",
        lambda checkout, commit, *, family, task_profile: dict(_DR_L0_BINDING),
    )
    monkeypatch.setattr(
        launcher._B, "_validate_runtime_asset_environment", lambda: {"kind": "test_runtime_assets"}
    )

    def verify(checkout, commit, pin, *, name):
        path = checkout / pin["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]
        return dict(pin), path

    monkeypatch.setattr(launcher._B, "_verify_tracked_file", verify)
    # [已删除 2026-08-05 安全门精简] 这里曾 stub 掉 _verify_frame0_artifact_source_commit /
    # _verify_frame0_probe_source_commit / _verify_commit_ancestor 三个 git 子进程校验。
    # 它们唯一的调用者是已退役的 _validate_retired_exact_frame0_lineage, 现在全仓零调用点,
    # 计划路径根本走不到, stub 已经是空转 —— 留着反而会在源函数被删时把上百个用例
    # 一起 AttributeError 打挂。原 stub 见 git 历史。

    def runtime_policy(*, path, checkout, lineage, arm):
        return _sealed(
            {
                "schema_version": 1,
                "kind": launcher.POLICY_MATERIALIZATION_KIND,
                "diagnostic_unauthorized": True,
                "arm_id": arm["arm_id"],
                "lineage_sha256": lineage["lineage_sha256"],
                "arm_contract_sha256": arm["arm_contract_sha256"],
                "runtime_policy_recipe_artifact": {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                "runtime_policy_recipe_sha256": "4" * 64,
                "dynamic_ready_binding_sha256": "5" * 64,
                "noise_std_type": "log",
                "configured_and_realized_init_noise_std": 0.02,
            }
        )

    monkeypatch.setattr(launcher, "_runtime_policy_materialization", runtime_policy)
    monkeypatch.setattr(
        launcher,
        "_audit_scale4096_terminal",
        lambda **_kwargs: copy.deepcopy(_terminal_acceptance_fixture()),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_four_grid_prelong_receipt",
        lambda value, *, checkout: {
            "artifact": copy.deepcopy(value),
            "schema_version": 1,
            "kind": launcher._Q.KIND,
            "status": "PASS",
            "content_sha256": "f" * 64,
        },
    )


def _raw_oracle_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qdes_manager_weight: float = -1.0,
    qdes_objective_weight: float | None = None,
    drift_raw_reward_sha: bool = False,
    hard_identity_mutation: dict | None = None,
) -> tuple[Path, dict]:
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="oracle32"
    )
    claim = launcher.build_plan(spec_path)["canonical_payload"]
    arm = claim["bundle"]["arm"]
    if qdes_objective_weight is None:
        qdes_objective_weight = arm["soft_weights"]["qdes_projection"]
    names_and_weights = {
        "death_penalty": arm["soft_weights"]["death_penalty"],
        "joint_limit": arm["soft_weights"]["joint_limit"],
        "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
    }
    terms = _required_effective_terms()
    for name, weight in sorted(names_and_weights.items()):
        terms.append(
            {
                "name": name,
                "callable": "fixture." + name,
                "weight": qdes_manager_weight if name == "qdes_projection_penalty" else weight,
                "params": (
                    {
                        "action_name": "joint_pos",
                        "shape_rate": 4.0,
                        "objective_weight": qdes_objective_weight,
                    }
                    if name == "qdes_projection_penalty"
                    else {}
                ),
            }
        )
    terms.sort(key=lambda term: term["name"])
    semantic = {"schema_version": 1, "terms": terms}
    reward_document = {
        **semantic,
        "sha256": launcher.canonical_sha256(semantic),
    }
    materialization = claim["materialization_inputs"]["arm_materialization"]
    reward_path = Path(materialization["runtime_effective_reward_artifact"]["path"])
    reward_file_sha = _write(reward_path, reward_document)
    materialization.update(
        {
            "runtime_effective_reward_artifact": {
                "path": str(reward_path),
                "sha256": reward_file_sha,
            },
            "runtime_effective_reward_sha256": reward_document["sha256"],
            "runtime_effective_reward_term_count": len(terms),
            "runtime_soft_weights": names_and_weights,
        }
    )

    hard_reward = copy.deepcopy(reward_document)
    if drift_raw_reward_sha:
        hard_reward["terms"].append(
            {
                "name": "unrelated_identity_drift",
                "callable": "fixture.unrelated_identity_drift",
                "weight": 0.0,
                "params": {},
            }
        )
        hard_reward["sha256"] = launcher.canonical_sha256(
            {"schema_version": 1, "terms": hard_reward["terms"]}
        )
    policy_materialization = claim["materialization_inputs"][
        "policy_recipe_materialization"
    ]
    ppo = arm["ppo"]
    hard_document = {
        "schema_version": 3,
        "target_mode": "action_ball",
        "actor_obs_contract": launcher.ACTOR_CONTRACT,
        "actor_obs_total_dim": launcher.ACTOR_WIDTH,
        "actor_obs_term_names": [
            name for name, _width in launcher.ACTOR_ORDERED_LAYOUT
        ],
        "actor_obs_term_dims": [
            width for _name, width in launcher.ACTOR_ORDERED_LAYOUT
        ],
        "critic_obs_contract": launcher.CRITIC_CONTRACT,
        "critic_obs_total_dim": launcher.CRITIC_WIDTH,
        "action_ball_211_trainability_contract": launcher.TRAINABILITY_CONTRACT,
        "task_wait_contract": launcher._hard_wait_contract(),
        "actor_obs_normalizer_identity": launcher.ACTOR_NORMALIZER_IDENTITY,
        "critic_obs_normalizer_identity": launcher.CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
        # 硬合同里带的是**本格真正跑的那一档**:噪声关 = DR-L0,噪声开 = DR-L0N。
        # 夹具从 arm 合同推导而不是抄死一档,否则换轴时又要"改测试让它变绿"。
        **(
            {"action_ball_dr_l0n": copy.deepcopy(_DR_L0N_PAYLOAD)}
            if arm["policy_observation_corruption"]
            else {"action_ball_dr_l0": copy.deepcopy(_DR_L0_PAYLOAD)}
        ),
        "action_ball_training": {
            "runtime": {
                "target_provider": {
                    "source": "online_solver",
                    "recipe": "current_lm",
                    "validity_mask": [True, True, True],
                    "target_observation_noise": False,
                    "immutable_tape": None,
                    "exact_question_answer_reuse": {"enabled": True},
                }
            }
        },
        "effective_reward_recipe": hard_reward,
        "action_ball_ppo_runner_recipe": {
            "sha256": policy_materialization["runtime_policy_recipe_sha256"],
            "recipe": {
                "algorithm": {
                    "schedule": ppo["schedule"],
                    "learning_rate": ppo["learning_rate"],
                    "desired_kl": 0.01,
                    "clip_param": 0.2,
                    "num_learning_epochs": 5,
                    "num_mini_batches": 4,
                    "entropy_coef": arm["entropy_coef"],
                },
                "policy": {
                    "actor_hidden_dims": arm["actor_hidden_dims"],
                    "critic_hidden_dims": arm["critic_hidden_dims"],
                    "init_noise_std": arm["init_noise_std"],
                    "noise_std_type": arm["noise_std_type"],
                },
            },
        },
    }
    if hard_identity_mutation:
        hard_document.update(hard_identity_mutation)
    checkout = Path(claim["spec"]["source"]["checkout"])
    runtime_dir = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs/rsl_rl"
        / launcher.EXPERIMENT_NAME
        / (
            "fixture_"
            + Path(claim["spec"]["namespace"]).name
            + "-DIAGNOSTIC_UNAUTHORIZED"
        )
    )
    hard_path = runtime_dir / "params/training_contract.json"
    hard_sha = _write(hard_path, hard_document)
    claim["runtime_sources"] = {
        "training entrypoint": {"sha256": "6" * 64},
        "A211 DR-L0 task profile": {"sha256": "7" * 64},
    }
    lineage = claim["bundle"]["lineage"]
    bindings = {
        "source_sha256": "6" * 64,
        "task_sha256": "7" * 64,
        "hard_contract_sha256": hard_sha,
        "reward_sha256": hard_reward["sha256"],
        "policy_sha256": policy_materialization["runtime_policy_recipe_sha256"],
        "policy_contract_sha256": policy_materialization[
            "runtime_policy_recipe_sha256"
        ],
        "dynamic_ready_sha256": policy_materialization[
            "dynamic_ready_binding_sha256"
        ],
        "dynamic_ready_artifact_sha256": lineage["dynamic_ready_artifact"][
            "sha256"
        ],
        "dynamic_ready_nominal_hold_sha256": lineage[
            "dynamic_ready_nominal_receipt"
        ]["sha256"],
        "manifest_sha256": lineage["action_manifest"]["sha256"],
        "motion_sha256": lineage["motion"]["sha256"],
        "target_provider_contract_sha256": launcher.canonical_sha256(
            hard_document["action_ball_training"]["runtime"]["target_provider"]
        ),
    }
    oracle_path = tmp_path / "raw-oracle32.json"
    _write(
        oracle_path,
        {
            "schema_version": 2,
            "kind": "action_ball_teacher_qdes_dynamic_oracle_v2",
            "diagnostic_unauthorized": True,
            "bindings": bindings,
            "completion": {
                "exact_strike_observed_nonterminal": 32,
                "pre_strike_or_same_step_unknown": 0,
                "control_steps": 32,
            },
            "phase_by_termination": {},
            "exact_strike": {},
            "capture_rejection": {},
            "measurement_contract": {},
            "safety_exposure": {
                "termination": {},
                "projection": {},
                "soft_limit": {},
                "reference_guard": {},
            },
            "teacher_qdes": {},
            "episodes": 32,
        },
    )

    class TrainingContract:
        @staticmethod
        def validate_schema3_contract_structure(document):
            return None

        @staticmethod
        def validate_action_ball_training_authorization(document):
            return True

        @staticmethod
        def action_ball_dr_l0_contract_payload():
            return copy.deepcopy(_DR_L0_PAYLOAD)

        @staticmethod
        def action_ball_dr_l0_contract_sha256():
            return _DR_L0_CONTRACT_SHA256

        @staticmethod
        def action_ball_dr_l0n_contract_payload():
            return copy.deepcopy(_DR_L0N_PAYLOAD)

    monkeypatch.setattr(
        launcher._OLD,
        "_load_training_contract_module",
        lambda checkout: TrainingContract,
    )
    monkeypatch.setattr(
        launcher._OLD,
        "_oracle32_acceptance_failures",
        lambda **kwargs: [],
    )
    return oracle_path, claim


def _flatten_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _flatten_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_strings(child)
    elif isinstance(value, str):
        yield value


def test_two_formal_a211_grid_cells_are_exact():
    # 2026-08-05 层级对齐(exp §5.6 第 3/7 条):death -300.0 -> -10.0。
    # 2026-08-05 第二轴改版(第二次,exp §5.6.2d):四格共用 fixed lr1e-4 与标准初始化 +
    # sigma 1.0 + scalar;唯一差异是本体感观测噪声开关(A0 关 / A1 开)。
    expected = {
        launcher.ARM_IDS[0]: (-10.0, -5.0, "metrics_only", "fixed", 1e-4),
        launcher.ARM_IDS[1]: (-10.0, -5.0, "metrics_only", "fixed", 1e-4),
    }
    assert tuple(launcher.ARMS) == launcher.ARM_IDS
    for arm_id, values in expected.items():
        arm = launcher._arm_contract(arm_id)
        assert (arm["soft_weights"]["death_penalty"], arm["soft_weights"]["qdes_limit"], arm["reference_guard_mode"], arm["ppo"]["schedule"], arm["ppo"]["learning_rate"]) == values
        assert arm["actor_hidden_dims"] == arm["critic_hidden_dims"] == [512, 256, 128]
        assert arm["entropy_coef"] == 0.01
    assert {
        (arm["ppo"]["schedule"], arm["ppo"]["learning_rate"])
        for arm in launcher.ARMS.values()
    } == {("fixed", 1.0e-4)}
    noise_off = launcher._arm_contract(launcher.ARM_IDS[0])
    noise_on = launcher._arm_contract(launcher.ARM_IDS[1])
    # 探索包本轮**不是**差异轴:两格逐字相同。
    for arm in (noise_off, noise_on):
        assert (
            arm["actor_init_mode"],
            arm["init_noise_std"],
            arm["noise_std_type"],
            arm["four_sigma_hard_inner_gate_applies"],
        ) == ("default", 1.0, "scalar", False)
    assert noise_off["policy_observation_corruption"] is False
    assert noise_on["policy_observation_corruption"] is True
    assert noise_off["dr_level_identity"] == launcher.DR_LEVEL_IDENTITY_OBS_NOISE_OFF
    assert noise_on["dr_level_identity"] == launcher.DR_LEVEL_IDENTITY_OBS_NOISE_ON
    assert noise_off["proprioceptive_observation_noise_channels"] is None
    assert (
        noise_on["proprioceptive_observation_noise_channels"]
        == launcher._F.PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS
    )
    # 任务通道两格都无噪:那会改支撑集,等于换题。
    assert noise_off["task_channel_observation_noise"] is False
    assert noise_on["task_channel_observation_noise"] is False
    # 对照实验前提:除观测噪声包之外逐字段相同。
    varying = {
        "arm_id",
        "four_grid_cell_id",
        "arm_contract_sha256",
        "observation_noise_axis",
        "policy_observation_corruption",
        "proprioceptive_observation_noise_channels",
        "task_channel_observation_noise",
        "dr_level_identity",
    }
    assert set(noise_off) == set(noise_on)
    for key in noise_off:
        if key in varying:
            continue
        assert noise_off[key] == noise_on[key], key
    assert noise_off["arm_contract_sha256"] != noise_on["arm_contract_sha256"]
    assert all(
        arm["reference_guard_mode"] == "metrics_only"
        and set(arm["soft_weights"].values()) == {-10.0, -5.0}
        for arm in launcher.ARMS.values()
    )
    manifest = launcher._isaac_four_grid_manifest()
    assert [
        row["cell_id"]
        for row in manifest["cells"]
        if row["task_family"] == "A211"
    ] == list(launcher.ARM_IDS)
    for arm_id in launcher.ARM_IDS:
        arm = launcher._arm_contract(arm_id)
        cell = launcher._four_grid_cell(arm_id, task_family="A211")
        assert arm["four_grid_cell_id"] == arm_id
        assert arm["isaac_four_grid_manifest_sha256"] == manifest["content_sha256"]
        assert arm["ppo"] == cell["ppo"]
        assert arm["ppo_adaptation_axis"] == cell["ppo_adaptation_axis"]
        assert arm["contact_sigma_adaptation"] is False


@pytest.mark.parametrize(
    "retired_arm_id",
    (
        "L0-corrected-metrics-fixed-lr1e4",
        "L1-corrected-metrics-fixed-lr1e3",
        "L2-corrected-metrics-adaptive-lr1e4",
        "L3-corrected-metrics-adaptive-lr1e3",
    ),
)
def test_historical_a211_arm_ids_are_not_formal_grid_selectors(retired_arm_id):
    with pytest.raises(launcher.LaunchRefused, match="two formal A211 grid cells"):
        launcher._arm_contract(retired_arm_id)


def test_required_a211_task_reward_contract_is_complete_and_exact():
    expected_stds = {
        "racket_position_coarse": 0.20,
        "racket_velocity_coarse": 1.50,
        "racket_normal_coarse": 1.0,
        "racket_position": 0.50,
        "racket_velocity": 3.0,
        "racket_normal": 2.10,
        "racket_position_precision": 0.075,
        "racket_velocity_precision": 0.50,
        "racket_normal_precision": 0.262,
    }
    required_task_terms = set(expected_stds) | {
        "strike_capture_bonus",
        "virtual_pass_net",
        "virtual_landing_dense",
        "virtual_landing",
    }
    assert required_task_terms.issubset(launcher.REQUIRED_EFFECTIVE_TERMS)
    assert {
        name: launcher.REQUIRED_EFFECTIVE_TERMS[name]["params"]["std"]
        for name in expected_stds
    } == expected_stds
    assert all(
        launcher.REQUIRED_EFFECTIVE_TERMS[name]["callable"].startswith(
            "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
        )
        for name in required_task_terms
    )


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("callable", "pkg.drifted", "callable differs"),
        ("weight", 0.0, "not positive"),
        ("std", 0.21, "gate identity differs"),
    ),
)
def test_required_a211_task_reward_contract_fails_closed(field, value, match):
    terms = _required_effective_terms()
    term = next(row for row in terms if row["name"] == "racket_position_coarse")
    if field == "std":
        term["params"]["std"] = value
    else:
        term[field] = value
    with pytest.raises(launcher.LaunchRefused, match=match):
        launcher._require_effective_learnability_terms(terms)


def test_a211_v2_actor_layout_binds_localizer_and_body_gyro_exact_slices():
    identity = launcher._actor_layout_identity()
    assert identity["schema_version"] == 2
    assert identity["total_dim"] == 211
    assert identity["ordered_terms"][0] == {
        "name": "actual_base_pose_lin_vel_world",
        "width": 12,
        "slice": [0, 12],
    }
    assert identity["ordered_terms"][1] == {
        "name": "base_ang_vel_body",
        "width": 3,
        "slice": [12, 15],
    }
    assert identity["sensor_sources"]["actual_base_pose_lin_vel_world"][
        "angular_velocity_included"
    ] is False
    assert identity["sensor_sources"]["base_ang_vel_body"]["producer"] == (
        "mdp.action_ball_base_ang_vel_body"
    )
    assert launcher.canonical_sha256(
        {key: value for key, value in identity.items() if key != "content_sha256"}
    ) == identity["content_sha256"]


def test_a211_v2_lineage_rejects_same_width_pre_imu_layout(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    old = copy.deepcopy(lineage["actor_layout_identity"])
    old["ordered_terms"][0]["name"] = "stage1_base_state_world"
    old["ordered_terms"][0]["width"] = 15
    old["ordered_terms"][0]["slice"] = [0, 15]
    old["ordered_terms"].pop(1)
    old.pop("content_sha256")
    old["content_sha256"] = launcher.canonical_sha256(old)
    lineage["actor_layout_identity"] = old
    lineage_path = Path(spec["source"]["checkout"]) / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


def test_a211_safe_wait_uses_split_ready_then_public_teacher_bridge(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    checkout = Path(_spec["source"]["checkout"])
    physical = json.loads(
        (checkout / lineage["dynamic_ready_artifact"]["path"]).read_text()
    )["physical_ready"]
    frame0 = json.loads(
        (checkout / lineage["teacher_frame0_artifact"]["path"]).read_text()
    )["frame0"]
    assert physical["joint_vel_radps"] == [0.0] * 31
    assert physical["joint_pos_rad"] != frame0["joint_pos_rad"]
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    plan_lineage = payload["bundle"]["lineage"]
    assert "frame0_exact_receipt" not in plan_lineage
    authority = plan_lineage["split_ready_reset_wait_authority"]
    unsigned = dict(authority)
    claim_sha256 = unsigned.pop("claim_sha256")
    assert authority["kind"] == launcher.SPLIT_READY_RESET_WAIT_GATE_KIND
    assert claim_sha256 == launcher.canonical_sha256(unsigned)
    assert authority["hidden_wait_required_policy_steps"] == 25
    assert authority["observed_policy_steps"] == 60
    assert authority["observed_physics_steps"] == 240
    assert authority["time_to_teacher_start_at_reveal_s"] == pytest.approx(
        0.6923759904781779
    )
    assert authority["initial_center_timing_authority"]["timing_mode"] == (
        "a_online_solver"
    )
    assert authority["bridge_learning_signal"] == "dense_mimic_after_task_reveal"
    assert authority["passive_hold_after_reveal_required"] is False


def test_initial_center_timing_rejects_old_interior_tick92_receipt(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    lineage = _lineage(checkout)
    receipt_path = checkout / lineage["initial_center_task_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("canonical_sha256")
    receipt.update(
        {
            "sampling_stratum": "interior",
            "birth_sampling_stratum": "interior",
            "time_to_contact_tick": 92,
            "time_to_contact_s": 1.84,
            "pre_swing_wait_s": 1.84 - receipt["scaled_t_hit_s"],
        }
    )
    receipt["canonical_sha256"] = launcher.canonical_sha256(receipt)
    manifest = json.loads(
        (checkout / lineage["action_manifest"]["path"]).read_text(encoding="utf-8")
    )
    with pytest.raises(launcher.LaunchRefused, match="literal all-zero center"):
        launcher._initial_center_timing_authority(
            receipt=receipt,
            receipt_pin=lineage["initial_center_task_receipt"],
            action_manifest=manifest,
            action_manifest_pin=lineage["action_manifest"],
            motion_sha256=lineage["motion"]["sha256"],
            family="A",
        )


def test_a_family_initial_center_wait_law_is_unchanged(tmp_path):
    """2026-08-06 C 对齐运行时公式那次改动, A 族的门一字未变。

    A 一直执行的就是运行时那条 ``wait = time_to_contact_s - scaled_t_hit_s``。这里把
    A 的判定式、``derivation``/``timing_mode`` 字面量, 以及"改了 wait 就拒"三件事一起
    钉死, 这样以后任何人再动这个函数, A 侧走样会立刻变红。
    """

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    lineage = _lineage(checkout)
    receipt = json.loads(
        (checkout / lineage["initial_center_task_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (checkout / lineage["action_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    kwargs = {
        "receipt_pin": lineage["initial_center_task_receipt"],
        "action_manifest": manifest,
        "action_manifest_pin": lineage["action_manifest"],
        "motion_sha256": lineage["motion"]["sha256"],
        "family": "A",
    }
    timing = launcher._initial_center_timing_authority(
        receipt=receipt, **kwargs
    )
    assert timing["family"] == "A"
    assert timing["timing_mode"] == "a_online_solver"
    assert timing["derivation"] == "time_to_contact_s_minus_scaled_t_hit_s"
    assert timing["initial_center_time_to_teacher_start_at_reveal_s"] == (
        receipt["time_to_contact_s"] - receipt["scaled_t_hit_s"]
    )
    assert timing["initial_center_time_to_teacher_start_at_reveal_s"] == (
        pytest.approx(0.6923759904781779)
    )

    for wait in (
        receipt["time_to_contact_s"] - receipt["reference_t_hit_s"],
        receipt["pre_swing_wait_s"] + 1.0e-15,
        receipt["pre_swing_wait_s"] - 1.0e-15,
    ):
        tampered = dict(receipt)
        tampered.pop("canonical_sha256")
        tampered["pre_swing_wait_s"] = wait
        tampered["canonical_sha256"] = launcher.canonical_sha256(tampered)
        with pytest.raises(
            launcher.LaunchRefused, match="timing derivation differs"
        ):
            launcher._initial_center_timing_authority(
                receipt=tampered, **kwargs
            )


def test_a211_split_ready_rejects_equality_or_nonzero_reset_velocity(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    _spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    checkout = Path(spec["source"]["checkout"])
    dynamic_path = checkout / lineage["dynamic_ready_artifact"]["path"]
    dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
    nominal = json.loads(
        (checkout / lineage["dynamic_ready_nominal_receipt"]["path"]).read_text()
    )
    frame0 = json.loads(
        (checkout / lineage["teacher_frame0_artifact"]["path"]).read_text()
    )["frame0"]
    initial_center = json.loads(
        (checkout / lineage["initial_center_task_receipt"]["path"]).read_text()
    )
    manifest = json.loads(
        (checkout / lineage["action_manifest"]["path"]).read_text()
    )
    timing = launcher._initial_center_timing_authority(
        receipt=initial_center,
        receipt_pin=lineage["initial_center_task_receipt"],
        action_manifest=manifest,
        action_manifest_pin=lineage["action_manifest"],
        motion_sha256=lineage["motion"]["sha256"],
        family="A",
    )
    kwargs = {
        "nominal": nominal,
        "dynamic_pin": lineage["dynamic_ready_artifact"],
        "nominal_pin": lineage["dynamic_ready_nominal_receipt"],
        "teacher_frame0": frame0,
        "motion_sha256": lineage["motion"]["sha256"],
        "initial_center_timing_authority": timing,
    }
    assert launcher._split_ready_reset_wait_semantics(
        dynamic=dynamic, **kwargs
    )["observed_policy_steps"] == 60
    equal = copy.deepcopy(dynamic)
    equal["physical_ready"]["joint_pos_rad"] = list(frame0["joint_pos_rad"])
    equal["physical_ready"]["root_pos_w_m"] = list(frame0["root_pos_w_m"])
    equal["physical_ready"]["root_quat_wxyz"] = list(frame0["root_quat_wxyz"])
    equal.pop("content_sha256")
    equal["content_sha256"] = launcher.canonical_sha256(equal)
    with pytest.raises(launcher.LaunchRefused, match="separated teacher"):
        launcher._split_ready_reset_wait_semantics(dynamic=equal, **kwargs)
    moving = copy.deepcopy(dynamic)
    moving["physical_ready"]["joint_vel_radps"][0] = 1.0e-6
    moving.pop("content_sha256")
    moving["content_sha256"] = launcher.canonical_sha256(moving)
    with pytest.raises(launcher.LaunchRefused, match="separated teacher"):
        launcher._split_ready_reset_wait_semantics(dynamic=moving, **kwargs)


def test_a211_prelong_exec_environment_is_scale_only_and_recipe_bound():
    reward_sha = "a" * 64
    expected = {
        launcher.REWARD_PPO_ECONOMY_ENABLE_ENV: "1",
        launcher.PRELONG_SEMANTICS_ENABLE_ENV: "1",
        launcher.PRELONG_REWARD_RECIPE_SHA_ENV: reward_sha,
    }
    materialized = {"runtime_effective_reward_sha256": reward_sha}
    assert launcher._prelong_semantics_exec_environment(
        "scale4096", materialized
    ) == expected
    # 关键回归:materialize / recipe / oracle32 三个阶段的 arm_materialization 里
    # **没有** runtime_effective_reward_sha256 这个键(reward 还没产出)。这个 helper
    # 必须在取值之前就按 stage 短路,否则每条流水线的第一站都会在 exec 前 KeyError。
    planned = {
        "schema_version": 1,
        "kind": launcher.MATERIALIZATION_KIND,
        "arm_id": launcher.ARM_IDS[0],
    }
    assert "runtime_effective_reward_sha256" not in planned
    for stage in launcher.BUDGETS:
        if stage != "scale4096":
            assert launcher._prelong_semantics_exec_environment(
                stage, planned
            ) == {}
            assert launcher._prelong_semantics_exec_environment(stage, None) == {}
    # scale4096 上缺键 / 类型不对仍然是硬错,不许静默放行。
    for bad in ({}, None, True, {"runtime_effective_reward_sha256": True}):
        with pytest.raises(launcher.LaunchRefused):
            launcher._prelong_semantics_exec_environment("scale4096", bad)


def test_a211_update_profile_switch_is_exact_claim_bound_and_non_speed(
    tmp_path, monkeypatch
):
    assert launcher._update_profile_exec_environment({}) == {}
    assert launcher._update_profile_exec_environment(
        {launcher.UPDATE_PROFILE_ENV: "0"}
    ) == {launcher.UPDATE_PROFILE_ENV: "0"}
    assert launcher._update_profile_exec_environment(
        {launcher.UPDATE_PROFILE_ENV: "1"}
    ) == {launcher.UPDATE_PROFILE_ENV: "1"}
    for invalid in ("", "true", "2"):
        with pytest.raises(launcher.LaunchRefused, match="exactly 0 or 1"):
            launcher._update_profile_exec_environment(
                {launcher.UPDATE_PROFILE_ENV: invalid}
            )

    _patch_plan_environment(monkeypatch)
    monkeypatch.setenv(launcher.UPDATE_PROFILE_ENV, "1")
    _spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    normalized = launcher._validate_spec(spec)
    lineage = {
        **lineage,
        "teacher_frame0_artifact_content_sha256": "1" * 64,
        "split_ready_reset_wait_authority": {"claim_sha256": "2" * 64},
    }
    profile = launcher._output_contract(normalized, lineage)["update_profile"]
    assert profile["forwarded_value"] == "1"
    assert profile["mode"] == "profile_on_attribution_only"
    assert profile["speed_evidence_eligible"] is False
    assert profile["gpu_kernel_attribution_claimed"] is False
    _path, colocated, colocated_lineage = _case(
        tmp_path / "colocated-profile",
        arm_id=launcher.ARM_IDS[0],
        stage="scale4096",
        allow_colocation=True,
    )
    with pytest.raises(launcher.LaunchRefused, match="exclusive GPU claim"):
        launcher._output_contract(
            launcher._validate_spec(colocated),
            {
                **colocated_lineage,
                "teacher_frame0_artifact_content_sha256": "1" * 64,
                "split_ready_reset_wait_authority": {"claim_sha256": "2" * 64},
            },
        )

    monkeypatch.setenv(launcher.UPDATE_PROFILE_ENV, "0")
    off = launcher._output_contract(normalized, lineage)["update_profile"]
    assert off["forwarded_value"] == "0"
    assert off["mode"] == "explicit_profiler_off"
    assert off != profile


def test_a211_formal_lineage_has_no_bundle_or_immutable_tape(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert "bundle" not in lineage
    assert "immutable_tape" not in lineage
    assert payload["bundle"]["lineage"]["runtime_target_contract"] == (
        launcher._runtime_target_contract()
    )


def test_a211_accepts_60_tick_hold_but_rejects_less_than_hidden_wait(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    checkout = Path(spec["source"]["checkout"])
    receipt_path = checkout / lineage["dynamic_ready_nominal_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("content_sha256")
    assert launcher.build_plan(spec_path)["canonical_payload"]["bundle"][
        "lineage"
    ]["split_ready_reset_wait_authority"]["observed_policy_steps"] == 60
    receipt.update({
        "requested_duration_s": 0.48,
        "completed_duration_s": 0.48,
        "completed_policy_steps": 24,
        "completed_physics_steps": 96,
    })
    receipt["content_sha256"] = launcher.canonical_sha256(receipt)
    lineage["dynamic_ready_nominal_receipt"]["sha256"] = _write(
        receipt_path, receipt
    )
    lineage_path = checkout / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize("missing", launcher.FULL_ACTIVE_TERMINATIONS)
def test_a211_rejects_physical_ready_hold_missing_any_active_termination(
    tmp_path, monkeypatch, missing
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    checkout = Path(spec["source"]["checkout"])
    receipt_path = checkout / lineage["dynamic_ready_nominal_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("content_sha256")
    receipt["active_terminations"].remove(missing)
    receipt["content_sha256"] = launcher.canonical_sha256(receipt)
    lineage["dynamic_ready_nominal_receipt"]["sha256"] = _write(
        receipt_path, receipt
    )
    lineage_path = checkout / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "reference_termination", launcher.PROHIBITED_HOLD_REFERENCE_TERMINATIONS
)
def test_a211_rejects_physical_ready_hold_with_reference_envelope_active(
    tmp_path, monkeypatch, reference_termination
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    _rewrite_physical_ready_hold(
        spec_path,
        spec,
        lineage,
        lambda row: row["active_terminations"].append(reference_termination),
    )
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize("fraction", (0.0, 0.5, None))
def test_a211_rejects_physical_ready_hold_without_full_both_feet_contact(
    tmp_path, monkeypatch, fraction
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    _rewrite_physical_ready_hold(
        spec_path,
        spec,
        lineage,
        lambda row: row.__setitem__("both_feet_contact_fraction", fraction),
    )
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


def test_a211_rejects_resealed_reported_hard_gap_drift(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    _rewrite_physical_ready_hold(
        spec_path,
        spec,
        lineage,
        lambda row: row["joint_safety_telemetry"].__setitem__(
            "final_minimum_hard_gap_rad", 0.5
        ),
    )
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize("stage,budget", list(launcher.BUDGETS.items()))
def test_stage_budgets_are_code_owned(stage, budget):
    assert launcher.BUDGETS[stage] == budget


def test_formal_a211_scale_and_long_are_4096_envs():
    assert {
        stage: launcher.BUDGETS[stage][0]
        for stage in ("scale4096", "long4096")
    } == {"scale4096": 4096, "long4096": 4096}


def test_a211_leaf_freezes_stable_plant_static_sigma_and_coarse_widths():
    leaf = (
        SCRIPT.parent.parent
        / "cfg/task/HOPEPingPongActionBallA211VendorV2N1DRL0Learnability.yaml"
    ).read_text(encoding="utf-8")
    parent = (
        SCRIPT.parent.parent
        / "cfg/task/HOPEPingPongActionBallA211VendorV2N1Learnability.yaml"
    ).read_text(encoding="utf-8")
    assert "HOPEPingPongActionBallA211VendorV2N1Learnability@_here_" in leaf
    for marker in (
        "stable_ready_plant: true",
        "startup_physics_material: false",
        "startup_joint_default_pos: false",
        "policy_observation_corruption: false",
    ):
        assert marker in leaf
    for marker in (
        "num_envs: 4096",
        "adaptive_sigma: false",
        "adaptive_sigma_monotonic: false",
        "adaptive_sigma_normal: false",
        "racket_position_coarse_std: 0.20",
        "racket_velocity_coarse_std: 1.50",
    ):
        assert marker in parent
    assert launcher.REWARD_MATERIALIZATION_PROFILE == (
        "measured_vendor_v2_n1_static_v1"
    )


def test_plan_claim_is_a211_fresh_and_denies_retired_lineage(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, arm_id=launcher.ARM_IDS[1], stage="long4096")
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert payload["fresh_only"] is True
    assert payload["single_gpu"] is True
    assert payload["max_compute_pids_on_physical_gpu"] == 2
    assert payload["minimum_free_memory_mib"] == 8192
    assert payload["gpu_default_empty"] is True
    assert payload["vendor_v2_colocation_opt_in"] is False
    assert payload["bundle"]["lineage"]["actor_contract"] == launcher.ACTOR_CONTRACT
    assert payload["bundle"]["normalizers"] == launcher._normalizer_contract()
    assert payload["bundle"]["curriculum_scope"] == (
        launcher._curriculum_scope_contract()
    )
    assert payload["bundle"]["continuation_stop_gate"]["iter500_quantitative_threshold_status"] == "UNSET"
    assert payload["bundle"]["continuation_stop_gate"]["scale4096_required_for_long4096"] is True
    assert payload["materialization_inputs"]["predecessor_result"][
        "terminal_attestation"
    ]["completion"]["terminal_kind"] == "clean_completion"
    assert payload["output_contract"]["speed_benchmark_eligible"] is False
    assert payload["output_contract"]["rate_evidence_eligible"] is False
    assert payload["output_contract"]["rate_evidence_isolation"] == (
        "excluded_no_matched_abba_speed_stage"
    )
    flattened = "\n".join(_flatten_strings(payload)).lower()
    for retired in ("target_recipe", "target_validity_mask", "l194"):
        assert retired not in flattened


@pytest.mark.parametrize(
    "field,retired",
    (
        ("actor_contract", "action_ball_a225"),
        ("actor_contract", "action_ball_a210"),
        ("actor_width", 225),
        ("actor_width", 210),
        ("critic_contract", "action_ball_a225_critic_v1"),
        ("critic_contract", "action_ball_a210_critic_v1"),
        ("critic_width", 318),
        ("trainability_contract", "action_ball_a225_fixed_question_learnability_v1"),
        ("trainability_contract", "action_ball_a210_fixed_question_learnability_v1"),
        ("task_profile", "HOPEPingPongActionBallA225VendorV2N1Learnability"),
        ("gym_task", "HOPE-PingPong-ActionBall-A210Learnability-AgibotA3-v0"),
    ),
)
def test_structurally_resealed_retired_lineage_is_rejected(
    tmp_path, monkeypatch, field, retired
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    lineage[field] = retired
    lineage_path = Path(spec["source"]["checkout"]) / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "mutation",
    (
        {"actor_obs_contract": "action_ball_a225"},
        {"actor_obs_total_dim": 225},
        {"actor_obs_term_names": ["stage1_base_state_world"] + [
            name for name, _width in launcher.ACTOR_ORDERED_LAYOUT[2:]
        ]},
        {"critic_obs_contract": "action_ball_a210_critic_v1"},
        {"critic_obs_total_dim": 318},
        {
            "action_ball_211_trainability_contract":
                "action_ball_a225_fixed_question_learnability_v1"
        },
        {"actor_obs_normalizer_identity": "action_ball_a210_actor_norm_v1"},
        {"critic_obs_normalizer_identity": "action_ball_a225_critic_norm_v1"},
    ),
)
def test_structurally_resealed_retired_hard_contract_is_rejected(
    tmp_path, monkeypatch, mutation
):
    raw, claim = _raw_oracle_fixture(
        tmp_path, monkeypatch, hard_identity_mutation=mutation
    )
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_raw_oracle32(raw, claim=claim)


def test_retired_vocabulary_scan_treats_hashes_as_opaque():
    launcher._assert_no_retired_contract(
        {"spec_file_sha256": "0" * 12 + "c225" + "0" * 48},
        name="opaque digest",
    )
    with pytest.raises(launcher.LaunchRefused, match="retired ABI/arm token"):
        launcher._assert_no_retired_contract(
            {"obs_mode": "action_ball_c225"}, name="semantic value"
        )


def test_training_argv_pins_a211_lineage_bootstrap_and_optimizer(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, arm_id=launcher.ARM_IDS[1], stage="probe512")
    argv = launcher.build_plan(spec_path)["canonical_payload"]["training_argv"]
    for exact in (
        "task=HOPEPingPongActionBallA211VendorV2N1DRL0Learnability",
        "task.actor_obs_contract=action_ball_a211",
        "algo.policy.actor_hidden_dims=[512,256,128]",
        "algo.policy.critic_hidden_dims=[512,256,128]",
        "algo.algorithm.schedule=fixed",
        "algo.algorithm.learning_rate=0.0001",
        "+task.racket.reference_guard_mode=metrics_only",
        "task.domain_rand.stable_ready_plant=true",
        "task.racket.adaptive_sigma=false",
        "task.racket.adaptive_sigma_monotonic=false",
        "task.racket.adaptive_sigma_normal=false",
        "task.actions.control_step_action_delay_min=0",
        "task.actions.control_step_action_delay_max=0",
        "action_ball_dynamic_ready_bootstrap=true",
        # A1 = 标准 rsl_rl 初始化 + sigma 1.0 + scalar(exp §5.6.2c)
        "algo.policy.init_noise_std=1.0",
        "algo.policy.noise_std_type=scalar",
        "action_ball_actor_init_mode=default",
    ):
        assert exact in argv
    joined = "\n".join(argv)
    assert joined.count("task.domain_rand.stable_ready_plant=true") == 1
    assert "action_ball_policy_contract_sha256=" in joined
    assert "expected_effective_reward_recipe_sha256=" + "3" * 64 in argv
    assert "action_ball_manifest_sha256=" in joined
    assert "target_recipe" not in joined and "validity_mask" not in joined


@pytest.mark.parametrize(
    "arm_index,corruption",
    (
        (0, "false"),
        (1, "true"),
    ),
)
def test_training_argv_carries_each_cell_observation_noise_package(
    tmp_path, monkeypatch, arm_index, corruption
):
    """人话:两个格子的 argv 只在那一个布尔上不同,探索包与 PPO 三项完全一样。"""

    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, arm_id=launcher.ARM_IDS[arm_index], stage="probe512"
    )
    argv = launcher.build_plan(spec_path)["canonical_payload"]["training_argv"]
    # 四格相同的探索包。
    assert "algo.policy.init_noise_std=1.0" in argv
    assert "algo.policy.noise_std_type=scalar" in argv
    assert "action_ball_actor_init_mode=default" in argv
    assert "algo.algorithm.schedule=fixed" in argv
    assert "algo.algorithm.learning_rate=0.0001" in argv
    # 唯一的注册差异轴:整包 DR 元组写进 argv,只有最后这个布尔随格变。
    assert "task.domain_rand.stable_ready_plant=true" in argv
    assert "task.domain_rand.startup_physics_material=false" in argv
    assert "task.domain_rand.startup_joint_default_pos=false" in argv
    assert "task.domain_rand.policy_observation_corruption=%s" % corruption in argv
    # 标准初始化格必须与一个显式 bootstrap 同时给,否则 train.py 侧 fail-closed。
    assert "action_ball_dynamic_ready_bootstrap=true" in argv
    joined = "\n".join(argv)
    assert joined.count("algo.policy.init_noise_std=") == 1
    assert joined.count("algo.policy.noise_std_type=") == 1
    assert joined.count("action_ball_actor_init_mode=") == 1
    assert joined.count("task.domain_rand.policy_observation_corruption=") == 1
    # 任务/目标通道的噪声与这根轴无关,两格都必须保持关闭。
    assert "task.racket.action_ball_target_observation_noise=false" in argv


def test_a211_runtime_source_manifest_pins_the_a211_contract_source():
    assert launcher.A211_CONTRACT_SOURCE.endswith(
        "/action_ball_a211_trainability.py"
    )
    assert "action_ball_225_trainability.py" not in {
        path for path, _label in launcher.RUNTIME_SOURCE_PATHS
    }


def test_materialize_stage_publishes_and_binds_runtime_effective_reward(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert "+n1_vendor_sigma_profile=" + launcher.REWARD_MATERIALIZATION_PROFILE in payload[
        "training_argv"
    ]
    assert any(
        value.startswith("+action_ball_effective_reward_recipe_output_path=")
        for value in payload["training_argv"]
    )
    assert payload["output_contract"]["boot_marker"] == (
        "ACTION_BALL_EFFECTIVE_REWARD_RECIPE_MATERIALIZED_JSON"
    )

    arm = payload["bundle"]["arm"]
    names_and_weights = {
        "death_penalty": arm["soft_weights"]["death_penalty"],
        "joint_limit": arm["soft_weights"]["joint_limit"],
        "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
    }
    terms = _required_effective_terms()
    for name, weight in sorted(names_and_weights.items()):
        terms.append(
            {
                "name": name,
                "callable": "fixture." + name,
                "weight": -1.0 if name == "qdes_projection_penalty" else weight,
                "params": (
                    {"objective_weight": weight}
                    if name == "qdes_projection_penalty"
                    else {}
                ),
            }
        )
    terms.sort(key=lambda term: term["name"])
    semantic = {"schema_version": 1, "terms": terms}
    document = {**semantic, "sha256": launcher.canonical_sha256(semantic)}
    output = Path(payload["output_contract"]["effective_reward_recipe"])
    output.parent.mkdir(parents=True)
    _write(output, document)
    runtime = launcher._runtime_reward_materialization(
        path=output,
        planned=payload["materialization_inputs"]["arm_materialization"],
        arm=arm,
    )
    assert runtime["runtime_effective_reward_sha256"] == document["sha256"]
    assert runtime["runtime_soft_weights"] == names_and_weights

    next(term for term in document["terms"] if term["name"] == "death_penalty")["weight"] -= 1.0
    semantic = {"schema_version": 1, "terms": document["terms"]}
    document["sha256"] = launcher.canonical_sha256(semantic)
    output.unlink()
    _write(output, document)
    with pytest.raises(launcher.LaunchRefused, match="soft weights differ"):
        launcher._runtime_reward_materialization(
            path=output,
            planned=payload["materialization_inputs"]["arm_materialization"],
            arm=arm,
        )


def test_materialize_stage_refuses_missing_required_effective_term(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    arm = payload["bundle"]["arm"]
    terms = _required_effective_terms()
    terms = [term for term in terms if term["name"] != "virtual_landing"]
    for name, weight in sorted(
        {
            "death_penalty": arm["soft_weights"]["death_penalty"],
            "joint_limit": arm["soft_weights"]["joint_limit"],
            "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
            "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
        }.items()
    ):
        terms.append(
            {
                "name": name,
                "callable": "fixture." + name,
                "weight": -1.0 if name == "qdes_projection_penalty" else weight,
                "params": {"objective_weight": weight}
                if name == "qdes_projection_penalty"
                else {},
            }
        )
    terms.sort(key=lambda term: term["name"])
    semantic = {"schema_version": 1, "terms": terms}
    document = {**semantic, "sha256": launcher.canonical_sha256(semantic)}
    output = Path(payload["output_contract"]["effective_reward_recipe"])
    output.parent.mkdir(parents=True)
    _write(output, document)
    with pytest.raises(launcher.LaunchRefused, match="required effective term is absent: virtual_landing"):
        launcher._runtime_reward_materialization(
            path=output,
            planned=payload["materialization_inputs"]["arm_materialization"],
            arm=arm,
        )


def test_raw_oracle_accepts_projection_manager_minus_one_with_arm_objective(
    tmp_path, monkeypatch
):
    raw, claim = _raw_oracle_fixture(tmp_path, monkeypatch)
    receipt = launcher._validate_raw_oracle32(raw, claim=claim)
    assert receipt["runtime_effective_reward_sha256"] == claim[
        "materialization_inputs"
    ]["arm_materialization"]["runtime_effective_reward_sha256"]


@pytest.mark.parametrize(
    "binding",
    (
        "tape_canonical_sha256",
        "tape_base_question_sha256",
        "tape_target_producer_sha256",
        "tape_target_column_sha256",
    ),
)
def test_raw_oracle_rejects_retired_tape_binding_injection(
    tmp_path, monkeypatch, binding
):
    raw, claim = _raw_oracle_fixture(tmp_path, monkeypatch)
    document = json.loads(raw.read_text(encoding="utf-8"))
    document["bindings"][binding] = "f" * 64
    _write(raw, document)
    with pytest.raises(launcher.LaunchRefused, match="bindings keys differ"):
        launcher._validate_raw_oracle32(raw, claim=claim)


@pytest.mark.parametrize(
    "mutation",
    (
        {"qdes_objective_weight": -0.25},
        {"qdes_manager_weight": -0.5},
    ),
)
def test_raw_oracle_rejects_projection_objective_or_manager_weight_drift(
    tmp_path, monkeypatch, mutation
):
    raw, claim = _raw_oracle_fixture(tmp_path, monkeypatch, **mutation)
    with pytest.raises(launcher.LaunchRefused, match="soft weights differ"):
        launcher._validate_raw_oracle32(raw, claim=claim)


def test_raw_oracle_rejects_reward_sha_drift_from_revalidated_materialization(
    tmp_path, monkeypatch
):
    raw, claim = _raw_oracle_fixture(
        tmp_path, monkeypatch, drift_raw_reward_sha=True
    )
    with pytest.raises(launcher.LaunchRefused, match="lineage bindings differ"):
        launcher._validate_raw_oracle32(raw, claim=claim)


def test_recipe_stage_materializes_policy_before_oracle32(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    recipe_path, _spec, _ = _case(
        tmp_path / "recipe", arm_id=launcher.ARM_IDS[1], stage="recipe"
    )
    recipe = launcher.build_plan(recipe_path)["canonical_payload"]
    assert recipe["policy_recipe_materialization_only"] is True
    assert recipe["materialization_inputs"]["policy_recipe_materialization"] is None
    assert recipe["output_contract"]["boot_marker"] == (
        "ACTION_BALL_POLICY_RECIPE_MATERIALIZED"
    )
    assert (
        "task.racket.action_ball_policy_contract_sha256="
        + launcher.RECIPE_SENTINEL_POLICY_SHA256
    ) in recipe["training_argv"]
    assert any(
        value.startswith("action_ball_policy_recipe_output_path=")
        for value in recipe["training_argv"]
    )
    assert "policy_contract_sha256" not in recipe[
        "materialization_inputs"
    ]["arm_materialization"]

    oracle_path, _spec, _ = _case(
        tmp_path / "oracle", arm_id=launcher.ARM_IDS[1], stage="oracle32"
    )
    oracle = launcher.build_plan(oracle_path)["canonical_payload"]
    assert (
        "task.racket.action_ball_policy_contract_sha256=" + "4" * 64
    ) in oracle["training_argv"]
    assert not any(
        value.endswith("=" + launcher.RECIPE_SENTINEL_POLICY_SHA256)
        and "policy_contract_sha256" in value
        for value in oracle["training_argv"]
    )


def test_recipe_accepts_exact_legacy_reward_only_result_without_trusting_its_planned_policy(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="recipe"
    )
    result_path = Path(spec["arm_materialization"]["path"])
    result = json.loads(result_path.read_text())
    materialization = dict(result["arm_materialization"])
    materialization.pop("content_sha256")
    materialization["policy_contract_sha256"] = "9" * 64
    result["arm_materialization"] = _sealed(materialization)
    result.pop("policy_recipe_materialization")
    result.pop("content_sha256")
    spec["arm_materialization"]["sha256"] = _write(result_path, _sealed(result))
    _write(spec_path, spec)
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    normalized = payload["materialization_inputs"]["arm_materialization"]
    assert "policy_contract_sha256" not in normalized
    assert (
        "task.racket.action_ball_policy_contract_sha256="
        + launcher.RECIPE_SENTINEL_POLICY_SHA256
    ) in payload["training_argv"]


def test_runtime_policy_recipe_is_exact_arm_owned_and_no_observed_sha_is_baked(
    tmp_path, monkeypatch
):
    arm = launcher._arm_contract(launcher.ARM_IDS[1])
    lineage = {
        "lineage_sha256": "1" * 64,
        "motion": {"path": "motion", "sha256": "2" * 64},
        "dynamic_ready_artifact": {"path": "dynamic", "sha256": "3" * 64},
        "dynamic_ready_nominal_receipt": {
            "path": "nominal",
            "sha256": "4" * 64,
        },
    }
    runner = {
        "schema_version": 2,
        "runner": {
            "empirical_normalization": True,
            "init_at_random_ep_len": False,
        },
        "policy": {
            "actor_hidden_dims": arm["actor_hidden_dims"],
            "critic_hidden_dims": arm["critic_hidden_dims"],
            "init_noise_std": arm["init_noise_std"],
            "noise_std_type": arm["noise_std_type"],
        },
        "algorithm": {"entropy_coef": arm["entropy_coef"], **arm["ppo"]},
        "policy_initialization": {"fixture": True},
    }
    document = {
        "schema_version": 1,
        "kind": "action_ball_shared_ready_policy_recipe_materialization_v1",
        "action_count": 1,
        "action_order": ["take_061_unit04_bh"],
        "policy_contract_sha256": launcher.canonical_sha256(runner),
        "action_ball_ppo_runner_recipe": {
            "schema_version": 1,
            "sha256": launcher.canonical_sha256(runner),
            "recipe": runner,
        },
        "policy_bootstrap": {"fixture": True},
    }
    path = tmp_path / "policy.json"
    _write(path, document)

    def validate(
        value,
        *,
        checkout,
        bundle,
        expected_noise_std_type,
        expected_init_noise_std,
    ):
        # The borrowed validator must be asked for this arm's exploration
        # package, not the retired log/0.02 one.
        assert expected_noise_std_type == arm["noise_std_type"]
        assert expected_init_noise_std == arm["init_noise_std"]
        return {
            "artifact": dict(value),
            "policy_contract_sha256": document["policy_contract_sha256"],
            "dynamic_ready_binding_sha256": "5" * 64,
            "noise_std_type": expected_noise_std_type,
            "configured_and_realized_init_noise_std": expected_init_noise_std,
        }

    monkeypatch.setattr(launcher._OLD, "_validate_policy_materialization", validate)
    receipt = launcher._runtime_policy_materialization(
        path=path, checkout=tmp_path, lineage=lineage, arm=arm
    )
    assert receipt["runtime_policy_recipe_sha256"] == launcher.canonical_sha256(
        runner
    )
    assert "3a3" not in SCRIPT.read_text(encoding="utf-8")
    assert "f344" not in SCRIPT.read_text(encoding="utf-8")

    document["action_ball_ppo_runner_recipe"]["recipe"]["algorithm"][
        "learning_rate"
    ] = 0.5
    path.unlink()
    _write(path, document)
    with pytest.raises(launcher.LaunchRefused, match="selected A211 PPO arm"):
        launcher._runtime_policy_materialization(
            path=path, checkout=tmp_path, lineage=lineage, arm=arm
        )


def _policy_materialization_bytes(
    tmp_path, *, noise_std_type, init_noise_std, bootstrap_schema_version
):
    """One materialized policy recipe carrying an exact exploration package."""

    portable = {"portable_bootstrap": True}
    runner = {
        "policy": {
            "init_noise_std": init_noise_std,
            "noise_std_type": noise_std_type,
        },
        "policy_initialization": portable,
    }
    policy_sha = launcher.canonical_sha256(runner)
    bootstrap = {
        "schema_version": bootstrap_schema_version,
        "action_count": 1,
        "action_order": [launcher._OLD.ACTION_ID],
        "ready_source": {"identity": {"binding_sha256": "b" * 64}},
        "initialization": {
            "noise_std_type": noise_std_type,
            "init_noise_std": init_noise_std,
            "required_realized_init_noise_std": init_noise_std,
        },
    }
    document = {
        "schema_version": 1,
        "kind": "action_ball_shared_ready_policy_recipe_materialization_v1",
        "action_count": 1,
        "action_order": [launcher._OLD.ACTION_ID],
        "policy_contract_sha256": policy_sha,
        "action_ball_ppo_runner_recipe": {
            "schema_version": 1,
            "sha256": policy_sha,
            "recipe": runner,
        },
        "policy_bootstrap": bootstrap,
    }
    path = tmp_path / ("policy_%s.json" % noise_std_type)
    digest = _write(path, document)
    return {"path": str(path), "sha256": digest}, portable


def test_borrowed_policy_validator_takes_the_arm_s_sigma_not_the_retired_one(
    tmp_path, monkeypatch
):
    """The A211 recipe stage was unreachable: two gates demanded different sigmas.

    ``_runtime_policy_materialization`` first hands the emitted recipe to the
    borrowed vendor-v2 validator and then checks the same recipe against the
    selected arm.  The borrowed validator used to hardcode the retired log-std
    package (``schema_version`` 3 / ``noise_std_type`` "log" / sigma 0.02) while
    every A211 arm declares the standard scalar sigma 1.0 package, so no emitted
    recipe could satisfy both and the stage could never pass.  Nothing here
    loosens the equality -- the expected package just comes from the hashed arm
    contract instead of a stale literal.
    """

    import types

    stub_binding = {"binding_sha256": "b" * 64}

    def _stub_contract_module(checkout):
        return types.SimpleNamespace(
            load_action_ball_dynamic_ready_runtime_binding=(
                lambda **kwargs: stub_binding
            ),
            validate_action_ball_policy_bootstrap=(
                lambda bootstrap, *, expected_action_count: None
            ),
            action_ball_policy_bootstrap_scientific_identity=(
                lambda bootstrap, *, repo_root: {"portable_bootstrap": True}
            ),
        )

    monkeypatch.setattr(
        launcher._OLD, "_load_training_contract_module", _stub_contract_module
    )
    bundle = {
        "core": {
            "dynamic_ready": {
                "artifact": {"path": "a", "sha256": "1" * 64},
                "nominal_hold_receipt": {"path": "b", "sha256": "2" * 64},
            }
        },
        "motion": {"path": "m", "sha256": "3" * 64},
    }

    # Every A211 arm's declared package now passes the borrowed validator.
    for arm_id in launcher.ARM_IDS:
        arm = launcher._arm_contract(arm_id)
        assert (arm["noise_std_type"], arm["init_noise_std"]) == ("scalar", 1.0)
        pin, _portable = _policy_materialization_bytes(
            tmp_path / arm_id,
            noise_std_type=arm["noise_std_type"],
            init_noise_std=arm["init_noise_std"],
            bootstrap_schema_version=2,
        )
        validated = launcher._OLD._validate_policy_materialization(
            pin,
            checkout=tmp_path,
            bundle=bundle,
            expected_noise_std_type=arm["noise_std_type"],
            expected_init_noise_std=arm["init_noise_std"],
        )
        assert validated["noise_std_type"] == arm["noise_std_type"]
        assert (
            validated["configured_and_realized_init_noise_std"]
            == arm["init_noise_std"]
        )

    # The retired log/0.02 package is still the default, so the vendor-v2
    # callers that never pass the new keywords keep their exact old gate.
    log_pin, _ = _policy_materialization_bytes(
        tmp_path / "log",
        noise_std_type="log",
        init_noise_std=0.02,
        bootstrap_schema_version=3,
    )
    defaulted = launcher._OLD._validate_policy_materialization(
        log_pin, checkout=tmp_path, bundle=bundle
    )
    assert defaulted["noise_std_type"] == "log"
    assert defaulted["configured_and_realized_init_noise_std"] == 0.02

    # The gate is still exact in both directions.
    with pytest.raises(launcher._OLD.LaunchRefused, match="dynamic-ready N1 contract"):
        launcher._OLD._validate_policy_materialization(
            log_pin,
            checkout=tmp_path,
            bundle=bundle,
            expected_noise_std_type="scalar",
            expected_init_noise_std=1.0,
        )
    scalar_pin, _ = _policy_materialization_bytes(
        tmp_path / "scalar_default",
        noise_std_type="scalar",
        init_noise_std=1.0,
        bootstrap_schema_version=2,
    )
    with pytest.raises(launcher._OLD.LaunchRefused, match="dynamic-ready N1 contract"):
        launcher._OLD._validate_policy_materialization(
            scalar_pin, checkout=tmp_path, bundle=bundle
        )

    # The bootstrap ABI is derived from the sigma, never taken on trust.
    wrong_abi, _ = _policy_materialization_bytes(
        tmp_path / "wrong_abi",
        noise_std_type="scalar",
        init_noise_std=1.0,
        bootstrap_schema_version=3,
    )
    with pytest.raises(launcher._OLD.LaunchRefused, match="dynamic-ready N1 contract"):
        launcher._OLD._validate_policy_materialization(
            wrong_abi,
            checkout=tmp_path,
            bundle=bundle,
            expected_noise_std_type="scalar",
            expected_init_noise_std=1.0,
        )
    with pytest.raises(launcher._OLD.LaunchRefused, match="scalar or log"):
        launcher._OLD._validate_policy_materialization(
            scalar_pin,
            checkout=tmp_path,
            bundle=bundle,
            expected_noise_std_type="uniform",
            expected_init_noise_std=1.0,
        )


def test_full_stage_chain_is_enforced(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    for stage in (
        "materialize",
        "recipe",
        "oracle32",
        "scale4096",
        "long4096",
        "smoke",
        "probe512",
        "long512",
    ):
        spec_path, _spec, _lineage = _case(tmp_path / stage, arm_id=launcher.ARM_IDS[0], stage=stage)
        payload = launcher.build_plan(spec_path)["canonical_payload"]
        assert payload["spec"]["stage"] == stage
        if stage == "long4096":
            assert payload["materialization_inputs"][
                "four_grid_scale4096_receipt"
            ]["status"] == "PASS"
        else:
            assert payload["materialization_inputs"][
                "four_grid_scale4096_receipt"
            ] is None
    spec_path, spec, _ = _case(
        tmp_path / "missing-policy", arm_id=launcher.ARM_IDS[0], stage="oracle32"
    )
    spec["policy_recipe_materialization"] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="policy recipe receipt"):
        launcher.build_plan(spec_path)
    spec_path, spec, _ = _case(tmp_path / "missing", arm_id=launcher.ARM_IDS[0], stage="probe512")
    spec["predecessor_result"] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="completed smoke"):
        launcher.build_plan(spec_path)

    spec_path, spec, _ = _case(
        tmp_path / "long4096-missing-scale",
        arm_id=launcher.ARM_IDS[0],
        stage="long4096",
    )
    spec["predecessor_result"] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="completed scale4096"):
        launcher.build_plan(spec_path)

    spec_path, spec, _ = _case(
        tmp_path / "long4096-missing-all-four",
        arm_id=launcher.ARM_IDS[0],
        stage="long4096",
    )
    spec["four_grid_scale4096_receipt"] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="four-grid scale4096"):
        launcher.build_plan(spec_path)

    spec_path, spec, _ = _case(
        tmp_path / "scale4096-extra-all-four",
        arm_id=launcher.ARM_IDS[0],
        stage="scale4096",
    )
    spec["four_grid_scale4096_receipt"] = {
        "path": str(tmp_path / "foreign-aggregate.json"),
        "sha256": "f" * 64,
    }
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="four-grid scale4096"):
        launcher.build_plan(spec_path)


def test_cross_arm_or_oracle_content_drift_is_rejected(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="smoke")
    oracle_path = Path(spec["oracle32_receipt"]["path"])
    outer = json.loads(oracle_path.read_text())
    outer["oracle32_receipt"]["arm_id"] = launcher.ARM_IDS[1]
    receipt = dict(outer["oracle32_receipt"])
    receipt.pop("content_sha256")
    outer["oracle32_receipt"] = _sealed(receipt)
    outer.pop("content_sha256")
    outer = _sealed(outer)
    spec["oracle32_receipt"]["sha256"] = _write(oracle_path, outer)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="binding differs"):
        launcher.build_plan(spec_path)


def test_policy_recipe_artifact_sha_drift_is_rejected(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="oracle32"
    )
    recipe_result = json.loads(
        Path(spec["policy_recipe_materialization"]["path"]).read_text()
    )
    artifact = Path(
        recipe_result["policy_recipe_materialization"][
            "runtime_policy_recipe_artifact"
        ]["path"]
    )
    artifact.write_text("drifted-policy\n", encoding="utf-8")
    with pytest.raises(
        launcher.LaunchRefused, match="runtime policy materialization binding"
    ):
        launcher.build_plan(spec_path)


def test_default_empty_gpu_and_scale_long_colocation_claim_scope(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    default_path, _, _ = _case(tmp_path / "default", arm_id=launcher.ARM_IDS[0], stage="materialize")
    default = launcher.build_plan(default_path)["canonical_payload"]
    assert default["spec"]["gpu"]["require_empty"] is True
    assert default["output_contract"]["speed_benchmark_eligible"] is False
    assert default["output_contract"]["rate_evidence_eligible"] is False
    assert default["output_contract"]["rate_evidence_isolation"] == (
        "excluded_no_matched_abba_speed_stage"
    )

    opted_path, _, _ = _case(tmp_path / "opted", arm_id=launcher.ARM_IDS[0], stage="long4096", allow_colocation=True)
    opted = launcher.build_plan(opted_path)["canonical_payload"]
    assert opted["spec"]["gpu"]["require_empty"] is False
    assert opted["vendor_v2_colocation_opt_in"] is True
    assert opted["output_contract"]["speed_benchmark_eligible"] is False
    assert opted["output_contract"]["rate_evidence_eligible"] is False
    assert opted["output_contract"]["rate_evidence_isolation"] == "excluded_colocated_diagnostic"
    assert opted["output_contract"]["colocated_stage"] == "long4096"
    assert opted["output_contract"]["max_compute_processes_per_gpu"] == 2
    assert opted["output_contract"]["colocation_result_scope"] == "training_diagnostic_only"

    _path, colocated_scale, _lineage = _case(
        tmp_path / "colocated-scale",
        arm_id=launcher.ARM_IDS[0],
        stage="scale4096",
        allow_colocation=True,
    )
    scale_output = launcher.build_plan(_path)["canonical_payload"][
        "output_contract"
    ]
    assert scale_output["speed_benchmark_eligible"] is False
    assert scale_output["rate_evidence_eligible"] is False
    assert scale_output["colocated_stage"] == "scale4096"

    exclusive_scale_path, _, _ = _case(
        tmp_path / "exclusive-scale",
        arm_id=launcher.ARM_IDS[0],
        stage="scale4096",
    )
    exclusive_scale = launcher.build_plan(exclusive_scale_path)[
        "canonical_payload"
    ]["output_contract"]
    assert exclusive_scale["speed_benchmark_eligible"] is False
    assert exclusive_scale["rate_evidence_eligible"] is False
    assert exclusive_scale["rate_evidence_isolation"] == (
        "excluded_scale_finite_gate"
    )
    assert exclusive_scale["deferred_matched_speed_measurement"] == {
        "status": "DEFERRED_UNTIL_FINITE",
        "implemented_by_this_launcher": False,
        "workload_kind": "fixed_question_exact_answer_cache_consumer",
        "cold_first_distinct_question_solver_calls_required": 1,
        "steady_identical_question_solver_calls_required": 0,
        "novel_question_producer_unit": "seconds_per_4096_novel_questions",
        "producer_evidence_must_be_separate": True,
        "num_envs": 4096,
        "steps_per_env": 24,
        "warmup_updates": 10,
        "minimum_measured_updates": 50,
        "isolation": "exclusive_single_process_same_gpu",
        "abba_order": ["current_A", "current_C", "current_C", "current_A"],
        "main_timing_mode": "profiler_off",
        "profile_run_is_separate_non_speed_evidence": True,
        "scale4096_is_speed_or_rate_evidence": False,
    }

    _path, forbidden, _lineage = _case(
        tmp_path / "forbidden",
        arm_id=launcher.ARM_IDS[0],
        stage="recipe",
        allow_colocation=True,
    )
    with pytest.raises(launcher.LaunchRefused, match="scale4096/long4096"):
        launcher._validate_spec(forbidden)


def test_colocation_gpu_validation_is_cross_bound_and_fail_closed(tmp_path, monkeypatch):
    _, raw_spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="long4096", allow_colocation=True)
    spec = launcher._validate_spec(raw_spec)
    query = lambda index, uuid: {
        "total_memory_mib": 24576,
        "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
        "processes": [{"pid": 99}],
        "nvidia_smi_path": "/usr/bin/nvidia-smi",
        "nvidia_smi_sha256": "3" * 64,
    }
    monkeypatch.setattr(launcher, "_query_gpu_processes", query)
    monkeypatch.setattr(launcher, "_live_reservations", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launcher,
        "_validate_runtime_gpu_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(launcher.LaunchRefused("unknown GPU co-resident")),
    )
    with pytest.raises(launcher.LaunchRefused, match="unknown GPU"):
        launcher._verify_gpu_admission(spec, phase="pre_launch", current_namespace=None)

    monkeypatch.setattr(launcher, "_query_gpu_processes", lambda index, uuid: {**query(index, uuid), "free_memory_mib": launcher._A.MIN_VENDOR_V2_FREE_MEMORY_MIB - 1, "processes": []})
    with pytest.raises(launcher.LaunchRefused, match="below conservative headroom"):
        launcher._verify_gpu_admission(spec, phase="pre_launch", current_namespace=None)


def test_scale4096_executes_as_completion_stage_and_emits_natural_exit_result(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    monkeypatch.setenv(launcher.UPDATE_PROFILE_ENV, "1")
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[1], stage="scale4096")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    lock_file = tmp_path / "gpu.lock"
    monkeypatch.setattr(
        launcher,
        "_open_gpu_shared_lock",
        lambda path: os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600),
    )
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    phases = []

    def admission(spec, *, phase, current_namespace, require_current_compute=False, **kwargs):
        phases.append((phase, require_current_compute))
        return {"phase": phase}

    monkeypatch.setattr(launcher, "_verify_gpu_admission", admission)
    monkeypatch.setattr(
        launcher,
        "_write_reservation",
        lambda spec, digest: {"registry_path": lock_file, "claim": digest},
    )
    monkeypatch.setattr(launcher, "_release_reservation", lambda handle: None)

    def run(*args, **kwargs):
        state = Path(kwargs["env"]["KIT_BOOT_STATE_FILE"])
        state.write_text(
            "completion_exit_code=0\n"
            "terminal_kind=clean_completion\n"
            "terminal_exit_code=0\n",
            encoding="utf-8",
        )
        assert kwargs["env"]["KIT_WAIT_FOR_COMPLETION"] == "1"
        assert kwargs["env"][launcher.UPDATE_PROFILE_ENV] == "1"
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    result = launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    assert result["stage"] == "scale4096"
    assert result["completion"] == {
        "completion_exit_code": "0",
        "terminal_kind": "clean_completion",
        "terminal_exit_code": "0",
    }
    assert phases == [("pre_launch", False), ("post_completion", False)]
    assert result["output_contract"]["update_profile"]["forwarded_value"] == "1"
    assert result["output_contract"]["speed_benchmark_eligible"] is False
    assert Path(result["namespace"], "launch_result.json").is_file()


def _scale4096_terminal_artifacts(tmp_path: Path):
    checkout = tmp_path / "checkpoint-checkout"
    wbt = checkout / launcher._B.WBT_RELATIVE
    root = wbt / "logs" / "rsl_rl" / launcher.EXPERIMENT_NAME
    root.mkdir(parents=True)
    namespace = tmp_path / launcher.EXPERIMENT_NAME / "scale-terminal-fixture"
    namespace.mkdir(parents=True)
    run_dir = root / (
        "2026-08-04_12-34-56_"
        + namespace.name
        + "-DIAGNOSTIC_UNAUTHORIZED"
    )
    run_dir.mkdir()
    claim_sha = "a" * 64
    # 跑满 5 个 update 之后落盘的末位是 model_4.pt / iter=4(RSL-RL 的迭代变量在
    # 循环体内取 0..N-1)。字面量由 test_action_ball_4096x5_terminal_index.py 钉住。
    checkpoint_path = run_dir / "model_4.pt"
    checkpoint = {
        "iter": 4,
        "infos": {"training_launch_claim_sha256": claim_sha},
        "model_state_dict": {"weight": _FakeTensor([1.0, 1.0])},
        "optimizer_state_dict": {"state": {0: {"momentum": _FakeTensor([1.0, 1.0])}}},
        "obs_norm_state_dict": {"running_mean": _FakeTensor([0.0, 0.0, 0.0])},
        "privileged_obs_norm_state_dict": {"running_var": _FakeTensor([1.0] * 4)},
    }
    checkpoint_path.write_bytes(b"trusted exact Pod checkpoint fixture\n")

    lines = ["[INFO] Task: fixture | experiment: fixture | log: %s" % run_dir]
    for update in range(5):
        lines.extend(
            (
                "HOPE_JOINT_SAFETY_UPDATE_JSON="
                + json.dumps(
                    {
                        "event": "hope_joint_safety_diagnostic_compact_update",
                        "schema_version": 1,
                        "status": (
                            "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
                        ),
                        "ppo_update": update,
                        "counter_totals": {"actual_hard_edge_events": 0},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON="
                + json.dumps(
                    {
                        "event": "action_ball_actual_joint_forbidden_diagnostic_update",
                        "schema_version": 2,
                        "ppo_update": update,
                        "enabled": True,
                        "total_hard_terminal_count": 0,
                        "physx_control_position_limits": {
                            "enabled": True,
                            "by_joint": [
                                {
                                    "joint": "joint_00",
                                    "sides": {
                                        "lower": {"nonfinite_readback_observed": False},
                                        "upper": {"nonfinite_readback_observed": False},
                                    },
                                }
                            ],
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON="
                + json.dumps(
                    {
                        "event": "hope_reward_safety_transition_update",
                        "schema_version": 2,
                        "ppo_update": update,
                        "coverage": "complete_update",
                        "terminal_transitions": [],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
                + json.dumps(
                    {
                        "event": "hope_exact_behavior_update",
                        "schema_version": 1,
                        "ppo_update": update,
                        "counters": {
                            "task_wait_started_count": 12,
                            "task_reveal_reached_count": 10,
                            "termination_reason_base_fell_tilt_count": 0,
                            "termination_reason_base_fell_tilt_hidden_wait_count": 0,
                            "termination_reason_base_fell_tilt_revealed_pre_strike_count": 0,
                            "termination_reason_base_fell_tilt_post_strike_count": 0,
                            "termination_reason_base_too_low_count": 0,
                            "termination_reason_base_too_low_hidden_wait_count": 0,
                            "termination_reason_base_too_low_revealed_pre_strike_count": 0,
                            "termination_reason_base_too_low_post_strike_count": 0,
                            "termination_reason_robot_hit_table_count": 0,
                            "termination_reason_robot_hit_table_hidden_wait_count": 0,
                            "termination_reason_robot_hit_table_revealed_pre_strike_count": 0,
                            "termination_reason_robot_hit_table_post_strike_count": 0,
                            "ready_nonfinite_value_count": 0,
                            "strike_window_entry_racket_target_distance_nonfinite_count": 0,
                            "virtual_contact_nonfinite_reject_count": 0,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        lines.extend(_prelong_marker_lines(update))
    log_path = namespace / "run.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checkout, namespace, claim_sha, checkpoint_path, checkpoint, log_path


def _audit_terminal_fixture(
    checkout, namespace, claim, checkpoint, monkeypatch, *, load_error=None
):
    _FakeTorch.checkpoint = checkpoint
    _FakeTorch.load_error = load_error
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    return launcher._audit_scale4096_terminal(
        checkout=checkout,
        namespace=namespace,
        launch_claim_sha256=claim,
    )


def test_scale4096_terminal_checkpoint_and_safety_gate_accepts_valid_case(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, _checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    acceptance = _audit_terminal_fixture(
        checkout, namespace, claim, _checkpoint, monkeypatch
    )
    assert acceptance["checkpoint"]["path"] == str(checkpoint_path)
    # 跑满 5 个 update 之后落盘的末位是 model_4.pt / iter=4。
    assert acceptance["checkpoint"]["embedded_iteration"] == 4
    assert acceptance["checkpoint"]["filename_iteration"] == 4
    assert acceptance["checkpoint"]["load_mode"] == "torch_weights_only"
    assert acceptance["checkpoint"]["all_tensors_finite"] is True
    assert acceptance["prelong_gate"]["semantic_update_count"] == 5
    assert acceptance["prelong_gate"]["launch_claim_sha256"] == claim
    assert acceptance["prelong_gate"]["run_log_sha256"] == acceptance[
        "run_log"
    ]["sha256"]
    assert acceptance["prelong_gate"]["finite_model_sha256"] == acceptance[
        "checkpoint"
    ]["sha256"]
    assert acceptance["prelong_gate"]["gate"]["status"] == "PASS"
    assert acceptance["safety_counters"] == {
        "observed_ppo_updates": 5,
        "actual_hard_edge_event_count": 0,
        "actual_hard_terminal_count": 0,
        "joint_qdes_forbidden_terminal_count": 0,
        "joint_actual_forbidden_terminal_count": 0,
        "strict_hard_termination_count": 0,
        "table_contact_count": 0,
        "nonfinite_count": 0,
        "base_fell_tilt_terminal_count": 0,
        "base_too_low_terminal_count": 0,
        "physical_fall_by_reason_phase": {
            reason: {phase: 0 for phase in launcher.PHYSICAL_FALL_PHASES}
            for reason in launcher.PHYSICAL_FALL_REASONS
        },
        "table_contact_by_phase": {
            phase: 0 for phase in launcher.PHYSICAL_FALL_PHASES
        },
        "task_wait_started_by_update": [12] * 5,
        "task_wait_started_count": 60,
        "task_reveal_reached_by_update": [10] * 5,
        "task_reveal_reached_count": 50,
    }


@pytest.mark.parametrize("mode", ("missing", "duplicate"))
def test_scale4096_prelong_gate_requires_exactly_five_semantic_markers(
    tmp_path, monkeypatch, mode
):
    checkout, namespace, claim, _checkpoint_path, checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    prefix = launcher._S.PRELONG_SEMANTICS_MARKER_PREFIX
    lines = log_path.read_text(encoding="utf-8").splitlines()
    marker = next(
        line
        for line in lines
        if line.startswith(prefix) and '"ppo_update":4' in line
    )
    if mode == "missing":
        lines.remove(marker)
    else:
        lines.append(marker)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(launcher.LaunchRefused, match="exactly 5"):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


@pytest.mark.parametrize(
    "mutation,allow_nan,match",
    (
        (
            lambda row: row["task_invalid"].__setitem__(
                "task_reward_weighted_sum", 0.25
            ),
            False,
            "task_valid=0 leaked",
        ),
            (
                lambda row: row["reward_groups"][2].__setitem__(
                    "eligible_denominator", 10
                ),
                False,
                "strike-group denominator",
        ),
        (
            lambda row: row["reward_groups"][0].__setitem__(
                "weighted_sum", float("nan")
            ),
            True,
            "JSON is invalid",
        ),
    ),
)
def test_scale4096_prelong_gate_rejects_wait_denominator_or_nonfinite_drift(
    tmp_path, monkeypatch, mutation, allow_nan, match
):
    checkout, namespace, claim, _checkpoint_path, checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    _rewrite_prelong_marker(
        log_path, 2, mutation, allow_nan=allow_nan
    )
    with pytest.raises(launcher.LaunchRefused, match=match):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


def test_scale4096_terminal_checkpoint_gate_rejects_missing_checkpoint(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, _checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    checkpoint_path.unlink()
    with pytest.raises(launcher.LaunchRefused, match="checkpoint.*missing"):
        _audit_terminal_fixture(
            checkout, namespace, claim, _checkpoint, monkeypatch
        )


def test_scale4096_terminal_checkpoint_gate_rejects_corrupt_checkpoint(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, _checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    checkpoint_path.write_bytes(b"not a PyTorch checkpoint\n")
    with pytest.raises(launcher.LaunchRefused, match="weights-only load"):
        _audit_terminal_fixture(
            checkout,
            namespace,
            claim,
            _checkpoint,
            monkeypatch,
            load_error=ValueError("corrupt checkpoint"),
        )


@pytest.mark.parametrize(
    "state_key",
    (
        "model_state_dict",
        "optimizer_state_dict",
        "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    ),
)
def test_scale4096_terminal_checkpoint_gate_rejects_nonfinite_tensor(
    tmp_path, state_key, monkeypatch
):
    checkout, namespace, claim, checkpoint_path, checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    checkpoint[state_key] = {"bad": _FakeTensor([float("nan")])}
    with pytest.raises(launcher.LaunchRefused, match="non-finite tensor"):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


@pytest.mark.parametrize(
    "wrong_iteration",
    # 5 = 2026-08-07 之前那道差一格的门自己要的数字(预算而不是末位);
    # 3 = 真少跑了一格。两者都必须被拒。
    (5, 3),
)
def test_scale4096_terminal_checkpoint_gate_rejects_wrong_iteration(
    tmp_path, monkeypatch, wrong_iteration
):
    checkout, namespace, claim, checkpoint_path, checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    checkpoint["iter"] = wrong_iteration
    with pytest.raises(launcher.LaunchRefused, match="iteration/launch-claim"):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


@pytest.mark.parametrize("mode", ("missing", "mismatched"))
def test_scale4096_terminal_checkpoint_gate_requires_the_launch_claim(
    tmp_path, monkeypatch, mode
):
    """存档必须自陈它是哪一次发射产出的:键缺失或对不上都要拒。"""

    checkout, namespace, claim, _checkpoint_path, checkpoint, _log = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    if mode == "missing":
        checkpoint["infos"].pop("training_launch_claim_sha256")
    else:
        checkpoint["infos"]["training_launch_claim_sha256"] = "b" * 64
    with pytest.raises(launcher.LaunchRefused, match="iteration/launch-claim"):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


def test_scale4096_terminal_gate_rejects_missing_safety_counters(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, _checkpoint_path, _checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    rewritten = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        prefix = "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
        if line.startswith(prefix):
            row = json.loads(line[len(prefix) :])
            row["counters"].pop("virtual_contact_nonfinite_reject_count")
            line = prefix + json.dumps(row, sort_keys=True, separators=(",", ":"))
        rewritten.append(line)
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    with pytest.raises(launcher.LaunchRefused, match="nonfinite counters.*missing"):
        _audit_terminal_fixture(
            checkout, namespace, claim, _checkpoint, monkeypatch
        )


@pytest.mark.parametrize(
    "counter_kind",
    ("actual_hard", "joint_qdes", "joint_actual", "nonfinite"),
)
def test_scale4096_terminal_gate_rejects_observed_safety_event(
    tmp_path, counter_kind, monkeypatch
):
    checkout, namespace, claim, _checkpoint_path, checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    rewritten = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        prefix, separator, payload = line.partition("=")
        if not separator or not prefix.startswith("HOPE_"):
            rewritten.append(line)
            continue
        row = json.loads(payload)
        if row.get("ppo_update") == 0:
            if counter_kind == "actual_hard" and prefix == "HOPE_JOINT_SAFETY_UPDATE_JSON":
                row["counter_totals"]["actual_hard_edge_events"] = 1
            elif counter_kind == "table" and prefix == "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON":
                row["terminal_transitions"] = [
                    {"termination_terms": ["robot_hit_table"]}
                ]
            elif (
                counter_kind in {"joint_qdes", "joint_actual"}
                and prefix == "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON"
            ):
                row["terminal_transitions"] = [
                    {
                        "termination_terms": [
                            "joint_qdes_forbidden"
                            if counter_kind == "joint_qdes"
                            else "joint_actual_forbidden"
                        ]
                    }
                ]
            elif counter_kind == "nonfinite" and prefix == "HOPE_EXACT_BEHAVIOR_UPDATE_JSON":
                row["counters"]["ready_nonfinite_value_count"] = 1
        rewritten.append(
            prefix + "=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        )
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    with pytest.raises(
        launcher.LaunchRefused, match="implementation counters are nonzero"
    ):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


def test_scale4096_terminal_gate_reports_physical_fall_without_rejecting_finite(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, _checkpoint_path, checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    rewritten = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        prefix, separator, payload = line.partition("=")
        if not separator or not prefix.startswith("HOPE_"):
            rewritten.append(line)
            continue
        row = json.loads(payload)
        if row.get("ppo_update") == 0:
            if prefix == "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON":
                row["terminal_transitions"] = [
                    {"termination_terms": ["base_fell_tilt"]}
                ]
            elif prefix == "HOPE_EXACT_BEHAVIOR_UPDATE_JSON":
                row["counters"]["termination_reason_base_fell_tilt_count"] = 1
                row["counters"][
                    "termination_reason_base_fell_tilt_revealed_pre_strike_count"
                ] = 1
        rewritten.append(
            prefix + "=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        )
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    accepted = _audit_terminal_fixture(
        checkout, namespace, claim, checkpoint, monkeypatch
    )

    assert accepted["safety_counters"]["base_fell_tilt_terminal_count"] == 1
    assert accepted["safety_counters"]["strict_hard_termination_count"] == 0
    assert accepted["prelong_gate"]["gate"]["status"] == "PASS"


def test_scale4096_terminal_gate_reports_table_by_phase_without_rejecting_finite(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, _checkpoint_path, checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    rewritten = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        prefix, separator, payload = line.partition("=")
        if not separator or not prefix.startswith("HOPE_"):
            rewritten.append(line)
            continue
        row = json.loads(payload)
        if row.get("ppo_update") == 0:
            if prefix == "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON":
                row["terminal_transitions"] = [
                    {"termination_terms": ["robot_hit_table"]}
                ]
            elif prefix == "HOPE_EXACT_BEHAVIOR_UPDATE_JSON":
                row["counters"]["termination_reason_robot_hit_table_count"] = 1
                row["counters"][
                    "termination_reason_robot_hit_table_revealed_pre_strike_count"
                ] = 1
        rewritten.append(
            prefix + "=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        )
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    accepted = _audit_terminal_fixture(
        checkout, namespace, claim, checkpoint, monkeypatch
    )

    assert accepted["safety_counters"]["table_contact_count"] == 1
    assert accepted["safety_counters"]["strict_hard_termination_count"] == 0
    table = accepted["prelong_gate"]["gate"]["survival_denominators"][
        "robot_hit_table"
    ]
    assert table["by_phase"]["revealed_pre_strike"] == 1
    assert table["acceptance_threshold"] is None


def test_scale4096_terminal_gate_rejects_fall_reason_phase_nonconservation(
    tmp_path, monkeypatch
):
    checkout, namespace, claim, _checkpoint_path, checkpoint, log_path = (
        _scale4096_terminal_artifacts(tmp_path)
    )
    rewritten = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        prefix, separator, payload = line.partition("=")
        if not separator or not prefix.startswith("HOPE_"):
            rewritten.append(line)
            continue
        row = json.loads(payload)
        if (
            row.get("ppo_update") == 0
            and prefix == "HOPE_EXACT_BEHAVIOR_UPDATE_JSON"
        ):
            row["counters"]["termination_reason_base_too_low_count"] = 1
        rewritten.append(
            prefix + "=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        )
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    with pytest.raises(launcher.LaunchRefused, match="reason-by-phase counters"):
        _audit_terminal_fixture(
            checkout, namespace, claim, checkpoint, monkeypatch
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda result: result.__setitem__("completion", None),
        lambda result: result["completion"].__setitem__("terminal_exit_code", "9"),
        lambda result: result["output_contract"].__setitem__("ppo_update_count", 4),
        lambda result: result["output_contract"].__setitem__(
            "finite_model_save_interval", 100
        ),
    ),
)
def test_long4096_rejects_launch_accepted_without_exact_scale_terminal_receipt(
    tmp_path, monkeypatch, mutation
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="long4096"
    )
    predecessor_path = Path(spec["predecessor_result"]["path"])
    predecessor = json.loads(predecessor_path.read_text())
    predecessor.pop("content_sha256")
    mutation(predecessor)
    spec["predecessor_result"]["sha256"] = _write(
        predecessor_path, _sealed(predecessor)
    )
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="finite natural-exit receipt"):
        launcher.build_plan(spec_path)


def test_long4096_revalidates_prelong_gate_instead_of_trusting_resealed_receipt(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="long4096"
    )
    predecessor_path = Path(spec["predecessor_result"]["path"])
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    predecessor.pop("content_sha256")
    terminal = predecessor["terminal_acceptance"]
    terminal.pop("content_sha256")
    prelong = terminal["prelong_gate"]
    prelong.pop("content_sha256")
    prelong["gate"]["status"] = "BLOCKED"
    prelong["gate_sha256"] = launcher.canonical_sha256(prelong["gate"])
    prelong["content_sha256"] = launcher.canonical_sha256(prelong)
    terminal["content_sha256"] = launcher.canonical_sha256(terminal)
    spec["predecessor_result"]["sha256"] = _write(
        predecessor_path, _sealed(predecessor)
    )
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="terminal checkpoint/safety"):
        launcher.build_plan(spec_path)


def test_long4096_rejects_failure_branch_predecessor(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="long4096"
    )
    probe_path = tmp_path / (launcher.ARM_IDS[0] + ".probe512.json")
    spec["predecessor_result"] = {
        "path": str(probe_path),
        "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
    }
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="scale4096 predecessor result"):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "field",
    (
        "arm_materialization",
        "policy_recipe_materialization",
        "oracle32_receipt",
    ),
)
def test_long4096_rejects_scale_reward_policy_or_oracle_lineage_drift(
    tmp_path, monkeypatch, field
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="long4096"
    )
    predecessor_path = Path(spec["predecessor_result"]["path"])
    predecessor = json.loads(predecessor_path.read_text())
    predecessor.pop("content_sha256")
    predecessor[field]["content_sha256"] = "8" * 64
    spec["predecessor_result"]["sha256"] = _write(
        predecessor_path, _sealed(predecessor)
    )
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="predecessor arm/oracle lineage"):
        launcher.build_plan(spec_path)


def test_confirm_digest_mismatch_blocks_before_source_lock_or_namespace(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher._B, "_verify_clean_source", lambda *args: pytest.fail("source touched"))
    with pytest.raises(launcher.LaunchRefused, match="confirm-claim differs"):
        launcher.execute(plan, confirm_claim="f" * 64)
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


@pytest.mark.parametrize("mutation", ("outer_extra", "payload"))
def test_mutated_plan_envelope_blocks_before_gpu_lock_or_namespace(
    tmp_path, monkeypatch, mutation
):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    plan = launcher.build_plan(spec_path)
    if mutation == "outer_extra":
        plan["unsealed_extra"] = True
    else:
        plan["canonical_payload"]["bundle"]["normalizers"]["actor"][
            "state"
        ] = "mutated"
    monkeypatch.setattr(
        launcher,
        "_open_gpu_shared_lock",
        lambda *_args: pytest.fail("GPU lock touched"),
    )
    with pytest.raises(launcher.LaunchRefused):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


@pytest.mark.parametrize("mutation", ("authorization", "extra"))
def test_resealed_unsafe_payload_blocks_before_gpu_lock_or_namespace(
    tmp_path, monkeypatch, mutation
):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    plan = launcher.build_plan(spec_path)
    if mutation == "authorization":
        plan["canonical_payload"]["diagnostic_unauthorized"] = False
    else:
        plan["canonical_payload"]["unsealed_extra"] = True
    digest = launcher.canonical_sha256(plan["canonical_payload"])
    plan["launch_claim_sha256"] = digest
    monkeypatch.setattr(
        launcher,
        "_open_gpu_shared_lock",
        lambda *_args: pytest.fail("GPU lock touched"),
    )
    with pytest.raises(launcher.LaunchRefused):
        launcher.execute(plan, confirm_claim=digest)
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


def test_claim_namespace_is_no_clobber(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    original = (namespace / "launch_claim.json").read_bytes()
    with pytest.raises(launcher.LaunchRefused):
        launcher._B._claim_namespace(plan)
    assert (namespace / "launch_claim.json").read_bytes() == original


def test_a211_vendor_admission_adapter_reconstructs_lineage_output_contract(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    plan = launcher.build_plan(spec_path)
    payload = plan["canonical_payload"]
    assert launcher._ADMISSION._output_contract_from_payload is (
        launcher._admission_output_contract
    )
    assert launcher._ADMISSION._output_contract_from_payload(
        payload["spec"], payload
    ) == payload["output_contract"]


def test_pre_exec_admission_race_refuses_before_execve(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(
        launcher,
        "_verify_gpu_admission",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            launcher.LaunchRefused("pre_exec race occupied GPU")
        ),
    )
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("execve reached"))
    lock_path = Path(plan["canonical_payload"]["spec"]["gpu"]["lock_path"])
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(launcher.LaunchRefused, match="pre_exec race"):
            launcher._internal_exec(
                namespace / "launch_claim.json", plan["launch_claim_sha256"], lock_fd
            )
    finally:
        os.close(lock_fd)
    assert not (namespace / "pre_exec_gpu_admission.json").exists()


def test_post_boot_admission_failure_routes_exact_cleanup_and_spends_namespace(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="long512")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    lock_file = tmp_path / "gpu.lock"
    monkeypatch.setattr(
        launcher, "_open_gpu_shared_lock", lambda path: os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    )
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    phases = []

    def admission(spec, *, phase, current_namespace, require_current_compute=False, **kwargs):
        phases.append((phase, require_current_compute))
        if phase == "post_boot":
            raise launcher.LaunchRefused("post_boot unknown pid")
        return {"phase": phase}

    monkeypatch.setattr(launcher, "_verify_gpu_admission", admission)
    monkeypatch.setattr(
        launcher,
        "_write_reservation",
        lambda spec, digest: {"registry_path": lock_file, "claim": digest},
    )
    monkeypatch.setattr(launcher, "_release_reservation", lambda handle: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0})(),
    )
    cleanup_calls = []

    def cleanup(namespace, state, claim_sha, reason):
        cleanup_calls.append((namespace, state, claim_sha, reason))
        return {"cleanup": {"completed": True}, "path": namespace / "cleanup-failure.json"}

    monkeypatch.setattr(launcher, "_cleanup_post_boot_admission_failure", cleanup)
    with pytest.raises(launcher.LaunchRefused, match=r"cleanup completed.*cleanup-failure.json"):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    namespace = Path(plan["canonical_payload"]["spec"]["namespace"])
    assert namespace.is_dir()
    assert (namespace / "launch_claim.json").is_file()
    assert phases == [("pre_launch", False), ("post_boot", True)]
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][0] == namespace
    assert cleanup_calls[0][2] == plan["launch_claim_sha256"]


def test_claim_revalidation_detects_code_owned_bundle_mutation(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = Path(plan["canonical_payload"]["spec"]["namespace"])
    namespace.mkdir()
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["normalizers"]["actor"]["state"] = "donor"
    monkeypatch.setattr(launcher._B, "_validate_runtime_asset_claim", lambda value: value)
    with pytest.raises(launcher.LaunchRefused, match="drifted"):
        launcher._revalidate_claim_payload(payload)
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["isaac_four_grid_manifest"]["cells"].pop()
    with pytest.raises(launcher.LaunchRefused, match="drifted"):
        launcher._revalidate_claim_payload(payload)


def test_claim_revalidation_refuses_shared_four_grid_source_sha_drift(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    source = {
        "Isaac A211/C211 four-grid authority": {
            "path": launcher.FOUR_GRID_SOURCE,
            "sha256": "a" * 64,
        }
    }
    monkeypatch.setattr(
        launcher, "_runtime_sources", lambda _checkout, _commit: copy.deepcopy(source)
    )
    spec_path, _, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    plan = launcher.build_plan(spec_path)
    Path(plan["canonical_payload"]["spec"]["namespace"]).mkdir()
    source["Isaac A211/C211 four-grid authority"]["sha256"] = "b" * 64
    monkeypatch.setattr(
        launcher._B, "_validate_runtime_asset_claim", lambda value: value
    )
    with pytest.raises(launcher.LaunchRefused, match="runtime source identity drifted"):
        launcher._revalidate_claim_payload(plan["canonical_payload"])


@pytest.mark.parametrize("retired_key", ("target_recipe", "target_validity_mask", "resume_path", "checkpoint_path"))
def test_spec_rejects_retired_control_keys(tmp_path, retired_key):
    _, spec, _ = _case(tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize")
    spec[retired_key] = "forbidden"
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_spec(spec)


def test_spec_rejects_disconnected_same_named_experiment_root(tmp_path):
    _, spec, _ = _case(
        tmp_path, arm_id=launcher.ARM_IDS[0], stage="materialize"
    )
    detached = tmp_path / "detached" / launcher.EXPERIMENT_NAME
    detached.mkdir(parents=True)
    namespace = detached / "fresh"
    spec["namespace"] = str(namespace)
    spec["log_path"] = str(namespace / "run.log")
    with pytest.raises(launcher.LaunchRefused, match="checkout-local A211"):
        launcher._validate_spec(spec)


def test_template_python_symlink_defaults_to_exclusive(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    real_python = tmp_path / "real-python"
    real_python.write_text("#!/bin/sh\n")
    real_python.chmod(0o755)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)
    output = tmp_path / "template.json"
    root = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / launcher.EXPERIMENT_NAME
    )
    root.mkdir(parents=True)
    args = launcher._parser().parse_args([
        "template", "--output", str(output), "--checkout", str(checkout),
        "--commit-sha", "a" * 40, "--isaac-python", str(venv_python),
        "--arm-id", launcher.ARM_IDS[0], "--lineage-path", "a211.json",
        "--lineage-sha256", "b" * 64, "--stage", "materialize",
        "--gpu-index", "2", "--gpu-uuid", "GPU-12345678", "--owner", "Franco",
        "--namespace", str(root / "fresh"),
    ])
    launcher._write_template(args)
    document = json.loads(output.read_text())
    assert document["source"]["isaac_python"] == str(venv_python)
    assert launcher.COLOCATION_SPEC_KEY not in document
    assert document["gpu"]["require_empty"] is True


def test_parser_exposes_explicit_execute_and_hidden_exec():
    parser = launcher._parser()
    subparsers = next(action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction")
    assert set(subparsers.choices) == {"template", "plan", "execute", "_exec"}


def test_launcher_never_sets_or_repurposes_home():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"HOME"' not in source
    assert "$HOME" not in source


def test_launcher_trainability_literal_matches_the_contract_publisher():
    """Consumer and producer must spell the marker identically.

    The oracle32 ABI gate compares the hard document's
    ``action_ball_211_trainability_contract`` against this launcher's literal.
    train.py publishes it from training_contract's own literal, so if the two
    ever drift the gate refuses every run for a reason no log explains.
    """
    assert (
        launcher.TRAINABILITY_CONTRACT
        == _TRAINING_CONTRACT._ACTION_BALL_A211_TRAINABILITY_CONTRACT
    )


def test_hard_wait_contract_is_the_shape_training_contract_actually_emits():
    """The oracle32 ABI gate must compare the schema-3 block, not the spec block.

    The gate used to look for ``action_ball_task_wait_contract`` -- a key no
    producer in this repo has ever written -- and compare it against the
    spec-shaped ``_wait_contract()``.  That made the gate unsatisfiable, so the
    A211 oracle32 refused every run with "hard-contract ABI/authorization
    differs" no matter how healthy the run was.  Pin both the key and the shape
    to training_contract's own frozen authority.
    """
    assert (
        launcher._hard_wait_contract()
        == _TRAINING_CONTRACT._action_ball_211_wait_contract_facts()
    )
    # the spec shape and the hard shape are genuinely different objects
    assert launcher._hard_wait_contract() != launcher._wait_contract()
    # The gate must bind the real key as code.  Match on the quoted spellings so
    # the prose that explains this bug does not satisfy or trip the check.
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"task_wait_contract": _hard_wait_contract(),' in text
    assert '"action_ball_task_wait_contract"' not in text
