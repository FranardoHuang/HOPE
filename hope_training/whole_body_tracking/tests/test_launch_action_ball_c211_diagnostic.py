"""CPU-only fail-closed tests for the independent C211 launcher."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_action_ball_c211_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("launch_c211_diagnostic", SCRIPT)
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
    "path": launcher._FRAME0.DR_L0_MANIFEST_SOURCE,
    "file_sha256": "d" * 64,
    "contract_sha256": _DR_L0_CONTRACT_SHA256,
    "hard_contract_identity": "action_ball_dr_l0_exact_all_off_v1",
    "task_profile": launcher.TASK_PROFILE_ID,
}

PRODUCER_SCRIPT = SCRIPT.parent / "action_ball_c211_oracle_evidence.py"
PRODUCER_SPEC = importlib.util.spec_from_file_location(
    "action_ball_c211_oracle_evidence", PRODUCER_SCRIPT
)
producer = importlib.util.module_from_spec(PRODUCER_SPEC)
sys.modules[PRODUCER_SPEC.name] = producer
PRODUCER_SPEC.loader.exec_module(producer)

TRAINABILITY_SOURCE = (
    SCRIPT.parent.parent
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/"
    "action_ball_c211_trainability.py"
)
TRAINABILITY_SPEC = importlib.util.spec_from_file_location(
    "action_ball_c211_trainability_oracle_fixture", TRAINABILITY_SOURCE
)
trainability = importlib.util.module_from_spec(TRAINABILITY_SPEC)
sys.modules[TRAINABILITY_SPEC.name] = trainability
TRAINABILITY_SPEC.loader.exec_module(trainability)

# The exact runtime clock the launcher gate must agree with.  ``action_ball_runtime``
# is deliberately dependency-light (stdlib only), so a CPU-only launcher test can
# import the very function ``hope_commands`` re-derives every emitted receipt from.
RUNTIME_SOURCE = (
    SCRIPT.parent.parent
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "action_ball_runtime.py"
)
RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "action_ball_runtime_for_c211_launcher_test", RUNTIME_SOURCE
)
runtime = importlib.util.module_from_spec(RUNTIME_SPEC)
sys.modules[RUNTIME_SPEC.name] = runtime
RUNTIME_SPEC.loader.exec_module(runtime)

# Producer-emitted bytes, never hand-edited and never re-sealed: the center-stratum
# ``outcome_dense_only`` target receipt from the fixed-tape build.
INITIAL_CENTER_C_RECEIPT_SOURCE = Path(__file__).resolve().parents[3] / (
    "configs/action_ball_n1_measured_20260803/"
    "fresh_tape_seed0_20260803_take061_robust20n_r9_center/"
    "outcome_dense_only.target.task_receipt.v5.f9e0ddf178aa.json"
)


def _real_runner_preflight_facts() -> dict:
    """Build the fixture through the same validator used by the real runner."""

    cfg = SimpleNamespace(
        obs_mode=trainability.C211_ACTOR_CONTRACT,
        action_ball_211_construction_only=False,
        action_ball_211_trainability_contract=(
            trainability.C211_TRAINABILITY_CONTRACT
        ),
        critic_obs_contract=trainability.C211_CRITIC_CONTRACT,
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(
                action_ball_target_source="direct_ball",
                action_ball_reuse_exact_question_until_semantics_change=False,
                action_ball_initial_center_single_question=True,
                action_ball_target_recipe="outcome_dense_only",
                action_ball_target_validity_mask=[False, False, False],
                action_ball_task_wait_enabled=True,
                action_ball_task_wait_policy_dt_s=0.02,
                action_ball_task_wait_seed=20260804,
                action_ball_task_wait_min_wait_ticks=5,
                action_ball_task_wait_max_wait_ticks=25,
                action_ball_task_wait_episode_horizon_ticks=500,
                action_ball_task_wait_required_active_ticks=200,
            )
        ),
        observations=SimpleNamespace(critic=object()),
    )
    manager = SimpleNamespace(
        active_terms={
            "policy": [name for name, _dim in trainability.C211_ACTOR_LAYOUT],
            "critic": [name for name, _dim in trainability.C211_CRITIC_LAYOUT],
        },
        group_obs_term_dim={
            "policy": [
                (dim,) for _name, dim in trainability.C211_ACTOR_LAYOUT
            ],
            "critic": [
                (dim,) for _name, dim in trainability.C211_CRITIC_LAYOUT
            ],
        },
        group_obs_dim={
            "policy": (trainability.C211_ACTOR_WIDTH,),
            "critic": (trainability.C211_CRITIC_WIDTH,),
        },
    )
    runtime = SimpleNamespace(cfg=cfg, observation_manager=manager)
    runtime.unwrapped = runtime
    wrapped = SimpleNamespace(
        unwrapped=runtime,
        num_obs=trainability.C211_ACTOR_WIDTH,
        num_privileged_obs=trainability.C211_CRITIC_WIDTH,
    )
    actor_normalizer = object()
    critic_normalizer = object()
    runner = SimpleNamespace(
        env=wrapped,
        alg=SimpleNamespace(
            policy=SimpleNamespace(
                num_actor_obs=trainability.C211_ACTOR_WIDTH,
                num_critic_obs=trainability.C211_CRITIC_WIDTH,
            )
        ),
        empirical_normalization=True,
        _resolve_runtime_normalizer=lambda role: (
            ("actor_obs_normalizer", actor_normalizer, ())
            if role == "actor"
            else ("critic_obs_normalizer", critic_normalizer, ())
        ),
    )
    facts = trainability.validate_action_ball_c211_runner(runner)
    assert type(facts) is dict
    return facts


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


def _frame0_live_safety(action_id: str, motion_sha: str) -> dict:
    names = ["joint_%02d" % index for index in range(31)]
    joint = {
        "schema_version": 1,
        "complete": True,
        "joint_order": names,
        "current_actual_hard_edge_joint_count": 0,
        "current_actual_hard_edge_joint_names": [],
        "substep_actual_hard_edge_joint_count": 0,
        "substep_actual_hard_edge_joint_names": [],
        "final_minimum_hard_gap_rad": 1.0,
        "preterminal_joint_pos_rad": [0.0] * 31,
        "preterminal_joint_vel_radps": [0.0] * 31,
        "final_joint_pos_rad": [0.0] * 31,
        "final_joint_vel_radps": [0.0] * 31,
        "hard_lower_rad": [-1.0] * 31,
        "hard_upper_rad": [1.0] * 31,
    }
    return _sealed(
        {
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
            "requested_duration_s": 62 * launcher.POLICY_DT_S,
            "completed_duration_s": 62 * launcher.POLICY_DT_S,
            "completed_policy_steps": 62,
            "completed_physics_steps": 248,
            "terminal_reasons": [],
            "generic_terminated": False,
            "generic_truncated": False,
            "minimum_root_z_m": 0.9,
            "maximum_root_tilt_rad": 0.1,
            "both_feet_contact_fraction": 1.0,
            "joint_safety_telemetry": joint,
            "screenshots": [
                {"label": label, "sha256": ("%x" % (index + 1)) * 64}
                for index, label in enumerate(
                    (
                        "raw_env_reset", "physical_ready_after_reset_write",
                        "after_step_1", "after_step_10", "final",
                    )
                )
            ],
        }
    )


def _exact_zero_handoff_fields(
    *, motion_sha: str, joint_pos, root_pos, root_quat
):
    joint_pos = [float(value) for value in joint_pos]
    root_pos = [float(value) for value in root_pos]
    root_quat = [float(value) for value in root_quat]
    root_norm = math.sqrt(sum(value * value for value in root_quat))
    audit_quat = [value / root_norm for value in root_quat]
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
        "required_followup_policy_steps": launcher.PASSIVE_HOLD_SOAK_POLICY_STEPS,
        "required_followup_physics_steps": launcher.PASSIVE_HOLD_SOAK_PHYSICS_STEPS,
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
        "strike": (1.0, 9),
        "target": (0.0, 0),
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
            "per_term_weighted_dt_sum": {"motion": 1.0, "task": 0.0},
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
            profile=launcher._S.PRELONG_PROFILE_C211,
        ),
    ]


def _rewrite_prelong_marker(log_path: Path, update: int, mutate):
    prefix = launcher._S.PRELONG_SEMANTICS_MARKER_PREFIX
    rewritten = []
    matched = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            row = json.loads(line[len(prefix) :])
            if row["ppo_update"] == update:
                mutate(row)
                line = prefix + json.dumps(
                    row, sort_keys=True, separators=(",", ":")
                )
                matched += 1
        rewritten.append(line)
    assert matched == 1
    log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _lineage(checkout: Path) -> dict:
    repo = Path(__file__).resolve().parents[3]
    sources = {
        "motion": repo / (
            "assets/motions/chingmu73_measured_v4_20260803/"
            "hope_Take_061_unit04_BH.npz"
        ),
        "action_manifest": repo / (
            "configs/action_ball_n1_measured_20260803/"
            "fresh_core_seed0_20260803_take061_robust20n_r8_splitready/"
            "take_061_unit04_bh.full.manifest.v3.7d2139028427.json"
        ),
        "dynamic_ready_artifact": repo / (
            "configs/action_ball_n1_measured_20260803/"
            "evidence_holdpass_robust20n_20260803/"
            "take061.measured_teacher.yaw_aligned_full_seed.robust20n."
            "dynamic_ready.v2.json"
        ),
        "dynamic_ready_nominal_receipt": repo / (
            "configs/action_ball_n1_measured_20260803/"
            "evidence_holdpass_robust20n_20260803/"
            "take061.robust20n.nominal_hold.v1.json"
        ),
        "teacher_frame0_artifact": repo / (
            "configs/action_ball_n1_measured_20260803/"
            "a211_frame0_exact_20260803/"
            "take_061_unit04_bh.frame0_exact.v1.json"
        ),
    }
    pins = {}
    for key, source in sources.items():
        relative = (
            source.relative_to(repo)
            if key in ("motion", "action_manifest")
            else Path(source.name)
        )
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        pins[key] = {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    # 这份收据是 producer 原样吐出的字节, 一个数值都没改, canonical_sha256 也没重封。
    # 它是 outcome_dense_only(即 C 钉的 TARGET_RECIPE)在 center 层的目标收据, 天然满足
    # 运行时那条 `pre_swing_wait_s = time_to_contact_s - scaled_t_hit_s`。
    receipt_bytes = INITIAL_CENTER_C_RECEIPT_SOURCE.read_bytes()
    receipt_path = checkout / "initial_center_c.task_receipt.v5.json"
    receipt_path.write_bytes(receipt_bytes)
    pins["initial_center_task_receipt"] = {
        "path": receipt_path.name,
        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
    bundle_unsigned = {
        "schema_version": 4,
        "kind": launcher.C211_BUNDLE_KIND,
        "diagnostic_unauthorized": True,
        "action_id": launcher.ACTION_ID,
        "action_uid": launcher.ACTION_UID,
        "teacher_id": launcher.TEACHER_ID,
        "actor_contract": launcher.ACTOR_CONTRACT,
        "actor_width": launcher.ACTOR_WIDTH,
        "critic_contract": launcher.CRITIC_CONTRACT,
        "critic_width": launcher.CRITIC_WIDTH,
        "trainability_contract": launcher.TRAINABILITY_CONTRACT,
        "actor_normalizer_identity": launcher.ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": launcher.CRITIC_NORMALIZER_IDENTITY,
        "target_source": launcher.TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng": launcher._question_rng_contract(),
        "target_recipe": launcher.TARGET_RECIPE,
        "curriculum_scope": launcher._curriculum_scope_contract(),
        "target_validity_mask": list(launcher.TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(launcher.INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "motion": pins["motion"],
        "action_manifest": pins["action_manifest"],
        "initial_center_task_receipt": pins["initial_center_task_receipt"],
        "dynamic_ready_artifact": pins["dynamic_ready_artifact"],
        "dynamic_ready_nominal_receipt": pins[
            "dynamic_ready_nominal_receipt"
        ],
        "teacher_frame0_artifact": pins["teacher_frame0_artifact"],
        "dr_l0_manifest": dict(_DR_L0_BINDING),
    }
    bundle_path = checkout / "c211_bundle.json"
    pins["bundle"] = {
        "path": bundle_path.name,
        "sha256": _write(bundle_path, _sealed(bundle_unsigned)),
    }
    return {
        "schema_version": 4,
        "kind": launcher.LINEAGE_KIND,
        "actor_contract": launcher.ACTOR_CONTRACT,
        "actor_width": launcher.ACTOR_WIDTH,
        "critic_contract": launcher.CRITIC_CONTRACT,
        "critic_width": launcher.CRITIC_WIDTH,
        "trainability_contract": launcher.TRAINABILITY_CONTRACT,
        "actor_normalizer_identity": launcher.ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": launcher.CRITIC_NORMALIZER_IDENTITY,
        "task_profile": launcher.TASK_PROFILE_ID,
        "gym_task": launcher.GYM_TASK_ID,
        "target_semantics": launcher.TARGET_SEMANTICS,
        "curriculum_scope": launcher._curriculum_scope_contract(),
        "target_source": launcher.TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng": launcher._question_rng_contract(),
        "target_recipe": launcher.TARGET_RECIPE,
        "target_validity_mask": list(launcher.TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(launcher.INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "action_id": launcher.ACTION_ID,
        "action_uid": launcher.ACTION_UID,
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
    namespace=None,
    terminal_acceptance=None,
) -> dict:
    budget = launcher.BUDGETS[stage]
    if completion is None:
        completion = {
            "completion_exit_code": "0",
            "terminal_kind": "clean_completion",
            "terminal_exit_code": "0",
        }
    if output_contract is None:
        output_contract = {
            "ppo_update_count": budget[1],
            "finite_model_save_interval": budget[2],
        }
    unsigned = {
        "schema_version": 2,
        "kind": launcher.RESULT_KIND,
        "diagnostic_unauthorized": True,
        "accepted": True,
        "launch_claim_sha256": "1" * 64,
        "stage": stage,
        "namespace": (
            "/tmp/c211-fixture-" + stage
            if namespace is None
            else str(namespace)
        ),
        "completion": completion,
        "gpu_admission": {"phase": "fixture"},
        "output_contract": output_contract,
        "reward_materialization": materialization,
        "policy_recipe_materialization": policy,
        "oracle32_receipt": oracle,
        "predecessor_result": predecessor,
    }
    if terminal_acceptance is not None:
        unsigned["terminal_acceptance"] = terminal_acceptance
    return {"path": str(path), "sha256": _write(path, _sealed(unsigned))}


def _terminal_rows(
    terminations: "list[tuple[str, list[str]]] | None",
) -> "list[tuple[str, list[str]]]":
    """One (terminal_phase, termination_reasons) pair per closed attempt.

    Default = the historical fixture: 32 clean single strokes.  Mutation tests
    hand in falls / table hits / implementation-failure reasons instead.
    """

    rows = terminations or [
        ("post_strike", ["action_ball_single_stroke_complete"]) for _ in range(32)
    ]
    assert len(rows) == 32
    return [(phase, list(reasons)) for phase, reasons in rows]


def _termination_ledger(rows: "list[tuple[str, list[str]]]") -> dict:
    """Aggregate the same way the runtime producer does, from the rows alone."""

    by_reason: dict = {}
    phase_by_reason: dict = {
        "post_strike": {},
        "pre_strike_or_same_step_unknown": {},
    }
    for phase, reasons in rows:
        for reason in reasons:
            by_reason[reason] = by_reason.get(reason, 0) + 1
            phase_by_reason[phase][reason] = phase_by_reason[phase].get(reason, 0) + 1
    return {
        "allowed_reason": "action_ball_single_stroke_complete",
        "by_reason": by_reason,
        "unexpected_by_reason": {
            reason: count
            for reason, count in by_reason.items()
            if reason != "action_ball_single_stroke_complete"
        },
        "phase_by_reason": phase_by_reason,
    }


def _expected_termination_census(
    rows: "list[tuple[str, list[str]]]", *, wait_only_reset_excluded: int = 0
) -> dict:
    """The self-describing block the oracle32 receipt must carry.

    Deliberately recomputed here from the same rows the fixture wrote, so a
    receipt that reports a prettier census than its own 32 episodes fails.
    """

    ledger = _termination_ledger(rows)
    episodes_by_phase = {
        "post_strike": 0,
        "pre_strike_or_same_step_unknown": 0,
    }
    for phase, _reasons in rows:
        episodes_by_phase[phase] += 1
    return {
        "closed_attempt_episodes": len(rows),
        "source_episodes_consumed": len(rows) + wait_only_reset_excluded,
        "wait_only_reset_excluded": wait_only_reset_excluded,
        "episodes_by_terminal_phase": episodes_by_phase,
        "terminal_reason_totals": ledger["by_reason"],
        "terminal_reason_by_phase": ledger["phase_by_reason"],
        "single_stroke_complete_count": ledger["by_reason"].get(
            "action_ball_single_stroke_complete", 0
        ),
        "physical_fall_by_reason": {
            reason: ledger["by_reason"].get(reason, 0)
            for reason in launcher.PHYSICAL_FALL_REASONS
        },
        "robot_hit_table_count": ledger["by_reason"].get("robot_hit_table", 0),
        "implementation_strict_zero": {
            **{
                name: ledger["by_reason"].get(name, 0)
                for name in launcher.STRICT_HARD_TERMINATION_UNION
            },
            "projection_nonfinite_count": 0,
        },
    }


def _raw_oracle(
    path: Path,
    *,
    lineage: dict,
    lineage_sha: str,
    recipe: dict,
    materialization: dict,
    policy: dict,
    terminations: "list[tuple[str, list[str]]] | None" = None,
    wait_only_reset_excluded: int = 0,
    projection_nonfinite: int = 0,
) -> dict:
    params = path.parent / "params"
    params.mkdir(parents=True, exist_ok=True)
    actor_layout = (
        ("actual_base_pose_lin_vel_world", 12),
        ("base_ang_vel_body", 3),
        ("joint_pos", 31),
        ("joint_vel", 31),
        ("actions", 31),
        ("racket_site_achieved_now_heading", 9),
        ("teacher_joint_pos", 31),
        ("teacher_joint_vel", 31),
        ("racket_site_teacher_now_heading", 9),
        ("racket_site_teacher_at_reference_hit_heading", 9),
        ("incoming_ball_contact_position_heading", 3),
        ("incoming_ball_contact_velocity_heading", 3),
        ("incoming_ball_contact_spin_heading", 3),
        ("desired_base_xy_world", 2),
        ("time_to_contact", 1),
        ("time_to_teacher_start", 1),
        ("task_valid", 1),
    )
    critic_layout = (
        ("command", 62),
        ("motion_anchor_pos_b", 3),
        ("motion_anchor_ori_b", 6),
        ("body_pos", 42),
        ("body_ori", 84),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("joint_pos", 31),
        ("joint_vel", 31),
        ("actions", 31),
        ("racket_site_teacher_at_reference_hit_heading", 9),
        ("incoming_ball_contact_position_heading", 3),
        ("incoming_ball_contact_velocity_heading", 3),
        ("incoming_ball_contact_spin_heading", 3),
        ("desired_base_xy_world", 2),
        ("time_to_contact", 1),
        ("time_to_teacher_start", 1),
        ("task_valid", 1),
    )
    hard_contract = {
        "schema_version": 3,
        "target_mode": "action_ball",
        "actor_obs_contract": launcher.ACTOR_CONTRACT,
        "actor_obs_total_dim": launcher.ACTOR_WIDTH,
        "actor_obs_term_names": [name for name, _dim in actor_layout],
        "actor_obs_term_dims": [dim for _name, dim in actor_layout],
        "observation_history_lengths": [1] * len(actor_layout),
        "critic_obs_contract": launcher.CRITIC_CONTRACT,
        "critic_obs_total_dim": launcher.CRITIC_WIDTH,
        "critic_obs_term_names": [name for name, _dim in critic_layout],
        "critic_obs_term_dims": [dim for _name, dim in critic_layout],
        "actor_obs_normalizer_identity": launcher.ACTOR_NORMALIZER_IDENTITY,
        "critic_obs_normalizer_identity": launcher.CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
        "task_valid_required": True,
        "task_wait_contract": launcher._hard_wait_contract(),
        "question_source_contract": launcher._hard_question_source_contract(),
        "contact_target_absent": True,
        "c225_reward_contract": launcher._c211_reward_contract(),
        "action_ball_dr_l0": copy.deepcopy(_DR_L0_PAYLOAD),
        "action_ball_training": {
            "runtime": {
                "target_provider": {
                    "source": launcher.TARGET_SOURCE,
                    "recipe": launcher.TARGET_RECIPE,
                    "validity_mask": list(launcher.TARGET_VALIDITY_MASK),
                    "target_observation_noise": False,
                    "actor_width_unchanged": True,
                    "critic_width_unchanged": True,
                    "immutable_tape": None,
                    "exact_question_answer_reuse": {"enabled": False},
                }
            }
        },
    }
    hard_path = params / "training_contract.json"
    hard_sha = _write(hard_path, hard_contract)
    preflight_path = params / "c211_runner_preflight.json"
    preflight_sha = _write(
        preflight_path,
        _sealed(
            {
                "schema_version": 1,
                "kind": launcher.C211_RUNNER_PREFLIGHT_KIND,
                "diagnostic_unauthorized": True,
                "oracle_launch_claim_sha256": "1" * 64,
                "hard_contract_sha256": hard_sha,
                "marker": "ACTION_BALL_C211_TRAINABILITY_PREFLIGHT_JSON",
                "facts": _real_runner_preflight_facts(),
            }
        ),
    )
    selected_rows = [
        {
            "episode": index,
            "eligible_closed_swing": True,
            "classification": "selected_rubber",
            "contact_evidence_sha256": hashlib.sha256(
                ("selected-rubber-%d" % index).encode()
            ).hexdigest(),
        }
        for index in range(32)
    ]
    selected_path = params / "c211_selected_rubber_contact.json"
    selected_sha = _write(
        selected_path,
        _sealed(
            {
                "schema_version": 1,
                "kind": launcher.C211_SELECTED_RUBBER_KIND,
                "diagnostic_unauthorized": True,
                "oracle_launch_claim_sha256": "1" * 64,
                "action_id": launcher.ACTION_ID,
                "action_uid": launcher.ACTION_UID,
                "motion_sha256": lineage["motion"]["sha256"],
                "classifier_contract": "runtime_contact_pair_selected_rubber_v1",
                "classifier_source_sha256": "6" * 64,
                "geometry_authority_sha256": "7" * 64,
                "denominator_kind": "eligible_closed_swings",
                "episodes": selected_rows,
            }
        ),
    )
    incoming = {
        "incoming_ball_contact_position_heading": [0.5, 0.0, 1.05],
        "incoming_ball_contact_velocity_heading": [-2.8, 1.05, 0.24],
        "incoming_ball_contact_spin_heading": [0.0, 0.0, 0.0],
    }
    analytic_flight = {
        "evaluated": True,
        "finite": True,
        "landing_xy_m": [1.0, 0.0],
        "landing_valid": True,
        "net_crossed": True,
        "net_clear": True,
        "on_opponent_table": True,
        "source": "runtime_vb_one_shot_from_achieved_selected_rubber_contact",
    }
    predicted_outcome = {
        "evaluated": True,
        "predicted_net_clear": True,
        "predicted_legal_landing": True,
        "predicted_landing_xy_m": [1.0, 0.0],
        "source": "runtime_c225_achieved_flight_prediction_one_shot",
    }
    rows = _terminal_rows(terminations)
    episodes = [
        {
            "episode": index,
            "control_steps": 1,
            "terminal_phase": rows[index][0],
            "termination_reasons": list(rows[index][1]),
            "sampler_sample_index": index,
            "sampler_sample_sha256": hashlib.sha256(
                ("runtime-sample-%d" % index).encode()
            ).hexdigest(),
            "sampler_draw_start": index * 18,
            "sampler_draw_end": (index + 1) * 18,
            "incoming_ball_observation": {
                "source": "runtime_actor_and_critic_observation_terms",
                "actor": copy.deepcopy(incoming),
                "critic": copy.deepcopy(incoming),
            },
            "selected_rubber_evidence_sha256": launcher.canonical_sha256(
                selected_rows[index]
            ),
            "achieved_analytic_flight": copy.deepcopy(analytic_flight),
            "predicted_outcome": copy.deepcopy(predicted_outcome),
        }
        for index in range(32)
    ]
    bindings = {
        "oracle_launch_claim_sha256": "1" * 64,
        "lineage_sha256": lineage_sha,
        "recipe_contract_sha256": recipe["recipe_contract_sha256"],
        "reward_contract_sha256": materialization["reward_contract_sha256"],
        "runtime_effective_reward_sha256": materialization[
            "runtime_effective_reward_sha256"
        ],
        "runtime_policy_recipe_sha256": policy["runtime_policy_recipe_sha256"],
        "hard_contract_sha256": hard_sha,
        "motion_sha256": lineage["motion"]["sha256"],
        "manifest_sha256": lineage["action_manifest"]["sha256"],
        "dynamic_ready_artifact_sha256": lineage["dynamic_ready_artifact"]["sha256"],
        "dynamic_ready_nominal_receipt_sha256": lineage[
            "dynamic_ready_nominal_receipt"
        ]["sha256"],
    }
    observed_episodes = []
    for index, raw_episode in enumerate(episodes):
        observed_episodes.append(
            {
                "episode": index,
                "control_steps": 1,
                "terminal_phase": rows[index][0],
                "termination_reasons": list(rows[index][1]),
                "sampler_sample_index": raw_episode["sampler_sample_index"],
                "sampler_sample_sha256": raw_episode["sampler_sample_sha256"],
                "sampler_draw_start": raw_episode["sampler_draw_start"],
                "sampler_draw_end": raw_episode["sampler_draw_end"],
                "incoming_ball_observation": raw_episode["incoming_ball_observation"],
                "observed_selected_rubber_contact": {
                    "episode": index, "runtime_control_step": 1,
                    "task_valid": True, "eligible_closed_swing": True,
                    "exact_strike": True, "selected_face_sweep_contact": True,
                    "selected_face_bracketed": True, "selected_face_edge_safe": True,
                    "selected_face_geometry_finite": True,
                    "selected_face_closing_speed_positive": True,
                    "selected_face_normal_speed_consistent": True,
                    "wrong_surface_contact": False,
                    "edge_or_rim_ambiguous": False,
                    "between_planes_ambiguous": False,
                },
                "achieved_analytic_flight": copy.deepcopy(analytic_flight),
                "predicted_outcome": copy.deepcopy(predicted_outcome),
                "safety": {
                    # Mirror the live adapter: per-episode hard counters are
                    # exactly int(name in this episode's termination reasons).
                    "hard_termination_by_reason": {
                        name: int(name in rows[index][1])
                        for name in launcher.HARD_TERMINATION_UNION
                    },
                    "robot_table_contact_count": int(
                        "robot_hit_table" in rows[index][1]
                    ),
                    "projection_nonfinite_count": (
                        projection_nonfinite if index == 0 else 0
                    ),
                    "projection_observed_sample_count": 1,
                    "qdes_observed_sample_count": 1,
                    "actual_observed_sample_count": 1,
                    "reference_guard_sample_count": 1,
                },
                "teacher_qdes": {
                    "preclamp_max_abs_error_rad": 1.0e-7,
                    "teleport_used": False,
                },
            }
        )
    rollout_census = {
        "source_episodes_consumed": 32 + wait_only_reset_excluded,
        "wait_only_reset_excluded": wait_only_reset_excluded,
        "closed_attempts": 32,
    }
    observed = {
        "schema_version": 3,
        "kind": producer.INPUT_KIND,
        "diagnostic_unauthorized": True,
        "identity": {
            "action_id": launcher.ACTION_ID,
            "action_uid": launcher.ACTION_UID,
            "motion_sha256": lineage["motion"]["sha256"],
        },
        "bindings": bindings,
        "training_contract_path": str(hard_path),
        "runner_preflight_facts": json.loads(preflight_path.read_text())["facts"],
        "question_contract": launcher._question_contract(),
        "rollout_census": dict(rollout_census),
        "episodes": observed_episodes,
    }
    _write(path.parent / launcher.C211_OBSERVED_BUNDLE_FILENAME, observed)
    ledger = _termination_ledger(rows)
    unsigned = {
        "schema_version": 3,
        "kind": launcher.C211_RAW_ORACLE_KIND,
        "diagnostic_unauthorized": True,
        "bindings": bindings,
        "observed_oracle_bundle_content_sha256": launcher.canonical_sha256(observed),
        "training_contract_artifact": {
            "path": str(hard_path),
            "sha256": hard_sha,
        },
        "runner_preflight_artifact": {
            "path": str(preflight_path),
            "sha256": preflight_sha,
        },
        "question_contract": launcher._question_contract(),
        "completion": {
            "requested": 32,
            "terminal": 32,
            "single_stroke": ledger["by_reason"].get(
                "action_ball_single_stroke_complete", 0
            ),
            "control_steps": 32,
        },
        "episodes": episodes,
        "desired_contact_metrics": {
            "status": "INELIGIBLE",
            "reason": "target_validity_000_contact_target_absent",
        },
        "rollout_census": dict(rollout_census),
        "termination": copy.deepcopy(ledger),
        "safety": {
            "control_step_denominator": 32,
            "hard_termination_by_reason": {
                name: ledger["by_reason"].get(name, 0)
                for name in launcher.HARD_TERMINATION_UNION
            },
            "robot_table_contact_count": ledger["by_reason"].get(
                "robot_hit_table", 0
            ),
            "projection_nonfinite_count": projection_nonfinite,
            "projection_observed_sample_count": 32,
            "qdes_observed_sample_count": 32,
            "actual_observed_sample_count": 32,
            "reference_guard_sample_count": 32,
        },
        "selected_rubber_contact_artifact": {
            "path": str(selected_path),
            "sha256": selected_sha,
        },
        "teacher_qdes": {
            "control_step_denominator": 32,
            "preclamp_max_abs_error_rad": 1.0e-7,
            "teleport_used": False,
        },
    }
    document = _sealed(unsigned)
    _write(path, document)
    return document


def _scale4096_terminal_fixture(tmp_path: Path, checkout: Path):
    root = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / launcher.EXPERIMENT_NAME
    )
    root.mkdir(parents=True)
    namespace = (
        tmp_path
        / launcher.EXPERIMENT_NAME
        / "c211-scale4096-predecessor"
    )
    namespace.mkdir(parents=True)
    run_dir = root / (
        "2026-08-04_12-34-56_"
        + namespace.name
        + "-DIAGNOSTIC_UNAUTHORIZED"
    )
    run_dir.mkdir()
    checkpoint_path = run_dir / "model_5.pt"
    checkpoint = {
        "iter": 5,
        "infos": {"training_launch_claim_sha256": "1" * 64},
        "model_state_dict": {"weight": _FakeTensor([1.0, 1.0])},
        "optimizer_state_dict": {
            "state": {0: {"momentum": _FakeTensor([1.0, 1.0])}}
        },
        "obs_norm_state_dict": {
            "running_mean": _FakeTensor([0.0, 0.0, 0.0])
        },
        "privileged_obs_norm_state_dict": {
            "running_var": _FakeTensor([1.0, 1.0, 1.0, 1.0])
        },
    }
    checkpoint_path.write_bytes(b"trusted exact C211 Pod checkpoint fixture\n")

    lines = ["[INFO] Task: fixture | experiment: fixture | log: %s" % run_dir]
    for update in range(5):
        rows = (
            (
                "HOPE_JOINT_SAFETY_UPDATE_JSON=",
                {
                    "event": "hope_joint_safety_diagnostic_compact_update",
                    "schema_version": 1,
                    "status": (
                        "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
                    ),
                    "ppo_update": update,
                    "counter_totals": {"actual_hard_edge_events": 0},
                },
            ),
            (
                "HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON=",
                {
                    "event": (
                        "action_ball_actual_joint_forbidden_diagnostic_update"
                    ),
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
                                    "lower": {
                                        "nonfinite_readback_observed": False
                                    },
                                    "upper": {
                                        "nonfinite_readback_observed": False
                                    },
                                },
                            }
                        ],
                    },
                },
            ),
            (
                "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=",
                {
                    "event": "hope_reward_safety_transition_update",
                    "schema_version": 2,
                    "ppo_update": update,
                    "coverage": "complete_update",
                    "terminal_transitions": [],
                },
            ),
            (
                "HOPE_EXACT_BEHAVIOR_UPDATE_JSON=",
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
            ),
        )
        lines.extend(
            prefix + json.dumps(row, sort_keys=True, separators=(",", ":"))
            for prefix, row in rows
        )
        lines.extend(_prelong_marker_lines(update))
    log_path = namespace / "run.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    checkpoint_artifact = {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "filename_iteration": 5,
        "embedded_iteration": 5,
        "map_location": "cpu",
        "load_mode": "torch_weights_only",
        "tensor_groups": {
            "model": {"tensor_count": 1, "element_count": 2},
            "optimizer": {"tensor_count": 1, "element_count": 2},
            "actor_normalizer": {"tensor_count": 1, "element_count": 3},
            "critic_normalizer": {"tensor_count": 1, "element_count": 4},
        },
        "all_tensors_finite": True,
    }
    run_log_artifact = {
        "path": str(log_path),
        "size_bytes": log_path.stat().st_size,
        "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    }
    safety_counters = {
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
    prelong_gate = launcher._prelong_terminal_gate_binding(
        log_raw=log_path.read_bytes(),
        run_log=run_log_artifact,
        checkpoint=checkpoint_artifact,
        safety_counters=safety_counters,
        launch_claim_sha256="1" * 64,
    )
    unsigned = {
        "schema_version": 1,
        "kind": launcher.SCALE4096_TERMINAL_ACCEPTANCE_KIND,
        "diagnostic_unauthorized": True,
        "launch_claim_sha256": "1" * 64,
        "run_log": run_log_artifact,
        "checkpoint": checkpoint_artifact,
        "safety_counters": safety_counters,
        "prelong_gate": prelong_gate,
    }
    acceptance = _sealed(unsigned)
    return namespace, checkpoint_path, checkpoint, log_path, acceptance


def _chain(
    tmp_path: Path,
    checkout: Path,
    lineage_sha: str,
    lineage: dict,
    *,
    recipe_id: str,
    terminations: "list[tuple[str, list[str]]] | None" = None,
    wait_only_reset_excluded: int = 0,
    projection_nonfinite: int = 0,
):
    recipe = launcher._recipe_contract(recipe_id)
    planned = launcher._planned_materialization(
        recipe=recipe,
        lineage={
            "lineage_sha256": lineage_sha,
            "dr_l0_manifest": dict(_DR_L0_BINDING),
        },
    )
    reward_artifact = tmp_path / "c211.effective_reward.json"
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
            "runtime_effective_reward_term_count": 12,
            # 2026-08-05 层级对齐(exp §5.6 第 7 条):death -300.0 -> -10.0。
            "runtime_soft_weights": {
                "death_penalty": -10.0,
                "joint_limit": -5.0,
                "qdes_limit_barrier": -5.0,
                "qdes_projection_penalty": -5.0,
            },
        }
    )
    materialization = _sealed(materialization_unsigned)
    materialize = _result(
        tmp_path / "materialize.result.json",
        stage="materialize",
        materialization=materialization,
    )
    policy_artifact = tmp_path / "c211.policy.json"
    policy_artifact.write_text("fixture\n", encoding="utf-8")
    policy = _sealed(
        {
            "schema_version": 1,
            "kind": launcher.POLICY_MATERIALIZATION_KIND,
            "diagnostic_unauthorized": True,
            "recipe_id": recipe_id,
            "lineage_sha256": lineage_sha,
            "recipe_contract_sha256": recipe["recipe_contract_sha256"],
            "runtime_policy_recipe_artifact": {
                "path": str(policy_artifact),
                "sha256": hashlib.sha256(policy_artifact.read_bytes()).hexdigest(),
            },
            "runtime_policy_recipe_sha256": "4" * 64,
            "dynamic_ready_binding_sha256": "5" * 64,
            # 这里刻意从 recipe 合同取值而不是抄字面量:权威在 four-grid manifest,
            # 夹具再抄一遍只会在换轴时变成"改测试让它变绿"。launcher 侧的交叉校验
            # 仍然被下面 test_policy_materialization_* 那组变异用例覆盖。
            "noise_std_type": recipe["noise_std_type"],
            "configured_and_realized_init_noise_std": recipe["init_noise_std"],
        }
    )
    recipe_result = _result(
        tmp_path / "recipe.result.json",
        stage="recipe",
        materialization=materialization,
        policy=policy,
    )
    oracle_namespace = tmp_path / "oracle32.runtime"
    raw_oracle = oracle_namespace / "teacher_qdes_oracle_32ep.json"
    raw_document = _raw_oracle(
        raw_oracle,
        lineage=lineage,
        lineage_sha=lineage_sha,
        recipe=recipe,
        materialization=materialization,
        policy=policy,
        terminations=terminations,
        wait_only_reset_excluded=wait_only_reset_excluded,
        projection_nonfinite=projection_nonfinite,
    )
    observed_path = oracle_namespace / launcher.C211_OBSERVED_BUNDLE_FILENAME
    observed_document = json.loads(observed_path.read_text())
    oracle = _sealed(
        {
            "schema_version": 3,
            "kind": launcher.ORACLE32_KIND,
            "diagnostic_unauthorized": True,
            "verdict": "PASS",
            "episodes": 32,
            "recipe_id": recipe_id,
            "lineage_sha256": lineage_sha,
            "recipe_contract_sha256": recipe["recipe_contract_sha256"],
            "reward_contract_sha256": materialization["reward_contract_sha256"],
            "runtime_effective_reward_sha256": "3" * 64,
            "runtime_policy_recipe_sha256": "4" * 64,
            "actor_contract": launcher.ACTOR_CONTRACT,
            "actor_width": 211,
            "critic_contract": launcher.CRITIC_CONTRACT,
            "critic_width": 319,
            "trainability_contract": launcher.TRAINABILITY_CONTRACT,
            "target_source": launcher.TARGET_SOURCE,
            "question_source": "runtime_curriculum_sampler",
            "question_rng": launcher._question_rng_contract(),
            "target_recipe": launcher.TARGET_RECIPE,
            "target_validity_mask": [False, False, False],
            "incoming_ball_fields": list(launcher.INCOMING_BALL_FIELDS),
            "reset_inverse_solve": False,
            "online_solver_calls": 0,
            "online_lm_calls": 0,
            "seed": 0,
            "observed_oracle_bundle_artifact": {
                "path": str(observed_path),
                "sha256": hashlib.sha256(observed_path.read_bytes()).hexdigest(),
            },
            "observed_oracle_bundle_content_sha256": launcher.canonical_sha256(
                observed_document
            ),
            "raw_oracle_artifact": {
                "path": str(raw_oracle),
                "sha256": hashlib.sha256(raw_oracle.read_bytes()).hexdigest(),
            },
            "raw_oracle_kind": launcher.C211_RAW_ORACLE_KIND,
            "raw_oracle_content_sha256": raw_document["content_sha256"],
            "control_step_denominator": 32,
            "selected_rubber_episode_denominator": 32,
            "actual_selected_rubber_contact_count": 32,
            "termination_census": {
                **_expected_termination_census(
                    _terminal_rows(terminations),
                    wait_only_reset_excluded=wait_only_reset_excluded,
                ),
                "implementation_strict_zero": {
                    **_expected_termination_census(
                        _terminal_rows(terminations)
                    )["implementation_strict_zero"],
                    "projection_nonfinite_count": projection_nonfinite,
                },
            },
        }
    )
    oracle_result = _result(
        tmp_path / "oracle32.result.json",
        stage="oracle32",
        materialization=materialization,
        policy=policy,
        oracle=oracle,
        namespace=oracle_namespace,
    )
    scale_namespace, _checkpoint_path, checkpoint, _log_path, acceptance = (
        _scale4096_terminal_fixture(tmp_path, checkout)
    )
    _FakeTorch.checkpoint = checkpoint
    _FakeTorch.load_error = None
    scale_result = _result(
        tmp_path / "scale4096.result.json",
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
        namespace=scale_namespace,
        terminal_acceptance=acceptance,
    )
    return materialize, recipe_result, oracle_result, scale_result


def _case(
    tmp_path: Path,
    *,
    stage: str,
    recipe_id: str = launcher.C_OBS_NOISE_OFF_CELL_ID,
    allow_colocation: bool = False,
    terminations: "list[tuple[str, list[str]]] | None" = None,
    wait_only_reset_excluded: int = 0,
    projection_nonfinite: int = 0,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    lineage = _lineage(checkout)
    lineage_path = checkout / "c211_lineage.json"
    lineage_sha = _write(lineage_path, lineage)
    materialize, recipe, oracle, scale = _chain(
        tmp_path,
        checkout,
        lineage_sha,
        lineage,
        recipe_id=recipe_id,
        terminations=terminations,
        wait_only_reset_excluded=wait_only_reset_excluded,
        projection_nonfinite=projection_nonfinite,
    )
    root = (
        checkout
        / launcher._B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / launcher.EXPERIMENT_NAME
    )
    root.mkdir(parents=True, exist_ok=True)
    namespace = root / ("c211-" + stage)
    budget = launcher.BUDGETS[stage]
    four_grid_receipt_path = tmp_path / "c211.four-grid-scale4096.json"
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
        "recipe_id": recipe_id,
        "lineage": {"path": lineage_path.name, "sha256": lineage_sha},
        "materialization_result": None if stage == "materialize" else materialize,
        "recipe_result": None if stage in ("materialize", "recipe") else recipe,
        "oracle32_result": oracle if stage in ("scale4096", "long4096") else None,
        "predecessor_result": scale if stage == "long4096" else None,
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
    spec_path = tmp_path / (stage + ".spec.json")
    _write(spec_path, spec)
    return spec_path, spec, lineage


def _runtime_source_fixture(checkout: Path):
    output = {}
    for path, label in launcher.RUNTIME_SOURCE_PATHS:
        target = checkout / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("fixture: %s\n" % label, encoding="utf-8")
        output[label] = {
            "path": path,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    return output


def test_c211_runtime_source_closure_seals_live_oracle_adapter():
    assert (
        launcher.C211_LIVE_ORACLE_SOURCE,
        "C211 live runtime oracle adapter",
    ) in launcher.RUNTIME_SOURCE_PATHS


def _patch_plan_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)

    class FakeTrainingContractModule:
        @staticmethod
        def validate_schema3_contract_structure(contract):
            if contract.get("schema_version") != 3:
                raise ValueError("fixture schema-3 contract differs")

        @staticmethod
        def validate_action_ball_training_authorization(contract):
            return contract.get("target_mode") == "action_ball"

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
        lambda checkout: FakeTrainingContractModule,
    )
    monkeypatch.setattr(
        launcher._FRAME0,
        "_dr_l0_manifest_binding",
        lambda checkout, commit, *, family, task_profile: dict(_DR_L0_BINDING),
    )
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda checkout, commit: {
            "checkout": str(checkout),
            "commit_sha": commit,
            "clean": True,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_runtime_sources",
        lambda checkout, commit: _runtime_source_fixture(checkout),
    )
    monkeypatch.setattr(
        launcher._B,
        "_validate_runtime_asset_environment",
        lambda: {"kind": "test_runtime_assets"},
    )
    monkeypatch.setattr(
        launcher._B, "_validate_runtime_asset_claim", lambda value: None
    )

    def verify(checkout, commit, pin, *, name):
        path = checkout / pin["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
            raise launcher.LaunchRefused("%s file SHA differs" % name)
        return dict(pin), path

    monkeypatch.setattr(launcher._B, "_verify_tracked_file", verify)
    # [已删除 2026-08-05 安全门精简] 这里曾 stub 掉 _verify_frame0_artifact_source_commit /
    # _verify_frame0_probe_source_commit / _verify_commit_ancestor。C211 的后两个只是转发给
    # A211 的同名实现的薄壳, 三个的唯一调用者都是已退役的
    # _validate_retired_exact_frame0_lineage, 现在全仓零调用点, stub 已空转。原 stub 见 git 历史。
    monkeypatch.setattr(
        launcher, "_verify_c211_runtime_authorities", lambda _checkout: None
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


def _rewrite_oracle_raw(
    spec_path: Path, spec: dict, mutate, *, reseal_raw: bool = True
) -> None:
    result_path = Path(spec["oracle32_result"]["path"])
    result = json.loads(result_path.read_text())
    receipt = result["oracle32_receipt"]
    raw_path = Path(receipt["raw_oracle_artifact"]["path"])
    raw = json.loads(raw_path.read_text())
    mutate(raw)
    if reseal_raw:
        unsigned_raw = dict(raw)
        unsigned_raw.pop("content_sha256", None)
        raw["content_sha256"] = launcher.canonical_sha256(unsigned_raw)
    raw_file_sha = _write(raw_path, raw)
    receipt["raw_oracle_artifact"]["sha256"] = raw_file_sha
    receipt["raw_oracle_content_sha256"] = raw.get(
        "content_sha256", "0" * 64
    )
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("content_sha256")
    receipt["content_sha256"] = launcher.canonical_sha256(unsigned_receipt)
    unsigned_result = dict(result)
    unsigned_result.pop("content_sha256")
    result["content_sha256"] = launcher.canonical_sha256(unsigned_result)
    spec["oracle32_result"]["sha256"] = _write(result_path, result)
    _write(spec_path, spec)


def test_two_code_owned_c211_recipes_and_five_stage_chain_are_exact():
    assert tuple(launcher.BUDGETS) == (
        "materialize",
        "recipe",
        "oracle32",
        "scale4096",
        "long4096",
    )
    assert launcher.BUDGETS == {
        "materialize": (1, 0, 1),
        "recipe": (1, 0, 1),
        "oracle32": (1, 0, 1),
        "scale4096": (4096, 5, 1),
        "long4096": (4096, 1000, 100),
    }
    # 2026-08-05 第二轴改版(第二次,exp §5.6.2d):两格共用 fixed lr1e-4,探索包也共用
    # (标准初始化 + sigma 1.0 + scalar);唯一差异是本体感观测噪声开关。
    expected_ppo = {
        launcher.C_OBS_NOISE_OFF_CELL_ID: ("fixed", 1.0e-4),
        launcher.C_OBS_NOISE_ON_CELL_ID: ("fixed", 1.0e-4),
    }
    expected_exploration = {
        launcher.C_OBS_NOISE_OFF_CELL_ID: ("default", 1.0, "scalar", False),
        launcher.C_OBS_NOISE_ON_CELL_ID: ("default", 1.0, "scalar", False),
    }
    expected_observation_noise = {
        launcher.C_OBS_NOISE_OFF_CELL_ID: (
            False,
            launcher.DR_LEVEL_IDENTITY_OBS_NOISE_OFF,
            None,
        ),
        launcher.C_OBS_NOISE_ON_CELL_ID: (
            True,
            launcher.DR_LEVEL_IDENTITY_OBS_NOISE_ON,
            launcher._F.PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS,
        ),
    }
    assert launcher.RECIPE_IDS == tuple(expected_ppo)
    manifest = launcher._isaac_four_grid_manifest()
    for recipe_id, (schedule, learning_rate) in expected_ppo.items():
        recipe = launcher._recipe_contract(recipe_id)
        cell = launcher._four_grid_cell(recipe_id, task_family="C211")
        assert recipe["recipe_id"] == recipe_id
        assert recipe["four_grid_cell_id"] == recipe_id
        assert recipe["isaac_four_grid_manifest_sha256"] == manifest[
            "content_sha256"
        ]
        assert recipe["reference_guard_mode"] == "metrics_only"
        assert recipe["ppo"] == cell["ppo"]
        assert recipe["ppo"]["schedule"] == schedule
        assert recipe["ppo"]["learning_rate"] == learning_rate
        assert (
            recipe["actor_init_mode"],
            recipe["init_noise_std"],
            recipe["noise_std_type"],
            recipe["four_sigma_hard_inner_gate_applies"],
        ) == expected_exploration[recipe_id]
        assert (
            recipe["policy_observation_corruption"],
            recipe["dr_level_identity"],
            recipe["proprioceptive_observation_noise_channels"],
        ) == expected_observation_noise[recipe_id]
        # 任务通道永远无噪:那会改支撑集,等于换题。
        assert recipe["task_channel_observation_noise"] is False
        assert recipe["ppo_adaptation_axis"] == cell["ppo_adaptation_axis"]
        assert recipe["contact_sigma_adaptation"] is False
        assert recipe["actor_width"] == 211
        assert recipe["critic_width"] == 319
        assert list(recipe).count("trainability_contract") == 1
        assert recipe["trainability_contract"] == launcher.TRAINABILITY_CONTRACT
        assert recipe["fresh_normalizers_required"] is True
        assert recipe["foreign_checkpoint_reuse_prohibited"] is True
    # Match the exact shared schema-3 field spellings.  c225_reward_contract
    # is a retained key name whose value identity is C211 reward-v3.
    assert launcher._hard_wait_contract()["task_valid_actor_and_critic"] is True
    source_contract = launcher._hard_question_source_contract()
    assert source_contract["question_sampler"] == {
        "source": "runtime_curriculum_sampler",
        "cadence": "every_episode_reset",
        "curriculum_domain_levels_consulted_every_reset": True,
        "sampler_runs_every_reset": True,
        "initial_center_single_question": True,
        "initial_center_activation": "all_32_domain_levels_exact_zero",
        "initial_center_physical_support": "literal_profile_center_point",
        "initial_center_rng_draws": "normal_fixed_budget",
        "post_promotion_support": "zero_to_manifest_max_width_per_promoted_arm",
        "sampler_rng_reused_by_target_provider": False,
        "physical_rng_draw_count_authority": (
            "sample_receipt_draw_end_minus_draw_start"
        ),
        "zero_physical_rng_draw_claim_permitted": False,
        "selection": "sample_current_domain_levels",
        "checkpoint_resume": "exact_sampler_and_curriculum_state",
    }
    assert source_contract["target_provider"] == {
        "source": "direct_ball",
        "desired_contact_inverse": False,
        "exact_question_answer_cache": {"enabled": False},
        "online_inverse_solves_per_reset": 0,
        "online_inverse_solves_per_step": 0,
    }


@pytest.mark.parametrize(
    "retired_recipe_id",
    (
        "C0-corrected-metrics-fixedlr",
        "C-retired-fixed-lr1e3",
        "C-retired-adaptive-lr1e4",
    ),
)
def test_non_grid_c211_recipe_ids_are_rejected(retired_recipe_id):
    with pytest.raises(launcher.LaunchRefused, match="two formal C211 grid cells"):
        launcher._recipe_contract(retired_recipe_id)


def test_c211_leaf_freezes_4096_stable_plant_and_zero_base_shaping():
    profile = (
        SCRIPT.parent.parent
        / "cfg/task/HOPEPingPongActionBallC211VendorV2N1Learnability.yaml"
    ).read_text(encoding="utf-8")
    for marker in (
        "num_envs: 4096",
        "stable_ready_plant: true",
        "control_step_action_delay_min: 0",
        "control_step_action_delay_max: 0",
        "base_position_weight: 0.0",
        "adaptive_sigma: false",
        "adaptive_sigma_monotonic: false",
        "adaptive_sigma_normal: false",
    ):
        assert marker in profile
    assert launcher.BUDGETS["scale4096"][0] == 4096
    assert launcher.BUDGETS["long4096"][0] == 4096


def test_c211_reward_contract_is_exact_and_rejects_duplicate_or_target_income():
    contract = launcher._c211_reward_contract()
    assert contract["strike_bridge"] == {
        "term": "c225_strike_ball_paddle_center_proximity",
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.action_ball_c225_rewards."
            "c225_strike_ball_paddle_center_proximity"
        ),
        "weight": 240.0,
        "std_m": 0.15,
        "kernel": "cauchy_inverse_quadratic",
        "eligibility": "task_valid_active_swing_single_exact_strike_tick",
        "miss_retains_gradient": True,
    }
    assert contract["identity"] == "action_ball_c211_achieved_outcome_reward_v3"
    assert contract["task_valid_required"] is True
    assert contract["economics"] == {
        "policy_dt_s": 0.02,
        "task_valid_swing_mimic_undiscounted_cap": 2.8325,
        "task_reveal_discounted_gamma": 0.99,
        "task_reveal_contact_tick": 92,
        "task_valid_swing_mimic_discounted_cap": 1.7733077595610476,
        "strike_bridge_post_dt_peak": 4.8,
        "strike_bridge_discounted_at_contact": 1.9040534708257204,
        "legal_landing_post_dt_min": 8.4,
        "legal_landing_discounted_at_contact_min": 3.332093573945011,
        "ordering": "task_valid_swing_mimic_lt_strike_peak_lt_legal_landing",
    }
    assert contract["landing"]["weight"] == 700.0
    assert contract["landing"]["legal_opponent_table"] == (
        "0.6_plus_0.4_gaussian"
    )
    assert contract["landing"]["opponent_side_off_table"] == (
        "0.5_times_same_gaussian"
    )
    assert contract["landing"]["observed_physical_landing_available"] is False

    terms = [
        {
            "name": name,
            "callable": required["callable"],
            "weight": required["weight"],
            "params": copy.deepcopy(required["params"]),
        }
        for name, required in launcher.REQUIRED_OUTCOME_TERMS.items()
    ]
    terms.extend(
        {
            "name": name,
            "callable": required["callable"],
            "weight": 0.1,
            "params": copy.deepcopy(required["params"]),
        }
        for name, required in launcher.REQUIRED_PRIOR_TERMS.items()
    )
    launcher._require_c211_outcome_terms(terms)

    assert launcher.ALLOWED_TASK_DIRECTED_TERMS == {
        "c225_strike_ball_paddle_center_proximity",
        "virtual_landing",
    }


@pytest.mark.parametrize("name", launcher.PROHIBITED_TASK_DIRECTED_TERMS)
def test_c211_strict_task_reward_denylist_rejects_every_active_term(name):
    terms = [
        {
            "name": term_name,
            "callable": required["callable"],
            "weight": required["weight"],
            "params": copy.deepcopy(required["params"]),
        }
        for term_name, required in launcher.REQUIRED_OUTCOME_TERMS.items()
    ]
    terms.extend(
        {
            "name": term_name,
            "callable": required["callable"],
            "weight": 0.1,
            "params": copy.deepcopy(required["params"]),
        }
        for term_name, required in launcher.REQUIRED_PRIOR_TERMS.items()
    )
    terms.append(
        {
            "name": name,
            "callable": "pkg." + name,
            "weight": 1.0,
            "params": {},
        }
    )
    with pytest.raises(launcher.LaunchRefused, match="prohibited task-directed"):
        launcher._require_c211_outcome_terms(terms)


def test_c211_strict_task_reward_allowlist_rejects_unregistered_term():
    terms = [
        {
            "name": name,
            "callable": required["callable"],
            "weight": required["weight"],
            "params": copy.deepcopy(required["params"]),
        }
        for name, required in launcher.REQUIRED_OUTCOME_TERMS.items()
    ]
    terms.extend(
        {
            "name": name,
            "callable": required["callable"],
            "weight": 0.1,
            "params": copy.deepcopy(required["params"]),
        }
        for name, required in launcher.REQUIRED_PRIOR_TERMS.items()
    )
    terms.append(
        {
            "name": "virtual_unregistered_shaping",
            "callable": "pkg.unregistered",
            "weight": 1.0,
            "params": {},
        }
    )
    with pytest.raises(launcher.LaunchRefused, match="unregistered task-directed"):
        launcher._require_c211_outcome_terms(terms)


def test_c211_launcher_and_evidence_pin_the_v2_actor_abi():
    assert launcher.TRAINABILITY_CONTRACT == (
        "action_ball_c211_fixed_midpoint_learnability_v2"
    )
    assert launcher.ACTOR_NORMALIZER_IDENTITY == "action_ball_c211_actor_norm_v2"
    assert producer.TRAINABILITY_CONTRACT == launcher.TRAINABILITY_CONTRACT
    assert producer.ACTOR_NORMALIZER_IDENTITY == launcher.ACTOR_NORMALIZER_IDENTITY
    assert launcher.CRITIC_CONTRACT == "action_ball_c211_critic_v1"
    assert launcher.CRITIC_WIDTH == 319
    assert launcher.CRITIC_NORMALIZER_IDENTITY == "action_ball_c211_critic_norm_v1"


def test_c211_claim_seals_split_ready_hidden_wait_and_teacher_bridge(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    authority = plan["canonical_payload"]["bundle"]["lineage"][
        "split_ready_reset_wait_authority"
    ]
    unsigned = dict(authority)
    claim_sha256 = unsigned.pop("claim_sha256")
    assert authority["kind"] == launcher._FRAME0.SPLIT_READY_RESET_WAIT_GATE_KIND
    assert claim_sha256 == launcher.canonical_sha256(unsigned)
    assert plan["launch_claim_sha256"] == launcher.canonical_sha256(
        plan["canonical_payload"]
    )
    checkout = Path(spec["source"]["checkout"])
    receipt_pin = lineage_doc["dynamic_ready_nominal_receipt"]
    assert authority["hidden_wait_required_policy_steps"] == 25
    assert authority["hidden_wait_required_physics_steps"] == 100
    assert authority["observed_policy_steps"] == 60
    assert authority["observed_physics_steps"] == 240
    assert authority["physical_reset_source"] == "dynamic_ready.physical_ready"
    assert authority["teacher_source"] == "measured_motion.frame0"
    # C 与 A 同律: 等待 = 到达时间 - 缩放后的命中时刻, 就是运行时算的那条。
    receipt = json.loads(
        INITIAL_CENTER_C_RECEIPT_SOURCE.read_text(encoding="utf-8")
    )
    assert authority["time_to_teacher_start_at_reveal_s"] == (
        receipt["time_to_contact_s"] - receipt["scaled_t_hit_s"]
    )
    assert authority["time_to_teacher_start_at_reveal_s"] == pytest.approx(
        0.6923799138976297
    )
    assert authority["initial_center_timing_authority"]["timing_mode"] == (
        "c_direct_ball"
    )
    assert authority["bridge_learning_signal"] == "dense_mimic_after_task_reveal"
    assert authority["passive_hold_after_reveal_required"] is False
    argv = plan["canonical_payload"]["training_argv"]
    assert (
        "action_ball_dynamic_ready_nominal_receipt_path=%s"
        % (checkout / receipt_pin["path"])
    ) in argv
    assert (
        "action_ball_dynamic_ready_nominal_receipt_sha256=%s"
        % receipt_pin["sha256"]
    ) in argv


def _initial_center_case(tmp_path: Path):
    """Checkout + lineage + the untouched producer receipt the C gate reads."""

    checkout = tmp_path / "initial-center-checkout"
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
    return lineage, manifest, receipt


def _c_timing(lineage, manifest, receipt):
    return launcher._FRAME0._initial_center_timing_authority(
        receipt=receipt,
        receipt_pin=lineage["initial_center_task_receipt"],
        action_manifest=manifest,
        action_manifest_pin=lineage["action_manifest"],
        motion_sha256=lineage["motion"]["sha256"],
        family="C",
    )


def _resealed(receipt: dict, **changes) -> dict:
    """Mutation helper.  Only ever used to prove the gate REFUSES something."""

    mutated = dict(receipt)
    mutated.pop("canonical_sha256")
    mutated.update(changes)
    mutated["canonical_sha256"] = launcher.canonical_sha256(mutated)
    return mutated


def _runtime_timing(receipt: dict):
    return runtime.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=receipt["racket_site_velocity_w_mps"],
        time_to_contact_s=receipt["time_to_contact_s"],
        reference_t_hit_s=receipt["reference_t_hit_s"],
        reference_t_cycle_s=receipt["reference_t_cycle_s"],
        reference_racket_site_speed_mps=(
            receipt["reference_racket_site_speed_mps"]
        ),
        reaction_margin_s=receipt["reaction_margin_s"],
        teacher_rate_min=receipt["teacher_rate_min"],
        teacher_rate_max=receipt["teacher_rate_max"],
    )


def test_c211_initial_center_timing_is_the_runtime_wait_law(tmp_path):
    """C 的定时门算的就是运行时那条式子, 不是一条平行的 C 专用律。

    运行时 (``hope_commands``) 对所有题源, 包括 C 的 ``direct_ball``, 一律算
    ``pre_swing_wait_s = time_to_contact_s - scaled_t_hit_s``, 并用
    ``derive_action_teacher_site_timing`` 逐字段复核自己吐出的收据。这里直接调同一个
    函数, 证明 producer 原样吐出的收据一位不差地满足它, 且发射门给出同一个数。
    """

    lineage, manifest, receipt = _initial_center_case(tmp_path)
    live = _runtime_timing(receipt)
    assert live.teacher_rate == receipt["teacher_rate"]
    assert live.scaled_t_hit_s == receipt["scaled_t_hit_s"]
    assert live.pre_swing_wait_s == receipt["pre_swing_wait_s"]
    assert live.pre_swing_wait_s == (
        receipt["time_to_contact_s"] - receipt["scaled_t_hit_s"]
    )

    timing = _c_timing(lineage, manifest, receipt)
    assert timing["derivation"] == "time_to_contact_s_minus_scaled_t_hit_s"
    assert timing["timing_mode"] == "c_direct_ball"
    assert timing["family"] == "C"
    assert timing["initial_center_time_to_teacher_start_at_reveal_s"] == (
        live.pre_swing_wait_s
    )
    # 载体速率差约 15%: 教师 FK 载体 ~1.0, 这份 current_lm 载体 0.8513...。按 Franco
    # 2026-08-06 裁定接受, 交给 policy 泛化。
    assert timing["scaled_t_hit_s"] != timing["reference_t_hit_s"]
    assert 0.84 < receipt["teacher_rate"] < 0.86


def test_c211_initial_center_timing_refuses_a_tampered_wait(tmp_path):
    """把 wait 改掉再重封 SHA, 门必须照样拒 —— 改门不等于放宽门。"""

    lineage, manifest, receipt = _initial_center_case(tmp_path)
    assert _c_timing(lineage, manifest, receipt)

    for wait in (
        # 退役的那条 C 律: 等价于强制 teacher_rate 恰好 1.0。运行时从没实现过它,
        # 任何 producer 也没产出过满足它的收据。
        receipt["time_to_contact_s"] - receipt["reference_t_hit_s"],
        # 极小篡改也要拒: 门是精确相等, 不是 approx。
        receipt["pre_swing_wait_s"] + 1.0e-15,
        receipt["pre_swing_wait_s"] - 1.0e-15,
        receipt["pre_swing_wait_s"] + 0.05,
    ):
        tampered = _resealed(receipt, pre_swing_wait_s=wait)
        with pytest.raises(
            launcher._FRAME0.LaunchRefused, match="timing derivation differs"
        ):
            _c_timing(lineage, manifest, tampered)

    # 同族的其他时钟字段一并 fail-closed, 免得只把 wait 挡住而放过速率。
    for changes in (
        {"teacher_rate": receipt["teacher_rate"] * 1.01},
        {"scaled_t_hit_s": receipt["scaled_t_hit_s"] + 1.0e-15},
        {"time_to_contact_s": receipt["time_to_contact_s"] + 0.02},
    ):
        with pytest.raises(launcher._FRAME0.LaunchRefused):
            _c_timing(lineage, manifest, _resealed(receipt, **changes))


def test_c211_initial_center_timing_goes_red_if_the_runtime_clock_moves(
    tmp_path, monkeypatch
):
    """动运行时的公式, C 的门就变红 —— 证明门跟的是运行时, 不是一条抄下来的常量。"""

    lineage, manifest, receipt = _initial_center_case(tmp_path)
    original = runtime.derive_action_teacher_timing

    def shifted_clock(**kwargs):
        timing = original(**kwargs)
        return runtime.ActionTeacherTiming(
            required_racket_site_speed_mps=(
                timing.required_racket_site_speed_mps
            ),
            teacher_rate=timing.teacher_rate,
            scaled_t_hit_s=timing.scaled_t_hit_s,
            scaled_t_cycle_s=timing.scaled_t_cycle_s,
            pre_swing_wait_s=timing.pre_swing_wait_s - 0.05,
        )

    monkeypatch.setattr(runtime, "derive_action_teacher_timing", shifted_clock)
    moved = _runtime_timing(receipt)
    assert moved.pre_swing_wait_s != receipt["pre_swing_wait_s"]
    assert moved.scaled_t_hit_s == receipt["scaled_t_hit_s"]

    # 收据由"改过公式的运行时"产出 —— 现在的 C 门必须拒。
    with pytest.raises(
        launcher._FRAME0.LaunchRefused, match="timing derivation differs"
    ):
        _c_timing(
            lineage,
            manifest,
            _resealed(receipt, pre_swing_wait_s=moved.pre_swing_wait_s),
        )

    monkeypatch.undo()
    assert _runtime_timing(receipt).pre_swing_wait_s == (
        receipt["pre_swing_wait_s"]
    )
    assert _c_timing(lineage, manifest, receipt)[
        "initial_center_time_to_teacher_start_at_reveal_s"
    ] == receipt["pre_swing_wait_s"]


def test_c211_initial_center_receipt_is_producer_bytes_never_resealed(tmp_path):
    """夹具用的是 producer 原样吐出的字节, 没有手改数值再重封 canonical_sha256。"""

    lineage, _manifest, receipt = _initial_center_case(tmp_path)
    source = json.loads(
        INITIAL_CENTER_C_RECEIPT_SOURCE.read_text(encoding="utf-8")
    )
    assert receipt == source
    assert lineage["initial_center_task_receipt"]["sha256"] == hashlib.sha256(
        INITIAL_CENTER_C_RECEIPT_SOURCE.read_bytes()
    ).hexdigest()
    unsigned = dict(source)
    seal = unsigned.pop("canonical_sha256")
    assert seal == launcher.canonical_sha256(unsigned)


def test_c211_target_recipe_carrier_mismatch_stays_out_of_the_runtime(tmp_path):
    """载体错配核查: 收据的载体不是 C 运行时的载体, 但运行时一个字段都不读它。

    ``outcome_dense_only`` 的 producer 载体是 ``current_lm`` 逆解速度
    (``coherent_current_lm_carrier_mask_all_targets``), 而 C 的运行时在
    ``direct_ball`` 分支用的是教师 FK 参考行。差的只是这份标定收据里的速率/等待数字,
    它既不进 argv 也不是运行时题源, 所以是无害错配。
    """

    lineage, manifest, receipt = _initial_center_case(tmp_path)
    assert lineage["target_recipe"] == "outcome_dense_only"
    assert lineage["target_source"] == "direct_ball"
    assert lineage["reset_inverse_solve"] is False
    assert lineage["online_lm_calls"] == 0
    assert lineage["online_solver_calls"] == 0
    assert lineage["question_source"] == "runtime_curriculum_sampler"

    timing = _c_timing(lineage, manifest, receipt)
    assert timing["role"] == "calibration_receipt_not_runtime_question_source"
    # 收据里 C 用不到的逆解载体字段(位置/朝向/残差)从来没进过标定凭据。
    carrier_only = (
        "racket_site_target_w_m",
        "racket_site_velocity_w_mps",
        "racket_face_center_velocity_w_mps",
        "racket_command_quat_wxyz",
        "racket_normal_w",
        "solver_residual_m",
    )
    assert all(field in receipt for field in carrier_only)
    assert not any(field in timing for field in carrier_only)


def _audit_scale_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    checkpoint_mutation=None,
    log_mutation=None,
):
    checkout = tmp_path / "terminal-checkout"
    checkout.mkdir()
    namespace, checkpoint_path, checkpoint, log_path, _acceptance = (
        _scale4096_terminal_fixture(tmp_path, checkout)
    )
    if checkpoint_mutation is not None:
        checkpoint_mutation(checkpoint_path, checkpoint)
    if log_mutation is not None:
        log_mutation(log_path)
    _FakeTorch.checkpoint = checkpoint
    _FakeTorch.load_error = None
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    return launcher._audit_scale4096_terminal(
        checkout=checkout,
        namespace=namespace,
        launch_claim_sha256="1" * 64,
    )


def test_scale4096_terminal_audit_accepts_exact_model5_and_five_updates(
    tmp_path, monkeypatch
):
    acceptance = _audit_scale_terminal(tmp_path, monkeypatch)
    assert Path(acceptance["checkpoint"]["path"]).name == "model_5.pt"
    assert acceptance["checkpoint"]["embedded_iteration"] == 5
    assert acceptance["checkpoint"]["map_location"] == "cpu"
    assert acceptance["checkpoint"]["load_mode"] == "torch_weights_only"
    assert set(acceptance["checkpoint"]["tensor_groups"]) == {
        "model",
        "optimizer",
        "actor_normalizer",
        "critic_normalizer",
    }
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


def test_scale4096_terminal_audit_rejects_missing_exact_model5(
    tmp_path, monkeypatch
):
    def remove_checkpoint(path, _checkpoint):
        path.unlink()

    with pytest.raises(launcher.LaunchRefused, match="checkpoint.*missing"):
        _audit_scale_terminal(
            tmp_path,
            monkeypatch,
            checkpoint_mutation=remove_checkpoint,
        )


@pytest.mark.parametrize("binding", ("iteration", "claim"))
def test_scale4096_terminal_audit_rejects_wrong_checkpoint_binding(
    tmp_path, monkeypatch, binding
):
    def mutate_binding(_path, checkpoint):
        if binding == "iteration":
            checkpoint["iter"] = 4
        else:
            checkpoint["infos"]["training_launch_claim_sha256"] = "2" * 64

    with pytest.raises(
        launcher.LaunchRefused, match="iteration/launch-claim binding differs"
    ):
        _audit_scale_terminal(
            tmp_path,
            monkeypatch,
            checkpoint_mutation=mutate_binding,
        )


def test_scale4096_terminal_audit_rejects_truncated_five_update_telemetry(
    tmp_path, monkeypatch
):
    def truncate(log_path):
        lines = log_path.read_text(encoding="utf-8").splitlines()
        prefix = "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
        removed = False
        kept = []
        for line in lines:
            if line.startswith(prefix) and not removed:
                removed = True
                continue
            kept.append(line)
        assert removed
        log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    with pytest.raises(
        launcher.LaunchRefused, match="lacks exactly 5 contiguous terminal updates"
    ):
        _audit_scale_terminal(tmp_path, monkeypatch, log_mutation=truncate)


@pytest.mark.parametrize(
    "state_key",
    (
        "model_state_dict",
        "optimizer_state_dict",
        "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    ),
)
def test_scale4096_terminal_audit_rejects_nonfinite_checkpoint_tensor(
    tmp_path, monkeypatch, state_key
):
    def inject_nonfinite(_path, checkpoint):
        checkpoint[state_key] = {"bad": _FakeTensor([float("nan")])}

    with pytest.raises(launcher.LaunchRefused, match="non-finite tensor"):
        _audit_scale_terminal(
            tmp_path,
            monkeypatch,
            checkpoint_mutation=inject_nonfinite,
        )


@pytest.mark.parametrize(
    "counter_kind",
    ("hard", "joint_qdes", "joint_actual", "nonfinite"),
)
def test_scale4096_terminal_audit_rejects_wrong_safety_telemetry(
    tmp_path, monkeypatch, counter_kind
):
    def inject_wrong_counter(log_path):
        rewritten = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            prefix, separator, payload = line.partition("=")
            if not separator or not prefix.startswith("HOPE_"):
                rewritten.append(line)
                continue
            row = json.loads(payload)
            if row.get("ppo_update") == 0:
                if counter_kind == "hard" and prefix == "HOPE_JOINT_SAFETY_UPDATE_JSON":
                    row["counter_totals"]["actual_hard_edge_events"] = 1
                elif (
                    counter_kind == "table"
                    and prefix == "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON"
                ):
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
                elif (
                    counter_kind == "nonfinite"
                    and prefix == "HOPE_EXACT_BEHAVIOR_UPDATE_JSON"
                ):
                    row["counters"]["ready_nonfinite_value_count"] = 1
            rewritten.append(
                prefix
                + "="
                + json.dumps(row, sort_keys=True, separators=(",", ":"))
            )
        log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    with pytest.raises(
        launcher.LaunchRefused,
        match="implementation counters are nonzero",
    ):
        _audit_scale_terminal(
            tmp_path, monkeypatch, log_mutation=inject_wrong_counter
        )


def test_scale4096_terminal_audit_reports_physical_fall_without_rejecting_finite(
    tmp_path, monkeypatch
):
    def inject_fall(log_path):
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
                        {"termination_terms": ["base_too_low"]}
                    ]
                elif prefix == "HOPE_EXACT_BEHAVIOR_UPDATE_JSON":
                    row["counters"]["termination_reason_base_too_low_count"] = 1
                    row["counters"][
                        "termination_reason_base_too_low_post_strike_count"
                    ] = 1
            rewritten.append(
                prefix
                + "="
                + json.dumps(row, sort_keys=True, separators=(",", ":"))
            )
        log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    accepted = _audit_scale_terminal(
        tmp_path, monkeypatch, log_mutation=inject_fall
    )

    assert accepted["safety_counters"]["base_too_low_terminal_count"] == 1
    assert accepted["safety_counters"]["strict_hard_termination_count"] == 0
    assert accepted["prelong_gate"]["gate"]["status"] == "PASS"


def test_scale4096_terminal_audit_reports_table_by_phase_without_rejecting_finite(
    tmp_path, monkeypatch
):
    def inject_table(log_path):
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
                    row["counters"][
                        "termination_reason_robot_hit_table_count"
                    ] = 1
                    row["counters"][
                        "termination_reason_robot_hit_table_post_strike_count"
                    ] = 1
            rewritten.append(
                prefix
                + "="
                + json.dumps(row, sort_keys=True, separators=(",", ":"))
            )
        log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    accepted = _audit_scale_terminal(
        tmp_path, monkeypatch, log_mutation=inject_table
    )

    assert accepted["safety_counters"]["table_contact_count"] == 1
    assert accepted["safety_counters"]["strict_hard_termination_count"] == 0
    table = accepted["prelong_gate"]["gate"]["survival_denominators"][
        "robot_hit_table"
    ]
    assert table["by_phase"]["post_strike"] == 1
    assert table["acceptance_threshold"] is None


def test_scale4096_terminal_audit_rejects_fall_reason_phase_nonconservation(
    tmp_path, monkeypatch
):
    def inject_bad_phase(log_path):
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
                row["counters"]["termination_reason_base_fell_tilt_count"] = 1
            rewritten.append(
                prefix
                + "="
                + json.dumps(row, sort_keys=True, separators=(",", ":"))
            )
        log_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    with pytest.raises(launcher.LaunchRefused, match="reason-by-phase counters"):
        _audit_scale_terminal(
            tmp_path, monkeypatch, log_mutation=inject_bad_phase
        )


def test_long_plan_seals_true_c211_question_and_fresh_state(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="long4096")
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert payload["fresh_only"] is True
    assert payload["ppo_updates_authorized"] == 1000
    assert payload["output_contract"]["requested_ppo_update_count"] == 1000
    assert payload["output_contract"]["runtime_gate"] == "READY"
    assert payload["output_contract"]["runtime_dependencies"] == []
    assert payload["reset_inverse_solve"] is False
    assert payload["materialization_inputs"]["four_grid_scale4096_receipt"][
        "status"
    ] == "PASS"
    assert payload["bundle"]["question_contract"] == launcher._question_contract()
    assert payload["bundle"]["normalizers"] == launcher._normalizer_contract()
    assert payload["bundle"]["checkpoint_contract"]["input"] is None
    assert payload["bundle"]["checkpoint_contract"]["state"] == "fresh_empty"
    reset_wait = payload["bundle"]["lineage"][
        "split_ready_reset_wait_authority"
    ]
    assert reset_wait["claim_sha256"] == launcher.canonical_sha256(
        {
            key: value
            for key, value in reset_wait.items()
            if key != "claim_sha256"
        }
    )
    assert reset_wait["physical_reset_source"] == "dynamic_ready.physical_ready"
    assert reset_wait["teacher_source"] == "measured_motion.frame0"
    assert reset_wait["teacher_physical_birth_separated"] is True
    assert reset_wait["hidden_wait_required_policy_steps"] == 25
    assert reset_wait["observed_policy_steps"] == 60
    assert reset_wait["bridge_learning_signal"] == "dense_mimic_after_task_reveal"
    assert payload["bundle"]["curriculum_scope"] == (
        launcher._curriculum_scope_contract()
    )
    assert payload["materialization_inputs"]["predecessor_result"]["completion"][
        "terminal_kind"
    ] == "clean_completion"
    predecessor = payload["materialization_inputs"]["predecessor_result"]
    assert Path(predecessor["finite_model_artifact"]["path"]).name == "model_5.pt"
    assert predecessor["safety_counters"]["observed_ppo_updates"] == 5
    assert predecessor["prelong_gate"]["semantic_update_count"] == 5
    assert predecessor["prelong_gate"]["gate"]["status"] == "PASS"


@pytest.mark.parametrize(
    "field,retired",
    (
        ("kind", "action_ball_c225_fixed_midpoint_lineage_v1"),
        ("kind", "action_ball_c210_fixed_midpoint_lineage_v1"),
        ("actor_contract", "action_ball_c225"),
        ("actor_contract", "action_ball_c210"),
        ("actor_width", 225),
        ("actor_width", 210),
        ("critic_contract", "action_ball_c225_critic_v1"),
        ("critic_contract", "action_ball_c210_critic_v1"),
        ("critic_width", 318),
        ("trainability_contract", "action_ball_c225_fixed_midpoint_learnability_v1"),
        ("trainability_contract", "action_ball_c210_fixed_midpoint_learnability_v1"),
        ("trainability_contract", "action_ball_c211_fixed_midpoint_learnability_v1"),
        ("actor_normalizer_identity", "action_ball_c225_actor_norm_v1"),
        ("actor_normalizer_identity", "action_ball_c211_actor_norm_v1"),
        ("critic_normalizer_identity", "action_ball_c210_critic_norm_v1"),
    ),
)
def test_structurally_resealed_retired_c_lineage_is_rejected(
    tmp_path, monkeypatch, field, retired
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(tmp_path, stage="materialize")
    lineage[field] = retired
    lineage_path = Path(spec["source"]["checkout"]) / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "recipe_id,schedule,learning_rate,sigma,std_type,init_mode,corruption",
    (
        (
            launcher.C_OBS_NOISE_OFF_CELL_ID,
            "fixed",
            "0.0001",
            "1.0",
            "scalar",
            "default",
            "false",
        ),
        (
            launcher.C_OBS_NOISE_ON_CELL_ID,
            "fixed",
            "0.0001",
            "1.0",
            "scalar",
            "default",
            "true",
        ),
    ),
)
def test_training_argv_pins_c211_grid_cell_and_static_contact_sigma(
    tmp_path,
    monkeypatch,
    recipe_id,
    schedule,
    learning_rate,
    sigma,
    std_type,
    init_mode,
    corruption,
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, stage="recipe", recipe_id=recipe_id
    )
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    argv = payload["training_argv"]
    joined = "\n".join(argv)
    assert "task=%s" % launcher.TASK_PROFILE_ID in argv
    assert "task.actor_obs_contract=action_ball_c211" in joined
    assert "task.racket.action_ball_target_recipe=outcome_dense_only" in joined
    assert "task.racket.action_ball_target_source=direct_ball" in joined
    assert (
        "task.racket.action_ball_reuse_exact_question_until_semantics_change=false"
        in joined
    )
    assert "task.racket.action_ball_target_validity_mask=[false,false,false]" in joined
    assert "task.racket.action_ball_target_observation_noise=false" in joined
    assert "task.racket.adaptive_sigma=false" in joined
    assert "task.racket.adaptive_sigma_monotonic=false" in joined
    assert "task.racket.adaptive_sigma_normal=false" in joined
    assert "task.racket.target_noise_white=0.0" in joined
    assert "task.racket.target_noise_ar1_sigma=0.0" in joined
    assert not any(
        value.startswith("task.racket.action_ball_immutable_tape_")
        for value in argv
    )
    assert "task.racket.action_ball_diagnostic_unauthorized=true" in joined
    assert "+task.racket.reference_guard_mode=metrics_only" in joined
    assert "task.domain_rand.stable_ready_plant=true" in joined
    assert joined.count("task.domain_rand.stable_ready_plant=true") == 1
    assert "task.rewards.base_position_weight=0.0" in joined
    assert "task.actions.control_step_action_delay_min=0" in joined
    assert "task.actions.control_step_action_delay_max=0" in joined
    assert "algo.runner.empirical_normalization=true" in joined
    assert "algo.algorithm.schedule=" + schedule in argv
    assert "algo.algorithm.learning_rate=" + learning_rate in argv
    # 探索包本轮四格相同,所以两格的这三个 override 逐字一样。
    assert "algo.policy.init_noise_std=" + sigma in argv
    assert "algo.policy.noise_std_type=" + std_type in argv
    assert "action_ball_actor_init_mode=" + init_mode in argv
    assert joined.count("algo.policy.init_noise_std=") == 1
    assert joined.count("action_ball_actor_init_mode=") == 1
    # 唯一的注册差异轴:整包 DR 元组写进 argv,只有最后这个布尔随格变。
    assert "task.domain_rand.startup_physics_material=false" in argv
    assert "task.domain_rand.startup_joint_default_pos=false" in argv
    assert "task.domain_rand.policy_observation_corruption=" + corruption in argv
    assert joined.count("task.domain_rand.policy_observation_corruption=") == 1
    # 标准初始化格必须与一个显式 bootstrap 同时给,否则 train.py 侧 fail-closed。
    assert "action_ball_dynamic_ready_bootstrap=true" in argv
    assert payload["bundle"]["recipe"]["recipe_id"] == recipe_id
    assert payload["bundle"]["isaac_four_grid_manifest"] == (
        launcher._isaac_four_grid_manifest()
    )
    assert "resume=" not in joined.lower()
    assert "checkpoint=" not in joined.lower()
    assert "action_ball_a225" not in joined.lower()
    assert "l194" not in joined.lower()
    assert "current_lm" not in joined.lower()
    assert "online_solver" not in joined.lower()


def test_oracle_argv_uses_only_the_c211_runner_bound_bundle_hook(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="oracle32")
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    joined = "\n".join(payload["training_argv"])
    assert "+action_ball_c211_oracle_bundle_output_path=" in joined
    assert "+action_ball_c211_oracle_episodes=32" in joined
    assert "+action_ball_teacher_qdes_oracle_output_path=" not in joined
    assert payload["output_contract"]["c211_observed_oracle_bundle"].endswith(
        launcher.C211_OBSERVED_BUNDLE_FILENAME
    )


@pytest.mark.parametrize(
    "key,bad",
    (
        ("actor_contract", "action_ball_a225"),
        ("actor_width", 194),
        ("critic_width", 211),
        ("actor_normalizer_identity", "action_ball_a225_actor_norm_v1"),
        ("target_recipe", "current_lm"),
        ("target_validity_mask", [True, True, True]),
        ("incoming_ball_fields", ["desired_contact_position"]),
        ("reset_inverse_solve", True),
        ("online_solver_calls", 1),
        ("online_lm_calls", 1),
        ("target_source", "online_solver"),
        ("question_source", "immutable_tape"),
        ("question_rng", {"owner": "fixture"}),
        ("action_id", "take_061_unit05_bh"),
        ("action_uid", 1),
    ),
)
def test_lineage_rejects_foreign_or_noncausal_contract(
    tmp_path, monkeypatch, key, bad
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(tmp_path, stage="materialize")
    lineage[key] = bad
    lineage_path = Path(spec["source"]["checkout"]) / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


def test_c_lineage_cannot_relabel_legacy_bundle_as_true_c211(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, lineage = _case(tmp_path, stage="materialize")
    checkout = Path(spec["source"]["checkout"])
    bundle_path = checkout / lineage["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text())
    bundle["actor_contract"] = "action_ball_a225"
    bundle["actor_width"] = 194
    unsigned = dict(bundle)
    unsigned.pop("content_sha256")
    bundle["content_sha256"] = launcher.canonical_sha256(unsigned)
    lineage["bundle"]["sha256"] = _write(bundle_path, bundle)
    lineage_path = checkout / spec["lineage"]["path"]
    spec["lineage"]["sha256"] = _write(lineage_path, lineage)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize("stage", tuple(launcher.BUDGETS))
def test_stage_receipt_requirements_are_exact(tmp_path, stage):
    spec_path, spec, _lineage_doc = _case(tmp_path, stage=stage)
    required_key = {
        "materialize": "materialization_result",
        "recipe": "recipe_result",
        "oracle32": "oracle32_result",
        "scale4096": "predecessor_result",
        "long4096": "predecessor_result",
    }[stage]
    if stage == "materialize":
        spec[required_key] = {"path": "/tmp/foreign", "sha256": "0" * 64}
    elif stage == "recipe":
        spec[required_key] = spec["materialization_result"]
    elif stage == "oracle32":
        spec[required_key] = spec["recipe_result"]
    elif stage == "scale4096":
        spec[required_key] = spec["oracle32_result"]
    else:
        spec[required_key] = None
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_spec(spec)


def test_four_grid_aggregate_receipt_is_required_only_for_long4096(tmp_path):
    _path, spec, _lineage = _case(tmp_path / "missing", stage="long4096")
    spec["four_grid_scale4096_receipt"] = None
    with pytest.raises(launcher.LaunchRefused, match="four_grid_scale4096_receipt"):
        launcher._validate_spec(spec)

    _path, spec, _lineage = _case(tmp_path / "extra", stage="scale4096")
    spec["four_grid_scale4096_receipt"] = {
        "path": str(tmp_path / "aggregate.json"),
        "sha256": "f" * 64,
    }
    with pytest.raises(launcher.LaunchRefused, match="four_grid_scale4096_receipt"):
        launcher._validate_spec(spec)


def test_spec_rejects_disconnected_same_named_experiment_root(tmp_path):
    _, spec, _ = _case(tmp_path, stage="materialize")
    detached = tmp_path / "detached" / launcher.EXPERIMENT_NAME
    detached.mkdir(parents=True)
    namespace = detached / "fresh"
    spec["namespace"] = str(namespace)
    spec["log_path"] = str(namespace / "run.log")
    with pytest.raises(launcher.LaunchRefused, match="checkout-local C211"):
        launcher._validate_spec(spec)


def test_cross_lineage_or_broken_scale_terminal_is_rejected(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="long4096")
    predecessor = json.loads(Path(spec["predecessor_result"]["path"]).read_text())
    predecessor["completion"]["terminal_kind"] = "launch_accepted"
    predecessor_unsigned = dict(predecessor)
    predecessor_unsigned.pop("content_sha256")
    predecessor["content_sha256"] = launcher.canonical_sha256(predecessor_unsigned)
    spec["predecessor_result"]["sha256"] = _write(
        Path(spec["predecessor_result"]["path"]), predecessor
    )
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="natural-exit receipt"):
        launcher.build_plan(spec_path)


def test_long4096_reaudits_exact_scale_checkpoint_instead_of_trusting_receipt(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="long4096")
    predecessor = json.loads(Path(spec["predecessor_result"]["path"]).read_text())
    checkpoint_path = Path(predecessor["terminal_acceptance"]["checkpoint"]["path"])
    checkpoint_path.unlink()
    with pytest.raises(launcher.LaunchRefused, match="checkpoint.*missing"):
        launcher.build_plan(spec_path)


def test_long4096_reaudits_real_prelong_markers_instead_of_trusting_receipt(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="long4096")
    predecessor = json.loads(
        Path(spec["predecessor_result"]["path"]).read_text(encoding="utf-8")
    )
    log_path = Path(predecessor["terminal_acceptance"]["run_log"]["path"])
    _rewrite_prelong_marker(
        log_path,
        2,
        lambda row: row["reward_groups"][2].__setitem__(
            "eligible_denominator", 10
        ),
    )
    with pytest.raises(launcher.LaunchRefused, match="strike-group denominator"):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize("stage", ("materialize", "recipe", "oracle32"))
def test_upstream_result_requires_natural_exit_and_exact_budget(
    tmp_path, monkeypatch, stage
):
    _patch_plan_environment(monkeypatch)
    consumer = {
        "materialize": "recipe",
        "recipe": "oracle32",
        "oracle32": "scale4096",
    }[stage]
    spec_path, spec, _lineage_doc = _case(tmp_path, stage=consumer)
    key = {
        "materialize": "materialization_result",
        "recipe": "recipe_result",
        "oracle32": "oracle32_result",
    }[stage]
    result_path = Path(spec[key]["path"])
    result = json.loads(result_path.read_text())
    result["completion"]["terminal_exit_code"] = "125"
    unsigned = dict(result)
    unsigned.pop("content_sha256")
    result["content_sha256"] = launcher.canonical_sha256(unsigned)
    spec[key]["sha256"] = _write(result_path, result)
    _write(spec_path, spec)
    with pytest.raises(launcher.LaunchRefused, match="natural-exit receipt"):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "consumer,key,artifact_key",
    (
        ("recipe", "materialization_result", "runtime_effective_reward_artifact"),
        ("oracle32", "recipe_result", "runtime_policy_recipe_artifact"),
        ("scale4096", "oracle32_result", "raw_oracle_artifact"),
    ),
)
def test_consumed_receipt_requires_live_matching_artifact(
    tmp_path, monkeypatch, consumer, key, artifact_key
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage=consumer)
    result = json.loads(Path(spec[key]["path"]).read_text())
    receipt_key = {
        "materialization_result": "reward_materialization",
        "recipe_result": "policy_recipe_materialization",
        "oracle32_result": "oracle32_receipt",
    }[key]
    Path(result[receipt_key][artifact_key]["path"]).unlink()
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


def test_oracle_raw_wrong_file_hash_is_rejected(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    result = json.loads(Path(spec["oracle32_result"]["path"]).read_text())
    raw_path = Path(result["oracle32_receipt"]["raw_oracle_artifact"]["path"])
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(launcher.LaunchRefused, match="file SHA differs"):
        launcher.build_plan(spec_path)


def test_raw_schema_has_no_self_verdict_or_a_virtual_capture_semantics(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    result = json.loads(Path(spec["oracle32_result"]["path"]).read_text())
    raw_path = Path(result["oracle32_receipt"]["raw_oracle_artifact"]["path"])
    raw = json.loads(raw_path.read_text())
    assert "verdict" not in raw
    assert "capture_rejection" not in raw
    assert "preflight" not in raw
    assert raw["desired_contact_metrics"]["status"] == "INELIGIBLE"
    plan = launcher.build_plan(spec_path)["canonical_payload"]
    assert plan["ppo_updates_authorized"] == 5
    assert plan["output_contract"]["runtime_gate"] == "READY"


@pytest.mark.parametrize(
    "artifact_key",
    (
        "training_contract_artifact",
        "runner_preflight_artifact",
        "selected_rubber_contact_artifact",
    ),
)
def test_oracle_raw_nested_evidence_must_still_exist(
    tmp_path, monkeypatch, artifact_key
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    result = json.loads(Path(spec["oracle32_result"]["path"]).read_text())
    raw_path = Path(result["oracle32_receipt"]["raw_oracle_artifact"]["path"])
    raw = json.loads(raw_path.read_text())
    Path(raw[artifact_key]["path"]).unlink()
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "artifact_key",
    (
        "training_contract_artifact",
        "runner_preflight_artifact",
        "selected_rubber_contact_artifact",
    ),
)
def test_oracle_raw_nested_evidence_hash_is_checked(
    tmp_path, monkeypatch, artifact_key
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    result = json.loads(Path(spec["oracle32_result"]["path"]).read_text())
    raw_path = Path(result["oracle32_receipt"]["raw_oracle_artifact"]["path"])
    raw = json.loads(raw_path.read_text())
    artifact_path = Path(raw[artifact_key]["path"])
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    with pytest.raises(launcher.LaunchRefused, match="file SHA differs"):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "artifact_key,mutation",
    (
        (
            "runner_preflight_artifact",
            lambda row: row["facts"].__setitem__("runner_actor_width", 194),
        ),
        (
            "selected_rubber_contact_artifact",
            lambda row: row["episodes"][0].__setitem__(
                "classification", "no_contact"
            ),
        ),
    ),
)
def test_resealed_nested_runtime_claim_is_revalidated(
    tmp_path, monkeypatch, artifact_key, mutation
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")

    def mutate_nested(raw):
        artifact_path = Path(raw[artifact_key]["path"])
        artifact = json.loads(artifact_path.read_text())
        mutation(artifact)
        unsigned = dict(artifact)
        unsigned.pop("content_sha256")
        artifact["content_sha256"] = launcher.canonical_sha256(unsigned)
        raw[artifact_key]["sha256"] = _write(artifact_path, artifact)

    _rewrite_oracle_raw(spec_path, spec, mutate_nested)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


def test_resealed_hard_contract_cannot_relabel_actor_194_as_c211(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")

    def relabel(raw):
        hard_path = Path(raw["training_contract_artifact"]["path"])
        hard = json.loads(hard_path.read_text())
        hard["actor_obs_total_dim"] = 194
        hard_sha = _write(hard_path, hard)
        raw["training_contract_artifact"]["sha256"] = hard_sha
        raw["bindings"]["hard_contract_sha256"] = hard_sha
        preflight_path = Path(raw["runner_preflight_artifact"]["path"])
        preflight = json.loads(preflight_path.read_text())
        preflight["hard_contract_sha256"] = hard_sha
        unsigned = dict(preflight)
        unsigned.pop("content_sha256")
        preflight["content_sha256"] = launcher.canonical_sha256(unsigned)
        raw["runner_preflight_artifact"]["sha256"] = _write(
            preflight_path, preflight
        )

    _rewrite_oracle_raw(spec_path, spec, relabel)
    with pytest.raises(launcher.LaunchRefused, match="ABI differs"):
        launcher.build_plan(spec_path)


def test_resealed_synthetic_raw_oracle_is_not_evidence(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")

    def synthetic(raw):
        raw.clear()
        raw.update(
            {
                "schema_version": 1,
                "kind": "synthetic_oracle_pass",
                "diagnostic_unauthorized": True,
                "verdict": "PASS",
            }
        )

    _rewrite_oracle_raw(spec_path, spec, synthetic)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row.__setitem__("verdict", "PASS"),
        lambda row: row["desired_contact_metrics"].__setitem__(
            "status", "PASS"
        ),
        lambda row: row["episodes"][0]["incoming_ball_observation"][
            "critic"
        ].__setitem__(
            "incoming_ball_contact_spin_heading", [1.0, 2.0, 3.0]
        ),
        lambda row: row["question_contract"].__setitem__(
            "target_validity_mask", [True, True, True]
        ),
        lambda row: row["question_contract"].__setitem__("online_lm_calls", 1),
        lambda row: row["question_contract"].__setitem__(
            "question_source", "immutable_tape"
        ),
        lambda row: row["question_contract"]["question_rng"].__setitem__(
            "zero_draw_claim_permitted", True
        ),
        lambda row: row["completion"].__setitem__("requested", 31),
        lambda row: row["episodes"][0].__setitem__(
            "termination_reasons",
            ["action_ball_single_stroke_complete", "robot_hit_table"],
        ),
        lambda row: row["safety"].__setitem__("projection_nonfinite_count", 1),
        lambda row: row["selected_rubber_contact_artifact"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda row: row["episodes"][0]["predicted_outcome"].__setitem__(
            "predicted_landing_xy_m", [9.0, 9.0]
        ),
    ),
)
def test_resealed_raw_oracle_semantic_drift_is_rejected(
    tmp_path, monkeypatch, mutation
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    _rewrite_oracle_raw(spec_path, spec, mutation)
    with pytest.raises(launcher.LaunchRefused):
        launcher.build_plan(spec_path)


# ---------------------------------------------------------------------------
# oracle32 验收门重定范围 —— 2026-08-06
#
# 人话
# ====
# oracle32 跑的是**刚初始化、一步没训过**的 policy 的 32 集 rollout。旧门却要求它
# "32/32 都打完一整拍、一次没摔、一次没碰桌",也就是要求一个未开训的策略已经会
# 打乒乓球 —— 仓库自己在两处写着不该这么要求:
#
#   * §12.4:"bridge 的桌/跌倒/too-low 事件按 phase 作行为证据,
#             qdes-hard/actual-hard/nonfinite 才是实现 strict-zero"
#   * §8.3: "……不以『必须零次』循环要求未开训 policy 已经学会平衡"
#   * §5.6.4:参考实现 build_1 自己第 0 迭代 mean_ep_len 只有 23 tick、
#             iter 35..60 时 base_fell_tilt=1.00 —— 它自己也过不了旧门。
#
# 下面这组用例是这次重定范围的**变异测试**,三类各自独立:
#   A. 等强:两类实现故障(qdes-hard / actual-hard / nonfinite)非零时**仍然拒绝**;
#   B. 重定范围:未训练策略摔倒 / 碰桌**不再误拒**,而是被数出来写进收据;
#   C. 普查守恒:阶段x原因的计数必须和 32 集逐集重数完全对上,收据不能自报好看的总数。
# ---------------------------------------------------------------------------


def _untrained_policy_terminations() -> "list[tuple[str, list[str]]]":
    """A rollout that looks like a fresh policy: mostly falls, a few strokes.

    20 falls before the strike, 5 falls that also hit the table on the way down,
    3 table hits alone, 4 completed strokes.  Nothing here is an implementation
    failure; every one of these is behaviour the policy has not learned yet.
    """

    rows: "list[tuple[str, list[str]]]" = []
    rows += [("pre_strike_or_same_step_unknown", ["base_fell_tilt"])] * 20
    rows += [
        ("pre_strike_or_same_step_unknown", ["base_too_low", "robot_hit_table"])
    ] * 5
    rows += [("pre_strike_or_same_step_unknown", ["robot_hit_table"])] * 3
    rows += [("post_strike", ["action_ball_single_stroke_complete"])] * 4
    assert len(rows) == 32
    return rows


def test_untrained_policy_falls_and_table_hits_no_longer_refuse_oracle32(
    tmp_path, monkeypatch
):
    """B. 重定范围:未训练策略摔倒/碰桌不再误拒,而且必须在收据上数出来。"""

    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(
        tmp_path,
        stage="scale4096",
        terminations=_untrained_policy_terminations(),
        wait_only_reset_excluded=11,
    )
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    receipt = payload["materialization_inputs"]["oracle32_receipt"]
    assert receipt["verdict"] == "PASS"
    census = receipt["termination_census"]
    # 收据自陈 telemetry:一眼能看出这一跑各阶段各死法分别多少集。
    assert census["closed_attempt_episodes"] == 32
    assert census["episodes_by_terminal_phase"] == {
        "post_strike": 4,
        "pre_strike_or_same_step_unknown": 28,
    }
    assert census["physical_fall_by_reason"] == {
        "base_fell_tilt": 20,
        "base_too_low": 5,
    }
    assert census["robot_hit_table_count"] == 8
    assert census["single_stroke_complete_count"] == 4
    assert census["implementation_strict_zero"] == {
        "joint_actual_forbidden": 0,
        "joint_qdes_forbidden": 0,
        "projection_nonfinite_count": 0,
    }
    # 分母可见:32 次已关闭尝试之外,这一跑还烧掉了 11 次 WAIT 期猝死的复位。
    assert census["wait_only_reset_excluded"] == 11
    assert census["source_episodes_consumed"] == 43
    assert (
        census["closed_attempt_episodes"] + census["wait_only_reset_excluded"]
        == census["source_episodes_consumed"]
    )
    # 各阶段各原因加起来等于总集数(多原因集按集计一次的口径见 by-phase 表)。
    assert sum(census["episodes_by_terminal_phase"].values()) == 32
    for reason, total in census["terminal_reason_totals"].items():
        assert (
            sum(
                table.get(reason, 0)
                for table in census["terminal_reason_by_phase"].values()
            )
            == total
        )


def test_the_old_all_32_single_stroke_shape_still_passes(tmp_path, monkeypatch):
    """重定范围不是"换一个门":原来那份干净证据必须照样通过。"""

    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    payload = launcher.build_plan(spec_path)["canonical_payload"]
    census = payload["materialization_inputs"]["oracle32_receipt"][
        "termination_census"
    ]
    assert census["single_stroke_complete_count"] == 32
    assert census["physical_fall_by_reason"] == {
        "base_fell_tilt": 0,
        "base_too_low": 0,
    }
    assert census["robot_hit_table_count"] == 0


@pytest.mark.parametrize(
    "reason", ("joint_qdes_forbidden", "joint_actual_forbidden")
)
def test_implementation_hard_termination_still_refuses_oracle32(
    tmp_path, monkeypatch, reason
):
    """A. 等强:qdes-hard / actual-hard 只要出现一集就仍然拒绝。

    这里刻意让**整条证据链内部自洽**(safety 账、termination 账、收据 census 全都
    如实记了这一集),所以唯一能拒绝它的只可能是 strict-zero 规则本身,而不是某个
    SHA 对不上。
    """

    _patch_plan_environment(monkeypatch)
    rows = _untrained_policy_terminations()
    rows[0] = ("pre_strike_or_same_step_unknown", [reason])
    spec_path, _spec, _lineage_doc = _case(
        tmp_path, stage="scale4096", terminations=rows
    )
    with pytest.raises(
        launcher.LaunchRefused, match="implementation failure"
    ):
        launcher.build_plan(spec_path)


def test_projection_nonfinite_still_refuses_oracle32(tmp_path, monkeypatch):
    """A. 等强:投影里出现一次 NaN/Inf 就仍然拒绝。"""

    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(
        tmp_path,
        stage="scale4096",
        terminations=_untrained_policy_terminations(),
        projection_nonfinite=1,
    )
    with pytest.raises(
        launcher.LaunchRefused, match="implementation failure"
    ):
        launcher.build_plan(spec_path)


@pytest.mark.parametrize(
    "mutation,expected",
    (
        # 阶段x原因表少数一集 —— 守恒破了
        (
            lambda raw: raw["termination"]["phase_by_reason"][
                "pre_strike_or_same_step_unknown"
            ].__setitem__("base_fell_tilt", 19),
            "termination ledger differs from its own 32 episodes",
        ),
        # 总账多报一集
        (
            lambda raw: raw["termination"]["by_reason"].__setitem__(
                "base_fell_tilt", 21
            ),
            "termination ledger differs from its own 32 episodes",
        ),
        # 名单外账被抹平成空(旧门恰恰是靠"必须为空"放行的)
        (
            lambda raw: raw["termination"].__setitem__("unexpected_by_reason", {}),
            "termination ledger differs from its own 32 episodes",
        ),
        # completion 自报 32 次完整挥拍,和 32 集对不上
        (
            lambda raw: raw["completion"].__setitem__("single_stroke", 32),
            "single-stroke completion count differs",
        ),
        # safety 那条独立通道少报了摔倒
        (
            lambda raw: raw["safety"]["hard_termination_by_reason"].__setitem__(
                "base_fell_tilt", 0
            ),
            "safety ledger differs from its own termination census",
        ),
        # safety 少报撞桌
        (
            lambda raw: raw["safety"].__setitem__("robot_table_contact_count", 0),
            "safety ledger differs from its own termination census",
        ),
        # 出现一个词表外的终止名 —— 重定范围只放行已知的行为证据
        (
            lambda raw: raw["episodes"][0].__setitem__(
                "termination_reasons", ["anchor_pos"]
            ),
            "known terminal-reason set",
        ),
        # 同一集重复列同一个原因
        (
            lambda raw: raw["episodes"][0].__setitem__(
                "termination_reasons", ["base_fell_tilt", "base_fell_tilt"]
            ),
            "known terminal-reason set",
        ),
        # 一集没有任何终止原因
        (
            lambda raw: raw["episodes"][0].__setitem__("termination_reasons", []),
            "known terminal-reason set",
        ),
        # WAIT 期被丢掉的复位数被抹掉 —— 分母不能悄悄消失
        (
            lambda raw: raw["rollout_census"].__setitem__(
                "wait_only_reset_excluded", 0
            ),
            "rollout census does not close over its own resets",
        ),
        (
            lambda raw: raw["rollout_census"].__setitem__("closed_attempts", 31),
            "rollout census does not close over its own resets",
        ),
    ),
)
def test_oracle32_termination_census_must_conserve(
    tmp_path, monkeypatch, mutation, expected
):
    """C. 普查守恒:任何一处聚合数字和 32 集逐集重数对不上就拒绝。"""

    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(
        tmp_path,
        stage="scale4096",
        terminations=_untrained_policy_terminations(),
        wait_only_reset_excluded=11,
    )
    _rewrite_oracle_raw(spec_path, spec, mutation)
    with pytest.raises(launcher.LaunchRefused, match=expected):
        launcher.build_plan(spec_path)


def test_oracle32_strict_zero_scope_is_exactly_the_two_implementation_reasons():
    """门盯的对象本身也要被钉住,免得哪天有人把摔倒悄悄加回 strict-zero。"""

    assert launcher.STRICT_HARD_TERMINATION_UNION == (
        "joint_actual_forbidden",
        "joint_qdes_forbidden",
    )
    assert launcher.ORACLE_BEHAVIOR_TERMINATION_REASONS == (
        "base_fell_tilt",
        "base_too_low",
        "robot_hit_table",
    )
    assert set(launcher.ORACLE_ALLOWED_TERMINATION_REASONS) == {
        "action_ball_single_stroke_complete",
        *launcher.HARD_TERMINATION_UNION,
    }
    # 行为证据 + 实现故障 = 硬终止并集,不重不漏。
    assert set(launcher.ORACLE_BEHAVIOR_TERMINATION_REASONS) | set(
        launcher.STRICT_HARD_TERMINATION_UNION
    ) == set(launcher.HARD_TERMINATION_UNION)
    assert not set(launcher.ORACLE_BEHAVIOR_TERMINATION_REASONS) & set(
        launcher.STRICT_HARD_TERMINATION_UNION
    )
    # 这两项在同一个发射器里也被 scale4096 验收当作 strict zero 消费,同一套词表。
    assert launcher.PHYSICAL_FALL_REASONS == ("base_fell_tilt", "base_too_low")


def test_oracle_and_ppo_stages_are_no_longer_globally_blocked():
    assert launcher.BLOCKED_RUNTIME_STAGES == ()
    assert launcher.ORACLE_RUNTIME_DEPENDENCIES == ()


def test_launcher_pins_the_runner_before_c211_000_oracle_bundle_hook():
    markers = launcher.C211_ORACLE_HOOK_SOURCE_MARKERS[launcher.TRAIN_SOURCE]
    assert b"action_ball_c211_oracle_bundle_output_path" in markers
    assert b"ACTION_BALL_C211_OBSERVED_ORACLE_BUNDLE_JSON" in markers


def _producer_bundle_from_oracle_fixture(spec: dict) -> dict:
    result = json.loads(Path(spec["oracle32_result"]["path"]).read_text())
    receipt = result["oracle32_receipt"]
    raw = json.loads(Path(receipt["raw_oracle_artifact"]["path"]).read_text())
    preflight = json.loads(Path(raw["runner_preflight_artifact"]["path"]).read_text())
    episodes = []
    for item in raw["episodes"]:
        steps = item["control_steps"]
        episodes.append(
            {
                "episode": item["episode"],
                "control_steps": steps,
                "terminal_phase": item["terminal_phase"],
                "termination_reasons": item["termination_reasons"],
                "sampler_sample_index": item["sampler_sample_index"],
                "sampler_sample_sha256": item["sampler_sample_sha256"],
                "sampler_draw_start": item["sampler_draw_start"],
                "sampler_draw_end": item["sampler_draw_end"],
                "incoming_ball_observation": item["incoming_ball_observation"],
                "observed_selected_rubber_contact": {
                    "episode": item["episode"],
                    "runtime_control_step": steps,
                    "task_valid": True,
                    "eligible_closed_swing": True,
                    "exact_strike": True,
                    "selected_face_sweep_contact": True,
                    "selected_face_bracketed": True,
                    "selected_face_edge_safe": True,
                    "selected_face_geometry_finite": True,
                    "selected_face_closing_speed_positive": True,
                    "selected_face_normal_speed_consistent": True,
                    "wrong_surface_contact": False,
                    "edge_or_rim_ambiguous": False,
                    "between_planes_ambiguous": False,
                },
                "achieved_analytic_flight": item["achieved_analytic_flight"],
                "predicted_outcome": item["predicted_outcome"],
                "safety": {
                    "hard_termination_by_reason": {
                        name: int(name in item["termination_reasons"])
                        for name in launcher.HARD_TERMINATION_UNION
                    },
                    "robot_table_contact_count": int(
                        "robot_hit_table" in item["termination_reasons"]
                    ),
                    "projection_nonfinite_count": 0,
                    "projection_observed_sample_count": steps,
                    "qdes_observed_sample_count": steps,
                    "actual_observed_sample_count": steps,
                    "reference_guard_sample_count": steps,
                },
                "teacher_qdes": {
                    "preclamp_max_abs_error_rad": 0.0,
                    "teleport_used": False,
                },
            }
        )
    return {
        "schema_version": 3,
        "kind": producer.INPUT_KIND,
        "diagnostic_unauthorized": True,
        "identity": {
            "action_id": launcher.ACTION_ID,
            "action_uid": launcher.ACTION_UID,
            "motion_sha256": raw["bindings"]["motion_sha256"],
        },
        "bindings": raw["bindings"],
        "training_contract_path": raw["training_contract_artifact"]["path"],
        "runner_preflight_facts": preflight["facts"],
        "question_contract": raw["question_contract"],
        "rollout_census": copy.deepcopy(raw["rollout_census"]),
        "episodes": episodes,
    }


def test_observed_producer_sidecars_are_exactly_consumable(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    bundle = _producer_bundle_from_oracle_fixture(spec)
    namespace = tmp_path / "observed-oracle32"
    namespace.mkdir()
    published = producer.publish_bundle(bundle, namespace=namespace)

    raw_path = Path(published["raw_oracle_artifact"]["path"])
    raw = json.loads(raw_path.read_text())
    hard = json.loads(Path(bundle["training_contract_path"]).read_text())
    assert hard["actor_obs_term_names"][:2] == [
        "actual_base_pose_lin_vel_world",
        "base_ang_vel_body",
    ]
    assert hard["actor_obs_term_names"][10:13] == list(
        launcher.INCOMING_BALL_FIELDS
    )
    selected_path = Path(published["selected_rubber_contact_artifact"]["path"])
    selected = json.loads(selected_path.read_text())
    assert raw["kind"] == launcher.C211_RAW_ORACLE_KIND
    assert raw["desired_contact_metrics"] == {
        "status": "INELIGIBLE",
        "reason": "target_validity_000_contact_target_absent",
    }
    assert {row["classification"] for row in selected["episodes"]} == {
        "selected_rubber"
    }
    observed = bundle["episodes"][0]["observed_selected_rubber_contact"]
    assert selected["episodes"][0]["contact_evidence_sha256"] == (
        producer.canonical_sha256(observed)
    )
    assert selected["classifier_source_sha256"] == producer.sha256_file(
        PRODUCER_SCRIPT
    )
    assert selected["geometry_authority_sha256"] == (
        producer.geometry_authority_sha256()
    )

    result_path = Path(spec["oracle32_result"]["path"])
    result = json.loads(result_path.read_text())
    receipt = result["oracle32_receipt"]
    receipt["raw_oracle_artifact"] = published["raw_oracle_artifact"]
    observed_path = namespace / launcher.C211_OBSERVED_BUNDLE_FILENAME
    _write(observed_path, bundle)
    receipt["observed_oracle_bundle_artifact"] = {
        "path": str(observed_path),
        "sha256": hashlib.sha256(observed_path.read_bytes()).hexdigest(),
    }
    receipt["observed_oracle_bundle_content_sha256"] = launcher.canonical_sha256(
        bundle
    )
    receipt["raw_oracle_kind"] = raw["kind"]
    receipt["raw_oracle_content_sha256"] = raw["content_sha256"]
    receipt["control_step_denominator"] = raw["completion"]["control_steps"]
    receipt["selected_rubber_episode_denominator"] = 32
    receipt["actual_selected_rubber_contact_count"] = 32
    receipt["termination_census"] = _expected_termination_census(
        [(row["terminal_phase"], row["termination_reasons"]) for row in raw["episodes"]]
    )
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("content_sha256")
    receipt["content_sha256"] = launcher.canonical_sha256(unsigned_receipt)
    result["namespace"] = str(namespace)
    unsigned_result = dict(result)
    unsigned_result.pop("content_sha256")
    result["content_sha256"] = launcher.canonical_sha256(unsigned_result)
    spec["oracle32_result"]["sha256"] = _write(result_path, result)
    _write(spec_path, spec)

    payload = launcher.build_plan(spec_path)["canonical_payload"]
    assert payload["materialization_inputs"]["oracle32_receipt"]["verdict"] == "PASS"
    assert payload["output_contract"]["runtime_gate"] == "READY"
    with pytest.raises(producer.EvidenceError, match="no-clobber"):
        producer.publish_bundle(bundle, namespace=namespace)


def test_oracle_runner_facts_match_the_real_18_field_producer_contract():
    facts = _real_runner_preflight_facts()
    assert set(facts) == {
        "trainability_contract",
        "actor_contract",
        "critic_contract",
        "actor_width",
        "critic_width",
        "actor_normalizer_identity",
        "critic_normalizer_identity",
        "fresh_normalizers_required",
        "symmetric_critic_fallback_forbidden",
        "task_valid_required",
        "task_wait_contract",
        "question_source_contract",
        "contact_target_absent",
        "c225_reward_contract",
        "runner_actor_width",
        "runner_critic_width",
        "actor_normalizer_attribute",
        "critic_normalizer_attribute",
    }
    assert producer._validate_runner_facts(facts) == facts
    assert facts["task_valid_required"] is True
    assert facts["task_wait_contract"] == launcher._hard_wait_contract()
    assert facts["question_source_contract"] == (
        launcher._hard_question_source_contract()
    )


def test_runtime_oracle32_publishes_and_accepts_exactly_32_observed_rows(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    plan = launcher.build_plan(spec_path)["canonical_payload"]
    bundle = _producer_bundle_from_oracle_fixture(spec)
    namespace = tmp_path / "runtime-oracle32"
    namespace.mkdir()
    bundle_path = namespace / launcher.C211_OBSERVED_BUNDLE_FILENAME
    _write(bundle_path, bundle)
    receipt = launcher._runtime_oracle32_receipt(
        bundle_path=bundle_path,
        namespace=namespace,
        checkout=Path(spec["source"]["checkout"]),
        launch_claim_sha256="1" * 64,
        recipe=plan["bundle"]["recipe"],
        lineage=plan["bundle"]["lineage"],
        materialization=plan["materialization_inputs"]["reward_materialization"],
        policy=plan["materialization_inputs"]["policy_recipe_materialization"],
    )
    assert receipt["verdict"] == "PASS"
    assert receipt["episodes"] == 32
    assert receipt["selected_rubber_episode_denominator"] == 32
    assert receipt["actual_selected_rubber_contact_count"] == 32


def test_action_ball_hit_latch_closes_zero_hit_wait_and_early_terminal_honestly():
    source = (
        SCRIPT.parent.parent
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    ).read_text(encoding="utf-8")
    close = source.split("def _action_ball_close_attempts", 1)[1].split(
        "def _action_ball_retire_previous_births", 1
    )[0]
    strike = source.split("def _vb_book_strike_step", 1)[1].split(
        "def _book_sparse_reward_eligibility", 1
    )[0]
    assert '"C": torch.bincount(clamped[active]' in close
    assert '"H": torch.bincount(clamped[hit]' in close
    assert "hit = active & self._action_ball_attempt_hit[ids]" in close
    assert "active = active & self._action_ball_task_valid[ids]" in close
    assert 'additions["H"] <= additions["C"]' in close
    assert "hit = gate & self._action_ball_attempt_active" in strike
    assert "hit = hit & self._action_ball_task_valid" in strike
    assert "self._action_ball_attempt_hit |= hit" in strike


@pytest.mark.parametrize(
    "field,retired",
    (
        (
            "trainability_contract",
            "action_ball_c211_fixed_midpoint_learnability_v1",
        ),
        ("actor_normalizer_identity", "action_ball_c211_actor_norm_v1"),
        ("task_valid_required", False),
        ("task_wait_contract", {}),
        ("question_source_contract", {}),
    ),
)
def test_observed_producer_rejects_malformed_c211_runner_abi(
    tmp_path, monkeypatch, field, retired
):
    _patch_plan_environment(monkeypatch)
    _spec_path, spec, _lineage_doc = _case(tmp_path, stage="scale4096")
    bundle = _producer_bundle_from_oracle_fixture(spec)
    bundle["runner_preflight_facts"][field] = retired
    namespace = tmp_path / ("retired-" + field)
    namespace.mkdir()
    with pytest.raises(producer.EvidenceError, match="C211 ABI differs"):
        producer.publish_bundle(bundle, namespace=namespace)


def test_observed_contact_classifier_is_fail_closed():
    selected = {
        "episode": 0,
        "runtime_control_step": 1,
        "task_valid": True,
        "eligible_closed_swing": True,
        "exact_strike": True,
        "selected_face_sweep_contact": True,
        "selected_face_bracketed": True,
        "selected_face_edge_safe": True,
        "selected_face_geometry_finite": True,
        "selected_face_closing_speed_positive": True,
        "selected_face_normal_speed_consistent": True,
        "wrong_surface_contact": False,
        "edge_or_rim_ambiguous": False,
        "between_planes_ambiguous": False,
    }
    assert producer.classify_observed_contact(selected, episode=0)[0] == (
        "selected_rubber"
    )
    invalid = copy.deepcopy(selected)
    invalid["task_valid"] = False
    assert producer.classify_observed_contact(invalid, episode=0)[0] == "no_contact"
    overlap = copy.deepcopy(selected)
    overlap["wrong_surface_contact"] = True
    with pytest.raises(producer.EvidenceError, match="overlap"):
        producer.classify_observed_contact(overlap, episode=0)
    wrong = copy.deepcopy(selected)
    for key in (
        "selected_face_sweep_contact",
        "selected_face_bracketed",
        "selected_face_edge_safe",
        "selected_face_closing_speed_positive",
        "selected_face_normal_speed_consistent",
    ):
        wrong[key] = False
    wrong["wrong_surface_contact"] = True
    assert producer.classify_observed_contact(wrong, episode=0)[0] == "wrong_surface"
    unknown = copy.deepcopy(wrong)
    unknown["wrong_surface_contact"] = False
    unknown["selected_face_geometry_finite"] = False
    assert producer.classify_observed_contact(unknown, episode=0)[0] == "unknown"
    verdict_injection = {**selected, "classification": "selected_rubber"}
    with pytest.raises(producer.EvidenceError, match="keys differ"):
        producer.classify_observed_contact(verdict_injection, episode=0)


def test_live_oracle_authority_missing_still_fails_closed(tmp_path):
    with pytest.raises(launcher.LaunchRefused, match="live-oracle runtime authority is absent"):
        launcher._verify_c211_runtime_authorities(tmp_path)


def test_c211_prelong_exec_environment_is_scale_only_and_recipe_bound():
    reward_sha = "b" * 64
    expected = {
        launcher.REWARD_PPO_ECONOMY_ENABLE_ENV: "1",
        launcher.PRELONG_SEMANTICS_ENABLE_ENV: "1",
        launcher.PRELONG_REWARD_RECIPE_SHA_ENV: reward_sha,
    }
    materialized = {"runtime_effective_reward_sha256": reward_sha}
    assert launcher._prelong_semantics_exec_environment(
        "scale4096", materialized
    ) == expected
    # 关键回归:materialize / recipe / oracle32 三个阶段的 reward_materialization 里
    # **没有** runtime_effective_reward_sha256 这个键(reward 还没产出)。这个 helper
    # 必须在取值之前就按 stage 短路,否则每条流水线的第一站都会在 exec 前 KeyError。
    planned = {"schema_version": 1, "kind": launcher.MATERIALIZATION_KIND}
    assert "runtime_effective_reward_sha256" not in planned
    for stage in launcher.BUDGETS:
        if stage != "scale4096":
            assert launcher._prelong_semantics_exec_environment(
                stage, planned
            ) == {}
            assert launcher._prelong_semantics_exec_environment(stage, None) == {}
    for bad in ({}, None, False, {"runtime_effective_reward_sha256": False}):
        with pytest.raises(launcher.LaunchRefused):
            launcher._prelong_semantics_exec_environment("scale4096", bad)


def test_c211_update_profile_switch_is_exact_claim_bound_and_non_speed(
    tmp_path, monkeypatch
):
    assert launcher._update_profile_exec_environment({}) == {}
    for value in ("0", "1"):
        assert launcher._update_profile_exec_environment(
            {launcher.UPDATE_PROFILE_ENV: value}
        ) == {launcher.UPDATE_PROFILE_ENV: value}
    for invalid in ("", "true", "2"):
        with pytest.raises(launcher.LaunchRefused, match="exactly 0 or 1"):
            launcher._update_profile_exec_environment(
                {launcher.UPDATE_PROFILE_ENV: invalid}
            )

    _patch_plan_environment(monkeypatch)
    monkeypatch.setenv(launcher.UPDATE_PROFILE_ENV, "1")
    _spec_path, spec, _lineage = _case(tmp_path, stage="materialize")
    normalized = launcher._validate_spec(spec)
    profile = launcher._output_contract(normalized)["update_profile"]
    assert profile["forwarded_value"] == "1"
    assert profile["mode"] == "profile_on_attribution_only"
    assert profile["speed_evidence_eligible"] is False
    assert profile["gpu_kernel_attribution_claimed"] is False
    _path, colocated, _colocated_lineage = _case(
        tmp_path / "colocated-profile",
        stage="scale4096",
        allow_colocation=True,
    )
    with pytest.raises(launcher.LaunchRefused, match="exclusive GPU claim"):
        launcher._output_contract(launcher._validate_spec(colocated))

    monkeypatch.setenv(launcher.UPDATE_PROFILE_ENV, "0")
    off = launcher._output_contract(normalized)["update_profile"]
    assert off["forwarded_value"] == "0"
    assert off["mode"] == "explicit_profiler_off"
    assert off != profile


def test_materialize_claim_is_vendor_admission_revalidatable_and_no_clobber(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    validated = launcher._ADMISSION._validate_namespace_claim(
        namespace,
        plan["launch_claim_sha256"],
        checkout=Path(spec["source"]["checkout"]),
        commit=spec["source"]["commit_sha"],
        gpu_index=2,
        gpu_uuid="GPU-12345678",
        require_colocation_opt_in=False,
    )
    assert validated["spec"]["stage"] == "materialize"
    assert validated["training_argv"] == plan["canonical_payload"]["training_argv"]
    original_claim = (namespace / "launch_claim.json").read_bytes()
    with pytest.raises(launcher.LaunchRefused):
        launcher._B._claim_namespace(plan)
    assert (namespace / "launch_claim.json").read_bytes() == original_claim


def test_claim_revalidation_detects_question_or_fresh_state_mutation(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    namespace = launcher._B._claim_namespace(plan)
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["question_contract"]["online_solver_calls"] = 1
    with pytest.raises(launcher.LaunchRefused):
        launcher._revalidate_claim_payload(payload)
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["checkpoint_contract"]["input"] = str(namespace / "model.pt")
    with pytest.raises(launcher.LaunchRefused):
        launcher._revalidate_claim_payload(payload)
    payload = copy.deepcopy(plan["canonical_payload"])
    payload["bundle"]["isaac_four_grid_manifest"]["cells"].pop()
    with pytest.raises(launcher.LaunchRefused):
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
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    Path(plan["canonical_payload"]["spec"]["namespace"]).mkdir()
    source["Isaac A211/C211 four-grid authority"]["sha256"] = "b" * 64
    with pytest.raises(launcher.LaunchRefused, match="runtime source identity drifted"):
        launcher._revalidate_claim_payload(plan["canonical_payload"])


def test_default_empty_gpu_and_scale_long_colocation_are_sealed(tmp_path):
    spec_path, spec, _lineage_doc = _case(tmp_path / "empty", stage="materialize")
    normalized = launcher._validate_spec(spec)
    assert normalized["gpu"]["require_empty"] is True
    assert normalized[launcher.COLOCATION_SPEC_KEY] is False
    default_output = launcher._output_contract(normalized)
    assert default_output["speed_benchmark_eligible"] is False
    assert default_output["rate_evidence_eligible"] is False
    assert default_output["rate_evidence_isolation"] == (
        "excluded_no_matched_abba_speed_stage"
    )

    _path, colocated, _lineage = _case(
        tmp_path / "colocated", stage="long4096", allow_colocation=True
    )
    normalized = launcher._validate_spec(colocated)
    assert normalized["gpu"]["require_empty"] is False
    assert normalized[launcher.COLOCATION_SPEC_KEY] is True
    output = launcher._output_contract(normalized)
    assert output["speed_benchmark_eligible"] is False
    assert output["rate_evidence_eligible"] is False
    assert output["rate_evidence_isolation"] == "excluded_colocated_diagnostic"
    assert output["colocated_stage"] == "long4096"
    assert output["max_compute_processes_per_gpu"] == 2

    _path, colocated_scale, _lineage = _case(
        tmp_path / "colocated-scale", stage="scale4096", allow_colocation=True
    )
    scale_output = launcher._output_contract(
        launcher._validate_spec(colocated_scale)
    )
    assert scale_output["speed_benchmark_eligible"] is False
    assert scale_output["rate_evidence_eligible"] is False
    assert scale_output["colocated_stage"] == "scale4096"

    _path, exclusive_scale_spec, _lineage = _case(
        tmp_path / "exclusive-scale", stage="scale4096"
    )
    exclusive_scale = launcher._output_contract(
        launcher._validate_spec(exclusive_scale_spec)
    )
    assert exclusive_scale["speed_benchmark_eligible"] is False
    assert exclusive_scale["rate_evidence_eligible"] is False
    assert exclusive_scale["rate_evidence_isolation"] == (
        "excluded_scale_finite_gate"
    )
    assert exclusive_scale["deferred_matched_speed_measurement"][
        "abba_order"
    ] == ["current_A", "current_C", "current_C", "current_A"]
    assert exclusive_scale["deferred_matched_speed_measurement"][
        "implemented_by_this_launcher"
    ] is False
    assert exclusive_scale["deferred_matched_speed_measurement"][
        "workload_kind"
    ] == "direct_ball_sampler_consumer"
    assert exclusive_scale["deferred_matched_speed_measurement"][
        "online_solver_calls_required"
    ] == 0
    assert exclusive_scale["deferred_matched_speed_measurement"][
        "producer_evidence_must_be_separate"
    ] is True

    _path, forbidden, _lineage = _case(
        tmp_path / "forbidden", stage="oracle32", allow_colocation=True
    )
    with pytest.raises(launcher.LaunchRefused, match="scale4096/long4096"):
        launcher._validate_spec(forbidden)


def test_confirm_digest_mismatch_blocks_materialize_before_source_or_lock(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source touched")),
    )
    with pytest.raises(launcher.LaunchRefused, match="confirm-claim"):
        launcher.execute(plan, confirm_claim="0" * 64)
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


def test_mutated_plan_payload_blocks_before_source_lock_or_namespace(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    plan["canonical_payload"]["bundle"]["question_contract"][
        "online_solver_calls"
    ] = 1
    monkeypatch.setattr(
        launcher._B,
        "_verify_clean_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source touched")),
    )
    with pytest.raises(launcher.LaunchRefused, match="payload seal"):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


def test_mutated_outer_plan_envelope_blocks_before_gpu_lock_or_namespace(
    tmp_path, monkeypatch
):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    plan["unsealed_extra"] = True
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
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
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


def test_prelaunch_gpu_refusal_does_not_claim_namespace(tmp_path, monkeypatch):
    _patch_plan_environment(monkeypatch)
    spec_path, _spec, _lineage_doc = _case(tmp_path, stage="materialize")
    plan = launcher.build_plan(spec_path)
    monkeypatch.setattr(launcher, "_open_gpu_shared_lock", lambda path: 91)
    monkeypatch.setattr(launcher, "_lock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher, "_unlock_gpu_admission", lambda fd: None)
    monkeypatch.setattr(launcher.os, "close", lambda fd: None)
    monkeypatch.setattr(
        launcher,
        "_verify_gpu_admission",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            launcher.LaunchRefused("fixture GPU admission refused")
        ),
    )
    with pytest.raises(launcher.LaunchRefused, match="GPU admission refused"):
        launcher.execute(plan, confirm_claim=plan["launch_claim_sha256"])
    assert not Path(plan["canonical_payload"]["spec"]["namespace"]).exists()


@pytest.mark.parametrize(
    "text",
    (
        "completion_exit_code=0\nterminal_kind=clean_completion\nterminal_exit_code=1\n",
        "completion_exit_code=0\nterminal_kind=stale_timeout\nterminal_exit_code=0\n",
        "completion_exit_code=0\nterminal_kind=clean_completion\n",
        "completion_exit_code=0\ncompletion_exit_code=0\n"
        "terminal_kind=clean_completion\nterminal_exit_code=0\n",
    ),
)
def test_completion_state_rejects_nonexact_or_duplicate_rows(tmp_path, text):
    path = tmp_path / "completion.state"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(launcher.LaunchRefused):
        launcher._validate_completion_state(path)


def _c211_policy_recipe_bytes(path, *, noise_std_type, init_noise_std, recipe):
    """Write a policy-recipe artifact carrying one exact exploration package."""

    document = {
        "action_ball_ppo_runner_recipe": {
            "recipe": {
                "algorithm": {
                    "entropy_coef": recipe["entropy_coef"],
                    **recipe["ppo"],
                },
                "policy": {
                    "actor_hidden_dims": recipe["actor_hidden_dims"],
                    "critic_hidden_dims": recipe["critic_hidden_dims"],
                    "init_noise_std": init_noise_std,
                    "noise_std_type": noise_std_type,
                },
                "runner": {
                    "empirical_normalization": True,
                    "init_at_random_ep_len": False,
                },
            }
        }
    }
    path.write_bytes(launcher._B._canonical_bytes(document) + b"\n")
    return path


def _c211_policy_lineage():
    return {
        "lineage_sha256": "a" * 64,
        "dynamic_ready_artifact": {"path": "artifact.json", "sha256": "b" * 64},
        "dynamic_ready_nominal_receipt": {
            "path": "nominal.json",
            "sha256": "c" * 64,
        },
        "motion": {"path": "motion.npz", "sha256": "d" * 64},
    }


def test_c211_policy_gate_binds_the_registered_exploration_package(
    tmp_path, monkeypatch
):
    """C forwards its own registered sigma, and still refuses anything else.

    人话:借来的那台 A225 验证器默认要 log/0.02。四格 2026-08-05 把探索包定死成
    标准初始化 + sigma 1.0 + scalar 之后,那个默认值会把 C 的 recipe 阶段夹在两条
    互斥的门中间(先被要求 log/0.02,二十几行后又被要求等于 recipe 的 scalar/1.0),
    任何一份 recipe 都过不去。修法是把该期待哪个包交给注册的四格 cell。

    这不是放宽门,下面三段就是证据:
      1. 传出去的值必须逐字节等于 ``_recipe_contract`` 里的注册包,而那个包已被
         ``recipe_contract_sha256`` 封住 —— 不是这里现编的字面量;
      2. 借来的默认值和 C 的注册包不相等,所以一旦有人把这两个参数删掉,真验证器
         会当场拒 —— 转发是承重的,不是装饰;
      3. 产出的 recipe 只要偏离注册包一个字段,C 自己那道 ``expected_policy``
         仍然当场拒。
    """

    borrowed_defaults = (
        launcher._OLD._validate_policy_materialization.__kwdefaults__
    )
    seen = []

    def _record(value, *, checkout, bundle, **expectations):
        assert set(expectations) == {
            "expected_noise_std_type",
            "expected_init_noise_std",
        }, (
            "C211 must name its own exploration package; falling back to the "
            "borrowed A225 log/0.02 default cannot pass the four-grid contract"
        )
        seen.append(dict(expectations))
        return {
            "artifact": dict(value),
            "policy_contract_sha256": "e" * 64,
            "dynamic_ready_binding_sha256": "f" * 64,
            "noise_std_type": expectations["expected_noise_std_type"],
            "configured_and_realized_init_noise_std": expectations[
                "expected_init_noise_std"
            ],
        }

    monkeypatch.setattr(launcher._OLD, "_validate_policy_materialization", _record)
    lineage = _c211_policy_lineage()

    for recipe_id in launcher.RECIPE_IDS:
        recipe = launcher._recipe_contract(recipe_id)
        # 2. 借来的默认值不可能满足 C 的注册包。
        assert (
            borrowed_defaults["expected_noise_std_type"]
            != recipe["noise_std_type"]
        )
        assert (
            borrowed_defaults["expected_init_noise_std"]
            != recipe["init_noise_std"]
        )

        path = _c211_policy_recipe_bytes(
            tmp_path / (recipe_id + ".json"),
            noise_std_type=recipe["noise_std_type"],
            init_noise_std=recipe["init_noise_std"],
            recipe=recipe,
        )
        materialization = launcher._runtime_policy_materialization(
            path=path, checkout=tmp_path, lineage=lineage, recipe=recipe
        )
        # 1. 转发出去的就是注册包本身。
        assert seen[-1] == {
            "expected_noise_std_type": recipe["noise_std_type"],
            "expected_init_noise_std": recipe["init_noise_std"],
        }
        assert materialization["noise_std_type"] == recipe["noise_std_type"]
        assert (
            materialization["configured_and_realized_init_noise_std"]
            == recipe["init_noise_std"]
        )

    # 3. 偏离注册包一个字段就拒 —— 两个方向各来一次。
    recipe = launcher._recipe_contract(launcher.RECIPE_IDS[0])
    for wrong_type, wrong_sigma in (
        (borrowed_defaults["expected_noise_std_type"], recipe["init_noise_std"]),
        (recipe["noise_std_type"], borrowed_defaults["expected_init_noise_std"]),
    ):
        drifted = _c211_policy_recipe_bytes(
            tmp_path / ("drift_%s_%s.json" % (wrong_type, wrong_sigma)),
            noise_std_type=wrong_type,
            init_noise_std=wrong_sigma,
            recipe=recipe,
        )
        with pytest.raises(
            launcher.LaunchRefused, match="C211 runtime policy recipe differs"
        ):
            launcher._runtime_policy_materialization(
                path=drifted, checkout=tmp_path, lineage=lineage, recipe=recipe
            )


def test_launcher_never_sets_or_repurposes_home():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"HOME"' not in source
    assert '"CODEX_HOME"' not in source


def test_launcher_trainability_literal_matches_the_contract_publisher():
    """Consumer and producer must spell the C211 marker identically."""
    assert (
        launcher.TRAINABILITY_CONTRACT
        == _TRAINING_CONTRACT._ACTION_BALL_C211_TRAINABILITY_CONTRACT
    )


def test_hard_wait_contract_is_the_shape_training_contract_actually_emits():
    """C211's hard wait block must stay pinned to the producer's authority."""
    assert (
        launcher._hard_wait_contract()
        == _TRAINING_CONTRACT._action_ball_211_wait_contract_facts()
    )
