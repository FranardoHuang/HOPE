"""Host-only tests for the fail-closed offline Reward run audit."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_reward_run.py"
SPEC = importlib.util.spec_from_file_location("audit_reward_run_under_test", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


class FakeTensor:
    def __init__(self, values, dtype):
        self._array = np.asarray(values, dtype=dtype)
        self.dtype = "torch.{}".format(self._array.dtype.name)
        self.shape = self._array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return np.ascontiguousarray(self._array)

    def tolist(self):
        return self._array.tolist()

    def numel(self):
        return self._array.size

    def element_size(self):
        return self._array.itemsize


def _canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _recipe():
    terms = [
        {
            "name": "death_penalty",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
                "action_ball_safety_terminated"
            ),
            # 2026-08-05 层级对齐(exp §5.6 第 7 条):-300.0 -> -10.0。
            "weight": -10.0,
            "params": {
                "term_names": [
                    "base_fell_tilt",
                    "base_too_low",
                    "joint_actual_forbidden",
                    "joint_qdes_forbidden",
                    "robot_hit_table",
                ]
            },
        },
        {
            "name": "joint_limit",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
                "actual_joint_limit_barrier_v2"
            ),
            "weight": -5.0,
            "params": {
                "asset_cfg": {"name": "robot"},
                "margin_frac": 0.08,
                "penalty_floor": 0.25,
                "expected_joint_count": 31,
            },
        },
        {
            "name": "qdes_limit_barrier",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
                "qdes_limit_barrier_v2"
            ),
            "weight": -5.0,
            "params": {
                "action_name": "joint_pos",
                # 2026-08-05 带宽对齐(exp §5.6 第 9 条):qdes 通道 0.08 -> 0.05。
                "margin_frac": 0.05,
                "penalty_floor": 0.25,
            },
        },
        {
            "name": "qdes_limit_barrier_probe",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
                "qdes_limit_barrier_probe"
            ),
            "weight": 1.0,
            # train.py 把 term 的 params 整份拷给 probe,所以 probe 也跟到 0.05。
            "params": {"action_name": "joint_pos", "margin_frac": 0.05},
        },
        {
            "name": "racket_position",
            "callable": "whole_body_tracking.tasks.tracking.mdp.rewards.racket_position",
            "weight": 4.0,
            "params": {"std": 0.075},
        },
    ]
    payload = {"schema_version": 1, "terms": terms}
    return {**payload, "sha256": _sha(payload)}


def _manifest():
    return {
        "schema_version": 3,
        "manifest_id": "fixture_n2",
        "action_order": ["a", "b"],
        "actions": [
            {"action_id": "a", "action_uid": 101},
            {"action_id": "b", "action_uid": 202},
        ],
    }


def _activation():
    terms = [
        {
            "name": "death_penalty",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
                "action_ball_safety_terminated"
            ),
            "role": "objective",
            # 2026-08-05 层级对齐(exp §5.6 第 7 条):-300.0 -> -10.0,
            # weighted = raw 3.0 * -10.0 * dt 0.02 = -0.6(原 -18.0)。
            "weight": -10.0,
            "recipe_term_sha256": _sha(_recipe()["terms"][0]),
            "observed_environment_step_count": 3,
            "observed_sample_count": 6,
            "nonzero_sample_count": 3,
            "weighted_sum": -0.6,
            "raw_sum": 3.0,
            "raw_recovery": (
                "validated_weighted_eq_raw_times_weight_times_step_dt"
            ),
            "raw_recomposition_max_abs_error": 0.0,
            "eligibility": "unknown",
            "eligibility_reason": "term_specific_mask_unavailable",
        },
        {
            "name": "joint_limit",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
                "actual_joint_limit_barrier_v2"
            ),
            "role": "objective",
            "weight": -5.0,
            "recipe_term_sha256": _sha(_recipe()["terms"][1]),
            "observed_environment_step_count": 3,
            "observed_sample_count": 6,
            "nonzero_sample_count": 2,
            "weighted_sum": -0.3,
            "raw_sum": 3.0,
            "raw_recovery": (
                "validated_weighted_eq_raw_times_weight_times_step_dt"
            ),
            "raw_recomposition_max_abs_error": 0.0,
            "eligibility": "unknown",
            "eligibility_reason": "term_specific_mask_unavailable",
        },
        {
            "name": "qdes_limit_barrier",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
                "qdes_limit_barrier_v2"
            ),
            "role": "objective",
            "weight": -5.0,
            "recipe_term_sha256": _sha(_recipe()["terms"][2]),
            "observed_environment_step_count": 3,
            "observed_sample_count": 6,
            "nonzero_sample_count": 2,
            "weighted_sum": -0.3,
            "raw_sum": 3.0,
            "raw_recovery": (
                "validated_weighted_eq_raw_times_weight_times_step_dt"
            ),
            "raw_recomposition_max_abs_error": 0.0,
            "eligibility": "unknown",
            "eligibility_reason": "term_specific_mask_unavailable",
        },
        {
            "name": "qdes_limit_barrier_probe",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
                "qdes_limit_barrier_probe"
            ),
            "role": "diagnostic_probe",
            "weight": 1.0,
            "recipe_term_sha256": _sha(_recipe()["terms"][3]),
            "observed_environment_step_count": 3,
            "observed_sample_count": 6,
            "nonzero_sample_count": 0,
            "weighted_sum": 0.0,
            "raw_sum": 0.0,
            "raw_recovery": (
                "validated_weighted_eq_raw_times_weight_times_step_dt"
            ),
            "raw_recomposition_max_abs_error": 0.0,
            "eligibility": "unknown",
            "eligibility_reason": "term_specific_mask_unavailable",
        },
        {
            "name": "racket_position",
            "callable": "whole_body_tracking.tasks.tracking.mdp.rewards.racket_position",
            "role": "objective",
            "weight": 4.0,
            "recipe_term_sha256": _sha(_recipe()["terms"][4]),
            "observed_environment_step_count": 3,
            "observed_sample_count": 6,
            "nonzero_sample_count": 2,
            "weighted_sum": 0.24,
            "raw_sum": 3.0,
            "raw_recovery": (
                "validated_weighted_eq_raw_times_weight_times_step_dt"
            ),
            "raw_recomposition_max_abs_error": 0.0,
            "eligibility": "unknown",
            "eligibility_reason": "term_specific_mask_unavailable",
        },
    ]
    return {
        "event": AUDIT.ACTIVATION_EVENT,
        "schema_version": 1,
        "recipe_sha256": _recipe()["sha256"],
        "task_kind": "action_ball",
        "ppo_update": 4,
        "environment_step_count": 3,
        "expected_environment_step_count": 3,
        "num_envs": 2,
        "observed_sample_count": 6,
        "step_dt_s": 0.02,
        "common_step_counter_start": 10,
        "common_step_counter_end": 13,
        "objective_term_names": [
            "death_penalty",
            "joint_limit",
            "qdes_limit_barrier",
            "racket_position",
        ],
        "diagnostic_probe_term_names": ["qdes_limit_barrier_probe"],
        "reward_cache_contract": {
            "source": "isaaclab_reward_manager_private_step_cache",
            "step_cache_semantics": "raw_times_weight",
            "weighted_semantics": "raw_times_weight_times_step_dt",
            "total_reward_closure": "validated",
            "max_abs_error": 0.0,
        },
        "total_weighted_reward_sum": sum(
            row["weighted_sum"] for row in terms
        ),
        "terms": terms,
    }


def _episode_closure():
    term_names = [row["name"] for row in _recipe()["terms"]]
    return {
        "event": AUDIT.EPISODE_SEGMENTED_CLOSURE_EVENT,
        "schema_version": 1,
        "status": "PASS",
        "evidence_source": "synthetic_test_fixture",
        "capture_mode": "reward_manager_reset_pre_clear_hook",
        "task_kind": "action_ball",
        "ppo_update": 4,
        "recipe_sha256": _recipe()["sha256"],
        "step_dt_s": 0.02,
        "max_episode_length_s": 10.0,
        "num_envs": 2,
        "segment_key_fields": ["env_id", "reset_generation"],
        "all_reward_manager_term_names": term_names,
        "completed_episode_count": 0,
        "completed_episode_segments": [],
        "reset_batches": [],
        "open_episode_count": 2,
        "open_episode_segments": [
            {
                "env_id": env_id,
                "reset_generation": 0,
                "action_uid": action_uid,
                "step_count": 3,
                "reward_buf_sum": 0.0,
                "all_term_sum": 0.0,
                "reward_buf_vs_all_terms_abs_error": 0.0,
                "local_term_sums": [0.0] * len(term_names),
                "status": "OPEN_NOT_E2",
            }
            for env_id, action_uid in enumerate((101, 202))
        ],
        "dashboard_normalization": {
            "status": "NOT_OBSERVED_NO_RESET",
            "reset_batch_count": 0,
            "reason": "no RewardManager.reset batch occurred in this PPO update",
        },
        "checks": {
            "environment_step_count": 3,
            "reset_batch_count": 0,
            "completed_episode_count": 0,
            "manager_episode_sum_comparison_count": 30,
            "dashboard_term_comparison_count": 0,
            "reward_buf_term_sum_comparison_count": 2,
            "manager_clear_comparison_count": 0,
            "max_abs_manager_episode_sum_error": 0.0,
            "max_abs_dashboard_normalization_error": 0.0,
            "max_abs_reward_buf_vs_term_sum_error": 0.0,
            "max_abs_manager_clear_error": 0.0,
            "status": "PASS",
            "all_step_reward_buf_equals_all_term_sums": "PASS",
            "all_episode_sums_equal_captured_term_sums": "PASS",
            "all_observed_dashboard_values_normalized_exactly": (
                "NOT_OBSERVED_NO_RESET"
            ),
            "all_reset_episode_sums_cleared": "PASS",
            "exact_environment_step_coverage": "PASS",
        },
        "e2_eligible": False,
        "e2_ineligible_reason": (
            "no non-administrative completed live episode segment in this update"
        ),
    }


def _term(name, observed, nonzero, raw_sum, weighted_sum):
    return {
        "name": name,
        "observed_sample_count": observed,
        "nonzero_sample_count": nonzero,
        "raw_sum": raw_sum,
        "weighted_sum": weighted_sum,
    }


def _reward_group_rows(recipe, samples_by_group):
    taxonomy = AUDIT.REWARD_TAXONOMY.build_action_ball_reward_group_taxonomy(
        recipe["terms"]
    )
    taxonomy_by_name = {
        row["name"]: row for row in taxonomy["active_terms"]
    }
    total_positive = sum(
        max(value, 0.0)
        for values in samples_by_group.values()
        for value in values
    )
    total_negative = sum(
        min(value, 0.0)
        for values in samples_by_group.values()
        for value in values
    )
    rows = []
    for group in taxonomy["group_order"]:
        objectives = sorted(
            name
            for name, spec in taxonomy_by_name.items()
            if spec["group"] == group and spec["role"] == "objective"
        )
        probes = sorted(
            name
            for name, spec in taxonomy_by_name.items()
            if spec["group"] == group
            and spec["role"] == "diagnostic_probe"
        )
        values = samples_by_group.get(group, [])
        positive = sum(max(value, 0.0) for value in values)
        negative = sum(min(value, 0.0) for value in values)
        rows.append(
            {
                "group": group,
                "objective_term_names": objectives,
                "diagnostic_probe_term_names": probes,
                "eligibility": "reward_manager_evaluated_active_group_terms",
                "eligible_sample_count": len(values),
                "nonzero_sample_count": sum(value != 0.0 for value in values),
                "weighted_sum": sum(values),
                "weighted_p5": AUDIT.REWARD_TAXONOMY._linear_quantile(
                    values, 0.05
                ),
                "weighted_p50": AUDIT.REWARD_TAXONOMY._linear_quantile(
                    values, 0.50
                ),
                "weighted_p95": AUDIT.REWARD_TAXONOMY._linear_quantile(
                    values, 0.95
                ),
                "positive_weighted_sum": positive,
                "negative_weighted_sum": negative,
                "positive_return_fraction": (
                    None
                    if total_positive == 0.0
                    else positive / total_positive
                ),
                "negative_return_fraction": (
                    None
                    if total_negative == 0.0
                    else negative / total_negative
                ),
            }
        )
    return taxonomy, total_positive, total_negative, rows


def _per_action(recipe, manifest_sha):
    group = AUDIT.REWARD_TAXONOMY
    a_taxonomy, a_positive, a_negative, a_groups = _reward_group_rows(
        recipe,
        {
            group.ACTION_BALL_REWARD_GROUP_HOPE_TASK: [0.16, 0.0, 0.0],
            # 2026-08-05(exp §5.6 第 7 条):death -12.0 -> -0.4,组内合计 -12.2 -> -0.6。
            group.ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY: [
                -0.3,
                -0.3,
                0.0,
            ],
        },
    )
    b_taxonomy, b_positive, b_negative, b_groups = _reward_group_rows(
        recipe,
        {
            group.ACTION_BALL_REWARD_GROUP_HOPE_TASK: [0.08, 0.0, 0.0],
            # 2026-08-05(exp §5.6 第 7 条):death -6.0 -> -0.2,组内合计 -6.4 -> -0.6。
            group.ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY: [
                -0.2,
                -0.2,
                -0.2,
            ],
        },
    )
    assert a_taxonomy == b_taxonomy
    return {
        "event": AUDIT.PER_ACTION_EVENT,
        "schema_version": 2,
        "recipe_sha256": recipe["sha256"],
        "task_kind": "action_ball",
        "ppo_update": 4,
        "step_dt_s": 0.02,
        "manifest_sha256": manifest_sha,
        "action_order": ["a", "b"],
        "reward_group_taxonomy": a_taxonomy,
        "actions": [
            {
                "action_id": "a",
                "action_uid": 101,
                "observed_sample_count": 3,
                "positive_weighted_sum": a_positive,
                "negative_weighted_sum": a_negative,
                "reward_groups": a_groups,
                "terms": [
                    _term("death_penalty", 3, 2, 2.0, -0.4),
                    _term("joint_limit", 3, 1, 1.0, -0.1),
                    _term("qdes_limit_barrier", 3, 1, 1.0, -0.1),
                    _term("qdes_limit_barrier_probe", 3, 0, 0.0, 0.0),
                    _term("racket_position", 3, 1, 2.0, 0.16),
                ],
            },
            {
                "action_id": "b",
                "action_uid": 202,
                "observed_sample_count": 3,
                "positive_weighted_sum": b_positive,
                "negative_weighted_sum": b_negative,
                "reward_groups": b_groups,
                "terms": [
                    _term("death_penalty", 3, 1, 1.0, -0.2),
                    _term("joint_limit", 3, 1, 2.0, -0.2),
                    _term("qdes_limit_barrier", 3, 1, 2.0, -0.2),
                    _term("qdes_limit_barrier_probe", 3, 0, 0.0, 0.0),
                    _term("racket_position", 3, 1, 1.0, 0.08),
                ],
            },
        ],
    }


def _safety(recipe, manifest_sha):
    def transition(
        *,
        action_id,
        action_uid,
        reason_class,
        termination_term,
        env_id,
        common_step_counter,
        joint_policy_step_sequence,
        receipt,
    ):
        row = {
            "action_id": action_id,
            "action_uid": action_uid,
            "reason_classes": [reason_class],
            "primary_reason_class": reason_class,
            "termination_terms": [termination_term],
            "rising_termination_terms": [termination_term],
            "env_id": env_id,
            "common_step_counter": common_step_counter,
            "joint_policy_step_sequence": joint_policy_step_sequence,
            "reset_generation": 0,
            "swing_generation": joint_policy_step_sequence - 10,
            "birth_receipt_sha256": receipt,
            "timed_out_same_step": False,
            "pre_terminal_reason_mask": {
                name: False
                for name in (
                    "base_fell_tilt",
                    "base_too_low",
                    "joint_actual_forbidden",
                    "joint_qdes_forbidden",
                    "robot_hit_table",
                    "anchor_pos",
                    "anchor_ori",
                    "ee_body_pos",
                )
            },
            "post_terminal_reason_mask": {
                name: name == termination_term
                for name in (
                    "base_fell_tilt",
                    "base_too_low",
                    "joint_actual_forbidden",
                    "joint_qdes_forbidden",
                    "robot_hit_table",
                    "anchor_pos",
                    "anchor_ori",
                    "ee_body_pos",
                )
            },
            "death_raw_value": 1.0,
            # 2026-08-05(exp §5.6 第 7 条):-10.0 * dt 0.02 = -0.2(原 -6.0)。
            "death_weighted_contribution": -0.2,
            "death_activation": {
                "term_name": "death_penalty",
                "eligible": True,
                "active": True,
                "raw": 1.0,
                "weighted": -0.2,
                "step_dt_s": 0.02,
                "effective": True,
            },
            "reason_specific_penalties": [],
        }
        row["transition_id"] = AUDIT._safety_transition_id(4, row)
        return row

    return {
        "event": AUDIT.SAFETY_TRANSITION_EVENT,
        "schema_version": 2,
        "recipe_sha256": recipe["sha256"],
        "ppo_update": 4,
        "step_dt_s": 0.02,
        "coverage": "complete_update",
        "manifest_sha256": manifest_sha,
        "action_order": ["a", "b"],
        "soft_limit_term_names": [
            "joint_limit",
            "qdes_limit_barrier",
        ],
        "hard_safety_termination_term_names": [
            "base_fell_tilt",
            "base_too_low",
            "joint_actual_forbidden",
            "joint_qdes_forbidden",
            "robot_hit_table",
        ],
        "reference_envelope_termination_term_names": [
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
        ],
        "termination_term_order": [
            "base_fell_tilt",
            "base_too_low",
            "joint_actual_forbidden",
            "joint_qdes_forbidden",
            "robot_hit_table",
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
        ],
        "soft_limit_by_action_term": [
            {
                "action_id": "a",
                "action_uid": 101,
                "term_name": "joint_limit",
                "observed_sample_count": 3,
                "eligible_sample_count": 3,
                "active_sample_count": 1,
                "raw_sum": 1.0,
                "weighted_sum": -0.1,
                "terminated_active_sample_count": 1,
                "step_dt_s": 0.02,
                "effective": True,
                "terminal_reward": False,
            },
            {
                "action_id": "a",
                "action_uid": 101,
                "term_name": "qdes_limit_barrier",
                "observed_sample_count": 3,
                "eligible_sample_count": 3,
                "active_sample_count": 1,
                "raw_sum": 1.0,
                "weighted_sum": -0.1,
                "terminated_active_sample_count": 0,
                "step_dt_s": 0.02,
                "effective": True,
                "terminal_reward": False,
            },
            {
                "action_id": "b",
                "action_uid": 202,
                "term_name": "joint_limit",
                "observed_sample_count": 3,
                "eligible_sample_count": 3,
                "active_sample_count": 1,
                "raw_sum": 2.0,
                "weighted_sum": -0.2,
                "terminated_active_sample_count": 0,
                "step_dt_s": 0.02,
                "effective": True,
                "terminal_reward": False,
            },
            {
                "action_id": "b",
                "action_uid": 202,
                "term_name": "qdes_limit_barrier",
                "observed_sample_count": 3,
                "eligible_sample_count": 3,
                "active_sample_count": 1,
                "raw_sum": 2.0,
                "weighted_sum": -0.2,
                "terminated_active_sample_count": 0,
                "step_dt_s": 0.02,
                "effective": True,
                "terminal_reward": False,
            },
        ],
        "terminal_transitions": [
            transition(
                action_id="a",
                action_uid=101,
                reason_class="hard_limit",
                termination_term="joint_qdes_forbidden",
                env_id=0,
                common_step_counter=11,
                joint_policy_step_sequence=10,
                receipt="d" * 64,
            ),
            transition(
                action_id="a",
                action_uid=101,
                reason_class="table_hit",
                termination_term="robot_hit_table",
                env_id=0,
                common_step_counter=12,
                joint_policy_step_sequence=11,
                receipt="d" * 64,
            ),
            transition(
                action_id="b",
                action_uid=202,
                reason_class="fall",
                termination_term="base_fell_tilt",
                env_id=1,
                common_step_counter=13,
                joint_policy_step_sequence=12,
                receipt="e" * 64,
            ),
        ],
    }


def _action_ledger(manifest_sha):
    def row(**overrides):
        value = {
            "P": 2,
            "A": 2,
            "I": 2,
            "S": 2,
            "C": 0,
            "L": 0,
            "F": 0,
            "U_table": 0,
            "U_fall": 0,
            "U_collision": 0,
            "U_joint_qdes": 0,
            "U_joint_actual": 0,
            "X": 0,
        }
        value.update(overrides)
        return value

    return {
        "event": AUDIT.ACTION_BALL_LEDGER_EVENT,
        "schema_version": 1,
        "step": 4,
        "manifest_sha256": manifest_sha,
        "status": "report_only_requires_frozen_checkpoint_evidence",
        "action_order": ["a", "b"],
        "ledger": {
            "a": row(C=2, U_table=1, U_joint_qdes=1),
            "b": row(C=1, U_fall=1),
        },
        "solver_rejections": {"101": {}, "202": {}},
        "pool": {
            "101": {
                "requests": 2,
                "refill_calls": 1,
                "proposed": 2,
                "admitted": 2,
                "issued": 2,
                "discarded": 0,
                "pending": 0,
            },
            "202": {
                "requests": 2,
                "refill_calls": 1,
                "proposed": 2,
                "admitted": 2,
                "issued": 2,
                "discarded": 0,
                "pending": 0,
            },
        },
        "curriculum": {
            "101": {
                "phase": "center",
                "frontiers": {},
                "expected_domains": [],
            },
            "202": {
                "phase": "center",
                "frontiers": {},
                "expected_domains": [],
            },
        },
    }


def _sparse(index=(), values=()):
    index = [list(item) for item in index]
    values = list(values)
    return {
        "index": index,
        "value": values,
        "nonzero_cells": len(index),
        "event_count": sum(values),
    }


def _joint_sidecar_payload():
    receipts = ["d" * 64, "e" * 64]
    identity_rows = [
        {
            "action_episode_sequence": [0, 0],
            "episode_length": [index, index],
            "action_uid": [101, 202],
            "birth_generation": [0, 0],
            "swing_generation": [index, index],
            "birth_receipt_sha256": receipts,
        }
        for index in range(3)
    ]
    identity_hashes = [
        AUDIT._runner_identity_sha256(row, action_ball_enabled=True)
        for row in identity_rows
    ]
    counter_rows = [
        {
            "qdes_joint_count": _sparse(((0, 0),), (1,)),
            "policy_crossing_joint_count": _sparse(((0, 0),), (1,)),
            "substep_hard_crossing_joint_count": _sparse(((0, 0),), (1,)),
            "actual_hard_edge_joint_count": _sparse(),
        },
        {
            "qdes_joint_count": _sparse(),
            "policy_crossing_joint_count": _sparse(),
            "substep_hard_crossing_joint_count": _sparse(),
            "actual_hard_edge_joint_count": _sparse(),
        },
        {
            "qdes_joint_count": _sparse(),
            "policy_crossing_joint_count": _sparse(),
            "substep_hard_crossing_joint_count": _sparse(),
            "actual_hard_edge_joint_count": _sparse(),
        },
    ]
    policy_steps = []
    for index, counters in enumerate(counter_rows):
        policy_steps.append(
            {
                "policy_step_sequence": 10 + index,
                "policy_start_timestamp_s": float(index) * 0.02,
                "identity_sha256": identity_hashes[index],
                "minimum_hard_gap_rad": 0.02 if index == 0 else 1.0,
                "minimum_hard_gap_env_id": 0,
                "minimum_hard_gap_joint_id": 0,
                "per_action_minimum_hard_gap": {
                    "action_uid": [101, 202],
                    "minimum_gap_rad": (
                        [[0.02, 1.0], [1.0, 1.0]]
                        if index == 0
                        else [[1.0, 1.0], [1.0, 1.0]]
                    ),
                },
                "sparse_counters": counters,
            }
        )

    def transcript(sequence, *, crossing):
        count = 5
        q = [[0.0, 0.0] for _ in range(count)]
        if crossing:
            q[0][0] = 0.98
        qdot = [[0.0, 0.0] for _ in range(count)]
        lower_gap = [[value + 1.0 for value in row] for row in q]
        upper_gap = [[1.0 - value for value in row] for row in q]
        hard_crossing = [
            [crossing and index == 0, False] for index in range(count)
        ]
        actual_hard = [[False, False] for _ in range(count)]
        step_counter = [1, 0] if crossing else [0, 0]
        start = float(sequence - 10) * 0.02
        timestamps = tuple(start + index * 0.005 for index in range(count))
        return {
            "schema_version": 1,
            "policy_step_sequence": sequence,
            "policy_start_timestamp_s": start,
            "expected_apply_calls": 4,
            "physics_dt_s": 0.005,
            "apply_call_count": 4,
            "post_readback_count": 1,
            "complete": True,
            "record_count": count,
            "record_kind": ("apply", "apply", "apply", "apply", "post"),
            "call_index": tuple(range(count)),
            "timestamp_s": timestamps,
            "joint_pos_timestamp_s": timestamps,
            "joint_vel_timestamp_s": timestamps,
            "env_valid": FakeTensor([True] * count, np.bool_),
            "q": FakeTensor(q, np.float32),
            "qdot": FakeTensor(qdot, np.float32),
            "hard_lower_gap": FakeTensor(lower_gap, np.float32),
            "hard_upper_gap": FakeTensor(upper_gap, np.float32),
            "hard_crossing": FakeTensor(hard_crossing, np.bool_),
            "actual_hard_edge": FakeTensor(actual_hard, np.bool_),
            "qdes_env_latch": FakeTensor(bool(crossing), np.bool_),
            "crossing_env_latch": FakeTensor(bool(crossing), np.bool_),
            "qdes_joint_latch": FakeTensor(
                [bool(crossing), False], np.bool_
            ),
            "crossing_joint_latch": FakeTensor(
                [bool(crossing), False], np.bool_
            ),
            "qdes_joint_count": FakeTensor(step_counter, np.int64),
            "crossing_joint_count": FakeTensor(step_counter, np.int64),
            "substep_crossing_joint_latch": FakeTensor(
                [bool(crossing), False], np.bool_
            ),
            "substep_actual_joint_latch": FakeTensor(
                [False, False], np.bool_
            ),
            "substep_crossing_joint_count": FakeTensor(
                step_counter, np.int64
            ),
            "substep_actual_joint_count": FakeTensor([0, 0], np.int64),
            "step_qdes_joint_count": FakeTensor(step_counter, np.int64),
            "step_policy_crossing_joint_count": FakeTensor(
                step_counter, np.int64
            ),
        }

    def archive(
        sequence,
        env_id,
        action_uid,
        receipt,
        *,
        archive_sequence,
        crossing,
    ):
        value = {
            "storage": "full_forensic",
            "archive": {
                "archive_sequence": archive_sequence,
                "env_id": env_id,
                "policy_step_sequence": sequence,
                "action_episode_sequence": 0,
                "episode_length": sequence - 9,
                "episode_length_at_policy_start": sequence - 10,
                "episode_length_at_reset_hook": sequence - 9,
                "action_ball_enabled": True,
                "action_uid": action_uid,
                "birth_generation": 0,
                "swing_generation": sequence - 10,
                "birth_receipt_sha256": receipt,
                "reasons": (
                    ["unsafe", "reset"] if crossing else ["reset"]
                ),
                "reset_hook_observed": True,
                "terminated": True,
                "timed_out": False,
                "termination_status_available": True,
                "included_in_accumulator": True,
                "accumulator_consume_sequence": 0,
                "transcript": transcript(sequence, crossing=crossing),
                "payload_bytes": 0,
            },
        }
        value["archive"]["payload_bytes"] = AUDIT._joint_payload_bytes(
            value["archive"]
        )
        return value

    hard_lower = FakeTensor([-1.0, -1.0], np.float32)
    hard_upper = FakeTensor([1.0, 1.0], np.float32)
    scalar_contract = {
        "expected_apply_calls": 4,
        "physics_dt_s": 0.005,
        "margin_rad": 0.01,
        "margin_fraction": 0.01,
        "num_envs": 2,
        "joint_count": 2,
        "joint_names": ["j0", "j1"],
    }
    contract_digest = hashlib.sha256()
    contract_digest.update(
        json.dumps(
            scalar_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    for name, tensor in (
        ("hard_lower", hard_lower),
        ("hard_upper", hard_upper),
    ):
        contract_digest.update(name.encode("ascii"))
        contract_digest.update(str(tensor.dtype).encode("ascii"))
        contract_digest.update(tensor.numpy().tobytes(order="C"))
    payload = {
        "event": AUDIT.JOINT_SAFETY_EVENT,
        "schema_version": 2,
        "status": "prepared_before_optimizer",
        "rank": 0,
        "ppo_update": 4,
        "contract": {
            **scalar_contract,
            "hard_lower": hard_lower,
            "hard_upper": hard_upper,
            "sha256": contract_digest.hexdigest(),
        },
        "sequence": {
            "consume_sequence": 0,
            "first_policy_step_sequence": 10,
            "last_policy_step_sequence": 12,
            "last_archive_sequence": 2,
        },
        "completeness": {
            "all_rows_present": True,
            "all_policy_steps_complete": True,
            "expected_apply_readbacks": 4,
            "expected_post_readbacks": 1,
            "timestamp_invariant": True,
        },
        "identity": {
            "encoding": (
                "initial_full_plus_sparse_reset_birth_changes_and_dense_"
                "episode_length_swing_generation"
            ),
            "num_envs": 2,
            "action_ball_enabled": True,
            "initial": {
                "action_episode_sequence": [0, 0],
                "action_uid": [101, 202],
                "birth_generation": [0, 0],
            },
            "episode_length_int32": [[0, 0], [1, 1], [2, 2]],
            "swing_generation_int32": [[0, 0], [1, 1], [2, 2]],
            "changes": {
                "action_episode_sequence": {"index": [], "value": []},
                "action_uid": {"index": [], "value": []},
                "birth_generation": {"index": [], "value": []},
            },
            "initial_birth_receipt_sha256": receipts,
            "birth_receipt_changes": [],
            "per_step_identity_sha256": identity_hashes,
        },
        "policy_steps": policy_steps,
        "aggregate_sparse_counters": {
            "qdes_joint_count": _sparse(((0, 0),), (1,)),
            "policy_crossing_joint_count": _sparse(((0, 0),), (1,)),
            "substep_hard_crossing_joint_count": _sparse(((0, 0),), (1,)),
            "actual_hard_edge_joint_count": _sparse(),
        },
        "gaps": {
            "minimum_lower_gap_by_joint": [1.0, 1.0],
            "minimum_lower_gap_env_id_by_joint": [0, 0],
            "minimum_upper_gap_by_joint": [0.02, 1.0],
            "minimum_upper_gap_env_id_by_joint": [0, 0],
        },
        "fatal_flags": {
            "actual_hard_edge_event_count": 0,
            "nonpositive_physical_hard_gap_cell_count": 0,
        },
        "terminal": {
            "archive_count": 3,
            "entries": [
                archive(
                    10,
                    0,
                    101,
                    receipts[0],
                    archive_sequence=0,
                    crossing=True,
                ),
                archive(
                    11,
                    0,
                    101,
                    receipts[0],
                    archive_sequence=1,
                    crossing=False,
                ),
                archive(
                    12,
                    1,
                    202,
                    receipts[1],
                    archive_sequence=2,
                    crossing=False,
                ),
            ],
        },
        "budgets": {
            "core_payload_bytes": 0,
            "terminal_payload_bytes": 0,
            "total_payload_bytes": 0,
            "core_payload_max_bytes": 1024,
            "terminal_payload_max_bytes": 2048,
            "normal_serialized_max_bytes": 2048,
            "forensic_serialized_max_bytes": 4096,
        },
    }
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"terminal", "budgets"}
    }
    payload["budgets"]["core_payload_bytes"] = AUDIT._joint_payload_bytes(core)
    payload["budgets"]["terminal_payload_bytes"] = AUDIT._joint_payload_bytes(
        payload["terminal"]
    )
    payload["budgets"]["total_payload_bytes"] = (
        payload["budgets"]["core_payload_bytes"]
        + payload["budgets"]["terminal_payload_bytes"]
    )
    return payload


def _joint_safety(artifact_sha, artifact_size, commit_receipt, payload):
    return {
        "event": AUDIT.JOINT_SAFETY_EVENT,
        "schema_version": 2,
        "status": "optimizer_committed_and_ledger_acknowledged",
        "ppo_update": 4,
        "consume_sequence": 0,
        "num_envs": 2,
        "policy_step_count": 3,
        "first_policy_step_sequence": 10,
        "last_policy_step_sequence": 12,
        "complete_env_policy_steps": 6,
        "incomplete_env_policy_steps": 0,
        "minimum_hard_gap_rad": 0.02,
        "counter_totals": {
            "qdes_joint_count": 1,
            "policy_crossing_joint_count": 1,
            "substep_hard_crossing_joint_count": 1,
            "actual_hard_edge_joint_count": 0,
        },
        "fatal_flags": payload["fatal_flags"],
        "terminal_archive_count": 3,
        "terminal_reason_counts": {"reset": 3, "unsafe": 1},
        "per_policy_step_sparse_counters": [
            {
                "policy_step_sequence": row["policy_step_sequence"],
                "identity_sha256": row["identity_sha256"],
                "complete_env_count": 2,
                "incomplete_env_count": 0,
                "minimum_hard_gap_rad": row["minimum_hard_gap_rad"],
                "sparse_counters": {
                    name: {
                        "nonzero_cells": counter["nonzero_cells"],
                        "event_count": counter["event_count"],
                    }
                    for name, counter in row["sparse_counters"].items()
                },
            }
            for row in payload["policy_steps"]
        ],
        "identity_binding": (
            "lossless_initial_per_env_identity_plus_sparse_generation_"
            "transitions_and_per_step_sha256"
        ),
        "artifact": {
            "format": "torch_save_cpu",
            "schema_version": 2,
            "path": (
                "joint_safety_ledgers/"
                "ppo_update_00000004_rank_0000.prepared.pt"
            ),
            "sha256": artifact_sha,
            "size_bytes": artifact_size,
            "status": "prepared_before_optimizer",
        },
        "optimizer_commit": commit_receipt,
    }


def _fixture(tmp_path, *, include_complete=True):
    run_dir = tmp_path / "run"
    artifact = (
        run_dir
        / "joint_safety_ledgers"
        / "ppo_update_00000004_rank_0000.prepared.pt"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture-joint-safety-sidecar")
    sidecar_payload = _joint_sidecar_payload()
    recipe = _recipe()
    manifest = _manifest()
    recipe_path = run_dir / "params" / "effective_reward_recipe.json"
    recipe_path.parent.mkdir(parents=True)
    contract_path = run_dir / "params" / "training_contract.json"
    manifest_path = tmp_path / "manifest.json"
    events_path = tmp_path / "run.log"
    _write_json(recipe_path, recipe)
    _write_json(contract_path, {"effective_reward_recipe": recipe})
    _write_json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    commit_path = (
        run_dir
        / "joint_safety_ledgers"
        / "ppo_update_00000004_rank_0000.optimizer_commit.json"
    )
    commit_marker = {
        "event": "hope_joint_safety_optimizer_commit",
        "schema_version": 1,
        "ppo_update": 4,
        "rank": 0,
        "prepared_artifact_path": str(artifact.relative_to(run_dir)),
        "prepared_artifact_sha256": artifact_sha,
        "consume_sequence": 0,
        "status": "optimizer_succeeded_pending_ledger_ack",
    }
    _write_json(commit_path, commit_marker)
    commit_receipt = {
        "format": "canonical_json",
        "schema_version": 1,
        "path": str(commit_path.relative_to(run_dir)),
        "sha256": hashlib.sha256(commit_path.read_bytes()).hexdigest(),
        "size_bytes": commit_path.stat().st_size,
    }
    records = [
        _activation(),
        _episode_closure(),
        _action_ledger(manifest_sha),
        _joint_safety(
            artifact_sha,
            artifact.stat().st_size,
            commit_receipt,
            sidecar_payload,
        ),
    ]
    if include_complete:
        records.extend(
            [
                _per_action(recipe, manifest_sha),
                _safety(recipe, manifest_sha),
            ]
        )
    lines = []
    for record in records:
        prefix = {
            AUDIT.ACTIVATION_EVENT: (
                "HOPE_EFFECTIVE_REWARD_ACTIVATION_UPDATE_JSON="
            ),
            AUDIT.PER_ACTION_EVENT: (
                "HOPE_EFFECTIVE_REWARD_BY_ACTION_UPDATE_JSON="
            ),
            AUDIT.SAFETY_TRANSITION_EVENT: (
                "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON="
            ),
            AUDIT.EPISODE_SEGMENTED_CLOSURE_EVENT: (
                "HOPE_REWARD_EPISODE_SEGMENTED_CLOSURE_UPDATE_JSON="
            ),
            AUDIT.JOINT_SAFETY_EVENT: "HOPE_JOINT_SAFETY_UPDATE_JSON=",
        }.get(record["event"], "")
        lines.append(prefix + _canonical(record))
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recipe_path, manifest_path, events_path, run_dir, sidecar_payload


def _rewrite_event(path, event_name, mutate):
    rewritten = []
    found = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        object_start = line.find("{")
        if object_start < 0:
            rewritten.append(line)
            continue
        prefix = line[:object_start]
        record = json.loads(line[object_start:])
        if record.get("event") == event_name:
            mutate(record)
            found += 1
        rewritten.append(prefix + _canonical(record))
    assert found == 1
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def test_complete_artifact_set_passes_but_is_not_called_isaac_evidence(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "PASS"
    assert report["failures"] == []
    assert report["summary"]["per_action_available"] is True
    assert report["summary"]["negative_semantics_available"] is True
    runtime = report["reward_group_runtime"]
    assert runtime["reward_group_taxonomy"]["group_order"] == list(
        AUDIT.REWARD_TAXONOMY.ACTION_BALL_REWARD_GROUP_ORDER
    )
    assert [row["ppo_update"] for row in runtime["updates"]] == [4]
    assert [
        row["action_id"] for row in runtime["updates"][0]["actions"]
    ] == ["a", "b"]
    first_groups = {
        row["group"]: row
        for row in runtime["updates"][0]["actions"][0]["reward_groups"]
    }
    assert first_groups[
        AUDIT.REWARD_TAXONOMY.ACTION_BALL_REWARD_GROUP_HOPE_TASK
    ]["weighted_p50"] == 0.0
    assert first_groups[
        AUDIT.REWARD_TAXONOMY.ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY
    ]["negative_return_fraction"] == 1.0
    unsigned_runtime = dict(runtime)
    del unsigned_runtime["sha256"]
    assert runtime["sha256"] == _sha(unsigned_runtime)
    assert report["evidence_scope"] == "offline_artifact_consistency_only"
    assert report["isaac_runtime_evidence"] is False


def test_missing_or_fail_closed_episode_closure_is_rejected(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    kept = [
        line
        for line in events.read_text(encoding="utf-8").splitlines()
        if AUDIT.EPISODE_SEGMENTED_CLOSURE_EVENT not in line
    ]
    events.write_text("\n".join(kept) + "\n", encoding="utf-8")
    missing = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )
    assert missing["status"] == "FAIL_CLOSED"
    assert missing["checks"]["episode_segmented_reward_closure"][
        "status"
    ] == "FAIL_CLOSED"

    recipe, manifest, events, run_dir, sidecar = _fixture(
        tmp_path / "failed"
    )
    _rewrite_event(
        events,
        AUDIT.EPISODE_SEGMENTED_CLOSURE_EVENT,
        lambda record: record.__setitem__("status", "FAIL_CLOSED"),
    )
    failed = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )
    assert failed["status"] == "FAIL_CLOSED"
    assert "episode_closure_status" in {
        row["code"] for row in failed["failures"]
    }


def test_only_completed_live_reset_hook_receipt_is_labeled_isaac_e2(
    tmp_path,
):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def make_live_completed(record):
        term_names = record["all_reward_manager_term_names"]
        record["evidence_source"] = "live_isaac_reward_manager"
        record["completed_episode_count"] = 1
        record["completed_episode_segments"] = [
            {
                "env_id": 0,
                "reset_generation": 0,
                "segment_key": [0, 0],
                "action_uid": 101,
                "step_count": 3,
                "terminated": True,
                "timed_out": False,
                "administrative_reset": False,
                "reward_buf_sum": 0.0,
                "all_term_sum": 0.0,
                "reward_buf_vs_all_terms_abs_error": 0.0,
                "local_term_sums": [0.0] * len(term_names),
                "reward_manager_episode_sums": [0.0]
                * len(term_names),
            }
        ]
        record["open_episode_segments"][0]["reset_generation"] = 1
        record["open_episode_segments"][0]["step_count"] = 0
        record["reset_batches"] = [
            {
                "env_ids": [0],
                "reset_generations": [0],
                "administrative_reset": False,
                "normalization_divisor_s": 10.0,
                "terms": [
                    {
                        "name": name,
                        "reward_manager_episode_sum_mean": 0.0,
                        "expected_dashboard_value": 0.0,
                        "actual_dashboard_value": 0.0,
                        "abs_error": 0.0,
                    }
                    for name in term_names
                ],
                "status": "PASS",
            }
        ]
        record["dashboard_normalization"] = {
            "status": "PASS",
            "reset_batch_count": 1,
            "reason": None,
        }
        record["checks"]["reset_batch_count"] = 1
        record["checks"]["completed_episode_count"] = 1
        record["checks"]["dashboard_term_comparison_count"] = len(
            term_names
        )
        record["checks"]["manager_clear_comparison_count"] = len(
            term_names
        )
        record["checks"][
            "all_observed_dashboard_values_normalized_exactly"
        ] = "PASS"
        record["e2_eligible"] = True
        record["e2_ineligible_reason"] = None

    _rewrite_event(
        events,
        AUDIT.EPISODE_SEGMENTED_CLOSURE_EVENT,
        make_live_completed,
    )
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "PASS"
    assert report["isaac_runtime_evidence"] is True
    assert report["evidence_scope"] == (
        "offline_validation_of_live_isaac_reward_manager_receipts"
    )


def test_joint_table_overlap_closes_once_but_counts_both_raw_safety_flags(
    tmp_path,
):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def add_overlapping_table_mask(record):
        transition = record["terminal_transitions"][0]
        transition["reason_classes"] = ["table_hit", "hard_limit"]
        transition["primary_reason_class"] = "table_hit"
        transition["termination_terms"] = [
            "joint_qdes_forbidden",
            "robot_hit_table",
        ]
        transition["rising_termination_terms"] = [
            "joint_qdes_forbidden",
            "robot_hit_table",
        ]
        transition["post_terminal_reason_mask"]["robot_hit_table"] = True
        transition["transition_id"] = AUDIT._safety_transition_id(4, transition)

    _rewrite_event(
        events, AUDIT.SAFETY_TRANSITION_EVENT, add_overlapping_table_mask
    )

    def count_both_overlapping_raw_safety_flags(record):
        record["ledger"]["a"]["U_table"] = 2

    _rewrite_event(
        events,
        AUDIT.ACTION_BALL_LEDGER_EVENT,
        count_both_overlapping_raw_safety_flags,
    )
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )
    failure_codes = {failure["code"] for failure in report["failures"]}

    # This fixture intentionally loses the isolated-hard negative control.
    # Its overlapping raw masks still close the attempt once while proving
    # both the q_des and table safety signals independently.
    assert "hard_limit_isolated_trigger_unproven" in failure_codes
    assert "safety_action_outcome_binding" not in failure_codes

    def wrongly_hide_table_overlap_behind_joint_precedence(record):
        row = record["ledger"]["a"]
        row["U_table"] = 1

    _rewrite_event(
        events,
        AUDIT.ACTION_BALL_LEDGER_EVENT,
        wrongly_hide_table_overlap_behind_joint_precedence,
    )
    bad_report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert "safety_action_outcome_binding" in {
        failure["code"] for failure in bad_report["failures"]
    }


def test_activation_term_digest_binds_exact_recipe_params(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    qbar = next(
        term for term in payload["terms"] if term["name"] == "qdes_limit_barrier"
    )
    qbar["params"]["margin_frac"] = 0.09
    unsigned = {
        "schema_version": payload["schema_version"],
        "terms": payload["terms"],
    }
    payload["sha256"] = _sha(unsigned)
    _write_json(recipe, payload)
    _write_json(
        run_dir / "params" / "training_contract.json",
        {"effective_reward_recipe": payload},
    )
    for event_name in (
        AUDIT.ACTIVATION_EVENT,
        AUDIT.PER_ACTION_EVENT,
        AUDIT.SAFETY_TRANSITION_EVENT,
    ):
        _rewrite_event(
            events,
            event_name,
            lambda record, digest=payload["sha256"]: record.__setitem__(
                "recipe_sha256", digest
            ),
        )

    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "activation_recipe_binding" in {
        failure["code"] for failure in report["failures"]
    }


def test_composed_objective_needs_a_nonzero_runtime_activation(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def zero_racket_activation(record):
        row = next(
            item for item in record["terms"] if item["name"] == "racket_position"
        )
        row["nonzero_sample_count"] = 0
        row["raw_sum"] = 0.0
        row["weighted_sum"] = 0.0
        record["total_weighted_reward_sum"] = -220.8

    def zero_racket_per_action(record):
        for action in record["actions"]:
            row = next(
                item
                for item in action["terms"]
                if item["name"] == "racket_position"
            )
            row["nonzero_sample_count"] = 0
            row["raw_sum"] = 0.0
            row["weighted_sum"] = 0.0

    _rewrite_event(
        events, AUDIT.ACTIVATION_EVENT, zero_racket_activation
    )
    _rewrite_event(
        events, AUDIT.PER_ACTION_EVENT, zero_racket_per_action
    )
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "objective_activation_unproven" in {
        failure["code"] for failure in report["failures"]
    }


def test_joint_sidecar_content_must_close_to_stdout_receipt(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    bad_sidecar = copy.deepcopy(sidecar)
    bad_sidecar["aggregate_sparse_counters"]["qdes_joint_count"][
        "event_count"
    ] = 2

    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: bad_sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "joint_safety_artifact_content" in {
        failure["code"] for failure in report["failures"]
    }


def test_joint_sidecar_recomputes_contract_and_identity_hashes(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    for field, mutate in (
        (
            "contract",
            lambda payload: payload["contract"].__setitem__("sha256", "0" * 64),
        ),
        (
            "identity",
            lambda payload: payload["identity"][
                "per_step_identity_sha256"
            ].__setitem__(0, "0" * 64),
        ),
    ):
        bad_sidecar = copy.deepcopy(sidecar)
        mutate(bad_sidecar)
        report = AUDIT.audit_reward_run(
            recipe_path=recipe,
            event_paths=[events],
            manifest_path=manifest,
            run_dir=run_dir,
            joint_sidecar_loader=lambda _path, payload=bad_sidecar: payload,
        )
        assert report["status"] == "FAIL_CLOSED", field
        assert "joint_safety_artifact_content" in {
            failure["code"] for failure in report["failures"]
        }


def test_full_forensic_transcript_requires_complete_physics_schema(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    bad_sidecar = copy.deepcopy(sidecar)
    del bad_sidecar["terminal"]["entries"][0]["archive"]["transcript"]["q"]
    bad_sidecar["budgets"]["terminal_payload_bytes"] = AUDIT._joint_payload_bytes(
        bad_sidecar["terminal"]
    )
    bad_sidecar["budgets"]["total_payload_bytes"] = (
        bad_sidecar["budgets"]["core_payload_bytes"]
        + bad_sidecar["budgets"]["terminal_payload_bytes"]
    )

    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: bad_sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "joint_safety_artifact_content" in {
        failure["code"] for failure in report["failures"]
    }


def test_solver_pool_conservation_is_not_report_only(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def break_pool(record):
        record["pool"]["101"]["issued"] = 1

    _rewrite_event(events, AUDIT.ACTION_BALL_LEDGER_EVENT, break_pool)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "action_ledger_solver_pool_conservation" in {
        failure["code"] for failure in report["failures"]
    }


def test_duplicate_per_action_term_is_rejected(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def duplicate_term(record):
        record["actions"][0]["terms"].append(
            copy.deepcopy(record["actions"][0]["terms"][0])
        )

    _rewrite_event(events, AUDIT.PER_ACTION_EVENT, duplicate_term)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "per_action_term_duplicate" in {
        failure["code"] for failure in report["failures"]
    }


def test_per_action_reward_group_taxonomy_is_exact_recipe_bound(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def forge_taxonomy(record):
        active = record["reward_group_taxonomy"]["active_terms"]
        racket = next(row for row in active if row["name"] == "racket_position")
        racket["group"] = AUDIT.REWARD_TAXONOMY.ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY

    _rewrite_event(events, AUDIT.PER_ACTION_EVENT, forge_taxonomy)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "per_action_header" in {
        failure["code"] for failure in report["failures"]
    }
    assert report["reward_group_runtime"] is None


def test_per_action_reward_group_signed_sum_and_fraction_are_recomputed(
    tmp_path,
):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def forge_group_sum(record):
        group = next(
            row
            for row in record["actions"][0]["reward_groups"]
            if row["group"]
            == AUDIT.REWARD_TAXONOMY.ACTION_BALL_REWARD_GROUP_HOPE_TASK
        )
        group["weighted_sum"] += 1.0

    _rewrite_event(events, AUDIT.PER_ACTION_EVENT, forge_group_sum)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )
    assert "per_action_reward_group_closure" in {
        failure["code"] for failure in report["failures"]
    }

    recipe, manifest, events, run_dir, sidecar = _fixture(
        tmp_path / "fraction"
    )

    def forge_group_fraction(record):
        group = next(
            row
            for row in record["actions"][0]["reward_groups"]
            if row["group"]
            == AUDIT.REWARD_TAXONOMY.ACTION_BALL_REWARD_GROUP_HOPE_TASK
        )
        group["positive_return_fraction"] = 0.5

    _rewrite_event(events, AUDIT.PER_ACTION_EVENT, forge_group_fraction)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )
    assert "per_action_reward_group_fraction" in {
        failure["code"] for failure in report["failures"]
    }


def test_termination_reason_class_is_derived_from_term_names(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def mislabel(record):
        record["terminal_transitions"][0]["reason_classes"] = ["fall"]

    _rewrite_event(events, AUDIT.SAFETY_TRANSITION_EVENT, mislabel)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "termination_reason_mapping" in {
        failure["code"] for failure in report["failures"]
    }


def test_transition_id_binds_env_action_generation_and_reasons(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def drift_identity(record):
        record["terminal_transitions"][0]["reset_generation"] += 1

    _rewrite_event(events, AUDIT.SAFETY_TRANSITION_EVENT, drift_identity)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "terminal_transition_id_binding" in {
        failure["code"] for failure in report["failures"]
    }


def test_transition_identity_must_cross_bind_joint_terminal_archive(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def forge_receipt(record):
        transition = record["terminal_transitions"][0]
        transition["birth_receipt_sha256"] = "f" * 64
        transition["transition_id"] = AUDIT._safety_transition_id(4, transition)

    _rewrite_event(events, AUDIT.SAFETY_TRANSITION_EVENT, forge_receipt)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "terminal_transition_joint_archive_binding" in {
        failure["code"] for failure in report["failures"]
    }


def test_malformed_nested_event_returns_structured_fail_closed(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def corrupt_row(record):
        record["actions"][0] = None

    _rewrite_event(events, AUDIT.PER_ACTION_EVENT, corrupt_row)
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert report["checks"]["per_action_reward_accounting"]["status"] == "FAIL_CLOSED"
    assert any(
        failure["code"] in {
            "per_action_identity",
            "per_action_reward_accounting",
        }
        for failure in report["failures"]
    )


def test_current_aggregate_only_shape_fails_closed_on_missing_evidence(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(
        tmp_path, include_complete=False
    )
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    codes = {failure["code"] for failure in report["failures"]}
    assert "per_action_reward_accounting" in codes
    assert "negative_reward_semantics" in codes
    assert report["isaac_runtime_evidence"] is False


def test_recipe_sha_tamper_is_rejected_before_runtime_claims(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    _write_json(recipe, payload)

    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert report["checks"]["recipe_integrity"]["status"] == "FAIL_CLOSED"


def test_stacked_specific_terminal_penalty_is_rejected(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    payload["terms"].append(
        {
            "name": "table_hit_penalty",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp.rewards."
                "terminated_by_term"
            ),
            "weight": -1800.0,
            "params": {"term_name": "robot_hit_table"},
        }
    )
    payload["terms"] = sorted(payload["terms"], key=lambda item: item["name"])
    unsigned = {
        "schema_version": payload["schema_version"],
        "terms": payload["terms"],
    }
    payload["sha256"] = _sha(unsigned)
    _write_json(recipe, payload)

    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    codes = {failure["code"] for failure in report["failures"]}
    assert "terminal_specific_penalty_active" in codes


def test_each_soft_limit_needs_a_nonterminal_nonzero_runtime_sample(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def remove_nonterminal_qdes_coverage(record):
        for row in record["soft_limit_by_action_term"]:
            if row["term_name"] == "qdes_limit_barrier":
                row["terminated_active_sample_count"] = row[
                    "active_sample_count"
                ]

    _rewrite_event(
        events,
        AUDIT.SAFETY_TRANSITION_EVENT,
        remove_nonterminal_qdes_coverage,
    )
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "qdes_limit_barrier_nonterminal_trigger_unproven" in {
        failure["code"] for failure in report["failures"]
    }


def test_each_soft_limit_needs_an_observed_zero_interior_control(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def erase_joint_limit_zero_denominator(record):
        for row in record["soft_limit_by_action_term"]:
            if row["term_name"] == "joint_limit":
                row["observed_sample_count"] = row["active_sample_count"]
                row["eligible_sample_count"] = row["active_sample_count"]

    _rewrite_event(
        events,
        AUDIT.SAFETY_TRANSITION_EVENT,
        erase_joint_limit_zero_denominator,
    )
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "joint_limit_interior_zero_unproven" in {
        failure["code"] for failure in report["failures"]
    }


def test_positive_soft_intrusion_cannot_claim_a_subfloor_charge(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def graze_below_floor(record):
        row = next(
            item
            for item in record["soft_limit_by_action_term"]
            if item["action_id"] == "a"
            and item["term_name"] == "qdes_limit_barrier"
        )
        row["raw_sum"] = 0.01
        row["weighted_sum"] = -0.008

    _rewrite_event(
        events, AUDIT.SAFETY_TRANSITION_EVENT, graze_below_floor
    )
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "soft_limit_activation_binding" in {
        failure["code"] for failure in report["failures"]
    }


def test_terminal_event_must_carry_exactly_one_adopted_minus72_charge(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)

    def undercharge_table(record):
        row = next(
            item
            for item in record["terminal_transitions"]
            if item["primary_reason_class"] == "table_hit"
        )
        row["death_weighted_contribution"] = -71.0
        row["death_activation"]["weighted"] = -71.0

    _rewrite_event(
        events, AUDIT.SAFETY_TRANSITION_EVENT, undercharge_table
    )
    report = AUDIT.audit_reward_run(
        recipe_path=recipe,
        event_paths=[events],
        manifest_path=manifest,
        run_dir=run_dir,
        joint_sidecar_loader=lambda _path: sidecar,
    )

    assert report["status"] == "FAIL_CLOSED"
    assert "death_once_per_transition" in {
        failure["code"] for failure in report["failures"]
    }


def test_cli_writes_no_clobber_report_and_returns_gate_code(tmp_path):
    recipe, manifest, events, run_dir, sidecar = _fixture(tmp_path)
    output = tmp_path / "reward_audit.json"
    original_loader = AUDIT._torch_sidecar_loader
    AUDIT._torch_sidecar_loader = lambda _path: sidecar
    try:
        code = AUDIT.main(
            [
                "--recipe",
                str(recipe),
                "--events",
                str(events),
                "--manifest",
                str(manifest),
                "--run-dir",
                str(run_dir),
                "--output",
                str(output),
            ]
        )
    finally:
        AUDIT._torch_sidecar_loader = original_loader

    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    AUDIT._torch_sidecar_loader = lambda _path: sidecar
    try:
        assert (
            AUDIT.main(
                [
                    "--recipe",
                    str(recipe),
                    "--events",
                    str(events),
                    "--manifest",
                    str(manifest),
                    "--run-dir",
                    str(run_dir),
                    "--output",
                    str(output),
                ]
            )
            == 2
        )
    finally:
        AUDIT._torch_sidecar_loader = original_loader
