"""Host-only tests for per-PPO-update effective Reward activation evidence."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils"
    / "effective_reward_recipe.py"
)
SPEC = importlib.util.spec_from_file_location(
    "effective_reward_activation_under_test", MODULE_PATH
)
RECIPE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RECIPE)


def sparse_objective():
    pass


def negative_objective():
    pass


def dense_objective():
    pass


def activation_probe():
    pass


def is_terminated():
    pass


def action_ball_safety_terminated():
    pass


def qdes_limit_barrier_v2():
    pass


def actual_joint_limit_barrier_v2():
    pass


def terminated_by_term():
    pass


class StatefulInactiveCallable:
    def __call__(self):
        pass


@dataclasses.dataclass
class FakeTermCfg:
    func: object
    weight: float
    params: dict = dataclasses.field(default_factory=dict)


def _death_cfg(weight):
    return FakeTermCfg(
        action_ball_safety_terminated,
        weight,
        params={
            "term_names": [
                "base_fell_tilt",
                "base_too_low",
                "joint_actual_forbidden",
                "joint_qdes_forbidden",
                "robot_hit_table",
            ]
        },
    )


class NumpyTensorOps:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, np.ndarray)

    @staticmethod
    def detach(value):
        return value

    @staticmethod
    def as_tensor_like(values, like):
        return np.asarray(values, dtype=like.dtype)

    @staticmethod
    def isfinite(value):
        return np.isfinite(value)

    @staticmethod
    def logical_not(value):
        return np.logical_not(value)

    @staticmethod
    def greater(left, right):
        return np.greater(left, right)

    @staticmethod
    def abs(value):
        return np.abs(value)

    @staticmethod
    def sum(value, axis=None):
        return np.sum(value, axis=axis)

    @staticmethod
    def count_nonzero(value, axis=None):
        return np.count_nonzero(value, axis=axis)

    @staticmethod
    def max(value, axis=None):
        return np.max(value, axis=axis)

    @staticmethod
    def maximum(left, right):
        return np.maximum(left, right)

    @staticmethod
    def stack(values, axis=0):
        return np.stack(tuple(values), axis=axis)

    @staticmethod
    def to_host_list(value):
        return value.tolist()

    @staticmethod
    def to_host_scalar(value):
        return value.item() if hasattr(value, "item") else value


class FakeRewardManager:
    def __init__(self, term_cfgs, *, num_envs, max_episode_length_s):
        self.active_terms = list(term_cfgs)
        self._term_cfgs = dict(term_cfgs)
        self._max_episode_length_s = max_episode_length_s
        self._step_reward = np.zeros(
            (num_envs, len(self.active_terms)), dtype=np.float64
        )
        self._reward_buf = np.zeros(num_envs, dtype=np.float64)
        self._episode_sums = {
            name: np.zeros(num_envs, dtype=np.float64)
            for name in self.active_terms
        }

    def get_term_cfg(self, name):
        return self._term_cfgs[name]

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self._reward_buf.shape[0]))
        env_ids = list(env_ids)
        extras = {
            f"Episode_Reward/{name}": (
                float(np.mean(values[env_ids]))
                / self._max_episode_length_s
            )
            for name, values in self._episode_sums.items()
        }
        for values in self._episode_sums.values():
            values[env_ids] = 0.0
        return extras


class FakeEnv:
    def __init__(self, term_cfgs, *, num_envs=2, step_dt=0.1):
        self.step_dt = step_dt
        self.max_episode_length_s = 10.0
        self.common_step_counter = 0
        self.reward_manager = FakeRewardManager(
            term_cfgs,
            num_envs=num_envs,
            max_episode_length_s=self.max_episode_length_s,
        )

    def set_raw_step(self, values_by_name, *, reward_buf_transform=None):
        columns = []
        for name in self.reward_manager.active_terms:
            cfg = self.reward_manager.get_term_cfg(name)
            raw = np.asarray(values_by_name.get(name, [0.0, 0.0]), dtype=np.float64)
            columns.append(raw * float(cfg.weight))
        self.reward_manager._step_reward = np.stack(columns, axis=1)
        reward_buf = np.sum(
            self.reward_manager._step_reward * self.step_dt, axis=1
        )
        for index, name in enumerate(self.reward_manager.active_terms):
            self.reward_manager._episode_sums[name] += (
                self.reward_manager._step_reward[:, index] * self.step_dt
            )
        if reward_buf_transform is not None:
            reward_buf = reward_buf_transform(reward_buf)
        self.reward_manager._reward_buf = np.asarray(reward_buf, dtype=np.float64)
        self.common_step_counter += 1


def _ledger(env, *, expected_steps, task_kind="action_ball"):
    return RECIPE.EffectiveRewardActivationLedger(
        env,
        task_kind=task_kind,
        expected_environment_step_count=expected_steps,
        tensor_ops=NumpyTensorOps(),
    )


def _terms_by_name(record):
    return {term["name"]: term for term in record["terms"]}


_TERMINATION_ORDER = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)


def _termination_snapshot(*, num_envs=2, active=()):
    reasons = {
        name: np.zeros(num_envs, dtype=np.bool_)
        for name in _TERMINATION_ORDER
    }
    for name, env_ids in active:
        reasons[name][list(env_ids)] = True
    terminated = np.zeros(num_envs, dtype=np.bool_)
    for value in reasons.values():
        terminated |= value
    return {
        "term_order": _TERMINATION_ORDER,
        "terminated": terminated,
        "time_outs": np.zeros(num_envs, dtype=np.bool_),
        "reason_masks": reasons,
    }


def _action_bound_ledger(env, identity_state, termination_state):
    return RECIPE.ActionBoundRewardEvidenceLedger(
        env,
        expected_environment_step_count=1,
        action_contract={
            "action_order": ["a", "b"],
            "action_uids": [101, 202],
            "manifest": {"file_sha256": "a" * 64},
        },
        action_identity_provider=lambda: identity_state["value"],
        termination_snapshot_provider=lambda: termination_state["value"],
        tensor_ops=NumpyTensorOps(),
    )


def test_sparse_early_event_survives_later_zero_steps_with_sign_and_dt_identity():
    env = FakeEnv(
        {
            "negative": FakeTermCfg(negative_objective, -3.0),
            "sparse": FakeTermCfg(sparse_objective, 2.0),
        },
        step_dt=0.1,
    )
    ledger = _ledger(env, expected_steps=3)

    # Both activations happen in the first rollout step. The final manager cache
    # is all zeros, so a last-step-only logger would lose both events.
    env.set_raw_step({"negative": [2.0, 0.0], "sparse": [5.0, 0.0]})
    ledger.observe_after_environment_step()
    for _ in range(2):
        env.set_raw_step({})
        ledger.observe_after_environment_step()

    record = ledger.finish_update(ppo_update=7)
    terms = _terms_by_name(record)

    assert record["environment_step_count"] == 3
    assert record["observed_sample_count"] == 6
    assert record["common_step_counter_start"] == 0
    assert record["common_step_counter_end"] == 3
    assert terms["sparse"]["observed_sample_count"] == 6
    assert terms["sparse"]["nonzero_sample_count"] == 1
    assert terms["sparse"]["weighted_sum"] == pytest.approx(5.0 * 2.0 * 0.1)
    assert terms["sparse"]["raw_sum"] == pytest.approx(5.0)
    assert terms["negative"]["nonzero_sample_count"] == 1
    assert terms["negative"]["weighted_sum"] == pytest.approx(2.0 * -3.0 * 0.1)
    assert terms["negative"]["raw_sum"] == pytest.approx(2.0)
    assert terms["negative"]["raw_recovery"].startswith("validated_weighted_eq_raw")
    assert terms["negative"]["raw_recomposition_max_abs_error"] == pytest.approx(0.0)
    assert terms["negative"]["eligibility"] == "unknown"
    assert terms["negative"]["eligibility_reason"] == "term_specific_mask_unavailable"
    assert record["reward_cache_contract"]["total_reward_closure"] == "validated"

    canonical = RECIPE.canonical_effective_reward_activation_json(record)
    assert canonical == RECIPE.canonical_effective_reward_activation_json(record)


def test_zero_diagnostic_probe_is_separate_from_optimized_objective():
    env = FakeEnv(
        {
            "activation_probe": FakeTermCfg(activation_probe, 1.0),
            "dense": FakeTermCfg(dense_objective, 4.0),
        },
        step_dt=0.02,
    )
    ledger = _ledger(env, expected_steps=2, task_kind="upper_safe")
    for dense_values in ([1.0, 0.5], [0.25, 0.0]):
        env.set_raw_step(
            {
                "activation_probe": [0.0, 0.0],
                "dense": dense_values,
            }
        )
        ledger.observe_after_environment_step()

    record = ledger.finish_update(ppo_update=0)
    terms = _terms_by_name(record)
    probe = terms["activation_probe"]

    assert record["task_kind"] == "upper_safe"
    assert record["objective_term_names"] == ["dense"]
    assert record["diagnostic_probe_term_names"] == ["activation_probe"]
    assert probe["role"] == "diagnostic_probe"
    assert probe["observed_environment_step_count"] == 2
    assert probe["observed_sample_count"] == 4
    assert probe["nonzero_sample_count"] == 0
    assert probe["weighted_sum"] == 0.0
    assert probe["raw_sum"] == 0.0
    assert terms["dense"]["role"] == "objective"
    assert terms["dense"]["nonzero_sample_count"] == 3


def test_nonzero_diagnostic_probe_fails_closed_instead_of_becoming_objective():
    env = FakeEnv(
        {
            "activation_probe": FakeTermCfg(activation_probe, 1.0),
            "dense": FakeTermCfg(dense_objective, 1.0),
        }
    )
    ledger = _ledger(env, expected_steps=1)
    env.set_raw_step(
        {"activation_probe": [1.0, 0.0], "dense": [0.0, 0.0]}
    )
    ledger.observe_after_environment_step()

    with pytest.raises(
        RECIPE.RewardActivationLedgerError,
        match="diagnostic reward probe .* non-zero",
    ):
        ledger.finish_update(ppo_update=0)


@pytest.mark.parametrize(
    "reward_buf_transform",
    [
        lambda correct: correct / 0.05,
        lambda correct: -correct,
    ],
    ids=("missing-dt", "wrong-sign"),
)
def test_reward_cache_contract_rejects_missing_dt_or_wrong_sign(
    reward_buf_transform,
):
    env = FakeEnv(
        {"negative": FakeTermCfg(negative_objective, -3.0)},
        step_dt=0.05,
    )
    ledger = _ledger(env, expected_steps=1)
    env.set_raw_step(
        {"negative": [2.0, 1.0]},
        reward_buf_transform=reward_buf_transform,
    )
    ledger.observe_after_environment_step()

    with pytest.raises(
        RECIPE.RewardActivationLedgerError,
        match="_step_reward does not close to _reward_buf",
    ):
        ledger.finish_update(ppo_update=0)


def test_exact_environment_step_count_is_required_at_update_boundary():
    env = FakeEnv({"dense": FakeTermCfg(dense_objective, 1.0)})
    ledger = _ledger(env, expected_steps=2)
    env.set_raw_step({"dense": [1.0, 0.0]})
    ledger.observe_after_environment_step()

    with pytest.raises(
        RECIPE.RewardActivationLedgerError,
        match=r"observed 1 env\.step calls.*expected 2",
    ):
        ledger.finish_update(ppo_update=0)


def test_zero_weight_term_is_not_given_raw_or_activation_claims():
    env = FakeEnv(
        {
            "dense": FakeTermCfg(dense_objective, 1.0),
            "inactive": FakeTermCfg(StatefulInactiveCallable(), 0.0),
        }
    )
    ledger = _ledger(env, expected_steps=1)
    env.set_raw_step({"dense": [1.0, 0.0], "inactive": [999.0, 999.0]})
    ledger.observe_after_environment_step()
    record = ledger.finish_update(ppo_update=0)

    assert [term["name"] for term in record["terms"]] == ["dense"]


def test_action_bound_ledger_freezes_identity_and_proves_one_death_for_multi_reason():
    env = FakeEnv(
        {
            "death_penalty": _death_cfg(-3600.0),
            "joint_limit": FakeTermCfg(actual_joint_limit_barrier_v2, -40.0),
            "qdes_limit_barrier": FakeTermCfg(qdes_limit_barrier_v2, -40.0),
            "racket_position": FakeTermCfg(dense_objective, 4.0),
        },
        step_dt=0.02,
    )
    identity_state = {
        "value": {
            "action_uid": np.asarray([101, 202], dtype=np.int64),
            "reset_generation": np.asarray([7, 8], dtype=np.int64),
            "swing_generation": np.asarray([2, 3], dtype=np.int64),
            "birth_receipt_sha256": ("b" * 64, "c" * 64),
        }
    }
    termination_state = {"value": _termination_snapshot()}
    ledger = _action_bound_ledger(
        env, identity_state, termination_state
    )

    token = ledger.begin_environment_step()
    # Model an in-step terminal reset that changes the live action identity.
    # Accounting must stay bound to the action that actually produced the
    # transition, captured before env.step.
    identity_state["value"] = {
        "action_uid": np.asarray([202, 101], dtype=np.int64),
        "reset_generation": np.asarray([8, 9], dtype=np.int64),
        "swing_generation": np.asarray([0, 0], dtype=np.int64),
        "birth_receipt_sha256": ("d" * 64, "e" * 64),
    }
    env.set_raw_step(
        {
            "death_penalty": [1.0, 1.0],
            "joint_limit": [0.0, 2.0],
            "qdes_limit_barrier": [1.0, 0.0],
            "racket_position": [3.0, 4.0],
        }
    )
    termination_state["value"] = _termination_snapshot(
        active=(
            ("robot_hit_table", (0,)),
            ("joint_qdes_forbidden", (0,)),
            ("base_fell_tilt", (1,)),
        )
    )
    env.reward_manager.reset([0, 1])
    ledger.observe_after_environment_step(token)
    prepared = ledger.prepare_update(
        12, joint_first_policy_step_sequence=77
    )

    assert prepared["status"] == "frozen_validated_before_optimizer"
    conservation = prepared["action_ball_conservation"]
    assert conservation["status"] == "PASS"
    assert conservation["e2_eligible"] is True
    assert conservation["completed_episode_count"] == 2
    assert [
        row["segment_key"]
        for row in conservation["completed_episode_segments"]
    ] == [[0, 7], [1, 8]]
    assert conservation["dashboard_normalization"]["status"] == "PASS"
    by_action = {
        row["action_id"]: row for row in prepared["per_action"]["actions"]
    }
    assert by_action["a"]["observed_sample_count"] == 1
    assert by_action["b"]["observed_sample_count"] == 1
    a_terms = _terms_by_name(by_action["a"])
    b_terms = _terms_by_name(by_action["b"])
    assert a_terms["qdes_limit_barrier"]["raw_sum"] == pytest.approx(1.0)
    assert b_terms["joint_limit"]["raw_sum"] == pytest.approx(2.0)
    taxonomy = prepared["per_action"]["reward_group_taxonomy"]
    assert taxonomy["authority_sha256"] == (
        RECIPE.ACTION_BALL_REWARD_GROUP_TAXONOMY_AUTHORITY_SHA256
    )
    assert taxonomy["group_order"] == list(
        RECIPE.ACTION_BALL_REWARD_GROUP_ORDER
    )
    a_groups = {
        row["group"]: row for row in by_action["a"]["reward_groups"]
    }
    assert a_groups[
        RECIPE.ACTION_BALL_REWARD_GROUP_HOPE_TASK
    ]["weighted_sum"] == pytest.approx(3.0 * 4.0 * 0.02)
    assert a_groups[
        RECIPE.ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY
    ]["weighted_sum"] == pytest.approx(-72.8)
    assert a_groups[
        RECIPE.ACTION_BALL_REWARD_GROUP_MJLAB_STABILITY
    ]["eligible_sample_count"] == 0
    assert a_groups[
        RECIPE.ACTION_BALL_REWARD_GROUP_BEYONDMIMIC
    ]["weighted_p50"] is None
    assert a_groups[
        RECIPE.ACTION_BALL_REWARD_GROUP_HOPE_TASK
    ]["positive_return_fraction"] == pytest.approx(1.0)
    assert a_groups[
        RECIPE.ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY
    ]["negative_return_fraction"] == pytest.approx(1.0)

    soft = prepared["safety"]["soft_limit_by_action_term"]
    assert [(row["action_id"], row["term_name"]) for row in soft] == [
        ("a", "joint_limit"),
        ("a", "qdes_limit_barrier"),
        ("b", "joint_limit"),
        ("b", "qdes_limit_barrier"),
    ]
    assert all(row["effective"] is True for row in soft)
    assert all(row["terminal_reward"] is False for row in soft)

    transitions = prepared["safety"]["terminal_transitions"]
    assert len(transitions) == 2
    table_and_hard = transitions[0]
    assert table_and_hard["action_id"] == "a"
    assert table_and_hard["action_uid"] == 101
    assert table_and_hard["reset_generation"] == 7
    assert table_and_hard["birth_receipt_sha256"] == "b" * 64
    assert table_and_hard["reason_classes"] == ["table_hit", "hard_limit"]
    assert table_and_hard["primary_reason_class"] == "table_hit"
    assert table_and_hard["termination_terms"] == [
        "joint_qdes_forbidden",
        "robot_hit_table",
    ]
    assert table_and_hard["post_terminal_reason_mask"][
        "joint_qdes_forbidden"
    ]
    assert table_and_hard["post_terminal_reason_mask"]["robot_hit_table"]
    assert table_and_hard["death_raw_value"] == pytest.approx(1.0)
    assert table_and_hard["death_weighted_contribution"] == pytest.approx(
        -72.0
    )
    assert table_and_hard["reason_specific_penalties"] == []
    assert table_and_hard["joint_policy_step_sequence"] == 77
    assert transitions[1]["reason_classes"] == ["fall"]
    assert transitions[1]["primary_reason_class"] == "fall"
    assert transitions[1]["joint_policy_step_sequence"] == 77

    ledger.acknowledge_update(prepared)
    next_token = ledger.begin_environment_step()
    assert type(next_token) is object


def test_episode_closure_persists_open_sums_and_binds_external_reset():
    env = FakeEnv(
        {
            "death_penalty": _death_cfg(-3600.0),
            "joint_limit": FakeTermCfg(
                actual_joint_limit_barrier_v2, -40.0
            ),
            "qdes_limit_barrier": FakeTermCfg(
                qdes_limit_barrier_v2, -40.0
            ),
        },
        step_dt=0.02,
    )
    identity_state = {
        "value": {
            "action_uid": np.asarray([101, 202], dtype=np.int64),
            "reset_generation": np.asarray([0, 0], dtype=np.int64),
            "swing_generation": np.asarray([0, 0], dtype=np.int64),
            "birth_receipt_sha256": ("b" * 64, "c" * 64),
        }
    }
    termination_state = {"value": _termination_snapshot()}
    ledger = _action_bound_ledger(
        env, identity_state, termination_state
    )

    token = ledger.begin_environment_step()
    env.set_raw_step({"qdes_limit_barrier": [1.0, 2.0]})
    ledger.observe_after_environment_step(token)
    first = ledger.prepare_update(
        0, joint_first_policy_step_sequence=0
    )
    first_closure = first["action_ball_conservation"]
    assert first_closure["status"] == "PASS"
    assert first_closure["completed_episode_count"] == 0
    assert first_closure["dashboard_normalization"]["status"] == (
        "NOT_OBSERVED_NO_RESET"
    )
    assert first_closure["e2_eligible"] is False
    ledger.acknowledge_update(first)

    # A fenced global/frozen-evaluation reset can occur between PPO updates.
    # Its Reward segment must be captured without attributing it to either
    # adjacent optimizer rollout.
    env.reward_manager.reset([0])
    identity_state["value"] = {
        "action_uid": np.asarray([202, 202], dtype=np.int64),
        "reset_generation": np.asarray([1, 0], dtype=np.int64),
        "swing_generation": np.asarray([0, 0], dtype=np.int64),
        "birth_receipt_sha256": ("d" * 64, "c" * 64),
    }
    token = ledger.begin_environment_step()
    env.set_raw_step({})
    ledger.observe_after_environment_step(token)
    second = ledger.prepare_update(
        1, joint_first_policy_step_sequence=1
    )
    closure = second["action_ball_conservation"]
    assert closure["status"] == "PASS"
    assert closure["completed_episode_count"] == 1
    segment = closure["completed_episode_segments"][0]
    assert segment["segment_key"] == [0, 0]
    assert segment["administrative_reset"] is True
    assert segment["step_count"] == 1
    assert segment["reward_buf_sum"] == pytest.approx(-0.8)
    assert segment["reward_manager_episode_sums"] == pytest.approx(
        segment["local_term_sums"]
    )
    assert closure["dashboard_normalization"]["status"] == "PASS"
    assert closure["e2_eligible"] is False


def test_action_bound_ledger_rejects_reason_specific_terminal_stack():
    env = FakeEnv(
        {
            "death_penalty": _death_cfg(-3600.0),
            "joint_limit": FakeTermCfg(actual_joint_limit_barrier_v2, -40.0),
            "qdes_limit_barrier": FakeTermCfg(qdes_limit_barrier_v2, -40.0),
            "table_hit_penalty": FakeTermCfg(terminated_by_term, -100.0),
        },
        step_dt=0.02,
    )
    identity_state = {
        "value": {
            "action_uid": np.asarray([101, 202], dtype=np.int64),
            "reset_generation": np.asarray([0, 0], dtype=np.int64),
            "swing_generation": np.asarray([0, 0], dtype=np.int64),
            "birth_receipt_sha256": ("b" * 64, "c" * 64),
        }
    }
    termination_state = {"value": _termination_snapshot()}

    with pytest.raises(
        RECIPE.RewardActivationLedgerError,
        match="no generic/reason-specific stack",
    ):
        _action_bound_ledger(env, identity_state, termination_state)


def test_action_bound_taxonomy_rejects_unmapped_active_term():
    env = FakeEnv(
        {
            "death_penalty": _death_cfg(-3600.0),
            "joint_limit": FakeTermCfg(
                actual_joint_limit_barrier_v2, -40.0
            ),
            "qdes_limit_barrier": FakeTermCfg(
                qdes_limit_barrier_v2, -40.0
            ),
            "racket_position": FakeTermCfg(dense_objective, 4.0),
            "mystery_income": FakeTermCfg(dense_objective, 1.0),
        },
        step_dt=0.02,
    )
    identity_state = {
        "value": {
            "action_uid": np.asarray([101, 202], dtype=np.int64),
            "reset_generation": np.asarray([0, 0], dtype=np.int64),
            "swing_generation": np.asarray([0, 0], dtype=np.int64),
            "birth_receipt_sha256": ("b" * 64, "c" * 64),
        }
    }
    termination_state = {"value": _termination_snapshot()}

    with pytest.raises(
        RECIPE.RewardActivationLedgerError,
        match="has no authoritative taxonomy",
    ):
        _action_bound_ledger(env, identity_state, termination_state)


@pytest.mark.parametrize(
    "bad_term,bad_raw",
    (
        ("joint_limit", -1.0),
        ("racket_position", -1.0),
    ),
    ids=("negative_penalty_pays", "positive_objective_charges"),
)
def test_action_bound_runtime_rejects_wrong_signed_term_contribution(
    bad_term,
    bad_raw,
):
    env = FakeEnv(
        {
            "death_penalty": _death_cfg(-3600.0),
            "joint_limit": FakeTermCfg(
                actual_joint_limit_barrier_v2, -40.0
            ),
            "qdes_limit_barrier": FakeTermCfg(
                qdes_limit_barrier_v2, -40.0
            ),
            "racket_position": FakeTermCfg(dense_objective, 4.0),
        },
        step_dt=0.02,
    )
    identity_state = {
        "value": {
            "action_uid": np.asarray([101, 202], dtype=np.int64),
            "reset_generation": np.asarray([0, 0], dtype=np.int64),
            "swing_generation": np.asarray([0, 0], dtype=np.int64),
            "birth_receipt_sha256": ("b" * 64, "c" * 64),
        }
    }
    termination_state = {"value": _termination_snapshot()}
    ledger = _action_bound_ledger(
        env, identity_state, termination_state
    )

    token = ledger.begin_environment_step()
    env.set_raw_step({bad_term: [bad_raw, 0.0]})

    with pytest.raises(
        RECIPE.RewardActivationLedgerError,
        match="sign invariants",
    ):
        ledger.observe_after_environment_step(token)


def test_crafted_causal_probes_cover_each_objective_and_require_worse_return():
    recipe_terms = [
        {
            "name": "death_penalty",
            "callable": f"{__name__}.is_terminated",
            "weight": -3600.0,
            "params": {},
        },
        {
            "name": "joint_limit",
            "callable": f"{__name__}.actual_joint_limit_barrier_v2",
            "weight": -40.0,
            "params": {},
        },
        {
            "name": "qdes_limit_barrier",
            "callable": f"{__name__}.qdes_limit_barrier_v2",
            "weight": -40.0,
            "params": {},
        },
        {
            "name": "racket_position",
            "callable": f"{__name__}.dense_objective",
            "weight": 4.0,
            "params": {},
        },
    ]
    taxonomy = RECIPE.build_action_ball_reward_group_taxonomy(recipe_terms)
    probes = []
    for index, term in enumerate(taxonomy["active_terms"]):
        positive = term["expected_weight_sign"] == "positive"
        probes.append(
            {
                "term_name": term["name"],
                "callable": term["callable"],
                "changed_axes": [term["causal_axis"]],
                "frozen_context_sha256": "a" * 64,
                "baseline_state_sha256": f"{index + 1:064x}",
                "worsened_state_sha256": f"{index + 101:064x}",
                "baseline_raw": 1.0 if positive else 0.0,
                "worsened_raw": 0.0 if positive else 1.0,
            }
        )

    report = RECIPE.validate_action_ball_reward_causal_probes(
        recipe_terms, probes, step_dt=0.02
    )

    assert report["coverage"] == "every_active_objective_exactly_once"
    assert len(report["probes"]) == 4
    assert all(row["weighted_delta"] < 0.0 for row in report["probes"])
    bad = [dict(row) for row in probes]
    bad[-1]["baseline_raw"], bad[-1]["worsened_raw"] = (
        bad[-1]["worsened_raw"],
        bad[-1]["baseline_raw"],
    )
    with pytest.raises(RECIPE.RewardRecipeError, match="strict negative"):
        RECIPE.validate_action_ball_reward_causal_probes(recipe_terms, bad)
    with pytest.raises(RECIPE.RewardRecipeError, match="every active"):
        RECIPE.validate_action_ball_reward_causal_probes(
            recipe_terms, probes[:-1]
        )


@pytest.mark.parametrize(
    "step_dt,death_weight,joint_weight,qdes_weight,match",
    (
        (0.01, -3600.0, -40.0, -40.0, "step_dt=0.02"),
        (0.02, -1800.0, -40.0, -40.0, "death_penalty weight=-3600.0"),
        (0.02, -3600.0, -20.0, -40.0, "joint_limit weight=-40.0"),
        (0.02, -3600.0, -40.0, -20.0, "qdes_limit_barrier weight=-40.0"),
    ),
)
def test_action_bound_ledger_rejects_nonadopted_negative_dose(
    step_dt,
    death_weight,
    joint_weight,
    qdes_weight,
    match,
):
    env = FakeEnv(
        {
            "death_penalty": _death_cfg(death_weight),
            "joint_limit": FakeTermCfg(
                actual_joint_limit_barrier_v2, joint_weight
            ),
            "qdes_limit_barrier": FakeTermCfg(
                qdes_limit_barrier_v2, qdes_weight
            ),
        },
        step_dt=step_dt,
    )
    identity_state = {
        "value": {
            "action_uid": np.asarray([101, 202], dtype=np.int64),
            "reset_generation": np.asarray([0, 0], dtype=np.int64),
            "swing_generation": np.asarray([0, 0], dtype=np.int64),
            "birth_receipt_sha256": ("b" * 64, "c" * 64),
        }
    }
    termination_state = {"value": _termination_snapshot()}

    with pytest.raises(RECIPE.RewardActivationLedgerError, match=match):
        _action_bound_ledger(env, identity_state, termination_state)


def test_required_unsafe_reason_cannot_hide_behind_timeout_without_death():
    env = FakeEnv(
        {
            "death_penalty": _death_cfg(-3600.0),
            "joint_limit": FakeTermCfg(
                actual_joint_limit_barrier_v2, -40.0
            ),
            "qdes_limit_barrier": FakeTermCfg(
                qdes_limit_barrier_v2, -40.0
            ),
        },
        step_dt=0.02,
    )
    identity_state = {
        "value": {
            "action_uid": np.asarray([101, 202], dtype=np.int64),
            "reset_generation": np.asarray([0, 0], dtype=np.int64),
            "swing_generation": np.asarray([0, 0], dtype=np.int64),
            "birth_receipt_sha256": ("b" * 64, "c" * 64),
        }
    }
    termination_state = {"value": _termination_snapshot()}
    ledger = _action_bound_ledger(env, identity_state, termination_state)
    token = ledger.begin_environment_step()
    env.set_raw_step(
        {
            "death_penalty": [0.0, 0.0],
            "joint_limit": [0.0, 0.0],
            "qdes_limit_barrier": [0.0, 0.0],
        }
    )
    hidden = _termination_snapshot()
    hidden["reason_masks"]["robot_hit_table"][0] = True
    hidden["time_outs"][0] = True
    termination_state["value"] = hidden
    env.reward_manager.reset([0])
    identity_state["value"] = {
        **identity_state["value"],
        "reset_generation": np.asarray([1, 0], dtype=np.int64),
    }

    with pytest.raises(
        RECIPE.RewardActivationLedgerError,
        match="Reward/termination evidence violates",
    ):
        ledger.observe_after_environment_step(token)
