"""Host-only tests for the ActionBall 4096x5 semantic producer boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "action_ball_prelong_semantics.py"
)
RUNNER_PATH = MODULE_PATH.with_name("my_on_policy_runner.py")
SPEC = importlib.util.spec_from_file_location(
    "action_ball_prelong_semantics", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
SEMANTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEMANTICS)


def _counters():
    counters = {
        SEMANTICS.TASK_INVALID_OBSERVED_COUNTER: 7,
        SEMANTICS.TASK_INVALID_REWARD_SUM_COUNTER: 0.0,
        SEMANTICS.TASK_INVALID_REWARD_ELIGIBLE_COUNTER: 0,
        SEMANTICS.READY_MIMIC_REWARD_SUM_COUNTER: 0.5,
        SEMANTICS.READY_MIMIC_ELIGIBLE_COUNTER: 7,
        SEMANTICS.SWING_MIMIC_REWARD_SUM_COUNTER: 3.0,
        SEMANTICS.SWING_MIMIC_ELIGIBLE_COUNTER: 89993,
        SEMANTICS.EXACT_STRIKE_TIMING_COUNTER: 11,
        SEMANTICS.ELIGIBLE_CLOSED_SWING_COUNTER: 9,
        SEMANTICS.ACTUAL_CONTACT_COUNTER: 3,
        SEMANTICS.ACHIEVED_FLIGHT_COUNTER: 2,
        SEMANTICS.UNKNOWN_ATTRIBUTION_COUNTER: 0,
        "unrelated_existing_exact_behavior_counter": 19,
    }
    values = {
        "balance": (4.5, 98304),
        "mimic": (3.5, 90000),
        "strike": (1.5, 11),
        "target": (0.0, 3),
        "outcome": (0.5, 2),
    }
    for group, (income, eligible) in values.items():
        counters[SEMANTICS.reward_group_sum_counter(group)] = income
        counters[SEMANTICS.reward_group_eligible_counter(group)] = eligible
    return counters


def test_builds_fixed_purpose_order_with_explicit_true_eligibility():
    record = SEMANTICS.build_prelong_semantics_update(
        ppo_update=2, counters=_counters()
    )

    assert record["event"] == SEMANTICS.PRELONG_SEMANTICS_EVENT
    assert record["schema_version"] == 3
    assert record["reveal_to_playback_bridge"] is None
    assert record["profile"] == SEMANTICS.PRELONG_PROFILE_A211
    assert record["ppo_update"] == 2
    assert record["window"]["rollout_sample_count"] == 4096 * 24
    assert record["task_invalid"] == {
        "observed_sample_count": 7,
        "task_reward_weighted_sum": 0.0,
        "task_reward_eligible_denominator": 0,
        "eligibility_semantics": (
            "samples observed with the authoritative task_valid mask false; "
            "task reward eligibility is measured from true term masks"
        ),
    }
    assert record["mimic_task_phase_split"] == {
        "task_invalid_ready": {
            "weighted_sum": 0.5,
            "eligible_denominator": 7,
            "eligibility_semantics": (
                "active mimic-term income on samples whose authoritative "
                "pre-step task_valid mask is false"
            ),
        },
        "task_valid_swing": {
            "weighted_sum": 3.0,
            "eligible_denominator": 89993,
            "eligibility_semantics": (
                "active mimic-term income on samples whose authoritative "
                "pre-step task_valid mask is true"
            ),
        },
        "partition_semantics": (
            "the two masks are disjoint and exhaustive over the aggregate "
            "active-mimic sample union"
        ),
    }
    assert record["strike_timing"]["exact_strike_tick_denominator"] == 11
    assert record["hit"]["eligible_closed_swing_count"] == 9
    assert record["hit"]["actual_contact_numerator"] == 3
    assert record["achieved_flight"]["eligible_denominator"] == 2
    assert [row["group"] for row in record["reward_groups"]] == list(
        SEMANTICS.PRELONG_REWARD_GROUPS
    )
    assert all(row["eligibility_semantics"] for row in record["reward_groups"])
    assert record["unknown_attribution_count"] == 0


def test_marker_is_one_finite_canonical_json_line():
    line = SEMANTICS.prelong_semantics_marker_line(ppo_update=0, counters=_counters())
    assert line.startswith(SEMANTICS.PRELONG_SEMANTICS_MARKER_PREFIX)
    assert "\n" not in line
    payload = json.loads(line[len(SEMANTICS.PRELONG_SEMANTICS_MARKER_PREFIX) :])
    assert payload["ppo_update"] == 0
    assert payload["reward_groups"][4]["group"] == "outcome"


def test_zero_invalid_samples_is_valid_for_an_individual_update():
    counters = _counters()
    counters[SEMANTICS.TASK_INVALID_OBSERVED_COUNTER] = 0
    counters[SEMANTICS.READY_MIMIC_REWARD_SUM_COUNTER] = 0.0
    counters[SEMANTICS.READY_MIMIC_ELIGIBLE_COUNTER] = 0
    counters[SEMANTICS.SWING_MIMIC_REWARD_SUM_COUNTER] = 3.5
    counters[SEMANTICS.SWING_MIMIC_ELIGIBLE_COUNTER] = 90000
    record = SEMANTICS.build_prelong_semantics_update(ppo_update=4, counters=counters)
    assert record["task_invalid"]["observed_sample_count"] == 0


@pytest.mark.parametrize(
    ("counter", "value"),
    (
        (SEMANTICS.TASK_INVALID_REWARD_SUM_COUNTER, 0.25),
        (SEMANTICS.TASK_INVALID_REWARD_ELIGIBLE_COUNTER, 1),
    ),
)
def test_rejects_task_reward_leak_while_task_is_invalid(counter, value):
    counters = _counters()
    counters[counter] = value
    with pytest.raises(
        SEMANTICS.PrelongSemanticProducerError,
        match="task_valid=0 must have exactly zero",
    ):
        SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)


def test_rejects_contact_closed_nonconservation():
    counters = _counters()
    counters[SEMANTICS.ELIGIBLE_CLOSED_SWING_COUNTER] = 9
    counters[SEMANTICS.ACTUAL_CONTACT_COUNTER] = 10
    with pytest.raises(
        SEMANTICS.PrelongSemanticProducerError,
        match="contacts must be <= eligible closed swings",
    ):
        SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)


def test_rejects_unknown_attribution_instead_of_hiding_it():
    counters = _counters()
    counters[SEMANTICS.UNKNOWN_ATTRIBUTION_COUNTER] = 1
    with pytest.raises(
        SEMANTICS.PrelongSemanticProducerError,
        match="unknown attribution",
    ):
        SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)


def test_inapplicable_group_stays_present_as_zero_over_zero():
    counters = _counters()
    counters[SEMANTICS.reward_group_sum_counter("target")] = 0.0
    counters[SEMANTICS.reward_group_eligible_counter("target")] = 0
    record = SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)
    target = record["reward_groups"][3]
    assert target["group"] == "target"
    assert target["weighted_sum"] == 0.0
    assert target["eligible_denominator"] == 0


def test_rejects_nonzero_income_for_inapplicable_group():
    counters = _counters()
    counters[SEMANTICS.reward_group_sum_counter("target")] = 0.5
    counters[SEMANTICS.reward_group_eligible_counter("target")] = 0
    with pytest.raises(
        SEMANTICS.PrelongSemanticProducerError,
        match="target reward income is nonzero",
    ):
        SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)


@pytest.mark.parametrize(
    ("counter", "value", "message"),
    (
        (SEMANTICS.ELIGIBLE_CLOSED_SWING_COUNTER, True, "plain integer"),
        (SEMANTICS.ACTUAL_CONTACT_COUNTER, -1, "nonnegative"),
        (
            SEMANTICS.TASK_INVALID_OBSERVED_COUNTER,
            98305,
            "exceeds the fixed",
        ),
        (
            "prelong_balance_reward_weighted_sum",
            float("nan"),
            "must be finite",
        ),
    ),
)
def test_rejects_malformed_or_nonfinite_counter(counter, value, message):
    counters = _counters()
    counters[counter] = value
    with pytest.raises(SEMANTICS.PrelongSemanticProducerError, match=message):
        SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)


def test_every_required_counter_is_fail_closed():
    expected = set(SEMANTICS.required_prelong_counter_names())
    assert expected <= set(_counters())
    for missing in expected:
        counters = _counters()
        del counters[missing]
        with pytest.raises(
            SEMANTICS.PrelongSemanticProducerError,
            match="is missing",
        ):
            SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)


def test_cross_update_closure_is_not_compared_to_same_window_timing():
    counters = _counters()
    counters[SEMANTICS.EXACT_STRIKE_TIMING_COUNTER] = 0
    counters[SEMANTICS.reward_group_eligible_counter("strike")] = 0
    counters[SEMANTICS.reward_group_sum_counter("strike")] = 0.0
    counters[SEMANTICS.ELIGIBLE_CLOSED_SWING_COUNTER] = 4
    counters[SEMANTICS.ACTUAL_CONTACT_COUNTER] = 2

    record = SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)

    assert record["strike_timing"]["exact_strike_tick_denominator"] == 0
    assert record["hit"]["eligible_closed_swing_count"] == 4
    assert record["hit"]["actual_contact_numerator"] == 2


@pytest.mark.parametrize(
    ("counter", "value", "message"),
    (
        (
            SEMANTICS.reward_group_eligible_counter("strike"),
            12,
            "cannot exceed exact-strike timing",
        ),
        (
            SEMANTICS.ACHIEVED_FLIGHT_COUNTER,
            3,
            "outcome reward-group denominator",
        ),
    ),
)
def test_structural_event_denominators_match_reward_group_unions(
    counter, value, message
):
    counters = _counters()
    counters[counter] = value
    with pytest.raises(SEMANTICS.PrelongSemanticProducerError, match=message):
        SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)


@pytest.mark.parametrize(
    ("counter", "value", "message"),
    (
        (
            SEMANTICS.READY_MIMIC_ELIGIBLE_COUNTER,
            6,
            "ready-mimic denominator must equal",
        ),
        (
            SEMANTICS.SWING_MIMIC_ELIGIBLE_COUNTER,
            89992,
            "denominators must exhaust",
        ),
        (
            SEMANTICS.SWING_MIMIC_REWARD_SUM_COUNTER,
            2.9,
            "income must exhaust",
        ),
    ),
)
def test_ready_and_swing_mimic_partition_fails_closed(counter, value, message):
    counters = _counters()
    counters[counter] = value
    with pytest.raises(SEMANTICS.PrelongSemanticProducerError, match=message):
        SEMANTICS.build_prelong_semantics_update(ppo_update=0, counters=counters)


def test_exact_a211_and_c211_nonzero_term_sets_are_frozen():
    common_balance = {
        "upright_exp",
        "hit_unstable_support",
        "foot_slip_sq",
        "foot_velocity",
        "foot_soft_landing",
        "joint_torques",
        "undesired_contacts",
        "base_ang_vel_xy",
        "base_lin_vel_z",
        "joint_vel",
        "action_rate_clamped",
    }
    common_mimic = {
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_racket_position",
        "motion_racket_velocity",
        "motion_racket_normal",
        "motion_racket_long_axis",
    }
    a211 = SEMANTICS.prelong_group_term_weights(SEMANTICS.PRELONG_PROFILE_A211)
    c211 = SEMANTICS.prelong_group_term_weights(SEMANTICS.PRELONG_PROFILE_C211)
    assert set(a211["balance"]) == common_balance
    assert set(c211["balance"]) == common_balance
    assert set(a211["mimic"]) == common_mimic
    assert set(c211["mimic"]) == common_mimic
    assert set(a211["strike"]) == {"strike_capture_bonus"}
    assert set(c211["strike"]) == {"c225_strike_ball_paddle_center_proximity"}
    assert set(a211["target"]) == {
        "racket_progress",
        "racket_position_coarse",
        "racket_velocity_coarse",
        "racket_normal_coarse",
        "racket_position",
        "racket_velocity",
        "racket_normal",
        "racket_position_precision",
        "racket_velocity_precision",
        "racket_normal_precision",
    }
    assert c211["target"] == {}
    assert set(a211["outcome"]) == {
        "virtual_pass_net",
        "virtual_landing_dense",
        "virtual_landing",
    }
    assert set(c211["outcome"]) == {"virtual_landing"}
    assert sum(len(terms) for terms in a211.values()) == 34
    assert sum(len(terms) for terms in c211.values()) == 22
    assert a211["target"] == {
        "racket_progress": 10.0,
        "racket_position_coarse": 11.5,
        "racket_velocity_coarse": 11.5,
        "racket_normal_coarse": 5.75,
        "racket_position": 4.6,
        "racket_velocity": 0.575,
        "racket_normal": 0.575,
        "racket_position_precision": 0.575,
        "racket_velocity_precision": 0.2875,
        "racket_normal_precision": 0.575,
    }
    assert a211["outcome"]["virtual_landing"] == 700.0
    assert c211["strike"] == {
        "c225_strike_ball_paddle_center_proximity": 240.0
    }
    assert c211["outcome"] == {"virtual_landing": 700.0}
    assert SEMANTICS.PRELONG_EXCLUDED_SAFETY_TERM_WEIGHTS == {
        "death_penalty": -10.0,
        "joint_limit": -5.0,
        "qdes_limit_barrier": -5.0,
        "qdes_projection_penalty": -1.0,
    }
    assert SEMANTICS.PRELONG_REQUIRED_TERM_PARAMS == {
        "qdes_projection_penalty": {"objective_weight": -5.0}
    }
    a_callables = SEMANTICS.expected_prelong_callable_names(
        SEMANTICS.PRELONG_PROFILE_A211
    )
    c_callables = SEMANTICS.expected_prelong_callable_names(
        SEMANTICS.PRELONG_PROFILE_C211
    )
    assert a_callables["motion_body_pos"] == "motion_body_pos_swing_only"
    assert a_callables["racket_position_coarse"] == (
        "racket_position_coarse_tracking_cauchy"
    )
    assert a_callables["virtual_landing"] == "virtual_landing"
    assert "base_position" not in a_callables
    assert c_callables["virtual_landing"] == ("c225_landing_outcome_actual_contact")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("unknown", "unknown"),
        ("missing", "missing"),
        ("weight", "weight_drift"),
    ),
)
def test_nonzero_term_classification_fails_closed(mutation, message):
    weights = SEMANTICS.expected_prelong_nonzero_reward_weights(
        SEMANTICS.PRELONG_PROFILE_A211
    )
    if mutation == "unknown":
        weights["mystery_reward"] = 1.0
    elif mutation == "missing":
        del weights["upright_exp"]
    else:
        weights["upright_exp"] = 0.9
    with pytest.raises(SEMANTICS.PrelongSemanticLedgerError, match=message):
        SEMANTICS.classify_prelong_reward_profile(weights)


def test_dedicated_runtime_flag_is_isolated_and_recipe_sealed():
    recipe = "a" * 64
    assert (
        SEMANTICS.parse_prelong_runtime_request({}, reward_ppo_economy_requested=False)
        is None
    )
    assert (
        SEMANTICS.parse_prelong_runtime_request(
            {"HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_GATE": "1"},
            reward_ppo_economy_requested=True,
        )
        is None
    )
    assert (
        SEMANTICS.parse_prelong_runtime_request(
            {
                SEMANTICS.PRELONG_SEMANTICS_ENABLE_ENV: "1",
                SEMANTICS.PRELONG_SEMANTICS_RECIPE_SHA_ENV: recipe,
            },
            reward_ppo_economy_requested=True,
        )
        == recipe
    )


@pytest.mark.parametrize(
    ("environ", "economy", "message"),
    (
        (
            {SEMANTICS.PRELONG_SEMANTICS_ENABLE_ENV: "2"},
            True,
            "exactly 0, 1, or absent",
        ),
        (
            {SEMANTICS.PRELONG_SEMANTICS_ENABLE_ENV: "1"},
            True,
            "requires HOPE_ACTION_BALL_4096X5_PRELONG_REWARD_RECIPE_SHA256",
        ),
        (
            {
                SEMANTICS.PRELONG_SEMANTICS_ENABLE_ENV: "1",
                SEMANTICS.PRELONG_SEMANTICS_RECIPE_SHA_ENV: "a" * 64,
            },
            False,
            "requires HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_GATE=1",
        ),
        (
            {SEMANTICS.PRELONG_SEMANTICS_RECIPE_SHA_ENV: "a" * 64},
            False,
            "is present without",
        ),
        (
            {
                SEMANTICS.PRELONG_SEMANTICS_ENABLE_ENV: "1",
                SEMANTICS.PRELONG_SEMANTICS_RECIPE_SHA_ENV: "A" * 64,
            },
            True,
            "lowercase hexadecimal",
        ),
    ),
)
def test_dedicated_runtime_flag_rejects_ambiguous_enablement(environ, economy, message):
    with pytest.raises(SEMANTICS.PrelongSemanticLedgerError, match=message):
        SEMANTICS.parse_prelong_runtime_request(
            environ,
            reward_ppo_economy_requested=economy,
        )


class _FakeRewardCfg:
    def __init__(self, name, weight, callable_name):
        self.weight = weight

        def reward_func():
            return name

        reward_func.__name__ = callable_name
        reward_func.__qualname__ = callable_name
        reward_func.__module__ = "prelong_test_rewards"
        self.func = reward_func
        self.params = {}
        if name == "qdes_projection_penalty":
            self.params["objective_weight"] = -5.0
        if name in {
            "motion_global_anchor_ori",
            "motion_body_pos",
            "motion_body_ori",
            "motion_body_lin_vel",
            "motion_body_ang_vel",
            "motion_racket_position",
            "motion_racket_velocity",
            "motion_racket_normal",
            "motion_racket_long_axis",
        }:
            self.params["std"] = 0.2


class _FakeRewardManager:
    def __init__(self, profile):
        weights = SEMANTICS.expected_prelong_nonzero_reward_weights(profile)
        callable_names = SEMANTICS.expected_prelong_callable_names(profile)
        self.active_terms = list(weights)
        self._cfgs = {
            name: _FakeRewardCfg(name, weight, callable_names[name])
            for name, weight in weights.items()
        }
        self._indices = {name: index for index, name in enumerate(self.active_terms)}
        self._step_reward = torch.zeros(
            (SEMANTICS.PRELONG_NUM_ENVS, len(self.active_terms)),
            dtype=torch.float64,
        )
        self._reward_buf = torch.zeros(SEMANTICS.PRELONG_NUM_ENVS, dtype=torch.float64)

    def get_term_cfg(self, name):
        return self._cfgs[name]

    def set_cache(self, contributions=None, reward_buf_offset=0.0):
        self._step_reward.zero_()
        for name, value in (contributions or {}).items():
            self._step_reward[:, self._indices[name]] = value
        self._reward_buf.copy_(
            self._step_reward.sum(dim=1) * SEMANTICS.PRELONG_POLICY_DT_S
            + float(reward_buf_offset)
        )


class _FakeCommand:
    def __init__(self, profile):
        n = SEMANTICS.PRELONG_NUM_ENVS
        self.profile = profile
        self._action_ball_task_valid = torch.ones(n, dtype=torch.bool)
        self.pre_strike = torch.zeros(n, dtype=torch.bool)
        self.strike_window = torch.zeros(n, dtype=torch.bool)
        self.strike_window_pos = torch.zeros(n, dtype=torch.bool)
        self.strike_window_wide = torch.zeros(n, dtype=torch.bool)
        self.metrics = {"exact_strike_hit_rate": torch.zeros(n, dtype=torch.float64)}
        self._action_ball_attempt_active = torch.ones(n, dtype=torch.bool)
        self.vb_fired = torch.zeros(n, dtype=torch.bool)
        self.racket_pos_w = torch.zeros((n, 3), dtype=torch.float64)
        self._action_ball_ball_contact_target_w = torch.zeros(
            (n, 3), dtype=torch.float64
        )
        self.vb_landing_xy = torch.zeros((n, 2), dtype=torch.float64)
        self._vb_target_xy = torch.zeros(2, dtype=torch.float64)
        self.vb_landing_valid = torch.zeros(n, dtype=torch.bool)
        self.vb_net_crossed = torch.zeros(n, dtype=torch.bool)
        self.vb_net_clear = torch.zeros(n, dtype=torch.bool)
        self.cfg = SimpleNamespace(vb_landing_sigma=1.0)
        self.closed = 0
        self.contacts = 0

    def action_ball_target_component_valid(self, _name):
        return self.profile == SEMANTICS.PRELONG_PROFILE_A211

    def _action_ball_ledger_payload(self):
        return {
            "only_action": {
                "C": int(self.closed),
                "H": int(self.contacts),
            }
        }


def _runtime(profile=SEMANTICS.PRELONG_PROFILE_A211):
    manager = _FakeRewardManager(profile)
    command = _FakeCommand(profile)
    env = SimpleNamespace(
        reward_manager=manager,
        command_manager=SimpleNamespace(
            get_term=lambda name: command if name == "racket_target" else None
        ),
        step_dt=SEMANTICS.PRELONG_POLICY_DT_S,
        common_step_counter=0,
    )
    ledger = SEMANTICS.ActionBallPrelongSemanticsLedger(
        env,
        preregistered_effective_reward_recipe=(
            SEMANTICS.prelong_runtime_effective_reward_recipe_receipt(manager)
        ),
        require_bridge_telemetry=False,
    )
    return env, command, ledger


class _BridgeCommand(_FakeCommand):
    def __init__(self, profile):
        super().__init__(profile)
        n = SEMANTICS.PRELONG_NUM_ENVS
        self._action_ball_task_wait_schedule = SimpleNamespace(
            min_wait_ticks=5,
            max_wait_ticks=25,
            canonical_sha256="1" * 64,
        )
        self._action_ball_task_wait_total_ticks = torch.full(
            (n,), 5, dtype=torch.long
        )
        self._action_ball_task_wait_elapsed_ticks = torch.full(
            (n,), 5, dtype=torch.long
        )
        self._action_ball_reset_generation = torch.ones(n, dtype=torch.long)
        self._action_ball_task_valid.fill_(True)
        self.pre_strike.fill_(True)
        self.metrics.update(
            {
                "base_upright": torch.ones(n, dtype=torch.float64),
                "foot_contact_frac": torch.ones(n, dtype=torch.float64),
                "foot_slip_speed": torch.zeros(n, dtype=torch.float64),
            }
        )
        self._bridge_motion = SimpleNamespace(
            _action_ball_task_age_s=torch.full((n,), 0.10, dtype=torch.float64),
            _action_ball_time_to_contact_s=torch.full(
                (n,), 1.92, dtype=torch.float64
            ),
            _action_ball_teacher_rate=torch.full(
                (n,), 0.8, dtype=torch.float64
            ),
            _action_ball_scaled_t_hit_s=torch.full(
                (n,), 1.2, dtype=torch.float64
            ),
            _action_ball_pre_swing_wait_s=torch.full(
                (n,), 0.14, dtype=torch.float64
            ),
            imitation_eligible=torch.ones(n, dtype=torch.bool),
        )
        joints = 31
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.zeros((n, joints), dtype=torch.float64),
                joint_vel=torch.zeros((n, joints), dtype=torch.float64),
                joint_pos_limits=torch.tensor(
                    [[-1.0, 1.0]] * joints, dtype=torch.float64
                ),
                joint_vel_limits=torch.ones(joints, dtype=torch.float64) * 10.0,
                root_pos_w=torch.tensor(
                    [[0.0, 0.0, 1.0]], dtype=torch.float64
                ).repeat(n, 1),
                root_lin_vel_w=torch.zeros((n, 3), dtype=torch.float64),
            )
        )
        self._action_ball_task_by_env = [
            SimpleNamespace(
                action_uid=73,
                action_slot=0,
                mount_normal_sign=1,
                question_key="center",
            )
            for _ in range(n)
        ]
        self._action_ball_birth_by_env = [
            SimpleNamespace(question_key="center") for _ in range(n)
        ]
        self._action_ball_manifest = SimpleNamespace(
            actions=[SimpleNamespace(family="backhand")]
        )

    def _motion(self):
        return self._bridge_motion

    def action_ball_hard_contract(self):
        return {
            "action_order": [73],
            "timing": {
                "authority": "test_current_center_receipt",
                "policy_dt_s": SEMANTICS.PRELONG_POLICY_DT_S,
            },
            "target_provider": {
                "source": (
                    "online_solver"
                    if self.profile == SEMANTICS.PRELONG_PROFILE_A211
                    else "direct_ball"
                ),
                "recipe": (
                    "current_lm"
                    if self.profile == SEMANTICS.PRELONG_PROFILE_A211
                    else "outcome_dense_only"
                ),
            },
            "profiles": {"sampler_contract_sha256": "4" * 64},
            "sampling": {"initial_center_single_question": True},
        }


@pytest.fixture
def bridge_authority_imports(monkeypatch):
    question_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_question_cache"
    )
    question_module.exact_question_sha256 = SEMANTICS._canonical_payload_sha256
    command_module = types.ModuleType(
        "whole_body_tracking.tasks.tracking.mdp.hope_commands"
    )

    def exact_payload(
        *, action_uid, action_slot, birth, sample, mount_normal_sign
    ):
        return {
            "action_uid": action_uid,
            "action_slot": action_slot,
            "birth_question_key": birth.question_key,
            "sample_question_key": sample.question_key,
            "mount_normal_sign": mount_normal_sign,
        }

    command_module._action_ball_exact_question_payload = exact_payload
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.tasks.tracking.mdp.action_ball_question_cache",
        question_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.tasks.tracking.mdp.hope_commands",
        command_module,
    )
    return exact_payload


def _bridge_runtime(
    profile=SEMANTICS.PRELONG_PROFILE_A211,
    *,
    command_mutation=None,
    manager_mutation=None,
):
    manager = _FakeRewardManager(profile)
    if manager_mutation is not None:
        manager_mutation(manager)
    command = _BridgeCommand(profile)
    if command_mutation is not None:
        command_mutation(command)
    env = SimpleNamespace(
        reward_manager=manager,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(
                SEMANTICS.PRELONG_NUM_ENVS, dtype=torch.bool
            )
        ),
        step_dt=SEMANTICS.PRELONG_POLICY_DT_S,
        common_step_counter=0,
    )
    ledger = SEMANTICS.ActionBallPrelongSemanticsLedger(
        env,
        preregistered_effective_reward_recipe=(
            SEMANTICS.prelong_runtime_effective_reward_recipe_receipt(manager)
        ),
    )
    return env, command, ledger


def _bridge_step(env, command, ledger, *, contributions=None, terminate=False):
    token = ledger.begin_environment_step()
    env.reward_manager.set_cache(contributions=contributions)
    command._bridge_motion._action_ball_task_age_s.add_(
        SEMANTICS.PRELONG_POLICY_DT_S
    )
    env.termination_manager.terminated.fill_(terminate)
    env.common_step_counter += 1
    ledger.observe_after_environment_step(token)


def _finish_bridge_steps(env, command, ledger, already_observed):
    env.termination_manager.terminated.zero_()
    for _ in range(SEMANTICS.PRELONG_ROLLOUT_STEPS - already_observed):
        _bridge_step(env, command, ledger)


def test_bridge_conserves_mixed_wait_cohorts_and_reports_device_aggregates(
    bridge_authority_imports,
):
    half = SEMANTICS.PRELONG_NUM_ENVS // 2

    def install_mixed_wait(command):
        command._action_ball_task_wait_total_ticks[half:] = 25
        command._action_ball_task_wait_elapsed_ticks[half:] = 25
        command._bridge_motion._action_ball_task_age_s[half:] = 0.50
        command._bridge_motion._action_ball_time_to_contact_s[half:] = 2.32
        command._bridge_motion._action_ball_pre_swing_wait_s[half:] = 0.54

    env, command, ledger = _bridge_runtime(
        SEMANTICS.PRELONG_PROFILE_A211,
        command_mutation=install_mixed_wait,
    )
    mimic_weights = SEMANTICS.prelong_group_term_weights(
        SEMANTICS.PRELONG_PROFILE_A211
    )["mimic"]
    contributions = {
        name: weight * 0.8 for name, weight in mimic_weights.items()
    }
    contributions["racket_progress"] = 1.0

    for _ in range(3):
        _bridge_step(env, command, ledger, contributions=contributions)
    _finish_bridge_steps(env, command, ledger, already_observed=3)
    prepared = ledger.prepare_update(ppo_update=0)
    bridge = prepared.record["reveal_to_playback_bridge"]

    assert bridge["status"] == "active_fail_closed"
    authority = bridge["authority"]
    assert authority["family"] == "backhand"
    assert authority["target_source"] == "online_solver"
    assert authority["target_recipe"] == "current_lm"
    assert authority["question_sha256"] == SEMANTICS._canonical_payload_sha256(
        bridge_authority_imports(
            action_uid=73,
            action_slot=0,
            birth=command._action_ball_birth_by_env[0],
            sample=command._action_ball_task_by_env[0],
            mount_normal_sign=1,
        )
    )
    cohorts = {
        row["wait_ticks"]: row
        for row in bridge["lifetime_conservation"]["wait_cohorts"]
    }
    assert cohorts[5] == {
        "wait_ticks": 5,
        "reveal_count": half,
        "playback_start_count": half,
        "terminal_before_start_count": 0,
        "censored_count": 0,
    }
    assert cohorts[25] == {
        "wait_ticks": 25,
        "reveal_count": half,
        "playback_start_count": half,
        "terminal_before_start_count": 0,
        "censored_count": 0,
    }
    assert all(
        row["reveal_count"] == 0
        for wait_ticks, row in cohorts.items()
        if wait_ticks not in (5, 25)
    )
    timing = bridge["timing_at_reveal"]
    assert timing["reveal_count"] == SEMANTICS.PRELONG_NUM_ENVS
    assert timing["fields"]["time_to_contact_tick"]["mean"] == 91.0
    assert timing["fields"]["teacher_rate"]["mean"] == pytest.approx(0.8)
    assert timing["fields"]["scaled_t_hit_s"]["mean"] == pytest.approx(1.2)
    assert timing["fields"]["pre_swing_wait_s"]["mean"] == pytest.approx(0.04)
    assert timing["fields"]["expected_bridge_ticks"]["mean"] == 3.0

    window = bridge["window"]
    expected_samples = SEMANTICS.PRELONG_NUM_ENVS * 3
    expected_progress_income = expected_samples * SEMANTICS.PRELONG_POLICY_DT_S
    assert window["bridge_sample_count"] == expected_samples
    assert window["task_income_rule"] == (
        "racket_progress_only_base_position_absent_or_zero"
    )
    assert window["task_weighted_income_sum"] == pytest.approx(
        expected_progress_income
    )
    assert window["racket_progress_weighted_income_sum"] == pytest.approx(
        expected_progress_income
    )
    term_rows = {row["term"]: row for row in window["mimic_terms"]}
    assert set(term_rows) == set(mimic_weights)
    for name, weight in mimic_weights.items():
        row = term_rows[name]
        assert row["eligible_denominator"] == expected_samples
        assert row["raw_reward_sum_before_manager_weight"] == pytest.approx(
            expected_samples * 0.8
        )
        assert row["raw_kernel_sum_after_window_scale_removed"] == pytest.approx(
            expected_samples * 0.8
        )
        assert row["finite_error_denominator"] == expected_samples
        assert row["zero_kernel_count"] == 0
        assert row["weighted_income_sum"] == pytest.approx(
            expected_samples * weight * 0.8 * SEMANTICS.PRELONG_POLICY_DT_S
        )
    safety = window["safety"]
    assert safety["sample_count"] == expected_samples
    assert safety["minimum_physical_hard_gap_rad"] == 1.0
    assert safety["maximum_abs_qvel_over_physical_limit"] == 0.0
    assert safety["minimum_root_height_m"] == 1.0
    assert safety["maximum_root_height_m"] == 1.0
    assert safety["minimum_root_upright_cosine"] == 1.0
    assert safety["maximum_root_xy_speed_mps"] == 0.0
    assert safety["mean_foot_contact_fraction"] == 1.0
    assert safety["mean_foot_slip_speed_mps"] == 0.0
    assert safety["maximum_foot_slip_speed_mps"] == 0.0
    ledger.acknowledge_update(prepared)


def test_bridge_terminal_before_playback_closes_lifetime_conservation(
    bridge_authority_imports,
):
    env, command, ledger = _bridge_runtime(SEMANTICS.PRELONG_PROFILE_A211)
    _bridge_step(env, command, ledger, terminate=True)
    _finish_bridge_steps(env, command, ledger, already_observed=1)
    prepared = ledger.prepare_update(ppo_update=0)
    cohorts = {
        row["wait_ticks"]: row
        for row in prepared.record["reveal_to_playback_bridge"][
            "lifetime_conservation"
        ]["wait_cohorts"]
    }
    assert cohorts[5]["reveal_count"] == SEMANTICS.PRELONG_NUM_ENVS
    assert cohorts[5]["playback_start_count"] == 0
    assert cohorts[5]["terminal_before_start_count"] == SEMANTICS.PRELONG_NUM_ENVS
    assert cohorts[5]["censored_count"] == 0
    ledger.acknowledge_update(prepared)


def test_c_bridge_rejects_task_income_even_when_term_is_eligible(
    bridge_authority_imports,
):
    env, command, ledger = _bridge_runtime(SEMANTICS.PRELONG_PROFILE_C211)
    command.metrics["exact_strike_hit_rate"].fill_(1.0)
    _bridge_step(
        env,
        command,
        ledger,
        contributions={"c225_strike_ball_paddle_center_proximity": 1.0},
    )
    _finish_bridge_steps(env, command, ledger, already_observed=1)
    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="bridge_task_income_not_allowed",
    ):
        ledger.prepare_update(ppo_update=0)


def test_bridge_authority_rejects_mixed_question_cohort(
    bridge_authority_imports,
):
    manager = _FakeRewardManager(SEMANTICS.PRELONG_PROFILE_A211)
    command = _BridgeCommand(SEMANTICS.PRELONG_PROFILE_A211)
    command._action_ball_task_by_env[1].question_key = "different"
    env = SimpleNamespace(
        reward_manager=manager,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(
                SEMANTICS.PRELONG_NUM_ENVS, dtype=torch.bool
            )
        ),
        step_dt=SEMANTICS.PRELONG_POLICY_DT_S,
        common_step_counter=0,
    )
    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="contains multiple questions",
    ):
        SEMANTICS.ActionBallPrelongSemanticsLedger(
            env,
            preregistered_effective_reward_recipe=(
                SEMANTICS.prelong_runtime_effective_reward_recipe_receipt(
                    manager
                )
            ),
        )


def test_runtime_requires_bridge_schedule_by_default():
    manager = _FakeRewardManager(SEMANTICS.PRELONG_PROFILE_A211)
    command = _FakeCommand(SEMANTICS.PRELONG_PROFILE_A211)
    env = SimpleNamespace(
        reward_manager=manager,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        step_dt=SEMANTICS.PRELONG_POLICY_DT_S,
        common_step_counter=0,
    )
    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="requires the ActionBall task-wait schedule",
    ):
        SEMANTICS.ActionBallPrelongSemanticsLedger(
            env,
            preregistered_effective_reward_recipe=(
                SEMANTICS.prelong_runtime_effective_reward_recipe_receipt(
                    manager
                )
            ),
        )


def test_bridge_rejects_joint_position_limits_with_wrong_joint_width(
    bridge_authority_imports,
):
    def install_wrong_limits(command):
        command.robot.data.joint_pos_limits = torch.tensor(
            [[-1.0, 1.0]], dtype=torch.float64
        )

    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="joint_pos_limits are not broadcastable",
    ):
        _bridge_runtime(
            SEMANTICS.PRELONG_PROFILE_A211,
            command_mutation=install_wrong_limits,
        )


def test_a_bridge_allows_declared_zero_weight_base_position(
    bridge_authority_imports,
):
    def install_zero_base_position(manager):
        name = "base_position"
        manager.active_terms.append(name)
        manager._cfgs[name] = _FakeRewardCfg(name, 0.0, name)
        manager._indices[name] = len(manager._indices)
        manager._step_reward = torch.cat(
            (
                manager._step_reward,
                torch.zeros(
                    (SEMANTICS.PRELONG_NUM_ENVS, 1), dtype=torch.float64
                ),
            ),
            dim=1,
        )

    _env, _command, ledger = _bridge_runtime(
        SEMANTICS.PRELONG_PROFILE_A211,
        manager_mutation=install_zero_base_position,
    )
    assert ledger._bridge_enabled is True


@pytest.mark.parametrize("mutation", ("callable", "objective_param"))
def test_runtime_taxonomy_rejects_callable_or_objective_param_drift(mutation):
    profile = SEMANTICS.PRELONG_PROFILE_A211
    expected_recipe = SEMANTICS.prelong_runtime_effective_reward_recipe_receipt(
        _FakeRewardManager(profile)
    )
    manager = _FakeRewardManager(profile)
    command = _FakeCommand(profile)
    if mutation == "callable":
        manager._cfgs["upright_exp"].func.__name__ = "wrong_upright"
        message = "callable differs"
    else:
        manager._cfgs["qdes_projection_penalty"].params["objective_weight"] = -0.4
        message = "complete preregistered recipe"
    env = SimpleNamespace(
        reward_manager=manager,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        step_dt=SEMANTICS.PRELONG_POLICY_DT_S,
        common_step_counter=0,
    )
    with pytest.raises(SEMANTICS.PrelongSemanticLedgerError, match=message):
        SEMANTICS.ActionBallPrelongSemanticsLedger(
            env,
            preregistered_effective_reward_recipe=expected_recipe,
            require_bridge_telemetry=False,
        )


def test_complete_normalized_recipe_pin_rejects_preconstruction_param_drift():
    profile = SEMANTICS.PRELONG_PROFILE_A211
    expected_recipe = SEMANTICS.prelong_runtime_effective_reward_recipe_receipt(
        _FakeRewardManager(profile)
    )
    manager = _FakeRewardManager(profile)
    manager._cfgs["upright_exp"].params["std"] = 0.4472135954999579
    command = _FakeCommand(profile)
    env = SimpleNamespace(
        reward_manager=manager,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        step_dt=SEMANTICS.PRELONG_POLICY_DT_S,
        common_step_counter=0,
    )

    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="complete preregistered recipe",
    ):
        SEMANTICS.ActionBallPrelongSemanticsLedger(
            env,
            preregistered_effective_reward_recipe=expected_recipe,
            require_bridge_telemetry=False,
        )


def test_complete_normalized_params_cannot_drift_inside_rollout():
    env, _command, ledger = _runtime()
    token = ledger.begin_environment_step()
    env.reward_manager._cfgs["upright_exp"].params["std"] = 0.5
    env.reward_manager.set_cache()
    env.common_step_counter += 1

    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="complete normalized parameters changed",
    ):
        ledger.observe_after_environment_step(token)


def test_recipe_comparison_ignores_only_resolved_scene_entity_id_cache():
    expected = {
        "__config_type__": "isaaclab.managers.scene_entity_cfg.SceneEntityCfg",
        "fields": {
            "name": "robot",
            "joint_names": [".*"],
            "joint_ids": {"__slice__": [None, None, None]},
            "preserve_order": False,
        },
    }
    resolved = {
        "__config_type__": "isaaclab.managers.scene_entity_cfg.SceneEntityCfg",
        "fields": {
            "name": "robot",
            "joint_names": [".*"],
            "joint_ids": list(range(31)),
            "preserve_order": False,
        },
    }
    normalize = SEMANTICS._without_runtime_entity_resolution
    assert normalize(expected) == normalize(resolved)

    resolved["fields"]["joint_names"] = ["arm_.*"]
    assert normalize(expected) != normalize(resolved)


def _runtime_step(
    env, ledger, *, contributions=None, reward_buf_offset=0.0, in_step_mutation=None
):
    token = ledger.begin_environment_step()
    if in_step_mutation is not None:
        in_step_mutation()
    env.reward_manager.set_cache(
        contributions=contributions,
        reward_buf_offset=reward_buf_offset,
    )
    env.common_step_counter += 1
    ledger.observe_after_environment_step(token)
    return token


def _finish_zero_steps(env, ledger, already_observed):
    for _ in range(SEMANTICS.PRELONG_ROLLOUT_STEPS - already_observed):
        _runtime_step(env, ledger)


def test_a211_and_c211_masks_use_union_denominators_not_term_counts():
    _env, a_command, _ledger = _runtime(SEMANTICS.PRELONG_PROFILE_A211)
    a_command.pre_strike[:2] = True
    a_command.strike_window_pos[1:3] = True
    a_command.strike_window_wide[2:4] = True
    a_command.metrics["exact_strike_hit_rate"][:3] = 1.0
    a_command.vb_fired[:2] = True
    a_masks = SEMANTICS.prelong_eligibility_masks(
        a_command, SEMANTICS.PRELONG_PROFILE_A211
    )["groups"]
    assert int(a_masks["target"].sum().item()) == 4
    assert int(a_masks["strike"].sum().item()) == 3
    assert int(a_masks["outcome"].sum().item()) == 2

    _env, c_command, _ledger = _runtime(SEMANTICS.PRELONG_PROFILE_C211)
    c_command.metrics["exact_strike_hit_rate"][:3] = 1.0
    c_command.racket_pos_w[1, 0] = float("nan")
    c_command.vb_fired[:4] = True
    c_command.vb_landing_valid[:2] = True
    c_command.vb_net_crossed[:3] = True
    c_command.vb_net_clear[:1] = True
    c_snapshot = SEMANTICS.prelong_eligibility_masks(
        c_command, SEMANTICS.PRELONG_PROFILE_C211
    )
    c_masks = c_snapshot["groups"]
    assert int(c_snapshot["exact_strike_timing"].sum().item()) == 3
    assert int(c_masks["strike"].sum().item()) == 2
    assert int(c_masks["target"].sum().item()) == 0
    assert int(c_masks["outcome"].sum().item()) == 4


def test_c211_failed_contact_is_zero_over_one_outcome_opportunity():
    env, command, ledger = _runtime(SEMANTICS.PRELONG_PROFILE_C211)
    command.vb_fired[0] = True
    command.vb_landing_valid[0] = True
    command.vb_net_crossed[0] = True
    command.vb_net_clear[0] = False

    _runtime_step(env, ledger)
    command.vb_fired.zero_()
    _finish_zero_steps(env, ledger, already_observed=1)
    prepared = ledger.prepare_update(ppo_update=0)

    assert prepared.counters[
        SEMANTICS.reward_group_eligible_counter("outcome")
    ] == 1
    assert prepared.counters[SEMANTICS.reward_group_sum_counter("outcome")] == 0.0
    assert prepared.record["achieved_flight"]["eligible_denominator"] == 1
    ledger.acknowledge_update(prepared)


def test_c211_contacted_nonfinite_outcome_state_fails_closed():
    _env, command, _ledger = _runtime(SEMANTICS.PRELONG_PROFILE_C211)
    command.vb_fired[0] = True
    command.vb_landing_xy[0, 0] = float("nan")

    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="actual-contact outcome state must be finite",
    ):
        SEMANTICS.prelong_eligibility_masks(
            command, SEMANTICS.PRELONG_PROFILE_C211
        )


def test_pre_step_snapshot_preserves_hidden_wait_across_in_step_reveal():
    env, command, ledger = _runtime()
    command._action_ball_task_valid.zero_()

    _runtime_step(
        env,
        ledger,
        contributions={"upright_exp": 1.0},
        in_step_mutation=lambda: command._action_ball_task_valid.fill_(True),
    )
    _finish_zero_steps(env, ledger, already_observed=1)
    prepared = ledger.prepare_update(ppo_update=0)

    assert prepared.counters[SEMANTICS.TASK_INVALID_OBSERVED_COUNTER] == 4096
    assert prepared.counters[SEMANTICS.TASK_INVALID_REWARD_SUM_COUNTER] == 0.0
    assert prepared.counters[SEMANTICS.TASK_INVALID_REWARD_ELIGIBLE_COUNTER] == 0
    assert (
        prepared.counters[SEMANTICS.reward_group_eligible_counter("balance")]
        == 4096 * 24
    )
    marker = ledger.acknowledge_update(prepared)
    assert marker.startswith(SEMANTICS.PRELONG_SEMANTICS_MARKER_PREFIX)
    with pytest.raises(SEMANTICS.PrelongSemanticLedgerError, match="stale"):
        ledger.acknowledge_update(prepared)


def test_mimic_phase_split_is_disjoint_exhaustive_and_covers_every_active_term():
    env, command, ledger = _runtime()
    n = SEMANTICS.PRELONG_NUM_ENVS
    task_valid_count = 1024
    command._action_ball_task_valid.zero_()
    command._action_ball_task_valid[:task_valid_count] = True

    mimic_terms = tuple(
        SEMANTICS.prelong_group_term_weights(
            SEMANTICS.PRELONG_PROFILE_A211
        )["mimic"]
    )
    assert len(mimic_terms) == 9
    contributions = {}
    ready_rate_per_sample = 0.0
    swing_rate_per_sample = 0.0
    for index, name in enumerate(mimic_terms, start=1):
        values = torch.full((n,), float(index), dtype=torch.float64)
        values[:task_valid_count] *= 2.0
        contributions[name] = values
        ready_rate_per_sample += float(index)
        swing_rate_per_sample += float(2 * index)

    _runtime_step(
        env,
        ledger,
        contributions=contributions,
        # The frozen pre-step mask, rather than this reveal, owns the first
        # step's accounting partition.
        in_step_mutation=lambda: command._action_ball_task_valid.fill_(True),
    )
    _finish_zero_steps(env, ledger, already_observed=1)
    prepared = ledger.prepare_update(ppo_update=0)

    ready_count = n - task_valid_count
    swing_count = task_valid_count + (SEMANTICS.PRELONG_ROLLOUT_STEPS - 1) * n
    expected_ready_income = (
        ready_count * ready_rate_per_sample * SEMANTICS.PRELONG_POLICY_DT_S
    )
    expected_swing_income = (
        task_valid_count * swing_rate_per_sample * SEMANTICS.PRELONG_POLICY_DT_S
    )
    counters = prepared.counters
    assert counters[SEMANTICS.READY_MIMIC_ELIGIBLE_COUNTER] == ready_count
    assert counters[SEMANTICS.SWING_MIMIC_ELIGIBLE_COUNTER] == swing_count
    assert counters[SEMANTICS.READY_MIMIC_REWARD_SUM_COUNTER] == pytest.approx(
        expected_ready_income
    )
    assert counters[SEMANTICS.SWING_MIMIC_REWARD_SUM_COUNTER] == pytest.approx(
        expected_swing_income
    )
    assert (
        counters[SEMANTICS.READY_MIMIC_ELIGIBLE_COUNTER]
        + counters[SEMANTICS.SWING_MIMIC_ELIGIBLE_COUNTER]
        == counters[SEMANTICS.reward_group_eligible_counter("mimic")]
        == SEMANTICS.PRELONG_ROLLOUT_SAMPLES
    )
    assert (
        counters[SEMANTICS.READY_MIMIC_REWARD_SUM_COUNTER]
        + counters[SEMANTICS.SWING_MIMIC_REWARD_SUM_COUNTER]
        == pytest.approx(counters[SEMANTICS.reward_group_sum_counter("mimic")])
    )
    split = prepared.record["mimic_task_phase_split"]
    assert split["task_invalid_ready"]["eligible_denominator"] == ready_count
    assert split["task_valid_swing"]["eligible_denominator"] == swing_count
    ledger.acknowledge_update(prepared)


def test_step_tokens_reject_double_and_foreign_consumption():
    env, _command, ledger = _runtime()
    token = _runtime_step(env, ledger)
    with pytest.raises(SEMANTICS.PrelongSemanticLedgerError, match="stale"):
        ledger.observe_after_environment_step(token)

    current = ledger.begin_environment_step()
    with pytest.raises(SEMANTICS.PrelongSemanticLedgerError, match="stale"):
        ledger.observe_after_environment_step(token)
    ledger.abort_environment_step(current)


def test_dt_weighted_group_income_excludes_safety():
    env, _command, ledger = _runtime()
    _runtime_step(
        env,
        ledger,
        contributions={
            "upright_exp": 2.0,
            "death_penalty": 3.0,
        },
    )
    _finish_zero_steps(env, ledger, already_observed=1)
    prepared = ledger.prepare_update(ppo_update=0)

    expected = 2.0 * 0.02 * 4096
    assert prepared.counters[
        SEMANTICS.reward_group_sum_counter("balance")
    ] == pytest.approx(expected)
    assert all(
        prepared.counters[SEMANTICS.reward_group_sum_counter(group)] == 0.0
        for group in ("mimic", "strike", "target", "outcome")
    )
    ledger.acknowledge_update(prepared)


def test_diagnostic_probe_cannot_silently_change_reward_income():
    env, _command, ledger = _runtime()
    _runtime_step(
        env,
        ledger,
        contributions={"base_decel_activation_probe": 1.0},
    )
    _finish_zero_steps(env, ledger, already_observed=1)
    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="diagnostic_probe_reward_cache_nonzero",
    ):
        ledger.prepare_update(ppo_update=0)


def test_task_income_is_exact_zero_against_frozen_invalid_snapshot():
    env, command, ledger = _runtime()
    command._action_ball_task_valid.zero_()
    _runtime_step(
        env,
        ledger,
        contributions={"strike_capture_bonus": 1.0},
        in_step_mutation=lambda: command._action_ball_task_valid.fill_(True),
    )
    _finish_zero_steps(env, ledger, already_observed=1)
    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="outside_pre_step_eligibility",
    ):
        ledger.prepare_update(ppo_update=0)


def test_reward_buffer_cache_closure_fails_closed():
    env, _command, ledger = _runtime()
    _runtime_step(env, ledger, reward_buf_offset=1.0)
    _finish_zero_steps(env, ledger, already_observed=1)
    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="reward_buffer_closure",
    ):
        ledger.prepare_update(ppo_update=0)


def test_closed_contact_delta_is_separate_from_same_window_exact_ticks():
    env, command, ledger = _runtime()
    _finish_zero_steps(env, ledger, already_observed=0)
    command.closed = 1
    command.contacts = 1
    prepared = ledger.prepare_update(ppo_update=0)

    assert prepared.counters[SEMANTICS.EXACT_STRIKE_TIMING_COUNTER] == 0
    assert prepared.counters[SEMANTICS.ELIGIBLE_CLOSED_SWING_COUNTER] == 1
    assert prepared.counters[SEMANTICS.ACTUAL_CONTACT_COUNTER] == 1
    assert prepared.record["strike_timing"]["exact_strike_tick_denominator"] == 0
    assert prepared.record["hit"]["eligible_closed_swing_count"] == 1
    ledger.acknowledge_update(prepared)


def test_post_optimizer_boundary_closure_is_not_charged_to_next_rollout():
    env, command, ledger = _runtime()
    _finish_zero_steps(env, ledger, already_observed=0)
    first = ledger.prepare_update(ppo_update=0)
    assert first.counters[SEMANTICS.ELIGIBLE_CLOSED_SWING_COUNTER] == 0

    command.closed = 1
    command.contacts = 1
    marker = ledger.marker_line_for(first)
    assert marker == ledger.acknowledge_update(first)
    with pytest.raises(SEMANTICS.PrelongSemanticLedgerError, match="stale"):
        ledger.marker_line_for(first)

    _finish_zero_steps(env, ledger, already_observed=0)
    second = ledger.prepare_update(ppo_update=1)
    assert second.counters[SEMANTICS.ELIGIBLE_CLOSED_SWING_COUNTER] == 0
    assert second.counters[SEMANTICS.ACTUAL_CONTACT_COUNTER] == 0
    ledger.acknowledge_update(second)


def test_fallible_ack_validation_precedes_any_marker_emission():
    env, _command, ledger = _runtime()
    _finish_zero_steps(env, ledger, already_observed=0)
    prepared = ledger.prepare_update(ppo_update=0)
    prepared.marker_line += "tampered"
    emitted = []

    with pytest.raises(
        SEMANTICS.PrelongSemanticLedgerError,
        match="mutated before acknowledgement",
    ):
        acknowledgement = ledger.prepare_acknowledgement(prepared)
        emitted.append(acknowledgement.marker_line)
        acknowledgement.consume()
    assert emitted == []


def test_runner_wires_one_pre_step_token_and_one_canonical_marker():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    wrapper = source.index("def step_with_reward_activation")
    begin = source.index("prelong_semantics_ledger.begin_environment_step()", wrapper)
    env_step = source.index("result = original_env_step", begin)
    observe = source.index(
        "prelong_semantics_ledger.observe_after_environment_step", env_step
    )
    assert begin < env_step < observe
    assert source.count("print(prelong_marker_line, flush=True)") == 1

    update = source.index("def update_with_rollout_boundary")
    joint_commit = source.index("self._commit_diagnostic_joint_safety_update", update)
    boundary_service = source.index(
        "self._service_action_ball_frozen_evaluation", joint_commit
    )
    ack_prepare = source.index(
        "prelong_semantics_ledger.prepare_acknowledgement", boundary_service
    )
    marker_read = source.index("prelong_marker_line = prelong_ack.marker_line", ack_prepare)
    marker_print = source.index("print(prelong_marker_line, flush=True)", marker_read)
    marker_consume = source.index("acknowledged_line = prelong_ack.consume()", marker_print)
    assert (
        joint_commit
        < boundary_service
        < ack_prepare
        < marker_read
        < marker_print
        < marker_consume
    )
    assert "prelong_expected_recipe_sha256 = parse_prelong_runtime_request" in source
    assert "if prelong_expected_recipe_sha256 is not None:" in source
