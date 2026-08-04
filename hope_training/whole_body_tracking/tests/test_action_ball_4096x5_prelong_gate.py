"""Tests for the shared fail-closed 4096x5 pre-long telemetry gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "action_ball_4096x5_prelong_gate.py"
)
SPEC = importlib.util.spec_from_file_location("action_ball_4096x5_prelong_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _checkpoint_acceptance():
    return {
        "checkpoint": {
            "filename_iteration": 5,
            "embedded_iteration": 5,
            "load_mode": "torch_weights_only",
            "all_tensors_finite": True,
            "tensor_groups": {
                name: {"tensor_count": 1, "element_count": 2}
                for name in (
                    "model",
                    "optimizer",
                    "actor_normalizer",
                    "critic_normalizer",
                )
            },
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
                reason: {phase: 0 for phase in GATE.PHYSICAL_FALL_PHASES}
                for reason in GATE.PHYSICAL_FALL_REASONS
            },
            "table_contact_by_phase": {
                phase: 0 for phase in GATE.PHYSICAL_FALL_PHASES
            },
            "task_wait_started_by_update": [12] * 5,
            "task_wait_started_count": 60,
            "task_reveal_reached_by_update": [10] * 5,
            "task_reveal_reached_count": 50,
        },
    }


def _economy(update):
    return {
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
            "per_term_eligible_denominator": {"motion": 98304, "task": 98304},
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


def _groups(update):
    return {
        "event": "hope_effective_reward_activation_by_action_update",
        "schema_version": 2,
        "ppo_update": update,
        "actions": [
            {
                "action_id": "Take_061_unit04_BH",
                "reward_groups": [
                    {
                        "group": "motion",
                        "eligibility": "reward_manager_evaluated_active_group_terms",
                        "eligible_sample_count": 98304,
                        "weighted_sum": 4.0,
                    },
                    {
                        "group": "task",
                        "eligibility": "reward_manager_evaluated_active_group_terms",
                        "eligible_sample_count": 98304,
                        "weighted_sum": 0.0,
                    },
                ],
            }
        ],
    }


def _log():
    lines = []
    for update in range(5):
        lines.extend(
            (
                GATE.ECONOMY_PREFIX
                + json.dumps(_economy(update), sort_keys=True, separators=(",", ":")),
                GATE.GROUP_PREFIX
                + json.dumps(_groups(update), sort_keys=True, separators=(",", ":")),
            )
        )
    return "\n".join(lines) + "\n"


def _semantic(
    update,
    *,
    profile=GATE.PROFILE_A211,
    contacts=0,
    closed=9,
    exact_strike_ticks=9,
    flight_denominator=0,
    invalid_samples=7,
):
    if profile == GATE.PROFILE_A211:
        strike_income = 0.0
        target_income = 2.0
        target_denominator = 3
    elif profile == GATE.PROFILE_C211:
        strike_income = 2.0
        target_income = 0.0
        target_denominator = 0
    else:
        raise AssertionError(profile)
    return {
        "event": GATE.SEMANTIC_EVENT,
        "schema_version": GATE.SEMANTIC_SCHEMA_VERSION,
        "profile": profile,
        "ppo_update": update,
        "window": {
            "num_envs": 4096,
            "rollout_steps_per_env": 24,
            "rollout_sample_count": 98304,
            "reset_boundary": "same once-per-PPO-update transaction",
        },
        "task_invalid": {
            "observed_sample_count": invalid_samples,
            "task_reward_weighted_sum": 0.0,
            "task_reward_eligible_denominator": 0,
        },
        "strike_timing": {
            "exact_strike_tick_denominator": exact_strike_ticks,
        },
        "hit": {
            "eligible_closed_swing_count": closed,
            "actual_contact_numerator": contacts,
        },
        "achieved_flight": {"eligible_denominator": flight_denominator},
        "reward_groups": [
            {
                "group": "balance",
                "weighted_sum": 4.0,
                "eligible_denominator": 98304,
                "eligibility_semantics": "all_rollout_samples",
            },
            {
                "group": "mimic",
                "weighted_sum": 4.0,
                "eligible_denominator": 98304,
                "eligibility_semantics": "phase_eligible_mimic_samples",
            },
            {
                "group": "strike",
                "weighted_sum": strike_income,
                "eligible_denominator": exact_strike_ticks,
                "eligibility_semantics": "exact_strike_timing_ticks",
            },
            {
                "group": "target",
                "weighted_sum": target_income,
                "eligible_denominator": target_denominator,
                "eligibility_semantics": "task_valid_contact_target_opportunities",
            },
            {
                "group": "outcome",
                "weighted_sum": 0.0,
                "eligible_denominator": flight_denominator,
                "eligibility_semantics": "eligible_achieved_flights",
            },
        ],
        "unknown_attribution_count": 0,
    }


def _semantics(**kwargs):
    return [
        _semantic(
            update,
            invalid_samples=(7 if update == 0 else 0),
            **kwargs,
        )
        for update in range(5)
    ]


def test_gate_accepts_exact_five_updates_and_preserves_zero_over_c():
    result = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=_checkpoint_acceptance(),
        semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
    )
    assert result["status"] == "PASS"
    assert result["diagnostic_unauthorized"] is True
    assert result["ppo_updates"] == 5
    assert result["survival_denominators"]["task_reveal_reached_count"] == 50
    assert (
        result["survival_denominators"]["task_active_observed_sample_count"]
        == 4096 * 24 * 5 - 7
    )
    assert result["survival_denominators"]["nominal_strike_reached_count"] == 45
    assert result["survival_denominators"]["eligible_closed_swing_count"] == 45
    aggregate = result["opportunity_semantics"]["aggregate"]
    assert aggregate["profile"] == GATE.PROFILE_A211
    assert aggregate["actual_contact_numerator"] == 0
    assert aggregate["outcome_opportunity_denominator"] == 0
    assert (
        aggregate["reward_groups"]["balance"]["eligible_denominator"]
        == 4096 * 24 * 5
    )
    assert (
        aggregate["reward_groups"]["mimic"]["eligible_denominator"]
        == 4096 * 24 * 5
    )
    assert result["opportunity_semantics"]["updates"][0]["hit"] == "0/9"
    assert (
        result["opportunity_semantics"]["updates"][0][
            "achieved_flight_eligible_denominator"
        ]
        == 0
    )


def test_c211_gate_requires_strike_signal_but_not_initial_contact_or_outcome():
    result = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=_checkpoint_acceptance(),
        semantic_updates=_semantics(
            profile=GATE.PROFILE_C211,
            contacts=0,
            closed=9,
            flight_denominator=0,
        ),
    )

    aggregate = result["opportunity_semantics"]["aggregate"]
    assert aggregate["profile"] == GATE.PROFILE_C211
    assert aggregate["actual_contact_numerator"] == 0
    assert aggregate["outcome_opportunity_denominator"] == 0
    assert aggregate["reward_groups"]["strike"] == {
        "weighted_sum": 10.0,
        "eligible_denominator": 45,
    }


def test_current_markers_alone_fail_closed_on_missing_semantic_producer():
    with pytest.raises(GATE.PreLongGateRefused, match="MISSING_PRODUCER"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=None,
        )


def test_gate_rejects_four_or_duplicate_economy_updates():
    lines = _log().splitlines()
    missing = "\n".join(
        line
        for line in lines
        if not (
            line.startswith(GATE.ECONOMY_PREFIX)
            and '"ppo_update":4' in line
        )
    )
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_economy_updates(missing)

    duplicated = _log() + GATE.ECONOMY_PREFIX + json.dumps(_economy(4)) + "\n"
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_economy_updates(duplicated)


@pytest.mark.parametrize(
    "marker_kind,field,value",
    (
        ("economy", "schema_version", True),
        ("economy", "ppo_update", False),
        ("groups", "schema_version", True),
        ("groups", "ppo_update", False),
        ("semantic", "schema_version", True),
        ("semantic", "ppo_update", False),
    ),
)
def test_gate_rejects_boolean_update_and_schema_version(
    marker_kind, field, value
):
    if marker_kind == "semantic":
        rows = _semantics()
        rows[0][field] = value
        with pytest.raises(GATE.PreLongGateRefused, match="contiguous"):
            GATE.validate_semantic_updates(rows)
        return

    lines = []
    for update in range(5):
        row = _economy(update) if marker_kind == "economy" else _groups(update)
        if update == 0:
            row[field] = value
        prefix = GATE.ECONOMY_PREFIX if marker_kind == "economy" else GATE.GROUP_PREFIX
        lines.append(prefix + json.dumps(row, separators=(",", ":")))
    validator = (
        GATE.validate_economy_updates
        if marker_kind == "economy"
        else GATE.validate_group_income_updates
    )
    with pytest.raises(GATE.PreLongGateRefused, match="contiguous"):
        validator("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    "section,key,value,match",
    (
        ("ppo", "approx_kl", float("nan"), "not finite JSON"),
        ("ppo", "clip_fraction", 1.1, "clip_fraction"),
        ("gradient", "pre_clip_total_grad_norm", float("inf"), "not finite JSON"),
        ("policy", "policy_std_min", 0.0, "policy std"),
    ),
)
def test_gate_rejects_nonfinite_or_invalid_optimizer_health(
    section, key, value, match
):
    rows = []
    for update in range(5):
        economy = _economy(update)
        if update == 2:
            economy[section][key] = value
        rows.append(
            GATE.ECONOMY_PREFIX
            + json.dumps(economy, allow_nan=True, separators=(",", ":"))
        )
    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_economy_updates("\n".join(rows))


def test_task_invalid_reward_or_denominator_must_be_zero():
    rows = _semantics()
    rows[3]["task_invalid"]["task_reward_eligible_denominator"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="leaked task reward or eligibility"):
        GATE.validate_semantic_updates(rows)


def test_contact_numerator_must_not_exceed_closed_swing_denominator():
    with pytest.raises(GATE.PreLongGateRefused, match="contacts <= eligible closed"):
        GATE.validate_semantic_updates(_semantics(contacts=10, closed=9))


def test_exact_strike_timing_and_closed_swing_are_cross_update_denominators():
    rows = _semantics()
    rows[0]["strike_timing"]["exact_strike_tick_denominator"] = 9
    rows[0]["reward_groups"][2]["eligible_denominator"] = 9
    rows[0]["hit"]["eligible_closed_swing_count"] = 0
    rows[1]["strike_timing"]["exact_strike_tick_denominator"] = 0
    rows[1]["reward_groups"][2]["eligible_denominator"] = 0
    rows[1]["hit"]["eligible_closed_swing_count"] = 9
    accepted = GATE.validate_semantic_updates(rows)
    assert accepted["updates"][0]["hit"] == "0/0"
    assert accepted["updates"][1]["hit"] == "0/9"


def test_strike_group_may_finite_filter_raw_exact_strike_ticks():
    rows = _semantics()
    rows[0]["strike_timing"]["exact_strike_tick_denominator"] = 9
    rows[0]["reward_groups"][2]["eligible_denominator"] = 8

    accepted = GATE.validate_semantic_updates(rows)

    assert accepted["updates"][0]["exact_strike_tick_denominator"] == 9


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda row: row["reward_groups"][2].__setitem__(
                "eligible_denominator", 10
            ),
            "strike-group denominator",
        ),
        (
            lambda row: row["reward_groups"][4].__setitem__(
                "eligible_denominator", 1
            ),
            "outcome-group denominator",
        ),
        (
            lambda row: row["window"].__setitem__("num_envs", 4095),
            "fixed rollout window",
        ),
        (
            lambda row: row["reward_groups"][4].__setitem__(
                "weighted_sum", 0.5
            ),
            "zero true eligibility",
        ),
    ),
)
def test_semantic_window_and_denominator_conservation_are_fail_closed(
    mutation, match
):
    rows = _semantics()
    mutation(rows[2])
    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_semantic_updates(rows)


def test_semantic_markers_require_exactly_five_contiguous_updates():
    rows = _semantics()
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_semantic_updates(rows[:-1])
    duplicate = list(rows) + [dict(rows[-1])]
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_semantic_updates(duplicate)


def test_task_invalid_exercised_over_five_updates_not_every_update():
    rows = _semantics()
    assert rows[1]["task_invalid"]["observed_sample_count"] == 0
    accepted = GATE.validate_semantic_updates(rows)
    assert accepted["aggregate"]["task_invalid_observed_sample_count"] == 7
    for row in rows:
        row["task_invalid"]["observed_sample_count"] = 0
    with pytest.raises(GATE.PreLongGateRefused, match="did not exercise task_valid=0"):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize("group", ("balance", "mimic"))
def test_every_update_requires_full_balance_and_mimic_denominator(group):
    rows = _semantics()
    group_index = 0 if group == "balance" else 1
    rows[2]["reward_groups"][group_index]["eligible_denominator"] -= 1

    with pytest.raises(
        GATE.PreLongGateRefused,
        match=rf"{group} denominator must equal 98304",
    ):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize(
    "profile,group_index,match",
    (
        (GATE.PROFILE_A211, 3, "A211 five-update target"),
        (GATE.PROFILE_C211, 2, "C211 five-update strike"),
    ),
)
def test_profile_specific_learnability_signal_must_be_positive(
    profile, group_index, match
):
    rows = _semantics(profile=profile)
    for row in rows:
        row["reward_groups"][group_index]["weighted_sum"] = 0.0

    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize(
    "profile,group_index,match",
    (
        (GATE.PROFILE_A211, 3, "A211 five-update target"),
        (GATE.PROFILE_C211, 2, "C211 five-update strike"),
    ),
)
def test_profile_specific_learnability_denominator_must_be_positive(
    profile, group_index, match
):
    rows = _semantics(profile=profile)
    for row in rows:
        row["reward_groups"][group_index]["weighted_sum"] = 0.0
        row["reward_groups"][group_index]["eligible_denominator"] = 0

    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_semantic_updates(rows)


def test_all_zero_learning_signal_is_rejected():
    rows = _semantics()
    for row in rows:
        for group in row["reward_groups"]:
            group["weighted_sum"] = 0.0

    with pytest.raises(
        GATE.PreLongGateRefused,
        match="aggregate balance income must be nonzero",
    ):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize(
    "group_index,match",
    (
        (0, "aggregate balance income must be nonzero"),
        (1, "aggregate mimic income must be positive"),
    ),
)
def test_aggregate_balance_and_mimic_income_health_is_required(group_index, match):
    rows = _semantics()
    for row in rows:
        row["reward_groups"][group_index]["weighted_sum"] = 0.0

    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_semantic_updates(rows)


def test_unknown_attribution_and_terminal_safety_are_zero_tolerance():
    rows = _semantics()
    rows[0]["unknown_attribution_count"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="unknown attribution is nonzero"):
        GATE.validate_semantic_updates(rows)

    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"]["joint_qdes_forbidden_terminal_count"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="implementation counters"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )


def test_fall_and_too_low_are_reported_but_not_zero_tolerance_in_finite_gate():
    checkpoint = _checkpoint_acceptance()
    safety = checkpoint["safety_counters"]
    safety["base_fell_tilt_terminal_count"] = 3
    safety["base_too_low_terminal_count"] = 2
    safety["physical_fall_by_reason_phase"] = {
        "base_fell_tilt": {
            "hidden_wait": 1,
            "revealed_pre_strike": 2,
            "post_strike": 0,
        },
        "base_too_low": {
            "hidden_wait": 0,
            "revealed_pre_strike": 1,
            "post_strike": 1,
        },
    }

    accepted = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=checkpoint,
        semantic_updates=_semantics(),
    )

    balance = accepted["safety"]["balance_termination_counts"]
    assert balance["by_reason"] == {"base_fell_tilt": 3, "base_too_low": 2}
    assert "unvalidated_numeric_cutoff" in accepted["safety"][
        "finite_balance_termination_policy"
    ]
    behavior = accepted["survival_denominators"]["behavioral_terminations"]
    assert behavior["base_fell_tilt"]["phase_exposure_denominators"] == {
        "hidden_wait": 60,
        "revealed_pre_strike": 50,
        "post_strike": 45,
    }
    assert behavior["base_fell_tilt"]["phase_rates"] == {
        "hidden_wait": 1 / 60,
        "revealed_pre_strike": 2 / 50,
        "post_strike": 0.0,
    }
    assert behavior["base_fell_tilt"]["acceptance_threshold"] is None


@pytest.mark.parametrize(
    "counter",
    (
        "actual_hard_edge_event_count",
        "actual_hard_terminal_count",
        "joint_qdes_forbidden_terminal_count",
        "joint_actual_forbidden_terminal_count",
        "strict_hard_termination_count",
        "nonfinite_count",
    ),
)
def test_every_strict_safety_counter_remains_zero_tolerance(counter):
    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"][counter] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="implementation counters"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )


def test_reason_by_phase_and_reveal_denominators_fail_closed():
    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"]["base_fell_tilt_terminal_count"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="do not conserve"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )


def test_robot_hit_table_is_behavioral_phase_evidence_not_finite_strict_zero():
    checkpoint = _checkpoint_acceptance()
    safety = checkpoint["safety_counters"]
    safety["table_contact_count"] = 3
    safety["table_contact_by_phase"] = {
        "hidden_wait": 1,
        "revealed_pre_strike": 2,
        "post_strike": 0,
    }

    accepted = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=checkpoint,
        semantic_updates=_semantics(),
    )

    table = accepted["survival_denominators"]["robot_hit_table"]
    assert table["total_count"] == 3
    assert table["phase_exposure_denominators"] == {
        "hidden_wait": 60,
        "revealed_pre_strike": 50,
        "post_strike": 45,
    }
    assert table["acceptance_threshold"] is None


def test_robot_hit_table_phase_counts_must_conserve():
    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"]["table_contact_count"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="robot_hit_table.*conserve"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )

    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"]["task_reveal_reached_by_update"][2] = 0
    checkpoint["safety_counters"]["task_reveal_reached_count"] = 40
    with pytest.raises(GATE.PreLongGateRefused, match="every finite update"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )


@pytest.mark.parametrize("milestone", ("exact", "closed"))
def test_each_update_requires_nominal_strike_and_closed_swing_survival(milestone):
    rows = _semantics()
    if milestone == "exact":
        rows[2]["strike_timing"]["exact_strike_tick_denominator"] = 0
        rows[2]["reward_groups"][2]["eligible_denominator"] = 0
    else:
        rows[2]["hit"]["eligible_closed_swing_count"] = 0
    with pytest.raises(GATE.PreLongGateRefused, match="finite survival update 2"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=rows,
        )


def test_each_update_requires_task_active_samples():
    rows = _semantics()
    rows[2]["task_invalid"]["observed_sample_count"] = 4096 * 24
    with pytest.raises(GATE.PreLongGateRefused, match="no TASK_ACTIVE samples"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=rows,
        )


def test_checkpoint_audit_requires_all_finite_nonempty_state_groups():
    acceptance = _checkpoint_acceptance()
    del acceptance["checkpoint"]["tensor_groups"]["optimizer"]
    with pytest.raises(GATE.PreLongGateRefused, match="tensor-group coverage"):
        GATE.validate_checkpoint_audit(acceptance["checkpoint"])


def test_cli_reports_structured_blocker_when_semantic_input_is_absent(
    tmp_path, capsys
):
    log = tmp_path / "run.log"
    log.write_text(_log(), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(_checkpoint_acceptance()), encoding="utf-8")
    assert GATE.main(
        ["--run-log", str(log), "--checkpoint-acceptance", str(checkpoint)]
    ) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "BLOCKED"
    assert output["diagnostic_unauthorized"] is True
    assert output["reason"].startswith("MISSING_PRODUCER:")
