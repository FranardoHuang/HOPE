from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_n1_vendor_probe_gate_receipt.py"
)
SPEC = importlib.util.spec_from_file_location("vendor_probe_gate_materializer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class _FakeBoolTensor:
    def __init__(self, value: bool):
        self._value = value

    def all(self):
        return self

    def any(self):
        return self

    def item(self):
        return self._value


class _FakeTensor:
    def __init__(self, values, shape):
        self.values = list(values)
        self.shape = tuple(shape)

    def numel(self):
        return len(self.values)

    def item(self):
        if len(self.values) != 1:
            raise ValueError("not scalar")
        return self.values[0]

    def is_floating_point(self):
        return True

    def __lt__(self, value):
        return _FakeBoolTensor(any(item < value for item in self.values))


class _FakeTorch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)

    @staticmethod
    def isfinite(value):
        return _FakeBoolTensor(
            all(math.isfinite(float(item)) for item in value.values)
        )


def _behavior_records(stage: str) -> list[dict]:
    updates = M.EXPECTED_STAGES[stage]["max_iterations"]
    result = []
    for update in range(updates):
        counters = {
            "physical_fall_count": 0,
            "pre_strike_physical_fall_count": 0,
            "post_strike_physical_fall_count": 0,
            "terminal_reset_count": 0,
            "timeout_reset_count": 0,
            "non_physical_terminal_reset_count": 0,
            "ready_nonfinite_value_count": 0,
            "strike_opportunity_count": 1,
            "swing_start_count": 1,
            "swing_outcome_count": 1,
            "reference_guard_union_count": 0,
            "reference_guard_reference_only_count": 0,
            "reference_guard_reference_and_hard_count": 0,
            "reference_guard_hard_without_snapshot_count": 0,
            M._ENTRY_COUNT: 1,
            M._ENTRY_NONFINITE: 0,
            **{key: 0 for key in M._ENTRY_BUCKETS},
            "termination_reason_joint_actual_forbidden_count": 0,
            "termination_reason_joint_qdes_forbidden_count": 0,
            "termination_reason_robot_hit_table_count": 0,
        }
        counters[M._ENTRY_BUCKETS[0]] = 1
        if stage == "probe":
            counters.update(
                {
                    "table_guard_first_hit_total_count": 0,
                    **{
                        f"table_guard_first_hit_category_{name}_count": 0
                        for name in M._TABLE_ATTRIBUTION_CATEGORIES
                    },
                    **{
                        f"table_guard_first_hit_phase_{name}_count": 0
                        for name in M._TABLE_ATTRIBUTION_PHASES
                    },
                }
            )
        result.append(
            {
                "event": "hope_exact_behavior_update",
                "schema_version": 1,
                "ppo_update": update,
                "counters": counters,
            }
        )
    return result


def _joint_records(stage: str, *, actual_hard: int = 0) -> list[dict]:
    updates = M.EXPECTED_STAGES[stage]["max_iterations"]
    return [
        {
            "event": "hope_joint_safety_diagnostic_compact_update",
            "schema_version": 1,
            "status": "diagnostic_compact_optimizer_committed_and_ledger_acknowledged",
            "ppo_update": update,
            "num_envs": M.EXPECTED_STAGES[stage]["num_envs"],
            "policy_step_count": M.ROLLOUT_STEPS_PER_UPDATE,
            "minimum_hard_gap_rad": 0.1,
            "counter_totals": {
                "actual_hard_edge_events": actual_hard if update == 0 else 0,
                "qdes_events": 0,
                "policy_crossing_events": 0,
                "substep_crossing_events": 0,
                "policy_steps": M.EXPECTED_STAGES[stage]["num_envs"] * 24,
                "complete_policy_steps": M.EXPECTED_STAGES[stage]["num_envs"] * 24,
            },
        }
        for update in range(updates)
    ]


def test_behavior_gate_uses_bounded_rates_and_requires_reachability() -> None:
    stage = "probe"
    result = M._validate_behavior(
        _behavior_records(stage),
        stage=stage,
        updates=M.EXPECTED_STAGES[stage]["max_iterations"],
        num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
    )
    assert result["reachability_and_failure_rates"][
        "within_telemetry_thresholds"
    ] is True
    assert result["reachability_and_failure_rates"]["telemetry_only"] is True
    assert result["strike_window_entry_conservation"]["matches"] is True
    assert result["table_guard_attribution"] == {
        "enabled": True,
        "first_hit_total_count": 0,
        "terminal_count": 0,
        "category_counts": {
            name: 0 for name in M._TABLE_ATTRIBUTION_CATEGORIES
        },
        "phase_counts": {name: 0 for name in M._TABLE_ATTRIBUTION_PHASES},
        "sparse_cell_total_count": 0,
        "conserves": True,
        "telemetry_only": True,
    }

    no_strike = _behavior_records(stage)
    for row in no_strike:
        row["counters"]["strike_opportunity_count"] = 0
    no_strike_summary = M._validate_behavior(
        no_strike,
        stage=stage,
        updates=M.EXPECTED_STAGES[stage]["max_iterations"],
        num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
    )
    assert no_strike_summary["reachability_and_failure_rates"][
        "within_telemetry_thresholds"
    ] is False

    drifted_attribution = _behavior_records(stage)
    drifted_attribution[0]["counters"][
        "table_guard_first_hit_total_count"
    ] = 1
    attribution_summary = M._validate_behavior(
        drifted_attribution,
        stage=stage,
        updates=M.EXPECTED_STAGES[stage]["max_iterations"],
        num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
    )
    assert attribution_summary["table_guard_attribution"]["conserves"] is False
    assert attribution_summary["table_guard_attribution"]["telemetry_only"] is True

    no_entry = _behavior_records(stage)
    for row in no_entry:
        row["counters"][M._ENTRY_COUNT] = 0
        row["counters"][M._ENTRY_BUCKETS[0]] = 0
    no_entry_summary = M._validate_behavior(
        no_entry,
        stage=stage,
        updates=M.EXPECTED_STAGES[stage]["max_iterations"],
        num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
    )
    assert no_entry_summary["strike_window_entry_conservation"] == {
        "entry_count": 0,
        "finite_bucket_total": 0,
        "nonfinite_count": 0,
        "matches": True,
        "telemetry_only": True,
    }

    nonconserving_histogram = _behavior_records(stage)
    nonconserving_histogram[0]["counters"][M._ENTRY_COUNT] = 2
    summary = M._validate_behavior(
        nonconserving_histogram,
        stage=stage,
        updates=M.EXPECTED_STAGES[stage]["max_iterations"],
        num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
    )
    assert summary["strike_window_entry_conservation"]["matches"] is False
    assert summary["strike_window_entry_conservation"]["telemetry_only"] is True


def test_old_failed_probe_actual_hard_cannot_mint_pass() -> None:
    stage = "probe"
    # The historical 89082 probe had 14,086 actual-hard events.  One event is
    # already enough to prove that a PASS receipt cannot be materialized.
    with pytest.raises(M.ReceiptRefused, match="zero gate"):
        M._validate_joint_safety(
            _joint_records(stage, actual_hard=14_086),
            [],
            updates=M.EXPECTED_STAGES[stage]["max_iterations"],
            num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
        )

    predicted_only = _joint_records(stage)
    predicted_only[0]["counter_totals"]["policy_crossing_events"] = 17
    predicted_only[0]["counter_totals"]["substep_crossing_events"] = 9
    accepted = M._validate_joint_safety(
        predicted_only,
        [],
        updates=M.EXPECTED_STAGES[stage]["max_iterations"],
        num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
    )
    assert accepted["aggregate_counter_totals"]["policy_crossing_events"] == 17


def test_integrated_probe_crosses_push_timer_lower_bound() -> None:
    duration = (
        M.EXPECTED_STAGES["probe"]["max_iterations"]
        * M.ROLLOUT_STEPS_PER_UPDATE
        * M.POLICY_DT_S
    )
    assert M.PUSH_INTERVAL_RANGE_S == (1.0, 3.0)
    assert duration == 2.4
    assert duration > M.PUSH_INTERVAL_RANGE_S[0]
    assert duration < M.PUSH_INTERVAL_RANGE_S[1]


def test_policy_bootstrap_receipt_requires_fresh_log_std_at_point_zero_two() -> None:
    row = {
        "event": "hope_action_ball_policy_bootstrap",
        "schema_version": 1,
        "applied_fresh": True,
        "noise_std_type": "log",
        "parameter_name": "log_std",
        "parameter_shape": [31],
        "parameter_count": 31,
        "configured_init_noise_std": 0.02,
        "realized_policy_std_min": 0.02,
        "realized_policy_std_mean": 0.02,
        "realized_policy_std_max": 0.02,
    }
    assert M._validate_policy_bootstrap([row]) == row

    scalar = dict(row, noise_std_type="scalar", parameter_name="std")
    with pytest.raises(M.ReceiptRefused, match="exact log_std"):
        M._validate_policy_bootstrap([scalar])

    wrong_sigma = dict(row, realized_policy_std_mean=0.021)
    with pytest.raises(M.ReceiptRefused, match="sigma 0.02"):
        M._validate_policy_bootstrap([wrong_sigma])


def _economy_records() -> list[dict]:
    names = [f"term_{index:02d}" for index in range(30)]
    samples = 4096 * 24
    stats = {
        "min": -2.0,
        "mean": 0.0,
        "p50": 0.0,
        "p95": 1.0,
        "p99": 1.5,
        "max": 2.0,
    }
    result = []
    for update in range(5):
        weighted = {name: 0.1 for name in names}
        result.append(
            {
                "event": "hope_action_ball_reward_ppo_economy_update",
                "schema_version": 1,
                "status": "PASS",
                "ppo_update": update,
                "gate": {
                    "num_envs": 4096,
                    "steps_per_env_per_update": 24,
                    "rollout_samples_per_update": samples,
                },
                "reward": {
                    "pre_advantage_reward_min_mean_p50_p95_p99_max": dict(stats),
                    "return_min_mean_p50_p95_p99_max": dict(stats),
                    "return_std": 2.0,
                    "explained_variance": 0.5,
                    "value_prediction_min_mean_p50_p95_p99_max": dict(stats),
                    "value_residual_min_mean_p50_p95_p99_max": dict(stats),
                    "per_term_raw_sum": {name: 1.0 for name in names},
                    "per_term_weighted_dt_sum": weighted,
                    "per_term_eligible_denominator": {
                        name: samples for name in names
                    },
                    "per_term_denominator_semantics": (
                        "all_rollout_environment_samples_including_gated_zero"
                    ),
                    "reward_manager_total_sum": math.fsum(weighted.values()),
                    "per_term_closure_error": {name: 0.0 for name in names},
                    "reward_manager_closure_max_abs_error": 0.0,
                    "recipe_sha256": "a" * 64,
                    "pre_advantage_reward_semantics": (
                        "ppo_storage_reward_after_timeout_bootstrap"
                    ),
                },
                "advantage": {
                    "pre_normalization_mean_std_min_max": {
                        "mean": 1.0,
                        "std": 2.0,
                        "min": -3.0,
                        "max": 5.0,
                    },
                    "post_normalization_mean_std_min_max": {
                        "mean": 0.0,
                        "std": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                    },
                    "post_normalization_finite": True,
                    "dtype_tolerance": 5.0e-5,
                    "normalization_population": "whole_rollout_98304_samples",
                },
                "ppo": {
                    "surrogate_loss": -0.1,
                    "value_loss": 0.2,
                    "entropy_mean": 0.3,
                    "loss_entropy_semantics": (
                        "arithmetic_mean_over_20_optimizer_minibatches"
                    ),
                    "approx_kl": 0.01,
                    "approx_kl_semantics": (
                        "final_policy_vs_rollout_policy_whole_rollout"
                    ),
                    "learning_rate": 1.0e-3 if update < 4 else 1.0e-5,
                    "clip_fraction": 0.1,
                    "clip_fraction_semantics": (
                        "final_policy_probability_ratio_outside_ppo_clip_whole_rollout"
                    ),
                },
                "gradient": {
                    "pre_clip_actor_mean_parameter_grad_norm": 2.0,
                    "pre_clip_critic_parameter_grad_norm": 3.0,
                    "pre_clip_std_parameter_grad_norm": 0.5,
                    "pre_clip_total_grad_norm": 4.0,
                    "post_clip_total_grad_norm": 1.0,
                    "pre_clip_actor_mean_parameter_grad_norm_distribution": {
                        "min": 1.0,
                        "mean": 2.0,
                        "max": 3.0,
                    },
                    "pre_clip_critic_parameter_grad_norm_distribution": {
                        "min": 2.0,
                        "mean": 3.0,
                        "max": 4.0,
                    },
                    "pre_clip_std_parameter_grad_norm_distribution": {
                        "min": 0.25,
                        "mean": 0.5,
                        "max": 0.75,
                    },
                    "pre_clip_total_grad_norm_distribution": {
                        "min": 3.0,
                        "mean": 4.0,
                        "max": 5.0,
                    },
                    "post_clip_total_grad_norm_distribution": {
                        "min": 0.75,
                        "mean": 1.0,
                        "max": 1.0,
                    },
                    "clip_factor_distribution": {
                        "min": 0.2,
                        "mean": 0.3,
                        "max": 1.0,
                    },
                    "max_grad_norm": 1.0,
                    "aggregation": "arithmetic_mean_over_optimizer_minibatches",
                    "optimizer_minibatch_count": 20,
                },
                "policy": {
                    "noise_std_type": "log",
                    "policy_std_min": 0.01,
                    "policy_std_mean": 0.02,
                    "policy_std_max": 0.03,
                },
                "checks": {
                    "all_required_fields_present": True,
                    "all_required_values_finite": True,
                    "reward_sum_closure": "PASS",
                    "post_advantage_zero_mean_unit_std": "PASS",
                    "noise_std_type_log": True,
                    "policy_std_strictly_positive": True,
                },
            }
        )
    return result


def test_reward_ppo_economy_requires_complete_finite_five_update_pass() -> None:
    records = _economy_records()
    result = M._validate_reward_ppo_economy(
        records,
        updates=5,
        num_envs=4096,
        expected_recipe_sha256="a" * 64,
    )
    assert result["status"] == "PASS"
    assert result["summary"]["active_reward_term_count"] == 30
    assert result["summary"]["learning_rate_floor_update_count"] == 1
    assert result["summary"]["all_required_fields_present_and_finite"] is True

    std_lr = [
        {
            "ppo_update": update,
            "policy_std_min": 0.01,
            "policy_std_mean": 0.02,
            "policy_std_max": 0.03,
            "learning_rate": 1.0e-3 if update < 4 else 1.0e-5,
        }
        for update in range(5)
    ]
    assert M._cross_validate_std_lr_and_economy(std_lr, result) == {
        "policy_std_exact": True,
        "learning_rate_exact": True,
    }
    mismatched = [dict(row) for row in std_lr]
    mismatched[0]["policy_std_mean"] = 0.021
    with pytest.raises(M.ReceiptRefused, match="markers disagree"):
        M._cross_validate_std_lr_and_economy(mismatched, result)

    missing = _economy_records()
    del missing[0]["gradient"]["pre_clip_std_parameter_grad_norm"]
    with pytest.raises(M.ReceiptRefused, match="gradient fields"):
        M._validate_reward_ppo_economy(
            missing,
            updates=5,
            num_envs=4096,
            expected_recipe_sha256="a" * 64,
        )

    nonfinite = _economy_records()
    nonfinite[0]["ppo"]["value_loss"] = float("nan")
    with pytest.raises(M.ReceiptRefused, match="must be finite"):
        M._validate_reward_ppo_economy(
            nonfinite,
            updates=5,
            num_envs=4096,
            expected_recipe_sha256="a" * 64,
        )

    unnormalized = _economy_records()
    unnormalized[0]["advantage"][
        "post_normalization_mean_std_min_max"
    ]["mean"] = 1.0e-3
    with pytest.raises(M.ReceiptRefused, match="zero-mean/unit-std"):
        M._validate_reward_ppo_economy(
            unnormalized,
            updates=5,
            num_envs=4096,
            expected_recipe_sha256="a" * 64,
        )

    all_floor = _economy_records()
    for row in all_floor:
        row["ppo"]["learning_rate"] = 1.0e-5
    with pytest.raises(M.ReceiptRefused, match="floor for all updates"):
        M._validate_reward_ppo_economy(
            all_floor,
            updates=5,
            num_envs=4096,
            expected_recipe_sha256="a" * 64,
        )

    hidden_clip_violation = _economy_records()
    hidden_clip_violation[0]["gradient"][
        "post_clip_total_grad_norm_distribution"
    ]["max"] = 1.01
    with pytest.raises(M.ReceiptRefused, match="gradient clip contract"):
        M._validate_reward_ppo_economy(
            hidden_clip_violation,
            updates=5,
            num_envs=4096,
            expected_recipe_sha256="a" * 64,
        )


def test_completion_marker_is_exact_once_and_binds_both_contracts() -> None:
    row = {
        "cleanup_complete": True,
        "completed_ppo_updates": 5,
        "event": "hope_training_complete",
        "num_envs": 4096,
        "schema_version": 1,
        "stage": "probe",
        "training_contract_sha256": "a" * 64,
        "training_launch_claim_sha256": "b" * 64,
        "vendor_runtime_training_contract_sha256": "c" * 64,
    }
    assert M._validate_completion(
        [row],
        stage="probe",
        num_envs=4096,
        updates=5,
        expected_claim_sha256="b" * 64,
        expected_hard_contract_sha256="a" * 64,
        expected_vendor_contract_sha256="c" * 64,
    ) == row
    with pytest.raises(M.ReceiptRefused, match="exactly one"):
        M._validate_completion(
            [],
            stage="probe",
            num_envs=4096,
            updates=5,
            expected_claim_sha256="b" * 64,
            expected_hard_contract_sha256="a" * 64,
            expected_vendor_contract_sha256="c" * 64,
        )
    tampered = dict(row)
    tampered["vendor_runtime_training_contract_sha256"] = "d" * 64
    with pytest.raises(M.ReceiptRefused, match="identity differs"):
        M._validate_completion(
            [tampered],
            stage="probe",
            num_envs=4096,
            updates=5,
            expected_claim_sha256="b" * 64,
            expected_hard_contract_sha256="a" * 64,
            expected_vendor_contract_sha256="c" * 64,
        )


def test_checkpoint_normalizer_requires_persisted_finite_state_and_count() -> None:
    state = {
        "_mean": _FakeTensor([0.0] * 194, (194,)),
        "_std": _FakeTensor([1.0] * 194, (194,)),
        "count": _FakeTensor([10.0], ()),
    }
    summary = M._normalizer_checkpoint_summary(
        state,
        role="actor",
        expected_features=194,
        torch_module=_FakeTorch,
    )
    assert summary["feature_count"] == 194
    assert summary["count"] == 10.0

    with pytest.raises(M.ReceiptRefused, match="lacks actor"):
        M._normalizer_checkpoint_summary(
            None,
            role="actor",
            expected_features=194,
            torch_module=_FakeTorch,
        )
    nonfinite = dict(state)
    nonfinite["_mean"] = _FakeTensor(
        [float("nan"), *([0.0] * 193)], (194,)
    )
    with pytest.raises(M.ReceiptRefused, match="non-finite"):
        M._normalizer_checkpoint_summary(
            nonfinite,
            role="actor",
            expected_features=194,
            torch_module=_FakeTorch,
        )
    zero_count = dict(state)
    zero_count["count"] = _FakeTensor([0.0], ())
    with pytest.raises(M.ReceiptRefused, match="finite and positive"):
        M._normalizer_checkpoint_summary(
            zero_count,
            role="actor",
            expected_features=194,
            torch_module=_FakeTorch,
        )
    negative_scale = dict(state)
    negative_scale["_std"] = _FakeTensor(
        [-1.0, *([1.0] * 193)], (194,)
    )
    with pytest.raises(M.ReceiptRefused, match="scale is negative"):
        M._normalizer_checkpoint_summary(
            negative_scale,
            role="actor",
            expected_features=194,
            torch_module=_FakeTorch,
        )


def test_checkpoint_normalizer_count_must_not_regress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = [10.0, 11.0, 9.0, 12.0, 13.0]
    for index in M.EXPECTED_CHECKPOINT_INDICES["probe"]:
        (tmp_path / f"model_{index}.pt").touch()

    def fake_summary(_path, *, expected_iteration, **_kwargs):
        normalizer = {"count": counts[expected_iteration]}
        return {
            "actor_normalizer": dict(normalizer),
            "critic_normalizer": dict(normalizer),
        }

    monkeypatch.setattr(M, "_checkpoint_summary", fake_summary)
    with pytest.raises(M.ReceiptRefused, match="count regressed"):
        M._inspect_checkpoints(
            tmp_path,
            stage="probe",
            expected_claim_sha256="a" * 64,
            expected_contract_sha256="b" * 64,
        )


def _push_records(stage: str) -> list[dict]:
    updates = M.EXPECTED_STAGES[stage]["max_iterations"]
    result = []
    for update in range(updates):
        active = update == updates - 1
        result.append(
            {
                "event": "hope_push_velocity_diagnostic_update",
                "schema_version": 1,
                "ppo_update": update,
                "counters": {
                    "event_call_count": 1 if active else 0,
                    "env_application_count": 7 if active else 0,
                    "delta_nonfinite_element_count": 0,
                    "axes": {
                        axis: {
                            "observed_delta_min": -0.05 if active else None,
                            "observed_delta_max": 0.05 if active else None,
                            "below_range_count": 0,
                            "above_range_count": 0,
                        }
                        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
                    },
                },
            }
        )
    return result


def test_runtime_push_counter_is_required_not_inferred_from_duration() -> None:
    records = _push_records("probe")
    result = M._validate_push_velocity(
        records, stage="probe", updates=5, num_envs=4096
    )
    assert result["aggregate"]["env_application_count"] == 7
    assert result["aggregate"]["six_axis_extrema_finite_and_in_range"] is True
    assert set(result["axis_extrema"]) == {
        "x", "y", "z", "roll", "pitch", "yaw"
    }

    missing = _push_records("probe")
    missing[-1]["counters"]["event_call_count"] = 0
    missing[-1]["counters"]["env_application_count"] = 0
    for values in missing[-1]["counters"]["axes"].values():
        values["observed_delta_min"] = None
        values["observed_delta_max"] = None
    with pytest.raises(M.ReceiptRefused, match="nonzero velocity push"):
        M._validate_push_velocity(
            missing, stage="probe", updates=5, num_envs=4096
        )

    one_sided = _push_records("probe")
    for values in one_sided[-1]["counters"]["axes"].values():
        values["observed_delta_min"] = 0.01
        values["observed_delta_max"] = 0.05
    accepted_one_sided = M._validate_push_velocity(
        one_sided, stage="probe", updates=5, num_envs=4096
    )
    assert accepted_one_sided["aggregate"]["event_call_count"] == 1

    out_of_range = _push_records("probe")
    out_of_range[-1]["counters"]["axes"]["x"]["above_range_count"] = 1
    with pytest.raises(M.ReceiptRefused, match="range breach"):
        M._validate_push_velocity(
            out_of_range, stage="probe", updates=5, num_envs=4096
        )

    lying_range_counters = _push_records("probe")
    lying_range_counters[-1]["counters"]["axes"]["x"][
        "observed_delta_max"
    ] = 0.251
    with pytest.raises(M.ReceiptRefused, match="raw extrema"):
        M._validate_push_velocity(
            lying_range_counters, stage="probe", updates=5, num_envs=4096
        )

    missing_active_extrema = _push_records("probe")
    missing_active_extrema[-1]["counters"]["axes"]["x"][
        "observed_delta_max"
    ] = None
    with pytest.raises(M.ReceiptRefused, match="omits observed extrema"):
        M._validate_push_velocity(
            missing_active_extrema, stage="probe", updates=5, num_envs=4096
        )


def test_scientific_argv_excludes_only_operational_axes() -> None:
    argv = [
        "/python",
        "/train.py",
        "task=HOPEPingPongActionBallA3VendorV1",
        M._V.STABLE_READY_PLANT_OVERRIDE,
        M._V.VENDOR_POLICY_NOISE_STD_OVERRIDE,
        f"task.actor_obs_contract={M.ACTOR_OBS_CONTRACT}",
        "seed=0",
        "num_envs=4096",
        "max_iterations=5",
        "algo.runner.save_interval=1",
        "device=cuda:0",
        "run_name=probe",
        M._V.TABLE_ATTRIBUTION_PROBE_OVERRIDE,
        "task.table_contact_attribution_diagnostic_extra=true",
        "task.table_contact_other_diagnostic=true",
    ]
    normalized, digest = M._scientific_argv(argv)
    assert "seed=0" in normalized
    assert not any(item.startswith("num_envs=") for item in normalized)
    assert M._V.TABLE_ATTRIBUTION_PROBE_OVERRIDE not in normalized
    assert "task.table_contact_attribution_diagnostic_extra=true" in normalized
    assert "task.table_contact_other_diagnostic=true" in normalized
    assert len(digest) == 64


def test_materialized_receipt_is_self_reference_free_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_commit = "b" * 40
    evidence_commit = gate_commit
    identity = {"action_id": "bh_loop_c", "scientific_argv_canonical_sha256": "c" * 64}
    monkeypatch.setattr(
        M,
        "_verify_gate_source",
        lambda *_args, **_kwargs: {"path": M.PRODUCER_SOURCE, "sha256": "d" * 64},
    )
    monkeypatch.setattr(
        M,
        "_stage_evidence",
        lambda *_args, expected_stage, **_kwargs: (
            {
                "stage": expected_stage,
                "source_commit": evidence_commit,
                "control_step_action_delay": {
                    "training_contract_sha256": "e" * 64
                },
                "runtime_abi": {"runtime": {"version": "same"}},
            },
            identity,
        ),
    )
    monkeypatch.setattr(
        M.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    receipt = M.materialize(
        gate_checkout=tmp_path,
        gate_source_commit=gate_commit,
        evidence_source_commit=evidence_commit,
        probe_namespace=tmp_path,
        probe_run_dir=tmp_path,
        receipt_repo_path="configs/n1_vendor_probe_gate_20260731/pass.json",
        long_spec_repo_path=(
            "configs/n1_vendor_launch_20260731/bh_loop_c.long.template.json"
        ),
    )
    assert receipt["verdict"] == "PASS"
    assert receipt["schema_version"] == 3
    assert receipt["kind"] == "n1_vendor_probe_gate_receipt_v3"
    assert receipt["producer"]["algorithm"] == "exact_integrated_probe_v3"
    assert set(receipt) == {
        "schema_version",
        "kind",
        "verdict",
        "producer",
        "evidence_source_commit",
        "scientific_identity",
        "stages",
        "acceptance",
        "successor_policy",
        "authorization",
        "content_sha256",
    }
    assert set(receipt["stages"]) == {"probe"}
    assert receipt["acceptance"] == {
        "integrated_probe_exact_pass": True,
        "finite_checkpoints": True,
        "normalizer_checkpoint_persistence": True,
        "runtime_abi_exact": True,
        "fresh_log_std_initialization_exact": True,
        "control_step_delay_exact": True,
        "positive_policy_std_and_finite_lr": True,
        "reward_ppo_economy_runtime_pass": True,
        "zero_actual_hard_edge": True,
        "zero_qdes_edge": True,
        "zero_nonfinite": True,
        "push_timer_lower_bound_crossed": True,
        "velocity_only_push_observed_nonzero": True,
        "velocity_push_six_axis_extrema_finite_and_in_range": True,
        "natural_training_completion": True,
    }
    assert "push_evidence" not in json.dumps(receipt)
    assert receipt["producer"]["self_reference_free"] is True
    assert "artifact_commit" not in json.dumps(receipt)
    assert receipt["content_sha256"] == M._canonical_sha(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )

    output = tmp_path / "receipt.json"
    M._write_no_clobber(output, receipt)
    with pytest.raises(M.ReceiptRefused, match="no-clobber"):
        M._write_no_clobber(output, receipt)
    assert output.read_bytes() == M._canonical_bytes(receipt) + b"\n"


def test_cli_accepts_only_one_integrated_probe_namespace_and_run_dir() -> None:
    option_strings = {
        option
        for action in M._parser()._actions
        for option in action.option_strings
    }
    assert "--probe-namespace" in option_strings
    assert "--probe-run-dir" in option_strings
    assert "--push-namespace" not in option_strings
    assert "--push-run-dir" not in option_strings


def test_content_tamper_changes_seal() -> None:
    unsigned = {"schema_version": 1, "kind": M.RECEIPT_KIND, "verdict": "PASS"}
    sealed = {**unsigned, "content_sha256": M._canonical_sha(unsigned)}
    tampered = copy.deepcopy(sealed)
    tampered["verdict"] = "FAIL"
    assert tampered["content_sha256"] != M._canonical_sha(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
