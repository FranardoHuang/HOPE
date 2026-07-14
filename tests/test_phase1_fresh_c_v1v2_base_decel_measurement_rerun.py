from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = (
    ROOT
    / "configs"
    / "phase1_fresh_c_v1v2_base_decel_measurement_rerun_queue_20260715.yaml"
)
PREVIOUS_QUEUE_PATH = (
    ROOT / "configs" / "phase1_fresh_c_v1v2_decel_interaction_queue_20260714.yaml"
)
SCRIPT = ROOT / "scripts" / "run_lean_training_queue.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "v1v2_base_decel_measurement_rerun_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = _load_module()


def _override_map(arguments: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for argument in arguments:
        raw_key, separator, value = argument.partition("=")
        assert separator == "="
        key = raw_key.removeprefix("++").removeprefix("+")
        assert key not in result
        result[key] = value
    return result


def test_replacement_pair_is_pod2_only_and_cannot_plan_a_launch():
    queue = Q.load_queue(QUEUE_PATH)

    assert queue["launch_authorized"] is False
    assert queue["preregistration_status"] == (
        "blocked_measurement_source_successor_required"
    )
    assert queue["dispatch_pods"] == ["pod2"]
    assert {slot.pod for slot in Q.slots(queue)} == {"pod2"}
    assert [job["status"] for job in queue["jobs"]] == ["blocked", "blocked"]
    assert all("312669c" in job["blocker"] for job in queue["jobs"])
    assert all("strict full-scene terminal probe" in job["blocker"] for job in queue["jobs"])
    assert Q.cmd_plan(queue, live=False)["assignments"] == []


def test_reference_source_and_required_main_hardening_are_exact_but_not_launch_sufficient():
    queue = Q.load_queue(QUEUE_PATH)
    closure = queue["instrumentation_closure"]

    assert closure["telemetry_reference_source_commit"] == (
        "312669c7bd61f8fc8f5ea99c8e94cfc3ffae9b94"
    )
    assert closure["required_main_hardening_commit"] == (
        "f00c49779174fdf43f861a29ceda0f985be04f31"
    )
    assert closure["telemetry_reference_contains_required_main_hardening"] is False
    assert closure["source_rebind_required_before_launch"] is True
    assert closure["strict_full_scene_terminal_probe_required_after_rebind"] is True

    for job in queue["jobs"]:
        assert job["source"]["checkout"] == (
            "/workspace/codexschema/nohope_p1_post_swing_activation_312669c"
        )
        assert job["source"]["commit"] == (
            "312669c7bd61f8fc8f5ea99c8e94cfc3ffae9b94"
        )
        assert job["runtime_binding"] is True


def test_five_post_swing_counts_are_frozen_with_arithmetic_closure():
    queue = Q.load_queue(QUEUE_PATH)
    expected = {
        "buffer_not_ready": "Live/motion/post_swing_replay_buffer_not_ready_reset_count",
        "eligible_denominator": "Live/motion/post_swing_replay_eligible_reset_count",
        "random_not_selected": "Live/motion/post_swing_replay_random_not_selected_reset_count",
        "selected_numerator": "Live/motion/post_swing_replay_selected_reset_count",
        "started_numerator": "Live/motion/post_swing_replay_started_reset_count",
    }

    assert queue["instrumentation_closure"]["available_post_swing_replay_metrics"] == expected
    replay = queue["decision_contract"]["activation"]["post_swing_replay_start"]
    assert replay["common_probability"] == 0.25
    assert replay["tensorboard_metrics"] == expected
    assert replay["selected_plus_random_not_selected_must_equal_eligible"] is True
    assert replay["started_must_equal_selected"] is True
    assert replay["eligible_denominator_must_be_positive"] is True
    assert replay["selected_numerator_must_be_positive"] is True
    assert replay["started_numerator_must_be_positive"] is True
    assert [
        replay["realized_started_fraction_min"],
        replay["realized_started_fraction_max"],
    ] == [0.20, 0.30]


def test_v1_v2_and_base_deceleration_activation_gaps_are_explicit_and_minimal():
    queue = Q.load_queue(QUEUE_PATH)
    closure = queue["instrumentation_closure"]
    missing = closure["missing_minimum_runtime_metrics"]

    assert missing == {
        "v1_free_wrist_linear_velocity": {
            "eligible_denominator": "Live/Reward/v1_free_wrist_vel_mimic_eligible_sample_count",
            "exclusion_numerator": "Live/Reward/v1_free_wrist_vel_mimic_excluded_sample_count",
        },
        "v2_strike_window_imitation_scale": {
            "eligible_denominator": "Live/Reward/v2_strike_window_imitation_eligible_sample_count",
            "scaled_window_numerator": "Live/Reward/v2_strike_window_imitation_scaled_sample_count",
        },
        "base_deceleration": {
            "eligible_denominator": "Live/Reward/base_decel_eligible_sample_count",
            "nonzero_reward_numerator": "Live/Reward/base_decel_nonzero_reward_sample_count",
        },
    }
    existing = closure["existing_base_deceleration_metrics"]
    assert existing["weighted_reward_mean"] == "Live/Reward/base_decel"
    assert existing["zero_masked_speed_mean"] == (
        "Live/racket_target/base_speed_xy_prestrike"
    )
    assert existing["sufficient_for_activation"] is False
    assert "eligible sample denominator" in existing["reason"]

    contract = queue["decision_contract"]
    assert contract["activation_must_precede_behavior_comparison"] is True
    assert contract["missing_activation_makes_run"] == (
        "invalid_instrumentation_blocked"
    )
    activation = contract["activation"]
    assert activation["v1_free_wrist_linear_velocity"][
        "telemetry_status_in_reference_source"
    ] == "missing"
    assert activation["v2_strike_window_imitation_scale"][
        "telemetry_status_in_reference_source"
    ] == "missing"
    assert activation["base_deceleration"][
        "telemetry_status_in_reference_source"
    ] == "missing_eligible_denominator"
    assert activation["v1_free_wrist_linear_velocity"][
        "exclusion_must_equal_eligible"
    ] is True
    assert activation["v2_strike_window_imitation_scale"][
        "scaled_window_must_equal_eligible"
    ] is True
    assert activation["base_deceleration"][
        "nonzero_reward_must_not_exceed_eligible"
    ] is True


def test_replacement_preserves_the_previous_science_recipe_but_not_its_namespace():
    queue = Q.load_queue(QUEUE_PATH)
    previous = Q.load_queue(PREVIOUS_QUEUE_PATH)
    control, treatment = queue["jobs"]
    previous_control = next(
        job
        for job in previous["jobs"]
        if job["id"] == "fresh_c_v1v2_base_decel_matched_control_retry_v2"
    )
    previous_treatment = next(
        job for job in previous["jobs"] if job["id"] == "fresh_c_v1v2_base_decel_w1"
    )

    for new, old in (
        (control, previous_control),
        (treatment, previous_treatment),
    ):
        for identity in ("action", "motion", "bank", "exam", "seed", "budget", "milestones"):
            assert new[identity] == old[identity]
        assert new["recipe"] == old["recipe"]
        assert new["run_name"] != old["run_name"]
        assert new["run_dir"] != old["run_dir"]

    assert control["seed"] == treatment["seed"] == 3
    assert control["budget"] == treatment["budget"] == {
        "num_envs": 4096,
        "max_iterations": 1001,
        "save_interval": 100,
    }
    assert control["milestones"] == treatment["milestones"] == [200, 500, 1000]


def test_base_deceleration_weight_remains_the_only_pair_delta():
    queue = Q.load_queue(QUEUE_PATH)
    control, treatment = queue["jobs"]

    assert control["recipe"]["base"] == treatment["recipe"]["base"]
    control_delta = _override_map(control["recipe"]["delta"])
    treatment_delta = _override_map(treatment["recipe"]["delta"])
    assert set(control_delta) == set(treatment_delta)
    differences = {
        key: (control_delta[key], treatment_delta[key])
        for key in control_delta
        if control_delta[key] != treatment_delta[key]
    }
    assert differences == {"task.rewards.base_decel_weight": ("0.0", "1.0")}
    assert control_delta["task.motion.post_swing_start_prob"] == "0.25"
    assert treatment_delta["task.motion.post_swing_start_prob"] == "0.25"
    assert control_delta["task.rewards.free_wrist_vel_mimic"] == "true"
    assert control_delta["task.rewards.motion_scale_in_window"] == "0.25"


def test_fresh_namespaces_have_no_old_attempt_or_placeholder():
    queue = Q.load_queue(QUEUE_PATH)
    raw = QUEUE_PATH.read_text(encoding="utf-8")

    assert "placeholder" not in raw.lower()
    assert "/path/to/" not in raw.lower()
    assert "phase1_fresh_c_v1v2_decel_interaction_20260714/runs/" not in raw
    assert "control_seed3_retry_v2_20260714" not in raw
    assert len({job["run_name"] for job in queue["jobs"]}) == 2
    assert len({job["run_dir"] for job in queue["jobs"]}) == 2
    for job in queue["jobs"]:
        assert job["run_name"].endswith("_20260715")
        assert job["run_dir"].startswith(
            "/workspace/codexschema/phase1_fresh_c_v1v2_base_decel_measurement_rerun_20260715/runs/"
        )
        assert job["status"] == "blocked"


def test_behavior_rules_cannot_run_before_activation_or_promote():
    queue = Q.load_queue(QUEUE_PATH)
    contract = queue["decision_contract"]

    assert contract["aggregation"] == "trailing_21_updates_at_each_milestone"
    assert contract["milestone_rules"][200]["purpose"].startswith(
        "direction_only_after_all_activation"
    )
    assert contract["milestone_rules"][500] == {
        "purpose": "causal_screen_after_all_activation_and_finite_checks",
        "base_speed_xy_prestrike_treatment_over_control_max": 0.90,
        "pre_fall_rate_treatment_minus_control_max": 0.02,
        "precision_pass_rate_treatment_minus_control_min": -0.05,
        "stop_or_promote_allowed": False,
    }
    assert contract["milestone_rules"][1000]["purpose"].startswith(
        "terminal_training_diagnostic_after_all_activation"
    )
    assert contract["second_seed_authorized"] is False
    assert contract["judge_authorized"] is False
    assert contract["promotion_authorized"] is False
