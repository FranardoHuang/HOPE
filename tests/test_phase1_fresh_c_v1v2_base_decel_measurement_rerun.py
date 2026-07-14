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


def test_replacement_pair_is_pod2_only_and_plans_exact_hard_slots_after_probe():
    queue = Q.load_queue(QUEUE_PATH)

    assert queue["launch_authorized"] is True
    assert queue["preregistration_status"] == (
        "exact_inference_fix_terminal_full_scene_probe_passed"
    )
    assert queue["dispatch_pods"] == ["pod2"]
    assert {slot.pod for slot in Q.slots(queue)} == {"pod2"}
    assert [job["status"] for job in queue["jobs"]] == ["ready", "ready"]
    assert all(job["blocker"] is None for job in queue["jobs"])
    assert [
        (row["job_id"], row["resource"])
        for row in Q.cmd_plan(queue, live=False)["assignments"]
    ] == [
        ("fresh_c_v1v2_base_decel_measurement_control_v4", "pod2/gpu1"),
        ("fresh_c_v1v2_base_decel_measurement_w1_v4", "pod2/gpu2"),
    ]


def test_successor_source_and_exact_terminal_probe_evidence_are_bound():
    queue = Q.load_queue(QUEUE_PATH)
    closure = queue["instrumentation_closure"]

    assert closure["telemetry_reference_source_commit"] == (
        "312669c7bd61f8fc8f5ea99c8e94cfc3ffae9b94"
    )
    assert closure["required_main_hardening_commit"] == (
        "f00c49779174fdf43f861a29ceda0f985be04f31"
    )
    assert closure["telemetry_reference_contains_required_main_hardening"] is False
    assert closure["successor_source_commit"] == (
        "2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e"
    )
    assert closure["failed_runtime_source_commit"] == (
        "0f3900a612863faf326dca6ad3e8d38bfe8df3c9"
    )
    assert closure["inference_mode_counter_reset_fix_in_successor"] is True
    assert closure["hard_slot_scheduler_commit"] == (
        "8b0a08414aef390d3b45664c2cd3746e87453fff"
    )
    assert closure["successor_contains_required_main_hardening"] is True
    assert closure["source_rebind_required_before_launch"] is False
    assert closure["strict_full_scene_terminal_probe_required_after_rebind"] is True
    assert closure["strict_full_scene_terminal_probe_satisfied"] is True

    evidence = queue["strict_full_scene_probe_evidence"]
    assert evidence["result_file_sha256"] == (
        "4b12854c5deca075ddf886fea3c5806aa0838b1d2bc9d3739e2fa13cd1840b27"
    )
    assert evidence["result_content_sha256"] == (
        "4cbc9fc0bf7a5e5bdc5dfaa06386463e325dab141945344d0b0064b1b55fb083"
    )
    assert evidence["claim_content_sha256"] == (
        "52298bf11cb16e11cd67a198ca713c542423d576d1052e4178e42868b9bcfb9f"
    )
    assert evidence["checkpoint_sha256"] == (
        "68d9809bdc29c041a21fe775006adeafeacf6ff0a1d7b86cbbfe7bd042598713"
    )
    assert evidence["hard_contract_sha256"] == (
        "451cda47227f8e78e4f3dcae3cbf22d7ddf88b4c5fbaf348e698963d8bc12291"
    )
    assert evidence["actual_num_envs"] == 4096
    assert evidence["physical_ball_enabled"] is True
    assert evidence["physical_scene_entities"] == [
        "pb_ball", "pb_table", "pb_table_visual"
    ]
    assert evidence["embedded_iteration"] == 1
    assert evidence["nonfinite_floating_elements"] == 0
    assert evidence["isolated_process_group_empty"] is True
    assert evidence["terminal_status"] == "passed"
    assert evidence["unlock_authorized"] is True

    for job in queue["jobs"]:
        assert job["source"]["checkout"] == (
            "/workspace/codexschema/nohope_p1_activation_successor_2c2d70d"
        )
        assert job["source"]["commit"] == (
            "2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e"
        )
        assert job["runtime_binding"] is True


def test_pair_is_hard_bound_away_from_yikang_gpu0():
    queue = Q.load_queue(QUEUE_PATH)
    control, treatment = queue["jobs"]

    assert control["resource"] == {
        "policy": "dispatch_gpu_round_robin",
        "required_slot": "pod2/gpu1",
    }
    assert treatment["resource"] == {
        "policy": "dispatch_gpu_round_robin",
        "required_slot": "pod2/gpu2",
    }
    assert all(job["resource"].get("preferred_slot") is None for job in queue["jobs"])


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


def test_successor_exposes_exact_v1_v2_and_raw_base_deceleration_metrics():
    queue = Q.load_queue(QUEUE_PATH)
    closure = queue["instrumentation_closure"]
    metrics = closure["successor_runtime_metrics"]

    assert metrics == {
        "v1_free_wrist_linear_velocity": {
            "eligible_denominator": "Live/motion/v1_velocity_mimic_eligible_sample_count",
            "exclusion_numerator": "Live/motion/v1_held_wrist_excluded_sample_count",
        },
        "v2_strike_window_imitation_scale": {
            "eligible_denominator": "Live/motion/v2_strike_window_eligible_imitation_sample_count",
            "scaled_window_numerator": "Live/motion/v2_quarter_scaled_strike_window_imitation_sample_count",
        },
        "base_deceleration": {
            "sampling_phase": "reward_manager_before_reset_and_command",
            "probe_weight": 1.0,
            "probe_return": "exact_zero_per_environment",
            "eligible_denominator": "Live/racket_target/base_decel_eligible_sample_count",
            "raw_kernel_sum": "Live/racket_target/base_decel_raw_kernel_sum",
            "raw_kernel_nonzero_numerator": "Live/racket_target/base_decel_raw_kernel_nonzero_sample_count",
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
        "telemetry_status_in_successor_source"
    ] == "instrumented_probe_pending"
    assert activation["v2_strike_window_imitation_scale"][
        "telemetry_status_in_successor_source"
    ] == "instrumented_probe_pending"
    assert activation["base_deceleration"][
        "telemetry_status_in_successor_source"
    ] == "instrumented_probe_pending"
    assert activation["v1_free_wrist_linear_velocity"][
        "exclusion_must_equal_eligible"
    ] is True
    assert activation["v2_strike_window_imitation_scale"][
        "scaled_window_must_equal_eligible"
    ] is True
    base = activation["base_deceleration"]
    assert base["denominator_definition"] == (
        "pre_strike_and_not_in_hold_environment_samples_observed_every_reward_manager_step_before_reset_and_command"
    )
    assert metrics["base_deceleration"]["sampling_phase"] == (
        "reward_manager_before_reset_and_command"
    )
    assert metrics["base_deceleration"]["probe_weight"] == 1.0
    assert metrics["base_deceleration"]["probe_return"] == (
        "exact_zero_per_environment"
    )
    assert base["raw_kernel_nonzero_must_not_exceed_eligible"] is True
    for arm in ("control", "treatment"):
        assert base[arm]["eligible_denominator_must_be_positive"] is True
        assert base[arm]["raw_kernel_sum_must_be_positive"] is True
        assert base[arm]["raw_kernel_nonzero_numerator_must_be_positive"] is True


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
    assert "control_seed3_v3_20260715" not in raw
    assert "w1_seed3_v3_20260715" not in raw
    assert len({job["run_name"] for job in queue["jobs"]}) == 2
    assert len({job["run_dir"] for job in queue["jobs"]}) == 2
    for job in queue["jobs"]:
        assert job["run_name"].endswith("_20260715")
        assert job["run_dir"].startswith(
            "/workspace/codexschema/phase1_fresh_c_v1v2_base_decel_measurement_rerun_20260715/runs/"
        )
        assert job["status"] == "ready"


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
