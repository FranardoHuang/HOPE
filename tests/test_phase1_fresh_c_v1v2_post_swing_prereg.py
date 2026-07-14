from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = (
    ROOT
    / "configs"
    / "phase1_fresh_c_v1v2_post_swing_interaction_queue_20260714.yaml"
)
SCRIPT = ROOT / "scripts" / "run_lean_training_queue.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "v1v2_post_swing_queue_under_test", SCRIPT
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


def test_pair_is_pod2_only_but_launch_blocked_until_instrumented_source_probe():
    queue = Q.load_queue(QUEUE_PATH)

    assert queue["launch_authorized"] is False
    assert queue["preregistration_status"] == (
        "activation_instrumentation_source_pinned_strict_full_scene_probe_pending"
    )
    assert queue["dispatch_pods"] == ["pod2"]
    assert {slot.pod for slot in Q.slots(queue)} == {"pod2"}
    assert [job["status"] for job in queue["jobs"]] == ["blocked", "blocked"]
    assert all("strict full-scene probe" in job["blocker"] for job in queue["jobs"])
    assert all(job["runtime_binding"] is True for job in queue["jobs"])

    plan = Q.cmd_plan(queue, live=False)
    assert plan["assignments"] == []
    activated = [dict(job, status="ready", blocker=None) for job in queue["jobs"]]
    queue["jobs"] = activated
    assert [(job["id"], slot.name) for job, slot in Q._assign(
        queue, {slot.name: 0 for slot in Q.slots(queue)}
    )] == [
        ("fresh_c_v1v2_post_swing_p025_control", "pod2/gpu1"),
        ("fresh_c_v1v2_post_swing_p050", "pod2/gpu2"),
    ]


def test_pair_freezes_same_inputs_budget_seed_and_exact_instrumented_source():
    queue = Q.load_queue(QUEUE_PATH)
    control, treatment = queue["jobs"]

    for identity in ("action", "motion", "bank", "exam", "source", "seed", "budget", "milestones"):
        assert control[identity] == treatment[identity]
    assert control["seed"] == 3
    assert control["budget"] == {
        "num_envs": 4096,
        "max_iterations": 1001,
        "save_interval": 100,
    }
    assert control["milestones"] == [200, 500, 1000]
    assert control["source"]["checkout"] == (
        "/workspace/codexschema/nohope_p1_post_swing_activation_3ced5a2"
    )
    assert control["source"]["commit"] == (
        "3ced5a218eab322ebc4ebea6c73ecf64ee47cc5e"
    )
    ignored = control["source"]["ignored_runtime_asset"]
    assert ignored == {
        "target_relative_path": "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
        "donor": {
            "checkout": "/workspace/codexschema/nohope",
            "commit": "6d93bcb16c422a2f42748c2dc99432559653480b",
            "relative_path": "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
        },
        "file_count": 46,
        "total_file_bytes": 15378264,
        "tree_content_sha256": "0137f59b1fe45e7d5f8fa731bedca905f5466bc98e8d1354081fe071d60426c6",
        "symlinks_forbidden": True,
        "target_must_be_gitignored": True,
    }
    assert control["motion"]["bindings"] == {
        "motion_file": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_forehand_v4rg_cal.npz",
        "motion_file_2": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_backhand_v4rg_cal.npz",
    }
    assert control["bank"]["train_path"].endswith(
        "s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz"
    )
    assert control["exam"]["family"] == "signed_face_rebound_k100_v1"


def test_post_swing_probability_is_the_only_matched_pair_delta():
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
    assert differences == {
        "task.motion.post_swing_start_prob": ("0.25", "0.50")
    }
    expected_common = {
        "task.rewards.free_wrist_vel_mimic": "true",
        "task.rewards.motion_scale_in_window": "0.25",
        "task.rewards.base_decel_weight": "0.0",
        "task.rewards.joint_velocity_limit_hinge_weight": "0.0",
        "task.rewards.joint_velocity_limit_hinge_margin": "0.85",
        "task.rewards.racket_face_conditional_guidance_weight": "0.0",
    }
    for key, value in expected_common.items():
        assert control_delta[key] == value
        assert treatment_delta[key] == value


def test_replay_activation_precedes_fixed_budget_recovery_and_precision_screen():
    queue = Q.load_queue(QUEUE_PATH)
    contract = queue["decision_contract"]

    assert contract["activation_required_before_comparison"] is True
    replay = contract["activation"]["post_swing_replay_start"]
    assert replay["denominator_definition"] == (
        "true_episode_resets_after_post_swing_buffer_min_fill"
    )
    assert replay["selected_numerator_definition"] == (
        "eligible_resets_selected_by_the_frozen_probability_draw"
    )
    assert replay["started_numerator_definition"] == (
        "selected_resets_after_root_and_joint_state_writes_return"
    )
    assert replay["tensorboard_metrics"] == {
        "buffer_not_ready": "Live/motion/post_swing_replay_buffer_not_ready_reset_count",
        "eligible_denominator": "Live/motion/post_swing_replay_eligible_reset_count",
        "random_not_selected": "Live/motion/post_swing_replay_random_not_selected_reset_count",
        "selected_numerator": "Live/motion/post_swing_replay_selected_reset_count",
        "started_numerator": "Live/motion/post_swing_replay_started_reset_count",
    }
    assert replay["selected_plus_random_not_selected_must_equal_eligible"] is True
    assert replay["started_must_equal_selected"] is True
    assert replay["control"] == {
        "probability": 0.25,
        "eligible_denominator_must_be_positive": True,
        "selected_numerator_must_be_positive": True,
        "started_numerator_must_be_positive": True,
        "realized_started_fraction_min": 0.20,
        "realized_started_fraction_max": 0.30,
    }
    assert replay["treatment"] == {
        "probability": 0.50,
        "eligible_denominator_must_be_positive": True,
        "selected_numerator_must_be_positive": True,
        "started_numerator_must_be_positive": True,
        "realized_started_fraction_min": 0.45,
        "realized_started_fraction_max": 0.55,
    }

    at_200 = contract["milestone_rules"][200]
    at_500 = contract["milestone_rules"][500]
    at_1000 = contract["milestone_rules"][1000]
    assert at_200["purpose"] == "direction_only_keep_training_if_active_and_finite"
    assert at_500 == {
        "purpose": "fixed_budget_recovery_balance_screen",
        "pre_strike_fall_rate_treatment_minus_control_max": -0.03,
        "post_strike_fall_rate_treatment_minus_control_max": 0.02,
        "swing_completion_rate_treatment_minus_control_min": 0.05,
        "precision_pass_rate_treatment_minus_control_min": -0.05,
        "stop_or_promote_allowed": False,
    }
    assert at_1000["purpose"] == "terminal_training_diagnostic"
    assert contract["precision_pass_rates"] == [
        "racket_position_pass",
        "racket_velocity_pass",
        "signed_racket_normal_pass",
        "strike_composite_pass",
    ]
    assert contract["non_compensable_safety_metrics"] == [
        "self_hit_rate",
        "unsafe_contact_rate",
    ]
    assert contract["safety_failure_cannot_be_offset_by_reward_or_precision_gain"] is True
    assert contract["second_seed_authorized"] is False
    assert contract["judge_authorized"] is False
    assert contract["promotion_authorized"] is False


def test_namespaces_are_new_concrete_and_contain_no_legacy_source_or_placeholder():
    queue = Q.load_queue(QUEUE_PATH)
    raw = QUEUE_PATH.read_text(encoding="utf-8")

    assert "placeholder" not in raw.lower()
    assert "/path/to/" not in raw.lower()
    assert "nohope_p1_077e70c" not in raw
    assert "nohope_p1_caeb9ad" not in raw
    assert "nohope_conditional_face_guidance_61007e9" not in raw
    assert "phase1_fresh_c_v1v2_decel_interaction_20260714" not in raw
    assert len({job["run_name"] for job in queue["jobs"]}) == 2
    assert len({job["run_dir"] for job in queue["jobs"]}) == 2
    for job in queue["jobs"]:
        assert job["id"].startswith("fresh_c_v1v2_post_swing_")
        assert job["run_name"].startswith("phase1_fresh_c_v1v2_post_swing_")
        assert job["run_dir"].startswith(
            "/workspace/codexschema/phase1_fresh_c_v1v2_post_swing_interaction_20260714/runs/"
        )
