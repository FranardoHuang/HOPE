from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = (
    ROOT
    / "configs"
    / "phase1_fresh_c_v1v2_decel_interaction_queue_20260714.yaml"
)
ACTIVE_QUEUE_PATH = ROOT / "configs" / "phase1_fresh_c_mechanism_queue_20260714.yaml"
SCRIPT = ROOT / "scripts" / "run_lean_training_queue.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v1v2_decel_queue_under_test", SCRIPT)
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


def _write(tmp_path: Path, queue: dict) -> Path:
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(queue, sort_keys=False), encoding="utf-8")
    return path


def test_preregistered_pair_is_pod2_only_blocked_and_unassigned():
    queue = Q.load_queue(QUEUE_PATH)

    assert queue["launch_authorized"] is False
    assert queue["preregistration_status"] == "blocked_pending_p1_source_binding"
    assert queue["dispatch_pods"] == ["pod2"]
    assert [job["id"] for job in queue["jobs"]] == [
        "fresh_c_v1v2_base_decel_matched_control",
        "fresh_c_v1v2_base_decel_w1",
    ]
    assert all(job["status"] == "blocked" for job in queue["jobs"])
    assert all("P1" in job["blocker"] for job in queue["jobs"])
    assert Q.cmd_plan(queue, live=False)["assignments"] == []


def test_pair_copies_the_complete_active_recipe_and_freezes_one_seed_budget():
    queue = Q.load_queue(QUEUE_PATH)
    active = Q.load_queue(ACTIVE_QUEUE_PATH)
    active_v1v2 = next(
        job
        for job in active["jobs"]
        if job["id"] == "fresh_c_v1_v2_combined_retry_v2"
    )

    for job in queue["jobs"]:
        assert job["recipe"]["base"] == active_v1v2["recipe"]["base"]
        assert job["seed"] == 3
        assert job["budget"] == {
            "num_envs": 4096,
            "max_iterations": 1001,
            "save_interval": 100,
        }
        assert job["milestones"] == [200, 500, 1000]
        assert job["resource"] == {"policy": "dispatch_gpu_round_robin"}


def test_base_deceleration_weight_is_the_only_matched_pair_delta():
    queue = Q.load_queue(QUEUE_PATH)
    control, treatment = queue["jobs"]

    for identity in ("motion", "bank", "exam", "source", "seed", "budget", "milestones", "resource"):
        assert control[identity] == treatment[identity]
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
    assert control_delta == {
        "task.rewards.free_wrist_vel_mimic": "true",
        "task.rewards.motion_scale_in_window": "0.25",
        "task.rewards.base_decel_weight": "0.0",
        "task.motion.post_swing_start_prob": "0.25",
        "task.rewards.joint_velocity_limit_hinge_weight": "0.0",
        "task.rewards.joint_velocity_limit_hinge_margin": "0.85",
        "task.rewards.racket_face_conditional_guidance_weight": "0.0",
    }


def test_source_placeholder_has_two_independent_fail_closed_guards(tmp_path):
    queue = Q.load_queue(QUEUE_PATH)
    for job in queue["jobs"]:
        assert job["source"] == {
            "checkout": "/workspace/BLOCKED_PLACEHOLDER_P1_RUNTIME_BINDING_SOURCE",
            "commit": "0" * 40,
        }

    zero_commit = copy.deepcopy(queue)
    zero_commit["launch_authorized"] = True
    zero_commit["jobs"][0]["status"] = "ready"
    zero_commit["jobs"][0]["blocker"] = None
    with pytest.raises(Q.QueueError, match="all-zero placeholder"):
        Q.load_queue(_write(tmp_path, zero_commit))

    placeholder_checkout = copy.deepcopy(queue)
    placeholder_checkout["launch_authorized"] = True
    placeholder_checkout["jobs"][0]["status"] = "ready"
    placeholder_checkout["jobs"][0]["blocker"] = None
    placeholder_checkout["jobs"][0]["source"]["commit"] = "1" * 40
    with pytest.raises(Q.QueueError, match="still a placeholder"):
        Q.load_queue(_write(tmp_path, placeholder_checkout))


def test_early_decision_requires_activation_balance_and_precision_without_promotion():
    queue = Q.load_queue(QUEUE_PATH)
    contract = queue["decision_contract"]

    assert contract["activation_required_before_comparison"] is True
    activation = contract["activation"]
    assert activation["v1_free_wrist_linear_velocity"]["control_and_treatment"] == {
        "setting": True,
        "eligible_denominator_must_be_positive": True,
        "exclusion_numerator_must_be_positive": True,
    }
    assert activation["v2_strike_window_imitation_scale"]["control_and_treatment"] == {
        "setting": 0.25,
        "eligible_denominator_must_be_positive": True,
        "scaled_window_numerator_must_be_positive": True,
    }
    assert activation["base_deceleration"]["control"]["weight"] == 0.0
    assert activation["base_deceleration"]["treatment"]["weight"] == 1.0

    at_500 = contract["milestone_rules"][500]
    assert at_500["base_speed_xy_prestrike_treatment_over_control_max"] == 0.90
    assert at_500["pre_fall_rate_treatment_minus_control_max"] == 0.02
    assert at_500["precision_pass_rate_treatment_minus_control_min"] == -0.05
    assert contract["precision_pass_rates"] == [
        "racket_position_pass",
        "racket_velocity_pass",
        "signed_racket_normal_pass",
        "strike_composite_pass",
    ]
    assert contract["second_seed_authorized"] is False
    assert contract["judge_authorized"] is False
    assert contract["promotion_authorized"] is False
