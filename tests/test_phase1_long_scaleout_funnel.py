from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs" / "phase1_long_scaleout_funnel_20260715.yaml"


def _load():
    data = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    return data, data["jobs"]


def _delta(job):
    return dict(item.split("=", 1) for item in job["recipe"]["delta"])


def test_six_long_rows_are_pod2_only_single_seed_and_round_robin():
    data, jobs = _load()
    assert data["dispatch_pods"] == ["pod2"]
    assert data["launch_authorized"] is True
    assert data["strict_full_scene_probe_evidence"]["terminal_status"] == "passed"
    assert len(jobs) == 7
    assert jobs[2]["id"] == "p1_long_v2_only_seed3"
    assert jobs[2]["status"] == "rejected"
    assert jobs[-1]["id"] == "p1_long_v2_only_seed3_retry_v2"
    assert jobs[-1]["status"] == "ready"
    assert all(job["status"] == "ready" for index, job in enumerate(jobs) if index != 2)
    assert {job["seed"] for job in jobs} == {3}
    assert {job["budget"]["num_envs"] for job in jobs} == {4096}
    assert {job["budget"]["max_iterations"] for job in jobs} == {10001}
    assert {tuple(job["milestones"]) for job in jobs} == {
        (200, 500, 1000, 2000, 3000, 6000, 10000)
    }
    assert [job["resource"]["required_slot"] for job in jobs] == [
        "pod2/gpu0",
        "pod2/gpu1",
        "pod2/gpu0",
        "pod2/gpu1",
        "pod2/gpu0",
        "pod2/gpu1",
        "pod2/gpu0",
    ]
    assert {job["source"]["commit"] for job in jobs} == {
        "2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e"
    }
    assert all(job["runtime_binding"] is True for job in jobs)


def test_rows_complete_two_factor_and_two_dose_curves_without_second_seed():
    _, jobs = _load()
    deltas = {job["id"]: _delta(job) for job in jobs}
    common = {
        "task.rewards.base_decel_weight": "0.0",
        "task.motion.post_swing_start_prob": "0.0",
        "task.rewards.racket_face_conditional_guidance_weight": "0.0",
        "task.rewards.joint_velocity_limit_hinge_margin": "0.85",
    }
    for delta in deltas.values():
        for key, value in common.items():
            actual_key = f"++{key}" if key == "task.rewards.racket_face_conditional_guidance_weight" else key
            assert delta[actual_key] == value

    assert (
        deltas["p1_long_v1_only_seed3"]["++task.rewards.free_wrist_vel_mimic"],
        deltas["p1_long_v1_only_seed3"]["++task.rewards.motion_scale_in_window"],
    ) == ("true", "1.0")
    assert (
        deltas["p1_long_v2_only_seed3"]["++task.rewards.free_wrist_vel_mimic"],
        deltas["p1_long_v2_only_seed3"]["++task.rewards.motion_scale_in_window"],
    ) == ("false", "0.25")
    assert deltas["p1_long_v2_only_seed3_retry_v2"] == deltas[
        "p1_long_v2_only_seed3"
    ]
    assert deltas["p1_long_qdot_w1_seed3"][
        "task.rewards.joint_velocity_limit_hinge_weight"
    ] == "-1.0"
    assert deltas["p1_long_qdot_w2p5_seed3"][
        "task.rewards.joint_velocity_limit_hinge_weight"
    ] == "-2.5"
    assert deltas["p1_long_foot_orientation_w0_seed3"][
        "task.rewards.foot_orientation_weight"
    ] == "0.0"
    assert deltas["p1_long_foot_orientation_w0p6_seed3"][
        "task.rewards.foot_orientation_weight"
    ] == "-0.6"


def test_early_checkpoints_cannot_reject_sparse_hit_learning():
    data, _ = _load()
    contract = data["decision_contract"]
    assert contract["checkpoints_200_500_1000"] == (
        "structural_activation_and_direction_only"
    )
    assert contract["sparse_hit_behavior_may_stop_before_minimum_eligible_events"] is False
    assert contract["second_seed_authorized"] is False
    assert contract["promotion_authorized"] is False
