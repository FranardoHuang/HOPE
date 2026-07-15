from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs" / "phase1_long_no_replay_funnel_20260715.yaml"


def _jobs():
    data = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    return data, {job["id"]: job for job in data["jobs"]}


def test_long_funnel_is_three_single_seed_gpu2_rows():
    data, jobs = _jobs()
    assert data["dispatch_pods"] == ["pod2"]
    assert data["strict_full_scene_probe_evidence"]["terminal_status"] == "passed"
    assert len(jobs) == 4
    assert jobs["p1_long_no_replay_control_seed3"]["status"] == "rejected"
    assert {
        job_id for job_id, job in jobs.items() if job["status"] == "ready"
    } == {
        "p1_long_no_replay_qdot_w5_seed3",
        "p1_long_no_replay_v1v2_seed3",
        "p1_long_no_replay_control_seed3_retry_v2",
    }
    assert {job["seed"] for job in jobs.values()} == {3}
    assert {job["budget"]["max_iterations"] for job in jobs.values()} == {10001}
    assert {tuple(job["milestones"]) for job in jobs.values()} == {
        (200, 500, 1000, 2000, 3000, 6000, 10000)
    }
    assert {job["resource"]["required_slot"] for job in jobs.values()} == {
        "pod2/gpu2"
    }
    assert {job["source"]["commit"] for job in jobs.values()} == {
        "2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e"
    }
    assert {job["runtime_binding"] for job in jobs.values()} == {True}


def test_long_funnel_has_one_variable_per_treatment_and_no_replay():
    _, jobs = _jobs()
    deltas = {
        job_id: dict(item.split("=", 1) for item in job["recipe"]["delta"])
        for job_id, job in jobs.items()
    }
    assert {delta["task.motion.post_swing_start_prob"] for delta in deltas.values()} == {"0.0"}
    control = deltas["p1_long_no_replay_control_seed3"]
    qdot = deltas["p1_long_no_replay_qdot_w5_seed3"]
    v1v2 = deltas["p1_long_no_replay_v1v2_seed3"]
    assert {key for key in control if control[key] != qdot[key]} == {
        "task.rewards.joint_velocity_limit_hinge_weight"
    }
    assert {key for key in control if control[key] != v1v2[key]} == {
        "++task.rewards.free_wrist_vel_mimic",
        "++task.rewards.motion_scale_in_window",
    }
    assert deltas["p1_long_no_replay_control_seed3_retry_v2"] == control
