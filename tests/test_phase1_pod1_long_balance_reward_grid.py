from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs" / "phase1_pod1_long_balance_reward_grid_20260715.yaml"


def _load():
    data = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    return data, data["jobs"]


def _delta(job):
    return dict(item.split("=", 1) for item in job["recipe"]["delta"])


def test_twelve_distinct_single_seed_jobs_fill_pod1_by_rounds():
    data, jobs = _load()
    assert data["dispatch_pods"] == ["pod1"]
    assert data["pods"]["pod1"]["max_trainers_per_gpu"] == 4
    assert len(jobs) == 12
    assert len({job["id"] for job in jobs}) == 12
    assert all(job["status"] == "ready" for job in jobs)
    assert {job["seed"] for job in jobs} == {3}
    assert {job["budget"]["num_envs"] for job in jobs} == {4096}
    assert {job["budget"]["max_iterations"] for job in jobs} == {10001}
    assert {tuple(job["milestones"]) for job in jobs} == {
        (200, 500, 1000, 2000, 3000, 6000, 10000)
    }
    assert [job["resource"]["required_slot"] for job in jobs] == [
        f"pod1/gpu{gpu}" for _round in range(4) for gpu in range(3)
    ]
    assert {job["source"]["commit"] for job in jobs} == {
        "2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e"
    }
    assert all(job["runtime_binding"] is True for job in jobs)


def test_balance_grid_is_non_striking_arm_by_episode_length():
    _, jobs = _load()
    cells = set()
    for job in jobs[:6]:
        delta = _delta(job)
        cells.add(
            (
                float(delta["task.env.episode_length_s"]),
                delta["++task.rewards.free_non_striking_arm_mimic"],
            )
        )
        assert delta["task.rewards.racket_position_weight"] == "14.0"
        assert delta["task.rewards.racket_velocity_weight"] == "10.0"
        assert delta["task.rewards.racket_normal_weight"] == "5.0"
    assert cells == {
        (seconds, free) for seconds in (10.0, 16.0, 24.0) for free in ("false", "true")
    }


def test_reward_rows_change_only_the_three_dense_strike_weights():
    data, jobs = _load()
    triples = []
    for job in jobs[6:]:
        delta = _delta(job)
        assert delta["task.env.episode_length_s"] == "10.0"
        assert delta["++task.rewards.free_non_striking_arm_mimic"] == "false"
        triples.append(
            tuple(
                float(delta[key])
                for key in (
                    "task.rewards.racket_position_weight",
                    "task.rewards.racket_velocity_weight",
                    "task.rewards.racket_normal_weight",
                )
            )
        )
    assert triples == [
        (9.67, 9.67, 9.66),
        (19.0, 5.0, 5.0),
        (5.0, 19.0, 5.0),
        (5.0, 5.0, 19.0),
        (4.0, 0.5, 0.5),
        (28.0, 20.0, 10.0),
    ]
    contract = data["decision_contract"]
    assert contract["sparse_hit_behavior_may_stop_before_minimum_eligible_events"] is False
    assert contract["second_seed_authorized"] is False
    assert contract["promotion_authorized"] is False
