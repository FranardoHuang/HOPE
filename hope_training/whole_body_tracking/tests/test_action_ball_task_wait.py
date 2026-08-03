import copy
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "action_ball_task_wait.py"
)
SPEC = importlib.util.spec_from_file_location(
    "action_ball_task_wait_test_target", MODULE_PATH
)
W = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = W
SPEC.loader.exec_module(W)


def _schedule(**overrides):
    values = {
        "seed": 20260804,
        "min_wait_ticks": 3,
        "max_wait_ticks": 17,
        "episode_horizon_ticks": 128,
        "required_active_ticks": 64,
    }
    values.update(overrides)
    return W.ActionBallTaskWaitSchedule(**values)


def test_schedule_roundtrip_and_canonical_sha_are_content_bound():
    schedule = _schedule()
    restored = W.ActionBallTaskWaitSchedule.from_dict(schedule.to_dict())

    assert restored == schedule
    assert restored.canonical_sha256 == schedule.canonical_sha256
    assert schedule.to_dict()["counter_algorithm"] == W.COUNTER_ALGORITHM
    assert schedule.to_dict()["unit"] == "policy_tick"

    changed = _schedule(seed=schedule.seed + 1)
    assert changed.canonical_sha256 != schedule.canonical_sha256

    tampered = schedule.to_dict()
    tampered["max_wait_ticks"] += 1
    with pytest.raises(ValueError, match="canonical SHA mismatch"):
        W.ActionBallTaskWaitSchedule.from_dict(tampered)


@pytest.mark.parametrize(
    "overrides",
    (
        {"seed": True},
        {"min_wait_ticks": 0},
        {"max_wait_ticks": 2},
        {"episode_horizon_ticks": 1},
        {"required_active_ticks": 0},
        {
            "max_wait_ticks": 70,
            "required_active_ticks": 64,
            "episode_horizon_ticks": 128,
        },
    ),
)
def test_schedule_rejects_invalid_bounds_or_horizon(overrides):
    with pytest.raises(ValueError):
        _schedule(**overrides)


def test_counter_assignment_is_deterministic_bounded_and_arm_neutral():
    schedule_for_a = _schedule()
    schedule_for_c = W.ActionBallTaskWaitSchedule.from_dict(
        schedule_for_a.to_dict()
    )

    rows_a = [
        schedule_for_a.assignment(env_id=env_id, reset_generation=generation)
        for env_id in range(8)
        for generation in range(1, 6)
    ]
    rows_c = [
        schedule_for_c.assignment(env_id=env_id, reset_generation=generation)
        for env_id in range(8)
        for generation in range(1, 6)
    ]

    assert rows_a == rows_c
    assert all(
        schedule_for_a.min_wait_ticks
        <= row.wait_ticks
        <= schedule_for_a.max_wait_ticks
        for row in rows_a
    )
    assert len({row.wait_ticks for row in rows_a}) > 1
    assert all(row.schedule_canonical_sha256 == schedule_for_a.canonical_sha256 for row in rows_a)


def test_known_counter_vector_pins_algorithm_and_integer_tick_result():
    schedule = _schedule()
    assignment = schedule.assignment(env_id=7, reset_generation=11)

    assert assignment.wait_ticks == 9
    assert assignment.rejection_round == 0
    assert assignment == schedule.assignment(env_id=7, reset_generation=11)
    assert len(assignment.canonical_sha256) == 64


def test_wait_task_validity_has_atomic_start_and_no_pre_task_fields():
    schedule = _schedule(min_wait_ticks=5, max_wait_ticks=5)
    assignment = schedule.assignment(env_id=0, reset_generation=1)

    at_reset = W.wait_task_validity(schedule, assignment, elapsed_ticks=0)
    before = W.wait_task_validity(schedule, assignment, elapsed_ticks=4)
    start = W.wait_task_validity(schedule, assignment, elapsed_ticks=5)
    active = W.wait_task_validity(schedule, assignment, elapsed_ticks=9)

    assert at_reset.to_dict() == {
        "phase": "WAIT",
        "wait_active": True,
        "task_active": False,
        "task_fields_valid": False,
        "ball_fields_valid": False,
        "clocks_valid": False,
        "task_started_this_tick": False,
        "wait_remaining_ticks": 5,
        "task_age_ticks": 0,
    }
    assert before.phase == "WAIT"
    assert before.wait_remaining_ticks == 1
    assert start.phase == "TASK"
    assert start.task_started_this_tick
    assert start.task_fields_valid and start.ball_fields_valid and start.clocks_valid
    assert start.task_age_ticks == 0
    assert active.phase == "TASK"
    assert not active.task_started_this_tick
    assert active.task_age_ticks == 4


def test_validity_rejects_forged_assignment_and_elapsed_beyond_horizon():
    schedule = _schedule()
    assignment = schedule.assignment(env_id=1, reset_generation=2)
    forged = W.TaskWaitAssignment(
        schedule_canonical_sha256=assignment.schedule_canonical_sha256,
        env_id=assignment.env_id,
        reset_generation=assignment.reset_generation,
        wait_ticks=(
            schedule.min_wait_ticks
            if assignment.wait_ticks != schedule.min_wait_ticks
            else schedule.min_wait_ticks + 1
        ),
        rejection_round=assignment.rejection_round,
    )

    with pytest.raises(ValueError, match="canonical schedule row"):
        W.wait_task_validity(schedule, forged, elapsed_ticks=0)
    with pytest.raises(ValueError, match="elapsed_ticks"):
        W.wait_task_validity(
            schedule,
            assignment,
            elapsed_ticks=schedule.episode_horizon_ticks + 1,
        )


def test_checkpoint_highwater_roundtrip_rejects_replay_and_gap():
    schedule = _schedule()
    highwater = W.ActionBallTaskWaitHighwater(schedule)
    first = highwater.record(env_id=3, reset_generation=1)
    second = highwater.record(env_id=3, reset_generation=2)
    other = highwater.record(env_id=9, reset_generation=1)

    highwater.assert_recorded(first)
    highwater.assert_recorded(second)
    highwater.assert_recorded(other)
    state = highwater.state_dict()
    restored = W.ActionBallTaskWaitHighwater.from_state_dict(schedule, state)
    assert restored.state_dict() == state
    restored.assert_recorded(second)
    assert restored.record(env_id=3, reset_generation=3) == schedule.assignment(
        env_id=3, reset_generation=3
    )

    with pytest.raises(ValueError, match="advance exactly once"):
        highwater.record(env_id=3, reset_generation=2)
    with pytest.raises(ValueError, match="advance exactly once"):
        highwater.record(env_id=9, reset_generation=3)


def test_checkpoint_highwater_is_self_sealed_and_schedule_bound():
    schedule = _schedule()
    highwater = W.ActionBallTaskWaitHighwater(schedule)
    highwater.record(env_id=1, reset_generation=1)
    highwater.record(env_id=4, reset_generation=1)
    state = highwater.state_dict()

    tampered = copy.deepcopy(state)
    tampered["highwater_by_env"][0][1] = 2
    with pytest.raises(ValueError, match="canonical SHA mismatch"):
        W.ActionBallTaskWaitHighwater.from_state_dict(schedule, tampered)

    unsorted = copy.deepcopy(state)
    unsorted["highwater_by_env"].reverse()
    unsorted_payload = dict(unsorted)
    unsorted_payload.pop("canonical_sha256")
    unsorted["canonical_sha256"] = W._canonical_sha256(unsorted_payload)
    with pytest.raises(ValueError, match="strictly env-sorted"):
        W.ActionBallTaskWaitHighwater.from_state_dict(schedule, unsorted)

    with pytest.raises(ValueError, match="different schedule"):
        W.ActionBallTaskWaitHighwater.from_state_dict(
            _schedule(seed=schedule.seed + 1), state
        )


def test_checkpoint_highwater_rejects_unrecorded_assignment():
    schedule = _schedule()
    highwater = W.ActionBallTaskWaitHighwater(schedule)
    highwater.record(env_id=5, reset_generation=1)

    with pytest.raises(ValueError, match="above checkpoint highwater"):
        highwater.assert_recorded(
            schedule.assignment(env_id=5, reset_generation=2)
        )


def test_a211_c211_frozen_schedule_identity_is_pinned():
    schedule = W.ActionBallTaskWaitSchedule(
        seed=20260804,
        min_wait_ticks=5,
        max_wait_ticks=25,
        episode_horizon_ticks=500,
        required_active_ticks=200,
    )

    assert schedule.canonical_sha256 == (
        "58aa7bb62406d301df619caf7026af8d595f4b8cd9594ea8441b4c89997d400e"
    )


def test_runtime_owner_masks_exact_events_rewards_and_causal_c_rewards():
    mdp = MODULE_PATH.parent / "mdp"
    commands_source = (mdp / "hope_commands.py").read_text(encoding="utf-8")
    rewards_source = (mdp / "hope_rewards.py").read_text(encoding="utf-8")
    causal_source = (mdp / "action_ball_c225_rewards.py").read_text(
        encoding="utf-8"
    )

    assert "self._action_ball_task_valid = torch.ones(" in commands_source
    assert "exact_strike = exact_strike & self._action_ball_task_valid" in commands_source
    assert "active = active & self._action_ball_task_valid[ids]" in commands_source
    assert "capture = capture & task_valid" in commands_source
    assert "net_clear = net_clear & task_valid" in commands_source
    assert "landing_valid = landing_valid & task_valid" in commands_source
    assert "legal_return = legal_return & task_valid" in commands_source
    assert '"wait_countdown_is_public": False' in commands_source
    assert "def action_ball_task_valid_mask(" in rewards_source
    # Dense target windows/base/progress/guidance and every contact/outcome
    # reward each carry an explicit final task-valid intersection.  Balance
    # and non-task imitation functions intentionally remain outside it.
    assert rewards_source.count("action_ball_task_valid_mask(cmd)") >= 16
    for balance_function in (
        "def hold_ready(",
        "def hold_heading(",
        "def pre_strike_foot_slip(",
        "def strike_proj_grav_xy(",
    ):
        start = rewards_source.index(balance_function)
        end = rewards_source.find("\ndef ", start + len(balance_function))
        function_source = rewards_source[start : None if end < 0 else end]
        assert "action_ball_task_valid_mask" not in function_source
    assert causal_source.count("action_ball_task_valid_mask(cmd)") >= 2
