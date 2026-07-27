"""Host-only contract tests for the pure task-first curriculum."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "task_first_curriculum.py"
)


def _load_module():
    name = "task_first_curriculum_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = _load_module()
SHA = "a" * 64


def _config(**overrides):
    values = {
        "min_attempts": 100,
        "enter_success_lower_bound": 0.90,
        "exit_success_lower_bound": 0.70,
        "enter_unsafe_upper_bound": 0.05,
        "exit_unsafe_upper_bound": 0.10,
        "enter_dwell_updates": 1,
        "exit_dwell_updates": 2,
        "max_stall_updates": 100,
        "stall_policy": "fail",
    }
    values.update(overrides)
    return C.GateConfig(**values)


GOOD = C.OutcomeCounts(attempts=100, successes=100, unsafe_failures=0)
NEUTRAL = C.OutcomeCounts(attempts=100, successes=80, unsafe_failures=0)
BAD = C.OutcomeCounts(attempts=100, successes=40, unsafe_failures=20)
INSUFFICIENT = C.OutcomeCounts(attempts=20, successes=20, unsafe_failures=0)


def _curriculum(actions=("fh",), **config_overrides):
    return C.TaskFirstCurriculum(
        manifest_sha256=SHA,
        action_order=actions,
        gate_config=_config(**config_overrides),
    )


def _advance_single(curriculum, evidence):
    return curriculum.advance({curriculum.action_order[0]: evidence})[0]


def test_wilson_interval_has_expected_bounds_and_rejects_invalid_counts():
    assert C.wilson_interval(0, 0) == (0.0, 1.0)
    lower, upper = C.wilson_interval(50, 100)
    assert lower == pytest.approx(0.4038315304)
    assert upper == pytest.approx(0.5961684696)
    zero_lower, zero_upper = C.wilson_interval(0, 100)
    assert zero_lower == 0.0
    assert zero_upper == pytest.approx(0.0369934982)
    assert C.wilson_interval(90, 100)[0] < C.wilson_interval(100, 100)[0]

    with pytest.raises(TypeError, match="plain integers"):
        C.wilson_interval(True, 1)
    with pytest.raises(ValueError, match="between zero and attempts"):
        C.wilson_interval(2, 1)
    with pytest.raises(ValueError, match="positive"):
        C.wilson_interval(0, 1, z=0.0)


def test_gate_uses_success_lcb_unsafe_ucb_minimum_attempts_and_hysteresis():
    good = C.evaluate_gate(GOOD, _config())
    assert good.enter_ok is True
    assert good.exit_bad is False
    assert good.success_lower >= 0.90
    assert good.unsafe_upper <= 0.05

    insufficient = C.evaluate_gate(INSUFFICIENT, _config())
    assert insufficient.enter_ok is False
    assert insufficient.exit_bad is False
    assert "minimum_attempts" in insufficient.enter_blockers

    neutral = C.evaluate_gate(NEUTRAL, _config())
    assert neutral.enter_ok is False
    assert neutral.exit_bad is False

    bad = C.evaluate_gate(BAD, _config())
    assert bad.enter_ok is False
    assert bad.exit_bad is True
    assert set(bad.exit_reasons) == {"success_lower_bound", "unsafe_upper_bound"}


def test_gate_config_and_outcomes_fail_closed_on_incoherent_values():
    with pytest.raises(ValueError, match="at least exit"):
        _config(enter_success_lower_bound=0.60, exit_success_lower_bound=0.70)
    with pytest.raises(ValueError, match="must not exceed"):
        _config(enter_unsafe_upper_bound=0.20, exit_unsafe_upper_bound=0.10)
    with pytest.raises(ValueError, match="longer dwell"):
        _config(enter_dwell_updates=3, max_stall_updates=2)
    with pytest.raises(ValueError, match="disjoint"):
        C.OutcomeCounts(attempts=10, successes=8, unsafe_failures=3)
    with pytest.raises(TypeError, match="plain integer"):
        C.OutcomeCounts(attempts=True, successes=0)


def test_scaled_ranges_expand_asymmetrically_from_center_at_fixed_levels():
    assert C.scaled_interval(2.0, (-2.0, 6.0), 0.0) == (2.0, 2.0)
    assert C.scaled_interval(2.0, (-2.0, 6.0), 0.25) == (1.0, 3.0)
    assert C.scaled_interval(2.0, (-2.0, 6.0), 1.0) == (-2.0, 6.0)
    assert C.scaled_axis_ranges(
        centers=(0.0, 10.0),
        full_ranges=((-2.0, 6.0), (4.0, 12.0)),
        level=0.5,
    ) == ((-1.0, 3.0), (7.0, 11.0))

    centers = {axis: (float(index),) for index, axis in enumerate(C.AXES)}
    full = {
        axis: ((float(index - 1), float(index + 2)),)
        for index, axis in enumerate(C.AXES)
    }
    levels = dict(zip(C.AXES, C.LEVELS[:4]))
    result = C.compute_axis_ranges(centers, full, levels)
    assert tuple(result) == C.AXES
    assert result["position"] == ((0.0, 0.0),)
    assert result["speed"] == ((0.75, 1.5),)

    with pytest.raises(ValueError, match="center must lie"):
        C.scaled_interval(5.0, (0.0, 4.0), 0.5)
    with pytest.raises(ValueError, match="one of"):
        C.compute_axis_ranges(centers, full, {**levels, "base": 0.3})
    with pytest.raises(ValueError, match="keys"):
        C.compute_axis_ranges(
            {key: value for key, value in centers.items() if key != "base"},
            full,
            levels,
        )


@pytest.mark.parametrize("action_count", [1, 5, 93])
def test_balanced_sampler_supports_arbitrary_n_and_never_drifts_by_more_than_one(
    action_count,
):
    actions = tuple(f"action_{index:03d}" for index in range(action_count))
    sampler = C.BalancedActionSampler(actions)
    draws = sampler.sample(action_count * 3 + max(0, action_count - 1))
    counts = Counter(draws)
    assert set(counts) == set(actions)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert draws[:action_count] == actions


def test_balanced_sampler_resume_is_deterministic_and_strict():
    sampler = C.BalancedActionSampler(("a", "b", "c", "d", "e"))
    assert sampler.sample(7) == ("a", "b", "c", "d", "e", "a", "b")
    state = sampler.state_dict()
    resumed = C.BalancedActionSampler(("a", "b", "c", "d", "e"))
    resumed.load_state_dict(state)
    assert resumed.sample(17) == sampler.sample(17)

    broken = deepcopy(state)
    broken["cursor"] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        C.BalancedActionSampler(("a", "b", "c", "d", "e")).load_state_dict(
            broken
        )
    with pytest.raises(ValueError, match="does not match"):
        C.BalancedActionSampler(("a", "b", "c", "d", "e")).load_state_dict(
            {**state, "action_order": ["b", "a", "c", "d", "e"]}
        )
    with pytest.raises(ValueError, match="keys"):
        resumed.load_state_dict({**state, "extra": 1})


def test_actions_progress_independently_without_pooled_success():
    curriculum = _curriculum(("fh_drive", "bh_drive"))
    result = curriculum.advance({"fh_drive": GOOD, "bh_drive": BAD})
    assert result[0].kind == "promote"
    assert result[1].kind == "hold"
    assert curriculum.axis_level("fh_drive", "position") == 0.25
    assert curriculum.axis_level("bh_drive", "position") == 0.0

    for _ in range(3):
        curriculum.advance({"fh_drive": GOOD, "bh_drive": NEUTRAL})
    assert curriculum.axis_level("fh_drive", "position") == 1.0
    assert curriculum.axis_level("fh_drive", "speed") == 0.0
    assert curriculum.active_axis("fh_drive") == "speed"
    curriculum.advance({"fh_drive": GOOD, "bh_drive": NEUTRAL})
    assert curriculum.axis_level("fh_drive", "speed") == 0.25
    assert curriculum.axis_level("bh_drive", "position") == 0.0


def test_four_axes_promote_sequentially_one_fixed_level_per_update():
    curriculum = _curriculum()
    promoted_axes = []
    previous = curriculum.level_indices("fh")
    for _ in range(len(C.AXES) * (len(C.LEVELS) - 1)):
        result = _advance_single(curriculum, GOOD)
        current = curriculum.level_indices("fh")
        assert sum(current) == sum(previous) + 1
        promoted_axes.append(result.axis)
        previous = current

    assert promoted_axes == [
        "position",
        "position",
        "position",
        "position",
        "speed",
        "speed",
        "speed",
        "speed",
        "face",
        "face",
        "face",
        "face",
        "base",
        "base",
        "base",
        "base",
    ]
    assert curriculum.is_complete("fh") is True
    assert curriculum.levels("fh") == {axis: 1.0 for axis in C.AXES}
    assert _advance_single(curriculum, GOOD).kind == "complete_hold"
    assert _advance_single(curriculum, BAD).kind == "complete_hold"
    rollback = _advance_single(curriculum, BAD)
    assert rollback.kind == "retreat"
    assert rollback.axis == "base"
    assert (rollback.from_level, rollback.to_level) == (1.0, 0.75)


def test_exit_dwell_rolls_back_exactly_one_level_including_prior_axis_frontier():
    curriculum = _curriculum(exit_dwell_updates=2)
    for _ in range(4):
        _advance_single(curriculum, GOOD)
    assert curriculum.levels("fh") == {
        "position": 1.0,
        "speed": 0.0,
        "face": 0.0,
        "base": 0.0,
    }

    assert _advance_single(curriculum, BAD).kind == "hold"
    rollback = _advance_single(curriculum, BAD)
    assert rollback.kind == "retreat"
    assert rollback.axis == "position"
    assert (rollback.from_level, rollback.to_level) == (1.0, 0.75)
    assert curriculum.level_indices("fh") == (3, 0, 0, 0)

    _advance_single(curriculum, GOOD)
    _advance_single(curriculum, GOOD)
    assert curriculum.level_indices("fh") == (4, 1, 0, 0)
    _advance_single(curriculum, BAD)
    rollback = _advance_single(curriculum, BAD)
    assert rollback.axis == "speed"
    assert (rollback.from_level, rollback.to_level) == (0.25, 0.0)
    assert curriculum.level_indices("fh") == (4, 0, 0, 0)

    at_center = _curriculum(exit_dwell_updates=1)
    assert _advance_single(at_center, BAD).kind == "hold"
    assert at_center.level_indices("fh") == (0, 0, 0, 0)


def test_neutral_and_single_bad_update_preserve_enter_evidence_but_reset_exit_dwell():
    curriculum = _curriculum(
        enter_dwell_updates=3,
        exit_dwell_updates=2,
        max_stall_updates=20,
    )
    assert _advance_single(curriculum, GOOD).kind == "hold"
    assert _advance_single(curriculum, NEUTRAL).kind == "hold"
    assert _advance_single(curriculum, BAD).kind == "hold"
    assert _advance_single(curriculum, NEUTRAL).kind == "hold"
    assert _advance_single(curriculum, GOOD).kind == "hold"
    promoted = _advance_single(curriculum, GOOD)
    assert promoted.kind == "promote"
    assert curriculum.axis_level("fh", "position") == 0.25


def test_fail_stall_policy_is_atomic_across_actions():
    curriculum = _curriculum(
        ("advancing", "stalled"),
        max_stall_updates=2,
        stall_policy="fail",
    )
    curriculum.advance({"advancing": GOOD, "stalled": INSUFFICIENT})
    before = curriculum.state_dict()
    with pytest.raises(C.CurriculumStalledError, match="stalled"):
        curriculum.advance({"advancing": GOOD, "stalled": INSUFFICIENT})
    assert curriculum.state_dict() == before
    assert curriculum.axis_level("advancing", "position") == 0.25


def test_freeze_stall_policy_freezes_only_the_stalled_action():
    curriculum = _curriculum(
        ("healthy", "stalled"),
        max_stall_updates=2,
        stall_policy="freeze",
    )
    curriculum.advance({"healthy": GOOD, "stalled": INSUFFICIENT})
    results = curriculum.advance({"healthy": GOOD, "stalled": INSUFFICIENT})
    assert results[0].kind == "promote"
    assert results[1].kind == "freeze"
    assert curriculum.axis_level("healthy", "position") == 0.5
    assert curriculum.is_frozen("stalled") is True
    results = curriculum.advance({"healthy": GOOD, "stalled": GOOD})
    assert results[1].kind == "frozen_hold"
    assert curriculum.axis_level("stalled", "position") == 0.0


def test_state_round_trip_restores_progress_dwell_and_sampler_future_exactly():
    curriculum = _curriculum(("a", "b"), enter_dwell_updates=2)
    curriculum.advance({"a": GOOD, "b": NEUTRAL})
    curriculum.sample_actions(7)
    state = curriculum.state_dict()
    json.dumps(state)

    resumed = _curriculum(("a", "b"), enter_dwell_updates=2)
    resumed.load_state_dict(json.loads(json.dumps(state)))
    assert resumed.state_dict() == state
    assert resumed.sample_actions(25) == curriculum.sample_actions(25)
    assert resumed.advance({"a": GOOD, "b": NEUTRAL}) == curriculum.advance(
        {"a": GOOD, "b": NEUTRAL}
    )
    assert resumed.state_dict() == curriculum.state_dict()


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda state: state.update(schema_version=999), "schema_version"),
        (lambda state: state.update(manifest_sha256="b" * 64), "manifest_sha256"),
        (lambda state: state.update(action_order=["b", "a"]), "action_order"),
        (lambda state: state.update(extra=True), "keys"),
        (lambda state: state.update(levels=[0.0, 1.0]), "levels"),
        (
            lambda state: state["gate_config"].update(min_attempts=101),
            "gate_config",
        ),
        (
            lambda state: state["progress"]["a"]["level_indices"].__setitem__(1, 1),
            "sequential",
        ),
        (
            lambda state: state["sampler"].update(schema_version=999),
            "sampler schema_version",
        ),
        (
            lambda state: state["progress"]["a"]["enter_dwell"].__setitem__(1, 1),
            "non-active-axis",
        ),
        (
            lambda state: state["progress"]["a"]["stall_updates"].__setitem__(0, 100),
            "stall counter",
        ),
    ],
)
def test_state_load_rejects_identity_or_schema_drift_without_partial_mutation(
    mutation,
    match,
):
    curriculum = _curriculum(("a", "b"))
    baseline = curriculum.state_dict()
    candidate = deepcopy(baseline)
    mutation(candidate)
    with pytest.raises((TypeError, ValueError), match=match):
        curriculum.load_state_dict(candidate)
    assert curriculum.state_dict() == baseline


def test_reachable_frozen_state_round_trips_but_impossible_frozen_state_fails():
    curriculum = _curriculum(
        ("a",),
        max_stall_updates=2,
        stall_policy="freeze",
    )
    _advance_single(curriculum, INSUFFICIENT)
    _advance_single(curriculum, INSUFFICIENT)
    state = curriculum.state_dict()
    resumed = _curriculum(
        ("a",),
        max_stall_updates=2,
        stall_policy="freeze",
    )
    resumed.load_state_dict(deepcopy(state))
    assert resumed.state_dict() == state

    impossible = deepcopy(state)
    impossible["progress"]["a"]["stall_updates"][0] = 1
    with pytest.raises(ValueError, match="frozen state"):
        resumed.load_state_dict(impossible)
    assert resumed.state_dict() == state


def test_curriculum_constructs_and_serializes_ninety_three_actions():
    actions = tuple(f"clip_{index:03d}" for index in range(93))
    curriculum = _curriculum(actions)
    assert curriculum.action_order == actions
    assert curriculum.sample_actions(190)[:93] == actions
    state = curriculum.state_dict()
    assert tuple(state["progress"]) == actions
    assert len(state["progress"]) == 93


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"manifest_sha256": "A" * 64, "action_order": ("a",)}, "lowercase"),
        ({"manifest_sha256": SHA, "action_order": ()}, "at least one"),
        ({"manifest_sha256": SHA, "action_order": ("a", "a")}, "duplicate"),
    ],
)
def test_curriculum_identity_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        C.TaskFirstCurriculum(gate_config=_config(), **kwargs)
