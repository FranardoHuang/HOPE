"""Host-only contract tests for the exact strike-window entry distance probe."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
COMMAND_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_commands.py"
)
SOURCE = COMMAND_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(COMMAND_PATH))
COMMAND_CLASS = next(
    node
    for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
)


def _probe_class():
    constant_names = {
        "_STRIKE_WINDOW_ENTRY_DISTANCE_BIN_EDGES_M",
        "_STRIKE_WINDOW_ENTRY_DISTANCE_BIN_NAMES",
        "_STRIKE_WINDOW_ENTRY_DISTANCE_COUNT",
        "_STRIKE_WINDOW_ENTRY_DISTANCE_SUM",
        "_STRIKE_WINDOW_ENTRY_DISTANCE_NONFINITE_COUNT",
        "_STRIKE_WINDOW_ENTRY_DISTANCE_BUCKET_COUNTERS",
    }
    constants = []
    for node in TREE.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = {
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if targets & constant_names:
            constants.append(node)
    assert {
        target.id
        for node in constants
        for target in node.targets
        if isinstance(target, ast.Name)
    } == constant_names

    method_names = {
        "_ensure_strike_window_entry_distance_probe_state",
        "_rearm_strike_window_entry_distance_probe",
        "_book_strike_window_entry_distance_probe",
        "_strike_window_entry_distance_probe_exact_state",
        "_stage_strike_window_entry_distance_probe_exact_state",
        "consume_exact_behavior_decision_counters",
    }
    methods = [
        node
        for node in COMMAND_CLASS.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {method.name for method in methods} == method_names
    probe = ast.ClassDef(
        name="Probe",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            *constants,
            probe,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return namespace


PROBE = _probe_class()


class Harness(PROBE["Probe"]):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self._action_ball_enabled = False
        self.ledger = {
            PROBE["_STRIKE_WINDOW_ENTRY_DISTANCE_COUNT"]: torch.zeros(
                (), dtype=torch.long
            ),
            PROBE[
                "_STRIKE_WINDOW_ENTRY_DISTANCE_NONFINITE_COUNT"
            ]: torch.zeros((), dtype=torch.long),
            PROBE["_STRIKE_WINDOW_ENTRY_DISTANCE_SUM"]: torch.zeros(
                (), dtype=torch.float64
            ),
        }
        self.ledger.update(
            {
                name: torch.zeros((), dtype=torch.long)
                for name in PROBE[
                    "_STRIKE_WINDOW_ENTRY_DISTANCE_BUCKET_COUNTERS"
                ]
            }
        )

    def _ensure_exact_behavior_decision_counters(self):
        return self.ledger

    def materialize_action_ball_diagnostic_metrics_for_report(self):
        return None

    def consume_sparse_reward_eligibility_counters(self):
        return {}


def _consume(harness: Harness):
    return harness.consume_exact_behavior_decision_counters()


def test_entry_histogram_is_disjoint_conservative_and_not_recounted_across_updates():
    harness = Harness(9)
    distances = torch.tensor(
        [0.075, 0.10, 0.18, 0.25, 0.40, 0.60, 0.85, 1.20, float("nan")],
        dtype=torch.float32,
    )
    outside = torch.zeros(9, dtype=torch.bool)
    inside = torch.ones(9, dtype=torch.bool)

    with torch.inference_mode():
        harness._book_strike_window_entry_distance_probe(outside, distances)
        harness._book_strike_window_entry_distance_probe(inside, distances)
        # Staying in the window is not a second out->in event.
        harness._book_strike_window_entry_distance_probe(
            inside, distances * 0.5
        )
    first = _consume(harness)

    count_key = PROBE["_STRIKE_WINDOW_ENTRY_DISTANCE_COUNT"]
    sum_key = PROBE["_STRIKE_WINDOW_ENTRY_DISTANCE_SUM"]
    nonfinite_key = PROBE[
        "_STRIKE_WINDOW_ENTRY_DISTANCE_NONFINITE_COUNT"
    ]
    bucket_keys = PROBE[
        "_STRIKE_WINDOW_ENTRY_DISTANCE_BUCKET_COUNTERS"
    ]
    assert first[count_key].item() == 9
    assert first[nonfinite_key].item() == 1
    assert [first[key].item() for key in bucket_keys] == [1] * 8
    assert (
        sum(first[key].item() for key in bucket_keys)
        + first[nonfinite_key].item()
        == first[count_key].item()
    )
    assert first[sum_key].item() == pytest.approx(
        sum((0.075, 0.10, 0.18, 0.25, 0.40, 0.60, 0.85, 1.20)),
        abs=1e-6,
    )

    # Ledger consumption starts a disjoint PPO reporting window but must not rearm the swing.
    harness._book_strike_window_entry_distance_probe(inside, distances)
    second = _consume(harness)
    assert second[count_key].item() == 0
    assert all(second[key].item() == 0 for key in bucket_keys)


def test_partial_reset_or_wrap_rearms_only_selected_environments():
    harness = Harness(4)
    inside = torch.ones(4, dtype=torch.bool)
    distance = torch.tensor([0.05, 0.10, 0.25, 1.25])
    harness._book_strike_window_entry_distance_probe(inside, distance)
    _consume(harness)

    harness._rearm_strike_window_entry_distance_probe(torch.tensor([1, 3]))
    harness._book_strike_window_entry_distance_probe(
        torch.tensor([True, True, True, False]), distance
    )
    # env3 remains armed while outside; it books on its later entry. env0/env2 stay disarmed.
    harness._book_strike_window_entry_distance_probe(
        torch.tensor([False, False, False, True]), distance
    )
    snapshot = _consume(harness)
    assert snapshot[
        PROBE["_STRIKE_WINDOW_ENTRY_DISTANCE_COUNT"]
    ].item() == 2
    assert sum(
        snapshot[key].item()
        for key in PROBE[
            "_STRIKE_WINDOW_ENTRY_DISTANCE_BUCKET_COUNTERS"
        ]
    ) == 2


def test_exact_resume_latch_roundtrip_is_strict_and_side_effect_free_while_staging():
    donor = Harness(4)
    donor._ensure_strike_window_entry_distance_probe_state()
    donor._strike_window_entry_armed.copy_(
        torch.tensor([True, False, True, False])
    )
    state = donor._strike_window_entry_distance_probe_exact_state()
    assert state == [True, False, True, False]

    recipient = Harness(4)
    recipient._ensure_strike_window_entry_distance_probe_state()
    before = recipient._strike_window_entry_armed.clone()
    staged = recipient._stage_strike_window_entry_distance_probe_exact_state(
        state
    )
    assert torch.equal(recipient._strike_window_entry_armed, before)
    recipient._strike_window_entry_armed.copy_(staged)
    assert recipient._strike_window_entry_distance_probe_exact_state() == state

    for bad in (None, [True], [True, False, 1, False], (True,) * 4):
        with pytest.raises(ValueError, match="one boolean per environment"):
            recipient._stage_strike_window_entry_distance_probe_exact_state(bad)


def test_probe_is_wired_only_to_metrics_resample_ledger_and_v6_exact_state():
    assert "_ACTION_BALL_STATE_SCHEMA_VERSION = 6" in SOURCE

    def method_source(name: str) -> str:
        node = next(
            item
            for item in COMMAND_CLASS.body
            if isinstance(item, ast.FunctionDef) and item.name == name
        )
        segment = ast.get_source_segment(SOURCE, node)
        assert segment is not None
        return segment

    update = method_source("_update_metrics")
    resample = method_source("_resample_command")
    save = method_source("_action_ball_exact_resume_state_dict")
    load = method_source("_action_ball_load_exact_resume_state_dict")
    assert "self._book_strike_window_entry_distance_probe(" in update
    assert "self.strike_window, pos_err" in update
    assert "self._rearm_strike_window_entry_distance_probe(env_ids_t)" in resample
    assert '"strike_window_entry_armed"' in save
    assert "_stage_strike_window_entry_distance_probe_exact_state" in load
    assert "self._strike_window_entry_armed.copy_(" in load
