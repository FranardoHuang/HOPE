"""Host/source checks for fixed-question target-metric eligibility.

The four fixed-question arms share physical strike opportunities.  Missing target columns are
``not measured`` rather than failed, and only a complete 111 target may report the three-channel
composite.  These tests extract the two pure helpers from ``hope_commands.py`` so they remain
Isaac-free while exercising the production truth table.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMANDS_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_commands.py"
)
COMMANDS = COMMANDS_PATH.read_text(encoding="utf-8")


def _eligibility_helpers():
    tree = ast.parse(COMMANDS, filename=str(COMMANDS_PATH))
    names = {
        "_action_ball_target_metric_eligibility",
        "_action_ball_target_metric_eligible_counts",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(COMMANDS_PATH), "exec"), namespace)
    return (
        namespace["_action_ball_target_metric_eligible_counts"],
        namespace["_action_ball_target_metric_eligibility"],
    )


def test_fixed_question_target_metric_eligibility_truth_table():
    eligible_counts, eligible_masks = _eligibility_helpers()
    physical = torch.tensor([True, False, True, True])

    expected = {
        (True, True, True): (True, True, True, True),
        (True, False, True): (True, False, True, False),
        (False, False, False): (False, False, False, False),
    }
    for validity, channel_truth in expected.items():
        masks = eligible_masks(physical, validity)
        counts = eligible_counts(7.5, validity)
        for mask, channel_valid, count in zip(masks, channel_truth, counts):
            assert torch.equal(
                mask,
                physical if channel_valid else torch.zeros_like(physical),
            )
            assert count == (7.5 if channel_valid else 0.0)


def test_update_metrics_keeps_physical_denominator_and_masks_target_ledgers():
    tree = ast.parse(COMMANDS, filename=str(COMMANDS_PATH))
    command_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
    )
    update = next(
        node
        for node in command_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_update_metrics"
    )
    source = ast.get_source_segment(COMMANDS, update)
    assert source is not None

    # Physical completion is unchanged; only target-defined numerators and denominators are masked.
    assert "exact_strike.sum(dtype=pos_err.dtype)" in source
    assert "(pos_err * pos_target_eligible).sum()" in source
    assert "(vel_err * vel_target_eligible).sum()" in source
    assert "(normal_err_rad * face_target_eligible).sum()" in source
    assert "& composite_target_eligible" in source
    assert "_action_ball_target_metric_eligible_counts(" in source

    for channel in ("pos", "vel", "normal", "composite"):
        assert f'("{channel}",' in source
    assert (
        'f"strike_{_channel}_target_eligible_sample_count_decayed"'
        in COMMANDS
    )
    assert (
        'f"strike_{_channel}_target_eligible_sample_count_decayed_{_cname}"'
        in COMMANDS
    )


def test_invalid_target_channels_do_not_update_held_failure_metrics():
    tree = ast.parse(COMMANDS, filename=str(COMMANDS_PATH))
    command_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
    )
    update = next(
        node
        for node in command_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_update_metrics"
    )
    source = ast.get_source_segment(COMMANDS, update)
    assert source is not None

    for gate in (
        "in_win & target_pos_valid",
        "in_win & target_vel_valid",
        "in_win & target_face_valid",
        "pos_target_eligible",
        "vel_target_eligible",
        "face_target_eligible",
    ):
        assert gate in source
    assert "self._action_ball_enabled and target_pos_valid" in source
