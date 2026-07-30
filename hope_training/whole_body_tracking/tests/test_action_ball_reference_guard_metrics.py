"""Host-only tests for ActionBall reference-guard metrics-only mode."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
SOURCE = MDP / "action_ball_reference_guard.py"
COMMANDS = (MDP / "hope_commands.py").read_text(encoding="utf-8")
REWARDS = (MDP / "hope_rewards.py").read_text(encoding="utf-8")
ENV_CFG = (
    MDP.parent / "config" / "agibot_a3" / "hope_env_cfg.py"
).read_text(encoding="utf-8")

SPEC = importlib.util.spec_from_file_location(
    "action_ball_reference_guard_test_target", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _reference_gate_functions():
    tree = ast.parse(REWARDS, filename=str(MDP / "hope_rewards.py"))
    names = {
        "_action_ball_reference_terminations_mask",
        "_gate_reference_termination",
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
    exec(compile(module, str(MDP / "hope_rewards.py"), "exec"), namespace)
    return tuple(namespace[name] for name in sorted(names))


def _ledger():
    return {
        name: torch.zeros((), dtype=torch.long)
        for name in MODULE.REFERENCE_GUARD_COUNTER_NAMES
    }


def _record_step(
    tracker,
    ledger,
    *,
    token,
    anchor_pos,
    anchor_ori,
    ee_body_pos,
    pre,
    strike,
    center,
):
    common = {
        "step_token": token,
        "pre_strike": torch.tensor(pre, dtype=torch.bool),
        "strike_window": torch.tensor(strike, dtype=torch.bool),
        "center_phase": torch.tensor(center, dtype=torch.bool),
        "ledger": ledger,
    }
    for reason, values in (
        ("anchor_pos", anchor_pos),
        ("anchor_ori", anchor_ori),
        ("ee_body_pos", ee_body_pos),
    ):
        tracker.record(
            reason=reason,
            raw_mask=torch.tensor(values, dtype=torch.bool),
            **common,
        )


def test_mode_truth_table_and_default_are_explicit():
    assert MODULE.validate_reference_guard_mode("phase_gated") == "phase_gated"
    assert MODULE.validate_reference_guard_mode("metrics_only") == "metrics_only"
    for bad in (None, False, "off", "metrics-only"):
        with pytest.raises(ValueError, match="reference_guard_mode"):
            MODULE.validate_reference_guard_mode(bad)
    assert MODULE.REFERENCE_GUARD_HARD_REASONS == (
        "base_fell_tilt",
        "base_too_low",
        "robot_hit_table",
        "joint_qdes_forbidden",
        "joint_actual_forbidden",
    )
    assert len(MODULE.REFERENCE_GUARD_COUNTER_SCHEMA_SHA256) == 64
    assert all(
        character in "0123456789abcdef"
        for character in MODULE.REFERENCE_GUARD_COUNTER_SCHEMA_SHA256
    )
    assert len(MODULE.REFERENCE_GUARD_CONTRACT_SHA256) == 64
    assert (
        MODULE.REFERENCE_GUARD_CONTRACT_PAYLOAD["hard_reasons"]
        == list(MODULE.REFERENCE_GUARD_HARD_REASONS)
    )

    assert (
        "reference_guard_mode: str = _REFERENCE_GUARD_PHASE_GATED"
        in COMMANDS
    )
    assert (
        "reference_guard_mode='metrics_only' is ActionBall-only"
        in COMMANDS
    )
    # Default phase-gated behavior still returns the exact frozen center latch.
    method = COMMANDS.split(
        "def action_ball_reference_terminations_enabled("
    )[1].split("\n    def ", 1)[0]
    assert "return self._action_ball_reference_term_center_latch" in method
    # The treatment changes only the three reference verdicts and reuses one
    # cached all-false mask rather than allocating on every control step.
    assert (
        "_action_ball_reference_term_disabled_mask" in method
    )
    for hard_name in (
        "base_fell_tilt",
        "base_too_low",
        "joint_qdes_forbidden",
        "joint_actual_forbidden",
        "robot_hit_table",
    ):
        assert hard_name in ENV_CFG


def test_three_reasons_union_timing_phase_and_hard_overlap_truth_table():
    tracker = MODULE.ActionBallReferenceGuardMetrics(
        num_envs=4, device="cpu"
    )
    ledger = _ledger()
    _record_step(
        tracker,
        ledger,
        token=7,
        anchor_pos=(True, False, False, True),
        anchor_ori=(False, True, False, True),
        ee_body_pos=(False, False, True, False),
        pre=(True, True, False, False),
        strike=(False, True, False, True),
        center=(True, True, False, False),
    )
    assert ledger["reference_guard_sample_count"].item() == 4
    assert ledger["reference_guard_anchor_pos_count"].item() == 2
    assert ledger["reference_guard_anchor_ori_count"].item() == 2
    assert ledger["reference_guard_ee_body_pos_count"].item() == 1
    assert ledger["reference_guard_union_count"].item() == 4
    assert ledger["reference_guard_reference_only_count"].item() == 4
    assert ledger["reference_guard_reference_and_hard_count"].item() == 0
    assert ledger["reference_guard_pre_sample_count"].item() == 1
    assert ledger["reference_guard_strike_sample_count"].item() == 2
    assert ledger["reference_guard_post_sample_count"].item() == 1
    assert ledger["reference_guard_center_union_count"].item() == 2
    assert ledger["reference_guard_noncenter_union_count"].item() == 2

    # env1 is reference+hard; env3 is reference-only.  A partial reset must
    # move only env1 out of the provisional reference-only bucket.
    tracker.adjust_hard_overlap(
        env_ids=torch.tensor([1, 3], dtype=torch.long),
        hard_mask=torch.tensor([True, False], dtype=torch.bool),
        step_token=7,
        ledger=ledger,
    )
    assert ledger["reference_guard_reference_only_count"].item() == 3
    assert ledger["reference_guard_reference_and_hard_count"].item() == 1
    assert ledger["reference_guard_strike_reference_only_count"].item() == 1
    assert (
        ledger["reference_guard_strike_reference_and_hard_count"].item()
        == 1
    )
    assert ledger["reference_guard_center_reference_only_count"].item() == 1
    assert (
        ledger["reference_guard_center_reference_and_hard_count"].item()
        == 1
    )
    tracker.validate_conservation(ledger)
    # Reset plumbing may revisit a selected row.  Reclassification is
    # device-only and idempotent, so the second visit cannot double count.
    tracker.adjust_hard_overlap(
        env_ids=torch.tensor([1], dtype=torch.long),
        hard_mask=torch.tensor([True], dtype=torch.bool),
        step_token=7,
        ledger=ledger,
    )
    assert ledger["reference_guard_reference_only_count"].item() == 3
    assert ledger["reference_guard_reference_and_hard_count"].item() == 1
    tracker.validate_conservation(ledger)


def test_partial_reset_adjustment_does_not_leak_into_next_step():
    tracker = MODULE.ActionBallReferenceGuardMetrics(
        num_envs=3, device="cpu"
    )
    ledger = _ledger()
    _record_step(
        tracker,
        ledger,
        token=10,
        anchor_pos=(True, True, False),
        anchor_ori=(False, False, False),
        ee_body_pos=(False, False, False),
        pre=(True, True, True),
        strike=(False, False, False),
        center=(True, True, True),
    )
    tracker.adjust_hard_overlap(
        env_ids=torch.tensor([0], dtype=torch.long),
        hard_mask=torch.tensor([True], dtype=torch.bool),
        step_token=10,
        ledger=ledger,
    )
    _record_step(
        tracker,
        ledger,
        token=11,
        anchor_pos=(False, False, True),
        anchor_ori=(False, False, False),
        ee_body_pos=(False, False, False),
        pre=(False, False, False),
        strike=(False, False, False),
        center=(False, False, False),
    )
    # Step 10: two union rows, one hard. Step 11: one fresh reference-only.
    assert ledger["reference_guard_union_count"].item() == 3
    assert ledger["reference_guard_reference_only_count"].item() == 2
    assert ledger["reference_guard_reference_and_hard_count"].item() == 1
    assert ledger["reference_guard_post_reference_only_count"].item() == 1
    assert (
        ledger["reference_guard_post_reference_and_hard_count"].item()
        == 0
    )
    tracker.validate_conservation(ledger)


def test_reset_without_snapshot_is_async_noop_unless_it_was_hard():
    tracker = MODULE.ActionBallReferenceGuardMetrics(
        num_envs=2, device="cpu"
    )
    ledger = _ledger()
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    tracker.adjust_hard_overlap(
        env_ids=env_ids,
        hard_mask=torch.tensor([False, False]),
        step_token=0,
        ledger=ledger,
    )
    tracker.validate_conservation(ledger)

    tracker.adjust_hard_overlap(
        env_ids=env_ids,
        hard_mask=torch.tensor([True, False]),
        step_token=1,
        ledger=ledger,
    )
    assert (
        ledger["reference_guard_hard_without_snapshot_count"].item() == 1
    )
    with pytest.raises(RuntimeError, match="hard reset without"):
        tracker.validate_conservation(ledger)


def test_counter_conservation_rejects_union_partition_drift():
    tracker = MODULE.ActionBallReferenceGuardMetrics(
        num_envs=2, device="cpu"
    )
    ledger = _ledger()
    _record_step(
        tracker,
        ledger,
        token=3,
        anchor_pos=(True, False),
        anchor_ori=(False, False),
        ee_body_pos=(False, False),
        pre=(True, True),
        strike=(False, False),
        center=(True, True),
    )
    tracker.validate_conservation(ledger)
    ledger["reference_guard_reference_only_count"].sub_(1)
    with pytest.raises(RuntimeError, match="does not partition"):
        tracker.validate_conservation(ledger)


def test_incomplete_or_duplicate_step_fails_before_counters_drift():
    tracker = MODULE.ActionBallReferenceGuardMetrics(
        num_envs=2, device="cpu"
    )
    ledger = _ledger()
    kwargs = {
        "raw_mask": torch.tensor([True, False]),
        "pre_strike": torch.tensor([True, True]),
        "strike_window": torch.tensor([False, False]),
        "center_phase": torch.tensor([True, True]),
        "ledger": ledger,
    }
    tracker.record(reason="anchor_pos", step_token=1, **kwargs)
    with pytest.raises(RuntimeError, match="ran twice"):
        tracker.record(reason="anchor_pos", step_token=1, **kwargs)
    with pytest.raises(RuntimeError, match="before all three"):
        tracker.record(reason="anchor_ori", step_token=2, **kwargs)
    assert all(value.item() == 0 for value in ledger.values())


def test_metrics_only_records_raw_before_returning_false_and_never_adds_reward():
    for reason in MODULE.REFERENCE_GUARD_REASONS:
        assert f'reason="{reason}"' in REWARDS
    gate = REWARDS.split("def _gate_reference_termination(")[1].split(
        "\ndef ", 1
    )[0]
    assert "_action_ball_reference_terminations_mask(" in gate
    mask = REWARDS.split(
        "def _action_ball_reference_terminations_mask("
    )[1].split("\ndef ", 1)[0]
    assert mask.index("action_ball_record_reference_guard_raw") < mask.index(
        "return getter()"
    )
    # Instrumentation is not declared as a RewardTerm.
    assert "func=mdp.reference_guard" not in ENV_CFG


def test_metrics_only_gate_records_raw_but_returns_no_reference_reset():
    functions = {
        function.__name__: function for function in _reference_gate_functions()
    }
    gate = functions["_gate_reference_termination"]
    recorded = []
    racket = SimpleNamespace(
        cfg=SimpleNamespace(reference_guard_mode="metrics_only"),
        action_ball_record_reference_guard_raw=lambda reason, mask: recorded.append(
            (reason, mask.clone())
        ),
        action_ball_reference_terminations_enabled=lambda: torch.zeros(
            3, dtype=torch.bool
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_term=lambda name: racket
            if name == "racket_target"
            else (_ for _ in ()).throw(KeyError(name))
        )
    )
    raw = torch.tensor([True, False, True])
    verdict = gate(env, raw, reason="anchor_pos")
    assert torch.equal(verdict, torch.zeros_like(raw))
    assert len(recorded) == 1
    assert recorded[0][0] == "anchor_pos"
    assert torch.equal(recorded[0][1], raw)


def test_default_phase_gate_does_not_touch_metrics_recorder():
    functions = {
        function.__name__: function for function in _reference_gate_functions()
    }
    gate = functions["_gate_reference_termination"]

    def forbidden_recorder(*_args):
        raise AssertionError("default phase_gated path called metrics recorder")

    mask = torch.tensor([True, False, True])
    racket = SimpleNamespace(
        cfg=SimpleNamespace(reference_guard_mode="phase_gated"),
        action_ball_record_reference_guard_raw=forbidden_recorder,
        action_ball_reference_terminations_enabled=lambda: mask,
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_term=lambda name: racket
            if name == "racket_target"
            else (_ for _ in ()).throw(KeyError(name))
        )
    )
    raw = torch.tensor([True, True, False])
    assert torch.equal(
        gate(env, raw, reason="anchor_ori"),
        torch.tensor([True, False, False]),
    )
