"""ActionBall diagnostic host-packet and async-invariant regression tests.

The optimized diagnostic path preserves exact contact/capture truth and the historical
Python-float telemetry recurrences.  It only combines already-reduced scalars into the one
exact-any host transfer that diagnostic VirtualBall already requires, and moves pure invariant
checks to ordered CUDA assertions.  Formal/default execution remains synchronously fail-fast.

The CUDA synchronization and wall-time acceptance run on a Pod; these CPU tests use the real
``hope_commands.py`` through the repository's Isaac-free module fixture.
"""

from __future__ import annotations

import inspect
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_reward_flags_mdp import hope_commands_mod  # noqa: E402


def _yaw_quats(yaws: list[float]) -> torch.Tensor:
    yaw = torch.as_tensor(yaws, dtype=torch.float32)
    quat = torch.zeros(len(yaw), 4)
    quat[:, 0] = torch.cos(0.5 * yaw)
    quat[:, 3] = torch.sin(0.5 * yaw)
    return quat


def _hold_command(
    yaws: list[float],
    *,
    diagnostic: bool,
):
    command_type = hope_commands_mod.RacketTargetCommand
    command = command_type.__new__(command_type)
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(root_quat_w=_yaw_quats(yaws))
    )
    command.cfg = types.SimpleNamespace(
        virtual_ball=True,
        vb_metrics_only=False,
    )
    command._action_ball_enabled = diagnostic
    command._action_ball_diagnostic_unauthorized = diagnostic
    command._diagnostic_hold_recovery_metric_scalars = ()
    command._hold_edge_pending = torch.zeros(len(yaws), dtype=torch.bool)
    command._previous_in_hold = torch.zeros(len(yaws), dtype=torch.bool)
    command._hold_start_yaw = torch.zeros(len(yaws))
    command._heading_expiry_sum_acc = 0.0
    command._heading_expiry_n_acc = 0.0
    command._recovery_spawn_sum_acc = 0.0
    command._recovery_expiry_sum_acc = 0.0
    command._recovery_n_acc = 0.0
    return command


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_diagnostic_packet_matches_individual_reads_with_one_batch(
    monkeypatch,
    dtype: torch.dtype,
):
    calls: list[int] = []
    original = hope_commands_mod._batched_host_scalar_values

    def counted(values):
        values = tuple(values)
        calls.append(len(values))
        return original(values)

    monkeypatch.setattr(
        hope_commands_mod,
        "_batched_host_scalar_values",
        counted,
    )
    metrics = (
        torch.tensor(1.25, dtype=dtype),
        torch.tensor(-2.5, dtype=dtype),
        torch.tensor(7.0, dtype=dtype),
    )
    result = hope_commands_mod._action_ball_diagnostic_host_packet(
        orphan_exact=torch.tensor(False),
        identity_drift=torch.tensor(True),
        exact_any=torch.tensor(True),
        metric_scalars=metrics,
    )

    assert result[:3] == (False, True, True)
    assert result[3] == tuple(float(value) for value in metrics)
    assert calls == [6]


def test_tensor_predicate_validation_keeps_precise_cpu_failure():
    validate = hope_commands_mod._action_ball_validate_tensor_predicate
    validate(torch.tensor([True, True]), "healthy", async_validate=False)
    validate(torch.tensor([True, True]), "healthy", async_validate=True)

    for async_validate in (False, True):
        with pytest.raises(RuntimeError, match="kept failure text"):
            validate(
                torch.tensor([True, False]),
                "kept failure text",
                async_validate=async_validate,
            )
    with pytest.raises(TypeError, match="exact boolean"):
        validate(torch.tensor(True), "bad flag", async_validate=1)
    with pytest.raises(TypeError, match="tensor predicate"):
        validate(True, "bad predicate", async_validate=False)


def test_diagnostic_hold_recovery_staging_matches_legacy_accumulators():
    legacy = _hold_command([0.5, 0.1, -0.6], diagnostic=False)
    diagnostic = _hold_command([0.5, 0.1, -0.6], diagnostic=True)
    started = torch.tensor([True, True, True])
    legacy._update_hold_recovery_metrics(started)
    diagnostic._update_hold_recovery_metrics(started)

    # The first diagnostic step is retained on device.  Simulate the following step's fused
    # exact-any packet before asking the helper to stage another row.
    assert len(diagnostic._diagnostic_hold_recovery_metric_scalars) == 5
    assert diagnostic._heading_expiry_n_acc == 0.0
    diagnostic._apply_hold_recovery_host_values(
        hope_commands_mod._batched_host_scalar_values(
            diagnostic._diagnostic_hold_recovery_metric_scalars
        )
    )
    diagnostic._diagnostic_hold_recovery_metric_scalars = ()

    legacy.robot.data.root_quat_w = _yaw_quats([0.2, 0.05, -0.1])
    diagnostic.robot.data.root_quat_w = _yaw_quats([0.2, 0.05, -0.1])
    expired = torch.tensor([False, False, False])
    legacy._update_hold_recovery_metrics(expired)
    diagnostic._update_hold_recovery_metrics(expired)
    diagnostic._flush_diagnostic_hold_recovery_metric_scalars()

    names = (
        "_heading_expiry_sum_acc",
        "_heading_expiry_n_acc",
        "_recovery_spawn_sum_acc",
        "_recovery_expiry_sum_acc",
        "_recovery_n_acc",
    )
    for name in names:
        assert getattr(diagnostic, name) == pytest.approx(
            getattr(legacy, name)
        )
    assert diagnostic._diagnostic_hold_recovery_metric_scalars == ()


def test_diagnostic_hold_recovery_staging_refuses_overwrite():
    command = _hold_command([0.4], diagnostic=True)
    command._update_hold_recovery_metrics(torch.tensor([True]))
    with pytest.raises(RuntimeError, match="was not consumed"):
        command._update_hold_recovery_metrics(torch.tensor([True]))


def test_diagnostic_hotpath_source_guards():
    command_type = hope_commands_mod.RacketTargetCommand
    validation = inspect.getsource(
        hope_commands_mod._action_ball_validate_tensor_predicate
    )
    evaluate = inspect.getsource(command_type._vb_evaluate)
    timing = inspect.getsource(command_type._compute_strike_timing)
    history = inspect.getsource(
        command_type._action_ball_store_virtual_contact_history
    )
    sparse = inspect.getsource(
        command_type._book_sparse_reward_eligibility
    )
    metrics = inspect.getsource(command_type._update_metrics)
    consume = inspect.getsource(
        command_type.consume_exact_behavior_decision_counters
    )

    assert "_assert_async" in validation
    assert "if not bool(scalar):" in validation
    assert "_action_ball_diagnostic_host_packet(" in evaluate
    assert (
        "active action-ball task has no receipt-owned Motion timing"
        in timing
    )
    assert "bool((active & ~raw_state_finite).any())" not in history
    assert "bool((partition & mask).any())" not in sparse
    assert "torch.equal(partition, exact_strike)" not in sparse
    assert "+ _pending_hold_recovery_metric_scalars" in metrics
    assert metrics.index(
        "self._apply_hold_recovery_host_values("
    ) < metrics.index("self._decay_swing_accounting(decay)")
    assert (
        "self._flush_diagnostic_hold_recovery_metric_scalars()"
        in consume
    )
