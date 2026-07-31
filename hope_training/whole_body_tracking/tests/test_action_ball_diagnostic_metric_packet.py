"""ActionBall diagnostic host-packet and async-invariant regression tests.

The optimized diagnostic path preserves exact contact/capture truth and the historical
telemetry recurrences.  Training-relevant exact metrics still share the required exact-any host
packet; reporting-only swing/VirtualBall EMAs remain float64 device scalars until the PPO update
boundary.  Pure invariant checks use ordered CUDA assertions.  Formal/default execution remains
synchronously fail-fast.

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


def _report_command():
    command = _hold_command([0.0], diagnostic=True)
    command.cfg.exact_success_min_count = 1.0
    command.cfg.rally_legacy_metrics = True
    command._clip_names = {0: "block"}
    command._exact_n_acc = 1.0
    command._exact_n_acc_c = {0: 1.0}
    command._swing_starts_acc = 2.0
    command._swing_starts_acc_c = {0: 0.0}
    command._prestrike_fall_acc = 0.0
    command._poststrike_fall_acc = 0.0
    command._prestrike_fall_acc_c = {0: 0.0}
    command._poststrike_fall_acc_c = {0: 0.0}
    command._drift_n_acc = 0.0
    command._drift_sum_acc = 0.0
    command._drift_fwd_sum_acc = 0.0
    command._station_offset_start_sum_acc = 0.0
    command._tracking_loss_acc = 0.0
    command._tracking_loss_acc_c = {0: 0.0}
    command._rally_starts_acc = 0.0
    command._rally_returns_acc = 0.0
    command._rally_starts_acc_c = {0: 0.0}
    command._rally_returns_acc_c = {0: 0.0}
    command._vb_exact_acc = 0.0
    command._vb_hit_acc = 0.0
    command._vb_net_acc = 0.0
    command._vb_land_valid_acc = 0.0
    command._vb_inb_acc = 0.0
    command._vb_exact_acc_c = {0: 0.0}
    command._vb_hit_acc_c = {0: 0.0}
    command._vb_inb_acc_c = {0: 0.0}
    names = (
        "swing_completion_rate",
        "pre_strike_fall_rate",
        "post_strike_fall_rate",
        "base_drift_per_swing",
        "base_drift_fwd_per_swing",
        "base_station_offset_at_swing_start",
        "base_heading_abs_at_swing_start",
        "base_heading_hold_expiry_count",
        "heading_recovery_spawn_yaw",
        "heading_recovery_expiry_yaw",
        "heading_recovery_count",
        "tracking_loss_rate",
        "virtual_return_rate_rally",
        "virtual_hit_rate",
        "virtual_net_clear_rate",
        "virtual_land_valid_rate",
        "virtual_land_inbounds_rate",
        "virtual_return_rate",
        "virtual_return_rate_rally_legacy",
        "swing_completion_rate_block",
        "pre_strike_fall_rate_block",
        "post_strike_fall_rate_block",
        "tracking_loss_rate_block",
        "virtual_return_rate_rally_block",
        "virtual_return_rate_block",
        "virtual_hit_rate_block",
        "virtual_return_rate_rally_block_legacy",
    )
    command.metrics = {
        name: torch.full((1,), -1.0) for name in names
    }
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

    # Pure diagnostic telemetry stays on device and accepts subsequent steps without a host read.
    assert diagnostic._diagnostic_hold_recovery_metric_scalars == ()
    assert diagnostic._heading_expiry_n_acc == 0.0

    legacy.robot.data.root_quat_w = _yaw_quats([0.2, 0.05, -0.1])
    diagnostic.robot.data.root_quat_w = _yaw_quats([0.2, 0.05, -0.1])
    expired = torch.tensor([False, False, False])
    legacy._update_hold_recovery_metrics(expired)
    diagnostic._update_hold_recovery_metrics(expired)
    diagnostic._flush_action_ball_diagnostic_device_telemetry()

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


def test_diagnostic_device_telemetry_has_one_update_boundary_host_read(
    monkeypatch,
):
    command = _hold_command([0.4], diagnostic=True)
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
    command._update_hold_recovery_metrics(torch.tensor([True]))
    command._update_hold_recovery_metrics(torch.tensor([True]))
    assert calls == []
    command._flush_action_ball_diagnostic_device_telemetry()
    assert calls == [5]
    command._flush_action_ball_diagnostic_device_telemetry()
    assert calls == [5]


def test_report_boundary_refreshes_private_and_public_metrics():
    command = _report_command()

    increments = (
        ("swing", ("_swing_starts_acc_c", 0), 2.0),
        ("swing", ("_rally_starts_acc", None), 2.0),
        ("swing", ("_rally_returns_acc", None), 1.0),
        ("swing", ("_rally_starts_acc_c", 0), 2.0),
        ("swing", ("_rally_returns_acc_c", 0), 1.0),
        ("swing", ("_drift_n_acc", None), 1.0),
        ("swing", ("_drift_sum_acc", None), 0.2),
        ("swing", ("_drift_fwd_sum_acc", None), 0.1),
        ("swing", ("_station_offset_start_sum_acc", None), 0.05),
        ("swing", ("_prestrike_fall_acc", None), 1.0),
        ("swing", ("_prestrike_fall_acc_c", 0), 1.0),
        ("swing", ("_heading_expiry_sum_acc", None), 0.4),
        ("swing", ("_heading_expiry_n_acc", None), 2.0),
        ("swing", ("_recovery_spawn_sum_acc", None), 0.6),
        ("swing", ("_recovery_expiry_sum_acc", None), 0.2),
        ("swing", ("_recovery_n_acc", None), 2.0),
        ("swing", ("_tracking_loss_acc", None), 1.0),
        ("swing", ("_tracking_loss_acc_c", 0), 1.0),
        ("vb", ("_vb_exact_acc", None), 2.0),
        ("vb", ("_vb_hit_acc", None), 1.0),
        ("vb", ("_vb_net_acc", None), 1.0),
        ("vb", ("_vb_land_valid_acc", None), 1.0),
        ("vb", ("_vb_inb_acc", None), 1.0),
        ("vb", ("_vb_exact_acc_c", 0), 2.0),
        ("vb", ("_vb_hit_acc_c", 0), 1.0),
        ("vb", ("_vb_inb_acc_c", 0), 1.0),
    )
    for group, target, value in increments:
        command._action_ball_diagnostic_device_telemetry_add(
            group,
            target,
            torch.tensor(value),
        )

    with pytest.raises(RuntimeError, match="pending on device"):
        command.assert_action_ball_diagnostic_metrics_materialized_for_report()
    assert command.metrics["virtual_return_rate"].item() == -1.0

    command.materialize_action_ball_diagnostic_metrics_for_report()
    command.assert_action_ball_diagnostic_metrics_materialized_for_report()

    assert command._vb_exact_acc == pytest.approx(2.0)
    assert command._vb_hit_acc == pytest.approx(1.0)
    assert command._rally_starts_acc == pytest.approx(2.0)
    assert command._rally_returns_acc == pytest.approx(1.0)
    assert command._swing_starts_acc_c[0] == pytest.approx(2.0)
    assert command._tracking_loss_acc == pytest.approx(1.0)
    expected = {
        "swing_completion_rate": 0.5,
        "pre_strike_fall_rate": 0.5,
        "post_strike_fall_rate": 0.0,
        "base_drift_per_swing": 0.2,
        "base_drift_fwd_per_swing": 0.1,
        "base_station_offset_at_swing_start": 0.05,
        "base_heading_abs_at_swing_start": 0.2,
        "base_heading_hold_expiry_count": 2.0,
        "heading_recovery_spawn_yaw": 0.3,
        "heading_recovery_expiry_yaw": 0.1,
        "heading_recovery_count": 2.0,
        "tracking_loss_rate": 0.5,
        "virtual_return_rate_rally": 0.5,
        "virtual_hit_rate": 0.5,
        "virtual_net_clear_rate": 1.0,
        "virtual_land_valid_rate": 1.0,
        "virtual_land_inbounds_rate": 1.0,
        "virtual_return_rate": 0.5,
        "virtual_return_rate_rally_legacy": 0.5,
        "swing_completion_rate_block": 0.5,
        "pre_strike_fall_rate_block": 0.5,
        "post_strike_fall_rate_block": 0.0,
        "tracking_loss_rate_block": 0.5,
        "virtual_return_rate_rally_block": 0.5,
        "virtual_return_rate_block": 0.5,
        "virtual_hit_rate_block": 0.5,
        "virtual_return_rate_rally_block_legacy": 0.5,
    }
    assert set(command.metrics) == set(expected)
    for name, value in expected.items():
        assert command.metrics[name].item() == pytest.approx(value)


def test_report_boundary_accepts_inference_tensor_slots_and_metrics():
    command = _report_command()
    with torch.inference_mode():
        command.metrics = {
            name: value.clone() for name, value in command.metrics.items()
        }
        command._action_ball_diagnostic_device_telemetry_add(
            "swing",
            ("_swing_starts_acc", None),
            torch.tensor(2.0),
        )

    command.materialize_action_ball_diagnostic_metrics_for_report()

    assert command._swing_starts_acc == pytest.approx(4.0)
    assert command.metrics["swing_completion_rate"].item() == pytest.approx(
        0.25
    )
    slot = command._action_ball_diagnostic_device_telemetry["swing"][
        ("_swing_starts_acc", None)
    ]
    assert slot.item() == pytest.approx(0.0)


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
    swing = inspect.getsource(command_type._count_swing_starts)
    book = inspect.getsource(command_type._vb_book_strike_step)
    metrics = inspect.getsource(command_type._update_metrics)
    consume = inspect.getsource(
        command_type.consume_exact_behavior_decision_counters
    )
    materialize = inspect.getsource(
        command_type.materialize_action_ball_diagnostic_metrics_for_report
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
    assert (
        "_action_ball_diagnostic_device_telemetry_add("
        in swing
    )
    assert (
        "_action_ball_diagnostic_device_telemetry_recur("
        in book
    )
    assert "_pending_hold_recovery_metric_scalars" not in metrics
    assert (
        "self.materialize_action_ball_diagnostic_metrics_for_report()"
        in consume
    )
    assert "_refresh_action_ball_diagnostic_public_metrics()" in materialize
