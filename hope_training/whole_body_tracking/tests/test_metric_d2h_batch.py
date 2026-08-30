"""Exact-strike metric D2H batching parity tests.

The production change keeps every reduction and every Python-float EMA recurrence unchanged.  It
only stacks the already-reduced scalars before one CPU transfer, replacing 10 + 8*N independent
CUDA stream drains.  These tests intentionally run without Isaac by reusing the repository's
module stubs; the CUDA synchronization/profile acceptance is run on a Pod.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_reward_flags_mdp import hope_commands_mod  # noqa: E402


def test_full_mdp_metric_deferral_recipe_rejects_a_real_rollout_consumer():
    cfg = SimpleNamespace(
        action_ball_diagnostic_unauthorized=True,
        adaptive_sigma=False,
        adaptive_sigma_monotonic=False,
        adaptive_sigma_normal=False,
        virtual_ball=False,
        vb_metrics_only=False,
        shadow_ball=False,
        physical_ball=False,
        achieved_target_mix_prob=0.0,
    )
    predicate = hope_commands_mod._full_mdp_command_metric_deferral_recipe
    assert predicate(cfg, {0: "take061"}, True) is True
    cfg.adaptive_sigma = True
    assert predicate(cfg, {0: "take061"}, True) is False


def _exact_reductions(
    *,
    exact: torch.Tensor,
    pos_err: torch.Tensor,
    vel_err: torch.Tensor,
    normal_err_rad: torch.Tensor,
    families: torch.Tensor,
    num_buckets: int,
) -> list[torch.Tensor]:
    normal_err_deg = normal_err_rad * (180.0 / math.pi)
    pass_pos = (pos_err < 0.075) & exact
    pass_vel = (vel_err < 0.5) & exact
    pass_normal = (normal_err_deg < 15.0) & exact
    pass_comp = pass_pos & pass_vel & pass_normal
    pass_5cm = (pos_err < 0.05) & exact
    pass_10cm = (pos_err < 0.10) & exact
    values = [
        exact.sum(dtype=pos_err.dtype),
        pass_comp.sum(dtype=pos_err.dtype),
        pass_pos.sum(dtype=pos_err.dtype),
        pass_vel.sum(dtype=pos_err.dtype),
        pass_5cm.sum(dtype=pos_err.dtype),
        pass_10cm.sum(dtype=pos_err.dtype),
        pass_normal.sum(dtype=pos_err.dtype),
        (pos_err * exact).sum(),
        (vel_err * exact).sum(),
        (normal_err_rad * exact).sum(),
    ]
    for bucket in range(num_buckets):
        selected = exact & (families == bucket)
        selected_float = selected.float()
        values.extend(
            (
                selected.sum(dtype=pos_err.dtype),
                (pass_pos & selected).sum(dtype=pos_err.dtype),
                (pass_vel & selected).sum(dtype=pos_err.dtype),
                (pass_normal & selected).sum(dtype=pos_err.dtype),
                (pass_comp & selected).sum(dtype=pos_err.dtype),
                (pos_err * selected_float).sum(),
                (vel_err * selected_float).sum(),
                (normal_err_deg * selected_float).sum(),
            )
        )
    return values


@pytest.mark.parametrize("num_buckets", (1, 5, 73))
@pytest.mark.parametrize("has_exact_samples", (False, True))
def test_batched_host_scalars_equal_individual_float_reads(
    num_buckets: int, has_exact_samples: bool
):
    num_envs = 4096
    rows = torch.arange(num_envs)
    families = rows.remainder(num_buckets)
    exact = rows.remainder(41).eq(0) if has_exact_samples else torch.zeros(
        num_envs, dtype=torch.bool
    )
    # Nontrivial float32 values exercise the same post-reduction float32 -> Python-double
    # conversion used by the production error sums.
    pos_err = (rows.float().remainder(113) + 0.25) / 1000.0
    vel_err = (rows.float().remainder(127) + 0.5) / 100.0
    normal_err_rad = (rows.float().remainder(89) + 0.75) / 100.0
    reductions = _exact_reductions(
        exact=exact,
        pos_err=pos_err,
        vel_err=vel_err,
        normal_err_rad=normal_err_rad,
        families=families,
        num_buckets=num_buckets,
    )

    legacy = tuple(float(value) for value in reductions)
    batched = hope_commands_mod._batched_host_scalar_values(reductions)

    assert batched == legacy
    assert all(type(value) is float for value in batched)


def test_batched_values_preserve_python_ema_recurrence_exactly():
    exact = torch.tensor([True, False, True, True])
    pos_err = torch.tensor([0.01, 0.50, 0.08, 0.12], dtype=torch.float32)
    vel_err = torch.tensor([0.20, 0.30, 0.60, 0.40], dtype=torch.float32)
    normal_err_rad = torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=torch.float32)
    reductions = _exact_reductions(
        exact=exact,
        pos_err=pos_err,
        vel_err=vel_err,
        normal_err_rad=normal_err_rad,
        families=torch.tensor([0, 1, 0, 1]),
        num_buckets=2,
    )
    legacy_values = tuple(float(value) for value in reductions)
    batched_values = hope_commands_mod._batched_host_scalar_values(reductions)
    old = tuple(0.125 * (index + 1) for index in range(len(reductions)))
    decay = 0.99

    legacy_ema = tuple(decay * before + value for before, value in zip(old, legacy_values))
    batched_ema = tuple(decay * before + value for before, value in zip(old, batched_values))

    assert batched_ema == legacy_ema


@pytest.mark.parametrize("num_envs", (0, 4096, 8192))
def test_float32_boolean_counts_match_the_legacy_int64_host_value(num_envs: int):
    mask = torch.arange(num_envs).remainder(3).eq(0)

    legacy = float(mask.sum())
    batched_count = float(mask.sum(dtype=torch.float32))

    assert num_envs < 2**24
    assert batched_count == legacy


def test_batched_host_scalar_contract_rejects_non_scalars():
    with pytest.raises(ValueError, match="scalar tensors"):
        hope_commands_mod._batched_host_scalar_values([torch.zeros(2)])
    with pytest.raises(ValueError, match="scalar tensors"):
        hope_commands_mod._batched_host_scalar_values([1.0])
    with pytest.raises(ValueError, match="common dtype and device"):
        hope_commands_mod._batched_host_scalar_values(
            [torch.tensor(1, dtype=torch.int64), torch.tensor(1.0)]
        )
    assert hope_commands_mod._batched_host_scalar_values([]) == ()


def test_update_metrics_batches_only_the_targeted_exact_reductions():
    source = inspect.getsource(
        hope_commands_mod.RacketTargetCommand._update_metrics
    )

    assert source.count("_batched_host_scalar_values(") == 1
    for retired_individual_read in (
        "float(exact_strike.sum())",
        "float(pass_comp.sum())",
        "float(pass_pos.sum())",
        "float(pass_vel.sum())",
        "float(_pass_5cm.sum())",
        "float(_pass_10cm.sum())",
        "float(pass_normal.sum())",
        "float((pos_err * exact_strike).sum())",
        "float((vel_err * exact_strike).sum())",
        "float((normal_err_rad * exact_strike).sum())",
        "float(_sel.sum())",
        "float((pass_pos & _sel).sum())",
        "float((pass_vel & _sel).sum())",
        "float((pass_normal & _sel).sum())",
        "float((pass_comp & _sel).sum())",
    ):
        assert retired_individual_read not in source
    # The behavior-coupled EMA remains a Python-float, per-step recurrence; this is not a
    # device-only or rollout-boundary semantic change.
    assert (
        "self._exact_n_acc = decay * self._exact_n_acc "
        "+ next(_exact_metric_values)"
    ) in source
    assert "self._update_adaptive_sigma(" in source
    assert "_composite_target_enough, _composite_target_denom" in source
    assert "self._curr_perturb_scale + float(" in source


def test_per_clip_batched_values_keep_the_legacy_state_transition_order():
    source = inspect.getsource(
        hope_commands_mod.RacketTargetCommand._update_metrics
    )

    buffered = source.index("_exact_metric_bucket_values = {")
    completion_report = source.index(
        'self.metrics[f"swing_completion_rate_{_cn}"]'
    )
    per_clip_update = source.index(
        "self._exact_n_acc_c[_c] = (",
        buffered,
    )
    exact_quality_report = source.index(
        'self.metrics[f"strike_pos_pass_exact_{_cn}"]'
    )

    # The single D2H happens early, but the per-clip EMA mutation stays after the historic
    # swing-completion read and immediately before the exact-quality report.
    assert buffered < completion_report < per_clip_update < exact_quality_report


def _full_mdp_metric_boundary_command(device):
    command = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    command.device = device
    command.cfg = SimpleNamespace(exact_success_min_count=1.0)
    command._action_ball_full_mdp_command_metrics_device_enabled = True
    command._action_ball_full_mdp_command_metric_state = torch.arange(
        1, 24, dtype=torch.float64, device=device
    )
    command._action_ball_full_mdp_command_metric_pending_row = None
    command._action_ball_full_mdp_command_metric_step_count = 49
    for name in (
        "_exact_n_acc", "_exact_pass_comp_acc", "_exact_pass_pos_acc",
        "_exact_pass_vel_acc", "_exact_pass_5cm_acc", "_exact_pass_10cm_acc",
        "_exact_pass_normal_acc", "_exact_pos_err_sum", "_exact_vel_err_sum",
        "_exact_nrm_err_sum", "_heading_expiry_sum_acc", "_heading_expiry_n_acc",
        "_recovery_spawn_sum_acc", "_recovery_expiry_sum_acc", "_recovery_n_acc",
    ):
        setattr(command, name, -1.0)
    for name in (
        "_exact_n_acc_c", "_exact_pass_pos_acc_c", "_exact_pass_vel_acc_c",
        "_exact_pass_normal_acc_c", "_exact_pass_comp_acc_c",
        "_exact_pos_err_sum_c", "_exact_vel_err_sum_c", "_exact_nrm_err_sum_c",
    ):
        setattr(command, name, {0: -1.0})
    command._exact_composite_rate = -1.0
    return command


def _full_mdp_metric_host_snapshot(command):
    return (
        command._exact_n_acc,
        command._exact_pass_comp_acc,
        command._exact_n_acc_c[0],
        command._heading_expiry_sum_acc,
        command._recovery_n_acc,
        command._exact_composite_rate,
    )


def test_full_mdp_metric_boundary_validates_before_host_mutation():
    command = _full_mdp_metric_boundary_command(torch.device("cpu"))
    before = _full_mdp_metric_host_snapshot(command)
    with pytest.raises(RuntimeError, match="PPO span differs"):
        command._materialize_action_ball_full_mdp_command_metrics(48)
    assert command._action_ball_full_mdp_command_metric_step_count == 49
    assert _full_mdp_metric_host_snapshot(command) == before

    command._exact_n_acc_c.pop(0)
    with pytest.raises(RuntimeError, match="bucket key is missing"):
        command._materialize_action_ball_full_mdp_command_metrics(49)
    assert command._action_ball_full_mdp_command_metric_step_count == 49
    command._exact_n_acc_c[0] = -1.0
    assert _full_mdp_metric_host_snapshot(command) == before

    state_before = command._action_ball_full_mdp_command_metric_state.clone()
    with pytest.raises(RuntimeError, match="complete device 0-D tensors"):
        command._action_ball_full_mdp_command_metric_row(
            (torch.zeros(1),) * 5, width=5
        )
    assert torch.equal(command._action_ball_full_mdp_command_metric_state, state_before)


def test_full_mdp_metric_boundary_success_commits_fixed_snapshot():
    command = _full_mdp_metric_boundary_command(torch.device("cpu"))
    command._action_ball_full_mdp_command_metric_state[0] = 3.21
    command._action_ball_full_mdp_command_metric_state[1] = 0.1
    command._materialize_action_ball_full_mdp_command_metrics(49)
    assert command._action_ball_full_mdp_command_metric_step_count == 0
    assert command._exact_n_acc == 3.21
    assert command._exact_n_acc_c[0] == 11.0
    assert command._heading_expiry_sum_acc == 19.0
    assert command._recovery_n_acc == 23.0
    assert command._exact_composite_rate == 0.1 / 3.21
    assert command._exact_composite_rate != 0.1 * (1.0 / 3.21)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA profiler unavailable")
def test_full_mdp_metric_boundary_cuda_counts_physical_d2h_and_failure_order():
    device = torch.device("cuda:0")

    def transfer_count(command, expected, error=None):
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            if error is None:
                command._materialize_action_ball_full_mdp_command_metrics(expected)
            else:
                with pytest.raises(RuntimeError, match=error):
                    command._materialize_action_ball_full_mdp_command_metrics(expected)
            torch.cuda.synchronize(device)
        return sum(
            event.name.startswith("Memcpy DtoH")
            for event in profiler.events()
        )

    mismatch = _full_mdp_metric_boundary_command(device)
    assert transfer_count(mismatch, 48, "PPO span differs") == 0
    assert mismatch._action_ball_full_mdp_command_metric_step_count == 49

    nonfinite = _full_mdp_metric_boundary_command(device)
    nonfinite._action_ball_full_mdp_command_metric_state[7] = float("nan")
    before = _full_mdp_metric_host_snapshot(nonfinite)
    assert transfer_count(nonfinite, 49, "non-finite") == 1
    assert nonfinite._action_ball_full_mdp_command_metric_step_count == 49
    assert _full_mdp_metric_host_snapshot(nonfinite) == before

    success = _full_mdp_metric_boundary_command(device)
    assert transfer_count(success, 49) == 1
    assert success._action_ball_full_mdp_command_metric_step_count == 0


def test_full_mdp_metric_hot_path_keeps_host_packets_off_active_branch():
    update = inspect.getsource(hope_commands_mod.RacketTargetCommand._update_metrics)
    hold = inspect.getsource(
        hope_commands_mod.RacketTargetCommand._update_hold_recovery_metrics
    )
    materialize = inspect.getsource(
        hope_commands_mod.RacketTargetCommand._materialize_action_ball_full_mdp_command_metrics
    )
    assert update.index("command_metric_state[:10]") < update.index(
        "_batched_host_scalar_values(_exact_metric_tensors)"
    )
    assert hold.index("command_metric_state[18:].add_") < hold.index(
        "_batched_host_scalar_values(metric_scalars)"
    )
    assert '.to(device="cpu", non_blocking=False)' in materialize
    assert "reciprocal" not in update
