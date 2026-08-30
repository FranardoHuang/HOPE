"""Exact/hold command metric D2H and FullMDP device-recurrence parity tests.

The fresh diagnostic path keeps the two presentation-only packets in one float64 device state and
materializes it once per PPO update; consumer-bearing recipes retain the immediate Python path.
These tests intentionally run without Isaac by reusing the repository's module stubs; CUDA
synchronization and end-to-end fixed-tape acceptance remain Pod checks.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_reward_flags_mdp import hope_commands_mod  # noqa: E402


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
    # Consumer-bearing fallback recipes retain the Python-double per-step recurrence.
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


def _full_mdp_device_metric_command(num_envs: int = 4):
    command_type = hope_commands_mod.RacketTargetCommand
    command = command_type.__new__(command_type)
    command.cfg = types.SimpleNamespace(
        target_mode="action_ball_full_mdp",
        action_ball_diagnostic_unauthorized=True,
        adaptive_sigma=False,
        adaptive_sigma_monotonic=False,
        adaptive_sigma_normal=False,
        virtual_ball=False,
        vb_metrics_only=False,
        shadow_ball=False,
        physical_ball=False,
        exact_success_min_count=1.0,
        rally_legacy_metrics=False,
    )
    command._action_ball_full_mdp_enabled = True
    command._action_ball_enabled = False
    command._task_first_enabled = False
    command._action_ball_full_mdp_device_r05_owner = object()
    command._action_ball_full_mdp_racket_epoch_owner = object()
    command._shadow = None
    command._physical = None
    command._clip_names = {0: "fixed_action"}
    for attribute in command._action_ball_full_mdp_exact_global_metric_attributes():
        setattr(command, attribute, 0.0)
    for attribute in command._action_ball_full_mdp_exact_bucket_metric_attributes():
        setattr(command, attribute, {0: 0.0})
    for attribute, value in zip(
        command._action_ball_full_mdp_hold_metric_attributes(),
        (0.25, 1.0, 0.5, 0.125, 1.0),
    ):
        setattr(command, attribute, value)
    command._exact_composite_rate = 0.0
    command._swing_starts_acc = 12.0
    command._swing_starts_acc_c = {0: 8.0}

    global_names = (
        "swing_completion_rate",
        "strike_composite_success_exact",
        "strike_pos_pass_exact",
        "strike_vel_pass_exact",
        "strike_normal_pass_exact",
        "exact_strike_pos_success_5cm",
        "exact_strike_pos_success_10cm",
        "exact_strike_sample_count_decayed",
        "strike_pos_target_eligible_sample_count_decayed",
        "strike_vel_target_eligible_sample_count_decayed",
        "strike_normal_target_eligible_sample_count_decayed",
        "strike_composite_target_eligible_sample_count_decayed",
        "base_heading_abs_at_swing_start",
        "base_heading_hold_expiry_count",
        "heading_recovery_spawn_yaw",
        "heading_recovery_expiry_yaw",
        "heading_recovery_count",
    )
    bucket_names = (
        "swing_completion_rate_fixed_action",
        "strike_pos_pass_exact_fixed_action",
        "strike_vel_pass_exact_fixed_action",
        "strike_normal_pass_exact_fixed_action",
        "strike_composite_success_exact_fixed_action",
        "racket_pos_error_exact_strike_fixed_action",
        "racket_vel_error_exact_strike_fixed_action",
        "racket_normal_error_deg_exact_strike_fixed_action",
        "strike_pos_target_eligible_sample_count_decayed_fixed_action",
        "strike_vel_target_eligible_sample_count_decayed_fixed_action",
        "strike_normal_target_eligible_sample_count_decayed_fixed_action",
        "strike_composite_target_eligible_sample_count_decayed_fixed_action",
    )
    command.metrics = {
        name: torch.full((num_envs,), -123.0, dtype=torch.float32)
        for name in global_names + bucket_names
    }
    return command


def _manual_metric_step(state, row, hold, decay):
    before_bucket = list(state[10:18])
    for index in range(10):
        state[index] = decay * state[index] + float(row[index])
    for index in range(18, 23):
        state[index] = decay * state[index]

    exact_n = state[0]
    exact_scale = 1.0 / max(exact_n, 1.0e-6) if exact_n >= 1.0 else 0.0
    bucket_n_before = before_bucket[0]
    hold_heading_scale = (
        1.0 / max(state[19], 1.0e-6) if state[19] >= 1.0 else 0.0
    )
    hold_recovery_scale = (
        1.0 / max(state[22], 1.0e-6) if state[22] >= 1.0 else 0.0
    )
    public = {
        "swing_completion_rate": min(exact_n / 12.0, 1.0),
        "swing_completion_rate_fixed_action": min(
            bucket_n_before / 8.0, 1.0
        ),
        "strike_composite_success_exact": state[1] * exact_scale,
        "strike_pos_pass_exact": state[2] * exact_scale,
        "strike_vel_pass_exact": state[3] * exact_scale,
        "exact_strike_pos_success_5cm": state[4] * exact_scale,
        "exact_strike_pos_success_10cm": state[5] * exact_scale,
        "strike_normal_pass_exact": state[6] * exact_scale,
        "exact_strike_sample_count_decayed": exact_n,
        "base_heading_abs_at_swing_start": state[18] * hold_heading_scale,
        "base_heading_hold_expiry_count": state[19],
        "heading_recovery_spawn_yaw": state[20] * hold_recovery_scale,
        "heading_recovery_expiry_yaw": state[21] * hold_recovery_scale,
        "heading_recovery_count": state[22],
    }
    for channel in ("pos", "vel", "normal", "composite"):
        public[f"strike_{channel}_target_eligible_sample_count_decayed"] = exact_n

    for index in range(8):
        state[10 + index] = decay * state[10 + index] + float(row[10 + index])
    bucket_n = state[10]
    bucket_scale = 1.0 / max(bucket_n, 1.0e-6) if bucket_n >= 1.0 else 0.0
    for name, value in (
        ("strike_pos_pass_exact_fixed_action", state[11] * bucket_scale),
        ("strike_vel_pass_exact_fixed_action", state[12] * bucket_scale),
        ("strike_normal_pass_exact_fixed_action", state[13] * bucket_scale),
        ("strike_composite_success_exact_fixed_action", state[14] * bucket_scale),
        ("racket_pos_error_exact_strike_fixed_action", state[15] * bucket_scale),
        ("racket_vel_error_exact_strike_fixed_action", state[16] * bucket_scale),
        (
            "racket_normal_error_deg_exact_strike_fixed_action",
            state[17] * bucket_scale,
        ),
    ):
        public[name] = value
    for channel in ("pos", "vel", "normal", "composite"):
        public[
            f"strike_{channel}_target_eligible_sample_count_decayed_fixed_action"
        ] = bucket_n

    for index in range(5):
        state[18 + index] += float(hold[index])
    return public


def _device_metric_step(command, row, hold, decay):
    packed = command._begin_action_ball_full_mdp_exact_metric_device_step(
        row.unbind(), decay=decay, bucket_order=(0,)
    )
    command._decay_action_ball_full_mdp_hold_metric_device_state(decay)
    command._publish_action_ball_full_mdp_pre_bucket_device_metrics()
    command._publish_action_ball_full_mdp_exact_global_device_metrics()
    command._finish_action_ball_full_mdp_exact_metric_device_step(
        packed, decay=decay, bucket_order=(0,)
    )
    command._publish_action_ball_full_mdp_exact_bucket_device_metrics()
    command._add_action_ball_full_mdp_hold_metric_device_values(hold.unbind())


def test_full_mdp_device_fixed_tape_preserves_public_reset_timing_and_boundary_snapshot(
    monkeypatch,
):
    command = _full_mdp_device_metric_command()
    manual = [0.0] * 18 + [0.25, 1.0, 0.5, 0.125, 1.0]
    learning_state = {
        "action": torch.arange(8, dtype=torch.float32),
        "reward": torch.arange(4, dtype=torch.float32),
        "observation": torch.arange(12, dtype=torch.float32),
        "reset": torch.tensor([False, True, False, False]),
        "rng": torch.random.get_rng_state().clone(),
    }
    learning_before = {
        name: value.clone() for name, value in learning_state.items()
    }
    command._fixed_tape_learning_state = learning_state

    for step in range(48):
        count = float(1 + (step % 3)) if step % 7 == 0 else 0.0
        passes = (
            count,
            count if step % 2 == 0 else 0.0,
            count,
            count if step % 3 else 0.0,
            count if step % 2 == 0 else 0.0,
            count,
            count if step % 5 else 0.0,
            count * (0.01 + step * 0.001),
            count * (0.1 + step * 0.002),
            count * (0.02 + step * 0.001),
        )
        bucket = (
            passes[0],
            passes[2],
            passes[3],
            passes[6],
            passes[1],
            passes[7],
            passes[8],
            count * (2.0 + step * 0.01),
        )
        row = torch.tensor(passes + bucket, dtype=torch.float32)
        hold = torch.tensor(
            (
                0.25 if step in (3, 19) else 0.0,
                1.0 if step in (3, 19) else 0.0,
                0.5 if step == 19 else 0.0,
                0.125 if step == 19 else 0.0,
                1.0 if step == 19 else 0.0,
            ),
            dtype=torch.float32,
        )
        if step in (11, 32):
            for metric in command.metrics.values():
                metric[torch.tensor([1, 3])] = 0.0
        expected_public = _manual_metric_step(manual, row, hold, 0.99)
        _device_metric_step(command, row, hold, 0.99)
        for name, expected in expected_public.items():
            value = torch.tensor(expected, dtype=torch.float32)
            assert torch.equal(
                command.metrics[name], value.expand_as(command.metrics[name])
            ), (step, name)

    calls = []
    original = hope_commands_mod._batched_host_scalar_values

    def counted(values):
        values = tuple(values)
        calls.append(len(values))
        return original(values)

    monkeypatch.setattr(hope_commands_mod, "_batched_host_scalar_values", counted)
    command.materialize_action_ball_diagnostic_metrics_for_report(
        expected_full_mdp_command_metric_step_counts=(48,)
    )
    command.assert_action_ball_diagnostic_metrics_materialized_for_report()

    assert calls == [23]
    order = command._action_ball_full_mdp_command_metric_order((0,))
    actual = []
    for attribute, bucket in order:
        value = getattr(command, attribute)
        actual.append(value if bucket is None else value[bucket])
    assert tuple(actual) == tuple(manual)
    for name, before in learning_before.items():
        assert torch.equal(command._fixed_tape_learning_state[name], before), name


@pytest.mark.parametrize(
    ("owner_field", "value"),
    (
        ("cfg.target_mode", "action_ball"),
        ("cfg.action_ball_diagnostic_unauthorized", False),
        ("cfg.adaptive_sigma", True),
        ("cfg.virtual_ball", True),
        ("_action_ball_full_mdp_enabled", False),
        ("_action_ball_enabled", True),
        ("_action_ball_full_mdp_device_r05_owner", None),
        ("_action_ball_full_mdp_racket_epoch_owner", None),
        ("_shadow", object()),
        ("_physical", object()),
    ),
)
def test_full_mdp_device_metric_predicate_rejects_real_consumer_counterexamples(
    owner_field, value
):
    command = _full_mdp_device_metric_command()
    assert command._action_ball_full_mdp_device_command_metrics_enabled()
    owner, _, field = owner_field.partition(".")
    if field:
        setattr(getattr(command, owner), field, value)
    else:
        setattr(command, owner, value)
    assert not command._action_ball_full_mdp_device_command_metrics_enabled()


def test_full_mdp_hot_path_routes_both_packets_around_host_materialization():
    command_type = hope_commands_mod.RacketTargetCommand
    update = inspect.getsource(command_type._update_metrics)
    hold = inspect.getsource(command_type._update_hold_recovery_metrics)
    materialize = inspect.getsource(
        command_type._materialize_action_ball_full_mdp_command_metrics
    )

    assert update.index(
        "_begin_action_ball_full_mdp_exact_metric_device_step"
    ) < update.index("_batched_host_scalar_values(")
    assert hold.index(
        "_add_action_ball_full_mdp_hold_metric_device_values"
    ) < hold.index("_batched_host_scalar_values(")
    assert update.count("_batched_host_scalar_values(") == 1
    assert hold.count("_batched_host_scalar_values(") == 1
    assert materialize.count("_batched_host_scalar_values(") == 1


def test_full_mdp_lean_runtime_drains_command_metrics_before_epoch_packet():
    runtime_path = os.path.join(
        os.path.dirname(hope_commands_mod.__file__),
        "action_ball_full_mdp_lean_runtime.py",
    )
    with open(runtime_path, encoding="utf-8") as handle:
        source = handle.read()
    method_start = source.index("    def prepare_pre_optimizer_ppo_boundary(")
    method_end = source.index("    def ", method_start + 8)
    method = source[method_start:method_end]

    command_drain = method.index(
        '"materialize_action_ball_diagnostic_metrics_for_report"'
    )
    epoch_drain = method.index("self._epoch.prepare_drain()")
    assert command_drain < epoch_drain
    assert "completed_delta % num_envs != 0" in method
    assert "(policy_steps, policy_steps + 1)" in method
    assert "expected_full_mdp_command_metric_step_counts" in method
