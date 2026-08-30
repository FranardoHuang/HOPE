"""Command metric D2H batching and update-boundary parity tests.

The production change keeps every reduction and every Python-float EMA recurrence unchanged.  The
FullMDP diagnostic path retains chronological exact-quality and hold/recovery rows on device and
transfers each fixed-width tape once at the PPO boundary; configurations with an immediate
consumer keep the existing control-step path.  The CUDA synchronization and wall-time acceptance
remains a Pod-only check.
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


def _deferred_exact_command(num_buckets: int = 1):
    command_type = hope_commands_mod.RacketTargetCommand
    command = command_type.__new__(command_type)
    command.cfg = types.SimpleNamespace(
        adaptive_sigma=False,
        adaptive_sigma_monotonic=False,
        adaptive_sigma_normal=False,
        target_mode="action_ball_full_mdp",
        ref_perturb_success_gated=True,
        exact_success_min_count=1.0,
        virtual_ball=False,
        vb_metrics_only=False,
        shadow_ball=False,
        physical_ball=False,
        rally_legacy_metrics=False,
    )
    command._action_ball_full_mdp_enabled = True
    command._action_ball_enabled = False
    command._task_first_enabled = False
    command._action_ball_diagnostic_unauthorized = True
    command._action_ball_full_mdp_device_r05_owner = object()
    command._action_ball_full_mdp_racket_epoch_owner = object()
    command._shadow = None
    command._physical = None
    command._action_ball_target_validity_mask = (True, True, True)
    command._clip_names = {index: f"clip{index}" for index in range(num_buckets)}
    for attribute in (
        "_exact_n_acc",
        "_exact_pass_comp_acc",
        "_exact_pass_pos_acc",
        "_exact_pass_vel_acc",
        "_exact_pass_5cm_acc",
        "_exact_pass_10cm_acc",
        "_exact_pass_normal_acc",
        "_exact_pos_err_sum",
        "_exact_vel_err_sum",
        "_exact_nrm_err_sum",
    ):
        setattr(command, attribute, 0.0)
    for attribute in (
        "_exact_n_acc_c",
        "_exact_pass_pos_acc_c",
        "_exact_pass_vel_acc_c",
        "_exact_pass_normal_acc_c",
        "_exact_pass_comp_acc_c",
        "_exact_pos_err_sum_c",
        "_exact_vel_err_sum_c",
        "_exact_nrm_err_sum_c",
    ):
        setattr(command, attribute, {index: 0.0 for index in range(num_buckets)})
    command._exact_composite_rate = 0.0
    command._swing_starts_acc = 8.0
    command._prestrike_fall_acc = 0.0
    command._poststrike_fall_acc = 0.0
    command._drift_n_acc = 0.0
    command._drift_sum_acc = 0.0
    command._drift_fwd_sum_acc = 0.0
    command._station_offset_start_sum_acc = 0.0
    command._heading_expiry_n_acc = 0.0
    command._heading_expiry_sum_acc = 0.0
    command._recovery_n_acc = 0.0
    command._recovery_spawn_sum_acc = 0.0
    command._recovery_expiry_sum_acc = 0.0
    command._rally_starts_acc = 0.0
    command._rally_returns_acc = 0.0
    command._vb_exact_acc = 0.0
    command._vb_hit_acc = 0.0
    command._vb_net_acc = 0.0
    command._vb_land_valid_acc = 0.0
    command._vb_inb_acc = 0.0
    command._swing_starts_acc_c = {
        index: 4.0 for index in range(num_buckets)
    }
    for attribute in (
        "_prestrike_fall_acc_c",
        "_poststrike_fall_acc_c",
        "_rally_starts_acc_c",
        "_rally_returns_acc_c",
        "_vb_exact_acc_c",
        "_vb_hit_acc_c",
        "_vb_inb_acc_c",
    ):
        setattr(command, attribute, {index: 0.0 for index in range(num_buckets)})
    global_metrics = (
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
    )
    bucket_metrics = (
        "swing_completion_rate",
        "strike_pos_pass_exact",
        "strike_vel_pass_exact",
        "strike_normal_pass_exact",
        "strike_composite_success_exact",
        "racket_pos_error_exact_strike",
        "racket_vel_error_exact_strike",
        "racket_normal_error_deg_exact_strike",
        "strike_pos_target_eligible_sample_count_decayed",
        "strike_vel_target_eligible_sample_count_decayed",
        "strike_normal_target_eligible_sample_count_decayed",
        "strike_composite_target_eligible_sample_count_decayed",
    )
    metric_names = list(global_metrics)
    for bucket_name in command._clip_names.values():
        metric_names.extend(
            f"{name}_{bucket_name}" for name in bucket_metrics
        )
    command.metrics = {name: torch.zeros(1) for name in metric_names}
    return command


def test_full_mdp_exact_rows_replay_python_ema_in_chronological_order(monkeypatch):
    command = _deferred_exact_command(num_buckets=1)
    width = 18
    first = torch.arange(1, width + 1, dtype=torch.float32)
    second = torch.arange(101, 101 + width, dtype=torch.float32)
    calls = []
    original = hope_commands_mod._batched_host_scalar_rows

    def counted(rows):
        rows = tuple(rows)
        calls.append(len(rows))
        return original(rows)

    monkeypatch.setattr(hope_commands_mod, "_batched_host_scalar_rows", counted)
    command._stage_action_ball_full_mdp_exact_metric_row(
        first.unbind(), decay=0.9, bucket_order=(0,)
    )
    command._stage_action_ball_full_mdp_exact_metric_row(
        second.unbind(), decay=0.8, bucket_order=(0,)
    )

    assert command._exact_n_acc == 0.0
    assert calls == []
    assert command._flush_action_ball_full_mdp_exact_metric_rows()
    assert calls == [2]
    expected = tuple(0.8 * (0.9 * 0.0 + float(a)) + float(b) for a, b in zip(first, second))
    globals_in_order = (
        command._exact_n_acc,
        command._exact_pass_comp_acc,
        command._exact_pass_pos_acc,
        command._exact_pass_vel_acc,
        command._exact_pass_5cm_acc,
        command._exact_pass_10cm_acc,
        command._exact_pass_normal_acc,
        command._exact_pos_err_sum,
        command._exact_vel_err_sum,
        command._exact_nrm_err_sum,
    )
    assert globals_in_order == pytest.approx(expected[:10], abs=0.0)
    assert command._exact_n_acc_c[0] == pytest.approx(expected[10], abs=0.0)
    assert not command._flush_action_ball_full_mdp_exact_metric_rows()


def test_full_mdp_exact_rows_preserve_every_global_and_bucket_float_transition():
    command = _deferred_exact_command(num_buckets=1)
    global_attributes = (
        "_exact_n_acc",
        "_exact_pass_comp_acc",
        "_exact_pass_pos_acc",
        "_exact_pass_vel_acc",
        "_exact_pass_5cm_acc",
        "_exact_pass_10cm_acc",
        "_exact_pass_normal_acc",
        "_exact_pos_err_sum",
        "_exact_vel_err_sum",
        "_exact_nrm_err_sum",
    )
    bucket_attributes = (
        "_exact_n_acc_c",
        "_exact_pass_pos_acc_c",
        "_exact_pass_vel_acc_c",
        "_exact_pass_normal_acc_c",
        "_exact_pass_comp_acc_c",
        "_exact_pos_err_sum_c",
        "_exact_vel_err_sum_c",
        "_exact_nrm_err_sum_c",
    )
    before = tuple(0.125 * (index + 1) for index in range(18))
    for attribute, value in zip(global_attributes, before[:10]):
        setattr(command, attribute, value)
    for attribute, value in zip(bucket_attributes, before[10:]):
        getattr(command, attribute)[0] = value

    first = torch.arange(1, 19, dtype=torch.float32)
    second = torch.arange(101, 119, dtype=torch.float32)
    command._stage_action_ball_full_mdp_exact_metric_row(
        first.unbind(), decay=0.5, bucket_order=(0,)
    )
    command._stage_action_ball_full_mdp_exact_metric_row(
        second.unbind(), decay=0.25, bucket_order=(0,)
    )
    command.materialize_action_ball_diagnostic_metrics_for_report(
        expected_full_mdp_exact_row_counts=(2,)
    )

    actual = tuple(getattr(command, name) for name in global_attributes) + tuple(
        getattr(command, name)[0] for name in bucket_attributes
    )
    expected = tuple(
        0.25 * (0.5 * old + float(a)) + float(b)
        for old, a, b in zip(before, first, second)
    )
    assert actual == expected


def test_full_mdp_deferred_fixed_tape_matches_independent_public_formulas():
    deferred = _deferred_exact_command(num_buckets=2)
    immediate = _deferred_exact_command(num_buckets=2)
    global_attributes = (
        "_exact_n_acc",
        "_exact_pass_comp_acc",
        "_exact_pass_pos_acc",
        "_exact_pass_vel_acc",
        "_exact_pass_5cm_acc",
        "_exact_pass_10cm_acc",
        "_exact_pass_normal_acc",
        "_exact_pos_err_sum",
        "_exact_vel_err_sum",
        "_exact_nrm_err_sum",
    )
    bucket_attributes = (
        "_exact_n_acc_c",
        "_exact_pass_pos_acc_c",
        "_exact_pass_vel_acc_c",
        "_exact_pass_normal_acc_c",
        "_exact_pass_comp_acc_c",
        "_exact_pos_err_sum_c",
        "_exact_vel_err_sum_c",
        "_exact_nrm_err_sum_c",
    )
    bucket_order = (0, 1)
    rows = (
        (0.9, torch.arange(1, 27, dtype=torch.float32)),
        (0.8, torch.arange(101, 127, dtype=torch.float32)),
        (0.7, torch.arange(201, 227, dtype=torch.float32)),
    )

    for decay, row in rows:
        deferred._stage_action_ball_full_mdp_exact_metric_row(
            row.unbind(), decay=decay, bucket_order=bucket_order
        )
        values = iter(float(value) for value in row)
        for attribute in global_attributes:
            setattr(
                immediate,
                attribute,
                decay * getattr(immediate, attribute) + next(values),
            )
        for bucket in bucket_order:
            for attribute in bucket_attributes:
                accumulators = getattr(immediate, attribute)
                accumulators[bucket] = (
                    decay * accumulators[bucket] + next(values)
                )

    deferred.materialize_action_ball_diagnostic_metrics_for_report(
        expected_full_mdp_exact_row_counts=(3,)
    )
    n = immediate._exact_n_acc
    expected = {
        "swing_completion_rate": min(n / immediate._swing_starts_acc, 1.0),
        "strike_composite_success_exact": immediate._exact_pass_comp_acc / n,
        "strike_pos_pass_exact": immediate._exact_pass_pos_acc / n,
        "strike_vel_pass_exact": immediate._exact_pass_vel_acc / n,
        "strike_normal_pass_exact": immediate._exact_pass_normal_acc / n,
        "exact_strike_pos_success_5cm": immediate._exact_pass_5cm_acc / n,
        "exact_strike_pos_success_10cm": immediate._exact_pass_10cm_acc / n,
        "exact_strike_sample_count_decayed": n,
        "strike_pos_target_eligible_sample_count_decayed": n,
        "strike_vel_target_eligible_sample_count_decayed": n,
        "strike_normal_target_eligible_sample_count_decayed": n,
        "strike_composite_target_eligible_sample_count_decayed": n,
    }
    for bucket in bucket_order:
        suffix = immediate._clip_names[bucket]
        bucket_n = immediate._exact_n_acc_c[bucket]
        expected.update(
            {
                f"swing_completion_rate_{suffix}": min(
                    bucket_n / immediate._swing_starts_acc_c[bucket], 1.0
                ),
                f"strike_pos_pass_exact_{suffix}": (
                    immediate._exact_pass_pos_acc_c[bucket] / bucket_n
                ),
                f"strike_vel_pass_exact_{suffix}": (
                    immediate._exact_pass_vel_acc_c[bucket] / bucket_n
                ),
                f"strike_normal_pass_exact_{suffix}": (
                    immediate._exact_pass_normal_acc_c[bucket] / bucket_n
                ),
                f"strike_composite_success_exact_{suffix}": (
                    immediate._exact_pass_comp_acc_c[bucket] / bucket_n
                ),
                f"racket_pos_error_exact_strike_{suffix}": (
                    immediate._exact_pos_err_sum_c[bucket] / bucket_n
                ),
                f"racket_vel_error_exact_strike_{suffix}": (
                    immediate._exact_vel_err_sum_c[bucket] / bucket_n
                ),
                f"racket_normal_error_deg_exact_strike_{suffix}": (
                    immediate._exact_nrm_err_sum_c[bucket] / bucket_n
                ),
                f"strike_pos_target_eligible_sample_count_decayed_{suffix}": bucket_n,
                f"strike_vel_target_eligible_sample_count_decayed_{suffix}": bucket_n,
                f"strike_normal_target_eligible_sample_count_decayed_{suffix}": bucket_n,
                f"strike_composite_target_eligible_sample_count_decayed_{suffix}": bucket_n,
            }
        )

    assert deferred.metrics
    assert set(deferred.metrics) == set(expected)
    for name, value in sorted(expected.items()):
        actual = deferred.metrics[name]
        independent = torch.tensor([value], dtype=actual.dtype)
        assert torch.equal(actual, independent), name


def test_full_mdp_exact_deferral_keeps_immediate_consumers_on_old_path():
    command = _deferred_exact_command()
    assert command._action_ball_full_mdp_deferred_exact_metrics_enabled()
    command.cfg.adaptive_sigma = True
    assert not command._action_ball_full_mdp_deferred_exact_metrics_enabled()
    command.cfg.adaptive_sigma = False
    command.cfg.target_mode = "reference_perturbed"
    assert not command._action_ball_full_mdp_deferred_exact_metrics_enabled()
    command.cfg.target_mode = "action_ball_full_mdp"
    command.cfg.virtual_ball = True
    assert not command._action_ball_full_mdp_deferred_exact_metrics_enabled()
    command.cfg.virtual_ball = False
    command._action_ball_diagnostic_unauthorized = False
    assert not command._action_ball_full_mdp_deferred_exact_metrics_enabled()
    command._action_ball_diagnostic_unauthorized = True
    command._action_ball_full_mdp_device_r05_owner = None
    assert not command._action_ball_full_mdp_deferred_exact_metrics_enabled()


@pytest.mark.parametrize(
    ("row_count", "expected", "accepted"),
    (
        (1, (1,), True),
        (47, (48, 49), False),
        (48, (48, 49), True),
        (49, (48, 49), True),
        (50, (48, 49), False),
    ),
)
def test_full_mdp_exact_drain_checks_span_and_batches_once(
    monkeypatch, row_count: int, expected: tuple[int, ...], accepted: bool
):
    command = _deferred_exact_command()
    calls = []
    original = hope_commands_mod._batched_host_scalar_rows

    def counted(rows):
        rows = tuple(rows)
        calls.append(len(rows))
        return original(rows)

    monkeypatch.setattr(hope_commands_mod, "_batched_host_scalar_rows", counted)
    for index in range(row_count):
        row = torch.arange(18, dtype=torch.float32) + float(index)
        command._stage_action_ball_full_mdp_exact_metric_row(
            row.unbind(), decay=0.99, bucket_order=(0,)
        )

    if not accepted:
        with pytest.raises(RuntimeError, match="rollout row count differs"):
            command.materialize_action_ball_diagnostic_metrics_for_report(
                expected_full_mdp_exact_row_counts=expected
            )
        assert calls == []
        assert len(command._action_ball_full_mdp_pending_exact_metric_rows) == row_count
        return

    command.materialize_action_ball_diagnostic_metrics_for_report(
        expected_full_mdp_exact_row_counts=expected
    )
    command.assert_action_ball_diagnostic_metrics_materialized_for_report()
    command.materialize_action_ball_diagnostic_metrics_for_report()

    assert calls == [row_count]


def test_full_mdp_exact_public_metrics_use_all_true_validity_without_legacy_mask():
    command = _deferred_exact_command()
    del command._action_ball_target_validity_mask
    metric_names = (
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
    )
    command.metrics = {name: torch.zeros(1) for name in metric_names}
    command._exact_n_acc = 4.0
    command._exact_pass_comp_acc = 1.0
    command._exact_pass_pos_acc = 2.0
    command._exact_pass_vel_acc = 3.0
    command._exact_pass_normal_acc = 4.0
    command._exact_pass_5cm_acc = 1.0
    command._exact_pass_10cm_acc = 2.0

    command._refresh_action_ball_exact_public_metrics()

    assert command.metrics["strike_composite_success_exact"].item() == 0.25
    assert command.metrics["strike_pos_pass_exact"].item() == 0.5
    assert command.metrics["strike_vel_pass_exact"].item() == 0.75
    assert command.metrics["strike_normal_pass_exact"].item() == 1.0
    assert command.metrics[
        "strike_composite_target_eligible_sample_count_decayed"
    ].item() == 4.0


def test_full_mdp_exact_rows_reject_nonfinite_values_and_decay():
    command = _deferred_exact_command()
    finite = torch.zeros(18, dtype=torch.float32)
    nonfinite = finite.clone()
    nonfinite[3] = float("nan")

    with pytest.raises(RuntimeError, match="decay differs"):
        command._stage_action_ball_full_mdp_exact_metric_row(
            finite.unbind(), decay=float("nan"), bucket_order=(0,)
        )
    command._stage_action_ball_full_mdp_exact_metric_row(
        nonfinite.unbind(), decay=0.99, bucket_order=(0,)
    )
    with pytest.raises(RuntimeError, match="non-finite scalar"):
        command.materialize_action_ball_diagnostic_metrics_for_report(
            expected_full_mdp_exact_row_counts=(1,)
        )
    assert len(command._action_ball_full_mdp_pending_exact_metric_rows) == 1


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
    # The behavior-coupled EMA remains a Python-float, per-step recurrence; FullMDP
    # replays these exact Python transitions at the drain.
    assert (
        "self._exact_n_acc = _exact_metric_recurrence_decay * self._exact_n_acc "
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


_HOLD_RECOVERY_ATTRIBUTES = (
    "_heading_expiry_sum_acc",
    "_heading_expiry_n_acc",
    "_recovery_spawn_sum_acc",
    "_recovery_expiry_sum_acc",
    "_recovery_n_acc",
)


def _deferred_hold_command():
    command = _deferred_exact_command()
    command._action_ball_full_mdp_pending_hold_recovery_metric_rows = []
    command._hold_edge_pending = torch.zeros(3, dtype=torch.bool)
    command._previous_in_hold = torch.zeros(3, dtype=torch.bool)
    command._hold_start_yaw = torch.zeros(3, dtype=torch.float32)
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(root_quat_w=torch.zeros(3, 4))
    )
    command.robot.data.root_quat_w[:, 0] = 1.0
    for name in (
        "base_heading_abs_at_swing_start",
        "base_heading_hold_expiry_count",
        "heading_recovery_spawn_yaw",
        "heading_recovery_expiry_yaw",
        "heading_recovery_count",
    ):
        command.metrics[name] = torch.full((1,), -1.0)
    return command


def test_full_mdp_hold_recovery_fixed_tape_matches_python_double_and_public_metrics():
    deferred = _deferred_hold_command()
    legacy = _deferred_hold_command()
    legacy._action_ball_full_mdp_enabled = False
    before = (0.125, 1.25, 0.375, 0.625, 2.5)
    for command in (deferred, legacy):
        for attribute, value in zip(_HOLD_RECOVERY_ATTRIBUTES, before):
            setattr(command, attribute, value)

    rows = (
        (0.99, torch.tensor([0.25, 1.0, 0.5, 0.125, 1.0])),
        (0.875, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])),
        (0.5, torch.tensor([0.75, 2.0, 1.25, 0.25, 2.0])),
    )
    for decay, row in rows:
        deferred._stage_action_ball_full_mdp_hold_recovery_metric_row(
            row.unbind(),
            decay=decay,
        )
        for attribute in _HOLD_RECOVERY_ATTRIBUTES:
            setattr(legacy, attribute, decay * getattr(legacy, attribute))
        legacy._apply_hold_recovery_host_values(
            tuple(float(value) for value in row)
        )

    deferred.materialize_action_ball_diagnostic_metrics_for_report()
    legacy._refresh_action_ball_diagnostic_public_metrics()

    assert tuple(
        getattr(deferred, attribute)
        for attribute in _HOLD_RECOVERY_ATTRIBUTES
    ) == tuple(
        getattr(legacy, attribute)
        for attribute in _HOLD_RECOVERY_ATTRIBUTES
    )
    for name in (
        "base_heading_abs_at_swing_start",
        "base_heading_hold_expiry_count",
        "heading_recovery_spawn_yaw",
        "heading_recovery_expiry_yaw",
        "heading_recovery_count",
    ):
        assert torch.equal(deferred.metrics[name], legacy.metrics[name]), name


@pytest.mark.parametrize("row_count", (48, 49))
def test_full_mdp_exact_and_hold_tapes_use_two_update_boundary_reads(
    monkeypatch,
    row_count: int,
):
    command = _deferred_hold_command()
    calls = []
    original = hope_commands_mod._batched_host_scalar_rows

    def counted(rows):
        rows = tuple(rows)
        calls.append((len(rows), int(rows[0].numel())))
        return original(rows)

    monkeypatch.setattr(hope_commands_mod, "_batched_host_scalar_rows", counted)
    for index in range(row_count):
        exact = torch.arange(18, dtype=torch.float32) + float(index)
        hold = torch.arange(5, dtype=torch.float32) + float(index)
        command._stage_action_ball_full_mdp_exact_metric_row(
            exact.unbind(), decay=0.99, bucket_order=(0,)
        )
        command._stage_action_ball_full_mdp_hold_recovery_metric_row(
            hold.unbind(), decay=0.99
        )

    command.materialize_action_ball_diagnostic_metrics_for_report(
        expected_full_mdp_exact_row_counts=(48, 49)
    )
    command.assert_action_ball_diagnostic_metrics_materialized_for_report()
    command.materialize_action_ball_diagnostic_metrics_for_report()

    assert calls == [(row_count, 18), (row_count, 5)]


def test_full_mdp_hold_row_count_mismatch_fails_before_any_transfer(monkeypatch):
    command = _deferred_hold_command()
    calls = []
    monkeypatch.setattr(
        hope_commands_mod,
        "_batched_host_scalar_rows",
        lambda rows: calls.append(tuple(rows)),
    )
    for _ in range(48):
        command._stage_action_ball_full_mdp_exact_metric_row(
            torch.zeros(18).unbind(), decay=0.99, bucket_order=(0,)
        )
    for _ in range(47):
        command._stage_action_ball_full_mdp_hold_recovery_metric_row(
            torch.zeros(5).unbind(), decay=0.99
        )

    with pytest.raises(
        RuntimeError,
        match="exact/hold metric rollout row counts differ",
    ):
        command.materialize_action_ball_diagnostic_metrics_for_report(
            expected_full_mdp_exact_row_counts=(48,)
        )

    assert calls == []
    assert len(command._action_ball_full_mdp_pending_exact_metric_rows) == 48
    assert len(
        command._action_ball_full_mdp_pending_hold_recovery_metric_rows
    ) == 47


def test_full_mdp_exact_and_hold_allowed_span_mismatch_fails_before_transfer(
    monkeypatch,
):
    command = _deferred_hold_command()
    calls = []
    monkeypatch.setattr(
        hope_commands_mod,
        "_batched_host_scalar_rows",
        lambda rows: calls.append(tuple(rows)),
    )
    for _ in range(48):
        command._stage_action_ball_full_mdp_exact_metric_row(
            torch.zeros(18).unbind(), decay=0.99, bucket_order=(0,)
        )
    for _ in range(49):
        command._stage_action_ball_full_mdp_hold_recovery_metric_row(
            torch.zeros(5).unbind(), decay=0.99
        )

    with pytest.raises(
        RuntimeError,
        match="exact/hold metric rollout row counts differ",
    ):
        command.materialize_action_ball_diagnostic_metrics_for_report(
            expected_full_mdp_exact_row_counts=(48, 49)
        )

    assert calls == []
    assert len(command._action_ball_full_mdp_pending_exact_metric_rows) == 48
    assert len(
        command._action_ball_full_mdp_pending_hold_recovery_metric_rows
    ) == 49


def test_full_mdp_hold_pending_and_nonfinite_rows_fail_closed():
    command = _deferred_hold_command()
    row = torch.zeros(5)
    row[2] = float("nan")
    command._stage_action_ball_full_mdp_hold_recovery_metric_row(
        row.unbind(), decay=0.99
    )

    with pytest.raises(RuntimeError, match="pending on device"):
        command.assert_action_ball_diagnostic_metrics_materialized_for_report()
    with pytest.raises(RuntimeError, match="non-finite scalar"):
        command.materialize_action_ball_diagnostic_metrics_for_report()

    assert len(
        command._action_ball_full_mdp_pending_hold_recovery_metric_rows
    ) == 1


def test_full_mdp_second_tape_failure_cannot_partially_replay_exact_rows():
    command = _deferred_hold_command()
    command._exact_n_acc = 7.0
    command._heading_expiry_sum_acc = 11.0
    command._stage_action_ball_full_mdp_exact_metric_row(
        torch.arange(1, 19, dtype=torch.float32).unbind(),
        decay=0.75,
        bucket_order=(0,),
    )
    hold = torch.zeros(5)
    hold[2] = float("nan")
    command._stage_action_ball_full_mdp_hold_recovery_metric_row(
        hold.unbind(),
        decay=0.75,
    )

    with pytest.raises(RuntimeError, match="non-finite scalar"):
        command.materialize_action_ball_diagnostic_metrics_for_report(
            expected_full_mdp_exact_row_counts=(1,)
        )

    assert command._exact_n_acc == 7.0
    assert command._heading_expiry_sum_acc == 11.0
    assert len(command._action_ball_full_mdp_pending_exact_metric_rows) == 1
    assert len(
        command._action_ball_full_mdp_pending_hold_recovery_metric_rows
    ) == 1


def test_full_mdp_exact_and_hold_decay_drift_fails_before_transfer(monkeypatch):
    command = _deferred_hold_command()
    calls = []
    monkeypatch.setattr(
        hope_commands_mod,
        "_batched_host_scalar_rows",
        lambda rows: calls.append(tuple(rows)),
    )
    command._stage_action_ball_full_mdp_exact_metric_row(
        torch.zeros(18).unbind(), decay=0.99, bucket_order=(0,)
    )
    command._stage_action_ball_full_mdp_hold_recovery_metric_row(
        torch.zeros(5).unbind(), decay=0.98
    )

    with pytest.raises(RuntimeError, match="decay chronology differs"):
        command.materialize_action_ball_diagnostic_metrics_for_report(
            expected_full_mdp_exact_row_counts=(1,)
        )

    assert calls == []


def test_full_mdp_hold_update_stages_no_host_read_and_handles_missing_hold(
    monkeypatch,
):
    command = _deferred_hold_command()
    calls = []
    original = hope_commands_mod._batched_host_scalar_rows

    def counted(rows):
        rows = tuple(rows)
        calls.append((len(rows), int(rows[0].numel())))
        return original(rows)

    monkeypatch.setattr(hope_commands_mod, "_batched_host_scalar_rows", counted)
    command._update_hold_recovery_metrics(
        torch.tensor([True, True, True]),
        decay=0.9,
    )
    command._update_hold_recovery_metrics(None, decay=0.8)

    assert calls == []
    assert len(
        command._action_ball_full_mdp_pending_hold_recovery_metric_rows
    ) == 2
    command.materialize_action_ball_diagnostic_metrics_for_report()
    assert calls == [(2, 5)]


def test_full_mdp_hold_hotpath_skips_python_decay_and_threads_step_decay():
    command_type = hope_commands_mod.RacketTargetCommand
    decay_source = inspect.getsource(command_type._decay_swing_accounting)
    update_source = inspect.getsource(command_type._update_metrics)
    footwork_source = inspect.getsource(command_type._update_footwork_signals)

    assert (
        "if not self._action_ball_full_mdp_deferred_exact_metrics_enabled():"
        in decay_source
    )
    assert "hold_recovery_decay=decay" in update_source
    assert "decay=hold_recovery_decay" in footwork_source
