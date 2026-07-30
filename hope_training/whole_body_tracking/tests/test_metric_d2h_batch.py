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
    # The behavior-coupled EMA remains a Python-float, per-step recurrence; this is not a
    # device-only or rollout-boundary semantic change.
    assert (
        "self._exact_n_acc = decay * self._exact_n_acc "
        "+ next(_exact_metric_values)"
    ) in source
    assert "self._update_adaptive_sigma(enough, denom)" in source
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
