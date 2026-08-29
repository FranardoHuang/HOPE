"""Fixed analytic and shared-backend contracts for the paddle motion prior."""

from pathlib import Path
import math
import sys

import pytest
import torch


MDP = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
sys.path.insert(0, str(MDP))
import action_ball_full_mdp_paddle_prior as P  # noqa: E402
import action_ball_full_mdp_reward_contract as C  # noqa: E402


def test_direct_paddle_successor_strengthens_only_dynamic_playback_rows():
    assert C.PADDLE_MOTION_PRIOR_PLAYBACK_SCALE == 4.0
    assert tuple(
        spec.manager_weight for spec in C.PADDLE_MOTION_PRIOR_SPECS
    ) == (1.0, 1.0, 1.0, 0.5)
    assert tuple(spec.std for spec in C.PADDLE_MOTION_PRIOR_SPECS) == (
        0.075,
        0.50,
        0.2617993877991494,
        0.17453292519943295,
    )
    assert tuple(spec.coarse_std for spec in C.PADDLE_MOTION_PRIOR_SPECS) == (
        0.30,
        2.0,
        1.0471975511965976,
        0.6981317007977318,
    )
    assert all(
        spec.command_name == "racket_target"
        and spec.scale_during_playback == 4.0
        for spec in C.PADDLE_MOTION_PRIOR_SPECS
    )


def test_fixed_composite_kernel_matches_independent_analytic_values():
    error = torch.tensor([0.0, 0.075, 0.30, 0.60], dtype=torch.float64)
    actual = P.coarse_precision_kernel(
        error, precision_std=0.075, coarse_std=0.30
    )
    expected = torch.tensor(
        [
            1.0,
            0.5 * math.exp(-1.0) + 0.5 / (1.0 + 0.25**2),
            0.5 * math.exp(-16.0) + 0.25,
            0.5 * math.exp(-64.0) + 0.1,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-15)


def test_vectorized_mu_kernel_is_the_same_four_channel_kernel_as_isaac():
    errors = torch.tensor(
        [[0.075, 0.50, math.radians(15.0), math.radians(10.0)]],
        dtype=torch.float64,
    )
    precision = torch.tensor(
        [spec.std for spec in C.PADDLE_MOTION_PRIOR_SPECS], dtype=torch.float64
    )
    coarse = torch.tensor(
        [spec.coarse_std for spec in C.PADDLE_MOTION_PRIOR_SPECS],
        dtype=torch.float64,
    )
    vectorized = P.kernels(
        errors, precision_stds=precision, coarse_stds=coarse
    )
    scalar_columns = torch.stack(
        [
            P.coarse_precision_kernel(
                errors[:, index],
                precision_std=spec.std,
                coarse_std=spec.coarse_std,
            )
            for index, spec in enumerate(C.PADDLE_MOTION_PRIOR_SPECS)
        ],
        dim=1,
    )
    torch.testing.assert_close(vectorized, scalar_columns, rtol=0.0, atol=0.0)


def test_vectorized_kernel_preserves_scalar_nonfinite_results_exactly():
    errors = torch.tensor(
        [
            [float("nan"), 0.50, float("inf"), math.radians(10.0)],
            [0.075, float("-inf"), math.radians(15.0), float("nan")],
        ],
        dtype=torch.float64,
    )
    precision = torch.tensor(
        [spec.std for spec in C.PADDLE_MOTION_PRIOR_SPECS], dtype=torch.float64
    )
    coarse = torch.tensor(
        [spec.coarse_std for spec in C.PADDLE_MOTION_PRIOR_SPECS],
        dtype=torch.float64,
    )
    vectorized = P.kernels(
        errors, precision_stds=precision, coarse_stds=coarse
    )
    scalar_columns = torch.stack(
        [
            P.coarse_precision_kernel(
                errors[:, index],
                precision_std=spec.std,
                coarse_std=spec.coarse_std,
            )
            for index, spec in enumerate(C.PADDLE_MOTION_PRIOR_SPECS)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        vectorized, scalar_columns, rtol=0.0, atol=0.0, equal_nan=True
    )


def test_tracking_errors_are_physical_units_and_preserve_nonfinite_evidence():
    actual_position = torch.tensor(
        [[3.0, 4.0, 0.0], [float("nan"), 0.0, 0.0]], dtype=torch.float32
    )
    actual_velocity = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32
    )
    actual_face = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    )
    actual_long = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    )
    zeros = torch.zeros((2, 3), dtype=torch.float32)
    teacher_axis = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    )
    error = P.tracking_errors(
        actual_position,
        actual_velocity,
        actual_face,
        actual_long,
        zeros,
        zeros,
        teacher_axis,
        teacher_axis,
    )
    assert tuple(error.shape) == (2, 4)
    assert error.is_contiguous()
    assert error[0].tolist() == pytest.approx(
        [5.0, 2.0, 0.0, math.pi / 2.0]
    )
    assert torch.isnan(error[1, 0])
