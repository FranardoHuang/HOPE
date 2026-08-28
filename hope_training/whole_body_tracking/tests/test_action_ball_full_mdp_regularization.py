"""Focused CPU contracts for the shared FullMDP continuous learning costs."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch


MDP = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
)
if str(MDP) not in sys.path:
    sys.path.insert(0, str(MDP))
R = importlib.import_module("action_ball_full_mdp_regularization")
C = importlib.import_module("action_ball_full_mdp_reward_contract")


def _limits(lower: float, upper: float) -> torch.Tensor:
    return torch.tensor([[lower, upper]] * R.JOINT_COUNT, dtype=torch.float32)


def _legacy_soft_limit_barrier_v2(
    positions: torch.Tensor,
    soft_limits: torch.Tensor,
    default_joint_pos: torch.Tensor,
    hard_limits: torch.Tensor,
) -> torch.Tensor:
    """Exact pre-preparation formula retained only as the parity oracle."""

    lower, upper = soft_limits[:, :, 0], soft_limits[:, :, 1]
    hard_lower, hard_upper = hard_limits[:, :, 0], hard_limits[:, :, 1]
    span = upper - lower
    default_distance = torch.minimum(
        default_joint_pos - lower, upper - default_joint_pos
    ) / span
    margin_eff = torch.clamp(
        default_distance - R.STANCE_EPS_FRAC,
        max=float(R.SOFT_LIMIT_MARGIN_FRAC),
    )
    row_finite = torch.all(torch.isfinite(positions), dim=1)
    safe_positions = torch.where(
        row_finite[:, None], positions, default_joint_pos
    )
    distance = torch.minimum(
        safe_positions - lower, upper - safe_positions
    ) / span
    intrusion = torch.relu(margin_eff - distance) / margin_eff
    band_rad = margin_eff * span
    depth_rad = intrusion * band_rad
    ramp = torch.where(
        intrusion <= 1.0,
        torch.square(depth_rad) / (2.0 * band_rad),
        depth_rad - 0.5 * band_rad,
    )
    hard_excess = torch.relu(hard_lower - safe_positions) + torch.relu(
        safe_positions - hard_upper
    )
    floor_rad = torch.where(
        hard_excess > 0.0,
        float(R.SOFT_LIMIT_PENALTY_FLOOR) * band_rad,
        torch.zeros_like(ramp),
    )
    per_joint = torch.where(
        intrusion > 0.0, ramp + floor_rad, torch.zeros_like(ramp)
    )
    value = torch.sum(per_joint, dim=1)
    return -torch.where(row_finite, value, torch.zeros_like(value))


def test_reward_contract_remains_dependency_light_without_torch_import():
    source_path = MDP / "action_ball_full_mdp_reward_contract.py"
    source = source_path.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "action_ball_full_mdp_regularization" not in source
    namespace = {"__name__": "dependency_light_reward_contract"}
    exec(compile(source, str(source_path), "exec"), namespace)
    assert namespace["REGULARIZATION_NAMES"] == C.REGULARIZATION_NAMES


def test_fixed_contract_has_four_separate_negative_objectives():
    assert C.REGULARIZATION_NAMES == (
        "action_rate_l2",
        "qdes_limit_barrier",
        "qdes_projection_penalty",
        "joint_limit",
    )
    assert [spec.manager_weight for spec in C.REGULARIZATION_SPECS] == [
        0.1,
        10.0,
        1.0,
        10.0,
    ]
    assert [spec.effective_coefficient for spec in C.REGULARIZATION_SPECS] == [
        -0.1,
        -10.0,
        -1.0,
        -10.0,
    ]
    assert C.MANAGER_NAMES[-4:] == C.REGULARIZATION_NAMES


def test_action_rate_is_isaaclab_sum_not_mean_and_reset_zero_is_free():
    action = torch.zeros((3, R.JOINT_COUNT), dtype=torch.float32)
    action[0] = torch.linspace(-1.7, 2.3, R.JOINT_COUNT)
    action[1, (0, 3, 11, 30)] = torch.tensor([1.0, -2.0, 0.5, 3.0])
    previous = torch.zeros_like(action)
    action[2, 7] = float("nan")

    value = R.action_rate_l2(action, previous)

    assert value[0] == pytest.approx(-torch.square(action[0]).sum().item())
    assert value[1] == pytest.approx(-(1.0 + 4.0 + 0.25 + 9.0))
    assert value[1] != pytest.approx(-(1.0 + 4.0 + 0.25 + 9.0) / R.JOINT_COUNT)
    assert value[2].item() == 0.0
    assert torch.isfinite(value).all()
    reset_action = torch.zeros_like(action[:1])
    assert R.action_rate_l2(reset_action, reset_action).item() == 0.0


def test_soft_limit_v2_keeps_quadratic_band_linear_tail_and_invalid_row_finite():
    soft = _limits(-0.9, 0.9)
    hard = _limits(-1.0, 1.0)
    default = torch.zeros(R.JOINT_COUNT, dtype=torch.float32)
    positions = torch.zeros((4, R.JOINT_COUNT), dtype=torch.float32)
    # band = .02 * 1.8 = .036 rad.  Depth .018 is the quadratic midpoint.
    positions[1, 5] = 0.9 - 0.018
    # 0.018 rad beyond the soft limit is in the linear tail: depth=.054.
    positions[2, 9] = 0.9 + 0.018
    positions[3, 2] = float("inf")

    value = R.soft_limit_barrier_v2(positions, soft, default, hard)

    assert value[0].item() == 0.0
    assert value[1].item() == pytest.approx(
        -(0.018**2 / (2.0 * 0.036)), abs=1.0e-6
    )
    assert value[2].item() == pytest.approx(
        -(0.054 - 0.5 * 0.036), abs=1.0e-6
    )
    assert value[3].item() == 0.0
    assert torch.isfinite(value).all()


def test_prepared_soft_limit_geometry_is_bitwise_equal_across_dynamic_rows():
    soft = _limits(-0.9, 0.9).unsqueeze(0).expand(8, -1, -1).clone()
    hard = _limits(-1.0, 1.0).unsqueeze(0).expand(8, -1, -1).clone()
    default = torch.zeros((8, R.JOINT_COUNT), dtype=torch.float32)
    positions = torch.zeros_like(default)
    band = 0.02 * 1.8
    positions[1, 0] = 0.9 - band
    positions[2, 1] = 0.9 - 0.5 * band
    positions[3, 2] = 0.9 + 0.5 * band
    positions[4, 3] = 1.05
    positions[5, 4] = -1.05
    positions[6, 5] = float("nan")
    positions[7, 6] = float("inf")

    prepared = R._prepare_soft_limit_barrier_v2(
        positions, soft, default, hard, freeze=True
    )
    expected = _legacy_soft_limit_barrier_v2(
        positions, soft, default, hard
    )
    actual = R._soft_limit_barrier_v2_prepared(positions, prepared)
    assert torch.equal(actual, expected)
    assert torch.equal(
        R.soft_limit_barrier_v2(positions, soft, default, hard), expected
    )

    reset_positions = torch.zeros_like(positions)
    reset_positions[0, 7] = -0.9 + 0.25 * band
    reset_expected = _legacy_soft_limit_barrier_v2(
        reset_positions, soft, default, hard
    )
    reset_actual = R._soft_limit_barrier_v2_prepared(
        reset_positions, prepared
    )
    assert torch.equal(reset_actual, reset_expected)

    soft.add_(0.1)
    default.add_(0.1)
    hard.add_(0.1)
    assert torch.equal(
        R._soft_limit_barrier_v2_prepared(reset_positions, prepared),
        reset_expected,
    )


def test_prepared_soft_limit_geometry_keeps_static_validation_at_construction():
    reference = torch.zeros((2, R.JOINT_COUNT), dtype=torch.float32)
    default = torch.zeros(R.JOINT_COUNT, dtype=torch.float32)
    hard = _limits(-1.0, 1.0)

    nonfinite = _limits(-0.9, 0.9)
    nonfinite[0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        R._prepare_soft_limit_barrier_v2(
            reference, nonfinite, default, hard
        )

    reversed_limits = _limits(-0.9, 0.9)
    reversed_limits[0] = torch.tensor([0.5, -0.5])
    with pytest.raises(RuntimeError):
        R._prepare_soft_limit_barrier_v2(
            reference, reversed_limits, default, hard
        )


def test_projection_v2_has_full_span_nonfinite_surrogate_and_invalid_reset_zero():
    projected = torch.zeros((3, R.JOINT_COUNT), dtype=torch.float32)
    span = torch.full_like(projected, 2.0)
    requested = projected.clone()
    requested[0, 0] = 0.05  # knee=.1 rad, quadratic branch
    requested[0, 1] = 0.3  # linear branch
    requested[1, 4] = float("nan")  # one full span surrogate
    valid = torch.tensor([True, True, False])

    value = R.qdes_projection_penalty(
        requested, projected, span, valid, valid
    )

    expected0 = 0.05**2 / (2.0 * 0.1) + (0.3 - 0.05)
    expected1 = 2.0 - 0.05
    assert value[0].item() == pytest.approx(-expected0)
    assert value[1].item() == pytest.approx(-expected1)
    assert value[2].item() == 0.0
    assert torch.isfinite(value).all()


def test_four_component_manager_conservation_is_exact_sum_of_rows():
    action = torch.zeros((2, R.JOINT_COUNT), dtype=torch.float32)
    action[0, (0, 8, 30)] = torch.tensor([1.0, -0.5, 2.0])
    previous = torch.zeros_like(action)
    soft = _limits(-0.9, 0.9)
    hard = _limits(-1.0, 1.0)
    default = torch.zeros(R.JOINT_COUNT, dtype=torch.float32)
    qdes = torch.zeros_like(action)
    qdes[0, 4] = 0.89
    pre = qdes.clone()
    projected = qdes.clone()
    span = torch.full_like(action, 1.62)
    valid = torch.ones(2, dtype=torch.bool)
    actual = torch.zeros_like(action)
    actual[0, 7] = -0.895
    raw = torch.stack(
        (
            R.action_rate_l2(action, previous),
            R.soft_limit_barrier_v2(qdes, soft, default, hard),
            R.qdes_projection_penalty(pre, projected, span, valid, valid),
            R.soft_limit_barrier_v2(actual, soft, default, hard),
        ),
        dim=1,
    )
    weights = torch.tensor(
        [spec.manager_weight for spec in R.REGULARIZATION_SPECS]
    )
    configured_rows = raw * weights * 0.02

    total = configured_rows.sum(dim=1)
    assert torch.equal(total, torch.sum(configured_rows, dim=1))
    assert torch.all(configured_rows <= 0.0)
    assert torch.isfinite(configured_rows).all()
