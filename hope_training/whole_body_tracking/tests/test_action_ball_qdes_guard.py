from __future__ import annotations

import torch

from action_ball_qdes_guard import action_ball_qdes_guard


def _guard(pre, *, previous=None, valid=None, q=None, qd=None):
    pre = torch.as_tensor(pre, dtype=torch.float32)
    if pre.ndim == 1:
        pre = pre.unsqueeze(0)
    n, j = pre.shape
    previous = (
        torch.zeros_like(pre) if previous is None else torch.as_tensor(
            previous, dtype=pre.dtype
        ).reshape(n, j)
    )
    valid = (
        torch.ones(n, dtype=torch.bool) if valid is None else torch.as_tensor(
            valid, dtype=torch.bool
        ).reshape(n)
    )
    q = torch.zeros_like(pre) if q is None else torch.as_tensor(
        q, dtype=pre.dtype
    ).reshape(n, j)
    qd = torch.zeros_like(pre) if qd is None else torch.as_tensor(
        qd, dtype=pre.dtype
    ).reshape(n, j)
    hard_lower = torch.full_like(pre, -1.0)
    hard_upper = torch.full_like(pre, 1.0)
    soft_lower = torch.full_like(pre, -0.9)
    soft_upper = torch.full_like(pre, 0.9)
    return action_ball_qdes_guard(
        pre_clamp_qdes=pre,
        previous_executable_qdes=previous,
        previous_executable_valid=valid,
        default_qdes=torch.zeros_like(pre),
        soft_lower=soft_lower,
        soft_upper=soft_upper,
        hard_lower=hard_lower,
        hard_upper=hard_upper,
        joint_pos=q,
        joint_vel=qd,
        policy_dt_s=0.02,
        hard_margin_rad=0.0,
        hard_margin_fraction=0.05,
        project_finite_without_termination=True,
        projection_soft_inset_fraction=0.05,
    )


def test_finite_proposal_projects_to_soft_and_hard_inset_without_terminal():
    result = _guard([[2.0, -2.0]])
    # hard +/-1 -> soft +/-0.9 -> another 5% of soft span = +/-0.81.
    assert torch.allclose(result.executable_qdes, torch.tensor([[0.81, -0.81]]))
    assert not bool(result.hard_violation_env.any())
    assert result.qdes_forbidden_request.all()
    assert not result.qdes_safety_violation.any()


def test_nonfinite_request_uses_previous_target_and_remains_terminal():
    result = _guard(
        [[float("nan"), float("inf")]], previous=[[0.2, -0.3]]
    )
    assert torch.allclose(result.finite_fallback_qdes, torch.tensor([[0.2, -0.3]]))
    assert torch.isfinite(result.executable_qdes).all()
    assert result.qdes_safety_violation.all()
    assert bool(result.hard_violation_env[0])


def test_predicted_crossing_uses_maximum_inward_target_and_terminates():
    result = _guard([[0.0]], q=[[0.85]], qd=[[5.0]])
    assert bool(result.crossing_violation[0, 0])
    assert bool(result.upper_crossing_risk[0, 0])
    assert bool(result.unambiguous_crossing_risk[0, 0])
    assert bool(result.hard_violation_env[0])
    assert torch.allclose(result.brake_target, torch.tensor([[0.75]]))
    assert torch.allclose(result.maximum_inward_target, torch.tensor([[-0.81]]))
    assert torch.allclose(result.executable_qdes, torch.tensor([[-0.81]]))


def test_dual_side_crossing_retains_bounded_velocity_horizon_target():
    result = _guard([[0.0]], q=[[0.95]], qd=[[-100.0]])
    assert bool(result.lower_crossing_risk[0, 0])
    assert bool(result.upper_crossing_risk[0, 0])
    assert not bool(result.unambiguous_crossing_risk[0, 0])
    assert torch.allclose(result.brake_target, torch.tensor([[0.81]]))
    assert torch.allclose(result.executable_qdes, result.brake_target)


def test_first_step_uses_default_when_previous_target_is_invalid():
    result = _guard(
        [[float("nan")]], previous=[[float("nan")]], valid=[False]
    )
    assert torch.equal(result.finite_fallback_qdes, torch.zeros((1, 1)))
    assert torch.isfinite(result.executable_qdes).all()
