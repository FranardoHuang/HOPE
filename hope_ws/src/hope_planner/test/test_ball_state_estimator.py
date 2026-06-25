"""Stage 1 unit tests (HOPE_7DOF...Planner_Reference_Setup.md Section 3 gates)."""

import numpy as np

from hope_planner.ball_state_estimator import BallStateEstimator
from hope_planner.constants import PlannerConfig


def _dt():
    return 1.0 / 360.0


def test_constant_velocity_returns_correct_velocity():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    v_true = np.array([1.0, -0.5, 0.0])
    p0 = np.array([0.2, -0.3, 0.5])  # z constant > bounce tol -> no false bounce
    dt = _dt()
    for i in range(20):
        est.push(i * dt, p0 + v_true * (i * dt))
    assert est.ready
    _, v_est, _ = est.estimate()
    assert np.allclose(v_est, v_true, atol=1e-3)


def test_parabolic_z_returns_correct_vertical_velocity():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    z0, vz0, g = 1.0, 0.0, -9.81
    n = 20
    for i in range(n):
        t = i * dt
        p = np.array([0.0, 0.0, z0 + vz0 * t + 0.5 * g * t * t])
        est.push(t, p)
    _, v_est, t_ref = est.estimate()
    expected_vz = vz0 + g * t_ref  # derivative at the latest (reference) sample
    assert abs(v_est[2] - expected_vz) < 1e-3


def test_bounce_pattern_clears_buffer():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    zs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.002, 0.1]  # descend -> contact -> rise
    for i, z in enumerate(zs):
        est.push(i * dt, np.array([0.0, 0.0, z]))
    assert est.bounce_detected
    # reset() runs before the rising sample is appended -> only that sample remains
    assert len(est.t_buffer) == 1
    assert not est.ready


def test_fewer_than_six_samples_not_ready():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    for i in range(5):
        est.push(i * dt, np.array([0.0, 0.0, 0.5]))
    assert not est.ready
