import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "solve_take061_stable_support_plant_feasible.py"
)
SPEC = importlib.util.spec_from_file_location("take061_plant_feasible", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_plant_compensated_qdes_matches_pd_inverse_equation():
    q = np.array([0.2, -0.3])
    qd = np.array([0.4, -0.5])
    tau = np.array([2.0, -3.0])
    kp = np.array([20.0, 30.0])
    kd = np.array([1.0, 2.0])
    qdes = M.plant_compensated_qdes(q, qd, tau, kp, kd)
    np.testing.assert_allclose(kp * (qdes - q) - kd * qd, tau)


def test_timewarp_reduces_candidate_velocity_but_not_fixed_task_velocity():
    fps = 10.0
    sites = np.stack([np.arange(5), np.zeros(5), np.zeros(5)], axis=1) / fps
    target = sites.copy()
    frames = np.array([1, 2, 3])
    assert M.contact_velocity_error(sites, target, fps, 1.0, frames) == pytest.approx(0.0)
    assert M.contact_velocity_error(sites, target, fps, 2.0, frames) == pytest.approx(0.5)


def test_minimum_jerk_has_exact_endpoints_and_zero_endpoint_rates():
    start = np.array([0.1, -0.2])
    goal = np.array([0.4, 0.7])
    q, qd, qdd = M.minimum_jerk_samples(start, goal, duration_s=0.6, dt=0.02)
    np.testing.assert_allclose(q[0], start)
    np.testing.assert_allclose(q[-1], goal)
    np.testing.assert_allclose(qd[[0, -1]], 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd[[0, -1]], 0.0, atol=1e-12)


def test_choose_first_admitted_preserves_minimum_dof_order():
    assert M.choose_first_admitted([{"admitted": False}, {"admitted": True}]) == 1
    assert M.choose_first_admitted([{"admitted": False}]) is None


def test_no_replace_refuses_to_overwrite(tmp_path):
    output = tmp_path / "artifact.json"
    M._write_no_replace(output, b"first")
    with pytest.raises(FileExistsError):
        M._write_no_replace(output, b"second")
    assert output.read_bytes() == b"first"

