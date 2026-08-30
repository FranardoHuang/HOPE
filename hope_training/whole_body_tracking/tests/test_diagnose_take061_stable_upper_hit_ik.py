import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_take061_stable_upper_hit_ik.py"
SPEC = importlib.util.spec_from_file_location("stable_upper_hit_ik_under_test", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def test_strict_bounds_never_return_projector_edge():
    lower, upper = M.strict_executable_bounds(np.array([-1.0, -0.2]), np.array([1.0, 0.3]))
    assert np.all(lower > [-1.0, -0.2])
    assert np.all(upper < [1.0, 0.3])


def test_strict_bounds_reject_empty_open_interval():
    with pytest.raises(M.DiagnosticError, match="invalid executable"):
        M.strict_executable_bounds(np.array([0.0]), np.array([1.0e-6]))


def test_minimum_jerk_duration_uses_peak_derivative_and_per_joint_limits():
    duration, rows = M.minimum_jerk_duration_s(
        np.array([0.0, 1.0]), np.array([2.0, 1.5]), np.array([4.0, 0.25])
    )
    assert rows.tolist() == pytest.approx([0.9375, 3.75])
    assert duration == pytest.approx(3.75)


def test_optimized_joint_contract_is_waist_plus_right_arm_only():
    assert M.OPTIMIZED_JOINTS[:3] == (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    assert len(M.OPTIMIZED_JOINTS) == 10
    assert all("left_" not in name and "hip" not in name for name in M.OPTIMIZED_JOINTS)
