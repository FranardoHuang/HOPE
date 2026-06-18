"""Quaternion utility tests (Planner_Reference_Setup.md Section 8 gate)."""

import numpy as np

from hope_planner.quaternion_utils import normal_to_quaternion


def test_quaternion_is_normalized():
    normals = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.2, -0.7, 0.4]),
    ]
    for n in normals:
        q = normal_to_quaternion(n)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9)


def test_quaternion_normalized_with_constrain_up():
    for n in [np.array([1.0, 0.2, 0.3]), np.array([0.5, -0.5, 0.7])]:
        q = normal_to_quaternion(n, constrain_up=True)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
