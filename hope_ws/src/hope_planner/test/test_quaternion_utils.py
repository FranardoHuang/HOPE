"""Quaternion utility tests (Planner_Reference_Setup.md Section 8 gate)."""

import numpy as np

from hope_planner.quaternion_utils import normal_to_quaternion


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


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


def test_quaternion_rotates_local_x_to_normal():
    normals = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.2, -0.7, 0.4]),
    ]
    local_x = np.array([1.0, 0.0, 0.0])
    for n in normals:
        target = n / np.linalg.norm(n)
        q = normal_to_quaternion(n)
        assert np.allclose(_quat_to_matrix(q) @ local_x, target, atol=1e-9)


def test_quaternion_normalized_with_constrain_up():
    for n in [np.array([1.0, 0.2, 0.3]), np.array([0.5, -0.5, 0.7])]:
        q = normal_to_quaternion(n, constrain_up=True)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
