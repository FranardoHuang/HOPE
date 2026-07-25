"""Tests for canonical-ready-relative root-pose path encoding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "canonical_root_pose_codec.py"
)
_SPEC = importlib.util.spec_from_file_location("canonical_root_pose_codec", _SCRIPT)
codec = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = codec
_SPEC.loader.exec_module(codec)


def _qz(angle):
    angle = np.asarray(angle, dtype=np.float64)
    return np.stack(
        (
            np.cos(angle / 2.0),
            np.zeros_like(angle),
            np.zeros_like(angle),
            np.sin(angle / 2.0),
        ),
        axis=-1,
    )


def _quat_distance(left, right):
    dot = np.sum(left * right, axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def test_roundtrip_relative_to_nonidentity_ready():
    ready = _qz(np.deg2rad(72.0))
    relative = codec.rotation_vector_to_quat_wxyz(
        np.column_stack(
            (
                np.linspace(0.0, 0.15, 8),
                np.linspace(0.0, -0.08, 8),
                np.linspace(0.0, 0.25, 8),
            )
        )
    )
    world = codec.quat_multiply_wxyz(ready, relative)
    pos = np.column_stack(
        (
            np.linspace(0.1, 0.2, 8),
            np.linspace(-0.2, -0.1, 8),
            np.linspace(0.9, 0.92, 8),
        )
    )
    encoded = codec.encode_root_pose(
        pos, world, canonical_ready_root_quat_wxyz=ready
    )
    decoded_pos, decoded_quat = codec.decode_root_pose(
        encoded.coordinates, canonical_ready_root_quat_wxyz=ready
    )
    np.testing.assert_array_equal(decoded_pos, pos)
    assert float(np.max(_quat_distance(decoded_quat, world))) < 5.0e-8
    np.testing.assert_allclose(encoded.coordinates[0, 3:], 0.0, atol=1.0e-15)


def test_quaternion_input_signs_do_not_change_coordinates():
    ready = np.array([1.0, 0.0, 0.0, 0.0])
    world = _qz(np.linspace(0.0, 0.4, 5))
    alternating = world.copy()
    alternating[1::2] *= -1.0
    pos = np.zeros((5, 3))
    normal = codec.encode_root_pose(
        pos, world, canonical_ready_root_quat_wxyz=ready
    )
    signed = codec.encode_root_pose(
        pos, alternating, canonical_ready_root_quat_wxyz=ready
    )
    np.testing.assert_allclose(normal.coordinates, signed.coordinates, atol=1e-15)


def test_exponential_and_logarithm_handle_zero_and_small_angles():
    vector = np.array([[0.0, 0.0, 0.0], [1e-12, -2e-12, 3e-12]])
    quat = codec.rotation_vector_to_quat_wxyz(vector)
    rebuilt = codec.quat_wxyz_to_rotation_vector(quat)
    np.testing.assert_allclose(rebuilt, vector, rtol=1e-10, atol=1e-15)


def test_pi_branch_cut_fails_closed():
    ready = np.array([1.0, 0.0, 0.0, 0.0])
    pos = np.zeros((2, 3))
    world = codec.rotation_vector_to_quat_wxyz(
        np.array([[0.0, 0.0, 0.0], [np.pi - 1e-4, 0.0, 0.0]])
    )
    with pytest.raises(codec.RootPoseCodecError, match="branch cut"):
        codec.encode_root_pose(
            pos,
            world,
            canonical_ready_root_quat_wxyz=ready,
            pi_margin_rad=1e-3,
        )


def test_discontinuous_rotation_vector_sampling_fails_closed():
    ready = np.array([1.0, 0.0, 0.0, 0.0])
    pos = np.zeros((2, 3))
    world = codec.rotation_vector_to_quat_wxyz(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    with pytest.raises(codec.RootPoseCodecError, match="discontinuous"):
        codec.encode_root_pose(
            pos,
            world,
            canonical_ready_root_quat_wxyz=ready,
            max_rotation_vector_step_rad=0.5,
        )


@pytest.mark.parametrize(
    "position, quaternion",
    [
        (np.zeros((3, 2)), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))),
        (np.zeros((3, 3)), np.zeros((3, 4))),
        (
            np.zeros((3, 3)),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [np.nan, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_bad_pose_inputs_fail_closed(position, quaternion):
    with pytest.raises(codec.RootPoseCodecError):
        codec.encode_root_pose(
            position,
            quaternion,
            canonical_ready_root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )


def test_report_is_explicit_about_local_chart():
    ready = np.array([1.0, 0.0, 0.0, 0.0])
    encoded = codec.encode_root_pose(
        np.zeros((2, 3)),
        _qz(np.array([0.0, 0.1])),
        canonical_ready_root_quat_wxyz=ready,
    )
    assert encoded.report["algorithm"] == "canonical_ready_relative_so3_log_v1"
    assert encoded.report["coordinate_order"][3] == "root_rotvec_x_ready"
    assert encoded.report["limitations"]


def _vee(skew_matrix):
    return np.stack(
        (
            skew_matrix[..., 2, 1],
            skew_matrix[..., 0, 2],
            skew_matrix[..., 1, 0],
        ),
        axis=-1,
    )


def _quat_matrix(quat):
    w, x, y, z = quat
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def test_rotation_vector_rate_matches_world_angular_velocity_central_difference():
    ready = codec.rotation_vector_to_quat_wxyz(np.array([0.2, -0.1, 0.3]))
    vector = np.array([[0.4, -0.2, 0.15], [-0.1, 0.3, 0.2]])
    rate = np.array([[0.7, -0.4, 0.2], [0.1, 0.5, -0.3]])
    actual = codec.rotation_vector_rate_to_world_angular_velocity(
        vector,
        rate,
        canonical_ready_root_quat_wxyz=ready,
    )
    epsilon = 1.0e-7
    expected = []
    ready_matrix = _quat_matrix(ready)
    for row, row_rate in zip(vector, rate):
        center = ready_matrix @ _quat_matrix(
            codec.rotation_vector_to_quat_wxyz(row)
        )
        plus = ready_matrix @ _quat_matrix(
            codec.rotation_vector_to_quat_wxyz(row + epsilon * row_rate)
        )
        minus = ready_matrix @ _quat_matrix(
            codec.rotation_vector_to_quat_wxyz(row - epsilon * row_rate)
        )
        spatial_skew = ((plus - minus) / (2.0 * epsilon)) @ center.T
        expected.append(_vee(spatial_skew))
    np.testing.assert_allclose(actual, np.asarray(expected), atol=2.0e-9, rtol=0.0)


def test_root_coordinate_velocity_decodes_world_twist_and_zero_endpoints():
    ready = codec.rotation_vector_to_quat_wxyz(
        np.array([0.0, 0.0, np.pi / 2.0])
    )
    coordinates = np.zeros((3, 6))
    coordinates[1, 3] = 0.4
    velocity = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    linear, angular = codec.root_coordinate_velocity_to_world_twist(
        coordinates,
        velocity,
        canonical_ready_root_quat_wxyz=ready,
    )
    np.testing.assert_array_equal(linear, velocity[:, :3])
    np.testing.assert_array_equal(angular[[0, 2]], np.zeros((2, 3)))
    np.testing.assert_allclose(angular[1], [0.0, 0.5, 0.0], atol=1.0e-12)


def test_rotation_vector_second_derivative_matches_angular_velocity_derivative():
    ready = codec.rotation_vector_to_quat_wxyz(
        np.array([0.17, -0.09, 0.23])
    )
    vector = np.array(
        [[0.31, -0.22, 0.11], [-0.08, 0.27, 0.19]],
        dtype=np.float64,
    )
    first = np.array(
        [[0.63, -0.37, 0.24], [0.14, 0.42, -0.31]],
        dtype=np.float64,
    )
    second = np.array(
        [[-0.21, 0.16, 0.09], [0.07, -0.18, 0.23]],
        dtype=np.float64,
    )
    omega, alpha = (
        codec.rotation_vector_derivatives_to_world_angular_kinematics(
            vector,
            first,
            second,
            canonical_ready_root_quat_wxyz=ready,
        )
    )
    np.testing.assert_allclose(
        omega,
        codec.rotation_vector_rate_to_world_angular_velocity(
            vector,
            first,
            canonical_ready_root_quat_wxyz=ready,
        ),
        atol=2.0e-15,
        rtol=0.0,
    )

    epsilon = 2.0e-6

    def velocity_at(offset):
        shifted_vector = (
            vector
            + offset * first
            + 0.5 * offset * offset * second
        )
        shifted_first = first + offset * second
        return codec.rotation_vector_rate_to_world_angular_velocity(
            shifted_vector,
            shifted_first,
            canonical_ready_root_quat_wxyz=ready,
        )

    expected_alpha = (
        velocity_at(epsilon) - velocity_at(-epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        alpha, expected_alpha, atol=2.0e-10, rtol=2.0e-9
    )


def test_rotation_vector_second_derivative_is_regular_at_zero():
    ready = codec.rotation_vector_to_quat_wxyz(
        np.array([-0.1, 0.2, 0.3])
    )
    vector = np.zeros((2, 3), dtype=np.float64)
    first = np.array(
        [[0.8, -0.4, 0.2], [-0.3, 0.1, 0.7]],
        dtype=np.float64,
    )
    second = np.array(
        [[-0.2, 0.5, 0.1], [0.4, -0.6, 0.3]],
        dtype=np.float64,
    )
    omega, alpha = (
        codec.rotation_vector_derivatives_to_world_angular_kinematics(
            vector,
            first,
            second,
            canonical_ready_root_quat_wxyz=ready,
        )
    )
    rotation = _quat_matrix(ready)
    np.testing.assert_allclose(
        omega, np.einsum("ij,tj->ti", rotation, first), atol=1.0e-15
    )
    # dJ_left/ds contributes 0.5 * (phi_s x phi_s), which is exactly zero.
    np.testing.assert_allclose(
        alpha, np.einsum("ij,tj->ti", rotation, second), atol=1.0e-15
    )


@pytest.mark.parametrize("radius", [0.0099, 0.0101])
def test_rotation_vector_second_derivative_is_stable_across_series_boundary(
    radius,
):
    ready = codec.rotation_vector_to_quat_wxyz(
        np.array([0.1, -0.2, 0.3])
    )
    vector = np.array([[radius, 0.2 * radius, -0.1 * radius]])
    first = np.array([[1.0e4, -2.0e3, 3.0e3]])
    second = np.array([[100.0, -50.0, 20.0]])
    _, alpha = (
        codec.rotation_vector_derivatives_to_world_angular_kinematics(
            vector,
            first,
            second,
            canonical_ready_root_quat_wxyz=ready,
        )
    )
    epsilon = 3.0e-9

    def velocity_at(offset):
        return codec.rotation_vector_rate_to_world_angular_velocity(
            vector + offset * first + 0.5 * offset * offset * second,
            first + offset * second,
            canonical_ready_root_quat_wxyz=ready,
        )

    expected = (
        velocity_at(epsilon) - velocity_at(-epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(alpha, expected, atol=1.0e-2, rtol=1.0e-7)


def test_rotation_vector_second_derivative_bad_inputs_fail_closed():
    ready = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(codec.RootPoseCodecError):
        codec.rotation_vector_derivatives_to_world_angular_kinematics(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.zeros((2, 2)),
            canonical_ready_root_quat_wxyz=ready,
        )
    with pytest.raises(codec.RootPoseCodecError, match="branch cut"):
        codec.rotation_vector_derivatives_to_world_angular_kinematics(
            np.array([[np.pi - 1.0e-4, 0.0, 0.0]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            canonical_ready_root_quat_wxyz=ready,
            pi_margin_rad=1.0e-3,
        )
