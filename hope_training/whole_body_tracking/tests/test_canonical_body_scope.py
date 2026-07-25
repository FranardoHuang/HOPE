"""Pure-CPU tests for the canonical A3 body-scope preprocessing contract."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "canonical_body_scope.py"
)
_SPEC = importlib.util.spec_from_file_location("canonical_body_scope", _SCRIPT)
scope = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scope
_SPEC.loader.exec_module(scope)


def _quat_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """ZYX RPY -> wxyz test helper."""

    roll, pitch, yaw = np.broadcast_arrays(
        np.asarray(roll, np.float64),
        np.asarray(pitch, np.float64),
        np.asarray(yaw, np.float64),
    )
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ),
        axis=-1,
    )


def _fixture(frames: int = 6, *, shuffled: bool = False):
    names = list(scope.A3_JOINT_NAMES)
    if shuffled:
        names = names[11:] + names[:11]
    q = np.zeros((frames, 31), dtype=np.float64)
    for index in range(31):
        q[:, index] = index * 0.01 + np.linspace(0.0, 0.005, frames)
    ready_q = np.linspace(-0.2, 0.2, 31)
    root_pos = np.column_stack(
        (
            3.0 + np.linspace(0.0, 0.03, frames),
            -2.0 + np.linspace(0.0, -0.02, frames),
            1.04 + np.linspace(0.0, 0.005, frames),
        )
    )
    root_yaw = np.deg2rad(80.0 + np.linspace(0.0, 20.0, frames))
    root_quat = _quat_from_rpy(
        np.deg2rad(np.linspace(2.0, 3.0, frames)),
        np.deg2rad(np.linspace(-1.0, -2.0, frames)),
        root_yaw,
    )
    ready_pos = np.array([0.4, -0.3, 1.08])
    ready_quat = _quat_from_rpy(
        np.deg2rad(1.0), np.deg2rad(-2.0), np.deg2rad(-35.0)
    )
    return names, q, ready_q, root_pos, root_quat, ready_pos, ready_quat


def test_upper_resolves_groups_by_name_and_folds_complete_relative_root_so3():
    names, q, ready_q, root_pos, root_quat, ready_pos, ready_quat = _fixture(
        shuffled=True
    )
    result = scope.project_upper_body_scope(
        source_joint_pos=q,
        source_root_pos_w=root_pos,
        source_root_quat_w=root_quat,
        joint_names=names,
        canonical_ready_joint_pos=ready_q,
        canonical_ready_root_pos_w=ready_pos,
        canonical_ready_root_quat_w=ready_quat,
    )
    index = {name: i for i, name in enumerate(names)}
    fixed = [index[name] for name in scope.UPPER_READY_FIXED_JOINT_NAMES]
    np.testing.assert_array_equal(
        result.joint_pos[:, fixed],
        np.broadcast_to(ready_q[fixed], (q.shape[0], len(fixed))),
    )
    for name in scope.UPPER_ACTIVE_JOINT_NAMES:
        if name not in scope.WAIST_JOINT_NAMES:
            np.testing.assert_array_equal(
                result.joint_pos[:, index[name]], q[:, index[name]]
            )

    source_root = scope.quat_to_matrix_wxyz(root_quat)
    relative_root = np.einsum(
        "ji,tjk->tik", source_root[0], source_root
    )
    source_waist = scope.waist_zxy_matrix(
        q[:, index["waist_yaw_joint"]],
        q[:, index["waist_roll_joint"]],
        q[:, index["waist_pitch_joint"]],
    )
    expected_torso = relative_root @ source_waist
    output_waist = scope.waist_zxy_matrix(
        result.joint_pos[:, index["waist_yaw_joint"]],
        result.joint_pos[:, index["waist_roll_joint"]],
        result.joint_pos[:, index["waist_pitch_joint"]],
    )
    np.testing.assert_allclose(output_waist, expected_torso, atol=1e-12)
    for name in scope.WAIST_JOINT_NAMES:
        np.testing.assert_allclose(
            result.joint_pos[0, index[name]], q[0, index[name]], atol=1e-12
        )
    # The source starts at a global station orientation.  That constant
    # orientation is deliberately not copied into the waist.
    fold = result.report["relative_root_so3_fold"]
    assert fold["global_frame0_orientation_folded"] is False
    assert fold["max_reconstruction_error_rad"] < 1.0e-7


def test_upper_root_is_exact_canonical_ready_and_report_is_json_safe():
    names, q, ready_q, root_pos, root_quat, ready_pos, ready_quat = _fixture()
    result = scope.project_upper_body_scope(
        source_joint_pos=q,
        source_root_pos_w=root_pos,
        source_root_quat_w=root_quat,
        joint_names=names,
        canonical_ready_joint_pos=ready_q,
        canonical_ready_root_pos_w=ready_pos,
        canonical_ready_root_quat_w=ready_quat,
    )
    np.testing.assert_allclose(
        result.root_pos_w, np.broadcast_to(ready_pos, result.root_pos_w.shape)
    )
    np.testing.assert_allclose(
        result.root_quat_w, np.broadcast_to(ready_quat, result.root_quat_w.shape)
    )
    assert result.report["root_policy"]["mode"] == "fixed_canonical_ready"
    assert result.report["deleted_root_motion_gate"]["passed"] is True
    json.dumps(result.report, allow_nan=False)


def test_upper_waist_range_and_deleted_translation_fail_closed():
    names, q, ready_q, root_pos, _, ready_pos, ready_quat = _fixture()
    frames = q.shape[0]
    index = {name: i for i, name in enumerate(names)}
    for name in scope.WAIST_JOINT_NAMES:
        q[:, index[name]] = 0.0
    excessive_tilt = _quat_from_rpy(
        np.deg2rad(np.linspace(0.0, 25.0, frames)),
        np.zeros(frames),
        np.full(frames, np.deg2rad(40.0)),
    )
    with pytest.raises(scope.BodyScopeError, match="waist_roll"):
        scope.project_upper_body_scope(
            source_joint_pos=q,
            source_root_pos_w=root_pos,
            source_root_quat_w=excessive_tilt,
            joint_names=names,
            canonical_ready_joint_pos=ready_q,
            canonical_ready_root_pos_w=ready_pos,
            canonical_ready_root_quat_w=ready_quat,
        )

    translated = root_pos.copy()
    translated[-1, 0] = translated[0, 0] + 0.11
    benign_quat = scope.yaw_quat_wxyz(np.deg2rad(np.linspace(40.0, 45.0, frames)))
    with pytest.raises(scope.BodyScopeError, match="translation"):
        scope.project_upper_body_scope(
            source_joint_pos=q,
            source_root_pos_w=translated,
            source_root_quat_w=benign_quat,
            joint_names=names,
            canonical_ready_joint_pos=ready_q,
            canonical_ready_root_pos_w=ready_pos,
            canonical_ready_root_quat_w=ready_quat,
        )


def test_full_applies_one_atomic_se2_and_preserves_joints_and_local_motion():
    names, q, _, root_pos, root_quat, ready_pos, ready_quat = _fixture()
    result = scope.align_full_body_scope(
        source_joint_pos=q,
        source_root_pos_w=root_pos,
        source_root_quat_w=root_quat,
        joint_names=names,
        canonical_ready_root_pos_w=ready_pos,
        canonical_ready_root_quat_w=ready_quat,
    )
    np.testing.assert_array_equal(result.joint_pos, q)
    np.testing.assert_allclose(result.root_pos_w[0, :2], ready_pos[:2], atol=1e-12)
    got_yaw = scope.yaw_from_quat_wxyz(result.root_quat_w)
    ready_yaw = float(scope.yaw_from_quat_wxyz(ready_quat))
    assert abs(float(scope.wrap_to_pi(got_yaw[0] - ready_yaw))) < 1e-12

    delta_yaw = result.report["root_policy"]["se2"]["yaw_rad"]
    expected_local_xy = scope.rotate_xy(
        root_pos[:, :2] - root_pos[0, :2], delta_yaw
    )
    np.testing.assert_allclose(
        result.root_pos_w[:, :2] - result.root_pos_w[0, :2],
        expected_local_xy,
        atol=1e-12,
    )
    source_relative_yaw = np.unwrap(scope.yaw_from_quat_wxyz(root_quat))
    source_relative_yaw -= source_relative_yaw[0]
    output_relative_yaw = np.unwrap(got_yaw)
    output_relative_yaw -= output_relative_yaw[0]
    np.testing.assert_allclose(output_relative_yaw, source_relative_yaw, atol=1e-12)
    np.testing.assert_array_equal(result.root_pos_w[:, 2], root_pos[:, 2])
    assert result.report["root_policy"]["se2"]["applied_once_to_all_frames"] is True


def test_full_grounding_is_explicit_bounded_and_marks_new_asset():
    names, q, _, root_pos, root_quat, ready_pos, ready_quat = _fixture()
    result = scope.align_full_body_scope(
        source_joint_pos=q,
        source_root_pos_w=root_pos,
        source_root_quat_w=root_quat,
        joint_names=names,
        canonical_ready_root_pos_w=ready_pos,
        canonical_ready_root_quat_w=ready_quat,
        grounding_z_offset_m=-0.02,
    )
    np.testing.assert_allclose(result.root_pos_w[:, 2], root_pos[:, 2] - 0.02)
    assert result.report["grounding"] == {
        "requested_explicitly": True,
        "z_policy": "constant_bounded_grounding_offset",
        "z_offset_m": -0.02,
        "max_abs_correction_m": 0.05,
        "requires_new_derived_asset_record": True,
        "passed": True,
    }

    with pytest.raises(scope.BodyScopeError, match="grounding correction"):
        scope.align_full_body_scope(
            source_joint_pos=q,
            source_root_pos_w=root_pos,
            source_root_quat_w=root_quat,
            joint_names=names,
            canonical_ready_root_pos_w=ready_pos,
            canonical_ready_root_quat_w=ready_quat,
            grounding_z_offset_m=0.051,
        )


def test_joint_domain_and_dispatch_fail_loud():
    names, q, ready_q, root_pos, root_quat, ready_pos, ready_quat = _fixture()
    bad_names = list(names)
    bad_names[-1] = "mystery_joint"
    with pytest.raises(scope.BodyScopeError, match="A3 31-DOF"):
        scope.project_upper_body_scope(
            source_joint_pos=q,
            source_root_pos_w=root_pos,
            source_root_quat_w=root_quat,
            joint_names=bad_names,
            canonical_ready_joint_pos=ready_q,
            canonical_ready_root_pos_w=ready_pos,
            canonical_ready_root_quat_w=ready_quat,
        )
    with pytest.raises(scope.BodyScopeError, match="upper.*full"):
        scope.preprocess_body_scope("other")


def test_quaternion_and_se2_helpers_use_wxyz_world_z_convention():
    yaw = np.deg2rad(np.array([-170.0, -10.0, 40.0, 179.0]))
    np.testing.assert_allclose(
        scope.wrap_to_pi(scope.yaw_from_quat_wxyz(scope.yaw_quat_wxyz(yaw)) - yaw),
        0.0,
        atol=1e-12,
    )
    points = np.array([[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(
        scope.rotate_xy(points, math.pi / 2.0),
        np.array([[0.0, 1.0], [-2.0, 0.0]]),
        atol=1e-12,
    )
