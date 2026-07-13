"""CPU regression for the teacher-reference pelvis velocity point/frame conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
MODULE_PATH = WBT_ROOT / "scripts" / "mujoco_eval_onnx.py"
A3_MJCF = (
    REPO_ROOT
    / "agi"
    / "A3_MuJoCo_Sim"
    / "aimrt_mujoco_sim"
    / "src"
    / "models"
    / "bin"
    / "cfg"
    / "model"
    / "a3_pingpong"
    / "a3_pingpong.xml"
)


def _load_evaluator_module():
    spec = importlib.util.spec_from_file_location("mj_eval_com_reset_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_reference_reset_respects_declared_pelvis_velocity_point_and_frames():
    """Exercise both velocity-point branches against the real non-zero-offset A3 pelvis."""

    mujoco = pytest.importorskip("mujoco")
    evaluator = _load_evaluator_module()
    model = mujoco.MjModel.from_xml_path(str(A3_MJCF))
    data = mujoco.MjData(model)
    pelvis_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link"
    )
    free_jids = np.flatnonzero(
        (np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE))
        & (np.asarray(model.jnt_bodyid) == pelvis_bid)
    )
    assert free_jids.shape == (1,)
    assert int(model.jnt_bodyid[free_jids[0]]) == pelvis_bid
    assert int(model.jnt_qposadr[free_jids[0]]) == 0
    assert int(model.jnt_dofadr[free_jids[0]]) == 0

    # Build only the fields used by the production reset method.  Joint addresses come from the
    # real model, so this test remains sensitive to the A3 freejoint/layout contract without
    # requiring an ONNX policy or Isaac runtime.
    robot = object.__new__(evaluator.MujocoRobot)
    robot.mj = mujoco
    robot.model = model
    robot.data = data
    robot.pelvis_bid = pelvis_bid
    articulated_jids = [
        jid
        for jid in range(model.njnt)
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE
    ]
    robot.qadr = np.asarray([model.jnt_qposadr[jid] for jid in articulated_jids], dtype=int)
    robot.vadr = np.asarray([model.jnt_dofadr[jid] for jid in articulated_jids], dtype=int)
    assert robot.qadr.shape == robot.vadr.shape == (31,)

    stand_kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    q_artic = np.asarray(model.key_qpos[stand_kid, robot.qadr], dtype=np.float64)
    qd_artic = np.linspace(-0.2, 0.2, len(robot.vadr), dtype=np.float64)

    # Deliberately use a tilted/yawed pelvis and a world angular velocity with all components.
    # MuJoCo and the motion schema both use scalar-first wxyz for root orientation.
    axis = np.asarray([0.37, -0.58, 0.72], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.83
    root_quat_wxyz = np.concatenate(
        ([np.cos(angle / 2.0)], axis * np.sin(angle / 2.0))
    )
    root_pos_w = np.asarray([0.23, -0.17, 1.14], dtype=np.float64)
    requested_com_lin_vel_w = np.asarray([0.71, -0.43, 0.26], dtype=np.float64)
    requested_ang_vel_w = np.asarray([1.17, -0.64, 0.91], dtype=np.float64)

    # The production A3 pelvis has a material link-origin -> COM offset (~12.7 cm).
    pelvis_com_pos_b = np.asarray(model.body_ipos[pelvis_bid], dtype=np.float64)
    assert np.linalg.norm(pelvis_com_pos_b) > 0.12
    # Its off-diagonal fullinertia also compiles to a non-identity inertial-frame rotation.  This
    # makes mjOBJ_BODY/local observably different from the pelvis link frame used by the policy.
    assert np.linalg.norm(model.body_iquat[pelvis_bid, 1:]) > 1.0e-3

    robot.reset_to_reference(
        root_pos=root_pos_w,
        root_quat=root_quat_wxyz,
        root_lin_w=requested_com_lin_vel_w,
        root_ang_w=requested_ang_vel_w,
        root_lin_vel_point=evaluator.ROOT_LIN_VEL_POINT_CENTER_OF_MASS,
        q_artic=q_artic,
        qd_artic=qd_artic,
    )

    R_wb = evaluator.mat_from_quat(root_quat_wxyz)
    pelvis_com_offset_w = R_wb @ pelvis_com_pos_b
    expected_origin_lin_vel_w = requested_com_lin_vel_w - np.cross(
        requested_ang_vel_w, pelvis_com_offset_w
    )

    # body_ipos is already expressed in the body/link frame; body_iquat is irrelevant to the COM
    # point.  These assertions also pin the freejoint angular qvel to the pelvis body frame.
    np.testing.assert_allclose(
        data.xipos[pelvis_bid] - data.xpos[pelvis_bid], pelvis_com_offset_w, atol=1e-12
    )
    np.testing.assert_allclose(data.qvel[0:3], expected_origin_lin_vel_w, atol=1e-12)
    np.testing.assert_allclose(data.qvel[3:6], R_wb.T @ requested_ang_vel_w, atol=1e-12)
    np.testing.assert_allclose(
        robot.pelvis_ang_vel_body(), data.qvel[3:6], atol=1e-12
    )

    pelvis_origin_velocity_w = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_XBODY,
        pelvis_bid,
        pelvis_origin_velocity_w,
        0,
    )
    np.testing.assert_allclose(
        pelvis_origin_velocity_w[:3], requested_ang_vel_w, atol=1e-12
    )
    np.testing.assert_allclose(
        pelvis_origin_velocity_w[3:6], expected_origin_lin_vel_w, atol=1e-12
    )

    pelvis_com_velocity_w = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        pelvis_bid,
        pelvis_com_velocity_w,
        0,
    )
    np.testing.assert_allclose(
        pelvis_com_velocity_w[:3], requested_ang_vel_w, atol=1e-12
    )
    np.testing.assert_allclose(
        pelvis_com_velocity_w[3:6], requested_com_lin_vel_w, atol=1e-12
    )

    requested_origin_lin_vel_w = np.asarray([-0.34, 0.52, -0.19], dtype=np.float64)
    robot.reset_to_reference(
        root_pos=root_pos_w,
        root_quat=root_quat_wxyz,
        root_lin_w=requested_origin_lin_vel_w,
        root_ang_w=requested_ang_vel_w,
        root_lin_vel_point=evaluator.ROOT_LIN_VEL_POINT_LINK_ORIGIN,
        q_artic=q_artic,
        qd_artic=qd_artic,
    )
    np.testing.assert_allclose(data.qvel[0:3], requested_origin_lin_vel_w, atol=1e-12)

    legacy_origin_velocity_w = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_XBODY,
        pelvis_bid,
        legacy_origin_velocity_w,
        0,
    )
    np.testing.assert_allclose(
        legacy_origin_velocity_w[3:6], requested_origin_lin_vel_w, atol=1e-12
    )

    legacy_com_velocity_w = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        pelvis_bid,
        legacy_com_velocity_w,
        0,
    )
    np.testing.assert_allclose(
        legacy_com_velocity_w[3:6],
        requested_origin_lin_vel_w
        + np.cross(requested_ang_vel_w, pelvis_com_offset_w),
        atol=1e-12,
    )
