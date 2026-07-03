"""Tests for the sim2real frame alignment (pure numpy, no ROS / no ONNX)."""

import math

import numpy as np

from hope_wbc_runner.reference_clock import swing_sign_from_target_y
from hope_wbc_runner.world_frame import (
    ImuYawAligner,
    OriginCapture,
    TableToPolicy,
    TargetGate,
    base_relative_target,
)


def _quat_z(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# TableToPolicy
# ---------------------------------------------------------------------------

def test_table_to_policy_yaw0_translation():
    # robot boots at table (-0.5, -0.7625) facing +x; floor at table z=-0.76.
    t2p = TableToPolicy(origin_xy_table=np.array([-0.5, -0.7625]), yaw_table=0.0)
    # planner target on the hit plane, table frame: x=0, y=table centerline, z=15cm above surface
    p = t2p.pos(np.array([0.0, -0.7625, 0.15]))
    np.testing.assert_allclose(p, [0.5, 0.0, 0.91], atol=1e-12)  # z = 0.15 + 0.76 above floor


def test_table_to_policy_velocity_is_rotation_only():
    t2p = TableToPolicy(origin_xy_table=np.array([-0.5, -0.7625]), yaw_table=0.0)
    v = t2p.vec(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(v, [1.0, 2.0, 3.0], atol=1e-12)  # yaw 0: untouched (no translation!)


def test_table_to_policy_yaw90():
    # robot facing table +y: a point 1 m to the table-left is 1 m AHEAD of the robot
    t2p = TableToPolicy(origin_xy_table=np.array([0.0, 0.0]), yaw_table=math.pi / 2)
    p = t2p.pos(np.array([0.0, 1.0, -0.76]))          # on the floor, 1 m along table +y
    np.testing.assert_allclose(p, [1.0, 0.0, 0.0], atol=1e-12)
    v = t2p.vec(np.array([0.0, 2.0, 0.0]))            # table +y velocity -> policy +x
    np.testing.assert_allclose(v, [2.0, 0.0, 0.0], atol=1e-12)


def test_table_to_policy_quat_yaw_composition():
    t2p = TableToPolicy(origin_xy_table=np.array([0.0, 0.0]), yaw_table=math.pi / 2)
    # a body whose yaw is +90 deg in the table frame has yaw 0 in the policy frame
    q = t2p.quat(_quat_z(math.pi / 2))
    np.testing.assert_allclose(np.abs(q @ IDENTITY), 1.0, atol=1e-12)


def test_roundtrip_relative_quantities_cancel_origin():
    # base-relative target must be identical whether computed in table or policy frame
    t2p = TableToPolicy(origin_xy_table=np.array([-0.5, -0.7625]), yaw_table=0.0)
    tgt_table = np.array([0.1, -0.45, 0.2])
    base_table = np.array([-0.45, -0.75, 0.03])
    rel_policy = t2p.pos(tgt_table) - t2p.pos(base_table)
    np.testing.assert_allclose(rel_policy, tgt_table - base_table, atol=1e-12)


# ---------------------------------------------------------------------------
# Swing side: the bug this module exists to fix
# ---------------------------------------------------------------------------

def test_backhand_target_in_table_frame_selects_backhand():
    """In the TABLE frame every y is negative (table spans 0..-1.525), so the raw
    world-Y sign convention would ALWAYS pick forehand. Base-relative Y must be used."""
    t2p = TableToPolicy(origin_xy_table=np.array([-0.5, -0.7625]), yaw_table=0.0)
    # backhand: target 0.3 m to the robot's LEFT of its centerline y=-0.7625
    tgt_table = np.array([0.1, -0.4625, 0.25])
    assert tgt_table[1] < 0.0                     # raw sign would say "forehand" — wrong
    tgt_policy = t2p.pos(tgt_table)
    base_policy = np.array([0.0, 0.0, 0.79])      # robot at its start pose
    rel = base_relative_target(tgt_policy, base_policy, IDENTITY)
    assert rel[1] > 0.0
    assert swing_sign_from_target_y(rel[1]) == -1.0   # backhand


def test_forehand_target_in_table_frame_selects_forehand():
    t2p = TableToPolicy(origin_xy_table=np.array([-0.5, -0.7625]), yaw_table=0.0)
    tgt_table = np.array([0.1, -1.0625, 0.25])    # 0.3 m to the robot's RIGHT
    rel = base_relative_target(t2p.pos(tgt_table), np.array([0.0, 0.0, 0.79]), IDENTITY)
    assert rel[1] < 0.0
    assert swing_sign_from_target_y(rel[1]) == 1.0    # forehand


def test_base_relative_target_uses_yaw_heading_frame():
    # base yawed +90 deg: a target ahead of the BODY is along policy +y
    base = np.array([0.0, 0.0, 0.8])
    tgt = np.array([0.0, 0.7, 1.0])
    rel = base_relative_target(tgt, base, _quat_z(math.pi / 2))
    np.testing.assert_allclose(rel, [0.7, 0.0, 0.2], atol=1e-12)


# ---------------------------------------------------------------------------
# OriginCapture / ImuYawAligner
# ---------------------------------------------------------------------------

def test_origin_capture_averages_xy():
    cap = OriginCapture(n_samples=4)
    for p in ([1.0, 2.0, 0.5], [1.2, 1.8, 0.4], [0.8, 2.2, 0.6]):
        assert cap.push(np.array(p)) is None
    xy = cap.push(np.array([1.0, 2.0, 0.5]))
    np.testing.assert_allclose(xy, [1.0, 2.0], atol=1e-12)
    assert cap.done
    assert cap.push(np.array([9.0, 9.0, 9.0])) is None  # locked after capture


def test_imu_yaw_aligner_rezeros_boot_yaw():
    al = ImuYawAligner(n_samples=5)
    for _ in range(5):
        al.push(_quat_z(math.radians(30.0)))
    assert al.done
    assert abs(al.offset + math.radians(30.0)) < 1e-9
    q = al.correct(_quat_z(math.radians(30.0)))
    # corrected quat has yaw 0
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
    assert abs(yaw) < 1e-9


def test_imu_yaw_aligner_circular_mean_across_wrap():
    al = ImuYawAligner(n_samples=2)
    al.push(_quat_z(math.pi - 0.01))
    al.push(_quat_z(-math.pi + 0.01))
    assert al.done
    # mean of +/-(pi-0.01) is pi (not 0): offset must be ~ -pi (mod 2pi)
    assert abs(abs(al.offset) - math.pi) < 1e-6


def test_imu_yaw_aligner_passthrough_before_done():
    al = ImuYawAligner(n_samples=10)
    q = _quat_z(0.7)
    np.testing.assert_allclose(al.correct(q), q, atol=1e-12)


# ---------------------------------------------------------------------------
# TargetGate
# ---------------------------------------------------------------------------

def test_target_gate_accepts_trained_box():
    gate = TargetGate()
    ok, why = gate.check(np.array([0.65, -0.35, 0.85]), np.array([1.5, 1.2, 0.5]))
    assert ok, why


def test_target_gate_rejects_out_of_box_and_fast():
    gate = TargetGate()
    assert not gate.check(np.array([1.5, 0.0, 0.9]), np.zeros(3))[0]    # x too far
    assert not gate.check(np.array([0.6, -1.2, 0.9]), np.zeros(3))[0]   # |y| too far
    assert not gate.check(np.array([0.6, 0.0, 0.2]), np.zeros(3))[0]    # z too low (near legs)
    assert not gate.check(np.array([0.6, 0.0, 0.9]), np.array([4.0, 0.0, 0.0]))[0]  # too fast


def test_target_gate_disabled_accepts_everything():
    gate = TargetGate(enabled=False)
    assert gate.check(np.array([99.0, 99.0, -99.0]), np.array([99.0, 0.0, 0.0]))[0]
