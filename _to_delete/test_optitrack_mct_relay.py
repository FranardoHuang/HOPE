"""Smoke tests: the OptiTrack relay maps NamedPoseArray onto the HOPE contract.

Covers the review findings for ``optitrack_mct_relay``:

  * the ball rides at ``/poses`` index 0 with its quaternion preserved (strict
    6-DOF rigid-body ball — the HOPE spec);
  * ``/poses`` is ball-gated: frames without a ball entry publish nothing;
  * stale non-ball bodies are pruned after ``body_stale_s`` (a calibration-time
    ``Table`` must not be re-emitted forever);
  * ``pose_array_order`` is validated (unknown keys / missing ``ball`` raise).

Skipped automatically when ``rclpy`` or the vendored
``motion_capture_tracking_interfaces`` package is unavailable (needs a sourced,
built ROS 2 workspace).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import types

import pytest

rclpy = pytest.importorskip("rclpy")
mct_msgs = pytest.importorskip("motion_capture_tracking_interfaces.msg")

from geometry_msgs.msg import PoseArray  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "optitrack_mct_relay"

_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
SPIN = (0.5, 0.5, 0.5, 0.5)  # a distinctly non-identity unit quaternion


def _load_relay_module() -> types.ModuleType:
    loader = importlib.machinery.SourceFileLoader("optitrack_mct_relay_script", str(_SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _named_frame(entries, sec: int, frame: str = "world"):
    msg = mct_msgs.NamedPoseArray()
    msg.header.frame_id = frame
    msg.header.stamp.sec = sec
    for name, xyz, quat in entries:
        np_ = mct_msgs.NamedPose()
        np_.name = name
        np_.pose.position.x, np_.pose.position.y, np_.pose.position.z = xyz
        (np_.pose.orientation.x, np_.pose.orientation.y,
         np_.pose.orientation.z, np_.pose.orientation.w) = quat
        msg.poses.append(np_)
    return msg


@pytest.fixture()
def ros_context():
    ctx = rclpy.Context()
    rclpy.init(context=ctx, args=None)
    yield ctx
    rclpy.shutdown(context=ctx)


def _make_relay(module, ctx, **overrides):
    return module.OptitrackMctRelay(
        mct_msgs.NamedPoseArray,
        context=ctx,
        parameter_overrides=[Parameter(k, value=v) for k, v in overrides.items()],
    )


def _pump(executor, publish, predicate, timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        publish()
        executor.spin_once(timeout_sec=0.05)


def test_ball_first_with_quaternion_and_gating(ros_context):
    """Ball at index 0 with quaternion preserved; ball-less frames publish nothing."""
    module = _load_relay_module()
    relay = _make_relay(module, ros_context)
    helper = rclpy.create_node("mct_relay_test_helper", context=ros_context)
    received: list[PoseArray] = []
    helper.create_subscription(PoseArray, "/poses", received.append, _QOS)
    pub = helper.create_publisher(mct_msgs.NamedPoseArray, "/optitrack/poses", _QOS)

    executor = SingleThreadedExecutor(context=ros_context)
    executor.add_node(relay)
    executor.add_node(helper)
    ball_less = _named_frame([("P1", (0.1, -0.9, 1.0), IDENTITY)], sec=1)
    with_ball = _named_frame(
        [("Ball", (1.25, -0.5, 0.31), SPIN), ("P1", (0.1, -0.9, 1.0), IDENTITY)], sec=2
    )
    try:
        # Ball-less frames first: nothing may appear on /poses.
        for _ in range(10):
            pub.publish(ball_less)
            executor.spin_once(timeout_sec=0.02)
        assert not received, "/poses published without a ball entry (gating broken)"

        _pump(executor, lambda: pub.publish(with_ball), lambda: received)
        assert received, "relay never published /poses for a ball-bearing frame"
        out = received[-1]
        assert out.poses[0].position.x == pytest.approx(1.25)   # ball is index 0
        q = out.poses[0].orientation
        assert (q.x, q.y, q.z, q.w) == tuple(pytest.approx(c) for c in SPIN)
        assert out.header.stamp.sec == 2                        # ball stamp passed through
    finally:
        relay.destroy_node()
        helper.destroy_node()


def test_stale_bodies_pruned_from_poses(ros_context):
    """A Table seen once (calibration) must drop out of /poses after body_stale_s."""
    module = _load_relay_module()
    relay = _make_relay(module, ros_context, body_stale_s=0.5)
    helper = rclpy.create_node("mct_relay_test_helper2", context=ros_context)
    received: list[PoseArray] = []
    helper.create_subscription(PoseArray, "/poses", received.append, _QOS)
    pub = helper.create_publisher(mct_msgs.NamedPoseArray, "/optitrack/poses", _QOS)

    executor = SingleThreadedExecutor(context=ros_context)
    executor.add_node(relay)
    executor.add_node(helper)
    calib = _named_frame(
        [("Ball", (1.0, -0.5, 0.3), IDENTITY), ("Table", (0.0, 0.0, 0.0), IDENTITY)], sec=10
    )
    # Much later (stamp-wise), the Table asset is disabled: ball-only frames.
    competition = _named_frame([("Ball", (1.1, -0.6, 0.4), IDENTITY)], sec=20)
    try:
        _pump(executor, lambda: pub.publish(calib), lambda: received)
        assert received and len(received[-1].poses) == 2, "calibration frame should carry ball+Table"

        received.clear()
        _pump(executor, lambda: pub.publish(competition), lambda: received)
        assert received, "relay stopped publishing after Table disappeared"
        assert len(received[-1].poses) == 1, (
            "stale Table pose re-emitted after body_stale_s — pruning broken"
        )
        assert received[-1].poses[0].position.x == pytest.approx(1.1)
    finally:
        relay.destroy_node()
        helper.destroy_node()


def test_pose_array_order_validated(ros_context):
    """Unknown keys (e.g. 'Ball' instead of internal 'ball') must raise, not silently misfeed."""
    module = _load_relay_module()
    with pytest.raises(Exception, match="pose_array_order"):
        _make_relay(module, ros_context, pose_array_order=["Ball", "Table", "P1", "P2"])
    with pytest.raises(Exception, match="ball"):
        _make_relay(module, ros_context, pose_array_order=["Table", "P1", "P2"])
