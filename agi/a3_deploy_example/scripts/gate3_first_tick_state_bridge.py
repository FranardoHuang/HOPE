#!/usr/bin/env python3
"""Read-only vendor-sim state bridge for the production first-tick JSON.

This sidecar subscribes three already-published Agibot MuJoCo diagnostic topics
and exposes their closest-receipt values through one mode-0600 kernel-locked file. It
never publishes, resets, commands, estimates, differentiates, or fills missing
values with zero. The topics carry no common simulator-sample sequence, so the
output is explicitly inexact; C++ rejects incomplete/stale/skewed records.

The ROS imports live inside ``main`` so the ABI packer remains dependency-light
and can be unit-tested on a non-ROS host.
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
from pathlib import Path
import signal
import stat
import struct
import threading
from typing import Iterable


MAGIC = 0x3153544633475050  # "PPG3FTS1" in little-endian memory.
VERSION = 1
WIRE = struct.Struct("<QIIQ9q20d")
SEQUENCE_OFFSET = 16
DEFAULT_SHM = "/dev/shm/pp_gate3_first_tick_state_v1"


def _finite(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("native simulator message contains NaN/Inf")
    return result


def _unit_quaternion_wxyz(values: Iterable[float]) -> tuple[float, ...]:
    result = _finite(values)
    if len(result) != 4:
        raise ValueError("quaternion must have four elements")
    norm = math.sqrt(sum(value * value for value in result))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"native quaternion is not unit length: {norm:.17g}")
    return result


def _canonical_new_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or path != Path(os.path.normpath(str(path))):
        raise ValueError("--shm must be an absolute lexical-canonical path")
    parent = path.parent
    if not parent.is_dir():
        raise ValueError(f"--shm parent does not exist: {parent}")
    canonical_parent = Path(os.path.realpath(parent))
    expected = canonical_parent / path.name
    if expected != path:
        raise ValueError(f"--shm is not canonical (expected {expected})")
    cursor = Path(path.anchor)
    for component in path.parts[1:-1]:
        cursor /= component
        if stat.S_ISLNK(os.lstat(cursor).st_mode):
            raise ValueError(f"symlink path component is forbidden: {cursor}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"state bridge refuses to clobber {path}")
    return path


class NativeStateFile:
    """Exclusive owned kernel-locked file with the C++ ``NativeStateWire`` ABI."""

    def __init__(self, path: str):
        self.path = _canonical_new_path(path)
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.fd = os.open(self.path, flags, 0o600)
        os.fchmod(self.fd, 0o600)
        os.ftruncate(self.fd, WIRE.size)
        opened = os.fstat(self.fd)
        self.identity = opened.st_dev, opened.st_ino
        self.lock = threading.Lock()
        self.sequence = 0
        nan3 = (math.nan,) * 3
        nan4 = (math.nan,) * 4
        self.values: dict[str, tuple[float, ...] | int] = {
            "base_pose_stamp_ns": 0,
            "base_twist_stamp_ns": 0,
            "racket_pose_stamp_ns": 0,
            "base_pose_receive_monotonic_ns": 0,
            "base_twist_receive_monotonic_ns": 0,
            "racket_pose_receive_monotonic_ns": 0,
            "base_pose_receive_system_ns": 0,
            "base_twist_receive_system_ns": 0,
            "racket_pose_receive_system_ns": 0,
            "base_position_world": nan3,
            "base_quaternion_wxyz": nan4,
            "base_linear_velocity_world": nan3,
            "base_angular_velocity_world": nan3,
            "racket_position_world": nan3,
            "racket_quaternion_wxyz": nan4,
        }
        self._commit_locked()

    def _commit_locked(self) -> None:
        self.sequence += 2
        value = self.values
        packed = WIRE.pack(
            MAGIC,
            VERSION,
            WIRE.size,
            self.sequence,
            int(value["base_pose_stamp_ns"]),
            int(value["base_twist_stamp_ns"]),
            int(value["racket_pose_stamp_ns"]),
            int(value["base_pose_receive_monotonic_ns"]),
            int(value["base_twist_receive_monotonic_ns"]),
            int(value["racket_pose_receive_monotonic_ns"]),
            int(value["base_pose_receive_system_ns"]),
            int(value["base_twist_receive_system_ns"]),
            int(value["racket_pose_receive_system_ns"]),
            *value["base_position_world"],
            *value["base_quaternion_wxyz"],
            *value["base_linear_velocity_world"],
            *value["base_angular_velocity_world"],
            *value["racket_position_world"],
            *value["racket_quaternion_wxyz"],
        )
        # Kernel-mediated cross-process transaction: Python holds LOCK_EX for
        # one whole-record pwrite; C++ holds LOCK_SH for one whole-record pread.
        # No cross-language mmap atomic/fence assumption is used.
        import fcntl
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            written = os.pwrite(self.fd, packed, 0)
            if written != WIRE.size:
                raise OSError(f"short native-state pwrite: {written}/{WIRE.size}")
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def update(self, **values: tuple[float, ...] | int) -> None:
        with self.lock:
            # A freshly received replay must not make an old simulator sample
            # look current. Each topic's source header must advance strictly;
            # a reset/regression requires restarting this one-shot sidecar.
            for key, raw_value in values.items():
                if not key.endswith("_stamp_ns") or "receive_" in key:
                    continue
                next_stamp = int(raw_value)
                previous_stamp = int(self.values[key])
                if previous_stamp > 0 and next_stamp <= previous_stamp:
                    raise ValueError(
                        f"native topic header stamp did not advance: "
                        f"{key}={next_stamp} <= {previous_stamp}"
                    )
            self.values.update(values)
            self._commit_locked()

    def close(self) -> None:
        if getattr(self, "fd", -1) >= 0:
            os.close(self.fd)
            self.fd = -1
        try:
            st = os.lstat(self.path)
            if not stat.S_ISLNK(st.st_mode) and (st.st_dev, st.st_ino) == self.identity:
                os.unlink(self.path)
        except FileNotFoundError:
            pass


def _stamp_ns(header: object) -> int:
    stamp = header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value <= 0:
        raise ValueError("native topic header stamp must be positive")
    if header.frame_id != "odom":
        raise ValueError(f"native topic frame_id must be 'odom', got {header.frame_id!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm", default=DEFAULT_SHM)
    args = parser.parse_args()

    # Imported only for the actual ROS sidecar.  Importing this module for ABI
    # tests performs no ROS discovery and creates no file/process.
    import rclpy
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from rclpy.qos import qos_profile_sensor_data

    output = NativeStateFile(args.shm)
    # Covers rclpy.init/node/subscription failures that occur before the spin
    # try/finally is installed. Identity-checked close is idempotent.
    atexit.register(output.close)
    rclpy.init()
    node = rclpy.create_node("gate3_first_tick_state_bridge")

    def monotonic_ns() -> int:
        import time
        return time.monotonic_ns()

    def system_ns() -> int:
        import time
        return time.time_ns()

    def base_pose(message: PoseStamped) -> None:
        try:
            output.update(
                base_pose_stamp_ns=_stamp_ns(message.header),
                base_pose_receive_monotonic_ns=monotonic_ns(),
                base_pose_receive_system_ns=system_ns(),
                base_position_world=_finite((
                    message.pose.position.x,
                    message.pose.position.y,
                    message.pose.position.z,
                )),
                base_quaternion_wxyz=_unit_quaternion_wxyz((
                    message.pose.orientation.w,
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                )),
            )
        except ValueError as exc:
            node.get_logger().error(f"reject /sim/a3/pelvis_pose: {exc}")

    def base_twist(message: TwistStamped) -> None:
        try:
            output.update(
                base_twist_stamp_ns=_stamp_ns(message.header),
                base_twist_receive_monotonic_ns=monotonic_ns(),
                base_twist_receive_system_ns=system_ns(),
                base_linear_velocity_world=_finite((
                    message.twist.linear.x,
                    message.twist.linear.y,
                    message.twist.linear.z,
                )),
                base_angular_velocity_world=_finite((
                    message.twist.angular.x,
                    message.twist.angular.y,
                    message.twist.angular.z,
                )),
            )
        except ValueError as exc:
            node.get_logger().error(f"reject /sim/a3/pelvis_twist: {exc}")

    def racket_pose(message: PoseStamped) -> None:
        try:
            output.update(
                racket_pose_stamp_ns=_stamp_ns(message.header),
                racket_pose_receive_monotonic_ns=monotonic_ns(),
                racket_pose_receive_system_ns=system_ns(),
                racket_position_world=_finite((
                    message.pose.position.x,
                    message.pose.position.y,
                    message.pose.position.z,
                )),
                racket_quaternion_wxyz=_unit_quaternion_wxyz((
                    message.pose.orientation.w,
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                )),
            )
        except ValueError as exc:
            node.get_logger().error(f"reject /sim/a3/right_racket_pose: {exc}")

    node.create_subscription(
        PoseStamped, "/sim/a3/pelvis_pose", base_pose, qos_profile_sensor_data
    )
    node.create_subscription(
        TwistStamped, "/sim/a3/pelvis_twist", base_twist, qos_profile_sensor_data
    )
    node.create_subscription(
        PoseStamped,
        "/sim/a3/right_racket_pose",
        racket_pose,
        qos_profile_sensor_data,
    )
    node.get_logger().info(
        f"read-only Gate3 first-tick native state -> {output.path}; "
        "no publisher/reset/command exists"
    )

    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        output.close()
        atexit.unregister(output.close)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
