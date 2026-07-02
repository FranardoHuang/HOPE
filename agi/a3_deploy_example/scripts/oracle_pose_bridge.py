#!/usr/bin/env python3
# ============================================================================
#  SIMULATION-ONLY oracle pelvis-pose bridge.
# ============================================================================
#  Subscribes the MuJoCo sim ground-truth pelvis pose (ROS 2, geometry_msgs/
#  PoseStamped on /sim/a3/pelvis_pose) and writes it into a small shared-memory
#  file that the ping-pong runner reads (a3_pingpong/pp_oracle_pose.hpp), so the
#  runner needs ZERO ROS 2 linkage.
#
#  >>> HARD SAFETY RULE: RUN THIS ONLY IN SIMULATION. <<<
#  The real robot has no ground-truth pelvis pose. Running this on hardware is
#  meaningless and the runner's oracle mode is itself sim-only guarded.
#
#  Wire layout (must match pp_oracle_pose.hpp, little-endian, packed, 72 bytes),
#  written with a simple seqlock (odd seq during write, even when stable):
#    uint64 seq | double stamp_wall_ns | double pos[3] | double quat_wxyz[4]
#
#  Usage (inside the ROS 2 env that can see the sim's /sim/a3 topics):
#    source <ros2 setup> ; source <sim msgs overlay if needed>
#    python3 scripts/oracle_pose_bridge.py \
#        --topic /sim/a3/pelvis_pose --shm /dev/shm/pp_oracle_pelvis
import argparse
import mmap
import os
import struct
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

STRUCT = struct.Struct("<Qd3d4d")  # seq, stamp_ns, pos[3], quat_wxyz[4]  -> 72 bytes
SIZE = STRUCT.size
assert SIZE == 72, SIZE


class OraclePoseBridge(Node):
    def __init__(self, topic: str, shm_path: str, best_effort: bool):
        super().__init__("pp_oracle_pose_bridge")
        # create/truncate the shm file to the fixed size
        self._fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(self._fd, SIZE)
        self._mm = mmap.mmap(self._fd, SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
        self._seq = 0
        self._n = 0
        qos = QoSProfile(depth=10)
        if best_effort:
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(PoseStamped, topic, self._on_pose, qos)
        self.get_logger().info(
            f"[SIM-ONLY oracle bridge] {topic} -> {shm_path} ({SIZE} B). "
            f"DO NOT RUN ON HARDWARE.")
        self.create_timer(2.0, self._heartbeat)

    def _heartbeat(self):
        self.get_logger().info(f"oracle bridge: {self._n} poses written")

    def _on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        stamp_ns = time.clock_gettime(time.CLOCK_REALTIME) * 1e9
        # seqlock: bump to odd (writing), write payload, bump to even (stable)
        self._seq += 1  # odd
        self._mm.seek(0)
        self._mm.write(STRUCT.pack(self._seq, stamp_ns,
                                   p.x, p.y, p.z, q.w, q.x, q.y, q.z))
        self._seq += 1  # even
        self._mm.seek(0)
        self._mm.write(struct.pack("<Q", self._seq))
        self._n += 1

    def destroy_node(self):
        try:
            self._mm.close()
            os.close(self._fd)
        finally:
            super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/sim/a3/pelvis_pose")
    ap.add_argument("--shm", default="/dev/shm/pp_oracle_pelvis")
    ap.add_argument("--best-effort", action="store_true",
                    help="use BEST_EFFORT QoS (match the sim publisher if needed)")
    args = ap.parse_args()
    rclpy.init()
    node = OraclePoseBridge(args.topic, args.shm, args.best_effort)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
