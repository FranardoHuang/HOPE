"""Agibot A3 hardware bridge: adapts WBC runner ↔ AGI body-drive topics.

The WBC runner (hope_wbc_runner) expects a single merged joint-state topic and
publishes a single merged joint-command. The AGI AimRT backend exposes four
separate body-drive topics for waist/neck/arm/leg.  This node bridges the gap:

  STATE (robot → WBC runner)
    /body_drive/waist_joint_state   (joint_msgs/JointState,  3 joints)
    /body_drive/arm_joint_state     (joint_msgs/JointState, 14 joints)
    /body_drive/leg_joint_state     (joint_msgs/JointState, 12 joints)
    /body_drive/neck_joint_state    (joint_msgs/JointState,  2 joints)
    /body_drive/pelvis_imu/data     (sensor_msgs/Imu)
            │
            ▼  (merged 31-DOF)
    /a3/joint_states  (joint_msgs/JointState, 31 joints)
    /a3/imu           (sensor_msgs/Imu)

  COMMAND (WBC runner → robot)
    /a3/joint_command               (joint_msgs/JointCommand, 31 joints)
            │
            ▼  (split by group)
    /body_drive/waist_joint_command (joint_msgs/JointCommand,  3 joints)
    /body_drive/arm_joint_command   (joint_msgs/JointCommand, 14 joints)
    /body_drive/leg_joint_command   (joint_msgs/JointCommand, 12 joints)
    /body_drive/neck_joint_command  (joint_msgs/JointCommand,  2 joints)

Joint assignment is by EXACT NAME matching (robust to ordering changes):
  waist: waist_yaw_joint, waist_roll_joint, waist_pitch_joint
  neck:  head_yaw_joint, head_pitch_joint
  arm:   *_shoulder_*, *_elbow_*, *_wrist_* (both arms)
  leg:   *_hip_*, *_knee_*, *_ankle_* (both legs)

NOTE: requires joint_msgs to be on the ROS2 overlay path.  In the hope
distrobox this comes from the AGI sim overlay; source it before launching.
"""

from __future__ import annotations

from collections import OrderedDict

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


# ---------------------------------------------------------------------------
# A3 31-DOF backend joint order (mirrors a3_pingpong::backend_joint_order()).
# Index = backend slot; value = joint name.
# ---------------------------------------------------------------------------
_BACKEND_JOINT_ORDER: list[str] = [
    # waist [0..2]
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # neck [3..4]
    "head_yaw_joint", "head_pitch_joint",
    # left arm [5..11]
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # right arm [12..18]
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # left leg [19..24]
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # right leg [25..30]
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

_WAIST_NAMES: frozenset[str] = frozenset(
    ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
_NECK_NAMES: frozenset[str] = frozenset(
    ["head_yaw_joint", "head_pitch_joint"])
_ARM_NAMES: frozenset[str] = frozenset(
    ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
     "left_elbow_joint",
     "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
     "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
     "right_elbow_joint",
     "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"])
_LEG_NAMES: frozenset[str] = frozenset(
    ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
     "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
     "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
     "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"])

_GROUP_SETS: dict[str, frozenset[str]] = {
    "waist": _WAIST_NAMES,
    "neck":  _NECK_NAMES,
    "arm":   _ARM_NAMES,
    "leg":   _LEG_NAMES,
}


def _joint_group(name: str) -> str | None:
    for g, names in _GROUP_SETS.items():
        if name in names:
            return g
    return None


class AgibotHardwareBridgeNode(Node):
    """Bridges WBC runner ↔ AGI A3 body-drive topics."""

    def __init__(self) -> None:
        super().__init__("agibot_hardware_bridge")

        # --- lazy joint_msgs import (only available when the AGI sim overlay is sourced) ---
        try:
            from joint_msgs.msg import JointCommand, JointState, State, Command
            self._JointState = JointState
            self._JointCommand = JointCommand
            self._State = State
            self._Command = Command
        except ImportError as e:
            self.get_logger().fatal(
                f"Cannot import joint_msgs: {e}.  "
                "Source the AGI sim overlay before launching: "
                "source .../setup_ros2_msgs.bash  (or source the AGI dist overlay).")
            raise

        self.declare_parameter("body_drive_ns", "/body_drive")
        self.declare_parameter("joint_states_topic", "/a3/joint_states")
        self.declare_parameter("joint_command_topic", "/a3/joint_command")
        self.declare_parameter("imu_topic", "/a3/imu")
        # Which body-drive IMU to forward (pelvis or torso).
        self.declare_parameter("imu_source", "pelvis")   # "pelvis" | "torso"
        # Publish merged state at this rate (Hz) using the latest per-group data.
        # Set 0 to publish only on incoming state (reactive, lower latency but
        # potentially mismatched timestamps).
        self.declare_parameter("merge_publish_rate_hz", 0.0)

        ns = str(self.get_parameter("body_drive_ns").value).rstrip("/")
        states_topic = str(self.get_parameter("joint_states_topic").value)
        cmd_topic = str(self.get_parameter("joint_command_topic").value)
        imu_topic = str(self.get_parameter("imu_topic").value)
        imu_source = str(self.get_parameter("imu_source").value)
        merge_hz = float(self.get_parameter("merge_publish_rate_hz").value)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Commands MUST be RELIABLE: the AGI AimRT ros2 backend subscribes the
        # /body_drive/*_joint_command topics with RELIABLE QoS (a3_aimrt_config.ros2.yaml),
        # and a BEST_EFFORT publisher never matches a RELIABLE subscriber — the robot
        # would silently receive nothing.
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Per-group latest JointState (state, set by each group's callback).
        # Keyed by group name; value = {name: (pos, vel)} dict.
        self._latest_state: dict[str, dict[str, tuple[float, float]]] = {
            "waist": {}, "neck": {}, "arm": {}, "leg": {},
        }
        self._state_received: dict[str, bool] = {
            "waist": False, "neck": False, "arm": False, "leg": False,
        }
        self._latest_stamp = None   # most recent header stamp across all groups

        # --- STATE subscribers: AGI backend -> bridge ---
        for group in ("waist", "arm", "leg", "neck"):
            topic = f"{ns}/{group}_joint_state"
            self.create_subscription(
                self._JointState, topic,
                lambda msg, g=group: self._on_joint_state(g, msg),
                sensor_qos,
            )
            self.get_logger().info(f"subscribing state: {topic}")

        # --- STATE publisher: bridge -> WBC runner ---
        self._states_pub = self.create_publisher(
            self._JointState, states_topic, sensor_qos)
        self.get_logger().info(f"publishing merged state: {states_topic}")

        # --- IMU subscriber + publisher ---
        imu_bd_topic = f"{ns}/pelvis_imu/data" if imu_source == "pelvis" \
                       else f"{ns}/torso_imu/data"
        self.create_subscription(Imu, imu_bd_topic, self._on_imu, sensor_qos)
        self._imu_pub = self.create_publisher(Imu, imu_topic, sensor_qos)
        self.get_logger().info(
            f"forwarding IMU: {imu_bd_topic} -> {imu_topic}")

        # --- COMMAND subscriber: WBC runner -> bridge ---
        self.create_subscription(
            self._JointCommand, cmd_topic, self._on_joint_command, sensor_qos)
        self.get_logger().info(f"subscribing command: {cmd_topic}")

        # --- COMMAND publishers: bridge -> AGI backend (RELIABLE, see command_qos) ---
        self._cmd_pubs: dict[str, object] = {}
        for group in ("waist", "arm", "leg", "neck"):
            topic = f"{ns}/{group}_joint_command"
            self._cmd_pubs[group] = self.create_publisher(
                self._JointCommand, topic, command_qos)
            self.get_logger().info(f"publishing command: {topic} (RELIABLE)")

        # Timer mode REPLACES reactive publishing (see _on_joint_state).
        self._timer_mode = merge_hz > 0
        if self._timer_mode:
            self.create_timer(1.0 / merge_hz, self._publish_merged_state)
            self.get_logger().info(f"timer-based merged state at {merge_hz:.0f} Hz")

        self.get_logger().info("agibot_hardware_bridge ready")

    # -----------------------------------------------------------------------
    # STATE: body_drive/* -> /a3/joint_states
    # -----------------------------------------------------------------------

    def _on_joint_state(self, group: str, msg) -> None:
        """Update the per-group joint cache and publish merged state."""
        self._state_received[group] = True
        if msg.header.stamp.sec > 0 or msg.header.stamp.nanosec > 0:
            self._latest_stamp = msg.header.stamp

        for s in msg.joints:
            self._latest_state[group][s.name] = (s.position, s.velocity)

        # Reactive publish (only when no timer is configured): emit on each update.
        if not self._timer_mode:
            self._publish_merged_state()

    def _publish_merged_state(self) -> None:
        # ALL four groups must have reported at least once: a partially-filled merge
        # would emit q=0 for entire missing limbs, and because every joint NAME is
        # present the runner's coverage check cannot detect it — the policy would see
        # (and react to) a robot with limbs at zero.
        if not all(self._state_received.values()):
            missing = [g for g, ok in self._state_received.items() if not ok]
            self.get_logger().warn(
                f"merged state withheld: no data yet from group(s) {missing}",
                throttle_duration_sec=5.0)
            return

        out = self._JointState()
        if self._latest_stamp is not None:
            out.header.stamp = self._latest_stamp
        else:
            out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = ""

        states: list = []
        for name in _BACKEND_JOINT_ORDER:
            group = _joint_group(name)
            if group is None:
                continue
            pos, vel = self._latest_state[group].get(name, (0.0, 0.0))
            s = self._State()
            s.name = name
            s.sequence = 0
            s.position = pos
            s.velocity = vel
            s.effort = 0.0
            states.append(s)

        out.joints = states
        self._states_pub.publish(out)

    # -----------------------------------------------------------------------
    # IMU: /body_drive/pelvis_imu/data -> /a3/imu
    # -----------------------------------------------------------------------

    def _on_imu(self, msg: Imu) -> None:
        self._imu_pub.publish(msg)

    # -----------------------------------------------------------------------
    # COMMAND: /a3/joint_command -> body_drive/*_joint_command
    # -----------------------------------------------------------------------

    def _on_joint_command(self, msg) -> None:
        """Split the combined 31-DOF command into 4 group-specific messages."""
        groups: dict[str, list] = {g: [] for g in _GROUP_SETS}
        for jc in msg.joints:
            g = _joint_group(jc.name)
            if g is None:
                self.get_logger().warn(
                    f"unknown joint in /a3/joint_command: '{jc.name}' — skipping",
                    throttle_duration_sec=5.0)
                continue
            groups[g].append(jc)

        for group, cmds in groups.items():
            if not cmds:
                continue
            out = self._JointCommand()
            out.header = msg.header
            out.joints = cmds
            self._cmd_pubs[group].publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AgibotHardwareBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
