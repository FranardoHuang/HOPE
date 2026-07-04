"""ROS 2 node wrapping the HOPE 7-DOF racket planner.

Subscribes to ball position from the mocap `/poses` stream (the avatar_pro relay
on the Avatar Pro / Chingmu VRPN path) and publishes the desired racket state on
`/racket/command` as the shared
hope_msgs/RacketCommand. Diagnostics are published at 10 Hz.

Per HOPE rules the racket pose is never measured by motion capture; the
humanoid must achieve the commanded racket state via its own forward
kinematics. See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md.
"""

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

# hope_msgs is OPTIONAL (FLAT-ONLY mode): on the robot MDU only the std_msgs flat topics
# are consumed (by the C++ --planner runner), and building hope_msgs typesupport for
# aarch64 is exactly the dependency the flat wire exists to avoid. Without hope_msgs the
# node still runs and publishes /racket/command_flat + /a3/base_pose_flat; only the
# Path-B /racket/command publisher is skipped.
try:
    from hope_msgs.msg import RacketCommand
except ImportError:  # flat-only environment (e.g. the MDU)
    RacketCommand = None

from .constants import BallPhysics, PlannerConfig
from .planner import HOPEPlanner


class HOPEPlannerNode(Node):
    """ROS 2 node for the HOPE model-based planner."""

    def __init__(self):
        super().__init__("hope_planner")

        self.declare_parameter("ball_rigid_body_name", "pingpong_ball")
        self.declare_parameter("ball_pose_index", 0)
        self.declare_parameter("x_hit", 0.0)
        self.declare_parameter("target_land_x", 2.055)
        self.declare_parameter("target_land_y", -0.7625)
        self.declare_parameter("delta_t_flight", 0.5)
        self.declare_parameter("drag_k", 0.5)
        self.declare_parameter("restitution_h", 0.75)
        self.declare_parameter("restitution_v", 0.85)
        self.declare_parameter("restitution_racket", 0.88)
        # --- ADAPTIVE hit plane (2026-07-04): x_hit follows the LIVE robot position ---
        # The trained policy WALKS to the strike (walk-and-strike lunge, ~0.5-0.8 m): after
        # one return the robot stands AT the old static plane, so subsequent plans land at
        # base-rel x ~ 0 and the runner's reachability gate rejects them (one swing per
        # session). With a robot pose feed the plane tracks the robot:
        #   x_hit = clamp(robot_x + x_hit_offset, x_hit_min, x_hit_max)
        # x_hit_offset = the policy's comfortable strike reach (base-rel box center ~0.67).
        # x_hit_max PROTECTS THE TABLE: the lunge marches the robot forward each swing; the
        # clamp stops the plane (and therefore the robot) at the table edge minus reach.
        # Empty robot_pose_topic -> static x_hit (legacy behavior).
        # Arena: robot_pose_topic:=/P1/pose (mocap relay, TABLE frame, position-only).
        # AGI sim: robot_pose_topic:=/sim/a3/pelvis_pose (sim world == table frame).
        self.declare_parameter("robot_pose_topic", "")
        self.declare_parameter("x_hit_offset", 0.67)
        self.declare_parameter("x_hit_min", -0.30)
        self.declare_parameter("x_hit_max", 0.30)   # table edge x=0 + racket reach margin
        # --- FLAT outputs for the AGI native C++ runner (Path A binary, --planner) ---
        # The C++ a3_deploy_onnx_ref_pingpong subscribes std_msgs/Float64MultiArray (it avoids
        # vendoring hope_msgs typesupport on aarch64). We MIRROR /racket/command as a flat array
        # and stream the robot base pose (from robot_pose_topic) as a second flat array so the
        # runner's external_base localization has a live base (the ONLY control path since 2026-07-04).
        self.declare_parameter("publish_flat_cmd", True)
        self.declare_parameter("racket_flat_topic", "/racket/command_flat")
        self.declare_parameter("base_flat_topic", "/a3/base_pose_flat")
        # marker-cluster -> base_link offset (table frame). /P1/pose is the marker cluster; the
        # policy base is the pelvis. In sim (robot_pose_topic=/sim/a3/pelvis_pose) it is already
        # the pelvis, so [0,0,0]. Set per venue (mirrors wbc_runner marker_to_base_xyz / G8).
        self.declare_parameter("marker_to_base_xyz", [0.0, 0.0, 0.0])

        self._ball_index = int(self.get_parameter("ball_pose_index").value)
        self._x_hit_offset = float(self.get_parameter("x_hit_offset").value)
        self._x_hit_min = float(self.get_parameter("x_hit_min").value)
        self._x_hit_max = float(self.get_parameter("x_hit_max").value)
        self._robot_x = None          # latest robot X (table frame); None -> static x_hit
        self._publish_flat = bool(self.get_parameter("publish_flat_cmd").value)
        self._marker_to_base = np.array(
            [float(v) for v in self.get_parameter("marker_to_base_xyz").value])

        config = PlannerConfig(
            x_hit=self.get_parameter("x_hit").value,
            target_land=np.array([
                self.get_parameter("target_land_x").value,
                self.get_parameter("target_land_y").value,
                0.0,
            ]),
            delta_t_flight=self.get_parameter("delta_t_flight").value,
            C_r=self.get_parameter("restitution_racket").value,
        )
        physics = BallPhysics(
            k=self.get_parameter("drag_k").value,
            C_h=self.get_parameter("restitution_h").value,
            C_v=self.get_parameter("restitution_v").value,
        )

        self.planner = HOPEPlanner(physics=physics, config=config)

        # Best-effort, depth-1 QoS for high-rate mocap topics (REP-2003 sensor style).
        mocap_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PoseArray, "/poses", self._poses_cb, mocap_qos)
        if RacketCommand is not None:
            self.cmd_pub = self.create_publisher(RacketCommand, "/racket/command", command_qos)
        else:
            self.cmd_pub = None
            self.get_logger().warn(
                "hope_msgs not available -> FLAT-ONLY mode: /racket/command is NOT "
                "published (fine for the C++ --planner runner; Path B needs hope_msgs).")
        self.diag_pub = self.create_publisher(DiagnosticArray, "/planner/diagnostics", 1)

        # Flat outputs for the AGI native C++ runner (--planner). Same RELIABLE QoS as
        # /racket/command so the AimRT ros2 subscriber (declared RELIABLE) matches.
        self.flat_cmd_pub = None
        self.flat_base_pub = None
        if self._publish_flat:
            self.flat_cmd_pub = self.create_publisher(
                Float64MultiArray, str(self.get_parameter("racket_flat_topic").value), command_qos)
            self.flat_base_pub = self.create_publisher(
                Float64MultiArray, str(self.get_parameter("base_flat_topic").value), command_qos)

        robot_pose_topic = str(self.get_parameter("robot_pose_topic").value)
        if robot_pose_topic:
            from geometry_msgs.msg import PoseStamped

            def _robot_pose_cb(msg: PoseStamped) -> None:
                p = msg.pose.position
                self._robot_x = float(p.x)
                # Stream the base pose to the C++ runner (external_base localization). Apply the
                # marker->base offset; the runner uses POSITION only (mocap is position-only) so
                # the quaternion is informational. Publishes at the robot_pose_topic rate.
                if self.flat_base_pub is not None:
                    bx = float(p.x) + float(self._marker_to_base[0])
                    by = float(p.y) + float(self._marker_to_base[1])
                    bz = float(p.z) + float(self._marker_to_base[2])
                    q = msg.pose.orientation
                    m = Float64MultiArray()
                    # [schema, valid, x, y, z, qw, qx, qy, qz]
                    m.data = [1.0, 1.0, bx, by, bz,
                              float(q.w), float(q.x), float(q.y), float(q.z)]
                    self.flat_base_pub.publish(m)

            self.create_subscription(PoseStamped, robot_pose_topic, _robot_pose_cb, mocap_qos)
            self.get_logger().info(
                f"adaptive x_hit ON: robot pose from '{robot_pose_topic}', "
                f"x_hit = clamp(robot_x + {self._x_hit_offset:.2f}, "
                f"[{self._x_hit_min:.2f}, {self._x_hit_max:.2f}])")

        # Diagnostics at 10 Hz.
        self._n_received = 0
        self._n_valid = 0
        self._last_valid = False
        self._last_tts = float("nan")
        self.create_timer(0.1, self._publish_diagnostics)

        self.get_logger().info(
            f"HOPE planner started - x_hit={config.x_hit:.2f}, "
            f"target={config.target_land}, ball_pose_index={self._ball_index}"
        )

    def _poses_cb(self, msg: PoseArray) -> None:
        self._n_received += 1
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if len(msg.poses) <= self._ball_index:
            return

        # NOTE: PoseArray carries no names. Configure ball_pose_index to match
        # the ball's slot in the /poses ordering (the avatar_pro relay puts the
        # ball first), or swap this for a /tf lookup keyed on ball_rigid_body_name.
        pose = msg.poses[self._ball_index]
        p_ball = np.array([pose.position.x, pose.position.y, pose.position.z])

        # adaptive hit plane: track the live robot (see robot_pose_topic above). Mutating
        # config.x_hit is safe — the predictor reads it per predict() call.
        if self._robot_x is not None:
            self.planner.config.x_hit = float(
                np.clip(self._robot_x + self._x_hit_offset, self._x_hit_min, self._x_hit_max))

        cmd = self.planner.update(t, p_ball)
        if cmd is None:
            self._last_valid = False
            self._last_tts = float("nan")
            return

        self._last_valid = cmd.valid
        tts = self.planner.time_to_strike
        self._last_tts = tts if tts is not None else float("nan")
        if cmd.valid:
            self._n_valid += 1

        if self.cmd_pub is not None:
            out = RacketCommand()
            out.header = msg.header
            out.header.frame_id = "world"
            out.position.x = float(cmd.p_intercept[0])
            out.position.y = float(cmd.p_intercept[1])
            out.position.z = float(cmd.p_intercept[2])
            out.velocity.x = float(cmd.v_racket[0])
            out.velocity.y = float(cmd.v_racket[1])
            out.velocity.z = float(cmd.v_racket[2])
            # RacketCommand.normal carries the unit face normal (Vector3), not an
            # orientation. IK-based controllers that need a full quaternion can call
            # quaternion_utils.normal_to_quaternion(cmd.n_racket, constrain_up=True).
            out.normal.x = float(cmd.n_racket[0])
            out.normal.y = float(cmd.n_racket[1])
            out.normal.z = float(cmd.n_racket[2])
            out.strike_time = float(cmd.t_strike)
            out.time_to_strike = float(self._last_tts)
            out.ball_velocity_outgoing.x = float(cmd.v_ball_outgoing[0])
            out.ball_velocity_outgoing.y = float(cmd.v_ball_outgoing[1])
            out.ball_velocity_outgoing.z = float(cmd.v_ball_outgoing[2])
            out.valid = bool(cmd.valid)
            out.clears_net = bool(cmd.clears_net)
            out.bypasses_net_posts = bool(cmd.bypasses_net_posts)
            out.predicted_bounces = int(cmd.num_bounces)
            self.cmd_pub.publish(out)

        # Mirror to the flat topic for the AGI C++ runner (--planner). swing_sign is left 0
        # (the runner derives the side from the base-relative target y); frame_code 0 = world.
        if self.flat_cmd_pub is not None:
            fm = Float64MultiArray()
            # [schema, valid, swing_sign, px, py, pz, vx, vy, vz, tts, strike_time, frame_code]
            fm.data = [
                1.0, 1.0 if cmd.valid else 0.0, 0.0,
                float(cmd.p_intercept[0]), float(cmd.p_intercept[1]), float(cmd.p_intercept[2]),
                float(cmd.v_racket[0]), float(cmd.v_racket[1]), float(cmd.v_racket[2]),
                float(self._last_tts) if self._last_tts == self._last_tts else 0.0,
                float(cmd.t_strike), 0.0,
            ]
            self.flat_cmd_pub.publish(fm)

    def _publish_diagnostics(self) -> None:
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "hope_planner"
        status.hardware_id = "hope_planner"
        if self._n_received == 0:
            status.level = DiagnosticStatus.WARN
            status.message = "no /poses received yet"
        elif self._last_valid:
            status.level = DiagnosticStatus.OK
            status.message = "valid racket command"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "running; no valid strike"
        status.values = [
            KeyValue(key="poses_received", value=str(self._n_received)),
            KeyValue(key="valid_commands", value=str(self._n_valid)),
            KeyValue(key="last_valid", value=str(self._last_valid)),
            KeyValue(key="time_to_strike_s", value=f"{self._last_tts:.4f}"),
        ]
        arr.status = [status]
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = HOPEPlannerNode()
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
