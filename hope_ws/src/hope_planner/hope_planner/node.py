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

from hope_msgs.msg import RacketCommand

from .ball_kalman_estimator import BallKalmanEstimator
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
        self.declare_parameter("drag_k", 0.1261)          # venue fit 2026-07-03 (configs/ball_physics_venue.yaml)
        self.declare_parameter("restitution_h", 0.64)     # no-spin grip equivalent (1 - a_t)
        self.declare_parameter("restitution_v", 0.9215)   # venue table e_n
        self.declare_parameter("restitution_racket", 0.654)  # paddle e const; e(u_n) exp form applied in racket_target_planner
        self.declare_parameter("use_kalman", False)          # shadow-run the EKF next to the polyfit estimator

        self._ball_index = int(self.get_parameter("ball_pose_index").value)

        config = PlannerConfig(
            x_hit=self.get_parameter("x_hit").value,
            target_land=np.array([
                self.get_parameter("target_land_x").value,
                self.get_parameter("target_land_y").value,
                0.0,
            ]),
            delta_t_flight=self.get_parameter("delta_t_flight").value,
            C_r=self.get_parameter("restitution_racket").value,
            use_kalman=bool(self.get_parameter("use_kalman").value),
        )
        physics = BallPhysics(
            k=self.get_parameter("drag_k").value,
            C_h=self.get_parameter("restitution_h").value,
            C_v=self.get_parameter("restitution_v").value,
        )

        self.planner = HOPEPlanner(physics=physics, config=config)

        # Flag-gated EKF SHADOW estimator: fed the same measurements as the
        # legacy polyfit estimator; the planner still ACTS on the legacy path.
        # Only position/velocity deltas are published (diagnostics) so the
        # EKF can be validated on live venue data before promotion.
        self._kf = BallKalmanEstimator(config, physics) if config.use_kalman else None
        self._kf_pos_delta = float("nan")
        self._kf_vel_delta = float("nan")

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
        self.cmd_pub = self.create_publisher(RacketCommand, "/racket/command", command_qos)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/planner/diagnostics", 1)

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

        cmd = self.planner.update(t, p_ball)

        if self._kf is not None:
            self._kf.push(t, p_ball)
            if self._kf.ready and self.planner.estimator.ready:
                p_kf, v_kf, _ = self._kf.estimate()
                p_leg, v_leg, _ = self.planner.estimator.estimate()
                self._kf_pos_delta = float(np.linalg.norm(p_kf - p_leg))
                self._kf_vel_delta = float(np.linalg.norm(v_kf - v_leg))
            else:
                self._kf_pos_delta = float("nan")
                self._kf_vel_delta = float("nan")

        if cmd is None:
            self._last_valid = False
            self._last_tts = float("nan")
            return

        self._last_valid = cmd.valid
        tts = self.planner.time_to_strike
        self._last_tts = tts if tts is not None else float("nan")
        if cmd.valid:
            self._n_valid += 1

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
        if self._kf is not None:
            status.values += [
                KeyValue(key="kf_pos_delta_m", value=f"{self._kf_pos_delta:.4f}"),
                KeyValue(key="kf_vel_delta_mps", value=f"{self._kf_vel_delta:.4f}"),
                KeyValue(key="kf_rejected", value=str(self._kf.rejected_count)),
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
