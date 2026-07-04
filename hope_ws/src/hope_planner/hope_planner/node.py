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

from .ball_kalman_estimator import BallKalmanEstimator
from .constants import BallPhysics, PlannerConfig
from .planner import HOPEPlanner
from .strike_spec_planner import StrikeSpecPlanner


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
        self.declare_parameter("publish_strike_spec", False)  # diagnostics-only strike-spec inverse solve
        self.declare_parameter("racket_speed_budget", 10.0)   # m/s cap for the spec solve — diagnostic
                                                              # sanity bound, above venue strike speeds
                                                              # (paddle u_n fit envelope tops out 7.2 m/s)
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
        # --- FLAT outputs for the AGI native C++ runner (--planner, the ONLY control path) ---
        # The C++ a3_deploy_onnx_ref_pingpong subscribes std_msgs/Float64MultiArray (it avoids
        # vendoring hope_msgs typesupport on aarch64). We MIRROR /racket/command as a flat array
        # and stream the robot base pose (from robot_pose_topic) as a second flat array so the
        # runner's external_base localization has a live base.
        self.declare_parameter("publish_flat_cmd", True)
        self.declare_parameter("racket_flat_topic", "/racket/command_flat")
        self.declare_parameter("base_flat_topic", "/a3/base_pose_flat")
        # marker-cluster -> base_link offset (table frame). /P1/pose is the marker cluster; the
        # policy base is the pelvis. In sim (robot_pose_topic=/sim/a3/pelvis_pose) it is already
        # the pelvis, so [0,0,0]. Set per venue (mirrors hope_world_frame.yaml mocap_to_base_link / G8).
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

        # Flag-gated strike-spec DIAGNOSTICS: inverse-solve the racket control
        # variables (face tilt, v_n, v_t) + their landing sensitivities next to
        # the existing racket command. Does NOT touch the command path. The LM
        # solve costs ~0.3 s, so it is throttled to at most 1 Hz rather than
        # running per 300 Hz mocap frame.
        self._publish_strike_spec = bool(self.get_parameter("publish_strike_spec").value)
        self._racket_speed_budget = float(self.get_parameter("racket_speed_budget").value)
        self._spec_planner = (
            StrikeSpecPlanner(physics=physics, config=config)
            if self._publish_strike_spec else None
        )
        self._last_spec = None
        self._spec_next_t = float("-inf")

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

        # Strike-spec diagnostics AFTER the command publish so the solve
        # latency never delays the command itself.
        if self._spec_planner is not None and cmd.valid and t >= self._spec_next_t:
            self._spec_next_t = t + 1.0
            strike = self.planner.strike_target
            if strike is not None and strike.valid:
                # Legacy command path is spin-blind -> omega None (zeros);
                # promote to the EKF/spin estimate when that path lands.
                self._last_spec = self._spec_planner.solve(
                    strike.p_ball, strike.v_ball, None,
                    self.planner.config.target_land[:2],
                    self._racket_speed_budget,
                )
                if self._last_spec is not None:
                    s = self._last_spec
                    self.get_logger().info(
                        "strike spec: tilt=(%.2f, %.2f) deg  v_n=%.2f  |v_t|=%.2f m/s  "
                        "land=(%.3f, %.3f)  sens: %.3f m/deg pitch, %.3f m/deg yaw, "
                        "%.3f m/(m/s) v_n, %.3f m/(m/s) v_t"
                        % (
                            s.tilt_pitch_deg, s.tilt_yaw_deg, s.v_n_signed,
                            float(np.linalg.norm(s.v_t_vec)),
                            s.landing_xy[0], s.landing_xy[1],
                            float(np.linalg.norm(s.d_landing_d_pitch)),
                            float(np.linalg.norm(s.d_landing_d_yaw)),
                            float(np.linalg.norm(s.d_landing_d_v_n)),
                            float(np.linalg.norm(s.d_landing_d_v_t)),
                        )
                    )

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
        if self._publish_strike_spec:
            s = self._last_spec
            if s is None:
                status.values.append(KeyValue(key="spec_valid", value="False"))
            else:
                status.values += [
                    KeyValue(key="spec_valid", value="True"),
                    KeyValue(key="spec_tilt_pitch_deg", value=f"{s.tilt_pitch_deg:.3f}"),
                    KeyValue(key="spec_tilt_yaw_deg", value=f"{s.tilt_yaw_deg:.3f}"),
                    KeyValue(key="spec_v_n_mps", value=f"{s.v_n_signed:.3f}"),
                    KeyValue(key="spec_v_t_mps", value=f"{np.linalg.norm(s.v_t_vec):.3f}"),
                    KeyValue(key="spec_landing_x_m", value=f"{s.landing_xy[0]:.3f}"),
                    KeyValue(key="spec_landing_y_m", value=f"{s.landing_xy[1]:.3f}"),
                    # Landing-sensitivity norms = the control-precision budget:
                    # how much landing error one unit of control error buys.
                    KeyValue(key="spec_dland_dpitch_m_per_deg",
                             value=f"{np.linalg.norm(s.d_landing_d_pitch):.4f}"),
                    KeyValue(key="spec_dland_dyaw_m_per_deg",
                             value=f"{np.linalg.norm(s.d_landing_d_yaw):.4f}"),
                    KeyValue(key="spec_dland_dvn_m_per_mps",
                             value=f"{np.linalg.norm(s.d_landing_d_v_n):.4f}"),
                    KeyValue(key="spec_dland_dvt_m_per_mps",
                             value=f"{np.linalg.norm(s.d_landing_d_v_t):.4f}"),
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
