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
        # PER-SIDE aim/flight (2026-07-08, from the Gate-3 rally vel-gate finding): the two
        # trained clips return in OPPOSITE cross-court directions (fh vy [+0.96,+1.96], bh vy
        # [-1.21,-0.21] world), so NO single target_land_y makes both sides' demanded racket
        # velocity land inside the trained boxes (best single aim = 5/10 sweep serves in-band;
        # the C++ runner's vel gate correctly stands on the rest). When set (non-NaN) the aim
        # and flight time switch per predicted arrival SIDE — fh if arrival_y - robot_y < -0.11
        # (the trained band midpoint, mirroring the runner's nearest-station split). The side
        # uses the LAST valid plan's intercept (one 300 Hz frame of lag; settles ~1.5 s before
        # any engage). NaN (default) = legacy single aim; arena values live in the yaml.
        self.declare_parameter("target_land_y_fh", float("nan"))
        self.declare_parameter("target_land_y_bh", float("nan"))
        self.declare_parameter("delta_t_flight_fh", float("nan"))
        self.declare_parameter("delta_t_flight_bh", float("nan"))
        self.declare_parameter("drag_k", 0.1261)          # venue fit 2026-07-03 (configs/ball_physics_venue.yaml)
        self.declare_parameter("restitution_h", 0.64)     # no-spin grip equivalent (1 - a_t)
        self.declare_parameter("restitution_v", 0.9215)   # venue table e_n
        self.declare_parameter("restitution_racket", 0.654)  # paddle e const; e(u_n) exp form applied in racket_target_planner
        self.declare_parameter("use_kalman", False)          # shadow-run the EKF next to the polyfit estimator
        self.declare_parameter("publish_strike_spec", True)   # strike-spec inverse solve (fast path made it
                                                       # a per-tick production channel, franco 2026-07-07;
                                                       # False restores the diagnostics-off legacy)
        self.declare_parameter("racket_speed_budget", 10.0)   # m/s cap for the spec solve — diagnostic
                                                              # sanity bound, above venue strike speeds
                                                              # (paddle u_n fit envelope tops out 7.2 m/s)
        # --- FAST strike spec (2026-07-06, flag-gated, DEFAULT OFF) ---
        # use_fast_strike_spec=True upgrades the strike-spec path from the 1 Hz
        # throttled scalar LM solve (~0.45 s each — why it was diagnostics-only) to the
        # PRODUCTION fast solver (strike_spec_fast.FastStrikeSpecPlanner): numpy-batched
        # Jacobian probes + adaptive 远粗近细 integration (spec_dt_integrate_coarse)
        # + warm start from the previous solve + LM budget strike_spec_max_iter,
        # sensitivities OFF the hot path (strike_spec_sensitivities=True re-enables
        # them per solve, on demand). Replanning is DECIMATED to strike_spec_rate_hz
        # (30-50 Hz sensible; /poses runs up to 300 Hz) and the cached spec is served
        # between solves. Measured on N=200 venue scenarios (Mac CPU, benchmark
        # --variants prod): med ~15 ms / p90 ~42 ms per solve at the SAME ~19 mm
        # oracle landing error as the 0.45 s baseline.
        # The published command topics (/racket/command, /racket/command_flat) are
        # UNTOUCHED by every flag here — same topic, same schema, same values.
        # Deploy flip: publish_strike_spec:=true use_fast_strike_spec:=true
        #              (optionally dt_integrate_coarse:=0.02 for the Stage-2 speedup).
        self.declare_parameter("use_fast_strike_spec", True)  # DEFAULT ON since 2026-07-07 (franco:
                                                      # 双档粒度作废,每 tick 全精度重算;flip to False
                                                      # only to reproduce the legacy 1 Hz throttled path)
        self.declare_parameter("strike_spec_rate_hz", 50.0)   # replan decimation = policy tick rate
                                                       # (每 tick 一解,franco 2026-07-07)
        self.declare_parameter("strike_spec_max_iter", 6)     # LM budget from a warm start
        self.declare_parameter("spec_dt_integrate_coarse", 0.02)  # adaptive cruise for the SPEC
                                                              # solve only (own config copy)
        self.declare_parameter("strike_spec_sensitivities", False)  # sensitivities on demand
        # Stage-2 adaptive integrator (EXISTING PlannerConfig.dt_integrate_coarse flag,
        # plumbed): 0.0 = OFF = legacy fixed-dt 1 kHz Euler, byte-identical. 0.02 cuts
        # Stage-2 predict ~3.9 ms -> ~1.0 ms per /poses tick (same benchmark). Only the
        # predictor honors the flag (predict / integrate_to_table_plane / net-clear);
        # RacketTargetPlanner's own integrations keep the legacy fixed dt either way.
        self.declare_parameter("dt_integrate_coarse", 0.02)  # DEFAULT ON since 2026-07-07: adaptive
                                                     # cruise (event sub-steps keep legacy dt);
                                                     # 0.0 restores byte-identical fixed-dt Euler
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
        # HITTER-PURE decoupling (2026-07-07): the paper's virtual hit plane is FIXED (§IV-B,
        # x = -1.37 m table frame) and the ROBOT walks to a commanded station behind it (Fig. 4);
        # the adaptive plane above inverts that causality (plane chases the robot -> the
        # documented "+x march"). For the 110-D hitter_pure deploy profile set this FALSE:
        # robot_pose_topic keeps feeding /a3/base_pose_flat (the runner needs the live base),
        # but x_hit stays the static parameter and the runner derives the station from the
        # target (station_x = x_hit - 0.70, the trained plane offset). Default TRUE = legacy
        # behavior for the 175/177 lineages.
        self.declare_parameter("x_hit_follow_robot", True)
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
        # Z offset added to ALL PUBLISHED OUTPUTS (both flats + /racket/command) converting the
        # planner's working frame into the POLICY world frame (z=0 at the FLOOR — the training
        # frame the C++ runner's gates expect: base_low 0.7, target z in [0.55,1.40]).
        # Planner INTERNALS (bounce plane z=0, net check, target_land) stay in the MOCAP frame;
        # at the arena that is the G5 calibration with z=0 at the TABLE SURFACE, so set 0.76
        # (= TableParams.height). Sim feeds are already floor-origin -> keep 0.0.
        # (Field 2026-07-07: without this, arena base z ~0.15 < base_low 0.7 -> engage never fires.)
        self.declare_parameter("policy_z_offset", 0.0)
        # Table +Y edge in the working frame (TableParams.y_max). Arena default 0.0
        # (origin at near-left corner, table at y<=0); the SIM harness centers the
        # table on the robot -> hope_planner.sim.yaml sets 0.7825.
        self.declare_parameter("table_y_max", 0.0)

        self._ball_index = int(self.get_parameter("ball_pose_index").value)
        self._x_hit_offset = float(self.get_parameter("x_hit_offset").value)
        self._x_hit_min = float(self.get_parameter("x_hit_min").value)
        self._x_hit_max = float(self.get_parameter("x_hit_max").value)
        self._x_hit_follow_robot = bool(self.get_parameter("x_hit_follow_robot").value)
        self._robot_x = None          # latest robot X (table frame); None -> static x_hit
        self._robot_y = None          # latest robot Y (table frame); side split for per-side aim
        # per-side aim/flight (NaN = disabled); consumed in _poses_cb before the solve
        self._land_y_fh = float(self.get_parameter("target_land_y_fh").value)
        self._land_y_bh = float(self.get_parameter("target_land_y_bh").value)
        self._dtf_fh = float(self.get_parameter("delta_t_flight_fh").value)
        self._dtf_bh = float(self.get_parameter("delta_t_flight_bh").value)
        self._per_side_aim = not (np.isnan(self._land_y_fh) and np.isnan(self._land_y_bh)
                                  and np.isnan(self._dtf_fh) and np.isnan(self._dtf_bh))
        self._last_intercept_y = None  # last valid plan's arrival y (side memory)
        self._publish_flat = bool(self.get_parameter("publish_flat_cmd").value)
        self._marker_to_base = np.array(
            [float(v) for v in self.get_parameter("marker_to_base_xyz").value])
        self._policy_z_offset = float(self.get_parameter("policy_z_offset").value)

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
            dt_integrate_coarse=float(self.get_parameter("dt_integrate_coarse").value),
        )
        physics = BallPhysics(
            k=self.get_parameter("drag_k").value,
            C_h=self.get_parameter("restitution_h").value,
            C_v=self.get_parameter("restitution_v").value,
        )

        from .constants import TableParams
        table = TableParams(y_max=float(self.get_parameter("table_y_max").value))
        self.planner = HOPEPlanner(physics=physics, config=config, table=table)

        # Flag-gated EKF SHADOW estimator: fed the same measurements as the
        # legacy polyfit estimator; the planner still ACTS on the legacy path.
        # Only position/velocity deltas are published (diagnostics) so the
        # EKF can be validated on live venue data before promotion.
        self._kf = BallKalmanEstimator(config, physics) if config.use_kalman else None
        self._kf_pos_delta = float("nan")
        self._kf_vel_delta = float("nan")

        # Flag-gated strike-spec path: inverse-solve the racket control
        # variables (face tilt, v_n, v_t) + their landing sensitivities next to
        # the existing racket command. Does NOT touch the command path.
        #   legacy (use_fast_strike_spec=False): scalar LM solve ~0.45 s ->
        #     throttled to at most 1 Hz. Byte-identical to the pre-flag node.
        #   fast   (use_fast_strike_spec=True): FastStrikeSpecPlanner.solve_fast_spec
        #     (~15 ms med) -> replans at strike_spec_rate_hz, warm-started, cached
        #     spec served between solves; log still throttled to 1 Hz.
        self._publish_strike_spec = bool(self.get_parameter("publish_strike_spec").value)
        self._racket_speed_budget = float(self.get_parameter("racket_speed_budget").value)
        self._use_fast_spec = bool(self.get_parameter("use_fast_strike_spec").value)
        self._spec_period = 1.0 / max(float(self.get_parameter("strike_spec_rate_hz").value), 1e-3)
        self._spec_max_iter = int(self.get_parameter("strike_spec_max_iter").value)
        self._spec_sens = bool(self.get_parameter("strike_spec_sensitivities").value)
        self._spec_planner = None
        if self._publish_strike_spec:
            if self._use_fast_spec:
                import dataclasses

                from .strike_spec_fast import FastStrikeSpecPlanner

                # Own config COPY so the spec solver's adaptive cruise cannot
                # leak into the legacy Stage-2 path (which keeps the shared
                # config and the separate dt_integrate_coarse parameter).
                spec_config = dataclasses.replace(
                    config,
                    dt_integrate_coarse=float(
                        self.get_parameter("spec_dt_integrate_coarse").value),
                )
                self._spec_planner = FastStrikeSpecPlanner(physics=physics, config=spec_config)
            else:
                self._spec_planner = StrikeSpecPlanner(physics=physics, config=config)
        self._last_spec = None
        self._spec_next_t = float("-inf")
        self._spec_log_next_t = float("-inf")
        self._spec_warm_q = None      # previous fast solve's q, warm start

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
                self._robot_y = float(p.y)
                # Stream the base pose to the C++ runner (external_base localization). Apply the
                # marker->base offset; the runner uses POSITION only (mocap is position-only) so
                # the quaternion is informational. Publishes at the robot_pose_topic rate.
                if self.flat_base_pub is not None:
                    bx = float(p.x) + float(self._marker_to_base[0])
                    by = float(p.y) + float(self._marker_to_base[1])
                    bz = float(p.z) + float(self._marker_to_base[2]) + self._policy_z_offset
                    q = msg.pose.orientation
                    m = Float64MultiArray()
                    # [schema, valid, x, y, z, qw, qx, qy, qz]
                    m.data = [1.0, 1.0, bx, by, bz,
                              float(q.w), float(q.x), float(q.y), float(q.z)]
                    self.flat_base_pub.publish(m)

            self.create_subscription(PoseStamped, robot_pose_topic, _robot_pose_cb, mocap_qos)
            if self._x_hit_follow_robot:
                self.get_logger().info(
                    f"adaptive x_hit ON: robot pose from '{robot_pose_topic}', "
                    f"x_hit = clamp(robot_x + {self._x_hit_offset:.2f}, "
                    f"[{self._x_hit_min:.2f}, {self._x_hit_max:.2f}])")
            else:
                self.get_logger().info(
                    f"x_hit FIXED at {config.x_hit:.2f} (hitter_pure profile, paper §IV-B); "
                    f"robot pose from '{robot_pose_topic}' feeds the base flat only")

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
        if self._spec_planner is not None and self._use_fast_spec:
            self.get_logger().info(
                f"FAST strike spec ON: replan {1.0 / self._spec_period:.0f} Hz, "
                f"iter budget {self._spec_max_iter}, spec dt_coarse="
                f"{self._spec_planner.config.dt_integrate_coarse:.3f} s, "
                f"sensitivities {'per-solve' if self._spec_sens else 'off (on demand)'}"
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
        # config.x_hit is safe — the predictor reads it per predict() call. Disabled by
        # x_hit_follow_robot=false (hitter_pure profile: FIXED plane, paper §IV-B).
        if self._robot_x is not None and self._x_hit_follow_robot:
            self.planner.config.x_hit = float(
                np.clip(self._robot_x + self._x_hit_offset, self._x_hit_min, self._x_hit_max))

        # PER-SIDE aim/flight (see the parameter block): switch target_land_y/delta_t_flight
        # by the last valid plan's arrival side. Mutating config is safe (read per solve).
        if self._per_side_aim and self._robot_y is not None and self._last_intercept_y is not None:
            fh = (self._last_intercept_y - self._robot_y) < -0.11
            land_y = self._land_y_fh if fh else self._land_y_bh
            dtf = self._dtf_fh if fh else self._dtf_bh
            if not np.isnan(land_y):
                self.planner.config.target_land[1] = land_y
            if not np.isnan(dtf):
                self.planner.config.delta_t_flight = dtf

        # CRASH GUARD (field 2026-07-07): garbage measurements (e.g. a mocap feed in
        # millimetres) made the outgoing-velocity solve raise FloatingPointError and
        # KILLED the node mid-demo. A planner glitch must degrade to "no command"
        # (the runner's safe stand), never to a dead planner.
        try:
            cmd = self.planner.update(t, p_ball)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().warning(
                f"planner solve failed ({type(exc).__name__}: {exc}) - treating as no-solution; "
                "if persistent, check the mocap feed (units/units-of-metres, frame, outliers)",
                throttle_duration_sec=2.0)
            cmd = None

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
            self._last_intercept_y = float(cmd.p_intercept[1])  # per-side aim memory

        if self.cmd_pub is not None:
            out = RacketCommand()
            out.header = msg.header
            out.header.frame_id = "world"
            out.position.x = float(cmd.p_intercept[0])
            out.position.y = float(cmd.p_intercept[1])
            out.position.z = float(cmd.p_intercept[2]) + self._policy_z_offset
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
                float(cmd.p_intercept[0]), float(cmd.p_intercept[1]),
                float(cmd.p_intercept[2]) + self._policy_z_offset,
                float(cmd.v_racket[0]), float(cmd.v_racket[1]), float(cmd.v_racket[2]),
                float(self._last_tts) if self._last_tts == self._last_tts else 0.0,
                float(cmd.t_strike), 0.0,
            ]
            self.flat_cmd_pub.publish(fm)

        # Strike-spec solve AFTER the command publish so the solve latency
        # never delays the command itself. Legacy: 1 Hz throttle (scalar LM,
        # ~0.45 s). Fast (use_fast_strike_spec): replan every _spec_period
        # (default 40 Hz), warm-started from the previous solve, cached spec
        # served between solves, log throttled to 1 Hz.
        if self._spec_planner is not None and cmd.valid and t >= self._spec_next_t:
            self._spec_next_t = t + (self._spec_period if self._use_fast_spec else 1.0)
            strike = self.planner.strike_target
            if strike is not None and strike.valid:
                # Legacy command path is spin-blind -> omega None (zeros);
                # promote to the EKF/spin estimate when that path lands.
                if self._use_fast_spec:
                    # Budget rule (measured, benchmark run_full_tick): the
                    # tight strike_spec_max_iter budget only converges FROM a
                    # warm start; a cold solve (first tick of a rally / after
                    # a failed solve) gets the solver's default budget once,
                    # then later ticks ride the cheap warm path.
                    self._last_spec = self._spec_planner.solve_fast_spec(
                        strike.p_ball, strike.v_ball, None,
                        self.planner.config.target_land[:2],
                        self._racket_speed_budget,
                        max_iter=(self._spec_max_iter
                                  if self._spec_warm_q is not None else None),
                        q0=self._spec_warm_q,
                        with_sensitivities=self._spec_sens,
                    )
                    if self._last_spec is not None:
                        sp = self._last_spec
                        self._spec_warm_q = np.array([
                            sp.tilt_pitch_deg, sp.tilt_yaw_deg, sp.v_n_signed,
                            sp.v_t_vec[0], sp.v_t_vec[1],
                        ])
                    else:
                        self._spec_warm_q = None  # cold restart next solve
                else:
                    self._last_spec = self._spec_planner.solve(
                        strike.p_ball, strike.v_ball, None,
                        self.planner.config.target_land[:2],
                        self._racket_speed_budget,
                    )
                log_now = self._last_spec is not None and (
                    not self._use_fast_spec or t >= self._spec_log_next_t)
                if log_now:
                    self._spec_log_next_t = t + 1.0
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
