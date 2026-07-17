"""ROS 2 node wrapping the HOPE 7-DOF racket planner.

Subscribes to ball position from the mocap `/poses` stream (the avatar_pro relay
on the Avatar Pro / Chingmu VRPN path) and publishes the desired racket state on
`/racket/command` as the shared
hope_msgs/RacketCommand. Diagnostics are published at 10 Hz.

Per HOPE rules the racket pose is never measured by motion capture; the
humanoid must achieve the commanded racket state via its own forward
kinematics. See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md.
"""

import time

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
from .flat_command_wire import (
    BASE_FLAT_SCHEMA_V1,
    BASE_FLAT_SCHEMA_V2_EPOCH,
    RACKET_FLAT_SCHEMA_V2_FACE179,
    RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
    RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
    pack_base_pose_flat,
    pack_invalid_racket_command_flat,
    pack_racket_command_flat_fail_closed,
)
from .task_lifecycle import (
    FormalBallTrackBoundary,
    FormalBallTrackDecision,
    FormalTaskLifecycle,
    FormalTaskRevision,
    FormalTaskState,
)
from .node_runtime_contract import (
    FormalBaseBarrierRejection,
    FormalBasePosePlausibilityGuard,
    FormalBaseSourceState,
    FormalSourceFrameContract,
    FormalWireCounters,
    FormalWireExhaustion,
    SolveCadence,
    SourceStampGuard,
    SwingSideSelector,
    base_pose_is_fresh,
    corrected_base_pose,
    latency_compensated_time_to_strike,
    ros_source_to_monotonic,
    ros_stamp_fields_to_seconds,
    validate_formal_source_clock_mode,
)
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
        # Prediction horizon is a launch/profile parameter. Gate3's ~1.89 s
        # arrivals need margin above the historical 2.0 s boundary because the
        # earliest fit can briefly predict a later crossing.
        self.declare_parameter("max_predict_time", 2.0)
        # Ingest every mocap sample, but solve/publish at the 50 Hz policy cadence.
        # ``0.0`` remains an explicit diagnostic override; it is no longer the
        # production default because a 300 Hz venue stream would otherwise run
        # Stage 1-3 in every subscription callback and starve base localization.
        self.declare_parameter("solve_period_s", 0.02)
        # PER-SIDE aim/flight (2026-07-08, from the Gate-3 rally vel-gate finding): the two
        # trained clips return in OPPOSITE cross-court directions (fh vy [+0.96,+1.96], bh vy
        # [-1.21,-0.21] world), so NO single target_land_y makes both sides' demanded racket
        # velocity land inside the trained boxes (best single aim = 5/10 sweep serves in-band;
        # the C++ runner's vel gate correctly stands on the rest). When set (non-NaN), aim and
        # flight time switch from THIS Stage-2 intercept's corrected base-yaw Y, using the
        # formal side Schmitt band below; Stage 3 is then rerun without estimator/strike-time
        # lag. NaN (default) = legacy single aim; arena values live in the yaml.
        self.declare_parameter("target_land_y_fh", float("nan"))
        self.declare_parameter("target_land_y_bh", float("nan"))
        self.declare_parameter("delta_t_flight_fh", float("nan"))
        self.declare_parameter("delta_t_flight_bh", float("nan"))
        # Explicit planner->policy side contract. The split is in base-yaw axes;
        # default 0.0 preserves the legacy runtime y-sign decision away from
        # the hysteresis band.
        self.declare_parameter("swing_side_split_y", 0.0)
        self.declare_parameter("swing_side_hysteresis_y", 0.04)
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
        # --- SHADOW spec solver (2026-07-12, flag-gated, DEFAULT OFF) ---
        # use_shadow_solver=True (use_* prefix = the package's boolean-enable
        # convention, matching use_kalman / use_fast_strike_spec) double-runs every
        # per-tick fast strike-spec solve through a SECOND backend (shadow_solver.py)
        # and reconciles the two — the promotion route for the torch batched solver,
        # same pattern as use_kalman above ("shadow-run the EKF next to the polyfit
        # estimator"): the planner still ACTS only on the production solve; the shadow
        # result goes nowhere except diagnostics KeyValues and the optional CSV.
        # A shadow exception is swallowed and counted, never a dead planner; a shadow
        # slower than its wall budget self-disables permanently (ShadowHarness).
        # NOTE: shadow mode is a VALIDATION mode — the double-run + CSV write are
        # synchronous in _poses_cb, so it doubles solve-tick occupancy and
        # shadow_log_path must be fast LOCAL disk (see shadow_solver.py docstring).
        # Requires the fast spec path (publish_strike_spec + use_fast_strike_spec) —
        # there is no per-tick solve to shadow otherwise. False (default)
        # short-circuits before any backend is constructed or timed: the command
        # path stays byte-identical.
        self.declare_parameter("use_shadow_solver", False)
        self.declare_parameter("shadow_solver_backend", "echo")  # "echo" = re-run the
                                                       # production solver (pipeline
                                                       # check, diff must be 0);
                                                       # "torch" fail-louds at startup
                                                       # until claude's solver lands
        self.declare_parameter("shadow_log_path", "")  # reconciliation CSV (one row per
                                                       # double-run); "" = no file written
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
        # documented "+x march"). For the 110-D hitter_pure deploy profile keep this FALSE:
        # robot_pose_topic keeps feeding /a3/base_pose_flat (the runner needs the live base),
        # but x_hit stays the static parameter and the runner derives the station from the
        # target (station_x = x_hit - 0.70, the trained plane offset).
        #
        # Keep the shared default TRUE for the current 175/177 multi-swing runtime. The
        # 110-D x-locked HitterPure profile must load hope_planner.hitter_pure.yaml, which
        # explicitly sets this FALSE. Plane semantics are therefore part of the model contract,
        # not a global default that can silently change an already-deployed policy.
        self.declare_parameter("x_hit_follow_robot", True)
        # --- FLAT outputs for the AGI native C++ runner (--planner, the ONLY control path) ---
        # The C++ a3_deploy_onnx_ref_pingpong subscribes std_msgs/Float64MultiArray (it avoids
        # vendoring hope_msgs typesupport on aarch64). We MIRROR /racket/command as a flat array
        # and stream the robot base pose (from robot_pose_topic) as a second flat array so the
        # runner's external_base localization has a live base.
        self.declare_parameter("publish_flat_cmd", True)
        self.declare_parameter("racket_flat_topic", "/racket/command_flat")
        self.declare_parameter("base_flat_topic", "/a3/base_pose_flat")
        # Formal side must expire with the C++ external-base localization gate;
        # receive-time monotonic age avoids trusting a stale/foreign ROS stamp.
        self.declare_parameter("base_pose_max_age_s", 0.2)
        self.declare_parameter("ball_source_stamp_max_age_s", 0.2)
        self.declare_parameter("source_stamp_future_tolerance_s", 0.02)
        self.declare_parameter("formal_source_clock_mode", "system")
        self.declare_parameter("formal_ball_source_frame_id", "")
        self.declare_parameter("formal_base_source_frame_id", "")
        self.declare_parameter("formal_common_frame_required", True)
        # Schema 1 is legacy. Schema 2 added the face tail but had no cross-topic
        # causality. Formal Gate3/179 must opt into schema 3, whose payload binds
        # a shared base-control epoch, strict sequence and source monotonic stamp.
        # Schema 4 adds one physical-ball task id and strictly increasing revisions.
        self.declare_parameter("racket_flat_schema", 1)
        self.declare_parameter("formal_task_no_ball_rearm_s", 0.10)
        self.declare_parameter("formal_task_plane_close_margin_m", 0.02)
        self.declare_parameter("formal_task_deadline_close_grace_s", 0.08)
        self.declare_parameter("formal_task_inbound_vx_threshold_mps", -0.30)
        self.declare_parameter("formal_task_outbound_vx_threshold_mps", 0.30)
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
        self._robot_position_w = None
        self._robot_quaternion_wxyz = None
        self._formal_base_state = FormalBaseSourceState()
        self._formal_base_pose_guard = FormalBasePosePlausibilityGuard()
        # per-side aim/flight (NaN = disabled); consumed in _poses_cb before the solve
        self._land_y_fh = float(self.get_parameter("target_land_y_fh").value)
        self._land_y_bh = float(self.get_parameter("target_land_y_bh").value)
        self._dtf_fh = float(self.get_parameter("delta_t_flight_fh").value)
        self._dtf_bh = float(self.get_parameter("delta_t_flight_bh").value)
        self._per_side_aim = not (np.isnan(self._land_y_fh) and np.isnan(self._land_y_bh)
                                  and np.isnan(self._dtf_fh) and np.isnan(self._dtf_bh))
        self._last_intercept_y = None  # last valid plan's arrival y (diagnostics)
        self._side_selector = SwingSideSelector(
            split_y=float(self.get_parameter("swing_side_split_y").value),
            hysteresis_y=float(self.get_parameter("swing_side_hysteresis_y").value),
        )
        self._solve_cadence = SolveCadence(
            period_s=float(self.get_parameter("solve_period_s").value)
        )
        self._publish_flat = bool(self.get_parameter("publish_flat_cmd").value)
        self._racket_flat_schema = int(self.get_parameter("racket_flat_schema").value)
        if self._racket_flat_schema not in (1, 2, 3, 4):
            raise ValueError(
                "racket_flat_schema must be 1 (legacy), 2 (face), "
                "3 (formal epoch), or 4 (formal task revision)"
            )
        if self._racket_flat_schema in (2, 3, 4) and not self._publish_flat:
            raise ValueError("face racket schemas require publish_flat_cmd=true")
        self._base_flat_schema = (
            BASE_FLAT_SCHEMA_V2_EPOCH
            if self._racket_flat_schema in (
                RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
                RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
            )
            else BASE_FLAT_SCHEMA_V1
        )
        self._wire_counters = FormalWireCounters()
        self._task_lifecycle = None
        self._task_boundary = None
        self._pending_ball_discontinuity = False
        if self._racket_flat_schema == RACKET_FLAT_SCHEMA_V4_FACE179_TASK:
            self._task_lifecycle = FormalTaskLifecycle(
                control_epoch=self._wire_counters.control_epoch
            )
            self._task_boundary = FormalBallTrackBoundary(
                no_ball_rearm_s=float(
                    self.get_parameter("formal_task_no_ball_rearm_s").value
                ),
                plane_close_margin_m=float(
                    self.get_parameter("formal_task_plane_close_margin_m").value
                ),
                deadline_close_grace_s=float(
                    self.get_parameter(
                        "formal_task_deadline_close_grace_s"
                    ).value
                ),
                inbound_vx_threshold_mps=float(
                    self.get_parameter(
                        "formal_task_inbound_vx_threshold_mps"
                    ).value
                ),
                outbound_vx_threshold_mps=float(
                    self.get_parameter(
                        "formal_task_outbound_vx_threshold_mps"
                    ).value
                ),
            )
        self._base_pose_max_age_s = float(
            self.get_parameter("base_pose_max_age_s").value
        )
        # Pure helper validates the launch parameter at startup.
        base_pose_is_fresh(None, time.monotonic(), self._base_pose_max_age_s)
        source_future_tolerance_s = float(
            self.get_parameter("source_stamp_future_tolerance_s").value
        )
        self._base_source_guard = SourceStampGuard(
            max_age_s=self._base_pose_max_age_s,
            future_tolerance_s=source_future_tolerance_s,
        )
        self._ball_source_guard = SourceStampGuard(
            max_age_s=float(self.get_parameter("ball_source_stamp_max_age_s").value),
            future_tolerance_s=source_future_tolerance_s,
        )
        self._formal_source_frames = None
        if self._racket_flat_schema in (
            RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
            RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
        ):
            if not self.has_parameter("use_sim_time"):
                raise ValueError(
                    "formal schema 3 requires ROS to declare use_sim_time"
                )
            validate_formal_source_clock_mode(
                self.get_parameter("use_sim_time").value,
                self.get_parameter("formal_source_clock_mode").value,
            )
            self._formal_source_frames = FormalSourceFrameContract(
                ball_frame_id=self.get_parameter(
                    "formal_ball_source_frame_id"
                ).value,
                base_frame_id=self.get_parameter(
                    "formal_base_source_frame_id"
                ).value,
                common_frame_required=self.get_parameter(
                    "formal_common_frame_required"
                ).value,
            )
        self._base_ready_cadence = SolveCadence(period_s=0.1)
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
            max_predict_time=float(self.get_parameter("max_predict_time").value),
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

        # Flag-gated SHADOW solver harness (see the parameter block). None = off =
        # the only new code on the hot path is `is not None` checks; the harness,
        # backend and CSV recorder are never constructed.
        self._shadow = None
        self._shadow_problem_cls = None
        self._n_shadow_node_errors = 0   # harness-call failures caught in _poses_cb
        self._shadow_disabled_reason = None  # set when requested but not runnable —
                                             # surfaced in diagnostics every cycle so a
                                             # mis-launch is never fail-quiet (an empty
                                             # CSV must not pass for "zero diffs")
        if bool(self.get_parameter("use_shadow_solver").value):
            if self._spec_planner is None or not self._use_fast_spec:
                self._shadow_disabled_reason = "fast_spec_path_off"
                self.get_logger().warning(
                    "use_shadow_solver=True needs the per-tick fast spec path "
                    "(publish_strike_spec:=true use_fast_strike_spec:=true); "
                    "shadow DISABLED — nothing to reconcile against "
                    "(diagnostics will carry shadow_backend=DISABLED(fast_spec_path_off)).")
            else:
                from .shadow_solver import ShadowHarness, ShadowProblem, make_backend

                # NOTE: a torch backend request fail-louds HERE (NotImplementedError
                # at construction) until claude's solver lands — by design, so a
                # mis-launched shadow dies at startup instead of running echo-less.
                backend = make_backend(
                    str(self.get_parameter("shadow_solver_backend").value),
                    physics=self._spec_planner.physics,
                    config=self._spec_planner.config,
                    table=self._spec_planner.table,
                )
                log_path = str(self.get_parameter("shadow_log_path").value) or None
                self._shadow = ShadowHarness(backend, log_path)
                self._shadow_problem_cls = ShadowProblem
                self.get_logger().info(
                    f"SHADOW spec solver ON: backend={backend.name}, "
                    f"log={'off' if log_path is None else log_path} "
                    "(production command path unaffected; diffs in diagnostics + CSV)")

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
                received_now = time.monotonic()
                # Expire the previous source-age lease before inspecting or
                # admitting this callback's candidate. If this advances the
                # epoch, a delayed pre-barrier source cannot immediately
                # recover it merely because the DDS receive is new.
                expired_this_callback = self._expire_formal_base_if_needed(received_now)
                base_source_monotonic_s = received_now
                base_source_stamp_s = None
                if self._racket_flat_schema in (
                    RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
                    RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
                ):
                    try:
                        self._formal_source_frames.validate_base(msg.header.frame_id)
                    except ValueError as exc:
                        # A base-frame mismatch invalidates localization and
                        # therefore owns a new control epoch. If source-age
                        # expiry already advanced it above, do not double-step.
                        self._revoke_formal_base(
                            "base source frame_id rejected",
                            force_epoch=not expired_this_callback,
                        )
                        self.get_logger().warning(
                            f"base source frame_id rejected ({exc}); geometry suppressed",
                            throttle_duration_sec=2.0,
                        )
                        return
                    try:
                        source_stamp_s = self._source_stamp_s(msg)
                        base_source_stamp_s = source_stamp_s
                        now_ros_s = self._now_ros_s()
                        self._base_source_guard.validate(
                            source_stamp_s, now_ros_s
                        )
                        base_source_monotonic_s = self._source_monotonic_from_ros(
                            source_stamp_s,
                            now_ros_s=now_ros_s,
                            now_monotonic_s=received_now,
                        )
                        self._formal_base_state.validate_candidate(
                            base_source_monotonic_s,
                            now_monotonic_s=received_now,
                            max_age_s=self._base_pose_max_age_s,
                        )
                    except ValueError as exc:
                        barrier_only_rejection = isinstance(
                            exc, FormalBaseBarrierRejection
                        )
                        self._revoke_formal_base(
                            "stale/regressing robot marker source",
                            force_epoch=(
                                not expired_this_callback
                                and not barrier_only_rejection
                            ),
                        )
                        self.get_logger().warning(
                            f"robot marker source stamp rejected ({exc})",
                            throttle_duration_sec=2.0,
                        )
                        return
                p = msg.pose.position
                q = msg.pose.orientation
                try:
                    base_w, quat_wxyz = corrected_base_pose(
                        (float(p.x), float(p.y), float(p.z)),
                        (float(q.w), float(q.x), float(q.y), float(q.z)),
                        self._marker_to_base,
                        policy_z_offset=self._policy_z_offset,
                    )
                    if self._racket_flat_schema in (
                        RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
                        RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
                    ):
                        self._formal_base_pose_guard.validate(
                            base_w, quat_wxyz, base_source_monotonic_s
                        )
                except ValueError as exc:
                    # Do not keep using an older yaw frame after the newest
                    # base sample proves unusable. Formal schema 3 publishes an
                    # invalid revocation until a complete corrected pose lands.
                    self._revoke_formal_base(
                        "invalid robot marker pose",
                        force_epoch=not expired_this_callback,
                    )
                    self.get_logger().warning(
                        f"invalid robot marker pose ({exc}); base update suppressed",
                        throttle_duration_sec=2.0,
                    )
                    return
                # Adaptive x, side selection, and the published flat consume the
                # exact same corrected base. The local marker->base offset must
                # not be added directly in world axes after the robot turns.
                bx, by, bz = (float(v) for v in base_w)
                admission_now = time.monotonic()
                # Recheck at the actual state-install boundary: geometry work
                # must not let the prior lease age out between callback entry
                # and candidate admission without advancing the epoch first.
                expired_during_admission = self._expire_formal_base_if_needed(
                    admission_now
                )
                expired_this_callback = (
                    expired_this_callback or expired_during_admission
                )
                try:
                    if self._racket_flat_schema in (
                        RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
                        RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
                    ):
                        self._formal_base_state.accept_source(
                            base_source_monotonic_s,
                            now_monotonic_s=admission_now,
                            max_age_s=self._base_pose_max_age_s,
                        )
                    else:
                        # Legacy face schema has no source stamp. Preserve its
                        # receive-age behavior without weakening schema 3.
                        self._formal_base_state.lease.accept(
                            received_now,
                            now_monotonic_s=admission_now,
                            max_age_s=self._base_pose_max_age_s,
                        )
                except ValueError as exc:
                    barrier_only_rejection = isinstance(
                        exc, FormalBaseBarrierRejection
                    )
                    self._revoke_formal_base(
                        "base source lease rejected",
                        force_epoch=(
                            not expired_this_callback
                            and not barrier_only_rejection
                        ),
                    )
                    self.get_logger().warning(
                        f"base source lease rejected ({exc}); geometry suppressed",
                        throttle_duration_sec=2.0,
                    )
                    return
                if self._racket_flat_schema in (
                    RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
                    RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
                ):
                    self._formal_base_pose_guard.commit(
                        base_w, quat_wxyz, base_source_monotonic_s
                    )
                if base_source_stamp_s is not None:
                    self._base_source_guard.commit(base_source_stamp_s)
                self._robot_x = bx
                self._robot_position_w = base_w.copy()
                self._robot_quaternion_wxyz = quat_wxyz.copy()
                self._publish_flat_base(
                    valid=True,
                    position_w=(bx, by, bz),
                    quaternion_wxyz=tuple(float(v) for v in quat_wxyz),
                    source_monotonic_s=base_source_monotonic_s,
                )
                # READY is tied to this exact mapped source stamp, not to its
                # newly received callback. A source that ages out during the
                # callback can therefore never announce receive-fresh READY.
                self._emit_base_ready_from_new_sample(
                    base_source_monotonic_s, time.monotonic()
                )

            self.create_subscription(PoseStamped, robot_pose_topic, _robot_pose_cb, mocap_qos)
            if self._x_hit_follow_robot:
                self.get_logger().warning(
                    f"adaptive x_hit ON (FOLLOW-MODE): robot pose from '{robot_pose_topic}', "
                    f"x_hit = clamp(robot_x + {self._x_hit_offset:.2f}, "
                    f"[{self._x_hit_min:.2f}, {self._x_hit_max:.2f}]). "
                    "Required by current 175/177 rallies; incompatible with HitterPure/x-locked "
                    "policies, which must load hope_planner.hitter_pure.yaml.")
            else:
                self.get_logger().info(
                    f"x_hit FIXED at {config.x_hit:.2f} (x-locked / hitter_pure profile, paper "
                    f"§IV-B); robot pose from '{robot_pose_topic}' feeds the base flat only")

        # Diagnostics at 10 Hz.
        self._n_received = 0
        self._n_planner_valid = 0
        self._n_valid = 0
        self._n_flat_contract_rejected = 0
        self._n_custom_mirror_rejected = 0
        self._last_valid = False
        self._last_tts = float("nan")
        self._last_effective_tts = float("nan")
        self._last_ball_source_age_s = float("nan")
        self._last_ball_plane_distance_m = float("nan")
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

    def _next_sequence(self, field: str) -> int:
        try:
            if field == "_racket_sequence":
                return self._wire_counters.next_racket()
            if field == "_base_sequence":
                return self._wire_counters.next_base()
            raise ValueError(f"unknown formal wire sequence field {field}")
        except FormalWireExhaustion as exc:
            self._publish_terminal_wire_exhaustion(str(exc))

    def _advance_control_epoch(self) -> None:
        try:
            self._wire_counters.advance_epoch()
            if self._task_lifecycle is not None:
                self._task_lifecycle.observe_epoch(
                    self._wire_counters.control_epoch
                )
                assert self._task_boundary is not None
                self._task_boundary.reset_epoch()
                self._pending_ball_discontinuity = False
        except FormalWireExhaustion as exc:
            self._publish_terminal_wire_exhaustion(str(exc))

    def _publish_terminal_wire_exhaustion(self, reason: str) -> None:
        """Spend the reserved MAX counters on one terminal double revoke."""

        epoch, racket_sequence, base_sequence = (
            self._wire_counters.reserve_terminal_invalid()
        )
        stamp = time.monotonic()
        errors = []
        if self.flat_cmd_pub is not None:
            try:
                msg = Float64MultiArray()
                racket_kwargs = dict(
                    schema=self._racket_flat_schema,
                    control_epoch=epoch,
                    command_sequence=racket_sequence,
                    base_sequence_ref=base_sequence,
                    source_monotonic_s=stamp,
                )
                if self._racket_flat_schema == RACKET_FLAT_SCHEMA_V4_FACE179_TASK:
                    racket_kwargs.update(task_id=0, task_revision=0)
                msg.data = pack_invalid_racket_command_flat(**racket_kwargs)
                self.flat_cmd_pub.publish(msg)
            except Exception as exc:
                errors.append(f"racket={type(exc).__name__}: {exc}")
        if self.flat_base_pub is not None:
            try:
                msg = Float64MultiArray()
                msg.data = pack_base_pose_flat(
                    schema=BASE_FLAT_SCHEMA_V2_EPOCH,
                    valid=False,
                    position_w=(0.0, 0.0, 0.0),
                    quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    control_epoch=epoch,
                    base_sequence=base_sequence,
                    source_monotonic_s=stamp,
                )
                self.flat_base_pub.publish(msg)
            except Exception as exc:
                errors.append(f"base={type(exc).__name__}: {exc}")
        detail = "; ".join(errors) if errors else "terminal double revoke published"
        raise RuntimeError(f"{reason}; {detail}; runner restart is required")

    def _source_stamp_s(self, msg) -> float:
        return ros_stamp_fields_to_seconds(
            msg.header.stamp.sec, msg.header.stamp.nanosec
        )

    def _now_ros_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1.0e-9

    def _source_monotonic_from_ros(
        self, source_stamp_s: float, *, now_ros_s: float, now_monotonic_s: float
    ) -> float:
        # Same-host clock mapping: retain the source's age instead of granting
        # a new freshness lease at planner publication time.
        return ros_source_to_monotonic(
            source_stamp_s, now_ros_s, now_monotonic_s
        )

    def _publish_flat_racket_invalid(
        self,
        source_monotonic_s: float | None = None,
        *,
        task_revision: FormalTaskRevision | None = None,
    ) -> None:
        if self.flat_cmd_pub is None:
            return
        kwargs = {"schema": self._racket_flat_schema}
        if self._racket_flat_schema in (
            RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
            RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
        ):
            kwargs.update(
                control_epoch=self._wire_counters.control_epoch,
                command_sequence=self._next_sequence("_racket_sequence"),
                # Bind every racket event, including revocations, to the most
                # recent base publication *attempt*. If that attempt failed or
                # has not reached the subscriber yet, the runner sees a tuple
                # mismatch instead of reusing its older base sample.
                base_sequence_ref=self._wire_counters.base_sequence,
                source_monotonic_s=(
                    time.monotonic()
                    if source_monotonic_s is None
                    else float(source_monotonic_s)
                ),
            )
        if self._racket_flat_schema == RACKET_FLAT_SCHEMA_V4_FACE179_TASK:
            if task_revision is None:
                kwargs.update(task_id=0, task_revision=0)
            else:
                if task_revision.control_epoch != self._wire_counters.control_epoch:
                    raise RuntimeError(
                        "task revision epoch does not match the live control epoch"
                    )
                kwargs.update(
                    task_id=task_revision.task_id,
                    task_revision=task_revision.task_revision,
                )
        racket = Float64MultiArray()
        racket.data = pack_invalid_racket_command_flat(**kwargs)
        self.flat_cmd_pub.publish(racket)

    def _disarm_formal_task(self) -> None:
        """Forget task identity when the ball source itself is untrustworthy."""

        if self._task_lifecycle is None:
            return
        self._task_lifecycle.disarm(self._wire_counters.control_epoch)
        assert self._task_boundary is not None
        self._task_boundary.reset_epoch()
        self._pending_ball_discontinuity = False

    def _formal_task_active(self) -> bool:
        return (
            self._task_lifecycle is not None
            and self._task_lifecycle.state is FormalTaskState.ACTIVE
        )

    def _publish_task_close(
        self, source_monotonic_s: float | None
    ) -> FormalTaskRevision | None:
        """Close one live task and publish its final task-scoped invalid row."""

        assert self._task_lifecycle is not None
        terminal = self._task_lifecycle.close(
            self._wire_counters.control_epoch
        )
        if terminal is not None:
            self._publish_flat_racket_invalid(
                source_monotonic_s, task_revision=terminal
            )
            if self.cmd_pub is not None:
                try:
                    mirror = RacketCommand()
                    mirror.normal.x = 1.0
                    mirror.valid = False
                    self.cmd_pub.publish(mirror)
                except Exception as exc:
                    self._n_custom_mirror_rejected += 1
                    self.get_logger().error(
                        "optional RacketCommand task-close mirror failed after "
                        f"formal flat publication: {type(exc).__name__}: {exc}",
                        throttle_duration_sec=2.0,
                    )
        return terminal

    def _observe_formal_task_present(
        self,
        *,
        source_time_s: float,
        source_monotonic_s: float | None,
        raw_ball_position: np.ndarray,
        predicted_strike_time_s: float | None,
    ) -> tuple[bool, bool]:
        """Update the schema-4 ball boundary.

        Returns ``(continue_current_sample, inbound_track_ready)``. A task is
        closed before any newer command for that sample is considered. A
        physically proven new inbound ball may close/rearm in one callback,
        but transport sequence numbers and solver validity never create tasks.
        """

        if self._task_lifecycle is None:
            return True, False
        assert self._task_boundary is not None
        ball_x = float(raw_ball_position[0])
        ball_vx = None
        if self.planner.estimator.ready:
            p_est, v_est, _ = self.planner.estimator.estimate()
            ball_x = float(p_est[0])
            ball_vx = float(v_est[0])
        self._last_ball_plane_distance_m = (
            ball_x - float(self.planner.config.x_hit)
        )
        discontinuity = bool(
            self._pending_ball_discontinuity
            or self.planner.estimator.discontinuity_detected
        )
        # The estimator deliberately clears its fit window on the jump, so a
        # velocity estimate can be unavailable on the detection tick. Retain
        # that physical-boundary proof until a later fit classifies direction.
        self._pending_ball_discontinuity = (
            discontinuity if ball_vx is None else False
        )
        decision = self._task_boundary.observe_present(
            source_time_s,
            ball_x_m=ball_x,
            ball_vx_mps=ball_vx,
            discontinuity_detected=discontinuity,
            task_active=self._formal_task_active(),
            strike_plane_x_m=float(self.planner.config.x_hit),
            predicted_strike_time_s=predicted_strike_time_s,
        )
        if decision in (
            FormalBallTrackDecision.CLOSE_ACTIVE,
            FormalBallTrackDecision.CLOSE_AND_REARM,
        ):
            self._publish_task_close(source_monotonic_s)
            if decision is FormalBallTrackDecision.CLOSE_ACTIVE:
                return False, False
        if decision in (
            FormalBallTrackDecision.SAFE_REARM,
            FormalBallTrackDecision.CLOSE_AND_REARM,
        ):
            self._task_lifecycle.explicit_rearm(
                self._wire_counters.control_epoch,
                no_ball_or_new_serve_confirmed=True,
            )
        inbound = ball_vx is not None and (
            ball_vx <= self._task_boundary.inbound_vx_threshold_mps
        )
        return True, inbound

    def _observe_formal_task_absent(
        self, *, source_time_s: float, source_monotonic_s: float | None
    ) -> None:
        """Handle a trustworthy no-ball tick without inventing a new task."""

        if self._task_lifecycle is None:
            return
        assert self._task_boundary is not None
        decision = self._task_boundary.observe_absent(
            source_time_s, task_active=self._formal_task_active()
        )
        if decision is FormalBallTrackDecision.CLOSE_ACTIVE:
            self._publish_task_close(source_monotonic_s)
            return
        revision = self._task_lifecycle.publish(
            self._wire_counters.control_epoch,
            inbound_track_ready=False,
            solver_valid=False,
        )
        self._publish_flat_racket_invalid(
            source_monotonic_s, task_revision=revision
        )

    def _publish_flat_base(
        self, *, valid: bool, position_w=None, quaternion_wxyz=None,
        source_monotonic_s: float | None = None,
    ) -> None:
        if self.flat_base_pub is None:
            return
        position_w = (0.0, 0.0, 0.0) if position_w is None else position_w
        quaternion_wxyz = (
            (1.0, 0.0, 0.0, 0.0) if quaternion_wxyz is None else quaternion_wxyz
        )
        kwargs = {
            "schema": self._base_flat_schema,
            "valid": valid,
            "position_w": position_w,
            "quaternion_wxyz": quaternion_wxyz,
        }
        if self._base_flat_schema == BASE_FLAT_SCHEMA_V2_EPOCH:
            kwargs.update(
                control_epoch=self._wire_counters.control_epoch,
                base_sequence=self._next_sequence("_base_sequence"),
                source_monotonic_s=(
                    time.monotonic()
                    if source_monotonic_s is None
                    else float(source_monotonic_s)
                ),
            )
        base = Float64MultiArray()
        base.data = pack_base_pose_flat(**kwargs)
        self.flat_base_pub.publish(base)

    def _formal_base_is_fresh(self, now_monotonic_s: float) -> bool:
        return self._formal_base_state.lease.fresh(
            now_monotonic_s, self._base_pose_max_age_s
        )

    def _clear_formal_base_geometry(self) -> None:
        self._robot_x = None
        self._robot_position_w = None
        self._robot_quaternion_wxyz = None
        self._side_selector.sign = 0.0

    def _publish_formal_revocation(self, reason: str) -> None:
        """Publish canonical formal racket and base invalid rows.

        The formal flat topics are the control plane. Clearing Python state is
        not a revoke, and the optional custom message is published only after
        both formal invalid rows have been attempted.
        """

        if self._racket_flat_schema not in (
            RACKET_FLAT_SCHEMA_V2_FACE179,
            RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
            RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
        ):
            return
        flat_errors = []
        try:
            self._publish_flat_racket_invalid()
        except Exception as exc:
            flat_errors.append(f"racket={type(exc).__name__}: {exc}")
        try:
            self._publish_flat_base(valid=False)
        except Exception as exc:
            flat_errors.append(f"base={type(exc).__name__}: {exc}")
        if self.cmd_pub is not None:
            try:
                mirror = RacketCommand()
                mirror.normal.x = 1.0
                mirror.valid = False
                self.cmd_pub.publish(mirror)
            except Exception as exc:
                self._n_custom_mirror_rejected += 1
                self.get_logger().error(
                    "optional RacketCommand revoke mirror failed after formal flat "
                    f"publication: {type(exc).__name__}: {exc}",
                    throttle_duration_sec=2.0,
                )
        self.get_logger().warning(
            f"{reason}; formal racket/base epoch revoked",
            throttle_duration_sec=2.0,
        )
        if flat_errors:
            raise RuntimeError(
                "formal dual-topic revocation attempted but failed: " + "; ".join(flat_errors)
            )

    def _revoke_formal_base(self, reason: str, *, force_epoch: bool = False) -> bool:
        transitioned = self._formal_base_state.revoke(
            time.monotonic(), force=force_epoch
        )
        self._clear_formal_base_geometry()
        if transitioned:
            if self._racket_flat_schema in (
                RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
                RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
            ):
                self._advance_control_epoch()
            self._publish_formal_revocation(reason)
        return transitioned

    def _expire_formal_base_if_needed(self, now_monotonic_s: float) -> bool:
        if self._racket_flat_schema not in (
            RACKET_FLAT_SCHEMA_V2_FACE179,
            RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
            RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
        ) or not self._formal_base_state.expire_before_admission(
            now_monotonic_s, self._base_pose_max_age_s
        ):
            return False
        self._clear_formal_base_geometry()
        if self._racket_flat_schema in (
            RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
            RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
        ):
            self._advance_control_epoch()
        self._publish_formal_revocation("corrected base pose stale")
        return True

    def _emit_base_ready_from_new_sample(
        self, base_source_monotonic_s: float, now_monotonic_s: float
    ) -> bool:
        """Emit at most 10 Hz only while this exact source lease is fresh."""

        if (self._racket_flat_schema not in (
                RACKET_FLAT_SCHEMA_V2_FACE179,
                RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
                RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
            ) or self.flat_cmd_pub is None
                or not self._formal_base_state.ready_for_source(
                    base_source_monotonic_s,
                    now_monotonic_s,
                    self._base_pose_max_age_s,
                )
                or not self._base_ready_cadence.admit(now_monotonic_s)):
            return False
        self.get_logger().info("HOPE planner READY: corrected base pose fresh")
        return True

    def _poses_cb(self, msg: PoseArray) -> None:
        self._n_received += 1
        ball_source_monotonic_s = None
        formal_epoch_schema = self._racket_flat_schema in (
            RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
            RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
        )
        if formal_epoch_schema:
            try:
                self._formal_source_frames.validate_ball(msg.header.frame_id)
            except ValueError as exc:
                # Ball-frame failure revokes only the racket command. Base
                # localization/epoch remains independently valid, and neither
                # estimator may ingest a sample in the wrong frame.
                self._last_valid = False
                self._last_tts = float("nan")
                self._last_effective_tts = float("nan")
                self._last_ball_source_age_s = float("nan")
                self._disarm_formal_task()
                self._publish_flat_racket_invalid()
                self.get_logger().warning(
                    f"ball source frame_id rejected ({exc}); formal racket command revoked",
                    throttle_duration_sec=2.0,
                )
                return
        try:
            t = self._source_stamp_s(msg)
        except ValueError as exc:
            if formal_epoch_schema:
                self._disarm_formal_task()
                self._publish_flat_racket_invalid()
            self.get_logger().warning(
                f"ball source stamp fields rejected ({exc})",
                throttle_duration_sec=2.0,
            )
            return
        # Preserve the older schema-3 no-pose behavior. Schema 4 first proves
        # the source clock so an empty PoseArray can safely contribute to the
        # explicit no-ball/new-serve boundary.
        if (
            len(msg.poses) <= self._ball_index
            and self._racket_flat_schema != RACKET_FLAT_SCHEMA_V4_FACE179_TASK
        ):
            if formal_epoch_schema:
                self._publish_flat_racket_invalid()
            return
        if formal_epoch_schema:
            try:
                now_ros_s = self._now_ros_s()
                self._ball_source_guard.validate(float(t), now_ros_s)
                now_monotonic_s = time.monotonic()
                ball_source_monotonic_s = self._source_monotonic_from_ros(
                    float(t),
                    now_ros_s=now_ros_s,
                    now_monotonic_s=now_monotonic_s,
                )
            except ValueError as exc:
                self._last_valid = False
                self._last_tts = float("nan")
                self._last_effective_tts = float("nan")
                self._last_ball_source_age_s = float("nan")
                self._disarm_formal_task()
                self._publish_flat_racket_invalid(ball_source_monotonic_s)
                self.get_logger().warning(
                    f"ball source stamp rejected ({exc}); formal racket command revoked",
                    throttle_duration_sec=2.0,
                )
                return

        if len(msg.poses) <= self._ball_index:
            # Reaching this branch is schema 4 with a validated source stamp.
            self._ball_source_guard.commit(float(t))
            self._last_valid = False
            self._last_tts = float("nan")
            self._last_effective_tts = float("nan")
            self._last_ball_source_age_s = max(0.0, self._now_ros_s() - float(t))
            self._observe_formal_task_absent(
                source_time_s=float(t),
                source_monotonic_s=ball_source_monotonic_s,
            )
            return

        # NOTE: PoseArray carries no names. Configure ball_pose_index to match
        # the ball's slot in the /poses ordering (the avatar_pro relay puts the
        # ball first), or swap this for a /tf lookup keyed on ball_rigid_body_name.
        pose = msg.poses[self._ball_index]
        p_ball = np.array([pose.position.x, pose.position.y, pose.position.z])
        if formal_epoch_schema:
            if not np.isfinite(p_ball).all():
                self._last_valid = False
                self._last_effective_tts = float("nan")
                self._last_ball_source_age_s = float("nan")
                self._disarm_formal_task()
                self._publish_flat_racket_invalid(ball_source_monotonic_s)
                return
            self._ball_source_guard.commit(float(t))

        # Keep every 300 Hz measurement in both estimators, but only solve and
        # publish on admitted cadence ticks. A skipped callback deliberately
        # does not republish the cached command, so it cannot refresh stale
        # data at the C++ subscriber. Expire the independent base lease before
        # cadence admission: a skipped ball solve must never postpone a safety
        # revoke until the 10 Hz diagnostics timer.
        self._expire_formal_base_if_needed(time.monotonic())
        try:
            solve_now = self._solve_cadence.admit(t)
        except ValueError as exc:
            self.get_logger().warning(
                f"invalid planner timestamp ({exc}); sample suppressed",
                throttle_duration_sec=2.0,
            )
            return
        if not solve_now:
            self.planner.push_measurement(t, p_ball)
            if self._task_lifecycle is not None:
                self._pending_ball_discontinuity = bool(
                    self._pending_ball_discontinuity
                    or self.planner.estimator.discontinuity_detected
                )
            if self._kf is not None:
                self._kf.push(t, p_ball)
            return

        # adaptive hit plane: track the live robot (see robot_pose_topic above). Mutating
        # config.x_hit is safe — the predictor reads it per predict() call. Disabled by
        # x_hit_follow_robot=false (hitter_pure profile: FIXED plane, paper §IV-B).
        if self._robot_x is not None and self._x_hit_follow_robot:
            self.planner.config.x_hit = float(
                np.clip(self._robot_x + self._x_hit_offset, self._x_hit_min, self._x_hit_max))

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

        predicted_strike_time_s = None
        strike_target = self.planner.strike_target
        if strike_target is not None:
            candidate_strike_time = float(strike_target.t_strike)
            if np.isfinite(candidate_strike_time) and candidate_strike_time >= 0.0:
                predicted_strike_time_s = candidate_strike_time
        continue_sample, inbound_track_ready = self._observe_formal_task_present(
            source_time_s=float(t),
            source_monotonic_s=ball_source_monotonic_s,
            raw_ball_position=p_ball,
            predicted_strike_time_s=predicted_strike_time_s,
        )
        if not continue_sample:
            self._last_valid = False
            self._last_tts = float("nan")
            self._last_effective_tts = float("nan")
            return

        if cmd is None:
            self._last_valid = False
            self._last_tts = float("nan")
            self._last_effective_tts = float("nan")
            if self._task_lifecycle is not None:
                revision = self._task_lifecycle.publish(
                    self._wire_counters.control_epoch,
                    inbound_track_ready=inbound_track_ready,
                    solver_valid=False,
                )
                self._publish_flat_racket_invalid(
                    ball_source_monotonic_s, task_revision=revision
                )
            elif self._racket_flat_schema in (
                RACKET_FLAT_SCHEMA_V2_FACE179,
                RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
            ):
                self._publish_flat_racket_invalid(ball_source_monotonic_s)
            if self.cmd_pub is not None and self._racket_flat_schema in (2, 3, 4):
                out = RacketCommand()
                out.header = msg.header
                out.header.frame_id = "world"
                out.normal.x = 1.0
                out.valid = False
                self.cmd_pub.publish(out)
            return

        # Select the clip from THIS Stage-2 intercept and the corrected base,
        # then optionally rerun Stage 3 if an opt-in per-side aim changed. A
        # formal schema-3 command without a base/Stage-2 tuple keeps sign=0 and
        # is revoked by the wire packer instead of guessing a clip.
        swing_sign = 0.0
        strike = self.planner.strike_target
        if (cmd.valid and strike is not None and strike.valid
                and self._robot_position_w is not None
                and self._robot_quaternion_wxyz is not None):
            intercept_w = np.asarray(strike.p_ball, dtype=float).copy()
            intercept_w[2] += self._policy_z_offset
            try:
                swing_sign = self._side_selector.select(
                    intercept_w,
                    self._robot_position_w,
                    self._robot_quaternion_wxyz,
                )
            except ValueError as exc:
                self.get_logger().warning(
                    f"invalid base-yaw side tuple ({exc}); formal command will be revoked",
                    throttle_duration_sec=2.0,
                )
                swing_sign = 0.0
            if self._per_side_aim and swing_sign in (-1.0, 1.0):
                forehand = swing_sign > 0.0
                land_y = self._land_y_fh if forehand else self._land_y_bh
                dtf = self._dtf_fh if forehand else self._dtf_bh
                aim_changed = False
                if not np.isnan(land_y) and self.planner.config.target_land[1] != land_y:
                    self.planner.config.target_land[1] = land_y
                    aim_changed = True
                if not np.isnan(dtf) and self.planner.config.delta_t_flight != dtf:
                    self.planner.config.delta_t_flight = dtf
                    aim_changed = True
                if aim_changed:
                    cmd = self.planner.replan_latest()
                    if cmd is None:
                        self._last_valid = False
                        if self._task_lifecycle is not None:
                            revision = self._task_lifecycle.publish(
                                self._wire_counters.control_epoch,
                                inbound_track_ready=inbound_track_ready,
                                solver_valid=False,
                            )
                            self._publish_flat_racket_invalid(
                                ball_source_monotonic_s,
                                task_revision=revision,
                            )
                        elif self._racket_flat_schema in (2, 3):
                            self._publish_flat_racket_invalid(ball_source_monotonic_s)
                        return

        task_revision = None
        if self._task_lifecycle is not None:
            task_revision = self._task_lifecycle.publish(
                self._wire_counters.control_epoch,
                inbound_track_ready=inbound_track_ready,
                solver_valid=bool(cmd.valid),
            )
        wire_command_valid = bool(cmd.valid) and (
            self._task_lifecycle is None or task_revision is not None
        )
        self._last_valid = wire_command_valid
        tts = self.planner.time_to_strike
        if (
            tts is not None
            and self._racket_flat_schema == RACKET_FLAT_SCHEMA_V4_FACE179_TASK
        ):
            publish_now_ros_s = self._now_ros_s()
            self._last_ball_source_age_s = max(
                0.0, publish_now_ros_s - float(t)
            )
            # Keep the wire value sample-relative. Its source_monotonic_s maps
            # the original header stamp, and the C++ mailbox subtracts the
            # resulting end-to-end age exactly once. The compensated value is
            # diagnostic only; publishing it would double-count latency.
            self._last_tts = float(tts)
            self._last_effective_tts = latency_compensated_time_to_strike(
                float(tts), float(t), publish_now_ros_s
            )
        else:
            self._last_tts = tts if tts is not None else float("nan")
            self._last_effective_tts = self._last_tts
        if cmd.valid:
            self._n_planner_valid += 1

        # Validate the formal flat command before publishing either wire. If
        # a formal face command is malformed, both outputs publish a safe invalid
        # row/message; the optional hope_msgs mirror must not claim validity
        # after the official flat path revoked the command.
        flat_data = None
        flat_contract_error = None
        if self.flat_cmd_pub is not None:
            formal_wire = {}
            if self._racket_flat_schema in (
                RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
                RACKET_FLAT_SCHEMA_V4_FACE179_TASK,
            ):
                formal_wire = {
                    "control_epoch": self._wire_counters.control_epoch,
                    "command_sequence": self._next_sequence("_racket_sequence"),
                    "base_sequence_ref": self._wire_counters.base_sequence,
                    "source_monotonic_s": ball_source_monotonic_s,
                }
            if self._racket_flat_schema == RACKET_FLAT_SCHEMA_V4_FACE179_TASK:
                formal_wire.update(
                    task_id=(0 if task_revision is None else task_revision.task_id),
                    task_revision=(
                        0
                        if task_revision is None
                        else task_revision.task_revision
                    ),
                )
            try:
                if (
                    self._racket_flat_schema == RACKET_FLAT_SCHEMA_V4_FACE179_TASK
                    and task_revision is None
                ):
                    # The producer has not yet observed the explicit new-ball
                    # rearm barrier. Do not leak a plausible target under an
                    # anonymous identity while waiting for that proof.
                    flat_data = pack_invalid_racket_command_flat(
                        schema=self._racket_flat_schema, **formal_wire
                    )
                else:
                    flat_data, flat_contract_error = pack_racket_command_flat_fail_closed(
                        schema=self._racket_flat_schema,
                        valid=wire_command_valid,
                        swing_sign=swing_sign,
                        position_w=(
                            float(cmd.p_intercept[0]),
                            float(cmd.p_intercept[1]),
                            float(cmd.p_intercept[2]) + self._policy_z_offset,
                        ),
                        velocity_w=cmd.v_racket,
                        time_to_strike=(
                            float(self._last_tts)
                            if self._last_tts == self._last_tts
                            else 0.0
                        ),
                        strike_time=float(cmd.t_strike),
                        frame_code=0,
                        # StrikeSpec exposes the physical opponent-facing contact face B. The C++
                        # 179 runner converts B->raw mount +Y/A with the selected clip sign; do not
                        # pre-flip here and never flip position/velocity.
                        normal_cmd_w=cmd.n_racket,
                        rho=0.0,
                        **formal_wire,
                    )
            except (IndexError, TypeError, ValueError, OverflowError) as exc:
                if self._racket_flat_schema not in (2, 3, 4):
                    raise
                flat_data = pack_invalid_racket_command_flat(
                    schema=self._racket_flat_schema,
                    **formal_wire,
                )
                flat_contract_error = str(exc)
            if flat_contract_error is not None:
                self._last_valid = False
                self._n_flat_contract_rejected += 1
                self.get_logger().error(
                    "formal racket command rejected; published valid=0 revocation instead: "
                    f"{flat_contract_error}",
                    throttle_duration_sec=2.0,
                )

        # The flat wire is the formal Gate-3 control path. Publish it before
        # touching the optional custom-message mirror: a conversion or DDS
        # exception in hope_msgs must never suppress a fresh flat command or
        # revocation and leave the previous face tuple alive.
        if self.flat_cmd_pub is not None:
            fm = Float64MultiArray()
            fm.data = flat_data
            self.flat_cmd_pub.publish(fm)

        if self.cmd_pub is not None:
            try:
                out = RacketCommand()
                out.header = msg.header
                out.header.frame_id = "world"
                if flat_contract_error is not None or (
                    self._racket_flat_schema == RACKET_FLAT_SCHEMA_V4_FACE179_TASK
                    and task_revision is None
                ):
                    # Canonical invalid mirror: finite zeros plus opponent-facing
                    # +X normal. Consumers see the same formal-flat revocation.
                    out.normal.x = 1.0
                    out.valid = False
                else:
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
                    out.valid = wire_command_valid
                    out.clears_net = bool(cmd.clears_net)
                    out.bypasses_net_posts = bool(cmd.bypasses_net_posts)
                    out.predicted_bounces = int(cmd.num_bounces)
                self.cmd_pub.publish(out)
            except Exception as exc:  # optional mirror; formal flat already published above
                self._n_custom_mirror_rejected += 1
                self.get_logger().error(
                    "optional RacketCommand mirror failed after formal flat publication: "
                    f"{type(exc).__name__}: {exc}",
                    throttle_duration_sec=2.0,
                )

        # `valid_commands` is the control-visible count, not merely a successful
        # planner solve. A formal payload rejected into valid=0 must never be
        # reported as accepted while the C++ subscriber correctly revokes it.
        if self._last_valid:
            self._n_valid += 1
            self._last_intercept_y = float(cmd.p_intercept[1])  # accepted per-side aim memory

        # Strike-spec solve AFTER the command publish so the solve latency
        # never delays the command itself. Legacy: 1 Hz throttle (scalar LM,
        # ~0.45 s). Fast (use_fast_strike_spec): replan every _spec_period
        # (default 40 Hz), warm-started from the previous solve, cached spec
        # served between solves, log throttled to 1 Hz.
        if self._spec_planner is not None and self._last_valid and t >= self._spec_next_t:
            self._spec_next_t = t + (self._spec_period if self._use_fast_spec else 1.0)
            strike = self.planner.strike_target
            if strike is not None and strike.valid:
                # Legacy command path is spin-blind -> omega None (zeros);
                # promote to the EKF/spin estimate when that path lands.
                if self._use_fast_spec:
                    # SHADOW snapshot BEFORE the production solve: q0/max_iter must
                    # capture the warm state the production solve consumes THIS tick
                    # (the solve below overwrites _spec_warm_q). Copies, so neither
                    # the backend nor the CSV can alias live planner state. Any
                    # snapshot failure disables this tick's shadow only.
                    # .active: once the harness wall-budget self-disable trips,
                    # skip even the snapshot/timing cost.
                    shadow_problem = None
                    if self._shadow is not None and self._shadow.active:
                        try:
                            shadow_problem = self._shadow_problem_cls(
                                t=t,
                                p_ball=np.array(strike.p_ball, dtype=float),
                                v_ball=np.array(strike.v_ball, dtype=float),
                                omega_ball=None,
                                target_land_xy=np.array(
                                    self.planner.config.target_land[:2], dtype=float),
                                racket_speed_budget=self._racket_speed_budget,
                                max_iter=(self._spec_max_iter
                                          if self._spec_warm_q is not None else None),
                                q0=(None if self._spec_warm_q is None
                                    else self._spec_warm_q.copy()),
                                with_sensitivities=self._spec_sens,
                                # explicit on purpose: the production call below
                                # does not override tol_m (solver default TOL_M);
                                # if it ever does, this snapshot must mirror it.
                                tol_m=None,
                            )
                            t_solve0 = time.perf_counter()
                        except Exception as exc:
                            self._n_shadow_node_errors += 1
                            shadow_problem = None
                            self.get_logger().warning(
                                f"shadow snapshot failed ({type(exc).__name__}: {exc}); "
                                "production solve unaffected",
                                throttle_duration_sec=5.0)
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
                    # SHADOW double-run AFTER the production solve (its latency never
                    # delays the cached-spec serving, same reasoning as the command
                    # publish above). ShadowHarness.run() already never raises; the
                    # belt-and-braces except is for harness-construction-level bugs —
                    # a shadow can only cost us a counter, never the command path.
                    if shadow_problem is not None:
                        prod_wall = time.perf_counter() - t_solve0
                        try:
                            self._shadow.run(shadow_problem, self._last_spec, prod_wall)
                        except Exception as exc:
                            self._n_shadow_node_errors += 1
                            self.get_logger().warning(
                                f"shadow run failed ({type(exc).__name__}: {exc}); "
                                "production solve unaffected",
                                throttle_duration_sec=5.0)
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
        # The timer expires the control lease but never manufactures READY from
        # an older cached sample. READY is sample-driven in _robot_pose_cb.
        self._expire_formal_base_if_needed(time.monotonic())
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
            KeyValue(key="planner_valid_commands", value=str(self._n_planner_valid)),
            KeyValue(key="valid_commands", value=str(self._n_valid)),
            KeyValue(
                key="flat_contract_rejected",
                value=str(self._n_flat_contract_rejected),
            ),
            KeyValue(
                key="custom_mirror_rejected",
                value=str(self._n_custom_mirror_rejected),
            ),
            KeyValue(key="last_valid", value=str(self._last_valid)),
            KeyValue(key="time_to_strike_s", value=f"{self._last_tts:.4f}"),
            KeyValue(
                key="runner_effective_time_to_strike_s",
                value=f"{self._last_effective_tts:.4f}",
            ),
            KeyValue(
                key="ball_source_age_s",
                value=f"{self._last_ball_source_age_s:.4f}",
            ),
            KeyValue(
                key="ball_to_strike_plane_x_m",
                value=f"{self._last_ball_plane_distance_m:.4f}",
            ),
        ]
        if self._task_lifecycle is not None:
            status.values += [
                KeyValue(
                    key="formal_task_state",
                    value=self._task_lifecycle.state.value,
                ),
                KeyValue(
                    key="formal_task_id",
                    value=str(self._task_lifecycle.active_task_id or 0),
                ),
                KeyValue(
                    key="formal_task_revision",
                    value=str(self._task_lifecycle.active_revision),
                ),
            ]
        if self._kf is not None:
            status.values += [
                KeyValue(key="kf_pos_delta_m", value=f"{self._kf_pos_delta:.4f}"),
                KeyValue(key="kf_vel_delta_mps", value=f"{self._kf_vel_delta:.4f}"),
                KeyValue(key="kf_rejected", value=str(self._kf.rejected_count)),
            ]
        if self._shadow is not None:
            # One numeric metric per key (the kf_* / spec_* convention) so
            # diagnostic_aggregator / plotjuggler can threshold and trend the
            # counters; the per-field record lives in the shadow_log_path CSV.
            d = self._shadow.last_diffs or {}
            d_land = d.get("d_landing_xy_l2_m", float("nan"))
            backend_v = self._shadow.backend.name
            if not self._shadow.active:
                backend_v += f":DISABLED({self._shadow.disabled_reason})"
            status.values += [
                KeyValue(key="shadow_backend", value=backend_v),
                KeyValue(key="shadow_runs", value=str(self._shadow.n_runs)),
                KeyValue(key="shadow_exc",
                         value=str(self._shadow.n_shadow_exceptions)),
                KeyValue(key="shadow_rec_fail",
                         value=str(self._shadow.n_record_failures)),
                KeyValue(key="shadow_over_budget",
                         value=str(self._shadow.n_over_budget)),
                KeyValue(key="shadow_node_exc",
                         value=str(self._n_shadow_node_errors)),
                KeyValue(key="shadow_last_d_land_m", value=f"{d_land:.3e}"),
            ]
        elif self._shadow_disabled_reason is not None:
            # use_shadow_solver=True was requested but nothing is being
            # reconciled — keep saying so on every cycle (fail-loud, so an
            # empty CSV can never be mistaken for an all-zero-diff session).
            status.values.append(KeyValue(
                key="shadow_backend",
                value=f"DISABLED({self._shadow_disabled_reason})"))
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
