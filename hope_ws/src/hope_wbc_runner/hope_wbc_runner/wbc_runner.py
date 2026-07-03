"""ROS 2 node: WBC runner for model_15200 (staged, safety-gated).

Subscribes ``/racket/command`` (hope_msgs/RacketCommand, e.g. from planner_imitate),
builds the exact 180-D observation, runs the exported model_15200 ONNX
DETERMINISTICALLY (mean action, no dither), decodes joint position targets, logs
them, and - ONLY in hardware mode with the enable+estop gate satisfied -
publishes low-level joint commands (joint_msgs/JointCommand: position + per-joint
kp/kd, the position+PD interface the A3 backend expects).

Three modes (param ``mode``):
  A. dry_run  (default) - log only; NEVER publishes. State = synthetic (perfect
                          tracking of the reference) so it runs standalone with
                          just planner_imitate. This is Milestone 1.
  B. shadow             - subscribe REAL robot/sim state, build obs, predict, log.
                          NEVER publishes (robot is driven by something else).
  C. hardware           - publish joint targets, but ONLY when hardware_enable is
                          true AND /hope/estop is false. Defense in depth.

Publish gate (all must hold): mode == "hardware"  AND  hardware_enable  AND  not estop.
dry_run and shadow can NEVER publish, regardless of the other flags.

Deterministic by design: dither_scale defaults to 0, so learned_std.npy is not
needed (MuJoCo showed dither is unnecessary for stability and hurts the backhand).
"""

import csv
import math
import os

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from hope_msgs.msg import RacketCommand

from .frame_math import projected_gravity_body, quat_rotate, yaw_quat
from .obs_builder import (
    N_JOINTS,
    OBS_DIM_175,
    RacketTarget,
    RobotState,
    build_obs,
    build_obs_175,
    synthetic_state_from_refs,
)
from .onnx_policy import OnnxPolicy
from .reference_clock import (
    ClipLayout,
    clip_id_from_swing_sign,
    clip_phase,
    swing_sign_from_target_y,
    time_step_for,
)
from .world_frame import (
    ImuYawAligner,
    OriginCapture,
    TableToPolicy,
    TargetGate,
    base_relative_target,
)

_MODES = ("dry_run", "shadow", "hardware")


class WbcRunnerNode(Node):
    def __init__(self):
        super().__init__("wbc_runner")

        # --- params ---
        self.declare_parameter("mode", "dry_run")          # dry_run | shadow | hardware
        self.declare_parameter("hardware_enable", False)   # must be true to publish in hardware mode
        self.declare_parameter("onnx_path", "")            # REQUIRED: exported policy.onnx
        self.declare_parameter("std_path", "")             # optional; only for dither_scale>0
        self.declare_parameter("dither_scale", 0.0)        # 0 = deterministic (default, recommended)
        self.declare_parameter("control_rate_hz", 50.0)    # must match training (50 Hz)
        self.declare_parameter("seg_len", [95, 105])       # (forehand, backhand) clip frame counts
        self.declare_parameter("strike_phase", [0.36, 0.50])  # (forehand, backhand) contact phase
        self.declare_parameter("command_timeout_s", 0.5)   # stale /racket/command -> idle (stand)
        self.declare_parameter("base_target_xy", [0.0, 0.0])  # desired base XY (world)
        # state source
        self.declare_parameter("state_source", "synthetic")   # synthetic | ros
        self.declare_parameter("joint_state_topic", "/a3/joint_states")
        self.declare_parameter("imu_topic", "/a3/imu")
        self.declare_parameter("nominal_base_height", 0.95)    # fallback pelvis z when no localisation
        self.declare_parameter("torso_offset_z", 0.20)         # fallback torso-above-pelvis z
        self.declare_parameter("base_pose_topic", "")          # optional geometry_msgs/PoseStamped pelvis pose (TABLE frame)
        self.declare_parameter("joint_state_type", "sensor_msgs")  # "sensor_msgs" (std) | "joint_msgs" (a3 sim)
        # --- sim2real frame alignment: HOPE table frame -> policy frame (world_frame.py) ---
        # Applied to /racket/command with frame_id=="world" AND to base_pose_topic (both are
        # published by the mocap chain in the HOPE TABLE frame: origin table corner, z=0 at
        # the table surface). frame_id=="base_link" (planner_imitate) passes through unchanged.
        self.declare_parameter("robot_start_xy_table", [-0.5, -0.7625])  # boot base_link ground XY (table frame); MEASURE/confirm
        self.declare_parameter("robot_start_yaw_table", 0.0)   # boot heading (rad, table frame); 0 = facing +x/opponent — PLACEMENT CONVENTION
        self.declare_parameter("table_floor_z", -0.76)         # floor z in the table frame (surface z=0)
        self.declare_parameter("origin_autocapture", True)     # refine origin XY from the first N base-pose samples
        self.declare_parameter("origin_capture_samples", 60)
        self.declare_parameter("marker_to_base_xyz", [0.0, 0.0, 0.0])  # P1 marker-cluster -> base_link, BASE frame (MEASURE; see hope_world_frame.yaml)
        self.declare_parameter("use_mocap_orientation", False)  # mocap is position-only -> keep IMU orientation (default)
        self.declare_parameter("imu_yaw_align", True)           # re-zero drifting IMU yaw at boot (robot standing still)
        self.declare_parameter("imu_yaw_align_samples", 100)
        self.declare_parameter("base_pose_timeout_s", 0.5)      # stale mocap base pose -> nominal fallback + warn
        # --- target reachability gate (base-relative; reject OOD targets -> stand) ---
        self.declare_parameter("target_gate_enable", True)
        self.declare_parameter("target_gate_x", [0.20, 0.90])
        self.declare_parameter("target_gate_y_abs", 0.85)
        self.declare_parameter("target_gate_z", [0.55, 1.40])
        self.declare_parameter("target_gate_speed_max", 3.5)
        # --- swing-completion latch: the planner stops publishing once the ball is hit ---
        # (v_x flips sign), which would otherwise revert to stand mid-follow-through after
        # command_timeout_s. If the command went stale with the strike imminent/past
        # (tts <= latch_commit_tts_s), finish the swing through the clip end instead.
        self.declare_parameter("latch_swing_completion", True)
        self.declare_parameter("latch_commit_tts_s", 0.35)
        # output / safety topics
        self.declare_parameter("joint_command_topic", "/a3/joint_command")
        self.declare_parameter("estop_topic", "/hope/estop")
        self.declare_parameter("enable_topic", "/hope/hardware_enable")
        # logging
        self.declare_parameter("csv_path", "")

        gp = self.get_parameter
        self._mode = str(gp("mode").value)
        if self._mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got '{self._mode}'")
        onnx_path = str(gp("onnx_path").value).strip()
        if not onnx_path:
            raise ValueError("onnx_path is required (path to the exported model_15200 policy.onnx)")
        self._dither = float(gp("dither_scale").value)
        std_path = str(gp("std_path").value).strip() or None
        self.policy = OnnxPolicy(onnx_path, std_path=std_path if self._dither > 0 else None)

        rate = float(gp("control_rate_hz").value)
        self._dt = 1.0 / max(rate, 1e-3)
        # Clip layout: the ONNX metadata (clip_seg_lengths / clip_strike_phases, baked at
        # export) is the truth; the yaml values are only a fallback for older exports.
        # Trusting a stale yaml against a newer model mis-times every strike (the exact
        # bug that drove v2 clips at v1 frame indices in the C++ runner).
        seg_len_cfg = tuple(int(v) for v in gp("seg_len").value)
        phase_cfg = tuple(float(v) for v in gp("strike_phase").value)
        seg_len = self.policy.clip_seg_lengths or seg_len_cfg
        phases = self.policy.clip_strike_phases or phase_cfg
        if self.policy.clip_seg_lengths and (seg_len != seg_len_cfg or phases != phase_cfg):
            self.get_logger().warn(
                f"clip layout from ONNX metadata seg_len={seg_len} strike_phase={phases} "
                f"OVERRIDES yaml seg_len={seg_len_cfg} strike_phase={phase_cfg} (yaml is stale).")
        elif not self.policy.clip_seg_lengths:
            self.get_logger().warn(
                "ONNX has no clip_seg_lengths/clip_strike_phases metadata (old export) — "
                f"trusting yaml seg_len={seg_len} strike_phase={phases}. Verify they match this model!")
        self._layout = ClipLayout(seg_len=seg_len, strike_phase=phases, step_dt=self._dt)
        self._timeout = float(gp("command_timeout_s").value)
        self._base_target_xy = np.array([float(v) for v in gp("base_target_xy").value])
        self._state_source = str(gp("state_source").value)
        self._hardware_enable = bool(gp("hardware_enable").value)
        self._estop = False

        # --- sim2real frame alignment (table -> policy frame) ---
        self._t2p = TableToPolicy(
            origin_xy_table=np.array([float(v) for v in gp("robot_start_xy_table").value]),
            yaw_table=float(gp("robot_start_yaw_table").value),
            floor_z_table=float(gp("table_floor_z").value))
        self._origin_capture = (OriginCapture(int(gp("origin_capture_samples").value))
                                if bool(gp("origin_autocapture").value) else None)
        self._marker_to_base = np.array([float(v) for v in gp("marker_to_base_xyz").value])
        self._use_mocap_ori = bool(gp("use_mocap_orientation").value)
        self._yaw_aligner = (ImuYawAligner(int(gp("imu_yaw_align_samples").value))
                             if bool(gp("imu_yaw_align").value) else None)
        self._base_pose_timeout = float(gp("base_pose_timeout_s").value)
        self._gate = TargetGate(
            x_range=tuple(float(v) for v in gp("target_gate_x").value),
            y_abs_max=float(gp("target_gate_y_abs").value),
            z_range=tuple(float(v) for v in gp("target_gate_z").value),
            speed_max=float(gp("target_gate_speed_max").value),
            enabled=bool(gp("target_gate_enable").value))
        self._latch = bool(gp("latch_swing_completion").value)
        self._latch_commit_tts = float(gp("latch_commit_tts_s").value)
        self._warned_capture_pending = False

        # state
        self._last_action = np.zeros(N_JOINTS)
        self._last_cmd = None              # latest VALID RacketCommand (invalid ones don't overwrite)
        self._last_cmd_t = None            # wall time it arrived
        self._last_invalid_t = None        # wall time of the latest valid=False message
        self._cmd_warned_frame = False
        self._n_ticks = 0
        self._n_published = 0

        # --- pubs / subs ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(RacketCommand, "/racket/command", self._cmd_cb, qos)
        self.create_subscription(Bool, str(gp("estop_topic").value), self._estop_cb, 1)
        self.create_subscription(Bool, str(gp("enable_topic").value), self._enable_cb, 1)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/wbc_runner/diagnostics", 1)

        # joint command publisher + state subs are created lazily (only when needed) so dry-run
        # has NO hard dependency on joint_msgs being built.
        self._cmd_pub = None
        self._joint_state = None     # (q, qd) latest from ros, Isaac order
        self._imu = None             # (quat_w, ang_vel_b) latest (IMU frame, yaw unreferenced)
        self._base_pose_raw = None   # (pos_TABLE, quat_TABLE) latest from base_pose_topic
        self._base_pose_t = None     # wall time the base pose arrived (staleness check)
        self._joint_coverage_logged = False
        self._obs_verified = False
        self._joint_matched = -1     # set from the real joint-state names (-1 = unknown / synthetic)
        self._joint_missing = -1
        if self._mode == "hardware":
            self._cmd_pub = self._make_joint_command_pub(str(gp("joint_command_topic").value), qos)
        if self._mode == "shadow" and self._state_source != "ros":
            self.get_logger().warn(
                "mode=shadow but state_source!=ros: predicting on SYNTHETIC state, NOT the real robot. "
                "Pass state_source:=ros (+ joint_state_topic / imu_topic) to shadow the real A3.")
        if self._state_source == "ros":
            self._subscribe_robot_state(
                str(gp("joint_state_topic").value), str(gp("imu_topic").value),
                str(gp("base_pose_topic").value), str(gp("joint_state_type").value), qos)

        # CSV
        self._csv_file = self._csv = None
        csv_path = str(gp("csv_path").value).strip()
        if csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(
                ["t", "mode", "published", "swing_type", "clip_id", "time_step", "clip_phase",
                 "time_to_strike", "valid", "obs_norm", "action_norm", "target_q_max_abs",
                 # --- obs-verify columns (shadow bring-up: inspect IMU frame / joint mapping over time) ---
                 "proj_grav_x", "proj_grav_y", "proj_grav_z",
                 "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
                 "joint_order_matched_count", "missing_joint_count", "base_pose_source",
                 # --- sim2real frame columns (policy frame) ---
                 "latched", "base_x", "base_y", "base_z", "tgt_x", "tgt_y", "tgt_z"]
                + [f"target_q_{i}" for i in range(N_JOINTS)])

        self._t0 = self._now()
        self.create_timer(self._dt, self._tick)

        self.get_logger().info(
            f"wbc_runner started | mode={self._mode} | state_source={self._state_source} | "
            f"rate={rate:.0f} Hz | dither={self._dither} (deterministic={self._dither == 0.0}) | "
            f"obs={self.policy.obs_dim}-D ({'deploy_parity/racket-FK' if self.policy.obs_dim == OBS_DIM_175 else 'full'}) | "
            f"seg_len={self._layout.seg_len} strike_phase={self._layout.strike_phase}")
        self.get_logger().info(
            "[frames] table->policy: origin_xy_table="
            f"{np.round(self._t2p.origin_xy_table, 3).tolist()} yaw={self._t2p.yaw_table:.3f} rad "
            f"({self._t2p.origin_source}, autocapture={'on' if self._origin_capture else 'off'}) | "
            f"imu_yaw_align={'on' if self._yaw_aligner else 'OFF'} | "
            f"mocap_orientation={'TRUSTED' if self._use_mocap_ori else 'ignored (position-only)'} | "
            f"target_gate={'on' if self._gate.enabled else 'OFF'} "
            f"x={self._gate.x_range} |y|<={self._gate.y_abs_max} z={self._gate.z_range} | "
            f"swing_latch={'on' if self._latch else 'off'} (commit tts<={self._latch_commit_tts:.2f}s)")
        self.get_logger().warn(
            "SAFETY: publishes joint commands ONLY when mode=hardware AND hardware_enable AND "
            f"!estop. Now: mode={self._mode}, hardware_enable={self._hardware_enable}, estop={self._estop}. "
            f"{'WILL PUBLISH when enabled' if self._mode == 'hardware' else 'WILL NOT PUBLISH (log only)'}.")

    # ----- time -----
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ----- callbacks -----
    def _cmd_cb(self, msg: RacketCommand):
        # Only VALID commands become the active swing command. valid=False (the real
        # planner emits it as soon as the ball passes the hit plane / prediction is
        # lost) must NOT clobber the in-progress swing's target — it is recorded as a
        # timestamp so _tick can decide between aborting (wind-up) and finishing the
        # swing (post-commit latch).
        if msg.valid:
            self._last_cmd = msg
            self._last_cmd_t = self._now()
        else:
            self._last_invalid_t = self._now()
        if msg.header.frame_id not in ("world", "base_link", "") and not self._cmd_warned_frame:
            self._cmd_warned_frame = True
            self.get_logger().warn(
                f"/racket/command frame_id='{msg.header.frame_id}': unknown frame — treating as "
                "'world' (HOPE table frame). Expected 'world' (real hope_planner; converted to the "
                "policy frame via robot_start_*_table) or 'base_link' (planner_imitate; passed "
                "through as policy-frame).")

    def _estop_cb(self, msg: Bool):
        was = self._estop
        self._estop = bool(msg.data)
        if self._estop and not was:
            self.get_logger().warn("ESTOP engaged -> publishing disabled, commanding STAND.")

    def _enable_cb(self, msg: Bool):
        self._hardware_enable = bool(msg.data)
        self.get_logger().warn(f"hardware_enable -> {self._hardware_enable}")

    # ----- main loop -----
    def _tick(self):
        self._n_ticks += 1
        t = self._now() - self._t0
        cmd = self._last_cmd

        # hard stand conditions: estop, or never received a valid command
        if self._estop or cmd is None:
            self._stand(t, "estop" if self._estop else "no_command")
            return

        # tts decays since the command arrived (planner publishes the value at send time)
        age = self._now() - self._last_cmd_t
        tts = float(cmd.time_to_strike) - age
        stale = age > self._timeout
        invalid_now = self._last_invalid_t is not None and self._last_invalid_t > self._last_cmd_t
        latched = False
        if stale or invalid_now:
            # Planner went quiet or flipped to valid=False. The REAL planner does BOTH the
            # moment the ball passes the hit plane, which is EXPECTED mid-swing — aborting
            # to stand there would cut the follow-through. If the strike was imminent/past
            # when it happened (tts <= commit threshold), finish the swing; if it happened
            # during the wind-up (mocap/planner died, prediction lost), abort to stand.
            if not (self._latch and tts <= self._latch_commit_tts):
                self._stand(t, "planner_invalid" if invalid_now else "stale")
                return
            latched = True

        # racket target -> POLICY frame (world_frame.py; table-frame cmds transformed)
        tgt_pos_w, tgt_vel_w = self._target_to_policy(cmd)

        # swing side + reachability from the BASE-RELATIVE target. NOTE: the raw world-Y
        # sign convention only works for base_link targets; in the TABLE frame the whole
        # table is y<0 and raw Y would ALWAYS pick forehand.
        base_pos, base_quat, _src = self._localized_base()
        tgt_b = base_relative_target(tgt_pos_w, base_pos, base_quat)
        # gate on [x_rel, y_rel, z_above_floor]: training boxes are env-frame z
        ok, why = self._gate.check(np.array([tgt_b[0], tgt_b[1], tgt_pos_w[2]]), tgt_vel_w)
        if not ok:
            self.get_logger().warn(
                f"[target gate] racket target base-rel ({tgt_b[0]:+.2f},{tgt_b[1]:+.2f},{tgt_b[2]:+.2f}) "
                f"rejected ({why}) -> stand. Out-of-distribution targets cause deploy falls.",
                throttle_duration_sec=2.0)
            self._stand(t, "target_gate", base_pos=base_pos, tgt_w=tgt_pos_w)
            return
        swing_sign = swing_sign_from_target_y(float(tgt_b[1]))
        clip_id = clip_id_from_swing_sign(swing_sign)
        time_step = time_step_for(self._layout, clip_id, tts)
        cphase = clip_phase(self._layout, clip_id, time_step)

        if latched:
            # stand once the latched swing's follow-through is complete (clip end reached)
            follow_through_s = (self._layout.seg_start[clip_id] + self._layout.seg_len[clip_id] - 1
                                - self._layout.strike_frame(clip_id)) * self._layout.step_dt
            if tts < -follow_through_s:
                self._stand(t, "swing_complete")
                return

        refs = self.policy.refs(time_step)
        state = self._get_state(refs)
        target = RacketTarget(pos_w=tgt_pos_w, vel_w=tgt_vel_w,
                              swing_sign=swing_sign, time_to_strike=tts,
                              base_target_xy=self._base_target_xy)

        # obs contract auto-selected from the loaded ONNX (mirrors the C++ runner):
        # 175 = deploy_parity (racket-FK-relative target, no world-base terms);
        # 180 = full (model_15200 era).
        builder = build_obs_175 if self.policy.obs_dim == OBS_DIM_175 else build_obs
        obs = builder(refs, state, target, self._last_action, self.policy.default_q)
        self._verify_obs_once(state)
        action = self.policy.action(obs, time_step, dither_scale=self._dither)
        self._last_action = action
        target_q = self.policy.target_q(action)

        self._emit(t, target_q, swing=("forehand" if swing_sign > 0 else "backhand"),
                   clip_id=clip_id, time_step=time_step, cphase=cphase, tts=tts, valid=True,
                   obs_norm=float(np.linalg.norm(obs)), action_norm=float(np.linalg.norm(action)),
                   idle_reason="", latched=latched, base_pos=state.base_pos_w, tgt_w=tgt_pos_w)

    def _stand(self, t, reason, base_pos=None, tgt_w=None):
        target_q = self.policy.default_q.copy()      # hold nominal stand pose
        self._last_action = np.zeros(N_JOINTS)
        self._emit(t, target_q, swing="stand", clip_id=-1, time_step=-1, cphase=float("nan"),
                   tts=float("nan"), valid=False, obs_norm=float("nan"), action_norm=0.0,
                   idle_reason=reason, base_pos=base_pos, tgt_w=tgt_w)

    # ----- sim2real frame plumbing -----
    def _target_to_policy(self, cmd):
        """RacketCommand pos/vel -> policy frame.

        frame_id 'base_link' (planner_imitate) is already policy-frame robot-relative and
        passes through; 'world' / anything else is the HOPE TABLE frame and is transformed
        via TableToPolicy (origin = robot boot pose, from config, refined by mocap capture).
        """
        pos = np.array([cmd.position.x, cmd.position.y, cmd.position.z])
        vel = np.array([cmd.velocity.x, cmd.velocity.y, cmd.velocity.z])
        if cmd.header.frame_id == "base_link":
            return pos, vel
        if (self._origin_capture is not None and not self._origin_capture.done
                and not self._warned_capture_pending):
            self._warned_capture_pending = True
            self.get_logger().warn(
                "world-frame racket target arrived BEFORE the mocap origin capture finished — "
                "falling back to the configured robot_start_xy_table as the policy origin. "
                f"Check that base_pose_topic ('{self.get_parameter('base_pose_topic').value}') is alive "
                "and the robot stood still at startup.")
        return self._t2p.pos(pos), self._t2p.vec(vel)

    def _fresh_base_pose_raw(self):
        """Latest (pos_TABLE, quat_TABLE) from mocap, or None if never received / stale."""
        if self._base_pose_raw is None:
            return None
        if self._now() - self._base_pose_t > self._base_pose_timeout:
            self.get_logger().warn(
                f"mocap base pose STALE (>{self._base_pose_timeout:.1f}s) -> nominal-origin fallback; "
                "world-frame targets + the anchor obs degrade. Check the mocap relay / P1 rigid body.",
                throttle_duration_sec=5.0)
            return None
        return self._base_pose_raw

    def _localized_base(self):
        """Best (base_pos_w, base_quat_w, source) estimate in the POLICY frame.

        Orientation: yaw-aligned IMU (mocap is position-only; its quat is identity/garbage)
        unless use_mocap_orientation. Position: mocap marker -> TableToPolicy -> +marker
        offset; nominal upright at the origin when no mocap. Shared by the obs state (ros
        mode), the swing-side pick, and the target gate so they can never disagree.
        """
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self._imu is not None:
            imu_quat, _ = self._imu
            quat = self._yaw_aligner.correct(imu_quat) if self._yaw_aligner is not None else imu_quat
        pos = np.array([0.0, 0.0, float(self.get_parameter("nominal_base_height").value)])
        src = "nominal"
        raw = self._fresh_base_pose_raw()
        if raw is not None:
            pos_table, quat_table = raw
            if self._use_mocap_ori:
                quat = self._t2p.quat(quat_table)
            pos = self._t2p.pos(pos_table)
            if np.any(self._marker_to_base):
                pos = pos + quat_rotate(yaw_quat(quat), self._marker_to_base)
            src = "mocap"
        return pos, quat, src

    # ----- state assembly -----
    def _get_state(self, refs) -> RobotState:
        if self._state_source == "synthetic":
            return synthetic_state_from_refs(refs, self.policy.default_q)
        # ros: joints + IMU from real/sim; base/torso WORLD pose is the sim2real localisation gap.
        if self._joint_state is None or self._imu is None:
            self.get_logger().warn(
                "state_source='ros' but joint_state/imu not received yet; falling back to synthetic.",
                once=True)
            return synthetic_state_from_refs(refs, self.policy.default_q)
        q, qd = self._joint_state
        _imu_quat, ang_vel_b = self._imu
        # policy-frame localisation: mocap position (table->policy) + yaw-aligned IMU
        # orientation (position-only mocap can NOT provide a usable quaternion).
        base_pos_w, base_quat_w, _src = self._localized_base()
        torso_pos_w = base_pos_w + np.array([0.0, 0.0, float(self.get_parameter("torso_offset_z").value)])
        return RobotState(base_pos_w=base_pos_w, base_quat_w=base_quat_w,
                          torso_pos_w=torso_pos_w, torso_quat_w=base_quat_w,
                          base_ang_vel_b=ang_vel_b, q=q, qd=qd)

    def _verify_obs_once(self, state):
        """One-time sanity print of the obs frames built from REAL state (shadow bring-up check)."""
        # only meaningful once real IMU is flowing (synthetic state has no IMU to check)
        if self._obs_verified or self._state_source != "ros" or self._imu is None:
            return
        self._obs_verified = True
        pg = projected_gravity_body(state.base_quat_w)
        yaw_info = "disabled"
        if self._yaw_aligner is not None:
            yaw_info = (f"offset {math.degrees(self._yaw_aligner.offset):+.1f} deg"
                        if self._yaw_aligner.done else "CAPTURING (robot must stand still)")
        self.get_logger().info(
            "[obs verify] first real-state observation built. Sanity checks:\n"
            f"  base_quat_w (w,x,y,z) = {np.round(state.base_quat_w, 3).tolist()}\n"
            f"  projected_gravity (body) = {np.round(pg, 3).tolist()}  "
            f"(upright should be ~[0,0,-1]; if it's far off, the IMU orientation frame is wrong)\n"
            f"  base_ang_vel (body gyro) = {np.round(state.base_ang_vel_b, 3).tolist()}  "
            f"(should be ~0 when standing still)\n"
            f"  base_pos_w (POLICY frame) = {np.round(state.base_pos_w, 3).tolist()}  "
            f"({'mocap via table->policy' if self._base_pose_raw is not None else 'NOMINAL fallback'}; "
            f"expect ~[0, 0, pelvis_height] with the robot at its start pose)\n"
            f"  policy origin (table frame) = {np.round(self._t2p.origin_xy_table, 3).tolist()} "
            f"yaw={self._t2p.yaw_table:.3f} rad ({self._t2p.origin_source})\n"
            f"  imu yaw-align: {yaw_info}\n"
            f"  joint q[:3] = {np.round(state.q[:3], 3).tolist()} (Isaac order; first 3 = "
            f"{self.policy.joint_names[:3]})")

    def _verify_values(self):
        """Per-row obs-verify values for the CSV, from the LATEST real state:
        (proj_grav[3], base_ang_vel[3], joint_matched, joint_missing, base_pose_source).
        nan / 'synthetic' until real IMU/joints are flowing (state_source=ros)."""
        if self._state_source != "ros" or self._imu is None:
            nan3 = (float("nan"), float("nan"), float("nan"))
            src = "synthetic" if self._state_source != "ros" else "no_imu_yet"
            return nan3, nan3, self._joint_matched, self._joint_missing, src
        _quat, ang = self._imu
        _pos, quat, src = self._localized_base()   # match _get_state exactly
        pg = projected_gravity_body(quat)
        return tuple(pg), tuple(ang), self._joint_matched, self._joint_missing, src

    # ----- output + logging -----
    def _emit(self, t, target_q, swing, clip_id, time_step, cphase, tts, valid,
              obs_norm, action_norm, idle_reason, latched=False, base_pos=None, tgt_w=None):
        publish = (self._mode == "hardware") and self._hardware_enable and (not self._estop)
        if publish and self._cmd_pub is not None:
            self._cmd_pub_publish(target_q)
            self._n_published += 1

        # console: one line per ~0.5 s (avoid 50 Hz spam)
        if self._n_ticks % max(int(0.5 / self._dt), 1) == 1:
            tag = "PUBLISH" if publish else f"log-only/{self._mode}"
            extra = f" idle({idle_reason})" if idle_reason else (" LATCHED" if latched else "")
            self.get_logger().info(
                f"[{tag}] {swing}{extra} ts={time_step} phase={cphase:.2f} tts={tts:+.2f} "
                f"|act|={action_norm:.2f} max|q*|={float(np.max(np.abs(target_q))):.2f}")

        if self._csv is not None:
            pg, av, matched, missing, src = self._verify_values()
            nan3 = [float("nan")] * 3
            bp = list(base_pos) if base_pos is not None else nan3
            tw = list(tgt_w) if tgt_w is not None else nan3
            self._csv.writerow(
                [f"{t:.4f}", self._mode, int(publish), swing, clip_id, time_step,
                 f"{cphase:.4f}", f"{tts:.4f}", int(valid), f"{obs_norm:.4f}",
                 f"{action_norm:.4f}", f"{float(np.max(np.abs(target_q))):.4f}",
                 f"{pg[0]:.4f}", f"{pg[1]:.4f}", f"{pg[2]:.4f}",
                 f"{av[0]:.4f}", f"{av[1]:.4f}", f"{av[2]:.4f}",
                 matched, missing, src,
                 int(latched), f"{bp[0]:.4f}", f"{bp[1]:.4f}", f"{bp[2]:.4f}",
                 f"{tw[0]:.4f}", f"{tw[1]:.4f}", f"{tw[2]:.4f}"]
                + [f"{v:.5f}" for v in target_q])
            self._csv_file.flush()
        self._publish_diag(swing, publish, time_step, tts, idle_reason, latched)

    def _publish_diag(self, swing, publish, time_step, tts, idle_reason, latched=False):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = "wbc_runner"
        st.hardware_id = "wbc_runner"
        st.level = DiagnosticStatus.WARN if (self._estop or idle_reason) else DiagnosticStatus.OK
        st.message = f"{self._mode} {swing} {'PUBLISHING' if publish else 'log-only'}"
        yaw_align = ("disabled" if self._yaw_aligner is None
                     else f"{self._yaw_aligner.offset:+.4f}" if self._yaw_aligner.done else "capturing")
        st.values = [
            KeyValue(key="mode", value=self._mode),
            KeyValue(key="publishing", value=str(publish)),
            KeyValue(key="hardware_enable", value=str(self._hardware_enable)),
            KeyValue(key="estop", value=str(self._estop)),
            KeyValue(key="swing_type", value=swing),
            KeyValue(key="time_step", value=str(time_step)),
            KeyValue(key="time_to_strike_s", value=f"{tts:.3f}"),
            KeyValue(key="published_count", value=str(self._n_published)),
            KeyValue(key="idle_reason", value=idle_reason),
            KeyValue(key="latched", value=str(latched)),
            KeyValue(key="origin_source", value=self._t2p.origin_source),
            KeyValue(key="imu_yaw_offset_rad", value=yaw_align),
        ]
        arr.status = [st]
        self.diag_pub.publish(arr)

    # ----- joint_msgs (lazy: only hardware mode imports/uses it) -----
    def _make_joint_command_pub(self, topic, qos):
        # lazy import: only hardware mode needs joint_msgs on the path
        from joint_msgs.msg import Command, JointCommand
        self._JointCommand = JointCommand
        self._Command = Command
        self.get_logger().info(f"hardware mode: joint command publisher on '{topic}' (joint_msgs/JointCommand)")
        return self.create_publisher(JointCommand, topic, qos)

    def _cmd_pub_publish(self, target_q):
        msg = self._JointCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        joints = []
        for i, name in enumerate(self.policy.joint_names):
            c = self._Command()
            c.name = name
            c.sequence = self._n_published
            c.position = float(target_q[i])
            c.velocity = 0.0
            c.effort = 0.0
            c.stiffness = float(self.policy.kp[i])
            c.damping = float(self.policy.kd[i])
            joints.append(c)
        msg.joints = joints
        self._cmd_pub.publish(msg)

    def _subscribe_robot_state(self, joint_topic, imu_topic, base_pose_topic, joint_state_type, qos):
        from sensor_msgs.msg import Imu
        name_to_idx = {n: i for i, n in enumerate(self.policy.joint_names)}

        def fill_from_arrays(names, positions, velocities):
            # joint-order coverage: how many of the 31 policy joints are present by name.
            nameset = set(names)
            missing = [n for n in self.policy.joint_names if n not in nameset]
            self._joint_missing = len(missing)
            self._joint_matched = N_JOINTS - self._joint_missing
            q = np.array(self.policy.default_q, copy=True)
            qd = np.zeros(N_JOINTS)
            for k, n in enumerate(names):
                i = name_to_idx.get(n)
                if i is not None:
                    q[i] = positions[k]
                    qd[i] = velocities[k] if velocities is not None and k < len(velocities) else 0.0
            self._joint_state = (q, qd)
            if not self._joint_coverage_logged:          # one-time console summary
                self._joint_coverage_logged = True
                lvl = self.get_logger().info if not missing else self.get_logger().warn
                lvl(f"[joint order] matched {self._joint_matched}/{N_JOINTS} policy joints by name in '{joint_topic}'.")
                if missing:
                    self.get_logger().warn(
                        f"[joint order] {len(missing)} joints NOT found (left at default): {missing[:6]}"
                        f"{' ...' if len(missing) > 6 else ''}. The A3 joint names must match the ONNX "
                        "metadata joint_names; fix the bridge naming or add a remap before trusting shadow obs.")

        if joint_state_type == "joint_msgs":
            from joint_msgs.msg import JointState as JMState   # a3 sim: header + State[] {name,position,velocity}
            def js_cb(msg):
                fill_from_arrays([s.name for s in msg.joints],
                                 [s.position for s in msg.joints],
                                 [s.velocity for s in msg.joints])
            self.create_subscription(JMState, joint_topic, js_cb, qos)
        else:  # default: the ROS-standard sensor_msgs/JointState (parallel name/position/velocity arrays)
            from sensor_msgs.msg import JointState as SMState
            def js_cb(msg):
                fill_from_arrays(list(msg.name), list(msg.position),
                                 list(msg.velocity) if msg.velocity else None)
            self.create_subscription(SMState, joint_topic, js_cb, qos)

        def imu_cb(msg):
            quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
            if np.linalg.norm(quat) < 1e-6:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            # IMU angular_velocity is the pelvis BODY-frame gyro (matches the obs base_ang_vel term).
            ang = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            self._imu = (quat, ang)
            # boot yaw-align: the pelvis-IMU yaw is unreferenced (drifts boot-to-boot);
            # capture it while the robot stands still and re-zero (world_frame.ImuYawAligner)
            if self._yaw_aligner is not None and not self._yaw_aligner.done:
                self._yaw_aligner.push(quat)
                if self._yaw_aligner.done:
                    self.get_logger().info(
                        f"[yaw-align] IMU boot yaw captured; applying offset "
                        f"{math.degrees(self._yaw_aligner.offset):+.1f} deg (policy yaw := 0 at boot).")

        self.create_subscription(Imu, imu_topic, imu_cb, qos)
        self.get_logger().info(
            f"state_source=ros: joints '{joint_topic}' ({joint_state_type}/JointState), imu '{imu_topic}'.")

        if base_pose_topic:
            from geometry_msgs.msg import PoseStamped
            def bp_cb(msg):
                p = msg.pose.position
                o = msg.pose.orientation
                pos_table = np.array([p.x, p.y, p.z])
                quat_table = np.array([o.w, o.x, o.y, o.z])
                if np.linalg.norm(quat_table) < 1e-6:
                    quat_table = np.array([1.0, 0.0, 0.0, 0.0])
                self._base_pose_raw = (pos_table, quat_table)
                self._base_pose_t = self._now()
                # policy-origin capture: average the boot marker XY, then shift by the
                # (yaw0-rotated) marker->base offset so the origin is under base_link.
                if self._origin_capture is not None and not self._origin_capture.done:
                    xy = self._origin_capture.push(pos_table)
                    if xy is not None:
                        c, s = math.cos(self._t2p.yaw_table), math.sin(self._t2p.yaw_table)
                        off = self._marker_to_base
                        xy = xy + np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
                        self._t2p.set_origin_xy(xy, "mocap_capture")
                        self.get_logger().info(
                            f"[origin capture] policy origin set from mocap: table-frame XY "
                            f"({xy[0]:+.3f}, {xy[1]:+.3f}), yaw {self._t2p.yaw_table:.3f} rad "
                            f"(robot_start_yaw_table — placement convention, NOT measured).")
            self.create_subscription(PoseStamped, base_pose_topic, bp_cb, qos)
            self.get_logger().info(
                f"state_source=ros: base pose from '{base_pose_topic}' (geometry_msgs/PoseStamped, "
                "HOPE TABLE frame -> converted to the policy frame; position used, orientation "
                f"{'used' if self._use_mocap_ori else 'IGNORED (position-only mocap; IMU orientation used)'}).")
        else:
            self.get_logger().warn(
                "no base_pose_topic: absolute base/torso pose is the sim2real localisation gap -> "
                "using a nominal upright pose at the policy origin (nominal_base_height/torso_offset_z). "
                "World-frame racket targets are still converted with the CONFIGURED robot start pose, "
                "but walking/anchor obs are approximate until the mocap base pose is wired.")

    def destroy_node(self):
        if self._csv_file is not None:
            self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WbcRunnerNode()
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
