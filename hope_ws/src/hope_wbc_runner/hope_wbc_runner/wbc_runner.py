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

from .frame_math import projected_gravity_body
from .obs_builder import N_JOINTS, RacketTarget, RobotState, build_obs, synthetic_state_from_refs
from .onnx_policy import OnnxPolicy
from .reference_clock import (
    ClipLayout,
    clip_id_from_swing_sign,
    clip_phase,
    swing_sign_from_target_y,
    time_step_for,
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
        self.declare_parameter("base_pose_topic", "")          # optional geometry_msgs/PoseStamped pelvis pose (world)
        self.declare_parameter("joint_state_type", "sensor_msgs")  # "sensor_msgs" (std) | "joint_msgs" (a3 sim)
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
        self._layout = ClipLayout(
            seg_len=tuple(int(v) for v in gp("seg_len").value),
            strike_phase=tuple(float(v) for v in gp("strike_phase").value),
            step_dt=self._dt,
        )
        self._timeout = float(gp("command_timeout_s").value)
        self._base_target_xy = np.array([float(v) for v in gp("base_target_xy").value])
        self._state_source = str(gp("state_source").value)
        self._hardware_enable = bool(gp("hardware_enable").value)
        self._estop = False

        # state
        self._last_action = np.zeros(N_JOINTS)
        self._last_cmd = None              # latest RacketCommand
        self._last_cmd_t = None            # wall time it arrived
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
        self._imu = None             # (quat_w, ang_vel_b) latest
        self._base_pose = None       # (pos_w, quat_w) from base_pose_topic, if provided
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
                 "joint_order_matched_count", "missing_joint_count", "base_pose_source"]
                + [f"target_q_{i}" for i in range(N_JOINTS)])

        self._t0 = self._now()
        self.create_timer(self._dt, self._tick)

        self.get_logger().info(
            f"wbc_runner started | mode={self._mode} | state_source={self._state_source} | "
            f"rate={rate:.0f} Hz | dither={self._dither} (deterministic={self._dither == 0.0}) | "
            f"seg_len={self._layout.seg_len} strike_phase={self._layout.strike_phase}")
        self.get_logger().warn(
            "SAFETY: publishes joint commands ONLY when mode=hardware AND hardware_enable AND "
            f"!estop. Now: mode={self._mode}, hardware_enable={self._hardware_enable}, estop={self._estop}. "
            f"{'WILL PUBLISH when enabled' if self._mode == 'hardware' else 'WILL NOT PUBLISH (log only)'}.")

    # ----- time -----
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ----- callbacks -----
    def _cmd_cb(self, msg: RacketCommand):
        self._last_cmd = msg
        self._last_cmd_t = self._now()
        if msg.header.frame_id not in ("base_link", "") and not self._cmd_warned_frame:
            self._cmd_warned_frame = True
            self.get_logger().warn(
                f"/racket/command frame_id='{msg.header.frame_id}'. This runner treats targets as the "
                "policy robot-ground frame (planner_imitate frame_id='base_link'). 'world' targets need a "
                "world->robot transform (robot pose unmeasured) - NOT applied here yet.")

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
        stale = (self._last_cmd_t is None) or (self._now() - self._last_cmd_t > self._timeout)

        # idle / stand: no command, stale, estop, or planner says valid=False
        if cmd is None or stale or self._estop or not cmd.valid:
            reason = ("estop" if self._estop else "no_command" if cmd is None else
                      "stale" if stale else "planner_invalid")
            target_q = self.policy.default_q.copy()      # hold nominal stand pose
            self._last_action = np.zeros(N_JOINTS)
            self._emit(t, target_q, swing="stand", clip_id=-1, time_step=-1, cphase=float("nan"),
                       tts=float("nan"), valid=False, obs_norm=float("nan"), action_norm=0.0,
                       idle_reason=reason)
            return

        # active swing: drive the reference clock from the planner's time_to_strike
        target_y = cmd.position.y
        swing_sign = swing_sign_from_target_y(target_y)
        clip_id = clip_id_from_swing_sign(swing_sign)
        # tts decays since the command arrived (planner publishes the value at send time)
        tts = float(cmd.time_to_strike) - (self._now() - self._last_cmd_t)
        time_step = time_step_for(self._layout, clip_id, tts)
        cphase = clip_phase(self._layout, clip_id, time_step)

        refs = self.policy.refs(time_step)
        state = self._get_state(refs)
        target = RacketTarget(
            pos_w=np.array([cmd.position.x, cmd.position.y, cmd.position.z]),
            vel_w=np.array([cmd.velocity.x, cmd.velocity.y, cmd.velocity.z]),
            swing_sign=swing_sign, time_to_strike=tts, base_target_xy=self._base_target_xy)

        obs = build_obs(refs, state, target, self._last_action, self.policy.default_q)
        self._verify_obs_once(state)
        action = self.policy.action(obs, time_step, dither_scale=self._dither)
        self._last_action = action
        target_q = self.policy.target_q(action)

        self._emit(t, target_q, swing=("forehand" if swing_sign > 0 else "backhand"),
                   clip_id=clip_id, time_step=time_step, cphase=cphase, tts=tts, valid=True,
                   obs_norm=float(np.linalg.norm(obs)), action_norm=float(np.linalg.norm(action)),
                   idle_reason="")

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
        base_quat_w, ang_vel_b = self._imu
        if self._base_pose is not None:
            base_pos_w, base_quat_w = self._base_pose      # real localisation (preferred)
        else:
            base_pos_w = np.array([0.0, 0.0, float(self.get_parameter("nominal_base_height").value)])
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
        from .frame_math import projected_gravity_body
        pg = projected_gravity_body(state.base_quat_w)
        self.get_logger().info(
            "[obs verify] first real-state observation built. Sanity checks:\n"
            f"  base_quat_w (w,x,y,z) = {np.round(state.base_quat_w, 3).tolist()}\n"
            f"  projected_gravity (body) = {np.round(pg, 3).tolist()}  "
            f"(upright should be ~[0,0,-1]; if it's far off, the IMU orientation frame is wrong)\n"
            f"  base_ang_vel (body gyro) = {np.round(state.base_ang_vel_b, 3).tolist()}  "
            f"(should be ~0 when standing still)\n"
            f"  base_pos_w = {np.round(state.base_pos_w, 3).tolist()}  "
            f"({'real base_pose_topic' if self._base_pose is not None else 'NOMINAL fallback'})\n"
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
        quat, ang = self._imu
        if self._base_pose is not None:
            quat = self._base_pose[1]   # match _get_state: base orientation from base_pose when present
            src = "real"
        else:
            src = "nominal"
        pg = projected_gravity_body(quat)
        return tuple(pg), tuple(ang), self._joint_matched, self._joint_missing, src

    # ----- output + logging -----
    def _emit(self, t, target_q, swing, clip_id, time_step, cphase, tts, valid,
              obs_norm, action_norm, idle_reason):
        publish = (self._mode == "hardware") and self._hardware_enable and (not self._estop)
        if publish and self._cmd_pub is not None:
            self._cmd_pub_publish(target_q)
            self._n_published += 1

        # console: one line per ~0.5 s (avoid 50 Hz spam)
        if self._n_ticks % max(int(0.5 / self._dt), 1) == 1:
            tag = "PUBLISH" if publish else f"log-only/{self._mode}"
            extra = f" idle({idle_reason})" if idle_reason else ""
            self.get_logger().info(
                f"[{tag}] {swing}{extra} ts={time_step} phase={cphase:.2f} tts={tts:+.2f} "
                f"|act|={action_norm:.2f} max|q*|={float(np.max(np.abs(target_q))):.2f}")

        if self._csv is not None:
            pg, av, matched, missing, src = self._verify_values()
            self._csv.writerow(
                [f"{t:.4f}", self._mode, int(publish), swing, clip_id, time_step,
                 f"{cphase:.4f}", f"{tts:.4f}", int(valid), f"{obs_norm:.4f}",
                 f"{action_norm:.4f}", f"{float(np.max(np.abs(target_q))):.4f}",
                 f"{pg[0]:.4f}", f"{pg[1]:.4f}", f"{pg[2]:.4f}",
                 f"{av[0]:.4f}", f"{av[1]:.4f}", f"{av[2]:.4f}",
                 matched, missing, src]
                + [f"{v:.5f}" for v in target_q])
            self._csv_file.flush()
        self._publish_diag(swing, publish, time_step, tts, idle_reason)

    def _publish_diag(self, swing, publish, time_step, tts, idle_reason):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = "wbc_runner"
        st.hardware_id = "wbc_runner"
        st.level = DiagnosticStatus.WARN if (self._estop or idle_reason) else DiagnosticStatus.OK
        st.message = f"{self._mode} {swing} {'PUBLISHING' if publish else 'log-only'}"
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

        self.create_subscription(Imu, imu_topic, imu_cb, qos)
        self.get_logger().info(
            f"state_source=ros: joints '{joint_topic}' ({joint_state_type}/JointState), imu '{imu_topic}'.")

        if base_pose_topic:
            from geometry_msgs.msg import PoseStamped
            def bp_cb(msg):
                p = msg.pose.position
                o = msg.pose.orientation
                self._base_pose = (np.array([p.x, p.y, p.z]),
                                   np.array([o.w, o.x, o.y, o.z]))
            self.create_subscription(PoseStamped, base_pose_topic, bp_cb, qos)
            self.get_logger().info(f"state_source=ros: base pose from '{base_pose_topic}' (geometry_msgs/PoseStamped).")
        else:
            self.get_logger().warn(
                "no base_pose_topic: absolute base/torso WORLD pose is the sim2real localisation gap -> "
                "using a nominal upright pose (nominal_base_height/torso_offset_z). The anchor-POSITION "
                "obs term is approximate until you wire a real base-pose estimate.")

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
