"""Lightweight MuJoCo sim-to-sim runner for the HOPE A3 ping-pong BeyondMimic/HOPE ONNX policy.

Runs the EXACT policy contract used in Isaac training, but in MuJoCo (a different physics engine),
to verify the exported ONNX before hardware. NO retraining, NO reward changes, NO target-sampling
changes. This deliberately does NOT use the official Agibot 1570D HITTER-tokenizer C++ harness and
does NOT convert the policy to 29D — it runs the 31D BeyondMimic ONNX as-is.

SCOPE: this script is the IN-REPO STRIKE-METRICS tool (composite/pos/vel/normal pass rates, per-clip
breakdowns). The OFFICIAL deploy validation sim is the vendor C++/aimrt harness under
agi/A3_MuJoCo_Sim/aimrt_mujoco_sim (the real-hardware gate; deploy-faithful robot validation, no
strike metrics). The vendor/deploy runner drives a CONTINUOUS reference clock with NO per-swing
teleports — which is what the multiswing reset mode below mirrors on the training side.

=============================================================================================
POLICY CONTRACT (verified against the exported ONNX metadata + checkpoint weights, 2026-06-27)
=============================================================================================
ONNX (logs/.../basecouple03_resume/exported/policy.onnx):
  inputs : obs[1,180] (float32), time_step[1,1] (float32)
  outputs: actions[1,31], joint_pos[1,31], joint_vel[1,31],
           body_pos_w[1,14,3], body_quat_w[1,14,4], body_lin_vel_w[1,14,3], body_ang_vel_w[1,14,3]
  -> outputs[1:] are the REFERENCE motion (the BeyondMimic clip) indexed by `time_step`. We use them
     as the single source of truth for the reference command + anchor (NO npz body-order guessing).

ACTOR OBSERVATION = 180D, concatenated in THIS order (verified: actor.0.weight is (512,180)):
   1. command            62  = cat(ref_joint_pos[31], ref_joint_vel[31])   (generated_commands("motion"))
   2. motion_anchor_pos_b  3  ref torso pose in the ROBOT torso (anchor) frame
   3. motion_anchor_ori_b  6  same, orientation as first 2 columns of the rotation matrix (6D rot rep)
   4. base_ang_vel         3  pelvis angular velocity in the PELVIS BODY frame (IMU gyro)
   5. joint_pos           31  q - default_joint_pos   (Isaac articulation order)
   6. joint_vel           31  qdot                     (Isaac articulation order)
   7. actions             31  last raw policy action (the 31-vec from the previous ONNX call)
   8. projected_gravity    3  gravity unit vec in the PELVIS BODY frame (IMU)
   9. base_target_pos_b    2  desired base XY in the yaw-heading base frame
  10. racket_target_pos_b  3  desired racket pos in the yaw-heading base frame
  11. racket_target_vel_w  3  desired racket velocity in WORLD frame
  12. time_to_strike       1  seconds until the strike frame
  13. swing_type           1  +1 forehand / -1 backhand
  (NOTE: the user's brief said 129D/command=13/ori=4 — that is WRONG vs the actual weights.
   The real contract is 180D with command=62 and motion_anchor_ori_b=6. We build the real one.)

FRAME CONVENTIONS (critical — these are NOT all world frame):
  * world frame ........ MuJoCo global frame. env origin = (0,0,0) (single env, matches Isaac env 0).
  * pelvis BODY frame .. base_ang_vel and projected_gravity are expressed here (IMU-measurable).
                         base_ang_vel_b = R_pelvis^T @ omega_world ; proj_grav_b = R_pelvis^T @ [0,0,-1].
  * yaw-heading base ... racket_target_pos_b / base_target_pos_b use yaw_quat(pelvis_quat) (Z-rotation
                         ONLY), then quat_rotate_inverse. So they are heading-relative, not full-3D.
  * anchor (torso) frame motion_anchor_* express the REFERENCE torso pose relative to the ROBOT torso.

JOINT ORDER:
  * Isaac articulation order = ONNX metadata `joint_names` (31, interleaved L/R; e.g. left_hip_pitch,
    right_hip_pitch, waist_yaw, ...). The obs, action, default_joint_pos, action_scale, kp, kd are ALL
    in this order. We work internally in this order.
  * MuJoCo qpos/qvel/actuators are in MJCF declaration order (waist, head, L-arm, R-arm, L-leg, R-leg).
  * We build a name-based permutation between the two and only convert at the sim boundary.

ACTION DECODE (Isaac articulation order):  target_q = default_joint_pos + raw_action * action_scale
  (use_default_offset=True; no action clipping in training — clip_actions=null). default_joint_pos,
  action_scale, kp(stiffness), kd(damping) are all read from the ONNX metadata.

PD CONTROL (per physics substep, target_q held across the control step / decimation):
  torque = kp * (target_q - q) - kd * qdot   then clipped to the actuator effort limits.
  The PD gains ARE the official Agibot a3_kps/a3_kds (the Isaac config transcribes them). To avoid
  DOUBLE damping we zero the MJCF passive joint damping + frictionloss on the 31 actuated DOFs (Isaac's
  ImplicitActuator models damping via kd only); armature is kept (physical, present in both). Use
  --keep-passive to leave the MJCF damping/frictionloss in (a harder, less faithful test).

CONTROL FREQUENCY: 50 Hz. Isaac used sim_dt=0.005 * decimation=4. We mirror that (--sim-dt/--decimation).

REFERENCE-STATE-INIT (matches Isaac): each new swing (episode reset OR clip wrap) teleports the robot
  to the reference pose at that clip's first frame (root pose from ref body 0 = pelvis_link, joints from
  ref_joint_pos). Isaac does exactly this in MotionCommand._resample_command, so an episode contains
  several teleport-initialised swings until a termination (fall) or the 10 s timeout.

OBSERVATION NOISE: training adds small uniform obs corruption; deployment/sim-to-sim feeds CLEAN obs
  (the ONNX is deterministic). We feed clean obs and document it; sensor noise is a separate concern.

=============================================================================================
P0 FIX (2026-07-04): all-zero scores for the multiswing generation — TWO root causes
=============================================================================================
1. OBS NORMALIZATION (the actual all-zero bug, affects EVERY generation).
   Every run trains with rsl_rl `empirical_normalization: true`, i.e. the actor consumes
   (obs - mean) / (std + 0.01) with running stats stored in the checkpoint's obs_norm_state_dict.
   The export chain (scripts/play.py -> _OnnxPolicyExporter, and standalone_onnx_export.py) bakes the
   RAW actor with NO normalizer (verified 2026-07-04 by zero-point matching the p21_E and old-gen
   hopex ONNXs against their checkpoints: ONNX == raw actor to ~1e-6; the checkpoints DO carry
   non-trivial stats, mean |max| ~4, var up to ~13). Feeding raw obs to a normalized-obs actor
   lobotomizes the policy: in MuJoCo it staggered forward-right ~0.5 m on a canonical trajectory
   regardless of the sampled target (the 0.53 +/- 0.06 m "systematic offset"), tripped the
   anchor_pos/ee_body_pos tracking guards at ~52 steps, and only the backhand strike frame
   (44 steps in) was ever inside an episode — forehand (65 steps in) never evaluated. It could not
   even hold a nominal stand (deploy-faithful fell at ~1.1 s). Diagnosed with a perfect-tracking
   probe: with the robot PINNED to the reference each step, actions still exploded (|a| -> ~60)
   through the last-action feedback obs; with the normalizer applied they are sane.
   FIX: this runner now loads an `obs_norm.npz` sidecar (keys mean/std/eps; produced from the SAME
   checkpoint as the ONNX by scripts/make_std_sidecar.py) and applies (obs - mean)/(std + eps)
   before inference. Resolution: --obs-norm PATH > auto (<onnx_dir>/obs_norm.npz) > loud WARNING and
   raw obs (only correct for a hypothetical normalizer-free training run or a future export that
   bakes the normalizer in). NOTE FOR DEPLOY: the C++ runner consuming these ONNX files raw has the
   same defect — either bake the normalizer into the export or normalize obs on the robot.
2. EPISODE PROTOCOL (--reset-mode): the old harness teleported the robot to the next clip's first
   frame on EVERY clip wrap (legacy RSI generation, wrap_teleport=true). The multiswing generation
   (HOPEPingPong/DeployParity, 2026-07+) trains with wrap_teleport=false + a pre-swing HOLD of
   U[0,100] control steps (reference frozen at the swing's first frame, time_to_strike pinned at its
   per-clip max) and the robot physically carries itself between swings. New flag:
     --reset-mode teleport   : byte-identical legacy behavior (teleport per swing, no holds).
     --reset-mode multiswing : no wrap teleports; per-swing pre-swing hold sampled from
                               --hold-steps-range (training default 0..100); last_action persists
                               across wraps (training only zeroes it on true episode resets); adds
                               the absolute balance terminations (tilt > 0.7 rad, pelvis z < 0.5 m)
                               that HOPEDeployParityTerminationsCfg trains with, alongside the
                               inherited tracking guards.
     --reset-mode auto (default): ONNX metadata `wrap_teleport` when present, else multiswing.
   Episode RESETS still reference-state-init at the sampled clip's first frame in both modes
   (training's dominant reset path); the 10 s timeout matches episode_length_s.
   Also fixed here: the strike-phase precedence is resolved (CLI > ONNX metadata > legacy builtin)
   BEFORE anything is printed or computed — the old code printed a misleading
   "strike_phase_per_clip in effect" line from the builtin default before metadata resolution
   (the metadata value did take effect for the metrics, but the double print hid what ran).

=============================================================================================
TARGET SOURCE (--target-source, 2026-07-04): where the per-strike racket targets come from.
  boxes (default) ....... per-clip training boxes (mode A, in-distribution vs training). This is
                          the pre-existing behavior, byte-identical (same RNG stream, same CSVs).
  venue-balls (mode B) .. distribution-driven REALISM eval: sample an INCOMING BALL at-strike
                          state from the fitted venue distribution (configs/
                          incoming_ball_venue.yaml pooled/matchlike spec — the eval principle in
                          that file), invert it through the StrikeSpec planner (hope_ws/src/
                          hope_planner, imported lazily by path) into the racket (pos, vel,
                          normal) that ball DEMANDS for a sampled opponent-half landing, and feed
                          that through the unchanged target pipeline. Strikes are scored exactly
                          like mode A (pos/vel/normal pass + composite) AND, for CONTACTED strikes
                          (training capture gate: pos_err < 0.095 m, approach > 0.3 m/s), the
                          virtual return of the ACHIEVED racket state vs the SAMPLED ball is
                          rolled out (venue contact model + drag/Magnus flight) to a landing:
                          the mode-B summary reports the landing-in-bounds/net-clear rate
                          ("回球成功率" headline) + median landing error vs the intended target,
                          and the strikes CSV gains ball/landing columns. Frames, geometry and v1
                          caveats (independent box sampling ignores the documented correlations;
                          human-height contact z) are documented in scripts/venue_ball_sampler.py.
                          Not supported together with --deploy-faithful (v1).
                          COUNTERFACTUAL (committed 2026-07-05; was the ad-hoc 2026-07-04 analysis
                          that found 0/25 -> 25/25): every venue strike is ALSO scored with the
                          DEMANDED face normal swapped in for the achieved one (same achieved
                          pos/vel/pos_err) — cf_* CSV columns + summary rows. It isolates the
                          normal channel: cf return rate >> actual return rate means the face
                          orientation alone is what fails the return (the 175-D contract has no
                          normal channel; docs/motion_and_contract_v3.md).
  venue-balls + --venue-fixed-normal (path A, 2026-07-05): the StrikeSpec inversion PINS the
                          face normal at the swing side's clip reference normal
                          (solve_fixed_normal — velocity-only inversion): the planner adapts to
                          the clip-locked face the policy actually produces. The demanded normal
                          becomes reachable, so the return_success_rate under this flag is the
                          ZERO-TRAINING DEPLOYMENT CEILING of the current policy + an adapted
                          planner. Compare against the free-normal baseline + its counterfactual.

SWITCH-STRESS (--switch-stress P, 2026-07-05; R11's missing benefit ruler): deploy-parity
  mid-swing clip-switch stress protocol for the training-like multiswing rollout. Each control
  step, with probability P, the reference clock aborts the swing exactly like the deploy runner
  does when the planner changes its mind (commands.py clip_switch_prob semantics /
  pp_reference_clock.hpp tts clamp): uniform new clip, reference jumped to its windup frame,
  fresh pre-swing hold + racket target, robot state untouched. While ON, tracking-guard
  terminations are DISABLED (balance falls + timeout only): the reference jump makes imitation
  guards fire spuriously, and the question is deploy falls. Reports switches, falls, 2 s
  post-switch survival, and hit rates on post-switch swings vs clean swings (the within-run
  baseline). P=0 (default) draws nothing — byte-identical baseline behavior.

=============================================================================================
USAGE (motion env that has mujoco + onnxruntime, e.g. hope-motion-py310):
  python scripts/mujoco_eval_onnx.py \
      --onnx logs/rsl_rl/agibot_a3_hope/2026-06-27_18-14-06_basecouple03_resume/exported/policy.onnx \
      --mjcf agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
      --motion-files logs/rsl_rl/eval_motion/fh.npz logs/rsl_rl/eval_motion/bh.npz \
      --noise-scales 0.0 0.05 --steps 1200 --seed 0
  (paths are relative to the whole_body_tracking/ dir; --mjcf is relative to the repo root by default)
=============================================================================================
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np

# Validated pure-numpy racket forward-kinematics reference (racket pos in the pelvis frame from the
# 31 Isaac-order joint angles; validated to ~2e-6 m vs Isaac). Used ONLY by the 175-D deploy_parity
# obs path to reframe racket_target_pos_b relative to the current racket FK. Import from the sibling
# module (same scripts/ dir) so it works regardless of the caller's cwd.
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from racket_fk_ref import racket_pos_pelvis  # noqa: E402

# -------------------------------------------------------------------------------------------------
# Constants pulled from the verified training config (HOPEPingPong.yaml uniform-mode + RacketTargetCmd).
# These are the RUNTIME values (YAML overrides applied), not the python cfg defaults.
# -------------------------------------------------------------------------------------------------
ANCHOR_BODY = "torso_Link"               # ONNX metadata anchor_body_name
# 14 tracked bodies, in ONNX metadata `body_names` order (index 0 = root pelvis, 7 = torso anchor).
TRACKED_BODIES = [
    "pelvis_link", "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
    "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link", "torso_Link",
    "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
    "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link",
]
ANCHOR_TRACKED_IDX = TRACKED_BODIES.index(ANCHOR_BODY)   # 7
ROOT_TRACKED_IDX = 0                                     # pelvis_link
# End-effector bodies for the ee_body_pos termination (indices INTO the 14 tracked list).
EE_TRACKED_IDX = [TRACKED_BODIES.index(n) for n in
                  ["left_ankle_roll_Link", "right_ankle_roll_Link",
                   "left_wrist_yaw_Link", "right_wrist_yaw_Link"]]
FEET_BODIES = ["left_ankle_roll_Link", "right_ankle_roll_Link"]

# Termination thresholds (tracking_env_cfg.TerminationsCfg).
TERM_ANCHOR_POS_Z = 0.25     # |ref_torso_z - robot_torso_z| > 0.25  -> fall
TERM_ANCHOR_ORI = 0.8        # |proj_grav_z(ref) - proj_grav_z(robot)| > 0.8 -> fall
TERM_EE_POS_Z = 0.25         # any ee |z(ref_relative) - z(robot)| > 0.25 -> fall

# --deploy-faithful fall thresholds — mirror the training BALANCE terminations (hope_env_cfg
# RealSensor TerminationsCfg: bad_orientation limit_angle=0.7 rad, root_height_below_minimum=0.5 m).
# In deploy-faithful mode these are the ONLY episode-enders (no tracking guards, no 10 s timeout).
DF_FALL_TILT_RAD = 0.7       # acos(-proj_grav_z) > 0.7 rad (~40 deg from upright) -> fall
DF_FALL_ROOT_Z_MIN = 0.5     # pelvis z below 0.5 m -> fall

# RacketTargetCommand uniform-mode sampling. DEFAULTS UPDATED 2026-07-03 to mirror the CURRENT
# training config (cfg/task/HOPEPingPongDeployParity.yaml, 2026-07-02 blade re-plane): PER-CLIP
# pos/vel boxes centered on each clip's reference BLADE strike state. The legacy shared box below is
# the fallback when POS/VEL_RANGE_PER_CLIP is None (e.g. --pos-z-range override) — it matches the OLD
# HOPEPingPong.yaml wrist-era generation; evaluating a DeployParity model on it puts the targets
# out-of-training-distribution.
POS_RANGE_PER_CLIP = (
    ((0.58, 0.78), (-0.64, -0.24), (0.72, 0.92)),   # forehand: blade strike (0.68, -0.44, 0.82)
    ((0.56, 0.76), (-0.07, 0.33), (0.93, 1.13)),    # backhand: blade strike (0.66,  0.13, 1.03)
)
VEL_RANGE_PER_CLIP = (
    ((1.05, 2.05), (0.96, 1.96), (0.31, 1.11)),     # forehand: blade clean strike vel (1.55, 1.46, 0.71)
    ((1.61, 2.61), (-1.21, -0.21), (0.00, 0.71)),   # backhand: blade clean strike vel (2.11, -0.71, 0.31)
)
RACKET_POS_X_RANGE = (0.40, 0.40)        # legacy fallback: fixed strike plane (x), rel. to env origin
RACKET_POS_Y_ABS_RANGE = (0.05, 0.45)    # legacy fallback: |y|; sign set per clip
RACKET_POS_Z_RANGE = (0.70, 1.05)        # legacy fallback
RACKET_VEL_X_RANGE = (1.5, 3.5)          # legacy fallback
RACKET_VEL_Y_RANGE = (-1.0, 1.0)         # legacy fallback
RACKET_VEL_Z_RANGE = (0.0, 1.5)          # legacy fallback
BASE_TARGET_X_RANGE = (-0.10, 0.10)
BASE_TARGET_Y_RANGE = (-0.10, 0.10)
BASE_COUPLE_BLEND = 0.3                  # weak base->racket Y coupling
BASE_COUPLE_MAX_OFFSET = 0.20
FOREHAND_ON_NEGATIVE_Y = True            # forehand (clip 0) target on -y
# forehand / backhand contact phase. MUST match the trained model's task YAML
# `racket.strike_phase_per_clip`. Resolution order at runtime: --strike-phase-per-clip CLI override >
# `clip_strike_phases` baked in the ONNX metadata (scripts/play.py; same keys the C++ runner uses) >
# this built-in fallback. The fallback stays at the LEGACY value because only metadata-less (= old)
# exports ever reach it: v1 clips (0.36, 0.50); model_32200-era backhand 0.74 (dead recovery frame —
# strike metrics collapse to ~0). Current v2-blade/_hopex clips (0.47, 0.333) always arrive via ONNX
# metadata; a wrong phase silently collapses strike metrics, so pass the era explicitly for old models.
STRIKE_PHASE_PER_CLIP = (0.36, 0.50)
STRIKE_WINDOW_S = 0.12
# Strike-success acceptance thresholds (RacketTargetCommandCfg) — identical to Isaac's exact metric.
STRIKE_POS_THRESH = 0.075                 # m   strike_success_pos_thresh
STRIKE_VEL_THRESH = 0.5                   # m/s strike_success_vel_thresh  (||actual - target|| 3-vec)
STRIKE_NORMAL_THRESH_DEG = 15.0           # deg strike_success_normal_thresh_deg  (acos(dot) of unit normals)
# Racket face normal = local +Y axis of the racket frame (== wrist frame; mount_quat is identity).
MOUNT_NORMAL_AXIS = 1
MOUNT_NORMAL_SIGN = 1.0
WRIST_TRACKED_IDX = TRACKED_BODIES.index("right_wrist_yaw_Link")   # 13; racket frame == this body's frame
CLIP_NAMES = {0: "forehand", 1: "backhand"}


# =================================================================================================
# Quaternion / frame math (numpy, scalar-first w,x,y,z — MuJoCo and IsaacLab both use this order).
# =================================================================================================
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / (w * w + x * x + y * y + z * z)


def mat_from_quat(q):
    """3x3 rotation matrix R (world<-body) from quaternion (w,x,y,z). Matches IsaacLab matrix_from_quat."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_rotate(q, v):
    return mat_from_quat(q) @ v


def quat_rotate_inverse(q, v):
    return mat_from_quat(q).T @ v


def yaw_quat(q):
    """Quaternion of the yaw-only (Z) rotation of q. Matches IsaacLab yaw_quat."""
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def subtract_frame_transforms(t01, q01, t02, q02):
    """Pose of frame 2 expressed in frame 1. Returns (t12, q12). Matches IsaacLab."""
    q10 = quat_inv(q01)
    t12 = quat_rotate(q10, t02 - t01)
    q12 = quat_mul(q10, q02)
    return t12, q12


# =================================================================================================
# ONNX policy wrapper
# =================================================================================================
class OnnxPolicy:
    def __init__(self, onnx_path, obs_norm="auto"):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ins = {i.name: i.shape for i in self.sess.get_inputs()}
        outs = [o.name for o in self.sess.get_outputs()]
        assert "obs" in ins and "time_step" in ins, f"unexpected ONNX inputs: {ins}"
        self.obs_dim = int(ins["obs"][1])
        assert self.obs_dim in (175, 179, 180), \
            f"expected obs dim 180 (base), 175 (deploy_parity) or 179 (deploy_parity + face command), got {self.obs_dim}"
        # 175-D deploy_parity = 180-D MINUS motion_anchor_pos_b(3) and base_target_pos_b(2), with the
        # racket_target_pos_b term reframed relative to the CURRENT racket FK (not the base). See build_obs.
        # 179-D (stage 1, 2026-07-06) = 175-D + racket_target_normal_cmd tail: DEMANDED face normal
        # (3, world) + zero-filled rho placeholder (1) — the frozen contract-day 175->179 layout
        # (train.py face_command_obs appends the term LAST, so the 175 prefix is byte-identical).
        self.deploy_parity = (self.obs_dim in (175, 179))
        self.face_command = (self.obs_dim == 179)
        self.out_names = outs
        md = self.sess.get_modelmeta().custom_metadata_map
        # --- metadata (FAIL LOUDLY if anything is missing) -------------------------------------
        required = ["joint_names", "default_joint_pos", "action_scale", "joint_stiffness",
                    "joint_damping", "body_names", "anchor_body_name", "observation_names"]
        missing = [k for k in required if k not in md or not md[k].strip()]
        if missing:
            raise SystemExit(
                "[FATAL] ONNX is missing required metadata keys: " + ", ".join(missing) +
                "\nThis runner needs them to map joints/gains. Re-export with attach_onnx_metadata "
                "(scripts/play.py) so they are baked in.")
        self.joint_names = md["joint_names"].split(",")
        self.default_q = np.array([float(v) for v in md["default_joint_pos"].split(",")], np.float64)
        self.action_scale = np.array([float(v) for v in md["action_scale"].split(",")], np.float64)
        self.kp = np.array([float(v) for v in md["joint_stiffness"].split(",")], np.float64)
        self.kd = np.array([float(v) for v in md["joint_damping"].split(",")], np.float64)
        self.body_names = md["body_names"].split(",")
        # optional clip-clock metadata (baked by scripts/play.py; the same keys the C++ deploy
        # runner uses to override its built-in clip layout)
        self.clip_strike_phases = None
        if md.get("clip_strike_phases", "").strip():
            self.clip_strike_phases = tuple(float(v) for v in md["clip_strike_phases"].split(","))
        self.clip_seg_lengths = None
        if md.get("clip_seg_lengths", "").strip():
            self.clip_seg_lengths = tuple(int(float(v)) for v in md["clip_seg_lengths"].split(","))
        # optional episode-semantics metadata (future exports may bake the training wrap_teleport
        # flag; none do as of 2026-07-04 — --reset-mode auto then defaults to multiswing).
        self.wrap_teleport_meta = None
        if md.get("wrap_teleport", "").strip():
            self.wrap_teleport_meta = md["wrap_teleport"].strip().lower() in ("1", "true", "yes")
        n = len(self.joint_names)
        assert n == 31 and self.default_q.shape == (31,) and self.action_scale.shape == (31,), \
            f"expected 31 joints, got {n}"
        assert self.body_names == TRACKED_BODIES, \
            f"ONNX body_names != expected tracked order:\n {self.body_names}\n {TRACKED_BODIES}"
        # --- empirical obs normalization (P0 fix 2026-07-04, see module docstring #1) -------------
        # Every training run uses rsl_rl empirical_normalization=true, but the export chain bakes the
        # RAW actor. The `obs_norm.npz` sidecar (mean/std/eps from the checkpoint's
        # obs_norm_state_dict; scripts/make_std_sidecar.py writes it next to the ONNX) restores the
        # training-time obs transform (obs - mean) / (std + eps). Without it, a normalized-obs model
        # is evaluated on garbage and scores ~0 with a very consistent staggering pathology.
        self.obs_mean = self.obs_std = None
        self.obs_eps = 1e-2                      # rsl_rl EmpiricalNormalization default
        self.obs_norm_path = None
        # Double-normalization guard: exports made with standalone_onnx_export.py --bake-obs-norm
        # carry obs_norm_baked=1 in metadata and must NOT get the sidecar on top.
        if str(md.get("obs_norm_baked", "0")) == "1":
            print("[mj-sim2sim] obs normalization BAKED into the ONNX graph (metadata) — sidecar skipped")
            obs_norm = "off"
        if obs_norm != "off":
            path = obs_norm if obs_norm not in (None, "auto") else \
                os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "obs_norm.npz")
            if os.path.isfile(path):
                d = np.load(path)
                mean = np.asarray(d["mean"], np.float64).reshape(-1)
                std = np.asarray(d["std"], np.float64).reshape(-1)
                assert mean.shape == (self.obs_dim,) and std.shape == (self.obs_dim,), (
                    f"obs_norm sidecar dim {mean.shape}/{std.shape} != obs dim {self.obs_dim} "
                    f"({path}) — sidecar from a different obs contract/checkpoint?")
                self.obs_mean, self.obs_std = mean, std
                self.obs_eps = float(d["eps"]) if "eps" in d else 1e-2
                self.obs_norm_path = path
            elif obs_norm not in (None, "auto"):
                raise SystemExit(f"[FATAL] --obs-norm sidecar not found: {path}")

    def normalize_obs(self, obs):
        """Training-time empirical obs normalization (identity if no sidecar loaded)."""
        if self.obs_mean is None:
            return obs
        return (obs - self.obs_mean) / (self.obs_std + self.obs_eps)

    def refs(self, time_step):
        """Reference motion at `time_step` (obs-independent). Returns dict of arrays in metadata order."""
        obs = np.zeros((1, self.obs_dim), np.float32)
        ts = np.array([[float(time_step)]], np.float32)
        o = self.sess.run(None, {"obs": obs, "time_step": ts})
        names = self.out_names
        d = {names[i]: o[i][0] for i in range(len(names))}
        return d

    def action(self, obs, time_step):
        ts = np.array([[float(time_step)]], np.float32)
        obs = self.normalize_obs(obs)
        o = self.sess.run(None, {"obs": obs[None].astype(np.float32), "time_step": ts})
        return o[0][0].astype(np.float64)    # mean action (31,), Isaac articulation order


# =================================================================================================
# MuJoCo robot wrapper (handles the articulation<->MJCF joint permutation + FK reads)
# =================================================================================================
class MujocoRobot:
    def __init__(self, mjcf_path, joint_names, body_names, sim_dt, keep_passive, pd_mode, kd_for_implicit=None):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.opt.timestep = sim_dt
        self.pd_mode = pd_mode      # "explicit" (torque kp*e - kd*qd) or "implicit" (kp torque + kd as
                                    #  passive damping integrated by MuJoCo's implicitfast integrator,
                                    #  matching Isaac's ImplicitActuator semantics).
        self.data = mujoco.MjData(self.model)

        def jid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        def bid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        def aid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # Per actuated joint (in ARTICULATION order): qpos addr, qvel/dof addr, actuator id.
        self.qadr = np.array([self.model.jnt_qposadr[jid(n)] for n in joint_names], int)
        self.vadr = np.array([self.model.jnt_dofadr[jid(n)] for n in joint_names], int)
        self.act_id = np.array([aid(n + "_motor") for n in joint_names], int)
        assert (self.act_id >= 0).all(), "missing <joint>_motor actuator(s) in MJCF"
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_id, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_id, 1].copy()

        # Body ids for the 14 tracked bodies, the pelvis (free base) and torso (anchor).
        self.tracked_bid = np.array([bid(n) for n in body_names], int)
        self.pelvis_bid = bid("pelvis_link")
        self.torso_bid = bid(ANCHOR_BODY)
        self.racket_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_racket")
        self.feet_bid = [bid(n) for n in FEET_BODIES]
        self.feet_geoms = {g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] in self.feet_bid}

        # Zero MJCF passive damping + frictionloss on the 31 actuated DOFs (kd provides damping;
        # avoids double-damping vs Isaac's ImplicitActuator). Armature is kept (physical).
        if not keep_passive:
            self.model.dof_damping[self.vadr] = 0.0
            self.model.dof_frictionloss[self.vadr] = 0.0
        if pd_mode == "implicit":
            # Match Isaac's ImplicitActuator: the kd damping is integrated IMPLICITLY (stable + no
            # under-shoot of fast swings at a 5 ms step). Put kd into the passive joint damping and
            # use MuJoCo's implicitfast integrator; the control torque then applies kp only.
            assert kd_for_implicit is not None
            self.model.dof_damping[self.vadr] = kd_for_implicit
            self.model.opt.integrator = int(self.mj.mjtIntegrator.mjINT_IMPLICITFAST)

    # --- state reads (all returned in ARTICULATION order or world/body frame as named) ----------
    def q_artic(self):
        return self.data.qpos[self.qadr].copy()

    def qd_artic(self):
        return self.data.qvel[self.vadr].copy()

    def body_pos(self, bid):
        return self.data.xpos[bid].copy()

    def body_quat(self, bid):
        return self.data.xquat[bid].copy()

    def pelvis_ang_vel_body(self):
        """Pelvis angular velocity in the pelvis BODY frame (== IMU gyro). Uses mj_objectVelocity."""
        res = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY,
                                  self.pelvis_bid, res, 1)  # flg_local=1 -> body frame
        return res[:3].copy()   # [angular(3), linear(3)] -> angular

    def projected_gravity_body(self):
        R = self.data.xmat[self.pelvis_bid].reshape(3, 3)   # world<-body
        return R.T @ np.array([0.0, 0.0, -1.0])

    def tracked_pos(self):
        return self.data.xpos[self.tracked_bid].copy()      # (14,3) world

    def racket_pos(self):
        return self.data.site_xpos[self.racket_site].copy()  # world

    def racket_lin_vel_w(self):
        # World-frame linear velocity of the racket site (== pingpang_red_Link origin; coincident with
        # Isaac's racket body, whose data.body_lin_vel_w is the analytic rigid-body origin velocity).
        res = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_SITE,
                                  self.racket_site, res, 0)  # flg_local=0 -> world frame; [ang(3), lin(3)]
        return res[3:6].copy()

    def racket_normal_w(self):
        # Actual racket face normal in world = local +Y axis of the racket(=wrist) frame.
        # site has identity orientation rel. to the wrist, so site_xmat == wrist world rotation.
        R = self.data.site_xmat[self.racket_site].reshape(3, 3)
        return R[:, MOUNT_NORMAL_AXIS] * MOUNT_NORMAL_SIGN

    def foot_contact_frac(self):
        ncon = self.data.ncon
        feet_in_contact = set()
        for i in range(ncon):
            c = self.data.contact[i]
            for g in (c.geom1, c.geom2):
                if g in self.feet_geoms:
                    feet_in_contact.add(self.model.geom_bodyid[g])
        return len(feet_in_contact) / max(len(self.feet_bid), 1)

    def reset_to_reference(self, root_pos, root_quat, root_lin_w, root_ang_w, q_artic):
        """Reference-state-init: teleport base + joints to the reference pose/vel (world frame inputs)."""
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat
        self.data.qpos[self.qadr] = q_artic
        # MuJoCo free-joint qvel: linear in WORLD frame, angular in the BODY frame.
        R = mat_from_quat(root_quat)
        self.data.qvel[0:3] = root_lin_w
        self.data.qvel[3:6] = R.T @ root_ang_w
        self.data.qvel[self.vadr] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def reset_to_stand(self, root_pos, root_quat, q_artic):
        """--deploy-faithful episode init: nominal stand (default_joint_pos, upright root at standing
        height), ALL velocities zero. This mirrors how the deployed robot enters MOTION from PD_STAND
        (pp_policy.hpp) — it is deliberately NOT reference-state-init."""
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat
        self.data.qpos[self.qadr] = q_artic
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def apply_pd_and_step(self, target_q_artic, kp, kd, decimation):
        """Hold target_q across `decimation` physics substeps, recomputing PD torque each substep.
        explicit: tau = kp*(tgt-q) - kd*qd (full PD as motor force).
        implicit: tau = kp*(tgt-q) only; kd is the passive joint damping integrated by implicitfast."""
        for _ in range(decimation):
            q = self.data.qpos[self.qadr]
            qd = self.data.qvel[self.vadr]
            tau = kp * (target_q_artic - q)
            if self.pd_mode == "explicit":
                tau = tau - kd * qd
            tau = np.clip(tau, self.ctrl_lo, self.ctrl_hi)
            self.data.ctrl[self.act_id] = tau
            self.mj.mj_step(self.model, self.data)
        return tau   # last substep torque (for logging)


# =================================================================================================
# RacketTargetCommand port (uniform mode, unified 2-clip) — only the obs-feeding quantities.
# =================================================================================================
class RacketCommand:
    def __init__(self, seg_start, seg_len, step_dt, rng, target_normal_per_clip, origin=np.zeros(3),
                 vel_ranges_per_clip=None, pos_ranges_per_clip=None):
        self.seg_start = seg_start          # (num_clips,)
        self.seg_len = seg_len
        self.step_dt = step_dt
        self.rng = rng
        self.origin = origin                # env origin (world). Single env -> (0,0,0), like Isaac env 0.
        # Per-clip target paddle normal: the imitated swing's reference face normal at strike (unified
        # uniform mode uses this, NOT a velocity-derived normal). Precomputed from the ref wrist quat.
        self.target_normal_per_clip = target_normal_per_clip
        # DIAGNOSTIC (eval-only, --eval-per-clip-vel-targets): optional per-clip racket target-velocity
        # boxes. None -> the training-default per-clip blade boxes (VEL_RANGE_PER_CLIP); the legacy
        # shared box (RACKET_VEL_*_RANGE) applies only if that is ALSO None (--pos-z-range legacy mode).
        # When set, it is a list indexed by clip_id; each entry is (x_range, y_range, z_range) and it
        # overrides both defaults. This ONLY changes which target velocity the MuJoCo RacketCommand
        # samples at eval time — it does NOT touch the policy, ONNX, rewards, or any training code.
        self.vel_ranges_per_clip = vel_ranges_per_clip
        # Optional per-clip POSITION boxes ((x_range, y_range, z_range) per clip, SIGNED y used
        # directly) — matches training's racket_pos_range_per_clip semantics and REPLACES the
        # fixed-plane + |y|-sign + z-range logic when set. Needed to evaluate blade-centered-box
        # checkpoints in-distribution (the built-in constants are the legacy fixed-plane ranges).
        self.pos_ranges_per_clip = pos_ranges_per_clip
        # state
        self.racket_target_pos_w = np.zeros(3)
        self.racket_target_vel_w = np.zeros(3)
        self.racket_target_normal_w = np.array([0.0, 1.0, 0.0])
        self.base_target_pos_w = np.zeros(2)
        self.swing_sign = 1.0
        self.time_to_strike = 0.0

    def _u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def resample(self, clip_id):
        """New swing: sample racket target (pos/vel), base target, swing sign — matches uniform mode."""
        o = self.origin
        # racket target position (world). Precedence: explicit per-clip boxes from the CLI/scoreboard
        # (--pos-range-per-clip / task-YAML auto-forward; training racket.pos_range_per_clip parity,
        # SIGNED y) > module-default current-generation blade boxes (POS_RANGE_PER_CLIP; disabled via
        # --pos-z-range) > legacy shared fixed-x-plane + |y|-per-clip-sign + z-range box.
        pos_boxes = self.pos_ranges_per_clip if self.pos_ranges_per_clip is not None else POS_RANGE_PER_CLIP
        if pos_boxes is not None:
            px_r, py_r, pz_r = pos_boxes[min(clip_id, len(pos_boxes) - 1)]
            self.racket_target_pos_w = np.array([o[0] + self._u(*px_r),
                                                 o[1] + self._u(*py_r),
                                                 o[2] + self._u(*pz_r)])
        else:
            px = o[0] + self._u(*RACKET_POS_X_RANGE)
            ymag = self._u(*RACKET_POS_Y_ABS_RANGE)
            fh_sign = -1.0 if FOREHAND_ON_NEGATIVE_Y else 1.0
            sign = fh_sign if clip_id == 0 else -fh_sign  # forehand clip0 on -y, backhand clip1 on +y
            py = o[1] + sign * ymag
            pz = o[2] + self._u(*RACKET_POS_Z_RANGE)
            self.racket_target_pos_w = np.array([px, py, pz])
        # racket target velocity (world): independent box sample. Precedence: explicit per-clip boxes
        # (--vel-range-per-clip / task-YAML auto-forward, or --eval-per-clip-vel-targets diagnostic) >
        # module-default per-clip blade boxes (VEL_RANGE_PER_CLIP) > the legacy shared box.
        if self.vel_ranges_per_clip is not None:
            vx_r, vy_r, vz_r = self.vel_ranges_per_clip[clip_id]
        elif VEL_RANGE_PER_CLIP is not None:
            vx_r, vy_r, vz_r = VEL_RANGE_PER_CLIP[min(clip_id, len(VEL_RANGE_PER_CLIP) - 1)]
        else:
            vx_r, vy_r, vz_r = RACKET_VEL_X_RANGE, RACKET_VEL_Y_RANGE, RACKET_VEL_Z_RANGE
        self.racket_target_vel_w = np.array([self._u(*vx_r),
                                             self._u(*vy_r),
                                             self._u(*vz_r)])
        # swing sign: clip 0 -> +1 (forehand), clip 1 -> -1 (backhand).
        self.swing_sign = 1.0 if clip_id == 0 else -1.0
        # target paddle normal = the per-clip reference face normal at strike (unified uniform mode).
        self.racket_target_normal_w = self.target_normal_per_clip[clip_id]
        # base target XY (world): spawn + weak Y coupling to racket target + small jitter.
        base_xy = o[:2].copy()
        racket_y_off = self.racket_target_pos_w[1] - o[1]
        base_xy[1] += float(np.clip(BASE_COUPLE_BLEND * racket_y_off,
                                    -BASE_COUPLE_MAX_OFFSET, BASE_COUPLE_MAX_OFFSET))
        base_xy[0] += self._u(*BASE_TARGET_X_RANGE)
        base_xy[1] += self._u(*BASE_TARGET_Y_RANGE)
        self.base_target_pos_w = base_xy

    def set_external_target(self, pos_w, vel_w, normal_w, clip_id):
        """Mode-B (venue-balls) target injection: same state writes as resample(), but with an
        externally computed (ball-demanded) pos/vel/normal instead of box draws. The base-target
        coupling is kept identical to resample() so the policy sees the same obs semantics."""
        o = self.origin
        self.racket_target_pos_w = np.asarray(pos_w, np.float64).copy()
        self.racket_target_vel_w = np.asarray(vel_w, np.float64).copy()
        self.racket_target_normal_w = np.asarray(normal_w, np.float64).copy()
        self.swing_sign = 1.0 if clip_id == 0 else -1.0
        base_xy = o[:2].copy()
        racket_y_off = self.racket_target_pos_w[1] - o[1]
        base_xy[1] += float(np.clip(BASE_COUPLE_BLEND * racket_y_off,
                                    -BASE_COUPLE_MAX_OFFSET, BASE_COUPLE_MAX_OFFSET))
        base_xy[0] += self._u(*BASE_TARGET_X_RANGE)
        base_xy[1] += self._u(*BASE_TARGET_Y_RANGE)
        self.base_target_pos_w = base_xy

    def update_strike_timing(self, clip_id, time_step):
        seg_start = self.seg_start[clip_id]
        seg_len = self.seg_len[clip_id]
        phase = STRIKE_PHASE_PER_CLIP[clip_id]
        strike_step = seg_start + int(round(phase * (seg_len - 1)))
        self.time_to_strike = (strike_step - time_step) * self.step_dt

    # --- base-relative observation projections (yaw-heading base frame) -------------------------
    def racket_target_pos_b(self, base_pos_w, base_quat_w):
        return quat_rotate_inverse(yaw_quat(base_quat_w), self.racket_target_pos_w - base_pos_w)

    def racket_target_pos_b_rel_fk(self, base_pos_w, base_quat_w, racket_pos_w):
        """175-D deploy_parity variant: racket target expressed in the yaw-heading base frame but
        RELATIVE TO THE CURRENT RACKET FK position instead of the base origin. The base position
        cancels (target - racket_fk), so this obs term is deploy-honest (no world base position).
        racket_pos_w = base_pos_w + R(base_quat_w) @ racket_pos_pelvis(q)."""
        return quat_rotate_inverse(yaw_quat(base_quat_w), self.racket_target_pos_w - racket_pos_w)

    def base_target_pos_b(self, base_pos_w, base_quat_w):
        delta = np.array([self.base_target_pos_w[0] - base_pos_w[0],
                          self.base_target_pos_w[1] - base_pos_w[1], 0.0])
        return quat_rotate_inverse(yaw_quat(base_quat_w), delta)[:2]


# =================================================================================================
# Observation builder. Two contracts, detected from the ONNX obs dim:
#   180-D (base)         : full BeyondMimic obs (has motion_anchor_pos_b + base_target_pos_b, and
#                          racket_target_pos_b is BASE-relative).
#   175-D (deploy_parity): DROPS motion_anchor_pos_b(3) and base_target_pos_b(2), and reframes
#                          racket_target_pos_b to be relative to the CURRENT RACKET FK (deploy-honest,
#                          no world base position leaks in). Everything else is byte-identical.
# =================================================================================================
def build_obs(refs, robot: MujocoRobot, racket: RacketCommand, last_action, default_q,
              deploy_parity=False, face_command=False):
    # robot base (pelvis = root) world pose
    base_pos_w = robot.body_pos(robot.pelvis_bid)
    base_quat_w = robot.body_quat(robot.pelvis_bid)
    # robot anchor (torso) world pose
    robot_anchor_pos_w = robot.body_pos(robot.torso_bid)
    robot_anchor_quat_w = robot.body_quat(robot.torso_bid)
    # reference anchor (torso) world pose, from ONNX side-outputs (body_names order)
    ref_anchor_pos_w = refs["body_pos_w"][ANCHOR_TRACKED_IDX]
    ref_anchor_quat_w = refs["body_quat_w"][ANCHOR_TRACKED_IDX]

    # 1. command (62): reference joint pos + vel
    command = np.concatenate([refs["joint_pos"], refs["joint_vel"]])
    # 2/3. motion_anchor_pos_b (3) + ori_b (6): ref anchor expressed in robot anchor frame
    pos_b, ori_q = subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w,
                                             ref_anchor_pos_w, ref_anchor_quat_w)
    ori_b6 = mat_from_quat(ori_q)[:, :2].reshape(-1)     # first 2 columns -> [R00,R01,R10,R11,R20,R21]
    # 4. base_ang_vel (3) body frame
    base_ang_vel = robot.pelvis_ang_vel_body()
    # 5/6. joint pos/vel rel (31 each), articulation order
    q = robot.q_artic()
    qd = robot.qd_artic()
    joint_pos_rel = q - default_q
    joint_vel_rel = qd
    # 7. last action (31)
    # 8. projected gravity (3) body frame
    proj_grav = robot.projected_gravity_body()
    # 11. racket_target_vel_w (3)
    racket_vel_w = racket.racket_target_vel_w
    # 12. time_to_strike (1)
    tts = np.array([racket.time_to_strike])
    # 13. swing_type (1)
    swing = np.array([racket.swing_sign])

    if deploy_parity:
        # 175-D deploy_parity: DROP motion_anchor_pos_b(3) and base_target_pos_b(2); reframe
        # racket_target_pos_b relative to the CURRENT racket FK.
        #   racket_pos_w = base_pos_w + R(base_quat_w) @ racket_pos_pelvis(q)   (q = current joints)
        racket_pos_w = base_pos_w + mat_from_quat(base_quat_w) @ racket_pos_pelvis(q)
        racket_tgt_b = racket.racket_target_pos_b_rel_fk(base_pos_w, base_quat_w, racket_pos_w)
        parts = [
            command, ori_b6, base_ang_vel, joint_pos_rel, joint_vel_rel,
            last_action, proj_grav, racket_tgt_b, racket_vel_w, tts, swing,
        ]
        if face_command:
            # stage-1 face-command tail: DEMANDED face normal (world; box mode = the clip
            # reference, venue mode = the per-ball StrikeSpec demand) + zero rho placeholder.
            parts.append(racket.racket_target_normal_w)
            parts.append(np.zeros(1))
        obs = np.concatenate(parts).astype(np.float64)
        want = 179 if face_command else 175
        assert obs.shape == (want,), f"obs dim {obs.shape} != {want} (deploy_parity)"
    else:
        # 9. base_target_pos_b (2)
        base_tgt_b = racket.base_target_pos_b(base_pos_w, base_quat_w)
        # 10. racket_target_pos_b (3) base-relative
        racket_tgt_b = racket.racket_target_pos_b(base_pos_w, base_quat_w)
        obs = np.concatenate([
            command, pos_b, ori_b6, base_ang_vel, joint_pos_rel, joint_vel_rel,
            last_action, proj_grav, base_tgt_b, racket_tgt_b, racket_vel_w, tts, swing,
        ]).astype(np.float64)
        assert obs.shape == (180,), f"obs dim {obs.shape} != 180"
    return obs, base_quat_w, robot_anchor_pos_w, robot_anchor_quat_w, ref_anchor_pos_w, ref_anchor_quat_w


# =================================================================================================
# Termination checks (replicate tracking_env_cfg terminations, single env)
# =================================================================================================
def body_pos_relative_w(ref_body_pos, ref_anchor_pos, ref_anchor_quat, robot_anchor_pos, robot_anchor_quat):
    """Reference body positions re-anchored to the robot's actual (xy, yaw); z from the reference anchor.
    Mirrors MotionCommand._update_command body_pos_relative_w."""
    delta_pos = robot_anchor_pos.copy()
    delta_pos[2] = ref_anchor_pos[2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
    out = np.zeros_like(ref_body_pos)
    for i in range(ref_body_pos.shape[0]):
        out[i] = delta_pos + quat_rotate(delta_ori, ref_body_pos[i] - ref_anchor_pos)
    return out


def check_terminations(refs, robot, robot_anchor_pos, robot_anchor_quat, ref_anchor_pos, ref_anchor_quat):
    reasons = []
    # anchor_pos (z only): |ref torso z - robot torso z| > 0.25
    if abs(ref_anchor_pos[2] - robot_anchor_pos[2]) > TERM_ANCHOR_POS_Z:
        reasons.append("anchor_pos")
    # anchor_ori: |proj_grav_z(ref) - proj_grav_z(robot)| > 0.8
    g = np.array([0.0, 0.0, -1.0])
    pg_ref = quat_rotate_inverse(ref_anchor_quat, g)[2]
    pg_rob = quat_rotate_inverse(robot_anchor_quat, g)[2]
    if abs(pg_ref - pg_rob) > TERM_ANCHOR_ORI:
        reasons.append("anchor_ori")
    # ee_body_pos (z only): any ee |z(ref_relative) - z(robot)| > 0.25
    ref_rel = body_pos_relative_w(refs["body_pos_w"], ref_anchor_pos, ref_anchor_quat,
                                  robot_anchor_pos, robot_anchor_quat)
    robot_tracked = robot.tracked_pos()
    ee_err_z = np.abs(ref_rel[EE_TRACKED_IDX, 2] - robot_tracked[EE_TRACKED_IDX, 2])
    if np.any(ee_err_z > TERM_EE_POS_Z):
        reasons.append("ee_body_pos")
    return reasons


# =================================================================================================
# Strike-success accumulator (Isaac exact-strike metric, replicated as a simple mean over exact
# frames — the converged value of Isaac's decay=0.99 EMA. Tracks overall + per swing-type clip).
# =================================================================================================
class StrikeAcc:
    def __init__(self):
        self.n = 0
        self.pos_err = self.vel_err = self.nrm_err = 0.0
        self.pos_pass = self.vel_pass = self.nrm_pass = self.comp = 0
        # speed-magnitude diagnostics (separate the "too slow / wrong direction" question)
        self.act_speed = self.tgt_speed = self.speed_err = 0.0   # ||v||, ||v_tgt||, ||v||-||v_tgt||
        # failure-mode counts (mutually-exclusive pos/vel breakdown + raw per-channel fails)
        self.pos_fail = self.vel_fail = self.nrm_fail = 0
        self.pos_only_fail = self.vel_only_fail = self.pos_and_vel_fail = 0

    def add(self, pos_err, vel_err, nrm_err_deg, act_speed, tgt_speed):
        pp = pos_err < STRIKE_POS_THRESH
        pv = vel_err < STRIKE_VEL_THRESH
        pn = nrm_err_deg < STRIKE_NORMAL_THRESH_DEG
        self.n += 1
        self.pos_err += pos_err; self.vel_err += vel_err; self.nrm_err += nrm_err_deg
        self.pos_pass += pp; self.vel_pass += pv; self.nrm_pass += pn; self.comp += (pp and pv and pn)
        self.act_speed += act_speed; self.tgt_speed += tgt_speed
        self.speed_err += (act_speed - tgt_speed)
        pf, vf, nf = (not pp), (not pv), (not pn)
        self.pos_fail += pf; self.vel_fail += vf; self.nrm_fail += nf
        self.pos_only_fail += (pf and not vf)
        self.vel_only_fail += (vf and not pf)
        self.pos_and_vel_fail += (pf and vf)

    def rate(self, k):
        """Mean of accumulator `k` over the strike samples (nan if none)."""
        return (getattr(self, k) / self.n) if self.n else float("nan")

    def count(self, k):
        """Raw integer count of accumulator `k` (0 if none)."""
        return int(getattr(self, k))


# =================================================================================================
# Mode-B (venue-balls) virtual-return accumulator: one add() per exact-strike frame.
# =================================================================================================
class VenueAcc:
    def __init__(self):
        self.n = 0
        self.contacted = self.landing_valid = self.on_opp = self.net_clear = self.landed_ok = 0
        self.land_errs = []          # ||achieved - intended|| for CONTACTED strikes w/ valid landing
        self.demanded_speed = 0.0    # |v_r| the spec demanded (target_speed)

    def add(self, ret, demanded_speed):
        self.n += 1
        self.contacted += ret.contacted
        self.demanded_speed += demanded_speed
        if ret.contacted:
            self.landing_valid += ret.landing_valid
            self.on_opp += ret.on_opponent
            self.net_clear += ret.net_clear
            self.landed_ok += ret.landed_ok
            if ret.landing_valid and not math.isnan(ret.land_err):
                self.land_errs.append(ret.land_err)

    def metrics(self):
        nan = float("nan")
        n_c = self.contacted
        return dict(
            n_strikes=self.n,
            contacted=n_c,
            contact_rate=(n_c / self.n) if self.n else nan,
            landing_valid_rate=(self.landing_valid / n_c) if n_c else nan,   # of contacted
            in_bounds_rate=(self.on_opp / n_c) if n_c else nan,              # of contacted
            net_clear_rate=(self.net_clear / n_c) if n_c else nan,           # of contacted
            landed_ok=self.landed_ok,
            # headline 回球成功率: legal return (contact + in-bounds + net clear) per strike CHANCE
            return_success_rate=(self.landed_ok / self.n) if self.n else nan,
            land_err_median=(float(np.median(self.land_errs)) if self.land_errs else nan),
            land_err_mean=(float(np.mean(self.land_errs)) if self.land_errs else nan),
            demanded_speed_mean=(self.demanded_speed / self.n) if self.n else nan,
        )


# =================================================================================================
# Viewer markers (VISUALIZATION ONLY — no physics, no collision, no reward, no observation effect).
# Drawn into the viewer's user scene each frame; they never touch model/data/qpos/obs/action.
# =================================================================================================
def _draw_markers(viewer, mujoco, racket, robot, ball_pos):
    scn = viewer.user_scn
    eye = np.eye(3).flatten()

    def add(pos, radius, rgba):
        i = scn.ngeom
        if i >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], float), np.asarray(pos, float),
            eye, np.asarray(rgba, np.float32))
        scn.ngeom = i + 1

    scn.ngeom = 0
    add(racket.racket_target_pos_w, 0.045, [0.10, 0.90, 0.10, 0.55])  # green  = racket TARGET point
    add(robot.racket_pos(),         0.035, [0.95, 0.20, 0.20, 0.95])  # red    = ACTUAL racket center
    if ball_pos is not None:
        add(ball_pos,               0.020, [1.00, 0.55, 0.00, 1.00])  # orange = visual-only incoming ball


# =================================================================================================
# Rollout for one noise scale
# =================================================================================================
def run_rollout(policy, robot, refs_table, seg_start, seg_len, num_clips, step_dt, decimation,
                noise_scale, std_vec, n_steps, max_ep_len, rng, csv_writer, mode_label,
                target_normal_per_clip, strike_csv_writer=None, viewer=None, realtime=True,
                vel_ranges_per_clip=None, pos_ranges_per_clip=None, df=None,
                reset_mode="teleport", hold_range=(0, 100), venue_sampler=None,
                switch_stress=0.0):
    racket = RacketCommand(seg_start, seg_len, step_dt, rng, target_normal_per_clip,
                           vel_ranges_per_clip=vel_ranges_per_clip,
                           pos_ranges_per_clip=pos_ranges_per_clip)
    strike = {"all": StrikeAcc(), "forehand": StrikeAcc(), "backhand": StrikeAcc()}
    multiswing = (reset_mode == "multiswing") and (df is None)
    # --- mode B (venue-balls): per-rollout accumulators + the current swing's sampled ball -------
    assert venue_sampler is None or df is None, "venue-balls + --deploy-faithful unsupported (v1)"
    venue = {"all": VenueAcc(), "forehand": VenueAcc(), "backhand": VenueAcc()}
    # mode-B counterfactual: same achieved kinematics, DEMANDED normal swapped in (see docstring).
    venue_cf = {"all": VenueAcc(), "forehand": VenueAcc(), "backhand": VenueAcc()}
    cur_venue_strike = [None]     # VenueStrike of the swing in flight (list = py2-style nonlocal)
    if venue_sampler is not None:
        venue_sampler.reset_counters()
    # --- switch-stress (deploy-parity mid-swing clip switch; see docstring) ----------------------
    stress = (switch_stress > 0.0) and (df is None)
    assert not (stress and venue_sampler is not None), "--switch-stress + venue-balls unsupported (v1)"
    assert not stress or multiswing, "--switch-stress needs the multiswing protocol"
    # per-swing provenance + per-rollout stress counters (inert when stress is off)
    swing_from_switch = False        # current swing was started by a mid-swing switch
    last_switch = {"step": None, "mid": False}   # most recent switch (for the 2 s fall window)
    surv_window = int(round(2.0 / step_dt))      # "survived the switch" horizon: 2 s = 100 steps
    sw = dict(n_switches=0, n_midswing=0, n_inhold=0, n_prestrike=0,
              falls_2s=0, falls_2s_midswing=0)
    strike_sw = {"clean": StrikeAcc(), "postswitch": StrikeAcc()}

    def sample_swing():
        """Pick the next swing's clip (+ ball, in venue mode). boxes mode draws the clip exactly
        like the legacy code (SAME rng call, byte-identical stream); venue mode is ball-first —
        the clip follows from the sampled ball's y side."""
        if venue_sampler is None:
            return int(rng.integers(0, num_clips)), None
        vs = venue_sampler.sample(rng)
        return vs.clip, vs

    def apply_target(c, vs):
        """Set the swing's racket target: legacy per-clip box resample (mode A, byte-identical) or
        the venue ball's StrikeSpec demand (mode B)."""
        if vs is None:
            racket.resample(c)
        else:
            racket.set_external_target(vs.target_pos_w, vs.target_vel_w, vs.target_normal_w, c)
            cur_venue_strike[0] = vs

    def sample_hold():
        """Pre-swing HOLD length (multiswing only): training freezes the reference at the swing's
        first frame for U[hold_range] control steps on EVERY resample (reset AND wrap). Teleport
        mode draws nothing so its RNG stream stays byte-identical to the legacy harness."""
        if not multiswing:
            return 0
        return int(rng.integers(int(hold_range[0]), int(hold_range[1]) + 1))

    def fresh_swing():
        """Sample a clip, set time_step to its start, ref-state-init the robot, resample racket target."""
        clip, vs = sample_swing()
        ts = int(seg_start[clip])
        r = refs_table[ts]
        robot.reset_to_reference(
            root_pos=r["body_pos_w"][ROOT_TRACKED_IDX], root_quat=r["body_quat_w"][ROOT_TRACKED_IDX],
            root_lin_w=r["body_lin_vel_w"][ROOT_TRACKED_IDX], root_ang_w=r["body_ang_vel_w"][ROOT_TRACKED_IDX],
            q_artic=r["joint_pos"])
        apply_target(clip, vs)
        racket.update_strike_timing(clip, ts)
        return clip, ts

    # ---------------------------------------------------------------------------------------------
    # --deploy-faithful machinery (df is None => this whole block is inert and the training-like
    # protocol runs unchanged). Mirrors pp_policy.hpp's single-swing/rest deploy logic:
    #   nominal-stand episode start (NEVER reference-state-init) -> hold the windup reference at
    #   seg_start with obs time_to_strike pinned at the per-clip in-training max -> advance the clip
    #   ONE frame per control step through its FULL length (final frames included) -> rest at the
    #   NEXT swing's windup (new clip + freshly resampled racket target, like a training resample)
    #   -> repeat. NO teleports ever; episodes end only on a REAL fall (tilt / root-height).
    # ---------------------------------------------------------------------------------------------
    def df_strike_step(c):
        return int(seg_start[c]) + int(round(STRIKE_PHASE_PER_CLIP[c] * (int(seg_len[c]) - 1)))

    dfs = None
    if df is not None:
        dfs = {
            "phase": "hold", "left": 0, "completed": True, "next_clip": 0,
            "swing_starts": [0] * num_clips, "swing_completions": [0] * num_clips,
            "fall_times_s": [],
        }

        def df_pick_clip():
            if df["clip_mode"] == "fh":
                return 0
            if df["clip_mode"] == "bh":
                return min(1, num_clips - 1)
            c = dfs["next_clip"] % num_clips          # "both": strict fh/bh alternation per swing
            dfs["next_clip"] += 1
            return c

        def df_new_swing(phase, steps_left):
            """Arm the NEXT swing: pick its clip, resample its racket target (exactly what a training
            resample would do), pin the clock at that clip's windup (seg_start => obs time_to_strike
            = the per-clip in-training max, matching deploy's tts clamp), hold for `steps_left`."""
            c = df_pick_clip()
            racket.resample(c)
            ts = int(seg_start[c])
            racket.update_strike_timing(c, ts)
            dfs["phase"] = phase
            dfs["left"] = int(steps_left)
            dfs["completed"] = True                    # no swing in flight yet
            return c, ts

        def df_start_episode():
            """Nominal stand: default_joint_pos + upright root at standing height, zero velocity —
            how the deployed robot enters MOTION. NEVER reference-state-init."""
            robot.reset_to_stand(df["stand_root_pos"], df["stand_root_quat"], policy.default_q)
            return df_new_swing("hold", df["hold_steps"])

        def df_fall_reasons():
            """Deploy-faithful fall = the training BALANCE terminations only (hope_env_cfg RealSensor:
            bad_orientation tilt > 0.7 rad OR root_height_below_minimum pelvis z < 0.5 m)."""
            reasons = []
            pg = robot.projected_gravity_body()
            tilt = math.acos(max(-1.0, min(1.0, -float(pg[2]))))   # angle from upright
            if tilt > DF_FALL_TILT_RAD:
                reasons.append("fall_tilt")
            if float(robot.body_pos(robot.pelvis_bid)[2]) < DF_FALL_ROOT_Z_MIN:
                reasons.append("fall_root_z")
            return reasons

    if df is None:
        clip, time_step = fresh_swing()
        hold_left = sample_hold()
    else:
        clip, time_step = df_start_episode()
        hold_left = 0
    last_action = np.zeros(31)
    ep_len = 0

    # accumulators
    ep_lengths, term_reasons = [], []
    n_term_early = n_timeout = 0
    roll_acc, pitch_acc, torquemax_acc, footc_acc, n_acc = 0.0, 0.0, 0.0, 0.0, 0
    racket_err_acc, racket_err_n = 0.0, 0        # mean over the ±strike_window_s gate
    racket_exact_acc, racket_exact_n = 0.0, 0    # pos err at the EXACT strike frame (|tts| <= 0.5*step_dt)
    racket_velerr_acc = 0.0                       # racket vel err at the exact strike frame (Isaac bottleneck)
    fell = 0
    exact_tol = 0.5 * step_dt + 1e-6
    frame_clock = time.perf_counter()   # realtime pacing reference for the viewer (no effect headless)

    for step in range(n_steps):
        refs = refs_table[time_step]
        obs, base_quat_w, ra_pos, ra_quat, refa_pos, refa_quat = build_obs(
            refs, robot, racket, last_action, policy.default_q, deploy_parity=policy.deploy_parity,
            face_command=getattr(policy, "face_command", False))

        mean = policy.action(obs, time_step)
        action = mean if noise_scale <= 0.0 else mean + noise_scale * std_vec * rng.standard_normal(31)
        last_action = action.copy()

        target_q = policy.default_q + action * policy.action_scale
        tau = robot.apply_pd_and_step(target_q, policy.kp, policy.kd, decimation)
        ep_len += 1

        # --- viewer (visualization only; does not touch sim/obs/action) ---
        if viewer is not None:
            if not viewer.is_running():
                break
            # visual-only "incoming ball": approaches the target along +x, arriving at strike time
            # (venue mode: back-extrapolated along the SAMPLED ball velocity instead).
            if venue_sampler is not None and cur_venue_strike[0] is not None:
                ball_pos = (racket.racket_target_pos_w
                            - cur_venue_strike[0].ball_vel_w * max(racket.time_to_strike, 0.0))
            else:
                ball_pos = (racket.racket_target_pos_w
                            + np.array([1.0, 0.0, 0.0]) * max(racket.time_to_strike, 0.0) * 3.0)
            _draw_markers(viewer, robot.mj, racket, robot, ball_pos)
            viewer.sync()
            if realtime:
                frame_clock += step_dt
                sleep_t = frame_clock - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)

        # --- metrics (post-step state) ---
        bq = robot.body_quat(robot.pelvis_bid)
        # roll/pitch from pelvis quat (deg)
        w, x, y, z = bq
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        roll_d, pitch_d = math.degrees(roll), math.degrees(pitch)
        torque_max = float(np.max(np.abs(tau)))
        foot_c = robot.foot_contact_frac()
        roll_acc += abs(roll_d); pitch_acc += abs(pitch_d)
        torquemax_acc += torque_max; footc_acc += foot_c; n_acc += 1
        # racket tracking error inside the strike window
        racket_err = float("nan")
        if abs(racket.time_to_strike) <= STRIKE_WINDOW_S:
            racket_err = float(np.linalg.norm(robot.racket_pos() - racket.racket_target_pos_w))
            racket_err_acc += racket_err; racket_err_n += 1
            if abs(racket.time_to_strike) <= exact_tol:   # exact strike frame (Isaac exact_strike mask)
                # All three error channels at the exact strike frame (same arithmetic as Isaac):
                #   pos_err = ||racket_pos_w - target_pos_w||
                #   vel_err = ||racket_lin_vel_w - target_vel_w||   (full 3-vec norm)
                #   nrm_err = acos(dot(unit normals)) in degrees
                pos_err = racket_err
                act_vel_w = robot.racket_lin_vel_w()
                tgt_vel_w = racket.racket_target_vel_w
                vel_err = float(np.linalg.norm(act_vel_w - tgt_vel_w))
                act_speed = float(np.linalg.norm(act_vel_w))
                tgt_speed = float(np.linalg.norm(tgt_vel_w))
                nrm = robot.racket_normal_w()
                tgt_nrm = racket.racket_target_normal_w
                cos_a = float(np.clip(np.dot(nrm, tgt_nrm), -1.0, 1.0))
                nrm_err_deg = math.degrees(math.acos(cos_a))
                strike["all"].add(pos_err, vel_err, nrm_err_deg, act_speed, tgt_speed)
                strike[CLIP_NAMES[clip]].add(pos_err, vel_err, nrm_err_deg, act_speed, tgt_speed)
                if stress:   # hit-rate split: swings born of a switch vs clean swings
                    strike_sw["postswitch" if swing_from_switch else "clean"].add(
                        pos_err, vel_err, nrm_err_deg, act_speed, tgt_speed)
                racket_exact_acc += pos_err; racket_exact_n += 1
                racket_velerr_acc += vel_err
                # --- mode B: virtual return of the ACHIEVED racket state vs the SAMPLED ball ---
                venue_extra = []
                if venue_sampler is not None and cur_venue_strike[0] is not None:
                    vs = cur_venue_strike[0]
                    ret = venue_sampler.score_return(
                        vs, racket_pos_w=robot.racket_pos(), racket_vel_w=act_vel_w,
                        racket_normal_w=nrm, pos_err=pos_err)
                    venue["all"].add(ret, tgt_speed)
                    venue[CLIP_NAMES[clip]].add(ret, tgt_speed)
                    # COUNTERFACTUAL: same achieved pos/vel/pos_err, DEMANDED normal swapped in —
                    # isolates the normal channel (deterministic rescore, no RNG involved).
                    ret_cf = venue_sampler.score_return(
                        vs, racket_pos_w=robot.racket_pos(), racket_vel_w=act_vel_w,
                        racket_normal_w=vs.target_normal_w, pos_err=pos_err)
                    venue_cf["all"].add(ret_cf, tgt_speed)
                    venue_cf[CLIP_NAMES[clip]].add(ret_cf, tgt_speed)
                    lx = "" if math.isnan(ret.landing_xy[0]) else f"{ret.landing_xy[0]:.4f}"
                    ly = "" if math.isnan(ret.landing_xy[1]) else f"{ret.landing_xy[1]:.4f}"
                    lerr = "" if math.isnan(ret.land_err) else f"{ret.land_err:.4f}"
                    cx = "" if math.isnan(ret_cf.landing_xy[0]) else f"{ret_cf.landing_xy[0]:.4f}"
                    cy = "" if math.isnan(ret_cf.landing_xy[1]) else f"{ret_cf.landing_xy[1]:.4f}"
                    cerr = "" if math.isnan(ret_cf.land_err) else f"{ret_cf.land_err:.4f}"
                    venue_extra = [
                        f"{vs.ball_vel_w[0]:.4f}", f"{vs.ball_vel_w[1]:.4f}", f"{vs.ball_vel_w[2]:.4f}",
                        f"{vs.ball_spin_w[0]:.4f}", f"{vs.ball_spin_w[1]:.4f}", f"{vs.ball_spin_w[2]:.4f}",
                        int(ret.contacted),
                        f"{vs.intended_landing_xy[0]:.4f}", f"{vs.intended_landing_xy[1]:.4f}",
                        lx, ly, int(ret.landed_ok), lerr, int(ret.net_clear),
                        int(ret_cf.contacted), cx, cy, int(ret_cf.landed_ok), cerr,
                        int(ret_cf.net_clear),
                    ]
                # --- per-strike CSV row (one line per exact-strike sample) ---
                if strike_csv_writer is not None:
                    pp = pos_err < STRIKE_POS_THRESH
                    pv = vel_err < STRIKE_VEL_THRESH
                    pn = nrm_err_deg < STRIKE_NORMAL_THRESH_DEG
                    racket_pos_w = robot.racket_pos()
                    tgt_pos_w = racket.racket_target_pos_w
                    base_pos_w = robot.body_pos(robot.pelvis_bid)
                    strike_csv_writer.writerow([
                        mode_label, step, len(ep_lengths), CLIP_NAMES[clip], f"{racket.swing_sign:+.0f}",
                        f"{racket.time_to_strike:.4f}",
                        f"{pos_err:.4f}", f"{vel_err:.4f}", f"{nrm_err_deg:.3f}",
                        int(pp), int(pv), int(pn), int(pp and pv and pn),
                        f"{act_vel_w[0]:.4f}", f"{act_vel_w[1]:.4f}", f"{act_vel_w[2]:.4f}",
                        f"{tgt_vel_w[0]:.4f}", f"{tgt_vel_w[1]:.4f}", f"{tgt_vel_w[2]:.4f}",
                        f"{act_speed:.4f}", f"{tgt_speed:.4f}",
                        f"{racket_pos_w[0]:.4f}", f"{racket_pos_w[1]:.4f}", f"{racket_pos_w[2]:.4f}",
                        f"{tgt_pos_w[0]:.4f}", f"{tgt_pos_w[1]:.4f}", f"{tgt_pos_w[2]:.4f}",
                        f"{base_pos_w[0]:.4f}", f"{base_pos_w[1]:.4f}", f"{base_pos_w[2]:.4f}",
                    ] + venue_extra + ([int(swing_from_switch)] if stress else []))

        # --- terminations ---
        if df is None:
            # training-like: tracking-guard resets + 10 s timeout. Under --switch-stress the
            # tracking guards are OFF (the reference jump fires them spuriously; the question
            # is deploy falls) — balance terminations + timeout only.
            reasons = [] if stress else check_terminations(refs, robot, ra_pos, ra_quat,
                                                           refa_pos, refa_quat)
            if multiswing:
                # HOPEDeployParityTerminationsCfg adds ABSOLUTE balance terminations on top of the
                # inherited tracking guards — a real fall/sink ends the episode regardless of clip.
                pg = robot.projected_gravity_body()
                tilt = math.acos(max(-1.0, min(1.0, -float(pg[2]))))
                if tilt > DF_FALL_TILT_RAD:
                    reasons.append("fall_tilt")
                if float(robot.body_pos(robot.pelvis_bid)[2]) < DF_FALL_ROOT_Z_MIN:
                    reasons.append("fall_root_z")
            timeout = ep_len >= max_ep_len
        else:
            # deploy-faithful: only REAL falls end an episode (no tracking guards, no timeout)
            reasons = df_fall_reasons()
            timeout = False
        terminated = len(reasons) > 0

        if csv_writer is not None:
            csv_writer.writerow([
                mode_label, step, time_step, clip, f"{racket.swing_sign:+.0f}",
                f"{racket.time_to_strike:.4f}", f"{roll_d:.3f}", f"{pitch_d:.3f}",
                f"{ra_pos[0]:.4f}", f"{ra_pos[1]:.4f}", f"{ra_pos[2]:.4f}", f"{refa_pos[2]:.4f}",
                f"{np.mean(np.abs(target_q)):.4f}", f"{np.max(np.abs(target_q)):.4f}",
                f"{torque_max:.2f}", f"{foot_c:.2f}",
                ("" if math.isnan(racket_err) else f"{racket_err:.4f}"),
                f"{float(np.linalg.norm(robot.racket_lin_vel_w())):.4f}",
                ep_len, ("|".join(reasons) if terminated else ("timeout" if timeout else "")),
            ])

        if terminated or timeout:
            ep_lengths.append(ep_len)
            if terminated:
                n_term_early += 1; fell += 1; term_reasons.extend(reasons)
                # switch-stress fall attribution: a fall within 2 s of the most recent switch
                # counts against that switch ("did the mid-swing abort knock it over").
                if stress and last_switch["step"] is not None \
                        and (step - last_switch["step"]) <= surv_window:
                    sw["falls_2s"] += 1
                    if last_switch["mid"]:
                        sw["falls_2s_midswing"] += 1
            else:
                n_timeout += 1
            if df is not None:
                dfs["fall_times_s"].append(ep_len * step_dt)   # time-to-fall from episode start
            ep_len = 0
            if df is None:
                clip, time_step = fresh_swing()
                hold_left = sample_hold()
            else:
                clip, time_step = df_start_episode()   # fresh nominal stand — NEVER ref-state-init
            last_action = np.zeros(31)
            swing_from_switch = False
            last_switch = {"step": None, "mid": False}
            continue

        if df is None:
            # --- advance the motion clock; wrap within the env's current segment (multi-swing per episode) ---
            wrapped = False
            if multiswing and hold_left > 0:
                # Pre-swing HOLD (training parity, MotionCommand._update_command): the reference
                # clock is FROZEN at the swing's first frame ("the ball is not here yet") and
                # time_to_strike stays pinned at its per-clip max. The robot keeps being simulated.
                hold_left -= 1
            else:
                time_step += 1
                seg_end = int(seg_start[clip]) + int(seg_len[clip])
                if time_step >= seg_end:
                    # clip wrap mid-episode: sample the next swing + resample its target. Teleport
                    # mode = legacy RSI generation (Isaac wrap_teleport=true): ref-state-init the
                    # robot + zero last_action. Multiswing mode = current generation
                    # (wrap_teleport=false): NO teleport — the policy physically carries the body
                    # into the new swing's windup during the pre-swing hold; last_action persists
                    # (training only zeroes it on true episode resets). ep_len is NOT reset either
                    # way (the episode continues across swings until a fall/timeout).
                    wrapped = True
                    swing_from_switch = False       # a natural wrap starts a CLEAN swing
                    clip, vs = sample_swing()
                    time_step = int(seg_start[clip])
                    if not multiswing:
                        r = refs_table[time_step]
                        robot.reset_to_reference(
                            root_pos=r["body_pos_w"][ROOT_TRACKED_IDX], root_quat=r["body_quat_w"][ROOT_TRACKED_IDX],
                            root_lin_w=r["body_lin_vel_w"][ROOT_TRACKED_IDX], root_ang_w=r["body_ang_vel_w"][ROOT_TRACKED_IDX],
                            q_artic=r["joint_pos"])
                        last_action = np.zeros(31)
                    apply_target(clip, vs)
                    hold_left = sample_hold()
            # --- switch-stress injection (deploy-parity mid-swing clip switch; the commands.py
            # clip_switch_prob semantics): per-step Bernoulli, suppressed on a step that already
            # wrapped (training masks sw[wrap_ids]=False); HELD swings can switch too (the
            # planner may change its mind while waiting). Routes through the SAME resample path
            # as a wrap — uniform new clip, windup frame, fresh hold + target — and the robot's
            # physical state is untouched.
            if stress and not wrapped and float(rng.uniform(0.0, 1.0)) < switch_stress:
                mid = (hold_left == 0)
                sw["n_switches"] += 1
                sw["n_midswing" if mid else "n_inhold"] += 1
                if racket.time_to_strike > exact_tol:
                    sw["n_prestrike"] += 1          # aborted BEFORE its strike -> strike lost
                last_switch = {"step": step, "mid": mid}
                swing_from_switch = True
                clip, vs = sample_swing()
                time_step = int(seg_start[clip])
                apply_target(clip, vs)
                hold_left = sample_hold()
        else:
            # --- deploy-faithful swing schedule: hold -> play the WHOLE clip once -> rest -> repeat.
            # NO teleports; last_action carries across swings (the deployed policy runs continuously).
            if dfs["phase"] == "swing":
                if not dfs["completed"] and time_step >= df_strike_step(clip):
                    dfs["completed"] = True                       # reached the strike frame ALIVE
                    dfs["swing_completions"][clip] += 1
                if time_step >= int(seg_start[clip]) + int(seg_len[clip]) - 1:
                    # final clip frame has been played -> rest at the NEXT swing's windup
                    clip, time_step = df_new_swing("rest", df["rest_steps"])
                else:
                    time_step += 1
            else:  # "hold" (episode start) or "rest" (between swings): clock pinned at the windup
                dfs["left"] -= 1
                if dfs["left"] <= 0:
                    dfs["phase"] = "swing"
                    dfs["swing_starts"][clip] += 1
                    dfs["completed"] = False
                    time_step += 1                                # first advancing frame after windup
        racket.update_strike_timing(clip, time_step)

    total_term = n_term_early + n_timeout
    from collections import Counter
    rc = Counter(term_reasons)

    def clip_metrics(acc):
        """Full per-clip strike breakdown (means + raw failure-mode counts)."""
        return dict(
            n_strikes=acc.n,
            strike_composite_success_exact=acc.rate("comp"),
            strike_pos_pass_exact=acc.rate("pos_pass"),
            strike_vel_pass_exact=acc.rate("vel_pass"),
            strike_normal_pass_exact=acc.rate("nrm_pass"),
            racket_pos_err_exact=acc.rate("pos_err"),
            racket_vel_err_exact=acc.rate("vel_err"),          # full 3-vec ||actual-target|| mean
            racket_normal_err_exact=acc.rate("nrm_err"),
            actual_speed_exact=acc.rate("act_speed"),
            target_speed_exact=acc.rate("tgt_speed"),
            speed_error_scalar=acc.rate("speed_err"),          # mean(||v|| - ||v_tgt||)
            full_vel_vec_err=acc.rate("vel_err"),              # same as racket_vel_err_exact, by name
            pos_fail=acc.count("pos_fail"),
            vel_fail=acc.count("vel_fail"),
            normal_fail=acc.count("nrm_fail"),
            pos_only_fail=acc.count("pos_only_fail"),
            vel_only_fail=acc.count("vel_only_fail"),
            pos_and_vel_fail=acc.count("pos_and_vel_fail"),
        )

    out = dict(
        mode=mode_label, noise_scale=noise_scale,
        mean_ep_len=(sum(ep_lengths) / len(ep_lengths)) if ep_lengths else float("nan"),
        n_episodes=len(ep_lengths), n_term_early=n_term_early, n_timeout=n_timeout,
        terminated_rate=(n_term_early / total_term) if total_term else float("nan"),
        term_breakdown=dict(rc), fell=fell,
        base_roll_deg=roll_acc / max(n_acc, 1), base_pitch_deg=pitch_acc / max(n_acc, 1),
        torque_max_mean=torquemax_acc / max(n_acc, 1), foot_contact_frac=footc_acc / max(n_acc, 1),
        racket_pos_err_strike=(racket_err_acc / racket_err_n) if racket_err_n else float("nan"),
        racket_pos_err_exact=(racket_exact_acc / racket_exact_n) if racket_exact_n else float("nan"),
        racket_vel_err_exact=(racket_velerr_acc / racket_exact_n) if racket_exact_n else float("nan"),
        n_strikes=racket_exact_n,
        # --- full Isaac-matching strike composite metrics (overall + fh/bh) ---
        strike_composite_success_exact=strike["all"].rate("comp"),
        strike_pos_pass_exact=strike["all"].rate("pos_pass"),
        strike_vel_pass_exact=strike["all"].rate("vel_pass"),
        strike_normal_pass_exact=strike["all"].rate("nrm_pass"),
        racket_normal_err_exact=strike["all"].rate("nrm_err"),
        strike_composite_forehand=strike["forehand"].rate("comp"),
        strike_composite_backhand=strike["backhand"].rate("comp"),
        n_strikes_fh=strike["forehand"].n, n_strikes_bh=strike["backhand"].n,
        # --- full per-clip breakdown dicts (the diagnostic the report is built from) ---
        clip_all=clip_metrics(strike["all"]),
        clip_forehand=clip_metrics(strike["forehand"]),
        clip_backhand=clip_metrics(strike["backhand"]),
    )
    if venue_sampler is not None:
        out["venue"] = dict(
            all=venue["all"].metrics(),
            forehand=venue["forehand"].metrics(),
            backhand=venue["backhand"].metrics(),
            cf_all=venue_cf["all"].metrics(),
            cf_forehand=venue_cf["forehand"].metrics(),
            cf_backhand=venue_cf["backhand"].metrics(),
            sampler=venue_sampler.counters(),
        )
    if stress:
        clean, post = strike_sw["clean"], strike_sw["postswitch"]
        n_sw = sw["n_switches"]
        out["switch_stress"] = dict(
            p=switch_stress,
            n_switches=n_sw, n_midswing=sw["n_midswing"], n_inhold=sw["n_inhold"],
            n_prestrike_aborts=sw["n_prestrike"],
            falls=fell,
            falls_within_2s=sw["falls_2s"], falls_within_2s_midswing=sw["falls_2s_midswing"],
            survival_2s=(1.0 - sw["falls_2s"] / n_sw) if n_sw else float("nan"),
            survival_2s_midswing=(1.0 - sw["falls_2s_midswing"] / sw["n_midswing"])
                                 if sw["n_midswing"] else float("nan"),
            strikes_clean=clean.n, composite_clean=clean.rate("comp"),
            pos_pass_clean=clean.rate("pos_pass"), vel_pass_clean=clean.rate("vel_pass"),
            nrm_pass_clean=clean.rate("nrm_pass"),
            strikes_postswitch=post.n, composite_postswitch=post.rate("comp"),
            pos_pass_postswitch=post.rate("pos_pass"), vel_pass_postswitch=post.rate("vel_pass"),
            nrm_pass_postswitch=post.rate("nrm_pass"),
        )
    if df is not None:
        starts, comps = dfs["swing_starts"], dfs["swing_completions"]
        tot_s, tot_c = sum(starts), sum(comps)
        ftimes = dfs["fall_times_s"]
        dfd = dict(
            swing_starts=tot_s, swing_completions=tot_c,
            # UNCONDITIONAL completion rate: swings interrupted by a fall count as started-not-completed.
            completion_rate=(tot_c / tot_s) if tot_s else float("nan"),
            falls=fell,
            mean_time_to_fall_s=(sum(ftimes) / len(ftimes)) if ftimes else float("nan"),
            min_time_to_fall_s=(min(ftimes) if ftimes else float("nan")),
            fall_times_s=[round(t, 2) for t in ftimes],
        )
        for c in range(num_clips):
            nm = CLIP_NAMES.get(c, f"clip{c}")
            dfd[f"swing_starts_{nm}"] = starts[c]
            dfd[f"swing_completions_{nm}"] = comps[c]
            dfd[f"completion_rate_{nm}"] = (comps[c] / starts[c]) if starts[c] else float("nan")
        out["df"] = dfd
    return out


# =================================================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    wbt = os.path.dirname(here)                 # whole_body_tracking/
    repo = os.path.dirname(os.path.dirname(wbt))  # HOPE/ (whole_body_tracking is hope_training/whole_body_tracking)
    default_run = os.path.join(
        wbt, "logs/rsl_rl/agibot_a3_hope/2026-06-27_18-14-06_basecouple03_resume")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", default=os.path.join(default_run, "exported/policy.onnx"))
    p.add_argument("--std", default=os.path.join(default_run, "exported/learned_std.npy"),
                   help="learned_std.npy sidecar for the dither mode (31,). Optional if only noise_scale=0.")
    p.add_argument("--mjcf", default=os.path.join(
        repo, "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"))
    p.add_argument("--motion-files", nargs="+", default=[
        os.path.join(wbt, "logs/rsl_rl/eval_motion/fh.npz"),
        os.path.join(wbt, "logs/rsl_rl/eval_motion/bh.npz")],
        help="motion clips in TRAINING order (clip0=forehand, clip1=backhand). Used for segment lengths.")
    p.add_argument("--noise-scales", nargs="+", type=float, default=[0.0, 0.05])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--sim-dt", type=float, default=0.005, help="MuJoCo physics dt (Isaac used 0.005)")
    p.add_argument("--decimation", type=int, default=4, help="physics substeps per 50 Hz control step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-passive", action="store_true",
                   help="keep MJCF joint damping+frictionloss (harder, less faithful to Isaac)")
    p.add_argument("--pd-mode", choices=["explicit", "implicit"], default="explicit",
                   help="explicit: torque=kp*e-kd*qd. implicit: kp torque + kd as passive damping via "
                        "MuJoCo implicitfast (matches Isaac's ImplicitActuator; less fast-swing undershoot).")
    p.add_argument("--out-dir", default=None, help="where to write the CSV (default: ONNX run dir)")
    p.add_argument("--viewer", action="store_true",
                   help="launch the MuJoCo passive viewer to watch the robot (keeps all metric/CSV "
                        "behavior). Adds visualization-only markers: green=racket target, red=actual "
                        "racket, orange=incoming ball. Does NOT affect physics/obs/action/rewards.")
    p.add_argument("--no-realtime", action="store_true",
                   help="with --viewer, run as fast as possible instead of pacing to ~real time.")
    # --- DIAGNOSTIC (eval-only): per-clip racket target-velocity sampling ----------------------------
    # The trained policy uses a single clip-INDEPENDENT velocity box for both swings (target_mode:
    # uniform). That box (mean |v| ~2.7 m/s) fits the forehand but overshoots the backhand (achievable
    # ~2.0-2.2 m/s), so backhand vel_pass is low. This flag lets us TEST, with NO retrain and NO change
    # to the policy/ONNX/rewards, whether sampling a lower backhand target velocity recovers backhand
    # vel_pass/composite — i.e. whether per-clip velocity targets are worth adding to TRAINING later.
    p.add_argument("--eval-per-clip-vel-targets", action="store_true",
                   help="DIAGNOSTIC: sample DIFFERENT racket target-velocity boxes for forehand vs "
                        "backhand at eval time (forehand unchanged; backhand lowered). Eval-only — does "
                        "NOT change the policy, ONNX, rewards, or any training config.")
    p.add_argument("--fh-vel-x-range", nargs=2, type=float, default=[1.5, 3.5],
                   help="forehand target vel x range (only with --eval-per-clip-vel-targets)")
    p.add_argument("--fh-vel-y-range", nargs=2, type=float, default=[-1.0, 1.0],
                   help="forehand target vel y range (only with --eval-per-clip-vel-targets)")
    p.add_argument("--fh-vel-z-range", nargs=2, type=float, default=[0.0, 1.5],
                   help="forehand target vel z range (only with --eval-per-clip-vel-targets)")
    p.add_argument("--bh-vel-x-range", nargs=2, type=float, default=[1.2, 2.4],
                   help="backhand target vel x range (only with --eval-per-clip-vel-targets)")
    p.add_argument("--bh-vel-y-range", nargs=2, type=float, default=[-1.0, 1.0],
                   help="backhand target vel y range (only with --eval-per-clip-vel-targets)")
    p.add_argument("--bh-vel-z-range", nargs=2, type=float, default=[0.0, 1.2],
                   help="backhand target vel z range (only with --eval-per-clip-vel-targets)")
    # DIAGNOSTIC (eval-only): override the strike phase and/or target z-range to MATCH a model trained
    # with a different config (e.g. a backhand strike_phase fix). The defaults mirror the training YAML;
    # when you eval a model trained with strike_phase_per_clip=[0.36,0.50] you MUST pass the same here, or
    # the eval measures the wrong frame. Does NOT touch the policy/ONNX/rewards.
    p.add_argument("--ee-term-z", type=float, default=None,
                   help="override the ee_body_pos termination z-threshold (training default 0.25 m). "
                        "ee_body_pos is a TRAINING reset guard, not a deployment condition; raise it "
                        "(e.g. 100) to let swings run past a loose wind-up and measure the true strike/"
                        "fall rate without the guard cutting episodes off mid-swing.")
    p.add_argument("--strike-phase-per-clip", nargs="+", type=float, default=None,
                   help="DIAGNOSTIC: override per-clip strike phase. Must match the trained model's "
                        "strike_phase_per_clip. Default: ONNX metadata clip_strike_phases when present "
                        "(all current exports incl. v2-blade 0.47/0.333), else the built-in legacy "
                        "(0.36, 0.50); pass 0.36 0.74 for model_32200-era backhand. A wrong phase "
                        "silently collapses strike metrics to ~0.")
    # Per-clip TARGET BOXES matching the trained task YAML (racket.pos_range_per_clip /
    # vel_range_per_clip). Explicit values here override the module-default blade boxes. 12 floats:
    #   fh_x_lo fh_x_hi fh_y_lo fh_y_hi fh_z_lo fh_z_hi bh_x_lo bh_x_hi bh_y_lo bh_y_hi bh_z_lo bh_z_hi
    p.add_argument("--pos-range-per-clip", nargs=12, type=float, default=None,
                   help="per-clip racket target POSITION boxes (see comment; matches training "
                        "racket.pos_range_per_clip; signed y).")
    p.add_argument("--vel-range-per-clip", nargs=12, type=float, default=None,
                   help="per-clip racket target VELOCITY boxes (compact alternative to "
                        "--eval-per-clip-vel-targets + six --fh/bh-vel-*-range args).")
    p.add_argument("--pos-z-range", nargs=2, type=float, default=None,
                   help="DIAGNOSTIC: override the LEGACY shared-box target z-range (e.g. 0.85 1.25). "
                        "NOTE: this disables the default per-clip blade pos AND vel boxes entirely "
                        "(full legacy shared-box generation in effect). Default: per-clip blade boxes.")
    # --- DEPLOY-FAITHFUL evaluation mode (default OFF; existing behavior byte-identical when off) --
    p.add_argument("--deploy-faithful", action="store_true",
                   help="evaluate with the DEPLOYED episode protocol (pp_policy.hpp single-swing/rest "
                        "logic) instead of the training-like one: start from a nominal stand "
                        "(default_joint_pos, XML 'stand' keyframe root, zero velocity), hold the windup "
                        "reference with time_to_strike pinned at the per-clip in-training max, advance "
                        "the clip ONE frame per control step through its FULL length, rest at the NEXT "
                        "swing's windup (fresh racket target), repeat. NO reference-state-init, NO clip-"
                        "wrap teleports, NO tracking-guard terminations, NO timeout — episodes end only "
                        "on a real fall (tilt > 0.7 rad or pelvis z < 0.5 m).")
    p.add_argument("--df-hold-steps", type=int, default=50,
                   help="[--deploy-faithful] control steps to hold the FIRST windup after the "
                        "nominal-stand episode start (50 = 1.0 s at 50 Hz).")
    p.add_argument("--df-rest-steps", type=int, default=75,
                   help="[--deploy-faithful] control steps to rest at the next swing's windup between "
                        "swings (75 = 1.5 s at 50 Hz).")
    p.add_argument("--df-clips", choices=["fh", "bh", "both"], default="both",
                   help="[--deploy-faithful] which clip(s) to swing: fh=forehand only, bh=backhand "
                        "only, both=strict forehand/backhand alternation per swing (default).")
    # --- P0 fix 2026-07-04 (see module docstring): obs normalization + episode protocol ------------
    p.add_argument("--obs-norm", default="auto",
                   help="empirical obs-normalization sidecar (obs_norm.npz with mean/std/eps from the "
                        "checkpoint's obs_norm_state_dict; scripts/make_std_sidecar.py writes it). "
                        "'auto' (default) = <onnx_dir>/obs_norm.npz when present. ALL training runs "
                        "use empirical_normalization=true but the exports bake the RAW actor, so "
                        "evaluating without the sidecar lobotomizes the policy (all-zero scores).")
    p.add_argument("--no-obs-norm", action="store_true",
                   help="feed RAW obs even if an obs_norm.npz sidecar exists (legacy/broken behavior; "
                        "only correct for an export that already bakes the normalizer in).")
    p.add_argument("--reset-mode", choices=["auto", "teleport", "multiswing"], default="auto",
                   help="clip-wrap protocol for the training-like rollout. teleport = legacy RSI "
                        "generation (ref-state-init at every clip wrap, wrap_teleport=true era). "
                        "multiswing = current generation (wrap_teleport=false): NO wrap teleports, "
                        "pre-swing hold (--hold-steps-range) with time_to_strike pinned, last_action "
                        "persists across wraps, plus the absolute balance terminations "
                        "(tilt/root-height) that DeployParity trains with. auto (default) = ONNX "
                        "metadata 'wrap_teleport' when present, else multiswing.")
    p.add_argument("--hold-steps-range", nargs=2, type=int, default=[0, 100],
                   help="[--reset-mode multiswing] pre-swing hold U[lo,hi] control steps at every "
                        "swing start (training MotionCommandCfg.hold_steps_range default 0 100).")
    # --- MODE B: distribution-driven realism eval (2026-07-04; see module docstring TARGET SOURCE
    # + scripts/venue_ball_sampler.py for frames/geometry/caveats). Default boxes = mode A,
    # byte-identical to the pre-existing behavior.
    p.add_argument("--target-source", choices=["boxes", "venue-balls"], default="boxes",
                   help="boxes (default): racket targets from the per-clip training boxes (mode A, "
                        "in-distribution). venue-balls (mode B): sample INCOMING BALLS from the "
                        "fitted venue matchlike distribution (configs/incoming_ball_venue.yaml), "
                        "derive the racket (pos,vel,normal) each ball demands via the StrikeSpec "
                        "inverse planner (hope_ws/src/hope_planner), score strikes as usual PLUS "
                        "the virtual return landing of contacted strikes (回球成功率).")
    p.add_argument("--venue-speed-budget", type=float, default=10.0,
                   help="[venue-balls] max |v_r| the StrikeSpec solve may demand (m/s); specs "
                        "beyond it are rejected+resampled. Default 10.0 = the hope_planner node's "
                        "racket_speed_budget config default (diagnostic, effectively uncapped).")
    p.add_argument("--venue-landing-x-range", nargs=2, type=float, default=None,
                   help="[venue-balls] landing-target x box on the opponent half, env frame. "
                        "Default: geometry-derived [net_x+0.3 (dink guard), far_x-0.2] = "
                        "[2.17, 3.04] with the training virtual table (near edge x=0.5).")
    p.add_argument("--venue-landing-y-range", nargs=2, type=float, default=[-0.5, 0.5],
                   help="[venue-balls] landing-target y box (env frame; table half-width 0.7625).")
    p.add_argument("--venue-table-near-x", type=float, default=None,
                   help="[venue-balls] near table edge x in the env frame (default 0.5 = training "
                        "vb_table_near_x; the robot stands 0.5 m behind its table end).")
    p.add_argument("--venue-table-surface-z", type=float, default=None,
                   help="[venue-balls] table surface height above the env origin (default 0.76 = "
                        "training vb_table_surface_z).")
    p.add_argument("--venue-fh-y-split", type=float, default=None,
                   help="[venue-balls] swing-side split on the ball's y: y < split -> forehand "
                        "(targets on -y), else backhand. Default -0.155 = midpoint between the "
                        "training forehand/backhand box y edges.")
    p.add_argument("--venue-max-tries", type=int, default=100,
                   help="[venue-balls] max ball redraws per swing when the StrikeSpec solve "
                        "rejects (no-converge / speed budget); exceeding it is FATAL.")
    p.add_argument("--venue-fixed-normal", action="store_true",
                   help="[venue-balls] PATH A (docs/motion_and_contract_v3.md §6): pin the "
                        "StrikeSpec face normal at the swing side's clip reference normal and "
                        "solve velocity only (solve_fixed_normal) — the planner adapts to the "
                        "policy's clip-locked face. The reported return_success_rate is the "
                        "ZERO-TRAINING deployment ceiling of the current policy + an adapted "
                        "planner. Expect more solve rejections (landing DOF given up).")
    # --- SWITCH-STRESS protocol (2026-07-05): R11/R11b's benefit ruler — see module docstring --
    p.add_argument("--switch-stress", type=float, default=0.0, metavar="P",
                   help="deploy-parity mid-swing clip-switch stress protocol (multiswing rollout "
                        "only; NOT with --deploy-faithful / venue-balls). Each control step, with "
                        "probability P, the reference clock aborts the swing the way the deploy "
                        "runner does when the planner changes its mind (commands.py "
                        "clip_switch_prob / pp_reference_clock.hpp): uniform new clip, windup "
                        "frame, fresh hold + racket target, robot state untouched. Tracking-guard "
                        "terminations are DISABLED while on (balance falls + timeout only). "
                        "Reports switches, falls, 2 s post-switch survival, and post-switch vs "
                        "clean-swing hit rates. 0.0 (default) = off, byte-identical baseline. "
                        "Reference: training dose 0.002/step ~ 24-28%%/swing; suggested stress "
                        "dose 0.01 ~ 75%%/swing.")
    args = p.parse_args()

    if args.venue_fixed_normal and args.target_source != "venue-balls":
        raise SystemExit("[FATAL] --venue-fixed-normal only means something with "
                         "--target-source venue-balls")
    if args.switch_stress > 0.0:
        if args.deploy_faithful:
            raise SystemExit("[FATAL] --switch-stress + --deploy-faithful is unsupported (v1): "
                             "the df swing scheduler owns its own clip clock.")
        if args.target_source == "venue-balls":
            raise SystemExit("[FATAL] --switch-stress + --target-source venue-balls is "
                             "unsupported (v1): one stressor per protocol.")

    # Apply eval-only overrides to the module globals BEFORE any precompute/rollout reads them.
    global STRIKE_PHASE_PER_CLIP, RACKET_POS_Z_RANGE, TERM_EE_POS_Z, POS_RANGE_PER_CLIP, VEL_RANGE_PER_CLIP
    if args.strike_phase_per_clip is not None:
        STRIKE_PHASE_PER_CLIP = tuple(args.strike_phase_per_clip)
        print(f"[mj-sim2sim] OVERRIDE strike_phase_per_clip -> {STRIKE_PHASE_PER_CLIP} (eval-only)")
    if args.ee_term_z is not None:
        TERM_EE_POS_Z = float(args.ee_term_z)
        print(f"[mj-sim2sim] OVERRIDE ee_body_pos termination z-threshold -> {TERM_EE_POS_Z} m "
              f"(training default 0.25; large value = deployment-realistic, no tracking-guard cutoff)")
    if args.pos_z_range is not None:
        # The z override belongs to the LEGACY shared box, so drop back to it entirely — pos AND vel
        # (a hybrid legacy-pos + per-clip-vel distribution matches no training generation).
        RACKET_POS_Z_RANGE = tuple(args.pos_z_range)
        POS_RANGE_PER_CLIP = None
        VEL_RANGE_PER_CLIP = None
        print(f"[mj-sim2sim] OVERRIDE pos_z_range -> {RACKET_POS_Z_RANGE} (eval-only; per-clip pos AND "
              f"vel boxes DISABLED, full legacy shared-box generation in effect)")

    step_dt = args.sim_dt * args.decimation
    assert abs(step_dt - 0.02) < 1e-9, f"control dt {step_dt} != 0.02 (50 Hz). adjust --sim-dt/--decimation"
    max_ep_len = int(round(10.0 / step_dt))   # 10 s episode -> 500 steps

    print(f"[mj-sim2sim] onnx={args.onnx}")
    print(f"[mj-sim2sim] mjcf={args.mjcf}")
    policy = OnnxPolicy(args.onnx, obs_norm=("off" if args.no_obs_norm else args.obs_norm))
    contract = ('deploy_parity + FACE COMMAND tail (demanded normal 3 + rho placeholder)'
                if getattr(policy, 'face_command', False) else
                ('deploy_parity: racket_target_pos_b relative to racket FK, no anchor_pos/base_target'
                 if policy.deploy_parity else 'base: full 180-D BeyondMimic obs'))
    print(f"[mj-sim2sim] obs_dim={policy.obs_dim} "
          f"({contract}) "
          f"joints={len(policy.joint_names)} "
          f"control={1/step_dt:.0f}Hz (sim_dt={args.sim_dt}, decim={args.decimation})")

    # --- obs normalization status (P0 fix #1) — a missing sidecar silently zeroes every metric for
    # a normalized-obs model, so make the state of this transform impossible to miss.
    if policy.obs_mean is not None:
        print(f"[mj-sim2sim] obs normalization: ON (sidecar {policy.obs_norm_path}; "
              f"(obs-mean)/(std+{policy.obs_eps:g}), mean|max|={np.abs(policy.obs_mean).max():.2f}, "
              f"std max={policy.obs_std.max():.2f})")
    elif args.no_obs_norm:
        print("[mj-sim2sim] obs normalization: OFF (--no-obs-norm)")
    else:
        print("[mj-sim2sim] WARNING: obs normalization sidecar NOT FOUND (<onnx_dir>/obs_norm.npz). "
              "All known training runs use empirical_normalization=true while the export bakes the "
              "RAW actor — without the sidecar such a model is fed unnormalized obs and scores ~0 "
              "with a staggering/early-termination pathology. Create it with "
              "scripts/make_std_sidecar.py --checkpoint <the model_<N>.pt the ONNX came from>.")

    # strike-phase resolution: CLI (handled above) > ONNX clip metadata > built-in legacy fallback.
    # Resolved (and printed ONCE, with its source) BEFORE any strike-frame precompute.
    if args.strike_phase_per_clip is None:
        if policy.clip_strike_phases:
            STRIKE_PHASE_PER_CLIP = policy.clip_strike_phases
            print(f"[mj-sim2sim] strike_phase_per_clip in effect: {STRIKE_PHASE_PER_CLIP} "
                  f"(from ONNX metadata clip_strike_phases)")
        else:
            print(f"[mj-sim2sim] strike_phase_per_clip in effect: {STRIKE_PHASE_PER_CLIP} "
                  f"(built-in legacy fallback — no clip_strike_phases in ONNX metadata; pass "
                  f"--strike-phase-per-clip to match the trained cfg if this is not a v1-clip model)")
    else:
        print(f"[mj-sim2sim] strike_phase_per_clip in effect: {STRIKE_PHASE_PER_CLIP} (CLI override; "
              f"must match the model's training YAML)")

    # --- episode/reset protocol resolution (P0 fix #2) --------------------------------------------
    reset_mode = args.reset_mode
    if reset_mode == "auto":
        md_wrap = getattr(policy, "wrap_teleport_meta", None)
        if md_wrap is not None:
            reset_mode = "teleport" if md_wrap else "multiswing"
            print(f"[mj-sim2sim] reset mode: {reset_mode} (from ONNX metadata wrap_teleport={md_wrap})")
        else:
            reset_mode = "multiswing"
            print("[mj-sim2sim] reset mode: multiswing (auto default — no episode-semantics metadata; "
                  "current generation trains wrap_teleport=false. Pass --reset-mode teleport for "
                  "legacy RSI-per-swing models.)")
    else:
        print(f"[mj-sim2sim] reset mode: {reset_mode} (CLI)")
    if reset_mode == "multiswing":
        print(f"[mj-sim2sim]   multiswing: no wrap teleports; pre-swing hold U{tuple(args.hold_steps_range)} "
              f"steps (ref frozen at windup, tts pinned); + balance terminations "
              f"(tilt>{DF_FALL_TILT_RAD} rad, pelvis z<{DF_FALL_ROOT_Z_MIN} m)")
    if args.switch_stress > 0.0:
        if reset_mode != "multiswing":
            raise SystemExit("[FATAL] --switch-stress needs the multiswing protocol (got "
                             f"reset_mode={reset_mode}); a teleport-era model has no deploy-"
                             "parity swing-to-swing carry to stress.")
        print(f"[mj-sim2sim] SWITCH-STRESS protocol: p={args.switch_stress}/step — mid-swing "
              f"clip switch (commands.py clip_switch semantics: uniform new clip, windup frame, "
              f"fresh hold + target, NO teleport). Tracking guards OFF (balance falls + timeout "
              f"only). Post-switch vs clean-swing hit rates reported.")

    # std sidecar (only needed if a noise_scale > 0 is requested)
    std_vec = None
    if any(s > 0 for s in args.noise_scales):
        if not os.path.isfile(args.std):
            raise SystemExit(f"[FATAL] dither mode requested but std sidecar not found: {args.std}\n"
                             f"        Create it from the checkpoint: np.save(.../learned_std.npy, "
                             f"torch.load(model.pt)['model_state_dict']['std'])")
        std_vec = np.load(args.std).astype(np.float64).reshape(-1)
        assert std_vec.shape == (31,), f"std sidecar shape {std_vec.shape} != (31,)"
        print(f"[mj-sim2sim] learned std: mean={std_vec.mean():.4f} min={std_vec.min():.4f} max={std_vec.max():.4f}")

    # segment lengths from the motion npz frame counts (clip0 then clip1, like MotionLoader)
    seg_len = []
    for f in args.motion_files:
        if not os.path.isfile(f):
            raise SystemExit(f"[FATAL] motion file not found: {f}")
        seg_len.append(int(np.load(f)["joint_pos"].shape[0]))
    seg_len = np.array(seg_len, int)
    seg_start = np.zeros(len(seg_len), int)
    if len(seg_len) > 1:
        seg_start[1:] = np.cumsum(seg_len)[:-1]
    num_clips = len(seg_len)
    T = int(seg_len.sum())
    print(f"[mj-sim2sim] motion: {num_clips} clips, seg_len={seg_len.tolist()}, "
          f"seg_start={seg_start.tolist()}, T={T}")
    if policy.clip_seg_lengths and tuple(seg_len.tolist()) != tuple(policy.clip_seg_lengths):
        print(f"[mj-sim2sim] WARNING: motion npz seg_len {tuple(seg_len.tolist())} != ONNX "
              f"clip_seg_lengths {tuple(policy.clip_seg_lengths)} — these are probably NOT the "
              f"clips this model was trained/exported with.")

    robot = MujocoRobot(args.mjcf, policy.joint_names, policy.body_names, args.sim_dt, args.keep_passive,
                        args.pd_mode, kd_for_implicit=policy.kd)
    print(f"[mj-sim2sim] PD mode: {args.pd_mode}"
          + ("  (kd as passive damping + implicitfast integrator)" if args.pd_mode == "implicit" else ""))

    # --- deploy-faithful config (nominal-stand root from the XML 'stand' keyframe when present) ---
    df_cfg = None
    if args.deploy_faithful:
        stand_root_pos = np.array([0.0, 0.0, 0.93])          # fallback: standing height, identity yaw
        stand_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        kid = robot.mj.mj_name2id(robot.model, robot.mj.mjtObj.mjOBJ_KEY, "stand")
        if kid >= 0:
            kq = np.asarray(robot.model.key_qpos[kid], np.float64)
            stand_root_pos = kq[0:3].copy()
            stand_root_quat = kq[3:7].copy() / np.linalg.norm(kq[3:7])
            stand_src = f"XML 'stand' keyframe (root z={stand_root_pos[2]:.4f})"
        else:
            stand_src = "fallback (root z=0.93, identity yaw)"
        df_cfg = dict(hold_steps=args.df_hold_steps, rest_steps=args.df_rest_steps,
                      clip_mode=args.df_clips, stand_root_pos=stand_root_pos,
                      stand_root_quat=stand_root_quat)
        print(f"[mj-sim2sim] DEPLOY-FAITHFUL mode: nominal-stand init from {stand_src}; "
              f"hold={args.df_hold_steps} rest={args.df_rest_steps} steps; clips={args.df_clips}; "
              f"NO teleports / NO tracking-guard terminations / NO timeout "
              f"(fall = tilt>{DF_FALL_TILT_RAD} rad or pelvis z<{DF_FALL_ROOT_Z_MIN} m)")

    # Precompute the reference table (refs depend only on time_step) -> one ONNX call per frame, once.
    refs_table = [policy.refs(ts) for ts in range(T)]

    # GROUNDING check (2026-07-03): this gate's target boxes (POS/VEL_RANGE_PER_CLIP) and the
    # deploy runner's scripted targets assume clips RE-GROUNDED to face +X (frame-0 pelvis yaw ~0,
    # scripts/reground_hope_frame.py). A raw clip (registry v4: yaw ~+82/+86 deg) trains a
    # TURN-AND-WALK policy that can pass THIS gate (true poses fed to obs = oracle localization)
    # yet fail on deploy under perfect_tracking, whose base obs cannot see the footwork.
    for c in range(num_clips):
        q0 = refs_table[int(seg_start[c])]["body_quat_w"][ROOT_TRACKED_IDX]
        yaw0 = math.degrees(math.atan2(2.0 * (q0[0] * q0[3] + q0[1] * q0[2]),
                                       1.0 - 2.0 * (q0[2] * q0[2] + q0[3] * q0[3])))
        flag = ("  ** NOT RE-GROUNDED — turn-and-walk policy; a PASS here does NOT clear "
                "perfect_tracking deploy (needs oracle/mocap) **" if abs(yaw0) > 10.0 else "")
        print(f"[mj-sim2sim] clip {c} baked frame-0 pelvis yaw = {yaw0:+.1f} deg{flag}")

    # Per-clip TARGET paddle normal (unified uniform mode): the imitated swing's reference face normal
    # at its strike frame = local +Y of the reference wrist(=racket) frame at strike_step.
    target_normal_per_clip = []
    for c in range(num_clips):
        strike_step = int(seg_start[c]) + int(round(STRIKE_PHASE_PER_CLIP[c] * (seg_len[c] - 1)))
        ref_wrist_quat = refs_table[strike_step]["body_quat_w"][WRIST_TRACKED_IDX]
        target_normal_per_clip.append(mat_from_quat(ref_wrist_quat)[:, MOUNT_NORMAL_AXIS] * MOUNT_NORMAL_SIGN)
    target_normal_per_clip = np.array(target_normal_per_clip)

    # --- MODE B (venue-balls) sampler: lazy import so mode A never needs hope_planner ----------
    venue_sampler = None
    if args.target_source == "venue-balls":
        if args.deploy_faithful:
            raise SystemExit("[FATAL] --target-source venue-balls + --deploy-faithful is "
                             "unsupported (v1): the df swing scheduler owns its own resample path.")
        import venue_ball_sampler as _vbs   # sibling module (scripts/ is on sys.path, top of file)
        kw = {}
        if args.venue_table_near_x is not None:
            kw["table_near_x"] = args.venue_table_near_x
        if args.venue_table_surface_z is not None:
            kw["table_surface_z"] = args.venue_table_surface_z
        if args.venue_fh_y_split is not None:
            kw["fh_y_split"] = args.venue_fh_y_split
        venue_sampler = _vbs.VenueBallSampler(
            repo_root=repo, ref_normal_per_clip=target_normal_per_clip, num_clips=num_clips,
            landing_x_range=args.venue_landing_x_range,
            landing_y_range=tuple(args.venue_landing_y_range),
            speed_budget=args.venue_speed_budget, max_tries=args.venue_max_tries,
            fixed_normal=args.venue_fixed_normal, **kw)
        print(f"[mj-sim2sim] MODE B — target source: VENUE BALLS "
              f"(spec mirrors {_vbs.VENUE_YAML_REL}, pooled matchlike)")
        if args.venue_fixed_normal:
            print("[mj-sim2sim]   FIXED-NORMAL inversion (path A): StrikeSpec normal PINNED at "
                  "the clip reference face; velocity-only solve. return_success_rate = the "
                  "zero-training ceiling of current policy + adapted planner.")
        print(f"[mj-sim2sim]   incoming ball: contact_pos(venue frame)={_vbs.VENUE_CONTACT_POS_Q10_Q90} "
              f"vel={_vbs.VENUE_VEL_BOX_MATCHLIKE} |spin|<= {_vbs.VENUE_SPIN_ABS_MAX} rad/s (isotropic)")
        print(f"[mj-sim2sim]   virtual table (env frame): near_x={venue_sampler.table_near_x} "
              f"net_x={venue_sampler.net_x:.2f} far_x={venue_sampler.far_x:.2f} "
              f"surface_z={venue_sampler.table_surface_z} (training vb parity)")
        print(f"[mj-sim2sim]   landing target box: x={venue_sampler.landing_x_range} "
              f"y={venue_sampler.landing_y_range}; speed budget {venue_sampler.speed_budget} m/s; "
              f"fh/bh split at ball y={venue_sampler.fh_y_split}")
        print(f"[mj-sim2sim]   contact box in env frame: "
              f"x=[{_vbs.VENUE_CONTACT_POS_Q10_Q90[0][0] + venue_sampler.net_x:.2f}, "
              f"{_vbs.VENUE_CONTACT_POS_Q10_Q90[0][1] + venue_sampler.net_x:.2f}] "
              f"z=[{_vbs.VENUE_CONTACT_POS_Q10_Q90[2][0] + venue_sampler.table_surface_z:.2f}, "
              f"{_vbs.VENUE_CONTACT_POS_Q10_Q90[2][1] + venue_sampler.table_surface_z:.2f}] — "
              f"NOTE: human-height contacts, mostly ABOVE the trained strike boxes (realism test)")
        print(f"[mj-sim2sim]   v1 caveat: independent box sampling; the venue correlations "
              f"(corr(vx,vz)=-0.44 etc.) are NOT enforced, only the sign structure (vx<0)")

    # DIAGNOSTIC: per-clip eval target-velocity boxes (clip 0 = forehand, clip 1 = backhand). None ->
    # faithful baseline (single training box for both clips). num_clips>2 reuse the backhand box.
    def _boxes12(vals):
        fh = ((vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]))
        bh = ((vals[6], vals[7]), (vals[8], vals[9]), (vals[10], vals[11]))
        return [fh if c == 0 else bh for c in range(num_clips)]

    vel_ranges_per_clip = None
    if venue_sampler is not None:
        # mode B: RacketCommand.resample() is never called — every target comes from the sampled
        # ball's StrikeSpec demand, so ALL box config (per-clip/legacy/CLI) is inert this run.
        print("[mj-sim2sim] per-clip target boxes: INERT (--target-source venue-balls; targets "
              "are ball-demanded via StrikeSpec)")
    elif args.vel_range_per_clip is not None:
        vel_ranges_per_clip = _boxes12(args.vel_range_per_clip)
        print(f"[mj-sim2sim] per-clip eval velocity boxes (training parity): "
              f"fh={vel_ranges_per_clip[0]} bh={vel_ranges_per_clip[-1]}")
    elif args.eval_per_clip_vel_targets:
        fh = (tuple(args.fh_vel_x_range), tuple(args.fh_vel_y_range), tuple(args.fh_vel_z_range))
        bh = (tuple(args.bh_vel_x_range), tuple(args.bh_vel_y_range), tuple(args.bh_vel_z_range))
        vel_ranges_per_clip = [fh if c == 0 else bh for c in range(num_clips)]
        print("[mj-sim2sim] per-clip eval velocity targets: ENABLED (DIAGNOSTIC, eval-only; "
              "policy/ONNX/rewards/training UNCHANGED)")
        print(f"[mj-sim2sim]   forehand vel box: x={fh[0]} y={fh[1]} z={fh[2]}")
        print(f"[mj-sim2sim]   backhand vel box: x={bh[0]} y={bh[1]} z={bh[2]}")
    elif VEL_RANGE_PER_CLIP is not None:
        print("[mj-sim2sim] target sampling: training-default PER-CLIP blade boxes (DeployParity "
              "2026-07-02 re-plane)")
        print(f"[mj-sim2sim]   forehand pos/vel: {POS_RANGE_PER_CLIP[0] if POS_RANGE_PER_CLIP else 'legacy shared'} / {VEL_RANGE_PER_CLIP[0]}")
        print(f"[mj-sim2sim]   backhand pos/vel: {POS_RANGE_PER_CLIP[1] if POS_RANGE_PER_CLIP else 'legacy shared'} / {VEL_RANGE_PER_CLIP[1]}")
    else:
        print("[mj-sim2sim] per-clip eval velocity targets: DISABLED (baseline — both clips use the "
              f"training box x={RACKET_VEL_X_RANGE} y={RACKET_VEL_Y_RANGE} z={RACKET_VEL_Z_RANGE})")

    pos_ranges_per_clip = None
    if venue_sampler is not None:
        pass    # mode B: position boxes inert too (single INERT line printed above)
    elif args.pos_range_per_clip is not None:
        pos_ranges_per_clip = _boxes12(args.pos_range_per_clip)
        print(f"[mj-sim2sim] per-clip eval position boxes (training parity): "
              f"fh={pos_ranges_per_clip[0]} bh={pos_ranges_per_clip[-1]}")
    else:
        print("[mj-sim2sim] per-clip eval position boxes: DISABLED (legacy fixed-plane sampling "
              f"x={RACKET_POS_X_RANGE} |y|={RACKET_POS_Y_ABS_RANGE} z={RACKET_POS_Z_RANGE})")

    out_dir = args.out_dir or default_run
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "mujoco_sim2sim_log.csv")
    csv_f = open(csv_path, "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["mode", "step", "time_step", "clip", "swing_sign", "time_to_strike",
                 "base_roll_deg", "base_pitch_deg", "torso_x", "torso_y", "torso_z", "ref_torso_z",
                 "target_q_mean_abs", "target_q_max_abs", "torque_max", "foot_contact_frac",
                 "racket_pos_err_strike", "racket_speed", "episode_len", "term_reason"])

    # Second CSV: one row per EXACT-strike sample (the fh/bh failure-breakdown raw data).
    strike_csv_path = os.path.join(out_dir, "mujoco_sim2sim_strikes.csv")
    strike_csv_f = open(strike_csv_path, "w", newline="")
    scw = csv.writer(strike_csv_f)
    strike_cols = [
        "mode", "step", "episode", "clip_name", "swing_type", "time_to_strike",
        "pos_err", "vel_err", "normal_err_deg",
        "pos_pass", "vel_pass", "normal_pass", "composite_pass",
        "actual_racket_vel_w_x", "actual_racket_vel_w_y", "actual_racket_vel_w_z",
        "target_racket_vel_w_x", "target_racket_vel_w_y", "target_racket_vel_w_z",
        "actual_speed", "target_speed",
        "racket_pos_w_x", "racket_pos_w_y", "racket_pos_w_z",
        "racket_target_pos_w_x", "racket_target_pos_w_y", "racket_target_pos_w_z",
        "base_pos_w_x", "base_pos_w_y", "base_pos_w_z",
    ]
    if venue_sampler is not None:
        # mode-B extras (only written in venue-balls mode -> mode A CSVs stay byte-identical).
        # ball state at strike == racket target pos columns above; landed_ok = contacted AND
        # valid landing AND on-opponent-half AND net cleared (the legal-return definition).
        # cf_* = the COUNTERFACTUAL rescore (same achieved pos/vel, DEMANDED normal swapped in).
        strike_cols += [
            "ball_v_x", "ball_v_y", "ball_v_z", "ball_w_x", "ball_w_y", "ball_w_z",
            "contacted", "intended_land_x", "intended_land_y",
            "achieved_land_x", "achieved_land_y", "landed_ok", "land_err_m", "net_clear",
            "cf_contacted", "cf_achieved_land_x", "cf_achieved_land_y", "cf_landed_ok",
            "cf_land_err_m", "cf_net_clear",
        ]
    if args.switch_stress > 0.0:
        # switch-stress extra (only in stress runs -> baseline CSVs stay byte-identical):
        # was this strike's swing started by a mid-swing switch?
        strike_cols += ["born_of_switch"]
    scw.writerow(strike_cols)

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        print("[mj-sim2sim] MuJoCo passive viewer launched "
              f"(realtime={'off' if args.no_realtime else 'on'}). Close the window to stop.")

    results = []
    for ns in args.noise_scales:
        rng = np.random.default_rng(args.seed)   # same seed per mode -> identical target/clip sequence
        print(f"\n[mj-sim2sim] >>> rollout noise_scale={ns}")
        res = run_rollout(policy, robot, refs_table, seg_start, seg_len, num_clips, step_dt,
                          args.decimation, ns, std_vec, args.steps, max_ep_len, rng, cw,
                          mode_label=f"ns={ns}", target_normal_per_clip=target_normal_per_clip,
                          strike_csv_writer=scw, viewer=viewer, realtime=not args.no_realtime,
                          vel_ranges_per_clip=vel_ranges_per_clip,
                          pos_ranges_per_clip=pos_ranges_per_clip, df=df_cfg,
                          reset_mode=reset_mode, hold_range=tuple(args.hold_steps_range),
                          venue_sampler=venue_sampler, switch_stress=args.switch_stress)
        results.append(res)
    csv_f.close()
    strike_csv_f.close()
    if viewer is not None:
        viewer.close()

    # ---- summary table ----
    print("\n" + "=" * 92)
    print(f"MuJoCo sim-to-sim | {os.path.basename(args.onnx)} | {args.steps} steps | seed {args.seed}")
    print("-" * 92)
    cols = [f"{r['mode']:>16s}" for r in results]
    print(f"{'metric':28s}" + "".join(cols))
    def row(label, key, fmt="{:16.4f}"):
        print(f"{label:28s}" + "".join(fmt.format(r[key]) if isinstance(r[key], float) else f"{str(r[key]):>16s}"
                                       for r in results))
    row("mean_episode_length", "mean_ep_len")
    row("terminated_rate", "terminated_rate")
    row("n_episodes", "n_episodes")
    row("early/timeout", "n_term_early", "{:16d}")
    print(f"{'  (timeouts)':28s}" + "".join(f"{r['n_timeout']:16d}" for r in results))
    row("base_roll_deg(|mean|)", "base_roll_deg")
    row("base_pitch_deg(|mean|)", "base_pitch_deg")
    row("torque_max(mean)", "torque_max_mean")
    row("foot_contact_frac", "foot_contact_frac")
    row("racket_err@window(m)", "racket_pos_err_strike")
    print("-" * 92)
    row("strike_composite_succ_exact", "strike_composite_success_exact")
    row("  forehand", "strike_composite_forehand")
    row("  backhand", "strike_composite_backhand")
    row("strike_pos_pass_exact", "strike_pos_pass_exact")
    row("strike_vel_pass_exact", "strike_vel_pass_exact")
    row("strike_normal_pass_exact", "strike_normal_pass_exact")
    row("racket_pos_err@exact(m)", "racket_pos_err_exact")
    row("racket_vel_err@exact(m/s)", "racket_vel_err_exact")
    row("racket_normal_err@exact(deg)", "racket_normal_err_exact")
    row("n_strikes (exact)", "n_strikes", "{:16d}")
    print(f"{'  fh/bh strikes':28s}" + "".join(f"{str(str(r['n_strikes_fh'])+'/'+str(r['n_strikes_bh'])):>16s}"
                                               for r in results))
    print("-" * 92)
    row("fell(count)", "fell", "{:16d}")
    print(f"{'term_breakdown':28s}" + "".join(f"{str(r['term_breakdown']):>16s}" for r in results))
    print("=" * 92)

    # ---- per-clip (forehand vs backhand) failure breakdown ----
    CLIP_ROWS = [
        ("n_strikes",                  "n_strikes",                       "{:16d}"),
        ("composite_succ_exact",       "strike_composite_success_exact",  "{:16.4f}"),
        ("pos_pass_exact",             "strike_pos_pass_exact",           "{:16.4f}"),
        ("vel_pass_exact",             "strike_vel_pass_exact",           "{:16.4f}"),
        ("normal_pass_exact",          "strike_normal_pass_exact",        "{:16.4f}"),
        ("racket_pos_err@exact(m)",    "racket_pos_err_exact",            "{:16.4f}"),
        ("racket_vel_err@exact(m/s)",  "racket_vel_err_exact",            "{:16.4f}"),
        ("full_vel_vec_err(m/s)",      "full_vel_vec_err",                "{:16.4f}"),
        ("racket_normal_err(deg)",     "racket_normal_err_exact",         "{:16.4f}"),
        ("actual_speed@exact(m/s)",    "actual_speed_exact",              "{:16.4f}"),
        ("target_speed@exact(m/s)",    "target_speed_exact",              "{:16.4f}"),
        ("speed_err_scalar(m/s)",      "speed_error_scalar",              "{:16.4f}"),
        ("pos_fail(count)",            "pos_fail",                        "{:16d}"),
        ("vel_fail(count)",            "vel_fail",                        "{:16d}"),
        ("normal_fail(count)",         "normal_fail",                     "{:16d}"),
        ("pos_only_fail(count)",       "pos_only_fail",                   "{:16d}"),
        ("vel_only_fail(count)",       "vel_only_fail",                   "{:16d}"),
        ("pos_and_vel_fail(count)",    "pos_and_vel_fail",                "{:16d}"),
    ]
    for clip_key, clip_label in (("clip_forehand", "FOREHAND"), ("clip_backhand", "BACKHAND")):
        print(f"\n{clip_label} per-clip breakdown")
        print("-" * 92)
        print(f"{'metric':28s}" + "".join(cols))
        for label, key, fmt in CLIP_ROWS:
            print(f"{label:28s}" + "".join(
                (fmt.format(r[clip_key][key]) if isinstance(r[clip_key][key], (int, float))
                 and not (isinstance(r[clip_key][key], float) and math.isnan(r[clip_key][key]))
                 else f"{str(r[clip_key][key]):>16s}")
                for r in results))
    print("=" * 92)

    # ---- MODE B (venue-balls) summary: distribution-driven realism + virtual return landing ----
    if venue_sampler is not None:
        print("\nMODE B — VENUE-BALL REALISM (incoming balls ~ venue matchlike spec; racket "
              "targets = StrikeSpec demands;\n         return landing = achieved racket state x "
              "sampled ball through the venue contact+flight model)")
        print("-" * 92)
        print(f"{'metric':28s}" + "".join(cols))

        def vrow(label, key, sub="all", fmt="{:16.4f}"):
            vals = []
            for r in results:
                v = r["venue"][sub].get(key, float("nan"))
                if isinstance(v, bool) or isinstance(v, (int, np.integer)):
                    vals.append(f"{int(v):16d}")
                elif isinstance(v, float):
                    vals.append(fmt.format(v) if not math.isnan(v) else f"{'nan':>16s}")
                else:
                    vals.append(f"{str(v):>16s}")
            print(f"{label:28s}" + "".join(vals))

        vrow("n_strikes (exact)", "n_strikes")
        vrow("ball_contacted (n)", "contacted")
        vrow("contact_rate", "contact_rate")
        vrow("landing_valid|contact", "landing_valid_rate")
        vrow("in_bounds|contact", "in_bounds_rate")
        vrow("net_clear|contact", "net_clear_rate")
        vrow("landed_ok (n)", "landed_ok")
        vrow("RETURN SUCCESS (回球成功率)", "return_success_rate")
        vrow("land_err_median(m)", "land_err_median")
        vrow("land_err_mean(m)", "land_err_mean")
        vrow("demanded_|v_r|_mean(m/s)", "demanded_speed_mean")
        for sub, nm in (("forehand", "fh"), ("backhand", "bh")):
            vrow(f"  {nm}: n_strikes", "n_strikes", sub=sub)
            vrow(f"  {nm}: contact_rate", "contact_rate", sub=sub)
            vrow(f"  {nm}: return_success", "return_success_rate", sub=sub)
        print("-" * 92)
        print("COUNTERFACTUAL — DEMANDED normal swapped into the achieved strike (same achieved "
              "pos/vel/pos_err):\n  CF >> actual return rate = the face-orientation channel "
              "ALONE fails the return (no normal channel in the obs contract)")
        vrow("CF contact_rate", "contact_rate", sub="cf_all")
        vrow("CF RETURN SUCCESS", "return_success_rate", sub="cf_all")
        vrow("CF land_err_median(m)", "land_err_median", sub="cf_all")
        for sub, nm in (("cf_forehand", "fh"), ("cf_backhand", "bh")):
            vrow(f"  {nm}: CF return_success", "return_success_rate", sub=sub)
        print("-" * 92)
        vrow("spec_solve_fails", "solve_fail", sub="sampler")
        vrow("sign_rejects", "sign_reject", sub="sampler")
        vrow("mean_solve_iters", "mean_solve_iters", sub="sampler")
        print("=" * 92)

    # ---- deploy-faithful swing-schedule report ----
    if args.deploy_faithful:
        print("\nDEPLOY-FAITHFUL report (nominal-stand start, hold -> full-clip swing -> rest; NO teleports)")
        print("-" * 92)
        print(f"{'metric':28s}" + "".join(cols))

        def dfrow(label, key, fmt="{:16.4f}"):
            vals = []
            for r in results:
                v = r["df"].get(key, float("nan"))
                if isinstance(v, bool) or isinstance(v, int):
                    vals.append(f"{v:16d}")
                elif isinstance(v, float):
                    vals.append(fmt.format(v) if not math.isnan(v) else f"{'nan':>16s}")
                else:
                    vals.append(f"{str(v):>16s}")
            print(f"{label:28s}" + "".join(vals))

        dfrow("swing_starts(total)", "swing_starts")
        dfrow("swing_completions(total)", "swing_completions")
        dfrow("completion_rate(uncond)", "completion_rate")
        for cname in ("forehand", "backhand"):
            dfrow(f"  {cname}_starts", f"swing_starts_{cname}")
            dfrow(f"  {cname}_completions", f"swing_completions_{cname}")
            dfrow(f"  {cname}_completion_rate", f"completion_rate_{cname}")
        dfrow("falls", "falls")
        dfrow("mean_time_to_fall(s)", "mean_time_to_fall_s")
        dfrow("min_time_to_fall(s)", "min_time_to_fall_s")
        print(f"{'fall_times_s':28s}" + "".join(f"{str(r['df']['fall_times_s']):>16s}" for r in results))
        print("=" * 92)

    # ---- switch-stress report: R11/R11b's benefit ruler (deploy-parity mid-swing aborts) ----
    if args.switch_stress > 0.0:
        print(f"\nSWITCH-STRESS report (p={args.switch_stress}/step mid-swing clip switch; "
              f"tracking guards OFF, balance falls only)")
        print("-" * 92)
        print(f"{'metric':28s}" + "".join(cols))

        def swrow(label, key, fmt="{:16.4f}"):
            vals = []
            for r in results:
                v = r["switch_stress"].get(key, float("nan"))
                if isinstance(v, bool) or isinstance(v, (int, np.integer)):
                    vals.append(f"{int(v):16d}")
                elif isinstance(v, float):
                    vals.append(fmt.format(v) if not math.isnan(v) else f"{'nan':>16s}")
                else:
                    vals.append(f"{str(v):>16s}")
            print(f"{label:28s}" + "".join(vals))

        swrow("switches(total)", "n_switches")
        swrow("  mid-swing switches", "n_midswing")
        swrow("  in-hold switches", "n_inhold")
        swrow("  pre-strike aborts", "n_prestrike_aborts")
        swrow("falls(total)", "falls")
        swrow("falls within 2s of switch", "falls_within_2s")
        swrow("SURVIVAL 2s post-switch", "survival_2s")
        swrow("  mid-swing only", "survival_2s_midswing")
        swrow("strikes on clean swings", "strikes_clean")
        swrow("  composite (clean)", "composite_clean")
        swrow("strikes on post-switch", "strikes_postswitch")
        swrow("  COMPOSITE (post-switch)", "composite_postswitch")
        swrow("  pos_pass (post-switch)", "pos_pass_postswitch")
        swrow("  vel_pass (post-switch)", "vel_pass_postswitch")
        swrow("  nrm_pass (post-switch)", "nrm_pass_postswitch")
        print("=" * 92)

    print(f"[mj-sim2sim] per-step CSV   -> {csv_path}")
    print(f"[mj-sim2sim] per-strike CSV -> {strike_csv_path}\n")


if __name__ == "__main__":
    main()
