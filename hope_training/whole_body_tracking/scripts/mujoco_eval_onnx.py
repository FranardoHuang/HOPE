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
POLICY CONTRACTS (resolved from input width AND exact ONNX metadata; 2026-07-10 audit)
=============================================================================================
Supported actor inputs:
  180 full BeyondMimic; 175 deploy_parity; 177 hitter_footwork; 179 face-command;
  181 face-command + station anchor; 110 hitter_pure (HITTER Table-I structure).
For 110, width alone is NOT accepted: actor_obs_contract/mode/term dims and this exact term order
must be present in metadata:
  base_ang_vel(3), joint_pos(31), joint_vel(31), actions(31), projected_gravity(3),
  base_forward_xy(2), base_target_delta_xy(2), racket_target_rel_base(3),
  racket_target_vel_w(3), time_to_strike(1).

Every ONNX has:
  inputs : obs[1,D] (float32), time_step[1,1] (float32)
  outputs: actions[1,31], joint_pos[1,31], joint_vel[1,31],
           body_pos_w[1,14,3], body_quat_w[1,14,4], body_lin_vel_w[1,14,3], body_ang_vel_w[1,14,3]
  -> outputs[1:] are the REFERENCE motion (the BeyondMimic clip) indexed by `time_step`. We use them
     as the single source of truth for the reference command + anchor (NO npz body-order guessing).

LEGACY FULL ACTOR OBSERVATION = 180D, concatenated in THIS order:
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
  explicit: torque = kp*(target_q-q)-kd*qdot; implicit: kp torque with kd inserted as MuJoCo
  implicitfast damping. Schema-3 auto-profile reads the per-joint actuator type; observation width
  is not actuator provenance. Native viscous damping and frictionloss are controlled independently.
  Non-zero Isaac/PhysX joint friction is dimensionless and load-dependent, so a direct numeric
  MuJoCo frictionloss mapping is explicitly diagnostic/inexact rather than formal parity.

CONTROL FREQUENCY: 50 Hz. Isaac used sim_dt=0.005 * decimation=4. We mirror that (--sim-dt/--decimation).

RESET/WRAP: metadata chooses teleport vs continuous multiswing and the hold distribution/reference.
Reference-state init now includes reference joint velocity. For 110-D Rally/V3, the evaluator still
does not reproduce the full true-reset stand/yaw mixture; it prints this limitation and must not be
cited as an exact reset-distribution match. Clip wraps in multiswing mode never teleport. Formal
BankExam is deliberately different: every immutable question starts from the same complete MJCF
named ``stand`` keyframe via ``mj_resetDataKeyframe`` with qvel/act/ctrl/last_action zero. Missing
``stand`` is fatal. Clip-start teacher-reference reset is retained only as an explicit inexact,
within-lineage diagnostic and is content-addressed per clip.

OBSERVATION NOISE: training adds small uniform obs corruption; deployment/sim-to-sim feeds CLEAN obs
  (the ONNX is deterministic). We feed clean obs and document it; sensor noise is a separate concern.

=============================================================================================
P0 FIX (2026-07-04): all-zero scores for the multiswing generation — TWO root causes
=============================================================================================
1. OBS NORMALIZATION (the actual all-zero bug, affects EVERY generation).
   Normalized runs consume
   (obs - mean) / (std + 0.01) with running stats stored in the checkpoint's obs_norm_state_dict.
   Historical exports sometimes contained the RAW actor; current native/standalone exporters write
   explicit `empirical_normalization` and `obs_norm_baked` truth. Feeding raw obs to a normalized actor
   lobotomizes the policy: in MuJoCo it staggered forward-right ~0.5 m on a canonical trajectory
   regardless of the sampled target (the 0.53 +/- 0.06 m "systematic offset"), tripped the
   anchor_pos/ee_body_pos tracking guards at ~52 steps, and only the backhand strike frame
   (44 steps in) was ever inside an episode — forehand (65 steps in) never evaluated. It could not
   even hold a nominal stand (deploy-faithful fell at ~1.1 s). Diagnosed with a perfect-tracking
   probe: with the robot PINNED to the reference each step, actions still exploded (|a| -> ~60)
   through the last-action feedback obs; with the normalizer applied they are sane.
   FIX: baked models skip sidecars; normalized raw models require the SAME-checkpoint obs_norm.npz;
   raw-trained models ignore stale sidecars. Contradictory/missing 110-D provenance is fatal.
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
   Outside formal BankExam, episode RESETS still reference-state-init at the sampled clip's first
   frame in both modes (training's dominant reset path); the 10 s timeout matches episode_length_s.
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
                          scored by scripts/virtual_return_scorer.py: fitted contact + the Isaac
                          metric's fixed-step RK4 drag/Magnus rollout to the ball-centre table
                          plane. The scorer source, venue YAML, full score spec, and their hashes
                          are bound into the execution contract and summary evidence. Therefore
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

CONTRACT ALIGNMENT (2026-07-08, fixE retrial follow-up — both flags DEFAULT OFF = the exam
  protocol stays byte-identical to every score already on the books):
  --qdes-clamp        clamp the decoded q_des to the MJCF soft joint limits (0.9 x range about the
                      midpoint == Isaac soft_joint_pos_limit_factor) before the PD. Training
                      (ClampedJointPositionAction, default ON since 2026-07-06) and the C++ deploy
                      runner (pp_joint_limits.hpp) BOTH clamp; unflagged, this exam is the only leg
                      of train/deploy/eval that feeds unclamped q_des — it can wrongly pass a
                      "clamp-rider" policy (q_des far past the limits buying torque the runner will
                      never grant) or wrongly kill a healthy policy that leans on the clamp.
                      Recommended ON for every new exam; state is printed in the report header.
  --hold-ref stand    multiswing pre-swing HOLD reference = READY STAND (joint refs = default_q,
                      ref vel = 0), the 2026-07-05+ training hold semantics (commands.py). The
                      default ("clip") keeps the legacy frozen-windup-frame reference the
                      pre-07-05 generations trained on. Examining a 07-05+ generation with "clip"
                      is the 07-07 incident shape (generation-mismatched hold reference) — pass
                      "stand" for those arms. Both toggles were adjudicated harmless on a healthy
                      arm (fixC six-cell retrial: composite unchanged) and are exam-side only.

STRIKING FACE (--mount-normal-sign-per-clip, 2026-07-09, franco 拍板"哪面拍子超前就是哪面"):
  per-clip ±1 face-sign table (clip order). A unified fh+bh policy strikes with OPPOSITE paddle
  faces; the legacy single-face (+Y) scoring pins the backhand face error at ~115-137° (the
  M3b/CF-swap=1.000 signature). Flag ON scores each swing's REAL striking face — achieved normal
  and the boxes-mode reference target normal both get the clip's sign (training
  racket.mount_normal_sign_per_clip semantics); venue/bank demanded normals are untouched. Signs
  are OFFLINE constants from the reference clip's contact frame (scripts/suggest_face_sign.py),
  never the live paddle velocity. Default OFF = byte-identical to every booked score.

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
import hashlib
import json
import math
import os
import time
import traceback
from collections import Counter

import numpy as np

# Validated pure-numpy racket forward-kinematics reference (racket pos in the pelvis frame from the
# 31 Isaac-order joint angles; validated to ~2e-6 m vs Isaac). Used ONLY by the 175-D deploy_parity
# obs path to reframe racket_target_pos_b relative to the current racket FK. Import from the sibling
# module (same scripts/ dir) so it works regardless of the caller's cwd.
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from racket_fk_ref import racket_pos_pelvis  # noqa: E402
import virtual_return_scorer as _virtual_return_scorer  # noqa: E402


VIRTUAL_RETURN_SCORER_RELATIVE_PATH = (
    "hope_training/whole_body_tracking/scripts/virtual_return_scorer.py"
)
BALL_PHYSICS_CONFIG_RELATIVE_PATH = "configs/ball_physics_venue.yaml"

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
# 177-D hitter_footwork base-station coupling (training base_couple_mode=reference_reach):
# station_xy = racket_target_xy - per-clip reference base->racket reach offset at strike + jitter.
# Jitter ranges mirror the hitter task YAML (racket.base_target_x_range / base_target_y_range).
HITTER_BASE_JITTER_X = (-0.05, 0.05)
HITTER_BASE_JITTER_Y = (-0.10, 0.10)
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
# 每 clip 击球面符号表(--mount-normal-sign-per-clip,2026-07-09 franco 拍板"哪面拍子超前就是哪面")。
# 病根:统一正反手策略用拍子相反的两面击球(正手=红面/+Y,反手=黑面/−Y),判卷只按 +Y 单面算拍面
# 误差会把反手钉在 ~115-137°(CF 换拍面=1.000 签名)。开表后判卷按该 clip 的实际击球面翻面再算:
# 实测法向和(boxes 模式的)参考目标法向同乘该 clip 符号——和训练侧 mount_normal_sign_per_clip 同语义。
# 符号是**离线固定常量**(参考 clip 触球帧的超前面,scripts/suggest_face_sign.py 算),绝不在运行时
# 用当前拍速动态定——训练早期拍面可能整个反着,动态符号会把"反面"合法化。None(默认)= 全部 clip 用
# 标量 MOUNT_NORMAL_SIGN,判卷行为与账上每一份成绩逐位一致。
MOUNT_NORMAL_SIGN_PER_CLIP = None


def face_sign_for_clip(clip_id):
    """该 clip 击球面的符号。表没开(None,默认)= 标量 MOUNT_NORMAL_SIGN,现役判卷行为逐位不变。"""
    if MOUNT_NORMAL_SIGN_PER_CLIP is None:
        return MOUNT_NORMAL_SIGN
    return MOUNT_NORMAL_SIGN_PER_CLIP[clip_id]
WRIST_TRACKED_IDX = TRACKED_BODIES.index("right_wrist_yaw_Link")   # 13; racket frame == this body's frame
CLIP_NAMES = {0: "forehand", 1: "backhand"}
# 110-D hitter_pure fallback geometry mirrors HOPEPingPongHitterPure.yaml. New 110-D exports are
# expected to carry these boxes in ONNX metadata; the fallback is only available through the
# explicit --allow-hitter-pure-defaults escape hatch so an old/stale recipe cannot be scored by
# accident. Position x/y are station-relative; z and all velocities are world-frame quantities.
HP_POS_RANGE_PER_CLIP = (
    ((0.51, 0.51), (-0.65, -0.15), (0.67, 0.97)),
    ((0.51, 0.51), (-0.05, 0.45), (0.88, 1.18)),
)
HP_VEL_RANGE_PER_CLIP = (
    ((1.05, 2.05), (0.96, 1.96), (0.31, 1.11)),
    ((1.61, 2.61), (-1.21, -0.21), (0.00, 0.71)),
)
HP_BASE_TARGET_RANGE = ((0.0, 0.0), (-0.40, 0.40))
# Frozen 110-D actor layout. Merely matching the total input width is not enough: feeding a
# different 110-column layout is a syntactically valid ONNX call and a scientifically invalid exam.
HITTER_PURE_OBS_NAMES = (
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "projected_gravity",
    "base_forward_xy",
    "base_target_delta_xy",
    "racket_target_rel_base",
    "racket_target_vel_w",
    "time_to_strike",
)
HITTER_PURE_OBS_DIMS = (3, 31, 31, 31, 3, 2, 2, 3, 3, 1)
DEPLOY_PARITY_OBS_NAMES = (
    "command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel",
    "actions", "projected_gravity", "racket_target_pos_b", "racket_target_vel_w",
    "time_to_strike", "swing_type",
)
DEPLOY_PARITY_OBS_DIMS = (62, 6, 3, 31, 31, 31, 3, 3, 3, 1, 1)
FULL_OBS_NAMES = (
    "command", "motion_anchor_pos_b", "motion_anchor_ori_b", "base_ang_vel",
    "joint_pos", "joint_vel", "actions", "projected_gravity", "base_target_pos_b",
    "racket_target_pos_b", "racket_target_vel_w", "time_to_strike", "swing_type",
)
FULL_OBS_DIMS = (62, 3, 6, 3, 31, 31, 31, 3, 2, 3, 3, 1, 1)
HITTER_FOOTWORK_OBS_NAMES = DEPLOY_PARITY_OBS_NAMES[:7] + (
    "base_target_pos_b",
) + DEPLOY_PARITY_OBS_NAMES[7:]
HITTER_FOOTWORK_OBS_DIMS = DEPLOY_PARITY_OBS_DIMS[:7] + (2,) + DEPLOY_PARITY_OBS_DIMS[7:]
FACE179_OBS_NAMES = DEPLOY_PARITY_OBS_NAMES + ("racket_target_normal_cmd",)
FACE179_OBS_DIMS = DEPLOY_PARITY_OBS_DIMS + (4,)
STATION181_OBS_NAMES = FACE179_OBS_NAMES + ("station_anchor_err_b",)
STATION181_OBS_DIMS = FACE179_OBS_DIMS + (2,)
FORMAL_ACTOR_CONTRACTS = {
    110: ("hitter_pure", "hitter_pure", HITTER_PURE_OBS_NAMES, HITTER_PURE_OBS_DIMS),
    175: ("deploy_parity", "deploy_parity", DEPLOY_PARITY_OBS_NAMES, DEPLOY_PARITY_OBS_DIMS),
    177: ("hitter_footwork", "hitter_footwork", HITTER_FOOTWORK_OBS_NAMES,
          HITTER_FOOTWORK_OBS_DIMS),
    179: ("deploy_parity_face179", "deploy_parity", FACE179_OBS_NAMES, FACE179_OBS_DIMS),
    180: ("full", "full", FULL_OBS_NAMES, FULL_OBS_DIMS),
    181: ("deploy_parity_station181", "deploy_parity", STATION181_OBS_NAMES,
          STATION181_OBS_DIMS),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalizer_state_sha256(mean, std, eps, count):
    """Canonical hash used by ``make_std_sidecar.py`` for the normalization payload."""
    digest = hashlib.sha256()
    digest.update(np.asarray(mean, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(std, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray([eps], dtype="<f4").tobytes())
    digest.update(np.asarray([count], dtype="<i8").tobytes())
    return digest.hexdigest()


def is_sha256(value):
    value = str(value).strip().lower()
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


FORMAL_READY_STATE_MODE = "mjcf_named_keyframe:stand:v1"
TEACHER_REFERENCE_READY_STATE_MODE = "teacher_reference_clip_start:v1"
CONTINUOUS_READY_STATE_MODE = "continuous_previous_question:v1"
DEPLOY_NOMINAL_READY_STATE_MODE = "deploy_nominal_stand:v1"


def canonical_contract_sha256(value):
    """Hash a finite JSON contract with stable key/order/float serialization."""
    payload = json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ready_state_snapshot_contract(*, mode, qpos, qvel, act, ctrl, last_action,
                                  time_s=0.0, qacc_warmstart=(), mocap_pos=(),
                                  mocap_quat=(), userdata=()):
    """Content identity of every state channel that survives into the first actor step.

    ``mj_resetDataKeyframe`` also clears solver warm-start state; that reset operation is part of
    the named mode.  The hash intentionally covers qpos/qvel/act/ctrl and the policy-side
    ``last_action`` because omitting any one of them allows two candidates to begin from different
    physical or observable states while claiming the same ready state.
    """
    arrays = {}
    for name, value in (
        ("time_s", [time_s]),
        ("qpos", qpos), ("qvel", qvel), ("act", act), ("ctrl", ctrl),
        ("qacc_warmstart", qacc_warmstart),
        ("mocap_pos", mocap_pos), ("mocap_quat", mocap_quat),
        ("userdata", userdata),
        ("last_action", last_action),
    ):
        source = np.asarray(value, dtype=np.float64)
        array = source.reshape(-1)
        if not np.isfinite(array).all():
            raise ValueError(f"ready state {name} must contain only finite values")
        arrays[name] = {"shape": list(source.shape), "values": array.tolist()}
    body = {
        "schema_version": 1,
        "kind": "hope_mujoco_ready_state",
        "mode": str(mode),
        "state": arrays,
    }
    body["sha256"] = canonical_contract_sha256(body)
    return body


def aggregate_teacher_reference_ready_contract(per_clip):
    """Bind every clip-specific reference reset without pretending it is a common state."""
    items = []
    for clip, contract in enumerate(per_clip):
        if contract.get("mode") != TEACHER_REFERENCE_READY_STATE_MODE:
            raise ValueError("teacher-reference aggregate contains a different ready-state mode")
        if not is_sha256(contract.get("sha256", "")):
            raise ValueError("teacher-reference aggregate contains an invalid state SHA")
        items.append({"clip": clip, "ready_state_sha256": contract["sha256"]})
    body = {
        "schema_version": 1,
        "kind": "hope_mujoco_ready_state_set",
        "mode": TEACHER_REFERENCE_READY_STATE_MODE,
        "per_clip": items,
    }
    body["sha256"] = canonical_contract_sha256(body)
    return body


def resolve_ready_state_mode(requested, *, target_source, deploy_faithful,
                             allow_inexact_contract):
    """Resolve CLI spelling into an auditable reset contract, failing closed on mixed protocols."""
    if requested not in ("auto", "stand-keyframe", "teacher-reference"):
        raise ValueError(f"unknown ready-state selection {requested!r}")
    if deploy_faithful:
        if requested != "auto":
            raise SystemExit(
                "[FATAL] --deploy-faithful owns the deploy nominal-stand reset; do not combine "
                "it with an incompatible --ready-state override"
            )
        return DEPLOY_NOMINAL_READY_STATE_MODE
    if requested == "auto":
        mode = (
            FORMAL_READY_STATE_MODE
            if target_source == "bank" else TEACHER_REFERENCE_READY_STATE_MODE
        )
    elif requested == "stand-keyframe":
        mode = FORMAL_READY_STATE_MODE
    else:
        mode = TEACHER_REFERENCE_READY_STATE_MODE
    if (
        target_source == "bank"
        and mode == TEACHER_REFERENCE_READY_STATE_MODE
        and not allow_inexact_contract
    ):
        raise SystemExit(
            "[FATAL] BankExam teacher-reference reset is candidate-dependent and diagnostic only; "
            "pass --allow-inexact-contract explicitly or use the default shared stand keyframe"
        )
    return mode


def training_hold_protocol_active(*, reset_mode, deploy_faithful_cfg, venue_sampler):
    """One source of truth for hold-aware tracking guards in main and rollout scopes."""

    multiswing = reset_mode == "multiswing" and deploy_faithful_cfg is None
    bank_schedule = bool(
        venue_sampler is not None and hasattr(venue_sampler, "schedule")
    )
    return bool(multiswing or bank_schedule)


def stage1_question_bank_module_path(wbt_root):
    """Return the standalone bank loader without importing the Isaac task package."""

    path = os.path.join(
        os.path.abspath(wbt_root),
        "source", "whole_body_tracking", "whole_body_tracking", "tasks", "tracking", "mdp",
        "stage1_question_bank.py",
    )
    if not os.path.isfile(path):
        raise SystemExit(f"[FATAL] standalone stage1 question-bank loader not found: {path}")
    return path


def materialize_ready_state_contract(robot, refs_table, seg_start, mode, action_dim=31):
    """Exercise the real reset path once and return its content-addressed contract."""
    zero_action = np.zeros(int(action_dim), dtype=np.float64)
    if mode == FORMAL_READY_STATE_MODE:
        robot.reset_to_named_keyframe("stand")
        return robot.ready_state_snapshot(mode, zero_action)
    if mode != TEACHER_REFERENCE_READY_STATE_MODE:
        raise ValueError(f"unsupported ready-state mode {mode!r}")
    per_clip = []
    for ts_value in np.asarray(seg_start, dtype=np.int64).reshape(-1):
        ts = int(ts_value)
        refs = refs_table[ts]
        robot.reset_to_reference(
            root_pos=refs["body_pos_w"][ROOT_TRACKED_IDX],
            root_quat=refs["body_quat_w"][ROOT_TRACKED_IDX],
            root_lin_w=refs["body_lin_vel_w"][ROOT_TRACKED_IDX],
            root_ang_w=refs["body_ang_vel_w"][ROOT_TRACKED_IDX],
            q_artic=refs["joint_pos"], qd_artic=refs["joint_vel"],
        )
        per_clip.append(robot.ready_state_snapshot(mode, zero_action))
    return aggregate_teacher_reference_ready_contract(per_clip)


def build_evaluation_execution_contract(
    *, robot, policy, mjcf_sha256, evaluator_sha256, ready_state_contract,
    sim_dt, decimation, pd_mode, passive_damping_mode, frictionloss_mode,
    qdes_clamp, one_question_reset, plant_semantics=None, protocol_semantics=None,
    virtual_return_scorer_contract=None,
):
    """Bind the actual common plant/protocol, excluding candidate identity and exam outcomes."""
    require_contract(is_sha256(mjcf_sha256), "execution contract requires a valid MJCF SHA")
    require_contract(is_sha256(evaluator_sha256), "execution contract requires evaluator SHA")
    body = {
        "schema_version": 1,
        "kind": "hope_mujoco_bank_execution_contract",
        "mjcf_sha256": mjcf_sha256,
        "evaluator_source_sha256": evaluator_sha256,
        "ready_state_mode": ready_state_contract["mode"],
        "ready_state_sha256": ready_state_contract["sha256"],
        "physics_step_dt_s": float(sim_dt),
        "control_decimation": int(decimation),
        "policy_step_dt_s": float(sim_dt) * int(decimation),
        "pd_mode": str(pd_mode),
        "passive_damping_mode": str(passive_damping_mode),
        "frictionloss_mode": str(frictionloss_mode),
        "qdes_clamp": bool(qdes_clamp),
        "one_question_one_reset": bool(one_question_reset),
        "obs_dim": int(policy.obs_dim),
        "joint_names": list(policy.joint_names),
        "default_joint_pos": np.asarray(policy.default_q, np.float64).tolist(),
        "action_scale": np.asarray(policy.action_scale, np.float64).tolist(),
        "joint_stiffness": np.asarray(policy.kp, np.float64).tolist(),
        "joint_damping": np.asarray(policy.kd, np.float64).tolist(),
        "qdes_joint_pos_limits": np.column_stack(
            (robot.soft_jnt_lo, robot.soft_jnt_hi)
        ).astype(np.float64).tolist(),
        "mujoco_actuated_dof_damping": np.asarray(
            robot.model.dof_damping[robot.vadr], np.float64
        ).tolist(),
        "mujoco_actuated_dof_frictionloss": np.asarray(
            robot.model.dof_frictionloss[robot.vadr], np.float64
        ).tolist(),
        "mujoco_actuated_dof_armature": np.asarray(
            robot.model.dof_armature[robot.vadr], np.float64
        ).tolist(),
        "mujoco_actuator_ctrlrange": np.column_stack(
            (robot.ctrl_lo, robot.ctrl_hi)
        ).astype(np.float64).tolist(),
        "mujoco_integrator": int(robot.model.opt.integrator),
        "resolved_joint_actuator_types": [str(value) for value in getattr(
            robot, "actuator_types", []
        )],
        "joint_velocity_limits": (
            np.asarray(robot.joint_velocity_limits, np.float64).tolist()
            if getattr(robot, "joint_velocity_limits", None) is not None else []
        ),
        "velocity_limit_proxy_allowed": bool(
            getattr(robot, "allow_velocity_limit_proxy", True)
        ),
        "plant_semantics": dict(plant_semantics or {}),
        "protocol_semantics": dict(protocol_semantics or {}),
    }
    if virtual_return_scorer_contract is not None:
        scorer_contract = dict(virtual_return_scorer_contract)
        scorer_sha = scorer_contract.pop("sha256", "")
        require_contract(
            is_sha256(scorer_sha)
            and canonical_contract_sha256(scorer_contract) == scorer_sha,
            "execution contract requires an internally consistent virtual-return scorer contract",
        )
        body["virtual_return_scorer_contract"] = dict(virtual_return_scorer_contract)
    body["sha256"] = canonical_contract_sha256(body)
    return body


def require_contract(condition, message):
    """Fail closed even under ``python -O`` (safety contracts must never use ``assert``)."""
    if not condition:
        raise SystemExit(f"[FATAL] {message}")


def json_ready(value):
    """Convert evaluator results to strict JSON (non-finite metrics become null)."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def build_virtual_return_scorer(repo_root, venue_sampler):
    """Create the production scorer and a portable, content-addressed scoring contract.

    ``venue_ball_sampler`` continues to own question/ball generation.  Scoring is evaluator-owned
    and reads the repository's venue-fit YAML explicitly, so an environment override cannot change
    a formal exam silently.  Paths are recorded separately in the result artifact; the hashed
    contract uses repository-relative names and content hashes so it is stable across machines.
    """

    repo_root = os.path.abspath(os.fspath(repo_root))
    source_path = os.path.abspath(_virtual_return_scorer.__file__)
    expected_source_path = os.path.join(repo_root, VIRTUAL_RETURN_SCORER_RELATIVE_PATH)
    config_path = os.path.join(repo_root, BALL_PHYSICS_CONFIG_RELATIVE_PATH)
    require_contract(
        os.path.isfile(source_path)
        and os.path.isfile(expected_source_path)
        and os.path.samefile(source_path, expected_source_path),
        "virtual-return scorer import did not resolve to the repository-owned production source: "
        f"imported={source_path}, expected={expected_source_path}",
    )
    require_contract(os.path.isfile(config_path), f"ball-physics config missing: {config_path}")

    params = _virtual_return_scorer.load_venue_params(config_path)
    spec = _virtual_return_scorer.VirtualReturnSpec(
        table_surface_z=float(venue_sampler.table_surface_z),
        net_x=float(venue_sampler.net_x),
        far_x=float(venue_sampler.far_x),
        half_width=float(venue_sampler.half_w),
        net_height=float(venue_sampler.table.net_height),
    )
    scorer = _virtual_return_scorer.VirtualReturnScorer(params, spec)
    physics_fields = (
        "k_d", "k_m", "g", "ball_radius", "inertia_coeff", "paddle_a_t", "paddle_b_t",
        "paddle_mu", "paddle_e_g1", "paddle_e_g2",
    )
    spec_fields = (
        "table_surface_z", "net_x", "far_x", "half_width", "net_height", "capture_radius",
        "min_approach_speed", "rollout_h", "rollout_steps",
    )
    contract = {
        "schema_version": 1,
        "kind": "hope_virtual_return_scorer_contract",
        "implementation": "virtual_return_scorer.VirtualReturnScorer",
        "source": {
            "repo_relative_path": VIRTUAL_RETURN_SCORER_RELATIVE_PATH,
            "sha256": sha256_file(source_path),
        },
        "physics_config": {
            "repo_relative_path": BALL_PHYSICS_CONFIG_RELATIVE_PATH,
            "sha256": sha256_file(config_path),
        },
        "physics_parameters": {
            name: float(getattr(params, name)) for name in physics_fields
        },
        "score_spec": {
            name: (int(getattr(spec, name)) if name == "rollout_steps"
                   else float(getattr(spec, name)))
            for name in spec_fields
        },
    }
    contract["sha256"] = canonical_contract_sha256(contract)
    return scorer, contract


def score_virtual_return(scorer, strike, racket_pos_w, racket_vel_w, racket_normal_w, pos_err):
    """Production adapter from a sampled exam strike to the authoritative scorer API."""

    return scorer.score(
        ball_vel=strike.ball_vel_w,
        ball_spin=strike.ball_spin_w,
        racket_pos=racket_pos_w,
        racket_vel=racket_vel_w,
        racket_normal=racket_normal_w,
        pos_err=pos_err,
        intended_landing_xy=strike.intended_landing_xy,
    )


def inspect_onnx_obs_normalization(onnx_path):
    """Inspect the saved graph's ``obs`` dataflow for a Sub->Div normalization prefix.

    Returns ``(True|False|None, explanation)``. ``None`` means the graph shape could not be proven;
    formal metadata schemas fail closed in that case. Metadata is deliberately not consulted here.
    """
    try:
        import onnx
    except ImportError:
        return None, "python package 'onnx' is unavailable"
    try:
        model = onnx.load(onnx_path)
    except Exception as exc:
        return None, f"cannot load ONNX graph: {exc}"

    consumers = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    passthrough = {"Identity", "Cast", "Reshape", "Flatten"}

    def reaches_div(names):
        queue = list(names)
        seen_names = set()
        while queue:
            name = queue.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            for node in consumers.get(name, ()):
                if node.op_type == "Div" and name in node.input:
                    return True
                if node.op_type in passthrough:
                    queue.extend(node.output)
        return False

    queue = ["obs"]
    seen_names = set()
    reached_linear = False
    while queue:
        name = queue.pop()
        if name in seen_names:
            continue
        seen_names.add(name)
        for node in consumers.get(name, ()):
            if node.op_type == "Sub" and name in node.input:
                if reaches_div(node.output):
                    return True, "graph obs path contains Sub->Div before actor"
                return None, "graph obs path contains Sub without a following Div"
            if node.op_type in ("Gemm", "MatMul"):
                reached_linear = True
            elif node.op_type in passthrough:
                queue.extend(node.output)
            else:
                return None, f"unrecognized first obs-path op {node.op_type}"
    if reached_linear:
        return False, "graph obs path reaches actor linear layer without Sub->Div"
    return None, "could not trace obs input to normalization or actor linear layer"


def post_step_time_to_strike(time_to_strike, step_dt, clock_advances):
    """Return the strike clock paired with the state *after* one control step.

    The actor consumes the observation at the current motion frame, then MuJoCo advances the
    physical state.  Isaac's command manager advances the motion clock before it computes the
    post-physics racket metrics, so an advancing swing must be graded at ``tts - step_dt``.  A
    held/resting swing keeps its clock pinned.  Keeping this arithmetic in one pure helper makes
    the otherwise easy-to-reintroduce one-control-step grading offset directly testable.
    """
    tts = float(time_to_strike)
    dt = float(step_dt)
    if not math.isfinite(tts) or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"finite time_to_strike and positive step_dt required, got {tts}, {dt}")
    return tts - dt if bool(clock_advances) else tts


def reset_sampler_for_paired_rollout(sampler):
    """Reset counters and rewind an immutable BankExam paper for a paired rollout column."""
    if sampler is None:
        return
    sampler.reset_counters()
    if hasattr(sampler, "rewind_schedule"):
        sampler.rewind_schedule()


def paired_rollout_rngs(seed):
    """Return independent target/schedule and action-noise RNG streams for one rollout column."""
    target_seed, noise_seed = np.random.SeedSequence(int(seed)).spawn(2)
    return np.random.default_rng(target_seed), np.random.default_rng(noise_seed)


def formal_bank_step_cap(schedule, seg_len, max_ep_len):
    """Conservative control-step cap that can finish every finite schedule item."""
    if not schedule or int(max_ep_len) <= 0:
        raise ValueError("non-empty schedule and positive episode cap required")
    lengths = np.asarray(seg_len, dtype=np.int64)
    per_attempt = max(
        max(int(max_ep_len), int(item.hold_steps) + int(lengths[item.clip]) + 2)
        for item in schedule
    )
    return len(schedule) * per_attempt


def attempt_ledger_flags(reason, details, *, scheduled_exam):
    """Canonical phase-1 attempt eligibility/censor/fall classification."""
    reason = str(reason)
    details = tuple(str(value) for value in details)
    censored = reason.startswith("truncated")
    physical_fall = any(value in ("fall_tilt", "fall_root_z") for value in details)
    return {
        "censored": censored,
        "eligible": bool(scheduled_exam and not censored),
        "physical_fall": physical_fall,
        "guard_reset": bool(reason.startswith("fall") and not physical_fall),
    }


def summarize_attempt_records(records, num_clips):
    """Build unconditional attempt metrics, including deaths before the strike frame."""

    def group(rows):
        n = len(rows)
        n_exact = sum(bool(row.get("exact", False)) for row in rows)
        n_composite = sum(bool(row.get("exact_composite", False)) for row in rows)
        return dict(
            n_attempts=n,
            n_reached_exact=n_exact,
            n_composite=n_composite,
            exact_reach_rate=(n_exact / n) if n else float("nan"),
            composite_rate_per_attempt=(n_composite / n) if n else float("nan"),
            composite_rate_given_exact=(n_composite / n_exact) if n_exact else float("nan"),
            finalize_reason_counts=dict(Counter(str(row["reason"]) for row in rows)),
        )

    summary = group(records)
    summary["per_clip"] = {
        CLIP_NAMES.get(c, f"clip_{c}"): group([row for row in records if int(row["clip"]) == c])
        for c in range(int(num_clips))
    }
    return summary
# --qdes-clamp soft-limit factor: Isaac ArticulationCfg soft_joint_pos_limit_factor (robots/
# agibot_a3.py) — the training ClampedJointPositionAction clamps to soft_joint_pos_limits, which
# Isaac derives by shrinking each joint's range by this factor about its midpoint.
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9


def soft_joint_limits(jnt_range, factor=SOFT_JOINT_POS_LIMIT_FACTOR):
    """Isaac-style soft joint position limits from an (N, 2) [lo, hi] range array: shrink each
    LIMITED joint's range by `factor` about its midpoint (soft_joint_pos_limits semantics).
    Unlimited joints (MJCF hi <= lo) get (-inf, +inf) so a clamp never touches them."""
    jnt_range = np.asarray(jnt_range, float)
    mid = 0.5 * (jnt_range[:, 0] + jnt_range[:, 1])
    half = 0.5 * (jnt_range[:, 1] - jnt_range[:, 0]) * factor
    limited = jnt_range[:, 1] > jnt_range[:, 0]
    lo = np.where(limited, mid - half, -np.inf)
    hi = np.where(limited, mid + half, np.inf)
    return lo, hi


def validate_formal_bank_execution_contract(policy, *, physics_step_dt_s,
                                            policy_step_dt_s, control_decimation,
                                            qdes_clamp):
    """Fail closed unless the evaluator exactly reproduces the schema-3 execution contract.

    The formal exam must not reconstruct q_des bounds from whichever MJCF happens to be passed:
    training's effective soft limits and both clock rates are immutable checkpoint facts.  The
    returned ``(lo, hi)`` arrays are therefore the bounds the rollout must actually apply.
    """
    require_contract(
        policy.training_contract_exact == "1",
        "formal BankExam requires training_contract_exact=1",
    )
    require_contract(
        policy.training_contract_schema_version == "3",
        "formal BankExam requires training_contract_schema_version exactly 3",
    )
    require_contract(
        is_sha256(policy.training_contract_sha256),
        "formal BankExam requires a valid training_contract_sha256",
    )
    require_contract(
        is_sha256(policy.source_checkpoint_sha256),
        "formal BankExam requires a valid source_checkpoint_sha256",
    )
    limits = policy.qdes_joint_pos_limits
    require_contract(
        isinstance(limits, np.ndarray) and limits.shape == (31, 2)
        and np.isfinite(limits).all() and np.all(limits[:, 0] <= limits[:, 1]),
        "formal BankExam requires finite qdes_joint_pos_limits metadata shaped (31,2)",
    )
    for name, runtime, recorded in (
        ("physics_step_dt_s", physics_step_dt_s, policy.physics_step_dt_s),
        ("policy_step_dt_s", policy_step_dt_s, policy.policy_step_dt_s),
    ):
        require_contract(
            recorded is not None and math.isclose(
                float(runtime), float(recorded), rel_tol=0.0, abs_tol=1e-12
            ),
            f"formal BankExam runtime {name}={runtime!r} != training metadata {recorded!r}",
        )
    require_contract(
        policy.control_decimation is not None
        and int(control_decimation) == int(policy.control_decimation),
        "formal BankExam runtime control_decimation="
        f"{control_decimation} != training metadata {policy.control_decimation}",
    )
    require_contract(
        policy.qdes_clamp_meta is True and bool(qdes_clamp),
        "formal BankExam requires qdes_clamp=1 in training metadata and evaluator runtime",
    )
    actuator_types = getattr(policy, "joint_actuator_types", None)
    require_contract(
        actuator_types is not None
        and len(actuator_types) == 31
        and all(value in ("implicit", "explicit") for value in actuator_types),
        "formal BankExam requires one bound implicit|explicit actuator type per joint",
    )
    for name, values, strictly_positive in (
        ("joint_effort_limits", getattr(policy, "joint_effort_limits", None), True),
        ("joint_armature", getattr(policy, "joint_armature", None), False),
        ("joint_friction_coefficients", getattr(policy, "joint_friction_coefficients", None), False),
        ("joint_velocity_limits", getattr(policy, "joint_velocity_limits", None), True),
    ):
        require_contract(
            isinstance(values, np.ndarray) and values.shape == (31,)
            and np.isfinite(values).all()
            and np.all(values > 0.0 if strictly_positive else values >= 0.0),
            f"formal BankExam requires 31 finite "
            f"{'positive' if strictly_positive else 'non-negative'} {name}",
        )
    require_contract(
        getattr(policy, "joint_friction_backend", None) == "physx"
        and getattr(policy, "joint_friction_semantics", None)
        == "load_dependent_spatial_force_coefficient"
        and getattr(policy, "joint_friction_units", None) == "dimensionless",
        "formal BankExam requires explicit PhysX load-dependent dimensionless friction semantics",
    )
    # MuJoCo ``frictionloss`` is a constant Coulomb torque in N m.  Isaac/PhysX joint friction is
    # a dimensionless coefficient multiplying transmitted spatial force.  Equal numeric values do
    # not describe the same plant.  Exact evaluation is possible only when the bound coefficient is
    # identically zero; non-zero contracts may run solely through --allow-inexact-contract as an
    # explicitly labelled direct-number proxy.
    require_contract(
        np.array_equal(
            getattr(policy, "joint_friction_coefficients", None),
            np.zeros(31, np.float64),
        ),
        "formal BankExam cannot reproduce non-zero PhysX load-dependent joint-friction "
        "coefficients with MuJoCo constant-Nm frictionloss; use --allow-inexact-contract only "
        "for a labelled proxy, or train/calibrate a zero/equivalent friction contract",
    )
    return limits[:, 0].copy(), limits[:, 1].copy()


def stand_hold_refs(refs, default_q):
    """READY-STAND hold reference (2026-07-05+ training hold semantics, commands.py): joint refs =
    the default stand, ref joint vel = 0. Returns a shallow COPY of `refs` — body/anchor reference
    entries pass through untouched (training's hold only re-points the JOINT command), and the
    refs_table entry itself is never mutated."""
    refs = dict(refs)
    refs["joint_pos"] = default_q
    refs["joint_vel"] = np.zeros_like(refs["joint_vel"])
    return refs


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
        input_meta = {item.name: item for item in self.sess.get_inputs()}
        output_meta = {item.name: item for item in self.sess.get_outputs()}
        ins = {name: item.shape for name, item in input_meta.items()}
        expected_outputs = {
            "actions": [1, 31],
            "joint_pos": [1, 31],
            "joint_vel": [1, 31],
            "body_pos_w": [1, 14, 3],
            "body_quat_w": [1, 14, 4],
            "body_lin_vel_w": [1, 14, 3],
            "body_ang_vel_w": [1, 14, 3],
        }
        require_contract(
            list(input_meta) == ["obs", "time_step"],
            f"unexpected ONNX inputs/order: {list(input_meta)}",
        )
        require_contract(
            list(output_meta) == list(expected_outputs),
            f"unexpected ONNX outputs/order: {list(output_meta)}",
        )
        require_contract(
            all(item.type == "tensor(float)" for item in [*input_meta.values(), *output_meta.values()]),
            "all ONNX inputs/outputs must be float32 tensors",
        )
        require_contract(ins["time_step"] == [1, 1], f"time_step shape must be [1,1], got {ins['time_step']}")
        for name, shape in expected_outputs.items():
            require_contract(
                output_meta[name].shape == shape,
                f"ONNX output {name} shape {output_meta[name].shape} != {shape}",
            )
        self.obs_dim = int(ins["obs"][1])
        require_contract(self.obs_dim in (110, 175, 177, 179, 180, 181), (
            f"expected obs dim 180 (base), 175 (deploy_parity), 177 (hitter_footwork), "
            f"179 (deploy_parity + face command), 181 (179 + station anchor), or "
            f"110 (hitter_pure), got {self.obs_dim}"))
        require_contract(ins["obs"] == [1, self.obs_dim], f"obs shape must be fixed [1,D], got {ins['obs']}")
        # 175-D deploy_parity = 180-D MINUS motion_anchor_pos_b(3) and base_target_pos_b(2), with the
        # racket_target_pos_b term reframed relative to the CURRENT racket FK (not the base). See build_obs.
        # 177-D hitter_footwork = the 175-D layout PLUS base_target_pos_b(2) re-inserted after
        # projected_gravity (relative-Δ station footwork channel); racket reframe stays FK-relative.
        # 179-D (stage 1, 2026-07-06) = 175-D + racket_target_normal_cmd tail: DEMANDED face normal
        # (3, world) + zero-filled rho placeholder (1) — the frozen contract-day 175->179 layout
        # (train.py face_command_obs appends the term LAST, so the 175 prefix is byte-identical).
        # 181-D (R10c, 2026-07-09) = 179-D + station_anchor_err_b(2) tail: world-frame station anchor
        # (spawn-point constant = env origin) minus current base XY, yaw-heading base frame — franco's
        # "planner p_base 站位锚" (the anchor stays put while the trunk drifts, so the policy SEES the
        # drift; contract deploy_parity_station181). Station-before-face is the CONTRACT-DAY layout;
        # this tail order is the training-side pure-tail-pad transition (see actor_observation_contract).
        self.deploy_parity = (self.obs_dim in (175, 179, 181))
        self.face_command = (self.obs_dim in (179, 181))
        self.station_obs = (self.obs_dim == 181)
        self.hitter = (self.obs_dim == 177)
        self.hitter_pure = (self.obs_dim == 110)
        self.out_names = list(output_meta)
        md = self.sess.get_modelmeta().custom_metadata_map
        self.metadata = dict(md)
        self.metadata_schema_version = str(md.get("hope_metadata_schema_version", "")).strip()
        try:
            self.metadata_schema_number = int(self.metadata_schema_version or "0")
        except ValueError:
            raise SystemExit(
                f"[FATAL] invalid hope_metadata_schema_version={self.metadata_schema_version!r}"
            )
        self.formal_schema = self.metadata_schema_number >= 2
        self.evaluation_contract_exact = self.formal_schema
        self.training_contract_exact = str(
            md.get("training_contract_exact", "")
        ).strip()
        if self.training_contract_exact not in ("", "0", "1"):
            raise SystemExit("[FATAL] training_contract_exact metadata must be 0|1")
        if self.training_contract_exact != "1":
            self.evaluation_contract_exact = False
        self.training_contract_schema_version = str(
            md.get("training_contract_schema_version", "")
        ).strip()
        self.training_contract_sha256 = str(
            md.get("training_contract_sha256", "")
        ).strip().lower()
        if self.training_contract_sha256 and (
            len(self.training_contract_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.training_contract_sha256)
        ):
            raise SystemExit("[FATAL] invalid training_contract_sha256 metadata")
        self.episode_length_s_meta = None
        if str(md.get("episode_length_s", "")).strip():
            try:
                episode_length_s = float(md["episode_length_s"])
            except ValueError as exc:
                raise SystemExit(
                    f"[FATAL] invalid episode_length_s metadata {md['episode_length_s']!r}"
                ) from exc
            if not math.isfinite(episode_length_s) or episode_length_s <= 0.0:
                raise SystemExit(
                    f"[FATAL] episode_length_s metadata must be finite and positive, got "
                    f"{episode_length_s!r}"
                )
            self.episode_length_s_meta = episode_length_s
        self.stage1_source_family_sha256 = str(
            md.get("stage1_source_family_sha256", "")
        ).strip().lower()
        if self.stage1_source_family_sha256 and (
            len(self.stage1_source_family_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.stage1_source_family_sha256)
        ):
            raise SystemExit(
                "[FATAL] invalid stage1_source_family_sha256 in ONNX metadata"
            )
        self.stage1_question_bank_exact = str(
            md.get("stage1_question_bank_exact", "")
        ).strip()
        if self.stage1_question_bank_exact not in ("", "0", "1"):
            raise SystemExit(
                "[FATAL] stage1_question_bank_exact metadata must be 0|1 when present"
            )
        self.stage1_bank_schema_version = str(
            md.get("stage1_bank_schema_version", "")
        ).strip()
        self.stage1_bank_split = str(md.get("stage1_bank_split", "")).strip()
        self.stage1_train_bank_sha256 = str(
            md.get("stage1_train_bank_sha256", "")
        ).strip().lower()
        if self.stage1_train_bank_sha256 and (
            len(self.stage1_train_bank_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.stage1_train_bank_sha256)
        ):
            raise SystemExit("[FATAL] invalid stage1_train_bank_sha256 in ONNX metadata")
        self.source_checkpoint_sha256 = str(
            md.get("source_checkpoint_sha256", "")
        ).strip().lower()
        if self.source_checkpoint_sha256 and (
            len(self.source_checkpoint_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.source_checkpoint_sha256)
        ):
            raise SystemExit("[FATAL] invalid source_checkpoint_sha256 in ONNX metadata")
        qdes_meta_raw = str(md.get("qdes_clamp", "")).strip()
        if qdes_meta_raw not in ("", "0", "1"):
            raise SystemExit("[FATAL] qdes_clamp metadata must be 0|1")
        self.qdes_clamp_meta = None if not qdes_meta_raw else qdes_meta_raw == "1"
        self.qdes_joint_pos_limits = None
        qdes_limits_raw = str(md.get("qdes_joint_pos_limits", "")).strip()
        if qdes_limits_raw:
            try:
                qdes_values = np.asarray(
                    [float(value) for value in qdes_limits_raw.split(",")], np.float64
                )
            except ValueError as exc:
                raise SystemExit(
                    "[FATAL] invalid qdes_joint_pos_limits metadata"
                ) from exc
            if qdes_values.shape != (62,) or not np.isfinite(qdes_values).all():
                raise SystemExit(
                    "[FATAL] qdes_joint_pos_limits metadata must contain 62 finite floats "
                    "(lo0,hi0,...,lo30,hi30)"
                )
            self.qdes_joint_pos_limits = qdes_values.reshape(31, 2)
            if np.any(self.qdes_joint_pos_limits[:, 0] > self.qdes_joint_pos_limits[:, 1]):
                raise SystemExit("[FATAL] qdes_joint_pos_limits contains lo > hi")

        def optional_positive_float(key):
            raw = str(md.get(key, "")).strip()
            if not raw:
                return None
            try:
                value = float(raw)
            except ValueError as exc:
                raise SystemExit(f"[FATAL] invalid {key} metadata {raw!r}") from exc
            if not math.isfinite(value) or value <= 0.0:
                raise SystemExit(f"[FATAL] {key} must be finite and positive, got {value!r}")
            return value

        self.physics_step_dt_s = optional_positive_float("physics_step_dt_s")
        self.policy_step_dt_s = optional_positive_float("policy_step_dt_s")
        control_decimation_raw = str(md.get("control_decimation", "")).strip()
        self.control_decimation = None
        if control_decimation_raw:
            try:
                control_decimation = int(control_decimation_raw)
            except ValueError as exc:
                raise SystemExit(
                    f"[FATAL] invalid control_decimation metadata {control_decimation_raw!r}"
                ) from exc
            if control_decimation <= 0 or str(control_decimation) != control_decimation_raw:
                raise SystemExit(
                    "[FATAL] control_decimation metadata must be a canonical positive integer"
                )
            self.control_decimation = control_decimation
        self.joint_actuator_types = None
        actuator_types_raw = str(md.get("joint_actuator_types", "")).strip()
        if actuator_types_raw:
            actuator_types = tuple(value.strip() for value in actuator_types_raw.split(","))
            if len(actuator_types) != 31 or any(
                value not in ("implicit", "explicit") for value in actuator_types
            ):
                raise SystemExit(
                    "[FATAL] joint_actuator_types must contain 31 implicit|explicit values"
                )
            self.joint_actuator_types = actuator_types

        def optional_joint_vector(key, *, strictly_positive):
            raw = str(md.get(key, "")).strip()
            if not raw:
                return None
            try:
                values = np.asarray([float(value) for value in raw.split(",")], np.float64)
            except ValueError as exc:
                raise SystemExit(f"[FATAL] invalid {key} metadata") from exc
            invalid = (
                values.shape != (31,)
                or not np.isfinite(values).all()
                or np.any(values <= 0.0 if strictly_positive else values < 0.0)
            )
            if invalid:
                qualifier = "positive" if strictly_positive else "non-negative"
                raise SystemExit(
                    f"[FATAL] {key} metadata must contain 31 finite {qualifier} values"
                )
            return values

        self.joint_effort_limits = optional_joint_vector(
            "joint_effort_limits", strictly_positive=True
        )
        self.joint_armature = optional_joint_vector(
            "joint_armature", strictly_positive=False
        )
        self.joint_friction_coefficients = optional_joint_vector(
            "joint_friction_coefficients", strictly_positive=False
        )
        self.joint_velocity_limits = optional_joint_vector(
            "joint_velocity_limits", strictly_positive=True
        )
        self.joint_friction_backend = str(
            md.get("joint_friction_backend", "")
        ).strip()
        self.joint_friction_semantics = str(
            md.get("joint_friction_semantics", "")
        ).strip()
        self.joint_friction_units = str(md.get("joint_friction_units", "")).strip()
        if self.training_contract_exact == "1" and self.training_contract_schema_version == "3":
            missing_plant = [
                key for key, value in (
                    ("joint_actuator_types", self.joint_actuator_types),
                    ("joint_effort_limits", self.joint_effort_limits),
                    ("joint_armature", self.joint_armature),
                    ("joint_friction_coefficients", self.joint_friction_coefficients),
                    ("joint_velocity_limits", self.joint_velocity_limits),
                    ("joint_friction_backend", self.joint_friction_backend),
                    ("joint_friction_semantics", self.joint_friction_semantics),
                    ("joint_friction_units", self.joint_friction_units),
                ) if value is None or (isinstance(value, str) and value == "")
            ]
            if missing_plant:
                raise SystemExit(
                    "[FATAL] exact schema-3 ONNX lacks actuator-plant metadata: "
                    + ", ".join(missing_plant)
                )
            if (
                self.joint_friction_backend != "physx"
                or self.joint_friction_semantics
                != "load_dependent_spatial_force_coefficient"
                or self.joint_friction_units != "dimensionless"
            ):
                raise SystemExit(
                    "[FATAL] exact schema-3 ONNX has unsupported joint-friction semantics; "
                    "expected physx/load_dependent_spatial_force_coefficient/dimensionless"
                )
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
        self.observation_names = tuple(v.strip() for v in md["observation_names"].split(","))
        # optional clip-clock metadata (baked by scripts/play.py; the same keys the C++ deploy
        # runner uses to override its built-in clip layout)
        self.clip_strike_phases = None
        if md.get("clip_strike_phases", "").strip():
            self.clip_strike_phases = tuple(float(v) for v in md["clip_strike_phases"].split(","))
        self.clip_seg_lengths = None
        if md.get("clip_seg_lengths", "").strip():
            self.clip_seg_lengths = tuple(int(float(v)) for v in md["clip_seg_lengths"].split(","))
        self.motion_clip_sha256 = tuple(
            v.strip() for v in md.get("motion_clip_sha256", "").split(",") if v.strip()
        )
        # per-clip reference base->racket reach offset at the strike frame (dx0,dy0,dx1,dy1,...).
        # Needed to derive the 177-D hitter base station from the racket target (base_couple_mode
        # reference_reach). Baked by utils/exporter.py since 2026-07-06; older 177 exports lack it —
        # main() computes a fallback from refs_table (same arithmetic as training).
        self.ref_reach_offset_xy = None
        if md.get("ref_reach_offset_xy", "").strip():
            vals = [float(v) for v in md["ref_reach_offset_xy"].split(",")]
            self.ref_reach_offset_xy = [np.array(vals[i:i + 2]) for i in range(0, len(vals), 2)]
        # optional episode-semantics metadata (future exports may bake the training wrap_teleport
        # flag; none do as of 2026-07-04 — --reset-mode auto then defaults to multiswing).
        self.wrap_teleport_meta = None
        if md.get("wrap_teleport", "").strip():
            self.wrap_teleport_meta = md["wrap_teleport"].strip().lower() in ("1", "true", "yes")
        self.motion_hold_steps_range_meta = None
        if md.get("motion_hold_steps_range", "").strip():
            hold_vals = tuple(int(float(v)) for v in md["motion_hold_steps_range"].split(","))
            if len(hold_vals) != 2 or not (0 <= hold_vals[0] <= hold_vals[1]):
                raise SystemExit(
                    f"[FATAL] invalid motion_hold_steps_range metadata: {hold_vals}"
                )
            self.motion_hold_steps_range_meta = hold_vals
        self.motion_hold_reference_meta = md.get("motion_hold_reference", "").strip() or None
        self.motion_stand_start_prob_meta = (
            float(md["motion_stand_start_prob"])
            if md.get("motion_stand_start_prob", "").strip() else None
        )
        self.motion_stand_start_min_hold_meta = (
            int(float(md["motion_stand_start_min_hold"]))
            if md.get("motion_stand_start_min_hold", "").strip() else None
        )
        self.motion_stand_start_yaw_range_meta = None
        if md.get("motion_stand_start_yaw_range", "").strip():
            yaw_vals = tuple(float(v) for v in md["motion_stand_start_yaw_range"].split(","))
            if len(yaw_vals) != 2 or yaw_vals[0] > yaw_vals[1]:
                raise SystemExit(
                    f"[FATAL] invalid motion_stand_start_yaw_range metadata: {yaw_vals}"
                )
            self.motion_stand_start_yaw_range_meta = yaw_vals
        # 110-D HitterPure task geometry and face provenance. These values are part of the policy
        # contract, not evaluator tuning: scoring against a different plane/box answers a different
        # scientific question.
        self.hp_pos_range_per_clip = self._parse_hp_boxes(
            md.get("hitter_pure_pos_range_per_clip", "")
        )
        self.hp_vel_range_per_clip = self._parse_hp_boxes(
            md.get("hitter_pure_vel_range_per_clip", "")
        )
        self.hp_base_target_range = None
        if md.get("hitter_pure_base_target_range", "").strip():
            vals = [float(x) for x in md["hitter_pure_base_target_range"].split(",")]
            if len(vals) != 4:
                raise SystemExit("[FATAL] hitter_pure_base_target_range metadata needs 4 floats")
            self.hp_base_target_range = ((vals[0], vals[1]), (vals[2], vals[3]))
        self.mount_normal_sign_per_clip_meta = None
        if md.get("mount_normal_sign_per_clip", "").strip():
            self.mount_normal_sign_per_clip_meta = tuple(
                float(x) for x in md["mount_normal_sign_per_clip"].split(",")
            )
        if self.hitter_pure:
            if not self.formal_schema:
                raise SystemExit(
                    "[FATAL] 110-D ONNX requires hope_metadata_schema_version>=2; re-export with "
                    "the current exporter."
                )
            contract = md.get("actor_obs_contract", "").strip()
            obs_mode = md.get("actor_obs_mode", "").strip()
            total_dim = md.get("actor_obs_total_dim", "").strip()
            term_dims_raw = md.get("actor_obs_term_dims", "").strip()
            term_dims = tuple(int(float(v)) for v in term_dims_raw.split(",")) if term_dims_raw else ()
            if contract != "hitter_pure" or obs_mode != "hitter_pure":
                raise SystemExit(
                    "[FATAL] 110-D ONNX lacks the exact HitterPure contract provenance: "
                    f"actor_obs_contract={contract!r}, actor_obs_mode={obs_mode!r}. Re-export; "
                    "input width alone cannot identify a column layout."
                )
            if total_dim != "110" or term_dims != HITTER_PURE_OBS_DIMS:
                raise SystemExit(
                    "[FATAL] 110-D ONNX actor term dimensions do not match HitterPure: "
                    f"total={total_dim!r}, dims={term_dims}, expected={HITTER_PURE_OBS_DIMS}."
                )
            if self.observation_names != HITTER_PURE_OBS_NAMES:
                raise SystemExit(
                    "[FATAL] 110-D ONNX observation_names mismatch.\n"
                    f"Expected: {HITTER_PURE_OBS_NAMES}\nActual:   {self.observation_names}\n"
                    "Refusing to silently feed the right number of columns in the wrong order."
                )
        if self.training_contract_exact == "1":
            expected_contract, expected_mode, expected_names, expected_dims = (
                FORMAL_ACTOR_CONTRACTS[self.obs_dim]
            )
            actor_contract = md.get("actor_obs_contract", "").strip()
            actor_mode = md.get("actor_obs_mode", "").strip()
            actor_total = md.get("actor_obs_total_dim", "").strip()
            actor_dims_raw = md.get("actor_obs_term_dims", "").strip()
            try:
                actor_dims = tuple(int(float(v)) for v in actor_dims_raw.split(","))
            except ValueError as exc:
                raise SystemExit(
                    f"[FATAL] invalid actor_obs_term_dims metadata {actor_dims_raw!r}"
                ) from exc
            if (
                actor_contract != expected_contract
                or actor_mode != expected_mode
                or actor_total != str(self.obs_dim)
                or actor_dims != expected_dims
                or self.observation_names != expected_names
            ):
                raise SystemExit(
                    "[FATAL] formal actor observation registry mismatch: "
                    f"expected contract/mode/total/dims/names="
                    f"{expected_contract}/{expected_mode}/{self.obs_dim}/{expected_dims}/"
                    f"{expected_names}; got {actor_contract}/{actor_mode}/{actor_total}/"
                    f"{actor_dims}/{self.observation_names}"
                )
        n = len(self.joint_names)
        require_contract(
            n == 31 and self.default_q.shape == (31,) and self.action_scale.shape == (31,),
            f"expected 31 joints/default_q/action_scale, got n={n}, "
            f"default_q={self.default_q.shape}, action_scale={self.action_scale.shape}",
        )
        require_contract(
            self.body_names == TRACKED_BODIES,
            f"ONNX body_names != expected tracked order:\n {self.body_names}\n {TRACKED_BODIES}",
        )
        # --- empirical obs normalization -----------------------------------------------------------
        # There are two independent facts: whether training normalized observations, and whether
        # that transform is baked into this graph. Schema-v2 exports must state both. Historical
        # exports predate those keys, so they retain the explicit legacy sidecar fallback rather than
        # being assigned provenance that cannot be inferred from graph width or file adjacency.
        self.obs_mean = self.obs_std = None
        self.obs_eps = 1e-2                      # rsl_rl EmpiricalNormalization default
        self.obs_norm_path = None
        norm_baked_raw = str(md.get("obs_norm_baked", "")).strip()
        empirical_raw = str(md.get("empirical_normalization", "")).strip()
        trained_norm_raw = str(md.get("trained_with_obs_norm", "")).strip()
        if self.formal_schema and (norm_baked_raw not in ("0", "1") or
                                   empirical_raw not in ("0", "1")):
            raise SystemExit(
                "[FATAL] schema-v2 ONNX must explicitly declare obs_norm_baked and "
                "empirical_normalization as 0|1; got "
                f"{norm_baked_raw!r}/{empirical_raw!r}/{trained_norm_raw!r}. Re-export."
            )
        if norm_baked_raw not in ("", "0", "1"):
            raise SystemExit(
                f"[FATAL] invalid obs_norm_baked metadata {norm_baked_raw!r}; expected 0|1"
            )
        if empirical_raw not in ("", "0", "1"):
            raise SystemExit(
                f"[FATAL] invalid empirical_normalization metadata {empirical_raw!r}; expected 0|1"
            )
        if trained_norm_raw not in ("", "0", "1"):
            raise SystemExit(
                f"[FATAL] invalid trained_with_obs_norm metadata {trained_norm_raw!r}; expected 0|1"
            )
        if empirical_raw and trained_norm_raw and empirical_raw != trained_norm_raw:
            raise SystemExit(
                "[FATAL] empirical_normalization and trained_with_obs_norm metadata disagree"
            )
        if self.formal_schema and not trained_norm_raw:
            self.evaluation_contract_exact = False
        declared_baked = None if not norm_baked_raw else norm_baked_raw == "1"
        graph_baked, graph_norm_reason = inspect_onnx_obs_normalization(onnx_path)
        if self.formal_schema and graph_baked is None:
            raise SystemExit(
                "[FATAL] cannot prove schema-v2 ONNX observation-normalization dataflow: "
                f"{graph_norm_reason}"
            )
        if (graph_baked is not None and declared_baked is not None and
                graph_baked != declared_baked):
            raise SystemExit(
                "[FATAL] ONNX obs_norm_baked metadata contradicts graph dataflow: "
                f"metadata={int(declared_baked)}, graph={int(graph_baked)} ({graph_norm_reason})"
            )
        self.obs_norm_baked = bool(graph_baked if graph_baked is not None else declared_baked)
        if graph_baked is not None:
            print(
                f"[mj-sim2sim] graph-proven obs_norm_baked={int(graph_baked)}: "
                f"{graph_norm_reason}"
            )
        self.empirical_normalization = None if not empirical_raw else empirical_raw == "1"
        if self.obs_norm_baked and self.empirical_normalization is False:
            raise SystemExit(
                "[FATAL] contradictory ONNX metadata: obs_norm_baked=1 but "
                "empirical_normalization=0"
            )
        explicit_raw_override = obs_norm == "off"
        if (self.formal_schema and explicit_raw_override and
                self.empirical_normalization is True and not self.obs_norm_baked):
            raise SystemExit(
                "[FATAL] --no-obs-norm would feed raw observations to a schema-v2 model that "
                "declares empirical_normalization=1 and obs_norm_baked=0. Formal scores fail "
                "closed; provide the matching obs_norm.npz sidecar."
            )
        # Double-normalization guard: exports made with standalone_onnx_export.py --bake-obs-norm
        # carry obs_norm_baked=1 in metadata and must NOT get the sidecar on top.
        if self.obs_norm_baked:
            print("[mj-sim2sim] obs normalization BAKED into the ONNX graph (metadata) — sidecar skipped")
            obs_norm = "off"
        elif self.empirical_normalization is False:
            if obs_norm not in (None, "auto", "off"):
                raise SystemExit(
                    "[FATAL] --obs-norm was supplied for an ONNX that declares "
                    "empirical_normalization=0; refusing to apply an out-of-contract transform."
                )
            auto_sidecar = os.path.join(
                os.path.dirname(os.path.abspath(onnx_path)), "obs_norm.npz"
            )
            if os.path.isfile(auto_sidecar):
                print(f"[mj-sim2sim] stale normalization sidecar ignored (model declares "
                      f"empirical_normalization=0): {auto_sidecar}")
            obs_norm = "off"
        if obs_norm != "off":
            path = obs_norm if obs_norm not in (None, "auto") else \
                os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "obs_norm.npz")
            if os.path.isfile(path):
                with np.load(path) as d:
                    if "mean" not in d or "std" not in d:
                        raise SystemExit(f"[FATAL] obs_norm sidecar lacks mean/std: {path}")
                    mean = np.asarray(d["mean"], np.float64).reshape(-1)
                    std = np.asarray(d["std"], np.float64).reshape(-1)
                    eps = float(d["eps"]) if "eps" in d else 1e-2
                    count = int(d["count"]) if "count" in d else None
                    checkpoint_sha = (
                        str(np.asarray(d["source_checkpoint_sha256"]).item()).strip().lower()
                        if "source_checkpoint_sha256" in d else ""
                    )
                    state_sha = (
                        str(np.asarray(d["normalizer_state_sha256"]).item()).strip().lower()
                        if "normalizer_state_sha256" in d else ""
                    )
                if mean.shape != (self.obs_dim,) or std.shape != (self.obs_dim,):
                    raise SystemExit(
                        f"[FATAL] obs_norm sidecar dim {mean.shape}/{std.shape} != obs dim "
                        f"{self.obs_dim} ({path}) — wrong observation contract/checkpoint?"
                    )
                if (not np.isfinite(mean).all() or not np.isfinite(std).all() or
                        np.any(std < 0.0) or not math.isfinite(eps) or eps < 0.0 or
                        np.any(std + eps <= 0.0)):
                    raise SystemExit(
                        f"[FATAL] obs_norm sidecar must contain finite mean, finite non-negative "
                        f"std, and finite eps>=0 with std+eps>0: {path}"
                    )
                if self.formal_schema and (count is None or count <= 0):
                    raise SystemExit(
                        f"[FATAL] schema-v2 normalized model requires obs_norm sidecar count>0: {path}"
                    )
                if checkpoint_sha and not is_sha256(checkpoint_sha):
                    raise SystemExit(
                        f"[FATAL] invalid source_checkpoint_sha256 inside obs_norm sidecar: {path}"
                    )
                if state_sha:
                    if not is_sha256(state_sha):
                        raise SystemExit(
                            f"[FATAL] invalid normalizer_state_sha256 inside obs_norm sidecar: {path}"
                        )
                    actual_state_sha = normalizer_state_sha256(mean, std, eps, count)
                    if state_sha != actual_state_sha:
                        raise SystemExit(
                            "[FATAL] obs_norm payload does not match its normalizer_state_sha256: "
                            f"declared={state_sha}, actual={actual_state_sha}, path={path}"
                        )
                if self.metadata_schema_number >= 3 and (not checkpoint_sha or not state_sha):
                    raise SystemExit(
                        "[FATAL] schema-v3+ normalized raw ONNX requires sidecar-internal "
                        "source_checkpoint_sha256 and normalizer_state_sha256"
                    )
                expected_checkpoint_sha = str(
                    md.get("source_checkpoint_sha256", md.get("checkpoint_sha256", ""))
                ).strip().lower()
                if expected_checkpoint_sha:
                    if not checkpoint_sha or checkpoint_sha != expected_checkpoint_sha:
                        raise SystemExit(
                            "[FATAL] obs_norm source checkpoint does not match ONNX metadata: "
                            f"expected={expected_checkpoint_sha}, sidecar={checkpoint_sha or '<missing>'}"
                        )
                expected_norm_sha = str(md.get("obs_norm_sidecar_sha256", "")).strip().lower()
                if expected_norm_sha:
                    actual_norm_sha = sha256_file(path)
                    if actual_norm_sha != expected_norm_sha:
                        raise SystemExit(
                            "[FATAL] obs_norm sidecar SHA256 does not match ONNX metadata: "
                            f"expected={expected_norm_sha}, actual={actual_norm_sha}, path={path}"
                        )
                elif self.metadata_schema_number >= 3:
                    raise SystemExit(
                        "[FATAL] schema-v3+ normalized raw ONNX requires "
                        "obs_norm_sidecar_sha256 metadata"
                    )
                elif self.formal_schema:
                    self.evaluation_contract_exact = False
                    print(
                        "[mj-sim2sim] WARNING: schema-v2 normalization sidecar is not SHA-bound "
                        "to the ONNX; evaluation_contract_exact=false (legacy-unbound-sidecar)"
                    )
                self.obs_mean, self.obs_std = mean, std
                self.obs_eps = eps
                self.obs_norm_path = path
            elif obs_norm not in (None, "auto"):
                raise SystemExit(f"[FATAL] --obs-norm sidecar not found: {path}")
            elif self.empirical_normalization is True:
                raise SystemExit(
                    "[FATAL] this ONNX declares empirical_normalization=1 and obs_norm_baked=0, "
                    f"but no sidecar exists at {path}. Create it from the SAME checkpoint with "
                    "scripts/make_std_sidecar.py."
                )
        if explicit_raw_override and self.empirical_normalization is True and not self.obs_norm_baked:
            print("[mj-sim2sim] WARNING: explicit raw-observation override on a model trained with "
                  "empirical normalization; this is a diagnostic, not a valid policy score")

    @staticmethod
    def _parse_hp_boxes(value):
        value = value.strip()
        if not value:
            return None
        boxes = []
        for part in value.split(";"):
            vals = [float(x) for x in part.split(",")]
            if len(vals) != 6:
                raise SystemExit(
                    f"[FATAL] hitter_pure box metadata needs 6 floats per clip, got {part!r}"
                )
            boxes.append(((vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5])))
        return boxes

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
    def __init__(self, mjcf_path, joint_names, body_names, sim_dt, keep_native_damping,
                 keep_frictionloss, pd_mode, kd_for_implicit=None, *, actuator_types=None,
                 joint_armature=None, joint_frictionloss_proxy=None,
                 joint_velocity_limits=None, joint_effort_limits=None,
                 require_bound_plant_match=False, allow_velocity_limit_proxy=True):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.opt.timestep = sim_dt
        self.pd_mode = pd_mode
        # Per-joint execution is contract data.  A single observation width never proves whether
        # training used implicit PhysX drives or an explicit actuator model.
        if actuator_types is None:
            actuator_types = [pd_mode] * len(joint_names)
        self.actuator_types = np.asarray(tuple(actuator_types), dtype=object)
        require_contract(
            self.actuator_types.shape == (len(joint_names),)
            and all(value in ("implicit", "explicit") for value in self.actuator_types),
            "actuator_types must contain one implicit|explicit value per joint",
        )
        self.implicit_mask = self.actuator_types == "implicit"
        self.explicit_mask = self.actuator_types == "explicit"
        self.allow_velocity_limit_proxy = bool(allow_velocity_limit_proxy)
        self.velocity_limit_hit_count = 0
        self.velocity_limit_peak_ratio = 0.0
        self.data = mujoco.MjData(self.model)

        def named_id(obj_type, name, kind):
            value = mujoco.mj_name2id(self.model, obj_type, name)
            require_contract(value >= 0, f"MJCF missing {kind} {name!r}")
            return value

        def jid(name): return named_id(mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
        def bid(name): return named_id(mujoco.mjtObj.mjOBJ_BODY, name, "body")
        def aid(name): return named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name, "actuator")

        # Per actuated joint (in ARTICULATION order): qpos addr, qvel/dof addr, actuator id.
        self.qadr = np.array([self.model.jnt_qposadr[jid(n)] for n in joint_names], int)
        self.vadr = np.array([self.model.jnt_dofadr[jid(n)] for n in joint_names], int)
        self.act_id = np.array([aid(n + "_motor") for n in joint_names], int)

        def bound_vector(values, name, *, positive):
            if values is None:
                return None
            out = np.asarray(values, np.float64)
            require_contract(
                out.shape == (len(joint_names),) and np.isfinite(out).all()
                and np.all(out > 0.0 if positive else out >= 0.0),
                f"{name} must contain one finite "
                f"{'positive' if positive else 'non-negative'} value per joint",
            )
            return out

        bound_armature = bound_vector(joint_armature, "joint_armature", positive=False)
        bound_effort = bound_vector(
            joint_effort_limits, "joint_effort_limits", positive=True
        )
        self.joint_velocity_limits = bound_vector(
            joint_velocity_limits, "joint_velocity_limits", positive=True
        )
        frictionloss_proxy = bound_vector(
            joint_frictionloss_proxy, "joint_frictionloss_proxy", positive=False
        )

        if bound_armature is not None:
            source = self.model.dof_armature[self.vadr].copy()
            if require_bound_plant_match:
                require_contract(
                    np.allclose(source, bound_armature, rtol=0.0, atol=1e-10),
                    "formal BankExam MJCF armature disagrees with training metadata: "
                    f"max_abs={float(np.max(np.abs(source - bound_armature))):.3g}",
                )
            self.model.dof_armature[self.vadr] = bound_armature
        if bound_effort is not None:
            source_lo = self.model.actuator_ctrlrange[self.act_id, 0].copy()
            source_hi = self.model.actuator_ctrlrange[self.act_id, 1].copy()
            if require_bound_plant_match:
                require_contract(
                    np.allclose(source_lo, -bound_effort, rtol=0.0, atol=1e-9)
                    and np.allclose(source_hi, bound_effort, rtol=0.0, atol=1e-9),
                    "formal BankExam MJCF actuator ctrlrange disagrees with bound effort limits",
                )
            self.model.actuator_ctrlrange[self.act_id, 0] = -bound_effort
            self.model.actuator_ctrlrange[self.act_id, 1] = bound_effort
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_id, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_id, 1].copy()

        # Soft joint position limits (ARTICULATION order) for the optional --qdes-clamp: the
        # training ClampedJointPositionAction and the C++ deploy runner (pp_joint_limits.hpp)
        # both clamp q_des to these; precomputed unconditionally (read only when the flag is on).
        self.soft_jnt_lo, self.soft_jnt_hi = soft_joint_limits(
            np.array([self.model.jnt_range[jid(n)] for n in joint_names], float))

        # Body ids for the 14 tracked bodies, the pelvis (free base) and torso (anchor).
        self.tracked_bid = np.array([bid(n) for n in body_names], int)
        self.pelvis_bid = bid("pelvis_link")
        self.torso_bid = bid(ANCHOR_BODY)
        self.racket_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_racket")
        self.feet_bid = [bid(n) for n in FEET_BODIES]
        self.feet_geoms = {g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] in self.feet_bid}

        # Native viscous damping and dry friction are separate pieces of the plant. Isaac uses
        # actuator kd plus a PhysX load-dependent joint-friction coefficient, not MuJoCo's extra
        # viscous damping or constant-Nm frictionloss. Conflating them behind --keep-passive made
        # the effective plant uninspectable.
        if not keep_native_damping:
            self.model.dof_damping[self.vadr] = 0.0
        if not keep_frictionloss:
            self.model.dof_frictionloss[self.vadr] = 0.0
        if frictionloss_proxy is not None:
            # Diagnostic only: these numbers are dimensionless PhysX coefficients, while MuJoCo
            # interprets them as constant N m Coulomb torque.  The caller must already have marked
            # the evaluation inexact; this assignment merely preserves the historical proxy.
            self.model.dof_frictionloss[self.vadr] = frictionloss_proxy
        if np.any(self.implicit_mask):
            # Match Isaac's ImplicitActuator: the kd damping is integrated IMPLICITLY (stable + no
            # under-shoot of fast swings at a 5 ms step). Put kd into the passive joint damping and
            # use MuJoCo's implicitfast integrator; the control torque then applies kp only.
            require_contract(kd_for_implicit is not None, "implicit PD requires kd_for_implicit")
            # ADD kd to whatever passive damping survives above (2026-07-05 fix): the old
            # assignment OVERWROTE dof_damping, so --keep-passive + implicit silently lost
            # the MJCF passive damping — the AGI plant has BOTH (passive + our commanded kd).
            kd_for_implicit = np.asarray(kd_for_implicit, np.float64)
            require_contract(
                kd_for_implicit.shape == self.implicit_mask.shape,
                "kd_for_implicit shape does not match actuator contract",
            )
            self.model.dof_damping[self.vadr[self.implicit_mask]] += (
                kd_for_implicit[self.implicit_mask]
            )
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
        # World-frame linear velocity of the racket SITE (= pingpang_red_Link origin).  Do not equate
        # this with Isaac Lab 2.1 data.body_lin_vel_w: that legacy property is the link COM-point
        # velocity even though body_pos_w is the link origin.  Training now uses
        # body_link_lin_vel_w (+ omega x wrist->site in fixed-link fallback), so both simulators grade
        # the same rigid point.
        res = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_SITE,
                                  self.racket_site, res, 0)  # flg_local=0 -> world frame; [ang(3), lin(3)]
        return res[3:6].copy()

    def racket_normal_w(self, sign=None):
        # Actual racket face normal in world = local +Y axis of the racket(=wrist) frame.
        # site has identity orientation rel. to the wrist, so site_xmat == wrist world rotation.
        # sign: 击球面符号覆盖(每 clip 常量,face_sign_for_clip);None = 标量 MOUNT_NORMAL_SIGN,
        # 现役行为逐位不变。
        R = self.data.site_xmat[self.racket_site].reshape(3, 3)
        return R[:, MOUNT_NORMAL_AXIS] * (MOUNT_NORMAL_SIGN if sign is None else float(sign))

    def foot_contact_frac(self):
        ncon = self.data.ncon
        feet_in_contact = set()
        for i in range(ncon):
            c = self.data.contact[i]
            for g in (c.geom1, c.geom2):
                if g in self.feet_geoms:
                    feet_in_contact.add(self.model.geom_bodyid[g])
        return len(feet_in_contact) / max(len(self.feet_bid), 1)

    def reset_to_reference(self, root_pos, root_quat, root_lin_w, root_ang_w, q_artic, qd_artic):
        """Reference-state-init with a complete MuJoCo episode-state reset.

        Overwriting only ``qpos``/``qvel`` leaves solver and actuator history such as
        ``qacc_warmstart``, ``ctrl`` and ``act`` from the previous question.  That violates the
        formal BankExam's one-question/one-reset contract and can make an otherwise identical
        schedule depend on the preceding policy trajectory.  ``mj_resetData`` clears every data-
        side hidden state while preserving the already-configured model (timestep, damping and
        friction choices); the requested reference state is then installed explicitly.
        """
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat
        self.data.qpos[self.qadr] = q_artic
        # MuJoCo free-joint qvel: linear in WORLD frame, angular in the BODY frame.
        R = mat_from_quat(root_quat)
        self.data.qvel[0:3] = root_lin_w
        self.data.qvel[3:6] = R.T @ root_ang_w
        self.data.qvel[self.vadr] = qd_artic
        self.mj.mj_forward(self.model, self.data)

    def reset_to_stand(self, root_pos, root_quat, q_artic):
        """--deploy-faithful episode init: nominal stand (default_joint_pos, upright root at standing
        height), ALL velocities zero. This mirrors how the deployed robot enters MOTION from PD_STAND
        (pp_policy.hpp) — it is deliberately NOT reference-state-init."""
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat
        self.data.qpos[self.qadr] = q_artic
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def reset_to_named_keyframe(self, key_name="stand"):
        """Reset the complete MuJoCo state to a named keyframe, then enforce a static ready state.

        The formal cross-teacher ruler uses the *same full qpos* for every policy.  Reconstructing
        only the free-root pose plus ``policy.default_q`` is insufficient: those defaults belong to
        the candidate and can differ, while an absent-key fallback silently changes the experiment.
        """
        kid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_KEY, str(key_name))
        require_contract(kid >= 0, f"MJCF missing required named keyframe {key_name!r}")
        self.mj.mj_resetDataKeyframe(self.model, self.data, kid)
        self.data.time = 0.0
        self.data.qvel[:] = 0.0
        if self.data.act.size:
            self.data.act[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def ready_state_snapshot(self, mode, last_action):
        return ready_state_snapshot_contract(
            mode=mode,
            qpos=self.data.qpos,
            qvel=self.data.qvel,
            act=self.data.act,
            ctrl=self.data.ctrl,
            time_s=self.data.time,
            qacc_warmstart=self.data.qacc_warmstart,
            mocap_pos=self.data.mocap_pos,
            mocap_quat=self.data.mocap_quat,
            userdata=self.data.userdata,
            last_action=last_action,
        )

    def apply_pd_and_step(self, target_q_artic, kp, kd, decimation):
        """Hold target_q across `decimation` physics substeps, recomputing PD torque each substep.
        explicit: tau = kp*(tgt-q) - kd*qd (full PD as motor force).
        implicit: tau = kp*(tgt-q) only; kd is the passive joint damping integrated by implicitfast."""
        for _ in range(decimation):
            q = self.data.qpos[self.qadr]
            qd = self.data.qvel[self.vadr]
            tau = kp * (target_q_artic - q)
            tau[self.explicit_mask] -= kd[self.explicit_mask] * qd[self.explicit_mask]
            tau = np.clip(tau, self.ctrl_lo, self.ctrl_hi)
            self.data.ctrl[self.act_id] = tau
            self.mj.mj_step(self.model, self.data)
            if self.joint_velocity_limits is not None:
                qd_after = self.data.qvel[self.vadr]
                ratio = np.abs(qd_after) / self.joint_velocity_limits
                peak_ratio = float(np.max(ratio))
                self.velocity_limit_peak_ratio = max(
                    self.velocity_limit_peak_ratio, peak_ratio
                )
                hit = ratio > (1.0 + 1e-9)
                if np.any(hit):
                    self.velocity_limit_hit_count += int(np.count_nonzero(hit))
                    if not self.allow_velocity_limit_proxy:
                        names = np.flatnonzero(hit).tolist()
                        raise SystemExit(
                            "[FATAL] formal BankExam reached bound PhysX joint-velocity limit "
                            f"on articulation indices {names}; MuJoCo lacks the same braking "
                            "constraint, so this trajectory is not exact"
                        )
                    self.data.qvel[self.vadr] = np.clip(
                        qd_after, -self.joint_velocity_limits, self.joint_velocity_limits
                    )
                    self.mj.mj_forward(self.model, self.data)
        return tau   # last substep torque (for logging)


# =================================================================================================
# RacketTargetCommand port (uniform mode, unified 2-clip) — only the obs-feeding quantities.
# =================================================================================================
class RacketCommand:
    def __init__(self, seg_start, seg_len, step_dt, rng, target_normal_per_clip, origin=np.zeros(3),
                 vel_ranges_per_clip=None, pos_ranges_per_clip=None, ref_reach_offset_xy=None,
                 hp_cfg=None):
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
        # 177-D hitter_footwork: per-clip reference base->racket reach offset (list of (dx,dy)).
        # When set, base targets use training's reference_reach coupling (station follows the racket
        # target) instead of the legacy 180-era spawn + weak-Y-blend sampling.
        self.ref_reach_offset_xy = ref_reach_offset_xy
        # 110-D HitterPure: station-first sampling followed by a station-relative racket plane.
        # None keeps every existing 175/177/179/180/181 path and RNG stream unchanged.
        self.hp_cfg = hp_cfg
        # state
        self.racket_target_pos_w = np.zeros(3)
        self.racket_target_vel_w = np.zeros(3)
        self.racket_target_normal_w = np.array([0.0, 1.0, 0.0])
        self.base_target_pos_w = np.zeros(2)
        # the swing's SAMPLED station (deploy-faithful hold pins base_target_pos_w to the live base
        # for a Δ=0 obs, then restores this at swing start — see run_rollout)
        self.station_pos_w = np.zeros(2)
        # 181-D station anchor (R10c): world-frame CONSTANT = env origin XY, exactly the training
        # buffer (station_anchor_pos_w = env origin + offset, offset defaults 0). NOT resampled, NOT
        # pinned during holds — it is the world anchor whose whole point is staying put while the
        # trunk drifts. 人话:出生点常数,漂了这 2 维误差自己变大。
        self.station_anchor_pos_w = np.asarray(origin[:2], np.float64).copy()
        self.swing_sign = 1.0
        self.time_to_strike = 0.0

    def _u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def resample(self, clip_id):
        """New swing: sample racket target (pos/vel), base target, swing sign — matches uniform mode."""
        if self.hp_cfg is not None:
            self._resample_hitter_pure(clip_id)
            return
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
        # base target XY (world): reference_reach station (hitter) or legacy blend — see helper.
        self._sample_base_target(clip_id)

    def _resample_hitter_pure(self, clip_id):
        """Mirror training ``_sample_targets_hitter_pure`` for one MuJoCo environment.

        Sample the world station independently, then sample station-relative racket x/y and
        absolute z, world velocity, and the velocity-direction face target. Every swing redraws
        the station just as reset/wrap calls use ``resample_base=True`` in training.
        """
        hp = self.hp_cfg
        x_range, y_range = hp["base_range"]
        station = np.array([
            self.origin[0] + self._u(*x_range),
            self.origin[1] + self._u(*y_range),
        ])
        pos_box = hp["pos_boxes"][min(clip_id, len(hp["pos_boxes"]) - 1)]
        vel_box = hp["vel_boxes"][min(clip_id, len(hp["vel_boxes"]) - 1)]
        if hp.get("targets_center", False):
            offset = np.array([0.5 * (lo + hi) for lo, hi in pos_box])
            velocity = np.array([0.5 * (lo + hi) for lo, hi in vel_box])
        else:
            offset = np.array([self._u(lo, hi) for lo, hi in pos_box])
            velocity = np.array([self._u(lo, hi) for lo, hi in vel_box])
        self.base_target_pos_w = station.copy()
        self.station_pos_w = station.copy()
        self.racket_target_pos_w = np.array([
            station[0] + offset[0],
            station[1] + offset[1],
            self.origin[2] + offset[2],
        ])
        self.racket_target_vel_w = velocity
        self.racket_target_normal_w = velocity / (np.linalg.norm(velocity) + 1e-6)
        self.swing_sign = 1.0 if clip_id == 0 else -1.0

    def _sample_base_target(self, clip_id):
        """Base station (world XY). hitter_footwork (ref_reach_offset_xy set): training
        base_couple_mode=reference_reach — station = racket_target_xy - per-clip reference
        base->racket reach at strike + jitter, so standing AT the station puts the racket target
        at the clip's reference reach. Legacy 180-D: spawn + weak Y blend + jitter. Both branches
        draw exactly 2 uniforms, keeping the RNG stream length identical across contracts."""
        o = self.origin
        if self.ref_reach_offset_xy is not None:
            reach = self.ref_reach_offset_xy[min(clip_id, len(self.ref_reach_offset_xy) - 1)]
            base_xy = self.racket_target_pos_w[:2] - reach
            base_xy[0] += self._u(*HITTER_BASE_JITTER_X)
            base_xy[1] += self._u(*HITTER_BASE_JITTER_Y)
        else:
            base_xy = o[:2].copy()
            racket_y_off = self.racket_target_pos_w[1] - o[1]
            base_xy[1] += float(np.clip(BASE_COUPLE_BLEND * racket_y_off,
                                        -BASE_COUPLE_MAX_OFFSET, BASE_COUPLE_MAX_OFFSET))
            base_xy[0] += self._u(*BASE_TARGET_X_RANGE)
            base_xy[1] += self._u(*BASE_TARGET_Y_RANGE)
        self.base_target_pos_w = base_xy
        self.station_pos_w = base_xy.copy()

    def set_external_target(self, pos_w, vel_w, normal_w, clip_id):
        """Mode-B (venue-balls) target injection: same state writes as resample(), but with an
        externally computed (ball-demanded) pos/vel/normal instead of box draws. The base-target
        coupling is kept identical to resample() so the policy sees the same obs semantics."""
        self.racket_target_pos_w = np.asarray(pos_w, np.float64).copy()
        self.racket_target_vel_w = np.asarray(vel_w, np.float64).copy()
        self.racket_target_normal_w = np.asarray(normal_w, np.float64).copy()
        self.swing_sign = 1.0 if clip_id == 0 else -1.0
        self._sample_base_target(clip_id)

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

    def station_anchor_err_b(self, base_pos_w, base_quat_w):
        """181-D station-anchor channel: (world anchor − current base XY) in the yaw-heading base
        frame — same math as base_target_pos_b, but against the CONSTANT spawn anchor (training:
        RacketTargetCommand.station_anchor_err_b)."""
        delta = np.array([self.station_anchor_pos_w[0] - base_pos_w[0],
                          self.station_anchor_pos_w[1] - base_pos_w[1], 0.0])
        return quat_rotate_inverse(yaw_quat(base_quat_w), delta)[:2]


# =================================================================================================
# Observation builder. Contracts, detected from the ONNX obs dim:
#   180-D (base)           : full BeyondMimic obs (has motion_anchor_pos_b + base_target_pos_b, and
#                            racket_target_pos_b is BASE-relative).
#   175-D (deploy_parity)  : DROPS motion_anchor_pos_b(3) and base_target_pos_b(2), and reframes
#                            racket_target_pos_b to be relative to the CURRENT RACKET FK (deploy-honest,
#                            no world base position leaks in). Everything else is byte-identical.
#   177-D (hitter_footwork): the 175-D layout PLUS base_target_pos_b(2) re-inserted after
#                            projected_gravity — a RELATIVE Δxy station channel in the yaw-heading
#                            base frame (Δ=0 == "already at station"); racket reframe stays FK-rel.
#   179-D (face command)   : 175-D + racket_target_normal_cmd(4) tail (demanded normal + rho=0).
#   181-D (station anchor) : 179-D + station_anchor_err_b(2) tail — world spawn-constant anchor
#                            minus current base XY, yaw-heading base frame (R10c; the anchor stays
#                            put through holds/swings so trunk drift is visible to the policy).
#   110-D (hitter_pure)    : HITTER Table-I actor: no reference stream/swing_type; world-frame
#                            base forward, station delta, racket-relative-to-base target, velocity,
#                            and time-to-strike after the 99-D A3 proprioceptive prefix.
# =================================================================================================
def build_obs(refs, robot: MujocoRobot, racket: RacketCommand, last_action, default_q,
              deploy_parity=False, face_command=False, hitter=False, station=False,
              hitter_pure=False):
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

    if hitter_pure:
        # Exact HITTER_PURE contract (actor_observation_contract.HITTER_PURE / C++ build_obs_110).
        # Targets stay in world coordinates; e_base,x gives the network the heading explicitly.
        forward_w = mat_from_quat(base_quat_w)[:, 0]
        base_forward_xy = forward_w[:2] / (np.linalg.norm(forward_w[:2]) + 1e-6)
        base_target_delta_xy = racket.base_target_pos_w - base_pos_w[:2]
        racket_target_rel_base = racket.racket_target_pos_w - base_pos_w
        obs = np.concatenate([
            base_ang_vel,
            joint_pos_rel,
            joint_vel_rel,
            last_action,
            proj_grav,
            base_forward_xy,
            base_target_delta_xy,
            racket_target_rel_base,
            racket_vel_w,
            tts,
        ]).astype(np.float64)
        require_contract(obs.shape == (110,), f"obs dim {obs.shape} != 110 (hitter_pure)")
    elif hitter:
        # 177-D hitter_footwork: the 175-D deploy_parity layout PLUS base_target_pos_b(2)
        # inserted after projected_gravity. Same FK-relative racket reframe as 175.
        racket_pos_w = base_pos_w + mat_from_quat(base_quat_w) @ racket_pos_pelvis(q)
        racket_tgt_b = racket.racket_target_pos_b_rel_fk(base_pos_w, base_quat_w, racket_pos_w)
        base_tgt_b = racket.base_target_pos_b(base_pos_w, base_quat_w)
        obs = np.concatenate([
            command, ori_b6, base_ang_vel, joint_pos_rel, joint_vel_rel,
            last_action, proj_grav, base_tgt_b, racket_tgt_b, racket_vel_w, tts, swing,
        ]).astype(np.float64)
        require_contract(obs.shape == (177,), f"obs dim {obs.shape} != 177 (hitter_footwork)")
    elif deploy_parity:
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
        if station:
            # 181-D station-anchor tail (R10c): AFTER the face channel — the 179 prefix must stay
            # byte-identical (pure-tail pad warm start). Station without face has no legal shape
            # (177 would collide with hitter_footwork's inserted-station layout).
            require_contract(face_command, "station channel requires the face channel (181 = 179 + 2)")
            parts.append(racket.station_anchor_err_b(base_pos_w, base_quat_w))
        obs = np.concatenate(parts).astype(np.float64)
        want = 181 if station else (179 if face_command else 175)
        require_contract(obs.shape == (want,), f"obs dim {obs.shape} != {want} (deploy_parity)")
    else:
        # 9. base_target_pos_b (2)
        base_tgt_b = racket.base_target_pos_b(base_pos_w, base_quat_w)
        # 10. racket_target_pos_b (3) base-relative
        racket_tgt_b = racket.racket_target_pos_b(base_pos_w, base_quat_w)
        obs = np.concatenate([
            command, pos_b, ori_b6, base_ang_vel, joint_pos_rel, joint_vel_rel,
            last_action, proj_grav, base_tgt_b, racket_tgt_b, racket_vel_w, tts, swing,
        ]).astype(np.float64)
        require_contract(obs.shape == (180,), f"obs dim {obs.shape} != 180")
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
                target_normal_per_clip, strike_csv_writer=None, attempt_csv_writer=None,
                viewer=None, realtime=True,
                vel_ranges_per_clip=None, pos_ranges_per_clip=None, df=None,
                reset_mode="teleport", hold_range=(0, 100), venue_sampler=None,
                virtual_return_scorer=None,
                switch_stress=0.0, qdes_clamp=False, hold_ref="clip", hp_cfg=None,
                action_noise_rng=None, bank_one_question_reset=False,
                ready_state_contract=None, mjcf_sha256="", execution_contract_sha256=""):
    if action_noise_rng is None:
        # Direct legacy callers retain deterministic behavior; main() always passes an independent
        # stream so action dithering cannot perturb target/question/hold scheduling.
        action_noise_rng = rng
    racket = RacketCommand(seg_start, seg_len, step_dt, rng, target_normal_per_clip,
                           vel_ranges_per_clip=vel_ranges_per_clip,
                           pos_ranges_per_clip=pos_ranges_per_clip,
                           ref_reach_offset_xy=(policy.ref_reach_offset_xy if policy.hitter else None),
                           hp_cfg=hp_cfg)
    strike = {"all": StrikeAcc(), "forehand": StrikeAcc(), "backhand": StrikeAcc()}
    multiswing = (reset_mode == "multiswing") and (df is None)
    bank_schedule = bool(venue_sampler is not None and hasattr(venue_sampler, "schedule"))
    one_question_reset = bool(bank_schedule and bank_one_question_reset)
    training_hold_protocol = training_hold_protocol_active(
        reset_mode=reset_mode, deploy_faithful_cfg=df, venue_sampler=venue_sampler
    )
    require_contract(
        isinstance(ready_state_contract, dict)
        and is_sha256(ready_state_contract.get("sha256", "")),
        "rollout requires a content-addressed ready-state contract",
    )
    require_contract(is_sha256(mjcf_sha256), "rollout requires the exact MJCF SHA256")
    require_contract(
        is_sha256(execution_contract_sha256),
        "rollout requires the exact evaluator execution-contract SHA256",
    )
    if bank_schedule and policy.evaluation_contract_exact:
        require_contract(
            ready_state_contract.get("mode") == FORMAL_READY_STATE_MODE,
            "formal BankExam requires the shared MJCF named stand keyframe ready state",
        )
    # --- mode B (venue-balls): per-rollout accumulators + the current swing's sampled ball -------
    require_contract(venue_sampler is None or df is None, "venue-balls + --deploy-faithful unsupported (v1)")
    require_contract(
        (venue_sampler is None) == (virtual_return_scorer is None),
        "venue/bank sampling and authoritative virtual-return scoring must be enabled together",
    )
    require_contract(
        not bank_schedule or not switch_stress,
        "immutable BankExam schedule is incompatible with switch stress",
    )
    venue = {"all": VenueAcc(), "forehand": VenueAcc(), "backhand": VenueAcc()}
    # mode-B counterfactual: same achieved kinematics, DEMANDED normal swapped in (see docstring).
    venue_cf = {"all": VenueAcc(), "forehand": VenueAcc(), "backhand": VenueAcc()}
    cur_venue_strike = [None]     # VenueStrike of the swing in flight (list = py2-style nonlocal)
    reset_sampler_for_paired_rollout(venue_sampler)
    # One record is opened for EVERY sampled target, before any pre-swing hold.  It is closed exactly
    # once on completion, switch, fall, timeout, or rollout truncation.  Exact-frame strike metrics
    # remain available conditionally, while this ledger supplies the honest unconditional denominator.
    attempt_records = []
    attempt_cur = {}
    attempt_seq = [0]
    attempt_action_noise_rng = [action_noise_rng]
    attempt_ready = [None]

    def attempt_start(c, vs=None):
        if attempt_cur.get("open", False):
            raise RuntimeError(
                f"attempt {attempt_cur.get('attempt_id')} replaced without a finalize reason"
            )
        require_contract(
            isinstance(attempt_ready[0], dict)
            and is_sha256(attempt_ready[0].get("sha256", "")),
            "attempt opened before its actual initial state was content-addressed",
        )
        attempt_cur.clear()
        attempt_cur.update(
            open=True,
            attempt_id=int(attempt_seq[0]),
            clip=int(c),
            schedule_index=int(getattr(vs, "schedule_index", -1)),
            question_sequence_index=int(getattr(vs, "schedule_index", -1)),
            bank_row=int(getattr(vs, "bank_row", -1)),
            question_id=str(getattr(vs, "question_id", "")),
            hold_steps=int(getattr(vs, "hold_steps", 0)),
            attempt_seed=int(getattr(vs, "attempt_seed", 0)),
            schedule_sha256=str(getattr(vs, "schedule_sha256", "")),
            repeat=int(getattr(vs, "repeat", 0)),
            ready_state_mode=str(attempt_ready[0]["mode"]),
            ready_state_sha256=str(attempt_ready[0]["sha256"]),
            mjcf_sha256=str(mjcf_sha256),
            execution_contract_sha256=str(execution_contract_sha256),
            exact=False,
            exact_composite=False,
        )
        if bank_schedule:
            require_contract(
                vs is not None and attempt_cur["schedule_index"] == int(attempt_seq[0]),
                "BankExam schedule index is not contiguous/in order",
            )
            require_contract(
                attempt_cur["question_id"] and is_sha256(attempt_cur["schedule_sha256"]),
                "BankExam attempt lacks question/schedule provenance",
            )
            # Re-seed at every question.  A fall or a different model's episode length therefore
            # cannot shift the Gaussian stream of any later question/noise column.
            attempt_action_noise_rng[0] = np.random.default_rng(attempt_cur["attempt_seed"])
        attempt_seq[0] += 1

    def attempt_mark_exact(composite, hit=False, returned=False):
        if not attempt_cur.get("open", False):
            raise RuntimeError("exact-strike sample has no open target attempt")
        attempt_cur["exact"] = True
        attempt_cur["exact_composite"] = bool(composite)
        attempt_cur["hit"] = bool(hit)
        attempt_cur["returned"] = bool(returned)

    def attempt_finalize(reason, details=()):
        if not attempt_cur.get("open", False):
            raise RuntimeError(f"attempt finalized twice or before target sample: {reason}")
        rec = dict(attempt_cur)
        rec.pop("open", None)
        rec["reason"] = str(reason)
        rec["details"] = tuple(str(value) for value in details)
        rec.update(attempt_ledger_flags(
            rec["reason"], rec["details"], scheduled_exam=bank_schedule
        ))
        rec.setdefault("hit", False)
        rec.setdefault("returned", False)
        attempt_records.append(rec)
        if attempt_csv_writer is not None:
            attempt_csv_writer.writerow([
                mode_label,
                rec["attempt_id"],
                rec["schedule_index"],
                rec["question_sequence_index"],
                CLIP_NAMES.get(rec["clip"], f"clip_{rec['clip']}"),
                rec["bank_row"],
                rec["question_id"],
                rec["repeat"],
                rec["hold_steps"],
                rec["attempt_seed"],
                rec["schedule_sha256"],
                rec["ready_state_mode"],
                rec["ready_state_sha256"],
                rec["mjcf_sha256"],
                rec["execution_contract_sha256"],
                int(rec["eligible"]),
                int(rec["censored"]),
                int(rec["physical_fall"]),
                int(rec["guard_reset"]),
                int(rec["hit"]),
                int(rec["returned"]),
                int(rec["exact"]),
                int(rec["exact_composite"]),
                rec["reason"],
                "|".join(rec["details"]),
            ])
        attempt_cur.clear()

    def attempt_phase_reason(event, in_hold):
        if attempt_cur.get("exact", False):
            phase = "post_strike"
        elif in_hold:
            phase = "hold"
        else:
            phase = "pre_strike"
        return f"{event}_{phase}"

    # HitterPure station ruler on advancing swing frames only. Every record carries a close-out
    # reason so a fall/switch abort can never masquerade as a successfully completed swing.
    hp_track = hp_cfg is not None
    hp_records = []
    hp_cur = {}

    def hp_reset():
        hp_cur.clear()
        hp_cur.update(
            max_x=0.0, last_dx_abs=float("nan"), last_dy_abs=float("nan"),
            last_dxy=float("nan"), last_yaw_abs_deg=float("nan"),
            initial_dy_abs=float("nan"), record_open=False, active=False, clip=None,
            exact=False, exact_dx_abs=float("nan"), exact_dy_abs=float("nan"),
            exact_dxy=float("nan"), exact_yaw_abs_deg=float("nan"),
            exact_composite=False,
        )

    def hp_start(c):
        if not hp_track:
            return
        if hp_cur.get("record_open", False):
            raise RuntimeError("HitterPure target replaced without finalizing its attempt")
        hp_reset()
        hp_cur["record_open"] = True
        hp_cur["clip"] = int(c)

    def hp_finalize(reason):
        if hp_cur["record_open"]:
            rec = dict(hp_cur)
            rec["reason"] = str(reason)
            hp_records.append(rec)
        hp_reset()

    hp_reset()
    # --- switch-stress (deploy-parity mid-swing clip switch; see docstring) ----------------------
    stress = (switch_stress > 0.0) and (df is None)
    require_contract(not (stress and venue_sampler is not None), "--switch-stress + venue-balls unsupported (v1)")
    require_contract(not stress or multiswing, "--switch-stress needs the multiswing protocol")
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
        attempt_start(c, vs)
        hp_start(c)

    def sample_hold():
        """Pre-swing HOLD length (multiswing only): training freezes the reference at the swing's
        first frame for U[hold_range] control steps on EVERY resample (reset AND wrap). Teleport
        mode draws nothing so its RNG stream stays byte-identical to the legacy harness."""
        if bank_schedule:
            require_contract(
                cur_venue_strike[0] is not None,
                "BankExam hold requested before a schedule item was applied",
            )
            return int(cur_venue_strike[0].hold_steps)
        if not multiswing:
            return 0
        return int(rng.integers(int(hold_range[0]), int(hold_range[1]) + 1))

    ready_mode = str(ready_state_contract["mode"])
    per_clip_ready_sha = {
        int(item["clip"]): str(item["ready_state_sha256"])
        for item in ready_state_contract.get("per_clip", [])
    }
    zero_last_action = np.zeros(31, dtype=np.float64)

    def reset_question_state(c, ts):
        """Install and verify the physical state used before a question's first actor step."""
        if ready_mode == FORMAL_READY_STATE_MODE:
            robot.reset_to_named_keyframe("stand")
            expected_sha = str(ready_state_contract["sha256"])
        elif ready_mode == TEACHER_REFERENCE_READY_STATE_MODE:
            r = refs_table[ts]
            robot.reset_to_reference(
                root_pos=r["body_pos_w"][ROOT_TRACKED_IDX],
                root_quat=r["body_quat_w"][ROOT_TRACKED_IDX],
                root_lin_w=r["body_lin_vel_w"][ROOT_TRACKED_IDX],
                root_ang_w=r["body_ang_vel_w"][ROOT_TRACKED_IDX],
                q_artic=r["joint_pos"], qd_artic=r["joint_vel"],
            )
            expected_sha = per_clip_ready_sha.get(int(c), "")
        else:
            raise SystemExit(f"[FATAL] unsupported ready-state mode {ready_mode!r}")
        actual = robot.ready_state_snapshot(ready_mode, zero_last_action)
        require_contract(
            actual["sha256"] == expected_sha,
            f"ready-state reset is not reproducible for clip {c}: "
            f"expected {expected_sha}, observed {actual['sha256']}",
        )
        attempt_ready[0] = actual

    def mark_continuous_ready(last_action_value):
        attempt_ready[0] = robot.ready_state_snapshot(
            CONTINUOUS_READY_STATE_MODE, last_action_value
        )

    def fresh_swing():
        """Sample a clip, install the declared ready state, and arm its target."""
        clip, vs = sample_swing()
        ts = int(seg_start[clip])
        reset_question_state(clip, ts)
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
            attempt_start(c)
            hp_start(c)
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
            attempt_ready[0] = robot.ready_state_snapshot(
                DEPLOY_NOMINAL_READY_STATE_MODE, zero_last_action
            )
            require_contract(
                attempt_ready[0]["sha256"] == ready_state_contract["sha256"],
                "deploy nominal ready state did not reproduce its preflight hash",
            )
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

    last_action = zero_last_action.copy()
    if df is None:
        clip, time_step = fresh_swing()
        hold_left = sample_hold()
    else:
        clip, time_step = df_start_episode()
        hold_left = 0
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
        # --hold-ref stand: the multiswing pre-swing HOLD imitates READY STAND (joint refs =
        # default_q, ref vel = 0) — lockstep with the 2026-07-05+ training hold semantics
        # (commands.py), instead of the frozen windup-frame reference the pre-07-05 generations
        # trained on. Default "clip" keeps the frozen-frame reference byte-identical. The refs
        # swap happens BEFORE build_obs AND the tracking-guard terminations read `refs`, exactly
        # like training grades its own hold against the stand command.
        if hold_ref == "stand" and df is None and training_hold_protocol and hold_left > 0:
            refs = stand_hold_refs(refs, policy.default_q)
        # DF hold/rest = READY-STAND reference (2026-07-05, lockstep with training
        # commands.joint_pos + C++ pp_policy level-0): a frozen clock imitates the
        # default stand (joint refs = default_q, ref vel = 0), not frame 0's
        # asymmetric mid-crouch. Swing phases feed the raw clip refs unchanged.
        if df is not None and dfs["phase"] != "swing":
            refs = dict(refs)
            refs["joint_pos"] = policy.default_q
            refs["joint_vel"] = np.zeros_like(refs["joint_vel"])
            # 177 hitter: hold/rest keeps the SAMPLED station live (world anchor), exactly like
            # training (pbase pays through the hold and the Δ obs is real). Feeding Δ=0 here was
            # tried first ("already at station" — the mocap-dropout fallback) and is WRONG as the
            # nominal hold: it removes the only signal anchoring the base, and the policy free-
            # wanders 1-2 m during holds then falls off-station (2026-07-06 CSV phase analysis:
            # falls at torso x 1.0-2.0 m with station boxes at ±0.1). Δ=0 stays the DROPOUT
            # fallback only. PP_DF_HOLD_DZERO=1 restores the old pinning for A/B.
            if policy.hitter and os.environ.get("PP_DF_HOLD_DZERO", "0") == "1":
                racket.base_target_pos_w = robot.body_pos(robot.pelvis_bid)[:2].copy()
        obs, base_quat_w, ra_pos, ra_quat, refa_pos, refa_quat = build_obs(
            refs, robot, racket, last_action, policy.default_q, deploy_parity=policy.deploy_parity,
            face_command=getattr(policy, "face_command", False), hitter=policy.hitter,
            station=getattr(policy, "station_obs", False), hitter_pure=policy.hitter_pure)

        mean = policy.action(obs, time_step)
        action = (mean if noise_scale <= 0.0 else
                  mean + noise_scale * std_vec
                  * attempt_action_noise_rng[0].standard_normal(31))
        last_action = action.copy()

        target_q = policy.default_q + action * policy.action_scale
        # --qdes-clamp: clamp the processed q_des to the soft joint limits BEFORE the PD — the
        # training ClampedJointPositionAction (hope_actions.py, default ON since 2026-07-06) and
        # the C++ deploy runner (pp_joint_limits.hpp) both do; without this flag the MuJoCo exam
        # is the only leg of train/deploy/eval that grants unclamped q_des torque.
        if qdes_clamp:
            target_q = np.clip(target_q, robot.soft_jnt_lo, robot.soft_jnt_hi)
        tau = robot.apply_pd_and_step(target_q, policy.kp, policy.kd, decimation)
        ep_len += 1

        # The policy above consumes the current clock. Isaac then grades the post-physics state
        # after MotionCommand has advanced that clock. Keep those two instants distinct: reusing the
        # actor-input tts here delayed exact-strike grading by one 20 ms control step.
        clock_advances = ((dfs["phase"] == "swing") if df is not None else
                          not (training_hold_protocol and hold_left > 0))
        racket.time_to_strike = post_step_time_to_strike(
            racket.time_to_strike, step_dt, clock_advances
        )

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
        if hp_track:
            swing_now = (dfs["phase"] == "swing") if df is not None else \
                (not multiswing or hold_left <= 0)
            if swing_now:
                base_pos = robot.body_pos(robot.pelvis_bid)
                dxy = np.asarray(racket.station_pos_w, np.float64) - base_pos[:2]
                dx_abs, dy_abs = abs(float(dxy[0])), abs(float(dxy[1]))
                fwd = mat_from_quat(bq)[:2, 0]
                yaw_abs_deg = abs(math.degrees(math.atan2(float(fwd[1]), float(fwd[0]))))
                if not hp_cur["active"]:
                    hp_cur["initial_dy_abs"] = dy_abs
                    hp_cur["clip"] = int(clip)
                hp_cur["max_x"] = max(hp_cur["max_x"], dx_abs)
                hp_cur["last_dx_abs"] = dx_abs
                hp_cur["last_dy_abs"] = dy_abs
                hp_cur["last_dxy"] = float(np.linalg.norm(dxy))
                hp_cur["last_yaw_abs_deg"] = yaw_abs_deg
                hp_cur["active"] = True
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
                # 判卷按该 clip 的实际击球面取法向(离线常量表;表没开 = 现役单面行为逐位不变)。
                # boxes 模式的参考目标法向在预计算处已同乘同一符号;venue/bank 的需求法向来自球/规划器,
                # 不翻——翻的只是"我们给策略记分的那一面"。
                nrm = robot.racket_normal_w(sign=face_sign_for_clip(clip))
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
                attempt_hit = False
                attempt_returned = False
                if venue_sampler is not None and cur_venue_strike[0] is not None:
                    vs = cur_venue_strike[0]
                    ret = score_virtual_return(
                        virtual_return_scorer, vs, racket_pos_w=robot.racket_pos(),
                        racket_vel_w=act_vel_w, racket_normal_w=nrm, pos_err=pos_err)
                    venue["all"].add(ret, tgt_speed)
                    venue[CLIP_NAMES[clip]].add(ret, tgt_speed)
                    attempt_hit = bool(ret.contacted)
                    attempt_returned = bool(ret.landed_ok)
                    # COUNTERFACTUAL: same achieved pos/vel/pos_err, DEMANDED normal swapped in —
                    # isolates the normal channel (deterministic rescore, no RNG involved).
                    ret_cf = score_virtual_return(
                        virtual_return_scorer, vs, racket_pos_w=robot.racket_pos(),
                        racket_vel_w=act_vel_w, racket_normal_w=vs.target_normal_w,
                        pos_err=pos_err)
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
                pp = pos_err < STRIKE_POS_THRESH
                pv = vel_err < STRIKE_VEL_THRESH
                pn = nrm_err_deg < STRIKE_NORMAL_THRESH_DEG
                attempt_mark_exact(
                    pp and pv and pn, hit=attempt_hit, returned=attempt_returned
                )
                if hp_track:
                    base_pos = robot.body_pos(robot.pelvis_bid)
                    dxy = np.asarray(racket.station_pos_w, np.float64) - base_pos[:2]
                    fwd = mat_from_quat(robot.body_quat(robot.pelvis_bid))[:2, 0]
                    hp_cur["exact"] = True
                    hp_cur["exact_dx_abs"] = abs(float(dxy[0]))
                    hp_cur["exact_dy_abs"] = abs(float(dxy[1]))
                    hp_cur["exact_dxy"] = float(np.linalg.norm(dxy))
                    hp_cur["exact_yaw_abs_deg"] = abs(
                        math.degrees(math.atan2(float(fwd[1]), float(fwd[0])))
                    )
                    hp_cur["exact_composite"] = bool(pp and pv and pn)
                # --- per-strike CSV row (one line per exact-strike sample) ---
                if strike_csv_writer is not None:
                    racket_pos_w = robot.racket_pos()
                    tgt_pos_w = racket.racket_target_pos_w
                    base_pos_w = robot.body_pos(robot.pelvis_bid)
                    strike_csv_writer.writerow([
                        mode_label, step, len(ep_lengths), attempt_cur.get("attempt_id", -1),
                        attempt_cur.get("schedule_index", -1),
                        attempt_cur.get("question_sequence_index", -1),
                        attempt_cur.get("bank_row", -1),
                        attempt_cur.get("question_id", ""),
                        attempt_cur.get("repeat", 0),
                        attempt_cur.get("hold_steps", 0),
                        attempt_cur.get("attempt_seed", 0),
                        attempt_cur.get("schedule_sha256", ""),
                        CLIP_NAMES[clip], f"{racket.swing_sign:+.0f}",
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
            # Training-like: reference-relative tracking guards are swing-only. A held reset
            # intentionally combines ready-stand joints with the next clip's windup reference;
            # grading that mixed state against clip envelopes kills valid questions before the
            # actor reaches strike. Absolute tilt/height guards below remain live during holds.
            in_training_hold = training_hold_protocol and hold_left > 0
            # Under --switch-stress the
            # tracking guards are OFF (the reference jump fires them spuriously; the question
            # is deploy falls) — balance terminations + timeout only.
            reasons = (
                [] if (stress or in_training_hold)
                else check_terminations(refs, robot, ra_pos, ra_quat, refa_pos, refa_quat)
            )
            if training_hold_protocol:
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
            in_hold_now = ((dfs["phase"] != "swing") if df is not None else
                           (training_hold_protocol and hold_left > 0))
            close_reason = attempt_phase_reason(
                "fall" if terminated else "timeout", in_hold=in_hold_now
            )
            attempt_finalize(close_reason, reasons if terminated else ("episode_timeout",))
            if hp_track:
                hp_finalize(close_reason)
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
            if bank_schedule and venue_sampler.exhausted:
                break
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
            if training_hold_protocol and hold_left > 0:
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
                    attempt_finalize("completed")
                    if hp_track:
                        hp_finalize("completed")
                    if one_question_reset:
                        ep_lengths.append(ep_len)
                        ep_len = 0
                    if bank_schedule and venue_sampler.exhausted:
                        break
                    swing_from_switch = False       # a natural wrap starts a CLEAN swing
                    clip, vs = sample_swing()
                    time_step = int(seg_start[clip])
                    if not multiswing or one_question_reset:
                        reset_question_state(clip, time_step)
                        last_action = np.zeros(31)
                    else:
                        # Explicit continuity diagnostics start from the previous question's
                        # terminal physical/action state; record that truth per attempt instead of
                        # falsely attaching the common-reset hash.
                        mark_continuous_ready(last_action)
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
                switch_reason = attempt_phase_reason("switch", in_hold=not mid)
                attempt_finalize(switch_reason)
                if hp_track:
                    hp_finalize(switch_reason)
                clip, vs = sample_swing()
                time_step = int(seg_start[clip])
                mark_continuous_ready(last_action)
                apply_target(clip, vs)
                hold_left = sample_hold()
        else:
            # --- deploy-faithful swing schedule: hold -> play the WHOLE clip once -> rest -> repeat.
            # NO teleports; last_action carries across swings (the deployed policy runs continuously).
            if dfs["phase"] == "swing":
                if not dfs["completed"] and racket.time_to_strike <= exact_tol:
                    # Post-physics clock: the exact frame reached above counts immediately, not on
                    # the next actor step. This shares the same contact instant as strike grading.
                    dfs["completed"] = True
                    dfs["swing_completions"][clip] += 1
                if time_step >= int(seg_start[clip]) + int(seg_len[clip]) - 1:
                    # final clip frame has been played -> rest at the NEXT swing's windup
                    attempt_finalize("completed")
                    if hp_track:
                        hp_finalize("completed")
                    mark_continuous_ready(last_action)
                    clip, time_step = df_new_swing("rest", df["rest_steps"])
                else:
                    time_step += 1
            else:  # "hold" (episode start) or "rest" (between swings): clock pinned at the windup
                dfs["left"] -= 1
                if dfs["left"] <= 0:
                    dfs["phase"] = "swing"
                    dfs["swing_starts"][clip] += 1
                    dfs["completed"] = False
                    # restore the armed swing's SAMPLED station (the hitter hold pins the base
                    # target to the live base for Δ=0; no-op for non-hitter contracts)
                    racket.base_target_pos_w = racket.station_pos_w.copy()
                    time_step += 1                                # first advancing frame after windup
        racket.update_strike_timing(clip, time_step)

    if attempt_cur.get("open", False):
        in_hold_now = ((dfs["phase"] != "swing") if df is not None else
                       (training_hold_protocol and hold_left > 0))
        trunc_reason = attempt_phase_reason("truncated", in_hold=in_hold_now)
        attempt_finalize(trunc_reason, ("rollout_step_budget",))
        if hp_track:
            hp_finalize(trunc_reason)

    if bank_schedule:
        expected_ids = tuple(venue_sampler.schedule_question_ids)
        actual_ids = tuple(record["question_id"] for record in attempt_records)
        censored_count = sum(bool(record.get("censored", False)) for record in attempt_records)
        if (len(attempt_records) != len(venue_sampler.schedule)
                or not venue_sampler.exhausted or censored_count):
            raise SystemExit(
                "[FATAL] BankExam safety step cap exhausted before the immutable paper completed: "
                f"finished={len(attempt_records)}/{len(venue_sampler.schedule)}, "
                f"censored={censored_count}, n_steps_cap={n_steps}. Raise --steps or use "
                "--steps 0 for the computed cap."
            )
        require_contract(
            actual_ids == expected_ids,
            "BankExam question-id order differs from its immutable schedule",
        )
        require_contract(
            len(set(actual_ids)) == len(actual_ids),
            "BankExam question-id sequence contains a duplicate/wrap",
        )
        require_contract(
            venue_sampler.asked == venue_sampler.selected
            and all(value == 0 for value in venue_sampler.wrapped)
            and venue_sampler.n_samples == len(venue_sampler.schedule),
            "BankExam sampler denominator/cursor counters disagree with the completed schedule",
        )

    total_term = n_term_early + n_timeout
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
        evaluation_contract_exact=bool(policy.evaluation_contract_exact),
        ready_state_mode=ready_state_contract["mode"],
        ready_state_sha256=ready_state_contract["sha256"],
        mjcf_sha256=mjcf_sha256,
        execution_contract_sha256=execution_contract_sha256,
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
    if bank_schedule:
        out["exam_schedule"] = {
            "schema_version": 1,
            "sha256": venue_sampler.schedule_sha256,
            "bank_sha256": venue_sampler.bank_sha256,
            "seed": venue_sampler.schedule_seed,
            "size": len(venue_sampler.schedule),
            "one_question_reset": bool(one_question_reset),
            "question_id_order": [record["question_id"] for record in attempt_records],
            "items": [
                {
                    "schedule_index": record["schedule_index"],
                    "question_sequence_index": record["question_sequence_index"],
                    "clip": record["clip"],
                    "bank_row": record["bank_row"],
                    "question_id": record["question_id"],
                    "repeat": record["repeat"],
                    "hold_steps": record["hold_steps"],
                    "attempt_seed": record["attempt_seed"],
                    "ready_state_mode": record["ready_state_mode"],
                    "ready_state_sha256": record["ready_state_sha256"],
                    "mjcf_sha256": record["mjcf_sha256"],
                    "execution_contract_sha256": record["execution_contract_sha256"],
                    "finalize_reason": record["reason"],
                    "eligible": record["eligible"],
                    "censored": record["censored"],
                    "physical_fall": record["physical_fall"],
                    "guard_reset": record["guard_reset"],
                    "hit": record["hit"],
                    "returned": record["returned"],
                }
                for record in attempt_records
            ],
        }
    out["attempts"] = summarize_attempt_records(attempt_records, num_clips)
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
        attempt_groups = {
            "all": out["attempts"],
            "forehand": out["attempts"]["per_clip"].get("forehand", {}),
            "backhand": out["attempts"]["per_clip"].get("backhand", {}),
        }
        for venue_key, attempt_key in (
            ("all", "all"), ("forehand", "forehand"), ("backhand", "backhand"),
            ("cf_all", "all"), ("cf_forehand", "forehand"), ("cf_backhand", "backhand"),
        ):
            metrics = out["venue"][venue_key]
            attempts = int(attempt_groups[attempt_key].get("n_attempts", 0))
            metrics["n_attempts"] = attempts
            metrics["exact_reach_rate_per_attempt"] = (
                metrics["n_strikes"] / attempts if attempts else float("nan")
            )
            metrics["contact_rate_per_attempt"] = (
                metrics["contacted"] / attempts if attempts else float("nan")
            )
            metrics["return_success_rate_per_attempt"] = (
                metrics["landed_ok"] / attempts if attempts else float("nan")
            )
    if hp_track:
        reason_counts = Counter(r["reason"] for r in hp_records)
        completed_records = [r for r in hp_records if r["reason"] == "completed"]
        exact_records = [r for r in hp_records if r["exact"]]

        def hp_mean(records, key):
            values = np.asarray([r[key] for r in records if math.isfinite(float(r[key]))], np.float64)
            return float(values.mean()) if values.size else float("nan")

        def hp_group(records):
            exact = [r for r in records if r["exact"]]
            return dict(
                n_records=len(records),
                n_measured=sum(bool(r["active"]) for r in records),
                n_completed=sum(r["reason"] == "completed" for r in records),
                n_exact_alive=len(exact),
                exact_station_dx_abs_mean=hp_mean(exact, "exact_dx_abs"),
                exact_station_dy_abs_mean=hp_mean(exact, "exact_dy_abs"),
                exact_station_dxy_mean=hp_mean(exact, "exact_dxy"),
                exact_yaw_abs_deg_mean=hp_mean(exact, "exact_yaw_abs_deg"),
                exact_composite_rate=(
                    sum(bool(r["exact_composite"]) for r in exact) / len(exact)
                    if exact else float("nan")
                ),
            )

        peaks = np.asarray([r["max_x"] for r in completed_records], np.float64)
        y_bins = (("lt_0p2", 0.0, 0.2), ("0p2_to_0p4", 0.2, 0.4),
                  ("ge_0p4", 0.4, float("inf")))
        out["hp"] = dict(
            n_attempts=len(hp_records),
            n_swings_measured=sum(bool(r["active"]) for r in hp_records),
            n_completed=len(completed_records),
            n_exact_alive=len(exact_records),
            finalize_reason_counts=dict(reason_counts),
            base_x_excursion_mean=float(peaks.mean()) if peaks.size else float("nan"),
            base_x_excursion_p90=float(np.percentile(peaks, 90.0)) if peaks.size else float("nan"),
            base_x_excursion_max=float(peaks.max()) if peaks.size else float("nan"),
            base_x_at_completed_end_mean=hp_mean(completed_records, "last_dx_abs"),
            exact_station_dx_abs_mean=hp_mean(exact_records, "exact_dx_abs"),
            exact_station_dy_abs_mean=hp_mean(exact_records, "exact_dy_abs"),
            exact_station_dxy_mean=hp_mean(exact_records, "exact_dxy"),
            exact_yaw_abs_deg_mean=hp_mean(exact_records, "exact_yaw_abs_deg"),
            per_clip={
                CLIP_NAMES.get(c, f"clip_{c}"): hp_group(
                    [r for r in hp_records if r["clip"] == c]
                )
                for c in range(num_clips)
            },
            initial_station_y_bins={
                name: hp_group([
                    r for r in hp_records
                    if math.isfinite(float(r["initial_dy_abs"])) and lo <= r["initial_dy_abs"] < hi
                ])
                for name, lo, hi in y_bins
            },
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
    p.add_argument("--steps", type=int, default=None,
                   help="rollout safety cap. Formal BankExam must finish its entire immutable "
                        "schedule before this cap; 0 computes a conservative cap from K and the "
                        "episode timeout. Default: auto for bank, 1200 for other target sources.")
    p.add_argument("--sim-dt", type=float, default=0.005, help="MuJoCo physics dt (Isaac used 0.005)")
    p.add_argument("--decimation", type=int, default=4, help="physics substeps per 50 Hz control step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-passive", action="store_true",
                   help="DEPRECATED shorthand for --passive-damping mjcf --frictionloss mjcf")
    p.add_argument("--passive-damping", choices=["auto", "zero", "mjcf"], default="auto",
                   help="native MJCF viscous joint damping. auto: zero (the actuator kd supplies "
                        "the current training plant's viscous damping).")
    p.add_argument("--frictionloss", choices=["auto", "zero", "mjcf"], default="auto",
                   help="MJCF constant-Nm dry joint friction. auto: schema-3 zero when the PhysX "
                        "coefficient is zero, otherwise a labelled direct-number diagnostic proxy; "
                        "legacy HitterPure keeps MJCF and other legacy contracts use zero.")
    p.add_argument("--pd-mode", choices=["auto", "explicit", "implicit"], default="auto",
                   help="explicit: torque=kp*e-kd*qd. implicit: kp torque + kd as passive damping via "
                        "MuJoCo implicitfast. auto: schema-3 per-joint actuator contract, otherwise "
                        "implicit for legacy HitterPure and explicit for other legacy actors.")
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
    p.add_argument("--anchor-term-z", type=float, default=None,
                   help="override the anchor_pos termination threshold (|ref torso z - robot torso "
                        "z|, training default 0.25 m). Same class as --ee-term-z: a TRAINING reset "
                        "guard. Bank/face policies deviate from the replay reference BY DESIGN "
                        "(2026-07-06: 456 hold-phase kills on arm B before any swing) — raise to "
                        "100 for bank exams; real falls stay caught by fall_root_z.")
    p.add_argument("--anchor-term-ori", type=float, default=None,
                   help="override the anchor_ori termination threshold (projected-gravity z gap, "
                        "training default 0.8). Same training-guard class as --anchor-term-z.")
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
    p.add_argument("--targets-center", action="store_true",
                   help="[110-D hitter_pure] use per-clip racket position/velocity box centers "
                        "instead of random draws; the station still samples its full box.")
    p.add_argument("--hp-base-target-range", nargs=4, type=float, default=None,
                   metavar=("X_LO", "X_HI", "Y_LO", "Y_HI"),
                   help="[110-D hitter_pure] explicit world station box around env origin. "
                        "Overrides ONNX metadata; required for metadata-less models unless "
                        "--allow-hitter-pure-defaults is passed.")
    p.add_argument("--allow-hitter-pure-defaults", action="store_true",
                   help="[110-D only] explicitly allow built-in task-YAML mirror boxes when ONNX "
                        "geometry metadata is absent. Without this flag missing provenance is fatal.")
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
    p.add_argument("--hold-steps-range", nargs=2, type=int, default=None,
                   help="[--reset-mode multiswing] pre-swing hold U[lo,hi] control steps at every "
                        "swing start. Default: ONNX motion_hold_steps_range metadata; legacy "
                        "non-110 models fall back to 0 100. Metadata-less 110 requires CLI.")
    p.add_argument("--episode-length-s", type=float, default=None,
                   help="DIAGNOSTIC compatibility override for exports that predate the "
                        "episode_length_s metadata. A value that disagrees with present metadata "
                        "is fatal; supplying it for an unbound old model marks the score inexact.")
    # --- MODE B: distribution-driven realism eval (2026-07-04; see module docstring TARGET SOURCE
    # + scripts/venue_ball_sampler.py for frames/geometry/caveats). Default boxes = mode A,
    # byte-identical to the pre-existing behavior.
    p.add_argument("--exam-bank", default=None,
                   help="[bank] stage-1 exam bank npz (gen_stage1_questions.py --split exam "
                        "product). REQUIRED with --target-source bank; loaded through "
                        "stage1_question_bank.load_question_bank so the meta guards "
                        "(grip_applied/rally_yaw_applied) are enforced, never bypassed.")
    p.add_argument("--exam-schedule-k", type=int, default=None,
                   help="[bank] fixed total number of stratified exam questions, sampled without "
                        "replacement when the immutable schedule is materialized. Default: every "
                        "exam-bank row exactly once.")
    p.add_argument("--exam-schedule-json", default=None,
                   help="[bank] shared balanced schema-v3 schedule artifact produced by the "
                        "evaluator-owned BankExam adapter. The artifact is content-addressed, "
                        "validated against the complete exam bank, and consumed unchanged by "
                        "both MuJoCo and Isaac. Mutually exclusive with --exam-schedule-k.")
    p.add_argument("--exam-continuity-diagnostic", action="store_true",
                   help="[bank, DIAGNOSTIC ONLY] keep robot/action state across scheduled questions "
                        "instead of the formal one-question/one-reset ruler. Uses the same finite "
                        "paper but stamps evaluation_contract_exact=false.")
    p.add_argument(
        "--ready-state", choices=["auto", "stand-keyframe", "teacher-reference"],
        default="auto",
        help="question initial-state contract. auto = the MJCF named 'stand' keyframe for BankExam, "
             "the deploy nominal stand for --deploy-faithful, and clip-start teacher reference for "
             "legacy within-lineage diagnostics. "
             "stand-keyframe resets the complete MuJoCo state with mj_resetDataKeyframe then "
             "forces qvel/act/ctrl/last_action to zero; a missing key is fatal. "
             "teacher-reference is candidate-dependent and always stamps the evaluation inexact; "
             "BankExam additionally requires --allow-inexact-contract.",
    )
    p.add_argument("--allow-inexact-contract", action="store_true",
                   help="DIAGNOSTIC ONLY: allow a legacy/unbound exam bank, missing old artifact "
                        "provenance, or an explicit old episode timeout. The summary is stamped "
                        "evaluation_contract_exact=false and must not be booked as a formal score.")
    p.add_argument("--target-source", choices=["boxes", "venue-balls", "bank"], default="boxes",
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
    # --- STAGE EXAMS (2026-07-06, franco's staged-question doctrine) -------------------------
    p.add_argument("--venue-contact-fixed", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                   help="[venue-balls] STAGE-1 exam: pin the incoming-ball contact point (env "
                        "frame; e.g. the v5-bh _cal anchor blade point). Default: venue box.")
    p.add_argument("--venue-spin-max", type=float, default=None,
                   help="[venue-balls] cap the isotropic incoming-spin magnitude (rad/s); 0 = "
                        "spinless stage-1 balls. Default: 34 (venue matchlike).")
    p.add_argument("--venue-vel-box", nargs=6, type=float, default=None,
                   metavar=("VX_LO", "VX_HI", "VY_LO", "VY_HI", "VZ_LO", "VZ_HI"),
                   help="[venue-balls] override the incoming velocity box (speed tier / staged "
                        "curriculum). Default: venue matchlike.")
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
    p.add_argument("--qdes-clamp", action="store_true",
                   help="clamp the decoded q_des to the MJCF soft joint limits (0.9 x range about "
                        "the midpoint = Isaac soft_joint_pos_limit_factor) before the PD — matches "
                        "BOTH the training ClampedJointPositionAction (default ON since "
                        "2026-07-06) and the C++ deploy runner (pp_joint_limits.hpp). Default OFF "
                        "= legacy exam behavior, byte-identical to every score on the books. "
                        "Recommended ON for every new exam; the state is printed in the report "
                        "header either way (fixE retrial 2026-07-08: unflagged, the exam is the "
                        "only unclamped leg of train/deploy/eval).")
    p.add_argument("--mount-normal-sign-per-clip", nargs="+", type=float, default=None,
                   metavar="SIGN",
                   help="per-clip striking-FACE sign table (one +1/-1 per --motion-files clip, in "
                        "clip order; e.g. 1 -1 = forehand red/+Y face, backhand black/-Y face). "
                        "Score each swing's REAL striking face (franco 2026-07-09 '哪面拍子超前就是"
                        "哪面'): the achieved normal AND the boxes-mode per-clip reference target "
                        "normal are both multiplied by the clip's sign before the face error — same "
                        "semantics as training racket.mount_normal_sign_per_clip. Signs are OFFLINE "
                        "per-clip constants from the reference clip's contact frame "
                        "(scripts/suggest_face_sign.py), NEVER derived from the live paddle velocity "
                        "at eval time. Default OFF = legacy single-face scoring, byte-identical to "
                        "every score on the books. The table serves the metric/reference channels "
                        "ONLY (2026-07-09 face-frame ruling): it is REJECTED with --target-source "
                        "bank (bank scoring/obs are +Y/A-frame on both sides) and with any 179/"
                        "181-D face-obs model (the face lane is +Y/A-frame in training) — both "
                        "exit 2. Legit use: boxes/venue scoring of non-face models on flipped-"
                        "face clips.")
    p.add_argument("--hold-ref", choices=["auto", "clip", "stand"], default="auto",
                   help="multiswing pre-swing HOLD reference semantics. 'auto' uses ONNX metadata "
                        "(legacy non-110 fallback: clip). 'clip' = "
                        "freeze the windup frame's raw clip reference — what pre-2026-07-05 "
                        "generations trained on. 'stand' = READY-STAND (joint refs = default_q, "
                        "ref vel = 0; the 2026-07-05+ commands.py hold semantics) — use it for "
                        "arms trained on/after 2026-07-05 or the hold segment grades against a "
                        "generation-mismatched reference (the 07-07 incident shape). Inert outside "
                        "the multiswing protocol (teleport / --deploy-faithful).")
    args = p.parse_args()

    if args.venue_fixed_normal and args.target_source != "venue-balls":
        raise SystemExit("[FATAL] --venue-fixed-normal only means something with "
                         "--target-source venue-balls")
    if args.target_source == "bank":
        if not args.exam_bank:
            raise SystemExit("[FATAL] --target-source bank requires --exam-bank <exam npz> "
                             "(gen_stage1_questions.py --split exam product)")
        if args.venue_contact_fixed or args.venue_spin_max is not None or args.venue_vel_box:
            raise SystemExit("[FATAL] --target-source bank: the exam paper comes SOLELY from "
                             "the bank; --venue-contact-fixed/--venue-spin-max/--venue-vel-box "
                             "would silently contradict it — drop them.")
        if args.deploy_faithful:
            raise SystemExit("[FATAL] --target-source bank + --deploy-faithful is unsupported "
                             "(v1): the df swing scheduler owns its own resample path.")
        if args.exam_schedule_k is not None and args.exam_schedule_k <= 0:
            raise SystemExit("[FATAL] --exam-schedule-k must be a positive integer")
        if args.exam_schedule_json and args.exam_schedule_k is not None:
            raise SystemExit(
                "[FATAL] --exam-schedule-json and --exam-schedule-k are mutually exclusive"
            )
    elif (args.exam_schedule_k is not None or args.exam_schedule_json
          or args.exam_continuity_diagnostic):
        raise SystemExit(
            "[FATAL] --exam-schedule-k/--exam-schedule-json/"
            "--exam-continuity-diagnostic require --target-source bank"
        )
    ready_state_mode = resolve_ready_state_mode(
        args.ready_state,
        target_source=args.target_source,
        deploy_faithful=args.deploy_faithful,
        allow_inexact_contract=args.allow_inexact_contract,
    )
    if args.steps is None:
        args.steps = 0 if args.target_source == "bank" else 1200
    if args.steps <= 0 and args.target_source != "bank":
        raise SystemExit("[FATAL] --steps must be positive outside formal BankExam")
    if args.switch_stress > 0.0:
        if args.deploy_faithful:
            raise SystemExit("[FATAL] --switch-stress + --deploy-faithful is unsupported (v1): "
                             "the df swing scheduler owns its own clip clock.")
        if args.target_source != "boxes":
            raise SystemExit("[FATAL] --switch-stress + --target-source "
                             f"{args.target_source} is unsupported (v1): one stressor per "
                             "protocol.")

    # Apply eval-only overrides to the module globals BEFORE any precompute/rollout reads them.
    global STRIKE_PHASE_PER_CLIP, RACKET_POS_Z_RANGE, TERM_EE_POS_Z, POS_RANGE_PER_CLIP, VEL_RANGE_PER_CLIP
    global TERM_ANCHOR_POS_Z, TERM_ANCHOR_ORI, MOUNT_NORMAL_SIGN_PER_CLIP
    if args.mount_normal_sign_per_clip is not None:
        # 互斥守卫①(2026-07-09 单翻病定案):bank 考卷按 +Y(A)约定双侧不翻判分(题库行原样
        # 进 obs 与打分),传符号表 = 实测翻面 vs A 约定目标 = 反手拍面误差被打成 ~180°−x 的错判。
        if args.target_source == "bank":
            raise SystemExit("[FATAL] --mount-normal-sign-per-clip is incompatible with "
                             "--target-source bank: bank scoring/obs are +Y(A)-frame on BOTH "
                             "sides by design (双不翻;2026-07-09 face-frame 定案). Drop the flag.")
        _signs = tuple(float(s) for s in args.mount_normal_sign_per_clip)
        # fail-loud:符号表长度必须 = clip 数(照训练侧 _mount_signs_cfg / _strike_phases_cfg 先例),
        # 符号只认 ±1 —— 静默截断/回退会让某个 clip 按错误的一面判分还不吭声。
        if len(_signs) != len(args.motion_files):
            raise SystemExit(f"[FATAL] --mount-normal-sign-per-clip has {len(_signs)} entries but "
                             f"{len(args.motion_files)} motion file(s) were given — one striking-"
                             f"face sign per clip, in --motion-files order.")
        if any(s not in (1.0, -1.0) for s in _signs):
            raise SystemExit(f"[FATAL] --mount-normal-sign-per-clip entries must be +1 or -1, "
                             f"got {_signs}")
        MOUNT_NORMAL_SIGN_PER_CLIP = _signs
        print(f"[mj-sim2sim] OVERRIDE mount_normal_sign_per_clip -> {_signs} (判卷按每 clip 的实际"
              f"击球面翻面再算拍面误差;eval-only,与训练 racket.mount_normal_sign_per_clip 同语义)")
    if args.strike_phase_per_clip is not None:
        STRIKE_PHASE_PER_CLIP = tuple(args.strike_phase_per_clip)
        print(f"[mj-sim2sim] OVERRIDE strike_phase_per_clip -> {STRIKE_PHASE_PER_CLIP} (eval-only)")
    if args.anchor_term_z is not None:
        TERM_ANCHOR_POS_Z = float(args.anchor_term_z)
        print(f"[mj-sim2sim] OVERRIDE anchor_pos termination threshold -> {TERM_ANCHOR_POS_Z} m "
              f"(training reset guard; eval-only)")
    if args.anchor_term_ori is not None:
        TERM_ANCHOR_ORI = float(args.anchor_term_ori)
        print(f"[mj-sim2sim] OVERRIDE anchor_ori termination threshold -> {TERM_ANCHOR_ORI} "
              f"(training reset guard; eval-only)")
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
    require_contract(
        abs(step_dt - 0.02) < 1e-9,
        f"control dt {step_dt} != 0.02 (50 Hz). adjust --sim-dt/--decimation",
    )

    print(f"[mj-sim2sim] onnx={args.onnx}")
    print(f"[mj-sim2sim] mjcf={args.mjcf}")
    policy = OnnxPolicy(args.onnx, obs_norm=("off" if args.no_obs_norm else args.obs_norm))
    if ready_state_mode == TEACHER_REFERENCE_READY_STATE_MODE:
        policy.evaluation_contract_exact = False
        print(
            "[mj-sim2sim] ready state: teacher clip-start reference (candidate-dependent; "
            "within-lineage diagnostic only, evaluation_contract_exact=false)"
        )
    if args.episode_length_s is not None:
        if not math.isfinite(args.episode_length_s) or args.episode_length_s <= 0.0:
            raise SystemExit("[FATAL] --episode-length-s must be finite and positive")
        if (policy.episode_length_s_meta is not None
                and not math.isclose(args.episode_length_s, policy.episode_length_s_meta,
                                     rel_tol=0.0, abs_tol=1e-9)):
            raise SystemExit(
                f"[FATAL] --episode-length-s={args.episode_length_s:g} disagrees with ONNX "
                f"metadata {policy.episode_length_s_meta:g}; an evaluator override may not change "
                "the trained episode contract"
            )
        episode_length_s = float(args.episode_length_s)
        if policy.episode_length_s_meta is None:
            policy.evaluation_contract_exact = False
            print(
                "[mj-sim2sim] WARNING: episode timeout supplied by CLI because ONNX lacks "
                "episode_length_s; evaluation_contract_exact=false"
            )
    elif policy.episode_length_s_meta is not None:
        episode_length_s = policy.episode_length_s_meta
    else:
        episode_length_s = 10.0
        policy.evaluation_contract_exact = False
        print(
            "[mj-sim2sim] WARNING: ONNX lacks episode_length_s; using legacy 10 s fallback and "
            "marking evaluation_contract_exact=false (pass --episode-length-s only for a "
            "documented diagnostic replay)"
        )
    max_ep_len = int(round(episode_length_s / step_dt))
    require_contract(max_ep_len > 0, "episode timeout rounds to zero control steps")
    print(
        f"[mj-sim2sim] episode timeout: {episode_length_s:g} s = {max_ep_len} control steps "
        f"({'ONNX metadata' if policy.episode_length_s_meta is not None else 'diagnostic/legacy'})"
    )
    bound_schema3_plant = (
        policy.training_contract_exact == "1"
        and policy.training_contract_schema_version == "3"
    )
    if args.pd_mode == "auto":
        if bound_schema3_plant and policy.joint_actuator_types is not None:
            resolved_actuator_types = tuple(policy.joint_actuator_types)
            actuator_source = "schema-3 training contract"
        else:
            legacy_pd = "implicit" if policy.hitter_pure else "explicit"
            resolved_actuator_types = (legacy_pd,) * 31
            actuator_source = "legacy observation-width fallback"
    else:
        resolved_actuator_types = (args.pd_mode,) * 31
        actuator_source = "CLI override"
    actuator_kinds = sorted(set(resolved_actuator_types))
    pd_mode = actuator_kinds[0] if len(actuator_kinds) == 1 else "mixed"
    if args.keep_passive and (args.passive_damping != "auto" or args.frictionloss != "auto"):
        raise SystemExit(
            "[FATAL] --keep-passive is a deprecated shorthand; do not combine it with "
            "--passive-damping/--frictionloss."
        )
    if args.keep_passive:
        passive_damping_mode = frictionloss_mode = "mjcf"
        plant_source = "deprecated --keep-passive"
    else:
        passive_damping_mode = (
            "zero" if args.passive_damping == "auto" else args.passive_damping
        )
        if args.frictionloss == "auto":
            if bound_schema3_plant and policy.joint_friction_coefficients is not None:
                frictionloss_mode = (
                    "zero" if np.array_equal(
                        policy.joint_friction_coefficients, np.zeros(31, np.float64)
                    ) else "contract-proxy"
                )
            else:
                frictionloss_mode = "mjcf" if policy.hitter_pure else "zero"
        else:
            frictionloss_mode = args.frictionloss
        plant_source = "contract auto profile" if (
            args.passive_damping == "auto" and args.frictionloss == "auto"
        ) else "CLI override"
    if frictionloss_mode == "contract-proxy":
        policy.evaluation_contract_exact = False
    qdes_clamp = bool(args.qdes_clamp or policy.hitter_pure)
    formal_qdes_limits = None
    formal_execution_contract_ok = False
    if args.target_source == "bank":
        try:
            formal_qdes_limits = validate_formal_bank_execution_contract(
                policy,
                physics_step_dt_s=args.sim_dt,
                policy_step_dt_s=step_dt,
                control_decimation=args.decimation,
                qdes_clamp=qdes_clamp,
            )
            formal_execution_contract_ok = True
        except SystemExit as exc:
            if not args.allow_inexact_contract:
                raise
            policy.evaluation_contract_exact = False
            print(
                f"[mj-sim2sim] WARNING: {exc}; diagnostic escape enabled, using MJCF-derived "
                "q_des bounds and evaluation_contract_exact=false"
            )
        if policy.qdes_clamp_meta is None:
            policy.evaluation_contract_exact = False
            if not args.allow_inexact_contract:
                raise SystemExit(
                    "[FATAL] bank-eval ONNX lacks the trained qdes_clamp contract; re-export"
                )
        elif policy.qdes_clamp_meta != qdes_clamp:
            if not args.allow_inexact_contract:
                raise SystemExit(
                    f"[FATAL] evaluator q_des clamp={qdes_clamp} disagrees with training "
                    f"contract={policy.qdes_clamp_meta}"
                )
            policy.evaluation_contract_exact = False
            print(
                "[mj-sim2sim] WARNING: q_des clamp override changes the training contract; "
                "evaluation_contract_exact=false"
            )
        if policy.qdes_clamp_meta is not True:
            if not args.allow_inexact_contract:
                raise SystemExit(
                    "[FATAL] formal BankExam requires a policy trained/evaluated with q_des "
                    "clamping, matching the C++ deployment path"
                )
            policy.evaluation_contract_exact = False
        bank_profile_violations = []
        if policy.joint_actuator_types is None:
            bank_profile_violations.append("missing per-joint actuator integration contract")
        elif tuple(resolved_actuator_types) != tuple(policy.joint_actuator_types):
            bank_profile_violations.append(
                f"actuator_types={pd_mode} ({actuator_source}) disagree with training contract"
            )
        if passive_damping_mode != "zero":
            bank_profile_violations.append(
                f"passive_damping={passive_damping_mode} (required zero)"
            )
        if (
            policy.joint_friction_coefficients is not None
            and np.any(policy.joint_friction_coefficients != 0.0)
        ):
            bank_profile_violations.append(
                "non-zero PhysX dimensionless/load-dependent joint friction has no exact "
                f"MuJoCo frictionloss equivalent (resolved={frictionloss_mode})"
            )
        elif frictionloss_mode != "zero":
            bank_profile_violations.append(
                f"frictionloss={frictionloss_mode} (required zero)"
            )
        for name, value in (
            ("anchor_term_z", args.anchor_term_z),
            ("anchor_term_ori", args.anchor_term_ori),
            ("ee_term_z", args.ee_term_z),
        ):
            if value is not None:
                bank_profile_violations.append(f"{name} override={value}")
        if args.keep_passive:
            bank_profile_violations.append("deprecated keep_passive override")
        if args.deploy_faithful:
            bank_profile_violations.append("deploy_faithful is a different protocol")
        if args.switch_stress > 0.0:
            bank_profile_violations.append("switch_stress is a different protocol")
        if bank_profile_violations:
            if not args.allow_inexact_contract:
                raise SystemExit(
                    "[FATAL] formal BankExam profile violation(s): "
                    + "; ".join(bank_profile_violations)
                )
            policy.evaluation_contract_exact = False
            print(
                "[mj-sim2sim] WARNING: diagnostic BankExam profile override(s): "
                + "; ".join(bank_profile_violations)
                + "; evaluation_contract_exact=false"
            )
    if policy.hitter_pure:
        if args.target_source != "boxes":
            raise SystemExit(
                f"[FATAL] 110-D hitter_pure currently supports --target-source boxes only; "
                f"{args.target_source!r} bypasses its station-first/velocity-normal contract."
            )
        if args.pos_z_range is not None or args.eval_per_clip_vel_targets:
            raise SystemExit(
                "[FATAL] --pos-z-range/--eval-per-clip-vel-targets are legacy samplers and inert "
                "under 110-D hitter_pure; use --pos-range-per-clip/--vel-range-per-clip."
            )
        meta_signs = policy.mount_normal_sign_per_clip_meta
        if meta_signs is not None:
            if len(meta_signs) != len(args.motion_files) or any(s not in (1.0, -1.0) for s in meta_signs):
                raise SystemExit(
                    f"[FATAL] invalid mount_normal_sign_per_clip ONNX metadata: {meta_signs}"
                )
        if MOUNT_NORMAL_SIGN_PER_CLIP is None:
            if meta_signs is not None:
                MOUNT_NORMAL_SIGN_PER_CLIP = meta_signs
                print(f"[mj-sim2sim] 110 hitter_pure face signs from ONNX metadata: {meta_signs}")
            else:
                raise SystemExit(
                    "[FATAL] 110-D ONNX lacks mount_normal_sign_per_clip metadata. Re-export, pass "
                    "--mount-normal-sign-per-clip explicitly. Face signs cannot use the geometry "
                    "fallback because base HitterPure and Rally recipes have different provenance."
                )
        elif meta_signs is not None and tuple(MOUNT_NORMAL_SIGN_PER_CLIP) != tuple(meta_signs):
            print("[mj-sim2sim] WARN: CLI face signs override different ONNX metadata: "
                  f"cli={MOUNT_NORMAL_SIGN_PER_CLIP} metadata={meta_signs}")
    # 互斥守卫②(obs 维度触发,盖所有模式含 boxes/venue):face-obs 模型(179/181-D)的 face 通道
    # 在训练里永远是 +Y(A)约定(bank 行原样进 obs;hope_rewards._face_pair)。开符号表会让 boxes/
    # venue 的目标法向翻到击球面(B)喂进 obs = 训练没见过的镜像分布,判也判不对。按元数据可自省
    # (exporter: face_obs_convention=mount_plusY_A),但守卫以实测 obs 维度为准,老模型也拦。
    if policy.face_command and MOUNT_NORMAL_SIGN_PER_CLIP is not None:
        raise SystemExit("[FATAL] --mount-normal-sign-per-clip with a face-obs model "
                         f"(obs_dim={policy.obs_dim}): the face lane is +Y(A)-frame in training; "
                         "flipping target normals here feeds the policy a mirrored face command "
                         "it never saw. Drop the flag (all target sources).")
    contract_desc = {
        110: "hitter_pure: HITTER Table-I actor; world target deltas + base forward, no ref stream",
        175: "deploy_parity: racket_target_pos_b relative to racket FK, no anchor_pos/base_target",
        177: "hitter_footwork: deploy_parity + base_target_pos_b(2) station Δxy after proj_grav",
        179: "deploy_parity + FACE COMMAND tail (demanded normal 3 + rho placeholder)",
        180: "base: full 180-D BeyondMimic obs",
        181: "deploy_parity + face tail + STATION ANCHOR tail (spawn-constant world anchor Δxy 2)",
    }[policy.obs_dim]
    print(f"[mj-sim2sim] obs_dim={policy.obs_dim} ({contract_desc}) "
          f"joints={len(policy.joint_names)} "
          f"control={1/step_dt:.0f}Hz (sim_dt={args.sim_dt}, decim={args.decimation})")

    # --- obs normalization status (P0 fix #1) — a missing sidecar silently zeroes every metric for
    # a normalized-obs model, so make the state of this transform impossible to miss.
    if policy.obs_mean is not None:
        print(f"[mj-sim2sim] obs normalization: ON (sidecar {policy.obs_norm_path}; "
              f"(obs-mean)/(std+{policy.obs_eps:g}), mean|max|={np.abs(policy.obs_mean).max():.2f}, "
              f"std max={policy.obs_std.max():.2f})")
    elif policy.obs_norm_baked:
        print("[mj-sim2sim] obs normalization: ON (baked into ONNX graph)")
    elif policy.empirical_normalization is False:
        print("[mj-sim2sim] obs normalization: OFF (training metadata declares raw observations)")
    elif args.no_obs_norm:
        print("[mj-sim2sim] obs normalization: OFF (--no-obs-norm)")
    else:
        print("[mj-sim2sim] WARNING: obs normalization sidecar NOT FOUND (<onnx_dir>/obs_norm.npz). "
              "All known training runs use empirical_normalization=true while the export bakes the "
              "RAW actor — without the sidecar such a model is fed unnormalized obs and scores ~0 "
              "with a staggering/early-termination pathology. Create it with "
              "scripts/make_std_sidecar.py --checkpoint <the model_<N>.pt the ONNX came from>.")
    print(f"[mj-sim2sim] artifact_contract_exact_preflight={policy.evaluation_contract_exact}")

    # strike-phase resolution: CLI (handled above) > ONNX clip metadata > built-in legacy fallback.
    # Resolved (and printed ONCE, with its source) BEFORE any strike-frame precompute.
    if args.strike_phase_per_clip is None:
        if policy.clip_strike_phases:
            STRIKE_PHASE_PER_CLIP = policy.clip_strike_phases
            print(f"[mj-sim2sim] strike_phase_per_clip in effect: {STRIKE_PHASE_PER_CLIP} "
                  f"(from ONNX metadata clip_strike_phases)")
        else:
            if policy.hitter_pure:
                raise SystemExit(
                    "[FATAL] 110-D ONNX lacks clip_strike_phases metadata. Re-export; the exact "
                    "strike frame is part of the trained contract and must not use a legacy guess."
                )
            print(f"[mj-sim2sim] strike_phase_per_clip in effect: {STRIKE_PHASE_PER_CLIP} "
                  f"(built-in legacy fallback — no clip_strike_phases in ONNX metadata; pass "
                  f"--strike-phase-per-clip to match the trained cfg if this is not a v1-clip model)")
    else:
        print(f"[mj-sim2sim] strike_phase_per_clip in effect: {STRIKE_PHASE_PER_CLIP} (CLI override; "
              f"must match the model's training YAML)")

    # --- episode/reset protocol resolution (P0 fix #2) --------------------------------------------
    if args.hold_steps_range is not None:
        hold_steps_range = tuple(int(v) for v in args.hold_steps_range)
        hold_range_source = "CLI"
    elif policy.motion_hold_steps_range_meta is not None:
        hold_steps_range = policy.motion_hold_steps_range_meta
        hold_range_source = "ONNX metadata"
    elif policy.hitter_pure:
        raise SystemExit(
            "[FATAL] metadata-less 110-D model needs --hold-steps-range. Base Pure trains [0,0] "
            "while Rally/V3 train [25,125]; guessing changes the exam distribution."
        )
    else:
        hold_steps_range = (0, 100)
        hold_range_source = "legacy evaluator fallback"
    if not (0 <= hold_steps_range[0] <= hold_steps_range[1]):
        raise SystemExit(f"[FATAL] invalid --hold-steps-range {hold_steps_range}")

    if args.hold_ref != "auto":
        hold_ref = args.hold_ref
        hold_ref_source = "CLI"
    elif policy.motion_hold_reference_meta in ("clip", "stand"):
        hold_ref = policy.motion_hold_reference_meta
        hold_ref_source = "ONNX metadata"
    elif policy.hitter_pure:
        raise SystemExit(
            "[FATAL] metadata-less 110-D model needs --hold-ref stand|clip. Current HitterPure "
            "holds use READY-STAND; silently using the legacy windup reference can create false "
            "tracking-guard terminations."
        )
    else:
        hold_ref = "clip"
        hold_ref_source = "legacy evaluator fallback"

    reset_mode = args.reset_mode
    bank_one_question_reset = False
    if args.target_source == "bank" and not args.exam_continuity_diagnostic:
        if args.reset_mode != "auto":
            raise SystemExit(
                "[FATAL] formal BankExam owns a one-question/one-reset protocol; do not pass "
                "--reset-mode. Use --exam-continuity-diagnostic for the separate carry-state ruler."
            )
        # Internally retain the training hold/balance semantics; run_rollout's explicit bank flag
        # resets robot/action/episode state after every completed schedule item.
        reset_mode = "multiswing"
        bank_one_question_reset = True
        print("[mj-sim2sim] reset mode: bank-one-question-reset (formal immutable-paper ruler; "
              "robot + last_action reset for every question)")
    elif args.target_source == "bank" and args.exam_continuity_diagnostic:
        if args.reset_mode not in ("auto", "multiswing"):
            raise SystemExit(
                "[FATAL] --exam-continuity-diagnostic requires multiswing carry-state semantics"
            )
        reset_mode = "multiswing"
        policy.evaluation_contract_exact = False
        print("[mj-sim2sim] reset mode: bank-continuity-diagnostic (same immutable paper, "
              "NO per-question reset; evaluation_contract_exact=false)")
    elif reset_mode == "auto":
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
    if args.target_source == "bank":
        protocol_missing = (
            policy.motion_hold_steps_range_meta is None
            or policy.motion_hold_reference_meta not in ("clip", "stand")
        )
        protocol_mismatch = (
            policy.motion_hold_steps_range_meta is not None
            and tuple(hold_steps_range) != tuple(policy.motion_hold_steps_range_meta)
        ) or (
            policy.motion_hold_reference_meta in ("clip", "stand")
            and hold_ref != policy.motion_hold_reference_meta
        )
        if protocol_missing or protocol_mismatch:
            if not args.allow_inexact_contract:
                raise SystemExit(
                    "[FATAL] BankExam reset/hold protocol is missing or disagrees with ONNX "
                    f"training metadata: hold="
                    f"{hold_steps_range}/{policy.motion_hold_steps_range_meta}, ref="
                    f"{hold_ref}/{policy.motion_hold_reference_meta}"
                )
            policy.evaluation_contract_exact = False
            print(
                "[mj-sim2sim] WARNING: reset/hold override changes or cannot prove the training "
                "protocol; evaluation_contract_exact=false"
            )
    if reset_mode == "multiswing":
        hold_desc = ("READY-STAND ref during holds (joint refs=default_q, vel 0; 2026-07-05+ "
                     "training hold semantics)" if hold_ref == "stand"
                     else "ref frozen at windup (legacy pre-07-05 hold semantics)")
        continuity_text = (
            "one reset per immutable question" if bank_one_question_reset
            else "no wrap teleports / carry state across questions"
        )
        print(f"[mj-sim2sim]   multiswing mechanics: {continuity_text}; pre-swing hold "
              f"U{hold_steps_range} "
              f"steps [{hold_range_source}] ({hold_desc} [{hold_ref_source}], tts pinned); "
              f"+ balance terminations "
              f"(tilt>{DF_FALL_TILT_RAD} rad, pelvis z<{DF_FALL_ROOT_Z_MIN} m)")
    elif hold_ref == "stand":
        print(f"[mj-sim2sim] NOTE: --hold-ref stand is INERT outside the multiswing protocol "
              f"(reset mode: {reset_mode})")
    if policy.hitter_pure:
        print("[mj-sim2sim] 110 evaluator limitation: training's true-reset mixture "
              f"stand_start_prob={policy.motion_stand_start_prob_meta}, "
              f"min_hold={policy.motion_stand_start_min_hold_meta}, "
              f"yaw_range={policy.motion_stand_start_yaw_range_meta} is recorded but not mixed into "
              "this training-like harness; use --deploy-faithful for a 100% stand-entry stress, "
              "and do not cite either as an exact reset-distribution match.")
    print(f"[mj-sim2sim] q_des clamp: "
          + ("ON — decoded q_des clamped to soft joint limits (0.9 x range; train "
             "ClampedJointPositionAction == deploy pp_joint_limits parity)" if qdes_clamp
             else "OFF (legacy exam, byte-identical to booked scores; --qdes-clamp recommended "
                  "for new exams — training clamps by default since 2026-07-06 and the C++ "
                  "runner always has)"))
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
    std_sha256 = None
    std_manifest_sha256 = None
    if any(s > 0 for s in args.noise_scales):
        if not os.path.isfile(args.std):
            raise SystemExit(f"[FATAL] dither mode requested but std sidecar not found: {args.std}\n"
                             f"        Create it from the checkpoint: np.save(.../learned_std.npy, "
                             f"torch.load(model.pt)['model_state_dict']['std'])")
        std_vec = np.load(args.std).astype(np.float64).reshape(-1)
        require_contract(std_vec.shape == (31,), f"std sidecar shape {std_vec.shape} != (31,)")
        require_contract(
            np.isfinite(std_vec).all() and np.all(std_vec > 0.0),
            "std sidecar must contain 31 finite positive values",
        )
        std_sha256 = sha256_file(args.std)
        std_manifest_path = args.std + ".meta.json"
        if os.path.isfile(std_manifest_path):
            try:
                with open(std_manifest_path, encoding="utf-8") as stream:
                    std_manifest = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"[FATAL] invalid learned-std manifest {std_manifest_path}: {exc}"
                ) from exc
            require_contract(
                isinstance(std_manifest, dict) and std_manifest.get("schema_version") == 1,
                "learned-std manifest must be a schema-v1 object",
            )
            require_contract(
                std_manifest.get("shape") == [31]
                and std_manifest.get("dtype") == "float32",
                f"learned-std manifest shape/dtype mismatch: {std_manifest}",
            )
            require_contract(
                std_manifest.get("std_file_sha256") == std_sha256,
                "learned-std file SHA does not match its manifest",
            )
            payload_sha = hashlib.sha256(
                np.asarray(std_vec, dtype="<f4").tobytes(order="C")
            ).hexdigest()
            require_contract(
                std_manifest.get("std_payload_sha256") == payload_sha,
                "learned-std payload SHA does not match its manifest",
            )
            std_checkpoint_sha = str(
                std_manifest.get("source_checkpoint_sha256", "")
            ).strip().lower()
            require_contract(
                len(std_checkpoint_sha) == 64
                and all(ch in "0123456789abcdef" for ch in std_checkpoint_sha),
                "learned-std manifest has invalid source checkpoint SHA",
            )
            if policy.source_checkpoint_sha256:
                require_contract(
                    std_checkpoint_sha == policy.source_checkpoint_sha256,
                    "learned-std sidecar belongs to a different checkpoint than the ONNX",
                )
            else:
                policy.evaluation_contract_exact = False
                print(
                    "[mj-sim2sim] WARNING: learned std is bound to a checkpoint but old ONNX is "
                    "not; evaluation_contract_exact=false"
                )
            std_manifest_sha256 = sha256_file(std_manifest_path)
        else:
            policy.evaluation_contract_exact = False
            print(
                "[mj-sim2sim] WARNING: learned_std.npy has no checkpoint-binding manifest; "
                "evaluation_contract_exact=false"
            )
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
    strict_motion_contract = policy.hitter_pure or args.target_source == "bank"
    if strict_motion_contract:
        contract_label = "HitterPure" if policy.hitter_pure else "BankExam"
        if policy.clip_seg_lengths is None:
            if args.target_source == "bank" and args.allow_inexact_contract:
                policy.evaluation_contract_exact = False
                print(
                    "[mj-sim2sim] WARNING: old bank-eval ONNX lacks clip_seg_lengths; "
                    "evaluation_contract_exact=false"
                )
            else:
                raise SystemExit(
                    f"[FATAL] {contract_label} ONNX lacks clip_seg_lengths metadata; re-export"
                )
        elif (len(policy.clip_seg_lengths) != num_clips
              or tuple(seg_len.tolist()) != tuple(policy.clip_seg_lengths)):
            raise SystemExit(
                f"[FATAL] {contract_label} motion clips do not match ONNX metadata: "
                f"npz={tuple(seg_len.tolist())}, onnx={tuple(policy.clip_seg_lengths)}. "
                "Wrong clips invalidate the policy's embedded reference buffers."
            )
        actual_clip_sha = tuple(sha256_file(path) for path in args.motion_files)
        if len(policy.motion_clip_sha256) != num_clips:
            if args.target_source == "bank" and args.allow_inexact_contract:
                policy.evaluation_contract_exact = False
                print(
                    "[mj-sim2sim] WARNING: old bank-eval ONNX lacks one motion SHA per clip; "
                    "evaluation_contract_exact=false"
                )
            else:
                raise SystemExit(
                    f"[FATAL] {contract_label} ONNX lacks one motion_clip_sha256 per clip; re-export"
                )
        elif actual_clip_sha != policy.motion_clip_sha256:
            raise SystemExit(
                f"[FATAL] {contract_label} motion clip SHA256 mismatch. Equal frame counts are "
                f"not proof of identity.\nONNX: {policy.motion_clip_sha256}\nFiles: {actual_clip_sha}"
            )
        if len(STRIKE_PHASE_PER_CLIP) != num_clips or any(
                not math.isfinite(float(p)) or not (0.0 <= float(p) <= 1.0)
                for p in STRIKE_PHASE_PER_CLIP):
            raise SystemExit(
                f"[FATAL] {contract_label} strike phases must have one finite [0,1] value per "
                f"clip; got {STRIKE_PHASE_PER_CLIP} for {num_clips} clips"
            )
        if policy.clip_strike_phases is None:
            if args.target_source == "bank" and args.allow_inexact_contract:
                policy.evaluation_contract_exact = False
                print(
                    "[mj-sim2sim] WARNING: old bank-eval ONNX lacks clip strike phases; "
                    "evaluation_contract_exact=false"
                )
            else:
                raise SystemExit(
                    f"[FATAL] {contract_label} ONNX lacks clip_strike_phases; re-export"
                )
        elif (len(policy.clip_strike_phases) != num_clips
              or any(not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=5e-5)
                     for a, b in zip(policy.clip_strike_phases, STRIKE_PHASE_PER_CLIP))):
            raise SystemExit(
                f"[FATAL] {contract_label} strike phases disagree: ONNX="
                f"{policy.clip_strike_phases}, evaluator={STRIKE_PHASE_PER_CLIP}"
            )
    elif policy.clip_seg_lengths and tuple(seg_len.tolist()) != tuple(policy.clip_seg_lengths):
        print(f"[mj-sim2sim] WARNING: motion npz seg_len {tuple(seg_len.tolist())} != ONNX "
              f"clip_seg_lengths {tuple(policy.clip_seg_lengths)} — these are probably NOT the "
              f"clips this model was trained/exported with.")

    robot = MujocoRobot(
        args.mjcf, policy.joint_names, policy.body_names, args.sim_dt,
        keep_native_damping=(passive_damping_mode == "mjcf"),
        keep_frictionloss=(frictionloss_mode == "mjcf"),
        pd_mode=pd_mode, kd_for_implicit=policy.kd,
        actuator_types=resolved_actuator_types,
        joint_armature=policy.joint_armature if bound_schema3_plant else None,
        joint_frictionloss_proxy=(
            policy.joint_friction_coefficients
            if bound_schema3_plant and frictionloss_mode == "contract-proxy" else None
        ),
        joint_velocity_limits=(
            policy.joint_velocity_limits if bound_schema3_plant else None
        ),
        joint_effort_limits=(policy.joint_effort_limits if bound_schema3_plant else None),
        require_bound_plant_match=formal_execution_contract_ok,
        allow_velocity_limit_proxy=not formal_execution_contract_ok,
    )
    if formal_qdes_limits is not None:
        robot.soft_jnt_lo, robot.soft_jnt_hi = (
            formal_qdes_limits[0].copy(), formal_qdes_limits[1].copy()
        )
        print("[mj-sim2sim] q_des bounds: schema-3 training metadata (31x2), not MJCF reconstruction")
    print(f"[mj-sim2sim] PD mode: {pd_mode} [{actuator_source}]"
          + ("  (implicit joints: kd as damping + implicitfast)"
             if "implicit" in resolved_actuator_types else ""))
    print(f"[mj-sim2sim] plant: native_damping={passive_damping_mode}, "
          f"frictionloss={frictionloss_mode} [{plant_source}]")
    if frictionloss_mode == "contract-proxy":
        print(
            "[mj-sim2sim] WARNING: direct-number friction proxy maps dimensionless PhysX "
            "load-dependent coefficients to MuJoCo constant N-m frictionloss; "
            "evaluation_contract_exact=false"
        )
        policy.evaluation_contract_exact = False

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

    # The formal cross-teacher ruler starts every question from one model-owned state, never from
    # the candidate teacher's clip.  Materializing through the real reset path makes a missing
    # ``stand`` keyframe fatal and proves the snapshot can be reproduced before any score is taken.
    if ready_state_mode == DEPLOY_NOMINAL_READY_STATE_MODE:
        require_contract(df_cfg is not None, "deploy nominal ready state requires deploy-faithful mode")
        robot.reset_to_stand(
            df_cfg["stand_root_pos"], df_cfg["stand_root_quat"], policy.default_q
        )
        ready_state_contract = robot.ready_state_snapshot(
            ready_state_mode, np.zeros(len(policy.joint_names), np.float64)
        )
    else:
        ready_state_contract = materialize_ready_state_contract(
            robot, refs_table, seg_start, ready_state_mode, action_dim=len(policy.joint_names)
        )
    mjcf_sha256 = sha256_file(args.mjcf)
    friction_coefficients = getattr(policy, "joint_friction_coefficients", None)
    friction_proxy = bool(
        bound_schema3_plant
        and friction_coefficients is not None
        and np.any(np.asarray(friction_coefficients, np.float64) != 0.0)
    )
    plant_semantics = {
        "training_joint_actuator_types": list(
            getattr(policy, "joint_actuator_types", None) or []
        ),
        "joint_friction_backend": str(getattr(policy, "joint_friction_backend", "")),
        "joint_friction_semantics": str(getattr(policy, "joint_friction_semantics", "")),
        "joint_friction_units": str(getattr(policy, "joint_friction_units", "")),
        "nonzero_physx_frictionloss_direct_number_proxy": friction_proxy,
        "formal_training_execution_metadata_validated": bool(formal_execution_contract_ok),
    }
    print(
        f"[mj-sim2sim] ready-state mode={ready_state_contract['mode']} "
        f"sha256={ready_state_contract['sha256']}"
    )

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
    # at its strike frame = local +Y of the reference wrist(=racket) frame at strike_step, times the
    # per-clip striking-face sign (--mount-normal-sign-per-clip; 表没开 = 标量 MOUNT_NORMAL_SIGN,逐位
    # 不变)。参考目标法向和实测法向同乘同一符号 = 训练侧 _ensure_reference_strike_state 同语义;boxes
    # 模式下两边同翻,拍面误差数值不变,变化只出现在 venue/bank(需求法向来自球,不翻)。
    target_normal_per_clip = []
    for c in range(num_clips):
        strike_step = int(seg_start[c]) + int(round(STRIKE_PHASE_PER_CLIP[c] * (seg_len[c] - 1)))
        ref_wrist_quat = refs_table[strike_step]["body_quat_w"][WRIST_TRACKED_IDX]
        target_normal_per_clip.append(
            mat_from_quat(ref_wrist_quat)[:, MOUNT_NORMAL_AXIS] * face_sign_for_clip(c))
    target_normal_per_clip = np.array(target_normal_per_clip)

    # 177-D hitter_footwork: per-clip reference base->racket reach offset (station coupling).
    # Metadata (utils/exporter.py) wins; metadata-less 177 exports (pre-2026-07-06) get a fallback
    # computed from the baked refs — same arithmetic as training _ensure_reference_strike_state:
    # reach_xy = reference blade world xy - reference pelvis world xy at the clip's strike frame.
    if policy.hitter and policy.ref_reach_offset_xy is None:
        reach_fallback = []
        for c in range(num_clips):
            strike_step = int(seg_start[c]) + int(round(STRIKE_PHASE_PER_CLIP[c] * (seg_len[c] - 1)))
            r = refs_table[strike_step]
            ref_root_pos = r["body_pos_w"][ROOT_TRACKED_IDX]
            ref_root_quat = r["body_quat_w"][ROOT_TRACKED_IDX]
            blade_w = ref_root_pos + mat_from_quat(ref_root_quat) @ racket_pos_pelvis(r["joint_pos"])
            reach_fallback.append(blade_w[:2] - ref_root_pos[:2])
        policy.ref_reach_offset_xy = reach_fallback
        print("[mj-sim2sim] 177 hitter: ONNX lacks ref_reach_offset_xy metadata — computed from "
              "refs: " + ", ".join(f"clip{c}=({v[0]:+.3f},{v[1]:+.3f})"
                                   for c, v in enumerate(reach_fallback)) +
              "  (re-export with the patched exporter to bake it)")
    elif policy.hitter:
        print("[mj-sim2sim] 177 hitter: ref_reach_offset_xy from ONNX metadata: "
              + ", ".join(f"clip{c}=({v[0]:+.3f},{v[1]:+.3f})"
                          for c, v in enumerate(policy.ref_reach_offset_xy)))

    # --- MODE B (venue-balls) sampler: lazy import so mode A never needs hope_planner ----------
    venue_sampler = None
    shared_schedule_artifact = None
    exam_bank_sha_before = None
    stage1_qb_path = None
    if args.target_source == "bank":
        exam_bank_sha_before = sha256_file(args.exam_bank)
        formal_bank_fields_ok = (
            formal_execution_contract_ok
            and policy.stage1_question_bank_exact == "1"
            and policy.stage1_bank_schema_version == "3"
            and policy.stage1_bank_split == "train"
            and bool(policy.stage1_source_family_sha256)
            and bool(policy.stage1_train_bank_sha256)
            and bool(policy.source_checkpoint_sha256)
        )
        if not formal_bank_fields_ok:
            if not args.allow_inexact_contract:
                raise SystemExit(
                    "[FATAL] ONNX lacks an immutable schema-3 execution/train-bank/checkpoint "
                    "binding. Required: exact training contract schema=3 with q_des limits/dt/"
                    "decimation, stage1_question_bank_exact=1, bank schema=3/split=train, train-bank "
                    "SHA, source-family SHA and source-checkpoint SHA. Re-export from a checkpoint "
                    "whose params/training_contract.json records the bank; historical replay needs "
                    "the explicit diagnostic escape hatch."
                )
            policy.evaluation_contract_exact = False
            print(
                "[mj-sim2sim] WARNING: ONNX has no immutable train-bank/checkpoint binding; "
                "evaluation_contract_exact=false"
            )
        stage1_qb_path = stage1_question_bank_module_path(wbt)
        # The loader module itself is NumPy/Torch-only, but importing it through the task package
        # pulls Isaac Lab. Bind the current checkout's standalone module explicitly so the CPU
        # evaluator never depends on an ambient HOPE_STAGE1_QB shell variable.
        os.environ["HOPE_STAGE1_QB"] = stage1_qb_path
        import venue_ball_sampler as _vbs   # sibling module (scripts/ is on sys.path, top of file)
        kw = {}
        if args.venue_table_near_x is not None:
            kw["table_near_x"] = args.venue_table_near_x
        if args.venue_table_surface_z is not None:
            kw["table_surface_z"] = args.venue_table_surface_z
        if args.venue_fh_y_split is not None:
            kw["fh_y_split"] = args.venue_fh_y_split
        venue_sampler = _vbs.BankExamSampler(
            repo_root=repo, ref_normal_per_clip=target_normal_per_clip, num_clips=num_clips,
            bank_path=args.exam_bank,
            allow_legacy_bank=args.allow_inexact_contract,
            runtime_motion_files=args.motion_files,
            runtime_segment_lengths=seg_len,
            runtime_strike_phases=STRIKE_PHASE_PER_CLIP,
            expected_source_family_sha256=policy.stage1_source_family_sha256 or None,
            schedule_seed=args.seed,
            schedule_k=args.exam_schedule_k,
            schedule_hold_range=hold_steps_range,
            landing_x_range=args.venue_landing_x_range,
            landing_y_range=tuple(args.venue_landing_y_range),
            speed_budget=args.venue_speed_budget, max_tries=args.venue_max_tries, **kw)
        exam_bank_sha_after = sha256_file(args.exam_bank)
        require_contract(
            exam_bank_sha_before == exam_bank_sha_after == venue_sampler.bank_sha256,
            "exam bank changed while the validated arrays/sampler were being loaded: "
            f"before={exam_bank_sha_before}, after={exam_bank_sha_after}, "
            f"sampler={venue_sampler.bank_sha256}",
        )
        if args.exam_schedule_json:
            import bank_exam_schedule as _bes

            _question_ids = _bes.derive_sampler_question_ids(
                venue_sampler, allow_inexact_contract=args.allow_inexact_contract
            )
            shared_schedule_artifact = _bes.load_schedule_artifact(
                args.exam_schedule_json,
                expected_bank_sha256=venue_sampler.bank_sha256,
                expected_clip_names=venue_sampler.clip_names,
                expected_question_ids=_question_ids,
            )
            _bes.install_schedule_on_sampler(
                venue_sampler,
                shared_schedule_artifact,
                allow_inexact_contract=args.allow_inexact_contract,
            )
            hold_steps_range = tuple(shared_schedule_artifact.hold_range)
            print(
                "[mj-sim2sim] shared schema-v3 exam schedule installed: "
                f"{os.path.abspath(args.exam_schedule_json)} "
                f"K={len(venue_sampler.schedule)} "
                f"quota={shared_schedule_artifact.per_clip_quota}/clip "
                f"sha256={shared_schedule_artifact.schedule_sha256}"
            )
        # A question either completes its hold+clip, or the episode timeout/fall finalizes it sooner.
        # Multiplying that proven per-attempt upper bound by K makes --steps a safety cap rather than
        # the experiment's denominator.  Explicit smaller caps are allowed only to fail loudly if K
        # cannot finish (useful for the truncation regression test).
        computed_bank_step_cap = formal_bank_step_cap(
            venue_sampler.schedule, seg_len, max_ep_len
        )
        if args.steps == 0:
            args.steps = computed_bank_step_cap
            print(
                f"[mj-sim2sim] BankExam --steps auto safety cap={args.steps} "
                f"(K={len(venue_sampler.schedule)} finite questions; timeout/clip upper bound)"
            )
        else:
            print(
                f"[mj-sim2sim] BankExam explicit --steps cap={args.steps}; K must complete or "
                "evaluation exits nonzero"
            )
        policy.evaluation_contract_exact = bool(
            policy.evaluation_contract_exact and venue_sampler.contract_exact
        )
        print(f"[mj-sim2sim] MODE B — target source: EXAM BANK (official S1 paper; same-source "
              f"questions, loader meta guards enforced)")
        print(f"[mj-sim2sim]   bank: {args.exam_bank}")
        if not policy.evaluation_contract_exact and not args.allow_inexact_contract:
            raise SystemExit(
                "[FATAL] bank evaluation contract is not exact (old artifact/bank provenance or "
                "episode semantics are missing). Regenerate/export a bound artifact; the explicit "
                "--allow-inexact-contract escape hatch is diagnostic only."
            )
        print(
            f"[mj-sim2sim] evaluation_contract_exact="
            f"{str(bool(policy.evaluation_contract_exact)).lower()}"
        )
        for _ln in venue_sampler.denominator_report():
            print(f"[mj-sim2sim] {_ln}")
    elif args.target_source == "venue-balls":
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
            fixed_normal=args.venue_fixed_normal,
            contact_fixed_env=args.venue_contact_fixed, spin_abs_max=args.venue_spin_max,
            vel_box=(None if args.venue_vel_box is None else
                     tuple((args.venue_vel_box[i], args.venue_vel_box[i + 1]) for i in (0, 2, 4))),
            **kw)
        if args.venue_contact_fixed or args.venue_spin_max is not None or args.venue_vel_box:
            print(f"[mj-sim2sim]   STAGE EXAM overrides: contact_fixed={args.venue_contact_fixed} "
                  f"spin_max={args.venue_spin_max} vel_box={args.venue_vel_box}")
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

    virtual_return_scorer = None
    virtual_return_scorer_contract = None
    if venue_sampler is not None:
        virtual_return_scorer, virtual_return_scorer_contract = build_virtual_return_scorer(
            repo, venue_sampler
        )

    execution_contract = build_evaluation_execution_contract(
        robot=robot,
        policy=policy,
        mjcf_sha256=mjcf_sha256,
        evaluator_sha256=sha256_file(os.path.abspath(__file__)),
        ready_state_contract=ready_state_contract,
        sim_dt=args.sim_dt,
        decimation=args.decimation,
        pd_mode=pd_mode,
        passive_damping_mode=passive_damping_mode,
        frictionloss_mode=frictionloss_mode,
        qdes_clamp=qdes_clamp,
        one_question_reset=bank_one_question_reset,
        plant_semantics=plant_semantics,
        protocol_semantics={
            "target_source": args.target_source,
            "reset_mode": reset_mode,
            "hold_ref": hold_ref,
            "hold_steps_range": [int(value) for value in hold_steps_range],
            "exam_hold_semantics": (
                shared_schedule_artifact.hold_semantics
                if shared_schedule_artifact is not None
                else "legacy-sampler-unspecified"
            ),
            "tracking_guards_ignored_during_hold": training_hold_protocol_active(
                reset_mode=reset_mode,
                deploy_faithful_cfg=df_cfg,
                venue_sampler=venue_sampler,
            ),
            "stage1_question_bank_loader_sha256": (
                sha256_file(stage1_qb_path) if stage1_qb_path is not None else None
            ),
            "episode_length_s": float(episode_length_s),
            "max_episode_control_steps": int(max_ep_len),
            "termination_anchor_pos_z_m": float(TERM_ANCHOR_POS_Z),
            "termination_anchor_orientation_projected_gravity_z": float(TERM_ANCHOR_ORI),
            "termination_end_effector_z_m": float(TERM_EE_POS_Z),
            "absolute_fall_tilt_rad": float(DF_FALL_TILT_RAD),
            "absolute_fall_root_z_min_m": float(DF_FALL_ROOT_Z_MIN),
            "strike_position_error_threshold_m": float(STRIKE_POS_THRESH),
            "strike_velocity_error_threshold_mps": float(STRIKE_VEL_THRESH),
            "strike_normal_error_threshold_deg": float(STRIKE_NORMAL_THRESH_DEG),
            "formal_bank_execution_metadata_validated": bool(formal_execution_contract_ok),
            "exam_schedule_schema_version": (
                int(shared_schedule_artifact.schema_version)
                if shared_schedule_artifact is not None else (1 if venue_sampler is not None else None)
            ),
            "exam_schedule_sha256": (
                str(venue_sampler.schedule_sha256) if venue_sampler is not None else None
            ),
        },
        virtual_return_scorer_contract=virtual_return_scorer_contract,
    )
    print(
        f"[mj-sim2sim] mjcf_sha256={mjcf_sha256} "
        f"execution_contract_sha256={execution_contract['sha256']}"
    )
    if virtual_return_scorer_contract is not None:
        print(
            "[mj-sim2sim] authoritative virtual-return scorer: "
            f"contract_sha256={virtual_return_scorer_contract['sha256']} "
            f"source_sha256={virtual_return_scorer_contract['source']['sha256']} "
            f"physics_config_sha256="
            f"{virtual_return_scorer_contract['physics_config']['sha256']}"
        )

    # DIAGNOSTIC: per-clip eval target-velocity boxes (clip 0 = forehand, clip 1 = backhand). None ->
    # faithful baseline (single training box for both clips). num_clips>2 reuse the backhand box.
    def _boxes12(vals):
        fh = ((vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]))
        bh = ((vals[6], vals[7]), (vals[8], vals[9]), (vals[10], vals[11]))
        return [fh if c == 0 else bh for c in range(num_clips)]

    hp_cfg = None
    if policy.hitter_pure:
        def hp_pick(cli_value, metadata_value, fallback, label):
            if cli_value is not None:
                return _boxes12(cli_value), "CLI"
            if metadata_value is not None:
                return metadata_value, "ONNX metadata"
            if args.allow_hitter_pure_defaults:
                return list(fallback), "explicit built-in fallback"
            raise SystemExit(
                f"[FATAL] 110-D ONNX lacks {label} metadata. Re-export, pass the matching CLI "
                "box, or use --allow-hitter-pure-defaults explicitly."
            )

        hp_pos, hp_pos_src = hp_pick(
            args.pos_range_per_clip,
            policy.hp_pos_range_per_clip,
            HP_POS_RANGE_PER_CLIP,
            "hitter_pure_pos_range_per_clip",
        )
        hp_vel, hp_vel_src = hp_pick(
            args.vel_range_per_clip,
            policy.hp_vel_range_per_clip,
            HP_VEL_RANGE_PER_CLIP,
            "hitter_pure_vel_range_per_clip",
        )
        if args.hp_base_target_range is not None:
            vals = args.hp_base_target_range
            hp_base, hp_base_src = ((vals[0], vals[1]), (vals[2], vals[3])), "CLI"
        elif policy.hp_base_target_range is not None:
            hp_base, hp_base_src = policy.hp_base_target_range, "ONNX metadata"
        elif args.allow_hitter_pure_defaults:
            hp_base, hp_base_src = HP_BASE_TARGET_RANGE, "explicit built-in fallback"
        else:
            raise SystemExit(
                "[FATAL] 110-D ONNX lacks hitter_pure_base_target_range metadata. Re-export, "
                "pass --hp-base-target-range, or use --allow-hitter-pure-defaults explicitly."
            )
        if len(hp_pos) != num_clips or len(hp_vel) != num_clips:
            raise SystemExit(
                f"[FATAL] HitterPure boxes must have one entry per clip: pos={len(hp_pos)} "
                f"vel={len(hp_vel)} clips={num_clips}"
            )
        all_ranges = [axis for box in hp_pos + hp_vel for axis in box] + list(hp_base)
        if any(not np.isfinite([lo, hi]).all() or lo > hi for lo, hi in all_ranges):
            raise SystemExit(f"[FATAL] invalid HitterPure range (finite lo<=hi required): {all_ranges}")
        hp_cfg = dict(
            pos_boxes=hp_pos,
            vel_boxes=hp_vel,
            base_range=hp_base,
            targets_center=bool(args.targets_center),
        )
        print(f"[mj-sim2sim] 110 hitter_pure station x/y={hp_base} [{hp_base_src}]")
        print(f"[mj-sim2sim]   station-relative racket pos [{hp_pos_src}]: {hp_pos}")
        print(f"[mj-sim2sim]   world racket velocity [{hp_vel_src}]: {hp_vel}")
        print("[mj-sim2sim]   target normal=velocity direction; actual face signs="
              f"{MOUNT_NORMAL_SIGN_PER_CLIP}; targets="
              f"{'box centers' if args.targets_center else 'uniform draws'}")

    vel_ranges_per_clip = None
    if venue_sampler is not None:
        # mode B: RacketCommand.resample() is never called — every target comes from the sampled
        # ball's StrikeSpec demand, so ALL box config (per-clip/legacy/CLI) is inert this run.
        print("[mj-sim2sim] per-clip target boxes: INERT (--target-source venue-balls; targets "
              "are ball-demanded via StrikeSpec)")
    elif policy.hitter_pure:
        print("[mj-sim2sim] legacy target samplers: INERT (110-D hitter_pure sampler active)")
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
    elif policy.hitter_pure:
        pass
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
        "mode", "step", "episode", "attempt_id", "schedule_index", "question_sequence_index",
        "bank_row", "question_id", "repeat", "hold_steps", "attempt_seed",
        "schedule_sha256", "clip_name", "swing_type", "time_to_strike",
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

    # Third CSV: one row for every sampled target, including attempts that die during hold or before
    # exact strike. This is the unconditional denominator artifact; the strike CSV remains the
    # conditional exact-frame diagnostic.
    attempt_csv_path = os.path.join(out_dir, "mujoco_sim2sim_attempts.csv")
    attempt_csv_f = open(attempt_csv_path, "w", newline="")
    acw = csv.writer(attempt_csv_f)
    acw.writerow([
        "mode", "attempt_id", "schedule_index", "question_sequence_index", "clip_name",
        "bank_row", "question_id", "repeat", "hold_steps", "attempt_seed",
        "schedule_sha256", "ready_state_mode", "ready_state_sha256", "mjcf_sha256",
        "execution_contract_sha256", "eligible", "censored", "physical_fall", "guard_reset",
        "hit", "returned", "reached_exact", "exact_composite",
        "finalize_reason", "termination_details",
    ])

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        print("[mj-sim2sim] MuJoCo passive viewer launched "
              f"(realtime={'off' if args.no_realtime else 'on'}). Close the window to stop.")

    results = []
    paired_bank_order = None
    for ns in args.noise_scales:
        # Separate streams are essential for a paired exam: ns=0 consumes no Gaussian draws while
        # ns>0 consumes 31/step. Sharing one RNG would therefore change later questions/holds even
        # with the same seed and a reset bank cursor.
        rng, action_noise_rng = paired_rollout_rngs(args.seed)
        print(f"\n[mj-sim2sim] >>> rollout noise_scale={ns}")
        res = run_rollout(policy, robot, refs_table, seg_start, seg_len, num_clips, step_dt,
                          args.decimation, ns, std_vec, args.steps, max_ep_len, rng, cw,
                          mode_label=f"ns={ns}", target_normal_per_clip=target_normal_per_clip,
                          strike_csv_writer=scw, attempt_csv_writer=acw,
                          viewer=viewer, realtime=not args.no_realtime,
                          vel_ranges_per_clip=vel_ranges_per_clip,
                          pos_ranges_per_clip=pos_ranges_per_clip, df=df_cfg,
                          reset_mode=reset_mode, hold_range=hold_steps_range,
                          venue_sampler=venue_sampler,
                          virtual_return_scorer=virtual_return_scorer,
                          switch_stress=args.switch_stress,
                          qdes_clamp=qdes_clamp, hold_ref=hold_ref, hp_cfg=hp_cfg,
                          action_noise_rng=action_noise_rng,
                          bank_one_question_reset=bank_one_question_reset,
                          ready_state_contract=ready_state_contract,
                          mjcf_sha256=mjcf_sha256,
                          execution_contract_sha256=execution_contract["sha256"])
        if args.target_source == "bank":
            current_order = tuple(res["exam_schedule"]["question_id_order"])
            if paired_bank_order is None:
                paired_bank_order = current_order
            require_contract(
                current_order == paired_bank_order
                and current_order == tuple(venue_sampler.schedule_question_ids),
                "paired BankExam model/noise columns did not answer identical question IDs/order",
            )
            require_contract(
                res["exam_schedule"]["sha256"] == venue_sampler.schedule_sha256,
                "paired BankExam rollout reported a different schedule SHA",
            )
        results.append(res)
    csv_f.close()
    strike_csv_f.close()
    attempt_csv_f.close()
    if viewer is not None:
        viewer.close()

    velocity_limit_diagnostics = {
        "hit_count": int(robot.velocity_limit_hit_count),
        "peak_abs_velocity_over_limit": float(robot.velocity_limit_peak_ratio),
        "proxy_clamp_applied": bool(
            robot.allow_velocity_limit_proxy and robot.velocity_limit_hit_count > 0
        ),
    }
    if velocity_limit_diagnostics["hit_count"] > 0:
        policy.evaluation_contract_exact = False
        for result in results:
            result["evaluation_contract_exact"] = False

    # ---- summary table ----
    print("\n" + "=" * 92)
    print(f"MuJoCo sim-to-sim | {os.path.basename(args.onnx)} | {args.steps} steps | seed {args.seed}"
          f" | qdes_clamp={'ON' if qdes_clamp else 'OFF'} | hold_ref={hold_ref}"
          f" | hold_range={hold_steps_range} | pd={pd_mode}"
          f" | damping={passive_damping_mode} | friction={frictionloss_mode}")
    print(
        "[mj-sim2sim] joint velocity limits: "
        f"hits={velocity_limit_diagnostics['hit_count']} "
        f"peak_ratio={velocity_limit_diagnostics['peak_abs_velocity_over_limit']:.6g} "
        f"proxy_clamp={str(velocity_limit_diagnostics['proxy_clamp_applied']).lower()}"
    )
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
    print(f"{'attempts (all targets)':28s}" + "".join(
        f"{r['attempts']['n_attempts']:16d}" for r in results
    ))
    print(f"{'reached exact / attempts':28s}" + "".join(
        f"{r['attempts']['exact_reach_rate']:16.4f}" for r in results
    ))
    print(f"{'composite / attempts':28s}" + "".join(
        f"{r['attempts']['composite_rate_per_attempt']:16.4f}" for r in results
    ))
    print(f"{'attempt finalize reasons':28s}" + "".join(
        f"{str(r['attempts']['finalize_reason_counts']):>16s}" for r in results
    ))
    print("-" * 92)
    row("fell(count)", "fell", "{:16d}")
    print(f"{'term_breakdown':28s}" + "".join(f"{str(r['term_breakdown']):>16s}" for r in results))
    if results and "hp" in results[0]:
        print("-" * 92)

        def hp_row(label, key, fmt="{:16.4f}"):
            values = []
            for result in results:
                value = result["hp"][key]
                if isinstance(value, (int, np.integer)):
                    values.append(f"{int(value):16d}")
                elif isinstance(value, float) and not math.isnan(value):
                    values.append(fmt.format(value))
                else:
                    values.append(f"{str(value):>16s}")
            print(f"{label:28s}" + "".join(values))

        hp_row("hp_base_x_exc_mean(m)", "base_x_excursion_mean")
        hp_row("hp_base_x_exc_p90(m)", "base_x_excursion_p90")
        hp_row("hp_base_x_exc_max(m)", "base_x_excursion_max")
        hp_row("hp_base_x_completed_end(m)", "base_x_at_completed_end_mean")
        hp_row("hp_exact_station_dx(m)", "exact_station_dx_abs_mean")
        hp_row("hp_exact_station_dy(m)", "exact_station_dy_abs_mean")
        hp_row("hp_exact_station_dxy(m)", "exact_station_dxy_mean")
        hp_row("hp_exact_yaw_abs(deg)", "exact_yaw_abs_deg_mean")
        hp_row("hp_swings_measured", "n_swings_measured")
        hp_row("hp_attempts(all targets)", "n_attempts")
        hp_row("hp_swings_completed", "n_completed")
        hp_row("hp_exact_alive", "n_exact_alive")
        print(f"{'hp_finalize_reasons':28s}" + "".join(
            f"{str(r['hp']['finalize_reason_counts']):>16s}" for r in results
        ))
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
              "sampled ball through the authoritative Isaac-parity virtual-return scorer)")
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

        vrow("attempts (all targets)", "n_attempts")
        vrow("exact reach / attempt", "exact_reach_rate_per_attempt")
        vrow("CONTACT / ATTEMPT", "contact_rate_per_attempt")
        vrow("RETURN SUCCESS / ATTEMPT", "return_success_rate_per_attempt")
        vrow("n_strikes (exact)", "n_strikes")
        vrow("ball_contacted (n)", "contacted")
        vrow("contact_rate | exact", "contact_rate")
        vrow("landing_valid|contact", "landing_valid_rate")
        vrow("in_bounds|contact", "in_bounds_rate")
        vrow("net_clear|contact", "net_clear_rate")
        vrow("landed_ok (n)", "landed_ok")
        vrow("return_success | exact", "return_success_rate")
        vrow("land_err_median(m)", "land_err_median")
        vrow("land_err_mean(m)", "land_err_mean")
        vrow("demanded_|v_r|_mean(m/s)", "demanded_speed_mean")
        for sub, nm in (("forehand", "fh"), ("backhand", "bh")):
            vrow(f"  {nm}: attempts", "n_attempts", sub=sub)
            vrow(f"  {nm}: n_strikes", "n_strikes", sub=sub)
            vrow(f"  {nm}: contact/attempt", "contact_rate_per_attempt", sub=sub)
            vrow(f"  {nm}: return/attempt", "return_success_rate_per_attempt", sub=sub)
            vrow(f"  {nm}: return|exact", "return_success_rate", sub=sub)
        print("-" * 92)
        print("COUNTERFACTUAL — DEMANDED normal swapped into the achieved strike (same achieved "
              "pos/vel/pos_err):\n  CF >> actual return rate = the face-orientation channel "
              "ALONE fails the return (no normal channel in the obs contract)")
        vrow("CF contact_rate", "contact_rate", sub="cf_all")
        vrow("CF RETURN / ATTEMPT", "return_success_rate_per_attempt", sub="cf_all")
        vrow("CF return | exact", "return_success_rate", sub="cf_all")
        vrow("CF land_err_median(m)", "land_err_median", sub="cf_all")
        for sub, nm in (("cf_forehand", "fh"), ("cf_backhand", "bh")):
            vrow(f"  {nm}: CF return/attempt", "return_success_rate_per_attempt", sub=sub)
            vrow(f"  {nm}: CF return|exact", "return_success_rate", sub=sub)
        print("-" * 92)
        vrow("spec_solve_fails", "solve_fail", sub="sampler")
        vrow("sign_rejects", "sign_reject", sub="sampler")
        vrow("mean_solve_iters", "mean_solve_iters", sub="sampler")
        if hasattr(venue_sampler, "denominator_report"):
            print("-" * 92)
            print("DENOMINATORS (判卷分母法则 — return rates are meaningless without these):")
            for _ln in venue_sampler.denominator_report():
                print(_ln)
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

    summary_path = os.path.join(out_dir, "mujoco_sim2sim_summary.json")
    input_artifacts = {
        "onnx": {"path": os.path.abspath(args.onnx), "sha256": sha256_file(args.onnx)},
        "mjcf": {"path": os.path.abspath(args.mjcf), "sha256": sha256_file(args.mjcf)},
        "motions": [
            {"path": os.path.abspath(path), "sha256": sha256_file(path)}
            for path in args.motion_files
        ],
        "evaluator_source": {
            "path": os.path.abspath(__file__),
            "sha256": sha256_file(os.path.abspath(__file__)),
        },
    }
    if args.exam_bank:
        input_artifacts["exam_bank"] = {
            "path": os.path.abspath(args.exam_bank),
            "sha256": sha256_file(args.exam_bank),
        }
    if std_sha256 is not None:
        input_artifacts["learned_std"] = {
            "path": os.path.abspath(args.std),
            "sha256": std_sha256,
            "manifest_sha256": std_manifest_sha256,
        }
    if policy.obs_norm_path:
        input_artifacts["obs_norm"] = {
            "path": os.path.abspath(policy.obs_norm_path),
            "sha256": sha256_file(policy.obs_norm_path),
        }
    if venue_sampler is not None:
        input_artifacts["venue_sampler_source"] = {
            "path": os.path.abspath(__import__("venue_ball_sampler").__file__),
            "sha256": sha256_file(os.path.abspath(__import__("venue_ball_sampler").__file__)),
        }
        input_artifacts["virtual_return_scorer_source"] = {
            "path": os.path.abspath(_virtual_return_scorer.__file__),
            "sha256": virtual_return_scorer_contract["source"]["sha256"],
        }
        input_artifacts["virtual_return_scorer_physics_config"] = {
            "path": os.path.abspath(virtual_return_scorer.params.source_path),
            "sha256": virtual_return_scorer_contract["physics_config"]["sha256"],
        }
        if args.exam_schedule_json:
            input_artifacts["exam_schedule_artifact"] = {
                "path": os.path.abspath(args.exam_schedule_json),
                "sha256": sha256_file(args.exam_schedule_json),
                "schedule_sha256": shared_schedule_artifact.schedule_sha256,
                "schema_version": shared_schedule_artifact.schema_version,
            }
        if getattr(venue_sampler, "bank_meta", None):
            input_artifacts["bank_physics_contract_sha256"] = venue_sampler.bank_meta.get(
                "physics_contract_sha256"
            )
    summary = {
        "schema_version": 3,
        "onnx": os.path.abspath(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "evaluation_contract_exact": bool(policy.evaluation_contract_exact),
        "ready_state": ready_state_contract,
        "ready_state_mode": ready_state_contract["mode"],
        "ready_state_sha256": ready_state_contract["sha256"],
        "mjcf_sha256": mjcf_sha256,
        "execution_contract": execution_contract,
        "execution_contract_sha256": execution_contract["sha256"],
        "joint_velocity_limit_diagnostics": velocity_limit_diagnostics,
        "control_step_dt_s": step_dt,
        "arguments": vars(args),
        "input_artifacts": input_artifacts,
        "results": results,
        "artifacts": {
            "per_step_csv": {
                "path": os.path.abspath(csv_path), "sha256": sha256_file(csv_path)
            },
            "per_strike_csv": {
                "path": os.path.abspath(strike_csv_path), "sha256": sha256_file(strike_csv_path)
            },
            "per_attempt_csv": {
                "path": os.path.abspath(attempt_csv_path), "sha256": sha256_file(attempt_csv_path)
            },
        },
    }
    if virtual_return_scorer_contract is not None:
        summary["virtual_return_scorer_contract"] = virtual_return_scorer_contract
        summary["virtual_return_scorer_contract_sha256"] = virtual_return_scorer_contract["sha256"]
    if args.target_source == "bank":
        schedule_runtime = {
            "sha256": venue_sampler.schedule_sha256,
            "bank_sha256": venue_sampler.bank_sha256,
            "seed": venue_sampler.schedule_seed,
            "size": len(venue_sampler.schedule),
            "one_question_reset": bool(bank_one_question_reset),
            "ready_state_mode": ready_state_contract["mode"],
            "ready_state_sha256": ready_state_contract["sha256"],
            "mjcf_sha256": mjcf_sha256,
            "execution_contract_sha256": execution_contract["sha256"],
            "common_random_numbers": {
                "action_noise": "standard Gaussian stream re-seeded by question attempt_seed; "
                                "noise_scale only multiplies the shared draws",
                "repeat": 0,
            },
            "items": [
                {
                    "schedule_index": item.schedule_index,
                    "question_sequence_index": item.schedule_index,
                    "clip": item.clip,
                    "bank_row": item.bank_row,
                    "question_id": item.question_id,
                    "repeat": item.repeat,
                    "hold_steps": item.hold_steps,
                    "attempt_seed": item.attempt_seed,
                }
                for item in venue_sampler.schedule
            ],
        }
        if shared_schedule_artifact is not None:
            import bank_exam_schedule as _bes

            schedule_runtime["schema_version"] = shared_schedule_artifact.schema_version
            schedule_runtime["shared_artifact"] = _bes.artifact_document(
                shared_schedule_artifact
            )
            schedule_runtime["artifact_path"] = os.path.abspath(args.exam_schedule_json)
            schedule_runtime["artifact_file_sha256"] = sha256_file(args.exam_schedule_json)
        else:
            schedule_runtime["schema_version"] = 1
            schedule_runtime["shared_artifact"] = None
        summary["exam_schedule"] = schedule_runtime
    summary_tmp = summary_path + ".tmp"
    with open(summary_tmp, "w", encoding="utf-8") as stream:
        json.dump(json_ready(summary), stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
    os.replace(summary_tmp, summary_path)

    print(f"[mj-sim2sim] per-step CSV   -> {csv_path}")
    print(f"[mj-sim2sim] per-strike CSV -> {strike_csv_path}\n")
    print(f"[mj-sim2sim] per-attempt CSV -> {attempt_csv_path}\n")
    print(f"[mj-sim2sim] summary JSON    -> {summary_path}\n")


def cli_entrypoint():
    """Run the evaluator with an explicit process-status contract."""
    try:
        result = main()
        return 0 if result is None else int(result)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
