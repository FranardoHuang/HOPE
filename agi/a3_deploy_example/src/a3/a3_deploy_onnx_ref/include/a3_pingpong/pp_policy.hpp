// model_15200 front-end as a drop-in A3PolicyDriver CommandFn. Per tick:
//   scripted racket target -> reference clock -> ONNX refs -> 180-D obs ->
//   action -> target_q (Isaac) -> scatter to 31 SDK slots -> RobotCommand.
// NECK PASSIVE: head slots [3,4] are held at nominal (q=0) with AGI's fixed PD
// (kp=40, kd=2); the model's neck outputs are ignored for hardware command.
//
// ===================== WHAT IS SCRIPTED vs LEARNED =====================
// The swing JOINT TRAJECTORY is NOT hard-coded. Every tick the learned ONNX
// policy emits a fresh 31-DOF action; q_des = default_q + action*action_scale.
// The forehand/backhand BODY POSTURE is learned (encoded in the policy weights +
// the baked reference clip the ONNX carries as obs-independent side-outputs).
//   LEARNED (ONNX, per tick):  31 joint actions -> q_des; kp/kd from metadata.
//   REFERENCE (ONNX side-out): command[0:62] ref joint_pos/vel + tracked body
//                              poses, indexed by time_step (the strike clock).
//   SCRIPTED (this file, C++):  the racket TARGET (pos/vel/normal-sign), the
//                              strike clock (time_to_strike -> time_step), and
//                              the forehand/backhand SELECT (swing_dir_). There
//                              is NO live ball tracker / planner -- ScriptedTarget
//                              is a fixed front-right TEST target. Pressing f/b
//                              only flips the target y-sign + swing_type and picks
//                              the matching baked clip; it does not load new poses.
//   OVERWRITTEN AFTER ONNX:    neck slots [3,4] forced passive; legs forced to
//                              nominal iff --legs-passive; q_des clamped to A3
//                              joint limits (safety). Nothing else is overridden.
// To hit REAL balls, replace ScriptedTarget with planner output (pos/vel/normal/
// hit-time from a ball-trajectory estimator); the policy/obs/decode stay as-is.
// ======================================================================
//
// Depends only on Eigen + onnxruntime + robot_io_backend.hpp (plain structs),
// so it is unit-testable off-robot. The CommandFn signature matches
// a3_deploy::CommandFn exactly (assignable without including the AimRT driver).
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "a3_pingpong/pp_base_estimator.hpp"
#include "a3_pingpong/pp_joint_limits.hpp"
#include "a3_pingpong/pp_joint_map.hpp"
#include "a3_pingpong/pp_obs_builder.hpp"
#include "a3_pingpong/pp_onnx_policy.hpp"
#include "a3_pingpong/pp_oracle_pose.hpp"
#include "a3_pingpong/pp_planner_input.hpp"
#include "a3_pingpong/pp_reference_clock.hpp"
#include "a3_policy_parameters.hpp"      // ::a3_pd_stand_kps / kds (official robust-stand gains)
#include "robot_io/a3_layout_extra.hpp"  // robot_io::kA3PolicyToSdkIdx (29->31 scatter)
#include "robot_io/robot_io_backend.hpp"

namespace a3_pingpong {

// AGI neck-passive constants (from expand_to_backend.hpp).
constexpr int kHeadSlot0 = 3;
constexpr int kHeadSlot1 = 4;
constexpr double kHeadPosRad = 0.0;
constexpr double kHeadKp = 40.0;
constexpr double kHeadKd = 2.0;
// Backend MuJoCo slot layout: legs are slots [19..30] (12 DOF).
constexpr int kLegSlotStart = 19;
constexpr int kLegSlotCount = 12;
// Waist is slots [0..2] (waist_yaw, waist_roll, waist_pitch).
constexpr int kWaistSlotStart = 0;
constexpr int kWaistSlotCount = 3;

// torso_Link's frame origin relative to the pelvis: offset (-0.02,0,0.005) carried
// through waist_yaw (Isaac q[2], +Z) and waist_roll (Isaac q[5], +X). The torso
// anchor sits ~at the waist (≈ base + 5 mm up), NOT 0.25 m up — verified against
// the model's reference body_pos_w[7] ≈ pelvis + 0.005. Using a fixed (0,0,1.20)
// torso put a ~0.25 m bias into the motion_anchor_pos observation.
inline Vec3 torso_pos_from_base(const Vec3& base_pos, const Vec4& base_quat,
                                const Eigen::VectorXd& q_isaac) {
  const double wy = q_isaac.size() > 2 ? q_isaac[2] : 0.0;  // waist_yaw
  const double wr = q_isaac.size() > 5 ? q_isaac[5] : 0.0;  // waist_roll
  const Mat3 Rwaist = (Eigen::AngleAxisd(wy, Vec3::UnitZ()) *
                       Eigen::AngleAxisd(wr, Vec3::UnitX())).toRotationMatrix();
  const Vec3 off = Rwaist * Vec3(-0.02, 0.0, 0.005);
  return base_pos + mat_from_quat(base_quat) * off;
}

// How the localization-dependent obs terms (motion_anchor_pos_b,
// racket_target_pos_b, base_target_pos_b) get their robot world pose. The 180-D
// obs LAYOUT is identical in all three modes; only the *values* of these terms
// change. See SIM_DEPLOY_REHEARSAL.md.
enum class LocMode {
  kFabricated,       // A (legacy): nominal frozen base pose + waist-FK torso.
                     //   -> motion_anchor_pos_b is a FICTIONAL tracking error.
  kPerfectTracking,  // B (hardware-safe): assume position tracking is perfect.
                     //   torso_pos_w := ref anchor (motion_anchor_pos_b == 0);
                     //   base_pos_w := ref pelvis (racket/base target relative to
                     //   where we SHOULD be). Real IMU still drives orientation.
  kOracle,           // C (SIMULATION ONLY): true MuJoCo pelvis pose from the shm
                     //   bridge. NEVER available on hardware (shm file absent).
  kExternalBase,     // HARDWARE planner mode: real base POSITION from a live mocap
                     //   localizer (PpBasePoseInput / /a3/base_pose_flat), in the SAME
                     //   world frame as the planner's racket target. Orientation stays
                     //   the yaw-aligned IMU (mocap is position-only). Stale stream ->
                     //   SAFE fallback to perfect_tracking + loud warn (like oracle).
};

struct PpPolicyConfig {
  int level = 1;                 // 0 = hold wind-up (quasi-stand), 1 = periodic forehand
  bool legs_passive = false;     // hold leg joints at nominal (firm PD) — for a HOISTED demo
                                 // where balance isn't needed; stops leg twitch from the
                                 // nominal-base-position obs gap. Arm+waist still swing.
  bool waist_passive = false;    // ALSO hold the waist (slots 0..2) at nominal. The policy
                                 // commands waist_pitch to its forward limit (+0.419) which,
                                 // with the forehand arms reaching forward, pushes the CoM past
                                 // the feet → a STATIC leg hold can't rebalance → tips forward.
                                 // Freezing the waist keeps the torso CoM over the feet for an
                                 // ARMS-ONLY ground swing (with --official-stand → official gains).
  bool auto_leg_hold = false;    // dynamic hold: at level 0 (ready/windup) HOLD legs+waist (stable
                                 // stand, avoids the frozen-windup OOD foot-lift); at level 1 (swing)
                                 // RELEASE them (full-body self-balancing swing). The driver flips
                                 // set_legs_passive/set_waist_passive each tick from the level.
  double leg_smooth_alpha = 1.0; // EMA low-pass on the POLICY-DRIVEN leg q_des: out = a*in + (1-a)*prev.
                                 // 1.0 = off (no smoothing). <1 removes the tick-to-tick jitter that
                                 // stiff weight-bearing gains (--leg-stand-gains) amplify into a TWITCH;
                                 // ~0.2-0.3 = moderate (tau ~3-4 ticks @50Hz). Seeded from nominal so the
                                 // release does not jump; no-op when legs are HELD. See --leg-smooth-alpha.
  double leg_clamp_rad = 0.0;    // 0 = off. >0 clamps each POLICY-DRIVEN leg slot (level-1
                                 // released swing) to nominal ± this band, capping the deep
                                 // crouch-and-lean the trained swing commands (hip_pitch -0.6..
                                 // -0.77, ankle_pitch -0.7..-0.9 rad) that the real robot cannot
                                 // hold standing -> knees sink. Keeps legs near the proven upright
                                 // stand while leaving room for small balance moves. No-op when
                                 // legs are HELD (already nominal). See --leg-clamp-rad.
  bool use_base_estimator = false;  // leg-FK + IMU pelvis-height estimate (planted feet).
                                    // ON for the ground test; OFF on the hoist (feet hang ->
                                    // planted assumption invalid -> use nominal height).
                                    // Only affects kFabricated mode.
  // HARDWARE-SAFE DEFAULT: perfect_tracking. kFabricated synthesizes a fictional
  // world-tracking error (the documented "deploy buzz" mode) and must NOT be the
  // default on hardware. The A/B/C rehearsal selects fabricated explicitly via
  // --loc-mode. See LocMode + SIM_DEPLOY_REHEARSAL.md.
  LocMode loc_mode = LocMode::kPerfectTracking;  // A/B/C localization mode (see LocMode).
  // DEFAULT FLIPPED TO TRUE (2026-07-03): with yaw_align (below, default on) the base yaw is
  // expressed relative to the ENGAGE heading — it starts at identity and then tracks the robot's
  // REAL turning, which is exactly what training saw (targets rotated by the current base yaw,
  // hope_commands.racket_target_pos_b_rel). The old false default (identity yaw) predates
  // yaw_align and silently mixed frames: the racket-FK world conversion uses the full
  // yaw-aligned quat while the target rotation ignored yaw — fine only while the robot never
  // turns. model_9000 TURNS ~84 deg by design (v4 clips are baked facing world +Y), so identity
  // yaw would rotate the target obs ~84 deg OOD mid-motion. Revert with --no-imu-yaw (only
  // sensible for a non-turning model, e.g. p4).
  bool use_imu_yaw_for_targets = true;  // see build_obs_180(use_base_yaw_for_targets)
  double oracle_max_age_s = 0.1;    // reject oracle samples older than this (stale bridge/sim).
  double dt = 0.02;              // 50 Hz
  double strike_period = 3.0;    // seconds between strikes (level 1)
  double strike_lead_frac = 0.7; // strike occurs at this fraction of each cycle
  // SINGLE-SWING / REST mode: the periodic level-1 clock WRAPS every strike_period —
  // the reference SNAPS from the clip's end pose back to the windup frame mid-stance.
  // Training never tracks that transition (clip wraps TELEPORT the robot in Isaac), and
  // the backhand end->windup pose gap is large enough that the snap topples the free
  // base (observed: p4 backhand survives swing 1, collapses right after the first wrap;
  // forehand's smaller gap survives). single_swing: after the clip has fully played
  // (tts below the clip's end), auto-drop to level 0 (held stand / windup hold) instead
  // of wrapping — press 1 to swing again from a clean windup start (which the policy
  // provably handles). swing_rest_s >= 0: additionally auto re-arm level 1 after that
  // many seconds of rest (continuous demo without ever snapping).
  bool single_swing = false;
  double swing_rest_s = -1.0;    // <0 = no auto re-arm (manual '1' per swing)
  // ===================== LIVE PLANNER MODE (Path B, official) =====================
  // When planner_mode: the racket target is NO LONGER the scripted per-clip box center.
  // A real planner feeds PpRacketTargetInput (over AimRT /racket/command_flat) and a mocap
  // localizer feeds PpBasePoseInput (LocMode::kExternalBase). Each ComputeCommand tick,
  // PlannerEngageStep_ reproduces the proven Python wbc_runner._tick engage machine:
  // gate a fresh VALID command (timeout / invalid-flutter grace / min-tts / base-low /
  // reachability), then set_swing_dir + set_level(1) + FREEZE the target. The existing
  // swing clock, tts clamps, single-swing completion and mid-swing latch execute the swing
  // UNCHANGED. planner_mode implies single_swing (one clip per engage, then a held stand).
  bool planner_mode = false;
  double engage_min_tts_s = 1.0;      // never START a swing later than this (deep-clip snap -> fall)
  double planner_invalid_grace_s = 0.25;  // a valid cmd still engages if an invalid arrived within this
  double command_timeout_s = 0.5;     // no fresh VALID command within this -> stand
  double base_low_z = 0.7;            // base below this (fallen/crouched) -> refuse to engage
  double hold_anchor_x_b = 0.40;      // base-rel x of the ready-hold target between swings (racket-reach)
  // Post-swing hold budget: run the POLICY hold this long after a completed swing (it must
  // actively balance out of the follow-through — a static stand cannot), then blend to the
  // STATIC official stand until the next engage. The model's level-0 policy hold only has
  // ~5 s of margin (Gate 2.5: scripted m0 hold falls at ~5 s; closed-loop: post-swing hold
  // degrades at ~5-10 s) — never park on it.
  double hold_recover_s = 2.5;
  double hold_blend_s = 0.8;          // q_des ramp measured-pose -> nominal at the switch
  double external_base_max_age_s = 0.2;   // reject base-pose samples older than this (stale mocap)
  // Reachability gate (base-relative x,y + world z + speed); mirrors wbc_runner target_gate.
  bool target_gate_enable = true;
  double gate_x_lo = 0.20, gate_x_hi = 0.90;
  double gate_y_abs = 0.85;
  double gate_z_lo = 0.55, gate_z_hi = 1.40;
  double gate_speed_max = 3.5;
  // YAW-ALIGN (hardware fix, 2026-07-02): the pelvis AND torso IMU yaws are NOT
  // world-referenced on the real robot (boot-to-boot drift; MDU captures show a constant
  // fictional -12/-15/-38.5 deg yaw error in motion_anchor_ori_b while training reset noise
  // is only +-11 deg). Two obs terms consume the raw quats: motion_anchor_ori_b (torso vs
  // clip-frame reference anchor) and the racket-FK world conversion (R(base_quat)*fk vs the
  // identity-yaw target frame) — the old use_imu_yaw_for_targets=false fix only bypassed the
  // TARGET rotation. With yaw_align, each IMU's yaw is captured at the moment the policy
  // engages (SHADOW/MOTION entry; robot standing, facing its operational forward = the clip
  // world +x) and its inverse is left-multiplied onto every subsequent sample, so attitudes
  // are expressed relative to the entry heading. No-op in sim (yaw ~ 0 at spawn).
  bool yaw_align = true;
  double swing_speed = 1.0;      // <1.0 stretches the swing in real time so the
                                 // hardware actuators can actually track it
                                 // (native speed under-shoots + strains loudly).
                                 // The clip frame AND obs time_to_strike slow
                                 // together, so the (frame,tts) pair stays on the
                                 // training manifold — just evolves slower.
  // Scripted swing direction at startup: false=forehand (clip 0), true=backhand
  // (clip 1). Toggle live with the f/b keys. No live planner — this is the
  // scripted TEST path.
  bool start_backhand = false;
  // Scripted racket TEST targets, PER CLIP, chosen inside the TRAINED sampling boxes.
  // RE-SYNCED 2026-07-03 to the model_9000 generation (run 2026-07-03_02-01-17,
  // cfg/task/HOPEPingPongDeployParity.yaml 2026-07-02 blade re-plane; same constants as
  // mujoco_eval_onnx.py POS/VEL_RANGE_PER_CLIP):
  //   clip0 forehand: pos x[0.58,0.78] y[-0.64,-0.24] z[0.72,0.92]  vel x[1.05,2.05] y[ 0.96, 1.96] z[0.31,1.11]
  //   clip1 backhand: pos x[0.56,0.76] y[-0.07, 0.33] z[0.93,1.13]  vel x[1.61,2.61] y[-1.21,-0.21] z[0.00,0.71]
  // Targets sit at the BOX CENTERS = each clip's reference BLADE strike state (fh (0.68,-0.44,0.82)
  // vel (1.55,1.46,0.71); bh (0.66,0.13,1.03) vel (2.11,-0.71,0.36)). The previous values
  // (fh/bh pos x=0.45, bh vel x=1.50) were the OLD explicitpd_ft-era boxes — 0.11-0.13 m BELOW the
  // new x ranges = an OOD command obs on every tick of the swing.
  // ⚠ model_9000 is a WALK-AND-STRIKE policy: with these world-fixed targets it turns ~84 deg and
  // displaces its base 0.4-0.65 m before contact (measured in the deploy-faithful MuJoCo gate).
  // That footwork only closes the loop when the localization source reports the REAL base motion
  // (sim: --oracle-pelvis; hardware: mocap). Under perfect_tracking the base obs stays pinned to
  // the reference pelvis and the strike loop runs OPEN — validate in sim with BOTH loc modes.
  Vec3 racket_pos_w_clip[2] = {Vec3(0.68, -0.44, 0.82), Vec3(0.66, 0.13, 1.03)};
  Vec3 racket_vel_w_clip[2] = {Vec3(1.55, 1.46, 0.71), Vec3(2.11, -0.71, 0.36)};
  // sim2real localisation gap: no global base/torso pose -> nominal (matches
  // the Python wbc_runner shadow behavior). base orientation uses the real IMU.
  Vec3 nominal_base_pos_w = Vec3(0.0, 0.0, 0.95);
  Vec3 nominal_torso_pos_w = Vec3(0.0, 0.0, 1.20);
  Vec4 nominal_torso_quat_w = Vec4(1.0, 0.0, 0.0, 0.0);
};

class PpPolicy {
 public:
  PpPolicy(const std::string& onnx_path, PpPolicyConfig cfg = {})
      : onnx_(onnx_path), cfg_(cfg), level_(cfg.level),
        swing_speed_(cfg.swing_speed), swing_dir_(cfg.start_backhand ? -1 : 1),
        legs_passive_(cfg.legs_passive), waist_passive_(cfg.waist_passive),
        leg_clamp_rad_(cfg.leg_clamp_rad), leg_smooth_alpha_(cfg.leg_smooth_alpha),
        last_action_(Eigen::VectorXd::Zero(kNumJoints)) {
    if (!build_src_to_sdk(onnx_.joint_names(), isaac_to_sdk_))
      throw std::runtime_error("pingpong: ONNX joint_names do not map onto the backend layout");
    // Planner-mode pre-engage hold target: seed pos/vel from the forehand box center (the
    // same in-training values the SCRIPTED hold uses) so the level-0 hold obs before the
    // first serve is on-manifold; each engage overwrites them with the frozen command.
    // The y seed matters: a centered (y=0) hold target sits ~0.3 m LEFT of the forehand
    // ready racket -> the policy leans/reaches toward it, sinks, and tips (observed in the
    // headless closed-loop). Box-center y keeps the hold at the trained ready stance.
    planner_frozen_vel_w_ = cfg_.racket_vel_w_clip[0];
    planner_hold_z_w_ = cfg_.racket_pos_w_clip[0][2];
    planner_hold_pos_b_engage_ =
        Vec3(cfg_.hold_anchor_x_b, cfg_.racket_pos_w_clip[0][1], 0.0);
    // Reference-clock layout: prefer the ONNX-baked per-clip metadata (new exports carry
    // clip_seg_lengths/clip_strike_phases). The ClipLayout default is the LEGACY v1 layout
    // ({95,105}/{0.36,0.50}, model_15200-era); driving a v2-baked model with it serves the
    // wrong reference frames every tick (forehand strike ~0.6 s early, follow-through clamped,
    // "backhand" spliced across the clip boundary) — the 2026-07-02 stale-clock deploy bug.
    if (onnx_.has_clip_layout()) {
      if (onnx_.clip_seg_lengths().size() != 2)
        throw std::runtime_error("pingpong: ONNX clip layout metadata does not have 2 clips");
      clip_.seg_len[0] = static_cast<int>(std::lround(onnx_.clip_seg_lengths()[0]));
      clip_.seg_len[1] = static_cast<int>(std::lround(onnx_.clip_seg_lengths()[1]));
      clip_.strike_phase[0] = onnx_.clip_strike_phases()[0];
      clip_.strike_phase[1] = onnx_.clip_strike_phases()[1];
      std::fprintf(stderr,
          "[pp] clip layout from ONNX metadata: seg_len={%d,%d} strike_phase={%.3f,%.3f} "
          "(strike frames %d/%d)\n",
          clip_.seg_len[0], clip_.seg_len[1], clip_.strike_phase[0], clip_.strike_phase[1],
          clip_.strike_frame(0), clip_.strike_frame(1));
      // Baked-clip GROUNDING check (2026-07-03): the runner (and training's actor obs!) consume
      // the RAW clip-world reference. A properly re-grounded clip (scripts/reground_hope_frame.py,
      // e.g. the hopex lineage / p4) has frame-0 pelvis yaw == 0 -> refs coincide with the engage
      // heading and a strike-in-place policy deploys cleanly under perfect_tracking. A NON-re-
      // grounded clip (e.g. registry v4, pelvis yaw ~+82/+86 deg) trains a TURN-AND-WALK policy
      // whose footwork perfect_tracking cannot observe (base obs pinned to the ref pelvis) ->
      // open strike loop on hardware. Print per-clip baked yaw so a raw-clip model is never a
      // silent surprise again.
      for (int c = 0; c < 2; ++c) {
        const auto r0 = onnx_.refs(clip_.seg_start(c));
        const Vec4& q0 = r0.anchor_quat_w;
        const double yaw0 = std::atan2(2.0 * (q0[0] * q0[3] + q0[1] * q0[2]),
                                       1.0 - 2.0 * (q0[2] * q0[2] + q0[3] * q0[3]));
        const double yaw0_deg = yaw0 * 180.0 / M_PI;
        std::fprintf(stderr, "[pp] clip %d baked frame-0 anchor yaw = %+.1f deg%s\n", c, yaw0_deg,
                     (std::fabs(yaw0_deg) > 20.0)
                         ? "  ** NOT RE-GROUNDED: policy will TURN toward the clip heading and "
                           "step to its target; that footwork is INVISIBLE under perfect_tracking "
                           "(open strike loop). Use oracle/mocap localization, or deploy a model "
                           "trained on re-grounded (+X, yaw~0) clips. **"
                         : "");
      }
      // Periodic-wrap guard (2026-07-03): in the default periodic mode the tts clock wraps
      // (1-strike_lead_frac)*strike_period seconds after the strike. If that is shorter than a
      // clip's follow-through, the reference SNAPS back to the windup MID-FOLLOW-THROUGH — an
      // untracked transition that topples the free base (the p4 backhand failure signature).
      // v4 clips have LONG follow-throughs (fh 1.46 s / bh 1.74 s vs the 0.9 s default budget).
      if (!cfg_.single_swing && cfg_.swing_rest_s < 0.0) {
        const double post_budget = (1.0 - cfg_.strike_lead_frac) * cfg_.strike_period;
        for (int c = 0; c < 2; ++c) {
          const double follow_s =
              (clip_.seg_len[c] - 1 - (clip_.strike_frame(c) - clip_.seg_start(c))) * cfg_.dt;
          if (post_budget < follow_s - 1e-9) {
            std::fprintf(stderr,
                "[pp WARN] periodic mode wraps %.2f s after the strike but clip %d's "
                "follow-through is %.2f s -> the reference will SNAP mid-follow-through "
                "(untrained; topples the free base). Run with --single-swing or --swing-rest S, "
                "or raise strike_period.\n",
                post_budget, c, follow_s);
          }
        }
      }
    } else {
      std::fprintf(stderr,
          "[pp WARN] ONNX carries NO clip layout metadata -> using the hardcoded LEGACY v1 "
          "layout seg_len={%d,%d} strike_phase={%.2f,%.2f}. Only correct for v1-clip models "
          "(model_15200); a v2-baked model will swing against the WRONG reference frames.\n",
          clip_.seg_len[0], clip_.seg_len[1], clip_.strike_phase[0], clip_.strike_phase[1]);
    }
    nominal_q_sdk_ = to_sdk_order(onnx_.default_q(), isaac_to_sdk_);  // nominal pose in SDK order
    leg_qdes_smooth_ = nominal_q_sdk_;  // seed the leg q_des EMA at nominal (no jump on first release)
    // Official robust-stand PD gains (a3_pd_stand_*, 29-DOF policy view) scattered
    // to the 31 SDK slots via kA3PolicyToSdkIdx; neck slots get the fixed head PD.
    official_kp_sdk_ = Eigen::VectorXd::Zero(kNumJoints);
    official_kd_sdk_ = Eigen::VectorXd::Zero(kNumJoints);
    for (int i = 0; i < robot_io::kA3PolicyDof; ++i) {
      const int sdk = robot_io::kA3PolicyToSdkIdx[i];
      official_kp_sdk_[sdk] = a3_pd_stand_kps[i];
      official_kd_sdk_[sdk] = a3_pd_stand_kds[i];
    }
    for (int s : {kHeadSlot0, kHeadSlot1}) { official_kp_sdk_[s] = kHeadKp; official_kd_sdk_[s] = kHeadKd; }
    last_q_des_ = last_q_meas_ = last_qd_meas_ = Eigen::VectorXd::Zero(kNumJoints);
  }

  // Attach a SIM-ONLY oracle pelvis-pose source (shared by main; only consulted
  // when loc_mode == kOracle). On hardware the shm file is absent so the reader
  // fails to open and oracle mode falls back with a loud warning.
  void SetOracle(std::shared_ptr<PpOraclePose> oracle) { oracle_ = std::move(oracle); }

  // Attach LIVE planner inputs (Path B). racket_in feeds the racket target; base_in feeds
  // LocMode::kExternalBase. Both are written by the AimRT subscriber thread and read here
  // from the driver thread (each is internally lock-guarded + age-gated). Only consulted
  // when cfg_.planner_mode is set; absent/stale streams degrade to a held stand.
  void SetRacketInput(std::shared_ptr<PpRacketTargetInput> r) { racket_in_ = std::move(r); }
  void SetBasePoseInput(std::shared_ptr<PpBasePoseInput> b) { base_in_ = std::move(b); }
  bool planner_mode() const { return cfg_.planner_mode; }
  std::string planner_status() const {
    std::lock_guard<std::mutex> lk(planner_mu_);
    return planner_status_;
  }

  LocMode loc_mode() const { return cfg_.loc_mode; }
  const char* loc_mode_name() const {
    switch (cfg_.loc_mode) {
      case LocMode::kFabricated: return "fabricated(A)";
      case LocMode::kPerfectTracking: return "perfect_tracking(B)";
      case LocMode::kOracle: return "oracle(C)";
      case LocMode::kExternalBase: return "external_base(mocap)";
    }
    return "?";
  }

  // --- obs-debug snapshot (full 180-D obs + flags), read by the status thread ---
  struct ObsDebug {
    Eigen::VectorXd obs;          // last 180-D observation
    bool valid = false;
    bool oracle_enabled = false;  // loc_mode == kOracle
    bool oracle_fresh = false;    // a fresh oracle sample was used this tick
    double oracle_age_s = -1.0;   // age of last oracle sample (s), -1 if n/a
    std::uint64_t sync_miss = 0;  // ticks seen with sync_aligned == false (cumulative)
  };
  ObsDebug take_obs_debug() {
    std::lock_guard<std::mutex> lk(obs_mu_);
    ObsDebug d;
    d.obs = last_obs_;
    d.valid = last_obs_.size() == onnx_.obs_dim();
    d.oracle_enabled = (cfg_.loc_mode == LocMode::kOracle);
    d.oracle_fresh = oracle_fresh_;
    d.oracle_age_s = oracle_age_s_;
    d.sync_miss = sync_miss_;
    return d;
  }
  const Eigen::VectorXd& last_obs_unsafe() const { return last_obs_; }

  // --- live per-joint diagnostics (SDK order), read by the status thread ---
  struct DiagSnapshot {
    Eigen::VectorXd q_des, q_meas, qd_meas;                    // instantaneous
    Eigen::VectorXd des_range, meas_range, err_peak, qd_peak;  // over the window
    bool valid = false;                                        // window had samples
  };
  // Copy out diagnostics and reset the rolling window. Thread-safe.
  DiagSnapshot take_diag() {
    std::lock_guard<std::mutex> lk(diag_mu_);
    DiagSnapshot d;
    d.q_des = last_q_des_; d.q_meas = last_q_meas_; d.qd_meas = last_qd_meas_;
    if (ranges_init_) {
      d.des_range = des_hi_ - des_lo_;
      d.meas_range = meas_hi_ - meas_lo_;
      d.err_peak = err_peak_;
      d.qd_peak = qd_peak_;
      d.valid = true;
    }
    ranges_init_ = false;  // start a fresh window
    return d;
  }

  // Official robust stand (matches AGI's PD_STAND): pose = nominal (== a3_default_angles),
  // gains = production a3_pd_stand_*. All in 31-DOF SDK order.
  const Eigen::VectorXd& official_stand_q() const { return nominal_q_sdk_; }
  const Eigen::VectorXd& official_stand_kp() const { return official_kp_sdk_; }
  const Eigen::VectorXd& official_stand_kd() const { return official_kd_sdk_; }

  // Runtime swing level: 0 = hold wind-up (quasi-stand), 1 = periodic forehand.
  // Any EXTERNAL level change (keyboard, safety guard) cancels a pending swing-rest
  // auto re-arm — a guard trip must never re-enter the swing on its own.
  void set_level(int lvl) { rest_rearm_armed_.store(false); level_.store(lvl); }
  int level() const { return level_.load(); }

  // Live swing-speed tuning (real-time stretch; <1.0 slower). Clamped to a sane range.
  void set_swing_speed(double s) { swing_speed_.store(std::max(0.05, std::min(2.0, s))); }
  double swing_speed() const { return swing_speed_.load(); }

  // Live swing DIRECTION (scripted test path; no live planner). +1 = forehand
  // (target -y, baked clip 0), -1 = backhand (target +y, baked clip 1). Flips the
  // scripted target's y-sign and swing_type; the reference clock then selects the
  // matching baked clip via clip_id_from_swing_sign.
  //
  // MID-SWING LATCH (2026-07-04): applying a dir flip while a swing is in progress
  // snaps the 62-D reference obs from clip A's mid-swing frame to clip B's windup
  // while the BODY is mid-swing — the exact OOD transition training never contains
  // (clips only switch at a completed wrap + hold; see the Python runner's
  // active-swing lock and the free-base 'b'-key falls). At level 1 the request is
  // QUEUED and applied at the next safe boundary (level 0, or the next windup start
  // of the periodic/single-swing clock) in ComputeCommand.
  void set_swing_dir(int d) {
    const int want = d >= 0 ? 1 : -1;
    if (level_.load() == 1 && swing_dir_.load() != want) {
      pending_swing_dir_.store(want);
      std::fprintf(stderr, "[pp] swing dir -> %s QUEUED (mid-swing switch is OOD; "
                   "applies at the next windup/hold)\n", want > 0 ? "FOREHAND" : "BACKHAND");
      return;
    }
    swing_dir_.store(want);
  }
  int swing_dir() const { return swing_dir_.load(); }
  const char* swing_dir_name() const { return swing_dir_.load() >= 0 ? "FOREHAND" : "BACKHAND"; }

  // Re-capture the yaw-align offsets on the next policy tick. Called by the driver
  // whenever the mode transitions INTO SHADOW/MOTION from PASSIVE/PD_STAND (the robot
  // may have been turned/moved between engagements).
  void rearm_yaw_align() { yaw_align_pending_.store(true); }

  // Full-body gate: true => leg q_des is overwritten to nominal (NOT a full-body
  // test); false => the policy's leg actions pass through (31-DOF command check).
  // Atomic so --auto-leg-hold can flip it per-tick from the driver thread while the
  // status thread reads it. Initialised from cfg in the constructor.
  bool legs_passive() const { return legs_passive_.load(); }
  bool waist_passive() const { return waist_passive_.load(); }
  void set_legs_passive(bool v) { legs_passive_.store(v); }
  void set_waist_passive(bool v) { waist_passive_.store(v); }

  PpRacketTarget ScriptedTarget(std::uint64_t tick_idx) const {
    PpRacketTarget tg;
    const int dir = swing_dir_.load();  // +1 forehand (clip0) / -1 backhand (clip1)
    const int clip = dir >= 0 ? 0 : 1;
    // Per-clip target from the trained sampling boxes (see PpPolicyConfig) — no mirroring.
    tg.pos_w = cfg_.racket_pos_w_clip[clip];
    tg.vel_w = cfg_.racket_vel_w_clip[clip];
    tg.swing_sign = (dir >= 0 ? 1.0 : -1.0);  // +1 fore / -1 back
    tg.base_target_xy = Vec2::Zero();
    // Swing clock measured from the origin set on each level->1 entry (see ComputeCommand),
    // so a release from a long level-0 hold starts the swing at the WINDUP (matching the held
    // pose) instead of snapping into the free-running mid-cycle phase, which would mismatch the
    // body and lurch the robot. swing_speed<1 stretches the clock.
    const std::uint64_t origin = swing_clock_origin_.load();
    const double t = (tick_idx >= origin ? tick_idx - origin : 0) * cfg_.dt * swing_speed_.load();
    if (level_.load() == 0) {
      tg.time_to_strike = 5.0;  // far away -> clock holds at clip start (wind-up)
    } else if (cfg_.planner_mode) {
      // LIVE PLANNER: linear clock seeded from the ENGAGE-time tts (clamped to the clip's
      // windup length at engage) so the reference strike aligns with the ball's arrival.
      // Same no-wrap semantics as single_swing; completion still trips on tts < min_tts.
      tg.time_to_strike = planner_tts0_ - t;
    } else if (cfg_.single_swing || cfg_.swing_rest_s >= 0.0) {
      // SINGLE-SWING: linear clock, NO fmod wrap. The periodic schedule bounds tts to
      // [-(1-lead)*period, lead*period] = [-0.9, 2.1], which (a) never reaches the clip's
      // end (backhand needs tts=-1.76) so the follow-through frames 227..270 never play,
      // and (b) SNAPS the reference end->windup every period. Linear tts plays the WHOLE
      // clip once; ComputeCommand then drops to level 0 when the clip has fully played.
      tg.time_to_strike = cfg_.strike_lead_frac * cfg_.strike_period - t;
    } else {
      const double cyc = std::fmod(t, cfg_.strike_period);
      tg.time_to_strike = cfg_.strike_lead_frac * cfg_.strike_period - cyc;  // windup->strike->follow-through
    }
    return tg;
  }

  // CommandFn body. Fills a full 31-slot RobotCommand (SDK order). Always valid.
  bool ComputeCommand(std::uint64_t tick_idx, const robot_io::RobotState& state,
                      robot_io::RobotCommand& cmd) {
    // LIVE PLANNER (Path B): decide engage/hold from the latest planner command and drive
    // the EXISTING swing controls (set_swing_dir/set_level + freeze). Runs before the swing
    // clock logic so the 0->1 edge below resets the clock to the windup as usual. No-op in
    // the scripted/keyboard path (planner_mode == false).
    if (cfg_.planner_mode) PlannerEngageStep_(tick_idx);
    // Reset the swing clock to its windup on level 0->1 (release from hold) OR on a
    // forehand<->backhand switch. Either way the swing must (re)start from its WINDUP
    // (tts -> clip start, matching the current near-stand body) rather than snap into the
    // free-running mid-cycle phase. Pressing 'b' mid-forehand WITHOUT this reset jumps the
    // backhand reference straight to a mid-swing frame while the body is still in a
    // forehand-end pose -> reference/body mismatch -> lurch -> FALL (forehand is fine only
    // because it gets this clean windup start at MOTION entry).
    // Apply a QUEUED dir flip (set_swing_dir latch) only at a safe boundary: held stand,
    // or while the swing clock still sits at the windup start (tts clamped at max — early
    // cycle / just released). There the flip re-selects the OTHER clip's windup, the same
    // reference-pose family the clock reset produces anyway; mid-swing it would snap the
    // obs reference across clips (the 'b'-mid-forehand OOD fall).
    {
      const int pend = pending_swing_dir_.load();
      if (pend != 0 && (level_.load() == 0 || last_tts_at_windup_)) {
        pending_swing_dir_.store(0);
        swing_dir_.store(pend);
        std::fprintf(stderr, "[pp] queued swing dir applied -> %s\n",
                     pend > 0 ? "FOREHAND" : "BACKHAND");
      }
    }
    const int swing_lvl_now = level_.load();
    const int swing_dir_now = swing_dir_.load();
    if ((swing_lvl_now == 1 && swing_level_prev_ != 1) || swing_dir_now != swing_dir_prev_)
      swing_clock_origin_.store(tick_idx);
    swing_level_prev_ = swing_lvl_now;
    swing_dir_prev_ = swing_dir_now;
    PpRacketTarget tg = ScriptedTarget(tick_idx);
    const int clip_id = clip_id_from_swing_sign(tg.swing_sign);
    // Clamp time_to_strike to the clip's IN-TRAINING maximum. Training computes
    // tts = (strike_frame - current_frame)*dt from the actual clip frame, so its max is
    // (strike_frame - seg_start)*dt (backhand 0.86 s, forehand 1.30 s). The scripted schedule
    // instead feeds raw 2.1 s at cycle start / 5.0 s at hold — an OOD (tts, windup-frame)
    // pairing the policy never saw (worst for backhand: 1.24 s of OOD input right before the
    // swing; observed to precede the free-base backhand fall). Clamping makes the windup state
    // exactly the training state "at windup frame, tts=max". The reference clock is unaffected
    // (it already clamps ts to seg_start for any tts >= this bound).
    const double max_tts =
        (clip_.strike_frame(clip_id) - clip_.seg_start(clip_id)) * clip_.step_dt;
    if (tg.time_to_strike > max_tts) tg.time_to_strike = max_tts;
    last_tts_at_windup_ = (tg.time_to_strike >= max_tts - 1e-9);
    // SINGLE-SWING / REST (see PpPolicyConfig): once the clip has fully played, drop to
    // level 0 (held stand) instead of letting the periodic clock WRAP the reference from
    // the end pose back to windup (an untracked-in-training snap that topples the backhand).
    // min_tts = tts at the clip's last frame; below it the clock is clamped at the end.
    if ((cfg_.single_swing || cfg_.swing_rest_s >= 0.0) && swing_lvl_now == 1) {
      const double min_tts = (clip_.strike_frame(clip_id) -
                              (clip_.seg_start(clip_id) + clip_.seg_len[clip_id] - 1)) *
                             clip_.step_dt;
      if (tg.time_to_strike < min_tts) {
        level_.store(0);
        if (cfg_.planner_mode) planner_hold_start_tick_ = tick_idx;  // recovery-window clock
        if (cfg_.swing_rest_s >= 0.0) {
          rest_rearm_tick_ = tick_idx + static_cast<std::uint64_t>(
              std::max(0.0, cfg_.swing_rest_s) / std::max(cfg_.dt, 1e-6));
          rest_rearm_armed_ = true;
        }
        std::fprintf(stderr, "[pp] swing complete -> level 0 (held stand)%s\n",
                     cfg_.swing_rest_s >= 0.0 ? " (auto re-arm after rest)" : "; press 1 to swing again");
      }
    }
    // Auto re-arm after the rest (only if WE dropped the level; a manual '0' clears it).
    // NOT in planner mode: there a swing re-engages only on a fresh VALID command
    // (PlannerEngageStep_); rest_rearm_tick_ is reused there purely as the rest timer.
    if (!cfg_.planner_mode && rest_rearm_armed_ && level_.load() == 0 &&
        tick_idx >= rest_rearm_tick_) {
      rest_rearm_armed_ = false;
      level_.store(1);  // next tick's 0->1 edge resets the swing clock to windup
    }
    // MIN-side OBS clamp (2026-07-04): the reference clock clamps the FRAME at the clip
    // end, but the raw tts kept decreasing into values training never paired with the
    // frozen end frame (periodic mode with a raised strike_period). Clamp the OBS tts to
    // the in-training minimum, symmetric with the max clamp above. AFTER the completion
    // check on purpose — that check needs the raw sub-minimum tts to detect clip end.
    {
      const double min_tts_clip = (clip_.strike_frame(clip_id) -
                                   (clip_.seg_start(clip_id) + clip_.seg_len[clip_id] - 1)) *
                                  clip_.step_dt;
      if (tg.time_to_strike < min_tts_clip) tg.time_to_strike = min_tts_clip;
    }
    const int time_step = clip_.time_step_for(clip_id, tg.time_to_strike);

    const PpRefs refs = onnx_.refs(time_step);

    if (!state.sync_aligned) ++sync_miss_;  // dropped/unaligned state packet count

    PpRobotState st;
    st.q = from_sdk_order(state.q, isaac_to_sdk_);    // SDK -> Isaac
    st.qd = from_sdk_order(state.dq, isaac_to_sdk_);
    st.base_quat_w = state.imu_quat_wxyz;             // real pelvis IMU orientation
    st.base_ang_vel_b = state.imu_gyro;               // real pelvis gyro (body frame)
    // torso ORIENTATION from the real secondary (torso) IMU when available
    // (identity was wrong -> broke the anchor term -> robot fell). Measurable in
    // every mode, so always use the real value.
    st.torso_quat_w = state.has_secondary_imu ? state.sec_imu_quat_wxyz
                                              : cfg_.nominal_torso_quat_w;

    // YAW-ALIGN (see PpPolicyConfig::yaw_align). Capture each IMU's yaw on the first
    // policy tick after (re)engage, then express every subsequent attitude relative to
    // that entry heading. Fixes the boot-drift yaw polluting motion_anchor_ori_b and the
    // racket-FK world conversion on hardware; no-op in sim where spawn yaw ~ 0.
    if (cfg_.yaw_align) {
      if (yaw_align_pending_.load()) {
        // UPRIGHT + STATIONARY GUARD (2026-07-04): capturing while the robot is tilted,
        // turning, or fallen bakes a garbage offset into EVERY subsequent obs (yaw of a
        // fallen quat is ill-defined; observed in the ROS runner: all base-relative
        // targets rotated ~125 deg, magnitude untouched). Defer the capture until the
        // robot is upright (proj gravity ~[0,0,-1]) and still (|gyro| small); warn while
        // waiting so a hoisted/leaning engage is visible instead of silently wrong.
        // body-frame gravity z from the raw base quat (w,x,y,z): R(q)^T·[0,0,-1] |_z
        const double gz = 2.0 * (st.base_quat_w[1] * st.base_quat_w[1] +
                                 st.base_quat_w[2] * st.base_quat_w[2]) - 1.0;
        const double gyro_n = st.base_ang_vel_b.norm();
        if (gz > -0.95 || gyro_n > 0.5) {
          if (++yaw_align_defer_ticks_ % 50 == 1) {
            std::fprintf(stderr,
                "[pp WARN] yaw-align DEFERRED: robot not upright/still (gravZ=%+.2f "
                "|gyro|=%.2f); stand the robot at its heading to capture.\n", gz, gyro_n);
          }
        } else {
          yaw_align_pending_.store(false);
          yaw_align_defer_ticks_ = 0;
          yaw0_base_inv_ = quat_inv(yaw_quat(st.base_quat_w));
          yaw0_torso_inv_ = quat_inv(yaw_quat(st.torso_quat_w));
        const auto yaw_deg = [](const Vec4& q) {
          return std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                            1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])) * 180.0 / M_PI;
        };
        std::fprintf(stderr,
            "[pp] yaw-align captured at policy engage: base_yaw=%+.1f deg torso_yaw=%+.1f deg "
            "(subtracted from all subsequent IMU attitudes; robot heading at engage == clip +x)\n",
            yaw_deg(st.base_quat_w), yaw_deg(st.torso_quat_w));
        }
      }
      st.base_quat_w = quat_mul(yaw0_base_inv_, st.base_quat_w);
      st.torso_quat_w = quat_mul(yaw0_torso_inv_, st.torso_quat_w);
    }
    if (!state.has_secondary_imu && !sec_imu_warned_) {
      sec_imu_warned_ = true;
      std::fprintf(stderr,
          "[pp WARN] secondary (torso) IMU ABSENT -> torso orientation falls back to "
          "identity; motion_anchor_ori_b (and the anchor frame feeding "
          "motion_anchor_pos_b) will be WRONG. Do NOT run MOTION on hardware without "
          "a working torso IMU.\n");
    }

    // --- localization-dependent world pose (3 modes; obs LAYOUT unchanged) ---
    oracle_fresh_ = false;
    oracle_age_s_ = -1.0;
    switch (cfg_.loc_mode) {
      case LocMode::kOracle: {  // ===== C: SIMULATION ONLY (true MuJoCo pose) =====
        PpOracleSample s;
        if (oracle_ && oracle_->Latest(s, cfg_.oracle_max_age_s)) {
          st.base_pos_w = s.pos;       // true world pelvis position
          st.base_quat_w = s.quat;     // true world pelvis orientation
          oracle_fresh_ = true;
          oracle_age_s_ = s.age_s;
          st.torso_pos_w = torso_pos_from_base(st.base_pos_w, st.base_quat_w, st.q);
        } else {  // stale/missing oracle -> SAFE fallback to perfect-tracking, warn LOUDLY
          // 2026-07-03: this used to warn ONCE and then silently degrade — an --oracle-pelvis
          // A/B run with the bridge down produced a perfect_tracking run in disguise (the two
          // "different" loc-mode tests were identical). Now: repeat the warning every ~2 s and
          // mark the fallback in oracle_fresh_ so the [obs] status line shows fresh=0.
          oracle_fresh_ = false;
          if ((oracle_warn_tick_++ % 100) == 0) {
            std::fprintf(stderr,
                "[pp ORACLE] NO FRESH SAMPLE (bridge down / stale shm?) -> running as "
                "perfect-tracking. This is NOT an oracle run — start "
                "scripts/run_oracle.sh first and require 'fresh=1' in the [obs] line.\n");
          }
          st.base_pos_w = refs.ref_pelvis_pos_w;
          st.torso_pos_w = refs.anchor_pos_w;
        }
        break;
      }
      case LocMode::kPerfectTracking: {  // ===== B: assume perfect position tracking =====
        // racket/base-target relative to where the pelvis SHOULD be (the reference),
        // and zero the anchor POSITION error so the policy is not fed a fictional
        // world-tracking error. Orientation terms stay real (IMU above).
        st.base_pos_w = refs.ref_pelvis_pos_w;
        st.torso_pos_w = refs.anchor_pos_w;   // -> motion_anchor_pos_b == 0
        break;
      }
      case LocMode::kExternalBase: {  // ===== HARDWARE planner: live mocap base POSITION =====
        // Mocap is position-only, so take the real base POSITION and keep the yaw-aligned
        // IMU orientation (st.base_quat_w set above). Torso position follows the base by FK.
        PpBaseSample s;
        if (base_in_ && base_in_->Latest(s, cfg_.external_base_max_age_s)) {
          st.base_pos_w = s.pos;
          st.torso_pos_w = torso_pos_from_base(st.base_pos_w, st.base_quat_w, st.q);
          base_fresh_ = true;
          oracle_age_s_ = s.age_s;
        } else {  // stale/absent mocap -> SAFE fallback to perfect-tracking + loud warn.
          base_fresh_ = false;
          if ((base_warn_tick_++ % 100) == 0) {
            std::fprintf(stderr,
                "[pp EXT-BASE] NO FRESH mocap base sample (relay down / stale?) -> running as "
                "perfect-tracking; planner engage is BLOCKED until the base stream returns.\n");
          }
          st.base_pos_w = refs.ref_pelvis_pos_w;
          st.torso_pos_w = refs.anchor_pos_w;
        }
        break;
      }
      case LocMode::kFabricated:  // ===== A: legacy fabricated nominal pose =====
      default: {
        st.base_pos_w = cfg_.nominal_base_pos_w;
        if (cfg_.use_base_estimator)  // leg-FK + IMU pelvis height (planted-foot stance)
          st.base_pos_w[2] = estimate_base_height(st.q, st.base_quat_w);
        // torso POSITION = base + waist-FK offset (~base + 5 mm up).
        st.torso_pos_w = torso_pos_from_base(st.base_pos_w, st.base_quat_w, st.q);
        break;
      }
    }
    last_base_pos_ = st.base_pos_w;
    last_base_quat_w_ = st.base_quat_w;  // yaw-aligned; PlannerEngageStep_ gates on it next tick

    // LIVE PLANNER static stand at level 0 — in TWO regimes:
    //   (a) pre-FIRST-engage (the Python runner's proven _stand-until-engage design;
    //       running the policy hold from a cold stand knelt the robot within ~2 s), and
    //   (b) POST-RECOVERY: hold_recover_s after a completed swing. The policy hold must
    //       run first (it actively balances out of the follow-through — a static stand
    //       cannot), but the model's level-0 hold only has ~5 s of margin (Gate 2.5 +
    //       closed-loop falls), so after the recovery window we blend to the static
    //       official stand and stay there until the next engage.
    // Localization/engage above still run every tick; an engage (level 0->1) exits this
    // branch and main's blend covers the stand -> swing transition (the Gate-2-proven
    // MOTION-entry path). q_des ramps measured -> nominal over hold_blend_s so the stiff
    // official gains never snap onto a displaced pose (the kp-2000 catapult class).
    {
      // Handoff is QUIESCENCE-GATED, not time-only: a timed switch fell 0.6 s after the
      // blend began (the robot still carried follow-through momentum — the documented
      // "blended static stand cannot balance out of the follow-through" failure). The
      // policy hold keeps actively balancing until the robot is upright AND still; a
      // force-switch at recover+3 s bounds the stay inside the fragile ~5-10 s window.
      const double t_since =
          (tick_idx - planner_hold_start_tick_) * cfg_.dt;
      const bool upright_still =
          projected_gravity_body(st.base_quat_w)[2] < -0.95 &&
          st.base_ang_vel_b.norm() < 0.4 &&
          (st.qd.size() == 0 || st.qd.cwiseAbs().maxCoeff() < 1.0);
      const bool post_recovery = planner_have_hold_ &&
          ((t_since > cfg_.hold_recover_s && upright_still) ||
           t_since > cfg_.hold_recover_s + 3.0);
      // STICKY: once static engages it stays until the next swing (level 1). A quiescence
      // condition that flaps re-enters the branch every few ticks — policy/static command
      // CHATTER with a restarted blend each time (observed: 9 re-entries then a fall).
      if (cfg_.planner_mode && !planner_engaged_ && level_.load() == 0 &&
          (!planner_have_hold_ || post_recovery || planner_static_active_)) {
        if (!planner_static_active_) {
          planner_static_active_ = true;
          planner_static_start_tick_ = tick_idx;
          planner_static_q0_ = state.q.size() == kNumJoints ? state.q : nominal_q_sdk_;
          if (planner_have_hold_)
            std::fprintf(stderr,
                "[pp] post-swing recovery done -> STATIC official stand until next engage\n");
        }
        const double a = std::min(1.0,
            (tick_idx - planner_static_start_tick_) * cfg_.dt /
                std::max(cfg_.hold_blend_s, 1e-3));
        cmd.q_des = (1.0 - a) * planner_static_q0_ + a * nominal_q_sdk_;
        cmd.dq_des = Eigen::VectorXd::Zero(kNumJoints);
        cmd.tau_ff = Eigen::VectorXd::Zero(kNumJoints);
        cmd.kp = official_kp_sdk_;
        cmd.kd = official_kd_sdk_;
        return true;
      }
      // Reset the sticky latch only when a swing actually runs (engage exited the branch);
      // never mid-hold, or quiescence flapping chatters between policy and static commands.
      if (level_.load() == 1) planner_static_active_ = false;
    }

    // LIVE PLANNER target override (Path B): the swing clock already set tg.time_to_strike,
    // swing_sign, clip_id and time_step above; here we only swap the REACH POINT. During a
    // swing use the FROZEN world target; while holding, use a base-anchored ready target at
    // racket-reach x (so the footwork policy is not commanded to walk to a fixed world point
    // during the hold — the wbc_runner rest-hold semantics). Untouched when not planner_mode.
    if (cfg_.planner_mode) {
      if (planner_engaged_) {   // active swing -> frozen world target
        tg.pos_w = planner_frozen_pos_w_;
        tg.vel_w = planner_frozen_vel_w_;
      } else {                  // idle/rest (incl. before the first engage) -> base-anchored hold
        const Vec4 base_yaw = yaw_quat(st.base_quat_w);
        Vec3 hb(cfg_.hold_anchor_x_b, planner_hold_pos_b_engage_[1], 0.0);
        tg.pos_w = st.base_pos_w + quat_rotate(base_yaw, hb);
        tg.pos_w[2] = planner_hold_z_w_;
        tg.vel_w = planner_frozen_vel_w_;
      }
    }

    last_proj_grav_ = projected_gravity_body(st.base_quat_w);

    // 175-D deploy_parity (new policy) vs 180-D full (model_15200). Auto-selected from the loaded
    // ONNX input dim. build_obs_175 drops motion_anchor_pos_b + base_target_pos_b and reframes the
    // racket target to be relative to the CURRENT racket FK (pp_racket_fk.hpp) — no world base pos.
    const Eigen::VectorXd obs = (onnx_.obs_dim() == kObsDim175)
        ? build_obs_175(refs, st, tg, last_action_, onnx_.default_q(), cfg_.use_imu_yaw_for_targets)
        : build_obs_180(refs, st, tg, last_action_, onnx_.default_q(), cfg_.use_imu_yaw_for_targets);
    { std::lock_guard<std::mutex> lk(obs_mu_); last_obs_ = obs; }  // for obs-debug
    const Eigen::VectorXd action = onnx_.mean_action(obs, time_step);
    const Eigen::VectorXd tq_isaac = onnx_.target_q(action);

    Eigen::VectorXd q_sdk = to_sdk_order(tq_isaac, isaac_to_sdk_);
    Eigen::VectorXd kp_sdk = to_sdk_order(onnx_.kp(), isaac_to_sdk_);
    Eigen::VectorXd kd_sdk = to_sdk_order(onnx_.kd(), isaac_to_sdk_);

    // NECK PASSIVE: ignore model neck outputs; hold head at nominal w/ fixed PD.
    for (int s : {kHeadSlot0, kHeadSlot1}) {
      q_sdk[s] = kHeadPosRad;
      kp_sdk[s] = kHeadKp;
      kd_sdk[s] = kHeadKd;
    }

    // LEGS PASSIVE (hoisted demo): hold legs at nominal stand with the trained
    // leg PD. Removes leg twitch caused by balance corrections against the
    // nominal-base-position obs (no localisation). Arm + waist still swing.
    if (legs_passive_.load()) {
      // Hold legs at nominal with the TRAINED leg PD (ran clean on the hoist).
      // The stiff official ground-stand gains buzz/swing a hoisted robot, so
      // they are NOT used here; they live behind --official-stand for Step 2.
      // With --auto-leg-hold this flips true at level 0 / false at level 1.
      for (int s = kLegSlotStart; s < kLegSlotStart + kLegSlotCount; ++s)
        q_sdk[s] = nominal_q_sdk_[s];
    }

    // WAIST PASSIVE: hold the waist (slots 0..2) at nominal so the torso stays
    // upright. The policy drives waist_pitch to its forward limit which (with the
    // forehand arms forward) shifts the CoM past the feet — a static leg hold can't
    // rebalance that. Freezing the waist makes the swing ARMS-ONLY but keeps the
    // CoM over the base of support. Gains: official ground-stand kp/kd applied in
    // a3_pingpong_main when --official-stand is set (else the trained waist PD).
    if (waist_passive_.load()) {
      for (int s = kWaistSlotStart; s < kWaistSlotStart + kWaistSlotCount; ++s)
        q_sdk[s] = nominal_q_sdk_[s];
    }

    // LEG q_des CLAMP (released full-body swing only): the trained swing commands a
    // deep crouch-and-lean (hip_pitch -0.6..-0.77, knee +0.6, ankle_pitch -0.7..-0.9
    // rad) that assumes planted-foot contact dynamics. On the real robot that posture
    // is not a stable static stand -> tracking it sinks the knees / pitches forward.
    // Clamp each policy-driven leg slot to nominal ± leg_clamp_rad_ to keep the legs
    // near the proven upright stand while leaving room for small balance moves. The
    // policy still sees the true measured q in obs, so its feedback loop is intact.
    // No-op when legs are HELD (already nominal) or when the band is 0 (off).
    if (leg_clamp_rad_ > 0.0 && !legs_passive_.load()) {
      for (int s = kLegSlotStart; s < kLegSlotStart + kLegSlotCount; ++s) {
        const double lo = nominal_q_sdk_[s] - leg_clamp_rad_;
        const double hi = nominal_q_sdk_[s] + leg_clamp_rad_;
        q_sdk[s] = std::min(std::max(q_sdk[s], lo), hi);
      }
    }

    // LEG q_des LOW-PASS (released swing only): EMA-smooth the leg q_des so stiff
    // weight-bearing gains (--leg-stand-gains, kp~2000) track a SMOOTH reference
    // instead of the policy's tick-to-tick jitter (which they amplify into a TWITCH).
    // Runs AFTER the clamp, so the EMA of in-band values stays in band. Seeded from
    // nominal; while legs are HELD it tracks nominal so the next release starts smooth.
    if (leg_smooth_alpha_ < 1.0 && leg_qdes_smooth_.size() == kNumJoints) {
      const double a = leg_smooth_alpha_;
      const bool released = !legs_passive_.load();
      for (int s = kLegSlotStart; s < kLegSlotStart + kLegSlotCount; ++s) {
        leg_qdes_smooth_[s] = released ? (a * q_sdk[s] + (1.0 - a) * leg_qdes_smooth_[s])
                                       : q_sdk[s];  // held: track nominal, re-seed for release
        q_sdk[s] = leg_qdes_smooth_[s];
      }
    }

    // SAFETY: clamp q_des to the MJCF/URDF joint position limits before publish.
    // In-range commands are untouched (no-op for in-distribution actions); a
    // nonzero count means the policy commanded out-of-limit -> a red flag we warn
    // about once (check gains / targets / loc mode before continuing on hardware).
    const Eigen::VectorXd q_preclamp = q_sdk;  // pre-clamp, for the per-joint audit
    last_clamp_count_ = clamp_q_to_limits(q_sdk);
    ++clamp_ticks_;
    for (int i = 0; i < kNumJoints; ++i) {  // per-backend-slot clamp stats (waist_roll audit)
      const double viol = std::abs(q_preclamp[i] - q_sdk[i]);
      if (viol > 1e-9) {
        ++clamp_count_[i];
        if (viol > clamp_max_viol_[i]) clamp_max_viol_[i] = viol;
      }
    }
    if (last_clamp_count_ > 0 && !clamp_warned_) {
      clamp_warned_ = true;
      std::fprintf(stderr,
          "[pp WARN] q_des clamped to joint limits on %d joint(s) (policy commanded "
          "out-of-range; check gains/targets/loc-mode)\n",
          last_clamp_count_);
    }

    // One-shot comprehensive FIRST-TICK debug dump (joint pos/vel, IMU/gravity,
    // full per-block obs stats, raw ONNX action stats, decoded q_des/kp/kd) for
    // bring-up + AGI staff review. Fires on the first policy tick only.
    if (!dbg_done_) {
      dbg_done_ = true;
      LogFirstTick(obs, action, q_sdk, kp_sdk, kd_sdk, st, state, time_step);
    }

    cmd.q_des = q_sdk;
    cmd.kp = kp_sdk;
    cmd.kd = kd_sdk;
    cmd.dq_des = Eigen::VectorXd::Zero(kNumJoints);
    cmd.tau_ff = Eigen::VectorXd::Zero(kNumJoints);

    // --- diagnostics: snapshot + rolling per-joint ranges (SDK order) ---
    if (state.q.size() == kNumJoints && state.dq.size() == kNumJoints) {
      const Eigen::VectorXd err = (q_sdk - state.q).cwiseAbs();
      const Eigen::VectorXd qda = state.dq.cwiseAbs();
      std::lock_guard<std::mutex> lk(diag_mu_);
      last_q_des_ = q_sdk;
      last_q_meas_ = state.q;
      last_qd_meas_ = state.dq;
      if (!ranges_init_) {
        des_lo_ = des_hi_ = q_sdk;
        meas_lo_ = meas_hi_ = state.q;
        err_peak_ = err;
        qd_peak_ = qda;
        ranges_init_ = true;
      } else {
        des_lo_ = des_lo_.cwiseMin(q_sdk);     des_hi_ = des_hi_.cwiseMax(q_sdk);
        meas_lo_ = meas_lo_.cwiseMin(state.q); meas_hi_ = meas_hi_.cwiseMax(state.q);
        err_peak_ = err_peak_.cwiseMax(err);
        qd_peak_ = qd_peak_.cwiseMax(qda);
      }
    }

    last_action_ = action;
    last_time_step_ = time_step;
    return true;
  }

  // Bind to a std::function with the a3_deploy::CommandFn signature.
  std::function<bool(std::uint64_t, const robot_io::RobotState&, robot_io::RobotCommand&)>
  AsCommandFn() {
    return [this](std::uint64_t tick, const robot_io::RobotState& s,
                  robot_io::RobotCommand& c) { return ComputeCommand(tick, s, c); };
  }

  int last_time_step() const { return last_time_step_; }
  Vec3 last_proj_grav() const { return last_proj_grav_; }

  // Refresh the IMU-derived diagnostic (projected gravity) from the latest backend
  // state, INDEPENDENT of the policy running. ComputeCommand only runs in
  // SHADOW/MOTION, so without this the status/trace gravity FREEZES at the [0,0,-1]
  // default in PASSIVE/PD_STAND -- which hides whether the robot is actually upright
  // on the ground (the whole point of the PD_STAND ground check). Call every tick in
  // every mode. Diagnostic-only: does NOT touch the published command.
  void observe_imu(const robot_io::RobotState& state) {
    if (state.imu_quat_wxyz.size() == 4 && state.imu_quat_wxyz.norm() > 0.5)
      last_proj_grav_ = projected_gravity_body(state.imu_quat_wxyz);
  }

  // --- clamp audit (per backend slot; for the waist_roll mismatch investigation) ---
  int last_clamp_count() const { return last_clamp_count_; }       // # joints clamped last tick
  std::uint64_t clamp_ticks() const { return clamp_ticks_; }       // ticks the clamp has run
  std::uint64_t clamp_count_for(int slot) const {                  // times this slot was clamped
    return (slot >= 0 && slot < kNumJoints) ? clamp_count_[slot] : 0;
  }
  double clamp_max_viol_for(int slot) const {                      // worst out-of-range amount (rad)
    return (slot >= 0 && slot < kNumJoints) ? clamp_max_viol_[slot] : 0.0;
  }
  int worst_clamped_slot() const {                                 // most-clamped backend slot (-1 none)
    int w = -1; std::uint64_t best = 0;
    for (int i = 0; i < kNumJoints; ++i)
      if (clamp_count_[i] > best) { best = clamp_count_[i]; w = i; }
    return w;
  }
  Vec3 last_base_pos() const { return last_base_pos_; }
  const Eigen::VectorXd& last_action() const { return last_action_; }
  const std::array<int, 31>& isaac_to_sdk() const { return isaac_to_sdk_; }
  PpOnnxPolicy& onnx() { return onnx_; }

 private:
  void set_planner_status_(const char* s) {
    std::lock_guard<std::mutex> lk(planner_mu_);
    if (planner_status_ != s) planner_status_ = s;
  }

  // Live-planner engage machine (Path B). Reproduces the PROVEN Python wbc_runner._tick:
  // while a swing runs, the target is FROZEN and the existing clock/completion owns it (no
  // mid-swing abort on planner flutter); at idle, gate a fresh VALID command (timeout /
  // invalid-grace / min-tts / base-low / reachability) and, if it passes, FREEZE the target
  // and drive the EXISTING controls (set_swing_dir + set_level(1)). Uses the PREVIOUS tick's
  // localized base (1-tick lag @50 Hz is negligible) so it can run before localization.
  void PlannerEngageStep_(std::uint64_t tick_idx) {
    if (level_.load() == 1) { set_planner_status_("swinging"); return; }  // frozen, in flight
    planner_engaged_ = false;  // level 0: idle/hold (ready-hold override uses planner_have_hold_)

    // Inter-swing rest: the completion path armed rest_rearm_tick_ (planner mode never
    // auto-re-arms; it is reused purely as a settle timer). Hold until it elapses.
    if (rest_rearm_armed_ && tick_idx < rest_rearm_tick_) { set_planner_status_("rest"); return; }

    if (!racket_in_) { set_planner_status_("no_input"); return; }
    const auto snap = racket_in_->Latest();
    if (!snap.has_valid) { set_planner_status_("no_command"); return; }

    const double tts = snap.cmd.time_to_strike - snap.valid_age_s;  // decays since send
    if (snap.valid_age_s > cfg_.command_timeout_s) { set_planner_status_("stale"); return; }
    if (snap.invalid_after && snap.valid_age_s > cfg_.planner_invalid_grace_s) {
      set_planner_status_("planner_invalid"); return;
    }
    if (tts < cfg_.engage_min_tts_s) { set_planner_status_("too_late"); return; }

    // A stale localization frame makes the base obs (and this gate) incoherent -> block
    // engage. Covers BOTH live-base modes: external_base (mocap) and oracle (sim GT) —
    // a stale-oracle run silently degrades to the reference pelvis, and engaging on that
    // fictional base would gate/aim against a position the robot is not at.
    if ((cfg_.loc_mode == LocMode::kExternalBase && !base_fresh_) ||
        (cfg_.loc_mode == LocMode::kOracle && !oracle_fresh_)) {
      set_planner_status_("no_base"); return;
    }

    const Vec3 base_pos = last_base_pos_;
    const Vec4 base_yaw = yaw_quat(last_base_quat_w_);
    if (base_pos[2] < cfg_.base_low_z) { set_planner_status_("base_low"); return; }

    // Racket target -> policy WORLD frame. frame_code 0 = same world as the base (planner
    // table frame == mocap world). frame_code 1 = base_link-relative -> lift to world
    // (BOTH position and velocity rotate; a translated-only velocity would mix frames
    // once the robot has turned).
    Vec3 pos_w = snap.cmd.pos_w;
    Vec3 vel_w = snap.cmd.vel_w;
    if (snap.cmd.frame_code == 1) {
      pos_w = base_pos + quat_rotate(base_yaw, snap.cmd.pos_w);
      vel_w = quat_rotate(base_yaw, snap.cmd.vel_w);
    }

    const Vec3 tgt_b = quat_rotate_inverse(base_yaw, pos_w - base_pos);
    if (cfg_.target_gate_enable) {
      const bool ok = tgt_b[0] >= cfg_.gate_x_lo && tgt_b[0] <= cfg_.gate_x_hi &&
                      std::abs(tgt_b[1]) <= cfg_.gate_y_abs &&
                      pos_w[2] >= cfg_.gate_z_lo && pos_w[2] <= cfg_.gate_z_hi &&
                      vel_w.norm() <= cfg_.gate_speed_max;
      if (!ok) {
        // Throttled detail print (mirrors the Python runner's gate warn): without the
        // inputs a rejection is undebuggable at the venue.
        if ((gate_warn_tick_++ % 50) == 0) {
          std::fprintf(stderr,
              "[pp gate] REJECT base-rel (%+.2f,%+.2f) z_w=%.2f |v|=%.2f tts=%.2f "
              "(need x[%.2f,%.2f] |y|<=%.2f z[%.2f,%.2f] v<=%.2f)\n",
              tgt_b[0], tgt_b[1], pos_w[2], vel_w.norm(), tts,
              cfg_.gate_x_lo, cfg_.gate_x_hi, cfg_.gate_y_abs,
              cfg_.gate_z_lo, cfg_.gate_z_hi, cfg_.gate_speed_max);
        }
        set_planner_status_("target_gate"); return;
      }
    }

    // Swing side from the BASE-RELATIVE y (raw world-y is always <0 in the table frame).
    const double sign = swing_sign_from_target_y(tgt_b[1]);

    // ENGAGE: freeze target, lock side, release the swing. set_swing_dir applies immediately
    // at level 0; the 0->1 edge in ComputeCommand resets the swing clock to the windup.
    // tts0 is stored CLAMPED to the clip's windup length and DRIVES the swing clock
    // (ScriptedTarget planner branch: tts = tts0 - t), so the STRIKE fires when the ball
    // arrives — not a fixed clip-length after engage. Mirrors wbc_runner's
    // `"tts0": min(tts, max_tts0)`; without the transfer every strike would be late by
    // (max_tts - planner_tts).
    {
      const int eng_clip = clip_id_from_swing_sign(sign);
      const double max_tts0 =
          (clip_.strike_frame(eng_clip) - clip_.seg_start(eng_clip)) * clip_.step_dt;
      planner_tts0_ = std::min(tts, max_tts0);
    }
    planner_frozen_pos_w_ = pos_w;
    planner_frozen_vel_w_ = vel_w;
    planner_frozen_sign_ = sign;
    planner_hold_pos_b_engage_ = tgt_b;
    planner_hold_z_w_ = pos_w[2];
    planner_have_hold_ = true;
    planner_engaged_ = true;
    set_swing_dir(sign >= 0.0 ? 1 : -1);
    set_level(1);
    std::fprintf(stderr,
        "[pp engage] %s locked: tgt base-rel (%+.2f,%+.2f,%+.2f) tts=%.2fs (clock tts0=%.2fs)\n",
        sign > 0 ? "forehand" : "backhand", tgt_b[0], tgt_b[1], tgt_b[2], tts, planner_tts0_);
    set_planner_status_("engage");
  }

  // One-shot first-tick diagnostic dump (stderr). action = raw Isaac-order policy
  // output; q_sdk/kp_sdk/kd_sdk = final backend-slot command; st = policy-frame
  // robot state; state = raw backend RobotState (SDK order).
  void LogFirstTick(const Eigen::VectorXd& obs, const Eigen::VectorXd& action,
                    const Eigen::VectorXd& q_sdk, const Eigen::VectorXd& kp_sdk,
                    const Eigen::VectorXd& kd_sdk, const PpRobotState& st,
                    const robot_io::RobotState& state, int time_step) const {
    auto S = [](const Eigen::VectorXd& v) {
      char b[96];
      std::snprintf(b, sizeof b, "min=%+.4f mean=%+.4f max=%+.4f |.|=%.4f",
                    v.minCoeff(), v.mean(), v.maxCoeff(), v.norm());
      return std::string(b);
    };
    std::fprintf(stderr,
        "\n===================== [pp FIRST-TICK DEBUG] =====================\n");
    std::fprintf(stderr, " loc_mode=%s  time_step=%d  swing_level=%d  obs_dim=%d act_dim=%d\n",
                 loc_mode_name(), time_step, level_.load(), (int)obs.size(), (int)action.size());
    const Vec3 g = last_proj_grav_;
    std::fprintf(stderr,
        " IMU: base_quat(wxyz)=[%+.3f %+.3f %+.3f %+.3f] proj_grav=[%+.3f %+.3f %+.3f]\n"
        "      gyro=[%+.3f %+.3f %+.3f] sec_imu=%d torso_quat=[%+.3f %+.3f %+.3f %+.3f]\n",
        st.base_quat_w[0], st.base_quat_w[1], st.base_quat_w[2], st.base_quat_w[3],
        g[0], g[1], g[2], state.imu_gyro[0], state.imu_gyro[1], state.imu_gyro[2],
        (int)state.has_secondary_imu, st.torso_quat_w[0], st.torso_quat_w[1],
        st.torso_quat_w[2], st.torso_quat_w[3]);
    if (state.q.size() == kNumJoints) std::fprintf(stderr, " STATE(SDK) q : %s\n", S(state.q).c_str());
    if (state.dq.size() == kNumJoints) std::fprintf(stderr, " STATE(SDK) qd: %s\n", S(state.dq).c_str());
    struct Blk { const char* n; int lo; int len; };
    static const Blk blks180[] = {
        {"command", 0, 62}, {"motion_anchor_pos_b", 62, 3}, {"motion_anchor_ori_b", 65, 6},
        {"base_ang_vel", 71, 3}, {"joint_pos_rel", 74, 31}, {"joint_vel", 105, 31},
        {"actions(last)", 136, 31}, {"projected_gravity", 167, 3}, {"base_target_pos_b", 170, 2},
        {"racket_target_pos_b", 172, 3}, {"racket_target_vel_w", 175, 3},
        {"time_to_strike", 178, 1}, {"swing_type", 179, 1}};
    // deploy_parity 175-D: motion_anchor_pos_b + base_target_pos_b dropped; racket_target_pos_b is
    // relative to the CURRENT racket FK (not base). Everything after motion_anchor_ori_b shifts down 3.
    static const Blk blks175[] = {
        {"command", 0, 62}, {"motion_anchor_ori_b", 62, 6}, {"base_ang_vel", 68, 3},
        {"joint_pos_rel", 71, 31}, {"joint_vel", 102, 31}, {"actions(last)", 133, 31},
        {"projected_gravity", 164, 3}, {"racket_target_pos_b(relFK)", 167, 3},
        {"racket_target_vel_w", 170, 3}, {"time_to_strike", 173, 1}, {"swing_type", 174, 1}};
    const bool dp = (obs.size() == kObsDim175);
    std::fprintf(stderr, " OBS blocks (%d-D):\n", (int)obs.size());
    const Blk* blks = dp ? blks175 : blks180;
    const int nblk = dp ? (int)(sizeof(blks175) / sizeof(Blk)) : (int)(sizeof(blks180) / sizeof(Blk));
    for (int i = 0; i < nblk; ++i)
      std::fprintf(stderr, "   %-24s [%3d:%3d] %s\n", blks[i].n, blks[i].lo, blks[i].lo + blks[i].len,
                   S(obs.segment(blks[i].lo, blks[i].len)).c_str());
    std::fprintf(stderr, " ACTION(raw,Isaac)[31]: %s\n", S(action).c_str());
    std::fprintf(stderr, " Q_DES(SDK)[31]       : %s\n", S(q_sdk).c_str());
    std::fprintf(stderr, " KP(SDK)[31]          : %s\n", S(kp_sdk).c_str());
    std::fprintf(stderr, " KD(SDK)[31]          : %s\n", S(kd_sdk).c_str());
    if (state.q.size() == kNumJoints) {
      const Eigen::VectorXd e = (q_sdk - state.q).cwiseAbs();
      int wi = 0;
      e.maxCoeff(&wi);
      std::fprintf(stderr, " |q_des-q_meas| max=%.4f at %s (slot %d)\n", e[wi],
                   backend_joint_order()[wi].c_str(), wi);
    }
    std::fprintf(stderr,
        " NECK passive: slots[%d,%d] q=%.2f kp=%.1f kd=%.1f (model neck output dropped)\n",
        kHeadSlot0, kHeadSlot1, q_sdk[kHeadSlot0], kp_sdk[kHeadSlot0], kd_sdk[kHeadSlot0]);
    std::fprintf(stderr,
        " SAFETY: q_des clamped on %d/31 joint(s) | sec(torso) IMU=%d -> torso ori %s\n",
        last_clamp_count_, (int)state.has_secondary_imu,
        state.has_secondary_imu ? "from IMU" : "IDENTITY-FALLBACK(!)");
    std::fprintf(stderr,
        "=================================================================\n\n");
  }

  PpOnnxPolicy onnx_;
  PpPolicyConfig cfg_;
  std::atomic<int> level_;
  std::atomic<double> swing_speed_;
  std::atomic<int> swing_dir_;  // +1 forehand / -1 backhand (scripted; live f/b toggle)
  std::atomic<int> pending_swing_dir_{0};  // queued mid-swing dir flip (0 = none); see set_swing_dir
  bool last_tts_at_windup_ = true;  // last tick's clock sat at the windup start (safe flip point)
  int yaw_align_defer_ticks_ = 0;   // ticks spent waiting for upright+still before yaw capture
  std::atomic<bool> legs_passive_{false};   // hold legs at nominal (dyn: --auto-leg-hold flips by level)
  std::atomic<bool> waist_passive_{false};  // hold waist at nominal (dyn: --auto-leg-hold flips by level)
  double leg_clamp_rad_ = 0.0;              // clamp policy-driven leg q_des to nominal ± band (0=off)
  double leg_smooth_alpha_ = 1.0;           // EMA low-pass on released leg q_des (1=off)
  Eigen::VectorXd leg_qdes_smooth_;         // EMA state for the leg q_des low-pass (seeded to nominal)
  std::atomic<std::uint64_t> swing_clock_origin_{0};  // tick offset; reset on each level->1 entry
  std::uint64_t rest_rearm_tick_ = 0;                 // driver thread only
  std::atomic<bool> rest_rearm_armed_{false};         // cleared by any external set_level()
  // yaw-align state (see PpPolicyConfig::yaw_align / rearm_yaw_align)
  std::atomic<bool> yaw_align_pending_{true};
  Vec4 yaw0_base_inv_ = Vec4(1.0, 0.0, 0.0, 0.0);     // driver thread only
  Vec4 yaw0_torso_inv_ = Vec4(1.0, 0.0, 0.0, 0.0);    // driver thread only
  int swing_level_prev_ = 0;                          // ComputeCommand (driver thread) only
  int swing_dir_prev_ = 1;                            // detect f<->b switch -> restart swing at windup
  ClipLayout clip_;
  std::array<int, 31> isaac_to_sdk_{};
  Eigen::VectorXd nominal_q_sdk_;
  Eigen::VectorXd official_kp_sdk_;
  Eigen::VectorXd official_kd_sdk_;
  Eigen::VectorXd last_action_;
  int last_time_step_ = -1;
  Vec3 last_proj_grav_ = Vec3(0.0, 0.0, -1.0);
  Vec3 last_base_pos_ = Vec3(0.0, 0.0, 0.95);
  Vec4 last_base_quat_w_ = Vec4(1.0, 0.0, 0.0, 0.0);  // yaw-aligned; planner gate uses last tick's
  bool dbg_done_ = false;
  bool sec_imu_warned_ = false;  // one-shot warn when torso IMU is absent
  bool clamp_warned_ = false;    // one-shot warn when q_des hits a joint limit
  int last_clamp_count_ = 0;     // # joints clamped on the last tick
  std::uint64_t clamp_ticks_ = 0;                        // ticks the clamp ran
  std::array<std::uint64_t, kNumJoints> clamp_count_{};  // per-slot clamp hit count
  std::array<double, kNumJoints> clamp_max_viol_{};      // per-slot max out-of-range (rad)

  // --- localization / oracle (sim-only) + obs-debug state ---
  std::shared_ptr<PpOraclePose> oracle_;       // only used when loc_mode == kOracle
  bool oracle_fresh_ = false;
  std::uint64_t oracle_warn_tick_ = 0;  // repeat the stale-oracle warning every ~2 s (100 ticks)
  double oracle_age_s_ = -1.0;
  std::uint64_t sync_miss_ = 0;

  // --- LIVE PLANNER inputs + engage state (Path B; driver-thread only unless noted) ---
  std::shared_ptr<PpRacketTargetInput> racket_in_;  // written by AimRT subscriber thread
  std::shared_ptr<PpBasePoseInput> base_in_;        // written by AimRT subscriber thread
  bool base_fresh_ = false;             // a fresh external-base sample was used this tick
  std::uint64_t base_warn_tick_ = 0;    // repeat the stale-mocap warning every ~2 s
  std::uint64_t gate_warn_tick_ = 0;    // throttle the target-gate rejection detail print
  bool planner_engaged_ = false;        // a planner swing is active (frozen target in flight)
  bool planner_have_hold_ = false;      // at least one swing engaged (diagnostic)
  double planner_tts0_ = 0.0;           // engage-time tts, clamped to the clip windup length;
                                        // seeds the swing clock so the strike meets the ball
  Vec3 planner_frozen_pos_w_ = Vec3::Zero();
  // Hold/pre-engage target velocity: initialised in the ctor to the forehand box-center
  // vel (a ZERO vel target is outside every trained target-vel box = an obs state
  // training never saw); overwritten by each engage's frozen velocity.
  Vec3 planner_frozen_vel_w_ = Vec3::Zero();
  double planner_frozen_sign_ = 1.0;
  // base-rel target at engage (hold anchor); defaults = a centered, racket-reachable ready
  // stance so the pre-first-engage hold is safe even before any command arrives.
  Vec3 planner_hold_pos_b_engage_ = Vec3(0.40, 0.0, 0.0);
  double planner_hold_z_w_ = 0.90;
  // post-swing recovery clock + static-stand blend state (driver thread only)
  std::uint64_t planner_hold_start_tick_ = 0;
  bool planner_static_active_ = false;
  std::uint64_t planner_static_start_tick_ = 0;
  Eigen::VectorXd planner_static_q0_;
  mutable std::mutex planner_mu_;
  std::string planner_status_ = "init";
  mutable std::mutex obs_mu_;
  Eigen::VectorXd last_obs_;

  // --- live diagnostics (written in ComputeCommand, read by status thread) ---
  mutable std::mutex diag_mu_;
  Eigen::VectorXd last_q_des_, last_q_meas_, last_qd_meas_;
  Eigen::VectorXd des_lo_, des_hi_, meas_lo_, meas_hi_, err_peak_, qd_peak_;
  bool ranges_init_ = false;
};

}  // namespace a3_pingpong
