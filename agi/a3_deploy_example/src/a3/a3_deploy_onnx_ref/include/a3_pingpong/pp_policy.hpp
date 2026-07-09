// Ping-pong policy front-end as a drop-in A3PolicyDriver CommandFn. Per tick:
//   scripted/planner racket target -> reference clock -> ONNX refs ->
//   180/177/175-D obs (auto-selected from the model input dim) ->
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
  // ============== 110-D hitter_pure additions (2026-07-07, HITTER-paper deploy) =============
  // The 110 engage gate is METADATA-driven (per-clip z bands + station geometry from the
  // ONNX hitter_pure boxes), replacing the fixed base-relative box above. These bound the
  // remaining free parameters:
  double gate_station_step_max = 0.85;  // max |derived station − current base| xy (m); trained
                                        // stations span ±0.40 vs spawn and up to ~0.8 m between
                                        // consecutive swings (paper Fig. 4 goes to ±0.75-0.8)
  double gate_z_margin = 0.05;          // slack around the per-clip trained z band (m)
  // Per-clip trained VELOCITY box gate (2026-07-08, from the first rally-gate fall): the old
  // |v|<=3.5 speed cap accepted a planner demand of (0.9,+0.18,0.7) — vy 0.18 vs the trained
  // fh box [0.96,1.96] — and the swing executed on an out-of-distribution velocity command
  // (follow-through charged +0.57 m off-station; trained follow-through drift is 0.01-0.02 m).
  // Engage + mid-swing streaming now also require vel_w inside the per-clip
  // hitter_pure_vel_range_per_clip metadata box, per axis, +- this margin (m/s).
  // Raise via --vel-gate-margin if a venue's demanded returns sit just outside the box —
  // but read the REJECT(110) vel print first: a far-out demand is a planner mistuning
  // (delta_t_flight / target_land aim), not a gate problem.
  double gate_vel_margin = 0.30;
  // STREAM-until-contact (paper Fig. 3: the planner refines the prediction to ~0 error at
  // contact; the paper's WBC consumes the stream — there is NO lock-at-engage). While a swing
  // flies, same-side commands passing the band gate keep updating WHERE (pos/vel); WHEN stays
  // the engage-latched clip clock (training never varies tts mid-swing — the training analog
  // of streaming WHERE is racket.midswing_resample_prob, whose tts floor this mirrors).
  // ⚠ DEFAULT OFF (2026-07-08): the deployed baseline TRAINED with midswing_resample_prob
  // = 0.0 (HOPEPingPongHitterPure.yaml:187) — for it, every mid-swing target update is an
  // untrained obs transition, and streaming also moves the derived STATION mid-swing, which
  // even the training-side resample contract holds fixed. Enable with --stream-target ONLY
  // for a model actually trained with midswing_resample_prob > 0.
  bool stream_target = false;           // 110-D models only; other contracts keep the lock
  double stream_tts_floor_s = 0.30;     // freeze the target inside the last 0.3 s before strike
  // DEMO-ROBUSTNESS velocity mode (--vel-box-center, 2026-07-08): command the per-clip
  // TRAINED BOX-CENTER velocity (== the reference swing's strike velocity, the manifold
  // the policy is most robust on) instead of the planner's solved velocity. The planner
  // still owns WHERE (pos) and WHEN (tts); only the outgoing-shot aim precision is given
  // up (the return goes roughly where the human demo's returns went). Rationale: the
  // planner's physically-solvable velocities intersect the trained box only near its
  // low-z corner (rally-gate measurement: demanded vz 0.08-0.10 vs trained center 0.71),
  // and off-center vel commands erode the swing margin in the stricter AGI sim.
  bool vel_cmd_box_center = false;
  // ENGAGE HEADING GATE (2026-07-08, rally run-3 fall): training swings always START facing
  // ~+x (episode resets; reference clips yaw at most ±20° MID-swing and END back at ~0-6°),
  // but a divergent follow-through can leave the real robot 30-70° off heading, and the
  // 110-D obs are world-frame — an engage from a yawed stand is far outside the trained
  // start distribution (measured: engage at ~-30° yaw -> |act| 58, 2 m sprint, violent
  // fall). Refuse to engage while the (yaw-aligned) base heading is off by more than this;
  // status shows "yawed". Recovery = the operator re-stand ('s' -> square the robot -> 'm').
  double engage_yaw_max_deg = 20.0;
  // Stricter heading bound for the STATIC-stand handoff (rally run 5: a +17° handoff —
  // legal under the 20° engage gate — tipped ~3 s after the gains froze; the static stand
  // needs a genuinely square stance, while an engage merely needs a near-trained start).
  double static_handoff_yaw_max_deg = 10.0;
  // MOTION-ENTRY SETTLE (2026-07-08): run 3 engaged a leftover in-flight serve on the SAME
  // tick MOTION started (the robot was seconds off the stand-gain catch). Give the stand
  // this long before the first engage of a MOTION session.
  double engage_settle_s = 1.0;
  // 110-D LEVEL-0 station semantics (2026-07-07 fix): false (nominal) = the 177-style
  // fixed-world hold anchor — idle actively station-keeps, pulling the base back after every
  // follow-through so displacement can NOT accumulate across swings (the Gate-2.5 P7 creep).
  // true = the legacy Δ=0 idle (station := current base), kept ONLY for A/B: it let the robot
  // free-creep between swings (12200 P7 fall) and diverges outright for hold-trained rally
  // models (18000 P2 fall) — see the level-0 branch comment.
  bool idle_station_dzero_110 = false;
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
  // RE-SYNCED 2026-07-07 to the model_12200_hitterpure generation (HOPEPingPongHitterPure,
  // run 2026-07-07_13-28-13; the 110-D hitter_pure contract). hitter_pure fixes the striking
  // plane at x=0.70 RELATIVE to the station (both clips' blade reach ≈ 0.70) and samples only y/z:
  //   clip0 forehand: pos x[0.70,0.70] y[-0.65,-0.15] z[0.67,0.97]  vel x[1.05,2.05] y[ 0.96, 1.96] z[0.31,1.11]
  //   clip1 backhand: pos x[0.70,0.70] y[-0.05, 0.45] z[0.88,1.18]  vel x[1.61,2.61] y[-1.21,-0.21] z[0.00,0.71]
  // Targets sit at the BOX CENTERS = each clip's reference BLADE strike state (fh (0.70,-0.40,0.82)
  // vel (1.55,1.46,0.71); bh (0.70,0.20,1.03) vel (2.11,-0.71,0.36)); y-centers ≈ the ref reach
  // (fh −0.409 / bh +0.185). The prior model_9000 values (x-plane 0.68/0.66) were 2-4 cm off the
  // hitter_pure fixed 0.70 plane = a small OOD offset on the world racket_target_rel_base channel.
  // (The 110-D ENGAGE gate z/vel bands come from ONNX hitter_pure_* metadata, not these constants;
  // these feed only the SCRIPTED Gate 2/2.5 "ball".)
  // ⚠ model_9000 is a WALK-AND-STRIKE policy: with these world-fixed targets it turns ~84 deg and
  // displaces its base 0.4-0.65 m before contact (measured in the deploy-faithful MuJoCo gate).
  // That footwork only closes the loop when the localization source reports the REAL base motion
  // (sim: --oracle-pelvis; hardware: mocap). Under perfect_tracking the base obs stays pinned to
  // the reference pelvis and the strike loop runs OPEN — validate in sim with BOTH loc modes.
  // ⚠ DO NOT pull x below the box centers to reduce the scripted forward walk: tried
  // 2026-07-05 with model_19400_holdfix2 — x=0.58 (box edge) AND x=0.63 (half-way) BOTH
  // fail Gate 2.5 P3b (post-swing hold fall; the shallower strike leaves a follow-through
  // pose the policy never recovers from — post-swing recovery is only trained near the
  // reference blade point). Fixed-x-plane / y-only strikes need the retrain rider
  // (base-relative fixed-reach-x target sampling), not a runner-side constant.
  Vec3 racket_pos_w_clip[2] = {Vec3(0.70, -0.40, 0.82), Vec3(0.70, 0.20, 1.03)};
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
    // 177-D hitter_footwork: resolve the per-clip base-station reach offsets. The runner
    // derives the deploy-time base STATION from the racket target as
    //   station_xy = target_xy - reach_offset_xy[clip]
    // (training base_couple_mode=reference_reach: standing AT the station puts the racket
    // target at the clip's reference reach). Prefer the ONNX-baked ref_reach_offset_xy
    // metadata (exports since 2026-07-06); else compute from the baked refs at each clip's
    // strike frame (same arithmetic as training _ensure_reference_strike_state). The station
    // channel is the whole point of the 177 contract — refuse to run without it rather than
    // silently feeding a garbage station.
    if (onnx_.obs_dim() == kObsDim177) {
      if (onnx_.has_reach_offsets()) {
        if (onnx_.reach_offsets().size() < 2)
          throw std::runtime_error(
              "pingpong: 177 model's ref_reach_offset_xy metadata does not have 2 clips");
        reach_offset_clip_[0] = onnx_.reach_offsets()[0];
        reach_offset_clip_[1] = onnx_.reach_offsets()[1];
        std::fprintf(stderr,
            "[pp] 177 hitter: reach offsets from ONNX metadata: fh=(%+.3f,%+.3f) "
            "bh=(%+.3f,%+.3f)\n",
            reach_offset_clip_[0][0], reach_offset_clip_[0][1],
            reach_offset_clip_[1][0], reach_offset_clip_[1][1]);
      } else if (onnx_.has_clip_layout()) {
        for (int c = 0; c < 2; ++c)
          reach_offset_clip_[c] = onnx_.reach_offset_from_refs(clip_.strike_frame(c));
        std::fprintf(stderr,
            "[pp WARN] 177 hitter: ONNX lacks ref_reach_offset_xy metadata -> computed from "
            "the baked refs: fh=(%+.3f,%+.3f) bh=(%+.3f,%+.3f). Re-export with the patched "
            "exporter (scripts/export_onnx_hitter.sh) to bake it.\n",
            reach_offset_clip_[0][0], reach_offset_clip_[0][1],
            reach_offset_clip_[1][0], reach_offset_clip_[1][1]);
      } else {
        throw std::runtime_error(
            "pingpong: 177 hitter model without clip layout OR reach-offset metadata — "
            "cannot derive the base station; re-export with scripts/export_onnx_hitter.sh");
      }
    }
    // 110-D hitter_pure (2026-07-07): resolve the per-side station geometry from the baked
    // sampling boxes — station_xy = target_xy − (plane_x, y_band_center)[side] (the paper's
    // §V-B-3 heuristic computes p̂_base downstream of the ball planner; here = the runner).
    // The per-clip z bands also drive the engage gate. Preference order: hitter_pure box
    // metadata (exports via scripts/export_onnx_hitter_pure.sh) → ref_reach_offset_xy
    // (numerically ≈ the box centers by construction: fh (0.699,−0.409) / bh (0.706,+0.185)
    // vs box (0.70,−0.40)/(0.70,+0.20)) → refs-FK fallback. Refuse to run blind.
    if (onnx_.obs_dim() == kObsDim110) {
      if (onnx_.has_hitter_pure_boxes() && onnx_.hp_pos_boxes().size() >= 2) {
        for (int c = 0; c < 2; ++c) {
          const auto& b = onnx_.hp_pos_boxes()[c];  // {x_lo,x_hi,y_lo,y_hi,z_lo,z_hi}
          reach_offset_clip_[c] = Vec2(b[0], 0.5 * (b[2] + b[3]));
          hp_y_band_[c] = Vec2(b[2], b[3]);
          hp_z_band_[c] = Vec2(b[4], b[5]);
        }
        if (onnx_.hp_vel_boxes().size() >= 2) {  // per-clip trained vel box -> engage/stream gate
          hp_vel_box_[0] = onnx_.hp_vel_boxes()[0];
          hp_vel_box_[1] = onnx_.hp_vel_boxes()[1];
          hp_vel_box_set_ = true;
        }
        std::fprintf(stderr,
            "[pp] 110 hitter_pure: station geometry from ONNX boxes: plane_x=%.2f "
            "fh y[%.2f,%.2f] z[%.2f,%.2f]  bh y[%.2f,%.2f] z[%.2f,%.2f]\n",
            reach_offset_clip_[0][0], hp_y_band_[0][0], hp_y_band_[0][1], hp_z_band_[0][0],
            hp_z_band_[0][1], hp_y_band_[1][0], hp_y_band_[1][1], hp_z_band_[1][0],
            hp_z_band_[1][1]);
      } else if (onnx_.has_reach_offsets() && onnx_.reach_offsets().size() >= 2) {
        reach_offset_clip_[0] = onnx_.reach_offsets()[0];
        reach_offset_clip_[1] = onnx_.reach_offsets()[1];
        for (int c = 0; c < 2; ++c) {
          hp_y_band_[c] = Vec2(reach_offset_clip_[c][1] - 0.25, reach_offset_clip_[c][1] + 0.25);
          hp_z_band_[c] = Vec2(cfg.gate_z_lo, cfg.gate_z_hi);
        }
        std::fprintf(stderr,
            "[pp WARN] 110 hitter_pure: ONNX lacks hitter_pure box metadata -> station from "
            "ref_reach_offset_xy fh=(%+.3f,%+.3f) bh=(%+.3f,%+.3f), WIDE z gate. Re-export "
            "with scripts/export_onnx_hitter_pure.sh to bake the trained boxes.\n",
            reach_offset_clip_[0][0], reach_offset_clip_[0][1], reach_offset_clip_[1][0],
            reach_offset_clip_[1][1]);
      } else if (onnx_.has_clip_layout()) {
        for (int c = 0; c < 2; ++c) {
          reach_offset_clip_[c] = onnx_.reach_offset_from_refs(clip_.strike_frame(c));
          hp_y_band_[c] = Vec2(reach_offset_clip_[c][1] - 0.25, reach_offset_clip_[c][1] + 0.25);
          hp_z_band_[c] = Vec2(cfg.gate_z_lo, cfg.gate_z_hi);
        }
        std::fprintf(stderr,
            "[pp WARN] 110 hitter_pure: no box/reach metadata -> refs-FK fallback "
            "fh=(%+.3f,%+.3f) bh=(%+.3f,%+.3f). Re-export to bake the trained boxes.\n",
            reach_offset_clip_[0][0], reach_offset_clip_[0][1], reach_offset_clip_[1][0],
            reach_offset_clip_[1][1]);
      } else {
        throw std::runtime_error(
            "pingpong: 110 hitter_pure model without box/reach/clip metadata — cannot derive "
            "the base station; re-export with scripts/export_onnx_hitter_pure.sh");
      }
      // Idle-hold seeds on the trained manifold: ready racket at the fixed plane in front of
      // the fh band center, at the fh band-center height (hitter_pure trains NO hold — idle
      // must look like 'standing at station, next target at comfortable reach, tts pinned').
      planner_hold_pos_b_engage_ =
          Vec3(reach_offset_clip_[0][0], reach_offset_clip_[0][1], 0.0);
      planner_hold_z_w_ = 0.5 * (hp_z_band_[0][0] + hp_z_band_[0][1]);
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
  // may have been turned/moved between engagements). Also drops the 177-D hold-station
  // anchor so it re-captures at the robot's NEW spot (a stale anchor from before the
  // move would command a walk back to the old position).
  void rearm_yaw_align() {
    yaw_align_pending_.store(true);
    hold_station_set_ = false;
    arm_quiet_ticks_ = 0;    // fresh MOTION entry: restart the arm-hold sustained-quiet clock
    arm_hold_armed_ = true;  // ...and re-arm the pre-swing arm hold
    // PLANNER-MODE swing-state reset (2026-07-08): SHADOW and MOTION run ComputeCommand on
    // DIFFERENT clock domains (SHADOW = a free-running local counter, MOTION = the publish-
    // gated driver tick). A swing engaged during a SHADOW preview leaves level 1 + a
    // swing_clock_origin_ from the SHADOW clock; entering MOTION then resumes that phantom
    // swing against an incoherent clock (tts frozen or deeply negative — engage locked out
    // or an instant snap). Every SHADOW/MOTION entry from PASSIVE/PD_STAND must start from
    // a clean stand: level 0, no engage latch, no stale rest timer. The next valid planner
    // command re-engages normally. (Scripted mode is untouched — the operator owns level.)
    if (cfg_.planner_mode) {
      pending_swing_dir_.store(0);
      rest_rearm_armed_.store(false);
      level_.store(0);
      planner_engaged_ = false;
      planner_static_active_ = false;
      planner_entry_pending_.store(true);  // restart the engage settle clock (engage_settle_s)
    }
  }

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
    if (cfg_.planner_mode && planner_entry_pending_.exchange(false))
      planner_entry_tick_ = tick_idx;  // first tick of this SHADOW/MOTION session (settle clock)
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
    // ANY 1->0 edge restarts the planner post-swing recovery clock, not just the normal
    // completion path (which also sets it, idempotently). Without this, an EXTERNAL
    // set_level(0) mid-swing (squat/tilt guard, operator key) leaves the clock stale from
    // the PREVIOUS hold — post_recovery reads instantly true and the stiff static stand
    // freezes onto a tilted, still-moving robot, skipping the policy recovery entirely.
    if (cfg_.planner_mode && swing_lvl_now == 0 && swing_level_prev_ == 1)
      planner_hold_start_tick_ = tick_idx;
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

    PpRefs refs = onnx_.refs(time_step);
    // HOLD = a STATIONARY reference (2026-07-05, train==deploy lockstep): clip frame 0
    // is a mid-crouch TRANSIENT (knee +7.8 rad/s, torso -1.11 m/s down) — feeding its
    // raw velocities through the whole hold taught the policy to fight a phantom squat
    // (the Gate 2.5 P2 3-5 s bare-hold tip). Training now zeroes the reference joint
    // velocities on held envs (commands.py joint_vel); mirror it in every policy-hold
    // state (level 0 = scripted hold AND the planner post-swing recovery hold).
    if (level_.load() == 0) {
      refs.joint_vel.setZero();
      // ...and the hold JOINT reference is the READY STAND, not the windup crouch
      // (2026-07-05 lockstep with training commands.joint_pos): frame 0 is an
      // asymmetric mid-crouch — imitating it during hold produced the splayed-feet
      // crouch-stand. The release into the swing is the trained stand_start regime.
      refs.joint_pos = onnx_.default_q();
    }

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
      // NEAR-STATION guard (2026-07-08, from the rally-gate fall): never hand the stiff
      // STATIC stand a robot that is parked far off its hold station — the walked stance
      // is staggered/leaning and the official gains freeze it there (measured: forced
      // switch at 0.83 m off-station -> tip). Off-station, the POLICY hold keeps actively
      // walking home; the switch waits until it arrives. No-op when no anchor is set
      // (dropout / pre-engage: near_station true -> legacy behavior).
      const bool near_station = !hold_station_set_ ||
          (Vec2(st.base_pos_w[0], st.base_pos_w[1]) - hold_station_w_).norm() < 0.3;
      // ...and NEAR-HEADING (2026-07-08 rally run 4): the backhand follow-through can leave
      // the robot yawed 35-55° (execution over-rotation; the reference ends at ~0°). The
      // static stand FROZE that yawed/staggered stance and it tipped seconds later, while
      // the POLICY hold both balances actively and — being the trained pre-strike state
      // (which always faces +x in training) — is the only thing in the chain with a
      // heading-restoring feedback loop. Off-heading: stay on the policy hold (g25-proven
      // to 20 s); the engage heading gate keeps swings blocked until square.
      const double hold_yaw = std::atan2(
          2.0 * (st.base_quat_w[0] * st.base_quat_w[3] +
                 st.base_quat_w[1] * st.base_quat_w[2]),
          1.0 - 2.0 * (st.base_quat_w[2] * st.base_quat_w[2] +
                       st.base_quat_w[3] * st.base_quat_w[3]));
      const bool near_heading =
          std::fabs(hold_yaw) < cfg_.static_handoff_yaw_max_deg * M_PI / 180.0;
      const bool post_recovery = planner_have_hold_ && near_station && near_heading &&
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
          // The policy is out of control from here until the next engage: zero the
          // last-action obs so the engage's first policy tick reads a training-style
          // reset (stand start, zero prev action) instead of a seconds-stale action.
          last_action_.setZero();
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
      } else if (onnx_.obs_dim() == kObsDim110) {
        // 110 hitter_pure idle (2026-07-08 fix, from the first rally-gate fall): the hold
        // target must be WORLD-FIXED at the hold-station anchor — the same obs family as
        // the Gate-2.5-proven scripted hold (world-fixed box-center target + box-center
        // vel; P2 held 20 s on it). The first design anchored it to the LIVE base (0.70
        // ahead of wherever the robot is, re-anchored per tick): a moving carrot with NO
        // positional feedback — racket_target_rel_base never closes however far the robot
        // walks, and the rally gate measured the policy hold charging +0.83 m off-station
        // in ~1 s on it (then the forced static handoff tipped from the walked stance).
        // Anchoring at the station makes a forward drift SHORTEN the observed reach, so
        // the trained pre-strike response pulls the robot back. Geometry is PER-SIDE
        // (last swing side = the side the hold tts clamp already assumes): plane_x +
        // y-band center at the anchor, z-band mid height, trained box-center velocity
        // (the frozen streamed vel could be out-of-band — it is what the LAST swing flew).
        // Offset in WORLD axes, deliberately NOT rotated by the base yaw (2026-07-08 review):
        // training's hold target is world-fixed (station + the +x plane offset in WORLD),
        // and the engage-side station derivation already uses world-frame reach offsets.
        // Rotating by the live yaw would make the target ORBIT the station while a yawed
        // robot re-squares — an obs the (rally2-)trained recovery never sees.
        const int hc = clip_id_from_swing_sign(swing_dir_.load() >= 0 ? 1.0 : -1.0);
        const Vec2 anchor_xy = hold_station_set_
            ? hold_station_w_
            : Vec2(st.base_pos_w[0], st.base_pos_w[1]);  // pre-anchor tick / dropout
        tg.pos_w = Vec3(anchor_xy[0] + reach_offset_clip_[hc][0],
                        anchor_xy[1] + reach_offset_clip_[hc][1],
                        0.5 * (hp_z_band_[hc][0] + hp_z_band_[hc][1]));
        tg.vel_w = cfg_.racket_vel_w_clip[hc];
      } else {                  // idle/rest (incl. before the first engage) -> base-anchored hold
        const Vec4 base_yaw = yaw_quat(st.base_quat_w);
        Vec3 hb(cfg_.hold_anchor_x_b, planner_hold_pos_b_engage_[1], 0.0);
        tg.pos_w = st.base_pos_w + quat_rotate(base_yaw, hb);
        tg.pos_w[2] = planner_hold_z_w_;
        tg.vel_w = planner_frozen_vel_w_;
      }
    }

    last_proj_grav_ = projected_gravity_body(st.base_quat_w);

    // 177-D hitter_footwork base-station channel (base_target_pos_b = yaw-frame Δxy from the
    // current base to the commanded station). During a swing the station rides the SAME reach
    // point the swing uses (scripted box center or frozen planner target) minus the per-clip
    // reference reach — training's base_couple_mode=reference_reach coupling. During level-0
    // holds the station is a FIXED WORLD ANCHOR (captured at hold entry / carried over from
    // the completed swing): the live Δ to that anchor is the policy's balance signal —
    // training pays pbase through every hold, so the policy leans on this channel to stay
    // put. Feeding Δ=0 through holds was the first design and is WRONG as the nominal path:
    // it removes the only anchor and the policy free-wanders meters during holds, then falls
    // off-station (2026-07-06 MuJoCo deploy-faithful CSV phase analysis: falls at |torso|
    // 1-2 m with ±0.1 m stations; live-station holds: model_17400 0 falls x 3 seeds).
    // Δ=0 remains ONLY the localization-dropout fallback (perfect_tracking / fabricated /
    // stale mocap/oracle), where any nonzero Δ would be fictional and chased open-loop.
    // 2026-07-07: the fixed-world anchor now applies to 110-D hitter_pure TOO (it was Δ=0
    // at idle — see the idle_station_dzero_110 branch below for the refuting evidence).
    if (onnx_.obs_dim() == kObsDim177 || onnx_.obs_dim() == kObsDim110) {
      const bool base_real =
          (cfg_.loc_mode == LocMode::kOracle && oracle_fresh_) ||
          (cfg_.loc_mode == LocMode::kExternalBase && base_fresh_);
      if (!base_real) {
        tg.base_target_xy = Vec2(st.base_pos_w[0], st.base_pos_w[1]);  // dropout: Δ=0
        hold_station_set_ = false;  // re-anchor at the CURRENT spot when localization returns
      } else if (level_.load() == 1) {
        const int c = clip_id_from_swing_sign(tg.swing_sign);
        tg.base_target_xy = Vec2(tg.pos_w[0], tg.pos_w[1]) - reach_offset_clip_[c];
        hold_station_w_ = tg.base_target_xy;  // post-swing hold recovers AT the strike station
        hold_station_set_ = true;
      } else if (onnx_.obs_dim() == kObsDim110 && cfg_.idle_station_dzero_110) {
        // LEGACY 110 idle: Δ=0 (station := current base). First design, justified as
        // "hitter_pure trains NO hold so idle never demands station-keeping" — REFUTED by the
        // 2026-07-07 Gate-2.5 evidence: with Δ=0 there is NO pull-back between swings, so the
        // follow-through displacement ACCUMULATES across cycles against the world-fixed
        // scripted target (the model_12200 P7 fall), and a hold-TRAINED rally model
        // (model_18000) outright DIVERGES in it — walked +0.94 m THROUGH the target
        // (racket_rel_base x +0.69 -> -0.27 while base_target_dxy pinned 0), yawed ~70°, obs
        // blow-up, fell 12 s into the P2 hold. Training-side truth: hitter_pure stations are
        // x ±0.10 / y ±0.40 with drift 0.01-0.02 m/swing — the policy NEVER trains forward
        // locomotion; an unanchored idle walks it straight out of distribution (the observed
        // pigeon-toed creep). Kept ONLY as a compile-time A/B fallback; the nominal 110 path
        // is the 177-style fixed-world anchor below (idle at an anchor == "pre-strike at the
        // station", which hitter_pure trains every swing).
        tg.base_target_xy = Vec2(st.base_pos_w[0], st.base_pos_w[1]);
      } else {
        if (!hold_station_set_) {  // fresh hold (pre-first-engage / after re-localization)
          hold_station_w_ = Vec2(st.base_pos_w[0], st.base_pos_w[1]);
          hold_station_set_ = true;
        }
        tg.base_target_xy = hold_station_w_;
      }
    }

    // 175-D deploy_parity vs 177-D hitter_footwork vs 180-D full (model_15200) vs 110-D
    // hitter_pure. Auto-selected from the loaded ONNX input dim. build_obs_175 drops
    // motion_anchor_pos_b + base_target_pos_b and reframes the racket target relative to the
    // CURRENT racket FK (pp_racket_fk.hpp) — no world base pos. build_obs_177 = the 175 layout
    // + base_target_pos_b(2) re-inserted (above). build_obs_110 = HITTER Table-I exact: NO
    // reference stream/swing_type, WORLD-frame deltas + e_base,x (refs never enter the obs —
    // the clip clock above only schedules tts and the graph's time_step input).
    const Eigen::VectorXd obs = (onnx_.obs_dim() == kObsDim110)
        ? build_obs_110(st, tg, last_action_, onnx_.default_q())
        : (onnx_.obs_dim() == kObsDim175)
        ? build_obs_175(refs, st, tg, last_action_, onnx_.default_q(), cfg_.use_imu_yaw_for_targets)
        : (onnx_.obs_dim() == kObsDim177)
        ? build_obs_177(refs, st, tg, last_action_, onnx_.default_q(), cfg_.use_imu_yaw_for_targets)
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
    if (level_.load() == 1) {  // in flight
      // 110-D STREAMING (paper Fig. 3): keep consuming same-side refinements while the swing
      // flies. Every other contract keeps the proven frozen-target behavior.
      if (onnx_.obs_dim() == kObsDim110 && cfg_.stream_target) StreamTargetStep_(tick_idx);
      set_planner_status_("swinging");
      return;
    }
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
    // Late gate. 110: PER-CLIP (the backhand windup 0.87 s < the legacy 1.0 s constant —
    // a scalar gate would make backhand unreachable under the wait-for-tts semantics below);
    // side is not chosen yet, so gate on the LOOSER (SHORTER-windup) clip here and re-check
    // per-clip after side selection. ⚠ 2026-07-08 fix: this used windup_MAX, which made the
    // pre-side cutoff min(1.0, 0.9*1.30)=1.0 s — ABOVE the backhand's whole engage window
    // [0.9*0.87, 0.87] = [0.78, 0.87] s, so every backhand serve died here as too_late and
    // the side-specific gate below never ran (backhand mathematically unreachable in planner
    // mode). The pre-side cutoff must be the MIN of the per-clip cutoffs. Legacy contracts
    // keep the scalar behavior unchanged.
    if (onnx_.obs_dim() == kObsDim110) {
      const double windup_min = std::min(
          (clip_.strike_frame(0) - clip_.seg_start(0)) * clip_.step_dt,
          (clip_.strike_frame(1) - clip_.seg_start(1)) * clip_.step_dt);
      if (tts < std::min(cfg_.engage_min_tts_s, 0.9 * windup_min)) {
        set_planner_status_("too_late"); return;
      }
    } else if (tts < cfg_.engage_min_tts_s) { set_planner_status_("too_late"); return; }

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

    // MOTION-entry settle: no engage for the first engage_settle_s of a session (see cfg).
    if (tick_idx < planner_entry_tick_ +
                       static_cast<std::uint64_t>(cfg_.engage_settle_s / std::max(cfg_.dt, 1e-6))) {
      set_planner_status_("settling"); return;
    }
    // HEADING gate (see cfg.engage_yaw_max_deg): last_base_quat_w_ is yaw-aligned, so its yaw
    // is the drift from the engage heading (~ world +x). Swinging from a yawed stand is OOD.
    {
      const double yaw = std::atan2(
          2.0 * (last_base_quat_w_[0] * last_base_quat_w_[3] +
                 last_base_quat_w_[1] * last_base_quat_w_[2]),
          1.0 - 2.0 * (last_base_quat_w_[2] * last_base_quat_w_[2] +
                       last_base_quat_w_[3] * last_base_quat_w_[3]));
      if (std::fabs(yaw) > cfg_.engage_yaw_max_deg * M_PI / 180.0) {
        if ((gate_warn_tick_++ % 100) == 0)
          std::fprintf(stderr,
              "[pp gate] REJECT yawed: base heading %+.0f deg off the engage heading "
              "(max %.0f) — swings must start square; re-stand ('s', square, 'm') if this "
              "persists\n", yaw * 180.0 / M_PI, cfg_.engage_yaw_max_deg);
        set_planner_status_("yawed"); return;
      }
    }

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

    // Swing side. 110-D hitter_pure: the paper's §V-B-3 heuristic, implemented as
    // NEAREST-STATION — candidate station per side = target_xy − (plane_x, band_center_y),
    // pick the side needing the smaller step. The legacy y<0 split is WRONG for the pure
    // bands (the bh band [−0.05,0.45] crosses y=0: a bh-region ball at station-rel y ∈
    // [−0.10,0) would grab the fh clip + a ~0.6 m wrong station). Legacy contracts keep
    // the y-sign split.
    double sign;
    if (onnx_.obs_dim() == kObsDim110) {
      const Vec2 tgt_xy(pos_w[0], pos_w[1]);
      const Vec2 base_xy(base_pos[0], base_pos[1]);
      const double d_fh = (tgt_xy - reach_offset_clip_[0] - base_xy).norm();
      const double d_bh = (tgt_xy - reach_offset_clip_[1] - base_xy).norm();
      sign = (d_fh <= d_bh) ? 1.0 : -1.0;
    } else {
      // BASE-RELATIVE y (raw world-y is always <0 in the table frame).
      sign = swing_sign_from_target_y(tgt_b[1]);
    }
    const int eng_clip = clip_id_from_swing_sign(sign);
    const double max_tts0 =
        (clip_.strike_frame(eng_clip) - clip_.seg_start(eng_clip)) * clip_.step_dt;

    if (cfg_.target_gate_enable) {
      bool ok;
      if (onnx_.obs_dim() == kObsDim110) {
        // METADATA-driven gate against the TRAINED distribution: per-clip z band, required
        // station step, speed cap. No fixed base-relative box — the paper's robot WALKS to
        // targets the arm alone cannot cover (Fig. 4), so reachability is a station question.
        const Vec2 station =
            Vec2(pos_w[0], pos_w[1]) - reach_offset_clip_[eng_clip];
        const double step = (station - Vec2(base_pos[0], base_pos[1])).norm();
        // Per-clip trained VELOCITY box (metadata), per axis ± gate_vel_margin. A demand
        // outside it is an obs the policy never trained a swing for — the 2026-07-08 rally
        // fall executed vy=+0.18 against the fh box [0.96,1.96] and charged off-station.
        // Moot under --vel-box-center (the demand is replaced by the box center anyway).
        const bool vel_ok = cfg_.vel_cmd_box_center || !hp_vel_box_set_ ||
                            vel_in_hp_box_(eng_clip, vel_w);
        ok = pos_w[2] >= hp_z_band_[eng_clip][0] - cfg_.gate_z_margin &&
             pos_w[2] <= hp_z_band_[eng_clip][1] + cfg_.gate_z_margin &&
             step <= cfg_.gate_station_step_max &&
             vel_w.norm() <= cfg_.gate_speed_max && vel_ok;
        if (!ok && (gate_warn_tick_++ % 50) == 0) {
          const auto& vb = hp_vel_box_[eng_clip];
          std::fprintf(stderr,
              "[pp gate] REJECT(110) %s z_w=%.2f (band[%.2f,%.2f]±%.2f) station_step=%.2f "
              "(<=%.2f) |v|=%.2f (<=%.2f) vel=(%+.2f,%+.2f,%+.2f)%s tts=%.2f\n",
              sign > 0 ? "fh" : "bh", pos_w[2], hp_z_band_[eng_clip][0],
              hp_z_band_[eng_clip][1], cfg_.gate_z_margin, step, cfg_.gate_station_step_max,
              vel_w.norm(), cfg_.gate_speed_max, vel_w[0], vel_w[1], vel_w[2],
              vel_ok ? ""
                     : (hp_vel_box_set_
                            ? " OUT-OF-BAND"
                            : ""),
              tts);
          if (!vel_ok)
            std::fprintf(stderr,
                "[pp gate]   trained vel box (clip %d): x[%.2f,%.2f] y[%.2f,%.2f] "
                "z[%.2f,%.2f] ±%.2f — planner demand out of the trained envelope; retune "
                "the planner (delta_t_flight / target_land aim), or --vel-gate-margin\n",
                eng_clip, vb[0], vb[1], vb[2], vb[3], vb[4], vb[5], cfg_.gate_vel_margin);
        }
      } else {
        ok = tgt_b[0] >= cfg_.gate_x_lo && tgt_b[0] <= cfg_.gate_x_hi &&
             std::abs(tgt_b[1]) <= cfg_.gate_y_abs &&
             pos_w[2] >= cfg_.gate_z_lo && pos_w[2] <= cfg_.gate_z_hi &&
             vel_w.norm() <= cfg_.gate_speed_max;
        if (!ok && (gate_warn_tick_++ % 50) == 0) {
          // Throttled detail print (mirrors the Python runner's gate warn): without the
          // inputs a rejection is undebuggable at the venue.
          std::fprintf(stderr,
              "[pp gate] REJECT base-rel (%+.2f,%+.2f) z_w=%.2f |v|=%.2f tts=%.2f "
              "(need x[%.2f,%.2f] |y|<=%.2f z[%.2f,%.2f] v<=%.2f)\n",
              tgt_b[0], tgt_b[1], pos_w[2], vel_w.norm(), tts,
              cfg_.gate_x_lo, cfg_.gate_x_hi, cfg_.gate_y_abs,
              cfg_.gate_z_lo, cfg_.gate_z_hi, cfg_.gate_speed_max);
        }
      }
      if (!ok) { set_planner_status_("target_gate"); return; }
    }

    // Strike-time alignment. 110: WAIT-until-tts (paper: the hit time comes from the
    // virtual-plane crossing and the strike fires when the ball arrives). The legacy clamp
    // planner_tts0_ = min(tts, max_tts0) starts the clip early and lets the strike frame
    // fire (planner_tts − max_tts0) seconds BEFORE the ball (bh: >1 s early on a slow lob
    // = multi-decimeter miss). Wait at ready until the decaying tts enters the windup
    // window, then engage with the strike frame exactly on the predicted arrival. Per-clip
    // late gate re-check (side is now known).
    if (onnx_.obs_dim() == kObsDim110) {
      if (tts < std::min(cfg_.engage_min_tts_s, 0.9 * max_tts0)) {
        set_planner_status_("too_late"); return;
      }
      if (tts > max_tts0) { set_planner_status_("waiting_tts"); return; }
      planner_tts0_ = tts;
    } else {
      // ENGAGE: tts0 stored CLAMPED to the clip's windup length; DRIVES the swing clock
      // (ScriptedTarget planner branch: tts = tts0 - t). Mirrors wbc_runner's
      // `"tts0": min(tts, max_tts0)`; without the transfer every strike would be late by
      // (max_tts - planner_tts).
      planner_tts0_ = std::min(tts, max_tts0);
    }
    planner_frozen_pos_w_ = pos_w;
    planner_frozen_vel_w_ = (onnx_.obs_dim() == kObsDim110 && cfg_.vel_cmd_box_center)
                                ? cfg_.racket_vel_w_clip[eng_clip]
                                : vel_w;
    planner_frozen_sign_ = sign;
    planner_hold_pos_b_engage_ = tgt_b;
    planner_hold_z_w_ = pos_w[2];
    planner_have_hold_ = true;
    planner_engaged_ = true;
    set_swing_dir(sign >= 0.0 ? 1 : -1);
    set_level(1);
    std::fprintf(stderr,
        "[pp engage] %s %s: tgt base-rel (%+.2f,%+.2f,%+.2f) tts=%.2fs (clock tts0=%.2fs)\n",
        sign > 0 ? "forehand" : "backhand",
        (onnx_.obs_dim() == kObsDim110 && cfg_.stream_target) ? "engaged (streaming)" : "locked",
        tgt_b[0], tgt_b[1], tgt_b[2], tts, planner_tts0_);
    set_planner_status_("engage");
  }

  // 110-D stream-until-contact (paper Fig. 3: the planner's prediction error converges to ~0
  // at contact and the WBC consumes the stream — there is no lock-at-engage in HITTER).
  // Refresh WHERE (pos/vel) from the latest valid command while the swing flies; the side and
  // the swing clock (WHEN) stay engage-latched (training never varies tts mid-swing). Guards:
  // fresh+valid, same side under the nearest-station heuristic (a planner re-side mid-swing
  // is ignored), locked-side band membership, speed cap, and a tts floor mirroring training's
  // midswing_resample_tts_floor so the final approach is not perturbed.
  void StreamTargetStep_(std::uint64_t tick_idx) {
    if (!racket_in_) return;
    const auto snap = racket_in_->Latest();
    if (!snap.has_valid || snap.invalid_after) return;
    if (snap.valid_age_s > cfg_.command_timeout_s) return;
    const std::uint64_t origin = swing_clock_origin_.load();
    const double t = (tick_idx >= origin ? tick_idx - origin : 0) * cfg_.dt * swing_speed_.load();
    if (planner_tts0_ - t < cfg_.stream_tts_floor_s) return;  // freeze near the strike
    Vec3 pos_w = snap.cmd.pos_w;
    Vec3 vel_w = snap.cmd.vel_w;
    if (snap.cmd.frame_code == 1) {
      const Vec4 base_yaw = yaw_quat(last_base_quat_w_);
      pos_w = last_base_pos_ + quat_rotate(base_yaw, snap.cmd.pos_w);
      vel_w = quat_rotate(base_yaw, snap.cmd.vel_w);
    }
    const int c = clip_id_from_swing_sign(planner_frozen_sign_);
    const Vec2 tgt_xy(pos_w[0], pos_w[1]);
    const Vec2 base_xy(last_base_pos_[0], last_base_pos_[1]);
    if ((tgt_xy - reach_offset_clip_[1 - c] - base_xy).norm() <
        (tgt_xy - reach_offset_clip_[c] - base_xy).norm())
      return;  // nearest-station now prefers the OTHER side: keep the locked target
    if (pos_w[2] < hp_z_band_[c][0] - cfg_.gate_z_margin ||
        pos_w[2] > hp_z_band_[c][1] + cfg_.gate_z_margin)
      return;
    if (vel_w.norm() > cfg_.gate_speed_max) return;
    // Same trained-vel-box membership as engage: a mid-swing refinement must not drag the
    // velocity command out of the trained envelope (keep the engage-gated value instead).
    if (!cfg_.vel_cmd_box_center && hp_vel_box_set_ && !vel_in_hp_box_(c, vel_w)) return;
    planner_frozen_pos_w_ = pos_w;
    if (!cfg_.vel_cmd_box_center) planner_frozen_vel_w_ = vel_w;  // box-center mode: vel stays pinned
  }

  // vel_w inside the per-clip trained hitter_pure velocity box, per axis ± gate_vel_margin.
  bool vel_in_hp_box_(int clip, const Vec3& v) const {
    const auto& b = hp_vel_box_[clip];
    const double m = cfg_.gate_vel_margin;
    return v[0] >= b[0] - m && v[0] <= b[1] + m &&
           v[1] >= b[2] - m && v[1] <= b[3] + m &&
           v[2] >= b[4] - m && v[2] <= b[5] + m;
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
    // hitter_footwork 177-D: the 175 layout + base_target_pos_b(2) station Δxy re-inserted
    // after projected_gravity; everything after it shifts up 2.
    static const Blk blks177[] = {
        {"command", 0, 62}, {"motion_anchor_ori_b", 62, 6}, {"base_ang_vel", 68, 3},
        {"joint_pos_rel", 71, 31}, {"joint_vel", 102, 31}, {"actions(last)", 133, 31},
        {"projected_gravity", 164, 3}, {"base_target_pos_b", 167, 2},
        {"racket_target_pos_b(relFK)", 169, 3}, {"racket_target_vel_w", 172, 3},
        {"time_to_strike", 175, 1}, {"swing_type", 176, 1}};
    // hitter_pure 110-D (HITTER Table-I exact): no reference stream, no swing_type;
    // world-frame deltas + e_base,x. Matches training contract `hitter_pure`.
    static const Blk blks110[] = {
        {"base_ang_vel", 0, 3}, {"joint_pos_rel", 3, 31}, {"joint_vel", 34, 31},
        {"actions(last)", 65, 31}, {"projected_gravity", 96, 3}, {"base_forward_xy", 99, 2},
        {"base_target_delta_xy(world)", 101, 2}, {"racket_target_rel_base(world)", 103, 3},
        {"racket_target_vel_w", 106, 3}, {"time_to_strike", 109, 1}};
    std::fprintf(stderr, " OBS blocks (%d-D):\n", (int)obs.size());
    const Blk* blks = (obs.size() == kObsDim175) ? blks175
                    : (obs.size() == kObsDim177) ? blks177
                    : (obs.size() == kObsDim110) ? blks110
                                                 : blks180;
    const int nblk = (obs.size() == kObsDim175) ? (int)(sizeof(blks175) / sizeof(Blk))
                   : (obs.size() == kObsDim177) ? (int)(sizeof(blks177) / sizeof(Blk))
                   : (obs.size() == kObsDim110) ? (int)(sizeof(blks110) / sizeof(Blk))
                                                : (int)(sizeof(blks180) / sizeof(Blk));
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
  // 177-D hitter_footwork + 110-D hitter_pure: per-clip station geometry.
  // station_xy = racket_target_xy - reach_offset_clip_[clip]. 177: reference base->racket
  // reach at the strike frame (ONNX metadata or refs fallback). 110: (fixed_plane_x,
  // y_band_center) from the baked hitter_pure sampling boxes (≈ the same numbers by
  // construction). Zero for 175/180 models (never read there).
  Vec2 reach_offset_clip_[2] = {Vec2::Zero(), Vec2::Zero()};
  // 110-D hitter_pure only: trained per-clip target bands (engage gate + streaming gate).
  // Defaults = the legacy shared gate; overwritten from ONNX metadata in the ctor.
  Vec2 hp_y_band_[2] = {Vec2(-0.65, -0.15), Vec2(-0.05, 0.45)};
  Vec2 hp_z_band_[2] = {Vec2(0.55, 1.40), Vec2(0.55, 1.40)};
  // Per-clip trained velocity boxes {x_lo,x_hi,y_lo,y_hi,z_lo,z_hi} from the ONNX
  // hitter_pure_vel_range_per_clip metadata; gate engage + streaming when set.
  std::array<double, 6> hp_vel_box_[2] = {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
  bool hp_vel_box_set_ = false;
  // 177-D hold-station anchor (driver thread only): the fixed WORLD station fed to the
  // base_target obs during level-0 holds (captured at hold entry; carried from the last
  // swing's station after a completed swing). Cleared on localization dropout and by
  // rearm_yaw_align() (mode re-entry — the robot may have been carried/moved).
  Vec2 hold_station_w_ = Vec2::Zero();
  bool hold_station_set_ = false;
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
  // MOTION/SHADOW session start (engage settle clock); pending set by rearm_yaw_align.
  std::uint64_t planner_entry_tick_ = 0;
  std::atomic<bool> planner_entry_pending_{true};
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
