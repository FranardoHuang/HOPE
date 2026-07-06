// 180/175/177-D observation builders. 180 ported from
// hope_ws/.../hope_wbc_runner/obs_builder.py (build_obs). Layout (total 180):
//   command(62) = ref joint_pos[31] + ref joint_vel[31]
//   motion_anchor_pos_b(3), motion_anchor_ori_b(6)
//   base_ang_vel(3), joint_pos_rel(31), joint_vel(31), last_action(31),
//   projected_gravity(3), base_target_pos_b(2), racket_target_pos_b(3),
//   racket_target_vel_w(3), time_to_strike(1), swing_type(1)
// 175 = deploy_parity (drops anchor_pos + base_target, racket target FK-relative);
// 177 = hitter_footwork (the 175 layout + base_target_pos_b(2) re-inserted after
// projected_gravity — the HITTER-style relative-Δ station footwork channel).
#pragma once

#include <Eigen/Dense>

#include "a3_pingpong/pp_frame_math.hpp"
#include "a3_pingpong/pp_racket_fk.hpp"

namespace a3_pingpong {

constexpr int kObsDim = 180;
constexpr int kObsDim175 = 175;
constexpr int kObsDim177 = 177;
constexpr int kNumJoints = 31;
constexpr int kAnchorTrackedIdx = 7;  // torso_Link in the 14-body tracked order

// Reference motion at the current time_step (the ONNX side-outputs). Only the
// anchor body (index 7) is needed from the 14-body pose arrays.
struct PpRefs {
  Eigen::VectorXd joint_pos;  // (31)
  Eigen::VectorXd joint_vel;  // (31)
  Vec3 anchor_pos_w;          // body_pos_w[7]  (torso anchor)
  Vec4 anchor_quat_w;         // body_quat_w[7]  (w,x,y,z)
  Vec3 ref_pelvis_pos_w;      // body_pos_w[0]  (reference pelvis/root; used only by
                              // the perfect-tracking localization mode for the
                              // racket/base-target obs terms - NOT part of the obs itself)
};

// Robot state (all world poses in the policy frame). torso/base world pose is
// the sim2real localisation gap on the real robot (nominal there).
struct PpRobotState {
  Vec3 base_pos_w;
  Vec4 base_quat_w;
  Vec3 torso_pos_w;
  Vec4 torso_quat_w;
  Vec3 base_ang_vel_b;        // pelvis gyro, pelvis BODY frame
  Eigen::VectorXd q;          // (31) Isaac order
  Eigen::VectorXd qd;         // (31)
};

// Planner target, already in the policy world frame.
struct PpRacketTarget {
  Vec3 pos_w;
  Vec3 vel_w;
  double swing_sign;          // +1 forehand / -1 backhand
  double time_to_strike;      // seconds
  Vec2 base_target_xy = Vec2::Zero();
};

// Assemble the 180-D observation (double precision; cast to float at the ONNX
// boundary). last_action and default_q are length-31.
// use_base_yaw_for_targets: rotate the (base/racket) world targets into the
// robot's yaw-heading frame using the REAL base-IMU yaw. TRUE reproduces the
// training/sim2sim transform exactly (default, so the parity golden + sim are
// untouched). On HARDWARE the pelvis IMU yaw is NOT world-referenced (it drifts
// boot-to-boot, e.g. 130-165 deg, and disagrees with the torso IMU), so feeding
// it here rotates the scripted target away from where it should be. Pass FALSE
// there: the scripted target is already expressed in the robot's nominal heading
// (+x), so we use identity yaw. This is a no-op whenever base yaw ~ 0 (sim), and
// it does NOT affect projected_gravity (yaw-invariant) or the anchor terms.
inline Eigen::VectorXd build_obs_180(const PpRefs& refs, const PpRobotState& state,
                                     const PpRacketTarget& target,
                                     const Eigen::VectorXd& last_action,
                                     const Eigen::VectorXd& default_q,
                                     bool use_base_yaw_for_targets = true) {
  Eigen::VectorXd obs(kObsDim);
  int o = 0;

  // 1. command (62) = ref joint_pos[31] ++ ref joint_vel[31]
  obs.segment(o, kNumJoints) = refs.joint_pos; o += kNumJoints;
  obs.segment(o, kNumJoints) = refs.joint_vel; o += kNumJoints;

  // 2/3. ref anchor (torso) expressed in robot anchor (torso) frame
  Vec3 pos_b;
  Vec4 ori_q;
  subtract_frame_transforms(state.torso_pos_w, state.torso_quat_w,
                            refs.anchor_pos_w, refs.anchor_quat_w, pos_b, ori_q);
  const Mat3 R = mat_from_quat(ori_q);
  // 6D rot = first two COLUMNS of R, row-major: R00,R01,R10,R11,R20,R21
  obs.segment<3>(o) = pos_b; o += 3;
  obs[o++] = R(0, 0); obs[o++] = R(0, 1);
  obs[o++] = R(1, 0); obs[o++] = R(1, 1);
  obs[o++] = R(2, 0); obs[o++] = R(2, 1);

  // 4. base angular velocity (3), body frame
  obs.segment<3>(o) = state.base_ang_vel_b; o += 3;

  // 5/6. joint pos rel / joint vel (31 each)
  obs.segment(o, kNumJoints) = state.q - default_q; o += kNumJoints;
  obs.segment(o, kNumJoints) = state.qd; o += kNumJoints;

  // 7. last action (31)
  obs.segment(o, kNumJoints) = last_action; o += kNumJoints;

  // 8. projected gravity (3), body frame
  obs.segment<3>(o) = projected_gravity_body(state.base_quat_w); o += 3;

  // 9. base target XY in yaw-heading base frame (2)
  // yaw-heading from the real IMU (training) OR identity (hardware: the absolute
  // IMU yaw is unreferenced; the scripted target lives in the robot's +x frame).
  const Vec4 yq = use_base_yaw_for_targets ? yaw_quat(state.base_quat_w)
                                           : Vec4(1.0, 0.0, 0.0, 0.0);
  Vec3 delta_base(target.base_target_xy[0] - state.base_pos_w[0],
                  target.base_target_xy[1] - state.base_pos_w[1], 0.0);
  const Vec3 base_tgt_b = quat_rotate_inverse(yq, delta_base);
  obs[o++] = base_tgt_b[0];
  obs[o++] = base_tgt_b[1];

  // 10. racket target pos in yaw-heading base frame (3)
  obs.segment<3>(o) = quat_rotate_inverse(yq, target.pos_w - state.base_pos_w); o += 3;

  // 11. racket target velocity, world (3)
  obs.segment<3>(o) = target.vel_w; o += 3;

  // 12/13. time_to_strike (1), swing_type (1)
  obs[o++] = target.time_to_strike;
  obs[o++] = target.swing_sign;

  return obs;  // o == 180
}

// Assemble the 175-D "deploy_parity" observation. Layout (total 175):
//   command(62), motion_anchor_ori_b(6), base_ang_vel(3), joint_pos_rel(31),
//   joint_vel(31), last_action(31), projected_gravity(3), racket_target_pos_b(3),
//   racket_target_vel_w(3), time_to_strike(1), swing_type(1)
//
// Differences from build_obs_180 (everything else is byte-identical):
//   1. motion_anchor_pos_b (the 3-vec before the 6D rot) is REMOVED.
//   2. base_target_pos_b (the 2-vec block) is REMOVED.
//   3. racket_target_pos_b uses the CURRENT racket FK world position as the
//      reference point instead of the base position:
//        180: quat_rotate_inverse(yq, target.pos_w - state.base_pos_w)
//        175: quat_rotate_inverse(yq, target.pos_w - racket_pos_w)
//      where racket_pos_w = base_pos_w + R(base_quat_w) * racket_pos_pelvis(q).
//      On the real robot this cancels the (unobservable) world base position,
//      matching the deploy-honest training recipe.
//
// Signature matches build_obs_180 exactly. See build_obs_180 for the meaning of
// use_base_yaw_for_targets.
inline Eigen::VectorXd build_obs_175(const PpRefs& refs, const PpRobotState& state,
                                     const PpRacketTarget& target,
                                     const Eigen::VectorXd& last_action,
                                     const Eigen::VectorXd& default_q,
                                     bool use_base_yaw_for_targets = true) {
  Eigen::VectorXd obs(kObsDim175);
  int o = 0;

  // 1. command (62) = ref joint_pos[31] ++ ref joint_vel[31]
  obs.segment(o, kNumJoints) = refs.joint_pos; o += kNumJoints;
  obs.segment(o, kNumJoints) = refs.joint_vel; o += kNumJoints;

  // 2. motion_anchor_ori_b (6) — ref anchor (torso) orientation in robot anchor
  //    (torso) frame. NOTE: motion_anchor_pos_b (3) is intentionally NOT emitted.
  Vec3 pos_b;  // computed but dropped from the 175 layout
  Vec4 ori_q;
  subtract_frame_transforms(state.torso_pos_w, state.torso_quat_w,
                            refs.anchor_pos_w, refs.anchor_quat_w, pos_b, ori_q);
  const Mat3 R = mat_from_quat(ori_q);
  // 6D rot = first two COLUMNS of R, row-major: R00,R01,R10,R11,R20,R21
  obs[o++] = R(0, 0); obs[o++] = R(0, 1);
  obs[o++] = R(1, 0); obs[o++] = R(1, 1);
  obs[o++] = R(2, 0); obs[o++] = R(2, 1);

  // 3. base angular velocity (3), body frame
  obs.segment<3>(o) = state.base_ang_vel_b; o += 3;

  // 4/5. joint pos rel / joint vel (31 each)
  obs.segment(o, kNumJoints) = state.q - default_q; o += kNumJoints;
  obs.segment(o, kNumJoints) = state.qd; o += kNumJoints;

  // 6. last action (31)
  obs.segment(o, kNumJoints) = last_action; o += kNumJoints;

  // 7. projected gravity (3), body frame
  obs.segment<3>(o) = projected_gravity_body(state.base_quat_w); o += 3;

  // 8. racket target pos, relative to the CURRENT racket FK position, in the
  //    yaw-heading base frame (3). base_target_pos_b (2) is NOT emitted.
  const Vec4 yq = use_base_yaw_for_targets ? yaw_quat(state.base_quat_w)
                                           : Vec4(1.0, 0.0, 0.0, 0.0);
  const Vec3 racket_pos_w =
      state.base_pos_w + mat_from_quat(state.base_quat_w) * racket_pos_pelvis(state.q);
  obs.segment<3>(o) = quat_rotate_inverse(yq, target.pos_w - racket_pos_w); o += 3;

  // 9. racket target velocity, world (3)
  obs.segment<3>(o) = target.vel_w; o += 3;

  // 10/11. time_to_strike (1), swing_type (1)
  obs[o++] = target.time_to_strike;
  obs[o++] = target.swing_sign;

  return obs;  // o == 175
}

// Assemble the 177-D "hitter_footwork" observation. Layout (total 177):
//   command(62), motion_anchor_ori_b(6), base_ang_vel(3), joint_pos_rel(31),
//   joint_vel(31), last_action(31), projected_gravity(3), base_target_pos_b(2),
//   racket_target_pos_b(3), racket_target_vel_w(3), time_to_strike(1), swing_type(1)
//
// = build_obs_175 PLUS the base_target_pos_b(2) block from build_obs_180 re-inserted
// between projected_gravity and racket_target_pos_b. Training semantics
// (hope_commands.py base_target_pos_b): Δxy from the CURRENT base to the commanded
// station, rotated into the yaw-heading base frame. Callers must set
// target.base_target_xy to the WORLD station; for the Δ=0 graceful fallback (hold /
// no real base localization: contract "already at station") set it to the CURRENT
// base xy so the delta vanishes. Everything else is byte-identical to build_obs_175.
inline Eigen::VectorXd build_obs_177(const PpRefs& refs, const PpRobotState& state,
                                     const PpRacketTarget& target,
                                     const Eigen::VectorXd& last_action,
                                     const Eigen::VectorXd& default_q,
                                     bool use_base_yaw_for_targets = true) {
  Eigen::VectorXd obs(kObsDim177);
  int o = 0;

  // 1. command (62) = ref joint_pos[31] ++ ref joint_vel[31]
  obs.segment(o, kNumJoints) = refs.joint_pos; o += kNumJoints;
  obs.segment(o, kNumJoints) = refs.joint_vel; o += kNumJoints;

  // 2. motion_anchor_ori_b (6) — ref anchor (torso) orientation in robot anchor
  //    (torso) frame. motion_anchor_pos_b (3) is intentionally NOT emitted.
  Vec3 pos_b;  // computed but dropped from the 177 layout
  Vec4 ori_q;
  subtract_frame_transforms(state.torso_pos_w, state.torso_quat_w,
                            refs.anchor_pos_w, refs.anchor_quat_w, pos_b, ori_q);
  const Mat3 R = mat_from_quat(ori_q);
  // 6D rot = first two COLUMNS of R, row-major: R00,R01,R10,R11,R20,R21
  obs[o++] = R(0, 0); obs[o++] = R(0, 1);
  obs[o++] = R(1, 0); obs[o++] = R(1, 1);
  obs[o++] = R(2, 0); obs[o++] = R(2, 1);

  // 3. base angular velocity (3), body frame
  obs.segment<3>(o) = state.base_ang_vel_b; o += 3;

  // 4/5. joint pos rel / joint vel (31 each)
  obs.segment(o, kNumJoints) = state.q - default_q; o += kNumJoints;
  obs.segment(o, kNumJoints) = state.qd; o += kNumJoints;

  // 6. last action (31)
  obs.segment(o, kNumJoints) = last_action; o += kNumJoints;

  // 7. projected gravity (3), body frame
  obs.segment<3>(o) = projected_gravity_body(state.base_quat_w); o += 3;

  // 8. base target XY in yaw-heading base frame (2) — the footwork station channel.
  const Vec4 yq = use_base_yaw_for_targets ? yaw_quat(state.base_quat_w)
                                           : Vec4(1.0, 0.0, 0.0, 0.0);
  Vec3 delta_base(target.base_target_xy[0] - state.base_pos_w[0],
                  target.base_target_xy[1] - state.base_pos_w[1], 0.0);
  const Vec3 base_tgt_b = quat_rotate_inverse(yq, delta_base);
  obs[o++] = base_tgt_b[0];
  obs[o++] = base_tgt_b[1];

  // 9. racket target pos, relative to the CURRENT racket FK position, in the
  //    yaw-heading base frame (3) — same deploy-honest reframe as build_obs_175.
  const Vec3 racket_pos_w =
      state.base_pos_w + mat_from_quat(state.base_quat_w) * racket_pos_pelvis(state.q);
  obs.segment<3>(o) = quat_rotate_inverse(yq, target.pos_w - racket_pos_w); o += 3;

  // 10. racket target velocity, world (3)
  obs.segment<3>(o) = target.vel_w; o += 3;

  // 11/12. time_to_strike (1), swing_type (1)
  obs[o++] = target.time_to_strike;
  obs[o++] = target.swing_sign;

  return obs;  // o == 177
}

}  // namespace a3_pingpong
