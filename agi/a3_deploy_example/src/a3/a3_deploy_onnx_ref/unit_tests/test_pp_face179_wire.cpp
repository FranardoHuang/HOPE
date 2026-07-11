#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "a3_pingpong/pp_face179_contract.hpp"
#include "a3_pingpong/pp_obs_builder.hpp"

namespace {

void MakeInputs(a3_pingpong::PpRefs& refs, a3_pingpong::PpRobotState& state,
                a3_pingpong::PpRacketTarget& target, Eigen::VectorXd& last_action,
                Eigen::VectorXd& default_q) {
  refs.joint_pos = Eigen::VectorXd::Zero(a3_pingpong::kNumJoints);
  refs.joint_vel = Eigen::VectorXd::Zero(a3_pingpong::kNumJoints);
  refs.anchor_pos_w = a3_pingpong::Vec3(0.0, 0.0, 1.2);
  refs.anchor_quat_w = a3_pingpong::Vec4(1.0, 0.0, 0.0, 0.0);
  refs.ref_pelvis_pos_w = a3_pingpong::Vec3(0.0, 0.0, 0.95);
  state.base_pos_w = a3_pingpong::Vec3(0.0, 0.0, 0.95);
  state.base_quat_w = a3_pingpong::Vec4(1.0, 0.0, 0.0, 0.0);
  state.torso_pos_w = refs.anchor_pos_w;
  state.torso_quat_w = refs.anchor_quat_w;
  state.base_ang_vel_b = a3_pingpong::Vec3::Zero();
  state.q = Eigen::VectorXd::Zero(a3_pingpong::kNumJoints);
  state.qd = Eigen::VectorXd::Zero(a3_pingpong::kNumJoints);
  target.pos_w = a3_pingpong::Vec3(0.7, -0.4, 0.82);
  target.vel_w = a3_pingpong::Vec3(1.5, 1.4, 0.7);
  target.swing_sign = 1.0;
  target.time_to_strike = 0.4;
  target.face_command_valid = true;
  target.normal_cmd_w = a3_pingpong::Vec3(0.6, 0.8, 0.0);
  target.rho = 0.0;
  last_action = Eigen::VectorXd::Zero(a3_pingpong::kNumJoints);
  default_q = Eigen::VectorXd::Zero(a3_pingpong::kNumJoints);
}

TEST(PpFace179Wire, ObservationKeepsExact175PrefixAndAtomicFaceTail) {
  a3_pingpong::PpRefs refs;
  a3_pingpong::PpRobotState state;
  a3_pingpong::PpRacketTarget target;
  Eigen::VectorXd last_action;
  Eigen::VectorXd default_q;
  MakeInputs(refs, state, target, last_action, default_q);
  const auto base = a3_pingpong::build_obs_175(
      refs, state, target, last_action, default_q);
  const auto face = a3_pingpong::build_obs_179(
      refs, state, target, last_action, default_q);
  ASSERT_EQ(face.size(), a3_pingpong::kObsDim179);
  EXPECT_DOUBLE_EQ((face.head(a3_pingpong::kObsDim175) - base).cwiseAbs().maxCoeff(), 0.0);
  EXPECT_NEAR(face[175], 0.6, 1e-12);
  EXPECT_NEAR(face[176], 0.8, 1e-12);
  EXPECT_NEAR(face[177], 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(face[178], 0.0);
}

TEST(PpFace179Wire, MissingFaceOrNonzeroRhoFailsClosed) {
  a3_pingpong::PpRefs refs;
  a3_pingpong::PpRobotState state;
  a3_pingpong::PpRacketTarget target;
  Eigen::VectorXd last_action;
  Eigen::VectorXd default_q;
  MakeInputs(refs, state, target, last_action, default_q);
  target.face_command_valid = false;
  EXPECT_THROW(
      a3_pingpong::build_obs_179(refs, state, target, last_action, default_q),
      std::invalid_argument);
  target.face_command_valid = true;
  target.rho = 0.1;
  EXPECT_THROW(
      a3_pingpong::build_obs_179(refs, state, target, last_action, default_q),
      std::invalid_argument);
}

TEST(PpFace179Wire, FormalMetadataRejectsEveryWrongSemanticField) {
  const std::string sha(64, 'a');
  a3_pingpong::PpFace179MetadataContract valid{
      "1", "shared_plus_y", "mount_plusY_A", "1", "3", "train", sha, sha};
  EXPECT_NO_THROW(a3_pingpong::ValidatePpFace179MetadataContract(valid));

  auto wrong = valid;
  wrong.face_command_enabled = "0";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
  wrong = valid;
  wrong.face_command_pairing = "legacy_signed_vs_A";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
  wrong = valid;
  wrong.face_obs_convention = "wrong_frame";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
  wrong = valid;
  wrong.question_bank_exact = "0";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
  wrong = valid;
  wrong.bank_schema_version = "legacy";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
  wrong = valid;
  wrong.bank_split = "exam";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
  wrong = valid;
  wrong.train_bank_sha256 = std::string(64, 'A');
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
  wrong = valid;
  wrong.source_family_sha256 = "short";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
}

}  // namespace
