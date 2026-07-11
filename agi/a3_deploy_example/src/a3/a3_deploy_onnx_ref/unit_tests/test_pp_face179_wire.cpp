#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <locale>

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

a3_pingpong::PpFace179MetadataContract ValidFaceContract() {
  // Contract-bound representative geometry for the immutable formal train bank. The heavy NPZ is
  // ignored; the exact real-bank prospective fixture is configs/phase1_face179_*_20260712.json.
  const std::string bank_sha =
      "2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700";
  const std::string family_sha =
      "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5";
  a3_pingpong::PpFaceNormalEnvelopeMetadata envelope{
      "1",
      "world_table_frame0",
      "mount_plusY_A",
      "shared_plus_y",
      "per_clip_sign_preserving_spherical_mean_cap_v1",
      "0.0002",
      "0.000001",
      "0.000001",
      "forehand,backhand",
      "1,-1",
      "0.8,0.6,0;-0.8,0.6,0",
      "0.8,0.6,0;-0.8,0.6,0",
      "0.974278,0.972078",
      "757,724",
      bank_sha,
      family_sha,
      ""};
  envelope.payload_sha256 = a3_pingpong::PpSha256Hex(
      a3_pingpong::BuildPpFaceNormalEnvelopePayload(envelope));
  return {"1", "shared_plus_y", "mount_plusY_A", "1", "3", "train",
          bank_sha, family_sha, "1,-1", envelope};
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
  const auto valid = ValidFaceContract();
  EXPECT_EQ(valid.normal_envelope.payload_sha256,
            "d47b426c41df974fbdd83adb68cf563f87844958b3a4e28a7b3d80d64a7ddc88");
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
  wrong = valid;
  wrong.mount_normal_sign_per_clip = "1,1";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
  wrong = valid;
  wrong.normal_envelope.mount_normal_sign_per_clip = "1,1";
  wrong.normal_envelope.payload_sha256 = a3_pingpong::PpSha256Hex(
      a3_pingpong::BuildPpFaceNormalEnvelopePayload(wrong.normal_envelope));
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
}

TEST(PpFace179Wire, NormalEnvelopeRejectsMissingMalformedOrUnboundMetadata) {
  const auto valid = ValidFaceContract();
  auto wrong = valid;
  wrong.normal_envelope = {};
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);

  wrong = valid;
  wrong.normal_envelope.frame = "base_link";
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);

  wrong = valid;
  wrong.normal_envelope.centers = "1,0,0";
  wrong.normal_envelope.payload_sha256 = a3_pingpong::PpSha256Hex(
      a3_pingpong::BuildPpFaceNormalEnvelopePayload(wrong.normal_envelope));
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);

  wrong = valid;
  wrong.normal_envelope.centers = "nan,0.6,0;0.8,-0.6,0";
  wrong.normal_envelope.payload_sha256 = a3_pingpong::PpSha256Hex(
      a3_pingpong::BuildPpFaceNormalEnvelopePayload(wrong.normal_envelope));
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);

  wrong = valid;
  wrong.normal_envelope.min_dots = "1.1,0.9";
  wrong.normal_envelope.payload_sha256 = a3_pingpong::PpSha256Hex(
      a3_pingpong::BuildPpFaceNormalEnvelopePayload(wrong.normal_envelope));
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);

  wrong = valid;
  wrong.normal_envelope.row_counts = "64,0";
  wrong.normal_envelope.payload_sha256 = a3_pingpong::PpSha256Hex(
      a3_pingpong::BuildPpFaceNormalEnvelopePayload(wrong.normal_envelope));
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);

  wrong = valid;
  wrong.normal_envelope.train_bank_sha256 = std::string(64, 'b');
  wrong.normal_envelope.payload_sha256 = a3_pingpong::PpSha256Hex(
      a3_pingpong::BuildPpFaceNormalEnvelopePayload(wrong.normal_envelope));
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);

  wrong = valid;
  wrong.normal_envelope.min_dots = "0.8,0.9";  // stale payload hash must catch field edits first
  EXPECT_THROW(a3_pingpong::ValidatePpFace179MetadataContract(wrong), std::invalid_argument);
}

TEST(PpFace179Wire, PhysicalWireBConvertsPerClipToRawAThenGatesSelectedCap) {
  const auto envelope = a3_pingpong::ValidatePpFace179MetadataContract(ValidFaceContract());
  const auto fh_a = envelope.WireBToRawA(0, 0.8, 0.6, 0.0);
  const auto bh_a = envelope.WireBToRawA(1, 0.8, -0.6, 0.0);
  EXPECT_DOUBLE_EQ(fh_a[0], 0.8);
  EXPECT_DOUBLE_EQ(fh_a[1], 0.6);
  EXPECT_DOUBLE_EQ(fh_a[2], 0.0);
  EXPECT_DOUBLE_EQ(bh_a[0], -0.8);
  EXPECT_DOUBLE_EQ(bh_a[1], 0.6);
  EXPECT_DOUBLE_EQ(bh_a[2], -0.0);
  EXPECT_TRUE(envelope.AllowsRawA(0, fh_a[0], fh_a[1], fh_a[2]));
  EXPECT_TRUE(envelope.AllowsRawA(1, bh_a[0], bh_a[1], bh_a[2]));
  // Feeding physical-B directly into raw-A or using the wrong clip sign fails.
  EXPECT_FALSE(envelope.AllowsRawA(1, 0.8, -0.6, 0.0));
  EXPECT_FALSE(envelope.AllowsRawA(0, bh_a[0], bh_a[1], bh_a[2]));
  // Physical x>0 and unit length are necessary but no longer sufficient.
  EXPECT_FALSE(envelope.AllowsRawA(0, 1.0, 0.0, 0.0));
  EXPECT_FALSE(envelope.AllowsRawA(1, -1.0, 0.0, 0.0));
  EXPECT_FALSE(envelope.AllowsRawA(2, 0.8, 0.6, 0.0));
  EXPECT_FALSE(envelope.AllowsRawA(0, 0.8 * 1.001, 0.6 * 1.001, 0.0));
}

TEST(PpFace179Wire, RuntimeAlsoRequiresRawACandidateInsideReferenceHemisphere) {
  a3_pingpong::PpFaceNormalEnvelope envelope;
  envelope.centers[0] = {1.0, 0.0, 0.0};
  envelope.reference_normals[0] = {0.1, 0.99498743710662, 0.0};
  envelope.min_dots[0] = 0.05;
  envelope.runtime_unit_tolerance = 1e-6;
  envelope.runtime_dot_tolerance = 1e-6;
  // This unit candidate is inside the permissive center cap (dot=.1) but lies in the opposite
  // reference hemisphere. The independent reference gate must reject it.
  EXPECT_FALSE(envelope.AllowsRawA(0, 0.1, -0.99498743710662, 0.0));

  const double on_boundary_y = envelope.runtime_dot_tolerance;
  const double on_boundary_x = std::sqrt(1.0 - on_boundary_y * on_boundary_y);
  envelope.centers[0] = {on_boundary_x, on_boundary_y, 0.0};
  envelope.reference_normals[0] = {0.0, 1.0, 0.0};
  envelope.min_dots[0] = 0.9;
  // The contract is an open reference hemisphere: equality at tolerance must also fail.
  EXPECT_FALSE(envelope.AllowsRawA(0, on_boundary_x, on_boundary_y, 0.0));
}

TEST(PpFace179Wire, MetadataShaUsesStandardSha256) {
  EXPECT_EQ(a3_pingpong::PpSha256Hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(a3_pingpong::PpSha256Hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

class CommaDecimalPunct : public std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};

TEST(PpFace179Wire, HashAndNumericParsingIgnoreProcessLocale) {
  const std::locale previous = std::locale();
  std::locale::global(std::locale(std::locale::classic(), new CommaDecimalPunct));
  EXPECT_EQ(a3_pingpong::PpSha256Hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_DOUBLE_EQ(a3_pingpong::PpEnvelopeFiniteDouble("0.974278", "test"), 0.974278);
  EXPECT_THROW(a3_pingpong::PpEnvelopeFiniteDouble("0,974278", "test"),
               std::invalid_argument);
  EXPECT_NO_THROW(a3_pingpong::ValidatePpFace179MetadataContract(ValidFaceContract()));
  std::locale::global(previous);
}

}  // namespace
