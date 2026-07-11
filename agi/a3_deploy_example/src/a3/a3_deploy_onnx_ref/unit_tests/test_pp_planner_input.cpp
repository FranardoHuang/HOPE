#include <gtest/gtest.h>

#include <chrono>
#include <limits>
#include <thread>
#include <vector>

#include "a3_pingpong/pp_planner_input.hpp"

namespace {

TEST(PpPlannerInput, RacketRejectsWrongSchemaAndNonFiniteValidPayload) {
  a3_pingpong::PpRacketTargetInput input;
  std::vector<double> msg = {2, 1, 1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0};
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[0] = 1;
  msg[4] = std::numeric_limits<double>::quiet_NaN();
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[4] = -0.4;
  input.SetFromFlat(msg);
  const auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_NEAR(snap.cmd.pos_w[0], 0.7, 1e-12);
  EXPECT_NEAR(snap.cmd.vel_w[1], 1.4, 1e-12);
}

TEST(PpPlannerInput, InvalidRacketMessageCannotOverwriteLastGoodCommand) {
  a3_pingpong::PpRacketTargetInput input;
  const std::vector<double> valid = {1, 1, 1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0};
  input.SetFromFlat(valid);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  std::vector<double> invalid = valid;
  invalid[1] = 0;
  invalid[3] = 99.0;
  input.SetFromFlat(invalid);

  const auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_TRUE(snap.invalid_after);
  EXPECT_NEAR(snap.cmd.pos_w[0], 0.7, 1e-12);
}

TEST(PpPlannerInput, Face179SchemaBindsUnitNormalAndZeroRhoAtomically) {
  a3_pingpong::PpRacketTargetInput input;
  const std::vector<double> valid = {
      2, 1, -1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
      0.6, 0.8, 0.0, 0.0};
  input.SetFromFlat(valid);
  const auto snap = input.Latest();
  ASSERT_TRUE(snap.has_valid);
  EXPECT_GE(snap.valid_age_s, 0.0);
  EXPECT_TRUE(snap.cmd.has_face_command);
  EXPECT_NEAR(snap.cmd.normal_cmd[0], 0.6, 1e-12);
  EXPECT_NEAR(snap.cmd.normal_cmd[1], 0.8, 1e-12);
  EXPECT_DOUBLE_EQ(snap.cmd.rho, 0.0);
}

TEST(PpPlannerInput, Face179RejectsMalformedOrNonOpponentFacingCommand) {
  a3_pingpong::PpRacketTargetInput input;
  std::vector<double> msg = {
      2, 1, -1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
      0.6, 0.8, 0.0, 0.0};
  msg.pop_back();
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg.push_back(0.0);
  msg[12] = 1.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[12] = 0.6;
  msg[15] = 0.1;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[15] = 0.0;
  msg[11] = 1.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[11] = 0.0;
  msg[12] = -1.0;
  msg[13] = 0.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);

  msg[12] = 0.0;
  msg[14] = 1.0;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);
}

TEST(PpPlannerInput, MalformedFace179RevokesEarlierValidCommand) {
  const std::vector<double> valid = {
      2, 1, -1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0, 0,
      0.6, 0.8, 0.0, 0.0};
  const auto expect_revoked = [&valid](std::vector<double> malformed) {
    a3_pingpong::PpRacketTargetInput input;
    input.SetFromFlat(valid);
    ASSERT_TRUE(input.Latest().has_valid);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    input.SetFromFlat(malformed);
    const auto snap = input.Latest();
    ASSERT_TRUE(snap.has_valid);
    EXPECT_TRUE(snap.invalid_after);
    EXPECT_NEAR(snap.cmd.normal_cmd[0], 0.6, 1e-12);
  };

  auto wrong_length = valid;
  wrong_length.pop_back();
  expect_revoked(wrong_length);

  auto nonfinite = valid;
  nonfinite[12] = std::numeric_limits<double>::quiet_NaN();
  expect_revoked(nonfinite);

  auto nonunit = valid;
  nonunit[12] = 1.0;
  expect_revoked(nonunit);

  auto nonzero_rho = valid;
  nonzero_rho[15] = 0.1;
  expect_revoked(nonzero_rho);

  auto unsupported_frame = valid;
  unsupported_frame[11] = 1.0;
  expect_revoked(unsupported_frame);

  auto wrong_face = valid;
  wrong_face[12] = -0.6;
  wrong_face[13] = -0.8;
  expect_revoked(wrong_face);

  auto unknown_schema = valid;
  unknown_schema[0] = 3.0;
  expect_revoked(unknown_schema);

  auto fractional_schema = valid;
  fractional_schema[0] = 1.5;
  expect_revoked(fractional_schema);

  auto nonfinite_schema = valid;
  nonfinite_schema[0] = std::numeric_limits<double>::quiet_NaN();
  expect_revoked(nonfinite_schema);

  expect_revoked({});
}

TEST(PpPlannerInput, HugeFiniteOrFractionalSchemaIsIgnoredWithoutConversion) {
  a3_pingpong::PpRacketTargetInput input;
  std::vector<double> msg = {1e300, 1, 1, 0.7, -0.4, 0.8, 1.5, 1.4, 0.7, 1.0, 0.0};
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);
  msg[0] = 1.5;
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest().has_valid);
}

TEST(PpPlannerInput, BaseRejectsInvalidPoseAndNormalizesQuaternion) {
  a3_pingpong::PpBasePoseInput input;
  a3_pingpong::PpBaseSample sample;
  std::vector<double> msg = {1, 1, 1.0, 2.0, 0.95, 0.0, 0.0, 0.0, 0.0};
  input.SetFromFlat(msg);
  EXPECT_FALSE(input.Latest(sample, 1.0));

  msg[5] = 2.0;
  input.SetFromFlat(msg);
  ASSERT_TRUE(input.Latest(sample, 1.0));
  EXPECT_NEAR(sample.pos[2], 0.95, 1e-12);
  EXPECT_NEAR(sample.quat.norm(), 1.0, 1e-12);
  EXPECT_NEAR(sample.quat[0], 1.0, 1e-12);
}

}  // namespace
