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
