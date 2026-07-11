#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "a3_pingpong/pp_onnx_policy.hpp"

namespace {

TEST(PpOnnxPolicy, LoadsRealModelWithStableInputTypeInfoLifetime) {
  const char* model_path = std::getenv("A3_PP_ONNX_PATH");
  if (model_path == nullptr || std::string(model_path).empty()) {
    GTEST_SKIP() << "set A3_PP_ONNX_PATH to a real exported policy.onnx";
  }

  a3_pingpong::PpOnnxPolicy policy(model_path, true);
  const int obs_dim = policy.obs_dim();
  EXPECT_TRUE(obs_dim == a3_pingpong::kObsDim || obs_dim == a3_pingpong::kObsDim179 ||
              obs_dim == a3_pingpong::kObsDim177 || obs_dim == a3_pingpong::kObsDim175 ||
              obs_dim == a3_pingpong::kObsDim110);
  EXPECT_EQ(policy.joint_names().size(), static_cast<std::size_t>(a3_pingpong::kNumJoints));
  EXPECT_EQ(policy.default_q().size(), a3_pingpong::kNumJoints);
}

}  // namespace
