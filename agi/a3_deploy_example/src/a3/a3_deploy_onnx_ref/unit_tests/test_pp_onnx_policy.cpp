#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "a3_pingpong/pp_onnx_policy.hpp"

namespace {

TEST(PpOnnxPolicy, LoadsOnlyPublishableRealModelWithStableInputTypeInfoLifetime) {
  const char* model_path = std::getenv("A3_PP_ONNX_PATH");
  if (model_path == nullptr || std::string(model_path).empty()) {
    GTEST_SKIP() << "set A3_PP_ONNX_PATH to a real exported policy.onnx";
  }

  a3_pingpong::PpOnnxPolicy policy(model_path);
  const int obs_dim = policy.obs_dim();
  EXPECT_TRUE(obs_dim == a3_pingpong::kObsDim || obs_dim == a3_pingpong::kObsDim179 ||
              obs_dim == a3_pingpong::kObsDim177 || obs_dim == a3_pingpong::kObsDim175 ||
              obs_dim == a3_pingpong::kObsDim110);
  EXPECT_EQ(policy.joint_names().size(), static_cast<std::size_t>(a3_pingpong::kNumJoints));
  EXPECT_EQ(policy.default_q().size(), a3_pingpong::kNumJoints);
  EXPECT_TRUE(policy.has_schema3_execution_contract());
  EXPECT_TRUE(policy.training_contract_exact());
  EXPECT_TRUE(policy.publishable_model_contract());
  if (obs_dim == a3_pingpong::kObsDim179) {
    EXPECT_FALSE(policy.face_normal_envelope().payload_sha256.empty());
    EXPECT_EQ(policy.face_normal_envelope().mount_normal_signs[0], 1.0);
    EXPECT_EQ(policy.face_normal_envelope().mount_normal_signs[1], -1.0);
  }
}

}  // namespace
