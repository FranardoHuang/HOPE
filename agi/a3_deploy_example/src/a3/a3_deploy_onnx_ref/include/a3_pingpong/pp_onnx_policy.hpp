// ONNX session wrapper for model_15200. Ported from
// hope_ws/.../hope_wbc_runner/onnx_policy.py. Inputs: obs[1,180], time_step[1,1].
// Outputs: actions[1,31] + reference side-outputs joint_pos[31], joint_vel[31],
// body_pos_w[14,3], body_quat_w[14,4] (clip baked in -> obs-independent refs).
// Metadata: joint_names, default_joint_pos, action_scale, joint_stiffness(kp),
// joint_damping(kd), body_names. Decode: target_q = default_q + action*scale.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "a3_pingpong/pp_obs_builder.hpp"

namespace a3_pingpong {

class PpOnnxPolicy {
 public:
  explicit PpOnnxPolicy(const std::string& model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "pp_onnx"),
        mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), so);

    // verify obs input dim
    auto in0 = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in0.size() != 2 || in0[1] != kObsDim)
      throw std::runtime_error("ONNX obs input is not [1,180]");

    Ort::AllocatorWithDefaultOptions alloc;
    auto md = session_->GetModelMetadata();
    joint_names_ = SplitCsv(LookupMeta(md, alloc, "joint_names"));
    body_names_ = SplitCsv(LookupMeta(md, alloc, "body_names"));
    default_q_ = ToVec(LookupMeta(md, alloc, "default_joint_pos"));
    action_scale_ = ToVec(LookupMeta(md, alloc, "action_scale"));
    kp_ = ToVec(LookupMeta(md, alloc, "joint_stiffness"));
    kd_ = ToVec(LookupMeta(md, alloc, "joint_damping"));
    if ((int)joint_names_.size() != kNumJoints || default_q_.size() != kNumJoints)
      throw std::runtime_error("ONNX metadata joint count != 31");
  }

  const std::vector<std::string>& joint_names() const { return joint_names_; }
  const std::vector<std::string>& body_names() const { return body_names_; }
  const Eigen::VectorXd& default_q() const { return default_q_; }
  const Eigen::VectorXd& action_scale() const { return action_scale_; }
  const Eigen::VectorXd& kp() const { return kp_; }
  const Eigen::VectorXd& kd() const { return kd_; }

  // Reference motion at time_step (obs-independent side-outputs).
  PpRefs refs(int time_step) {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(kObsDim);
    auto out = Run(zero, time_step, {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w"});
    PpRefs r;
    r.joint_pos = Map(out[0], kNumJoints);
    r.joint_vel = Map(out[1], kNumJoints);
    const float* bp = out[2].GetTensorData<float>();  // [1,14,3]
    const float* bq = out[3].GetTensorData<float>();  // [1,14,4]
    r.anchor_pos_w = Vec3(bp[kAnchorTrackedIdx * 3 + 0], bp[kAnchorTrackedIdx * 3 + 1],
                          bp[kAnchorTrackedIdx * 3 + 2]);
    r.anchor_quat_w = Vec4(bq[kAnchorTrackedIdx * 4 + 0], bq[kAnchorTrackedIdx * 4 + 1],
                           bq[kAnchorTrackedIdx * 4 + 2], bq[kAnchorTrackedIdx * 4 + 3]);
    // reference pelvis/root = tracked body index 0 (perfect-tracking mode only)
    r.ref_pelvis_pos_w = Vec3(bp[0], bp[1], bp[2]);
    return r;
  }

  // Deterministic mean action (31), Isaac order.
  Eigen::VectorXd mean_action(const Eigen::VectorXd& obs, int time_step) {
    auto out = Run(obs, time_step, {"actions"});
    return Map(out[0], kNumJoints);
  }

  // target_q = default_q + action * action_scale (Isaac order).
  Eigen::VectorXd target_q(const Eigen::VectorXd& action) const {
    return default_q_ + action.cwiseProduct(action_scale_);
  }

 private:
  std::vector<Ort::Value> Run(const Eigen::VectorXd& obs, int time_step,
                              const std::vector<const char*>& out_names) {
    obs_f_.resize(kObsDim);
    for (int i = 0; i < kObsDim; ++i) obs_f_[i] = static_cast<float>(obs[i]);
    float ts = static_cast<float>(time_step);
    std::array<int64_t, 2> obs_shape{1, kObsDim};
    std::array<int64_t, 2> ts_shape{1, 1};
    std::array<Ort::Value, 2> ins{
        Ort::Value::CreateTensor<float>(mem_, obs_f_.data(), kObsDim, obs_shape.data(), 2),
        Ort::Value::CreateTensor<float>(mem_, &ts, 1, ts_shape.data(), 2)};
    static const char* in_names[2] = {"obs", "time_step"};
    return session_->Run(Ort::RunOptions{nullptr}, in_names, ins.data(), 2,
                         out_names.data(), out_names.size());
  }

  static Eigen::VectorXd Map(const Ort::Value& v, int n) {
    const float* p = v.GetTensorData<float>();
    Eigen::VectorXd out(n);
    for (int i = 0; i < n; ++i) out[i] = p[i];
    return out;
  }
  static std::string LookupMeta(Ort::ModelMetadata& md, Ort::AllocatorWithDefaultOptions& a,
                                const char* key) {
    auto s = md.LookupCustomMetadataMapAllocated(key, a);
    if (!s) throw std::runtime_error(std::string("ONNX missing metadata: ") + key);
    return std::string(s.get());
  }
  static std::vector<std::string> SplitCsv(const std::string& s) {
    std::vector<std::string> out; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
  }
  static Eigen::VectorXd ToVec(const std::string& s) {
    auto t = SplitCsv(s); Eigen::VectorXd v(t.size());
    for (size_t i = 0; i < t.size(); ++i) v[i] = std::stod(t[i]);
    return v;
  }

  Ort::Env env_;
  Ort::MemoryInfo mem_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> joint_names_, body_names_;
  Eigen::VectorXd default_q_, action_scale_, kp_, kd_;
  std::vector<float> obs_f_;
};

}  // namespace a3_pingpong
