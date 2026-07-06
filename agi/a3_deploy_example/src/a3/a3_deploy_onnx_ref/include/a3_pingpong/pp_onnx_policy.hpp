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

    // detect obs input dim: 180 (full), 175 (deploy_parity) or 177 (hitter_footwork).
    // The obs builder + buffers use obs_dim_.
    auto in0 = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in0.size() != 2 ||
        (in0[1] != kObsDim && in0[1] != kObsDim175 && in0[1] != kObsDim177))
      throw std::runtime_error("ONNX obs input is not [1,180], [1,177] or [1,175]");
    obs_dim_ = static_cast<int>(in0[1]);

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

    // ZERO-GAIN GUARD. These gains go verbatim into the published RobotCommand; a non-positive
    // kp/kd means that joint receives NO PD torque (limp). The 2026-07-02 explicitpd_ft export
    // baked kp=kd=0 for all IdealPD (arms/waist/ankle) joints because the exporter read PhysX
    // drive gains, which explicit actuators null -> the free-base robot collapsed. Fail fast
    // instead of silently deploying a torqueless robot.
    for (int i = 0; i < kNumJoints; ++i) {
      if (kp_[i] <= 0.0 || kd_[i] <= 0.0)
        throw std::runtime_error(
            "ONNX metadata has non-positive PD gain for joint '" + joint_names_[i] +
            "' (kp=" + std::to_string(kp_[i]) + ", kd=" + std::to_string(kd_[i]) +
            "): broken export (exporter must bake NOMINAL actuator gains, "
            "data.default_joint_stiffness — not the PhysX drive gains). Re-export or patch "
            "the model metadata; refusing to run.");
    }

    // OPTIONAL per-clip reference-clock layout (clip_seg_lengths / clip_strike_phases). New
    // exports carry these so the runner's ClipLayout matches the BAKED clips; legacy models
    // (model_15200, v1 clips) predate the keys and fall back to the hardcoded v1 layout.
    const std::string seg_s = LookupMetaOptional(md, alloc, "clip_seg_lengths");
    const std::string pha_s = LookupMetaOptional(md, alloc, "clip_strike_phases");
    if (!seg_s.empty() && !pha_s.empty()) {
      const Eigen::VectorXd seg = ToVec(seg_s);
      const Eigen::VectorXd pha = ToVec(pha_s);
      if (seg.size() == pha.size() && seg.size() >= 1) {
        clip_seg_lengths_.assign(seg.data(), seg.data() + seg.size());
        clip_strike_phases_.assign(pha.data(), pha.data() + pha.size());
      }
    }

    // OPTIONAL per-clip reference base->racket reach offset at the strike frame
    // (dx0,dy0,dx1,dy1,...). Baked by utils/exporter.py since 2026-07-06 for 177-D
    // hitter_footwork models: the runner derives the deploy-time base STATION from the
    // racket target as station_xy = target_xy - reach_offset_xy[clip] (training
    // base_couple_mode=reference_reach). PpPolicy computes a refs-based fallback for
    // metadata-less 177 exports.
    const std::string reach_s = LookupMetaOptional(md, alloc, "ref_reach_offset_xy");
    if (!reach_s.empty()) {
      const Eigen::VectorXd v = ToVec(reach_s);
      for (int i = 0; i + 1 < v.size(); i += 2) reach_offsets_.push_back(Vec2(v[i], v[i + 1]));
    }
  }

  int obs_dim() const { return obs_dim_; }
  // Per-clip layout baked by new exports (empty on legacy models -> caller keeps its default).
  bool has_clip_layout() const { return !clip_seg_lengths_.empty(); }
  const std::vector<double>& clip_seg_lengths() const { return clip_seg_lengths_; }
  const std::vector<double>& clip_strike_phases() const { return clip_strike_phases_; }
  // Per-clip reference reach offsets (empty when the export predates the metadata key).
  bool has_reach_offsets() const { return !reach_offsets_.empty(); }
  const std::vector<Vec2>& reach_offsets() const { return reach_offsets_; }

  // Reference base->racket reach offset at a given time_step, computed from the baked refs:
  // blade world xy (ref pelvis pose + racket FK on the ref joints) minus ref pelvis xy. Same
  // arithmetic as training's _ensure_reference_strike_state — the fallback for 177 models
  // whose export predates the ref_reach_offset_xy metadata key.
  Vec2 reach_offset_from_refs(int time_step) {
    auto out = Run(Eigen::VectorXd::Zero(obs_dim_), time_step,
                   {"joint_pos", "body_pos_w", "body_quat_w"});
    const Eigen::VectorXd ref_q = Map(out[0], kNumJoints);
    const float* bp = out[1].GetTensorData<float>();  // [1,14,3]; tracked body 0 = pelvis
    const float* bq = out[2].GetTensorData<float>();  // [1,14,4]
    const Vec3 pelvis_pos(bp[0], bp[1], bp[2]);
    const Vec4 pelvis_quat(bq[0], bq[1], bq[2], bq[3]);
    const Vec3 blade_w = pelvis_pos + mat_from_quat(pelvis_quat) * racket_pos_pelvis(ref_q);
    return Vec2(blade_w[0] - pelvis_pos[0], blade_w[1] - pelvis_pos[1]);
  }
  const std::vector<std::string>& joint_names() const { return joint_names_; }
  const std::vector<std::string>& body_names() const { return body_names_; }
  const Eigen::VectorXd& default_q() const { return default_q_; }
  const Eigen::VectorXd& action_scale() const { return action_scale_; }
  const Eigen::VectorXd& kp() const { return kp_; }
  const Eigen::VectorXd& kd() const { return kd_; }

  // Reference motion at time_step (obs-independent side-outputs).
  PpRefs refs(int time_step) {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(obs_dim_);
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
    obs_f_.resize(obs_dim_);
    for (int i = 0; i < obs_dim_; ++i) obs_f_[i] = static_cast<float>(obs[i]);
    float ts = static_cast<float>(time_step);
    std::array<int64_t, 2> obs_shape{1, obs_dim_};
    std::array<int64_t, 2> ts_shape{1, 1};
    std::array<Ort::Value, 2> ins{
        Ort::Value::CreateTensor<float>(mem_, obs_f_.data(), obs_dim_, obs_shape.data(), 2),
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
  static std::string LookupMetaOptional(Ort::ModelMetadata& md,
                                        Ort::AllocatorWithDefaultOptions& a, const char* key) {
    auto s = md.LookupCustomMetadataMapAllocated(key, a);
    return s ? std::string(s.get()) : std::string{};
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
  std::vector<double> clip_seg_lengths_, clip_strike_phases_;  // empty on legacy exports
  std::vector<Vec2> reach_offsets_;  // per-clip ref base->racket reach; empty on old exports
  Eigen::VectorXd default_q_, action_scale_, kp_, kd_;
  std::vector<float> obs_f_;
  int obs_dim_ = kObsDim;  // detected from the model input at load (180 full / 175 deploy_parity)
};

}  // namespace a3_pingpong
