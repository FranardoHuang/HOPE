// ONNX session wrapper for model_15200. Ported from
// hope_ws/.../hope_wbc_runner/onnx_policy.py. Inputs: obs[1,180], time_step[1,1].
// Outputs: actions[1,31] + reference side-outputs joint_pos[31], joint_vel[31],
// body_pos_w[14,3], body_quat_w[14,4] (clip baked in -> obs-independent refs).
// Metadata: joint_names, default_joint_pos, action_scale, joint_stiffness(kp),
// joint_damping(kd), body_names. Decode: target_q = default_q + action*scale.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "a3_pingpong/pp_obs_builder.hpp"
#include "a3_pingpong/pp_face179_contract.hpp"

namespace a3_pingpong {

class PpOnnxPolicy {
 public:
  explicit PpOnnxPolicy(const std::string& model_path,
                        bool allow_legacy_model_diagnostic = false)
      : env_(ORT_LOGGING_LEVEL_WARNING, "pp_onnx"),
        mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), so);

    // detect obs input dim: 180 (full), 179 (deploy_parity_face179),
    // 175 (deploy_parity), 177 (hitter_footwork) or 110 (hitter_pure).
    if (session_->GetInputCount() != 2)
      throw std::runtime_error("ONNX must have exactly two inputs: obs and time_step");
    // TensorTypeAndShapeInfo borrows storage from TypeInfo in ONNX Runtime's C++ API. Keep the
    // owning TypeInfo alive while reading element type and shape; chaining through a temporary
    // leaves a dangling handle and has manifested as length_error/bad_alloc on real 175/179 models.
    const auto in0_type_info = session_->GetInputTypeInfo(0);
    const auto in0_tensor_info = in0_type_info.GetTensorTypeAndShapeInfo();
    auto in0 = in0_tensor_info.GetShape();
    if (in0_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        in0.size() != 2 || in0[0] != 1 ||
        (in0[1] != kObsDim && in0[1] != kObsDim179 && in0[1] != kObsDim175 && in0[1] != kObsDim177 &&
         in0[1] != kObsDim110))
      throw std::runtime_error(
          "ONNX obs input is not [1,180], [1,179], [1,177], [1,175] or [1,110]");
    obs_dim_ = static_cast<int>(in0[1]);

    Ort::AllocatorWithDefaultOptions alloc;
    const auto obs_name = session_->GetInputNameAllocated(0, alloc);
    const auto time_name = session_->GetInputNameAllocated(1, alloc);
    const auto in1_type_info = session_->GetInputTypeInfo(1);
    const auto in1_tensor_info = in1_type_info.GetTensorTypeAndShapeInfo();
    if (!obs_name || !time_name || std::string(obs_name.get()) != "obs" ||
        std::string(time_name.get()) != "time_step" ||
        in1_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        in1_tensor_info.GetShape() != std::vector<int64_t>({1, 1}))
      throw std::runtime_error(
          "ONNX input contract must be float obs[1,D], float time_step[1,1] in that order");
    auto md = session_->GetModelMetadata();
    joint_names_ = SplitCsv(LookupMeta(md, alloc, "joint_names"));
    body_names_ = SplitCsv(LookupMeta(md, alloc, "body_names"));
    default_q_ = ToVec(LookupMeta(md, alloc, "default_joint_pos"));
    action_scale_ = ToVec(LookupMeta(md, alloc, "action_scale"));
    kp_ = ToVec(LookupMeta(md, alloc, "joint_stiffness"));
    kd_ = ToVec(LookupMeta(md, alloc, "joint_damping"));
    if ((int)joint_names_.size() != kNumJoints || default_q_.size() != kNumJoints ||
        action_scale_.size() != kNumJoints || kp_.size() != kNumJoints ||
        kd_.size() != kNumJoints)
      throw std::runtime_error("ONNX metadata joint count != 31");
    if (std::unordered_set<std::string>(joint_names_.begin(), joint_names_.end()).size() !=
        joint_names_.size())
      throw std::runtime_error("ONNX metadata joint_names contains duplicates");

    // Observation-normalization provenance is a deployment safety contract. This C++ runner has
    // no sidecar normalization stage, so a policy trained with empirical normalization is runnable
    // only when that transform is baked into the ONNX graph. Loading a raw actor here produces
    // finite but meaningless commands, which is more dangerous than a shape error.
    const std::string empirical_norm = LookupMetaOptional(md, alloc, "empirical_normalization");
    const std::string trained_norm = LookupMetaOptional(md, alloc, "trained_with_obs_norm");
    const std::string obs_norm_baked = LookupMetaOptional(md, alloc, "obs_norm_baked");
    const std::string training_contract_exact =
        LookupMetaOptional(md, alloc, "training_contract_exact");
    training_contract_exact_ = training_contract_exact == "1";
    const std::string schema = LookupMetaOptional(md, alloc, "hope_metadata_schema_version");
    auto valid_bit = [](const std::string& s) { return s.empty() || s == "0" || s == "1"; };
    if (!valid_bit(empirical_norm) || !valid_bit(trained_norm) || !valid_bit(obs_norm_baked) ||
        !valid_bit(training_contract_exact))
      throw std::runtime_error(
          "ONNX normalization metadata must be 0|1: empirical_normalization='" + empirical_norm +
          "' obs_norm_baked='" + obs_norm_baked + "'");
    if (!trained_norm.empty() && trained_norm != empirical_norm)
      throw std::runtime_error(
          "ONNX trained_with_obs_norm and empirical_normalization metadata disagree");
    if (obs_norm_baked == "1" && empirical_norm == "0")
      throw std::runtime_error(
          "ONNX normalization metadata contradicts itself: baked=1 but empirical=0");
    const bool formal_contract = schema == "2";
    if (!schema.empty() && schema != "1" && schema != "2")
      throw std::runtime_error("unsupported HOPE ONNX metadata schema version '" + schema + "'");
    if (!allow_legacy_model_diagnostic && !formal_contract)
      throw std::runtime_error(
          "publish-capable ONNX requires hope_metadata_schema_version=2; legacy or missing "
          "metadata requires explicit --allow-legacy-model-diagnostic with --no-publish");
    if (formal_contract && obs_norm_baked != "1")
      throw std::runtime_error(
          "formal schema-v2/110-D ONNX must declare obs_norm_baked=1. This C++ runner has no "
          "sidecar normalization stage; a raw actor is finite but behaviorally invalid. Re-export "
          "with the checkpoint obs_norm_state_dict baked into the graph.");
    if (formal_contract && empirical_norm != "1")
      throw std::runtime_error(
          "formal schema-v2/110-D ONNX must explicitly declare empirical_normalization=1; "
          "missing/disabled normalization provenance is not accepted for hardware deployment");
    if (formal_contract && !allow_legacy_model_diagnostic &&
        (trained_norm != "1" || training_contract_exact != "1"))
      throw std::runtime_error(
          "publish-capable formal ONNX requires trained_with_obs_norm=1 and "
          "training_contract_exact=1; re-export from the checkpoint's immutable training contract");
    if (!formal_contract && obs_norm_baked != "1") {
      if (!allow_legacy_model_diagnostic)
        throw std::runtime_error(
            "legacy ONNX has raw or unknown observation normalization provenance. It may only be "
            "opened with explicit --allow-legacy-model-diagnostic plus --no-publish; re-export "
            "as schema v2 with obs_norm_baked=1 before any command-capable run.");
      std::fprintf(stderr,
          "[pp DIAGNOSTIC] legacy/raw ONNX normalization is untrusted; allowed only because "
          "--allow-legacy-model-diagnostic and process-wide no-publish are both active.\n");
    }

    // Schema-v2 describes the ONNX packaging/layout.  Formal hardware eligibility additionally
    // requires the immutable schema-v3 *training* execution contract: exact checkpoint binding,
    // decoder limits, and both clock rates. Only the separate explicit legacy-model diagnostic
    // escape may inspect an older export with missing fields; plain no-publish remains strict.
    // Any field that is present must still be well-formed.
    training_contract_schema_version_ =
        LookupMetaOptional(md, alloc, "training_contract_schema_version");
    training_contract_sha256_ = LookupMetaOptional(md, alloc, "training_contract_sha256");
    source_checkpoint_sha256_ = LookupMetaOptional(md, alloc, "source_checkpoint_sha256");
    const std::string qdes_clamp = LookupMetaOptional(md, alloc, "qdes_clamp");
    const std::string physics_dt = LookupMetaOptional(md, alloc, "physics_step_dt_s");
    const std::string policy_dt = LookupMetaOptional(md, alloc, "policy_step_dt_s");
    const std::string decimation = LookupMetaOptional(md, alloc, "control_decimation");
    const std::string qdes_limits =
        LookupMetaOptional(md, alloc, "qdes_joint_pos_limits");
    const std::string racket_control_point =
        LookupMetaOptional(md, alloc, "racket_control_point_offset_wrist_m");

    if (!training_contract_schema_version_.empty() &&
        training_contract_schema_version_ != "1" &&
        training_contract_schema_version_ != "2" &&
        training_contract_schema_version_ != "3")
      throw std::runtime_error(
          "ONNX training_contract_schema_version must be one of diagnostic 1/2 or formal 3");
    if (!training_contract_sha256_.empty() &&
        !IsLowerHexSha256(training_contract_sha256_))
      throw std::runtime_error(
          "ONNX training_contract_sha256 must contain exactly 64 lowercase hex characters");
    if (!source_checkpoint_sha256_.empty() &&
        !IsLowerHexSha256(source_checkpoint_sha256_))
      throw std::runtime_error(
          "ONNX source_checkpoint_sha256 must contain exactly 64 lowercase hex characters");
    if (!qdes_clamp.empty() && qdes_clamp != "0" && qdes_clamp != "1")
      throw std::runtime_error("ONNX qdes_clamp metadata must be 0|1");
    if (!physics_dt.empty()) physics_step_dt_s_ = ParsePositiveFinite(physics_dt, "physics_step_dt_s");
    if (!policy_dt.empty()) policy_step_dt_s_ = ParsePositiveFinite(policy_dt, "policy_step_dt_s");
    if (!decimation.empty())
      control_decimation_ = ParseCanonicalPositiveInt(decimation, "control_decimation");
    if (!qdes_limits.empty()) {
      const Eigen::VectorXd flat = ToFiniteVecStrict(qdes_limits, "qdes_joint_pos_limits");
      if (flat.size() != 2 * kNumJoints)
        throw std::runtime_error(
            "ONNX qdes_joint_pos_limits must contain 62 values in Isaac/action order");
      qdes_soft_lo_.resize(kNumJoints);
      qdes_soft_hi_.resize(kNumJoints);
      for (int i = 0; i < kNumJoints; ++i) {
        const double lo = flat[2 * i];
        const double hi = flat[2 * i + 1];
        if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
          throw std::runtime_error(
              "ONNX qdes_joint_pos_limits contains a non-finite or non-increasing pair at '" +
              joint_names_[i] + "'");
        if (default_q_[i] < lo || default_q_[i] > hi)
          throw std::runtime_error(
              "ONNX default_joint_pos lies outside qdes_joint_pos_limits at '" +
              joint_names_[i] + "'");
        qdes_soft_lo_[i] = lo;
        qdes_soft_hi_[i] = hi;
      }
    }
    bool racket_control_point_matches_runtime = false;
    if (!racket_control_point.empty()) {
      const Eigen::VectorXd recorded = ToFiniteVecStrict(
          racket_control_point, "racket_control_point_offset_wrist_m");
      if (recorded.size() != 3)
        throw std::runtime_error(
            "ONNX racket_control_point_offset_wrist_m must contain exactly three values");
      const Vec3 expected = racket_control_point_offset_wrist_m();
      constexpr double kControlPointToleranceM = 1e-9;
      if ((recorded - expected).cwiseAbs().maxCoeff() > kControlPointToleranceM)
        throw std::runtime_error(
            "ONNX racket_control_point_offset_wrist_m does not match the C++ FK control point "
            "(pingpang_red_Link origin); refusing a mixed wrist/racket contract");
      racket_control_point_matches_runtime = true;
    }
    if (std::isfinite(physics_step_dt_s_) && std::isfinite(policy_step_dt_s_) &&
        control_decimation_ > 0) {
      const double reconstructed =
          physics_step_dt_s_ * static_cast<double>(control_decimation_);
      if (std::fabs(policy_step_dt_s_ - reconstructed) > 1e-12)
        throw std::runtime_error(
            "ONNX policy_step_dt_s does not equal physics_step_dt_s * control_decimation");
    }

    const bool schema3_complete =
        training_contract_exact == "1" &&
        training_contract_schema_version_ == "3" &&
        IsLowerHexSha256(training_contract_sha256_) &&
        IsLowerHexSha256(source_checkpoint_sha256_) &&
        qdes_clamp == "1" && std::isfinite(physics_step_dt_s_) &&
        std::isfinite(policy_step_dt_s_) && control_decimation_ > 0 &&
        qdes_soft_lo_.size() == kNumJoints && racket_control_point_matches_runtime;
    if (formal_contract && !allow_legacy_model_diagnostic && !schema3_complete)
      throw std::runtime_error(
          "publish-capable schema-v2 ONNX requires an exact schema-v3 training contract: "
          "training_contract_schema_version=3, valid contract/checkpoint SHA-256 values, "
          "qdes_clamp=1 with 31 soft limit pairs, self-consistent execution timing, and an "
          "FK-matched racket_control_point_offset_wrist_m");
    has_schema3_execution_contract_ = schema3_complete;
    if (allow_legacy_model_diagnostic && formal_contract && !schema3_complete) {
      std::fprintf(stderr,
          "[pp DIAGNOSTIC] schema-v2 ONNX lacks a complete schema-v3 execution contract; "
          "hardware publication is not eligible.\n");
    }
    const std::string effort_s = LookupMetaOptional(md, alloc, "joint_effort_limits");
    if (!effort_s.empty()) effort_limits_ = ToVec(effort_s);
    if (effort_limits_.size() != kNumJoints) {
      if (!allow_legacy_model_diagnostic)
        throw std::runtime_error(
            "publish-capable ONNX requires 31 joint_effort_limits so the runner can "
            "reject commands whose requested PD effort exceeds the trained actuator envelope");
      effort_limits_ = Eigen::VectorXd();
      std::fprintf(stderr,
          "[pp DIAGNOSTIC] ONNX lacks a 31-joint effort envelope; allowed only by explicit "
          "legacy-model diagnostic under no-publish.\n");
    } else {
      for (int i = 0; i < effort_limits_.size(); ++i)
        if (!std::isfinite(effort_limits_[i]) || effort_limits_[i] <= 0.0)
          throw std::runtime_error(
              "ONNX joint_effort_limits contains a non-positive/non-finite value");
    }

    if (formal_contract) {
      std::string expected_contract;
      std::string expected_mode;
      std::vector<std::string> expected_names;
      std::vector<int> expected_dims;
      if (obs_dim_ == kObsDim110) {
        expected_contract = "hitter_pure";
        expected_mode = "hitter_pure";
        expected_names = {
            "base_ang_vel", "joint_pos", "joint_vel", "actions", "projected_gravity",
            "base_forward_xy", "base_target_delta_xy", "racket_target_rel_base",
            "racket_target_vel_w", "time_to_strike"};
        expected_dims = {3, 31, 31, 31, 3, 2, 2, 3, 3, 1};
      } else if (obs_dim_ == kObsDim175) {
        expected_contract = "deploy_parity";
        expected_mode = "deploy_parity";
        expected_names = {
            "command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel",
            "actions", "projected_gravity", "racket_target_pos_b", "racket_target_vel_w",
            "time_to_strike", "swing_type"};
        expected_dims = {62, 6, 3, 31, 31, 31, 3, 3, 3, 1, 1};
      } else if (obs_dim_ == kObsDim179) {
        expected_contract = "deploy_parity_face179";
        expected_mode = "deploy_parity";
        expected_names = {
            "command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel",
            "actions", "projected_gravity", "racket_target_pos_b", "racket_target_vel_w",
            "time_to_strike", "swing_type", "racket_target_normal_cmd"};
        expected_dims = {62, 6, 3, 31, 31, 31, 3, 3, 3, 1, 1, 4};
      } else if (obs_dim_ == kObsDim177) {
        expected_contract = "hitter_footwork";
        expected_mode = "hitter_footwork";
        expected_names = {
            "command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel",
            "actions", "projected_gravity", "base_target_pos_b", "racket_target_pos_b",
            "racket_target_vel_w", "time_to_strike", "swing_type"};
        expected_dims = {62, 6, 3, 31, 31, 31, 3, 2, 3, 3, 1, 1};
      } else if (obs_dim_ == kObsDim) {
        expected_contract = "full";
        expected_mode = "full";
        expected_names = {
            "command", "motion_anchor_pos_b", "motion_anchor_ori_b", "base_ang_vel",
            "joint_pos", "joint_vel", "actions", "projected_gravity", "base_target_pos_b",
            "racket_target_pos_b", "racket_target_vel_w", "time_to_strike", "swing_type"};
        expected_dims = {62, 3, 6, 3, 31, 31, 31, 3, 2, 3, 3, 1, 1};
      }
      const std::string contract = LookupMeta(md, alloc, "actor_obs_contract");
      const std::string mode = LookupMeta(md, alloc, "actor_obs_mode");
      const std::string total = LookupMeta(md, alloc, "actor_obs_total_dim");
      const std::vector<std::string> names =
          SplitCsv(LookupMeta(md, alloc, "observation_names"));
      const Eigen::VectorXd dims = ToVec(LookupMeta(md, alloc, "actor_obs_term_dims"));
      bool dims_match = dims.size() == static_cast<int>(expected_dims.size());
      for (int i = 0; dims_match && i < dims.size(); ++i)
        dims_match = std::isfinite(dims[i]) &&
                     std::fabs(dims[i] - std::round(dims[i])) < 1e-9 &&
                     static_cast<int>(std::lround(dims[i])) ==
                         expected_dims[static_cast<size_t>(i)];
      if (expected_names.empty() || contract != expected_contract || mode != expected_mode ||
          total != std::to_string(obs_dim_) ||
          names != expected_names || !dims_match)
        throw std::runtime_error(
            "formal ONNX observation layout metadata does not match the exact runner contract; "
            "refusing a right-width/wrong-column deployment");
      if (obs_dim_ == kObsDim179) {
        face_normal_envelope_ = ValidatePpFace179MetadataContract(PpFace179MetadataContract{
            LookupMeta(md, alloc, "face_command_enabled"),
            LookupMeta(md, alloc, "face_command_pairing"),
            LookupMeta(md, alloc, "face_obs_convention"),
            LookupMeta(md, alloc, "stage1_question_bank_exact"),
            LookupMeta(md, alloc, "stage1_bank_schema_version"),
            LookupMeta(md, alloc, "stage1_bank_split"),
            LookupMeta(md, alloc, "stage1_train_bank_sha256"),
            LookupMeta(md, alloc, "stage1_source_family_sha256"),
            LookupMeta(md, alloc, "mount_normal_sign_per_clip"),
            PpFaceNormalEnvelopeMetadata{
                LookupMeta(md, alloc, "stage1_normal_envelope_schema_version"),
                LookupMeta(md, alloc, "stage1_normal_envelope_frame"),
                LookupMeta(md, alloc, "stage1_normal_envelope_face_convention"),
                LookupMeta(md, alloc, "stage1_normal_envelope_pairing"),
                LookupMeta(md, alloc, "stage1_normal_envelope_algorithm"),
                LookupMeta(md, alloc, "stage1_normal_envelope_bank_row_unit_tolerance"),
                LookupMeta(md, alloc, "stage1_normal_envelope_runtime_unit_tolerance"),
                LookupMeta(md, alloc, "stage1_normal_envelope_runtime_dot_tolerance"),
                LookupMeta(md, alloc, "stage1_normal_envelope_clip_order"),
                LookupMeta(md, alloc, "stage1_normal_envelope_mount_normal_sign_per_clip"),
                LookupMeta(md, alloc, "stage1_normal_envelope_centers"),
                LookupMeta(md, alloc, "stage1_normal_envelope_reference_normals"),
                LookupMeta(md, alloc, "stage1_normal_envelope_min_dots"),
                LookupMeta(md, alloc, "stage1_normal_envelope_row_counts"),
                LookupMeta(md, alloc, "stage1_normal_envelope_train_bank_sha256"),
                LookupMeta(md, alloc, "stage1_normal_envelope_source_family_sha256"),
                LookupMeta(md, alloc, "stage1_normal_envelope_payload_sha256")}});
      }
    }

    if (formal_contract) {
      static const std::vector<std::string> expected_bodies = {
          "pelvis_link", "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
          "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link", "torso_Link",
          "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
          "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link"};
      const std::string anchor = LookupMeta(md, alloc, "anchor_body_name");
      if (body_names_ != expected_bodies || anchor != "torso_Link")
        throw std::runtime_error(
            "formal ONNX body_names/anchor do not match the runner's exact 14-body order "
            "(torso_Link must be anchor index 7)");
    }

    // ZERO-GAIN GUARD. These gains go verbatim into the published RobotCommand; a non-positive
    // kp/kd means that joint receives NO PD torque (limp). The 2026-07-02 explicitpd_ft export
    // baked kp=kd=0 for all IdealPD (arms/waist/ankle) joints because the exporter read PhysX
    // drive gains, which explicit actuators null -> the free-base robot collapsed. Fail fast
    // instead of silently deploying a torqueless robot.
    for (int i = 0; i < kNumJoints; ++i) {
      if (!std::isfinite(default_q_[i]) || !std::isfinite(action_scale_[i]) ||
          !std::isfinite(kp_[i]) || !std::isfinite(kd_[i]))
        throw std::runtime_error("ONNX joint metadata contains a non-finite value at '" +
                                 joint_names_[i] + "'");
      if (action_scale_[i] <= 0.0)
        throw std::runtime_error("ONNX action_scale must be positive for joint '" +
                                 joint_names_[i] + "'");
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
      if (seg.size() != pha.size() || seg.size() < 1)
        throw std::runtime_error("ONNX clip layout lengths/phases have incompatible sizes");
      for (int i = 0; i < seg.size(); ++i) {
        if (!std::isfinite(seg[i]) || seg[i] <= 0.0 ||
            std::fabs(seg[i] - std::round(seg[i])) > 1e-9 ||
            !std::isfinite(pha[i]) || pha[i] < 0.0 || pha[i] > 1.0)
          throw std::runtime_error("ONNX clip layout contains invalid length/phase");
      }
      clip_seg_lengths_.assign(seg.data(), seg.data() + seg.size());
      clip_strike_phases_.assign(pha.data(), pha.data() + pha.size());
    }
    if (formal_contract && clip_seg_lengths_.size() != 2)
      throw std::runtime_error("formal ONNX requires exactly two validated clip layouts");

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

    // OPTIONAL hitter_pure sampling geometry (110-D models, 2026-07-07). Baked by
    // utils/exporter.py when the training task ran target_mode=hitter_pure:
    //   hitter_pure_pos_range_per_clip: "x_lo,x_hi,y_lo,y_hi,z_lo,z_hi;..." per clip,
    //     STATION-RELATIVE x/y (x degenerate = the fixed striking plane, y = the swing
    //     band), z ABSOLUTE above the floor.
    //   hitter_pure_vel_range_per_clip: same 6-tuple format, world-frame velocity box.
    //   hitter_pure_base_target_range: "x_lo,x_hi,y_lo,y_hi" — the independent trained
    //     station box around the spawn (informational; deploy stations come from balls).
    // PpPolicy derives the station from a racket target as
    //   station_xy = target_xy − (x_plane, y_band_center)[clip]
    // and gates engage against the trained z/vel envelopes.
    const std::string hp_pos_s = LookupMetaOptional(md, alloc, "hitter_pure_pos_range_per_clip");
    const std::string hp_vel_s = LookupMetaOptional(md, alloc, "hitter_pure_vel_range_per_clip");
    const std::string hp_base_s = LookupMetaOptional(md, alloc, "hitter_pure_base_target_range");
    auto parse_boxes = [](const std::string& s, std::vector<std::array<double, 6>>& out) {
      std::stringstream ss(s);
      std::string clip;
      while (std::getline(ss, clip, ';')) {
        const Eigen::VectorXd v = ToVec(clip);
        if (v.size() == 6) out.push_back({v[0], v[1], v[2], v[3], v[4], v[5]});
      }
    };
    if (!hp_pos_s.empty()) parse_boxes(hp_pos_s, hp_pos_boxes_);
    if (!hp_vel_s.empty()) parse_boxes(hp_vel_s, hp_vel_boxes_);
    if (!hp_base_s.empty()) {
      const Eigen::VectorXd v = ToVec(hp_base_s);
      if (v.size() != 4)
        throw std::runtime_error("hitter_pure_base_target_range must contain exactly 4 values");
      hp_base_range_ = {v[0], v[1], v[2], v[3]};
    }
    if (obs_dim_ == kObsDim110) {
      if (hp_pos_boxes_.size() != 2 || hp_vel_boxes_.size() != 2)
        throw std::runtime_error(
            "formal 110-D ONNX requires exactly two position and velocity boxes");
      auto valid_box = [](const std::array<double, 6>& b) {
        for (double x : b)
          if (!std::isfinite(x)) return false;
        return b[0] <= b[1] && b[2] <= b[3] && b[4] <= b[5];
      };
      if (!valid_box(hp_pos_boxes_[0]) || !valid_box(hp_pos_boxes_[1]) ||
          !valid_box(hp_vel_boxes_[0]) || !valid_box(hp_vel_boxes_[1]))
        throw std::runtime_error("formal 110-D ONNX contains a non-finite/inverted box");
      if (hp_base_s.empty() || !std::all_of(hp_base_range_.begin(), hp_base_range_.end(),
                                            [](double x) { return std::isfinite(x); }) ||
          hp_base_range_[0] > hp_base_range_[1] || hp_base_range_[2] > hp_base_range_[3])
        throw std::runtime_error(
            "formal 110-D ONNX requires a finite, ordered hitter_pure_base_target_range");
      constexpr double kPlaneTolerance = 1e-6;
      if (std::fabs(hp_pos_boxes_[0][0] - hp_pos_boxes_[0][1]) > kPlaneTolerance ||
          std::fabs(hp_pos_boxes_[1][0] - hp_pos_boxes_[1][1]) > kPlaneTolerance ||
          std::fabs(hp_pos_boxes_[0][0] - hp_pos_boxes_[1][0]) > kPlaneTolerance)
        throw std::runtime_error(
            "formal 110-D ONNX position boxes must share one degenerate fixed strike plane");
    }
    publishable_model_contract_ =
        formal_contract && empirical_norm == "1" && trained_norm == "1" &&
        obs_norm_baked == "1" && schema3_complete &&
        effort_limits_.size() == kNumJoints;
    if (!allow_legacy_model_diagnostic && !publishable_model_contract_)
      throw std::runtime_error(
          "publish-capable model contract did not survive complete parsed-metadata validation");
  }

  int obs_dim() const { return obs_dim_; }
  // Per-clip layout baked by new exports (empty on legacy models -> caller keeps its default).
  bool has_clip_layout() const { return !clip_seg_lengths_.empty(); }
  const std::vector<double>& clip_seg_lengths() const { return clip_seg_lengths_; }
  const std::vector<double>& clip_strike_phases() const { return clip_strike_phases_; }
  // Per-clip reference reach offsets (empty when the export predates the metadata key).
  bool has_reach_offsets() const { return !reach_offsets_.empty(); }
  const std::vector<Vec2>& reach_offsets() const { return reach_offsets_; }
  // hitter_pure sampling geometry (empty on non-pure exports). Box layout per clip:
  // {x_lo, x_hi, y_lo, y_hi, z_lo, z_hi} — pos boxes station-relative x/y + absolute z.
  bool has_hitter_pure_boxes() const { return !hp_pos_boxes_.empty(); }
  const std::vector<std::array<double, 6>>& hp_pos_boxes() const { return hp_pos_boxes_; }
  const std::vector<std::array<double, 6>>& hp_vel_boxes() const { return hp_vel_boxes_; }
  const std::array<double, 4>& hp_base_range() const { return hp_base_range_; }

  // Reference base->racket reach offset at a given time_step, computed from the baked refs:
  // blade world xy (ref pelvis pose + racket FK on the ref joints) minus ref pelvis xy. Same
  // arithmetic as training's _ensure_reference_strike_state — the fallback for 177 models
  // whose export predates the ref_reach_offset_xy metadata key.
  Vec2 reach_offset_from_refs(int time_step) {
    auto out = Run(Eigen::VectorXd::Zero(obs_dim_), time_step,
                   {"joint_pos", "body_pos_w", "body_quat_w"});
    const Eigen::VectorXd ref_q = Map(out[0], kNumJoints);
    RequireShape(out[1], {1, 14, 3}, "body_pos_w");
    RequireShape(out[2], {1, 14, 4}, "body_quat_w");
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
  const Eigen::VectorXd& effort_limits() const { return effort_limits_; }
  bool has_schema3_execution_contract() const { return has_schema3_execution_contract_; }
  bool training_contract_exact() const { return training_contract_exact_; }
  bool publishable_model_contract() const { return publishable_model_contract_; }
  double physics_step_dt_s() const { return physics_step_dt_s_; }
  double policy_step_dt_s() const { return policy_step_dt_s_; }
  int control_decimation() const { return control_decimation_; }
  const std::string& training_contract_sha256() const {
    return training_contract_sha256_;
  }
  const std::string& source_checkpoint_sha256() const {
    return source_checkpoint_sha256_;
  }
  const Eigen::VectorXd& qdes_soft_lo() const { return qdes_soft_lo_; }
  const Eigen::VectorXd& qdes_soft_hi() const { return qdes_soft_hi_; }
  const PpFaceNormalEnvelope& face_normal_envelope() const {
    return face_normal_envelope_;
  }
  bool face_normal_within_training_envelope(int clip, const Vec3& normal_w) const {
    return obs_dim_ == kObsDim179 &&
           face_normal_envelope_.AllowsRawA(
               clip, normal_w[0], normal_w[1], normal_w[2]);
  }
  Vec3 face_normal_raw_a_from_wire_b(int clip, const Vec3& normal_wire_b) const {
    if (obs_dim_ != kObsDim179)
      throw std::logic_error("physical-B to raw-A face conversion is 179-D only");
    const auto converted = face_normal_envelope_.WireBToRawA(
        clip, normal_wire_b[0], normal_wire_b[1], normal_wire_b[2]);
    return Vec3(converted[0], converted[1], converted[2]);
  }

  // Run the actor once before the backend/driver starts. ONNX Runtime may lazily allocate and
  // compile kernels on the first actor invocation; doing that in the publish callback can trip
  // the command deadline despite healthy steady-state inference.
  void prewarm_actor() {
    const Eigen::VectorXd action = mean_action(Eigen::VectorXd::Zero(obs_dim_), 0);
    if (action.size() != kNumJoints || !action.allFinite())
      throw std::runtime_error("ONNX actor prewarm returned an invalid action");
  }

  // Reference motion at time_step (obs-independent side-outputs).
  PpRefs refs(int time_step) {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(obs_dim_);
    auto out = Run(zero, time_step, {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w"});
    PpRefs r;
    r.joint_pos = Map(out[0], kNumJoints);
    r.joint_vel = Map(out[1], kNumJoints);
    RequireShape(out[2], {1, 14, 3}, "body_pos_w");
    RequireShape(out[3], {1, 14, 4}, "body_quat_w");
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
    if (action.size() != kNumJoints)
      throw std::runtime_error("ONNX action output size != 31");
    Eigen::VectorXd target = default_q_ + action.cwiseProduct(action_scale_);
    // Strategy clamp: reproduce ClampedJointPositionAction's schema-v3 *soft* limits in the
    // actor/action order.  The runner's SDK-order hard clamp remains a separate outer safety
    // envelope and therefore also protects PD_STAND/reference-playback/non-policy commands.
    if (qdes_soft_lo_.size() == kNumJoints && qdes_soft_hi_.size() == kNumJoints) {
      target = target.cwiseMax(qdes_soft_lo_).cwiseMin(qdes_soft_hi_);
    }
    return target;
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
    if (!v.IsTensor()) throw std::runtime_error("ONNX output is not a tensor");
    const auto info = v.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != static_cast<size_t>(n))
      throw std::runtime_error("ONNX output tensor type/element count violates the contract");
    const float* p = v.GetTensorData<float>();
    Eigen::VectorXd out(n);
    for (int i = 0; i < n; ++i) out[i] = p[i];
    return out;
  }
  static void RequireShape(const Ort::Value& v, std::initializer_list<int64_t> expected,
                           const char* name) {
    if (!v.IsTensor())
      throw std::runtime_error(std::string("ONNX output '") + name + "' is not a tensor");
    const auto info = v.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> want(expected);
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape() != want)
      throw std::runtime_error(std::string("ONNX output '") + name +
                               "' has the wrong type/shape");
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
  static Eigen::VectorXd ToFiniteVecStrict(const std::string& value, const char* key) {
    const auto tokens = SplitCsv(value);
    Eigen::VectorXd parsed(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
      size_t consumed = 0;
      try {
        parsed[static_cast<int>(i)] = std::stod(tokens[i], &consumed);
      } catch (const std::exception&) {
        throw std::runtime_error(std::string("ONNX ") + key +
                                 " contains a non-numeric value");
      }
      if (consumed != tokens[i].size() ||
          !std::isfinite(parsed[static_cast<int>(i)]))
        throw std::runtime_error(std::string("ONNX ") + key +
                                 " contains a malformed or non-finite value");
    }
    return parsed;
  }
  static bool IsLowerHexSha256(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
  }
  static double ParsePositiveFinite(const std::string& value, const char* key) {
    size_t consumed = 0;
    double parsed = 0.0;
    try {
      parsed = std::stod(value, &consumed);
    } catch (const std::exception&) {
      throw std::runtime_error(std::string("ONNX ") + key +
                               " metadata is not a finite positive number");
    }
    if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0)
      throw std::runtime_error(std::string("ONNX ") + key +
                               " metadata is not a finite positive number");
    return parsed;
  }
  static int ParseCanonicalPositiveInt(const std::string& value, const char* key) {
    size_t consumed = 0;
    long parsed = 0;
    try {
      parsed = std::stol(value, &consumed, 10);
    } catch (const std::exception&) {
      throw std::runtime_error(std::string("ONNX ") + key +
                               " metadata is not a canonical positive integer");
    }
    if (consumed != value.size() || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max() || value != std::to_string(parsed))
      throw std::runtime_error(std::string("ONNX ") + key +
                               " metadata is not a canonical positive integer");
    return static_cast<int>(parsed);
  }

  Ort::Env env_;
  Ort::MemoryInfo mem_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> joint_names_, body_names_;
  std::vector<double> clip_seg_lengths_, clip_strike_phases_;  // empty on legacy exports
  std::vector<Vec2> reach_offsets_;  // per-clip ref base->racket reach; empty on old exports
  std::vector<std::array<double, 6>> hp_pos_boxes_, hp_vel_boxes_;  // hitter_pure geometry
  std::array<double, 4> hp_base_range_ = {0.0, 0.0, 0.0, 0.0};
  Eigen::VectorXd default_q_, action_scale_, kp_, kd_, effort_limits_;
  Eigen::VectorXd qdes_soft_lo_, qdes_soft_hi_;  // schema-v3 Isaac/action-order soft limits
  std::string training_contract_schema_version_;
  std::string training_contract_sha256_;
  std::string source_checkpoint_sha256_;
  double physics_step_dt_s_ = std::numeric_limits<double>::quiet_NaN();
  double policy_step_dt_s_ = std::numeric_limits<double>::quiet_NaN();
  int control_decimation_ = 0;
  bool has_schema3_execution_contract_ = false;
  bool training_contract_exact_ = false;
  bool publishable_model_contract_ = false;
  PpFaceNormalEnvelope face_normal_envelope_{};  // mandatory and content-bound for 179-D only
  std::vector<float> obs_f_;
  int obs_dim_ = kObsDim;  // detected from the model input at load (180 full / 175 deploy_parity)
};

}  // namespace a3_pingpong
