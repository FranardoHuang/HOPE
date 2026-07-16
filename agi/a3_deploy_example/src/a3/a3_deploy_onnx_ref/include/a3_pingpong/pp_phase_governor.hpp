#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "a3_pingpong/pp_frame_math.hpp"

namespace a3_pingpong {

// C++ mirror of training's phase_governor_v1.  The task identity and target
// tuple may be revised while the wind-up is active, but phase never moves
// backwards and neither phase rate nor phase acceleration can jump outside the
// immutable model-bound profile. Revision envelopes remain relative to the
// consumer's begin snapshot: deployment uses a latest-value mailbox, so an
// adjacent-revision allowance would be transport-order dependent and could
// ratchet without limit.
struct PpPhaseGovernorProfile {
  std::string contract_version;
  int schema_version = 0;
  std::string profile_sha256;
  double policy_dt_s = 0.0;
  double min_tts_s = 0.0;
  double max_tts_s = 0.0;
  double max_phase_rate_per_s = 0.0;
  double max_phase_acceleration_per_s2 = 0.0;
  double max_deadline_revision_delta_s = 0.0;
  double max_position_revision_delta_m = 0.0;
  double max_velocity_revision_delta_mps = 0.0;
  double max_normal_revision_delta_rad = 0.0;
  double normal_unit_tolerance = 0.0;
  double early_deadline_tolerance_s = 0.0;

  bool Valid() const {
    return contract_version == "phase_governor_v1" && schema_version == 1 &&
        profile_sha256.size() == 64 &&
        std::all_of(profile_sha256.begin(), profile_sha256.end(), [](char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }) &&
        std::isfinite(policy_dt_s) && policy_dt_s > 0.0 &&
        std::isfinite(min_tts_s) && min_tts_s > 0.0 &&
        std::isfinite(max_tts_s) && max_tts_s > min_tts_s &&
        std::isfinite(max_phase_rate_per_s) && max_phase_rate_per_s > 0.0 &&
        std::isfinite(max_phase_acceleration_per_s2) &&
        max_phase_acceleration_per_s2 > 0.0 &&
        std::isfinite(max_deadline_revision_delta_s) &&
        max_deadline_revision_delta_s >= 0.0 &&
        std::isfinite(max_position_revision_delta_m) &&
        max_position_revision_delta_m >= 0.0 &&
        std::isfinite(max_velocity_revision_delta_mps) &&
        max_velocity_revision_delta_mps >= 0.0 &&
        std::isfinite(max_normal_revision_delta_rad) &&
        max_normal_revision_delta_rad >= 0.0 &&
        max_normal_revision_delta_rad <= 3.14159265358979323846 &&
        std::isfinite(normal_unit_tolerance) &&
        normal_unit_tolerance >= 0.0 && normal_unit_tolerance < 1.0 &&
        std::isfinite(early_deadline_tolerance_s) &&
        early_deadline_tolerance_s >= 0.0;
  }
};

// The model contract and runtime transport switch are a two-key cutover. A
// revision-trained model must never execute behind the legacy schema-3 path,
// and a legacy model must never be fed live task revisions it was not trained
// to consume.
inline void RequirePpPlannerTaskRevisionDoubleKey(bool model_trained,
                                                  bool runtime_enabled) {
  if (model_trained != runtime_enabled) {
    throw std::runtime_error(
        "pingpong: planner task revision model metadata and runtime flag must be "
        "enabled or disabled together");
  }
}

struct PpPhaseRevision {
  std::uint64_t control_epoch = 0;
  std::uint64_t task_id = 0;
  std::uint64_t task_revision = 0;
  std::uint64_t command_sequence = 0;
  double source_monotonic_s = -1.0;
  Vec3 target_position_m = Vec3::Zero();
  Vec3 target_velocity_mps = Vec3::Zero();
  Vec3 target_normal = Vec3(1.0, 0.0, 0.0);
  double desired_tts_s = 0.0;
};

enum class PpPhaseDecision {
  kAccepted,
  kInactive,
  kPostContact,
  kMalformed,
  kDifferentTask,
  kOldOrDuplicate,
  kTargetEnvelope,
  kDeadlineEnvelope,
  kUnreachableDeadline,
};

class PpPhaseGovernor {
 public:
  explicit PpPhaseGovernor(PpPhaseGovernorProfile profile)
      : profile_(std::move(profile)) {
    if (!profile_.Valid())
      throw std::invalid_argument("invalid phase_governor_v1 profile");
  }

  static double MinimumFinishTime(double distance, double initial_rate,
                                  double maximum_rate,
                                  double maximum_acceleration) {
    if (distance <= 0.0) return 0.0;
    const double rate = std::min(maximum_rate, std::max(0.0, initial_rate));
    const double accelerate_time =
        std::max(0.0, (maximum_rate - rate) / maximum_acceleration);
    const double accelerate_distance = rate * accelerate_time +
        0.5 * maximum_acceleration * accelerate_time * accelerate_time;
    if (distance <= accelerate_distance) {
      return (-rate + std::sqrt(rate * rate +
              2.0 * maximum_acceleration * distance)) /
          maximum_acceleration;
    }
    return accelerate_time + (distance - accelerate_distance) / maximum_rate;
  }

  PpPhaseDecision Begin(const PpPhaseRevision& revision,
                        double local_monotonic_s) {
    if (revision.task_revision != 1)
      return PpPhaseDecision::kMalformed;
    return BeginConsumerSnapshot(revision, local_monotonic_s);
  }

  // Deployment consumes a latest-value mailbox.  If task N was already being
  // refined while task N-1 finished its follow-through, the first snapshot the
  // runner can act on may be revision 2+.  Producer-side Begin() still enforces
  // revision 1; this consumer adapter preserves the latest positive revision
  // as the ordering high-water mark while using identical phase dynamics.
  PpPhaseDecision BeginConsumerSnapshot(const PpPhaseRevision& revision,
                                        double local_monotonic_s) {
    if (active_) return PpPhaseDecision::kDifferentTask;
    if (!RevisionWellFormed_(revision) || !std::isfinite(local_monotonic_s) ||
        local_monotonic_s < 0.0)
      return PpPhaseDecision::kMalformed;
    const PpPhaseRevision canonical_revision = CanonicalizedRevision_(revision);
    const double minimum = MinimumFinishTime(
        1.0, 0.0, profile_.max_phase_rate_per_s,
        profile_.max_phase_acceleration_per_s2);
    if (canonical_revision.desired_tts_s +
            profile_.early_deadline_tolerance_s < minimum)
      return PpPhaseDecision::kUnreachableDeadline;
    active_ = true;
    control_epoch_ = revision.control_epoch;
    task_id_ = revision.task_id;
    phase_ = 0.0;
    phase_rate_per_s_ = 0.0;
    local_monotonic_s_ = local_monotonic_s;
    deadline_local_s_ =
        local_monotonic_s + canonical_revision.desired_tts_s;
    baseline_deadline_local_s_ = deadline_local_s_;
    baseline_revision_ = canonical_revision;
    visible_revision_ = canonical_revision;
    slow_only_next_step_ = false;
    return PpPhaseDecision::kAccepted;
  }

  PpPhaseDecision Revise(const PpPhaseRevision& revision) {
    if (!active_) return PpPhaseDecision::kInactive;
    if (phase_ >= 1.0) return PpPhaseDecision::kPostContact;
    if (!RevisionWellFormed_(revision)) return PpPhaseDecision::kMalformed;
    const PpPhaseRevision canonical_revision = CanonicalizedRevision_(revision);
    if (canonical_revision.control_epoch != control_epoch_ ||
        canonical_revision.task_id != task_id_)
      return PpPhaseDecision::kDifferentTask;
    if (canonical_revision.task_revision <= visible_revision_.task_revision ||
        canonical_revision.command_sequence <= visible_revision_.command_sequence ||
        canonical_revision.source_monotonic_s <= visible_revision_.source_monotonic_s)
      return PpPhaseDecision::kOldOrDuplicate;
    if ((canonical_revision.target_position_m -
             baseline_revision_.target_position_m).norm() >
            profile_.max_position_revision_delta_m ||
        (canonical_revision.target_velocity_mps -
             baseline_revision_.target_velocity_mps).norm() >
            profile_.max_velocity_revision_delta_mps ||
        NormalAngle_(canonical_revision.target_normal,
                     baseline_revision_.target_normal) >
            profile_.max_normal_revision_delta_rad)
      return PpPhaseDecision::kTargetEnvelope;
    const double proposed_deadline =
        local_monotonic_s_ + canonical_revision.desired_tts_s;
    const double baseline_deadline_delta =
        proposed_deadline - baseline_deadline_local_s_;
    if (std::fabs(baseline_deadline_delta) >
        profile_.max_deadline_revision_delta_s)
      return PpPhaseDecision::kDeadlineEnvelope;
    const double minimum = MinimumFinishTime(
        std::max(0.0, 1.0 - phase_), phase_rate_per_s_,
        profile_.max_phase_rate_per_s,
        profile_.max_phase_acceleration_per_s2);
    if (canonical_revision.desired_tts_s +
            profile_.early_deadline_tolerance_s < minimum)
      return PpPhaseDecision::kUnreachableDeadline;
    const double deadline_delta = proposed_deadline - deadline_local_s_;
    deadline_local_s_ = proposed_deadline;
    visible_revision_ = canonical_revision;
    slow_only_next_step_ =
        deadline_delta > profile_.early_deadline_tolerance_s;
    return PpPhaseDecision::kAccepted;
  }

  void Advance() {
    if (!active_) return;
    const double dt = profile_.policy_dt_s;
    if (phase_ >= 1.0) {
      phase_rate_per_s_ = std::max(
          0.0, phase_rate_per_s_ -
              profile_.max_phase_acceleration_per_s2 * dt);
      local_monotonic_s_ += dt;
      slow_only_next_step_ = false;
      return;
    }
    const double new_time = local_monotonic_s_ + dt;
    const double remaining_phase = 1.0 - phase_;
    const double remaining_time_after_step =
        std::max(deadline_local_s_ - new_time, 0.0);
    double requested_rate = profile_.max_phase_rate_per_s;
    if (remaining_time_after_step > profile_.early_deadline_tolerance_s) {
      requested_rate = std::min(
          profile_.max_phase_rate_per_s,
          remaining_phase / remaining_time_after_step);
      const double earliest = MinimumFinishTime(
          remaining_phase, phase_rate_per_s_,
          profile_.max_phase_rate_per_s,
          profile_.max_phase_acceleration_per_s2);
      if (remaining_time_after_step <= earliest + dt)
        requested_rate = profile_.max_phase_rate_per_s;
    }
    if (slow_only_next_step_)
      requested_rate = std::min(requested_rate, phase_rate_per_s_);
    const double maximum_rate_delta =
        profile_.max_phase_acceleration_per_s2 * dt;
    const double rate_delta = std::clamp(
        requested_rate - phase_rate_per_s_,
        -maximum_rate_delta, maximum_rate_delta);
    const double new_rate = std::clamp(
        phase_rate_per_s_ + rate_delta, 0.0,
        profile_.max_phase_rate_per_s);
    const double phase_delta =
        std::max(0.0, 0.5 * (phase_rate_per_s_ + new_rate) * dt);
    double new_phase = std::min(1.0, phase_ + phase_delta);
    if (remaining_time_after_step > profile_.early_deadline_tolerance_s) {
      // Reserve one complete actor interval before contact.  The next task
      // revision is consumed later in this same policy tick; reaching phase 1
      // here would close the exact final policy_dt update before the actor can
      // observe it.
      const double next_rate = std::min(
          profile_.max_phase_rate_per_s,
          new_rate + profile_.max_phase_acceleration_per_s2 * dt);
      const double reserved_distance =
          0.5 * (new_rate + next_rate) * dt;
      const double precontact_cap =
          std::max(phase_, 1.0 - reserved_distance);
      new_phase = std::min(new_phase, precontact_cap);
    }
    phase_ = new_phase;
    phase_rate_per_s_ = new_rate;
    local_monotonic_s_ = new_time;
    slow_only_next_step_ = false;
  }

  void Complete() {
    active_ = false;
    phase_ = 0.0;
    phase_rate_per_s_ = 0.0;
    slow_only_next_step_ = false;
  }

  const PpPhaseGovernorProfile& profile() const { return profile_; }
  bool active() const { return active_; }
  std::uint64_t control_epoch() const { return control_epoch_; }
  std::uint64_t task_id() const { return task_id_; }
  double phase() const { return phase_; }
  double phase_rate_per_s() const { return phase_rate_per_s_; }
  double local_monotonic_s() const { return local_monotonic_s_; }
  double deadline_local_s() const { return deadline_local_s_; }
  double remaining_tts_s() const {
    return active_ ? std::max(0.0, deadline_local_s_ - local_monotonic_s_) : 0.0;
  }
  const PpPhaseRevision& visible_revision() const { return visible_revision_; }

 private:
  PpPhaseRevision CanonicalizedRevision_(PpPhaseRevision revision) const {
    const double tolerance = profile_.early_deadline_tolerance_s;
    if (std::fabs(revision.desired_tts_s - profile_.min_tts_s) <= tolerance)
      revision.desired_tts_s = profile_.min_tts_s;
    else if (std::fabs(revision.desired_tts_s - profile_.max_tts_s) <= tolerance)
      revision.desired_tts_s = profile_.max_tts_s;
    return revision;
  }

  bool RevisionWellFormed_(const PpPhaseRevision& revision) const {
    const double normal_norm = revision.target_normal.norm();
    return revision.control_epoch > 0 && revision.task_id > 0 &&
        revision.task_revision > 0 && revision.command_sequence > 0 &&
        std::isfinite(revision.source_monotonic_s) &&
        revision.source_monotonic_s >= 0.0 &&
        revision.target_position_m.allFinite() &&
        revision.target_velocity_mps.allFinite() &&
        revision.target_normal.allFinite() &&
        std::isfinite(normal_norm) &&
        std::fabs(normal_norm - 1.0) <= profile_.normal_unit_tolerance &&
        std::isfinite(revision.desired_tts_s) &&
        revision.desired_tts_s + profile_.early_deadline_tolerance_s >=
            profile_.min_tts_s &&
        revision.desired_tts_s - profile_.early_deadline_tolerance_s <=
            profile_.max_tts_s;
  }

  static double NormalAngle_(const Vec3& a, const Vec3& b) {
    const double denom = a.norm() * b.norm();
    if (!std::isfinite(denom) || denom <= 0.0)
      return std::numeric_limits<double>::infinity();
    return std::acos(std::clamp(a.dot(b) / denom, -1.0, 1.0));
  }

  PpPhaseGovernorProfile profile_;
  bool active_ = false;
  std::uint64_t control_epoch_ = 0;
  std::uint64_t task_id_ = 0;
  double phase_ = 0.0;
  double phase_rate_per_s_ = 0.0;
  double local_monotonic_s_ = 0.0;
  double deadline_local_s_ = 0.0;
  double baseline_deadline_local_s_ = 0.0;
  PpPhaseRevision baseline_revision_{};
  PpPhaseRevision visible_revision_{};
  bool slow_only_next_step_ = false;
};

}  // namespace a3_pingpong
