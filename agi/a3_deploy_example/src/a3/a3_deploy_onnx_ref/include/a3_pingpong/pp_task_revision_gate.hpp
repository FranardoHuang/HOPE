#pragma once

#include <cstdint>

namespace a3_pingpong {

// Pure consumer-side exactly-once gate for schema-4 planner tasks.  It has no
// transport, policy, simulator, or robot dependencies and is therefore usable
// as a source gate before pp_policy.hpp is allowed to consume live revisions.
enum class PpTaskRevisionDecision {
  kEngaged,
  kRevisionAccepted,
  kTaskInvalidObserved,
  kGlobalRevoke,
  kDisarmed,
  kNoActiveTask,
  kAlreadyActive,
  kConsumed,
  kDifferentTask,
  kOldOrDuplicate,
  kSideOrClipChanged,
  kMalformed,
  kEpochRegressed,
};

struct PpTaskRevisionEnvelope {
  std::uint64_t control_epoch = 0;
  std::uint64_t task_id = 0;
  std::uint64_t task_revision = 0;
  double swing_sign = 0.0;
  int clip_id = -1;
};

class PpTaskRevisionGate {
 public:
  // An explicit rearm is required initially and after epoch/global revoke.
  // Rearming an active task is rejected so it cannot erase consumption state.
  bool Rearm(std::uint64_t epoch) {
    const auto relation = ObserveEpoch_(epoch);
    if (relation == EpochRelation::kRegressed || active_) return false;
    armed_ = true;
    return true;
  }

  // Same-epoch disarm preserves the consumed high-water mark.  A later rearm
  // therefore cannot replay the task that was already linearized at engage.
  bool Disarm(std::uint64_t epoch) {
    const auto relation = ObserveEpoch_(epoch);
    if (relation == EpochRelation::kRegressed) return false;
    armed_ = false;
    ClearActive_();
    return true;
  }

  PpTaskRevisionDecision TryEngage(const PpTaskRevisionEnvelope& in) {
    const auto relation = ObserveEpoch_(in.control_epoch);
    if (relation == EpochRelation::kRegressed)
      return PpTaskRevisionDecision::kEpochRegressed;
    if (relation == EpochRelation::kAdvanced || !armed_)
      return PpTaskRevisionDecision::kDisarmed;
    if (!ValidActiveEnvelope_(in))
      return PpTaskRevisionDecision::kMalformed;
    if (active_) {
      return in.task_id == active_task_id_
                 ? PpTaskRevisionDecision::kAlreadyActive
                 : PpTaskRevisionDecision::kDifferentTask;
    }
    if (in.task_id <= last_consumed_task_id_)
      return PpTaskRevisionDecision::kConsumed;
    // Exactly-once linearization point: consume before exposing ACTIVE.  An
    // abort/disarm after this point can never make the same task engageable.
    last_consumed_task_id_ = in.task_id;
    active_task_id_ = in.task_id;
    active_revision_ = in.task_revision;
    frozen_swing_sign_ = in.swing_sign;
    frozen_clip_id_ = in.clip_id;
    active_ = true;
    return PpTaskRevisionDecision::kEngaged;
  }

  PpTaskRevisionDecision TryRevision(const PpTaskRevisionEnvelope& in) {
    const auto relation = ObserveEpoch_(in.control_epoch);
    if (relation == EpochRelation::kRegressed)
      return PpTaskRevisionDecision::kEpochRegressed;
    if (relation == EpochRelation::kAdvanced || !armed_)
      return PpTaskRevisionDecision::kDisarmed;
    if (!ValidActiveEnvelope_(in))
      return PpTaskRevisionDecision::kMalformed;
    if (!active_) {
      return in.task_id <= last_consumed_task_id_
                 ? PpTaskRevisionDecision::kConsumed
                 : PpTaskRevisionDecision::kNoActiveTask;
    }
    if (in.task_id != active_task_id_)
      return PpTaskRevisionDecision::kDifferentTask;
    if (in.swing_sign != frozen_swing_sign_ || in.clip_id != frozen_clip_id_)
      return PpTaskRevisionDecision::kSideOrClipChanged;
    if (in.task_revision <= active_revision_)
      return PpTaskRevisionDecision::kOldOrDuplicate;
    active_revision_ = in.task_revision;
    return PpTaskRevisionDecision::kRevisionAccepted;
  }

  // A positive pair is a task-scoped invalid refinement: advance its ordering
  // but keep the last good active target frozen.  Zero/zero is the only
  // anonymous global revoke and disarms immediately.  A mixed pair is invalid.
  PpTaskRevisionDecision ObserveInvalid(std::uint64_t epoch,
                                        std::uint64_t task_id,
                                        std::uint64_t task_revision) {
    const auto relation = ObserveEpoch_(epoch);
    if (relation == EpochRelation::kRegressed)
      return PpTaskRevisionDecision::kEpochRegressed;
    if ((task_id == 0) != (task_revision == 0))
      return PpTaskRevisionDecision::kMalformed;
    if (task_id == 0) {
      armed_ = false;
      ClearActive_();
      return PpTaskRevisionDecision::kGlobalRevoke;
    }
    if (!armed_ || relation == EpochRelation::kAdvanced)
      return PpTaskRevisionDecision::kDisarmed;
    if (!active_) {
      return task_id <= last_consumed_task_id_
                 ? PpTaskRevisionDecision::kConsumed
                 : PpTaskRevisionDecision::kNoActiveTask;
    }
    if (task_id != active_task_id_)
      return PpTaskRevisionDecision::kDifferentTask;
    if (task_revision <= active_revision_)
      return PpTaskRevisionDecision::kOldOrDuplicate;
    active_revision_ = task_revision;
    return PpTaskRevisionDecision::kTaskInvalidObserved;
  }

  // Completion leaves the session armed for a newer task, but the consumed
  // high-water mark survives.  Repeated completion or a different task fails.
  bool Complete(std::uint64_t epoch, std::uint64_t task_id) {
    const auto relation = ObserveEpoch_(epoch);
    if (relation != EpochRelation::kSame || !active_ ||
        task_id != active_task_id_)
      return false;
    ClearActive_();
    return true;
  }

  bool epoch_initialized() const { return epoch_initialized_; }
  std::uint64_t control_epoch() const { return control_epoch_; }
  bool armed() const { return armed_; }
  bool active() const { return active_; }
  std::uint64_t last_consumed_task_id() const {
    return last_consumed_task_id_;
  }
  std::uint64_t active_task_id() const { return active_task_id_; }
  std::uint64_t active_revision() const { return active_revision_; }
  double frozen_swing_sign() const { return frozen_swing_sign_; }
  int frozen_clip_id() const { return frozen_clip_id_; }

 private:
  enum class EpochRelation { kSame, kAdvanced, kRegressed };

  EpochRelation ObserveEpoch_(std::uint64_t epoch) {
    if (!epoch_initialized_ || epoch > control_epoch_) {
      epoch_initialized_ = true;
      control_epoch_ = epoch;
      armed_ = false;
      last_consumed_task_id_ = 0;
      ClearActive_();
      return EpochRelation::kAdvanced;
    }
    if (epoch < control_epoch_) return EpochRelation::kRegressed;
    return EpochRelation::kSame;
  }

  static bool ValidActiveEnvelope_(const PpTaskRevisionEnvelope& in) {
    return in.task_id > 0 && in.task_revision > 0 &&
           (in.swing_sign == -1.0 || in.swing_sign == 1.0) && in.clip_id >= 0;
  }

  void ClearActive_() {
    active_ = false;
    active_task_id_ = 0;
    active_revision_ = 0;
    frozen_swing_sign_ = 0.0;
    frozen_clip_id_ = -1;
  }

  bool epoch_initialized_ = false;
  std::uint64_t control_epoch_ = 0;
  bool armed_ = false;
  bool active_ = false;
  std::uint64_t last_consumed_task_id_ = 0;
  std::uint64_t active_task_id_ = 0;
  std::uint64_t active_revision_ = 0;
  double frozen_swing_sign_ = 0.0;
  int frozen_clip_id_ = -1;
};

}  // namespace a3_pingpong
