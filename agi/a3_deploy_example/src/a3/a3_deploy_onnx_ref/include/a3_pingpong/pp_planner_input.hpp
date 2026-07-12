// ============================================================================
//  Live planner inputs for the ping-pong runner (racket target + base pose).
// ============================================================================
//  These are the OFFICIAL-path replacement for ScriptedTarget: instead of a
//  fixed front-right TEST target baked into PpPolicyConfig, a real planner feeds
//  (pos/vel/side/time-to-strike) here and a mocap localizer feeds the base pose.
//
//  DELIVERY: the A3AimrtBackend subscribes std_msgs/Float64MultiArray over the
//  AimRT *ros2* backend (the pure-iceoryx body-drive path is untouched) and
//  calls SetFromFlat(...) on these holders from the subscriber thread. PpPolicy
//  reads the latest value from the 50 Hz driver thread. Both are lock-guarded.
//
//  WHY std_msgs/Float64MultiArray and not hope_msgs/RacketCommand: subscribing
//  the custom hope_msgs type would require its rosidl typesupport vendored+built
//  for aarch64 (the exact G1 pain we are avoiding). std_msgs is core ROS, already
//  in the runner's link closure, so the aarch64 cross-build needs no new msg pkg.
//
//  SAFETY: every read is age-gated. A stale/absent stream (planner crashed, LAN
//  drop, mocap loss) reports NOT-fresh so PpPolicy falls back to a held stand —
//  never a wild swing and never a fictional base pose.
//
//  Flat wire layouts (indices are fixed; element [0] is a schema tag):
//    RACKET schema 1 /racket/command_flat   (>=11 doubles)
//      [0]=schema(1)  [1]=valid(0/1)  [2]=swing_sign(+1 fore/-1 back)
//      [3..5]=pos_w(x,y,z)  [6..8]=vel_w(x,y,z)
//      [9]=time_to_strike(s)  [10]=strike_time(s, informational)
//      [11]=frame_code(0=world/table, 1=base_link)   (optional; default 0)
//    RACKET schema 2 face179 legacy          (exactly 16 doubles)
//      schema-1 prefix with mandatory frame_code at [11] and exact side +/-1
//      [12..14]=physical striking-face B normal (same world/table frame as pos/vel,
//               opponent-facing +X; converted per clip to raw mount +Y/A for actor input)
//      [15]=rho spin placeholder (Phase-1 requires exactly 0)
//    RACKET schema 3 formal face179          (exactly 20 doubles)
//      schema-2 prefix, [16]=shared control_epoch, [17]=command_sequence,
//      [18]=base_sequence_ref, [19]=same-host monotonic source stamp (seconds)
//    BASE schema 1 /a3/base_pose_flat        (>=9 doubles)
//      [0]=schema(1)  [1]=valid(0/1)  [2..4]=pos(x,y,z)
//      [5..8]=quat(w,x,y,z)
//    BASE schema 2 formal                    (exactly 12 doubles)
//      schema-1 prefix, [9]=shared control_epoch, [10]=base_sequence,
//      [11]=same-host monotonic source stamp (seconds)
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "a3_pingpong/pp_frame_math.hpp"

namespace a3_pingpong {

inline double PpNowMonotonicSec() {
  struct timespec ts {};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

constexpr std::uint64_t kPpMaxExactFloat64Integer = (std::uint64_t{1} << 53) - 1;

inline bool PpParseExactCounter(double raw, std::uint64_t& out) {
  if (!std::isfinite(raw) || raw < 0.0 ||
      raw > static_cast<double>(kPpMaxExactFloat64Integer) || std::floor(raw) != raw)
    return false;
  out = static_cast<std::uint64_t>(raw);
  return true;
}

enum class PpPlannerFreshnessDecision { kFresh, kStale, kRevoked };

inline PpPlannerFreshnessDecision EvaluatePpPlannerFreshness(
    double valid_age_s, double command_timeout_s, bool invalid_after,
    double invalid_grace_s, bool immediate_invalid_revoke = false) {
  if (!std::isfinite(valid_age_s) || valid_age_s < 0.0 ||
      !std::isfinite(command_timeout_s) || command_timeout_s < 0.0 ||
      valid_age_s > command_timeout_s) {
    return PpPlannerFreshnessDecision::kStale;
  }
  if (invalid_after && immediate_invalid_revoke)
    return PpPlannerFreshnessDecision::kRevoked;
  if (invalid_after &&
      (!std::isfinite(invalid_grace_s) || invalid_grace_s < 0.0 ||
       valid_age_s > invalid_grace_s)) {
    return PpPlannerFreshnessDecision::kRevoked;
  }
  return PpPlannerFreshnessDecision::kFresh;
}

// ------------------------------- racket target ------------------------------
struct PpRacketMsg {
  bool valid = false;
  double swing_sign = 1.0;     // +1 forehand / -1 backhand (planner's chosen side)
  bool has_explicit_side = false;  // true only for validated formal schema 2
  Vec3 pos_w = Vec3::Zero();   // racket intercept point (frame per frame_code)
  Vec3 vel_w = Vec3::Zero();   // desired racket velocity at strike
  double time_to_strike = 0.0; // seconds from the message stamp to strike
  double strike_time = 0.0;    // absolute/relative strike time (informational)
  int frame_code = 0;          // 0 = world/table, 1 = base_link
  bool has_face_command = false;
  Vec3 normal_cmd = Vec3(1.0, 0.0, 0.0);  // physical face B; world/table, unit, +X
  double rho = 0.0;             // reserved S3 spin lane; zero in Phase-1 schema 2
  bool has_formal_epoch = false;
  std::uint64_t control_epoch = 0;
  std::uint64_t command_sequence = 0;
  std::uint64_t base_sequence_ref = 0;
  double source_monotonic_s = -1.0;
};

// A latest-value mailbox with per-validity timestamps. The engage logic needs
// both "age of the newest VALID command" and "was an invalid seen AFTER it"
// (the planner_invalid_grace_s flutter tolerance from the Python runner).
class PpRacketTargetInput {
 public:
  explicit PpRacketTargetInput(
      std::shared_ptr<std::mutex> transaction_mu = std::make_shared<std::mutex>())
      : transaction_mu_(std::move(transaction_mu)) {}

  // Fed from the AimRT subscriber thread. `a` is the decoded Float64MultiArray.
  void SetFromFlat(const std::vector<double>& a) {
    std::lock_guard<std::mutex> transaction_lk(*transaction_mu_);
    if (a.empty() || !std::isfinite(a[0])) {
      RecordInvalidIfFaceActive();
      return;
    }
    if (a[0] != 1.0 && a[0] != 2.0 && a[0] != 3.0) {
      RecordInvalidIfFaceActive();
      return;
    }
    const bool face179 = a[0] == 2.0 || a[0] == 3.0;
    const bool formal = a[0] == 3.0;
    if (!formal) {
      bool formal_order_initialized = false;
      {
        std::lock_guard<std::mutex> lk(mu_);
        formal_order_initialized = formal_order_initialized_;
      }
      if (formal_order_initialized) {
        // Once the formal stream has established ordering, a recognized older
        // schema is a downgrade event, not a second live protocol.  Poison the
        // old tuple and require a later, causally fresh schema-3 command.
        RecordInvalidNow();
        return;
      }
    }
    // A packet that positively identifies itself as the formal face179 schema
    // but fails any subsequent validation must revoke a preceding command. If
    // it were merely ignored, a malformed live stream could keep an old swing
    // eligible for command_timeout_s. Schema 1 retains its historical
    // ignore-and-age-out behavior when no formal face command is active; a
    // malformed downgrade cannot preserve a previously active face tuple.
    const auto reject_malformed = [this, face179]() {
      if (face179)
        RecordInvalidNow();
      else
        RecordInvalidIfFaceActive();
    };
    if (a.size() < 2) {
      reject_malformed();
      return;
    }
    if ((!face179 && a.size() < 11) || (a[0] == 2.0 && a.size() != 16) ||
        (formal && a.size() != 20)) {
      reject_malformed();
      return;
    }
    if (!std::isfinite(a[1]) ||
        (a[1] != 0.0 && a[1] != 1.0)) {
      reject_malformed();
      return;
    }
    PpRacketMsg m;
    m.valid = a[1] != 0.0;
    const double now = PpNowMonotonicSec();
    if (formal) {
      if (!PpParseExactCounter(a[16], m.control_epoch) ||
          !PpParseExactCounter(a[17], m.command_sequence) ||
          !PpParseExactCounter(a[18], m.base_sequence_ref) ||
          !std::isfinite(a[19]) || a[19] < 0.0 || a[19] > now) {
        reject_malformed();
        return;
      }
      m.has_formal_epoch = true;
      m.source_monotonic_s = a[19];
    }
    const size_t body_required = face179 ? 16 : 11;
    for (size_t i = 2; i < body_required; ++i) {
      if (!std::isfinite(a[i])) {
        if (formal) {
          m.valid = false;
          CommitFormal(m, now);
        } else {
          reject_malformed();
        }
        return;
      }
    }
    if (!m.valid) {
      if (formal)
        CommitFormal(m, now);
      else
        RecordInvalidNow();
      return;
    }
    const auto reject_body = [this, formal, &m, now, &reject_malformed]() {
      if (formal) {
        m.valid = false;
        CommitFormal(m, now);
      } else {
        reject_malformed();
      }
    };
    if (a.size() >= 12 && (!std::isfinite(a[11]) || (a[11] != 0.0 && a[11] != 1.0))) {
      reject_body();
      return;
    }
    // Schema 1 keeps its historical yaw-heading interpretation of frame_code=1. The 179 tail
    // adds a full 3D normal, for which literal base-link versus yaw-heading semantics have not
    // been frozen; Phase-1 schema 2 is therefore world/table only.
    if (face179 && a[11] != 0.0) {
      reject_body();
      return;
    }
    if (face179 && a[2] != -1.0 && a[2] != 1.0) {
      reject_body();
      return;
    }
    m.swing_sign = a[2] >= 0.0 ? 1.0 : -1.0;
    m.has_explicit_side = face179;
    m.pos_w = Vec3(a[3], a[4], a[5]);
    m.vel_w = Vec3(a[6], a[7], a[8]);
    m.time_to_strike = a[9];
    m.strike_time = a[10];
    m.frame_code = a.size() >= 12 ? static_cast<int>(a[11]) : 0;
    if (face179) {
      m.normal_cmd = Vec3(a[12], a[13], a[14]);
      const double n = m.normal_cmd.norm();
      constexpr double kUnitNormalTolerance = 1e-6;
      constexpr double kOpponentFacingMinX = 1e-6;
      if (!std::isfinite(n) || std::fabs(n - 1.0) > kUnitNormalTolerance ||
          m.normal_cmd[0] <= kOpponentFacingMinX || a[15] != 0.0) {
        reject_body();
        return;
      }
      m.has_face_command = true;
      m.rho = a[15];
    }
    if (formal) {
      CommitFormal(m, now);
    } else {
      std::lock_guard<std::mutex> lk(mu_);
      last_valid_ = m;
      last_valid_wall_ = now;
      latest_event_valid_ = true;
      ++generation_;
      any_ = true;
    }
  }

  struct Snapshot {
    bool has_valid = false;      // a valid command was ever received
    PpRacketMsg cmd;             // the newest VALID command (invalids never overwrite)
    double valid_age_s = 1e9;    // wall age of that valid command
    double valid_received_wall_s = -1.0;  // local monotonic receive epoch
    bool invalid_after = false;  // an invalid arrived AFTER the newest valid
    std::uint64_t generation = 0;
  };

  Snapshot Latest() const {
    Snapshot s;
    std::lock_guard<std::mutex> lk(mu_);
    if (last_valid_wall_ < 0.0) return s;  // no valid yet
    // Read the clock only after acquiring the same lock used to publish a new
    // valid timestamp. Taking it before the lock can pair an older `now` with
    // a newer command and manufacture a negative age / longer TTS.
    const double now = PpNowMonotonicSec();
    s.has_valid = true;
    s.cmd = last_valid_;
    s.valid_age_s = last_valid_.has_formal_epoch
                        ? now - last_valid_.source_monotonic_s
                        : now - last_valid_wall_;
    s.valid_received_wall_s = last_valid_wall_;
    // Do not infer event order from wall-clock inequality: two callbacks may
    // legitimately observe the same clock tick.  The accepted-event state is
    // updated under mu_ and therefore makes an invalid deterministically win.
    s.invalid_after = !latest_event_valid_;
    s.generation = generation_.load(std::memory_order_acquire);
    return s;
  }

  bool has_any() const { return any_.load(); }
  std::shared_ptr<std::mutex> transaction_mutex() const { return transaction_mu_; }
  bool GenerationCurrent(std::uint64_t generation) const {
    return generation_.load(std::memory_order_acquire) == generation;
  }

 private:
  void CommitFormal(const PpRacketMsg& m, double now) {
    std::lock_guard<std::mutex> lk(mu_);
    const bool proven_old = formal_order_initialized_ &&
        m.control_epoch <= last_formal_epoch_ &&
        m.command_sequence <= last_formal_sequence_ &&
        m.source_monotonic_s <= last_formal_source_monotonic_s_;
    if (proven_old) return;
    const bool ordered = !formal_order_initialized_ ||
        (m.control_epoch >= last_formal_epoch_ &&
         m.command_sequence > last_formal_sequence_ &&
         m.source_monotonic_s > last_formal_source_monotonic_s_);
    if (!ordered) {
      last_invalid_wall_ = now;
      latest_event_valid_ = false;
      anonymous_revoke_source_barrier_ =
          std::max(anonymous_revoke_source_barrier_, now);
      ++generation_;
      any_ = true;
      return;
    }
    formal_order_initialized_ = true;
    last_formal_epoch_ = m.control_epoch;
    last_formal_sequence_ = m.command_sequence;
    last_formal_source_monotonic_s_ = m.source_monotonic_s;
    if (m.valid && m.source_monotonic_s > anonymous_revoke_source_barrier_) {
      last_valid_ = m;
      last_valid_wall_ = now;
      latest_event_valid_ = true;
    } else {
      last_invalid_wall_ = now;
      latest_event_valid_ = false;
      // An explicitly invalid row is new evidence and advances to its source
      // stamp. A valid row blocked only by the existing barrier is not new bad
      // evidence: keep T0 fixed so a constant-latency stream can cross it.
      if (!m.valid)
        anonymous_revoke_source_barrier_ = std::max(
            anonymous_revoke_source_barrier_, m.source_monotonic_s);
    }
    ++generation_;
    any_ = true;
  }

  void RecordInvalidNow() {
    const double now = PpNowMonotonicSec();
    std::lock_guard<std::mutex> lk(mu_);
    last_invalid_wall_ = now;
    latest_event_valid_ = false;
    anonymous_revoke_source_barrier_ =
        std::max(anonymous_revoke_source_barrier_, now);
    ++generation_;
    any_ = true;
  }

  void RecordInvalidIfFaceActive() {
    const double now = PpNowMonotonicSec();
    std::lock_guard<std::mutex> lk(mu_);
    if (last_valid_wall_ >= 0.0 && last_valid_.has_face_command) {
      last_invalid_wall_ = now;
      latest_event_valid_ = false;
      anonymous_revoke_source_barrier_ =
          std::max(anonymous_revoke_source_barrier_, now);
      ++generation_;
      any_ = true;
    }
  }

  mutable std::mutex mu_;
  std::shared_ptr<std::mutex> transaction_mu_;
  PpRacketMsg last_valid_{};
  double last_valid_wall_ = -1.0;
  double last_invalid_wall_ = -1.0;
  bool latest_event_valid_ = false;
  bool formal_order_initialized_ = false;
  std::uint64_t last_formal_epoch_ = 0;
  std::uint64_t last_formal_sequence_ = 0;
  double last_formal_source_monotonic_s_ = -1.0;
  double anonymous_revoke_source_barrier_ = -1.0;
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<bool> any_{false};
};

// -------------------------------- base pose ---------------------------------
struct PpBaseSample {
  Vec3 pos = Vec3::Zero();
  Vec4 quat = Vec4(1, 0, 0, 0);  // w,x,y,z
  double age_s = 1e9;
  double last_invalid_wall_s = -1.0;
  bool has_formal_epoch = false;
  std::uint64_t control_epoch = 0;
  std::uint64_t base_sequence = 0;
  double source_monotonic_s = -1.0;
  std::uint64_t generation = 0;
  std::uint64_t revocation_generation = 0;
};

// Immutable dual-language hard contract (mirrored by
// FormalBasePosePlausibilityGuard in the ROS producer). Coordinates are in the
// declared formal base source frame (`world` at the venue, vendor `odom` in
// Gate3), not an assumed global frame.
struct PpFormalBasePlausibilityConfig {
  Vec3 min_source = Vec3(-3.0, -3.0, 0.4);
  Vec3 max_source = Vec3(3.0, 3.0, 1.5);
  double linear_slack_m = 0.05;
  double max_linear_speed_mps = 8.0;
  double angular_slack_rad = 0.15;
  double max_angular_speed_radps = 12.0;
};

inline bool PpFormalBasePlausibilityConfigValid(
    const PpFormalBasePlausibilityConfig& cfg) {
  return cfg.min_source.allFinite() && cfg.max_source.allFinite() &&
      (cfg.min_source.array() < cfg.max_source.array()).all() &&
      std::isfinite(cfg.linear_slack_m) && cfg.linear_slack_m >= 0.0 &&
      std::isfinite(cfg.max_linear_speed_mps) &&
      cfg.max_linear_speed_mps >= 0.0 &&
      std::isfinite(cfg.angular_slack_rad) && cfg.angular_slack_rad >= 0.0 &&
      std::isfinite(cfg.max_angular_speed_radps) &&
      cfg.max_angular_speed_radps >= 0.0;
}

inline bool PpFormalBasePosePlausible(
    const PpBaseSample& sample,
    const PpFormalBasePlausibilityConfig& cfg) {
  if (!PpFormalBasePlausibilityConfigValid(cfg) || !sample.pos.allFinite() ||
      !sample.quat.allFinite())
    return false;
  if (sample.has_formal_epoch &&
      (!std::isfinite(sample.source_monotonic_s) ||
       sample.source_monotonic_s < 0.0))
    return false;
  const double quat_norm = sample.quat.norm();
  if (!std::isfinite(quat_norm) || quat_norm < 1e-6) return false;
  return (sample.pos.array() >= cfg.min_source.array()).all() &&
      (sample.pos.array() <= cfg.max_source.array()).all();
}

inline bool PpFormalBaseTransitionPlausible(
    const PpBaseSample& previous, const PpBaseSample& candidate,
    const PpFormalBasePlausibilityConfig& cfg) {
  if (!PpFormalBasePosePlausible(previous, cfg) ||
      !PpFormalBasePosePlausible(candidate, cfg))
    return false;
  const double dt =
      candidate.source_monotonic_s - previous.source_monotonic_s;
  if (!std::isfinite(dt) || dt <= 0.0) return false;
  if ((candidate.pos - previous.pos).norm() >
      cfg.linear_slack_m + cfg.max_linear_speed_mps * dt)
    return false;
  const Vec4 q_prev = previous.quat / previous.quat.norm();
  const Vec4 q_next = candidate.quat / candidate.quat.norm();
  const double dot = std::clamp(std::fabs(q_prev.dot(q_next)), 0.0, 1.0);
  const double angle = 2.0 * std::acos(dot);
  return angle <= cfg.angular_slack_rad + cfg.max_angular_speed_radps * dt;
}

// At 300 Hz this retains about 0.43 s, more than the formal 0.20 s source-age
// lease. The bound is explicit: overload/eviction fails closed instead of
// growing a control-history queue without limit.
constexpr std::size_t kPpFormalBaseHistoryCapacity = 128;

// A formal active swing owns the exact base lease captured at engage.  Normal
// same-epoch base refreshes advance the transaction generation but not this
// revocation generation, so they remain eligible.  Any accepted/anonymous
// invalid or epoch change invalidates the lease even if a valid recovery packet
// arrives before the next policy tick.
inline bool PpFormalBaseLeaseChanged(
    const PpBaseSample& current, std::uint64_t engaged_epoch,
    std::uint64_t engaged_revocation_generation) {
  return current.revocation_generation != engaged_revocation_generation ||
      (current.has_formal_epoch && current.control_epoch != engaged_epoch);
}

inline bool PpFormalBaseLeaseUsable(
    const PpBaseSample& current, bool current_fresh,
    std::uint64_t engaged_epoch,
    std::uint64_t engaged_revocation_generation) {
  return current_fresh && current.has_formal_epoch &&
      !PpFormalBaseLeaseChanged(
          current, engaged_epoch, engaged_revocation_generation);
}

// Mocap/localizer base pose in the declared formal base source frame (venue
// `world`, vendor Gate3 `odom`) shared with the racket command after the bound
// producer correction. Feeds LocMode::kExternalBase. A lost or implausible
// stream degrades to a loud stand, never a frozen fictional pose.
class PpBasePoseInput {
 public:
  explicit PpBasePoseInput(
      std::shared_ptr<std::mutex> transaction_mu = std::make_shared<std::mutex>(),
      PpFormalBasePlausibilityConfig plausibility = {})
      : transaction_mu_(std::move(transaction_mu)),
        plausibility_(std::move(plausibility)) {
    if (!PpFormalBasePlausibilityConfigValid(plausibility_))
      throw std::invalid_argument("invalid formal base plausibility contract");
  }

  void SetFromFlat(const std::vector<double>& a) {
    std::lock_guard<std::mutex> transaction_lk(*transaction_mu_);
    if (a.empty() || !std::isfinite(a[0]) || (a[0] != 1.0 && a[0] != 2.0)) {
      RecordInvalidNow();
      return;
    }
    const bool formal = a[0] == 2.0;
    if (!formal) {
      bool formal_order_initialized = false;
      {
        std::lock_guard<std::mutex> lk(mu_);
        formal_order_initialized = formal_order_initialized_;
      }
      if (formal_order_initialized) {
        // A live formal base stream cannot silently fall back to schema 1.
        // Treat the recognized downgrade as an anonymous revoke.
        RecordInvalidNow();
        return;
      }
    }
    if ((!formal && a.size() < 9) || (formal && a.size() != 12) ||
        !std::isfinite(a[1]) ||
        (a[1] != 0.0 && a[1] != 1.0)) {
      RecordInvalidNow();
      return;
    }
    const bool valid = a[1] != 0.0;
    const double now = PpNowMonotonicSec();
    std::uint64_t epoch = 0, sequence = 0;
    if (formal && (!PpParseExactCounter(a[9], epoch) ||
                   !PpParseExactCounter(a[10], sequence) || !std::isfinite(a[11]) ||
                   a[11] < 0.0 || a[11] > now)) {
      RecordInvalidNow();
      return;
    }
    for (size_t i = 2; i < 9; ++i) {
      if (!std::isfinite(a[i])) {
        if (formal)
          CommitFormalInvalid(epoch, sequence, a[11], now);
        else
          RecordInvalidNow();
        return;
      }
    }
    if (!valid) {
      if (formal)
        CommitFormalInvalid(epoch, sequence, a[11], now);
      else
        RecordInvalidNow();
      return;
    }
    Vec4 quat(a[5], a[6], a[7], a[8]);
    const double quat_norm = quat.norm();
    if (!std::isfinite(quat_norm) || quat_norm < 1e-6) {
      if (formal)
        CommitFormalInvalid(epoch, sequence, a[11], now);
      else
        RecordInvalidNow();
      return;
    }
    quat /= quat_norm;
    PpBaseSample candidate;
    candidate.pos = Vec3(a[2], a[3], a[4]);
    candidate.quat = quat;
    candidate.has_formal_epoch = formal;
    candidate.control_epoch = epoch;
    candidate.base_sequence = sequence;
    candidate.source_monotonic_s = formal ? a[11] : -1.0;
    std::lock_guard<std::mutex> lk(mu_);
    if (formal && FormalEventProvenOld(epoch, sequence, a[11])) return;
    const bool formal_order_valid =
        !formal || FormalOrderValid(epoch, sequence, a[11]);
    if (formal && formal_order_valid)
      // Consume every structurally ordered envelope exactly once even when a
      // local poison barrier makes its body ineligible. Otherwise the same
      // (epoch, sequence) could be replayed with a newer source stamp.
      UpdateFormalOrder(epoch, sequence, a[11]);
    if (formal && !formal_order_valid) {
      stamp_wall_ = -1.0;
      last_invalid_wall_ = now;
      anonymous_revoke_source_barrier_ =
          std::max(anonymous_revoke_source_barrier_, now);
      formal_valid_history_.clear();
      ++generation_;
      ++revocation_generation_;
      any_ = true;
      return;
    }
    if (formal && a[11] <= anonymous_revoke_source_barrier_) {
      // The ordered envelope was consumed above, but this candidate is old
      // relative to an existing poison barrier. Do not chase receive-now:
      // fixed-latency sources must be able to cross the original T0.
      stamp_wall_ = -1.0;
      last_invalid_wall_ = now;
      formal_valid_history_.clear();
      ++generation_;
      ++revocation_generation_;
      any_ = true;
      return;
    }
    const bool pose_plausible = PpFormalBasePosePlausible(
        candidate, plausibility_);
    const bool transition_plausible =
        !formal || !plausibility_baseline_initialized_ ||
        PpFormalBaseTransitionPlausible(
            plausibility_baseline_, candidate, plausibility_);
    if (!pose_plausible || !transition_plausible) {
      // This structurally ordered sample is new implausible geometry evidence,
      // unlike a candidate blocked only by the pre-existing barrier above.
      if (formal)
        anonymous_revoke_source_barrier_ = std::max(
            anonymous_revoke_source_barrier_, a[11]);
      stamp_wall_ = -1.0;
      last_invalid_wall_ = now;
      formal_valid_history_.clear();
      ++generation_;
      ++revocation_generation_;
      any_ = true;
      return;
    }
    if (formal) {
      if (!formal_valid_history_.empty() &&
          formal_valid_history_.back().control_epoch != epoch)
        formal_valid_history_.clear();
    }
    pos_ = candidate.pos;
    quat_ = candidate.quat;
    stamp_wall_ = now;
    has_formal_epoch_ = formal;
    control_epoch_ = epoch;
    base_sequence_ = sequence;
    source_monotonic_s_ = formal ? a[11] : -1.0;
    ++generation_;  // transaction version; normal refresh is not a revocation
    if (formal) {
      plausibility_baseline_ = candidate;
      plausibility_baseline_initialized_ = true;
      PpBaseSample history_sample;
      history_sample.pos = pos_;
      history_sample.quat = quat_;
      history_sample.has_formal_epoch = true;
      history_sample.control_epoch = epoch;
      history_sample.base_sequence = sequence;
      history_sample.source_monotonic_s = a[11];
      history_sample.generation = generation_.load(std::memory_order_acquire);
      history_sample.revocation_generation =
          revocation_generation_.load(std::memory_order_acquire);
      formal_valid_history_.push_back(history_sample);
      if (formal_valid_history_.size() > kPpFormalBaseHistoryCapacity)
        formal_valid_history_.pop_front();
    }
    any_ = true;
  }

  bool Latest(PpBaseSample& out, double max_age_s) const {
    std::lock_guard<std::mutex> lk(mu_);
    // Publish lease metadata even while revoked so an active policy tick can
    // observe a revoke followed by a fast recovery without losing the edge.
    out.last_invalid_wall_s = last_invalid_wall_;
    out.has_formal_epoch = has_formal_epoch_;
    out.control_epoch = control_epoch_;
    out.base_sequence = base_sequence_;
    out.source_monotonic_s = source_monotonic_s_;
    out.generation = generation_.load(std::memory_order_acquire);
    out.revocation_generation =
        revocation_generation_.load(std::memory_order_acquire);
    if (stamp_wall_ < 0.0) return false;
    out.pos = pos_;
    out.quat = quat_;
    out.age_s = has_formal_epoch_
                    ? PpNowMonotonicSec() - source_monotonic_s_
                    : PpNowMonotonicSec() - stamp_wall_;
    return out.age_s >= 0.0 && out.age_s <= max_age_s;
  }

  bool has_any() const { return any_.load(); }
  bool PosePlausible(const PpBaseSample& sample) const {
    return PpFormalBasePosePlausible(sample, plausibility_);
  }
  const PpFormalBasePlausibilityConfig& plausibility_config() const {
    return plausibility_;
  }
  bool ExactFormal(std::uint64_t epoch, std::uint64_t base_sequence,
                   PpBaseSample& out, double max_age_s) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!std::isfinite(max_age_s) || max_age_s < 0.0) return false;
    const auto current_revoke =
        revocation_generation_.load(std::memory_order_acquire);
    if (!has_formal_epoch_ || control_epoch_ != epoch) return false;
    for (auto it = formal_valid_history_.rbegin();
         it != formal_valid_history_.rend(); ++it) {
      if (it->control_epoch != epoch || it->base_sequence != base_sequence)
        continue;
      const double age_s = PpNowMonotonicSec() - it->source_monotonic_s;
      if (!std::isfinite(age_s) || age_s < 0.0 || age_s > max_age_s ||
          it->revocation_generation != current_revoke ||
          !PpFormalBasePosePlausible(*it, plausibility_))
        return false;
      out = *it;
      out.age_s = age_s;
      out.last_invalid_wall_s = last_invalid_wall_;
      out.generation = generation_.load(std::memory_order_acquire);
      out.revocation_generation = current_revoke;
      return true;
    }
    return false;
  }
  std::shared_ptr<std::mutex> transaction_mutex() const { return transaction_mu_; }
  bool GenerationCurrent(std::uint64_t generation) const {
    return generation_.load(std::memory_order_acquire) == generation;
  }

 private:
  bool FormalOrderValid(std::uint64_t epoch, std::uint64_t sequence,
                        double source_monotonic_s) const {
    return !formal_order_initialized_ ||
        (epoch >= last_formal_epoch_ && sequence > last_formal_sequence_ &&
         source_monotonic_s > last_formal_source_monotonic_s_);
  }

  bool FormalEventProvenOld(std::uint64_t epoch, std::uint64_t sequence,
                            double source_monotonic_s) const {
    return formal_order_initialized_ && epoch <= last_formal_epoch_ &&
        sequence <= last_formal_sequence_ &&
        source_monotonic_s <= last_formal_source_monotonic_s_;
  }

  void UpdateFormalOrder(std::uint64_t epoch, std::uint64_t sequence,
                         double source_monotonic_s) {
    formal_order_initialized_ = true;
    last_formal_epoch_ = epoch;
    last_formal_sequence_ = sequence;
    last_formal_source_monotonic_s_ = source_monotonic_s;
  }

  void CommitFormalInvalid(std::uint64_t epoch, std::uint64_t sequence,
                           double source_monotonic_s, double now) {
    std::lock_guard<std::mutex> lk(mu_);
    if (FormalEventProvenOld(epoch, sequence, source_monotonic_s)) return;
    if (FormalOrderValid(epoch, sequence, source_monotonic_s)) {
      UpdateFormalOrder(epoch, sequence, source_monotonic_s);
      anonymous_revoke_source_barrier_ =
          std::max(anonymous_revoke_source_barrier_, source_monotonic_s);
    } else {
      anonymous_revoke_source_barrier_ =
          std::max(anonymous_revoke_source_barrier_, now);
    }
    stamp_wall_ = -1.0;
    last_invalid_wall_ = now;
    formal_valid_history_.clear();
    ++generation_;
    ++revocation_generation_;
    any_ = true;
  }

  void RecordInvalidNow() {
    const double now = PpNowMonotonicSec();
    std::lock_guard<std::mutex> lk(mu_);
    stamp_wall_ = -1.0;
    last_invalid_wall_ = now;
    anonymous_revoke_source_barrier_ =
        std::max(anonymous_revoke_source_barrier_, now);
    formal_valid_history_.clear();
    ++generation_;
    ++revocation_generation_;
    any_ = true;
  }

  mutable std::mutex mu_;
  std::shared_ptr<std::mutex> transaction_mu_;
  Vec3 pos_ = Vec3::Zero();
  Vec4 quat_ = Vec4(1, 0, 0, 0);
  double stamp_wall_ = -1.0;
  double last_invalid_wall_ = -1.0;
  bool has_formal_epoch_ = false;
  std::uint64_t control_epoch_ = 0;
  std::uint64_t base_sequence_ = 0;
  double source_monotonic_s_ = -1.0;
  bool formal_order_initialized_ = false;
  std::uint64_t last_formal_epoch_ = 0;
  std::uint64_t last_formal_sequence_ = 0;
  double last_formal_source_monotonic_s_ = -1.0;
  double anonymous_revoke_source_barrier_ = -1.0;
  const PpFormalBasePlausibilityConfig plausibility_;
  bool plausibility_baseline_initialized_ = false;
  PpBaseSample plausibility_baseline_;
  std::deque<PpBaseSample> formal_valid_history_;
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<std::uint64_t> revocation_generation_{0};
  std::atomic<bool> any_{false};
};

template <class Fn>
bool PpWithPlannerInputsIfUnchanged(
    PpRacketTargetInput& racket, std::uint64_t racket_generation,
    PpBasePoseInput& base, std::uint64_t base_epoch,
    std::uint64_t base_sequence, double base_max_age_s, Fn&& fn) {
  const auto racket_tx = racket.transaction_mutex();
  const auto base_tx = base.transaction_mutex();
  if (!racket_tx || racket_tx != base_tx) return false;
  std::lock_guard<std::mutex> transaction_lk(*racket_tx);
  if (!racket.GenerationCurrent(racket_generation)) return false;
  PpBaseSample exact_base;
  if (!base.ExactFormal(
          base_epoch, base_sequence, exact_base, base_max_age_s))
    return false;
  PpBaseSample latest_base;
  if (!base.Latest(latest_base, base_max_age_s) ||
      !base.PosePlausible(latest_base))
    return false;
  return static_cast<bool>(
      std::forward<Fn>(fn)(exact_base, latest_base));
}

}  // namespace a3_pingpong
