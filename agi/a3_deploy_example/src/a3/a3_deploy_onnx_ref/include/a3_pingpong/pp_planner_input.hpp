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
//    RACKET  /racket/command_flat   (>=11 doubles)
//      [0]=schema(1)  [1]=valid(0/1)  [2]=swing_sign(+1 fore/-1 back)
//      [3..5]=pos_w(x,y,z)  [6..8]=vel_w(x,y,z)
//      [9]=time_to_strike(s)  [10]=strike_time(s, informational)
//      [11]=frame_code(0=world/table, 1=base_link)   (optional; default 0)
//    BASE    /a3/base_pose_flat     (>=9 doubles)
//      [0]=schema(1)  [1]=valid(0/1)  [2..4]=pos(x,y,z)
//      [5..8]=quat(w,x,y,z)
#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <vector>

#include "a3_pingpong/pp_frame_math.hpp"

namespace a3_pingpong {

inline double PpNowMonotonicSec() {
  struct timespec ts {};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// ------------------------------- racket target ------------------------------
struct PpRacketMsg {
  bool valid = false;
  double swing_sign = 1.0;     // +1 forehand / -1 backhand (planner's chosen side)
  Vec3 pos_w = Vec3::Zero();   // racket intercept point (frame per frame_code)
  Vec3 vel_w = Vec3::Zero();   // desired racket velocity at strike
  double time_to_strike = 0.0; // seconds from the message stamp to strike
  double strike_time = 0.0;    // absolute/relative strike time (informational)
  int frame_code = 0;          // 0 = world/table, 1 = base_link
};

// A latest-value mailbox with per-validity timestamps. The engage logic needs
// both "age of the newest VALID command" and "was an invalid seen AFTER it"
// (the planner_invalid_grace_s flutter tolerance from the Python runner).
class PpRacketTargetInput {
 public:
  // Fed from the AimRT subscriber thread. `a` is the decoded Float64MultiArray.
  void SetFromFlat(const std::vector<double>& a) {
    if (a.size() < 11) return;  // malformed -> ignore (last good value stands, ages out)
    if (!std::isfinite(a[0]) || a[0] != 1.0 || !std::isfinite(a[1]) ||
        (a[1] != 0.0 && a[1] != 1.0))
      return;
    PpRacketMsg m;
    m.valid = a[1] != 0.0;
    const double now = PpNowMonotonicSec();
    if (!m.valid) {
      std::lock_guard<std::mutex> lk(mu_);
      last_invalid_wall_ = now;
      any_ = true;
      return;
    }
    for (size_t i = 2; i < 11; ++i)
      if (!std::isfinite(a[i])) return;
    if (a.size() >= 12 && (!std::isfinite(a[11]) || (a[11] != 0.0 && a[11] != 1.0)))
      return;
    m.swing_sign = a[2] >= 0.0 ? 1.0 : -1.0;
    m.pos_w = Vec3(a[3], a[4], a[5]);
    m.vel_w = Vec3(a[6], a[7], a[8]);
    m.time_to_strike = a[9];
    m.strike_time = a[10];
    m.frame_code = a.size() >= 12 ? static_cast<int>(a[11]) : 0;
    std::lock_guard<std::mutex> lk(mu_);
    last_valid_ = m;
    last_valid_wall_ = now;
    any_ = true;
  }

  struct Snapshot {
    bool has_valid = false;      // a valid command was ever received
    PpRacketMsg cmd;             // the newest VALID command (invalids never overwrite)
    double valid_age_s = 1e9;    // wall age of that valid command
    bool invalid_after = false;  // an invalid arrived AFTER the newest valid
  };

  Snapshot Latest() const {
    Snapshot s;
    const double now = PpNowMonotonicSec();
    std::lock_guard<std::mutex> lk(mu_);
    if (last_valid_wall_ < 0.0) return s;  // no valid yet
    s.has_valid = true;
    s.cmd = last_valid_;
    s.valid_age_s = now - last_valid_wall_;
    s.invalid_after = last_invalid_wall_ > last_valid_wall_;
    return s;
  }

  bool has_any() const { return any_.load(); }

 private:
  mutable std::mutex mu_;
  PpRacketMsg last_valid_{};
  double last_valid_wall_ = -1.0;
  double last_invalid_wall_ = -1.0;
  std::atomic<bool> any_{false};
};

// -------------------------------- base pose ---------------------------------
struct PpBaseSample {
  Vec3 pos = Vec3::Zero();
  Vec4 quat = Vec4(1, 0, 0, 0);  // w,x,y,z
  double age_s = 1e9;
};

// Mocap/localizer base pose in the SAME world frame as the racket target. Feeds
// LocMode::kExternalBase. Age-gated exactly like the sim oracle so a lost mocap
// stream degrades to a loud stand, never a frozen fictional pose.
class PpBasePoseInput {
 public:
  void SetFromFlat(const std::vector<double>& a) {
    if (a.size() < 9) return;
    if (!std::isfinite(a[0]) || a[0] != 1.0 || !std::isfinite(a[1]) ||
        (a[1] != 0.0 && a[1] != 1.0))
      return;
    const bool valid = a[1] != 0.0;
    if (!valid) return;  // invalid pose -> let the last good sample age out to stale
    for (size_t i = 2; i < 9; ++i)
      if (!std::isfinite(a[i])) return;
    Vec4 quat(a[5], a[6], a[7], a[8]);
    const double quat_norm = quat.norm();
    if (!std::isfinite(quat_norm) || quat_norm < 1e-6) return;
    quat /= quat_norm;
    const double now = PpNowMonotonicSec();
    std::lock_guard<std::mutex> lk(mu_);
    pos_ = Vec3(a[2], a[3], a[4]);
    quat_ = quat;
    stamp_wall_ = now;
    any_ = true;
  }

  bool Latest(PpBaseSample& out, double max_age_s) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (stamp_wall_ < 0.0) return false;
    out.pos = pos_;
    out.quat = quat_;
    out.age_s = PpNowMonotonicSec() - stamp_wall_;
    return out.age_s >= 0.0 && out.age_s <= max_age_s;
  }

  bool has_any() const { return any_.load(); }

 private:
  mutable std::mutex mu_;
  Vec3 pos_ = Vec3::Zero();
  Vec4 quat_ = Vec4(1, 0, 0, 0);
  double stamp_wall_ = -1.0;
  std::atomic<bool> any_{false};
};

}  // namespace a3_pingpong
