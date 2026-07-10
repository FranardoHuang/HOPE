// Copyright (c) 2026, AgiBot Inc. All rights reserved.
// 对应 notes/a3_backend_plan.md §PR 8 Task 8.1
#include "a3_deploy/a3_policy_driver.hpp"

#include "a3_deploy/expand_to_backend.hpp"
#include "a3_deploy/safe_halt.hpp"
#include "a3_policy_parameters.hpp"
#include "robot_io/a3_layout_extra.hpp"

#include <sys/prctl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace a3_deploy {

namespace {

// NOTE: a3_sync::A3SyncLoop stamps RobotState.timestamp_ns using
// std::chrono::system_clock (see a3_sync_loop.cpp:27), so the watchdog
// must compare ages against the *same* clock. Using steady_clock here
// would mix epochs (wall-clock ~1.7e18 vs monotonic ~1e13) and the age
// check would be meaningless.
std::int64_t NowSystemNs() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::int64_t NowSteadyNs() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool IsValidCommand(const robot_io::RobotCommand& cmd, int dof) noexcept {
  if (cmd.q_des.size() != dof || cmd.dq_des.size() != dof ||
      cmd.tau_ff.size() != dof || cmd.kp.size() != dof || cmd.kd.size() != dof) {
    return false;
  }
  if (!cmd.q_des.allFinite() || !cmd.dq_des.allFinite() || !cmd.tau_ff.allFinite() ||
      !cmd.kp.allFinite() || !cmd.kd.allFinite()) {
    return false;
  }
  return (cmd.kp.array() >= 0.0).all() && (cmd.kd.array() >= 0.0).all();
}

void PreSizeCommandBuffer(robot_io::RobotCommand& cmd) {
  constexpr int N = robot_io::kA3Dof;
  cmd.q_des  = Eigen::VectorXd::Zero(N);
  cmd.dq_des = Eigen::VectorXd::Zero(N);
  cmd.tau_ff = Eigen::VectorXd::Zero(N);
  cmd.kp     = Eigen::VectorXd::Zero(N);
  cmd.kd     = Eigen::VectorXd::Zero(N);
}

void PoisonCommandBuffer(robot_io::RobotCommand& cmd) noexcept {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  cmd.q_des.setConstant(nan);
  cmd.dq_des.setConstant(nan);
  cmd.tau_ff.setConstant(nan);
  cmd.kp.setConstant(nan);
  cmd.kd.setConstant(nan);
}

}  // namespace

// ---------------------------------------------------------------------------
a3_rt::A3BasedTask::Options A3PolicyDriver::BuildBaseOptions_(
    const A3PolicyDriverOptions& opt) {
  a3_rt::A3BasedTask::Options b;
  b.name = "a3_policy";
  const double hz = (opt.policy_hz > 0.0) ? opt.policy_hz : 50.0;
  b.period_ns = static_cast<std::int64_t>(1.0e9 / hz);
  b.first_wake_monotonic_ns = opt.first_wake_monotonic_ns;
  b.sched = opt.sched;
  return b;
}

// ---------------------------------------------------------------------------
A3PolicyDriver::A3PolicyDriver(robot_io::RobotIOBackend& backend,
                               PolicyFn policy,
                               A3PolicyDriverOptions opt)
    : a3_rt::A3BasedTask(BuildBaseOptions_(opt)),
      backend_(backend),
      policy_(std::move(policy)),
      opt_(opt),
      watchdog_(opt.watchdog) {
  // Pre-size the reusable command buffer so RunOnce is alloc-free from tick 1.
  PreSizeCommandBuffer(cmd_out_);
  PreSizeCommandBuffer(safe_halt_cmd_);
  command_ = [this](std::uint64_t tick_idx,
                    const robot_io::RobotState& state,
                    robot_io::RobotCommand& command_out) {
    if (policy_) {
      policy_(tick_idx, state, q_out_buf_);
    } else {
      q_out_buf_.fill(0.0);
    }
    ExpandToBackend(q_out_buf_, a3_kps, a3_kds, command_out);
    return true;
  };
}

A3PolicyDriver::A3PolicyDriver(robot_io::RobotIOBackend& backend,
                               CommandFn command,
                               A3PolicyDriverOptions opt)
    : a3_rt::A3BasedTask(BuildBaseOptions_(opt)),
      backend_(backend),
      command_(std::move(command)),
      opt_(opt),
      watchdog_(opt.watchdog) {
  // Pre-size the reusable command buffer so RunOnce is alloc-free from tick 1.
  PreSizeCommandBuffer(cmd_out_);
  PreSizeCommandBuffer(safe_halt_cmd_);
}

A3PolicyDriver::~A3PolicyDriver() { StopDriver(); }

// ---------------------------------------------------------------------------
bool A3PolicyDriver::StartDriver() {
  if (driver_started_.exchange(true, std::memory_order_acq_rel)) return false;
  if (!std::isfinite(opt_.command_deadline_s) || opt_.command_deadline_s < 0.0 ||
      !std::isfinite(opt_.safe_halt_retry_period_s) ||
      opt_.safe_halt_retry_period_s <= 0.0 ||
      !std::isfinite(opt_.final_safe_halt_retry_period_s) ||
      opt_.final_safe_halt_retry_period_s < 0.0 ||
      opt_.final_safe_halt_max_attempts < 1 ||
      (opt_.command_deadline_s > 0.0 && opt_.sched.priority >= 99)) {
    driver_started_.store(false, std::memory_order_release);
    return false;
  }
  try {
    expected_dof_ = backend_.GetLayout().dof();
  } catch (...) {
    expected_dof_ = 0;
  }
  if (expected_dof_ != robot_io::kA3Dof) {
    driver_started_.store(false, std::memory_order_release);
    return false;
  }
  command_faulted_.store(false, std::memory_order_release);
  command_inflight_.store(false, std::memory_order_release);
  inflight_publish_expected_.store(false, std::memory_order_release);
  command_start_steady_ns_.store(0, std::memory_order_release);
  nonpublish_halt_pending_.store(false, std::memory_order_release);
  normal_command_live_.store(false, std::memory_order_release);
  last_safe_halt_attempt_steady_ns_.store(0, std::memory_order_release);
  publication_armed_.store(false, std::memory_order_release);
  final_safe_halt_succeeded_.store(true, std::memory_order_release);
  halt_generation_.store(0, std::memory_order_release);
  has_sent_command_.store(false, std::memory_order_release);
  std::atomic_store(&latest_state_, std::shared_ptr<const robot_io::RobotState>{});
  if (opt_.trigger_on_state) {
    std::lock_guard<std::mutex> lk(event_mtx_);
    event_state_.reset();
    event_state_seq_ = 0;
  }
  backend_.RegisterStateCallback(
      [this](const robot_io::RobotState& s) { OnBackendState_(s); });
  // The supervisor also owns publish->nonpublish edge halts, so it must run whenever a
  // mode predicate exists even when deadline monitoring itself is disabled.
  if (opt_.command_deadline_s > 0.0 || opt_.command_publish_expected) {
    if (command_watchdog_running_.exchange(true, std::memory_order_acq_rel)) {
      backend_.RegisterStateCallback({});
      driver_started_.store(false, std::memory_order_release);
      return false;
    }
    try {
      command_watchdog_thread_ =
          std::thread(&A3PolicyDriver::CommandWatchdogThreadMain_, this);
    } catch (...) {
      command_watchdog_running_.store(false, std::memory_order_release);
      backend_.RegisterStateCallback({});
      driver_started_.store(false, std::memory_order_release);
      return false;
    }
  }
  if (opt_.trigger_on_state) {
    if (event_running_.exchange(true, std::memory_order_acq_rel)) {
      command_watchdog_running_.store(false, std::memory_order_release);
      if (command_watchdog_thread_.joinable()) command_watchdog_thread_.join();
      backend_.RegisterStateCallback({});
      driver_started_.store(false, std::memory_order_release);
      return false;
    }
    try {
      event_thread_ = std::thread(&A3PolicyDriver::EventThreadMain_, this);
    } catch (...) {
      event_running_.store(false, std::memory_order_release);
      command_watchdog_running_.store(false, std::memory_order_release);
      if (command_watchdog_thread_.joinable()) command_watchdog_thread_.join();
      backend_.RegisterStateCallback({});
      driver_started_.store(false, std::memory_order_release);
      return false;
    }
    return true;
  }
  if (Start()) return true;
  command_watchdog_running_.store(false, std::memory_order_release);
  if (command_watchdog_thread_.joinable()) command_watchdog_thread_.join();
  backend_.RegisterStateCallback({});
  driver_started_.store(false, std::memory_order_release);
  return false;
}

bool A3PolicyDriver::StopDriver() {
  if (!driver_started_.exchange(false, std::memory_order_acq_rel)) {
    return final_safe_halt_succeeded_.load(std::memory_order_acquire);
  }
  if (opt_.trigger_on_state) {
    event_running_.store(false, std::memory_order_release);
    event_cv_.notify_all();
    if (event_thread_.joinable()) event_thread_.join();
  } else {
    Stop();
  }
  command_watchdog_running_.store(false, std::memory_order_release);
  if (command_watchdog_thread_.joinable()) command_watchdog_thread_.join();
  if (opt_.send_final_safe_halt_on_stop && SafetyCommandRequired_()) {
    final_safe_halt_succeeded_.store(false, std::memory_order_release);
    for (int attempt = 0; attempt < opt_.final_safe_halt_max_attempts; ++attempt) {
      if (SendSafeHalt_(std::atomic_load(&latest_state_), true)) {
        final_safe_halt_succeeded_.store(true, std::memory_order_release);
        break;
      }
      if (attempt + 1 < opt_.final_safe_halt_max_attempts &&
          opt_.final_safe_halt_retry_period_s > 0.0) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(opt_.final_safe_halt_retry_period_s));
      }
    }
  }
  backend_.RegisterStateCallback({});
  return final_safe_halt_succeeded_.load(std::memory_order_acquire);
}

bool A3PolicyDriver::CommitAuthorizationTransition(
    const std::function<void()>& transition,
    bool send_safe_halt_barrier) noexcept {
  std::lock_guard<std::mutex> send_lk(backend_send_mtx_);

  try {
    if (transition) transition();
  } catch (...) {
    const bool first = !command_faulted_.exchange(true, std::memory_order_acq_rel);
    if (first) command_failure_count_.fetch_add(1, std::memory_order_relaxed);
    if (opt_.send_safe_halt_on_command_failure && SafetyCommandRequired_()) {
      if (!SendSafeHaltLocked_(std::atomic_load(&latest_state_), false)) {
        nonpublish_halt_pending_.store(true, std::memory_order_release);
      }
    }
    return false;
  }

  // The transition itself and this barrier are one backend-send transaction.  A normal
  // command that acquired backend_send_mtx_ first is followed by this halt; a callback that
  // acquires it later observes the new authorization generation and is rejected.
  if (send_safe_halt_barrier && SafetyCommandRequired_()) {
    if (!SendSafeHaltLocked_(std::atomic_load(&latest_state_), false)) {
      const bool first = !command_faulted_.exchange(true, std::memory_order_acq_rel);
      if (first) command_failure_count_.fetch_add(1, std::memory_order_relaxed);
      nonpublish_halt_pending_.store(true, std::memory_order_release);
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
void A3PolicyDriver::OnBackendState_(const robot_io::RobotState& state) noexcept {
  // Copy into a heap-allocated snapshot so the RT thread can read it without
  // racing with the sync-loop's writes. shared_ptr refcount is thread-safe,
  // but the std::atomic<shared_ptr> specialisation is missing on GCC 11.4
  // libstdc++; use the free-function std::atomic_load / atomic_store instead
  // (same pattern as a3_aimrt_backend.cpp).
  try {
    auto snap = std::make_shared<const robot_io::RobotState>(state);
    std::atomic_store(&latest_state_, snap);
    if (opt_.trigger_on_state) {
      {
        std::lock_guard<std::mutex> lk(event_mtx_);
        event_state_ = std::move(snap);
        ++event_state_seq_;
      }
      event_cv_.notify_one();
    }
  } catch (...) {
    // noexcept contract — swallow any bad_alloc.
  }
}

void A3PolicyDriver::EventThreadMain_() noexcept {
  char name[16]{};
  std::strncpy(name, "a3_policy_evt", sizeof(name) - 1);
  prctl(PR_SET_NAME, name, 0, 0, 0);
  a3_rt::SetRtSchedFifo(opt_.sched.priority);
  a3_rt::PinCurrentThreadToCpu(opt_.sched.cpu);

  std::uint64_t seen_seq = 0;
  std::int64_t last_policy_state_stamp_ns = 0;
  const std::int64_t min_period_ns =
      opt_.trigger_min_period_ns > 0
          ? opt_.trigger_min_period_ns
          : static_cast<std::int64_t>(1.0e9 /
                                      (opt_.policy_hz > 0.0 ? opt_.policy_hz
                                                            : 50.0));

  while (event_running_.load(std::memory_order_acquire)) {
    std::shared_ptr<const robot_io::RobotState> state;
    {
      std::unique_lock<std::mutex> lk(event_mtx_);
      event_cv_.wait(lk, [&] {
        return !event_running_.load(std::memory_order_acquire) ||
               event_state_seq_ != seen_seq;
      });
      if (!event_running_.load(std::memory_order_acquire)) break;
      seen_seq = event_state_seq_;
      state = event_state_;
    }

    if (!state || !state->sync_complete || state->state_sync_ready_ns <= 0) {
      continue;
    }
    if (last_policy_state_stamp_ns > 0 &&
        state->timestamp_ns < last_policy_state_stamp_ns + min_period_ns) {
      continue;
    }
    last_policy_state_stamp_ns = state->timestamp_ns;

    const auto target_ns = state->state_sync_ready_ns +
                           std::max<std::int64_t>(0, opt_.trigger_offset_ns);
    const auto now_ns = NowSystemNs();
    if (target_ns > now_ns) {
      std::this_thread::sleep_until(
          std::chrono::system_clock::time_point(
              std::chrono::duration_cast<std::chrono::system_clock::duration>(
                  std::chrono::nanoseconds(target_ns))));
      if (!event_running_.load(std::memory_order_acquire)) break;
    }

    RunOnceWithState_(state);
  }
}

void A3PolicyDriver::CommandWatchdogThreadMain_() noexcept {
  char name[16]{};
  std::strncpy(name, "a3_cmd_watch", sizeof(name) - 1);
  prctl(PR_SET_NAME, name, 0, 0, 0);
  if (opt_.sched.priority > 0) {
    a3_rt::SetRtSchedFifo(opt_.sched.priority + 1);
  }
  a3_rt::PinCurrentThreadToCpu(opt_.sched.cpu);
  const auto deadline_ns = static_cast<std::int64_t>(
      std::max(0.0, opt_.command_deadline_s) * 1.0e9);
  const auto poll = std::chrono::microseconds(
      deadline_ns > 0
          ? std::max<std::int64_t>(250,
                                   std::min<std::int64_t>(1000, deadline_ns / 4 / 1000))
          : 1000);
  bool previous_publish_expected = CommandPublishExpected_();
  while (command_watchdog_running_.load(std::memory_order_acquire)) {
    const bool publish_expected = CommandPublishExpected_();
    if (!publish_expected && SafetyCommandRequired_() &&
        (previous_publish_expected || normal_command_live_.load(std::memory_order_acquire))) {
      // Persistent latch: a failed halt is retried below. This edge path does not wait for a
      // policy callback, so SHADOW/Stop cannot leave the last non-zero command resident.
      nonpublish_halt_pending_.store(true, std::memory_order_release);
    }
    previous_publish_expected = publish_expected;

    if (nonpublish_halt_pending_.load(std::memory_order_acquire)) {
      const auto retry_ns = static_cast<std::int64_t>(
          opt_.safe_halt_retry_period_s * 1.0e9);
      const auto now_ns = NowSteadyNs();
      const auto last_ns =
          last_safe_halt_attempt_steady_ns_.load(std::memory_order_acquire);
      if (now_ns - last_ns >= retry_ns) {
        if (SendSafeHalt_(std::atomic_load(&latest_state_), false)) {
          nonpublish_halt_pending_.store(false, std::memory_order_release);
        }
      }
    }

    if (deadline_ns > 0 && command_inflight_.load(std::memory_order_acquire)) {
      const auto started = command_start_steady_ns_.load(std::memory_order_acquire);
      const bool invocation_expected =
          inflight_publish_expected_.load(std::memory_order_acquire);
      if (invocation_expected && started > 0 &&
          NowSteadyNs() - started > deadline_ns) {
        const auto state = std::atomic_load(&latest_state_);
        LatchCommandFault_(state, true);
        const auto retry_ns = static_cast<std::int64_t>(
            opt_.safe_halt_retry_period_s * 1.0e9);
        const auto now_ns = NowSteadyNs();
        const auto last_ns =
            last_safe_halt_attempt_steady_ns_.load(std::memory_order_acquire);
        if (now_ns - last_ns >= retry_ns && SafetyCommandRequired_(true)) {
          SendSafeHalt_(state, false);
        }
      }
    }
    std::this_thread::sleep_for(poll);
  }
}

bool A3PolicyDriver::CommandPublishExpected_() noexcept {
  try {
    return opt_.command_publish_expected ? opt_.command_publish_expected() : true;
  } catch (...) {
    // The predicate is an authorization boundary.  Treat an exception as
    // non-publish and permanently fault the command path; returning true here
    // would silently fail open.  Do not call LatchCommandFault_ here because
    // this predicate is also evaluated while backend_send_mtx_ is held.
    const bool first = !command_faulted_.exchange(true, std::memory_order_acq_rel);
    if (first) command_failure_count_.fetch_add(1, std::memory_order_relaxed);
    if (opt_.send_safe_halt_on_command_failure && SafetyCommandRequired_()) {
      nonpublish_halt_pending_.store(true, std::memory_order_release);
    }
    return false;
  }
}

bool A3PolicyDriver::SafetyCommandRequired_(bool invocation_expected) const noexcept {
  return invocation_expected || publication_armed_.load(std::memory_order_acquire) ||
         has_sent_command_.load(std::memory_order_acquire);
}

bool A3PolicyDriver::SendSafeHalt_(
    const std::shared_ptr<const robot_io::RobotState>& state,
    bool final_frame) noexcept {
  // A halt and normal publish must be a total order. If a normal send already passed its
  // under-lock authorization check, it completes first and this halt is guaranteed last.
  std::lock_guard<std::mutex> send_lk(backend_send_mtx_);
  return SendSafeHaltLocked_(state, final_frame);
}

bool A3PolicyDriver::SendSafeHaltLocked_(
    const std::shared_ptr<const robot_io::RobotState>& state,
    bool final_frame) noexcept {
  last_safe_halt_attempt_steady_ns_.store(NowSteadyNs(), std::memory_order_release);
  // safe_halt_cmd_ is shared by the policy and supervisor threads; callers hold
  // backend_send_mtx_ while both building and publishing it.
  const Eigen::VectorXd empty;
  BuildSafeHaltCommand(state ? state->q : empty, safe_halt_cmd_);
  bool sent = false;
  try {
    sent = backend_.SendCommand(safe_halt_cmd_);
  } catch (...) {
    sent = false;
  }
  if (!sent) {
    send_failure_count_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  normal_command_live_.store(false, std::memory_order_release);
  nonpublish_halt_pending_.store(false, std::memory_order_release);
  halt_generation_.fetch_add(1, std::memory_order_release);
  has_sent_command_.store(true, std::memory_order_release);
  if (final_frame) {
    final_safe_halt_count_.fetch_add(1, std::memory_order_relaxed);
  } else {
    safe_halt_count_.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
}

void A3PolicyDriver::LatchCommandFault_(
    const std::shared_ptr<const robot_io::RobotState>& state,
    bool deadline) noexcept {
  const bool first = !command_faulted_.exchange(true, std::memory_order_acq_rel);
  if (!first) return;
  if (deadline) deadline_violation_count_.fetch_add(1, std::memory_order_relaxed);
  if (opt_.send_safe_halt_on_command_failure && SafetyCommandRequired_()) {
    // A failed SendCommand must not turn a one-shot fault path into a best-effort halt.
    // The independent supervisor owns the persistent retry latch, which matters especially
    // for trigger-on-state mode where no later state callback is guaranteed to arrive.
    if (!SendSafeHalt_(state, false)) {
      nonpublish_halt_pending_.store(true, std::memory_order_release);
    }
  }
}

// ---------------------------------------------------------------------------
void A3PolicyDriver::RunOnce() noexcept {
  // Non-RT-safe things to avoid here: new/malloc, std::cout, throws. All
  // buffers are pre-allocated members.
  auto s = std::atomic_load(&latest_state_);
  if (!s) return;  // No state yet; skip this tick.
  RunOnceWithState_(std::move(s));
}

void A3PolicyDriver::RunOnceWithState_(
    std::shared_ptr<const robot_io::RobotState> s) noexcept {
  const std::int64_t now = NowSystemNs();

  const bool complete = s->sync_complete;
  const bool aligned = complete && s->sync_aligned;
  const auto verdict = watchdog_.Check(now, s->timestamp_ns, aligned);

  if (command_faulted_.load(std::memory_order_acquire)) {
    if (opt_.send_safe_halt_on_command_failure && SafetyCommandRequired_()) {
      SendSafeHalt_(s, false);
    }
    return;
  }

  if (verdict == WatchdogVerdict::Ok && complete) {
    // Hand the RT thread the full state + its monotonic tick index. The
    // callback is responsible for gathering the 29-DOF MuJoCo policy view
    // out of state.q via robot_io::ExtractPolicyView, building the
    // obs_dict, running inference, and writing q_des_29_out (29-DOF
    // MuJoCo policy view — waist + arms + legs, neck excluded; see
    // notes/a3_dof_orderings.md). tick_counter_ is the current Ok-tick
    // count (0 on the very first Ok tick); it is post-incremented below
    // so the NEXT Ok tick sees one higher. The tokenizer replay uses
    // tick_idx to index its offline-baked stream. Short complete-but-
    // unaligned bursts are still allowed here; A3Watchdog escalates only
    // after the configured consecutive threshold.
    bool invocation_expected = true;
    bool authorization_tracked = false;
    std::uint64_t authorization_before = 0;
    const std::uint64_t halt_generation_before =
        halt_generation_.load(std::memory_order_acquire);
    if (command_) {
      invocation_expected = CommandPublishExpected_();
      if (opt_.command_authorization_token) {
        authorization_tracked = true;
        try {
          authorization_before = opt_.command_authorization_token();
        } catch (...) {
          command_failure_count_.fetch_add(1, std::memory_order_relaxed);
          LatchCommandFault_(s, false);
          return;
        }
      }
      inflight_publish_expected_.store(invocation_expected, std::memory_order_release);
      if (invocation_expected) publication_armed_.store(true, std::memory_order_release);
      const std::uint64_t tick_idx =
          policy_tick_count_.load(std::memory_order_relaxed);
      PoisonCommandBuffer(cmd_out_);
      const auto start_ns = NowSteadyNs();
      command_start_steady_ns_.store(start_ns, std::memory_order_release);
      command_inflight_.store(true, std::memory_order_release);
      bool publish = false;
      try {
        publish = command_(tick_idx, *s, cmd_out_);
      } catch (...) {
        command_inflight_.store(false, std::memory_order_release);
        inflight_publish_expected_.store(false, std::memory_order_release);
        command_failure_count_.fetch_add(1, std::memory_order_relaxed);
        LatchCommandFault_(s, false);
        return;
      }
      command_inflight_.store(false, std::memory_order_release);
      inflight_publish_expected_.store(false, std::memory_order_release);
      if (opt_.command_authorization_token) {
        bool authorization_changed = false;
        try {
          authorization_changed =
              opt_.command_authorization_token() != authorization_before;
        } catch (...) {
          command_failure_count_.fetch_add(1, std::memory_order_relaxed);
          LatchCommandFault_(s, false);
          return;
        }
        if (authorization_changed) {
          authorization_change_count_.fetch_add(1, std::memory_order_relaxed);
          // A normal operator/safety transition is not a command-path fault, but the old
          // command must never escape. Send a zero-gain frame even when both modes are
          // publish-capable (MOTION -> PASSIVE/PD_STAND), then let the next tick compute
          // under the new authorization.
          const bool transition_halt_already_ordered =
              halt_generation_.load(std::memory_order_acquire) != halt_generation_before;
          if (!transition_halt_already_ordered &&
              (invocation_expected || CommandPublishExpected_() || SafetyCommandRequired_())) {
            if (!SendSafeHalt_(s, false)) LatchCommandFault_(s, false);
          }
          return;
        }
      }
      if (!publish) {
        return;
      }
      if (opt_.command_deadline_s > 0.0 &&
          static_cast<double>(NowSteadyNs() - start_ns) * 1.0e-9 > opt_.command_deadline_s &&
          invocation_expected) {
        LatchCommandFault_(s, true);
        return;
      }
      if (command_faulted_.load(std::memory_order_acquire)) return;
      // Interlock twice: a callback that began in SHADOW must never publish if
      // mode changes to MOTION, and one that began in MOTION must not publish
      // a late command after SHADOW/SIGTERM is requested.
      if (!invocation_expected || !CommandPublishExpected_()) return;
    } else {
      q_out_buf_.fill(0.0);
      ExpandToBackend(q_out_buf_, a3_kps, a3_kds, cmd_out_);
    }
    if (!IsValidCommand(cmd_out_, expected_dof_)) {
      command_failure_count_.fetch_add(1, std::memory_order_relaxed);
      LatchCommandFault_(s, false);
      return;
    }
    bool sent = false;
    bool authorization_lost = false;
    bool halt_already_ordered = false;
    {
      std::lock_guard<std::mutex> send_lk(backend_send_mtx_);
      // Final commit check under the same lock used by the independent halt path. If the
      // transition happened before this lock, reject the old command. If it happens after
      // this check, the supervisor's halt waits for this send and is therefore ordered last.
      halt_already_ordered =
          halt_generation_.load(std::memory_order_acquire) != halt_generation_before;
      if (command_faulted_.load(std::memory_order_acquire) || halt_already_ordered ||
          (command_ && (!invocation_expected || !CommandPublishExpected_()))) {
        authorization_lost = true;
      }
      if (!authorization_lost && authorization_tracked) {
        try {
          authorization_lost =
              opt_.command_authorization_token() != authorization_before;
        } catch (...) {
          authorization_lost = true;
          command_failure_count_.fetch_add(1, std::memory_order_relaxed);
          command_faulted_.store(true, std::memory_order_release);
        }
      }
      if (!authorization_lost) {
        try {
          sent = backend_.SendCommand(cmd_out_);
        } catch (...) {
          sent = false;
        }
        if (sent) normal_command_live_.store(true, std::memory_order_release);
      }
    }
    if (authorization_lost) {
      if (!halt_already_ordered && SafetyCommandRequired_(invocation_expected) &&
          !SendSafeHalt_(s, false)) {
        // Authorization-token/predicate exceptions can arrive in the final under-lock
        // commit check. Preserve the same persistent retry guarantee as callback/deadline
        // faults instead of silently accepting a single failed halt attempt.
        nonpublish_halt_pending_.store(true, std::memory_order_release);
      }
      return;
    }
    if (!sent) {
      send_failure_count_.fetch_add(1, std::memory_order_relaxed);
      LatchCommandFault_(s, false);
      return;
    }
    has_sent_command_.store(true, std::memory_order_release);
    policy_tick_count_.fetch_add(1, std::memory_order_relaxed);
  } else {
    const bool current_expected = CommandPublishExpected_();
    if (!current_expected && !SafetyCommandRequired_()) return;
    if (!opt_.send_safe_halt_before_first_command && !SafetyCommandRequired_()) {
      return;
    }
    if (current_expected) publication_armed_.store(true, std::memory_order_release);
    SendSafeHalt_(s, false);
  }
}

}  // namespace a3_deploy
