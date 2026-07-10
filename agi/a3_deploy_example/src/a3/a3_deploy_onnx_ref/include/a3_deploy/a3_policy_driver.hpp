// Copyright (c) 2026, AgiBot Inc. All rights reserved.
// 对应 notes/a3_backend_plan.md §PR 8 Task 8.1
//
// A3PolicyDriver: a 50Hz RT loop (inheriting a3_rt::A3BasedTask from PR 4) that
// feeds a 29-DOF policy from the latest RobotState cached via a backend state
// callback. Output goes through ExpandToBackend() to a 31-DOF RobotCommand and
// is sent through RobotIOBackend::SendCommand. A3Watchdog decides each tick
// whether to run the policy or issue a safe-halt command.
#pragma once

#include "a3_deploy/strict_finite_math.hpp"

#include "a3_deploy/a3_watchdog.hpp"
#include "a3_rt/a3_based_task.hpp"
#include "a3_rt/a3_rt.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace a3_deploy {

// 29-DOF policy signature (PR 9). The RT thread calls this at `policy_hz`
// with:
//   - tick_idx: monotonic counter starting at 0, incremented AFTER every
//     successful call (i.e. only on the Watchdog::Ok branch — safe-halt
//     ticks do NOT advance the counter). This indexes the offline-baked
//     reference-motion stream via A3TokenizerReplay::At(tick_idx).
//   - state: latest synced RobotState (timestamp_ns already matches the
//     clock domain the watchdog uses).
//   - q_des_29_out: 29-float MuJoCo-ordered joint target (body only; neck
//     is added by ExpandToBackend).
using PolicyFn = std::function<void(
    std::uint64_t                   tick_idx,
    const robot_io::RobotState&     state,
    std::array<double, 29>&         q_des_29_out)>;

// Full-command signature for bring-up modes that must bypass the policy
// scatter path, e.g. PASSIVE with kp/kd/tau_ff all zero. Return true when
// command_out is valid and should be published; return false to skip this
// tick without sending any command.
using CommandFn = std::function<bool(
    std::uint64_t                   tick_idx,
    const robot_io::RobotState&     state,
    robot_io::RobotCommand&         command_out)>;

struct A3PolicyDriverOptions {
  double           policy_hz = 50.0;
  WatchdogConfig   watchdog{};
  a3_rt::RtSched   sched{};
  std::int64_t     first_wake_monotonic_ns = 0;
  bool             trigger_on_state = false;
  std::int64_t     trigger_offset_ns = 0;
  std::int64_t     trigger_min_period_ns = 0;
  bool             send_safe_halt_before_first_command = true;
  // Command callback safety. A positive deadline enables an independent supervisor that can
  // publish a safe halt even if inference blocks the policy thread. The predicate lets callers
  // declare intentional no-publish modes (e.g. SHADOW) so a slow preview never emits a command.
  double           command_deadline_s = 0.0;
  std::function<bool()> command_publish_expected{};
  // Optional authorization generation for mode/state machines. The driver snapshots this
  // token immediately before invoking CommandFn and compares it again before SendCommand.
  // Any change means the callback's command was computed under stale authority (for example,
  // MOTION -> PASSIVE inside a fall guard); the command is discarded and a zero-gain halt is
  // sent instead. A boolean predicate alone cannot catch transitions whose endpoints are both
  // publish-capable, such as MOTION -> PD_STAND/PASSIVE.
  std::function<std::uint64_t()> command_authorization_token{};
  bool             send_safe_halt_on_command_failure = true;
  bool             send_final_safe_halt_on_stop = false;
  double           safe_halt_retry_period_s = 0.02;
  int              final_safe_halt_max_attempts = 3;
  double           final_safe_halt_retry_period_s = 0.005;
};

class A3PolicyDriver : public a3_rt::A3BasedTask {
 public:
  A3PolicyDriver(robot_io::RobotIOBackend& backend,
                 PolicyFn policy,
                 A3PolicyDriverOptions opt);

  A3PolicyDriver(robot_io::RobotIOBackend& backend,
                 CommandFn command,
                 A3PolicyDriverOptions opt);

  ~A3PolicyDriver() override;

  // Must be called AFTER backend.Init(). Wires state callback and starts the
  // RT thread. Returns false on failure (already running, etc.).
  bool StartDriver();

  // Graceful stop. Joins the RT thread.
  // Returns false only when a required final safe-halt exhausted its bounded
  // retries. This is local SendCommand acceptance, not hardware acknowledgement.
  bool StopDriver();

  // Linearize a caller-owned authorization/mode mutation with every backend command send.
  // The callback MUST be short and noexcept in normal operation (typically one atomic mode
  // store).  If a normal command already won the send lock it completes before this transaction;
  // the optional zero-gain barrier is then ordered after it.  If this transaction wins first,
  // the callback's generation/token change makes every older in-flight policy result fail its
  // under-lock commit check.  Therefore, after a successful return, no command computed under the
  // previous authorization can be sent.
  bool CommitAuthorizationTransition(const std::function<void()>& transition,
                                     bool send_safe_halt_barrier = true) noexcept;

  const A3Watchdog& Watchdog() const noexcept { return watchdog_; }

  std::uint64_t PolicyTickCount() const noexcept { return policy_tick_count_.load(std::memory_order_relaxed); }
  std::uint64_t SafeHaltCount()   const noexcept { return safe_halt_count_.load(std::memory_order_relaxed); }
  std::uint64_t CommandFailureCount() const noexcept {
    return command_failure_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t DeadlineViolationCount() const noexcept {
    return deadline_violation_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t SendFailureCount() const noexcept {
    return send_failure_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t FinalSafeHaltCount() const noexcept {
    return final_safe_halt_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t AuthorizationChangeCount() const noexcept {
    return authorization_change_count_.load(std::memory_order_relaxed);
  }
  bool CommandFaulted() const noexcept {
    return command_faulted_.load(std::memory_order_acquire);
  }
  bool PublicationArmed() const noexcept {
    return publication_armed_.load(std::memory_order_acquire);
  }

  // Current value of the PR-9 PolicyFn tick counter (== PolicyTickCount).
  // Exposed as a distinct accessor so the test / diagnostic surface is
  // explicit about the contract (monotonic, advances only on Ok ticks).
  std::uint64_t TickIndex() const noexcept { return policy_tick_count_.load(std::memory_order_relaxed); }

 protected:
  void RunOnce() noexcept override;

 private:
  static a3_rt::A3BasedTask::Options BuildBaseOptions_(const A3PolicyDriverOptions& opt);

  void OnBackendState_(const robot_io::RobotState& state) noexcept;
  void EventThreadMain_() noexcept;
  void CommandWatchdogThreadMain_() noexcept;
  bool CommandPublishExpected_() noexcept;
  bool SafetyCommandRequired_(bool invocation_expected = false) const noexcept;
  void RunOnceWithState_(std::shared_ptr<const robot_io::RobotState> state) noexcept;
  bool SendSafeHalt_(const std::shared_ptr<const robot_io::RobotState>& state,
                     bool final_frame) noexcept;
  bool SendSafeHaltLocked_(const std::shared_ptr<const robot_io::RobotState>& state,
                           bool final_frame) noexcept;
  void LatchCommandFault_(const std::shared_ptr<const robot_io::RobotState>& state,
                          bool deadline) noexcept;

  robot_io::RobotIOBackend& backend_;
  PolicyFn                  policy_;
  CommandFn                 command_;
  A3PolicyDriverOptions     opt_;
  A3Watchdog                watchdog_;

  // Latest RobotState cached from the backend's state callback. Writer: the
  // backend's sync-loop thread (OnBackendState_); reader: the RT thread
  // (RunOnce). We use free-function std::atomic_load / atomic_store on
  // shared_ptr (GCC 11.4 libstdc++ workaround — the std::atomic<shared_ptr>
  // specialisation is not yet available there).
  std::shared_ptr<const robot_io::RobotState> latest_state_;

  std::mutex event_mtx_;
  std::condition_variable event_cv_;
  std::shared_ptr<const robot_io::RobotState> event_state_;
  std::thread event_thread_;
  std::uint64_t event_state_seq_{0};
  std::atomic<bool> event_running_{false};

  std::thread command_watchdog_thread_;
  std::atomic<bool> driver_started_{false};
  std::atomic<bool> command_watchdog_running_{false};
  std::atomic<bool> command_inflight_{false};
  std::atomic<bool> inflight_publish_expected_{false};
  std::atomic<std::int64_t> command_start_steady_ns_{0};
  std::atomic<bool> command_faulted_{false};
  // Single-writer arbitration for the backend's four-topic command transaction. Every
  // normal command and every halt takes this mutex. Normal sends re-check fault/mode/token
  // while holding it, so an older non-zero command can never overtake a halt.
  std::mutex backend_send_mtx_;
  std::atomic<bool> nonpublish_halt_pending_{false};
  // True iff the last successfully ordered backend transaction was a normal command.
  // Lets the supervisor recover a publish->nonpublish edge even if it first runs after
  // the transition and therefore has no previous predicate sample.
  std::atomic<bool> normal_command_live_{false};
  std::atomic<std::int64_t> last_safe_halt_attempt_steady_ns_{0};
  std::atomic<bool> publication_armed_{false};
  std::atomic<bool> final_safe_halt_succeeded_{true};
  int expected_dof_{0};

  std::atomic<std::uint64_t> policy_tick_count_{0};
  std::atomic<std::uint64_t> safe_halt_count_{0};
  std::atomic<std::uint64_t> command_failure_count_{0};
  std::atomic<std::uint64_t> deadline_violation_count_{0};
  std::atomic<std::uint64_t> send_failure_count_{0};
  std::atomic<std::uint64_t> final_safe_halt_count_{0};
  std::atomic<std::uint64_t> authorization_change_count_{0};
  // Increments under backend_send_mtx_ after every successfully accepted halt. A normal
  // callback snapshots it before inference and may not send if a halt overtook it.
  std::atomic<std::uint64_t> halt_generation_{0};
  std::atomic<bool> has_sent_command_{false};

  // Reusable buffers — avoid alloc in the RT hot path.
  std::array<double, 29> q_out_buf_{};
  robot_io::RobotCommand cmd_out_{};
  // The deadline supervisor may halt while the policy callback is still
  // writing cmd_out_. Keep its command storage disjoint to avoid a data race.
  robot_io::RobotCommand safe_halt_cmd_{};
};

}  // namespace a3_deploy
