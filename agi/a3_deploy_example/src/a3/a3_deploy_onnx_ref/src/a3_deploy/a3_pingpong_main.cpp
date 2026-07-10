// Minimal entry point for running model_15200 (ping-pong, 180-obs/31-act) on the
// A3 via AGI's native runner. Reuses robot_io::A3AimrtBackend (iceoryx/ros2 sync)
// + a3_deploy::A3PolicyDriver (50 Hz RT loop + watchdog + safe-halt) UNCHANGED;
// only the front-end is ours (a3_pingpong::PpPolicy CommandFn). AGI's original
// a3_deploy_onnx_ref + main.cpp are untouched (separate CMake target).
//
// Staged modes (keyboard, or --start MODE): PASSIVE (limp) -> PD_STAND (hold
// nominal) -> SHADOW (compute, no publish) -> MOTION (publish; --gain-scale for
// low-gain first try; 0/1 = swing level). Neck passive by default. Scripted
// racket targets only (no live planner).
//
// Usage:
//   a3_deploy_onnx_ref_pingpong --runtime-cfg PATH [--aimrt-cfg PATH]
//       [--start passive|pd_stand|shadow|motion] [--level 0|1]
//       [--gain-scale F] [--stand-kp K --stand-kd D]
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <termios.h>
#include <unistd.h>

#include <yaml-cpp/yaml.h>

#include "a3_deploy/a3_policy_driver.hpp"
#include "a3_pingpong/pp_policy.hpp"
#include "a3_pingpong/pp_reference_playback.hpp"
#include "a3_pingpong/pp_runtime_interlocks.hpp"
#include "robot_io/a3_aimrt_backend.hpp"

namespace {
namespace fs = std::filesystem;

std::atomic<bool> g_stop{false};
volatile std::sig_atomic_t g_signal_stop = 0;
void OnSig(int) { g_signal_stop = 1; }
bool StopRequested() {
  return g_signal_stop != 0 || g_stop.load(std::memory_order_acquire);
}

enum class Mode { kPassive, kPdStand, kShadow, kMotion, kReferencePlayback };
const char* ModeName(Mode m) {
  switch (m) {
    case Mode::kPassive: return "PASSIVE";
    case Mode::kPdStand: return "PD_STAND";
    case Mode::kShadow: return "SHADOW(no-publish)";
    case Mode::kMotion: return "MOTION";
    case Mode::kReferencePlayback: return "REFERENCE_PLAYBACK";
  }
  return "?";
}

// Atomically bind the mode value to a monotonically increasing authorization generation.
// A plain atomic<Mode> cannot tell the policy driver that MOTION -> PASSIVE happened while a
// callback was running because both endpoints are publish-capable. Keeping both fields in one
// atomic word also avoids the torn (new mode, old epoch) window of two separate atomics.
class ModeState {
 public:
  struct Snapshot {
    Mode mode;
    std::uint64_t generation;
    std::uint64_t token;
  };

  explicit ModeState(Mode initial) : state_(Encode_(initial, 0)) {}

  Mode load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
    return static_cast<Mode>(state_.load(order) & kModeMask);
  }

  // Must be invoked only by A3PolicyDriver::CommitAuthorizationTransition once the driver exists.
  // Keeping the primitive explicit makes accidental unsynchronised mode stores easy to audit.
  void store_under_driver_lock(Mode next) noexcept {
    std::uint64_t current = state_.load(std::memory_order_relaxed);
    for (;;) {
      const std::uint64_t generation = (current >> kModeBits) + 1;
      const std::uint64_t desired = Encode_(next, generation);
      if (state_.compare_exchange_weak(current, desired, std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {
        return;
      }
    }
  }

  std::uint64_t authorization_token() const noexcept {
    return state_.load(std::memory_order_acquire);
  }

  Snapshot snapshot() const noexcept {
    const std::uint64_t token = state_.load(std::memory_order_acquire);
    return Snapshot{static_cast<Mode>(token & kModeMask), token >> kModeBits, token};
  }

 private:
  static constexpr unsigned kModeBits = 8;
  static constexpr std::uint64_t kModeMask = (std::uint64_t{1} << kModeBits) - 1;
  static std::uint64_t Encode_(Mode mode, std::uint64_t generation) noexcept {
    return (generation << kModeBits) | static_cast<std::uint64_t>(mode);
  }

  std::atomic<std::uint64_t> state_;
};

bool ValidateCommandLine(int argc, char** argv, std::string& error) {
  static const std::unordered_set<std::string> value_flags = {
      "--aimrt-cfg", "--ankle-gain-scale", "--cmd-timeout", "--command-deadline-ms",
      "--engage-min-tts", "--effort-limit-ratio", "--gain-scale", "--invalid-grace", "--leg-clamp-rad",
      "--hold-recover", "--leg-gain-scale", "--leg-smooth-alpha", "--level", "--loc-mode", "--mode",
      "--model-path", "--motion-blend-sec", "--obs-csv", "--oracle-max-age",
      "--oracle-shm", "--ref-amplitude", "--ref-frequency", "--ref-gain-scale",
      "--ref-group", "--ref-max-err", "--ref-stale-ms", "--runtime-cfg",
      "--squat-guard-rad", "--stand-kd", "--stand-kp", "--start", "--swing-rest",
      "--swing-speed", "--tilt-guard", "--trace-csv", "--vel-gate-margin",
      "--warmup-sec"};
  static const std::unordered_set<std::string> switch_flags = {
      "--auto-leg-hold", "--backhand", "--base-estimator", "--dry-run",
      "--leg-stand-gains", "--legs-passive", "--no-fall-guard", "--no-imu-yaw",
      "--no-publish", "--no-yaw-align", "--official-stand", "--oracle-pelvis",
      "--perfect-tracking", "--planner", "--reference-playback", "--shadow-frozen-clock",
      "--single-swing", "--stream-target", "--use-imu-yaw", "--vel-box-center",
      "--waist-passive"};
  std::unordered_set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (value_flags.count(arg) == 0 && switch_flags.count(arg) == 0) {
      error = "unknown or positional argument '" + arg + "'";
      return false;
    }
    if (!seen.insert(arg).second) {
      error = "duplicate argument '" + arg + "'";
      return false;
    }
    if (value_flags.count(arg) != 0) {
      if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
        error = "argument '" + arg + "' requires exactly one value";
        return false;
      }
      ++i;
    }
  }
  return true;
}

std::string Flag(int argc, char** argv, const char* name, const std::string& def) {
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == name) return argv[i + 1];
  return def;
}
bool Has(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == name) return true;
  return false;
}
std::string Resolve(const std::string& p, const fs::path& base) {
  fs::path fp(p);
  if (fp.is_absolute() || fs::exists(fp)) return fp.lexically_normal().string();

  std::error_code ec;
  fs::path cursor = fs::weakly_canonical(base, ec);
  if (ec) cursor = fs::absolute(base, ec);
  if (ec) return (base / fp).lexically_normal().string();
  if (!fs::is_directory(cursor, ec)) cursor = cursor.parent_path();

  while (!cursor.empty()) {
    const fs::path candidate = (cursor / fp).lexically_normal();
    if (fs::exists(candidate)) return candidate.string();
    const fs::path parent = cursor.parent_path();
    if (parent == cursor) break;
    cursor = parent;
  }

  return (base / fp).lexically_normal().string();
}

std::string BuildBackendCfg(const YAML::Node& backend, const std::string& aimrt_override,
                            const fs::path& cfgdir, bool no_publish) {
  std::ostringstream ss;
  bool first = true;
  auto add = [&](const std::string& k, const std::string& v) {
    if (!first) ss << ',';
    ss << k << '=' << v;
    first = false;
  };
  std::string aimrt = aimrt_override.empty() ? backend["aimrt_cfg_path"].as<std::string>()
                                             : aimrt_override;
  add("cfg_file_path", Resolve(aimrt, cfgdir));
  if (!backend["sync_mode"]) add("sync_mode", "min_skew_pair");
  if (!backend["sync_hz"]) add("sync_hz", "100");
  for (auto it : backend) {
    const std::string k = it.first.as<std::string>();
    if (k == "aimrt_cfg_path") continue;
    if (it.second.IsScalar()) add(k, it.second.as<std::string>());
  }
  if (no_publish) add("publish_enabled", "false");
  return ss.str();
}

Mode ParseStartMode(const std::string& s, Mode def) {
  if (s == "passive") return Mode::kPassive;
  if (s == "pd_stand") return Mode::kPdStand;
  if (s == "shadow") return Mode::kShadow;
  if (s == "motion") return Mode::kMotion;
  return def;
}

a3_pingpong::RefPlaybackGroup ParseRefGroup(const std::string& s) {
  if (s == "neck" || s == "head" || s == "neck_head_hold") {
    return a3_pingpong::RefPlaybackGroup::kNeckHeadHold;
  }
  if (s == "waist") return a3_pingpong::RefPlaybackGroup::kWaist;
  if (s == "right_shoulder") return a3_pingpong::RefPlaybackGroup::kRightShoulder;
  if (s == "right_elbow_wrist") return a3_pingpong::RefPlaybackGroup::kRightElbowWrist;
  if (s == "right_arm") return a3_pingpong::RefPlaybackGroup::kRightArm;
  if (s == "waist_right_arm") return a3_pingpong::RefPlaybackGroup::kWaistRightArm;
  if (s == "legs" || s == "legs_hold") return a3_pingpong::RefPlaybackGroup::kLegsHold;
  if (s == "upper_body") return a3_pingpong::RefPlaybackGroup::kUpperBody;
  return a3_pingpong::RefPlaybackGroupFromInt(std::stoi(s));
}

// Per-joint tracking/amplitude block over the last status window (SHADOW/MOTION).
// cmd_range = how far the policy COMMANDS the joint to move; meas_range = how far
// it ACTUALLY moves; trk% = meas/cmd (low => the joint can't follow the command).
void PrintDiagBlock(const a3_pingpong::PpPolicy::DiagSnapshot& d, bool legs_passive) {
  if (!d.valid) return;
  const auto& nm = a3_pingpong::backend_joint_order();
  auto row = [&](int i) {
    const double cr = d.des_range[i], mr = d.meas_range[i];
    std::printf("   %-26s des=%+0.3f q=%+0.3f err=%+0.3f | cmdR=%.3f measR=%.3f trk=%3.0f%% qdpk=%.2f\n",
                nm[i].c_str(), d.q_des[i], d.q_meas[i], d.q_des[i] - d.q_meas[i],
                cr, mr, cr > 1e-3 ? 100.0 * mr / cr : 0.0, d.qd_peak[i]);
  };
  auto group = [&](const char* title, int lo, int hi) {
    std::printf("  -- %s --\n", title);
    for (int i = lo; i <= hi; ++i) row(i);
  };
  // worst tracker among a range (max cmd_range with low trk) for a compact summary
  auto summary = [&](const char* title, int lo, int hi) {
    int wi = lo; double worst = -1;
    for (int i = lo; i <= hi; ++i) {
      const double cr = d.des_range[i];
      const double miss = cr - d.meas_range[i];  // unfollowed command
      if (cr > 0.02 && miss > worst) { worst = miss; wi = i; }
    }
    std::printf("  -- %s -- worst: %s cmdR=%.3f measR=%.3f errpk=%.3f\n", title,
                nm[wi].c_str(), d.des_range[wi], d.meas_range[wi], d.err_peak[wi]);
  };
  std::printf(" [diag] (rad, last window)   des/q/err | cmdR=commanded-range measR=measured-range trk=follow%%\n");
  group("WAIST", 0, 2);
  group("RIGHT ARM (forehand)", 12, 18);
  group("LEFT ARM", 5, 11);
  if (legs_passive) summary("LEGS (held nominal)", 19, 30);
  else group("LEGS (policy-driven)", 19, 30);  // per-joint hip/knee/ankle des/q/err for knee-sink diag
  summary("NECK (passive)", 3, 4);
}

// Obs-debug block: obs vector stats + the localization-dependent slices, so you
// can confirm at a glance that motion_anchor_pos_b ~ 0 in perfect-tracking mode.
// Index map (see pp_obs_builder.hpp build_obs_180):
//   command [0..61] | anchor_pos_b [62..64] | anchor_ori_b [65..70] |
//   base_ang_vel [71..73] | joint_pos_rel [74..104] | joint_vel [105..135] |
//   last_action [136..166] | proj_grav [167..169] | base_target_pos_b [170..171] |
//   racket_target_pos_b [172..174] | racket_target_vel_w [175..177] |
//   time_to_strike [178] | swing_type [179]
void PrintObsDebugBlock(const a3_pingpong::PpPolicy::ObsDebug& d,
                        const Eigen::VectorXd& action,
                        const std::string& planner_status = "") {
  if (!d.valid) return;
  const auto& o = d.obs;
  const long n = (long)o.size();
  const double omin = o.minCoeff(), omax = o.maxCoeff(), omean = o.mean();
  std::printf(" [obs] loc=%s oracle(en=%d fresh=%d age=%.3fs) sync_miss=%llu | dim=%ld "
              "obs[min/mean/max]=[%.3f %.3f %.3f]\n",
              d.oracle_enabled ? "oracle" : "non-oracle",
              d.oracle_enabled ? 1 : 0, d.oracle_fresh ? 1 : 0, d.oracle_age_s,
              (unsigned long long)d.sync_miss, n, omin, omean, omax);
  char src[64];
  if (planner_status.empty()) std::snprintf(src, sizeof src, "SCRIPTED target -- no live planner");
  else std::snprintf(src, sizeof src, "PLANNER: %s", planner_status.c_str());
  // Per-layout goal-block slices. 2026-07-07 fix: this used to classify 175-vs-'everything
  // else = 180' and indexed o[170..179] on a 177-D (and would on a 110-D) obs — OUT OF BOUNDS
  // (Eigen UB in release). Every supported layout now has its own offsets; unknown dims print
  // only the stats line above.
  if (n == a3_pingpong::kObsDim110) {
    // hitter_pure: [96:99] grav, [99:101] e_base,x, [101:103] Δstation(world),
    // [103:106] racket rel base(world), [106:109] vel_w, [109] tts. No swing_type.
    std::printf("   e_base_x=[%+.3f %+.3f]  base_target_dxy=[%+.4f %+.4f]  "
                "racket_rel_base=[%+.4f %+.4f %+.4f]\n",
                o[99], o[100], o[101], o[102], o[103], o[104], o[105]);
    std::printf("   racket_target_vel_w=[%+.3f %+.3f %+.3f]  tts=%.3f  [%s]\n",
                o[106], o[107], o[108], o[109], src);
  } else if (n == a3_pingpong::kObsDim175 || n == a3_pingpong::kObsDim177 ||
             n == a3_pingpong::kObsDim) {
    const bool dp175 = (n == a3_pingpong::kObsDim175);
    const bool hp177 = (n == a3_pingpong::kObsDim177);
    if (!dp175 && !hp177) {  // 180 full: anchor_pos + base_target blocks exist
      const double anchor_pos_norm = o.segment<3>(62).norm();
      std::printf("   motion_anchor_pos_b=[%+.4f %+.4f %+.4f] |.|=%.4f  base_target_pos_b=[%+.4f %+.4f]\n",
                  o[62], o[63], o[64], anchor_pos_norm, o[170], o[171]);
    }
    if (hp177)
      std::printf("   base_target_pos_b=[%+.4f %+.4f]\n", o[167], o[168]);
    // racket_target_pos_b start: 180 -> 172 (rel base); 175 -> 167 (rel FK); 177 -> 169 (rel FK).
    const int rp = dp175 ? 167 : hp177 ? 169 : 172;
    std::printf("   racket_target_pos_b=[%+.4f %+.4f %+.4f]  racket_target_vel_w=[%+.3f %+.3f %+.3f]  "
                "tts=%.3f swing=%+.0f(%s)  [%s]\n",
                o[rp], o[rp + 1], o[rp + 2], o[rp + 3], o[rp + 4], o[rp + 5], o[rp + 6], o[rp + 7],
                o[rp + 7] >= 0 ? "FOREHAND" : "BACKHAND", src);
  }
  if (action.size() == a3_pingpong::kNumJoints)
    std::printf("   action[min/mean/max]=[%+.3f %+.3f %+.3f] |a|=%.3f\n",
                action.minCoeff(), action.mean(), action.maxCoeff(), action.norm());
}

void PrintRefDiagBlock(const a3_pingpong::RefPlaybackDiagSnapshot& d) {
  if (!d.valid) return;
  const auto& nm = a3_pingpong::backend_joint_order();
  std::printf(" [ref] group=%s moving=%d fault=%d reason=%s tick=%llu time=%.3f max_abs_err=%.4f\n",
              a3_pingpong::RefPlaybackGroupName(d.group), d.moving ? 1 : 0,
              d.faulted ? 1 : 0, d.fault_reason.empty() ? "-" : d.fault_reason.c_str(),
              (unsigned long long)d.tick, d.time_s, d.max_abs_err);
  for (int i = 0; i < d.active_count; ++i) {
    const int s = d.active_slots[i];
    const double qd = d.q_des.size() == a3_pingpong::kRefDof ? d.q_des[s] : 0.0;
    const double qm = d.q_meas.size() == a3_pingpong::kRefDof ? d.q_meas[s] : 0.0;
    const double kp = d.kp.size() == a3_pingpong::kRefDof ? d.kp[s] : 0.0;
    const double kd = d.kd.size() == a3_pingpong::kRefDof ? d.kd[s] : 0.0;
    std::printf("   %-26s sdk=%02d q_des=%+0.4f q_meas=%+0.4f err=%+0.4f "
                "kp=%0.2f kd=%0.2f group=%s tick=%llu time=%.3f\n",
                nm[s].c_str(), s, qd, qm, qd - qm, kp, kd,
                a3_pingpong::RefPlaybackGroupName(d.group),
                (unsigned long long)d.tick, d.time_s);
  }
}
}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffer so status survives kill
  if (std::signal(SIGINT, OnSig) == SIG_ERR ||
      std::signal(SIGTERM, OnSig) == SIG_ERR) {
    std::cerr << "failed to install SIGINT/SIGTERM handlers\n";
    return 1;
  }
  std::string cli_error;
  if (!ValidateCommandLine(argc, argv, cli_error)) {
    std::cerr << "invalid command line: " << cli_error << "\n";
    return 2;
  }
  const std::string cfg_path = Flag(argc, argv, "--runtime-cfg", "");
  if (cfg_path.empty()) {
    std::cerr << "usage: " << argv[0]
              << " --runtime-cfg PATH [--aimrt-cfg PATH] [--start passive|pd_stand|shadow|motion]"
                 " [--level 0|1]\n"
                 "       [--backhand] [--legs-passive] [--waist-passive] [--auto-leg-hold]"
                 " [--model-path PATH] [--gain-scale F] [--swing-speed F] [--stand-kp K --stand-kd D]\n"
                 "       [--reference-playback|--mode reference-playback]"
                 " [--no-publish|--dry-run] [--warmup-sec S]\n"
                 "       [--command-deadline-ms MS] (must be >0 and shorter than one policy period)\n"
                 "       [--effort-limit-ratio R] (0<R<=1; PD+measured effort safety envelope)\n"
                 "       [--planner] (LIVE planner: racket <- /racket/command_flat, base <- /a3/base_pose_flat over ros2;"
                 " [--engage-min-tts S] [--cmd-timeout S] [--invalid-grace S]"
                 " [--hold-recover S] [--vel-gate-margin M])\n"
                 "       [--loc-mode fabricated|perfect_tracking|oracle|external_base]"
                 " [--perfect-tracking] [--oracle-pelvis] [--no-imu-yaw]\n"
                 "       [--oracle-shm PATH] [--oracle-max-age S]"
                 " [--trace-csv PATH] [--obs-csv PATH] [--shadow-frozen-clock]\n"
                 "       [--leg-gain-scale F] [--ankle-gain-scale F] [--motion-blend-sec S]"
                 " [--squat-guard-rad R] [--tilt-guard G] [--leg-clamp-rad R] [--leg-stand-gains] [--leg-smooth-alpha A]\n";
    return 2;
  }
  const fs::path cfgdir = fs::path(cfg_path).parent_path();
  YAML::Node cfg = YAML::LoadFile(cfg_path);
  const double configured_policy_hz = cfg["policy_driver"]["policy_hz"]
                                          ? cfg["policy_driver"]["policy_hz"].as<double>()
                                          : 50.0;
  if (!std::isfinite(configured_policy_hz) || configured_policy_hz <= 0.0) {
    std::cerr << "policy_driver.policy_hz must be finite and positive; got "
              << configured_policy_hz << "\n";
    return 2;
  }

  const std::string run_mode = Flag(argc, argv, "--mode", "");
  if (!run_mode.empty() && run_mode != "reference-playback") {
    std::cerr << "--mode accepts only 'reference-playback'; use --start for staged runner modes\n";
    return 2;
  }
  const bool reference_playback_selected =
      Has(argc, argv, "--reference-playback") || run_mode == "reference-playback";
  const bool no_publish = Has(argc, argv, "--no-publish") || Has(argc, argv, "--dry-run");
  const bool no_yaw_align = Has(argc, argv, "--no-yaw-align");
  if (no_yaw_align && !no_publish) {
    std::cerr << "--no-yaw-align is diagnostic-only and requires --no-publish/--dry-run\n";
    return 2;
  }

  // LIVE PLANNER mode (Path B): racket target from /racket/command_flat + mocap base pose
  // from /a3/base_pose_flat, both over the AimRT ros2 backend; body-drive stays iceoryx.
  const bool planner_mode = Has(argc, argv, "--planner");

  const std::string aimrt_override =
      Resolve(Flag(argc, argv, "--aimrt-cfg", ""), cfgdir);
  std::string aimrt_override_arg =
      Has(argc, argv, "--aimrt-cfg") ? aimrt_override : std::string{};
  // In planner mode, default the AimRT transport cfg to the dual-plugin (iceoryx body-drive
  // + ros2 planner inputs) unless the operator passed an explicit --aimrt-cfg.
  if (planner_mode && aimrt_override_arg.empty())
    aimrt_override_arg = Resolve("a3_aimrt_config.pingpong_ros2body.yaml", cfgdir);

  const std::string model_path = Resolve(
      Flag(argc, argv, "--model-path", cfg["onnx"]["model_path"].as<std::string>()), cfgdir);
  const int level = std::stoi(Flag(argc, argv, "--level", "1"));
  if (level != 0 && level != 1) {
    std::cerr << "--level must be 0 or 1\n";
    return 2;
  }
  std::atomic<double> gain_scale{std::stod(Flag(argc, argv, "--gain-scale", "1.0"))};
  const double stand_kp = std::stod(Flag(argc, argv, "--stand-kp", "60"));
  const double stand_kd = std::stod(Flag(argc, argv, "--stand-kd", "4"));
  // The official a3_pd_stand gains (knee ~2000) are tuned for the robot bearing
  // its weight ON THE GROUND. On a HOIST they snap/buzz and swing the body, so
  // they are OFF by default; opt in only for free-standing on the ground
  // (Step 2). The hoisted demo uses the gentle flat PD that ran clean before.
  const bool official_stand = Has(argc, argv, "--official-stand");
  // Stretch the swing in real time (<1.0 = slower) so hardware actuators can
  // track it. Native (1.0) under-shoots and strains loudly on the real robot.
  const double swing_speed = std::stod(Flag(argc, argv, "--swing-speed", "1.0"));
  if (!std::isfinite(swing_speed) || swing_speed < 0.05 || swing_speed > 2.0) {
    std::cerr << "--swing-speed must be finite and within [0.05,2.0]\n";
    return 2;
  }
  const double effort_limit_ratio = std::stod(
      Flag(argc, argv, "--effort-limit-ratio", "1.0"));
  if (!std::isfinite(effort_limit_ratio) || effort_limit_ratio <= 0.0 ||
      effort_limit_ratio > 1.0) {
    std::cerr << "--effort-limit-ratio must be finite in (0,1]; values above the model's "
                 "actuator envelope are forbidden\n";
    return 2;
  }
  if (planner_mode && std::fabs(swing_speed - 1.0) > 1e-12) {
    std::cerr << "--planner requires --swing-speed 1.0: planner time_to_strike already owns "
                 "physical time, so scaling the reference clock again makes contact late\n";
    return 2;
  }

  // ---- localization mode (A/B/C) + sim-only oracle config ----
  // yaml: obs_debug.{loc_mode,use_sim_oracle_pelvis_pose,oracle_shm_path,
  //                  oracle_max_age_s,obs_csv}. CLI overrides yaml.
  YAML::Node odbg = cfg["obs_debug"] ? cfg["obs_debug"] : YAML::Node();
  auto odbg_str = [&](const char* k, const std::string& def) {
    return odbg[k] ? odbg[k].as<std::string>() : def;
  };
  std::string loc_mode_s = odbg_str("loc_mode", "perfect_tracking");  // hardware-safe default
  bool yaml_oracle = odbg["use_sim_oracle_pelvis_pose"] &&
                     odbg["use_sim_oracle_pelvis_pose"].as<bool>();
  if (yaml_oracle) loc_mode_s = "oracle";
  if (Has(argc, argv, "--loc-mode")) loc_mode_s = Flag(argc, argv, "--loc-mode", loc_mode_s);
  if (Has(argc, argv, "--perfect-tracking")) loc_mode_s = "perfect_tracking";
  if (Has(argc, argv, "--oracle-pelvis")) loc_mode_s = "oracle";
  // Planner mode needs the target frame to match a REAL base: default to live mocap base
  // (external_base) unless the operator picked a loc mode explicitly (e.g. --oracle-pelvis
  // for the sim closed-loop rehearsal, where sim ground truth plays the mocap role).
  const bool loc_mode_explicit = Has(argc, argv, "--loc-mode") ||
      Has(argc, argv, "--perfect-tracking") || Has(argc, argv, "--oracle-pelvis");
  const int loc_selector_count = (Has(argc, argv, "--loc-mode") ? 1 : 0) +
      (Has(argc, argv, "--perfect-tracking") ? 1 : 0) +
      (Has(argc, argv, "--oracle-pelvis") ? 1 : 0);
  if (loc_selector_count > 1) {
    std::cerr << "choose exactly one localization selector: --loc-mode, --perfect-tracking, "
                 "or --oracle-pelvis\n";
    return 2;
  }
  if (planner_mode && !loc_mode_explicit) loc_mode_s = "external_base";
  a3_pingpong::LocMode loc_mode = a3_pingpong::LocMode::kFabricated;
  if (loc_mode_s == "perfect_tracking" || loc_mode_s == "B" || loc_mode_s == "b")
    loc_mode = a3_pingpong::LocMode::kPerfectTracking;
  else if (loc_mode_s == "oracle" || loc_mode_s == "C" || loc_mode_s == "c")
    loc_mode = a3_pingpong::LocMode::kOracle;
  else if (loc_mode_s == "external_base" || loc_mode_s == "mocap")
    loc_mode = a3_pingpong::LocMode::kExternalBase;
  else if (loc_mode_s == "fabricated" || loc_mode_s == "A" || loc_mode_s == "a")
    loc_mode = a3_pingpong::LocMode::kFabricated;
  else { std::cerr << "unknown loc_mode '" << loc_mode_s << "'\n"; return 2; }
  const std::string oracle_shm =
      Flag(argc, argv, "--oracle-shm", odbg_str("oracle_shm_path", "/dev/shm/pp_oracle_pelvis"));
  const double oracle_max_age_s = std::stod(
      Flag(argc, argv, "--oracle-max-age", odbg["oracle_max_age_s"]
               ? std::to_string(odbg["oracle_max_age_s"].as<double>()) : "0.1"));
  const std::string obs_csv_path = Flag(argc, argv, "--obs-csv", odbg_str("obs_csv", ""));

  const std::string start_arg = Flag(argc, argv, "--start", "passive");
  if (start_arg != "passive" && start_arg != "pd_stand" && start_arg != "shadow" &&
      start_arg != "motion") {
    std::cerr << "--start must be passive|pd_stand|shadow|motion\n";
    return 2;
  }
  Mode default_mode = ParseStartMode(start_arg, Mode::kPassive);
  // Optional PD_STAND warmup: hold nominal for N s (robot settles upright),
  // then auto-switch to the requested mode. Matches a safe bring-up + lets a
  // non-interactive run reach MOTION from a stable stand.
  const double warmup_sec = std::stod(Flag(argc, argv, "--warmup-sec", "0"));
  const Mode target_mode = default_mode;
  ModeState mode{warmup_sec > 0 ? Mode::kPdStand : default_mode};

  // --- backend ---
  auto backend = std::make_unique<robot_io::A3AimrtBackend>();
  const std::string backend_cfg =
      BuildBackendCfg(cfg["backend"], aimrt_override_arg, cfgdir, no_publish);
  std::cout << "[pingpong] backend cfg: " << backend_cfg << "\n";
  if (!backend->Init(backend_cfg)) { std::cerr << "backend Init failed\n"; return 1; }
  std::cout << "[pingpong] A3AimrtBackend initialised; model=" << model_path << "\n";

  // --- our front-end ---
  a3_pingpong::PpPolicyConfig pcfg;
  // Planner mode MUST start idle (level 0): the swing level is driven ONLY by the engage
  // machine. Starting at level 1 makes PlannerEngageStep_ see "already swinging" on the
  // first tick and skip the real engage (target/velocity never frozen -> a dead swing).
  pcfg.level = planner_mode ? 0 : level;
  pcfg.legs_passive = Has(argc, argv, "--legs-passive");  // hold legs (hoisted demo)
  // Also hold the WAIST (slots 0..2) at nominal — keeps the torso CoM over the feet
  // when the static legs can't rebalance the policy's forward waist_pitch command.
  // ARMS-ONLY swing. With --official-stand the held waist uses official gains too.
  pcfg.waist_passive = Has(argc, argv, "--waist-passive");
  // AUTO LEG-HOLD: dynamically hold legs+waist at level 0 (stable ready stand, no
  // frozen-windup foot-lift) and release them at level 1 (full-body self-balancing swing).
  // Overrides the manual flags; the initial hold follows the START level.
  pcfg.auto_leg_hold = Has(argc, argv, "--auto-leg-hold");
  if (pcfg.auto_leg_hold) pcfg.legs_passive = pcfg.waist_passive = (level == 0);
  // GROUND held joints use AGI's official ground-stand gains (the ONLY config verified to
  // stand free on the ground) when --official-stand is set; the held POSE is identical
  // (nominal == official_stand_q), only the GAINS change. Released joints use the policy PD.
  // (Banner only — the gain loop recomputes this per-tick from the live hold state.)
  const bool legs_official_gains = pcfg.legs_passive && official_stand;
  const bool waist_official_gains = pcfg.waist_passive && official_stand;
  // LEG q_des CLAMP: cap how far the released (level-1) legs may deviate from the
  // nominal upright stand. The trained swing commands a deep crouch (hip_pitch/ankle
  // -0.6..-0.9 rad) that sinks the real robot; clamp to nominal ± band so the legs
  // stay weight-bearing while the arms+waist swing. 0 = off (full policy legs).
  pcfg.leg_clamp_rad = std::stod(Flag(argc, argv, "--leg-clamp-rad", "0.0"));
  // LEG q_des LOW-PASS: EMA-smooth the released leg q_des so stiff --leg-stand-gains track a
  // smooth reference instead of the policy jitter (which they amplify into a twitch). 1.0=off;
  // 0.2-0.3 = moderate. Clamp to (0,1].
  pcfg.leg_smooth_alpha = std::min(1.0, std::max(0.02, std::stod(Flag(argc, argv, "--leg-smooth-alpha", "1.0"))));
  pcfg.swing_speed = swing_speed;
  pcfg.dt = 1.0 / configured_policy_hz;
  pcfg.use_base_estimator = Has(argc, argv, "--base-estimator");  // leg-FK pelvis height (ground)
  pcfg.loc_mode = loc_mode;
  pcfg.diagnostic_no_publish = no_publish;
  // DEFAULT ON since 2026-07-03: with yaw_align the base yaw is engage-relative (starts at
  // identity, tracks the robot's REAL turning) — matching training's rotate-by-current-yaw
  // target transform. Required for turning models (model_9000 turns ~84 deg by design).
  // --no-imu-yaw reverts to the legacy identity-yaw transform (only sensible for a
  // non-turning model, e.g. p4); --use-imu-yaw is still accepted (now a no-op).
  pcfg.use_imu_yaw_for_targets = !Has(argc, argv, "--no-imu-yaw");
  // Scripted swing direction: default forehand; --backhand mirrors the target to
  // +y and selects the baked backhand clip. Toggle live with f/b. (No live planner.)
  pcfg.start_backhand = Has(argc, argv, "--backhand");
  pcfg.oracle_max_age_s = oracle_max_age_s;
  // SINGLE-SWING / REST: avoid the periodic clock's end->windup reference SNAP (untracked
  // in training; topples the free-base backhand). --single-swing: one swing per '1' press,
  // then held stand. --swing-rest S: swing, rest S seconds at held stand, swing again
  // (continuous demo, every swing from a clean windup start).
  pcfg.single_swing = Has(argc, argv, "--single-swing");
  if (Has(argc, argv, "--swing-rest"))
    pcfg.swing_rest_s = std::stod(Flag(argc, argv, "--swing-rest", "1.5"));
  // LIVE PLANNER: one clip per engage + a short settle before re-engaging (the wbc_runner
  // swing_rest semantics). single_swing gives the linear clock + clip-end completion the
  // engage machine relies on; swing_rest_s>=0 arms the inter-swing rest timer.
  pcfg.planner_mode = planner_mode;
  if (planner_mode) {
    pcfg.single_swing = true;
    if (!Has(argc, argv, "--swing-rest")) pcfg.swing_rest_s = 0.5;
    pcfg.engage_min_tts_s = std::stod(Flag(argc, argv, "--engage-min-tts", "1.0"));
    pcfg.command_timeout_s = std::stod(Flag(argc, argv, "--cmd-timeout", "0.5"));
    pcfg.planner_invalid_grace_s = std::stod(Flag(argc, argv, "--invalid-grace", "0.25"));
    pcfg.hold_recover_s = std::stod(Flag(argc, argv, "--hold-recover", "2.5"));
    // 110-D per-clip trained-velocity-box gate slack (m/s per axis); see PpPolicyConfig.
    pcfg.gate_vel_margin = std::stod(Flag(argc, argv, "--vel-gate-margin", "0.30"));
    // Mid-swing target streaming: OFF unless the model trained midswing_resample_prob > 0
    // (the 13200 baseline trained 0.0 — see PpPolicyConfig::stream_target).
    pcfg.stream_target = Has(argc, argv, "--stream-target");
    // Demo-robustness vel mode: command the trained box-center velocity; planner keeps
    // WHERE + WHEN (see PpPolicyConfig::vel_cmd_box_center).
    pcfg.vel_cmd_box_center = Has(argc, argv, "--vel-box-center");
  }
  // YAW-ALIGN default ON (hardware fix: boot-drifted IMU yaw polluted motion_anchor_ori_b
  // by a constant -12..-38 deg in MDU captures -> the policy fought a fictional torso yaw
  // error with legs/waist and fell during free-standing swings; no-op in sim). Opt out for
  // A/B debugging only.
  pcfg.yaw_align = !no_yaw_align;
  std::unique_ptr<a3_pingpong::PpPolicy> pp;
  try {
    pp = std::make_unique<a3_pingpong::PpPolicy>(model_path, pcfg);
  } catch (const std::exception& e) {
    std::cerr << "[pp PREFLIGHT] model/runtime contract rejected before backend start: "
              << e.what() << "\n";
    return 3;
  }

  // ---- LIVE PLANNER input wiring (Path B) ----
  // Backend AimRT subscribers (set BEFORE Start()) push decoded Float64MultiArrays into
  // thread-safe holders that PpPolicy reads on the 50 Hz driver thread. The racket topic
  // feeds the engage machine; the base topic feeds LocMode::kExternalBase.
  if (planner_mode) {
    auto racket_in = std::make_shared<a3_pingpong::PpRacketTargetInput>();
    auto base_in = std::make_shared<a3_pingpong::PpBasePoseInput>();
    pp->SetRacketInput(racket_in);
    pp->SetBasePoseInput(base_in);
    backend->SetRacketTargetCallback(
        [racket_in](const std::vector<double>& a) { racket_in->SetFromFlat(a); });
    backend->SetBasePoseCallback(
        [base_in](const std::vector<double>& a) { base_in->SetFromFlat(a); });
    std::cout << "[pingpong] LIVE PLANNER: racket <- /racket/command_flat, base <- "
                 "/a3/base_pose_flat (std_msgs/Float64MultiArray, ros2); body-drive iceoryx\n";
  }

  // ---- SIM-ONLY oracle localization wiring ----
  // The shm file is produced by scripts/oracle_pose_bridge.py (an rclpy node
  // subscribing /sim/a3/pelvis_pose). On hardware it does not exist, Open() fails,
  // and oracle mode falls back to perfect-tracking with a loud warning.
  if (loc_mode == a3_pingpong::LocMode::kOracle) {
    std::fprintf(stderr,
        "\n*** ORACLE LOCALIZATION ENABLED -- SIMULATION ONLY. DO NOT USE ON "
        "HARDWARE. ***\n    reading true MuJoCo pelvis pose from %s\n\n",
        oracle_shm.c_str());
    auto oracle = std::make_shared<a3_pingpong::PpOraclePose>();
    if (!oracle->Open(oracle_shm)) {
      std::fprintf(stderr,
          "[oracle] shm '%s' not present -> oracle UNAVAILABLE. Start the bridge:\n"
          "    python3 scripts/oracle_pose_bridge.py --shm %s\n"
          "  (oracle mode will fall back to perfect-tracking until then)\n",
          oracle_shm.c_str(), oracle_shm.c_str());
    } else {
      pp->SetOracle(oracle);
    }
  }
  std::cout << "[pingpong] localization mode = " << pp->loc_mode_name() << "\n";
  std::cout << "[pingpong] racket/base target yaw frame = "
            << (pcfg.use_imu_yaw_for_targets
                    ? "IMU-yaw (absolute; needs a real world-yaw localizer)"
                    : "robot-heading (+x; IMU yaw ignored -- hardware-safe)")
            << "\n";
  const Eigen::VectorXd stand_q =
      a3_pingpong::to_sdk_order(pp->onnx().default_q(), pp->isaac_to_sdk());
  // ONNX metadata follows the model/Isaac joint order, while RobotState and
  // RobotCommand use backend SDK slots.  Safety limits must be scattered by
  // name exactly like q/kp/kd; applying them positionally checks the wrong
  // actuator whenever the two orders differ.
  const Eigen::VectorXd effort_limits_sdk =
      pp->onnx().effort_limits().size() == a3_pingpong::kNumJoints
          ? a3_pingpong::to_sdk_order(pp->onnx().effort_limits(), pp->isaac_to_sdk())
          : Eigen::VectorXd{};
  a3_pingpong::RefPlaybackConfig rcfg;
  rcfg.dt = 1.0 / configured_policy_hz;
  rcfg.amplitude_rad = std::stod(Flag(argc, argv, "--ref-amplitude", "0.05"));
  rcfg.frequency_hz = std::stod(Flag(argc, argv, "--ref-frequency", "0.10"));
  rcfg.gain_scale = std::stod(
      Flag(argc, argv, "--ref-gain-scale",
           Flag(argc, argv, "--gain-scale", reference_playback_selected ? "0.25" : "1.0")));
  rcfg.max_abs_err_rad = std::stod(Flag(argc, argv, "--ref-max-err", "0.30"));
  rcfg.stale_ms = std::stod(Flag(argc, argv, "--ref-stale-ms", "250"));
  rcfg.legs_passive = pcfg.legs_passive;
  auto ref = std::make_unique<a3_pingpong::PpReferencePlayback>(pp->isaac_to_sdk(), rcfg);
  ref->SetGroup(ParseRefGroup(Flag(argc, argv, "--ref-group", "0")));
  std::cout << "[pingpong] joint map OK; neck PASSIVE (q=0,kp=" << a3_pingpong::kHeadKp
            << ",kd=" << a3_pingpong::kHeadKd << "); start=" << ModeName(default_mode)
            << " level=" << level << " gain_scale=" << gain_scale.load()
            << " swing_speed=" << swing_speed << "\n";
  if (reference_playback_selected) {
    std::cout << "[ref] selected: starts PASSIVE; press a group key 0..7 then r to move. "
              << "amp=" << ref->config().amplitude_rad
              << "rad freq=" << ref->config().frequency_hz
              << "Hz gain_scale=" << ref->config().gain_scale
              << " max_err=" << ref->config().max_abs_err_rad
              << " stale_ms=" << ref->config().stale_ms
              << " no_publish=" << (no_publish ? "true" : "false") << "\n";
  }

  // --- optional per-tick CSV trace (every joint: des/q/qd/kp/kd) for offline diag ---
  const std::string trace_path = Flag(argc, argv, "--trace-csv", "");
  if (!no_publish && (!trace_path.empty() || !obs_csv_path.empty())) {
    std::cerr << "--trace-csv/--obs-csv perform synchronous file I/O in the policy callback and "
                 "are therefore restricted to --no-publish/--dry-run until an async logger is "
                 "available\n";
    return 2;
  }
  std::ofstream trace;
  if (!trace_path.empty()) {
    trace.open(trace_path);
    if (trace) {
      const auto& nm = a3_pingpong::backend_joint_order();
      trace << "tick,ts,mode,level,gain,swing,legs_passive,gravx,gravy,gravz";
      for (const auto& n : nm) trace << ",des_" << n;
      for (const auto& n : nm) trace << ",q_" << n;
      for (const auto& n : nm) trace << ",qd_" << n;
      for (const auto& n : nm) trace << ",kp_" << n;
      for (const auto& n : nm) trace << ",kd_" << n;
      trace << "\n";
      std::cout << "[pingpong] trace CSV -> " << trace_path << "\n";
    } else {
      std::cerr << "[pingpong] WARN: cannot open trace csv " << trace_path << "\n";
    }
  }
  std::ofstream* trace_ptr = trace.is_open() ? &trace : nullptr;

  // --- optional per-tick OBS CSV (the full 180-D obs) for A/B/C comparison ---
  std::ofstream obscsv;
  if (!obs_csv_path.empty()) {
    obscsv.open(obs_csv_path);
    if (obscsv) {
      obscsv << "tick,ts,mode,loc_mode,oracle_fresh,oracle_age_s,sync_miss,legs_passive";
      for (int i = 0; i < pp->onnx().obs_dim(); ++i) obscsv << ",obs_" << i;
      for (int i = 0; i < a3_pingpong::kNumJoints; ++i) obscsv << ",act_" << i;
      obscsv << "\n";
      std::cout << "[pingpong] obs CSV -> " << obs_csv_path
                << " (loc_mode=" << pp->loc_mode_name() << ")\n";
    } else {
      std::cerr << "[pingpong] WARN: cannot open obs csv " << obs_csv_path << "\n";
    }
  }
  std::ofstream* obscsv_ptr = obscsv.is_open() ? &obscsv : nullptr;
  const int loc_mode_int = static_cast<int>(loc_mode);

  // SHADOW free-running clock. The driver's `tick` is PUBLISH-GATED: in SHADOW
  // nothing is published, so tick FREEZES at 0 -> the scripted swing clock never
  // advances -> the policy sits on the windup frame and the action converges to a
  // single (clamped) windup command. That is NOT a representative swing preview and
  // looks like a stuck/saturated policy. Fix: in SHADOW drive the scripted clock
  // from this local free-running counter so the obs evolves through the full swing
  // exactly like MOTION (still no publish -> safe). MOTION keeps the driver tick
  // (which intentionally pauses the swing clock during a safe-halt). Opt out with
  // --shadow-frozen-clock to restore the old frozen behavior.
  std::uint64_t shadow_tick = 0;
  const bool shadow_free_clock = !Has(argc, argv, "--shadow-frozen-clock");

  // GROUND-CONTACT gains. On the GROUND the LEGS bear weight: at a uniform low
  // --gain-scale the leg kp (trained ~150-250) becomes far too soft (e.g. ×0.05 ~
  // 10) -> knees sag -> robot falls forward. So scale the LEGS (stance/balance,
  // slots 19..30) by their OWN --leg-gain-scale (default: follow --gain-scale, i.e.
  // unchanged behavior; set e.g. 0.5-1.0 so the legs can actually hold weight on the
  // ground) while --gain-scale keeps the arms/waist (swing) gentle. Neck keeps its
  // fixed PD (unscaled). On the HOIST you can leave leg-gain following gain-scale.
  std::atomic<double> leg_gain_scale{
      Has(argc, argv, "--leg-gain-scale") ? std::stod(Flag(argc, argv, "--leg-gain-scale", "1.0"))
                                          : -1.0};  // <0 => follow --gain-scale
  // ANKLE is the standing-balance joint (ankle strategy: ankle_pitch torque keeps the
  // CoM over the feet). Its TRAINED kp is among the low ones, so a modest --leg-gain-scale
  // still leaves the ankle too soft -> the robot pitches FORWARD about the ankle. Scale the
  // 4 ankle slots (L pitch/roll = 23,24; R pitch/roll = 29,30) by their OWN
  // --ankle-gain-scale so the ankles can be stiff (hold upright) while hips/knees stay
  // gentle. Default: follow --leg-gain-scale. May exceed 1.0 (stiffer than training) if the
  // real ankle needs more than sim to resist tipping.
  std::atomic<double> ankle_gain_scale{
      Has(argc, argv, "--ankle-gain-scale") ? std::stod(Flag(argc, argv, "--ankle-gain-scale", "1.0"))
                                            : -1.0};  // <0 => follow --leg-gain-scale
  // SQUAT/TILT SAFETY GUARD (auto-leg-hold full-body swing only): revert to level 0 (re-engage the
  // held official stand) if a released leg SINKS (knee bends past nominal by > --squat-guard-rad) or
  // the body TILTS (|gravX| or |gravY| > --tilt-guard). <=0 disables that check. A hoist-test backstop.
  // DEFAULT 1.4 (was 0.6): the trip point is nominal_knee(0.247)+guard, and the v2 clip's OWN
  // reference crouch starts at knee 0.62 with swing flexion commanding ~1.14 — a 0.6 guard sits
  // INSIDE the trained swing envelope, so it fired on a healthy swing at ~5 deg tilt and the
  // kp-2000 official-stand snap-back CATAPULTED the robot backward (the captured 2026-07-02 fall).
  // 1.4 (trip at ~1.65) still catches a genuine leg collapse; real tilt is the tilt guard's job.
  const double squat_guard_rad = std::stod(Flag(argc, argv, "--squat-guard-rad", "1.4"));
  const double tilt_guard = std::stod(Flag(argc, argv, "--tilt-guard", "0.35"));
  // ALWAYS-ON FALL GUARD (2026-07-04): every free-base test showed a FALLEN robot happily
  // "swinging" on the floor — the only fall-adjacent check (tilt guard above) is gated on
  // --auto-leg-hold and only drops to level 0 (still publishing a stiff stand). If the
  // pelvis tilts past ~60 deg (|gravZ| < 0.5) for fall_guard_ticks consecutive ticks in
  // any publishing mode, drop to PASSIVE (zero gains): stiff commands on a downed robot
  // thrash it against the floor and burn motors. Disable (hoist rigs that pitch the
  // pelvis, debugging) with --no-fall-guard.
  const bool fall_guard = !Has(argc, argv, "--no-fall-guard");
  const double fall_guard_gz = -0.5;   // body-frame gravity z above this = fallen
  int fall_guard_ticks = 0;
  // LEG WEIGHT-BEARING: the released (level-1) legs default to the POLICY leg PD x --leg-gain-scale,
  // whose kp (~150 knee) is ~13x softer than AGI's official ground-stand knee kp (2000) -> the knees
  // SINK under real body load. --leg-stand-gains keeps the official ground-stand PD on the legs even
  // when RELEASED (policy still drives the CLAMPED q_des), so they bear weight like the level-0 hold
  // while making small swing-coupled moves. REQUIRES --official-stand; pair with a TIGHT --leg-clamp-rad
  // (0.15-0.20) since stiff gains drive the leg firmly to whatever the clamp allows. Default off.
  const bool leg_stand_gains = Has(argc, argv, "--leg-stand-gains");
  // POSE-BLEND on MOTION entry: ramp the published q_des from the pose at the moment
  // MOTION engaged to the policy target over this many seconds, so (now-stiff) legs
  // do NOT snap through the ~1.5-2 rad stand->windup jump. Convex blend of two
  // in-range poses -> stays in range. 0 disables.
  const double motion_blend_sec = std::stod(Flag(argc, argv, "--motion-blend-sec", "0.5"));
  const double policy_dt = 1.0 / configured_policy_hz;
  Mode prev_mode_for_blend = Mode::kPassive;       // driver-thread only (no race)
  std::uint64_t prev_mode_generation = mode.snapshot().generation;
  std::uint64_t prev_yaw_alignment_generation = pp->yaw_alignment_generation();
  std::uint64_t stand_enter_tick = 0;              // PD_STAND entry blend (2026-07-04)
  Eigen::VectorXd stand_blend_q_start;
  std::uint64_t motion_enter_tick = 0;
  Eigen::VectorXd blend_q_start;
  int prev_level_for_blend = level;                // re-arm the pose-blend on a level toggle
  int prev_swing_dir_for_blend = pcfg.start_backhand ? -1 : 1;  // re-arm on f<->b switch
  // AUTO LEG-HOLD: hold legs+waist at level 0 (stable ready stand, no frozen-windup foot-lift),
  // release them at level 1 (full-body self-balancing swing). The pose-blend (re-armed on the
  // level toggle below) ramps q_des from the current measured pose so the stiff official stand
  // gains do NOT snap the legs on the 1->0 re-engage (the "jump").
  const bool auto_leg_hold = pcfg.auto_leg_hold;

  // Filled immediately after driver construction and before StartDriver(). Every subsequent
  // mode mutation is committed while holding the driver's backend-send mutex, making the mode
  // generation change and its zero-gain barrier one linearizable transaction with SendCommand.
  a3_deploy::A3PolicyDriver* driver_for_transitions = nullptr;
  auto transition_mode = [&mode, &driver_for_transitions](Mode next) -> bool {
    const auto mutate = [&mode, next]() noexcept { mode.store_under_driver_lock(next); };
    bool ok = true;
    if (driver_for_transitions) {
      ok = driver_for_transitions->CommitAuthorizationTransition(mutate, true);
    } else {
      mutate();  // no driver/send path exists during construction
    }
    if (!ok) g_stop.store(true, std::memory_order_release);
    return ok;
  };
  bool outer_hard_clamp_warned = false;  // driver-thread only

  // --- mode-aware CommandFn (reuses driver's RT loop + watchdog) ---
  a3_pingpong::PpPolicy* ppp = pp.get();
  a3_pingpong::PpReferencePlayback* refp = ref.get();
  auto command_fn = [ppp, refp, &mode, &gain_scale, &leg_gain_scale, &ankle_gain_scale,
                     stand_q, stand_kp, stand_kd,
                     official_stand, auto_leg_hold, squat_guard_rad, tilt_guard, leg_stand_gains,
                     trace_ptr, obscsv_ptr, loc_mode_int,
                     &shadow_tick, shadow_free_clock, motion_blend_sec, policy_dt,
                     &prev_mode_for_blend, &prev_mode_generation,
                     &prev_yaw_alignment_generation,
                     &motion_enter_tick, &blend_q_start,
                     &prev_level_for_blend, &prev_swing_dir_for_blend,
                     &stand_enter_tick, &stand_blend_q_start,
                     &transition_mode, &outer_hard_clamp_warned,
                     fall_guard, fall_guard_gz, &fall_guard_ticks,
                     no_publish, effort_limit_ratio, effort_limits_sdk](
                        std::uint64_t tick, const robot_io::RobotState& st,
                        robot_io::RobotCommand& cmd) -> bool {
    // Once SIGINT/SIGTERM/q is observed, no subsequent callback may enter a
    // motion path—even if the status thread has not completed shutdown yet.
    const auto mode_snapshot = mode.snapshot();
    const Mode m = StopRequested() ? Mode::kPassive : mode_snapshot.mode;
    const int N = 31;
    bool publish = true;
    // Refresh the IMU-derived gravity diagnostic FIRST (every tick, every mode), so the squat/tilt
    // guard and the [status]/trace gravZ see the real base orientation. ComputeCommand (which also
    // sets it) does not run in PASSIVE/PD_STAND, so without this the ground checks read a frozen
    // [0,0,-1].
    ppp->observe_imu(st);
    // ALWAYS-ON FALL GUARD (see flag above): fallen -> PASSIVE, independent of --auto-leg-hold.
    if (fall_guard && (m == Mode::kMotion || m == Mode::kPdStand)) {
      const auto gfg = ppp->last_proj_grav();
      if (gfg[2] > fall_guard_gz) {
        if (++fall_guard_ticks >= 25) {  // ~0.5 s persistent at 50 Hz
          transition_mode(Mode::kPassive);
          fall_guard_ticks = 0;
          std::fprintf(stderr,
              "[pp SAFETY] FALL GUARD: gravZ=%+.2f > %.2f for 0.5 s -> PASSIVE (zero gains). "
              "Stand the robot up, then 's' -> 'm' to re-engage.\n", gfg[2], fall_guard_gz);
          // Do not finish the old MOTION computation. The driver's authorization-token
          // interlock will discard this callback and publish its independent zero-gain halt.
          // Populate a safe command as defense in depth if this callback is reused elsewhere.
          cmd.q_des = st.q.size() == N ? st.q : Eigen::VectorXd::Zero(N);
          cmd.dq_des = Eigen::VectorXd::Zero(N);
          cmd.tau_ff = Eigen::VectorXd::Zero(N);
          cmd.kp = Eigen::VectorXd::Zero(N);
          cmd.kd = Eigen::VectorXd::Zero(N);
          return true;
        }
      } else {
        fall_guard_ticks = 0;
      }
    }
    // SQUAT/TILT SAFETY GUARD (auto-leg-hold, full-body swing only): if a released leg SINKS (knee
    // bends past nominal by > squat_guard_rad) or the body TILTS (|gravX|/|gravY| > tilt_guard),
    // revert to level 0 so the held official stand re-stiffens the legs. Backstop for hoist tests.
    if (auto_leg_hold && ppp->level() == 1) {
      const auto g = ppp->last_proj_grav();
      bool trip = false; const char* why = "";
      if (tilt_guard > 0.0 && (std::abs(g[0]) > tilt_guard || std::abs(g[1]) > tilt_guard)) {
        trip = true; why = "tilt";
      }
      if (!trip && squat_guard_rad > 0.0 && st.q.size() == N) {
        const auto& nomq = ppp->official_stand_q();  // slots 22=L knee, 28=R knee
        if (std::abs(st.q[22] - nomq[22]) > squat_guard_rad ||
            std::abs(st.q[28] - nomq[28]) > squat_guard_rad) { trip = true; why = "knee-sink"; }
      }
      if (trip) {
        ppp->set_level(0);  // re-engage held stand (the auto-hold below stiffens legs+waist this tick)
        std::fprintf(stderr, "[pp SAFETY] %s guard tripped -> reverted to level 0 (held official "
                             "stand); press 1 to retry once stable\n", why);
      }
    }
    // AUTO LEG-HOLD: flip the leg+waist hold from the live level (0=hold ready, 1=release swing)
    // BEFORE the policy/gain code runs this tick (a guard trip above already forced level 0).
    if (auto_leg_hold) {
      const bool hold = (ppp->level() == 0);
      ppp->set_legs_passive(hold);
      ppp->set_waist_passive(hold);
    }
    // Re-arm the pose-blend on a MOTION entry, a level toggle, OR a forehand<->backhand switch,
    // so q_des ramps from the CURRENT measured pose -> no snap when stiff official stand gains
    // (re)engage at the 1->0 toggle, NOR when the reference jumps to the other clip's windup on a
    // dir switch (the swing clock restarts at windup in PpPolicy; this blends the command to match).
    const int cur_level = ppp->level();
    const int cur_swing_dir = ppp->swing_dir();
    const bool level_just_changed = (cur_level != prev_level_for_blend);
    const bool dir_just_changed = (cur_swing_dir != prev_swing_dir_for_blend);
    prev_level_for_blend = cur_level;
    prev_swing_dir_for_blend = cur_swing_dir;
    // Generation catches ABA transitions between callbacks (MOTION->PD->MOTION can end with the
    // same value but belongs to a new clock/authorization session).
    const bool mode_generation_changed = mode_snapshot.generation != prev_mode_generation;
    const bool motion_just_entered =
        m == Mode::kMotion && (prev_mode_for_blend != Mode::kMotion || mode_generation_changed);
    const bool stand_just_entered =
        m == Mode::kPdStand && (prev_mode_for_blend != Mode::kPdStand || mode_generation_changed);
    // Re-capture the IMU yaw-align offsets whenever the POLICY (SHADOW/MOTION) engages
    // from a non-policy mode — the operator may have moved/turned the robot in between.
    const bool policy_session_changed =
        (m == Mode::kMotion || m == Mode::kShadow) &&
        (m != prev_mode_for_blend || mode_generation_changed);
    if (policy_session_changed) {
      shadow_tick = 0;
      ppp->rearm_yaw_align();
    }
    prev_mode_for_blend = m;
    prev_mode_generation = mode_snapshot.generation;

    // Yaw capture is a mode-independent publication barrier.  Update the mode-session bookkeeping
    // before returning so a deferred capture is not rearmed on every tick.  The capture tick is
    // zero gain; on the following tick the generation edge restarts the relevant pose blend from
    // measured q, so time spent waiting upright/still never consumes the blend window.
    const bool yaw_barrier = ppp->HandleYawAlignment(st, cmd);
    if (yaw_barrier) {
      publish = (m != Mode::kShadow);
    } else {
    const std::uint64_t yaw_generation = ppp->yaw_alignment_generation();
    const bool yaw_alignment_changed =
        yaw_generation != prev_yaw_alignment_generation;
    prev_yaw_alignment_generation = yaw_generation;
    const bool rearm_blend = motion_just_entered || level_just_changed || dir_just_changed ||
        (yaw_alignment_changed && m == Mode::kMotion);
    const bool rearm_stand_blend =
        stand_just_entered || (yaw_alignment_changed && m == Mode::kPdStand);
    if (rearm_blend) motion_enter_tick = tick;
    if (m == Mode::kPassive) {  // limp: hold current pose, zero gains
      cmd.q_des = st.q.size() == N ? st.q : Eigen::VectorXd::Zero(N);
      cmd.dq_des = Eigen::VectorXd::Zero(N);
      cmd.tau_ff = Eigen::VectorXd::Zero(N);
      cmd.kp = Eigen::VectorXd::Zero(N);
      cmd.kd = Eigen::VectorXd::Zero(N);
    } else if (m == Mode::kPdStand) {  // hold nominal stand pose (== a3_default_angles)
      cmd.q_des = official_stand ? ppp->official_stand_q() : stand_q;
      cmd.dq_des = Eigen::VectorXd::Zero(N);
      cmd.tau_ff = Eigen::VectorXd::Zero(N);
      if (official_stand) {  // production ground-stand gains (free-standing, Step 2)
        cmd.kp = ppp->official_stand_kp();
        cmd.kd = ppp->official_stand_kd();
      } else {               // gentle flat PD — clean on a HOIST (default)
        cmd.kp = Eigen::VectorXd::Constant(N, stand_kp);
        cmd.kd = Eigen::VectorXd::Constant(N, stand_kd);
      }
      // pose-blend on PD_STAND ENTRY (2026-07-04): 's' pressed mid-swing used to slam the
      // stiff (kp~2000 knee) static stand target onto a moving robot with NO ramp — the
      // same catapult class as the auto-leg-hold level drop, and the one transition the
      // MOTION-only blend below did not cover. Ramp q_des from the entry pose.
      if (motion_blend_sec > 1e-6 && st.q.size() == N) {
        if (rearm_stand_blend) { stand_blend_q_start = st.q; stand_enter_tick = tick; }
        if (stand_blend_q_start.size() == N) {
          const double elapsed = static_cast<double>(tick - stand_enter_tick) * policy_dt;
          const double a = std::min(1.0, std::max(0.0, elapsed / motion_blend_sec));
          if (a < 1.0) cmd.q_des = (1.0 - a) * stand_blend_q_start + a * cmd.q_des;
        }
      }
    } else if (m == Mode::kReferencePlayback) {
      if (!refp->ComputeCommand(tick, st, cmd)) return false;
    } else {  // SHADOW or MOTION: run the policy
      // In SHADOW the driver's publish-gated `tick` is frozen, so drive the swing
      // from a free-running counter for a representative no-publish preview (the obs
      // then evolves through the swing like MOTION). MOTION uses the driver tick.
      const bool shadow = (m == Mode::kShadow);
      const std::uint64_t clk = (shadow && shadow_free_clock) ? shadow_tick : tick;
      if (shadow && shadow_free_clock) ++shadow_tick;
      if (!ppp->ComputeCommand(clk, st, cmd)) return false;
      // per-group gain: legs (slots 19..30) by --leg-gain-scale so they can bear
      // weight on the ground; arms+waist by --gain-scale (gentle swing); neck keeps
      // its fixed PD (unscaled). leg_gain<0 => follow gain-scale (hoist / legacy).
      const double g_arm = gain_scale.load();
      const double g_leg_o = leg_gain_scale.load();
      const double g_leg = (g_leg_o >= 0.0) ? g_leg_o : g_arm;
      const double g_ank_o = ankle_gain_scale.load();
      const double g_ank = (g_ank_o >= 0.0) ? g_ank_o : g_leg;  // ankle: own gain, else follow leg
      // Per-tick (so --auto-leg-hold's level toggle takes effect): legs/waist that are
      // HELD with --official-stand get AGI's ground-stand gains; released joints get the
      // policy gains scaled below. ppp->legs_passive()/waist_passive() reflect the live hold.
      // --leg-stand-gains keeps the WEIGHT-BEARING official gains on the legs even when
      // RELEASED (policy still drives the clamped q_des) so the knees don't sink under load.
      const bool leg_official = official_stand && (ppp->legs_passive() || leg_stand_gains);
      const bool waist_held_off = official_stand && ppp->waist_passive();
      if (cmd.kp.size() == N && cmd.kd.size() == N) {
        for (int i = 0; i < N; ++i) {
          if (i == a3_pingpong::kHeadSlot0 || i == a3_pingpong::kHeadSlot1) continue;  // neck fixed PD
          const bool is_leg = (i >= a3_pingpong::kLegSlotStart &&
                               i < a3_pingpong::kLegSlotStart + a3_pingpong::kLegSlotCount);
          const bool is_ankle = (i == 23 || i == 24 || i == 29 || i == 30);  // L/R ankle pitch+roll
          const bool is_waist = (i >= a3_pingpong::kWaistSlotStart &&
                                 i < a3_pingpong::kWaistSlotStart + a3_pingpong::kWaistSlotCount);
          if ((is_leg && leg_official) || (is_waist && waist_held_off)) {
            // GROUND weight-bearing joint (held, or released under --leg-stand-gains):
            // overwrite with AGI's official ground-stand gains VERBATIM (the config proven
            // to stand free on the ground); ignore --gain/--leg/--ankle-gain-scale so a
            // stray scale can't soften the stance.
            cmd.kp[i] = ppp->official_stand_kp()[i];
            cmd.kd[i] = ppp->official_stand_kd()[i];
            continue;
          }
          const double s = is_ankle ? g_ank : (is_leg ? g_leg : g_arm);
          cmd.kp[i] *= s; cmd.kd[i] *= s;
        }
      }
      // pose-blend on MOTION entry OR level toggle: ramp q_des from the entry/toggle pose
      // to the (new) target over motion_blend_sec, so stiff legs don't snap through the
      // windup jump NOR through the 1->0 official-stand re-engage. (convex combo of two
      // in-range poses -> in range; no clamp needed.)
      if (m == Mode::kMotion && motion_blend_sec > 1e-6) {
        if (rearm_blend && st.q.size() == N) blend_q_start = st.q;
        if (blend_q_start.size() == N && cmd.q_des.size() == N) {
          const double elapsed = static_cast<double>(tick - motion_enter_tick) * policy_dt;
          const double a = std::min(1.0, std::max(0.0, elapsed / motion_blend_sec));
          if (a < 1.0) cmd.q_des = (1.0 - a) * blend_q_start + a * cmd.q_des;
        }
      }
      publish = (m == Mode::kMotion);  // SHADOW computes but does not publish
      // --- OBS CSV row (only when the policy ran, so obs is current) ---
      if (obscsv_ptr) {
        const auto d = ppp->take_obs_debug();
        if (d.valid) {
          auto& o = *obscsv_ptr;
          o << tick << ',' << ppp->last_time_step() << ',' << static_cast<int>(m) << ','
            << loc_mode_int << ',' << (d.oracle_fresh ? 1 : 0) << ',' << d.oracle_age_s
            << ',' << d.sync_miss << ',' << (ppp->legs_passive() ? 1 : 0);
          for (int i = 0; i < d.obs.size(); ++i) o << ',' << d.obs[i];
          const auto& a = ppp->last_action();
          for (int i = 0; i < a3_pingpong::kNumJoints; ++i)
            o << ',' << (a.size() == a3_pingpong::kNumJoints ? a[i] : 0.0);
          o << '\n';
          static int oc = 0; if (++oc % 25 == 0) o.flush();  // single RT writer
        }
      }
    }
    }
    // Outer hardware safety clamp in SDK order.  This is intentionally outside PpPolicy so it
    // protects PD_STAND, reference playback, blend interpolation and future command sources too;
    // schema-v3 soft limits inside target_q remain the strategy/training-distribution clamp.
    if (cmd.q_des.size() == N) {
      const int hard_clamped = a3_pingpong::clamp_q_to_limits(cmd.q_des);
      if (hard_clamped > 0 && !outer_hard_clamp_warned) {
        outer_hard_clamp_warned = true;
        std::fprintf(stderr,
            "[pp SAFETY] outer SDK hard-limit clamp modified %d joint command(s); "
            "inspect the active source/blend before continuing.\n",
            hard_clamped);
      }
    }

    // Command-envelope guard, applied after every gain/pose blend and immediately before the
    // driver can publish.  This is not a continuous thermal model; it is the stricter invariant we
    // can prove today: requested PD+feedforward effort and reported joint effort must stay within
    // the per-joint actuator limits embedded from the training model.  The verbal 24/60 Nm figures
    // are not reinterpreted here as continuous ratings.
    if (publish && !no_publish && m != Mode::kPassive) {
      const bool shapes_ok = st.q.size() == N && st.dq.size() == N &&
          cmd.q_des.size() == N && cmd.dq_des.size() == N && cmd.tau_ff.size() == N &&
          cmd.kp.size() == N && cmd.kd.size() == N && effort_limits_sdk.size() == N;
      if (!shapes_ok) {
        std::fprintf(stderr,
            "[pp SAFETY] command envelope unavailable/incomplete in a publish-capable mode -> halt\n");
        transition_mode(Mode::kPassive);
        return false;
      }
      if (!a3_pingpong::MeasuredEffortFeedbackValid(st.tau_est, N)) {
        std::fprintf(stderr,
            "[pp SAFETY] measured effort feedback is missing or non-finite in a "
            "publish-capable mode -> PASSIVE+halt\n");
        transition_mode(Mode::kPassive);
        return false;
      }
      for (int i = 0; i < N; ++i) {
        const double limit = effort_limit_ratio * effort_limits_sdk[i];
        const double requested = cmd.kp[i] * (cmd.q_des[i] - st.q[i]) +
            cmd.kd[i] * (cmd.dq_des[i] - st.dq[i]) + cmd.tau_ff[i];
        const double measured = st.tau_est[i];
        if (!std::isfinite(requested) || std::fabs(requested) > limit ||
            std::fabs(measured) > limit) {
          std::fprintf(stderr,
              "[pp SAFETY] effort envelope joint=%d requested=%+.2f measured=%+.2f "
              "limit=%.2f (ratio %.2f) -> PASSIVE+halt\n",
              i, requested, measured, limit, effort_limit_ratio);
          transition_mode(Mode::kPassive);
          return false;
        }
      }
    }
    // --- CSV trace row (all modes; final post-gain command + measured state) ---
    if (trace_ptr) {
      auto& o = *trace_ptr;
      const auto g = ppp->last_proj_grav();
      const bool has = (st.q.size() == N && st.dq.size() == N &&
                        cmd.q_des.size() == N && cmd.kp.size() == N && cmd.kd.size() == N);
      o << tick << ',' << ppp->last_time_step() << ',' << static_cast<int>(m) << ','
        << ppp->level() << ',' << gain_scale.load() << ',' << ppp->swing_speed() << ','
        << (ppp->legs_passive() ? 1 : 0) << ','
        << g[0] << ',' << g[1] << ',' << g[2];
      for (int i = 0; i < N; ++i) o << ',' << (has ? cmd.q_des[i] : 0.0);
      for (int i = 0; i < N; ++i) o << ',' << (has ? st.q[i] : 0.0);
      for (int i = 0; i < N; ++i) o << ',' << (has ? st.dq[i] : 0.0);
      for (int i = 0; i < N; ++i) o << ',' << (has ? cmd.kp[i] : 0.0);
      for (int i = 0; i < N; ++i) o << ',' << (has ? cmd.kd[i] : 0.0);
      o << '\n';
      static int fc = 0; if (++fc % 25 == 0) o.flush();  // single RT writer
    }
    return publish;
  };

  if (StopRequested()) return 130;
  if (!backend->Start()) { std::cerr << "backend Start failed\n"; return 5; }
  if (StopRequested()) {
    backend->Stop();
    return 130;
  }
  std::cout << "[pingpong] backend started\n";

  a3_deploy::A3PolicyDriverOptions dopt;
  dopt.policy_hz = configured_policy_hz;
  const double yaml_deadline_ms = cfg["policy_driver"]["command_deadline_ms"]
                                      ? cfg["policy_driver"]["command_deadline_ms"].as<double>()
                                      : 15.0;
  const double command_deadline_ms = std::stod(
      Flag(argc, argv, "--command-deadline-ms", std::to_string(yaml_deadline_ms)));
  const double policy_period_ms = 1000.0 / dopt.policy_hz;
  if (!std::isfinite(dopt.policy_hz) || dopt.policy_hz <= 0.0 ||
      !std::isfinite(command_deadline_ms) || command_deadline_ms <= 0.0 ||
      command_deadline_ms >= policy_period_ms) {
    std::cerr << "invalid policy_driver: policy_hz=" << dopt.policy_hz
              << " command_deadline_ms=" << command_deadline_ms
              << " (deadline must be >0 and < one policy period)\n";
    backend->Stop();
    return 2;
  }
  dopt.command_deadline_s = command_deadline_ms * 1.0e-3;
  dopt.command_publish_expected = [&mode, no_publish]() {
    return !no_publish && !StopRequested() &&
           mode.load(std::memory_order_acquire) != Mode::kShadow;
  };
  dopt.command_authorization_token = [&mode]() {
    return mode.authorization_token();
  };
  dopt.send_safe_halt_on_command_failure = !no_publish;
  dopt.send_final_safe_halt_on_stop = !no_publish;
  a3_deploy::CommandFn cfn = command_fn;  // disambiguate the PolicyFn/CommandFn ctor
  a3_deploy::A3PolicyDriver driver(*backend, cfn, dopt);
  driver_for_transitions = &driver;
  if (!driver.StartDriver()) {
    driver_for_transitions = nullptr;
    std::cerr << "StartDriver failed\n";
    backend->Stop();
    return 6;
  }
  std::cout << "[pingpong] driver started @ " << dopt.policy_hz
            << " Hz; command deadline=" << command_deadline_ms << " ms\n";

  // ---- consolidated bring-up CONFIG banner (one place to eyeball every knob) ----
  char leg_gain_banner[16];
  {
    const double lgs = leg_gain_scale.load();
    if (lgs >= 0.0) std::snprintf(leg_gain_banner, sizeof leg_gain_banner, "%.2f", lgs);
    else std::snprintf(leg_gain_banner, sizeof leg_gain_banner, "=gain");
  }
  char ankle_gain_banner[16];
  {
    const double ags = ankle_gain_scale.load();
    if (ags >= 0.0) std::snprintf(ankle_gain_banner, sizeof ankle_gain_banner, "%.2f", ags);
    else std::snprintf(ankle_gain_banner, sizeof ankle_gain_banner, "=leg");
  }
  std::printf(
      "[pingpong] ================= RUN CONFIG =================\n"
      "[pingpong]  start_mode   = %-9s  (s=PD_STAND hold/NO swing, m=MOTION publish)\n"
      "[pingpong]  level        = %-9d  (0=hold/windup, 1=SWING)\n"
      "[pingpong]  swing_dir    = %-9s  (f=forehand / b=backhand keys)\n"
      "[pingpong]  target_src   = %s\n"
      "[pingpong]  action_src   = ONNX policy (LEARNED 31-DOF action every tick; q_des = default_q + a*action_scale)\n"
      "[pingpong]  post_onnx    = neck[3,4] HELD q=0 kp40 kd2 | legs %-6s | q_des CLAMPED to A3 limits (nothing else overridden)\n"
      "[pingpong]  loc_mode     = %s\n"
      "[pingpong]  legs_passive = %-9s  (true=legs HELD; validates UPPER-BODY/waist swing only)\n"
      "[pingpong]  leg_hold     = %-9s  (official=AGI ground-stand gains [GROUND, proven] | trained=ONNX leg PD [HOIST])\n"
      "[pingpong]  waist_hold   = %-9s  (official/trained=waist FROZEN at nominal [ARMS-ONLY swing] | swing=policy-driven)\n"
      "[pingpong]  auto_hold    = %-9s  (--auto-leg-hold: level0 HOLDS legs+waist [ready stand], level1 RELEASES [full-body swing])\n"
      "[pingpong]  gain_scale   = %-9.2f  (arms/waist swing)   swing_speed = %.2f\n"
      "[pingpong]  leg_gain     = %-9s  ankle_gain = %-7s  (ankle=balance joint; raise if tipping fwd)  motion_blend = %.2fs\n"
      "[pingpong]  leg_stand_g  = %-9s  (--leg-stand-gains: RELEASED legs use official ground-stand PD [weight-bearing, kp~2000 knee] vs policy PD x leg_gain)\n"
      "[pingpong]  safety       = squat_guard=%.2frad tilt_guard=%.2f leg_clamp=%.2frad leg_smooth=%.2f  (L1: revert L0 on sink/tilt; clamp+EMA the released legs; off=0/1.0)\n"
      "[pingpong]  publish      = %-9s  (--dry-run/--no-publish => NEVER publishes; SHADOW also no-publish)\n"
      "[pingpong]  model        = %s\n"
      "[pingpong]  trace_csv    = %s\n"
      "[pingpong]  obs_csv      = %s\n"
      "[pingpong] =============================================\n",
      ModeName(default_mode), level, pp->swing_dir_name(),
      planner_mode
          ? "PLANNER  (live: racket <- /racket/command_flat, base <- /a3/base_pose_flat over ros2; engage machine drives swing)"
          : "SCRIPTED (fixed front-right TEST target; NO live planner -- f/b only flips y-sign+clip)",
      pcfg.legs_passive ? "HELD" : "policy", pp->loc_mode_name(),
      pcfg.legs_passive ? "true" : "false",
      pcfg.legs_passive ? (legs_official_gains ? "official" : "trained") : "n/a (policy)",
      pcfg.waist_passive ? (waist_official_gains ? "official" : "trained") : "swing",
      pcfg.auto_leg_hold ? "ON" : "off",
      gain_scale.load(), swing_speed,
      leg_gain_banner, ankle_gain_banner, motion_blend_sec,
      leg_stand_gains ? "ON" : "off",
      squat_guard_rad, tilt_guard, pcfg.leg_clamp_rad, pcfg.leg_smooth_alpha,
      no_publish ? "DISABLED" : "enabled",
      model_path.c_str(),
      trace_path.empty() ? "<none>" : trace_path.c_str(),
      obs_csv_path.empty() ? "<none>" : obs_csv_path.c_str());

  // --- keyboard control (raw, non-blocking) ---
  std::cout << "[keys] p=PASSIVE(limp)  s=PD_STAND(hold, NO swing)  h=SHADOW(compute, no publish)"
               "  m=MOTION(publish)\n";
  std::cout << "[keys] 0=level0(hold/windup)  1=level1(SWING)  f=forehand  b=backhand"
               "  [=gain-  ]=gain+  ,=swing slower  .=swing faster  q=quit\n";
  std::cout << "[ref keys] (only in --reference-playback) 0=head hold 1=waist 2=R shoulder"
               " 3=R elbow/wrist 4=R arm 5=waist+R arm 6=legs hold 7=upper body"
               "  r=start ref  x=hold ref  c=clear ref fault\n";
  termios old_tio{};
  bool tty = isatty(STDIN_FILENO);
  if (tty) {
    tcgetattr(STDIN_FILENO, &old_tio);
    termios raw = old_tio;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 1;  // 100ms poll
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
  }
  std::thread kb([&]() {
    while (!StopRequested()) {
      char c = 0;
      if (tty && read(STDIN_FILENO, &c, 1) == 1) {
        switch (c) {
          case 'p': refp->Hold("operator_passive"); transition_mode(Mode::kPassive);
                    std::cout << "-> PASSIVE\n"; break;
          case 's': refp->Hold("operator_pd_stand"); transition_mode(Mode::kPdStand);
                    std::cout << "-> PD_STAND\n"; break;
          case 'h': transition_mode(Mode::kShadow); std::cout << "-> SHADOW (no publish)\n"; break;
          case 'm': transition_mode(Mode::kMotion); std::cout << "-> MOTION (PUBLISHING)\n"; break;
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7':
            if (reference_playback_selected || mode.load() == Mode::kReferencePlayback) {
              const int gi = c - '0';
              refp->SetGroup(a3_pingpong::RefPlaybackGroupFromInt(gi));
              refp->Hold("group_selected_hold");
              transition_mode(Mode::kReferencePlayback);
              std::cout << "-> ref group " << gi << " ("
                        << a3_pingpong::RefPlaybackGroupName(refp->group())
                        << "), HOLD; press r to move\n";
            } else if (planner_mode) {
              std::cout << "-> level key ignored (PLANNER drives swing engage)\n";
            } else if (a3_pingpong::ScriptedPhaseHotkeyBlocked(
                           no_publish, mode.load() == Mode::kMotion, ppp->level())) {
              std::cout << "[pp SAFETY] level key ignored: publish-capable scripted MOTION "
                           "is mid-swing; use 'p' for an immediate safe abort, or wait for "
                           "level 0 before changing phase\n";
            } else if (c == '0') {
              ppp->set_level(0); std::cout << "-> level 0 (hold)\n";
            } else if (c == '1') {
              ppp->set_level(1); std::cout << "-> level 1 (forehand)\n";
            }
            break;
          case '[': gain_scale.store(std::max(0.0, gain_scale.load() - 0.1));
                    std::cout << "gain_scale=" << gain_scale.load() << "\n"; break;
          case ']': gain_scale.store(std::min(1.0, gain_scale.load() + 0.1));
                    std::cout << "gain_scale=" << gain_scale.load() << "\n"; break;
          case ',':
          case '.': // swing-speed rescales the in-flight clock RETROACTIVELY (t is scaled from
                    // the engage origin): mid-swing it snaps tts and breaks the wait-until-tts
                    // strike alignment. Planner mode owns the clock — ignore, like 0/1/f/b.
                    if (planner_mode) { std::cout << "-> swing-speed key ignored (PLANNER owns the clock)\n"; break; }
                    if (a3_pingpong::ScriptedPhaseHotkeyBlocked(
                            no_publish, mode.load() == Mode::kMotion, ppp->level())) {
                      std::cout << "[pp SAFETY] swing-speed key ignored: publish-capable "
                                   "scripted MOTION is mid-swing; tune only at level 0 "
                                   "(SHADOW/--no-publish remain available)\n";
                      break;
                    }
                    ppp->set_swing_speed(ppp->swing_speed() + (c == '.' ? 0.1 : -0.1));
                    std::cout << "swing_speed=" << ppp->swing_speed() << "\n"; break;
          case 'f': if (planner_mode) { std::cout << "-> f/b ignored (PLANNER picks the side)\n"; break; }
                    ppp->set_swing_dir(+1);
                    std::cout << "-> swing dir = FOREHAND (scripted target -y, clip0)\n"; break;
          case 'b': if (planner_mode) { std::cout << "-> f/b ignored (PLANNER picks the side)\n"; break; }
                    ppp->set_swing_dir(-1);
                    std::cout << "-> swing dir = BACKHAND (scripted target +y, clip1)\n"; break;
          case 'r': refp->Start(); transition_mode(Mode::kReferencePlayback);
                    std::cout << "-> REFERENCE_PLAYBACK moving group="
                              << a3_pingpong::RefPlaybackGroupName(refp->group())
                              << "\n"; break;
          case 'x': refp->Hold("operator_hold"); transition_mode(Mode::kReferencePlayback);
                    std::cout << "-> REFERENCE_PLAYBACK hold current pose\n"; break;
          case 'c': refp->ClearFault();
                    std::cout << "-> ref fault cleared; press r to move\n"; break;
          case 'q': g_stop.store(true); break;
          default: break;
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  });

  // --- status loop ---
  std::uint64_t last_ticks = 0;
  auto t_start = std::chrono::steady_clock::now();
  auto t_prev = t_start;
  bool clamp_rate_warned = false;  // one-shot high-clamp-rate warning (waist_roll audit)
  bool warming = warmup_sec > 0;
  if (warming)
    std::printf("[pingpong] warmup: PD_STAND for %.1fs, then -> %s\n", warmup_sec,
                ModeName(target_mode));
  bool driver_faulted = false;
  while (!StopRequested()) {
    for (int i = 0; i < 100 && !StopRequested(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (driver.CommandFaulted()) break;
    }
    if (driver.CommandFaulted()) {
      driver_faulted = true;
      g_stop.store(true, std::memory_order_release);
      std::fprintf(stderr,
                   "[pp SAFETY] command path fault latched: callback_fail=%llu "
                   "deadline=%llu send_fail=%llu -> shutdown\n",
                   (unsigned long long)driver.CommandFailureCount(),
                   (unsigned long long)driver.DeadlineViolationCount(),
                   (unsigned long long)driver.SendFailureCount());
      break;
    }
    if (StopRequested()) break;
    const std::uint64_t ticks = driver.PolicyTickCount();
    const std::uint64_t halts = driver.SafeHaltCount();
    auto now = std::chrono::steady_clock::now();
    if (warming &&
        std::chrono::duration<double>(now - t_start).count() >= warmup_sec) {
      transition_mode(target_mode);
      warming = false;
      std::printf("[pingpong] warmup done -> %s\n", ModeName(target_mode));
    }
    double dt = std::chrono::duration<double>(now - t_prev).count();
    double hz = dt > 0 ? (ticks - last_ticks) / dt : 0;
    const auto g = ppp->last_proj_grav();
    const Mode cur_mode = mode.load();
    if (cur_mode == Mode::kReferencePlayback) {
      std::printf("[status] mode=%s ref_group=%s ref_moving=%d ref_fault=%d rate=%.1fHz "
                  "ticks=%llu halts=%llu ref_amp=%.3f ref_freq=%.3f ref_gain=%.2f\n",
                  ModeName(cur_mode), a3_pingpong::RefPlaybackGroupName(refp->group()),
                  refp->moving() ? 1 : 0, refp->faulted() ? 1 : 0, hz,
                  (unsigned long long)ticks, (unsigned long long)halts,
                  refp->config().amplitude_rad, refp->config().frequency_hz,
                  refp->config().gain_scale);
      PrintRefDiagBlock(refp->TakeDiag());
    } else {
      // Consume the rolling diag window + obs-debug ONCE this tick (take_diag
      // resets the window) and reuse for the status/[fullbody] lines + blocks.
      const auto diag = ppp->take_diag();
      const auto obsd = ppp->take_obs_debug();
      const Eigen::VectorXd& act = ppp->last_action();
      const double act_max = act.size() ? act.cwiseAbs().maxCoeff() : 0.0;
      // Peak single-joint commanded/measured range within a backend-slot group
      // (waist 0..2, Lleg 19..24, Rleg 25..30) for the full-body verification.
      auto grp_amp = [&diag](int lo, int hi, double& cmdR, double& measR) {
        cmdR = measR = 0.0;
        if (!diag.valid) return;
        for (int i = lo; i <= hi; ++i) {
          cmdR = std::max(cmdR, diag.des_range[i]);
          measR = std::max(measR, diag.meas_range[i]);
        }
      };
      double waist_c, waist_m, lleg_c, lleg_m, rleg_c, rleg_m;
      grp_amp(0, 2, waist_c, waist_m);
      grp_amp(19, 24, lleg_c, lleg_m);
      grp_amp(25, 30, rleg_c, rleg_m);
      // sdir=swing direction, maxact=max|action| (near 0 => ONNX not driving),
      // clamp=#joints clamped THIS tick, legs_passive=leg cmds held nominal (NOT a
      // full-body test), sync_miss=cumulative dropped/unaligned packets (must be 0).
      // ts advancing + |act| oscillating => the swing clock is live.
      std::printf("[status] mode=%s level=%d sdir=%s gain=%.2f sspeed=%.2f rate=%.1fHz ticks=%llu "
                  "halts=%llu sync_miss=%llu ts=%d |act|=%.2f maxact=%.2f clamp=%d legs_passive=%d "
                  "gravZ=%.2f baseZ=%.3f grav=[%.2f %.2f %.2f]\n",
                  ModeName(cur_mode), ppp->level(), ppp->swing_dir_name(),
                  gain_scale.load(), ppp->swing_speed(), hz, (unsigned long long)ticks,
                  (unsigned long long)halts, (unsigned long long)obsd.sync_miss,
                  ppp->last_time_step(), act.norm(), act_max, ppp->last_clamp_count(),
                  ppp->legs_passive() ? 1 : 0, g[2], ppp->last_base_pos()[2], g[0], g[1], g[2]);
      // Full-body command-vs-measured peak amplitude per group. legs_passive=1 =>
      // Lleg/Rleg cmdR ~ 0 (held nominal) => this is NOT a full-body test. With
      // legs_passive=0, leg cmdR>0 proves the policy DRIVES the legs; small
      // measR/cmdR while HOISTED is EXPECTED (feet bear no load) -> WARN not FAIL.
      std::printf("[fullbody] legs_passive=%d | waist cmdR=%.3f measR=%.3f | "
                  "Lleg cmdR=%.3f measR=%.3f | Rleg cmdR=%.3f measR=%.3f  (rad, peak this window)\n",
                  ppp->legs_passive() ? 1 : 0, waist_c, waist_m, lleg_c, lleg_m, rleg_c, rleg_m);
      PrintDiagBlock(diag, ppp->legs_passive());  // per-joint cmd-vs-meas block (SHADOW/MOTION)
      PrintObsDebugBlock(obsd, ppp->last_action(),
                         ppp->planner_mode() ? ppp->planner_status() : std::string{});  // obs slices + stats
      // one-shot warning if any joint is hitting its clamp on a large fraction of
      // ticks (the documented waist_roll mismatch): the policy keeps commanding
      // beyond the A3 limit. NOT a fault (clamp keeps it safe) — a tuning flag.
      const int wj = ppp->worst_clamped_slot();
      if (!clamp_rate_warned && wj >= 0 && ppp->clamp_ticks() > 100) {
        const double frac = static_cast<double>(ppp->clamp_count_for(wj)) /
                            static_cast<double>(ppp->clamp_ticks());
        if (frac > 0.20) {
          clamp_rate_warned = true;
          std::printf("[pingpong] WARN clamp-rate: %s clamped on %.0f%% of ticks "
                      "(max viol %.3f rad) -> policy commands beyond its A3 limit "
                      "(safe: clamped). See the waist_roll audit in the runbook.\n",
                      a3_pingpong::backend_joint_order()[wj].c_str(), 100.0 * frac,
                      ppp->clamp_max_viol_for(wj));
        }
      }
    }
    last_ticks = ticks; t_prev = now;
  }

  std::cout << "[pingpong] stopping...\n";
  transition_mode(Mode::kPassive);
  refp->Hold("process_shutdown");
  if (kb.joinable()) kb.join();
  if (tty) tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
  const bool final_halt_ok = driver.StopDriver();
  driver_for_transitions = nullptr;
  backend->Stop();
  if (!final_halt_ok) {
    std::cerr << "[pp SAFETY] final safe-halt exhausted retries; local delivery is unconfirmed. "
                 "Keep the external E-stop engaged and inspect the backend/controller timeout.\n";
  }
  std::cout << "[pingpong] done\n";
  return driver_faulted ? 7 : (final_halt_ok ? 0 : 8);
}
