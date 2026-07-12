// Inexact joined-source diagnostic for the first observed planner-engaged
// 179-D actor candidate. This schema is structurally forbidden as Gate3 evidence.
//
// This header deliberately has no Eigen, ROS, AimRT, ONNX Runtime, or MuJoCo
// dependency.  The production runner converts its native objects into the
// plain arrays below; dependency-light tests exercise the validation, stable
// state-source read, and atomic no-replace writer directly.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "a3_pingpong/pp_sha256.hpp"

namespace a3_pingpong {
namespace first_tick {

namespace fs = std::filesystem;

constexpr std::uint64_t kNativeStateMagic = 0x3153544633475050ULL;  // "PPG3FTS1" LE.
constexpr std::uint32_t kNativeStateVersion = 1;
constexpr const char* kDefaultNativeStatePath =
    "/dev/shm/pp_gate3_first_tick_state_v1";
constexpr std::int64_t kDefaultMaxSourceAgeNs = 100000000;   // 100 ms.
constexpr std::int64_t kDefaultMaxStampSkewNs = 20000000;    // 20 ms.
constexpr std::int64_t kDefaultMaxCrossSourceJoinSkewNs = 30000000;  // 30 ms.

inline bool PlannerActorCandidateEligible(int obs_dim, bool planner_mode,
                                          bool planner_engaged,
                                          bool planner_have_hold, int level,
                                          bool face_command_valid,
                                          double swing_type, double rho) {
  return obs_dim == 179 && planner_mode && planner_engaged && planner_have_hold &&
         level == 1 && face_command_valid &&
         (swing_type == 1.0 || swing_type == -1.0) && rho == 0.0;
}

// Written by scripts/gate3_first_tick_state_bridge.py from three existing,
// simulator-native ROS topics.  No member is inferred by the runner.  In
// particular base_linear_velocity_world is the vendor MuJoCo framelinvel
// sensor, never an integrated IMU value or a zero placeholder.
struct alignas(8) NativeStateWire {
  std::uint64_t magic = kNativeStateMagic;
  std::uint32_t version = kNativeStateVersion;
  std::uint32_t byte_size = sizeof(NativeStateWire);
  // The Python writer publishes only positive even committed generations.
  // Kernel flock, rather than an odd/even mmap seqlock, is the transaction.
  std::uint64_t sequence = 0;
  std::int64_t base_pose_stamp_ns = 0;
  std::int64_t base_twist_stamp_ns = 0;
  std::int64_t racket_pose_stamp_ns = 0;
  std::int64_t base_pose_receive_monotonic_ns = 0;
  std::int64_t base_twist_receive_monotonic_ns = 0;
  std::int64_t racket_pose_receive_monotonic_ns = 0;
  // A3SyncLoop's RobotState ready stamps use system_clock, so the bridge also
  // records system-clock receipt stamps for the cross-source join. Monotonic
  // receipts above are used only for freshness (never compared across epochs).
  std::int64_t base_pose_receive_system_ns = 0;
  std::int64_t base_twist_receive_system_ns = 0;
  std::int64_t racket_pose_receive_system_ns = 0;
  std::array<double, 3> base_position_world{};
  std::array<double, 4> base_quaternion_wxyz{{1.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> base_linear_velocity_world{};
  std::array<double, 3> base_angular_velocity_world{};
  std::array<double, 3> racket_position_world{};
  std::array<double, 4> racket_quaternion_wxyz{{1.0, 0.0, 0.0, 0.0}};
};
static_assert(sizeof(NativeStateWire) == 256,
              "Gate3 first-tick shared-state ABI changed");
static_assert(offsetof(NativeStateWire, sequence) == 16,
              "Gate3 first-tick seqlock offset changed");

inline std::int64_t MonotonicNowNs() {
  struct timespec ts {};
  if (::clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1;
  return static_cast<std::int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

inline bool IsFinite(const double value) { return std::isfinite(value); }

template <std::size_t N>
inline bool AllFinite(const std::array<double, N>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

inline bool AllFinite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

inline double QuaternionNorm(const std::array<double, 4>& q) {
  return std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

inline std::array<double, 3> ProjectedGravityBody(
    const std::array<double, 4>& q) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  return {{-2.0 * (x * z - w * y),
           -2.0 * (y * z + w * x),
           -(1.0 - 2.0 * (x * x + y * y))}};
}

inline bool IsLowerHex(std::string_view value, std::size_t length) {
  if (value.size() != length) return false;
  return std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

inline std::pair<std::int64_t, std::int64_t> StatMtime(const struct stat& st) {
#if defined(__APPLE__)
  return {static_cast<std::int64_t>(st.st_mtimespec.tv_sec),
          static_cast<std::int64_t>(st.st_mtimespec.tv_nsec)};
#else
  return {static_cast<std::int64_t>(st.st_mtim.tv_sec),
          static_cast<std::int64_t>(st.st_mtim.tv_nsec)};
#endif
}

inline bool IsCanonicalAbsolutePath(const fs::path& input, bool must_exist,
                                    std::string& error) {
  if (!input.is_absolute()) {
    error = "path must be absolute";
    return false;
  }
  if (input != input.lexically_normal()) {
    error = "path must use its lexical canonical spelling";
    return false;
  }
  const fs::path existing = must_exist ? input : input.parent_path();
  std::error_code ec;
  const fs::path canonical_existing = fs::canonical(existing, ec);
  if (ec) {
    error = "cannot canonicalize " + existing.string() + ": " + ec.message();
    return false;
  }
  const fs::path expected = must_exist ? canonical_existing
                                       : canonical_existing / input.filename();
  if (expected != input) {
    error = "path is not canonical (expected " + expected.string() + ")";
    return false;
  }

  // canonical() follows symlinks, so explicitly reject a symlink at every
  // spelling component as well as a symlink leaf.
  fs::path cursor = input.root_path();
  for (const auto& component : input.relative_path()) {
    cursor /= component;
    if (!must_exist && cursor == input) break;
    const auto status = fs::symlink_status(cursor, ec);
    if (ec) {
      error = "cannot lstat " + cursor.string() + ": " + ec.message();
      return false;
    }
    if (fs::is_symlink(status)) {
      error = "symlink path component is forbidden: " + cursor.string();
      return false;
    }
  }
  return true;
}

class NativeStateSource {
 public:
  NativeStateSource() = default;
  NativeStateSource(const NativeStateSource&) = delete;
  NativeStateSource& operator=(const NativeStateSource&) = delete;
  ~NativeStateSource() { Close(); }

  bool Open(const std::string& path, std::string& error) {
    Close();
    fs::path canonical(path);
    if (!IsCanonicalAbsolutePath(canonical, true, error)) return false;
    struct stat lst {};
    if (::lstat(path.c_str(), &lst) != 0) {
      error = "lstat state source failed: " + std::string(std::strerror(errno));
      return false;
    }
    if (S_ISLNK(lst.st_mode) || !S_ISREG(lst.st_mode)) {
      error = "state source must be a non-symlink regular file";
      return false;
    }
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd_ < 0) {
      error = "open state source failed: " + std::string(std::strerror(errno));
      return false;
    }
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_dev != lst.st_dev || st.st_ino != lst.st_ino ||
        st.st_size != static_cast<off_t>(sizeof(NativeStateWire))) {
      error = "state source identity/type/size changed or is invalid";
      Close();
      return false;
    }
    if (st.st_uid != ::geteuid() || (st.st_mode & 0777) != 0600) {
      error = "state source must be owned by this uid and mode 0600";
      Close();
      return false;
    }
    path_ = path;
    return true;
  }

  bool Latest(NativeStateWire& out, std::string& error,
              std::int64_t max_age_ns = kDefaultMaxSourceAgeNs,
              std::int64_t max_stamp_skew_ns = kDefaultMaxStampSkewNs) const {
    if (fd_ < 0) {
      error = "state source is not open";
      return false;
    }
    if (max_age_ns <= 0 || max_stamp_skew_ns < 0) {
      error = "state source age/skew limits are invalid";
      return false;
    }
    // Cross-process synchronization is a kernel-mediated flock transaction,
    // not a language-level atomic/fence assumption over mmap. The Python
    // writer holds LOCK_EX around one pwrite; this reader holds LOCK_SH around
    // one complete pread and the identity checks.
    while (::flock(fd_, LOCK_SH) != 0) {
      if (errno == EINTR) continue;
      error = "cannot lock state source: " + std::string(std::strerror(errno));
      return false;
    }
    struct stat before {};
    struct stat after {};
    const bool before_ok = ::fstat(fd_, &before) == 0;
    std::size_t offset = 0;
    bool read_ok = before_ok;
    while (read_ok && offset < sizeof(out)) {
      const ssize_t n = ::pread(fd_, reinterpret_cast<char*>(&out) + offset,
                                sizeof(out) - offset, static_cast<off_t>(offset));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) { read_ok = false; break; }
      offset += static_cast<std::size_t>(n);
    }
    const bool after_ok = ::fstat(fd_, &after) == 0;
    const int unlock_rc = ::flock(fd_, LOCK_UN);
    if (!read_ok || !after_ok || unlock_rc != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mode != after.st_mode || StatMtime(before) != StatMtime(after)) {
      error = "state source locked read/identity validation failed";
      return false;
    }
    if (out.magic != kNativeStateMagic || out.version != kNativeStateVersion ||
        out.byte_size != sizeof(NativeStateWire) || out.sequence == 0 ||
        (out.sequence & 1ULL) != 0ULL) {
      error = "state source magic/version/size/sequence is invalid";
      return false;
    }
    const std::array<std::int64_t, 3> stamps{{
        out.base_pose_stamp_ns, out.base_twist_stamp_ns,
        out.racket_pose_stamp_ns}};
    const std::array<std::int64_t, 3> receives_monotonic{{
        out.base_pose_receive_monotonic_ns,
        out.base_twist_receive_monotonic_ns,
        out.racket_pose_receive_monotonic_ns}};
    const std::array<std::int64_t, 3> receives_system{{
        out.base_pose_receive_system_ns,
        out.base_twist_receive_system_ns,
        out.racket_pose_receive_system_ns}};
    if (std::any_of(stamps.begin(), stamps.end(), [](std::int64_t v) { return v <= 0; }) ||
        std::any_of(receives_monotonic.begin(), receives_monotonic.end(),
                    [](std::int64_t v) { return v <= 0; }) ||
        std::any_of(receives_system.begin(), receives_system.end(),
                    [](std::int64_t v) { return v <= 0; })) {
      error = "state source is incomplete (all three native topics are required)";
      return false;
    }
    const auto stamp_mm = std::minmax_element(stamps.begin(), stamps.end());
    if (*stamp_mm.second - *stamp_mm.first > max_stamp_skew_ns) {
      error = "native topic header stamps exceed the allowed skew";
      return false;
    }
    const std::int64_t now = MonotonicNowNs();
    if (now <= 0) {
      error = "cannot read CLOCK_MONOTONIC";
      return false;
    }
    for (std::int64_t received : receives_monotonic) {
      const std::int64_t age = now - received;
      if (age < 0 || age > max_age_ns) {
        error = "native topic snapshot is stale or has a future receive time";
        return false;
      }
    }
    if (!AllFinite(out.base_position_world) ||
        !AllFinite(out.base_quaternion_wxyz) ||
        !AllFinite(out.base_linear_velocity_world) ||
        !AllFinite(out.base_angular_velocity_world) ||
        !AllFinite(out.racket_position_world) ||
        !AllFinite(out.racket_quaternion_wxyz)) {
      error = "native topic snapshot contains NaN/Inf";
      return false;
    }
    if (std::fabs(QuaternionNorm(out.base_quaternion_wxyz) - 1.0) > 1e-6 ||
        std::fabs(QuaternionNorm(out.racket_quaternion_wxyz) - 1.0) > 1e-6) {
      error = "native base/racket quaternion is not unit length";
      return false;
    }
    return true;
  }

  const std::string& path() const { return path_; }

 private:
  void Close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    path_.clear();
  }

  int fd_ = -1;
  std::string path_;
};

struct TargetEvidence {
  std::array<double, 3> position_world{};
  std::array<double, 3> velocity_world{};
  std::array<double, 3> normal_raw_mount_a_world{};
  double rho = 0.0;
  double time_to_strike = 0.0;
  double swing_type = 0.0;
  bool valid = false;
};

struct Evidence {
  std::string model_path;
  std::string model_sha256;
  std::string training_contract_sha256;
  std::string source_checkpoint_sha256;
  std::string native_state_path;
  std::uint64_t policy_tick = 0;
  std::int64_t robot_state_timestamp_ns = 0;
  std::int64_t robot_state_tick = 0;
  std::int64_t robot_state_data_ready_ns = 0;
  std::int64_t robot_state_sync_ready_ns = 0;
  bool robot_state_sync_complete = false;
  bool robot_state_sync_aligned = false;
  std::int64_t robot_state_sync_skew_ns = 0;
  double policy_base_source_age_s = -1.0;
  int reference_time_step = -1;
  std::vector<std::string> joint_names;
  std::vector<double> qpos;
  std::vector<double> qvel;
  std::vector<double> base_pose;
  // The 179 observation uses the yaw-aligned IMU/external-localizer policy
  // pose, while qpos/base_pose is the native vendor-world state. Recording
  // both (and validating position + tilt consistency) makes that relation
  // explicit instead of pretending obs was built from qpos[0:7] verbatim.
  std::vector<double> policy_base_pose;
  std::vector<double> racket_pose;
  std::vector<double> racket_fk_position_world;
  double racket_fk_position_error_m = 0.0;
  std::vector<double> obs;
  std::vector<double> action;
  TargetEvidence target;
  NativeStateWire native_state;
};

inline std::string JsonEscape(std::string_view input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (unsigned char c : input) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          static constexpr char hex[] = "0123456789abcdef";
          out += "\\u00";
          out.push_back(hex[(c >> 4) & 0xf]);
          out.push_back(hex[c & 0xf]);
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  return out;
}

inline std::string Number(double value) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return out.str();
}

inline std::string JsonArray(const std::vector<double>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ',';
    out += Number(values[i]);
  }
  out += ']';
  return out;
}

template <std::size_t N>
inline std::string JsonArray(const std::array<double, N>& values) {
  return JsonArray(std::vector<double>(values.begin(), values.end()));
}

inline std::string JsonStringArray(const std::vector<std::string>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ',';
    out += "\"" + JsonEscape(values[i]) + "\"";
  }
  out += ']';
  return out;
}

inline const std::string& NativeSourceContractText() {
  static const std::string value =
      "schema=pp_gate3_first_tick_native_state_v1\n"
      "base_pose=/sim/a3/pelvis_pose|geometry_msgs/PoseStamped|odom|native_framepos_framequat\n"
      "base_twist=/sim/a3/pelvis_twist|geometry_msgs/TwistStamped|odom|native_framelinvel_frameangvel\n"
      "racket_pose=/sim/a3/right_racket_pose|geometry_msgs/PoseStamped|odom|native_site_framepos_framequat\n"
      "racket_site=right_racket|wrist_local_pos_m=0.21021,0.032078,0.032036|"
      "equals_formal_racket_control_point_offset_wrist_m\n"
      "quaternion_order=wxyz\nvelocity_frame=world_odom\n"
      "sample_join=closest_receipt_window_not_common_sim_sequence\n"
      "planner_snapshot_exact=false\n"
      "no_base_velocity_estimation=true\n";
  return value;
}

inline bool ValidateEvidence(const Evidence& e, std::string& error) {
  if (!IsLowerHex(e.model_sha256, 64) ||
      !IsLowerHex(e.training_contract_sha256, 64) ||
      !IsLowerHex(e.source_checkpoint_sha256, 64)) {
    error = "model/training/checkpoint SHA fields must be lowercase SHA-256";
    return false;
  }
  if (e.joint_names.size() != 31 || e.qpos.size() != 38 || e.qvel.size() != 37 ||
      e.base_pose.size() != 7 || e.policy_base_pose.size() != 7 ||
      e.racket_pose.size() != 7 || e.obs.size() != 179 ||
      e.action.size() != 31 || e.racket_fk_position_world.size() != 3) {
    error = "first-tick vector lengths must be qpos38/qvel37/base7/racket7/obs179/action31";
    return false;
  }
  std::vector<std::string> sorted_names = e.joint_names;
  std::sort(sorted_names.begin(), sorted_names.end());
  if (std::adjacent_find(sorted_names.begin(), sorted_names.end()) != sorted_names.end() ||
      std::any_of(sorted_names.begin(), sorted_names.end(),
                  [](const std::string& name) { return name.empty(); })) {
    error = "joint_names must contain 31 distinct non-empty names";
    return false;
  }
  if (!AllFinite(e.qpos) || !AllFinite(e.qvel) || !AllFinite(e.base_pose) ||
      !AllFinite(e.racket_pose) || !AllFinite(e.obs) || !AllFinite(e.action) ||
      !AllFinite(e.policy_base_pose) ||
      !AllFinite(e.racket_fk_position_world) ||
      !IsFinite(e.racket_fk_position_error_m) ||
      !AllFinite(e.target.position_world) || !AllFinite(e.target.velocity_world) ||
      !AllFinite(e.target.normal_raw_mount_a_world) || !IsFinite(e.target.rho) ||
      !IsFinite(e.target.time_to_strike) || !IsFinite(e.target.swing_type)) {
    error = "first-tick diagnostic contains NaN/Inf";
    return false;
  }
  const std::array<double, 4> policy_quat{{
      e.policy_base_pose[3], e.policy_base_pose[4], e.policy_base_pose[5],
      e.policy_base_pose[6]}};
  if (std::fabs(QuaternionNorm(policy_quat) - 1.0) > 1e-6) {
    error = "policy observation base quaternion is not unit length";
    return false;
  }
  double policy_native_position_error_sq = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    const double delta = e.policy_base_pose[i] - e.base_pose[i];
    policy_native_position_error_sq += delta * delta;
  }
  if (std::sqrt(policy_native_position_error_sq) > 0.03) {
    error = "policy-localized base and native base differ by more than 3cm";
    return false;
  }
  const std::array<double, 4> native_quat{{
      e.base_pose[3], e.base_pose[4], e.base_pose[5], e.base_pose[6]}};
  const auto policy_gravity = ProjectedGravityBody(policy_quat);
  const auto native_gravity = ProjectedGravityBody(native_quat);
  double gravity_error_sq = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    const double delta = policy_gravity[i] - native_gravity[i];
    gravity_error_sq += delta * delta;
  }
  if (std::sqrt(gravity_error_sq) > 0.02) {
    error = "policy yaw-aligned IMU tilt and native base tilt are inconsistent";
    return false;
  }
  if (e.racket_fk_position_error_m < 0.0 || e.racket_fk_position_error_m > 0.005) {
    error = "native right_racket site differs from formal control-point FK by more than 5mm";
    return false;
  }
  double racket_error_sq = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    const double delta = e.racket_fk_position_world[i] - e.racket_pose[i];
    racket_error_sq += delta * delta;
  }
  if (std::fabs(std::sqrt(racket_error_sq) - e.racket_fk_position_error_m) > 1e-12) {
    error = "racket FK/site consistency error is not self-consistent";
    return false;
  }
  if (!e.target.valid || (e.target.swing_type != 1.0 && e.target.swing_type != -1.0) ||
      e.target.rho != 0.0 ||
      std::fabs(std::sqrt(
          e.target.normal_raw_mount_a_world[0] * e.target.normal_raw_mount_a_world[0] +
          e.target.normal_raw_mount_a_world[1] * e.target.normal_raw_mount_a_world[1] +
          e.target.normal_raw_mount_a_world[2] * e.target.normal_raw_mount_a_world[2]) - 1.0) >
          1e-6) {
    error = "strict 179 target candidate must be valid, side +/-1, rho zero, and carry a unit raw-A normal";
    return false;
  }
  for (std::size_t i = 0; i < 7; ++i) {
    if (e.qpos[i] != e.base_pose[i]) {
      error = "base_pose must be byte-value identical to qpos[0:7]";
      return false;
    }
  }
  const std::array<double, 7> native_base{{
      e.native_state.base_position_world[0], e.native_state.base_position_world[1],
      e.native_state.base_position_world[2], e.native_state.base_quaternion_wxyz[0],
      e.native_state.base_quaternion_wxyz[1], e.native_state.base_quaternion_wxyz[2],
      e.native_state.base_quaternion_wxyz[3]}};
  const std::array<double, 7> native_racket{{
      e.native_state.racket_position_world[0], e.native_state.racket_position_world[1],
      e.native_state.racket_position_world[2], e.native_state.racket_quaternion_wxyz[0],
      e.native_state.racket_quaternion_wxyz[1], e.native_state.racket_quaternion_wxyz[2],
      e.native_state.racket_quaternion_wxyz[3]}};
  for (std::size_t i = 0; i < 7; ++i) {
    if (e.base_pose[i] != native_base[i] || e.racket_pose[i] != native_racket[i]) {
      error = "base/racket pose vectors must be the unmodified native source values";
      return false;
    }
  }
  for (std::size_t i = 0; i < 3; ++i) {
    if (e.qvel[i] != e.native_state.base_linear_velocity_world[i] ||
        e.qvel[3 + i] != e.native_state.base_angular_velocity_world[i]) {
      error = "qvel root twist must be the unmodified native source value";
      return false;
    }
  }
  if (e.native_state.magic != kNativeStateMagic ||
      e.native_state.version != kNativeStateVersion ||
      e.native_state.byte_size != sizeof(NativeStateWire) ||
      e.native_state.sequence == 0 || (e.native_state.sequence & 1ULL) != 0ULL) {
    error = "joined native-source record is not a stable v1 generation";
    return false;
  }
  const auto& ns = e.native_state;
  if (ns.base_pose_stamp_ns <= 0 || ns.base_twist_stamp_ns <= 0 ||
      ns.racket_pose_stamp_ns <= 0 || ns.base_pose_receive_monotonic_ns <= 0 ||
      ns.base_twist_receive_monotonic_ns <= 0 ||
      ns.racket_pose_receive_monotonic_ns <= 0 ||
      ns.base_pose_receive_system_ns <= 0 ||
      ns.base_twist_receive_system_ns <= 0 ||
      ns.racket_pose_receive_system_ns <= 0 ||
      !AllFinite(ns.base_position_world) ||
      !AllFinite(ns.base_quaternion_wxyz) ||
      !AllFinite(ns.base_linear_velocity_world) ||
      !AllFinite(ns.base_angular_velocity_world) ||
      !AllFinite(ns.racket_position_world) ||
      !AllFinite(ns.racket_quaternion_wxyz) ||
      std::fabs(QuaternionNorm(ns.base_quaternion_wxyz) - 1.0) > 1e-6 ||
      std::fabs(QuaternionNorm(ns.racket_quaternion_wxyz) - 1.0) > 1e-6) {
    error = "native state provenance is incomplete/non-finite/non-unit";
    return false;
  }
  const std::array<std::int64_t, 3> native_header_stamps{{
      ns.base_pose_stamp_ns, ns.base_twist_stamp_ns, ns.racket_pose_stamp_ns}};
  const auto native_stamp_mm =
      std::minmax_element(native_header_stamps.begin(), native_header_stamps.end());
  if (*native_stamp_mm.second - *native_stamp_mm.first > kDefaultMaxStampSkewNs) {
    error = "native topic header stamps exceed the diagnostic 20ms skew";
    return false;
  }
  if (e.reference_time_step < 0 || e.robot_state_timestamp_ns <= 0 ||
      e.robot_state_tick < 0 || e.robot_state_data_ready_ns <= 0 ||
      e.robot_state_sync_ready_ns <= 0 || !e.robot_state_sync_complete ||
      !e.robot_state_sync_aligned || e.robot_state_sync_skew_ns < 0 ||
      !std::isfinite(e.policy_base_source_age_s) ||
      e.policy_base_source_age_s < 0.0 || e.policy_base_source_age_s > 0.1 ||
      e.native_state_path.empty()) {
    error = "tick/state/native-source provenance is incomplete";
    return false;
  }
  const std::array<std::int64_t, 5> join_times{{
      e.robot_state_data_ready_ns, e.robot_state_sync_ready_ns,
      e.native_state.base_pose_receive_system_ns,
      e.native_state.base_twist_receive_system_ns,
      e.native_state.racket_pose_receive_system_ns}};
  const auto join_mm = std::minmax_element(join_times.begin(), join_times.end());
  if (*join_mm.second - *join_mm.first > kDefaultMaxCrossSourceJoinSkewNs) {
    error = "RobotState and native sidecar receipts exceed the 30ms join window";
    return false;
  }
  return true;
}

inline std::string TargetPayload(const TargetEvidence& target) {
  return std::string("{") +
      "\"frame\":\"world_table\"," +
      "\"normal_semantics\":\"raw_mount_plus_y_a_actor_command\"," +
      "\"position\":" + JsonArray(target.position_world) + ',' +
      "\"rho\":" + Number(target.rho) + ',' +
      "\"swing_type\":" + Number(target.swing_type) + ',' +
      "\"time_to_strike\":" + Number(target.time_to_strike) + ',' +
      "\"valid\":" + (target.valid ? "true" : "false") + ',' +
      "\"velocity\":" + JsonArray(target.velocity_world) + ',' +
      "\"normal\":" + JsonArray(target.normal_raw_mount_a_world) + "}";
}

inline std::string BuildPayload(const Evidence& e) {
  const std::string qpos_json = JsonArray(e.qpos);
  const std::string qvel_json = JsonArray(e.qvel);
  const std::string base_json = JsonArray(e.base_pose);
  const std::string policy_base_json = JsonArray(e.policy_base_pose);
  const std::string racket_json = JsonArray(e.racket_pose);
  const std::string obs_json = JsonArray(e.obs);
  const std::string action_json = JsonArray(e.action);
  const std::string target_json = TargetPayload(e.target);
  const std::string source_contract_sha = PpSha256Hex(NativeSourceContractText());
  const auto& n = e.native_state;
  const std::array<double, 4> policy_quat{{
      e.policy_base_pose[3], e.policy_base_pose[4], e.policy_base_pose[5],
      e.policy_base_pose[6]}};
  const std::array<double, 4> native_quat{{
      e.base_pose[3], e.base_pose[4], e.base_pose[5], e.base_pose[6]}};
  const auto policy_gravity = ProjectedGravityBody(policy_quat);
  const auto native_gravity = ProjectedGravityBody(native_quat);
  double base_position_error_sq = 0.0;
  double gravity_error_sq = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    const double pos_delta = e.policy_base_pose[i] - e.base_pose[i];
    base_position_error_sq += pos_delta * pos_delta;
    const double gravity_delta = policy_gravity[i] - native_gravity[i];
    gravity_error_sq += gravity_delta * gravity_delta;
  }
  double quaternion_dot = 0.0;
  for (std::size_t i = 0; i < 4; ++i) quaternion_dot += policy_quat[i] * native_quat[i];

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << '{'
      << "\"all_finite\":true,"
      << "\"artifact_kind\":\"gate3_first_tick_joined_source_diagnostic\","
      << "\"base_pose\":" << base_json << ','
      << "\"base_pose_sha256\":\"" << PpSha256Hex(base_json) << "\","
      << "\"execution_semantics\":\"shadow_policy_compute_no_command_publish\","
      << "\"evaluation_contract_exact\":false,"
      << "\"exactness\":{"
      << "\"inexact_reasons\":["
      << "\"vendor_native_topics_lack_common_sim_sample_sequence\","
      << "\"planner_same_tick_snapshot_and_payload_epoch_not_merged\","
      << "\"runtime_binary_config_publisher_transitive_closure_not_bound\"],"
      << "\"native_sample_alignment_exact\":false,"
      << "\"planner_snapshot_exact\":false,"
      << "\"runtime_artifact_closure_exact\":false,"
      << "\"source_binary_binding_exact\":false,"
      << "\"source_semantics_closure_exact\":false},"
      << "\"joint_names\":" << JsonStringArray(e.joint_names) << ','
      << "\"joint_order\":\"vendor_mujoco_backend_31\","
      << "\"join_semantics\":\"closest_receipt_window_not_common_sim_tick\","
      << "\"layout\":{"
      << "\"base_pose\":\"xyz_quat_wxyz\","
      << "\"obs\":\"deploy_parity_face179\","
      << "\"qpos\":\"joined_diagnostic_free_xyz_quat_wxyz_then_31_joint_names\","
      << "\"qvel\":\"joined_diagnostic_free_linear_xyz_angular_xyz_then_31_joint_names\","
      << "\"racket_pose\":\"xyz_quat_wxyz\"},"
      << "\"model\":{"
      << "\"path\":\"" << JsonEscape(e.model_path) << "\","
      << "\"sha256\":\"" << e.model_sha256 << "\","
      << "\"source_checkpoint_sha256\":\"" << e.source_checkpoint_sha256 << "\","
      << "\"training_contract_sha256\":\"" << e.training_contract_sha256 << "\"},"
      << "\"native_state_source\":{"
      << "\"base_linear_velocity_semantics\":\"vendor_mujoco_native_framelinvel_world_not_estimated\","
      << "\"base_pose_receive_monotonic_ns\":" << n.base_pose_receive_monotonic_ns << ','
      << "\"base_pose_receive_system_ns\":" << n.base_pose_receive_system_ns << ','
      << "\"base_pose_stamp_ns\":" << n.base_pose_stamp_ns << ','
      << "\"base_twist_receive_monotonic_ns\":" << n.base_twist_receive_monotonic_ns << ','
      << "\"base_twist_receive_system_ns\":" << n.base_twist_receive_system_ns << ','
      << "\"base_twist_stamp_ns\":" << n.base_twist_stamp_ns << ','
      << "\"contract_sha256\":\"" << source_contract_sha << "\","
      << "\"path\":\"" << JsonEscape(e.native_state_path) << "\","
      << "\"racket_pose_receive_monotonic_ns\":" << n.racket_pose_receive_monotonic_ns << ','
      << "\"racket_pose_receive_system_ns\":" << n.racket_pose_receive_system_ns << ','
      << "\"racket_pose_stamp_ns\":" << n.racket_pose_stamp_ns << ','
      << "\"sample_alignment_exact\":false,"
      << "\"sequence\":" << n.sequence << "},"
      << "\"obs\":" << obs_json << ','
      << "\"obs_sha256\":\"" << PpSha256Hex(obs_json) << "\","
      << "\"policy_action\":" << action_json << ','
      << "\"policy_action_sha256\":\"" << PpSha256Hex(action_json) << "\","
      << "\"policy_tick\":" << e.policy_tick << ','
      << "\"policy_observation_base_pose\":" << policy_base_json << ','
      << "\"policy_observation_base_pose_sha256\":\""
      << PpSha256Hex(policy_base_json) << "\","
      << "\"policy_observation_base_semantics\":\"external_position_plus_yaw_aligned_pelvis_imu\","
      << "\"policy_observation_base_source_age_s\":"
      << Number(e.policy_base_source_age_s) << ','
      << "\"policy_vs_native_base_position_error_m\":"
      << Number(std::sqrt(base_position_error_sq)) << ','
      << "\"policy_vs_native_projected_gravity_error\":"
      << Number(std::sqrt(gravity_error_sq)) << ','
      << "\"policy_vs_native_quaternion_abs_dot\":"
      << Number(std::fabs(quaternion_dot)) << ','
      << "\"qpos\":" << qpos_json << ','
      << "\"qpos_sha256\":\"" << PpSha256Hex(qpos_json) << "\","
      << "\"qvel\":" << qvel_json << ','
      << "\"qvel_sha256\":\"" << PpSha256Hex(qvel_json) << "\","
      << "\"quaternion_order\":\"wxyz\","
      << "\"racket_pose\":" << racket_json << ','
      << "\"racket_control_point_fk_position_world\":"
      << JsonArray(e.racket_fk_position_world) << ','
      << "\"racket_control_point_fk_position_error_m\":"
      << Number(e.racket_fk_position_error_m) << ','
      << "\"racket_pose_semantics\":\"vendor_mujoco_native_right_racket_site\","
      << "\"racket_pose_sha256\":\"" << PpSha256Hex(racket_json) << "\","
      << "\"reference_time_step\":" << e.reference_time_step << ','
      << "\"robot_state_tick\":" << e.robot_state_tick << ','
      << "\"robot_state_data_ready_ns\":" << e.robot_state_data_ready_ns << ','
      << "\"robot_state_sync_aligned\":"
      << (e.robot_state_sync_aligned ? "true" : "false") << ','
      << "\"robot_state_sync_complete\":"
      << (e.robot_state_sync_complete ? "true" : "false") << ','
      << "\"robot_state_sync_ready_ns\":" << e.robot_state_sync_ready_ns << ','
      << "\"robot_state_sync_skew_ns\":" << e.robot_state_sync_skew_ns << ','
      << "\"robot_state_timestamp_ns\":" << e.robot_state_timestamp_ns << ','
      << "\"schema_version\":1,"
      << "\"target\":" << target_json << ','
      << "\"target_sha256\":\"" << PpSha256Hex(target_json) << "\"}"
      ;
  return out.str();
}

class ExclusiveJsonSink {
 public:
  ExclusiveJsonSink() = default;
  ExclusiveJsonSink(const ExclusiveJsonSink&) = delete;
  ExclusiveJsonSink& operator=(const ExclusiveJsonSink&) = delete;
  ~ExclusiveJsonSink() {
    if (dir_fd_ >= 0) ::close(dir_fd_);
  }

  bool Prepare(const std::string& output_path, std::string& error) {
    if (dir_fd_ >= 0) {
      error = "first-tick output sink is already prepared";
      return false;
    }
    const fs::path path(output_path);
    if (!IsCanonicalAbsolutePath(path, false, error)) return false;
    if (path.filename().empty() || path.filename() == "." || path.filename() == "..") {
      error = "first-tick output filename is invalid";
      return false;
    }
    struct stat existing {};
    const int lstat_rc = ::lstat(output_path.c_str(), &existing);
    const int lstat_errno = errno;
    if (lstat_rc == 0 || lstat_errno != ENOENT) {
      error = lstat_rc == 0
                  ? "first-tick output already exists (no-clobber)"
                  : "cannot lstat first-tick output: " +
                        std::string(std::strerror(lstat_errno));
      return false;
    }
    dir_fd_ = ::open(path.parent_path().c_str(),
                     O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (dir_fd_ < 0) {
      error = "cannot pin first-tick output directory: " +
              std::string(std::strerror(errno));
      return false;
    }
    leaf_ = path.filename().string();
    path_ = output_path;
    return true;
  }

  bool Write(const Evidence& evidence, std::string& error) {
    if (dir_fd_ < 0) {
      error = "first-tick output sink is not prepared";
      return false;
    }
    if (written_) {
      error = "first-tick output is one-shot";
      return false;
    }
    if (!ValidateEvidence(evidence, error)) return false;
    const std::string payload = BuildPayload(evidence);
    const std::string bytes =
        std::string("{\"artifact_kind\":\"gate3_first_tick_joined_source_diagnostic\","
                    "\"evaluation_contract_exact\":false,\"payload\":") +
        payload +
        ",\"payload_sha256\":\"" + PpSha256Hex(payload) + "\"}\n";

    struct stat existing {};
    if (::fstatat(dir_fd_, leaf_.c_str(), &existing, AT_SYMLINK_NOFOLLOW) == 0 ||
        errno != ENOENT) {
      error = "first-tick output appeared before write (no-clobber)";
      return false;
    }
    static std::atomic<std::uint64_t> serial{0};
    int temp_fd = -1;
    std::string temp;
    for (int attempt = 0; attempt < 32 && temp_fd < 0; ++attempt) {
      temp = "." + leaf_ + ".tmp." + std::to_string(::getpid()) + "." +
             std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
      temp_fd = ::openat(dir_fd_, temp.c_str(),
                         O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                         0600);
      if (temp_fd < 0 && errno != EEXIST) break;
    }
    if (temp_fd < 0) {
      error = "cannot create exclusive first-tick temporary file: " +
              std::string(std::strerror(errno));
      return false;
    }
    if (::fchmod(temp_fd, 0600) != 0) {
      const int saved = errno;
      ::close(temp_fd);
      ::unlinkat(dir_fd_, temp.c_str(), 0);
      error = "cannot enforce mode 0600 on first-tick temporary file: " +
              std::string(std::strerror(saved));
      return false;
    }
    bool ok = true;
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      const ssize_t n = ::write(temp_fd, bytes.data() + offset, bytes.size() - offset);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) { ok = false; break; }
      offset += static_cast<std::size_t>(n);
    }
    if (ok && ::fsync(temp_fd) != 0) ok = false;
    if (::close(temp_fd) != 0) ok = false;
    if (!ok) {
      const int saved = errno;
      ::unlinkat(dir_fd_, temp.c_str(), 0);
      error = "cannot durably write first-tick temporary file: " +
              std::string(std::strerror(saved));
      return false;
    }
    if (::linkat(dir_fd_, temp.c_str(), dir_fd_, leaf_.c_str(), 0) != 0) {
      const int saved = errno;
      ::unlinkat(dir_fd_, temp.c_str(), 0);
      error = "atomic no-replace first-tick link failed: " +
              std::string(std::strerror(saved));
      return false;
    }
    if (::fsync(dir_fd_) != 0) {
      const int saved = errno;
      ::unlinkat(dir_fd_, leaf_.c_str(), 0);
      ::unlinkat(dir_fd_, temp.c_str(), 0);
      (void)::fsync(dir_fd_);
      error = "cannot durably commit first-tick output directory: " +
              std::string(std::strerror(saved));
      return false;
    }
    if (::unlinkat(dir_fd_, temp.c_str(), 0) != 0 || ::fsync(dir_fd_) != 0) {
      const int saved = errno;
      (void)::unlinkat(dir_fd_, leaf_.c_str(), 0);
      (void)::unlinkat(dir_fd_, temp.c_str(), 0);
      (void)::fsync(dir_fd_);
      error = "cannot finish first-tick no-replace directory transaction: " +
              std::string(std::strerror(saved));
      return false;
    }
    written_ = true;
    return true;
  }

  const std::string& path() const { return path_; }
  bool written() const { return written_; }

 private:
  int dir_fd_ = -1;
  std::string leaf_;
  std::string path_;
  bool written_ = false;
};

inline bool StableRegularFileBytesAndSha256(const std::string& path,
                                            std::string& bytes,
                                            std::string& digest,
                                            std::string& error) {
  bytes.clear();
  digest.clear();
  if (!IsCanonicalAbsolutePath(fs::path(path), true, error)) return false;
  struct stat lst {};
  if (::lstat(path.c_str(), &lst) != 0 || S_ISLNK(lst.st_mode) || !S_ISREG(lst.st_mode)) {
    error = "hash input must be a canonical non-symlink regular file";
    return false;
  }
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    error = "cannot open hash input: " + std::string(std::strerror(errno));
    return false;
  }
  struct stat before {};
  if (::fstat(fd, &before) != 0 || before.st_dev != lst.st_dev ||
      before.st_ino != lst.st_ino || before.st_size != lst.st_size ||
      StatMtime(before) != StatMtime(lst)) {
    error = "hash input identity changed before read";
    ::close(fd);
    return false;
  }
  if (before.st_size < 0 || static_cast<std::uint64_t>(before.st_size) >
                              static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    error = "hash input is too large";
    ::close(fd);
    return false;
  }
  bytes.resize(static_cast<std::size_t>(before.st_size));
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t n = ::read(fd, bytes.data() + offset, bytes.size() - offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      error = "short/error read while hashing input";
      ::close(fd);
      return false;
    }
    offset += static_cast<std::size_t>(n);
  }
  struct stat after {};
  const bool stable = ::fstat(fd, &after) == 0 &&
      before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
      before.st_size == after.st_size && before.st_mode == after.st_mode &&
      StatMtime(before) == StatMtime(after);
  ::close(fd);
  if (!stable) {
    error = "hash input identity changed during read";
    return false;
  }
  digest = PpSha256Hex(bytes);
  return true;
}

inline bool StableRegularFileSha256(const std::string& path, std::string& digest,
                                    std::string& error) {
  std::string bytes;
  return StableRegularFileBytesAndSha256(path, bytes, digest, error);
}

}  // namespace first_tick
}  // namespace a3_pingpong
