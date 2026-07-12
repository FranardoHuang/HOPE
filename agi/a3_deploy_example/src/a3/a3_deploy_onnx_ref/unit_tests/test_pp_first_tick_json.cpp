#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "a3_pingpong/pp_first_tick_json.hpp"

namespace {
namespace ft = a3_pingpong::first_tick;
namespace fs = std::filesystem;

class TempDir {
 public:
  TempDir() {
    std::string pattern = (fs::temp_directory_path() / "pp-ft-json-XXXXXX").string();
    std::vector<char> chars(pattern.begin(), pattern.end());
    chars.push_back('\0');
    char* made = ::mkdtemp(chars.data());
    if (!made) throw std::runtime_error("mkdtemp failed");
    path = fs::canonical(made);
  }
  ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
  fs::path path;
};

ft::NativeStateWire ValidNativeState() {
  ft::NativeStateWire n;
  n.sequence = 2;
  const auto mono = ft::MonotonicNowNs();
  const auto system = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  n.base_pose_stamp_ns = system - 3000000;
  n.base_twist_stamp_ns = system - 2000000;
  n.racket_pose_stamp_ns = system - 1000000;
  n.base_pose_receive_monotonic_ns = mono - 3000000;
  n.base_twist_receive_monotonic_ns = mono - 2000000;
  n.racket_pose_receive_monotonic_ns = mono - 1000000;
  n.base_pose_receive_system_ns = system - 3000000;
  n.base_twist_receive_system_ns = system - 2000000;
  n.racket_pose_receive_system_ns = system - 1000000;
  n.base_position_world = {{0.1, -0.2, 1.0}};
  n.base_quaternion_wxyz = {{1.0, 0.0, 0.0, 0.0}};
  n.base_linear_velocity_world = {{0.125, -0.25, 0.375}};
  n.base_angular_velocity_world = {{0.01, 0.02, 0.03}};
  n.racket_position_world = {{0.7, -0.4, 0.9}};
  n.racket_quaternion_wxyz = {{1.0, 0.0, 0.0, 0.0}};
  return n;
}

ft::Evidence ValidEvidence() {
  ft::Evidence e;
  e.model_path = "/canonical/model.onnx";
  e.model_sha256 = std::string(64, 'b');
  e.training_contract_sha256 = std::string(64, 'c');
  e.source_checkpoint_sha256 = std::string(64, 'd');
  e.native_state_path = "/dev/shm/pp_gate3_first_tick_state_v1";
  e.policy_tick = 7;
  const auto system = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  e.robot_state_timestamp_ns = system - 4000000;
  e.robot_state_tick = 11;
  e.robot_state_data_ready_ns = system - 4000000;
  e.robot_state_sync_ready_ns = system - 3500000;
  e.robot_state_sync_complete = true;
  e.robot_state_sync_aligned = true;
  e.robot_state_sync_skew_ns = 1000000;
  e.policy_base_source_age_s = 0.004;
  e.reference_time_step = 12;
  e.native_state = ValidNativeState();
  for (int i = 0; i < 31; ++i) e.joint_names.push_back("joint_" + std::to_string(i));
  e.qpos.assign(e.native_state.base_position_world.begin(),
                e.native_state.base_position_world.end());
  e.qpos.insert(e.qpos.end(), e.native_state.base_quaternion_wxyz.begin(),
                e.native_state.base_quaternion_wxyz.end());
  e.qpos.resize(38, 0.0);
  e.qvel.assign(e.native_state.base_linear_velocity_world.begin(),
                e.native_state.base_linear_velocity_world.end());
  e.qvel.insert(e.qvel.end(), e.native_state.base_angular_velocity_world.begin(),
                e.native_state.base_angular_velocity_world.end());
  e.qvel.resize(37, 0.0);
  e.base_pose.assign(e.qpos.begin(), e.qpos.begin() + 7);
  e.policy_base_pose = e.base_pose;
  e.racket_pose.assign(e.native_state.racket_position_world.begin(),
                       e.native_state.racket_position_world.end());
  e.racket_pose.insert(e.racket_pose.end(),
                       e.native_state.racket_quaternion_wxyz.begin(),
                       e.native_state.racket_quaternion_wxyz.end());
  e.racket_fk_position_world.assign(e.native_state.racket_position_world.begin(),
                                    e.native_state.racket_position_world.end());
  e.racket_fk_position_error_m = 0.0;
  e.obs.assign(179, 0.0);
  e.action.assign(31, 0.0);
  e.target.position_world = {{0.7, -0.4, 0.9}};
  e.target.velocity_world = {{1.0, 1.0, 0.5}};
  e.target.normal_raw_mount_a_world = {{1.0, 0.0, 0.0}};
  e.target.rho = 0.0;
  e.target.time_to_strike = 0.8;
  e.target.swing_type = 1.0;
  e.target.valid = true;
  return e;
}

void WriteNativeState(const fs::path& path, const ft::NativeStateWire& state) {
  const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(::ftruncate(fd, sizeof(state)), 0);
  ASSERT_EQ(::flock(fd, LOCK_EX), 0);
  ASSERT_EQ(::pwrite(fd, &state, sizeof(state), 0),
            static_cast<ssize_t>(sizeof(state)));
  ASSERT_EQ(::flock(fd, LOCK_UN), 0);
  ASSERT_EQ(::close(fd), 0);
}

TEST(PpFirstTickJson, KernelLockedSourcePreservesNativeLinearVelocity) {
  TempDir tmp;
  const fs::path path = tmp.path / "state";
  WriteNativeState(path, ValidNativeState());
  ft::NativeStateSource source;
  std::string error;
  ASSERT_TRUE(source.Open(path.string(), error)) << error;
  ft::NativeStateWire got;
  ASSERT_TRUE(source.Latest(got, error)) << error;
  EXPECT_DOUBLE_EQ(got.base_linear_velocity_world[0], 0.125);
  EXPECT_DOUBLE_EQ(got.base_linear_velocity_world[1], -0.25);
  EXPECT_DOUBLE_EQ(got.base_linear_velocity_world[2], 0.375);
}

TEST(PpFirstTickJson, IdleDoesNotConsumeFormalOneShot) {
  EXPECT_FALSE(ft::PlannerActorCandidateEligible(
      179, true, false, false, 0, false, 1.0, 0.0));
  EXPECT_FALSE(ft::PlannerActorCandidateEligible(
      179, true, false, true, 1, true, 1.0, 0.0));
  EXPECT_FALSE(ft::PlannerActorCandidateEligible(
      179, true, true, true, 1, false, 1.0, 0.0));
  EXPECT_TRUE(ft::PlannerActorCandidateEligible(
      179, true, true, true, 1, true, -1.0, 0.0));
}

TEST(PpFirstTickJson, SourceRejectsIncompleteNonfiniteAndSymlink) {
  TempDir tmp;
  auto bad = ValidNativeState();
  bad.base_twist_stamp_ns = 0;
  bad.base_linear_velocity_world[0] = std::numeric_limits<double>::quiet_NaN();
  const fs::path path = tmp.path / "bad";
  WriteNativeState(path, bad);
  ft::NativeStateSource source;
  std::string error;
  ASSERT_TRUE(source.Open(path.string(), error)) << error;
  ft::NativeStateWire got;
  EXPECT_FALSE(source.Latest(got, error));

  const fs::path odd_path = tmp.path / "odd";
  auto odd = ValidNativeState();
  odd.sequence = 3;
  WriteNativeState(odd_path, odd);
  ft::NativeStateSource odd_source;
  ASSERT_TRUE(odd_source.Open(odd_path.string(), error)) << error;
  EXPECT_FALSE(odd_source.Latest(got, error));

  const fs::path link = tmp.path / "link";
  ASSERT_EQ(::symlink(path.c_str(), link.c_str()), 0);
  ft::NativeStateSource linked;
  EXPECT_FALSE(linked.Open(link.string(), error));
}

TEST(PpFirstTickJson, AtomicNoReplaceAndContentBinding) {
  TempDir tmp;
  const fs::path output = tmp.path / "first.json";
  ft::ExclusiveJsonSink sink;
  std::string error;
  ASSERT_TRUE(sink.Prepare(output.string(), error)) << error;
  ASSERT_TRUE(sink.Write(ValidEvidence(), error)) << error;
  std::ifstream in(output);
  const std::string bytes((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  EXPECT_NE(bytes.find("\"payload_sha256\""), std::string::npos);
  EXPECT_NE(bytes.find("\"evaluation_contract_exact\":false"),
            std::string::npos);
  EXPECT_NE(bytes.find("\"planner_snapshot_exact\":false"),
            std::string::npos);
  EXPECT_EQ(bytes.find("\"source_commit\""), std::string::npos);
  EXPECT_NE(bytes.find("\"qpos_sha256\""), std::string::npos);
  EXPECT_NE(bytes.find("\"obs_sha256\""), std::string::npos);
  EXPECT_NE(bytes.find("vendor_mujoco_native_framelinvel_world_not_estimated"),
            std::string::npos);
  struct stat st {};
  ASSERT_EQ(::stat(output.c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 0777, 0600);
  EXPECT_FALSE(sink.Write(ValidEvidence(), error));
  std::ifstream in_again(output);
  const std::string bytes_again((std::istreambuf_iterator<char>(in_again)),
                                std::istreambuf_iterator<char>());
  EXPECT_EQ(bytes_again, bytes);

  const fs::path raced_output = tmp.path / "raced.json";
  ft::ExclusiveJsonSink raced_sink;
  ASSERT_TRUE(raced_sink.Prepare(raced_output.string(), error)) << error;
  const int competitor = ::open(raced_output.c_str(),
                                O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
  ASSERT_GE(competitor, 0);
  constexpr char kCompetitor[] = "competitor-owned\n";
  ASSERT_EQ(::write(competitor, kCompetitor, sizeof(kCompetitor) - 1),
            static_cast<ssize_t>(sizeof(kCompetitor) - 1));
  ASSERT_EQ(::close(competitor), 0);
  EXPECT_FALSE(raced_sink.Write(ValidEvidence(), error));
  std::ifstream raced_in(raced_output);
  const std::string raced_bytes((std::istreambuf_iterator<char>(raced_in)),
                                std::istreambuf_iterator<char>());
  EXPECT_EQ(raced_bytes, kCompetitor);
}

TEST(PpFirstTickJson, InvalidEvidenceLeavesNoArtifact) {
  TempDir tmp;
  const fs::path output = tmp.path / "first.json";
  ft::ExclusiveJsonSink sink;
  std::string error;
  ASSERT_TRUE(sink.Prepare(output.string(), error)) << error;
  auto bad = ValidEvidence();
  bad.qvel[0] = 0.0;  // may not replace the native 0.125 m/s with a placeholder.
  EXPECT_FALSE(sink.Write(bad, error));
  EXPECT_FALSE(fs::exists(output));
}

TEST(PpFirstTickJson, ModelHashBindsTheBytesReturnedToTheOnnxLoader) {
  TempDir tmp;
  const fs::path model = tmp.path / "model.onnx";
  const std::string expected("onnx\0bytes", 10);
  const int fd = ::open(model.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(::write(fd, expected.data(), expected.size()),
            static_cast<ssize_t>(expected.size()));
  ASSERT_EQ(::close(fd), 0);
  std::string loaded;
  std::string digest;
  std::string error;
  ASSERT_TRUE(ft::StableRegularFileBytesAndSha256(
      model.string(), loaded, digest, error)) << error;
  EXPECT_EQ(loaded, expected);
  EXPECT_EQ(digest, a3_pingpong::PpSha256Hex(expected));
}

TEST(PpFirstTickJson, CrossSourceTimingAndPolicyBaseRelationFailClosed) {
  std::string error;
  auto bad_time = ValidEvidence();
  bad_time.robot_state_sync_ready_ns -= 1000000000LL;
  EXPECT_FALSE(ft::ValidateEvidence(bad_time, error));

  auto bad_policy_base = ValidEvidence();
  bad_policy_base.policy_base_pose[0] += 0.10;
  EXPECT_FALSE(ft::ValidateEvidence(bad_policy_base, error));
}

}  // namespace
