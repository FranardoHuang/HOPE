#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "a3_pingpong/pp_first_tick_json.hpp"

namespace ft = a3_pingpong::first_tick;
namespace fs = std::filesystem;

ft::NativeStateWire Native(std::int64_t system, std::int64_t monotonic) {
  ft::NativeStateWire n;
  n.sequence = 2;
  n.base_pose_stamp_ns = system - 3000000;
  n.base_twist_stamp_ns = system - 2000000;
  n.racket_pose_stamp_ns = system - 1000000;
  n.base_pose_receive_monotonic_ns = monotonic - 3000000;
  n.base_twist_receive_monotonic_ns = monotonic - 2000000;
  n.racket_pose_receive_monotonic_ns = monotonic - 1000000;
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

ft::Evidence Evidence(const ft::NativeStateWire& n, std::int64_t system) {
  ft::Evidence e;
  e.model_path = "/canonical/model.onnx";
  e.model_sha256 = std::string(64, 'b');
  e.training_contract_sha256 = std::string(64, 'c');
  e.source_checkpoint_sha256 = std::string(64, 'd');
  e.native_state_path = "/dev/shm/pp_gate3_first_tick_state_v1";
  e.policy_tick = 7;
  e.robot_state_timestamp_ns = system - 4000000;
  e.robot_state_tick = 11;
  e.robot_state_data_ready_ns = system - 4000000;
  e.robot_state_sync_ready_ns = system - 3500000;
  e.robot_state_sync_complete = true;
  e.robot_state_sync_aligned = true;
  e.robot_state_sync_skew_ns = 1000000;
  e.policy_base_source_age_s = 0.004;
  e.reference_time_step = 12;
  e.native_state = n;
  for (int i = 0; i < 31; ++i) e.joint_names.push_back("joint_" + std::to_string(i));
  e.qpos.assign(n.base_position_world.begin(), n.base_position_world.end());
  e.qpos.insert(e.qpos.end(), n.base_quaternion_wxyz.begin(), n.base_quaternion_wxyz.end());
  e.qpos.resize(38, 0.0);
  e.qvel.assign(n.base_linear_velocity_world.begin(), n.base_linear_velocity_world.end());
  e.qvel.insert(e.qvel.end(), n.base_angular_velocity_world.begin(),
                n.base_angular_velocity_world.end());
  e.qvel.resize(37, 0.0);
  e.base_pose.assign(e.qpos.begin(), e.qpos.begin() + 7);
  e.policy_base_pose = e.base_pose;
  e.racket_pose.assign(n.racket_position_world.begin(), n.racket_position_world.end());
  e.racket_pose.insert(e.racket_pose.end(), n.racket_quaternion_wxyz.begin(),
                       n.racket_quaternion_wxyz.end());
  e.racket_fk_position_world.assign(n.racket_position_world.begin(),
                                    n.racket_position_world.end());
  e.obs.assign(179, 0.0);
  e.action.assign(31, 0.0);
  e.target.position_world = {{0.7, -0.4, 0.9}};
  e.target.velocity_world = {{1.0, 1.0, 0.5}};
  e.target.normal_raw_mount_a_world = {{1.0, 0.0, 0.0}};
  e.target.time_to_strike = 0.8;
  e.target.swing_type = 1.0;
  e.target.valid = true;
  return e;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  assert(!ft::PlannerActorCandidateEligible(
      179, true, false, false, 0, false, 1.0, 0.0));
  assert(ft::PlannerActorCandidateEligible(
      179, true, true, true, 1, true, -1.0, 0.0));

  const auto system = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  const auto monotonic = ft::MonotonicNowNs();
  const auto native = Native(system, monotonic);
  const fs::path root = fs::canonical(argv[1]);
  const fs::path state_path = root / "native";
  const int state_fd = ::open(state_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  assert(state_fd >= 0);
  assert(::ftruncate(state_fd, sizeof(native)) == 0);
  assert(::flock(state_fd, LOCK_EX) == 0);
  assert(::pwrite(state_fd, &native, sizeof(native), 0) ==
         static_cast<ssize_t>(sizeof(native)));
  assert(::flock(state_fd, LOCK_UN) == 0);
  assert(::close(state_fd) == 0);
  ft::NativeStateSource source;
  std::string error;
  assert(source.Open(state_path.string(), error));
  ft::NativeStateWire read;
  assert(source.Latest(read, error));
  assert(read.base_linear_velocity_world[0] == 0.125);

  auto odd = native;
  odd.sequence = 3;
  const fs::path odd_state_path = root / "native-odd";
  const int odd_fd = ::open(odd_state_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  assert(odd_fd >= 0);
  assert(::ftruncate(odd_fd, sizeof(odd)) == 0);
  assert(::pwrite(odd_fd, &odd, sizeof(odd), 0) ==
         static_cast<ssize_t>(sizeof(odd)));
  assert(::close(odd_fd) == 0);
  ft::NativeStateSource odd_source;
  assert(odd_source.Open(odd_state_path.string(), error));
  ft::NativeStateWire odd_read;
  assert(!odd_source.Latest(odd_read, error));

  auto evidence = Evidence(read, system);
  assert(ft::ValidateEvidence(evidence, error));
  const fs::path output = root / "first.json";
  ft::ExclusiveJsonSink sink;
  assert(sink.Prepare(output.string(), error));
  assert(sink.Write(evidence, error));
  std::ifstream input(output);
  const std::string bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  assert(bytes.find("\"payload_sha256\"") != std::string::npos);
  assert(bytes.find("not_estimated") != std::string::npos);
  assert(!sink.Write(evidence, error));

  const fs::path raced_output = root / "raced.json";
  ft::ExclusiveJsonSink raced_sink;
  assert(raced_sink.Prepare(raced_output.string(), error));
  const int competitor = ::open(raced_output.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  assert(competitor >= 0);
  constexpr char kCompetitor[] = "competitor-owned\n";
  assert(::write(competitor, kCompetitor, sizeof(kCompetitor) - 1) ==
         static_cast<ssize_t>(sizeof(kCompetitor) - 1));
  assert(::close(competitor) == 0);
  assert(!raced_sink.Write(evidence, error));
  std::ifstream raced_input(raced_output);
  const std::string raced_bytes((std::istreambuf_iterator<char>(raced_input)),
                                std::istreambuf_iterator<char>());
  assert(raced_bytes == kCompetitor);

  const fs::path model_path = root / "model.onnx";
  const std::string expected_model("onnx\0bytes", 10);
  const int model_fd = ::open(model_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  assert(model_fd >= 0);
  assert(::write(model_fd, expected_model.data(), expected_model.size()) ==
         static_cast<ssize_t>(expected_model.size()));
  assert(::close(model_fd) == 0);
  std::string loaded_model;
  std::string loaded_model_sha;
  assert(ft::StableRegularFileBytesAndSha256(
      model_path.string(), loaded_model, loaded_model_sha, error));
  assert(loaded_model == expected_model);
  assert(loaded_model_sha == a3_pingpong::PpSha256Hex(expected_model));

  auto fabricated_velocity = evidence;
  fabricated_velocity.qvel[0] = 0.0;
  assert(!ft::ValidateEvidence(fabricated_velocity, error));
  auto stale_join = evidence;
  stale_join.robot_state_sync_ready_ns -= 1000000000LL;
  assert(!ft::ValidateEvidence(stale_join, error));
  return 0;
}
