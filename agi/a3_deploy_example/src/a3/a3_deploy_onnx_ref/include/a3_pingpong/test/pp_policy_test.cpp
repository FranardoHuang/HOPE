// Off-robot test of the full CommandFn path: synthetic RobotState (robot at
// nominal, upright IMU) -> ComputeCommand -> RobotCommand. Verifies sizes,
// finiteness, neck-passive (q=0,kp=40,kd=2), non-neck gains == scattered
// metadata, target_q in range, and that the reference clock sweeps a swing.
//   ./pp_policy_test <policy.onnx>
#include <cstdio>

#include "a3_pingpong/pp_policy.hpp"

using namespace a3_pingpong;

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s policy.onnx\n", argv[0]); return 2; }
  PpPolicyConfig cfg; cfg.level = 1;
  PpPolicy pol(argv[1], cfg);

  // synthetic state: robot AT nominal pose (default_q scattered to SDK), upright IMU
  robot_io::RobotState st;
  st.q = to_sdk_order(pol.onnx().default_q(), pol.isaac_to_sdk());
  st.dq = Eigen::VectorXd::Zero(31);
  st.imu_quat_wxyz = Eigen::Vector4d(1, 0, 0, 0);
  st.imu_gyro = Eigen::Vector3d::Zero();

  Eigen::VectorXd expect_kp = to_sdk_order(pol.onnx().kp(), pol.isaac_to_sdk());
  Eigen::VectorXd expect_kd = to_sdk_order(pol.onnx().kd(), pol.isaac_to_sdk());

  int fails = 0;
  double max_abs_q = 0, max_abs_a = 0;
  std::printf("tick  time_step  |action|  max|q_des|  (swing sweep)\n");
  for (std::uint64_t tick = 0; tick <= 160; tick += 16) {
    robot_io::RobotCommand cmd;
    bool ok = pol.ComputeCommand(tick, st, cmd);
    if (!ok) { fails++; continue; }
    // sizes
    if (cmd.q_des.size() != 31 || cmd.kp.size() != 31 || cmd.kd.size() != 31 ||
        cmd.dq_des.size() != 31 || cmd.tau_ff.size() != 31) { std::printf("size FAIL\n"); fails++; }
    if (!cmd.q_des.allFinite() || !cmd.kp.allFinite() || !cmd.kd.allFinite()) { std::printf("nan FAIL\n"); fails++; }
    // neck passive
    for (int s : {kHeadSlot0, kHeadSlot1}) {
      if (std::abs(cmd.q_des[s]) > 1e-12 || std::abs(cmd.kp[s] - kHeadKp) > 1e-9 ||
          std::abs(cmd.kd[s] - kHeadKd) > 1e-9) { std::printf("neck FAIL slot %d\n", s); fails++; }
    }
    // non-neck gains == scattered metadata
    for (int s = 0; s < 31; ++s) {
      if (s == kHeadSlot0 || s == kHeadSlot1) continue;
      if (std::abs(cmd.kp[s] - expect_kp[s]) > 1e-9 || std::abs(cmd.kd[s] - expect_kd[s]) > 1e-9) {
        std::printf("gain FAIL slot %d\n", s); fails++; break;
      }
    }
    double mq = cmd.q_des.cwiseAbs().maxCoeff(), ma = pol.last_action().norm();
    max_abs_q = std::max(max_abs_q, mq); max_abs_a = std::max(max_abs_a, ma);
    std::printf(" %3llu     %3d      %6.3f     %6.3f\n", (unsigned long long)tick,
                pol.last_time_step(), ma, mq);
  }
  // sanity: targets stay in a plausible joint range, clock reaches the strike frame
  bool range_ok = max_abs_q < 3.5;
  std::printf("max|q_des|=%.3f max|action|=%.3f fails=%d\n", max_abs_q, max_abs_a, fails);
  bool pass = (fails == 0) && range_ok;
  std::printf("%s\n", pass ? "POLICY CALLBACK PASS" : "POLICY CALLBACK FAIL");
  return pass ? 0 : 1;
}
