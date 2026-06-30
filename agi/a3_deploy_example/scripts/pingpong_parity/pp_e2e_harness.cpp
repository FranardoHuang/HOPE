// End-to-end C++ front-end exercise: synthetic nominal RobotState -> ComputeCommand.
#include <cstdio>
#include <cmath>
#include "a3_pingpong/pp_policy.hpp"
#include "robot_io/robot_io_backend.hpp"
using namespace a3_pingpong;
static void stats(const char* nm, const Eigen::VectorXd& v){
  bool fin=v.allFinite();
  std::printf("  %-10s n=%ld min=%+.4f max=%+.4f mean=%+.4f |.|=%.4f finite=%d\n",
    nm, (long)v.size(), v.minCoeff(), v.maxCoeff(), v.mean(), v.norm(), (int)fin);
}
int main(int argc, char** argv){
  std::string model = argc>1?argv[1]:"dist/a3_deploy_x86_64/models/model_15200.onnx";
  PpPolicyConfig cfg; cfg.level = argc>2?atoi(argv[2]):1;
  cfg.loc_mode = LocMode::kPerfectTracking;
  PpPolicy pp(model, cfg);
  Eigen::VectorXd q_sdk = to_sdk_order(pp.onnx().default_q(), pp.isaac_to_sdk());
  robot_io::RobotState st;
  st.q=q_sdk; st.dq=Eigen::VectorXd::Zero(31); st.tau_est=Eigen::VectorXd::Zero(31);
  st.imu_quat_wxyz=Eigen::Vector4d(1,0,0,0); st.imu_gyro=Eigen::Vector3d::Zero();
  st.sync_aligned=true; st.sync_complete=true;
  // run several ticks (clock advances)
  robot_io::RobotCommand cmd;
  for(int t=0;t<3;++t){ if(!pp.ComputeCommand(t,st,cmd)){std::printf("ComputeCommand FALSE\n");return 3;} }
  std::printf("=== ComputeCommand OK (level=%d, loc=%s) ===\n", cfg.level, pp.loc_mode_name());
  std::printf("dims: q_des=%ld dq_des=%ld tau_ff=%ld kp=%ld kd=%ld (expect 31)\n",
    (long)cmd.q_des.size(),(long)cmd.dq_des.size(),(long)cmd.tau_ff.size(),(long)cmd.kp.size(),(long)cmd.kd.size());
  stats("q_des", cmd.q_des); stats("kp", cmd.kp); stats("kd", cmd.kd);
  stats("action", pp.last_action());
  const auto& B = backend_joint_order();
  std::printf("neck slots: %s q=%.3f kp=%.1f kd=%.1f | %s q=%.3f kp=%.1f kd=%.1f\n",
    B[3].c_str(),cmd.q_des[3],cmd.kp[3],cmd.kd[3], B[4].c_str(),cmd.q_des[4],cmd.kp[4],cmd.kd[4]);
  // sanity: any huge q_des?
  double worst=0; int wi=0; for(int i=0;i<31;++i){double a=std::fabs(cmd.q_des[i]); if(a>worst){worst=a;wi=i;}}
  std::printf("max|q_des|=%.3f at %s\n", worst, B[wi].c_str());
  bool bad = !cmd.q_des.allFinite()||!cmd.kp.allFinite()||!cmd.kd.allFinite()||worst>3.5;
  std::printf("RESULT: %s\n", bad?"FAIL (nan/inf/huge)":"PASS (finite, bounded)");
  return bad?1:0;
}
