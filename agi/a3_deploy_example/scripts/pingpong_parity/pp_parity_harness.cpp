// Standalone C++<->Python ONNX parity harness for the ping-pong runner.
// Loads model_15200.onnx via the SAME PpOnnxPolicy used in the deploy runner,
// reads an obs[180] from a text file, runs the bundled onnxruntime, and prints
// action[31] and target_q[31]. No ROS/AimRT needed.
//   usage: pp_parity_harness <model.onnx> <obs.txt> <time_step>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include "a3_pingpong/pp_onnx_policy.hpp"

static Eigen::VectorXd read_vec(const char* path, int n) {
  std::ifstream f(path);
  std::vector<double> v;
  double x;
  while (f >> x) v.push_back(x);
  if ((int)v.size() != n) {
    std::fprintf(stderr, "expected %d values in %s, got %zu\n", n, path, v.size());
    std::exit(2);
  }
  Eigen::VectorXd out(n);
  for (int i = 0; i < n; ++i) out[i] = v[i];
  return out;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model.onnx> <obs.txt> <time_step>\n", argv[0]);
    return 1;
  }
  using namespace a3_pingpong;
  PpOnnxPolicy onnx(argv[1]);
  // obs dim from the loaded model (175 deploy_parity / 180 full) — a hardcoded kObsDim
  // made this harness reject every deploy_parity model.
  Eigen::VectorXd obs = read_vec(argv[2], onnx.obs_dim());
  int ts = std::atoi(argv[3]);

  Eigen::VectorXd act = onnx.mean_action(obs, ts);
  Eigen::VectorXd tq = onnx.target_q(act);

  // joint_names / metadata sanity
  std::fprintf(stderr, "[cpp] joints=%zu obsdim=%d ts=%d  act min/max/mean=%.5f/%.5f/%.5f\n",
               onnx.joint_names().size(), onnx.obs_dim(), ts, act.minCoeff(), act.maxCoeff(), act.mean());

  std::ostringstream a, t;
  a.precision(17); t.precision(17);
  for (int i = 0; i < kNumJoints; ++i) { a << act[i] << (i + 1 < kNumJoints ? " " : ""); }
  for (int i = 0; i < kNumJoints; ++i) { t << tq[i] << (i + 1 < kNumJoints ? " " : ""); }
  std::printf("ACT %s\n", a.str().c_str());
  std::printf("TQ %s\n", t.str().c_str());
  return 0;
}
