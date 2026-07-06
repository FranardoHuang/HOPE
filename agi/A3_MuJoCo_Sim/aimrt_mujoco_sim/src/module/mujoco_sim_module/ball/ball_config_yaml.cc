// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
//
// The ONLY ball-dynamics translation unit that depends on yaml-cpp: it parses configs/ball_physics_venue.yaml
// (the cross-language single source of truth) into a BallPhysicsConfig. Kept separate so the math kernel
// (ball_dynamics.cc) stays dependency-light and standalone-testable.

#include <stdexcept>

#include "mujoco_sim_module/ball/ball_dynamics.h"
#include "yaml-cpp/yaml.h"

namespace aimrt_mujoco_sim::mujoco_sim_module::ball {

BallPhysicsConfig BallPhysicsConfig::LoadFromYaml(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const std::exception& e) {
    throw std::runtime_error("ball_physics.yaml load failed (" + path + "): " + e.what());
  }

  BallPhysicsConfig cfg;
  const auto& b = root["ball"];
  cfg.ball.mass = b["mass"].as<double>();
  cfg.ball.radius = b["radius"].as<double>();
  cfg.ball.inertia_coeff = b["inertia_coeff"].as<double>();
  cfg.air_rho = root["air"]["rho"].as<double>();

  const auto& f = root["flight"];
  cfg.flight.k_d = f["k_d"].as<double>();
  cfg.flight.k_m = f["k_m"].as<double>();
  cfg.flight.g = f["g"].as<double>();
  cfg.flight.rk4_h = f["rk4_h"].as<double>();
  cfg.flight.ball_radius = cfg.ball.radius;
  // Venue F2 saturating-Magnus data (behavior switch use_saturating_magnus is NOT read from the
  // yaml — it stays a per-run code-level flag, default off).
  if (f["cl_slope"] && f["cl_max"] && f["cl_slope"].as<double>() > 0.0) {
    cfg.flight.magnus_sr_sat = f["cl_max"].as<double>() / f["cl_slope"].as<double>();
  }

  auto parse_contact = [&](const YAML::Node& n) {
    ContactParams cp;
    cp.e_eff = n["e_eff"].as<double>();
    cp.a_t = n["a_t"].as<double>();
    cp.b_t = n["b_t"].as<double>();
    cp.mu_safety = n["mu_safety"].as<double>();
    cp.ball_radius = cfg.ball.radius;
    cp.inertia_coeff = cfg.ball.inertia_coeff;
    // Venue F4: velocity-dependent restitution (paddle section carries these keys).
    if (n["e_exp_g1"] && n["e_exp_g2"]) {
      cp.e_exp_g1 = n["e_exp_g1"].as<double>();
      cp.e_exp_g2 = n["e_exp_g2"].as<double>();
      cp.use_e_exp = true;
    }
    return cp;
  };
  cfg.table = parse_contact(root["contact"]["table"]);
  cfg.paddle = parse_contact(root["contact"]["paddle"]);
  return cfg;
}

}  // namespace aimrt_mujoco_sim::mujoco_sim_module::ball
