// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
//
// BallPhysics: the MuJoCo-facing driver for the spin-aware ping-pong ball.
//
// The ball is a MOCAP body in the MJCF, so MuJoCo never integrates it (no double-integration, no
// keyframe-qpos change) and never collides it (collision-inert). BallPhysics OWNS the ball's
// (p, v, omega) in the world frame, advances it each timestep with the experimentally-fitted flight +
// spin-equation contact model (ball_dynamics), detects racket hits from the robot's racket site, predicts
// the outgoing-shot landing point at each hit, and writes the pose back to d_->mocap_pos for rendering /
// downstream publishing. Identical math to the Isaac training env and the Record reference.

#pragma once

#include <string>

#include "mujoco/mujoco.h"
#include "mujoco_sim_module/ball/ball_dynamics.h"

namespace aimrt_mujoco_sim::mujoco_sim_module::ball {

class BallPhysics {
 public:
  // Resolve the ball mocap body + racket site and load configs/ball_physics.yaml. Returns true (and
  // sets Enabled()) iff a mocap ball body named `ball_body_name` exists. Paddle-hit + landing prediction
  // are additionally gated on finding the racket site (logged if absent).
  bool Init(mjModel* m, mjData* d, const std::string& config_path);

  bool Enabled() const { return enabled_; }

  // Advance the ball one MuJoCo timestep. Call AFTER mj_step, inside the sim lock, so it reads the
  // freshly-stepped racket pose and owns the final ball pose for this frame.
  void Step();

  // The most recent outgoing-shot landing prediction (updated on each racket hit).
  const LandingResult& LastLanding() const { return last_landing_; }
  bool ConsumeNewLanding();  // true once per new hit; clears the flag.

  // --- scene/config knobs (defaults match the MJCF table + HOPE frame; tune to the real cell) -------
  std::string ball_body_name = "ball";
  std::string racket_site_name = "right_racket";
  int racket_normal_axis = 1;            // racket-local face-normal axis (0=x, 1=y, 2=z)
  double racket_contact_radius = 0.075;  // m

 private:
  void WriteMocap();

  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;
  bool enabled_ = false;

  BallPhysicsConfig cfg_;
  TablePlane table_;

  int ball_body_id_ = -1;
  int mocap_id_ = -1;
  int racket_site_id_ = -1;  // < 0 => paddle-hit disabled

  Vec3 p_ = {0.0, 0.0, 0.0};
  Vec3 v_ = {0.0, 0.0, 0.0};
  Vec3 omega_ = {0.0, 0.0, 0.0};
  bool paddle_latch_ = false;

  LandingResult last_landing_;
  bool has_new_landing_ = false;
};

}  // namespace aimrt_mujoco_sim::mujoco_sim_module::ball
