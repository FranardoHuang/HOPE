// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
//
// Golden-value regression for the ball-dynamics kernel. The expected numbers were produced by the
// fitted Record reference (Record/analysis/{contact_model/spin_equation,flight_model/simulator}.py) for
// the SAME inputs and cross-checked to <1e-6 (see the repo's Python test test_ball_physics_vs_record.py
// and the C++ parity check). This guards the C++ port against drift without needing the Record folder.

#include "mujoco_sim_module/ball/ball_dynamics.h"

#include <gtest/gtest.h>

namespace aimrt_mujoco_sim::mujoco_sim_module::ball {

namespace {
constexpr double kTol = 1e-6;
}

TEST(BallDynamics, TableBounceMatchesOracle) {
  BallPhysicsConfig cfg;  // defaults == configs/ball_physics.yaml
  Vec3 vm = {3.0, -1.0, -4.0}, vr = {0.0, 0.0, 0.0}, n = {0.0, 0.0, 1.0}, w = {10.0, -5.0, 2.0};
  ContactResult r = PredictContact(vm, vr, n, w, cfg.table);
  // GRIPPY table (e_n=0.908, grip k=0.369): tangential speed partly kept + converted to spin; the
  // spin about the normal (omega_plus[2]) is preserved. Re-baselined 2026-06-30 (OptiTrack recal).
  EXPECT_NEAR(r.v_plus[0], 1.856100000000, kTol);
  EXPECT_NEAR(r.v_plus[1], -0.704800000000, kTol);
  EXPECT_NEAR(r.v_plus[2], 3.632000000000, kTol);  // 0.908 * 4.0 normal restitution
  EXPECT_NEAR(r.omega_plus[0], 32.140000000000, kTol);
  EXPECT_NEAR(r.omega_plus[1], 80.792500000000, kTol);
  EXPECT_NEAR(r.omega_plus[2], 2.000000000000, kTol);
}

TEST(BallDynamics, PaddleHitMatchesOracle) {
  BallPhysicsConfig cfg;
  Vec3 vm = {-5.0, 0.5, 1.0}, vr = {2.0, 0.0, 1.0}, n = {0.0, 1.0, 0.0}, w = {0.0, 30.0, 0.0};
  ContactResult r = PredictContact(vm, vr, n, w, cfg.paddle);
  EXPECT_NEAR(r.v_plus[0], -3.171004001760, kTol);
  EXPECT_NEAR(r.v_plus[1], -0.231598399295, kTol);
  EXPECT_NEAR(r.v_plus[2], 1.000000000000, kTol);
  EXPECT_NEAR(r.omega_plus[2], -137.174699867828, kTol);
}

TEST(BallDynamics, NormalRestitutionIsEN) {
  // A pure vertical drop must rebound at exactly e_n = 0.908.
  BallPhysicsConfig cfg;
  Vec3 vm = {0.0, 0.0, -5.0}, vr = {0.0, 0.0, 0.0}, n = {0.0, 0.0, 1.0}, w = {0.0, 0.0, 0.0};
  ContactResult r = PredictContact(vm, vr, n, w, cfg.table);
  EXPECT_NEAR(r.v_plus[2], 0.908 * 5.0, kTol);
  EXPECT_NEAR(r.v_plus[0], 0.0, kTol);
  EXPECT_NEAR(r.v_plus[1], 0.0, kTol);
}

TEST(BallDynamics, SimulateBouncesOnlyWithinTable) {
  BallPhysicsConfig cfg;
  TablePlane t;  // HOPE-frame ITTF table: surface 0.76, x in [0,2.74], y in [-1.525,0]
  const double cz = t.surface_z + cfg.ball.radius;

  // Drop straight down INSIDE the table footprint -> bounces, rebounds above the surface.
  Vec3 p_in = {1.0, -0.7625, t.surface_z + 1.0}, v_in = {0, 0, 0}, w_in = {0, 0, 0};
  int nb_in = Simulate(p_in, v_in, w_in, 0.6, cfg.flight, cfg.table, t, cfg.ball.radius);
  EXPECT_EQ(nb_in, 1);
  EXPECT_GT(p_in[2], cz);

  // Drop straight down BEYOND the far edge (x > x_max) -> no bounce, falls through the plane height.
  Vec3 p_out = {3.5, -0.7625, t.surface_z + 1.0}, v_out = {0, 0, 0}, w_out = {0, 0, 0};
  int nb_out = Simulate(p_out, v_out, w_out, 0.6, cfg.flight, cfg.table, t, cfg.ball.radius);
  EXPECT_EQ(nb_out, 0);
  EXPECT_LT(p_out[2], cz);
}

TEST(BallDynamics, LandingMatchesOracle) {
  BallPhysicsConfig cfg;
  Vec3 p = {0.0, 0.0, 0.9}, v = {6.0, 0.5, 0.5}, w = {0.0, 40.0, 0.0};
  TablePlane table;
  table.surface_z = 0.0;
  table.x_min = -5.0; table.x_max = 5.0; table.y_min = -5.0; table.y_max = 5.0; table.net_x = -100.0;
  LandingResult r = PredictLanding(p, v, w, cfg.flight, table, cfg.ball.radius, 2.0, 1.0e-3);
  EXPECT_TRUE(r.valid);
  // Re-baselined 2026-06-30 for the OptiTrack flight recal (k_m 0.006 -> 0.0042; lower Magnus -> the
  // topspun ball carries farther). Pure flight, so unaffected by the grippy-table change.
  EXPECT_NEAR(r.t_flight, 0.473477, 1e-4);
  EXPECT_NEAR(r.point[0], 2.404731, 1e-4);
  EXPECT_NEAR(r.point[1], 0.202032, 1e-4);
  EXPECT_NEAR(r.point[2], cfg.ball.radius, 1e-4);
}

}  // namespace aimrt_mujoco_sim::mujoco_sim_module::ball
