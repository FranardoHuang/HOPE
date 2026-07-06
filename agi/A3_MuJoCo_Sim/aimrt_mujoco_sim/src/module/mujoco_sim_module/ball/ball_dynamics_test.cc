// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
//
// Golden-value regression for the ball-dynamics kernel. The expected numbers were produced by the
// fitted Record reference (Record/analysis/{contact_model/spin_equation,flight_model/simulator}.py)
// driven with the VENUE-FIT constants (configs/ball_physics_venue.yaml, 2026-07-03: e_n=0.9215,
// paddle e(u_n)=0.759*exp(-0.0441*u_n) / a_t=0.52 / mu=0.5, k_d=0.1261, k_m=0.00444) for the SAME
// inputs and cross-checked to <1e-6 (see the repo's Python test test_ball_physics_vs_record.py and
// the C++ parity check). This guards the C++ port against drift without needing the Record folder.

#include "mujoco_sim_module/ball/ball_dynamics.h"

#include <gtest/gtest.h>

namespace aimrt_mujoco_sim::mujoco_sim_module::ball {

namespace {
constexpr double kTol = 1e-6;
}

TEST(BallDynamics, TableBounceMatchesOracle) {
  BallPhysicsConfig cfg;  // defaults == configs/ball_physics_venue.yaml
  Vec3 vm = {3.0, -1.0, -4.0}, vr = {0.0, 0.0, 0.0}, n = {0.0, 0.0, 1.0}, w = {10.0, -5.0, 2.0};
  ContactResult r = PredictContact(vm, vr, n, w, cfg.table);
  // GRIPPY table (e_n=0.9215, grip k=0.369): tangential speed partly kept + converted to spin; the
  // spin about the normal (omega_plus[2]) is preserved. Re-baselined 2026-07-05 (venue fit).
  EXPECT_NEAR(r.v_plus[0], 1.856100000000, kTol);
  EXPECT_NEAR(r.v_plus[1], -0.704800000000, kTol);
  EXPECT_NEAR(r.v_plus[2], 3.686000000000, kTol);  // 0.9215 * 4.0 normal restitution
  EXPECT_NEAR(r.omega_plus[0], 32.140000000000, kTol);
  EXPECT_NEAR(r.omega_plus[1], 80.792500000000, kTol);
  EXPECT_NEAR(r.omega_plus[2], 2.000000000000, kTol);
}

TEST(BallDynamics, PaddleHitMatchesOracle) {
  BallPhysicsConfig cfg;
  Vec3 vm = {-5.0, 0.5, 1.0}, vr = {2.0, 0.0, 1.0}, n = {0.0, 1.0, 0.0}, w = {0.0, 30.0, 0.0};
  ContactResult r = PredictContact(vm, vr, n, w, cfg.paddle);
  // Venue paddle: F4 e(u_n)=0.759*exp(-0.0441*u_n) -> e=0.742447 at this contact's u_n;
  // a_t=0.52 (velocity-channel fit) -> far less spin transfer than the v0.1 placeholder.
  EXPECT_NEAR(r.v_plus[0], -4.564388196222, kTol);
  EXPECT_NEAR(r.v_plus[1], -0.371223607556, kTol);
  EXPECT_NEAR(r.v_plus[2], 1.000000000000, kTol);
  EXPECT_NEAR(r.omega_plus[2], -32.670885283342, kTol);
}

TEST(BallDynamics, NormalRestitutionIsEN) {
  // A pure vertical drop must rebound at exactly e_n = 0.9215.
  BallPhysicsConfig cfg;
  Vec3 vm = {0.0, 0.0, -5.0}, vr = {0.0, 0.0, 0.0}, n = {0.0, 0.0, 1.0}, w = {0.0, 0.0, 0.0};
  ContactResult r = PredictContact(vm, vr, n, w, cfg.table);
  EXPECT_NEAR(r.v_plus[2], 0.9215 * 5.0, kTol);
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
  // Re-baselined 2026-07-05 for the venue flight fit (k_d 0.1222 -> 0.1261, k_m 0.0042 -> 0.00444).
  // Pure flight, so unaffected by the contact-parameter changes.
  EXPECT_NEAR(r.t_flight, 0.472795, 1e-4);
  EXPECT_NEAR(r.point[0], 2.389817, 1e-4);
  EXPECT_NEAR(r.point[1], 0.200879, 1e-4);
  EXPECT_NEAR(r.point[2], cfg.ball.radius, 1e-4);
}

}  // namespace aimrt_mujoco_sim::mujoco_sim_module::ball
