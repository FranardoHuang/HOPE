// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
//
// Experimentally-calibrated, spin-aware ping-pong ball physics (flight + contact + landing).
//
// C++ port of the fitted Record reference (Python/numpy):
//   * flight  : Record/analysis/flight_model/simulator.py   (flight_accel + rk4 + bounce state machine)
//   * contact : Record/analysis/contact_model/spin_equation.py (predict_contact impulse model)
// The constants come from configs/ball_physics_venue.yaml (the SAME file the Isaac training env reads), so the
// two simulators and the offline reference share one source of truth.
//
// This header is MuJoCo-free and dependency-light (only <array>/<vector>/<string>): the core math
// (PredictContact / FlightAccel / Rk4Step / PredictLanding) has no yaml-cpp dependency and is
// unit-testable standalone. Only BallPhysicsConfig::LoadFromYaml pulls in yaml-cpp.
//
// SI units throughout: m, m/s, rad/s, s, kg.

#pragma once

#include <array>
#include <string>

namespace aimrt_mujoco_sim::mujoco_sim_module::ball {

using Vec3 = std::array<double, 3>;

// --- small vector helpers (header-inline so tests and the component share them) ----------------
inline Vec3 Add(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 Sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 Scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double Dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double Norm(const Vec3& a);
Vec3 Normalize(const Vec3& a, double eps = 1e-12);

// --- parameter structs -------------------------------------------------------------------------
struct BallParams {
  double mass = 0.0034;              // kg — the coated 3.4 g MATCH ball (venue fit 2026-07-03)
  double radius = 0.020;
  double inertia_coeff = 2.0 / 3.0;  // hollow sphere: I = c m R^2
};

struct FlightParams {
  double k_d = 0.1261;    // 1/m quadratic-drag accel coefficient (venue fit, coated ball)
  double k_m = 0.00444;   // Magnus coefficient for PHYSICAL spin (venue sidespin channel). YAML overrides.
  double g = 9.81;        // m/s^2 gravity magnitude (acts along -Z)
  double rk4_h = 5.0e-4;  // s reference RK4 step
  double ball_radius = 0.020;  // R, for the spin ratio SR = R|omega|/|v|
  // Venue F2 saturating lift law, expressed relative to its own low-SR linearization:
  // k_m_eff(SR) = k_m / (1 + SR / magnus_sr_sat), with magnus_sr_sat = cl_max / cl_slope.
  // Identical to constant k_m at low SR; only bends where the fit has no coverage.
  // Behavior switch defaults OFF (oracle parity); flip use_saturating_magnus per run.
  double magnus_sr_sat = 0.55 / 0.87;
  bool use_saturating_magnus = false;
};

struct ContactParams {
  // Defaults == configs/ball_physics_venue.yaml `contact.table` (venue fit 2026-07-03): e_n=0.9215
  // (forensics-corrected; was 0.908 on the old table), grip coeff a_t=k=0.369 (retained v0
  // OptiTrack fit, venue refit degenerate), b_t=0, Coulomb cap disabled (mu_safety high).
  // Paddle params are set explicitly in BallPhysicsConfig.
  double e_eff = 0.9215;
  double a_t = 0.369;
  double b_t = 0.0;
  double mu_safety = 2.0;
  double ball_radius = 0.020;       // R
  double inertia_coeff = 2.0 / 3.0; // c
  // Venue F4 (paddle only): velocity-dependent normal restitution
  // e(u_n) = e_exp_g1 * exp(e_exp_g2 * |u_n|), clamped to [0.05, 0.95] in PredictContact;
  // e_eff stays as the documented constant fallback when use_e_exp is false.
  double e_exp_g1 = 0.0;
  double e_exp_g2 = 0.0;
  bool use_e_exp = false;
};

// Table plane + court rectangle (world frame). Normal assumed ~+Z for this sim.
struct TablePlane {
  double surface_z = 0.76;  // table-top height (world Z)
  double x_min = 0.0, x_max = 2.74;
  double y_min = -1.525, y_max = 0.0;
  double net_x = 1.37;      // opponent half = x > net_x
};

struct BallPhysicsConfig {
  BallParams ball;
  double air_rho = 1.20;
  FlightParams flight;
  // Defaults equal configs/ball_physics_venue.yaml (table = the ContactParams default; paddle set
  // explicitly here so a default-constructed config is fully correct even before LoadFromYaml).
  // Paddle: FIRST REAL RACKET FIT — e_eff 0.654 constant fallback, F4 e(u_n) enabled by default,
  // a_t 0.52 velocity-channel fit, b_t 0, mu_safety 0.5 (cap never binds on venue data).
  ContactParams table;
  ContactParams paddle{0.654, 0.52, 0.0, 0.5, 0.020, 2.0 / 3.0, 0.759, -0.0441, true};

  // Parse configs/ball_physics_venue.yaml (yaml-cpp). Folds ball.radius / ball.inertia_coeff into
  // both ContactParams; reads the paddle e_exp_g1/g2 (-> use_e_exp) and flight cl_slope/cl_max
  // (-> magnus_sr_sat) keys when present. Throws std::runtime_error on a missing/unreadable file
  // or missing keys.
  static BallPhysicsConfig LoadFromYaml(const std::string& path);
};

// --- contact model (port of spin_equation.predict_contact) -------------------------------------
struct ContactResult {
  Vec3 v_plus;
  Vec3 omega_plus;
};

// v_r = contact-point velocity (0 for the static table), n = face normal (auto-oriented).
ContactResult PredictContact(const Vec3& v_minus, const Vec3& v_r, const Vec3& n,
                             const Vec3& omega_minus, const ContactParams& p);

// --- flight model (port of simulator.flight_accel / rk4) ---------------------------------------
Vec3 FlightAccel(const Vec3& v, const Vec3& omega, const FlightParams& f);
void Rk4Step(const Vec3& p, const Vec3& v, const Vec3& omega, double h, const FlightParams& f,
             Vec3& p_out, Vec3& v_out);

// --- landing prediction (flight half of simulator.simulate's first bounce) ---------------------
struct LandingResult {
  bool valid = false;        // a descending table-plane crossing was found within the horizon
  bool in_bounds = false;    // crossing falls inside the table rectangle
  bool on_opponent = false;  // in_bounds AND on the opponent half (x > net_x)
  double t_flight = 0.0;     // time from the hit to the landing (s)
  Vec3 point = {0.0, 0.0, 0.0};
};

// Forward-integrate (p, v, omega) to the first descending crossing of z = table.surface_z + ball_radius.
LandingResult PredictLanding(const Vec3& p, const Vec3& v, const Vec3& omega, const FlightParams& f,
                             const TablePlane& table, double ball_radius, double max_time = 2.0,
                             double dt = 1.0e-3, int bisect_iters = 24);

// Advance (p, v, omega) IN PLACE by `duration` seconds with flight + table bounces (RK4 substeps at
// f.rk4_h; bounces resolved at their true sub-step time via bisection). Port of simulator.simulate; used
// to drive the live in-sim ball one MuJoCo timestep at a time. Returns the number of table bounces.
int Simulate(Vec3& p, Vec3& v, Vec3& omega, double duration, const FlightParams& f,
             const ContactParams& table_contact, const TablePlane& table, double ball_radius,
             int bisect_iters = 24);

}  // namespace aimrt_mujoco_sim::mujoco_sim_module::ball
