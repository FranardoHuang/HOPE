// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.

#include "mujoco_sim_module/ball/ball_dynamics.h"

#include <algorithm>
#include <cmath>

namespace aimrt_mujoco_sim::mujoco_sim_module::ball {

namespace {
constexpr double kEps = 1e-12;
}  // namespace

double Norm(const Vec3& a) { return std::sqrt(Dot(a, a)); }

Vec3 Normalize(const Vec3& a, double eps) { return Scale(a, 1.0 / (Norm(a) + eps)); }

// NOTE: BallPhysicsConfig::LoadFromYaml lives in ball_config_yaml.cc (the only TU that needs yaml-cpp),
// so this math kernel stays dependency-light and unit-testable standalone.

// ---------------------------------------------------------------- contact model
ContactResult PredictContact(const Vec3& v_minus, const Vec3& v_r, const Vec3& n_in,
                             const Vec3& omega_minus, const ContactParams& p) {
  const double R = p.ball_radius;
  const double c = p.inertia_coeff;

  // Orient n so the incoming relative velocity approaches the face.
  Vec3 n = Normalize(n_in);
  if (Dot(Sub(v_minus, v_r), n) > 0.0) n = Scale(n, -1.0);

  const Vec3 r = Scale(n, -R);
  const Vec3 u = Sub(Add(v_minus, Cross(omega_minus, r)), v_r);
  const double u_n_signed = Dot(u, n);
  const Vec3 u_t_vec = Sub(u, Scale(n, u_n_signed));
  const double u_t_mag = Norm(u_t_vec);
  const double u_n_abs = std::abs(u_n_signed);

  // Venue F4: velocity-dependent normal restitution e(u_n) = g1 * exp(g2 * |u_n|),
  // clamped so far-out-of-envelope contacts stay physical (paddle only; table constant-e).
  const double e_eff =
      p.use_e_exp ? std::clamp(p.e_exp_g1 * std::exp(p.e_exp_g2 * u_n_abs), 0.05, 0.95) : p.e_eff;

  const double cos_theta = u_n_abs / (std::hypot(u_t_mag, u_n_signed) + kEps);
  const double raw = (p.a_t + p.b_t * cos_theta) * u_t_mag;
  const double cap = p.mu_safety * (1.0 + e_eff) * u_n_abs;
  const double s = std::clamp(raw, 0.0, cap);

  Vec3 delta_v_t = {0.0, 0.0, 0.0};
  if (u_t_mag > kEps) delta_v_t = Scale(u_t_vec, -s / u_t_mag);

  const Vec3 delta_v_n = Scale(n, -(1.0 + e_eff) * u_n_signed);
  const Vec3 delta_omega = Scale(Cross(n, delta_v_t), -(1.0 / (c * R)));

  ContactResult out;
  out.v_plus = Add(Add(v_minus, delta_v_n), delta_v_t);
  out.omega_plus = Add(omega_minus, delta_omega);
  return out;
}

// ---------------------------------------------------------------- flight model
Vec3 FlightAccel(const Vec3& v, const Vec3& omega, const FlightParams& f) {
  const double sp = Norm(v);
  Vec3 a = {0.0, 0.0, -f.g};
  a = Sub(a, Scale(v, f.k_d * sp));
  if (f.k_m != 0.0) {
    double k_m_eff = f.k_m;
    // Venue F2 saturating lift: k_m_eff(SR) = k_m / (1 + SR/sr_sat), SR = R|omega|/|v|.
    // Off by default (oracle parity); see FlightParams.
    if (f.use_saturating_magnus && f.magnus_sr_sat > 0.0) {
      const double sr = f.ball_radius * Norm(omega) / (sp + kEps);
      k_m_eff /= 1.0 + sr / f.magnus_sr_sat;
    }
    a = Add(a, Scale(Cross(omega, v), k_m_eff));
  }
  return a;
}

void Rk4Step(const Vec3& p, const Vec3& v, const Vec3& omega, double h, const FlightParams& f,
             Vec3& p_out, Vec3& v_out) {
  const Vec3 a1 = FlightAccel(v, omega, f);
  const Vec3 a2 = FlightAccel(Add(v, Scale(a1, 0.5 * h)), omega, f);
  const Vec3 a3 = FlightAccel(Add(v, Scale(a2, 0.5 * h)), omega, f);
  const Vec3 a4 = FlightAccel(Add(v, Scale(a3, h)), omega, f);

  // v_new = v + h/6 (a1 + 2 a2 + 2 a3 + a4)
  Vec3 v_sum = Add(Add(a1, Scale(a2, 2.0)), Add(Scale(a3, 2.0), a4));
  v_out = Add(v, Scale(v_sum, h / 6.0));

  // p_new = p + h/6 (v + 2(v + h/2 a1) + 2(v + h/2 a2) + (v + h a3))   [matches simulator.rk4]
  Vec3 k1 = v;
  Vec3 k2 = Add(v, Scale(a1, 0.5 * h));
  Vec3 k3 = Add(v, Scale(a2, 0.5 * h));
  Vec3 k4 = Add(v, Scale(a3, h));
  Vec3 p_sum = Add(Add(k1, Scale(k2, 2.0)), Add(Scale(k3, 2.0), k4));
  p_out = Add(p, Scale(p_sum, h / 6.0));
}

// ---------------------------------------------------------------- landing prediction
LandingResult PredictLanding(const Vec3& p_in, const Vec3& v_in, const Vec3& omega,
                             const FlightParams& f, const TablePlane& table, double ball_radius,
                             double max_time, double dt, int bisect_iters) {
  const double contact_z = table.surface_z + ball_radius;
  LandingResult res;

  Vec3 p = p_in;
  Vec3 v = v_in;
  double t = 0.0;
  const int n_steps = std::max(1, static_cast<int>(std::lround(max_time / dt)));

  for (int i = 0; i < n_steps; ++i) {
    Vec3 p_next, v_next;
    Rk4Step(p, v, omega, dt, f, p_next, v_next);

    const bool crossing = (p[2] > contact_z) && (p_next[2] <= contact_z) && (v[2] < 0.0);
    if (crossing) {
      // Bisection on f in [0, dt] so z(f) == contact_z (descending).
      double lo = 0.0, hi = dt;
      for (int b = 0; b < bisect_iters; ++b) {
        const double mid = 0.5 * (lo + hi);
        Vec3 pm, vm;
        Rk4Step(p, v, omega, mid, f, pm, vm);
        if (pm[2] > contact_z)
          lo = mid;
        else
          hi = mid;
      }
      Vec3 p_cross, v_cross;
      Rk4Step(p, v, omega, 0.5 * (lo + hi), f, p_cross, v_cross);

      res.valid = true;
      res.t_flight = t + 0.5 * (lo + hi);
      res.point = p_cross;
      res.in_bounds = (p_cross[0] >= table.x_min) && (p_cross[0] <= table.x_max) &&
                      (p_cross[1] >= table.y_min) && (p_cross[1] <= table.y_max);
      res.on_opponent = res.in_bounds && (p_cross[0] > table.net_x);
      return res;
    }

    p = p_next;
    v = v_next;
    t += dt;
  }
  return res;  // valid = false: no crossing within the horizon
}

int Simulate(Vec3& p, Vec3& v, Vec3& omega, double duration, const FlightParams& f,
             const ContactParams& table_contact, const TablePlane& table, double ball_radius,
             int bisect_iters) {
  const double contact_z = table.surface_z + ball_radius;
  const double h = f.rk4_h;
  const Vec3 n_up = {0.0, 0.0, 1.0};
  int n_bounces = 0;
  double t = 0.0;

  while (t < duration - 1e-12) {
    const double step = std::min(h, duration - t);
    Vec3 p2, v2;
    Rk4Step(p, v, omega, step, f, p2, v2);

    const bool crossing = (p[2] > contact_z) && (p2[2] <= contact_z) && (v[2] < 0.0);
    bool bounced = false;
    if (crossing) {
      double lo = 0.0, hi = step;
      for (int b = 0; b < bisect_iters; ++b) {
        const double mid = 0.5 * (lo + hi);
        Vec3 pm, vm;
        Rk4Step(p, v, omega, mid, f, pm, vm);
        if (pm[2] > contact_z)
          lo = mid;
        else
          hi = mid;
      }
      const double fc = 0.5 * (lo + hi);
      Vec3 pc, vc;
      Rk4Step(p, v, omega, fc, f, pc, vc);
      // Only bounce if the crossing falls inside the table rectangle (mirrors the Isaac in_xy guard);
      // a ball that crosses the table plane beyond the edge has missed and keeps falling.
      const bool in_bounds = (pc[0] >= table.x_min) && (pc[0] <= table.x_max) &&
                             (pc[1] >= table.y_min) && (pc[1] <= table.y_max);
      if (in_bounds) {
        ContactResult cr = PredictContact(vc, {0.0, 0.0, 0.0}, n_up, omega, table_contact);
        p = pc;
        v = cr.v_plus;
        omega = cr.omega_plus;
        t += fc;
        ++n_bounces;
        bounced = true;
      }
    }
    if (!bounced) {
      p = p2;
      v = v2;
      t += step;
    }
  }
  return n_bounces;
}

}  // namespace aimrt_mujoco_sim::mujoco_sim_module::ball
