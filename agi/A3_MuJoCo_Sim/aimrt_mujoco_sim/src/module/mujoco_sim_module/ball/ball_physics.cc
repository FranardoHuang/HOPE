// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.

#include "mujoco_sim_module/ball/ball_physics.h"

#include <exception>

namespace aimrt_mujoco_sim::mujoco_sim_module::ball {

bool BallPhysics::Init(mjModel* m, mjData* d, const std::string& config_path) {
  m_ = m;
  d_ = d;

  try {
    cfg_ = BallPhysicsConfig::LoadFromYaml(config_path);
  } catch (const std::exception&) {
    return false;  // bad/empty path: disable the ball rather than crash the sim (module logs).
  }

  ball_body_id_ = mj_name2id(m_, mjOBJ_BODY, ball_body_name.c_str());
  if (ball_body_id_ < 0) return false;
  mocap_id_ = m_->body_mocapid[ball_body_id_];
  if (mocap_id_ < 0) return false;  // ball must be a mocap body (so MuJoCo never integrates it).

  racket_site_id_ = mj_name2id(m_, mjOBJ_SITE, racket_site_name.c_str());  // < 0 => paddle disabled

  // Authoritative initial state = the body's spawn pose; at rest (a serve can set v_/omega_ later).
  p_ = {d_->mocap_pos[3 * mocap_id_ + 0], d_->mocap_pos[3 * mocap_id_ + 1], d_->mocap_pos[3 * mocap_id_ + 2]};
  v_ = {0.0, 0.0, 0.0};
  omega_ = {0.0, 0.0, 0.0};

  // table_ keeps its struct defaults (HOPE-frame ITTF table); MUST match the MJCF table geom.
  enabled_ = true;
  return true;
}

void BallPhysics::Step() {
  if (!enabled_) return;
  has_new_landing_ = false;
  const double dt = m_->opt.timestep;

  // --- Paddle hit (moving racket face) + outgoing-shot landing prediction. ---
  if (racket_site_id_ >= 0) {
    const mjtNum* sp = d_->site_xpos + 3 * racket_site_id_;
    const mjtNum* sm = d_->site_xmat + 9 * racket_site_id_;  // row-major 3x3 (world orientation)
    const Vec3 racket_pos = {sp[0], sp[1], sp[2]};
    // racket-local face-normal axis expressed in world = column `racket_normal_axis` of site_xmat.
    const int a = racket_normal_axis;
    const Vec3 face_normal = {sm[a], sm[3 + a], sm[6 + a]};

    mjtNum vel6[6];
    mj_objectVelocity(m_, d_, mjOBJ_SITE, racket_site_id_, vel6, /*flg_local=*/0);
    const Vec3 racket_vel = {vel6[3], vel6[4], vel6[5]};  // [rot(3), lin(3)] -> linear part

    const Vec3 to_ball = Sub(p_, racket_pos);
    const double dist = Norm(to_ball);
    const double radius_sum = racket_contact_radius + cfg_.ball.radius;
    const bool approaching = Dot(Sub(v_, racket_vel), to_ball) < 0.0;

    if (!paddle_latch_ && dist < radius_sum && approaching) {
      ContactResult r = PredictContact(v_, racket_vel, face_normal, omega_, cfg_.paddle);
      v_ = r.v_plus;
      omega_ = r.omega_plus;
      last_landing_ = PredictLanding(p_, v_, omega_, cfg_.flight, table_, cfg_.ball.radius);
      has_new_landing_ = true;
    }
    paddle_latch_ = dist < (radius_sum + 1e-3);
  }

  // --- Flight + table bounces over one timestep. ---
  Simulate(p_, v_, omega_, dt, cfg_.flight, cfg_.table, table_, cfg_.ball.radius);

  WriteMocap();
}

void BallPhysics::WriteMocap() {
  d_->mocap_pos[3 * mocap_id_ + 0] = p_[0];
  d_->mocap_pos[3 * mocap_id_ + 1] = p_[1];
  d_->mocap_pos[3 * mocap_id_ + 2] = p_[2];
  // Identity orientation (we track spin internally; the visual sphere is rotation-agnostic).
  d_->mocap_quat[4 * mocap_id_ + 0] = 1.0;
  d_->mocap_quat[4 * mocap_id_ + 1] = 0.0;
  d_->mocap_quat[4 * mocap_id_ + 2] = 0.0;
  d_->mocap_quat[4 * mocap_id_ + 3] = 0.0;
}

bool BallPhysics::ConsumeNewLanding() {
  const bool v = has_new_landing_;
  has_new_landing_ = false;
  return v;
}

}  // namespace aimrt_mujoco_sim::mujoco_sim_module::ball
