"""Table-tennis environment: mocap-calibrated, spin-aware ball physics on top of PhysX.

PhysX integrates gravity and prevents tunnelling, but it does NOT model (a) air drag / Magnus, nor (b)
the fitted spin-coupled bounce. This subclass supplies both via a **physics-step callback** that runs
every physics substep (400 Hz):

  1. **Aerodynamics** — quadratic drag + Magnus from the ball's spin (:func:`.ball.compute_aero_wrench`),
     injected as an external force.
  2. **Contacts (code-driven)** — when the ball crosses the table plane (descending) or meets the
     racket, the experimentally-fitted spin-equation impulse (:func:`.physics.predict_contact`) sets the
     post-contact velocity AND spin directly. PhysX restitution is neutralized for the ball
     (``geometry.BounceMaterials.ball_restitution = 0``) so it does not fight these writes; it still
     resolves penetration (anti-tunnelling, helped by CCD).
  3. **Landing prediction** — at the instant of a racket hit, forward-integrate the post-hit state with
     the SAME flight model (:func:`.physics.predict_landing`) to where the shot will land. Cached per env
     for the reward / observation managers.

All fitted constants come from the shared ``configs/ball_physics.yaml`` (:mod:`.physics.params`), so the
sim, the analytic predictor, and the MuJoCo C++ sim share one source of truth. If the callback cannot be
registered, the env still runs on PhysX gravity + (zero-restitution) contacts.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat, quat_rotate_inverse

from . import geometry
from .ball import compute_aero_wrench
from .physics import get_ball_physics, predict_contact, predict_landing
from .table_tennis_env_cfg import TableTennisEnvCfg

# Racket hit footprint: the A3 paddle is ~7.5 cm radius (hope_planner.constants.racket_radius). A hit is
# registered when the ball centre comes within (this + ball radius) of the racket body origin.
RACKET_CONTACT_RADIUS: float = 0.075
# Candidate racket body names on the A3 articulation (the red hitting face), most specific first.
_RACKET_BODY_CANDIDATES = ("pingpang_red_Link", "pingpang_black_Link", "right_racket", "pingbang_ball_Link")
_CONTACT_EPS: float = 1e-3  # latch hysteresis (m)


class TableTennisEnv(ManagerBasedRLEnv):
    """Manager-based table-tennis env with per-substep aerodynamics + spin-equation contacts."""

    cfg: TableTennisEnvCfg

    def __init__(self, cfg: TableTennisEnvCfg, render_mode: str | None = None, **kwargs):
        # Must exist before super().__init__ (which resets all envs and calls _reset_idx).
        self._physics_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        self._aero_active = False
        self._paddle_active = False
        self._setup_ball_physics()

    # -------------------------------------------------------------- step hook
    def step(self, action):
        """Run one env step, then consume the one-shot ``_hit_event``.

        The hit flag is raised in the physics callback (a racket contact during this step's substeps)
        and read by every per-hit term — rewards (landing, net clearance) and the privileged landing
        observation — while ``super().step`` computes them. Clearing it HERE, once, after the step
        returns means all those terms see the same flag for the step in which the hit happened; no single
        term may consume it and starve the others.
        """
        out = super().step(action)
        if getattr(self, "_physics_ready", False):
            self._hit_event[:] = False
        return out

    # ------------------------------------------------------------------ setup
    def _setup_ball_physics(self) -> None:
        self._ball = self.scene["ball"]
        self._aero_cfg = self.cfg.ball_aerodynamics

        # Shared, mocap-fitted constants (single source of truth).
        self._phys = get_ball_physics()
        self._ball_mass = float(self._phys.ball.mass)

        # Table plane + court rectangle in the HOPE frame (table surface at z = 0). Ball radius comes from
        # the shared YAML (single source of truth), consistent with the PhysX ball built from it.
        self._contact_z = float(self._phys.ball.radius)              # ball-centre height at contact
        self._table_x = (0.0, float(geometry.TABLE_LENGTH))
        self._table_y = (float(-geometry.TABLE_WIDTH), 0.0)
        self._net_x = float(geometry.NET_X)

        # Per-env state buffers (always created so obs/reward managers are valid even if aero is off).
        n, dev = self.num_envs, self.device
        self._aero_force = torch.zeros(n, 1, 3, device=dev)
        self._aero_torque = torch.zeros(n, 1, 3, device=dev)
        self._table_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._paddle_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hit_event = torch.zeros(n, dtype=torch.bool, device=dev)
        self._predicted_landing_xy = torch.zeros(n, 2, device=dev)
        self._predicted_landing_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._predicted_landing_in_opp = torch.zeros(n, dtype=torch.bool, device=dev)
        self._predicted_net_z = torch.zeros(n, device=dev)
        self._predicted_net_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._physics_ready = True

        if not self._aero_cfg.enabled:
            return

        # Locate the racket body on the robot (paddle-hit detection is optional — skip if absent).
        try:
            robot = self.scene["robot"]
            self._robot = robot
            self._racket_body_idx = next(
                robot.body_names.index(name) for name in _RACKET_BODY_CANDIDATES if name in robot.body_names
            )
            self._paddle_active = True
            import omni.log

            omni.log.info(
                "[TableTennisEnv] paddle-hit + landing/net prediction ACTIVE; racket body "
                f"'{robot.body_names[self._racket_body_idx]}' (idx {self._racket_body_idx}). "
                "If this line is absent, _paddle_active is False and the landing/pass_net rewards are zero."
            )
        except (KeyError, StopIteration):
            import omni.log

            omni.log.warn(
                "[TableTennisEnv] no racket body found on the robot "
                f"(looked for {_RACKET_BODY_CANDIDATES}); paddle-hit + landing prediction disabled."
            )

        try:
            self.sim.add_physics_callback("hope_ball_physics", self._apply_ball_physics)
            self._aero_active = True
        except Exception as exc:  # pragma: no cover - never block the sim on physics-callback setup
            import omni.log

            omni.log.warn(
                f"[TableTennisEnv] could not register the ball-physics callback ({exc!r}); "
                f"the ball will fly on PhysX gravity + (zero-restitution) contacts only."
            )

    # -------------------------------------------------------------- callback
    def _apply_ball_physics(self, dt: float) -> None:
        """Per-substep: aerodynamic force, then spin-equation table-bounce / paddle-hit, then landing."""
        lin_vel_w = self._ball.data.root_lin_vel_w
        ang_vel_w = self._ball.data.root_ang_vel_w
        quat_w = self._ball.data.root_quat_w
        pos_h = self._ball.data.root_pos_w - self.scene.env_origins  # HOPE-frame ball position

        # --- 1. Aerodynamics (drag + Magnus), world -> body frame. ----------
        force_w, torque_w = compute_aero_wrench(lin_vel_w, ang_vel_w, self._ball_mass, self._aero_cfg)
        self._aero_force[:, 0, :] = quat_rotate_inverse(quat_w, force_w)
        self._aero_torque[:, 0, :] = quat_rotate_inverse(quat_w, torque_w)
        self._ball.set_external_force_and_torque(self._aero_force, self._aero_torque)
        self._ball.write_data_to_sim()

        # --- 2a. Table bounce (static surface, n = +Z, v_r = 0). ------------
        x, y, z = pos_h[:, 0], pos_h[:, 1], pos_h[:, 2]
        in_xy = (x >= self._table_x[0]) & (x <= self._table_x[1]) & (y >= self._table_y[0]) & (y <= self._table_y[1])
        table_hit = (~self._table_latch) & in_xy & (z <= self._contact_z) & (lin_vel_w[:, 2] < 0.0)
        if torch.any(table_hit):
            n_up = torch.zeros_like(lin_vel_w)
            n_up[:, 2] = 1.0
            zero_vr = torch.zeros_like(lin_vel_w)
            v_plus, w_plus = predict_contact(lin_vel_w, zero_vr, n_up, ang_vel_w, self._phys.table)
            self._write_ball_velocity(table_hit, v_plus, w_plus)
        # Latch while at/below the surface; clears once the ball rises back above it.
        self._table_latch = z <= (self._contact_z + _CONTACT_EPS)

        # --- 2b. Paddle hit (moving face) + 3. landing prediction. ----------
        if self._paddle_active:
            self._handle_paddle(pos_h, lin_vel_w, ang_vel_w)

    def _handle_paddle(self, pos_h: torch.Tensor, lin_vel_w: torch.Tensor, ang_vel_w: torch.Tensor) -> None:
        rd = self._robot.data
        racket_pos_w = rd.body_pos_w[:, self._racket_body_idx, :]
        racket_quat_w = rd.body_quat_w[:, self._racket_body_idx, :]
        racket_vel_w = rd.body_lin_vel_w[:, self._racket_body_idx, :]
        # Paddle face normal = racket-local +Y (matches tracking RacketTargetCommand mount_normal_axis=1).
        face_normal = matrix_from_quat(racket_quat_w)[:, :, 1]

        ball_pos_w = self._ball.data.root_pos_w
        to_ball = ball_pos_w - racket_pos_w
        dist = torch.linalg.norm(to_ball, dim=-1)
        radius_sum = RACKET_CONTACT_RADIUS + float(geometry.BALL_RADIUS)
        # Approaching = ball moving toward the racket centre (sign-free; orient_normal handles the face).
        approaching = torch.sum((lin_vel_w - racket_vel_w) * to_ball, dim=-1) < 0.0
        hit = (~self._paddle_latch) & (dist < radius_sum) & approaching

        if torch.any(hit):
            v_plus, w_plus = predict_contact(lin_vel_w, racket_vel_w, face_normal, ang_vel_w, self._phys.paddle)
            self._write_ball_velocity(hit, v_plus, w_plus)

            ids = hit.nonzero(as_tuple=False).squeeze(-1)
            res = predict_landing(
                pos_h[ids], v_plus[ids], w_plus[ids], self._phys.flight,
                contact_z=self._contact_z, table_x=self._table_x, table_y=self._table_y, net_x=self._net_x,
            )
            self._predicted_landing_xy[ids] = res.xy
            self._predicted_landing_valid[ids] = res.valid
            self._predicted_landing_in_opp[ids] = res.on_opponent
            self._predicted_net_z[ids] = res.net_z
            self._predicted_net_valid[ids] = res.net_valid
            self._hit_event[ids] = True

        self._paddle_latch = dist < (radius_sum + _CONTACT_EPS)

    def _write_ball_velocity(self, mask: torch.Tensor, v_plus: torch.Tensor, w_plus: torch.Tensor) -> None:
        """Write post-contact [lin(3), ang(3)] velocity for the masked envs only."""
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        vel = torch.cat([v_plus[ids], w_plus[ids]], dim=-1)
        self._ball.write_root_velocity_to_sim(vel, env_ids=ids)

    # ----------------------------------------------------------------- reset
    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if not self._physics_ready:
            return
        self._table_latch[env_ids] = False
        self._paddle_latch[env_ids] = False
        self._hit_event[env_ids] = False
        self._predicted_landing_valid[env_ids] = False
        self._predicted_landing_in_opp[env_ids] = False
        self._predicted_landing_xy[env_ids] = 0.0
        self._predicted_net_valid[env_ids] = False
        self._predicted_net_z[env_ids] = 0.0
