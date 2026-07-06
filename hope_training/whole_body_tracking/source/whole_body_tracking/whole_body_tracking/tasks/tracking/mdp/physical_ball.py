"""PHYSICAL ball + table for the tracking task — Phase A TRUTH INSTRUMENT (metrics only).

WHAT THIS IS: a flag-gated (``RacketTargetCommandCfg.physical_ball`` /
``HOPEPingPongAgibotA3EnvCfg.physical_ball`` / task-yaml top-level ``physical_ball``) real PhysX
rigid-body ball, one per env, plus a real static PhysX table collider, that realizes each swing's
question-bank incoming ball PHYSICALLY:

* SERVE — when the per-swing question is known (bank resample) and ``time_to_strike`` enters the
  serve horizon, the ball is launched from the venue-model REVERSE-TIME integrated state
  (:func:`back_integrate_incoming`: RK4 of the fitted flight law with a negative step, TRUNCATED
  at the table plane — the FINAL BALLISTIC SEGMENT only), ``tts_effective`` seconds before the
  strike, so that forward flight arrives at the question's contact point with the question's
  incoming velocity exactly at the exact-strike frame.
* FLIGHT — PhysX gravity + the per-physics-substep venue aero wrench (drag + Magnus,
  ``F = m(-k_d|v|v + k_m omega x v)``, world->body rotated because Isaac Lab 2.1
  ``set_external_force_and_torque`` applies wrenches in the BODY frame — the
  table_tennis_env.py / shadow_ball.py mechanism, re-used verbatim).
* TABLE BOUNCE — CODE-DRIVEN: a descending crossing of ``surface_z + R`` inside the table
  footprint triggers :func:`predict_table_contact` (the fitted angle-dependent tangential-impulse
  contact with the VENUE TABLE params ``contact.table`` of configs/ball_physics_venue.yaml:
  constant e_eff, v_r = 0, n = +z). PhysX NEVER resolves the ball's contacts (see below), so the
  fitted model is the single bounce authority — no restitution double-count.
* ROBOT PASS-THROUGH — the ball's collider is DISABLED (``collision_enabled=False``), which
  filters ball<->robot AND ball<->table PhysX contacts in one switch. This is deliberate:
  (a) Phase A has NO in-engine racket impulse (the fitted racket contact stays analytic in the
  reward path; porting it into the engine is PHASE B, out of scope here), so a robot collision
  would be an unfitted PhysX artifact; (b) the table bounce must be code-authoritative anyway
  (PhysX restitution cannot represent the fitted spin-dependent contact). The static table
  collider is still spawned (Phase-B ready, and the scene is honest about where the table is).

PHYSICS BASIS: scripts/isaac_ball_inloop_check.py validated exactly this injection pattern
(batched single view, per-substep body-frame venue aero wrench) with an in-loop result of PhysX
flight matching the venue RK4 reference to a 17 mm SYSTEMATIC landing offset — that number is the
expected floor for ``pb_serve_err_m`` here (reverse-RK4 launch -> forward PhysX-Euler flight).

WHAT THIS IS NOT: a TRUTH INSTRUMENT only. Reward and observation streams are COMPLETELY
untouched even when the flag is on — no reward term, no obs term, no bank-target logic reads any
of this; the analytic virtual ball (:mod:`virtual_ball`) remains the reward machine. The value is
per-strike ground truth: ``pb_serve_err_m`` / ``pb_serve_vel_err`` measure how exactly the engine
delivers the question's (contact point, incoming velocity) at the strike frame, and the
post-strike flight/bounce/landing metrics record what the real ball did.

HONESTY NOTES (read before trusting the numbers):

* Phase A serves ONLY the FINAL BALLISTIC SEGMENT (post-last-bounce): the reverse integration
  TRUNCATES at the table plane (last state strictly above ``surface_z + R + SERVE_PLANE_MARGIN``)
  and returns the per-env ``tts_effective`` it actually covered. Questions whose real history
  includes the incoming table bounce (rising contact velocities — ~11% of the bank — and moderate
  tts generally) launch LATER, ``tts_effective`` before the strike, from ON the incoming
  trajectory, so forward flight for ``tts_effective`` still arrives exactly at the question
  (contact, velocity) — the arrival guarantee the instrument needs. The pre-bounce segment is
  OUT OF SCOPE until the bounce-aware serve (bounce-map inversion — future work). This fix is
  the seed=1 pod-defect root cause: un-truncated back-integration put rising-contact launches
  under/inside the table (pb_serve_err_m 0.58 m); seed=0 had merely been lucky with questions.
* The strike itself applies NO impulse (Phase B): the ball flies THROUGH the strike point and the
  robot, descends behind it, and its first descending ``surface+R`` crossing is recorded as the
  landing (same plane convention as virtual_ball.coarse_landing / the shadow ball).
* The pre-strike arc cannot trigger the table bounce by construction (the bounce fires only on
  DESCENDING in-bounds crossings; the inbound arc's minimum over the table is the contact point
  itself, which sits above ``surface+R``). If an out-of-envelope question ever does descend
  through the plane in-bounds pre-strike, the bounce is applied anyway (physical consistency) and
  ``pb_serve_err_m`` reports the damage honestly.

This module is importable WITHOUT Isaac (torch-only at top level; sibling modules are loaded by
file path when the package import is unavailable). Pure helpers are unit-tested Isaac-free in
``tests/test_physical_ball_helpers.py``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------------------------- #
# Sibling modules: package import in the training env; file-path load for Isaac-free tests
# (the mdp package __init__ pulls isaaclab, so standalone loading cannot go through it).
# --------------------------------------------------------------------------------------------- #
try:  # pragma: no cover - trivial import plumbing
    from whole_body_tracking.tasks.tracking.mdp import shadow_ball as _sb
    from whole_body_tracking.tasks.tracking.mdp import virtual_ball as _vb
except Exception:  # standalone (tests / scripts without isaaclab on the path)

    def _load_sibling(fname: str, name: str):
        import importlib.util
        import sys

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclass resolution needs the module registered during exec
        spec.loader.exec_module(mod)
        return mod

    _sb = _load_sibling("shadow_ball.py", "physical_ball._shadow_ball")
    _vb = _load_sibling("virtual_ball.py", "physical_ball._virtual_ball")

# Serve horizon (s): the ball launches when time_to_strike first drops to/below this. Bounded for
# (a) the same env-footprint reason as shadow_ball.PRESTRIKE_HORIZON_S, (b) the incoming-bounce
# honesty window (module docstring), and (c) reverse-time drag blowup: integrating quadratic drag
# BACKWARD anti-amplifies speed with a finite-time singularity, measured at t* ~ 1.3 s for venue
# contact states (speed ~8-10 m/s at 0.6 s, ~22-25 m/s at 1.0 s, divergent by ~1.3 s) — 0.6 s
# keeps a >2x margin below both the blowup and BACKINT_SPEED_CAP.
SERVE_HORIZON_S = 0.6
# Reverse-integration speed cap (m/s): rows whose backward speed WOULD exceed this stop stepping
# (the crossing step is rejected), so the helper stays finite for any requested tts (up to and
# beyond 1.5 s) instead of hitting the reverse-drag singularity. The cap NEVER engages within
# the venue velocity envelope for t_back <= ~1.0 s (backward speeds reach ~22-32 m/s there —
# tested); capped rows report the shorter ``tts_effective`` they actually integrated, and the
# roundtrip guarantee holds over that span like any truncated row.
BACKINT_SPEED_CAP = 40.0
# Table-plane truncation margin (m): the reverse integration STOPS at the last state strictly
# above z = surface_z + ball_radius + SERVE_PLANE_MARGIN. Root cause of the seed=1 pod defect
# (pb_serve_err_m = 0.58 m / pb_serve_vel_err = 1.19): for rising-contact questions (vz >= 0,
# ~11% of the bank) and moderate tts the pure backward path dips below the table plane — in
# reality that segment is PRE-BOUNCE — so the launch state sat under/inside the table and the
# serve was garbage. Phase A serves only the FINAL ballistic segment; the pre-bounce segment is
# the future bounce-aware serve (module docstring).
SERVE_PLANE_MARGIN = 5e-3
# Reverse-integration step used by the manager at serve time (helper default is 1e-3 for tests).
# RK4 truncation at 5 ms over <= 0.6 s is sub-mm — far below the 17 mm engine-integration floor.
SERVE_BACKINT_H = 5e-3
# Park position (env-local): far below the table, out of sight; rewritten kinematically every
# control step so nothing (gravity, stale forces) can accumulate on a parked ball.
PARK_POS_ENV = (0.0, 0.0, -10.0)
# Post-strike balls below this env-local z are done (landing recorded or hopeless) -> park.
KILL_Z_ENV = -2.0

_MODE_PARKED = 0   # waiting for the question / for tts to enter the serve horizon
_MODE_INBOUND = 1  # launched; PhysX + aero wrench own the flight; strike frame not yet reached
_MODE_POST = 2     # past the strike frame (no impulse — Phase B); flying until landing/kill


# --------------------------------------------------------------------------------------------- #
# Pure helpers (torch-only; unit-tested Isaac-free)
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TableContactParams:
    """Venue TABLE contact constants (configs/ball_physics_venue.yaml ``contact.table``) +
    the ball constants the contact math needs. The table is static (v_r = 0, n = +z) and uses a
    CONSTANT e_eff (the F4 velocity-dependent restitution applies to the PADDLE only)."""

    e_eff: float
    a_t: float
    b_t: float
    mu: float
    ball_radius: float
    inertia_coeff: float
    source_path: str


def load_venue_table_params(path: str | None = None) -> TableContactParams:
    """Read the table-contact block from the SAME venue yaml the flight/paddle constants use."""
    import yaml

    path = path or _vb.default_venue_yaml_path()
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    tab = raw["contact"]["table"]
    return TableContactParams(
        e_eff=float(tab["e_eff"]),
        a_t=float(tab["a_t"]),
        b_t=float(tab["b_t"]),
        mu=float(tab["mu_safety"]),
        ball_radius=float(raw["ball"]["radius"]),
        inertia_coeff=float(raw["ball"]["inertia_coeff"]),
        source_path=os.path.abspath(path),
    )


def back_integrate_incoming(
    contact_pos: torch.Tensor,
    incoming_vel: torch.Tensor,
    omega: torch.Tensor,
    tts: torch.Tensor,
    prm,
    h: float = 1e-3,
    surface_z: float = 0.76,
    margin: float = SERVE_PLANE_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Venue-model REVERSE-TIME integration from the contact state, TRUNCATED at the table plane.

    Integrates the fitted flight ODE ``a = g - k_d|v|v + k_m (omega x v)`` backward
    (``virtual_ball.rk4_step`` with a NEGATIVE per-env step) for up to ``tts`` seconds, STOPPING
    per env at the last state strictly above ``z = surface_z + ball_radius + margin`` (and below
    the ``BACKINT_SPEED_CAP`` reverse-drag guard). The returned state is ON the incoming
    trajectory, so forward flight from ``(launch_pos, launch_vel)`` for ``tts_effective`` seconds
    arrives at ``(contact_pos, incoming_vel)`` — roundtrip error is RK4 truncation only (sub-mm
    at h = 1e-3; tested untruncated at tts 0.3/0.6/1.0 s and truncated for rising/long-tts
    cases).

    WHY TRUNCATE (seed=1 pod defect): for rising contact velocities (vz >= 0) and moderate tts
    the pure backward path dips below the table plane — in reality that segment is PRE-BOUNCE —
    and an un-truncated launch sat under/inside the table (pb_serve_err_m 0.58 m). Phase A
    serves only the FINAL BALLISTIC SEGMENT (post-last-bounce); realizing the pre-bounce segment
    (bounce-map inversion) is the future bounce-aware serve. ``tts_effective`` runs ~0.14-0.35 s
    for typical bank contact heights — the serve simply fires later.

    Vectorized over envs with PER-ENV step size: ``n = ceil(max(tts)/h)`` fixed-length loop,
    ``h_i = tts_i / n`` (envs with smaller tts get a smaller, MORE accurate step; ``tts_i = 0``
    rows take identity steps); a row that would step below the plane (or past the speed cap)
    rejects that step and freezes, so ``tts_effective`` is an exact multiple of its ``h_i``.
    Frame-free in xy; ``surface_z`` must be given in the SAME frame as ``contact_pos`` (env-local
    ``vb_table_surface_z`` when positions are env-local, or origin-shifted when world — the
    tracking env grids are z-flat so the manager passes world contact points with the env-local
    plane unchanged). Omega is constant in flight (the fit's assumption).

    Args:
        contact_pos: (N, 3) question contact point.
        incoming_vel: (N, 3) question incoming velocity AT the contact point.
        omega: (N, 3) question incoming spin (rad/s, constant in flight).
        tts: (N,) time to strike in seconds (clamped at 0 from below).
        prm: ``virtual_ball.VirtualBallParams`` (venue flight constants).
        h: nominal reverse step size (s).
        surface_z: table surface height in the frame of ``contact_pos``.
        margin: extra clearance above ``surface_z + ball_radius`` where truncation stops.

    Returns:
        ``(launch_pos, launch_vel, tts_effective)``: (N, 3), (N, 3), (N,). ``tts_effective ==
        tts`` where nothing truncated; smaller where the plane (or the speed cap) cut the span.
    """
    t_back = tts.clamp(min=0.0)
    t_max = float(t_back.max().item()) if t_back.numel() else 0.0
    if t_max <= 0.0:
        return contact_pos.clone(), incoming_vel.clone(), torch.zeros_like(t_back)
    z_min = float(surface_z) + float(prm.ball_radius) + float(margin)
    n_steps = max(1, int(math.ceil(t_max / float(h))))
    h_i = (t_back / float(n_steps)).unsqueeze(-1)  # (N, 1), broadcasts through rk4_step
    p, v = contact_pos, incoming_vel
    t_eff = torch.zeros_like(t_back)
    alive = torch.ones_like(t_back, dtype=torch.bool)
    for _ in range(n_steps):
        p_new, v_new = _vb.rk4_step(p, v, omega, -h_i, prm)
        # Accept the step only where the row is still integrating AND the new state stays
        # strictly above the truncation plane AND below the reverse-drag speed cap; a rejected
        # step freezes the row at the last valid state (its candidate recomputes identically and
        # keeps being rejected — no NaN path).
        ok = (
            alive
            & (p_new[:, 2] > z_min)
            & (torch.linalg.norm(v_new, dim=-1) < BACKINT_SPEED_CAP)
        )
        okc = ok.unsqueeze(-1)
        p = torch.where(okc, p_new, p)
        v = torch.where(okc, v_new, v)
        t_eff = t_eff + h_i.squeeze(-1) * ok
        alive = ok
    return p, v, t_eff


def predict_table_contact(
    v_minus: torch.Tensor,
    omega_minus: torch.Tensor,
    tp: TableContactParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fitted table bounce: the spin-equation contact with the venue TABLE params.

    Same angle-dependent tangential-impulse math as ``virtual_ball.predict_paddle_contact`` /
    ``ball_physics_fit/contact_model.predict_contact`` specialized to the static table:
    ``v_r = 0``, ``n = +z`` (the bounce caller guarantees a descending ball, so the oriented
    normal is +z by construction), CONSTANT ``e_eff``::

        u    = v- + w- x (-R n)
        s    = clip((a_t + b_t cos(theta)) |u_t|, 0, mu (1+e) |u_n|)
        dv_t = -s unit(u_t);  dv_n = -(1+e) u_n n;  dw = -(1/(cR)) n x dv_t

    Inputs (N, 3); returns ``(v_plus, omega_plus)``.
    """
    eps = 1e-9
    R = tp.ball_radius
    c = tp.inertia_coeff
    n = torch.zeros_like(v_minus)
    n[:, 2] = 1.0
    r = -R * n

    u = v_minus + torch.cross(omega_minus, r, dim=-1)
    u_n_signed = torch.sum(u * n, dim=-1, keepdim=True)          # (N, 1), < 0 for a descending ball
    u_t_vec = u - u_n_signed * n
    u_t_mag = torch.linalg.norm(u_t_vec, dim=-1, keepdim=True)
    u_n_abs = torch.abs(u_n_signed)

    cos_theta = u_n_abs / (torch.hypot(u_t_mag, u_n_signed) + eps)
    raw = (tp.a_t + tp.b_t * cos_theta) * u_t_mag
    cap = tp.mu * (1.0 + tp.e_eff) * u_n_abs
    s = torch.clamp(raw, min=0.0).minimum(cap)

    safe_dir = u_t_vec / (u_t_mag + eps)
    delta_v_t = torch.where(u_t_mag > eps, -s * safe_dir, torch.zeros_like(u_t_vec))
    delta_v_n = -(1.0 + tp.e_eff) * u_n_signed * n
    delta_omega = -(1.0 / (c * R)) * torch.cross(n, delta_v_t, dim=-1)

    return v_minus + delta_v_n + delta_v_t, omega_minus + delta_omega


def schedule_serves(
    parked: torch.Tensor,
    tts: torch.Tensor,
    horizon_s: float = SERVE_HORIZON_S,
    min_tts_s: float = 0.0,
) -> torch.Tensor:
    """Which parked envs launch their ball THIS control step.

    A parked env serves the first step its ``time_to_strike`` enters ``(min_tts_s, horizon_s]``:
    beyond the horizon it keeps waiting (parked far below the table); at/below ``min_tts_s``
    (the manager passes one control step) there is no physics window left for the ball to fly, so
    it never serves — the strike is then counted as ``pb_missed_serve`` (happens when a resample
    lands inside the last control step before the strike, or the reference clock jumps).
    """
    return parked & (tts <= float(horizon_s)) & (tts > float(min_tts_s))


def table_bounds_mask(
    xy: torch.Tensor, near_x: float, table_len: float, half_w: float
) -> torch.Tensor:
    """Env-local footprint test for the code-driven bounce: x in [near, near+len], |y| <= half_w."""
    return (
        (xy[:, 0] >= float(near_x))
        & (xy[:, 0] <= float(near_x) + float(table_len))
        & (xy[:, 1].abs() <= float(half_w))
    )


# --------------------------------------------------------------------------------------------- #
# Manager (owned/called by RacketTargetCommand when cfg.physical_ball is on)
# --------------------------------------------------------------------------------------------- #
class PhysicalBallManager:
    """Drives the per-env physical ball through PARKED -> INBOUND -> POST. Metrics only.

    SEAM (deliberate, single integration surface): all control-rate work hooks into
    ``RacketTargetCommand`` — ``update(exact_strike)`` once per control step from
    ``_update_metrics`` (after ``_vb_evaluate``), ``on_resample(env_ids)`` from
    ``_resample_command``; the per-substep aero wrench + bounce/landing detection run in a
    ``sim.add_physics_callback`` (the table_tennis_env.py mechanism). Chosen over an interval
    event term (no access to the per-swing resample/tts/question stream without cross-manager
    coupling) and over a scene-entity update (no view of command state at all); the shadow-ball
    driver already proved this seam mech-clean, so both measurement channels share one shape.
    """

    def __init__(self, command, env):
        self._cmd = command
        self._env = env
        self.device = command.device
        n = command.num_envs

        try:
            self._ball = env.scene["pb_ball"]
        except KeyError as exc:
            raise KeyError(
                "PhysicalBallManager: scene entity 'pb_ball' not found. physical_ball=True "
                "requires the scene attachment from hope_env_cfg.attach_physical_ball_scene "
                "(run automatically by HOPEPingPongAgibotA3EnvCfg.__post_init__ or the train.py "
                "task.physical_ball override translation)."
            ) from exc

        # Venue constants: flight via the same loader as the reward path; table contact + mass
        # from the same YAML (single source of truth).
        import yaml as _yaml

        self._prm = _vb.load_venue_params()
        self._tp = load_venue_table_params(self._prm.source_path)
        with open(self._prm.source_path, "r") as fh:
            self._mass = float(_yaml.safe_load(fh)["ball"]["mass"])

        # Virtual-table landmarks (env-local), same convention as the vb reward path. geometry.py
        # is pure python; fall back to a file-path load when the package import is unavailable
        # (Isaac-free harness tests drive the real manager through mocks).
        try:
            from whole_body_tracking.tasks.table_tennis import geometry as _tt_geom
        except Exception:
            import importlib.util as _ilu
            import sys as _sys

            _geo_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "table_tennis", "geometry.py"
            )
            _spec = _ilu.spec_from_file_location("physical_ball._tt_geometry", _geo_path)
            _tt_geom = _ilu.module_from_spec(_spec)
            _sys.modules["physical_ball._tt_geometry"] = _tt_geom  # dataclass resolution needs this
            _spec.loader.exec_module(_tt_geom)

        self._near_x = float(command.cfg.vb_table_near_x)
        self._table_len = float(_tt_geom.TABLE_LENGTH)
        self._half_w = float(_tt_geom.TABLE_WIDTH) / 2.0
        self._z_thr = float(command.cfg.vb_table_surface_z) + float(self._prm.ball_radius)

        # Lifecycle + event buffers.
        self._mode = torch.full((n,), _MODE_PARKED, dtype=torch.long, device=self.device)
        self._landed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._land_new = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._land_xy = torch.zeros(n, 2, device=self.device)
        self._bounce_new = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Truncation latch: set on any candidate step whose serve is delayed by the plane
        # truncation; COUNTED into pb_serve_truncated_count exactly once, at CONSUMPTION (the
        # serve, or the strike if it never served), where it is also cleared — and ONLY there.
        # Deliberately NOT cleared in on_resample: upstream _resample_command can repeat within
        # one physical wait (motion.just_resampled stays latched across steps at low env counts
        # — the seed=1 exposing config), and a resample-cleared latch either re-counts per
        # candidate step (counting while waiting: the observed 1945 counts vs 110 serves) or
        # never counts at all (counting at serve: the final repeat's fresh discovery serves
        # un-delayed). Consumption events are the only cadence-invariant swing boundary; a latch
        # carried from an aborted wait into the next swing's consumption keeps the AGGREGATE
        # honest (one count per physical wait that ever hit truncation).
        self._trunc_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Per-swing tts_effective cache: the final-ballistic-segment length is a trajectory
        # property fixed when the question is sampled, so it is DISCOVERED once (first candidate
        # step) and cached; waiting steps only compare tts against it instead of re-running the
        # full reverse integration every step (the wasted-compute half of the same defect).
        self._teff_cache = torch.zeros(n, device=self.device)
        self._teff_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._prev_pos_env = torch.zeros(n, 3, device=self.device)
        # Reusable wrench buffers (num_envs, 1 body, 3), zeroed like table_tennis_env.py.
        self._force_b = torch.zeros(n, 1, 3, device=self.device)
        self._torque_b = torch.zeros(n, 1, 3, device=self.device)
        self._identity_quat = torch.zeros(n, 4, device=self.device)
        self._identity_quat[:, 0] = 1.0
        self._park_pos_env = torch.tensor(PARK_POS_ENV, device=self.device).expand(n, 3)

        # Cumulative counters + sample-weighted EMAs (vb-metric discipline: decay only on
        # event-carrying steps — exact at large env counts, slightly stale at small).
        self._serve_count = 0.0
        self._meas_count = 0.0
        self._missed_serve_count = 0.0
        self._trunc_count = 0.0
        self._bounce_count = 0.0
        self._land_count = 0.0
        self._land_on_table_count = 0.0
        self._serve_err_acc = 0.0
        self._serve_vel_err_acc = 0.0
        self._serve_n_acc = 0.0
        m = command.metrics
        m["pb_serve_err_m"] = torch.zeros(n, device=self.device)
        m["pb_serve_vel_err"] = torch.zeros(n, device=self.device)
        m["pb_serve_count"] = torch.zeros(n, device=self.device)
        m["pb_strike_meas_count"] = torch.zeros(n, device=self.device)
        m["pb_missed_serve_count"] = torch.zeros(n, device=self.device)
        m["pb_serve_truncated_count"] = torch.zeros(n, device=self.device)
        m["pb_bounce_count"] = torch.zeros(n, device=self.device)
        m["pb_land_count"] = torch.zeros(n, device=self.device)
        m["pb_land_on_table_count"] = torch.zeros(n, device=self.device)
        m["pb_land_x"] = torch.zeros(n, device=self.device)
        m["pb_land_y"] = torch.zeros(n, device=self.device)

        # Per-substep aero + bounce/landing via the table_tennis_env.py physics-callback
        # mechanism. Defensive like the shadow driver: on registration failure the ball still
        # flies on PhysX gravity alone and events are detected at the control rate (degraded
        # measurement) — never block training.
        self._cb_active = False
        try:
            env.sim.add_physics_callback("hope_physical_ball", self._on_physics_step)
            self._cb_active = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(
                f"[PhysicalBallManager] could not register the physics callback ({exc!r}); "
                "the physical ball flies on PhysX gravity only and bounce/landing detection "
                "degrades to the CONTROL rate.",
                flush=True,
            )
        print(
            f"[PhysicalBallManager] PHYSICAL ball ON (truth instrument, metrics-only): "
            f"R={self._prm.ball_radius} m, mass={self._mass} kg, k_d={self._prm.k_d}, "
            f"k_m={self._prm.k_m}, table e_eff={self._tp.e_eff} a_t={self._tp.a_t} "
            f"mu={self._tp.mu} (code-driven bounce), serve horizon={SERVE_HORIZON_S}s, "
            f"bounce plane z={self._z_thr:.4f} env-local, x in "
            f"[{self._near_x:.2f}, {self._near_x + self._table_len:.2f}], |y|<={self._half_w:.3f}. "
            f"Racket impulse = Phase B (ball passes through the robot).",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # control-rate hooks (called from RacketTargetCommand)
    # ------------------------------------------------------------------ #
    def on_resample(self, env_ids) -> None:
        """New question for these envs (reset or clip wrap): park until tts enters the horizon."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._mode[ids] = _MODE_PARKED
        self._landed[ids] = False
        self._land_new[ids] = False
        self._bounce_new[ids] = False
        # NOTE: _trunc_flag is NOT cleared here — it is consumed (counted + cleared) at the
        # serve/strike only, so pb_serve_truncated_count stays exactly-once-per-wait even when
        # upstream resamples repeat within one wait (see the latch comment in __init__).
        self._teff_valid[ids] = False  # new question -> new trajectory -> re-discover its segment
        self._prev_valid[ids] = False

    def update(self, exact_strike: torch.Tensor) -> None:
        """Once per control step from ``_update_metrics`` (after ``_vb_evaluate``)."""
        cmd = self._cmd
        origins = self._env.scene.env_origins
        step_dt = float(self._env.step_dt)

        # 1) fold bounce/landing events flagged by the physics callback since last control step.
        self._consume_events()

        # 2) serve: parked envs whose tts entered (step_dt, SERVE_HORIZON_S] are CANDIDATES.
        #    FIRST candidate step per swing runs the reverse integration (env-local frame,
        #    vb-convention) once to DISCOVER the final-ballistic-segment length tts_effective,
        #    which is cached — subsequent WAITING steps only compare tts against the cache, no
        #    re-integration. A row launches only when its remaining tts fits inside the
        #    un-truncated segment (tts <= tts_effective): un-truncated discoveries serve on the
        #    spot from that same integration; TRUNCATED rows serve LATER (one more integration on
        #    the serve step, over exactly t_back = tts <= tts_effective — un-truncated by
        #    construction), from ON the incoming trajectory, so forward flight for exactly tts
        #    seconds still arrives at the question (contact, velocity) at the exact-strike frame
        #    (tts is an exact multiple of step_dt — bank runs forbid retiming). Rows whose whole
        #    final segment is shorter than one control step never serve and are counted at the
        #    strike as pb_missed_serve. Cost: <= 2 integrations per swing (was: one per waiting
        #    step). Truncation is LATCHED here but counted only at consumption (serve/strike).
        just_served = torch.zeros_like(self._landed)
        cand = schedule_serves(self._mode == _MODE_PARKED, cmd.time_to_strike,
                               SERVE_HORIZON_S, min_tts_s=step_dt)
        discover = cand & ~self._teff_valid
        serve_cached = cand & self._teff_valid & (cmd.time_to_strike <= self._teff_cache + 1e-6)
        integ = discover | serve_cached
        if bool(integ.any()):
            t_back = cmd.time_to_strike.clamp(min=0.0, max=SERVE_HORIZON_S)
            pos_env, vel_w, t_eff = back_integrate_incoming(
                cmd.racket_target_pos_w - origins, cmd.vb_vel_in_w, cmd.vb_spin_in_w,
                t_back, self._prm, h=SERVE_BACKINT_H,
                surface_z=float(cmd.cfg.vb_table_surface_z), margin=SERVE_PLANE_MARGIN,
            )
            self._teff_cache = torch.where(discover, t_eff, self._teff_cache)
            self._teff_valid |= discover
            # 1e-4 truncation tolerance: t_eff is a float32 per-step sum (~1e-5 noise); a row
            # truly truncated within the last 1e-4 s costs <= |v|*1e-4 ~ 0.4 mm at the strike —
            # far below the 17 mm engine floor.
            due = integ & (t_eff >= t_back - 1e-4)
            self._trunc_flag |= discover & ~due  # latch only; counted at serve/strike
            just_served = due
            if bool(due.any()):
                ids = torch.where(due)[0]
                pose = torch.cat([origins[ids] + pos_env[ids], self._identity_quat[ids]], dim=-1)
                vel6 = torch.cat([vel_w[ids], cmd.vb_spin_in_w[ids]], dim=-1)
                self._ball.write_root_pose_to_sim(pose, env_ids=ids)
                self._ball.write_root_velocity_to_sim(vel6, env_ids=ids)
                self._mode[ids] = _MODE_INBOUND
                self._landed[ids] = False
                self._prev_valid[ids] = False
                self._serve_count += float(len(ids))
                # One-per-swing truncation accounting, CONSUMED at the serve: invariant under
                # upstream resample repeats (each repeat clears the latch, the next candidate
                # step re-latches it, the single serve consumes it once).
                delayed = due & self._trunc_flag
                if bool(delayed.any()):
                    self._trunc_count += float(delayed.sum())
                self._trunc_flag = self._trunc_flag & ~due

        # 3) strike-frame truth measurement (the instrument's headline numbers). just_served envs
        #    are excluded (their write hasn't been integrated yet); with tts > step_dt at serve
        #    time this overlap cannot occur anyway.
        meas = exact_strike & (self._mode == _MODE_INBOUND) & ~just_served
        if bool(meas.any()):
            serve_err = torch.linalg.norm(
                self._ball.data.root_pos_w - cmd.racket_target_pos_w, dim=-1
            )
            vel_err = torch.linalg.norm(
                self._ball.data.root_lin_vel_w - cmd.vb_vel_in_w, dim=-1
            )
            decay = float(cmd.cfg.exact_success_decay)
            self._serve_err_acc = decay * self._serve_err_acc + float(serve_err[meas].sum())
            self._serve_vel_err_acc = decay * self._serve_vel_err_acc + float(vel_err[meas].sum())
            self._serve_n_acc = decay * self._serve_n_acc + float(meas.sum())
            self._meas_count += float(meas.sum())
            # Phase A: no racket impulse — the ball continues THROUGH the strike point/robot.
            self._mode[meas] = _MODE_POST
        # Strike frame reached while still parked (resampled inside the last control step, or the
        # question was never realizable this swing): count the unserved strike, and consume the
        # truncation latch of swings that were delayed but never got a serve window (t_eff <
        # one control step) — the OTHER once-per-swing consumption point.
        missed = exact_strike & (self._mode == _MODE_PARKED)
        if bool(missed.any()):
            self._missed_serve_count += float(missed.sum())
            late = missed & self._trunc_flag
            if bool(late.any()):
                self._trunc_count += float(late.sum())
            self._trunc_flag = self._trunc_flag & ~missed
        # Clock jumped past the strike WITHOUT an exact-strike frame (deploy-parity mid-swing clip
        # switch): the inbound flight is no longer measurable — let it fly out as POST (silently:
        # the gate was never evaluated; mirrors the shadow driver's stale handling).
        stale = (self._mode == _MODE_INBOUND) & (cmd.time_to_strike < -0.5 * step_dt) & ~meas
        if bool(stale.any()):
            self._mode[stale] = _MODE_POST

        # 4) retire post-strike balls that recorded their landing and fell away, then park-drive.
        done = (self._mode == _MODE_POST) & (
            (self._ball.data.root_pos_w[:, 2] - origins[:, 2]) < KILL_Z_ENV
        )
        if bool(done.any()):
            self._mode[done] = _MODE_PARKED
        parked = self._mode == _MODE_PARKED
        if bool(parked.any()):
            ids = torch.where(parked)[0]
            pose = torch.cat([origins[ids] + self._park_pos_env[ids], self._identity_quat[ids]], dim=-1)
            vel6 = torch.zeros(len(ids), 6, device=self.device)
            self._ball.write_root_pose_to_sim(pose, env_ids=ids)
            self._ball.write_root_velocity_to_sim(vel6, env_ids=ids)
            self._prev_valid[parked] = False

        # 5) metrics (broadcast counters; land_x/y held per env at its most recent landing).
        m = cmd.metrics
        m["pb_serve_count"][:] = self._serve_count
        m["pb_strike_meas_count"][:] = self._meas_count
        m["pb_missed_serve_count"][:] = self._missed_serve_count
        m["pb_serve_truncated_count"][:] = self._trunc_count
        m["pb_bounce_count"][:] = self._bounce_count
        m["pb_land_count"][:] = self._land_count
        m["pb_land_on_table_count"][:] = self._land_on_table_count
        if self._serve_n_acc >= 1.0:
            m["pb_serve_err_m"][:] = self._serve_err_acc / self._serve_n_acc
            m["pb_serve_vel_err"][:] = self._serve_vel_err_acc / self._serve_n_acc

        # Degraded fallback: no physics callback -> detect bounce/landing at the control rate.
        if not self._cb_active:
            self._detect_bounce_and_landing()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _consume_events(self) -> None:
        """Fold events flagged since the last control step into the held metrics/counters."""
        new = self._land_new
        if bool(new.any()):
            m = self._cmd.metrics
            m["pb_land_x"] = torch.where(new, self._land_xy[:, 0], m["pb_land_x"])
            m["pb_land_y"] = torch.where(new, self._land_xy[:, 1], m["pb_land_y"])
            on_table = new & table_bounds_mask(
                self._land_xy, self._near_x, self._table_len, self._half_w
            )
            self._land_count += float(new.sum())
            self._land_on_table_count += float(on_table.sum())
            self._land_new.zero_()
        if bool(self._bounce_new.any()):
            self._bounce_count += float(self._bounce_new.sum())
            self._bounce_new.zero_()

    def _detect_bounce_and_landing(self) -> None:
        """Descending surface+R crossing scan on the current vs previous ball sample (any rate).

        In-bounds crossings get the CODE-DRIVEN venue table bounce (velocity/spin rewritten, ball
        snapped back to the plane at the interpolated crossing point). The first POST-strike
        crossing (in-bounds or not — the shadow/vb landing-plane convention) is recorded as the
        landing. Pre-strike in-bounds crossings bounce too (physical consistency; see module
        docstring — cannot occur for in-envelope questions).
        """
        active = self._mode != _MODE_PARKED
        if not bool(active.any()):
            self._prev_valid.zero_()
            return
        origins = self._env.scene.env_origins
        pos_env = self._ball.data.root_pos_w - origins
        crossed, xy = _sb.landing_crossing(self._prev_pos_env, pos_env, self._z_thr)
        evt = active & self._prev_valid & crossed

        if bool(evt.any()):
            # landing record: first post-strike crossing.
            land = evt & (self._mode == _MODE_POST) & ~self._landed
            if bool(land.any()):
                self._land_xy = torch.where(land.unsqueeze(-1), xy, self._land_xy)
                self._landed |= land
                self._land_new |= land

            # code-driven bounce: in-bounds crossings only (off the ends/sides the ball just
            # keeps falling toward the floor — no floor model, it parks at KILL_Z_ENV).
            bounce = evt & table_bounds_mask(xy, self._near_x, self._table_len, self._half_w)
            if bool(bounce.any()):
                v_minus = self._ball.data.root_lin_vel_w
                w_minus = self._ball.data.root_ang_vel_w
                v_plus, w_plus = predict_table_contact(v_minus, w_minus, self._tp)
                ids = torch.where(bounce)[0]
                new_pos_env = pos_env.clone()
                new_pos_env[ids, 0] = xy[ids, 0]
                new_pos_env[ids, 1] = xy[ids, 1]
                new_pos_env[ids, 2] = self._z_thr
                pose = torch.cat(
                    [origins[ids] + new_pos_env[ids], self._ball.data.root_quat_w[ids]], dim=-1
                )
                vel6 = torch.cat([v_plus[ids], w_plus[ids]], dim=-1)
                self._ball.write_root_pose_to_sim(pose, env_ids=ids)
                self._ball.write_root_velocity_to_sim(vel6, env_ids=ids)
                self._bounce_new |= bounce
                # compare-from state for the next scan = the snapped-back position.
                pos_env = torch.where(bounce.unsqueeze(-1), new_pos_env, pos_env)

        self._prev_pos_env.copy_(pos_env)
        self._prev_valid.copy_(active)

    def _on_physics_step(self, dt: float) -> None:
        """Physics-substep callback (table_tennis_env.py mechanism): aero wrench + bounce scan.

        Asset ``data`` buffers are lazily refreshed against the sim timestamp, so reads here are
        per-substep fresh. The wrench is written for the FULL batch every substep (zeros where
        parked) so a just-parked ball never keeps a stale external force.
        """
        active = self._mode != _MODE_PARKED
        lin_vel_w = self._ball.data.root_lin_vel_w
        ang_vel_w = self._ball.data.root_ang_vel_w
        force_w = _sb.venue_aero_force(lin_vel_w, ang_vel_w, self._mass, self._prm.k_d, self._prm.k_m)
        force_w = force_w * active.unsqueeze(-1)
        # Isaac Lab 2.1 applies external wrenches in the BODY frame at the COM: rotate world->body.
        self._force_b[:, 0, :] = _sb.quat_rotate_inverse_wxyz(self._ball.data.root_quat_w, force_w)
        self._ball.set_external_force_and_torque(self._force_b, self._torque_b)
        self._ball.write_data_to_sim()

        self._detect_bounce_and_landing()
