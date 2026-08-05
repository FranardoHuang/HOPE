#!/usr/bin/env python3
"""Invert the *measured* ball-table restitution into a MuJoCo ``solref``.

WHY THIS SCRIPT EXISTS (plain language)
---------------------------------------
MuJoCo has no coefficient-of-restitution parameter.  A bounce is whatever the
soft-constraint reference dynamics produce.  Our authority is a venue
measurement -- ``e = 0.9215`` on 58 gated table bounces, cross-checked by an
independent OptiTrack rig at ``0.9102`` CI95 ``[0.8825, 0.9311]`` -- so the job
is to *invert* that number into ``solref``, and then prove the inversion by
dropping a real ball in the real engine and reading the rebound height.

THE ENGINE MATH WE ARE INVERTING
--------------------------------
``mujoco_warp/_src/constraint.py`` (and MuJoCo core) build the contact's
reference acceleration as::

    k = 1/(dmax^2 * timeconst^2 * dampratio^2)     b = 2/(dmax * timeconst)
    k = -solref[0]/dmax^2   if solref[0] <= 0      # "direct" parameterization
    b = -solref[1]/dmax     if solref[1] <= 0
    aref = -k*imp*pos - b*vel

For a single active contact the solver lands on::

    J*a = (1 - imp) * a_free + imp * aref

(exactly, because a contact's ``efc_invweight`` for a free body against static
world equals ``1/m``, which is also ``J M^-1 J'``; verified numerically --
``body_invweight0[ball] = 294.117 = 1/0.0034``).  One explicit-Euler step of a
ball arriving with normal velocity ``vel = -|v|`` and penetration ``pos = -d``
therefore leaves::

    v_next = |v| * (imp*b*dt - 1)  +  dt*imp^2*k*d  -  dt*(1-imp)*g

so, to first order,

    e  ~=  imp*b*dt - 1  +  dt^2*imp^2*k          (d ~= |v|*dt)

Three consequences drive every choice below.

1. **The rebound is velocity-level.**  The real ball-table dwell is 1-3 ms
   (``ball_physics_venue.yaml: capture.timing.contact_dwell_ms = 2``) and the
   plant timestep is 1 ms, so the contact is *inherently* sub-resolved: no
   ``(timeconst, dampratio)`` pair can both keep the ball from sinking and give
   the contact enough steps to behave like a continuous damped spring.  The
   honest formulation is the damping term.
2. **We need ``timeconst`` below the ``REFSAFE`` floor.**  ``REFSAFE`` clamps
   ``timeconst >= 2*dt``; at ``dt = 1 ms`` that caps ``imp*b*dt`` at ``1.0``,
   i.e. ``e = 0``.  That is exactly why the incumbent native path reads
   ``e = 0.131`` instead of ``0.9215``.  The *negative* (direct) solref
   parameterization bypasses ``REFSAFE`` without touching ``opt.disableflags``
   -- which matters, because the robot plant is verified at ``disableflags=0``
   and we are not allowed to perturb it.
3. **Keep the stiffness term small.**  ``k`` enters ``e`` as ``dt^2*imp^2*k``
   with a per-bounce jitter of the same size (the penetration at detection is
   uniform on ``(0, |v|dt]``).  ``k`` is therefore chosen from the *resting*
   requirement (``d_rest = g/(k*imp)``), not from the bounce, and then folded
   out of ``b`` by the numerical solve.

We also keep ``solimp = (d, d, ...)`` with ``dmin == dmax`` so the impedance is
constant instead of ramping with penetration -- that removes the one remaining
source of impact-speed dependence, matching the measurement (venue F3: ``e``
flat in ``v_n``, slope ``+0.005/m/s`` CI ``[-0.007, +0.018]``).

WHAT IS MEASURED
----------------
Every world drops a ball from a known height ``h`` onto a table box and we read
the **rebound height ratio**, ``e = sqrt(h_rebound / h)``.  With
``opt.density = opt.viscosity = 0`` there is no drag, so that ratio is exact,
not a fit.  ``mujoco_warp`` batches ``geom_solref`` per world, so one GPU run
sweeps the whole (parameter x impact-speed) grid at once.

Usage
-----
  python calibrate_restitution.py --sweep          # response surface + solve
  python calibrate_restitution.py --confirm        # single param, e(v_n) curve
  python calibrate_restitution.py --spin-probe     # tangential grip a_t check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict

import numpy as np

# --------------------------------------------------------------------------
# Measured authority (configs/ball_physics_venue.yaml + the OptiTrack refit).
# --------------------------------------------------------------------------
E_TABLE_MEASURED = 0.9215          # venue, 58 gated bounces, v_n 1.0-4.5 m/s
E_TABLE_OPTITRACK = 0.9102         # independent rig, CI95 [0.8825, 0.9311]
E_ACCEPT = (0.88, 0.93)            # acceptance band handed down for this task
A_T_TABLE_MEASURED = 0.369         # tangential grip gain (OptiTrack 101 bounces)
V_N_ENVELOPE = (1.0, 4.5)          # validated normal-impact-speed range, m/s

BALL_RADIUS = 0.02
BALL_MASS = 0.0034
BALL_INERTIA_COEFF = 2.0 / 3.0     # hollow sphere
BALL_DIAGINERTIA = BALL_INERTIA_COEFF * BALL_MASS * BALL_RADIUS**2  # 9.0667e-07

GRAVITY = 9.81
TIMESTEP = 0.001                   # vendor plant, non-negotiable

# Constant-impedance contact: dmin == dmax kills the penetration-dependent ramp.
BALL_SOLIMP = (0.95, 0.95, 0.001, 0.5, 2.0)
DMAX = BALL_SOLIMP[1]

# Ball sliding friction.  TWO requirements pin this, see
# `pyramidal_effective_impedance` below:
#   (1) grip: a table bounce must *stick*, which needs
#       mu >= a_t*(u_t/v_n)/(1+e) = 0.192*(u_t/v_n); mu = 1.0 covers u_t/v_n <= 5.2,
#       well past anything in the validated envelope (the venue fit's own
#       mu_safety = 2.0 "never binds").
#   (2) cone decoupling: a MuJoCo *pyramidal* contact folds mu into the normal
#       channel's invweight.  mu = 1.0 is the unique value at which the effective
#       normal impedance equals solimp's dmax, i.e. the restitution stops
#       depending on the friction choice.  Verified: condim=1 and
#       condim=3 / mu=1.0 return bit-identical e.
BALL_FRICTION = (1.0, 0.005, 0.0001)

# k trades *resting* sag against *bounce-to-bounce* scatter, and nothing else:
# the ball is first seen already penetrating by d ~ U(0, |v|dt], and that d
# enters e through the k term.  Measured battery (10 drop heights, v_n 1.0-8.0):
#
#     k       e spread     resting penetration
#     3e2     0.00078          0.68 mm
#     1e3     0.00090          0.43 mm     <- adopted
#     3e3     0.00191          0.18 mm
#     1e4     0.00710          0.054 mm
#     3e4     0.02234          0.018 mm    (e_max 0.9312 leaves the accept band)
#
# 1e3 keeps the scatter at +-0.0005 -- two orders below the measurement's own
# CI width -- while a resting ball sinks only 2.1% of its radius.
DEFAULT_K = 1.0e3

# One measured Newton step on top of the analytic seed.  The closed form assumes
# the contact is first seen at the mean penetration |v|dt/2; the realised mean
# over the drop battery differs slightly.  Re-derive with --confirm whenever dt,
# mu, k, solimp or the cone change -- none of them are free parameters.
B_MEASURED_CORRECTION = 0.39       # 1/s, at k = 1e3, mu = 1.0, dt = 1e-3


def pyramidal_effective_impedance(mu: float, imp: float = DMAX,
                                  impratio: float = 1.0,
                                  condim: int = 3) -> float:
    """Fraction of ``aref`` a MuJoCo contact actually achieves in one step.

    ``mujoco_warp/_src/constraint.py::_efc_contact_update`` inflates the row
    invweight of a **pyramidal** contact by the friction coefficient::

        w = w0 * (1 + mu^2) * 2*mu^2 / impratio          (condim > 1)

    A purely normal impact excites all ``nrows = 2*(condim-1)`` pyramid rows
    symmetrically; their aggregate response is ``A = nrows/m`` (the tangential
    halves cancel), so with ``D = imp/(w*(1-imp))`` the solver lands on
    ``a = (x/(1+x)) * aref`` with::

        x = A*D = nrows * imp * impratio / (2*mu^2*(1+mu^2)*(1-imp))

    ``condim == 1`` has no friction rows and the fraction is ``imp`` itself.
    Note ``x = nrows*imp*impratio/(2*(1-imp))`` at ``mu = 1``, which for
    ``condim = 3`` gives exactly ``imp`` back -- the decoupling point.
    """
    if condim <= 1:
        return imp
    nrows = 2 * (condim - 1)
    if mu <= 0.0:
        return 1.0
    x = nrows * imp * impratio / (2.0 * mu * mu * (1.0 + mu * mu) * (1.0 - imp))
    return x / (1.0 + x)


def predict_e(b: float, k: float, mu: float, imp: float = DMAX,
              dt: float = TIMESTEP, impratio: float = 1.0,
              condim: int = 3) -> float:
    """Closed-form restitution of the direct-solref contact.

    ``v_next = v*(1 - ieff*b*dt) + dt*ieff*imp*k*d`` for penetration ``d``; the
    ball is first *seen* penetrating by ``d ~ U(0, |v|dt]``, whose mean gives
    the ``0.5*dt^2`` term.
    """
    ieff = pyramidal_effective_impedance(mu, imp, impratio, condim)
    return ieff * b * dt - 1.0 + 0.5 * dt * dt * ieff * imp * k


def analytic_seed_b(e_target: float, k: float, mu: float = BALL_FRICTION[0],
                    imp: float = DMAX, dt: float = TIMESTEP,
                    impratio: float = 1.0, condim: int = 3) -> float:
    """Invert :func:`predict_e` for ``b``.  This is the whole method."""
    ieff = pyramidal_effective_impedance(mu, imp, impratio, condim)
    stiffness_bias = 0.5 * dt * dt * ieff * imp * k
    return (1.0 + e_target - stiffness_bias) / (ieff * dt)


def calibrated_b(e_target: float = E_TABLE_MEASURED, k: float = DEFAULT_K,
                 mu: float = BALL_FRICTION[0]) -> float:
    """The shipped value: analytic seed + the one measured correction."""
    return analytic_seed_b(e_target, k, mu=mu) + B_MEASURED_CORRECTION


def kb_to_solref(k: float, b: float, dmax: float = DMAX) -> tuple[float, float]:
    """Physical (k, b) -> the negative solref pair MuJoCo stores."""
    return (-k * dmax * dmax, -b * dmax)


def solref_to_kb(s0: float, s1: float, dmax: float = DMAX) -> tuple[float, float]:
    return (-s0 / (dmax * dmax), -s1 / dmax)


# --------------------------------------------------------------------------
# Probe model.  Same <option> as the verified A3 plant, minus the robot.
# --------------------------------------------------------------------------

_PROBE_XML = """
<mujoco model="restitution_probe">
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 -{g}" integrator="Euler" solver="Newton"
          iterations="100" ls_iterations="50" tolerance="1e-8"
          ls_tolerance="0.01" ccd_iterations="35" impratio="1"
          cone="{cone}" jacobian="auto" density="0" viscosity="0"/>
  <worldbody>
    <geom name="table" type="box" size="1.37 0.7625 0.025" pos="0 0 -0.025"
          contype="1" conaffinity="7" condim="3"
          friction="1.5 0.005 0.0001" solref="0.005 1"/>
    <body name="ball" pos="0 0 0.5">
      <joint name="ball_free" type="free" damping="0" armature="0"
             frictionloss="0" stiffness="0"/>
      <inertial pos="0 0 0" mass="{m}" diaginertia="{I} {I} {I}"/>
      <geom name="ball" type="sphere" size="{r}" contype="1" conaffinity="7"
            condim="{condim}" priority="1" friction="{mu} {mut} {mur}"
            solref="{s0} {s1}" solimp="{d0} {d1} {w} {mid} {p}"/>
    </body>
  </worldbody>
{pair_block}
</mujoco>
"""

# An explicit <pair> is the ONLY way to give a contact its own `solreffriction`,
# and `solreffriction` is only honoured by the *elliptic* cone (see
# mujoco_warp/_src/constraint.py::_efc_contact_update -- the pyramidal branch
# never reads it).  Together they are the remedy for the tangential over-grip
# that a velocity-level restitution otherwise forces onto the friction rows.
_PAIR_BLOCK = """  <contact>
    <pair geom1="ball" geom2="table" condim="{condim}"
          friction="{mu} {mu} {mut} {mur} {mur}"
          solref="{s0} {s1}" solreffriction="{f0} {f1}"
          solimp="{d0} {d1} {w} {mid} {p}"/>
  </contact>"""

TABLE_TOP_Z = 0.0        # the probe table's top face
REST_Z = TABLE_TOP_Z + BALL_RADIUS


def probe_model(s0: float, s1: float, mu: float | None = None,
                condim: int = 3, cone: str = "pyramidal",
                solreffriction: tuple[float, float] | None = None):
    import mujoco

    mu_v = BALL_FRICTION[0] if mu is None else mu
    common = dict(
        dt=TIMESTEP, g=GRAVITY, m=BALL_MASS, I=BALL_DIAGINERTIA, r=BALL_RADIUS,
        mu=mu_v, mut=BALL_FRICTION[1], mur=BALL_FRICTION[2],
        s0=s0, s1=s1, condim=condim, cone=cone,
        d0=BALL_SOLIMP[0], d1=BALL_SOLIMP[1], w=BALL_SOLIMP[2],
        mid=BALL_SOLIMP[3], p=BALL_SOLIMP[4],
    )
    pair_block = ""
    if solreffriction is not None:
        pair_block = _PAIR_BLOCK.format(f0=solreffriction[0],
                                        f1=solreffriction[1], **common)
    xml = _PROBE_XML.format(pair_block=pair_block, **common)
    return mujoco.MjModel.from_xml_string(xml)


# --------------------------------------------------------------------------
# Bounce analysis (pure numpy, engine agnostic).
# --------------------------------------------------------------------------


@dataclass
class Bounce:
    world: int
    drop_height_m: float
    v_n_analytic: float
    contact_step: int
    dwell_steps: int
    penetration_max_m: float
    rebound_height_m: float
    e_height: float
    e_velocity: float
    ok: bool
    reason: str


def analyse(z: np.ndarray, vz: np.ndarray, heights: np.ndarray) -> list[Bounce]:
    """First bounce per world, from the recorded ball-centre height."""
    out: list[Bounce] = []
    nsteps, nworld = z.shape
    for w in range(nworld):
        zi, vi, h = z[:, w], vz[:, w], float(heights[w])
        v_n = math.sqrt(2.0 * GRAVITY * h)
        inside = zi < REST_Z
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            out.append(Bounce(w, h, v_n, -1, 0, 0.0, 0.0, float("nan"),
                              float("nan"), False, "never touched the table"))
            continue
        i0 = int(idx[0])
        # leave = first step at or after i0 that is back above the rest height
        after = np.nonzero(~inside[i0:])[0]
        if after.size == 0:
            out.append(Bounce(w, h, v_n, i0, nsteps - i0,
                              float(REST_Z - zi[i0:].min()), 0.0,
                              float("nan"), float("nan"), False,
                              "never separated (sank / stuck)"))
            continue
        i1 = i0 + int(after[0])
        pen = float(max(0.0, REST_Z - zi[i0:i1].min()))
        # apex before the next touch
        nxt = np.nonzero(inside[i1:])[0]
        i2 = i1 + int(nxt[0]) if nxt.size else nsteps
        seg = zi[i1:i2]
        if seg.size == 0:
            out.append(Bounce(w, h, v_n, i0, i1 - i0, pen, 0.0, float("nan"),
                              float("nan"), False, "no post-bounce samples"))
            continue
        apex = float(seg.max())
        rebound = apex - REST_Z
        if i2 >= nsteps and seg.argmax() == seg.size - 1:
            reason = "apex not bracketed -- run more steps"
            ok = False
        else:
            reason = ""
            ok = True
        e_h = math.sqrt(max(rebound, 0.0) / h)
        v_in = float(vi[i0 - 1]) if i0 > 0 else float(vi[0])
        v_out = float(vi[i1])
        e_v = abs(v_out) / abs(v_in) if v_in else float("nan")
        out.append(Bounce(w, h, v_n, i0, i1 - i0, pen, rebound, e_h, e_v,
                          ok and rebound > 0, reason))
    return out


# --------------------------------------------------------------------------
# GPU driver.
# --------------------------------------------------------------------------


def run_drops(params: list[tuple[float, float]], heights: list[float],
              steps: int, spin: tuple[float, float] | None = None,
              device: str = "cuda:0", mu: float | None = None,
              condim: int = 3, cone: str = "pyramidal",
              solreffriction: tuple[float, float] | None = None) -> dict:
    """One batched GPU run.  world = param_index * len(heights) + height_index.

    ``params`` are raw ``(solref[0], solref[1])`` pairs -- negative for the
    direct parameterization, positive for ``(timeconst, dampratio)``.
    ``spin`` is an optional ``(vx, omega_y)`` applied to every world, used by
    the tangential-grip probe.
    """
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp
    import torch

    nh = len(heights)
    nworld = len(params) * nh

    mjm = probe_model(*params[0], mu=mu, condim=condim, cone=cone,
                      solreffriction=solreffriction)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    ball_gid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball")

    wm = mjwarp.put_model(mjm)
    # Batch solref across worlds: mujoco_warp indexes [worldid % shape[0]].
    # An explicit <pair> overrides the geom pair, so batch whichever is live.
    if mjm.npair > 0:
        sr = np.repeat(wm.pair_solref.numpy(), nworld, axis=0).copy()
        for p, (s0, s1) in enumerate(params):
            sr[p * nh:(p + 1) * nh, 0, 0] = s0
            sr[p * nh:(p + 1) * nh, 0, 1] = s1
        wm.pair_solref = wp.array(sr, dtype=wp.vec2)
    else:
        sr = np.repeat(wm.geom_solref.numpy(), nworld, axis=0).copy()
        for p, (s0, s1) in enumerate(params):
            sr[p * nh:(p + 1) * nh, ball_gid, 0] = s0
            sr[p * nh:(p + 1) * nh, ball_gid, 1] = s1
        wm.geom_solref = wp.array(sr, dtype=wp.vec2)

    wd = mjwarp.put_data(mjm, mjd, nworld=nworld, nconmax=8, njmax=32)

    qpos = wp.to_torch(wd.qpos)
    qvel = wp.to_torch(wd.qvel)
    h_all = np.tile(np.asarray(heights, dtype=np.float64), len(params))
    qpos[:] = 0.0
    qpos[:, 3] = 1.0
    qpos[:, 2] = torch.as_tensor(REST_Z + h_all, dtype=qpos.dtype,
                                 device=qpos.device)
    qvel[:] = 0.0
    if spin is not None:
        qvel[:, 0] = spin[0]
        qvel[:, 4] = spin[1]
    mjwarp.forward(wm, wd)

    z = torch.zeros((steps + 1, nworld), dtype=torch.float32, device=qpos.device)
    vz = torch.zeros_like(z)
    vx = torch.zeros_like(z)
    wy = torch.zeros_like(z)
    z[0], vz[0], vx[0], wy[0] = qpos[:, 2], qvel[:, 2], qvel[:, 0], qvel[:, 4]
    for i in range(steps):
        mjwarp.step(wm, wd)
        z[i + 1] = qpos[:, 2]
        vz[i + 1] = qvel[:, 2]
        vx[i + 1] = qvel[:, 0]
        wy[i + 1] = qvel[:, 4]
    torch.cuda.synchronize()

    return {
        "z": z.cpu().numpy().astype(np.float64),
        "vz": vz.cpu().numpy().astype(np.float64),
        "vx": vx.cpu().numpy().astype(np.float64),
        "wy": wy.cpu().numpy().astype(np.float64),
        "heights": h_all,
        "nworld": nworld,
        "ball_gid": ball_gid,
    }


def run_drops_cpu(s0: float, s1: float, heights: list[float],
                  steps: int, mu: float | None = None,
                  condim: int = 3, cone: str = "pyramidal",
                  solreffriction: tuple[float, float] | None = None) -> dict:
    """Same drop on stock CPU MuJoCo -- an engine cross-check, not a fit."""
    import mujoco

    mjm = probe_model(s0, s1, mu=mu, condim=condim, cone=cone,
                      solreffriction=solreffriction)
    nworld = len(heights)
    z = np.zeros((steps + 1, nworld))
    vz = np.zeros((steps + 1, nworld))
    for w, h in enumerate(heights):
        mjd = mujoco.MjData(mjm)
        mjd.qpos[:] = 0.0
        mjd.qpos[2] = REST_Z + h
        mjd.qpos[3] = 1.0
        mjd.qvel[:] = 0.0
        mujoco.mj_forward(mjm, mjd)
        z[0, w], vz[0, w] = mjd.qpos[2], mjd.qvel[2]
        for i in range(steps):
            mujoco.mj_step(mjm, mjd)
            z[i + 1, w], vz[i + 1, w] = mjd.qpos[2], mjd.qvel[2]
    return {"z": z, "vz": vz, "heights": np.asarray(heights, dtype=np.float64)}


# --------------------------------------------------------------------------
# Modes.
# --------------------------------------------------------------------------


def _srf(args) -> tuple[float, float] | None:
    """``--solreffriction a b`` -> an explicit <pair>; None -> geom contact."""
    if not getattr(args, "solreffriction", None):
        return None
    a, b = args.solreffriction
    return (float(a), float(b))


def steps_for(heights) -> int:
    hmax = max(heights)
    t_fall = math.sqrt(2.0 * hmax / GRAVITY)
    return int(math.ceil((t_fall * (1.0 + E_TABLE_MEASURED) + 0.15) / TIMESTEP))


DEFAULT_HEIGHTS = [0.0510, 0.100, 0.200, 0.350, 0.550, 0.750, 0.920, 1.032]


def mode_sweep(args) -> dict:
    heights = DEFAULT_HEIGHTS
    ks = [float(x) for x in args.k_grid]
    payload: dict = {"family": args.family, "heights_m": heights,
                     "v_n_m_s": [math.sqrt(2 * GRAVITY * h) for h in heights],
                     "grids": []}
    mu = args.mu if args.mu is not None else BALL_FRICTION[0]
    for k in ks:
        seed = analytic_seed_b(E_TABLE_MEASURED, k, mu=mu, condim=args.condim)
        bs = np.linspace(seed * (1 - args.span), seed * (1 + args.span),
                         args.n_param)
        params = [kb_to_solref(k, float(b)) for b in bs]
        steps = steps_for(heights)
        raw = run_drops(params, heights, steps, device=args.device,
                        mu=args.mu, condim=args.condim, cone=args.cone,
                        solreffriction=_srf(args))
        bounces = analyse(raw["z"], raw["vz"], raw["heights"])
        nh = len(heights)
        rows = []
        for p, b in enumerate(bs):
            grp = bounces[p * nh:(p + 1) * nh]
            e_h = np.array([g.e_height for g in grp], dtype=np.float64)
            rows.append({
                "k": k, "b": float(b),
                "solref": list(kb_to_solref(k, float(b))),
                "e_height_by_v_n": e_h.tolist(),
                "e_height_mean": float(np.nanmean(e_h)),
                "e_height_slope_per_m_s": float(
                    np.polyfit([g.v_n_analytic for g in grp], e_h, 1)[0]
                ) if np.isfinite(e_h).all() else float("nan"),
                "penetration_max_mm": float(
                    1e3 * max(g.penetration_max_m for g in grp)),
                "dwell_steps_max": int(max(g.dwell_steps for g in grp)),
                "all_ok": bool(all(g.ok for g in grp)),
            })
        # solve b* for the mean-over-envelope e
        means = np.array([r["e_height_mean"] for r in rows])
        b_star = float("nan")
        good = np.isfinite(means)
        if good.sum() >= 2:
            b_star = float(np.interp(E_TABLE_MEASURED, means[good], bs[good]))
        payload["grids"].append({
            "k": k, "analytic_seed_b": seed, "b_solved": b_star,
            "solref_solved": list(kb_to_solref(k, b_star))
            if math.isfinite(b_star) else None,
            "steps": steps, "rows": rows,
        })
    return payload


def mode_confirm(args) -> dict:
    k, b = args.k, args.b
    s0, s1 = kb_to_solref(k, b)
    heights = DEFAULT_HEIGHTS + [float(x) for x in args.extra_heights]
    steps = steps_for(heights)
    raw = run_drops([(s0, s1)], heights, steps, device=args.device,
                    mu=args.mu, condim=args.condim, cone=args.cone,
                    solreffriction=_srf(args))
    bounces = analyse(raw["z"], raw["vz"], raw["heights"])
    e_h = np.array([b_.e_height for b_ in bounces])
    v_n = np.array([b_.v_n_analytic for b_ in bounces])
    slope, intercept = np.polyfit(v_n, e_h, 1)

    cpu = None
    if args.cpu_crosscheck:
        rawc = run_drops_cpu(s0, s1, heights, steps, mu=args.mu,
                             condim=args.condim, cone=args.cone,
                             solreffriction=_srf(args))
        bc = analyse(rawc["z"], rawc["vz"], rawc["heights"])
        cpu = {
            "e_height": [b_.e_height for b_ in bc],
            "e_height_mean": float(np.nanmean([b_.e_height for b_ in bc])),
            "max_abs_diff_vs_gpu": float(
                np.nanmax(np.abs(np.array([b_.e_height for b_ in bc]) - e_h))),
        }

    return {
        "k": k, "b": b, "solref": [s0, s1], "solimp": list(BALL_SOLIMP),
        "friction": [args.mu if args.mu is not None else BALL_FRICTION[0],
                     BALL_FRICTION[1], BALL_FRICTION[2]],
        "condim": args.condim, "timestep": TIMESTEP,
        "steps": steps,
        "bounces": [asdict(x) for x in bounces],
        "e_height_mean": float(np.nanmean(e_h)),
        "e_height_min": float(np.nanmin(e_h)),
        "e_height_max": float(np.nanmax(e_h)),
        "e_height_slope_per_m_s": float(slope),
        "e_height_at_v_n_3": float(slope * 3.0 + intercept),
        "in_accept_band": bool(np.all((e_h >= E_ACCEPT[0]) & (e_h <= E_ACCEPT[1]))),
        "penetration_max_mm": float(1e3 * max(x.penetration_max_m for x in bounces)),
        "dwell_steps_max": int(max(x.dwell_steps for x in bounces)),
        "cpu_crosscheck": cpu,
    }


def mode_spin(args) -> dict:
    """Does a MuJoCo Coulomb contact reproduce the measured grip a_t = 0.369?

    Theory: for a hollow sphere (I = c m R^2, c = 2/3) a *sticking* bounce
    changes the tangential centre velocity by ``dv_t = -u_t/(1 + 1/c) = -0.4 u_t``
    where ``u_t`` is the contact-point tangential velocity.  0.4 is therefore the
    rigid-body ceiling, and the measurement 0.369 sits at 92% of it.
    """
    k, b = args.k, args.b
    s0, s1 = kb_to_solref(k, b)
    heights = [0.35]
    steps = steps_for(heights)
    results = []
    for vx0, wy0 in [(2.0, 0.0), (0.0, 100.0), (2.0, 100.0), (2.0, -100.0),
                     (4.0, 0.0), (1.0, 50.0)]:
        raw = run_drops([(s0, s1)], heights, steps, spin=(vx0, wy0),
                        device=args.device, mu=args.mu, condim=args.condim,
                        cone=args.cone, solreffriction=_srf(args))
        z, vx, wy = raw["z"][:, 0], raw["vx"][:, 0], raw["wy"][:, 0]
        inside = z < REST_Z
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            continue
        i0 = int(idx[0])
        after = np.nonzero(~inside[i0:])[0]
        i1 = i0 + int(after[0])
        vx_in, vx_out = float(vx[i0 - 1]), float(vx[i1])
        wy_in, wy_out = float(wy[i0 - 1]), float(wy[i1])
        u_in = vx_in - BALL_RADIUS * wy_in
        dvx = vx_out - vx_in
        dwy = wy_out - wy_in
        results.append({
            "vx0": vx0, "omega_y0": wy0,
            "contact_point_u_t": u_in,
            "dv_t": dvx, "d_omega_y": dwy,
            "a_t_measured_in_sim": (-dvx / u_in) if abs(u_in) > 1e-9 else None,
            "spin_consistency_ratio": (-dwy * BALL_RADIUS * BALL_INERTIA_COEFF
                                       / dvx) if abs(dvx) > 1e-9 else None,
            "u_t_after": float(vx[i1] - BALL_RADIUS * wy[i1]),
        })
    return {
        "solref": [s0, s1],
        "rigid_stick_ceiling_a_t": 1.0 / (1.0 + 1.0 / BALL_INERTIA_COEFF),
        "measured_table_a_t": A_T_TABLE_MEASURED,
        "cases": results,
    }


def mode_validate(args) -> dict:
    """Does the closed form predict the engine?  Sweep mu, compare, report error.

    This is the falsification test for the inversion: if ``predict_e`` and the
    dropped ball disagree, the inversion is a curve fit, not a derivation.
    """
    heights = DEFAULT_HEIGHTS
    steps = steps_for(heights)
    k, b = args.k, args.b
    rows = []
    for mu in [0.3, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        raw = run_drops([kb_to_solref(k, b)], heights, steps,
                        device=args.device, mu=mu, condim=args.condim,
                        cone=args.cone, solreffriction=_srf(args))
        bs = analyse(raw["z"], raw["vz"], raw["heights"])
        e_h = np.array([x.e_height for x in bs])
        rows.append({
            "mu": mu,
            "impedance_effective": pyramidal_effective_impedance(
                mu, condim=args.condim),
            "e_predicted": predict_e(b, k, mu, condim=args.condim),
            "e_measured_mean": float(np.nanmean(e_h)),
            "abs_error": abs(predict_e(b, k, mu, condim=args.condim)
                             - float(np.nanmean(e_h))),
        })
    return {"k": k, "b": b, "rows": rows,
            "max_abs_error": max(r["abs_error"] for r in rows)}


def _ieff(args) -> float:
    mu = args.mu if args.mu is not None else BALL_FRICTION[0]
    if args.cone == "elliptic":
        return pyramidal_effective_impedance(mu, condim=1)
    return pyramidal_effective_impedance(mu, condim=args.condim)


def mode_rest(args) -> dict:
    """Where does a *resting* ball sit?  ``k`` is chosen by this, not by e."""
    k, b = args.k, args.b
    s0, s1 = kb_to_solref(k, b)
    raw = run_drops([(s0, s1)], [0.002], 4000, device=args.device,
                    mu=args.mu, condim=args.condim, cone=args.cone,
                    solreffriction=_srf(args))
    z = raw["z"][:, 0]
    settle = z[-500:]
    return {
        "k": k, "solref": [s0, s1],
        "analytic_rest_penetration_mm": 1e3 * (1.0 - _ieff(args)) * GRAVITY
        / (_ieff(args) * k * DMAX),
        "measured_rest_penetration_mm": float(1e3 * (REST_Z - settle.mean())),
        "measured_rest_penetration_peak_to_peak_mm": float(
            1e3 * (settle.max() - settle.min())),
        "final_z_m": float(z[-1]),
        "fraction_of_ball_radius": float((REST_Z - settle.mean()) / BALL_RADIUS),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--spin-probe", action="store_true")
    p.add_argument("--validate-model", action="store_true")
    p.add_argument("--rest", action="store_true")
    p.add_argument("--family", choices=("direct", "standard"), default="direct")
    p.add_argument("--k-grid", nargs="*", default=[3e3, 1e4, 3e4])
    p.add_argument("--n-param", type=int, default=21)
    p.add_argument("--span", type=float, default=0.06)
    p.add_argument("--k", type=float, default=DEFAULT_K)
    p.add_argument("--b", type=float, default=None)
    p.add_argument("--extra-heights", nargs="*", default=[])
    p.add_argument("--mu", type=float, default=None,
                   help="ball sliding friction; the pyramidal cone couples it "
                        "into the normal response, so e depends on it")
    p.add_argument("--condim", type=int, default=3)
    p.add_argument("--cone", choices=("pyramidal", "elliptic"),
                   default="pyramidal",
                   help="plant value is pyramidal (MuJoCo default); elliptic is "
                        "a registered deviation that unlocks solreffriction")
    p.add_argument("--solreffriction", nargs=2, default=None,
                   help="emit an explicit <pair> carrying this friction-row "
                        "solref; only the elliptic cone reads it")
    p.add_argument("--cpu-crosscheck", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    mu_eff = args.mu if args.mu is not None else BALL_FRICTION[0]
    if args.b is None:
        args.b = analytic_seed_b(E_TABLE_MEASURED, args.k, mu=mu_eff,
                                 condim=args.condim)

    payload: dict = {
        "measured_authority": {
            "e_table_venue": E_TABLE_MEASURED,
            "e_table_optitrack": E_TABLE_OPTITRACK,
            "accept_band": list(E_ACCEPT),
            "v_n_envelope": list(V_N_ENVELOPE),
            "a_t_table": A_T_TABLE_MEASURED,
        },
        "analytic": {
            "formula": "e = ieff*b*dt - 1 + 0.5*dt^2*ieff*imp*k",
            "impedance_effective_formula":
                "ieff = x/(1+x), x = 2*(condim-1)*imp*impratio"
                " / (2*mu^2*(1+mu^2)*(1-imp))   [pyramidal cone]",
            "dmax": DMAX, "timestep": TIMESTEP, "mu": mu_eff,
            "condim": args.condim,
            "impedance_effective": pyramidal_effective_impedance(
                mu_eff, condim=args.condim),
            "seed_b_for_k": {
                str(k): analytic_seed_b(E_TABLE_MEASURED, float(k), mu=mu_eff,
                                        condim=args.condim)
                for k in args.k_grid},
            "b_used": args.b, "k_used": args.k,
            "solref_used": list(kb_to_solref(args.k, args.b)),
            "e_predicted": predict_e(args.b, args.k, mu_eff,
                                     condim=args.condim),
        },
    }
    if args.sweep:
        payload["sweep"] = mode_sweep(args)
    if args.confirm:
        payload["confirm"] = mode_confirm(args)
    if args.spin_probe:
        payload["spin_probe"] = mode_spin(args)
    if args.validate_model:
        payload["validate_model"] = mode_validate(args)
    if args.rest:
        payload["rest"] = mode_rest(args)

    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
