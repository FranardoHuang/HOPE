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

HOW THE RESULT IS GRADED
------------------------
``--confirm`` is the mode that claims "the ball bounces right", so it is the
mode that is allowed to say NO.  It exits non-zero when a gate fires.  Four
gates, in plain language:

* **e_within_field_band** -- every bounce sits inside what the venue
  measurement can resolve.  A *sanity floor*, deliberately labelled as weak:
  the band is 0.0235 wide against a simulator spread of order 1e-4.
* **e_mean_matches_authority** -- the inversion still lands on 0.9215.
* **impact_speed_spread_is_integrator_artefact_only** -- how far ``e`` moves
  across ``v_n`` 1.0-4.5 m/s.  The contact is designed to be speed-independent,
  so what is left is the Euler step, with the closed form ``dt^2*ieff*imp*k``.
  **This is the gate that catches a wrong ``k``.**
* **e_vs_v_n_slope_within_venue_ci** -- no invented speed dependence.

The last two need the impact-speed envelope actually swept.  A single drop
height reports ``NOT_MEASURED`` for them, and **NOT_MEASURED is not a pass**
(exit 5).

Usage
-----
  python calibrate_restitution.py --sweep          # response surface + solve
  python calibrate_restitution.py --confirm        # single param, e(v_n) curve
  python calibrate_restitution.py --spin-probe     # tangential grip a_t check

  # mutation test: the shipped k must pass, 10x k must fail
  python calibrate_restitution.py --confirm --calibrated-b --k 1000    # -> 0
  python calibrate_restitution.py --confirm --calibrated-b --k 10000   # -> 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict

import numpy as np

# --------------------------------------------------------------------------
# Measured authority and the acceptance gates.
#
# PROVENANCE.  Every number below is traceable to a line in this repo.  Nothing
# here is "handed down".
#
#   E_TABLE_MEASURED  0.9215
#     `configs/ball_physics_venue.yaml` -> `contact.table.e_eff`.  58 gated
#     table bounces over v_n 1.0-4.5 m/s.  The same block states the forensics
#     estimator over all 218 bounces (0.925, CI95 [0.920, 0.937]) and an
#     explicit "+- 0.005 systematic".
#   E_TABLE_OPTITRACK 0.9102
#     `configs/ball_physics_optitrack_20260730.yaml` -> `contact.table.e_eff`
#     note: independent rig, pure-bounce drag-corrected estimator, n = 20,
#     CI95 [0.8825, 0.9311].  That CI *contains* the venue value; the venue
#     value is the one that ships, so it is the one we invert.
#   E_FIELD_SIGMA     0.005
#     the venue fit's own stated systematic on table `e` (same yaml block).
#     This is a *field* sigma.  It is ~1400x the simulator's world-to-world
#     sigma (measured 3.5e-6), which is exactly why the simulator's own spread
#     must never be used to size an acceptance band.
#   E_ITTF_BAND       (0.876, 0.931)
#     quoted in `configs/ball_physics_optitrack_20260730.yaml` and
#     `docs/ball_physics_optitrack_20260730.md`.
#   E_SLOPE_CI_PER_M_S (-0.007, +0.018) per m/s
#     venue F3 after the contact-time fix: "flat, slope +0.005/m/s
#     CI [-0.007, +0.018]" (same yaml block; `docs/ball_physics_fit_report.md`
#     F3 row quotes the companion figures e_n 0.920, MAD 0.03 at n = 58).
#
# WHERE THE OLD (0.88, 0.93) CAME FROM: **UNCONFIRMED -- OPEN**.
#   The incumbent band's only comment was "acceptance band handed down for this
#   task".  Searched: the introducing commit (3d2fce66) and its message, every
#   `configs/*.yaml`, `docs/ball_physics*`, and the experiment record.  No
#   source states who set it or from which measurement.  It is numerically the
#   ITTF band (0.876, 0.931) rounded inward to two decimals, which is the
#   obvious reconstruction -- but that is a RECONSTRUCTION, not a citation.  Do
#   not promote it to a source without evidence.
# --------------------------------------------------------------------------
E_TABLE_MEASURED = 0.9215          # venue, 58 gated bounces, v_n 1.0-4.5 m/s
E_TABLE_OPTITRACK = 0.9102         # independent rig, CI95 [0.8825, 0.9311]
A_T_TABLE_MEASURED = 0.369         # tangential grip gain (OptiTrack 101 bounces)
V_N_ENVELOPE = (1.0, 4.5)          # validated normal-impact-speed range, m/s

E_FIELD_SIGMA = 0.005              # VENUE systematic on e -- a field sigma
E_ACCEPT_N_SIGMA = 3.0
E_ITTF_BAND = (0.876, 0.931)
E_ACCEPT_INCUMBENT = (0.88, 0.93)  # the unprovenanced band this replaces

_E_LO = E_TABLE_MEASURED - E_ACCEPT_N_SIGMA * E_FIELD_SIGMA   # 0.9065
_E_HI = E_TABLE_MEASURED + E_ACCEPT_N_SIGMA * E_FIELD_SIGMA   # 0.9365
# A fail-closed gate is never allowed to loosen, so each edge is clamped inward
# against the incumbent.  Only the lower edge actually moves (0.88 -> 0.9065);
# the upper stays at the incumbent 0.93, which is also the ITTF ceiling 0.931
# rounded in.  Width 0.05 -> 0.0235.
E_ACCEPT = (max(_E_LO, E_ACCEPT_INCUMBENT[0]),
            min(_E_HI, E_ACCEPT_INCUMBENT[1]))

# THE BAND ABOVE IS A PHYSICS SANITY FLOOR AND NOTHING MORE.  0.0235 wide
# against a simulator sigma of 3.5e-6 is still ~6,800 sim-sigma, so it cannot
# detect a mis-parameterised contact.  Proof, from the shipped k-mutation
# battery: multiplying the contact stiffness k by 10 moves e by at most 0.0043
# from the authority, which any field-sized band still passes.  The gates that
# CAN fail are below, and they are sized by the inversion's own derived
# precision instead of by the field.
#
# One rule sets both: **the calibration's own error budget must sit at least 3x
# INSIDE the field's stated systematic**, so that no calibration artefact can
# be mistaken for -- or hide inside -- a real measured effect.
E_CAL_TOL = E_FIELD_SIGMA / 3.0    # 1.667e-3
E_MEAN_TOL = E_CAL_TOL             # |e_mean - authority|
E_SPREAD_TOL = E_CAL_TOL           # e_max - e_min across the v_n envelope
E_SLOPE_CI_PER_M_S = (-0.007, 0.018)

# A slope or a spread only means something if the run actually swept the
# envelope.  Below this coverage the two envelope gates report NOT_MEASURED --
# which is NOT a pass.
E_MIN_DISTINCT_IMPACT_SPEEDS = 8
E_COVERAGE_FRACTION = 0.90         # of V_N_ENVELOPE's 3.5 m/s span

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
#     3e3     0.00191          0.18 mm     (spread gate: marginal, 1.15x over)
#     1e4     0.00710          0.054 mm    (spread gate: FAIL, 4.3x over)
#     3e4     0.02234          0.018 mm    (spread gate: FAIL; also e_max
#                                           0.9312 leaves the accept band)
#
# 1e3 keeps the scatter at +-0.0005 -- two orders below the measurement's own
# CI width -- while a resting ball sinks only 2.1% of its radius.  The "spread
# gate" annotations are `E_SPREAD_TOL` above; the incumbent (0.88, 0.93) band
# passed BOTH 1e3 and 1e4, which is what made it an empty test.
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
# The acceptance verdict.  Shared by this script and by a3_court_env.py so the
# two receipts cannot drift apart.
# --------------------------------------------------------------------------


def effective_impedance(mu: float | None = None, imp: float = DMAX,
                        condim: int = 3, cone: str = "pyramidal") -> float:
    """``ieff`` for the cone actually in use.

    The *elliptic* branch never folds ``mu`` into the normal channel, so it
    behaves like ``condim == 1`` for restitution purposes.
    """
    mu_v = BALL_FRICTION[0] if mu is None else float(mu)
    if cone == "elliptic":
        return pyramidal_effective_impedance(mu_v, imp, condim=1)
    return pyramidal_effective_impedance(mu_v, imp, condim=condim)


def discretization_spread(k: float = DEFAULT_K, mu: float | None = None,
                          imp: float = DMAX, dt: float = TIMESTEP,
                          condim: int = 3, cone: str = "pyramidal") -> float:
    """Full width of the impact-speed spread the Euler step alone produces.

    The contact is first *seen* penetrating by ``d ~ U(0, |v|dt]``, and ``d``
    enters ``e`` through the stiffness term as ``dt*ieff*imp*k*d/|v|``.  Sweep
    ``d`` over its whole uniform range and ``e`` sweeps ``dt^2*ieff*imp*k``.

    In plain language: **this is not physics, it is the integrator.**  It is
    the same number whatever ball or table you use, it scales linearly with
    ``k``, and it is the statistic ``E_SPREAD_TOL`` polices.  Measured against
    the shipped battery it is exact: predicted 9.025e-4 at ``k = 1e3``,
    observed ``e_max - e_min = 8.98e-4``.
    """
    return dt * dt * effective_impedance(mu, imp, condim, cone) * imp * float(k)


def _gate(name: str, plain: str, verdict: str, **extra) -> dict:
    out = {"gate": name, "plain": plain, "verdict": verdict}
    out.update(extra)
    return out


def restitution_verdict(e_values, v_n_values, *, k: float = DEFAULT_K,
                        mu: float | None = None, imp: float = DMAX,
                        dt: float = TIMESTEP, condim: int = 3,
                        cone: str = "pyramidal", n_worlds: int | None = None,
                        context: str = "") -> dict:
    """Grade a batch of measured bounces against the field authority.

    Returns a self-describing receipt: every gate carries its own statistic,
    its own limit, its own provenance and a one-line plain-language reading, so
    the JSON can be read without this source file.

    ``verdict`` is one of ``PASS`` / ``FAIL`` / ``NOT_MEASURED``.
    **``NOT_MEASURED`` is not a pass.**  A run that never swept the impact-speed
    envelope has not tested the two gates that can actually fail, and says so.
    """
    e = np.asarray(e_values, dtype=np.float64).ravel()
    v = np.asarray(v_n_values, dtype=np.float64).ravel()
    if v.size != e.size:
        v = np.full(e.size, np.nan)
    finite = np.isfinite(e)
    e, v = e[finite], v[finite]
    n_worlds = int(n_worlds if n_worlds is not None else e.size)

    ieff = effective_impedance(mu, imp, condim, cone)
    predicted_spread = discretization_spread(k, mu, imp, dt, condim, cone)

    provenance = {
        "authority_e": E_TABLE_MEASURED,
        "authority_source": "configs/ball_physics_venue.yaml :: "
                            "contact.table.e_eff -- 58 gated bounces, "
                            "v_n 1.0-4.5 m/s",
        "independent_rig_e": E_TABLE_OPTITRACK,
        "independent_rig_source": "configs/ball_physics_optitrack_20260730.yaml"
                                  " :: contact.table.e_eff note -- n=20, "
                                  "CI95 [0.8825, 0.9311], contains the venue "
                                  "value",
        "field_sigma": E_FIELD_SIGMA,
        "field_sigma_source": "the venue fit's own stated '+- 0.005 "
                              "systematic' on table e (same yaml block)",
        "accept_band": list(E_ACCEPT),
        "accept_band_rule": f"authority +- {E_ACCEPT_N_SIGMA:g} field sigma, "
                            "each edge clamped inward against the incumbent "
                            f"{list(E_ACCEPT_INCUMBENT)} so no fail-closed "
                            "gate loosens",
        "accept_band_replaced": list(E_ACCEPT_INCUMBENT),
        "accept_band_replaced_provenance": "UNCONFIRMED -- OPEN. The incumbent "
                                           "comment said only 'handed down'; "
                                           "no repo source names an origin. "
                                           "Numerically it is the ITTF band "
                                           f"{list(E_ITTF_BAND)} rounded "
                                           "inward, but that is a "
                                           "reconstruction, not a citation.",
        "ittf_band": list(E_ITTF_BAND),
        "slope_ci_per_m_s": list(E_SLOPE_CI_PER_M_S),
        "slope_ci_source": "venue F3 after the contact-time fix: 'flat, slope "
                           "+0.005/m/s CI [-0.007, +0.018]'",
        "calibration_tolerance": E_CAL_TOL,
        "calibration_tolerance_rule": "field sigma / 3 -- the calibration's "
                                      "own error budget must sit at least 3x "
                                      "inside the field's stated systematic, "
                                      "so no calibration artefact can be "
                                      "mistaken for a real measured effect",
    }

    if e.size == 0:
        return {
            "verdict": "NOT_MEASURED",
            "verdict_plain": "no bounce was recorded at all, so nothing was "
                             "tested; this is NOT a pass",
            "context": context,
            "gates": [],
            "failed_gates": [],
            "not_measured_gates": ["all"],
            "provenance": provenance,
            "independent_samples": {
                "n_worlds": n_worlds,
                "n_bounces_analysed": 0,
                "worlds_are_not_independent_samples": True,
            },
        }

    e_mean = float(e.mean())
    e_min, e_max = float(e.min()), float(e.max())
    e_spread = e_max - e_min
    e_bias = e_mean - E_TABLE_MEASURED

    v_finite = v[np.isfinite(v)]
    distinct = int(np.unique(np.round(v_finite, 9)).size) if v_finite.size else 0
    span = float(v_finite.max() - v_finite.min()) if v_finite.size else 0.0
    envelope_span = V_N_ENVELOPE[1] - V_N_ENVELOPE[0]
    covered = (distinct >= E_MIN_DISTINCT_IMPACT_SPEEDS
               and span >= E_COVERAGE_FRACTION * envelope_span)

    gates: list[dict] = []

    # 1 -- physics sanity floor.  Weak by construction; say so in the receipt.
    band_ok = bool(np.all((e >= E_ACCEPT[0]) & (e <= E_ACCEPT[1])))
    gates.append(_gate(
        "e_within_field_band",
        "every bounce lands inside what the venue measurement can resolve. "
        "This is a SANITY FLOOR, not evidence of a good inversion: the band is "
        f"{E_ACCEPT[1] - E_ACCEPT[0]:.4f} wide against a simulator spread of "
        "order 1e-4, so a badly mis-parameterised contact still passes it.",
        "PASS" if band_ok else "FAIL",
        statistic={"e_min": e_min, "e_max": e_max},
        limit=list(E_ACCEPT),
        band_width=E_ACCEPT[1] - E_ACCEPT[0],
        worlds_outside=int(np.count_nonzero(
            (e < E_ACCEPT[0]) | (e > E_ACCEPT[1]))),
    ))

    # 2 -- the inversion still lands on the authority.
    bias_ok = abs(e_bias) <= E_MEAN_TOL
    gates.append(_gate(
        "e_mean_matches_authority",
        "the mean bounce still lands on the venue number to within a third of "
        "the venue's own systematic. Reads a calibration drift, not a physics "
        "claim.",
        "PASS" if bias_ok else "FAIL",
        statistic={"e_mean": e_mean, "bias_vs_authority": e_bias},
        limit=E_MEAN_TOL,
        bias_in_field_sigma=e_bias / E_FIELD_SIGMA,
    ))

    # 3 -- the impact-speed spread is still integrator noise and nothing more.
    #
    # The statistic is max(measured, closed form) on purpose.  The *measured*
    # range grows with how many drop heights you sample -- 8 heights under-read
    # the true range -- so a sparse sweep could hide a big k.  The *closed form*
    # is sample-size independent but was derived for the bare geom contact and
    # under-reads the court scene by ~1.5x.  Each covers the other's blind spot,
    # so the gate takes the worse of the two.
    e_spread_gated = max(e_spread, predicted_spread)
    if covered:
        spread_verdict = "PASS" if e_spread_gated <= E_SPREAD_TOL else "FAIL"
        spread_reason = ""
    else:
        spread_verdict = "NOT_MEASURED"
        spread_reason = (
            f"the run swept {distinct} distinct impact speed(s) over "
            f"{span:.3f} m/s; the envelope is {V_N_ENVELOPE[0]}-"
            f"{V_N_ENVELOPE[1]} m/s and this gate needs at least "
            f"{E_MIN_DISTINCT_IMPACT_SPEEDS} distinct speeds covering "
            f"{E_COVERAGE_FRACTION:.0%} of it. At one drop height the observed "
            "spread is mujoco-warp non-determinism, not the impact-speed "
            "artefact this gate polices.")
    gates.append(_gate(
        "impact_speed_spread_is_integrator_artefact_only",
        "how much e moves across the validated impact-speed envelope. The "
        "contact is designed to be speed-independent, so whatever is left is "
        "the Euler step seeing the ball at a random penetration -- a closed "
        "form, dt^2*ieff*imp*k. THIS IS THE GATE THAT CATCHES A WRONG k. The "
        "statistic is the worse of the measured range and the closed form: a "
        "sparse sweep under-reads the range, and the closed form under-reads "
        "the court scene, so neither alone is fail-closed.",
        spread_verdict,
        statistic={"e_spread_gated": e_spread_gated,
                   "e_spread_measured": e_spread,
                   "e_spread_closed_form": predicted_spread},
        limit=E_SPREAD_TOL,
        predicted_from_closed_form=predicted_spread,
        predicted_over_limit_x=predicted_spread / E_SPREAD_TOL,
        measured_over_limit_x=e_spread / E_SPREAD_TOL,
        k_used=float(k), ieff=ieff,
        not_measured_reason=spread_reason or None,
    ))

    # 4 -- no manufactured velocity dependence.
    slope = None
    if covered and e.size > 2:
        slope = float(np.polyfit(v, e, 1)[0])
        slope_verdict = ("PASS" if E_SLOPE_CI_PER_M_S[0] <= slope
                         <= E_SLOPE_CI_PER_M_S[1] else "FAIL")
        slope_reason = ""
    else:
        slope_verdict = "NOT_MEASURED"
        slope_reason = (
            "a slope needs more than one drop height. With a single impact "
            "speed there is no slope to report -- and reporting 0.0 would be "
            "read as 'measured, and flat', which is a different claim.")
    gates.append(_gate(
        "e_vs_v_n_slope_within_venue_ci",
        "the simulator must not invent a speed dependence the venue did not "
        "measure. The venue measured e flat in v_n; its own CI is the limit.",
        slope_verdict,
        statistic={"slope_per_m_s": slope},
        limit=list(E_SLOPE_CI_PER_M_S),
        v_n_span_measured=[float(v_finite.min()), float(v_finite.max())]
        if v_finite.size else None,
        v_n_envelope=list(V_N_ENVELOPE),
        not_measured_reason=slope_reason or None,
    ))

    failed = [g["gate"] for g in gates if g["verdict"] == "FAIL"]
    unmeasured = [g["gate"] for g in gates if g["verdict"] == "NOT_MEASURED"]
    if failed:
        verdict = "FAIL"
        plain = ("at least one acceptance gate fired; the ball's bounce is not "
                 "the calibrated one")
    elif unmeasured:
        verdict = "NOT_MEASURED"
        plain = ("nothing failed, but the gates that can actually fail were "
                 "never exercised -- this run is NOT evidence that the ball "
                 "bounce is verified")
    else:
        verdict = "PASS"
        plain = ("every gate ran and passed over the full validated "
                 "impact-speed envelope")

    return {
        "verdict": verdict,
        "verdict_plain": plain,
        "context": context,
        "gates": gates,
        "failed_gates": failed,
        "not_measured_gates": unmeasured,
        "provenance": provenance,
        "summary": {
            "e_mean": e_mean, "e_min": e_min, "e_max": e_max,
            "e_spread": e_spread, "bias_vs_authority": e_bias,
            "slope_per_m_s": slope,
            "envelope_covered": bool(covered),
        },
        "independent_samples": {
            "n_worlds": n_worlds,
            "n_bounces_analysed": int(e.size),
            "n_distinct_impact_speeds": distinct,
            "worlds_are_not_independent_samples": True,
            "independent_dof_for_e_mean": 1,
            "independent_conditions_for_e_vs_v_n": distinct,
            "what_varies_between_worlds_at_one_drop_height":
                "mujoco-warp scheduling non-determinism only -- no measurement "
                "noise, no parameter sampling, no random seed enters this "
                "probe",
            "plain":
                f"{n_worlds} worlds is {n_worlds} repeats of a deterministic "
                "drop, not "
                f"{n_worlds} samples. The mean has ONE independent degree of "
                "freedom however many worlds run; only the number of DISTINCT "
                "impact speeds buys information, and only about the e-vs-v_n "
                "curve. Do not quote a standard error computed over worlds.",
        },
    }


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

    verdict = restitution_verdict(
        e_h, v_n, k=k, mu=args.mu, condim=args.condim, cone=args.cone,
        n_worlds=len(bounces),
        context=f"calibrate_restitution --confirm, k={k:g}, b={b:g}, "
                f"cone={args.cone}, {len(bounces)} drop heights")

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
        "e_height_spread": float(np.nanmax(e_h) - np.nanmin(e_h)),
        "e_height_slope_per_m_s": float(slope),
        "e_height_at_v_n_3": float(slope * 3.0 + intercept),
        "in_accept_band": bool(np.all((e_h >= E_ACCEPT[0]) & (e_h <= E_ACCEPT[1]))),
        "restitution_acceptance": verdict,
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
    return effective_impedance(args.mu, condim=args.condim, cone=args.cone)


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
    p.add_argument("--calibrated-b", action="store_true",
                   help="use the SHIPPED b recipe (analytic seed + the one "
                        "measured correction) instead of the bare analytic "
                        "seed; this is what a3_court_env.py actually runs, so "
                        "it is what a mutation test must mutate against")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    mu_eff = args.mu if args.mu is not None else BALL_FRICTION[0]
    if args.b is None:
        if args.calibrated_b:
            args.b = calibrated_b(E_TABLE_MEASURED, args.k, mu=mu_eff)
        else:
            args.b = analytic_seed_b(E_TABLE_MEASURED, args.k, mu=mu_eff,
                                     condim=args.condim)

    payload: dict = {
        "measured_authority": {
            "e_table_venue": E_TABLE_MEASURED,
            "e_table_optitrack": E_TABLE_OPTITRACK,
            "accept_band": list(E_ACCEPT),
            "accept_band_replaced": list(E_ACCEPT_INCUMBENT),
            "accept_band_rule": f"authority +- {E_ACCEPT_N_SIGMA:g} x field "
                                f"sigma {E_FIELD_SIGMA:g}, clamped inward "
                                "against the incumbent so nothing loosens",
            "field_sigma": E_FIELD_SIGMA,
            "calibration_tolerance": E_CAL_TOL,
            "e_mean_tolerance": E_MEAN_TOL,
            "e_spread_tolerance": E_SPREAD_TOL,
            "slope_ci_per_m_s": list(E_SLOPE_CI_PER_M_S),
            "v_n_envelope": list(V_N_ENVELOPE),
            "a_t_table": A_T_TABLE_MEASURED,
            "b_recipe": "calibrated_b (shipped)" if args.calibrated_b
            else "analytic_seed_b",
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
            "impact_speed_spread_predicted": discretization_spread(
                args.k, args.mu, condim=args.condim, cone=args.cone),
        },
    }
    rc = 0
    try:
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
    except BaseException as exc:                       # noqa: BLE001
        # A run that dies still has to leave a receipt behind, otherwise the
        # only runs without evidence are the ones that needed it most.
        payload["status"] = "crashed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        if args.json_out:
            with open(args.json_out, "w") as fh:
                fh.write(json.dumps(payload, indent=2, default=str))
        raise

    # Fail-closed.  `--confirm` is the mode that claims "the ball bounces
    # right", so it is the mode that must be able to say NO.  PASS is the only
    # verdict that exits 0: NOT_MEASURED is not a pass.
    conf = payload.get("confirm")
    if conf is not None:
        acc = conf["restitution_acceptance"]
        payload["status"] = f"restitution_{acc['verdict'].lower()}"
        payload["restitution_verdict"] = acc["verdict"]
        if acc["verdict"] == "FAIL":
            rc = 4
        elif acc["verdict"] == "NOT_MEASURED":
            rc = 5
    else:
        payload["status"] = "no_confirm_mode_no_restitution_verdict"

    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text)
    if rc:
        print(f"RESTITUTION_ACCEPTANCE {payload['restitution_verdict']}: "
              f"{payload['confirm']['restitution_acceptance']['verdict_plain']}",
              file=sys.stderr)
        for g in payload["confirm"]["restitution_acceptance"]["gates"]:
            if g["verdict"] != "PASS":
                print(f"  [{g['verdict']}] {g['gate']}: "
                      f"stat={g['statistic']} limit={g['limit']}",
                      file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
