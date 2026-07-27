"""Net-clearance annotation tests (strike_spec_planner.compute_net_clearance + StrikeSpec).

The helper re-integrates the post-strike flight with the planner's own venue model
(drag + gravity + Magnus, Euler @ config.dt_integrate) and interpolates the ball
CENTER height at the net plane x = table.net_x (planner HOPE frame: z = 0 at the
TABLE SURFACE); clears = z_at_net > net_height + ball_radius. The solver ANNOTATES
(clears_net / net_z_margin_m, computed post-solve) — it never rejects on the net.

ADVERSARIAL CROSS-CHECK (team rule for coordinate-adjacent deliveries): the flat
FAIL case is anchored to an INDEPENDENT hand-derived drag-parabola value, not to the
helper's own integrator — see the derivation in test_flat_trajectory_fails_net.
"""

from __future__ import annotations

import numpy as np

from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
from hope_planner.strike_spec_planner import StrikeSpecPlanner, compute_net_clearance

PHYS = BallPhysics()
CFG = PlannerConfig()
TAB = TableParams()
NET_CLEAR_Z = TAB.net_height + PHYS.radius  # 0.1525 + 0.02 = 0.1725 m, center-based


def _predictor() -> BallTrajectoryPredictor:
    return BallTrajectoryPredictor(PHYS, CFG, TAB)


# --------------------------------------------------------------------- #
# direct unit tests of the pure helper
# --------------------------------------------------------------------- #


def test_flat_trajectory_fails_net():
    """Flat 8 m/s drive from z = 0.25: crosses the net plane BELOW net top + R.

    Independent hand check (1-D quadratic drag horizontally + gravity with the
    first-order drag relief on the vertical channel; k = 0.1261, g = 9.81):
      a      = k v0                       = 1.0088 1/s
      t_net  = (e^{k x_net} - 1)/a        = 0.186932 s      (x_net = 1.37)
      drop0  = g t^2 / 2                  = 0.171399 m      (vacuum drop)
      dz1    = g (t^2/2 - I/a)            = 0.009860 m      (drag relief,
               I = int_0^t ln(1+a s) ds = [(1+at)(ln(1+at)-1)+1]/a)
      z_net ~= 0.25 - (drop0 - dz1)       = 0.088462 m  ->  margin ~= -0.084038 m
    The helper's fully-coupled integration gives margin = -0.084601 m — 0.56 mm
    from the hand value (the residual is the neglected |v|-vs-vx drag coupling,
    which the hand estimate correctly UNDER-drags). Both sit ~8.4 cm below the
    0.1725 m clearance plane: decisive FAIL, and the agreement pins the frame
    (z=0 = surface), net_x, ball-radius term and margin sign all at once.
    """
    clears, margin = compute_net_clearance(
        np.array([0.0, -0.7625, 0.25]), np.array([8.0, 0.0, 0.0]), None, _predictor()
    )
    assert clears is False
    assert margin < 0.0
    # anchor to the INDEPENDENT hand-derived number (3 mm tolerance covers the
    # analytically-neglected coupling, measured at 0.56 mm)
    assert abs(margin - (-0.084038)) < 0.003


def test_lofted_trajectory_clears_net():
    """A 3+3 m/s lofted return from z = 0.30 sails ~0.32 m above the clearance plane."""
    clears, margin = compute_net_clearance(
        np.array([0.0, -0.7625, 0.30]), np.array([3.0, 0.0, 3.0]), None, _predictor()
    )
    assert clears is True
    assert 0.25 < margin < 0.45  # measured 0.3174 m at venue params


def test_lands_short_of_net_is_nan():
    """A low flat ball that hits the table BEFORE x = net_x never crosses the net
    plane: clears False with margin nan (nan > 0 is False, so consumers gating on
    the margin stay consistent with the bool)."""
    clears, margin = compute_net_clearance(
        np.array([0.0, -0.7625, 0.10]), np.array([6.0, 0.0, 0.0]), None, _predictor()
    )
    assert clears is False
    assert np.isnan(margin)


def test_same_step_net_crossing_is_ordered_against_center_contact_plane():
    """When net and landing occur in one step, compare fractions against z=R.

    Start at center z=3 cm with a 2 cm ball and descend 2 cm in one 0.1 s
    step: physical contact is halfway through the step.  A net at x=4 cm is
    crossed first; one at x=6 cm is reached only after the ball has landed.
    The assertion is run through fixed and adaptive control flow.
    """
    physics = BallPhysics(
        k=0.0,
        g=np.zeros(3),
        radius=0.02,
    )
    p0 = np.array([0.0, 0.0, 0.03])
    v0 = np.array([1.0, 0.0, -0.2])
    for coarse in (0.0, 0.2):
        cfg = PlannerConfig(
            dt_integrate=0.1,
            dt_integrate_coarse=coarse,
            max_predict_time=0.4,
        )
        before = BallTrajectoryPredictor(
            physics, cfg, TableParams(net_x=0.04, net_height=0.0)
        )
        clears, margin = compute_net_clearance(p0, v0, None, before)
        assert clears is True
        assert margin > 0.0

        after = BallTrajectoryPredictor(
            physics, cfg, TableParams(net_x=0.06, net_height=0.0)
        )
        clears, margin = compute_net_clearance(p0, v0, None, after)
        assert clears is False
        assert np.isnan(margin)


def test_struck_past_net_plane_is_nan():
    """No +x crossing exists when the strike point is already at/past the net."""
    clears, margin = compute_net_clearance(
        np.array([TAB.net_x + 0.1, -0.7625, 0.30]), np.array([3.0, 0.0, 3.0]), None,
        _predictor(),
    )
    assert clears is False
    assert np.isnan(margin)


# --------------------------------------------------------------------- #
# end-to-end: solve() / solve_fixed_normal() populate the annotation
# --------------------------------------------------------------------- #


def test_solve_deep_target_annotates_clears_net():
    """A converged solve() on a deep opponent-half target must carry clears_net
    True with a positive margin, and the annotation must equal a direct helper
    re-run on the spec's own (v_plus, omega_plus) — one flight model, one number."""
    planner = StrikeSpecPlanner(physics=PHYS, config=CFG, table=TAB)
    p_strike = np.array([0.0, -0.7625, 0.3])
    v_ball = np.array([-4.0, 0.3, -0.8])
    spec = planner.solve(
        p_strike, v_ball, None, np.array([2.2, -0.5]), racket_speed_budget=10.0
    )
    assert spec is not None
    assert spec.landing_xy[0] > TAB.net_x  # deep target: the flight does cross the net
    assert spec.clears_net is True
    assert spec.net_z_margin_m > 0.0
    clears2, margin2 = compute_net_clearance(
        p_strike, spec.v_plus, spec.omega_plus, planner.predictor
    )
    assert clears2 is True
    assert np.isclose(margin2, spec.net_z_margin_m, atol=1e-12)


def test_solve_fixed_normal_annotates_clears_net():
    """The pinned-normal sibling assembles through the same _build_spec: annotated too."""
    planner = StrikeSpecPlanner(physics=PHYS, config=CFG, table=TAB)
    n = np.array([0.95, 0.05, 0.3])
    n = n / np.linalg.norm(n)
    spec = planner.solve_fixed_normal(
        np.array([0.0, -0.7625, 0.3]), np.array([-4.0, 0.3, -0.8]), None,
        np.array([2.3, -0.5]), racket_speed_budget=10.0, n_fixed=n,
    )
    assert spec is not None
    assert spec.clears_net is True
    assert spec.net_z_margin_m > 0.0
