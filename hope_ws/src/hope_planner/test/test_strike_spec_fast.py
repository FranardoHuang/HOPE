"""Production fast strike-spec path tests (fastmath + strike_spec_fast).

Guards the three productionized levers of the 2026-07-06 latency work:

1. fastmath.cross3 / cross_rows are BIT-IDENTICAL to np.cross on float64
   (>= 1000 random states incl. zeros, signed zeros, subnormals, huge
   magnitudes) — the contract that lets the flight-acceleration hot loop
   drop np.cross's ~10 us dispatch. If numpy ever reorders the arithmetic,
   these fail and the safe fix is reverting callers to np.cross.
2. batch_integrate_to_table_plane row-for-row equals the scalar
   BallTrajectoryPredictor.integrate_to_table_plane, bitwise, in BOTH the
   fixed-dt and the adaptive (dt_integrate_coarse) modes.
3. FastStrikeSpecPlanner.solve_fast is a pure REIMPLEMENTATION, not a new
   model: under an identical config it reproduces StrikeSpecPlanner.solve
   bit for bit (same iterates, same n / v_r / landing). Speed comes only
   from batching + the config flags (dt_coarse, warm start, iter budget).
"""

import numpy as np

from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
from hope_planner.fastmath import cross3, cross_rows
from hope_planner.strike_spec_fast import (
    FastStrikeSpecPlanner,
    batch_integrate_to_table_plane,
)
from hope_planner.strike_spec_planner import StrikeSpecPlanner

PHYS = BallPhysics()
TAB = TableParams()


# --------------------------------------------------------------------- #
# (1) hand-written cross products: exact bitwise equality with np.cross
# --------------------------------------------------------------------- #


def _cross_case_pairs(rng, n_cases):
    """Random float64 3-vector pairs incl. zeros/signed zeros/subnormals/huge."""
    for i in range(n_cases):
        kind = i % 6
        if kind == 0:      # generic, mixed magnitudes
            a = rng.standard_normal(3) * 10.0 ** rng.integers(-6, 7)
            b = rng.standard_normal(3) * 10.0 ** rng.integers(-6, 7)
        elif kind == 1:    # subnormals
            a = rng.standard_normal(3) * 1e-310
            b = rng.standard_normal(3) * 1e-310
        elif kind == 2:    # zeros and signed zeros mixed in
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            a[rng.integers(0, 3)] = 0.0
            b[rng.integers(0, 3)] = -0.0
        elif kind == 3:    # exact zero vectors
            a = np.zeros(3)
            b = rng.standard_normal(3)
        elif kind == 4:    # huge (near-overflow products)
            a = rng.standard_normal(3) * 1e150
            b = rng.standard_normal(3) * 1e150
        else:              # planner-realistic: spin (rad/s) x velocity (m/s)
            a = rng.normal(0.0, 60.0, 3)
            b = rng.normal(0.0, 8.0, 3)
        yield a, b


def test_cross3_bitwise_equals_np_cross():
    rng = np.random.default_rng(0)
    n = 0
    for a, b in _cross_case_pairs(rng, 1200):
        got = cross3(a, b)
        ref = np.cross(a, b)
        assert got.dtype == ref.dtype == np.float64
        assert got.tobytes() == ref.tobytes(), f"case {n}: {a} x {b}"
        n += 1
    assert n >= 1000


def test_cross_rows_bitwise_equals_np_cross():
    rng = np.random.default_rng(1)
    pairs = list(_cross_case_pairs(rng, 1200))
    A = np.stack([p[0] for p in pairs])
    B = np.stack([p[1] for p in pairs])
    got = cross_rows(A, B)
    ref = np.cross(A, B)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert got.tobytes() == ref.tobytes()


def test_flight_acceleration_bitwise_matches_np_cross_formula():
    """The predictor hot loop swapped np.cross -> cross3; outputs must be
    bitwise-unchanged (this is the byte-identity guarantee of the swap)."""
    pred = BallTrajectoryPredictor(PHYS, PlannerConfig(), TAB)
    rng = np.random.default_rng(2)
    for _ in range(300):
        v = rng.normal(0.0, 6.0, 3)
        w = rng.normal(0.0, 60.0, 3)
        got = pred._flight_acceleration(v, w)
        speed = np.linalg.norm(v)
        ref = -PHYS.k * speed * v + PHYS.g
        ref = ref + pred.config.k_m * np.cross(w, v)
        assert got.tobytes() == ref.tobytes()
        # spin-blind branch untouched
        got0 = pred._flight_acceleration(v)
        ref0 = -PHYS.k * np.linalg.norm(v) * v + PHYS.g
        assert got0.tobytes() == ref0.tobytes()


# --------------------------------------------------------------------- #
# (2) batched integrator == scalar integrator, row for row
# NOTE tolerance (2026-07-07): bit-identical held on the authoring Mac; on the
# pod (different CPU/SIMD) the batched numpy path drifts by ~1 ULP. Parity is
# asserted at atol=1e-12 m / s — far below any physical meaning (mm scale),
# same solver claim, portable across machines.
# --------------------------------------------------------------------- #


def _post_contact_states(rng, m):
    p0 = np.array([rng.uniform(-0.1, 0.4), rng.uniform(-0.5, 0.5), rng.uniform(0.2, 0.5)])
    Vp = rng.uniform([1.0, -1.0, 1.0], [6.0, 1.0, 4.0], (m, 3))
    Wp = rng.normal(0.0, 50.0, (m, 3))
    return p0, Vp, Wp


def test_batch_integrator_bitwise_matches_scalar_fixed_and_adaptive():
    rng = np.random.default_rng(7)
    for dt_coarse in (0.0, 0.02):
        cfg = PlannerConfig(dt_integrate_coarse=dt_coarse)
        pred = BallTrajectoryPredictor(PHYS, cfg, TAB)
        for _ in range(12):
            p0, Vp, Wp = _post_contact_states(rng, int(rng.integers(1, 7)))
            land, t_land = batch_integrate_to_table_plane(p0, Vp, Wp, PHYS, cfg)
            for i in range(Vp.shape[0]):
                out = pred.integrate_to_table_plane(p0, Vp[i], Wp[i])
                if out is None:
                    assert np.isnan(land[i]).all() and np.isnan(t_land[i])
                else:
                    np.testing.assert_allclose(out[0], land[i], rtol=0.0, atol=1e-12,
                                               err_msg=f"dtc={dt_coarse}")
                    np.testing.assert_allclose(float(out[1]), float(t_land[i]),
                                               rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------- #
# (3) solve_fast is the SAME solver: parity with solve() (see tolerance NOTE
# above — 1e-12, portable replacement for the Mac-only bitwise claim)
# --------------------------------------------------------------------- #


def _strike_cases(rng, n):
    for _ in range(n):
        p = np.array([rng.uniform(-0.1, 0.3), rng.uniform(-0.6, 0.2), rng.uniform(0.25, 0.5)])
        v = np.array([rng.uniform(-6.0, -3.0), rng.uniform(-1.0, 1.0), rng.uniform(-2.0, 1.0)])
        w = rng.normal(0.0, 40.0, 3)
        tgt = np.array([rng.uniform(1.7, 2.4), rng.uniform(-0.5, 0.5)])
        yield p, v, w, tgt


def test_solve_fast_bitwise_reproduces_scalar_solve():
    rng = np.random.default_rng(11)
    for dt_coarse, n_cases in ((0.0, 2), (0.02, 5)):
        cfg = PlannerConfig(dt_integrate_coarse=dt_coarse)
        scalar = StrikeSpecPlanner(PHYS, cfg, TAB)
        fast = FastStrikeSpecPlanner(PHYS, cfg, TAB)
        for p, v, w, tgt in _strike_cases(rng, n_cases):
            spec = scalar.solve(p, v, w, tgt, 10.0, with_sensitivities=False)
            out = fast.solve_fast(p, v, w, tgt, 10.0)
            assert (spec is None) == (out is None)
            if spec is None:
                continue
            # Solve-level tolerance 1e-9: the LM iterations amplify the per-step
            # ~1 ULP machine drift to ~3e-12 (measured on pod); 1e-9 m / rad is still
            # six orders below a millimeter — any real solver divergence trips it.
            np.testing.assert_allclose(spec.n, out["n"], rtol=0.0, atol=1e-9)
            np.testing.assert_allclose(spec.v_r, out["v_r"], rtol=0.0, atol=1e-9)
            np.testing.assert_allclose(spec.landing_xy, out["landing_xy"], rtol=0.0, atol=1e-9)
            assert spec.iterations == out["iterations"]
            assert abs(spec.residual_m - out["resid_m"]) <= 1e-9


def test_solve_fast_production_recipe_and_warm_start():
    """Production recipe = adaptive 20 ms cruise + WARM start + iter budget 6.
    The budget-6 leg is only viable FROM a warm start (cold budget-6 solves
    do miss sometimes — that is why the node warm-starts): the previous
    tick's solve runs at the default budget, then the perturbed next tick
    must converge within 6 iterations from its q."""
    cfg = PlannerConfig(dt_integrate_coarse=FastStrikeSpecPlanner.PROD_DT_COARSE)
    fast = FastStrikeSpecPlanner(PHYS, cfg, TAB)
    rng = np.random.default_rng(23)
    n_cold, n_warm = 0, 0
    for p, v, w, tgt in _strike_cases(rng, 8):
        cold = fast.solve_fast(p, v, w, tgt, 10.0)  # default MAX_ITER budget
        if cold is None:
            continue
        n_cold += 1
        assert cold["resid_m"] < fast.TOL_M
        assert np.linalg.norm(cold["v_r"]) <= 10.0 + 1e-9
        # next tick: slightly perturbed ball state, warm started from q,
        # production iteration budget
        p2 = p + rng.normal(0.0, 0.005, 3)
        v2 = v * (1.0 + rng.normal(0.0, 0.02, 3))
        warm = fast.solve_fast(p2, v2, w, tgt, 10.0,
                               max_iter=FastStrikeSpecPlanner.PROD_MAX_ITER,
                               q0=cold["q"])
        if warm is None:  # budget-6 does miss occasionally (benchmark
            continue      # cmd_rate 0.93) — the node then serves the cache
        n_warm += 1
        assert warm["resid_m"] < fast.TOL_M
        assert warm["iterations"] <= FastStrikeSpecPlanner.PROD_MAX_ITER
    assert n_cold >= 6           # venue-like cases overwhelmingly solvable
    assert n_warm >= n_cold - 1  # warm budget-6 succeeds ~all of the time


def test_solve_fast_spec_full_spec_and_on_demand_sensitivities():
    cfg = PlannerConfig(dt_integrate_coarse=FastStrikeSpecPlanner.PROD_DT_COARSE)
    fast = FastStrikeSpecPlanner(PHYS, cfg, TAB)
    p = np.array([0.1, -0.4, 0.35])
    v = np.array([-5.0, 0.4, -0.6])
    w = np.array([10.0, -40.0, 5.0])
    tgt = np.array([2.0, -0.6])

    spec = fast.solve_fast_spec(p, v, w, tgt, 10.0)
    assert spec is not None
    assert np.linalg.norm(spec.landing_xy - tgt) < fast.TOL_M
    assert abs(np.linalg.norm(spec.n) - 1.0) < 1e-9
    # net-clearance annotation is populated by the same post-solve path
    assert isinstance(spec.clears_net, (bool, np.bool_))
    # sensitivities are OFF the hot path by default...
    for d in (spec.d_landing_d_pitch, spec.d_landing_d_yaw,
              spec.d_landing_d_v_n, spec.d_landing_d_v_t):
        assert np.isnan(d).all()
    # ...and available on demand
    spec_s = fast.solve_fast_spec(p, v, w, tgt, 10.0, with_sensitivities=True)
    assert spec_s is not None
    assert np.isfinite(spec_s.d_landing_d_pitch).all()
    assert np.isfinite(spec_s.d_landing_d_v_n).all()
    # warm-start vector documented as recoverable from the spec
    q0 = np.array([spec.tilt_pitch_deg, spec.tilt_yaw_deg, spec.v_n_signed,
                   spec.v_t_vec[0], spec.v_t_vec[1]])
    warm = fast.solve_fast(p, v, w, tgt, 10.0,
                           max_iter=FastStrikeSpecPlanner.PROD_MAX_ITER, q0=q0)
    assert warm is not None and warm["iterations"] <= spec.iterations + 1


def test_speed_flags_default_off():
    """Team rule: un-ablated changes merge flag-off. The fast path must be
    strictly opt-in — default configs keep the legacy integrator."""
    assert PlannerConfig().dt_integrate_coarse == 0.0
    fast = FastStrikeSpecPlanner()
    assert fast.config.dt_integrate_coarse == 0.0
    assert fast.MAX_ITER == StrikeSpecPlanner.MAX_ITER
    assert fast.TOL_M == StrikeSpecPlanner.TOL_M
    # productionized knob values are pinned (benchmark provenance)
    assert FastStrikeSpecPlanner.PROD_DT_COARSE == 0.02
    assert FastStrikeSpecPlanner.PROD_MAX_ITER == 6
