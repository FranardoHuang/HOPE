"""Accuracy-vs-latency benchmark for every HOPE strike-planner candidate.

Motivation (2026-07-06): in the sim closed loop the strike command was observed
to cost ~0.4 s per solve — 20x the 20 ms policy tick. This benchmark measures,
on ONE fixed set of venue-realistic incoming balls (sampled from
configs/incoming_ball_venue.yaml at-strike stats, fixed seed):

  per-call wall-clock (median / p90, time.perf_counter, warmed)
  + ACCURACY through the venue oracle: each candidate's racket command
    (face normal n + contact-point velocity v_r) is replayed through the
    TRAINING-side venue physics (virtual_ball.predict_paddle_contact +
    coarse_landing, RK4 @ 2 ms) and scored on landing error vs the
    candidate's own intended target, net clearance, and failure rate.

Candidates
----------
  user_quick        RacketTargetPlanner.plan (Stage-3 mirror law + drag
                    shooting solve + net fallback), TRUE strike state given.
  jiayi_pipeline    "jiayi's planner" = the deployed 3-stage pipeline
                    (git ed5cca9, dongc1/Jiayi Zhu): BallTrajectoryPredictor
                    .predict (Stage 2, 1 kHz Euler, spin-blind legacy path)
                    + RacketTargetPlanner.plan (Stage 3) — exactly what
                    node.py runs per /poses tick. There is NO separate C++
                    planner: the AGI C++ runner only consumes
                    /racket/command_flat (pp_planner_conductor.py is pure
                    orchestration). Timed from a pre-strike ball state.
  strike_spec       StrikeSpecPlanner.solve (LM over tilt+v_n+v_t), defaults.
  fixed_normal      StrikeSpecPlanner.solve_fixed_normal at the mirror-law
                    seed normal (clip-locked-face mode).
  torch_batchK      strike_spec_torch.solve_strike_specs, amortized per
                    question at batch 1 and 64 (kernel-launch-bound).

Speed-up prototypes (all flag-parameterized; measured as extra table rows so
the speed/accuracy trade is a CURVE, not a guess)
-------------------------------------------------
  ss_iter8 / ss_nosens        reduced LM budget / skip sensitivity rollouts
  ss_warm                     warm start q0 from the previous tick's solution
  ss_adapt10 / ss_adapt20     远粗近细 adaptive integration
                              (PlannerConfig.dt_integrate_coarse = 10/20 ms
                              RK4 cruise + legacy 1 ms Euler near events)
  ss_twostage                 coarse-solve (5 ms Euler, tol 2 cm) ->
                              fine-polish (1 ms, warm start), SMASH-style
  ss_fastnp / ss_fastnp_a10   numpy-BATCHED LM probes (base + 5 Jacobian
                              rollouts integrated as one (6,3) batch)
  ss_combo                    adapt10 + warm + iter6 + nosens (per-tick mode)

PRODUCTION path (2026-07-06: the winner ss_fastnp_a20_warm was PRODUCTIONIZED
into hope_planner/strike_spec_fast.py — the fast rows below import THAT
shipped implementation, there is no benchmark-local fork anymore; flags in
node.py: use_fast_strike_spec / strike_spec_rate_hz / dt_integrate_coarse)
------------------------------------------------------------------------
  ss_fastnp_a20_warm          production solve_fast: batched probes + 20 ms
                              cruise + warm start + iter budget 6
  prod_solve_fast_spec        the node's actual call (solve_fast_spec):
                              adds StrikeSpec assembly + net-clear annotation
  tick_S1S2S3(legacy)         ONE whole node tick, deployed path:
                              S1 polyfit + S2 predict 1 kHz + RTP.plan
  tick_S1S2fastSS(prod)       ONE whole node tick, fast path: S1 polyfit +
                              S2 adaptive predict + solve_fast (warm, iter 6)
  --variants prod runs exactly these four next to the always-on baselines.

HONEST BOUNDARY: all timings are Mac CPU (this machine) — a proxy. Deploy
SoC / pod numbers need a follow-up run of the SAME command there:

  /opt/anaconda3/bin/python3 hope_ws/src/hope_planner/benchmarks/benchmark_planner_latency.py --n 200 --seed 0

(on the pod: the training venv python from repo root). The oracle and numpy
planners both score the physical ball-center contact plane z = ball radius.
`--oracle-plane surface` is retained only as a historical counterfactual for
quantifying the pre-2026-07-10 bare-surface convention bug.

Run:
  /opt/anaconda3/bin/python3 hope_ws/src/hope_planner/benchmarks/benchmark_planner_latency.py --n 200
  ... --profile        # component/line-level timing of the 0.4 s hot path
  ... --json out.json  # machine-readable rows
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)  # hope_ws/src/hope_planner
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_ROOT, "..", "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from hope_planner.ball_contact import predict_paddle_contact as np_contact  # noqa: E402
from hope_planner.ball_state_estimator import BallStateEstimator  # noqa: E402
from hope_planner.ball_trajectory_predictor import (  # noqa: E402
    BallTrajectoryPredictor,
    StrikeTarget,
)
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams  # noqa: E402
from hope_planner.racket_target_planner import RacketTargetPlanner  # noqa: E402
from hope_planner.strike_spec_fast import (  # noqa: E402
    FastStrikeSpecPlanner,
    batch_integrate_to_table_plane,
)
from hope_planner.strike_spec_planner import StrikeSpecPlanner  # noqa: E402

# Backward-compat alias: the batched integrator + fast planner were PROTOTYPED
# here (2026-07-06) and then productionized into hope_planner.strike_spec_fast;
# the benchmark now imports the ONE production implementation (no fork to
# drift). ss_fastnp_* rows therefore measure the shipped code path.
_batch_integrate_to_plane = batch_integrate_to_table_plane

TABLE_Y_MAX = 0.7625        # sim harness centers the table on the robot
NET_X = TableParams().net_x  # 1.37 (planner frame: z = 0 at the table surface)
NET_TOP = TableParams().net_height
SPEED_BUDGET = 10.0          # node.py racket_speed_budget default
INCOMING_YAML = os.path.join(_REPO_ROOT, "configs", "incoming_ball_venue.yaml")
VENUE_YAML = os.path.join(_REPO_ROOT, "configs", "ball_physics_venue.yaml")
_MDP_DIR = os.path.join(
    _REPO_ROOT, "hope_training", "whole_body_tracking", "source",
    "whole_body_tracking", "whole_body_tracking", "tasks", "tracking", "mdp",
)

try:  # torch is optional: without it the oracle + torch candidate are skipped
    import torch  # noqa: F401

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# venue-realistic scenarios
# --------------------------------------------------------------------------- #


@dataclass
class Scenario:
    """One incoming ball: TRUE state at the strike + a pre-strike observation."""

    p_strike: np.ndarray   # (3,) planner frame (z above table surface)
    v_strike: np.ndarray   # (3,)
    w_strike: np.ndarray   # (3,) rad/s
    target_xy: np.ndarray  # (2,) intended landing point (opponent half)
    p_pre: np.ndarray      # (3,) ball state `lookback` s BEFORE the strike
    v_pre: np.ndarray      # (3,)
    lookback: float        # s (post-bounce arc; used by the jiayi_pipeline candidate)


def load_incoming_stats(path: str = INCOMING_YAML) -> dict:
    import yaml

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return raw["at_strike"]


def sample_scenarios(n: int, seed: int, path: str = INCOMING_YAML) -> List[Scenario]:
    """Sample n at-strike states from the venue fit + back-integrate 0.25 s.

    Positions: uniform inside the fitted q10-q90 box (yaml x is net-relative
    -> planner x = NET_X + x_rel). Velocities: per-component normal
    (vel_mean/std) clipped to q10-q90. Spin: random axis, |w| ~ N(mean, 10)
    clipped to q10-q90. NOTE the fitted cross-correlations (corr(vx,vz)=-0.44
    etc.) are NOT applied — the yaml only mirrors the diagonal; the full MVN
    lives in the pod JSON (incoming_dist.json).
    """
    st = load_incoming_stats(path)
    rng = np.random.default_rng(seed)
    vel_mean = np.asarray(st["vel_mean"], float)
    vel_std = np.asarray(st["vel_std"], float)
    vq = st["vel_q10_q90"]
    vlo = np.array([vq["x"][0], vq["y"][0], vq["z"][0]])
    vhi = np.array([vq["x"][1], vq["y"][1], vq["z"][1]])
    pq = st["contact_pos_q10_q90"]
    plo = np.array([pq["x"][0], pq["y"][0], pq["z"][0]])
    phi = np.array([pq["x"][1], pq["y"][1], pq["z"][1]])
    w_lo, w_hi = st["spin_mag_q10_q90"]
    w_mean = float(st["spin_mag_mean"])

    # back-integration uses the SAME flight model (legacy config, scalar RK4)
    pred = BallTrajectoryPredictor(BallPhysics(), PlannerConfig(), TableParams(y_max=TABLE_Y_MAX))

    scenarios: List[Scenario] = []
    for _ in range(n):
        p_rel = rng.uniform(plo, phi)
        p = np.array([NET_X + p_rel[0], p_rel[1], p_rel[2]])
        v = np.clip(rng.normal(vel_mean, vel_std), vlo, vhi)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        w = axis * float(np.clip(rng.normal(w_mean, 10.0), w_lo, w_hi))
        tgt = np.array([rng.uniform(NET_X + 0.30, TableParams().length - 0.20),
                        rng.uniform(-0.50, 0.50)])

        # pre-strike state: integrate the SAME ODE backward (h < 0) until
        # 0.25 s or the post-bounce arc ends (z would dip under 2 cm).
        h = -5e-4
        pp, vv = p.copy(), v.copy()
        lookback = 0.0
        while lookback < 0.25:
            p_n, v_n = pred._rk4_flight_step(pp, vv, w, h)
            if p_n[2] <= 0.02:
                break
            pp, vv = p_n, v_n
            lookback += -h
        scenarios.append(Scenario(p, v, w, tgt, pp, vv, lookback))
    return scenarios


# --------------------------------------------------------------------------- #
# venue oracle (training-side virtual_ball, loaded by file — no Isaac import)
# --------------------------------------------------------------------------- #


def _load_virtual_ball():
    pkg = types.ModuleType("wbt_mdp_bench")
    pkg.__path__ = [_MDP_DIR]
    sys.modules.setdefault("wbt_mdp_bench", pkg)
    name = "wbt_mdp_bench.virtual_ball"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_MDP_DIR, "virtual_ball.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_strike_spec_torch():
    _load_virtual_ball()
    # strike_spec_torch does "from .virtual_ball import ..." — give it the package
    name = "wbt_mdp_bench.strike_spec_torch"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_MDP_DIR, "strike_spec_torch.py"),
        submodule_search_locations=None,
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "wbt_mdp_bench"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class VenueOracle:
    """Ground truth: venue-fit contact + RK4 flight from virtual_ball.py.

    plane="radius": landing extracted at z = ball radius (ball CENTER when the
    ball physically touches the table — the venue extraction convention).
    plane="surface": z = 0 (historical counterfactual; not physical contact).
    """

    def __init__(self, plane: str = "radius", h: float = 0.002, horizon: float = 1.2):
        if not HAVE_TORCH:
            raise RuntimeError("torch unavailable — the venue oracle needs it")
        self.vb = _load_virtual_ball()
        self.prm = self.vb.load_venue_params(VENUE_YAML)
        self.surface_z = self.prm.ball_radius if plane == "radius" else 0.0
        self.h = h
        self.n_steps = int(round(horizon / h))
        self.clear_z = NET_TOP + self.prm.ball_radius  # ball-center net clearance

    def score(
        self,
        scenarios: Sequence[Scenario],
        n_cmd: np.ndarray,
        v_r_cmd: np.ndarray,
        intended_xy: np.ndarray,
        produced: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        import torch

        N = len(scenarios)
        land_err = np.full(N, np.nan)
        net_margin = np.full(N, np.nan)
        landed = np.zeros(N, bool)
        net_ok = np.zeros(N, bool)
        idx = np.where(produced)[0]
        if idx.size == 0:
            return dict(land_err=land_err, net_margin=net_margin, landed=landed, net_ok=net_ok)

        P = torch.tensor(np.stack([scenarios[i].p_strike for i in idx]), dtype=torch.float64)
        V = torch.tensor(np.stack([scenarios[i].v_strike for i in idx]), dtype=torch.float64)
        W = torch.tensor(np.stack([scenarios[i].w_strike for i in idx]), dtype=torch.float64)
        Ncmd = torch.tensor(n_cmd[idx], dtype=torch.float64)
        Vr = torch.tensor(v_r_cmd[idx], dtype=torch.float64)
        v_plus, w_plus = self.vb.predict_paddle_contact(V, Vr, Ncmd, W, self.prm)
        out = self.vb.coarse_landing(
            P, v_plus, w_plus, self.prm,
            surface_z=self.surface_z, net_x=NET_X, h=self.h, n_steps=self.n_steps,
        )
        lxy = out["land_xy"].numpy()
        lv = out["land_valid"].numpy()
        nz = out["net_z"].numpy()
        nv = out["net_valid"].numpy()
        land_err[idx] = np.linalg.norm(lxy - intended_xy[idx], axis=1)
        land_err[idx[~lv]] = np.nan
        landed[idx] = lv
        net_margin[idx] = np.where(nv, nz - self.clear_z, np.nan)
        net_ok[idx] = nv & (nz > self.clear_z)
        return dict(land_err=land_err, net_margin=net_margin, landed=landed, net_ok=net_ok)


# --------------------------------------------------------------------------- #
# candidate harness
# --------------------------------------------------------------------------- #


@dataclass
class CandidateResult:
    name: str
    times_s: np.ndarray
    produced: np.ndarray
    n_cmd: np.ndarray
    v_r_cmd: np.ndarray
    intended_xy: np.ndarray
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _run_candidate(
    name: str,
    scenarios: Sequence[Scenario],
    call: Callable[[Scenario], Optional[Tuple[np.ndarray, np.ndarray]]],
    notes: str = "",
    warmup: int = 2,
) -> CandidateResult:
    """Time `call` per scenario (warmed). call -> (n, v_r) or None."""
    for s in scenarios[: min(warmup, len(scenarios))]:
        call(s)
    N = len(scenarios)
    times = np.zeros(N)
    produced = np.zeros(N, bool)
    n_cmd = np.zeros((N, 3))
    v_r = np.zeros((N, 3))
    tgt = np.zeros((N, 2))
    for i, s in enumerate(scenarios):
        t0 = time.perf_counter()
        out = call(s)
        times[i] = time.perf_counter() - t0
        tgt[i] = s.target_xy
        if out is not None:
            produced[i] = True
            n_cmd[i], v_r[i] = out
    return CandidateResult(name, times, produced, n_cmd, v_r, tgt, notes)


def run_user_quick(scenarios: Sequence[Scenario]) -> CandidateResult:
    """(a) the user's quick mirror-law planner (Stage 3 only, true strike state)."""
    physics, config, table = BallPhysics(), PlannerConfig(), TableParams(y_max=TABLE_Y_MAX)
    rtp = RacketTargetPlanner(physics, config, table)

    def call(s: Scenario):
        config.target_land = np.array([s.target_xy[0], s.target_xy[1], 0.0])
        strike = StrikeTarget(
            p_ball=s.p_strike.copy(), v_ball=s.v_strike.copy(),
            t_strike=0.4, num_bounces=1, valid=True,
        )
        cmd = rtp.plan(strike)
        if cmd is None or not cmd.valid:
            return None
        return cmd.n_racket, cmd.v_racket

    return _run_candidate("user_quick(rtp)", scenarios, call,
                          notes="spin-blind mirror law; true strike state")


def run_jiayi_pipeline(scenarios: Sequence[Scenario]) -> CandidateResult:
    """(d) jiayi's planner = the deployed Stage-2+3 pipeline, per-tick cost.

    predict() from a pre-strike observation (spin-blind — the legacy node
    path passes no omega), then plan(). x_hit is set to the true strike
    plane per scenario (the node's adaptive-x_hit mode). Timed together:
    this is what node.py pays on EVERY /poses tick (up to 300 Hz).
    """
    physics, config, table = BallPhysics(), PlannerConfig(), TableParams(y_max=TABLE_Y_MAX)
    pred = BallTrajectoryPredictor(physics, config, table)
    rtp = RacketTargetPlanner(physics, config, table)
    intercept_err: List[float] = []

    def call(s: Scenario):
        config.x_hit = float(s.p_strike[0])
        config.target_land = np.array([s.target_xy[0], s.target_xy[1], 0.0])
        strike = pred.predict(s.p_pre.copy(), s.v_pre.copy(), 0.0)
        if not strike.valid:
            return None
        cmd = rtp.plan(strike)
        if cmd is None or not cmd.valid:
            return None
        intercept_err.append(float(np.linalg.norm(strike.p_ball - s.p_strike)))
        return cmd.n_racket, cmd.v_racket

    res = _run_candidate("jiayi_pipeline(S2+S3)", scenarios, call,
                         notes="deployed per-tick path (predict+plan), spin-blind")
    if intercept_err:
        res.extra["intercept_err_med_mm"] = 1e3 * float(np.median(intercept_err))
    return res


def run_strike_spec(
    scenarios: Sequence[Scenario],
    name: str = "strike_spec(LM)",
    dt_coarse: float = 0.0,
    max_iter: Optional[int] = None,
    tol_m: Optional[float] = None,
    warm: bool = False,
    with_sensitivities: bool = True,
    notes: str = "",
    seed: int = 1234,
) -> CandidateResult:
    """(b) + prototype variants of StrikeSpecPlanner.solve.

    warm=True emulates the previous tick: an UNTIMED solve on a perturbed
    copy of the scenario (p ±5 mm, v ±2 %) provides q0 for the timed solve.
    """
    physics = BallPhysics()
    config = PlannerConfig(dt_integrate_coarse=dt_coarse)
    table = TableParams(y_max=TABLE_Y_MAX)
    planner = StrikeSpecPlanner(physics, config, table)
    rng = np.random.default_rng(seed)
    iters: List[int] = []
    warm_q0: Dict[int, np.ndarray] = {}

    if warm:  # previous-tick solutions, prepared outside the timed section
        for i, s in enumerate(scenarios):
            p_prev = s.p_strike + rng.normal(0.0, 0.005, 3)
            v_prev = s.v_strike * (1.0 + rng.normal(0.0, 0.02, 3))
            spec = planner.solve(p_prev, v_prev, s.w_strike, s.target_xy, SPEED_BUDGET,
                                 with_sensitivities=False)
            if spec is not None:
                warm_q0[i] = np.array([spec.tilt_pitch_deg, spec.tilt_yaw_deg,
                                       spec.v_n_signed, spec.v_t_vec[0], spec.v_t_vec[1]])
    index = {id(s): i for i, s in enumerate(scenarios)}

    def call(s: Scenario):
        q0 = warm_q0.get(index.get(id(s))) if warm else None
        spec = planner.solve(s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET,
                             max_iter=max_iter, tol_m=tol_m, q0=q0,
                             with_sensitivities=with_sensitivities)
        if spec is None:
            return None
        iters.append(spec.iterations)
        return spec.n, spec.v_r

    res = _run_candidate(name, scenarios, call, notes=notes)
    if iters:
        res.extra["lm_iters_med"] = float(np.median(iters))
    return res


def run_fixed_normal(scenarios: Sequence[Scenario]) -> CandidateResult:
    """(c) solve_fixed_normal at the mirror-law seed normal (clip-locked face)."""
    physics, config, table = BallPhysics(), PlannerConfig(), TableParams(y_max=TABLE_Y_MAX)
    planner = StrikeSpecPlanner(physics, config, table)

    def call(s: Scenario):
        phi0, theta0, _ = planner._initial_guess(s.p_strike, s.v_strike, s.target_xy)
        n_fixed = planner._normal_from_tilt(phi0, theta0, 0.0, 0.0)
        spec = planner.solve_fixed_normal(
            s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET, n_fixed)
        if spec is None:
            return None
        return spec.n, spec.v_r

    return _run_candidate("fixed_normal(LM3)", scenarios, call,
                          notes="face pinned at mirror-law normal")


def run_two_stage(scenarios: Sequence[Scenario]) -> CandidateResult:
    """(iv) coarse-solve (5 ms Euler, tol 2 cm) -> fine-polish (1 ms, warm)."""
    physics, table = BallPhysics(), TableParams(y_max=TABLE_Y_MAX)
    coarse = StrikeSpecPlanner(physics, PlannerConfig(dt_integrate=0.005), table)
    fine = StrikeSpecPlanner(physics, PlannerConfig(), table)

    def call(s: Scenario):
        rough = coarse.solve(s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET,
                             max_iter=10, tol_m=0.02, with_sensitivities=False)
        q0 = None
        if rough is not None:
            q0 = np.array([rough.tilt_pitch_deg, rough.tilt_yaw_deg,
                           rough.v_n_signed, rough.v_t_vec[0], rough.v_t_vec[1]])
        spec = fine.solve(s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET,
                          max_iter=6, q0=q0, with_sensitivities=False)
        if spec is None:
            return None
        return spec.n, spec.v_r

    return _run_candidate("ss_twostage(5ms->1ms)", scenarios, call,
                          notes="coarse tol 2 cm then 6-iter fine polish")


# --------------------------------------------------------------------------- #
# (iii) numpy-batched LM probes — the PRODUCTION implementation
# (hope_planner.strike_spec_fast.FastStrikeSpecPlanner, imported above)
# --------------------------------------------------------------------------- #


def run_fast_np(
    scenarios: Sequence[Scenario],
    name: str,
    dt_coarse: float = 0.0,
    max_iter: Optional[int] = None,
    warm: bool = False,
    notes: str = "",
    seed: int = 1234,
) -> CandidateResult:
    physics = BallPhysics()
    config = PlannerConfig(dt_integrate_coarse=dt_coarse)
    planner = FastStrikeSpecPlanner(physics, config, TableParams(y_max=TABLE_Y_MAX))
    rng = np.random.default_rng(seed)
    warm_q0: Dict[int, np.ndarray] = {}
    if warm:  # previous-tick solutions, prepared outside the timed section
        for i, s in enumerate(scenarios):
            p_prev = s.p_strike + rng.normal(0.0, 0.005, 3)
            v_prev = s.v_strike * (1.0 + rng.normal(0.0, 0.02, 3))
            out = planner.solve_fast(p_prev, v_prev, s.w_strike, s.target_xy, SPEED_BUDGET)
            if out is not None:
                warm_q0[i] = out["q"]
    index = {id(s): i for i, s in enumerate(scenarios)}

    def call(s: Scenario):
        q0 = warm_q0.get(index.get(id(s))) if warm else None
        out = planner.solve_fast(s.p_strike, s.v_strike, s.w_strike, s.target_xy,
                                 SPEED_BUDGET, max_iter=max_iter, q0=q0)
        if out is None:
            return None
        return out["n"], out["v_r"]

    return _run_candidate(name, scenarios, call, notes=notes)


# --------------------------------------------------------------------------- #
# production-path candidates (import the SHIPPED planner, node-shaped calls)
# --------------------------------------------------------------------------- #


def run_prod_spec(
    scenarios: Sequence[Scenario],
    name: str = "prod_solve_fast_spec",
    dt_coarse: float = FastStrikeSpecPlanner.PROD_DT_COARSE,
    max_iter: int = FastStrikeSpecPlanner.PROD_MAX_ITER,
    seed: int = 1234,
) -> CandidateResult:
    """The node's fast strike-spec call, timed end to end: solve_fast_spec
    with the production defaults (adaptive 20 ms cruise, warm start, iter 6,
    sensitivities OFF) INCLUDING full StrikeSpec assembly + the post-solve
    net-clearance annotation — i.e. what node.py pays per decimated solve
    when use_fast_strike_spec is on."""
    physics = BallPhysics()
    config = PlannerConfig(dt_integrate_coarse=dt_coarse)
    planner = FastStrikeSpecPlanner(physics, config, TableParams(y_max=TABLE_Y_MAX))
    rng = np.random.default_rng(seed)
    warm_q0: Dict[int, np.ndarray] = {}
    iters: List[int] = []
    for i, s in enumerate(scenarios):  # previous tick, untimed (same as run_fast_np)
        p_prev = s.p_strike + rng.normal(0.0, 0.005, 3)
        v_prev = s.v_strike * (1.0 + rng.normal(0.0, 0.02, 3))
        out = planner.solve_fast(p_prev, v_prev, s.w_strike, s.target_xy, SPEED_BUDGET)
        if out is not None:
            warm_q0[i] = out["q"]
    index = {id(s): i for i, s in enumerate(scenarios)}

    def call(s: Scenario):
        spec = planner.solve_fast_spec(
            s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET,
            max_iter=max_iter, q0=warm_q0.get(index.get(id(s))),
            with_sensitivities=False)
        if spec is None:
            return None
        iters.append(spec.iterations)
        return spec.n, spec.v_r

    res = _run_candidate(name, scenarios, call,
                         notes="node path: fast spec + net-clear annotation")
    if iters:
        res.extra["lm_iters_med"] = float(np.median(iters))
    return res


def run_full_tick(scenarios: Sequence[Scenario], fast: bool) -> CandidateResult:
    """One WHOLE node tick: S1 polyfit (push+estimate) + S2 predict + S3.

    Mocap samples are synthesized noise-free at config.mocap_hz along the
    same flight ODE from the pre-strike observation. Untimed setup fills the
    estimator window and (fast mode) runs the previous tick's solve for the
    warm start; the TIMED section is exactly one /poses tick: push newest
    sample -> estimate -> Stage-2 predict -> Stage-3 plan.

      fast=False: legacy deployed tick  = S1 + S2 (1 kHz Euler) + RTP.plan
      fast=True : production fast tick  = S1 + S2 (adaptive 20 ms cruise)
                  + FastStrikeSpecPlanner.solve_fast (warm, iter 6)

    Both S3 calls are spin-blind (omega None), matching today's node (no
    live spin estimate), so oracle accuracy for BOTH rows carries the real
    spin-blind Stage-2 error — compare their latency, not their mm."""
    physics, table = BallPhysics(), TableParams(y_max=TABLE_Y_MAX)
    n_iters: List[int] = []

    # --- untimed per-scenario prep: mocap history + previous-tick warm state ---
    preps: Dict[int, dict] = {}
    for i, s in enumerate(scenarios):
        config = PlannerConfig(
            x_hit=float(s.p_strike[0]),
            target_land=np.array([s.target_xy[0], s.target_xy[1], 0.0]),
            dt_integrate_coarse=FastStrikeSpecPlanner.PROD_DT_COARSE if fast else 0.0,
        )
        pred = BallTrajectoryPredictor(physics, config, table)
        h = 1.0 / config.mocap_hz
        p, v = s.p_pre.copy(), s.v_pre.copy()
        samples = [(0.0, p.copy())]
        for j in range(config.fit_window + 1):
            p, v = pred._rk4_flight_step(p, v, s.w_strike, h)
            if p[0] <= config.x_hit + 0.02 or p[2] <= 0.0:
                break
            samples.append((float((j + 1) * h), p.copy()))
        if len(samples) < config.fit_window + 1:
            continue  # ball reaches the plane inside the fit window -> no tick
        window = samples[:-1][-config.fit_window:]
        prep = dict(
            config=config, pred=pred,
            t_buf=[t for t, _ in window], p_buf=[q for _, q in window],
            z_hist=[float(q[2]) for _, q in window[-3:]],
            t_new=samples[-1][0], p_new=samples[-1][1],
            rtp=RacketTargetPlanner(physics, config, table),
            planner_fast=FastStrikeSpecPlanner(physics, config, table) if fast else None,
            warm_q=None,
        )
        if fast:  # previous tick's solve (window without the newest sample)
            est_prev = BallStateEstimator(config)
            for t_i, p_i in window:
                est_prev.push(t_i, p_i)
            p_e, v_e, t_e = est_prev.estimate()
            strike_prev = pred.predict(p_e, v_e, t_e)
            if strike_prev.valid:
                # DEFAULT iteration budget here: the previous tick converges
                # cold once, every later tick then rides the budget-6 warm
                # path (same pattern as the node: first solve of a rally is
                # cold, budget-6 only pays off warm).
                out_prev = prep["planner_fast"].solve_fast(
                    strike_prev.p_ball, strike_prev.v_ball, None,
                    s.target_xy, SPEED_BUDGET)
                if out_prev is not None:
                    prep["warm_q"] = out_prev["q"]
        preps[i] = prep
    index = {id(s): i for i, s in enumerate(scenarios)}

    def call(s: Scenario):
        prep = preps.get(index[id(s)])
        if prep is None:
            return None
        # Estimator state restore is idempotent (warmup calls reuse it) and
        # cheap (~us, list copies) next to the ms-scale tick being timed.
        est = BallStateEstimator(prep["config"])
        est.t_buffer = list(prep["t_buf"])
        est.p_buffer = list(prep["p_buf"])
        est._z_hist = list(prep["z_hist"])
        # --- the actual node tick: push -> estimate -> predict -> plan ---
        est.push(prep["t_new"], prep["p_new"])
        p_e, v_e, t_e = est.estimate()
        if v_e[0] >= 0.0:
            return None
        strike = prep["pred"].predict(p_e, v_e, t_e)
        if not strike.valid:
            return None
        if fast:
            out = prep["planner_fast"].solve_fast(
                strike.p_ball, strike.v_ball, None, s.target_xy, SPEED_BUDGET,
                max_iter=FastStrikeSpecPlanner.PROD_MAX_ITER, q0=prep["warm_q"])
            if out is None:
                return None
            n_iters.append(out["iterations"])
            return out["n"], out["v_r"]
        cmd = prep["rtp"].plan(strike)
        if cmd is None or not cmd.valid:
            return None
        return cmd.n_racket, cmd.v_racket

    name = "tick_S1S2fastSS(prod)" if fast else "tick_S1S2S3(legacy)"
    notes = ("full node tick, fast spec path (spin-blind S2)" if fast
             else "full node tick, deployed RTP path (spin-blind S2)")
    res = _run_candidate(name, scenarios, call, notes=notes)
    if n_iters:
        res.extra["lm_iters_med"] = float(np.median(n_iters))
    return res


def run_torch_batched(scenarios: Sequence[Scenario], batch: int) -> CandidateResult:
    """(e) strike_spec_torch batched; per-question amortized wall-clock."""
    import torch

    st = _load_strike_spec_torch()
    vb = _load_virtual_ball()
    prm = vb.load_venue_params(VENUE_YAML)
    N = len(scenarios)
    P = torch.tensor(np.stack([s.p_strike for s in scenarios]), dtype=torch.float64)
    V = torch.tensor(np.stack([s.v_strike for s in scenarios]), dtype=torch.float64)
    W = torch.tensor(np.stack([s.w_strike for s in scenarios]), dtype=torch.float64)
    T = torch.tensor(np.stack([s.target_xy for s in scenarios]), dtype=torch.float64)

    def solve_chunk(sl):
        return st.solve_strike_specs(
            P[sl], V[sl], W[sl], T[sl], prm,
            surface_z=float(prm.ball_radius),  # physical landing plane
            net_x=NET_X, speed_budget=SPEED_BUDGET,
        )

    solve_chunk(slice(0, min(batch, N)))  # warmup

    times = np.zeros(N)
    produced = np.zeros(N, bool)
    n_cmd = np.zeros((N, 3))
    v_r = np.zeros((N, 3))
    tgt = np.stack([s.target_xy for s in scenarios])
    for start in range(0, N, batch):
        sl = slice(start, min(start + batch, N))
        m = sl.stop - sl.start
        t0 = time.perf_counter()
        out = solve_chunk(sl)
        dt_amort = (time.perf_counter() - t0) / m
        times[sl] = dt_amort
        ok = out["ok"].numpy()
        produced[sl] = ok
        n_cmd[sl] = out["n"].numpy()
        v_r[sl] = out["v_r"].numpy()
    return CandidateResult(
        f"torch_batch{batch}", times, produced, n_cmd, v_r, tgt,
        notes=f"amortized per question over batches of {batch} (CPU)",
    )


# --------------------------------------------------------------------------- #
# summary / reporting
# --------------------------------------------------------------------------- #


def summarize(res: CandidateResult, score: Dict[str, np.ndarray]) -> Dict[str, object]:
    t_ms = res.times_s * 1e3
    prod = res.produced
    le = score["land_err"]
    ok_land = np.isfinite(le)
    row: Dict[str, object] = {
        "name": res.name,
        "med_ms": float(np.median(t_ms)),
        "p90_ms": float(np.percentile(t_ms, 90)),
        "max_ms": float(np.max(t_ms)),
        "cmd_rate": float(np.mean(prod)),
        "land_rate": float(np.mean(score["landed"][prod])) if prod.any() else float("nan"),
        "land_med_mm": float(1e3 * np.nanmedian(le)) if ok_land.any() else float("nan"),
        "land_p90_mm": float(1e3 * np.nanpercentile(le, 90)) if ok_land.any() else float("nan"),
        "net_ok_rate": float(np.mean(score["net_ok"][prod])) if prod.any() else float("nan"),
        "net_margin_med_mm": (
            float(1e3 * np.nanmedian(score["net_margin"]))
            if np.isfinite(score["net_margin"]).any() else float("nan")
        ),
        "notes": res.notes,
    }
    row.update({k: v for k, v in res.extra.items()})
    return row


_COLS = [
    ("name", 24), ("med_ms", 9), ("p90_ms", 9), ("cmd_rate", 9),
    ("land_med_mm", 12), ("land_p90_mm", 12), ("net_ok_rate", 12), ("notes", 0),
]


def format_table(rows: List[Dict[str, object]]) -> str:
    def fmt(v, width):
        if isinstance(v, float):
            s = f"{v:8.2f}" if abs(v) >= 0.01 or v == 0 else f"{v:8.4f}"
        else:
            s = str(v)
        return s.ljust(width) if width else s

    lines = ["  ".join(h.ljust(w) if w else h for h, w in _COLS)]
    lines.append("-" * 110)
    for r in rows:
        lines.append("  ".join(fmt(r.get(h, ""), w) for h, w in _COLS))
    return "\n".join(lines)


def run_benchmark(
    n: int = 200,
    seed: int = 0,
    oracle_plane: str = "radius",
    torch_batches: Sequence[int] = (1, 64),
    variants: str = "all",
) -> Tuple[List[Dict[str, object]], List[CandidateResult]]:
    """Run every candidate on one shared scenario set. Importable entry point."""
    scenarios = sample_scenarios(n, seed)
    oracle = VenueOracle(plane=oracle_plane)

    results: List[CandidateResult] = [
        run_user_quick(scenarios),
        run_jiayi_pipeline(scenarios),
        run_strike_spec(scenarios),                       # (b) baseline
        run_fixed_normal(scenarios),                      # (c)
    ]
    if variants == "all":
        results += [
            run_strike_spec(scenarios, "ss_iter8", max_iter=8,
                            notes="LM budget 8"),
            run_strike_spec(scenarios, "ss_nosens", with_sensitivities=False,
                            notes="skip sensitivity rollouts"),
            run_strike_spec(scenarios, "ss_warm", warm=True, max_iter=8,
                            notes="q0 from prev tick, budget 8"),
            run_strike_spec(scenarios, "ss_adapt10", dt_coarse=0.01,
                            notes="RK4 cruise 10 ms + fine 1 ms"),
            run_strike_spec(scenarios, "ss_adapt20", dt_coarse=0.02,
                            notes="RK4 cruise 20 ms + fine 1 ms"),
            run_two_stage(scenarios),
            run_fast_np(scenarios, "ss_fastnp", notes="batched numpy probes, 1 ms"),
            run_fast_np(scenarios, "ss_fastnp_a10", dt_coarse=0.01,
                        notes="batched probes + 10 ms cruise"),
            run_strike_spec(scenarios, "ss_combo", dt_coarse=0.01, warm=True,
                            max_iter=6, with_sensitivities=False,
                            notes="adapt10+warm+iter6+nosens (per-tick mode)"),
        ]
    if variants in ("all", "smoke", "prod"):
        results += [
            run_fast_np(scenarios, "ss_fastnp_a20_warm", dt_coarse=0.02,
                        max_iter=6, warm=True,
                        notes="PRODUCTION strike_spec_fast, a20+warm+iter6"),
        ]
    if variants in ("all", "smoke"):
        results += [
            run_fast_np(scenarios, "ss_fastnp_a40_warm", dt_coarse=0.04,
                        max_iter=6, warm=True,
                        notes="40 ms cruise: python per-call floor probe"),
        ]
    if variants in ("all", "prod"):
        results += [
            run_prod_spec(scenarios),
            run_full_tick(scenarios, fast=False),
            run_full_tick(scenarios, fast=True),
        ]
    if HAVE_TORCH:
        for b in torch_batches:
            results.append(run_torch_batched(scenarios, batch=b))

    rows = [summarize(r, oracle.score(scenarios, r.n_cmd, r.v_r_cmd,
                                      r.intended_xy, r.produced))
            for r in results]
    return rows, results


# --------------------------------------------------------------------------- #
# hot-loop profile (deliverable 2: name the loop, with numbers)
# --------------------------------------------------------------------------- #


def profile_hot_loop(seed: int = 0) -> str:
    """Component-level timing of the worst candidate (StrikeSpecPlanner.solve)."""
    import cProfile
    import io
    import pstats

    s = sample_scenarios(3, seed)[-1]
    physics, config, table = BallPhysics(), PlannerConfig(), TableParams(y_max=TABLE_Y_MAX)
    planner = StrikeSpecPlanner(physics, config, table)
    lines: List[str] = []

    # per-step python cost of the Euler inner loop
    n_steps = {"count": 0}
    orig_accel = planner.predictor._flight_acceleration

    def counting_accel(v, omega=None):
        n_steps["count"] += 1
        return orig_accel(v, omega)

    planner.predictor._flight_acceleration = counting_accel
    forwards = {"count": 0}
    orig_fwd = planner._forward

    def counting_fwd(*a, **k):
        forwards["count"] += 1
        return orig_fwd(*a, **k)

    planner._forward = counting_fwd

    planner.solve(s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET)  # warm
    n_steps["count"] = 0
    forwards["count"] = 0
    t0 = time.perf_counter()
    spec = planner.solve(s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET)
    dt_full = time.perf_counter() - t0
    steps_full, fwd_full = n_steps["count"], forwards["count"]

    n_steps["count"] = 0
    forwards["count"] = 0
    t0 = time.perf_counter()
    planner.solve(s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET,
                  with_sensitivities=False)
    dt_nosens = time.perf_counter() - t0
    steps_nosens, fwd_nosens = n_steps["count"], forwards["count"]

    lines.append("HOT LOOP: StrikeSpecPlanner.solve -> _forward -> "
                 "BallTrajectoryPredictor.integrate_to_table_plane (1 kHz python Euler)")
    if spec is not None:
        lines.append(f"  LM iterations                : {spec.iterations}")
    lines.append(f"  full solve                   : {dt_full*1e3:8.1f} ms, "
                 f"{fwd_full} forward rollouts, {steps_full} Euler/flight steps "
                 f"-> {dt_full/max(steps_full,1)*1e6:6.1f} us/step (python)")
    lines.append(f"  solve w/o sensitivities      : {dt_nosens*1e3:8.1f} ms, "
                 f"{fwd_nosens} rollouts, {steps_nosens} steps")
    lines.append(f"  sensitivity+netclear overhead: {(dt_full-dt_nosens)*1e3:8.1f} ms "
                 f"({steps_full-steps_nosens} steps)")

    # single rollout / stage-2 predict cost for scale
    planner.predictor._flight_acceleration = orig_accel
    pred = BallTrajectoryPredictor(physics, PlannerConfig(x_hit=float(s.p_strike[0])), table)
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        v_plus, w_plus = np_contact(s.v_strike, np.array([1.0, 0.0, 0.5]),
                                    np.array([0.9, 0.0, 0.44]), s.w_strike, physics, config)
        planner.predictor.integrate_to_table_plane(s.p_strike, v_plus, w_plus)
    lines.append(f"  one contact+flight rollout   : {(time.perf_counter()-t0)/reps*1e3:8.2f} ms")
    t0 = time.perf_counter()
    for _ in range(reps):
        pred.predict(s.p_pre.copy(), s.v_pre.copy(), 0.0)
    lines.append(f"  Stage-2 predict (per tick)   : {(time.perf_counter()-t0)/reps*1e3:8.2f} ms")
    pred_a = BallTrajectoryPredictor(
        physics, PlannerConfig(x_hit=float(s.p_strike[0]), dt_integrate_coarse=0.02), table)
    pred_a.predict(s.p_pre.copy(), s.v_pre.copy(), 0.0)
    t0 = time.perf_counter()
    for _ in range(reps):
        pred_a.predict(s.p_pre.copy(), s.v_pre.copy(), 0.0)
    lines.append(f"  Stage-2 predict, adapt 20 ms : {(time.perf_counter()-t0)/reps*1e3:8.2f} ms")

    buf = io.StringIO()
    pr = cProfile.Profile()
    pr.enable()
    planner.solve(s.p_strike, s.v_strike, s.w_strike, s.target_xy, SPEED_BUDGET)
    pr.disable()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(12)
    lines.append("")
    lines.append("cProfile (one solve, cumulative top 12):")
    lines.append(buf.getvalue())
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle-plane", choices=("radius", "surface"), default="radius")
    ap.add_argument("--torch-batches", type=int, nargs="*", default=[1, 64])
    ap.add_argument("--variants", choices=("all", "smoke", "prod", "none"), default="all")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args(argv)

    if args.profile:
        print(profile_hot_loop(args.seed))
        print()

    rows, _ = run_benchmark(
        n=args.n, seed=args.seed, oracle_plane=args.oracle_plane,
        torch_batches=tuple(args.torch_batches), variants=args.variants,
    )
    print(f"N={args.n} venue scenarios (seed {args.seed}), oracle: virtual_ball RK4@2ms, "
          f"plane={args.oracle_plane}; Mac CPU wall-clock (proxy — rerun on the deploy SoC).")
    print(format_table(rows))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"rows -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
