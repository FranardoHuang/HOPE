"""Shadow strike-spec solver harness: double-run a SECOND solver next to the
production fast path and reconcile the two, tick by tick.

This is the wiring shell for the torch batched strike-spec solver (NOW.md
"拍面反解 torch 版上部署"): before any new solver is allowed near the command
path it must SHADOW the production FastStrikeSpecPlanner on live ticks — same
inputs, its own solve, per-field diffs to a CSV — exactly the promotion route
the EKF took (node.py use_kalman: "shadow-run the EKF next to the polyfit
estimator"). The planner still ACTS only on the production solve; the shadow
result is never published anywhere except diagnostics and the CSV.

ZERO rclpy in this module (pure python + numpy): the node owns all ROS I/O and
hands the harness plain arrays/floats, so the backends and the reconciliation
logic are unit-testable on any machine and reusable by offline replay tools.

I/O CONTRACT for backends (this is what the torch solver must implement)
------------------------------------------------------------------------
``SolverBackend.solve(problem: ShadowProblem) -> Optional[StrikeSpec]``

ShadowProblem is a frozen snapshot of ONE production solve call — the exact
argument list of FastStrikeSpecPlanner.solve_fast_spec as issued by
node._poses_cb (captured BEFORE the production solve mutates any warm state):

  p_ball, v_ball   (3,) float  predicted ball state AT the strike, world frame
                   (planner.strike_target — Stage-2 output, z=0 = table surface)
  omega_ball       (3,) float or None; None = spin-blind zeros (the production
                   path is currently spin-blind, so shadows see None today)
  target_land_xy   (2,) float  desired first-bounce point (per-side aim already
                   resolved by the node — the shadow sees the FINAL target)
  racket_speed_budget  float, m/s cap on |v_r|
  max_iter         int or None; the node's warm/cold budget rule, pre-applied:
                   strike_spec_max_iter when a warm start exists, None (=solver
                   default budget) on a cold tick
  q0               (5,) float or None; the PRODUCTION warm start of THIS tick
                   (pitch_deg, yaw_deg, v_n, v_t1, v_t2). Echo must consume it
                   to be bit-identical. A batch/torch backend MAY ignore it —
                   whether it does is part of the contract to settle (see
                   open questions in the deploy notes).
  with_sensitivities  bool; production tick default is False (hot path)
  tol_m            float or None; landing-residual acceptance tolerance.
                   None (the production tick value today — the node never
                   overrides it) = the solver default, StrikeSpecPlanner.TOL_M
                   = 0.005 m. Snapshotted so that if the node ever starts
                   passing tol_m explicitly the shadow sees the same value.

Return: a strike_spec_planner.StrikeSpec (world frame, same field semantics),
or None for "no admissible spec" — None is a VALID answer and is reconciled
as such, not an error. Raising is also allowed: the harness catches every
exception, counts it and logs it in the CSV row; it never reaches the
command path.

ACCEPTANCE RULE behind None (pinned numbers — a backend must apply the SAME
rule itself, not return best-effort specs for the shell to filter; the shell
does NOT filter):
  - landing residual: accept iff |landing_xy - target_land_xy| < tol
    (strict <), tol = problem.tol_m if not None else
    StrikeSpecPlanner.TOL_M = 0.005 m;
  - speed budget: accept iff |v_r| <= racket_speed_budget + 1e-9
    (see strike_spec_fast.solve_fast / strike_spec_planner.solve).

CONSTRUCTION-TIME INPUTS: make_backend receives the production spec planner's
own physics (BallPhysics — venue fit: drag_k, e(u_n), Magnus), config
(PlannerConfig) and table (TableParams). PlannerConfig fields the production
solve actually consumes (a non-integrator backend may ignore the dt_* ones,
everything else is solution-relevant):
  delta_t_flight, C_r, e_exp_g1, e_exp_g2   -> _initial_guess / v_n seed
  k_m                                        -> Magnus in paddle contact
  dt_integrate, dt_integrate_coarse,
  max_predict_time                           -> flight integrator only

Reconciliation CSV (one row per double-run; ShadowHarness.CSV_FIELDS):
  tick_t                  mocap stamp of the tick (s)
  prod_wall_ms / shadow_wall_ms   wall time of each solve
  prod_valid / shadow_valid       1 = returned a spec, 0 = returned None
  d_n_l2                  face-normal L2 diff (unitless, unit vectors)
  d_v_r_l2_mps            racket contact velocity L2 diff (m/s)
  d_landing_xy_l2_m       predicted landing point L2 diff (m)
  d_landing_time_s        |flight-time difference| (s)
  shadow_exception        "" or "ExcType: message" (row still written)
Diffs are NaN whenever either side returned None (valid flags disambiguate).

LATENCY / BLOCKING CAVEAT (shadow mode is a VALIDATION mode, not free):
the double-run and the CSV write are SYNCHRONOUS inside the node's /poses
callback. Exceptions are contained, but blocking is not — a hung filesystem
blocks write()/flush() WITHOUT raising OSError, stalling command publication.
Therefore shadow_log_path MUST point at fast local disk (never NFS/SD/USB/
network mounts), and even on a healthy disk an echo shadow roughly doubles
per-solve-tick callback occupancy (~15 ms med / ~42 ms p90 per solve on the
benchmark), dropping extra 300 Hz mocap frames while it runs. A per-solve
wall budget (ShadowHarness wall_budget_s, default 0.25 s, auto-disable after
max_over_budget=10 over-budget solves — same self-disable pattern as
ShadowRecorder) caps how long a slow backend can keep doing that, but cannot
interrupt a single blocked write.

BYTE-IDENTITY WHEN OFF: nothing in this module executes unless the node
constructs a harness, and node.py only does that when the use_shadow_solver
parameter (default False) is True — the flag check short-circuits before any
backend/recorder is built, before any timing call, before this module is even
imported. With the flag at its default the node's published topics, solve
sequence, warm-start state and diagnostics values are byte-identical to a tree
without this file.
"""

from __future__ import annotations

import abc
import csv
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import BallPhysics, PlannerConfig, TableParams
from .strike_spec_fast import FastStrikeSpecPlanner
from .strike_spec_planner import StrikeSpec


@dataclass(frozen=True)
class ShadowProblem:
    """Frozen snapshot of one production solve_fast_spec call (see module doc).

    The node builds this BEFORE running the production solve, so q0/max_iter
    reflect the warm state the production solve actually consumed this tick.
    """

    t: float                              # tick timestamp (mocap stamp, s)
    p_ball: np.ndarray                    # (3,) ball position at strike
    v_ball: np.ndarray                    # (3,) ball velocity at strike
    omega_ball: Optional[np.ndarray]      # (3,) spin or None = spin-blind
    target_land_xy: np.ndarray            # (2,) final landing target
    racket_speed_budget: float            # m/s cap on |v_r|
    max_iter: Optional[int]               # node's warm/cold budget rule, pre-applied
    q0: Optional[np.ndarray]              # (5,) production warm start or None
    with_sensitivities: bool              # production hot-path default: False
    tol_m: Optional[float] = None         # landing acceptance tol; None = solver
                                          # default StrikeSpecPlanner.TOL_M (5 mm)
                                          # — the node never overrides it today


class SolverBackend(abc.ABC):
    """A shadow solver: same problem in, StrikeSpec (or None) out."""

    #: short name for diagnostics / CSV provenance
    name: str = "abstract"

    @abc.abstractmethod
    def solve(self, problem: ShadowProblem) -> Optional[StrikeSpec]:
        """May return None (no admissible spec) or raise (harness catches)."""


class EchoBackend(SolverBackend):
    """Re-run the PRODUCTION solver on the snapshot: validates the pipeline.

    Own FastStrikeSpecPlanner instance (no state shared with the node's) built
    from the same physics/config/table, fed the same snapshot incl. the same
    warm start — solve_fast is deterministic numpy, so every diff column must
    be exactly 0.0 whenever both sides return a spec, and the valid flags must
    always agree. Any nonzero diff means the SNAPSHOT is wrong (an input the
    node failed to capture), which is precisely what this backend exists to
    catch before the torch backend makes diffs meaningful.
    """

    name = "echo"

    def __init__(
        self,
        physics: Optional[BallPhysics] = None,
        config: Optional[PlannerConfig] = None,
        table: Optional[TableParams] = None,
    ):
        self._solver = FastStrikeSpecPlanner(physics=physics, config=config, table=table)

    def solve(self, problem: ShadowProblem) -> Optional[StrikeSpec]:
        return self._solver.solve_fast_spec(
            problem.p_ball,
            problem.v_ball,
            problem.omega_ball,
            problem.target_land_xy,
            problem.racket_speed_budget,
            max_iter=problem.max_iter,
            tol_m=problem.tol_m,
            q0=problem.q0,
            with_sensitivities=problem.with_sensitivities,
        )


class TorchBackend(SolverBackend):
    """Placeholder for claude's torch batched strike-spec solver.

    Deliberately fail-loud AT CONSTRUCTION (not first solve): a launch that
    selects shadow_solver_backend:=torch before the solver lands must die at
    node startup with this message, never run silently echo-less.
    """

    name = "torch"

    # Startup-facing message: MUST stay English — launch operators cross lanes
    # (franco runs deploy profiles) and every runtime logger/exception string in
    # this package is English by convention. (中文注释可以,报错字符串不行。)
    _NOT_LANDED = (
        "TorchBackend: the torch batched strike-spec solver has not landed yet. "
        "I/O contract: module docstring of hope_planner/shadow_solver.py — "
        "solve(ShadowProblem) -> Optional[StrikeSpec], world frame, unit normal, "
        "None = no admissible spec. Until it lands, launch with "
        "shadow_solver_backend:=echo to validate the shadow wiring."
    )

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self._NOT_LANDED)

    def solve(self, problem: ShadowProblem) -> Optional[StrikeSpec]:
        # unreachable (construction raises); concrete override keeps the class
        # instantiable as far as abc is concerned so OUR message is what a
        # torch launch dies with, not abc's generic TypeError
        raise NotImplementedError(self._NOT_LANDED)


def make_backend(
    name: str,
    physics: Optional[BallPhysics] = None,
    config: Optional[PlannerConfig] = None,
    table: Optional[TableParams] = None,
) -> SolverBackend:
    """Backend factory for the node's shadow_solver_backend parameter."""
    if name == "echo":
        return EchoBackend(physics=physics, config=config, table=table)
    if name == "torch":
        return TorchBackend(physics=physics, config=config, table=table)
    raise ValueError(
        f"unknown shadow_solver_backend '{name}' (expected 'echo' or 'torch')")


def spec_field_diffs(
    prod: Optional[StrikeSpec], shadow: Optional[StrikeSpec],
) -> dict:
    """Per-field reconciliation diffs between two specs (NaN if either is None).

    The four compared fields are the ones a consumer would act on: face normal
    (方向), racket contact velocity (速度), predicted landing point (位置) and
    flight time (时刻). Solver-internal fields (tilt parameterization, LM
    iterations) are deliberately NOT compared — two backends may reach the
    same physical spec through different parameterizations.
    """
    nan = float("nan")
    out = {
        "prod_valid": int(prod is not None),
        "shadow_valid": int(shadow is not None),
        "d_n_l2": nan,
        "d_v_r_l2_mps": nan,
        "d_landing_xy_l2_m": nan,
        "d_landing_time_s": nan,
    }
    if prod is None or shadow is None:
        return out
    out["d_n_l2"] = float(np.linalg.norm(np.asarray(prod.n) - np.asarray(shadow.n)))
    out["d_v_r_l2_mps"] = float(
        np.linalg.norm(np.asarray(prod.v_r) - np.asarray(shadow.v_r)))
    out["d_landing_xy_l2_m"] = float(
        np.linalg.norm(np.asarray(prod.landing_xy) - np.asarray(shadow.landing_xy)))
    out["d_landing_time_s"] = float(abs(prod.landing_time - shadow.landing_time))
    return out


class ShadowRecorder:
    """Append-only CSV writer for reconciliation rows; path None/'' = disabled.

    Lazy-opens on the first row (header written once for a fresh file), flushes
    per row so a crashed session keeps its data. Any I/O failure disables the
    recorder permanently (returns False; the harness counts it) — a full disk
    must never take down the shadow, let alone the node.

    WARNING: write_row runs synchronously in the node's /poses callback and a
    HUNG mount blocks instead of raising — the self-disable only catches
    OSError, not blocking. Log to fast LOCAL disk only (see the module
    docstring's latency caveat); never NFS/SD/USB/network mounts.
    """

    def __init__(self, path: Optional[str], fieldnames: list):
        self._path = path or None
        self._fieldnames = list(fieldnames)
        self._fh = None
        self._writer = None
        self._failed = False

    @property
    def enabled(self) -> bool:
        return self._path is not None and not self._failed

    def write_row(self, row: dict) -> bool:
        """True = written (or recorder disabled by config); False = I/O failure."""
        if self._path is None:
            return True  # logging not requested — not a failure
        if self._failed:
            return False
        try:
            if self._writer is None:
                self._fh = open(self._path, "a", newline="")
                self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
                if self._fh.tell() == 0:
                    self._writer.writeheader()
            self._writer.writerow(row)
            self._fh.flush()
            return True
        except OSError:
            self._failed = True
            try:
                if self._fh is not None:
                    self._fh.close()
            except OSError:
                pass
            self._fh = None
            self._writer = None
            return False

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
            self._writer = None


class ShadowHarness:
    """Owns the backend + recorder + counters; guarantees run() never raises.

    The node calls run() once per production solve, right after it, with the
    pre-solve snapshot and the production result. Everything that can fail
    (backend solve, diffing, CSV I/O) is contained here and surfaced only as
    counters (the node splits them into per-key diagnostics KeyValues;
    diag_value() remains as a compact one-line summary for logs/offline tools).

    WALL BUDGET: exceptions are contained but latency is not — the double-run
    is synchronous in the /poses callback, so a slow backend (e.g. torch
    JIT/graph warmup, 100 ms+ first solves) delays every subsequent command
    publish while it runs. Each shadow solve is therefore checked against
    wall_budget_s; after max_over_budget over-budget solves the harness
    disables itself PERMANENTLY (active -> False, run() becomes a no-op) —
    the same self-disable pattern as ShadowRecorder. The default budget
    (0.25 s) with max_over_budget=10 tolerates a one-off warmup spike but
    kills a backend that is persistently too slow to shadow at 50 Hz.
    """

    CSV_FIELDS = [
        "tick_t",
        "prod_wall_ms",
        "shadow_wall_ms",
        "prod_valid",
        "shadow_valid",
        "d_n_l2",
        "d_v_r_l2_mps",
        "d_landing_xy_l2_m",
        "d_landing_time_s",
        "shadow_exception",
    ]

    #: per-solve wall clock cap / how many over-budget solves before self-disable
    WALL_BUDGET_S = 0.25
    MAX_OVER_BUDGET = 10

    def __init__(
        self,
        backend: SolverBackend,
        log_path: Optional[str] = None,
        wall_budget_s: Optional[float] = WALL_BUDGET_S,
        max_over_budget: int = MAX_OVER_BUDGET,
    ):
        self.backend = backend
        self.recorder = ShadowRecorder(log_path, self.CSV_FIELDS)
        self.wall_budget_s = wall_budget_s   # None = no cap (offline replay)
        self.max_over_budget = int(max_over_budget)
        self.n_runs = 0
        self.n_shadow_exceptions = 0   # backend.solve raised
        self.n_record_failures = 0     # CSV row lost to I/O
        self.n_over_budget = 0         # shadow solves slower than wall_budget_s
        self.disabled_reason: Optional[str] = None  # set once, permanent
        self.last_diffs: Optional[dict] = None

    @property
    def active(self) -> bool:
        """False once the wall-budget self-disable tripped (permanent)."""
        return self.disabled_reason is None

    def run(
        self,
        problem: ShadowProblem,
        production_spec: Optional[StrikeSpec],
        production_wall_s: float,
    ) -> dict:
        """Double-run one tick; NEVER raises (contract with the command path)."""
        if self.disabled_reason is not None:
            return {}  # self-disabled: no backend call, no I/O, no timing
        try:
            return self._run_inner(problem, production_spec, production_wall_s)
        except Exception as exc:  # diff/record bug must not reach the node
            self.n_shadow_exceptions += 1
            return {"shadow_exception": f"{type(exc).__name__}: {exc}"}

    def _run_inner(self, problem, production_spec, production_wall_s) -> dict:
        self.n_runs += 1
        shadow_spec = None
        exc_text = ""
        t0 = time.perf_counter()
        try:
            shadow_spec = self.backend.solve(problem)
        except Exception as exc:  # any backend failure is data, not a crash
            self.n_shadow_exceptions += 1
            exc_text = f"{type(exc).__name__}: {exc}"
        shadow_wall_s = time.perf_counter() - t0
        if self.wall_budget_s is not None and shadow_wall_s > self.wall_budget_s:
            self.n_over_budget += 1
            if self.n_over_budget >= self.max_over_budget:
                # Permanent self-disable: a backend this slow is stealing the
                # /poses callback from the command path (see class docstring).
                self.disabled_reason = (
                    f"{self.n_over_budget} solves over the "
                    f"{self.wall_budget_s * 1e3:.0f} ms wall budget "
                    f"(last {shadow_wall_s * 1e3:.1f} ms)")

        diffs = spec_field_diffs(production_spec, shadow_spec)
        self.last_diffs = diffs
        row = {
            "tick_t": f"{problem.t:.6f}",
            "prod_wall_ms": f"{production_wall_s * 1e3:.3f}",
            "shadow_wall_ms": f"{shadow_wall_s * 1e3:.3f}",
            "prod_valid": diffs["prod_valid"],
            "shadow_valid": diffs["shadow_valid"],
            "d_n_l2": f"{diffs['d_n_l2']:.9e}",
            "d_v_r_l2_mps": f"{diffs['d_v_r_l2_mps']:.9e}",
            "d_landing_xy_l2_m": f"{diffs['d_landing_xy_l2_m']:.9e}",
            "d_landing_time_s": f"{diffs['d_landing_time_s']:.9e}",
            "shadow_exception": exc_text,
        }
        if not self.recorder.write_row(row):
            self.n_record_failures += 1
        return diffs

    def diag_value(self) -> str:
        """Compact one-line summary (logs / offline tools; the node publishes
        the same counters as per-key diagnostics KeyValues instead)."""
        d = self.last_diffs or {}
        d_land = d.get("d_landing_xy_l2_m", float("nan"))
        s = (
            f"backend={self.backend.name} runs={self.n_runs} "
            f"exc={self.n_shadow_exceptions} rec_fail={self.n_record_failures} "
            f"over_budget={self.n_over_budget} last_d_land_m={d_land:.3e}"
        )
        if self.disabled_reason is not None:
            s += f" DISABLED({self.disabled_reason})"
        return s
