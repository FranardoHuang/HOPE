"""Shadow solver harness tests (shadow_solver.py) — ZERO rclpy, pure numpy.

Guards the four promises the shadow wiring makes before the torch backend
lands (NOW.md 拍面反解 torch 版上部署, shadow-first promotion route):

1. EchoBackend re-running the production FastStrikeSpecPlanner on the node's
   snapshot reconciles to diff EXACTLY 0.0 — the pipeline check that must be
   green before any nonzero diff can be blamed on a new solver — AND, as the
   negative control, the diff detector returns NONZERO for two genuinely
   different solves (so an all-zero CSV cannot come from a vacuous compare).
2. TorchBackend fail-louds AT CONSTRUCTION with a message pointing at the I/O
   contract (a torch launch before the solver lands must die at startup).
3. The harness NEVER raises: backend exceptions are swallowed, counted, and
   recorded in the CSV row; recorder I/O failures only bump a counter; a
   backend persistently over the wall budget self-disables permanently.
4. CSV row format: stable column set, one row per double-run, values parse.

node.py wiring (parameter short-circuit, snapshot-before-solve, diagnostics
KeyValue) imports rclpy and is NOT testable here — left for pod colcon test /
a live shadow session (see the branch notes).
"""

from __future__ import annotations

import csv
import math
import time

import numpy as np
import pytest

from hope_planner.constants import PlannerConfig
from hope_planner.shadow_solver import (
    EchoBackend,
    ShadowHarness,
    ShadowProblem,
    SolverBackend,
    TorchBackend,
    make_backend,
    spec_field_diffs,
)
from hope_planner.strike_spec_fast import FastStrikeSpecPlanner


def _production_setup():
    """Production-like fast planner (node defaults: adaptive 20 ms cruise)."""
    cfg = PlannerConfig(dt_integrate_coarse=FastStrikeSpecPlanner.PROD_DT_COARSE)
    return FastStrikeSpecPlanner(config=cfg)


def _solvable_problem(t=100.0, q0=None, max_iter=None, target_land_xy=(1.80, 0.29)):
    """Venue-like strike snapshot known to solve (mirrors node's call shape:
    spin-blind omega None, with_sensitivities False on the hot path)."""
    return ShadowProblem(
        t=t,
        p_ball=np.array([0.28, -0.10, 0.34]),
        v_ball=np.array([-4.47, 0.33, -1.17]),
        omega_ball=None,
        target_land_xy=np.array(target_land_xy, dtype=float),
        racket_speed_budget=10.0,
        max_iter=max_iter,
        q0=q0,
        with_sensitivities=False,
    )


def _production_solve(planner, problem):
    """The node's production call, argument for argument."""
    return planner.solve_fast_spec(
        problem.p_ball, problem.v_ball, problem.omega_ball,
        problem.target_land_xy, problem.racket_speed_budget,
        max_iter=problem.max_iter, q0=problem.q0,
        with_sensitivities=problem.with_sensitivities,
    )


class _RaisingBackend(SolverBackend):
    name = "raising"

    def solve(self, problem):
        raise RuntimeError("boom")


class _NoneBackend(SolverBackend):
    name = "none"

    def solve(self, problem):
        return None


# --------------------------------------------------------------------- #
# (1) echo reconciles to exactly zero — cold AND warm-started ticks
# --------------------------------------------------------------------- #


def test_echo_backend_diff_is_exactly_zero_cold_and_warm():
    prod = _production_setup()
    echo = EchoBackend(physics=prod.physics, config=prod.config, table=prod.table)

    # cold tick (q0 None, solver default budget — the node's cold branch)
    problem = _solvable_problem()
    spec = _production_solve(prod, problem)
    assert spec is not None
    diffs = spec_field_diffs(spec, echo.solve(problem))
    assert diffs["prod_valid"] == 1 and diffs["shadow_valid"] == 1
    for key in ("d_n_l2", "d_v_r_l2_mps", "d_landing_xy_l2_m", "d_landing_time_s"):
        assert diffs[key] == 0.0, f"{key}: echo must be bit-identical, got {diffs[key]}"

    # warm tick: the node snapshots the SAME q0/max_iter it hands production
    q0 = np.array([spec.tilt_pitch_deg, spec.tilt_yaw_deg, spec.v_n_signed,
                   spec.v_t_vec[0], spec.v_t_vec[1]])
    warm_problem = _solvable_problem(
        t=100.02, q0=q0, max_iter=FastStrikeSpecPlanner.PROD_MAX_ITER)
    warm_spec = _production_solve(prod, warm_problem)
    assert warm_spec is not None
    warm_diffs = spec_field_diffs(warm_spec, echo.solve(warm_problem))
    for key in ("d_n_l2", "d_v_r_l2_mps", "d_landing_xy_l2_m", "d_landing_time_s"):
        assert warm_diffs[key] == 0.0


def test_negative_control_diff_detector_fires_on_different_solves():
    """NEGATIVE CONTROL: the diff machinery must return NONZERO for two
    genuinely different solves. Without this, a regression that makes the
    comparison vacuous (echo handed the production spec object, a field
    compared against itself after a rename, ...) keeps every ==0.0 assertion
    green and the shadow CSV certifies ANY torch backend as perfect parity."""
    prod = _production_setup()
    problem = _solvable_problem()
    spec = _production_solve(prod, problem)
    assert spec is not None
    # same strike state, aim pulled 20 cm shorter -> a genuinely different
    # admissible spec (verified to solve cold from this snapshot)
    shifted = _solvable_problem(target_land_xy=(1.60, 0.29))
    shifted_spec = _production_solve(prod, shifted)
    assert shifted_spec is not None
    diffs = spec_field_diffs(spec, shifted_spec)
    assert diffs["prod_valid"] == 1 and diffs["shadow_valid"] == 1
    for key in ("d_n_l2", "d_v_r_l2_mps", "d_landing_xy_l2_m", "d_landing_time_s"):
        assert math.isfinite(diffs[key]), f"{key} must be finite for two specs"
        assert diffs[key] > 0.0, (
            f"{key}: diff detector returned 0 for two DIFFERENT solves — "
            "the comparison is vacuous")
    # the aim shift itself must show up at roughly its own magnitude
    assert diffs["d_landing_xy_l2_m"] > 0.1


def test_echo_via_harness_end_to_end(tmp_path):
    prod = _production_setup()
    log = tmp_path / "shadow.csv"
    harness = ShadowHarness(
        EchoBackend(physics=prod.physics, config=prod.config, table=prod.table),
        str(log))
    problem = _solvable_problem()
    spec = _production_solve(prod, problem)
    diffs = harness.run(problem, spec, production_wall_s=0.015)
    assert harness.n_runs == 1
    assert harness.n_shadow_exceptions == 0
    assert harness.n_record_failures == 0
    assert diffs["d_landing_xy_l2_m"] == 0.0
    # diag string is a single compact value (one diagnostics KeyValue)
    s = harness.diag_value()
    assert "backend=echo" in s and "runs=1" in s and "exc=0" in s
    assert "\n" not in s


# --------------------------------------------------------------------- #
# (2) torch placeholder is fail-loud at construction
# --------------------------------------------------------------------- #


def test_torch_backend_fail_loud_at_construction():
    with pytest.raises(NotImplementedError) as exc_info:
        TorchBackend()
    # the error must point the reader at the I/O contract location
    assert "shadow_solver" in str(exc_info.value)
    assert "StrikeSpec" in str(exc_info.value)


def test_make_backend_factory():
    prod = _production_setup()
    backend = make_backend("echo", physics=prod.physics, config=prod.config)
    assert isinstance(backend, EchoBackend)
    with pytest.raises(NotImplementedError):
        make_backend("torch")
    with pytest.raises(ValueError):
        make_backend("lm_fused_v9")


# --------------------------------------------------------------------- #
# (3) exception isolation: harness.run never raises
# --------------------------------------------------------------------- #


def test_harness_swallows_backend_exception_and_records_it(tmp_path):
    prod = _production_setup()
    log = tmp_path / "shadow.csv"
    harness = ShadowHarness(_RaisingBackend(), str(log))
    problem = _solvable_problem()
    spec = _production_solve(prod, problem)

    diffs = harness.run(problem, spec, production_wall_s=0.01)  # must not raise
    assert harness.n_shadow_exceptions == 1
    assert diffs["prod_valid"] == 1 and diffs["shadow_valid"] == 0
    assert math.isnan(diffs["d_landing_xy_l2_m"])

    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 1
    assert rows[0]["shadow_exception"] == "RuntimeError: boom"
    assert rows[0]["prod_valid"] == "1" and rows[0]["shadow_valid"] == "0"


def test_harness_counts_recorder_failure_without_raising(tmp_path):
    # unopenable path (a directory) -> recorder fails, harness only counts
    harness = ShadowHarness(_NoneBackend(), str(tmp_path))
    harness.run(_solvable_problem(), None, production_wall_s=0.0)
    assert harness.n_record_failures == 1
    assert harness.n_shadow_exceptions == 0
    harness.run(_solvable_problem(), None, production_wall_s=0.0)  # stays disabled
    assert harness.n_record_failures == 2


def test_harness_wall_budget_self_disables_permanently():
    """A backend persistently slower than the wall budget disables the shadow
    (permanently) instead of stealing the /poses callback forever."""

    class _SlowBackend(SolverBackend):
        name = "slow"

        def __init__(self):
            self.calls = 0

        def solve(self, problem):
            self.calls += 1
            time.sleep(0.002)
            return None

    backend = _SlowBackend()
    harness = ShadowHarness(backend, None, wall_budget_s=1e-4, max_over_budget=2)
    assert harness.active

    harness.run(_solvable_problem(), None, production_wall_s=0.0)
    assert harness.active and harness.n_over_budget == 1

    harness.run(_solvable_problem(), None, production_wall_s=0.0)
    assert not harness.active
    assert harness.n_over_budget == 2
    assert "wall budget" in harness.disabled_reason
    assert "DISABLED" in harness.diag_value()

    # disabled = permanent no-op: backend not called, counters frozen
    calls_before = backend.calls
    out = harness.run(_solvable_problem(), None, production_wall_s=0.0)
    assert out == {}
    assert backend.calls == calls_before
    assert harness.n_runs == 2


def test_wall_budget_none_never_disables():
    harness = ShadowHarness(_NoneBackend(), None, wall_budget_s=None)
    for _ in range(3):
        harness.run(_solvable_problem(), None, production_wall_s=0.0)
    assert harness.active and harness.n_over_budget == 0


def test_both_none_reconciles_as_valid_flags_not_error():
    harness = ShadowHarness(_NoneBackend(), None)  # logging off
    diffs = harness.run(_solvable_problem(), None, production_wall_s=0.0)
    assert diffs["prod_valid"] == 0 and diffs["shadow_valid"] == 0
    assert math.isnan(diffs["d_n_l2"])
    assert harness.n_shadow_exceptions == 0
    assert harness.n_record_failures == 0  # path None = logging not requested


# --------------------------------------------------------------------- #
# (4) CSV row format
# --------------------------------------------------------------------- #


def test_csv_format_header_and_parseable_rows(tmp_path):
    prod = _production_setup()
    log = tmp_path / "shadow.csv"
    harness = ShadowHarness(
        EchoBackend(physics=prod.physics, config=prod.config, table=prod.table),
        str(log))
    problem = _solvable_problem()
    spec = _production_solve(prod, problem)
    harness.run(problem, spec, production_wall_s=0.0151)
    harness.run(_solvable_problem(t=100.02), spec, production_wall_s=0.0032)

    with open(log) as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ShadowHarness.CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 2  # one row per double-run, header written once
    assert float(rows[0]["tick_t"]) == 100.0
    assert abs(float(rows[0]["prod_wall_ms"]) - 15.1) < 1e-6
    assert float(rows[0]["shadow_wall_ms"]) >= 0.0
    for key in ("d_n_l2", "d_v_r_l2_mps", "d_landing_xy_l2_m", "d_landing_time_s"):
        assert float(rows[0][key]) == 0.0  # echo
    assert rows[0]["shadow_exception"] == ""


def test_csv_append_keeps_single_header(tmp_path):
    """A restarted session appends to an existing CSV without a second header."""
    log = tmp_path / "shadow.csv"
    for _ in range(2):  # two harness lifetimes, same file
        harness = ShadowHarness(_NoneBackend(), str(log))
        harness.run(_solvable_problem(), None, production_wall_s=0.0)
        harness.recorder.close()
    lines = open(log).read().splitlines()
    assert len(lines) == 3  # 1 header + 2 rows
    assert lines[0].split(",") == ShadowHarness.CSV_FIELDS
