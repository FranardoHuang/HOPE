"""CI smoke for the planner accuracy-vs-latency benchmark (N=5 end-to-end).

Full run (the real numbers, ~15 min on a Mac):
  /opt/anaconda3/bin/python3 hope_ws/src/hope_planner/benchmarks/benchmark_planner_latency.py --n 200
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("yaml")

_BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "benchmarks")
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import benchmark_planner_latency as bpl  # noqa: E402


def test_scenarios_are_venue_realistic_and_deterministic():
    a = bpl.sample_scenarios(5, seed=3)
    b = bpl.sample_scenarios(5, seed=3)
    for s, t in zip(a, b):
        np.testing.assert_array_equal(s.p_strike, t.p_strike)
        np.testing.assert_array_equal(s.v_strike, t.v_strike)
    for s in a:
        assert -0.2 < s.p_strike[0] < 0.5          # our side, near the hit plane
        assert 0.2 < s.p_strike[2] < 0.55          # venue contact heights
        assert s.v_strike[0] < 0.0                 # incoming
        assert s.lookback > 0.0 and s.p_pre[0] > s.p_strike[0]


@pytest.mark.skipif(not bpl.HAVE_TORCH, reason="oracle needs torch")
def test_benchmark_end_to_end_n5():
    rows, results = bpl.run_benchmark(
        n=5, seed=1, torch_batches=(5,), variants="smoke")
    names = {r["name"] for r in rows}
    assert {"user_quick(rtp)", "jiayi_pipeline(S2+S3)", "strike_spec(LM)",
            "fixed_normal(LM3)", "ss_fastnp_a20_warm", "torch_batch5"} <= names
    for r in rows:
        assert r["med_ms"] > 0.0
    by_name = {r["name"]: r for r in rows}
    # the baseline LM solver must produce commands that land near its target
    base = by_name["strike_spec(LM)"]
    assert base["cmd_rate"] > 0.5
    assert base["land_med_mm"] < 100.0
    # the fast prototype must be materially faster than the baseline
    assert by_name["ss_fastnp_a20_warm"]["med_ms"] < base["med_ms"]
    # table renders
    assert "strike_spec(LM)" in bpl.format_table(rows)


def test_benchmark_imports_the_production_fast_planner():
    """The fast rows must measure the SHIPPED implementation, not a fork."""
    from hope_planner.strike_spec_fast import (
        FastStrikeSpecPlanner,
        batch_integrate_to_table_plane,
    )

    assert bpl.FastStrikeSpecPlanner is FastStrikeSpecPlanner
    assert bpl._batch_integrate_to_plane is batch_integrate_to_table_plane


@pytest.mark.skipif(not bpl.HAVE_TORCH, reason="oracle needs torch")
def test_benchmark_production_variant_n4():
    rows, _ = bpl.run_benchmark(n=4, seed=2, torch_batches=(), variants="prod")
    by_name = {r["name"]: r for r in rows}
    for name in ("ss_fastnp_a20_warm", "prod_solve_fast_spec",
                 "tick_S1S2S3(legacy)", "tick_S1S2fastSS(prod)"):
        assert name in by_name, name
        assert by_name[name]["med_ms"] > 0.0
    # the production full tick must be materially faster than the legacy tick
    assert (by_name["tick_S1S2fastSS(prod)"]["med_ms"]
            < by_name["tick_S1S2S3(legacy)"]["med_ms"])
