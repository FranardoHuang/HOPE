from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_rally_intervals.py"
SPEC = importlib.util.spec_from_file_location("analyze_rally_intervals", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _row(take: str, time_s: float, paddle: str):
    return {"take": take, "t_c": time_s, "paddle": paddle}


def test_extracts_only_consecutive_bounded_aba_triples():
    rows = [
        _row("rally", 0.0, "p1"),
        _row("rally", 0.8, "p2"),
        _row("rally", 1.7, "p1"),
        _row("rally", 2.8, "p2"),
        _row("rally", 7.0, "p1"),  # second leg too long for the overlapping p2->p1->p2
        _row("dianqiu_nospin", 0.0, "p1"),
        _row("dianqiu_nospin", 0.4, "p2"),
        _row("dianqiu_nospin", 0.8, "p1"),
    ]

    samples = audit.extract_aba(rows, max_leg_s=2.5)

    assert len(samples) == 2
    assert samples[0]["paddle"] == "p1"
    assert samples[0]["self_to_opponent_s"] == pytest.approx(0.8)
    assert samples[0]["opponent_to_self_s"] == pytest.approx(0.9)
    assert samples[0]["same_player_interval_s"] == pytest.approx(1.7)
    assert samples[1]["paddle"] == "p2"
    assert samples[1]["same_player_interval_s"] == pytest.approx(2.0)


def test_linear_quantile_matches_the_documented_interpolation():
    assert audit.linear_quantile([1.0, 2.0, 4.0], 0.5) == pytest.approx(2.0)
    assert audit.linear_quantile([1.0, 2.0, 4.0], 0.1) == pytest.approx(1.2)
    with pytest.raises(ValueError, match="empty"):
        audit.linear_quantile([], 0.5)
