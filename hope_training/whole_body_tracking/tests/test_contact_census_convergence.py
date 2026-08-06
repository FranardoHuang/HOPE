"""The contact census must refuse to sign a peak it did not watch converge.

Plain English: these tests hold the census gate to the two failures the
2026-08-06 readiness review found in it.

  1. A 3000-step window reported a *lower bound* and called it a peak.  The
     ctrl=0 long run was still setting new records at step 11,640 of 12,000,
     so every "headroom Nx" built on it was unsigned.  The gate must call
     that NOT_CONVERGED.
  2. Peaks were compared across friction cones (plant-pyramidal 136 against
     court-elliptic 95), which is a 4-rows-vs-3-rows bookkeeping delta
     dressed up as "the table catches the robot".  The gate must refuse the
     comparison outright.

Everything here is pure logic -- no torch, no mujoco, no GPU -- so it runs in
CI on the same host that cannot import the simulator.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_CENSUS_PY = (Path(__file__).resolve().parents[1]
              / "mjlab_lane" / "contact_census.py")


def _load():
    spec = importlib.util.spec_from_file_location("contact_census", _CENSUS_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


cc = _load()


# ---------------------------------------------------------------------------
# The module must stay importable without the simulator stack.
# ---------------------------------------------------------------------------


def test_importable_without_torch_or_mujoco():
    import sys

    # exec_module already succeeded above; assert it did not drag the heavy
    # stack in as a side effect, because that is what would break CI hosts.
    fresh = _load()
    assert fresh.SCHEMA == "contact_census/2"
    assert "mjlab" not in sys.modules


# ---------------------------------------------------------------------------
# Peak / running-max bookkeeping.
# ---------------------------------------------------------------------------


def test_peak_and_step_distinguishes_no_samples_from_measured_zero():
    assert cc.peak_and_step([]) == (None, 0)
    # measured, and the answer really is zero -- step index is 1, not 0
    assert cc.peak_and_step([0, 0, 0]) == (0, 1)


def test_peak_and_step_reports_first_attainment():
    assert cc.peak_and_step([1, 5, 3, 5, 2]) == (5, 2)


def test_running_max_is_monotone_and_ends_on_the_peak():
    vals = [1, 4, 2, 9, 3, 3, 7]
    series = cc.running_max_series(vals, stride=2)
    ys = [pt[1] for pt in series]
    assert ys == sorted(ys)
    assert series[-1][0] == len(vals)          # final step always emitted
    assert series[-1][1] == max(vals)


def test_running_max_series_honours_a_sampling_gap():
    series = cc.running_max_series([3, 1, 8], stride=1, step_gap=20)
    assert [pt[0] for pt in series] == [1, 21, 41]


# ---------------------------------------------------------------------------
# The convergence rule itself.  These are the mutation tests: each one is a
# shape that the old fixed-window census signed off on.
# ---------------------------------------------------------------------------


def test_the_historical_ctrl0_long_run_is_judged_not_converged():
    """peak at 11,640 of 12,000 == 97% of the run.  Must not pass."""
    v = cc.signal_convergence(peak_at_step=11640, steps=12000,
                              stall_steps=3000)
    assert v["converged"] is False
    assert v["steps_since_last_new_record"] == 360
    assert any("97%" in r or "still climbing" in r
               for r in v["not_converged_because"])


def test_the_historical_plant_census_is_judged_not_converged():
    """peak_at_step 2534 / 3000 == 84%: the old receipt's own number."""
    v = cc.signal_convergence(peak_at_step=2534, steps=3000, stall_steps=1000)
    assert v["converged"] is False


def test_a_genuinely_stalled_signal_converges():
    v = cc.signal_convergence(peak_at_step=905, steps=12000, stall_steps=3000)
    assert v["converged"] is True
    assert v["not_converged_because"] == []


def test_stall_alone_is_not_enough_when_the_peak_lands_late():
    """Stalled 4000 steps, but the record still sits at 80% of the run."""
    v = cc.signal_convergence(peak_at_step=16000, steps=20000,
                              stall_steps=3000)
    assert v["converged"] is False


def test_late_peak_alone_is_not_enough_when_the_stall_is_short():
    v = cc.signal_convergence(peak_at_step=100, steps=1000, stall_steps=3000)
    assert v["converged"] is False


def test_no_samples_never_converges():
    v = cc.signal_convergence(peak_at_step=0, steps=0, stall_steps=1)
    assert v["converged"] is False
    assert "no samples recorded" in v["not_converged_because"]


# ---------------------------------------------------------------------------
# Headroom naming: an unconverged number must not be quotable as a margin.
# ---------------------------------------------------------------------------


def test_converged_headroom_is_named_plainly():
    h = cc.headroom_block(188, 572, converged=True, denominator="nefc")
    assert h["headroom_x"] == 3.043
    assert h["usable_as_evidence"] is True
    assert "headroom_x_lower_bound" not in h


def test_unconverged_headroom_is_renamed_and_flagged():
    h = cc.headroom_block(106, 572, converged=False, denominator="nefc")
    assert "headroom_x" not in h
    assert h["headroom_x_lower_bound"] == 5.396
    assert h["usable_as_evidence"] is False


def test_zero_and_missing_peaks_report_no_ratio_at_all():
    for peak in (None, 0):
        h = cc.headroom_block(peak, 572, converged=True, denominator="nefc")
        assert "headroom_x" not in h
        assert h["headroom_x_unknown"] is None
        assert h["usable_as_evidence"] is False


# ---------------------------------------------------------------------------
# Verdict folding.
# ---------------------------------------------------------------------------


def _sig(peak, converged=True, required=True):
    return {"peak": peak, "required": required,
            "convergence": {"converged": converged}}


def test_no_samples_outranks_every_other_verdict():
    v, why = cc.scenario_verdict(
        {"nefc_rows_per_world": _sig(None)},
        overflow_flags=["BROADPHASE"],
        ref_hits={"nefc_rows_per_world": "over"})
    assert v == "NO_SAMPLES"
    assert "nefc_rows_per_world" in why


def test_engine_overflow_outranks_a_headroom_claim():
    v, _ = cc.scenario_verdict({"a": _sig(10)},
                               overflow_flags=["EPA_HORIZON"], ref_hits={})
    assert v == "ENGINE_OVERFLOW"


def test_exceeding_the_reference_allocation_fails():
    v, _ = cc.scenario_verdict({"a": _sig(600)}, overflow_flags=[],
                               ref_hits={"a": "over"})
    assert v == "OVER_REFERENCE_ALLOCATION"


def test_exactly_filling_the_reference_allocation_still_fails():
    """nefc == njmax fits exactly, but zero headroom is not a pass."""
    v, _ = cc.scenario_verdict({"a": _sig(572)}, overflow_flags=[],
                               ref_hits={"a": "at"})
    assert v == "AT_REFERENCE_ALLOCATION"


def test_not_converged_fails_even_with_room_to_spare():
    v, why = cc.scenario_verdict({"a": _sig(95, converged=False)},
                                 overflow_flags=[], ref_hits={})
    assert v == "NOT_CONVERGED"
    assert "still climbing" in why


def test_a_clean_scenario_passes():
    v, _ = cc.scenario_verdict({"a": _sig(95)}, overflow_flags=[], ref_hits={})
    assert v == "PASS_CONVERGED"


def test_a_signal_marked_not_required_cannot_veto_on_absence():
    v, _ = cc.scenario_verdict({"a": _sig(95), "b": _sig(None, required=False)},
                               overflow_flags=[], ref_hits={})
    assert v == "PASS_CONVERGED"


def test_run_verdict_is_the_worst_scenario_not_the_best():
    assert cc.worst_verdict(["PASS_CONVERGED", "NOT_CONVERGED"]) == "NOT_CONVERGED"
    assert cc.worst_verdict(["NOT_CONVERGED", "ENGINE_OVERFLOW"]) == "ENGINE_OVERFLOW"
    assert cc.worst_verdict([]) == "NO_SAMPLES"


# ---------------------------------------------------------------------------
# Cone bookkeeping and the same-cone guard.
# ---------------------------------------------------------------------------


def test_rows_per_contact_matches_mujocos_own_arithmetic():
    assert cc.rows_per_contact("pyramidal", 3) == 4
    assert cc.rows_per_contact("elliptic", 3) == 3
    assert cc.rows_per_contact("pyramidal", 1) == 1
    assert cc.cone_name_from_int(0) == "pyramidal"
    assert cc.cone_name_from_int(1) == "elliptic"


def _receipt(cone, verdict="PASS_CONVERGED", peaks=None, scen_cone=None,
             converged=True):
    peaks = peaks or {"randpose": 188}
    return {
        "scene": "court", "nworld": 4096, "verdict": verdict,
        "cone": {"built": cone, "requested": cone,
                 "rows_per_contact": cc.rows_per_contact(cone, 3)},
        "scenarios": {
            name: {"cone": scen_cone or cone,
                   "signals": {"nefc_rows_per_world": {
                       "peak": peak,
                       "convergence": {"converged": converged}}}}
            for name, peak in peaks.items()
        },
    }


def test_a_receipt_whose_scenarios_disagree_on_the_cone_is_rejected():
    bad = _receipt("elliptic", scen_cone="pyramidal")
    with pytest.raises(cc.CensusComparisonError):
        cc.assert_single_cone(bad)


def test_a_receipt_with_no_cone_recorded_is_rejected():
    with pytest.raises(cc.CensusComparisonError):
        cc.assert_single_cone({"cone": {}, "scenarios": {}})


def test_the_false_causality_diff_is_refused():
    """plant-pyramidal 136 vs court-elliptic 95 -- the 08-06 mistake."""
    plant = _receipt("pyramidal", peaks={"zero": 136})
    court = _receipt("elliptic", peaks={"zero": 95})
    with pytest.raises(cc.CensusComparisonError) as exc:
        cc.compare_receipts(plant, court, "plant", "court")
    assert "cone conversion" in str(exc.value)


def test_an_unconverged_cell_is_dropped_not_averaged_in():
    """One side still climbing -> that cell is refused, with the reason."""
    a = _receipt("elliptic", peaks={"zero": 106})
    b = _receipt("elliptic", verdict="NOT_CONVERGED", peaks={"zero": 95},
                 converged=False)
    with pytest.raises(cc.CensusComparisonError) as exc:
        cc.compare_receipts(a, b, "a", "b")
    assert "no signal converged on both sides" in str(exc.value)


def test_a_converged_sibling_signal_still_compares():
    """Per-cell, not per-receipt: zero may be unconverged while bang is not."""
    def two_sig(zero_conv, bang_conv):
        r = _receipt("pyramidal", verdict="NOT_CONVERGED",
                     peaks={"zero": 159, "bang": 167})
        r["scenarios"]["zero"]["signals"]["nefc_rows_per_world"][
            "convergence"] = {"converged": zero_conv}
        r["scenarios"]["bang"]["signals"]["nefc_rows_per_world"][
            "convergence"] = {"converged": bang_conv}
        return r

    a = two_sig(False, True)
    b = two_sig(True, True)
    b["scenarios"]["zero"]["signals"]["nefc_rows_per_world"]["peak"] = 131
    b["scenarios"]["bang"]["signals"]["nefc_rows_per_world"]["peak"] = 174
    diff = cc.compare_receipts(a, b, "plant", "court")
    assert diff["scenarios_compared"] == ["bang"]
    assert diff["peaks"]["bang"]["nefc_rows_per_world"]["delta"] == 7
    assert [c["scenario"] for c in diff["refused_cells"]] == ["zero"]
    assert "plant" in diff["refused_cells"][0]["reason"]


def test_same_cone_converged_receipts_diff_cleanly():
    a = _receipt("pyramidal", peaks={"zero": 136, "randpose": 240})
    b = _receipt("pyramidal", peaks={"zero": 115})
    diff = cc.compare_receipts(a, b, "plant", "court")
    assert diff["cone"] == "pyramidal"
    assert diff["same_cone_enforced"] is True
    assert diff["scenarios_compared"] == ["zero"]
    assert diff["only_in_a"] == ["randpose"]
    row = diff["peaks"]["zero"]["nefc_rows_per_world"]
    assert row["delta"] == -21
    assert row["ratio"] == 0.846


# ---------------------------------------------------------------------------
# Engine overflow bitmask decode -- including the four types that never print.
# ---------------------------------------------------------------------------


def test_overflow_bit_positions_match_the_engine_enum():
    assert cc.decode_overflow_mask(0) == []
    assert cc.decode_overflow_mask(1 << 0) == ["NEFC"]
    assert cc.decode_overflow_mask(1 << 2) == ["BROADPHASE"]
    assert cc.decode_overflow_mask(1 << 8) == ["EPA_HORIZON"]
    assert cc.decode_overflow_mask((1 << 2) | (1 << 3)) == ["BROADPHASE",
                                                            "NARROWPHASE"]


def test_the_four_silent_overflow_types_are_covered():
    silent = ("HFIELD", "CONTACT_MATCH", "NVMAX", "EPA_HORIZON")
    for name in silent:
        bit = cc.OVERFLOW_FLAG_NAMES.index(name)
        assert cc.decode_overflow_mask(1 << bit) == [name]
