"""Unit tests for the STROKE GUARD in scripts/gen_stage1_questions.py (快筛层落地).

Pure CPU, NO mujoco/torch/planner: the generator module is loaded by file path and only
the guard-layer symbols are exercised (StrokeGuard / blade_acc_envelope /
resolve_stroke_guard / stroke_disposition) plus a mini bank loop that replicates the
main()-loop disposition logic exactly.

Design under test = docs/research/stroke_interface_survey_2026-07-09.md §3.1:
  * a_min = (v*² − v0²)/(2·L_deep) is a NECESSARY-condition lower bound -> the guard may
    only sound-reject (a_min > a_max); PASS != feasible.
  * L_deep = the stroke-ledger 口径 (deep frame -> contact frame arc length), the ONE
    implementation in extend_stroke.deep_frame_and_L.
  * a_max = empirical blade-point Cartesian |acc| envelope over the budget clips × scale
    (the time-law family's v4rg×1.5 budget convention, taken at the blade point).
  * v1 分母法则: default stats mode counts but does NOT drop (报"若开拦会拦掉 N 题");
    only --stroke-guard enforce really drops; off disables entirely.
  * fail-loud: degenerate stroke / non-finite v* / missing budget data must raise, never
    silently skip (a silent skip falsifies the denominator report).

Run:  pytest hope_training/whole_body_tracking/tests/test_stroke_guard_stage1.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gsq = _load("gsq_stroke_test", "gen_stage1_questions.py")

FPS = 50.0
NB = 32          # body count >= RACKET_BODY+1 (synthesize_timing.RACKET_BODY == 31)


# --------------------------------------------------------------- synthetic clips -- #
def _write_clip(path, blade_xyz: np.ndarray, fps: float = FPS) -> None:
    """Minimal npz whose synthesize_timing.blade_positions equals blade_xyz + const.

    Identity quats everywhere -> blade = body_pos_w[:, 31] + MOUNT_OFFSET (a constant,
    which cancels in every difference/derivative the guard takes).
    """
    T = blade_xyz.shape[0]
    pos = np.zeros((T, NB, 3), dtype=np.float64)
    pos[:, 31] = blade_xyz
    quat = np.zeros((T, NB, 4), dtype=np.float64)
    quat[..., 0] = 1.0
    np.savez(path, fps=np.array([int(fps)], dtype=np.int64),
             body_pos_w=pos.astype(np.float32), body_quat_w=quat.astype(np.float32))


def _quadratic_blade(T: int, acc: float, fps: float = FPS) -> np.ndarray:
    """Constant-acceleration straight-line blade path: max |a| via np.gradient×2 == acc."""
    t = np.arange(T) / fps
    out = np.zeros((T, 3))
    out[:, 0] = 0.5 * acc * t**2
    return out


# --------------------------------------------------------------------- a_min math -- #
def test_amin_formula_and_v0():
    g = gsq.StrokeGuard("clip", L_deep=0.5, deep_frame=3, contact_frame=10, a_max=10.0)
    assert g.a_min(2.0) == pytest.approx(4.0)          # v*²/(2L) = 4/1.0
    assert g.a_min(0.0) == pytest.approx(0.0)
    g2 = gsq.StrokeGuard("clip", 0.5, 3, 10, 10.0, v0=1.0)
    assert g2.a_min(2.0) == pytest.approx(3.0)         # (4-1)/1.0
    # v* below v0 -> non-positive bound -> trivially passes (mathematically consistent)
    a, over = g2.check(0.5)
    assert a < 0.0 and over is False


def test_v_star_cap_inverts_the_bound():
    g = gsq.StrokeGuard("clip", L_deep=0.72, deep_frame=0, contact_frame=5, a_max=9.0)
    cap = g.v_star_cap
    assert g.a_min(cap) == pytest.approx(g.a_max)      # cap sits exactly on the boundary
    _, over_lo = g.check(cap * 0.999)
    _, over_hi = g.check(cap * 1.001)
    assert over_lo is False and over_hi is True


# ----------------------------------------------------------- ledger 口径 (L_deep) -- #
def test_ledger_parity_with_extend_stroke():
    """Guard L_deep must be bitwise the extend_stroke deep_frame_and_L value (单一实现)."""
    rng = np.random.default_rng(7)
    blade = np.cumsum(rng.normal(scale=0.02, size=(40, 3)), axis=0)
    c = 30
    es = gsq._stroke_ledger()
    d_ref, L_ref = es.deep_frame_and_L(blade, c)
    g = gsq.StrokeGuard.from_blade_path("clip", blade, c, a_max=5.0)
    assert g.deep_frame == d_ref
    assert g.L_deep == L_ref
    assert g.contact_frame == c


def test_ledger_L_hand_checked_value():
    # straight retreat to -0.3 m then straight advance to +0.5 m: deep frame is the
    # turning point, L_deep = 0.8 m exactly (per-frame displacement sum, ledger 口径)
    x = np.concatenate([np.linspace(0.0, -0.3, 4), np.linspace(-0.2, 0.5, 8)])
    blade = np.zeros((len(x), 3))
    blade[:, 0] = x
    g = gsq.StrokeGuard.from_blade_path("clip", blade, len(x) - 1, a_max=100.0)
    assert g.deep_frame == 3
    assert g.L_deep == pytest.approx(0.8)
    assert g.a_min(2.0) == pytest.approx(4.0 / 1.6)


# ------------------------------------------------------------------- fail-loud ---- #
def test_degenerate_stroke_fails_loud():
    # contact frame IS the euclidean-farthest point's own position for a frozen path:
    # dist all zero -> d = 0, L = 0 -> refuse (never silently ship an unpriceable clip)
    frozen = np.zeros((20, 3))
    with pytest.raises(SystemExit, match="L_deep"):
        gsq.StrokeGuard.from_blade_path("frozen", frozen, 10, a_max=5.0)


def test_contact_frame_without_runup_fails_loud():
    blade = _quadratic_blade(20, 1.0)
    with pytest.raises(SystemExit, match="pre-contact"):
        gsq.StrokeGuard.from_blade_path("clip", blade, 0, a_max=5.0)
    with pytest.raises(SystemExit, match="pre-contact"):
        gsq.StrokeGuard.from_blade_path("clip", blade, 20, a_max=5.0)  # out of range


def test_nonfinite_blade_path_fails_loud():
    blade = _quadratic_blade(20, 1.0)
    blade[7, 1] = np.nan
    with pytest.raises(SystemExit, match="non-finite"):
        gsq.StrokeGuard.from_blade_path("clip", blade, 15, a_max=5.0)


def test_nonfinite_vstar_fails_loud():
    g = gsq.StrokeGuard("clip", 0.5, 3, 10, a_max=10.0)
    with pytest.raises(SystemExit, match="v\\*"):
        g.a_min(float("nan"))
    with pytest.raises(SystemExit, match="v\\*"):
        g.check(float("inf"))


def test_bad_a_max_fails_loud():
    with pytest.raises(SystemExit, match="a_max"):
        gsq.StrokeGuard("clip", 0.5, 3, 10, a_max=0.0)
    with pytest.raises(SystemExit, match="a_max"):
        gsq.StrokeGuard("clip", 0.5, 3, 10, a_max=float("nan"))


# ------------------------------------------------------- a_max envelope (budget) -- #
def test_blade_acc_envelope_known_value(tmp_path):
    p = tmp_path / "budget_a.npz"
    _write_clip(p, _quadratic_blade(60, acc=3.0))
    per = gsq.blade_acc_envelope([str(p)])
    # np.gradient×2 is exact on quadratics; residual = float32 storage quantization of
    # body_pos_w (production clips store float32 too) -> tolerance 1e-3 relative
    assert per[p.name] == pytest.approx(3.0, rel=1e-3)


def test_resolve_stroke_guard_pools_max_and_scales(tmp_path):
    pa, pb = tmp_path / "fh.npz", tmp_path / "bh.npz"
    _write_clip(pa, _quadratic_blade(60, acc=3.0))
    _write_clip(pb, _quadratic_blade(60, acc=8.0))
    a_max, per = gsq.resolve_stroke_guard("stats", [str(pa), str(pb)], scale=1.5)
    assert per[pa.name] == pytest.approx(3.0, rel=1e-3)
    assert per[pb.name] == pytest.approx(8.0, rel=1e-3)
    assert a_max == pytest.approx(12.0, rel=1e-3)       # max(3, 8) × 1.5 — 同预算口径


def test_resolve_stroke_guard_off_needs_nothing():
    a_max, per = gsq.resolve_stroke_guard("off", None, 1.5)
    assert a_max is None and per == {}


@pytest.mark.parametrize("mode", ["stats", "enforce"])
def test_guard_on_without_budget_clips_fails_loud(mode):
    with pytest.raises(SystemExit, match="--stroke-budget-clips"):
        gsq.resolve_stroke_guard(mode, None, 1.5)
    with pytest.raises(SystemExit, match="--stroke-budget-clips"):
        gsq.resolve_stroke_guard(mode, [], 1.5)


def test_budget_clip_missing_body_arrays_fails_loud(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, fps=np.array([50]), joint_pos=np.zeros((30, 31)))  # no body arrays
    with pytest.raises(SystemExit, match="body"):
        gsq.blade_acc_envelope([str(p)])


def test_budget_clip_too_short_fails_loud(tmp_path):
    p = tmp_path / "short.npz"
    _write_clip(p, _quadratic_blade(3, acc=1.0))
    with pytest.raises(SystemExit, match="short"):
        gsq.blade_acc_envelope([str(p)])


# --------------------------------------------- 分母法则: stats 不拦, enforce 才拦 -- #
def _mini_bank_loop(mode: str, guard, v_stars):
    """Replicates the main()-loop guard block verbatim: check -> disposition -> keep."""
    kept, a_mins = [], []
    for i, v in enumerate(v_stars):
        a_min_q, over = guard.check(v)
        if gsq.stroke_disposition(mode, over) == "reject":
            continue
        kept.append(i)
        a_mins.append(a_min_q)
    return kept, a_mins


def test_stats_counts_but_ships_everything():
    guard = gsq.StrokeGuard("clip", L_deep=0.5, deep_frame=3, contact_frame=10, a_max=4.0)
    v = [1.0, 2.0, 3.0, 4.0]                 # a_min = 1, 4, 9, 16 vs a_max 4
    kept, a_mins = _mini_bank_loop("stats", guard, v)
    assert kept == [0, 1, 2, 3]              # stats mode NEVER drops (v1 默认不拦)
    assert guard.over_budget_count == 2      # "若开拦会拦掉 2 题" — franco's number
    assert a_mins == pytest.approx([1.0, 4.0, 9.0, 16.0])  # flagged ones stay priceable


def test_enforce_rejects_over_budget_only():
    guard = gsq.StrokeGuard("clip", L_deep=0.5, deep_frame=3, contact_frame=10, a_max=4.0)
    v = [1.0, 2.0, 3.0, 4.0]
    kept, a_mins = _mini_bank_loop("enforce", guard, v)
    assert kept == [0, 1]                    # a_min > a_max 的题拒发货
    assert guard.over_budget_count == 2      # …并计数进分母报表
    assert max(a_mins) <= guard.a_max


def test_boundary_question_ships():
    # a_min == a_max is NOT over budget (reject requires strict >, the sound direction:
    # only a PROVEN-impossible question may be dropped)
    guard = gsq.StrokeGuard("clip", L_deep=0.5, deep_frame=3, contact_frame=10, a_max=4.0)
    _, over = guard.check(2.0)               # a_min = 4.0 exactly
    assert over is False
    assert guard.over_budget_count == 0


def test_disposition_table():
    assert gsq.stroke_disposition("off", True) == "ship"
    assert gsq.stroke_disposition("off", False) == "ship"
    assert gsq.stroke_disposition("stats", False) == "ship"
    assert gsq.stroke_disposition("stats", True) == "flag"
    assert gsq.stroke_disposition("enforce", False) == "ship"
    assert gsq.stroke_disposition("enforce", True) == "reject"


# ------------------------------------------- offline audit (existing v2 exam 卷) -- #
sga = _load("sga_stroke_test", "stroke_guard_bank_audit.py")


def _write_bank(path, name: str, v_stars, anchor_frame=None) -> None:
    """Minimal bank npz: demanded_vel rows with |v| = v_stars + real meta_json."""
    v = np.zeros((len(v_stars), 3))
    v[:, 0] = v_stars
    meta = dict(grip_applied=True, rally_yaw_applied=True,
                clips={name: ({} if anchor_frame is None
                              else dict(anchor_frame=int(anchor_frame)))})
    np.savez(path, **{f"{name}/demanded_vel": v},
             meta_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))


def _stroke_clip(tmp_path, fname: str, L: float = 0.5, T: int = 12):
    """Clip whose blade retreats then advances: L_deep == L at contact frame T-1."""
    x = np.concatenate([np.linspace(0.0, -L / 2, 4), np.linspace(-L / 2 + L / 14, L / 2, 7)])
    blade = np.zeros((len(x), 3))
    blade[:, 0] = x
    p = tmp_path / fname
    _write_clip(p, blade)
    return str(p), len(x) - 1


def test_bank_audit_counts_match_generator_guard(tmp_path):
    clip_path, c = _stroke_clip(tmp_path, "clip.npz", L=0.5)
    budget = tmp_path / "budget.npz"
    _write_clip(budget, _quadratic_blade(60, acc=4.0))     # a_max = 4×1.5 = 6 m/s²
    # L_deep from float32 clip storage; price the v* list against the SAME guard
    es = gsq._stroke_ledger()
    blade = es.st.blade_positions(dict(np.load(clip_path)))
    d_ref, L_ref = es.deep_frame_and_L(blade, c)
    v_stars = [1.0, 2.0, 3.0, 4.0]
    expect_over = sum(v * v / (2 * L_ref) > 6.0 * (4.0 / 4.0) for v in v_stars)  # vs ~6
    bank = tmp_path / "bank_exam.npz"
    _write_bank(bank, "backhand", v_stars, anchor_frame=c)
    res = sga.audit_bank(str(bank), {"backhand": clip_path}, [str(budget)])
    row = res["backhand"]
    assert row["questions"] == 4
    assert row["L_deep_m"] == pytest.approx(L_ref)
    assert row["over_budget"] == expect_over > 0           # v5 反手式:高 v* 短 L 被拦
    assert res["_meta"]["a_max_mps2"] == pytest.approx(6.0, rel=1e-3)


def test_bank_audit_missing_anchor_frame_fails_loud(tmp_path):
    clip_path, c = _stroke_clip(tmp_path, "clip.npz")
    budget = tmp_path / "budget.npz"
    _write_clip(budget, _quadratic_blade(60, acc=4.0))
    bank = tmp_path / "bank_exam.npz"
    _write_bank(bank, "backhand", [1.0, 2.0], anchor_frame=None)   # meta lacks the anchor
    with pytest.raises(SystemExit, match="anchor_frame"):
        sga.audit_bank(str(bank), {"backhand": clip_path}, [str(budget)])
    # explicit override unblocks it (fail-loud, then human supplies the fact)
    res = sga.audit_bank(str(bank), {"backhand": clip_path}, [str(budget)],
                         anchor_overrides={"backhand": c})
    assert res["backhand"]["questions"] == 2


def test_bank_audit_missing_clip_entry_fails_loud(tmp_path):
    clip_path, c = _stroke_clip(tmp_path, "clip.npz")
    budget = tmp_path / "budget.npz"
    _write_clip(budget, _quadratic_blade(60, acc=4.0))
    bank = tmp_path / "bank_exam.npz"
    _write_bank(bank, "backhand", [1.0], anchor_frame=c)
    with pytest.raises(SystemExit, match="forehand"):
        sga.audit_bank(str(bank), {"forehand": clip_path}, [str(budget)])


# ------------------------------------------------------------------ CLI defaults -- #
def test_cli_default_is_stats_and_off_is_available():
    """v1 拍板: 默认 = stats (统计不拦), enforce 必须显式传, off 可关闭."""
    import argparse
    src = (Path(_SCRIPTS) / "gen_stage1_questions.py").read_text()
    assert '"--stroke-guard", choices=("off", "stats", "enforce"), default="stats"' in src
    # argparse-level double check via a scratch parser mirroring main()'s declaration
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke-guard", choices=("off", "stats", "enforce"), default="stats")
    assert ap.parse_args([]).stroke_guard == "stats"
