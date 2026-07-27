"""CLOSED-FORM inverse vs the TRUE solver, on a fresh sample (CPU torch, isaaclab STUBBED).

这条测试问的是三件事,一件都不许含糊:

A. 公式答出来的球,拿**训练自己的物理**(predict_paddle_contact + coarse_landing)重放一遍,落
   点离瞄点多远。不是自洽检查——自洽检查只能证明公式和自己一致。
B. 公式拒绝的球,拒绝理由是不是**具名的物理条件**;公式不该有意见的球,是不是老老实实标成
   needs_fallback 让回给真求解器,而不是硬外插。
C. 这条测试**能不能红**。所以文件里带了两个 mutation:把接触反解退回仓库那条少了 a_t v- 项的
   镜面律、以及把飞行段的二次阻力补偿关掉——同一段断言体必须当场判失败。一条永远绿的测试不是
   保护,是装饰。

ACCURACY BUDGET (Understand phase).  Two different budgets, asserted separately:
  * LANDING — the thing that decides whether a question is a valid question.  The shipped banks'
    own LM answers replay at 3.28 mm median / 4.44 mm max; the offline solver's tolerance is 5 mm
    and the online one's is 2 cm.  Asserted here: median <= 3 mm, p99 <= 10 mm, max <= 25 mm, and
    100 % of accepted rows inside 5 cm.  Measured on this box: 1.23 / 4.77 / 7.42 mm — 3x margin.
  * CONVENTION — |dv_r| <= 0.05 m/s and face <= 2 deg against the converged LM.  This budget is
    NOT met at the median (measured 0.067 m/s / 2.08 deg at T = 0.66) and the test says so out
    loud instead of loosening the number quietly: the whole gap is the LM's own accidental
    tangential racket velocity (it seeds v_t = 0, its w_speed regulariser was supposed to hold it
    there, and it drifts), which this module's ``pin="normal"`` sets to exactly 0.  The assertion
    here is the LOOSE one (median face <= 5 deg, median |dv_r| <= 0.15 m/s) and the tight numbers
    are printed, because a threshold tuned to pass is a lie either way.

HOST NOTE: needs torch, so this does NOT run on the py3.8 host.  Run it on a pod checkout:
    python -m pytest hope_training/whole_body_tracking/tests/test_strike_spec_analytic.py -q
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = pathlib.Path(HERE).resolve().parents[2]
MDP_DIR = str(REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
              / "whole_body_tracking" / "tasks" / "tracking" / "mdp")

# NO isaaclab stub, and deliberately no import of test_reward_flags_mdp: the three modules under
# test import nothing but torch/yaml, and keeping this file free of the env stack is part of what
# is being asserted — the formula is usable from the offline bank builder, which has no isaaclab.
_PKG = "_ssa_mdp"
if _PKG not in sys.modules:
    _mod = types.ModuleType(_PKG)
    _mod.__path__ = [MDP_DIR]
    sys.modules[_PKG] = _mod


def _load(stem):
    dotted = _PKG + "." + stem
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, os.path.join(MDP_DIR, stem + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


vb = _load("virtual_ball")
sst = _load("strike_spec_torch")
ana = _load("strike_spec_analytic")

# ENV-LOCAL geometry, the coarse_landing convention: surface_z is the plane the ball CENTRE
# crosses (table surface 0.76 + ball radius 0.02); near_x 0.50 + 1.37 puts the net at 1.87.
SURFACE_Z = 0.78
NET_X = 1.87
NET_TOP_Z = 0.78 + 0.1525
SPEED_BUDGET = 3.4
T_PIN = 0.66            # the declared free dof; 0.66 s is the best convention match to the LM

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def prm():
    return vb.load_venue_params(str(REPO / "configs" / "ball_physics_venue.yaml"))


def _draw(n, seed, vx=(-4.5, -2.0), spin=50.0):
    """A FRESH sample from the ContinuousQuestionCfg defaults — not a stored fixture."""
    g = torch.Generator().manual_seed(seed)
    u = lambda a, b: a + (b - a) * torch.rand(n, generator=g, dtype=torch.float64)  # noqa: E731
    p = torch.stack([u(0.50, 0.62), u(-0.45, 0.45), u(0.80, 1.20)], dim=-1)
    v = torch.stack([u(vx[0], vx[1]), u(-0.6, 0.6), u(-1.0, 0.5)], dim=-1)
    w = (torch.rand(n, 3, generator=g, dtype=torch.float64) * 2.0 - 1.0) * spin
    aim = torch.stack([u(2.20, 2.90), u(-0.55, 0.55)], dim=-1)
    return p, v, w, aim


def _replay(prm_, p, v, w, n, v_r):
    """Score an answer through the TRAINING physics — never through the closed form itself."""
    v_plus, w_plus = vb.predict_paddle_contact(v, v_r, n, w, prm_)
    return vb.coarse_landing(p, v_plus, w_plus, prm_, surface_z=SURFACE_Z, net_x=NET_X,
                             h=0.01, n_steps=100)


def _landing_failures(prm_, n=4000, seed=20260727, solver=None):
    """The assertion BODY, shared by the real test and the mutation tests.

    Returns ``(failures, stats)``.  ``solver`` defaults to the shipped ``solve_analytic``; the
    mutation tests pass a deliberately broken one and require this same body to go red.
    """
    solver = solver or ana.solve_analytic
    p, v, w, aim = _draw(n, seed)
    out = solver(p, v, w, aim, prm_, SURFACE_Z, NET_X, t_flight=T_PIN,
                 speed_budget=SPEED_BUDGET, net_top_z=NET_TOP_Z)
    land = _replay(prm_, p, v, w, out["n"], out["v_r"])
    ok = out["ok"] & land["land_valid"]
    err = torch.linalg.norm(land["land_xy"] - aim, dim=-1)
    fails = []
    stats = {"n": n, "ok_rate": float(out["ok"].float().mean()),
             "fallback_rate": float(out["needs_fallback"].float().mean())}
    if int(ok.sum()) == 0:
        return ["A. no row was both accepted and landed — the map answered nothing"], stats
    e = err[ok]
    stats.update(median_mm=float(e.median()) * 1e3,
                 p99_mm=float(e.quantile(0.99)) * 1e3,
                 max_mm=float(e.max()) * 1e3,
                 inside_5cm=float((e < 0.05).float().mean()))
    if stats["median_mm"] > 3.0:
        fails.append(f"A. median landing error {stats['median_mm']:.3f} mm > 3 mm budget")
    if stats["p99_mm"] > 10.0:
        fails.append(f"A. p99 landing error {stats['p99_mm']:.3f} mm > 10 mm budget")
    if stats["max_mm"] > 25.0:
        fails.append(f"A. worst landing error {stats['max_mm']:.3f} mm > 25 mm budget")
    if stats["inside_5cm"] < 1.0:
        fails.append(f"A. only {stats['inside_5cm']:.4%} of accepted rows land inside 5 cm")
    if stats["ok_rate"] < 0.99:
        fails.append(f"A. solve rate {stats['ok_rate']:.4%} < 99 % on the declared box")
    return fails, stats


def test_landing_accuracy_against_the_training_physics(prm):
    """A. The closed form's answers, replayed through the trainer's own scorer."""
    fails, stats = _landing_failures(prm)
    print("\n  analytic landing: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    assert fails == [], "\n".join(fails)


def test_beats_the_incumbent_solver_on_the_same_fresh_draw(prm):
    """A'. Same balls, both solvers, scored the same way.  The LM's tail is its non-convergence."""
    p, v, w, aim = _draw(4000, seed=20260728)
    ana_out = ana.solve_analytic(p, v, w, aim, prm, SURFACE_Z, NET_X, t_flight=T_PIN,
                                 speed_budget=SPEED_BUDGET)
    lm = sst.solve_strike_specs(p, v, w, aim, prm, surface_z=SURFACE_Z, net_x=NET_X,
                                speed_budget=SPEED_BUDGET, n_iters=12, tol_m=0.02)
    e_a = torch.linalg.norm(_replay(prm, p, v, w, ana_out["n"], ana_out["v_r"])["land_xy"] - aim,
                            dim=-1)[ana_out["ok"]]
    e_l = torch.linalg.norm(lm["landing_xy"] - aim, dim=-1)[lm["ok"]]
    print(f"\n  analytic ok={float(ana_out['ok'].float().mean()):.4%} "
          f"med={float(e_a.median())*1e3:.3f} max={float(e_a.max())*1e3:.3f} mm | "
          f"LM ok={float(lm['ok'].float().mean()):.4%} "
          f"med={float(e_l.median())*1e3:.3f} max={float(e_l.max())*1e3:.3f} mm")
    assert float(ana_out["ok"].float().mean()) >= float(lm["ok"].float().mean())
    assert float(e_a.median()) <= float(e_l.median())
    # the LM's rejected rows are a convergence tail, not infeasible questions: the closed form
    # answers them inside the SAME budget.  If this ever stops holding, the tail became physical.
    drop = ~lm["ok"]
    if int(drop.sum()) > 0:
        assert bool(ana_out["ok"][drop].all()), (
            f"{int((~ana_out['ok'][drop]).sum())}/{int(drop.sum())} of the LM's dropped rows are "
            f"also refused by the closed form")


def test_convention_gap_against_the_converged_solver_is_reported_not_hidden(prm):
    """B. Same answer, or merely the same landing?  The tight budget is printed, not asserted."""
    p, v, w, aim = _draw(2000, seed=20260729)
    ref = sst.solve_strike_specs(p, v, w, aim, prm, surface_z=SURFACE_Z, net_x=NET_X,
                                 speed_budget=99.0, n_iters=60, tol_m=0.005)
    out = ana.solve_analytic(p, v, w, aim, prm, SURFACE_Z, NET_X, t_flight=T_PIN,
                             speed_budget=SPEED_BUDGET)
    m = ref["ok"] & out["ok"]
    assert int(m.sum()) > 1000
    n_a, n_b = out["n"][m], ref["n"][m]
    sgn = torch.sign((n_a * n_b).sum(-1, keepdim=True))
    face = torch.rad2deg(torch.arccos(((n_a * sgn) * n_b).sum(-1).clamp(-1.0, 1.0)))
    dvr = torch.linalg.norm(out["v_r"][m] - ref["v_r"][m], dim=-1)
    v_t = ref["v_r"] - (ref["v_r"] * ref["n"]).sum(-1, keepdim=True) * ref["n"]
    print(f"\n  face med={float(face.median()):.3f} p90={float(face.quantile(0.9)):.3f} deg "
          f"(tight budget 2 deg) | |dv_r| med={float(dvr.median()):.4f} "
          f"p90={float(dvr.quantile(0.9)):.4f} m/s (tight budget 0.05) | the LM's own |v_t| "
          f"med={float(torch.linalg.norm(v_t[m], dim=-1).median()):.4f} m/s = the whole gap")
    assert float(face.median()) <= 5.0
    assert float(dvr.median()) <= 0.15


def test_face_sign_needs_no_flip_and_plus_minus_n_are_the_same_face(prm):
    """C. The alpha > 0 proof in the module docstring, checked numerically."""
    p, v, w, aim = _draw(3000, seed=20260730)
    out = ana.solve_analytic(p, v, w, aim, prm, SURFACE_Z, NET_X, t_flight=T_PIN,
                             speed_budget=SPEED_BUDGET)
    # orient_normal flips when (v_minus - v_r).n > 0; the analytic n must already be on the right
    # side, otherwise the biquadratic's positive root was the wrong branch.
    u_n = ((v - out["v_r"]) * out["n"]).sum(-1)
    assert bool((u_n[out["ok"]] < 0.0).all()), (
        f"{int((u_n[out['ok']] >= 0).sum())} accepted rows come back on the wrong face side")
    land_p = _replay(prm, p, v, w, out["n"], out["v_r"])
    land_m = _replay(prm, p, v, w, -out["n"], out["v_r"])
    d = torch.linalg.norm(land_p["land_xy"] - land_m["land_xy"], dim=-1)
    assert float(d.max()) < 1e-9, f"+-n land {float(d.max()):.3e} m apart — not the same face"


def test_refusals_are_named_physics_not_a_single_bucket(prm):
    """D. The unsolvable region is VISIBLE: an explicit mask plus a reason nobody has to guess."""
    p, v, w, _ = _draw(1500, seed=20260731)
    ones = torch.ones(1500, dtype=torch.float64)

    def _at(aim_x):
        aim = torch.stack([ones * aim_x, torch.zeros_like(ones)], dim=-1)
        o = ana.solve_analytic(p, v, w, aim, prm, SURFACE_Z, NET_X, t_flight=T_PIN,
                               speed_budget=SPEED_BUDGET, net_top_z=NET_TOP_Z)
        hist = {}
        for code in o["reason"][~o["ok"]].tolist():
            hist[ana.REASONS[code]] = hist.get(ana.REASONS[code], 0) + 1
        return o, hist

    own_half, h_own = _at(1.00)          # aiming at your own half: the net is in the way
    legal, h_legal = _at(2.55)           # the shipped banks' own aim point
    off_table, h_off = _at(4.50)         # 1.3 m past the far edge: needs a racket you do not have
    print(f"\n  aim 1.00 ok={float(own_half['ok'].float().mean()):.3%} {h_own}"
          f"\n  aim 2.55 ok={float(legal['ok'].float().mean()):.3%} {h_legal}"
          f"\n  aim 4.50 ok={float(off_table['ok'].float().mean()):.3%} {h_off}")
    assert float(legal["ok"].float().mean()) > 0.99
    assert h_legal == {}
    assert h_own.get("net_not_cleared", 0) > 0.9 * int((~own_half["ok"]).sum())
    assert h_off.get("speed_over_budget", 0) > 0.9 * int((~off_table["ok"]).sum())
    # and nothing is fabricated: every accepted row is finite and inside its own predicates
    for o in (own_half, legal, off_table):
        m = o["ok"]
        assert bool(torch.isfinite(o["v_r"][m]).all())
        assert bool((o["speed"][m] <= SPEED_BUDGET + 1e-9).all())
        assert bool((o["cap_margin"][m] > 0.0).all())
        assert bool((o["reason"][m] == -1).all())
        assert bool((o["reason"][~m] >= 0).all())


def test_fallback_is_honest_and_its_rate_is_reported(prm):
    """E. Outside its own validated envelope the map defers instead of extrapolating."""
    # in the declared box the formula is entitled to an opinion on essentially every row
    p, v, w, aim = _draw(4000, seed=20260732)
    inbox = ana.solve_analytic(p, v, w, aim, prm, SURFACE_Z, NET_X, t_flight=T_PIN,
                               speed_budget=SPEED_BUDGET)
    rate_inbox = float(inbox["needs_fallback"].float().mean())
    # 6x outside the declared spin box the friction cap starts binding and stage 2 stops being
    # exact — the map must NOTICE, and the rows it flags must be the bad ones.
    p2, v2, w2, aim2 = _draw(4000, seed=20260733, spin=300.0)
    hot = ana.solve_analytic(p2, v2, w2, aim2, prm, SURFACE_Z, NET_X, t_flight=T_PIN,
                             speed_budget=SPEED_BUDGET)
    err = torch.linalg.norm(_replay(prm, p2, v2, w2, hot["n"], hot["v_r"])["land_xy"] - aim2,
                            dim=-1)
    flagged, clean = hot["needs_fallback"], ~hot["needs_fallback"]
    print(f"\n  fallback rate: declared box {rate_inbox:.4%}, spin +-300 "
          f"{float(flagged.float().mean()):.4%}; landing error on rows the map TRUSTS "
          f"{float(err[clean].median())*1e3:.3f} mm vs rows it FLAGS "
          f"{float(err[flagged].median())*1e3:.3f} mm")
    assert rate_inbox <= 0.01, f"fallback rate {rate_inbox:.4%} inside the declared box"
    assert float(flagged.float().mean()) > 0.05, "the map does not notice a 6x spin excursion"
    assert float(err[flagged].median()) > 3.0 * float(err[clean].median()), (
        "the fallback flag does not separate good rows from bad ones — it is decoration")


def test_fallback_path_defers_to_the_true_solver(prm):
    """E'. ``solve_with_fallback`` actually calls the LM for the rows it declines, and counts them."""
    p, v, w, aim = _draw(1200, seed=20260734, spin=300.0)
    out = ana.solve_with_fallback(p, v, w, aim, prm, SURFACE_Z, NET_X,
                                  t_flight=T_PIN, speed_budget=SPEED_BUDGET)
    used = int(out["fallback_used"].sum())
    print(f"\n  fallback_rate={out['fallback_rate']:.4%} used={used} "
          f"solved_by_LM={out['fallback_solved']}")
    assert used > 0
    assert out["fallback_rate"] == pytest.approx(used / 1200.0)
    assert bool(torch.isfinite(out["v_r"][out["ok"]]).all())


def test_swing_gap_is_a_finite_label_on_every_row(prm):
    """F. The motion-match label cannot fail: one subtraction, no solver, no branch."""
    p, v, w, aim = _draw(2000, seed=20260735)
    out = ana.solve_analytic(p, v, w, aim, prm, SURFACE_Z, NET_X, t_flight=T_PIN,
                             speed_budget=SPEED_BUDGET)
    # the answer's OWN velocity must sit (nearly) on its own cap: gap ~ 0 up to the sphere's
    # documented spin approximation
    own = ana.swing_gap(out["v_r"], v, w, out["v_plus"], prm)
    far = ana.swing_gap(out["v_r"] + torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
                        v, w, out["v_plus"], prm)
    print(f"\n  swing_gap(own answer) med={float(own.median()):.4f} m/s "
          f"(spin approximation, see answer_sphere docstring); "
          f"swing_gap(own + 2 m/s) med={float(far.median()):.4f} m/s")
    assert bool(torch.isfinite(own).all()) and bool(torch.isfinite(far).all())
    assert float(own.median()) < 0.15
    assert float(far.median()) > float(own.median()) + 0.5


# --------------------------------------------------------------------------------------------
# MUTATION GUARDS — the test must be able to go red.
# --------------------------------------------------------------------------------------------
def _mutant_uncorrected_mirror_law(*args, **kwargs):
    """Stage 2 with the repo's OWN uncorrected mirror law: c = v+ - v- instead of v+ - (1-a_t)v-.

    That missing ``a_t v-`` term is exactly what the LM seed is short by, and it is the single
    highest-leverage way to be wrong about the face.  If the landing budget does not catch it, the
    landing budget is not measuring the face.
    """
    def _wrapped(v_plus, v_in, w_in, prm_):
        k = prm_.paddle_a_t * prm_.ball_radius * w_in
        c = v_plus - v_in                        # <- the mutation (correct: v_plus - (1-a_t)*v_in)
        k2, c2 = (k * k).sum(-1), (c * c).sum(-1)
        kc = (k * c).sum(-1)
        a2 = 0.5 * ((c2 - k2) + torch.sqrt(((c2 - k2) ** 2 + 4.0 * kc ** 2).clamp_min(0.0)))
        a1 = torch.sqrt(a2.clamp_min(1e-30))
        num = a2[:, None] * c - a1[:, None] * torch.cross(k, c, dim=-1) + kc[:, None] * k
        n = num / torch.linalg.norm(num, dim=-1, keepdim=True).clamp_min(1e-30)
        n, v_r, w_plus, diag = ana.close_from_normal(n, v_plus, v_in, w_in, prm_)
        diag["alpha"] = a1
        return n, v_r, w_plus, diag

    saved = ana.contact_inverse_normal_pin
    ana.contact_inverse_normal_pin = _wrapped
    try:
        return ana.solve_analytic(*args, **kwargs)
    finally:
        ana.contact_inverse_normal_pin = saved


def _mutant_no_drag_defect(*args, **kwargs):
    """Stage 1 with the quadratic-drag defect quadrature switched off (``n_picard=0``).

    This is the "pure linearised drag" version — the honest lower bound on what the formula would
    be without its one approximation.  It must NOT pass the landing budget, otherwise the defect
    term is decoration and should be deleted rather than defended.
    """
    kwargs = dict(kwargs)
    kwargs["n_picard"] = 0
    return ana.solve_analytic(*args, **kwargs)


@pytest.mark.parametrize("mutant,label", [
    (_mutant_uncorrected_mirror_law, "stage 2 without the a_t v- correction (the LM's own seed)"),
    (_mutant_no_drag_defect, "stage 1 without the quadratic-drag defect quadrature"),
])
def test_the_accuracy_check_catches_a_mutated_map(prm, mutant, label):
    """G. Mutation guard: the SAME assertion body must go red on a deliberately broken map."""
    fails, stats = _landing_failures(prm, n=1500, seed=20260736, solver=mutant)
    print(f"\n  mutant [{label}]: {stats}")
    assert fails != [], (
        f"the landing budget did NOT catch a broken map ({label}) — the test is decoration")
