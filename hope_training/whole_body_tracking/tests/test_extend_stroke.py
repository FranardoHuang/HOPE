"""Unit tests for scripts/extend_stroke.py (backswing-deepening path morph).

Pure CPU, NO mujoco: the module is loaded by file path and the production MuJoCo FK is
replaced by an ANALYTIC stub — a planar 2-link arm driven by two of the 31 Isaac columns
plus a "waist" column that swings the whole arm. Blade position is then a closed form of
the joint angles, so ΔL / gradients / limits are all checkable by hand.

Covered (工程卡 A + docs/TIMELINE.md 07-09 晚四/晚五/晚六):
  * C1 bump profile: zeros + zero derivative at s0 / peak / s1; support ⊂ [s0, s1]
  * lock window: contact row AND [c-lock, end] bitwise; |v*| (clean-FD ±2) preserved exactly;
    frames before s0 and non-selected joints bitwise
  * ΔL targeting: +20% / +40% hit to tolerance; a_min = v*²/(2L) falls by the right factor
  * allocation: gradient direction lengthens the path; per-joint peak delta ∝ weight
  * URDF limits fail-loud: unreachable ΔL raises; a joint pinned in the deepening
    direction is `blocked` (dropped with WARN under --on-blocked drop, raises under fail)
  * forbidden set (晚六法则): waist_pitch/roll + legs rejected fail-loud
  * degenerate window (正手 d = 0) fails loud with the tool-gap message
  * unknown npz keys refused

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_extend_stroke.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _name in ("synthesize_timing", "extend_stroke"):
    _spec = importlib.util.spec_from_file_location(_name, _SCRIPTS / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
es = sys.modules["extend_stroke"]
st = sys.modules["synthesize_timing"]

FPS = 50.0
J = 31
NB = 32
NAMES = es.ISAAC_JOINT_NAMES

# columns the stub arm actually uses
C_WAIST_YAW = NAMES.index("waist_yaw_joint")
C_SH_PITCH = NAMES.index("right_shoulder_pitch_joint")
C_ELBOW = NAMES.index("right_elbow_joint")
C_WAIST_PITCH = NAMES.index("waist_pitch_joint")

L1, L2 = 0.30, 0.25          # stub link lengths [m]


# --------------------------------------------------------------- analytic stub FK -- #
def stub_blade(q: np.ndarray, frames=None) -> np.ndarray:
    """Planar 2-link arm, tilted by the waist pitch, then swung about z by the waist yaw.

    p_arm = [ L1·cos(a) + L2·cos(a+b),  0,  L1·sin(a) + L2·sin(a+b) ]   (a = shoulder
    pitch, b = elbow);  p = Rz(waist_yaw) · Ry(waist_pitch) · p_arm.
    All four driven columns therefore have a well-conditioned, nonzero blade gradient —
    which is what makes the `waist_pitch is pinned` test exercise the BLOCKED path rather
    than the trivial zero-weight path.
    """
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    yaw, a, b, pit = (q[:, C_WAIST_YAW], q[:, C_SH_PITCH], q[:, C_ELBOW], q[:, C_WAIST_PITCH])
    x = L1 * np.cos(a) + L2 * np.cos(a + b)
    z = L1 * np.sin(a) + L2 * np.sin(a + b)
    xp = x * np.cos(pit) + z * np.sin(pit)          # Ry(pitch)
    zp = -x * np.sin(pit) + z * np.cos(pit)
    return np.stack([xp * np.cos(yaw), xp * np.sin(yaw), zp], axis=1)


class LimitStub:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper
        self.effort, self.velocity = 100.0, 10.0


def limits(overrides: dict | None = None) -> dict:
    lim = {n: LimitStub(-3.0, 3.0) for n in NAMES}
    for n, (lo, hi) in (overrides or {}).items():
        lim[n] = LimitStub(lo, hi)
    return lim


def make_clip(T=60, contact=23, deep=13):
    """Backhand-shaped stub: the arm swings back to `deep` then forward through `contact`."""
    q = np.zeros((T, J), dtype=np.float32)
    s = np.arange(T, dtype=np.float64)
    # shoulder pitch: back-then-through, deepest (most negative) at `deep`
    q[:, C_SH_PITCH] = (-0.9 * np.sin(np.pi * np.clip(s / (2 * deep), 0, 1)) + 0.6 * s / T)
    q[:, C_ELBOW] = 0.55 + 0.35 * np.cos(np.pi * s / T)
    q[:, C_WAIST_YAW] = 0.25 * np.sin(2 * np.pi * s / (3 * T))
    # pinned flat, exactly like v5 反手的 waist_pitch(retarget clamping,整片贴在限位上)
    q[:, C_WAIST_PITCH] = 0.30
    dq = np.gradient(q.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    bp = np.zeros((T, NB, 3), dtype=np.float32)
    bp[:, st.RACKET_BODY] = stub_blade(q.astype(np.float64)).astype(np.float32)
    bq = np.zeros((T, NB, 4), dtype=np.float32)
    bq[..., 0] = 1.0
    bl = np.gradient(bp.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    data = {"fps": np.array([int(FPS)], dtype=np.int64), "joint_pos": q, "joint_vel": dq,
            "body_pos_w": bp, "body_quat_w": bq, "body_lin_vel_w": bl,
            "body_ang_vel_w": np.zeros_like(bp)}
    return data, contact / (T - 1)


ARM = ["right_shoulder_pitch_joint", "right_elbow_joint", "waist_yaw_joint"]


def run(data, phase, joints=ARM, frac=0.20, lim=None, **kw):
    return es.morph(data, phase, list(joints), frac, stub_blade, lim or limits(), **kw)


# ------------------------------------------------------------------ bump profile ---- #
def test_bump_profile_is_supported_and_peaked():
    P = es.bump_profile(60, 0, 13, 21)
    assert P[0] == 0.0 and P[13] == pytest.approx(1.0) and P[21] == 0.0
    assert (P[21:] == 0.0).all()                       # lock window untouched
    assert (P >= 0.0).all() and P.max() == pytest.approx(1.0)
    assert (np.diff(P[0:14]) > 0).all()                # unimodal: strictly up to the peak
    assert (np.diff(P[13:22]) < 0).all()               # ...strictly down to the lock window
    assert abs(P[14] - P[12]) < 5e-2                   # peak is a SMOOTH max (central diff ~0)
    # curvature is bounded by the analytic smoothstep max |P''| = 6/w² on the narrower branch
    # (w = 21-13 = 8). A C0 corner would instead show |Δ²P| ~ the peak slope, ~0.19 here.
    assert np.abs(np.diff(P, 2)).max() < 1.25 * 6.0 / min(13, 21 - 13) ** 2


def test_bump_profile_end_slopes_vanish_quadratically():
    """C1 proof: P'(s0) = P'(s1) = 0 analytically, so the one-sided difference across one
    frame must shrink like h² (ratio ≈ 4 per grid doubling). A C0-only corner would give
    ratio ≈ 2, a discontinuity ratio ≈ 1."""
    head, tail = [], []
    for k in (1, 2, 4):
        P = es.bump_profile(60 * k + 1, 0, 13 * k, 21 * k)
        head.append(P[1] - P[0])                        # slope leaving s0
        tail.append(P[21 * k - 1] - P[21 * k])          # slope arriving at s1
    for seq in (head, tail):
        assert seq[0] > 0
        assert 3.5 < seq[0] / seq[1] < 4.5
        assert 3.5 < seq[1] / seq[2] < 4.5


def test_bump_profile_degenerate_raises():
    with pytest.raises(SystemExit):
        es.bump_profile(60, 0, 0, 21)                  # peak == start (正手 d=0)


# ------------------------------------------------------------------ lock window ----- #
def test_lock_window_and_frozen_joints_bitwise():
    data, phase = make_clip()
    q_out, plan, info, _ = run(data, phase)
    q_src = np.asarray(data["joint_pos"], dtype=np.float64)
    c, s1 = info["contact_frame"], info["s1"]
    assert s1 == c - es.LOCK_FRAMES
    assert np.array_equal(q_out[s1:], q_src[s1:])         # lock window bitwise
    assert np.array_equal(q_out[c], q_src[c])             # contact row bitwise
    assert np.array_equal(q_out[0], q_src[0])             # ready pose held
    frozen = np.setdiff1d(np.arange(J), plan.cols)
    assert np.array_equal(q_out[:, frozen], q_src[:, frozen])
    assert info["frozen_joints_bitwise"]


def test_v_star_preserved_exactly():
    """The whole point of locking >= the clean-FD half-width: a_min's numerator can't drift."""
    data, phase = make_clip()
    q_out, _, info, blade_src = run(data, phase, frac=0.40)
    c = info["contact_frame"]
    blade_out = stub_blade(q_out)
    v0 = st.clean_speed_at(blade_src, c, 1.0 / FPS)
    v1 = st.clean_speed_at(blade_out, c, 1.0 / FPS)
    assert v1 == pytest.approx(v0, abs=1e-12)
    # and the ±2 stencil rows themselves are bitwise
    assert np.allclose(blade_out[c - es.LOCK_FRAMES], blade_src[c - es.LOCK_FRAMES], atol=0)


# --------------------------------------------------------------- ΔL targeting ------- #
@pytest.mark.parametrize("frac", [0.20, 0.40])
def test_extend_frac_hits_target_and_lowers_a_min(frac):
    data, phase = make_clip()
    _, _, info, _ = run(data, phase, frac=frac)
    assert info["extend_frac_out"] == pytest.approx(frac, rel=2e-3)
    L0, L1_ = info["L_deep_src_fk"], info["L_deep_out_fk"]
    assert L1_ > L0
    v = 3.405
    a0, a1 = es.a_min_of(v, L0), es.a_min_of(v, L1_)
    # a_min ∝ 1/L exactly
    assert a1 == pytest.approx(a0 / (1.0 + frac), rel=3e-3)


def test_deep_frame_and_L_matches_ledger_convention():
    blade = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [2, 0, 0], [0, 0, 0]], float)
    d, L = es.deep_frame_and_L(blade, c=4)
    assert d == 2                     # farthest (euclidean) pre-contact frame
    assert L == pytest.approx(1.0 + 2.0)   # arc length f2→f4, summed per-frame


def test_gradient_allocation_lengthens_and_scales_with_weight():
    data, phase = make_clip()
    _, plan, info, _ = run(data, phase, frac=0.30)
    q_src = np.asarray(data["joint_pos"], dtype=np.float64)
    # the morph moved the blade AWAY from contact at the peak frame
    d, c = info["peak_frame"], info["contact_frame"]
    p_c = stub_blade(q_src[c][None, :])[0]
    D0 = np.linalg.norm(stub_blade(q_src[d][None, :])[0] - p_c)
    q_out = plan.apply(q_src)
    D1 = np.linalg.norm(stub_blade(q_out[d][None, :])[0] - p_c)
    assert D1 > D0
    # peak delta of each joint == A · w_j  (allocation is exactly the weighted gradient)
    for i, n in enumerate(plan.names):
        assert info["peak_delta_rad"][n] == pytest.approx(plan.amp * plan.weights[i], abs=1e-6)
    assert max(abs(w) for w in plan.weights) == pytest.approx(1.0)


# ------------------------------------------------------------ direction refinement --- #
def test_refinement_never_increases_amplitude_and_keeps_invariants():
    data, phase = make_clip()
    q_src = np.asarray(data["joint_pos"], dtype=np.float64)
    _, p0, i0, _ = run(data, phase, frac=0.40, refine_iters=0)
    q1, p1, i1, _ = run(data, phase, frac=0.40, refine_iters=3)
    assert p1.amp <= p0.amp + 1e-9                       # monotone-improving by construction
    assert i1["refine"]["amp_final_rad"] <= i1["refine"]["amp_initial_rad"] + 1e-9
    assert i0["refine"]["iters_used"] == 0
    # both still hit the target and keep every hard invariant
    for info, qq in ((i0, None), (i1, q1)):
        assert info["extend_frac_out"] == pytest.approx(0.40, rel=2e-3)
    c, s1 = i1["contact_frame"], i1["s1"]
    assert np.array_equal(q1[s1:], q_src[s1:]) and np.array_equal(q1[c], q_src[c])
    frozen = np.setdiff1d(np.arange(J), p1.cols)
    assert np.array_equal(q1[:, frozen], q_src[:, frozen])


def test_refinement_reports_cosine_between_initial_and_final_direction():
    data, phase = make_clip()
    _, _, info, _ = run(data, phase, frac=0.40, refine_iters=3)
    cos = info["refine"]["cos_initial_final"]
    assert -1.0 - 1e-9 <= cos <= 1.0 + 1e-9
    if info["refine"]["iters_used"] == 0:
        assert cos == pytest.approx(1.0)


# -------------------------------------------------------- URDF limits are fail-loud -- #
def test_unreachable_target_fails_loud():
    data, phase = make_clip()
    # box every arm joint in tightly -> +200% deepening is impossible
    tight = {n: (-0.05, 1.0) for n in ARM}
    with pytest.raises(SystemExit, match="unreachable"):
        run(data, phase, frac=2.0, lim=limits(tight))


def test_morph_never_exceeds_limits():
    data, phase = make_clip()
    lo, hi = -0.35, 0.90
    q_out, plan, _, _ = run(data, phase, frac=0.20,
                            lim=limits({"waist_yaw_joint": (lo, hi)}))
    col = NAMES.index("waist_yaw_joint")
    src = np.asarray(data["joint_pos"], dtype=np.float64)[:, col]
    eff_lo, eff_hi = min(lo, src.min()), max(hi, src.max())   # grandfathered
    assert q_out[:, col].min() >= eff_lo - 1e-7
    assert q_out[:, col].max() <= eff_hi + 1e-7


def _pin_waist_pitch(data) -> dict:
    """URDF box collapsed onto waist_pitch's (constant) source value: zero headroom BOTH ways."""
    v = float(np.asarray(data["joint_pos"], dtype=np.float64)[:, C_WAIST_PITCH].max())
    return limits({"waist_pitch_joint": (v, v)})


def test_pinned_joint_is_blocked_and_dropped_with_warning(capsys):
    """v5 反手的 waist_pitch 形态:被 retarget 钳死在硬限位上,在加深方向零余量。"""
    data, phase = make_clip()
    joints = ARM + ["waist_pitch_joint"]
    _, plan, info, _ = run(data, phase, joints=joints, lim=_pin_waist_pitch(data),
                           on_blocked="drop")
    assert "waist_pitch_joint" in info["joints_blocked"]
    assert "waist_pitch_joint" not in info["joints_used"]
    assert C_WAIST_PITCH not in list(plan.cols)
    assert "ZERO deepening headroom" in capsys.readouterr().err
    # the surviving joints still hit the target — one pinned joint must not veto the set
    assert info["extend_frac_out"] == pytest.approx(0.20, rel=2e-3)


def test_pinned_joint_has_real_leverage_so_blocking_is_not_the_zero_weight_path():
    """Guard the guard: waist_pitch must have a NONZERO blade gradient in the stub."""
    data, phase = make_clip()
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    g = es.deep_gradient(stub_blade, q, d=13, c=23, cols=np.array([C_WAIST_PITCH]))
    assert abs(g[0]) > 1e-3


def test_pinned_joint_fails_loud_when_asked():
    data, phase = make_clip()
    with pytest.raises(SystemExit, match="ZERO deepening headroom"):
        run(data, phase, joints=ARM + ["waist_pitch_joint"],
            lim=_pin_waist_pitch(data), on_blocked="fail")


def test_strict_limits_rejects_saturated_source():
    data, phase = make_clip()
    v = float(np.asarray(data["joint_pos"], dtype=np.float64)[:, C_WAIST_PITCH].max())
    lim = limits({"waist_pitch_joint": (-0.1, v - 0.01)})       # source pokes out the top
    with pytest.raises(SystemExit, match="strict-limits"):
        run(data, phase, lim=lim, strict_limits=True)
    # non-strict: source saturation recorded (grandfathered), not fatal
    _, _, info, _ = run(data, phase, lim=lim, strict_limits=False)
    assert "waist_pitch_joint" in info["src_saturated_hi"]


# ------------------------------------------------------- 晚六 design law enforcement -- #
def test_forbidden_joints_rejected():
    assert set(es.PRESETS["armchain"]) == {
        "right_shoulder_yaw_joint", "right_shoulder_pitch_joint",
        "right_elbow_joint", "waist_yaw_joint"}
    forbid = es.resolve_forbid("legs,waist_pitch_roll")
    assert "waist_pitch_joint" in forbid and "waist_roll_joint" in forbid
    assert "left_knee_joint" in forbid and "right_ankle_roll_joint" in forbid
    # the v2 preset is disjoint from the forbidden set — the law holds by construction
    assert not (set(es.PRESETS["armchain"]) & set(forbid))
    # ...and the v1 waist preset is NOT (it is the negative control)
    assert set(es.PRESETS["waist"]) & set(forbid)


def test_forbid_alias_unknown_token():
    with pytest.raises(SystemExit):
        es.resolve_forbid("elbows")


def test_unknown_joint_name_rejected():
    with pytest.raises(SystemExit):
        es.resolve_joint_set("right_thumb_joint")


# ------------------------------------------------------------------ degenerate cases - #
def _monotone_forehand():
    """All driven joints monotone -> the blade sweeps one way -> deepest pre-contact frame = 0."""
    data, phase = make_clip()
    q = np.asarray(data["joint_pos"])
    T = q.shape[0]
    q[:, C_SH_PITCH] = np.linspace(-0.9, 0.6, T)
    q[:, C_ELBOW] = np.linspace(0.20, 0.90, T)
    q[:, C_WAIST_YAW] = np.linspace(-0.25, 0.25, T)
    data["joint_pos"] = q
    data["body_pos_w"][:, st.RACKET_BODY] = stub_blade(q.astype(np.float64)).astype(np.float32)
    return data, phase


def test_forehand_deep_at_zero_fails_loud_with_gap_message():
    """正手 d = 0(最深点即起手帧)→ 片内没有引拍窗。这是 v1 的已知工具缺口。"""
    data, phase = _monotone_forehand()
    blade = stub_blade(np.asarray(data["joint_pos"], dtype=np.float64))
    assert es.deep_frame_and_L(blade, c=23)[0] == 0        # the premise of this test
    with pytest.raises(SystemExit, match="引拍窗"):
        run(data, phase)


def test_peak_frame_override_rescues_degenerate_clip():
    data, phase = _monotone_forehand()
    _, _, info, _ = run(data, phase, peak_frame=10)
    assert info["peak_frame"] == 10
    assert info["extend_frac_out"] == pytest.approx(0.20, rel=2e-3)


def test_lock_window_eating_runup_fails():
    data, phase = make_clip(T=60, contact=1)
    with pytest.raises(SystemExit):
        run(data, 1 / 59)


def test_unknown_keys_refused():
    data, phase = make_clip()
    data["mystery"] = np.zeros((3, 2))
    with pytest.raises(SystemExit, match="unknown npz keys"):
        run(data, phase)


def test_a_min_formula():
    assert es.a_min_of(3.405, 0.497) == pytest.approx(11.66, rel=1e-3)   # ledger v5hLs bh
    assert es.a_min_of(2.488, 1.130) == pytest.approx(2.74, rel=1e-3)    # ledger v5hLs fh
