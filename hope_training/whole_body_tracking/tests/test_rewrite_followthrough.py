"""Unit tests for scripts/rewrite_followthrough.py (随挥段路径重写).

Pure CPU, NO mujoco: the module is loaded by file path; the production MuJoCo pieces
(self-collision gate, oracle dose scorer, CoM) are replaced by STUBS injected through the
same callables the searcher uses in production, so the search/guard/acceptance logic is
exercised for real, on a synthetic clip.

Covered (franco 07-10 定向的设计法则,逐条):
  * 锁段逐位: [0, c+2] AND the end pose bitwise; frozen joints bitwise; only the
    rewrite-domain interior rows may change
  * 接缝 C1: basis/plateau vanish with zero slope at both support endpoints; the FD
    velocity at c+2 (central) and at the last frame (one-sided) is preserved EXACTLY
  * 限位拦截: candidates beyond the (grandfathered) URDF box are rejected, final
    trajectory never exceeds it; strict mode refuses a saturated source
  * 自碰撞拦截 (C 在环): a stub collision gate blocks candidates BEFORE scoring and the
    search cannot cross it even when the objective says "go"
  * CoM 保险丝: candidates whose stub CoM displacement exceeds eps are rejected
  * 摩擦/τ 剂量守卫: a CoP improvement that worsens the friction dose is rejected
  * 剂量单调改善验收: accepted candidates form a strictly improving Score.key() sequence;
    the final dose is <= every accepted dose and < the source dose on a solvable problem
  * blend 模板: β=1 puts the plateau frames exactly on the joint-space straight line to
    the (unchanged) end pose; blend mode moves ONLY β
  * fail-loud: forbidden joints (legs / waist pitch/roll), unknown npz keys, degenerate
    rewrite window, dirty source (self-collision inside the domain)
  * 缝合契约: 派生速度只拼脏带 [lo-1, hi+1],带外行(含锁窗与末两帧)逐位保留源片
    存量——即使源片速度不是 gradient(存量 joint_pos) 的产物(float64 母带出身);
    changed 为空时输出六场与源片逐位相同;verify_output_contract fail-loud

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_rewrite_followthrough.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _name in ("synthesize_timing", "extend_stroke", "rewrite_followthrough"):
    _spec = importlib.util.spec_from_file_location(_name, _SCRIPTS / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
rf = sys.modules["rewrite_followthrough"]
es = sys.modules["extend_stroke"]

FPS = 50.0
J = 31
T = 60
CONTACT = 23
NAMES = rf.ISAAC_JOINT_NAMES

C_SH_PITCH = NAMES.index("right_shoulder_pitch_joint")
C_ELBOW = NAMES.index("right_elbow_joint")
C_WAIST_YAW = NAMES.index("waist_yaw_joint")

ARM3 = ["right_shoulder_pitch_joint", "right_elbow_joint", "waist_yaw_joint"]


# ------------------------------------------------------------------ synthetic clip -- #
def make_q(T_: int = T) -> np.ndarray:
    """Backhand-shaped joint trajectory with a lively follow-through to rewrite."""
    q = np.zeros((T_, J), dtype=np.float64)
    s = np.arange(T_, dtype=np.float64)
    q[:, C_SH_PITCH] = -0.9 * np.sin(np.pi * np.clip(s / 26.0, 0, 1)) + 0.6 * s / T_
    q[:, C_ELBOW] = 0.55 + 0.35 * np.cos(np.pi * s / T_)
    q[:, C_WAIST_YAW] = 0.25 * np.sin(2 * np.pi * s / (3 * T_))
    return q


def make_plan(q: np.ndarray, joints=ARM3, mode: str = "hybrid", K: int = 4):
    T_ = q.shape[0]
    s0, s1 = rf.rewrite_windows(T_, CONTACT)
    cols = np.array([NAMES.index(n) for n in joints])
    basis, peaks = rf.bump_basis(T_, s0, s1, K)
    template = None
    if mode in ("blend", "hybrid"):
        W = rf.plateau_window(T_, s0, s1)
        template = rf.retreat_template(q, cols, CONTACT, s0, s1, W)
    return rf.RewritePlan(cols=cols, names=list(joints), basis=basis, peaks=peaks,
                          template=template, coef=np.zeros((len(joints), len(basis))),
                          beta=0.0, s0=s0, s1=s1, c=CONTACT)


def pass_guard(q_out):
    return True, ""


class StubScorer:
    """Score = share of domain frames where shoulder pitch sits below `target`.

    Raising the shoulder-pitch coefficients monotonically improves it, so the searcher
    has a real gradient to follow; extra hooks let individual tests inject friction
    worsening / CoM displacement without touching the search code."""

    def __init__(self, q_src, s0, s1, target=0.5,
                 fric_fn=None, com_fn=None, tau_fn=None):
        self.s0, self.s1, self.target = s0, s1, target
        self.fric_fn = fric_fn or (lambda q: 0.0)
        self.tau_fn = tau_fn or (lambda q: 0.0)
        self.com_fn = com_fn or (lambda q: 0.0)
        self.calls = 0

    def __call__(self, q):
        self.calls += 1
        dom = q[self.s0 + 1: self.s1, C_SH_PITCH]
        gap = np.maximum(self.target - dom, 0.0)
        return rf.Score(dose_cop=float(np.mean(gap > 0.0)),
                        dose_fric=float(self.fric_fn(q)),
                        dose_tau=float(self.tau_fn(q)),
                        cop_area=float(gap.sum()),
                        com_dxy=float(self.com_fn(q)),
                        cop_frames=[int(t) for t in
                                    np.flatnonzero(gap > 0.0) + self.s0 + 1])


def run_search(q, plan, scorer, guards=None, **kw):
    log = []
    out = rf.coordinate_search(q, plan, scorer, guards or [], log=log, **kw)
    return (*out, log)


# ------------------------------------------------------------------ windows / basis - #
def test_rewrite_windows_v5hLs_numbers():
    # the real target clip: T=59, phase 0.391 -> c=23; lock [0,25], support [26,56]
    # (s1 = T-3: so3_derivative 末行拷贝的 stencil 读 q[T-3],那行必须留在域外)
    s0, s1 = rf.rewrite_windows(59, 23)
    assert (s0, s1) == (26, 56)


def test_rewrite_windows_too_short_fails_loud():
    with pytest.raises(SystemExit, match="太靠片尾"):
        rf.rewrite_windows(32, 25)      # support [28, 29] -> 1 frame < MIN_DOMAIN


def test_bump_basis_c1_support_and_peaks():
    s0, s1 = rf.rewrite_windows(T, CONTACT)
    B, peaks = rf.bump_basis(T, s0, s1, 4)
    assert B.shape == (4, T) and len(set(peaks)) == 4
    for k, p in enumerate(peaks):
        assert s0 < p < s1
        assert B[k, p] == pytest.approx(1.0)
        assert B[k, s0] == 0.0 and B[k, s1] == 0.0
        assert (B[k, : s0] == 0.0).all() and (B[k, s1:] == 0.0).all()


def test_bump_basis_clamps_k_when_domain_short(capsys):
    B, peaks = rf.bump_basis(40, 30, 38, 50)      # only 7 interior frames
    assert len(peaks) == 7
    assert "clamped" in capsys.readouterr().err


def test_plateau_window_c1_values():
    s0, s1, r = 26, 58, 6
    W = rf.plateau_window(T, s0, s1, ramp=r)
    assert W[s0] == 0.0 and W[s1] == 0.0
    assert (W[: s0] == 0.0).all() and (W[s1 + 1:] == 0.0).all()
    assert (W[s0 + r: s1 - r + 1] == 1.0).all()
    x = 1.0 / r                                    # first step off the ramp foot
    assert W[s0 + 1] == pytest.approx(3 * x**2 - 2 * x**3)   # smoothstep => slope -> 0
    assert W[s1 - 1] == pytest.approx(3 * x**2 - 2 * x**3)


def test_retreat_template_hits_straight_line_at_beta_one():
    q = make_q()
    plan = make_plan(q, mode="blend")
    s0, s1, r = plan.s0, plan.s1, max(2, (plan.s1 - plan.s0) // 5)
    cand = rf.replace(plan, beta=1.0, coef=plan.coef.copy())
    q_out = cand.apply(q)
    a, E = CONTACT + rf.LOCK_AFTER_CONTACT, T - 1
    for t in range(s0 + r, s1 - r + 1):            # plateau: W == 1 -> exactly the line
        frac = (t - a) / (E - a)
        line = q[a, plan.cols] * (1 - frac) + q[E, plan.cols] * frac
        np.testing.assert_allclose(q_out[t, plan.cols], line, atol=1e-12)
    assert np.array_equal(q_out[: s0 + 1], q[: s0 + 1])      # seams still bitwise
    assert np.array_equal(q_out[s1:], q[s1:])


# ------------------------------------------------------------- lock / seam invariants #
def test_lock_end_and_frozen_bitwise_under_random_plan():
    q = make_q()
    plan = make_plan(q)
    rng = np.random.default_rng(7)
    cand = rf.replace(plan, beta=0.63, coef=rng.normal(0, 0.2, plan.coef.shape))
    q_out = cand.apply(q)
    rf.assert_structure(q, q_out, plan.cols, plan.s0, plan.s1)  # must not raise
    assert np.array_equal(q_out[: CONTACT + 3], q[: CONTACT + 3])   # lock + seam head
    assert np.array_equal(q_out[-2:], q[-2:])                       # end pose + stencil row
    frozen = np.setdiff1d(np.arange(J), plan.cols)
    assert np.array_equal(q_out[:, frozen], q[:, frozen])
    assert not np.array_equal(q_out, q)                             # it DID rewrite something


def test_seam_velocity_preserved_exactly():
    q = make_q()
    plan = make_plan(q)
    cand = rf.replace(plan, beta=0.8, coef=np.full(plan.coef.shape, 0.3))
    q_out = cand.apply(q)
    r = rf.seam_residuals(q, q_out, CONTACT, FPS)
    assert r["lock_window_bitwise"] and r["contact_row_bitwise"] and r["end_pose_bitwise"]
    assert r["vel_residual_at_lock_end"] == 0.0
    assert r["vel_residual_at_end"] == 0.0


def test_assert_structure_catches_leaks():
    q = make_q()
    plan = make_plan(q)
    bad = q.copy()
    bad[CONTACT] += 1e-3                            # touch the contact row
    with pytest.raises(SystemExit, match="REWRITE BUG"):
        rf.assert_structure(q, bad, plan.cols, plan.s0, plan.s1)
    bad2 = q.copy()
    bad2[plan.s0 + 2, 0] += 1e-3                    # move a frozen (leg) joint
    with pytest.raises(SystemExit, match="frozen"):
        rf.assert_structure(q, bad2, plan.cols, plan.s0, plan.s1)


# ------------------------------------------------------------------ search behaviour - #
def test_search_improves_dose_monotonically():
    q = make_q()
    plan = make_plan(q, mode="field")
    scorer = StubScorer(q, plan.s0, plan.s1)
    best_plan, src, best, stats, log = run_search(q, plan, scorer, mode="field",
                                                  max_passes=6, max_evals=600)
    acc = [r for r in log if r["accepted"]]
    assert len(acc) >= 2, "search found no improving step on a solvable problem"
    keys = [(r["dose_cop"], r["cop_area"], r["dose_fric"], r["dose_tau"]) for r in acc]
    assert all(keys[i + 1] < keys[i] for i in range(len(keys) - 1)), "剂量键必须单调变好"
    assert best.key() < src.key()
    assert best.dose_cop < src.dose_cop or best.cop_area < src.cop_area
    q_out = best_plan.apply(q)
    rf.assert_structure(q, q_out, plan.cols, plan.s0, plan.s1)


def test_search_respects_limits_guard():
    q = make_q()
    plan = make_plan(q, mode="field")
    scorer = StubScorer(q, plan.s0, plan.s1, target=5.0)   # insatiable: always push up

    class LimitStub:
        def __init__(self, lo, hi):
            self.lower, self.upper = lo, hi
            self.effort, self.velocity = 100.0, 10.0

    hi_cap = float(q[:, C_SH_PITCH].max()) + 0.30
    lims = {n: LimitStub(-3.0, 3.0) for n in NAMES}
    lims["right_shoulder_pitch_joint"] = LimitStub(-3.0, hi_cap)
    guard, sat_hi, sat_lo = rf.make_limits_guard(q, plan.cols, lims, strict=False)
    assert sat_hi == [] and sat_lo == []
    best_plan, _, _, stats, _ = run_search(q, plan, scorer, guards=[("limits", guard)],
                                           mode="field", max_passes=6, max_evals=600)
    assert stats["rejects"].get("limits", 0) > 0, "the guard was never exercised"
    q_out = best_plan.apply(q)
    assert q_out[:, C_SH_PITCH].max() <= hi_cap + 1e-7


def test_strict_limits_refuses_saturated_source():
    q = make_q()
    plan = make_plan(q)

    class LimitStub:
        def __init__(self, lo, hi):
            self.lower, self.upper = lo, hi
            self.effort, self.velocity = 100.0, 10.0

    lims = {n: LimitStub(-3.0, 3.0) for n in NAMES}
    lims["right_elbow_joint"] = LimitStub(-3.0, float(q[:, C_ELBOW].max()) - 0.05)
    with pytest.raises(SystemExit, match="strict-limits"):
        rf.make_limits_guard(q, plan.cols, lims, strict=True)
    guard, sat_hi, _ = rf.make_limits_guard(q, plan.cols, lims, strict=False)
    assert "right_elbow_joint" in sat_hi            # grandfathered, recorded by name


def test_selfcol_gate_blocks_before_scoring():
    q = make_q()
    plan = make_plan(q, mode="field")
    ceiling = float(q[:, C_SH_PITCH].max()) + 0.40
    scored_q: list[np.ndarray] = []

    def collide_guard(q_out):
        if q_out[:, C_SH_PITCH].max() > ceiling:
            return False, "stub racket<->torso"
        return True, "clean"

    class RecordingScorer(StubScorer):
        def __call__(self, qq):
            scored_q.append(qq)
            return super().__call__(qq)

    scorer = RecordingScorer(q, plan.s0, plan.s1, target=5.0)
    best_plan, _, _, stats, _ = run_search(q, plan, scorer,
                                           guards=[("selfcol", collide_guard)],
                                           mode="field", max_passes=6, max_evals=600)
    assert stats["rejects"].get("selfcol", 0) > 0
    q_out = best_plan.apply(q)
    assert q_out[:, C_SH_PITCH].max() <= ceiling + 1e-9
    for qq in scored_q:                              # C 在环: 先自碰撞, 后打分
        assert qq[:, C_SH_PITCH].max() <= ceiling + 1e-9, \
            "a colliding candidate reached the oracle scorer"


def test_com_fuse_blocks():
    q = make_q()
    plan = make_plan(q, mode="field")
    com = lambda qq: 0.6 * float(np.abs(qq[:, C_WAIST_YAW] - q[:, C_WAIST_YAW]).max())
    scorer = StubScorer(q, plan.s0, plan.s1, target=5.0, com_fn=com)
    best_plan, _, best, stats, _ = run_search(q, plan, scorer, mode="field",
                                              max_passes=6, max_evals=600,
                                              com_eps=rf.COM_EPS_M)
    assert stats["rejects"].get("com_fuse", 0) > 0
    assert best.com_dxy <= rf.COM_EPS_M + 1e-12
    q_out = best_plan.apply(q)
    assert 0.6 * np.abs(q_out[:, C_WAIST_YAW] - q[:, C_WAIST_YAW]).max() <= rf.COM_EPS_M + 1e-12


def test_dose_guard_blocks_friction_worsening():
    q = make_q()
    plan = make_plan(q, mode="field")
    # ANY deviation from the source worsens friction => every improving move is rejected
    fric = lambda qq: 0.0 if np.array_equal(qq, q) else 0.5
    scorer = StubScorer(q, plan.s0, plan.s1, fric_fn=fric)
    best_plan, src, best, stats, log = run_search(q, plan, scorer, mode="field",
                                                  max_passes=3, max_evals=300)
    assert stats["rejects"].get("dose_guard", 0) > 0
    assert stats["accepted_steps"] == 0
    assert best.key() == src.key()
    assert np.array_equal(best_plan.apply(q), q)     # nothing accepted -> bitwise source


def test_blend_mode_moves_only_beta():
    q = make_q()
    plan = make_plan(q, mode="blend")
    scorer = StubScorer(q, plan.s0, plan.s1, target=0.2)
    best_plan, _, _, _, log = run_search(q, plan, scorer, mode="blend",
                                         max_passes=4, max_evals=200)
    assert np.all(best_plan.coef == 0.0)
    assert 0.0 <= best_plan.beta <= 1.0
    for r in log[1:]:
        assert r["coord"].startswith("beta")


def test_no_improvement_terminates_bitwise():
    q = make_q()
    plan = make_plan(q)
    flat = lambda qq: rf.Score(dose_cop=0.5, dose_fric=0.0, dose_tau=0.0, cop_area=1.0)
    best_plan, src, best, stats, log = run_search(q, plan, flat, mode="hybrid",
                                                  max_passes=4, max_evals=400)
    assert stats["accepted_steps"] == 0
    assert stats["rejects"].get("not_better", 0) > 0
    assert np.array_equal(best_plan.apply(q), q)


def test_eval_budget_respected():
    q = make_q()
    plan = make_plan(q, mode="field")
    scorer = StubScorer(q, plan.s0, plan.s1)
    _, _, _, stats, log = run_search(q, plan, scorer, mode="field",
                                     max_passes=50, max_evals=7)
    assert stats["budget_exhausted"]
    assert stats["n_evals"] <= 7
    assert len(log) <= 8                             # baseline + <= max_evals rows


# ------------------------------------------------------------------ fail-loud paths -- #
def test_forbidden_joints_rejected():
    for bad in ("left_knee_joint", "waist_pitch_joint", "waist_roll_joint",
                "right_ankle_roll_joint"):
        with pytest.raises(SystemExit, match="forbidden|冻结"):
            rf.resolve_joints(bad)
    for preset in rf.PRESETS.values():
        assert not (set(preset) & set(rf.FORBIDDEN_JOINTS))
    assert rf.resolve_joints("arm5") == list(rf.PRESETS["arm5"])


def test_unknown_joint_rejected():
    with pytest.raises(SystemExit, match="unknown joint"):
        rf.resolve_joints("right_thumb_joint")


def test_unknown_npz_keys_refused():
    data = {"fps": np.array([50]), "joint_pos": np.zeros((10, J), np.float32),
            "mystery": np.zeros(3)}
    with pytest.raises(SystemExit, match="unknown npz keys"):
        rf.validate_npz(data)


def test_bad_joint_dim_refused():
    data = {"fps": np.array([50]), "joint_pos": np.zeros((10, 7), np.float32)}
    with pytest.raises(SystemExit, match="joint_pos shape"):
        rf.validate_npz(data)


def test_dirty_source_refused():
    dirty = lambda qq: (False, "1 帧互穿, worst racket<->torso")
    with pytest.raises(SystemExit, match="拒绝在脏源"):
        rf.require_clean_source(dirty, make_q())


def test_score_key_ordering():
    a = rf.Score(dose_cop=0.30, dose_fric=0.9, dose_tau=0.9, cop_area=9.0)
    b = rf.Score(dose_cop=0.31, dose_fric=0.0, dose_tau=0.0, cop_area=0.1)
    assert a.key() < b.key()                         # CoP dose dominates everything
    c1 = rf.Score(dose_cop=0.30, dose_fric=0.0, dose_tau=0.0, cop_area=1.0)
    c2 = rf.Score(dose_cop=0.30, dose_fric=0.0, dose_tau=0.0, cop_area=2.0)
    assert c1.key() < c2.key()                       # area is the continuous tie-break

# ------------------------------------------------------------------ stitched rebuild -- #
class StubFK:
    """Pure-CPU linear FK stand-in: pelvis + one 'blade' body driven by shoulder pitch."""

    def __init__(self):
        self.names = ["pelvis_link", "blade_link"]

    def body_names(self):
        return self.names

    def fk(self, base_pos, base_quat, jd):
        v = float(jd["right_shoulder_pitch_joint"])
        p = np.stack([base_pos,
                      base_pos + np.array([0.3 * np.cos(v), 0.3 * np.sin(v), 0.5])])
        half = 0.5 * v
        q = np.stack([base_quat, np.array([np.cos(half), 0.0, 0.0, np.sin(half)])])
        return p, q

    def fk_with_com(self, base_pos, base_quat, jd):
        # Zero inertial offsets are sufficient for this pure-CPU seam test;
        # production MjFK returns MuJoCo data.xipos as the third value.
        p, q = self.fk(base_pos, base_quat, jd)
        return p, q, p.copy()


def make_npz_with_foreign_velocity_provenance(T_: int = T):
    """Source npz whose velocity fields are NOT gradient(stored positions) bitwise —
    模拟 float64 母带出身的源片(0710 对抗复核在 v5hLs 上实测差 4e-6)。"""
    fk = StubFK()
    q = make_q(T_).astype(np.float32)
    base_pos = np.zeros((T_, 3)) + np.array([0.1, 0.2, 0.8])
    base_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T_, 1))
    bp = np.zeros((T_, 2, 3), np.float32)
    bq = np.zeros((T_, 2, 4), np.float32)
    for t in range(T_):
        p, qm = fk.fk(base_pos[t], base_quat[t], dict(zip(NAMES, q[t].astype(np.float64))))
        bp[t], bq[t] = p.astype(np.float32), qm.astype(np.float32)
    dt = 1.0 / FPS
    rng = np.random.default_rng(3)
    noise = lambda shape: rng.uniform(-4e-6, 4e-6, shape)
    jv = (np.gradient(q.astype(np.float64), dt, axis=0) + noise(q.shape)).astype(np.float32)
    bl = (np.gradient(bp.astype(np.float64), dt, axis=0) + noise(bp.shape)).astype(np.float32)
    ba = np.zeros((T_, 2, 3), np.float32) + noise((T_, 2, 3)).astype(np.float32)
    data = dict(fps=np.array([50], np.int64), joint_pos=q, joint_vel=jv,
                body_pos_w=bp, body_quat_w=bq, body_lin_vel_w=bl, body_ang_vel_w=ba)
    data.update(rf.metadata_arrays(body_names=["pelvis_link", "body_1"]))
    assert not np.array_equal(np.gradient(q.astype(np.float64), dt, axis=0).astype(np.float32), jv)
    return data, fk


VEL_FIELDS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
ALL_FIELDS = ("joint_pos", "body_pos_w", "body_quat_w") + VEL_FIELDS


def test_rebuild_splices_velocities_only_in_dirty_band():
    data, fk = make_npz_with_foreign_velocity_provenance()
    q_src = np.asarray(data["joint_pos"], np.float64)
    plan = make_plan(q_src, joints=["right_shoulder_pitch_joint"], mode="field", K=4)
    cand = rf.replace(plan, coef=np.full(plan.coef.shape, 0.25))
    q_out = cand.apply(q_src)
    out, acc = rf.rebuild_npz_stitched(data, q_out, fk, [0, 1], plan.s0, plan.s1)
    lo, hi = min(acc["changed_frames"]), max(acc["changed_frames"])
    assert hi == plan.s1 - 1, "test must exercise the last rewritable row"
    assert acc["vel_dirty_band"] == [lo - 1, hi + 1]
    # 契约:脏带外(含锁窗与末两帧)六场逐位 = 源片,哪怕源片速度出处不同
    for k in ALL_FIELDS:
        assert np.array_equal(out[k][: lo - 1], np.asarray(data[k])[: lo - 1]), k
        assert np.array_equal(out[k][hi + 2:], np.asarray(data[k])[hi + 2:]), k
    for k in ("joint_pos", "body_pos_w", "body_quat_w"):   # pose 场更严:域外逐位
        assert np.array_equal(out[k][: lo], np.asarray(data[k])[: lo]), k
    # 末两帧(含 so3 末行拷贝)与锁窗必须逐位——0710 复核抓出的两处主病灶
    for k in ALL_FIELDS:
        assert np.array_equal(out[k][-2:], np.asarray(data[k])[-2:]), k
        assert np.array_equal(out[k][: CONTACT + 3], np.asarray(data[k])[: CONTACT + 3]), k
    # 带内:确为重算值(gradient of the NEW positions),不是源片存量
    dt = 1.0 / FPS
    jv_re = np.gradient(out["joint_pos"].astype(np.float64), dt, axis=0).astype(np.float32)
    assert np.array_equal(out["joint_vel"][lo - 1: hi + 2], jv_re[lo - 1: hi + 2])
    rep = rf.verify_output_contract(data, out, CONTACT, plan.s0, plan.s1)  # must not raise
    assert rep["all_fields_bitwise_outside"]


def test_rebuild_noop_is_bitwise_identity():
    data, fk = make_npz_with_foreign_velocity_provenance()
    q_src = np.asarray(data["joint_pos"], np.float64)
    plan = make_plan(q_src, joints=["right_shoulder_pitch_joint"], mode="field")
    out, acc = rf.rebuild_npz_stitched(data, q_src.copy(), fk, [0, 1], plan.s0, plan.s1)
    assert acc["n_changed"] == 0 and acc["vel_dirty_band"] is None
    for k in ALL_FIELDS:
        assert np.array_equal(out[k], np.asarray(data[k])), f"{k} 无改动时必须逐位恒等"


def test_verify_output_contract_catches_lock_leak():
    data, fk = make_npz_with_foreign_velocity_provenance()
    q_src = np.asarray(data["joint_pos"], np.float64)
    plan = make_plan(q_src, joints=["right_shoulder_pitch_joint"], mode="field")
    out, _ = rf.rebuild_npz_stitched(data, q_src.copy(), fk, [0, 1], plan.s0, plan.s1)
    out["joint_vel"] = np.array(out["joint_vel"], copy=True)
    out["joint_vel"][2, 0] += 1e-3                   # poison a lock-window velocity row
    with pytest.raises(SystemExit, match="泄漏"):
        rf.verify_output_contract(data, out, CONTACT, plan.s0, plan.s1)
