"""Unit tests for scripts/topp_budget_search.py (TOPP v2 外环预算搜索).

纯 CPU、不需要 mujoco/torch:模块按文件路径装载(同 test_synthesize_timing.py 手法),
用合成小 clip + 注入的假打分器测搜索逻辑本身——这正是 run_search 留 scorer 接缝的目的。
真打分器(OracleScorer/mj_inverse)在 pod 的 mjeval venv 里用真 clip 冒烟验证,不进单测。

覆盖(任务规格 2026-07-09):
  * 单调性:T_a 增大 → 候选输出真的变温和(匀加速段加速度 = q'·sdot*/T_a),
    连续型 τ 剂量不增
  * tight/fill 语义:tight = 不可行秒数降到温和端地板的最小 T_a;
    fill = 预算窗内最大可行 T_a(不给预警时间时 = ta_max)
  * timing-irreducible 检测:构造姿态绑定的违规(触球后路径段,任何 T_a 都要经过)
    → 地板剂量 > 0、每档都带同样残余、reducible/irreducible 分开报告
  * fail-loud:缺登记相位 / budget_frac 出界 / 未知 npz key / 打分器缺字段 /
    预警时间内无可行档 / 触球帧贴边 —— 全部拒跑

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_topp_budget_search.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "topp_budget_search.py"
_spec = importlib.util.spec_from_file_location("topp_budget_search", _SCRIPT)
tbs = importlib.util.module_from_spec(_spec)
sys.modules["topp_budget_search"] = tbs
_spec.loader.exec_module(tbs)
syn = tbs.syn

FPS = 50.0
J = 31
NB = 32          # 32 bodies,col 31 = right_wrist_yaw_Link(拍腕)


def make_clip(T=61, contact=40, blade_step=0.02, joint_amp=1.0):
    """合成源 clip(照 test_synthesize_timing 的手法):
    右肘(col 24)在路径上匀速走 0→joint_amp,拍腕沿 +x 每帧 blade_step 米、
    姿态恒等 → 拍路径导数恰 = blade_step m/frame,clean 拍速 = blade_step×FPS m/s。"""
    q = np.zeros((T, J), dtype=np.float32)
    q[:, 24] = np.linspace(0.0, joint_amp, T, dtype=np.float32)
    dq = np.gradient(q.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    bp = np.zeros((T, NB, 3), dtype=np.float32)
    bp[:, syn.RACKET_BODY, 0] = np.arange(T, dtype=np.float32) * blade_step
    bq = np.zeros((T, NB, 4), dtype=np.float32)
    bq[..., 0] = 1.0
    bl = np.gradient(bp.astype(np.float64), 1.0 / FPS, axis=0).astype(np.float32)
    ba = np.zeros_like(bp)
    return {
        "fps": np.array([int(FPS)], dtype=np.int64),
        "joint_pos": q, "joint_vel": dq,
        "body_pos_w": bp, "body_quat_w": bq,
        "body_lin_vel_w": bl, "body_ang_vel_w": ba,
    }, contact / (T - 1)


# ------------------------------------------------------------------ 假打分器 ----- #
def _acc_frames(npz_path):
    d = np.load(npz_path)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    q = np.asarray(d["joint_pos"], dtype=np.float64)
    acc = np.abs(np.diff(q, 2, axis=0)) * fps * fps   # (T-2, J) 二阶差分当"力矩需求"
    return d, fps, q, acc


def _zero_scores():
    return dict(dose_tau=0.0, sec_tau=0.0, dose_cop=0.0, sec_cop=0.0,
                dose_fric=0.0, sec_fric=0.0)


class AccMaxScorer:
    """连续型 τ 代理:剂量 = max(0, max|q̈|/cap - 1)(单位惯量,把加速度当力矩)。
    时间律里匀加速段 |q̈| = q'·sdot*/T_a,T_a 越大越温和 → 剂量单调不增,
    测的是"搜索产出的候选确实随 T_a 变温和"这个物理内容。"""
    def __init__(self, cap):
        self.cap = cap

    def __call__(self, npz_path, phase_out):
        _, _, _, acc = _acc_frames(npz_path)
        excess = max(0.0, float(acc.max()) / self.cap - 1.0)
        s = _zero_scores()
        s.update(dose_tau=excess, sec_tau=excess, max_acc=float(acc.max()))
        return s


class AccBinaryScorer:
    """逐帧超限型 τ 代理(oracle 剂量制同口径):违规帧 = 任一关节 |q̈| > cap,
    剂量 = 占比,秒数 = 帧数/fps。"""
    def __init__(self, cap):
        self.cap = cap

    def __call__(self, npz_path, phase_out):
        _, fps, _, acc = _acc_frames(npz_path)
        viol = (acc > self.cap).any(axis=1)
        s = _zero_scores()
        s.update(dose_tau=float(viol.sum()) / max(len(viol), 1),
                 sec_tau=float(viol.sum()) / fps)
        return s


class PostureScorer:
    """姿态绑定的违规(CoP 的 CPU 类比):|q24| > thresh 的帧算违规——
    这由路径位置决定,任何单调时间律都要经过同一段路径 → 时序不可约。"""
    def __init__(self, thresh=0.8):
        self.thresh = thresh

    def __call__(self, npz_path, phase_out):
        _, fps, q, _ = _acc_frames(npz_path)
        viol = np.abs(q[1:-1, 24]) > self.thresh
        s = _zero_scores()
        s.update(dose_cop=float(viol.sum()) / max(len(viol), 1),
                 sec_cop=float(viol.sum()) / fps)
        return s


class ComboScorer:
    """τ(可约,加速度型)+ CoP(不可约,姿态型)混合——测 reducible/irreducible 拆分。"""
    def __init__(self, cap, thresh=0.8):
        self.a, self.p = AccBinaryScorer(cap), PostureScorer(thresh)

    def __call__(self, npz_path, phase_out):
        s = self.a(npz_path, phase_out)
        s.update({k: v for k, v in self.p(npz_path, phase_out).items()
                  if k in ("dose_cop", "sec_cop")})
        return s


def search(tmp_path, data, phase, scorer, mode="tight", budget_frac=0.7, **kw):
    kw.setdefault("ta_grid_s", 0.2)
    kw.setdefault("body_mode", "interp")
    return tbs.run_search(data, phase, budget_frac, mode, scorer,
                          tmp_path / "cands", "clip", **kw)


# --------------------------------------------------------------- ① 单调性 -------- #
def test_ta_monotone_gentler(tmp_path):
    """T_a 网格升序 → 候选的峰值加速度不增(匀加速段 ∝ 1/T_a,尾端被固定的收拍
    减速段托底)→ 连续型 τ 剂量不增。"""
    data, phase = make_clip()
    rep, _ = search(tmp_path, data, phase, AccMaxScorer(cap=0.5), mode="fill")
    cands = rep["candidates"]
    tas = [cd["Ta_s"] for cd in cands]
    assert tas == sorted(tas) and len(tas) >= 5
    doses = [cd["dose_tau"] for cd in cands]
    assert all(a >= b - 1e-3 for a, b in zip(doses, doses[1:]))   # 不增
    assert doses[0] > doses[-1]                                   # 且真的降了
    maxacc = [cd["max_acc"] for cd in cands]
    assert all(a >= b - 1e-3 for a, b in zip(maxacc, maxacc[1:]))


# --------------------------------------------------------- ② tight/fill 语义 ----- #
def test_tight_picks_smallest_feasible_ta(tmp_path):
    """cap=2.0:匀加速段 |q̈| = 0.833/T_a → T_a∈{0.2,0.4} 违规、0.6 起干净。
    tight 必须选 0.6(最小可行),更紧的档必须标不可行。"""
    data, phase = make_clip()
    rep, _ = search(tmp_path, data, phase, AccBinaryScorer(cap=2.0), mode="tight")
    assert rep["selection"]["chosen_Ta_s"] == pytest.approx(0.6, abs=1e-6)
    by_ta = {cd["Ta_s"]: cd for cd in rep["candidates"]}
    assert not by_ta[0.2]["acceptable"] and not by_ta[0.4]["acceptable"]
    assert by_ta[0.6]["acceptable"] and by_ta[0.6]["sec_tau"] == 0.0
    assert rep["irreducible"]["sec_tau"] == 0.0          # 地板干净 = 全部可约
    assert not rep["irreducible"]["any_residue"]


def test_fill_picks_gentlest_ta(tmp_path):
    """fill 不给预警时间 = 摊到最温和端(ta_max 本人)。"""
    data, phase = make_clip()
    rep, _ = search(tmp_path, data, phase, AccBinaryScorer(cap=2.0), mode="fill")
    assert rep["selection"]["chosen_Ta_s"] == pytest.approx(rep["grid"]["ta_max_s"], abs=1e-6)
    rep_t, _ = search(tmp_path, data, phase, AccBinaryScorer(cap=2.0), mode="tight")
    assert rep_t["selection"]["chosen_Ta_s"] < rep["selection"]["chosen_Ta_s"]


def test_fill_respects_t_avail_window(tmp_path):
    """预警时间窗:t*(T_a) ≈ 0.8 + 0.5·T_a,t_avail=1.25 → fill 只能摊到 T_a=0.8。"""
    data, phase = make_clip()
    rep, _ = search(tmp_path, data, phase, AccBinaryScorer(cap=2.0), mode="fill",
                    t_avail_s=1.25)
    assert rep["selection"]["chosen_Ta_s"] == pytest.approx(0.8, abs=1e-6)
    assert rep["selection"]["chosen_t_star_s"] <= 1.25 + 1e-9


def test_speed_never_lowered(tmp_path):
    """速度绝不降:答案速度 = 源 clean 拍速,选中档输出拍速偏差 < 2%。"""
    data, phase = make_clip()
    rep, out = search(tmp_path, data, phase, AccBinaryScorer(cap=2.0), mode="tight")
    assert rep["answer"]["v_star_mps"] == pytest.approx(
        rep["source"]["clean_blade_speed_mps"], rel=1e-9)
    assert rep["output"]["blade_speed_dev_frac"] < 0.02
    k, T_out = rep["output"]["contact_frame"], rep["output"]["frames"]
    assert rep["output"]["phase_out"] == pytest.approx(k / (T_out - 1), abs=1e-6)
    assert out["joint_pos"].shape[0] == T_out


# ------------------------------------------------------ ③ irreducible 检测 ------- #
def test_posture_violation_is_irreducible(tmp_path):
    """姿态性违规(q24>0.8,触球后收拍段路径固定)对所有 T_a 秒数一样
    → 地板剂量 > 0、报告 any_residue、每档都可行、tight 退到最小 T_a。"""
    data, phase = make_clip()
    rep, _ = search(tmp_path, data, phase, PostureScorer(0.8), mode="tight",
                    sec_tol=0.06)
    assert rep["irreducible"]["sec_cop"] > 0.5            # 收拍段 ~0.59 s 消不掉
    assert rep["irreducible"]["any_residue"]
    assert all(cd["acceptable"] for cd in rep["candidates"])  # 残余人人平等
    assert rep["selection"]["chosen_Ta_s"] == rep["candidates"][0]["Ta_s"]
    # 每档的 CoP 秒数都贴着地板(时间律换挡救不动它)
    floor_sec = rep["irreducible"]["sec_cop"]
    for cd in rep["candidates"]:
        assert cd["sec_cop"] == pytest.approx(floor_sec, abs=0.06)


def test_reducible_vs_irreducible_split(tmp_path):
    """混合打分:τ 违规可约(加大 T_a 消掉),CoP 违规不可约(姿态绑定)。
    报告必须分开:irreducible.tau=0 / irreducible.cop>0;tight 选到
    τ 干净的最小档,其 reducible 超出 ≈ 0。"""
    data, phase = make_clip()
    rep, _ = search(tmp_path, data, phase, ComboScorer(cap=2.0, thresh=0.8),
                    mode="tight", sec_tol=0.06)
    assert rep["irreducible"]["sec_tau"] == 0.0
    assert rep["irreducible"]["sec_cop"] > 0.5
    assert rep["selection"]["chosen_Ta_s"] == pytest.approx(0.6, abs=1e-6)
    ex = rep["selection"]["reducible_excess_sec"]
    assert abs(ex["tau"]) <= 0.06 and abs(ex["cop"]) <= 0.06
    by_ta = {cd["Ta_s"]: cd for cd in rep["candidates"]}
    assert not by_ta[0.2]["acceptable"]                   # τ 可约部分还没消掉


# ------------------------------------------------------------- ④ fail-loud ------- #
def test_missing_phase_refused(tmp_path):
    data, _ = make_clip()
    with pytest.raises(SystemExit, match="相位"):
        search(tmp_path, data, None, AccBinaryScorer(2.0))


def test_budget_frac_out_of_range_refused(tmp_path):
    data, phase = make_clip()
    for bad in (0.0, -0.3, 1.5):
        with pytest.raises(SystemExit, match="budget_frac"):
            search(tmp_path, data, phase, AccBinaryScorer(2.0), budget_frac=bad)


def test_bad_mode_refused(tmp_path):
    data, phase = make_clip()
    with pytest.raises(SystemExit, match="mode"):
        search(tmp_path, data, phase, AccBinaryScorer(2.0), mode="banana")


def test_unknown_npz_key_refused(tmp_path):
    data, phase = make_clip()
    data["mystery_track"] = np.zeros((5, 3))
    with pytest.raises(SystemExit, match="key"):
        search(tmp_path, data, phase, AccBinaryScorer(2.0))


def test_contact_at_edge_refused(tmp_path):
    data, _ = make_clip()
    with pytest.raises(SystemExit):
        search(tmp_path, data, 1.0, AccBinaryScorer(2.0))   # 触球在末帧,没收拍路径


def test_scorer_missing_keys_refused(tmp_path):
    data, phase = make_clip()

    def half_scorer(path, phase_out):
        return dict(dose_tau=0.0)   # 缺 sec_* / cop / fric — 契约破裂

    with pytest.raises(SystemExit, match="打分器"):
        search(tmp_path, data, phase, half_scorer)


def test_no_feasible_within_t_avail_refused(tmp_path):
    """可行档最早也要 t* ≈ 1.1 s;预警时间只给 1.0 s → 拒跑并回灌接口守卫。"""
    data, phase = make_clip()
    with pytest.raises(SystemExit, match="不可行"):
        search(tmp_path, data, phase, AccBinaryScorer(cap=2.0), mode="tight",
               t_avail_s=1.0)


def test_cli_missing_phase_fails_before_heavy_work(tmp_path):
    """CLI 层 fail-loud:没给 --phase 也没给 --annotations → 在碰 mujoco/MJCF 之前拒跑。"""
    data, _ = make_clip()
    src = tmp_path / "clip.npz"
    np.savez(src, **data)
    with pytest.raises(SystemExit, match="相位"):
        tbs.main(["--input", str(src), "--output", str(tmp_path / "out.npz"),
                  "--budget-frac", "0.7", "--mode", "tight",
                  "--mjcf", "does_not_matter.xml", "--body-order", "nope.txt"])


def test_cli_annotation_without_stem_refused(tmp_path):
    """annotations 文件里查不到这个 clip 的 phase → 拒绝猜默认值。"""
    data, _ = make_clip()
    src = tmp_path / "clip.npz"
    np.savez(src, **data)
    ann = tmp_path / "ann.yaml"
    ann.write_text("clips:\n  some_other_clip:\n    phase: 0.5\n")
    with pytest.raises(SystemExit, match="phase"):
        tbs.main(["--input", str(src), "--output", str(tmp_path / "out.npz"),
                  "--annotations", str(ann),
                  "--budget-frac", "0.7", "--mode", "tight",
                  "--mjcf", "does_not_matter.xml", "--body-order", "nope.txt"])
