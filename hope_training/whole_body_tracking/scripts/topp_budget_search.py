#!/usr/bin/env python3
"""TOPP v2 (v1 实现) — 外环预算搜索:T_a 网格 × mj_inverse 可行性打分.

人话(这个工具在干什么)
    synthesize_timing.py 会自己选一个"运动学预算内最小"的加速时长 T_a,但它只看
    关节速度/加速度这种影子量。真约束(设计文档 §七/§九)是①逐关节力矩 |τ_req| ≤
    budget_frac × τ_max②接触可行性(立力≥0 / CoP∈支撑面 / 摩擦锥)——这些只有
    feasibility_oracle.py 的 mj_inverse 逆动力学才算得出来。所以本工具做外环搜索:

        对 T_a 网格上的每个候选:
            用 synthesize_timing 的时间律 + 重采样产一个候选 npz
            → 交给 feasibility_oracle 的机制逐帧打分(τ/CoP/摩擦三剂量)
        然后按模式选档:
            tight = 不可行剂量降到"温和端下限"的最小 T_a(贴限,时间最紧)
            fill  = 预算窗内(--t-avail-s)最大的可行 T_a(摊时,最温和)

    判决用剂量制不用峰值制(oracle 0709 校准结论):单帧尖峰不预测摔,预测摔的是
    不可行需求的时长占比。

timing-irreducible(时序不可约)剂量 —— 本报告最重要的一列
    把 T_a 放到最温和端(= ta_max,纯最小峰值加速度三角形)仍消不掉的违规剂量。
    这部分时间律救不了——它是姿态/路径绑定的(v5 家族的 CoP 违规预期落在这里)。
    报告里把每个候选的剂量拆成 irreducible(地板)+ reducible(候选超出地板的部分),
    这就是"时间律能不能救 v5"的直接答案:irreducible ≈ 0 ⇒ 能救;>0 ⇒ 只能救
    reducible 那部分,剩下的要动路径/姿态。

两轴消融(设计文档 §八)
    力矩余量档 budget_frac ∈ {0.5, 0.7, 0.9}(τ 预算 = budget_frac × τ_max,
    数值越小越保守)× 时间松弛档 {tight, fill}。本工具一次跑一个格子
    (一个 budget_frac + 一个 mode);扫两轴 = 多次调用。

可行性判据(选档用秒数不用占比,防"拖长时长稀释占比"假优化)
    候选可接受 ⟺ 每类违规的秒数 ≤ 温和端地板秒数 + --sec-tol。
    报告同时给占比(和 oracle 校准闸门可比)与秒数(选档口径)。

组合而非合并(尊重两个上游工具的 DELIBERATELY separate 约定)
    本文件只 import 调用 synthesize_timing.py(时间律/重采样)与仓库根
    scripts/feasibility_oracle.py(逆动力学打分),不改它们任何默认行为。

fail-loud
    缺登记相位、budget_frac 出 (0,1]、npz 有不认识的 key、打分器少返回字段、
    预警时间内无可行档 —— 全部拒跑并明说,绝不吞默认值。

退出码
    0 = 选中档不可行秒数为 0(时间律完全救得动)
    1 = 有 timing-irreducible 残余但低于 oracle FAIL 闸门(WARN 级)
    2 = 选中档剂量过 oracle FAIL 闸门(时间律救不动,得动路径/姿态)
    其余错误 = SystemExit 带人话消息。

USAGE(pod,mjeval venv:numpy + mujoco)
    /workspace/hope_mjeval_venv/bin/python \
        hope_training/whole_body_tracking/scripts/topp_budget_search.py \
        --input  .../hope_backhand_v4rg_cal.npz \
        --annotations hope_training/whole_body_tracking/cfg/strike_annotations.yaml \
        --output .../hope_backhand_v4rg_topp_b70_tight.npz \
        --budget-frac 0.7 --mode tight \
        --mjcf agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
        --body-order /workspace/franco/body_order_isaac.txt \
        --report out.json --md out.md

单元测试:tests/test_topp_budget_search.py(纯 CPU,合成 clip + 假打分器)。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# synthesize_timing 与本文件同目录,按它自己的路径约定 import(numpy-only at import)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_timing as syn  # noqa: E402


# ------------------------------------------------------------------ oracle 装载 -- #
def repo_root() -> Path:
    # scripts/ -> whole_body_tracking -> hope_training -> 仓库根
    return Path(__file__).resolve().parents[3]


def load_feasibility_oracle():
    """按文件路径装载仓库根 scripts/feasibility_oracle.py(与本目录不是同一个 scripts/).

    oracle 顶部 try-import mujoco,没装 mujoco 也能 import(打分时才真的需要),
    所以纯 CPU 测试可以拿它的 DOSE_* 闸门常数。找不到文件 = fail loud。
    """
    if "feasibility_oracle" in sys.modules:
        return sys.modules["feasibility_oracle"]
    p = repo_root() / "scripts" / "feasibility_oracle.py"
    if not p.is_file():
        raise SystemExit(f"feasibility_oracle.py 不在预期位置 {p} — 拒绝猜别的路径")
    spec = importlib.util.spec_from_file_location("feasibility_oracle", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["feasibility_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 打分器契约 --- #
# 打分器 = callable(npz_path, phase_out) -> dict,至少含以下键。
#   dose_* = 违规时长占比(和 oracle 校准闸门同口径)
#   sec_*  = 违规秒数(选档口径:占比会被"拖长时长"稀释,秒数不会)
REQUIRED_SCORE_KEYS = ("dose_tau", "dose_cop", "dose_fric", "sec_tau", "sec_cop", "sec_fric")
CHECKS = ("tau", "cop", "fric")
CHECK_LABEL = {"tau": "力矩(τ>budget_frac×τ_max)", "cop": "CoP 出支撑面", "fric": "摩擦锥"}


class OracleScorer:
    """生产打分器:feasibility_oracle 的 mj_inverse 机制逐帧算三剂量。

    直接复用 oracle 的 load_oracle_model + analyze_clip(不改其默认行为):
    CoP/摩擦剂量原样取 rep.doses(校准口径);τ 剂量按本工具的力矩余量档重算——
    违规帧 = 任一关节 |τ_req| > budget_frac × τ_max(rep.util 已是 |τ|/τ_max)。
    """

    def __init__(self, mjcf: str, body_order: str, mu: float, support_band: float,
                 budget_frac: float):
        self.fo = load_feasibility_oracle()
        self.om = self.fo.load_oracle_model(mjcf)  # 没装 mujoco 会 RuntimeError,fail loud
        self.body_order = body_order
        self.mu = mu
        self.support_band = support_band
        self.budget_frac = budget_frac

    def __call__(self, npz_path: Path, phase_out: float) -> dict:
        npz_path = Path(npz_path)
        stem = npz_path.name[: -len(".npz")]
        # 报告里的违规窗要相对触球帧,把候选自己的 phase_out 喂给 oracle 的登记口径
        rep = self.fo.analyze_clip(
            self.om, npz_path, {stem: {"phase": phase_out}},
            self.body_order, self.mu, self.support_band,
        )
        T, fps = rep.n_frames, rep.fps
        n_eval = max(T - 2, 1)              # 端点帧没有中心差分,oracle 同口径剔除
        util = rep.util[1 : T - 1]          # (n_eval, 31) |τ_req|/τ_max
        n_tau = int(((util > self.budget_frac).any(axis=1)).sum())
        dose_tau = n_tau / n_eval
        dose_cop = float(rep.doses["cop"])
        dose_fric = float(rep.doses["friction"])
        to_sec = n_eval / fps               # 占比 × 评估帧数 / fps = 秒数
        tq, cop, fric = (rep.checks[k] for k in ("torque", "cop", "friction"))
        return dict(
            dose_tau=dose_tau, dose_cop=dose_cop, dose_fric=dose_fric,
            sec_tau=dose_tau * to_sec, sec_cop=dose_cop * to_sec, sec_fric=dose_fric * to_sec,
            peak_tau_util=float(tq.peak), peak_tau_joint=str(tq.detail),
            cop_peak_m=float(max(cop.peak, 0.0)), fric_peak=float(fric.peak),
            oracle_verdict=str(rep.verdict), n_frames=int(T),
        )


# ------------------------------------------------------------------ 外环搜索 ----- #
def dose_gate_verdict(fo, dose_tau: float, dose_cop: float, dose_fric: float) -> str:
    """oracle 校准的剂量闸门(τ 剂量这里已按 budget_frac 口径算)。"""
    if (dose_cop >= fo.DOSE_COP_FAIL or dose_tau >= fo.DOSE_TAU_FAIL
            or dose_fric >= fo.DOSE_FRIC_FAIL):
        return "FAIL"
    if dose_cop >= fo.DOSE_COP_WARN or dose_fric > 0 or dose_tau >= fo.DOSE_TAU_WARN:
        return "WARN"
    return "PASS"


def _irreducible_block(floor: dict, sec_tol: float) -> dict:
    """时序不可约剂量块:最温和端(ta_max)候选的三剂量 = 时间律救不了的地板。"""
    blk = dict(Ta_s=floor["Ta_s"])
    for k in CHECKS:
        blk[f"dose_{k}"] = round(floor[f"dose_{k}"], 4)
        blk[f"sec_{k}"] = round(floor[f"sec_{k}"], 4)
    blk["any_residue"] = any(floor[f"sec_{k}"] > sec_tol for k in CHECKS)
    blk["note"] = ("时间放到最温和仍剩的违规=姿态/路径绑定,时间律救不了;"
                   "v5 家族的 CoP 预期落在这里")
    return blk


def run_search(data: dict, phase: float, budget_frac: float, mode: str, scorer,
               workdir: Path, stem: str, *,
               v_star: float | None = None, fps_out: float | None = None,
               min_cruise_s: float = 0.04, post_hold_s: float = 0.04,
               ta_grid_s: float = 0.05, t_avail_s: float | None = None,
               sec_tol: float = 0.025, body_mode: str = "interp", fk_ctx=None,
               keep_candidates: bool = False):
    """外环搜索主体。返回 (report dict, 选中档的输出 npz dict)。

    scorer 可注入(生产 = OracleScorer;测试 = 纯 CPU 假打分器),搜索逻辑本身
    不碰 mujoco——这就是单元测试的接缝。
    """
    # ---- fail-loud 入口检查 ------------------------------------------------------
    if mode not in ("tight", "fill"):
        raise SystemExit(f"mode 必须是 tight 或 fill,收到 {mode!r}")
    if not (0.0 < budget_frac <= 1.0):
        raise SystemExit(f"budget_frac 必须在 (0,1](力矩预算 = budget_frac×τ_max),收到 {budget_frac}")
    if phase is None:
        raise SystemExit("缺登记相位 phase — 拒跑(--phase 或 --annotations 必须给出一个)")
    unknown = [k for k in data.keys() if k not in syn.KNOWN_KEYS and not k.startswith("_")]
    if unknown:
        raise SystemExit(f"npz 有不认识的 key {unknown} — 不知道怎么重定时它们,拒跑")
    if ta_grid_s <= 0:
        raise SystemExit(f"--ta-grid-s 必须 > 0,收到 {ta_grid_s}")

    q = np.asarray(data["joint_pos"], dtype=np.float64)
    T_src, _J = q.shape
    fps_src = float(np.asarray(data["fps"]).reshape(-1)[0])
    if fps_out is None:
        fps_out = fps_src
    c = syn.contact_frame(phase, T_src)
    if not (0 < c < T_src - 1):
        raise SystemExit(f"触球帧 {c}/{T_src} 没给 run-up 或收拍留路径 — 拒跑")
    s_end = float(T_src - 1)

    # ---- 边界条件:触球帧拍速(默认复现源速;--strike-speed = 变速重解) ----------
    blade = syn.blade_positions(data)
    dpds = syn.blade_path_deriv_at(blade, c)
    if dpds <= 1e-9:
        raise SystemExit("触球帧拍路径导数 ~0 — phase 标错了?拒跑")
    v_src_clean = syn.clean_speed_at(blade, c, 1.0 / fps_src)
    if v_star is None:
        v_star = v_src_clean
    sdot_star = float(v_star) / dpds

    # ---- T_a 网格:从最紧到最温和端(ta_max = 触球前路径允许的最长加速) -----------
    tmax = syn.ta_max(c, sdot_star, min_cruise_s)  # run-up 装不下会自己 SystemExit
    grid = [float(t) for t in np.arange(ta_grid_s, tmax, ta_grid_s)]
    if not grid or grid[-1] < tmax - 1e-9:
        grid.append(float(tmax))                   # 最温和端必进网格 = 不可约剂量的测点

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cands: list[dict] = []
    for Ta in grid:
        law = syn.build_time_law(c, s_end, sdot_star, Ta, fps_out, post_hold_s)
        out, k_star, _ = syn.resample(data, law, fps_out, body_mode, fk_ctx)
        T_out = int(out["joint_pos"].shape[0])
        phase_out = k_star / (T_out - 1)
        p = workdir / f"{stem}__ta{Ta * 1000:04.0f}ms.npz"
        np.savez(p, **out)
        score = dict(scorer(p, phase_out))
        missing = [k for k in REQUIRED_SCORE_KEYS if k not in score]
        if missing:
            raise SystemExit(f"打分器少返回字段 {missing} — 契约破裂,拒跑")
        if not keep_candidates:
            p.unlink()
        cands.append(dict(
            Ta_s=round(Ta, 4), t_star_s=round(law.t_star, 4), t_end_s=round(law.t_end, 4),
            frames=T_out, contact_frame=int(k_star), phase_out=round(phase_out, 6),
            npz=(str(p) if keep_candidates else None), **score,
        ))

    # ---- 可行性:每类违规秒数不超"温和端地板 + tol"(见模块 docstring 的口径说明) --
    floor = cands[-1]  # ta_max = 最温和端 = timing-irreducible 剂量的定义点
    for cd in cands:
        cd["excess_sec"] = {k: round(cd[f"sec_{k}"] - floor[f"sec_{k}"], 4) for k in CHECKS}
        cd["acceptable"] = all(cd[f"sec_{k}"] <= floor[f"sec_{k}"] + sec_tol for k in CHECKS)
        cd["within_t_avail"] = bool(t_avail_s is None or cd["t_star_s"] <= t_avail_s + 1e-9)

    pool = [cd for cd in cands if cd["acceptable"] and cd["within_t_avail"]]
    if not pool:
        raise SystemExit(
            f"预警时间 t_avail={t_avail_s}s 内没有可行档(可行档最早触球 "
            f"{min((cd['t_star_s'] for cd in cands if cd['acceptable']), default=float('nan')):.2f}s)"
            " — 该题对该 ready 态不可行,回灌题库接口守卫")
    chosen = pool[0] if mode == "tight" else pool[-1]
    basis = ("tight=不可行秒数降到温和端地板的最小 T_a(时间最紧)" if mode == "tight"
             else "fill=预算窗内最大的可行 T_a(最温和,摊满时间)")

    # ---- 选中档重建输出 + 保真度检查(触球拍速/拍面不因重定时走样) ----------------
    law = syn.build_time_law(c, s_end, sdot_star, chosen["Ta_s"], fps_out, post_hold_s)
    out_final, k_star, _ = syn.resample(data, law, fps_out, body_mode, fk_ctx)
    blade_out = syn.blade_positions(out_final)
    v_out = syn.clean_speed_at(blade_out, k_star, 1.0 / fps_out)
    n_src = syn.blade_face_normals(data)[c]
    n_out = syn.blade_face_normals(out_final)[k_star]
    face_deg = float(np.degrees(np.arccos(np.clip(
        np.dot(n_src, n_out) / (np.linalg.norm(n_src) * np.linalg.norm(n_out)), -1, 1))))
    T_out = int(out_final["joint_pos"].shape[0])

    fo = load_feasibility_oracle()  # 只用 DOSE_* 闸门常数,CPU 可跑
    chosen_gate = dose_gate_verdict(fo, chosen["dose_tau"], chosen["dose_cop"], chosen["dose_fric"])

    report = dict(
        tool="topp_budget_search.py (TOPP v2 v1: 外环 T_a 搜索 × oracle 打分)",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        mode=mode, budget_frac=budget_frac, basis=basis,
        source=dict(frames=int(T_src), fps=fps_src, contact_frame=int(c),
                    phase=float(phase), clean_blade_speed_mps=round(v_src_clean, 4)),
        answer=dict(v_star_mps=round(float(v_star), 4),
                    v_star_source="source-clean" if abs(v_star - v_src_clean) < 1e-12 else "cli-override",
                    sdot_star_frames_per_s=round(sdot_star, 4)),
        grid=dict(ta_grid_s=ta_grid_s, ta_max_s=round(float(tmax), 4),
                  n_candidates=len(cands), t_avail_s=t_avail_s, sec_tol=sec_tol,
                  min_cruise_s=min_cruise_s, post_hold_s=post_hold_s),
        candidates=cands,
        # 时序不可约剂量 = 最温和端(ta_max)仍剩的违规:时间律救不了的部分
        irreducible=_irreducible_block(floor, sec_tol),
        selection=dict(chosen_Ta_s=chosen["Ta_s"], chosen_t_star_s=chosen["t_star_s"],
                       chosen_index=cands.index(chosen), n_acceptable=len(pool),
                       dose_gate_verdict=chosen_gate,
                       reducible_excess_sec=chosen["excess_sec"]),
        output=dict(frames=T_out, fps=float(fps_out), contact_frame=int(k_star),
                    phase_out=round(k_star / (T_out - 1), 6),
                    runup_s=round(k_star / fps_out, 4),
                    duration_s=round((T_out - 1) / fps_out, 4),
                    body_mode=body_mode,
                    blade_speed_clean_out_mps=round(v_out, 4),
                    blade_speed_dev_frac=round(abs(v_out - v_star) / v_star, 5),
                    face_normal_diff_deg=round(face_deg, 6)),
    )
    return report, out_final


# ------------------------------------------------------------------ 报告 --------- #
def report_md(rep: dict) -> str:
    s, a, g, sel, o, irr = (rep["source"], rep["answer"], rep["grid"],
                            rep["selection"], rep["output"], rep["irreducible"])
    lines = [
        f"# TOPP 预算搜索报告 — mode **{rep['mode']}** × budget_frac **{rep['budget_frac']}**",
        "",
        f"- 人话:在 {g['n_candidates']} 个加速时长候选(T_a ∈ [{g['ta_grid_s']}, "
        f"{g['ta_max_s']}] s)里,按「{rep['basis']}」选档;force 预算 = "
        f"{rep['budget_frac']}×τ_max,打分 = mj_inverse 三剂量(占比制,选档看秒数)。",
        f"- generated: {rep['generated_utc']}",
        f"- source: {s['frames']} 帧 @ {s['fps']:.0f} fps,触球 f{s['contact_frame']} "
        f"(phase {s['phase']}),clean 拍速 {s['clean_blade_speed_mps']:.3f} m/s",
        f"- 答案速度 |v*| = {a['v_star_mps']:.3f} m/s ({a['v_star_source']});"
        f"sdot* = {a['sdot_star_frames_per_s']:.2f} frames/s;速度绝不降,只调时间",
        f"- 选中:T_a = **{sel['chosen_Ta_s']:.3f} s**(触球时刻 t* = "
        f"{sel['chosen_t_star_s']:.3f} s;可行档 {sel['n_acceptable']}/{g['n_candidates']};"
        f"剂量闸门 {sel['dose_gate_verdict']})",
        f"- 输出:{o['frames']} 帧 @ {o['fps']:.0f} fps;触球 f{o['contact_frame']} → "
        f"**phase_out {o['phase_out']:.4f}**;拍速保真 {o['blade_speed_clean_out_mps']:.3f} m/s"
        f"(偏差 {o['blade_speed_dev_frac'] * 100:.2f}%);拍面差 {o['face_normal_diff_deg']:.4f}°;"
        f"body_mode={o['body_mode']}",
        "",
        f"## 时序不可约剂量(T_a→最温和端 {irr['Ta_s']:.3f} s 仍剩的违规)",
        "",
        f"- τ: 占比 {irr['dose_tau']:.3f} / {irr['sec_tau']:.2f} s | "
        f"CoP: 占比 {irr['dose_cop']:.3f} / {irr['sec_cop']:.2f} s | "
        f"摩擦: 占比 {irr['dose_fric']:.3f} / {irr['sec_fric']:.2f} s",
        f"- 人话:{irr['note']}。residue={'有' if irr['any_residue'] else '无'}。"
        f"选中档相对地板的 reducible 超出(秒): {sel['reducible_excess_sec']}",
        "",
        "## 候选表(τ 剂量按 budget_frac 口径;`*` = 选中档)",
        "",
        "| T_a [s] | t* [s] | τ 剂量 | τ 秒 | CoP 剂量 | CoP 秒 | 摩擦剂量 | 摩擦秒 | 可行 | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, cd in enumerate(rep["candidates"]):
        mark = " *" if i == sel["chosen_index"] else ""
        note = []
        if not cd["within_t_avail"]:
            note.append("超预警时间")
        if cd.get("oracle_verdict"):
            note.append(cd["oracle_verdict"])
        lines.append(
            f"| {cd['Ta_s']:.3f}{mark} | {cd['t_star_s']:.2f} | {cd['dose_tau']:.3f} | "
            f"{cd['sec_tau']:.2f} | {cd['dose_cop']:.3f} | {cd['sec_cop']:.2f} | "
            f"{cd['dose_fric']:.3f} | {cd['sec_fric']:.2f} | "
            f"{'✓' if cd['acceptable'] else '✗'} | {'/'.join(note) or '-'} |"
        )
    lines += [
        "",
        f"REGISTRY REMINDER: 选中资产时间轴是重合成的 — 到 cfg/strike_annotations.yaml "
        f"登记 phase_out = {o['phase_out']:.4f}(触球帧 {o['contact_frame']}/{o['frames']});"
        f"源视频帧约定作废。",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI ---------- #
def resolve_phase(args, stem: str) -> float:
    """登记相位:--phase 优先;否则查 --annotations;都没有 = 拒跑(fail loud)。"""
    if args.phase is not None:
        return float(args.phase)
    if args.annotations:
        fo = load_feasibility_oracle()
        ann = fo.load_annotations(args.annotations)
        entry = ann.get(stem)
        if not entry or "phase" not in entry:
            raise SystemExit(
                f"annotations {args.annotations} 里没有 {stem!r} 的 phase — 拒绝猜默认值")
        try:
            return float(entry["phase"])
        except (TypeError, ValueError):
            raise SystemExit(f"annotations 里 {stem!r} 的 phase={entry['phase']!r} 解析不了 — 拒跑")
    raise SystemExit("缺登记相位:--phase 或 --annotations 必须给出一个 — 拒跑")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="源 motion npz(路径先验)")
    ap.add_argument("--output", required=True, help="选中档写到这里")
    ap.add_argument("--phase", type=float, default=None,
                    help="源 clip 登记相位(不给就查 --annotations;都没有拒跑)")
    ap.add_argument("--annotations", default=None,
                    help="strike_annotations.yaml(clips.<stem>.phase 口径)")
    ap.add_argument("--budget-frac", type=float, required=True,
                    help="力矩余量档:τ 预算 = budget_frac×τ_max,消融档 {0.5,0.7,0.9}")
    ap.add_argument("--mode", choices=("tight", "fill"), required=True,
                    help="tight=最小可行 T_a(贴限);fill=预算窗内最大可行 T_a(摊时)")
    ap.add_argument("--ta-grid-s", type=float, default=0.05,
                    help="T_a 网格步长 [s](外环每格都要过一次 mj_inverse,别太细)")
    ap.add_argument("--t-avail-s", type=float, default=None,
                    help="预警时间:触球时刻 t* 必须 ≤ 此值;fill 在此窗内摊满")
    ap.add_argument("--strike-speed", type=float, default=None,
                    help="答案拍速 |v*| m/s(默认复现源 clean 拍速;给了 = 变速重解)")
    ap.add_argument("--fps-out", type=float, default=None, help="默认同源 fps")
    ap.add_argument("--min-cruise-s", type=float, default=0.04,
                    help="触球前最短匀速段(触球窗保护,同 synthesize_timing)")
    ap.add_argument("--post-contact-hold-s", type=float, default=0.04)
    ap.add_argument("--sec-tol", type=float, default=0.025,
                    help="可行判据容差 [s](吸收重采样 ±1 帧抖动)")
    ap.add_argument("--body-mode", choices=("fk", "interp"), default="fk",
                    help="fk=MuJoCo FK 重建(生产);interp=lerp/slerp(仅测试)")
    ap.add_argument("--mjcf", required=True,
                    help="a3_pingpong.xml:oracle 逆动力学 + fk 重建共用")
    ap.add_argument("--body-order", required=True,
                    help="body_pos_w 列序文件(oracle 支撑面/根位姿 + fk 列映射都要)")
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--support-band", type=float, default=0.03)
    ap.add_argument("--workdir", default=None,
                    help="候选 npz 落盘目录(默认建临时目录,跑完删)")
    ap.add_argument("--keep-candidates", action="store_true",
                    help="保留每个候选的 npz(默认打分完即删)")
    ap.add_argument("--report", default=None, help="JSON 报告写这里")
    ap.add_argument("--md", default=None, help="markdown 报告写这里")
    args = ap.parse_args(argv)

    if not (0.0 < args.budget_frac <= 1.0):
        raise SystemExit(f"--budget-frac 必须在 (0,1],收到 {args.budget_frac}")
    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"输入不存在:{in_path}")
    stem = in_path.name[: -len(".npz")] if in_path.name.endswith(".npz") else in_path.name
    phase = resolve_phase(args, stem)

    data = dict(np.load(in_path))

    fk_ctx = None
    if args.body_mode == "fk":
        fkm = syn.ctn.MjFK(args.mjcf, syn.ISAAC_JOINT_NAMES)
        names = fkm.body_names()
        order = [ln.strip() for ln in open(args.body_order) if ln.strip()]
        fk_ctx = (fkm, [names.index(n) for n in order])

    scorer = OracleScorer(args.mjcf, args.body_order, args.mu, args.support_band,
                          args.budget_frac)

    tmp_created = args.workdir is None
    workdir = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix=f"topp_{stem}_", dir=str(Path(args.output).parent or ".")))
    try:
        report, out_final = run_search(
            data, phase, args.budget_frac, args.mode, scorer, workdir, stem,
            v_star=args.strike_speed, fps_out=args.fps_out,
            min_cruise_s=args.min_cruise_s, post_hold_s=args.post_contact_hold_s,
            ta_grid_s=args.ta_grid_s, t_avail_s=args.t_avail_s, sec_tol=args.sec_tol,
            body_mode=args.body_mode, fk_ctx=fk_ctx,
            keep_candidates=args.keep_candidates)
    finally:
        if tmp_created and not args.keep_candidates:
            shutil.rmtree(workdir, ignore_errors=True)

    report["files"] = dict(input=str(in_path.resolve()), output=os.path.abspath(args.output),
                           mjcf=os.path.abspath(args.mjcf),
                           body_order=os.path.abspath(args.body_order))
    np.savez(args.output, **out_final)
    md = report_md(report)
    print(md)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.md:
        Path(args.md).write_text(md)

    # 退出码:0=时间律全救得动;1=有不可约残余(WARN 级);2=选中档仍过 FAIL 闸门
    if report["selection"]["dose_gate_verdict"] == "FAIL":
        return 2
    if report["irreducible"]["any_residue"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
