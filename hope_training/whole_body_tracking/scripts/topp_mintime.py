#!/usr/bin/env python3
"""MIN-TIME 双向重定时 (TOPP v3) — 统一可行性预算下,每个动作解自己的最快时间.

人话:v2(synthesize_timing_v2.py)只会"放慢"——oracle 判不可行的帧局部拉长,治病用。
v3 把 TOPP 的语义摆正(franco 2026-07-09 拍板):TOPP = "找到能做完这个动作的最快时间"。
统一的超参数 = 可行性预算,就两块:
    ① oracle 剂量闸门(feasibility_oracle 的 CoP / 摩擦锥 / τ 剂量,默认 = v2 校准值);
    ② 运动学硬边界(URDF 速度限位 × 余量 0.85 + |acc| 实证包络)——压缩必然顶到它。
每个 clip 在同一预算下解自己的 min-time:
    - 病动作(v5 反手):想快也快不了,预算顶回去 → 被放慢(退化为 v2 行为);
    - 健康动作(v4rg / swing):预算内有富余 → 被**加速**(总时长压到最小)。
franco 附带假说:弱侧学不动可能因为正反手时长/节奏差太多;统一 min-time 重定时把两侧
都推到同一预算的边界上,难度被拉平 → 报告把总时长/助跑时长放显眼处,供两侧对账。

ALGORITHM(外层缩放扫描 + 内层 oracle 在环修复)
    baseline:复用 v1.solve_min_ta(匀加速单参族 + 运动学验收)——迭代 0 的时间律。
    全局缩放 γ:触球锁窗(登记 phase ±0.1s)外,把基线时间律的 ρ 场整体置成 γ
        (γ<1 = 加速,γ>1 = 放慢);锁窗内永远 ρ≡1 → 触球行逐位保真、拍速 |v*| 不动
        (v1/v2 同约束,warp 的 grid-snap + 逐位 pin 原样复用)。
    内层修复(复用 v2 的贪心 bump):对候选跑 oracle 逐帧判卷 + 逐帧运动学利用率
        (audit 口径:|q̇| / (URDF×frac),|Δq̇|·fps / acc 包络);窗外超预算帧在其路径
        邻域抬 ρ 压回 —— 速度按线性律(|q̇|∝1/ρ)、加速度/τ/摩擦按开方律(∝1/ρ²)、
        CoP 按固定收缩迭代。验收 = oracle 三剂量全过闸 且 窗外运动学零越界(硬边界)。
        修不动(剩余越界全被锁窗钉死 / 时长超守卫 / 迭代耗尽)= 该 γ 不可行。
    外层:γ 从 1 起乘法下探(×compress-step),把整条梯子扫到 γ 下界(梯子越过下界时
        补探下界本身),全程不早停——修复后时长对 γ 非单调(bump 有过冲),平台期早停
        会错过更短的可行解(对抗复核 2026-07-09 实锤:fh_v5hLs 1.66→1.50s、fh_v4rg
        2.44→2.38s);修不动(不可行)才在最后一对(好 γ, 坏 γ)之间几何二分收尾。
        γ=1 都修不进预算 → 往上乘(×expand-step)找第一个可行的 γ 再二分收尾,梯子
        越过 γ 上界时补探上界本身;还没有 = fail loud(SystemExit)。
    收敛 = 预算内的最短总时长;全程记录候选,取全局最优(不信任单调性,见取舍)。
    语义务必读对:输出的 min-time 是**本搜索族(乘法梯子 × 内层贪心修复)内的最短**,
        是真 min-time 的上界——梯子点之间可能存在更短的可行解(非单调性所致),
        要更紧的界加密 --compress-step / --refine-steps 换 oracle 调用量。

取舍(为什么这么做,不是严格 TOPP / 凸优化):
    - 真逐点 TOPP 要求约束对 ṡ² 仿射闭式;CoP/摩擦锥经 mj_inverse 是黑盒,只能在环判卷
      → 贪心 bump + 外层缩放是 v2 已验证收敛的套路,直接复用,不另起炉灶。
    - 修复后的总时长对 γ 不保证单调(bump 有过冲),所以外层不做纯二分,而是乘法扫描
      + 收尾二分 + 全局记最优;代价 = 多几次 oracle 调用,换稳健。
    - 为什么外层是"全局 γ"而不是逐帧压缩优化:统一预算的语义就是一个旋钮;非均匀性
      全部由内层 bump 场提供(哪里有富余哪里就留在 γ,哪里超预算哪里被抬回)。
    - mean|acc| ≈ K/(T_out−1):压缩天然抬加速度剂量,acc 包络 + oracle 闸门就是压缩的
      两道地板;剂量是占比(flagged/T),压缩缩分母也会抬剂量 → 预算收紧 ⇒ min-time
      时长单调不减(测试⑤盯这个性质)。

NEW PHASE 口径(同 v1/v2):触球行 = 源帧 c 在输出网格 k* 的逐位拷贝;
    phase_out = k*/(T_out−1);视频约定帧对本资产弃用 → 登记 phase_out。

USAGE (pod, hope_mjeval_venv: numpy + mujoco)
    <venv>/bin/python hope_training/whole_body_tracking/scripts/topp_mintime.py \
        --input  .../v5_height_fix/hope_backhand_v5hLs_cal.npz --phase 0.391 \
        --output .../v5_height_fix/hope_backhand_v5mt_cal.npz \
        --budget-clips .../regen_0708_candidates/hope_backhand_v4rg_cal.npz \
                       .../regen_0708_candidates/hope_forehand_v4rg_cal.npz \
        --mjcf .../a3_pingpong.xml --body-order /workspace/franco/body_order_isaac.txt \
        --report bh_mt.json --md bh_mt.md

DEPENDENCIES: numpy always; mujoco 只在 --body-mode fk + 真 oracle 时(延迟导入)。
    纯数学核心(缩放场 / 修复环 / 外层搜索)CPU 单测:tests/test_topp_mintime.py
    (桩 oracle,v2 测试同手法)。勿改 v1/v2 的默认行为——本文件只 import 复用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

# 复用 v1(基线时间律/路径提取/拍面 FK)与 v2(ρ 场/warp/重采样/oracle 接口/bump 常数)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import synthesize_timing as v1        # noqa: E402  (numpy-only at import time)
import synthesize_timing_v2 as v2     # noqa: E402  (numpy-only at import time)

ISAAC_JOINT_NAMES = v1.ISAAC_JOINT_NAMES

# ---- 统一预算默认值:全部沿用 v2 的 oracle 校准口径(--dose-target 可收紧) ---------- #
DEFAULT_COP_GATE = v2.DOSE_TARGET     # 0.10  CoP 剂量闸门(主旋钮;v2 的 loop stop 目标)
DEFAULT_FRIC_GATE = v2.FRIC_ACCEPT    # 0.05  摩擦剂量闸门(= oracle DOSE_FRIC_FAIL)
DEFAULT_TAU_GATE = v2.TAU_ACCEPT      # 0.02  τ 剂量闸门(= oracle DOSE_TAU_WARN)
KIN_VEL_TARGET = 0.95                 # 修复时把越界帧的速度利用率压回到 0.95(留 5% 余)
KIN_ACC_TARGET = 0.95                 # 同上,加速度利用率目标
# ---- 外层搜索默认值 ------------------------------------------------------------------ #
DEFAULT_COMPRESS_STEP = 0.8           # 下探步长:γ ← γ×0.8(每步 -20% 时长上限)
DEFAULT_EXPAND_STEP = 1.3             # 上探步长:γ ← γ×1.3(γ=1 修不进预算时)
DEFAULT_SCALE_MIN = 0.35              # γ 下界(再压 = 时长 <35%,没有实证意义,防失控)
DEFAULT_SCALE_MAX = 4.0               # γ 上界(放慢 4 倍还不可行 = 几何问题,不是时序)
DEFAULT_REFINE_STEPS = 2              # 好/坏 γ 之间几何二分收尾步数
DEFAULT_MAX_INNER = 15                # 每个 γ 的内层修复迭代上限


# ====================================================================================== #
# ρ(s) 缩放场:v2 StretchField 放开 ρ≥1 限制 —— 窗外基线 = γ,bump 只往上抬              #
# ====================================================================================== #
@dataclass
class ScaledStretchField(v2.StretchField):
    """锁窗外基线 ρ=γ(γ<1 = 加速),锁窗内永远 ρ≡1(触球拍速不动)。
    bump 与 v2 完全同套路(高斯邻域 × 锁窗掩码 × 单轮增幅封顶),只把地板从 1 改成
    γ 基线:修复只能"往回抬",永远不会比 γ 更快 —— 加速的量由外层 γ 给,内层只负责治。"""
    rho_base: Optional[np.ndarray] = None   # (M,) γ 基线(锁窗内 1,窗外 γ,taper 平滑)

    @classmethod
    def build_scaled(cls, s_end: float, c: float, half: float, taper: float,
                     gamma: float, ds: float = 0.1) -> "ScaledStretchField":
        base = v2.StretchField.build(s_end, c, half, taper, ds)
        w = base.lock_weight(base.s_grid)              # 0 = 锁窗内, 1 = 窗外
        rho0 = 1.0 + (float(gamma) - 1.0) * w          # 窗内 1,窗外 γ,边缘平滑过渡
        return cls(s_grid=base.s_grid, rho=rho0.copy(), c=base.c, half=base.half,
                   taper=base.taper, rho_base=rho0.copy())

    def bump(self, centers: np.ndarray, factors: np.ndarray) -> None:
        if len(centers) == 0:
            return
        lw = self.lock_weight(self.s_grid)
        add = np.zeros_like(self.rho)
        for sc, fac in zip(np.asarray(centers, float), np.asarray(factors, float)):
            fac = float(min(fac, v2.RHO_STEP_MAX))     # 单轮增幅封顶(v2 同值)
            if fac <= 1.0:
                continue
            g = np.exp(-0.5 * ((self.s_grid - sc) / v2.BUMP_SIGMA_FR) ** 2)
            add = np.maximum(add, (fac - 1.0) * g * lw)
        self.rho = np.minimum(self.rho * (1.0 + add), v2.RHO_MAX)
        self.rho = np.maximum(self.rho, self.rho_base)  # 地板 = γ 基线(不是 1)

    def snapshot(self) -> "ScaledStretchField":
        return ScaledStretchField(self.s_grid.copy(), self.rho.copy(), self.c,
                                  self.half, self.taper, rho_base=self.rho_base.copy())


# ====================================================================================== #
# 逐帧运动学利用率(压缩的硬边界;audit 口径)                                            #
# ====================================================================================== #
def kin_utils(out: dict, vel_cap: np.ndarray, acc_budget: np.ndarray, fps_out: float):
    """输出网格上的逐帧运动学利用率,>1 = 越界。
    人话:速度 = 每个关节 |q̇| 除以 (URDF 限位×余量);加速度 = |Δq̇|·fps 除以实证包络
    (与 acc_envelope 的测量口径一致:np.diff(joint_vel)*fps)。帧 k 的加速度记 k→k+1。"""
    jv = np.asarray(out["joint_vel"], dtype=np.float64)
    vel_util = (np.abs(jv) / vel_cap[None, :]).max(axis=1)
    acc = np.abs(np.diff(jv, axis=0)) * fps_out
    acc_u = (acc / acc_budget[None, :]).max(axis=1)
    acc_util = np.concatenate([acc_u, [0.0]])
    return vel_util, acc_util


def _kin_bumps(vel_util: np.ndarray, acc_util: np.ndarray, s_out: np.ndarray, field):
    """运动学越界帧 → (路径中心, ρ 抬升倍率)。
    速度 ∝ 1/ρ → 线性律 fac = util/0.95;加速度主项 ∝ 1/ρ² → 开方律 fac = √(util/0.95)。
    锁窗内的越界由触球拍速 |v*| 几何钉死,时间律治不了 → 只计数(irreducible),不出 bump。"""
    lw = field.lock_weight(s_out)
    inside = lw < 0.5
    vel_bad = vel_util > 1.0
    acc_bad = acc_util > 1.0
    any_bad = vel_bad | acc_bad
    centers, factors = [], []
    for k in range(len(s_out)):
        if not any_bad[k] or inside[k]:
            continue
        fac = 1.0
        if vel_bad[k]:
            fac = max(fac, float(vel_util[k]) / KIN_VEL_TARGET)
        if acc_bad[k]:
            fac = max(fac, float(np.sqrt(acc_util[k] / KIN_ACC_TARGET)))
        centers.append(float(s_out[k]))
        factors.append(float(fac))
    breakdown = dict(
        kin_vel_bad=int(vel_bad.sum()), kin_acc_bad=int(acc_bad.sum()),
        kin_bad_in_window=int((any_bad & inside).sum()),
        kin_bad_out_window=int((any_bad & ~inside).sum()),
    )
    return np.asarray(centers, dtype=float), np.asarray(factors, dtype=float), breakdown


# ====================================================================================== #
# 内层:给定 γ 修复到预算内(可行)或判死(不可行)                                       #
# ====================================================================================== #
@dataclass
class InnerResult:
    feasible: bool
    reason: str                       # pass / dose_gate / window_locked / duration_guard / max_inner_iters
    gamma: float
    out: Optional[dict] = None
    warp: Optional["v2.WarpResult"] = None
    reading: Optional["v2.OracleReading"] = None
    reading0: Optional["v2.OracleReading"] = None   # 该 γ 未修复(迭代 0)的判卷
    field: Optional[ScaledStretchField] = None
    vel_util: Optional[np.ndarray] = None
    acc_util: Optional[np.ndarray] = None
    iters: int = 0
    duration_s: float = float("inf")
    breakdown: dict = dc_field(default_factory=dict)
    trace: List[dict] = dc_field(default_factory=list)


def repair_at_scale(gamma: float, data: dict, law: "v1.TimeLaw", judge: "v2.Judge",
                    stem: str, vel_cap: np.ndarray, acc_budget: np.ndarray,
                    fps_out: float, body_mode: str, fk_ctx, cop_gate: float,
                    fric_gate: float, tau_gate: float, max_inner: int,
                    src_duration: float, half: float, c: int) -> InnerResult:
    """内层修复环:γ 缩放候选 → oracle 判卷 + 运动学利用率 → 窗外超预算帧局部抬 ρ →
    重判,直到验收(= 该 γ 可行)或修不动(= 该 γ 不可行)。fail-quiet 是禁的:
    每一轮都进 trace,不可行原因写明白。"""
    field = ScaledStretchField.build_scaled(law.s_end, float(c), half,
                                            v2.LOCK_TAPER_FR, gamma)
    reading0 = None
    trace: List[dict] = []
    out = warp = reading = None
    vel_util = acc_util = None
    breakdown: dict = {}
    feasible, reason = False, "max_inner_iters"

    for it in range(max_inner + 1):
        warp = v2.warp_timeline(law, field, fps_out)
        out = v2.resample_at_s(data, warp.s_out, fps_out, body_mode, fk_ctx)
        phase_out = warp.k_star / (warp.T_out - 1)
        reading = judge(out, stem, phase_out)
        if reading0 is None:
            reading0 = reading
        vel_util, acc_util = kin_utils(out, vel_cap, acc_budget, fps_out)

        # oracle 侧越界帧(复用 v2 的映射:窗外才出 bump,窗内计 irreducible)
        oc, ofac, obr = v2._flagged_bumps(reading, warp.s_out, field)
        # 运动学侧越界帧(v3 新增:压缩的硬边界)
        kc, kfac, kbr = _kin_bumps(vel_util, acc_util, warp.s_out, field)
        breakdown = {**obr, **kbr}
        trace.append(dict(iter=it, verdict=reading.verdict,
                          cop=round(reading.doses.get("cop", 0.0), 4),
                          fric=round(reading.doses.get("friction", 0.0), 4),
                          tau=round(reading.doses.get("torque", 0.0), 4),
                          T_out=warp.T_out, duration_s=round(warp.duration_s, 4),
                          **breakdown))

        # 验收:oracle 三剂量全过闸(或 verdict PASS,v2 同口径)且窗外运动学零越界
        dose_ok = (reading.verdict == "PASS"
                   or (reading.doses.get("cop", 1.0) <= cop_gate
                       and reading.doses.get("friction", 0.0) <= fric_gate
                       and reading.doses.get("torque", 0.0) <= tau_gate))
        kin_ok = kbr["kin_bad_out_window"] == 0
        if dose_ok and kin_ok:
            feasible = True
            reason = "pass" if reading.verdict == "PASS" else "dose_gate"
            break

        centers = np.concatenate([oc, kc]) if (len(oc) + len(kc)) else np.array([])
        factors = np.concatenate([ofac, kfac]) if (len(ofac) + len(kfac)) else np.array([])
        if len(centers) == 0:
            reason = "window_locked"     # 剩余越界全被触球锁窗钉死 → 几何问题非时序
            break
        if warp.duration_s > v2.DURATION_GUARD_X * src_duration:
            reason = "duration_guard"    # 拉长超守卫倍数还不达标 → 判不可行防失控
            break
        field.bump(centers, factors)

    return InnerResult(feasible=feasible, reason=reason, gamma=float(gamma), out=out,
                       warp=warp, reading=reading, reading0=reading0,
                       field=field.snapshot(), vel_util=vel_util, acc_util=acc_util,
                       iters=len(trace) - 1, duration_s=float(warp.duration_s),
                       breakdown=breakdown, trace=trace)


# ====================================================================================== #
# 外层:γ 扫描,收敛 = 预算内的最短总时长                                                 #
# ====================================================================================== #
@dataclass
class MinTimeResult:
    best: InnerResult                 # 全局最优(预算内最短时长)
    gamma1: InnerResult               # γ=1 的修复结果(v2 语义等价,对照用)
    reading0: "v2.OracleReading"      # γ=1 迭代 0 = v1 基线原样的判卷(before)
    outer_trace: List[dict] = dc_field(default_factory=list)


def _outer_row(r: InnerResult) -> dict:
    return dict(gamma=round(r.gamma, 4), feasible=bool(r.feasible), reason=r.reason,
                iters=r.iters, T_out=(int(r.warp.T_out) if r.warp else None),
                duration_s=(round(r.duration_s, 4) if np.isfinite(r.duration_s) else None),
                cop=(round(r.reading.doses.get("cop", 0.0), 4) if r.reading else None),
                fric=(round(r.reading.doses.get("friction", 0.0), 4) if r.reading else None),
                tau=(round(r.reading.doses.get("torque", 0.0), 4) if r.reading else None))


def mintime_search(data: dict, law: "v1.TimeLaw", judge: "v2.Judge", stem: str,
                   vel_cap: np.ndarray, acc_budget: np.ndarray, fps_out: float,
                   body_mode: str, fk_ctx, cop_gate: float, fric_gate: float,
                   tau_gate: float, max_inner: int, src_duration: float, half: float,
                   c: int, compress_step: float, expand_step: float, scale_min: float,
                   scale_max: float, refine_steps: int) -> MinTimeResult:
    """外层 γ 搜索。可行 = 内层修进预算;目标 = 预算内最短总时长(全局记最优)。
    压缩侧把乘法梯子扫到 γ 下界(含下界收尾点)不早停:修复后时长对 γ 非单调,
    平台期早停会错过更短可行解(0709 对抗复核实锤);上探侧梯子越界时补探上界本身。"""
    outer_trace: List[dict] = []

    def inner(g: float) -> InnerResult:
        r = repair_at_scale(g, data, law, judge, stem, vel_cap, acc_budget, fps_out,
                            body_mode, fk_ctx, cop_gate, fric_gate, tau_gate,
                            max_inner, src_duration, half, c)
        outer_trace.append(_outer_row(r))
        return r

    def refine(lo: float, hi: float, best: InnerResult) -> InnerResult:
        # 好(hi)/坏(lo)γ 之间的几何二分收尾:再挤 refine_steps 步
        for _ in range(refine_steps):
            mid = float(np.sqrt(lo * hi))
            rm = inner(mid)
            if rm.feasible and rm.duration_s < best.duration_s - 1e-9:
                best, hi = rm, mid
            else:
                lo = mid
        return best

    r1 = inner(1.0)
    best: Optional[InnerResult] = r1 if r1.feasible else None

    if r1.feasible:
        # 下探压缩:健康动作在这里被加速。整条梯子扫到 γ 下界(越过时补探下界本身),
        # 全程记最优,不做平台期早停——修复后时长对 γ 非单调,早停会错过更短可行解
        # (对抗复核 0709:fh_v5hLs 平台期弃 1.50s、fh_v4rg 守卫尾巴藏 2.38s)。
        good_g = 1.0
        g = compress_step
        at_floor = False
        while True:
            if g < scale_min * (1.0 - 1e-9):
                if at_floor or good_g <= scale_min * (1.0 + 1e-9):
                    break
                g, at_floor = scale_min, True        # 梯子越过下界:补探 γ=下界 收尾
            r = inner(g)
            if r.feasible:
                if r.duration_s < best.duration_s - 1e-9:
                    best = r                          # 全局最优(时长帧格量化,严格短=至少短 1 帧)
                good_g = g
                if at_floor:
                    break
                g *= compress_step
                continue
            best = refine(g, good_g, best)           # 修不动:好坏之间二分收尾
            break
    else:
        # γ=1 都修不进预算(病重):往上乘找第一个可行的 γ,再往回二分挤;
        # 梯子越过 γ 上界时补探上界本身(fail-loud 文案里的 --scale-max 才名副其实)
        prev = 1.0
        g = expand_step
        while True:
            if g > scale_max * (1.0 + 1e-9):
                if prev >= scale_max * (1.0 - 1e-9):
                    break
                g = scale_max
            r = inner(g)
            if r.feasible:
                best = refine(prev, g, r)
                break
            prev = g
            g *= expand_step
        if best is None:
            bd = r1.breakdown
            raise SystemExit(
                "min-time 搜索失败:γ=1 修不进预算,放大到 γ={:.2f}(--scale-max)仍不可行。\n"
                "γ=1 末轮:reason={},CoP {:.4f}/fric {:.4f}/τ {:.4f}(闸门 {}/{}/{});"
                "窗内不可约:oracle {} 帧 / 运动学 {} 帧。\n"
                "窗内不可约 = 触球锁窗几何本身超预算,时间律治不了 → 走 path morph"
                "(改 |v*| 或空间路径),同 v2 口径;不硬造结果。".format(
                    scale_max, r1.reason,
                    r1.reading.doses.get("cop", float("nan")),
                    r1.reading.doses.get("friction", float("nan")),
                    r1.reading.doses.get("torque", float("nan")),
                    cop_gate, fric_gate, tau_gate,
                    bd.get("bad_in_window", "?"), bd.get("kin_bad_in_window", "?")))

    return MinTimeResult(best=best, gamma1=r1, reading0=r1.reading0,
                         outer_trace=outer_trace)


# ====================================================================================== #
# 顶层入口(与 v2.topp_lite 同形的 API,方便测试/复用)                                    #
# ====================================================================================== #
def mintime(data: dict, phase: float, vlim: np.ndarray, acc_budget: np.ndarray,
            judge: "v2.Judge", stem: str, v_star: Optional[float] = None,
            vel_limit_frac: float = 0.85, fps_out: Optional[float] = None,
            min_cruise_s: float = 0.04, post_hold_s: float = 0.04,
            ta_grid_s: float = 0.005, dense_dt_s: float = 0.002,
            body_mode: str = "interp", fk_ctx=None,
            max_inner: int = DEFAULT_MAX_INNER,
            cop_gate: float = DEFAULT_COP_GATE, fric_gate: float = DEFAULT_FRIC_GATE,
            tau_gate: float = DEFAULT_TAU_GATE,
            compress_step: float = DEFAULT_COMPRESS_STEP,
            expand_step: float = DEFAULT_EXPAND_STEP,
            scale_min: float = DEFAULT_SCALE_MIN, scale_max: float = DEFAULT_SCALE_MAX,
            refine_steps: int = DEFAULT_REFINE_STEPS):
    """统一预算 min-time 重定时。返回 (out, MinTimeResult, law, meta)。"""
    # ---- 参数护栏(fail loud,不静默修正) --------------------------------------- #
    if not (0.0 < compress_step < 1.0):
        raise SystemExit(f"--compress-step 必须在 (0,1),拿到 {compress_step}")
    if expand_step <= 1.0:
        raise SystemExit(f"--expand-step 必须 >1,拿到 {expand_step}")
    if not (0.0 < scale_min <= 1.0 <= scale_max):
        raise SystemExit(f"要求 0 < scale_min ≤ 1 ≤ scale_max,拿到 {scale_min}/{scale_max}")
    if min(cop_gate, fric_gate, tau_gate) < 0:
        raise SystemExit("剂量闸门不能为负")

    # ---- 与 v1/v2 完全同口径的预处理(路径/触球帧/拍速边界条件/预算) -------------- #
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    unknown = [k for k in data.keys() if k not in v1.KNOWN_KEYS and not k.startswith("_")]
    if unknown:
        raise SystemExit(f"unknown npz keys {unknown} — refusing to guess how to retime them")
    T_src, J = q.shape
    fps_src = float(np.asarray(data["fps"]).reshape(-1)[0])
    if fps_out is None:
        fps_out = fps_src
    c = v1.contact_frame(phase, T_src)
    s_end = float(T_src - 1)
    if not (0 < c < T_src - 1):
        raise SystemExit(f"contact frame {c} of {T_src} leaves no run-up or follow-through")

    blade = v1.blade_positions(data)
    dpds = v1.blade_path_deriv_at(blade, c)
    v_src_clean = v1.clean_speed_at(blade, c, 1.0 / fps_src)
    if v_star is None:
        v_star = v_src_clean
    if dpds <= 1e-9:
        raise SystemExit("blade path derivative ~0 at the contact frame — bad phase?")
    sdot_star = float(v_star) / dpds

    vel_cap = np.asarray(vlim, dtype=np.float64) * vel_limit_frac
    acc_budget = np.asarray(acc_budget, dtype=np.float64)
    if (acc_budget <= 0).any():
        floor = acc_budget[acc_budget > 0].min() if (acc_budget > 0).any() else 1.0
        acc_budget = np.where(acc_budget <= 0, floor, acc_budget)

    qp = np.gradient(q, axis=0)
    qpp = np.gradient(qp, axis=0)

    # 迭代 0 的时间律 = v1 匀加速单参族的 min-T_a(运动学验收在里面)
    law, base_info = v1.solve_min_ta(c, s_end, sdot_star, qp, qpp, vel_cap, acc_budget,
                                     fps_out, min_cruise_s, post_hold_s, ta_grid_s,
                                     dense_dt_s)

    half = v2.STRIKE_HALF_S * fps_src        # 锁窗半宽(路径帧;基线过触球时恰 ±0.1s)
    src_duration = (T_src - 1) / fps_src

    res = mintime_search(data, law, judge, stem, vel_cap, acc_budget, fps_out,
                         body_mode, fk_ctx, cop_gate, fric_gate, tau_gate, max_inner,
                         src_duration, half, c, compress_step, expand_step,
                         scale_min, scale_max, refine_steps)

    meta = dict(T_src=T_src, J=J, fps_src=fps_src, fps_out=fps_out, c=c, s_end=s_end,
                phase=phase, v_star=float(v_star), v_src_clean=v_src_clean, dpds=dpds,
                sdot_star=sdot_star, vel_cap=vel_cap, acc_budget=acc_budget,
                base_info=base_info, half=half, src_duration=src_duration,
                vel_limit_frac=vel_limit_frac,
                cop_gate=cop_gate, fric_gate=fric_gate, tau_gate=tau_gate,
                compress_step=compress_step, expand_step=expand_step,
                scale_min=scale_min, scale_max=scale_max, refine_steps=refine_steps,
                max_inner=max_inner)
    return res.best.out, res, law, meta


# ====================================================================================== #
# report                                                                                  #
# ====================================================================================== #
def build_report(data: dict, res: MinTimeResult, law: "v1.TimeLaw", meta: dict,
                 body_mode: str) -> dict:
    best, r1 = res.best, res.gamma1
    out, warp, field = best.out, best.warp, best.field
    k_star, T_out = warp.k_star, warp.T_out
    T_src, fps_src, c, fps_out = meta["T_src"], meta["fps_src"], meta["c"], meta["fps_out"]

    # 保真(v1/v2 同口径:触球行逐位、干净拍速、拍面法向、首帧健康)
    blade_out = v1.blade_positions(out)
    v_out_clean = v1.clean_speed_at(blade_out, k_star, 1.0 / fps_out)
    n_src = v1.blade_face_normals(data)[c]
    n_out = v1.blade_face_normals(out)[k_star]
    face_deg = float(np.degrees(np.arccos(np.clip(
        np.dot(n_src, n_out) / (np.linalg.norm(n_src) * np.linalg.norm(n_out)), -1, 1))))
    contact_bitwise = bool(np.array_equal(out["joint_pos"][k_star],
                                          np.asarray(data["joint_pos"])[c]))
    dq_out = np.asarray(out["joint_vel"], dtype=np.float64)
    first_frame_vel = float(np.abs(dq_out[0]).max())
    mean_acc_out = float(np.abs(np.diff(dq_out, axis=0) * fps_out).mean())
    dq_src = np.asarray(data["joint_vel"], dtype=np.float64)
    mean_acc_src = float(np.abs(np.diff(dq_src, axis=0) * fps_src).mean())
    speed_dev = abs(v_out_clean - meta["v_star"]) / meta["v_star"]

    src_duration = meta["src_duration"]
    dur = warp.duration_s
    dur_x_src = dur / src_duration
    direction = ("accelerated" if dur_x_src < 0.999
                 else ("slowed" if dur_x_src > 1.001 else "unchanged"))

    # 运动学利用率落点(窗外 = 硬边界必须干净;窗内 = |v*| 钉死,只报不判)
    lw = field.lock_weight(warp.s_out)
    inside = lw < 0.5
    kin = dict(
        vel_util_max_out_window=(round(float(best.vel_util[~inside].max()), 4)
                                 if (~inside).any() else None),
        acc_util_max_out_window=(round(float(best.acc_util[~inside].max()), 4)
                                 if (~inside).any() else None),
        vel_util_max_in_window=(round(float(best.vel_util[inside].max()), 4)
                                if inside.any() else None),
        kin_bad_in_window=best.breakdown.get("kin_bad_in_window", 0),
        note="窗内利用率由触球拍速 |v*| 几何钉死,时间律不可约;窗外必须 ≤1(硬边界)")

    # ρ 场分段(压缩段 rho_min < 1 直接可见)
    sg, rho = field.s_grid, field.rho
    half = meta["half"]
    pre = sg < (c - half)
    win = np.abs(sg - c) <= half
    post = sg > (c + half)

    def seg_stats(m):
        if not m.any():
            return dict(rho_mean=1.0, rho_min=1.0, rho_max=1.0)
        return dict(rho_mean=round(float(rho[m].mean()), 3),
                    rho_min=round(float(rho[m].min()), 3),
                    rho_max=round(float(rho[m].max()), 3))

    r0, rf = res.reading0, best.reading
    report = dict(
        tool="topp_mintime.py v3 (unified-budget min-time bidirectional retiming)",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        verdict=rf.verdict,
        direction=direction,
        chosen_scale=round(best.gamma, 4),
        feasible_reason=best.reason,
        budget=dict(cop_gate=meta["cop_gate"], fric_gate=meta["fric_gate"],
                    tau_gate=meta["tau_gate"], vel_limit_frac=meta["vel_limit_frac"],
                    kin_vel_target=KIN_VEL_TARGET, kin_acc_target=KIN_ACC_TARGET,
                    note="统一预算:oracle 三剂量闸门 + URDF 速度限位×余量 + |acc| 实证包络"),
        acceptance=dict(
            cop_dose_final=round(rf.doses.get("cop", 0.0), 4),
            fric_dose_final=round(rf.doses.get("friction", 0.0), 4),
            tau_dose_final=round(rf.doses.get("torque", 0.0), 4),
            within_budget=bool(best.feasible),
            kin_out_window_clean=bool(best.breakdown.get("kin_bad_out_window", 0) == 0)),
        durations=dict(
            source_s=round(src_duration, 4),
            baseline_gamma1_s=round(r1.duration_s, 4),
            baseline_gamma1_feasible=bool(r1.feasible),
            mintime_s=round(dur, 4),
            vs_source_x=round(dur_x_src, 4),
            vs_gamma1_x=round(dur / r1.duration_s, 4),
            note="franco 两侧难度拉平假说的对账数:同预算下正/反手各自的 min-time 总时长"),
        oracle_before=dict(verdict=r0.verdict, cop_dose=round(r0.doses.get("cop", 0.0), 4),
                           fric_dose=round(r0.doses.get("friction", 0.0), 4),
                           tau_dose=round(r0.doses.get("torque", 0.0), 4)),
        oracle_after=dict(verdict=rf.verdict, cop_dose=round(rf.doses.get("cop", 0.0), 4),
                          fric_dose=round(rf.doses.get("friction", 0.0), 4),
                          tau_dose=round(rf.doses.get("torque", 0.0), 4)),
        kin=kin,
        source=dict(frames=int(T_src), fps=fps_src, contact_frame=int(c),
                    phase=float(meta["phase"]), runup_s=round(c / fps_src, 4),
                    duration_s=round(src_duration, 4),
                    clean_blade_speed_mps=round(meta["v_src_clean"], 4),
                    mean_abs_acc=round(mean_acc_src, 3)),
        answer=dict(v_star_mps=round(float(meta["v_star"]), 4),
                    v_star_source=("source-clean"
                                   if abs(meta["v_star"] - meta["v_src_clean"]) < 1e-12
                                   else "cli-override"),
                    blade_dpds_m_per_frame=round(meta["dpds"], 6),
                    sdot_star_frames_per_s=round(meta["sdot_star"], 4)),
        baseline_law=dict(Ta_s=round(law.Ta, 4),
                          ta_max_s=round(meta["base_info"]["ta_max_s"], 4),
                          irreducible_kinematic=bool(meta["base_info"]["irreducible"])),
        stretch=dict(pre_approach=seg_stats(pre), lock_window=seg_stats(win),
                     follow_through=seg_stats(post),
                     rho_global_min=round(float(rho.min()), 3),
                     rho_global_max=round(float(rho.max()), 3)),
        output=dict(frames=int(T_out), fps=float(fps_out), contact_frame=int(k_star),
                    phase_out=round(k_star / (T_out - 1), 6),
                    runup_s=round(k_star / fps_out, 4),
                    duration_s=round((T_out - 1) / fps_out, 4),
                    runup_change_x=round((k_star / fps_out) / (c / fps_src), 3),
                    duration_change_x=round(((T_out - 1) / fps_out) / src_duration, 3),
                    wait_s=round(warp.wait_s, 4), body_mode=body_mode,
                    mean_abs_acc=round(mean_acc_out, 3)),
        fidelity=dict(contact_row_bitwise=contact_bitwise,
                      blade_speed_clean_out_mps=round(v_out_clean, 4),
                      blade_speed_dev_frac=round(speed_dev, 5),
                      face_normal_diff_deg=round(face_deg, 6),
                      first_frame_max_joint_vel=round(first_frame_vel, 4)),
        outer_trace=res.outer_trace,
        inner_trace_best=best.trace,
    )
    return report


def report_md(rep: dict) -> str:
    a, d, s, o, f, b, k = (rep["acceptance"], rep["durations"], rep["source"],
                           rep["output"], rep["fidelity"], rep["budget"], rep["kin"])
    ob, oa = rep["oracle_before"], rep["oracle_after"]
    st = rep["stretch"]
    zh = {"accelerated": "加速(预算内有富余,健康动作)",
          "slowed": "放慢(预算顶住,v2 语义兜底)",
          "unchanged": "不变(已在预算边界)"}[rep["direction"]]
    lines = [
        f"# min-time 双向重定时 (TOPP v3) — **{rep['verdict']}** / 方向 **{rep['direction']}** "
        f"(γ={rep['chosen_scale']}, {rep['feasible_reason']})",
        "",
        f"- generated: {rep['generated_utc']}",
        f"- 人话:统一预算(CoP≤{b['cop_gate']} / fric≤{b['fric_gate']} / τ≤{b['tau_gate']} / "
        f"速度≤URDF×{b['vel_limit_frac']:g} / |acc|≤实证包络)下,这个动作的最快时间 = "
        f"**{d['mintime_s']:.2f} s**;{zh}",
        f"- 时长对账:源 {d['source_s']:.2f} s (x{d['vs_source_x']:.2f}) | γ=1 基线修复后 "
        f"{d['baseline_gamma1_s']:.2f} s (x{d['vs_gamma1_x']:.2f} vs 基线, 基线可行="
        f"{d['baseline_gamma1_feasible']}) —— 两侧难度拉平假说请对比正/反手这一行",
        f"- oracle: {ob['verdict']} (CoP {ob['cop_dose']:.3f}/fric {ob['fric_dose']:.3f}/"
        f"τ {ob['tau_dose']:.3f}) -> {oa['verdict']} (CoP {oa['cop_dose']:.3f}/"
        f"fric {oa['fric_dose']:.3f}/τ {oa['tau_dose']:.3f}); within_budget={a['within_budget']}",
        f"- 运动学硬边界:窗外峰值利用率 vel {k['vel_util_max_out_window']} / "
        f"acc {k['acc_util_max_out_window']} (必须≤1, clean={a['kin_out_window_clean']}); "
        f"窗内 vel {k['vel_util_max_in_window']}(|v*| 钉死,不可约 {k['kin_bad_in_window']} 帧)",
        f"- 源: {s['frames']} 帧 @ {s['fps']:.0f} fps = {s['duration_s']:.2f} s; 触球 "
        f"f{s['contact_frame']} (phase {s['phase']}); 干净拍速 {s['clean_blade_speed_mps']:.3f} m/s",
        f"- 出: {o['frames']} 帧 @ {o['fps']:.0f} fps = {o['duration_s']:.2f} s; 触球 "
        f"f{o['contact_frame']} -> **phase_out {o['phase_out']:.4f}**; 助跑 x{o['runup_change_x']:.2f}; "
        f"wait {o['wait_s']:.3f} s; body_mode={o['body_mode']}; "
        f"mean|acc| {s['mean_abs_acc']:.2f} -> {o['mean_abs_acc']:.2f}",
        f"- ρ 场:pre [{st['pre_approach']['rho_min']:.2f},{st['pre_approach']['rho_max']:.2f}] "
        f"| 锁窗 {st['lock_window']['rho_mean']:.2f} | follow [{st['follow_through']['rho_min']:.2f},"
        f"{st['follow_through']['rho_max']:.2f}] | 全局 [{st['rho_global_min']:.2f},"
        f"{st['rho_global_max']:.2f}] (<1 = 压缩段, >1 = 修复段)",
        f"- 保真:触球行逐位={f['contact_row_bitwise']}; 拍速 {f['blade_speed_clean_out_mps']:.3f} m/s "
        f"(dev {f['blade_speed_dev_frac'] * 100:.2f}%); 拍面差 {f['face_normal_diff_deg']:.4f} deg; "
        f"首帧 max|q̇| {f['first_frame_max_joint_vel']:.3f} rad/s",
        "",
        "| γ | 可行 | 原因 | 内层轮数 | T_out | 时长 s | CoP | fric | τ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rep["outer_trace"]:
        lines.append(
            f"| {r['gamma']:.3f} | {r['feasible']} | {r['reason']} | {r['iters']} | "
            f"{r['T_out']} | {r['duration_s']} | {r['cop']} | {r['fric']} | {r['tau']} |")
    lines += ["",
              f"REGISTRY REMINDER: SYNTHESIZED timeline — register phase_out = "
              f"{o['phase_out']:.4f} (contact frame {o['contact_frame']} of {o['frames']}) "
              f"in cfg/strike_annotations.yaml. 视频约定帧对本资产弃用。"]
    return "\n".join(lines)


# ====================================================================================== #
# CLI                                                                                     #
# ====================================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="源 clip npz")
    ap.add_argument("--output", required=True, help="输出 npz(min-time 重定时资产)")
    ap.add_argument("--phase", type=float, required=True,
                    help="源 clip 的登记触球相位(strike_annotations.yaml)")
    ap.add_argument("--strike-speed", type=float, default=None,
                    help="答案拍速 |v*| m/s(默认=源干净拍速;改它=变速重解)")
    ap.add_argument("--budget-clips", nargs="+", required=True,
                    help="|acc| 实证包络的来源 clips(先例:v4rg 一对=可执行地板)")
    ap.add_argument("--budget-scale", type=float, default=1.5,
                    help="包络放大倍数(v1/v2 先例 1.5)")
    ap.add_argument("--vel-limit-frac", type=float, default=0.85,
                    help="URDF 速度限位余量(先例 0.85;压缩的硬边界)")
    ap.add_argument("--urdf", default=None, help="A3 URDF(默认 repo 内拷贝)")
    ap.add_argument("--mjcf", default=None, help="deploy/oracle MJCF(free-root);必填")
    ap.add_argument("--body-order", default=None, help="body 顺序文件;fk + oracle 必填")
    ap.add_argument("--fps-out", type=float, default=None, help="默认=源 fps")
    ap.add_argument("--min-cruise-s", type=float, default=0.04)
    ap.add_argument("--post-contact-hold-s", type=float, default=0.04)
    ap.add_argument("--ta-grid-s", type=float, default=0.005)
    ap.add_argument("--dense-dt-s", type=float, default=0.002)
    ap.add_argument("--body-mode", choices=("fk", "interp"), default="fk",
                    help="fk = MuJoCo FK 重建(产线);interp 只给 CPU 测试")
    ap.add_argument("--mu", type=float, default=0.8, help="oracle 摩擦系数")
    ap.add_argument("--support-band", type=float, default=0.03)
    ap.add_argument("--dose-target", type=float, default=DEFAULT_COP_GATE,
                    help=f"CoP 剂量闸门(统一预算主旋钮;默认 {DEFAULT_COP_GATE}=v2 校准值,"
                         "调小=收紧=时长只会变长)")
    ap.add_argument("--fric-gate", type=float, default=DEFAULT_FRIC_GATE,
                    help=f"摩擦剂量闸门(默认 {DEFAULT_FRIC_GATE}=oracle 校准值)")
    ap.add_argument("--tau-gate", type=float, default=DEFAULT_TAU_GATE,
                    help=f"τ 剂量闸门(默认 {DEFAULT_TAU_GATE}=oracle 校准值)")
    ap.add_argument("--max-inner-iters", type=int, default=DEFAULT_MAX_INNER,
                    help="每个 γ 的内层修复迭代上限")
    ap.add_argument("--compress-step", type=float, default=DEFAULT_COMPRESS_STEP,
                    help="外层下探步长 γ←γ×step(<1)")
    ap.add_argument("--expand-step", type=float, default=DEFAULT_EXPAND_STEP,
                    help="外层上探步长 γ←γ×step(>1;γ=1 不可行时)")
    ap.add_argument("--scale-min", type=float, default=DEFAULT_SCALE_MIN,
                    help="γ 下界(防失控压缩)")
    ap.add_argument("--scale-max", type=float, default=DEFAULT_SCALE_MAX,
                    help="γ 上界(放慢到此仍不可行=几何问题,fail loud)")
    ap.add_argument("--refine-steps", type=int, default=DEFAULT_REFINE_STEPS,
                    help="好/坏 γ 之间几何二分收尾步数")
    ap.add_argument("--oracle-workdir", default=None,
                    help="每轮判卷临时 npz 的目录(默认新 tmpdir)")
    ap.add_argument("--report", default=None, help="JSON 报告输出路径")
    ap.add_argument("--md", default=None, help="markdown 报告输出路径")
    args = ap.parse_args(argv)

    if args.urdf is None:
        args.urdf = os.path.normpath(os.path.join(
            _HERE, "../../..", "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"))
    from audit_motion_npz import parse_urdf_limits  # via sys.path insert
    limits = parse_urdf_limits(args.urdf)
    vlim = np.array([limits[n].velocity if limits[n].velocity is not None else np.inf
                     for n in ISAAC_JOINT_NAMES])
    env = v1.acc_envelope(args.budget_clips)
    acc_budget = env * args.budget_scale

    if not args.mjcf or not args.body_order:
        raise SystemExit("--mjcf and --body-order are required (oracle + FK)")

    fk_ctx = None
    if args.body_mode == "fk":
        fkm = v1.ctn.MjFK(args.mjcf, ISAAC_JOINT_NAMES)
        names = fkm.body_names()
        order = [ln.strip() for ln in open(args.body_order) if ln.strip()]
        fk_ctx = (fkm, [names.index(n) for n in order], tuple(order))

    judge = v2.RealOracle(args.mjcf, args.body_order, args.mu, args.support_band,
                          workdir=args.oracle_workdir)

    data = dict(np.load(args.input))
    stem = Path(args.input).name[:-4] if args.input.endswith(".npz") else Path(args.input).stem
    out, res, law, meta = mintime(
        data, args.phase, vlim, acc_budget, judge, stem,
        v_star=args.strike_speed, vel_limit_frac=args.vel_limit_frac,
        fps_out=args.fps_out, min_cruise_s=args.min_cruise_s,
        post_hold_s=args.post_contact_hold_s, ta_grid_s=args.ta_grid_s,
        dense_dt_s=args.dense_dt_s, body_mode=args.body_mode, fk_ctx=fk_ctx,
        max_inner=args.max_inner_iters, cop_gate=args.dose_target,
        fric_gate=args.fric_gate, tau_gate=args.tau_gate,
        compress_step=args.compress_step, expand_step=args.expand_step,
        scale_min=args.scale_min, scale_max=args.scale_max,
        refine_steps=args.refine_steps)

    rep = build_report(data, res, law, meta, args.body_mode)
    rep["files"] = dict(input=os.path.abspath(args.input), output=os.path.abspath(args.output))
    rep["budget_provenance"] = dict(clips=[os.path.abspath(p) for p in args.budget_clips],
                                    scale=args.budget_scale,
                                    envelope=[round(float(v), 3) for v in env])

    np.savez(args.output, **out)
    md = report_md(rep)
    print(md)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(rep, fh, indent=2)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(md)

    dev = rep["fidelity"]["blade_speed_dev_frac"]
    if dev > 0.02:
        print(f"** WARNING: contact blade-speed deviation {dev * 100:.2f}% > 2% **",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
