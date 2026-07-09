#!/usr/bin/env python3
"""Oracle-guided TIME-LAW SYNTHESIS v2 — TOPP-lite (针对性放慢不可行段) for motion .npz.

人话:v1(synthesize_timing.py)全局匀加速一个 T_a,压不动更低的加速度剂量(bh 卡在
mean|acc| 2.82,dose 0.232)。v2 不再全局匀加速——**只针对 A 层可行性判定器
(feasibility_oracle.py, mj_inverse 逐帧 CoP/摩擦锥/腰τ)判"不可行"的那些帧,在其路径邻域
局部拉长时间轴**:降 |q̇|(∝ṡ),降动力学需求(∝ṡ²,二次降)。触球窗(登记 phase±0.1s)
时间锁死 → 拍速 |v*| / 拍面逐位保真。总时长允许拉长(记 T_a/时长代价)。

设计文档: docs/research/robot_centric_timing_2026-07-09.md §七/八/九
    §七 可执行性真约束 = 腰力矩 + 接触可行性(CoP/摩擦锥);判决 = 不可行剂量制。
    §八/九 时间律 v2 = TOPP(τ 对 s̈ 仿射逐点);贪心近似够用,不追求最优性证明。

ALGORITHM (贪心 TOPP;task spec 2026-07-09 步骤 ①-⑤)
    path q(s):source frame index = 路径参数(与 v1 同,视频只当路径).
    baseline 时间律:REUSE v1.solve_min_ta(同预算:|q̇|≤0.85×URDF, |q̈|≤scale×v4rg 包络)
        → rest→匀加速→巡航(过触球,sdot* = |v*|/|dp/ds|)→hold→匀减速→rest.
        这是"当前时间律"的迭代 0(= v5syn 起点,oracle dose≈0.232).
    时间轴拉长 = 在 baseline 时间律上叠一个 **局部拉伸密度 ρ(s) ≥ 1**(以路径位置 s 为函数):
        new-time τ 满足 dτ/dt = ρ(s_base(t));ṡ_new = ṡ_base/ρ(局部降速);ρ=1 处原样穿过.
        触球窗 [c-0.1s·fps, c+0.1s·fps] 内 ρ≡1(锁死 → 触球拍速/拍面不变);两端本就静止.
    迭代:
        ① 对当前候选跑 feasibility_oracle 逐帧(mj_inverse + CoP/摩擦锥/腰τ).
        ② 找不可行帧(CoP 出支撑面 / 摩擦锥破 1 / 腰τ 破限),映回其路径位置 s_k.
        ③ 窗外的不可行帧:在其 s_k 邻域(高斯)抬高 ρ —— τ/摩擦按 √(over/target)(二次律),
           CoP 按固定收缩(非闭式,迭代收敛);窗内不可行帧无法放慢(锁死)= 记为几何不可约.
        ④ 重建 τ 映射 → 输出网格重采样 joint_pos + MuJoCo FK 重算 body_*(触球行逐位).
        ⑤ 重判;收敛 = verdict PASS 或 不可行剂量降到目标(< v5syn 0.232,力争 ≤ v4rg 0.167)
           或窗外无可放慢帧(剩余全在窗内 → 见下)或迭代上限.
    保留剂量最低的候选为输出.

⚠ 结论口径(task 2026-07-09):TOPP-lite 收敛后若 oracle 仍不 PASS 且剩余不可行帧**全落在
    触球锁窗内** = 触球窗几何本身不可行(非时序问题)——指向路径 morph(改 |v*| 或改空间路径)
    而非时间律。报告如实标注 residual 落点(窗内 vs 窗外),不硬造 PASS。

NEW PHASE 口径 (同 v1):触球行 = 源帧 c 在输出网格 k* 的逐位拷贝;phase_out = k*/(T_out-1).
    视频约定帧对本资产弃用 → 登记 phase_out.

USAGE (pod, hope_mjeval_venv:numpy + mujoco)
    <venv>/bin/python hope_training/whole_body_tracking/scripts/synthesize_timing_v2.py \
        --input  .../v5_height_fix/hope_backhand_v5hLs_cal.npz --phase 0.391 \
        --output .../v5_height_fix/hope_backhand_v5topp_cal.npz \
        --budget-clips .../regen_0708_candidates/hope_backhand_v4rg_cal.npz \
                       .../regen_0708_candidates/hope_forehand_v4rg_cal.npz \
        --mjcf .../a3_pingpong.xml --body-order /workspace/franco/body_order_isaac.txt \
        --report bh_topp.json --md bh_topp.md

DEPENDENCIES: numpy always; mujoco for --body-mode fk + the oracle (import deferred).
    Pure-math TOPP core (ρ warp / resample / lock / convergence) is unit-tested on CPU
    with a STUB oracle: tests/test_synthesize_timing_v2.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# reuse v1 (path extraction, blade FK, baseline law, budgets, FK resample bits) --------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import synthesize_timing as v1  # noqa: E402  (numpy-only at import time)

# feasibility oracle lives at REPO_ROOT/scripts/ (different tree) ----------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

ISAAC_JOINT_NAMES = v1.ISAAC_JOINT_NAMES

# ---- TOPP-lite defaults (greedy; see module docstring) -------------------------------- #
STRIKE_HALF_S = 0.10          # ±0.1 s lock window around contact (登记口径, phase±0.1s)
LOCK_TAPER_FR = 3.0           # frames over which ρ ramps 1 -> free at the window edge
BUMP_SIGMA_FR = 2.5           # gaussian half-width of a local ρ bump [source frames]
RHO_STEP_MAX = 1.6            # max multiplicative ρ growth per frame per iteration
RHO_MAX = 40.0                # absolute ρ ceiling (guards runaway duration)
TORQUE_TARGET = 0.95          # bring flagged |τ|/τmax down to this
FRICTION_TARGET = 0.95        # bring flagged |f_t|/(μ fz) down to this
COP_SHRINK = 0.85             # per-iter speed shrink for a flagged CoP frame (v -> 0.85 v)
DURATION_GUARD_X = 6.0        # refuse to stretch past this multiple of the source duration
DEFAULT_MAX_ITERS = 25
DOSE_ACCEPT = 0.232           # v5syn CoP dose — must beat this (acceptance ①)
V4RG_DOSE = 0.167             # v4rg CoP dose — the 'meets v4rg' report threshold
DOSE_TARGET = 0.10            # loop STOP target: comfortably below v4rg (leaves margin
                              # without chasing the absolute floor via denominator dilution)
# fric/tau dose tolerances at the stop point. NOT zero — even v4rg (fall 0.02, the floor)
# carries isolated torque spikes (172% one) & the whip's transient friction, which are
# demonstrably not fall-predictive (feasibility_oracle VERDICT note). Requiring exactly 0
# would force pointless over-stretching of the follow-through.
FRIC_ACCEPT = 0.05            # oracle DOSE_FRIC_FAIL
TAU_ACCEPT = 0.02             # oracle DOSE_TAU_WARN


# ====================================================================================== #
# Oracle reading (per-frame feasibility metrics on the OUTPUT grid)                       #
# ====================================================================================== #
@dataclass
class OracleReading:
    """Per-output-frame feasibility metrics + doses + verdict (the loop's ② input)."""
    cop_excess: np.ndarray        # (T,) signed dist to support hull [m]; >0 = outside
    fric_ratio: np.ndarray        # (T,) |f_t| / (μ fz)
    util_max: np.ndarray          # (T,) max_j |τ_j|/τ_max_j
    fz: np.ndarray                # (T,) required vertical ground force [N]
    doses: Dict[str, float]       # {'cop','friction','torque'}
    verdict: str                  # PASS / WARN / FAIL
    contact_frame: Optional[int] = None
    raw: object = None            # the underlying ClipReport (production only)

    @property
    def n(self) -> int:
        return int(self.cop_excess.shape[0])


# A judge maps (out_dict, stem, phase_out) -> OracleReading. Production = real mj_inverse
# oracle; tests inject a deterministic stub so the TOPP core runs without mujoco.
Judge = Callable[[dict, str, float], OracleReading]


class RealOracle:
    """Production judge: writes a temp npz and runs feasibility_oracle.analyze_clip."""

    def __init__(self, mjcf: str, body_order: str, mu: float, support_band: float,
                 workdir: Optional[str] = None):
        import feasibility_oracle as fo  # deferred (needs mujoco)
        self.fo = fo
        self.om = fo.load_oracle_model(mjcf)
        self.body_order = body_order
        self.mu = mu
        self.support_band = support_band
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="topp_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def __call__(self, out: dict, stem: str, phase_out: float) -> OracleReading:
        self._n += 1
        p = self.workdir / f"{stem}__topp_iter{self._n:02d}.npz"
        np.savez(p, **out)
        anno = {stem: {"phase": float(phase_out)}}
        rep = self.fo.analyze_clip(self.om, p, anno, self.body_order,
                                   self.mu, self.support_band)
        T = rep.n_frames
        util = rep.util
        util_max = (np.abs(util).max(axis=1) if util is not None
                    else np.zeros(T))
        cop = rep.cop_excess if rep.cop_excess is not None else np.full(T, np.nan)
        fric = rep.fric_ratio if rep.fric_ratio is not None else np.full(T, np.nan)
        fz = rep.fz if rep.fz is not None else np.zeros(T)
        return OracleReading(cop_excess=np.asarray(cop, float),
                             fric_ratio=np.asarray(fric, float),
                             util_max=np.asarray(util_max, float),
                             fz=np.asarray(fz, float),
                             doses=dict(rep.doses), verdict=rep.verdict,
                             contact_frame=rep.contact_frame, raw=rep)


# ====================================================================================== #
# ρ(s): local time-stretch density on the source-path grid                               #
# ====================================================================================== #
@dataclass
class StretchField:
    """ρ(s) ≥ 1 on a uniform source-path grid. ρ=1 => baseline timing passes through;
    ρ>1 => locally slower (dwell time multiplied by ρ). The contact lock window keeps
    ρ≡1 (contact blade speed preserved); a cosine taper ramps ρ 1->free at the edges."""
    s_grid: np.ndarray            # (M,) uniform 0..s_end [source frames]
    rho: np.ndarray               # (M,) >= 1
    c: float                      # contact path position [frames]
    half: float                   # lock half-window [frames]
    taper: float                  # edge taper [frames]

    @classmethod
    def build(cls, s_end: float, c: float, half: float, taper: float,
              ds: float = 0.1) -> "StretchField":
        M = int(np.floor(s_end / ds)) + 1
        s_grid = np.linspace(0.0, s_end, M)
        return cls(s_grid=s_grid, rho=np.ones(M), c=c, half=half, taper=taper)

    def lock_weight(self, s: np.ndarray) -> np.ndarray:
        """0 inside the lock window, smoothstep -> 1 beyond edge+taper (both sides)."""
        d = np.abs(np.asarray(s, float) - self.c) - self.half
        if self.taper <= 0:
            return (d > 0).astype(float)
        x = np.clip(d / self.taper, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)     # smoothstep

    def bump(self, centers: np.ndarray, factors: np.ndarray) -> None:
        """Raise ρ near each path center by (factor-1), gaussian-windowed & lock-masked.
        Overlapping bumps combine by the MAX additive part (gentler than product)."""
        if len(centers) == 0:
            return
        lw = self.lock_weight(self.s_grid)
        add = np.zeros_like(self.rho)
        for sc, fac in zip(centers, factors):
            fac = float(min(fac, RHO_STEP_MAX))
            if fac <= 1.0:
                continue
            g = np.exp(-0.5 * ((self.s_grid - sc) / BUMP_SIGMA_FR) ** 2)
            add = np.maximum(add, (fac - 1.0) * g * lw)
        self.rho = np.minimum(self.rho * (1.0 + add), RHO_MAX)
        self.rho = np.maximum(self.rho, 1.0)

    def of_s(self, s: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(s, float), self.s_grid, self.rho)


# ====================================================================================== #
# warp: baseline law s_base(t) + ρ(s) -> output-grid s(t), contact snapped onto the grid  #
# ====================================================================================== #
@dataclass
class WarpResult:
    s_out: np.ndarray             # (T_out,) source-path position per output frame
    T_out: int
    k_star: int                   # output contact frame
    wait_s: float                 # ready wait before motion (grid snap)
    t_end_warp: float             # warped motion end time (excl. wait/pad) [s]
    duration_s: float             # total output duration [s]


def warp_timeline(law: "v1.TimeLaw", field: StretchField, fps_out: float,
                  dense_dt_s: float = 0.001) -> WarpResult:
    """Integrate dτ/dt = ρ(s_base(t)) over the baseline law, snap the contact time onto
    the output grid via a ready-wait, and sample s on the uniform output grid."""
    t_star = float(law.t_star)
    # dense baseline sampling; force t_star to be an exact node (contact-row exactness)
    t_dense = np.arange(0.0, law.t_end + dense_dt_s, dense_dt_s)
    t_dense = np.unique(np.concatenate([t_dense, [t_star, law.t_end]]))
    s_dense, _, _ = law.s_sdot_sddot(t_dense)
    rho_dense = field.of_s(s_dense)
    # τ(t) = ∫ ρ dt  (trapezoid), τ(0)=0
    dtau = 0.5 * (rho_dense[1:] + rho_dense[:-1]) * np.diff(t_dense)
    tau_dense = np.concatenate([[0.0], np.cumsum(dtau)])
    idx_star = int(np.searchsorted(t_dense, t_star))
    tau_star = float(tau_dense[idx_star])
    tau_end = float(tau_dense[-1])

    # ready-wait so the contact lands EXACTLY on an output grid frame (v1's tw trick)
    k_star = int(np.ceil(tau_star * fps_out - 1e-9))
    k_star = max(k_star, 1)
    wait = k_star / fps_out - tau_star
    if wait < 0.0:
        wait = 0.0

    total = wait + tau_end
    T_out = int(np.ceil(total * fps_out - 1e-9)) + 1 + 1   # +1 rest-pad frame
    tau_out = np.arange(T_out, dtype=np.float64) / fps_out

    s_out = np.empty(T_out)
    for j in range(T_out):
        tau_m = tau_out[j] - wait
        if tau_m <= 0.0:
            s_out[j] = 0.0
        elif tau_m >= tau_end:
            s_out[j] = law.s_end
        else:
            t_j = float(np.interp(tau_m, tau_dense, t_dense))
            s_out[j] = float(law.s_sdot_sddot(np.array([t_j]))[0][0])
    s_out = np.clip(s_out, 0.0, law.s_end)
    # exactness pin: contact frame carries the source contact path position
    s_out[k_star] = law.c
    return WarpResult(s_out=s_out, T_out=T_out, k_star=k_star, wait_s=wait,
                      t_end_warp=tau_end, duration_s=(T_out - 1) / fps_out)


# ====================================================================================== #
# resample: source path sampled at s_out; joint_vel re-diff, body_* via MuJoCo FK         #
# ====================================================================================== #
def resample_at_s(data: dict, s_out: np.ndarray, fps_out: float,
                  body_mode: str = "interp", fk_ctx=None) -> dict:
    """Same conventions as v1.resample but driven by an explicit s(t) (the warp)."""
    q = np.asarray(data["joint_pos"])
    dt = 1.0 / fps_out
    jp = v1._interp_rows(q, s_out).astype(np.float32)
    jv = np.gradient(jp.astype(np.float64), dt, axis=0).astype(np.float32)

    if body_mode == "fk":
        fkm, cols = fk_ctx
        base_pos = v1._interp_rows(np.asarray(data["body_pos_w"], float)[:, 0], s_out)
        base_quat = v1._slerp_rows(np.asarray(data["body_quat_w"], float)[:, 0], s_out)
        pos_all, quat_all = v1.ctn.fk_series(fkm, base_pos, base_quat,
                                             jp.astype(np.float64), ISAAC_JOINT_NAMES)
        bp = pos_all[:, cols].astype(np.float32)
        bq = quat_all[:, cols].astype(np.float32)
    elif body_mode == "interp":
        bp = v1._interp_rows(np.asarray(data["body_pos_w"], float), s_out).astype(np.float32)
        bq = v1._slerp_rows(np.asarray(data["body_quat_w"], float), s_out).astype(np.float32)
    else:
        raise SystemExit(f"unknown --body-mode {body_mode!r}")

    bl = np.gradient(bp.astype(np.float64), dt, axis=0).astype(np.float32)
    ba = np.stack([v1.ctn.so3_derivative(bq[:, b].astype(np.float64), dt)
                   for b in range(bq.shape[1])], axis=1).astype(np.float32)
    return {"fps": np.array([int(round(fps_out))], dtype=np.int64),
            "joint_pos": jp, "joint_vel": jv, "body_pos_w": bp, "body_quat_w": bq,
            "body_lin_vel_w": bl, "body_ang_vel_w": ba}


# ====================================================================================== #
# the TOPP-lite loop                                                                      #
# ====================================================================================== #
def _flagged_bumps(reading: OracleReading, s_out: np.ndarray, field: StretchField
                   ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """From an oracle reading, return (path centers, ρ bump factors) for the WINDOW-OUTSIDE
    infeasible frames, plus a breakdown dict (incl. how many infeasible frames are locked
    INSIDE the window = geometrically irreducible)."""
    T = reading.n
    cop = reading.cop_excess
    fric = reading.fric_ratio
    util = reading.util_max
    lw = field.lock_weight(s_out)          # ~0 inside window, ~1 outside
    inside = lw < 0.5

    cop_bad = np.nan_to_num(cop, nan=-np.inf) > v1_cop_warn()
    fric_bad = np.nan_to_num(fric, nan=-np.inf) > 1.0
    tau_bad = util > 1.0
    any_bad = cop_bad | fric_bad | tau_bad

    centers, factors = [], []
    for k in range(T):
        if not any_bad[k] or inside[k]:
            continue
        fac = 1.0
        if tau_bad[k]:
            fac = max(fac, np.sqrt(util[k] / TORQUE_TARGET))
        if fric_bad[k]:
            fac = max(fac, np.sqrt(fric[k] / FRICTION_TARGET))
        if cop_bad[k]:
            fac = max(fac, 1.0 / COP_SHRINK)
        centers.append(float(s_out[k]))
        factors.append(float(fac))

    breakdown = dict(
        cop_bad=int(cop_bad.sum()), fric_bad=int(fric_bad.sum()),
        tau_bad=int(tau_bad.sum()),
        bad_in_window=int((any_bad & inside).sum()),
        bad_out_window=int((any_bad & ~inside).sum()),
        cop_bad_in_window=int((cop_bad & inside).sum()),
        cop_bad_out_window=int((cop_bad & ~inside).sum()),
    )
    return np.array(centers), np.array(factors), breakdown


def v1_cop_warn() -> float:
    """CoP 'outside at all' threshold — mirror the oracle's COP_WARN_M without importing it
    on CPU-test hosts (the value is a documented constant = 0.0 m)."""
    return 0.0


@dataclass
class ToppResult:
    out: dict
    stretch_field: StretchField
    warp: WarpResult
    reading0: OracleReading           # iteration-0 (baseline / v5syn-equivalent)
    reading: OracleReading            # accepted (best) reading
    iters: int
    trace: List[dict] = dc_field(default_factory=list)
    converged_reason: str = ""


def topp_lite(data: dict, phase: float, vlim: np.ndarray, acc_budget: np.ndarray,
              judge: Judge, stem: str, v_star: Optional[float] = None,
              vel_limit_frac: float = 0.85, fps_out: Optional[float] = None,
              min_cruise_s: float = 0.04, post_hold_s: float = 0.04,
              ta_grid_s: float = 0.005, dense_dt_s: float = 0.002,
              body_mode: str = "interp", fk_ctx=None,
              max_iters: int = DEFAULT_MAX_ITERS,
              dose_accept: float = DOSE_ACCEPT, dose_target: float = DOSE_TARGET
              ) -> Tuple[dict, ToppResult, "v1.TimeLaw", dict]:
    """Greedy oracle-guided time-law synthesis. Returns (out, ToppResult, law, meta)."""
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

    vel_cap = np.asarray(vlim, float) * vel_limit_frac
    acc_budget = np.asarray(acc_budget, float)
    if (acc_budget <= 0).any():
        floor = acc_budget[acc_budget > 0].min() if (acc_budget > 0).any() else 1.0
        acc_budget = np.where(acc_budget <= 0, floor, acc_budget)

    qp = np.gradient(q, axis=0)
    qpp = np.gradient(qp, axis=0)

    # baseline (iteration 0) law — v1's uniform-accel solve, same budgets
    law, base_info = v1.solve_min_ta(c, s_end, sdot_star, qp, qpp, vel_cap, acc_budget,
                                     fps_out, min_cruise_s, post_hold_s, ta_grid_s,
                                     dense_dt_s)

    half = STRIKE_HALF_S * fps_src
    field = StretchField.build(s_end, c, half, LOCK_TAPER_FR)
    src_duration = (T_src - 1) / fps_src

    def make_candidate(fld: StretchField):
        warp = warp_timeline(law, fld, fps_out)
        out = resample_at_s(data, warp.s_out, fps_out, body_mode, fk_ctx)
        phase_out = warp.k_star / (warp.T_out - 1)
        reading = judge(out, stem, phase_out)
        return out, warp, reading, phase_out

    out, warp, reading, phase_out = make_candidate(field)
    reading0 = reading
    best = dict(out=out, warp=warp, reading=reading, field=StretchField(
        field.s_grid.copy(), field.rho.copy(), field.c, field.half, field.taper),
        score=reading.doses.get("cop", 1.0))
    trace = [_trace_row(0, reading, warp, src_duration, {})]

    reason = "max_iters"
    for it in range(1, max_iters + 1):
        centers, factors, breakdown = _flagged_bumps(reading, warp.s_out, field)
        cop_dose = reading.doses.get("cop", 1.0)
        # accept? verdict PASS, or CoP dose target met with fric/tau within tolerance
        # (not exactly 0 — see FRIC_ACCEPT/TAU_ACCEPT rationale)
        fric_ok = reading.doses.get("friction", 0.0) <= FRIC_ACCEPT
        tau_ok = reading.doses.get("torque", 0.0) <= TAU_ACCEPT
        if reading.verdict == "PASS" or (cop_dose <= dose_target and fric_ok and tau_ok):
            reason = "pass" if reading.verdict == "PASS" else "dose_target"
            break
        if len(centers) == 0:
            reason = "no_out_of_window_flags"   # residual infeasibility is window-locked
            break
        if warp.duration_s > DURATION_GUARD_X * src_duration:
            reason = "duration_guard"
            break

        field.bump(centers, factors)
        out, warp, reading, phase_out = make_candidate(field)
        trace.append(_trace_row(it, reading, warp, src_duration, breakdown))

        score = reading.doses.get("cop", 1.0)
        if score < best["score"] - 1e-9:
            best = dict(out=out, warp=warp, reading=reading, field=StretchField(
                field.s_grid.copy(), field.rho.copy(), field.c, field.half, field.taper),
                score=score)

    # pick the best-dose candidate seen (may be the current or an earlier one)
    if reading.doses.get("cop", 1.0) <= best["score"] + 1e-9:
        best = dict(out=out, warp=warp, reading=reading,
                    field=field, score=reading.doses.get("cop", 1.0))

    res = ToppResult(out=best["out"], stretch_field=best["field"], warp=best["warp"],
                     reading0=reading0, reading=best["reading"], iters=len(trace) - 1,
                     trace=trace, converged_reason=reason)
    meta = dict(T_src=T_src, J=J, fps_src=fps_src, fps_out=fps_out, c=c, s_end=s_end,
                phase=phase, v_star=float(v_star), v_src_clean=v_src_clean, dpds=dpds,
                sdot_star=sdot_star, vel_cap=vel_cap, acc_budget=acc_budget,
                base_info=base_info, half=half, dose_accept=dose_accept,
                dose_target=dose_target, src_duration=src_duration)
    return best["out"], res, law, meta


def _trace_row(it: int, reading: OracleReading, warp: WarpResult,
               src_duration: float, breakdown: dict) -> dict:
    return dict(iter=it, verdict=reading.verdict,
                cop_dose=round(reading.doses.get("cop", 0.0), 4),
                fric_dose=round(reading.doses.get("friction", 0.0), 4),
                tau_dose=round(reading.doses.get("torque", 0.0), 4),
                T_out=warp.T_out, duration_s=round(warp.duration_s, 4),
                duration_x=round(warp.duration_s / src_duration, 3),
                **breakdown)


# ====================================================================================== #
# report                                                                                  #
# ====================================================================================== #
def build_report(data: dict, out: dict, res: ToppResult, law: "v1.TimeLaw", meta: dict,
                 body_mode: str) -> dict:
    T_src, fps_src, c = meta["T_src"], meta["fps_src"], meta["c"]
    fps_out = meta["fps_out"]
    warp, field = res.warp, res.stretch_field
    k_star, T_out = warp.k_star, warp.T_out

    # fidelity on the OUTPUT arrays
    blade_out = v1.blade_positions(out)
    v_out_clean = v1.clean_speed_at(blade_out, k_star, 1.0 / fps_out)
    n_src = v1.blade_face_normals(data)[c]
    n_out = v1.blade_face_normals(out)[k_star]
    face_deg = float(np.degrees(np.arccos(np.clip(
        np.dot(n_src, n_out) / (np.linalg.norm(n_src) * np.linalg.norm(n_out)), -1, 1))))
    contact_bitwise = bool(np.array_equal(out["joint_pos"][k_star],
                                          np.asarray(data["joint_pos"])[c]))
    dq_out = np.asarray(out["joint_vel"], float)
    first_frame_vel = float(np.abs(dq_out[0]).max())
    mean_acc_out = float(np.abs(np.diff(dq_out, axis=0) * fps_out).mean())
    dq_src = np.asarray(data["joint_vel"], float)
    mean_acc_src = float(np.abs(np.diff(dq_src, axis=0) * fps_src).mean())
    speed_dev = abs(v_out_clean - meta["v_star"]) / meta["v_star"]

    r0, rf = res.reading0, res.reading
    # per-segment stretch (ρ mean over pre-approach / lock window / follow-through)
    sg, rho = field.s_grid, field.rho
    half = meta["half"]
    pre = sg < (c - half)
    win = np.abs(sg - c) <= half
    post = sg > (c + half)
    seg = dict(
        pre_approach=dict(rho_mean=round(float(rho[pre].mean()) if pre.any() else 1.0, 3),
                          rho_max=round(float(rho[pre].max()) if pre.any() else 1.0, 3)),
        lock_window=dict(rho_mean=round(float(rho[win].mean()) if win.any() else 1.0, 3)),
        follow_through=dict(rho_mean=round(float(rho[post].mean()) if post.any() else 1.0, 3),
                            rho_max=round(float(rho[post].max()) if post.any() else 1.0, 3)),
        rho_peak=round(float(rho.max()), 3),
        rho_peak_at_frame=round(float(sg[int(np.argmax(rho))]), 2),
    )

    residual_in_window = bool(rf.doses.get("cop", 0) > V4RG_DOSE
                              and res.converged_reason == "no_out_of_window_flags")

    report = dict(
        tool="synthesize_timing_v2.py (oracle-guided TOPP-lite time-law synthesis)",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        verdict=rf.verdict,
        converged_reason=res.converged_reason,
        iterations=res.iters,
        acceptance=dict(
            cop_dose_final=round(rf.doses.get("cop", 0.0), 4),
            cop_dose_baseline=round(r0.doses.get("cop", 0.0), 4),
            beats_v5syn=bool(rf.doses.get("cop", 1.0) < meta["dose_accept"]),
            meets_v4rg=bool(rf.doses.get("cop", 1.0) <= V4RG_DOSE),
            dose_accept_threshold=meta["dose_accept"],
            v4rg_threshold=V4RG_DOSE,
            loop_stop_target=meta["dose_target"],
            residual_infeasibility_window_locked=residual_in_window,
        ),
        oracle_before=dict(verdict=r0.verdict, cop_dose=round(r0.doses.get("cop", 0.0), 4),
                           fric_dose=round(r0.doses.get("friction", 0.0), 4),
                           tau_dose=round(r0.doses.get("torque", 0.0), 4)),
        oracle_after=dict(verdict=rf.verdict, cop_dose=round(rf.doses.get("cop", 0.0), 4),
                          fric_dose=round(rf.doses.get("friction", 0.0), 4),
                          tau_dose=round(rf.doses.get("torque", 0.0), 4)),
        source=dict(frames=int(T_src), fps=fps_src, contact_frame=int(c),
                    phase=float(meta["phase"]), runup_s=round(c / fps_src, 4),
                    duration_s=round((T_src - 1) / fps_src, 4),
                    clean_blade_speed_mps=round(meta["v_src_clean"], 4),
                    mean_abs_acc=round(mean_acc_src, 3)),
        answer=dict(v_star_mps=round(float(meta["v_star"]), 4),
                    v_star_source="source-clean" if abs(meta["v_star"] - meta["v_src_clean"]) < 1e-12 else "cli-override",
                    blade_dpds_m_per_frame=round(meta["dpds"], 6),
                    sdot_star_frames_per_s=round(meta["sdot_star"], 4)),
        baseline_law=dict(Ta_s=round(law.Ta, 4), ta_max_s=round(meta["base_info"]["ta_max_s"], 4),
                          irreducible_kinematic=bool(meta["base_info"]["irreducible"])),
        stretch=seg,
        output=dict(frames=int(T_out), fps=float(fps_out), contact_frame=int(k_star),
                    phase_out=round(k_star / (T_out - 1), 6),
                    runup_s=round(k_star / fps_out, 4),
                    duration_s=round((T_out - 1) / fps_out, 4),
                    runup_change_x=round((k_star / fps_out) / (c / fps_src), 3),
                    duration_change_x=round(((T_out - 1) / fps_out) / ((T_src - 1) / fps_src), 3),
                    wait_s=round(warp.wait_s, 4), body_mode=body_mode,
                    mean_abs_acc=round(mean_acc_out, 3)),
        fidelity=dict(contact_row_bitwise=contact_bitwise,
                      blade_speed_clean_out_mps=round(v_out_clean, 4),
                      blade_speed_dev_frac=round(speed_dev, 5),
                      face_normal_diff_deg=round(face_deg, 6),
                      first_frame_max_joint_vel=round(first_frame_vel, 4)),
        trace=res.trace,
    )
    return report


def report_md(rep: dict) -> str:
    a, s, o, f, st = (rep["acceptance"], rep["source"], rep["output"],
                      rep["fidelity"], rep["stretch"])
    ob, oa = rep["oracle_before"], rep["oracle_after"]
    lines = [
        f"# Oracle-guided TOPP-lite time-law synthesis — **{rep['verdict']}** "
        f"(converged: {rep['converged_reason']}, {rep['iterations']} iters)",
        "",
        f"- generated: {rep['generated_utc']}",
        f"- **ACCEPTANCE ①**: CoP dose {a['cop_dose_baseline']:.3f} (baseline/v5syn-eq) -> "
        f"**{a['cop_dose_final']:.3f}** (final); beats v5syn {a['dose_accept_threshold']}: "
        f"**{a['beats_v5syn']}**; meets v4rg {a['v4rg_threshold']}: {a['meets_v4rg']} "
        f"(loop stop target {a['loop_stop_target']})",
        f"- oracle before: {ob['verdict']} (CoP {ob['cop_dose']:.3f} / fric {ob['fric_dose']:.3f} "
        f"/ τ {ob['tau_dose']:.3f})  ->  after: {oa['verdict']} (CoP {oa['cop_dose']:.3f} / "
        f"fric {oa['fric_dose']:.3f} / τ {oa['tau_dose']:.3f})",
        f"- source: {s['frames']} frames @ {s['fps']:.0f} fps = {s['duration_s']:.2f} s; contact "
        f"f{s['contact_frame']} (phase {s['phase']}); clean blade speed {s['clean_blade_speed_mps']:.3f} m/s",
        f"- output: {o['frames']} frames @ {o['fps']:.0f} fps = {o['duration_s']:.2f} s "
        f"(x{o['duration_change_x']:.2f} vs source); contact f{o['contact_frame']} -> "
        f"**phase_out {o['phase_out']:.4f}**; run-up x{o['runup_change_x']:.2f}; "
        f"wait {o['wait_s']:.3f} s; body_mode={o['body_mode']}; mean|acc| {s['mean_abs_acc']:.2f} -> {o['mean_abs_acc']:.2f}",
        f"- stretch ρ: pre-approach mean {st['pre_approach']['rho_mean']:.2f} "
        f"(max {st['pre_approach']['rho_max']:.2f}) | lock-window mean {st['lock_window']['rho_mean']:.2f} "
        f"| follow-through mean {st['follow_through']['rho_mean']:.2f} "
        f"(max {st['follow_through']['rho_max']:.2f}) | peak {st['rho_peak']:.2f} @ s={st['rho_peak_at_frame']:.1f}",
        f"- fidelity: contact row bitwise={f['contact_row_bitwise']}; blade speed out "
        f"{f['blade_speed_clean_out_mps']:.3f} m/s (dev {f['blade_speed_dev_frac'] * 100:.2f}%); "
        f"face diff {f['face_normal_diff_deg']:.4f} deg; frame-0 max|q̇| {f['first_frame_max_joint_vel']:.3f} rad/s",
    ]
    if a["residual_infeasibility_window_locked"]:
        lines.append(
            "- ⚠ **RESIDUAL INFEASIBILITY IS WINDOW-LOCKED**: TOPP-lite ran out of out-of-window "
            "frames to slow; the remaining infeasible frames sit INSIDE the ±0.1 s contact lock "
            "(geometry, not timing) — points to PATH MORPH (change |v*| or the spatial path), not "
            "a time law. Reported as-is, not forced to PASS.")
    lines += ["", "| iter | verdict | CoP dose | fric dose | τ dose | T_out | dur x | bad(out/in win) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rep["trace"]:
        lines.append(
            f"| {r['iter']} | {r['verdict']} | {r['cop_dose']:.3f} | {r['fric_dose']:.3f} | "
            f"{r['tau_dose']:.3f} | {r['T_out']} | {r['duration_x']:.2f} | "
            f"{r.get('bad_out_window', '-')}/{r.get('bad_in_window', '-')} |")
    lines += ["",
              f"REGISTRY REMINDER: SYNTHESIZED timeline — register phase_out = {o['phase_out']:.4f} "
              f"(contact frame {o['contact_frame']} of {o['frames']}) in cfg/strike_annotations.yaml. "
              f"Video frame convention does not apply (视频约定帧弃用)."]
    return "\n".join(lines)


# ====================================================================================== #
# CLI                                                                                     #
# ====================================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--phase", type=float, required=True,
                    help="registry contact phase of the SOURCE clip")
    ap.add_argument("--strike-speed", type=float, default=None,
                    help="answer blade speed |v*| m/s (default: source clean speed)")
    ap.add_argument("--budget-clips", nargs="+", required=True,
                    help="npz clips for the baseline-law |acc| budget (e.g. the v4rg pair)")
    ap.add_argument("--budget-scale", type=float, default=1.5)
    ap.add_argument("--vel-limit-frac", type=float, default=0.85)
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--mjcf", default=None, help="deploy/oracle MJCF (free-root); required")
    ap.add_argument("--body-order", default=None, help="body-order file; required for fk + oracle")
    ap.add_argument("--fps-out", type=float, default=None)
    ap.add_argument("--min-cruise-s", type=float, default=0.04)
    ap.add_argument("--post-contact-hold-s", type=float, default=0.04)
    ap.add_argument("--ta-grid-s", type=float, default=0.005)
    ap.add_argument("--dense-dt-s", type=float, default=0.002)
    ap.add_argument("--body-mode", choices=("fk", "interp"), default="fk",
                    help="fk = MuJoCo FK rebuild (PRODUCTION); interp = tests only")
    ap.add_argument("--mu", type=float, default=0.8, help="oracle friction coefficient")
    ap.add_argument("--support-band", type=float, default=0.03)
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    ap.add_argument("--dose-accept", type=float, default=DOSE_ACCEPT)
    ap.add_argument("--dose-target", type=float, default=DOSE_TARGET)
    ap.add_argument("--oracle-workdir", default=None,
                    help="dir for per-iteration temp npz (default: a fresh tmpdir)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)

    if args.urdf is None:
        args.urdf = os.path.normpath(os.path.join(
            _HERE, "../../..", "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"))
    from audit_motion_npz import parse_urdf_limits  # via v1's sys.path insert
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
        fk_ctx = (fkm, [names.index(n) for n in order])

    judge = RealOracle(args.mjcf, args.body_order, args.mu, args.support_band,
                       workdir=args.oracle_workdir)

    data = dict(np.load(args.input))
    stem = Path(args.input).name[:-4] if args.input.endswith(".npz") else Path(args.input).stem
    out, res, law, meta = topp_lite(
        data, args.phase, vlim, acc_budget, judge, stem,
        v_star=args.strike_speed, vel_limit_frac=args.vel_limit_frac, fps_out=args.fps_out,
        min_cruise_s=args.min_cruise_s, post_hold_s=args.post_contact_hold_s,
        ta_grid_s=args.ta_grid_s, dense_dt_s=args.dense_dt_s,
        body_mode=args.body_mode, fk_ctx=fk_ctx, max_iters=args.max_iters,
        dose_accept=args.dose_accept, dose_target=args.dose_target)

    rep = build_report(data, out, res, law, meta, args.body_mode)
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
        print(f"** WARNING: contact blade-speed deviation {dev * 100:.2f}% > 2% **", file=sys.stderr)
        return 1
    if not rep["acceptance"]["beats_v5syn"]:
        print(f"** WARNING: CoP dose {rep['acceptance']['cop_dose_final']:.3f} does NOT beat "
              f"v5syn {args.dose_accept} **", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
