#!/usr/bin/env python3
"""轴 A:把「击球速度 × r」重定时烤成独立动作资产(TOPP 语义的速度轴 bake)。

人话:给一条已登记 strike_phase 的参考动作(如 v4rg 正/反手),把触球瞬间的拍速
从原生 v0 改成 r×v0,其余什么都不改——空间路径、触球 pose(逐位)、拍面朝向(signed
face)、帧数、fps、触球帧号/相位全部保持,只重解时间律(哪一帧走到路径的哪个位置)。
输出一个可直接替换原资产的 npz + 一份单层内容 SHA 的 manifest JSON。

时间律(为什么这么解):
    输出帧格与源同长(T 帧、同 fps),warp s(k) 单调、s(c)=c(触球帧钉死)。
    触球锁窗(±0.1 s,≥ 干净拍速 stencil)内路径速率恒为 ρ,ρ 用二分法解到
    「输出资产上按训练口径(clean stencil ±2 帧)量出来的拍速 = r×v0」;
    锁窗外用二次时间律把亏/盈的路径长度匀回去(速率连续,r=1 时恒等,逐字节不变)。
    帧数不变 ⇒ 总时长不变、相位不变 ⇒ 各 r 变体对训练配置是 drop-in。

可行性判定(与 topp_mintime v3 的运动学硬边界同口径,audit 口径):
    逐关节 |q̇| ≤ URDF 速度限位 × --vel-limit-frac(先例 0.85);
    逐关节 |Δq̇|·fps ≤ 实证 |acc| 包络(--budget-clips,先例 v4rg 一对)× --budget-scale(1.5)。
    任何一帧任何关节越界 = physical-infeasible:manifest 里写明白是谁越界,
    并拒绝输出 npz(退出码 3)。禁止静默降速——目标拍速解不到就是不可行,不打折交货。
    (CoP/摩擦/τ 的 mj_inverse oracle 门不在本工具内;那是 pod 产线 topp_mintime 的活。)

--envelope 模式:先在梯子(默认 0.65/0.80/0.90/1.00/1.10/1.20/1.35)上逐点判卷,
    再从 r=1 向上扫 + 二分,给出该动作自身的 max 可行 r,全部写进 manifest。

分桶约定(Franco 拍板,写死在每份 manifest 里):
    train 0.80/1.00/1.20;interpolation 0.90/1.10;OOD 0.65/1.35。

减速档不再出资产(2026-07-27):r < 1 的档由**运行时慢放**交付(R14 分数时钟按
    speed_scale 播放:位置逐帧不动、qd×s、qdd×s²),bake 模式默认拒绝 r<1,理由和
    授权办法见拒绝信;真要历史对照用 --force-slowdown-bake。r ≥ 1 仍必须烤——
    时钟只往下走,round() 在 s>1 会跳过资产帧(可能正好跳掉触球帧)。
    每档交付方式由 canonical_playback_speed_gate.py 的证书判定,写进 --envelope
    manifest 的 delivery_plan(无证书 = 一律未授权,fail closed)。

USAGE(测试解释器即可,fk 需 mujoco;interp 只给 link_origin 测试 clip 用)
    python bake_topp_strike_speed.py \
        --motion hope_forehand_v4rg_cal.npz --phase 0.471 --speed-ratio 1.2 \
        --budget-clips hope_forehand_v4rg_cal.npz hope_backhand_v4rg_cal.npz \
        --out fh_speed1p20.npz --manifest fh_speed1p20.json
    python bake_topp_strike_speed.py --envelope \
        --motion hope_backhand_v4rg_cal.npz --phase 0.338 \
        --budget-clips ... --manifest bh_envelope.json

退出码:0 成功;1 合同/参数错误(fail loud);3 physical-infeasible(manifest 已写,资产拒发)。
带 canonical-v2 sidecar 的输入默认拒收;显式 comparator flag 只允许生成永久标记的历史对照。
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import synthesize_timing as v1     # noqa: E402  路径/拍面/干净拍速/重采样口径
import synthesize_timing_v2 as v2  # noqa: E402  resample_at_s + 锁窗常数
import topp_mintime as v3          # noqa: E402  内容 SHA / O_EXCL 写盘 / 回读校验
import canonical_playback_speed_gate as pbg  # noqa: E402  慢放门 + 离线交付计划

ISAAC_JOINT_NAMES = v1.ISAAC_JOINT_NAMES

# ---- Franco 拍板的速度轴分桶(写死;所有 manifest 原样携带) ------------------------- #
SPEED_RATIO_BUCKETS = {
    "train": (0.80, 1.00, 1.20),
    "interpolation": (0.90, 1.10),
    "OOD": (0.65, 1.35),
}
DEFAULT_LADDER = (0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35)

DEFAULT_WINDOW_S = v2.STRIKE_HALF_S   # 0.10 s:锁窗半宽 = 登记口径 phase±0.1s
DEFAULT_SPEED_TOL = 0.02              # 实际 vs 目标拍速的验收误差(manifest 断言)
DEFAULT_VEL_LIMIT_FRAC = 0.85         # URDF 速度限位余量(v1/v2/v3 先例)
DEFAULT_BUDGET_SCALE = 1.5            # |acc| 包络放大倍数(先例)
MIN_EDGE_RATE = 0.05                  # warp 段端速率下限(帧/帧;再小=近停滞,判不可行)
SOLVER_REL_TOL = 1e-4                 # ρ 二分收敛的相对拍速残差
SOLVER_MAX_ITERS = 60
FACE_DOT_GATE = 0.99                  # 全程拍面法向 vs 源(防镜像/翻面;同路径应≈1)
EXIT_INFEASIBLE = 3

#: r < 1 的减速档不再烤成独立资产:运行时 R14 分数时钟(mdp/commands.py)按
#: speed_scale 均匀慢放,位置逐帧不动、qd×s、qdd×s²,与烤入的减速片段等价而且免费。
#: 由 canonical_playback_speed_gate.py 验过才算数;越界照样是不可行,不打折交货。
SLOWDOWN_BAKE_EPS = 1e-9


def bucket_of(ratio: float) -> str:
    """r 属于哪个桶(train/interpolation/OOD);不在约定点上 = custom。"""
    for name, values in SPEED_RATIO_BUCKETS.items():
        if any(abs(float(ratio) - v) < 1e-9 for v in values):
            return name
    return "custom"


# ====================================================================================== #
# 输入校验(fail closed)                                                                  #
# ====================================================================================== #
def validate_source(data: dict) -> tuple[int, int, float]:
    """已知键、有限值、31 关节——不合格直接拒绝,不猜。返回 (T, J, fps)。"""
    unknown = [k for k in data.keys() if k not in v1.KNOWN_KEYS and not k.startswith("_")]
    if unknown:
        raise SystemExit(f"unknown npz keys {unknown} — refusing to guess how to retime them")
    for key in ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"):
        if key not in data:
            raise SystemExit(f"source npz missing required key {key!r}")
    for key, arr in data.items():
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise SystemExit(f"source npz key {key!r} contains non-finite values — refusing")
    q = np.asarray(data["joint_pos"])
    if q.ndim != 2 or q.shape[1] != len(ISAAC_JOINT_NAMES):
        raise SystemExit(f"joint_pos shape {q.shape} != (T, {len(ISAAC_JOINT_NAMES)})")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if not (np.isfinite(fps) and fps > 0):
        raise SystemExit(f"bad fps {fps}")
    return int(q.shape[0]), int(q.shape[1]), fps


# ====================================================================================== #
# 时间律:锁窗恒速 ρ + 窗外二次匀衡(帧数/触球帧不变)                                     #
# ====================================================================================== #
@dataclass
class SpeedWarp:
    ok: bool
    why: str                      # "" 可行;否则人话原因
    s: Optional[np.ndarray] = None   # (T,) 单调 warp;s[c]=c 逐位钉死
    m0: float = float("nan")      # 起始帧速率(帧/帧;r=1 时 =1)
    m1: float = float("nan")      # 末帧速率
    rho: float = float("nan")     # 锁窗内恒定路径速率
    w: int = 0                    # 锁窗半宽(帧)


def build_speed_warp(T: int, c: int, rho: float, w: int) -> SpeedWarp:
    """输出帧 k∈[0,T-1] 的路径位置 s(k):窗内 s=c+ρ(k−c);窗外二次律把剩余路径匀回端点。

    段端速率 m0=2μ_pre−ρ、m1=2μ_post−ρ(μ=段平均速率)保证速率连续;m 太小或窗把
    助跑/收拍整段吞掉 = 该 r 在此路径上物理不可行(时间重排救不了),fail closed。"""
    rho = float(rho)
    a = c - w                     # 助跑段帧数
    b = (T - 1) - (c + w)         # 收拍段帧数
    if a < 1 or b < 1:
        return SpeedWarp(False, f"lock window ±{w} frames leaves no run-up/follow-through "
                                f"(contact {c} of {T})")
    sa = c - rho * w              # 窗起点的路径位置
    sb = c + rho * w              # 窗终点的路径位置
    if sa < 1.0 or sb > T - 2.0:
        return SpeedWarp(False, f"window at rate rho={rho:.3f} consumes the whole "
                                f"run-up/follow-through path (sa={sa:.2f}, sb={sb:.2f})")
    m0 = 2.0 * (sa / a) - rho
    m1 = 2.0 * (((T - 1) - sb) / b) - rho
    if m0 < MIN_EDGE_RATE or m1 < MIN_EDGE_RATE:
        return SpeedWarp(False, f"warp would stall/reverse outside the window "
                                f"(edge rates m0={m0:.3f}, m1={m1:.3f} < {MIN_EDGE_RATE})")
    k = np.arange(T, dtype=np.float64)
    s = np.empty(T, dtype=np.float64)
    pre = k <= a
    s[pre] = m0 * k[pre] + (rho - m0) * k[pre] ** 2 / (2.0 * a)
    win = (k > a) & (k < c + w)
    s[win] = c + rho * (k[win] - c)
    post = k >= c + w
    t = k[post] - (c + w)
    s[post] = sb + rho * t + (m1 - rho) * t ** 2 / (2.0 * b)
    # 钉死不动点(浮点余差清零):端点 + 触球帧(逐位拷贝的前提)
    s[0], s[T - 1], s[c] = 0.0, float(T - 1), float(c)
    if not (np.diff(s) > 0).all():
        return SpeedWarp(False, "warp lost monotonicity — refusing")
    return SpeedWarp(True, "", s=s, m0=float(m0), m1=float(m1), rho=rho, w=w)


# ---- ρ 求解:让「输出资产上量出的干净拍速」= r×v0(不许静默降速) --------------------- #
def _blade_at(data: dict, s: np.ndarray) -> np.ndarray:
    """按输出资产同一套插值(pos 线性 + quat slerp + 拍面偏置)取分数帧的拍心位置。"""
    P = np.asarray(data["body_pos_w"], dtype=np.float64)[:, v1.RACKET_BODY]
    Q = np.asarray(data["body_quat_w"], dtype=np.float64)[:, v1.RACKET_BODY]
    pos = v1._interp_rows(P, s)
    quat = v1._slerp_rows(Q, s)
    return pos + np.einsum("tij,j->ti", v1.quat_to_rot(quat), v1.MOUNT_OFFSET)


def _proxy_clean_speed(data: dict, c: int, rho: float, fps: float) -> float:
    """clean stencil(±2 输出帧)在窗内恒速 ρ 下的拍速:‖blade(c+2ρ)−blade(c−2ρ)‖·fps/4。"""
    W = v1.CLEAN_VEL_WINDOW
    pts = _blade_at(data, np.array([c - W * rho, c + W * rho], dtype=np.float64))
    return float(np.linalg.norm(pts[1] - pts[0]) * fps / (2.0 * W))


def solve_window_rate(data: dict, T: int, c: int, fps: float, target_mps: float,
                      w: int) -> tuple[Optional[float], int, str]:
    """二分解锁窗速率 ρ:proxy 拍速命中 target(相对残差 ≤ 1e-4)。
    目标在单调可行的 ρ 区间内够不到 ⇒ 返回 (None, iters, 人话原因) = physical-infeasible。"""
    a, b = c - w, (T - 1) - (c + w)
    if a < 1 or b < 1:
        return None, 0, f"lock window ±{w} frames leaves no run-up/follow-through"
    rho_hi = min((c - 1.0) / w, (T - 2.0 - c) / w,
                 (2.0 * c - MIN_EDGE_RATE * a) / (a + 2.0 * w),
                 (2.0 * ((T - 1) - c) - MIN_EDGE_RATE * b) / (b + 2.0 * w))
    rho_lo = 1e-3
    if rho_hi <= rho_lo:
        return None, 0, "no monotone-feasible window rate exists for this contact placement"
    f_hi = _proxy_clean_speed(data, c, rho_hi, fps) - target_mps
    if f_hi < 0:
        return None, 0, (f"target strike speed {target_mps:.3f} m/s unreachable within the "
                         f"monotone warp (max reachable ≈ {f_hi + target_mps:.3f} m/s at "
                         f"rho={rho_hi:.3f}); refusing to under-deliver speed")
    f_lo = _proxy_clean_speed(data, c, rho_lo, fps) - target_mps
    if f_lo > 0:
        return None, 0, "target strike speed below the reachable minimum — bad ratio?"
    lo, hi = rho_lo, rho_hi
    # 先试解析种子 ρ=target/v0 对应的名义速率(≈r):r=1 时 proxy 恰=v0 → 原样返回 1.0,
    # 保证恒等 bake 逐字节不变;其余情况种子只是把括号收紧一半。
    seed = float(np.clip(target_mps / max(_proxy_clean_speed(data, c, 1.0, fps), 1e-12),
                         rho_lo, rho_hi))
    f_seed = _proxy_clean_speed(data, c, seed, fps) - target_mps
    if abs(f_seed) <= SOLVER_REL_TOL * target_mps:
        return seed, 1, ""
    if f_seed > 0:
        hi = seed
    else:
        lo = seed
    for it in range(1, SOLVER_MAX_ITERS + 1):
        mid = 0.5 * (lo + hi)
        f_mid = _proxy_clean_speed(data, c, mid, fps) - target_mps
        if abs(f_mid) <= SOLVER_REL_TOL * target_mps:
            return float(mid), it, ""
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi)), SOLVER_MAX_ITERS, ""


# ====================================================================================== #
# 运动学硬边界判卷(audit 口径,逐关节峰值 vs 限值)                                       #
# ====================================================================================== #
def kin_gate(out: dict, vlim: np.ndarray, acc_env: np.ndarray, vel_limit_frac: float,
             budget_scale: float, fps: float) -> dict:
    """输出资产的逐关节 |q̇| / |Δq̇|·fps 峰值 vs (URDF×余量) / (包络×倍数)。>1 = 越界。"""
    jv = np.asarray(out["joint_vel"], dtype=np.float64)
    vel_cap = np.asarray(vlim, dtype=np.float64) * vel_limit_frac
    acc_cap = np.asarray(acc_env, dtype=np.float64) * budget_scale
    if (acc_cap <= 0).any() or (vel_cap <= 0).any():
        raise SystemExit("velocity/acceleration caps must be strictly positive (floor the "
                         "acc envelope before calling)")
    vel_peak = np.abs(jv).max(axis=0)
    acc = np.abs(np.diff(jv, axis=0)) * fps
    acc_peak = acc.max(axis=0) if acc.shape[0] else np.zeros(jv.shape[1])
    vel_util = vel_peak / vel_cap
    acc_util = acc_peak / acc_cap
    offending = [
        dict(joint=ISAAC_JOINT_NAMES[j],
             vel_util=round(float(vel_util[j]), 4), acc_util=round(float(acc_util[j]), 4))
        for j in range(len(ISAAC_JOINT_NAMES))
        if vel_util[j] > 1.0 or acc_util[j] > 1.0
    ]
    per_joint = [
        dict(joint=ISAAC_JOINT_NAMES[j],
             qdot_peak=round(float(vel_peak[j]), 4),
             qdot_cap=round(float(vel_cap[j]), 4),
             qdot_urdf_limit=round(float(vlim[j]), 4),
             vel_util=round(float(vel_util[j]), 4),
             qddot_peak=round(float(acc_peak[j]), 4),
             qddot_cap=round(float(acc_cap[j]), 4),
             acc_util=round(float(acc_util[j]), 4))
        for j in range(len(ISAAC_JOINT_NAMES))
    ]
    return dict(feasible=len(offending) == 0,
                vel_util_max=round(float(vel_util.max()), 4),
                acc_util_max=round(float(acc_util.max()), 4),
                offending=offending, per_joint=per_joint,
                vel_limit_frac=vel_limit_frac, budget_scale=budget_scale)


# ====================================================================================== #
# bake 核心(纯函数,CLI 之外可单测)                                                      #
# ====================================================================================== #
@dataclass
class BakeResult:
    feasible: bool
    reasons: List[str]            # 空 = 可行;否则每条一个人话原因
    ratio: float
    v0_mps: float
    target_mps: float
    actual_mps: Optional[float] = None
    error_frac: Optional[float] = None
    rho: Optional[float] = None
    solver_iters: int = 0
    warp: Optional[SpeedWarp] = None
    out: Optional[dict] = None    # 判卷用;不可行时 CLI 绝不落盘
    kin: dict = dc_field(default_factory=dict)
    contact: dict = dc_field(default_factory=dict)
    T: int = 0
    c: int = 0
    fps: float = 0.0
    phase: float = 0.0


def bake(data: dict, phase: float, ratio: float, vlim: np.ndarray, acc_env: np.ndarray,
         *, vel_limit_frac: float = DEFAULT_VEL_LIMIT_FRAC,
         budget_scale: float = DEFAULT_BUDGET_SCALE, window_s: float = DEFAULT_WINDOW_S,
         speed_tol: float = DEFAULT_SPEED_TOL, body_mode: str = "interp",
         fk_ctx=None) -> BakeResult:
    """把一条 clip 烤成 r×v0 拍速变体。可行 ⇒ result.out 是完整输出资产字典;
    不可行 ⇒ feasible=False + reasons(禁止静默降速,没有打折资产)。"""
    if not (np.isfinite(ratio) and ratio > 0):
        raise SystemExit(f"--speed-ratio must be a positive finite number, got {ratio}")
    if not (0.0 < phase < 1.0):
        raise SystemExit(f"--phase must be in (0,1), got {phase}")
    T, J, fps = validate_source(data)
    c = v1.contact_frame(phase, T)
    if not (0 < c < T - 1):
        raise SystemExit(f"contact frame {c} of {T} leaves no run-up or follow-through")
    w = max(v1.CLEAN_VEL_WINDOW, int(round(window_s * fps)))
    if c - w < 1 or (T - 1) - (c + w) < 1:
        raise SystemExit(f"contact frame {c} too close to the clip edge for a ±{w}-frame "
                         f"lock window (T={T}) — cannot hold the strike speed stencil")
    if body_mode == "interp":
        point = str(np.asarray(data.get("body_lin_vel_point", "")).reshape(-1)[0])
        if point != "link_origin":
            raise SystemExit(
                f"body_mode=interp would silently change body_lin_vel semantics "
                f"(source point {point!r}); use --body-mode fk for production assets")
    elif body_mode == "fk":
        if fk_ctx is None:
            raise SystemExit("body_mode=fk requires an FK context (mjcf + body order)")
    else:
        raise SystemExit(f"unknown --body-mode {body_mode!r}")

    blade_src = v1.blade_positions(data)
    v0 = v1.clean_speed_at(blade_src, c, 1.0 / fps)
    dpds = v1.blade_path_deriv_at(blade_src, c)
    if dpds <= 1e-9 or v0 <= 1e-9:
        raise SystemExit("blade path derivative ~0 at the contact frame — bad phase?")
    target = float(ratio) * v0

    rho, iters, why = solve_window_rate(data, T, c, fps, target, w)
    base = BakeResult(feasible=False, reasons=[], ratio=float(ratio), v0_mps=v0,
                      target_mps=target, solver_iters=iters, T=T, c=c, fps=fps,
                      phase=float(phase))
    if rho is None:
        base.reasons = [f"physical-infeasible: {why}"]
        return base

    warp = build_speed_warp(T, c, rho, w)
    if not warp.ok:
        base.reasons = [f"physical-infeasible: {warp.why}"]
        base.rho = rho
        return base

    out = v2.resample_at_s(data, warp.s, fps, body_mode, fk_ctx)

    # ---- 保真断言(工具自身不变量,破了 = bug,fail loud 而不是 infeasible) -------- #
    if not np.array_equal(out["joint_pos"][c], np.asarray(data["joint_pos"])[c]):
        raise SystemExit("contact row is no longer a bitwise source copy — internal bug")
    if out["joint_pos"].shape[0] != T:
        raise SystemExit("frame count changed — internal bug")

    blade_out = v1.blade_positions(out)
    actual = v1.clean_speed_at(blade_out, c, 1.0 / fps)
    error = abs(actual - target) / target
    if error > speed_tol:
        raise SystemExit(f"actual strike speed {actual:.4f} m/s deviates "
                         f"{error * 100:.2f}% from target {target:.4f} m/s "
                         f"(tol {speed_tol * 100:.0f}%) — refusing to publish")

    n_src = v1.blade_face_normals(data)
    n_out = v1.blade_face_normals(out)
    n_src_at = v1._interp_rows(n_src, warp.s)
    n_src_at /= np.linalg.norm(n_src_at, axis=1, keepdims=True)
    face_dots = np.einsum("ij,ij->i", n_out, n_src_at) / np.linalg.norm(n_out, axis=1)
    face_min_dot = float(face_dots.min())
    face_contact_deg = float(np.degrees(np.arccos(np.clip(
        np.dot(n_out[c], n_src[c])
        / (np.linalg.norm(n_out[c]) * np.linalg.norm(n_src[c])), -1.0, 1.0))))

    kin = kin_gate(out, vlim, acc_env, vel_limit_frac, budget_scale, fps)

    reasons: List[str] = []
    if not kin["feasible"]:
        names = ", ".join(o["joint"] for o in kin["offending"][:6])
        reasons.append(
            f"physical-infeasible: kinematic hard limits exceeded "
            f"(vel_util_max {kin['vel_util_max']}, acc_util_max {kin['acc_util_max']}; "
            f"offending: {names})")
    if face_min_dot < FACE_DOT_GATE:
        reasons.append(f"physical-infeasible: blade face normal drifted "
                       f"(min dot {face_min_dot:.4f} < {FACE_DOT_GATE})")

    for key, arr in out.items():
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise SystemExit(f"output key {key!r} contains non-finite values — refusing")

    base.feasible = len(reasons) == 0
    base.reasons = reasons
    base.actual_mps = float(actual)
    base.error_frac = float(error)
    base.rho = float(rho)
    base.warp = warp
    base.out = out                # 判卷证据保留;CLI 只在可行时落盘
    base.kin = kin
    base.contact = dict(frame=int(c), phase=float(c) / (T - 1),
                        registered_phase=float(phase), row_bitwise=True,
                        face_normal_dev_deg=round(face_contact_deg, 6),
                        face_min_dot_over_clip=round(face_min_dot, 6))
    return base


# ====================================================================================== #
# --envelope:梯子判卷 + 向上扫描/二分找 max 可行 r                                       #
# ====================================================================================== #
def _row(res: BakeResult) -> dict:
    return dict(ratio=round(res.ratio, 4), bucket=bucket_of(res.ratio),
                feasible=bool(res.feasible),
                reasons=list(res.reasons),
                target_mps=round(res.target_mps, 4),
                actual_mps=(round(res.actual_mps, 4) if res.actual_mps is not None else None),
                error_frac=(round(res.error_frac, 6) if res.error_frac is not None else None),
                window_rate_rho=(round(res.rho, 4) if res.rho is not None else None),
                vel_util_max=res.kin.get("vel_util_max"),
                acc_util_max=res.kin.get("acc_util_max"))


def envelope_scan(data: dict, phase: float, vlim: np.ndarray, acc_env: np.ndarray,
                  ladder=DEFAULT_LADDER, cap: float = 3.0, tol: float = 0.01,
                  **bake_kw) -> dict:
    """先把梯子逐点判卷,再从 r=1 乘法上探(×1.15)到第一个不可行点,几何二分收尾。
    返回 {ladder: [...], max_feasible_ratio, search_trace}。r=1 不可行 = 源资产
    连自己的包络都过不了,上游就是坏的,fail loud。"""
    probe = lambda r: bake(data, phase, r, vlim, acc_env, **bake_kw)  # noqa: E731
    ladder_rows = [_row(probe(float(r))) for r in ladder]

    trace: List[dict] = []
    r1 = probe(1.0)
    trace.append(_row(r1))
    if not r1.feasible:
        raise SystemExit("native ratio r=1.0 fails its own feasibility envelope — the "
                         "source asset itself is broken; not searching further:\n  "
                         + "\n  ".join(r1.reasons))
    good, bad = 1.0, None
    g = 1.15
    while g <= cap * (1.0 + 1e-9):
        res = probe(g)
        trace.append(_row(res))
        if res.feasible:
            good = g
            g *= 1.15
        else:
            bad = g
            break
    if bad is None:
        bad_note = f"search capped at {cap} (still feasible)"
        max_feasible = good if good < cap else cap
    else:
        bad_note = ""
        while bad - good > tol:
            mid = 0.5 * (good + bad)
            res = probe(mid)
            trace.append(_row(res))
            if res.feasible:
                good = mid
            else:
                bad = mid
        max_feasible = good
    return dict(ladder=ladder_rows, max_feasible_ratio=round(float(max_feasible), 4),
                search_cap=cap, search_tol=tol, search_note=bad_note,
                search_trace=trace)


# ====================================================================================== #
# manifest(单层内容 SHA;Franco 拍板不搞多层审批链)                                       #
# ====================================================================================== #
def _limits_block(urdf_path: str, budget_clip_paths: List[str], acc_env: np.ndarray,
                  env_floored: bool, vel_limit_frac: float, budget_scale: float) -> dict:
    return dict(
        urdf=v3._file_evidence(urdf_path, "A3 URDF"),
        vel_limit_frac=vel_limit_frac,
        budget_clips=[v3._file_evidence(p, f"budget clip {i}")
                      for i, p in enumerate(budget_clip_paths)],
        budget_scale=budget_scale,
        acc_envelope=[round(float(x), 4) for x in acc_env],
        acc_envelope_zero_floored=bool(env_floored),
    )


def _source_block(motion_path: str, res_like: dict) -> dict:
    return dict(v3._file_evidence(motion_path, "source motion"), **res_like)


def _base_manifest(mode: str, args, source: dict, limits: dict) -> dict:
    return dict(
        tool="bake_topp_strike_speed.py v1 (axis A: strike-speed retiming bake)",
        mode=mode,
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        speed_ratio_buckets={k: list(v) for k, v in SPEED_RATIO_BUCKETS.items()},
        source=source,
        limits=limits,
        body_mode=args.body_mode,
        window_s=args.window_s,
        speed_tol=args.speed_tol,
        tool_provenance=dict(
            bake_topp_strike_speed=v3._file_evidence(__file__, "bake tool"),
            synthesize_timing=v3._file_evidence(v1.__file__, "dep v1"),
            synthesize_timing_v2=v3._file_evidence(v2.__file__, "dep v2"),
            topp_mintime=v3._file_evidence(v3.__file__, "dep v3"),
        ),
    )


def _write_manifest(path: Path, manifest: dict) -> None:
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
               ).encode("utf-8")
    v3._write_exclusive(path, lambda stream: stream.write(payload))
    if json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise SystemExit("manifest 回读不一致 — refusing")


# ====================================================================================== #
# CLI                                                                                     #
# ====================================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", required=True, help="源 clip npz(31 关节 50Hz 语义)")
    ap.add_argument("--phase", type=float, required=True,
                    help="登记触球相位(strike_annotations 口径)")
    ap.add_argument("--speed-ratio", type=float, default=None,
                    help="r:目标拍速 = r×原生干净拍速 v0(bake 模式必填)")
    ap.add_argument("--out", default=None, help="输出 npz(bake 模式必填;拒绝覆盖)")
    ap.add_argument("--manifest", required=True, help="manifest JSON(拒绝覆盖)")
    ap.add_argument("--envelope", action="store_true",
                    help="包络模式:梯子判卷 + 找 max 可行 r,只写 manifest 不出资产")
    ap.add_argument("--ladder", default=",".join(str(x) for x in DEFAULT_LADDER),
                    help="包络模式的 r 梯子(逗号分隔)")
    ap.add_argument("--envelope-cap", type=float, default=3.0, help="上探 r 上限")
    ap.add_argument("--envelope-tol", type=float, default=0.01, help="max r 二分精度")
    ap.add_argument("--budget-clips", nargs="+", required=True,
                    help="|acc| 实证包络来源(先例:v4rg 正反手一对)")
    ap.add_argument("--budget-scale", type=float, default=DEFAULT_BUDGET_SCALE)
    ap.add_argument("--vel-limit-frac", type=float, default=DEFAULT_VEL_LIMIT_FRAC)
    ap.add_argument("--urdf", default=None, help="A3 URDF(默认 repo 内拷贝)")
    ap.add_argument("--mjcf", default=None, help="fk 模式的 MJCF(默认 repo 内 a3_pingpong)")
    ap.add_argument("--body-mode", choices=("fk", "interp"), default="fk",
                    help="fk = MuJoCo FK 重建 body_*(产线);interp 只给 link_origin 测试 clip")
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S,
                    help="触球锁窗半宽秒数(恒速段;默认 0.10 = 登记口径)")
    ap.add_argument("--speed-tol", type=float, default=DEFAULT_SPEED_TOL,
                    help="实际 vs 目标拍速验收误差(默认 2%%)")
    ap.add_argument("--playback-certificate", default=None,
                    help="canonical_playback_speed_gate.py 的证书 JSON;交付计划据此"
                         "判定哪几档由运行时慢放覆盖(缺席 = 一律未授权,fail closed)")
    ap.add_argument("--force-slowdown-bake", action="store_true",
                    help="强行烤 r<1 的减速档(历史对照/取证用;正常产线不需要——"
                         "运行时慢放就是同一件事,而且不新增资产与 SHA 登记行)")
    ap.add_argument(
        "--allow-canonical-v2-legacy-comparator",
        action="store_true",
        help=(
            "仅历史比较:允许 canonical-v2/comparator 输入进入恒速锁窗旧语义;"
            "manifest 强制标为 legacy_window_frozen_comparator_only"
        ),
    )
    args = ap.parse_args(argv)

    if args.envelope:
        if args.speed_ratio is not None or args.out is not None:
            raise SystemExit("--envelope 模式不接受 --speed-ratio/--out(只写 manifest)")
    else:
        if args.speed_ratio is None or args.out is None:
            raise SystemExit("bake 模式必须给 --speed-ratio 和 --out")
        if (float(args.speed_ratio) < 1.0 - SLOWDOWN_BAKE_EPS
                and not args.force_slowdown_bake):
            raise SystemExit(
                f"refusing to bake a slowdown variant (r={args.speed_ratio} < 1): the "
                f"runtime already plays the native clip back slowly (R14 fractional clock, "
                f"speed_scale={args.speed_ratio} gives exactly q(t)->q({args.speed_ratio}t), "
                f"qd*{args.speed_ratio}, qdd*{args.speed_ratio}^2), so this asset would be a "
                f"duplicate that still costs an npz, a content SHA, and one allow-list row in "
                f"every question-bank family.\n"
                f"  authorise the ratio instead:  python canonical_playback_speed_gate.py "
                f"--certify --clip {Path(args.motion).name} --ratio {args.speed_ratio} ...\n"
                f"  if that gate REFUSES the ratio, a bake does not rescue it either — a "
                f"retime keeps the same poses, so the s-independent gravity term c0(q) is "
                f"identical; the source clip is the problem.\n"
                f"  --force-slowdown-bake still produces it for historical comparators.")

    playback_certificate = None
    if args.playback_certificate is not None:
        playback_certificate = json.loads(
            Path(args.playback_certificate).expanduser().read_text(encoding="utf-8"))

    manifest_path = Path(args.manifest).expanduser().absolute()
    if manifest_path.exists() or manifest_path.is_symlink():
        raise SystemExit(f"拒绝覆盖已有 manifest: {manifest_path}")
    out_path = None
    if not args.envelope:
        out_path = Path(args.out).expanduser().absolute()
        if out_path.exists() or out_path.is_symlink():
            raise SystemExit(f"拒绝覆盖已有资产: {out_path}")
        if out_path == manifest_path:
            raise SystemExit("--out 和 --manifest 必须是两个不同路径")

    output_paths = [manifest_path]
    if out_path is not None:
        output_paths.append(out_path)
    v3._refuse_output_aliases_inputs(
        [args.motion, *args.budget_clips],
        output_paths,
    )
    legacy_comparator = v3._legacy_comparator_guard(
        [
            ("source motion", args.motion),
            *[
                (f"budget clip {index}", path)
                for index, path in enumerate(args.budget_clips)
            ],
        ],
        allow_canonical_v2_legacy_comparator=(
            args.allow_canonical_v2_legacy_comparator
        ),
    )

    if args.urdf is None:
        args.urdf = os.path.normpath(os.path.join(
            _HERE, "../../..", "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"))
    if args.mjcf is None:
        args.mjcf = os.path.normpath(os.path.join(
            _HERE, "../../..",
            "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
            "a3_pingpong/a3_pingpong.xml"))

    from audit_motion_npz import parse_urdf_limits  # via sys.path insert
    limits = parse_urdf_limits(args.urdf)
    vlim = np.array([limits[n].velocity if limits[n].velocity is not None else np.inf
                     for n in ISAAC_JOINT_NAMES])
    env = v1.acc_envelope(args.budget_clips)
    env_floored = bool((env <= 0).any())
    if env_floored:  # 预算 clip 没动过的关节:给最小正预算(v1 先例),而不是 0 除
        floor = env[env > 0].min() if (env > 0).any() else 1.0
        env = np.where(env <= 0, floor, env)

    data = dict(np.load(args.motion, allow_pickle=False))
    fk_ctx = None
    if args.body_mode == "fk":
        if "body_names" not in data:
            raise SystemExit("fk 模式需要源 npz 自带 body_names 元数据(v3 schema)")
        fkm = v1.ctn.MjFK(args.mjcf, ISAAC_JOINT_NAMES)
        names = fkm.body_names()
        order = [str(x) for x in np.asarray(data["body_names"]).tolist()]
        missing = [n for n in order if n not in names]
        if missing:
            raise SystemExit(f"MJCF 缺 body {missing} — 模型与资产不匹配")
        fk_ctx = (fkm, [names.index(n) for n in order], tuple(order))

    limits_block = _limits_block(args.urdf, args.budget_clips, env, env_floored,
                                 args.vel_limit_frac, args.budget_scale)
    bake_kw = dict(vel_limit_frac=args.vel_limit_frac, budget_scale=args.budget_scale,
                   window_s=args.window_s, speed_tol=args.speed_tol,
                   body_mode=args.body_mode, fk_ctx=fk_ctx)

    if args.envelope:
        envl = envelope_scan(data, args.phase, vlim, env,
                             ladder=[float(x) for x in args.ladder.split(",")],
                             cap=args.envelope_cap, tol=args.envelope_tol, **bake_kw)
        T, _, fps = validate_source(data)
        c = v1.contact_frame(args.phase, T)
        v0 = v1.clean_speed_at(v1.blade_positions(data), c, 1.0 / fps)
        src = _source_block(args.motion, dict(
            frames=T, fps=fps, phase=args.phase, contact_frame=c,
            native_clean_speed_mps=round(v0, 4)))
        manifest = _base_manifest("envelope", args, src, limits_block)
        v3._mark_legacy_comparator(manifest, legacy_comparator)
        manifest["envelope"] = envl
        plan = pbg.plan_offline_speed_ladder(
            [row["ratio"] for row in envl["ladder"]], playback_certificate)
        manifest["delivery_plan"] = plan
        _write_manifest(manifest_path, manifest)
        print(f"[envelope] {Path(args.motion).name}: max feasible r = "
              f"{envl['max_feasible_ratio']} (native v0 {v0:.3f} m/s)")
        for row in envl["ladder"]:
            print(f"  r={row['ratio']:<5} {row['bucket']:<13} "
                  f"{'FEASIBLE  ' if row['feasible'] else 'INFEASIBLE'} "
                  f"actual={row['actual_mps']} m/s vel_util={row['vel_util_max']} "
                  f"acc_util={row['acc_util_max']}")
        print(f"[delivery] bakes required {plan['bakes_required']} of "
              f"{plan['bakes_without_this_gate']} ladder variants; "
              f"{plan['bakes_saved']} covered by runtime slow playback")
        return 0

    res = bake(data, args.phase, args.speed_ratio, vlim, env, **bake_kw)
    src = _source_block(args.motion, dict(
        frames=res.T, fps=res.fps, phase=res.phase, contact_frame=res.c,
        native_clean_speed_mps=round(res.v0_mps, 4)))
    manifest = _base_manifest("bake", args, src, limits_block)
    v3._mark_legacy_comparator(manifest, legacy_comparator)
    manifest["speed"] = dict(
        ratio=res.ratio, bucket=bucket_of(res.ratio),
        target_mps=round(res.target_mps, 4),
        actual_mps=(round(res.actual_mps, 4) if res.actual_mps is not None else None),
        error_frac=(round(res.error_frac, 6) if res.error_frac is not None else None),
        tol=args.speed_tol,
        window_rate_rho=(round(res.rho, 4) if res.rho is not None else None),
        solver_iters=res.solver_iters)
    manifest["feasibility"] = dict(
        verdict=("feasible" if res.feasible else "physical-infeasible"),
        reasons=list(res.reasons),
        vel_util_max=res.kin.get("vel_util_max"),
        acc_util_max=res.kin.get("acc_util_max"),
        offending=res.kin.get("offending", []),
        per_joint=res.kin.get("per_joint", []))
    manifest["contact"] = res.contact
    manifest["delivery"] = dict(
        kind=("forced_slowdown_bake" if res.ratio < 1.0 - SLOWDOWN_BAKE_EPS else "bake"),
        runtime_playback_max_ratio=pbg.RUNTIME_PLAYBACK_MAX_RATIO,
        rationale=("r<1 baked under --force-slowdown-bake; the runtime clock already "
                   "delivers uniform slow playback, so this asset is a duplicate"
                   if res.ratio < 1.0 - SLOWDOWN_BAKE_EPS else
                   "r>=1: above native, the runtime clock cannot deliver it "
                   "(round() would skip asset frames), so a bake is required"))
    if res.warp is not None:
        manifest["warp"] = dict(frames=res.T, window_half_frames=res.warp.w,
                                pre_edge_rate=round(res.warp.m0, 4),
                                post_edge_rate=round(res.warp.m1, 4),
                                monotone=True)

    if not res.feasible:
        manifest["output"] = None
        _write_manifest(manifest_path, manifest)
        print(f"[REFUSED] r={res.ratio} on {Path(args.motion).name}: physical-infeasible; "
              f"manifest written, NO asset (禁止静默降速).")
        for reason in res.reasons:
            print(f"  - {reason}")
        sys.exit(EXIT_INFEASIBLE)

    # 断言后的资产落盘:O_EXCL + 回读校验 + 内容 SHA 进 manifest(单层)
    assert res.error_frac is not None and res.error_frac <= args.speed_tol
    v3._write_exclusive(out_path, lambda stream: np.savez(stream, **res.out))
    try:
        v3._validate_npz_roundtrip(out_path, res.out)
        manifest["output"] = dict(path=str(out_path), bytes=out_path.stat().st_size,
                                  sha256=v3._sha256_file(out_path),
                                  frames=res.T, fps=res.fps,
                                  contact_frame=res.c,
                                  phase=round(res.contact["phase"], 6))
        _write_manifest(manifest_path, manifest)
    except BaseException:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        raise
    print(f"[baked] r={res.ratio} ({bucket_of(res.ratio)}): target {res.target_mps:.3f} "
          f"actual {res.actual_mps:.3f} m/s (err {res.error_frac * 100:.3f}%), "
          f"vel_util {res.kin['vel_util_max']} acc_util {res.kin['acc_util_max']} -> "
          f"{out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
