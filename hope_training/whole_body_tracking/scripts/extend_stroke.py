#!/usr/bin/env python3
"""Backswing-deepening PATH MORPH (引拍加深手术) for motion .npz clips — extend_stroke.

人话:franco 07-09 行程洞察——触球拍速 |v*| 定死后,触球前拍面行程 L 越短,加速度下界
a_min = v*²/(2L) 越高;时间律(放慢)治不了这个下界(v5 反手 min-time ×1.95 铁底就是它),
**只有加行程能治**。加行程的精确形态 = 让引拍(counter-movement)更深:选定的关节提前动、
往"把拍面推离触球点"的方向多走一段,再平滑地回到原轨。本工具做这台手术。

Design doc: docs/TIMELINE.md 07-09 晚四(path morph 定形)/晚五(反手引拍撞躯干)/晚六
(纠错:借行程用手臂链不用腰,"重心不要变"立为设计法则)。姊妹工具 synthesize_timing.py
(时间律合成)与 scripts/feasibility_oracle.py(A 层逆动力学打分)——本工具是 PATH 侧的手术刀,
那两把是 TIME 侧的尺子和裁判。三者刻意分文件:path / time / feasibility 是三件事,不要合并。

WHAT IT DOES
  1. 读源 clip,按登记相位定触球帧 c = round(phase*(T-1));**锁窗** [c-lock, T-1] 逐位冻结。
     lock 默认 = CLEAN_VEL_WINDOW = 2 帧 = 训练平价 clean-FD 拍速模板的半宽
     (v* = ‖blade[c+2]-blade[c-2]‖·fps/4)。只锁 c 一行会让 blade[c-2] 动 ⇒ v* 漂移 ⇒
     a_min = v*²/(2L) 的分子被污染。锁 2 帧 ⇒ **触球行逐位 + |v*| 逐位 + 拍面法向逐位**
     全部精确不变,ΔL 成为唯一自变量(这是反面对照与剂量对账成立的前提)。
  2. 最深帧 d = argmax_{s≤c} ‖blade(s) − blade(c)‖;L_deep = d→c 的逐帧位移弧长和
     (stroke_ledger_0709 同口径)。
  3. **形变场** δq_j(s) = A · w_j · P(s),P = 双段 smoothstep 凸包:
         P(s0)=0, P(d)=1, P(s1)=0, P'(s0)=P'(d)=P'(s1)=0, P≡0 外侧
     s0 = 起手帧(默认 0,ready 位保住),s1 = c − lock。**C1 由构造保证**。
  4. **方向 + 分配** w ∝ ∂D/∂q_j |_{frame d},D = ‖blade(q_d) − p_contact‖(中心差分梯度,
     归一化到 max|w| = 1)。即"最速加深方向在允许关节子空间上的投影"——雅可比加权,谁把拍子
     搬得远谁多干活,自然就是"达到 ΔL 且姿态退化最小"的分配。
     (为何不用"远离触球姿态的符号法":v5 反手的 waist_pitch 被 retarget 钳死在硬限位上,
      q(d) ≈ q(c),符号法退化成 0/噪声;梯度法给出真实的加深方向。)
     **方向再线性化**(--refine-iters,默认 1):初始梯度取在源姿态上,对"局部杠杆≈0 但远场杠杆大"
     的关节(v5 反手的肘就是)会低配权重。故在形变后的最深姿态上重取梯度,仅当新方向能以**更小
     的幅度 A** 达到同一 ΔL 时才接受 ⇒ 单调改善,且产物仍是严格的 δq = A·w·P(s) 形式(C1 不变)。
     注意 D(q) 沿单关节并非处处单调(实测 waist_pitch 前弯先减后增,谷底在 −0.2 rad 附近),
     所以这是局部方法:它找的是"最小侵入的加深",不是全局最优加深。
  5. **限位 fail-loud**。逐关节可用余量(仅在 P(s)>0 的帧上,按该关节自己的符号方向)
         m_j = min_{s: P(s)>0}  margin_j(s) / P(s)        [rad]
     幅度天花板 A_max = min_j m_j / |w_j|。既有饱和 grandfather(源片 waist_pitch 已贴上限,
     不能因此判源片死),但**禁止 morph 把它推得更远**。三档 loud:
       · m_j ≈ 0 的关节 = blocked(该关节在加深方向上零余量)→ --on-blocked drop(默认,
         剔除该关节 + WARN + 报告记名)或 fail(硬退出)。单个钉死关节不再连坐整个关节集。
       · 目标 ΔL 在 A_max 内不可达 → SystemExit,报文给出可达上限与 binding(关节,帧)。
       · 出片后任何一列越过(grandfather 后的)限位 → SystemExit。--strict-limits 关掉 grandfather。
  6. **CoM 保险丝**(晚六法则"重心不要变"):逐帧全身质心与源片对账,max|Δcom_xy| 超 --com-eps
     记 WARN 并进报告——不阻断产物(反面对照就是要拿到"腰系加深把质心带走"的数字)。
  7. 输出重建:joint_pos 形变;body_* 用 MuJoCo FK 在部署 MJCF 上重算(**root 位姿逐帧冻结**
     = 骨盆/腿不动);joint_vel 重新差分(np.gradient,csv_to_npz 约定);body 速度
     np.gradient / SO3 中心差分。
  8. **变换后自动串**:--retime v1 → synthesize_timing.synthesize() 同预算/限位口径重解时间律;
     --oracle → feasibility_oracle 给源片/形变片/重解片三段打分。报告直接给出 L_deep、
     a_min = v*²/(2L)、以及 CoP 剂量 vs **0.0905 纯时间律地板**(v5topp bh)的对账行。

关节集(--joints:预设名或逗号分隔关节名 —— **逐关节指定原生支持**)
    waist     = waist_yaw, waist_roll, waist_pitch     ← v1 默认腰系(含俯仰)= **反面对照**
                晚六:腰俯仰/侧滚正是 oracle 实测 τ binding(roll 127-129% / pitch 130%),
                且会把质心/支撑几何带走 → 预期 oracle 恶化。
    armchain  = right_shoulder_yaw(肩内旋), right_shoulder_pitch(肩俯仰),
                right_elbow(肘), waist_yaw(腰偏航)     ← v2 先遣分配集(晚六 + ff4b4bd 再精化)
                臂质量小 ⇒ CoM 近似不动;腰 yaw 绕竖直轴、不对抗重力、从未进 oracle top-8。

--forbid-joints(默认 legs):请求关节落入禁用集即 fail-loud。快捷名 legs / waist_pitch_roll。
    v2 法则跑法:--joints armchain --forbid-joints legs,waist_pitch_roll
    (预设本就不含它们;显式写上是把"法则被强制过"钉进报告。)

USAGE (pod, mjeval venv: numpy + mujoco)
    python hope_training/whole_body_tracking/scripts/extend_stroke.py \
        --input  .../v5_height_fix/hope_backhand_v5hLs_cal.npz --phase 0.391 \
        --output .../hope_backhand_v5hLsA20_cal.npz \
        --joints armchain --forbid-joints legs,waist_pitch_roll --extend-frac 0.20 \
        --mjcf .../a3_pingpong.xml --body-order /workspace/franco/body_order_isaac.txt \
        --retime v1 --retimed-output .../hope_backhand_v5hLsA20syn_cal.npz \
        --budget-clips .../hope_forehand_v4rg_cal.npz .../hope_backhand_v4rg_cal.npz \
        --oracle --report a20.json --md a20.md

EXIT CODES  0 = ok | 1 = WARN(CoM 保险丝跳闸 / 关节 blocked / 源片既有饱和)
            SystemExit = fail-loud(越限 / 禁用集 / ΔL 不可达 / 引拍窗不存在)

DEPENDENCIES: numpy always; mujoco for the production FK (--mjcf) and --oracle.
Unit tests: tests/test_extend_stroke.py (pure CPU, synthetic clips + analytic stub FK).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv_to_npz_mujoco as ctn  # noqa: E402  (numpy-only at import time)
import synthesize_timing as st  # noqa: E402
from audit_motion_npz import ISAAC_JOINT_NAMES, parse_urdf_limits  # noqa: E402

KNOWN_KEYS = st.KNOWN_KEYS
LOCK_FRAMES = st.CLEAN_VEL_WINDOW      # 2 — clean-FD strike-speed stencil half-width
ROOT_BODY_COL = 0                      # body_* column of pelvis_link (root pose, frozen)
RACKET_BODY_NAME = "right_wrist_yaw_Link"
COM_EPS_M = 0.02                       # CoM displacement fuse (晚六 "重心不要变")
BLOCKED_EPS = 1e-6                     # per-joint headroom below this = blocked [rad]

LEG_JOINTS = tuple(n for n in ISAAC_JOINT_NAMES
                   if any(k in n for k in ("hip_", "knee_", "ankle_")))
WAIST_PITCH_ROLL = ("waist_pitch_joint", "waist_roll_joint")

PRESETS = {
    # v1 默认瓶颈关节集 = 腰系(含俯仰)。晚六后转性为反面对照。
    "waist": ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
    # v2 先遣:手臂链(肩内旋/肩俯仰/肘)+ 腰偏航。禁腰俯仰/侧滚 + 双腿。
    "armchain": ("right_shoulder_yaw_joint", "right_shoulder_pitch_joint",
                 "right_elbow_joint", "waist_yaw_joint"),
}
FORBID_ALIASES = {"legs": LEG_JOINTS, "waist_pitch_roll": WAIST_PITCH_ROLL}

# 纯时间律地板:v5topp bh 的 oracle CoP 剂量(stroke_ledger_0709 §四)。行程手术的判据线。
TIMELAW_COP_FLOOR = 0.0905


# ------------------------------------------------------------------ small math -- #
def smoothstep(x: np.ndarray) -> np.ndarray:
    """C1 Hermite step on [0,1]: f(0)=0, f(1)=1, f'(0)=f'(1)=0."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def bump_profile(T: int, s0: int, d: int, s1: int) -> np.ndarray:
    """C1 unit bump over frame index: P(s0)=0, P(d)=1, P(s1)=0, P≡0 outside [s0, s1].

    Two smoothstep branches meeting at the peak d; P' vanishes at s0, d and s1, so
    q(s) + A·w·P(s) is C1 wherever q is (velocity-continuous by construction).
    """
    if not (s0 < d < s1):
        raise SystemExit(
            f"morph window degenerate: need s0 < peak < s1, got ({s0}, {d}, {s1})")
    s = np.arange(T, dtype=np.float64)
    P = np.zeros(T, dtype=np.float64)
    up = (s >= s0) & (s <= d)
    dn = (s > d) & (s <= s1)
    P[up] = smoothstep((s[up] - s0) / (d - s0))
    P[dn] = smoothstep((s1 - s[dn]) / (s1 - d))
    return P


def deep_frame_and_L(blade: np.ndarray, c: int) -> tuple[int, float]:
    """账本口径: d = 触球前离触球点欧氏最远帧;L_deep = d→c 的逐帧位移弧长和."""
    dist = np.linalg.norm(blade[: c + 1] - blade[c], axis=1)
    d = int(np.argmax(dist))
    L = float(np.sum(np.linalg.norm(np.diff(blade[d : c + 1], axis=0), axis=1)))
    return d, L


def a_min_of(v_star: float, L: float) -> float:
    """franco 行程界: a ≥ v*² / (2L) [m/s²]."""
    return float(v_star) ** 2 / (2.0 * float(L))


# ------------------------------------------------------------------ joint sets --- #
def resolve_joint_set(spec: str) -> list[str]:
    if spec in PRESETS:
        return list(PRESETS[spec])
    names = [t.strip() for t in spec.split(",") if t.strip()]
    if not names:
        raise SystemExit("--joints is empty")
    unknown = [n for n in names if n not in ISAAC_JOINT_NAMES]
    if unknown:
        raise SystemExit(f"unknown joint(s) {unknown} — not in the 31 Isaac columns")
    return names


def resolve_forbid(spec: str) -> list[str]:
    if not spec or spec.lower() == "none":
        return []
    out: list[str] = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok in FORBID_ALIASES:
            out.extend(FORBID_ALIASES[tok])
        elif tok in ISAAC_JOINT_NAMES:
            out.append(tok)
        else:
            raise SystemExit(f"unknown --forbid-joints token {tok!r} "
                             f"(aliases: {sorted(FORBID_ALIASES)})")
    return sorted(set(out))


# -------------------------------------------------------------------- morph ------ #
@dataclass
class MorphPlan:
    cols: np.ndarray        # (n,) joint column indices
    names: list[str]
    weights: np.ndarray     # (n,) signed, max|w| = 1
    grad: np.ndarray        # (n,) ∂D/∂q_j at the deep frame [m/rad]
    P: np.ndarray           # (T,) C1 unit bump
    s0: int
    d: int
    s1: int
    amp: float = 0.0

    def apply(self, q: np.ndarray, amp: float | None = None) -> np.ndarray:
        A = self.amp if amp is None else amp
        out = np.array(q, dtype=np.float64, copy=True)
        out[:, self.cols] += A * self.P[:, None] * self.weights[None, :]
        return out


def deep_gradient(blade_fn, q: np.ndarray, d: int, c: int, cols: np.ndarray,
                  eps: float = 1e-4) -> np.ndarray:
    """∂‖blade(q_d) − blade(q_c)‖ / ∂q_j at the deep frame (central FD; contact pose fixed).

    Both poses are evaluated at their OWN frame so the frozen root trajectory is applied
    correctly (blade_fn takes an explicit frame index per row).
    """
    p_c = blade_fn(q[c][None, :], frames=np.array([c]))[0]
    g = np.zeros(len(cols), dtype=np.float64)
    for i, j in enumerate(cols):
        qp = np.array(q[d], copy=True); qp[j] += eps
        qm = np.array(q[d], copy=True); qm[j] -= eps
        pp, pm = blade_fn(np.stack([qp, qm]), frames=np.array([d, d]))
        g[i] = (np.linalg.norm(pp - p_c) - np.linalg.norm(pm - p_c)) / (2.0 * eps)
    return g


def joint_headroom(q: np.ndarray, cols: np.ndarray, w: np.ndarray, P: np.ndarray,
                   lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Per-joint peak-delta headroom m_j [rad] and the binding frame of each.

    m_j = min over frames with P(s) > 0 of margin_j(s) / P(s), margin taken on the SIDE
    the joint is being pushed (sign of w_j). A(|w_j|) ≤ m_j is the per-joint constraint.
    """
    m = np.zeros(len(cols))
    bind: list[int] = []
    act = np.flatnonzero(P > 1e-12)
    for i, j in enumerate(cols):
        if w[i] == 0.0:                      # no leverage: cannot bind the amplitude
            m[i] = np.inf
            bind.append(-1)
            continue
        margin = (hi[j] - q[act, j]) if w[i] > 0 else (q[act, j] - lo[j])
        ratio = np.maximum(margin, 0.0) / P[act]
        k = int(np.argmin(ratio))
        m[i] = float(ratio[k])
        bind.append(int(act[k]))
    return m, bind


def solve_amplitude(q: np.ndarray, plan: MorphPlan, blade_fn, c: int,
                    L_target: float, A_max: float, tol: float = 1e-4,
                    iters: int = 60) -> tuple[float, float, int]:
    """Bisect A ∈ [0, A_max] so L_deep(morphed) == L_target. Fail-loud when unreachable."""
    def L_of(A: float) -> tuple[float, int]:
        d, L = deep_frame_and_L(blade_fn(plan.apply(q, A)), c)
        return L, d

    L0, _ = L_of(0.0)
    L_hi, _ = L_of(A_max)
    if L_hi < L0:
        raise SystemExit(
            f"morph direction SHORTENS the path (L {L0:.4f} → {L_hi:.4f} m at A_max) — "
            f"梯度方向与加深方向反号,拒绝出片(检查 --joints / --peak-frame)")
    if L_hi < L_target:
        raise SystemExit(
            f"target L_deep {L_target:.4f} m unreachable inside the URDF limits: max "
            f"achievable {L_hi:.4f} m (+{(L_hi / L0 - 1) * 100:.1f}% vs source {L0:.4f} m) "
            f"at A_max={A_max:.4f} rad — 加行程被限位挡住,报告须如实记这条缺口")

    lo_a, hi_a, A = 0.0, A_max, A_max
    for _ in range(iters):
        A = 0.5 * (lo_a + hi_a)
        L, _ = L_of(A)
        if abs(L - L_target) <= tol * L_target:
            break
        if L < L_target:
            lo_a = A
        else:
            hi_a = A
    L, d = L_of(A)
    return float(A), float(L), int(d)


def morph(data: dict, phase: float, joint_names: list[str], extend_frac: float,
          blade_fn, limits: dict, lock_frames: int = LOCK_FRAMES, s0: int = 0,
          peak_frame: int | None = None, strict_limits: bool = False,
          alloc: str = "grad", on_blocked: str = "drop", refine_iters: int = 1):
    """Source npz dict -> (morphed joint_pos (T,J) float64, plan, info dict)."""
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    unknown = [k for k in data.keys() if k not in KNOWN_KEYS and not k.startswith("_")]
    if unknown:
        raise SystemExit(f"unknown npz keys {unknown} — refusing to guess how to morph them")
    T, J = q.shape
    if J != len(ISAAC_JOINT_NAMES):
        raise SystemExit(f"joint_pos has {J} columns, expected {len(ISAAC_JOINT_NAMES)}")
    c = st.contact_frame(phase, T)
    s1 = c - int(lock_frames)
    if not (0 <= s0 < s1 < c < T):
        raise SystemExit(f"bad windows: s0={s0}, s1={s1}, contact={c}, T={T} "
                         f"(lock={lock_frames}) — 锁窗吃掉了引拍窗")

    blade_src = blade_fn(q)
    d_src, L_src = deep_frame_and_L(blade_src, c)
    d = int(d_src if peak_frame is None else peak_frame)
    if not (s0 < d < s1):
        raise SystemExit(
            f"deep frame {d} not strictly inside the morph window ({s0}, {s1}) — 最深点落在"
            f"起手帧或锁窗内:该 clip 片内没有引拍窗(正手 d=0 是已知形态)。用 --peak-frame "
            f"显式指定峰位,或先给 clip 前面补帧(v1 不支持 → 工具缺口,须记报告)")

    cols = np.array([ISAAC_JOINT_NAMES.index(n) for n in joint_names])
    g = deep_gradient(blade_fn, q, d, c, cols)
    if not np.isfinite(g).all() or np.abs(g).max() < 1e-9:
        raise SystemExit(f"deepening gradient ≈ 0 for joints {joint_names} — "
                         f"这些关节在最深帧对拍面深度没有杠杆,换关节集")
    if alloc == "grad":
        w = g / np.abs(g).max()
    elif alloc == "equal":
        w = np.sign(g)
    else:
        raise SystemExit(f"unknown --alloc {alloc!r}")

    P = bump_profile(T, s0, d, s1)

    # effective limits: grandfather pre-existing saturation, forbid making it worse
    lo = np.array([limits[n].lower for n in ISAAC_JOINT_NAMES], dtype=np.float64)
    hi = np.array([limits[n].upper for n in ISAAC_JOINT_NAMES], dtype=np.float64)
    src_lo_viol = [ISAAC_JOINT_NAMES[j] for j in range(J) if q[:, j].min() < lo[j] - 1e-9]
    src_hi_viol = [ISAAC_JOINT_NAMES[j] for j in range(J) if q[:, j].max() > hi[j] + 1e-9]
    if strict_limits and (src_lo_viol or src_hi_viol):
        raise SystemExit(f"--strict-limits: SOURCE clip already violates URDF limits "
                         f"(lo: {src_lo_viol}; hi: {src_hi_viol})")
    if not strict_limits:
        lo = np.minimum(lo, q.min(axis=0))
        hi = np.maximum(hi, q.max(axis=0))

    m, bind = joint_headroom(q, cols, w, P, lo, hi)
    # keep the FULL diagnostic picture of every REQUESTED joint before anything is dropped:
    # a blocked joint's gradient sign is the evidence for "which way its free travel points".
    grad_all = {n: round(float(g[i]), 6) for i, n in enumerate(joint_names)}
    headroom_all = {n: round(float(m[i]), 6) for i, n in enumerate(joint_names)}
    bind_all = {n: int(bind[i]) for i, n in enumerate(joint_names)}
    blocked = [i for i in range(len(cols)) if m[i] <= BLOCKED_EPS]
    blocked_names = [joint_names[i] for i in blocked]
    if blocked:
        msg = (f"joints with ZERO deepening headroom (pinned at a URDF limit in the "
               f"morph direction): {[(joint_names[i], f'f{bind[i]}') for i in blocked]}")
        if on_blocked == "fail":
            raise SystemExit(f"--on-blocked fail: {msg}")
        print(f"** WARNING: {msg} — dropped from the allocation **", file=sys.stderr)
        keep = [i for i in range(len(cols)) if i not in blocked]
        if not keep:
            raise SystemExit(f"every requested joint is blocked at a limit: {msg}")
        cols, w, g, m = cols[keep], w[keep], g[keep], m[keep]
        bind = [bind[i] for i in keep]
        joint_names = [joint_names[i] for i in keep]
        w = w / np.abs(w).max()
        m, bind = joint_headroom(q, cols, w, P, lo, hi)

    L_target = L_src * (1.0 + float(extend_frac))

    def attempt(w_try: np.ndarray):
        """(plan, A_max, binding) for a candidate direction, or None if it cannot reach ΔL."""
        mm, bb = joint_headroom(q, cols, w_try, P, lo, hi)
        aw = np.abs(w_try)
        ratios = np.where(aw > 0.0, mm / np.where(aw > 0.0, aw, 1.0), np.inf)
        k = int(np.argmin(ratios))
        A_max_ = float(ratios[k])
        if A_max_ <= BLOCKED_EPS or not np.isfinite(A_max_):
            return None
        pl = MorphPlan(cols=cols, names=list(joint_names), weights=w_try, grad=g, P=P,
                       s0=s0, d=d, s1=s1)
        try:
            A_, L_, dout_ = solve_amplitude(q, pl, blade_fn, c, L_target, A_max_)
        except SystemExit:
            return None
        pl.amp = A_
        return pl, A_max_, (joint_names[k], int(bb[k])), L_, dout_

    first = attempt(w)
    if first is None:
        mm, bb = joint_headroom(q, cols, w, P, lo, hi)
        aw = np.abs(w)
        ratios = np.where(aw > 0.0, mm / np.where(aw > 0.0, aw, 1.0), np.inf)
        k = int(np.argmin(ratios))
        A_max_ = float(ratios[k])
        # re-raise the informative failure from solve_amplitude (or the headroom one)
        if A_max_ <= BLOCKED_EPS or not np.isfinite(A_max_):
            raise SystemExit(f"no amplitude headroom: {joint_names[k]} @ f{bb[k]} (fail-loud)")
        pl = MorphPlan(cols=cols, names=list(joint_names), weights=w, grad=g, P=P,
                       s0=s0, d=d, s1=s1)
        solve_amplitude(q, pl, blade_fn, c, L_target, A_max_)   # raises with the real message
        raise SystemExit("unreachable ΔL (unexpected)")         # pragma: no cover

    plan, A_max, (b_joint, b_frame), L_out, d_out = first
    # --- direction refinement: re-linearize ∂D/∂q at the MORPHED deep pose ----------------
    # The initial gradient is taken at the source pose; joints whose leverage grows with the
    # deformation (v5 反手的肘 = 局部梯度 ≈ 0,远场杠杆很大) are otherwise under-used. Accept a
    # re-linearized direction only when it reaches the SAME ΔL with a SMALLER amplitude
    # (= less posture degradation), so refinement is monotone-improving by construction.
    w0, A0 = plan.weights.copy(), plan.amp
    n_refine, cos_change = 0, 1.0
    for _ in range(int(refine_iters)):
        g_new = deep_gradient(blade_fn, plan.apply(q), d, c, cols)
        if not np.isfinite(g_new).all() or np.abs(g_new).max() < 1e-12:
            break
        w_new = np.sign(g_new) if alloc == "equal" else g_new / np.abs(g_new).max()
        cand = attempt(w_new)
        if cand is None or cand[0].amp >= plan.amp * (1.0 - 1e-6):
            break
        plan, A_max, (b_joint, b_frame), L_out, d_out = cand
        n_refine += 1
    denom = np.linalg.norm(w0) * np.linalg.norm(plan.weights)
    if denom > 0:
        cos_change = float(np.dot(w0, plan.weights) / denom)

    A = plan.amp
    w, g = plan.weights, plan.grad
    q_out = plan.apply(q)

    # hard invariants — fail-loud, never silent
    if not np.array_equal(q_out[s1:], q[s1:]):
        raise SystemExit("lock window not bitwise — morph profile leaked past s1")
    frozen = np.setdiff1d(np.arange(J), cols)
    if not np.array_equal(q_out[:, frozen], q[:, frozen]):
        raise SystemExit("non-selected joints moved — allocation leaked")
    over_hi = [(ISAAC_JOINT_NAMES[j], float(q_out[:, j].max()), float(hi[j]))
               for j in range(J) if q_out[:, j].max() > hi[j] + 1e-7]
    over_lo = [(ISAAC_JOINT_NAMES[j], float(q_out[:, j].min()), float(lo[j]))
               for j in range(J) if q_out[:, j].min() < lo[j] - 1e-7]
    if over_hi or over_lo:
        raise SystemExit(f"URDF position limit violated by the morph — hi: {over_hi}; lo: {over_lo}")

    info = dict(
        contact_frame=c, s0=s0, s1=s1, lock_frames=int(lock_frames),
        deep_frame_src=int(d_src), deep_frame_out=int(d_out), peak_frame=int(d),
        L_deep_src_fk=float(L_src), L_deep_out_fk=float(L_out), L_target=float(L_target),
        extend_frac_req=float(extend_frac), extend_frac_out=float(L_out / L_src - 1.0),
        amp_rad=float(A), amp_max_rad=float(A_max),
        amp_binding_joint=b_joint, amp_binding_frame=b_frame,
        amp_headroom_frac=float(A / A_max),
        refine=dict(iters_used=int(n_refine), amp_initial_rad=float(A0),
                    amp_final_rad=float(A), cos_initial_final=round(float(cos_change), 6)),
        alloc=alloc, joints_used=list(joint_names), joints_blocked=blocked_names,
        grad_all_m_per_rad=grad_all, headroom_all_rad=headroom_all, binding_frame_all=bind_all,
        grad_m_per_rad={n: round(float(g[i]), 6) for i, n in enumerate(joint_names)},
        weights={n: round(float(w[i]), 6) for i, n in enumerate(joint_names)},
        headroom_rad={n: round(float(m[i]), 6) for i, n in enumerate(joint_names)},
        peak_delta_rad={n: round(float(A * w[i]), 6) for i, n in enumerate(joint_names)},
        src_saturated_hi=src_hi_viol, src_saturated_lo=src_lo_viol,
        frozen_joints_bitwise=True,
    )
    return q_out, plan, info, blade_src


# --------------------------------------------------------------- production FK --- #
class MjMorphFK:
    """float64 blade FK + whole-body CoM on the deploy MJCF; root pose frozen per frame."""

    def __init__(self, mjcf_path: str, joint_names: list[str]):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        if self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise SystemExit("MJCF root joint is not FREE — need a floating-base model")
        self.root_qadr = int(self.model.jnt_qposadr[0])
        adr = []
        for n in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid < 0:
                raise SystemExit(f"joint {n!r} not in MJCF")
            adr.append(int(self.model.jnt_qposadr[jid]))
        self.qadr = np.array(adr)
        self.racket_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            RACKET_BODY_NAME)
        if self.racket_bid < 0:
            raise SystemExit(f"body {RACKET_BODY_NAME!r} not in MJCF")
        self.root_pos = self.root_quat = None

    def bind_root(self, root_pos: np.ndarray, root_quat: np.ndarray) -> None:
        """Freeze the root trajectory the morph is evaluated against (pelvis/legs untouched)."""
        self.root_pos = np.asarray(root_pos, dtype=np.float64)
        self.root_quat = np.asarray(root_quat, dtype=np.float64)

    def _pose(self, t: int, q_row: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[self.root_qadr : self.root_qadr + 3] = self.root_pos[t]
        self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7] = self.root_quat[t]
        self.data.qpos[self.qadr] = q_row
        self.mujoco.mj_forward(self.model, self.data)

    def blade(self, q: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
        """Blade-center world positions (n,3) — mount FK, analyze_strike_phase convention."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        idx = np.arange(q.shape[0]) if frames is None else np.asarray(frames)
        out = np.zeros((q.shape[0], 3), dtype=np.float64)
        for k in range(q.shape[0]):
            self._pose(int(idx[k]), q[k])
            R = self.data.xmat[self.racket_bid].reshape(3, 3)
            out[k] = self.data.xpos[self.racket_bid] + R @ st.MOUNT_OFFSET
        return out

    def com(self, q: np.ndarray) -> np.ndarray:
        """Whole-body CoM (T,3) [m] (subtree_com of the world body)."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        out = np.zeros((q.shape[0], 3), dtype=np.float64)
        for t in range(q.shape[0]):
            self._pose(t, q[t])
            out[t] = self.data.subtree_com[0]
        return out


# ---------------------------------------------------------------- npz rebuild ---- #
def rebuild_npz(data: dict, q_out: np.ndarray, fk_ctx) -> dict:
    """Morphed joint_pos -> full npz dict (body_* via MuJoCo FK, root pose frozen)."""
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    jp = q_out.astype(np.float32)
    jv = np.gradient(jp.astype(np.float64), dt, axis=0).astype(np.float32)

    base_pos = np.asarray(data["body_pos_w"], dtype=np.float64)[:, ROOT_BODY_COL]
    base_quat = np.asarray(data["body_quat_w"], dtype=np.float64)[:, ROOT_BODY_COL]
    fkm, cols = fk_ctx
    pos_all, quat_all = ctn.fk_series(fkm, base_pos, base_quat,
                                      jp.astype(np.float64), ISAAC_JOINT_NAMES)
    bp = pos_all[:, cols].astype(np.float32)
    bq = quat_all[:, cols].astype(np.float32)
    bl = np.gradient(bp.astype(np.float64), dt, axis=0).astype(np.float32)
    ba = np.stack([ctn.so3_derivative(bq[:, b].astype(np.float64), dt)
                   for b in range(bq.shape[1])], axis=1).astype(np.float32)
    return {"fps": np.array([int(round(fps))], dtype=np.int64),
            "joint_pos": jp, "joint_vel": jv, "body_pos_w": bp, "body_quat_w": bq,
            "body_lin_vel_w": bl, "body_ang_vel_w": ba}


# ------------------------------------------------------------------- oracle ------ #
def load_oracle(repo_root: str):
    path = os.path.join(repo_root, "scripts", "feasibility_oracle.py")
    spec = importlib.util.spec_from_file_location("feasibility_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["feasibility_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


def score_with_oracle(npz_path: str, mjcf: str, body_order: str, annotations: str | None,
                      repo_root: str) -> dict:
    from pathlib import Path

    fo = load_oracle(repo_root)
    om = fo.load_oracle_model(mjcf)
    ann = fo.load_annotations(annotations)
    rep = fo.analyze_clip(om, Path(npz_path), ann, body_order, fo.DEFAULT_MU,
                          fo.DEFAULT_SUPPORT_BAND)
    return dict(verdict=rep.verdict, doses={k: float(v) for k, v in rep.doses.items()},
                peak_tau_util=float(rep.checks["torque"].peak),
                peak_tau_joint=rep.checks["torque"].detail,
                cop_max_out_cm=float(max(rep.checks["cop"].peak, 0.0) * 100.0),
                fric_peak=float(rep.checks["friction"].peak),
                min_fz_N=float(rep.checks["fz"].peak),
                top_joints=[{k: (float(v) if isinstance(v, (int, float)) else v)
                             for k, v in tj.items()} for tj in rep.top_joints[:5]])


# ------------------------------------------------------------------- report ------ #
def mean_abs_joint_acc(npz: dict) -> float:
    dq = np.asarray(npz["joint_vel"], dtype=np.float64)
    fps = float(np.asarray(npz["fps"]).reshape(-1)[0])
    return float(np.abs(np.diff(dq, axis=0) * fps).mean())


def build_report(data: dict, out: dict, info: dict, joints_req: list[str],
                 forbid: list[str], v_star: float, v_star_fk: float, v_star_out: float,
                 L_src_stored: float, L_out_npz: float, fk_vs_stored_blade_m: float,
                 com: dict | None, oracle_src: dict | None, oracle_out: dict | None,
                 retime: dict | None) -> dict:
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    c = info["contact_frame"]
    L0, L1 = info["L_deep_src_fk"], info["L_deep_out_fk"]
    a0, a1 = a_min_of(v_star, L0), a_min_of(v_star, L1)
    warn = bool((com and com["fuse_tripped"]) or info["joints_blocked"]
                or info["src_saturated_hi"] or info["src_saturated_lo"])
    jp_out, jp_src = np.asarray(out["joint_pos"]), np.asarray(data["joint_pos"])
    return dict(
        tool="extend_stroke.py (backswing-deepening path morph)",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        verdict="WARN" if warn else "ok",
        allocation=dict(joints_requested=joints_req, joints_used=info["joints_used"],
                        joints_blocked=info["joints_blocked"], forbidden=forbid,
                        alloc=info["alloc"], grad_m_per_rad=info["grad_m_per_rad"],
                        grad_all_m_per_rad=info["grad_all_m_per_rad"],
                        headroom_all_rad=info["headroom_all_rad"],
                        binding_frame_all=info["binding_frame_all"],
                        weights=info["weights"], headroom_rad=info["headroom_rad"],
                        peak_delta_rad=info["peak_delta_rad"], amp_rad=info["amp_rad"],
                        amp_max_rad=info["amp_max_rad"],
                        amp_headroom_frac=info["amp_headroom_frac"],
                        refine=info["refine"],
                        amp_binding=dict(joint=info["amp_binding_joint"],
                                         frame=info["amp_binding_frame"])),
        windows=dict(contact_frame=c, morph_start=info["s0"], morph_end=info["s1"],
                     lock_frames=info["lock_frames"], peak_frame=info["peak_frame"],
                     deep_frame_src=info["deep_frame_src"],
                     deep_frame_out=info["deep_frame_out"]),
        stroke=dict(v_star_mps=round(float(v_star), 4),
                    v_star_fk_mps=round(float(v_star_fk), 4),
                    v_star_out_mps=round(float(v_star_out), 4),
                    L_deep_src_m=round(L0, 4), L_deep_out_m=round(L1, 4),
                    L_deep_src_stored_m=round(float(L_src_stored), 4),
                    L_deep_out_npz_m=round(float(L_out_npz), 4),
                    extend_frac_req=info["extend_frac_req"],
                    extend_frac_out=round(info["extend_frac_out"], 4),
                    a_min_src=round(a0, 3), a_min_out=round(a1, 3),
                    a_min_drop_frac=round(1.0 - a1 / a0, 4)),
        fidelity=dict(
            contact_row_bitwise=bool(np.array_equal(jp_out[c], jp_src[c])),
            lock_window_bitwise=bool(np.array_equal(jp_out[info["s1"]:], jp_src[info["s1"]:])),
            frozen_joints_bitwise=bool(info["frozen_joints_bitwise"]),
            first_frame_bitwise=bool(np.array_equal(jp_out[0], jp_src[0])),
            # exact in real arithmetic (the ±2 stencil rows are bitwise); the residual is the
            # float32 quantization floor of body_pos_w (~1e-7 m -> ~2.5e-6 m/s), same floor
            # the v5hLs build acceptance quotes.
            v_star_preserved_exactly=bool(abs(v_star_out - v_star_fk) <= 1e-5 * v_star_fk),
            v_star_dev_vs_fk_mps=float(f"{abs(v_star_out - v_star_fk):.3e}"),
            v_star_dev_vs_stored_frac=round(float(abs(v_star_out - v_star) / v_star), 8),
            fk_vs_stored_blade_max_m=round(float(fk_vs_stored_blade_m), 6),
            first_frame_max_joint_vel=round(float(np.abs(np.asarray(out["joint_vel"])[0]).max()), 4),
            mean_abs_joint_acc_src=round(mean_abs_joint_acc(data), 3),
            mean_abs_joint_acc_out=round(mean_abs_joint_acc(out), 3),
        ),
        limits=dict(src_saturated_hi=info["src_saturated_hi"],
                    src_saturated_lo=info["src_saturated_lo"]),
        com=com, oracle_source=oracle_src, oracle_morphed=oracle_out, retime=retime,
        timelaw_cop_floor=TIMELAW_COP_FLOOR,
        source=dict(frames=int(jp_src.shape[0]), fps=fps, contact_frame=c,
                    runup_s=round(c / fps, 4)),
    )


def report_md(rep: dict) -> str:
    a, w, s, f = rep["allocation"], rep["windows"], rep["stroke"], rep["fidelity"]
    o, os_ = rep["oracle_morphed"], rep["oracle_source"]
    L = [
        f"# 引拍加深 path morph — **{rep['verdict']}**",
        "",
        f"- generated: {rep['generated_utc']}",
        f"- 关节集(请求): {', '.join(a['joints_requested'])}",
        f"- 关节集(实用): {', '.join(a['joints_used'])}"
        + (f"  |  **blocked(限位钉死,已剔除)**: " + ", ".join(
            f"{n}(∂D/∂q={a['grad_all_m_per_rad'][n]:+.4f} m/rad,加深方向余量 "
            f"{a['headroom_all_rad'][n]:.4f} rad @ f{a['binding_frame_all'][n]})"
            for n in a["joints_blocked"]) if a["joints_blocked"] else ""),
        f"- 禁用集: {', '.join(a['forbidden']) or '(none)'}  |  分配: {a['alloc']}",
        f"- 窗口: 起手 f{w['morph_start']} → 峰 f{w['peak_frame']} → 形变末 f{w['morph_end']} "
        f"→ **锁窗 f{w['morph_end']}..end 逐位冻结**(lock={w['lock_frames']} 帧;触球 f{w['contact_frame']})",
        f"- 幅度 A = {a['amp_rad']:.4f} rad(限位天花板 {a['amp_max_rad']:.4f},用掉 "
        f"{a['amp_headroom_frac'] * 100:.0f}%;binding = {a['amp_binding']['joint']} @ f{a['amp_binding']['frame']})",
        f"- 方向再线性化: {a['refine']['iters_used']} 轮,A {a['refine']['amp_initial_rad']:.4f} → "
        f"{a['refine']['amp_final_rad']:.4f} rad(cos⟨初,末⟩ = {a['refine']['cos_initial_final']:.4f})",
        f"- 逐关节峰值增量 [rad]: " + ", ".join(f"{k} {v:+.4f}" for k, v in a["peak_delta_rad"].items()),
        f"- **行程 L_deep {s['L_deep_src_m']:.4f} → {s['L_deep_out_m']:.4f} m "
        f"(+{s['extend_frac_out'] * 100:.1f}%)**;|v*| = {s['v_star_mps']:.3f} m/s(锁窗 ⇒ 逐位不变)"
        f" ⇒ **a_min {s['a_min_src']:.2f} → {s['a_min_out']:.2f} m/s² "
        f"(−{s['a_min_drop_frac'] * 100:.1f}%)**",
        f"- 保真: 触球行逐位={f['contact_row_bitwise']}; 锁窗逐位={f['lock_window_bitwise']}; "
        f"非选中关节逐位={f['frozen_joints_bitwise']}; 首帧逐位={f['first_frame_bitwise']}; "
        f"frame-0 max|q̇| {f['first_frame_max_joint_vel']:.3f} rad/s",
        f"- 关节 mean|acc| {f['mean_abs_joint_acc_src']:.2f} → {f['mean_abs_joint_acc_out']:.2f} rad/s²;"
        f" FK-vs-stored blade 最大偏差 {f['fk_vs_stored_blade_max_m'] * 1000:.3f} mm(口径对账)",
        f"- |v*| 对账: stored {s['v_star_mps']:.4f} / FK {s['v_star_fk_mps']:.4f} / morph 后 "
        f"{s['v_star_out_mps']:.4f} m/s(FK 口径逐位不变={f['v_star_preserved_exactly']}; "
        f"vs stored 偏差 {f['v_star_dev_vs_stored_frac'] * 100:.4f}%)",
    ]
    if rep["com"]:
        cm = rep["com"]
        L.append(f"- **CoM 保险丝**(晚六法则「重心不要变」): max|Δcom| xy "
                 f"{cm['max_dxy_m'] * 100:.2f} cm, z {cm['max_dz_m'] * 100:.2f} cm "
                 f"(ε={cm['eps_m'] * 100:.0f} cm) → {'**跳闸**' if cm['fuse_tripped'] else 'ok'}")
    if rep["limits"]["src_saturated_hi"] or rep["limits"]["src_saturated_lo"]:
        L.append(f"- 源片既有限位饱和(grandfather;morph 不得推得更远): "
                 f"hi={rep['limits']['src_saturated_hi']} lo={rep['limits']['src_saturated_lo']}")
    L += ["", "| 量 | 源 | morph 后 | Δ |", "|---|---|---|---|",
          f"| L_deep [m] | {s['L_deep_src_m']:.4f} | {s['L_deep_out_m']:.4f} "
          f"| +{s['extend_frac_out'] * 100:.1f}% |",
          f"| a_min = v*²/(2L) [m/s²] | {s['a_min_src']:.2f} | {s['a_min_out']:.2f} "
          f"| −{s['a_min_drop_frac'] * 100:.1f}% |"]
    if os_ and o:
        for k, lab in (("cop", "CoP 剂量"), ("torque", "τ 剂量"), ("friction", "摩擦剂量")):
            v0, v1 = os_["doses"].get(k, 0.0), o["doses"].get(k, 0.0)
            L.append(f"| {lab} | {v0:.4f} | {v1:.4f} | {v1 - v0:+.4f} |")
        L.append(f"| τ 峰 util | {os_['peak_tau_util'] * 100:.0f}% ({os_['peak_tau_joint']}) "
                 f"| {o['peak_tau_util'] * 100:.0f}% ({o['peak_tau_joint']}) | |")
        L.append(f"| CoP 最大外溢 [cm] | {os_['cop_max_out_cm']:.1f} | {o['cop_max_out_cm']:.1f} | |")
        L.append(f"| oracle 判决 | {os_['verdict']} | {o['verdict']} | |")
    L.append(f"| 关节 mean\\|acc\\| [rad/s²] | {f['mean_abs_joint_acc_src']:.2f} "
             f"| {f['mean_abs_joint_acc_out']:.2f} | |")
    if rep["retime"]:
        rt = rep["retime"]
        L += ["", f"**时间律重解**(synthesize_timing v1):T_a {rt['Ta_s']:.3f} s,片长 "
                  f"x{rt['duration_change_x']:.2f},phase_out {rt['phase_out']:.4f},"
                  f"拍速偏差 {rt['blade_speed_dev_frac'] * 100:.2f}%,verdict {rt['verdict']}"]
        if rt.get("oracle"):
            ro, floor = rt["oracle"], rep["timelaw_cop_floor"]
            d = ro["doses"]["cop"]
            L.append(f"- 重解后 oracle: **{ro['verdict']}** | CoP 剂量 **{d:.4f}** | τ 剂量 "
                     f"{ro['doses']['torque']:.4f} | 摩擦 {ro['doses']['friction']:.4f} "
                     f"| τ 峰 {ro['peak_tau_util'] * 100:.0f}% ({ro['peak_tau_joint']})")
            L.append(f"- **判据(晚四)**:重解后 CoP 剂量 {d:.4f} vs 纯时间律地板 {floor:.4f} → "
                     f"{'**破地板**' if d < floor else '未破地板'}")
    L += ["", f"REGISTRY REMINDER: 形变资产触球帧与源片同为 f{w['contact_frame']}(锁窗逐位),"
              f"phase 沿用源片登记值;若随后串了时间律,以 synthesize_timing 的 phase_out 为准。"]
    return "\n".join(L)


# -------------------------------------------------------------------------- CLI -- #
def repo_root_of(script: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(script)), "../../.."))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--phase", type=float, required=True,
                    help="registry contact phase of the SOURCE clip (strike_annotations.yaml)")
    ap.add_argument("--joints", default="armchain",
                    help="preset (waist|armchain) or comma-separated Isaac joint names")
    ap.add_argument("--forbid-joints", default="legs",
                    help="comma list / aliases (legs, waist_pitch_roll); fail-loud on overlap")
    ap.add_argument("--extend-frac", type=float, required=True,
                    help="target ΔL_deep as a fraction of the source L_deep (e.g. 0.20)")
    ap.add_argument("--alloc", choices=("grad", "equal"), default="grad")
    ap.add_argument("--on-blocked", choices=("drop", "fail"), default="drop",
                    help="joints with zero deepening headroom: drop (WARN) or fail-loud")
    ap.add_argument("--refine-iters", type=int, default=1,
                    help="re-linearize ∂D/∂q at the morphed deep pose; a candidate direction is "
                         "accepted only if it reaches the same ΔL with a smaller amplitude "
                         "(0 = pure local gradient)")
    ap.add_argument("--lock-frames", type=int, default=LOCK_FRAMES,
                    help="frames before contact frozen bitwise (>= clean-FD half-width 2)")
    ap.add_argument("--morph-start", type=int, default=0)
    ap.add_argument("--peak-frame", type=int, default=None,
                    help="override the bump peak (default: the source deep frame)")
    ap.add_argument("--strict-limits", action="store_true",
                    help="fail on ANY URDF limit violation, incl. pre-existing source saturation")
    ap.add_argument("--com-eps", type=float, default=COM_EPS_M,
                    help="CoM xy displacement fuse [m] (晚六 design law); WARN when exceeded")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--body-order", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--oracle", action="store_true",
                    help="score source + morphed (+ retimed) with feasibility_oracle")
    ap.add_argument("--retime", choices=("none", "v1"), default="none",
                    help="re-solve the time law on the morphed path (synthesize_timing v1)")
    ap.add_argument("--retimed-output", default=None)
    ap.add_argument("--budget-clips", nargs="+", default=None, help="required for --retime v1")
    ap.add_argument("--budget-scale", type=float, default=1.5)
    ap.add_argument("--vel-limit-frac", type=float, default=0.85)
    ap.add_argument("--report", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)

    root = repo_root_of(__file__)
    if args.urdf is None:
        args.urdf = os.path.join(root, "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf")
    if args.lock_frames < LOCK_FRAMES:
        raise SystemExit(f"--lock-frames {args.lock_frames} < clean-FD stencil half-width "
                         f"{LOCK_FRAMES}: |v*| would drift and a_min's numerator with it")

    joints = resolve_joint_set(args.joints)
    forbid = resolve_forbid(args.forbid_joints)
    clash = sorted(set(joints) & set(forbid))
    if clash:
        raise SystemExit(f"requested joints {clash} are in the forbidden set "
                         f"(晚六设计法则:禁腰俯仰/侧滚 + 双腿) — fail-loud")

    limits = parse_urdf_limits(args.urdf)
    data = dict(np.load(args.input))

    fk = MjMorphFK(args.mjcf, ISAAC_JOINT_NAMES)
    fk.bind_root(np.asarray(data["body_pos_w"], dtype=np.float64)[:, ROOT_BODY_COL],
                 np.asarray(data["body_quat_w"], dtype=np.float64)[:, ROOT_BODY_COL])

    q_out, plan, info, blade_src_fk = morph(
        data, args.phase, joints, args.extend_frac, fk.blade, limits,
        lock_frames=args.lock_frames, s0=args.morph_start, peak_frame=args.peak_frame,
        strict_limits=args.strict_limits, alloc=args.alloc, on_blocked=args.on_blocked,
        refine_iters=args.refine_iters)

    c = info["contact_frame"]
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    # |v*| and L on the STORED blade arrays = stroke_ledger parity; FK copy = cross-check
    blade_src_stored = st.blade_positions(data)
    v_star = st.clean_speed_at(blade_src_stored, c, 1.0 / fps)
    v_star_fk = st.clean_speed_at(blade_src_fk, c, 1.0 / fps)
    _, L_src_stored = deep_frame_and_L(blade_src_stored, c)
    fk_vs_stored = float(np.abs(blade_src_fk - blade_src_stored).max())

    com_src = fk.com(np.asarray(data["joint_pos"], dtype=np.float64))
    com_out = fk.com(q_out)
    dxy = float(np.linalg.norm(com_out[:, :2] - com_src[:, :2], axis=1).max())
    dz = float(np.abs(com_out[:, 2] - com_src[:, 2]).max())
    com = dict(max_dxy_m=dxy, max_dz_m=dz, eps_m=args.com_eps,
               fuse_tripped=bool(dxy > args.com_eps))

    fkm = ctn.MjFK(args.mjcf, ISAAC_JOINT_NAMES)
    names = fkm.body_names()
    order = [ln.strip() for ln in open(args.body_order) if ln.strip()]
    fk_ctx = (fkm, [names.index(n) for n in order])
    out = rebuild_npz(data, q_out, fk_ctx)
    np.savez(args.output, **out)
    blade_out_npz = st.blade_positions(out)
    _, L_out_npz = deep_frame_and_L(blade_out_npz, c)
    v_star_out = st.clean_speed_at(blade_out_npz, c, 1.0 / fps)

    oracle_src = oracle_out = None
    if args.oracle:
        oracle_src = score_with_oracle(args.input, args.mjcf, args.body_order,
                                       args.annotations, root)
        oracle_out = score_with_oracle(args.output, args.mjcf, args.body_order,
                                       args.annotations, root)

    retime = None
    if args.retime == "v1":
        if not args.budget_clips:
            raise SystemExit("--retime v1 requires --budget-clips")
        vlim = np.array([limits[n].velocity if limits[n].velocity is not None else np.inf
                         for n in ISAAC_JOINT_NAMES])
        budget = st.acc_envelope(args.budget_clips) * args.budget_scale
        rt_out, rt_rep = st.synthesize(out, args.phase, vlim, budget,
                                       vel_limit_frac=args.vel_limit_frac,
                                       body_mode="fk", fk_ctx=fk_ctx)
        rt_path = args.retimed_output or (os.path.splitext(args.output)[0] + "_syn.npz")
        np.savez(rt_path, **rt_out)
        retime = dict(path=os.path.abspath(rt_path), verdict=rt_rep["verdict"],
                      Ta_s=rt_rep["time_law"]["Ta_s"],
                      duration_change_x=rt_rep["output"]["duration_change_x"],
                      phase_out=rt_rep["output"]["phase_out"],
                      blade_speed_dev_frac=rt_rep["fidelity"]["blade_speed_dev_frac"],
                      mean_abs_acc=rt_rep["output"]["mean_abs_acc"])
        if args.oracle:
            retime["oracle"] = score_with_oracle(rt_path, args.mjcf, args.body_order,
                                                 args.annotations, root)

    rep = build_report(data, out, info, joints, forbid, v_star, v_star_fk, v_star_out,
                       L_src_stored, L_out_npz, fk_vs_stored, com,
                       oracle_src, oracle_out, retime)
    rep["files"] = dict(input=os.path.abspath(args.input), output=os.path.abspath(args.output),
                        urdf=os.path.abspath(args.urdf), mjcf=os.path.abspath(args.mjcf))
    md = report_md(rep)
    print(md)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(rep, fh, indent=2, default=float)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(md)
    return 1 if rep["verdict"] == "WARN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
