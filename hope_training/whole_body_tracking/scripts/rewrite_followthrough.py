#!/usr/bin/env python3
"""Follow-through PATH REWRITE (随挥段路径重写) for motion .npz clips — rewrite_followthrough.

人话:franco 07-10 凌晨定向——v5hLs 反手的 CoP 违规 38 帧里 28 帧(74%)住在触球锁窗+随挥段
(f21-f57)。随挥段没有球约束:球已经打出去了,那段轨迹长什么样是"刀的选择不是物理禁区"
(v1 引拍刀 extend_stroke 只动触球前,是它的选择)。旁证:时间律把随挥拉长到极限能把剂量
压到 0.038,但片长 ×5.4——说明随挥段的病是可治的,路径重写应该比纯慢放便宜。本工具做
这台手术:触球行及之前一根汗毛都不动,只重写触球之后的收拍路径,让 oracle 的 CoP 剂量
(摔跤预测主项)变小。

姊妹工具:extend_stroke.py(引拍刀,触球前)/ synthesize_timing(时间律)/
scripts/feasibility_oracle.py(裁判)/ audit_self_collision.py(C 卡)。本工具刻意复用
它们的原语(bump_profile、collect_hits、oracle 的逐帧动力学),不复制粘贴,防止口径漂移。

设计法则(franco 07-10 定向,不可违背)
  1. 锁死 [0, c+2] 逐位(c = round(phase·(T-1)),与 v1 引拍刀相反方向)。触球行、
     |v*| 的 clean-FD ±2 模板行、拍面法向全部精确不变。
  2. 重写域 = [c+3, end];末帧姿态回到源片末帧(收拍到同一 ready)。
  3. 自由度 = 手臂链(肩三轴/肘/腕酌情)+ 腰偏航;冻结腰俯仰/侧滚 + 双腿
     (oracle 实测 τ binding + 支撑几何不许动),CoM 位移 < 2 cm 保险丝。
  4. 接缝 C1:c+2 处与末帧处 q/q̇ 连续。做法:形变基元支撑端点取 s0=c+3、s1=T-2,
     端点值/斜率均为 0(双段 smoothstep 构造)——于是 c+2 的中心差分速度
     (q[c+3]-q[c+1])/2dt 和末帧的单侧差分 (q[E]-q[E-1])/dt 只读逐位行,q̇ 逐位保真。
  5. 逐帧 URDF 位置限位 fail-loud(源片既有饱和 grandfather,禁止推得更远)。
  6. **C 在环**:每个候选先过自碰撞(audit_self_collision.collect_hits 复用,
     只扫重写域;锁段行逐位 ⇒ 碰撞状态与源片相同),有任何互穿帧直接拒绝,再进 oracle。
  7. oracle 剂量是目标函数,CoP 主项。in-loop 打分器逐行照抄 analyze_clip 的动力学
     (mj_inverse + 支撑多边形 CoP),启动时与真 oracle 对账,数字对不上就拒绝开工。

算法(取舍讲清楚)
  参数化 = 形变场 + 收拍模板混合,二选一或混合:
    field  : δq_j(s) = Σ_k a_jk · P_k(s),P_k = 双段 smoothstep 凸包(峰位均匀铺在重写域
             内),端点 C1=0 由构造保证。参数量 = |关节| × K,每个候选全局合法。
    blend  : δq_j(s) = β · W(s) · (line_j(s) − q_j(s)),line = 从 q[c+2] 到 q[E] 的关节
             空间直线("最省事的收拍"先验),W = C1 平台窗(smoothstep 起落)。单旋钮。
    hybrid : β 和 a_jk 一起进坐标下降(默认)。先验给大步,场做微调。
  搜索 = oracle-在环贪心坐标下降:逐坐标试 ±step,守卫全过且剂量键严格变好才收,
  同方向连续延伸(最多 8 步);整轮无改善则步长减半,至下限收工。
    为什么不是轨迹优化/梯度法:oracle 不可导(剂量 = 离散计数 + mj_inverse),逐帧自由
    变量需要显式平滑约束和真 NLP;形变场把 C1/锁段/末帧全部变成构造性质,候选数几百个,
    每个 ~25ms,一分钟内收敛——和 v1 引拍刀验证过的配方同源。
    剂量键 = (CoP 剂量, CoP 正溢出面积, 摩擦剂量, τ 剂量) 字典序:剂量是台账货币
    (0709 回测摔跤预测),面积是连续 tie-breaker,防止搜索卡在离散台阶上。
  守卫(每候选,fail-closed,拒绝记名进报告):
    限位 → 自碰撞(C 在环)→ CoM 保险丝(xy < 2cm)→ 摩擦/τ 剂量不得劣于源片。

输出
  重写 npz:锁段与末帧行逐位 = 源片;重写域行由部署 MJCF FK 重算(root 逐帧冻结);
  joint_vel / body 速度按 csv_to_npz 约定重差分。报告 json+md:逐候选剂量三列
  (CoP/摩擦/τ)+ 自碰撞 + CoM + 接缝残差,源/终局剂量对账(in-loop 与文件复核双列)。

USAGE (pod, mjeval venv: numpy + mujoco)
    /workspace/hope_mjeval_venv/bin/python \
        hope_training/whole_body_tracking/scripts/rewrite_followthrough.py \
        --input  .../v5_height_fix/hope_backhand_v5hLs_cal.npz --phase 0.391 \
        --output .../hope_backhand_v5hLsFT_cal.npz \
        --joints arm5 --mode hybrid \
        --mjcf agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
        --body-order /workspace/franco/body_order_isaac.txt \
        --annotations hope_training/whole_body_tracking/cfg/strike_annotations.yaml \
        --report ft.json --md ft.md

EXIT CODES  0 = ok | 1 = WARN(无改善 / 文件复核剂量漂移 / 缝合口径偏差 / 腾空帧)
            SystemExit = fail-loud(锁段泄漏 / 限位 / 禁用关节 / 源片脏 / 打分器失配)

DEPENDENCIES: numpy always; mujoco for production (guards + oracle + FK). Unit tests are
pure CPU with stub guards/scorer: tests/test_rewrite_followthrough.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv_to_npz_mujoco as ctn  # noqa: E402  (numpy-only at import time)
import extend_stroke as es  # noqa: E402
import synthesize_timing as st  # noqa: E402
import audit_self_collision as asc  # noqa: E402  (mujoco import inside is guarded)
from audit_motion_npz import ISAAC_JOINT_NAMES, _ranges, parse_urdf_limits  # noqa: E402

KNOWN_KEYS = st.KNOWN_KEYS
LOCK_AFTER_CONTACT = 2          # lock [0, c+2]: contact row + clean-FD stencil rows c+1, c+2
COM_EPS_M = 0.02                # CoM xy fuse [m] (franco 07-10: 冻结双腿 + CoM<2cm 保险丝)
DEFAULT_BASIS_K = 4             # deformation bumps per joint
MIN_DOMAIN = 4                  # smallest usable support length s1 - s0 [frames]
FILE_DOSE_DRIFT_WARN = 0.02     # in-loop vs written-file CoP dose drift that trips WARN
STITCH_DEV_WARN_M = 0.002       # FK-vs-stored body row deviation at the seams that trips WARN

# 晚六/07-10 法则:腰俯仰/侧滚 + 双腿是 oracle 实测力矩 binding + 支撑几何,永久禁用。
FORBIDDEN_JOINTS = tuple(es.LEG_JOINTS) + tuple(es.WAIST_PITCH_ROLL)

PRESETS = {
    # 肩三轴 + 肘 + 腰偏航(默认;腕不动 = 拍面姿态少受干扰)
    "arm5": ("right_shoulder_pitch_joint", "right_shoulder_roll_joint",
             "right_shoulder_yaw_joint", "right_elbow_joint", "waist_yaw_joint"),
    # + 腕 roll/pitch("腕酌情":触球后拍面无球约束,允许参与卸力)
    "arm7": ("right_shoulder_pitch_joint", "right_shoulder_roll_joint",
             "right_shoulder_yaw_joint", "right_elbow_joint", "waist_yaw_joint",
             "right_wrist_roll_joint", "right_wrist_pitch_joint"),
}


# ------------------------------------------------------------------ pure geometry -- #
def rewrite_windows(T: int, c: int) -> tuple[int, int]:
    """(s0, s1) support endpoints of the deformation. Frames touched: (s0, s1) EXCLUSIVE.

    s0 = c + 3 keeps the central-difference velocity at c+2 bitwise (it reads c+1, c+3);
    s1 = T - 2 keeps the end pose AND its one-sided velocity (reads T-2, T-1) bitwise.
    """
    s0 = c + LOCK_AFTER_CONTACT + 1
    s1 = T - 2
    if s1 - s0 < MIN_DOMAIN:
        raise SystemExit(
            f"rewrite domain too short: support [{s0}, {s1}] has {max(s1 - s0, 0)} frames "
            f"(< {MIN_DOMAIN}) — 触球太靠片尾,随挥段没有可重写的肉 (T={T}, contact={c})")
    return s0, s1


def bump_basis(T: int, s0: int, s1: int, K: int) -> tuple[np.ndarray, list[int]]:
    """K C1 bumps (peaks evenly spread strictly inside (s0, s1)) -> basis (K_eff, T).

    Reuses extend_stroke.bump_profile: P(s0)=P(s1)=0 with zero slope, P≡0 outside.
    K is clamped to the number of distinct interior peaks (WARN, never silent)."""
    if K < 1:
        raise SystemExit(f"--basis-k must be >= 1, got {K}")
    peaks = np.unique(np.round(np.linspace(s0 + 1, s1 - 1, min(K, s1 - s0 - 1))).astype(int))
    if len(peaks) < K:
        print(f"** WARNING: basis K clamped {K} -> {len(peaks)} (domain has only "
              f"{s1 - s0 - 1} interior frames) **", file=sys.stderr)
    B = np.stack([es.bump_profile(T, s0, int(p), s1) for p in peaks])
    return B, [int(p) for p in peaks]


def plateau_window(T: int, s0: int, s1: int, ramp: int | None = None) -> np.ndarray:
    """C1 plateau on [s0, s1]: smoothstep up over `ramp` frames, hold 1, smoothstep down.

    W(s0)=W(s1)=0 with zero slope, W≡0 outside — the blend inherits seam C1 from it."""
    n = s1 - s0
    if n < MIN_DOMAIN:
        raise SystemExit(f"plateau window degenerate: s1-s0={n} < {MIN_DOMAIN}")
    r = max(2, n // 5) if ramp is None else int(ramp)
    r = min(r, n // 2)
    s = np.arange(T, dtype=np.float64)
    W = np.zeros(T, dtype=np.float64)
    up = (s >= s0) & (s < s0 + r)
    mid = (s >= s0 + r) & (s <= s1 - r)
    dn = (s > s1 - r) & (s <= s1)
    W[up] = es.smoothstep((s[up] - s0) / r)
    W[mid] = 1.0
    W[dn] = es.smoothstep((s1 - s[dn]) / r)
    return W


def retreat_template(q: np.ndarray, cols: np.ndarray, c: int, s0: int, s1: int,
                     W: np.ndarray) -> np.ndarray:
    """安全收拍模板方向场 V (T, n): W(s) · (joint-space line from q[c+2] to q[end] − q(s)).

    β·V pulls the follow-through toward the straightest legal retreat to the SAME ready
    pose; β=1 with W=1 makes the plateau frames exactly the straight line."""
    T = q.shape[0]
    a, E = c + LOCK_AFTER_CONTACT, T - 1
    if not (a < E):
        raise SystemExit(f"retreat template needs c+{LOCK_AFTER_CONTACT} < last frame")
    t = np.clip((np.arange(T, dtype=np.float64) - a) / (E - a), 0.0, 1.0)
    line = q[a, cols][None, :] * (1.0 - t)[:, None] + q[E, cols][None, :] * t[:, None]
    return W[:, None] * (line - q[:, cols])


# ------------------------------------------------------------------------- plan ---- #
@dataclass
class RewritePlan:
    cols: np.ndarray                 # (n,) joint column indices
    names: list[str]
    basis: np.ndarray                # (K, T) C1 bumps, zero outside (s0, s1)
    peaks: list[int]
    template: np.ndarray | None      # (T, n) blend direction, already windowed; None=field
    coef: np.ndarray                 # (n, K) bump coefficients [rad]
    beta: float                      # blend amount in [0, 1]
    s0: int
    s1: int
    c: int

    def delta(self) -> np.ndarray:
        d = (self.coef @ self.basis).T                       # (T, n)
        if self.template is not None and self.beta != 0.0:
            d = d + self.beta * self.template
        return d

    def apply(self, q: np.ndarray) -> np.ndarray:
        out = np.array(q, dtype=np.float64, copy=True)
        out[:, self.cols] += self.delta()
        return out


def assert_structure(q_src: np.ndarray, q_out: np.ndarray, cols: np.ndarray,
                     s0: int, s1: int) -> None:
    """Hard invariants every candidate must satisfy BY CONSTRUCTION — a failure is a bug."""
    if not np.array_equal(q_out[: s0 + 1], q_src[: s0 + 1]):
        raise SystemExit("REWRITE BUG: lock window / seam head not bitwise (basis leaked)")
    if not np.array_equal(q_out[s1:], q_src[s1:]):
        raise SystemExit("REWRITE BUG: tail seam / end pose not bitwise (basis leaked)")
    frozen = np.setdiff1d(np.arange(q_src.shape[1]), cols)
    if not np.array_equal(q_out[:, frozen], q_src[:, frozen]):
        raise SystemExit("REWRITE BUG: frozen joints moved (allocation leaked)")


def seam_residuals(q_src: np.ndarray, q_out: np.ndarray, c: int, fps: float) -> dict:
    """Measured (not assumed) C1 residuals at both seams. Exact zeros by construction."""
    a, E = c + LOCK_AFTER_CONTACT, q_src.shape[0] - 1
    vc = 0.5 * fps * ((q_out[a + 1] - q_out[a - 1]) - (q_src[a + 1] - q_src[a - 1]))
    ve = fps * ((q_out[E] - q_out[E - 1]) - (q_src[E] - q_src[E - 1]))
    return dict(
        lock_window_bitwise=bool(np.array_equal(q_out[: a + 1], q_src[: a + 1])),
        contact_row_bitwise=bool(np.array_equal(q_out[c], q_src[c])),
        end_pose_bitwise=bool(np.array_equal(q_out[E], q_src[E])),
        vel_residual_at_lock_end=float(np.abs(vc).max()),
        vel_residual_at_end=float(np.abs(ve).max()),
    )


# ------------------------------------------------------------------------ joints --- #
def resolve_joints(spec: str) -> list[str]:
    names = list(PRESETS[spec]) if spec in PRESETS else \
        [t.strip() for t in spec.split(",") if t.strip()]
    if not names:
        raise SystemExit("--joints is empty")
    unknown = [n for n in names if n not in ISAAC_JOINT_NAMES]
    if unknown:
        raise SystemExit(f"unknown joint(s) {unknown} — not in the 31 Isaac columns")
    clash = sorted(set(names) & set(FORBIDDEN_JOINTS))
    if clash:
        raise SystemExit(f"joints {clash} are PERMANENTLY forbidden here "
                         f"(法则:冻结腰俯仰/侧滚+双腿——τ binding + 支撑几何 + CoM)")
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def validate_npz(data: dict) -> tuple[np.ndarray, float]:
    unknown = [k for k in data.keys() if k not in KNOWN_KEYS and not k.startswith("_")]
    if unknown:
        raise SystemExit(f"unknown npz keys {unknown} — refusing to guess how to rewrite them")
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(ISAAC_JOINT_NAMES):
        raise SystemExit(f"joint_pos shape {q.shape}, expected (T, {len(ISAAC_JOINT_NAMES)})")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    return q, fps


def require_clean_source(selfcol_guard, q_src: np.ndarray) -> None:
    ok, info = selfcol_guard(q_src)
    if not ok:
        raise SystemExit(f"SOURCE clip already self-collides inside the rewrite domain "
                         f"({info}) — 拒绝在脏源上做随挥重写;先修源或换 clip")


# ------------------------------------------------------------------------- score --- #
@dataclass
class Score:
    dose_cop: float
    dose_fric: float
    dose_tau: float
    cop_area: float                  # Σ max(cop_excess, 0) over eval frames [m·frames]
    com_dxy: float = 0.0             # max CoM xy displacement vs source [m]
    com_dz: float = 0.0
    cop_frames: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def key(self) -> tuple:
        """Lexicographic objective: CoP dose (台账货币) > CoP area (连续 tie-break) > 摩擦 > τ."""
        return (round(self.dose_cop, 12), round(self.cop_area, 10),
                round(self.dose_fric, 12), round(self.dose_tau, 12))

    def as_row(self) -> dict:
        return dict(dose_cop=round(self.dose_cop, 6), dose_fric=round(self.dose_fric, 6),
                    dose_tau=round(self.dose_tau, 6), cop_area=round(self.cop_area, 6),
                    com_dxy_cm=round(self.com_dxy * 100.0, 3),
                    com_dz_cm=round(self.com_dz * 100.0, 3))


# ------------------------------------------------------------------------ search --- #
def coordinate_search(q_src: np.ndarray, plan: RewritePlan, scorer, pre_guards,
                      *, mode: str = "hybrid", com_eps: float = COM_EPS_M,
                      dose_slack: float = 0.0, coef_step: float = 0.08,
                      beta_step: float = 0.25, min_step: float = 0.005,
                      max_passes: int = 8, max_evals: int = 800, coef_cap: float = 1.2,
                      max_extend: int = 8, log: list | None = None):
    """Greedy coordinate descent, oracle-in-loop, guards fail-closed per candidate.

    Acceptance = ALL guards pass AND Score.key() strictly improves ⇒ the accepted-dose
    sequence is monotone by construction (验收项). Every evaluation is logged with its
    reject reason — nothing is silent. Returns (best_plan, src_score, best_score, stats)."""
    if log is None:
        log = []
    coords: list[tuple] = []
    if mode in ("blend", "hybrid"):
        if plan.template is None:
            raise SystemExit(f"--mode {mode} needs a retreat template (internal wiring bug)")
        coords.append(("beta",))
    if mode in ("field", "hybrid"):
        coords += [("coef", j, k) for j in range(len(plan.names))
                   for k in range(plan.basis.shape[0])]
    if not coords:
        raise SystemExit(f"unknown --mode {mode!r} (field|blend|hybrid)")

    def label(coord: tuple) -> str:
        if coord[0] == "beta":
            return "beta(收拍模板混合)"
        _, j, k = coord
        return f"{plan.names[j]}·P{k}@f{plan.peaks[k]}"

    src_score = scorer(q_src)
    best_plan, best = plan, src_score
    log.append(dict(idx=0, coord="baseline(源片)", value=0.0, accepted=True, reason="",
                    **src_score.as_row()))
    rejects: dict[str, int] = {}
    evals, passes_run, budget_out = 0, 0, False

    def stepped(pl: RewritePlan, coord: tuple, delta: float) -> RewritePlan | None:
        if coord[0] == "beta":
            nb = float(np.clip(pl.beta + delta, 0.0, 1.0))
            return None if nb == pl.beta else replace(pl, beta=nb, coef=pl.coef.copy())
        _, j, k = coord
        nv = float(np.clip(pl.coef[j, k] + delta, -coef_cap, coef_cap))
        if nv == pl.coef[j, k]:
            return None
        coef = pl.coef.copy()
        coef[j, k] = nv
        return replace(pl, coef=coef)

    def evaluate(cand: RewritePlan, coord: tuple, value: float) -> bool:
        nonlocal evals, best_plan, best, budget_out
        if evals >= max_evals:
            budget_out = True
            return False
        evals += 1
        q_out = cand.apply(q_src)
        assert_structure(q_src, q_out, plan.cols, plan.s0, plan.s1)
        row = dict(idx=evals, coord=label(coord), value=round(value, 6),
                   accepted=False, reason="")
        for gname, g in pre_guards:
            ok, info = g(q_out)
            if not ok:
                row["reason"] = f"{gname}:{info}"
                rejects[gname] = rejects.get(gname, 0) + 1
                log.append(row)
                return False
        sc = scorer(q_out)
        row.update(sc.as_row())
        reason = None
        if sc.com_dxy > com_eps:
            reason = f"com_fuse:Δxy {sc.com_dxy * 100:.2f}cm > {com_eps * 100:g}cm"
            rejects["com_fuse"] = rejects.get("com_fuse", 0) + 1
        elif (sc.dose_fric > src_score.dose_fric + dose_slack
              or sc.dose_tau > src_score.dose_tau + dose_slack):
            reason = (f"dose_guard:fric {sc.dose_fric:.4f}/τ {sc.dose_tau:.4f} 劣于源 "
                      f"{src_score.dose_fric:.4f}/{src_score.dose_tau:.4f}")
            rejects["dose_guard"] = rejects.get("dose_guard", 0) + 1
        elif not sc.key() < best.key():
            reason = "not_better"
            rejects["not_better"] = rejects.get("not_better", 0) + 1
        if reason:
            row["reason"] = reason
            log.append(row)
            return False
        row["accepted"] = True
        log.append(row)
        best_plan, best = cand, sc
        return True

    steps = {"beta": float(beta_step), "coef": float(coef_step)}
    for _ in range(max_passes):
        passes_run += 1
        improved = False
        for coord in coords:
            step = steps["beta" if coord[0] == "beta" else "coef"]
            for sgn in (+1.0, -1.0):
                took = False
                for _ext in range(max_extend):
                    cand = stepped(best_plan, coord, sgn * step)
                    if cand is None:
                        break
                    v = cand.beta if coord[0] == "beta" else cand.coef[coord[1], coord[2]]
                    if not evaluate(cand, coord, float(v)):
                        break
                    took = improved = True
                if took or budget_out:
                    break
            if budget_out:
                break
        if budget_out:
            break
        if not improved:
            steps["beta"] *= 0.5
            steps["coef"] *= 0.5
            if steps["coef"] < min_step and steps["beta"] < min_step:
                break

    stats = dict(n_evals=evals, passes_run=passes_run, rejects=rejects,
                 budget_exhausted=budget_out,
                 final_steps={k: round(v, 6) for k, v in steps.items()},
                 accepted_steps=sum(1 for r in log if r["accepted"]) - 1)
    return best_plan, src_score, best, stats


# ----------------------------------------------------------------- production guards - #
def make_limits_guard(q_src: np.ndarray, cols: np.ndarray, limits: dict,
                      strict: bool) -> tuple:
    """Per-frame URDF position-limit guard on the free joints (others are bitwise).

    grandfather 同 v1:源片既有饱和不判死,但禁止重写把它推得更远;--strict-limits 关掉。"""
    lo = np.array([limits[n].lower for n in ISAAC_JOINT_NAMES], dtype=np.float64)
    hi = np.array([limits[n].upper for n in ISAAC_JOINT_NAMES], dtype=np.float64)
    J = q_src.shape[1]
    sat_lo = [ISAAC_JOINT_NAMES[j] for j in range(J) if q_src[:, j].min() < lo[j] - 1e-9]
    sat_hi = [ISAAC_JOINT_NAMES[j] for j in range(J) if q_src[:, j].max() > hi[j] + 1e-9]
    if strict and (sat_lo or sat_hi):
        raise SystemExit(f"--strict-limits: SOURCE clip already violates URDF limits "
                         f"(lo: {sat_lo}; hi: {sat_hi})")
    if not strict:
        lo = np.minimum(lo, q_src.min(axis=0))
        hi = np.maximum(hi, q_src.max(axis=0))

    def guard(q_out: np.ndarray):
        sub = q_out[:, cols]
        over_hi = np.flatnonzero(sub.max(axis=0) > hi[cols] + 1e-7)
        over_lo = np.flatnonzero(sub.min(axis=0) < lo[cols] - 1e-7)
        if over_hi.size:
            i = int(over_hi[0])
            return False, (f"{ISAAC_JOINT_NAMES[cols[i]]} max {sub[:, i].max():+.4f} > "
                           f"hi {hi[cols[i]]:+.4f}")
        if over_lo.size:
            i = int(over_lo[0])
            return False, (f"{ISAAC_JOINT_NAMES[cols[i]]} min {sub[:, i].min():+.4f} < "
                           f"lo {lo[cols[i]]:+.4f}")
        return True, ""

    return guard, sat_hi, sat_lo


class SelfColGuard:
    """C 在环:重写域逐帧自碰撞(audit_self_collision.collect_hits 复用,vendor MJCF)。

    锁段/末帧行逐位 = 源片 ⇒ 域外碰撞状态不可能被本工具改变,只扫 [s0, s1] 即可。"""

    def __init__(self, sm, root_pos: np.ndarray, root_quat: np.ndarray, s0: int, s1: int):
        if asc.mujoco is None:
            raise SystemExit("mujoco is required for the self-collision gate")
        self.sm = sm
        self.data = asc.mujoco.MjData(sm.model)
        self.root_pos, self.root_quat = root_pos, root_quat
        self.s0, self.s1 = s0, s1

    def __call__(self, q_out: np.ndarray):
        sl = slice(self.s0, self.s1 + 1)
        qpos = asc.build_qpos(self.sm, q_out[sl], self.root_pos[sl], self.root_quat[sl])
        hits, colliding = asc.collect_hits(self.sm, self.data, qpos)
        n = int(colliding.sum())
        if n == 0:
            return True, "clean"
        worst = hits[0]
        return False, (f"{n} 帧互穿, worst {worst.body1}<->{worst.body2} "
                       f"深 {worst.depth_peak * 1000:.2f}mm "
                       f"@f{worst.depth_peak_frame + self.s0}")


class DoseScorer:
    """In-loop oracle:逐行照抄 feasibility_oracle.analyze_clip 的动力学与剂量算式。

    合法性前提(fail-loud 检查):root 位姿与双脚全程逐位冻结 ⇒ 支撑多边形/ground_z/支撑
    掩码可以从源片预计算一次。首个 score() 调用(必须是源片)钉下 CoM 基准;main() 里
    还会拿真 oracle 对账,两边剂量对不上直接拒绝开工。"""

    def __init__(self, fo, om, data: dict, body_names: list[str], mu: float, fps: float,
                 support_band: float):
        self.fo, self.om = fo, om
        mj = fo.mujoco
        b_idx = {n: i for i, n in enumerate(body_names)}
        for need in (fo.ROOT_BODY, *fo.FOOT_BODIES):
            if need not in b_idx:
                raise SystemExit(f"body {need!r} missing from the body order")
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
        self.root_pos = body_pos[:, b_idx[fo.ROOT_BODY]]
        self.root_quat = body_quat[:, b_idx[fo.ROOT_BODY]]
        T = body_pos.shape[0]
        self.T, self.dt, self.mu = T, 1.0 / fps, mu

        Rbuf = np.zeros(9)
        foot_world: dict[str, np.ndarray] = {}
        for fb in fo.FOOT_BODIES:
            i = b_idx[fb]
            corners = om.foot_corners[fb]
            world = np.zeros((T, len(corners), 3))
            for t in range(T):
                qn = body_quat[t, i] / np.linalg.norm(body_quat[t, i])
                mj.mju_quat2Mat(Rbuf, qn)
                world[t] = corners @ Rbuf.reshape(3, 3).T + body_pos[t, i]
            foot_world[fb] = world
        sole_z = {fb: foot_world[fb][:, :, 2].min(axis=1) for fb in fo.FOOT_BODIES}
        self.ground_z = float(min(sole_z[fb].min() for fb in fo.FOOT_BODIES))
        support = {fb: sole_z[fb] < self.ground_z + support_band for fb in fo.FOOT_BODIES}
        self.hulls: dict[int, np.ndarray] = {}
        self.flight_frames: list[int] = []
        for t in range(1, T - 1):
            feet_now = [fb for fb in fo.FOOT_BODIES if support[fb][t]]
            if feet_now:
                self.hulls[t] = fo.convex_hull_2d(
                    np.vstack([foot_world[fb][t][:, :2] for fb in feet_now]))
            else:
                self.flight_frames.append(t)
        self.d = mj.MjData(om.model)
        self.src_com: np.ndarray | None = None

    def __call__(self, joint_pos: np.ndarray) -> Score:
        fo, om, mj = self.fo, self.om, self.fo.mujoco
        T, d = self.T, self.d
        qpos = fo.build_qpos(om, joint_pos, self.root_pos, self.root_quat)
        qvel, qacc = fo.differentiate(om, qpos, self.dt)
        qfrc = np.zeros((T, om.nv))
        com = np.zeros((T, 3))
        for t in range(T):
            d.qpos[:] = qpos[t]
            d.qvel[:] = qvel[t]
            d.qacc[:] = qacc[t]
            mj.mj_inverse(om.model, d)
            qfrc[t] = d.qfrc_inverse
            com[t] = d.subtree_com[0]

        eval_mask = np.zeros(T, dtype=bool)
        eval_mask[1:-1] = True
        n_eval = max(int(eval_mask.sum()), 1)
        util = np.abs(qfrc[:, om.joint_dofadr]) / om.tau_max[None, :]
        util_e = np.where(eval_mask[:, None], util, 0.0)
        frame_peak = util_e.max(axis=1)
        dose_tau = float(np.sum(frame_peak > fo.TORQUE_FAIL)) / n_eval

        f_w = qfrc[:, 0:3]
        tau_w = np.zeros((T, 3))
        Rbuf = np.zeros(9)
        for t in range(T):
            mj.mju_quat2Mat(Rbuf, qpos[t, 3:7])
            tau_w[t] = Rbuf.reshape(3, 3) @ qfrc[t, 3:6]

        cop_excess = np.full(T, np.nan)
        fric_ratio = np.full(T, np.nan)
        for t in range(1, T - 1):
            hull = self.hulls.get(t)
            if hull is None:
                continue                    # flight frame: CoP/friction undefined
            fz = float(f_w[t, 2])
            fric_ratio[t] = float(np.hypot(f_w[t, 0], f_w[t, 1])) / (self.mu * max(fz, fo.COP_MIN_FZ))
            if fz > fo.COP_MIN_FZ:
                px, py = fo.cop_from_wrench(f_w[t], tau_w[t], qpos[t, 0:3], self.ground_z)
                cop_excess[t] = fo.signed_dist_to_hull(np.array([px, py]), hull)

        cop_v = np.nan_to_num(cop_excess, nan=-np.inf)
        fric_v = np.nan_to_num(fric_ratio, nan=-np.inf)
        dose_cop = float(np.sum(cop_v > fo.COP_WARN_M)) / n_eval
        dose_fric = float(np.sum(fric_v > fo.FRICTION_FAIL)) / n_eval
        cop_area = float(np.sum(np.clip(np.nan_to_num(cop_excess, nan=0.0), 0.0, None)))
        cop_frames = [int(t) for t in np.flatnonzero(cop_v > fo.COP_WARN_M)]

        if self.src_com is None:
            self.src_com = com.copy()
        dxy = float(np.linalg.norm((com - self.src_com)[:, :2], axis=1).max())
        dz = float(np.abs((com - self.src_com)[:, 2]).max())

        pk = int(np.argmax(frame_peak))
        return Score(
            dose_cop=dose_cop, dose_fric=dose_fric, dose_tau=dose_tau, cop_area=cop_area,
            com_dxy=dxy, com_dz=dz, cop_frames=cop_frames,
            detail=dict(
                cop_max_out_m=float(max(cop_v.max(), 0.0)),
                fric_peak=float(fric_v.max()),
                tau_peak_util=float(frame_peak[pk]),
                tau_peak_joint=ISAAC_JOINT_NAMES[int(np.argmax(util_e[pk]))],
                min_fz_N=float(f_w[1:-1, 2].min()),
                fric_frames=[int(t) for t in np.flatnonzero(fric_v > fo.FRICTION_FAIL)],
                tau_frames=[int(t) for t in np.flatnonzero(frame_peak > fo.TORQUE_FAIL)],
                flight_frames=list(self.flight_frames),
            ))


# ----------------------------------------------------------------- npz reconstruction #
def rebuild_npz_stitched(data: dict, q_out: np.ndarray, fkm, order_cols: list[int],
                         s0: int, s1: int) -> tuple[dict, dict]:
    """Stitched rebuild: untouched rows stay BITWISE source; only rewritten rows get FK.

    joint_pos rows outside (s0, s1) exclusive = source bitwise (强于 v1 的整片重算);
    body_pos/quat 同理逐位保留,重写行用部署 MJCF FK(root 逐帧冻结 = 源 root 行)重算,
    并对 FK 行做半球对齐(和相邻行同半球,否则 so3 差分会算出假 2π 角速度)。
    joint_vel / body 速度按 csv_to_npz 约定 np.gradient / so3 中心差分整片重差分——
    未触行的差分值只读逐位行,自动与源片口径一致。"""
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    q_src = np.asarray(data["joint_pos"], dtype=np.float64)
    changed = [int(t) for t in np.flatnonzero(np.any(q_out != q_src, axis=1))]
    if changed and (min(changed) <= s0 or max(changed) >= s1):
        raise SystemExit("REWRITE BUG: changed rows escaped the (s0, s1) interior")

    jp = np.array(data["joint_pos"], dtype=np.float32, copy=True)
    bp = np.array(data["body_pos_w"], dtype=np.float32, copy=True)
    bq = np.array(data["body_quat_w"], dtype=np.float32, copy=True)
    base_pos = np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0]
    base_quat = np.asarray(data["body_quat_w"], dtype=np.float64)[:, 0]

    hemi_flips = 0
    for t in changed:
        jp[t] = q_out[t].astype(np.float32)
        p, qm = fkm.fk(base_pos[t], base_quat[t],
                       dict(zip(ISAAC_JOINT_NAMES, q_out[t])))
        row_p = p[order_cols].astype(np.float32)
        row_q = qm[order_cols].astype(np.float32)
        flip = np.sum(row_q * bq[t - 1], axis=-1) < 0.0     # ascending t ⇒ t-1 已定稿
        row_q[flip] = -row_q[flip]
        hemi_flips += int(flip.sum())
        bp[t], bq[t] = row_p, row_q

    hemi_bad = int((np.sum(bq[1:] * bq[:-1], axis=-1) < 0.0).sum())
    jv = np.gradient(jp.astype(np.float64), dt, axis=0).astype(np.float32)
    bl = np.gradient(bp.astype(np.float64), dt, axis=0).astype(np.float32)
    ba = np.stack([ctn.so3_derivative(bq[:, b].astype(np.float64), dt)
                   for b in range(bq.shape[1])], axis=1).astype(np.float32)

    # 缝合口径对账:对 s0 / s1(未触帧)也跑一次 FK,和存量行比 —— 若源片 body_* 不是
    # 本 MJCF FK 的产物,缝合处会有一个这个量级的台阶,必须记名。
    stitch_dev = 0.0
    for t in (s0, s1):
        p, _ = fkm.fk(base_pos[t], base_quat[t], dict(zip(ISAAC_JOINT_NAMES, q_src[t])))
        stitch_dev = max(stitch_dev, float(np.abs(
            p[order_cols].astype(np.float32) - np.asarray(data["body_pos_w"])[t]).max()))

    out = {"fps": np.array([int(round(fps))], dtype=np.int64),
           "joint_pos": jp, "joint_vel": jv, "body_pos_w": bp, "body_quat_w": bq,
           "body_lin_vel_w": bl, "body_ang_vel_w": ba}
    acc = dict(changed_frames=changed, n_changed=len(changed),
               hemi_flips=hemi_flips, hemi_bad_adjacent=hemi_bad,
               fk_vs_stored_at_seams_m=stitch_dev)
    return out, acc


# ----------------------------------------------------------------------- reporting -- #
def split_frames(frames: list[int], lock_end: int) -> dict:
    locked = [f for f in frames if f <= lock_end]
    free = [f for f in frames if f > lock_end]
    return dict(total=len(frames), in_lock=len(locked), in_domain=len(free),
                lock_ranges=_ranges(locked), domain_ranges=_ranges(free))


def build_report(*, args, c, s0, s1, joints, plan_best, src_score, best_score, stats,
                 seams, acc, oracle_src, oracle_out_inloop_vs_file, selfcol_final,
                 v_star, com_eps, notes, log, sat_hi, sat_lo, peaks, fps, T) -> dict:
    lock_end = c + LOCK_AFTER_CONTACT
    return dict(
        tool="rewrite_followthrough.py (随挥段路径重写, oracle-在环)",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        verdict="WARN" if notes else "ok",
        notes=notes,
        files=dict(input=os.path.abspath(args.input), output=os.path.abspath(args.output),
                   mjcf=os.path.abspath(args.mjcf)),
        windows=dict(frames=T, fps=fps, contact_frame=c, lock_end=lock_end,
                     support=[s0, s1], rewritable=[s0 + 1, s1 - 1],
                     n_changed=acc["n_changed"]),
        allocation=dict(joints=joints, mode=args.mode, basis_k=len(peaks), peaks=peaks,
                        beta=round(float(plan_best.beta), 6),
                        coef_rad={n: [round(float(v), 6) for v in plan_best.coef[i]]
                                  for i, n in enumerate(plan_best.names)},
                        coef_peak_abs_rad=float(np.abs(plan_best.coef).max(initial=0.0)),
                        src_saturated_hi=sat_hi, src_saturated_lo=sat_lo),
        doses=dict(
            source_file=oracle_src,
            source_inloop=src_score.as_row() | dict(detail=src_score.detail),
            final_inloop=best_score.as_row() | dict(detail=best_score.detail),
            final_file=oracle_out_inloop_vs_file,
        ),
        cop_frames=dict(source=split_frames(src_score.cop_frames, lock_end),
                        final=split_frames(best_score.cop_frames, lock_end)),
        selfcol=selfcol_final,
        com=dict(max_dxy_m=best_score.com_dxy, max_dz_m=best_score.com_dz, eps_m=com_eps,
                 fuse_tripped=bool(best_score.com_dxy > com_eps)),
        seams=seams,
        stitch=acc,
        v_star=v_star,
        search=stats,
        candidates=log,
    )


def report_md(rep: dict, md_candidate_cap: int = 500) -> str:
    w, a, d, s = rep["windows"], rep["allocation"], rep["doses"], rep["search"]
    cf, sm, com = rep["cop_frames"], rep["seams"], rep["com"]
    src_f, src_i = d["source_file"], d["source_inloop"]
    fin_i, fin_f = d["final_inloop"], d["final_file"]
    L = [
        f"# 随挥段路径重写 — **{rep['verdict']}**",
        "",
        f"- generated: {rep['generated_utc']}",
        f"- 源片: `{rep['files']['input']}`",
        f"- 产物: `{rep['files']['output']}`",
        f"- 窗口(人话:触球行及之前一根汗毛不动,只改收拍): {w['frames']} 帧 @ {w['fps']:.0f}fps,"
        f"触球 f{w['contact_frame']},锁死 [0, f{w['lock_end']}] 逐位,重写域内实改 "
        f"{w['n_changed']} 帧(f{w['rewritable'][0]}..f{w['rewritable'][1]}),末帧逐位回 ready",
        f"- 自由度: {', '.join(a['joints'])}(腰俯仰/侧滚+双腿冻结 = 法则)",
        f"- 参数化: mode={a['mode']},基元 K={a['basis_k']}(峰位 f{a['peaks']}),"
        f"收拍模板混合 β={a['beta']:.3f},|系数|峰值 {a['coef_peak_abs_rad']:.4f} rad",
        f"- 搜索: {s['n_evals']} 次候选评估 / {s['accepted_steps']} 步收下 / "
        f"{s['passes_run']} 轮;拒绝记名 {s['rejects']}"
        + ("(预算耗尽)" if s["budget_exhausted"] else ""),
    ]
    if a["src_saturated_hi"] or a["src_saturated_lo"]:
        L.append(f"- 源片既有限位饱和(grandfather): hi={a['src_saturated_hi']} "
                 f"lo={a['src_saturated_lo']}")
    L += [
        "",
        "## 剂量对账(oracle 三列;CoP 是摔跤预测主项)",
        "",
        "| 量 | 源(oracle 文件) | 源(in-loop) | 终局(in-loop) | 终局(文件复核) |",
        "|---|---|---|---|---|",
        f"| CoP 剂量 | {src_f['doses']['cop']:.4f} | {src_i['dose_cop']:.4f} "
        f"| **{fin_i['dose_cop']:.4f}** | {fin_f['doses']['cop']:.4f} |",
        f"| 摩擦剂量 | {src_f['doses']['friction']:.4f} | {src_i['dose_fric']:.4f} "
        f"| {fin_i['dose_fric']:.4f} | {fin_f['doses']['friction']:.4f} |",
        f"| τ 剂量 | {src_f['doses']['torque']:.4f} | {src_i['dose_tau']:.4f} "
        f"| {fin_i['dose_tau']:.4f} | {fin_f['doses']['torque']:.4f} |",
        f"| CoP 正溢出面积 [m·帧] | - | {src_i['cop_area']:.4f} | {fin_i['cop_area']:.4f} | - |",
        f"| oracle 判决 | {src_f['verdict']} | - | - | **{fin_f['verdict']}** |",
        "",
        f"- CoP 违规帧: 源 {cf['source']['total']} 帧(锁窗内 {cf['source']['in_lock']} 帧 "
        f"{cf['source']['lock_ranges']} 本工具管不到 + 域内 {cf['source']['in_domain']} 帧 "
        f"{cf['source']['domain_ranges']}) → 终局 {cf['final']['total']} 帧"
        f"(锁窗 {cf['final']['in_lock']} + 域内 {cf['final']['in_domain']} 帧 "
        f"{cf['final']['domain_ranges']})",
        f"- 自碰撞(C 在环,每候选先查再打分): 终局全片审计 **{rep['selfcol']['verdict']}**"
        f",拍-躯干最小余隙 {rep['selfcol'].get('racket_torso_min_mm', '-')} mm",
        f"- CoM 保险丝: max|Δcom| xy {com['max_dxy_m'] * 100:.2f} cm / z "
        f"{com['max_dz_m'] * 100:.2f} cm(ε={com['eps_m'] * 100:g} cm)→ "
        f"{'**跳闸**' if com['fuse_tripped'] else 'ok'}",
        f"- 接缝 C1: 锁窗逐位={sm['lock_window_bitwise']},触球行逐位={sm['contact_row_bitwise']},"
        f"末帧逐位={sm['end_pose_bitwise']};q̇ 残差 c+2 处 {sm['vel_residual_at_lock_end']:.2e} "
        f"/ 末帧处 {sm['vel_residual_at_end']:.2e} rad/s(构造应为 0)",
        f"- |v*| 对账: 源 {rep['v_star']['src_mps']:.4f} → 出片 {rep['v_star']['out_mps']:.4f} m/s"
        f"(锁窗覆盖 ±2 模板行 ⇒ 逐位不变={rep['v_star']['bitwise']})",
        f"- 缝合口径: FK-vs-stored 缝帧偏差 {rep['stitch']['fk_vs_stored_at_seams_m'] * 1000:.3f} mm;"
        f"半球翻转 {rep['stitch']['hemi_flips']} 行,残留异向邻行 {rep['stitch']['hemi_bad_adjacent']}",
    ]
    if rep["notes"]:
        L += ["", "## WARN 记名", ""] + [f"- {n}" for n in rep["notes"]]
    acc_rows = [r for r in rep["candidates"] if r["accepted"]]
    L += ["", f"## 收下的步子({len(acc_rows) - 1} 步,剂量键单调变好 = 验收项)", "",
          "| # | 坐标 | 值 | CoP 剂量 | 面积 | 摩擦 | τ | CoM xy [cm] |",
          "|---|---|---|---|---|---|---|---|"]
    for r in acc_rows:
        L.append(f"| {r['idx']} | {r['coord']} | {r['value']:+.4f} | {r['dose_cop']:.4f} "
                 f"| {r['cop_area']:.4f} | {r['dose_fric']:.4f} | {r['dose_tau']:.4f} "
                 f"| {r['com_dxy_cm']:.2f} |")
    rows = rep["candidates"]
    L += ["", f"## 逐候选({len(rows)} 条;剂量三列 / 自碰撞 / CoM / 拒绝记名)", "",
          "| # | 坐标 | 值 | CoP | 摩擦 | τ | 面积 | CoM xy | 结果 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows[:md_candidate_cap]:
        dose = (f"{r['dose_cop']:.4f} | {r['dose_fric']:.4f} | {r['dose_tau']:.4f} "
                f"| {r['cop_area']:.4f} | {r['com_dxy_cm']:.2f}"
                if "dose_cop" in r else "- | - | - | - | -")
        res = "**收**" if r["accepted"] else r["reason"].replace("|", "\\|")
        L.append(f"| {r['idx']} | {r['coord']} | {r['value']:+.4f} | {dose} | {res} |")
    if len(rows) > md_candidate_cap:
        L.append(f"| … | (+{len(rows) - md_candidate_cap} 条,见 json) | | | | | | | |")
    L += ["", "REGISTRY REMINDER: 触球帧与片长均未动(不重排时间律),phase 沿用源片登记值;"
              "产物需重跑判炸器 L0 + C 卡全片 + oracle 三关后才可入训练资产表。"]
    return "\n".join(L)


# -------------------------------------------------------------------------- CLI ----- #
def repo_root_of(script: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(script)), "../../.."))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="源 clip npz(触球行及之前逐位保留)")
    ap.add_argument("--output", required=True, help="重写产物 npz")
    ap.add_argument("--phase", type=float, required=True,
                    help="登记触球相位(strike_annotations.yaml)——锁窗由它定")
    ap.add_argument("--joints", default="arm5",
                    help="preset arm5(肩三轴+肘+腰偏航)/ arm7(+腕 roll/pitch)/ 逗号关节名")
    ap.add_argument("--mode", choices=("field", "blend", "hybrid"), default="hybrid",
                    help="形变场 / 收拍模板混合 / 两者一起坐标下降(默认)")
    ap.add_argument("--basis-k", type=int, default=DEFAULT_BASIS_K,
                    help="每关节形变基元个数(域太短会自动收窄并 WARN)")
    ap.add_argument("--coef-step", type=float, default=0.08, help="系数初始步长 [rad]")
    ap.add_argument("--beta-step", type=float, default=0.25, help="β 初始步长")
    ap.add_argument("--min-step", type=float, default=0.005, help="步长下限,减半到此收工")
    ap.add_argument("--coef-cap", type=float, default=1.2, help="单系数绝对值上限 [rad]")
    ap.add_argument("--max-passes", type=int, default=8, help="坐标下降轮数上限")
    ap.add_argument("--max-evals", type=int, default=800, help="候选评估总预算")
    ap.add_argument("--com-eps", type=float, default=COM_EPS_M,
                    help="CoM xy 保险丝 [m],候选超过即拒绝(法则:重心不要变)")
    ap.add_argument("--dose-slack", type=float, default=0.0,
                    help="允许摩擦/τ 剂量比源片劣化的量(默认 0 = 一点不许)")
    ap.add_argument("--ramp", type=int, default=None, help="模板平台窗起落帧数(默认域长/5)")
    ap.add_argument("--strict-limits", action="store_true",
                    help="源片既有限位饱和也判死(默认 grandfather)")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--body-order", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--mu", type=float, default=None, help="摩擦系数(默认 oracle 的 0.8)")
    ap.add_argument("--support-band", type=float, default=None)
    ap.add_argument("--report", default=None, help="json 报告路径")
    ap.add_argument("--md", default=None, help="markdown 报告路径")
    args = ap.parse_args(argv)

    root = repo_root_of(__file__)
    if args.urdf is None:
        args.urdf = os.path.join(root, "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf")

    joints = resolve_joints(args.joints)
    data = dict(np.load(args.input))
    q_src, fps = validate_npz(data)
    T = q_src.shape[0]
    c = st.contact_frame(args.phase, T)
    if not (0 < c < T - 1):
        raise SystemExit(f"contact frame {c} out of range for T={T}")
    s0, s1 = rewrite_windows(T, c)
    cols = np.array([ISAAC_JOINT_NAMES.index(n) for n in joints])

    basis, peaks = bump_basis(T, s0, s1, args.basis_k)
    template = None
    if args.mode in ("blend", "hybrid"):
        W = plateau_window(T, s0, s1, args.ramp)
        template = retreat_template(q_src, cols, c, s0, s1, W)
    plan = RewritePlan(cols=cols, names=joints, basis=basis, peaks=peaks,
                       template=template, coef=np.zeros((len(joints), len(peaks))),
                       beta=0.0, s0=s0, s1=s1, c=c)

    limits = parse_urdf_limits(args.urdf)
    limits_guard, sat_hi, sat_lo = make_limits_guard(q_src, cols, limits, args.strict_limits)

    # body order (fail-loud: root column must really be the pelvis)
    order = [ln.strip() for ln in open(args.body_order) if ln.strip()]
    if len(order) != data["body_pos_w"].shape[1]:
        raise SystemExit(f"body order has {len(order)} names, npz has "
                         f"{data['body_pos_w'].shape[1]} bodies")
    if order[0] != "pelvis_link":
        raise SystemExit(f"body column 0 is {order[0]!r}, expected 'pelvis_link'")

    # --- C 在环: self-collision gate on the rewrite domain ---------------------------
    sm = asc.load_selfcol_model(args.mjcf)
    root_pos = np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0]
    root_quat = np.asarray(data["body_quat_w"], dtype=np.float64)[:, 0]
    selfcol_guard = SelfColGuard(sm, root_pos, root_quat, s0, s1)
    require_clean_source(selfcol_guard, q_src)

    # --- oracle in the loop ----------------------------------------------------------
    fo = es.load_oracle(root)
    om = fo.load_oracle_model(args.mjcf)
    mu = fo.DEFAULT_MU if args.mu is None else args.mu
    band = fo.DEFAULT_SUPPORT_BAND if args.support_band is None else args.support_band
    scorer = DoseScorer(fo, om, data, order, mu, fps, band)

    # 对账:in-loop 打分器 vs 真 oracle,数字对不上 = 打分器漂移,拒绝开工
    probe = scorer(q_src)
    oracle_src = es.score_with_oracle(args.input, args.mjcf, args.body_order,
                                      args.annotations, root)
    for k_mine, k_oracle in (("dose_cop", "cop"), ("dose_fric", "friction"),
                             ("dose_tau", "torque")):
        mine, ref = getattr(probe, k_mine), oracle_src["doses"][k_oracle]
        if abs(mine - ref) > 1e-9:
            raise SystemExit(f"in-loop scorer drifted from feasibility_oracle: "
                             f"{k_oracle} {mine:.6f} vs {ref:.6f} — 拒绝开工")

    # --- search ------------------------------------------------------------------------
    log: list[dict] = []
    plan_best, src_score, best_score, stats = coordinate_search(
        q_src, plan, scorer, [("limits", limits_guard), ("selfcol", selfcol_guard)],
        mode=args.mode, com_eps=args.com_eps, dose_slack=args.dose_slack,
        coef_step=args.coef_step, beta_step=args.beta_step, min_step=args.min_step,
        max_passes=args.max_passes, max_evals=args.max_evals, coef_cap=args.coef_cap,
        log=log)

    q_out = plan_best.apply(q_src)
    assert_structure(q_src, q_out, cols, s0, s1)
    ok, info = limits_guard(q_out)
    if not ok:
        raise SystemExit(f"final trajectory violates URDF limits ({info}) — should be unreachable")
    ok, info = selfcol_guard(q_out)
    if not ok:
        raise SystemExit(f"final trajectory self-collides ({info}) — should be unreachable")
    seams = seam_residuals(q_src, q_out, c, fps)

    # --- rebuild + write ---------------------------------------------------------------
    fkm = ctn.MjFK(args.mjcf, ISAAC_JOINT_NAMES)
    names = fkm.body_names()
    order_cols = [names.index(n) for n in order]
    out, acc = rebuild_npz_stitched(data, q_out, fkm, order_cols, s0, s1)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **out)

    # --- file-level verification ---------------------------------------------------------
    oracle_out = es.score_with_oracle(args.output, args.mjcf, args.body_order,
                                      args.annotations, root)
    ann = asc.load_annotations(args.annotations)
    sc_rep = asc.audit_clip(args.output, sm, annotations=ann, body_order=args.body_order)
    rt = sc_rep.clearance("racket-torso")
    selfcol_final = dict(verdict=sc_rep.verdict,
                         racket_torso_min_mm=None if rt is None else round(rt.min_dist * 1000, 1),
                         findings=[f.to_json() for f in sc_rep.findings if f.level != asc.PASS])

    blade_src = st.blade_positions(data)
    blade_out = st.blade_positions(out)
    v0 = st.clean_speed_at(blade_src, c, 1.0 / fps)
    v1 = st.clean_speed_at(blade_out, c, 1.0 / fps)
    v_star = dict(src_mps=v0, out_mps=v1, bitwise=bool(v0 == v1))

    notes: list[str] = []
    if best_score.key() >= src_score.key():
        notes.append("搜索无改善:终局剂量键未低于源片(域内可能没有可挤的水;看逐候选拒绝记名)")
    drift = abs(best_score.dose_cop - oracle_out["doses"]["cop"])
    if drift > FILE_DOSE_DRIFT_WARN:
        notes.append(f"文件复核 CoP 剂量与 in-loop 偏差 {drift:.4f} > {FILE_DOSE_DRIFT_WARN}"
                     f"(float32 重建挪动了边界帧,以文件复核为准)")
    if acc["fk_vs_stored_at_seams_m"] > STITCH_DEV_WARN_M:
        notes.append(f"缝合口径:FK 与存量 body 行在缝帧差 "
                     f"{acc['fk_vs_stored_at_seams_m'] * 1000:.2f} mm > "
                     f"{STITCH_DEV_WARN_M * 1000:g} mm(源片 body_* 非本 MJCF FK 产物?)")
    if acc["hemi_bad_adjacent"] > 0:
        notes.append(f"四元数半球残留异向邻行 {acc['hemi_bad_adjacent']} 处,角速度行可疑")
    if scorer.flight_frames:
        notes.append(f"源片存在腾空帧 {_ranges(scorer.flight_frames)}(CoP/摩擦无定义,已按 oracle 口径跳过)")
    if not v_star["bitwise"]:
        notes.append("|v*| 不逐位(不应发生:±2 模板行都在锁窗内)")
    if sc_rep.verdict == asc.FAIL:
        notes.append("终局全片自碰撞审计 FAIL(域外既有问题?本工具只保证域内干净)")

    rep = build_report(args=args, c=c, s0=s0, s1=s1, joints=joints, plan_best=plan_best,
                       src_score=src_score, best_score=best_score, stats=stats,
                       seams=seams, acc=acc, oracle_src=oracle_src,
                       oracle_out_inloop_vs_file=oracle_out, selfcol_final=selfcol_final,
                       v_star=v_star, com_eps=args.com_eps, notes=notes, log=log,
                       sat_hi=sat_hi, sat_lo=sat_lo, peaks=peaks, fps=fps, T=T)
    md = report_md(rep)
    print(md)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False, default=float)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(md)
    return 1 if rep["verdict"] == "WARN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
