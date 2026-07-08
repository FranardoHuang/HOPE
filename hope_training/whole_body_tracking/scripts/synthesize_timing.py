#!/usr/bin/env python3
"""Robot-centric TIME-LAW SYNTHESIS (时间律合成) for motion .npz clips — v1.

人话:视频只当"路径"用(关节空间里怎么走、经过什么次序),时间怎么走由机器人自己解:
ready 静止起步 → 匀加速把拍子加速到"答案拍速"→ 以该速度通过触球帧 → 触球后匀减速回静止。
加速要多久(T_a)不抄视频——取满足"逐关节速度 ≤ URDF 限位×余量、逐关节加速度 ≤ 预算
(v4rg 实证包络×scale)"的最小值;**速度不可达 = 延长加速时间(拉长 run-up,允许超过源片长),
绝不降速**。变速(R14 重写)= 换一个答案拍速 |v*| 重解同一时间律(--strike-speed)。

Design doc: docs/research/robot_centric_timing_2026-07-09.md (path/trajectory 分解,
TOPP 思想). Ablation arm B (v5syn); arm A = retime_motion_clip.py (v5hLt, 保视频节奏);
arm C = v4rg (对照). The two tools are DELIBERATELY separate files — different math,
different hypothesis; do not merge.

WHAT IT DOES (v1 = 从简但可验证)
  1. path extraction: s = source frame index (0..T-1), q(s) = linear interp of
     joint_pos (root pose from body col 0: lerp/slerp). No arc-length reparam in v1
     (可选轻度弧长重参数 — 报告里记为 not-applied).
  2. boundary condition: blade speed AT the contact frame (registry phase ->
     c = round(phase*(T-1))) must equal |v*| (default = the SOURCE's own clean
     blade speed at c — "复现原速但时序机器人本位"; --strike-speed overrides =
     天生支持变速). FK-numeric: sdot* = |v*| / |d p_blade/ds (c)| where p_blade
     comes from the stored body arrays (wrist body + mount FK, the
     analyze_strike_phase blade convention) — the npz IS the FK cache.
  3. time law s(t), piecewise-analytic, C1 (velocity-continuous):
       [ready wait  (grid snap; ṡ=0)]
       [uniform accel 0->sdot* over T_a]
       [cruise at sdot* >= --min-cruise-s through the contact  <- strike-window
        protection: the training-parity clean-FD strike velocity stays unbiased]
       [post-contact hold --post-contact-hold-s at sdot*]
       [uniform decel sdot*->0 over the remaining path]
       [rest pad to the output grid]
     T_a = the SMALLEST value on a --ta-grid-s grid satisfying, over the WHOLE
     profile (dense --dense-dt-s sampling of q̇ = q'(s)·ṡ, q̈ = q''(s)·ṡ² + q'(s)·s̈):
       per-joint |q̇| <= vel-limit-frac × URDF velocity limit
       per-joint |q̈| <= budget (= --budget-scale × per-joint max|acc| measured on
                                 the --budget-clips, e.g. the v4rg pair — 实证
                                 "机器人做得到"的包络; recorded in the report)
     IRREDUCIBLE VIOLATIONS: the contact-frame speed itself fixes q̇(c) = q'(c)·sdot*
     — no time law can lower it without降速 (forbidden). Feasibility is therefore
     judged RELATIVE to the gentlest member of the family (T_a = T_a_max, the pure
     minimal-peak-acceleration triangle): a candidate is accepted iff no joint is
     worse than max(1, gentlest utilization). If the gentlest profile already
     exceeds a budget the verdict is `budget_exceeded_irreducible` (reported per
     joint, npz still written — the L0 audit gates hard limits downstream).
  4. resample on the OUTPUT 50 fps grid: joint_pos interp (integer-s rows are
     BITWISE source copies — the contact row is one by construction, so the
     contact face normal/pose survive exactly); joint_vel RE-DIFFERENTIATED
     (np.gradient, the csv_to_npz convention); body_* recomputed with MuJoCo FK
     on the deploy MJCF (--body-mode fk, the csv_to_npz_mujoco path; body
     velocities np.gradient / SO3 central difference). --body-mode interp
     (lerp/slerp of the stored body arrays) is for CPU tests only — the report
     records the mode; production conversions must use fk.
  5. report (json + md): chosen T_a / scan bounds, wait/cruise/decel durations,
     duration & run-up change vs source, per-joint peak velocity/acceleration
     utilizations (top offenders), contact-frame blade-speed fidelity (analytic +
     clean-FD on the output grid vs |v*|), face-normal fidelity (deg vs source
     contact frame), first-frame health, NEW PHASE + registry reminder.

NEW PHASE 口径: contact row = bitwise copy of source frame c at output index k*;
phase_out = k*/(T_out-1). The video's frame convention is DEAD for this asset
(时间律合成资产,视频约定帧弃用) — register phase_out.

USAGE (pod, mjeval/hope_isaac venv: numpy + mujoco)
    python scripts/synthesize_timing.py \
        --input  .../v5_height_fix/hope_forehand_v5hLs_cal.npz --phase 0.678 \
        --output .../v5_height_fix/hope_forehand_v5syn_cal.npz \
        --budget-clips .../regen_0708_candidates/hope_forehand_v4rg_cal.npz \
                       .../regen_0708_candidates/hope_backhand_v4rg_cal.npz \
        --urdf .../URDF-JOINT-LINK.urdf \
        --body-mode fk --mjcf .../a3_pingpong.xml \
        --body-order /workspace/franco/body_order_isaac.txt \
        --report fh_syn.json --md fh_syn.md

DEPENDENCIES: numpy always; mujoco only for --body-mode fk (import deferred).
Unit tests: tests/test_synthesize_timing.py (pure CPU, synthetic clips).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv_to_npz_mujoco as ctn  # noqa: E402  (numpy-only at import time)
from audit_motion_npz import ISAAC_JOINT_NAMES, parse_urdf_limits  # noqa: E402

KNOWN_KEYS = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
              "body_lin_vel_w", "body_ang_vel_w")

# Blade conventions — MUST match analyze_strike_phase.py / RacketTargetCommand.
RACKET_BODY = 31           # right_wrist_yaw_Link column in body_* arrays
NORMAL_AXIS = 1            # blade face normal = wrist-frame +Y
MOUNT_OFFSET = np.array([0.210211399202899, 0.0320784994676765, 0.0320358706296689])
CLEAN_VEL_WINDOW = 2       # training-parity clean strike velocity stencil (frames)


# ------------------------------------------------------------------ small math -- #
def contact_frame(phase: float, T: int) -> int:
    """Registry convention: contact frame = round(phase * (T-1))."""
    return int(round(float(phase) * (T - 1)))


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """wxyz quat(s) (..,4) -> rotation matrices (..,3,3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def blade_positions(data: dict) -> np.ndarray:
    """Blade-center world positions (T,3) from the stored body arrays (FK cache)."""
    P = np.asarray(data["body_pos_w"], dtype=np.float64)[:, RACKET_BODY]
    Q = np.asarray(data["body_quat_w"], dtype=np.float64)[:, RACKET_BODY]
    return P + np.einsum("tij,j->ti", quat_to_rot(Q), MOUNT_OFFSET)


def blade_face_normals(data: dict) -> np.ndarray:
    Q = np.asarray(data["body_quat_w"], dtype=np.float64)[:, RACKET_BODY]
    return quat_to_rot(Q)[:, :, NORMAL_AXIS]


def clean_speed_at(blade_pos: np.ndarray, frame: int, dt: float,
                   window: int = CLEAN_VEL_WINDOW) -> float:
    """Training-parity clean blade speed: centered FD over +-window frames, edge-clamped."""
    T = blade_pos.shape[0]
    lo = max(0, frame - window)
    hi = min(T - 1, frame + window)
    return float(np.linalg.norm(blade_pos[hi] - blade_pos[lo]) / ((hi - lo) * dt))


def blade_path_deriv_at(blade_pos: np.ndarray, frame: int,
                        window: int = CLEAN_VEL_WINDOW) -> float:
    """|d p_blade / ds| at a frame [m/frame], same stencil as clean_speed_at.

    clean_speed = blade_path_deriv * fps when both use the same stencil, so the
    DEFAULT answer speed (source clean speed) maps to sdot* == source fps exactly.
    """
    T = blade_pos.shape[0]
    lo = max(0, frame - window)
    hi = min(T - 1, frame + window)
    return float(np.linalg.norm(blade_pos[hi] - blade_pos[lo]) / (hi - lo))


def _interp_rows(arr: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Linear interp of a (T, ...) array at fractional frame indices s (M,)."""
    T = arr.shape[0]
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, T - 1)
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (s - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
    out = arr[i0] * (1.0 - f) + arr[i1] * f
    exact = np.abs(s - np.round(s)) < 1e-9      # integer-s rows: bitwise source copies
    if exact.any():
        out[exact] = arr[np.round(s[exact]).astype(int)]
    return out


def _slerp_rows(quat: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Slerp of a (T,4) or (T,B,4) quat array at fractional frame indices."""
    T = quat.shape[0]
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, T - 1)
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (s - i0).astype(np.float64)
    blend = f.reshape((-1,) + (1,) * (quat.ndim - 1))
    out = ctn.quat_slerp(quat[i0].astype(np.float64), quat[i1].astype(np.float64), blend)
    exact = np.abs(s - np.round(s)) < 1e-9
    if exact.any():
        out[exact] = quat[np.round(s[exact]).astype(int)]
    return out


# ------------------------------------------------------------------- time law --- #
@dataclass
class TimeLaw:
    """Piecewise-analytic monotone time map s(t) (s in source-frame units).

    Segments: wait [0,tw) | accel [tw, tw+Ta) | cruise+hold [.., tD0) | decel
    [tD0, tD0+Td) | rest [tD0+Td, ...). Velocity-continuous by construction.
    """
    tw: float          # ready wait (grid snap), s=0, sdot=0
    Ta: float          # uniform-acceleration duration, 0 -> sdot_star
    sdot_star: float   # path speed through the contact [frames/s]
    c: float           # contact path position [frames]
    s_end: float       # last source frame index (T_src - 1)
    post_hold: float   # post-contact hold at sdot_star [s] (strike-window protection)

    @property
    def A(self) -> float:                 # accel magnitude [frames/s^2]
        return self.sdot_star / self.Ta

    @property
    def d_a(self) -> float:               # path distance covered while accelerating
        return 0.5 * self.sdot_star * self.Ta

    @property
    def t_star(self) -> float:            # contact time
        return self.tw + self.Ta + (self.c - self.d_a) / self.sdot_star

    @property
    def tD0(self) -> float:               # decel start time
        return self.t_star + self.post_hold

    @property
    def sD0(self) -> float:               # decel start path position
        return self.c + self.post_hold * self.sdot_star

    @property
    def D(self) -> float:                 # decel magnitude [frames/s^2]
        return self.sdot_star ** 2 / (2.0 * (self.s_end - self.sD0))

    @property
    def Td(self) -> float:
        return self.sdot_star / self.D

    @property
    def t_end(self) -> float:
        return self.tD0 + self.Td

    def s_sdot_sddot(self, t: np.ndarray):
        """Vectorized (s, sdot, sddot) at times t."""
        t = np.asarray(t, dtype=np.float64)
        s = np.empty_like(t); v = np.empty_like(t); a = np.empty_like(t)
        tw, Ta, ss = self.tw, self.Ta, self.sdot_star
        tA1 = tw + Ta
        m = t < tw
        s[m], v[m], a[m] = 0.0, 0.0, 0.0
        m = (t >= tw) & (t < tA1)
        tau = t[m] - tw
        s[m] = 0.5 * self.A * tau ** 2; v[m] = self.A * tau; a[m] = self.A
        m = (t >= tA1) & (t < self.tD0)
        s[m] = self.d_a + ss * (t[m] - tA1); v[m] = ss; a[m] = 0.0
        m = (t >= self.tD0) & (t < self.t_end)
        tau = t[m] - self.tD0
        s[m] = self.sD0 + ss * tau - 0.5 * self.D * tau ** 2
        v[m] = ss - self.D * tau; a[m] = -self.D
        m = t >= self.t_end
        s[m], v[m], a[m] = self.s_end, 0.0, 0.0
        return np.clip(s, 0.0, self.s_end), v, a


def build_time_law(c: float, s_end: float, sdot_star: float, Ta: float,
                   fps_out: float, post_hold_s: float) -> TimeLaw:
    """Assemble a TimeLaw and snap the contact time UP onto the output grid via tw."""
    law = TimeLaw(tw=0.0, Ta=Ta, sdot_star=sdot_star, c=c, s_end=s_end,
                  post_hold=post_hold_s)
    t_raw = law.t_star
    k = np.ceil(t_raw * fps_out - 1e-9)
    law.tw = float(k / fps_out - t_raw)
    if law.tw < 0.0:
        law.tw = 0.0
    return law


def ta_max(c: float, sdot_star: float, min_cruise_s: float) -> float:
    """Largest admissible T_a: acceleration must complete >= min_cruise_s before contact."""
    d_max = c - sdot_star * min_cruise_s
    if d_max <= 0.0:
        raise SystemExit(
            f"run-up too short: contact at path position {c:.1f} frames but the required "
            f"pre-contact cruise alone covers {sdot_star * min_cruise_s:.1f} frames — the "
            f"answer speed cannot be built up on this path (题对该 ready 态不可行)")
    return 2.0 * d_max / sdot_star


def profile_peaks(law: TimeLaw, qp: np.ndarray, qpp: np.ndarray,
                  dense_dt: float, t_lo: float = 0.0,
                  t_hi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint peak |q̇|, |q̈| over [t_lo, t_hi] (dense time sampling).

    q̇ = q'(s)·ṡ ; q̈ = q''(s)·ṡ² + q'(s)·s̈ — chain rule on the path derivatives
    (per-frame np.gradient arrays, linearly interpolated in s).
    """
    if t_hi is None:
        t_hi = law.t_end
    t = np.arange(t_lo, t_hi + dense_dt, dense_dt)
    s, v, a = law.s_sdot_sddot(t)
    qp_s = _interp_rows(qp, s)
    qpp_s = _interp_rows(qpp, s)
    qdot = qp_s * v[:, None]
    qddot = qpp_s * (v ** 2)[:, None] + qp_s * a[:, None]
    return np.abs(qdot).max(axis=0), np.abs(qddot).max(axis=0)


def solve_min_ta(c: float, s_end: float, sdot_star: float, qp: np.ndarray,
                 qpp: np.ndarray, vel_cap: np.ndarray, acc_budget: np.ndarray,
                 fps_out: float, min_cruise_s: float, post_hold_s: float,
                 ta_grid_s: float, dense_dt_s: float):
    """Smallest T_a on the grid whose PRE-CONTACT profile ([0, t*] — the only part
    T_a controls) violates no budget beyond the IRREDUCIBLE floor (= the gentlest
    profile T_a_max, the pure minimal-peak-acceleration triangle; e.g. the
    contact-frame speed itself is pinned by |v*|). The POST-CONTACT part (hold +
    uniform decel back to rest) is IDENTICAL for every T_a — it is checked once
    and reported separately, never traded against the accel phase.
    Returns (law, info dict)."""
    tmax = ta_max(c, sdot_star, min_cruise_s)

    def pre_peaks(Ta):
        law = build_time_law(c, s_end, sdot_star, Ta, fps_out, post_hold_s)
        pv, pa = profile_peaks(law, qp, qpp, dense_dt_s, t_hi=law.t_star)
        return law, pv / vel_cap, pa / acc_budget

    base_law, base_uv, base_ua = pre_peaks(tmax)
    allowed_v = np.maximum(1.0, base_uv) * (1.0 + 1e-6)
    allowed_a = np.maximum(1.0, base_ua) * (1.0 + 1e-6)
    irreducible = bool((base_uv > 1.0).any() or (base_ua > 1.0).any())

    chosen, cv, ca = base_law, base_uv, base_ua
    n_scanned = 0
    for Ta in np.arange(ta_grid_s, tmax, ta_grid_s):
        n_scanned += 1
        law, uv, ua = pre_peaks(float(Ta))
        if (uv <= allowed_v).all() and (ua <= allowed_a).all():
            chosen, cv, ca = law, uv, ua
            break

    # forced post-contact segment (same for all T_a): hold at sdot* + uniform decel
    post_v, post_a = profile_peaks(chosen, qp, qpp, dense_dt_s, t_lo=chosen.t_star)
    post_uv, post_ua = post_v / vel_cap, post_a / acc_budget

    info = dict(ta_max_s=float(tmax), ta_grid_s=float(ta_grid_s),
                n_scanned=int(n_scanned), irreducible=irreducible,
                base_vel_util=base_uv, base_acc_util=base_ua,
                vel_util=cv, acc_util=ca,
                post_vel_util=post_uv, post_acc_util=post_ua,
                decel_over_budget=bool((post_uv > 1.0).any() or (post_ua > 1.0).any()))
    return chosen, info


# ------------------------------------------------------------------- resample --- #
def resample(data: dict, law: TimeLaw, fps_out: float, body_mode: str = "interp",
             fk_ctx=None):
    """Sample the source path on the new time law's uniform output grid."""
    q = np.asarray(data["joint_pos"])
    T_out = int(np.ceil(law.t_end * fps_out - 1e-9)) + 1 + 1  # +1 rest pad frame
    t = np.arange(T_out, dtype=np.float64) / fps_out
    s, _, _ = law.s_sdot_sddot(t)
    k_star = int(round(law.t_star * fps_out))
    if abs(s[k_star] - law.c) > 1e-6:
        raise SystemExit(f"grid snap failed: s(k*={k_star}) = {s[k_star]:.6f} != c = {law.c}")

    dt = 1.0 / fps_out
    jp = _interp_rows(q, s).astype(np.float32)
    jv = np.gradient(jp.astype(np.float64), dt, axis=0).astype(np.float32)

    base_pos = _interp_rows(np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0], s)
    base_quat = _slerp_rows(np.asarray(data["body_quat_w"], dtype=np.float64)[:, 0], s)

    if body_mode == "fk":
        fkm, cols = fk_ctx
        pos_all, quat_all = ctn.fk_series(fkm, base_pos, base_quat,
                                          jp.astype(np.float64), ISAAC_JOINT_NAMES)
        bp = pos_all[:, cols].astype(np.float32)
        bq = quat_all[:, cols].astype(np.float32)
    elif body_mode == "interp":
        bp = _interp_rows(np.asarray(data["body_pos_w"], dtype=np.float64), s).astype(np.float32)
        bq = _slerp_rows(np.asarray(data["body_quat_w"], dtype=np.float64), s).astype(np.float32)
    else:
        raise SystemExit(f"unknown --body-mode {body_mode!r}")

    bl = np.gradient(bp.astype(np.float64), dt, axis=0).astype(np.float32)
    ba = np.stack([ctn.so3_derivative(bq[:, b].astype(np.float64), dt)
                   for b in range(bq.shape[1])], axis=1).astype(np.float32)

    out = {"fps": np.array([int(round(fps_out))], dtype=np.int64),
           "joint_pos": jp, "joint_vel": jv, "body_pos_w": bp, "body_quat_w": bq,
           "body_lin_vel_w": bl, "body_ang_vel_w": ba}
    return out, k_star, s


# ------------------------------------------------------------------ synthesize -- #
def acc_envelope(paths, joint_dim: int = 31) -> np.ndarray:
    """Per-joint empirical |acc| envelope over clips (audit口径: np.diff(joint_vel)*fps)."""
    env = np.zeros(joint_dim, dtype=np.float64)
    for p in paths:
        d = np.load(p)
        dq = np.asarray(d["joint_vel"], dtype=np.float64)
        fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        acc = np.abs(np.diff(dq, axis=0) * fps).max(axis=0)
        env = np.maximum(env, acc)
    return env


def synthesize(data: dict, phase: float, vlim: np.ndarray, acc_budget: np.ndarray,
               v_star: float | None = None, vel_limit_frac: float = 0.85,
               fps_out: float | None = None, min_cruise_s: float = 0.04,
               post_hold_s: float = 0.04, ta_grid_s: float = 0.005,
               dense_dt_s: float = 0.002, body_mode: str = "interp", fk_ctx=None):
    """Full synthesis: source npz dict -> (output npz dict, report dict)."""
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    unknown = [k for k in data.keys() if k not in KNOWN_KEYS and not k.startswith("_")]
    if unknown:
        raise SystemExit(f"unknown npz keys {unknown} — refusing to guess how to retime them")
    T_src, J = q.shape
    fps_src = float(np.asarray(data["fps"]).reshape(-1)[0])
    if fps_out is None:
        fps_out = fps_src
    c = contact_frame(phase, T_src)
    s_end = float(T_src - 1)
    if not (0 < c < T_src - 1):
        raise SystemExit(f"contact frame {c} of {T_src} leaves no run-up or follow-through")

    blade = blade_positions(data)
    dpds = blade_path_deriv_at(blade, c)                    # m / frame
    v_src_clean = clean_speed_at(blade, c, 1.0 / fps_src)   # m/s (source, training parity)
    if v_star is None:
        v_star = v_src_clean
    if dpds <= 1e-9:
        raise SystemExit("blade path derivative ~0 at the contact frame — bad phase?")
    sdot_star = float(v_star) / dpds                        # frames / s

    vel_cap = np.asarray(vlim, dtype=np.float64) * vel_limit_frac
    acc_budget = np.asarray(acc_budget, dtype=np.float64)
    if (acc_budget <= 0).any():
        # a joint the budget clips never moved: give it the smallest positive budget
        # (report notes it) instead of an impossible 0
        floor = acc_budget[acc_budget > 0].min() if (acc_budget > 0).any() else 1.0
        acc_budget = np.where(acc_budget <= 0, floor, acc_budget)

    qp = np.gradient(q, axis=0)          # dq/ds  [rad/frame]
    qpp = np.gradient(qp, axis=0)        # d²q/ds² [rad/frame²]

    law, info = solve_min_ta(c, s_end, sdot_star, qp, qpp, vel_cap, acc_budget,
                             fps_out, min_cruise_s, post_hold_s, ta_grid_s, dense_dt_s)
    out, k_star, s_grid = resample(data, law, fps_out, body_mode, fk_ctx)
    T_out = out["joint_pos"].shape[0]

    # ---- fidelity checks on the OUTPUT arrays ------------------------------------
    blade_out = blade_positions(out)
    v_out_clean = clean_speed_at(blade_out, k_star, 1.0 / fps_out)
    n_src = blade_face_normals(data)[c]
    n_out = blade_face_normals(out)[k_star]
    face_deg = float(np.degrees(np.arccos(np.clip(
        np.dot(n_src, n_out) / (np.linalg.norm(n_src) * np.linalg.norm(n_out)), -1, 1))))
    contact_bitwise = bool(np.array_equal(out["joint_pos"][k_star],
                                          np.asarray(data["joint_pos"])[c]))
    dq_out = np.asarray(out["joint_vel"], dtype=np.float64)
    first_frame_vel = float(np.abs(dq_out[0]).max())
    mean_acc_out = float(np.abs(np.diff(dq_out, axis=0) * fps_out).mean())
    dq_src = np.asarray(data["joint_vel"], dtype=np.float64)
    mean_acc_src = float(np.abs(np.diff(dq_src, axis=0) * fps_src).mean())

    verdict = "ok" if not info["irreducible"] else "budget_exceeded_irreducible"
    speed_dev = abs(v_out_clean - v_star) / v_star

    names = list(ISAAC_JOINT_NAMES) if J == len(ISAAC_JOINT_NAMES) else [f"j{i}" for i in range(J)]
    per_joint = [
        dict(joint=names[j],
             vel_util=round(float(info["vel_util"][j]), 4),
             acc_util=round(float(info["acc_util"][j]), 4),
             post_vel_util=round(float(info["post_vel_util"][j]), 4),
             post_acc_util=round(float(info["post_acc_util"][j]), 4),
             vel_cap=round(float(vel_cap[j]), 3),
             acc_budget=round(float(acc_budget[j]), 3))
        for j in range(J)
    ]

    report = dict(
        tool="synthesize_timing.py v1 (robot-centric time-law synthesis)",
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        verdict=verdict,
        source=dict(frames=int(T_src), fps=fps_src, contact_frame=int(c),
                    phase=float(phase), runup_s=round(c / fps_src, 4),
                    duration_s=round((T_src - 1) / fps_src, 4),
                    clean_blade_speed_mps=round(v_src_clean, 4),
                    mean_abs_acc=round(mean_acc_src, 3)),
        answer=dict(v_star_mps=round(float(v_star), 4),
                    v_star_source="source-clean" if abs(v_star - v_src_clean) < 1e-12 else "cli-override",
                    blade_dpds_m_per_frame=round(dpds, 6),
                    sdot_star_frames_per_s=round(sdot_star, 4)),
        time_law=dict(Ta_s=round(law.Ta, 4), ta_max_s=round(info["ta_max_s"], 4),
                      ta_grid_s=info["ta_grid_s"], wait_s=round(law.tw, 4),
                      cruise_pre_s=round(law.t_star - law.tw - law.Ta, 4),
                      post_hold_s=round(law.post_hold, 4), Td_s=round(law.Td, 4),
                      accel_frames_per_s2=round(law.A, 4),
                      decel_frames_per_s2=round(law.D, 4),
                      t_star_s=round(law.t_star, 4),
                      arc_length_reparam="not-applied (v1: s = frame index)"),
        output=dict(frames=int(T_out), fps=float(fps_out),
                    contact_frame=int(k_star),
                    phase_out=round(k_star / (T_out - 1), 6),
                    runup_s=round(k_star / fps_out, 4),
                    duration_s=round((T_out - 1) / fps_out, 4),
                    runup_change_x=round((k_star / fps_out) / (c / fps_src), 3),
                    duration_change_x=round(((T_out - 1) / fps_out) / ((T_src - 1) / fps_src), 3),
                    body_mode=body_mode, mean_abs_acc=round(mean_acc_out, 3)),
        fidelity=dict(contact_row_bitwise=contact_bitwise,
                      blade_speed_clean_out_mps=round(v_out_clean, 4),
                      blade_speed_dev_frac=round(speed_dev, 5),
                      face_normal_diff_deg=round(face_deg, 6),
                      first_frame_max_joint_vel=round(first_frame_vel, 4)),
        budgets=dict(vel_limit_frac=vel_limit_frac,
                     irreducible=info["irreducible"],
                     decel_over_budget=info["decel_over_budget"],
                     worst_vel_util=round(float(info["vel_util"].max()), 4),
                     worst_vel_joint=names[int(info["vel_util"].argmax())],
                     worst_acc_util=round(float(info["acc_util"].max()), 4),
                     worst_acc_joint=names[int(info["acc_util"].argmax())],
                     worst_post_acc_util=round(float(info["post_acc_util"].max()), 4),
                     worst_post_acc_joint=names[int(info["post_acc_util"].argmax())],
                     per_joint=per_joint),
    )
    return out, report


# ----------------------------------------------------------------------- report -- #
def report_md(rep: dict) -> str:
    s, a, tl, o, f, b = (rep["source"], rep["answer"], rep["time_law"],
                         rep["output"], rep["fidelity"], rep["budgets"])
    lines = [
        f"# Time-law synthesis report — **{rep['verdict']}**",
        "",
        f"- generated: {rep['generated_utc']}",
        f"- source: {s['frames']} frames @ {s['fps']:.0f} fps = {s['duration_s']:.2f} s; "
        f"contact f{s['contact_frame']} (phase {s['phase']}); run-up {s['runup_s']:.2f} s; "
        f"clean blade speed {s['clean_blade_speed_mps']:.3f} m/s; mean|acc| {s['mean_abs_acc']:.2f}",
        f"- answer speed |v*| = {a['v_star_mps']:.3f} m/s ({a['v_star_source']}); "
        f"|dp/ds|(c) = {a['blade_dpds_m_per_frame'] * 1000:.2f} mm/frame -> "
        f"sdot* = {a['sdot_star_frames_per_s']:.2f} frames/s",
        f"- time law: wait {tl['wait_s']:.3f} s | **T_a {tl['Ta_s']:.3f} s** "
        f"(scan max {tl['ta_max_s']:.3f} s) | cruise {tl['cruise_pre_s']:.3f} s | "
        f"post-hold {tl['post_hold_s']:.2f} s | T_d {tl['Td_s']:.3f} s; {tl['arc_length_reparam']}",
        f"- output: {o['frames']} frames @ {o['fps']:.0f} fps = {o['duration_s']:.2f} s "
        f"(x{o['duration_change_x']:.2f} vs source); contact f{o['contact_frame']} -> "
        f"**phase_out {o['phase_out']:.4f}**; run-up {o['runup_s']:.2f} s "
        f"(x{o['runup_change_x']:.2f}); body_mode={o['body_mode']}; mean|acc| {o['mean_abs_acc']:.2f}",
        f"- fidelity: contact row bitwise={f['contact_row_bitwise']}; blade speed out "
        f"{f['blade_speed_clean_out_mps']:.3f} m/s (dev {f['blade_speed_dev_frac'] * 100:.2f}%); "
        f"face diff {f['face_normal_diff_deg']:.4f} deg; frame-0 max|q̇| "
        f"{f['first_frame_max_joint_vel']:.3f} rad/s",
        f"- budgets: vel cap = {b['vel_limit_frac']:g} x URDF; PRE-contact worst vel util "
        f"{b['worst_vel_util']:.2f} ({b['worst_vel_joint']}); worst acc util "
        f"{b['worst_acc_util']:.2f} ({b['worst_acc_joint']}); irreducible={b['irreducible']}; "
        f"POST-contact (forced decel) worst acc util {b['worst_post_acc_util']:.2f} "
        f"({b['worst_post_acc_joint']}); decel_over_budget={b['decel_over_budget']}",
        "",
        "| joint | pre vel util | pre acc util | post vel | post acc | vel cap [rad/s] | acc budget [rad/s^2] |",
        "|---|---|---|---|---|---|---|",
    ]
    top = sorted(b["per_joint"], key=lambda r: -max(r["vel_util"], r["acc_util"],
                                                    r["post_vel_util"], r["post_acc_util"]))[:10]
    for r in top:
        lines.append(f"| {r['joint']} | {r['vel_util']:.2f} | {r['acc_util']:.2f} "
                     f"| {r['post_vel_util']:.2f} | {r['post_acc_util']:.2f} "
                     f"| {r['vel_cap']:.2f} | {r['acc_budget']:.2f} |")
    lines += [
        "",
        f"REGISTRY REMINDER: this asset's timeline is SYNTHESIZED — register phase_out = "
        f"{o['phase_out']:.4f} (contact frame {o['contact_frame']} of {o['frames']}) in "
        f"cfg/strike_annotations.yaml and in any strike_phase_per_clip training override. "
        f"The source video's frame convention does not apply (视频约定帧弃用).",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------------------- CLI -- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--phase", type=float, required=True,
                    help="registry contact phase of the SOURCE clip (strike_annotations.yaml)")
    ap.add_argument("--strike-speed", type=float, default=None,
                    help="answer blade speed |v*| m/s (default: the source's own clean blade "
                         "speed at the contact frame = 复现原速; override = 变速重解)")
    ap.add_argument("--budget-clips", nargs="+", required=True,
                    help="npz clips whose measured per-joint |acc| envelope defines the budget "
                         "(e.g. the v4rg pair — the proven-executable floor)")
    ap.add_argument("--budget-scale", type=float, default=1.5)
    ap.add_argument("--vel-limit-frac", type=float, default=0.85)
    ap.add_argument("--urdf", default=None,
                    help="A3 URDF with joint velocity limits (default: repo copy)")
    ap.add_argument("--fps-out", type=float, default=None, help="default: source fps")
    ap.add_argument("--min-cruise-s", type=float, default=0.04,
                    help="minimum constant-speed approach before contact (strike-window "
                         "protection for the clean-FD strike velocity)")
    ap.add_argument("--post-contact-hold-s", type=float, default=0.04,
                    help="constant-speed follow-through after contact before decel starts")
    ap.add_argument("--ta-grid-s", type=float, default=0.005)
    ap.add_argument("--dense-dt-s", type=float, default=0.002)
    ap.add_argument("--body-mode", choices=("fk", "interp"), default="fk",
                    help="fk = MuJoCo FK rebuild (PRODUCTION); interp = lerp/slerp of stored "
                         "body arrays (tests only)")
    ap.add_argument("--mjcf", default=None, help="deploy MJCF (required for --body-mode fk)")
    ap.add_argument("--body-order", default=None,
                    help="body-order file from csv_to_npz_mujoco --discover-map "
                         "(required for --body-mode fk)")
    ap.add_argument("--report", default=None, help="write the JSON report here")
    ap.add_argument("--md", default=None, help="write the markdown report here")
    args = ap.parse_args(argv)

    if args.urdf is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.urdf = os.path.normpath(os.path.join(
            here, "../../..", "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"))
    limits = parse_urdf_limits(args.urdf)
    vlim = np.array([limits[n].velocity if limits[n].velocity is not None else np.inf
                     for n in ISAAC_JOINT_NAMES])

    env = acc_envelope(args.budget_clips)
    acc_budget = env * args.budget_scale

    fk_ctx = None
    if args.body_mode == "fk":
        if not (args.mjcf and args.body_order):
            raise SystemExit("--body-mode fk requires --mjcf and --body-order")
        fkm = ctn.MjFK(args.mjcf, ISAAC_JOINT_NAMES)
        names = fkm.body_names()
        order = [ln.strip() for ln in open(args.body_order) if ln.strip()]
        fk_ctx = (fkm, [names.index(n) for n in order])

    data = dict(np.load(args.input))
    out, rep = synthesize(
        data, args.phase, vlim, acc_budget, v_star=args.strike_speed,
        vel_limit_frac=args.vel_limit_frac, fps_out=args.fps_out,
        min_cruise_s=args.min_cruise_s, post_hold_s=args.post_contact_hold_s,
        ta_grid_s=args.ta_grid_s, dense_dt_s=args.dense_dt_s,
        body_mode=args.body_mode, fk_ctx=fk_ctx)
    rep["files"] = dict(input=os.path.abspath(args.input),
                        output=os.path.abspath(args.output))
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
        print(f"** WARNING: contact blade-speed deviation {dev * 100:.2f}% > 2% acceptance **",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
