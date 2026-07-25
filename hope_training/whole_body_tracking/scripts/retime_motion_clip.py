#!/usr/bin/env python3
"""Non-uniform retiming (非均匀重定时) of a motion .npz — flatten the acceleration
profile WITHOUT touching the strike window.

人话:不做匀速慢放,而是"哪里猛就哪里放慢"。触球窗(登记相位 ±--window-s 秒)内
时间刻度锁死(恒等映射,窗内帧在新时间轴上逐位照抄——拍面/拍速零改动);窗外按
逐帧"运动强度"(全身 |ddq| 均值)做时间再分配:加速度大的段放慢、小的段可略压,
总时长最多拉长 --stretch-max(默认 1.3 = +30%)。目标给均值(--target-mean-acc,
二分找"最省时长的达标拉伸")或给峰值(--target-peak-acc)或直接定拉伸(--budget);
达不到目标就输出预算内最优,并在报告里写清"要达标还需拉到多少倍"。

WHY (franco 2026-07-09, NOW.md): uniform slow-play was rejected — it taxes the
healthy strike exactly as much as the sick out-of-window segments. A monotone time
reparameterization t -> tau stretching a segment locally by s scales its velocity
by 1/s and its acceleration by ~1/s^2, so spending stretch where |ddq| is large
flattens the profile at minimal total-duration cost. Fancy math yields to
verifiability: density-based reallocation + resample back to a uniform 50 fps grid.

GUARANTEES (asserted at build time, recorded in the report):
  * strike window: the pre-window time offset is rounded to an INTEGER number of
    output frames, so the new uniform grid lands EXACTLY on the original window
    frames; those rows are BITWISE copies of the source npz (joint_pos AND
    body_pos_w/body_quat_w) — contact-frame face normal / blade kinematics survive
    to the last bit. The tool refuses to ship if any window row is not bitwise.
  * monotone time map: every local stretch s is clipped to [--s-min, --s-max],
    s-min > 0, and asserted strictly increasing.
  * joint_vel is RE-DIFFERENTIATED (np.gradient on the new grid — the csv_to_npz
    convention); at the contact frame both central-difference neighbours are
    in-window bitwise rows, so the stored contact velocity reproduces the source
    pipeline's arithmetic (diff reported; expect float32 floor when the source
    joint_vel is itself gradient-consistent).
  * body_* outside the window: MuJoCo FK on the deploy MJCF (--mjcf +
    --body-order, the csv_to_npz_mujoco path) from the retimed joint_pos + the
    retimed root pose (body column 0); body velocities re-differentiated
    (np.gradient / SO3 central difference). Hosts without mujoco (unit tests):
    --body-mode interp lerps/slerps the stored body arrays — the report records
    the mode; PRODUCTION conversions must use --body-mode fk.
  * fail-closed on surprises: unknown npz keys whose leading axis is the frame
    axis refuse loudly (no silent guess about how to retime them).

NEW PHASE 口径: the printed phase_out is frame-content aligned BY CONSTRUCTION —
contact row = bitwise copy of the source contact frame at output index
K + (c - w0); phase_out = that index / (T_out - 1). No speed-peak convention
anywhere. Copy phase_out into cfg/strike_annotations.yaml when registering.

USAGE
    python scripts/retime_motion_clip.py \
        --input  hope_backhand_v5hLs_cal.npz --phase 0.391 \
        --output hope_backhand_v5hLt_cal.npz \
        --target-mean-acc 4.5 --stretch-max 1.3 \
        --mjcf .../a3_pingpong.xml --body-order /workspace/franco/body_order_isaac.txt \
        --report v5hLt_bh_retime.json --md v5hLt_bh_retime.md

DEPENDENCIES: numpy always; mujoco only for --body-mode fk (import deferred).
带 canonical-v2 sidecar 的输入默认拒收;显式 comparator flag 只允许生成永久标记的历史对照。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv_to_npz_mujoco as ctn  # noqa: E402  (numpy-only at import time)
import topp_mintime as legacy_v3  # noqa: E402  shared canonical-v2 legacy guard
from motion_kinematics_contract import (  # noqa: E402
    KINEMATICS_METADATA_KEYS,
    metadata_arrays,
    resolve_body_names,
)

KNOWN_KEYS = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
              "body_lin_vel_w", "body_ang_vel_w") + KINEMATICS_METADATA_KEYS


# ------------------------------------------------------------------ core math -- #
def contact_frame(phase: float, T: int) -> int:
    """Registry convention: contact frame = round(phase * (T-1))."""
    return int(round(float(phase) * (T - 1)))


def window_bounds(c: int, T: int, fps: float, window_s: float) -> tuple[int, int]:
    win_f = int(round(window_s * fps))
    return max(0, c - win_f), min(T - 1, c + win_f)


def audit_mean_acc(joint_vel: np.ndarray, fps: float) -> float:
    """EXACTLY the audit_motion_npz check-3 metric: mean |diff(joint_vel)*fps|."""
    return float(np.abs(np.diff(np.asarray(joint_vel, dtype=np.float64), axis=0) * fps).mean())


def audit_peak_acc(joint_vel: np.ndarray, fps: float) -> float:
    return float(np.abs(np.diff(np.asarray(joint_vel, dtype=np.float64), axis=0) * fps).max())


def intensity_profile(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    """Per-SEGMENT motion intensity (T-1,): whole-body mean |ddq| at the segment ends.

    Same flavour as the audit acc metric (mean over joints), so 'stretch where the
    metric is large' attacks the metric directly.
    """
    dt = 1.0 / fps
    q = np.asarray(joint_pos, dtype=np.float64)
    dq = np.gradient(q, dt, axis=0)
    ddq = np.gradient(dq, dt, axis=0)
    frame_int = np.abs(ddq).mean(axis=1)
    return 0.5 * (frame_int[:-1] + frame_int[1:])


def _smooth_edge_padded(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad_l = k // 2
    pad_r = k - 1 - pad_l
    xp = np.pad(x, (pad_l, pad_r), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def stretch_profile(seg_int: np.ndarray, a_ref: float, w0: int, w1: int,
                    s_min: float, s_max: float, smooth_frames: int,
                    ramp_frames: int) -> np.ndarray:
    """Per-segment stretch factors s (T-1,).

    s = sqrt(intensity / a_ref) (the 1/s^2 acceleration scaling inverted), clipped
    to [s_min, s_max], smoothed (edge-padded moving average), deviation-from-1
    cosine-tapered to 0 approaching the window, and EXACTLY 1.0 on the window
    segments [w0, w1-1].
    """
    s = np.sqrt(np.maximum(seg_int, 1e-12) / max(a_ref, 1e-12))
    s = np.clip(s, s_min, s_max)
    s = _smooth_edge_padded(s, smooth_frames)
    s = np.clip(s, s_min, s_max)  # smoothing cannot leave the sanctioned band
    dev = s - 1.0
    n = len(s)
    for i in range(n):
        if w0 <= i <= w1 - 1:
            dev[i] = 0.0
        elif ramp_frames > 0:
            d = (w0 - i) if i < w0 else (i - (w1 - 1))
            if d <= ramp_frames:
                dev[i] *= 0.5 * (1.0 - np.cos(np.pi * d / ramp_frames))
    s = 1.0 + dev
    # head/tail protection: compressing the first/last segment RAISES the frame-0 /
    # last-frame one-sided velocity (audit first-frame health) — never allow it.
    s[0] = max(s[0], 1.0)
    s[-1] = max(s[-1], 1.0)
    return s


def solve_stretch_for_budget(seg_int: np.ndarray, budget: float, w0: int, w1: int,
                             s_min: float, s_max: float, smooth_frames: int,
                             ramp_frames: int, iters: int = 60) -> np.ndarray:
    """Bisect a_ref so the mean stretch (= duration factor) hits `budget`.

    Monotone: larger a_ref -> smaller s -> shorter. Window segments are pinned at
    1.0, so the achievable factor is bounded; the closest achievable is returned.
    budget <= 1.0 with no compression allowed is BY DEFINITION the identity map —
    return it exactly (bisection would only approach s == 1 asymptotically and
    leave float dust on every resampled row).
    """
    if budget <= 1.0 and s_min >= 1.0:
        return np.ones(len(seg_int))
    lo, hi = 1e-6, 1e6  # a_ref bracket
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        s = stretch_profile(seg_int, mid, w0, w1, s_min, s_max, smooth_frames, ramp_frames)
        if s.mean() > budget:
            lo = mid      # too long -> raise a_ref
        else:
            hi = mid
    return stretch_profile(seg_int, np.sqrt(lo * hi), w0, w1, s_min, s_max,
                           smooth_frames, ramp_frames)


def build_time_map(s: np.ndarray, w0: int, fps: float) -> tuple[np.ndarray, int, float]:
    """Cumulative new-time knots tau (T,) with window grid alignment.

    Scales the pre-window stretches uniformly so the window start lands on an
    INTEGER output-frame index K — the whole window then lies exactly on the new
    uniform grid (window segments are 1.0 by construction).
    Returns (tau_seconds, K, pre_align_scale).
    """
    dt = 1.0 / fps
    s = np.asarray(s, dtype=np.float64).copy()
    if w0 > 0:
        pre = float(s[:w0].sum())
        K = max(1, int(round(pre)))
        scale = K / pre
        assert 0.7 < scale < 1.4, f"pre-window alignment scale {scale:.3f} out of sane range"
        s[:w0] *= scale
    else:
        K, scale = 0, 1.0
    tau = np.concatenate([[0.0], np.cumsum(s)]) * dt
    assert np.all(np.diff(tau) > 0.0), "time map must be strictly monotone"
    return tau, K, scale


def resample_fractional_index(tau: np.ndarray, fps: float,
                              snap_tol: float = 1e-6) -> np.ndarray:
    """Fractional SOURCE index for every output frame k at u_k = k/fps.

    Near-integer indices are snapped exactly so that grid-aligned frames become
    bitwise row copies downstream.
    """
    dt = 1.0 / fps
    n_out = int(np.floor(tau[-1] / dt + 1e-9)) + 1
    u = np.arange(n_out, dtype=np.float64) * dt
    phi = np.interp(u, tau, np.arange(len(tau), dtype=np.float64))
    r = np.round(phi)
    snap = np.abs(phi - r) < snap_tol
    phi[snap] = r[snap]
    return phi


def lerp_rows(x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Linear row interpolation; integer phi rows are BITWISE source copies."""
    x = np.asarray(x)
    i0 = np.floor(phi).astype(int)
    i1 = np.minimum(i0 + 1, x.shape[0] - 1)
    b = (phi - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    out = (x[i0].astype(np.float64) * (1.0 - b) + x[i1].astype(np.float64) * b).astype(x.dtype)
    exact = phi == np.floor(phi)
    out[exact] = x[i0[exact]]
    return out


def slerp_rows(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Quaternion slerp per row (wxyz, any (..., 4) trailing shape); integer phi
    rows are BITWISE source copies."""
    q = np.asarray(q)
    i0 = np.floor(phi).astype(int)
    i1 = np.minimum(i0 + 1, q.shape[0] - 1)
    b = (phi - i0).reshape((-1,) + (1,) * (q.ndim - 1)).astype(np.float64)
    out = ctn.quat_slerp(q[i0].astype(np.float64), q[i1].astype(np.float64), b).astype(q.dtype)
    exact = phi == np.floor(phi)
    out[exact] = q[i0[exact]]
    return out


# ------------------------------------------------------------------ pipeline --- #
def retime_joint_space(joint_pos: np.ndarray, fps: float, phase: float,
                       window_s: float, budget: float, s_min: float, s_max: float,
                       smooth_frames: int, ramp_frames: int):
    """One full joint-space retime at a fixed duration budget.

    Returns dict with phi, tau, K, scale, new joint_pos/joint_vel, window/contact
    indices (in and out), and the audit-style acc numbers.
    """
    T = joint_pos.shape[0]
    c = contact_frame(phase, T)
    w0, w1 = window_bounds(c, T, fps, window_s)
    seg_int = intensity_profile(joint_pos, fps)
    s = solve_stretch_for_budget(seg_int, budget, w0, w1, s_min, s_max,
                                 smooth_frames, ramp_frames)
    tau, K, scale = build_time_map(s, w0, fps)
    phi = resample_fractional_index(tau, fps)
    # window rows MUST be grid-aligned: output k = K + (i - w0) <-> source i
    for i in range(w0, w1 + 1):
        k = K + (i - w0)
        assert k < len(phi) and phi[k] == float(i), (
            f"window frame {i} not grid-aligned (k={k}, phi={phi[k] if k < len(phi) else 'OOB'})")
    q_new = lerp_rows(joint_pos, phi)
    # bitwise window guarantee (lerp_rows already copies exact rows; enforce + verify)
    q_new[K:K + (w1 - w0) + 1] = joint_pos[w0:w1 + 1]
    dq_new = np.gradient(q_new.astype(np.float64), 1.0 / fps, axis=0).astype(np.float32)
    k_c = K + (c - w0)
    return {
        "T_in": T, "T_out": q_new.shape[0], "c_in": c, "c_out": k_c,
        "w0": w0, "w1": w1, "K": K, "pre_align_scale": scale,
        "s": s, "tau": tau, "phi": phi,
        "joint_pos": q_new, "joint_vel": dq_new,
        "phase_out": k_c / (q_new.shape[0] - 1),
    }


def search_min_budget(joint_pos, fps, phase, window_s, s_min, s_max, smooth_frames,
                      ramp_frames, target_value, metric, stretch_max,
                      probe_max=3.0, tol=5e-3):
    """Smallest duration budget in [1.0, stretch_max] whose retime meets the target.

    metric: 'mean' (audit mean|acc|) or 'peak' (audit peak |acc|), evaluated on the
    RE-DIFFERENTIATED joint_vel of the actual retimed result (no approximation).
    Returns (budget_used, result_dict, attained, needed_budget_or_None):
    if even stretch_max misses the target, result is at stretch_max and
    needed_budget reports the bisected budget up to probe_max that WOULD attain it
    (None if not even probe_max suffices).
    """
    def value_at(budget):
        r = retime_joint_space(joint_pos, fps, phase, window_s, budget, s_min,
                               s_max, smooth_frames, ramp_frames)
        v = (audit_mean_acc if metric == "mean" else audit_peak_acc)(r["joint_vel"], fps)
        return v, r

    v_max, r_max = value_at(stretch_max)
    if v_max > target_value:  # not attainable inside the sanctioned budget
        needed = None
        v_probe, _ = value_at(probe_max)
        if v_probe <= target_value:
            lo, hi = stretch_max, probe_max
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                v, _ = value_at(mid)
                if v > target_value:
                    lo = mid
                else:
                    hi = mid
            needed = hi
        return stretch_max, r_max, False, needed
    lo, hi = 1.0, stretch_max
    v_lo, r_lo = value_at(lo)
    if v_lo <= target_value:
        return lo, r_lo, True, None
    r_hi = r_max
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        v, r = value_at(mid)
        if v > target_value:
            lo = mid
        else:
            hi, r_hi = mid, r
        if hi - lo < tol:
            break
    return hi, r_hi, True, None


def rebuild_bodies_fk(mjcf: str, body_order_file: str, base_pos, base_quat, joint_pos):
    """Return link poses and COM positions in the stored body-column order."""
    from audit_motion_npz import ISAAC_JOINT_NAMES  # same-dir import, numpy-only
    fkm = ctn.MjFK(mjcf, ISAAC_JOINT_NAMES)
    names = fkm.body_names()
    with open(body_order_file) as fh:
        body_order = [ln.strip() for ln in fh if ln.strip()]
    cols = [names.index(n) for n in body_order]
    pos_all, quat_all, com_all = ctn.fk_series_with_com(
        fkm,
        base_pos.astype(np.float64),
        base_quat.astype(np.float64),
        joint_pos.astype(np.float64),
        ISAAC_JOINT_NAMES,
    )
    return pos_all[:, cols], quat_all[:, cols], com_all[:, cols]


def read_body_order(path: str | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    with open(path) as fh:
        names = tuple(line.strip() for line in fh if line.strip())
    if not names or len(set(names)) != len(names):
        raise SystemExit("[retime] --body-order must contain unique non-empty names")
    return names


def retime_clip(data: dict, phase: float, window_s: float, s_min: float, s_max: float,
                smooth_frames: int, ramp_frames: int, body_mode: str,
                mjcf: str | None, body_order: str | None,
                budget: float | None, target_mean_acc: float | None,
                target_peak_acc: float | None, stretch_max: float):
    """Full retime of a standard motion npz dict. Returns (out_dict, report_dict)."""
    fps = float(np.array(data["fps"]).reshape(-1)[0])
    q = data["joint_pos"]
    T = q.shape[0]
    for key in data:
        if key in KNOWN_KEYS:
            continue
        arr = np.asarray(data[key])
        if arr.ndim >= 1 and arr.shape[0] == T:
            raise SystemExit(f"[retime] REFUSING: unknown time-axis key '{key}' "
                             f"(shape {arr.shape}) — teach the tool how to retime it first")

    targets = [t for t in (budget, target_mean_acc, target_peak_acc) if t is not None]
    if len(targets) != 1:
        raise SystemExit("[retime] give exactly ONE of --budget / --target-mean-acc / --target-peak-acc")

    if budget is not None:
        res = retime_joint_space(q, fps, phase, window_s, budget, s_min, s_max,
                                 smooth_frames, ramp_frames)
        budget_used, attained, needed = budget, True, None
        mode = f"fixed budget {budget:g}"
    else:
        metric = "mean" if target_mean_acc is not None else "peak"
        tval = target_mean_acc if target_mean_acc is not None else target_peak_acc
        budget_used, res, attained, needed = search_min_budget(
            q, fps, phase, window_s, s_min, s_max, smooth_frames, ramp_frames,
            tval, metric, stretch_max)
        mode = f"min budget for {metric}|acc| <= {tval:g} (cap {stretch_max:g})"

    phi, K, w0, w1 = res["phi"], res["K"], res["w0"], res["w1"]
    n_out = res["T_out"]
    win_out = slice(K, K + (w1 - w0) + 1)
    dt = 1.0 / fps

    # --- root pose (body col 0) on the new grid ------------------------------ #
    base_pos = lerp_rows(data["body_pos_w"][:, 0], phi)
    base_quat = slerp_rows(data["body_quat_w"][:, 0], phi)
    base_pos[win_out] = data["body_pos_w"][w0:w1 + 1, 0]
    base_quat[win_out] = data["body_quat_w"][w0:w1 + 1, 0]

    # --- bodies --------------------------------------------------------------- #
    fk_window_pos_diff = fk_window_quat_diff = None
    if body_mode == "fk":
        if not (mjcf and body_order):
            raise SystemExit("[retime] --body-mode fk requires --mjcf and --body-order")
        body_pos, body_quat, body_com = rebuild_bodies_fk(
            mjcf, body_order, base_pos, base_quat, res["joint_pos"]
        )
        body_pos = body_pos.astype(np.float32)
        body_quat = body_quat.astype(np.float32)
        velocity_path = body_com.astype(np.float64)
        velocity_point = "center_of_mass"
        # FK of bitwise window inputs must reproduce the stored rows (float32 floor)
        fk_window_pos_diff = float(np.abs(body_pos[win_out] - data["body_pos_w"][w0:w1 + 1]).max())
        qa, qb = body_quat[win_out], data["body_quat_w"][w0:w1 + 1]
        fk_window_quat_diff = float(np.minimum(np.abs(qa - qb), np.abs(qa + qb)).max())
    elif body_mode == "interp":
        body_pos = lerp_rows(data["body_pos_w"], phi)
        body_quat = slerp_rows(data["body_quat_w"], phi)
        # Interpolating stored link poses cannot reconstruct a COM path without
        # the MJCF inertial offsets.  Mark this diagnostic-only output honestly;
        # MotionLoader rejects it for formal training.
        velocity_path = body_pos.astype(np.float64)
        velocity_point = "link_origin"
    else:
        raise SystemExit(f"[retime] unknown --body-mode {body_mode}")
    # window rows: bitwise from source, whatever the body mode
    body_pos[win_out] = data["body_pos_w"][w0:w1 + 1]
    body_quat[win_out] = data["body_quat_w"][w0:w1 + 1]

    body_lin = np.gradient(velocity_path, dt, axis=0).astype(np.float32)
    body_ang = np.stack([ctn.so3_derivative(body_quat[:, b].astype(np.float64), dt)
                         for b in range(body_quat.shape[1])], axis=1).astype(np.float32)

    explicit_body_names = read_body_order(body_order)
    try:
        body_names = resolve_body_names(
            data,
            explicit_body_names=explicit_body_names,
            expected_count=body_pos.shape[1],
        )
    except ValueError as exc:
        raise SystemExit(f"[retime] {exc}") from None

    out = {
        "fps": np.array(data["fps"]).reshape(-1).astype(np.int64),
        "joint_pos": res["joint_pos"].astype(np.float32),
        "joint_vel": res["joint_vel"].astype(np.float32),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
    }
    out.update(metadata_arrays(
        body_names=body_names, body_lin_vel_point=velocity_point
    ))

    # --- verification --------------------------------------------------------- #
    assert np.array_equal(out["joint_pos"][win_out], q[w0:w1 + 1]), "window joint_pos not bitwise"
    assert np.array_equal(out["body_pos_w"][win_out], data["body_pos_w"][w0:w1 + 1])
    assert np.array_equal(out["body_quat_w"][win_out], data["body_quat_w"][w0:w1 + 1])
    c_in, c_out = res["c_in"], res["c_out"]
    assert contact_frame(res["phase_out"], n_out) == c_out, "phase_out does not round-trip"

    dq_in = data["joint_vel"]
    seg_all_in = np.abs(np.diff(np.asarray(dq_in, np.float64), axis=0) * fps)
    seg_all_out = np.abs(np.diff(np.asarray(out["joint_vel"], np.float64), axis=0) * fps)
    win_seg_in = seg_all_in[max(0, w0 - 1):w1 + 1]
    out_of_win_in = np.delete(seg_all_in, np.s_[max(0, w0 - 1):w1 + 1], axis=0)
    out_of_win_out = np.delete(seg_all_out, np.s_[max(0, K - 1):K + (w1 - w0) + 1], axis=0)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode, "budget_used": float(budget_used),
        "target_attained": bool(attained),
        "needed_budget_for_target": None if needed is None else float(needed),
        "fps": fps, "window_s": window_s,
        "frames": {"T_in": T, "T_out": n_out, "contact_in": c_in, "contact_out": c_out,
                   "window_in": [w0, w1], "window_out": [K, K + (w1 - w0)],
                   "K_align": K, "pre_align_scale": float(res["pre_align_scale"])},
        "duration_s": {"in": (T - 1) * dt, "out": (n_out - 1) * dt,
                       "stretch_total": float(res["tau"][-1] / ((T - 1) * dt)),
                       "tail_drop_ms": float((res["tau"][-1] - (n_out - 1) * dt) * 1000.0)},
        "phase": {"in": float(phase), "out": float(res["phase_out"]),
                  "convention": "frame-content aligned: contact row is a bitwise copy "
                                "of the source contact frame; phase_out = c_out/(T_out-1)"},
        "stretch_stats": {"min": float(res["s"].min()), "max": float(res["s"].max()),
                          "mean": float(res["s"].mean())},
        "acc_rad_s2": {
            "mean_in": audit_mean_acc(dq_in, fps),
            "mean_out": audit_mean_acc(out["joint_vel"], fps),
            "peak_in": audit_peak_acc(dq_in, fps),
            "peak_out": audit_peak_acc(out["joint_vel"], fps),
            "window_peak_in": float(win_seg_in.max()) if win_seg_in.size else None,
            "out_of_window_peak_in": float(out_of_win_in.max()) if out_of_win_in.size else None,
            "out_of_window_peak_out": float(out_of_win_out.max()) if out_of_win_out.size else None,
        },
        "first_frame": {
            "max_joint_vel_in": float(np.abs(dq_in[0]).max()),
            "max_joint_vel_out": float(np.abs(out["joint_vel"][0]).max()),
            "base_speed_in": float(np.linalg.norm(np.asarray(data["body_lin_vel_w"][0, 0], np.float64))),
            "base_speed_out": float(np.linalg.norm(np.asarray(body_lin[0, 0], np.float64))),
        },
        "fidelity": {
            "window_joint_pos_bitwise": True,
            "window_body_rows_bitwise": True,
            "body_mode": body_mode,
            "body_lin_vel_point": velocity_point,
            "motion_command_exact": bool(velocity_point == "center_of_mass"),
            "fk_window_max_pos_diff_m": fk_window_pos_diff,
            "fk_window_max_quat_diff": fk_window_quat_diff,
            "contact_joint_vel_max_abs_diff": float(
                np.abs(np.asarray(out["joint_vel"][c_out], np.float64)
                       - np.asarray(dq_in[c_in], np.float64)).max()),
        },
    }
    return out, report


# ---------------------------------------------------------------------- CLI ---- #
def _md_report(rep: dict, inp: str, outp: str) -> str:
    a, f, d, ph = rep["acc_rad_s2"], rep["frames"], rep["duration_s"], rep["phase"]
    lines = [
        f"# retime report — {os.path.basename(outp)}",
        "",
        f"- 人话:{os.path.basename(inp)} 非均匀重定时(触球窗锁死、窗外按 |ddq| 强度再分配),"
        f"总时长 {d['in']:.2f}s -> {d['out']:.2f}s(x{d['stretch_total']:.3f});"
        f"mean|acc| {a['mean_in']:.2f} -> {a['mean_out']:.2f} rad/s²,"
        f"峰值 {a['peak_in']:.1f} -> {a['peak_out']:.1f} rad/s²。",
        f"- mode: {rep['mode']} | budget used: {rep['budget_used']:.4g} | "
        f"target attained: {rep['target_attained']}"
        + ("" if rep["needed_budget_for_target"] is None
           else f" | NEEDED budget ~x{rep['needed_budget_for_target']:.3f}"),
        f"- frames: T {f['T_in']} -> {f['T_out']}; contact {f['contact_in']} -> {f['contact_out']}; "
        f"window in {f['window_in']} -> out {f['window_out']} (K={f['K_align']}, "
        f"pre-align x{f['pre_align_scale']:.4f})",
        f"- phase: {ph['in']:.3f} -> **{ph['out']:.6f}** ({ph['convention']})",
        f"- out-of-window peak|acc|: {a['out_of_window_peak_in']:.1f} -> "
        f"{a['out_of_window_peak_out']:.1f} rad/s² (window peak in source: {a['window_peak_in']:.1f})",
        f"- first frame: max|joint_vel| {rep['first_frame']['max_joint_vel_in']:.3f} -> "
        f"{rep['first_frame']['max_joint_vel_out']:.3f} rad/s",
        f"- fidelity: window joint_pos/body rows bitwise; body_mode={rep['fidelity']['body_mode']}; "
        f"contact joint_vel max|Δ|={rep['fidelity']['contact_joint_vel_max_abs_diff']:.2e} rad/s"
        + ("" if rep["fidelity"]["fk_window_max_pos_diff_m"] is None
           else f"; FK-vs-stored window: pos {rep['fidelity']['fk_window_max_pos_diff_m']:.2e} m, "
                f"quat {rep['fidelity']['fk_window_max_quat_diff']:.2e}"),
        f"- stretch s: min {rep['stretch_stats']['min']:.3f} / mean {rep['stretch_stats']['mean']:.3f} "
        f"/ max {rep['stretch_stats']['max']:.3f}; tail drop {d['tail_drop_ms']:.1f} ms",
        "",
    ]
    if rep.get("publication_class") == legacy_v3.LEGACY_COMPARATOR_CLASS:
        lines[2:2] = [
            f"- **publication_class: `{legacy_v3.LEGACY_COMPARATOR_CLASS}`**",
            "- training_authorized: false | deployment_authorized: false | "
            "hardware_authorized: false",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--phase", type=float, required=True,
                    help="registry contact phase of --input (frame = round(phase*(T-1)))")
    ap.add_argument("--window-s", type=float, default=0.1)
    ap.add_argument("--budget", type=float, default=None,
                    help="fixed duration factor (e.g. 1.2); mutually exclusive with targets")
    ap.add_argument("--target-mean-acc", type=float, default=None,
                    help="find the SMALLEST budget <= --stretch-max with mean|acc| <= this")
    ap.add_argument("--target-peak-acc", type=float, default=None)
    ap.add_argument("--stretch-max", type=float, default=1.3)
    ap.add_argument("--s-min", type=float, default=1.0,
                    help="local compression floor; <1.0 is opt-in ('低的段可略压') — "
                         "note compression raises velocities (URDF vel-limit risk)")
    ap.add_argument("--s-max", type=float, default=3.0)
    ap.add_argument("--smooth-frames", type=int, default=5)
    ap.add_argument("--ramp-frames", type=int, default=4)
    ap.add_argument("--body-mode", choices=("fk", "interp"), default="fk")
    ap.add_argument("--mjcf")
    ap.add_argument("--body-order")
    ap.add_argument("--report", help="write the JSON report here")
    ap.add_argument("--md", help="write the markdown report here")
    ap.add_argument(
        "--allow-canonical-v2-legacy-comparator",
        action="store_true",
        help=(
            "仅历史比较:允许 canonical-v2/comparator 输入进入冻结窗口旧语义;"
            "JSON report 强制标为 legacy_window_frozen_comparator_only"
        ),
    )
    args = ap.parse_args(argv)

    legacy_v3._refuse_output_aliases_inputs(
        [args.input],
        [
            args.output,
            *([args.report] if args.report else []),
            *([args.md] if args.md else []),
        ],
    )
    legacy_comparator = legacy_v3._legacy_comparator_guard(
        [("input clip", args.input)],
        allow_canonical_v2_legacy_comparator=(
            args.allow_canonical_v2_legacy_comparator
        ),
    )
    if legacy_comparator is not None and not args.report:
        raise SystemExit(
            "canonical-v2 legacy comparator 必须给 --report，禁止生成无机器降级标记的资产"
        )

    data = dict(np.load(args.input, allow_pickle=False))
    out, rep = retime_clip(
        data, phase=args.phase, window_s=args.window_s, s_min=args.s_min,
        s_max=args.s_max, smooth_frames=args.smooth_frames, ramp_frames=args.ramp_frames,
        body_mode=args.body_mode, mjcf=args.mjcf, body_order=args.body_order,
        budget=args.budget, target_mean_acc=args.target_mean_acc,
        target_peak_acc=args.target_peak_acc, stretch_max=args.stretch_max)
    legacy_v3._mark_legacy_comparator(rep, legacy_comparator)
    rep["input"], rep["output"] = args.input, args.output
    np.savez(args.output, **out)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(rep, fh, indent=2, allow_nan=False)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(_md_report(rep, args.input, args.output))
    a, d = rep["acc_rad_s2"], rep["duration_s"]
    print(f"[retime] {args.input} -> {args.output}: T {rep['frames']['T_in']} -> "
          f"{rep['frames']['T_out']} @ {rep['fps']:g} Hz (x{d['stretch_total']:.3f})")
    print(f"[retime] mean|acc| {a['mean_in']:.2f} -> {a['mean_out']:.2f} rad/s^2; "
          f"peak {a['peak_in']:.1f} -> {a['peak_out']:.1f}; target attained: {rep['target_attained']}"
          + ("" if rep["needed_budget_for_target"] is None
             else f" (needed ~x{rep['needed_budget_for_target']:.3f})"))
    print(f"[retime] NEW PHASE = {rep['phase']['out']:.6f} (contact frame "
          f"{rep['frames']['contact_out']} of {rep['frames']['T_out']}; frame-content aligned "
          f"by construction — window rows are bitwise source copies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
