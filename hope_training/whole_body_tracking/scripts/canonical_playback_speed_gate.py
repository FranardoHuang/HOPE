#!/usr/bin/env python3
"""慢放可行性门(playback-ratio feasibility gate):检查 q(t) -> q(st) 是否真的在包络里。

人话:任务要一个更慢的挥拍时,不必为每个速度重解一条新片段——把最快的那条按
s = 目标速/最高速 慢放就行。运行时**已经**在做这件事(mdp/commands.py 的 R14
分数时钟:time_steps_f += s,joint_vel/body_vel 乘 s,位置不动),本工具不重做它。
本工具做的是那件没人做的事:**在离线端把「这个 s 到底可不可行」验出来**,不可行
就大声拒绝,并据此把不必要的逐速度烤入片段砍掉。

为什么需要验,而不是"慢放当然更安全":
    路径 q 固定、时间按 1/s 缩放 ⇒ qd -> s·qd,qdd -> s²·qdd,于是逆动力学
        tau(s) = c0(q) + s·L(q,qd) + s²·Q(q,qd,qdd)
    c0 = 重力 + 纯位形偏置,**完全不随 s 缩放**。所以:
      * 一个自身重力力矩就超限的位形,慢放到 s->0 也救不回来(反而暴露成静态问题);
      * L(阻尼)项让 tau(s) 是抛物线,可能在区间**内部**取到比两端更大的值
        (顶点 s* = -L/(2Q)),只查端点会漏;
      * 有接触力(floating base / full scope)时这个二次律**根本不成立**——本工具
        用独立探针实测残差,残差超限就 fail loud,绝不假装认证。

三道筛子(每道都给自己的 max 可行 r,取最小值为准):
    ① kinematic_envelope: |qd|·s ≤ 速度限位,|qdd|·s² ≤ 加速度包络(纯运动学,
       与 bake 工具 kin_gate 同口径)。
    ② torque_quadratic: 上面的 c0 + sL + s²Q,逐帧逐关节在 [0, r] 上取闭式极值
       (含内部顶点),对 generalized-force 限位判卷;并用独立探针核对二次律本身。
    ③ ground_contact: 接地/浮基支撑筛(CoP/支撑多边形)。**本工具不实现**;
       support_mode 不是 fixed_base 时直接 IncompletePlaybackCertificate,
       拒绝只凭关节力矩给 full scope 发证。

用法::

    # 判一个 clip 的某个慢放比(需要 mjcf 才有力矩筛;没有就是 kinematic_only 半证)
    python canonical_playback_speed_gate.py --certify --clip fh.npz --ratio 0.8 \
        --budget-clips fh.npz bh.npz --mjcf a3_pingpong.xml \
        --certificate fh_playback_0p80.json

    # 用证书把"梯子上哪几档还需要烤片段"算出来(纯 CPU,不需要 mujoco)
    python canonical_playback_speed_gate.py --plan \
        --certificate fh_playback_0p80.json --ladder 0.65,0.8,0.9,1.0,1.1,1.2,1.35

退出码:0 可行;1 合同/参数错误(fail loud);3 physical-infeasible(证书已写);
        4 incomplete —— 必需的筛子没跑(如没给 --mjcf),**没有做出任何可行性主张**,
        这张半证不可用于授权慢放。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import canonical_torque_path_topp as ctp  # noqa: E402  DirectActuatorContract / 限位口径

EXIT_INFEASIBLE = 3
EXIT_INCOMPLETE = 4

#: 运行时慢放只允许 s <= 1(round() 时钟在 s>1 会跳过资产帧,可能正好跳过触球帧)。
RUNTIME_PLAYBACK_MAX_RATIO = 1.0

#: 二次律辨识用的探针(独立于辨识点 0/1/2),用来证明 tau(s) 真的是二次的。
DEFAULT_LAW_PROBE_RATIOS = (0.3, 0.55, 1.4)
DEFAULT_LAW_RELATIVE_TOLERANCE = 1.0e-8
DEFAULT_LAW_ABSOLUTE_TOLERANCE = 1.0e-9

InverseDynamics = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class PlaybackGateError(ValueError):
    """慢放门的输入/合同错误(fail loud,不给判卷结论)。"""


class PlaybackLawViolation(PlaybackGateError):
    """tau(s) 不是 c0 + sL + s²Q —— 有接触力等非二次项,拒绝据此发证。"""


class IncompletePlaybackCertificate(PlaybackGateError):
    """缺必需的物理筛子;**没有做出任何可行性主张**。"""

    def __init__(self, reason: str, *, missing: Sequence[str]) -> None:
        super().__init__(reason)
        self.report = {
            "status": "INCOMPLETE_FAIL_CLOSED",
            "reason": str(reason),
            "missing": [str(item) for item in missing],
        }


class PlaybackRatioInfeasible(PlaybackGateError):
    """请求的慢放比越出包络。"""

    def __init__(self, reason: str, certificate: Dict[str, Any]) -> None:
        super().__init__(reason)
        self.certificate = certificate


# ====================================================================================== #
# 数值内核:抛物线在区间上的闭式极值 + 首次出界比                                          #
# ====================================================================================== #
def _finite_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise PlaybackGateError(f"{name} contains non-finite values")
    return array


def quadratic_interval_extrema(
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    lo: float,
    hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """f(s) = c0 + c1·s + c2·s² 在 [lo, hi] 上的逐元素 (min, max, argmin, argmax)。

    端点 **加内部顶点** —— 只查端点会漏掉阻尼项造成的内部极值(实测占 16-35% 的
    (帧,关节) 条目),这正是 mutation 测试盯的那一行。
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
        raise PlaybackGateError(f"bad ratio interval [{lo}, {hi}]")
    c0 = _finite_array("c0", c0)
    c1 = _finite_array("c1", c1)
    c2 = _finite_array("c2", c2)

    def value_at(s):
        s = np.asarray(s, dtype=np.float64)
        return c0 + c1 * s + c2 * s * s

    f_lo = value_at(lo)
    f_hi = value_at(hi)
    best_max = np.maximum(f_lo, f_hi)
    best_min = np.minimum(f_lo, f_hi)
    arg_max = np.where(f_hi >= f_lo, hi, lo)
    arg_min = np.where(f_hi <= f_lo, hi, lo)

    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = np.where(c2 != 0.0, -c1 / (2.0 * np.where(c2 != 0.0, c2, 1.0)), np.nan)
    inside = np.isfinite(vertex) & (vertex > lo) & (vertex < hi)
    if np.any(inside):
        safe_vertex = np.where(inside, vertex, lo)
        f_vertex = value_at(safe_vertex)
        take_max = inside & (f_vertex > best_max)
        best_max = np.where(take_max, f_vertex, best_max)
        arg_max = np.where(take_max, safe_vertex, arg_max)
        take_min = inside & (f_vertex < best_min)
        best_min = np.where(take_min, f_vertex, best_min)
        arg_min = np.where(take_min, safe_vertex, arg_min)
    return best_min, best_max, arg_min, arg_max


def _smallest_positive_root(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """a·s² + b·s + c = 0 的最小正实根(无正实根 = +inf),数值稳定式。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    a, b, c = np.broadcast_arrays(a, b, c)
    disc = b * b - 4.0 * a * c
    real = disc >= 0.0
    sqrt_disc = np.sqrt(np.where(real, disc, 0.0))
    sign_b = np.where(b >= 0.0, 1.0, -1.0)
    q = -0.5 * (b + sign_b * sqrt_disc)
    with np.errstate(divide="ignore", invalid="ignore"):
        root_a = np.where(a != 0.0, q / np.where(a != 0.0, a, 1.0), np.inf)
        root_b = np.where(q != 0.0, c / np.where(q != 0.0, q, 1.0), np.inf)
    # q == 0 且 a != 0 ⇒ b == 0 且 c == 0 ⇒ 双重根 0(已经贴在边界上)。
    degenerate = (q == 0.0) & (a != 0.0)
    root_b = np.where(degenerate, 0.0, root_b)
    root_a = np.where(real, root_a, np.inf)
    root_b = np.where(real, root_b, np.inf)
    positive_a = np.where(root_a > 0.0, root_a, np.inf)
    positive_b = np.where(root_b > 0.0, root_b, np.inf)
    smallest = np.minimum(positive_a, positive_b)
    # 已经在 s=0 越界(c 与边界同号越出)的条目由调用方的静态筛报告,这里给 0。
    return np.where(np.isnan(smallest), np.inf, smallest)


def first_exit_ratio(
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    limit_lower: np.ndarray,
    limit_upper: np.ndarray,
) -> np.ndarray:
    """逐元素:从 s=0 出发,f(s) 第一次越出 [limit_lower, limit_upper] 的 s(不越 = inf)。

    s=0 时就在界外的条目返回 0.0 —— 那是 c0(重力)自己超限,任何慢放都救不了。
    """
    c0 = _finite_array("c0", c0)
    c1 = _finite_array("c1", c1)
    c2 = _finite_array("c2", c2)
    limit_lower = _finite_array("limit_lower", limit_lower)
    limit_upper = _finite_array("limit_upper", limit_upper)
    if np.any(limit_lower >= limit_upper):
        raise PlaybackGateError("generalized-force limits must satisfy lower < upper")
    already_out = (c0 > limit_upper) | (c0 < limit_lower)
    exit_upper = _smallest_positive_root(c2, c1, c0 - limit_upper)
    exit_lower = _smallest_positive_root(c2, c1, c0 - limit_lower)
    exit_ratio = np.minimum(exit_upper, exit_lower)
    return np.where(already_out, 0.0, exit_ratio)


def _utilisation(minimum: np.ndarray, maximum: np.ndarray,
                 limit_lower: np.ndarray, limit_upper: np.ndarray) -> np.ndarray:
    """占比:>1 = 越界。上界用 max/upper,下界用 min/lower(lower<0<upper)。"""
    return np.maximum(maximum / limit_upper, minimum / limit_lower)


# ====================================================================================== #
# 筛子 ①:纯运动学包络(不需要 mujoco)                                                    #
# ====================================================================================== #
@dataclass(frozen=True)
class KinematicScreenInputs:
    """慢放的运动学筛输入(与 bake_topp_strike_speed.kin_gate 同口径)。"""

    joint_vel: np.ndarray             # (T, J) 参考片段的原生关节速度
    fps: float
    velocity_limits: np.ndarray       # (J,) URDF 速度限位
    acceleration_envelope: np.ndarray  # (J,) 实证 |acc| 包络
    joint_names: Optional[Sequence[str]] = None
    vel_limit_frac: float = 0.85
    budget_scale: float = 1.5


def kinematic_playback_screen(inputs: KinematicScreenInputs, ratio: float) -> Dict[str, Any]:
    """|qd|·s ≤ vcap 且 |qdd|·s² ≤ acap。s 单调升 ⇒ 最坏点必在 s=ratio。"""
    jv = _finite_array("joint_vel", inputs.joint_vel)
    if jv.ndim != 2:
        raise PlaybackGateError(f"joint_vel must be (T, J), got {jv.shape}")
    fps = float(inputs.fps)
    if not (np.isfinite(fps) and fps > 0.0):
        raise PlaybackGateError(f"bad fps {inputs.fps}")
    vel_cap = _finite_array("velocity_limits", inputs.velocity_limits) * float(
        inputs.vel_limit_frac
    )
    acc_cap = _finite_array("acceleration_envelope", inputs.acceleration_envelope) * float(
        inputs.budget_scale
    )
    if vel_cap.shape != (jv.shape[1],) or acc_cap.shape != (jv.shape[1],):
        raise PlaybackGateError("velocity/acceleration caps must have one entry per joint")
    if np.any(vel_cap <= 0.0) or np.any(acc_cap <= 0.0):
        raise PlaybackGateError("velocity/acceleration caps must be strictly positive")

    vel_peak = np.abs(jv).max(axis=0)
    acc = np.abs(np.diff(jv, axis=0)) * fps
    acc_peak = acc.max(axis=0) if acc.shape[0] else np.zeros(jv.shape[1])

    s = float(ratio)
    vel_util = vel_peak * s / vel_cap
    acc_util = acc_peak * s * s / acc_cap
    with np.errstate(divide="ignore", invalid="ignore"):
        per_joint_max_ratio = np.minimum(
            np.where(vel_peak > 0.0, vel_cap / np.where(vel_peak > 0.0, vel_peak, 1.0), np.inf),
            np.where(acc_peak > 0.0,
                     np.sqrt(acc_cap / np.where(acc_peak > 0.0, acc_peak, 1.0)), np.inf),
        )
    max_ratio = float(np.min(per_joint_max_ratio))
    names = list(inputs.joint_names) if inputs.joint_names is not None else [
        f"dof_{index}" for index in range(jv.shape[1])
    ]
    offending = [
        {
            "joint": names[index],
            "vel_util": round(float(vel_util[index]), 6),
            "acc_util": round(float(acc_util[index]), 6),
        }
        for index in range(jv.shape[1])
        if vel_util[index] > 1.0 or acc_util[index] > 1.0
    ]
    return {
        "screen": "kinematic_envelope",
        "ran": True,
        "pass": len(offending) == 0,
        "ratio": s,
        "max_ratio": max_ratio,
        "vel_util_max": round(float(vel_util.max()), 6),
        "acc_util_max": round(float(acc_util.max()), 6),
        "vel_limit_frac": float(inputs.vel_limit_frac),
        "budget_scale": float(inputs.budget_scale),
        "offending": offending,
        "note": (
            "diagonal kinematic envelope only: no gravity, Coriolis or damping term. "
            "Never let this screen stand in for the torque screen."
        ),
    }


# ====================================================================================== #
# 筛子 ②:tau(s) = c0 + s·L + s²·Q(需要逆动力学)                                          #
# ====================================================================================== #
@dataclass(frozen=True)
class PlaybackQuadratic:
    """逐帧辨识出来的慢放二次律系数(每个都是 (T, nv))。"""

    c0: np.ndarray      # 重力 + 纯位形偏置:**不随 s 缩放**
    c1: np.ndarray      # 阻尼类:随 s 线性
    c2: np.ndarray      # 惯性 + 科氏:随 s²
    max_probe_residual: float
    probe_ratios: Tuple[float, ...]


@dataclass(frozen=True)
class TorqueScreenInputs:
    """力矩筛输入。qpos/qvel/qacc 是**原生 1× 播放**的参考轨迹。"""

    qpos: np.ndarray                # (T, nq)
    qvel: np.ndarray                # (T, nv)
    qacc: np.ndarray                # (T, nv)
    inverse_dynamics: InverseDynamics
    effort_lower: np.ndarray        # (nv,) 已扣掉干摩擦余量的 generalized-force 下限
    effort_upper: np.ndarray        # (nv,)
    dof_names: Optional[Sequence[str]] = None
    support_mode: str = "fixed_base"
    law_probe_ratios: Tuple[float, ...] = DEFAULT_LAW_PROBE_RATIOS
    law_relative_tolerance: float = DEFAULT_LAW_RELATIVE_TOLERANCE
    law_absolute_tolerance: float = DEFAULT_LAW_ABSOLUTE_TOLERANCE


def identify_playback_quadratic(inputs: TorqueScreenInputs) -> PlaybackQuadratic:
    """在 s ∈ {0, 1, 2} 上辨识,再用独立探针核对二次律;不成立就 fail loud。

    tau(s) := ID(q, s·qd, s²·qdd)。有接触力时它不是二次的(实测 full scope 残差
    1.96-13.3 N·m,对 3e-13 的 upper scope),探针会当场抓住并拒绝发证。
    """
    qpos = _finite_array("qpos", inputs.qpos)
    qvel = _finite_array("qvel", inputs.qvel)
    qacc = _finite_array("qacc", inputs.qacc)
    if qpos.ndim != 2 or qvel.ndim != 2 or qacc.ndim != 2:
        raise PlaybackGateError("qpos/qvel/qacc must all be 2-D (T, ...) arrays")
    frames = qpos.shape[0]
    if qvel.shape[0] != frames or qacc.shape[0] != frames:
        raise PlaybackGateError("qpos/qvel/qacc must have equal frame counts")
    if qvel.shape[1] != qacc.shape[1]:
        raise PlaybackGateError("qvel and qacc must have equal DoF counts")
    nv = qvel.shape[1]

    def evaluate(frame: int, s: float) -> np.ndarray:
        force = np.asarray(
            inputs.inverse_dynamics(
                qpos[frame].copy(),
                qvel[frame] * float(s),
                qacc[frame] * float(s) * float(s),
            ),
            dtype=np.float64,
        )
        if force.shape != (nv,):
            raise PlaybackGateError(
                f"inverse_dynamics must return shape ({nv},), got {force.shape}"
            )
        if not np.all(np.isfinite(force)):
            raise PlaybackGateError("inverse_dynamics returned non-finite generalized force")
        return force

    c0 = np.empty((frames, nv), dtype=np.float64)
    c1 = np.empty((frames, nv), dtype=np.float64)
    c2 = np.empty((frames, nv), dtype=np.float64)
    worst_residual = 0.0
    probes = tuple(float(value) for value in inputs.law_probe_ratios)
    if not probes:
        raise PlaybackGateError("at least one independent law probe ratio is required")
    for frame in range(frames):
        at_zero = evaluate(frame, 0.0)
        at_one = evaluate(frame, 1.0)
        at_two = evaluate(frame, 2.0)
        quadratic = 0.5 * (at_two - 2.0 * at_one + at_zero)
        linear = at_one - at_zero - quadratic
        c0[frame] = at_zero
        c1[frame] = linear
        c2[frame] = quadratic
        for probe in probes:
            actual = evaluate(frame, probe)
            predicted = at_zero + linear * probe + quadratic * probe * probe
            residual = float(np.max(np.abs(actual - predicted)))
            scale = max(1.0, float(np.max(np.abs(actual))), float(np.max(np.abs(predicted))))
            allowed = inputs.law_absolute_tolerance + inputs.law_relative_tolerance * scale
            if residual > allowed:
                raise PlaybackLawViolation(
                    "playback torque is not quadratic in the speed ratio: frame "
                    f"{frame}, probe s={probe}, residual={residual:.6g} N*m > "
                    f"allowed={allowed:.6g}. tau(s)=c0+s*L+s^2*Q holds only without "
                    "contact forces; a grounded/floating-base clip must go through the "
                    "support-polygon screen, not this one."
                )
            worst_residual = max(worst_residual, residual)
    return PlaybackQuadratic(
        c0=c0, c1=c1, c2=c2, max_probe_residual=worst_residual, probe_ratios=probes
    )


def torque_playback_screen(
    inputs: TorqueScreenInputs,
    ratio: float,
    *,
    ratio_floor: float = 0.0,
    law: Optional[PlaybackQuadratic] = None,
) -> Dict[str, Any]:
    """在整个 [ratio_floor, ratio] 区间上判卷(端点 + 内部顶点),并单列静态重力筛。"""
    if inputs.support_mode != "fixed_base":
        raise IncompletePlaybackCertificate(
            "the torque screen certifies fixed-base (upper-scope) clips only; a "
            f"support_mode={inputs.support_mode!r} clip converges to a STATIC BALANCE "
            "problem as s -> 0 and no joint-torque number can see that",
            missing=["ground_contact_support_polygon_screen"],
        )
    s = float(ratio)
    if not (np.isfinite(s) and s > 0.0):
        raise PlaybackGateError(f"ratio must be a positive finite number, got {ratio}")
    floor = float(ratio_floor)
    if not (np.isfinite(floor) and 0.0 <= floor <= s):
        raise PlaybackGateError(f"ratio_floor must satisfy 0 <= floor <= ratio, got {ratio_floor}")

    law = identify_playback_quadratic(inputs) if law is None else law
    nv = law.c0.shape[1]
    lower = _finite_array("effort_lower", inputs.effort_lower)
    upper = _finite_array("effort_upper", inputs.effort_upper)
    if lower.shape != (nv,) or upper.shape != (nv,):
        raise PlaybackGateError(f"effort limits must both have shape ({nv},)")
    if np.any(lower >= 0.0) or np.any(upper <= 0.0):
        raise PlaybackGateError(
            "effort limits must bracket zero (lower < 0 < upper); a joint that cannot "
            "hold zero generalized force has no admissible static pose"
        )

    names = list(inputs.dof_names) if inputs.dof_names is not None else [
        f"dof_{index}" for index in range(nv)
    ]

    # --- 静态重力筛:s 无关的那一项,单独报告(慢放救不了它) ------------------------- #
    static_util = _utilisation(law.c0, law.c0, lower, upper)
    static_max = float(np.max(static_util))
    static_index = int(np.argmax(static_util))
    static_frame, static_dof = np.unravel_index(static_index, static_util.shape)

    # --- 区间极值(含内部顶点) ------------------------------------------------------ #
    minimum, maximum, _, arg_max = quadratic_interval_extrema(law.c0, law.c1, law.c2, floor, s)
    util = _utilisation(minimum, maximum, lower, upper)
    util_max = float(np.max(util))
    worst_index = int(np.argmax(util))
    worst_frame, worst_dof = np.unravel_index(worst_index, util.shape)
    worst_ratio = float(arg_max[worst_frame, worst_dof])
    interior = (arg_max > floor + 1.0e-12) & (arg_max < s - 1.0e-12)
    interior_binding = int(np.count_nonzero(interior & (util > 1.0)))

    exit_ratio = first_exit_ratio(law.c0, law.c1, law.c2, lower, upper)
    max_ratio = float(np.min(exit_ratio))

    offending_mask = util > 1.0
    offending: List[Dict[str, Any]] = []
    if np.any(offending_mask):
        order = np.argsort(util, axis=None)[::-1]
        for flat in order[:16]:
            frame, dof = np.unravel_index(int(flat), util.shape)
            if util[frame, dof] <= 1.0:
                break
            offending.append(
                {
                    "frame": int(frame),
                    "dof": names[int(dof)],
                    "utilisation": round(float(util[frame, dof]), 6),
                    "at_ratio": round(float(arg_max[frame, dof]), 6),
                    "interior_vertex": bool(interior[frame, dof]),
                    "static_gravity_utilisation": round(float(static_util[frame, dof]), 6),
                }
            )
    return {
        "screen": "torque_quadratic",
        "ran": True,
        "pass": len(offending) == 0,
        "ratio": s,
        "ratio_interval": [floor, s],
        "max_ratio": max_ratio,
        "utilisation_max": round(util_max, 6),
        "utilisation_max_at": {
            "frame": int(worst_frame),
            "dof": names[int(worst_dof)],
            "ratio": round(worst_ratio, 6),
            "interior_vertex": bool(interior[worst_frame, worst_dof]),
        },
        "static_gravity": {
            "utilisation_max": round(static_max, 6),
            "pass": bool(static_max <= 1.0),
            "frame": int(static_frame),
            "dof": names[int(static_dof)],
            "note": (
                "c0(q) does NOT scale with s. If this fails, the clip is inadmissible "
                "at EVERY playback ratio, including arbitrarily slow ones."
            ),
        },
        "interior_vertex_violation_count": interior_binding,
        "quadratic_law_residual_max": float(law.max_probe_residual),
        "quadratic_law_probe_ratios": list(law.probe_ratios),
        "offending": offending,
    }


# ====================================================================================== #
# 证书                                                                                    #
# ====================================================================================== #
def certify_playback_ratio(
    ratio: float,
    kinematic: KinematicScreenInputs,
    torque: Optional[TorqueScreenInputs] = None,
    *,
    ratio_floor: float = 0.0,
    clip_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把三道筛子合成一张证书。verdict ∈ {feasible, infeasible, incomplete}。

    torque 缺席 ⇒ verdict="incomplete"、admissible=False:运动学包络里没有重力项,
    单独它**不足以**给慢放发证(DEFINITIONS.md 自己说那只是一阶对角下界)。
    """
    s = float(ratio)
    if not (np.isfinite(s) and s > 0.0):
        raise PlaybackGateError(f"ratio must be a positive finite number, got {ratio}")

    screens: Dict[str, Any] = {}
    screens["kinematic_envelope"] = kinematic_playback_screen(kinematic, s)
    missing: List[str] = []
    if torque is None:
        missing.append("torque_quadratic")
    else:
        screens["torque_quadratic"] = torque_playback_screen(
            torque, s, ratio_floor=ratio_floor
        )
    missing.append("ground_contact_support_polygon")

    ceilings = [float(screen["max_ratio"]) for screen in screens.values()]
    max_ratio = float(min(ceilings)) if ceilings else 0.0
    all_pass = all(bool(screen["pass"]) for screen in screens.values())
    complete = "torque_quadratic" in screens
    if not all_pass:
        # 任何**已跑**的筛子判否都是确定性拒绝:再补别的筛子也翻不过来。
        verdict = "infeasible"
    elif not complete:
        verdict = "incomplete"
    else:
        verdict = "feasible"

    certificate: Dict[str, Any] = {
        "tool": "canonical_playback_speed_gate.py v1 (uniform slow-playback feasibility gate)",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "law": "q(t) -> q(s*t): qd -> s*qd, qdd -> s^2*qdd, positions unchanged",
        "requested_ratio": s,
        "ratio_interval": [float(ratio_floor), s],
        "verdict": verdict,
        "admissible": bool(verdict == "feasible" and s <= max_ratio + 1.0e-12),
        "max_playback_ratio": max_ratio,
        "runtime_playback_max_ratio": RUNTIME_PLAYBACK_MAX_RATIO,
        "screens": screens,
        "screens_not_run": missing,
        "claim_scope": (
            "kinematic_only_NOT_shippable" if not complete else "fixed_base_joint_torque"
        ),
        "clip": dict(clip_evidence) if clip_evidence else None,
    }
    return certificate


def assert_playback_ratio_admissible(certificate: Dict[str, Any]) -> None:
    """证书不认可就抛 —— 调用方拿不到"沉默降速"这个选项。"""
    if certificate.get("admissible") is True:
        return
    raise PlaybackRatioInfeasible(
        "playback ratio {ratio} is NOT admissible (verdict={verdict}, "
        "max_playback_ratio={cap})".format(
            ratio=certificate.get("requested_ratio"),
            verdict=certificate.get("verdict"),
            cap=certificate.get("max_playback_ratio"),
        ),
        certificate,
    )


# ====================================================================================== #
# 离线交付计划:哪几档还真需要烤片段                                                       #
# ====================================================================================== #
def plan_offline_speed_ladder(
    ladder: Sequence[float],
    certificate: Optional[Dict[str, Any]] = None,
    *,
    native_ratio: float = 1.0,
    runtime_playback_max: float = RUNTIME_PLAYBACK_MAX_RATIO,
) -> Dict[str, Any]:
    """每一档怎么交付:native / runtime_playback / bake_required / 未授权。

    * r == native  -> ``native_source``(源资产本身,不需要任何烤入)
    * r <  native 且 ≤ 证书 max_playback_ratio -> ``runtime_playback``(**省下一次烤入**)
    * r >  runtime_playback_max                -> ``bake_required``(时钟不许上探)
    * 无证书 / 超出证书上限                     -> ``bake_required_playback_unauthorised``
      (fail closed:没验过就不许当成慢放覆盖)
    """
    cap: Optional[float] = None
    certified = False
    if certificate is not None:
        certified = certificate.get("verdict") == "feasible"
        if certified:
            cap = float(certificate["max_playback_ratio"])
    rows: List[Dict[str, Any]] = []
    for raw in ladder:
        ratio = float(raw)
        if not (np.isfinite(ratio) and ratio > 0.0):
            raise PlaybackGateError(f"ladder entries must be positive finite, got {raw}")
        if abs(ratio - native_ratio) <= 1.0e-12:
            delivery, why = "native_source", "the native clip already is this ratio"
        elif ratio > runtime_playback_max + 1.0e-12:
            delivery, why = (
                "bake_required",
                "above native: the runtime clock only slows down (round() skips asset "
                "frames above 1.0, possibly the contact frame itself)",
            )
        elif not certified:
            delivery, why = (
                "bake_required_playback_unauthorised",
                "no feasible playback certificate for this clip — run --certify first",
            )
        elif cap is not None and ratio > cap + 1.0e-12:
            delivery, why = (
                "bake_required_playback_unauthorised",
                "ratio {r} exceeds the certified playback ceiling {c}".format(r=ratio, c=cap),
            )
        else:
            delivery, why = (
                "runtime_playback",
                "uniform slow playback at speed_scale={s} reproduces this exactly".format(
                    s=ratio
                ),
            )
        rows.append({"ratio": ratio, "delivery": delivery, "why": why})

    bake_required = [row for row in rows if row["delivery"].startswith("bake_required")]
    playback = [row for row in rows if row["delivery"] == "runtime_playback"]
    baseline = [row for row in rows if row["delivery"] != "native_source"]
    return {
        "ladder": rows,
        "certified": certified,
        "certified_playback_ceiling": cap,
        "runtime_playback_max": float(runtime_playback_max),
        "bakes_without_this_gate": len(baseline),
        "bakes_required": len(bake_required),
        "bakes_saved": len(playback),
        "note": (
            "saved bakes are per-motion clip assets that no longer have to be produced, "
            "sha-registered, and appended to every question-bank family allow-list"
        ),
        "runtime_preconditions": [
            "the consuming arm must run the R14 retiming lane (task.motion.speed_scale_range "
            "!= (1,1)); an arm pinned at (1.0, 1.0) cannot play anything back slowly",
            "the planner task-revision phase governor must be OFF: mdp/commands.py refuses "
            "speed_scale_range/speed_scale_per_clip because the governor owns the sole "
            "reference clock — a governor arm still needs baked slowdown assets",
            "external BankExam requires native one-frame-per-step playback, so exam clips "
            "must be run at speed_scale = 1.0",
        ],
    }


# ====================================================================================== #
# MuJoCo 绑定(pod 端;host 上没有 mujoco 也能 import 本模块)                              #
# ====================================================================================== #
def mujoco_fixed_base_screen_inputs(
    mjcf_path: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    *,
    dof_names: Optional[Sequence[str]] = None,
) -> TorqueScreenInputs:
    """从 MJCF 造出力矩筛输入:限位口径与 canonical_torque_path_topp 完全一致。"""
    try:
        import mujoco  # type: ignore
    except ImportError as exc:  # pragma: no cover - pod-only path
        raise IncompletePlaybackCertificate(
            "mujoco is not installed; the torque screen cannot run here",
            missing=["mujoco"],
        ) from exc
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    contract = ctp.direct_actuator_contract_from_mujoco(model, support_mode="fixed_base")
    if int(contract.free_dof_count) != 0:
        raise IncompletePlaybackCertificate(
            "this MJCF has a floating root; slowing a grounded clip converges it to a "
            "static balance problem that the joint-torque screen cannot see",
            missing=["ground_contact_support_polygon_screen"],
        )
    lower, upper, _ = ctp.resolve_generalized_force_limits(contract)
    return TorqueScreenInputs(
        qpos=qpos,
        qvel=qvel,
        qacc=qacc,
        inverse_dynamics=ctp.MujocoSmoothInverseDynamics(model),
        effort_lower=lower,
        effort_upper=upper,
        dof_names=dof_names,
        support_mode="fixed_base",
    )


# ====================================================================================== #
# CLI                                                                                     #
# ====================================================================================== #
def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SystemExit("拒绝覆盖已有文件: {p}".format(p=path))
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w",
              encoding="utf-8") as stream:
        stream.write(text)
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise SystemExit("certificate 回读不一致 — refusing")


def _parse_ladder(text: str) -> List[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--certify", action="store_true", help="判一个 clip 的慢放比")
    parser.add_argument("--plan", action="store_true", help="按证书算离线交付计划")
    parser.add_argument("--clip", default=None, help="源 clip npz(certify 模式必填)")
    parser.add_argument("--ratio", type=float, default=None, help="请求的慢放比 s")
    parser.add_argument("--ratio-floor", type=float, default=0.0,
                        help="判卷区间下端(默认 0 = 含静态重力筛;运行时 hold 就是 s=0)")
    parser.add_argument("--budget-clips", nargs="+", default=None,
                        help="|acc| 实证包络来源(certify 模式必填)")
    parser.add_argument("--urdf", default=None, help="A3 URDF(默认 repo 内拷贝)")
    parser.add_argument("--mjcf", default=None,
                        help="力矩筛用的 MJCF;缺席 = kinematic_only 半证(不可发货)")
    parser.add_argument("--vel-limit-frac", type=float, default=0.85)
    parser.add_argument("--budget-scale", type=float, default=1.5)
    parser.add_argument("--certificate", default=None,
                        help="certify: 写出的证书路径;plan: 读入的证书路径")
    parser.add_argument("--ladder", default="0.65,0.8,0.9,1.0,1.1,1.2,1.35")
    parser.add_argument("--plan-out", default=None, help="plan 模式的输出 JSON(可选)")
    args = parser.parse_args(argv)

    if args.certify == args.plan:
        raise SystemExit("choose exactly one of --certify / --plan")

    if args.plan:
        if args.certificate is None:
            raise SystemExit("--plan 需要 --certificate(没有证书就没有慢放授权)")
        certificate = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
        plan = plan_offline_speed_ladder(_parse_ladder(args.ladder), certificate)
        for row in plan["ladder"]:
            print("  r={r:<6} {d:<38} {w}".format(
                r=row["ratio"], d=row["delivery"], w=row["why"]))
        print("[plan] bakes required {req} of {base} ladder variants; "
              "{saved} covered by runtime slow playback".format(
                  req=plan["bakes_required"], base=plan["bakes_without_this_gate"],
                  saved=plan["bakes_saved"]))
        if args.plan_out is not None:
            _write_json(Path(args.plan_out).expanduser().absolute(), plan)
        return 0

    if args.clip is None or args.ratio is None or args.budget_clips is None:
        raise SystemExit("--certify 需要 --clip / --ratio / --budget-clips")
    if args.certificate is None:
        raise SystemExit("--certify 需要 --certificate 输出路径")

    import synthesize_timing as v1  # noqa: E402  与 bake 工具同一套包络口径
    from audit_motion_npz import parse_urdf_limits  # noqa: E402

    if args.urdf is None:
        args.urdf = os.path.normpath(os.path.join(
            _HERE, "../../..", "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"))
    limits = parse_urdf_limits(args.urdf)
    joint_names = list(v1.ISAAC_JOINT_NAMES)
    velocity_limits = np.array(
        [limits[name].velocity if limits[name].velocity is not None else np.inf
         for name in joint_names],
        dtype=np.float64,
    )
    envelope = np.asarray(v1.acc_envelope(args.budget_clips), dtype=np.float64)
    if np.any(envelope <= 0.0):
        positive = envelope[envelope > 0.0]
        envelope = np.where(envelope <= 0.0, positive.min() if positive.size else 1.0, envelope)

    data = dict(np.load(args.clip, allow_pickle=False))
    kinematic = KinematicScreenInputs(
        joint_vel=np.asarray(data["joint_vel"], dtype=np.float64),
        fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
        velocity_limits=velocity_limits,
        acceleration_envelope=envelope,
        joint_names=joint_names,
        vel_limit_frac=args.vel_limit_frac,
        budget_scale=args.budget_scale,
    )
    torque = None
    if args.mjcf is not None:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
        joint_acc = np.gradient(joint_vel, 1.0 / kinematic.fps, axis=0)
        torque = mujoco_fixed_base_screen_inputs(
            args.mjcf, joint_pos, joint_vel, joint_acc, dof_names=joint_names
        )

    certificate = certify_playback_ratio(
        args.ratio,
        kinematic,
        torque,
        ratio_floor=args.ratio_floor,
        clip_evidence={"path": str(args.clip), "sha256": _sha256_file(args.clip)},
    )
    _write_json(Path(args.certificate).expanduser().absolute(), certificate)
    print("[gate] r={r} verdict={v} max_playback_ratio={cap}".format(
        r=certificate["requested_ratio"], v=certificate["verdict"],
        cap=certificate["max_playback_ratio"]))
    if certificate["verdict"] == "incomplete":
        print("  INCOMPLETE: screens not run {missing} — this certificate authorises "
              "nothing".format(missing=certificate["screens_not_run"]))
        return EXIT_INCOMPLETE
    if certificate["verdict"] != "feasible":
        for screen in certificate["screens"].values():
            for row in screen.get("offending", [])[:6]:
                print("  - {row}".format(row=row))
        return EXIT_INFEASIBLE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
