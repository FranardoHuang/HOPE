#!/usr/bin/env python3
"""可达拍面优化器（轴 B）—— 给定来球+触球点+拍速+目标落点，找"合法回球最优拍面法向"。

WHAT THIS IS（人话）
    输入一颗固定的来球（速度/旋转）、固定触球位置、固定拍速、指定对方台面落点，本工具在
    **signed 物理击球面半球**上搜索拍面法向 n*，让"合法回球质量"最高。输出两份答案：

        1. unconstrained physics optimum —— 纯球物理最优拍面（只受合法回球硬门约束）；
        2. robot-reachable optimum       —— 再加"手腕可达圆锥"约束后的最优拍面；

    以及两者的 reward gap（为可达性付出的回球质量代价）、n/-n 负控、objective landscape 采样、
    和按最终可达 signed yaw/pitch 的 train/interpolation/OOD 分桶建议（manifest 写死比例约定）。

物理模型（不自造第二套球物理）
    完全复用 scripts/virtual_return_scorer.py —— Isaac 训练判分的 NumPy 规范实现：
    venue 拟合的拍-球冲量律（predict_paddle_contact）、阻力+马格努斯 RK4 飞行
    （coarse_flight_events，固定 10 ms × 100 步）、球心过网/落台事件口径；
    桌面几何复用 hope_planner.constants.TableParams（ITTF 尺寸）+ venue_ball_sampler 的
    训练虚拟台摆位常量（near_x=0.5, surface_z=0.76，env 系）。

SIGNED FACE 语义（franco 2026-07-09 拍板："哪面拍子超前就是哪面"）
    硬门是**单面严格判据** n · v_hit > 0（v_hit = 拍速）。禁止取 abs —— SMASH Eq.11 的
    n/−n 对称形式（|n·(...)|）会把反面击球合法化，明确禁止复制。接触冲量律内部的
    orient_normal 是符号不变的平面动力学，不承担面身份判定；身份判定只走本工具的
    signed_face_gate。每次求解都对 -n* 跑负控并断言显著更差，不达标当场报错（fail-closed）。

硬门（先全过再比 reward；任一不过 = 该候选拿 GATED_REWARD 哨兵，绝不参与最优）
    1. signed face:   n · v_hit > 0（严格单面，无 abs）；
    2. 有效接触:      approach = v_hit · oriented_n > 0.3 m/s（判分器同款 phantom-block 门；
                      触球位置按输入给定视作 pos_err=0）；
    3. 过网:          球心在网平面高于 surface_z + net_height + ball_radius；
    4. 首落点对方台内: net_x < x ≤ far_x 且 |y| ≤ half_width（判分器同款边界）。

reward（只在四门全过后比较；权重是模块常量，manifest 全量记录）
    J_phys = 1.0(legal)
           + 0.5 · min(过网余量, 0.30)/0.30
           + 1.5 · exp(-落点误差/0.25)
           + 0.5 · min(|v_out|, 7.0)/7.0        （出球速度，cap= venue 拟合有效包络上界）
           + 0.25· min(|ω_out|, 34.0)/34.0      （spin 项，cap= venue 采样上限）
    reachable 目标另加 0.5 · (1 - 偏离可达轴角/半角)，且偏离角 ≤ 半角是硬门。
    设计红线：objective 不含"追踪自己选出的 target normal"一类自指项 —— reward 只吃
    回球结果量（余量/误差/速度/旋转），单测有源码守卫。

可达性近似（声明边界）
    当前用**简化圆锥**：拍面法向偏离 --reach-axis（默认 = 拍速方向，"超前面"先验）不超过
    --reach-half-angle-deg（默认 35°）。这只是手腕姿态可达域的零阶近似——真实可达域应由
    31 关节 IK + 关节限位余量给出（TODO: full IK joint-margin reachability；接入
    rewrite_followthrough 的 URDF 限位检查后替换本圆锥）。docstring 即边界声明。

分桶约定（写死进 manifest，各轴同一比例约定，轴 D 的 β 亦同）
    train 0.80/1.00/1.20 · interpolation 0.90/1.10 · OOD 0.65/1.35。
    建议表 = 把可达最优的 signed yaw/pitch 各乘比例，逐个重新过硬门评 reward，如实标注
    哪些缩放面仍是合法回球。

坐标/角度口径
    env 系：+x 指向对手，z 上。signed yaw = atan2(n_y, n_x)，signed pitch =
    atan2(n_z, hypot(n_x, n_y))，单位度，带符号（不取 abs，n 与 -n 角度不同——这就是 signed）。

USAGE
    python scripts/optimize_reachable_face.py \
        --ball-vel=-2.5,0.3,-0.8 --ball-spin 0,0,0 --contact-pos 0.6,-0.1,1.05 \
        --racket-vel 4.0,0.5,1.5 --target-landing 2.6,0.3 --json OUT.json

    Exit 0 = 两个最优都找到且负控通过；找不到任何合法拍面 / 负控不显著 = 当场报错退出。

输出 manifest（一层内容寻址，franco 拍板的精简治理：单层 sha + 单测，不搞多层审批链）
    {"kind": "hope_reachable_face_manifest", "schema_version": 1,
     "sha256": sha256(canonical payload), "payload": {...}}
    payload 内含 venue 物理 YAML 的文件 sha、全部输入回显、两个最优、reward gap、负控、
    landscape（全有限值）、分桶建议。

DEPENDENCIES
    numpy + PyYAML（virtual_return_scorer 读 venue 物理表）+ 仓库内 hope_planner（桌面几何，
    按固定仓库布局 lazy import）。不需要 isaac / torch / mujoco。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import virtual_return_scorer as vrs  # noqa: E402  物理唯一来源：冲量律 + RK4 飞行 + 事件口径
from venue_ball_sampler import (  # noqa: E402  训练虚拟台摆位 + venue 采样上限（几何唯一来源）
    DEFAULT_TABLE_NEAR_X,
    DEFAULT_TABLE_SURFACE_Z,
    VENUE_SPIN_ABS_MAX,
    hope_planner_path,
)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# ---- reward 权重/尺度（模块常量，manifest 全量记录）------------------------------------------
LEGAL_BASE = 1.0          # 合法回球基础分
W_NET, NET_MARGIN_CAP = 0.5, 0.30      # 过网余量，0.30 m 饱和（不奖励无限高吊）
W_LAND, LAND_ERR_SCALE = 1.5, 0.25     # 落点误差 exp 衰减尺度 0.25 m
W_SPEED, OUT_SPEED_CAP = 0.5, 7.0      # 出球速度，cap = venue 拟合有效包络上界 (m/s)
W_SPIN = 0.25
OUT_SPIN_CAP = float(VENUE_SPIN_ABS_MAX)   # 34 rad/s, venue 采样上限
W_REACH = 0.5             # 可达余量项（只进 reachable 目标）
GATED_REWARD = -1.0       # 硬门不过的有限哨兵值（landscape 里绝不出现 ±inf/nan）
NEG_CONTROL_MIN_GAP = 1.0  # n* 对 -n* 至少领先这么多（legal ≥ 1.0 > 0 > -1 结构保证）

BUCKET_CONVENTION = {
    "train": (0.80, 1.00, 1.20),
    "interpolation": (0.90, 1.10),
    "ood": (0.65, 1.35),
}
BUCKET_NOTE = "各轴同一比例约定（轴 D 的 β 亦同）；比例乘在可达最优的 signed yaw/pitch 上"

MANIFEST_KIND = "hope_reachable_face_manifest"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------------------------
def _vec(text, name, n):
    """CLI 逗号向量解析：长度/有限值 fail-closed。"""
    try:
        parts = [float(p) for p in str(text).split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be {n} comma-separated floats, got {text!r}") from exc
    arr = np.asarray(parts, dtype=np.float64)
    if arr.shape != (n,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be {n} finite floats, got {text!r}")
    return arr


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FaceGeometry:
    """env 系桌面几何（判分器 VirtualReturnSpec 同字段口径）。"""

    table_surface_z: float
    net_x: float
    far_x: float
    half_width: float
    net_height: float


def build_geometry(table_near_x=DEFAULT_TABLE_NEAR_X,
                   table_surface_z=DEFAULT_TABLE_SURFACE_Z) -> FaceGeometry:
    """桌面几何 = hope_planner ITTF TableParams + 训练虚拟台摆位（不自造第二套数字）。"""
    hp = hope_planner_path(REPO_ROOT)
    if not os.path.isdir(os.path.join(hp, "hope_planner")):
        raise SystemExit(f"[FATAL] table geometry needs hope_planner at {hp} "
                         f"(repo layout is fixed; found nothing there)")
    if hp not in sys.path:
        sys.path.insert(0, hp)
    from hope_planner.constants import TableParams

    table = TableParams()
    near = float(table_near_x)
    return FaceGeometry(
        table_surface_z=float(table_surface_z),
        net_x=near + float(table.net_x),
        far_x=near + float(table.length),
        half_width=float(table.width) / 2.0,
        net_height=float(table.net_height),
    )


@dataclass(frozen=True)
class Scenario:
    """一题的固定输入（env 系）。"""

    ball_vel: np.ndarray          # (3,) 来球触球瞬间速度，vx < 0（朝机器人）
    ball_spin: np.ndarray         # (3,) 来球旋转 rad/s
    contact_pos: np.ndarray       # (3,) 触球位置（= 拍面中心，pos_err 视作 0）
    racket_vel: np.ndarray        # (3,) 触球瞬间拍速 v_hit
    target_landing_xy: np.ndarray  # (2,) 指定落点（对方台内）


def build_scenario(*, ball_vel, ball_spin, contact_pos, racket_vel, target_landing_xy,
                   geometry: FaceGeometry) -> Scenario:
    """输入校验（fail-closed）：有限值、来球朝机器人、触球点在我方半区、落点在对方台内。"""
    v_in = np.asarray(ball_vel, dtype=np.float64)
    w_in = np.asarray(ball_spin, dtype=np.float64)
    pos = np.asarray(contact_pos, dtype=np.float64)
    v_r = np.asarray(racket_vel, dtype=np.float64)
    tgt = np.asarray(target_landing_xy, dtype=np.float64)
    for name, arr, shape in (("ball_vel", v_in, (3,)), ("ball_spin", w_in, (3,)),
                             ("contact_pos", pos, (3,)), ("racket_vel", v_r, (3,)),
                             ("target_landing_xy", tgt, (2,))):
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite shape {shape}, got {arr!r}")
    if not v_in[0] < 0.0:
        raise ValueError(f"ball_vel.x must be < 0 (ball approaching the robot), got {v_in[0]}")
    if not pos[0] < geometry.net_x:
        raise ValueError(f"contact_pos.x must be on our side of the net "
                         f"(< {geometry.net_x:.3f}), got {pos[0]}")
    if not pos[2] > 0.0:
        raise ValueError(f"contact_pos.z must be above the floor, got {pos[2]}")
    if not (geometry.net_x < tgt[0] <= geometry.far_x and abs(tgt[1]) <= geometry.half_width):
        raise ValueError(
            f"target_landing_xy must be on the opponent half in bounds "
            f"(x in ({geometry.net_x:.3f}, {geometry.far_x:.3f}], |y| <= "
            f"{geometry.half_width:.4f}), got {tgt}")
    return Scenario(ball_vel=v_in, ball_spin=w_in, contact_pos=pos, racket_vel=v_r,
                    target_landing_xy=tgt)


# ---------------------------------------------------------------------------------------------
def face_from_angles(yaw_deg, pitch_deg):
    """signed yaw/pitch（度，env 系 +x 朝对手）→ 单位法向。n 与 -n 的角度不同（这就是 signed）。"""
    y = math.radians(float(yaw_deg))
    p = math.radians(float(pitch_deg))
    return np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)],
                    dtype=np.float64)


def angles_from_face(normal):
    """单位法向 → (signed yaw, signed pitch) 度。"""
    n = np.asarray(normal, dtype=np.float64)
    return (math.degrees(math.atan2(n[1], n[0])),
            math.degrees(math.atan2(n[2], math.hypot(n[0], n[1]))))


def signed_face_gate(normal, racket_vel):
    """单面严格判据：n · v_hit > 0（"哪面超前就是哪面"）。

    禁止取绝对值——SMASH Eq.11 的 n/−n 对称形式会让反面击球同分，明确禁止复制。
    返回 (通过?, 原始点积)。
    """
    dot = float(np.dot(np.asarray(normal, dtype=np.float64),
                       np.asarray(racket_vel, dtype=np.float64)))
    return dot > 0.0, dot


def reward_physics_terms(*, net_margin, land_err, out_speed, out_spin_mag):
    """合法回球 reward（只吃回球结果量；不含任何"追踪选出的法向"自指项）。"""
    for name, value in (("net_margin", net_margin), ("land_err", land_err),
                        ("out_speed", out_speed), ("out_spin_mag", out_spin_mag)):
        if not np.isfinite(float(value)):
            raise ValueError(f"reward term {name} must be finite, got {value}")
    return (LEGAL_BASE
            + W_NET * min(max(float(net_margin), 0.0), NET_MARGIN_CAP) / NET_MARGIN_CAP
            + W_LAND * math.exp(-max(float(land_err), 0.0) / LAND_ERR_SCALE)
            + W_SPEED * min(max(float(out_speed), 0.0), OUT_SPEED_CAP) / OUT_SPEED_CAP
            + W_SPIN * min(max(float(out_spin_mag), 0.0), OUT_SPIN_CAP) / OUT_SPIN_CAP)


def reach_cone_margin(normal, reach_axis, half_angle_deg):
    """简化圆锥可达近似：返回 (偏离角度, 归一余量 1-angle/half, 硬门通过?)。

    边界声明：这是手腕可达域的零阶近似；完整 31 关节 IK + 限位余量是 TODO（见模块 docstring）。
    """
    axis = np.asarray(reach_axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"reach axis must be finite nonzero, got {reach_axis}")
    half = float(half_angle_deg)
    if not (0.0 < half <= 90.0) or not np.isfinite(half):
        raise ValueError(f"reach half angle must be in (0, 90] deg, got {half_angle_deg}")
    cos_a = float(np.clip(np.dot(np.asarray(normal, dtype=np.float64), axis / norm), -1.0, 1.0))
    angle = math.degrees(math.acos(cos_a))
    return angle, 1.0 - angle / half, angle <= half


def evaluate_face(normal, scenario: Scenario, params, geometry: FaceGeometry, *,
                  spec: "vrs.VirtualReturnSpec", rollout_h=None, rollout_steps=None) -> dict:
    """对一个候选拍面法向过四道硬门并评 J_phys；任一门不过 → reward=GATED_REWARD（有限哨兵）。

    物理全部转发 virtual_return_scorer：冲量律内的 orient_normal 是符号不变的平面动力学，
    面身份只由这里的 signed_face_gate 判定（判分器 score() 的同款分工）。
    """
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-6:
        raise ValueError(f"candidate normal must be unit, got norm={norm}")
    yaw, pitch = angles_from_face(n)
    out = {
        "normal": [float(x) for x in n],
        "signed_yaw_deg": yaw, "signed_pitch_deg": pitch,
        "signed_face_ok": False, "signed_dot": 0.0,
        "contact_ok": False, "approach_speed": float("nan"),
        "net_crossed": False, "net_clear": False, "net_margin_m": float("nan"),
        "landing_valid": False, "on_opponent": False,
        "landing_xy": [float("nan")] * 2, "landing_error_m": float("nan"),
        "flight_time_s": float("nan"),
        "outgoing_vel": [float("nan")] * 3, "outgoing_spin": [float("nan")] * 3,
        "outgoing_speed": float("nan"), "outgoing_spin_mag": float("nan"),
        "legal": False, "reward_physics": GATED_REWARD,
    }

    signed_ok, dot = signed_face_gate(n, scenario.racket_vel)
    out["signed_face_ok"], out["signed_dot"] = bool(signed_ok), dot
    if not signed_ok:
        return out  # 硬门 1 fail-closed：反面/垂直面直接出局，不跑飞行

    oriented = vrs.orient_normal(n, scenario.ball_vel, scenario.racket_vel)
    approach = float(np.dot(scenario.racket_vel, oriented))
    out["approach_speed"] = approach
    out["contact_ok"] = approach > spec.min_approach_speed
    if not out["contact_ok"]:
        return out  # 硬门 2：phantom-block 门（判分器同阈值）

    v_out, w_out = vrs.predict_paddle_contact(
        scenario.ball_vel, scenario.racket_vel, n, scenario.ball_spin, params)
    events = vrs.coarse_flight_events(
        scenario.contact_pos, v_out, w_out, params,
        contact_plane_z=geometry.table_surface_z + params.ball_radius,
        net_x=geometry.net_x,
        h=spec.rollout_h if rollout_h is None else float(rollout_h),
        n_steps=spec.rollout_steps if rollout_steps is None else int(rollout_steps))

    out["outgoing_vel"] = [float(x) for x in v_out]
    out["outgoing_spin"] = [float(x) for x in w_out]
    out["outgoing_speed"] = float(np.linalg.norm(v_out))
    out["outgoing_spin_mag"] = float(np.linalg.norm(w_out))
    out["net_crossed"] = bool(events.net_crossed)
    net_clear_z = geometry.table_surface_z + geometry.net_height + params.ball_radius
    if events.net_crossed:
        out["net_margin_m"] = float(events.net_z - net_clear_z)
    out["net_clear"] = bool(events.net_crossed and events.net_z > net_clear_z)
    out["landing_valid"] = bool(events.landing_valid)
    if events.landing_valid:
        out["landing_xy"] = [float(x) for x in events.landing_xy]
        out["flight_time_s"] = float(events.flight_time)
        out["on_opponent"] = bool(
            events.landing_xy[0] > geometry.net_x
            and events.landing_xy[0] <= geometry.far_x
            and abs(float(events.landing_xy[1])) <= geometry.half_width)
        out["landing_error_m"] = float(
            np.linalg.norm(events.landing_xy - scenario.target_landing_xy))

    legal = bool(out["net_clear"] and out["landing_valid"] and out["on_opponent"])
    out["legal"] = legal
    if legal:  # 硬门 3+4 全过才有 reward
        out["reward_physics"] = float(reward_physics_terms(
            net_margin=out["net_margin_m"], land_err=out["landing_error_m"],
            out_speed=out["outgoing_speed"], out_spin_mag=out["outgoing_spin_mag"]))
    return out


# ---------------------------------------------------------------------------------------------
def _objective(detail, reach):
    """候选的比较分：物理分 + （可达目标才有的）圆锥余量项；可达硬门不过 → GATED。"""
    if detail["reward_physics"] <= GATED_REWARD:
        return GATED_REWARD, None
    if reach is None:
        return detail["reward_physics"], None
    angle, margin, ok = reach_cone_margin(detail["normal"], reach["axis"],
                                          reach["half_angle_deg"])
    info = {"reach_angle_deg": angle, "reach_margin": margin, "reach_ok": bool(ok)}
    if not ok:
        return GATED_REWARD, info
    return detail["reward_physics"] + W_REACH * margin, info


def optimize_face(scenario, params, geometry, *, spec, reach=None, coarse_deg=6.0,
                  refine_levels=2, rollout_h=None, rollout_steps=None,
                  landscape_out=None):
    """整球面粗网格 + 局部细化的确定性搜索；找不到任何合法候选 fail-closed。

    reach=None → unconstrained physics optimum；reach={"axis","half_angle_deg"} → 可达最优。
    landscape_out（list）非 None 时收集粗网格行 [yaw, pitch, J_phys, legal]（全有限）。
    """
    step = float(coarse_deg)
    if not (0.5 <= step <= 45.0):
        raise ValueError(f"coarse_deg must be in [0.5, 45], got {coarse_deg}")

    def _eval(yaw, pitch):
        detail = evaluate_face(face_from_angles(yaw, pitch), scenario, params, geometry,
                               spec=spec, rollout_h=rollout_h, rollout_steps=rollout_steps)
        score, reach_info = _objective(detail, reach)
        if reach_info is not None:
            detail.update(reach_info)
        return score, detail

    best = None  # (score, yaw, pitch, detail)
    yaws = np.arange(-180.0, 180.0, step)
    pitches = np.arange(-85.0, 85.0 + 1e-9, step)
    for yaw in yaws:
        for pitch in pitches:
            score, detail = _eval(float(yaw), float(pitch))
            if landscape_out is not None:
                landscape_out.append([round(float(yaw), 6), round(float(pitch), 6),
                                      float(detail["reward_physics"]),
                                      int(detail["legal"])])
            if best is None or score > best[0]:
                best = (score, float(yaw), float(pitch), detail)

    if best is None or best[0] <= GATED_REWARD:
        kind = "reachable" if reach is not None else "unconstrained"
        raise SystemExit(
            f"[FATAL] no legal {kind} hitting face found (all candidates gated) — "
            f"check ball/racket kinematics{' or widen the reach cone' if reach else ''}")

    for _ in range(max(0, int(refine_levels))):
        step /= 3.0
        _, yaw0, pitch0, _ = best
        for dy in np.linspace(-3.0, 3.0, 7):
            for dp in np.linspace(-3.0, 3.0, 7):
                yaw = (yaw0 + dy * step + 180.0) % 360.0 - 180.0
                pitch = float(np.clip(pitch0 + dp * step, -89.5, 89.5))
                score, detail = _eval(yaw, pitch)
                if score > best[0]:
                    best = (score, yaw, pitch, detail)

    score, _, _, detail = best
    detail["objective_score"] = float(score)
    return detail


def negative_control(detail, scenario, params, geometry, *, spec,
                     rollout_h=None, rollout_steps=None):
    """n/-n 负控：评 -n*，必须 signed 门不过且 reward 显著更差，否则当场报错（fail-closed）。"""
    flipped = -np.asarray(detail["normal"], dtype=np.float64)
    ev = evaluate_face(flipped, scenario, params, geometry, spec=spec,
                       rollout_h=rollout_h, rollout_steps=rollout_steps)
    gap = float(detail["reward_physics"]) - float(ev["reward_physics"])
    if ev["signed_face_ok"] or ev["legal"] or gap < NEG_CONTROL_MIN_GAP:
        raise RuntimeError(
            f"n/-n negative control FAILED: flipped signed_ok={ev['signed_face_ok']} "
            f"legal={ev['legal']} gap={gap:.3f} < {NEG_CONTROL_MIN_GAP} — the signed gate "
            f"is not doing its job; refusing to emit a manifest")
    ev["reward_gap_vs_optimum"] = gap
    return ev


def bucket_suggestion(reachable_detail, scenario, params, geometry, *, spec,
                      rollout_h=None, rollout_steps=None):
    """按比例缩放可达最优的 signed yaw/pitch，逐个重过硬门，给 train/interp/OOD 建议表。"""
    yaw0 = float(reachable_detail["signed_yaw_deg"])
    pitch0 = float(reachable_detail["signed_pitch_deg"])
    table = {}
    for bucket, ratios in BUCKET_CONVENTION.items():
        rows = []
        for ratio in ratios:
            yaw, pitch = ratio * yaw0, ratio * pitch0
            ev = evaluate_face(face_from_angles(yaw, pitch), scenario, params, geometry,
                               spec=spec, rollout_h=rollout_h, rollout_steps=rollout_steps)
            rows.append({
                "ratio": float(ratio),
                "signed_yaw_deg": float(yaw), "signed_pitch_deg": float(pitch),
                "normal": ev["normal"], "legal": bool(ev["legal"]),
                "reward_physics": float(ev["reward_physics"]),
                "net_margin_m": None if not np.isfinite(ev["net_margin_m"])
                                else float(ev["net_margin_m"]),
                "landing_error_m": None if not np.isfinite(ev["landing_error_m"])
                                   else float(ev["landing_error_m"]),
            })
        table[bucket] = rows
    return table


# ---------------------------------------------------------------------------------------------
def _jsonable(obj):
    """payload 有限值化：NaN/Inf → None（canonical dump 用 allow_nan=False,双保险）。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def canonical_sha256(payload):
    """一层内容寻址：canonical JSON（排序键、紧凑分隔、禁 NaN）的 sha256。"""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_pipeline(scenario, *, geometry, params=None, reach_axis=None,
                 reach_half_angle_deg=35.0, coarse_deg=6.0, refine_levels=2,
                 rollout_h=None, rollout_steps=None):
    """完整求解：两个最优 + gap + 负控 + landscape + 分桶。返回 manifest payload（dict）。"""
    if params is None:
        params = vrs.load_venue_params()
    spec = vrs.VirtualReturnSpec(
        table_surface_z=geometry.table_surface_z, net_x=geometry.net_x,
        far_x=geometry.far_x, half_width=geometry.half_width,
        net_height=geometry.net_height)
    if reach_axis is None:
        reach_axis = scenario.racket_vel  # "超前面"先验：默认可达轴 = 拍速方向
    reach = {"axis": [float(x) for x in np.asarray(reach_axis, dtype=np.float64)],
             "half_angle_deg": float(reach_half_angle_deg)}
    # 圆锥参数先行校验（fail-closed，而不是在第一个候选处才炸）
    reach_cone_margin(face_from_angles(0.0, 0.0), reach["axis"], reach["half_angle_deg"])

    kw = dict(spec=spec, coarse_deg=coarse_deg, refine_levels=refine_levels,
              rollout_h=rollout_h, rollout_steps=rollout_steps)
    landscape = []
    unconstrained = optimize_face(scenario, params, geometry, reach=None,
                                  landscape_out=landscape, **kw)
    reachable = optimize_face(scenario, params, geometry, reach=reach, **kw)
    reward_gap = float(unconstrained["reward_physics"]) - float(reachable["reward_physics"])
    if reward_gap < -1e-9:
        raise RuntimeError(  # 约束不可能帮倒忙；出现负 gap = 搜索或门有 bug,fail-closed
            f"reward gap is negative ({reward_gap}); constrained beat unconstrained")

    neg_unc = negative_control(unconstrained, scenario, params, geometry, spec=spec,
                               rollout_h=rollout_h, rollout_steps=rollout_steps)
    neg_reach = negative_control(reachable, scenario, params, geometry, spec=spec,
                                 rollout_h=rollout_h, rollout_steps=rollout_steps)
    buckets = bucket_suggestion(reachable, scenario, params, geometry, spec=spec,
                                rollout_h=rollout_h, rollout_steps=rollout_steps)

    payload = {
        "tool": "optimize_reachable_face.py",
        "scenario": {
            "ball_vel": [float(x) for x in scenario.ball_vel],
            "ball_spin": [float(x) for x in scenario.ball_spin],
            "contact_pos": [float(x) for x in scenario.contact_pos],
            "racket_vel": [float(x) for x in scenario.racket_vel],
            "target_landing_xy": [float(x) for x in scenario.target_landing_xy],
        },
        "physics": {"venue_yaml": params.source_path,
                    "venue_yaml_sha256": _sha256_file(params.source_path),
                    "implementation": "virtual_return_scorer (RK4 flight + fitted paddle impulse)"},
        "geometry": {"table_surface_z": geometry.table_surface_z, "net_x": geometry.net_x,
                     "far_x": geometry.far_x, "half_width": geometry.half_width,
                     "net_height": geometry.net_height,
                     "source": "hope_planner.constants.TableParams + venue_ball_sampler 摆位"},
        "reward_weights": {"legal_base": LEGAL_BASE, "w_net": W_NET,
                           "net_margin_cap_m": NET_MARGIN_CAP, "w_land": W_LAND,
                           "land_err_scale_m": LAND_ERR_SCALE, "w_speed": W_SPEED,
                           "out_speed_cap": OUT_SPEED_CAP, "w_spin": W_SPIN,
                           "out_spin_cap": OUT_SPIN_CAP, "w_reach": W_REACH,
                           "gated_reward": GATED_REWARD},
        "optimizer": {"coarse_deg": float(coarse_deg), "refine_levels": int(refine_levels),
                      "rollout_h": spec.rollout_h if rollout_h is None else float(rollout_h),
                      "rollout_steps": (spec.rollout_steps if rollout_steps is None
                                        else int(rollout_steps))},
        "reachability": {"model": "cone-approx (零阶近似)", "axis": reach["axis"],
                         "half_angle_deg": reach["half_angle_deg"],
                         "todo": "full 31-joint IK joint-limit-margin reachability"},
        "unconstrained_optimum": unconstrained,
        "reachable_optimum": reachable,
        "reward_gap_physics": reward_gap,
        "negative_control": {"min_gap_required": NEG_CONTROL_MIN_GAP,
                             "unconstrained_flipped": neg_unc,
                             "reachable_flipped": neg_reach},
        "objective_landscape": {"grid": "yaw x pitch coarse grid (signed, deg)",
                                "columns": ["signed_yaw_deg", "signed_pitch_deg",
                                            "reward_physics", "legal"],
                                "gated_reward_sentinel": GATED_REWARD,
                                "rows": landscape},
        "bucket_convention": {k: list(v) for k, v in BUCKET_CONVENTION.items()},
        "bucket_note": BUCKET_NOTE,
        "bucket_suggestion": buckets,
    }
    return _jsonable(payload)


# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ball-vel", required=True, help="来球触球瞬间速度 vx,vy,vz（vx<0 朝机器人）")
    ap.add_argument("--ball-spin", default="0,0,0", help="来球旋转 wx,wy,wz (rad/s)，默认无旋")
    ap.add_argument("--contact-pos", required=True, help="触球位置 x,y,z（env 系，我方半区）")
    ap.add_argument("--racket-vel", required=True, help="触球瞬间拍速 vx,vy,vz")
    ap.add_argument("--target-landing", required=True, help="目标落点 x,y（对方台内）")
    ap.add_argument("--reach-axis", default=None,
                    help="可达圆锥轴 x,y,z（默认=拍速方向，'超前面'先验）")
    ap.add_argument("--reach-half-angle-deg", type=float, default=35.0,
                    help="可达圆锥半角（度，(0,90]；简化近似，完整 IK 余量是 TODO）")
    ap.add_argument("--coarse-deg", type=float, default=6.0, help="粗网格步长（度）")
    ap.add_argument("--refine-levels", type=int, default=2, help="局部细化层数")
    ap.add_argument("--rollout-steps", type=int, default=None,
                    help="飞行步数（默认判分器 100 步 × 10 ms）")
    ap.add_argument("--table-near-x", type=float, default=DEFAULT_TABLE_NEAR_X,
                    help="近台边 env x（训练虚拟台摆位默认 0.5）")
    ap.add_argument("--table-surface-z", type=float, default=DEFAULT_TABLE_SURFACE_Z,
                    help="台面 env z（默认 0.76）")
    ap.add_argument("--json", required=True, help="输出 manifest JSON 路径")
    args = ap.parse_args(argv)

    geometry = build_geometry(args.table_near_x, args.table_surface_z)
    scenario = build_scenario(
        ball_vel=_vec(args.ball_vel, "ball_vel", 3),
        ball_spin=_vec(args.ball_spin, "ball_spin", 3),
        contact_pos=_vec(args.contact_pos, "contact_pos", 3),
        racket_vel=_vec(args.racket_vel, "racket_vel", 3),
        target_landing_xy=_vec(args.target_landing, "target_landing", 2),
        geometry=geometry)
    reach_axis = None if args.reach_axis is None else _vec(args.reach_axis, "reach_axis", 3)

    payload = run_pipeline(
        scenario, geometry=geometry, reach_axis=reach_axis,
        reach_half_angle_deg=args.reach_half_angle_deg, coarse_deg=args.coarse_deg,
        refine_levels=args.refine_levels, rollout_steps=args.rollout_steps)
    manifest = {"kind": MANIFEST_KIND, "schema_version": SCHEMA_VERSION,
                "sha256": canonical_sha256(payload), "payload": payload}
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, allow_nan=False)

    u, r = payload["unconstrained_optimum"], payload["reachable_optimum"]
    print("[reachable-face] 人话：物理最优拍面 vs 可达最优拍面（signed yaw/pitch,度）+ 代价")
    print(f"  physics optimum:   yaw={u['signed_yaw_deg']:+7.2f} pitch={u['signed_pitch_deg']:+7.2f} "
          f"J_phys={u['reward_physics']:.4f} net_margin={u['net_margin_m']:.3f} m "
          f"land_err={u['landing_error_m']:.3f} m")
    print(f"  reachable optimum: yaw={r['signed_yaw_deg']:+7.2f} pitch={r['signed_pitch_deg']:+7.2f} "
          f"J_phys={r['reward_physics']:.4f} net_margin={r['net_margin_m']:.3f} m "
          f"land_err={r['landing_error_m']:.3f} m reach_angle={r.get('reach_angle_deg', 0.0):.1f}°")
    print(f"  reward gap (为可达性付出的物理分) = {payload['reward_gap_physics']:.4f}")
    print(f"  n/-n 负控：flipped signed 门不过,gap ≥ {NEG_CONTROL_MIN_GAP}(已断言)")
    print(f"  分桶约定：train {BUCKET_CONVENTION['train']} interp "
          f"{BUCKET_CONVENTION['interpolation']} OOD {BUCKET_CONVENTION['ood']}(写死进 manifest)")
    print(f"[reachable-face] manifest sha256={manifest['sha256'][:16]}… -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
