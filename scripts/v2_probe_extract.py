#!/usr/bin/env python3
"""v2 probe 出数提取器:tensorboard 事件文件 -> 定权计算器的 measured JSON(半自动)。

人话:probe 跑完后在 pod 上执行本脚本,它扫 run 目录的 TB 标量,把定权公式要的实测量
(ρ_I、ρ_Q、capture/legal 率、门内 landing 值、‖Δa‖² 分位、T_c)能自动算的自动算,
算不了的列出缺口让人工补。两步用法:

  # 第 1 步:看这个 run 到底记了哪些标量(键名因谱系/版本会漂,先侦察)
  /workspace/hope_isaac_venv/bin/python scripts/v2_probe_extract.py <run_dir> --list-tags

  # 第 2 步:提取(最后 N 个迭代的均值;缺什么会在 missing 里点名)
  /workspace/hope_isaac_venv/bin/python scripts/v2_probe_extract.py <run_dir> \
      --lineage v4rg_runtime_order_v3 --last-n 50 > measured_partial.json

设计约定(与 scripts/v2_weight_calibration.py 的输入一一对应):
- rho_I = Σ_六项(Episode_Reward/<motion_*> 均值) / (I_weight_sum × dt)——模仿核达成度;
  权重与 dt 从本脚本的 LINEAGE_WEIGHTS 表取(v2.2 默认包数值,换包要改表,fail-loud)。
- rho_Q = 窗内三核均值:Episode_Reward/racket_{position,velocity,normal} ÷ 权重 ÷ dt ÷ duty
  的均值(duty=窗步数/T_c,由 strike_window_hit_rate 实测得 duty,不用假设值)。
- p_capture / p_legal:优先读 Metrics/*capture*、*legal*、virtual_landing_valid 类计数标量;
  找不到就进 missing。
- E_land_value_per_legal:legal_base 模式下 = Episode_Reward/virtual_landing ÷ w_land ÷ dt
  ÷ (p_legal×每步拍频);probe 若跑的是旧 climb 模式,则由落点分布离线重算——进 missing,
  给出人工公式。
- action_rate_sq_p95:TB 只有均值时输出 mean 并在 notes 标注"p95≈3×mean 保守替代"。
- T_c_steps:每拍周期 = 1/(exact_strike 触发频率×dt);读 *exact*fire*/hit_rate 类标量推。

输出 JSON 结构:{"measured": {...可直接喂 calibrate 的键...}, "raw": {标量->均值},
"missing": [人工补的量+怎么补], "notes": [...]}
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

DT = 0.02
# v2.2 默认包权重(与 train.py _REWARD_PACK_V2_* 一致;换包/换谱系先改这里,不许静默错配)
LINEAGE_WEIGHTS = {
    "mimic_terms": {
        "motion_global_anchor_ori": 0.5,
        "motion_body_pos": 1.0,
        "motion_body_ori": 1.0,
        "motion_body_lin_vel": 1.0,
        "motion_body_ang_vel": 1.0,
    },
    "quality_terms": {
        "racket_position": 60.0,
        "racket_velocity": 45.0,
        "racket_normal": 35.0,
    },
    "landing_term": ("virtual_landing", 3600.0),
    "action_rate_weight": 0.2,  # 取值端为正数记账
}


def _load_scalars(run_dir: str) -> dict:
    """读 run 目录全部 event 文件的标量 -> {tag: [(step, value), ...]}(纯 TF 事件解析)。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # pod 的 hope_isaac_venv 自带 tensorboard
        raise SystemExit(f"tensorboard is required to parse events: {exc}")
    files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not files:
        raise SystemExit(f"no tensorboard event files under {run_dir}")
    out: dict = {}
    for f in files:
        acc = EventAccumulator(os.path.dirname(f), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            out.setdefault(tag, [])
            out[tag] += [(e.step, float(e.value)) for e in acc.Scalars(tag)]
    for tag in out:
        out[tag] = sorted(set(out[tag]))
    return out


def _tail_mean(series, last_n: int):
    vals = [v for _, v in series[-last_n:]]
    return sum(vals) / len(vals) if vals else None


def _find(scalars: dict, *needles: str):
    """大小写无关子串匹配唯一标量;0 个或多个都返回 None(宁缺勿错)。"""
    hits = [t for t in scalars if all(n.lower() in t.lower() for n in needles)]
    return hits[0] if len(hits) == 1 else None


def extract(run_dir: str, lineage: str, last_n: int) -> dict:
    scalars = _load_scalars(run_dir)
    raw = {t: _tail_mean(s, last_n) for t, s in scalars.items()}
    measured: dict = {"motion_lineage": lineage, "window_steps": 13}
    missing: list = []
    notes: list = []

    # ρ_I
    mim = LINEAGE_WEIGHTS["mimic_terms"]
    got = {}
    for name, w in mim.items():
        tag = _find(scalars, "Episode_Reward", name) or _find(scalars, name)
        if tag is not None:
            got[name] = raw[tag] / (w * DT)
    if len(got) == len(mim):
        measured["I_weight_sum"] = sum(mim.values())
        measured["rho_I"] = sum(raw_v * w for (n, raw_v), w in zip(got.items(), mim.values())) / sum(mim.values())
        # 上行等价于按权重加权的核均值;逐项值进 notes 备查
        notes.append({"rho_I_per_term": got})
    else:
        missing.append(f"rho_I: 只匹配到 {sorted(got)} — 用 --list-tags 对键名后改 _find 针脚")

    # duty 与 T_c(由窗命中率实测)
    tag_hit = _find(scalars, "strike_window_hit_rate")
    duty = raw.get(tag_hit) if tag_hit else None
    if duty and 0.01 < duty < 0.5:
        notes.append({"strike_window_hit_rate": duty, "check": "C1 修复生效应 ~0.05-0.08"})
        measured["T_c_steps"] = round(13.0 / duty)
    else:
        missing.append("T_c_steps: 找不到可信 strike_window_hit_rate;人工=13/窗命中率")

    # ρ_Q(窗内核均值 = Episode 均值 ÷ w ÷ dt ÷ duty)
    if duty:
        q = LINEAGE_WEIGHTS["quality_terms"]
        vals = []
        for name, w in q.items():
            tag = _find(scalars, "Episode_Reward", name) or _find(scalars, name)
            if tag is not None:
                vals.append(raw[tag] / (w * DT * duty))
        if len(vals) == 3:
            measured["rho_Q"] = sum(vals) / 3.0
        else:
            missing.append("rho_Q: 三核标量没配齐")

    # capture / legal
    for key, needles_list in (
        ("p_capture_measured", [("capture",), ("vb", "fired")]),
        ("p_legal_measured", [("legal",), ("net_clear",), ("landing_valid",)]),
    ):
        tag = None
        for needles in needles_list:
            tag = _find(scalars, *needles)
            if tag:
                break
        if tag:
            measured[key] = raw[tag]
            notes.append({key: tag})
        else:
            missing.append(f"{key}: 无匹配标量(用 --list-tags 找 vb/capture/legal 家族)")
    notes.append("p_capture_target/p_legal_target 是设计目标非实测——prereg 里定,默认沿用 0.7")

    # E_land_value_per_legal
    land_name, w_land = LINEAGE_WEIGHTS["landing_term"]
    tag = _find(scalars, "Episode_Reward", land_name) or _find(scalars, land_name)
    if tag and duty and measured.get("p_legal_measured"):
        fires_per_step = duty / 13.0  # exact_strike 每拍一次
        denom = w_land * DT * fires_per_step * measured["p_legal_measured"]
        if denom > 0:
            e_land = raw[tag] / denom
            if 0.0 < e_land <= 1.0:
                measured["E_land_value_per_legal"] = e_land
            else:
                missing.append(
                    f"E_land_value_per_legal: 反推值 {e_land:.3f} 出 (0,1] 值域——probe 若跑旧 "
                    "climb 模式属正常,改用落点分布离线算 base+(1-base)·exp(-d²/σ²) 的均值"
                )
    else:
        missing.append("E_land_value_per_legal: landing 标量/duty/legal 率缺一不可")

    # Δa 分位
    tag = _find(scalars, "Episode_Reward", "action_rate")
    if tag is not None:
        mean_sq = abs(raw[tag]) / (LINEAGE_WEIGHTS["action_rate_weight"] * DT)
        measured["action_rate_sq_p95"] = round(3.0 * mean_sq, 3)
        notes.append("action_rate_sq_p95 = 3×均值 保守替代(TB 无分布);acc 同法")
    tag = _find(scalars, "Episode_Reward", "action_acc")
    if tag is not None:
        measured["action_acc_sq_p95"] = round(3.0 * abs(raw[tag]) / (0.05 * DT), 3)

    return {"measured": measured, "raw": {k: v for k, v in raw.items() if v is not None},
            "missing": missing, "notes": notes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--lineage", default=None)
    ap.add_argument("--last-n", type=int, default=50)
    ap.add_argument("--list-tags", action="store_true")
    args = ap.parse_args()
    if args.list_tags:
        for tag in sorted(_load_scalars(args.run_dir)):
            print(tag)
        return 0
    if not args.lineage:
        raise SystemExit("--lineage is mandatory (校准表按动作谱系归档)")
    print(json.dumps(extract(args.run_dir, args.lineage, args.last_n), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
