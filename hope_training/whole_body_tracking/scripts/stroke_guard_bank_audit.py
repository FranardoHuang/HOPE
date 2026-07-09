#!/usr/bin/env python3
"""Offline stroke-guard audit of EXISTING stage-1 bank npz files (统计模式离线版).

人话:不重解题——直接拿已发货题库里每道题的需求拍速 |demanded_vel|,配上 clip 的行程
L_deep(引拍最深帧→触球帧拍面弧长,行程账本口径)和拍点加速度包络 a_max(v4rg 对实测
×1.5,时间律工具同一预算口径),报"**若开行程守卫会拦掉多少题**"。判据与出题器
gen_stage1_questions.py --stroke-guard 是同一实现(StrokeGuard / blade_acc_envelope /
resolve_stroke_guard 直接 import,不抄第二份公式)——必要条件 sound-reject,PASS ≠ 可行
(docs/research/stroke_interface_survey_2026-07-09.md §3.1)。

用途 = 行程守卫工单第 3 步:现役六对的 v2 exam 卷逐卷统计,franco 拍板"守卫默认开关"
的直接数据(预期 v5 反手卷拦得多:L_bh 0.497 m 短 + v* 高)。

触球帧口径(fail-loud,绝不静默猜):
  1. 默认取 bank meta_json 的 clips[<name>].anchor_frame —— 就是出题时用的锚帧;
  2. meta 里没有(老代际 bank)则必须手传 --anchor-frame name:frame,否则拒审。
blade 路径 = clip npz 存储 body 数组的 FK 缓存(synthesize_timing.blade_positions,
腕体+拍座偏置)。位置对 rally_yaw 不变,弧长对旋转不变,无需 canonicalize。

USAGE(pod,mjeval venv 或任意 numpy 环境):
    python scripts/stroke_guard_bank_audit.py \
        --bank /path/s1bank_v5hLs_exam.npz \
        --clip forehand:/path/hope_forehand_v5hLs_cal.npz \
        --clip backhand:/path/hope_backhand_v5hLs_cal.npz \
        --budget-clips /path/hope_forehand_v4rg_cal.npz /path/hope_backhand_v4rg_cal.npz

Unit tests: tests/test_stroke_guard_stage1.py (audit section). numpy-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# THE guard implementation (single source): gen_stage1_questions.py
gsq = _load_mod("sga_gen", os.path.join(HERE, "gen_stage1_questions.py"))


def _bank_meta(data) -> dict:
    if "meta_json" not in data:
        raise SystemExit("bank has no meta_json — pre-audit-round-2 bank; regenerate, or "
                         "audit is meaningless on a bank we cannot even date")
    try:
        return json.loads(bytes(np.asarray(data["meta_json"], dtype=np.uint8)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"bank meta_json is not valid JSON ({exc}) — legacy repr format; "
                         f"regenerate the bank before auditing") from None


def audit_bank(bank_path: str, clip_paths: dict, budget_clips, budget_scale: float = 1.5,
               v0: float = 0.0, anchor_overrides: dict | None = None) -> dict:
    """Per-clip stroke-guard stats over an existing bank. Returns {clip_name: row dict}.

    row = dict(questions, over_budget, frac, L_deep_m, deep_frame, contact_frame,
               a_max_mps2, v_star_cap_mps, v_star_max_mps). fail-loud on every missing
    ingredient (clip entry, anchor frame, demanded_vel) — 静默跳过会让分母变假.
    """
    anchor_overrides = dict(anchor_overrides or {})
    data = np.load(bank_path)
    meta = _bank_meta(data)
    a_max, env = gsq.resolve_stroke_guard("stats", list(budget_clips), budget_scale)
    es = gsq._stroke_ledger()

    out = {}
    for name, clip_npz in clip_paths.items():
        key = f"{name}/demanded_vel"
        if key not in data:
            raise SystemExit(f"bank {bank_path!r} has no clip {name!r} (missing {key!r})")
        v_stars = np.linalg.norm(np.asarray(data[key], dtype=np.float64), axis=1)
        if not len(v_stars):
            raise SystemExit(f"bank clip {name!r} is empty — nothing to audit")
        if name in anchor_overrides:
            c = int(anchor_overrides[name])
        else:
            c = ((meta.get("clips") or {}).get(name) or {}).get("anchor_frame")
            if c is None:
                raise SystemExit(
                    f"bank meta has no clips[{name!r}].anchor_frame (older-generation bank) "
                    f"— pass --anchor-frame {name}:<frame> (the contact frame the bank was "
                    f"generated at) instead of letting the audit guess")
        d = dict(np.load(clip_npz))
        missing = [k for k in ("body_pos_w", "body_quat_w") if k not in d]
        if missing:
            raise SystemExit(f"clip {clip_npz!r} lacks {missing} — need the stored body "
                             f"arrays (FK cache) for the blade path")
        blade = es.st.blade_positions(d)
        guard = gsq.StrokeGuard.from_blade_path(name, blade, int(c), a_max, v0=v0)
        a_mins = np.array([guard.check(float(v))[0] for v in v_stars])
        out[name] = dict(
            questions=int(len(v_stars)),
            over_budget=int(guard.over_budget_count),
            frac=guard.over_budget_count / len(v_stars),
            L_deep_m=guard.L_deep, deep_frame=guard.deep_frame,
            contact_frame=guard.contact_frame,
            a_max_mps2=guard.a_max, v_star_cap_mps=guard.v_star_cap,
            v_star_max_mps=float(v_stars.max()),
            a_min_med=float(np.median(a_mins)), a_min_max=float(a_mins.max()),
        )
    out["_meta"] = dict(bank=os.path.abspath(bank_path), a_max_mps2=a_max,
                        envelope_per_clip=env, budget_scale=budget_scale, v0_mps=v0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, help="existing stage-1 bank npz (e.g. a v2 exam 卷)")
    ap.add_argument("--clip", action="append", required=True,
                    help="name:path.npz — the SAME motion npz the bank was generated from "
                         "(repeatable; name = forehand/backhand)")
    ap.add_argument("--budget-clips", nargs="+", required=True,
                    help="npz clips whose blade-point |acc| envelope defines a_max "
                         "(v4rg 对 = the proven-executable floor)")
    ap.add_argument("--budget-scale", type=float, default=1.5,
                    help="a_max = envelope × this (default 1.5, 时间律预算口径)")
    ap.add_argument("--v0", type=float, default=0.0,
                    help="blade speed at stroke start [m/s] (default 0 = ready 静止起步)")
    ap.add_argument("--anchor-frame", action="append", default=[],
                    help="name:frame override when the bank meta lacks anchor_frame "
                         "(older banks); e.g. --anchor-frame backhand:23")
    args = ap.parse_args(argv)

    clips = {}
    for spec in args.clip:
        name, _, path = spec.partition(":")
        clips[name] = path
    overrides = {}
    for spec in args.anchor_frame:
        name, _, frame = spec.partition(":")
        overrides[name] = int(frame)

    res = audit_bank(args.bank, clips, args.budget_clips, args.budget_scale, args.v0,
                     overrides)
    m = res.pop("_meta")
    env_txt = ", ".join(f"{k} {v:.2f}" for k, v in m["envelope_per_clip"].items())
    print(f"[audit] bank={m['bank']}")
    print(f"[audit] a_max = max{{{env_txt}}} × {m['budget_scale']:g} = "
          f"{m['a_max_mps2']:.2f} m/s² (blade-point envelope; necessary-condition judge, "
          f"PASS != feasible)")
    print("| clip | questions | would-reject | frac | L_deep [m] | v* cap [m/s] | "
          "v* max in bank [m/s] | a_min med/max [m/s²] |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in res.items():
        print(f"| {name} | {r['questions']} | {r['over_budget']} | {r['frac']:.0%} "
              f"| {r['L_deep_m']:.3f} | {r['v_star_cap_mps']:.2f} | {r['v_star_max_mps']:.2f} "
              f"| {r['a_min_med']:.2f} / {r['a_min_max']:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
