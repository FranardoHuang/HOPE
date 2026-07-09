#!/usr/bin/env python3
"""离线"击球面符号"建议工具 —— franco 2026-07-09 拍板语义:"哪面拍子超前就是哪面"。

WHAT THIS IS(人话)
    乒乓拍有两面(红面=拍框局部 +Y,黑面=−Y)。统一正反手策略的两个挥拍用的是**相反的两面**
    击球,而训练/判卷的拍面法向从 mount 约定的单一符号(+Y)出发,把反手按"不击球的那一面"
    记分——反手拍面误差被钉在 ~115-137°,拍面目标(normal_mode=velocity)永不可达,综合成功
    清零(M3b 判死取证;考卷 CF 换拍面=1.000 同签名;jiayi origin/hitter b7b7dfc 同病)。
    本工具对每个**登记过触球帧**的参考 clip,在触球帧取拍面法向 n(+Y 基准,即训练 FK 在
    mount_normal_sign=+1 下算出的那根轴)与拍速 v,给出

        建议符号 sign = sign(n · v)      (超前面 = 实际击球面)

    的建议表,供 franco 审定后写进训练 YAML 的 racket.mount_normal_sign_per_clip /
    判卷的 --mount-normal-sign-per-clip。

为什么必须离线算(franco 语义修正 2026-07-09)
    符号是**每 clip 的固定常量**,从参考 clip 触球帧的运动学离线算出、登记进配置表;
    **绝不能**在训练/判卷运行时用策略当前的拍速方向动态定符号——训练早期动作没成型、
    拍面可能整个反着,动态符号会把"反面"合法化。运行时路径(hope_commands.py /
    mujoco_eval_onnx.py)只读常量表,这是设计约定,不是实现细节。

计算口径(与训练 FK 逐项对齐)
    * 法向 n:R(wrist_quat) 的第 mount_normal_axis(=1,+Y)列——和
      RacketTargetCommand._compute_racket_state / _ensure_reference_strike_state 在
      mount_normal_sign=+1 时的读数一致(mount_quat 恒等)。_cal 资产的握拍标定已烘进
      腕关节,所以裸 +Y 就是标定后的拍面方向。
    * 拍速 v:优先用训练同款"干净击球速度"——拍面中心 FK 位置(wrist_pos + R@mount_offset)
      对 ±clean_window(默认 2)帧的中心差分,边界截断(clean_reference_strike_velocity
      口径);同时给存储速度(body_lin_vel + ω×r)的叉验,两者符号不一致 = WARN 如实报。
    * 触球帧:只认 cfg/strike_annotations.yaml 登记值(phase),没登记当场报错(fail-loud,
      speed-peak 自动检测是已知陷阱,不做静默回退)。

USAGE
    python scripts/suggest_face_sign.py CLIP.npz [CLIP2.npz ...] \
        [--annotations cfg/strike_annotations.yaml] [--body-order LIST|FILE] \
        [--clean-window 2] [--json OUT.json]

    Exit code: 0 = 全部给出明确建议; 2 = 有 clip 贴边(|cos|<0.10)或干净/存储速度符号打架
    (建议表照打,但这些 clip 需要人工复核)。

DEPENDENCIES
    numpy + PyYAML(读登记表)。不需要 isaac / torch / mujoco。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 复用产线同款常量与工具:mount 偏置 / 四元数→旋转矩阵 / 登记表读取(analyze_strike_phase 是
# 这些 clip 的现役分析工具,口径就是训练 FK 的口径);体序解析复用判炸器的 fail-loud 实现。
from analyze_strike_phase import (  # noqa: E402
    CLEAN_VEL_WINDOW,
    DEFAULT_ANNOTATIONS,
    MOUNT_OFFSET,
    NORMAL_AXIS,
    _annotation_frame,
    _annotation_keys,
    _load_annotations,
    quat_to_rot,
)
from audit_motion_npz import resolve_body_order  # noqa: E402

WRIST_BODY_NAME = "right_wrist_yaw_Link"
LEGACY_WRIST_INDEX = 31          # 32 体 npz 的历史约定(analyze_strike_phase.RACKET_BODY)
AMBIGUOUS_COS = 0.10             # |cos(n,v)| 低于此值 = 法向和拍速几乎垂直,符号"贴边",要人工复核
MIN_SPEED = 0.05                 # m/s;触球帧拍速低于此值时 n·v 没有方向意义,直接判贴边


def _resolve_wrist_index(data, npz_path: str, n_bodies: int, body_order: str | None) -> tuple[int, str]:
    """腕体列号解析:显式/npz 内嵌/旁车文件的体序表优先(fail-loud 口径同判炸器);
    都没有时,仅对 32 体 npz 退回历史约定列 31(analyze_strike_phase 口径),并在表里注明。"""
    try:
        names = resolve_body_order(body_order, npz_path, n_bodies, data=data)
    except ValueError:
        if body_order is None and n_bodies == 32:
            return LEGACY_WRIST_INDEX, f"约定列 {LEGACY_WRIST_INDEX}(32 体 npz,无体序表)"
        raise
    if WRIST_BODY_NAME not in names:
        raise ValueError(f"body order has no {WRIST_BODY_NAME} (source resolved OK, {n_bodies} bodies)")
    idx = names.index(WRIST_BODY_NAME)
    return idx, f"体序表列 {idx}"


def compute_face_sign(npz_path: str, contact_frame: int, *, body_order: str | None = None,
                      clean_window: int = CLEAN_VEL_WINDOW) -> dict:
    """单 clip 的建议符号计算(纯函数,便于单测)。

    返回 dict:suggested_sign(±1)、cos_clean/cos_raw(n̂·v̂)、angle_deg(n 与 v 的夹角)、
    speed_clean/speed_raw、ambiguous(贴边)、raw_agrees(干净/存储速度符号一致)等。
    """
    d = np.load(npz_path)
    P = d["body_pos_w"].astype(np.float64)      # (T, B, 3)
    Q = d["body_quat_w"].astype(np.float64)     # (T, B, 4) wxyz
    T, n_bodies = P.shape[0], P.shape[1]
    fps = int(np.asarray(d["fps"]).reshape(-1)[0])
    f = int(contact_frame)
    if not (0 <= f < T):
        raise ValueError(f"contact frame {f} out of range [0, {T}) for {npz_path}")
    widx, widx_src = _resolve_wrist_index(d, npz_path, n_bodies, body_order)

    R = quat_to_rot(Q[:, widx])                                   # (T,3,3)
    normal = R[:, :, NORMAL_AXIS]                                 # +Y 基准拍面轴(sign=+1 的训练 FK 读数)
    offset_w = np.einsum("tij,j->ti", R, MOUNT_OFFSET)            # (T,3)
    blade = P[:, widx] + offset_w                                 # 拍面中心 FK 位置

    # 训练同款干净击球速度:±W 帧中心差分,边界截断(clean_reference_strike_velocity 口径)。
    W = max(1, int(clean_window))
    lo, hi = max(0, f - W), min(T - 1, f + W)
    if hi == lo:
        raise ValueError(f"clip too short for a finite difference at frame {f} (T={T})")
    v_clean = (blade[hi] - blade[lo]) * fps / (hi - lo)

    # 存储速度叉验:body_lin_vel + ω×r(存储速度是重采样/插值产物,和 FK 位置最多 ~1 m/s 不一致,
    # 只用来核符号,不作主口径)。缺列(极老资产)时跳过叉验。
    v_raw = None
    if "body_lin_vel_w" in d.files and "body_ang_vel_w" in d.files:
        v_raw = (d["body_lin_vel_w"].astype(np.float64)[f, widx]
                 + np.cross(d["body_ang_vel_w"].astype(np.float64)[f, widx], offset_w[f]))

    n_hat = normal[f] / max(np.linalg.norm(normal[f]), 1e-12)

    def _cos(v):
        s = np.linalg.norm(v)
        return float(np.dot(n_hat, v / s)) if s > 1e-12 else 0.0, float(s)

    cos_clean, speed_clean = _cos(v_clean)
    suggested = 1.0 if cos_clean > 0.0 else -1.0
    cos_raw, speed_raw = _cos(v_raw) if v_raw is not None else (None, None)
    raw_agrees = None if cos_raw is None else (cos_raw > 0.0) == (cos_clean > 0.0)
    ambiguous = abs(cos_clean) < AMBIGUOUS_COS or speed_clean < MIN_SPEED
    return dict(
        npz=str(npz_path), T=T, fps=fps, contact_frame=f, wrist_index=widx, wrist_source=widx_src,
        suggested_sign=suggested, cos_clean=cos_clean, speed_clean=speed_clean,
        cos_raw=cos_raw, speed_raw=speed_raw, raw_agrees=raw_agrees, ambiguous=ambiguous,
        angle_deg=math.degrees(math.acos(max(-1.0, min(1.0, cos_clean)))),
        normal_w=[float(x) for x in n_hat], vel_clean_w=[float(x) for x in v_clean],
    )


def _expected_sign(stem: str):
    """命名先验,仅作对照列:正手应 +1;反手(swing/v5 系实拍语义)预期 −1。算出来不是就如实报,
    拍板归 franco——预期列不参与任何计算。"""
    if "forehand" in stem:
        return 1.0
    if "backhand" in stem:
        return -1.0
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="motion npz 路径(体序:内嵌 body_names / 旁车 "
                                             "body_order*.txt / --body-order;32 体退回约定列 31)")
    ap.add_argument("--annotations", default=DEFAULT_ANNOTATIONS,
                    help="触球帧登记表(默认 cfg/strike_annotations.yaml);没登记的 clip 当场报错")
    ap.add_argument("--body-order", default=None,
                    help="体序表(逗号列表或每行一名的文件;判炸器同款解析)")
    ap.add_argument("--clean-window", type=int, default=CLEAN_VEL_WINDOW,
                    help=f"干净拍速中心差分半窗(帧,默认 {CLEAN_VEL_WINDOW} = 训练 "
                         f"clean_strike_vel_window 口径)")
    ap.add_argument("--json", default=None, help="把建议表另存为 JSON(入档用)")
    args = ap.parse_args(argv)

    ann_map = _load_annotations(args.annotations)
    rows = []
    for path in args.clips:
        if not os.path.isfile(path):
            raise SystemExit(f"[FATAL] motion npz not found: {path}")
        stem = Path(path).stem
        # 登记键解析口径同 analyze_strike_phase(stem;artifacts/<name>:vN/motion.npz 用父目录名)。
        key = next((k for k in _annotation_keys(stem, path) if k in ann_map), None)
        if key is None:
            raise SystemExit(
                f"[FATAL] {stem}: no contact-frame annotation in {args.annotations} — 触球帧必须用"
                f"登记值(speed-peak 自动检测是已知陷阱,不做静默回退)。先把 phase 登记进注册表。")
        d = np.load(path)
        frame = _annotation_frame(ann_map[key], int(d["body_pos_w"].shape[0]))
        row = compute_face_sign(path, frame, body_order=args.body_order, clean_window=args.clean_window)
        row["clip"] = key
        row["phase"] = ann_map[key].get("phase")
        row["expected_sign"] = _expected_sign(stem)
        rows.append(row)

    # ---- 建议表 ---------------------------------------------------------------------------
    print(f"[face-sign] 语义:sign = sign(n·v) @ 登记触球帧(哪面超前就是哪面);"
          f"n=+Y 基准拍面轴,v=±{max(1, int(args.clean_window))} 帧干净拍速(训练口径)")
    hdr = (f"{'clip':<34} {'f/T':>9} {'|v|m/s':>7} {'cos(n,v)':>9} {'角度°':>7} "
           f"{'建议':>5} {'预期':>5} {'叉验':>5}  备注")
    print(hdr)
    print("-" * len(hdr))
    worst = 0
    for r in rows:
        notes = []
        if r["ambiguous"]:
            notes.append(f"贴边(|cos|<{AMBIGUOUS_COS} 或 |v|<{MIN_SPEED}),人工复核")
            worst = 2
        if r["raw_agrees"] is False:
            notes.append("干净/存储速度符号打架,人工复核")
            worst = 2
        exp = r["expected_sign"]
        if exp is not None and exp != r["suggested_sign"] and not r["ambiguous"]:
            notes.append("与命名先验不符——如实报,拍板归 franco")
        raw_mark = {None: "n/a", True: "一致", False: "打架"}[r["raw_agrees"]]
        print(f"{r['clip']:<34} {r['contact_frame']:>4}/{r['T']:<4} {r['speed_clean']:>7.3f} "
              f"{r['cos_clean']:>+9.4f} {r['angle_deg']:>7.1f} {r['suggested_sign']:>+5.0f} "
              f"{('%+d' % exp) if exp is not None else '  ?':>5} {raw_mark:>5}  {'; '.join(notes)}")
    print(f"[face-sign] YAML 形如: task.racket.mount_normal_sign_per_clip=["
          + ",".join(f"{r['suggested_sign']:+.0f}" for r in rows)
          + "]  (顺序 = motion_file clip 顺序,自行核对)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
        print(f"[face-sign] JSON -> {args.json}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
