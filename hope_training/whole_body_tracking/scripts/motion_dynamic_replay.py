#!/usr/bin/env python3
"""L2 motion-only DYNAMIC replay (真动力学回放) for retargeted motion .npz clips.

人话（这工具是干嘛的）
    以前的三件套都证明不了"这条参考轨迹在真物理里做得出来"：
      * audit_motion_npz.py    (L0 判炸器)   只看运动学数字，没有模拟器；
      * audit_self_collision.py (L1 自碰撞)  只 mj_forward 摆姿势，不走动力学；
      * replay_npz.py           (Isaac 渲染) 逐帧覆盖 qpos，纯放片子。
    三者的共同上限：mj_step_calls == 0。本工具是第一档真动力学证据：机器人
    free base 站在地上，只在序列开始 reset 一次，然后把参考关节角当 q_des 喂给
    厂商 deploy nominal PD（Isaac ImplicitActuator 的 clip(P-D,±limit) 合同），每个 physics
    substep 真调 mujoco.mj_step，让机器人自己扛住重力、接触和惯性去追这条轨迹。
    整条 clip（ready→prepare→strike→follow-through→recover）跑完后再保持
    hold_after_s，看它有没有摔、脚滑没滑、CoM 出没出支撑面、末端回没回 ready。

结论边界（写死，不许扩大解释）
    motion-only PASS 只证明【参考轨迹本身动力学可执行】——一个理想 PD 跟踪器
    在厂商 MJCF 里能跟着它不摔。它：
      * 不替代 policy replay（策略可能学不出这个跟踪器，观测/延迟/噪声都没进来）；
      * 不替代 vendor Gate3（厂商验收是另一条独立链路，合同不同）；
      * FAIL 也不等于轨迹必炸——PD 增益是固定的 deploy nominal 表，不是最优跟踪器。
    它的用途是当 L2 前置闸门：连理想 PD 都跟不住的参考，不要送进训练。

已测事实（2026-07-20，本地 vendor MJCF a3_pingpong.xml 实测，pod 跑真 clip 前必读）
    纯厂商 PD 连"stand keyframe 静态站立"都撑不满 2 s：~1.6 s 前倾超 0.7 rad。
    这不是本工具的 bug，是纯关节 PD 的物理：踝 pitch 刚度 2×50 Nm/rad 远小于
    倒立摆失稳刚度 m*g*h ≈ 58.3 kg × 9.81 × 0.86 m ≈ 490 Nm/rad（保留/去掉 MJCF
    原生 damping+frictionloss 都摔，CoM 初始就在踝轴前 1.5 cm）。与
    mujoco_eval_onnx P0 记录"deploy-faithful 静态站立 ~1.1 s 摔"互证：真机靠
    减速箱不可反驱/策略在平衡，MuJoCo 里静态 q_des 必倒。所以在真机 MJCF 上
    跑本工具时，主控应把它当【相对分级器】用——比较不同 clip 的存活时长
    (fall.time_s)、滑移、饱和——或显式放宽 fall 门限，别指望绝对 PASS。
    合成小模型（测试用）几何上 PD 可稳，PASS/FAIL 全链路在测试里验证。

复用与来源（防漂移声明）
    * MuJoCo robot 加载 / PD/actuator 合同：直接 import 同目录
      mujoco_eval_onnx.MujocoRobot（含 Isaac total-PD clip、速度限幅代理、
      自碰撞逐 substep 扫描）。不复制该类。
    * 厂商 deploy PD 与 MJCF armature 表
      (_A3_DEPLOY_NOMINAL_PD/_A3_MJCF_EXACT_ARMATURE)：转录自 source/whole_body_tracking/
      whole_body_tracking/robots/agibot_a3.py 的 ImplicitActuatorCfg 各组
      （约行 222–366）。PD/effort 以 deploy/URDF 原件为准，armature 以
      a3_pingpong.xml 全精度字节为准；parkour DR 的分组表只提供随机化语义，
      不替换 nominal。那个模块顶层 import isaaclab，本地/CI 装不起，
      只能转录并用 host test fail-loud 对拍。
    * vendor MJCF 默认路径解析：镜像 mujoco_eval_onnx.main()（行 4219–4234）
      的 repo 根定位 + a3_pingpong.xml 相对路径。
    * npz 关节顺序：audit_motion_npz.ISAAC_JOINT_NAMES（31 DoF Isaac
      articulation 顺序），同一定义 import，不复制。

硬合同（每条都有测试或断言兜底）
    1. 单次 reset：序列开始前 reset 一次（clip 首帧关节角 + 首帧 root 位姿 +
       零速度），此后绝不 hoist、不逐帧覆盖 qpos、不中途 teleport；
       JSON 里 reset_count 恒等于 1。
    2. 每个 physics substep 真调 mj_step；mj_step_calls 显式计数，且
       sim time 必须严格递增、与 mj_step_calls*sim_dt 对账。
    3. 任何 NaN（输入 npz、sim 状态、汇总指标）→ fail-closed，verdict FAIL。
    4. --pair 正反手两条 npz 顺序执行（中间站立 hold，不 reset），作为
       no-reset 连续序列检查。

输出 JSON（--out）关键字段（人话对照）
    mj_step_calls            真调 mj_step 的次数（=执行 tick 数 × substeps）
    sim_time_monotonic       物理时间是否严格递增
    fall                     有没有摔（root 高度 / 倾角双判据，含首摔时刻）
    root_z_min_m             root 最低高度
    tilt_rad                 躯干倾角 p50/p95/p99/max
    feet.left/right          每只脚：接触时间比例、法向力均值、最长离地时长、
                             支撑期水平速度峰值、累计滑移
    stance_min_foot_dist_m   双支撑期两脚最小水平间距
    legs_crossed             是否交叉腿（骨盆坐标系左右脚横向间隔的最小值）
    com_support_margin_min_m CoM 水平投影到支撑多边形（双足接触点凸包）的
                             最小 margin（正=在里面）
    saturation               q 软限位越界样本数 / qdot 限幅命中数 / 力矩饱和数
    tracking_err_rad         参考 vs 实际关节角误差 mean/p95/max
    recover                  hold 结束时与最终参考帧的误差 + root 残余速度
    verdict                  PASS/FAIL + 逐门原因（gates 列表）

用法
    python scripts/motion_dynamic_replay.py --motion CLIP.npz --out OUT.json \
        [--mjcf PATH] [--hold-after-s 1.0] [--substeps 4]
    python scripts/motion_dynamic_replay.py --pair FH.npz BH.npz --out OUT.json

    退出码：0 = PASS，2 = FAIL / 输入不合法（fail-closed）。

依赖
    numpy + mujoco + 同目录 mujoco_eval_onnx / audit_motion_npz。无 isaac、无
    torch、无 onnxruntime（MujocoRobot 不摸 ONNX 路径）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mujoco_eval_onnx import (  # noqa: E402
    DF_FALL_ROOT_Z_MIN,
    DF_FALL_TILT_RAD,
    FEET_BODIES,
    TRACKED_BODIES,
    MujocoRobot,
    sha256_file,
)
from audit_motion_npz import ISAAC_JOINT_NAMES  # noqa: E402

try:  # keep the pure helpers importable on hosts without mujoco
    import mujoco
except ImportError:  # pragma: no cover - exercised only where mujoco is absent
    mujoco = None


# ---------------------------------------------------------------------------
# vendor plant contract
# ---------------------------------------------------------------------------
# 人话：厂商 deploy/URDF/MJCF 原件的执行器 nominal
# （kp/kd/力矩上限/速度上限/armature）。
# TRANSCRIBED from hope_training/whole_body_tracking/source/whole_body_tracking/
# whole_body_tracking/robots/agibot_a3.py, ImplicitActuatorCfg groups
# legs/feet/waist/head/arms (lines ~222-366).  That module cannot be imported here (top-level isaaclab
# import), so the numbers are copied verbatim.  The parkour training table supplies the DR ranges,
# delay, and push recipe elsewhere; it must not overwrite this exact nominal plant.  Keyed by the
# joint name WITHOUT the left_/right_ side prefix.
_A3_DEPLOY_NOMINAL_PD: Dict[str, Tuple[float, float, float, float]] = {
    #  base joint name         kp     kd    effort  velocity
    "hip_yaw_joint":         ( 80.0,  3.0,  220.0,  12.0),
    "hip_roll_joint":        (120.0,  4.0,  220.0,  12.0),
    "hip_pitch_joint":       ( 80.0,  3.0,  220.0,  12.0),
    "knee_joint":            (250.0,  8.0,  320.0,  14.6),
    "ankle_pitch_joint":     ( 50.0,  2.0,  118.2,  10.8),
    "ankle_roll_joint":      ( 50.0,  2.0,   54.75, 19.3),
    "waist_yaw_joint":       ( 85.0,  3.0,  220.0,  12.0),
    "waist_roll_joint":      ( 50.0,  2.0,   46.0,  22.7),
    "waist_pitch_joint":     ( 50.0,  2.0,  118.0,   9.2),
    "head_yaw_joint":        ( 40.0,  2.0,    6.0,  12.7),
    "head_pitch_joint":      ( 40.0,  2.0,    6.0,  12.7),
    "shoulder_pitch_joint":  ( 40.0,  3.0,   60.0,  13.6),
    "shoulder_roll_joint":   ( 40.0,  3.0,   60.0,  13.6),
    "shoulder_yaw_joint":    ( 30.0,  2.0,   24.0,  15.7),
    "elbow_joint":           ( 30.0,  2.0,   24.0,  15.7),
    "wrist_roll_joint":      ( 30.0,  2.0,   24.0,  15.7),
    "wrist_pitch_joint":     ( 20.0,  2.0,    6.0,  12.7),
    "wrist_yaw_joint":       ( 20.0,  2.0,    6.0,  12.7),
}

# Exact a3_pingpong.xml armature values.  Do not round these to the parkour table's
# six-digit groups: that changes all 29 body joints and erases the distinct pitch/yaw wrists.
_A3_MJCF_EXACT_ARMATURE: Dict[str, float] = {
    "hip_yaw_joint": 0.06646569891,
    "hip_roll_joint": 0.06646569891,
    "hip_pitch_joint": 0.06646569891,
    "knee_joint": 0.1203404,
    "ankle_pitch_joint": 0.06444060531,
    "ankle_roll_joint": 0.02012630058,
    "waist_yaw_joint": 0.06646569891,
    "waist_roll_joint": 0.01462087613,
    "waist_pitch_joint": 0.08820859156,
    "head_yaw_joint": 0.0008100893338,
    "head_pitch_joint": 0.0008100893338,
    "shoulder_pitch_joint": 0.01208336871,
    "shoulder_roll_joint": 0.01208336871,
    "shoulder_yaw_joint": 0.004967351303,
    "elbow_joint": 0.004967351303,
    "wrist_roll_joint": 0.004967351303,
    "wrist_pitch_joint": 0.0008100893338,
    "wrist_yaw_joint": 0.0008100893338,
}

# vendor MJCF default path, mirroring mujoco_eval_onnx.main() (lines 4219-4234):
# here = scripts/, wbt = whole_body_tracking/, repo = two levels above wbt.
DEFAULT_MJCF_REL = (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
ROOT_BODY = "pelvis_link"          # MujocoRobot hard-requires this freejoint root
DEFAULT_FPS_EXPECTED = 50          # 生产 clip 都是 50 Hz；非 50 只警告不拒绝
DEFAULT_SUBSTEPS = 4               # 50 Hz tick × 4 substeps = 0.005 s 物理步（同训练）
DEFAULT_HOLD_AFTER_S = 1.0         # clip 结束后继续保持最后一帧参考的时长
DEFAULT_HOLD_BETWEEN_S = 1.0       # --pair 两条 clip 之间的站立 hold 时长

PASS, FAIL = "PASS", "FAIL"


@dataclass(frozen=True)
class ReplayContract:
    """Plant contract: joint set + vendor actuator table, all in npz column order."""

    name: str
    joint_names: Tuple[str, ...]
    body_names: Tuple[str, ...]
    kp: np.ndarray
    kd: np.ndarray
    effort_limits: np.ndarray
    velocity_limits: np.ndarray
    armature: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.joint_names)
        if n == 0 or len(set(self.joint_names)) != n:
            raise ValueError("contract joint_names must be non-empty and unique")
        if not self.body_names:
            raise ValueError("contract body_names must be non-empty")
        for label, arr, positive in (
            ("kp", self.kp, True),
            ("kd", self.kd, False),
            ("effort_limits", self.effort_limits, True),
            ("velocity_limits", self.velocity_limits, True),
        ):
            if arr.shape != (n,) or not np.isfinite(arr).all():
                raise ValueError(f"contract {label} must be {n} finite values")
            if positive and np.any(arr <= 0.0):
                raise ValueError(f"contract {label} must be strictly positive")
            if not positive and np.any(arr < 0.0):
                raise ValueError(f"contract {label} must be non-negative")
        if self.armature is not None:
            if (
                self.armature.shape != (n,)
                or not np.isfinite(self.armature).all()
                or np.any(self.armature < 0.0)
            ):
                raise ValueError(f"contract armature must be {n} finite non-negative values")


def a3_contract() -> ReplayContract:
    """The exact vendor deploy/MJCF A3 nominal in Isaac articulation order."""
    kp, kd, eff, vel, arm = [], [], [], [], []
    for name in ISAAC_JOINT_NAMES:
        base = name
        for prefix in ("left_", "right_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        if base not in _A3_DEPLOY_NOMINAL_PD:
            raise ValueError(f"no vendor PD entry for joint {name!r} (base {base!r})")
        if base not in _A3_MJCF_EXACT_ARMATURE:
            raise ValueError(f"no vendor armature entry for joint {name!r} (base {base!r})")
        row = _A3_DEPLOY_NOMINAL_PD[base]
        kp.append(row[0]); kd.append(row[1]); eff.append(row[2]); vel.append(row[3])
        arm.append(_A3_MJCF_EXACT_ARMATURE[base])
    return ReplayContract(
        name="agibot_a3_deploy_mjcf_nominal",
        joint_names=tuple(ISAAC_JOINT_NAMES),
        body_names=tuple(TRACKED_BODIES),
        kp=np.asarray(kp, np.float64),
        kd=np.asarray(kd, np.float64),
        effort_limits=np.asarray(eff, np.float64),
        velocity_limits=np.asarray(vel, np.float64),
        armature=np.asarray(arm, np.float64),
    )


def contract_from_json(path: str) -> ReplayContract:
    """Load a plant contract from JSON (tests / non-A3 diagnostic models)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    required = {"name", "joint_names", "body_names", "kp", "kd",
                "effort_limits", "velocity_limits"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"contract json missing keys: {sorted(missing)}")
    return ReplayContract(
        name=str(raw["name"]),
        joint_names=tuple(str(x) for x in raw["joint_names"]),
        body_names=tuple(str(x) for x in raw["body_names"]),
        kp=np.asarray(raw["kp"], np.float64),
        kd=np.asarray(raw["kd"], np.float64),
        effort_limits=np.asarray(raw["effort_limits"], np.float64),
        velocity_limits=np.asarray(raw["velocity_limits"], np.float64),
        armature=(
            np.asarray(raw["armature"], np.float64)
            if "armature" in raw
            else None
        ),
    )


def resolve_default_mjcf() -> str:
    """repo-root-relative vendor MJCF, exactly like mujoco_eval_onnx.main()."""
    here = os.path.dirname(os.path.abspath(__file__))
    wbt = os.path.dirname(here)
    repo = os.path.dirname(os.path.dirname(wbt))
    return os.path.join(repo, DEFAULT_MJCF_REL)


# ---------------------------------------------------------------------------
# motion loading (fail-closed)
# ---------------------------------------------------------------------------

@dataclass
class MotionClip:
    path: str
    joint_pos: np.ndarray      # (T, J) reference joint angles, contract order
    fps: int
    root_pos0: np.ndarray      # (3,) frame-0 root position (world)
    root_quat0: np.ndarray     # (4,) frame-0 root quaternion (wxyz, unit)


def load_motion(path: str, n_joints: int) -> MotionClip:
    """Load one motion npz; any missing key / NaN / shape mismatch raises."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"motion npz not found: {path}")
    with np.load(p, allow_pickle=True) as data:
        files = set(data.files)
        for key in ("joint_pos", "fps", "body_pos_w", "body_quat_w"):
            if key not in files:
                raise ValueError(f"{p.name}: npz missing required key {key!r}")
        q = np.asarray(data["joint_pos"], np.float64)
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        body_pos = np.asarray(data["body_pos_w"], np.float64)
        body_quat = np.asarray(data["body_quat_w"], np.float64)
        body_names = None
        if "body_names" in files:
            body_names = [str(x) for x in np.asarray(data["body_names"]).reshape(-1)]
    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError(f"{p.name}: joint_pos must be (T>=2, J), got {q.shape}")
    if q.shape[1] != n_joints:
        raise ValueError(
            f"{p.name}: joint_pos has {q.shape[1]} columns, contract expects {n_joints}"
        )
    if fps <= 0:
        raise ValueError(f"{p.name}: fps must be positive, got {fps}")
    if not np.isfinite(q).all():
        raise ValueError(f"{p.name}: NaN/Inf in joint_pos (fail-closed)")
    if body_pos.ndim != 3 or body_pos.shape[0] != q.shape[0] or body_pos.shape[2] != 3:
        raise ValueError(f"{p.name}: body_pos_w shape {body_pos.shape} invalid")
    if body_quat.ndim != 3 or body_quat.shape[0] != q.shape[0] or body_quat.shape[2] != 4:
        raise ValueError(f"{p.name}: body_quat_w shape {body_quat.shape} invalid")
    if body_names is not None and body_names and body_names[0] != ROOT_BODY:
        raise ValueError(
            f"{p.name}: body_names[0]={body_names[0]!r}, expected root {ROOT_BODY!r} "
            "(column 0 of body_pos_w/body_quat_w must be the free root)"
        )
    root_pos0 = body_pos[0, 0].copy()
    root_quat0 = body_quat[0, 0].copy()
    if not (np.isfinite(root_pos0).all() and np.isfinite(root_quat0).all()):
        raise ValueError(f"{p.name}: NaN/Inf in frame-0 root pose (fail-closed)")
    norm = float(np.linalg.norm(root_quat0))
    if norm < 1e-6:
        raise ValueError(f"{p.name}: frame-0 root quaternion is zero")
    return MotionClip(
        path=str(p), joint_pos=q, fps=fps,
        root_pos0=root_pos0, root_quat0=root_quat0 / norm,
    )


# ---------------------------------------------------------------------------
# timeline (clip frames + standing holds; --pair concatenates with NO reset)
# ---------------------------------------------------------------------------

@dataclass
class Timeline:
    q_des: np.ndarray            # (N, J) per-tick reference
    is_clip_tick: np.ndarray     # (N,) True on real clip frames (tracking scope)
    segments: List[dict]         # [{name, kind, start_tick, end_tick}] inclusive-exclusive
    fps: int


def build_timeline(clips: Sequence[MotionClip], *, hold_after_s: float,
                   hold_between_s: float) -> Timeline:
    if not clips:
        raise ValueError("timeline needs at least one clip")
    fps = clips[0].fps
    for c in clips[1:]:
        if c.fps != fps:
            raise ValueError(
                f"--pair clips disagree on fps: {clips[0].path}={fps}, {c.path}={c.fps}"
            )
    if hold_after_s < 0.0 or hold_between_s < 0.0:
        raise ValueError("hold durations must be non-negative")
    chunks: List[np.ndarray] = []
    flags: List[np.ndarray] = []
    segments: List[dict] = []
    tick = 0

    def push(name: str, kind: str, arr: np.ndarray, clip_flag: bool):
        nonlocal tick
        if arr.shape[0] == 0:
            return
        chunks.append(arr)
        flags.append(np.full(arr.shape[0], clip_flag, bool))
        segments.append({"name": name, "kind": kind,
                         "start_tick": tick, "end_tick": tick + arr.shape[0]})
        tick += arr.shape[0]

    for i, clip in enumerate(clips):
        stem = Path(clip.path).stem
        push(stem, "clip", clip.joint_pos, True)
        last = clip.joint_pos[-1][None, :]
        if i + 1 < len(clips):
            hold_ticks = int(round(hold_between_s * fps))
            push(f"hold_between_{stem}", "hold",
                 np.repeat(last, hold_ticks, axis=0), False)
        else:
            hold_ticks = int(round(hold_after_s * fps))
            push(f"hold_after_{stem}", "hold",
                 np.repeat(last, hold_ticks, axis=0), False)
    return Timeline(
        q_des=np.concatenate(chunks, axis=0),
        is_clip_tick=np.concatenate(flags, axis=0),
        segments=segments,
        fps=fps,
    )


# ---------------------------------------------------------------------------
# support-polygon geometry (pure numpy, unit-tested without mujoco)
# ---------------------------------------------------------------------------

def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar 2D cross product (np.cross on 2-vectors is deprecated in numpy 2)."""
    return float(a[0] * b[1] - a[1] * b[0])


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, CCW, on (N,2) points (N>=1)."""
    pts = np.unique(np.asarray(points, np.float64), axis=0)
    if pts.shape[0] <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def half(seq):
        out: List[np.ndarray] = []
        for p in seq:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    hull = np.asarray(lower[:-1] + upper[:-1])
    return hull if hull.shape[0] >= 3 else pts


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-18:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def com_support_margin(com_xy: np.ndarray, contact_xy: np.ndarray) -> Optional[float]:
    """Signed margin of the CoM ground projection vs. the contact convex hull.

    正=在支撑多边形里面（到最近边的距离）；负=在外面（到多边形的距离取负）。
    接触点不足以张成面积（<3 个或共线）时永远取负——单点/一条线撑不住是常识,
    也是 fail-closed 的方向。无接触点返回 None（该 tick 不计入统计）。
    """
    pts = np.asarray(contact_xy, np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return None
    p = np.asarray(com_xy, np.float64).reshape(2)
    hull = _convex_hull(pts)
    if hull.shape[0] == 1:
        return -float(np.linalg.norm(p - hull[0]))
    if hull.shape[0] == 2:
        return -_point_segment_dist(p, hull[0], hull[1])
    edge_dists, inside = [], True
    for i in range(hull.shape[0]):
        a, b = hull[i], hull[(i + 1) % hull.shape[0]]
        if _cross2(b - a, p - a) < 0.0:   # CCW hull: negative cross => outside this edge
            inside = False
        edge_dists.append(_point_segment_dist(p, a, b))
    d = float(min(edge_dists))
    return d if inside else -d


# ---------------------------------------------------------------------------
# JSON hygiene (NaN anywhere -> fail-closed)
# ---------------------------------------------------------------------------

def sanitize_json(value, nonfinite: List[str], keypath: str = "$"):
    """Convert numpy scalars/arrays; record every non-finite float and None it."""
    if isinstance(value, dict):
        return {str(k): sanitize_json(v, nonfinite, f"{keypath}.{k}")
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v, nonfinite, f"{keypath}[{i}]")
                for i, v in enumerate(value)]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist(), nonfinite, keypath)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not math.isfinite(f):
            nonfinite.append(keypath)
            return None
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# the replay itself
# ---------------------------------------------------------------------------

@dataclass
class Thresholds:
    """Verdict gates（人话见 CLI help；默认值 = A3 部署跌倒判据 + 审计校准）。"""

    fall_root_z_min: float = DF_FALL_ROOT_Z_MIN     # root 低于此高度算摔 [m]
    fall_tilt_rad: float = DF_FALL_TILT_RAD         # 躯干倾角超此算摔 [rad]
    track_p95_max: float = 0.30                     # 跟踪误差 p95 上限 [rad]
    min_foot_contact_frac: float = 0.50             # 每脚最低接触时间比例
    max_foot_airtime_s: float = 1.0                 # 每脚最长连续离地 [s]
    max_cum_slip_m: float = 0.25                    # 每脚支撑期累计滑移上限 [m]
    com_margin_min: float = -0.05                   # CoM 支撑 margin 最小允许值 [m]
    ready_err_max: float = 0.35                     # hold 末端关节误差 max 上限 [rad]
    ready_root_lin_max: float = 0.25                # hold 末端 root 线速度上限 [m/s]
    ready_root_ang_max: float = 0.75                # hold 末端 root 角速度上限 [rad/s]


def build_robot(mjcf_path: str, contract: ReplayContract, sim_dt: float) -> MujocoRobot:
    """Vendor plant via mujoco_eval_onnx.MujocoRobot, training-parity settings.

    passive damping/frictionloss 归零 + 全 implicit + 绑定 effort/velocity 上限
    = MujocoRobot 的 isaac_total_pd_clip_exact 通道（Isaac ImplicitActuator 的
    clip(P-D,±L) 合同），与训练 plant 同一执行法。训练合同中的
    armature 会在内存中覆盖 MJCF 旧值，不修改资产文件。
    """
    n = len(contract.joint_names)
    return MujocoRobot(
        mjcf_path,
        list(contract.joint_names),
        list(contract.body_names),
        sim_dt,
        keep_native_damping=False,
        keep_frictionloss=False,
        pd_mode="implicit",
        kd_for_implicit=contract.kd,
        actuator_types=("implicit",) * n,
        joint_armature=contract.armature,
        joint_velocity_limits=contract.velocity_limits,
        joint_effort_limits=contract.effort_limits,
        allow_velocity_limit_proxy=True,
        allow_effort_limit_proxy=True,
        fail_on_self_contact=False,
    )


def _foot_tick_sample(robot: MujocoRobot):
    """Per-tick foot/floor contact readout: flags, normal force, contact points."""
    model, data = robot.model, robot.data
    n_feet = len(robot.feet_bid)
    in_contact = [False] * n_feet
    normal_force = [0.0] * n_feet
    contact_xy: List[np.ndarray] = []
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        for gf, go in ((g1, g2), (g2, g1)):
            if gf in robot.feet_geoms and not robot.robot_geom_mask[go]:
                side = robot.feet_bid.index(int(model.geom_bodyid[gf]))
                mujoco.mj_contactForce(model, data, i, f6)
                in_contact[side] = True
                normal_force[side] += abs(float(f6[0]))
                contact_xy.append(np.asarray(c.pos[:2], np.float64).copy())
                break
    return in_contact, normal_force, contact_xy


def _body_lin_vel_w(robot: MujocoRobot, bid: int) -> np.ndarray:
    res = np.zeros(6)
    mujoco.mj_objectVelocity(robot.model, robot.data, mujoco.mjtObj.mjOBJ_XBODY,
                             bid, res, 0)
    return res[3:6].copy()


def run_replay(robot: MujocoRobot, timeline: Timeline, *, substeps: int,
               contract: ReplayContract, thresholds: Thresholds) -> dict:
    """One reset, then pure dynamics over the whole timeline. Returns metrics."""
    if mujoco is None:  # pragma: no cover - guarded earlier by build_robot
        raise RuntimeError("mujoco is not installed")
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    fps = timeline.fps
    control_dt = 1.0 / fps
    sim_dt = float(robot.model.opt.timestep)
    n_ticks = timeline.q_des.shape[0]
    n_feet = len(robot.feet_bid)
    th = thresholds

    mj_step_calls = 0
    sim_time_monotonic = True
    state_nan_tick: Optional[int] = None
    fall = {"fell": False, "first_tick": None, "time_s": None, "reason": None}
    reset_count = 1  # the single allowed reset below; never incremented again

    tilt_list: List[float] = []
    root_z_list: List[float] = []
    track_err: List[np.ndarray] = []          # per clip tick: (J,) abs error
    seg_track: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(timeline.segments))}
    contact_ticks = np.zeros(n_feet, int)
    force_sum = np.zeros(n_feet)
    airtime_run = np.zeros(n_feet, int)
    airtime_max = np.zeros(n_feet, int)
    slip_cum = np.zeros(n_feet)
    support_speed_peak = np.zeros(n_feet)
    stance_min_dist = math.inf
    min_separation = math.inf
    legs_crossed = False
    com_margin_min = math.inf
    com_margin_seen = False
    q_soft_limit_samples = 0

    seg_of_tick = np.empty(n_ticks, int)
    for si, seg in enumerate(timeline.segments):
        seg_of_tick[seg["start_tick"]:seg["end_tick"]] = si

    executed_ticks = 0
    prev_time = float(robot.data.time)
    for tick in range(n_ticks):
        q_des = timeline.q_des[tick]
        robot.apply_pd_and_step(q_des, contract.kp, contract.kd, substeps)
        mj_step_calls += substeps
        executed_ticks += 1
        now = float(robot.data.time)
        if not now > prev_time:
            sim_time_monotonic = False
        prev_time = now

        qpos = robot.data.qpos
        qvel = robot.data.qvel
        if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
            state_nan_tick = tick  # fail-closed: stop stepping a poisoned state
            break

        root_z = float(qpos[2])
        tilt = float(math.acos(np.clip(-robot.projected_gravity_body()[2], -1.0, 1.0)))
        root_z_list.append(root_z)
        tilt_list.append(tilt)

        q = robot.q_artic()
        err = np.abs(q_des - q)
        if timeline.is_clip_tick[tick]:
            track_err.append(err)
        seg_track[int(seg_of_tick[tick])].append(err)
        q_soft_limit_samples += int(np.count_nonzero(
            (q < robot.soft_jnt_lo) | (q > robot.soft_jnt_hi)))

        in_contact, normal_force, contact_xy = _foot_tick_sample(robot)
        foot_pos = [robot.data.xpos[bid].copy() for bid in robot.feet_bid]
        for side in range(n_feet):
            if in_contact[side]:
                contact_ticks[side] += 1
                force_sum[side] += normal_force[side]
                airtime_run[side] = 0
                v_xy = float(np.hypot(*_body_lin_vel_w(robot, robot.feet_bid[side])[:2]))
                slip_cum[side] += v_xy * control_dt
                support_speed_peak[side] = max(support_speed_peak[side], v_xy)
            else:
                airtime_run[side] += 1
                airtime_max[side] = max(airtime_max[side], airtime_run[side])
        if n_feet >= 2 and in_contact[0] and in_contact[1]:
            stance_min_dist = min(
                stance_min_dist,
                float(np.hypot(*(foot_pos[0][:2] - foot_pos[1][:2]))),
            )
        if n_feet >= 2:
            # 骨盆坐标系里左脚应在 +Y 侧：左右脚位置差在骨盆左轴上的投影 < 0 = 交叉腿
            left_axis = robot.data.xmat[robot.pelvis_bid].reshape(3, 3)[:, 1][:2]
            axis_norm = float(np.linalg.norm(left_axis))
            if axis_norm > 1e-8:
                sep = float((foot_pos[0][:2] - foot_pos[1][:2]) @ (left_axis / axis_norm))
                min_separation = min(min_separation, sep)
                if sep < 0.0:
                    legs_crossed = True
        if contact_xy:
            com_xy = robot.data.subtree_com[robot.pelvis_bid][:2]
            margin = com_support_margin(com_xy, np.asarray(contact_xy))
            if margin is not None:
                com_margin_seen = True
                com_margin_min = min(com_margin_min, margin)

        if root_z < th.fall_root_z_min or tilt > th.fall_tilt_rad:
            fall = {
                "fell": True, "first_tick": tick, "time_s": now,
                "reason": (f"root_z {root_z:.3f} < {th.fall_root_z_min:g}"
                           if root_z < th.fall_root_z_min
                           else f"tilt {tilt:.3f} > {th.fall_tilt_rad:g} rad"),
            }
            break  # 摔了以后的数据没有意义，及时止损（verdict 已注定 FAIL）

    # trailing airtime runs count too (a foot that never came back down)
    for side in range(n_feet):
        airtime_max[side] = max(airtime_max[side], airtime_run[side])

    expected_calls = executed_ticks * substeps
    time_accounting_ok = (
        sim_time_monotonic
        and mj_step_calls == expected_calls
        and abs(float(robot.data.time) - mj_step_calls * sim_dt) < 1e-6 * max(1, mj_step_calls)
    )

    def pct(values: List[float], q: float) -> Optional[float]:
        return float(np.percentile(values, q)) if values else None

    track_flat = np.concatenate(track_err) if track_err else np.empty(0)
    q_end = robot.q_artic()
    q_des_end = timeline.q_des[min(executed_ticks, n_ticks) - 1]
    ready_err = np.abs(q_end - q_des_end)
    root_lin_speed = float(np.linalg.norm(robot.data.qvel[0:3]))
    root_ang_speed = float(np.linalg.norm(robot.data.qvel[3:6]))

    feet = {}
    denom = max(executed_ticks, 1)
    for side, name in enumerate(FEET_BODIES[:n_feet]):
        label = "left" if name.startswith("left") else "right"
        feet[label] = {
            "body": name,
            "contact_frac": float(contact_ticks[side] / denom),
            "normal_force_mean_n": (float(force_sum[side] / contact_ticks[side])
                                    if contact_ticks[side] else 0.0),
            "longest_airtime_s": float(airtime_max[side] * control_dt),
            "support_speed_peak_mps": float(support_speed_peak[side]),
            "slip_cum_m": float(slip_cum[side]),
        }

    segments_out = []
    for si, seg in enumerate(timeline.segments):
        errs = np.concatenate(seg_track[si]) if seg_track[si] else np.empty(0)
        segments_out.append({
            **seg,
            "executed": bool(seg["start_tick"] < executed_ticks),
            "track_err_rad": {
                "mean": float(errs.mean()) if errs.size else None,
                "p95": float(np.percentile(errs, 95)) if errs.size else None,
                "max": float(errs.max()) if errs.size else None,
            },
        })

    metrics = {
        "contract": contract.name,
        "reset_count": reset_count,
        "fps": fps,
        "substeps": substeps,
        "sim_dt_s": sim_dt,
        "planned_ticks": int(n_ticks),
        "executed_ticks": int(executed_ticks),
        "mj_step_calls": int(mj_step_calls),
        "expected_mj_step_calls": int(expected_calls),
        "sim_time_s": float(robot.data.time),
        "sim_time_monotonic": bool(sim_time_monotonic),
        "state_nan_tick": state_nan_tick,
        "fall": fall,
        "root_z_min_m": (float(min(root_z_list)) if root_z_list else None),
        "tilt_rad": {
            "p50": pct(tilt_list, 50), "p95": pct(tilt_list, 95),
            "p99": pct(tilt_list, 99),
            "max": (float(max(tilt_list)) if tilt_list else None),
        },
        "feet": feet,
        "stance_min_foot_dist_m": (float(stance_min_dist)
                                   if math.isfinite(stance_min_dist) else None),
        "legs_crossed": {
            "crossed": bool(legs_crossed),
            "min_separation_m": (float(min_separation)
                                 if math.isfinite(min_separation) else None),
        },
        "com_support_margin_min_m": (float(com_margin_min) if com_margin_seen else None),
        "saturation": {
            "q_soft_limit_samples": int(q_soft_limit_samples),
            "qdot_limit_hits": int(robot.velocity_limit_hit_count),
            "qdot_limit_peak_ratio": float(robot.velocity_limit_peak_ratio),
            "torque_limit_hits": int(robot.effort_limit_hit_count),
            "torque_limit_peak_ratio": float(robot.effort_limit_peak_ratio),
        },
        "tracking_err_rad": {
            "mean": float(track_flat.mean()) if track_flat.size else None,
            "p95": float(np.percentile(track_flat, 95)) if track_flat.size else None,
            "max": float(track_flat.max()) if track_flat.size else None,
        },
        "recover": {
            "ready_err_max_rad": float(ready_err.max()),
            "ready_err_mean_rad": float(ready_err.mean()),
            "root_lin_speed_mps": root_lin_speed,
            "root_ang_speed_radps": root_ang_speed,
        },
        "segments": segments_out,
    }
    metrics["verdict"] = build_verdict(metrics, th)
    return metrics


def build_verdict(metrics: dict, th: Thresholds) -> dict:
    """逐门 PASS/FAIL；任何一门 fail 或任何 NaN => 整体 FAIL（fail-closed）。"""
    gates: List[dict] = []

    def gate(name: str, ok: bool, reason: str):
        gates.append({"name": name, "ok": bool(ok), "reason": reason})

    nonfinite: List[str] = []
    sanitize_json({k: v for k, v in metrics.items() if k != "verdict"}, nonfinite)
    nan_free = metrics.get("state_nan_tick") is None and not nonfinite
    gate("finite", nan_free,
         "all state and metrics finite" if nan_free else
         f"NaN/Inf detected: state_nan_tick={metrics.get('state_nan_tick')}, "
         f"nonfinite_metrics={nonfinite[:5]}")

    complete = metrics["executed_ticks"] == metrics["planned_ticks"]
    gate("timeline_complete", complete,
         f"executed {metrics['executed_ticks']}/{metrics['planned_ticks']} ticks")

    steps_ok = (metrics["sim_time_monotonic"]
                and metrics["mj_step_calls"] == metrics["expected_mj_step_calls"]
                and metrics["mj_step_calls"] > 0)
    gate("mj_step_accounting", steps_ok,
         f"mj_step_calls={metrics['mj_step_calls']} expected="
         f"{metrics['expected_mj_step_calls']} monotonic={metrics['sim_time_monotonic']}")

    fell = metrics["fall"]["fell"]
    gate("no_fall", not fell,
         "stayed up" if not fell else f"fell: {metrics['fall']['reason']}")

    feet = metrics["feet"]
    ok_feet, feet_reason = True, []
    for label, f in feet.items():
        if f["contact_frac"] < th.min_foot_contact_frac:
            ok_feet = False
            feet_reason.append(f"{label} contact_frac {f['contact_frac']:.2f}")
        if f["longest_airtime_s"] > th.max_foot_airtime_s:
            ok_feet = False
            feet_reason.append(f"{label} airtime {f['longest_airtime_s']:.2f}s")
    gate("foot_contact", ok_feet,
         "both feet keep ground contact" if ok_feet else "; ".join(feet_reason))

    ok_slip, slip_reason = True, []
    for label, f in feet.items():
        if f["slip_cum_m"] > th.max_cum_slip_m:
            ok_slip = False
            slip_reason.append(f"{label} slip {f['slip_cum_m']:.3f} m")
    gate("foot_slip", ok_slip,
         "support-foot slip within budget" if ok_slip else "; ".join(slip_reason))

    crossed = metrics["legs_crossed"]["crossed"]
    gate("no_crossed_legs", not crossed,
         "feet stay on their own side" if not crossed else
         f"legs crossed (min separation {metrics['legs_crossed']['min_separation_m']})")

    margin = metrics["com_support_margin_min_m"]
    margin_ok = margin is not None and margin >= th.com_margin_min
    gate("com_support_margin", margin_ok,
         f"min margin {margin if margin is None else round(margin, 4)} m "
         f"(threshold {th.com_margin_min:g})")

    p95 = metrics["tracking_err_rad"]["p95"]
    track_ok = p95 is not None and p95 <= th.track_p95_max
    gate("tracking", track_ok,
         f"q error p95 {p95 if p95 is None else round(p95, 4)} rad "
         f"(threshold {th.track_p95_max:g})")

    rec = metrics["recover"]
    rec_ok = (rec["ready_err_max_rad"] <= th.ready_err_max
              and rec["root_lin_speed_mps"] <= th.ready_root_lin_max
              and rec["root_ang_speed_radps"] <= th.ready_root_ang_max)
    gate("recover_ready", rec_ok,
         f"ready err max {rec['ready_err_max_rad']:.3f} rad, root lin "
         f"{rec['root_lin_speed_mps']:.3f} m/s, ang {rec['root_ang_speed_radps']:.3f} rad/s")

    overall = PASS if all(g["ok"] for g in gates) else FAIL
    return {"overall": overall, "gates": gates}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--motion", help="单条 motion npz（31 DoF 关节参考轨迹，50 Hz）")
    src.add_argument("--pair", nargs=2, metavar=("FH_NPZ", "BH_NPZ"),
                     help="正反手两条 npz 顺序执行（中间站立 hold，不 reset）")
    p.add_argument("--mjcf", default=resolve_default_mjcf(),
                   help="vendor MJCF（默认 repo 根下 a3_pingpong.xml，同 mujoco_eval_onnx）")
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--hold-after-s", type=float, default=DEFAULT_HOLD_AFTER_S,
                   help="clip 结束后继续保持最后一帧参考的时长 [s]")
    p.add_argument("--hold-between-s", type=float, default=DEFAULT_HOLD_BETWEEN_S,
                   help="--pair 两条 clip 之间的站立 hold 时长 [s]")
    p.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS,
                   help="每个 50 Hz tick 的物理 substep 数（sim_dt = 1/fps/substeps）")
    p.add_argument("--contract-json", default=None,
                   help="诊断用 plant 合同 JSON（默认内置 A3 厂商 PD 表；测试的最小模型用这个）")
    d = Thresholds()
    p.add_argument("--fall-root-z-min", type=float, default=d.fall_root_z_min,
                   help="root 低于此高度算摔 [m]")
    p.add_argument("--fall-tilt-rad", type=float, default=d.fall_tilt_rad,
                   help="躯干倾角超此算摔 [rad]")
    p.add_argument("--track-p95-max", type=float, default=d.track_p95_max,
                   help="关节跟踪误差 p95 门限 [rad]")
    p.add_argument("--min-foot-contact-frac", type=float, default=d.min_foot_contact_frac,
                   help="每脚最低接触时间比例")
    p.add_argument("--max-foot-airtime-s", type=float, default=d.max_foot_airtime_s,
                   help="每脚最长连续离地 [s]")
    p.add_argument("--max-cum-slip-m", type=float, default=d.max_cum_slip_m,
                   help="每脚支撑期累计滑移上限 [m]")
    p.add_argument("--com-margin-min", type=float, default=d.com_margin_min,
                   help="CoM 支撑多边形 margin 最小允许值 [m]（负=允许短暂出圈）")
    p.add_argument("--ready-err-max", type=float, default=d.ready_err_max,
                   help="hold 末端关节误差 max 门限 [rad]")
    p.add_argument("--ready-root-lin-max", type=float, default=d.ready_root_lin_max,
                   help="hold 末端 root 线速度门限 [m/s]")
    p.add_argument("--ready-root-ang-max", type=float, default=d.ready_root_ang_max,
                   help="hold 末端 root 角速度门限 [rad/s]")
    args = p.parse_args(argv)

    if mujoco is None:
        print("[FAIL] mujoco is not installed in this environment", file=sys.stderr)
        return 2

    try:
        contract = (contract_from_json(args.contract_json)
                    if args.contract_json else a3_contract())
        clip_paths = [args.motion] if args.motion else list(args.pair)
        clips = [load_motion(path, len(contract.joint_names)) for path in clip_paths]
        timeline = build_timeline(
            clips, hold_after_s=args.hold_after_s, hold_between_s=args.hold_between_s)
        if timeline.fps != DEFAULT_FPS_EXPECTED:
            print(f"[WARN] clip fps={timeline.fps}, production contract is "
                  f"{DEFAULT_FPS_EXPECTED} Hz — timing gates are fps-relative")
        sim_dt = 1.0 / (timeline.fps * args.substeps)
        robot = build_robot(args.mjcf, contract, sim_dt)
        # 单次 reset（合同第 1 条）：clip 首帧位姿 + 零速度，此后只有 mj_step。
        robot.reset_to_stand(clips[0].root_pos0, clips[0].root_quat0,
                             clips[0].joint_pos[0])
        thresholds = Thresholds(
            fall_root_z_min=args.fall_root_z_min,
            fall_tilt_rad=args.fall_tilt_rad,
            track_p95_max=args.track_p95_max,
            min_foot_contact_frac=args.min_foot_contact_frac,
            max_foot_airtime_s=args.max_foot_airtime_s,
            max_cum_slip_m=args.max_cum_slip_m,
            com_margin_min=args.com_margin_min,
            ready_err_max=args.ready_err_max,
            ready_root_lin_max=args.ready_root_lin_max,
            ready_root_ang_max=args.ready_root_ang_max,
        )
        metrics = run_replay(robot, timeline, substeps=args.substeps,
                             contract=contract, thresholds=thresholds)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    metrics["motion_files"] = [str(Path(p_).resolve()) for p_ in clip_paths]
    metrics["mjcf"] = {"path": os.path.abspath(args.mjcf),
                       "sha256": sha256_file(args.mjcf)}
    metrics["disclaimer"] = (
        "motion-only PASS only proves the reference trajectory is dynamically "
        "executable under the vendor PD plant; it does NOT replace policy replay "
        "and does NOT replace vendor Gate3."
    )

    nonfinite: List[str] = []
    payload = sanitize_json(metrics, nonfinite)
    if nonfinite:
        # sanitize_json None-ed them; the finite gate has already failed, but the
        # verdict object may predate serialization — enforce FAIL here too.
        payload["verdict"]["overall"] = FAIL
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)

    verdict = payload["verdict"]["overall"]
    print(f"[motion-dynamic-replay] verdict={verdict} "
          f"mj_step_calls={payload['mj_step_calls']} "
          f"sim_time_s={payload['sim_time_s']:.3f} -> {out}")
    for g in payload["verdict"]["gates"]:
        mark = "ok " if g["ok"] else "FAIL"
        print(f"  [{mark}] {g['name']}: {g['reason']}")
    return 0 if verdict == PASS else 2


if __name__ == "__main__":
    sys.exit(main())
