#!/usr/bin/env python3
"""Axis-D footwork amplitude (β, 脚步幅度) — composition-spec generator SKELETON.

人话:这个工具把一条"共享横移脚步模块"的参考轨迹拆成结构化参数(左右脚分离向量、
双脚中心、root 有符号横向位移、左右脚 contact phase、落脚位置),再按脚步幅度 β 缩放
"位移类"量(root 位移与落脚点横向偏移),输出一份**组合规格 JSON**——不是最终 npz,
更不是可训练资产。支撑脚在自己的支撑期位置钉死;β 不碰 z 轴、不统一缩放全身、
更不是"把 hip-roll 放大一点"。

WHY A SPEC, NOT AN NPZ (frozen semantics, 2026-07-20)
    动作库里唯一的横移素材是 motion/{left,right}_dang{1,2}(catalog 角色
    shared_lateral_footwork_module),它们的 M0 stance gate 是 0/4 reject,且没有任何
    过动力学门的移动参考。所以本轮不可能产出可训练资产;本工具交付的是:
      1. β 的参数化合同(见 docs/interfaces/footwork_scale_contract.md);
      2. 一个可运行骨架:对合法输入生成组合规格 JSON;
      3. 对未来输入 fail-closed 的完整校验(stance 门、交叉检测、支撑滑移、
         contact phase 标注缺失即拒收)。
    任何由本规格派生的动作 <strike, footwork, signed distance, phase alignment,
    retiming> 都必须重新通过完整 Gate 链(见 stroke_footwork_composition.md)。

β BUCKETS (写死在每份输出 manifest 里;各泛化轴同一比例约定)
    train         : 0.80 / 1.00 / 1.20
    interpolation : 0.90 / 1.10
    OOD           : 0.65 / 1.35
    β 不在这张冻结网格上 → 直接拒收(分桶是预注册的,不接受任意 β)。

HARD REFUSALS (fail-closed, exit code 2)
    * β = 0:"原地"必须直接引用 catalog 里真实的 stationary_strike 资产,
      不允许把移动动作压扁冒充原地击球。
    * β < 0:方向翻转是镜像,镜像有自己的门链(v12 C3:关节映射/非对称/自碰/
      厂商动力学,且不得镜像右手挥拍),不是缩放。
    * 缺 contact phase 标注(npz 内嵌 left/right_foot_contact 或 --contact-json
      二选一,都没有就拒收)。
    * stance 门失败(阈值抄自 exact-GMR 卷宗的冻结判定,M0 四条 0/4 的已知数值
      在单测里作为负例):前后分量 |Δ|≤0.03 m;横向 signed 分量 |Δ|≤0.03 m;
      初始横向绝对分离 ≥0.05 m 且左右符号不翻;末态横向分离最多变窄 0.005 m
      (独立硬门——变窄 2.4 cm 即使落在 3 cm 带内也必须失败,right2 就是这么死的)。
    * 双脚交叉/最小站宽违例:源轨迹逐帧、以及 β 缩放后的落脚排程(支撑期锚点 +
      摆动期线性插值近似)都要过 sign-no-flip + 最小横向净距检查。
    * 支撑期滑移:标注为支撑的区间内脚水平漂移超预算 → 标注与轨迹矛盾,拒收。
    * 非有限值、body 顺序无法解析、朝向漂移过大(横移模块不该转身)、
      catalog 里 input_gate_status 为 rejected_* 的源资产(修好的素材必须换新
      asset_id 重登记再走门,不许借旧名)。

WHAT β ACTS ON (movement frame = 初始朝向对齐的水平坐标系)
    * root 有符号横向位移: d_target = β · d_source(achieved 值写进规格);
    * 落脚点横向偏移:每只脚第 k 个支撑锚点 y'_k = y_0 + β·(y_k − y_0),
      y_0 是该脚第一个支撑锚点;前后分量与高度不变;
    * 支撑脚支撑期:位置钉死(锚点即约束,不参与任何插值缩放);
    * 收步目标:末态恢复初始左右脚分离向量(recovery-ready 误差预算写进规格,
      横向变窄硬上限 0.005 m 沿用冻结判定)。

USAGE
    python scripts/scale_footwork.py \
        --footwork-npz M0_left1_fixed.npz --beta 1.20 \
        --direction left \
        --strike-asset-id hope_forehand_v4rg_cal --strike-phase 0.471 \
        [--strike-sha256 HEX] \
        [--catalog configs/motion_role_catalog.json --footwork-asset-id ID] \
        [--body-order LIST|FILE] [--contact-json contacts.json] \
        --output spec.json

    Exit code: 0 = spec written; 2 = refused (fail-closed, nothing written).

DEPENDENCIES: numpy + standard library only. No mujoco, no torch, no isaac.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

TOOL_NAME = "scale_footwork"
SPEC_SCHEMA_VERSION = 1
SPEC_KIND = "footwork_scale_composition_spec"

# --- frozen β buckets (同一比例约定横贯全部泛化轴;写死进每份 manifest) ------- #
BETA_BUCKETS: Dict[str, Tuple[float, ...]] = {
    "train": (0.80, 1.00, 1.20),
    "interpolation": (0.90, 1.10),
    "ood": (0.65, 1.35),
}
BETA_GRID: Tuple[float, ...] = tuple(sorted(b for bs in BETA_BUCKETS.values() for b in bs))
_BETA_TOL = 1e-9

# --- frozen stance-gate thresholds (exact-GMR 卷宗 2026-07-13 冻结判定) ------- #
STANCE_FORE_AFT_BAND_M = 0.03          # 前后分量绝对变化上限
STANCE_LATERAL_BAND_M = 0.03           # 横向 signed 分量绝对变化上限
STANCE_MIN_INITIAL_LATERAL_SEP_M = 0.05  # 初始横向绝对分离下限(符号还不许翻)
STANCE_MAX_TERMINAL_NARROWING_M = 0.005  # 末态最多允许的数值性变窄(独立硬门)

# --- tool defaults (非冻结;使用它们的实验必须预注册具体数值) ----------------- #
DEFAULT_MIN_LATERAL_CLEARANCE_M = 0.03   # 全程最小站宽/禁交叉净距
DEFAULT_SUPPORT_SLIP_BUDGET_M = 0.02     # 支撑期水平漂移预算(标注-轨迹一致性)
DEFAULT_SUPPORT_HEIGHT_EPS_M = 0.03      # 支撑帧高度带(z <= min z + eps)
DEFAULT_MIN_ROOT_DISPLACEMENT_M = 0.02   # "移动模块必须真的在移动"的工具级下限
DEFAULT_MAX_YAW_DRIFT_RAD = 0.26         # ~15°;横移模块不该转身
MIN_STANCE_WINDOW_FRAMES = 3             # 初/末双支撑窗至少帧数(稳健中位数用)

LEFT_FOOT_BODY = "left_ankle_roll_Link"
RIGHT_FOOT_BODY = "right_ankle_roll_Link"
N_JOINTS = 31

REQUIRES_FULL_GATE_CHAIN = [
    "runtime-order schema",
    "L0 finite/limit/endpoint audit (audit_motion_npz)",
    "vendor MJCF L1 self-collision + racket-handle clearance (audit_self_collision)",
    "table/net full-trajectory swept clearance >= 5 mm",
    "vendor MuJoCo dynamics / foot-contact replay",
]
FORBIDDEN_IMPLEMENTATIONS = [
    "hip_roll_only_amplification",
    "uniform_whole_body_scaling",
    "z_axis_scaling",
]


class FootworkScaleError(RuntimeError):
    """Fail-closed refusal. `code` is a stable machine tag; message is 人话."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


# ------------------------------------------------------------------ utilities #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise FootworkScaleError("nonfinite_input", f"{name} contains non-finite values")


def bucket_for_beta(beta: float) -> str:
    for bucket, values in BETA_BUCKETS.items():
        if any(abs(beta - v) <= _BETA_TOL for v in values):
            return bucket
    raise FootworkScaleError(
        "beta_off_grid",
        f"beta={beta!r} is not on the frozen preregistered grid {BETA_GRID}; "
        "arbitrary beta values are refused (buckets are preregistered)",
    )


def validate_beta(beta: float) -> str:
    """Return the bucket name for beta, or refuse (β=0 / β<0 / off-grid / nonfinite)."""
    if not (isinstance(beta, float) and math.isfinite(beta)):
        raise FootworkScaleError("nonfinite_input", f"beta={beta!r} is not a finite float")
    if abs(beta) <= _BETA_TOL:
        raise FootworkScaleError(
            "beta_zero_stationary",
            "beta=0 (d=0, stationary) is refused: the stationary case must directly "
            "reference a real stationary_strike asset from configs/motion_role_catalog.json; "
            "flattening a moving footwork clip to fake a stationary strike is not allowed",
        )
    if beta < 0.0:
        raise FootworkScaleError(
            "beta_negative_mirror",
            "beta<0 flips the movement direction, which is a MIRROR, not a scale; "
            "mirrored footwork has its own gate chain (v12 C3: joint mapping, asymmetry, "
            "self-collision, vendor dynamics; the right-hand racket swing is never mirrored)",
        )
    return bucket_for_beta(beta)


def yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Z-yaw from (..., 4) wxyz quaternions."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ------------------------------------------------------------------- loading #
def resolve_body_names(npz: Dict[str, np.ndarray], body_order: str | None,
                       n_bodies: int) -> List[str]:
    """Embedded `body_names` key, or --body-order LIST|FILE. Unresolved = refuse."""
    explicit: List[str] | None = None
    if body_order:
        is_file = False
        if "," not in body_order:
            try:
                is_file = Path(body_order).is_file()
            except OSError:
                is_file = False
        if is_file:
            explicit = [ln.strip() for ln in Path(body_order).read_text().splitlines()
                        if ln.strip() and not ln.startswith("#")]
        else:
            explicit = [n.strip() for n in body_order.split(",") if n.strip()]
    embedded: List[str] | None = None
    if "body_names" in npz:
        embedded = [str(n) for n in np.asarray(npz["body_names"]).reshape(-1)]
    if explicit is not None and embedded is not None and explicit != embedded:
        raise FootworkScaleError(
            "body_order_conflict", "--body-order disagrees with the npz-embedded body_names")
    names = explicit if explicit is not None else embedded
    if names is None:
        raise FootworkScaleError(
            "body_order_unresolved",
            "cannot resolve the body order: npz has no `body_names` key and no "
            "--body-order was given (fail-loud, never a silent guess)")
    if len(names) != n_bodies:
        raise FootworkScaleError(
            "body_order_unresolved",
            f"body order has {len(names)} names but body_pos_w has {n_bodies} bodies")
    return names


def load_footwork_clip(npz_path: Path, body_order: str | None) -> Dict[str, object]:
    if not npz_path.is_file():
        raise FootworkScaleError("input_missing", f"footwork npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    for key in ("fps", "joint_pos", "body_pos_w", "body_quat_w"):
        if key not in data:
            raise FootworkScaleError("npz_schema", f"footwork npz is missing key {key!r}")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if not (math.isfinite(fps) and fps > 0):
        raise FootworkScaleError("npz_schema", f"fps={fps!r} is not a positive finite scalar")
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != N_JOINTS:
        raise FootworkScaleError(
            "npz_schema", f"joint_pos shape {joint_pos.shape} != (T, {N_JOINTS})")
    if body_pos.ndim != 3 or body_pos.shape[2] != 3:
        raise FootworkScaleError("npz_schema", f"body_pos_w shape {body_pos.shape} != (T, nb, 3)")
    if body_quat.shape[:2] != body_pos.shape[:2] or body_quat.shape[2] != 4:
        raise FootworkScaleError("npz_schema", f"body_quat_w shape {body_quat.shape} != (T, nb, 4)")
    T = body_pos.shape[0]
    if joint_pos.shape[0] != T:
        raise FootworkScaleError("npz_schema", "joint_pos and body_pos_w frame counts disagree")
    if T < 2 * MIN_STANCE_WINDOW_FRAMES:
        raise FootworkScaleError("npz_schema", f"clip too short: {T} frames")
    _require_finite("joint_pos", joint_pos)
    _require_finite("body_pos_w", body_pos)
    _require_finite("body_quat_w", body_quat)

    names = resolve_body_names(data, body_order, body_pos.shape[1])
    try:
        li, ri = names.index(LEFT_FOOT_BODY), names.index(RIGHT_FOOT_BODY)
    except ValueError:
        raise FootworkScaleError(
            "body_order_unresolved",
            f"body order must contain {LEFT_FOOT_BODY!r} and {RIGHT_FOOT_BODY!r}; got {names}")
    return {
        "fps": fps, "T": T, "npz": data, "body_names": names,
        "root_pos": body_pos[:, 0, :], "root_quat": body_quat[:, 0, :],
        "left_pos": body_pos[:, li, :], "right_pos": body_pos[:, ri, :],
        "root_body": names[0],
    }


# ----------------------------------------------------------- contact phases #
def _validate_intervals(foot: str, raw: object, T: int) -> List[Tuple[int, int]]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise FootworkScaleError(
            "contact_phase_missing", f"{foot} foot has no support intervals")
    out: List[Tuple[int, int]] = []
    for item in raw:
        if (not isinstance(item, (list, tuple)) or len(item) != 2
                or not all(isinstance(v, (int, np.integer)) for v in item)):
            raise FootworkScaleError(
                "contact_phase_invalid", f"{foot} interval {item!r} is not [start, end] ints")
        s, e = int(item[0]), int(item[1])
        if not (0 <= s <= e < T):
            raise FootworkScaleError(
                "contact_phase_invalid", f"{foot} interval [{s},{e}] out of range [0,{T-1}]")
        out.append((s, e))
    out.sort()
    for (s0, e0), (s1, _e1) in zip(out, out[1:]):
        if s1 <= e0:
            raise FootworkScaleError(
                "contact_phase_invalid", f"{foot} support intervals overlap/touch: {out}")
    return out


def load_contact_phases(clip: Dict[str, object], contact_json: Path | None
                        ) -> Dict[str, List[Tuple[int, int]]]:
    """npz-embedded left/right_foot_contact 0/1 arrays, or --contact-json.
    Both absent -> refuse (contact phase 标注是硬输入,不做无标注自动猜测)."""
    T = int(clip["T"])
    npz: Dict[str, np.ndarray] = clip["npz"]  # type: ignore[assignment]
    embedded = "left_foot_contact" in npz and "right_foot_contact" in npz
    if contact_json is not None and embedded:
        raise FootworkScaleError(
            "contact_phase_conflict",
            "both npz-embedded contact arrays and --contact-json were given; "
            "pick one truth source")
    if contact_json is not None:
        if not contact_json.is_file():
            raise FootworkScaleError(
                "contact_phase_missing", f"--contact-json not found: {contact_json}")
        payload = json.loads(contact_json.read_text())
        if not isinstance(payload, dict) or "left" not in payload or "right" not in payload:
            raise FootworkScaleError(
                "contact_phase_invalid", "--contact-json must be {\"left\": [[s,e],...], \"right\": [...]}")
        return {"left": _validate_intervals("left", payload["left"], T),
                "right": _validate_intervals("right", payload["right"], T)}
    if embedded:
        result = {}
        for foot in ("left", "right"):
            arr = np.asarray(npz[f"{foot}_foot_contact"]).reshape(-1)
            if arr.shape[0] != T or not np.all(np.isin(arr, (0, 1))):
                raise FootworkScaleError(
                    "contact_phase_invalid", f"{foot}_foot_contact must be a (T,) 0/1 array")
            mask = arr.astype(bool)
            intervals, s = [], None
            for t in range(T):
                if mask[t] and s is None:
                    s = t
                elif not mask[t] and s is not None:
                    intervals.append((s, t - 1))
                    s = None
            if s is not None:
                intervals.append((s, T - 1))
            result[foot] = _validate_intervals(foot, intervals, T)
        return result
    raise FootworkScaleError(
        "contact_phase_missing",
        "no contact-phase annotation: the npz has no left/right_foot_contact arrays and "
        "no --contact-json was given. Contact phases are a REQUIRED input — this tool "
        "refuses to guess them (fail-closed)")


def support_mask(T: int, intervals: Sequence[Tuple[int, int]]) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    for s, e in intervals:
        m[s:e + 1] = True
    return m


# -------------------------------------------------------- parameterization #
def _both_support_window(left_m: np.ndarray, right_m: np.ndarray, *, tail: bool
                         ) -> Tuple[int, int]:
    both = left_m & right_m
    T = both.shape[0]
    idx = T - 1 if tail else 0
    if not both[idx]:
        raise FootworkScaleError(
            "contact_endpoints",
            "both feet must be in support at the clip's first and last frame "
            "(prepare / recover stances)")
    j = idx
    if tail:
        while j - 1 >= 0 and both[j - 1]:
            j -= 1
        w = (j, T - 1)
    else:
        while j + 1 < T and both[j + 1]:
            j += 1
        w = (0, j)
    if w[1] - w[0] + 1 < MIN_STANCE_WINDOW_FRAMES:
        raise FootworkScaleError(
            "contact_endpoints",
            f"double-support window {w} shorter than {MIN_STANCE_WINDOW_FRAMES} frames; "
            "a robust median needs a stable window, not one noisy video frame")
    return w


def _median_xy(pos: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    s, e = window
    return np.median(pos[s:e + 1, :], axis=0)


def extract_parameterization(clip: Dict[str, object],
                             phases: Dict[str, List[Tuple[int, int]]],
                             *,
                             support_height_eps: float,
                             support_slip_budget: float,
                             max_yaw_drift: float) -> Dict[str, object]:
    """Movement-frame parameterization: separation vector, feet center, root signed
    lateral displacement, per-interval support anchors, footfall positions."""
    T = int(clip["T"])
    left_pos: np.ndarray = clip["left_pos"]    # type: ignore[assignment]
    right_pos: np.ndarray = clip["right_pos"]  # type: ignore[assignment]
    root_pos: np.ndarray = clip["root_pos"]    # type: ignore[assignment]
    root_quat: np.ndarray = clip["root_quat"]  # type: ignore[assignment]
    left_m = support_mask(T, phases["left"])
    right_m = support_mask(T, phases["right"])

    init_w = _both_support_window(left_m, right_m, tail=False)
    term_w = _both_support_window(left_m, right_m, tail=True)

    # movement frame: world rotated by the median initial root yaw. 横移模块不该转身,
    # 朝向漂移过大直接拒收,所以一个全局 movement frame 是合法近似。
    yaw = yaw_from_quat_wxyz(root_quat)
    yaw0 = float(np.median(yaw[init_w[0]:init_w[1] + 1]))
    yaw_drift = float(np.max(np.abs(_wrap_angle(yaw - yaw0))))
    if yaw_drift > max_yaw_drift:
        raise FootworkScaleError(
            "yaw_drift",
            f"root yaw drifts {yaw_drift:.3f} rad > {max_yaw_drift:.3f} rad; a lateral "
            "footwork module must not turn — this clip is not a lateral shuffle")
    c, s = math.cos(yaw0), math.sin(yaw0)

    def to_movement(xy: np.ndarray) -> np.ndarray:
        """world (x, y) -> movement frame (fore_aft, lateral); lateral + = robot-left."""
        out = np.empty_like(xy)
        out[..., 0] = c * xy[..., 0] + s * xy[..., 1]
        out[..., 1] = -s * xy[..., 0] + c * xy[..., 1]
        return out

    left_mf = to_movement(left_pos[:, :2])
    right_mf = to_movement(right_pos[:, :2])
    root_mf = to_movement(root_pos[:, :2])

    # separation vector (left - right), robust medians over the two stance windows
    sep = left_mf - right_mf
    init_sep = _median_xy(sep, init_w)
    term_sep = _median_xy(sep, term_w)
    feet_center_init = _median_xy((left_mf + right_mf) * 0.5, init_w)
    feet_center_term = _median_xy((left_mf + right_mf) * 0.5, term_w)
    root_disp = float(_median_xy(root_mf, term_w)[1] - _median_xy(root_mf, init_w)[1])

    # per-interval support anchors + annotation-vs-trajectory consistency
    anchors: Dict[str, List[Dict[str, float]]] = {"left": [], "right": []}
    for foot, pos_mf, pos_w in (("left", left_mf, left_pos), ("right", right_mf, right_pos)):
        z = pos_w[:, 2]
        z_floor = float(np.min(z))
        for (a, b) in phases[foot]:
            seg_z = z[a:b + 1]
            if float(np.max(seg_z)) > z_floor + support_height_eps:
                raise FootworkScaleError(
                    "contact_height_contradiction",
                    f"{foot} foot annotated in support on [{a},{b}] but rises "
                    f"{float(np.max(seg_z)) - z_floor:.3f} m above its floor "
                    f"(> {support_height_eps:.3f} m): annotation contradicts trajectory")
            anchor = np.median(pos_mf[a:b + 1, :], axis=0)
            drift = float(np.max(np.linalg.norm(pos_mf[a:b + 1, :] - anchor, axis=1)))
            if drift > support_slip_budget:
                raise FootworkScaleError(
                    "support_slip",
                    f"{foot} foot slides {drift:.3f} m inside annotated support [{a},{b}] "
                    f"(budget {support_slip_budget:.3f} m): a planted foot must stay planted")
            anchors[foot].append({
                "start_frame": a, "end_frame": b,
                "fore_aft_m": float(anchor[0]), "lateral_m": float(anchor[1]),
                "height_m": float(np.median(seg_z)),
                "support_drift_m": drift,
            })

    # footfalls: every support interval after the first is a landing
    footfalls = {
        foot: [{"touchdown_frame": a["start_frame"],
                "fore_aft_m": a["fore_aft_m"], "lateral_m": a["lateral_m"]}
               for a in anchors[foot][1:]]
        for foot in ("left", "right")
    }
    return {
        "T": T,
        "movement_frame_yaw_rad": yaw0,
        "yaw_drift_rad": yaw_drift,
        "initial_window": init_w, "terminal_window": term_w,
        "initial_separation_m": {"fore_aft": float(init_sep[0]), "lateral": float(init_sep[1])},
        "terminal_separation_m": {"fore_aft": float(term_sep[0]), "lateral": float(term_sep[1])},
        "feet_center_initial_m": {"fore_aft": float(feet_center_init[0]),
                                  "lateral": float(feet_center_init[1])},
        "feet_center_terminal_m": {"fore_aft": float(feet_center_term[0]),
                                   "lateral": float(feet_center_term[1])},
        "root_signed_lateral_displacement_m": root_disp,
        "per_frame_separation_lateral_m": sep[:, 1],
        "support_anchors": anchors,
        "footfalls": footfalls,
    }


# ------------------------------------------------------------------ checks #
def stance_gate(param: Dict[str, object]) -> Dict[str, object]:
    """Frozen stance gate (thresholds preregistered in the exact-GMR dossier).
    Any failed check -> refuse. right2's 0.0243 m narrowing must die on the
    independent 0.005 m hard gate even though it sits inside the 0.03 m band."""
    init = param["initial_separation_m"]  # type: ignore[index]
    term = param["terminal_separation_m"]  # type: ignore[index]
    init_lat, term_lat = float(init["lateral"]), float(term["lateral"])
    fore_change = abs(float(term["fore_aft"]) - float(init["fore_aft"]))
    lat_change = abs(term_lat - init_lat)
    narrowing = max(0.0, abs(init_lat) - abs(term_lat))
    checks = {
        "fore_aft_band": {
            "value_m": fore_change, "limit_m": STANCE_FORE_AFT_BAND_M,
            "passed": fore_change <= STANCE_FORE_AFT_BAND_M},
        "lateral_band": {
            "value_m": lat_change, "limit_m": STANCE_LATERAL_BAND_M,
            "passed": lat_change <= STANCE_LATERAL_BAND_M},
        "initial_min_separation": {
            "value_m": abs(init_lat), "limit_m": STANCE_MIN_INITIAL_LATERAL_SEP_M,
            "passed": abs(init_lat) >= STANCE_MIN_INITIAL_LATERAL_SEP_M},
        "sign_no_flip": {
            "initial_sign": float(np.sign(init_lat)), "terminal_sign": float(np.sign(term_lat)),
            "passed": init_lat * term_lat > 0.0},
        "terminal_narrowing_hard_gate": {
            "value_m": narrowing, "limit_m": STANCE_MAX_TERMINAL_NARROWING_M,
            "passed": narrowing <= STANCE_MAX_TERMINAL_NARROWING_M},
    }
    passed = all(bool(c["passed"]) for c in checks.values())
    return {"passed": passed, "thresholds_frozen": True, "checks": checks}


def crossing_check(lat_sep: np.ndarray, expected_sign: float,
                   min_clearance: float) -> Dict[str, object]:
    """Per-frame no-crossing + minimum stance-width guard on a lateral-separation trace."""
    signed = lat_sep * expected_sign  # positive = healthy side
    min_sep = float(np.min(signed))
    bad = np.nonzero(signed < min_clearance)[0]
    return {
        "passed": bad.size == 0,
        "min_signed_separation_m": min_sep,
        "min_clearance_m": min_clearance,
        "first_violation_frame": int(bad[0]) if bad.size else None,
        "n_violation_frames": int(bad.size),
    }


def scale_schedule(param: Dict[str, object], beta: float,
                   min_clearance: float) -> Dict[str, object]:
    """β acts on displacements only: root displacement and footfall lateral offsets
    scale; support anchors stay pinned within their own support interval; fore-aft
    and height NEVER scale. Swing-phase lateral positions are approximated by linear
    interpolation between adjacent scaled anchors FOR THE CROSSING CHECK ONLY."""
    T = int(param["T"])  # type: ignore[arg-type]
    anchors: Dict[str, List[Dict[str, float]]] = param["support_anchors"]  # type: ignore[assignment]
    scaled_anchors: Dict[str, List[Dict[str, float]]] = {}
    lat_traces: Dict[str, np.ndarray] = {}
    for foot in ("left", "right"):
        alist = anchors[foot]
        y0 = alist[0]["lateral_m"]
        scaled = []
        for a in alist:
            scaled.append({
                "start_frame": a["start_frame"], "end_frame": a["end_frame"],
                "fore_aft_m": a["fore_aft_m"],                       # 前后不缩放
                "lateral_m": y0 + beta * (a["lateral_m"] - y0),      # 位移缩放
                "height_m": a["height_m"],                           # z 不缩放
            })
        scaled_anchors[foot] = scaled
        trace = np.empty(T, dtype=np.float64)
        trace[: scaled[0]["start_frame"] + 1] = scaled[0]["lateral_m"]
        for k, a in enumerate(scaled):
            trace[a["start_frame"]:a["end_frame"] + 1] = a["lateral_m"]
            if k + 1 < len(scaled):
                nxt = scaled[k + 1]
                gap = nxt["start_frame"] - a["end_frame"]
                if gap > 1:
                    ramp = np.linspace(a["lateral_m"], nxt["lateral_m"], gap + 1)
                    trace[a["end_frame"]:nxt["start_frame"] + 1] = ramp
        trace[scaled[-1]["end_frame"]:] = scaled[-1]["lateral_m"]
        lat_traces[foot] = trace

    init_lat = float(param["initial_separation_m"]["lateral"])  # type: ignore[index]
    expected_sign = 1.0 if init_lat >= 0 else -1.0
    scaled_sep = lat_traces["left"] - lat_traces["right"]
    cross = crossing_check(scaled_sep, expected_sign, min_clearance)
    d_source = float(param["root_signed_lateral_displacement_m"])  # type: ignore[arg-type]
    footfall_targets = {
        foot: [{"touchdown_frame": a["start_frame"],
                "fore_aft_m": a["fore_aft_m"], "lateral_m": a["lateral_m"]}
               for a in scaled_anchors[foot][1:]]
        for foot in ("left", "right")
    }
    return {
        "beta": beta,
        "achieved_root_signed_lateral_displacement_m": beta * d_source,
        "support_anchors_scaled": scaled_anchors,
        "footfall_targets": footfall_targets,
        "swing_model": "linear_interp_between_scaled_anchors (crossing-check approximation "
                       "only; the real swing path comes from the future constrained "
                       "whole-body solve, not from this spec)",
        "crossing_check_scaled": cross,
        "min_scaled_lateral_separation_m": cross["min_signed_separation_m"],
    }


# ------------------------------------------------------------------ catalog #
def catalog_lookup(catalog_path: Path, asset_id: str, expect_role: str) -> Dict[str, object]:
    payload = json.loads(catalog_path.read_text())
    for entry in payload.get("entries", []):
        if entry.get("asset_id") == asset_id:
            if entry.get("motion_role") != expect_role:
                raise FootworkScaleError(
                    "catalog_role_mismatch",
                    f"asset {asset_id!r} has motion_role={entry.get('motion_role')!r}, "
                    f"expected {expect_role!r}")
            return entry
    raise FootworkScaleError(
        "catalog_unknown_asset", f"asset {asset_id!r} not found in {catalog_path}")


# ------------------------------------------------------------------- spec #
def canonical_sha256(obj: Dict[str, object]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_spec(args: argparse.Namespace) -> Dict[str, object]:
    beta = float(args.beta)
    bucket = validate_beta(beta)

    if not (isinstance(args.strike_phase, float) and math.isfinite(args.strike_phase)
            and 0.0 < args.strike_phase < 1.0):
        raise FootworkScaleError(
            "strike_metadata_invalid",
            f"strike_phase={args.strike_phase!r} must be a finite float in (0, 1)")

    direction = args.direction
    catalog_info: Dict[str, object] = {}
    if args.catalog is not None:
        catalog_path = Path(args.catalog)
        if not catalog_path.is_file():
            raise FootworkScaleError("input_missing", f"catalog not found: {catalog_path}")
        strike_entry = catalog_lookup(catalog_path, args.strike_asset_id, "stationary_strike")
        if args.strike_sha256 and strike_entry.get("sha256") not in (None, args.strike_sha256):
            raise FootworkScaleError(
                "strike_metadata_invalid",
                f"--strike-sha256 disagrees with the catalog sha for {args.strike_asset_id!r}")
        catalog_info = {
            "catalog_path": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "strike_entry_sha256": strike_entry.get("sha256"),
        }
        if args.footwork_asset_id:
            fw_entry = catalog_lookup(catalog_path, args.footwork_asset_id,
                                      "shared_lateral_footwork_module")
            gate = str(fw_entry.get("input_gate_status", ""))
            if gate.startswith("rejected"):
                raise FootworkScaleError(
                    "footwork_source_gate_rejected",
                    f"footwork asset {args.footwork_asset_id!r} is input-gate rejected "
                    f"({gate}); a fixed clip must be re-registered under a NEW asset_id "
                    "and re-pass the gate chain — the old name cannot be borrowed")
            cat_dir = fw_entry.get("movement_direction")
            if direction is None:
                direction = cat_dir
            elif cat_dir is not None and cat_dir != direction:
                raise FootworkScaleError(
                    "direction_mismatch",
                    f"--direction {direction!r} disagrees with catalog movement_direction "
                    f"{cat_dir!r} for {args.footwork_asset_id!r}")
            catalog_info["footwork_entry_gate"] = gate or None
    if direction not in ("left", "right"):
        raise FootworkScaleError(
            "direction_missing",
            "movement direction unresolved: give --direction left|right or a catalog "
            "entry with movement_direction")

    npz_path = Path(args.footwork_npz)
    clip = load_footwork_clip(npz_path, args.body_order)
    contact_json = Path(args.contact_json) if args.contact_json else None
    phases = load_contact_phases(clip, contact_json)
    param = extract_parameterization(
        clip, phases,
        support_height_eps=args.support_height_eps,
        support_slip_budget=args.support_slip_budget,
        max_yaw_drift=args.max_yaw_drift,
    )

    # stance gate (frozen thresholds) — fail-closed
    gate = stance_gate(param)
    if not gate["passed"]:
        failed = [k for k, v in gate["checks"].items() if not v["passed"]]  # type: ignore[index]
        detail = "; ".join(
            f"{k}: value={gate['checks'][k].get('value_m')!r} "  # type: ignore[index]
            f"limit={gate['checks'][k].get('limit_m')!r}" for k in failed)  # type: ignore[index]
        raise FootworkScaleError(
            "stance_gate_failed",
            f"frozen stance gate failed ({', '.join(failed)}): {detail}. The same gate "
            "rejected all four M0 candidates (0/4); fix the motion, do not relax the gate")

    # source per-frame crossing / minimum stance width — fail-closed
    init_lat = float(param["initial_separation_m"]["lateral"])  # type: ignore[index]
    expected_sign = 1.0 if init_lat >= 0 else -1.0
    src_cross = crossing_check(
        np.asarray(param["per_frame_separation_lateral_m"]), expected_sign,
        args.min_lateral_clearance)
    if not src_cross["passed"]:
        raise FootworkScaleError(
            "foot_crossing",
            f"source clip violates the no-crossing / minimum-stance-width guard: "
            f"min signed separation {src_cross['min_signed_separation_m']:.3f} m "
            f"< clearance {args.min_lateral_clearance:.3f} m "
            f"(first violation frame {src_cross['first_violation_frame']})")

    # direction consistency + "a moving module must actually move"
    d = float(param["root_signed_lateral_displacement_m"])  # type: ignore[arg-type]
    if abs(d) < args.min_root_displacement:
        raise FootworkScaleError(
            "root_displacement_too_small",
            f"|root displacement| {abs(d):.3f} m < {args.min_root_displacement:.3f} m: "
            "a footwork module that does not move is not a footwork module; the "
            "stationary case must reference a real stationary_strike instead")
    want_sign = 1.0 if direction == "left" else -1.0
    if d * want_sign <= 0:
        raise FootworkScaleError(
            "direction_mismatch",
            f"declared direction {direction!r} expects displacement sign "
            f"{'+' if want_sign > 0 else '-'} but measured d={d:+.3f} m")

    scaled = scale_schedule(param, beta, args.min_lateral_clearance)
    if not scaled["crossing_check_scaled"]["passed"]:  # type: ignore[index]
        cc = scaled["crossing_check_scaled"]  # type: ignore[index]
        raise FootworkScaleError(
            "foot_crossing_scaled",
            f"beta={beta} scaled footfall schedule violates the no-crossing / "
            f"minimum-stance-width guard: min signed separation "
            f"{cc['min_signed_separation_m']:.3f} m < {args.min_lateral_clearance:.3f} m "
            f"(first violation frame {cc['first_violation_frame']})")

    param_out = {k: v for k, v in param.items() if k != "per_frame_separation_lateral_m"}
    spec: Dict[str, object] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "spec_kind": SPEC_KIND,
        "human_note": (
            "人话:这是脚步幅度 β 的组合规格,不是动作、不是可训练资产。它只说明"
            "\"把这条脚步模块的位移缩放 β 后,root 该到哪、脚该落哪、收步该回到什么站姿、"
            "验收要查什么\"。真正的全身轨迹由未来的受约束求解给出,且必须重过完整 Gate 链。"),
        "beta": beta,
        "beta_bucket": bucket,
        "beta_buckets": {k: list(v) for k, v in BETA_BUCKETS.items()},
        "movement_direction": direction,
        "inputs": {
            "footwork_npz": str(npz_path),
            "footwork_npz_sha256": sha256_file(npz_path),
            "footwork_asset_id": args.footwork_asset_id,
            "contact_phase_source": (
                str(contact_json) if contact_json else "npz_embedded"),
            "contact_json_sha256": sha256_file(contact_json) if contact_json else None,
            "strike": {
                "asset_id": args.strike_asset_id,
                "sha256": args.strike_sha256 or catalog_info.get("strike_entry_sha256"),
                "strike_phase": args.strike_phase,
            },
            **catalog_info,
        },
        "movement_frame": {
            "definition": "world rotated by the median initial root yaw; lateral + = robot-left",
            "yaw_rad": param["movement_frame_yaw_rad"],
            "yaw_drift_rad": param["yaw_drift_rad"],
        },
        "parameterization": param_out,
        "scaled": scaled,
        "checks": {
            "stance_gate": gate,
            "foot_crossing_source": src_cross,
            "foot_crossing_scaled": scaled["crossing_check_scaled"],
        },
        "recovery_ready_budget": {
            "human_note": (
                "人话:收步不是回到绝对位置,而是末态恢复初始左右脚分离向量。"
                "横向变窄 0.005 m 是独立硬门,3 cm 分量带兜整体近似。"),
            "target_terminal_separation_m": param["initial_separation_m"],
            "component_band_m": STANCE_LATERAL_BAND_M,
            "terminal_narrowing_hard_cap_m": STANCE_MAX_TERMINAL_NARROWING_M,
            "support_slip_budget_m": args.support_slip_budget,
            "frozen": ["component_band_m", "terminal_narrowing_hard_cap_m"],
            "tool_defaults_pending_preregistration": ["support_slip_budget_m"],
        },
        "forbidden_implementations": FORBIDDEN_IMPLEMENTATIONS,
        "training_authorized": False,
        "requires_full_gate_chain": REQUIRES_FULL_GATE_CHAIN,
    }
    spec["spec_sha256"] = canonical_sha256(spec)
    return spec


# --------------------------------------------------------------------- CLI #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Axis-D footwork amplitude β: emit a composition spec JSON "
                    "(never a trainable asset). Fail-closed on every gate.")
    p.add_argument("--footwork-npz", required=True, help="footwork module npz (schema keys "
                   "fps/joint_pos/body_pos_w/body_quat_w[/body_names/left,right_foot_contact])")
    p.add_argument("--beta", required=True, type=float,
                   help=f"footwork amplitude on the frozen grid {BETA_GRID}")
    p.add_argument("--direction", choices=("left", "right"), default=None,
                   help="movement direction (or resolved from the catalog entry)")
    p.add_argument("--strike-asset-id", required=True,
                   help="stationary_strike asset the footwork composes with")
    p.add_argument("--strike-phase", required=True, type=float,
                   help="nominal strike phase of the strike clip, in (0,1)")
    p.add_argument("--strike-sha256", default=None)
    p.add_argument("--catalog", default=None,
                   help="configs/motion_role_catalog.json for role/gate validation")
    p.add_argument("--footwork-asset-id", default=None)
    p.add_argument("--body-order", default=None, help="comma list or file; must contain "
                   f"{LEFT_FOOT_BODY} and {RIGHT_FOOT_BODY}")
    p.add_argument("--contact-json", default=None,
                   help='{"left": [[s,e],...], "right": [...]} support intervals')
    p.add_argument("--output", required=True, help="composition spec JSON path")
    p.add_argument("--min-lateral-clearance", type=float,
                   default=DEFAULT_MIN_LATERAL_CLEARANCE_M)
    p.add_argument("--support-slip-budget", type=float, default=DEFAULT_SUPPORT_SLIP_BUDGET_M)
    p.add_argument("--support-height-eps", type=float, default=DEFAULT_SUPPORT_HEIGHT_EPS_M)
    p.add_argument("--min-root-displacement", type=float,
                   default=DEFAULT_MIN_ROOT_DISPLACEMENT_M)
    p.add_argument("--max-yaw-drift", type=float, default=DEFAULT_MAX_YAW_DRIFT_RAD)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = build_spec(args)
    except FootworkScaleError as e:
        print(f"REFUSED {e}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    d = spec["scaled"]["achieved_root_signed_lateral_displacement_m"]  # type: ignore[index]
    print(f"OK beta={spec['beta']} bucket={spec['beta_bucket']} "
          f"achieved_root_displacement={d:+.4f} m -> {out}")
    print(f"spec_sha256={spec['spec_sha256']}")
    print("NOT a trainable asset: the derived composition must re-pass the full gate chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
