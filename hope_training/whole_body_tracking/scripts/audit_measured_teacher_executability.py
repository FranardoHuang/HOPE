#!/usr/bin/env python3
"""Can this measured clip be SENT to the robot as an open-loop position command?

人话:`_run_teacher_qdes_oracle` 把 measured clip 的关节角当 `q_des` 原样发下去。
本工具回答"这样发能不能行",并且把不能行的原因分成四个互不替代的问题,逐帧给数:

  1. 力矩够不够      每帧做逆动力学,算出实现这一帧需要多大关节力矩,逐关节对限幅。
                     这就是 §12.3 机械审计写"逐帧逆动力学力矩仍缺失"点名要的那一项。
  2. 脚踩没踩到地     参考帧的鞋底离地板多高。悬空的参考没有地面反力可用。
  3. 站不站得稳       参考帧的质心在不在双脚支撑多边形里。落在外面 = 静力学上站不住。
  4. 站姿对不对得上   参考的两脚间距,和 episode 出生姿态的两脚间距差多少。
                     差值要靠迈步才能消除,而开环位置指令不会迈步。

四问互不替代:力矩全部合格 **不等于** 这条 clip 可执行 —— 2/3/4 任何一条不过,
再完美的前馈力矩也没有用,因为关节力矩是内力,既不能凭空产生地面反力,也不能挪动被摩擦
钉住的脚。

前馈变换(问题 1 合格时才有意义):运行时 PD 是 `tau = kp*(q_des - q) - kd*qd`,
所以要让完美跟踪时 PD 输出等于前馈力矩 `tau_ff`,该发的位置指令是

    q_des[t] = q_ref[t] + (tau_ff[t] + kd * qd_ref[t]) / kp        (kp/kd 逐关节,不是标量)

等待期用的 `hold_qdes` 就是这条公式在 `qd_ref = qdd_ref = 0` 处的特例,本工具用它做
fail-closed 锚点:复现不出存档的保持力矩就拒绝出报告,不出一份没有校准的数。

逆动力学在 **Isaac 等效 plant** 上做(armature = 运行时表、`dof_damping = 0`、
`dof_frictionloss = 0`),因为这份 `tau_ff` 是要给 Isaac 用的;厂商 MJCF 自带的
`dof_damping`/`dof_frictionloss` 属于另一台 plant,差多少一并报出来。

用法(pod,需要 mujoco;不需要 GPU):
    CUDA_VISIBLE_DEVICES= /workspace/hope_isaac_venv/bin/python \
        hope_training/whole_body_tracking/scripts/audit_measured_teacher_executability.py \
        --dynamic-ready configs/.../take061...dynamic_ready.v2.json \
        --motion assets/motions/.../hope_Take_061_unit04_BH.npz \
        --mjcf agi/A3_MuJoCo_Sim/.../a3_pingpong.xml \
        --json out.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# 加速度靠数值微分得到,噪声会被放大两次,所以微分档位必须显式声明、并且逐档都跑一遍。
# 名字里的数字是 Savitzky-Golay 的 (窗口帧数, 多项式阶数);"raw" 是不平滑的中心差分。
DIFFERENTIATION_MODES: tuple[str, ...] = ("raw", "sg5_2", "sg7_2", "sg9_3", "sg13_3")
DEFAULT_MODE = "sg7_2"

# 鞋底顶点低于"**该脚自己**最低点 + 这个带宽"才算这只脚的支撑点。带宽取 6 mm:
# 比 LP 的 2 mm 接触判据宽,免得把"能站住但被判定悬空"错报成"站不住"。
#
# 「该脚自己」这四个字是 2026-08-07 修掉的一个真 bug。原来的写法把带宽锚在**两只脚合起来**
# 的最低顶点上,于是较高那只脚能不能进支撑多边形,取决于左右脚差多少毫米,而不取决于
# 它到底踩没踩到地:左右差 `4.8 mm` 时只捞到它最底下 `1.2 mm` 的一条边,半个多边形塌成
# 一条线;差超过 `6 mm` 时它整只脚消失。产出的裕度于是变成负数,并被读成"质心出支撑区"——
# 而真相是**量具把多边形算塌了**。变异证据:只把左脚鞋底顶点下移 `4.86 mm`(机器人姿态、
# 质心、关节角一律不动),现役规则的裕度从 `-11.95` 跳到 `+26.37 mm`。
SUPPORT_BAND_M = 6.0e-3

# 一只脚"算不算踩在地上",用的是它**自己**最低碰撞顶点落不落在地面 LP 的窗口里
# (下面那两个 LP 容差,不另立一套判据)。两只脚都不在窗口里 = 没有支撑多边形可言,
# 这时报 ``None`` 加一个具名状态,**不报一个负数假装质心出界**。
#
# 支撑多边形的最小宽度低于这个值就叫"退化":几何上它是一条线,任何"有符号裕度"都没有意义。
# 取 0.1 mm —— 比鞋底网格顶点坐标的量级小三个数量级,只会抓真正塌掉的那种。
DEGENERATE_SUPPORT_WIDTH_M = 1.0e-4

# 锚点容差:`kp*(hold_qdes - q_birth)` 与存档保持力矩的允许偏差。存档实测是 3e-15,
# 给到 1e-9 仍然是"完全一致"的量级,不给它偷偷漂的空间。
HOLD_ANCHOR_TOLERANCE_NM = 1.0e-9

# 地面 LP(`GroundContactConfig`)对"踩到地"的窗口:低于 -2 mm 算穿透、高于 +2 mm 算悬空。
# 两只脚必须同时落在这 4 mm 里,所以两只脚之间的高度差超过 4 mm 时,**任何刚体平移都救不了**。
# 这两个数是从 `canonical_torque_path_topp.GroundContactConfig` 的默认值抄过来的判据,
# 本工具只用来报"能不能靠平移修好",不参与任何准入。
LP_CONTACT_GAP_TOLERANCE_M = 2.0e-3
LP_PENETRATION_TOLERANCE_M = 2.0e-3


class ExecutabilityAuditError(RuntimeError):
    """本审计的任何 fail-closed 拒绝。"""


def _collision_sole_geoms(
    *,
    geom_bodyid: np.ndarray,
    geom_type: np.ndarray,
    geom_contype: np.ndarray,
    geom_conaffinity: np.ndarray,
    body_ids: Sequence[int],
    mesh_type: int,
) -> list[int]:
    """脚底顶点只能取**会碰撞**的那块网格。

    人话:A3 的每个 ankle_roll 上挂着两个 mesh —— 一个 collision(`contype=1 conaffinity=7`),
    一个纯视觉(`contype=0 conaffinity=0`)。视觉网格比 collision 网格**低 1.12 mm**。
    把视觉网格算进来,"离地"会少报 1.12 mm、"压地"会多报 1.12 mm;而 MuJoCo 的接触检测、
    以及产出 WAIT `hold_qdes` 的那套地面 LP,用的都是 collision 网格
    (`canonical_torque_path_topp` 选 geom 的条件就是 `contype != 0 and conaffinity != 0`)。
    两边必须问同一块几何,否则报出来的毫米数没人能跟 LP 的判定对上。
    """

    selected = [
        int(geom)
        for geom in range(int(np.asarray(geom_bodyid).shape[0]))
        if int(geom_bodyid[geom]) in set(int(body) for body in body_ids)
        and int(geom_type[geom]) == int(mesh_type)
        and int(geom_contype[geom]) != 0
        and int(geom_conaffinity[geom]) != 0
    ]
    if not selected:
        raise ExecutabilityAuditError(
            "MJCF exposes no collidable ankle-roll sole meshes; refusing to measure "
            "sole clearance against geometry MuJoCo does not collide"
        )
    return selected


def _savgol(values: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay 一阶导(逐列),边界靠平移窗口而不是补零,避免端点被拉平。"""

    count = values.shape[0]
    if window > count:
        raise ExecutabilityAuditError(
            f"differentiation window {window} exceeds the {count}-frame clip"
        )
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(offsets, poly + 1, increasing=True)
    pseudo = np.linalg.pinv(design)
    out = np.zeros_like(values)
    for index in range(count):
        low = min(max(index - half, 0), count - window)
        coeffs = pseudo @ values[low : low + window]
        u = float(index - low - half)
        weights = np.array(
            [k * u ** (k - 1) if k >= 1 else 0.0 for k in range(poly + 1)],
            dtype=np.float64,
        )
        out[index] = weights @ coeffs
    return out


def _derivative(values: np.ndarray, mode: str, dt: float) -> np.ndarray:
    if mode == "raw":
        out = np.zeros_like(values)
        out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
        out[0] = (values[1] - values[0]) / dt
        out[-1] = (values[-1] - values[-2]) / dt
        return out
    if not mode.startswith("sg"):
        raise ExecutabilityAuditError(f"unknown differentiation mode {mode!r}")
    window_text, _, poly_text = mode[2:].partition("_")
    return _savgol(values, int(window_text), int(poly_text)) / dt


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(points, 9), axis=0)
    if unique.shape[0] < 3:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def _chain(rows: np.ndarray) -> list[np.ndarray]:
        stack: list[np.ndarray] = []
        for point in rows:
            while len(stack) >= 2:
                cross = np.cross(stack[-1] - stack[-2], point - stack[-2])
                if cross > 0.0:
                    break
                stack.pop()
            stack.append(point)
        return stack

    lower = _chain(ordered)[:-1]
    upper = _chain(ordered[::-1])[:-1]
    return np.asarray(lower + upper)


def _hull_margin(point: np.ndarray, hull: np.ndarray) -> float:
    """点到凸包内部的有符号距离:正数 = 在里面,负数 = 在外面(米)。"""

    if hull.shape[0] < 3:
        return float("-inf")
    best = float("inf")
    count = hull.shape[0]
    for index in range(count):
        a = hull[index]
        edge = hull[(index + 1) % count] - a
        length = float(np.linalg.norm(edge))
        if length < 1.0e-12:
            continue
        outward = np.array([edge[1], -edge[0]], dtype=np.float64) / length
        best = min(best, -float(np.dot(point - a, outward)))
    return best


def _hull_area(hull: np.ndarray) -> float:
    """凸包面积(平方米);少于三个顶点就是 0。"""

    if hull.shape[0] < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def _hull_minimum_width(hull: np.ndarray) -> float:
    """凸包的最小宽度(米):沿每条边方向投影,取最薄的那一档。

    这是判"多边形有没有塌成一条线"的那把尺。塌掉的多边形上,任何"有符号裕度"
    都不是物理量,只是量具的产物,所以本工具遇到它时报 ``None`` 而不是报一个负数。
    """

    if hull.shape[0] < 3:
        return 0.0
    best = float("inf")
    count = hull.shape[0]
    for index in range(count):
        a = hull[index]
        edge = hull[(index + 1) % count] - a
        length = float(np.linalg.norm(edge))
        if length < 1.0e-12:
            continue
        normal = np.array([edge[1], -edge[0]], dtype=np.float64) / length
        offsets = (hull - a) @ normal
        best = min(best, float(offsets.max() - offsets.min()))
    return 0.0 if not math.isfinite(best) else best


def support_polygon(
    by_foot: Sequence[np.ndarray],
    com_floor_xy: np.ndarray,
    *,
    band_m: float = SUPPORT_BAND_M,
    contact_gap_tolerance_m: float | None = None,
    penetration_tolerance_m: float | None = None,
) -> dict[str, Any]:
    """从两只脚的鞋底顶点建支撑多边形,并**说清楚它是不是能用**。

    规则(2026-08-07 改)——

    * 每只脚的支撑点取**它自己**最低顶点 + ``band_m``,不再锚在两只脚合起来的最低点上;
    * 一只脚只有在**它自己**最低顶点落进地面 LP 窗口(``[-penetration, +gap]``)时才参与
      建多边形。悬在空中的脚不贡献支撑点 —— 这一条以前是漏的,于是"整条 clip 悬空 1 cm"
      也照样能算出一个支撑多边形来;
    * 建出来的多边形若少于三个顶点、或最小宽度低于 ``DEGENERATE_SUPPORT_WIDTH_M``,
      **不出裕度数字**,出一个具名状态。

    返回 ``margin_m`` 为 ``None`` 时,``status`` 说明原因,调用方不得把它当成"质心出界"。
    """

    if contact_gap_tolerance_m is None:
        contact_gap_tolerance_m = LP_CONTACT_GAP_TOLERANCE_M
    if penetration_tolerance_m is None:
        penetration_tolerance_m = LP_PENETRATION_TOLERANCE_M
    rows = [np.asarray(foot, np.float64) for foot in by_foot]
    if len(rows) != 2 or any(foot.ndim != 2 or foot.shape[1] != 3 for foot in rows):
        raise ExecutabilityAuditError(
            "support polygon needs exactly two (N, 3) sole vertex arrays"
        )
    lowest = [float(foot[:, 2].min()) for foot in rows]
    on_floor = [
        bool(-float(penetration_tolerance_m) <= value <= float(contact_gap_tolerance_m))
        for value in lowest
    ]
    bands = [
        foot[foot[:, 2] <= value + float(band_m)]
        for foot, value in zip(rows, lowest)
    ]
    per_foot_hulls = [_convex_hull_2d(band[:, :2]) for band in bands]
    contributing = [index for index in (0, 1) if on_floor[index]]
    detail: dict[str, Any] = {
        "band_rule": (
            "each foot's OWN lowest collision vertex + band; only feet whose own "
            "lowest vertex lies inside the ground-LP window contribute"
        ),
        "band_m": float(band_m),
        "contact_gap_tolerance_m": float(contact_gap_tolerance_m),
        "penetration_tolerance_m": float(penetration_tolerance_m),
        "degenerate_support_width_m": DEGENERATE_SUPPORT_WIDTH_M,
        "foot_lowest_vertex_z_m": lowest,
        "foot_on_floor": on_floor,
        "foot_support_vertex_count": [int(band.shape[0]) for band in bands],
        "foot_support_hull_min_width_m": [
            _hull_minimum_width(hull) for hull in per_foot_hulls
        ],
        "foot_support_hull_area_m2": [_hull_area(hull) for hull in per_foot_hulls],
        "contributing_feet": contributing,
    }
    if not contributing:
        detail.update(
            {
                "status": "NO_FOOT_ON_FLOOR",
                "hull_floor_xy_m": [],
                "hull_vertex_count": 0,
                "hull_area_m2": 0.0,
                "hull_minimum_width_m": 0.0,
                "margin_m": None,
            }
        )
        return detail
    support = np.vstack([bands[index] for index in contributing])
    hull = _convex_hull_2d(support[:, :2])
    width = _hull_minimum_width(hull)
    detail.update(
        {
            "hull_floor_xy_m": hull.tolist(),
            "hull_vertex_count": int(hull.shape[0]),
            "hull_area_m2": _hull_area(hull),
            "hull_minimum_width_m": width,
        }
    )
    if hull.shape[0] < 3 or width < DEGENERATE_SUPPORT_WIDTH_M:
        detail["status"] = "DEGENERATE_SUPPORT_POLYGON"
        detail["margin_m"] = None
        return detail
    detail["status"] = (
        "DOUBLE_SUPPORT" if len(contributing) == 2 else "SINGLE_FOOT_SUPPORT"
    )
    detail["margin_m"] = _hull_margin(
        np.asarray(com_floor_xy, np.float64)[:2], hull
    )
    return detail


class _Plant:
    """Isaac 等效的 MuJoCo plant,外加它的逆动力学副本(约束关掉)。"""

    def __init__(self, mujoco: Any, mjcf: Path, runtime_plant: dict) -> None:
        scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import mujoco_table_scene  # noqa: PLC0415  (只有 pod 上装了 mujoco 才 import 得动)

        scene = mujoco_table_scene.load_table_scene(
            mujoco, mjcf, collidable=True, action_ball_policy=True
        )
        model = scene.model
        self.names = list(runtime_plant["joint_names"])
        self.kp = np.asarray(runtime_plant["joint_stiffness"], np.float64)
        self.kd = np.asarray(runtime_plant["joint_damping"], np.float64)
        self.effort = np.asarray(runtime_plant["joint_effort_limits"], np.float64)
        self.qdes_lower = np.asarray(runtime_plant["executed_qdes_lower_rad"], np.float64)
        self.qdes_upper = np.asarray(runtime_plant["executed_qdes_upper_rad"], np.float64)
        armature = np.asarray(runtime_plant["joint_armature"], np.float64)
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.names
        ]
        if any(jid < 0 for jid in joint_ids):
            raise ExecutabilityAuditError("MJCF is missing a runtime joint name")
        self.qpos_adr = np.asarray(model.jnt_qposadr[joint_ids], np.int64)
        self.dof_adr = np.asarray(model.jnt_dofadr[joint_ids], np.int64)
        # 厂商 plant 的被动项先留档,再改成 Isaac 等效值 —— 差多少要报出来,不能悄悄抹掉。
        self.vendor_damping = np.asarray(model.dof_damping[self.dof_adr], np.float64).copy()
        self.vendor_frictionloss = np.asarray(
            model.dof_frictionloss[self.dof_adr], np.float64
        ).copy()
        model.opt.timestep = float(runtime_plant["physics_step_dt_s"])
        model.dof_armature[self.dof_adr] = armature
        model.dof_damping[self.dof_adr] = 0.0
        model.dof_frictionloss[self.dof_adr] = 0.0
        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        inverse = copy.copy(model)
        inverse.opt.disableflags = int(inverse.opt.disableflags) | int(
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        )
        self.inverse_model = inverse
        self.inverse_data = mujoco.MjData(inverse)
        self.ankle_bodies = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_Link")
            for side in ("left", "right")
        ]
        if any(body < 0 for body in self.ankle_bodies):
            raise ExecutabilityAuditError("MJCF is missing an ankle-roll body")
        self.sole_geoms_by_foot = [
            _collision_sole_geoms(
                geom_bodyid=np.asarray(model.geom_bodyid),
                geom_type=np.asarray(model.geom_type),
                geom_contype=np.asarray(model.geom_contype),
                geom_conaffinity=np.asarray(model.geom_conaffinity),
                body_ids=(body,),
                mesh_type=int(mujoco.mjtGeom.mjGEOM_MESH),
            )
            for body in self.ankle_bodies
        ]
        self.sole_geoms = [
            geom for foot in self.sole_geoms_by_foot for geom in foot
        ]

    def qpos(self, root_pos: np.ndarray, root_quat: np.ndarray, joints: np.ndarray) -> np.ndarray:
        out = np.zeros(int(self.model.nq), np.float64)
        out[0:3] = root_pos
        quat = np.asarray(root_quat, np.float64)
        out[3:7] = quat / np.linalg.norm(quat)
        out[self.qpos_adr] = joints
        return out

    def inverse_dynamics(
        self, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray
    ) -> np.ndarray:
        self.inverse_data.qpos[:] = qpos
        self.inverse_data.qvel[:] = qvel
        self.inverse_data.qacc[:] = qacc
        self.mujoco.mj_inverse(self.inverse_model, self.inverse_data)
        force = np.asarray(self.inverse_data.qfrc_inverse, np.float64).copy()
        if not np.all(np.isfinite(force)):
            raise ExecutabilityAuditError("inverse dynamics returned non-finite force")
        return force

    def _sole_vertices_world(self, geom: int) -> np.ndarray:
        mesh = int(self.model.geom_dataid[geom])
        start = int(self.model.mesh_vertadr[mesh])
        count = int(self.model.mesh_vertnum[mesh])
        verts = np.asarray(self.model.mesh_vert[start : start + count], np.float64)
        rot = np.asarray(self.data.geom_xmat[geom], np.float64).reshape(3, 3)
        return verts @ rot.T + np.asarray(self.data.geom_xpos[geom], np.float64)

    def ground_geometry(self, qpos: np.ndarray) -> dict:
        """一帧的鞋底离地高度、支撑多边形和质心裕度。

        质心裕度报**两个**,因为它们回答的不是同一个问题:

        * ``com_support_margin_m`` —— 只用**真的踩在地上**的那些脚,按每只脚自己的接触带建
          多边形(见 :func:`support_polygon`)。它衡量的是**这一帧照原样**站不站得住。
          没有脚落地、或者多边形塌成一条线时,这一项是 ``None``,原因写在
          ``support_polygon.status`` 里 —— **不出一个负数冒充"质心出支撑区"**。
        * ``com_footprint_margin_m`` —— 拿两只脚**完整鞋底**在地面上的投影当多边形。它等价于问
          "要是重定向把两只脚都放平踩实,质心在不在两脚之间"。两者差很多时,失衡是**重定向没踩地**
          的下游后果,不是动捕对象自己站不稳。
        """

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.mujoco.mj_comPos(self.model, self.data)
        com = np.asarray(self.data.subtree_com[0], np.float64)
        by_foot = [
            np.vstack([self._sole_vertices_world(geom) for geom in foot])
            for foot in self.sole_geoms_by_foot
        ]
        per_foot_lowest = [float(rows[:, 2].min()) for rows in by_foot]
        lowest = min(per_foot_lowest)
        polygon = support_polygon(by_foot, com[:2])
        footprint = _convex_hull_2d(np.vstack(by_foot)[:, :2])
        ankles = [np.asarray(self.data.xpos[body], np.float64) for body in self.ankle_bodies]
        return {
            "sole_lowest_vertex_z_m": lowest,
            "sole_lowest_vertex_z_by_foot_m": per_foot_lowest,
            "com_support_margin_m": polygon["margin_m"],
            "support_polygon": polygon,
            "com_footprint_margin_m": _hull_margin(com[:2], footprint),
            "stance_width_m": float(np.linalg.norm(ankles[0][:2] - ankles[1][:2])),
            "com_xy_m": com[:2].tolist(),
        }


def _load_clip(motion: Path) -> dict:
    payload = np.load(motion)
    for key in ("joint_pos", "body_pos_w", "body_quat_w", "fps"):
        if key not in payload.files:
            raise ExecutabilityAuditError(f"motion npz is missing {key!r}")
    return {
        "joint_pos": np.asarray(payload["joint_pos"], np.float64),
        "root_pos": np.asarray(payload["body_pos_w"], np.float64)[:, 0],
        "root_quat": np.asarray(payload["body_quat_w"], np.float64)[:, 0],
        "fps": float(np.asarray(payload["fps"]).reshape(-1)[0]),
    }


def _verify_hold_anchor(plant: _Plant, artifact: dict) -> dict:
    """Fail-closed 锚点:存档 hold 必须是"出生姿态 + 保持力矩/kp"这条公式的解。

    复现不了就整份报告作废 —— 说明本工具对 plant 或对 PD 律的理解和产出契约的那套不一致,
    这时候再往下算逐帧力矩只会给出一份没人能信的数。
    """

    birth = np.asarray(artifact["physical_ready"]["joint_pos_rad"], np.float64)
    hold = np.asarray(artifact["hold_candidate"]["hold_qdes_joint_pos_rad"], np.float64)
    archived = np.asarray(
        artifact["hold_candidate"]["actuator_generalized_force_runtime_order_nm"],
        np.float64,
    )
    residual = float(np.abs(plant.kp * (hold - birth) - archived).max())
    if not np.isfinite(residual) or residual > HOLD_ANCHOR_TOLERANCE_NM:
        raise ExecutabilityAuditError(
            "archived hold q_des does not reproduce the archived holding torque under "
            f"tau = kp*(q_des - q): max residual {residual:.3e} N*m exceeds "
            f"{HOLD_ANCHOR_TOLERANCE_NM:.1e}; refusing to report per-frame torque"
        )
    return {
        "max_residual_nm": residual,
        "tolerance_nm": HOLD_ANCHOR_TOLERANCE_NM,
        "semantics": "kp*(hold_qdes - q_birth) == archived holding torque",
    }


def audit(
    *, dynamic_ready: Path, motion: Path, mjcf: Path, modes: tuple[str, ...]
) -> dict:
    import mujoco  # noqa: PLC0415

    artifact = json.loads(dynamic_ready.read_text())
    plant = _Plant(mujoco, mjcf, artifact["runtime_plant"])
    anchor = _verify_hold_anchor(plant, artifact)

    clip = _load_clip(motion)
    frames = clip["joint_pos"].shape[0]
    step = 1.0 / clip["fps"]
    if clip["joint_pos"].shape[1] != len(plant.names):
        raise ExecutabilityAuditError("clip joint width does not match the runtime plant")

    nq, nv = int(plant.model.nq), int(plant.model.nv)
    qpos = np.stack(
        [
            plant.qpos(clip["root_pos"][t], clip["root_quat"][t], clip["joint_pos"][t])
            for t in range(frames)
        ]
    )
    # qvel 用 MuJoCo 自己的 mj_differentiatePos,这样四元数的切空间约定天生就是 MuJoCo 的。
    qvel = np.zeros((frames, nv), np.float64)
    for t in range(frames):
        low, high = max(t - 1, 0), min(t + 1, frames - 1)
        scratch = np.zeros(nv, np.float64)
        mujoco.mj_differentiatePos(
            plant.inverse_model, scratch, (high - low) * step, qpos[low], qpos[high]
        )
        qvel[t] = scratch

    torque_by_mode: dict[str, np.ndarray] = {}
    for mode in modes:
        qacc = _derivative(qvel, mode, step)
        rows = np.zeros((frames, len(plant.names)), np.float64)
        for t in range(frames):
            rows[t] = plant.inverse_dynamics(qpos[t], qvel[t], qacc[t])[plant.dof_adr]
        torque_by_mode[mode] = rows

    reference = torque_by_mode[modes[0] if DEFAULT_MODE not in modes else DEFAULT_MODE]
    ratio = np.abs(reference) / plant.effort[None, :]
    per_joint = []
    for index, name in enumerate(plant.names):
        column = np.abs(reference[:, index])
        per_joint.append(
            {
                "joint": name,
                "effort_limit_nm": float(plant.effort[index]),
                "max_abs_tau_nm": float(column.max()),
                "max_utilisation_fraction": float(ratio[:, index].max()),
                "frames_over_limit": int((ratio[:, index] > 1.0).sum()),
                "worst_frame": int(np.argmax(column)),
            }
        )
    per_joint.sort(key=lambda row: -row["max_utilisation_fraction"])

    sensitivity = {
        mode: {
            "max_abs_tau_nm": float(np.abs(rows).max()),
            "over_limit_cells": int((np.abs(rows) > plant.effort[None, :]).sum()),
            "worst_joint": plant.names[
                int(np.argmax((np.abs(rows) / plant.effort[None, :]).max(axis=0)))
            ],
            "worst_utilisation_fraction": float(
                (np.abs(rows) / plant.effort[None, :]).max()
            ),
        }
        for mode, rows in torque_by_mode.items()
    }

    # 前馈位置指令,以及它会不会被运行时的 qdes 包络截掉(截掉 = 前馈根本送不到)。
    qd_ref = qvel[:, plant.dof_adr]
    qdes_ff = clip["joint_pos"] + (reference + plant.kd[None, :] * qd_ref) / plant.kp[None, :]
    clipped = np.clip(qdes_ff, plant.qdes_lower[None, :], plant.qdes_upper[None, :])
    envelope = {
        "semantics": "q_des[t] = q_ref[t] + (tau_ff[t] + kd*qd_ref[t]) / kp, per joint",
        "cells_outside_executed_envelope": int((clipped != qdes_ff).sum()),
        "max_clip_shift_rad": float(np.abs(clipped - qdes_ff).max()),
        "frames": frames,
        "joints": len(plant.names),
    }

    ground_rows = [plant.ground_geometry(qpos[t]) for t in range(frames)]
    sole = np.array([row["sole_lowest_vertex_z_m"] for row in ground_rows])
    by_foot = np.array([row["sole_lowest_vertex_z_by_foot_m"] for row in ground_rows])
    # 支撑裕度可能是 ``None``(没脚落地 / 多边形退化)。那些帧**不算"质心出支撑区"**,
    # 它们单独进 ``support_polygon_status_counts``,否则量具的失效会被读成机器人的失衡。
    margin_defined = [
        float(row["com_support_margin_m"])
        for row in ground_rows
        if row["com_support_margin_m"] is not None
    ]
    footprint_margin = np.array([row["com_footprint_margin_m"] for row in ground_rows])
    stance = np.array([row["stance_width_m"] for row in ground_rows])
    status_counts: dict[str, int] = {}
    for row in ground_rows:
        key = str(row["support_polygon"]["status"])
        status_counts[key] = status_counts.get(key, 0) + 1

    # "整条 clip 悬空 1 cm,那把它整体压下去 1 cm 不就好了?" —— 这里把那个反问算成一道判据。
    # 一次刚体竖直平移 d 要让每一帧的两只脚都落进 LP 的接触窗口,需要
    #   max(两脚较高者) - d <= +gap   且   min(两脚较低者) - d >= -penetration
    # 解出来的 d 区间为空,就说明"平移一下"这条修法在几何上不存在,问题不是高度约定。
    highest_foot = float(by_foot.max())
    lowest_foot = float(by_foot.min())
    shift_upper = lowest_foot + LP_PENETRATION_TOLERANCE_M
    shift_lower = highest_foot - LP_CONTACT_GAP_TOLERANCE_M
    rigid_shift = {
        "semantics": (
            "is there ONE constant vertical offset that puts both feet inside the ground LP's "
            "[-penetration, +gap] window on every frame"
        ),
        "lp_contact_gap_tolerance_m": LP_CONTACT_GAP_TOLERANCE_M,
        "lp_penetration_tolerance_m": LP_PENETRATION_TOLERANCE_M,
        "required_shift_lower_bound_m": shift_lower,
        "required_shift_upper_bound_m": shift_upper,
        "feasible": bool(shift_lower <= shift_upper),
        "max_left_minus_right_mm": float(
            np.abs(by_foot[:, 0] - by_foot[:, 1]).max() * 1000.0
        ),
    }

    ready = artifact["physical_ready"]
    birth_geometry = plant.ground_geometry(
        plant.qpos(
            np.asarray(ready["root_pos_w_m"], np.float64),
            np.asarray(ready["root_quat_wxyz"], np.float64),
            np.asarray(ready["joint_pos_rad"], np.float64),
        )
    )

    return {
        "kind": "measured_teacher_open_loop_executability_audit_v1",
        "schema_version": 1,
        "diagnostic_unauthorized": True,
        "sources": {
            "dynamic_ready": str(dynamic_ready),
            "motion": str(motion),
            "mjcf": str(mjcf),
        },
        "clip": {"frames": frames, "fps": clip["fps"]},
        "hold_anchor": anchor,
        "q1_actuator_torque": {
            "differentiation_mode": DEFAULT_MODE if DEFAULT_MODE in modes else modes[0],
            "per_joint": per_joint,
            "filter_sensitivity": sensitivity,
        },
        "q2_ground_contact": {
            "geometry_measured": "ankle-roll COLLISION meshes only (contype!=0 and conaffinity!=0)",
            "sole_lowest_vertex_z_mm": {
                "clip_min": float(sole.min() * 1000.0),
                "clip_max": float(sole.max() * 1000.0),
                "clip_frame0": float(sole[0] * 1000.0),
                "birth": float(birth_geometry["sole_lowest_vertex_z_m"] * 1000.0),
                "clip_left_min": float(by_foot[:, 0].min() * 1000.0),
                "clip_left_max": float(by_foot[:, 0].max() * 1000.0),
                "clip_right_min": float(by_foot[:, 1].min() * 1000.0),
                "clip_right_max": float(by_foot[:, 1].max() * 1000.0),
            },
            "frames_with_no_floor_contact": int((sole > 2.0e-3).sum()),
            "single_rigid_vertical_shift": rigid_shift,
        },
        "q3_static_balance": {
            "com_support_margin_mm": {
                "semantics": (
                    "CoM vs the polygon built from the feet that are ACTUALLY on the "
                    "floor, each foot banded from its OWN lowest collision vertex; "
                    "reported only for frames whose polygon is defined and "
                    "non-degenerate -- a frame with no defined polygon is NOT counted "
                    "as CoM-outside-support"
                ),
                "frames_with_defined_polygon": len(margin_defined),
                "clip_min": (
                    float(min(margin_defined) * 1000.0) if margin_defined else None
                ),
                "clip_max": (
                    float(max(margin_defined) * 1000.0) if margin_defined else None
                ),
                "clip_mean": (
                    float(sum(margin_defined) / len(margin_defined) * 1000.0)
                    if margin_defined
                    else None
                ),
                "clip_frame0": (
                    None
                    if ground_rows[0]["com_support_margin_m"] is None
                    else float(ground_rows[0]["com_support_margin_m"] * 1000.0)
                ),
                "birth": (
                    None
                    if birth_geometry["com_support_margin_m"] is None
                    else float(birth_geometry["com_support_margin_m"] * 1000.0)
                ),
            },
            "com_footprint_margin_mm": {
                "semantics": (
                    "CoM vs the convex hull of BOTH complete soles projected on the floor, i.e. "
                    "the margin a retarget that actually planted both feet flat would have"
                ),
                "clip_min": float(footprint_margin.min() * 1000.0),
                "clip_max": float(footprint_margin.max() * 1000.0),
                "clip_mean": float(footprint_margin.mean() * 1000.0),
                "clip_frame0": float(footprint_margin[0] * 1000.0),
                "birth": float(birth_geometry["com_footprint_margin_m"] * 1000.0),
            },
            "frames_com_outside_support": sum(
                1 for value in margin_defined if value < 0.0
            ),
            "frames_support_polygon_undefined": frames - len(margin_defined),
            "support_polygon_status_counts": status_counts,
            "birth_support_polygon_status": str(
                birth_geometry["support_polygon"]["status"]
            ),
            "frames_com_outside_footprint": int((footprint_margin < 0.0).sum()),
            "support_band_m": SUPPORT_BAND_M,
            "degenerate_support_width_m": DEGENERATE_SUPPORT_WIDTH_M,
        },
        "q4_stance_gap": {
            "clip_stance_width_m": {
                "min": float(stance.min()),
                "max": float(stance.max()),
                "frame0": float(stance[0]),
            },
            "birth_stance_width_m": float(birth_geometry["stance_width_m"]),
            "gap_m": float(stance[0] - birth_geometry["stance_width_m"]),
        },
        "plant_deviation_vendor_vs_isaac": {
            "semantics": "vendor MJCF passive terms that the Isaac-equivalent plant zeroes",
            "dof_damping_nms_per_rad": {
                "min": float(plant.vendor_damping.min()),
                "max": float(plant.vendor_damping.max()),
            },
            "dof_frictionloss_nm": {
                "min": float(plant.vendor_frictionloss.min()),
                "max": float(plant.vendor_frictionloss.max()),
            },
        },
        "feedforward_command": envelope,
    }


def _mm(value: float | None) -> str:
    """支撑裕度可能没有定义;摘要里就得写「未定义」,不能印一个 0 或者一个负数。"""

    return "未定义" if value is None else f"{value:+.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-ready", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--modes", default=",".join(DIFFERENTIATION_MODES))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    modes = tuple(part for part in str(args.modes).split(",") if part)
    if not modes:
        raise ExecutabilityAuditError("at least one differentiation mode is required")
    report = audit(
        dynamic_ready=args.dynamic_ready,
        motion=args.motion,
        mjcf=args.mjcf,
        modes=modes,
    )
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.json is not None:
        args.json.write_text(text + "\n")
    print(text)

    torque = report["q1_actuator_torque"]["per_joint"][0]
    ground = report["q2_ground_contact"]
    balance = report["q3_static_balance"]
    stance = report["q4_stance_gap"]
    print(
        "\n人话摘要:"
        f"\n  1 力矩:最吃紧的是 {torque['joint']},{torque['max_abs_tau_nm']:.1f} N*m 对限幅 "
        f"{torque['effort_limit_nm']:.1f}({torque['max_utilisation_fraction'] * 100:.0f}%),"
        f"超限帧 {torque['frames_over_limit']}"
        f"\n  2 踩地:鞋底离地 左 {ground['sole_lowest_vertex_z_mm']['clip_left_min']:.1f}~"
        f"{ground['sole_lowest_vertex_z_mm']['clip_left_max']:.1f} mm / "
        f"右 {ground['sole_lowest_vertex_z_mm']['clip_right_min']:.1f}~"
        f"{ground['sole_lowest_vertex_z_mm']['clip_right_max']:.1f} mm,"
        f"{ground['frames_with_no_floor_contact']}/{report['clip']['frames']} 帧没接触地板"
        f"(出生姿态 {ground['sole_lowest_vertex_z_mm']['birth']:.1f} mm);"
        f"两脚最大高差 {ground['single_rigid_vertical_shift']['max_left_minus_right_mm']:.1f} mm,"
        f"整体平移能否修好:"
        f"{'能' if ground['single_rigid_vertical_shift']['feasible'] else '不能(平移救不了)'}"
        f"\n  3 站稳:质心裕度 {_mm(balance['com_support_margin_mm']['clip_min'])}~"
        f"{_mm(balance['com_support_margin_mm']['clip_max'])} mm,"
        f"{balance['frames_com_outside_support']}/{report['clip']['frames']} 帧质心在支撑面外;"
        f"另有 {balance['frames_support_polygon_undefined']}/{report['clip']['frames']} 帧"
        f"**根本建不出支撑多边形**(状态计数 {balance['support_polygon_status_counts']},"
        f"这些帧不算「出界」)"
        f"(出生姿态 {_mm(balance['com_support_margin_mm']['birth'])} mm / "
        f"{balance['birth_support_polygon_status']});"
        f"换成「两脚放平踩实」的鞋底footprint,裕度 "
        f"{balance['com_footprint_margin_mm']['clip_min']:+.1f}~"
        f"{balance['com_footprint_margin_mm']['clip_max']:+.1f} mm,"
        f"出界 {balance['frames_com_outside_footprint']}/{report['clip']['frames']} 帧"
        f"\n  4 站姿:clip {stance['clip_stance_width_m']['frame0']:.3f} m vs 出生 "
        f"{stance['birth_stance_width_m']:.3f} m,差 {stance['gap_m']:.3f} m",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
