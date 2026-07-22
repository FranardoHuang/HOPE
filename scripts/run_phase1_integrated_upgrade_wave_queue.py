#!/usr/bin/env python3
"""渲染集成升级波 3 臂队列的 SSH 命令（phase1_integrated_upgrade_wave_20260723）。

人话：这个程序只做三件事——(1) ``--plan`` 打印 3 臂计划表和推荐填充顺序；(2)
``--render-stage probe|science --render-job <id> --pod <pod1|pod2> --gpu <0|1|2>``
输出单臂的逐字 SSH 命令给人工核对后执行（本波不写死 pod/gpu，空槽即填，目标卡
在渲染时注入）；(3) ``--checklist`` 输出发射前依赖核对单。它自己绝不 SSH、绝不
发信号、绝不写远端。所有校验 fail-closed：YAML 缺键、C+S0 配方漂移、combo 档漂移、
push/力推键面漂移或混入 base、击球组 17/7/5/5/10 被动、软惩罚被叠加/减负、非法
pod/gpu、run_name 重复、闸门未开渲染被锁臂，都直接拒绝。

3 条 combo 臂（Franco 拍板：把已证明有用的升级一次性合体 go for a try，07-23 审查
变更已并入；父本只用 W；对照不重复买＝矩阵 w_c_s0）：
- combo_fresh：C+S0 + action_rate -0.2（jiayi V14 关节抽动=现役 -0.1 太小的证据）+
  全程高摩擦 静[1.0,1.6]/动[0.8,1.2] + 凹凸地形 2-6 cm + mjlab 档①三项（落地罚
  -3e-3 @300 N、抬脚罚 -0.01 @0.15 m【凹凸逼抬腿,只此臂】、二阶平滑 -0.05【07-22
  已接线 e995b5d5,action_acc 闸门已开】）+ qbar barrier -0.65/margin_frac 0.08 + 速度推
  ±0.35（w_p035 键面）+ 同冲量力推 68 N x 0.30 s（w_f035 键面，两组事件并存），
  fresh-from-random 从零 20001（rough 铁律）；
- combo_resume：同 combo_fresh 去掉凹凸与抬脚罚（平地谱系），从 W model_6700 续训
  13301；
- combo_franco：同 combo_resume + 反手动作（motion_file_2）换 franco_bh_loop_b
  （151f schema-2）——franco_contract 闸门锁死，等 franco_pipeline_20260722 交付
  grip 标定 bake + 锚入册 + 按族重绑题库三前置；Franco：换动作是全面升级的优先
  消融，launch_order 排最前（闸门未开跳过不阻塞）。

reward 比值守卫（Franco 强调）：击球组 17/7/5/5/10 一动不动（前三项钉在 W 配方串，
后两项禁止任何 override＝源码默认）；新增负项（action_rate/action_acc 连续 +
落地罚/抬脚罚/barrier 稀疏）每步合计量级 ≲ 击球收入 1/3（probe/首里程碑 tensorboard
实测核对）；软惩罚组不再叠加（penlight 教训）；速度推/力推是事件不是 reward。

远端命令不使用 scripts/launch_kit_training_locked.sh（其 180 s stale 门杀死过
v8/v9），改用 /workspace/bin/kit_boot_lock.sh + setsid nohup + 事后 artifact 验证。
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_QUEUE = Path("configs/phase1_integrated_upgrade_wave_20260723.yaml")
EXPECTED_QUEUE_ID = "phase1_integrated_upgrade_wave_20260723"
EXPECTED_NAMESPACE = "/workspace/codexschema/phase1_integrated_upgrade_wave_20260723"
EXPECTED_CHECKOUT = "/workspace/codexschema/nohope_iu_20260723"
FRANCO_MOTION_PLACEHOLDER = "PENDING_FRANCO_PIPELINE_DELIVERY"
KIT_BOOT_LOCK = "/workspace/bin/kit_boot_lock.sh"
PARENT_ITERATION = 6700
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
JOB_ID = re.compile(r"^(combo_fresh|combo_resume|combo_franco)$")
HYDRA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

ARM_ORDER = ("combo_fresh", "combo_resume", "combo_franco")
PARENT_ORDER = ("W",)
VALID_GPUS = (0, 1, 2)
# 同卡在跑 compute 进程必须 < 4 才允许发射（逐字继承矩阵/intel/CGF wave 预检）。
MAX_PROCS_PER_GPU = 4

# 人话：空槽即填的冻结顺序（Franco 07-23 拍板）——combo_franco 最先（换动作是全面
# 升级的优先消融）→ combo_resume → combo_fresh。franco 闸门未开时跳过不阻塞。
LAUNCH_ORDER = ("combo_franco", "combo_resume", "combo_fresh")
FRESH_ARMS = ("combo_fresh",)
FRANCO_ARMS = ("combo_franco",)
# 这些闸门任何一个是 false 时全部三臂拒渲染（三臂全带对应键面）。
ALL_ARM_GATES = (
    "groundfoot_contract", "push_contract", "force_push_contract",
    "qbar_contract", "action_acc_contract",
)

EXPECTED_PARENT_SHA256 = {
    "W": "2caab3dde3a0ac6c051ff8ac65385a641cac152aa3f84b640126b5ed7b96fcce",
}
EXPECTED_PARENT_RECIPES = {
    "W": [
        "task.rewards.racket_position_weight=17.0",
        "task.rewards.racket_velocity_weight=7.0",
        "task.rewards.racket_normal_weight=5.0",
        "task.rewards.foot_orientation_weight=-0.3",
        "task.rewards.prestrike_upright_weight=-1.0",
        "++task.rewards.free_non_striking_arm_mimic=true",
    ],
}

# 逐字照抄矩阵 C 档骨架，首行剂量 -0.2（jiayi V14 关节抽动 = 现役 -0.1 太小的证据）。
EXPECTED_TEMPORAL_IU = [
    "task.rewards.action_rate_weight=-0.2",
    "++task.rewards.processed_qdes_slew_hinge_weight=0.0",
    "++task.rewards.processed_qdes_slew_hinge_margin=0.85",
    "++task.rewards.processed_qdes_slew_hinge_recovery_start_s=0.2",
    "++task.rewards.processed_qdes_slew_hinge_recovery_end_s=1.55",
]

# 逐字照抄矩阵 S0 档（三个稳定机制 weight=0，参数键全显式 => 探针强制开）。
EXPECTED_STABILITY_S0 = [
    "++task.rewards.post_swing_settle_debt_weight=0.0",
    "++task.rewards.post_swing_settle_base_lin_margin_mps=0.3",
    "++task.rewards.post_swing_settle_base_lin_scale_mps=0.2",
    "++task.rewards.post_swing_settle_base_ang_margin_radps=0.5",
    "++task.rewards.post_swing_settle_base_ang_scale_radps=0.3",
    "++task.rewards.post_swing_settle_tilt_margin_rad=0.1",
    "++task.rewards.post_swing_settle_tilt_scale_rad=0.1",
    "++task.rewards.post_swing_settle_nominal_root_z_m=1.0684",
    "++task.rewards.post_swing_settle_root_height_deadband_m=0.05",
    "++task.rewards.post_swing_settle_root_height_scale_m=0.05",
    "++task.rewards.post_swing_settle_foot_slip_margin_mps=0.05",
    "++task.rewards.post_swing_settle_foot_slip_scale_mps=0.1",
    "++task.rewards.post_swing_settle_recovery_start_s=0.2",
    "++task.rewards.post_swing_settle_recovery_end_s=1.55",
    "++task.rewards.lower_body_stability_bundle_weight=0.0",
    "++task.rewards.lower_body_stability_min_stance_width_m=0.22",
    "++task.rewards.lower_body_stability_stance_scale_m=0.05",
    "++task.rewards.lower_body_stability_leg_velocity_margin_radps=1.0",
    "++task.rewards.lower_body_stability_leg_velocity_scale_radps=0.5",
    "++task.rewards.lower_body_stability_support_pre_s=0.3",
    "++task.rewards.lower_body_stability_support_post_s=0.4",
    "++task.rewards.lower_body_pose_imitation_weight=0.0",
    "++task.rewards.lower_body_pose_imitation_std=0.35",
    "++task.rewards.lower_body_pose_imitation_support_pre_s=0.3",
    "++task.rewards.lower_body_pose_imitation_support_post_s=0.4",
]

# grip：全程高摩擦（Franco 07-23 变更①：0.6 太低——摩擦当已知量拉高，失稳轴交给
# push/凹凸）。
GRIP_STATIC_RANGE = "++task.plant.robot_material_static_friction_range=[1.0,1.6]"
GRIP_DYNAMIC_RANGE = "++task.plant.robot_material_dynamic_friction_range=[0.8,1.2]"
# rough：随机凹凸地形高度场 2-6 cm（CGF rough 档逐字，只进 combo_fresh）。
ROUGH_HEIGHT_RANGE = "++task.plant.terrain_rough_height_range=[0.02,0.06]"
# footrw：mjlab 落地冲击（无量纲超阈倍数，mjlab -1e-5/N x 300 N 等效剂量 -3e-3）。
FOOTRW_WEIGHT = "++task.rewards.foot_soft_landing_weight=-0.003"
FOOTRW_THRESHOLD = "++task.rewards.foot_soft_landing_force_threshold_n=300.0"
# foot_clearance：抬脚罚（mjlab 档②配对项，只进凹凸臂——凹凸地面就是为了逼抬腿，
# 落地罚+抬脚罚成对；target 0.15 m = hope_rewards 注释"真要它抬腿跨步"档）。
FOOTCL_WEIGHT = "++task.rewards.foot_clearance_weight=-0.01"
FOOTCL_TARGET = "++task.rewards.foot_clearance_target_m=0.15"
# action_acc：mjlab 档①第三项（二阶平滑；07-22 已接线合入 main e995b5d5，
# action_acc_contract 闸门已开；剂量 -0.05 = action_rate 的 1/4，落在采纳文档
# "1/5~1/2 先小"带内）。
ACTION_ACC = "++task.rewards.action_acc_weight=-0.05"
# qbar：全关节 q_des 限位 barrier（我们 V14 验证档；margin_frac 是行程比例不是 rad；
# 稀疏项——贴限位才付费，常态 0；与 action_rate=-0.2 并存，intel 单变量互斥不适用）。
QBAR_WEIGHT = "++task.rewards.qdes_limit_barrier_weight=-0.65"
QBAR_MARGIN = "++task.rewards.qdes_limit_barrier_margin_frac=0.08"
# 速度推：w_p035 臂键面逐字（configs/phase1_push_robustness_20260721.yaml push.p035）。
PUSH_P035 = [
    "++task.push.enable=true",
    "++task.push.interval_range_s=[5.0,15.0]",
    "++task.push.vel_xy_mps=0.35",
    "++task.push.ang_vel_radps=0.0",
    "++task.push.ang_axes=none",
]
# 力推：w_f035 臂键面逐字（同冲量对表 p035：68.0 N x 0.30 s @ pelvis link 原点）。
FORCE_PUSH_F035 = [
    "++task.force_push.enable=true",
    "++task.force_push.interval_range_s=[5.0,15.0]",
    "++task.force_push.force_n=68.0",
    "++task.force_push.duration_s=0.3",
]

EXTRA_RESUME = [
    GRIP_STATIC_RANGE, GRIP_DYNAMIC_RANGE,
    FOOTRW_WEIGHT, FOOTRW_THRESHOLD,
    ACTION_ACC,
    QBAR_WEIGHT, QBAR_MARGIN,
    *PUSH_P035,
    *FORCE_PUSH_F035,
]
EXTRA_FRESH = [
    GRIP_STATIC_RANGE, GRIP_DYNAMIC_RANGE, ROUGH_HEIGHT_RANGE,
    FOOTRW_WEIGHT, FOOTRW_THRESHOLD,
    FOOTCL_WEIGHT, FOOTCL_TARGET,
    ACTION_ACC,
    QBAR_WEIGHT, QBAR_MARGIN,
    *PUSH_P035,
    *FORCE_PUSH_F035,
]

EXPECTED_TEMPORAL = {arm: EXPECTED_TEMPORAL_IU for arm in ARM_ORDER}
EXPECTED_STABILITY = {arm: EXPECTED_STABILITY_S0 for arm in ARM_ORDER}
EXPECTED_EXTRA = {
    "combo_fresh": EXTRA_FRESH,
    "combo_resume": EXTRA_RESUME,
    "combo_franco": EXTRA_RESUME,
}
EXPECTED_MOTION_ASSET = {
    "combo_fresh": "default",
    "combo_resume": "default",
    "combo_franco": "franco",
}

GROUNDFOOT_CLI_KEYS = [
    "task.plant.robot_material_static_friction_range",
    "task.plant.robot_material_dynamic_friction_range",
    "task.plant.terrain_rough_height_range",
    "task.rewards.foot_soft_landing_weight",
    "task.rewards.foot_soft_landing_force_threshold_n",
    "task.rewards.foot_clearance_weight",
    "task.rewards.foot_clearance_target_m",
]
PUSH_CLI_KEYS = [
    "task.push.enable",
    "task.push.interval_range_s",
    "task.push.vel_xy_mps",
    "task.push.ang_vel_radps",
    "task.push.ang_axes",
]
FORCE_PUSH_CLI_KEYS = [
    "task.force_push.enable",
    "task.force_push.interval_range_s",
    "task.force_push.force_n",
    "task.force_push.duration_s",
]
QBAR_CLI_KEYS = [
    "task.rewards.qdes_limit_barrier_weight",
    "task.rewards.qdes_limit_barrier_margin_frac",
]
ACTION_ACC_CLI_KEYS = [
    "task.rewards.action_acc_weight",
]
GATE_EXPECTED_KEYS = {
    "groundfoot_contract": GROUNDFOOT_CLI_KEYS,
    "push_contract": PUSH_CLI_KEYS,
    "force_push_contract": FORCE_PUSH_CLI_KEYS,
    "qbar_contract": QBAR_CLI_KEYS,
    "action_acc_contract": ACTION_ACC_CLI_KEYS,
}
# 键面纪律：每臂允许出现的"新键面"（plant 新键 + 脚部/平滑/barrier reward 键 + 推撞键）。
NEW_SURFACE_PREFIXES = (
    "task.plant.robot_material_", "task.plant.terrain_rough_",
    "task.plant.ground_", "task.plant.passive_damping_fold",
    "task.rewards.foot_soft_landing", "task.rewards.foot_clearance",
    "task.rewards.pre_strike_foot_slip_weight",
    "task.rewards.foot_slip_sq_weight", "task.rewards.foot_drag_weight",
    "task.rewards.action_acc", "task.rewards.qdes_limit_barrier",
    "task.push.", "task.force_push.",
)
_SURFACE_RESUME = {
    "task.plant.robot_material_static_friction_range",
    "task.plant.robot_material_dynamic_friction_range",
    "task.rewards.foot_soft_landing_weight",
    "task.rewards.foot_soft_landing_force_threshold_n",
    "task.rewards.action_acc_weight",
    "task.rewards.qdes_limit_barrier_weight",
    "task.rewards.qdes_limit_barrier_margin_frac",
    "task.push.enable", "task.push.interval_range_s",
    "task.push.vel_xy_mps", "task.push.ang_vel_radps", "task.push.ang_axes",
    "task.force_push.enable", "task.force_push.interval_range_s",
    "task.force_push.force_n", "task.force_push.duration_s",
}
EXPECTED_NEW_SURFACE = {
    "combo_fresh": _SURFACE_RESUME | {
        "task.plant.terrain_rough_height_range",
        "task.rewards.foot_clearance_weight",
        "task.rewards.foot_clearance_target_m",
    },
    "combo_resume": _SURFACE_RESUME,
    "combo_franco": _SURFACE_RESUME,
}

# 推撞键面语义双保险（冻结串改了也过不了这里，反之亦然）。
EXPECTED_PUSH_FACE = {
    "task.push.enable": "true",
    "task.push.interval_range_s": "[5.0,15.0]",
    "task.push.vel_xy_mps": "0.35",
    "task.push.ang_vel_radps": "0.0",
    "task.push.ang_axes": "none",
}
EXPECTED_FORCE_PUSH_FACE = {
    "task.force_push.enable": "true",
    "task.force_push.interval_range_s": "[5.0,15.0]",
    "task.force_push.force_n": "68.0",
    "task.force_push.duration_s": "0.3",
}

# reward 比值守卫：击球组 17/7/5/5/10 一动不动。前三项显式钉在 W 配方串；后两项
# （strike_success 5 / progress 10）禁止任何 override＝源码默认。
FROZEN_HITTING_DEFAULT_PREFIXES = (
    "task.rewards.racket_strike_success", "task.rewards.racket_progress",
)
EXPECTED_BUDGET_HITTING = {
    "racket_position_weight": 17.0,
    "racket_velocity_weight": 7.0,
    "racket_normal_weight": 5.0,
    "racket_strike_success_weight_default": 5.0,
    "racket_progress_weight_default": 10.0,
}
EXPECTED_BUDGET_NEGATIVE = {
    "action_rate_weight": -0.2,
    "action_acc_weight": -0.05,
    "foot_soft_landing_weight": -0.003,
    "foot_clearance_weight_fresh_only": -0.01,
    "qdes_limit_barrier_weight": -0.65,
}
# 软惩罚组不再叠加：全臂全额 -0.4/-0.3/-1.0，蹭滑/拖脚/挥拍前脚滑键一个不带。
SOFT_PENALTY_FULL_DOSE = {
    "task.rewards.racket_face_conditional_guidance_weight": "-0.4",
    "task.rewards.foot_orientation_weight": "-0.3",
    "task.rewards.prestrike_upright_weight": "-1.0",
}
FORBIDDEN_SOFT_STACK_KEYS = (
    "task.rewards.pre_strike_foot_slip_weight",
    "task.rewards.foot_slip_sq_weight",
    "task.rewards.foot_drag_weight",
)
FACE_GUIDANCE_BASE = "++task.rewards.racket_face_conditional_guidance_weight=-0.4"

EXPECTED_PODS = {
    "pod1": ("162.43.172.171", 18333),
    "pod2": ("162.43.172.181", 13146),
}

EXPECTED_CONTROLS = {
    "baseline_queue_id": "phase1_balance_temporal_matrix_20260720",
    "baseline_jobs": ["w_c_s0"],
    "baseline_run_names": ["p1btm_w_c_s0_seed3_20260720"],
}

EXPECTED_PROBE_BUDGET = {
    "num_envs": 4096, "num_steps_per_env": 24,
    "max_iterations": 2, "save_interval": 1,
}
EXPECTED_SCIENCE_BUDGET = {
    "num_envs": 4096, "num_steps_per_env": 24,
    "max_iterations": 13301, "save_interval": 100,
    "milestone_offsets_from_parent": [200, 500, 1000, 2000, 4000, 6600, 10000, 13300],
    "absolute_milestones": [6900, 7200, 7700, 8700, 10700, 13300, 16700, 20000],
}
EXPECTED_SCIENCE_FRESH_BUDGET = {
    "num_envs": 4096, "num_steps_per_env": 24,
    "max_iterations": 20001, "save_interval": 100,
    "absolute_milestones": [200, 500, 1000, 2000, 4000, 10700, 20000],
}
EXPECTED_WATCHDOG = {
    "boot_stall_timeout_s": 1800,
    "post_first_iteration_stall_timeout_s": 900,
    "retry_policy": "one_verbatim_retry_suffix_r2",
}

EXPECTED_STATIC_ASSETS = {
    "a3_runtime_asset_root": "/workspace/codexschema/nohope_balance_action_slew_20260720/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
    "preconverted_a3_usd": "/workspace/codexschema/simple_half_second_sprint_20260718/assets/a3_preconverted_usd/model.usd",
    "motion_forehand": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_forehand_v4rg_cal.npz",
    "motion_backhand": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_backhand_v4rg_cal.npz",
    "training_question_bank": "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_bank_rebind_v2/s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz",
}
FRANCO_MOTION_ASSET_KEY = "motion_backhand_franco"

# 续训臂 science/probe 命令必须含的键（值另行断言）；fresh 臂（combo_fresh）绝不携带
# 任何 checkpoint 键。
REQUIRED_RESUME_KEYS = (
    "checkpoint_path", "checkpoint_tolerant", "checkpoint_allow_missing_contract",
    "checkpoint_allow_contract_mismatch", "seed", "num_envs",
    "algo.runner.num_steps_per_env", "max_iterations", "algo.runner.save_interval",
    "run_name", "device", "logger",
)
REQUIRED_FRESH_KEYS = (
    "seed", "num_envs", "algo.runner.num_steps_per_env", "max_iterations",
    "algo.runner.save_interval", "run_name", "device", "logger",
)
CHECKPOINT_KEYS = (
    "checkpoint_path", "checkpoint_tolerant", "checkpoint_allow_missing_contract",
    "checkpoint_allow_contract_mismatch",
)


class QueueError(RuntimeError):
    """队列配置或渲染请求不安全，拒绝执行。"""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise QueueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise QueueError(f"{label} must be a list")
    return value


def _text(value: Any, label: str, *, safe_id: bool = False) -> str:
    if not isinstance(value, str) or not value or any(ord(ch) < 32 for ch in value):
        raise QueueError(f"{label} must be non-empty printable text")
    if safe_id and not SAFE_ID.fullmatch(value):
        raise QueueError(f"{label} must be a safe identifier")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise QueueError(f"{label} keys differ: missing={missing}, extra={extra}")


def _remote_path(value: Any, label: str) -> str:
    raw = _text(value, label)
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts or raw != str(path):
        raise QueueError(f"{label} must be a normalized absolute POSIX path")
    return raw


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise QueueError(f"{label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise QueueError(f"{label} must be a finite number")
    return result


def _override_key(argument: Any, label: str) -> str:
    raw = _text(argument, label)
    if "=" not in raw:
        raise QueueError(f"{label} must be a Hydra key=value override")
    key = raw.split("=", 1)[0].lstrip("+")
    if not HYDRA_KEY.fullmatch(key):
        raise QueueError(f"{label} has invalid Hydra key {key!r}")
    return key


def _override_map(arguments: Sequence[Any], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, argument in enumerate(arguments):
        raw = _text(argument, f"{label}[{index}]")
        key = _override_key(raw, f"{label}[{index}]")
        if key in result:
            raise QueueError(f"{label} sets Hydra key {key!r} more than once")
        result[key] = raw.split("=", 1)[1]
    return result


def _override_value(overrides: Sequence[str], key: str, label: str) -> float:
    for raw in overrides:
        if raw.lstrip("+").split("=", 1)[0] == key:
            return _finite_number(
                yaml.safe_load(raw.split("=", 1)[1]), f"{label}.{key}"
            )
    raise QueueError(f"{label} is missing required Hydra key {key}")


def _validate_commit_field(value: Any) -> str:
    commit = _text(value, "source.commit")
    if not COMMIT.fullmatch(commit) or commit == "0" * 40:
        raise QueueError(
            "source.commit must be a pinned 40-hex commit (this wave pins the "
            "prereg-day origin/main HEAD; no placeholder mode)"
        )
    return commit


def _franco_motion_is_placeholder(queue: Mapping[str, Any]) -> bool:
    return queue["assets"][FRANCO_MOTION_ASSET_KEY] == FRANCO_MOTION_PLACEHOLDER


def _validate_target(queue: Mapping[str, Any], pod: Any, gpu: Any) -> tuple[str, int]:
    """校验渲染时注入的目标卡：pod 必须是 pod1|pod2，gpu 必须是 0|1|2。"""

    if pod not in queue["pods"]:
        raise QueueError(
            f"--pod must be one of {sorted(queue['pods'])}, got {pod!r}"
        )
    if isinstance(gpu, str):
        if not gpu.isdigit():
            raise QueueError(f"--gpu must be one of {list(VALID_GPUS)}, got {gpu!r}")
        gpu = int(gpu)
    if type(gpu) is not int or gpu not in VALID_GPUS:
        raise QueueError(f"--gpu must be one of {list(VALID_GPUS)}, got {gpu!r}")
    if gpu not in queue["pods"][pod]["gpus"]:
        raise QueueError(f"gpu {gpu} is not offered by {pod}")
    return str(pod), gpu


def _validate_parent(name: str, parent: dict[str, Any]) -> None:
    _exact_keys(
        parent,
        {
            "human_name", "checkpoint_iteration", "checkpoint_path",
            "checkpoint_sha256", "hard_contract_path", "transfer_mode",
            "checkpoint_tolerant", "checkpoint_allow_missing_contract",
            "checkpoint_allow_contract_mismatch",
            "descendant_formal_exact_eligible", "recipe_overrides",
        },
        f"parents.{name}",
    )
    _text(parent["human_name"], f"parents.{name}.human_name")
    if parent["checkpoint_iteration"] != PARENT_ITERATION:
        raise QueueError(f"parents.{name} must use model_6700")
    checkpoint = _remote_path(parent["checkpoint_path"], f"parents.{name}.checkpoint_path")
    if not checkpoint.endswith("/model_6700.pt"):
        raise QueueError(f"parents.{name}.checkpoint_path must end in model_6700.pt")
    contract = _remote_path(parent["hard_contract_path"], f"parents.{name}.hard_contract_path")
    if contract != str(PurePosixPath(checkpoint).parent / "params" / "training_contract.json"):
        raise QueueError(f"parents.{name} hard contract is not adjacent to the checkpoint")
    sha = parent["checkpoint_sha256"]
    if sha is not None:
        if type(sha) is not str or not SHA256.fullmatch(sha):
            raise QueueError(f"parents.{name}.checkpoint_sha256 must be 64-hex or null")
        if sha != EXPECTED_PARENT_SHA256[name]:
            raise QueueError(
                f"parents.{name}.checkpoint_sha256 differs from the manifest-verified value"
            )
    if parent["transfer_mode"] != "full_policy_value_optimizer_normalizer":
        raise QueueError(f"parents.{name} must request full-state resume")
    for key, expected in {
        "checkpoint_tolerant": False,
        "checkpoint_allow_missing_contract": False,
        "checkpoint_allow_contract_mismatch": True,
        "descendant_formal_exact_eligible": False,
    }.items():
        if parent[key] is not expected:
            raise QueueError(f"parents.{name}.{key} must be {expected!r}")
    recipes = _list(parent["recipe_overrides"], f"parents.{name}.recipe_overrides")
    if recipes != EXPECTED_PARENT_RECIPES[name]:
        raise QueueError(f"parents.{name}.recipe_overrides changed from Wave A verbatim copy")


def _validate_mechanism(name: str, mechanism: dict[str, Any]) -> None:
    label = f"mechanisms.arms.{name}"
    _exact_keys(
        mechanism,
        {
            "human_name", "motion_backhand_asset",
            "temporal_overrides", "stability_overrides", "extra_overrides",
        },
        label,
    )
    _text(mechanism["human_name"], f"{label}.human_name")
    if mechanism["motion_backhand_asset"] != EXPECTED_MOTION_ASSET[name]:
        raise QueueError(
            f"{label}.motion_backhand_asset must be "
            f"{EXPECTED_MOTION_ASSET[name]!r} (only combo_franco swaps the "
            "backhand motion to franco_bh_loop_b)"
        )
    temporal = _list(mechanism["temporal_overrides"], f"{label}.temporal_overrides")
    if temporal != EXPECTED_TEMPORAL[name]:
        raise QueueError(f"{label}.temporal_overrides drifted from frozen design")
    stability = _list(mechanism["stability_overrides"], f"{label}.stability_overrides")
    if stability != EXPECTED_STABILITY[name]:
        raise QueueError(f"{label}.stability_overrides drifted from frozen design")
    extra = _list(mechanism["extra_overrides"], f"{label}.extra_overrides")
    if extra != EXPECTED_EXTRA[name]:
        raise QueueError(f"{label}.extra_overrides drifted from frozen design")

    # 语义双保险（冻结串改了也过不了这里，反之亦然）。
    action_rate = _override_value(temporal, "task.rewards.action_rate_weight", label)
    if action_rate != -0.2:
        raise QueueError(
            f"{label} action_rate must be exactly -0.2 (jiayi V14 chatter evidence dose)"
        )
    hinge = _override_value(
        temporal, "task.rewards.processed_qdes_slew_hinge_weight", label
    )
    if hinge != 0.0:
        raise QueueError(f"{label} slew hinge must stay 0 (C recipe verbatim)")
    margin = _override_value(
        temporal, "task.rewards.processed_qdes_slew_hinge_margin", label
    )
    if margin != 0.85:
        raise QueueError(f"{label} hinge margin must stay 0.85 (matrix C verbatim)")
    for key in (
        "task.rewards.post_swing_settle_debt_weight",
        "task.rewards.lower_body_stability_bundle_weight",
        "task.rewards.lower_body_pose_imitation_weight",
    ):
        if _override_value(stability, key, label) != 0.0:
            raise QueueError(f"{label} must keep all three mechanism weights at 0 (S0)")
    weight = _override_value(extra, "task.rewards.foot_soft_landing_weight", label)
    threshold = _override_value(
        extra, "task.rewards.foot_soft_landing_force_threshold_n", label
    )
    # 量纲铁律：输出是无量纲超阈倍数（threshold 归一、单脚封顶 3），mjlab -1e-5/N
    # 的等效剂量 = -1e-5 x threshold。别抄 -0.1（33 倍 mjlab 剂量）也别抄 -1e-4。
    if weight != -0.003 or threshold != 300.0:
        raise QueueError(
            f"{label} must be exactly weight -0.003 (mjlab-equivalent dose at "
            "threshold 300 N) / threshold 300.0"
        )
    if name in FRESH_ARMS:
        clearance = _override_value(extra, "task.rewards.foot_clearance_weight", label)
        target = _override_value(extra, "task.rewards.foot_clearance_target_m", label)
        if clearance != -0.01 or target != 0.15:
            raise QueueError(
                f"{label} foot clearance must be exactly -0.01 @ target 0.15 m "
                "(rough terrain exists to force leg lift; landing + clearance are a pair)"
            )
    action_acc = _override_value(extra, "task.rewards.action_acc_weight", label)
    if action_acc != -0.05:
        raise QueueError(
            f"{label} action_acc must be exactly -0.05 (action_rate/4, inside the "
            "mjlab-adoption 1/5~1/2 starting band; second-difference magnitudes are larger)"
        )
    qbar_weight = _override_value(extra, "task.rewards.qdes_limit_barrier_weight", label)
    qbar_margin = _override_value(
        extra, "task.rewards.qdes_limit_barrier_margin_frac", label
    )
    if qbar_weight != -0.65 or qbar_margin != 0.08:
        raise QueueError(
            f"{label} qbar barrier must be exactly -0.65 @ margin_frac 0.08 "
            "(V14-verified dose; margin_frac is a travel FRACTION, not radians)"
        )
    vel_xy = _override_value(extra, "task.push.vel_xy_mps", label)
    ang_vel = _override_value(extra, "task.push.ang_vel_radps", label)
    if vel_xy != 0.35 or ang_vel != 0.0:
        raise QueueError(
            f"{label} velocity push must be exactly the w_p035 face (vel 0.35, ang 0.0)"
        )
    force_n = _override_value(extra, "task.force_push.force_n", label)
    duration = _override_value(extra, "task.force_push.duration_s", label)
    if force_n != 68.0 or duration != 0.3:
        raise QueueError(
            f"{label} force push must be exactly the w_f035 face (68.0 N x 0.30 s, "
            "matched impulse to p035)"
        )
    for raw in stability:
        if raw.lstrip("+").split("=", 1)[0].endswith("_probe_weight"):
            raise QueueError(
                f"{label} must not pass *_probe_weight: train.py has no such CLI key "
                "(probes are auto-forced to 1.0 by the explicit weight keys)"
            )


def _validate_wiring_contract(
    contract: dict[str, Any], label: str, expected_keys: list[str]
) -> None:
    """wiring 闸门通用形状：确认布尔 + 冻结键名 + 人话说明。"""

    _exact_keys(
        contract,
        {"wiring_confirmed", "expected_cli_keys", "wiring_note"},
        label,
    )
    if not isinstance(contract["wiring_confirmed"], bool):
        raise QueueError(f"{label}.wiring_confirmed must be a bool")
    if contract["expected_cli_keys"] != expected_keys:
        raise QueueError(
            f"{label}.expected_cli_keys must be exactly the frozen key list"
        )
    _text(contract["wiring_note"], f"{label}.wiring_note")


def _validate_franco_contract(contract: dict[str, Any]) -> None:
    label = "franco_contract"
    _exact_keys(
        contract,
        {
            "wiring_confirmed", "motion_asset_key", "motion_human_name",
            "unlock_prerequisites", "wiring_note",
        },
        label,
    )
    if not isinstance(contract["wiring_confirmed"], bool):
        raise QueueError(f"{label}.wiring_confirmed must be a bool")
    if contract["motion_asset_key"] != FRANCO_MOTION_ASSET_KEY:
        raise QueueError(
            f"{label}.motion_asset_key must be {FRANCO_MOTION_ASSET_KEY!r}"
        )
    _text(contract["motion_human_name"], f"{label}.motion_human_name")
    prerequisites = _list(contract["unlock_prerequisites"], f"{label}.unlock_prerequisites")
    if len(prerequisites) != 3:
        raise QueueError(
            f"{label}.unlock_prerequisites must list exactly the three "
            "franco_pipeline_20260722 deliveries (grip calibration bake, anchor "
            "registration, family question-bank rebind)"
        )
    for index, item in enumerate(prerequisites):
        _text(item, f"{label}.unlock_prerequisites[{index}]")
    _text(contract["wiring_note"], f"{label}.wiring_note")


def _validate_reward_budget_contract(contract: dict[str, Any]) -> None:
    label = "reward_budget_contract"
    _exact_keys(
        contract,
        {
            "hitting_group_frozen", "new_negative_terms", "budget_rule",
            "soft_penalty_rule", "push_rule",
        },
        label,
    )
    hitting = _mapping(contract["hitting_group_frozen"], f"{label}.hitting_group_frozen")
    _exact_keys(hitting, set(EXPECTED_BUDGET_HITTING), f"{label}.hitting_group_frozen")
    for key, expected in EXPECTED_BUDGET_HITTING.items():
        if _finite_number(hitting[key], f"{label}.hitting_group_frozen.{key}") != expected:
            raise QueueError(
                f"{label}.hitting_group_frozen.{key} must be exactly {expected} "
                "(Franco: the 17/7/5/5/10 hitting group does not move)"
            )
    negative = _mapping(contract["new_negative_terms"], f"{label}.new_negative_terms")
    _exact_keys(negative, set(EXPECTED_BUDGET_NEGATIVE), f"{label}.new_negative_terms")
    for key, expected in EXPECTED_BUDGET_NEGATIVE.items():
        if _finite_number(negative[key], f"{label}.new_negative_terms.{key}") != expected:
            raise QueueError(f"{label}.new_negative_terms.{key} must be exactly {expected}")
    for key in ("budget_rule", "soft_penalty_rule", "push_rule"):
        _text(contract[key], f"{label}.{key}")


def _require_gates_open(queue: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """渲染闸门：五个 wiring 合同锁全部臂；franco 闸门 + 资产占位符锁 combo_franco。"""

    for gate in ALL_ARM_GATES:
        if queue[gate]["wiring_confirmed"] is not True:
            raise QueueError(
                f"{gate}.wiring_confirmed=false: every combo arm carries these keys "
                f"({', '.join(GATE_EXPECTED_KEYS[gate])}), refusing to render any "
                "command until the wiring is merged/verified at the pinned commit"
            )
    if job["id"] in FRANCO_ARMS:
        if queue["franco_contract"]["wiring_confirmed"] is not True:
            raise QueueError(
                "combo_franco is gated on the franco_pipeline_20260722 deliveries "
                "(grip calibration bake + anchor registration + family question-bank "
                "rebind); franco_contract.wiring_confirmed=false, refusing to render "
                "— the other two arms are unaffected"
            )
        if _franco_motion_is_placeholder(queue):
            raise QueueError(
                "assets.motion_backhand_franco is still the placeholder "
                f"{FRANCO_MOTION_PLACEHOLDER!r}; pin the delivered franco_bh_loop_b "
                "npz path (with sha256 on the ledger) before rendering combo_franco"
            )


def _job_is_fresh(job: Mapping[str, Any]) -> bool:
    return job["fresh_from_random"] is True


def _backhand_asset_key(queue: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    level = queue["mechanisms"]["arms"][job["arm"]]
    if level["motion_backhand_asset"] == "franco":
        return FRANCO_MOTION_ASSET_KEY
    return "motion_backhand"


def _validate_jobs(queue: dict[str, Any]) -> None:
    jobs = _list(queue["jobs"], "jobs")
    if len(jobs) != 3:
        raise QueueError("the queue must contain exactly 3 jobs")
    ids: set[str] = set()
    names: set[str] = set()
    dirs: set[str] = set()
    observed: list[str] = []
    for index, raw_job in enumerate(jobs):
        job = _mapping(raw_job, f"jobs[{index}]")
        _exact_keys(
            job,
            {"id", "parent", "fresh_from_random", "arm", "run_name", "run_dir"},
            f"jobs[{index}]",
        )
        job_id = _text(job["id"], f"jobs[{index}].id", safe_id=True)
        if JOB_ID.fullmatch(job_id) is None:
            raise QueueError(
                f"jobs[{index}].id must be one of {list(ARM_ORDER)}: {job_id!r}"
            )
        if job["arm"] != job_id:
            raise QueueError(f"jobs[{index}].arm must equal its id {job_id!r}")
        if job["parent"] != "W":
            raise QueueError(
                f"jobs[{index}].parent must be W (single-parent wave; combo_fresh "
                "is fresh via fresh_from_random, the W recipe is the design not "
                "the lineage)"
            )
        if not isinstance(job["fresh_from_random"], bool):
            raise QueueError(f"jobs[{index}].fresh_from_random must be a bool")
        if job["fresh_from_random"] != (job_id in FRESH_ARMS):
            raise QueueError(
                f"jobs[{index}].fresh_from_random must be true exactly for "
                f"{list(FRESH_ARMS)} (the rough-terrain combo must never resume a "
                "flat-ground checkpoint; the other two arms must resume W)"
            )
        run_name = _text(job["run_name"], f"jobs[{index}].run_name", safe_id=True)
        expected_name = f"p1iu_{job_id}_seed3_20260723"
        if run_name != expected_name:
            raise QueueError(f"jobs[{index}].run_name must be {expected_name!r}")
        run_dir = _remote_path(job["run_dir"], f"jobs[{index}].run_dir")
        if run_dir != f"{EXPECTED_NAMESPACE}/runs/{job_id}":
            raise QueueError(f"jobs[{index}].run_dir must be <root>/runs/{job_id}")
        if job_id in ids or run_name in names or run_dir in dirs:
            raise QueueError("duplicate job id, run_name, or run_dir")
        ids.add(job_id)
        names.add(run_name)
        dirs.add(run_dir)
        observed.append(job_id)
    if observed != list(ARM_ORDER):
        raise QueueError("jobs must keep the frozen combo_fresh..combo_franco ordering")

    order = _list(queue["launch_order"], "launch_order")
    if order != list(LAUNCH_ORDER):
        raise QueueError(
            "launch_order must keep the frozen combo_franco->combo_resume->"
            "combo_fresh fill order (Franco: the motion swap is the priority "
            "ablation of the integrated upgrade)"
        )
    if set(order) != ids:
        raise QueueError("launch_order must be a permutation of the 3 job ids")


def _validate_controls(controls: dict[str, Any]) -> None:
    _exact_keys(
        controls,
        set(EXPECTED_CONTROLS) | {"human_note"},
        "controls",
    )
    for key, expected in EXPECTED_CONTROLS.items():
        if controls[key] != expected:
            raise QueueError(
                f"controls.{key} must point at the matrix C+S0 baseline {expected!r}"
            )
    _text(controls["human_note"], "controls.human_note")


def _validate_queue(queue: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(
        queue,
        {
            "schema_version", "queue_id", "purpose", "simulation_only",
            "real_robot_authorized", "launch_authorized_by_default",
            "formal_exact_eligible", "evidence_class", "ssh", "pods", "namespace",
            "source", "watchdog", "assets", "controls", "parents", "common",
            "mechanisms", "groundfoot_contract", "push_contract",
            "force_push_contract", "qbar_contract", "action_acc_contract",
            "franco_contract", "reward_budget_contract", "probe_contract",
            "budgets", "launch_order", "jobs",
        },
        "queue",
    )
    if queue["schema_version"] != 1 or queue["queue_id"] != EXPECTED_QUEUE_ID:
        raise QueueError("queue identity/schema changed")
    _text(queue["purpose"], "purpose")
    for key, expected in {
        "simulation_only": True,
        "real_robot_authorized": False,
        "launch_authorized_by_default": False,
        "formal_exact_eligible": False,
    }.items():
        if queue[key] is not expected:
            raise QueueError(f"{key} must be {expected!r}")
    if queue["evidence_class"] != "diagnostic_only_intentional_parent_contract_mismatch":
        raise QueueError("evidence_class must remain diagnostic-only")

    ssh = _mapping(queue["ssh"], "ssh")
    _exact_keys(ssh, {"key"}, "ssh")
    if ssh["key"] != "~/.ssh/id_ed25519_runpod":
        raise QueueError("unexpected SSH key path")
    pods = _mapping(queue["pods"], "pods")
    if set(pods) != set(EXPECTED_PODS):
        raise QueueError("pods must be exactly pod1 and pod2")
    for name, pod in pods.items():
        pod = _mapping(pod, f"pods.{name}")
        _exact_keys(pod, {"host", "port", "gpus"}, f"pods.{name}")
        if (pod["host"], pod["port"]) != EXPECTED_PODS[name] or pod["gpus"] != [0, 1, 2]:
            raise QueueError(f"pods.{name} endpoint/GPU set changed")

    namespace = _mapping(queue["namespace"], "namespace")
    _exact_keys(namespace, {"root", "no_clobber", "automatic_retry"}, "namespace")
    if (
        _remote_path(namespace["root"], "namespace.root") != EXPECTED_NAMESPACE
        or namespace["no_clobber"] is not True
        or namespace["automatic_retry"] is not False
    ):
        raise QueueError("namespace must remain fresh, no-clobber, and no-retry")

    source = _mapping(queue["source"], "source")
    _exact_keys(
        source,
        {
            "checkout", "identity_mode", "commit", "worktree_relative", "python",
            "setup_relative", "trainer_relative", "locked_launcher_relative", "note",
        },
        "source",
    )
    if _remote_path(source["checkout"], "source.checkout") != EXPECTED_CHECKOUT:
        raise QueueError("source checkout changed")
    if source["identity_mode"] != "clean_detached_exact_commit":
        raise QueueError("source identity mode changed")
    _validate_commit_field(source["commit"])
    _remote_path(source["python"], "source.python")
    for key in ("worktree_relative", "setup_relative", "trainer_relative",
                "locked_launcher_relative"):
        value = _text(source[key], f"source.{key}")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value != str(path):
            raise QueueError(f"source.{key} must be a normalized relative path")
    _text(source["note"], "source.note")

    watchdog = _mapping(queue["watchdog"], "watchdog")
    _exact_keys(
        watchdog,
        set(EXPECTED_WATCHDOG) | {"retry_note"},
        "watchdog",
    )
    for key, expected in EXPECTED_WATCHDOG.items():
        if watchdog[key] != expected:
            raise QueueError(f"watchdog.{key} must be {expected!r}")
    _text(watchdog["retry_note"], "watchdog.retry_note")

    assets = _mapping(queue["assets"], "assets")
    _exact_keys(
        assets, set(EXPECTED_STATIC_ASSETS) | {FRANCO_MOTION_ASSET_KEY}, "assets"
    )
    for key, expected in EXPECTED_STATIC_ASSETS.items():
        if _remote_path(assets[key], f"assets.{key}") != expected:
            raise QueueError(f"assets.{key} changed from the frozen verbatim copy")
    franco_asset = _text(assets[FRANCO_MOTION_ASSET_KEY], f"assets.{FRANCO_MOTION_ASSET_KEY}")
    if franco_asset != FRANCO_MOTION_PLACEHOLDER:
        _remote_path(franco_asset, f"assets.{FRANCO_MOTION_ASSET_KEY}")
        if not franco_asset.endswith(".npz"):
            raise QueueError(
                f"assets.{FRANCO_MOTION_ASSET_KEY} must be either the literal "
                f"{FRANCO_MOTION_PLACEHOLDER!r} or an absolute .npz path"
            )

    _validate_controls(_mapping(queue["controls"], "controls"))

    parents = _mapping(queue["parents"], "parents")
    if set(parents) != set(PARENT_ORDER):
        raise QueueError("parents must be exactly W (single-parent wave)")
    for name, parent in parents.items():
        _validate_parent(name, _mapping(parent, f"parents.{name}"))

    common = _mapping(queue["common"], "common")
    _exact_keys(common, {"seed", "planner_revision_override", "base_overrides"}, "common")
    if type(common["seed"]) is not int or common["seed"] != 3:
        raise QueueError("common.seed must be integer 3")
    planner = _text(common["planner_revision_override"], "common.planner_revision_override")
    if _override_key(planner, "common.planner_revision_override") != "task.planner_revision":
        raise QueueError("planner_revision_override key changed")
    base = _list(common["base_overrides"], "common.base_overrides")
    base_map = _override_map(base, "common.base_overrides")
    if base_map.get("logger") != "tensorboard":
        raise QueueError("common.base_overrides must pin logger=tensorboard")
    if base_map.get("task.rewards.joint_velocity_limit_hinge_weight") != "0.0":
        raise QueueError("base recipe must explicitly keep the qdot hinge at zero")
    if base_map.get("task.motion.speed_scale_range") != "[1.0,1.0]":
        raise QueueError("base recipe must pin speed_scale_range=[1.0,1.0]")
    if base_map.get("task.plant.zero_joint_friction") != "true":
        raise QueueError("base recipe must keep the zero-joint-friction plant control")
    if FACE_GUIDANCE_BASE not in base:
        raise QueueError(
            "common.base_overrides must carry the verbatim full-dose "
            f"{FACE_GUIDANCE_BASE!r} line (soft penalties are untouched this wave)"
        )
    # 推撞键面绝不进 base：速度推/力推只允许作为 combo 臂的 extra_overrides 出现。
    forbidden_base = sorted(
        key for key in base_map
        if key.startswith("task.push.") or key.startswith("task.force_push.")
    )
    if forbidden_base:
        raise QueueError(
            f"push/force keys must live in the arm extras, never in base: {forbidden_base}"
        )
    frozen_hitting_base = sorted(
        key for key in base_map
        if any(key.startswith(prefix) for prefix in FROZEN_HITTING_DEFAULT_PREFIXES)
    )
    if frozen_hitting_base:
        raise QueueError(
            "the 17/7/5/5/10 hitting group does not move: strike_success/progress "
            f"overrides are forbidden anywhere, got {frozen_hitting_base}"
        )
    new_surface_base = sorted(
        key for key in base_map
        if key != "task.plant.zero_joint_friction"
        and any(key.startswith(prefix) or key == prefix for prefix in NEW_SURFACE_PREFIXES)
    )
    if new_surface_base:
        raise QueueError(
            f"common.base_overrides owns per-arm new-surface keys: {new_surface_base}"
        )
    per_cell_keys = {
        "task.rewards.action_rate_weight",
        "task.rewards.processed_qdes_slew_hinge_weight",
        "task.rewards.processed_qdes_slew_hinge_margin",
        "task.rewards.processed_qdes_slew_hinge_recovery_start_s",
        "task.rewards.processed_qdes_slew_hinge_recovery_end_s",
        "task.rewards.post_swing_settle_debt_weight",
        "task.rewards.lower_body_stability_bundle_weight",
        "task.rewards.lower_body_pose_imitation_weight",
        "checkpoint_path", "run_name", "seed", "num_envs", "max_iterations",
        "algo.runner.save_interval", "algo.runner.num_steps_per_env", "device",
        "task.rewards.racket_position_weight",
        "task.rewards.racket_velocity_weight",
        "task.rewards.racket_normal_weight",
        "task.rewards.foot_orientation_weight",
        "task.rewards.prestrike_upright_weight",
        "task.rewards.free_non_striking_arm_mimic",
    }
    overlap = set(base_map) & per_cell_keys
    if overlap:
        raise QueueError(f"common.base_overrides owns per-cell keys: {sorted(overlap)}")

    mechanisms = _mapping(queue["mechanisms"], "mechanisms")
    _exact_keys(mechanisms, {"arms"}, "mechanisms")
    arms = _mapping(mechanisms["arms"], "mechanisms.arms")
    if set(arms) != set(ARM_ORDER):
        raise QueueError(
            f"mechanisms.arms must be exactly {list(ARM_ORDER)}"
        )
    for name, mechanism in arms.items():
        _validate_mechanism(name, _mapping(mechanism, f"mechanisms.arms.{name}"))

    for gate, expected_keys in GATE_EXPECTED_KEYS.items():
        _validate_wiring_contract(_mapping(queue[gate], gate), gate, expected_keys)
    _validate_franco_contract(_mapping(queue["franco_contract"], "franco_contract"))
    _validate_reward_budget_contract(
        _mapping(queue["reward_budget_contract"], "reward_budget_contract")
    )

    _mapping(queue["probe_contract"], "probe_contract")

    budgets = _mapping(queue["budgets"], "budgets")
    _exact_keys(budgets, {"probe", "science", "science_fresh"}, "budgets")
    probe = _mapping(budgets["probe"], "budgets.probe")
    _exact_keys(probe, set(EXPECTED_PROBE_BUDGET), "budgets.probe")
    for key, expected in EXPECTED_PROBE_BUDGET.items():
        if _positive_int(probe[key], f"budgets.probe.{key}") != expected:
            raise QueueError(f"budgets.probe.{key} must be {expected}")
    science = _mapping(budgets["science"], "budgets.science")
    _exact_keys(science, set(EXPECTED_SCIENCE_BUDGET), "budgets.science")
    for key, expected in EXPECTED_SCIENCE_BUDGET.items():
        if science[key] != expected:
            raise QueueError(f"budgets.science.{key} must be {expected}")
    offsets = science["milestone_offsets_from_parent"]
    if [PARENT_ITERATION + offset for offset in offsets] != science["absolute_milestones"]:
        raise QueueError("science absolute milestones must equal parent + offsets")
    fresh = _mapping(budgets["science_fresh"], "budgets.science_fresh")
    _exact_keys(fresh, set(EXPECTED_SCIENCE_FRESH_BUDGET), "budgets.science_fresh")
    for key, expected in EXPECTED_SCIENCE_FRESH_BUDGET.items():
        if fresh[key] != expected:
            raise QueueError(f"budgets.science_fresh.{key} must be {expected}")

    _validate_jobs(queue)

    # 每臂两阶段都编译一遍：证明没有重复 Hydra 键、必带键齐全、键面纪律成立。
    for job in queue["jobs"]:
        for stage in ("probe", "science"):
            _training_argv(queue, job, stage)
    return queue


def load_queue(path: Path = DEFAULT_QUEUE) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except FileNotFoundError as exc:
        raise QueueError(f"queue config is missing: {path}") from exc
    except yaml.YAMLError as exc:
        raise QueueError(f"invalid YAML: {exc}") from exc
    return _validate_queue(_mapping(value, "queue"))


def _job_by_id(queue: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    for job in queue["jobs"]:
        if job["id"] == job_id:
            return job
    raise QueueError(f"unknown job id: {job_id!r}")


def _stage_run_dir(queue: Mapping[str, Any], job: Mapping[str, Any], stage: str) -> str:
    if stage == "science":
        return str(job["run_dir"])
    return f"{queue['namespace']['root']}/probes/{job['id']}"


def _stage_run_name(job: Mapping[str, Any], stage: str) -> str:
    if stage == "science":
        return str(job["run_name"])
    return f"p1iu_probe_{job['id']}_seed3_20260723"


def _science_budget(queue: Mapping[str, Any], job: Mapping[str, Any]) -> Mapping[str, Any]:
    return queue["budgets"]["science_fresh" if _job_is_fresh(job) else "science"]


def _training_argv(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str
) -> list[str]:
    if stage not in {"probe", "science"}:
        raise QueueError("stage must be probe or science")
    source = queue["source"]
    workdir = f"{source['checkout']}/{source['worktree_relative']}"
    parent = queue["parents"][job["parent"]]
    level = queue["mechanisms"]["arms"][job["arm"]]
    budget = (
        queue["budgets"]["probe"] if stage == "probe" else _science_budget(queue, job)
    )
    # fresh 臂（combo_fresh）绝不携带任何 checkpoint 键：平地 checkpoint 不能静默上
    # 粗糙地；配方串仍用 W（配方是实验设计，不是谱系）。
    checkpoint_args = (
        []
        if _job_is_fresh(job)
        else [
            f"checkpoint_path={parent['checkpoint_path']}",
            "checkpoint_tolerant=false",
            "checkpoint_allow_missing_contract=false",
            "checkpoint_allow_contract_mismatch=true",
        ]
    )
    backhand = queue["assets"][_backhand_asset_key(queue, job)]
    argv = [
        source["python"],
        f"{workdir}/{source['trainer_relative']}",
        *queue["common"]["base_overrides"],
        queue["common"]["planner_revision_override"],
        *parent["recipe_overrides"],
        *level["temporal_overrides"],
        *level["stability_overrides"],
        *level["extra_overrides"],
        f"motion_file={queue['assets']['motion_forehand']}",
        f"motion_file_2={backhand}",
        f"++task.racket.question_bank={queue['assets']['training_question_bank']}",
        *checkpoint_args,
        f"seed={queue['common']['seed']}",
        f"num_envs={budget['num_envs']}",
        f"algo.runner.num_steps_per_env={budget['num_steps_per_env']}",
        f"max_iterations={budget['max_iterations']}",
        f"algo.runner.save_interval={budget['save_interval']}",
        f"run_name={_stage_run_name(job, stage)}",
        # CUDA_VISIBLE_DEVICES=<gpu> 把物理卡钉死后，进程内只剩一张卡，
        # 所以 device 恒为 cuda:0（Wave A 验证过的组合）。
        "device=cuda:0",
    ]
    compiled = _override_map(argv[2:], f"{job['id']}.{stage}.argv")
    if set(compiled) & {"ros", "deploy", "real_robot", "motion_command"}:
        raise QueueError("real-robot/deploy arguments are forbidden")
    required = REQUIRED_FRESH_KEYS if _job_is_fresh(job) else REQUIRED_RESUME_KEYS
    for key in required:
        if key not in compiled:
            raise QueueError(f"{job['id']}.{stage} command is missing required key {key}")
    if _job_is_fresh(job):
        present = sorted(set(compiled) & set(CHECKPOINT_KEYS))
        if present:
            raise QueueError(
                f"{job['id']} is fresh-from-random; checkpoint keys {present} are "
                "forbidden (a flat-ground checkpoint must never silently resume "
                "onto rough terrain)"
            )
    # 键面纪律：每臂的"新键面"（plant 新键/脚部/平滑/barrier/推撞）必须逐字等于该臂的
    # 冻结集合——多一个少一个都拒绝（缺席 == 字节等价 no-op）。
    observed_surface = {
        key for key in compiled
        if key != "task.plant.zero_joint_friction"
        and any(key.startswith(prefix) or key == prefix for prefix in NEW_SURFACE_PREFIXES)
    }
    if observed_surface != EXPECTED_NEW_SURFACE[job["arm"]]:
        raise QueueError(
            f"{job['id']} new-surface keys must be exactly "
            f"{sorted(EXPECTED_NEW_SURFACE[job['arm']])}, got {sorted(observed_surface)}"
        )
    # 推撞键面语义双保险：w_p035 + w_f035 面逐字（值也钉死，两组事件并存）。
    for key, expected in {**EXPECTED_PUSH_FACE, **EXPECTED_FORCE_PUSH_FACE}.items():
        if compiled.get(key) != expected:
            raise QueueError(
                f"{job['id']} push/force face must be verbatim w_p035+w_f035: "
                f"{key}={expected}, got {compiled.get(key)!r}"
            )
    # reward 比值守卫：击球组 17/7/5/5/10 一动不动。
    if (
        compiled.get("task.rewards.racket_position_weight") != "17.0"
        or compiled.get("task.rewards.racket_velocity_weight") != "7.0"
        or compiled.get("task.rewards.racket_normal_weight") != "5.0"
    ):
        raise QueueError(
            f"{job['id']} must keep the 17/7/5 racket tracking weights untouched"
        )
    frozen_hitting = sorted(
        key for key in compiled
        if any(key.startswith(prefix) for prefix in FROZEN_HITTING_DEFAULT_PREFIXES)
    )
    if frozen_hitting:
        raise QueueError(
            f"{job['id']} must not override strike_success/progress (defaults 5/10 "
            f"are part of the frozen hitting group), got {frozen_hitting}"
        )
    # 软惩罚组不再叠加：全额三键钉死，蹭滑/拖脚/挥拍前脚滑键一个不带。
    for key, expected in SOFT_PENALTY_FULL_DOSE.items():
        if compiled.get(key) != expected:
            raise QueueError(
                f"{job['id']} must keep the full-dose soft penalty {key}={expected}, "
                f"got {compiled.get(key)!r}"
            )
    stacked = sorted(set(compiled) & set(FORBIDDEN_SOFT_STACK_KEYS))
    if stacked:
        raise QueueError(
            f"{job['id']} must not stack extra soft penalties (penlight lesson): "
            f"{stacked}"
        )
    # 动作资产纪律：只有 combo_franco 换反手；另两臂必须用 v4rg 反手。
    expected_backhand = queue["assets"][_backhand_asset_key(queue, job)]
    if compiled.get("motion_file_2") != expected_backhand:
        raise QueueError(f"{job['id']} motion_file_2 must be {expected_backhand!r}")
    if compiled.get("motion_file") != queue["assets"]["motion_forehand"]:
        raise QueueError(f"{job['id']} motion_file must stay the v4rg forehand")
    return argv


def _remote_body(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str, gpu: int
) -> str:
    commit = queue["source"]["commit"]
    _require_gates_open(queue, job)
    source = queue["source"]
    checkout = source["checkout"]
    workdir = f"{checkout}/{source['worktree_relative']}"
    setup = f"{workdir}/{source['setup_relative']}"
    trainer = f"{workdir}/{source['trainer_relative']}"
    parent = queue["parents"][job["parent"]]
    run_dir = _stage_run_dir(queue, job, stage)
    run_parent = str(PurePosixPath(run_dir).parent)
    run_log = f"{run_dir}/run.log"
    argv = _training_argv(queue, job, stage)
    required_files = [
        trainer,
        setup,
        queue["assets"]["preconverted_a3_usd"],
        queue["assets"]["motion_forehand"],
        queue["assets"][_backhand_asset_key(queue, job)],
        queue["assets"]["training_question_bank"],
    ]
    if not _job_is_fresh(job):
        required_files += [parent["checkpoint_path"], parent["hard_contract_path"]]
    file_checks = "\n".join(f"test -f {shlex.quote(path)}" for path in required_files)
    child = (
        f"env CUDA_VISIBLE_DEVICES={gpu} "
        f"HOPE_AGIBOT_A3_USD_PATH={shlex.quote(queue['assets']['preconverted_a3_usd'])} "
        "PYTHONUNBUFFERED=1 PYTHONPATH=\"${HOPE_WBT_PYTHONPATH}\" "
        + shlex.join(argv)
    )
    watchdog = queue["watchdog"]
    return f"""set -euo pipefail
test -d {shlex.quote(checkout)}
test "$(git -C {shlex.quote(checkout)} rev-parse HEAD)" = {shlex.quote(commit)}
test -z "$(git -C {shlex.quote(checkout)} status --porcelain)"
{file_checks}
test -d {shlex.quote(queue['assets']['a3_runtime_asset_root'])}
test -x {shlex.quote(KIT_BOOT_LOCK)}
gpu_output=$(nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader,nounits)
test "$(printf %s "$gpu_output" | sort -u | grep -c . || true)" -lt {MAX_PROCS_PER_GPU}
cd {shlex.quote(workdir)}
source {shlex.quote(setup)}
test ! -e {shlex.quote(run_dir)}
mkdir -p {shlex.quote(run_parent)}
mkdir {shlex.quote(run_dir)}
git -C {shlex.quote(checkout)} rev-parse HEAD > {shlex.quote(run_dir + '/source_commit.txt')}
export KIT_BOOT_MARKER='Learning iteration'
export KIT_BOOT_TIMEOUT_S={watchdog['boot_stall_timeout_s']}
export KIT_BOOT_STALE_TIMEOUT_S={watchdog['post_first_iteration_stall_timeout_s']}
setsid nohup {shlex.quote(KIT_BOOT_LOCK)} {shlex.quote(run_log)} {child} >> {shlex.quote(run_dir + '/launcher.out')} 2>&1 &
printf 'launched %s %s launcher_pid=%s\\n' {shlex.quote(job['id'])} {shlex.quote(stage)} "$!"
""".strip()


def _ssh_argv(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str, pod: str, gpu: Any
) -> list[str]:
    pod_name, gpu_index = _validate_target(queue, pod, gpu)
    endpoint = queue["pods"][pod_name]
    remote = _remote_body(queue, job, stage, gpu_index)
    key = os.path.expanduser(str(queue["ssh"]["key"]))
    return [
        "ssh", "-i", key, "-p", str(endpoint["port"]),
        "-o", "BatchMode=yes", f"root@{endpoint['host']}",
        f"bash -lc {shlex.quote(remote)}",
    ]


def render_command(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str, pod: str, gpu: Any
) -> str:
    return shlex.join(_ssh_argv(queue, job, stage, pod, gpu))


def _arm_summary(arm: str) -> str:
    if arm == "combo_fresh":
        return (
            "全升级合体·fresh：AR-0.2 + 高摩擦[1.0,1.6]/[0.8,1.2] + 凹凸 2-6 cm + "
            "落地罚 -3e-3 + 抬脚罚 -0.01@0.15 m + 二阶平滑 -0.05 + qbar -0.65 + "
            "速度推 ±0.35 + 力推 68 N（20001 从零，rough 铁律）"
        )
    if arm == "combo_resume":
        return (
            "全升级合体·续训：同 fresh 去凹凸与抬脚罚（平地谱系），W model_6700 + 13301"
        )
    return (
        "combo_resume + 反手换 franco_bh_loop_b（151f schema-2；franco 闸门锁死等 "
        "franco_pipeline 三交付；Franco：换动作是优先消融，排位最前）"
    )


def cmd_plan(queue: Mapping[str, Any]) -> str:
    lines = [
        f"queue: {queue['queue_id']}  3 臂 = W x {{combo_fresh,combo_resume,"
        "combo_franco}（集成升级合体），基础配方固定 C+S0 + AR-0.2",
        f"commit: {queue['source']['commit']}",
        "budgets: probe 4096env x 24steps x 2it | science 4096env x 24steps x 13301it "
        "save100（combo_fresh fresh 独立 20001it）",
        "watchdog: boot 停滞 1800 s / 首迭代后停滞 900 s 才判死；重试只许逐字 _r2 一次",
        "对照: 不重复买——矩阵 w_c_s0（C+S0、平地、默认摩擦、无 push、AR-0.1）就是对照；"
        "combo 赢了也不能归因单项（单项归因看 CGF/push/intel 各单变量波）",
        "比值守卫: 击球组 17/7/5/5/10 不动；新增负项(AR-0.2+二阶-0.05 连续,落地/抬脚/"
        "barrier 稀疏)每步合计 ≲ 击球收入 1/3，probe/首里程碑 tensorboard 实测核对",
        "槽位: 本波不写死 pod/gpu；空槽出现时渲染 --pod/--gpu 注入目标卡",
    ]
    for gate in ALL_ARM_GATES:
        confirmed = queue[gate]["wiring_confirmed"]
        if confirmed:
            lines.append(f"{gate}: 已确认（发射前仍须远端 grep 复核）")
        else:
            lines.append(
                f"{gate}: 锁定——三臂全带对应键面，全部拒渲染"
                + ("（action_acc 须实现合入 main 并重钉 commit 后再翻 true）"
                   if gate == "action_acc_contract" else "")
            )
    franco_gate = queue["franco_contract"]["wiring_confirmed"]
    franco_placeholder = _franco_motion_is_placeholder(queue)
    lines.append(
        "franco_contract: 已解锁（三前置交付+资产已钉）"
        if franco_gate and not franco_placeholder
        else "franco_contract: 锁定——combo_franco 等 franco_pipeline_20260722 交付 "
        "grip 标定 bake + 锚入册 + 按族重绑题库（另两臂不受影响）"
    )
    lines.append("")
    for job in queue["jobs"]:
        level = queue["mechanisms"]["arms"][job["arm"]]
        fresh_tag = "（fresh-from-random）" if _job_is_fresh(job) else ""
        lines.append(
            f"{job['id']:13s} {job['run_name']}{fresh_tag}\n"
            f"              人话: {level['human_name']}\n"
            f"              arm: {_arm_summary(job['arm'])}"
        )
    lines.append("")
    lines.append("推荐填充顺序（空槽出现时拉最前面还没发射且闸门已开的臂）：")
    for index, job_id in enumerate(queue["launch_order"], start=1):
        lines.append(f"  {index:2d}. {job_id}")
    return "\n".join(lines)


def cmd_checklist(queue: Mapping[str, Any]) -> str:
    parents = queue["parents"]
    lines = ["发射前依赖核对单（全部人工执行，本程序不 SSH）", ""]
    if queue["action_acc_contract"]["wiring_confirmed"] is not True:
        lines.append(
            "0. [阻塞] action_acc_contract.wiring_confirmed=false：mjlab 档①第三项 "
            "task.rewards.action_acc_weight 未确认接线（三臂全带此键，全部拒渲染）。"
            "谁实现谁补单测（照 foot_soft_landing 先例），合入 main 后重钉 "
            "source.commit 为新 40-hex、远端白名单核对一致再翻 true。"
        )
    lines += [
        f"1. 远端 checkout {queue['source']['checkout']} 存在、git status 干净、"
        f"HEAD == {queue['source']['commit']}（clean detached exact commit；本波钉的是 "
        "action_acc 接线合并后重钉的 origin/main HEAD e995b5d5，groundfoot b9c8fff2 "
        "与 push 4624c824 两次合并均为其祖先——host 已验证，远端仍须复核）。",
        "2. groundfoot 加检：grep 远端 checkout 在 exact commit 上的 train.py，确认"
        "白名单键逐字含 " + " / ".join(GROUNDFOOT_CLI_KEYS) + "。",
        "3. 推撞加检：grep 远端 train.py 确认 _PUSH_KEYS == (enable, interval_range_s, "
        "vel_xy_mps, ang_vel_radps, ang_axes) 且 _FORCE_PUSH_KEYS == (enable, "
        "interval_range_s, force_n, duration_s)；本波两组事件并存，键面逐字 = push 波 "
        "w_p035（±0.35 m/s）+ w_f035（68 N x 0.30 s @ pelvis link 原点，同冲量对表）。"
        "速度推/力推都是事件不是 reward。",
        "4. qbar/action_acc 加检：grep 远端 train.py 确认 qdes_limit_barrier_weight / "
        "qdes_limit_barrier_margin_frac / action_acc_weight 在白名单（action_acc 接线 "
        "= 本波重钉 commit e995b5d5，host 已 grep，远端仍须复核）；margin_frac=0.08 是"
        "行程比例不是 rad（07-22 语义裁定）；与 action_rate=-0.2 并存是本波设计"
        "（intel 单变量互斥不适用）。",
        "5. combo_fresh 一臂铁律：fresh-from-random——命令绝不带 checkpoint_path（渲染器"
        "已断言）；平地 checkpoint 上粗糙地会被 schema-3 ground_plant 合同块拒绝，这是"
        "设计不是事故。combo_fresh 判读只看粗糙地绝对水平与失稳率，与 w_c_s0 对比只作"
        "方向参考（谱系不同，结论必须带 caveat）。抬脚罚只在此臂（凹凸逼抬腿，"
        "落地罚+抬脚罚成对）。",
        "6. grip 两键会让新 checkpoint 长出 ground_plant 合同块 -> 与父本 W contract "
        "mismatch 属【有意】（checkpoint_allow_contract_mismatch=true 本来就开着，"
        "本波全谱系 diagnostic-only）；落地/抬脚/二阶/barrier 只动 reward weight，"
        "不进 plant 合同。",
        "7. combo_franco 三前置（franco_contract）：franco_pipeline_20260722 交付 "
        "grip 标定 bake + 空挥视觉锚点入册 + 按族重绑题库，三者齐了才把 "
        "assets.motion_backhand_franco 换成真实 npz 路径（sha256 入册）并翻 "
        "wiring_confirmed=true；在那之前该臂跳过不阻塞（Franco 排它最前，一开闸即首发）。",
        f"8. 两 pod 都确认 {KIT_BOOT_LOCK} 存在且可执行；本轮不用 "
        "launch_kit_training_locked.sh 的 180 s stale 门（v8/v9 死因）。",
        "9. 资产路径逐一 test -f / test -d（USD、正反手 motion、题库、A3 资产树；"
        "命令内已带 test -f 预检）。combo_franco 另加 franco_bh_loop_b npz 的 "
        "sha256sum 对账（入册值）。",
        "10. launch 前须 sha256sum 验证 parent checkpoint（combo_fresh 除外）：\n"
        f"   W: sha256sum {parents['W']['checkpoint_path']}\n"
        f"      期望 {parents['W']['checkpoint_sha256']}",
        f"11. namespace {queue['namespace']['root']} 全新（no_clobber）；3 个 run_dir 与 "
        "3 个 probe dir 都不得已存在。",
        "12. 排位与空槽：本波在全局队列排在 franco_pipeline_20260722 战役完成且卡上有"
        "空槽之后；发射前该卡 nvidia-smi compute 进程数 < 4（命令内已带预检），按 "
        "launch_order（combo_franco -> combo_resume -> combo_fresh，闸门未开跳过）拉"
        "最前面就绪的臂，渲染时 --render-job <id> --pod <pod> --gpu <g> 注入目标卡。",
        "13. 先跑 probe（每臂 2 个 update），自然退出后核对 run.log 出现 'Learning "
        "iteration'、无 fatal、checkpoint 存在（续训臂 model_6701.pt / combo_fresh "
        "model_1.pt），才允许发对应臂的 science（一格 smoke 通即发全矩阵）。",
        "14. 比值守卫核对（Franco 强调，本波特有）：probe + 首个里程碑（续训 "
        "model_6900 / fresh model_200）读 tensorboard 逐项 episode 均值，核对 "
        "(|action_rate| + |action_acc| + |foot_soft_landing| + |foot_clearance| + "
        "|qdes_limit_barrier|) / (racket_position + racket_velocity + racket_normal "
        "+ strike_success + progress) <= 1/3；超限该臂停发 science/停训裁决并记录读数。"
        "注：barrier/落地罚/抬脚罚是稀疏项（贴限位/落地/腾空才付费，常态 0），若其"
        "episode 均值异常大先查激活探针再谈停训。击球组 17/7/5/5/10 一动不动；"
        "软惩罚组全额不叠加。",
        "15. 发射节奏：两 pod 可并行；同 pod 内 boot 串行（kit_boot_lock 持锁），相邻两次 "
        "launch 错峰 >= 60 s。",
        "16. 日志摘要抓异常不抓预期：WARN 行必须全部进摘要（grep -n 'WARN' run.log）；"
        "q_des CLAMP ACTIVE 行必须进摘要（grep -Fn 'q_des CLAMP ACTIVE' run.log）；"
        "另加 grep 'ground_plant'/'terrain'/'push'/'force_push' 确认 plant 覆盖与两组"
        "推撞事件真的生效（applied 行里须出现对应 override）。",
        "17. watchdog：boot 停滞 1800 s、首个迭代后停滞 900 s 才算卡死；唯一允许的重试是"
        "逐字重发一次并给 run_name 加 _r2 后缀，仍需人工核对。",
        "18. 对照不重复买：矩阵 w_c_s0（p1btm_w_c_s0_seed3_20260720）就是本波对照；"
        "两 queue 的 source commit 不同时须 diff 训练路径无行为性改动并记录结论"
        "（本波必然不同——groundfoot/push/qbar wiring 是纯加法默认关，diff 结论须明确写"
        "『默认路径字节等价，host 测试有默认等价断言』），否则对照失效。",
        "19. 判读纪律：combo 臂赢了也不能归因单项——单项归因永远看 CGF/push/intel 各"
        "单变量波；W 后代 contract 有意 mismatch，胜者配方须另在 exact-lineage 重跑正名。",
    ]
    return "\n".join(lines)


def cmd_render(
    queue: Mapping[str, Any], stage: str, job_id: str, pod: str, gpu: Any
) -> str:
    if stage not in {"probe", "science"}:
        raise QueueError("--render-stage must be probe or science")
    job = _job_by_id(queue, job_id)
    pod_name, gpu_index = _validate_target(queue, pod, gpu)
    return (
        f"# {job['id']} {stage} {pod_name}/gpu{gpu_index} "
        f"run_name={_stage_run_name(job, stage)}\n"
        + render_command(queue, job, stage, pod_name, gpu_index)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--plan", action="store_true", help="打印 3 臂计划表（默认）")
    parser.add_argument("--render-stage", choices=("probe", "science"))
    parser.add_argument("--render-job", default=None, help="job id（配合 --render-stage）")
    parser.add_argument("--pod", default=None, help="渲染时注入的目标 pod（pod1|pod2）")
    parser.add_argument("--gpu", default=None, help="渲染时注入的目标 gpu（0|1|2）")
    parser.add_argument("--checklist", action="store_true", help="打印发射前依赖核对单")
    args = parser.parse_args(argv)
    try:
        queue = load_queue(args.queue)
        if args.render_stage is not None:
            if args.render_job is None or args.pod is None or args.gpu is None:
                raise QueueError(
                    "--render-stage requires --render-job <id> --pod <pod1|pod2> "
                    "--gpu <0|1|2>"
                )
            print(cmd_render(queue, args.render_stage, args.render_job, args.pod, args.gpu))
        elif args.checklist:
            print(cmd_checklist(queue))
        else:
            print(cmd_plan(queue))
    except QueueError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
