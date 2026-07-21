#!/usr/bin/env python3
"""渲染 Wave Q 8 臂 {W,V}x{spdmix,hstrong,fullbody,qbar} 情报波队列的 SSH 命令。

人话：这个程序只做三件事——(1) ``--plan`` 打印 8 臂计划表和推荐填充顺序；(2)
``--render-stage probe|science --render-job <id> --pod <pod1|pod2> --gpu <0|1|2>``
输出单臂的逐字 SSH 命令给人工核对后执行（本波不写死 pod/gpu，空槽即填，目标卡
在渲染时注入）；(3) ``--checklist`` 输出发射前依赖核对单。它自己绝不 SSH、绝不
发信号、绝不写远端。所有校验 fail-closed：YAML 缺键、占位 commit、C+S0 配方漂移、
情报档漂移、push/force 键面混入、非法 pod/gpu、run_name 重复，都直接拒绝。

四档情报臂（Franco 2026-07-21 情报，全部单变量对矩阵 C+S0 对照）：
- spdmix（v2）：参考动作换成 TOPP 烤入变速六 clip 列表——正手 0.8/1.0/1.2 + 反手
  0.8/1.0/1.1（帧数/fps/触球帧不变，触球行逐位相同），每次挥拍均匀抽一个 clip；播放
  时钟 speed_scale_range 保持 [1.0,1.0]（在线变速 v1 被 governor 单时钟守卫拒绝＝
  历史）。strike_phase/mount_normal_sign 两行 base 原位扩成六位，extra 带
  clip_family_per_clip 族表，motion_file/motion_file_2/题库三行换 assets 的
  spdmix_*（题库=按族重绑版，允许 SHA=两个 cal 原件+六个烤入）；
- hstrong：action_rate=0 + 恢复窗 q_des slew hinge -1.0（margin/窗与矩阵 H 档同）；
- fullbody：下肢姿态模仿 weight 0→2.0（std 0.35），支持窗放开到全程 pre/post 10.0 s
  （两阶段下肢方案第一阶段：静止击球下肢全程软模仿）；
- qbar：全关节 qdes 限位 barrier（Jiayi v14 去 top-k、逐关节 tail 求和），weight
  -0.65、margin 0.08，并把 raw action_rate 归零（barrier 是该臂唯一 qdes 惩罚），
  在 qbar_contract.qbar_wiring_confirmed=true 之前渲染被锁死（其余六臂不受影响）。

对照不重复买：矩阵 w_c_s0/v_c_s0（同 C+S0 配方、speed 恒 1.0、无 barrier）就是对照。
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


DEFAULT_QUEUE = Path("configs/phase1_intel_wave_20260721.yaml")
EXPECTED_QUEUE_ID = "phase1_intel_wave_20260721"
EXPECTED_NAMESPACE = "/workspace/codexschema/phase1_intel_wave_20260721"
EXPECTED_CHECKOUT = "/workspace/codexschema/nohope_push_20260721"
PLACEHOLDER_COMMIT = "PENDING_EXACT_COMMIT"
KIT_BOOT_LOCK = "/workspace/bin/kit_boot_lock.sh"
PARENT_ITERATION = 6700
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
JOB_ID = re.compile(r"^(w|v)_(spdmix|hstrong|fullbody|qbar)$")
HYDRA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

LEVEL_ORDER = ("spdmix", "hstrong", "fullbody", "qbar")
PARENT_ORDER = ("W", "V")
VALID_GPUS = (0, 1, 2)
# 同卡在跑 compute 进程必须 < 4 才允许发射（逐字继承矩阵预检）。
MAX_PROCS_PER_GPU = 4

# 人话：空槽即填的冻结顺序 = 价值排序——速度泛化（最高价值轴）→ 全身模仿 →
# 强 q_des 平滑 → qbar（闸门未开时跳过、不阻塞后面没有臂了所以顺延即可）。
LAUNCH_ORDER = (
    "w_spdmix", "v_spdmix",
    "w_fullbody", "v_fullbody",
    "w_hstrong", "v_hstrong",
    "w_qbar", "v_qbar",
)

EXPECTED_PARENT_SHA256 = {
    "W": "2caab3dde3a0ac6c051ff8ac65385a641cac152aa3f84b640126b5ed7b96fcce",
    "V": "ad9019100f199f23669829b0fbc4f8c2ad45c8073f930348f177da9487332716",
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
    "V": [
        "task.rewards.racket_position_weight=7.0",
        "task.rewards.racket_velocity_weight=17.0",
        "task.rewards.racket_normal_weight=5.0",
        "task.rewards.foot_orientation_weight=-0.6",
        "task.rewards.prestrike_upright_weight=-2.0",
        "++task.rewards.free_non_striking_arm_mimic=false",
    ],
}

# 逐字照抄矩阵 C 档（configs/phase1_balance_temporal_matrix_20260720.yaml）。
EXPECTED_TEMPORAL_C = [
    "task.rewards.action_rate_weight=-0.1",
    "++task.rewards.processed_qdes_slew_hinge_weight=0.0",
    "++task.rewards.processed_qdes_slew_hinge_margin=0.85",
    "++task.rewards.processed_qdes_slew_hinge_recovery_start_s=0.2",
    "++task.rewards.processed_qdes_slew_hinge_recovery_end_s=1.55",
]

# hstrong 档：action_rate 0、hinge -1.0；margin/窗三行逐字 = 矩阵 H 档（H 是 -0.25）。
EXPECTED_TEMPORAL_HSTRONG = [
    "task.rewards.action_rate_weight=0.0",
    "++task.rewards.processed_qdes_slew_hinge_weight=-1.0",
    "++task.rewards.processed_qdes_slew_hinge_margin=0.85",
    "++task.rewards.processed_qdes_slew_hinge_recovery_start_s=0.2",
    "++task.rewards.processed_qdes_slew_hinge_recovery_end_s=1.55",
]

# qbar 档：C 档去掉 raw action_rate（-0.10→0，Franco："把别的去掉"）——barrier 是该臂
# 唯一 qdes 惩罚；slew hinge 保持 0，margin/窗三行逐字 = 矩阵 C 档。
EXPECTED_TEMPORAL_QBAR = [
    "task.rewards.action_rate_weight=0.0",
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

# fullbody 档 = S0 三行替换：下肢姿态模仿 weight 0.0 -> 2.0，支持窗 pre/post
# 0.3/0.4 -> 10.0/10.0（≈ 全程；两阶段下肢方案第一阶段：静止击球下肢全程软模仿）。
FULLBODY_POSE_SWAPS = {
    "++task.rewards.lower_body_pose_imitation_weight=0.0":
        "++task.rewards.lower_body_pose_imitation_weight=2.0",
    "++task.rewards.lower_body_pose_imitation_support_pre_s=0.3":
        "++task.rewards.lower_body_pose_imitation_support_pre_s=10.0",
    "++task.rewards.lower_body_pose_imitation_support_post_s=0.4":
        "++task.rewards.lower_body_pose_imitation_support_post_s=10.0",
}
EXPECTED_STABILITY_FULLBODY = [
    FULLBODY_POSE_SWAPS.get(line, line) for line in EXPECTED_STABILITY_S0
]

# spdmix v2 档：speed_scale_range 全臂恒 [1.0,1.0]（governor 单时钟守卫，松不得）；
# 变速由六个 TOPP 烤入 clip 承担。strike_phase / mount_normal_sign 两行 base 原位
# 扩成六位（不新增键、不重复键），extra 带 clip_family_per_clip 族表。
SPEED_SCALE_BASE = "task.motion.speed_scale_range=[1.0,1.0]"
STRIKE_PHASE_BASE = "task.racket.strike_phase_per_clip=[0.471,0.338]"
STRIKE_PHASE_SPDMIX = (
    "task.racket.strike_phase_per_clip=[0.471,0.471,0.471,0.338,0.338,0.338]"
)
MOUNT_SIGN_BASE = "++task.racket.mount_normal_sign_per_clip=[1.0,-1.0]"
MOUNT_SIGN_SPDMIX = (
    "++task.racket.mount_normal_sign_per_clip=[1.0,1.0,1.0,-1.0,-1.0,-1.0]"
)
CLIP_FAMILY_SPDMIX = (
    "++task.motion.clip_family_per_clip="
    "[forehand,forehand,forehand,backhand,backhand,backhand]"
)

# qbar 档：全关节 qdes 限位 barrier（无 top-k）。键名按任务书冻结；wiring 由并行
# agent 落盘，主控核对远端 train.py 白名单一致后才翻 qbar_wiring_confirmed。
EXPECTED_QBAR_EXTRA = [
    "++task.rewards.qdes_limit_barrier_weight=-0.65",
    "++task.rewards.qdes_limit_barrier_margin_frac=0.08",
]
QBAR_CLI_KEYS = [
    "task.rewards.qdes_limit_barrier_weight",
    "task.rewards.qdes_limit_barrier_margin_frac",
]
QBAR_WEIGHT = -0.65
QBAR_MARGIN = 0.08

EXPECTED_BASE_REPLACEMENTS: dict[str, list[dict[str, str]]] = {
    "spdmix": [
        {"old": STRIKE_PHASE_BASE, "new": STRIKE_PHASE_SPDMIX},
        {"old": MOUNT_SIGN_BASE, "new": MOUNT_SIGN_SPDMIX},
    ],
    "hstrong": [],
    "fullbody": [],
    "qbar": [],
}
EXPECTED_TEMPORAL = {
    "spdmix": EXPECTED_TEMPORAL_C,
    "hstrong": EXPECTED_TEMPORAL_HSTRONG,
    "fullbody": EXPECTED_TEMPORAL_C,
    "qbar": EXPECTED_TEMPORAL_QBAR,
}
EXPECTED_STABILITY = {
    "spdmix": EXPECTED_STABILITY_S0,
    "hstrong": EXPECTED_STABILITY_S0,
    "fullbody": EXPECTED_STABILITY_FULLBODY,
    "qbar": EXPECTED_STABILITY_S0,
}
EXPECTED_EXTRA = {
    "spdmix": [CLIP_FAMILY_SPDMIX],
    "hstrong": [],
    "fullbody": [],
    "qbar": EXPECTED_QBAR_EXTRA,
}
# spdmix v2 铁律：全部 8 臂播放时钟恒 [1.0,1.0]——变速由烤入 clip 承担，governor
# 单时钟守卫不受扰；[0.8,1.2] 在线变速（v1）在任何臂出现都直接拒绝。
EXPECTED_SPEED_SCALE = {
    "spdmix": "[1.0,1.0]",
    "hstrong": "[1.0,1.0]",
    "fullbody": "[1.0,1.0]",
    "qbar": "[1.0,1.0]",
}
# spdmix v2 的每臂逐字键面（其余六臂 = base 原样两位表 + 无族表键）。
EXPECTED_STRIKE_PHASE = {
    level: (STRIKE_PHASE_SPDMIX if level == "spdmix" else STRIKE_PHASE_BASE).split("=", 1)[1]
    for level in LEVEL_ORDER
}
EXPECTED_MOUNT_SIGN = {
    level: (MOUNT_SIGN_SPDMIX if level == "spdmix" else MOUNT_SIGN_BASE).split("=", 1)[1]
    for level in LEVEL_ORDER
}
EXPECTED_CLIP_FAMILY = {
    level: (CLIP_FAMILY_SPDMIX.split("=", 1)[1] if level == "spdmix" else None)
    for level in LEVEL_ORDER
}

EXPECTED_PODS = {
    "pod1": ("162.43.172.171", 18333),
    "pod2": ("162.43.172.181", 13146),
}

EXPECTED_CONTROLS = {
    "baseline_queue_id": "phase1_balance_temporal_matrix_20260720",
    "baseline_jobs": ["w_c_s0", "v_c_s0"],
    "baseline_run_names": [
        "p1btm_w_c_s0_seed3_20260720", "p1btm_v_c_s0_seed3_20260720",
    ],
}

EXPECTED_PROBE_BUDGET = {
    "num_envs": 4096, "num_steps_per_env": 24,
    "max_iterations": 2, "save_interval": 1,
}
EXPECTED_SCIENCE_BUDGET = {
    "num_envs": 4096, "num_steps_per_env": 24,
    "max_iterations": 4001, "save_interval": 100,
    "milestone_offsets_from_parent": [200, 500, 1000, 2000, 4000],
    "absolute_milestones": [6900, 7200, 7700, 8700, 10700],
}
EXPECTED_WATCHDOG = {
    "boot_stall_timeout_s": 1800,
    "post_first_iteration_stall_timeout_s": 900,
    "retry_policy": "one_verbatim_retry_suffix_r2",
}

# spdmix v2 资产：TOPP 烤入变速片段 + 按族重绑题库（上传对账见同目录
# upload_manifest_spdmix_v2_20260722.json）。只有 spdmix 两臂用；其余六臂仍用
# cal 原件 + 原 train 题库，argv 逐字节不变。
SPDMIX_ASSET_DIR = "/workspace/codexschema/phase1_intel_wave_20260721/assets/topp_speed"
SPDMIX_FOREHAND_CLIPS = [
    f"{SPDMIX_ASSET_DIR}/hope_forehand_v4rg_speed0p80.npz",
    f"{SPDMIX_ASSET_DIR}/hope_forehand_v4rg_speed1p00.npz",
    f"{SPDMIX_ASSET_DIR}/hope_forehand_v4rg_speed1p20.npz",
]
SPDMIX_BACKHAND_CLIPS = [
    f"{SPDMIX_ASSET_DIR}/hope_backhand_v4rg_speed0p80.npz",
    f"{SPDMIX_ASSET_DIR}/hope_backhand_v4rg_speed1p00.npz",
    f"{SPDMIX_ASSET_DIR}/hope_backhand_v4rg_speed1p10.npz",
]
SPDMIX_QUESTION_BANK = (
    f"{SPDMIX_ASSET_DIR}/s1_v4rg_runtime_order_schema3_train_882fea4_family6_rebound.npz"
)

EXPECTED_ASSETS = {
    "a3_runtime_asset_root": "/workspace/codexschema/nohope_balance_action_slew_20260720/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
    "preconverted_a3_usd": "/workspace/codexschema/simple_half_second_sprint_20260718/assets/a3_preconverted_usd/model.usd",
    "motion_forehand": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_forehand_v4rg_cal.npz",
    "motion_backhand": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_backhand_v4rg_cal.npz",
    "training_question_bank": "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_bank_rebind_v2/s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz",
    "spdmix_motion_forehand_clips": SPDMIX_FOREHAND_CLIPS,
    "spdmix_motion_backhand_clips": SPDMIX_BACKHAND_CLIPS,
    "spdmix_training_question_bank": SPDMIX_QUESTION_BANK,
}

# science 阶段命令必须含的键（值另行断言）。本波剔除 push/force 键面：任何
# task.push.* / task.force_push.* 键出现在 argv 里都直接拒绝（见 _training_argv）。
REQUIRED_SCIENCE_KEYS = (
    "checkpoint_path", "checkpoint_tolerant", "checkpoint_allow_missing_contract",
    "checkpoint_allow_contract_mismatch", "seed", "num_envs",
    "algo.runner.num_steps_per_env", "max_iterations", "algo.runner.save_interval",
    "run_name", "device", "logger",
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


def _commit_is_placeholder(commit: str) -> bool:
    return commit == PLACEHOLDER_COMMIT


def _validate_commit_field(value: Any) -> str:
    commit = _text(value, "source.commit")
    if commit == PLACEHOLDER_COMMIT:
        return commit
    if not COMMIT.fullmatch(commit) or commit == "0" * 40:
        raise QueueError(
            "source.commit must be a non-placeholder 40-hex commit or the literal "
            f"{PLACEHOLDER_COMMIT!r}"
        )
    return commit


def _require_render_commit(queue: Mapping[str, Any]) -> str:
    commit = queue["source"]["commit"]
    if _commit_is_placeholder(commit):
        raise QueueError(
            "source.commit is still the placeholder; fill the exact merged 40-hex "
            "commit before rendering any SSH command"
        )
    return commit


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


def _validate_base_replacements(name: str, replacements: Any) -> None:
    label = f"mechanisms.intel.{name}.base_replacements"
    pairs = _list(replacements, label)
    if pairs != EXPECTED_BASE_REPLACEMENTS[name]:
        raise QueueError(f"{label} changed from frozen design")
    for index, raw_pair in enumerate(pairs):
        pair = _mapping(raw_pair, f"{label}[{index}]")
        _exact_keys(pair, {"old", "new"}, f"{label}[{index}]")
        old_key = _override_key(pair["old"], f"{label}[{index}].old")
        new_key = _override_key(pair["new"], f"{label}[{index}].new")
        if old_key != new_key:
            raise QueueError(
                f"{label}[{index}] must replace a base line in place "
                f"(same Hydra key), got {old_key!r} -> {new_key!r}"
            )


def _validate_mechanism(name: str, mechanism: dict[str, Any]) -> None:
    label = f"mechanisms.intel.{name}"
    _exact_keys(
        mechanism,
        {
            "human_name", "base_replacements", "temporal_overrides",
            "stability_overrides", "extra_overrides",
        },
        label,
    )
    _text(mechanism["human_name"], f"{label}.human_name")
    _validate_base_replacements(name, mechanism["base_replacements"])
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
    hinge = _override_value(
        temporal, "task.rewards.processed_qdes_slew_hinge_weight", label
    )
    action_rate = _override_value(temporal, "task.rewards.action_rate_weight", label)
    margin = _override_value(
        temporal, "task.rewards.processed_qdes_slew_hinge_margin", label
    )
    if margin != 0.85:
        raise QueueError(f"{label} hinge margin must stay 0.85 (matrix C/H verbatim)")
    if name == "hstrong":
        if action_rate != 0.0 or hinge != -1.0:
            raise QueueError(
                f"{label} must be exactly action_rate 0 / slew hinge -1.0"
            )
    elif name == "qbar":
        # Franco："把别的去掉"——qbar 去 raw action_rate，barrier 是唯一 qdes 惩罚。
        if action_rate != 0.0 or hinge != 0.0:
            raise QueueError(
                f"{label} must be exactly action_rate 0 / slew 0 "
                "(the barrier is this arm's only qdes penalty)"
            )
    else:
        if action_rate != -0.1 or hinge != 0.0:
            raise QueueError(f"{label} temporal must be exactly matrix C: -0.10 / slew 0")
    pose = _override_value(
        stability, "task.rewards.lower_body_pose_imitation_weight", label
    )
    if name == "fullbody":
        if pose != 2.0:
            raise QueueError(f"{label} must set lower-body pose imitation weight 2.0")
    elif pose != 0.0:
        raise QueueError(f"{label} must keep lower-body pose imitation weight 0")
    if _override_value(stability, "task.rewards.lower_body_pose_imitation_std", label) != 0.35:
        raise QueueError(f"{label} pose imitation std must stay 0.35")
    # fullbody 支持窗放开到全程（两阶段下肢方案第一阶段）；其余臂保持 S0 窄窗。
    window_pre, window_post = (10.0, 10.0) if name == "fullbody" else (0.3, 0.4)
    for key, expected in (
        ("task.rewards.lower_body_pose_imitation_support_pre_s", window_pre),
        ("task.rewards.lower_body_pose_imitation_support_post_s", window_post),
    ):
        if _override_value(stability, key, label) != expected:
            raise QueueError(
                f"{label} pose imitation window drifted (fullbody must be the "
                "full-episode 10.0/10.0 window, every other arm pre 0.3/post 0.4)"
            )
    for key in (
        "task.rewards.post_swing_settle_debt_weight",
        "task.rewards.lower_body_stability_bundle_weight",
    ):
        if _override_value(stability, key, label) != 0.0:
            raise QueueError(f"{label} must keep the other two mechanism weights at 0")
    if name == "qbar":
        weight = _override_value(extra, "task.rewards.qdes_limit_barrier_weight", label)
        barrier_margin = _override_value(
            extra, "task.rewards.qdes_limit_barrier_margin_frac", label
        )
        if weight != QBAR_WEIGHT or barrier_margin != QBAR_MARGIN:
            raise QueueError(
                f"{label} barrier must be exactly weight {QBAR_WEIGHT} / margin {QBAR_MARGIN}"
            )
        if not barrier_margin > 0.0:
            raise QueueError(f"{label} barrier margin must be positive")
    for raw in stability:
        if raw.lstrip("+").split("=", 1)[0].endswith("_probe_weight"):
            raise QueueError(
                f"{label} must not pass *_probe_weight: train.py has no such CLI key "
                "(probes are auto-forced to 1.0 by the explicit weight keys)"
            )


def _validate_qbar_contract(contract: dict[str, Any]) -> None:
    """qbar 渲染闸门：键名冻结 + wiring 确认布尔 + 人话说明。"""

    _exact_keys(
        contract,
        {"qbar_wiring_confirmed", "expected_cli_keys", "wiring_note"},
        "qbar_contract",
    )
    if not isinstance(contract["qbar_wiring_confirmed"], bool):
        raise QueueError("qbar_contract.qbar_wiring_confirmed must be a bool")
    if contract["expected_cli_keys"] != QBAR_CLI_KEYS:
        raise QueueError(
            "qbar_contract.expected_cli_keys must be exactly the two frozen "
            "task.rewards.qdes_limit_barrier_* keys"
        )
    _text(contract["wiring_note"], "qbar_contract.wiring_note")


def _require_qbar_wiring_confirmed(
    queue: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    """qbar 臂渲染闸门：wiring 未确认合入 source.commit 前拒绝出任何 qbar 命令。"""

    if job["intel"] != "qbar":
        return
    if queue["qbar_contract"]["qbar_wiring_confirmed"] is not True:
        raise QueueError(
            "qdes_limit_barrier wiring is not confirmed merged into source.commit "
            "(qbar_contract.qbar_wiring_confirmed=false); refusing to render any "
            "qbar command until the controller flips it after verifying the "
            "train.py whitelist at the frozen commit — the other six arms are "
            "unaffected"
        )


def _expected_jobs_order() -> list[tuple[str, str, str]]:
    """冻结的 jobs 排列：先 W 后 V，档位按 spdmix,hstrong,fullbody,qbar。"""

    result = []
    for parent in PARENT_ORDER:
        for level in LEVEL_ORDER:
            result.append((f"{parent.lower()}_{level}", parent, level))
    return result


def _validate_jobs(queue: dict[str, Any]) -> None:
    jobs = _list(queue["jobs"], "jobs")
    if len(jobs) != 8:
        raise QueueError("the queue must contain exactly 8 jobs")
    ids: set[str] = set()
    names: set[str] = set()
    dirs: set[str] = set()
    cells: set[tuple[str, str]] = set()
    observed: list[tuple[str, str, str]] = []
    for index, raw_job in enumerate(jobs):
        job = _mapping(raw_job, f"jobs[{index}]")
        _exact_keys(job, {"id", "parent", "intel", "run_name", "run_dir"}, f"jobs[{index}]")
        job_id = _text(job["id"], f"jobs[{index}].id", safe_id=True)
        match = JOB_ID.fullmatch(job_id)
        if match is None:
            raise QueueError(
                f"jobs[{index}].id must match w|v_spdmix|hstrong|fullbody|qbar: {job_id!r}"
            )
        parent, level = match.group(1).upper(), match.group(2)
        if (job["parent"], job["intel"]) != (parent, level):
            raise QueueError(f"jobs[{index}] axes do not match its id {job_id!r}")
        if job["parent"] not in PARENT_ORDER or job["intel"] not in LEVEL_ORDER:
            raise QueueError(f"jobs[{index}] references an unknown axis value")
        run_name = _text(job["run_name"], f"jobs[{index}].run_name", safe_id=True)
        expected_name = f"p1iq_{job_id}_seed3_20260721"
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
        cell = (parent, level)
        if cell in cells:
            raise QueueError(f"intel cell duplicated: {cell}")
        cells.add(cell)
        observed.append((job_id, parent, level))
    if len(cells) != 8:
        raise QueueError("parent x intel coverage is incomplete")
    if observed != _expected_jobs_order():
        raise QueueError("jobs must keep the frozen W-then-V, spdmix..qbar ordering")

    order = _list(queue["launch_order"], "launch_order")
    if order != list(LAUNCH_ORDER):
        raise QueueError(
            "launch_order must keep the frozen spdmix->fullbody->hstrong->qbar "
            "value-ranked fill order"
        )
    if set(order) != ids:
        raise QueueError("launch_order must be a permutation of the 8 job ids")


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
            "mechanisms", "qbar_contract", "probe_contract", "budgets",
            "launch_order", "jobs",
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
    _exact_keys(assets, set(EXPECTED_ASSETS), "assets")
    for key, expected in EXPECTED_ASSETS.items():
        if isinstance(expected, list):
            actual = _list(assets[key], f"assets.{key}")
            if [
                _remote_path(item, f"assets.{key}[{index}]")
                for index, item in enumerate(actual)
            ] != expected:
                raise QueueError(f"assets.{key} changed from the frozen spdmix v2 list")
        elif _remote_path(assets[key], f"assets.{key}") != expected:
            raise QueueError(f"assets.{key} changed from the frozen verbatim copy")

    _validate_controls(_mapping(queue["controls"], "controls"))

    parents = _mapping(queue["parents"], "parents")
    if set(parents) != set(PARENT_ORDER):
        raise QueueError("parents must be exactly W and V")
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
    if SPEED_SCALE_BASE not in base:
        raise QueueError(
            "common.base_overrides must carry the verbatim "
            f"{SPEED_SCALE_BASE!r} line (the governor single-clock pin on all arms)"
        )
    for required_line in (STRIKE_PHASE_BASE, MOUNT_SIGN_BASE):
        if required_line not in base:
            raise QueueError(
                "common.base_overrides must carry the verbatim "
                f"{required_line!r} line (the spdmix v2 in-place replacement target)"
            )
    forbidden_base = sorted(
        key for key in base_map
        if key.startswith("task.push.") or key.startswith("task.force_push.")
    )
    if forbidden_base:
        raise QueueError(
            f"push/force key surface is out of scope for this wave: {forbidden_base}"
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
        "task.rewards.qdes_limit_barrier_weight",
        "task.rewards.qdes_limit_barrier_margin_frac",
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
    _exact_keys(mechanisms, {"intel"}, "mechanisms")
    intel = _mapping(mechanisms["intel"], "mechanisms.intel")
    if set(intel) != set(LEVEL_ORDER):
        raise QueueError(
            "mechanisms.intel must be exactly spdmix, hstrong, fullbody, qbar"
        )
    for name, mechanism in intel.items():
        _validate_mechanism(name, _mapping(mechanism, f"mechanisms.intel.{name}"))

    _validate_qbar_contract(_mapping(queue["qbar_contract"], "qbar_contract"))

    _mapping(queue["probe_contract"], "probe_contract")

    budgets = _mapping(queue["budgets"], "budgets")
    _exact_keys(budgets, {"probe", "science"}, "budgets")
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

    _validate_jobs(queue)

    # 每臂两阶段都编译一遍：证明没有重复 Hydra 键、必带键齐全、spdmix 替换到位。
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
    return f"p1iq_probe_{job['id']}_seed3_20260721"


def _base_with_replacements(
    queue: Mapping[str, Any], level: Mapping[str, Any], label: str
) -> list[str]:
    """把该档的 base_replacements 原位应用到 common.base_overrides（不增不删键）。"""

    result = [str(item) for item in queue["common"]["base_overrides"]]
    for index, pair in enumerate(level["base_replacements"]):
        old, new = str(pair["old"]), str(pair["new"])
        if result.count(old) != 1:
            raise QueueError(
                f"{label}.base_replacements[{index}] target line must appear exactly "
                f"once in common.base_overrides: {old!r}"
            )
        result[result.index(old)] = new
    return result


def _motion_and_bank_paths(
    queue: Mapping[str, Any], job: Mapping[str, Any]
) -> tuple[list[str], list[str], str]:
    """该臂实际装载的 (正手 clip 列表, 反手 clip 列表, 题库路径)。

    人话：spdmix 两臂换成六个 TOPP 烤入变速片段 + 按族重绑题库；其余六臂保持两个
    cal 原件 + 原 train 题库，一字不动。
    """
    assets = queue["assets"]
    if job["intel"] == "spdmix":
        return (
            [str(path) for path in assets["spdmix_motion_forehand_clips"]],
            [str(path) for path in assets["spdmix_motion_backhand_clips"]],
            str(assets["spdmix_training_question_bank"]),
        )
    return (
        [str(assets["motion_forehand"])],
        [str(assets["motion_backhand"])],
        str(assets["training_question_bank"]),
    )


def _motion_list_value(paths: Sequence[str]) -> str:
    """单 clip 渲染裸路径（与历史命令逐字节一致）；多 clip 渲染 Hydra 列表字面量。"""
    if len(paths) == 1:
        return paths[0]
    return "[" + ",".join(paths) + "]"


def _training_argv(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str
) -> list[str]:
    if stage not in {"probe", "science"}:
        raise QueueError("stage must be probe or science")
    source = queue["source"]
    workdir = f"{source['checkout']}/{source['worktree_relative']}"
    parent = queue["parents"][job["parent"]]
    level = queue["mechanisms"]["intel"][job["intel"]]
    budget = queue["budgets"][stage]
    base = _base_with_replacements(queue, level, job["id"])
    motion_fh, motion_bh, question_bank = _motion_and_bank_paths(queue, job)
    argv = [
        source["python"],
        f"{workdir}/{source['trainer_relative']}",
        *base,
        queue["common"]["planner_revision_override"],
        *parent["recipe_overrides"],
        *level["temporal_overrides"],
        *level["stability_overrides"],
        *level["extra_overrides"],
        f"motion_file={_motion_list_value(motion_fh)}",
        f"motion_file_2={_motion_list_value(motion_bh)}",
        f"++task.racket.question_bank={question_bank}",
        f"checkpoint_path={parent['checkpoint_path']}",
        "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=true",
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
    forbidden = sorted(
        key for key in compiled
        if key.startswith("task.push.") or key.startswith("task.force_push.")
    )
    if forbidden:
        raise QueueError(
            f"{job['id']} carries push/force keys {forbidden}; the push/force key "
            "surface is out of scope for this wave (absent == byte-for-byte no-op)"
        )
    for key in REQUIRED_SCIENCE_KEYS:
        if key not in compiled:
            raise QueueError(f"{job['id']}.{stage} command is missing required key {key}")
    # spdmix v2 逐字纪律：全部 8 臂播放时钟恒 [1.0,1.0]（governor 单时钟守卫）；
    # 在线变速 [0.8,1.2]（v1）在任何臂出现都直接拒绝。
    speed = compiled.get("task.motion.speed_scale_range")
    if speed != EXPECTED_SPEED_SCALE[job["intel"]]:
        raise QueueError(
            f"{job['id']} speed_scale_range must be "
            f"{EXPECTED_SPEED_SCALE[job['intel']]!r}, got {speed!r}"
        )
    # spdmix v2 键面纪律：spdmix 两臂六位 strike_phase/mount_sign + 六位族表 + 六 clip
    # 变速列表 + 按族重绑题库；其余六臂两位表、无族表键、cal 原件 + 原题库（逐字节不变）。
    for key, expected_by_level in (
        ("task.racket.strike_phase_per_clip", EXPECTED_STRIKE_PHASE),
        ("task.racket.mount_normal_sign_per_clip", EXPECTED_MOUNT_SIGN),
        ("task.motion.clip_family_per_clip", EXPECTED_CLIP_FAMILY),
    ):
        expected_value = expected_by_level[job["intel"]]
        if compiled.get(key) != expected_value:
            raise QueueError(
                f"{job['id']} {key} must be {expected_value!r}, got {compiled.get(key)!r}"
            )
    motion_fh, motion_bh, question_bank = _motion_and_bank_paths(queue, job)
    for key, expected_value in (
        ("motion_file", _motion_list_value(motion_fh)),
        ("motion_file_2", _motion_list_value(motion_bh)),
        ("task.racket.question_bank", question_bank),
    ):
        if compiled.get(key) != expected_value:
            raise QueueError(
                f"{job['id']} {key} must be {expected_value!r}, got {compiled.get(key)!r}"
            )
    # 单变量纪律：qbar 两臂必带两个 barrier 键，其余六臂绝不出现 barrier 键。
    barrier_keys = sorted(
        key for key in compiled if key.startswith("task.rewards.qdes_limit_barrier")
    )
    if job["intel"] == "qbar":
        if barrier_keys != sorted(QBAR_CLI_KEYS):
            raise QueueError(
                f"{job['id']} must carry exactly the two frozen qdes_limit_barrier keys"
            )
    elif barrier_keys:
        raise QueueError(
            f"{job['id']} is not a qbar arm; barrier keys {barrier_keys} are "
            "forbidden (single variable: absent barrier == no-op)"
        )
    return argv


def _remote_body(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str, gpu: int
) -> str:
    commit = _require_render_commit(queue)
    _require_qbar_wiring_confirmed(queue, job)
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
    # 发射前逐文件 test -f：该臂真正装载的 motion clip（spdmix=六个烤入片段）与题库
    # （spdmix=按族重绑版）都要在场，缺一个立即拒绝。
    motion_fh, motion_bh, question_bank = _motion_and_bank_paths(queue, job)
    required_files = [
        trainer,
        setup,
        queue["assets"]["preconverted_a3_usd"],
        *motion_fh,
        *motion_bh,
        question_bank,
        parent["checkpoint_path"],
        parent["hard_contract_path"],
    ]
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


def _level_summary(level_name: str) -> str:
    if level_name == "spdmix":
        return (
            "v2 烤入变速六 clip 列表：正手 0.8/1.0/1.2 + 反手 0.8/1.0/1.1（TOPP 烤入、"
            "触球行逐位不变），clip_family 族表 + 按族重绑题库；speed_scale 恒 [1.0,1.0]"
        )
    if level_name == "hstrong":
        return "action_rate=0.0, slew hinge -1.0 @margin 0.85 窗 0.2-1.55s（H 档 -0.25 作剂量对照）"
    if level_name == "fullbody":
        return (
            "lower_body_pose_imitation +2.0 std 0.35 窗全程 pre10/post10"
            "（两阶段下肢方案第一阶段；S3 +0.5 窄窗作剂量对照）"
        )
    return (
        "qdes_limit_barrier weight -0.65 margin 0.08rad"
        "（全关节、无 top-k、逐关节求和）+ action_rate=0（barrier 是唯一 qdes 惩罚）"
    )


def cmd_plan(queue: Mapping[str, Any]) -> str:
    commit = queue["source"]["commit"]
    commit_line = (
        "commit: 占位符 PENDING_EXACT_COMMIT（渲染被锁死，主控合并 qbar wiring 后填 40-hex）"
        if _commit_is_placeholder(commit)
        else f"commit: {commit}"
    )
    qbar_gate = queue["qbar_contract"]["qbar_wiring_confirmed"]
    qbar_gate_line = (
        "qbar 渲染: 已解锁（qdes_limit_barrier wiring 已确认合入 source.commit）"
        if qbar_gate
        else "qbar 渲染: 锁定——qdes_limit_barrier wiring 未确认合入 source.commit，"
        "主控核对远端 train.py 白名单后翻 qbar_wiring_confirmed（其余六臂不受影响）"
    )
    lines = [
        f"queue: {queue['queue_id']}  8 臂 = {{W,V}} x {{spdmix,hstrong,fullbody,qbar}}"
        "（速度泛化/强 q_des 平滑/全身模仿/全关节 qdes barrier），基础配方固定 C+S0",
        commit_line,
        "budgets: probe 4096env x 24steps x 2it | science 4096env x 24steps x 4001it save100",
        "watchdog: boot 停滞 1800 s / 首迭代后停滞 900 s 才判死；重试只许逐字 _r2 一次",
        "对照: 不重复买——矩阵 w_c_s0/v_c_s0（同 C+S0 配方、speed 恒 1.0、无 barrier）就是对照",
        "槽位: 本波不写死 pod/gpu；空槽出现时渲染 --pod/--gpu 注入目标卡",
        qbar_gate_line,
        "",
    ]
    for job in queue["jobs"]:
        parent_human = queue["parents"][job["parent"]]["human_name"]
        level = queue["mechanisms"]["intel"][job["intel"]]
        lines.append(
            f"{job['id']:10s} {job['run_name']}\n"
            f"           人话: {parent_human} | {level['human_name']}\n"
            f"           intel: {_level_summary(job['intel'])}"
        )
    lines.append("")
    lines.append("推荐填充顺序（空槽出现时拉最前面还没发射的臂）：")
    for index, job_id in enumerate(queue["launch_order"], start=1):
        lines.append(f"  {index:2d}. {job_id}")
    return "\n".join(lines)


def cmd_checklist(queue: Mapping[str, Any]) -> str:
    parents = queue["parents"]
    commit = queue["source"]["commit"]
    lines = ["发射前依赖核对单（全部人工执行，本程序不 SSH）", ""]
    if _commit_is_placeholder(commit):
        lines.append(
            "0. [阻塞] source.commit 仍是占位符 PENDING_EXACT_COMMIT：主控合并 qbar "
            "wiring 后填入 40-hex 精确 commit，否则渲染器拒绝出命令。"
        )
    lines += [
        f"1. 远端 checkout {queue['source']['checkout']} 存在、git status 干净、"
        "HEAD == source.commit（clean detached exact commit）。",
        "2. qbar 两臂加检：grep 远端 checkout 在 exact commit 上的 train.py，"
        "确认 qdes_limit_barrier 白名单键逐字等于 "
        "task.rewards.qdes_limit_barrier_weight / task.rewards.qdes_limit_barrier_margin_frac"
        "（本地工作树冻结时 grep 无此 wiring，故 qbar_contract.qbar_wiring_confirmed="
        "false 锁死 qbar 渲染；主控核对一致后才翻 true，其余六臂不受影响；键名不同"
        "必须先停下重议再渲染，缺键则 boot 即 fail-loud）。",
        "3. spdmix 两臂加检（v2 烤入列表）：(a) grep 远端 checkout 在 exact commit 上的 "
        "train.py，确认 task.motion 白名单含 clip_family_per_clip（v2 改造 2026-07-22 "
        "落盘，主控合并并重钉 source.commit 后才发 spdmix；缺 wiring 则 boot 即 "
        "fail-loud）；(b) 渲染出的命令必须带六位 strike_phase/mount_normal_sign、"
        "++task.motion.clip_family_per_clip=[forehand x3, backhand x3]、motion_file/"
        "motion_file_2 六 clip 烤入列表与按族重绑题库 "
        f"{SPDMIX_QUESTION_BANK}；(c) 全部 8 臂 speed_scale_range 恒 [1.0,1.0]，任何臂"
        "出现 [0.8,1.2] 都是漂移（渲染器已逐字断言，人工再扫一眼）；(d) 六个烤入 npz + "
        "题库先 sha256sum 对 upload_manifest_spdmix_v2_20260722.json 逐一相等再发射。",
        f"4. 两 pod 都确认 {KIT_BOOT_LOCK} 存在且可执行；本轮不用 "
        "launch_kit_training_locked.sh 的 180 s stale 门（v8/v9 死因）。",
        "5. 资产路径逐一 test -f / test -d（USD、正反手 motion、题库、A3 资产树；"
        "spdmix 两臂另加六个烤入 npz + 按族重绑题库，命令内已带 test -f 预检）。",
        "6. launch 前须 sha256sum 验证 parent checkpoint：\n"
        f"   W: sha256sum {parents['W']['checkpoint_path']}\n"
        f"      期望 {parents['W']['checkpoint_sha256']}\n"
        f"   V: sha256sum {parents['V']['checkpoint_path']}\n"
        f"      期望 {parents['V']['checkpoint_sha256']}",
        f"7. namespace {queue['namespace']['root']} 全新（no_clobber）；8 个 run_dir 与 "
        "8 个 probe dir 都不得已存在。",
        "8. 空槽即填：谁先毕业谁腾卡；发射前该卡 nvidia-smi compute 进程数 < 4"
        "（命令内已带预检），按 launch_order 拉最前面就绪的臂（qbar 闸门未开时跳过 "
        "qbar 两臂），渲染时 --render-job <id> --pod <pod> --gpu <g> 注入目标卡。",
        "9. 先跑 probe（每臂 2 个 update），自然退出后核对 run.log 出现 'Learning iteration'"
        "、无 fatal、model_6701.pt 存在，才允许发对应臂的 science。",
        "10. 发射节奏：两 pod 可并行；同 pod 内 boot 串行（kit_boot_lock 持锁），相邻两次 "
        "launch 错峰 >= 60 s。",
        "11. 日志摘要抓异常不抓预期：WARN 行必须全部进摘要（grep -n 'WARN' run.log）；"
        "q_des CLAMP ACTIVE 行必须进摘要（grep -Fn 'q_des CLAMP ACTIVE' run.log）。",
        "12. watchdog：boot 停滞 1800 s、首个迭代后停滞 900 s 才算卡死；唯一允许的重试是"
        "逐字重发一次并给 run_name 加 _r2 后缀，仍需人工核对。",
        "13. 对照不重复买：矩阵 w_c_s0/v_c_s0（p1btm_w_c_s0_seed3_20260720 / "
        "p1btm_v_c_s0_seed3_20260720）就是本波对照，同 C+S0 配方、speed 恒 1.0、"
        "无 barrier，汇总时直接对比；两 queue 的 source commit 不同时须 diff 训练"
        "路径无行为性改动并记录结论，否则对照失效。",
        "14. 全部 8 臂 science 里程碑（6900/7200/7700/8700/10700）落盘后，汇总时按诊断"
        "谱系解读：W/V contract 有意 mismatch，胜者档位须另在 exact-lineage 重跑；"
        "qbar 判读须带 caveat——Jiayi 侧训练分布自述还没调好，其 v14 证据只作方向参考。",
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
    parser.add_argument("--plan", action="store_true", help="打印 8 臂计划表（默认）")
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
