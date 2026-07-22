#!/usr/bin/env python3
"""渲染抖动-地面-脚部消融波 8 臂队列的 SSH 命令（phase1_chatter_ground_foot_wave_20260722）。

人话：这个程序只做三件事——(1) ``--plan`` 打印 8 臂计划表和推荐填充顺序；(2)
``--render-stage probe|science --render-job <id> --pod <pod1|pod2> --gpu <0|1|2>``
输出单臂的逐字 SSH 命令给人工核对后执行（本波不写死 pod/gpu，空槽即填，目标卡
在渲染时注入）；(3) ``--checklist`` 输出发射前依赖核对单。它自己绝不 SSH、绝不
发信号、绝不写远端。所有校验 fail-closed：YAML 缺键、占位 commit、C+S0 配方漂移、
消融档漂移、push/force 键面混入、非法 pod/gpu、run_name 重复、闸门未开渲染被锁臂，
都直接拒绝。

8 档消融臂（父本只用 W；对照不重复买＝矩阵 w_c_s0）：
- ar02/ar05/ar10：action_rate -0.1 → -0.2/-0.5/-1.0（剂量曲线；jiayi V15 反向情报
  ＝降到 -0.01 靠投影约束，正反两说都记录）；
- grip：机器人 body 材质摩擦随机化下界抬高（静 0.3→0.6 / 动 0.3→0.5，脚不搓地假设）；
- rough：随机凹凸地形 2-6 cm，fresh-from-random 从零训 20001 个 update（平地
  checkpoint 不能静默上粗糙地；配方串仍用 W——配方是实验设计，不是谱系）；
- footrw：mjlab 落地冲击惩罚 -3e-3（无量纲超阈倍数量纲！mjlab -1e-5/N x 300 N 阈值
  的等效剂量；别抄 -0.1）；
- penlight：六个软惩罚统一降 ~1/3（含 07-22 新接线的蹭滑/拖脚剂量键；硬保护不动；Franco 第 8 条）；
- kdpassive：被动阻尼折进训练 kd——源码未接线，kdpassive_contract 闸门锁死。

grip/rough/footrw 受 groundfoot_contract 闸门锁（新键 wiring 已落盘本地未合并）；
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


DEFAULT_QUEUE = Path("configs/phase1_chatter_ground_foot_wave_20260722.yaml")
EXPECTED_QUEUE_ID = "phase1_chatter_ground_foot_wave_20260722"
EXPECTED_NAMESPACE = "/workspace/codexschema/phase1_chatter_ground_foot_wave_20260722"
EXPECTED_CHECKOUT = "/workspace/codexschema/nohope_cgf_20260722"
PLACEHOLDER_COMMIT = "PENDING_40HEX_AFTER_WIRING_MERGE"
KIT_BOOT_LOCK = "/workspace/bin/kit_boot_lock.sh"
PARENT_ITERATION = 6700
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
JOB_ID = re.compile(r"^(ar02|ar05|ar10|grip|rough|footrw|penlight|kdpassive)$")
HYDRA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

ARM_ORDER = (
    "ar02", "ar05", "ar10", "grip", "rough", "footrw", "penlight", "kdpassive",
)
PARENT_ORDER = ("W",)
VALID_GPUS = (0, 1, 2)
# 同卡在跑 compute 进程必须 < 4 才允许发射（逐字继承矩阵/intel wave 预检）。
MAX_PROCS_PER_GPU = 4

# 人话：空槽即填的冻结顺序 = 价值排序（Franco 拍板）：grip → footrw → ar05 →
# penlight → rough → ar02 → ar10 → kdpassive。闸门未开的臂跳过、不阻塞后面的臂。
LAUNCH_ORDER = (
    "grip", "footrw", "ar05", "penlight", "rough", "ar02", "ar10", "kdpassive",
)
# groundfoot 闸门锁的四臂（新 plant/foot/蹭滑拖脚键）；kdpassive 闸门单锁 kdpassive。
GROUNDFOOT_GATED_ARMS = ("grip", "rough", "footrw", "penlight")
FRESH_ARMS = ("rough",)

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

# 逐字照抄矩阵 C 档（= intel wave 的 EXPECTED_TEMPORAL_C）。
EXPECTED_TEMPORAL_C = [
    "task.rewards.action_rate_weight=-0.1",
    "++task.rewards.processed_qdes_slew_hinge_weight=0.0",
    "++task.rewards.processed_qdes_slew_hinge_margin=0.85",
    "++task.rewards.processed_qdes_slew_hinge_recovery_start_s=0.2",
    "++task.rewards.processed_qdes_slew_hinge_recovery_end_s=1.55",
]


def _temporal_with_action_rate(weight: str) -> list[str]:
    return [f"task.rewards.action_rate_weight={weight}", *EXPECTED_TEMPORAL_C[1:]]


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

# penlight：base 里唯一被减负的行（拍面条件引导 -0.4 → -0.13）。
FACE_GUIDANCE_BASE = "++task.rewards.racket_face_conditional_guidance_weight=-0.4"
FACE_GUIDANCE_PENLIGHT = "++task.rewards.racket_face_conditional_guidance_weight=-0.13"
# penlight：W 配方里被减负的两行（脚朝向、挥拍前直立）。
FOOT_ORI_BASE = "task.rewards.foot_orientation_weight=-0.3"
FOOT_ORI_PENLIGHT = "task.rewards.foot_orientation_weight=-0.1"
UPRIGHT_BASE = "task.rewards.prestrike_upright_weight=-1.0"
UPRIGHT_PENLIGHT = "task.rewards.prestrike_upright_weight=-0.33"
# penlight：挥拍前脚滑现役有效值 -0.4（DeployParity YAML 第 177 行，argv 里原本不带
# 这个键）→ 显式 -0.13；触地脚蹭滑/拖脚（源码常开 -1.0/-0.5，07-22 新接 CLI 键）
# → -0.33/-0.17。
SLIP_PENLIGHT = "task.rewards.pre_strike_foot_slip_weight=-0.13"
SLIPSQ_PENLIGHT = "++task.rewards.foot_slip_sq_weight=-0.33"
DRAG_PENLIGHT = "++task.rewards.foot_drag_weight=-0.17"

# grip：机器人 body 材质摩擦随机化范围（现役 (0.3,1.6)/(0.3,1.2)，下界 0.3 很滑）。
GRIP_STATIC_RANGE = "++task.plant.robot_material_static_friction_range=[0.6,1.6]"
GRIP_DYNAMIC_RANGE = "++task.plant.robot_material_dynamic_friction_range=[0.5,1.2]"
# rough：随机凹凸地形高度场 2-6 cm。
ROUGH_HEIGHT_RANGE = "++task.plant.terrain_rough_height_range=[0.02,0.06]"
# footrw：mjlab 落地冲击（无量纲超阈倍数，mjlab 等效剂量 -3e-3）。
FOOTRW_WEIGHT = "++task.rewards.foot_soft_landing_weight=-0.003"
FOOTRW_THRESHOLD = "++task.rewards.foot_soft_landing_force_threshold_n=300.0"
# kdpassive：源码未接线的开关（闸门锁死渲染）。
KDPASSIVE_FOLD = "++task.plant.passive_damping_fold=true"

EXPECTED_BASE_REPLACEMENTS: dict[str, list[dict[str, str]]] = {
    arm: [] for arm in ARM_ORDER
}
EXPECTED_BASE_REPLACEMENTS["penlight"] = [
    {"old": FACE_GUIDANCE_BASE, "new": FACE_GUIDANCE_PENLIGHT},
]
EXPECTED_PARENT_RECIPE_REPLACEMENTS: dict[str, list[dict[str, str]]] = {
    arm: [] for arm in ARM_ORDER
}
EXPECTED_PARENT_RECIPE_REPLACEMENTS["penlight"] = [
    {"old": FOOT_ORI_BASE, "new": FOOT_ORI_PENLIGHT},
    {"old": UPRIGHT_BASE, "new": UPRIGHT_PENLIGHT},
]
EXPECTED_TEMPORAL = {
    "ar02": _temporal_with_action_rate("-0.2"),
    "ar05": _temporal_with_action_rate("-0.5"),
    "ar10": _temporal_with_action_rate("-1.0"),
    "grip": EXPECTED_TEMPORAL_C,
    "rough": EXPECTED_TEMPORAL_C,
    "footrw": EXPECTED_TEMPORAL_C,
    "penlight": EXPECTED_TEMPORAL_C,
    "kdpassive": EXPECTED_TEMPORAL_C,
}
EXPECTED_ACTION_RATE = {
    "ar02": -0.2, "ar05": -0.5, "ar10": -1.0,
    "grip": -0.1, "rough": -0.1, "footrw": -0.1,
    "penlight": -0.1, "kdpassive": -0.1,
}
EXPECTED_STABILITY = {arm: EXPECTED_STABILITY_S0 for arm in ARM_ORDER}
EXPECTED_EXTRA = {
    "ar02": [],
    "ar05": [],
    "ar10": [],
    "grip": [GRIP_STATIC_RANGE, GRIP_DYNAMIC_RANGE],
    "rough": [ROUGH_HEIGHT_RANGE],
    "footrw": [FOOTRW_WEIGHT, FOOTRW_THRESHOLD],
    "penlight": [SLIP_PENLIGHT, SLIPSQ_PENLIGHT, DRAG_PENLIGHT],
    "kdpassive": [KDPASSIVE_FOLD],
}

GROUNDFOOT_CLI_KEYS = [
    "task.plant.robot_material_static_friction_range",
    "task.plant.robot_material_dynamic_friction_range",
    "task.plant.terrain_rough_height_range",
    "task.rewards.foot_soft_landing_weight",
    "task.rewards.foot_soft_landing_force_threshold_n",
    "task.rewards.foot_slip_sq_weight",
    "task.rewards.foot_drag_weight",
]
KDPASSIVE_CLI_KEYS = ["task.plant.passive_damping_fold"]
# 单变量纪律：每臂允许出现的"新键面"（task.plant 新键 + 脚部 reward 键 + 挥拍前脚滑）。
NEW_SURFACE_PREFIXES = (
    "task.plant.robot_material_", "task.plant.terrain_rough_",
    "task.plant.ground_", "task.plant.passive_damping_fold",
    "task.rewards.foot_soft_landing", "task.rewards.foot_clearance",
    "task.rewards.pre_strike_foot_slip_weight",
    "task.rewards.foot_slip_sq_weight", "task.rewards.foot_drag_weight",
)
EXPECTED_NEW_SURFACE = {
    "ar02": set(), "ar05": set(), "ar10": set(),
    "grip": {
        "task.plant.robot_material_static_friction_range",
        "task.plant.robot_material_dynamic_friction_range",
    },
    "rough": {"task.plant.terrain_rough_height_range"},
    "footrw": {
        "task.rewards.foot_soft_landing_weight",
        "task.rewards.foot_soft_landing_force_threshold_n",
    },
    "penlight": {
        "task.rewards.pre_strike_foot_slip_weight",
        "task.rewards.foot_slip_sq_weight",
        "task.rewards.foot_drag_weight",
    },
    "kdpassive": {"task.plant.passive_damping_fold"},
}

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

EXPECTED_ASSETS = {
    "a3_runtime_asset_root": "/workspace/codexschema/nohope_balance_action_slew_20260720/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
    "preconverted_a3_usd": "/workspace/codexschema/simple_half_second_sprint_20260718/assets/a3_preconverted_usd/model.usd",
    "motion_forehand": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_forehand_v4rg_cal.npz",
    "motion_backhand": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_backhand_v4rg_cal.npz",
    "training_question_bank": "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_bank_rebind_v2/s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz",
}

# 续训臂 science/probe 命令必须含的键（值另行断言）；fresh 臂（rough）绝不携带任何
# checkpoint 键。push/force 键面继续全禁。
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
            "source.commit is still the placeholder; merge the 2026-07-22 wiring "
            "branch, re-checkout on the pod, and pin the exact merged 40-hex commit "
            "before rendering any SSH command"
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


def _validate_replacement_pairs(
    pairs: Any, expected: list[dict[str, str]], label: str
) -> None:
    pairs = _list(pairs, label)
    if pairs != expected:
        raise QueueError(f"{label} changed from frozen design")
    for index, raw_pair in enumerate(pairs):
        pair = _mapping(raw_pair, f"{label}[{index}]")
        _exact_keys(pair, {"old", "new"}, f"{label}[{index}]")
        old_key = _override_key(pair["old"], f"{label}[{index}].old")
        new_key = _override_key(pair["new"], f"{label}[{index}].new")
        if old_key != new_key:
            raise QueueError(
                f"{label}[{index}] must replace a line in place "
                f"(same Hydra key), got {old_key!r} -> {new_key!r}"
            )


def _validate_mechanism(name: str, mechanism: dict[str, Any]) -> None:
    label = f"mechanisms.arms.{name}"
    _exact_keys(
        mechanism,
        {
            "human_name", "base_replacements", "parent_recipe_replacements",
            "temporal_overrides", "stability_overrides", "extra_overrides",
        },
        label,
    )
    _text(mechanism["human_name"], f"{label}.human_name")
    _validate_replacement_pairs(
        mechanism["base_replacements"],
        EXPECTED_BASE_REPLACEMENTS[name],
        f"{label}.base_replacements",
    )
    _validate_replacement_pairs(
        mechanism["parent_recipe_replacements"],
        EXPECTED_PARENT_RECIPE_REPLACEMENTS[name],
        f"{label}.parent_recipe_replacements",
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
    if action_rate != EXPECTED_ACTION_RATE[name]:
        raise QueueError(
            f"{label} action_rate must be exactly {EXPECTED_ACTION_RATE[name]}"
        )
    hinge = _override_value(
        temporal, "task.rewards.processed_qdes_slew_hinge_weight", label
    )
    if hinge != 0.0:
        raise QueueError(f"{label} slew hinge must stay 0 (single variable per arm)")
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
    if name == "footrw":
        weight = _override_value(
            extra, "task.rewards.foot_soft_landing_weight", label
        )
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
    for raw in stability:
        if raw.lstrip("+").split("=", 1)[0].endswith("_probe_weight"):
            raise QueueError(
                f"{label} must not pass *_probe_weight: train.py has no such CLI key "
                "(probes are auto-forced to 1.0 by the explicit weight keys)"
            )


def _validate_gate_contract(
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


def _require_gates_open(queue: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """渲染闸门：grip/rough/footrw 等 groundfoot wiring；kdpassive 等自己的实现。"""

    if job["id"] in GROUNDFOOT_GATED_ARMS and (
        queue["groundfoot_contract"]["wiring_confirmed"] is not True
    ):
        raise QueueError(
            f"{job['id']} needs the 2026-07-22 ground/foot wiring merged into "
            "source.commit (groundfoot_contract.wiring_confirmed=false); refusing "
            "to render until the controller verifies the train.py whitelist at the "
            "frozen commit — the other four arms are unaffected"
        )
    if job["id"] == "kdpassive" and (
        queue["kdpassive_contract"]["wiring_confirmed"] is not True
    ):
        raise QueueError(
            "kdpassive's passive_damping_fold key has NO source wiring at all "
            "(kdpassive_contract.wiring_confirmed=false); implement + test + merge "
            "before rendering — the other seven arms are unaffected"
        )


def _job_is_fresh(job: Mapping[str, Any]) -> bool:
    return job["fresh_from_random"] is True


def _validate_jobs(queue: dict[str, Any]) -> None:
    jobs = _list(queue["jobs"], "jobs")
    if len(jobs) != 8:
        raise QueueError("the queue must contain exactly 8 jobs")
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
                f"jobs[{index}].parent must be W (single-parent wave; rough is "
                "fresh via fresh_from_random, the W recipe is the design not the "
                "lineage)"
            )
        if not isinstance(job["fresh_from_random"], bool):
            raise QueueError(f"jobs[{index}].fresh_from_random must be a bool")
        if job["fresh_from_random"] != (job_id in FRESH_ARMS):
            raise QueueError(
                f"jobs[{index}].fresh_from_random must be true exactly for "
                f"{list(FRESH_ARMS)} (rough must never resume a flat-ground "
                "checkpoint; every other arm must resume W)"
            )
        run_name = _text(job["run_name"], f"jobs[{index}].run_name", safe_id=True)
        expected_name = f"p1cgf_{job_id}_seed3_20260722"
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
        raise QueueError("jobs must keep the frozen ar02..kdpassive ordering")

    order = _list(queue["launch_order"], "launch_order")
    if order != list(LAUNCH_ORDER):
        raise QueueError(
            "launch_order must keep the frozen grip->footrw->ar05->penlight->rough->"
            "ar02->ar10->kdpassive value-ranked fill order"
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
            "mechanisms", "groundfoot_contract", "kdpassive_contract",
            "probe_contract", "budgets", "launch_order", "jobs",
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
        if _remote_path(assets[key], f"assets.{key}") != expected:
            raise QueueError(f"assets.{key} changed from the frozen verbatim copy")

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
            "common.base_overrides must carry the verbatim "
            f"{FACE_GUIDANCE_BASE!r} line (the penlight in-place replacement target)"
        )
    forbidden_base = sorted(
        key for key in base_map
        if key.startswith("task.push.") or key.startswith("task.force_push.")
    )
    if forbidden_base:
        raise QueueError(
            f"push/force key surface is out of scope for this wave: {forbidden_base}"
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

    _validate_gate_contract(
        _mapping(queue["groundfoot_contract"], "groundfoot_contract"),
        "groundfoot_contract",
        GROUNDFOOT_CLI_KEYS,
    )
    _validate_gate_contract(
        _mapping(queue["kdpassive_contract"], "kdpassive_contract"),
        "kdpassive_contract",
        KDPASSIVE_CLI_KEYS,
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

    # 每臂两阶段都编译一遍：证明没有重复 Hydra 键、必带键齐全、单变量键面纪律成立。
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
    return f"p1cgf_probe_{job['id']}_seed3_20260722"


def _science_budget(queue: Mapping[str, Any], job: Mapping[str, Any]) -> Mapping[str, Any]:
    return queue["budgets"]["science_fresh" if _job_is_fresh(job) else "science"]


def _with_replacements(
    lines: Sequence[str], replacements: Sequence[Mapping[str, str]], label: str
) -> list[str]:
    """把 {old,new} 替换对原位应用到一份 override 串（不增不删键）。"""

    result = [str(item) for item in lines]
    for index, pair in enumerate(replacements):
        old, new = str(pair["old"]), str(pair["new"])
        if result.count(old) != 1:
            raise QueueError(
                f"{label}[{index}] target line must appear exactly once: {old!r}"
            )
        result[result.index(old)] = new
    return result


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
    base = _with_replacements(
        queue["common"]["base_overrides"],
        level["base_replacements"],
        f"{job['id']}.base_replacements",
    )
    recipes = _with_replacements(
        parent["recipe_overrides"],
        level["parent_recipe_replacements"],
        f"{job['id']}.parent_recipe_replacements",
    )
    # fresh 臂（rough）绝不携带任何 checkpoint 键：平地 checkpoint 不能静默上粗糙地；
    # 配方串仍用 W（配方是实验设计，不是谱系）。
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
    argv = [
        source["python"],
        f"{workdir}/{source['trainer_relative']}",
        *base,
        queue["common"]["planner_revision_override"],
        *recipes,
        *level["temporal_overrides"],
        *level["stability_overrides"],
        *level["extra_overrides"],
        f"motion_file={queue['assets']['motion_forehand']}",
        f"motion_file_2={queue['assets']['motion_backhand']}",
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
    forbidden = sorted(
        key for key in compiled
        if key.startswith("task.push.") or key.startswith("task.force_push.")
    )
    if forbidden:
        raise QueueError(
            f"{job['id']} carries push/force keys {forbidden}; the push/force key "
            "surface is out of scope for this wave (absent == byte-for-byte no-op)"
        )
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
    # 单变量纪律：每臂的"新键面"（plant 新键/脚部 reward/挥拍前脚滑）必须逐字等于
    # 该臂的冻结集合——别的臂出现即拒绝（缺席 == 字节等价 no-op）。
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
    # penlight 数值双保险 + 其余臂软惩罚全额（漂移即拒）。
    face = compiled.get("task.rewards.racket_face_conditional_guidance_weight")
    foot_ori = compiled.get("task.rewards.foot_orientation_weight")
    upright = compiled.get("task.rewards.prestrike_upright_weight")
    if job["arm"] == "penlight":
        if (face, foot_ori, upright) != ("-0.13", "-0.1", "-0.33"):
            raise QueueError(
                "penlight must reduce face/foot-orientation/upright to exactly "
                f"-0.13/-0.1/-0.33, got {(face, foot_ori, upright)}"
            )
        if compiled.get("task.rewards.pre_strike_foot_slip_weight") != "-0.13":
            raise QueueError("penlight must set pre_strike_foot_slip_weight=-0.13")
        if (
            compiled.get("task.rewards.foot_slip_sq_weight") != "-0.33"
            or compiled.get("task.rewards.foot_drag_weight") != "-0.17"
        ):
            raise QueueError(
                "penlight must reduce foot_slip_sq/foot_drag to exactly -0.33/-0.17"
            )
    else:
        if (face, foot_ori, upright) != ("-0.4", "-0.3", "-1.0"):
            raise QueueError(
                f"{job['id']} must keep the full-dose soft penalties "
                f"-0.4/-0.3/-1.0, got {(face, foot_ori, upright)}"
            )
    return argv


def _remote_body(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str, gpu: int
) -> str:
    commit = _require_render_commit(queue)
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
        queue["assets"]["motion_backhand"],
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
    if arm in {"ar02", "ar05", "ar10"}:
        dose = {"ar02": "-0.2", "ar05": "-0.5", "ar10": "-1.0"}[arm]
        return f"action_rate -0.1 -> {dose}（剂量曲线；对照 w_c_s0 = -0.1 档）"
    if arm == "grip":
        return "机器人材质摩擦随机化抬高：静 [0.6,1.6] 动 [0.5,1.2]（现役下界 0.3 很滑）"
    if arm == "rough":
        return "随机凹凸地形 2-6 cm，fresh-from-random 20001 updates（不传 resume）"
    if arm == "footrw":
        return "落地冲击罚 -3e-3 @300 N（无量纲超阈倍数；mjlab -1e-5/N 等效剂量）"
    if arm == "penlight":
        return (
            "软惩罚降 1/3：face -0.13 / foot_ori -0.1 / upright -0.33 / "
            "pre-slip -0.13 / slip_sq -0.33 / drag -0.17"
        )
    return "passive_damping_fold=true（源码未接线，闸门锁死）"


def cmd_plan(queue: Mapping[str, Any]) -> str:
    commit = queue["source"]["commit"]
    commit_line = (
        f"commit: 占位符 {PLACEHOLDER_COMMIT}（渲染被锁死，主控合并 07-22 wiring 后填 40-hex）"
        if _commit_is_placeholder(commit)
        else f"commit: {commit}"
    )
    groundfoot_gate = queue["groundfoot_contract"]["wiring_confirmed"]
    kd_gate = queue["kdpassive_contract"]["wiring_confirmed"]
    groundfoot_line = (
        "groundfoot 渲染: 已解锁（地面/脚部新键 wiring 已确认合入 source.commit）"
        if groundfoot_gate
        else "groundfoot 渲染: 锁定——grip/rough/footrw/penlight 四臂等 07-22 新键 wiring "
        "合并钉 commit 后翻 wiring_confirmed（其余四臂不受影响）"
    )
    kd_line = (
        "kdpassive 渲染: 已解锁（passive_damping_fold 实现+单测已确认合入）"
        if kd_gate
        else "kdpassive 渲染: 锁定——passive_damping_fold 源码完全未接线，实现+单测"
        "合入 main 后才翻（其余七臂不受影响）"
    )
    lines = [
        f"queue: {queue['queue_id']}  8 臂 = W x {{ar02,ar05,ar10,grip,rough,footrw,"
        "penlight,kdpassive}（抖动-地面-脚部消融），基础配方固定 C+S0",
        commit_line,
        "budgets: probe 4096env x 24steps x 2it | science 4096env x 24steps x 13301it "
        "save100（rough fresh 独立 20001it）",
        "watchdog: boot 停滞 1800 s / 首迭代后停滞 900 s 才判死；重试只许逐字 _r2 一次",
        "对照: 不重复买——矩阵 w_c_s0（同 C+S0 配方、平地、默认摩擦、软惩罚全额）就是对照",
        "槽位: 本波不写死 pod/gpu；空槽出现时渲染 --pod/--gpu 注入目标卡",
        groundfoot_line,
        kd_line,
        "",
    ]
    for job in queue["jobs"]:
        level = queue["mechanisms"]["arms"][job["arm"]]
        fresh_tag = "（fresh-from-random）" if _job_is_fresh(job) else ""
        lines.append(
            f"{job['id']:10s} {job['run_name']}{fresh_tag}\n"
            f"           人话: {level['human_name']}\n"
            f"           arm: {_arm_summary(job['arm'])}"
        )
    lines.append("")
    lines.append("推荐填充顺序（空槽出现时拉最前面还没发射且闸门已开的臂）：")
    for index, job_id in enumerate(queue["launch_order"], start=1):
        lines.append(f"  {index:2d}. {job_id}")
    return "\n".join(lines)


def cmd_checklist(queue: Mapping[str, Any]) -> str:
    parents = queue["parents"]
    commit = queue["source"]["commit"]
    lines = ["发射前依赖核对单（全部人工执行，本程序不 SSH）", ""]
    if _commit_is_placeholder(commit):
        lines.append(
            f"0. [阻塞] source.commit 仍是占位符 {PLACEHOLDER_COMMIT}：先把 07-22 "
            "工作分支（脚部 reward + 地面/地形键 + YAML-null/预检/精确续训三搬运）合并，"
            "在 pod 重新 checkout 后填入 40-hex 精确 commit，否则渲染器拒绝出命令。"
        )
    lines += [
        f"1. 远端 checkout {queue['source']['checkout']} 存在、git status 干净、"
        "HEAD == source.commit（clean detached exact commit）。",
        "2. grip/rough/footrw/penlight 四臂加检（groundfoot 闸门）：grep 远端 checkout 在 exact "
        "commit 上的 train.py，确认白名单键逐字含 "
        + " / ".join(GROUNDFOOT_CLI_KEYS)
        + "（wiring 已在本地落盘+host 测试全绿：test_ground_plant_wiring.py 38 绿、"
        "test_foot_contact_shaping.py 42 绿；主控核对远端一致后才翻 "
        "groundfoot_contract.wiring_confirmed=true；键名不同必须先停下重议再渲染）。",
        "3. kdpassive 一臂加检（kdpassive 闸门）：task.plant.passive_damping_fold 源码"
        "完全未接线——实现+单测合入 main、远端白名单核对一致后才翻 "
        "kdpassive_contract.wiring_confirmed=true；在那之前该臂跳过不阻塞。",
        "4. rough 一臂铁律：fresh-from-random——命令绝不带 checkpoint_path（渲染器已"
        "断言）；平地 checkpoint 上粗糙地会被 schema-3 ground_plant 合同块拒绝，这是"
        "设计不是事故。rough 判读只看粗糙地绝对水平与失稳率，与 w_c_s0 对比只作方向"
        "参考（谱系不同，结论必须带 caveat）。",
        "5. grip 两键会让新 checkpoint 长出 ground_plant 合同块 -> 与父本 W contract "
        "mismatch 属【有意】（checkpoint_allow_contract_mismatch=true 本来就开着，"
        "本波全谱系 diagnostic-only）；footrw 只动 reward weight，不进合同。",
        f"6. 两 pod 都确认 {KIT_BOOT_LOCK} 存在且可执行；本轮不用 "
        "launch_kit_training_locked.sh 的 180 s stale 门（v8/v9 死因）。",
        "7. 资产路径逐一 test -f / test -d（USD、正反手 motion、题库、A3 资产树；"
        "命令内已带 test -f 预检）。",
        "8. launch 前须 sha256sum 验证 parent checkpoint（rough 除外）：\n"
        f"   W: sha256sum {parents['W']['checkpoint_path']}\n"
        f"      期望 {parents['W']['checkpoint_sha256']}",
        f"9. namespace {queue['namespace']['root']} 全新（no_clobber）；8 个 run_dir 与 "
        "8 个 probe dir 都不得已存在。",
        "10. 空槽即填：谁先毕业谁腾卡；发射前该卡 nvidia-smi compute 进程数 < 4"
        "（命令内已带预检），按 launch_order 拉最前面就绪且闸门已开的臂，渲染时 "
        "--render-job <id> --pod <pod> --gpu <g> 注入目标卡。",
        "11. 先跑 probe（每臂 2 个 update），自然退出后核对 run.log 出现 'Learning "
        "iteration'、无 fatal、checkpoint 存在（续训臂 model_6701.pt / rough 臂 "
        "model_1.pt），才允许发对应臂的 science（一格 smoke 通即发全矩阵）。",
        "12. 发射节奏：两 pod 可并行；同 pod 内 boot 串行（kit_boot_lock 持锁），相邻两次 "
        "launch 错峰 >= 60 s。",
        "13. 日志摘要抓异常不抓预期：WARN 行必须全部进摘要（grep -n 'WARN' run.log）；"
        "q_des CLAMP ACTIVE 行必须进摘要（grep -Fn 'q_des CLAMP ACTIVE' run.log）；"
        "grip/rough 另加 grep 'ground_plant'/'terrain' 确认 plant 覆盖真的生效"
        "（applied 行里须出现对应 override）。",
        "14. watchdog：boot 停滞 1800 s、首个迭代后停滞 900 s 才算卡死；唯一允许的重试是"
        "逐字重发一次并给 run_name 加 _r2 后缀，仍需人工核对。",
        "15. 对照不重复买：矩阵 w_c_s0（p1btm_w_c_s0_seed3_20260720）就是本波对照；"
        "两 queue 的 source commit 不同时须 diff 训练路径无行为性改动并记录结论"
        "（本波必然不同——07-22 wiring 是纯加法默认关，diff 结论须明确写"
        "『默认路径字节等价，host 测试有默认等价断言』），否则对照失效。",
        "16. 全部 8 臂 science 里程碑落盘后按诊断谱系解读：W 后代 contract 有意 "
        "mismatch，胜者档位须另在 exact-lineage 重跑；ar 判读带上 jiayi V15 反向情报"
        "（他把 action_rate 降到 -0.01 靠投影约束）；kdpassive 判读带上『jiayi 侧训练"
        "分布自述还没调好』caveat。",
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
