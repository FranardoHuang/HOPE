#!/usr/bin/env python3
"""渲染 Wave P 14 臂 {W,V}x{p02,p035,p05,p08,yaw,ang,fast} push 鲁棒性队列的 SSH 命令。

人话：这个程序只做三件事——(1) ``--plan`` 打印 12 臂计划表和推荐填充顺序；(2)
``--render-stage probe|science --render-job <id> --pod <pod1|pod2> --gpu <0|1|2>``
输出单臂的逐字 SSH 命令给人工核对后执行（本波不写死 pod/gpu，空槽即填，目标卡
在渲染时注入）；(3) ``--checklist`` 输出发射前依赖核对单。它自己绝不 SSH、绝不
发信号、绝不写远端。所有校验 fail-closed：YAML 缺键、占位 commit、C+S0 配方漂移、
push 档位漂移、ang_axes 与角速度幅度错配、非法 pod/gpu、run_name 重复，都直接拒绝。

温度/稳定配方固定为矩阵 C+S0（无 push 对照 = 矩阵 w_c_s0/v_c_s0，不重复买）。
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


DEFAULT_QUEUE = Path("configs/phase1_push_robustness_20260721.yaml")
EXPECTED_QUEUE_ID = "phase1_push_robustness_20260721"
EXPECTED_NAMESPACE = "/workspace/codexschema/phase1_push_robustness_20260721"
EXPECTED_CHECKOUT = "/workspace/codexschema/nohope_push_20260721"
PLACEHOLDER_COMMIT = "PENDING_EXACT_COMMIT"
KIT_BOOT_LOCK = "/workspace/bin/kit_boot_lock.sh"
PARENT_ITERATION = 6700
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
JOB_ID = re.compile(r"^(w|v)_(p02|p035|p05|p08|yaw|ang|fast)$")
HYDRA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

PUSH_ORDER = ("p02", "p035", "p05", "p08", "yaw", "ang", "fast")
PARENT_ORDER = ("W", "V")
VALID_GPUS = (0, 1, 2)
# 同卡在跑 compute 进程必须 < 4 才允许发射（逐字继承矩阵预检）。
MAX_PROCS_PER_GPU = 4

# 人话：空槽即填的冻结顺序——parent 与档位交错，前 6 位覆盖全部 6 档。
LAUNCH_ORDER = (
    "w_p02", "v_p035", "w_p05", "v_yaw", "w_ang", "v_fast",
    "w_p035", "v_p02", "w_yaw", "v_p05", "w_fast", "v_ang",
    "w_p08", "v_p08",
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


def _push_overrides(interval: str, xy: str, ang: str, axes: str) -> list[str]:
    # 键名与工作树 train.py 的 _PUSH_KEYS 白名单逐字一致：
    # ("enable", "interval_range_s", "vel_xy_mps", "ang_vel_radps", "ang_axes")。
    # vel_xy_mps 由 training_contract.push_robot_event_block 单源展开成 x/y 对称
    # 区间（z 永不推）；ang_axes 选轴（none|yaw|rpy），ang_vel_radps 给幅度。
    # 渲染前仍须 grep 远端 checkout 在 exact commit 上的 train.py 再核对一遍。
    return [
        "++task.push.enable=true",
        f"++task.push.interval_range_s={interval}",
        f"++task.push.vel_xy_mps={xy}",
        f"++task.push.ang_vel_radps={ang}",
        f"++task.push.ang_axes={axes}",
    ]


EXPECTED_PUSH = {
    "p02": _push_overrides("[5.0,15.0]", "0.2", "0.0", "none"),
    "p035": _push_overrides("[5.0,15.0]", "0.35", "0.0", "none"),
    "p05": _push_overrides("[5.0,15.0]", "0.5", "0.0", "none"),
    "yaw": _push_overrides("[5.0,15.0]", "0.35", "0.5", "yaw"),
    "ang": _push_overrides("[5.0,15.0]", "0.35", "0.5", "rpy"),
    "fast": _push_overrides("[1.0,3.0]", "0.35", "0.0", "none"),
    # 2026-07-21 追加：±0.5 在 W 父本零摔，补 ±0.8 上界框住"必须应对"区（capture point ~26 cm）。
    "p08": _push_overrides("[5.0,15.0]", "0.8", "0.0", "none"),
}
ALLOWED_XY_MAGNITUDES = (0.2, 0.35, 0.5, 0.8)
ALLOWED_ANGULAR_MAGNITUDES = (0.0, 0.5)
# train.py/training_contract 的合法轴选择（PUSH_ROBOT_ANG_AXES 同款）。
ALLOWED_ANG_AXES = ("none", "yaw", "rpy")

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
    "max_iterations": 10001, "save_interval": 100,
    "milestone_offsets_from_parent": [200, 500, 1000, 2000, 4000, 6000, 10000],
    "absolute_milestones": [6900, 7200, 7700, 8700, 10700, 12700, 16700],
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

# science 阶段命令必须含的键（值另行断言）。
REQUIRED_SCIENCE_KEYS = (
    "checkpoint_path", "checkpoint_tolerant", "checkpoint_allow_missing_contract",
    "checkpoint_allow_contract_mismatch", "seed", "num_envs",
    "algo.runner.num_steps_per_env", "max_iterations", "algo.runner.save_interval",
    "run_name", "device", "logger", "task.push.enable",
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


def _override_range(overrides: Sequence[str], key: str, label: str) -> tuple[float, float]:
    for raw in overrides:
        if raw.lstrip("+").split("=", 1)[0] == key:
            value = yaml.safe_load(raw.split("=", 1)[1])
            pair = _list(value, f"{label}.{key}")
            if len(pair) != 2:
                raise QueueError(f"{label}.{key} must be a [lo,hi] pair")
            lo = _finite_number(pair[0], f"{label}.{key}[0]")
            hi = _finite_number(pair[1], f"{label}.{key}[1]")
            return lo, hi
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


def _validate_recipe(recipe: dict[str, Any]) -> None:
    """C+S0 配方必须与矩阵 yaml 的 C 档/S0 档逐字一致（防漂移）。"""

    _exact_keys(recipe, {"temporal_c", "stability_s0"}, "recipe")
    temporal = _mapping(recipe["temporal_c"], "recipe.temporal_c")
    _exact_keys(temporal, {"human_name", "overrides"}, "recipe.temporal_c")
    _text(temporal["human_name"], "recipe.temporal_c.human_name")
    overrides = _list(temporal["overrides"], "recipe.temporal_c.overrides")
    if overrides != EXPECTED_TEMPORAL_C:
        raise QueueError(
            "recipe.temporal_c.overrides drifted from the matrix C level verbatim copy"
        )
    action_rate = _override_value(overrides, "task.rewards.action_rate_weight", "recipe.temporal_c")
    hinge = _override_value(
        overrides, "task.rewards.processed_qdes_slew_hinge_weight", "recipe.temporal_c"
    )
    if action_rate != -0.1 or hinge != 0.0:
        raise QueueError("recipe.temporal_c weights must be exactly C: -0.10 / slew 0")

    stability = _mapping(recipe["stability_s0"], "recipe.stability_s0")
    _exact_keys(stability, {"human_name", "overrides"}, "recipe.stability_s0")
    _text(stability["human_name"], "recipe.stability_s0.human_name")
    overrides = _list(stability["overrides"], "recipe.stability_s0.overrides")
    if overrides != EXPECTED_STABILITY_S0:
        raise QueueError(
            "recipe.stability_s0.overrides drifted from the matrix S0 level verbatim copy"
        )
    for key in (
        "task.rewards.post_swing_settle_debt_weight",
        "task.rewards.lower_body_stability_bundle_weight",
        "task.rewards.lower_body_pose_imitation_weight",
    ):
        if _override_value(overrides, key, "recipe.stability_s0") != 0.0:
            raise QueueError("recipe.stability_s0 must keep all three mechanism weights at 0")
    for raw in overrides:
        if raw.lstrip("+").split("=", 1)[0].endswith("_probe_weight"):
            raise QueueError(
                "recipe.stability_s0 must not pass *_probe_weight: train.py has no such "
                "CLI key (probes are auto-forced to 1.0 by the explicit weight keys)"
            )


def _validate_push(name: str, mechanism: dict[str, Any]) -> None:
    _exact_keys(mechanism, {"human_name", "overrides"}, f"mechanisms.push.{name}")
    _text(mechanism["human_name"], f"mechanisms.push.{name}.human_name")
    overrides = _list(mechanism["overrides"], f"mechanisms.push.{name}.overrides")
    if overrides != EXPECTED_PUSH[name]:
        raise QueueError(f"mechanisms.push.{name}.overrides changed from frozen design")
    label = f"mechanisms.push.{name}"
    compiled = _override_map(overrides, label)
    if set(compiled) != {
        "task.push.enable", "task.push.interval_range_s", "task.push.vel_xy_mps",
        "task.push.ang_vel_radps", "task.push.ang_axes",
    }:
        raise QueueError(
            f"{label} must set exactly the five train.py _PUSH_KEYS whitelist keys"
        )
    if compiled["task.push.enable"] != "true":
        raise QueueError(f"{label} must set task.push.enable=true")
    lo, hi = _override_range(overrides, "task.push.interval_range_s", label)
    if not 0.0 < lo < hi:
        raise QueueError(f"{label} interval must satisfy 0 < min < max")
    xy = _override_value(overrides, "task.push.vel_xy_mps", label)
    if xy not in ALLOWED_XY_MAGNITUDES:
        raise QueueError(
            f"{label} vel_xy_mps must be one of {ALLOWED_XY_MAGNITUDES} (symmetric "
            "±a on x/y is assembled by push_robot_event_block; z is never pushed)"
        )
    ang = _override_value(overrides, "task.push.ang_vel_radps", label)
    if ang < 0.0 or ang not in ALLOWED_ANGULAR_MAGNITUDES:
        raise QueueError(
            f"{label} ang_vel_radps must be a magnitude in {ALLOWED_ANGULAR_MAGNITUDES}"
        )
    axes = compiled["task.push.ang_axes"]
    if axes not in ALLOWED_ANG_AXES:
        raise QueueError(f"{label} ang_axes must be one of {ALLOWED_ANG_AXES}")
    if (axes == "none") != (ang == 0.0):
        raise QueueError(
            f"{label} ang_axes/ang_vel_radps mismatch: axes='none' iff ang=0 "
            "(train.py contract refuses the other combinations at boot)"
        )


def _expected_jobs_order() -> list[tuple[str, str, str]]:
    """冻结的 jobs 排列：先 W 后 V，档位按 p02..fast。"""

    result = []
    for parent in PARENT_ORDER:
        for level in PUSH_ORDER:
            result.append((f"{parent.lower()}_{level}", parent, level))
    return result


def _validate_jobs(queue: dict[str, Any]) -> None:
    jobs = _list(queue["jobs"], "jobs")
    if len(jobs) != 14:
        raise QueueError("the queue must contain exactly 14 jobs")
    ids: set[str] = set()
    names: set[str] = set()
    dirs: set[str] = set()
    cells: set[tuple[str, str]] = set()
    observed: list[tuple[str, str, str]] = []
    for index, raw_job in enumerate(jobs):
        job = _mapping(raw_job, f"jobs[{index}]")
        _exact_keys(job, {"id", "parent", "push", "run_name", "run_dir"}, f"jobs[{index}]")
        job_id = _text(job["id"], f"jobs[{index}].id", safe_id=True)
        match = JOB_ID.fullmatch(job_id)
        if match is None:
            raise QueueError(
                f"jobs[{index}].id must match w|v_p02|p035|p05|yaw|ang|fast: {job_id!r}"
            )
        parent, push = match.group(1).upper(), match.group(2)
        if (job["parent"], job["push"]) != (parent, push):
            raise QueueError(f"jobs[{index}] axes do not match its id {job_id!r}")
        if job["parent"] not in PARENT_ORDER or job["push"] not in PUSH_ORDER:
            raise QueueError(f"jobs[{index}] references an unknown axis value")
        run_name = _text(job["run_name"], f"jobs[{index}].run_name", safe_id=True)
        expected_name = f"p1push_{job_id}_seed3_20260721"
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
        cell = (parent, push)
        if cell in cells:
            raise QueueError(f"push cell duplicated: {cell}")
        cells.add(cell)
        observed.append((job_id, parent, push))
    if len(cells) != 14:
        raise QueueError("parent x push coverage is incomplete")
    if observed != _expected_jobs_order():
        raise QueueError("jobs must keep the frozen W-then-V, p02..fast ordering")

    order = _list(queue["launch_order"], "launch_order")
    if order != list(LAUNCH_ORDER):
        raise QueueError(
            "launch_order must keep the frozen parent/level interleaved fill order"
        )
    if set(order) != ids:
        raise QueueError("launch_order must be a permutation of the 12 job ids")


def _validate_controls(controls: dict[str, Any]) -> None:
    _exact_keys(
        controls,
        set(EXPECTED_CONTROLS) | {"human_note"},
        "controls",
    )
    for key, expected in EXPECTED_CONTROLS.items():
        if controls[key] != expected:
            raise QueueError(
                f"controls.{key} must point at the matrix no-push baseline {expected!r}"
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
            "recipe", "mechanisms", "probe_contract", "budgets", "launch_order",
            "jobs",
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
            raise QueueError(f"assets.{key} changed from the Wave A verbatim copy")

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
    per_cell_keys = {
        "task.rewards.action_rate_weight",
        "task.rewards.processed_qdes_slew_hinge_weight",
        "task.rewards.processed_qdes_slew_hinge_margin",
        "task.rewards.processed_qdes_slew_hinge_recovery_start_s",
        "task.rewards.processed_qdes_slew_hinge_recovery_end_s",
        "task.rewards.post_swing_settle_debt_weight",
        "task.rewards.lower_body_stability_bundle_weight",
        "task.rewards.lower_body_pose_imitation_weight",
        "task.push.enable",
        "task.push.interval_range_s",
        "task.push.vel_xy_mps",
        "task.push.ang_vel_radps",
        "task.push.ang_axes",
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

    _validate_recipe(_mapping(queue["recipe"], "recipe"))

    mechanisms = _mapping(queue["mechanisms"], "mechanisms")
    _exact_keys(mechanisms, {"push"}, "mechanisms")
    push = _mapping(mechanisms["push"], "mechanisms.push")
    if set(push) != set(PUSH_ORDER):
        raise QueueError("mechanisms.push must be exactly p02, p035, p05, p08, yaw, ang, fast")
    for name, mechanism in push.items():
        _validate_push(name, _mapping(mechanism, f"mechanisms.push.{name}"))

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

    # 每臂两阶段都编译一遍：证明没有重复 Hydra 键、必带键齐全。
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
    return f"p1push_probe_{job['id']}_seed3_20260721"


def _training_argv(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str
) -> list[str]:
    if stage not in {"probe", "science"}:
        raise QueueError("stage must be probe or science")
    source = queue["source"]
    workdir = f"{source['checkout']}/{source['worktree_relative']}"
    parent = queue["parents"][job["parent"]]
    temporal = queue["recipe"]["temporal_c"]
    stability = queue["recipe"]["stability_s0"]
    push = queue["mechanisms"]["push"][job["push"]]
    budget = queue["budgets"][stage]
    argv = [
        source["python"],
        f"{workdir}/{source['trainer_relative']}",
        *queue["common"]["base_overrides"],
        queue["common"]["planner_revision_override"],
        *parent["recipe_overrides"],
        *temporal["overrides"],
        *stability["overrides"],
        *push["overrides"],
        f"motion_file={queue['assets']['motion_forehand']}",
        f"motion_file_2={queue['assets']['motion_backhand']}",
        f"++task.racket.question_bank={queue['assets']['training_question_bank']}",
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
    for key in REQUIRED_SCIENCE_KEYS:
        if key not in compiled:
            raise QueueError(f"{job['id']}.{stage} command is missing required key {key}")
    return argv


def _remote_body(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str, gpu: int
) -> str:
    commit = _require_render_commit(queue)
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


def _push_params(queue: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    overrides = queue["mechanisms"]["push"][job["push"]]["overrides"]
    label = job["id"]
    interval = _override_range(overrides, "task.push.interval_range_s", label)
    xy = _override_value(overrides, "task.push.vel_xy_mps", label)
    ang = _override_value(overrides, "task.push.ang_vel_radps", label)
    axes = _override_map(overrides, label)["task.push.ang_axes"]
    return {"interval_s": interval, "xy_mps": xy, "ang_radps": ang, "axes": axes}


def cmd_plan(queue: Mapping[str, Any]) -> str:
    commit = queue["source"]["commit"]
    commit_line = (
        "commit: 占位符 PENDING_EXACT_COMMIT（渲染被锁死，主控合并 push wiring 后填 40-hex）"
        if _commit_is_placeholder(commit)
        else f"commit: {commit}"
    )
    lines = [
        f"queue: {queue['queue_id']}  12 臂 = {{W,V}} x {{p02,p035,p05,yaw,ang,fast}}，配方固定 C+S0",
        commit_line,
        "budgets: probe 4096env x 24steps x 2it | science 4096env x 24steps x 10001it save100",
        "watchdog: boot 停滞 1800 s / 首迭代后停滞 900 s 才判死；重试只许逐字 _r2 一次",
        "对照: 不重复买——矩阵 w_c_s0/v_c_s0（同 C+S0 配方、无 push）就是无 push 基线",
        "槽位: 本波不写死 pod/gpu；矩阵谁毕业谁腾卡，渲染时 --pod/--gpu 注入目标卡",
        "",
    ]
    for job in queue["jobs"]:
        parent_human = queue["parents"][job["parent"]]["human_name"]
        push_human = queue["mechanisms"]["push"][job["push"]]["human_name"]
        params = _push_params(queue, job)
        lines.append(
            f"{job['id']:8s} {job['run_name']}\n"
            f"         人话: {parent_human} | {push_human}\n"
            f"         push: interval={params['interval_s'][0]}-{params['interval_s'][1]}s "
            f"vx=vy=±{params['xy_mps']}m/s ang=±{params['ang_radps']}rad/s "
            f"axes={params['axes']}"
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
            "0. [阻塞] source.commit 仍是占位符 PENDING_EXACT_COMMIT：主控合并 push "
            "wiring 后填入 40-hex 精确 commit，否则渲染器拒绝出命令。"
        )
    lines += [
        f"1. 远端 checkout {queue['source']['checkout']} 存在、git status 干净、"
        "HEAD == source.commit（clean detached exact commit）。",
        "2. grep 远端 checkout 的 train.py 复核 push wiring 在 exact commit 里：\n"
        "   _PUSH_KEYS 白名单必须逐字等于 (enable, interval_range_s, vel_xy_mps, "
        "ang_vel_radps, ang_axes)，即本 yaml 每臂五个 task.push.* 键（本地工作树已 "
        "grep 核对一致；若远端键名不同必须先停下重议再渲染，缺一个键则 boot 即 "
        "fail-loud；ang_axes='none' 必须配 ang_vel_radps=0，yaw|rpy 必须配 >0）。",
        f"3. 两 pod 都确认 {KIT_BOOT_LOCK} 存在且可执行；本轮不用 "
        "launch_kit_training_locked.sh 的 180 s stale 门（v8/v9 死因）。",
        "4. 五条资产路径逐一 test -f / test -d（USD、正反手 motion、题库、A3 资产树）。",
        "5. launch 前须 sha256sum 验证 parent checkpoint：\n"
        f"   W: sha256sum {parents['W']['checkpoint_path']}\n"
        f"      期望 {parents['W']['checkpoint_sha256']}\n"
        f"   V: sha256sum {parents['V']['checkpoint_path']}\n"
        f"      期望 {parents['V']['checkpoint_sha256']}",
        f"6. namespace {queue['namespace']['root']} 全新（no_clobber）；12 个 run_dir 与 "
        "12 个 probe dir 都不得已存在。",
        "7. 空槽即填：矩阵 24 格谁先毕业谁腾卡；发射前该卡 nvidia-smi compute 进程数 "
        "< 4（命令内已带预检），按 launch_order 拉最前面就绪的臂，渲染时 "
        "--render-job <id> --pod <pod> --gpu <g> 注入目标卡。",
        "8. 先跑 probe（每臂 2 个 update），自然退出后核对 run.log 出现 'Learning iteration'"
        "、无 fatal、model_6701.pt 存在，才允许发对应臂的 science。",
        "9. 发射节奏：两 pod 可并行；同 pod 内 boot 串行（kit_boot_lock 持锁），相邻两次 "
        "launch 错峰 >= 60 s。",
        "10. 日志摘要抓异常不抓预期：WARN 行必须全部进摘要（grep -n 'WARN' run.log）；"
        "q_des CLAMP ACTIVE 行必须进摘要（grep -Fn 'q_des CLAMP ACTIVE' run.log）。",
        "11. watchdog：boot 停滞 1800 s、首个迭代后停滞 900 s 才算卡死；唯一允许的重试是"
        "逐字重发一次并给 run_name 加 _r2 后缀，仍需人工核对。",
        "12. 对照不重复买：矩阵 w_c_s0/v_c_s0（p1btm_w_c_s0_seed3_20260720 / "
        "p1btm_v_c_s0_seed3_20260720）就是本波的无 push 对照，同 C+S0 配方、"
        "push_robot=None，汇总时直接对比。",
        "13. 全部 12 臂 science 里程碑（6900/7200/7700/8700/10700/12700/16700）落盘后，"
        "汇总时按诊断谱系解读：W/V contract 有意 mismatch，胜者档位须另在 exact-lineage 重跑。",
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
    parser.add_argument("--plan", action="store_true", help="打印 12 臂计划表（默认）")
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
