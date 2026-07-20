#!/usr/bin/env python3
"""渲染 24 格 {W,V}x{N,C,H}x{S0..S3} 平衡×时序矩阵的 probe/science SSH 命令。

人话：这个程序只做三件事——(1) ``--plan`` 打印 24 格计划表；(2)
``--render-stage probe|science`` 输出逐字 SSH 命令给人工核对后执行；(3)
``--checklist`` 输出发射前依赖核对单。它自己绝不 SSH、绝不发信号、绝不写远端。
所有校验 fail-closed：YAML 缺键、占位 commit、权重符号错、run_name 重复、
单卡超 4 条，都直接拒绝。

本轮是 lean runner：没有 manifest-SHA 多层审批链，主控人工核对渲染结果后执行。
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


DEFAULT_QUEUE = Path("configs/phase1_balance_temporal_matrix_20260720.yaml")
EXPECTED_QUEUE_ID = "phase1_balance_temporal_matrix_20260720"
EXPECTED_NAMESPACE = "/workspace/codexschema/phase1_balance_temporal_matrix_20260720d"
EXPECTED_CHECKOUT = "/workspace/codexschema/nohope_btm_20260720"
PLACEHOLDER_COMMIT = "PENDING_EXACT_COMMIT"
KIT_BOOT_LOCK = "/workspace/bin/kit_boot_lock.sh"
PARENT_ITERATION = 6700
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
JOB_ID = re.compile(r"^(w|v)_(n|c|h)_(s[0-3])$")
HYDRA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

TEMPORAL_ORDER = ("N", "C", "H")
STABILITY_ORDER = ("S0", "S1", "S2", "S3")
PARENT_ORDER = ("W", "V")
GPU_SLOTS = (
    ("pod1", 0), ("pod1", 1), ("pod1", 2),
    ("pod2", 0), ("pod2", 1), ("pod2", 2),
)
JOBS_PER_GPU = 4

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

_SLEW_TAIL = [
    "++task.rewards.processed_qdes_slew_hinge_margin=0.85",
    "++task.rewards.processed_qdes_slew_hinge_recovery_start_s=0.2",
    "++task.rewards.processed_qdes_slew_hinge_recovery_end_s=1.55",
]
EXPECTED_TEMPORAL = {
    "N": [
        "task.rewards.action_rate_weight=0.0",
        "++task.rewards.processed_qdes_slew_hinge_weight=0.0",
        *_SLEW_TAIL,
    ],
    "C": [
        "task.rewards.action_rate_weight=-0.1",
        "++task.rewards.processed_qdes_slew_hinge_weight=0.0",
        *_SLEW_TAIL,
    ],
    "H": [
        "task.rewards.action_rate_weight=0.0",
        "++task.rewards.processed_qdes_slew_hinge_weight=-0.25",
        *_SLEW_TAIL,
    ],
}

# 每个 S 档的三个机制 weight（settle_debt, bundle, pose_imitation）——互斥开启。
EXPECTED_STABILITY_WEIGHTS = {
    "S0": (0.0, 0.0, 0.0),
    "S1": (-0.25, 0.0, 0.0),
    "S2": (0.0, -0.25, 0.0),
    "S3": (0.0, 0.0, 0.5),
}


def _stability_overrides(settle: str, bundle: str, pose: str) -> list[str]:
    # 注意：三个机制都没有独立的 *_probe_weight CLI 键（train.py _REWARD_KEYS 白名单
    # 没收；写了会在 boot 时 fail-loud）。显式写机制 weight 键本身就会让 train.py
    # 强制把对应测量探针 weight 设为 1.0——所以每格三个 weight 都显式写。
    return [
        f"++task.rewards.post_swing_settle_debt_weight={settle}",
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
        f"++task.rewards.lower_body_stability_bundle_weight={bundle}",
        "++task.rewards.lower_body_stability_min_stance_width_m=0.22",
        "++task.rewards.lower_body_stability_stance_scale_m=0.05",
        "++task.rewards.lower_body_stability_leg_velocity_margin_radps=1.0",
        "++task.rewards.lower_body_stability_leg_velocity_scale_radps=0.5",
        "++task.rewards.lower_body_stability_support_pre_s=0.3",
        "++task.rewards.lower_body_stability_support_post_s=0.4",
        f"++task.rewards.lower_body_pose_imitation_weight={pose}",
        "++task.rewards.lower_body_pose_imitation_std=0.35",
        "++task.rewards.lower_body_pose_imitation_support_pre_s=0.3",
        "++task.rewards.lower_body_pose_imitation_support_post_s=0.4",
    ]


EXPECTED_STABILITY = {
    "S0": _stability_overrides("0.0", "0.0", "0.0"),
    "S1": _stability_overrides("-0.25", "0.0", "0.0"),
    "S2": _stability_overrides("0.0", "-0.25", "0.0"),
    "S3": _stability_overrides("0.0", "0.0", "0.5"),
}

EXPECTED_PODS = {
    "pod1": ("162.43.172.171", 18333),
    "pod2": ("162.43.172.181", 13146),
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


def _validate_temporal(name: str, mechanism: dict[str, Any]) -> None:
    _exact_keys(mechanism, {"human_name", "overrides"}, f"mechanisms.temporal.{name}")
    _text(mechanism["human_name"], f"mechanisms.temporal.{name}.human_name")
    overrides = _list(mechanism["overrides"], f"mechanisms.temporal.{name}.overrides")
    if overrides != EXPECTED_TEMPORAL[name]:
        raise QueueError(f"mechanisms.temporal.{name}.overrides changed from frozen design")
    label = f"mechanisms.temporal.{name}"
    action_rate = _override_value(overrides, "task.rewards.action_rate_weight", label)
    hinge = _override_value(
        overrides, "task.rewards.processed_qdes_slew_hinge_weight", label
    )
    if action_rate > 0.0 or hinge > 0.0:
        raise QueueError(f"{label} weight sign error: penalties must be <= 0")


def _validate_stability(name: str, mechanism: dict[str, Any]) -> None:
    _exact_keys(mechanism, {"human_name", "overrides"}, f"mechanisms.stability.{name}")
    _text(mechanism["human_name"], f"mechanisms.stability.{name}.human_name")
    overrides = _list(mechanism["overrides"], f"mechanisms.stability.{name}.overrides")
    if overrides != EXPECTED_STABILITY[name]:
        raise QueueError(f"mechanisms.stability.{name}.overrides changed from frozen design")
    label = f"mechanisms.stability.{name}"
    settle = _override_value(overrides, "task.rewards.post_swing_settle_debt_weight", label)
    bundle = _override_value(
        overrides, "task.rewards.lower_body_stability_bundle_weight", label
    )
    pose = _override_value(
        overrides, "task.rewards.lower_body_pose_imitation_weight", label
    )
    if settle > 0.0 or bundle > 0.0:
        raise QueueError(f"{label} weight sign error: settle debt/bundle must be <= 0")
    if pose < 0.0:
        raise QueueError(f"{label} weight sign error: pose imitation must be >= 0")
    if (settle, bundle, pose) != EXPECTED_STABILITY_WEIGHTS[name]:
        raise QueueError(f"{label} weights differ from frozen design")
    if sum(1 for value in (settle, bundle, pose) if value != 0.0) > 1:
        raise QueueError(f"{label} enables more than one mechanism (S1/S2/S3 are exclusive)")
    for raw in overrides:
        if raw.lstrip("+").split("=", 1)[0].endswith("_probe_weight"):
            raise QueueError(
                f"{label} must not pass *_probe_weight: train.py has no such CLI key "
                "(probes are auto-forced to 1.0 by the explicit weight keys)"
            )
    start = _override_value(overrides, "task.rewards.post_swing_settle_recovery_start_s", label)
    end = _override_value(overrides, "task.rewards.post_swing_settle_recovery_end_s", label)
    if not 0.0 <= start < end:
        raise QueueError(f"{label} settle recovery window must satisfy 0 <= start < end")


def _expected_assignment() -> list[tuple[str, str, str, str, int]]:
    """按 (S, T, parent) 排序后轮流发到六个 GPU 槽的冻结分配。"""

    cells = [
        (stability, temporal, parent)
        for stability in STABILITY_ORDER
        for temporal in TEMPORAL_ORDER
        for parent in PARENT_ORDER
    ]
    result = []
    for index, (stability, temporal, parent) in enumerate(cells):
        pod, gpu = GPU_SLOTS[index % len(GPU_SLOTS)]
        job_id = f"{parent.lower()}_{temporal.lower()}_{stability.lower()}"
        result.append((job_id, temporal, stability, pod, gpu))
    return result


def _validate_jobs(queue: dict[str, Any]) -> None:
    jobs = _list(queue["jobs"], "jobs")
    if len(jobs) != 24:
        raise QueueError("the queue must contain exactly 24 jobs")
    ids: set[str] = set()
    names: set[str] = set()
    dirs: set[str] = set()
    cells: set[tuple[str, str, str]] = set()
    per_slot: dict[tuple[str, int], int] = {}
    observed: list[tuple[str, str, str, str, int]] = []
    for index, raw_job in enumerate(jobs):
        job = _mapping(raw_job, f"jobs[{index}]")
        _exact_keys(
            job,
            {"id", "parent", "temporal", "stability", "pod", "gpu", "run_name", "run_dir"},
            f"jobs[{index}]",
        )
        job_id = _text(job["id"], f"jobs[{index}].id", safe_id=True)
        match = JOB_ID.fullmatch(job_id)
        if match is None:
            raise QueueError(f"jobs[{index}].id must match w|v_n|c|h_s0..s3: {job_id!r}")
        parent, temporal, stability = (
            match.group(1).upper(), match.group(2).upper(), match.group(3).upper()
        )
        if (job["parent"], job["temporal"], job["stability"]) != (parent, temporal, stability):
            raise QueueError(f"jobs[{index}] axes do not match its id {job_id!r}")
        if job["parent"] not in PARENT_ORDER or job["temporal"] not in TEMPORAL_ORDER \
                or job["stability"] not in STABILITY_ORDER:
            raise QueueError(f"jobs[{index}] references an unknown axis value")
        run_name = _text(job["run_name"], f"jobs[{index}].run_name", safe_id=True)
        expected_name = f"p1btm_{job_id}_seed3_20260720"
        if run_name != expected_name:
            raise QueueError(f"jobs[{index}].run_name must be {expected_name!r}")
        run_dir = _remote_path(job["run_dir"], f"jobs[{index}].run_dir")
        if run_dir != f"{EXPECTED_NAMESPACE}/runs/{job_id}":
            raise QueueError(f"jobs[{index}].run_dir must be <root>/runs/{job_id}")
        if job["pod"] not in EXPECTED_PODS:
            raise QueueError(f"jobs[{index}].pod is unknown")
        if type(job["gpu"]) is not int or job["gpu"] not in (0, 1, 2):
            raise QueueError(f"jobs[{index}].gpu must be 0, 1, or 2")
        if job_id in ids or run_name in names or run_dir in dirs:
            raise QueueError("duplicate job id, run_name, or run_dir")
        ids.add(job_id)
        names.add(run_name)
        dirs.add(run_dir)
        cell = (parent, temporal, stability)
        if cell in cells:
            raise QueueError(f"matrix cell duplicated: {cell}")
        cells.add(cell)
        slot = (job["pod"], job["gpu"])
        per_slot[slot] = per_slot.get(slot, 0) + 1
        if per_slot[slot] > JOBS_PER_GPU:
            raise QueueError(f"GPU slot {slot} holds more than {JOBS_PER_GPU} jobs")
        observed.append((job_id, temporal, stability, job["pod"], job["gpu"]))
    if len(cells) != 24:
        raise QueueError("T x S x parent coverage is incomplete")
    if set(per_slot) != set(GPU_SLOTS) or any(
        count != JOBS_PER_GPU for count in per_slot.values()
    ):
        raise QueueError("every one of the six GPUs must hold exactly four jobs")
    if observed != _expected_assignment():
        raise QueueError(
            "jobs must keep the frozen (S,T,parent)-sorted round-robin GPU assignment"
        )


def _validate_queue(queue: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(
        queue,
        {
            "schema_version", "queue_id", "purpose", "simulation_only",
            "real_robot_authorized", "launch_authorized_by_default",
            "formal_exact_eligible", "evidence_class", "ssh", "pods", "namespace",
            "source", "watchdog", "assets", "parents", "common", "mechanisms",
            "probe_contract", "budgets", "jobs",
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
    _exact_keys(mechanisms, {"temporal", "stability"}, "mechanisms")
    temporal = _mapping(mechanisms["temporal"], "mechanisms.temporal")
    if set(temporal) != set(TEMPORAL_ORDER):
        raise QueueError("mechanisms.temporal must be exactly N, C, H")
    for name, mechanism in temporal.items():
        _validate_temporal(name, _mapping(mechanism, f"mechanisms.temporal.{name}"))
    stability = _mapping(mechanisms["stability"], "mechanisms.stability")
    if set(stability) != set(STABILITY_ORDER):
        raise QueueError("mechanisms.stability must be exactly S0, S1, S2, S3")
    for name, mechanism in stability.items():
        _validate_stability(name, _mapping(mechanism, f"mechanisms.stability.{name}"))

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

    # 每格两阶段都编译一遍：证明没有重复 Hydra 键、必带键齐全。
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
    return f"p1btm_probe_{job['id']}_seed3_20260720"


def _training_argv(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str
) -> list[str]:
    if stage not in {"probe", "science"}:
        raise QueueError("stage must be probe or science")
    source = queue["source"]
    workdir = f"{source['checkout']}/{source['worktree_relative']}"
    parent = queue["parents"][job["parent"]]
    temporal = queue["mechanisms"]["temporal"][job["temporal"]]
    stability = queue["mechanisms"]["stability"][job["stability"]]
    budget = queue["budgets"][stage]
    argv = [
        source["python"],
        f"{workdir}/{source['trainer_relative']}",
        *queue["common"]["base_overrides"],
        queue["common"]["planner_revision_override"],
        *parent["recipe_overrides"],
        *temporal["overrides"],
        *stability["overrides"],
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


def _remote_body(queue: Mapping[str, Any], job: Mapping[str, Any], stage: str) -> str:
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
        f"env CUDA_VISIBLE_DEVICES={job['gpu']} "
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
gpu_output=$(nvidia-smi -i {job['gpu']} --query-compute-apps=pid --format=csv,noheader,nounits)
test -z "$gpu_output"
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


def _ssh_argv(queue: Mapping[str, Any], job: Mapping[str, Any], stage: str) -> list[str]:
    pod = queue["pods"][job["pod"]]
    remote = _remote_body(queue, job, stage)
    key = os.path.expanduser(str(queue["ssh"]["key"]))
    return [
        "ssh", "-i", key, "-p", str(pod["port"]),
        "-o", "BatchMode=yes", f"root@{pod['host']}",
        f"bash -lc {shlex.quote(remote)}",
    ]


def render_command(queue: Mapping[str, Any], job: Mapping[str, Any], stage: str) -> str:
    return shlex.join(_ssh_argv(queue, job, stage))


def _job_weights(queue: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, float]:
    temporal = queue["mechanisms"]["temporal"][job["temporal"]]["overrides"]
    stability = queue["mechanisms"]["stability"][job["stability"]]["overrides"]
    label = job["id"]
    return {
        "action_rate": _override_value(temporal, "task.rewards.action_rate_weight", label),
        "qdes_slew_hinge": _override_value(
            temporal, "task.rewards.processed_qdes_slew_hinge_weight", label
        ),
        "settle_debt": _override_value(
            stability, "task.rewards.post_swing_settle_debt_weight", label
        ),
        "stability_bundle": _override_value(
            stability, "task.rewards.lower_body_stability_bundle_weight", label
        ),
        "pose_imitation": _override_value(
            stability, "task.rewards.lower_body_pose_imitation_weight", label
        ),
    }


def cmd_plan(queue: Mapping[str, Any]) -> str:
    commit = queue["source"]["commit"]
    commit_line = (
        "commit: 占位符 PENDING_EXACT_COMMIT（渲染被锁死，主控合并后填 40-hex）"
        if _commit_is_placeholder(commit)
        else f"commit: {commit}"
    )
    lines = [
        f"queue: {queue['queue_id']}  24 格 = {{W,V}} x {{N,C,H}} x {{S0..S3}}",
        commit_line,
        "budgets: probe 4096env x 24steps x 2it | science 4096env x 24steps x 10001it save100",
        "watchdog: boot 停滞 1800 s / 首迭代后停滞 900 s 才判死；重试只许逐字 _r2 一次",
        "",
    ]
    for job in queue["jobs"]:
        parent_human = queue["parents"][job["parent"]]["human_name"]
        temporal_human = queue["mechanisms"]["temporal"][job["temporal"]]["human_name"]
        stability_human = queue["mechanisms"]["stability"][job["stability"]]["human_name"]
        weights = _job_weights(queue, job)
        weight_text = " ".join(f"{key}={value}" for key, value in weights.items())
        lines.append(
            f"{job['id']:8s} {job['pod']}/gpu{job['gpu']}  {job['run_name']}\n"
            f"         人话: {parent_human} | {temporal_human} | {stability_human}\n"
            f"         {weight_text}"
        )
    return "\n".join(lines)


def cmd_checklist(queue: Mapping[str, Any]) -> str:
    parents = queue["parents"]
    commit = queue["source"]["commit"]
    lines = ["发射前依赖核对单（全部人工执行，本程序不 SSH）", ""]
    if _commit_is_placeholder(commit):
        lines.append(
            "0. [阻塞] source.commit 仍是占位符 PENDING_EXACT_COMMIT：主控合并所有并行"
            "实现后填入 40-hex 精确 commit，否则渲染器拒绝出命令。"
        )
    lines += [
        f"1. 远端 checkout {queue['source']['checkout']} 存在、git status 干净、"
        "HEAD == source.commit（clean detached exact commit）。",
        "2. grep 远端 checkout 的 train.py 复核 S1 实现在 exact commit 里：\n"
        "   post_swing_settle_debt_weight 与全部 post_swing_settle_* 参数键都在 "
        "_REWARD_KEYS 白名单里，且三个机制的 probe.weight=1.0 由显式 weight 键自动强制"
        "（本地 worktree 已 grep 验证；缺一个键则所有 24 格 boot 即 fail-loud）。",
        f"3. 两 pod 都确认 {KIT_BOOT_LOCK} 存在且可执行；本轮不用 "
        "launch_kit_training_locked.sh 的 180 s stale 门（v8/v9 死因）。",
        "4. 五条资产路径逐一 test -f / test -d（USD、正反手 motion、题库、A3 资产树）。",
        "5. launch 前须 sha256sum 验证 parent checkpoint：\n"
        f"   W: sha256sum {parents['W']['checkpoint_path']}\n"
        f"      期望 {parents['W']['checkpoint_sha256']}\n"
        f"   V: sha256sum {parents['V']['checkpoint_path']}\n"
        f"      期望 {parents['V']['checkpoint_sha256']}",
        f"6. namespace {queue['namespace']['root']} 全新（no_clobber）；24 个 run_dir 与 "
        "24 个 probe dir 都不得已存在。",
        "7. 每张目标卡 nvidia-smi 零 compute 进程后才允许发射该卡。",
        "8. 先跑 probe（每格 2 个 update），自然退出后核对 run.log 出现 'Learning iteration'"
        "、无 fatal、model_6701.pt 存在，才允许发对应格的 science。",
        "9. 发射节奏：两 pod 可并行；同 pod 内 boot 串行（kit_boot_lock 持锁），相邻两次 "
        "launch 错峰 >= 60 s。",
        "10. 日志摘要抓异常不抓预期：WARN 行必须全部进摘要（grep -n 'WARN' run.log）；"
        "q_des CLAMP ACTIVE 行必须进摘要（grep -Fn 'q_des CLAMP ACTIVE' run.log）。",
        "11. watchdog：boot 停滞 1800 s、首个迭代后停滞 900 s 才算卡死；唯一允许的重试是"
        "逐字重发一次并给 run_name 加 _r2 后缀，仍需人工核对。",
        "12. 全部 24 格 science 里程碑（6900/7200/7700/8700/10700/12700/16700）落盘后，"
        "汇总时按诊断谱系解读：W/V contract 有意 mismatch，胜者机制须另在 exact-lineage 重跑。",
    ]
    return "\n".join(lines)


def cmd_render(queue: Mapping[str, Any], stage: str, job_selector: str) -> str:
    if stage not in {"probe", "science"}:
        raise QueueError("--render-stage must be probe or science")
    if job_selector == "all":
        jobs = list(queue["jobs"])
    else:
        jobs = [_job_by_id(queue, job_selector)]
    blocks = []
    for job in jobs:
        blocks.append(
            f"# {job['id']} {stage} {job['pod']}/gpu{job['gpu']} "
            f"run_name={_stage_run_name(job, stage)}\n"
            + render_command(queue, job, stage)
        )
    return "\n\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--plan", action="store_true", help="打印 24 格计划表（默认）")
    parser.add_argument("--render-stage", choices=("probe", "science"))
    parser.add_argument("--job", default=None, help="job id 或 all（配合 --render-stage）")
    parser.add_argument("--checklist", action="store_true", help="打印发射前依赖核对单")
    args = parser.parse_args(argv)
    try:
        queue = load_queue(args.queue)
        if args.render_stage is not None:
            if args.job is None:
                raise QueueError("--render-stage requires --job <id>|all")
            print(cmd_render(queue, args.render_stage, args.job))
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
