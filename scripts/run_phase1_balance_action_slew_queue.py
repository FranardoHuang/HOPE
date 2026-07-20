#!/usr/bin/env python3
"""Validate and render the W/V processed-qdes slew ablation queue.

The default invocation is strictly NO-LAUNCH: it validates the checked-in YAML
and prints a six-cell plan without SSH or trainer commands.  Passing
``--authorize-launch`` only renders commands for a human/parent orchestrator;
this program uses read-only local Git subprocesses to verify ``origin/main``
authority, but has no SSH, signal, trainer, or remote-write execution path.
Rendering is fail-closed behind a separately reviewed launch manifest and the
current authoritative NOW claim.
The training stage additionally consumes six cryptographically bound probe
receipts; a boolean approval flag is deliberately insufficient.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_QUEUE = Path("configs/phase1_balance_action_slew_20260720.yaml")
QUEUE_CONFIG_RELATIVE = "configs/phase1_balance_action_slew_20260720.yaml"
QUEUE_RUNNER_RELATIVE = "scripts/run_phase1_balance_action_slew_queue.py"
LAUNCH_MANIFEST_RELATIVE = "configs/phase1_balance_action_slew_launch_manifest_20260720.json"
AUTHORITY_NOW_TITLE = "- **[11｜P1] 稳定机制 Wave A/B。**"
AUTHORITY_OWNER_EXECUTOR = "责任人 franco；执行者 Codex；执行分支"
AUTHORITY_BRANCH = "Franco_codex/balance-ablation-round-20260720"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
HYDRA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_QUEUE_ID = "phase1_balance_action_slew_20260720"
EXPECTED_NAMESPACE = "/workspace/codexschema/phase1_balance_action_slew_v4_20260720"
EXPECTED_SOURCE = "/workspace/codexschema/nohope_balance_action_slew_20260720"
EXPECTED_REMOTE_SOURCE_COMMIT = "54c9a62656f0e60e5bb41cbcfa0e5a972b793906"
PARENT_ITERATION = 6700
PROBE_NUM_ENVS = 4096
NUM_STEPS_PER_ENV = 24
EXPECTED_SAMPLES_PER_UPDATE = PROBE_NUM_ENVS * NUM_STEPS_PER_ENV
EXPECTED_JOBS = {
    "w_c": ("W", "C", "pod1", 0),
    "w_n": ("W", "N", "pod1", 1),
    "w_h": ("W", "H", "pod1", 2),
    "v_c": ("V", "C", "pod2", 0),
    "v_n": ("V", "N", "pod2", 1),
    "v_h": ("V", "H", "pod2", 2),
}
EXPECTED_MECHANISMS = {
    "C": (-0.10, 0.0),
    "N": (0.0, 0.0),
    "H": (0.0, -0.25),
}
FACTOR_KEYS = {
    "task.rewards.action_rate_weight",
    "task.rewards.processed_qdes_slew_hinge_weight",
    "task.rewards.processed_qdes_slew_hinge_margin",
    "task.rewards.processed_qdes_slew_hinge_recovery_start_s",
    "task.rewards.processed_qdes_slew_hinge_recovery_end_s",
}
EXPECTED_STARTUP_CHECKS = [
    "reviewed_launch_manifest_binds_source_config_runner_and_inputs",
    "source_checkout_exists_and_is_clean",
    "required_source_and_asset_paths_are_regular",
    "parent_checkpoint_is_model_6700_full_state_and_finite",
    "assigned_gpu_has_zero_compute_processes",
    "hydra_configuration_resolves_without_kit",
    "output_directory_is_new_and_no_clobber",
    "queue_claim_is_exclusively_published_before_trainer_start",
    "first_training_iteration_appears",
    "fatal_log_scan_is_clean",
    "probe_natural_exit_terminal_checkpoint_counter_consistency_and_gpu_release",
]
PROBE_COUNTER_TAGS = {
    "observed_sample_count": "Live/processed_qdes_slew/observed_sample_count",
    "previous_qdes_valid_sample_count": "Live/processed_qdes_slew/previous_qdes_valid_sample_count",
    "previous_qdes_invalid_first_step_sample_count": "Live/processed_qdes_slew/previous_qdes_invalid_first_step_sample_count",
    "recovery_eligible_sample_count": "Live/processed_qdes_slew/recovery_eligible_sample_count",
    "reward_enabled_eligible_sample_count": "Live/processed_qdes_slew/reward_enabled_eligible_sample_count",
    "tail_active_sample_count": "Live/processed_qdes_slew/tail_active_sample_count",
    "above_margin_joint_count": "Live/processed_qdes_slew/above_margin_joint_count",
    "gated_tail_value_sum": "Live/processed_qdes_slew/gated_tail_value_sum",
    "racket_swing_outcome_count": "Live/racket_target/swing_outcome_count",
    "racket_swing_completion_count": "Live/racket_target/swing_completion_count",
    "racket_physical_fall_count": "Live/racket_target/physical_fall_count",
    "racket_pre_strike_physical_fall_count": "Live/racket_target/pre_strike_physical_fall_count",
    "racket_post_strike_physical_fall_count": "Live/racket_target/post_strike_physical_fall_count",
    "racket_strike_opportunity_count": "Live/racket_target/strike_opportunity_count",
    "racket_virtual_legal_return_count": "Live/racket_target/virtual_legal_return_count",
    "racket_ready_tilt_eligible_sample_count": "Live/racket_target/ready_tilt_eligible_sample_count",
    "racket_ready_tilt_rad_sum": "Live/racket_target/ready_tilt_rad_sum",
    "racket_ready_nonfinite_value_count": "Live/racket_target/ready_nonfinite_value_count",
    "qdot_observed_sample_count": "Live/qdot/observed_sample_count",
    "qdot_excess_sample_count": "Live/qdot/excess_sample_count",
    "qdot_normalized_excess_square_sum": "Live/qdot/normalized_excess_square_sum",
}
PROBE_COUNTERS = tuple(PROBE_COUNTER_TAGS)
PROBE_FLOAT_COUNTERS = frozenset(
    {
        "gated_tail_value_sum",
        "racket_ready_tilt_rad_sum",
        "qdot_normalized_excess_square_sum",
    }
)
PROBE_STEPS = (6700, 6701)
EXPECTED_PROBE_RUNTIME = {
    "normal_exit": True,
    "exit_code": 0,
    "fatal_log_scan_clean": True,
    "leader_absent": True,
    "process_group_absent": True,
    "gpu_released": True,
}


class QueueError(RuntimeError):
    """The queue or requested command-generation operation is unsafe."""


class LoadedQueue(dict[str, Any]):
    """Validated queue carrying the exact local bytes used for authorization."""

    source_path: Path


@dataclass(frozen=True)
class LaunchManifest:
    envelope: dict[str, Any]
    content: dict[str, Any]
    file_sha256: str
    content_sha256: str
    path: Path


def _git_read(repo_root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise QueueError(
            f"origin/main authority Git check failed ({' '.join(args)}): {detail}"
        )
    return completed.stdout


def _authority_now_entry(text: str) -> str:
    start = text.find(AUTHORITY_NOW_TITLE)
    if start < 0:
        raise QueueError("Wave A/B claim is missing from origin/main NOW")
    end = text.find("\n- **[", start + len(AUTHORITY_NOW_TITLE))
    entry = text[start:] if end < 0 else text[start:end]
    if AUTHORITY_OWNER_EXECUTOR not in entry or AUTHORITY_BRANCH not in entry:
        raise QueueError(
            "Wave A/B owner, executor, and branch are not bound in one origin/main NOW entry"
        )
    return entry


def _validate_origin_main_launch_authority(
    queue: Mapping[str, Any], manifest: LaunchManifest
) -> dict[str, str]:
    """Fail closed unless this exact renderer is authorized by fetched origin/main."""

    repo_root = Path(__file__).resolve().parents[1]
    head = _git_read(repo_root, "rev-parse", "HEAD").decode().strip()
    origin_main = _git_read(
        repo_root, "rev-parse", "refs/remotes/origin/main"
    ).decode().strip()
    if head != origin_main:
        raise QueueError(
            "authorized rendering requires HEAD == fetched origin/main; run git fetch origin main"
        )
    tracked = (
        QUEUE_CONFIG_RELATIVE,
        QUEUE_RUNNER_RELATIVE,
        LAUNCH_MANIFEST_RELATIVE,
        "docs/NOW.md",
    )
    dirty = _git_read(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        *tracked,
    )
    if dirty:
        raise QueueError("origin/main launch-authority files have tracked worktree changes")
    now_raw = _git_read(repo_root, "show", "refs/remotes/origin/main:docs/NOW.md")
    try:
        now_text = now_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QueueError("origin/main NOW is not UTF-8") from exc
    entry = _authority_now_entry(now_text)
    expected_files = {
        QUEUE_CONFIG_RELATIVE: manifest.content["queue_files"]["config"]["sha256"],
        QUEUE_RUNNER_RELATIVE: manifest.content["queue_files"]["runner"]["sha256"],
        LAUNCH_MANIFEST_RELATIVE: manifest.file_sha256,
    }
    for relative, expected_sha in expected_files.items():
        tracked_raw = _git_read(
            repo_root, "show", f"refs/remotes/origin/main:{relative}"
        )
        if hashlib.sha256(tracked_raw).hexdigest() != expected_sha:
            raise QueueError(f"origin/main tracked bytes do not match authority: {relative}")
    if queue["queue_id"] not in entry:
        raise QueueError("origin/main Wave A/B claim does not bind the exact queue id")
    return {
        "origin_main_commit": origin_main,
        "now_entry_sha256": hashlib.sha256(entry.encode("utf-8")).hexdigest(),
        "human_owner": "franco",
        "executor": "Codex",
        "branch": AUTHORITY_BRANCH,
    }


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


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QueueError("value is not canonical finite JSON") from exc
    return encoded


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_document(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _stat_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _read_regular_bytes(path: Path, label: str) -> bytes:
    """Read a stable regular non-symlink file without following a final link."""

    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise QueueError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise QueueError(f"{label} must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise QueueError(f"cannot safely open {label}: {path}") from exc
    try:
        opened = os.fstat(fd)
        if _stat_signature(opened) != _stat_signature(before):
            raise QueueError(f"{label} changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after_path = path.lstat()
    except FileNotFoundError as exc:
        raise QueueError(f"{label} vanished while reading: {path}") from exc
    if (
        _stat_signature(before) != _stat_signature(after_fd)
        or _stat_signature(before) != _stat_signature(after_path)
    ):
        raise QueueError(f"{label} changed while reading: {path}")
    return b"".join(chunks)


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QueueError(f"{label} is not JSON: {path}") from exc
    value = _mapping(value, label)
    if raw != _json_document(value):
        raise QueueError(f"{label} must use canonical JSON plus one newline: {path}")
    return value, raw


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or not SHA256.fullmatch(value):
        raise QueueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _require_commit(value: Any, label: str) -> str:
    if type(value) is not str or not COMMIT.fullmatch(value) or value == "0" * 40:
        raise QueueError(f"{label} must be a non-placeholder 40-character commit")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise QueueError(f"{label} must be a list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise QueueError(f"{label} keys differ: missing={missing}, extra={extra}")


def _text(value: Any, label: str, *, safe_id: bool = False) -> str:
    if not isinstance(value, str) or not value or any(ord(ch) < 32 for ch in value):
        raise QueueError(f"{label} must be non-empty printable text")
    if safe_id and not SAFE_ID.fullmatch(value):
        raise QueueError(f"{label} must be a safe identifier")
    return value


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


def _nonpositive(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number > 0.0:
        raise QueueError(f"{label} must be <= 0 (positive values reward slew)")
    return number


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


def _validate_parent(name: str, parent: dict[str, Any]) -> None:
    _exact_keys(
        parent,
        {
            "human_name",
            "checkpoint_iteration",
            "checkpoint_path",
            "hard_contract_path",
            "transfer_mode",
            "checkpoint_tolerant",
            "checkpoint_allow_missing_contract",
            "checkpoint_allow_contract_mismatch",
            "descendant_formal_exact_eligible",
            "recipe_overrides",
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
    if parent["transfer_mode"] != "full_policy_value_optimizer_normalizer":
        raise QueueError(f"parents.{name} must request full-state resume")
    expected_bools = {
        "checkpoint_tolerant": False,
        "checkpoint_allow_missing_contract": False,
        "checkpoint_allow_contract_mismatch": True,
        "descendant_formal_exact_eligible": False,
    }
    for key, expected in expected_bools.items():
        if parent[key] is not expected:
            raise QueueError(f"parents.{name}.{key} must be {expected!r}")
    overrides = _override_map(
        _list(parent["recipe_overrides"], f"parents.{name}.recipe_overrides"),
        f"parents.{name}.recipe_overrides",
    )
    expected_keys = {
        "task.rewards.racket_position_weight",
        "task.rewards.racket_velocity_weight",
        "task.rewards.racket_normal_weight",
        "task.rewards.foot_orientation_weight",
        "task.rewards.prestrike_upright_weight",
        "task.rewards.free_non_striking_arm_mimic",
    }
    if set(overrides) != expected_keys or set(overrides) & FACTOR_KEYS:
        raise QueueError(f"parents.{name}.recipe_overrides changed scientific axes")


def _validate_mechanism(name: str, mechanism: dict[str, Any]) -> None:
    _exact_keys(
        mechanism,
        {
            "human_name",
            "dense_action_rate_weight",
            "processed_qdes_slew_hinge_weight",
            "processed_qdes_slew_hinge_margin",
            "processed_qdes_slew_hinge_recovery_start_s",
            "processed_qdes_slew_hinge_recovery_end_s",
        },
        f"mechanisms.{name}",
    )
    _text(mechanism["human_name"], f"mechanisms.{name}.human_name")
    dense = _nonpositive(
        mechanism["dense_action_rate_weight"],
        f"mechanisms.{name}.dense_action_rate_weight",
    )
    hinge = _nonpositive(
        mechanism["processed_qdes_slew_hinge_weight"],
        f"mechanisms.{name}.processed_qdes_slew_hinge_weight",
    )
    expected_dense, expected_hinge = EXPECTED_MECHANISMS[name]
    if dense != expected_dense or hinge != expected_hinge:
        raise QueueError(
            f"mechanisms.{name} must be dense={expected_dense}, hinge={expected_hinge}"
        )
    margin = _finite_number(
        mechanism["processed_qdes_slew_hinge_margin"],
        f"mechanisms.{name}.processed_qdes_slew_hinge_margin",
    )
    start = _finite_number(
        mechanism["processed_qdes_slew_hinge_recovery_start_s"],
        f"mechanisms.{name}.processed_qdes_slew_hinge_recovery_start_s",
    )
    end = _finite_number(
        mechanism["processed_qdes_slew_hinge_recovery_end_s"],
        f"mechanisms.{name}.processed_qdes_slew_hinge_recovery_end_s",
    )
    if margin != 0.85 or start != 0.20 or end != 1.55 or not 0.0 < margin < 1.0 or not 0.0 <= start < end:
        raise QueueError(
            f"mechanisms.{name} must use margin=0.85 and recovery=[0.20,1.55]"
        )


def _validate_budget(name: str, budget: dict[str, Any]) -> None:
    if name == "probe":
        _exact_keys(
            budget,
            {
                "num_envs", "num_steps_per_env", "additional_updates", "max_iterations",
                "save_interval", "exclusive_iteration_upper_bound",
                "terminal_checkpoint_iteration", "terminal_checkpoint_basename",
            },
            "budgets.probe",
        )
        expected = (4096, 24, 2, 2, 1, 6702, 6701, "model_6701.pt")
        actual = (
            budget["num_envs"], budget["num_steps_per_env"], budget["additional_updates"],
            budget["max_iterations"], budget["save_interval"],
            budget["exclusive_iteration_upper_bound"],
            budget["terminal_checkpoint_iteration"],
            budget["terminal_checkpoint_basename"],
        )
    else:
        _exact_keys(
            budget,
            {
                "num_envs", "num_steps_per_env", "additional_updates", "max_iterations",
                "save_interval", "offsets_from_parent", "absolute_milestones",
            },
            "budgets.train",
        )
        expected = (4096, 24, 1001, 1001, 100, [200, 500, 1000], [6900, 7200, 7700])
        actual = (
            budget["num_envs"], budget["num_steps_per_env"], budget["additional_updates"],
            budget["max_iterations"], budget["save_interval"],
            budget["offsets_from_parent"], budget["absolute_milestones"],
        )
    for key in ("num_envs", "num_steps_per_env", "additional_updates", "max_iterations", "save_interval"):
        _positive_int(budget[key], f"budgets.{name}.{key}")
    if actual != expected:
        raise QueueError(f"budgets.{name} must encode the fixed probe/milestone budget")


def _validate_launch_manifest_contract(value: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema_version", "status", "manifest_is_separate_reviewed_input",
            "required_cli", "required_bindings", "claim_publish",
            "probe_receipt_basename", "probe_verifier_publish", "train_unlock",
            "automatic_probe_approval", "note",
        },
        "launch_manifest_contract",
    )
    expected = {
        "schema_version": 1,
        "status": "explicit_reviewed_manifest_required_per_authorized_render",
        "manifest_is_separate_reviewed_input": True,
        "required_cli": ["--launch-manifest", "--expected-launch-manifest-sha256"],
        "required_bindings": [
            "exact_clean_source_commit_and_required_source_file_sha256",
            "queue_config_and_queue_runner_sha256",
            "a3_runtime_asset_tree_sha256_file_count_and_total_bytes",
            "preconverted_usd_exact_file_and_six_file_bundle_tree_motion_bank_checkpoint_and_parent_contract_sha256",
        ],
        "claim_publish": "exclusive_no_clobber_before_trainer_start",
        "probe_receipt_basename": "probe_receipt.json",
        "probe_verifier_publish": "exclusive_no_clobber_after_natural_exit",
        "train_unlock": "all_six_local_probe_receipts_must_verify",
        "automatic_probe_approval": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise QueueError(f"launch_manifest_contract.{key} changed")
    _text(value["note"], "launch_manifest_contract.note")


def _validate_queue(queue: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(
        queue,
        {
            "schema_version", "queue_id", "purpose", "simulation_only",
            "real_robot_authorized", "launch_authorized_by_default",
            "formal_exact_eligible", "evidence_class", "ssh", "pods",
            "namespace", "source", "assets", "parents", "common",
            "launch_manifest_contract", "mechanisms", "measurement_contract", "budgets",
            "startup_checks", "stop_contract", "jobs",
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
    _validate_launch_manifest_contract(
        _mapping(queue["launch_manifest_contract"], "launch_manifest_contract")
    )

    ssh = _mapping(queue["ssh"], "ssh")
    _exact_keys(ssh, {"key"}, "ssh")
    if ssh["key"] != "~/.ssh/id_ed25519_runpod":
        raise QueueError("unexpected SSH key path")
    pods = _mapping(queue["pods"], "pods")
    if set(pods) != {"pod1", "pod2"}:
        raise QueueError("pods must be exactly pod1 and pod2")
    expected_pods = {
        "pod1": ("162.43.172.171", 18333),
        "pod2": ("162.43.172.181", 13146),
    }
    for name, pod in pods.items():
        pod = _mapping(pod, f"pods.{name}")
        _exact_keys(pod, {"host", "port", "gpus"}, f"pods.{name}")
        if (pod["host"], pod["port"]) != expected_pods[name] or pod["gpus"] != [0, 1, 2]:
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
            "checkout", "identity_mode", "worktree_relative", "python",
            "commit",
            "setup_relative", "trainer_relative", "locked_launcher_relative",
            "required_relative_files", "note",
        },
        "source",
    )
    if _remote_path(source["checkout"], "source.checkout") != EXPECTED_SOURCE:
        raise QueueError("source checkout changed")
    if source["identity_mode"] != "clean_detached_exact_commit":
        raise QueueError("source identity mode changed")
    if source["commit"] != EXPECTED_REMOTE_SOURCE_COMMIT:
        raise QueueError("source must remain pinned to the reviewed exact remote commit")
    _remote_path(source["python"], "source.python")
    for key in ("worktree_relative", "setup_relative", "trainer_relative", "locked_launcher_relative"):
        value = _text(source[key], f"source.{key}")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value != str(path):
            raise QueueError(f"source.{key} must be a normalized relative path")
    required_source_files = _list(source["required_relative_files"], "source.required_relative_files")
    if len(required_source_files) != len(set(required_source_files)) or not required_source_files:
        raise QueueError("source.required_relative_files must be unique and non-empty")
    for index, raw in enumerate(required_source_files):
        path = PurePosixPath(_text(raw, f"source.required_relative_files[{index}]"))
        if path.is_absolute() or ".." in path.parts:
            raise QueueError("required source paths must be relative and traversal-free")
    _text(source["note"], "source.note")

    assets = _mapping(queue["assets"], "assets")
    _exact_keys(
        assets,
        {
            "a3_runtime_asset_root", "preconverted_a3_usd", "motion_forehand",
            "motion_backhand", "training_question_bank",
        },
        "assets",
    )
    for key, value in assets.items():
        _remote_path(value, f"assets.{key}")

    parents = _mapping(queue["parents"], "parents")
    if set(parents) != {"W", "V"}:
        raise QueueError("parents must be exactly W and V")
    for name, parent in parents.items():
        _validate_parent(name, _mapping(parent, f"parents.{name}"))

    common = _mapping(queue["common"], "common")
    _exact_keys(
        common,
        {
            "seed", "curriculum", "qdot_limit_hinge_weight",
            "initial_tts_weights", "planner_revision_override", "base_overrides",
        },
        "common",
    )
    if type(common["seed"]) is not int or common["seed"] != 3:
        raise QueueError("common.seed must be integer 3")
    if common["curriculum"] != "short_focus" or _finite_number(
        common["qdot_limit_hinge_weight"], "common.qdot_limit_hinge_weight"
    ) != 0.0:
        raise QueueError("common must remain short_focus with qdot hinge disabled")
    weights = _mapping(common["initial_tts_weights"], "common.initial_tts_weights")
    expected_weights = {
        "late_stress": 0.10, "exact_half_second": 0.45,
        "fast_deploy": 0.40, "broad_arrival": 0.05,
    }
    if set(weights) != set(expected_weights):
        raise QueueError("short_focus component set changed")
    for key, expected in expected_weights.items():
        if _finite_number(weights[key], f"initial_tts_weights.{key}") != expected:
            raise QueueError("short_focus weights changed")
    planner = _text(common["planner_revision_override"], "common.planner_revision_override")
    if _override_key(planner, "common.planner_revision_override") != "task.planner_revision":
        raise QueueError("planner_revision_override key changed")
    base = _list(common["base_overrides"], "common.base_overrides")
    base_map = _override_map(base, "common.base_overrides")
    if base_map.get("task.rewards.joint_velocity_limit_hinge_weight") != "0.0":
        raise QueueError("base recipe must explicitly keep qdot hinge at zero")
    forbidden_base = FACTOR_KEYS | {
        "checkpoint_path", "run_name", "seed", "num_envs", "max_iterations",
        "algo.runner.save_interval", "device",
        "task.rewards.racket_position_weight",
        "task.rewards.racket_velocity_weight",
        "task.rewards.racket_normal_weight",
        "task.rewards.foot_orientation_weight",
        "task.rewards.prestrike_upright_weight",
        "task.rewards.free_non_striking_arm_mimic",
    }
    overlap = set(base_map) & forbidden_base
    if overlap:
        raise QueueError(f"common.base_overrides owns per-cell keys: {sorted(overlap)}")

    mechanisms = _mapping(queue["mechanisms"], "mechanisms")
    if set(mechanisms) != {"C", "N", "H"}:
        raise QueueError("mechanisms must be exactly C, N, H")
    for name, mechanism in mechanisms.items():
        _validate_mechanism(name, _mapping(mechanism, f"mechanisms.{name}"))

    measurement = _mapping(queue["measurement_contract"], "measurement_contract")
    _exact_keys(
        measurement,
        {
            "enabled_in_all_cells", "source", "phase",
            "reset_first_step_excluded", "required_before_scientific_interpretation",
        },
        "measurement_contract",
    )
    if measurement != {
        "enabled_in_all_cells": True,
        "source": "processed_qdes_slew_hinge_probe",
        "phase": "recovery_0p20_to_1p55_seconds",
        "reset_first_step_excluded": True,
        "required_before_scientific_interpretation": True,
    }:
        raise QueueError("measurement contract changed")

    budgets = _mapping(queue["budgets"], "budgets")
    if set(budgets) != {"probe", "train"}:
        raise QueueError("budgets must contain probe and train")
    _validate_budget("probe", _mapping(budgets["probe"], "budgets.probe"))
    _validate_budget("train", _mapping(budgets["train"], "budgets.train"))
    if queue["startup_checks"] != EXPECTED_STARTUP_CHECKS:
        raise QueueError("startup_checks must remain exact and ordered")

    stop = _mapping(queue["stop_contract"], "stop_contract")
    _exact_keys(
        stop,
        {
            "automatic_stop", "stop_command_generated", "exact_numeric_pgid_only",
            "required_identity_fields", "launcher_state_basename",
            "leader_identity_basename", "pre_term_identity_basename",
            "pre_kill_identity_basename", "rule",
        },
        "stop_contract",
    )
    if (
        stop["automatic_stop"] is not False
        or stop["stop_command_generated"] is not False
        or stop["exact_numeric_pgid_only"] is not True
        or stop["required_identity_fields"] != ["pid", "pgid", "leader_starttime_ticks", "command"]
    ):
        raise QueueError("stop contract must forbid automatic/broad signalling")
    expected_basenames = {
        "launcher_state_basename": "run.log.launch",
        "leader_identity_basename": "run.log.launch.leader.json",
        "pre_term_identity_basename": "run.log.launch.pre_term.json",
        "pre_kill_identity_basename": "run.log.launch.pre_kill.json",
    }
    for key, expected in expected_basenames.items():
        if stop[key] != expected:
            raise QueueError(f"stop_contract.{key} changed")
    _text(stop["rule"], "stop_contract.rule")

    jobs = _list(queue["jobs"], "jobs")
    if len(jobs) != 6:
        raise QueueError("the queue must contain exactly six jobs")
    ids: set[str] = set()
    names: set[str] = set()
    dirs: set[str] = set()
    slots: set[tuple[str, int]] = set()
    normalized_jobs: list[dict[str, Any]] = []
    for index, raw_job in enumerate(jobs):
        job = _mapping(raw_job, f"jobs[{index}]")
        _exact_keys(
            job, {"id", "parent", "mechanism", "pod", "gpu", "run_name", "run_dir"},
            f"jobs[{index}]",
        )
        job_id = _text(job["id"], f"jobs[{index}].id", safe_id=True)
        run_name = _text(job["run_name"], f"jobs[{index}].run_name", safe_id=True)
        run_dir = _remote_path(job["run_dir"], f"jobs[{index}].run_dir")
        if job_id in ids or run_name in names or run_dir in dirs:
            raise QueueError("duplicate job id, run_name, or run_dir")
        if type(job["gpu"]) is not int:
            raise QueueError(f"jobs[{index}].gpu must be an integer")
        slot = (job["pod"], job["gpu"])
        if slot in slots:
            raise QueueError("duplicate GPU assignment")
        ids.add(job_id)
        names.add(run_name)
        dirs.add(run_dir)
        slots.add(slot)
        if job_id not in EXPECTED_JOBS or (
            job["parent"], job["mechanism"], job["pod"], job["gpu"]
        ) != EXPECTED_JOBS[job_id]:
            raise QueueError(f"job {job_id!r} changed its matrix cell or GPU")
        expected_dir = f"{EXPECTED_NAMESPACE}/runs/{job_id}"
        if run_dir != expected_dir:
            raise QueueError(f"job {job_id!r} must use fresh run dir {expected_dir}")
        normalized_jobs.append(job)
    if ids != set(EXPECTED_JOBS) or slots != {
        ("pod1", 0), ("pod1", 1), ("pod1", 2),
        ("pod2", 0), ("pod2", 1), ("pod2", 2),
    }:
        raise QueueError("six-cell matrix or six unique GPU slots is incomplete")

    # Compile both stages now.  This proves every cell has exactly one value
    # for every Hydra key and that the only within-parent scientific factors
    # are the five explicitly registered action-slew keys.
    for job in normalized_jobs:
        for stage in ("probe", "train"):
            _training_argv(queue, job, stage)
    for parent_name in ("W", "V"):
        parent_jobs = [job for job in normalized_jobs if job["parent"] == parent_name]
        invariant_maps = []
        for job in parent_jobs:
            compiled = _override_map(_training_argv(queue, job, "train")[2:], job["id"])
            for key in FACTOR_KEYS | {"run_name", "device"}:
                compiled.pop(key, None)
            invariant_maps.append(compiled)
        if not all(item == invariant_maps[0] for item in invariant_maps[1:]):
            raise QueueError(f"parent {parent_name} cells differ outside action-slew factors")
    return queue


def load_queue(path: Path = DEFAULT_QUEUE) -> LoadedQueue:
    path = path.resolve()
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise QueueError(f"invalid YAML: {exc}") from exc
    result = LoadedQueue(_validate_queue(_mapping(value, "queue")))
    result.source_path = path
    return result


def _manifest_source_relative_paths(queue: Mapping[str, Any]) -> list[str]:
    worktree = PurePosixPath(queue["source"]["worktree_relative"])
    paths = {
        str(worktree / queue["source"]["setup_relative"]),
        str(worktree / queue["source"]["trainer_relative"]),
        str(worktree / queue["source"]["locked_launcher_relative"]),
    }
    paths.update(
        str(worktree / relative)
        for relative in queue["source"]["required_relative_files"]
    )
    return sorted(paths)


def _artifact_sha256(value: Any, label: str) -> str:
    result = _require_sha256(value, label)
    if result == "0" * 64:
        raise QueueError(f"{label} must not be a placeholder digest")
    return result


def _load_launch_manifest(
    queue: Mapping[str, Any],
    path: Path | None,
    expected_file_sha256: str | None,
) -> LaunchManifest:
    if path is None or expected_file_sha256 is None:
        raise QueueError(
            "command generation requires both --launch-manifest and "
            "--expected-launch-manifest-sha256; final source/input hashes are pending"
        )
    expected_file_sha256 = _artifact_sha256(
        expected_file_sha256, "expected launch manifest file SHA256"
    )
    envelope, raw = _read_canonical_json(path.resolve(), "launch manifest")
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise QueueError("launch manifest file SHA256 differs from reviewed authority")
    _exact_keys(envelope, {"schema_version", "content", "content_sha256"}, "launch manifest")
    if envelope["schema_version"] != 1:
        raise QueueError("launch manifest schema_version must be 1")
    content = _mapping(envelope["content"], "launch manifest content")
    content_sha256 = _artifact_sha256(
        envelope["content_sha256"], "launch manifest content SHA256"
    )
    if _canonical_sha256(content) != content_sha256:
        raise QueueError("launch manifest canonical content digest mismatch")
    _exact_keys(
        content,
        {
            "schema_version", "queue_id", "queue_files", "source", "assets",
            "parents",
        },
        "launch manifest content",
    )
    if content["schema_version"] != 1 or content["queue_id"] != queue["queue_id"]:
        raise QueueError("launch manifest queue identity/schema mismatch")

    queue_files = _mapping(content["queue_files"], "launch manifest queue_files")
    _exact_keys(queue_files, {"config", "runner"}, "launch manifest queue_files")
    local_root = Path(__file__).resolve().parents[1]
    loaded_queue_path = getattr(queue, "source_path", None)
    expected_queue_path = (local_root / QUEUE_CONFIG_RELATIVE).resolve()
    if loaded_queue_path is None or Path(loaded_queue_path).resolve() != expected_queue_path:
        raise QueueError("launch authorization accepts only the checked-in canonical queue path")
    local_bindings = {
        "config": (QUEUE_CONFIG_RELATIVE, expected_queue_path),
        "runner": (QUEUE_RUNNER_RELATIVE, (local_root / QUEUE_RUNNER_RELATIVE).resolve()),
    }
    local_sha: dict[str, str] = {}
    for name, (relative, local_path) in local_bindings.items():
        binding = _mapping(queue_files[name], f"launch manifest queue_files.{name}")
        _exact_keys(binding, {"path", "sha256"}, f"launch manifest queue_files.{name}")
        if binding["path"] != relative:
            raise QueueError(f"launch manifest queue_files.{name}.path changed")
        observed = hashlib.sha256(_read_regular_bytes(local_path, f"local {name}")).hexdigest()
        expected = _artifact_sha256(binding["sha256"], f"queue_files.{name}.sha256")
        if observed != expected:
            raise QueueError(f"launch manifest does not bind the current {name} bytes")
        local_sha[name] = observed

    source = _mapping(content["source"], "launch manifest source")
    _exact_keys(source, {"checkout", "commit", "required_file_sha256"}, "launch manifest source")
    if _remote_path(source["checkout"], "launch manifest source.checkout") != queue["source"]["checkout"]:
        raise QueueError("launch manifest source checkout changed")
    if _require_commit(source["commit"], "launch manifest source.commit") != queue["source"]["commit"]:
        raise QueueError("launch manifest source commit differs from the reviewed remote C1")
    required = _mapping(source["required_file_sha256"], "source.required_file_sha256")
    expected_paths = _manifest_source_relative_paths(queue)
    if set(required) != set(expected_paths):
        raise QueueError("launch manifest source required-file set is incomplete or extra")
    for relative in expected_paths:
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or str(posix) != relative:
            raise QueueError("launch manifest source path is unsafe")
        _artifact_sha256(required[relative], f"source.required_file_sha256[{relative!r}]")

    assets = _mapping(content["assets"], "launch manifest assets")
    if set(assets) != set(queue["assets"]):
        raise QueueError("launch manifest asset set is incomplete or extra")
    for name in ("motion_forehand", "motion_backhand", "training_question_bank"):
        binding = _mapping(assets[name], f"launch manifest assets.{name}")
        _exact_keys(binding, {"path", "sha256"}, f"launch manifest assets.{name}")
        if _remote_path(binding["path"], f"assets.{name}.path") != queue["assets"][name]:
            raise QueueError(f"launch manifest asset path changed: {name}")
        _artifact_sha256(binding["sha256"], f"assets.{name}.sha256")
    usd = _mapping(assets["preconverted_a3_usd"], "launch manifest preconverted USD")
    _exact_keys(
        usd,
        {
            "path", "sha256", "bundle_root", "bundle_tree_sha256",
            "file_count", "total_file_bytes", "symlinks_forbidden",
        },
        "launch manifest preconverted USD",
    )
    usd_path = _remote_path(usd["path"], "preconverted USD path")
    if usd_path != queue["assets"]["preconverted_a3_usd"] or not usd_path.endswith("/model.usd"):
        raise QueueError("launch manifest preconverted USD file path changed")
    expected_bundle_root = str(PurePosixPath(usd_path).parent)
    if _remote_path(usd["bundle_root"], "preconverted USD bundle root") != expected_bundle_root:
        raise QueueError("preconverted USD bundle root must be the exact model.usd parent")
    _artifact_sha256(usd["sha256"], "preconverted USD file SHA256")
    _artifact_sha256(usd["bundle_tree_sha256"], "preconverted USD bundle tree SHA256")
    if usd["file_count"] != 6:
        raise QueueError("preconverted USD bundle must contain the registered six files")
    _positive_int(usd["total_file_bytes"], "preconverted USD bundle total_file_bytes")
    if usd["symlinks_forbidden"] is not True:
        raise QueueError("preconverted USD bundle must forbid symlinks")
    tree = _mapping(assets["a3_runtime_asset_root"], "launch manifest A3 tree")
    _exact_keys(
        tree,
        {"path", "tree_sha256", "file_count", "total_file_bytes", "symlinks_forbidden"},
        "launch manifest A3 tree",
    )
    if _remote_path(tree["path"], "A3 tree path") != queue["assets"]["a3_runtime_asset_root"]:
        raise QueueError("launch manifest A3 runtime tree path changed")
    _artifact_sha256(tree["tree_sha256"], "A3 tree SHA256")
    _positive_int(tree["file_count"], "A3 tree file_count")
    _positive_int(tree["total_file_bytes"], "A3 tree total_file_bytes")
    if tree["symlinks_forbidden"] is not True:
        raise QueueError("launch manifest A3 tree must forbid symlinks")

    parents = _mapping(content["parents"], "launch manifest parents")
    if set(parents) != {"W", "V"}:
        raise QueueError("launch manifest parents must be W and V")
    for name in ("W", "V"):
        binding = _mapping(parents[name], f"launch manifest parents.{name}")
        _exact_keys(
            binding,
            {
                "checkpoint_path", "checkpoint_sha256", "hard_contract_path",
                "hard_contract_sha256",
            },
            f"launch manifest parents.{name}",
        )
        parent = queue["parents"][name]
        if _remote_path(binding["checkpoint_path"], f"parents.{name}.checkpoint_path") != parent["checkpoint_path"]:
            raise QueueError(f"launch manifest parent {name} checkpoint path changed")
        if _remote_path(binding["hard_contract_path"], f"parents.{name}.hard_contract_path") != parent["hard_contract_path"]:
            raise QueueError(f"launch manifest parent {name} hard-contract path changed")
        _artifact_sha256(binding["checkpoint_sha256"], f"parents.{name}.checkpoint_sha256")
        _artifact_sha256(binding["hard_contract_sha256"], f"parents.{name}.hard_contract_sha256")
    return LaunchManifest(envelope, content, file_sha256, content_sha256, path.resolve())


def _hydra_number(value: Any) -> str:
    number = float(value)
    if number == 0.0:
        return "0.0"
    return str(number)


def _mechanism_overrides(mechanism: Mapping[str, Any]) -> list[str]:
    return [
        "task.rewards.action_rate_weight="
        + _hydra_number(mechanism["dense_action_rate_weight"]),
        "++task.rewards.processed_qdes_slew_hinge_weight="
        + _hydra_number(mechanism["processed_qdes_slew_hinge_weight"]),
        "++task.rewards.processed_qdes_slew_hinge_margin="
        + _hydra_number(mechanism["processed_qdes_slew_hinge_margin"]),
        "++task.rewards.processed_qdes_slew_hinge_recovery_start_s="
        + _hydra_number(mechanism["processed_qdes_slew_hinge_recovery_start_s"]),
        "++task.rewards.processed_qdes_slew_hinge_recovery_end_s="
        + _hydra_number(mechanism["processed_qdes_slew_hinge_recovery_end_s"]),
    ]


def _stage_run_dir(queue: Mapping[str, Any], job: Mapping[str, Any], stage: str) -> str:
    if stage == "train":
        return str(job["run_dir"])
    return f"{queue['namespace']['root']}/probes/{job['id']}"


def _stage_run_name(job: Mapping[str, Any], stage: str) -> str:
    if stage == "train":
        return str(job["run_name"])
    return f"phase1_balance_slew_probe5_{job['id']}_seed3_20260720"


def _training_argv(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str
) -> list[str]:
    if stage not in {"probe", "train"}:
        raise QueueError("stage must be probe or train")
    source = queue["source"]
    workdir = f"{source['checkout']}/{source['worktree_relative']}"
    parent = queue["parents"][job["parent"]]
    mechanism = queue["mechanisms"][job["mechanism"]]
    budget = queue["budgets"][stage]
    argv = [
        source["python"],
        f"{workdir}/{source['trainer_relative']}",
        *queue["common"]["base_overrides"],
        queue["common"]["planner_revision_override"],
        *parent["recipe_overrides"],
        *_mechanism_overrides(mechanism),
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
        "device=cuda:0",
    ]
    compiled = _override_map(argv[2:], f"{job['id']}.{stage}.argv")
    if set(compiled) & {"ros", "deploy", "real_robot", "motion_command"}:
        raise QueueError("real-robot/deploy arguments are forbidden")
    return argv


def _claim_paths(
    queue: Mapping[str, Any], job: Mapping[str, Any], stage: str
) -> tuple[str, str]:
    run_dir = _stage_run_dir(queue, job, stage)
    if stage == "probe":
        return (
            f"{run_dir}/full_scene_probe_claim.json",
            f"{run_dir}/full_scene_probe_binding.json",
        )
    return f"{run_dir}/queue_claim.json", f"{run_dir}/run_binding.json"


def _probe_terminal_status_path(
    queue: Mapping[str, Any], job: Mapping[str, Any]
) -> str:
    return f"{_stage_run_dir(queue, job, 'probe')}/terminal_status.json"


def _probe_supervisor_prefix(
    queue: Mapping[str, Any], job: Mapping[str, Any]
) -> list[str]:
    claim_path, binding_path = _claim_paths(queue, job, "probe")
    run_dir = _stage_run_dir(queue, job, "probe")
    return [
        queue["source"]["python"],
        "-B",
        "-c",
        PROBE_SUPERVISOR_PROGRAM,
        _probe_terminal_status_path(queue, job),
        binding_path,
        claim_path,
        f"{run_dir}/run.log",
        queue["source"]["checkout"],
        queue["source"]["commit"],
        job["id"],
        job["pod"],
        str(job["gpu"]),
        _stage_run_name(job, "probe"),
        run_dir,
        "--",
    ]


def _probe_receipt_set_digest(receipts: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "job_id": item["job_id"],
            "file_sha256": item["file_sha256"],
            "content_sha256": item["content_sha256"],
            "probe_claim_content_sha256": item["probe_claim_content_sha256"],
        }
        for item in receipts
    ]
    return _canonical_sha256(normalized)


def _build_claim(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    stage: str,
    manifest: LaunchManifest,
    *,
    probe_receipts: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], list[str]]:
    if stage not in {"probe", "train"}:
        raise QueueError("stage must be probe or train")
    if stage == "probe" and probe_receipts:
        raise QueueError("probe claims cannot consume probe receipts")
    if stage == "train" and len(probe_receipts) != len(EXPECTED_JOBS):
        raise QueueError("train claim requires all six verified probe receipts")
    claim_path, binding_path = _claim_paths(queue, job, stage)
    argv_without_claim = _training_argv(queue, job, stage)
    if stage == "train":
        argv_without_claim = [
            *argv_without_claim,
            f"++training_queue_claim_path={claim_path}",
            f"++training_run_binding_path={binding_path}",
        ]
    budget = queue["budgets"][stage]
    if stage == "probe":
        budget_binding = {
            "num_envs": budget["num_envs"],
            "num_steps_per_env": budget["num_steps_per_env"],
            "max_iterations": budget["max_iterations"],
            "save_interval": budget["save_interval"],
            "milestones": [budget["terminal_checkpoint_iteration"]],
            "parent_iteration": PARENT_ITERATION,
            "exclusive_iteration_upper_bound": budget["exclusive_iteration_upper_bound"],
            "terminal_checkpoint_iteration": budget["terminal_checkpoint_iteration"],
        }
    else:
        budget_binding = {
            "num_envs": budget["num_envs"],
            "num_steps_per_env": budget["num_steps_per_env"],
            "max_iterations": budget["max_iterations"],
            "save_interval": budget["save_interval"],
            "milestones": list(budget["absolute_milestones"]),
        }
    content: dict[str, Any] = {
        "schema_version": 1,
        "queue_id": queue["queue_id"],
        "job_id": job["id"],
        "stage": stage,
        "parent": job["parent"],
        "mechanism": job["mechanism"],
        "pod": job["pod"],
        "gpu": job["gpu"],
        "run_name": _stage_run_name(job, stage),
        "run_dir": _stage_run_dir(queue, job, stage),
        "source": {
            "checkout": manifest.content["source"]["checkout"],
            "commit": manifest.content["source"]["commit"],
            "required_file_sha256": manifest.content["source"]["required_file_sha256"],
        },
        "launch_manifest": {
            "file_sha256": manifest.file_sha256,
            "content_sha256": manifest.content_sha256,
        },
        "queue_files": manifest.content["queue_files"],
        "inputs": manifest.content["assets"],
        "parent_artifact": manifest.content["parents"][job["parent"]],
        "budget": budget_binding,
        "simulation_only": True,
        "real_robot_authorized": False,
        "formal_exact_eligible": False,
        "automatic_retry": False,
        "training_argv_without_claim": argv_without_claim,
    }
    if stage == "probe":
        content.update(
            {
                "purpose": "balance_action_slew_probe_not_science",
                "not_science": True,
                "attestable": False,
                "promotable": False,
                "supervisor_argv_prefix": _probe_supervisor_prefix(queue, job),
            }
        )
    else:
        receipt_bindings = [dict(item) for item in probe_receipts]
        content.update(
            {
                "purpose": None,
                "probe_receipts": receipt_bindings,
                "probe_receipt_set_sha256": _probe_receipt_set_digest(receipt_bindings),
            }
        )
    digest = _canonical_sha256(content)
    training_argv = [
        *argv_without_claim,
        f"++training_launch_claim_sha256={digest}",
    ]
    claim = {
        "schema_version": 2,
        "content": content,
        "content_sha256": digest,
        "training_argv": training_argv,
    }
    return claim, training_argv


CHECKPOINT_AUDIT_PROGRAM = r'''
import math
import sys
import torch

path = sys.argv[1]
expected = int(sys.argv[2])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
if not isinstance(checkpoint, dict) or type(checkpoint.get("iter")) is not int:
    raise SystemExit("checkpoint must be a mapping with an integer iter")
if checkpoint["iter"] != expected:
    raise SystemExit("checkpoint iteration mismatch")
optimizer = checkpoint.get("optimizer_state_dict")
if not isinstance(optimizer, dict) or not optimizer.get("state") or not optimizer.get("param_groups"):
    raise SystemExit("checkpoint optimizer state is missing")
required = (
    "model_state_dict",
    "optimizer_state_dict",
    "obs_norm_state_dict",
    "privileged_obs_norm_state_dict",
)
for name in required:
    if not isinstance(checkpoint.get(name), dict) or not checkpoint[name]:
        raise SystemExit(f"checkpoint {name} is missing or empty")
def audit(name, value):
    seen = set()
    tensors = 0
    floating = 0
    nonfinite = 0
    def visit(item):
        nonlocal tensors, floating, nonfinite
        if isinstance(item, torch.Tensor):
            tensors += 1
            if torch.is_floating_point(item) or torch.is_complex(item):
                floating += item.numel()
                nonfinite += item.numel() - int(torch.isfinite(item).sum().item())
            return
        if isinstance(item, float) and not math.isfinite(item):
            nonfinite += 1
            return
        if isinstance(item, dict):
            if id(item) in seen:
                return
            seen.add(id(item))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            if id(item) in seen:
                return
            seen.add(id(item))
            for child in item:
                visit(child)
    visit(value)
    if tensors <= 0 or floating <= 0 or nonfinite != 0:
        raise SystemExit(f"checkpoint {name} has no floating tensor state or is non-finite")
for name in required:
    audit(name, checkpoint[name])
'''.strip()


PROBE_SUPERVISOR_PROGRAM = r'''
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

if len(sys.argv) < 14 or sys.argv[12] != "--":
    raise SystemExit("probe supervisor requires bound metadata -- TRAIN_ARGV")
(
    status_path, binding_path, claim_path, run_log, source_checkout,
    expected_commit, job_id, pod, gpu_raw, run_name, run_dir,
) = sys.argv[1:12]
argv = sys.argv[13:]
gpu = int(gpu_raw)
def canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
def stable_bytes(path, label):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"{label} must be a regular non-symlink file")
    signature = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if signature(before) != signature(opened):
            raise SystemExit(f"{label} changed while opening")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if signature(before) != signature(after_fd) or signature(before) != signature(after):
        raise SystemExit(f"{label} changed while reading")
    return b"".join(chunks)
def load_document(path, label):
    raw = stable_bytes(path, label)
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical(value) + b"\n":
        raise SystemExit(f"{label} must be canonical JSON")
    digest = hashlib.sha256(canonical(value["content"])).hexdigest()
    if value.get("content_sha256") != digest:
        raise SystemExit(f"{label} content digest mismatch")
    return value
def publish(path, value, label):
    payload = canonical(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload): offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
def proc_identity(pid):
    proc = Path("/proc") / str(pid)
    stat_before = (proc / "stat").read_text(encoding="utf-8")
    cmdline = (proc / "cmdline").read_bytes()
    pgid = os.getpgid(pid)
    stat_after = (proc / "stat").read_text(encoding="utf-8")
    def starttime(value):
        close = value.rfind(")")
        fields = value[close + 2:].split()
        if close < 0 or len(fields) <= 19 or not fields[19].isdigit():
            raise SystemExit("process stat lacks starttime")
        return int(fields[19])
    before_start = starttime(stat_before)
    if starttime(stat_after) != before_start:
        raise SystemExit("process identity changed while binding")
    process_argv = [part.decode("utf-8", "strict") for part in cmdline.split(b"\0") if part]
    return {"pid": pid, "pgid": pgid, "starttime_ticks": before_start, "argv": process_argv}
claim = load_document(claim_path, "probe queue claim")
claim_digest = claim["content_sha256"]
if claim.get("schema_version") != 2 or claim.get("training_argv") != argv:
    raise SystemExit("probe queue claim/argv mismatch")
claim_content = claim["content"]
if claim_content.get("source", {}).get("commit") != expected_commit:
    raise SystemExit("probe claim source commit mismatch")
environment = dict(os.environ)
environment["GIT_OPTIONAL_LOCKS"] = "0"
def git(*args):
    return subprocess.run(
        ["git", "-C", source_checkout, *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
    ).stdout.strip()
if git("rev-parse", "HEAD") != expected_commit or git("status", "--porcelain", "--untracked-files=all"):
    raise SystemExit("probe source is not the exact clean claimed commit")
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
supervisor = proc_identity(os.getpid())
if supervisor["pid"] != supervisor["pgid"]:
    raise SystemExit("probe supervisor is not its process-group leader")
child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
trainer = proc_identity(child.pid)
if trainer["pgid"] != supervisor["pgid"] or trainer["argv"] != argv:
    raise SystemExit("probe trainer identity differs from claimed child")
rsl_dir = None
binding = None
marker_observed = False
assert child.stdout is not None
for line in iter(child.stdout.readline, b""):
    text = line.decode("utf-8", "replace")
    match = re.search(r"\| log: (\S+)\s*$", text)
    if match:
        candidate = Path(match.group(1))
        expected_root = Path(source_checkout) / "hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball"
        try:
            relative = candidate.relative_to(expected_root)
        except ValueError:
            raise SystemExit("probe RSL directory is outside the claimed source log root")
        if len(relative.parts) != 1 or not relative.name.endswith("_" + run_name):
            raise SystemExit("probe RSL directory does not match claimed run_name")
        rsl_dir = str(candidate)
    if b"Learning iteration" in line and not marker_observed:
        if rsl_dir is None:
            raise SystemExit("first iteration appeared before the RSL directory binding")
        binding_content = {
            "schema_version": 1,
            "job_id": job_id,
            "claim_path": claim_path,
            "claim_content_sha256": claim_digest,
            "binding_path": binding_path,
            "rsl_log_dir": rsl_dir,
            "process": trainer,
            "supervisor_process": supervisor,
            "pod": pod,
            "gpu": gpu,
            "source": claim_content["source"],
            "source_state_at_binding": {"head": expected_commit, "clean": True},
            "run_name": run_name,
            "run_dir": run_dir,
            "milestones": claim_content["budget"]["milestones"],
            "training_argv": argv,
            "purpose": claim_content["purpose"],
            "not_science": True,
            "attestable": False,
            "promotable": False,
        }
        binding = {"schema_version": 1, "content": binding_content, "content_sha256": hashlib.sha256(canonical(binding_content)).hexdigest()}
        publish(binding_path, binding, "probe run binding")
        marker_observed = True
    sys.stdout.buffer.write(line)
    sys.stdout.buffer.flush()
child.stdout.close()
return_code = child.wait()
ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
content = {
    "schema_version": 1,
    "started_utc": started,
    "ended_utc": ended,
    "child_argv": argv,
    "exit_code": return_code,
    "normal_exit": return_code == 0 and marker_observed and binding is not None,
    "claim_content_sha256": claim_digest,
    "binding_content_sha256": None if binding is None else binding["content_sha256"],
    "supervisor_process": supervisor,
    "trainer_process": trainer,
    "run_log": run_log,
    "first_iteration_observed": marker_observed,
}
document = {
    "schema_version": 1,
    "content": content,
    "content_sha256": hashlib.sha256(canonical(content)).hexdigest(),
}
publish(status_path, document, "probe terminal status")
raise SystemExit(return_code if return_code >= 0 else 128 - return_code)
'''.strip()


REMOTE_PREFLIGHT_PROGRAM = r'''
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

manifest = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
content = manifest["content"]
checkout = Path(content["source"]["checkout"])
def canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
if hashlib.sha256(canonical(content)).hexdigest() != manifest["content_sha256"]:
    raise SystemExit("manifest canonical digest mismatch")
def stable_file(path, label):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"{label} is not a regular non-symlink file: {path}")
    signature = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if signature(before) != signature(opened):
            raise SystemExit(f"{label} changed while opening: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if signature(before) != signature(after_fd) or signature(before) != signature(after):
        raise SystemExit(f"{label} changed while hashing: {path}")
    return b"".join(chunks)
def verify_file(path, expected, label):
    observed = hashlib.sha256(stable_file(path, label)).hexdigest()
    if observed != expected:
        raise SystemExit(f"{label} SHA256 mismatch")
environment = dict(os.environ)
environment["GIT_OPTIONAL_LOCKS"] = "0"
def git(*args):
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
    ).stdout.strip()
if git("rev-parse", "HEAD") != content["source"]["commit"]:
    raise SystemExit("source HEAD differs from manifest")
if git("status", "--porcelain", "--untracked-files=all"):
    raise SystemExit("source checkout is dirty")
for relative, expected in content["source"]["required_file_sha256"].items():
    verify_file(checkout / relative, expected, f"source file {relative}")
for name in ("preconverted_a3_usd", "motion_forehand", "motion_backhand", "training_question_bank"):
    item = content["assets"][name]
    verify_file(item["path"], item["sha256"], f"asset {name}")
for name, item in content["parents"].items():
    verify_file(item["checkpoint_path"], item["checkpoint_sha256"], f"parent {name} checkpoint")
    verify_file(item["hard_contract_path"], item["hard_contract_sha256"], f"parent {name} hard contract")
def verify_tree(root_value, expected_sha, expected_count, expected_bytes, label):
    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise SystemExit(f"{label} root must be a real directory")
    entries = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            if candidate.is_symlink():
                raise SystemExit(f"{label} contains a symlink: {candidate}")
        for filename in files:
            candidate = current_path / filename
            if candidate.is_symlink():
                raise SystemExit(f"{label} contains a symlink: {candidate}")
            data = stable_file(candidate, f"{label} file")
            entries.append({
                "path": candidate.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            })
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise SystemExit(f"{label} is empty")
    if len(entries) != expected_count or sum(item["size"] for item in entries) != expected_bytes:
        raise SystemExit(f"{label} count/byte total mismatch")
    if hashlib.sha256(canonical(entries)).hexdigest() != expected_sha:
        raise SystemExit(f"{label} canonical SHA256 mismatch")
a3_tree = content["assets"]["a3_runtime_asset_root"]
verify_tree(
    a3_tree["path"], a3_tree["tree_sha256"], a3_tree["file_count"],
    a3_tree["total_file_bytes"], "A3 runtime asset tree",
)
usd_bundle = content["assets"]["preconverted_a3_usd"]
verify_tree(
    usd_bundle["bundle_root"], usd_bundle["bundle_tree_sha256"],
    usd_bundle["file_count"], usd_bundle["total_file_bytes"],
    "preconverted A3 USD six-file bundle",
)
print("REMOTE_MANIFEST_PREFLIGHT_OK", flush=True)
'''.strip()


BINDING_AUDIT_PROGRAM = r'''
import hashlib
import json
from pathlib import Path
import sys

binding_path = Path(sys.argv[1])
claim_path = Path(sys.argv[2])
expected_claim = sys.argv[3]
expected_job = sys.argv[4]
def load(path):
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    canonical = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
    if raw != canonical:
        raise SystemExit(f"non-canonical JSON: {path}")
    return value
claim = load(claim_path)
binding = load(binding_path)
if claim.get("content_sha256") != expected_claim:
    raise SystemExit("queue claim digest mismatch")
if hashlib.sha256(json.dumps(claim["content"], allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest() != expected_claim:
    raise SystemExit("queue claim content mismatch")
content = binding.get("content")
digest = hashlib.sha256(json.dumps(content, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
if binding.get("schema_version") != 1 or binding.get("content_sha256") != digest:
    raise SystemExit("run binding digest mismatch")
if content.get("job_id") != expected_job or content.get("claim_content_sha256") != expected_claim:
    raise SystemExit("run binding job/claim mismatch")
if content.get("training_argv") != claim.get("training_argv"):
    raise SystemExit("run binding argv mismatch")
if content.get("source_state_at_binding") != {"head": claim["content"]["source"]["commit"], "clean": True}:
    raise SystemExit("run binding source proof mismatch")
print("RUN_BINDING_OK", flush=True)
'''.strip()


PROBE_VERIFIER_PROGRAM = r'''
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

spec = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
def canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
def signature(item):
    return (item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns)
def stable_bytes(path, label):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if signature(before) != signature(opened):
            raise SystemExit(f"{label} changed while opening")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if signature(before) != signature(after_fd) or signature(before) != signature(after):
        raise SystemExit(f"{label} changed while reading")
    return b"".join(chunks)
def load_document(path, label):
    raw = stable_bytes(path, label)
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical(value) + b"\n":
        raise SystemExit(f"{label} is not canonical JSON")
    if hashlib.sha256(canonical(value["content"])).hexdigest() != value.get("content_sha256"):
        raise SystemExit(f"{label} canonical content digest mismatch")
    return value, raw
claim, claim_raw = load_document(spec["claim_path"], "queue claim")
if claim.get("schema_version") != 2 or claim.get("content_sha256") != spec["claim_content_sha256"]:
    raise SystemExit("queue claim identity mismatch")
if claim.get("training_argv") != [*claim["content"]["training_argv_without_claim"], f"++training_launch_claim_sha256={spec['claim_content_sha256']}"]:
    raise SystemExit("queue claim does not self-bind training argv")
binding, binding_raw = load_document(spec["binding_path"], "run binding")
bound = binding["content"]
if bound.get("job_id") != spec["job_id"] or bound.get("claim_content_sha256") != spec["claim_content_sha256"]:
    raise SystemExit("run binding job/claim mismatch")
if bound.get("training_argv") != claim.get("training_argv"):
    raise SystemExit("run binding argv mismatch")
if bound.get("pod") != spec["pod"] or bound.get("gpu") != spec["gpu"]:
    raise SystemExit("run binding resource mismatch")
if bound.get("purpose") != "balance_action_slew_probe_not_science" or bound.get("not_science") is not True:
    raise SystemExit("probe binding purpose mismatch")
terminal, terminal_raw = load_document(spec["terminal_status_path"], "terminal status")
terminal_content = terminal["content"]
if terminal_content.get("exit_code") != 0 or terminal_content.get("normal_exit") is not True:
    raise SystemExit("probe did not exit naturally with status zero")
if terminal_content.get("child_argv") != claim.get("training_argv"):
    raise SystemExit("terminal status argv differs from claim")
supervisor = bound.get("supervisor_process")
trainer = bound.get("process")
if not isinstance(supervisor, dict) or not isinstance(trainer, dict):
    raise SystemExit("probe binding lacks trainer/supervisor identities")
if (
    terminal_content.get("claim_content_sha256") != spec["claim_content_sha256"]
    or terminal_content.get("binding_content_sha256") != binding["content_sha256"]
    or terminal_content.get("supervisor_process") != supervisor
    or terminal_content.get("trainer_process") != trainer
    or terminal_content.get("run_log") != spec["run_log"]
    or terminal_content.get("first_iteration_observed") is not True
):
    raise SystemExit("terminal status provenance binding mismatch")
pgid = supervisor.get("pgid")
if type(pgid) is not int or pgid <= 0 or supervisor.get("pid") != pgid or trainer.get("pgid") != pgid:
    raise SystemExit("probe process-group binding is invalid")
for pid in (supervisor.get("pid"), trainer.get("pid")):
    if type(pid) is not int or pid <= 0:
        raise SystemExit("probe binding has invalid PID")
    if (Path("/proc") / str(pid)).exists():
        raise SystemExit("bound probe process is still live")
for proc_dir in Path("/proc").iterdir():
    if not proc_dir.name.isdigit():
        continue
    try:
        stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    close = stat_text.rfind(")")
    fields = stat_text[close + 2:].split() if close >= 0 else []
    if len(fields) > 2 and fields[2].isdigit() and int(fields[2]) == pgid:
        raise SystemExit("probe process group still has a live member")
gpu_output = subprocess.run(
    ["nvidia-smi", "-i", str(spec["gpu"]), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
).stdout
gpu_lines = [line.strip() for line in gpu_output.splitlines() if line.strip()]
if any(not re.fullmatch(r"[1-9][0-9]*", line) for line in gpu_lines):
    raise SystemExit("nvidia-smi returned nonnumeric nonempty output")
if gpu_lines:
    raise SystemExit("assigned GPU was not released")
log_raw = stable_bytes(spec["run_log"], "probe log")
log_text = log_raw.decode("utf-8", "replace")
fatal = re.compile(r"(^|[^A-Za-z0-9_])(Fatal|Traceback|OutOfMemoryError|OutOfMemory|out of memory|OOM|bad_alloc|Segmentation fault)([^A-Za-z0-9_]|$)", re.I | re.M)
if fatal.search(log_text):
    raise SystemExit("probe log contains a semantic fatal marker")
for marker in spec["expected_applied_markers"]:
    if log_text.count(marker) != 1:
        raise SystemExit(f"probe log lacks one exact applied marker: {marker}")
rsl_dir = Path(bound["rsl_log_dir"])
if not rsl_dir.is_dir() or not rsl_dir.name.endswith("_" + claim["content"]["run_name"]):
    raise SystemExit("bound RSL directory is missing or has wrong run name")
checkpoint_path = rsl_dir / spec["terminal_checkpoint_basename"]
checkpoint_raw = stable_bytes(checkpoint_path, "terminal checkpoint")
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
if stable_bytes(checkpoint_path, "terminal checkpoint") != checkpoint_raw:
    raise SystemExit("terminal checkpoint changed while loading")
if (
    not isinstance(checkpoint, dict)
    or type(checkpoint.get("iter")) is not int
    or checkpoint["iter"] != spec["terminal_checkpoint_iteration"]
):
    raise SystemExit("terminal checkpoint iteration mismatch")
optimizer = checkpoint.get("optimizer_state_dict")
if not isinstance(optimizer, dict) or not optimizer.get("state") or not optimizer.get("param_groups"):
    raise SystemExit("terminal checkpoint optimizer state/param_groups are missing or empty")
def audit_state(name):
    value = checkpoint.get(name)
    if not isinstance(value, dict) or not value:
        raise SystemExit(f"terminal checkpoint {name} is missing or empty")
    seen = set()
    tensors = floating = nonfinite = 0
    def visit(item):
        nonlocal tensors, floating, nonfinite
        if isinstance(item, torch.Tensor):
            tensors += 1
            if torch.is_floating_point(item) or torch.is_complex(item):
                floating += item.numel()
                nonfinite += item.numel() - int(torch.isfinite(item).sum().item())
            return
        if isinstance(item, float) and not math.isfinite(item):
            nonfinite += 1
        elif isinstance(item, dict):
            if id(item) in seen:
                return
            seen.add(id(item))
            for child in item.values(): visit(child)
        elif isinstance(item, (list, tuple)):
            if id(item) in seen:
                return
            seen.add(id(item))
            for child in item: visit(child)
    visit(value)
    if tensors <= 0 or floating <= 0 or nonfinite:
        raise SystemExit(f"terminal checkpoint {name} has absent/non-finite floating state")
    return {"tensor_count": tensors, "floating_elements": floating, "nonfinite_elements": nonfinite}
state_audit = {name: audit_state(name) for name in (
    "model_state_dict", "optimizer_state_dict", "obs_norm_state_dict", "privileged_obs_norm_state_dict"
)}
infos = checkpoint.get("infos")
if not isinstance(infos, dict) or infos.get("training_launch_claim_sha256") != spec["claim_content_sha256"]:
    raise SystemExit("terminal checkpoint launch-claim lineage mismatch")
if type(infos.get("training_contract_lineage_exact")) is not int or infos["training_contract_lineage_exact"] != 0:
    raise SystemExit("intentional parent mismatch must retain exact lineage value 0")
hard_path = rsl_dir / "params" / "training_contract.json"
hard_raw = stable_bytes(hard_path, "hard training contract")
hard_sha = hashlib.sha256(hard_raw).hexdigest()
if infos.get("training_contract_sha256") != hard_sha:
    raise SystemExit("terminal checkpoint hard-contract SHA mismatch")
try:
    hard_contract = json.loads(hard_raw.decode("utf-8"))
except (UnicodeDecodeError, json.JSONDecodeError):
    raise SystemExit("hard training contract is not JSON")
slew_contract = hard_contract.get("processed_qdes_slew_hinge_reward") if isinstance(hard_contract, dict) else None
if not isinstance(slew_contract, dict):
    raise SystemExit("hard training contract lacks processed qdes slew section")
for key, expected in spec["expected_processed_qdes_contract"].items():
    if slew_contract.get(key) != expected:
        raise SystemExit(f"hard training contract processed qdes mismatch: {key}")
accumulator = EventAccumulator(str(rsl_dir), size_guidance={"scalars": 0})
accumulator.Reload()
rows = {step: {} for step in spec["expected_steps"]}
for counter, tag in spec["counter_tags"].items():
    if tag not in accumulator.Tags().get("scalars", []):
        raise SystemExit(f"missing probe metric tag: {tag}")
    events = accumulator.Scalars(tag)
    if [event.step for event in events] != spec["expected_steps"]:
            raise SystemExit(f"probe metric has wrong/duplicate steps: {tag}")
    for event in events:
        value = float(event.value)
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"probe metric is invalid: {tag}")
        rows[event.step][counter] = value
integer_counters = set(spec["counter_tags"]) - set(spec["float_counter_names"])
for step in spec["expected_steps"]:
    row = rows[step]
    for counter in integer_counters:
        if abs(row[counter] - round(row[counter])) > 1e-6:
            raise SystemExit(f"counter is not integral at step {step}: {counter}")
        row[counter] = int(round(row[counter]))
    observed = row["observed_sample_count"]
    valid = row["previous_qdes_valid_sample_count"]
    invalid = row["previous_qdes_invalid_first_step_sample_count"]
    eligible = row["recovery_eligible_sample_count"]
    enabled = row["reward_enabled_eligible_sample_count"]
    tail = row["tail_active_sample_count"]
    joints = row["above_margin_joint_count"]
    tail_sum = row["gated_tail_value_sum"]
    if observed != valid + invalid or observed != spec["expected_samples_per_update"] or not (0 <= eligible <= valid <= observed):
        raise SystemExit(f"activation denominator inconsistency at step {step}")
    if not (0 <= tail <= eligible and tail <= joints <= 15 * tail):
        raise SystemExit(f"activation tail/joint inconsistency at step {step}")
    tail_bound = joints / 15.0
    # TensorBoard records the environment reduction as float32.  Keep this
    # bound fail-closed at one part per million while allowing normal reduction
    # rounding at 4096 environments (which can exceed a fixed 1e-6 tolerance).
    tail_tolerance = max(1.0, abs(tail_bound)) * 1.0e-6
    if tail_sum < 0.0 or tail_sum > tail_bound + tail_tolerance:
        raise SystemExit(f"activation tail/value bound inconsistency at step {step}")
    if tail == 0 and (joints != 0 or tail_sum != 0.0):
        raise SystemExit(f"inactive tail has nonzero joint/value evidence at step {step}")
    expected_enabled = eligible if spec["mechanism"] == "H" else 0
    if enabled != expected_enabled:
        raise SystemExit(f"reward-enabled activation mismatch at step {step}")
    outcome = row["racket_swing_outcome_count"]
    completion = row["racket_swing_completion_count"]
    physical = row["racket_physical_fall_count"]
    pre_fall = row["racket_pre_strike_physical_fall_count"]
    post_fall = row["racket_post_strike_physical_fall_count"]
    if outcome <= 0 or not (0 <= completion <= outcome):
        raise SystemExit(f"behavior completion denominator inconsistency at step {step}")
    if pre_fall + post_fall != physical or not (0 <= physical <= outcome):
        raise SystemExit(f"behavior physical-fall closeout inconsistency at step {step}")
    strike = row["racket_strike_opportunity_count"]
    legal = row["racket_virtual_legal_return_count"]
    if strike <= 0 or not (0 <= legal <= strike):
        raise SystemExit(f"behavior legal-return denominator inconsistency at step {step}")
    ready = row["racket_ready_tilt_eligible_sample_count"]
    ready_sum = row["racket_ready_tilt_rad_sum"]
    if ready <= 0 or ready_sum < 0.0 or row["racket_ready_nonfinite_value_count"] != 0:
        raise SystemExit(f"ready-tilt denominator/value inconsistency at step {step}")
    qdot_observed = row["qdot_observed_sample_count"]
    qdot_excess = row["qdot_excess_sample_count"]
    qdot_sum = row["qdot_normalized_excess_square_sum"]
    if qdot_observed != spec["expected_samples_per_update"]:
        raise SystemExit(f"qdot observed denominator inconsistency at step {step}")
    if not (0 <= qdot_excess <= qdot_observed):
        raise SystemExit(f"qdot excess denominator inconsistency at step {step}")
    if (qdot_excess == 0 and qdot_sum != 0.0) or (qdot_excess > 0 and qdot_sum <= 0.0):
        raise SystemExit(f"qdot excess/value inconsistency at step {step}")
activation_rows = [{"step": step, **rows[step]} for step in spec["expected_steps"]]
totals = {counter: sum(row[counter] for row in activation_rows) for counter in spec["counter_tags"]}
if totals["previous_qdes_invalid_first_step_sample_count"] < spec["num_envs"]:
    raise SystemExit("two-update probe did not observe one full reset-invalid denominator")
if totals["recovery_eligible_sample_count"] <= 0:
    raise SystemExit("two-update probe did not observe any recovery-eligible sample")
receipt_content = {
    "schema_version": 1,
    "queue_id": spec["queue_id"],
    "job_id": spec["job_id"],
    "parent": spec["parent"],
    "mechanism": spec["mechanism"],
    "pod": spec["pod"],
    "gpu": spec["gpu"],
    "status": "passed",
    "launch_manifest": spec["launch_manifest"],
    "probe_claim_content_sha256": spec["claim_content_sha256"],
    "probe_verifier_program_sha256": spec["verifier_program_sha256"],
    "artifacts": {
        "queue_claim_file_sha256": hashlib.sha256(claim_raw).hexdigest(),
        "run_binding_file_sha256": hashlib.sha256(binding_raw).hexdigest(),
        "run_binding_content_sha256": binding["content_sha256"],
        "terminal_status_file_sha256": hashlib.sha256(terminal_raw).hexdigest(),
        "terminal_status_content_sha256": terminal["content_sha256"],
        "terminal_checkpoint_path": str(checkpoint_path),
        "terminal_checkpoint_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
        "terminal_checkpoint_iteration": checkpoint["iter"],
        "hard_contract_path": str(hard_path),
        "hard_contract_sha256": hard_sha,
    },
    "checkpoint_state_audit": state_audit,
    "activation": {
        "expected_steps": spec["expected_steps"],
        "expected_samples_per_update": spec["expected_samples_per_update"],
        "rows": activation_rows,
        "totals": totals,
    },
    "runtime": spec["expected_runtime"],
}
receipt = {"schema_version": 1, "content": receipt_content, "content_sha256": hashlib.sha256(canonical(receipt_content)).hexdigest()}
payload = canonical(receipt) + b"\n"
output = Path(spec["receipt_path"])
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(output, flags, 0o600)
try:
    offset = 0
    while offset < len(payload): offset += os.write(fd, payload[offset:])
    os.fsync(fd)
finally:
    os.close(fd)
print(json.dumps({"status": "passed", "receipt_path": str(output), "content_sha256": receipt["content_sha256"]}, sort_keys=True), flush=True)
'''.strip()


PROBE_VERIFIER_PROGRAM_SHA256 = hashlib.sha256(
    PROBE_VERIFIER_PROGRAM.encode("utf-8")
).hexdigest()


def _validate_activation_payload(value: Any, mechanism: str, label: str) -> None:
    activation = _mapping(value, label)
    _exact_keys(
        activation,
        {"expected_steps", "expected_samples_per_update", "rows", "totals"},
        label,
    )
    if activation["expected_steps"] != list(PROBE_STEPS):
        raise QueueError(f"{label}.expected_steps must be {list(PROBE_STEPS)}")
    if activation["expected_samples_per_update"] != EXPECTED_SAMPLES_PER_UPDATE:
        raise QueueError(
            f"{label}.expected_samples_per_update must be {EXPECTED_SAMPLES_PER_UPDATE}"
        )
    rows = _list(activation["rows"], f"{label}.rows")
    if len(rows) != len(PROBE_STEPS):
        raise QueueError(f"{label}.rows must contain exactly two updates")
    calculated: dict[str, float] = {counter: 0.0 for counter in PROBE_COUNTERS}
    for expected_step, raw_row in zip(PROBE_STEPS, rows):
        row = _mapping(raw_row, f"{label}.rows[{expected_step}]")
        _exact_keys(row, {"step", *PROBE_COUNTERS}, f"{label}.rows[{expected_step}]")
        if row["step"] != expected_step:
            raise QueueError(f"{label} has wrong or reordered update steps")
        integers: dict[str, int] = {}
        for counter in PROBE_COUNTERS:
            number = _finite_number(row[counter], f"{label}.{expected_step}.{counter}")
            if number < 0:
                raise QueueError(f"{label}.{expected_step}.{counter} must be nonnegative")
            if counter not in PROBE_FLOAT_COUNTERS:
                if not number.is_integer():
                    raise QueueError(f"{label}.{expected_step}.{counter} must be integral")
                integers[counter] = int(number)
            calculated[counter] += number
        observed = integers["observed_sample_count"]
        valid = integers["previous_qdes_valid_sample_count"]
        invalid = integers["previous_qdes_invalid_first_step_sample_count"]
        eligible = integers["recovery_eligible_sample_count"]
        enabled = integers["reward_enabled_eligible_sample_count"]
        tail = integers["tail_active_sample_count"]
        joints = integers["above_margin_joint_count"]
        tail_sum = float(row["gated_tail_value_sum"])
        if observed != valid + invalid or observed != EXPECTED_SAMPLES_PER_UPDATE or not 0 <= eligible <= valid <= observed:
            raise QueueError(f"{label} activation denominator inconsistency at {expected_step}")
        if not 0 <= tail <= eligible or not tail <= joints <= 15 * tail:
            raise QueueError(f"{label} tail/joint inconsistency at {expected_step}")
        tail_bound = joints / 15.0
        tail_tolerance = max(1.0, abs(tail_bound)) * 1.0e-6
        if tail_sum < 0.0 or tail_sum > tail_bound + tail_tolerance:
            raise QueueError(f"{label} tail/value bound inconsistency at {expected_step}")
        if tail == 0 and (joints != 0 or tail_sum != 0.0):
            raise QueueError(f"{label} inactive tail has nonzero evidence at {expected_step}")
        expected_enabled = eligible if mechanism == "H" else 0
        if enabled != expected_enabled:
            raise QueueError(f"{label} reward-enabled counter mismatch at {expected_step}")
        outcome = integers["racket_swing_outcome_count"]
        completion = integers["racket_swing_completion_count"]
        physical = integers["racket_physical_fall_count"]
        pre_fall = integers["racket_pre_strike_physical_fall_count"]
        post_fall = integers["racket_post_strike_physical_fall_count"]
        if outcome <= 0 or not 0 <= completion <= outcome:
            raise QueueError(f"{label} behavior completion denominator inconsistency at {expected_step}")
        if pre_fall + post_fall != physical or not 0 <= physical <= outcome:
            raise QueueError(f"{label} behavior physical-fall closeout inconsistency at {expected_step}")
        strike = integers["racket_strike_opportunity_count"]
        legal = integers["racket_virtual_legal_return_count"]
        if strike <= 0 or not 0 <= legal <= strike:
            raise QueueError(f"{label} behavior legal-return denominator inconsistency at {expected_step}")
        ready = integers["racket_ready_tilt_eligible_sample_count"]
        ready_sum = float(row["racket_ready_tilt_rad_sum"])
        if (
            ready <= 0
            or ready_sum < 0.0
            or integers["racket_ready_nonfinite_value_count"] != 0
        ):
            raise QueueError(f"{label} ready-tilt denominator/value inconsistency at {expected_step}")
        qdot_observed = integers["qdot_observed_sample_count"]
        qdot_excess = integers["qdot_excess_sample_count"]
        qdot_sum = float(row["qdot_normalized_excess_square_sum"])
        if qdot_observed != EXPECTED_SAMPLES_PER_UPDATE:
            raise QueueError(f"{label} qdot observed denominator inconsistency at {expected_step}")
        if not 0 <= qdot_excess <= qdot_observed:
            raise QueueError(f"{label} qdot excess denominator inconsistency at {expected_step}")
        if (qdot_excess == 0 and qdot_sum != 0.0) or (
            qdot_excess > 0 and qdot_sum <= 0.0
        ):
            raise QueueError(f"{label} qdot excess/value inconsistency at {expected_step}")
    totals = _mapping(activation["totals"], f"{label}.totals")
    if set(totals) != set(PROBE_COUNTERS):
        raise QueueError(f"{label}.totals counter set changed")
    for counter, expected in calculated.items():
        observed = _finite_number(totals[counter], f"{label}.totals.{counter}")
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6):
            raise QueueError(f"{label}.totals.{counter} does not sum the rows")
    if calculated["previous_qdes_invalid_first_step_sample_count"] < 4096:
        raise QueueError(f"{label} did not observe a full reset-invalid denominator")
    if calculated["recovery_eligible_sample_count"] <= 0:
        raise QueueError(f"{label} did not observe any recovery-eligible sample")


def _validate_probe_receipt(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    manifest: LaunchManifest,
    path: Path,
) -> dict[str, str]:
    receipt, raw = _read_canonical_json(path, f"probe receipt {job['id']}")
    _exact_keys(receipt, {"schema_version", "content", "content_sha256"}, "probe receipt")
    if receipt["schema_version"] != 1:
        raise QueueError(f"probe receipt {job['id']} schema_version must be 1")
    content = _mapping(receipt["content"], f"probe receipt {job['id']} content")
    content_sha = _artifact_sha256(
        receipt["content_sha256"], f"probe receipt {job['id']} content SHA256"
    )
    if _canonical_sha256(content) != content_sha:
        raise QueueError(f"probe receipt {job['id']} canonical digest mismatch")
    expected_claim, _ = _build_claim(queue, job, "probe", manifest)
    expected_identity = {
        "schema_version": 1,
        "queue_id": queue["queue_id"],
        "job_id": job["id"],
        "parent": job["parent"],
        "mechanism": job["mechanism"],
        "pod": job["pod"],
        "gpu": job["gpu"],
        "status": "passed",
        "launch_manifest": {
            "file_sha256": manifest.file_sha256,
            "content_sha256": manifest.content_sha256,
        },
        "probe_claim_content_sha256": expected_claim["content_sha256"],
        "probe_verifier_program_sha256": PROBE_VERIFIER_PROGRAM_SHA256,
    }
    for key, expected in expected_identity.items():
        if content.get(key) != expected:
            raise QueueError(f"probe receipt {job['id']} identity mismatch: {key}")
    _exact_keys(
        content,
        {
            *expected_identity,
            "artifacts", "checkpoint_state_audit", "activation", "runtime",
        },
        f"probe receipt {job['id']} content",
    )
    artifacts = _mapping(content["artifacts"], f"probe receipt {job['id']} artifacts")
    _exact_keys(
        artifacts,
        {
            "queue_claim_file_sha256", "run_binding_file_sha256",
            "run_binding_content_sha256", "terminal_status_file_sha256",
            "terminal_status_content_sha256", "terminal_checkpoint_path",
            "terminal_checkpoint_sha256", "terminal_checkpoint_iteration",
            "hard_contract_path", "hard_contract_sha256",
        },
        f"probe receipt {job['id']} artifacts",
    )
    for key in (
        "queue_claim_file_sha256", "run_binding_file_sha256",
        "run_binding_content_sha256", "terminal_status_file_sha256",
        "terminal_status_content_sha256", "terminal_checkpoint_sha256",
        "hard_contract_sha256",
    ):
        _artifact_sha256(artifacts[key], f"probe receipt {job['id']} {key}")
    expected_claim_file_sha256 = hashlib.sha256(
        _json_document(expected_claim)
    ).hexdigest()
    if artifacts["queue_claim_file_sha256"] != expected_claim_file_sha256:
        raise QueueError(
            f"probe receipt {job['id']} queue claim file SHA does not match the canonical expected claim"
        )
    checkpoint_path = _remote_path(
        artifacts["terminal_checkpoint_path"],
        f"probe receipt {job['id']} terminal checkpoint path",
    )
    if not checkpoint_path.endswith("/model_6701.pt") or artifacts["terminal_checkpoint_iteration"] != 6701:
        raise QueueError(f"probe receipt {job['id']} terminal checkpoint is not model_6701")
    _remote_path(artifacts["hard_contract_path"], f"probe receipt {job['id']} hard contract path")
    state = _mapping(content["checkpoint_state_audit"], f"probe receipt {job['id']} state audit")
    required_state = {
        "model_state_dict", "optimizer_state_dict", "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    }
    if set(state) != required_state:
        raise QueueError(f"probe receipt {job['id']} full-state audit set changed")
    for name, raw_audit in state.items():
        audit = _mapping(raw_audit, f"probe receipt {job['id']} state {name}")
        _exact_keys(audit, {"tensor_count", "floating_elements", "nonfinite_elements"}, f"state {name}")
        _positive_int(audit["tensor_count"], f"state {name}.tensor_count")
        _positive_int(audit["floating_elements"], f"state {name}.floating_elements")
        if audit["nonfinite_elements"] != 0:
            raise QueueError(f"probe receipt {job['id']} state {name} is non-finite")
    runtime = _mapping(content["runtime"], f"probe receipt {job['id']} runtime")
    if runtime != EXPECTED_PROBE_RUNTIME:
        raise QueueError(f"probe receipt {job['id']} lacks a clean terminal runtime proof")
    _validate_activation_payload(
        content["activation"], job["mechanism"], f"probe receipt {job['id']} activation"
    )
    return {
        "job_id": job["id"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha,
        "probe_claim_content_sha256": expected_claim["content_sha256"],
    }


def _load_probe_receipts(
    queue: Mapping[str, Any], manifest: LaunchManifest, directory: Path | None
) -> list[dict[str, str]]:
    if directory is None:
        raise QueueError(
            "train command generation requires --probe-receipts-dir containing all six verifier receipts"
        )
    root = directory.resolve()
    if root.is_symlink() or not root.is_dir():
        raise QueueError("probe receipts directory must be a real directory")
    results = []
    for job in queue["jobs"]:
        path = root / job["id"] / "probe_receipt.json"
        results.append(_validate_probe_receipt(queue, job, manifest, path))
    if {item["job_id"] for item in results} != set(EXPECTED_JOBS):
        raise QueueError("probe receipt set is incomplete")
    return results


def _probe_verifier_spec(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    manifest: LaunchManifest,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = _stage_run_dir(queue, job, "probe")
    claim_path, binding_path = _claim_paths(queue, job, "probe")
    mechanism = queue["mechanisms"][job["mechanism"]]
    dense = _hydra_number(mechanism["dense_action_rate_weight"])
    hinge = _hydra_number(mechanism["processed_qdes_slew_hinge_weight"])
    margin = _hydra_number(mechanism["processed_qdes_slew_hinge_margin"])
    start = _hydra_number(mechanism["processed_qdes_slew_hinge_recovery_start_s"])
    end = _hydra_number(mechanism["processed_qdes_slew_hinge_recovery_end_s"])
    return {
        "schema_version": 1,
        "queue_id": queue["queue_id"],
        "job_id": job["id"],
        "parent": job["parent"],
        "mechanism": job["mechanism"],
        "pod": job["pod"],
        "gpu": job["gpu"],
        "launch_manifest": {
            "file_sha256": manifest.file_sha256,
            "content_sha256": manifest.content_sha256,
        },
        "claim_path": claim_path,
        "binding_path": binding_path,
        "claim_content_sha256": claim["content_sha256"],
        "terminal_status_path": _probe_terminal_status_path(queue, job),
        "run_log": f"{run_dir}/run.log",
        "receipt_path": f"{run_dir}/probe_receipt.json",
        "terminal_checkpoint_basename": "model_6701.pt",
        "terminal_checkpoint_iteration": 6701,
        "num_envs": queue["budgets"]["probe"]["num_envs"],
        "num_steps_per_env": queue["budgets"]["probe"]["num_steps_per_env"],
        "expected_samples_per_update": (
            queue["budgets"]["probe"]["num_envs"]
            * queue["budgets"]["probe"]["num_steps_per_env"]
        ),
        "expected_steps": list(PROBE_STEPS),
        "counter_tags": dict(PROBE_COUNTER_TAGS),
        "float_counter_names": sorted(PROBE_FLOAT_COUNTERS),
        "expected_processed_qdes_contract": {
            "schema_version": 1,
            "enabled": float(mechanism["processed_qdes_slew_hinge_weight"]) < 0.0,
            "weight": float(mechanism["processed_qdes_slew_hinge_weight"]),
            "margin": float(mechanism["processed_qdes_slew_hinge_margin"]),
            "recovery_start_s": float(mechanism["processed_qdes_slew_hinge_recovery_start_s"]),
            "recovery_end_s": float(mechanism["processed_qdes_slew_hinge_recovery_end_s"]),
            "action_name": "joint_pos",
            "command_name": "racket_target",
            "joint_count": 15,
        },
        "expected_applied_markers": [
            f"[train.py]     rewards.action_rate_l2.weight={dense}",
            f"[train.py]     rewards.processed_qdes_slew_hinge.weight={hinge}",
            f"[train.py]     rewards.processed_qdes_slew_hinge.params.margin={margin}",
            f"[train.py]     rewards.processed_qdes_slew_hinge.params.recovery_start_s={start}",
            f"[train.py]     rewards.processed_qdes_slew_hinge.params.recovery_end_s={end}",
            (
                "[train.py]     rewards.processed_qdes_slew_hinge_probe="
                f"(margin={margin},recovery={start}..{end},weight=1.0)"
            ),
        ],
        "expected_runtime": dict(EXPECTED_PROBE_RUNTIME),
        "verifier_program_sha256": PROBE_VERIFIER_PROGRAM_SHA256,
    }


def _remote_launch_body(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    stage: str,
    manifest: LaunchManifest,
    claim: Mapping[str, Any],
    training_argv: Sequence[str],
) -> str:
    source = queue["source"]
    checkout = source["checkout"]
    workdir = f"{checkout}/{source['worktree_relative']}"
    setup = f"{workdir}/{source['setup_relative']}"
    launcher = f"{workdir}/{source['locked_launcher_relative']}"
    trainer = f"{workdir}/{source['trainer_relative']}"
    run_dir = _stage_run_dir(queue, job, stage)
    run_parent = str(PurePosixPath(run_dir).parent)
    run_log = f"{run_dir}/run.log"
    state = f"{run_log}.launch"
    parent = queue["parents"][job["parent"]]
    argv = list(training_argv)
    compose_argv = [argv[0], argv[1], "--cfg", "job", "--resolve", *argv[2:]]
    required = [
        trainer,
        setup,
        launcher,
        *[
            f"{workdir}/{relative}"
            for relative in source["required_relative_files"]
        ],
        queue["assets"]["preconverted_a3_usd"],
        queue["assets"]["motion_forehand"],
        queue["assets"]["motion_backhand"],
        queue["assets"]["training_question_bank"],
        parent["checkpoint_path"],
        parent["hard_contract_path"],
    ]
    regular_checks = "\n".join(f"test -f {shlex.quote(path)}" for path in required)
    manifest_b64 = base64.b64encode(_canonical_bytes(manifest.envelope)).decode("ascii")
    remote_preflight = shlex.join(
        [source["python"], "-B", "-c", REMOTE_PREFLIGHT_PROGRAM, manifest_b64]
    )
    checkpoint_audit = shlex.join(
        [
            source["python"], "-B", "-c", CHECKPOINT_AUDIT_PROGRAM,
            parent["checkpoint_path"], str(PARENT_ITERATION),
        ]
    )
    child_environment = (
        f"env CUDA_VISIBLE_DEVICES={job['gpu']} "
        f"HOPE_AGIBOT_A3_USD_PATH={shlex.quote(queue['assets']['preconverted_a3_usd'])} "
        "PYTHONUNBUFFERED=1 PYTHONPATH=\"${HOPE_WBT_PYTHONPATH}\""
    )
    compose = f"{child_environment} {shlex.join(compose_argv)}"
    launched_argv = (
        [*_probe_supervisor_prefix(queue, job), *argv]
        if stage == "probe"
        else argv
    )
    launch = (
        f"{shlex.quote(launcher)} {shlex.quote(run_log)} "
        f"{child_environment} {shlex.join(launched_argv)}"
    )
    metadata = json.dumps(
        {
            "schema_version": 1,
            "queue_id": queue["queue_id"],
            "job_id": job["id"],
            "stage": stage,
            "parent": job["parent"],
            "mechanism": job["mechanism"],
            "pod": job["pod"],
            "gpu": job["gpu"],
            "simulation_only": True,
            "real_robot_authorized": False,
            "formal_exact_eligible": False,
            "automatic_stop": False,
            "checkpoint_allow_contract_mismatch": True,
            "training_argv": argv,
            "launch_manifest_file_sha256": manifest.file_sha256,
            "launch_manifest_content_sha256": manifest.content_sha256,
            "queue_claim_content_sha256": claim["content_sha256"],
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    manifest_document = _json_document(manifest.envelope).decode("utf-8")
    claim_document = _json_document(claim).decode("utf-8")
    claim_path, binding_path = _claim_paths(queue, job, stage)
    binding_audit = shlex.join(
        [
            source["python"], "-B", "-c", BINDING_AUDIT_PROGRAM,
            binding_path, claim_path, claim["content_sha256"], job["id"],
        ]
    )
    fatal_pattern = (
        r"(^|[^[:alnum:]_])(Fatal|Traceback|OutOfMemory|out of memory|OOM|"
        r"bad_alloc|Segmentation fault)([^[:alnum:]_]|$)"
    )
    # No broad process operation is present here.  The reviewed launcher binds
    # pid/pgid/starttime/argv and may only clean its own exact group on boot
    # failure.  Later stop metadata is emitted, but this harness emits no stop.
    return f"""set -euo pipefail
test -d {shlex.quote(checkout)}
{regular_checks}
{remote_preflight}
{checkpoint_audit}
gpu_output=$(nvidia-smi -i {job['gpu']} --query-compute-apps=pid --format=csv,noheader,nounits)
if test -n "$gpu_output"; then
  if printf '%s\n' "$gpu_output" | awk 'NF != 1 || $1 !~ /^[1-9][0-9]*$/ {{exit 1}}'; then
    printf '%s\n' 'assigned GPU already has a compute process' >&2
  else
    printf '%s\n' 'nvidia-smi returned nonnumeric nonempty output' >&2
  fi
  exit 1
fi
cd {shlex.quote(workdir)}
source {shlex.quote(setup)}
{compose} >/dev/null
test ! -e {shlex.quote(run_dir)}
mkdir -p {shlex.quote(run_parent)}
mkdir {shlex.quote(run_dir)}
( set -o noclobber; printf %s {shlex.quote(manifest_document)} > {shlex.quote(run_dir + '/launch_manifest.json')} )
( set -o noclobber; printf %s {shlex.quote(claim_document)} > {shlex.quote(claim_path)} )
( set -o noclobber; printf %s {shlex.quote(metadata)} > {shlex.quote(run_dir + '/launch_spec.json')} )
( set -o noclobber; git -C {shlex.quote(checkout)} rev-parse HEAD > {shlex.quote(run_dir + '/source_commit.txt')} )
export KIT_BOOT_MARKER='Learning iteration'
export KIT_BOOT_TIMEOUT_S=900
export KIT_BOOT_STALE_TIMEOUT_S=180
{launch}
test -s {shlex.quote(state)}
grep -Eq '^pid=[1-9][0-9]*$' {shlex.quote(state)}
grep -Eq '^pgid=[1-9][0-9]*$' {shlex.quote(state)}
grep -Eq '^leader_starttime_ticks=[1-9][0-9]*$' {shlex.quote(state)}
grep -Fq -- 'Learning iteration' {shlex.quote(run_log)}
test -s {shlex.quote(binding_path)}
{binding_audit}
if grep -Eiq -- {shlex.quote(fatal_pattern)} {shlex.quote(run_log)}; then
  echo 'fatal log scan failed; no automatic retry or broad signal is authorized' >&2
  exit 1
fi
""".strip()


def _stop_metadata(queue: Mapping[str, Any], job: Mapping[str, Any], stage: str) -> dict[str, Any]:
    run_log = f"{_stage_run_dir(queue, job, stage)}/run.log"
    return {
        "automatic_stop": False,
        "stop_command_generated": False,
        "exact_numeric_pgid_only": True,
        "required_revalidation": ["pid", "pgid", "leader_starttime_ticks", "command"],
        "launcher_state": f"{run_log}.launch",
        "leader_identity": f"{run_log}.launch.leader.json",
        "pre_term_identity_if_later_reviewed": f"{run_log}.launch.pre_term.json",
        "pre_kill_identity_if_later_reviewed": f"{run_log}.launch.pre_kill.json",
    }


def _wrap_ssh(queue: Mapping[str, Any], job: Mapping[str, Any], remote: str) -> list[str]:
    pod = queue["pods"][job["pod"]]
    key = os.path.expanduser(queue["ssh"]["key"])
    return [
        "ssh", "-i", key, "-p", str(pod["port"]),
        "-o", "BatchMode=yes", f"root@{pod['host']}",
        f"bash -lc {shlex.quote(remote)}",
    ]


def _ssh_argv(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    stage: str,
    manifest: LaunchManifest,
    claim: Mapping[str, Any],
    training_argv: Sequence[str],
) -> list[str]:
    remote = _remote_launch_body(
        queue, job, stage, manifest, claim, training_argv
    )
    return _wrap_ssh(queue, job, remote)


def _probe_verifier_ssh_argv(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    manifest: LaunchManifest,
    claim: Mapping[str, Any],
) -> list[str]:
    source = queue["source"]
    workdir = f"{source['checkout']}/{source['worktree_relative']}"
    setup = f"{workdir}/{source['setup_relative']}"
    spec_b64 = base64.b64encode(
        _canonical_bytes(_probe_verifier_spec(queue, job, manifest, claim))
    ).decode("ascii")
    verifier = shlex.join(
        [source["python"], "-B", "-c", PROBE_VERIFIER_PROGRAM, spec_b64]
    )
    remote = f"""set -euo pipefail
cd {shlex.quote(workdir)}
source {shlex.quote(setup)}
env CUDA_VISIBLE_DEVICES={job['gpu']} HOPE_AGIBOT_A3_USD_PATH={shlex.quote(queue['assets']['preconverted_a3_usd'])} PYTHONUNBUFFERED=1 PYTHONPATH=\"${{HOPE_WBT_PYTHONPATH}}\" {verifier}
""".strip()
    return _wrap_ssh(queue, job, remote)


def cmd_plan(queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "plan",
        "queue_id": queue["queue_id"],
        "dry_run": True,
        "launch_authorized": False,
        "commands_emitted": False,
        "simulation_only": True,
        "real_robot_authorized": False,
        "formal_exact_eligible": False,
        "launch_manifest_gate": {
            "status": "blocked_manifest_not_supplied_to_this_invocation",
            "required_cli": ["--launch-manifest", "--expected-launch-manifest-sha256"],
            "reason": (
                "this no-launch invocation did not supply the separately reviewed manifest "
                "and exact file SHA256; a checked-in manifest is not ambient authority"
            ),
        },
        "probe_gate": {
            "required_before_train_command_generation": True,
            **queue["budgets"]["probe"],
            "expected_samples_per_update": EXPECTED_SAMPLES_PER_UPDATE,
        },
        "train_budget": dict(queue["budgets"]["train"]),
        "jobs": [
            {
                "job_id": job["id"],
                "parent": f"{job['parent']}@model_6700",
                "mechanism": job["mechanism"],
                "resource": f"{job['pod']}/gpu{job['gpu']}",
                "run_name": job["run_name"],
                "run_dir": job["run_dir"],
                "weights": {
                    "dense_action_rate": queue["mechanisms"][job["mechanism"]]["dense_action_rate_weight"],
                    "processed_qdes_slew_hinge": queue["mechanisms"][job["mechanism"]]["processed_qdes_slew_hinge_weight"],
                },
            }
            for job in queue["jobs"]
        ],
        "next": (
            "after separately reviewing a complete manifest, rerun with --authorize-launch "
            "--stage probe --launch-manifest PATH --expected-launch-manifest-sha256 SHA256"
        ),
    }


def cmd_launch_commands(
    queue: Mapping[str, Any],
    *,
    stage: str,
    launch_manifest_path: Path | None = None,
    expected_launch_manifest_sha256: str | None = None,
    probe_receipts_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_launch_manifest(
        queue, launch_manifest_path, expected_launch_manifest_sha256
    )
    launch_authority = _validate_origin_main_launch_authority(queue, manifest)
    probe_receipts = (
        _load_probe_receipts(queue, manifest, probe_receipts_dir)
        if stage == "train"
        else []
    )
    if stage == "probe" and probe_receipts_dir is not None:
        raise QueueError("--probe-receipts-dir is valid only for --stage train")
    rows = []
    for job in queue["jobs"]:
        claim, training_argv = _build_claim(
            queue, job, stage, manifest, probe_receipts=probe_receipts
        )
        argv = _ssh_argv(queue, job, stage, manifest, claim, training_argv)
        row = {
            "job_id": job["id"],
            "stage": stage,
            "resource": f"{job['pod']}/gpu{job['gpu']}",
            "run_dir": _stage_run_dir(queue, job, stage),
            "queue_claim_content_sha256": claim["content_sha256"],
            "ssh_argv": argv,
            "launch_command": shlex.join(argv),
            "stop_metadata": _stop_metadata(queue, job, stage),
        }
        if stage == "probe":
            verifier_argv = _probe_verifier_ssh_argv(
                queue, job, manifest, claim
            )
            row.update(
                {
                    "probe_receipt_remote_path": (
                        f"{_stage_run_dir(queue, job, 'probe')}/probe_receipt.json"
                    ),
                    "probe_verifier_program_sha256": PROBE_VERIFIER_PROGRAM_SHA256,
                    "probe_verifier_ssh_argv": verifier_argv,
                    "probe_verifier_command": shlex.join(verifier_argv),
                }
            )
        rows.append(row)
    return {
        "mode": "launch_commands",
        "queue_id": queue["queue_id"],
        "dry_run": True,
        "command_generation_only": True,
        "no_ssh_executed": True,
        "launch_authorized": True,
        "commands_emitted": True,
        "stage": stage,
        "launch_manifest": {
            "path": str(manifest.path),
            "file_sha256": manifest.file_sha256,
            "content_sha256": manifest.content_sha256,
        },
        "origin_main_authority": launch_authority,
        "probe_receipt_set_sha256": (
            _probe_receipt_set_digest(probe_receipts) if probe_receipts else None
        ),
        "probe_receipt_count": len(probe_receipts),
        "simulation_only": True,
        "real_robot_authorized": False,
        "automatic_retry": False,
        "jobs": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--stage", choices=("probe", "train"), default="probe")
    parser.add_argument(
        "--authorize-launch",
        action="store_true",
        help="render launch commands; never executes them",
    )
    parser.add_argument(
        "--launch-manifest",
        type=Path,
        help="separately reviewed canonical JSON binding source/config/input SHA256 values",
    )
    parser.add_argument(
        "--expected-launch-manifest-sha256",
        help="reviewed SHA256 of the exact canonical launch-manifest file bytes",
    )
    parser.add_argument(
        "--probe-receipts-dir",
        type=Path,
        help="for train: directory containing JOB_ID/probe_receipt.json for all six cells",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = load_queue(args.queue.resolve())
        if args.authorize_launch:
            result = cmd_launch_commands(
                queue,
                stage=args.stage,
                launch_manifest_path=args.launch_manifest,
                expected_launch_manifest_sha256=args.expected_launch_manifest_sha256,
                probe_receipts_dir=args.probe_receipts_dir,
            )
        else:
            if any(
                value is not None
                for value in (
                    args.launch_manifest,
                    args.expected_launch_manifest_sha256,
                    args.probe_receipts_dir,
                )
            ):
                raise QueueError("launch manifest/receipt options require --authorize-launch")
            result = cmd_plan(queue)
    except (KeyError, OSError, QueueError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
