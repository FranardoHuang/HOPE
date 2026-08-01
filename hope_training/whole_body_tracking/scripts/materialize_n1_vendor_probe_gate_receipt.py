#!/usr/bin/env python3
"""Materialize one canonical PASS receipt for the vendor N1 long gate.

The materializer is deliberately host-only.  It consumes the immutable launch
claim, complete run log and finite checkpoints from one integrated ``probe``
run.  That single 4096-environment, five-update run must prove both the core
probe contract and runtime-observed vendor velocity pushes.  It never starts
Kit or training.  The output is an exclusive, canonical JSON file; an existing
path is permanently no-clobber.

The receipt is self-reference free: it binds one clean gate-code/evidence
commit, but not the future artifact commit that will track the receipt.  That
later commit is constrained by the launcher to add
only the exact receipt/long-spec paths and ``docs/**``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence, Union


_THIS_FILE = Path(__file__).resolve()
_VENDOR_LAUNCHER_FILE = _THIS_FILE.with_name(
    "launch_n1_vendor_baseline_diagnostic.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_hope_vendor_probe_gate_launcher", _VENDOR_LAUNCHER_FILE
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load vendor launcher: {_VENDOR_LAUNCHER_FILE}")
_V = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V)
_B = _V._B


SCHEMA_VERSION = 3
RECEIPT_KIND = "n1_vendor_probe_gate_receipt_v3"
PRODUCER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_n1_vendor_probe_gate_receipt.py"
)
ACTOR_OBS_CONTRACT = "action_ball_table_pose_twist_heading_task_teacher_start_v2"
ROLLOUT_STEPS_PER_UPDATE = 24
POLICY_DT_S = 0.02
PUSH_INTERVAL_RANGE_S = (1.0, 3.0)
PUSH_VELOCITY_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
PUSH_VELOCITY_BOUNDS = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (-0.1, 0.1),
    "roll": (-0.26, 0.26),
    "pitch": (-0.26, 0.26),
    "yaw": (-0.39, 0.39),
}
EXPECTED_STAGES = {
    "probe": {"num_envs": 4096, "max_iterations": 5, "save_interval": 1},
}
EXPECTED_CHECKPOINT_INDICES = {
    "probe": (0, 1, 2, 3, 4),
}
BEHAVIOR_RATE_LIMITS = {
    "probe": {"table_contact_per_env_step": 0.005, "fall_per_env_step": 0.001},
}
MIN_CONSERVATIVE_EPISODE_AGE_STEPS = 60.0

_MARKERS = {
    "policy_bootstrap": "HOPE_ACTION_BALL_POLICY_BOOTSTRAP_JSON=",
    "abi": "HOPE_RSL_RL_RUNTIME_ABI_JSON=",
    "delay": "HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON=",
    "std_lr": "HOPE_POLICY_STD_UPDATE_JSON=",
    "reward_ppo_economy": (
        "HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_UPDATE_JSON="
    ),
    "joint_safety": "HOPE_JOINT_SAFETY_UPDATE_JSON=",
    "joint_safety_fatal": "HOPE_JOINT_SAFETY_FATAL_JSON=",
    "behavior": "HOPE_EXACT_BEHAVIOR_UPDATE_JSON=",
    "push_velocity": "HOPE_PUSH_VELOCITY_DIAGNOSTIC_UPDATE_JSON=",
    "completion": "HOPE_TRAINING_COMPLETE_JSON=",
}
_ENTRY_COUNT = "strike_window_entry_racket_target_distance_count"
_ENTRY_NONFINITE = "strike_window_entry_racket_target_distance_nonfinite_count"
_ENTRY_BUCKETS = (
    "strike_window_entry_racket_target_distance_le_0p075m_count",
    "strike_window_entry_racket_target_distance_gt_0p075m_le_0p15m_count",
    "strike_window_entry_racket_target_distance_gt_0p15m_le_0p20m_count",
    "strike_window_entry_racket_target_distance_gt_0p20m_le_0p30m_count",
    "strike_window_entry_racket_target_distance_gt_0p30m_le_0p50m_count",
    "strike_window_entry_racket_target_distance_gt_0p50m_le_0p70m_count",
    "strike_window_entry_racket_target_distance_gt_0p70m_le_1p00m_count",
    "strike_window_entry_racket_target_distance_gt_1p00m_count",
)
_TABLE_ATTRIBUTION_CATEGORIES = (
    "proxy_conservative_only",
    "proxy_exact_overlap",
    "blade_conservative_only",
    "blade_exact_overlap",
    "nonfinite",
)
_TABLE_ATTRIBUTION_PHASES = ("pre", "strike", "post", "recovery")
_VARIABLE_ARG_PREFIXES = (
    "device=",
    "num_envs=",
    "max_iterations=",
    "algo.runner.save_interval=",
    "run_name=",
    "+n1_vendor_diagnostic_stage=",
    "+vendor_runtime_training_contract_sha256=",
)
# The raw launch claim and argv SHA bind this exact diagnostic switch.  It is
# excluded only from the scientific projection because it changes evidence
# collection, not training semantics.  No broad ``task.table_*`` prefix
# belongs here.
_DIAGNOSTIC_ARG_PREFIXES = (
    "+task.table_contact_attribution_diagnostic=",
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ReceiptRefused(RuntimeError):
    """Raised before writing when source evidence is not an exact PASS."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptRefused(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReceiptRefused(f"{name} contains non-finite JSON: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptRefused(f"{name} is not strict UTF-8 JSON: {exc}") from exc
    if type(value) is not dict:
        raise ReceiptRefused(f"{name} must contain one JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReceiptRefused(f"value is not canonical finite JSON: {exc}") from exc


def _canonical_sha(value: Any) -> str:
    # Content seals and launch-claim SHAs hash canonical JSON bytes only.  The
    # persisted file has a trailing newline, whose identity is covered by the
    # separate file_sha256 field.
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_stable_bytes(path: Path, *, name: str) -> bytes:
    path = _real_file(path, name=name)
    before = path.lstat()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReceiptRefused(f"{name} cannot be read: {exc}") from exc
    after = path.lstat()
    if _file_identity(before) != _file_identity(after) or len(raw) != after.st_size:
        raise ReceiptRefused(f"{name} changed while being read")
    return raw


def _real_file(value: Union[str, Path], *, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ReceiptRefused(f"{name} must be absolute")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReceiptRefused(f"{name} cannot be inspected: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path:
        raise ReceiptRefused(f"{name} must be one real regular file")
    return path


def _real_dir(value: Union[str, Path], *, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ReceiptRefused(f"{name} must be absolute")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReceiptRefused(f"{name} cannot be inspected: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or path.resolve(strict=True) != path:
        raise ReceiptRefused(f"{name} must be one real directory")
    return path


def _repo_path(value: str, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ReceiptRefused(f"{name} must be one non-empty POSIX repo path")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ReceiptRefused(f"{name} must be normalized and repo-relative")
    return path.as_posix()


def _run_git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if result.returncode != 0:
        raise ReceiptRefused(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _verify_gate_source(checkout: Path, commit: str) -> dict[str, str]:
    checkout = _real_dir(checkout, name="gate checkout")
    if _COMMIT_RE.fullmatch(commit or "") is None:
        raise ReceiptRefused("gate source commit must be 40 lowercase hex")
    if _run_git(checkout, "rev-parse", "HEAD") != commit:
        raise ReceiptRefused("gate checkout HEAD differs from gate source commit")
    if _run_git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ReceiptRefused("gate source checkout must be exactly clean")
    producer = checkout / PRODUCER_SOURCE
    if producer.resolve(strict=True) != _THIS_FILE:
        raise ReceiptRefused("running producer is not the selected checkout source")
    tracked = _run_git(checkout, "ls-files", "--error-unmatch", PRODUCER_SOURCE)
    if tracked != PRODUCER_SOURCE:
        raise ReceiptRefused("producer source is not tracked")
    return {"path": PRODUCER_SOURCE, "sha256": _sha_file(producer)}


def _load_claim(namespace: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    claim_path = _real_file(namespace / "launch_claim.json", name="launch claim")
    raw = _read_stable_bytes(claim_path, name="launch claim")
    claim = _strict_json_bytes(raw, name="launch claim")
    if raw != _canonical_bytes(claim) + b"\n":
        raise ReceiptRefused("launch claim is not canonical JSON plus newline")
    payload = claim.get("canonical_payload")
    claim_sha = claim.get("launch_claim_sha256")
    if (
        claim.get("schema_version") != 1
        or claim.get("kind") != _V.CLAIM_KIND
        or type(payload) is not dict
        or _SHA_RE.fullmatch(claim_sha or "") is None
        or _canonical_sha(payload) != claim_sha
    ):
        raise ReceiptRefused("launch claim envelope or digest differs")
    return claim, payload, {
        "path": str(claim_path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "launch_claim_sha256": claim_sha,
    }


def _parse_markers(log_path: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    log_path = _real_file(log_path, name="run log")
    records = {name: [] for name in _MARKERS}
    raw_log = _read_stable_bytes(log_path, name="run log")
    try:
        text = raw_log.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReceiptRefused(f"run log is not UTF-8: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        for name, prefix in _MARKERS.items():
            if not line.startswith(prefix):
                continue
            raw = line[len(prefix) :].strip().encode("utf-8")
            row = _strict_json_bytes(raw, name=f"{name} marker line {line_number}")
            row["_line"] = line_number
            records[name].append(row)
    return records, hashlib.sha256(raw_log).hexdigest()


def _nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ReceiptRefused(f"{name} must be a non-negative integer")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ReceiptRefused(f"{name} must be finite and positive")
    return float(value)


def _validate_policy_bootstrap(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pin the fresh vendor policy parameterization before PPO mutates it."""

    if len(records) != 1:
        raise ReceiptRefused(
            "stage requires exactly one fresh policy bootstrap marker"
        )
    row = dict(records[0])
    row.pop("_line", None)
    values = (
        row.get("realized_policy_std_min"),
        row.get("realized_policy_std_mean"),
        row.get("realized_policy_std_max"),
    )
    if (
        set(row)
        != {
            "event",
            "schema_version",
            "applied_fresh",
            "noise_std_type",
            "parameter_name",
            "parameter_shape",
            "parameter_count",
            "configured_init_noise_std",
            "realized_policy_std_min",
            "realized_policy_std_mean",
            "realized_policy_std_max",
        }
        or row.get("event") != "hope_action_ball_policy_bootstrap"
        or row.get("schema_version") != 1
        or row.get("applied_fresh") is not True
        or row.get("noise_std_type") != "log"
        or row.get("parameter_name") != "log_std"
        or row.get("parameter_shape") != [31]
        or row.get("parameter_count") != 31
        or row.get("configured_init_noise_std") != 0.02
        or any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), 0.02, rel_tol=0.0, abs_tol=1.0e-8
            )
            for value in values
        )
        or not values[0] <= values[1] <= values[2]
    ):
        raise ReceiptRefused(
            "fresh policy bootstrap is not exact log_std with realized sigma 0.02"
        )
    return row


def _validate_abi(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 1:
        raise ReceiptRefused("stage requires exactly one runtime ABI marker")
    row = dict(records[0])
    row.pop("_line", None)
    caps = row.get("capabilities")
    runtime = row.get("runtime")
    distributions = (
        runtime.get("distributions") if type(runtime) is dict else None
    )
    normalizer_binding = caps.get("normalizer_binding") if type(caps) is dict else None
    normalizers = (
        normalizer_binding.get("normalizers")
        if type(normalizer_binding) is dict
        else None
    )
    if (
        row.get("event") != "hope_rsl_rl_runtime_abi"
        or row.get("schema_version") != 1
        or type(runtime) is not dict
        or set(runtime)
        != {"distributions", "package_origin", "runner_module", "runner_origin"}
        or type(distributions) is not list
        or not distributions
        or any(
            type(item) is not dict
            or set(item) != {"name", "version"}
            or type(item["name"]) is not str
            or not item["name"]
            or type(item["version"]) is not str
            or not item["version"]
            for item in distributions
        )
        or type(runtime.get("package_origin")) is not str
        or not runtime["package_origin"]
        or not Path(runtime["package_origin"]).is_absolute()
        or type(runtime.get("runner_module")) is not str
        or not runtime["runner_module"].startswith("rsl_rl.")
        or type(runtime.get("runner_origin")) is not str
        or not runtime["runner_origin"]
        or not Path(runtime["runner_origin"]).is_absolute()
        or type(caps) is not dict
        or caps.get("empirical_normalization_preflight") is not True
        or caps.get("positive_realized_policy_std_guard") is not True
        or type(normalizer_binding) is not dict
        or normalizer_binding.get("empirical_normalization") is not True
        or type(normalizers) is not dict
        or set(normalizers) != {"actor", "critic"}
        or caps.get("policy_std_abi")
        != {
            "noise_std_type": "log",
            "parameter_name": "log_std",
            "parameter_shape": [31],
            "parameter_count": 31,
        }
    ):
        raise ReceiptRefused("runtime ABI marker is incomplete")
    for role, expected_features in (("actor", 194), ("critic", 318)):
        binding = normalizers[role]
        shapes = binding.get("state_shapes") if type(binding) is dict else None
        semantic = binding.get("semantic_buffers") if type(binding) is dict else None
        mean_key = semantic.get("mean") if type(semantic) is dict else None
        mean_shape = shapes.get(mean_key) if type(shapes) is dict else None
        if (
            binding.get("enabled") is not True
            or type(mean_shape) is not list
            or not mean_shape
            or any(type(value) is not int or value <= 0 for value in mean_shape)
            or math.prod(mean_shape) != expected_features
        ):
            raise ReceiptRefused(
                f"runtime ABI {role} normalizer feature shape differs"
            )
    return row


def _validate_delay(records: list[dict[str, Any]], *, num_envs: int) -> dict[str, Any]:
    if len(records) != 1:
        raise ReceiptRefused("stage requires exactly one action-delay marker")
    row = dict(records[0])
    row.pop("_line", None)
    terms = row.get("delay_terms")
    if (
        row.get("event") != "hope_control_step_action_delay_runtime"
        or row.get("schema_version") != 1
        or _SHA_RE.fullmatch(row.get("training_contract_sha256") or "") is None
        or type(terms) is not list
        or len(terms) != 1
    ):
        raise ReceiptRefused("action-delay marker is incomplete")
    term = terms[0]
    histogram = term.get("lag_histogram") if type(term) is dict else None
    expected_contract = {
        "schema_version": 1,
        "enabled": True,
        "semantic_unit": "policy_control_step",
        "sample_timing": "once_per_episode_reset",
        "distribution": "discrete_uniform_inclusive",
        "min_steps": 0,
        "max_steps": 2,
        "shared_across_all_31_joints": True,
        "history_fill": "safe_default_or_action_specific_hold",
    }
    if (
        row.get("active_action_term_names") != ["joint_pos"]
        or term.get("term_name") != "joint_pos"
        or term.get("schema_version") != 1
        or term.get("kind")
        != "whole_body_tracking.policy_control_step_action_delay_receipt"
        or term.get("num_envs") != num_envs
        or term.get("initialized_env_count") != num_envs
        or type(histogram) is not dict
        or set(histogram) != {"0", "1", "2"}
        or any(type(value) is not int or value < 0 for value in histogram.values())
        or any(value <= 0 for value in histogram.values())
        or sum(histogram.values()) != num_envs
        or term.get("contract") != expected_contract
    ):
        raise ReceiptRefused("action-delay histogram/contract is incomplete")
    return row


def _validate_completion(
    records: list[dict[str, Any]],
    *,
    stage: str,
    num_envs: int,
    updates: int,
    expected_claim_sha256: str,
    expected_hard_contract_sha256: str,
    expected_vendor_contract_sha256: str,
) -> dict[str, Any]:
    if len(records) != 1:
        raise ReceiptRefused("stage requires exactly one natural-completion marker")
    row = dict(records[0])
    row.pop("_line", None)
    expected = {
        "cleanup_complete": True,
        "completed_ppo_updates": updates,
        "event": "hope_training_complete",
        "num_envs": num_envs,
        "schema_version": 1,
        "stage": stage,
        "training_contract_sha256": expected_hard_contract_sha256,
        "training_launch_claim_sha256": expected_claim_sha256,
        "vendor_runtime_training_contract_sha256": (
            expected_vendor_contract_sha256
        ),
    }
    if row != expected:
        raise ReceiptRefused("natural-completion marker identity differs")
    return row


def _validate_std_lr(records: list[dict[str, Any]], *, updates: int) -> list[dict[str, Any]]:
    if len(records) != updates:
        raise ReceiptRefused("policy std/LR marker count differs from PPO budget")
    result = []
    for expected, source in enumerate(records):
        row = dict(source)
        row.pop("_line", None)
        if (
            row.get("event") != "hope_policy_std_update"
            or row.get("schema_version") != 1
            or row.get("ppo_update") != expected
            or row.get("noise_std_type") != "log"
            or row.get("parameter_name") != "log_std"
            or row.get("parameter_shape") != [31]
            or row.get("parameter_count") != 31
        ):
            raise ReceiptRefused("policy std/LR update sequence differs")
        minimum = _finite_positive(row.get("policy_std_min"), name="policy std min")
        mean = _finite_positive(row.get("policy_std_mean"), name="policy std mean")
        maximum = _finite_positive(row.get("policy_std_max"), name="policy std max")
        learning_rate = _finite_positive(
            row.get("learning_rate"), name="learning rate"
        )
        if (
            not minimum <= mean <= maximum
            or type(row.get("learning_rate_at_floor")) is not bool
            or row["learning_rate_at_floor"]
            is not (learning_rate <= 1.0e-5)
        ):
            raise ReceiptRefused("policy std ordering/LR floor marker differs")
        result.append(row)
    return result


_ECONOMY_DISTRIBUTION_KEYS = {"min", "mean", "p50", "p95", "p99", "max"}
_ECONOMY_ADVANTAGE_KEYS = {"mean", "std", "min", "max"}
_ECONOMY_MIN_MEAN_MAX_KEYS = {"min", "mean", "max"}


def _finite_number(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ReceiptRefused(f"{name} must be {qualifier}")
    return float(value)


def _finite_economy_stats(
    value: Any, *, name: str, keys: set[str]
) -> dict[str, float]:
    if type(value) is not dict or set(value) != keys:
        raise ReceiptRefused(f"{name} statistic keys differ")
    result = {
        key: _finite_number(item, name=f"{name}.{key}")
        for key, item in value.items()
    }
    if "p50" in result and not (
        result["min"]
        <= result["p50"]
        <= result["p95"]
        <= result["p99"]
        <= result["max"]
    ):
        raise ReceiptRefused(f"{name} quantiles are not ordered")
    if not result["min"] <= result["mean"] <= result["max"]:
        raise ReceiptRefused(f"{name} mean lies outside its extrema")
    return result


def _validate_reward_ppo_economy(
    records: list[dict[str, Any]],
    *,
    updates: int,
    num_envs: int,
    expected_recipe_sha256: str,
) -> dict[str, Any]:
    """Validate all runtime fields required by the static economy receipt."""

    if len(records) != updates:
        raise ReceiptRefused(
            "reward/PPO economy marker count differs from PPO budget"
        )
    rollout_samples = num_envs * ROLLOUT_STEPS_PER_UPDATE
    normalized = []
    learning_rates = []
    closure_errors = []
    post_advantage_mean_errors = []
    post_advantage_std_errors = []
    pre_clip_total_norms = []
    post_clip_total_norms = []
    term_names: Union[set[str], None] = None

    for expected_update, source in enumerate(records):
        row = dict(source)
        row.pop("_line", None)
        if (
            set(row)
            != {
                "event",
                "schema_version",
                "status",
                "ppo_update",
                "gate",
                "reward",
                "advantage",
                "ppo",
                "gradient",
                "policy",
                "checks",
            }
            or row.get("event")
            != "hope_action_ball_reward_ppo_economy_update"
            or row.get("schema_version") != 1
            or row.get("status") != "PASS"
            or row.get("ppo_update") != expected_update
            or row.get("gate")
            != {
                "num_envs": num_envs,
                "steps_per_env_per_update": ROLLOUT_STEPS_PER_UPDATE,
                "rollout_samples_per_update": rollout_samples,
            }
        ):
            raise ReceiptRefused(
                "reward/PPO economy update envelope or gate differs"
            )

        reward = row["reward"]
        expected_reward_keys = {
            "pre_advantage_reward_min_mean_p50_p95_p99_max",
            "return_min_mean_p50_p95_p99_max",
            "return_std",
            "explained_variance",
            "value_prediction_min_mean_p50_p95_p99_max",
            "value_residual_min_mean_p50_p95_p99_max",
            "per_term_raw_sum",
            "per_term_weighted_dt_sum",
            "per_term_eligible_denominator",
            "per_term_denominator_semantics",
            "reward_manager_total_sum",
            "per_term_closure_error",
            "reward_manager_closure_max_abs_error",
            "recipe_sha256",
            "pre_advantage_reward_semantics",
        }
        if type(reward) is not dict or set(reward) != expected_reward_keys:
            raise ReceiptRefused("reward/PPO economy reward fields differ")
        for key in (
            "pre_advantage_reward_min_mean_p50_p95_p99_max",
            "return_min_mean_p50_p95_p99_max",
            "value_prediction_min_mean_p50_p95_p99_max",
            "value_residual_min_mean_p50_p95_p99_max",
        ):
            _finite_economy_stats(
                reward[key], name=f"reward/PPO economy {key}",
                keys=_ECONOMY_DISTRIBUTION_KEYS,
            )
        _finite_positive(reward["return_std"], name="return std")
        _finite_number(
            reward["explained_variance"], name="explained variance"
        )
        if reward["pre_advantage_reward_semantics"] != (
            "ppo_storage_reward_after_timeout_bootstrap"
        ):
            raise ReceiptRefused(
                "reward/PPO economy pre-advantage reward semantics differ"
            )
        raw = reward["per_term_raw_sum"]
        weighted = reward["per_term_weighted_dt_sum"]
        denominators = reward["per_term_eligible_denominator"]
        per_term_error = reward["per_term_closure_error"]
        if (
            type(raw) is not dict
            or type(weighted) is not dict
            or type(denominators) is not dict
            or type(per_term_error) is not dict
            or not raw
            or set(raw) != set(weighted)
            or set(raw) != set(denominators)
            or set(raw) != set(per_term_error)
            or len(raw) != 30
            or list(raw) != sorted(raw)
        ):
            raise ReceiptRefused(
                "reward/PPO economy per-term key set/order differs"
            )
        current_names = set(raw)
        if term_names is None:
            term_names = current_names
        elif current_names != term_names:
            raise ReceiptRefused(
                "reward/PPO economy per-term identity changed across updates"
            )
        for name in sorted(current_names):
            _finite_number(raw[name], name=f"reward raw sum {name}")
            _finite_number(weighted[name], name=f"reward weighted sum {name}")
            if denominators[name] != rollout_samples:
                raise ReceiptRefused(
                    f"reward term {name!r} denominator is not whole-rollout exact"
                )
            _finite_number(
                per_term_error[name],
                name=f"reward term closure error {name}",
                nonnegative=True,
            )
        if reward["per_term_denominator_semantics"] != (
            "all_rollout_environment_samples_including_gated_zero"
        ):
            raise ReceiptRefused(
                "reward/PPO economy per-term denominator semantics differ"
            )
        if reward["recipe_sha256"] != expected_recipe_sha256:
            raise ReceiptRefused("reward/PPO economy recipe SHA differs")
        total = _finite_number(
            reward["reward_manager_total_sum"],
            name="reward manager total sum",
        )
        closure = abs(math.fsum(float(value) for value in weighted.values()) - total)
        tolerance = max(1.0e-6, abs(total) * 1.0e-6)
        manager_closure = _finite_number(
            reward["reward_manager_closure_max_abs_error"],
            name="reward manager closure max error",
            nonnegative=True,
        )
        if closure > tolerance or manager_closure > tolerance:
            raise ReceiptRefused(
                "reward/PPO economy reward sum closure exceeds dtype tolerance"
            )
        closure_errors.append(max(closure, manager_closure))

        advantage = row["advantage"]
        if (
            type(advantage) is not dict
            or set(advantage)
            != {
                "pre_normalization_mean_std_min_max",
                "post_normalization_mean_std_min_max",
                "post_normalization_finite",
                "dtype_tolerance",
                "normalization_population",
            }
            or advantage["post_normalization_finite"] is not True
            or advantage["normalization_population"]
            != "whole_rollout_98304_samples"
        ):
            raise ReceiptRefused("reward/PPO economy advantage fields differ")
        before = _finite_economy_stats(
            advantage["pre_normalization_mean_std_min_max"],
            name="pre-normalization advantage",
            keys=_ECONOMY_ADVANTAGE_KEYS,
        )
        after = _finite_economy_stats(
            advantage["post_normalization_mean_std_min_max"],
            name="post-normalization advantage",
            keys=_ECONOMY_ADVANTAGE_KEYS,
        )
        if before["std"] < 0.0 or after["std"] < 0.0:
            raise ReceiptRefused("reward/PPO economy advantage std is negative")
        advantage_tolerance = _finite_positive(
            advantage["dtype_tolerance"], name="advantage dtype tolerance"
        )
        mean_error = abs(after["mean"])
        std_error = abs(after["std"] - 1.0)
        if mean_error > advantage_tolerance or std_error > advantage_tolerance:
            raise ReceiptRefused(
                "post-normalization advantage is not zero-mean/unit-std"
            )
        post_advantage_mean_errors.append(mean_error)
        post_advantage_std_errors.append(std_error)

        ppo = row["ppo"]
        if (
            type(ppo) is not dict
            or set(ppo)
            != {
                "surrogate_loss",
                "value_loss",
                "entropy_mean",
                "loss_entropy_semantics",
                "approx_kl",
                "approx_kl_semantics",
                "learning_rate",
                "clip_fraction",
                "clip_fraction_semantics",
            }
            or ppo["approx_kl_semantics"]
            != "final_policy_vs_rollout_policy_whole_rollout"
            or ppo["clip_fraction_semantics"]
            != "final_policy_probability_ratio_outside_ppo_clip_whole_rollout"
            or ppo["loss_entropy_semantics"]
            != "arithmetic_mean_over_20_optimizer_minibatches"
        ):
            raise ReceiptRefused("reward/PPO economy PPO fields differ")
        for name in (
            "surrogate_loss",
            "value_loss",
            "entropy_mean",
            "approx_kl",
        ):
            _finite_number(ppo[name], name=f"PPO {name}")
        learning_rate = _finite_positive(
            ppo["learning_rate"], name="PPO learning rate"
        )
        clip_fraction = _finite_number(
            ppo["clip_fraction"], name="PPO clip fraction", nonnegative=True
        )
        if clip_fraction > 1.0:
            raise ReceiptRefused("PPO clip fraction exceeds one")
        learning_rates.append(learning_rate)

        gradient = row["gradient"]
        gradient_fields = (
            "pre_clip_actor_mean_parameter_grad_norm",
            "pre_clip_critic_parameter_grad_norm",
            "pre_clip_std_parameter_grad_norm",
            "pre_clip_total_grad_norm",
            "post_clip_total_grad_norm",
            "max_grad_norm",
        )
        gradient_distribution_fields = (
            "pre_clip_actor_mean_parameter_grad_norm_distribution",
            "pre_clip_critic_parameter_grad_norm_distribution",
            "pre_clip_std_parameter_grad_norm_distribution",
            "pre_clip_total_grad_norm_distribution",
            "post_clip_total_grad_norm_distribution",
            "clip_factor_distribution",
        )
        if (
            type(gradient) is not dict
            or set(gradient)
            != {
                *gradient_fields,
                *gradient_distribution_fields,
                "aggregation",
                "optimizer_minibatch_count",
            }
            or gradient["aggregation"]
            != "arithmetic_mean_over_optimizer_minibatches"
            or gradient["optimizer_minibatch_count"] != 20
        ):
            raise ReceiptRefused("reward/PPO economy gradient fields differ")
        gradient_values = {
            name: _finite_number(
                gradient[name], name=f"gradient {name}", nonnegative=True
            )
            for name in gradient_fields
        }
        distributions = {
            name: _finite_economy_stats(
                gradient[name],
                name=f"gradient {name}",
                keys=_ECONOMY_MIN_MEAN_MAX_KEYS,
            )
            for name in gradient_distribution_fields
        }
        if any(
            value < 0.0
            for distribution in distributions.values()
            for value in distribution.values()
        ):
            raise ReceiptRefused(
                "reward/PPO economy gradient distributions must be non-negative"
            )
        scalar_to_distribution = {
            "pre_clip_actor_mean_parameter_grad_norm": (
                "pre_clip_actor_mean_parameter_grad_norm_distribution"
            ),
            "pre_clip_critic_parameter_grad_norm": (
                "pre_clip_critic_parameter_grad_norm_distribution"
            ),
            "pre_clip_std_parameter_grad_norm": (
                "pre_clip_std_parameter_grad_norm_distribution"
            ),
            "pre_clip_total_grad_norm": (
                "pre_clip_total_grad_norm_distribution"
            ),
            "post_clip_total_grad_norm": (
                "post_clip_total_grad_norm_distribution"
            ),
        }
        if any(
            gradient_values[scalar]
            != distributions[distribution]["mean"]
            for scalar, distribution in scalar_to_distribution.items()
        ):
            raise ReceiptRefused(
                "reward/PPO economy gradient scalar/distribution means differ"
            )
        if (
            gradient_values["max_grad_norm"] != 1.0
            or distributions[
                "post_clip_total_grad_norm_distribution"
            ]["max"]
            > 1.0 + 1.0e-5
            or distributions["clip_factor_distribution"]["max"] > 1.0
        ):
            raise ReceiptRefused("reward/PPO economy gradient clip contract differs")
        pre_clip_total_norms.append(
            gradient_values["pre_clip_total_grad_norm"]
        )
        post_clip_total_norms.append(
            gradient_values["post_clip_total_grad_norm"]
        )

        policy = row["policy"]
        if type(policy) is not dict or set(policy) != {
            "noise_std_type",
            "policy_std_min",
            "policy_std_mean",
            "policy_std_max",
        } or policy["noise_std_type"] != "log":
            raise ReceiptRefused("reward/PPO economy policy fields differ")
        policy_min = _finite_positive(
            policy["policy_std_min"], name="policy std min"
        )
        policy_mean = _finite_positive(
            policy["policy_std_mean"], name="policy std mean"
        )
        policy_max = _finite_positive(
            policy["policy_std_max"], name="policy std max"
        )
        if not policy_min <= policy_mean <= policy_max:
            raise ReceiptRefused("reward/PPO economy policy std ordering differs")
        if row["checks"] != {
            "all_required_fields_present": True,
            "all_required_values_finite": True,
            "reward_sum_closure": "PASS",
            "post_advantage_zero_mean_unit_std": "PASS",
            "noise_std_type_log": True,
            "policy_std_strictly_positive": True,
        }:
            raise ReceiptRefused("reward/PPO economy check vector differs")
        normalized.append(row)

    floor_count = sum(value <= 1.0e-5 for value in learning_rates)
    if floor_count == updates:
        raise ReceiptRefused(
            "reward/PPO economy learning rate stayed at the 1e-5 floor for all updates"
        )
    return {
        "status": "PASS",
        "updates": normalized,
        "summary": {
            "update_count": updates,
            "rollout_samples_per_update": rollout_samples,
            "active_reward_term_count": len(term_names or ()),
            "learning_rate_floor_update_count": floor_count,
            "all_updates_learning_rate_at_floor": False,
            "max_reward_closure_abs_error": max(closure_errors),
            "max_post_advantage_mean_abs_error": max(
                post_advantage_mean_errors
            ),
            "max_post_advantage_std_abs_error": max(
                post_advantage_std_errors
            ),
            "max_pre_clip_total_grad_norm": max(pre_clip_total_norms),
            "max_post_clip_total_grad_norm": max(post_clip_total_norms),
            "noise_std_type": "log",
            "all_required_fields_present_and_finite": True,
        },
    }


def _cross_validate_std_lr_and_economy(
    std_lr: list[dict[str, Any]], economy: dict[str, Any]
) -> dict[str, bool]:
    """Require both independently parsed runtime markers to agree exactly."""

    economy_updates = economy.get("updates")
    if type(economy_updates) is not list or len(std_lr) != len(economy_updates):
        raise ReceiptRefused(
            "policy std/LR and reward/PPO economy update counts differ"
        )
    for expected, (std_row, economy_row) in enumerate(
        zip(std_lr, economy_updates)
    ):
        policy = economy_row.get("policy")
        ppo = economy_row.get("ppo")
        if (
            type(policy) is not dict
            or type(ppo) is not dict
            or std_row.get("ppo_update") != expected
            or economy_row.get("ppo_update") != expected
            or std_row.get("policy_std_min") != policy.get("policy_std_min")
            or std_row.get("policy_std_mean") != policy.get("policy_std_mean")
            or std_row.get("policy_std_max") != policy.get("policy_std_max")
            or std_row.get("learning_rate") != ppo.get("learning_rate")
        ):
            raise ReceiptRefused(
                "policy std/LR and reward/PPO economy markers disagree"
            )
    return {
        "policy_std_exact": True,
        "learning_rate_exact": True,
    }


def _sum_numeric_counters(
    records: Iterable[Mapping[str, Any]]
) -> dict[str, Union[int, float]]:
    totals: dict[str, Union[int, float]] = {}
    for record in records:
        counters = record.get("counters")
        if type(counters) is not dict:
            raise ReceiptRefused("behavior marker lacks counters")
        for key, value in counters.items():
            if type(key) is not str or type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ReceiptRefused("behavior counters must be finite and non-negative")
            totals[key] = totals.get(key, 0) + value
    return dict(sorted(totals.items()))


def _validate_behavior(
    records: list[dict[str, Any]], *, stage: str, updates: int, num_envs: int
) -> dict[str, Any]:
    if stage != "probe":
        raise ReceiptRefused("integrated gate accepts only the probe stage")
    if len(records) != updates:
        raise ReceiptRefused("behavior marker count differs from PPO budget")
    for expected, row in enumerate(records):
        if row.get("event") != "hope_exact_behavior_update" or row.get("schema_version") != 1 or row.get("ppo_update") != expected:
            raise ReceiptRefused("behavior update sequence differs")
    normalized_updates = []
    for source in records:
        row = dict(source)
        row.pop("_line", None)
        normalized_updates.append(row)
    totals = _sum_numeric_counters(normalized_updates)
    physical = _nonnegative_int(totals.get("physical_fall_count"), name="physical fall count")
    pre = _nonnegative_int(totals.get("pre_strike_physical_fall_count"), name="pre-strike fall count")
    post = _nonnegative_int(totals.get("post_strike_physical_fall_count"), name="post-strike fall count")
    terminal = _nonnegative_int(totals.get("terminal_reset_count"), name="terminal reset count")
    nonphysical = _nonnegative_int(totals.get("non_physical_terminal_reset_count"), name="nonphysical terminal count")
    physical_partition_matches = physical == pre + post
    terminal_partition_matches = terminal == physical + nonphysical
    entry = _nonnegative_int(totals.get(_ENTRY_COUNT), name="strike entry count")
    entry_nonfinite = _nonnegative_int(totals.get(_ENTRY_NONFINITE), name="strike entry nonfinite")
    bucket_total = sum(_nonnegative_int(totals.get(key), name=key) for key in _ENTRY_BUCKETS)
    entry_histogram_matches = entry == bucket_total + entry_nonfinite
    union = _nonnegative_int(totals.get("reference_guard_union_count"), name="reference union")
    reference_only = _nonnegative_int(totals.get("reference_guard_reference_only_count"), name="reference only")
    reference_hard = _nonnegative_int(totals.get("reference_guard_reference_and_hard_count"), name="reference hard")
    hard_without_snapshot = _nonnegative_int(
        totals.get("reference_guard_hard_without_snapshot_count"),
        name="reference hard without snapshot",
    )
    reference_guard_matches = (
        union == reference_only + reference_hard and hard_without_snapshot == 0
    )
    hard_terminal_reasons = {
        key: value
        for key, value in totals.items()
        if key.startswith("termination_reason_")
        and (
            "joint_actual_forbidden" in key
            or "joint_qdes_forbidden" in key
        )
        and value != 0
    }
    if entry_nonfinite != 0 or totals.get("ready_nonfinite_value_count") != 0 or hard_terminal_reasons:
        raise ReceiptRefused(
            "behavior evidence contains joint-hard/nonfinite failure: "
            f"{hard_terminal_reasons}"
        )
    terminal_reasons = {
        key[len("termination_reason_") : -len("_count")]: value
        for key, value in totals.items()
        if key.startswith("termination_reason_") and key.endswith("_count")
    }
    env_policy_steps = num_envs * ROLLOUT_STEPS_PER_UPDATE * updates
    table_count = sum(
        value for key, value in terminal_reasons.items() if "table" in key
    )
    attribution_total = _nonnegative_int(
        totals.get("table_guard_first_hit_total_count"),
        name="table-guard first-hit total",
    )
    category_counts = {
        name: _nonnegative_int(
            totals.get(f"table_guard_first_hit_category_{name}_count"),
            name=f"table-guard category {name}",
        )
        for name in _TABLE_ATTRIBUTION_CATEGORIES
    }
    phase_counts = {
        name: _nonnegative_int(
            totals.get(f"table_guard_first_hit_phase_{name}_count"),
            name=f"table-guard phase {name}",
        )
        for name in _TABLE_ATTRIBUTION_PHASES
    }
    cell_total = sum(
        _nonnegative_int(value, name=f"table-guard cell {key}")
        for key, value in totals.items()
        if key.startswith("table_guard_first_hit_cell_")
    )
    attribution_conserves = (
        attribution_total == table_count
        and sum(category_counts.values()) == attribution_total
        and sum(phase_counts.values()) == attribution_total
        and cell_total == attribution_total
        and category_counts["nonfinite"] == 0
    )
    if category_counts["nonfinite"] != 0:
        raise ReceiptRefused(
            "table-guard attribution contains non-finite evidence"
        )
    table_attribution = {
        "enabled": True,
        "first_hit_total_count": attribution_total,
        "terminal_count": table_count,
        "category_counts": category_counts,
        "phase_counts": phase_counts,
        "sparse_cell_total_count": cell_total,
        "conserves": attribution_conserves,
        "telemetry_only": True,
    }
    table_rate = float(table_count) / float(env_policy_steps)
    fall_rate = float(physical) / float(env_policy_steps)
    rate_limits = BEHAVIOR_RATE_LIMITS[stage]
    timeout = _nonnegative_int(totals.get("timeout_reset_count"), name="timeout count")
    conservative_age = float(env_policy_steps) / float(num_envs + terminal + timeout)
    strike = _nonnegative_int(totals.get("strike_opportunity_count"), name="strike opportunity count")
    swing_start = _nonnegative_int(totals.get("swing_start_count"), name="swing start count")
    swing_outcome = _nonnegative_int(totals.get("swing_outcome_count"), name="swing outcome count")
    within_telemetry_thresholds = not (
        table_rate > rate_limits["table_contact_per_env_step"]
        or fall_rate > rate_limits["fall_per_env_step"]
        or conservative_age < MIN_CONSERVATIVE_EPISODE_AGE_STEPS
        or entry <= 0
        or strike <= 0
        or swing_start <= 0
        or swing_outcome <= 0
    )
    return {
        "updates": normalized_updates,
        "aggregate_counters": totals,
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        "terminal_conservation": {
            "terminal_reset_count": terminal,
            "physical_fall_count": physical,
            "non_physical_terminal_reset_count": nonphysical,
            "physical_partition_matches": physical_partition_matches,
            "terminal_partition_matches": terminal_partition_matches,
            "telemetry_only": True,
        },
        "strike_window_entry_conservation": {
            "entry_count": entry,
            "finite_bucket_total": bucket_total,
            "nonfinite_count": entry_nonfinite,
            "matches": entry_histogram_matches,
            "telemetry_only": True,
        },
        "reference_guard_conservation": {
            "union_count": union,
            "reference_only_count": reference_only,
            "reference_and_hard_count": reference_hard,
            "hard_without_snapshot_count": hard_without_snapshot,
            "matches": reference_guard_matches,
            "telemetry_only": True,
        },
        "table_guard_attribution": table_attribution,
        "reachability_and_failure_rates": {
            "environment_policy_step_denominator": env_policy_steps,
            "table_contact_count": table_count,
            "table_contact_per_env_step": table_rate,
            "table_contact_per_env_step_limit": rate_limits[
                "table_contact_per_env_step"
            ],
            "physical_fall_count": physical,
            "physical_fall_per_env_step": fall_rate,
            "physical_fall_per_env_step_limit": rate_limits[
                "fall_per_env_step"
            ],
            "conservative_mean_episode_age_steps": conservative_age,
            "minimum_conservative_mean_episode_age_steps": (
                MIN_CONSERVATIVE_EPISODE_AGE_STEPS
            ),
            "strike_opportunity_count": strike,
            "swing_start_count": swing_start,
            "swing_outcome_count": swing_outcome,
            "within_telemetry_thresholds": within_telemetry_thresholds,
            "telemetry_only": True,
        },
    }


def _validate_joint_safety(
    records: list[dict[str, Any]],
    fatal_records: list[dict[str, Any]],
    *,
    updates: int,
    num_envs: int,
) -> dict[str, Any]:
    if fatal_records:
        raise ReceiptRefused("joint-safety fatal marker is present")
    if len(records) != updates:
        raise ReceiptRefused("joint-safety marker count differs from PPO budget")
    totals: dict[str, int] = {}
    minimum_gap: Union[float, None] = None
    normalized = []
    for expected, source in enumerate(records):
        row = dict(source)
        row.pop("_line", None)
        counters = row.get("counter_totals")
        if (
            row.get("event")
            != "hope_joint_safety_diagnostic_compact_update"
            or row.get("schema_version") != 1
            or row.get("status")
            != "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
            or row.get("ppo_update") != expected
            or row.get("num_envs") != num_envs
            or row.get("policy_step_count") != ROLLOUT_STEPS_PER_UPDATE
            or type(counters) is not dict
            or counters.get("policy_steps")
            != num_envs * ROLLOUT_STEPS_PER_UPDATE
            or counters.get("complete_policy_steps")
            != num_envs * ROLLOUT_STEPS_PER_UPDATE
        ):
            raise ReceiptRefused("joint-safety update sequence differs")
        for key, value in counters.items():
            totals[key] = totals.get(key, 0) + _nonnegative_int(value, name=f"joint safety {key}")
        gap = row.get("minimum_hard_gap_rad")
        if type(gap) not in (int, float) or not math.isfinite(float(gap)) or float(gap) <= 0:
            raise ReceiptRefused("joint-safety minimum hard gap must stay positive")
        minimum_gap = float(gap) if minimum_gap is None else min(minimum_gap, float(gap))
        normalized.append(row)
    forbidden = {
        key: totals.get(key, 0)
        for key in (
            "actual_hard_edge_events",
            "qdes_events",
        )
        if totals.get(key, 0) != 0
    }
    if forbidden:
        raise ReceiptRefused(f"joint-safety zero gate failed: {forbidden}")
    return {
        "updates": normalized,
        "aggregate_counter_totals": dict(sorted(totals.items())),
        "minimum_hard_gap_rad": minimum_gap,
        "fatal_marker_count": 0,
    }


def _validate_push_velocity(
    records: list[dict[str, Any]], *, stage: str, updates: int, num_envs: int
) -> dict[str, Any]:
    if stage != "probe":
        raise ReceiptRefused("integrated gate accepts only the probe stage")
    if len(records) != updates:
        raise ReceiptRefused(
            "velocity-push diagnostic marker count differs from PPO budget"
        )
    event_calls = 0
    env_applications = 0
    nonfinite = 0
    below = 0
    above = 0
    observed_axis_extrema: dict[str, list[float]] = {
        axis: [] for axis in PUSH_VELOCITY_AXES
    }
    normalized = []
    for expected, source in enumerate(records):
        row = dict(source)
        row.pop("_line", None)
        counters = row.get("counters")
        axes = counters.get("axes") if type(counters) is dict else None
        if (
            row.get("event") != "hope_push_velocity_diagnostic_update"
            or row.get("schema_version") != 1
            or row.get("ppo_update") != expected
            or type(axes) is not dict
            or set(axes) != set(PUSH_VELOCITY_AXES)
        ):
            raise ReceiptRefused("velocity-push diagnostic update differs")
        row_event_calls = _nonnegative_int(
            counters.get("event_call_count"), name="push event call count"
        )
        event_calls += row_event_calls
        row_env_applications = _nonnegative_int(
            counters.get("env_application_count"),
            name="push environment application count",
        )
        env_applications += row_env_applications
        if (row_event_calls == 0) != (row_env_applications == 0):
            raise ReceiptRefused(
                "push event-call and environment-application counts disagree"
            )
        nonfinite += _nonnegative_int(
            counters.get("delta_nonfinite_element_count"),
            name="push nonfinite delta count",
        )
        for axis, values in axes.items():
            if type(values) is not dict:
                raise ReceiptRefused(f"push axis {axis} evidence is incomplete")
            below += _nonnegative_int(
                values.get("below_range_count"), name=f"push {axis} below count"
            )
            above += _nonnegative_int(
                values.get("above_range_count"), name=f"push {axis} above count"
            )
            minimum = values.get("observed_delta_min")
            maximum = values.get("observed_delta_max")
            if row_event_calls == 0 and (minimum is not None or maximum is not None):
                raise ReceiptRefused("empty push window reports fabricated extrema")
            if row_event_calls > 0 and (minimum is None or maximum is None):
                raise ReceiptRefused("active push window omits observed extrema")
            for label, value in (("minimum", minimum), ("maximum", maximum)):
                if value is not None and (
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                ):
                    raise ReceiptRefused(
                        f"push {axis} observed {label} is non-finite"
                    )
            if minimum is not None and maximum is not None:
                if float(minimum) > float(maximum):
                    raise ReceiptRefused(
                        f"push {axis} observed extrema are reversed"
                    )
                lower, upper = PUSH_VELOCITY_BOUNDS[axis]
                if float(minimum) < lower or float(maximum) > upper:
                    raise ReceiptRefused(
                        f"push {axis} raw extrema exceed the pinned vendor range"
                    )
                observed_axis_extrema[axis].extend(
                    (float(minimum), float(maximum))
                )
        normalized.append(row)
    if nonfinite != 0 or below != 0 or above != 0:
        raise ReceiptRefused("velocity-push evidence contains nonfinite/range breach")
    if event_calls <= 0 or env_applications <= 0:
        raise ReceiptRefused(
            "integrated probe did not observe a nonzero velocity push"
        )
    axis_extrema = {
        axis: {
            "observed_delta_min": min(values),
            "observed_delta_max": max(values),
            "finite_and_in_range": True,
        }
        for axis, values in observed_axis_extrema.items()
    }
    return {
        "updates": normalized,
        "aggregate": {
            "event_call_count": event_calls,
            "env_application_count": env_applications,
            "delta_nonfinite_element_count": nonfinite,
            "below_range_count": below,
            "above_range_count": above,
            "six_axis_extrema_finite_and_in_range": True,
        },
        "axis_extrema": axis_extrema,
    }


def _normalizer_checkpoint_summary(
    state: Any,
    *,
    role: str,
    expected_features: int,
    torch_module: Any,
) -> dict[str, Any]:
    if not isinstance(state, Mapping) or not state:
        raise ReceiptRefused(f"checkpoint lacks {role} normalizer state")
    aliases = {
        "mean": {"mean", "_mean", "running_mean"},
        "scale": {
            "std",
            "_std",
            "running_std",
            "var",
            "_var",
            "variance",
            "running_var",
        },
        "count": {"count", "_count", "running_count"},
    }
    semantic: dict[str, list[str]] = {key: [] for key in aliases}
    tensors = []
    for key, value in state.items():
        if type(key) is not str or not torch_module.is_tensor(value):
            raise ReceiptRefused(
                f"checkpoint {role} normalizer must contain string-keyed tensors"
            )
        if int(value.numel()) <= 0:
            raise ReceiptRefused(f"checkpoint {role} normalizer tensor is empty")
        if not bool(torch_module.isfinite(value).all().item()):
            raise ReceiptRefused(
                f"checkpoint {role} normalizer contains non-finite tensors"
            )
        tensors.append(value)
        leaf = key.rsplit(".", 1)[-1]
        for semantic_name, names in aliases.items():
            if leaf in names:
                semantic[semantic_name].append(key)
    if any(len(semantic[name]) != 1 for name in ("mean", "scale", "count")):
        raise ReceiptRefused(
            f"checkpoint {role} normalizer semantic buffers are ambiguous"
        )
    mean_key = semantic["mean"][0]
    scale_key = semantic["scale"][0]
    count_key = semantic["count"][0]
    mean = state[mean_key]
    scale = state[scale_key]
    count = state[count_key]
    if (
        tuple(mean.shape) != tuple(scale.shape)
        or int(mean.numel()) != expected_features
        or int(scale.numel()) != expected_features
        or int(count.numel()) != 1
        or not bool(mean.is_floating_point())
        or not bool(scale.is_floating_point())
    ):
        raise ReceiptRefused(
            f"checkpoint {role} normalizer feature/count shape differs"
        )
    if bool((scale < 0).any().item()):
        raise ReceiptRefused(
            f"checkpoint {role} normalizer scale is negative"
        )
    try:
        count_value = float(count.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReceiptRefused(
            f"checkpoint {role} normalizer count is not real-valued"
        ) from exc
    if not math.isfinite(count_value) or count_value <= 0.0:
        raise ReceiptRefused(
            f"checkpoint {role} normalizer count must be finite and positive"
        )
    return {
        "state_keys": sorted(state),
        "mean_key": mean_key,
        "scale_key": scale_key,
        "count_key": count_key,
        "feature_count": expected_features,
        "count": count_value,
        "tensor_count": len(tensors),
        "element_count": sum(int(value.numel()) for value in tensors),
        "all_finite": True,
    }


def _checkpoint_summary(
    path: Path,
    *,
    expected_iteration: int,
    expected_claim_sha256: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Pod/runtime dependency
        raise ReceiptRefused("PyTorch is required to inspect checkpoints") from exc
    raw = _read_stable_bytes(path, name="checkpoint")
    try:
        checkpoint = torch.load(
            io.BytesIO(raw), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ReceiptRefused(f"checkpoint cannot be loaded: {path}: {exc}") from exc
    infos = checkpoint.get("infos") if type(checkpoint) is dict else None
    if (
        checkpoint.get("iter") != expected_iteration
        or type(infos) is not dict
        or infos.get("training_launch_claim_sha256") != expected_claim_sha256
        or infos.get("training_contract_sha256") != expected_contract_sha256
    ):
        raise ReceiptRefused(
            "checkpoint iteration/launch-claim/training-contract lineage differs"
        )
    state = checkpoint.get("model_state_dict") if type(checkpoint) is dict else None
    if not isinstance(state, Mapping) or not state:
        raise ReceiptRefused(f"checkpoint lacks model_state_dict: {path}")
    tensors = []
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise ReceiptRefused(f"checkpoint state {name!r} is not a tensor")
        tensors.append(value)
    if not tensors or not all(bool(torch.isfinite(value).all().item()) for value in tensors):
        raise ReceiptRefused(f"checkpoint contains non-finite model tensors: {path}")
    if "std" in state or "log_std" not in state:
        raise ReceiptRefused(
            "checkpoint policy std ABI must contain log_std and no legacy std"
        )
    log_std = state["log_std"]
    if tuple(log_std.shape) != (31,) or int(log_std.numel()) != 31:
        raise ReceiptRefused(
            "checkpoint log_std must cover exactly 31 action dimensions"
        )
    realized_std = torch.exp(log_std.detach())
    realized_values = torch.stack(
        (realized_std.min(), realized_std.mean(), realized_std.max())
    ).tolist()
    if (
        not all(math.isfinite(float(value)) for value in realized_values)
        or float(realized_values[0]) <= 0.0
        or not realized_values[0] <= realized_values[1] <= realized_values[2]
    ):
        raise ReceiptRefused(
            "checkpoint realized log_std is not finite and strictly positive"
        )
    actor_normalizer = _normalizer_checkpoint_summary(
        checkpoint.get("obs_norm_state_dict"),
        role="actor",
        expected_features=194,
        torch_module=torch,
    )
    critic_normalizer = _normalizer_checkpoint_summary(
        checkpoint.get("privileged_obs_norm_state_dict"),
        role="critic",
        expected_features=318,
        torch_module=torch,
    )
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "embedded_iteration": expected_iteration,
        "training_launch_claim_sha256": expected_claim_sha256,
        "training_contract_sha256": expected_contract_sha256,
        "tensor_count": len(tensors),
        "element_count": sum(int(value.numel()) for value in tensors),
        "all_finite": True,
        "policy_std_parameter": {
            "noise_std_type": "log",
            "parameter_name": "log_std",
            "parameter_shape": [31],
            "parameter_count": 31,
            "realized_policy_std_min": float(realized_values[0]),
            "realized_policy_std_mean": float(realized_values[1]),
            "realized_policy_std_max": float(realized_values[2]),
        },
        "actor_normalizer": actor_normalizer,
        "critic_normalizer": critic_normalizer,
    }


def _inspect_checkpoints(
    run_dir: Path,
    *,
    stage: str,
    expected_claim_sha256: str,
    expected_contract_sha256: str,
) -> list[dict[str, Any]]:
    expected = EXPECTED_CHECKPOINT_INDICES[stage]
    observed: dict[int, Path] = {}
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_([0-9]+)\.pt", path.name)
        if match:
            observed[int(match.group(1))] = _real_file(path, name="checkpoint")
    if tuple(sorted(observed)) != expected:
        raise ReceiptRefused(
            f"{stage} checkpoint indices differ: expected={expected}, "
            f"observed={tuple(sorted(observed))}"
        )
    result = []
    previous_counts: Union[dict[str, float], None] = None
    for index in expected:
        path = observed[index]
        summary = _checkpoint_summary(
            path,
            expected_iteration=index,
            expected_claim_sha256=expected_claim_sha256,
            expected_contract_sha256=expected_contract_sha256,
        )
        counts = {
            role: summary[f"{role}_normalizer"]["count"]
            for role in ("actor", "critic")
        }
        if previous_counts is not None and any(
            counts[role] < previous_counts[role] for role in counts
        ):
            raise ReceiptRefused(
                f"{stage} checkpoint normalizer count regressed"
            )
        previous_counts = counts
        result.append(
            {
                "index": index,
                "path": str(path),
                **summary,
            }
        )
    return result


def _scientific_argv(argv: Any) -> tuple[list[str], str]:
    if type(argv) is not list or any(type(item) is not str for item in argv):
        raise ReceiptRefused("claim training argv must be a list of strings")
    normalized = [
        item
        for item in argv[2:]
        if not item.startswith(
            _VARIABLE_ARG_PREFIXES + _DIAGNOSTIC_ARG_PREFIXES
        )
    ]
    if normalized.count(_V.STABLE_READY_PLANT_OVERRIDE) != 1:
        raise ReceiptRefused("scientific argv must contain stable-ready exactly once")
    if normalized.count(_V.VENDOR_POLICY_NOISE_STD_OVERRIDE) != 1:
        raise ReceiptRefused(
            "scientific argv must contain vendor log_std exactly once"
        )
    actor_arg = f"task.actor_obs_contract={ACTOR_OBS_CONTRACT}"
    if normalized.count(actor_arg) != 1:
        raise ReceiptRefused("scientific argv actor observation contract differs")
    return normalized, _canonical_sha(normalized)


def _source_pin(runtime_sources: Mapping[str, Any], label: str) -> dict[str, str]:
    value = runtime_sources.get(label)
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ReceiptRefused(f"claim lacks exact runtime source pin: {label}")
    return {"path": value["path"], "sha256": value["sha256"]}


def _action_artifact_pin(
    action: Any, field: str, *, layer: str
) -> dict[str, str]:
    """Resolve one materialized action pin from the exact registry source."""

    try:
        pin = dict(
            _V._R.require_materialized_pin(
                getattr(action, field),
                action_id=action.action_id,
                layer=layer,
            )
        )
    except (AttributeError, _V._R.VendorActionRegistryError) as exc:
        raise ReceiptRefused(
            f"claim action registry lacks materialized {layer}: {exc}"
        ) from exc
    if (
        set(pin) != {"path", "sha256"}
        or type(pin["path"]) is not str
        or not pin["path"]
        or type(pin["sha256"]) is not str
        or _SHA_RE.fullmatch(pin["sha256"]) is None
    ):
        raise ReceiptRefused(f"claim action registry {layer} pin is invalid")
    return pin


def _contact_timing(payload: Mapping[str, Any], checkout: Path) -> dict[str, Any]:
    bundle = payload.get("bundle")
    contact = bundle.get("contact_alignment") if type(bundle) is dict else None
    if type(contact) is not dict:
        raise ReceiptRefused("claim lacks validated contact alignment pin")
    path = checkout / contact["path"]
    contact_raw = _read_stable_bytes(path, name="contact alignment")
    if hashlib.sha256(contact_raw).hexdigest() != contact["sha256"]:
        raise ReceiptRefused("contact alignment file SHA differs from claim")
    document = _strict_json_bytes(contact_raw, name="contact alignment")
    timing = document.get("timing")
    if type(timing) is not dict or timing.get("manifest_t_hit_s") != timing.get("motion_t_hit_s"):
        raise ReceiptRefused("contact t_hit timing is not exact")
    return {
        "fps_hz": timing["fps_hz"],
        "frame_count": timing["frame_count"],
        "contact_frame": timing["contact_frame"],
        "t_hit_s": timing["motion_t_hit_s"],
        "t_cycle_s": timing["motion_t_cycle_s"],
        "t_hit_abs_error_s": timing["t_hit_abs_error_s"],
        "t_cycle_abs_error_s": timing["t_cycle_abs_error_s"],
        "center_gate_distance_m": bundle["contact_summary"]["center_gate_distance_m"],
        "center_threshold_m": bundle["contact_summary"]["center_threshold_m"],
    }


def _scientific_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = payload.get("spec")
    source = spec.get("source") if type(spec) is dict else None
    bundle = payload.get("bundle")
    runtime_sources = payload.get("runtime_sources")
    authority = payload.get("vendor_runtime_authority")
    if not all(type(value) is dict for value in (spec, source, bundle, runtime_sources, authority)):
        raise ReceiptRefused("claim scientific identity is incomplete")
    try:
        action = _V._R.get_action_config(spec.get("action_id"))
    except _V._R.VendorActionRegistryError as exc:
        raise ReceiptRefused(f"claim action is not registry-authorized: {exc}") from exc
    expected_bundle = _action_artifact_pin(
        action, "contact_bundle", layer="contact bundle"
    )
    expected_identity = _action_artifact_pin(
        action,
        "required_identity_manifest",
        layer="required identity manifest",
    )
    expected_authority = _action_artifact_pin(
        action,
        "runtime_authority_receipt",
        layer="runtime authority receipt",
    )
    expected_contract = _action_artifact_pin(
        action, "runtime_contract", layer="runtime contract"
    )
    expected_fixed_domain = _action_artifact_pin(
        action,
        "fixed_domain_initial_receipt",
        layer="fixed-domain initial receipt",
    )
    expected_reward_economy = _action_artifact_pin(
        action, "reward_economy_receipt", layer="reward economy receipt"
    )
    authority_contract = authority.get("runtime_training_contract")
    verified_runtime = authority.get("verified_vendor_runtime")
    if spec.get("bundle") != expected_bundle:
        raise ReceiptRefused(
            "claim bundle differs from its action-specific registry pin"
        )
    if spec.get(_V.VENDOR_CONTRACT_FIELD) != expected_contract["sha256"]:
        raise ReceiptRefused(
            "claim vendor contract differs from its action-specific registry pin"
        )
    if spec.get(_V.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD) != expected_fixed_domain:
        raise ReceiptRefused(
            "claim fixed-domain initial receipt differs from its "
            "action-specific registry pin"
        )
    if spec.get(_V.REWARD_ECONOMY_RECEIPT_FIELD) != expected_reward_economy:
        raise ReceiptRefused(
            "claim reward economy receipt differs from its action-specific "
            "registry pin"
        )
    if (
        authority.get("receipt_path") != expected_authority["path"]
        or authority.get("receipt_sha256") != expected_authority["sha256"]
        or type(authority_contract) is not dict
        or authority_contract.get("path") != expected_contract["path"]
        or authority_contract.get("sha256") != expected_contract["sha256"]
        or authority_contract.get("schema_version") != 3
        or type(verified_runtime) is not dict
        or verified_runtime.get("action_id") != action.action_id
    ):
        raise ReceiptRefused(
            "claim runtime authority differs from its action-specific registry pins"
        )
    checkout = _real_dir(source["checkout"], name="evidence checkout")
    argv, science_sha = _scientific_argv(payload.get("training_argv"))
    dynamic = bundle.get("dynamic_ready")
    if type(dynamic) is not dict:
        raise ReceiptRefused("claim lacks dynamic-ready pins")
    motion = bundle.get("motion")
    if type(motion) is not dict:
        raise ReceiptRefused("claim lacks motion pin")
    return {
        "action_id": spec["action_id"],
        "scope": spec["scope"],
        "seed": spec["seed"],
        "task_profile": _source_pin(
            runtime_sources, f"immutable task profile {_V.TASK_PROFILE_ID}"
        ),
        "robot_source": _source_pin(runtime_sources, "vendor A3 robot source"),
        "action_registry": _source_pin(
            runtime_sources, "A3 vendor action registry"
        ),
        "bundle": expected_bundle,
        "dynamic_ready_candidate": dict(dynamic["artifact"]),
        "nominal_hold_receipt": dict(dynamic["nominal_hold_receipt"]),
        "vendor_runtime_training_contract_sha256": spec[_V.VENDOR_CONTRACT_FIELD],
        "required_identity": expected_identity,
        "runtime_authority_receipt": expected_authority,
        "runtime_authority_receipt_sha256": authority["receipt_sha256"],
        "policy_contract_sha256": spec["policy_contract_sha256"],
        "sigma_profile": spec.get(
            _V.SIGMA_PROFILE_FIELD, _V.STATIC_SIGMA_PROFILE
        ),
        "sigma_variant_scientific_identity_sha256": spec.get(
            _V.SIGMA_VARIANT_IDENTITY_FIELD,
            spec["policy_contract_sha256"],
        ),
        "effective_reward_recipe_sha256": spec[
            "expected_effective_reward_recipe_sha256"
        ],
        _V.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD: expected_fixed_domain,
        _V.REWARD_ECONOMY_RECEIPT_FIELD: expected_reward_economy,
        "stable_ready_plant_override_count": argv.count(
            _V.STABLE_READY_PLANT_OVERRIDE
        ),
        "actor_observation_contract": ACTOR_OBS_CONTRACT,
        "scientific_argv_canonical_sha256": science_sha,
        "motion": dict(motion),
        "contact_timing": _contact_timing(payload, checkout),
    }


def _stage_evidence(namespace: Path, run_dir: Path, *, expected_stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    namespace = _real_dir(namespace, name=f"{expected_stage} namespace")
    run_dir = _real_dir(run_dir, name=f"{expected_stage} run directory")
    _claim, payload, claim_pin = _load_claim(namespace)
    spec = payload.get("spec")
    expected_budget = EXPECTED_STAGES[expected_stage]
    if (
        type(spec) is not dict
        or spec.get("stage") != expected_stage
        or any(spec.get(key) != value for key, value in expected_budget.items())
        or spec.get("namespace") != str(namespace)
        or Path(spec.get("log_path", "")) != namespace / "run.log"
    ):
        raise ReceiptRefused(f"{expected_stage} claim/spec/budget differs")
    expected_run_suffix = f"_{namespace.name}-DIAGNOSTIC_UNAUTHORIZED"
    if not run_dir.name.endswith(expected_run_suffix):
        raise ReceiptRefused(
            f"{expected_stage} run directory is not bound to claim run_name"
        )
    source_commit = spec.get("source", {}).get("commit_sha")
    if _COMMIT_RE.fullmatch(source_commit or "") is None:
        raise ReceiptRefused("stage source commit is invalid")
    markers, log_sha = _parse_markers(namespace / "run.log")
    updates = expected_budget["max_iterations"]
    policy_bootstrap = _validate_policy_bootstrap(
        markers["policy_bootstrap"]
    )
    abi = _validate_abi(markers["abi"])
    delay = _validate_delay(markers["delay"], num_envs=expected_budget["num_envs"])
    # These are deliberately different contract layers.  The live hard SHA
    # covers the complete task materialization and is bound through delay,
    # checkpoints, and completion.  The vendor SHA covers the reviewed A3
    # plant authority and is independently bound by spec/argv/completion.
    # Requiring equality would reject the valid 89082 hierarchy
    # (hard=5727fc46..., vendor=38974f1b...).
    hard_contract_sha256 = delay["training_contract_sha256"]
    std_lr = _validate_std_lr(markers["std_lr"], updates=updates)
    reward_ppo_economy = _validate_reward_ppo_economy(
        markers["reward_ppo_economy"],
        updates=updates,
        num_envs=expected_budget["num_envs"],
        expected_recipe_sha256=spec[
            "expected_effective_reward_recipe_sha256"
        ],
    )
    reward_ppo_economy_cross_source = _cross_validate_std_lr_and_economy(
        std_lr, reward_ppo_economy
    )
    joint = _validate_joint_safety(
        markers["joint_safety"],
        markers["joint_safety_fatal"],
        updates=updates,
        num_envs=expected_budget["num_envs"],
    )
    behavior = _validate_behavior(
        markers["behavior"],
        stage=expected_stage,
        updates=updates,
        num_envs=expected_budget["num_envs"],
    )
    push_velocity = _validate_push_velocity(
        markers["push_velocity"],
        stage=expected_stage,
        updates=updates,
        num_envs=expected_budget["num_envs"],
    )
    checkpoints = _inspect_checkpoints(
        run_dir,
        stage=expected_stage,
        expected_claim_sha256=claim_pin["launch_claim_sha256"],
        expected_contract_sha256=hard_contract_sha256,
    )
    completion = _validate_completion(
        markers["completion"],
        stage=expected_stage,
        num_envs=expected_budget["num_envs"],
        updates=updates,
        expected_claim_sha256=claim_pin["launch_claim_sha256"],
        expected_hard_contract_sha256=hard_contract_sha256,
        expected_vendor_contract_sha256=spec[_V.VENDOR_CONTRACT_FIELD],
    )
    stage = {
        "stage": expected_stage,
        "namespace": str(namespace),
        "run_directory": str(run_dir),
        "launch_claim": claim_pin,
        "run_log": {"path": str(namespace / "run.log"), "sha256": log_sha},
        "source_commit": source_commit,
        "budget": dict(expected_budget),
        "checkpoints": checkpoints,
        "policy_bootstrap": policy_bootstrap,
        "runtime_abi": abi,
        "control_step_action_delay": delay,
        "policy_std_lr_updates": std_lr,
        "reward_ppo_economy": reward_ppo_economy,
        "reward_ppo_economy_cross_source": (
            reward_ppo_economy_cross_source
        ),
        "joint_safety": joint,
        "behavior": behavior,
        "push_velocity_diagnostic": push_velocity,
        "training_completion": completion,
    }
    sources = payload.get(_V.INTEGRATED_PROBE_CLAIM_FIELD)
    if sources != {
        label: {"path": pin["path"], "sha256": pin["sha256"]}
        for label, pin in _V.INTEGRATED_PROBE_RUNTIME_SOURCE_PINS.items()
    }:
        raise ReceiptRefused("integrated probe runtime source pins differ")
    duration = updates * ROLLOUT_STEPS_PER_UPDATE * POLICY_DT_S
    if duration <= PUSH_INTERVAL_RANGE_S[0]:
        raise ReceiptRefused(
            "integrated probe duration does not cross push timer lower bound"
        )
    stage["push_timer_control_flow"] = {
        "runtime_sources": sources,
        "push_semantics": "velocity_only",
        "interval_range_s": list(PUSH_INTERVAL_RANGE_S),
        "rollout_steps_per_update": ROLLOUT_STEPS_PER_UPDATE,
        "policy_dt_s": POLICY_DT_S,
        "duration_s": duration,
        "interval_lower_bound_s": PUSH_INTERVAL_RANGE_S[0],
        "duration_crosses_interval_lower_bound": True,
        "runtime_observation_required": True,
        "push_counter": {
            "kind": "runtime_observed_nonzero_v2",
            "event_call_count": push_velocity["aggregate"][
                "event_call_count"
            ],
            "minimum_event_call_count": 1,
            "environment_application_count": push_velocity[
                "aggregate"
            ]["env_application_count"],
            "minimum_environment_application_count": 1,
        },
    }
    return stage, _scientific_identity(payload)


def materialize(
    *,
    gate_checkout: Path,
    gate_source_commit: str,
    evidence_source_commit: str,
    probe_namespace: Path,
    probe_run_dir: Path,
    receipt_repo_path: str,
    long_spec_repo_path: str,
) -> dict[str, Any]:
    producer_pin = _verify_gate_source(gate_checkout, gate_source_commit)
    if _COMMIT_RE.fullmatch(evidence_source_commit or "") is None:
        raise ReceiptRefused("evidence source commit must be 40 lowercase hex")
    if evidence_source_commit != gate_source_commit:
        raise ReceiptRefused(
            "integrated probe evidence must come from the exact gate-code source commit"
        )
    if subprocess.run(
        ["git", "-C", str(gate_checkout), "merge-base", "--is-ancestor", evidence_source_commit, gate_source_commit],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise ReceiptRefused("evidence source commit is not an ancestor of gate source")
    probe, probe_identity = _stage_evidence(
        probe_namespace, probe_run_dir, expected_stage="probe"
    )
    if probe["source_commit"] != evidence_source_commit:
        raise ReceiptRefused(
            "integrated probe source commit differs from exact evidence source"
        )
    receipt_repo_path = _repo_path(receipt_repo_path, name="receipt repo path")
    long_spec_repo_path = _repo_path(long_spec_repo_path, name="long spec repo path")
    if (
        not receipt_repo_path.startswith("configs/n1_vendor_probe_gate_20260731/")
        or not receipt_repo_path.endswith(".json")
    ):
        raise ReceiptRefused("receipt repo path must use the fixed gate config class")
    if (
        not long_spec_repo_path.startswith("configs/n1_vendor_launch_20260731/")
        or ".long." not in Path(long_spec_repo_path).name
        or not long_spec_repo_path.endswith(".json")
    ):
        raise ReceiptRefused("long spec path must use the fixed long-template class")
    acceptance = {
        "integrated_probe_exact_pass": True,
        "finite_checkpoints": True,
        "normalizer_checkpoint_persistence": True,
        "runtime_abi_exact": True,
        "fresh_log_std_initialization_exact": True,
        "control_step_delay_exact": True,
        "positive_policy_std_and_finite_lr": True,
        "reward_ppo_economy_runtime_pass": True,
        "zero_actual_hard_edge": True,
        "zero_qdes_edge": True,
        "zero_nonfinite": True,
        "push_timer_lower_bound_crossed": True,
        "velocity_only_push_observed_nonzero": True,
        "velocity_push_six_axis_extrema_finite_and_in_range": True,
        "natural_training_completion": True,
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "verdict": "PASS",
        "producer": {
            "source": producer_pin,
            "gate_source_commit": gate_source_commit,
            "algorithm": "exact_integrated_probe_v3",
            "self_reference_free": True,
        },
        "evidence_source_commit": evidence_source_commit,
        "scientific_identity": probe_identity,
        "stages": {"probe": probe},
        "acceptance": acceptance,
        "successor_policy": {
            "required_gate_source_ancestor_commit": gate_source_commit,
            "allowed_artifact_descendant_diff": {
                "exact_paths": [receipt_repo_path, long_spec_repo_path],
                "prefixes": ["docs/"],
            },
        },
        "authorization": {
            "vendor_n1_long_launch": True,
            "formal_evidence": False,
            "curriculum_promotion": False,
            "resume": False,
            "export": False,
            "judge": False,
            "deployment": False,
            "hardware": False,
        },
    }
    receipt["content_sha256"] = _canonical_sha(receipt)
    return receipt


def _write_no_clobber(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ReceiptRefused("output must have an existing absolute parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        raise ReceiptRefused(f"receipt output is no-clobber: {exc}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(document) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-checkout", required=True)
    parser.add_argument("--gate-source-commit", required=True)
    parser.add_argument("--evidence-source-commit", required=True)
    parser.add_argument("--probe-namespace", required=True)
    parser.add_argument("--probe-run-dir", required=True)
    parser.add_argument("--receipt-repo-path", required=True)
    parser.add_argument("--long-spec-repo-path", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = materialize(
            gate_checkout=Path(args.gate_checkout),
            gate_source_commit=args.gate_source_commit,
            evidence_source_commit=args.evidence_source_commit,
            probe_namespace=Path(args.probe_namespace),
            probe_run_dir=Path(args.probe_run_dir),
            receipt_repo_path=args.receipt_repo_path,
            long_spec_repo_path=args.long_spec_repo_path,
        )
        _write_no_clobber(Path(args.output), receipt)
        print(
            json.dumps(
                {
                    "verdict": "PASS",
                    "content_sha256": receipt["content_sha256"],
                    "output": args.output,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except ReceiptRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
