#!/usr/bin/env python3
"""Fail-closed, A2/B2-only signed-face L1 launcher and terminal finalizer.

The default is a read-only plan.  A launch creates exactly one new arm claim
on its preregistered empty GPU and returns after the reviewed, host-wide Kit
boot lock observes the hard-contract marker.  A2 is the control on Pod1 GPU0;
B2 is the signed-face guidance arm on Pod2 GPU0.  Each host invocation is
one-shot and each natural exit and terminal ``model_13824.pt`` is finalized
independently; cross-host pair interpretation remains deliberately absent.

This program contains no judge, activation, L2, second-seed, retry, or robot
path.  The wrapper may signal only the exact PGID that it created when Kit
fails before its marker; this launcher never discovers or signals processes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time
from typing import Any


MANIFEST_ID = "phase1-signed-face-a2-b2-hot-explicit-zero-friction-l1-cross-pod-20260714-v2"
CELL_IDS = ("A2", "B2")
TRAINING_COMMIT = "4467d79f1ed425a4263f0caaad2f661e1ec737ad"
TRAINING_TREE = "497db1d8f2d7fb1b554337928f098a2951d4cf0d"
SOURCE_CHECKOUT = "/workspace/codexschema/nohope_signed_face_a2b2_hot_l1_v2_4467d79"
ARTIFACT_ROOT = "/workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714"
RUN_ROOT = f"{ARTIFACT_ROOT}/runs/l1"
CONTROL_ROOT = f"{ARTIFACT_ROOT}/control/v2"
TRAINING_ENV_SHA256_BY_CELL = {
    "A2": "0c89d8b3d34c49e14a1883bf81316367b1089520a65444cb9f03a82d9c007c9e",
    "B2": "82d3249107a2e5a826d1a3d8b3b59dc6c9e60555e415116256865178b7691992",
}
EXPECTED_PYTHONPATH = ":".join((
    f"{SOURCE_CHECKOUT}/hope_training/whole_body_tracking/source/whole_body_tracking",
    "/workspace/IsaacLab/source/isaaclab",
    "/workspace/IsaacLab/source/isaaclab_tasks",
    "/workspace/IsaacLab/source/isaaclab_assets",
    "/workspace/IsaacLab/source/isaaclab_rl",
))
ROOT_CONFIRMATION = (
    "ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_A2_B2_HOT_EXPLICIT_ZERO_FRICTION_"
    "L1_CROSS_POD_V2"
)
ZERO_FRICTION_ARG = "task.plant.zero_joint_friction=true"
CLAIM_ARG_PREFIX = "++training_launch_claim_sha256="
CLAIM_PLACEHOLDER = "<computed-after-atomic-claim>"
CARB_THREAD_ARG = "++kit_carb_tasking_thread_count=16"
TBB_THREAD_ARG = "++kit_tbb_thread_count=16"
THREAD_MARKER = (
    "[train.py] KIT_THREAD_CAP_OK: carb.tasking=16 omni.tbb=16 useOmniJob=false"
)
HARD_CONTRACT_MARKER = "[train.py] hard training contract:"
ZERO_FRICTION_RUNTIME_MARKER = (
    "[train.py] ZERO_FRICTION_RUNTIME_OK: 31/31 instantiated PhysX joint "
    "friction coefficients are exactly 0.0"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
FAILURE_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemory|Segmentation fault|"
    r"\b(?:NaN|Inf)\b|\bKilled\b|malloc|bad_alloc",
    re.IGNORECASE,
)

# Exact static identity of the recipe list.  This avoids copying a second long
# recipe into Python while still making any manifest edit fail closed.
BASE_RECIPE_SHA256 = "40e3c659e0b124b0136dfbd695da83e2d030bc7fac9e6ff9a70d88b66e654a8f"
EXPECTED_HISTORICAL = {
    "predecessor_plan_commit": "c3c60f000b56d9276291c5349eca44a794f242a2",
    "predecessor_status": "plan_static_only_runtime_consumer_required",
    "predecessor_namespace_adopted_for_runtime": False,
    "old_v6_a_b_evidence_adopted": False,
    "old_v6_rejection_reason": (
        "old A/B emitted contracts did not bind the signed-face guidance weight"
    ),
}

EXPECTED_CURRENT_ONLY_SHARED = {
    "motion_adaptive_alpha": 0.001,
    "motion_adaptive_kernel_size": 1,
    "motion_adaptive_lambda": 0.8,
    "motion_adaptive_uniform_ratio": 0.1,
    "motion_clip_switch_prob": 0.0,
    "motion_event_timing": {"mode": "disabled"},
    "motion_post_swing_buffer_size": 4096,
    "motion_post_swing_min_fill": 256,
    "motion_post_swing_min_hold": 25,
    "motion_post_swing_start_prob": 0.25,
    "motion_rsi_skip_settle_frames": 0,
    "motion_stagger_hold_max_steps": 150,
    "motion_stagger_initial_clock": False,
    "racket_midswing_resample_prob": 0.0,
    "racket_midswing_resample_tts_floor": 0.3,
    "racket_strike_phase": 0.47,
    "racket_strike_window_pos_s": None,
    "racket_strike_window_s": 0.12,
    "racket_strike_window_wide_s": None,
    "racket_target_bias_per_swing": 0.0,
    "racket_target_delay_steps": 2,
    "racket_target_dropout_prob": 0.0,
    "racket_target_jitter_pos_per_s": 0.0,
    "racket_target_jitter_vel_per_s": 0.0,
    "racket_target_noise_ar1_rho": 0.717,
    "racket_target_noise_ar1_sigma": 0.0052,
    "racket_target_noise_white": 0.0019,
    "racket_target_post_strike_dropout_s": 0.0,
}

EXPECTED_GUIDANCE_BY_CELL = {
    "A2": {
        "position": {"weight": 0.0, "command_name": "racket_target", "d_max": 0.5},
        "signed_face": {
            "weight": 0.0, "command_name": "racket_target", "theta_max": math.pi,
        },
    },
    "B2": {
        "position": {"weight": 0.0, "command_name": "racket_target", "d_max": 0.5},
        "signed_face": {
            "weight": -0.4, "command_name": "racket_target", "theta_max": math.pi,
        },
    },
}


class ContractError(RuntimeError):
    """One preregistered source, runtime, or scientific invariant changed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if type(value) is not str or not SHA_RE.fullmatch(value):
        raise ContractError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def strictly_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            strictly_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strictly_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def require_exact(actual: Any, expected: Any, label: str) -> None:
    if not strictly_equal(actual, expected):
        raise ContractError(f"{label} changed or has a bool/int type confusion")


def require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ContractError(
            f"{label} keys changed: missing={sorted(keys - actual)} "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except ContractError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} root must be an object")
    return value


def write_json_exclusive(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def cells(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("cells")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ContractError("manifest must contain exactly two cells")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or type(item.get("cell_id")) is not str:
            raise ContractError("each cell must be an object with a string cell_id")
        if item["cell_id"] in result:
            raise ContractError("duplicate cell_id")
        result[item["cell_id"]] = item
    if tuple(result) != CELL_IDS:
        raise ContractError("cell order and identity must be exactly A2 then B2")
    return result


def require_explicit_zero_friction_recipe(
    shared: dict[str, Any], recipe: list[str],
) -> None:
    """Bind the declarative plant fact to one exact Hydra argv leaf."""

    require_exact(shared.get("zero_joint_friction"), True, "zero-friction declaration")
    require_exact(
        shared.get("zero_joint_friction_argv"), ZERO_FRICTION_ARG,
        "zero-friction argv declaration",
    )
    matches = [
        part for part in recipe
        if part.startswith("task.plant.zero_joint_friction=")
        or part.startswith("++task.plant.zero_joint_friction=")
    ]
    if matches != [ZERO_FRICTION_ARG]:
        raise ContractError(
            "zero-friction declaration requires exactly one literal "
            "task.plant.zero_joint_friction=true argv"
        )


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("manifest_id") != MANIFEST_ID:
        raise ContractError("unexpected manifest schema or identity")
    expected_top = {
        "status": "machine_preregistered_runtime_launcher_root_confirmation_required",
        "human_owner": "Franco",
        "executor": "Codex",
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "automatic_retry_forbidden": True,
        "activation_authorized": False,
        "judge_authorized": False,
        "l2_authorized": False,
        "second_seed_authorized": False,
    }
    for key, expected in expected_top.items():
        require_exact(manifest.get(key), expected, f"manifest {key}")

    source = manifest.get("source")
    runtime = manifest.get("runtime")
    inputs = manifest.get("inputs")
    shared = manifest.get("shared_training_contract")
    transition = manifest.get("hot_start_contract_transition")
    pair = manifest.get("pair_contract")
    schedule = manifest.get("execution_schedule")
    evaluation = manifest.get("evaluation")
    if not all(
        isinstance(v, dict)
        for v in (source, runtime, inputs, shared, transition, pair, schedule, evaluation)
    ):
        raise ContractError(
            "source/runtime/inputs/shared/transition/pair/schedule/evaluation must be objects"
        )

    require_exact(source.get("expected_training_commit"), TRAINING_COMMIT, "training commit")
    require_exact(source.get("expected_training_tree"), TRAINING_TREE, "training tree")
    require_exact(source.get("training_checkout"), SOURCE_CHECKOUT, "training checkout")
    require_exact(source.get("wbt_relative_path"), "hope_training/whole_body_tracking", "WBT path")
    require_exact(
        source.get("required_ignored_asset_directory"),
        "source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
        "required ignored A3 asset directory",
    )

    expected_runtime = {
        "expected_host_by_cell": {"A2": "pod1", "B2": "pod2"},
        "artifact_root": ARTIFACT_ROOT,
        "external_control_root": CONTROL_ROOT,
        "run_root": RUN_ROOT,
        "source_asset_root": "/workspace/codexschema/phase1_fresh_20260711",
        "isaac_python": "/workspace/hope_isaac_venv/bin/python",
        "isaaclab_root": "/workspace/IsaacLab",
        "training_environment_sha256_by_cell": TRAINING_ENV_SHA256_BY_CELL,
        "gpu0_uuid_by_host": {
            "pod1": "GPU-889b1712-8d89-0536-5c9e-e79aae30523d",
            "pod2": "GPU-449c8b80-f4a6-2d03-6e8a-b8ac68dea23d",
        },
        "pair_finalize_host": "pod1",
        "pair_input_root": f"{ARTIFACT_ROOT}/pair_inputs/v1",
        "pair_result_path": f"{RUN_ROOT}/paired_l1_result.json",
        "locked_launcher_relative_path": "scripts/launch_kit_training_locked.sh",
        "kit_boot_marker": HARD_CONTRACT_MARKER,
        "zero_friction_runtime_marker": ZERO_FRICTION_RUNTIME_MARKER,
        "kit_boot_timeout_seconds": 900,
        "poll_seconds": 5,
        "initial_gpu_must_have_zero_compute_processes": True,
        "maximum_live_trainers_per_assigned_gpu": 1,
        "minimum_free_gpu_memory_mib_before_launch": 4500,
        "minimum_host_available_memory_mib": 65536,
        "root_launch_confirmation": ROOT_CONFIRMATION,
        "launch_state_basename": "run.log.launch",
        "launch_contract_basename": "launch_contract.json",
        "runtime_verified_basename": "runtime_verified.json",
        "training_log_basename": "run.log",
        "failure_basename": "launch_failure.json",
        "terminal_result_basename": "terminal_result.json",
        "pair_result_basename": "paired_l1_result.json",
    }
    for key, expected in expected_runtime.items():
        require_exact(runtime.get(key), expected, f"runtime {key}")
    require_exact(runtime.get("kit_thread_cap_contract"), {
        "carb_tasking_thread_count": 16,
        "omni_tbb_thread_count": 16,
        "use_omni_job": False,
        "runtime_marker": THREAD_MARKER,
    }, "Kit 16/16 thread contract")
    require_exact(runtime.get("runtime_closure"), {
        "isaaclab": {
            "root": "/workspace/IsaacLab",
            "commit": "21f7136325136ca3f6ca4e0a8125edffe5c24f7e",
            "tree": "9ce79305538001b0451d3745f47076c57c5d6006",
            "must_be_clean": True,
        },
        "python_entrypoint": "/workspace/hope_isaac_venv/bin/python",
    }, "runtime closure")

    expected_inputs = {
        "forehand_motion": ("f2cb2d9f5d27cefbcee0b790000fcd979abaf02894d4fcad061ebca27f141687", False),
        "backhand_motion": ("1722553375cd28f9b2d567c01b1a5fc6bcd149fa12cadb20e5202a9153367534", False),
        "schema3_train_bank": ("3a9d8851c1c0b13ef82f58228ea1cf83213157c70d72daa514f1bed3a3885b71", True),
        "schema3_train_bank_rebind_report": ("9fffed0308eb0102e3575c3a255e9466c04f45e6c0c303cefb5541a19decbb37", True),
        "hot_parent_checkpoint": ("478efa8d163ec53dbade328c5de18947f6c068df78cbadff8e46a29844bdc9e6", True),
    }
    if set(inputs) != set(expected_inputs):
        raise ContractError("input set changed")
    for name, (digest, absolute) in expected_inputs.items():
        require_exact(inputs[name].get("sha256"), digest, f"{name} SHA")
        path_key = "path" if absolute else "relative_path"
        path = Path(str(inputs[name].get(path_key, "")))
        if path.is_absolute() is not absolute:
            raise ContractError(f"{name} path kind changed")
    bank = inputs["schema3_train_bank"]
    require_exact(bank.get("schema_version"), 3, "train bank schema")
    require_exact(bank.get("split"), "train", "train bank split")
    require_exact(bank.get("physics_contract_sha256"), "09dfe8999c54e36b258fe54b5ec3da5d9816ff3be3675963b919371d7f4afb95", "physics contract")
    require_exact(bank.get("source_family_sha256"), "9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db", "bank family")
    require_exact(inputs["hot_parent_checkpoint"], {
        "run_name": "phase1_fresh_v3_SZ_seed3",
        "path": "/workspace/codexschema/nohope/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-11_05-39-31_phase1_fresh_v3_SZ_seed3/model_13800.pt",
        "sha256": "478efa8d163ec53dbade328c5de18947f6c068df78cbadff8e46a29844bdc9e6",
        "embedded_iteration": 13800,
        "embedded_training_contract_schema_version": 3,
        "embedded_training_contract_sha256": "3a3b3d956e19d47f7e6f0a157159dc96c8f09d8345c436a776c8c7e99c0b9972",
        "embedded_training_contract_lineage_exact": True,
        "adjacent_training_contract_path": "/workspace/codexschema/nohope/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-11_05-39-31_phase1_fresh_v3_SZ_seed3/params/training_contract.json",
        "adjacent_training_contract_sha256": "3a3b3d956e19d47f7e6f0a157159dc96c8f09d8345c436a776c8c7e99c0b9972",
    }, "hot parent checkpoint")

    expected_shared = {
        "training_seed": 3,
        "initialization": "hot_representation_transfer",
        "num_envs": 512,
        "max_iterations": 25,
        "save_interval": 100,
        "expected_terminal_checkpoint_iteration": 13824,
        "expected_child_lineage_exact": False,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1.0, -1.0],
        "zero_joint_friction": True,
        "zero_joint_friction_argv": ZERO_FRICTION_ARG,
        "motion_kinematics_exact": True,
        "question_bank_schema_version": 3,
        "question_bank_split": "train",
        "actor_observation_contract": "deploy_parity_face179",
        "actor_observation_dim": 179,
        "action_dim": 31,
        "strike_phase_per_clip": [0.471, 0.338],
        "event_timing_mode": "disabled",
        "positional_guidance_weight": 0.0,
        "face_guidance_theta_max": math.pi,
    }
    for key, expected in expected_shared.items():
        require_exact(shared.get(key), expected, f"shared contract {key}")
    recipe = shared.get("base_recipe")
    if not isinstance(recipe, list) or any(type(part) is not str for part in recipe):
        raise ContractError("base recipe must be a string argv list")
    require_explicit_zero_friction_recipe(shared, recipe)
    if canonical_sha256(recipe) != BASE_RECIPE_SHA256:
        raise ContractError("base recipe changed")
    forbidden_recipe = (
        "racket_guidance_weight", "racket_face_guidance_weight",
        "racket_face_guidance_theta_max", "training_launch_claim_sha256",
        "kit_carb_tasking_thread_count", "kit_tbb_thread_count",
    )
    if any(any(token in part for token in forbidden_recipe) for part in recipe):
        raise ContractError("per-cell guidance/claim/thread facts leaked into the base recipe")

    require_exact(transition, {
        "classification": "explicit_inexact_representation_transfer",
        "checkpoint_allow_missing_contract": False,
        "checkpoint_allow_contract_mismatch": True,
        "checkpoint_tolerant": False,
        "all_parent_and_current_common_fields_must_match_except_exact_allowlist": True,
        "allowed_changed_common_fields": {
            "question_bank": {
                "parent": {
                    "sha256": "2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700",
                    "schema_version": 3, "split": "train",
                    "source_family_sha256": "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5",
                    "exact": True,
                },
                "current": {
                    "sha256": "3a9d8851c1c0b13ef82f58228ea1cf83213157c70d72daa514f1bed3a3885b71",
                    "schema_version": 3, "split": "train",
                    "source_family_sha256": "9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db",
                    "exact": True,
                },
            },
        },
        "allowed_current_only_top_level_keys": [
            "motion_adaptive_alpha", "motion_adaptive_kernel_size", "motion_adaptive_lambda",
            "motion_adaptive_uniform_ratio", "motion_clip_switch_prob", "motion_event_timing",
            "motion_post_swing_buffer_size", "motion_post_swing_min_fill", "motion_post_swing_min_hold",
            "motion_post_swing_start_prob", "motion_rsi_skip_settle_frames", "motion_stagger_hold_max_steps",
            "motion_stagger_initial_clock", "racket_guidance_reward", "racket_midswing_resample_prob",
            "racket_midswing_resample_tts_floor", "racket_strike_phase", "racket_strike_window_pos_s",
            "racket_strike_window_s", "racket_strike_window_wide_s", "racket_target_bias_per_swing",
            "racket_target_delay_steps", "racket_target_dropout_prob", "racket_target_jitter_pos_per_s",
            "racket_target_jitter_vel_per_s", "racket_target_noise_ar1_rho", "racket_target_noise_ar1_sigma",
            "racket_target_noise_white", "racket_target_post_strike_dropout_s",
        ],
        "expected_current_only_shared_values": EXPECTED_CURRENT_ONLY_SHARED,
        "expected_racket_guidance_reward_by_cell": EXPECTED_GUIDANCE_BY_CELL,
        "prohibited_claim": "A2/B2 descendants are never fresh/exact evidence and cannot be relabeled by adjacency to the current contract",
    }, "hot-start contract transition")

    expected_cells = {
        "A2": {
            "cell_id": "A2",
            "human_name": "hot-start explicit-zero-friction signed-face control",
            "causal_role": "hot_control",
            "run_name": "phase1_signed_face_l1_a2b2_v2_A2_hot_control_seed3",
            "host": "pod1", "gpu": 0,
            "face_guidance_weight": 0.0,
            "expected_lineage_exact": False,
        },
        "B2": {
            "cell_id": "B2",
            "human_name": "hot-start explicit-zero-friction signed-face guidance",
            "causal_role": "hot_guidance",
            "run_name": "phase1_signed_face_l1_a2b2_v2_B2_hot_guidance_seed3",
            "host": "pod2", "gpu": 0,
            "face_guidance_weight": -0.4,
            "expected_lineage_exact": False,
        },
    }
    require_exact(cells(manifest), expected_cells, "A2/B2 cells")
    for item in expected_cells.values():
        if not RUN_RE.fullmatch(item["run_name"]):
            raise ContractError("unsafe run name")
    require_exact(pair, {
        "sole_scientific_difference": "signed_face_guidance_weight",
        "normalized_recipe_and_hard_contract_must_match_after_removing_weight": True,
        "same_parent_checkpoint_required": True,
        "same_seed_and_budget_required": True,
        "old_v6_a_b_evidence_adopted": False,
    }, "pair contract")
    require_exact(schedule, {
        "mode": "independent_one_shot_cross_pod_same_gpu_index",
        "ordered_cells": ["A2", "B2"],
        "same_host": False,
        "distinct_assigned_hosts": True,
        "same_physical_gpu_index_on_distinct_hosts": True,
        "kit_boot_serialized_by_each_host_lock": True,
        "one_new_atomic_claim_per_invocation": True,
        "assigned_gpu_must_be_empty_before_each_claim": True,
        "cells_have_no_cross_host_runtime_predecessor": True,
        "exact_cell_and_host_are_required_per_invocation": True,
        "fresh_artifact_arm_and_training_run_absence_required_before_claim": True,
        "stop_on_any_failure": True,
        "automatic_retry_forbidden": True,
    }, "execution schedule")
    require_exact(manifest.get("historical_read_only_evidence"), EXPECTED_HISTORICAL, "historical read-only evidence")
    require_exact(evaluation, {
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "same_immutable_signed_paper_required_before_any_later_decision": True,
        "terminal_result_scope": "finite lineage-inexact hot-start explicit-zero-friction L1 checkpoint provenance only",
    }, "evaluation boundary")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json_object(path, "A2/B2 manifest")
    validate_manifest(manifest)
    return manifest


def repo_for_static_source(manifest: dict[str, Any]) -> Path:
    runtime_checkout = Path(manifest["source"]["training_checkout"])
    return runtime_checkout if runtime_checkout.is_dir() else Path(__file__).resolve().parents[1]


def verify_static_source(manifest: dict[str, Any]) -> dict[str, Any]:
    repo = repo_for_static_source(manifest)
    source = manifest["source"]
    try:
        tree = git_output(repo, "rev-parse", f"{TRAINING_COMMIT}^{{tree}}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"training source commit is unavailable in {repo}") from exc
    if tree != TRAINING_TREE:
        raise ContractError("training source tree changed")
    return {"commit": TRAINING_COMMIT, "tree": tree}


def input_paths(manifest: dict[str, Any]) -> dict[str, tuple[Path, str]]:
    root = Path(manifest["runtime"]["source_asset_root"])
    inputs = manifest["inputs"]
    return {
        "forehand_motion": (root / inputs["forehand_motion"]["relative_path"], inputs["forehand_motion"]["sha256"]),
        "backhand_motion": (root / inputs["backhand_motion"]["relative_path"], inputs["backhand_motion"]["sha256"]),
        "schema3_train_bank": (Path(inputs["schema3_train_bank"]["path"]), inputs["schema3_train_bank"]["sha256"]),
        "schema3_train_bank_rebind_report": (Path(inputs["schema3_train_bank_rebind_report"]["path"]), inputs["schema3_train_bank_rebind_report"]["sha256"]),
        "hot_parent_checkpoint": (Path(inputs["hot_parent_checkpoint"]["path"]), inputs["hot_parent_checkpoint"]["sha256"]),
        "hot_parent_adjacent_contract": (
            Path(inputs["hot_parent_checkpoint"]["adjacent_training_contract_path"]),
            inputs["hot_parent_checkpoint"]["adjacent_training_contract_sha256"],
        ),
    }


def optimization_recipe(manifest: dict[str, Any], cell_id: str) -> dict[str, Any]:
    shared = manifest["shared_training_contract"]
    cell = cells(manifest)[cell_id]
    return {
        "initialization": "hot_representation_transfer",
        "parent_checkpoint_sha256": manifest["inputs"]["hot_parent_checkpoint"]["sha256"],
        "parent_training_contract_sha256": manifest["inputs"]["hot_parent_checkpoint"]["embedded_training_contract_sha256"],
        "training_seed": shared["training_seed"],
        "num_envs": shared["num_envs"],
        "max_iterations": shared["max_iterations"],
        "save_interval": shared["save_interval"],
        "zero_joint_friction": shared["zero_joint_friction"],
        "zero_joint_friction_argv": shared["zero_joint_friction_argv"],
        "positional_guidance_weight": shared["positional_guidance_weight"],
        "signed_face_guidance_weight": cell["face_guidance_weight"],
        "face_guidance_theta_max": shared["face_guidance_theta_max"],
    }


def build_command(manifest: dict[str, Any], cell_id: str, claim_sha: str) -> list[str]:
    if claim_sha != CLAIM_PLACEHOLDER:
        require_sha(claim_sha, "training launch claim")
    shared = manifest["shared_training_contract"]
    runtime = manifest["runtime"]
    cell = cells(manifest)[cell_id]
    paths = input_paths(manifest)
    require_explicit_zero_friction_recipe(shared, shared["base_recipe"])
    command = [
        runtime["isaac_python"], "scripts/train.py", *shared["base_recipe"],
        f"seed={shared['training_seed']}",
        f"num_envs={shared['num_envs']}",
        f"max_iterations={shared['max_iterations']}",
        f"algo.runner.save_interval={shared['save_interval']}",
        f"run_name={cell['run_name']}",
        f"checkpoint_path={paths['hot_parent_checkpoint'][0]}", "checkpoint_tolerant=false",
        "checkpoint_allow_missing_contract=false",
        "checkpoint_allow_contract_mismatch=true",
        f"motion_file={paths['forehand_motion'][0]}",
        f"motion_file_2={paths['backhand_motion'][0]}",
        "task.racket.strike_phase_per_clip=[0.471,0.338]",
        f"++task.racket.question_bank={paths['schema3_train_bank'][0]}",
        "++task.racket.face_command_pairing=shared_plus_y",
        f"++task.rewards.racket_guidance_weight={shared['positional_guidance_weight']}",
        f"++task.rewards.racket_face_guidance_weight={cell['face_guidance_weight']}",
        f"++task.rewards.racket_face_guidance_theta_max={shared['face_guidance_theta_max']}",
        CARB_THREAD_ARG, TBB_THREAD_ARG,
        f"{CLAIM_ARG_PREFIX}{claim_sha}",
    ]
    prefixes = (
        "checkpoint_path=", "checkpoint_tolerant=",
        "checkpoint_allow_missing_contract=",
        "checkpoint_allow_contract_mismatch=",
        "++task.rewards.racket_guidance_weight=",
        "++task.rewards.racket_face_guidance_weight=",
        "++task.rewards.racket_face_guidance_theta_max=",
        "++kit_carb_tasking_thread_count=", "++kit_tbb_thread_count=",
        CLAIM_ARG_PREFIX,
    )
    for prefix in prefixes:
        if sum(part.startswith(prefix) for part in command) != 1:
            raise ContractError(f"constructed command must contain exactly one {prefix}")
    zero_flags = [
        part for part in command
        if part.startswith("task.plant.zero_joint_friction=")
        or part.startswith("++task.plant.zero_joint_friction=")
    ]
    if zero_flags != [ZERO_FRICTION_ARG]:
        raise ContractError("constructed command lost exact zero-friction argv")
    forbidden = ("ros2", "run_deploy", "joint_command", "real_robot", "/dev/")
    if any(any(token in part.lower() for token in forbidden) for part in command):
        raise ContractError("constructed command contains a robot/runtime token")
    return command


def normalized_command(manifest: dict[str, Any], cell_id: str) -> list[str]:
    result = []
    for part in build_command(manifest, cell_id, CLAIM_PLACEHOLDER):
        if part.startswith("run_name="):
            part = "run_name=<paired>"
        elif part.startswith("++task.rewards.racket_face_guidance_weight="):
            part = "++task.rewards.racket_face_guidance_weight=<causal-axis>"
        elif part.startswith(CLAIM_ARG_PREFIX):
            part = f"{CLAIM_ARG_PREFIX}<paired>"
        result.append(part)
    return result


def expected_source_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest["source"]
    return {
        "commit": source["expected_training_commit"],
        "tree": source["expected_training_tree"],
        "checkout": source["training_checkout"],
    }


def build_claim(
    manifest: dict[str, Any], *, manifest_sha: str, launcher_sha: str,
    cell_id: str, arm_dir: Path, arm_identity: dict[str, int],
) -> dict[str, Any]:
    require_sha(manifest_sha, "manifest SHA")
    require_sha(launcher_sha, "launcher SHA")
    if set(arm_identity) != {"device", "inode"} or any(
        type(value) is not int or value <= 0 for value in arm_identity.values()
    ):
        raise ContractError("atomic claim directory identity is invalid")
    cell = cells(manifest)[cell_id]
    expected_dir = Path(manifest["runtime"]["run_root"]) / cell["run_name"]
    if arm_dir != expected_dir:
        raise ContractError("atomic claim path changed")
    return {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "training_source": expected_source_identity(manifest),
        "stage": "l1",
        "cell_id": cell_id,
        "run_name": cell["run_name"],
        "parent_checkpoint": manifest["inputs"]["hot_parent_checkpoint"],
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "execution_lane": {
            "host": cell["host"],
            "physical_gpu": cell["gpu"],
            "cuda_visible_devices": str(cell["gpu"]),
            "local_training_device": "cuda:0",
            "training_environment_sha256": TRAINING_ENV_SHA256_BY_CELL[cell_id],
        },
        "expected_terminal_checkpoint_iteration": 13824,
        "expected_child_training_contract_lineage_exact": False,
        "claim_directory": {
            "path": str(arm_dir),
            "st_dev": arm_identity["device"],
            "st_ino": arm_identity["inode"],
        },
    }


def build_plan(manifest: dict[str, Any], manifest_path: Path, launcher_path: Path) -> dict[str, Any]:
    if normalized_command(manifest, "A2") != normalized_command(manifest, "B2"):
        raise ContractError("A2/B2 commands differ outside run name, claim, and face weight")
    return {
        "artifact_kind": "phase1_signed_face_a2_b2_hot_l1_cross_pod_plan",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "launcher_sha256": sha256_file(launcher_path),
        "training_source": verify_static_source(manifest),
        "simulation_only": True,
        "writes_or_launches_performed": False,
        "ordered_cells": list(CELL_IDS),
        "one_new_atomic_claim_per_invocation": True,
        "execution_lanes": {
            cell_id: {
                "host": cells(manifest)[cell_id]["host"],
                "physical_gpu": cells(manifest)[cell_id]["gpu"],
                "local_training_device": "cuda:0",
            }
            for cell_id in CELL_IDS
        },
        "cross_host_independent_one_shot": True,
        "same_parent_checkpoint_sha256": manifest["inputs"]["hot_parent_checkpoint"]["sha256"],
        "expected_child_lineage_exact": False,
        "explicit_zero_friction_argv_by_cell": {
            cell_id: ZERO_FRICTION_ARG for cell_id in CELL_IDS
        },
        "commands": {
            cell_id: shlex.join(build_command(manifest, cell_id, CLAIM_PLACEHOLDER))
            for cell_id in CELL_IDS
        },
        "decision_boundary": {
            "activation": False, "judge": False, "l2": False,
            "second_seed": False, "stop_or_promote": False,
            "automatic_retry": False, "real_robot_commands": False,
        },
    }


def no_symlink_existing_components(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ContractError(f"{label} contains a symlink component: {current}")


def require_regular(path: Path, label: str) -> os.stat_result:
    no_symlink_existing_components(path, label)
    try:
        result = path.stat()
    except OSError as exc:
        raise ContractError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
        raise ContractError(f"{label} must be a single-link regular file: {path}")
    return result


def require_directory(path: Path, label: str) -> os.stat_result:
    no_symlink_existing_components(path, label)
    try:
        result = path.stat()
    except OSError as exc:
        raise ContractError(f"{label} is missing: {path}") from exc
    if not stat.S_ISDIR(result.st_mode):
        raise ContractError(f"{label} must be a directory: {path}")
    return result


def verify_external_control_location(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path
) -> dict[str, Any]:
    """Bind runtime/finalizer invocations to one immutable external control snapshot."""

    root = Path(manifest["runtime"]["external_control_root"])
    require_directory(root, "external control root")
    expected_manifest = root / "phase1_signed_face_a2b2_l1_prereg_20260714.json"
    expected_launcher = root / "run_phase1_signed_face_a2b2_l1.py"
    if manifest_path != expected_manifest or launcher_path != expected_launcher:
        raise ContractError("runtime modes require the exact external control paths")
    receipts = {}
    for label, path in (("manifest", manifest_path), ("launcher", launcher_path)):
        result = require_regular(path, f"external control {label}")
        if result.st_mode & 0o222:
            raise ContractError(f"external control {label} must be read-only")
        receipts[label] = {
            "path": str(path), "sha256": sha256_file(path),
            "device": result.st_dev, "inode": result.st_ino,
        }
    return receipts


def identity(result: os.stat_result) -> dict[str, int]:
    return {"device": result.st_dev, "inode": result.st_ino}


def asset_tree(root: Path) -> dict[str, Any]:
    require_directory(root, "A3 runtime asset root")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"A3 runtime asset contains symlink: {path}")
        if path.is_dir():
            continue
        result = require_regular(path, "A3 runtime asset file")
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": result.st_size,
            "sha256": sha256_file(path),
        })
    return {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "tree_content_sha256": canonical_sha256({"files": rows}),
    }


def exact_environment_payload(
    manifest: dict[str, Any], wbt: Path, cell_id: str
) -> dict[str, str]:
    runtime = manifest["runtime"]
    gpu = cells(manifest)[cell_id]["gpu"]
    isaac = Path(runtime["isaaclab_root"])
    entries = [
        wbt / "source/whole_body_tracking",
        isaac / "source/isaaclab", isaac / "source/isaaclab_tasks",
        isaac / "source/isaaclab_assets", isaac / "source/isaaclab_rl",
    ]
    pythonpath = ":".join(str(path) for path in entries)
    if pythonpath != EXPECTED_PYTHONPATH:
        raise ContractError("source-first PYTHONPATH entries or order changed")
    return {
        "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "LOGNAME": "root", "USER": "root", "SHELL": "/bin/bash",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
        "HOPE_ISAAC_PYTHON": runtime["isaac_python"],
        "HOPE_ISAACLAB_ROOT": str(isaac),
        "HOPE_WBT_PYTHONPATH": pythonpath, "PYTHONPATH": pythonpath,
        "OMNI_KIT_ACCEPT_EULA": "YES", "TMPDIR": "/workspace/tmp",
        "PIP_CACHE_DIR": "/workspace/.cache/pip", "XDG_CACHE_HOME": "/workspace/.cache",
        "WANDB_DIR": "/workspace/codexschema/.wandb",
        "WANDB_ENTITY": "BerkeleyPingPong",
        "WANDB_REGISTRY_ORG": "dongc_1-university-of-california-berkeley-org",
        "WANDB_PROJECT": "hope_wbc", "WANDB_MOTION_PROJECT": "csv_to_npz",
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
        "HOPE_PREREG_HOST": cells(manifest)[cell_id]["host"],
        "HOPE_PREREG_CELL": cell_id,
    }


def build_environment(
    manifest: dict[str, Any], wbt: Path, cell_id: str
) -> dict[str, str]:
    if (wbt / "setup_train_env.local.sh").exists():
        raise ContractError("untracked setup_train_env.local.sh is forbidden")
    isaac = Path(manifest["runtime"]["isaaclab_root"])
    entries = [
        wbt / "source/whole_body_tracking",
        isaac / "source/isaaclab", isaac / "source/isaaclab_tasks",
        isaac / "source/isaaclab_assets", isaac / "source/isaaclab_rl",
    ]
    for entry in entries:
        require_directory(entry, "PYTHONPATH entry")
    environment = exact_environment_payload(manifest, wbt, cell_id)
    for key in ("TMPDIR", "XDG_CACHE_HOME", "WANDB_DIR"):
        require_directory(Path(environment[key]), f"environment {key}")
    if canonical_sha256(environment) != TRAINING_ENV_SHA256_BY_CELL[cell_id]:
        raise ContractError("explicit source-first training environment changed")
    return environment


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    uuid_rows = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(uuid_rows) != 1 or not uuid_rows[0].startswith("GPU-"):
        raise ContractError("cannot read exact GPU UUID")
    free = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(free) != 1 or not free[0].strip().isdigit():
        raise ContractError("cannot read exact GPU free memory")
    rows = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    pids = sorted({int(row.strip()) for row in rows if row.strip().isdigit()})
    trainers = []
    for pid in pids:
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.is_file() and b"scripts/train.py" in cmdline.read_bytes():
            trainers.append(pid)
    return {
        "gpu": gpu,
        "uuid": uuid_rows[0].strip(),
        "free_memory_mib": int(free[0]),
        "compute_pids": pids,
        "trainer_pids": trainers,
    }


def verify_gpu_host_identity(
    manifest: dict[str, Any], host: str, snapshot: dict[str, Any]
) -> None:
    require_exact(snapshot.get("gpu"), 0, "host identity GPU index")
    require_exact(
        snapshot.get("uuid"),
        manifest["runtime"]["gpu0_uuid_by_host"][host],
        f"{host} hardware GPU0 UUID",
    )


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ContractError("/proc/meminfo lacks MemAvailable")


def verify_runtime_closure(manifest: dict[str, Any], python: Path) -> dict[str, Any]:
    expected = manifest["runtime"]["runtime_closure"]
    isaac = Path(expected["isaaclab"]["root"])
    if git_output(isaac, "rev-parse", "HEAD") != expected["isaaclab"]["commit"]:
        raise ContractError("IsaacLab commit changed")
    if git_output(isaac, "rev-parse", "HEAD^{tree}") != expected["isaaclab"]["tree"]:
        raise ContractError("IsaacLab tree changed")
    if git_output(isaac, "status", "--porcelain", "--untracked-files=all"):
        raise ContractError("IsaacLab checkout is dirty")
    require_exact(str(python), expected["python_entrypoint"], "Python entrypoint")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ContractError("Python entrypoint is unavailable")
    return expected


def verify_runtime(
    manifest: dict[str, Any], cell_id: str, observed_host: str,
    *, require_empty_gpu: bool
) -> dict[str, Any]:
    require_exact(observed_host, cells(manifest)[cell_id]["host"], "execution host")
    require_exact(
        manifest["runtime"]["expected_host_by_cell"][cell_id],
        observed_host,
        "manifest execution host",
    )
    source = manifest["source"]
    runtime = manifest["runtime"]
    checkout = Path(source["training_checkout"])
    require_directory(checkout, "training checkout")
    if git_output(checkout, "rev-parse", "HEAD") != TRAINING_COMMIT:
        raise ContractError("training checkout commit changed")
    if git_output(checkout, "rev-parse", "HEAD^{tree}") != TRAINING_TREE:
        raise ContractError("training checkout tree changed")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ContractError("training checkout must be exact and clean")
    wbt = checkout / source["wbt_relative_path"]
    require_directory(wbt, "whole-body training root")
    target = wbt / source["required_ignored_asset_directory"]
    require_directory(target, "required ignored A3 asset directory")

    verified_inputs = {}
    for name, (path, expected) in input_paths(manifest).items():
        require_regular(path, name)
        if sha256_file(path) != expected:
            raise ContractError(f"{name} SHA changed")
        verified_inputs[name] = {"path": str(path), "sha256": expected}
    bank_metadata = verify_schema3_bank_metadata(
        Path(manifest["inputs"]["schema3_train_bank"]["path"]),
        manifest["inputs"]["schema3_train_bank"],
    )
    python = Path(runtime["isaac_python"])
    locked = wbt / runtime["locked_launcher_relative_path"]
    if not python.is_file():
        raise ContractError("Isaac Python entrypoint is missing")
    require_regular(locked, "locked Kit launcher")
    if not os.access(python, os.X_OK) or not os.access(locked, os.X_OK):
        raise ContractError("Python or locked launcher is not executable")
    environment = build_environment(manifest, wbt, cell_id)
    closure = verify_runtime_closure(manifest, python)
    parent = manifest["inputs"]["hot_parent_checkpoint"]
    parent_checkpoint = Path(parent["path"])
    if parent_checkpoint.name != "model_13800.pt":
        raise ContractError("hot parent filename no longer binds iteration 13800")
    parent_audit = checkpoint_audit(
        python, parent_checkpoint, require_launch_claim=False
    )
    parent_expected = {
        "iter": 13800,
        "training_contract_schema_version": 3,
        "training_contract_sha256": parent["embedded_training_contract_sha256"],
        "training_contract_lineage_exact": 1,
        "nonfinite_floating_elements": 0,
    }
    for key, expected in parent_expected.items():
        require_exact(parent_audit.get(key), expected, f"hot parent checkpoint {key}")
    if available_memory_mib() < runtime["minimum_host_available_memory_mib"]:
        raise ContractError("host available memory is below the frozen floor")
    assigned_gpu = cells(manifest)[cell_id]["gpu"]
    gpu = gpu_snapshot(assigned_gpu)
    verify_gpu_host_identity(manifest, observed_host, gpu)
    if require_empty_gpu and gpu["compute_pids"]:
        raise ContractError(
            f"{cell_id} assigned GPU{assigned_gpu} must be empty, found {gpu['compute_pids']}"
        )
    if gpu["free_memory_mib"] < runtime["minimum_free_gpu_memory_mib_before_launch"]:
        raise ContractError("GPU free memory is below the frozen floor")
    return {
        "checkout": checkout, "wbt": wbt, "python": python, "locked": locked,
        "environment": environment,
        "runtime_closure": closure, "verified_inputs": verified_inputs,
        "schema3_train_bank_metadata": bank_metadata,
        "gpu_snapshot": gpu, "assigned_gpu": assigned_gpu,
        "training_environment_sha256": TRAINING_ENV_SHA256_BY_CELL[cell_id],
        "observed_host_confirmation": observed_host,
        "hot_parent_checkpoint_audit": parent_audit,
        "required_ignored_asset_directory": str(target),
    }


def parse_state(path: Path) -> dict[str, str]:
    require_regular(path, "launch state")
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def process_starttime(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return -1
    close = raw.rfind(")")
    if close < 0:
        raise ContractError("cannot parse process stat")
    fields = raw[close + 2:].split()
    if len(fields) <= 19 or not fields[19].isdigit():
        raise ContractError("process stat lacks starttime")
    return int(fields[19])


def locate_training_run(wbt: Path, run_name: str, timeout: int = 90) -> Path:
    root = wbt / "logs/rsl_rl/agibot_a3_hope_virtualball"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hits = sorted(
            path for path in root.glob(f"*_{run_name}")
            if path.is_dir() and (path / "params/training_contract.json").is_file()
        ) if root.is_dir() else []
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ContractError(f"run name maps to multiple training directories: {run_name}")
        time.sleep(2)
    raise ContractError(f"training run/contract did not materialize: {run_name}")


def verify_schema3_bank_metadata(
    path: Path, expected: dict[str, Any],
) -> dict[str, Any]:
    """Bind physics from exact NPZ metadata, outside the five-key compact record."""

    require_regular(path, "schema-3 train bank")
    if sha256_file(path) != expected["sha256"]:
        raise ContractError("schema-3 train-bank file SHA changed")
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            if "meta_json" not in archive.files:
                raise ContractError("schema-3 train bank has no meta_json")
            raw = bytes(np.asarray(archive["meta_json"], dtype=np.uint8).tolist())
        metadata = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"cannot parse schema-3 train-bank metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ContractError("schema-3 train-bank metadata must be an object")
    for key in (
        "schema_version", "split", "source_family_sha256",
        "physics_contract_sha256",
    ):
        require_exact(metadata.get(key), expected[key], f"bank metadata {key}")
    family = metadata.get("source_family_contract")
    if not isinstance(family, dict):
        raise ContractError("schema-3 train bank lacks source_family_contract")
    require_exact(
        family.get("physics_contract_sha256"), expected["physics_contract_sha256"],
        "source-family physics-contract SHA",
    )
    require_exact(
        canonical_sha256(family), expected["source_family_sha256"],
        "source-family contract canonical SHA",
    )
    return {
        "path": str(path),
        "sha256": expected["sha256"],
        "schema_version": expected["schema_version"],
        "split": expected["split"],
        "source_family_sha256": expected["source_family_sha256"],
        "physics_contract_sha256": expected["physics_contract_sha256"],
        "source_family_contract_sha256_recomputed": canonical_sha256(family),
    }


def verify_hard_contract(
    path: Path, manifest: dict[str, Any], cell_id: str,
    *, bank_metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    require_regular(path, "adjacent hard contract")
    contract = read_json_object(path, "adjacent hard contract")
    shared = manifest["shared_training_contract"]
    recipe = shared.get("base_recipe")
    if not isinstance(recipe, list) or any(type(part) is not str for part in recipe):
        raise ContractError("hard-contract verification requires the exact base recipe")
    require_explicit_zero_friction_recipe(shared, recipe)
    expected = {
        "schema_version": 3,
        "actor_obs_contract": "deploy_parity_face179",
        "actor_obs_total_dim": 179,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1.0, -1.0],
        "strike_phase_per_clip": [0.471, 0.338],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
        "motion_event_timing": {"mode": "disabled"},
        "racket_guidance_reward": {
            "position": {"weight": 0.0, "command_name": "racket_target", "d_max": 0.5},
            "signed_face": {
                "weight": cells(manifest)[cell_id]["face_guidance_weight"],
                "command_name": "racket_target",
                "theta_max": math.pi,
            },
        },
    }
    for key, wanted in expected.items():
        require_exact(contract.get(key), wanted, f"hard contract {key}")
    if len(contract.get("joint_names", [])) != 31 or len(contract.get("action_joint_ids", [])) != 31:
        raise ContractError("hard contract does not bind 31 joints/actions")
    friction = contract.get("joint_friction_coefficients")
    if (
        not isinstance(friction, list) or len(friction) != 31
        or any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) != 0.0
            for value in friction
        )
    ):
        raise ContractError("hard contract is not 31/31 zero-friction")
    clips = contract.get("motion_clips")
    if not isinstance(clips, list) or [item.get("sha256") for item in clips if isinstance(item, dict)] != [
        manifest["inputs"]["forehand_motion"]["sha256"],
        manifest["inputs"]["backhand_motion"]["sha256"],
    ]:
        raise ContractError("hard contract motion order/SHA changed")
    bank = contract.get("question_bank")
    expected_bank = manifest["inputs"]["schema3_train_bank"]
    exact_keys = {"sha256", "schema_version", "split", "source_family_sha256", "exact"}
    if not isinstance(bank, dict) or set(bank) != exact_keys:
        raise ContractError(
            "hard contract question_bank must be the exact trainer-emitted five-key shape"
        )
    require_exact(bank, {
        "sha256": expected_bank["sha256"],
        "schema_version": 3,
        "split": "train",
        "source_family_sha256": expected_bank["source_family_sha256"],
        "exact": True,
    }, "compact hard-contract train bank")
    if bank_metadata is None:
        bank_metadata = verify_schema3_bank_metadata(
            Path(expected_bank["path"]), expected_bank
        )
    require_exact(bank_metadata.get("sha256"), bank["sha256"], "bank file↔hard-contract SHA")
    require_exact(
        bank_metadata.get("source_family_sha256"), bank["source_family_sha256"],
        "bank family↔hard-contract SHA",
    )
    require_exact(
        bank_metadata.get("physics_contract_sha256"),
        expected_bank["physics_contract_sha256"],
        "independent bank physics contract",
    )
    return sha256_file(path), contract


def verify_hot_transition(
    parent: dict[str, Any], current: dict[str, Any],
    manifest: dict[str, Any], cell_id: str,
) -> None:
    transition = manifest["hot_start_contract_transition"]
    bank_transition = transition["allowed_changed_common_fields"]["question_bank"]
    require_exact(parent.get("question_bank"), bank_transition["parent"], "parent question bank")
    require_exact(current.get("question_bank"), bank_transition["current"], "current question bank")
    parent_keys = set(parent)
    current_keys = set(current)
    if parent_keys - current_keys:
        raise ContractError("current hard contract dropped a parent field")
    allowed = set(transition["allowed_current_only_top_level_keys"])
    if current_keys - parent_keys != allowed:
        raise ContractError("current-only hard-contract field set changed")
    expected_values = dict(transition["expected_current_only_shared_values"])
    expected_values["racket_guidance_reward"] = transition[
        "expected_racket_guidance_reward_by_cell"
    ][cell_id]
    require_exact(
        {key: current[key] for key in allowed},
        expected_values,
        f"{cell_id} preregistered current-only hard-contract values",
    )
    for key in parent_keys - {"question_bank"}:
        require_exact(current[key], parent[key], f"hot transition common field {key}")


def verify_parent_to_current_contract(
    current: dict[str, Any], manifest: dict[str, Any], cell_id: str,
) -> dict[str, Any]:
    parent_spec = manifest["inputs"]["hot_parent_checkpoint"]
    parent_path = Path(parent_spec["adjacent_training_contract_path"])
    require_regular(parent_path, "hot parent adjacent hard contract")
    if sha256_file(parent_path) != parent_spec["adjacent_training_contract_sha256"]:
        raise ContractError("hot parent adjacent hard-contract SHA changed")
    parent = read_json_object(parent_path, "hot parent adjacent hard contract")
    verify_hot_transition(parent, current, manifest, cell_id)
    return parent


def verify_pair_contracts(contracts: dict[str, dict[str, Any]]) -> None:
    if set(contracts) != set(CELL_IDS):
        raise ContractError("paired hard-contract audit requires A2 and B2")
    normalized = {}
    weights = {}
    for cell_id in CELL_IDS:
        value = copy.deepcopy(contracts[cell_id])
        try:
            signed = value["racket_guidance_reward"]["signed_face"]
            weights[cell_id] = signed["weight"]
            signed["weight"] = "<causal-axis>"
        except (KeyError, TypeError) as exc:
            raise ContractError("hard contract lacks signed-face causal fact") from exc
        normalized[cell_id] = value
    require_exact(weights, {"A2": 0.0, "B2": -0.4}, "paired hard-contract weights")
    if normalized["A2"] != normalized["B2"]:
        raise ContractError("A2/B2 hard contracts differ outside signed-face guidance weight")


CHECKPOINT_AUDIT_KEYS = {
    "iter", "training_contract_schema_version", "training_contract_sha256",
    "training_contract_lineage_exact", "training_launch_claim_sha256",
    "floating_tensor_count", "floating_elements", "nonfinite_floating_elements",
}

TERMINAL_RESULT_KEYS = {
    "artifact_kind", "schema_version", "manifest_id", "manifest_sha256",
    "launcher_sha256", "cell_id", "run_name", "optimization_recipe",
    "execution_lane", "training_source", "training_launch_claim",
    "training_launch_claim_sha256", "training_run_dir", "hard_contract_path",
    "hard_contract_sha256", "terminal_checkpoint_path",
    "terminal_checkpoint_sha256", "terminal_checkpoint_file_identity",
    "terminal_checkpoint_size_bytes", "checkpoint_audit", "training_log_sha256",
    "training_log_file_identity", "exact_trainer_natural_exit_observed",
    "gpu_empty_terminal_barrier_observed", "kit_thread_cap_marker_occurrences",
    "instantiated_zero_friction_marker_occurrences", "activation", "judge", "l2",
    "second_seed", "stop_or_promote", "real_robot_commands_executed",
}


CHECKPOINT_CODE = r"""
import json,sys,torch
obj=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
stack=[obj]; seen=set(); tensors=elements=nonfinite=0
while stack:
 value=stack.pop()
 if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
  tensors+=1; elements+=value.numel(); nonfinite+=int((~torch.isfinite(value)).sum().item())
 elif isinstance(value,dict) and id(value) not in seen:
  seen.add(id(value)); stack.extend(value.values())
 elif isinstance(value,(list,tuple)) and id(value) not in seen:
  seen.add(id(value)); stack.extend(value)
infos=obj.get('infos') if isinstance(obj,dict) else {}; infos=infos if isinstance(infos,dict) else {}
print(json.dumps({'iter':obj.get('iter'),'training_contract_schema_version':infos.get('training_contract_schema_version'),'training_contract_sha256':infos.get('training_contract_sha256'),'training_contract_lineage_exact':infos.get('training_contract_lineage_exact'),'training_launch_claim_sha256':infos.get('training_launch_claim_sha256'),'floating_tensor_count':tensors,'floating_elements':elements,'nonfinite_floating_elements':nonfinite},sort_keys=True))
"""


def checkpoint_audit(
    python: Path, checkpoint: Path, *, require_launch_claim: bool = True
) -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [str(python), "-c", CHECKPOINT_CODE, str(checkpoint)],
            text=True, stderr=subprocess.STDOUT,
        )
        result = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ContractError(f"checkpoint audit failed: {exc}") from exc
    require_exact_keys(result, CHECKPOINT_AUDIT_KEYS, "checkpoint audit")
    for key in ("iter", "training_contract_schema_version", "training_contract_lineage_exact", "floating_tensor_count", "floating_elements", "nonfinite_floating_elements"):
        if type(result[key]) is not int:
            raise ContractError(f"checkpoint audit {key} has wrong exact type")
    require_sha(result["training_contract_sha256"], "checkpoint hard contract SHA")
    if require_launch_claim:
        require_sha(result["training_launch_claim_sha256"], "checkpoint launch claim SHA")
    elif result["training_launch_claim_sha256"] is not None:
        require_sha(result["training_launch_claim_sha256"], "optional parent launch claim SHA")
    if result["floating_tensor_count"] <= 0 or result["floating_elements"] <= 0:
        raise ContractError("checkpoint contains no floating tensors")
    if result["nonfinite_floating_elements"] != 0:
        raise ContractError("checkpoint contains NaN/Inf")
    return result


def expected_arm_paths(manifest: dict[str, Any], cell_id: str) -> dict[str, Path]:
    runtime = manifest["runtime"]
    arm = Path(runtime["run_root"]) / cells(manifest)[cell_id]["run_name"]
    return {
        "arm": arm,
        "launch": arm / runtime["launch_contract_basename"],
        "runtime": arm / runtime["runtime_verified_basename"],
        "state": arm / runtime["launch_state_basename"],
        "log": arm / runtime["training_log_basename"],
        "failure": arm / runtime["failure_basename"],
        "result": arm / runtime["terminal_result_basename"],
    }


def reconstruct_claim(
    manifest: dict[str, Any], manifest_sha: str, launcher_sha: str,
    cell_id: str, launch: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    claim = launch.get("training_launch_claim")
    digest = require_sha(launch.get("training_launch_claim_sha256"), "recorded launch claim")
    if not isinstance(claim, dict) or canonical_sha256(claim) != digest:
        raise ContractError("recorded launch claim bytes/digest changed")
    directory = claim.get("claim_directory")
    if not isinstance(directory, dict):
        raise ContractError("recorded launch claim lacks directory identity")
    expected = build_claim(
        manifest, manifest_sha=manifest_sha, launcher_sha=launcher_sha,
        cell_id=cell_id, arm_dir=expected_arm_paths(manifest, cell_id)["arm"],
        arm_identity={"device": directory.get("st_dev"), "inode": directory.get("st_ino")},
    )
    require_exact(claim, expected, "reconstructed atomic launch claim")
    if canonical_sha256(expected) != digest:
        raise ContractError("launch claim does not bind reconstructed source/recipe/directory")
    return expected, digest


def verify_launch_and_runtime(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    cell_id: str,
) -> dict[str, Any]:
    paths = expected_arm_paths(manifest, cell_id)
    if paths["failure"].exists():
        raise ContractError(f"{cell_id} has a preserved failure marker; automatic retry is forbidden")
    launch = read_json_object(paths["launch"], f"{cell_id} launch contract")
    runtime = read_json_object(paths["runtime"], f"{cell_id} runtime verification")
    manifest_sha = sha256_file(manifest_path)
    launcher_sha = sha256_file(launcher_path)
    control = verify_external_control_location(manifest, manifest_path, launcher_path)
    claim, claim_sha = reconstruct_claim(manifest, manifest_sha, launcher_sha, cell_id, launch)
    expected_command = build_command(manifest, cell_id, claim_sha)
    common = {
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "training_source": expected_source_identity(manifest),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "command": expected_command,
        "external_control": control,
        "execution_lane": claim["execution_lane"],
        "training_environment_sha256": TRAINING_ENV_SHA256_BY_CELL[cell_id],
    }
    for key, expected in common.items():
        require_exact(launch.get(key), expected, f"{cell_id} launch {key}")
        require_exact(runtime.get(key), expected, f"{cell_id} runtime {key}")
    require_exact(launch.get("explicit_pythonpath"), EXPECTED_PYTHONPATH, f"{cell_id} launch PYTHONPATH")
    require_exact(runtime.get("explicit_pythonpath"), EXPECTED_PYTHONPATH, f"{cell_id} runtime PYTHONPATH")
    snapshot = launch.get("gpu_snapshot_before")
    if not isinstance(snapshot, dict):
        raise ContractError(f"{cell_id} launch lacks assigned-GPU snapshot")
    require_exact(snapshot.get("gpu"), cells(manifest)[cell_id]["gpu"], f"{cell_id} assigned GPU")
    require_exact(snapshot.get("compute_pids"), [], f"{cell_id} pre-claim GPU compute PIDs")
    arm_stat = require_directory(paths["arm"], f"{cell_id} atomic claim directory")
    require_exact(identity(arm_stat), {
        "device": claim["claim_directory"]["st_dev"],
        "inode": claim["claim_directory"]["st_ino"],
    }, f"{cell_id} atomic claim directory identity")
    state = parse_state(paths["state"])
    if not state.get("pid", "").isdigit() or state.get("pgid") != state.get("pid"):
        raise ContractError(f"{cell_id} launch state lost pid==pgid")
    require_exact(runtime.get("pid"), int(state["pid"]), f"{cell_id} runtime pid")
    require_exact(runtime.get("pgid"), int(state["pgid"]), f"{cell_id} runtime pgid")
    if type(runtime.get("process_starttime_ticks")) is not int or runtime["process_starttime_ticks"] <= 0:
        raise ContractError(f"{cell_id} runtime starttime is invalid")
    run_dir = Path(str(runtime.get("training_run_dir", "")))
    expected_logs = Path(manifest["source"]["training_checkout"]) / manifest["source"]["wbt_relative_path"] / "logs/rsl_rl/agibot_a3_hope_virtualball"
    require_directory(run_dir, f"{cell_id} training run directory")
    if run_dir.parent != expected_logs or not run_dir.name.endswith(f"_{cells(manifest)[cell_id]['run_name']}"):
        raise ContractError(f"{cell_id} runtime evidence binds a foreign training run")
    require_exact(identity(run_dir.stat()), runtime.get("training_run_directory_identity"), f"{cell_id} training run identity")
    contract_path = Path(str(runtime.get("hard_contract_path", "")))
    if contract_path != run_dir / "params/training_contract.json":
        raise ContractError(f"{cell_id} adjacent contract path changed")
    contract_stat = require_regular(contract_path, f"{cell_id} adjacent hard contract")
    require_exact(identity(contract_stat), runtime.get("hard_contract_file_identity"), f"{cell_id} hard contract identity")
    hard_sha, hard_contract = verify_hard_contract(contract_path, manifest, cell_id)
    verify_parent_to_current_contract(hard_contract, manifest, cell_id)
    require_exact(runtime.get("hard_contract_sha256"), hard_sha, f"{cell_id} hard contract SHA")
    require_exact(runtime.get("checkpoint_absent_at_runtime_verification"), True, f"{cell_id} runtime checkpoint boundary")
    require_exact(
        runtime.get("instantiated_zero_friction_marker_verified"), True,
        f"{cell_id} runtime zero-friction marker boundary",
    )
    require_exact(runtime.get("activation_started"), False, f"{cell_id} activation boundary")
    require_exact(runtime.get("judge_started"), False, f"{cell_id} judge boundary")
    return {
        "paths": paths, "launch": launch, "runtime": runtime,
        "claim": claim, "claim_sha": claim_sha, "command": expected_command,
        "run_dir": run_dir, "contract_path": contract_path,
        "hard_sha": hard_sha, "hard_contract": hard_contract,
    }


def audit_terminal(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    preflight: dict[str, Any], cell_id: str, *, stable_delay: float = 1.0,
) -> dict[str, Any]:
    evidence = verify_launch_and_runtime(manifest, manifest_path, launcher_path, cell_id)
    runtime = evidence["runtime"]
    pid = runtime["pid"]
    if process_starttime(pid) == runtime["process_starttime_ticks"]:
        raise ContractError(f"{cell_id} exact trainer is still alive; finalize is read-only")
    assigned_gpu = cells(manifest)[cell_id]["gpu"]
    gpu = gpu_snapshot(assigned_gpu)
    if gpu["compute_pids"]:
        raise ContractError(
            f"{cell_id} terminal barrier requires assigned GPU{assigned_gpu} empty, "
            f"found {gpu['compute_pids']}"
        )
    log_path = evidence["paths"]["log"]
    log_stat = require_regular(log_path, f"{cell_id} training log")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if FAILURE_RE.search(text):
        raise ContractError(f"{cell_id} log contains a hard failure signature")
    if text.count(THREAD_MARKER) != 1:
        raise ContractError(f"{cell_id} log lacks exactly one 16/16 thread marker")
    if text.count(ZERO_FRICTION_RUNTIME_MARKER) != 1:
        raise ContractError(
            f"{cell_id} log lacks exactly one instantiated zero-friction marker"
        )
    if HARD_CONTRACT_MARKER not in text:
        raise ContractError(f"{cell_id} log lacks hard-contract marker")
    checkpoint = evidence["run_dir"] / "model_13824.pt"
    before = require_regular(checkpoint, f"{cell_id} terminal model_13824.pt")
    if before.st_size <= 0:
        raise ContractError(f"{cell_id} terminal checkpoint is empty")
    runtime_stat = require_regular(evidence["paths"]["runtime"], f"{cell_id} runtime verification")
    if before.st_ctime_ns <= runtime_stat.st_ctime_ns:
        raise ContractError(f"{cell_id} model_13824.pt does not postdate runtime verification")
    if stable_delay:
        time.sleep(stable_delay)
    after = require_regular(checkpoint, f"{cell_id} stable terminal model_13824.pt")
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ContractError(f"{cell_id} model_13824.pt is still changing")
    checkpoint_sha = sha256_file(checkpoint)
    audit = checkpoint_audit(preflight["python"], checkpoint)
    expected_audit = {
        "iter": 13824,
        "training_contract_schema_version": 3,
        "training_contract_sha256": evidence["hard_sha"],
        "training_contract_lineage_exact": 0,
        "training_launch_claim_sha256": evidence["claim_sha"],
        "nonfinite_floating_elements": 0,
    }
    for key, expected in expected_audit.items():
        require_exact(audit.get(key), expected, f"{cell_id} checkpoint {key}")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise ContractError(f"{cell_id} model_13824.pt changed during audit")
    return {
        "artifact_kind": "phase1_signed_face_a2_b2_l1_terminal_result",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "launcher_sha256": sha256_file(launcher_path),
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "execution_lane": evidence["claim"]["execution_lane"],
        "training_source": expected_source_identity(manifest),
        "training_launch_claim": evidence["claim"],
        "training_launch_claim_sha256": evidence["claim_sha"],
        "training_run_dir": str(evidence["run_dir"]),
        "hard_contract_path": str(evidence["contract_path"]),
        "hard_contract_sha256": evidence["hard_sha"],
        "terminal_checkpoint_path": str(checkpoint),
        "terminal_checkpoint_sha256": checkpoint_sha,
        "terminal_checkpoint_file_identity": identity(after),
        "terminal_checkpoint_size_bytes": after.st_size,
        "checkpoint_audit": audit,
        "training_log_sha256": sha256_file(log_path),
        "training_log_file_identity": identity(log_stat),
        "exact_trainer_natural_exit_observed": True,
        "gpu_empty_terminal_barrier_observed": True,
        "kit_thread_cap_marker_occurrences": 1,
        "instantiated_zero_friction_marker_occurrences": 1,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "real_robot_commands_executed": False,
    }


def load_and_verify_terminal_result(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    preflight: dict[str, Any], cell_id: str,
) -> dict[str, Any]:
    path = expected_arm_paths(manifest, cell_id)["result"]
    recorded = read_json_object(path, f"{cell_id} terminal result")
    expected = audit_terminal(
        manifest, manifest_path, launcher_path, preflight, cell_id, stable_delay=0.0
    )
    require_exact(recorded, expected, f"{cell_id} terminal result replay")
    return recorded


def select_one_shot_cell(manifest: dict[str, Any], cell_id: str, host: str) -> str:
    if cell_id not in CELL_IDS:
        raise ContractError("one-shot launch requires exact cell A2 or B2")
    require_exact(host, cells(manifest)[cell_id]["host"], "cell host")
    paths = expected_arm_paths(manifest, cell_id)
    if paths["failure"].exists():
        raise ContractError(f"{cell_id} failure is preserved; no automatic retry is allowed")
    if paths["arm"].exists():
        raise ContractError(f"{cell_id} fresh namespace is already claimed")
    return cell_id


def launch_one(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    cell_id: str, host: str, root_confirm: str | None,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise ContractError("launch-one requires root on the simulator Pod")
    if root_confirm != ROOT_CONFIRMATION:
        raise ContractError("launch-one requires the exact simulation-only confirmation token")
    control = verify_external_control_location(manifest, manifest_path, launcher_path)
    cell_id = select_one_shot_cell(manifest, cell_id, host)
    preflight = verify_runtime(
        manifest, cell_id, host, require_empty_gpu=True
    )
    paths = expected_arm_paths(manifest, cell_id)
    run_root = Path(manifest["runtime"]["run_root"])
    no_symlink_existing_components(run_root, "run root")
    run_root.mkdir(parents=True, exist_ok=True)
    require_directory(run_root, "run root")
    # No glob-as-latest: the exact run name must be absent before its one claim.
    logs_root = preflight["wbt"] / "logs/rsl_rl/agibot_a3_hope_virtualball"
    if logs_root.is_dir() and list(logs_root.glob(f"*_{cells(manifest)[cell_id]['run_name']}")):
        raise ContractError(f"{cell_id} exact training run name already exists")
    paths["arm"].mkdir(exist_ok=False)
    arm_identity = identity(paths["arm"].stat())
    manifest_sha = sha256_file(manifest_path)
    launcher_sha = sha256_file(launcher_path)
    claim = build_claim(
        manifest, manifest_sha=manifest_sha, launcher_sha=launcher_sha,
        cell_id=cell_id, arm_dir=paths["arm"], arm_identity=arm_identity,
    )
    claim_sha = canonical_sha256(claim)
    command = build_command(manifest, cell_id, claim_sha)
    launch_contract = {
        "artifact_kind": "phase1_signed_face_a2_b2_l1_launch_contract",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "training_source": expected_source_identity(manifest),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "command": command,
        "external_control": control,
        "atomic_claim_directory_identity": arm_identity,
        "gpu_snapshot_before": preflight["gpu_snapshot"],
        "execution_lane": claim["execution_lane"],
        "training_environment_sha256": preflight["training_environment_sha256"],
        "explicit_pythonpath": preflight["environment"]["PYTHONPATH"],
        "kit_thread_caps": manifest["runtime"]["kit_thread_cap_contract"],
        "verified_inputs": preflight["verified_inputs"],
        "runtime_closure": preflight["runtime_closure"],
        "automatic_retry": False,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "real_robot_commands_forbidden": True,
    }
    write_json_exclusive(paths["launch"], launch_contract)
    environment = preflight["environment"].copy()
    environment.update({
        "KIT_BOOT_MARKER": HARD_CONTRACT_MARKER,
        "KIT_BOOT_TIMEOUT_S": str(manifest["runtime"]["kit_boot_timeout_seconds"]),
        "KIT_BOOT_POLL_S": str(manifest["runtime"]["poll_seconds"]),
        "KIT_BOOT_STATE_FILE": str(paths["state"]),
    })
    completed = subprocess.run(
        [str(preflight["locked"]), str(paths["log"]), *command],
        cwd=preflight["wbt"], env=environment, check=False,
    )
    if completed.returncode != 0:
        failure = {
            "artifact_kind": "phase1_signed_face_a2_b2_l1_launch_failure",
            "schema_version": 1,
            "manifest_sha256": manifest_sha,
            "launcher_sha256": launcher_sha,
            "cell_id": cell_id,
            "wrapper_returncode": completed.returncode,
            "launch_contract_sha256": sha256_file(paths["launch"]),
            "automatic_retry": False,
            "manual_diagnosis_required": True,
        }
        if paths["state"].is_file():
            failure["launch_state_sha256"] = sha256_file(paths["state"])
        if paths["log"].is_file():
            failure["training_log_sha256"] = sha256_file(paths["log"])
        write_json_exclusive(paths["failure"], failure)
        raise ContractError(
            f"{cell_id} reviewed wrapper failed rc={completed.returncode}; evidence preserved, no retry"
        )
    state = parse_state(paths["state"])
    if not state.get("pid", "").isdigit() or state.get("pgid") != state.get("pid"):
        raise ContractError(f"{cell_id} wrapper did not record isolated pid==pgid")
    pid = int(state["pid"])
    starttime = process_starttime(pid)
    if starttime <= 0:
        raise ContractError(f"{cell_id} trainer exited before runtime verification")
    run_dir = locate_training_run(preflight["wbt"], cells(manifest)[cell_id]["run_name"])
    run_stat = require_directory(run_dir, f"{cell_id} training run directory")
    contract_path = run_dir / "params/training_contract.json"
    contract_stat = require_regular(contract_path, f"{cell_id} adjacent hard contract")
    hard_sha, hard_contract = verify_hard_contract(contract_path, manifest, cell_id)
    verify_parent_to_current_contract(hard_contract, manifest, cell_id)
    checkpoint = run_dir / "model_13824.pt"
    if checkpoint.exists():
        raise ContractError(f"{cell_id} terminal checkpoint appeared before runtime verification")
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    if log_text.count(ZERO_FRICTION_RUNTIME_MARKER) != 1:
        raise ContractError(
            f"{cell_id} log lacks exactly one instantiated zero-friction marker"
        )
    if THREAD_MARKER not in log_text:
        raise ContractError(f"{cell_id} log lacks verified 16/16 thread marker")
    runtime_verified = {
        "artifact_kind": "phase1_signed_face_a2_b2_l1_runtime_verified",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "training_source": expected_source_identity(manifest),
        "training_launch_claim": claim,
        "training_launch_claim_sha256": claim_sha,
        "command": command,
        "external_control": control,
        "execution_lane": claim["execution_lane"],
        "training_environment_sha256": preflight["training_environment_sha256"],
        "pid": pid,
        "pgid": pid,
        "process_starttime_ticks": starttime,
        "training_run_dir": str(run_dir),
        "training_run_directory_identity": identity(run_stat),
        "hard_contract_path": str(contract_path),
        "hard_contract_file_identity": identity(contract_stat),
        "hard_contract_sha256": hard_sha,
        "checkpoint_absent_at_runtime_verification": True,
        "explicit_pythonpath": preflight["environment"]["PYTHONPATH"],
        "kit_thread_caps_verified": True,
        "instantiated_zero_friction_marker_verified": True,
        "activation_started": False,
        "judge_started": False,
        "l2_started": False,
        "second_seed_started": False,
        "real_robot_commands_executed": False,
    }
    write_json_exclusive(paths["runtime"], runtime_verified)
    return {
        "status": "one_arm_claimed_and_runtime_verified",
        "cell_id": cell_id,
        "pid": pid,
        "training_launch_claim_sha256": claim_sha,
        "hard_contract_sha256": hard_sha,
        "next_action": "wait for natural exit, then finalize this exact cell on its empty GPU",
    }


def finalize_cell(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    cell_id: str, host: str,
) -> dict[str, Any]:
    if cell_id not in CELL_IDS:
        raise ContractError("finalize-cell requires A2 or B2")
    require_exact(host, cells(manifest)[cell_id]["host"], "cell host")
    preflight = verify_runtime(manifest, cell_id, host, require_empty_gpu=True)
    paths = expected_arm_paths(manifest, cell_id)
    if paths["result"].exists():
        raise ContractError(f"{cell_id} terminal result already exists; no-clobber finalization")
    result = audit_terminal(manifest, manifest_path, launcher_path, preflight, cell_id)
    write_json_exclusive(paths["result"], result)
    return {
        "status": "l1_terminal_checkpoint_finite_exactly_bound",
        "cell_id": cell_id,
        "terminal_result_path": str(paths["result"]),
        "terminal_result_sha256": sha256_file(paths["result"]),
        "hard_contract_sha256": result["hard_contract_sha256"],
        "training_launch_claim_sha256": result["training_launch_claim_sha256"],
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
    }


def pair_input_paths(manifest: dict[str, Any]) -> dict[str, dict[str, Path]]:
    root = Path(manifest["runtime"]["pair_input_root"])
    return {
        cell_id: {
            "terminal": root / cell_id / "terminal_result.json",
            "hard": root / cell_id / "training_contract.json",
            "checkpoint": root / cell_id / "model_13824.pt",
        }
        for cell_id in CELL_IDS
    }


def verify_copied_terminal_result(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    cell_id: str, terminal_path: Path, hard_path: Path, checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    terminal_stat = require_regular(terminal_path, f"{cell_id} copied terminal result")
    hard_stat = require_regular(hard_path, f"{cell_id} copied hard contract")
    checkpoint_stat = require_regular(checkpoint_path, f"{cell_id} copied model_13824.pt")
    if any(result.st_mode & 0o222 for result in (terminal_stat, hard_stat, checkpoint_stat)):
        raise ContractError(f"{cell_id} pair inputs must be read-only")
    terminal = read_json_object(terminal_path, f"{cell_id} copied terminal result")
    require_exact_keys(terminal, TERMINAL_RESULT_KEYS, f"{cell_id} copied terminal result")
    manifest_sha = sha256_file(manifest_path)
    launcher_sha = sha256_file(launcher_path)
    expected_scalars = {
        "artifact_kind": "phase1_signed_face_a2_b2_l1_terminal_result",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "cell_id": cell_id,
        "run_name": cells(manifest)[cell_id]["run_name"],
        "training_source": expected_source_identity(manifest),
        "optimization_recipe": optimization_recipe(manifest, cell_id),
        "exact_trainer_natural_exit_observed": True,
        "gpu_empty_terminal_barrier_observed": True,
        "kit_thread_cap_marker_occurrences": 1,
        "instantiated_zero_friction_marker_occurrences": 1,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "real_robot_commands_executed": False,
    }
    for key, expected in expected_scalars.items():
        require_exact(terminal.get(key), expected, f"{cell_id} copied terminal {key}")
    claim = terminal.get("training_launch_claim")
    claim_sha = require_sha(
        terminal.get("training_launch_claim_sha256"),
        f"{cell_id} copied terminal claim SHA",
    )
    if not isinstance(claim, dict) or canonical_sha256(claim) != claim_sha:
        raise ContractError(f"{cell_id} copied terminal claim digest changed")
    directory = claim.get("claim_directory")
    if not isinstance(directory, dict):
        raise ContractError(f"{cell_id} copied terminal claim lacks directory identity")
    expected_claim = build_claim(
        manifest,
        manifest_sha=manifest_sha,
        launcher_sha=launcher_sha,
        cell_id=cell_id,
        arm_dir=expected_arm_paths(manifest, cell_id)["arm"],
        arm_identity={
            "device": directory.get("st_dev"),
            "inode": directory.get("st_ino"),
        },
    )
    require_exact(claim, expected_claim, f"{cell_id} copied terminal claim")
    require_exact(
        terminal.get("execution_lane"), expected_claim["execution_lane"],
        f"{cell_id} copied terminal execution lane",
    )
    training_run_dir = Path(str(terminal.get("training_run_dir", "")))
    expected_logs = (
        Path(manifest["source"]["training_checkout"])
        / manifest["source"]["wbt_relative_path"]
        / "logs/rsl_rl/agibot_a3_hope_virtualball"
    )
    if (
        not training_run_dir.is_absolute()
        or training_run_dir.parent != expected_logs
        or not training_run_dir.name.endswith(f"_{cells(manifest)[cell_id]['run_name']}")
    ):
        raise ContractError(f"{cell_id} copied terminal binds a foreign training run")
    require_exact(
        terminal.get("hard_contract_path"),
        str(training_run_dir / "params/training_contract.json"),
        f"{cell_id} copied terminal hard-contract path",
    )
    require_exact(
        terminal.get("terminal_checkpoint_path"),
        str(training_run_dir / "model_13824.pt"),
        f"{cell_id} copied terminal checkpoint path",
    )
    hard_sha = sha256_file(hard_path)
    require_exact(
        terminal.get("hard_contract_sha256"), hard_sha,
        f"{cell_id} copied terminal hard-contract SHA",
    )
    audit = terminal.get("checkpoint_audit")
    require_exact_keys(audit, CHECKPOINT_AUDIT_KEYS, f"{cell_id} copied checkpoint audit")
    for key, expected in {
        "iter": 13824,
        "training_contract_schema_version": 3,
        "training_contract_sha256": hard_sha,
        "training_contract_lineage_exact": 0,
        "training_launch_claim_sha256": claim_sha,
        "nonfinite_floating_elements": 0,
    }.items():
        require_exact(audit.get(key), expected, f"{cell_id} copied checkpoint {key}")
    if type(audit.get("floating_tensor_count")) is not int or audit["floating_tensor_count"] <= 0:
        raise ContractError(f"{cell_id} copied checkpoint has no floating tensors")
    if type(audit.get("floating_elements")) is not int or audit["floating_elements"] <= 0:
        raise ContractError(f"{cell_id} copied checkpoint has no floating elements")
    recorded_checkpoint_sha = require_sha(
        terminal.get("terminal_checkpoint_sha256"),
        f"{cell_id} copied terminal checkpoint SHA",
    )
    require_exact(
        terminal.get("terminal_checkpoint_size_bytes"), checkpoint_stat.st_size,
        f"{cell_id} copied terminal checkpoint byte count",
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    require_exact(
        recorded_checkpoint_sha, checkpoint_sha,
        f"{cell_id} copied terminal checkpoint bytes/SHA",
    )
    recomputed_audit = checkpoint_audit(
        Path(manifest["runtime"]["isaac_python"]), checkpoint_path
    )
    require_exact(
        audit, recomputed_audit,
        f"{cell_id} copied terminal checkpoint recursive audit",
    )
    checkpoint_after = require_regular(
        checkpoint_path, f"{cell_id} stable copied model_13824.pt"
    )
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    if any(
        getattr(checkpoint_stat, key) != getattr(checkpoint_after, key)
        for key in stable_fields
    ):
        raise ContractError(f"{cell_id} copied checkpoint changed during recursive audit")
    hard_digest, hard = verify_hard_contract(hard_path, manifest, cell_id)
    require_exact(hard_digest, hard_sha, f"{cell_id} copied hard digest")
    verify_parent_to_current_contract(hard, manifest, cell_id)
    return terminal, hard


def verify_pair_checkpoint_architecture(
    terminals: dict[str, dict[str, Any]],
) -> None:
    if set(terminals) != set(CELL_IDS):
        raise ContractError("paired checkpoint audit requires A2 and B2")
    for key in ("floating_tensor_count", "floating_elements"):
        values = {
            cell_id: terminals[cell_id]["checkpoint_audit"][key]
            for cell_id in CELL_IDS
        }
        if values["A2"] != values["B2"]:
            raise ContractError(f"A2/B2 checkpoint shared architecture {key} differs")


def finalize_pair(
    manifest: dict[str, Any], manifest_path: Path, launcher_path: Path,
    host: str,
) -> dict[str, Any]:
    require_exact(host, manifest["runtime"]["pair_finalize_host"], "pair finalize host")
    if os.geteuid() != 0:
        raise ContractError("finalize-pair requires root on the simulator Pod")
    control = verify_external_control_location(manifest, manifest_path, launcher_path)
    host_snapshot = gpu_snapshot(0)
    verify_gpu_host_identity(manifest, host, host_snapshot)
    inputs = pair_input_paths(manifest)
    terminals: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for cell_id in CELL_IDS:
        terminals[cell_id], contracts[cell_id] = verify_copied_terminal_result(
            manifest, manifest_path, launcher_path, cell_id,
            inputs[cell_id]["terminal"], inputs[cell_id]["hard"],
            inputs[cell_id]["checkpoint"],
        )
    verify_pair_contracts(contracts)
    verify_pair_checkpoint_architecture(terminals)
    output = Path(manifest["runtime"]["pair_result_path"])
    if output.exists():
        raise ContractError("paired L1 result already exists; no-clobber finalization")
    result = {
        "artifact_kind": "phase1_signed_face_a2_b2_cross_pod_l1_pair_result",
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "launcher_sha256": sha256_file(launcher_path),
        "external_control": control,
        "pair_finalize_host": host,
        "pair_finalize_gpu0_uuid": host_snapshot["uuid"],
        "ordered_cells": list(CELL_IDS),
        "terminal_result_sha256_by_cell": {
            cell_id: sha256_file(inputs[cell_id]["terminal"])
            for cell_id in CELL_IDS
        },
        "hard_contract_sha256_by_cell": {
            cell_id: sha256_file(inputs[cell_id]["hard"])
            for cell_id in CELL_IDS
        },
        "terminal_checkpoint_sha256_by_cell": {
            cell_id: terminals[cell_id]["terminal_checkpoint_sha256"]
            for cell_id in CELL_IDS
        },
        "only_hard_contract_difference": "racket_guidance_reward.signed_face.weight",
        "all_current_only_values_preregistered_and_pair_equal": True,
        "both_model_13824_finite_iter13824_lineage0": True,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "real_robot_commands_executed": False,
    }
    write_json_exclusive(output, result)
    return {
        "status": "paired_l1_provenance_complete_no_decision",
        "paired_result_path": str(output),
        "paired_result_sha256": sha256_file(output),
        "judge": False,
        "l2": False,
        "second_seed": False,
    }


def default_manifest_path(launcher: Path | None = None) -> Path:
    launcher = Path(__file__).resolve() if launcher is None else launcher.resolve()
    external = launcher.parent / "phase1_signed_face_a2b2_l1_prereg_20260714.json"
    if external.is_file():
        return external
    return launcher.parents[1] / "configs/phase1_signed_face_a2b2_l1_prereg_20260714.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--manifest", type=Path,
        default=default_manifest_path(),
    )
    value.add_argument(
        "--mode",
        choices=(
            "plan", "static-validate", "validate-runtime", "launch-one",
            "finalize-cell", "finalize-pair",
        ),
        default="plan",
    )
    value.add_argument("--cell", choices=CELL_IDS)
    value.add_argument("--host", choices=("pod1", "pod2"))
    value.add_argument("--root-confirm")
    value.add_argument("--plan-output", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    launcher_path = Path(__file__).resolve()
    try:
        manifest = load_manifest(manifest_path)
        if args.mode == "static-validate":
            result = {"status": "static_valid_no_writes", "training_source": verify_static_source(manifest)}
        elif args.mode == "plan":
            result = build_plan(manifest, manifest_path, launcher_path)
            if args.plan_output is not None:
                write_json_exclusive(args.plan_output.resolve(), result)
        elif args.mode == "validate-runtime":
            if args.cell is None or args.host is None:
                raise ContractError("validate-runtime requires exact --cell and --host")
            verify_external_control_location(manifest, manifest_path, launcher_path)
            preflight = verify_runtime(
                manifest, args.cell, args.host, require_empty_gpu=True
            )
            result = {
                "status": "runtime_validated_no_launch",
                "cell_id": args.cell,
                "host": args.host,
                "gpu_snapshot": preflight["gpu_snapshot"],
                "explicit_pythonpath": preflight["environment"]["PYTHONPATH"],
                "hot_parent_checkpoint_audit": preflight["hot_parent_checkpoint_audit"],
            }
        elif args.mode == "launch-one":
            if args.cell is None or args.host is None:
                raise ContractError("launch-one requires exact --cell and --host")
            result = launch_one(
                manifest, manifest_path, launcher_path,
                args.cell, args.host, args.root_confirm,
            )
        elif args.mode == "finalize-cell":
            if args.cell is None or args.host is None:
                raise ContractError("finalize-cell requires exact --cell and --host")
            result = finalize_cell(
                manifest, manifest_path, launcher_path, args.cell, args.host
            )
        elif args.mode == "finalize-pair":
            if args.host is None:
                raise ContractError("finalize-pair requires exact --host pod1")
            result = finalize_pair(
                manifest, manifest_path, launcher_path, args.host
            )
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (ContractError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[signed-face-a2-b2-l1] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
