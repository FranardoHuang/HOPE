#!/usr/bin/env python3
"""Plan or launch one isolated measured-racket VendorV2 N1 diagnostic.

This launcher deliberately does not modify or reuse the formal VendorV1 launcher,
experiment name or namespace.  It is a thin safety adapter over the reviewed N1
diagnostic primitives: exact clean commit, tracked scientific blobs, external runtime
asset pins, fresh/no-clobber namespace and an explicit physical GPU UUID.  The default
still requires an empty GPU.  One claim-owned opt-in may admit a second VendorV2
diagnostic process, but only after its PID, memory use, checkout and no-clobber
namespace receipt are cross-validated; unknown co-residents fail closed.

Five recipes are code-owned.  All use the existing fixed-194 actor / 318-D critic canary
ABI and an exact immutable tape; only target validity/content differs.  The mask is constant
within each independent arm, so it is not an observation column.  Invalid target columns are
zero-filled.  This cannot be promoted to the final varying-ball/N73 ABI.  Reset installs a
tape row and never solves an inverse problem::

    current_lm                    111
    analytic_full                111
    analytic_no_velocity         101
    teacher_pos_face_no_velocity 101
    outcome_dense_only           000

``physical_ball=false`` means the analytic virtual ball/scorer is authoritative.  It does
not claim PhysX paddle contact.  Every stage is fresh, delay-0, single-GPU and
``diagnostic_unauthorized``.  A zero-PPO ``materialize`` stage first publishes the exact
fully composed reward receipt.  A separate zero-PPO ``recipe`` stage must consume that
receipt and publishes the exact dynamic-ready policy recipe.  ``oracle2`` consumes both,
runs two live teacher-qdes episodes and performs zero PPO updates.  ``oracle32`` reuses the
same lineage, exact-process cleanup and post-completion admission, but runs 32 episodes and
applies the code-owned tracking/capture/safety/exposure gate.  Smoke/probe512/long512/probe must
consume both artifacts, so neither SHA is guessed or inherited from an older lineage.
``plan`` is read-only; ``launch`` recomputes the plan and requires its exact claim digest.
No arbitrary Hydra override or resume input exists.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


THIS_FILE = Path(__file__).resolve()
BASE_FILE = THIS_FILE.with_name("launch_n1_reward_screen_diagnostic.py")
BASE_SPEC = importlib.util.spec_from_file_location("_measured_vendor_v2_base", BASE_FILE)
if BASE_SPEC is None or BASE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import diagnostic launcher base")
_B = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(_B)

ADMISSION_FILE = THIS_FILE.with_name("vendor_v2_gpu_admission.py")
ADMISSION_SPEC = importlib.util.spec_from_file_location(
    "_measured_vendor_v2_gpu_admission", ADMISSION_FILE
)
if ADMISSION_SPEC is None or ADMISSION_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import VendorV2 GPU admission module")
_A = importlib.util.module_from_spec(ADMISSION_SPEC)
ADMISSION_SPEC.loader.exec_module(_A)

EXACT_GROUP_FILE = THIS_FILE.with_name("exact_process_group.py")
EXACT_GROUP_SPEC = importlib.util.spec_from_file_location(
    "_measured_vendor_v2_exact_process_group", EXACT_GROUP_FILE
)
if EXACT_GROUP_SPEC is None or EXACT_GROUP_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import exact process-group helper")
_G = importlib.util.module_from_spec(EXACT_GROUP_SPEC)
sys.modules[EXACT_GROUP_SPEC.name] = _G
EXACT_GROUP_SPEC.loader.exec_module(_G)

SCHEMA_VERSION = 2
SPEC_KIND = "n1_measured_vendor_v2_diagnostic_spec_v2"
CLAIM_KIND = "n1_measured_vendor_v2_diagnostic_claim_v2"
EXPERIMENT_NAME = "agibot_a3_action_ball_measured_vendor_v2_n1_diagnostic"
TASK_PROFILE_ID = "HOPEPingPongActionBallA3VendorV2N1Diagnostic"
TASK_PROFILE_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV2N1Diagnostic.yaml"
)
VENDOR_V2_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA3VendorV2.yaml"
)
LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_measured_vendor_v2_diagnostic.py"
)
ADMISSION_SOURCE = (
    "hope_training/whole_body_tracking/scripts/vendor_v2_gpu_admission.py"
)
EXACT_GROUP_SOURCE = (
    "hope_training/whole_body_tracking/scripts/exact_process_group.py"
)
BASE_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)
MATERIALIZER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_measured_action_ball_n1_bundle.py"
)
TAPE_PRODUCER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_action_ball_n1_fixed_tape_variants.py"
)
HOPE_COMMANDS_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
)
ACTOR_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/actor_observation_contract.py"
)
FIXED_TAPE_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_fixed_question_tape.py"
)
TRAINING_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/training_contract.py"
)
EFFECTIVE_REWARD_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/effective_reward_recipe.py"
)
# Tonight's fixed-question canary deliberately retains the already exercised fixed-194 actor and
# 318-D critic.  Validity is constant per independent arm and does not need to identify an arm to
# the policy.  This is not the future varying-ball/N73 ABI.
ACTOR_CONTRACT = "action_ball_table_pose_twist_heading_task_teacher_start_v2"
BUNDLE_KIND = "measured_action_ball_n1_diagnostic_bundle_v1"
PHYSICAL_BALL_SEMANTICS = "analytic_virtual_ball_authoritative_physx_disabled"
ACTION_ID = "take_061_unit04_bh"
MEASURED_UID = "Take_061_unit04_BH"
ACTION_UID = 5527597793770800
MOTION_PATH = (
    "assets/motions/chingmu73_measured_v4_20260803/"
    "hope_Take_061_unit04_BH.npz"
)
MOTION_SHA256 = "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
RACKET_ALIGNMENT_GATES = frozenset(
    {
        "full_position_p95_le_0p05_m",
        "full_face_p95_le_10_deg",
        "full_long_axis_p95_le_10_deg",
        "full_so3_p95_le_10_deg",
        "hit_position_le_0p05_m",
        "hit_face_le_5_deg",
        "hit_long_axis_le_5_deg",
        "hit_so3_le_5_deg",
        "hit_velocity_direction_observable",
        "hit_velocity_direction_le_15_deg",
        "hit_velocity_relative_le_0p20",
    }
)
RECIPES: Mapping[str, tuple[bool, bool, bool]] = {
    "current_lm": (True, True, True),
    "analytic_full": (True, True, True),
    "analytic_no_velocity": (True, False, True),
    "teacher_pos_face_no_velocity": (True, False, True),
    "outcome_dense_only": (False, False, False),
}
TARGET_ORDER = ("position", "velocity", "face")
REWARD_MATERIALIZATION_PROFILE = "measured_vendor_v2_n1_static_v1"
REWARD_RECIPE_FILENAME = "measured_vendor_v2_effective_reward_recipe.json"
POLICY_RECIPE_FILENAME = "measured_vendor_v2_dynamic_ready_policy_recipe.json"
TEACHER_QDES_ORACLE_FILENAME = "teacher_qdes_oracle_2ep.json"
TEACHER_QDES_ORACLE32_FILENAME = "teacher_qdes_oracle_32ep.json"
TEACHER_ORACLE_STAGES = frozenset(("oracle2", "oracle32"))
ORACLE32_EPISODES = 32
ORACLE32_ACCEPTANCE = {
    "schema_version": 1,
    "single_stroke_terminal_count": 32,
    "exact_strike_observed_count": 32,
    "position_error_m_strict_lt": 0.075,
    "velocity_error_mps_strict_lt": 0.5,
    "face_error_deg_strict_lt": 15.0,
    "opportunity_count": 32,
    "capture_count": 32,
    "reject_count_max": 0,
    "unknown_attribution_count_max": 0,
    "unexpected_termination_count_max": 0,
    "projection_nonfinite_sample_count_max": 0,
    "preclamp_max_abs_error_rad_max": 2.0e-6,
}


def _is_teacher_oracle_stage(stage: str) -> bool:
    return stage in TEACHER_ORACLE_STAGES


def _teacher_oracle_episodes(stage: str) -> int:
    if stage == "oracle2":
        return 2
    if stage == "oracle32":
        return ORACLE32_EPISODES
    raise LaunchRefused("stage is not a teacher-qdes oracle")


def _teacher_oracle_filename(stage: str) -> str:
    if stage == "oracle2":
        return TEACHER_QDES_ORACLE_FILENAME
    if stage == "oracle32":
        return TEACHER_QDES_ORACLE32_FILENAME
    raise LaunchRefused("stage is not a teacher-qdes oracle")
# 下面三张严格键表的**生产方全在本仓**(scripts/train.py 的 teacher-qdes oracle 收据),
# 消费方是本文件。生产方多写一个字段,消费方就当场 LaunchRefused —— 那是设计要的
# fail-closed,但不该等到发射当天才发现。因此三张表都提成模块级常量,由
# tests/test_launch_n1_measured_vendor_v2_diagnostic.py 的
# ``test_teacher_qdes_receipt_key_sets_match_the_train_py_producer`` 直接 AST 读
# train.py 的字面量逐一对齐:漂了在 host 测试就红,不必先烧一次 Pod 时间。
# train.py 已经在本发射器的 TRAIN_SOURCE 钉子表里,配对是合法的 provenance 关系。
TEACHER_ORACLE_REJECT_KEYS = (
    "virtual_contact_face_reject_count",
    "virtual_contact_geometry_reject_count",
    "virtual_contact_nonfinite_reject_count",
    "virtual_contact_u_n_below_fit_reject_count",
    "virtual_contact_u_n_above_fit_reject_count",
)
TEACHER_QDES_RECEIPT_KEYS = (
    "control_step_denominator",
    "preclamp_max_abs_error_rad",
    "raw_action_max_abs",
    "teleport_used",
    "wait_hold_command_steps",
    "teacher_reference_command_steps",
)
TEACHER_CAPTURE_REJECTION_KEYS = (
    "opportunities",
    "captures",
    "rejects",
    "conserved",
)
RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64
# ``noise_std_type`` is owned by cfg/algo/ppo.yaml.  This must be a normal
# Hydra override; ``+`` is reserved for keys absent from the composed config.
POLICY_NOISE_STD_OVERRIDE = "algo.policy.noise_std_type=log"
DISABLED_PUSH_DORMANT_FIELDS = (
    "recipe",
    "interval_range_s",
    "combined_exclusive",
    "velocity_range",
)
BUNDLE_KEYS = (
    "schema_version",
    "artifact_type",
    "action_id",
    "action_uid",
    "measured_uid",
    "source_manifest",
    "fixed_n1_source_manifest",
    "motion",
    "racket_alignment_audit",
    "racket_alignment",
    "measured_bank_receipt",
    "measured_manifest_build_report",
    "measured_provenance",
    "core_contact_bundle",
    "prepared_core_bundle",
    "task_profile",
    "immutable_tape_build_report",
    "immutable_tape",
    "mechanical_audit",
    "mechanical_selection",
    "target_recipe",
    "target_validity",
    "runtime_contract",
    "claims",
)
DYNAMIC_READY_V2_KIND = "agibot_a3_action_dynamic_ready_candidate_v2"
_DYNAMIC_READY_V2_KEYS = (
    "schema_version",
    "kind",
    "action_id",
    "robot",
    "authorization",
    "ready_source",
    "sources",
    "teacher_reference",
    "physical_birth_composition",
    "physical_ready",
    "physical_birth_static_evidence",
    "runtime_plant",
    "hold_candidate",
    "required_next_gate",
    "non_claims",
    "producer",
    "content_sha256",
)
_MEASURED_DYNAMIC_READY_SOURCE_KEYS = (
    "stable_motion",
    "measured_bank_receipt",
    "measured_mechanical_audit",
    "physical_birth_seed",
    "mujoco_model",
    "runtime_training_contract",
)
_NOMINAL_HOLD_RECEIPT_KEYS = (
    "schema_version",
    "kind",
    "verdict",
    "action_id",
    "artifact",
    "motion_sha256",
    "teacher_reference_unchanged",
    "teacher_physical_birth_separated",
    "candidate_physical_birth_written",
    "candidate_hold_qdes_and_delay_history_installed",
    "plant_contract_match",
    "control_step_action_delay_runtime",
    "active_terminations",
    "requested_duration_s",
    "completed_duration_s",
    "completed_policy_steps",
    "completed_physics_steps",
    "terminal_reasons",
    "generic_terminated",
    "generic_truncated",
    "minimum_root_z_m",
    "maximum_root_tilt_rad",
    "both_feet_contact_fraction",
    "joint_safety_telemetry",
    "screenshots",
    "content_sha256",
)
BUDGETS = {
    "materialize": (1, 0, 1),
    "recipe": (1, 0, 1),
    "oracle2": (1, 0, 1),
    "oracle32": (1, 0, 1),
    "smoke": (1, 2, 1),
    # The full 4096-env A3 scene can exceed the 30-minute boot-staleness
    # watchdog before its first rollout on Pod1.  This bounded diagnostic
    # budget separates recipe learnability from that scene-scaling gate.
    "probe512": (512, 5, 1),
    # Promotion from the five-update recipe canary is still diagnostic-only,
    # fresh and bounded.  This is long enough to expose contact learning while
    # retaining periodic finite checkpoints for early causal review.
    "long512": (512, 1000, 100),
    "probe": (4096, 5, 1),
}
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CUDA_LAUNCH_BLOCKING_SPEC_KEY = "cuda_launch_blocking"
VENDOR_V2_COLOCATION_SPEC_KEY = "allow_vendor_v2_colocation"
MAX_VENDOR_V2_COMPUTE_PIDS = _A.MAX_VENDOR_V2_COMPUTE_PIDS
# Keep at least 8 GiB physically free after every admission check.  This is a
# code-owned conservative floor, not an operator-tunable capacity estimate.
MIN_VENDOR_V2_FREE_MEMORY_MIB = _A.MIN_VENDOR_V2_FREE_MEMORY_MIB
GPU_RESERVATION_FILENAME = _A.GPU_RESERVATION_FILENAME
GPU_NAMESPACE_RECEIPT_FILENAME = _A.GPU_NAMESPACE_RECEIPT_FILENAME
GPU_NAMESPACE_RECEIPT_ENV = _A.GPU_NAMESPACE_RECEIPT_ENV
GPU_NAMESPACE_RECEIPT_SHA_ENV = _A.GPU_NAMESPACE_RECEIPT_SHA_ENV
RUNTIME_SOURCE_PATHS = (
    (LAUNCHER_SOURCE, "VendorV2 N1 launcher"),
    (ADMISSION_SOURCE, "VendorV2 GPU admission"),
    (EXACT_GROUP_SOURCE, "exact process-group helper"),
    (BASE_LAUNCHER_SOURCE, "N1 safety base"),
    (MATERIALIZER_SOURCE, "measured N1 materializer"),
    (TAPE_PRODUCER_SOURCE, "offline fixed-tape producer"),
    (_B.TRAIN_SOURCE, "training entrypoint"),
    (TASK_PROFILE_SOURCE, "VendorV2 N1 task leaf"),
    (VENDOR_V2_SOURCE, "VendorV2 parent task"),
    (_B.KIT_LAUNCHER_SOURCE, "locked Kit launcher"),
    (HOPE_COMMANDS_SOURCE, "ActionBall runtime"),
    (ACTOR_CONTRACT_SOURCE, "actor observation contract"),
    (FIXED_TAPE_SOURCE, "immutable fixed-question tape runtime"),
    (TRAINING_CONTRACT_SOURCE, "dynamic-ready policy contract"),
    (EFFECTIVE_REWARD_SOURCE, "effective reward receipt contract"),
)

LaunchRefused = _B.LaunchRefused
canonical_sha256 = _B.canonical_sha256


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    return _B._exact_dict(value, keys, name=name)


def _validate_external_pin(value: Any, *, name: str) -> tuple[dict[str, str], Path]:
    row = _exact_dict(value, ("path", "sha256"), name=name)
    path = _B._absolute_path(row["path"], name=f"{name}.path", must_exist=True)
    _B._stable_regular_file(path, name=name)
    expected = _B._sha256(row["sha256"], name=f"{name}.sha256")
    observed = _B.sha256_file(path)
    if observed != expected:
        raise LaunchRefused(
            f"{name} file SHA differs: pin={expected} observed={observed}"
        )
    return {"path": str(path), "sha256": expected}, path


def _validate_reward_materialization(value: Any) -> dict[str, Any]:
    pin, path = _validate_external_pin(value, name="reward materialization")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="reward materialization")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("reward materialization must be canonical JSON plus newline")
    row = _exact_dict(
        document,
        ("schema_version", "terms", "sha256"),
        name="reward materialization",
    )
    if row["schema_version"] != 1 or type(row["terms"]) is not list:
        raise LaunchRefused("reward materialization schema differs")
    digest = canonical_sha256(
        {"schema_version": row["schema_version"], "terms": row["terms"]}
    )
    if row["sha256"] != digest:
        raise LaunchRefused("reward materialization semantic SHA differs")
    names = [
        term.get("name")
        for term in row["terms"]
        if type(term) is dict and type(term.get("name")) is str
    ]
    if (
        len(names) != len(row["terms"])
        or names != sorted(names)
        or len(names) != len(set(names))
    ):
        raise LaunchRefused("reward materialization terms are not canonical/unique")
    return {
        "artifact": pin,
        "effective_reward_recipe_sha256": digest,
        "term_count": len(names),
    }


def _validate_policy_materialization_header(value: Any) -> dict[str, Any]:
    pin, path = _validate_external_pin(value, name="policy materialization")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="policy materialization")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("policy materialization must be canonical JSON plus newline")
    row = _exact_dict(
        document,
        (
            "schema_version",
            "kind",
            "action_count",
            "action_order",
            "policy_contract_sha256",
            "action_ball_ppo_runner_recipe",
            "policy_bootstrap",
        ),
        name="policy materialization",
    )
    policy_sha = _B._sha256(
        row["policy_contract_sha256"], name="materialized policy contract SHA"
    )
    if policy_sha == RECIPE_SENTINEL_POLICY_SHA256:
        raise LaunchRefused("materialized policy contract cannot be the recipe sentinel")
    if (
        row["schema_version"] != 1
        or row["kind"]
        != "action_ball_shared_ready_policy_recipe_materialization_v1"
        or row["action_count"] != 1
        or row["action_order"] != [ACTION_ID]
    ):
        raise LaunchRefused("policy materialization header differs")
    return {
        "artifact": pin,
        "policy_contract_sha256": policy_sha,
        "document": row,
    }


def _isaac_python_entry(value: Any) -> Path:
    """Validate the real executable while preserving the venv entry pathname."""

    entry = _B._absolute_path(
        value, name="source.isaac_python", must_exist=True
    )
    try:
        real = entry.resolve(strict=True)
        info = real.stat()
    except OSError as exc:
        raise LaunchRefused("source.isaac_python cannot resolve to a real file") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(real, os.X_OK):
        raise LaunchRefused(
            "source.isaac_python must resolve to an executable regular file"
        )
    # Do not return ``real``.  Executing a resolved venv symlink bypasses the venv's
    # interpreter discovery and can silently drop its installed packages.
    return entry


def _validate_gpu(value: Any, *, allow_colocation: bool) -> dict[str, Any]:
    """Validate the physical GPU and bind empty/co-resident policy together."""

    row = _exact_dict(
        value,
        ("index", "uuid", "owner", "lock_path", "require_empty"),
        name="spec.gpu",
    )
    index = _B._plain_int(row["index"], name="spec.gpu.index", maximum=31)
    uuid = row["uuid"]
    if (
        type(uuid) is not str
        or not uuid.startswith("GPU-")
        or len(uuid) < 8
        or "," in uuid
        or "\n" in uuid
    ):
        raise LaunchRefused("spec.gpu.uuid must be an explicit GPU UUID")
    owner = row["owner"]
    if (
        type(owner) is not str
        or owner != owner.strip()
        or not owner
        or owner.lower()
        in {"codex", "claude", "fable", "agent", "unassigned"}
    ):
        raise LaunchRefused("spec.gpu.owner must be an explicit human name")
    lock_path = _B._absolute_path(row["lock_path"], name="spec.gpu.lock_path")
    expected_lock = Path("/tmp/hope_lean_queue_gpu%d.lock" % index)
    if lock_path != expected_lock:
        raise LaunchRefused("spec.gpu.lock_path must be %s" % expected_lock)
    expected_empty = not allow_colocation
    if row["require_empty"] is not expected_empty:
        raise LaunchRefused(
            "spec.gpu.require_empty must be %s when "
            "allow_vendor_v2_colocation=%s"
            % (str(expected_empty).lower(), str(allow_colocation).lower())
        )
    return {
        "index": index,
        "uuid": uuid,
        "owner": owner,
        "lock_path": str(lock_path),
        "require_empty": expected_empty,
    }


def _validate_spec(document: dict[str, Any], *, claimed: bool = False) -> dict[str, Any]:
    keys = (
        "schema_version",
        "kind",
        "source",
        "action_id",
        "bundle",
        "target_recipe",
        "target_validity_mask",
        "reward_materialization",
        "policy_materialization",
        "policy_contract_sha256",
        "expected_effective_reward_recipe_sha256",
        "seed",
        "stage",
        "num_envs",
        "max_iterations",
        "save_interval",
        "gpu",
        "namespace",
        "log_path",
    )
    actual_keys = frozenset(document) if type(document) is dict else frozenset()
    required_keys = frozenset(keys)
    optional_keys = frozenset(
        (CUDA_LAUNCH_BLOCKING_SPEC_KEY, VENDOR_V2_COLOCATION_SPEC_KEY)
    )
    if not required_keys.issubset(actual_keys) or not actual_keys.issubset(
        required_keys | optional_keys
    ):
        raise LaunchRefused(
            "launch spec keys differ: missing=%s extra=%s"
            % (
                sorted(required_keys - actual_keys),
                sorted(actual_keys - required_keys - optional_keys),
            )
        )
    row = dict(document)
    cuda_launch_blocking = row.get(CUDA_LAUNCH_BLOCKING_SPEC_KEY, False)
    if type(cuda_launch_blocking) is not bool:
        raise LaunchRefused("cuda_launch_blocking must be a boolean")
    allow_vendor_v2_colocation = row.get(VENDOR_V2_COLOCATION_SPEC_KEY, False)
    if type(allow_vendor_v2_colocation) is not bool:
        raise LaunchRefused("allow_vendor_v2_colocation must be a boolean")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused("launch spec schema/kind differs")
    source = _exact_dict(
        row["source"], ("checkout", "commit_sha", "isaac_python"), name="spec.source"
    )
    checkout = _B._absolute_path(source["checkout"], name="source.checkout", must_exist=True)
    commit = source["commit_sha"]
    if type(commit) is not str or _B.COMMIT_RE.fullmatch(commit) is None:
        raise LaunchRefused("source.commit_sha must be exact lowercase 40-hex")
    isaac_python = _isaac_python_entry(source["isaac_python"])
    action_id = row["action_id"]
    if action_id != ACTION_ID:
        raise LaunchRefused("action_id must be the code-owned %s" % ACTION_ID)
    bundle = _exact_dict(row["bundle"], ("path", "sha256"), name="spec.bundle")
    recipe = row["target_recipe"]
    if type(recipe) is not str or recipe not in RECIPES:
        raise LaunchRefused("target_recipe must be one of the five code-owned recipes")
    if row["target_validity_mask"] != list(RECIPES[recipe]):
        raise LaunchRefused("target_validity_mask differs from target_recipe")
    seed = _B._plain_int(row["seed"], name="seed", maximum=(1 << 31) - 1)
    if seed != 0:
        raise LaunchRefused("this fixed first-wave launcher requires seed 0")
    stage = row["stage"]
    if stage not in BUDGETS:
        raise LaunchRefused(
            "stage must be materialize, recipe, oracle2, oracle32, smoke, "
            "probe512, long512, or probe"
        )
    if (
        stage in ("materialize", "recipe") or _is_teacher_oracle_stage(stage)
    ) and recipe != "current_lm":
        raise LaunchRefused(
            "%s stage must use the code-owned current_lm identity arm" % stage
        )
    reward_materialization = None
    policy_materialization = None
    if stage == "materialize":
        if (
            row["reward_materialization"] is not None
            or row["policy_materialization"] is not None
            or row["expected_effective_reward_recipe_sha256"] is not None
            or row["policy_contract_sha256"] is not None
        ):
            raise LaunchRefused(
                "materialize stage must not predeclare reward/policy identities"
            )
        reward_sha = None
        policy_sha = None
    else:
        reward_materialization = _validate_reward_materialization(
            row["reward_materialization"]
        )
        reward_sha = _B._sha256(
            row["expected_effective_reward_recipe_sha256"],
            name="effective reward SHA",
        )
        if reward_sha != reward_materialization["effective_reward_recipe_sha256"]:
            raise LaunchRefused(
                "expected reward SHA differs from its materialized receipt"
            )
        if stage == "recipe":
            if (
                row["policy_materialization"] is not None
                or row["policy_contract_sha256"] is not None
            ):
                raise LaunchRefused(
                    "recipe stage must materialize rather than predeclare policy identity"
                )
            policy_sha = None
        else:
            policy_materialization = _validate_policy_materialization_header(
                row["policy_materialization"]
            )
            policy_sha = _B._sha256(
                row["policy_contract_sha256"], name="policy contract SHA"
            )
            if policy_sha != policy_materialization["policy_contract_sha256"]:
                raise LaunchRefused(
                    "policy contract SHA differs from its materialized recipe"
                )
    expected_budget = BUDGETS[stage]
    actual_budget = (
        _B._plain_int(row["num_envs"], name="num_envs", minimum=1),
        _B._plain_int(row["max_iterations"], name="max_iterations", minimum=0),
        _B._plain_int(row["save_interval"], name="save_interval", minimum=1),
    )
    if actual_budget != expected_budget:
        raise LaunchRefused(
            "%s budget must be exactly %s" % (stage, expected_budget)
        )
    gpu = _validate_gpu(row["gpu"], allow_colocation=allow_vendor_v2_colocation)
    namespace = _B._absolute_path(row["namespace"], name="namespace")
    if SAFE_COMPONENT.fullmatch(namespace.name or "") is None:
        raise LaunchRefused("namespace basename is unsafe")
    log_path = _B._absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must equal <namespace>/run.log")
    if os.path.lexists(namespace):
        if not claimed:
            raise LaunchRefused("namespace already exists and is permanently spent")
        info = namespace.lstat()
        if not stat.S_ISDIR(info.st_mode) or namespace.resolve(strict=True) != namespace:
            raise LaunchRefused("claimed namespace is not a real directory")
    elif claimed:
        raise LaunchRefused("claimed namespace vanished")
    parent = namespace.parent
    if not parent.exists() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise LaunchRefused("namespace parent must be an existing real directory")
    if parent.name != EXPERIMENT_NAME:
        raise LaunchRefused(
            "namespace parent must be the dedicated VendorV2 diagnostic root %s"
            % EXPERIMENT_NAME
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": commit,
            "isaac_python": str(isaac_python),
        },
        "action_id": action_id,
        "bundle": dict(bundle),
        "target_recipe": recipe,
        "target_validity_mask": list(RECIPES[recipe]),
        "reward_materialization": (
            None
            if reward_materialization is None
            else reward_materialization["artifact"]
        ),
        "policy_materialization": (
            None
            if policy_materialization is None
            else policy_materialization["artifact"]
        ),
        "policy_contract_sha256": policy_sha,
        "expected_effective_reward_recipe_sha256": reward_sha,
        "seed": seed,
        "stage": stage,
        "num_envs": actual_budget[0],
        "max_iterations": actual_budget[1],
        "save_interval": actual_budget[2],
        "gpu": gpu,
        "namespace": str(namespace),
        "log_path": str(log_path),
        CUDA_LAUNCH_BLOCKING_SPEC_KEY: cuda_launch_blocking,
        VENDOR_V2_COLOCATION_SPEC_KEY: allow_vendor_v2_colocation,
    }


def _cuda_launch_blocking_environment(
    spec: Mapping[str, Any],
) -> dict[str, str]:
    """Return the one claim-owned synchronous-CUDA diagnostic setting."""

    if spec[CUDA_LAUNCH_BLOCKING_SPEC_KEY] is True:
        return {"CUDA_LAUNCH_BLOCKING": "1"}
    return {}


def _pin_tracked(checkout: Path, commit: str, relative: str, *, name: str) -> dict[str, str]:
    path = checkout / relative
    pin = {"path": relative, "sha256": _B.sha256_file(path)}
    normalized, _path = _B._verify_tracked_file(checkout, commit, pin, name=name)
    return normalized


def _runtime_sources(checkout: Path, commit: str) -> dict[str, dict[str, str]]:
    rows = {}
    for relative, name in RUNTIME_SOURCE_PATHS:
        rows[name] = _pin_tracked(checkout, commit, relative, name=name)
    if THIS_FILE != checkout / LAUNCHER_SOURCE:
        raise LaunchRefused("running launcher is not the selected checkout launcher")
    if ADMISSION_FILE != checkout / ADMISSION_SOURCE:
        raise LaunchRefused("running admission module is not from the selected checkout")
    if EXACT_GROUP_FILE != checkout / EXACT_GROUP_SOURCE:
        raise LaunchRefused("running process-group helper is not from the selected checkout")
    # These markers are the minimum source-level proof that the new ABI is wired.  Bundle/runtime
    # receipt validation still owns semantics; absence fails before GPU/namespace mutation.
    command_bytes = (checkout / HOPE_COMMANDS_SOURCE).read_bytes()
    for marker in (
        b"action_ball_target_source",
        b"action_ball_immutable_tape_path",
        b"action_ball_immutable_tape_sha256",
        b"action_ball_target_recipe",
        b"action_ball_target_validity_mask",
        b"immutable_tape",
    ):
        if marker not in command_bytes:
            raise LaunchRefused(
                "immutable-tape runtime dependency is not wired: missing %s"
                % marker.decode("ascii")
            )
    actor_bytes = (checkout / ACTOR_CONTRACT_SOURCE).read_bytes()
    if ACTOR_CONTRACT.encode("ascii") not in actor_bytes:
        raise LaunchRefused("existing fixed-194 ActionBall actor ABI is not wired")
    return rows


def _validate_measured_dynamic_ready_v2(
    checkout: Path,
    commit: str,
    value: Any,
    *,
    action_id: str,
    motion_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Validate the measured launcher's exact schema-v2 hold handoff."""

    row = _B._exact_dict(
        value, _B._DYNAMIC_READY_KEYS, name="measured N1 bundle.dynamic_ready"
    )
    artifact_pin, candidate = _B._load_tracked_json(
        checkout,
        commit,
        row["artifact"],
        name="measured N1 dynamic-ready artifact",
    )
    candidate = _B._exact_dict(
        candidate,
        _DYNAMIC_READY_V2_KEYS,
        name="measured N1 dynamic-ready artifact",
    )
    candidate_content_sha = _B._verify_content_seal(
        candidate,
        name="measured N1 dynamic-ready artifact",
        ensure_ascii=True,
    )
    robot = _B._exact_dict(
        candidate["robot"],
        ("family", "joint_names"),
        name="measured N1 dynamic-ready robot",
    )
    sources = _B._exact_dict(
        candidate["sources"],
        _MEASURED_DYNAMIC_READY_SOURCE_KEYS,
        name="measured N1 dynamic-ready sources",
    )
    stable_motion = _B._exact_dict(
        sources["stable_motion"],
        ("path", "sha256", "frame_index"),
        name="measured N1 dynamic-ready stable motion",
    )
    required_gate = _B._exact_dict(
        candidate["required_next_gate"],
        ("kind", "minimum_horizon_semantics", "zero_terminal_required"),
        name="measured N1 dynamic-ready required gate",
    )
    authorization = _B._exact_dict(
        candidate["authorization"],
        (
            "training_authorized",
            "deployment_authorized",
            "hardware_authorized",
            "isaac_nominal_hold_validated",
        ),
        name="measured N1 dynamic-ready authorization",
    )
    if (
        candidate["schema_version"] != 2
        or candidate["kind"] != DYNAMIC_READY_V2_KIND
        or candidate["action_id"] != action_id
        or robot["family"] != "AgiBot A3"
        or type(robot["joint_names"]) is not list
        or len(robot["joint_names"]) != 31
        or len(set(robot["joint_names"])) != 31
        or any(type(name) is not str or not name for name in robot["joint_names"])
        or stable_motion["frame_index"] != 0
        or stable_motion["sha256"] != motion_sha256
        or type(candidate["runtime_plant"]) is not dict
        or not candidate["runtime_plant"]
        or any(flag is not False for flag in authorization.values())
        or required_gate["kind"] != _B.NOMINAL_HOLD_RECEIPT_KIND
        or required_gate["minimum_horizon_semantics"]
        != "validated_t_hit_plus_reaction_margin"
    ):
        raise LaunchRefused(
            "measured launch requires the exact schema-v2 A3 action/motion plant"
        )

    receipt_pin, receipt = _B._load_tracked_json(
        checkout,
        commit,
        row["nominal_hold_receipt"],
        name="measured N1 nominal-hold receipt",
    )
    receipt = _B._exact_dict(
        receipt,
        _NOMINAL_HOLD_RECEIPT_KEYS,
        name="measured N1 nominal-hold receipt",
    )
    _B._verify_content_seal(
        receipt,
        name="measured N1 nominal-hold receipt",
        ensure_ascii=False,
    )
    receipt_artifact = _B._exact_dict(
        receipt["artifact"],
        ("path", "sha256", "content_sha256"),
        name="measured N1 nominal-hold receipt artifact",
    )
    required_terminations = required_gate["zero_terminal_required"]
    active_terminations = receipt["active_terminations"]
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != _B.NOMINAL_HOLD_RECEIPT_KIND
        or receipt["verdict"] != "PASS"
        or receipt["action_id"] != action_id
        or receipt["motion_sha256"] != motion_sha256
        or receipt["teacher_reference_unchanged"] is not True
        or receipt["teacher_physical_birth_separated"] is not True
        or receipt["candidate_physical_birth_written"] is not True
        or receipt["candidate_hold_qdes_and_delay_history_installed"] is not True
        or receipt["plant_contract_match"] is not True
        or receipt["terminal_reasons"] != []
        or receipt["generic_terminated"] is not False
        or receipt["generic_truncated"] is not False
        or receipt_artifact["sha256"] != artifact_pin["sha256"]
        or receipt_artifact["content_sha256"] != candidate_content_sha
        or type(required_terminations) is not list
        or type(active_terminations) is not list
        or not all(
            type(reason) is str and reason in active_terminations
            for reason in required_terminations
        )
    ):
        raise LaunchRefused(
            "measured nominal-hold receipt does not prove the exact schema-v2 "
            "action/motion plant with zero terminal"
        )
    try:
        binding = _load_training_contract_module(
            checkout
        ).load_action_ball_dynamic_ready_runtime_binding(
            artifact_path=str(checkout / artifact_pin["path"]),
            artifact_sha256=artifact_pin["sha256"],
            nominal_hold_receipt_path=str(checkout / receipt_pin["path"]),
            nominal_hold_receipt_sha256=receipt_pin["sha256"],
            action_order=[action_id],
            motion_paths=[str(checkout / MOTION_PATH)],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchRefused(
            f"measured schema-v2 dynamic-ready runtime binding is invalid: {exc}"
        ) from exc
    if (
        binding.get("schema_version") != 2
        or binding.get("kind") != "action_ball_dynamic_ready_runtime_binding_v2"
        or binding.get("action_order") != [action_id]
        or binding.get("motion_sha256_per_action") != [motion_sha256]
    ):
        raise LaunchRefused(
            "measured dynamic-ready runtime binding identity differs"
        )
    return {
        "artifact": artifact_pin,
        "nominal_hold_receipt": receipt_pin,
    }


def _validate_bundle(
    checkout: Path,
    commit: str,
    pin: dict[str, Any],
    *,
    action_id: str,
    recipe: str,
    seed: int,
) -> dict[str, Any]:
    normalized, bundle = _B._load_tracked_json(checkout, commit, pin, name="measured N1 bundle")
    bundle = _exact_dict(bundle, BUNDLE_KEYS, name="measured N1 bundle")
    if bundle["schema_version"] != 1 or bundle["artifact_type"] != BUNDLE_KIND:
        raise LaunchRefused("measured bundle schema/kind differs")
    if bundle["action_id"] != action_id or bundle["target_recipe"] != recipe:
        raise LaunchRefused("measured bundle action/recipe differs")
    if bundle["action_uid"] != ACTION_UID or bundle["measured_uid"] != MEASURED_UID:
        raise LaunchRefused("measured bundle numeric/string action identity differs")
    prepared_pin, prepared = _B._load_tracked_json(
        checkout,
        commit,
        bundle["prepared_core_bundle"],
        name="prepared measured N1 core bundle",
    )
    prepared_keys = {
        "schema_version",
        "artifact_type",
        "action_id",
        "action_uid",
        "measured_uid",
        "source_manifest",
        "fixed_n1_source_manifest",
        "motion",
        "racket_alignment_audit",
        "racket_alignment",
        "measured_bank_receipt",
        "measured_manifest_build_report",
        "measured_provenance",
        "core_contact_bundle",
        "task_profile",
        "mechanical_audit",
        "mechanical_selection",
        "claims",
    }
    if (
        set(prepared) != prepared_keys
        or prepared.get("schema_version") != 1
        or prepared.get("artifact_type") != "measured_action_ball_n1_prepared_core_v1"
        or prepared.get("action_id") != ACTION_ID
        or prepared.get("action_uid") != ACTION_UID
        or prepared.get("measured_uid") != MEASURED_UID
        or any(
            bundle[key] != prepared[key]
            for key in prepared_keys - {"artifact_type"}
        )
    ):
        raise LaunchRefused("final bundle differs from its exact prepared core")
    if prepared.get("claims", {}).get("dynamic_ready_status") != "PASS":
        raise LaunchRefused("prepared core is not exact dynamic-ready plus nominal-hold PASS")
    validity = bundle["target_validity"]
    if validity != {"order": list(TARGET_ORDER), "mask": list(RECIPES[recipe])}:
        raise LaunchRefused("measured bundle validity mask differs")
    task_pin, _task_path = _B._verify_tracked_file(
        checkout,
        commit,
        bundle["task_profile"],
        name="VendorV2 N1 task profile",
    )
    if task_pin["path"] != TASK_PROFILE_SOURCE:
        raise LaunchRefused("bundle task profile is not the isolated VendorV2 N1 leaf")
    legacy_dynamic_ready_validator = _B._validate_dynamic_ready
    try:
        _B._validate_dynamic_ready = _validate_measured_dynamic_ready_v2
        core = _B._validate_bundle(
            checkout,
            commit,
            bundle["core_contact_bundle"],
            expected_action=action_id,
            expected_scope="full",
            require_dynamic_ready=True,
        )
    finally:
        _B._validate_dynamic_ready = legacy_dynamic_ready_validator
    motion_pin, _motion_path = _B._verify_tracked_file(
        checkout, commit, bundle["motion"], name="measured motion"
    )
    if motion_pin != {"path": MOTION_PATH, "sha256": MOTION_SHA256}:
        raise LaunchRefused("measured bundle motion is not the code-owned exact clip")
    if core["motion"] != motion_pin:
        raise LaunchRefused("core bundle and measured wrapper motion differ")
    alignment_pin, alignment_report = _B._load_tracked_json(
        checkout,
        commit,
        bundle["racket_alignment_audit"],
        name="independent racket FK alignment audit",
    )
    alignment = bundle["racket_alignment"]
    if (
        alignment_report.get("admitted") is not True
        or alignment_report.get("finite") is not True
        or alignment_report.get("motion_sha256") != motion_pin["sha256"]
        or type(alignment_report.get("gates")) is not dict
        or set(alignment_report["gates"]) != RACKET_ALIGNMENT_GATES
        or any(value is not True for value in alignment_report["gates"].values())
        or alignment_report.get("uid") != MEASURED_UID
        or type(alignment) is not dict
        or alignment.get("motion_sha256") != motion_pin["sha256"]
        or alignment.get("all_11_gates_pass") is not True
        or alignment.get("diagnostic_unauthorized") is not True
    ):
        raise LaunchRefused("independent racket FK alignment evidence differs")
    bank_pin, bank = _B._load_tracked_json(
        checkout,
        commit,
        bundle["measured_bank_receipt"],
        name="measured bank receipt",
    )
    report_pin, report = _B._load_tracked_json(
        checkout,
        commit,
        bundle["measured_manifest_build_report"],
        name="measured manifest build report",
    )
    if (
        report.get("measured_bank_receipt_sha256") != bank_pin["sha256"]
        or report.get("file_sha256") != bundle["source_manifest"].get("sha256")
        or report.get("racket_authority") != "measured_channel"
        or bank.get("authorization", {}).get("training") is not False
        or bank.get("authorization", {}).get("mechanical_admission") is not False
        or type(bundle["measured_provenance"]) is not dict
    ):
        raise LaunchRefused("measured bank/build provenance boundary differs")
    _B._load_tracked_json(
        checkout,
        commit,
        bundle["source_manifest"],
        name="measured source manifest",
    )
    _B._load_tracked_json(
        checkout,
        commit,
        bundle["fixed_n1_source_manifest"],
        name="fixed N1 source manifest",
    )
    _B._load_tracked_json(
        checkout,
        commit,
        bundle["mechanical_audit"],
        name="mechanical audit",
    )
    selection = bundle["mechanical_selection"]
    if (
        type(selection) is not dict
        or selection.get("motion_sha256") != motion_pin["sha256"]
        or selection.get("kinematic_limit_verdict") != "PASS"
        or selection.get("mechanical_verdict") not in ("PASS", "UNKNOWN")
        or selection.get("diagnostic_unauthorized") is not True
    ):
        raise LaunchRefused("mechanical diagnostic selection is not eligible")
    report_pin, _report = _B._load_tracked_json(
        checkout,
        commit,
        bundle["immutable_tape_build_report"],
        name="immutable tape build report",
    )
    materializer_spec = importlib.util.spec_from_file_location(
        "_measured_vendor_v2_materializer_validation",
        checkout / MATERIALIZER_SOURCE,
    )
    if materializer_spec is None or materializer_spec.loader is None:
        raise LaunchRefused("cannot import measured N1 materializer")
    materializer = importlib.util.module_from_spec(materializer_spec)
    sys.modules[materializer_spec.name] = materializer
    try:
        materializer_spec.loader.exec_module(materializer)
        if set(BUNDLE_KEYS) != set(materializer.FINAL_BUNDLE_KEYS):
            raise LaunchRefused("finalize/launcher bundle schema differs")
        validated_report_pin, tape = materializer._validate_tape_build_report(
            checkout,
            checkout / report_pin["path"],
            report_pin["sha256"],
            action_uid=ACTION_UID,
            motion_sha=MOTION_SHA256,
            recipe=recipe,
        )
    except Exception as exc:
        raise LaunchRefused(
            "immutable tape build report failed validation: %s" % exc
        ) from exc
    if validated_report_pin != report_pin:
        raise LaunchRefused("immutable tape build report pin differs")
    tape_pin, _tape_path = _B._verify_tracked_file(
        checkout, commit, tape["artifact"], name="immutable target tape"
    )
    if (
        bundle["immutable_tape"] != tape_pin
        or tape["target_validity"] != validity
        or tape["sampler_seed"] != seed
        or tape["source_identity"]["manifest_sha256"]
        != core["manifest"]["sha256"]
        or tape["source_identity"]["physics_sha256"]
        != core["profile_pins"]["physics_profile_sha256"]
        or tape["source_identity"]["solver_sha256"]
        != core["profile_pins"]["solver_profile_sha256"]
    ):
        raise LaunchRefused("immutable tape canonical identity/lineage differs")
    runtime = bundle["runtime_contract"]
    if runtime != {
        "target_source": "immutable_tape",
        "reset_inverse_solve": False,
        "control_step_action_delay": [0, 0],
        "physical_ball_semantics": PHYSICAL_BALL_SEMANTICS,
        "canary_contract": "fixed_question_ablation_canary_v1",
        "actor_obs_contract": ACTOR_CONTRACT,
        "actor_width": 194,
        "critic_width": 318,
        "final_varying_ball_abi": False,
        "target_validity_is_fixed_recipe_constant": True,
        "invalid_target_columns_zero_filled": True,
        "invalid_target_columns_masked_from_reward": True,
        "target_noise_disabled": True,
        "adaptive_sigma_disabled": True,
    }:
        raise LaunchRefused("measured bundle runtime contract differs")
    claims = bundle["claims"]
    required_false = (
        "training_authorized",
    )
    required_true = (
        "diagnostic_unauthorized",
        "formal_evidence_prohibited",
        "promotion_prohibited",
        "export_prohibited",
        "deployment_prohibited",
        "hardware_prohibited",
    )
    if (
        type(claims) is not dict
        or any(claims.get(key) is not False for key in required_false)
        or any(claims.get(key) is not True for key in required_true)
    ):
        raise LaunchRefused("measured bundle diagnostic authority boundary differs")
    return {
        "bundle": normalized,
        "prepared_core_bundle": prepared_pin,
        "action_id": action_id,
        "target_recipe": recipe,
        "target_validity": validity,
        "motion": motion_pin,
        "racket_alignment_audit": alignment_pin,
        "racket_alignment": alignment,
        "measured_bank_receipt": bank_pin,
        "measured_manifest_build_report": report_pin,
        "core": core,
        "task_profile": task_pin,
        "immutable_tape_build_report": report_pin,
        "immutable_tape": tape_pin,
        "tape_row_count": tape["row_count"],
        "tape_sampler_seed": tape["sampler_seed"],
        "selected_target_lineage": tape["selected_target_lineage"],
        "mechanical_selection": selection,
        "runtime_contract": runtime,
    }


def _load_training_contract_module(checkout: Path):
    module_spec = importlib.util.spec_from_file_location(
        "_measured_vendor_v2_training_contract",
        checkout / TRAINING_CONTRACT_SOURCE,
    )
    if module_spec is None or module_spec.loader is None:
        raise LaunchRefused("cannot import dynamic-ready training contract")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise LaunchRefused(
            f"cannot import dynamic-ready training contract: {exc}"
        ) from exc
    return module


# The emitted ``policy_bootstrap`` ABI is a function of the exploration package,
# not a free parameter: ``train.py._action_ball_policy_bootstrap_schema_version``
# stamps 2 for the scalar sigma and 3 for the log sigma on every dynamic-ready
# bootstrap.  Mirroring that mapping here keeps the caller from being able to ask
# for a sigma/ABI pair the runtime can never emit.
_DYNAMIC_READY_BOOTSTRAP_SCHEMA_BY_NOISE_STD_TYPE = {"scalar": 2, "log": 3}


def _validate_policy_materialization(
    value: Any,
    *,
    checkout: Path,
    bundle: Mapping[str, Any],
    expected_noise_std_type: str = "log",
    expected_init_noise_std: float = 0.02,
) -> dict[str, Any]:
    """Bind the emitted PPO recipe to one exact exploration package.

    人话:这道门要的是"发射方声明的探索包"和"运行时真吐出来的探索包"逐字节相等,
    不是某一个写死的数。默认值仍是本文件自己那台 N1 vendor-v2 诊断用的 log/0.02,
    所以老调用点一字未改;新的四格发射器把自己 arm 合同里的 sigma 传进来。

    为什么要参数化:2026-08-05 四格把探索包定死成标准 rsl_rl 初始化 + sigma 1.0 +
    scalar(A211/C211 两格 arm 表和 arm_id 里的 ``sigma1p0`` 都是这么写的),但这里
    仍然写死 log/0.02。于是 A211 的 recipe 阶段同时被两条互斥的门夹住:先在这里被
    要求 log/0.02,二十几行后又被 ``_runtime_policy_materialization`` 要求等于
    arm 的 scalar/1.0。任何一份 recipe 都不可能同时满足,该阶段无法通过。
    """

    if expected_noise_std_type not in _DYNAMIC_READY_BOOTSTRAP_SCHEMA_BY_NOISE_STD_TYPE:
        raise LaunchRefused(
            "expected policy noise_std_type must be scalar or log"
        )
    if (
        type(expected_init_noise_std) not in (int, float)
        or not math.isfinite(float(expected_init_noise_std))
        or float(expected_init_noise_std) <= 0.0
    ):
        raise LaunchRefused(
            "expected policy init_noise_std must be one finite positive number"
        )
    expected_init_noise_std = float(expected_init_noise_std)
    expected_bootstrap_schema = _DYNAMIC_READY_BOOTSTRAP_SCHEMA_BY_NOISE_STD_TYPE[
        expected_noise_std_type
    ]
    header = _validate_policy_materialization_header(value)
    row = header["document"]
    policy_sha = header["policy_contract_sha256"]
    runner_recipe = row["action_ball_ppo_runner_recipe"]
    bootstrap = row["policy_bootstrap"]
    training_contract = _load_training_contract_module(checkout)
    dynamic = bundle["core"]["dynamic_ready"]
    try:
        expected_binding = (
            training_contract.load_action_ball_dynamic_ready_runtime_binding(
                artifact_path=str(checkout / dynamic["artifact"]["path"]),
                artifact_sha256=dynamic["artifact"]["sha256"],
                nominal_hold_receipt_path=str(
                    checkout / dynamic["nominal_hold_receipt"]["path"]
                ),
                nominal_hold_receipt_sha256=(
                    dynamic["nominal_hold_receipt"]["sha256"]
                ),
                action_order=[ACTION_ID],
                motion_paths=[str(checkout / bundle["motion"]["path"])],
            )
        )
        training_contract.validate_action_ball_policy_bootstrap(
            bootstrap, expected_action_count=1
        )
        portable_bootstrap = (
            training_contract.action_ball_policy_bootstrap_scientific_identity(
                bootstrap, repo_root=checkout
            )
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise LaunchRefused(
            f"materialized policy failed dynamic-ready validation: {exc}"
        ) from exc
    runner_payload = (
        runner_recipe.get("recipe") if type(runner_recipe) is dict else None
    )
    runner_sha = None
    if type(runner_payload) is dict:
        runner_sha = canonical_sha256(runner_payload)
    initialization = (
        bootstrap.get("initialization") if type(bootstrap) is dict else None
    )
    ready_source = (
        bootstrap.get("ready_source") if type(bootstrap) is dict else None
    )
    identity = (
        ready_source.get("identity") if type(ready_source) is dict else None
    )
    init_noise_std = (
        initialization.get("init_noise_std")
        if type(initialization) is dict
        else None
    )
    realized_noise_std = (
        initialization.get("required_realized_init_noise_std")
        if type(initialization) is dict
        else None
    )
    if (
        type(runner_recipe) is not dict
        or runner_recipe.get("sha256") != policy_sha
        or type(runner_payload) is not dict
        or runner_sha != policy_sha
        or runner_payload.get("policy_initialization") != portable_bootstrap
        or type(bootstrap) is not dict
        or bootstrap.get("schema_version") != expected_bootstrap_schema
        or bootstrap.get("action_count") != 1
        or bootstrap.get("action_order") != [ACTION_ID]
        or type(identity) is not dict
        or identity.get("binding_sha256")
        != expected_binding["binding_sha256"]
        or type(initialization) is not dict
        or initialization.get("noise_std_type") != expected_noise_std_type
        or type(init_noise_std) not in (int, float)
        or float(init_noise_std) != expected_init_noise_std
        or type(realized_noise_std) not in (int, float)
        or float(realized_noise_std) != expected_init_noise_std
    ):
        raise LaunchRefused(
            "materialized policy is not the exact %s-std dynamic-ready N1 contract"
            % expected_noise_std_type
        )
    return {
        "artifact": header["artifact"],
        "policy_contract_sha256": policy_sha,
        "dynamic_ready_binding_sha256": expected_binding["binding_sha256"],
        "noise_std_type": expected_noise_std_type,
        "configured_and_realized_init_noise_std": expected_init_noise_std,
    }


def _oracle32_acceptance_failures(
    *,
    completion: Mapping[str, Any],
    observed: int,
    exact_summary: Mapping[str, Mapping[str, Any]],
    capture: Mapping[str, Any],
    unknown: int,
    termination: Mapping[str, Any],
    projection: Mapping[str, Any],
    qdes: Mapping[str, Any],
    soft_limit: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> list[str]:
    """Return every failed preregistered oracle32 check in stable order."""

    acceptance = ORACLE32_ACCEPTANCE
    failed = []
    if completion["single_stroke"] != acceptance["single_stroke_terminal_count"]:
        failed.append("single_stroke_terminal_count")
    if observed != acceptance["exact_strike_observed_count"]:
        failed.append("exact_strike_observed_count")
    for name, summary_name in (
        ("position_error_m_strict_lt", "position"),
        ("velocity_error_mps_strict_lt", "velocity"),
        ("face_error_deg_strict_lt", "face"),
    ):
        maximum = exact_summary[summary_name]["max"]
        if maximum is None or not maximum < acceptance[name]:
            failed.append(name)
    if capture["opportunities"] != acceptance["opportunity_count"]:
        failed.append("opportunity_count")
    if capture["captures"] != acceptance["capture_count"]:
        failed.append("capture_count")
    if sum(capture["rejects"].values()) > acceptance["reject_count_max"]:
        failed.append("reject_count")
    if unknown > acceptance["unknown_attribution_count_max"]:
        failed.append("unknown_attribution_count")
    if termination["unexpected_total"] > acceptance[
        "unexpected_termination_count_max"
    ]:
        failed.append("unexpected_termination_count")
    if projection["nonfinite_sample_count"] > acceptance[
        "projection_nonfinite_sample_count_max"
    ]:
        failed.append("projection_nonfinite_sample_count")
    if qdes["preclamp_max_abs_error_rad"] > acceptance[
        "preclamp_max_abs_error_rad_max"
    ]:
        failed.append("preclamp_max_abs_error_rad")
    if projection["observed_sample_count"] != completion["control_steps"]:
        failed.append("projection_observed_sample_count")
    for channel_name in ("qdes", "actual"):
        if (
            soft_limit[channel_name]["observed_sample_count"]
            != completion["control_steps"]
        ):
            failed.append("%s_soft_limit_observed_sample_count" % channel_name)
    if (
        reference["mode"] != "metrics_only"
        or reference["available"] is not True
        or reference["sample_count"] != completion["control_steps"]
    ):
        failed.append("reference_exposure_denominator")
    return failed


def _validate_teacher_qdes_oracle(
    value: Any, *, spec: Mapping[str, Any], claim: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the fresh oracle output against its complete launch claim."""

    pin, path = _validate_external_pin(value, name="teacher-qdes oracle")
    expected_path = Path(spec["namespace"]) / _teacher_oracle_filename(
        spec["stage"]
    )
    if path != expected_path:
        raise LaunchRefused("teacher-qdes oracle path differs from output contract")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="teacher-qdes oracle")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("teacher-qdes oracle must be canonical JSON plus newline")
    row = _exact_dict(
        document,
        (
            "schema_version", "kind", "diagnostic_unauthorized", "bindings",
            "completion", "phase_by_termination", "exact_strike",
            "capture_rejection", "measurement_contract", "safety_exposure",
            "teacher_qdes", "episodes",
        ),
        name="teacher-qdes oracle",
    )
    if (
        row["schema_version"] != 2
        or row["kind"] != "action_ball_teacher_qdes_dynamic_oracle_v2"
        or row["diagnostic_unauthorized"] is not True
    ):
        raise LaunchRefused("teacher-qdes oracle schema/kind/authorization differs")
    bindings = _exact_dict(
        row["bindings"],
        (
            "source_sha256", "task_sha256", "hard_contract_sha256",
            "reward_sha256", "policy_sha256", "policy_contract_sha256",
            "dynamic_ready_sha256", "dynamic_ready_artifact_sha256",
            "dynamic_ready_nominal_hold_sha256", "manifest_sha256",
            "motion_sha256", "tape_file_sha256", "tape_canonical_sha256",
            "tape_base_question_sha256", "tape_target_producer_sha256",
            "tape_target_column_sha256",
        ),
        name="teacher-qdes oracle bindings",
    )
    for name, digest in bindings.items():
        _B._sha256(digest, name="teacher-qdes oracle %s" % name)
    checkout = Path(spec["source"]["checkout"])
    root = checkout / _B.WBT_RELATIVE / "logs/rsl_rl" / EXPERIMENT_NAME
    suffix = "_%s-DIAGNOSTIC_UNAUTHORIZED" % Path(spec["namespace"]).name
    candidates = (
        [] if not root.is_dir()
        else [candidate for candidate in root.iterdir() if candidate.name.endswith(suffix)]
    )
    if (
        len(candidates) != 1
        or not stat.S_ISDIR(candidates[0].lstat().st_mode)
        or candidates[0].resolve(strict=True) != candidates[0]
    ):
        raise LaunchRefused("teacher-qdes oracle has no unique hard contract")
    contracts = [candidates[0] / "params/training_contract.json"]
    _B._stable_regular_file(contracts[0], name="teacher-qdes hard contract")
    hard_contract = _B._strict_json_bytes(
        contracts[0].read_bytes(), name="teacher-qdes hard contract"
    )
    try:
        _load_training_contract_module(
            checkout
        ).validate_schema3_contract_structure(hard_contract)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchRefused(
            "teacher-qdes hard contract failed complete schema-3 validation: %s"
            % exc
        ) from exc
    bundle = claim["bundle"]
    sources = claim["runtime_sources"]
    policy = claim["materialization_inputs"]["policy"]
    expected = {
        "source_sha256": sources["training entrypoint"]["sha256"],
        "task_sha256": sources["VendorV2 N1 task leaf"]["sha256"],
        "hard_contract_sha256": _B.sha256_file(contracts[0]),
        "reward_sha256": spec["expected_effective_reward_recipe_sha256"],
        "policy_sha256": spec["policy_contract_sha256"],
        "policy_contract_sha256": spec["policy_contract_sha256"],
        "dynamic_ready_sha256": policy["dynamic_ready_binding_sha256"],
        "dynamic_ready_artifact_sha256": bundle["core"]["dynamic_ready"]["artifact"]["sha256"],
        "dynamic_ready_nominal_hold_sha256": bundle["core"]["dynamic_ready"]["nominal_hold_receipt"]["sha256"],
        "manifest_sha256": bundle["core"]["manifest"]["sha256"],
        "motion_sha256": bundle["motion"]["sha256"],
        "tape_file_sha256": bundle["immutable_tape"]["sha256"],
        "tape_canonical_sha256": bundle["selected_target_lineage"]["tape_canonical_sha256"],
        "tape_base_question_sha256": bundle["selected_target_lineage"]["base_question_sha256"],
        "tape_target_producer_sha256": bundle["selected_target_lineage"]["target_producer_sha256"],
        "tape_target_column_sha256": bundle["selected_target_lineage"]["target_column_sha256"],
    }
    if any(bindings[name] != digest for name, digest in expected.items()):
        raise LaunchRefused("teacher-qdes oracle lineage bindings differ from claim")
    try:
        hard_action_ball = hard_contract["action_ball_training"]
        hard_preflight = hard_action_ball["preflight"]
        hard_dynamic = hard_action_ball["policy_bootstrap"]["ready_source"][
            "identity"
        ]
        hard_dynamic_rows = hard_dynamic["rows"]
        hard_tape = hard_action_ball["runtime"]["target_provider"][
            "immutable_tape"
        ]
        hard_projection = hard_action_ball["qdes_projection_penalty"]
        hard_reference_payload = hard_action_ball["runtime"]["reference_guard"][
            "contract_payload"
        ]
        hard_reference_reasons = hard_reference_payload["reference_reasons"]
        hard_reasons = hard_reference_payload["hard_reasons"]
        hard_reference_counter_names = hard_reference_payload["counter_names"]
        hard_reference_counter_schema_sha256 = hard_reference_payload[
            "counter_schema_sha256"
        ]
        if (
            type(hard_reference_reasons) is not list
            or type(hard_reasons) is not list
            or type(hard_reference_counter_names) is not list
            or any(
                type(name) is not str or not name
                for name in (
                    *hard_reference_reasons,
                    *hard_reasons,
                    *hard_reference_counter_names,
                )
            )
            or len(set(hard_reference_counter_names))
            != len(hard_reference_counter_names)
        ):
            raise KeyError("oracle requires exact reference-guard schema")
        _B._sha256(
            hard_reference_counter_schema_sha256,
            name="teacher-qdes reference counter schema",
        )
        if (
            type(hard_projection) is not dict
            or hard_projection.get("schema_version") != 2
            or hard_projection.get("weight_independent_exposure") is not True
            or hard_projection.get("exposure_denominator")
            != "control_step_observed_sample_count"
            or hard_projection.get("hypothetical_unweighted_penalty")
            != "projection_penalty_value_sum"
            or hard_projection.get("per_joint_exposure") is not True
        ):
            raise KeyError("oracle requires weight-independent projection exposure")
        if (
            spec["stage"] == "oracle32"
            and hard_action_ball["runtime"]["reference_guard"].get("mode")
            != "metrics_only"
        ):
            raise KeyError("oracle32 requires metrics-only reference exposure")
        hard_lineage = hard_tape["target_lineage"]
        hard_motion = hard_contract["motion_clips"]
        if (
            type(hard_dynamic_rows) is not list
            or len(hard_dynamic_rows) != 1
            or type(hard_motion) is not list
            or len(hard_motion) != 1
        ):
            raise KeyError("oracle requires one dynamic-ready row and one motion")
        hard_expected = {
            "reward_sha256": hard_action_ball["effective_reward_recipe_sha256"],
            "policy_sha256": hard_contract["action_ball_ppo_runner_recipe"][
                "sha256"
            ],
            "policy_contract_sha256": hard_preflight["policy_contract_sha256"],
            "dynamic_ready_sha256": hard_dynamic["binding_sha256"],
            "dynamic_ready_artifact_sha256": hard_dynamic_rows[0]["artifact"][
                "sha256"
            ],
            "dynamic_ready_nominal_hold_sha256": hard_dynamic_rows[0][
                "nominal_hold_receipt"
            ]["sha256"],
            "manifest_sha256": hard_preflight["manifest"]["file_sha256"],
            "motion_sha256": hard_motion[0]["sha256"],
            "tape_file_sha256": hard_tape["file_sha256"],
            "tape_canonical_sha256": hard_tape["canonical_sha256"],
            "tape_base_question_sha256": hard_tape["base_question_sha256"],
            "tape_target_producer_sha256": hard_lineage[
                "target_producer_sha256"
            ],
            "tape_target_column_sha256": hard_lineage["target_column_sha256"],
        }
    except (KeyError, TypeError) as exc:
        raise LaunchRefused(
            "teacher-qdes hard contract lacks exact oracle lineage"
        ) from exc
    if any(bindings[name] != digest for name, digest in hard_expected.items()):
        raise LaunchRefused("teacher-qdes hard-contract lineage differs")

    completion = _exact_dict(
        row["completion"],
        (
            "requested", "terminal", "single_stroke",
            "exact_strike_observed_nonterminal",
            "pre_strike_or_same_step_unknown", "control_steps",
        ),
        name="teacher-qdes oracle completion",
    )
    expected_episodes = _teacher_oracle_episodes(spec["stage"])
    episodes = row["episodes"]
    if (
        type(episodes) is not list
        or completion["requested"] != expected_episodes
        or completion["terminal"] != expected_episodes
        or len(episodes) != expected_episodes
        or type(completion["control_steps"]) is not int
        or completion["control_steps"] < expected_episodes
        or completion["control_steps"]
        != sum(
            episode.get("control_steps", 0)
            for episode in episodes
            if type(episode) is dict
        )
    ):
        raise LaunchRefused(
            "teacher-qdes oracle did not complete exactly %d episodes"
            % expected_episodes
        )
    phases = ("post_strike", "pre_strike_or_same_step_unknown")
    observed = unknown = 0
    exact_records = []
    derived: dict[str, dict[str, int]] = {phase: {} for phase in phases}
    for index, episode in enumerate(episodes):
        episode = _exact_dict(
            episode,
            ("episode", "control_steps", "terminal_phase", "termination_reasons", "exact_strike"),
            name="teacher-qdes oracle episode",
        )
        phase = episode["terminal_phase"]
        reasons = episode["termination_reasons"]
        exact = episode["exact_strike"]
        if (
            episode["episode"] != index
            or type(episode["control_steps"]) is not int
            or episode["control_steps"] < 1
            or phase not in phases
            or type(reasons) is not list
            or not reasons
            or any(type(reason) is not str or not reason for reason in reasons)
            or len(set(reasons)) != len(reasons)
            or (phase == "post_strike") != (type(exact) is dict)
        ):
            raise LaunchRefused("teacher-qdes oracle episode attribution differs")
        if exact is not None:
            exact = _exact_dict(
                exact,
                ("position_error_m", "velocity_error_mps", "face_error_deg", "captured"),
                name="teacher-qdes oracle exact strike",
            )
            if (
                any(
                    type(exact[name]) not in (int, float)
                    or not math.isfinite(float(exact[name]))
                    or exact[name] < 0
                    for name in ("position_error_m", "velocity_error_mps", "face_error_deg")
                )
                or type(exact["captured"]) is not bool
            ):
                raise LaunchRefused("teacher-qdes oracle exact strike values differ")
            exact_records.append(exact)
        observed += int(exact is not None)
        unknown += int(phase == "pre_strike_or_same_step_unknown")
        for reason in reasons:
            derived[phase][reason] = derived[phase].get(reason, 0) + 1
    reported_phases = row["phase_by_termination"]
    if type(reported_phases) is not dict or set(reported_phases) != set(phases):
        raise LaunchRefused("teacher-qdes oracle phase ledger keys differ")
    term_keys = None
    for phase in phases:
        phase_row = reported_phases[phase]
        if (
            type(phase_row) is not dict
            or any(
                type(name) is not str or type(count) is not int or count < 0
                for name, count in phase_row.items()
            )
        ):
            raise LaunchRefused("teacher-qdes oracle phase ledger values differ")
        term_keys = set(phase_row) if term_keys is None else term_keys
        if set(phase_row) != term_keys or any(
            count != derived[phase].get(name, 0)
            for name, count in phase_row.items()
        ):
            raise LaunchRefused("teacher-qdes oracle phase ledger differs")
    if (
        completion["exact_strike_observed_nonterminal"] != observed
        or completion["pre_strike_or_same_step_unknown"] != unknown
        or completion["single_stroke"]
        != sum(
            "action_ball_single_stroke_complete" in episode["termination_reasons"]
            for episode in episodes
        )
    ):
        raise LaunchRefused("teacher-qdes oracle phase/completion ledger differs")
    exact_summary = _exact_dict(
        row["exact_strike"], ("position", "velocity", "face"),
        name="teacher-qdes oracle exact summary",
    )
    for output_name, record_name in (
        ("position", "position_error_m"),
        ("velocity", "velocity_error_mps"),
        ("face", "face_error_deg"),
    ):
        summary = _exact_dict(
            exact_summary[output_name], ("denominator", "values", "mean", "max"),
            name="teacher-qdes oracle %s summary" % output_name,
        )
        values = [record[record_name] for record in exact_records]
        if (
            summary["denominator"] != observed
            or summary["values"] != values
            or summary["mean"] != (None if not values else sum(values) / len(values))
            or summary["max"] != (None if not values else max(values))
        ):
            raise LaunchRefused("teacher-qdes oracle exact summary differs")
    qdes = row["teacher_qdes"]
    capture = row["capture_rejection"]
    if (
        type(qdes) is not dict
        or set(qdes) != set(TEACHER_QDES_RECEIPT_KEYS)
        or qdes.get("teleport_used") is not False
        or qdes.get("control_step_denominator") != completion["control_steps"]
        or any(
            type(qdes.get(name)) not in (int, float)
            or not math.isfinite(float(qdes[name]))
            or qdes[name] < 0
            for name in ("preclamp_max_abs_error_rad", "raw_action_max_abs")
        )
        # 人话:等待阶段发的是"保持 q_des",揭示后发老师姿态;两个计数必须
        # 恰好把总步数分完,收据才说得清哪一步是被什么驱动的。
        or any(
            type(qdes.get(name)) is not int or qdes[name] < 0
            for name in (
                "wait_hold_command_steps", "teacher_reference_command_steps",
            )
        )
        or qdes["wait_hold_command_steps"]
        + qdes["teacher_reference_command_steps"]
        != completion["control_steps"]
        or type(capture) is not dict
        or set(capture) != set(TEACHER_CAPTURE_REJECTION_KEYS)
        or capture.get("conserved") is not True
        or type(capture.get("opportunities")) is not int
        or type(capture.get("captures")) is not int
        or type(capture.get("rejects")) is not dict
        or set(capture["rejects"]) != set(TEACHER_ORACLE_REJECT_KEYS)
        or any(
            type(name) is not str or type(count) is not int or count < 0
            for name, count in capture["rejects"].items()
        )
        or capture["opportunities"] < 0
        or capture["captures"] < 0
        or capture["captures"] + sum(capture["rejects"].values())
        != capture["opportunities"]
        or capture["captures"]
        != sum(bool(record["captured"]) for record in exact_records)
        or observed > capture["opportunities"]
        or capture["opportunities"] - observed > unknown
    ):
        raise LaunchRefused("teacher-qdes oracle qdes/capture ledger differs")

    measurement = _exact_dict(
        row["measurement_contract"],
        (
            "single_stroke_requested",
            "exact_strike_thresholds",
            "projection_exposure_is_weight_independent",
        ),
        name="teacher-qdes oracle measurement contract",
    )
    thresholds = _exact_dict(
        measurement["exact_strike_thresholds"],
        (
            "position_error_m_strict_lt",
            "velocity_error_mps_strict_lt",
            "face_error_deg_strict_lt",
        ),
        name="teacher-qdes oracle exact-strike thresholds",
    )
    if (
        measurement["single_stroke_requested"] != expected_episodes
        or measurement["projection_exposure_is_weight_independent"] is not True
        or thresholds
        != {
            "position_error_m_strict_lt": 0.075,
            "velocity_error_mps_strict_lt": 0.5,
            "face_error_deg_strict_lt": 15.0,
        }
    ):
        raise LaunchRefused("teacher-qdes oracle measurement contract differs")

    exposure = _exact_dict(
        row["safety_exposure"],
        ("projection", "soft_limit", "reference_guard", "termination"),
        name="teacher-qdes oracle safety exposure",
    )
    projection = _exact_dict(
        exposure["projection"],
        (
            "observed_sample_count",
            "projected_sample_count",
            "nonfinite_sample_count",
            "hypothetical_unweighted_penalty_sum",
            "max_normalized_projection_distance",
            "mean_normalized_projection_distance",
            "joints",
        ),
        name="teacher-qdes oracle projection exposure",
    )
    for name in (
        "observed_sample_count",
        "projected_sample_count",
        "nonfinite_sample_count",
    ):
        if type(projection[name]) is not int or projection[name] < 0:
            raise LaunchRefused("teacher-qdes oracle projection counts differ")
    if (
        projection["projected_sample_count"] > projection["observed_sample_count"]
        or projection["nonfinite_sample_count"]
        > projection["projected_sample_count"]
    ):
        raise LaunchRefused("teacher-qdes oracle projection counts differ")
    for name in (
        "hypothetical_unweighted_penalty_sum",
        "max_normalized_projection_distance",
        "mean_normalized_projection_distance",
    ):
        if (
            type(projection[name]) not in (int, float)
            or not math.isfinite(float(projection[name]))
            or projection[name] < 0
        ):
            raise LaunchRefused("teacher-qdes oracle projection values differ")
    joints = projection["joints"]
    if type(joints) is not list or len(joints) != 31:
        raise LaunchRefused("teacher-qdes oracle projection joint ledger differs")
    for joint_index, joint in enumerate(joints):
        joint = _exact_dict(
            joint,
            (
                "joint_index",
                "trigger_count",
                "lower_count",
                "upper_count",
                "mean_normalized_distance",
                "max_normalized_distance",
            ),
            name="teacher-qdes oracle projection joint",
        )
        if (
            joint["joint_index"] != joint_index
            or any(
                type(joint[name]) is not int or joint[name] < 0
                for name in ("trigger_count", "lower_count", "upper_count")
            )
            or joint["trigger_count"] > projection["observed_sample_count"]
            or joint["lower_count"] + joint["upper_count"]
            > joint["trigger_count"]
            or any(
                type(joint[name]) not in (int, float)
                or not math.isfinite(float(joint[name]))
                or joint[name] < 0
                for name in (
                    "mean_normalized_distance",
                    "max_normalized_distance",
                )
            )
        ):
            raise LaunchRefused("teacher-qdes oracle projection joint ledger differs")

    soft_limit = _exact_dict(
        exposure["soft_limit"],
        ("qdes", "actual"),
        name="teacher-qdes oracle soft-limit exposure",
    )
    for channel_name in ("qdes", "actual"):
        channel = _exact_dict(
            soft_limit[channel_name],
            (
                "observed_sample_count",
                "intrusion_sample_count",
                "intrusion_joint_count",
                "reward_enabled_sample_count",
                "max_intrusion_depth_frac",
                "hypothetical_unweighted_barrier_sum",
            ),
            name="teacher-qdes oracle %s soft-limit exposure" % channel_name,
        )
        if any(
            type(channel[name]) is not int or channel[name] < 0
            for name in (
                "observed_sample_count",
                "intrusion_sample_count",
                "intrusion_joint_count",
                "reward_enabled_sample_count",
            )
        ) or any(
            type(channel[name]) not in (int, float)
            or not math.isfinite(float(channel[name]))
            or channel[name] < 0
            for name in (
                "max_intrusion_depth_frac",
                "hypothetical_unweighted_barrier_sum",
            )
        ):
            raise LaunchRefused("teacher-qdes oracle soft-limit values differ")
        if (
            channel["intrusion_sample_count"] > channel["observed_sample_count"]
            or channel["reward_enabled_sample_count"]
            > channel["observed_sample_count"]
        ):
            raise LaunchRefused("teacher-qdes oracle soft-limit counts differ")

    reference = _exact_dict(
        exposure["reference_guard"],
        (
            "mode",
            "available",
            "counter_schema_sha256",
            "counter_names",
            "counters",
            "sample_count",
            "union_count",
            "reference_only_count",
            "reference_and_hard_count",
        ),
        name="teacher-qdes oracle reference exposure",
    )
    if (
        type(reference["mode"]) is not str
        or type(reference["available"]) is not bool
        or reference["counter_schema_sha256"]
        != hard_reference_counter_schema_sha256
        or reference["counter_names"] != hard_reference_counter_names
    ):
        raise LaunchRefused("teacher-qdes oracle reference exposure differs")
    if reference["available"]:
        counters = reference["counters"]
        if (
            type(counters) is not dict
            or set(counters) != set(hard_reference_counter_names)
            or any(type(value) is not int or value < 0 for value in counters.values())
            or any(
            type(reference[name]) is not int or reference[name] < 0
            for name in (
                "sample_count",
                "union_count",
                "reference_only_count",
                "reference_and_hard_count",
            )
            )
            or reference["union_count"] != (
            reference["reference_only_count"]
            + reference["reference_and_hard_count"]
            )
            or reference["sample_count"]
            != counters["reference_guard_sample_count"]
            or reference["union_count"]
            != counters["reference_guard_union_count"]
            or reference["reference_only_count"]
            != counters["reference_guard_reference_only_count"]
            or reference["reference_and_hard_count"]
            != counters["reference_guard_reference_and_hard_count"]
        ):
            raise LaunchRefused("teacher-qdes oracle reference ledger differs")
    elif reference["counters"] is not None or any(
        reference[name] is not None for name in (
            "sample_count",
            "union_count",
            "reference_only_count",
            "reference_and_hard_count",
        )
    ):
        raise LaunchRefused("teacher-qdes oracle unavailable reference ledger differs")

    termination = _exact_dict(
        exposure["termination"],
        (
            "active_terms",
            "allowed_terminal_reason",
            "unexpected_total",
            "unexpected_by_reason",
        ),
        name="teacher-qdes oracle termination exposure",
    )
    unexpected = termination["unexpected_by_reason"]
    expected_active_terms = set(
        hard_reference_reasons
        + hard_reasons
        + ["time_out", "action_ball_single_stroke_complete"]
    )
    if (
        type(termination["active_terms"]) is not list
        or len(termination["active_terms"]) != len(set(termination["active_terms"]))
        or set(termination["active_terms"]) != expected_active_terms
        or term_keys != expected_active_terms
        or set(unexpected) != expected_active_terms - {
            "action_ball_single_stroke_complete"
        }
        or
        termination["allowed_terminal_reason"]
        != "action_ball_single_stroke_complete"
        or type(termination["unexpected_total"]) is not int
        or termination["unexpected_total"] < 0
        or type(unexpected) is not dict
        or any(
            type(name) is not str or type(count) is not int or count < 0
            for name, count in unexpected.items()
        )
        or termination["unexpected_total"] != sum(unexpected.values())
        or termination["unexpected_total"]
        != sum(
            count
            for phase in phases
            for name, count in reported_phases[phase].items()
            if name != "action_ball_single_stroke_complete"
        )
    ):
        raise LaunchRefused("teacher-qdes oracle termination exposure differs")

    verdict = None
    if spec["stage"] == "oracle32":
        failed = _oracle32_acceptance_failures(
            completion=completion,
            observed=observed,
            exact_summary=exact_summary,
            capture=capture,
            unknown=unknown,
            termination=termination,
            projection=projection,
            qdes=qdes,
            soft_limit=soft_limit,
            reference=reference,
        )
        verdict = {
            "accepted": not failed,
            "failed_checks": failed,
            "acceptance": ORACLE32_ACCEPTANCE,
        }
        if failed:
            raise LaunchRefused(
                "teacher-qdes oracle32 acceptance failed: %s" % ",".join(failed)
            )
    return {
        "artifact": pin,
        "hard_contract": {"path": str(contracts[0]), "sha256": bindings["hard_contract_sha256"]},
        "completion": completion,
        "bindings": bindings,
        "safety_exposure": exposure,
        "oracle32_verdict": verdict,
    }


def _check_rsl_namespace(checkout: Path, namespace_name: str) -> None:
    root = checkout / _B.WBT_RELATIVE / "logs/rsl_rl" / EXPERIMENT_NAME
    if not root.exists():
        return
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise LaunchRefused("RSL experiment root is not a real directory")
    suffix = "_%s-DIAGNOSTIC_UNAUTHORIZED" % namespace_name
    spent = [child.name for child in root.iterdir() if child.name.endswith(suffix)]
    if spent:
        raise LaunchRefused("trainer run_name is already spent: %s" % sorted(spent)[0])


def _training_argv(spec: dict[str, Any], bundle: dict[str, Any]) -> list[str]:
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / _B.WBT_RELATIVE
    core = bundle["core"]
    dynamic = core["dynamic_ready"]
    motion = checkout / bundle["motion"]["path"]
    manifest = checkout / core["manifest"]["path"]
    tape = checkout / bundle["immutable_tape"]["path"]
    validity = json.dumps(spec["target_validity_mask"], separators=(",", ":"))
    list_one = json.dumps([str(motion)], separators=(",", ":"))
    action_one = json.dumps([spec["action_id"]], separators=(",", ":"))
    policy_sha = (
        RECIPE_SENTINEL_POLICY_SHA256
        if spec["stage"] in ("materialize", "recipe")
        else spec["policy_contract_sha256"]
    )
    argv = [
        spec["source"]["isaac_python"],
        str(wbt / "scripts/train.py"),
        "task=%s" % TASK_PROFILE_ID,
        "algo=ppo",
        "algo.policy.init_noise_std=0.02",
        POLICY_NOISE_STD_OVERRIDE,
        "headless=true",
        "logger=tensorboard",
        "video=false",
        "device=cuda:0",
        "seed=%d" % spec["seed"],
        "num_envs=%d" % spec["num_envs"],
        "max_iterations=%d" % spec["max_iterations"],
        "algo.runner.save_interval=%d" % spec["save_interval"],
        "run_name=%s-DIAGNOSTIC_UNAUTHORIZED" % Path(spec["namespace"]).name,
        "task.experiment_name=%s" % EXPERIMENT_NAME,
        "task.actor_obs_contract=%s" % ACTOR_CONTRACT,
        "action_ball_dynamic_ready_bootstrap=true",
        "action_ball_dynamic_ready_artifact_path=%s"
        % (checkout / dynamic["artifact"]["path"]),
        "action_ball_dynamic_ready_artifact_sha256=%s"
        % dynamic["artifact"]["sha256"],
        "action_ball_dynamic_ready_nominal_receipt_path=%s"
        % (checkout / dynamic["nominal_hold_receipt"]["path"]),
        "action_ball_dynamic_ready_nominal_receipt_sha256=%s"
        % dynamic["nominal_hold_receipt"]["sha256"],
        "motion_file=%s" % list_one,
        "task.racket.clip_names=%s" % action_one,
        "task.racket.action_ball_manifest_path=%s" % manifest,
        "task.racket.action_ball_manifest_sha256=%s" % core["manifest"]["sha256"],
        "task.racket.action_ball_policy_contract_sha256=%s"
        % policy_sha,
        "task.racket.action_ball_diagnostic_unauthorized=true",
        "task.motion.action_ball_diagnostic_split_ready_teacher=true",
        "task.racket.action_ball_seed=%d" % spec["seed"],
        "task.racket.action_ball_target_source=immutable_tape",
        "task.racket.action_ball_immutable_tape_path=%s" % tape,
        "task.racket.action_ball_immutable_tape_sha256=%s"
        % bundle["immutable_tape"]["sha256"],
        "task.racket.action_ball_target_recipe=%s" % spec["target_recipe"],
        "task.racket.action_ball_target_validity_mask=%s" % validity,
        "task.racket.action_ball_target_observation_noise=false",
        "task.racket.adaptive_sigma=false",
        "task.racket.adaptive_sigma_monotonic=false",
        "task.racket.adaptive_sigma_normal=false",
        "task.racket.target_noise_white=0.0",
        "task.racket.target_noise_ar1_sigma=0.0",
        "task.actions.control_step_action_delay_min=0",
        "task.actions.control_step_action_delay_max=0",
        "task.push.enable=false",
        *("~task.push.%s" % field for field in DISABLED_PUSH_DORMANT_FIELDS),
        "task.physical_ball=false",
        "task.racket.virtual_ball=true",
        "task.racket.action_ball_pool_refill_rows=1",
        "task.racket.question_bank=",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
    ]
    if spec["stage"] == "materialize":
        argv.extend(
            [
                "+n1_vendor_sigma_profile=%s"
                % REWARD_MATERIALIZATION_PROFILE,
                "+action_ball_effective_reward_recipe_output_path=%s"
                % (Path(spec["namespace"]) / REWARD_RECIPE_FILENAME),
            ]
        )
    else:
        argv.append(
            "expected_effective_reward_recipe_sha256=%s"
            % spec["expected_effective_reward_recipe_sha256"]
        )
        if spec["stage"] == "recipe":
            argv.append(
                "action_ball_policy_recipe_output_path=%s"
                % (Path(spec["namespace"]) / POLICY_RECIPE_FILENAME)
            )
        elif _is_teacher_oracle_stage(spec["stage"]):
            argv.extend(
                [
                    "+action_ball_teacher_qdes_oracle_output_path=%s"
                    % (
                        Path(spec["namespace"])
                        / _teacher_oracle_filename(spec["stage"])
                    ),
                    "+action_ball_teacher_qdes_oracle_episodes=%d"
                    % _teacher_oracle_episodes(spec["stage"]),
                ]
            )
            if spec["stage"] == "oracle32":
                argv.append(
                    "+task.racket.reference_guard_mode=metrics_only"
                )
    return argv


def _output_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec["stage"] == "materialize":
        return {
            "ppo_update_count": 0,
            "effective_reward_recipe": str(
                Path(spec["namespace"]) / REWARD_RECIPE_FILENAME
            ),
            "policy_recipe": None,
            "teacher_qdes_oracle": None,
            "boot_marker": "ACTION_BALL_EFFECTIVE_REWARD_RECIPE_MATERIALIZED_JSON",
        }
    if spec["stage"] == "recipe":
        return {
            "ppo_update_count": 0,
            "effective_reward_recipe": None,
            "policy_recipe": str(
                Path(spec["namespace"]) / POLICY_RECIPE_FILENAME
            ),
            "teacher_qdes_oracle": None,
            "boot_marker": "ACTION_BALL_POLICY_RECIPE_MATERIALIZED",
        }
    if _is_teacher_oracle_stage(spec["stage"]):
        return {
            "ppo_update_count": 0,
            "effective_reward_recipe": None,
            "policy_recipe": None,
            "teacher_qdes_oracle": str(
                Path(spec["namespace"])
                / _teacher_oracle_filename(spec["stage"])
            ),
            "boot_marker": "ACTION_BALL_TEACHER_QDES_ORACLE_COMPLETE_JSON",
            "teacher_qdes_oracle_episodes": _teacher_oracle_episodes(
                spec["stage"]
            ),
            "teacher_qdes_oracle_acceptance": (
                copy.deepcopy(ORACLE32_ACCEPTANCE)
                if spec["stage"] == "oracle32"
                else None
            ),
        }
    return {
        "ppo_update_count": spec["max_iterations"],
        "effective_reward_recipe": None,
        "policy_recipe": None,
        "teacher_qdes_oracle": None,
        "boot_marker": "Learning iteration",
    }


_ADMISSION = _A.VendorV2GPUAdmission(
    base=_B,
    schema_version=SCHEMA_VERSION,
    claim_kind=CLAIM_KIND,
    experiment_name=EXPERIMENT_NAME,
    colocation_spec_key=VENDOR_V2_COLOCATION_SPEC_KEY,
    physical_ball_semantics=PHYSICAL_BALL_SEMANTICS,
    runtime_source_paths=RUNTIME_SOURCE_PATHS,
    launcher_source=LAUNCHER_SOURCE,
    admission_source=ADMISSION_SOURCE,
    exact_group_source=EXACT_GROUP_SOURCE,
    exact_group=_G,
    canonical_sha256=canonical_sha256,
    exact_dict=_exact_dict,
    validate_spec=_validate_spec,
    output_contract=_output_contract,
    training_argv=_training_argv,
)
_open_gpu_shared_lock = _ADMISSION._open_gpu_shared_lock
_lock_gpu_admission = _ADMISSION._lock_gpu_admission
_unlock_gpu_admission = _ADMISSION._unlock_gpu_admission
_proc_starttime = _ADMISSION._proc_starttime
_proc_environment = _ADMISSION._proc_environment
_proc_executable = _ADMISSION._proc_executable
_proc_cmdline = _ADMISSION._proc_cmdline
_stable_canonical_json = _ADMISSION._stable_canonical_json
_validate_namespace_claim = _ADMISSION._validate_namespace_claim
_validate_runtime_gpu_process = _ADMISSION._validate_runtime_gpu_process
_query_gpu_processes = _ADMISSION._query_gpu_processes
_live_runtime_handoff = _ADMISSION._live_runtime_handoff
_live_reservations = _ADMISSION._live_reservations
_reservation_document = _ADMISSION._reservation_document
_runtime_namespace_receipt = _ADMISSION._runtime_namespace_receipt
_cleanup_post_boot_admission_failure = (
    _ADMISSION._cleanup_post_boot_admission_failure
)


def _verify_gpu_admission(
    spec: Mapping[str, Any],
    *,
    phase: str,
    current_namespace: Path | None,
    require_current_compute: bool = False,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    return _ADMISSION._verify_gpu_admission(
        spec,
        phase=phase,
        current_namespace=current_namespace,
        require_current_compute=require_current_compute,
        proc_root=proc_root,
        query_gpu_processes=_query_gpu_processes,
        validate_runtime_gpu_process=_validate_runtime_gpu_process,
        live_reservations=_live_reservations,
    )


def build_plan(spec_path: Path) -> dict[str, Any]:
    spec_path = _B._absolute_path(str(spec_path), name="--spec", must_exist=True)
    _B._stable_regular_file(spec_path, name="launch spec")
    raw = spec_path.read_bytes()
    document = _B._strict_json_bytes(raw, name="launch spec")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("launch spec must be canonical JSON plus newline")
    spec = _validate_spec(document)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    source = _B._verify_clean_source(checkout, commit)
    sources = _runtime_sources(checkout, commit)
    assets = _B._validate_runtime_asset_environment()
    bundle = _validate_bundle(
        checkout,
        commit,
        spec["bundle"],
        action_id=spec["action_id"],
        recipe=spec["target_recipe"],
        seed=spec["seed"],
    )
    materialization_inputs = {
        "reward": (
            None
            if spec["stage"] == "materialize"
            else _validate_reward_materialization(
                spec["reward_materialization"]
            )
        ),
        "policy": (
            None
            if spec["stage"] in ("materialize", "recipe")
            else _validate_policy_materialization(
                spec["policy_materialization"],
                checkout=checkout,
                bundle=bundle,
            )
        ),
    }
    if (
        materialization_inputs["reward"] is not None
        and materialization_inputs["reward"][
            "effective_reward_recipe_sha256"
        ]
        != spec["expected_effective_reward_recipe_sha256"]
    ):
        raise LaunchRefused("reward materialization drifted after spec validation")
    if (
        materialization_inputs["policy"] is not None
        and materialization_inputs["policy"]["policy_contract_sha256"]
        != spec["policy_contract_sha256"]
    ):
        raise LaunchRefused("policy materialization drifted after spec validation")
    _check_rsl_namespace(checkout, Path(spec["namespace"]).name)
    argv = _training_argv(spec, bundle)
    output_contract = _output_contract(spec)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "diagnostic_unauthorized": True,
        "formal_evidence_prohibited": True,
        "promotion_prohibited": True,
        "resume_prohibited": True,
        "export_prohibited": True,
        "deployment_prohibited": True,
        "hardware_prohibited": True,
        "single_gpu": True,
        "max_compute_pids_on_physical_gpu": MAX_VENDOR_V2_COMPUTE_PIDS,
        "minimum_free_memory_mib": MIN_VENDOR_V2_FREE_MEMORY_MIB,
        "gpu_default_empty": not spec[VENDOR_V2_COLOCATION_SPEC_KEY],
        "vendor_v2_colocation_opt_in": spec[VENDOR_V2_COLOCATION_SPEC_KEY],
        "fresh_only": True,
        "reward_materialization_only": spec["stage"] == "materialize",
        "policy_recipe_materialization_only": spec["stage"] == "recipe",
        "teacher_qdes_oracle_only": _is_teacher_oracle_stage(spec["stage"]),
        "ppo_updates_authorized": output_contract["ppo_update_count"],
        "control_step_action_delay": 0,
        "reset_inverse_solve": False,
        "physical_ball_semantics": PHYSICAL_BALL_SEMANTICS,
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
        "source": source,
        "runtime_sources": sources,
        "runtime_assets": assets,
        "bundle": bundle,
        "materialization_inputs": materialization_inputs,
        "output_contract": output_contract,
        "boot_marker": output_contract["boot_marker"],
        "training_argv": argv,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(payload),
        "canonical_payload": payload,
    }


def _internal_exec(claim_path: Path, claim_sha: str, lock_fd: int) -> int:
    path = _B._absolute_path(str(claim_path), name="internal claim", must_exist=True)
    raw = path.read_bytes()
    outer = _B._strict_json_bytes(raw, name="internal claim")
    if raw != _B._canonical_bytes(outer) + b"\n":
        raise LaunchRefused("internal claim is not canonical")
    outer = _exact_dict(
        outer,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="internal claim",
    )
    if (
        outer["schema_version"] != SCHEMA_VERSION
        or outer["kind"] != CLAIM_KIND
        or outer["launch_claim_sha256"] != claim_sha
        or canonical_sha256(outer["canonical_payload"]) != claim_sha
    ):
        raise LaunchRefused("internal claim digest differs")
    payload = outer["canonical_payload"]
    spec = _validate_spec(payload["spec"], claimed=True)
    checkout = Path(spec["source"]["checkout"])
    _B._verify_clean_source(checkout, spec["source"]["commit_sha"])
    sources = _runtime_sources(checkout, spec["source"]["commit_sha"])
    if sources != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted")
    assets = _B._validate_runtime_asset_claim(payload["runtime_assets"])
    bundle = _validate_bundle(
        checkout,
        spec["source"]["commit_sha"],
        spec["bundle"],
        action_id=spec["action_id"],
        recipe=spec["target_recipe"],
        seed=spec["seed"],
    )
    materialization_inputs = {
        "reward": (
            None
            if spec["stage"] == "materialize"
            else _validate_reward_materialization(
                spec["reward_materialization"]
            )
        ),
        "policy": (
            None
            if spec["stage"] in ("materialize", "recipe")
            else _validate_policy_materialization(
                spec["policy_materialization"],
                checkout=checkout,
                bundle=bundle,
            )
        ),
    }
    if (
        bundle != payload["bundle"]
        or materialization_inputs != payload["materialization_inputs"]
        or _output_contract(spec) != payload["output_contract"]
        or payload["boot_marker"] != payload["output_contract"]["boot_marker"]
        or _training_argv(spec, bundle) != payload["training_argv"]
    ):
        raise LaunchRefused("bundle or training argv drifted")
    lock_path = Path(spec["gpu"]["lock_path"])
    info = os.fstat(lock_fd)
    path_info = lock_path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)
    ):
        raise LaunchRefused("inherited GPU lock identity differs")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise LaunchRefused("inherited GPU lock is not held") from exc
    _lock_gpu_admission(lock_fd)
    try:
        gpu = _verify_gpu_admission(
            spec,
            phase="pre_exec",
            current_namespace=Path(spec["namespace"]),
        )
        _B._write_exclusive_json(
            Path(spec["namespace"]) / "pre_exec_gpu_admission.json",
            {
                "schema_version": 1,
                "kind": "measured_vendor_v2_pre_exec_gpu_admission_v2",
                "launch_claim_sha256": claim_sha,
                "gpu": gpu,
            },
        )
        namespace_receipt, namespace_receipt_sha = _runtime_namespace_receipt(
            spec, claim_sha
        )
    finally:
        _unlock_gpu_admission(lock_fd)
    wbt = checkout / _B.WBT_RELATIVE
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(wbt / "source/whole_body_tracking"),
        "CUDA_VISIBLE_DEVICES": str(spec["gpu"]["index"]),
        "HYDRA_FULL_ERROR": "1",
        "WANDB_MODE": "offline",
        "HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256": claim_sha,
        GPU_NAMESPACE_RECEIPT_ENV: str(namespace_receipt),
        GPU_NAMESPACE_RECEIPT_SHA_ENV: namespace_receipt_sha,
        **_B._runtime_asset_exec_environment(assets),
        **_cuda_launch_blocking_environment(spec),
    }
    os.chdir(wbt)
    os.execve(payload["training_argv"][0], payload["training_argv"], environment)
    raise AssertionError("execve returned")


def _validate_oracle_completion_state(path: Path) -> dict[str, str]:
    _B._stable_regular_file(path, name="oracle completion state")
    observed: dict[str, str] = {}
    required = {"completion_exit_code", "terminal_kind", "terminal_exit_code"}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in required:
            if key in observed:
                raise LaunchRefused("oracle completion state has duplicate %s" % key)
            observed[key] = value
    if observed != {
        "completion_exit_code": "0",
        "terminal_kind": "clean_completion",
        "terminal_exit_code": "0",
    }:
        raise LaunchRefused("oracle workload did not exit cleanly and uniquely")
    return observed


def launch(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    expected = _B._sha256(confirm_claim, name="--confirm-claim")
    if expected != plan["launch_claim_sha256"]:
        raise LaunchRefused("--confirm-claim differs from freshly recomputed plan")
    spec = plan["canonical_payload"]["spec"]
    checkout = Path(spec["source"]["checkout"])
    _B._verify_clean_source(checkout, spec["source"]["commit_sha"])
    _B._validate_runtime_asset_claim(plan["canonical_payload"]["runtime_assets"])
    lock_fd = _open_gpu_shared_lock(Path(spec["gpu"]["lock_path"]))
    namespace = None
    try:
        _lock_gpu_admission(lock_fd)
        try:
            first = _verify_gpu_admission(
                spec,
                phase="pre_launch",
                current_namespace=None,
            )
            namespace = _B._claim_namespace(plan)
            _B._write_exclusive_json(
                namespace / GPU_RESERVATION_FILENAME,
                _reservation_document(spec, expected),
            )
            _B._write_exclusive_json(
                namespace / "pre_launch_gpu_admission.json",
                {
                    "schema_version": 1,
                    "kind": "measured_vendor_v2_pre_launch_gpu_admission_v2",
                    "launch_claim_sha256": expected,
                    "gpu": first,
                },
            )
        finally:
            _unlock_gpu_admission(lock_fd)
        state = Path(spec["log_path"] + ".launch")
        internal = [
            spec["source"]["isaac_python"],
            str(checkout / LAUNCHER_SOURCE),
            "_exec",
            "--claim",
            str(namespace / "launch_claim.json"),
            "--claim-sha256",
            expected,
            "--gpu-lock-fd",
            str(lock_fd),
        ]
        environment = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": os.environ.get("HOME", "/root"),
            "LANG": "C",
            "LC_ALL": "C",
            "KIT_BOOT_MARKER": plan["canonical_payload"]["boot_marker"],
            "KIT_BOOT_TIMEOUT_S": "2700",
            "KIT_BOOT_STALE_TIMEOUT_S": "1800",
            "KIT_BOOT_POLL_S": "5",
            "KIT_BOOT_STATE_FILE": str(state),
            **(
                {
                    "KIT_WAIT_FOR_COMPLETION": "1",
                    "KIT_COMPLETION_TIMEOUT_S": "120",
                }
                if _is_teacher_oracle_stage(spec["stage"])
                else {}
            ),
        }
        result = subprocess.run(
            [str(checkout / _B.KIT_LAUNCHER_SOURCE), spec["log_path"], *internal],
            cwd=checkout / _B.WBT_RELATIVE,
            env=environment,
            pass_fds=(lock_fd,),
            check=False,
        )
        if result.returncode != 0:
            raise LaunchRefused(
                "locked Kit launcher returned %d; namespace remains spent" % result.returncode
            )
        oracle_completion = (
            _validate_oracle_completion_state(state)
            if _is_teacher_oracle_stage(spec["stage"])
            else None
        )
        _lock_gpu_admission(lock_fd)
        try:
            try:
                final_gpu = _verify_gpu_admission(
                    spec,
                    phase=(
                        "post_completion"
                        if _is_teacher_oracle_stage(spec["stage"])
                        else "post_boot"
                    ),
                    current_namespace=namespace,
                    require_current_compute=not _is_teacher_oracle_stage(
                        spec["stage"]
                    ),
                )
                _B._write_exclusive_json(
                    namespace
                    / (
                        "post_completion_gpu_admission.json"
                        if _is_teacher_oracle_stage(spec["stage"])
                        else "post_boot_gpu_admission.json"
                    ),
                    {
                        "schema_version": 1,
                        "kind": (
                            "measured_vendor_v2_post_completion_gpu_admission_v1"
                            if _is_teacher_oracle_stage(spec["stage"])
                            else "measured_vendor_v2_post_boot_gpu_admission_v1"
                        ),
                        "launch_claim_sha256": expected,
                        "gpu": final_gpu,
                    },
                )
            except (LaunchRefused, FileNotFoundError, ValueError, OSError) as exc:
                if _is_teacher_oracle_stage(spec["stage"]):
                    raise LaunchRefused(
                        "post-completion admission refused after exact clean exit"
                    ) from exc
                failure = _cleanup_post_boot_admission_failure(
                    namespace,
                    state,
                    expected,
                    str(exc),
                )
                cleanup = failure["cleanup"]
                outcome = (
                    "completed"
                    if cleanup["completed"] is True
                    else "incomplete"
                )
                raise LaunchRefused(
                    "post-boot admission refused; exact current-trainer cleanup "
                    "%s; failure receipt=%s"
                    % (outcome, failure["path"])
                ) from exc
        finally:
            _unlock_gpu_admission(lock_fd)
        materialized_reward = None
        materialized_policy = None
        teacher_qdes_oracle = None
        output_contract = plan["canonical_payload"]["output_contract"]
        if spec["stage"] == "materialize":
            output_path = Path(output_contract["effective_reward_recipe"])
            materialized_reward = _validate_reward_materialization(
                {
                    "path": str(output_path),
                    "sha256": _B.sha256_file(output_path),
                }
            )
        elif spec["stage"] == "recipe":
            output_path = Path(output_contract["policy_recipe"])
            materialized_policy = _validate_policy_materialization(
                {
                    "path": str(output_path),
                    "sha256": _B.sha256_file(output_path),
                },
                checkout=checkout,
                bundle=plan["canonical_payload"]["bundle"],
            )
        elif _is_teacher_oracle_stage(spec["stage"]):
            output_path = Path(output_contract["teacher_qdes_oracle"])
            teacher_qdes_oracle = _validate_teacher_qdes_oracle(
                {
                    "path": str(output_path),
                    "sha256": _B.sha256_file(output_path),
                },
                spec=spec,
                claim=plan["canonical_payload"],
            )
        return {
            "schema_version": 1,
            "kind": "n1_measured_vendor_v2_diagnostic_launch_result_v2",
            "launch_claim_sha256": expected,
            "stage": spec["stage"],
            "namespace": str(namespace),
            "log_path": spec["log_path"],
            "state_path": str(state),
            "gpu": spec["gpu"],
            "post_boot_gpu_admission": (
                None if _is_teacher_oracle_stage(spec["stage"]) else final_gpu
            ),
            "post_completion_gpu_admission": (
                final_gpu if _is_teacher_oracle_stage(spec["stage"]) else None
            ),
            "oracle_completion": oracle_completion,
            "output_contract": output_contract,
            "materialized_effective_reward_recipe": materialized_reward,
            "materialized_policy_recipe": materialized_policy,
            "teacher_qdes_oracle": teacher_qdes_oracle,
            "ppo_update_count": output_contract["ppo_update_count"],
            "diagnostic_unauthorized": True,
            "accepted": True,
        }
    finally:
        os.close(lock_fd)


def _write_template(args: argparse.Namespace) -> dict[str, Any]:
    budget = BUDGETS[args.stage]
    allow_colocation = bool(
        getattr(args, VENDOR_V2_COLOCATION_SPEC_KEY, False)
    )
    namespace = Path(args.namespace).resolve(strict=False)
    isaac_python = _isaac_python_entry(args.isaac_python)
    reward_pin = None
    policy_pin = None
    if args.stage != "materialize":
        if (
            args.reward_materialization_path is None
            or args.reward_materialization_sha256 is None
        ):
            raise LaunchRefused(
                "%s template requires the exact reward materialization path/SHA"
                % args.stage
            )
        reward_materialization = _validate_reward_materialization(
            {
                "path": args.reward_materialization_path,
                "sha256": args.reward_materialization_sha256,
            }
        )
        reward_pin = reward_materialization["artifact"]
        reward_sha = reward_materialization[
            "effective_reward_recipe_sha256"
        ]
    else:
        if (
            args.reward_materialization_path is not None
            or args.reward_materialization_sha256 is not None
        ):
            raise LaunchRefused(
                "materialize template must not accept a reward materialization"
            )
        reward_sha = None
    if args.stage not in ("materialize", "recipe"):
        if (
            args.policy_materialization_path is None
            or args.policy_materialization_sha256 is None
        ):
            raise LaunchRefused(
                "%s template requires the exact policy materialization path/SHA"
                % args.stage
            )
        policy_materialization = _validate_policy_materialization_header(
            {
                "path": args.policy_materialization_path,
                "sha256": args.policy_materialization_sha256,
            }
        )
        policy_pin = policy_materialization["artifact"]
        policy_sha = policy_materialization["policy_contract_sha256"]
    else:
        if (
            args.policy_materialization_path is not None
            or args.policy_materialization_sha256 is not None
        ):
            raise LaunchRefused(
                "%s template must not accept a policy materialization"
                % args.stage
            )
        policy_sha = None
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(Path(args.checkout).resolve(strict=True)),
            "commit_sha": args.commit_sha,
            "isaac_python": str(isaac_python),
        },
        "action_id": args.action_id,
        "bundle": {"path": args.bundle_path, "sha256": args.bundle_sha256},
        "target_recipe": args.target_recipe,
        "target_validity_mask": list(RECIPES[args.target_recipe]),
        "reward_materialization": reward_pin,
        "policy_materialization": policy_pin,
        "policy_contract_sha256": policy_sha,
        "expected_effective_reward_recipe_sha256": reward_sha,
        "seed": args.seed,
        "stage": args.stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "gpu": {
            "index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "owner": args.owner,
            "lock_path": "/tmp/hope_lean_queue_gpu%d.lock" % args.gpu_index,
            "require_empty": not allow_colocation,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }
    if getattr(args, CUDA_LAUNCH_BLOCKING_SPEC_KEY, False):
        document[CUDA_LAUNCH_BLOCKING_SPEC_KEY] = True
    if allow_colocation:
        document[VENDOR_V2_COLOCATION_SPEC_KEY] = True
    output = Path(args.output).resolve(strict=False)
    _B._write_exclusive_json(output, document)
    return {"status": "CREATED", "spec": str(output), "target_recipe": args.target_recipe}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--output", required=True)
    template.add_argument("--checkout", required=True)
    template.add_argument("--commit-sha", required=True)
    template.add_argument("--isaac-python", required=True)
    template.add_argument("--action-id", required=True)
    template.add_argument("--bundle-path", required=True)
    template.add_argument("--bundle-sha256", required=True)
    template.add_argument("--target-recipe", required=True, choices=tuple(RECIPES))
    template.add_argument("--reward-materialization-path")
    template.add_argument("--reward-materialization-sha256")
    template.add_argument("--policy-materialization-path")
    template.add_argument("--policy-materialization-sha256")
    template.add_argument("--seed", required=True, type=int, choices=(0,))
    template.add_argument("--stage", required=True, choices=tuple(BUDGETS))
    template.add_argument("--gpu-index", required=True, type=int)
    template.add_argument("--gpu-uuid", required=True)
    template.add_argument("--owner", required=True)
    template.add_argument("--namespace", required=True)
    template.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="diagnostic-only: set CUDA_LAUNCH_BLOCKING=1 in the trainer",
    )
    template.add_argument(
        "--allow-vendor-v2-colocation",
        action="store_true",
        help=(
            "diagnostic-only: admit one already verified VendorV2 process on "
            "the same physical GPU (hard max: two compute PIDs)"
        ),
    )
    for command in ("plan", "launch"):
        child = sub.add_parser(command)
        child.add_argument("--spec", required=True)
        if command == "launch":
            child.add_argument("--confirm-claim", required=True)
    internal = sub.add_parser("_exec", help=argparse.SUPPRESS)
    internal.add_argument("--claim", required=True)
    internal.add_argument("--claim-sha256", required=True)
    internal.add_argument("--gpu-lock-fd", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "template":
            result = _write_template(args)
        elif args.command == "_exec":
            return _internal_exec(Path(args.claim), args.claim_sha256, args.gpu_lock_fd)
        else:
            plan = build_plan(Path(args.spec))
            if args.command == "plan":
                result = plan
            else:
                result = launch(plan, confirm_claim=args.confirm_claim)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    except (LaunchRefused, FileNotFoundError, ValueError, OSError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
