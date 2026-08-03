#!/usr/bin/env python3
"""Plan or explicitly execute the fresh-only C211 diagnostic chain.

This launcher owns a C211-only lineage and claim surface.  It cannot consume an
A225 or historical fixed-194 lineage, normalizer, checkpoint, or result.  The
only authorized chain is materialize -> recipe -> oracle32 -> scale4096 ->
long4096.  Planning is read-only; execution requires the freshly recomputed
claim digest and permanently spends a new namespace.
"""

from __future__ import annotations

import argparse
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


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_FILE = SCRIPT_DIR / "launch_n1_reward_screen_diagnostic.py"
ADMISSION_FILE = SCRIPT_DIR / "vendor_v2_gpu_admission.py"
EXACT_GROUP_FILE = SCRIPT_DIR / "exact_process_group.py"
OLD_VALIDATOR_FILE = SCRIPT_DIR / "launch_n1_measured_vendor_v2_diagnostic.py"
TASK_WAIT_FILE = (
    SCRIPT_DIR.parent
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/action_ball_task_wait.py"
)


def _load_helper(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot import helper %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_B = _load_helper("_c211_diagnostic_base", BASE_FILE)
_A = _load_helper("_c211_diagnostic_gpu_admission", ADMISSION_FILE)
_G = _load_helper("_c211_diagnostic_exact_process_group", EXACT_GROUP_FILE)
_OLD = _load_helper("_c211_diagnostic_oracle_validator", OLD_VALIDATOR_FILE)
_W = _load_helper("_c211_task_wait_schedule", TASK_WAIT_FILE)

LaunchRefused = _B.LaunchRefused

SCHEMA_VERSION = 2
SPEC_KIND = "action_ball_c211_diagnostic_spec_v2"
CLAIM_KIND = "action_ball_c211_diagnostic_claim_v2"
LINEAGE_KIND = "action_ball_c211_fixed_midpoint_lineage_v1"
C211_BUNDLE_KIND = "action_ball_c211_fixed_midpoint_bundle_v1"
RECIPE_KIND = "action_ball_c211_matched_recipe_v1"
MATERIALIZATION_KIND = "action_ball_c211_reward_materialization_v1"
POLICY_MATERIALIZATION_KIND = "action_ball_c211_policy_materialization_v1"
ORACLE32_KIND = "action_ball_c211_oracle32_receipt_v1"
C211_RAW_ORACLE_KIND = "action_ball_c211_oracle_raw_evidence_v1"
C211_RUNNER_PREFLIGHT_KIND = "action_ball_c211_runner_preflight_evidence_v1"
C211_SELECTED_RUBBER_KIND = (
    "action_ball_c211_selected_rubber_contact_evidence_v1"
)
RESULT_KIND = "action_ball_c211_diagnostic_launch_result_v1"
FRAME0_EXACT_ARTIFACT_KIND = "action_ball_c211_frame0_exact_artifact_v1"
FRAME0_EXACT_RECEIPT_KIND = "action_ball_c211_frame0_exact_receipt_v1"
FRAME0_EXACT_SOURCE_KIND = "action_ball_c211_teacher_motion_frame0_exact_v1"
EXPERIMENT_NAME = "agibot_a3_action_ball_c211_diagnostic"

RECIPE_ID = "C0-corrected-phase-fixedlr"
ACTOR_CONTRACT = "action_ball_c211"
ACTOR_WIDTH = 211
CRITIC_CONTRACT = "action_ball_c211_critic_v1"
CRITIC_WIDTH = 319
TRAINABILITY_CONTRACT = "action_ball_c211_fixed_midpoint_learnability_v1"
ACTOR_NORMALIZER_IDENTITY = "action_ball_c211_actor_norm_v1"
CRITIC_NORMALIZER_IDENTITY = "action_ball_c211_critic_norm_v1"
TASK_PROFILE_ID = "HOPEPingPongActionBallC211VendorV2N1Learnability"
GYM_TASK_ID = "HOPE-PingPong-ActionBall-C211Learnability-AgibotA3-v0"
TARGET_SEMANTICS = "c211_incoming_ball_p_v_spin_outcome_dense_v1"
TARGET_RECIPE = "outcome_dense_only"
TARGET_VALIDITY_MASK = (False, False, False)
ACTION_ID = "take_061_unit04_bh"
ACTION_UID = 5527597793770800
TEACHER_ID = "Take_061_unit04_BH"
INCOMING_BALL_FIELDS = (
    "incoming_ball_contact_position_heading",
    "incoming_ball_contact_velocity_heading",
    "incoming_ball_contact_spin_heading",
)
PHYSICAL_BALL_SEMANTICS = "analytic_virtual_ball_authoritative_physx_disabled"
REWARD_MATERIALIZATION_PROFILE = "measured_vendor_v2_n1_static_v1"
REWARD_RECIPE_FILENAME = "c211_effective_reward_recipe.json"
POLICY_RECIPE_FILENAME = "c211_dynamic_ready_policy_recipe.json"
RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64
POLICY_DT_S = 0.02
WAIT_SCHEDULE = _W.ActionBallTaskWaitSchedule(
    seed=20260804,
    min_wait_ticks=5,
    max_wait_ticks=25,
    episode_horizon_ticks=500,
    required_active_ticks=200,
).to_dict()
COLOCATION_SPEC_KEY = "allow_vendor_v2_colocation"
HARD_TERMINATION_UNION = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)

REQUIRED_OUTCOME_TERMS: Mapping[str, Mapping[str, Any]] = {
    "c225_strike_ball_paddle_center_proximity": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.action_ball_c225_rewards."
            "c225_strike_ball_paddle_center_proximity"
        ),
        "weight": 220.0,
        "params": {"command_name": "racket_target", "std": 0.15},
    },
    "virtual_landing": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.action_ball_c225_rewards."
            "c225_landing_outcome_actual_contact"
        ),
        "weight": 500.0,
        "params": {
            "command_name": "racket_target",
            "mode": "legal_base",
            "base_frac": 0.6,
            "off_table_frac": 0.5,
            "settle_delay_s": 0.0,
        },
    },
}
REQUIRED_PRIOR_TERMS: Mapping[str, Mapping[str, Any]] = {
    "upright_exp": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.upright_exp",
        "params": {"std": 0.4472135954999579},
    },
    "motion_body_pos": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
            "motion_body_pos_swing_only"
        ),
        "params": {"command_name": "motion"},
    },
    "motion_body_ori": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
            "motion_body_ori_swing_only"
        ),
        "params": {"command_name": "motion"},
    },
    "motion_body_lin_vel": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.rewards."
            "motion_global_body_linear_velocity_error_exp"
        ),
        "params": {"command_name": "motion"},
    },
    "motion_body_ang_vel": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.rewards."
            "motion_global_body_angular_velocity_error_exp"
        ),
        "params": {"command_name": "motion"},
    },
    "motion_racket_position": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
            "motion_racket_position_tracking_cauchy"
        ),
        "params": {"command_name": "racket_target", "scale_in_strike_window": 1.0},
    },
    "motion_racket_velocity": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
            "motion_racket_velocity_tracking_cauchy"
        ),
        "params": {"command_name": "racket_target", "scale_in_strike_window": 1.0},
    },
    "motion_racket_normal": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
            "motion_racket_normal_tracking_cauchy"
        ),
        "params": {"command_name": "racket_target", "scale_in_strike_window": 1.0},
    },
    "motion_racket_long_axis": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.hope_rewards."
            "motion_racket_long_axis_tracking_cauchy"
        ),
        "params": {"command_name": "racket_target", "scale_in_strike_window": 1.0},
    },
}
PROHIBITED_CONTACT_TARGET_TERMS = (
    "racket_position",
    "racket_velocity",
    "racket_normal",
    "racket_position_coarse",
    "racket_velocity_coarse",
    "racket_normal_coarse",
    "racket_position_precision",
    "racket_velocity_precision",
    "racket_normal_precision",
    "racket_strike_success",
)
PROHIBITED_DUPLICATE_OUTCOME_TERMS = (
    "strike_capture_bonus",
    "virtual_pass_net",
    "virtual_landing_dense",
)

LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_action_ball_c211_diagnostic.py"
)
ADMISSION_SOURCE = "hope_training/whole_body_tracking/scripts/vendor_v2_gpu_admission.py"
EXACT_GROUP_SOURCE = "hope_training/whole_body_tracking/scripts/exact_process_group.py"
BASE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)
TRAIN_SOURCE = "hope_training/whole_body_tracking/scripts/train.py"
OLD_VALIDATOR_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_measured_vendor_v2_diagnostic.py"
)
TRAINING_CONTRACT_SOURCE = _OLD.TRAINING_CONTRACT_SOURCE
TASK_WAIT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/action_ball_task_wait.py"
)
KIT_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
)
TASK_PROFILE_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallC211VendorV2N1Learnability.yaml"
)
C211_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/action_ball_c211_trainability.py"
)
C211_ENV_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
C211_REWARD_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_c225_rewards.py"
)
C211_MDP_EXPORT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/__init__.py"
)
EFFECTIVE_REWARD_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/effective_reward_recipe.py"
)
C211_REGISTRY_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/__init__.py"
)

RUNTIME_SOURCE_PATHS = (
    (LAUNCHER_SOURCE, "VendorV2 N1 launcher"),
    (ADMISSION_SOURCE, "VendorV2 GPU admission"),
    (EXACT_GROUP_SOURCE, "exact process-group helper"),
    (BASE_SOURCE, "no-clobber base helper"),
    (KIT_LAUNCHER_SOURCE, "locked Kit launcher"),
    (TRAIN_SOURCE, "training entrypoint"),
    (OLD_VALIDATOR_SOURCE, "oracle32 acceptance validator"),
    (TRAINING_CONTRACT_SOURCE, "dynamic-ready policy contract"),
    (TASK_WAIT_SOURCE, "pre-task wait schedule contract"),
    (TASK_PROFILE_SOURCE, "C211 task profile"),
    (C211_CONTRACT_SOURCE, "C211 trainability contract"),
    (C211_ENV_SOURCE, "C211 environment config"),
    (C211_REWARD_SOURCE, "C211 causal reward functions"),
    (C211_MDP_EXPORT_SOURCE, "C211 reward MDP export"),
    (EFFECTIVE_REWARD_SOURCE, "effective reward receipt taxonomy"),
    (C211_REGISTRY_SOURCE, "C211 Gym registration"),
)

BUDGETS: Mapping[str, tuple[int, int, int]] = {
    "materialize": (1, 0, 1),
    "recipe": (1, 0, 1),
    "oracle32": (1, 0, 1),
    "scale4096": (4096, 5, 1),
    "long4096": (4096, 1000, 100),
}
STAGE_ORDER = tuple(BUDGETS)
BLOCKED_RUNTIME_STAGES = ("oracle32", "scale4096", "long4096")
ORACLE_RUNTIME_BLOCKER = "C211_ORACLE_NOT_IMPLEMENTED"
ORACLE_RUNTIME_DEPENDENCIES = (
    "C211_000_RAW_EXPORTER_NOT_IMPLEMENTED",
    "C211_RUNNER_PREFLIGHT_RECEIPT_NOT_IMPLEMENTED",
    "C211_SELECTED_RUBBER_AUTHORITY_NOT_IMPLEMENTED",
)
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PIN_KEYS = ("path", "sha256")
FOREIGN_VALUE_TOKENS = (
    "action_ball_a225",
    "a225",
    "l194",
    "fixed-194",
)


def canonical_sha256(value: Any) -> str:
    return _B.canonical_sha256(value)


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    return _B._exact_dict(value, tuple(keys), name=name)


def _assert_c211_only(value: Any, *, name: str) -> None:
    """Reject foreign lineage vocabulary while treating hashes as opaque."""

    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) is not str:
                raise LaunchRefused("%s contains a non-string key" % name)
            lowered = key.lower()
            if lowered.endswith("sha256") or lowered == "commit_sha":
                continue
            _assert_c211_only(child, name=name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_c211_only(child, name=name)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in FOREIGN_VALUE_TOKENS):
            raise LaunchRefused("%s contains a foreign ABI/lineage token" % name)


def _pin(value: Any, *, name: str) -> dict[str, str]:
    row = _exact_dict(value, PIN_KEYS, name=name)
    return {
        "path": _B._relative_path(row["path"], name="%s.path" % name),
        "sha256": _B._sha256(row["sha256"], name="%s.sha256" % name),
    }


def _external_pin(value: Any, *, name: str) -> tuple[dict[str, str], Path]:
    row = _exact_dict(value, PIN_KEYS, name=name)
    path = _B._absolute_path(row["path"], name="%s.path" % name, must_exist=True)
    _B._stable_regular_file(path, name=name)
    digest = _B._sha256(row["sha256"], name="%s.sha256" % name)
    if _B.sha256_file(path) != digest:
        raise LaunchRefused("%s file SHA differs" % name)
    return {"path": str(path), "sha256": digest}, path


def _canonical_external_json(
    value: Any, *, name: str
) -> tuple[dict[str, str], dict[str, Any]]:
    pin, path = _external_pin(value, name=name)
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name=name)
    if raw != _B._canonical_bytes(document) + b"\n" or type(document) is not dict:
        raise LaunchRefused("%s must be a canonical JSON object plus newline" % name)
    return pin, document


def _tracked_json(
    checkout: Path, commit: str, value: Any, *, name: str
) -> tuple[dict[str, str], dict[str, Any]]:
    pin = _pin(value, name=name)
    normalized, path = _B._verify_tracked_file(checkout, commit, pin, name=name)
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name=name)
    if raw != _B._canonical_bytes(document) + b"\n" or type(document) is not dict:
        raise LaunchRefused("%s must be a canonical JSON object plus newline" % name)
    return normalized, document


def _wait_contract() -> dict[str, Any]:
    return {
        "policy_dt_s": POLICY_DT_S,
        "schedule": dict(WAIT_SCHEDULE),
        "in_loop_expansion_prohibited": True,
    }


def _hard_wait_contract() -> dict[str, Any]:
    """Exact schema-3 field names emitted by training_contract.py."""

    return {
        "identity": "action_ball_pre_task_wait_schedule_v1",
        "policy_dt_s": POLICY_DT_S,
        "seed": WAIT_SCHEDULE["seed"],
        "min_wait_ticks": WAIT_SCHEDULE["min_wait_ticks"],
        "max_wait_ticks": WAIT_SCHEDULE["max_wait_ticks"],
        "episode_horizon_ticks": WAIT_SCHEDULE["episode_horizon_ticks"],
        "required_active_ticks": WAIT_SCHEDULE["required_active_ticks"],
        "schedule_canonical_sha256": WAIT_SCHEDULE["canonical_sha256"],
        "task_valid_actor_and_critic": True,
        "wait_task_ball_base_and_clocks_masked": True,
        "wait_remaining_observed": False,
    }


def _hard_question_source_contract() -> dict[str, Any]:
    """Exact fixed-band/offline-inverse scope emitted by schema-3 runtime."""

    return {
        "identity": "action_ball_211_question_source_scope_v1",
        "current_immutable_tape": {
            "scope": "diagnostic_n1_early_fixed_band_only",
            "final_curriculum_frozen": False,
        },
        "final_curriculum": {
            "source": "pregenerated_cached_band_question_bank",
            "generation": "offline_before_rollout",
            "reset_selection": "index_one_bank_row",
            "online_inverse_solves_per_reset": 0,
            "online_inverse_solves_per_step": 0,
            "wait_remaining_observed": False,
        },
    }


def _curriculum_scope_contract() -> dict[str, Any]:
    return {
        "current_tape_scope": "diagnostic_n1_early_fixed_band_only",
        "permanent_single_question_curriculum": False,
        "final_curriculum_source": "pregenerated_cached_band_question_bank",
        "reset_question_selection": "index_pregenerated_bank_row",
        "online_inverse_solve_calls": 0,
    }


def _verify_frame0_artifact_source_commit(
    checkout: Path, artifact_source_commit: str, artifact_pin: Mapping[str, str]
) -> None:
    committed = subprocess.run(
        [
            "git", "-C", str(checkout), "show",
            artifact_source_commit + ":" + artifact_pin["path"],
        ],
        check=False,
        capture_output=True,
    )
    if (
        committed.returncode != 0
        or hashlib.sha256(committed.stdout).hexdigest() != artifact_pin["sha256"]
    ):
        raise LaunchRefused("C211 frame0-exact artifact source commit differs")


def _question_contract() -> dict[str, Any]:
    return {
        "target_source": "immutable_tape",
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "target_observation_noise": False,
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "desired_contact_fields_observed": False,
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
    }


def _c211_reward_contract() -> dict[str, Any]:
    return {
        "identity": "action_ball_c211_achieved_outcome_reward_v2",
        "desired_contact_position_velocity_face_consumed": False,
        "task_valid_required": True,
        "strike_bridge": {
            "term": "c225_strike_ball_paddle_center_proximity",
            "callable": REQUIRED_OUTCOME_TERMS[
                "c225_strike_ball_paddle_center_proximity"
            ]["callable"],
            "weight": 220.0,
            "std_m": 0.15,
            "kernel": "cauchy_inverse_quadratic",
            "eligibility": "task_valid_active_swing_single_exact_strike_tick",
            "miss_retains_gradient": True,
        },
        "economics": {
            "policy_dt_s": 0.02,
            "compatible_swing_motion_static_max": 3.6575,
            "strike_bridge_post_dt_peak": 4.4,
            "legal_landing_post_dt_min": 6.0,
            "ordering": "motion_lt_strike_peak_lt_legal_landing",
        },
        "landing": {
            "term": "virtual_landing",
            "callable": REQUIRED_OUTCOME_TERMS["virtual_landing"]["callable"],
            "weight": 500.0,
            "evidence_source": (
                "analytic_prediction_from_achieved_selected_rubber_contact"
            ),
            "observed_physical_landing_available": False,
            "eligibility": (
                "task_valid_and_actual_selected_rubber_contact_and_finite_landing_plane_"
                "and_net_crossed_and_net_clear"
            ),
            "legal_opponent_table": "0.6_plus_0.4_gaussian",
            "opponent_side_off_table": "0.5_times_same_gaussian",
            "miss_or_invalid_or_hypothetical": 0.0,
            "sigma_m": 1.0,
        },
        "legacy_duplicate_outcome_terms_active": False,
        "rollout0_required_priors": list(REQUIRED_PRIOR_TERMS),
    }


def _recipe_contract() -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "kind": RECIPE_KIND,
        "recipe_id": RECIPE_ID,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "fresh_normalizers_required": True,
        "foreign_checkpoint_reuse_prohibited": True,
        "init_noise_std": 0.02,
        "noise_std_type": "log",
        "entropy_coef": 0.01,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "reference_guard_mode": "phase_gated",
        "soft_weights": {
            "death_penalty": -30.0,
            "qdes_limit": -0.5,
            "qdes_projection": -0.5,
            "joint_limit": -0.5,
        },
        "ppo": {
            "schedule": "fixed",
            "learning_rate": 1.0e-4,
            "desired_kl": 0.01,
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
        },
    }
    return {**unsigned, "recipe_contract_sha256": canonical_sha256(unsigned)}


def _validate_lineage(
    checkout: Path, commit: str, value: Any
) -> dict[str, Any]:
    pin, row = _tracked_json(checkout, commit, value, name="C211 lineage")
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "trainability_contract",
            "trainability_contract",
            "actor_normalizer_identity",
            "critic_normalizer_identity",
            "task_profile",
            "gym_task",
            "target_semantics",
            "curriculum_scope",
            "target_recipe",
            "target_validity_mask",
            "incoming_ball_fields",
            "reset_inverse_solve",
            "online_solver_calls",
            "online_lm_calls",
            "physical_rng_draws",
            "action_id",
            "action_uid",
            "teacher_id",
            "seed",
            "bundle",
            "motion",
            "immutable_tape",
            "action_manifest",
            "dynamic_ready_artifact",
            "dynamic_ready_nominal_receipt",
            "frame0_exact_artifact",
            "frame0_exact_receipt",
        ),
        name="C211 lineage",
    )
    _assert_c211_only(row, name="C211 lineage")
    expected = {
        "schema_version": 1,
        "kind": LINEAGE_KIND,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "actor_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "task_profile": TASK_PROFILE_ID,
        "gym_task": GYM_TASK_ID,
        "target_semantics": TARGET_SEMANTICS,
        "curriculum_scope": _curriculum_scope_contract(),
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
    }
    for key, wanted in expected.items():
        if row[key] != wanted:
            raise LaunchRefused("C211 lineage %s differs" % key)
    action_id = row["action_id"]
    action_uid = row["action_uid"]
    teacher_id = row["teacher_id"]
    if (
        type(action_id) is not str
        or SAFE_COMPONENT.fullmatch(action_id) is None
        or type(action_uid) is not int
        or action_uid <= 0
        or type(teacher_id) is not str
        or SAFE_COMPONENT.fullmatch(teacher_id) is None
    ):
        raise LaunchRefused("C211 action/teacher identity is unsafe")
    if (
        action_id != ACTION_ID
        or action_uid != ACTION_UID
        or teacher_id != TEACHER_ID
    ):
        raise LaunchRefused("C211 first-wave action/teacher identity differs")
    seed = _B._plain_int(
        row["seed"], name="C211 lineage seed", maximum=(1 << 31) - 1
    )
    if seed != 0:
        raise LaunchRefused("C211 first-wave lineage requires seed 0")
    pins = {}
    paths = {}
    for key in (
        "bundle",
        "motion",
        "immutable_tape",
        "action_manifest",
        "dynamic_ready_artifact",
        "dynamic_ready_nominal_receipt",
        "frame0_exact_artifact",
        "frame0_exact_receipt",
    ):
        normalized, _path = _B._verify_tracked_file(
            checkout,
            commit,
            _pin(row[key], name="lineage.%s" % key),
            name="C211 %s" % key,
        )
        pins[key] = normalized
        paths[key] = _path

    documents = {}
    for key in (
        "bundle",
        "immutable_tape",
        "action_manifest",
        "dynamic_ready_artifact",
        "dynamic_ready_nominal_receipt",
        "frame0_exact_artifact",
        "frame0_exact_receipt",
    ):
        document = _B._strict_json_bytes(
            paths[key].read_bytes(), name="C211 %s" % key
        )
        if type(document) is not dict:
            raise LaunchRefused("C211 %s must be a JSON object" % key)
        documents[key] = document

    bundle = _sealed_row(
        documents["bundle"],
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "action_id",
            "action_uid",
            "teacher_id",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "trainability_contract",
            "actor_normalizer_identity",
            "critic_normalizer_identity",
            "target_recipe",
            "curriculum_scope",
            "target_validity_mask",
            "incoming_ball_fields",
            "reset_inverse_solve",
            "online_solver_calls",
            "online_lm_calls",
            "physical_rng_draws",
            "motion",
            "immutable_tape",
            "action_manifest",
            "dynamic_ready_artifact",
            "dynamic_ready_nominal_receipt",
            "frame0_exact_artifact",
            "frame0_exact_receipt",
        ),
        name="C211 bundle",
    )
    expected_bundle = {
        "schema_version": 1,
        "kind": C211_BUNDLE_KIND,
        "diagnostic_unauthorized": True,
        "action_id": action_id,
        "action_uid": action_uid,
        "teacher_id": teacher_id,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "actor_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "target_recipe": TARGET_RECIPE,
        "curriculum_scope": _curriculum_scope_contract(),
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
        "motion": pins["motion"],
        "immutable_tape": pins["immutable_tape"],
        "action_manifest": pins["action_manifest"],
        "dynamic_ready_artifact": pins["dynamic_ready_artifact"],
        "dynamic_ready_nominal_receipt": pins[
            "dynamic_ready_nominal_receipt"
        ],
        "frame0_exact_artifact": pins["frame0_exact_artifact"],
        "frame0_exact_receipt": pins["frame0_exact_receipt"],
    }
    if any(bundle[key] != wanted for key, wanted in expected_bundle.items()):
        raise LaunchRefused("C211 bundle semantics or input closure differs")

    tape = documents["immutable_tape"]
    tape_unsigned = dict(tape)
    tape_seal = tape_unsigned.pop("canonical_sha256", None)
    question = tape.get("question")
    target = (
        tape.get("targets", {}).get(TARGET_RECIPE)
        if type(tape.get("targets")) is dict
        else None
    )
    reset = tape.get("reset_semantics")
    if (
        tape.get("schema_version") != 1
        or tape.get("kind") != "action_ball_n1_immutable_single_question_tape"
        or tape.get("diagnostic_unauthorized") is not True
        or tape.get("row_count") != 1
        or _B._sha256(tape_seal, name="C211 tape canonical SHA")
        != canonical_sha256(tape_unsigned)
        or type(question) is not dict
        or question.get("action_uid") != action_uid
        or question.get("motion_sha256") != pins["motion"]["sha256"]
        or any(
            type(question.get(field)) is not list
            or len(question[field]) != 3
            for field in (
                "ball_contact_w_m",
                "incoming_velocity_w_mps",
                "incoming_spin_w_radps",
            )
        )
        or type(target) is not dict
        or target.get("recipe") != TARGET_RECIPE
        or target.get("validity_mask") != list(TARGET_VALIDITY_MASK)
        or type(reset) is not dict
        or reset.get("online_lm_calls") != 0
        or reset.get("physical_rng_draws") != 0
    ):
        raise LaunchRefused("C211 immutable tape semantics differ")

    manifest = documents["action_manifest"]
    actions = manifest.get("actions")
    action = actions[0] if type(actions) is list and len(actions) == 1 else None
    if (
        manifest.get("schema_version") != 3
        or manifest.get("action_order") != [action_id]
        or manifest.get("mobility_mode") != "no_move"
        or type(action) is not dict
        or action.get("action_id") != action_id
        or action.get("action_uid") != action_uid
        or action.get("motion_path") != pins["motion"]["path"]
        or action.get("motion_sha256") != pins["motion"]["sha256"]
    ):
        raise LaunchRefused("C211 action manifest closure differs")

    for key, nominal in (
        ("dynamic_ready_artifact", False),
        ("dynamic_ready_nominal_receipt", True),
    ):
        dynamic = documents[key]
        dynamic_unsigned = dict(dynamic)
        dynamic_seal = dynamic_unsigned.pop("content_sha256", None)
        if (
            _B._sha256(dynamic_seal, name="C211 %s content SHA" % key)
            != canonical_sha256(dynamic_unsigned)
            or dynamic.get("action_id") != action_id
            or dynamic.get("motion_sha256") != pins["motion"]["sha256"]
            or (nominal and dynamic.get("verdict") != "PASS")
            or (
                not nominal
                and dynamic.get("kind")
                != "agibot_a3_action_dynamic_ready_candidate_v2"
            )
        ):
            raise LaunchRefused("C211 %s closure differs" % key)
    artifact = _sealed_row(
        documents["frame0_exact_artifact"],
        (
            "schema_version", "kind", "diagnostic_unauthorized", "action_id",
            "source_kind", "motion_sha256", "task_close_ticks", "policy_dt_s",
            "wait_schedule_canonical_sha256",
        ),
        name="C211 frame0-exact artifact",
    )
    receipt = _sealed_row(
        documents["frame0_exact_receipt"],
        (
            "schema_version", "kind", "diagnostic_unauthorized", "verdict",
            "source_kind", "action_id", "motion_sha256", "artifact_file_sha256",
            "artifact_content_sha256", "artifact_source_commit",
            "task_close_ticks", "policy_dt_s",
            "wait_schedule_canonical_sha256",
        ),
        name="C211 frame0-exact receipt",
    )
    task_close_ticks = artifact["task_close_ticks"]
    if (
        artifact["schema_version"] != 1
        or artifact["kind"] != FRAME0_EXACT_ARTIFACT_KIND
        or artifact["diagnostic_unauthorized"] is not True
        or artifact["source_kind"] != FRAME0_EXACT_SOURCE_KIND
        or artifact["action_id"] != action_id
        or artifact["motion_sha256"] != pins["motion"]["sha256"]
        or type(task_close_ticks) is not int
        or not 1 <= task_close_ticks <= WAIT_SCHEDULE["required_active_ticks"]
        or artifact["policy_dt_s"] != POLICY_DT_S
        or artifact["wait_schedule_canonical_sha256"]
        != WAIT_SCHEDULE["canonical_sha256"]
    ):
        raise LaunchRefused("C211 frame0-exact artifact binding differs")
    artifact_content_sha = artifact["content_sha256"]
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != FRAME0_EXACT_RECEIPT_KIND
        or receipt["diagnostic_unauthorized"] is not True
        or receipt["source_kind"] != FRAME0_EXACT_SOURCE_KIND
        or receipt["verdict"] != "PASS"
        or receipt["action_id"] != action_id
        or receipt["motion_sha256"] != pins["motion"]["sha256"]
        or receipt["artifact_file_sha256"]
        != pins["frame0_exact_artifact"]["sha256"]
        or receipt["artifact_content_sha256"] != artifact_content_sha
        or receipt["task_close_ticks"] != task_close_ticks
        or receipt["policy_dt_s"] != POLICY_DT_S
        or receipt["wait_schedule_canonical_sha256"]
        != WAIT_SCHEDULE["canonical_sha256"]
    ):
        raise LaunchRefused("C211 frame0-exact receipt binding differs")
    artifact_source_commit = receipt["artifact_source_commit"]
    if (
        type(artifact_source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", artifact_source_commit) is None
    ):
        raise LaunchRefused("C211 frame0-exact artifact source commit is malformed")
    _verify_frame0_artifact_source_commit(
        checkout, artifact_source_commit, pins["frame0_exact_artifact"]
    )
    return {
        **expected,
        "action_id": action_id,
        "action_uid": action_uid,
        "teacher_id": teacher_id,
        "seed": seed,
        **pins,
        "tape_canonical_sha256": tape_seal,
        "tape_question_sha256": canonical_sha256(question),
        "frame0_exact_artifact_content_sha256": artifact_content_sha,
        "frame0_exact_receipt_content_sha256": receipt["content_sha256"],
        "artifact": pin,
        "lineage_sha256": pin["sha256"],
    }


def _planned_materialization(
    *, recipe: Mapping[str, Any], lineage: Mapping[str, Any]
) -> dict[str, Any]:
    reward = {
        "question_contract": _question_contract(),
        "soft_weights": recipe["soft_weights"],
        "reference_guard_mode": recipe["reference_guard_mode"],
        "c211_reward_contract": _c211_reward_contract(),
        "required_positive_outcome_terms": list(REQUIRED_OUTCOME_TERMS),
        "desired_contact_reward_terms_prohibited": True,
        "duplicate_predicted_outcome_terms_prohibited": list(
            PROHIBITED_DUPLICATE_OUTCOME_TERMS
        ),
    }
    unsigned = {
        "schema_version": 1,
        "kind": MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "recipe_id": RECIPE_ID,
        "lineage_sha256": lineage["lineage_sha256"],
        "recipe_contract_sha256": recipe["recipe_contract_sha256"],
        "reward_contract_sha256": canonical_sha256(reward),
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _sealed_row(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    row = _exact_dict(value, tuple(keys) + ("content_sha256",), name=name)
    unsigned = dict(row)
    seal = unsigned.pop("content_sha256")
    if _B._sha256(seal, name="%s content SHA" % name) != canonical_sha256(unsigned):
        raise LaunchRefused("%s content seal differs" % name)
    return row


def _validated_stage_result(
    value: Any, *, expected_stage: str, name: str
) -> tuple[dict[str, str], dict[str, Any]]:
    pin, row = _canonical_external_json(value, name=name)
    row = _sealed_row(
        row,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "accepted",
            "launch_claim_sha256",
            "stage",
            "namespace",
            "completion",
            "gpu_admission",
            "output_contract",
            "reward_materialization",
            "policy_recipe_materialization",
            "oracle32_receipt",
            "predecessor_result",
        ),
        name=name,
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != RESULT_KIND
        or row["diagnostic_unauthorized"] is not True
        or row["accepted"] is not True
        or row["stage"] != expected_stage
        or _B._sha256(row["launch_claim_sha256"], name="%s claim SHA" % name)
        != row["launch_claim_sha256"]
        or type(row["namespace"]) is not str
        or not row["namespace"]
    ):
        raise LaunchRefused("%s identity differs" % name)
    expected_completion = {
        "completion_exit_code": "0",
        "terminal_kind": "clean_completion",
        "terminal_exit_code": "0",
    }
    budget = BUDGETS[expected_stage]
    if (
        expected_stage in ("materialize", "recipe", "oracle32", "scale4096")
        and (
            row["completion"] != expected_completion
            or type(row["output_contract"]) is not dict
            or row["output_contract"].get("ppo_update_count") != budget[1]
            or row["output_contract"].get("finite_model_save_interval") != budget[2]
        )
    ):
        raise LaunchRefused("%s lacks its exact finite natural-exit receipt" % name)
    return pin, row


def _validate_materialization(
    value: Any, *, recipe: Mapping[str, Any], lineage: Mapping[str, Any]
) -> dict[str, Any]:
    pin, result = _validated_stage_result(
        value, expected_stage="materialize", name="C211 materialize result"
    )
    if (
        result["policy_recipe_materialization"] is not None
        or result["oracle32_receipt"] is not None
        or result["predecessor_result"] is not None
    ):
        raise LaunchRefused("C211 materialize result contains downstream receipts")
    row = _sealed_row(
        result["reward_materialization"],
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "recipe_id",
            "lineage_sha256",
            "recipe_contract_sha256",
            "reward_contract_sha256",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "trainability_contract",
            "runtime_effective_reward_artifact",
            "runtime_effective_reward_sha256",
            "runtime_effective_reward_term_count",
            "runtime_soft_weights",
        ),
        name="C211 reward materialization",
    )
    planned = _planned_materialization(recipe=recipe, lineage=lineage)
    for key in (
        "schema_version",
        "kind",
        "diagnostic_unauthorized",
        "recipe_id",
        "lineage_sha256",
        "recipe_contract_sha256",
        "reward_contract_sha256",
        "actor_contract",
        "actor_width",
        "critic_contract",
        "critic_width",
        "trainability_contract",
    ):
        if row[key] != planned[key]:
            raise LaunchRefused("C211 reward materialization binding differs")
    artifact, _artifact_path = _external_pin(
        row["runtime_effective_reward_artifact"], name="runtime reward artifact"
    )
    if (
        _B._sha256(
            row["runtime_effective_reward_sha256"], name="runtime reward SHA"
        )
        != row["runtime_effective_reward_sha256"]
        or type(row["runtime_effective_reward_term_count"]) is not int
        or row["runtime_effective_reward_term_count"] <= 0
    ):
        raise LaunchRefused("C211 runtime reward materialization differs")
    expected_weights = {
        "death_penalty": recipe["soft_weights"]["death_penalty"],
        "joint_limit": recipe["soft_weights"]["joint_limit"],
        "qdes_limit_barrier": recipe["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": recipe["soft_weights"]["qdes_projection"],
    }
    if row["runtime_soft_weights"] != expected_weights:
        raise LaunchRefused("C211 runtime reward soft weights differ")
    return {
        "materialize_result": pin,
        **{key: row[key] for key in row if key != "runtime_effective_reward_artifact"},
        "runtime_effective_reward_artifact": dict(artifact),
    }


def _validate_policy_materialization(
    value: Any,
    *,
    recipe: Mapping[str, Any],
    lineage: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    pin, result = _validated_stage_result(
        value, expected_stage="recipe", name="C211 recipe result"
    )
    if (
        result["oracle32_receipt"] is not None
        or result["predecessor_result"] is not None
        or type(result["reward_materialization"]) is not dict
        or result["reward_materialization"].get("content_sha256")
        != materialization["content_sha256"]
    ):
        raise LaunchRefused("C211 recipe result reward lineage differs")
    row = _sealed_row(
        result["policy_recipe_materialization"],
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "recipe_id",
            "lineage_sha256",
            "recipe_contract_sha256",
            "runtime_policy_recipe_artifact",
            "runtime_policy_recipe_sha256",
            "dynamic_ready_binding_sha256",
            "noise_std_type",
            "configured_and_realized_init_noise_std",
        ),
        name="C211 policy materialization",
    )
    expected = {
        "schema_version": 1,
        "kind": POLICY_MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "recipe_id": RECIPE_ID,
        "lineage_sha256": lineage["lineage_sha256"],
        "recipe_contract_sha256": recipe["recipe_contract_sha256"],
        "noise_std_type": recipe["noise_std_type"],
        "configured_and_realized_init_noise_std": recipe["init_noise_std"],
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("C211 policy materialization binding differs")
    artifact, _artifact_path = _external_pin(
        row["runtime_policy_recipe_artifact"], name="runtime policy artifact"
    )
    for key in (
        "runtime_policy_recipe_sha256",
        "dynamic_ready_binding_sha256",
    ):
        if _B._sha256(row[key], name=key) != row[key]:
            raise LaunchRefused("C211 policy materialization SHA differs")
    return {
        "recipe_result": pin,
        **row,
        "runtime_policy_recipe_artifact": dict(artifact),
    }


def _finite_vec3(value: Any, *, name: str) -> list[float | int]:
    if (
        type(value) is not list
        or len(value) != 3
        or any(
            type(component) not in (int, float)
            or not math.isfinite(float(component))
            for component in value
        )
    ):
        raise LaunchRefused("%s must be a finite three-vector" % name)
    return list(value)


def _validate_c211_hard_contract(
    value: Any,
    *,
    checkout: Path,
    oracle_namespace: Path,
    lineage: Mapping[str, Any],
) -> dict[str, str]:
    pin, path = _external_pin(value, name="C211 hard training contract")
    expected_path = oracle_namespace / "params" / "training_contract.json"
    if path.resolve(strict=True) != expected_path.resolve(strict=True):
        raise LaunchRefused("C211 hard training contract path differs")
    contract = _B._strict_json_bytes(
        path.read_bytes(), name="C211 hard training contract"
    )
    if type(contract) is not dict:
        raise LaunchRefused("C211 hard training contract must be a JSON object")
    try:
        module = _OLD._load_training_contract_module(checkout)
        module.validate_schema3_contract_structure(contract)
        authorized = module.validate_action_ball_training_authorization(contract)
    except LaunchRefused:
        raise
    except Exception as exc:
        raise LaunchRefused(
            "C211 hard training contract validation failed: %s" % exc
        ) from exc
    if authorized is not True:
        raise LaunchRefused("C211 hard training contract is not action-ball authorized")
    expected = {
        "schema_version": 3,
        "target_mode": "action_ball",
        "actor_obs_contract": ACTOR_CONTRACT,
        "actor_obs_total_dim": ACTOR_WIDTH,
        "critic_obs_contract": CRITIC_CONTRACT,
        "critic_obs_total_dim": CRITIC_WIDTH,
        "actor_obs_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_obs_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
        "task_valid_required": True,
        "task_wait_contract": _hard_wait_contract(),
        "question_source_contract": _hard_question_source_contract(),
        "contact_target_absent": True,
        # The shared schema kept this historical field spelling when C225 was
        # retired.  Its value identity is nevertheless the C211 reward-v2
        # contract and is revalidated in full below.
        "c225_reward_contract": _c211_reward_contract(),
    }
    if any(contract.get(key) != wanted for key, wanted in expected.items()):
        raise LaunchRefused("C211 hard training contract ABI differs")
    actor_names = contract.get("actor_obs_term_names")
    actor_dims = contract.get("actor_obs_term_dims")
    critic_names = contract.get("critic_obs_term_names")
    critic_dims = contract.get("critic_obs_term_dims")
    if (
        type(actor_names) is not list
        or type(actor_dims) is not list
        or type(critic_names) is not list
        or type(critic_dims) is not list
        or actor_names[9:12] != list(INCOMING_BALL_FIELDS)
        or actor_dims[9:12] != [3, 3, 3]
        or critic_names[11:14] != list(INCOMING_BALL_FIELDS)
        or critic_dims[11:14] != [3, 3, 3]
        or actor_names[-1:] != ["task_valid"]
        or actor_dims[-1:] != [1]
        or critic_names[-1:] != ["task_valid"]
        or critic_dims[-1:] != [1]
    ):
        raise LaunchRefused("C211 hard training contract incoming-ball layout differs")
    try:
        target = contract["action_ball_training"]["runtime"]["target_provider"]
        tape = target["immutable_tape"]
    except (KeyError, TypeError) as exc:
        raise LaunchRefused("C211 hard training contract target provider is missing") from exc
    expected_tape_path = checkout / lineage["immutable_tape"]["path"]
    try:
        observed_tape_path = Path(str(tape.get("path", ""))).resolve(strict=True)
        expected_tape_path = expected_tape_path.resolve(strict=True)
    except (AttributeError, OSError) as exc:
        raise LaunchRefused("C211 hard training contract tape path is invalid") from exc
    if (
        type(target) is not dict
        or target.get("source") != "immutable_tape"
        or target.get("recipe") != TARGET_RECIPE
        or target.get("validity_mask") != list(TARGET_VALIDITY_MASK)
        or target.get("target_observation_noise") is not False
        or target.get("actor_width_unchanged") is not True
        or target.get("critic_width_unchanged") is not True
        or type(tape) is not dict
        or observed_tape_path != expected_tape_path
        or tape.get("file_sha256") != lineage["immutable_tape"]["sha256"]
        or tape.get("canonical_sha256") != lineage["tape_canonical_sha256"]
        or tape.get("base_question_sha256") != lineage["tape_question_sha256"]
        or tape.get("online_lm_calls") != 0
        or tape.get("physical_rng_draws") != 0
    ):
        raise LaunchRefused("C211 hard training contract 000/tape contract differs")
    return pin


def _validate_c211_runner_preflight(
    value: Any,
    *,
    oracle_namespace: Path,
    launch_claim_sha256: str,
    hard_contract_sha256: str,
) -> dict[str, Any]:
    pin, document = _canonical_external_json(
        value, name="C211 runner preflight evidence"
    )
    path = Path(pin["path"])
    expected_path = oracle_namespace / "params" / "c211_runner_preflight.json"
    if path.resolve(strict=True) != expected_path.resolve(strict=True):
        raise LaunchRefused("C211 runner preflight evidence path differs")
    row = _sealed_row(
        document,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "oracle_launch_claim_sha256",
            "hard_contract_sha256",
            "marker",
            "facts",
        ),
        name="C211 runner preflight evidence",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != C211_RUNNER_PREFLIGHT_KIND
        or row["diagnostic_unauthorized"] is not True
        or row["oracle_launch_claim_sha256"] != launch_claim_sha256
        or row["hard_contract_sha256"] != hard_contract_sha256
        or row["marker"] != "ACTION_BALL_C211_TRAINABILITY_PREFLIGHT_JSON"
    ):
        raise LaunchRefused("C211 runner preflight binding differs")
    facts = _exact_dict(
        row["facts"],
        (
            "trainability_contract",
            "actor_contract",
            "critic_contract",
            "actor_width",
            "critic_width",
            "actor_normalizer_identity",
            "critic_normalizer_identity",
            "fresh_normalizers_required",
            "symmetric_critic_fallback_forbidden",
            "contact_target_absent",
            "c225_reward_contract",
            "runner_actor_width",
            "runner_critic_width",
            "actor_normalizer_attribute",
            "critic_normalizer_attribute",
        ),
        name="C211 runner preflight facts",
    )
    expected = {
        "trainability_contract": TRAINABILITY_CONTRACT,
        "actor_contract": ACTOR_CONTRACT,
        "critic_contract": CRITIC_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_width": CRITIC_WIDTH,
        "actor_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
        "contact_target_absent": True,
        "c225_reward_contract": _c211_reward_contract(),
        "runner_actor_width": ACTOR_WIDTH,
        "runner_critic_width": CRITIC_WIDTH,
    }
    if any(facts[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("C211 runner preflight ABI differs")
    actor_attr = facts["actor_normalizer_attribute"]
    critic_attr = facts["critic_normalizer_attribute"]
    if (
        type(actor_attr) is not str
        or not actor_attr
        or type(critic_attr) is not str
        or not critic_attr
        or actor_attr == critic_attr
    ):
        raise LaunchRefused("C211 runner preflight normalizer proof differs")
    return {"artifact": pin, "content_sha256": row["content_sha256"]}


def _validate_c211_selected_rubber(
    value: Any,
    *,
    oracle_namespace: Path,
    launch_claim_sha256: str,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    pin, document = _canonical_external_json(
        value, name="C211 selected-rubber contact evidence"
    )
    path = Path(pin["path"])
    expected_path = oracle_namespace / "params" / "c211_selected_rubber_contact.json"
    if path.resolve(strict=True) != expected_path.resolve(strict=True):
        raise LaunchRefused("C211 selected-rubber evidence path differs")
    row = _sealed_row(
        document,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "oracle_launch_claim_sha256",
            "action_id",
            "action_uid",
            "motion_sha256",
            "classifier_contract",
            "classifier_source_sha256",
            "geometry_authority_sha256",
            "denominator_kind",
            "episodes",
        ),
        name="C211 selected-rubber contact evidence",
    )
    for key in ("classifier_source_sha256", "geometry_authority_sha256"):
        if (
            _B._sha256(row[key], name="C211 selected-rubber %s" % key)
            != row[key]
            or row[key] == "0" * 64
        ):
            raise LaunchRefused("C211 selected-rubber authority SHA differs")
    if (
        row["schema_version"] != 1
        or row["kind"] != C211_SELECTED_RUBBER_KIND
        or row["diagnostic_unauthorized"] is not True
        or row["oracle_launch_claim_sha256"] != launch_claim_sha256
        or row["action_id"] != lineage["action_id"]
        or row["action_uid"] != lineage["action_uid"]
        or row["motion_sha256"] != lineage["motion"]["sha256"]
        or row["classifier_contract"]
        != "runtime_contact_pair_selected_rubber_v1"
        or row["denominator_kind"] != "eligible_closed_swings"
    ):
        raise LaunchRefused("C211 selected-rubber evidence binding differs")
    episodes = row["episodes"]
    if type(episodes) is not list or len(episodes) != 32:
        raise LaunchRefused("C211 selected-rubber evidence must contain 32 rows")
    buckets = {
        name: 0
        for name in (
            "selected_rubber",
            "no_contact",
            "wrong_surface",
            "edge_or_rim_ambiguous",
            "between_planes_ambiguous",
            "unknown",
        )
    }
    row_sha256 = []
    for index, episode in enumerate(episodes):
        item = _exact_dict(
            episode,
            (
                "episode",
                "eligible_closed_swing",
                "classification",
                "contact_evidence_sha256",
            ),
            name="C211 selected-rubber episode",
        )
        classification = item["classification"]
        evidence_sha = _B._sha256(
            item["contact_evidence_sha256"],
            name="C211 selected-rubber contact evidence SHA",
        )
        if (
            item["episode"] != index
            or item["eligible_closed_swing"] is not True
            or classification not in buckets
            or evidence_sha == "0" * 64
        ):
            raise LaunchRefused("C211 selected-rubber episode evidence differs")
        buckets[classification] += 1
        row_sha256.append(canonical_sha256(item))
    if buckets != {
        "selected_rubber": 32,
        "no_contact": 0,
        "wrong_surface": 0,
        "edge_or_rim_ambiguous": 0,
        "between_planes_ambiguous": 0,
        "unknown": 0,
    }:
        raise LaunchRefused("C211 selected-rubber denominator did not pass")
    return {
        "artifact": pin,
        "content_sha256": row["content_sha256"],
        "row_sha256": row_sha256,
        "eligible_episode_denominator": len(episodes),
        "actual_selected_rubber_contact_count": buckets["selected_rubber"],
    }


def _validate_c211_raw_oracle(
    path: Path,
    *,
    checkout: Path,
    oracle_namespace: Path,
    launch_claim_sha256: str,
    recipe: Mapping[str, Any],
    lineage: Mapping[str, Any],
    materialization: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse C-owned evidence and derive PASS; the producer has no verdict field."""

    _B._stable_regular_file(path, name="C211 raw oracle evidence")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="C211 raw oracle evidence")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("C211 raw oracle evidence must be canonical JSON plus newline")
    row = _sealed_row(
        document,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "bindings",
            "training_contract_artifact",
            "runner_preflight_artifact",
            "question_contract",
            "completion",
            "episodes",
            "desired_contact_metrics",
            "termination",
            "safety",
            "selected_rubber_contact_artifact",
            "teacher_qdes",
        ),
        name="C211 raw oracle evidence",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != C211_RAW_ORACLE_KIND
        or row["diagnostic_unauthorized"] is not True
    ):
        raise LaunchRefused("C211 raw oracle evidence schema differs")

    hard_contract = _validate_c211_hard_contract(
        row["training_contract_artifact"],
        checkout=checkout,
        oracle_namespace=oracle_namespace,
        lineage=lineage,
    )
    bindings = _exact_dict(
        row["bindings"],
        (
            "oracle_launch_claim_sha256",
            "lineage_sha256",
            "recipe_contract_sha256",
            "reward_contract_sha256",
            "runtime_effective_reward_sha256",
            "runtime_policy_recipe_sha256",
            "hard_contract_sha256",
            "motion_sha256",
            "manifest_sha256",
            "tape_file_sha256",
            "tape_canonical_sha256",
            "tape_base_question_sha256",
            "dynamic_ready_artifact_sha256",
            "dynamic_ready_nominal_receipt_sha256",
        ),
        name="C211 raw oracle bindings",
    )
    expected_bindings = {
        "oracle_launch_claim_sha256": launch_claim_sha256,
        "lineage_sha256": lineage["lineage_sha256"],
        "recipe_contract_sha256": recipe["recipe_contract_sha256"],
        "reward_contract_sha256": materialization["reward_contract_sha256"],
        "runtime_effective_reward_sha256": materialization[
            "runtime_effective_reward_sha256"
        ],
        "runtime_policy_recipe_sha256": policy[
            "runtime_policy_recipe_sha256"
        ],
        "hard_contract_sha256": hard_contract["sha256"],
        "motion_sha256": lineage["motion"]["sha256"],
        "manifest_sha256": lineage["action_manifest"]["sha256"],
        "tape_file_sha256": lineage["immutable_tape"]["sha256"],
        "tape_canonical_sha256": lineage["tape_canonical_sha256"],
        "tape_base_question_sha256": lineage["tape_question_sha256"],
        "dynamic_ready_artifact_sha256": lineage[
            "dynamic_ready_artifact"
        ]["sha256"],
        "dynamic_ready_nominal_receipt_sha256": lineage[
            "dynamic_ready_nominal_receipt"
        ]["sha256"],
    }
    if any(bindings[key] != wanted for key, wanted in expected_bindings.items()):
        raise LaunchRefused("C211 raw oracle evidence lineage binding differs")
    for key, value in bindings.items():
        if _B._sha256(value, name="C211 raw oracle %s" % key) != value:
            raise LaunchRefused("C211 raw oracle binding SHA differs")

    _validate_c211_runner_preflight(
        row["runner_preflight_artifact"],
        oracle_namespace=oracle_namespace,
        launch_claim_sha256=launch_claim_sha256,
        hard_contract_sha256=hard_contract["sha256"],
    )
    selected = _validate_c211_selected_rubber(
        row["selected_rubber_contact_artifact"],
        oracle_namespace=oracle_namespace,
        launch_claim_sha256=launch_claim_sha256,
        lineage=lineage,
    )

    question = _exact_dict(
        row["question_contract"],
        (
            "target_source",
            "target_recipe",
            "target_validity_mask",
            "target_observation_noise",
            "incoming_ball_fields",
            "reset_inverse_solve",
            "online_solver_calls",
            "online_lm_calls",
            "physical_rng_draws",
            "immutable_tape_file_sha256",
            "immutable_tape_canonical_sha256",
            "immutable_tape_base_question_sha256",
        ),
        name="C211 raw oracle question contract",
    )
    expected_question = {
        "target_source": "immutable_tape",
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "target_observation_noise": False,
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
        "immutable_tape_file_sha256": lineage["immutable_tape"]["sha256"],
        "immutable_tape_canonical_sha256": lineage["tape_canonical_sha256"],
        "immutable_tape_base_question_sha256": lineage["tape_question_sha256"],
    }
    if question != expected_question:
        raise LaunchRefused("C211 raw oracle 000/tape/counter contract differs")

    desired = _exact_dict(
        row["desired_contact_metrics"], ("status", "reason"),
        name="C211 raw oracle desired-contact metrics",
    )
    if desired != {
        "status": "INELIGIBLE",
        "reason": "target_validity_000_contact_target_absent",
    }:
        raise LaunchRefused("C211 000 desired-contact metrics must be ineligible")

    completion = _exact_dict(
        row["completion"],
        ("requested", "terminal", "single_stroke", "control_steps"),
        name="C211 raw oracle completion",
    )
    control_steps = completion["control_steps"]
    if (
        completion["requested"] != 32
        or completion["terminal"] != 32
        or completion["single_stroke"] != 32
        or type(control_steps) is not int
        or control_steps < 32
    ):
        raise LaunchRefused("C211 raw oracle did not complete 32 single strokes")

    episodes = row["episodes"]
    if type(episodes) is not list or len(episodes) != 32:
        raise LaunchRefused("C211 raw oracle episode ledger must contain 32 rows")
    episode_step_sum = 0
    for index, episode in enumerate(episodes):
        item = _exact_dict(
            episode,
            (
                "episode",
                "control_steps",
                "terminal_phase",
                "termination_reasons",
                "tape_question_index",
                "tape_question_sha256",
                "incoming_ball_observation",
                "selected_rubber_evidence_sha256",
            ),
            name="C211 raw oracle episode",
        )
        incoming = _exact_dict(
            item["incoming_ball_observation"],
            ("source", "actor", "critic"),
            name="C211 incoming-ball observation",
        )
        actor = _exact_dict(
            incoming["actor"], INCOMING_BALL_FIELDS,
            name="C211 actor incoming-ball observation",
        )
        critic = _exact_dict(
            incoming["critic"], INCOMING_BALL_FIELDS,
            name="C211 critic incoming-ball observation",
        )
        for field in INCOMING_BALL_FIELDS:
            _finite_vec3(actor[field], name="C211 actor %s" % field)
            _finite_vec3(critic[field], name="C211 critic %s" % field)
        if (
            item["episode"] != index
            or type(item["control_steps"]) is not int
            or item["control_steps"] <= 0
            or item["terminal_phase"] != "post_strike"
            or item["termination_reasons"]
            != ["action_ball_single_stroke_complete"]
            or item["tape_question_index"] != 0
            or item["tape_question_sha256"] != lineage["tape_question_sha256"]
            or incoming["source"]
            != "runtime_actor_and_critic_observation_terms"
            or actor != critic
            or item["selected_rubber_evidence_sha256"]
            != selected["row_sha256"][index]
        ):
            raise LaunchRefused("C211 raw oracle episode evidence differs")
        episode_step_sum += item["control_steps"]
    if episode_step_sum != control_steps:
        raise LaunchRefused("C211 raw oracle episode/control-step ledger differs")

    termination = _exact_dict(
        row["termination"],
        ("allowed_reason", "by_reason", "unexpected_by_reason", "phase_by_reason"),
        name="C211 raw oracle termination",
    )
    if termination != {
        "allowed_reason": "action_ball_single_stroke_complete",
        "by_reason": {"action_ball_single_stroke_complete": 32},
        "unexpected_by_reason": {},
        "phase_by_reason": {
            "post_strike": {"action_ball_single_stroke_complete": 32},
            "pre_strike_or_same_step_unknown": {},
        },
    }:
        raise LaunchRefused("C211 raw oracle termination ledger differs")

    safety = _exact_dict(
        row["safety"],
        (
            "control_step_denominator",
            "hard_termination_by_reason",
            "robot_table_contact_count",
            "projection_nonfinite_count",
            "projection_observed_sample_count",
            "qdes_observed_sample_count",
            "actual_observed_sample_count",
            "reference_guard_sample_count",
        ),
        name="C211 raw oracle safety",
    )
    hard = _exact_dict(
        safety["hard_termination_by_reason"], HARD_TERMINATION_UNION,
        name="C211 raw oracle hard termination ledger",
    )
    if (
        safety["control_step_denominator"] != control_steps
        or any(hard[name] != 0 for name in HARD_TERMINATION_UNION)
        or safety["robot_table_contact_count"] != 0
        or safety["projection_nonfinite_count"] != 0
        or any(
            safety[key] != control_steps
            for key in (
                "projection_observed_sample_count",
                "qdes_observed_sample_count",
                "actual_observed_sample_count",
                "reference_guard_sample_count",
            )
        )
    ):
        raise LaunchRefused("C211 raw oracle safety denominator differs")

    teacher = _exact_dict(
        row["teacher_qdes"],
        (
            "control_step_denominator",
            "preclamp_max_abs_error_rad",
            "teleport_used",
        ),
        name="C211 raw oracle teacher qdes",
    )
    error = teacher["preclamp_max_abs_error_rad"]
    if (
        teacher["control_step_denominator"] != control_steps
        or type(error) not in (int, float)
        or not math.isfinite(float(error))
        or error < 0.0
        or error > 2.0e-6
        or teacher["teleport_used"] is not False
    ):
        raise LaunchRefused("C211 raw oracle teacher-qdes evidence differs")
    return {
        "kind": C211_RAW_ORACLE_KIND,
        "content_sha256": row["content_sha256"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "control_steps": control_steps,
        "selected_rubber_episode_denominator": selected[
            "eligible_episode_denominator"
        ],
        "actual_selected_rubber_contact_count": selected[
            "actual_selected_rubber_contact_count"
        ],
    }


def _validate_oracle32(
    value: Any,
    *,
    checkout: Path,
    recipe: Mapping[str, Any],
    lineage: Mapping[str, Any],
    materialization: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    pin, result = _validated_stage_result(
        value, expected_stage="oracle32", name="C211 oracle32 result"
    )
    if (
        result["predecessor_result"] is not None
        or type(result["reward_materialization"]) is not dict
        or result["reward_materialization"].get("content_sha256")
        != materialization["content_sha256"]
        or type(result["policy_recipe_materialization"]) is not dict
        or result["policy_recipe_materialization"].get("content_sha256")
        != policy["content_sha256"]
    ):
        raise LaunchRefused("C211 oracle32 result lineage differs")
    row = _sealed_row(
        result["oracle32_receipt"],
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "verdict",
            "episodes",
            "recipe_id",
            "lineage_sha256",
            "recipe_contract_sha256",
            "reward_contract_sha256",
            "runtime_effective_reward_sha256",
            "runtime_policy_recipe_sha256",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "trainability_contract",
            "target_recipe",
            "target_validity_mask",
            "incoming_ball_fields",
            "reset_inverse_solve",
            "online_solver_calls",
            "online_lm_calls",
            "physical_rng_draws",
            "seed",
            "raw_oracle_artifact",
            "raw_oracle_kind",
            "raw_oracle_content_sha256",
            "control_step_denominator",
            "selected_rubber_episode_denominator",
            "actual_selected_rubber_contact_count",
        ),
        name="C211 oracle32 receipt",
    )
    expected = {
        "schema_version": 1,
        "kind": ORACLE32_KIND,
        "diagnostic_unauthorized": True,
        "verdict": "PASS",
        "episodes": 32,
        "recipe_id": RECIPE_ID,
        "lineage_sha256": lineage["lineage_sha256"],
        "recipe_contract_sha256": recipe["recipe_contract_sha256"],
        "reward_contract_sha256": materialization["reward_contract_sha256"],
        "runtime_effective_reward_sha256": materialization[
            "runtime_effective_reward_sha256"
        ],
        "runtime_policy_recipe_sha256": policy["runtime_policy_recipe_sha256"],
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "physical_rng_draws": 0,
        "seed": lineage["seed"],
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("C211 oracle32 binding differs")
    artifact, _artifact_path = _external_pin(
        row["raw_oracle_artifact"], name="raw oracle artifact"
    )
    oracle_namespace = _B._absolute_path(
        result["namespace"], name="C211 oracle32 namespace", must_exist=True
    )
    expected_raw_path = oracle_namespace / "teacher_qdes_oracle_32ep.json"
    if _artifact_path.resolve(strict=True) != expected_raw_path.resolve(strict=True):
        raise LaunchRefused("C211 raw oracle artifact path differs")
    raw_facts = _validate_c211_raw_oracle(
        _artifact_path,
        checkout=checkout,
        oracle_namespace=oracle_namespace,
        launch_claim_sha256=result["launch_claim_sha256"],
        recipe=recipe,
        lineage=lineage,
        materialization=materialization,
        policy=policy,
    )
    expected_raw = {
        "raw_oracle_kind": raw_facts["kind"],
        "raw_oracle_content_sha256": raw_facts["content_sha256"],
        "control_step_denominator": raw_facts["control_steps"],
        "selected_rubber_episode_denominator": raw_facts[
            "selected_rubber_episode_denominator"
        ],
        "actual_selected_rubber_contact_count": raw_facts[
            "actual_selected_rubber_contact_count"
        ],
    }
    if any(row[key] != wanted for key, wanted in expected_raw.items()):
        raise LaunchRefused("C211 oracle32 receipt differs from parsed raw evidence")
    return {
        "oracle32_result": pin,
        **row,
        "raw_oracle_artifact": dict(artifact),
    }


def _validate_scale_predecessor(
    value: Any,
    *,
    materialization: Mapping[str, Any],
    policy: Mapping[str, Any],
    oracle32: Mapping[str, Any],
) -> dict[str, Any]:
    pin, result = _validated_stage_result(
        value, expected_stage="scale4096", name="C211 scale4096 result"
    )
    expected_completion = {
        "completion_exit_code": "0",
        "terminal_kind": "clean_completion",
        "terminal_exit_code": "0",
    }
    if (
        result["completion"] != expected_completion
        or type(result["reward_materialization"]) is not dict
        or result["reward_materialization"].get("content_sha256")
        != materialization["content_sha256"]
        or type(result["policy_recipe_materialization"]) is not dict
        or result["policy_recipe_materialization"].get("content_sha256")
        != policy["content_sha256"]
        or type(result["oracle32_receipt"]) is not dict
        or result["oracle32_receipt"].get("content_sha256")
        != oracle32["content_sha256"]
        or result["predecessor_result"] is not None
        or type(result["output_contract"]) is not dict
        or result["output_contract"].get("ppo_update_count") != 5
        or result["output_contract"].get("finite_model_save_interval") != 1
    ):
        raise LaunchRefused(
            "C211 long4096 requires the exact finite natural-exit scale4096 result"
        )
    return {
        "artifact": pin,
        "stage": "scale4096",
        "launch_claim_sha256": result["launch_claim_sha256"],
        "content_sha256": result["content_sha256"],
        "completion": expected_completion,
    }


def _validate_gpu(value: Any, *, allow_colocation: bool) -> dict[str, Any]:
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
        or owner.lower() in {"codex", "claude", "fable", "agent", "unassigned"}
    ):
        raise LaunchRefused("spec.gpu.owner must be an explicit human name")
    lock_path = _B._absolute_path(row["lock_path"], name="spec.gpu.lock_path")
    expected_lock = Path("/tmp/hope_lean_queue_gpu%d.lock" % index)
    if lock_path != expected_lock:
        raise LaunchRefused("spec.gpu.lock_path must be %s" % expected_lock)
    expected_empty = not allow_colocation
    if row["require_empty"] is not expected_empty:
        raise LaunchRefused("spec.gpu.require_empty differs from colocation policy")
    return {
        "index": index,
        "uuid": uuid,
        "owner": owner,
        "lock_path": str(lock_path),
        "require_empty": expected_empty,
    }


def _isaac_python_entry(value: Any) -> Path:
    entry = _B._absolute_path(value, name="source.isaac_python", must_exist=True)
    try:
        real = entry.resolve(strict=True)
        info = real.stat()
    except OSError as exc:
        raise LaunchRefused("source.isaac_python cannot resolve to a real file") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(real, os.X_OK):
        raise LaunchRefused(
            "source.isaac_python must resolve to an executable regular file"
        )
    return entry


def _validate_spec(document: Any, *, claimed: bool = False) -> dict[str, Any]:
    required = frozenset(
        (
            "schema_version",
            "kind",
            "source",
            "recipe_id",
            "lineage",
            "materialization_result",
            "recipe_result",
            "oracle32_result",
            "predecessor_result",
            "stage",
            "num_envs",
            "max_iterations",
            "save_interval",
            "wait_contract",
            "frame0_exact_artifact_sha256",
            "frame0_exact_receipt_sha256",
            "gpu",
            "namespace",
            "log_path",
        )
    )
    actual = frozenset(document) if type(document) is dict else frozenset()
    optional = frozenset((COLOCATION_SPEC_KEY,))
    if not required.issubset(actual) or not actual.issubset(required | optional):
        raise LaunchRefused(
            "C211 launch spec keys differ: missing=%s extra=%s"
            % (sorted(required - actual), sorted(actual - required - optional))
        )
    row = dict(document)
    _assert_c211_only(row, name="C211 launch spec")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused("C211 launch spec schema/kind differs")
    if row["recipe_id"] != RECIPE_ID:
        raise LaunchRefused("C211 launcher accepts only its code-owned recipe")
    if row["wait_contract"] != _wait_contract():
        raise LaunchRefused("C211 launch wait schedule differs")
    frame0_artifact_sha = _B._sha256(
        row["frame0_exact_artifact_sha256"],
        name="spec.frame0_exact_artifact_sha256",
    )
    frame0_receipt_sha = _B._sha256(
        row["frame0_exact_receipt_sha256"],
        name="spec.frame0_exact_receipt_sha256",
    )
    allow_colocation = row.get(COLOCATION_SPEC_KEY, False)
    if type(allow_colocation) is not bool:
        raise LaunchRefused("allow_vendor_v2_colocation must be a boolean")
    source = _exact_dict(
        row["source"], ("checkout", "commit_sha", "isaac_python"), name="spec.source"
    )
    checkout = _B._absolute_path(source["checkout"], name="source.checkout", must_exist=True)
    commit = source["commit_sha"]
    if type(commit) is not str or _B.COMMIT_RE.fullmatch(commit) is None:
        raise LaunchRefused("source.commit_sha must be exact lowercase 40-hex")
    python = _isaac_python_entry(source["isaac_python"])
    stage = row["stage"]
    if stage not in BUDGETS:
        raise LaunchRefused(
            "stage must be materialize, recipe, oracle32, scale4096, or long4096"
        )
    actual_budget = (
        _B._plain_int(row["num_envs"], name="num_envs", minimum=1),
        _B._plain_int(row["max_iterations"], name="max_iterations", minimum=0),
        _B._plain_int(row["save_interval"], name="save_interval", minimum=1),
    )
    if actual_budget != BUDGETS[stage]:
        raise LaunchRefused("%s budget must be exactly %s" % (stage, BUDGETS[stage]))
    namespace = _B._absolute_path(row["namespace"], name="namespace")
    if SAFE_COMPONENT.fullmatch(namespace.name or "") is None:
        raise LaunchRefused("namespace basename is unsafe")
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
        raise LaunchRefused("namespace parent must be the dedicated C211 experiment root")
    log_path = _B._absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must equal <namespace>/run.log")

    requirements = {
        "materialization_result": stage != "materialize",
        "recipe_result": stage not in ("materialize", "recipe"),
        "oracle32_result": stage in ("scale4096", "long4096"),
        "predecessor_result": stage == "long4096",
    }
    for key, needed in requirements.items():
        if needed is not (row[key] is not None):
            raise LaunchRefused("%s receipt requirement differs for %s" % (stage, key))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": commit,
            "isaac_python": str(python),
        },
        "recipe_id": RECIPE_ID,
        "lineage": _pin(row["lineage"], name="spec.lineage"),
        "materialization_result": row["materialization_result"],
        "recipe_result": row["recipe_result"],
        "oracle32_result": row["oracle32_result"],
        "predecessor_result": row["predecessor_result"],
        "stage": stage,
        "num_envs": actual_budget[0],
        "max_iterations": actual_budget[1],
        "save_interval": actual_budget[2],
        "wait_contract": _wait_contract(),
        "frame0_exact_artifact_sha256": frame0_artifact_sha,
        "frame0_exact_receipt_sha256": frame0_receipt_sha,
        "gpu": _validate_gpu(row["gpu"], allow_colocation=allow_colocation),
        "namespace": str(namespace),
        "log_path": str(log_path),
        COLOCATION_SPEC_KEY: allow_colocation,
    }


def _runtime_sources(checkout: Path, commit: str) -> dict[str, dict[str, str]]:
    output = {}
    for relative, label in RUNTIME_SOURCE_PATHS:
        normalized, _path = _B._verify_tracked_file(
            checkout,
            commit,
            {"path": relative, "sha256": _B.sha256_file(checkout / relative)},
            name=label,
        )
        output[label] = normalized
    return output


def _normalizer_contract() -> dict[str, Any]:
    return {
        "actor": {"identity": ACTOR_NORMALIZER_IDENTITY, "state": "fresh_empty"},
        "critic": {"identity": CRITIC_NORMALIZER_IDENTITY, "state": "fresh_empty"},
        "distinct_objects_required": True,
        "foreign_state_reuse_prohibited": True,
    }


def _checkpoint_contract() -> dict[str, Any]:
    return {
        "input": None,
        "state": "fresh_empty",
        "resume_prohibited": True,
        "foreign_lineage_reuse_prohibited": True,
        "output_actor_contract": ACTOR_CONTRACT,
        "output_actor_width": ACTOR_WIDTH,
        "output_critic_contract": CRITIC_CONTRACT,
        "output_critic_width": CRITIC_WIDTH,
    }


def _termination_contract() -> dict[str, Any]:
    return {
        "hard_union": list(HARD_TERMINATION_UNION),
        "single_stroke_terminal": "action_ball_single_stroke_complete",
        "finite_horizon_terminal": "time_out",
    }


def _continuation_stop_gate() -> dict[str, Any]:
    return {
        "exact_stage_budget_required": True,
        "hard_termination_count_max": 0,
        "table_contact_count_max": 0,
        "nonfinite_count_max": 0,
        "finite_model_required_when_updates_positive": True,
        "oracle32_pass_required_for_training_stages": True,
        "scale4096_required_for_long4096": True,
        "automatic_promotion_prohibited": True,
    }


def _training_argv(
    spec: Mapping[str, Any], lineage: Mapping[str, Any], recipe: Mapping[str, Any]
) -> list[str]:
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / _B.WBT_RELATIVE
    ppo = recipe["ppo"]
    weights = recipe["soft_weights"]
    materialization = (
        _planned_materialization(recipe=recipe, lineage=lineage)
        if spec["stage"] == "materialize"
        else _validate_materialization(
            spec["materialization_result"], recipe=recipe, lineage=lineage
        )
    )
    policy = (
        None
        if spec["stage"] in ("materialize", "recipe")
        else _validate_policy_materialization(
            spec["recipe_result"],
            recipe=recipe,
            lineage=lineage,
            materialization=materialization,
        )
    )
    policy_sha = (
        RECIPE_SENTINEL_POLICY_SHA256
        if policy is None
        else policy["runtime_policy_recipe_sha256"]
    )
    action_list = json.dumps([lineage["action_id"]], separators=(",", ":"))
    motion_list = json.dumps(
        [str(checkout / lineage["motion"]["path"])], separators=(",", ":")
    )
    argv = [
        spec["source"]["isaac_python"],
        str(wbt / "scripts/train.py"),
        "task=%s" % TASK_PROFILE_ID,
        "algo=ppo",
        "headless=true",
        "logger=tensorboard",
        "video=false",
        "device=cuda:0",
        "seed=%d" % lineage["seed"],
        "num_envs=%d" % spec["num_envs"],
        "max_iterations=%d" % spec["max_iterations"],
        "algo.runner.save_interval=%d" % spec["save_interval"],
        "algo.runner.empirical_normalization=true",
        "algo.policy.actor_hidden_dims=[512,256,128]",
        "algo.policy.critic_hidden_dims=[512,256,128]",
        "algo.policy.init_noise_std=0.02",
        "algo.policy.noise_std_type=log",
        "algo.algorithm.entropy_coef=0.01",
        "algo.algorithm.schedule=%s" % ppo["schedule"],
        "algo.algorithm.learning_rate=%s" % format(ppo["learning_rate"], ".12g"),
        "algo.algorithm.desired_kl=0.01",
        "algo.algorithm.clip_param=0.2",
        "algo.algorithm.num_learning_epochs=5",
        "algo.algorithm.num_mini_batches=4",
        "run_name=%s-DIAGNOSTIC_UNAUTHORIZED" % Path(spec["namespace"]).name,
        "task.experiment_name=%s" % EXPERIMENT_NAME,
        "task.gym_task=%s" % GYM_TASK_ID,
        "task.actor_obs_contract=%s" % ACTOR_CONTRACT,
        "action_ball_dynamic_ready_bootstrap=true",
        "action_ball_dynamic_ready_artifact_path=%s"
        % (checkout / lineage["dynamic_ready_artifact"]["path"]),
        "action_ball_dynamic_ready_artifact_sha256=%s"
        % lineage["dynamic_ready_artifact"]["sha256"],
        "action_ball_dynamic_ready_nominal_receipt_path=%s"
        % (checkout / lineage["dynamic_ready_nominal_receipt"]["path"]),
        "action_ball_dynamic_ready_nominal_receipt_sha256=%s"
        % lineage["dynamic_ready_nominal_receipt"]["sha256"],
        "motion_file=%s" % motion_list,
        "task.racket.clip_names=%s" % action_list,
        "task.racket.action_ball_manifest_path=%s"
        % (checkout / lineage["action_manifest"]["path"]),
        "task.racket.action_ball_manifest_sha256=%s"
        % lineage["action_manifest"]["sha256"],
        "task.racket.action_ball_policy_contract_sha256=%s" % policy_sha,
        "task.racket.action_ball_seed=%d" % lineage["seed"],
        "task.racket.action_ball_target_source=immutable_tape",
        "task.racket.action_ball_target_recipe=%s" % TARGET_RECIPE,
        "task.racket.action_ball_target_validity_mask=[false,false,false]",
        "task.racket.action_ball_target_observation_noise=false",
        "task.racket.adaptive_sigma=false",
        "task.racket.adaptive_sigma_monotonic=false",
        "task.racket.adaptive_sigma_normal=false",
        "task.racket.target_noise_white=0.0",
        "task.racket.target_noise_ar1_sigma=0.0",
        "task.racket.action_ball_immutable_tape_path=%s"
        % (checkout / lineage["immutable_tape"]["path"]),
        "task.racket.action_ball_immutable_tape_sha256=%s"
        % lineage["immutable_tape"]["sha256"],
        "task.racket.action_ball_diagnostic_unauthorized=true",
        "+task.racket.reference_guard_mode=%s" % recipe["reference_guard_mode"],
        "task.rewards.death_penalty_weight=%s" % weights["death_penalty"],
        "task.rewards.qdes_limit_barrier_weight=%s" % weights["qdes_limit"],
        "+task.rewards.qdes_projection_penalty_weight=%s"
        % weights["qdes_projection"],
        "task.rewards.joint_limit_weight=%s" % weights["joint_limit"],
        "task.actions.control_step_action_delay_min=0",
        "task.actions.control_step_action_delay_max=0",
        "task.push.enable=false",
        "task.physical_ball=false",
        "task.racket.virtual_ball=true",
        "task.racket.question_bank=",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
    ]
    if spec["stage"] == "materialize":
        argv.extend(
            (
                "+n1_vendor_sigma_profile=%s" % REWARD_MATERIALIZATION_PROFILE,
                "+action_ball_effective_reward_recipe_output_path=%s"
                % (Path(spec["namespace"]) / REWARD_RECIPE_FILENAME),
            )
        )
    else:
        argv.append(
            "expected_effective_reward_recipe_sha256=%s"
            % materialization["runtime_effective_reward_sha256"]
        )
        if spec["stage"] == "recipe":
            argv.append(
                "action_ball_policy_recipe_output_path=%s"
                % (Path(spec["namespace"]) / POLICY_RECIPE_FILENAME)
            )
    if spec["stage"] == "oracle32":
        argv.extend(
            (
                "+action_ball_teacher_qdes_oracle_output_path=%s"
                % (Path(spec["namespace"]) / "teacher_qdes_oracle_32ep.json"),
                "+action_ball_teacher_qdes_oracle_episodes=32",
            )
        )
    return argv


def _output_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    stage = spec["stage"]
    runtime_blocked = stage in BLOCKED_RUNTIME_STAGES
    output = {
        "ppo_update_count": 0 if runtime_blocked else spec["max_iterations"],
        "requested_ppo_update_count": spec["max_iterations"],
        "finite_model_save_interval": spec["save_interval"],
        "effective_reward_recipe": None,
        "policy_recipe": None,
        "teacher_qdes_oracle32": None,
        "boot_marker": "Learning iteration",
        "speed_benchmark_eligible": not spec[COLOCATION_SPEC_KEY],
        "diagnostic_unauthorized": True,
        "runtime_gate": ORACLE_RUNTIME_BLOCKER if runtime_blocked else "READY",
        "runtime_dependencies": (
            list(ORACLE_RUNTIME_DEPENDENCIES) if runtime_blocked else []
        ),
        "wait_contract": _wait_contract(),
        "frame0_exact_artifact_sha256": spec[
            "frame0_exact_artifact_sha256"
        ],
        "frame0_exact_receipt_sha256": spec[
            "frame0_exact_receipt_sha256"
        ],
    }
    namespace = Path(spec["namespace"])
    if stage == "materialize":
        output["effective_reward_recipe"] = str(namespace / REWARD_RECIPE_FILENAME)
        output["boot_marker"] = "ACTION_BALL_EFFECTIVE_REWARD_RECIPE_MATERIALIZED_JSON"
    elif stage == "recipe":
        output["policy_recipe"] = str(namespace / POLICY_RECIPE_FILENAME)
        output["boot_marker"] = "ACTION_BALL_POLICY_RECIPE_MATERIALIZED"
    elif stage == "oracle32":
        output["teacher_qdes_oracle32"] = str(
            namespace / "teacher_qdes_oracle_32ep.json"
        )
        output["boot_marker"] = "ACTION_BALL_TEACHER_QDES_ORACLE_COMPLETE_JSON"
    return output


def _materialization_inputs(
    spec: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    materialization = (
        _planned_materialization(recipe=recipe, lineage=lineage)
        if spec["stage"] == "materialize"
        else _validate_materialization(
            spec["materialization_result"], recipe=recipe, lineage=lineage
        )
    )
    policy = (
        None
        if spec["stage"] in ("materialize", "recipe")
        else _validate_policy_materialization(
            spec["recipe_result"],
            recipe=recipe,
            lineage=lineage,
            materialization=materialization,
        )
    )
    oracle32 = (
        _validate_oracle32(
            spec["oracle32_result"],
            checkout=Path(spec["source"]["checkout"]),
            recipe=recipe,
            lineage=lineage,
            materialization=materialization,
            policy=policy,
        )
        if spec["stage"] in ("scale4096", "long4096")
        else None
    )
    predecessor = (
        _validate_scale_predecessor(
            spec["predecessor_result"],
            materialization=materialization,
            policy=policy,
            oracle32=oracle32,
        )
        if spec["stage"] == "long4096"
        else None
    )
    return {
        "reward_materialization": materialization,
        "policy_recipe_materialization": policy,
        "oracle32_receipt": oracle32,
        "predecessor_result": predecessor,
    }


def _admission_training_argv(
    spec: Mapping[str, Any], bundle: Mapping[str, Any]
) -> list[str]:
    row = _exact_dict(
        bundle,
        (
            "lineage",
            "recipe",
            "question_contract",
            "normalizers",
            "checkpoint_contract",
            "termination_contract",
            "continuation_stop_gate",
            "curriculum_scope",
        ),
        name="C211 claim bundle",
    )
    return _training_argv(spec, row["lineage"], row["recipe"])


_ADMISSION = _A.VendorV2GPUAdmission(
    base=_B,
    schema_version=SCHEMA_VERSION,
    claim_kind=CLAIM_KIND,
    experiment_name=EXPERIMENT_NAME,
    colocation_spec_key=COLOCATION_SPEC_KEY,
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
    training_argv=_admission_training_argv,
)
_open_gpu_shared_lock = _ADMISSION._open_gpu_shared_lock
_lock_gpu_admission = _ADMISSION._lock_gpu_admission
_unlock_gpu_admission = _ADMISSION._unlock_gpu_admission
_query_gpu_processes = _ADMISSION._query_gpu_processes
_validate_runtime_gpu_process = _ADMISSION._validate_runtime_gpu_process
_live_reservations = _ADMISSION._live_reservations
_reservation_document = _ADMISSION._reservation_document
_runtime_namespace_receipt = _ADMISSION._runtime_namespace_receipt
_cleanup_post_boot_admission_failure = _ADMISSION._cleanup_post_boot_admission_failure


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
    path = _B._absolute_path(str(spec_path), name="--spec", must_exist=True)
    _B._stable_regular_file(path, name="C211 launch spec")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="C211 launch spec")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("C211 launch spec must be canonical JSON plus newline")
    spec = _validate_spec(document)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    source = _B._verify_clean_source(checkout, commit)
    runtime_sources = _runtime_sources(checkout, commit)
    runtime_assets = _B._validate_runtime_asset_environment()
    lineage = _validate_lineage(checkout, commit, spec["lineage"])
    if (
        spec["frame0_exact_artifact_sha256"]
        != lineage["frame0_exact_artifact"]["sha256"]
        or spec["frame0_exact_receipt_sha256"]
        != lineage["frame0_exact_receipt"]["sha256"]
    ):
        raise LaunchRefused("C211 spec frame0-exact pins differ from lineage")
    recipe = _recipe_contract()
    inputs = _materialization_inputs(
        spec, recipe=recipe, lineage=lineage
    )
    output_contract = _output_contract(spec)
    bundle = {
        "lineage": lineage,
        "recipe": recipe,
        "question_contract": _question_contract(),
        "normalizers": _normalizer_contract(),
        "checkpoint_contract": _checkpoint_contract(),
        "termination_contract": _termination_contract(),
        "continuation_stop_gate": _continuation_stop_gate(),
        "curriculum_scope": _curriculum_scope_contract(),
    }
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
        "max_compute_pids_on_physical_gpu": _A.MAX_VENDOR_V2_COMPUTE_PIDS,
        "minimum_free_memory_mib": _A.MIN_VENDOR_V2_FREE_MEMORY_MIB,
        "gpu_default_empty": not spec[COLOCATION_SPEC_KEY],
        "vendor_v2_colocation_opt_in": spec[COLOCATION_SPEC_KEY],
        "fresh_only": True,
        "reward_materialization_only": spec["stage"] == "materialize",
        "policy_recipe_materialization_only": spec["stage"] == "recipe",
        "teacher_qdes_oracle_only": spec["stage"] == "oracle32",
        "ppo_updates_authorized": output_contract["ppo_update_count"],
        "control_step_action_delay": 0,
        "reset_inverse_solve": False,
        "physical_ball_semantics": PHYSICAL_BALL_SEMANTICS,
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
        "source": source,
        "runtime_sources": runtime_sources,
        "runtime_assets": runtime_assets,
        "bundle": bundle,
        "materialization_inputs": inputs,
        "output_contract": output_contract,
        "boot_marker": output_contract["boot_marker"],
        "training_argv": _training_argv(spec, lineage, recipe),
    }
    _assert_c211_only(payload, name="C211 launch claim")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(payload),
        "canonical_payload": payload,
    }


def _revalidate_claim_payload(
    payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = _validate_spec(payload["spec"], claimed=True)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    if _B._verify_clean_source(checkout, commit) != payload["source"]:
        raise LaunchRefused("clean source claim drifted")
    if _runtime_sources(checkout, commit) != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted")
    _B._validate_runtime_asset_claim(payload["runtime_assets"])
    lineage = _validate_lineage(checkout, commit, spec["lineage"])
    recipe = _recipe_contract()
    expected_bundle = {
        "lineage": lineage,
        "recipe": recipe,
        "question_contract": _question_contract(),
        "normalizers": _normalizer_contract(),
        "checkpoint_contract": _checkpoint_contract(),
        "termination_contract": _termination_contract(),
        "continuation_stop_gate": _continuation_stop_gate(),
        "curriculum_scope": _curriculum_scope_contract(),
    }
    expected_inputs = _materialization_inputs(
        spec, recipe=recipe, lineage=lineage
    )
    if (
        payload["spec"] != spec
        or payload["bundle"] != expected_bundle
        or payload["materialization_inputs"] != expected_inputs
        or payload["output_contract"] != _output_contract(spec)
        or payload["boot_marker"] != payload["output_contract"]["boot_marker"]
        or payload["training_argv"] != _training_argv(spec, lineage, recipe)
    ):
        raise LaunchRefused("C211 claim lineage, output contract, or argv drifted")
    _assert_c211_only(payload, name="C211 launch claim")
    return spec, lineage, recipe


def _runtime_soft_weights(
    terms: Any, *, recipe: Mapping[str, Any]
) -> dict[str, float]:
    if type(terms) is not list:
        raise LaunchRefused("C211 runtime effective reward terms are malformed")
    by_name = {}
    for term in terms:
        if type(term) is not dict or type(term.get("name")) is not str:
            raise LaunchRefused("C211 runtime effective reward term is malformed")
        if term["name"] in by_name:
            raise LaunchRefused("C211 runtime reward term name is duplicated")
        by_name[term["name"]] = term
    expected = {
        "death_penalty": recipe["soft_weights"]["death_penalty"],
        "joint_limit": recipe["soft_weights"]["joint_limit"],
        "qdes_limit_barrier": recipe["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": recipe["soft_weights"]["qdes_projection"],
    }
    observed = {}
    for name, wanted in expected.items():
        term = by_name.get(name)
        if type(term) is not dict:
            raise LaunchRefused("C211 runtime soft-weight term is absent: %s" % name)
        if name == "qdes_projection_penalty":
            params = term.get("params")
            value = params.get("objective_weight") if type(params) is dict else None
            if term.get("weight") != -1.0:
                value = None
        else:
            value = term.get("weight")
        if type(value) not in (int, float) or value != wanted:
            raise LaunchRefused("C211 runtime soft weight differs: %s" % name)
        observed[name] = float(value)
    return {name: observed[name] for name in sorted(observed)}


def _require_c211_outcome_terms(terms: Any) -> None:
    if type(terms) is not list:
        raise LaunchRefused("C211 runtime effective reward terms are malformed")
    by_name = {}
    for term in terms:
        if type(term) is not dict or type(term.get("name")) is not str:
            raise LaunchRefused("C211 runtime effective reward term is malformed")
        if term["name"] in by_name:
            raise LaunchRefused("C211 runtime reward term name is duplicated")
        by_name[term["name"]] = term
    for name, required in REQUIRED_OUTCOME_TERMS.items():
        term = by_name.get(name)
        params = term.get("params") if type(term) is dict else None
        if (
            type(term) is not dict
            or term.get("callable") != required["callable"]
            or term.get("weight") != required["weight"]
            or type(params) is not dict
            or params != required["params"]
        ):
            raise LaunchRefused("C211 required achieved-outcome term differs: %s" % name)
    for name, required in REQUIRED_PRIOR_TERMS.items():
        term = by_name.get(name)
        params = term.get("params") if type(term) is dict else None
        if (
            type(term) is not dict
            or term.get("callable") != required["callable"]
            or type(term.get("weight")) not in (int, float)
            or term["weight"] <= 0.0
            or type(params) is not dict
            or any(params.get(key) != value for key, value in required["params"].items())
        ):
            raise LaunchRefused("C211 required motion-prior term differs: %s" % name)
    for name in PROHIBITED_CONTACT_TARGET_TERMS:
        term = by_name.get(name)
        if term is not None and term.get("weight") != 0.0:
            raise LaunchRefused("C211 desired-contact reward term is active: %s" % name)
    for name in PROHIBITED_DUPLICATE_OUTCOME_TERMS:
        term = by_name.get(name)
        if term is not None and term.get("weight") != 0.0:
            raise LaunchRefused("C211 duplicate contact/outcome term is active: %s" % name)


def _runtime_reward_materialization(
    *, path: Path, planned: Mapping[str, Any], recipe: Mapping[str, Any]
) -> dict[str, Any]:
    validated = _OLD._validate_reward_materialization(
        {"path": str(path), "sha256": _B.sha256_file(path)}
    )
    document = _B._strict_json_bytes(
        path.read_bytes(), name="C211 reward materialization"
    )
    _require_c211_outcome_terms(document["terms"])
    weights = _runtime_soft_weights(document["terms"], recipe=recipe)
    unsigned = {
        key: value for key, value in planned.items() if key != "content_sha256"
    }
    unsigned.update(
        {
            "runtime_effective_reward_artifact": validated["artifact"],
            "runtime_effective_reward_sha256": validated[
                "effective_reward_recipe_sha256"
            ],
            "runtime_effective_reward_term_count": validated["term_count"],
            "runtime_soft_weights": weights,
        }
    )
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _runtime_policy_materialization(
    *,
    path: Path,
    checkout: Path,
    lineage: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = {
        "core": {
            "dynamic_ready": {
                "artifact": lineage["dynamic_ready_artifact"],
                "nominal_hold_receipt": lineage["dynamic_ready_nominal_receipt"],
            }
        },
        "motion": lineage["motion"],
    }
    try:
        validated = _OLD._validate_policy_materialization(
            {"path": str(path), "sha256": _B.sha256_file(path)},
            checkout=checkout,
            bundle=bundle,
        )
    except _OLD.LaunchRefused as exc:
        raise LaunchRefused("C211 runtime policy recipe validation failed") from exc
    document = _B._strict_json_bytes(path.read_bytes(), name="C211 policy recipe")
    runner = document.get("action_ball_ppo_runner_recipe", {}).get("recipe", {})
    expected_policy = {
        "actor_hidden_dims": recipe["actor_hidden_dims"],
        "critic_hidden_dims": recipe["critic_hidden_dims"],
        "init_noise_std": recipe["init_noise_std"],
        "noise_std_type": recipe["noise_std_type"],
    }
    expected_algorithm = {"entropy_coef": recipe["entropy_coef"], **recipe["ppo"]}
    if (
        type(runner) is not dict
        or type(runner.get("policy")) is not dict
        or any(runner["policy"].get(key) != value for key, value in expected_policy.items())
        or type(runner.get("algorithm")) is not dict
        or any(
            runner["algorithm"].get(key) != value
            for key, value in expected_algorithm.items()
        )
        or type(runner.get("runner")) is not dict
        or runner["runner"].get("empirical_normalization") is not True
        or runner["runner"].get("init_at_random_ep_len") is not False
    ):
        raise LaunchRefused("C211 runtime policy recipe differs")
    unsigned = {
        "schema_version": 1,
        "kind": POLICY_MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "recipe_id": RECIPE_ID,
        "lineage_sha256": lineage["lineage_sha256"],
        "recipe_contract_sha256": recipe["recipe_contract_sha256"],
        "runtime_policy_recipe_artifact": validated["artifact"],
        "runtime_policy_recipe_sha256": validated["policy_contract_sha256"],
        "dynamic_ready_binding_sha256": validated["dynamic_ready_binding_sha256"],
        "noise_std_type": validated["noise_std_type"],
        "configured_and_realized_init_noise_std": validated[
            "configured_and_realized_init_noise_std"
        ],
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


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
    raw_spec = payload.get("spec") if type(payload) is dict else None
    if type(raw_spec) is dict and raw_spec.get("stage") in BLOCKED_RUNTIME_STAGES:
        raise LaunchRefused(
            "%s: true C211 needs an independent 000/p-v-spin oracle consumer"
            % ORACLE_RUNTIME_BLOCKER
        )
    spec, _lineage, _recipe = _revalidate_claim_payload(payload)
    lock_path = Path(spec["gpu"]["lock_path"])
    descriptor_info = os.fstat(lock_fd)
    path_info = lock_path.lstat()
    if (
        not stat.S_ISREG(descriptor_info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or (descriptor_info.st_dev, descriptor_info.st_ino)
        != (path_info.st_dev, path_info.st_ino)
    ):
        raise LaunchRefused("inherited GPU lock identity differs")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise LaunchRefused("inherited GPU lock is not held") from exc
    _lock_gpu_admission(lock_fd)
    try:
        gpu = _verify_gpu_admission(
            spec, phase="pre_exec", current_namespace=Path(spec["namespace"])
        )
        _B._write_exclusive_json(
            Path(spec["namespace"]) / "pre_exec_gpu_admission.json",
            {
                "schema_version": 1,
                "kind": "action_ball_c211_pre_exec_gpu_admission_v1",
                "launch_claim_sha256": claim_sha,
                "gpu": gpu,
            },
        )
        namespace_receipt, namespace_receipt_sha = _runtime_namespace_receipt(
            spec, claim_sha
        )
    finally:
        _unlock_gpu_admission(lock_fd)
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / _B.WBT_RELATIVE
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(wbt / "source/whole_body_tracking"),
        "CUDA_VISIBLE_DEVICES": str(spec["gpu"]["index"]),
        "HYDRA_FULL_ERROR": "1",
        "WANDB_MODE": "offline",
        "HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256": claim_sha,
        _A.GPU_NAMESPACE_RECEIPT_ENV: str(namespace_receipt),
        _A.GPU_NAMESPACE_RECEIPT_SHA_ENV: namespace_receipt_sha,
        **_B._runtime_asset_exec_environment(payload["runtime_assets"]),
    }
    os.chdir(wbt)
    argv = payload["training_argv"]
    os.execve(argv[0], argv, environment)
    raise AssertionError("execve returned")


def _validate_completion_state(path: Path) -> dict[str, str]:
    _B._stable_regular_file(path, name="C211 completion state")
    observed: dict[str, str] = {}
    required = {"completion_exit_code", "terminal_kind", "terminal_exit_code"}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in required:
            if key in observed:
                raise LaunchRefused("C211 completion state has duplicate %s" % key)
            observed[key] = value
    expected = {
        "completion_exit_code": "0",
        "terminal_kind": "clean_completion",
        "terminal_exit_code": "0",
    }
    if observed != expected:
        raise LaunchRefused("C211 workload did not exit cleanly and uniquely")
    return observed


def _completion_stage(stage: str) -> bool:
    return stage in ("materialize", "recipe", "oracle32", "scale4096")


def execute(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    expected = _B._sha256(confirm_claim, name="--confirm-claim")
    if expected != plan["launch_claim_sha256"]:
        raise LaunchRefused("--confirm-claim differs from freshly recomputed plan")
    payload = plan["canonical_payload"]
    if canonical_sha256(payload) != expected:
        raise LaunchRefused("launch plan payload seal differs before execution")
    spec = payload["spec"]
    if spec["stage"] in BLOCKED_RUNTIME_STAGES:
        raise LaunchRefused(
            "%s: true C211 needs an independent 000/p-v-spin oracle consumer"
            % ORACLE_RUNTIME_BLOCKER
        )
    checkout = Path(spec["source"]["checkout"])
    _B._verify_clean_source(checkout, spec["source"]["commit_sha"])
    _B._validate_runtime_asset_claim(payload["runtime_assets"])
    lock_fd = _open_gpu_shared_lock(Path(spec["gpu"]["lock_path"]))
    namespace = None
    try:
        _lock_gpu_admission(lock_fd)
        try:
            first = _verify_gpu_admission(
                spec, phase="pre_launch", current_namespace=None
            )
            namespace = _B._claim_namespace(plan)
            _B._write_exclusive_json(
                namespace / _A.GPU_RESERVATION_FILENAME,
                _reservation_document(spec, expected),
            )
            _B._write_exclusive_json(
                namespace / "pre_launch_gpu_admission.json",
                {
                    "schema_version": 1,
                    "kind": "action_ball_c211_pre_launch_gpu_admission_v1",
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
        completion_stage = _completion_stage(spec["stage"])
        environment = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "KIT_BOOT_MARKER": payload["boot_marker"],
            "KIT_BOOT_TIMEOUT_S": "2700",
            "KIT_BOOT_STALE_TIMEOUT_S": "1800",
            "KIT_BOOT_POLL_S": "5",
            "KIT_BOOT_STATE_FILE": str(state),
            **(
                {
                    "KIT_WAIT_FOR_COMPLETION": "1",
                    "KIT_COMPLETION_TIMEOUT_S": "7200",
                }
                if completion_stage
                else {}
            ),
        }
        result = subprocess.run(
            [str(checkout / KIT_LAUNCHER_SOURCE), spec["log_path"], *internal],
            cwd=checkout / _B.WBT_RELATIVE,
            env=environment,
            pass_fds=(lock_fd,),
            check=False,
        )
        if result.returncode != 0:
            raise LaunchRefused(
                "locked Kit launcher returned %d; namespace remains spent"
                % result.returncode
            )
        completion = _validate_completion_state(state) if completion_stage else None
        _lock_gpu_admission(lock_fd)
        try:
            try:
                final_gpu = _verify_gpu_admission(
                    spec,
                    phase="post_completion" if completion_stage else "post_boot",
                    current_namespace=namespace,
                    require_current_compute=not completion_stage,
                )
            except (LaunchRefused, FileNotFoundError, ValueError, OSError) as exc:
                if completion_stage:
                    raise LaunchRefused(
                        "post-completion admission refused after exact clean exit"
                    ) from exc
                failure = _cleanup_post_boot_admission_failure(
                    namespace, state, expected, str(exc)
                )
                raise LaunchRefused(
                    "post-boot admission refused; failure receipt=%s" % failure["path"]
                ) from exc
        finally:
            _unlock_gpu_admission(lock_fd)
        inputs = payload["materialization_inputs"]
        materialization = inputs["reward_materialization"]
        policy = inputs["policy_recipe_materialization"]
        oracle32 = inputs["oracle32_receipt"]
        if spec["stage"] == "materialize":
            materialization = _runtime_reward_materialization(
                path=Path(payload["output_contract"]["effective_reward_recipe"]),
                planned=materialization,
                recipe=payload["bundle"]["recipe"],
            )
        elif spec["stage"] == "recipe":
            policy = _runtime_policy_materialization(
                path=Path(payload["output_contract"]["policy_recipe"]),
                checkout=checkout,
                lineage=payload["bundle"]["lineage"],
                recipe=payload["bundle"]["recipe"],
            )
        elif spec["stage"] == "oracle32":  # pragma: no cover - blocked above
            raise LaunchRefused(ORACLE_RUNTIME_BLOCKER)
        unsigned = {
            "schema_version": 1,
            "kind": RESULT_KIND,
            "diagnostic_unauthorized": True,
            "accepted": True,
            "launch_claim_sha256": expected,
            "stage": spec["stage"],
            "namespace": str(namespace),
            "completion": completion,
            "gpu_admission": final_gpu,
            "output_contract": payload["output_contract"],
            "reward_materialization": materialization,
            "policy_recipe_materialization": policy,
            "oracle32_receipt": oracle32,
            "predecessor_result": inputs["predecessor_result"],
        }
        launch_result = {**unsigned, "content_sha256": canonical_sha256(unsigned)}
        _B._write_exclusive_json(namespace / "launch_result.json", launch_result)
        return launch_result
    finally:
        os.close(lock_fd)


def _paired_pin(path: str | None, digest: str | None, *, name: str) -> dict | None:
    if (path is None) != (digest is None):
        raise LaunchRefused("%s path/SHA must be supplied together" % name)
    return None if path is None else {"path": path, "sha256": digest}


def _write_template(args: argparse.Namespace) -> dict[str, Any]:
    budget = BUDGETS[args.stage]
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(Path(args.checkout).resolve(strict=True)),
            "commit_sha": args.commit_sha,
            "isaac_python": str(_isaac_python_entry(args.isaac_python)),
        },
        "recipe_id": RECIPE_ID,
        "lineage": {"path": args.lineage_path, "sha256": args.lineage_sha256},
        "materialization_result": _paired_pin(
            args.materialization_result_path,
            args.materialization_result_sha256,
            name="materialization result",
        ),
        "recipe_result": _paired_pin(
            args.recipe_result_path,
            args.recipe_result_sha256,
            name="recipe result",
        ),
        "oracle32_result": _paired_pin(
            args.oracle32_result_path,
            args.oracle32_result_sha256,
            name="oracle32 result",
        ),
        "predecessor_result": _paired_pin(
            args.predecessor_result_path,
            args.predecessor_result_sha256,
            name="predecessor result",
        ),
        "stage": args.stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "wait_contract": _wait_contract(),
        "frame0_exact_artifact_sha256": args.frame0_exact_artifact_sha256,
        "frame0_exact_receipt_sha256": args.frame0_exact_receipt_sha256,
        "gpu": {
            "index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "owner": args.owner,
            "lock_path": "/tmp/hope_lean_queue_gpu%d.lock" % args.gpu_index,
            "require_empty": not args.allow_colocation,
        },
        "namespace": str(Path(args.namespace).resolve(strict=False)),
        "log_path": str(Path(args.namespace).resolve(strict=False) / "run.log"),
    }
    if args.allow_colocation:
        document[COLOCATION_SPEC_KEY] = True
    normalized = _validate_spec(document)
    if not args.allow_colocation:
        normalized.pop(COLOCATION_SPEC_KEY)
    output = Path(args.output).resolve(strict=False)
    _B._write_exclusive_json(output, normalized)
    return {"status": "CREATED", "spec": str(output), "stage": args.stage}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    template = sub.add_parser("template", help="write one canonical C211 spec")
    template.add_argument("--output", required=True)
    template.add_argument("--checkout", required=True)
    template.add_argument("--commit-sha", required=True)
    template.add_argument("--isaac-python", required=True)
    template.add_argument("--lineage-path", required=True)
    template.add_argument("--lineage-sha256", required=True)
    template.add_argument("--frame0-exact-artifact-sha256", required=True)
    template.add_argument("--frame0-exact-receipt-sha256", required=True)
    template.add_argument("--stage", choices=STAGE_ORDER, required=True)
    template.add_argument("--materialization-result-path")
    template.add_argument("--materialization-result-sha256")
    template.add_argument("--recipe-result-path")
    template.add_argument("--recipe-result-sha256")
    template.add_argument("--oracle32-result-path")
    template.add_argument("--oracle32-result-sha256")
    template.add_argument("--predecessor-result-path")
    template.add_argument("--predecessor-result-sha256")
    template.add_argument("--gpu-index", type=int, required=True)
    template.add_argument("--gpu-uuid", required=True)
    template.add_argument("--owner", required=True)
    template.add_argument("--namespace", required=True)
    template.add_argument("--allow-colocation", action="store_true")

    plan = sub.add_parser("plan", help="print a read-only digest-bound claim")
    plan.add_argument("--spec", required=True)

    run = sub.add_parser("execute", help="spend a namespace and launch explicitly")
    run.add_argument("--spec", required=True)
    run.add_argument("--confirm-claim", required=True)

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
        elif args.command == "plan":
            result = build_plan(Path(args.spec))
        elif args.command == "execute":
            result = execute(
                build_plan(Path(args.spec)), confirm_claim=args.confirm_claim
            )
        else:
            return _internal_exec(
                Path(args.claim), args.claim_sha256, args.gpu_lock_fd
            )
    except (LaunchRefused, FileNotFoundError, OSError, ValueError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
