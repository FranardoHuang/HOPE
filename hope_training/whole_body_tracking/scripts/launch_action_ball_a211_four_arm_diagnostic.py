#!/usr/bin/env python3
"""Plan or explicitly execute the fail-closed A211 side of the Isaac 2x2.

This planner is deliberately separate from the historical fixed-194 launcher.  It
defaults to read-only planning; execution requires an exact recomputed claim digest.
A canonical spec selects one of two code-owned A211 PPO cells and one finite stage,
binds a tracked A211 lineage, requires a fresh namespace on one physical GPU, and
emits a digest-bound claim.  A zero-PPO ``materialize`` stage first publishes the
exact runtime reward recipe; a separate zero-PPO ``recipe`` stage consumes it and
publishes the exact dynamic-ready PPO recipe before oracle32 may run.  The GPU is
empty by default; exact VendorV2 colocation is available only through an explicit
opt-in and is excluded from speed evidence.

The primary diagnostic chain uses a finite five-update 4096-env scale stage before
the 1000-update 4096-env long stage.  Explicit same-family two-process colocation is
restricted to scale4096/long4096 and remains ineligible for rate evidence.  The
long stage requires the exact natural-exit scale result for the same
arm, ABI, reward/policy contract, seed and A211 lineage.
The smaller smoke/probe512/long512 branch remains failure-diagnosis-only and cannot
substitute for that scale terminal receipt.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
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
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_FILE = SCRIPT_DIR / "launch_n1_reward_screen_diagnostic.py"
ADMISSION_FILE = SCRIPT_DIR / "vendor_v2_gpu_admission.py"
EXACT_GROUP_FILE = SCRIPT_DIR / "exact_process_group.py"
OLD_VALIDATOR_FILE = SCRIPT_DIR / "launch_n1_measured_vendor_v2_diagnostic.py"
FOUR_GRID_FILE = SCRIPT_DIR / "action_ball_211_four_grid_contract.py"
FOUR_GRID_BARRIER_FILE = (
    SCRIPT_DIR / "action_ball_211_four_grid_prelong_barrier.py"
)
PRELONG_GATE_FILE = SCRIPT_DIR / "action_ball_4096x5_prelong_gate.py"
PRELONG_SEMANTICS_FILE = (
    SCRIPT_DIR.parent
    / "source/whole_body_tracking/whole_body_tracking/utils/"
    "action_ball_prelong_semantics.py"
)
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


_B = _load_helper("_a211_four_arm_base", BASE_FILE)
_A = _load_helper("_a211_four_arm_gpu_admission", ADMISSION_FILE)
_G = _load_helper("_a211_four_arm_exact_process_group", EXACT_GROUP_FILE)
_OLD = _load_helper("_a211_four_arm_oracle32_validator", OLD_VALIDATOR_FILE)
_F = _load_helper("_a211_c211_four_grid_authority", FOUR_GRID_FILE)
_Q = _load_helper("_a211_c211_four_grid_prelong_barrier", FOUR_GRID_BARRIER_FILE)
_P = _load_helper("_a211_4096x5_prelong_gate", PRELONG_GATE_FILE)
_S = _load_helper("_a211_4096x5_prelong_semantics", PRELONG_SEMANTICS_FILE)
_W = _load_helper("_a211_task_wait_schedule", TASK_WAIT_FILE)

LaunchRefused = _B.LaunchRefused

SCHEMA_VERSION = 2
SPEC_KIND = "action_ball_a211_four_arm_diagnostic_spec_v2"
CLAIM_KIND = "action_ball_a211_four_arm_diagnostic_claim_v2"
LINEAGE_KIND = "action_ball_a211_split_ready_online_question_dr_l0_lineage_v5"
MATERIALIZATION_KIND = "action_ball_a211_arm_materialization_v1"
POLICY_MATERIALIZATION_KIND = "action_ball_a211_policy_recipe_materialization_v1"
ORACLE32_KIND = "action_ball_a211_oracle32_receipt_v1"
RESULT_KIND = "action_ball_a211_four_arm_diagnostic_launch_result_v1"
SCALE4096_TERMINAL_ACCEPTANCE_KIND = (
    "action_ball_a211_scale4096_terminal_acceptance_v2"
)
FRAME0_EXACT_ARTIFACT_KIND = "action_ball_frame0_exact_artifact_v2"
FRAME0_EXACT_RECEIPT_KIND = "action_ball_frame0_exact_receipt_v2"
FRAME0_EXACT_SOURCE_KIND = "action_ball_teacher_motion_frame0_exact_v1"
FRAME0_LIVE_RECEIPT_KIND = "isaac_action_ball_nominal_hold_v1"
PASSIVE_HOLD_SOAK_GATE_KIND = "action_ball_a211_passive_hold_soak_gate_v1"
SPLIT_READY_RESET_WAIT_GATE_KIND = (
    "action_ball_a211_split_ready_reset_wait_gate_v1"
)
TEACHER_FRAME0_ARTIFACT_KIND = "action_ball_a211_frame0_exact_artifact_v1"
TEACHER_FRAME0_SOURCE_KIND = "action_ball_a211_teacher_motion_frame0_exact_v1"
SPLIT_READY_DYNAMIC_ARTIFACT_SHA256 = (
    "ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069"
)
SPLIT_READY_NOMINAL_HOLD_SHA256 = (
    "c8b92a28203cbf9b9a4f6dee784d6cc08f3f279672d8a9fc886aa6d92b5bb19b"
)
SPLIT_READY_TEACHER_FRAME0_ARTIFACT_SHA256 = (
    "ad17d984559776e90c70182ac4c0361c01de95094859e725162fb958defdbc54"
)
INITIAL_CENTER_TIMING_AUTHORITY_KIND = (
    "action_ball_initial_center_timing_authority_v1"
)
PRELONG_SEMANTICS_ENABLE_ENV = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_SEMANTICS"
)
PRELONG_REWARD_RECIPE_SHA_ENV = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_REWARD_RECIPE_SHA256"
)
REWARD_PPO_ECONOMY_ENABLE_ENV = "HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_GATE"
UPDATE_PROFILE_ENV = "HOPE_ACTION_BALL_UPDATE_PROFILE"
UPDATE_PROFILE_JSON_PREFIX = "HOPE_ACTION_BALL_UPDATE_PROFILE_JSON="
_FRAME0_HANDOFF_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "selection_semantics",
        "state_sha256_semantics",
        "physical_ready_state_sha256",
        "teacher_frame0_state_sha256",
        "mjcf_audit_state_sha256",
        "stored_root_quaternion_norm",
        "mjcf_audit_root_quat_wxyz",
        "mjcf_audit_quaternion_semantics",
        "stored_teacher_and_physical_quaternion_unchanged",
        "endpoints_bitwise_equal",
        "physical_ready_joint_velocity_exact_zero",
        "teacher_static_endpoint_joint_velocity_exact_zero",
        "measured_motion_velocity_channels_consumed",
        "not_a_motion_velocity_continuity_claim",
        "certified_transition_s",
        "required_min_wait_s",
        "torque_speed_curve_required",
        "torque_speed_non_requirement_reason",
        "runtime_transition_reference_required",
        "required_followup_hold_gate",
        "required_followup_policy_steps",
        "required_followup_physics_steps",
        "diagnostic_unauthorized",
        "training_authorized",
    )
)
_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS = {
    "left_sole_floor_slack_m": 1.0e-4,
    "right_sole_floor_slack_m": 1.0e-4,
    "left_contact_load_slack_n": 1.0e-1,
    "right_contact_load_slack_n": 1.0e-1,
    "support_margin_slack_m": 1.0e-3,
    "joint_position_slack_rad": 2.0e-2,
    "qdes_slack_rad": 2.0e-2,
    "torque_slack_nm": 2.0,
    "table_clearance_slack_m": 1.0e-2,
    "root_height_slack_m": 2.0e-2,
    "root_tilt_slack_rad": 2.0e-2,
    "collision_slack_m": 5.0e-3,
    "ground_lp_residual_slack": 5.0e-8,
}
FRAME0_RECEIPT_PROBE_SOURCE_PATHS = (
    "hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py",
    "hope_training/whole_body_tracking/scripts/run_action_ball_a211_frame0_nominal_hold.py",
    "hope_training/whole_body_tracking/scripts/consume_action_ball_a211_frame0_nominal_hold.py",
)
EXPERIMENT_NAME = "agibot_a3_action_ball_a211_four_arm_diagnostic"

ACTOR_CONTRACT = "action_ball_a211"
ACTOR_WIDTH = 211
CRITIC_CONTRACT = "action_ball_a211_critic_v1"
CRITIC_WIDTH = 319
TRAINABILITY_CONTRACT = "action_ball_a211_fixed_question_learnability_v2"
ACTOR_NORMALIZER_IDENTITY = "action_ball_a211_actor_norm_v2"
CRITIC_NORMALIZER_IDENTITY = "action_ball_a211_critic_norm_v1"
TASK_PROFILE_ID = "HOPEPingPongActionBallA211VendorV2N1DRL0Learnability"
GYM_TASK_ID = "HOPE-PingPong-ActionBall-A211Learnability-AgibotA3-v0"
TARGET_SEMANTICS = "a211_desired_contact_v1"
ACTION_ID = "take_061_unit04_bh"
ACTION_UID = 5527597793770800
TEACHER_ID = "Take_061_unit04_BH"
PHYSICAL_BALL_SEMANTICS = "analytic_virtual_ball_authoritative_physx_disabled"
REWARD_MATERIALIZATION_PROFILE = "measured_vendor_v2_n1_static_v1"
REWARD_RECIPE_FILENAME = "a211_effective_reward_recipe.json"
POLICY_RECIPE_FILENAME = "a211_dynamic_ready_policy_recipe.json"
RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64
POLICY_DT_S = 0.02
PHYSICAL_READY_HOLD_POLICY_STEPS = 200
PHYSICAL_READY_HOLD_PHYSICS_STEPS = 800
WAIT_SCHEDULE = _W.ActionBallTaskWaitSchedule(
    seed=20260804,
    min_wait_ticks=5,
    max_wait_ticks=25,
    episode_horizon_ticks=500,
    required_active_ticks=200,
).to_dict()
ACTOR_ORDERED_LAYOUT = (
    ("actual_base_pose_lin_vel_world", 12),
    ("base_ang_vel_body", 3),
    ("joint_pos", 31),
    ("joint_vel", 31),
    ("actions", 31),
    ("racket_site_achieved_now_heading", 9),
    ("teacher_joint_pos", 31),
    ("teacher_joint_vel", 31),
    ("racket_site_teacher_now_heading", 9),
    ("racket_site_teacher_at_reference_hit_heading", 9),
    ("task_desired_contact_position_heading", 3),
    ("task_desired_contact_velocity_heading", 3),
    ("task_desired_contact_face_heading", 3),
    ("desired_base_xy_world", 2),
    ("time_to_contact", 1),
    ("time_to_teacher_start", 1),
    ("task_valid", 1),
)
assert sum(width for _name, width in ACTOR_ORDERED_LAYOUT) == ACTOR_WIDTH
COLOCATION_SPEC_KEY = "allow_vendor_v2_colocation"
COLOCATED_STAGES = ("scale4096", "long4096")
MAX_COLOCATED_PROCESSES_PER_GPU = 2
assert MAX_COLOCATED_PROCESSES_PER_GPU == _A.MAX_VENDOR_V2_COMPUTE_PIDS
HARD_TERMINATION_UNION = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)
STRICT_HARD_TERMINATION_UNION = (
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
)
PHYSICAL_FALL_REASONS = ("base_fell_tilt", "base_too_low")
PHYSICAL_FALL_PHASES = (
    "hidden_wait",
    "revealed_pre_strike",
    "post_strike",
)
TASK_WAIT_STARTED_COUNTER = "task_wait_started_count"
TASK_REVEAL_REACHED_COUNTER = "task_reveal_reached_count"
PROHIBITED_HOLD_REFERENCE_TERMINATIONS = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)
FULL_ACTIVE_TERMINATIONS = (
    "time_out",
    "base_fell_tilt",
    "base_too_low",
    "robot_hit_table",
    "joint_qdes_forbidden",
    "joint_actual_forbidden",
)

# A211 is a learnability diagnostic, not merely a safety-weight ablation.  The
# runtime recipe must therefore prove that the actual dense learning channels
# and their gate identities were composed, rather than sealing only the four
# arm-owned soft weights below.
REQUIRED_EFFECTIVE_TERMS: Mapping[str, Mapping[str, Any]] = {
    "upright_exp": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.upright_exp",
        "params": {"std": 0.4472135954999579},
    },
    "motion_body_pos": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.motion_body_pos_swing_only",
        "params": {"command_name": "motion"},
    },
    "motion_body_ori": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.motion_body_ori_swing_only",
        "params": {"command_name": "motion"},
    },
    "motion_body_lin_vel": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.rewards.motion_global_body_linear_velocity_error_exp",
        "params": {"command_name": "motion"},
    },
    "motion_body_ang_vel": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.rewards.motion_global_body_angular_velocity_error_exp",
        "params": {"command_name": "motion"},
    },
    "motion_racket_position": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.motion_racket_position_tracking_cauchy",
        "params": {"command_name": "racket_target", "scale_in_strike_window": 1.0},
    },
    "motion_racket_velocity": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.motion_racket_velocity_tracking_cauchy",
        "params": {"command_name": "racket_target", "scale_in_strike_window": 1.0},
    },
    "motion_racket_normal": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.motion_racket_normal_tracking_cauchy",
        "params": {"command_name": "racket_target", "scale_in_strike_window": 1.0},
    },
    "motion_racket_long_axis": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.motion_racket_long_axis_tracking_cauchy",
        "params": {"command_name": "racket_target", "scale_in_strike_window": 1.0},
    },
    "racket_position_coarse": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_position_coarse_tracking_cauchy",
        "params": {"command_name": "racket_target", "std": 0.20},
    },
    "racket_velocity_coarse": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_velocity_coarse_tracking_cauchy",
        "params": {"command_name": "racket_target", "std": 1.50},
    },
    "racket_normal_coarse": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_normal_coarse_tracking_cauchy",
        "params": {"command_name": "racket_target", "std": 1.0},
    },
    "racket_position": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_position_tracking_exp",
        "params": {"command_name": "racket_target", "std": 0.50},
    },
    "racket_velocity": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_velocity_tracking_exp",
        "params": {"command_name": "racket_target", "std": 3.0},
    },
    "racket_normal": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_normal_tracking_exp",
        "params": {"command_name": "racket_target", "std": 2.10},
    },
    "racket_position_precision": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_position_tracking_exp",
        "params": {"command_name": "racket_target", "std": 0.075},
    },
    "racket_velocity_precision": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_velocity_tracking_exp",
        "params": {"command_name": "racket_target", "std": 0.50},
    },
    "racket_normal_precision": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.racket_normal_tracking_exp",
        "params": {"command_name": "racket_target", "std": 0.262},
    },
    "strike_capture_bonus": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.strike_capture_bonus",
        "params": {"command_name": "racket_target"},
    },
    "virtual_pass_net": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.virtual_pass_net",
        "params": {"command_name": "racket_target"},
    },
    "virtual_landing_dense": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.virtual_landing_dense_actual_contact",
        "params": {"command_name": "racket_target"},
    },
    "virtual_landing": {
        "callable": "whole_body_tracking.tasks.tracking.mdp.hope_rewards.virtual_landing",
        "params": {"command_name": "racket_target", "mode": "legal_base", "base_frac": 0.6, "settle_delay_s": 0.0},
    },
}

LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_action_ball_a211_four_arm_diagnostic.py"
)
ADMISSION_SOURCE = (
    "hope_training/whole_body_tracking/scripts/vendor_v2_gpu_admission.py"
)
EXACT_GROUP_SOURCE = (
    "hope_training/whole_body_tracking/scripts/exact_process_group.py"
)
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
    "HOPEPingPongActionBallA211VendorV2N1DRL0Learnability.yaml"
)
RETAINED_TASK_PROFILE_PARENT_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBallA211VendorV2N1Learnability.yaml"
)
DR_L0_MANIFEST_SOURCE = (
    "configs/action_ball_n1_measured_20260803/"
    "action_ball_211_dr_l0_learnability_candidate.v1.json"
)
A211_CONTRACT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/action_ball_a211_trainability.py"
)
A211_ENV_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
A211_REGISTRY_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/__init__.py"
)
FOUR_GRID_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_211_four_grid_contract.py"
)
PRELONG_GATE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_4096x5_prelong_gate.py"
)
FOUR_GRID_BARRIER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_211_four_grid_prelong_barrier.py"
)
PRELONG_SEMANTICS_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/action_ball_prelong_semantics.py"
)
ACTION_BALL_SAMPLING_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_sampling.py"
)
ACTION_BALL_COMMAND_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
)
ACTION_BALL_QUESTION_CACHE_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_question_cache.py"
)

RUNTIME_SOURCE_PATHS = (
    (LAUNCHER_SOURCE, "VendorV2 N1 launcher"),
    (FOUR_GRID_SOURCE, "Isaac A211/C211 four-grid authority"),
    (FOUR_GRID_BARRIER_SOURCE, "Isaac A211/C211 all-four pre-long barrier"),
    (PRELONG_GATE_SOURCE, "shared ActionBall 4096x5 pre-long terminal gate"),
    (PRELONG_SEMANTICS_SOURCE, "ActionBall 4096x5 semantic marker schema"),
    (ACTION_BALL_SAMPLING_SOURCE, "ActionBall curriculum question sampler"),
    (ACTION_BALL_COMMAND_SOURCE, "ActionBall runtime command integration"),
    (ACTION_BALL_QUESTION_CACHE_SOURCE, "A211 exact-question answer cache"),
    (ADMISSION_SOURCE, "VendorV2 GPU admission"),
    (EXACT_GROUP_SOURCE, "exact process-group helper"),
    (BASE_SOURCE, "no-clobber base helper"),
    (KIT_LAUNCHER_SOURCE, "locked Kit launcher"),
    (TRAIN_SOURCE, "training entrypoint"),
    (OLD_VALIDATOR_SOURCE, "oracle32 acceptance validator"),
    (TRAINING_CONTRACT_SOURCE, "dynamic-ready policy contract"),
    (TASK_WAIT_SOURCE, "pre-task wait schedule contract"),
    (TASK_PROFILE_SOURCE, "A211 DR-L0 task profile"),
    (RETAINED_TASK_PROFILE_PARENT_SOURCE, "A211 inherited task-profile parent"),
    (DR_L0_MANIFEST_SOURCE, "ActionBall DR-L0 launch manifest"),
    (A211_CONTRACT_SOURCE, "A211 trainability contract"),
    (A211_ENV_SOURCE, "A211 environment config"),
    (A211_REGISTRY_SOURCE, "A211 Gym registration"),
)

ISAAC_FOUR_GRID_KIND = _F.KIND
A_BOOTSTRAP_CELL_ID = _F.A_BOOTSTRAP_CELL_ID
A_STANDARD_INIT_CELL_ID = _F.A_STANDARD_INIT_CELL_ID
C_BOOTSTRAP_CELL_ID = _F.C_BOOTSTRAP_CELL_ID
C_STANDARD_INIT_CELL_ID = _F.C_STANDARD_INIT_CELL_ID
ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS = _F.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
ACTOR_INIT_MODE_DEFAULT = _F.ACTOR_INIT_MODE_DEFAULT
ISAAC_FOUR_GRID_CELL_IDS = _F.CELL_IDS
ARM_IDS = _F.FAMILY_CELL_IDS["A211"]
A_SELECTED_TAPE_VARIANT = "current_lm"


def _arm(
    death: float,
    qdes: float,
    projection: float,
    joint: float,
    guard: str,
    schedule: str,
    learning_rate: float,
    actor_init_mode: str,
    init_noise_std: float,
    noise_std_type: str,
) -> dict[str, Any]:
    return {
        "soft_weights": {
            "death_penalty": death,
            "qdes_limit": qdes,
            "qdes_projection": projection,
            "joint_limit": joint,
        },
        "reference_guard_mode": guard,
        "ppo": {
            "schedule": schedule,
            "learning_rate": learning_rate,
            "desired_kl": 0.01,
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
        },
        "actor_init_mode": actor_init_mode,
        "init_noise_std": init_noise_std,
        "noise_std_type": noise_std_type,
    }


# 2026-08-05 层级对齐(exp §5.6 第 7 条):death -300.0 -> -10.0(post-dt -6.0 -> -0.2)。
# 2026-08-05 第二轴改版(exp §5.6.2c):PPO schedule 对照降级为 later,四格共用 fixed lr1e-4;
# 第二格改跑标准 rsl_rl 初始化 + sigma 1.0 + scalar(对齐 BeyondMimic / build_1),第一格
# 维持零权重 bootstrap + 钉死 bias + sigma 0.1 + log。两格其余设定逐字节相同。
# 本表是 four-grid manifest 的手抄副本,_arm_contract() 逐字段比对两者,不同步就 LaunchRefused。
ARMS: Mapping[str, dict[str, Any]] = {
    ARM_IDS[0]: _arm(
        -10.0,
        -5.0,
        -5.0,
        -5.0,
        "metrics_only",
        "fixed",
        1.0e-4,
        ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS,
        0.1,
        "log",
    ),
    ARM_IDS[1]: _arm(
        -10.0,
        -5.0,
        -5.0,
        -5.0,
        "metrics_only",
        "fixed",
        1.0e-4,
        ACTOR_INIT_MODE_DEFAULT,
        1.0,
        "scalar",
    ),
}

BUDGETS: Mapping[str, tuple[int, int, int]] = {
    "materialize": (1, 0, 1),
    "recipe": (1, 0, 1),
    "oracle32": (1, 0, 1),
    "smoke": (1, 2, 1),
    "probe512": (512, 5, 1),
    "long512": (512, 1000, 100),
    "scale4096": (4096, 5, 1),
    "long4096": (4096, 1000, 100),
}
FORMAL_GRID_STAGE_ORDER = _F.FORMAL_STAGE_ORDER

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PIN_KEYS = ("path", "sha256")
FORBIDDEN_KEY_FRAGMENTS = (
    "target_recipe",
    "target_validity_mask",
    "resume",
    "checkpoint",
)
FORBIDDEN_VALUE_TOKENS = (
    "action_ball_a225",
    "a225",
    "action_ball_a210",
    "a210",
    "action_ball_c225",
    "c225",
    "action_ball_c210",
    "c210",
    "action_ball_c211",
    "c211",
    "l194",
    "action_ball_a211_fixed_question_learnability_v1",
    "action_ball_a211_actor_norm_v1",
)


def canonical_sha256(value: Any) -> str:
    return _B.canonical_sha256(value)


def _float_override_token(value: Any, *, name: str) -> str:
    """Emit a Hydra override that always parses back as a float.

    人话:".12g" 会把 1.0 打成 "1",Hydra 读回来就是 int。init_noise_std 必须始终是浮点,
    所以这里用 repr 保住小数点。
    """

    if type(value) is not float or value != value or value in (
        float("inf"),
        float("-inf"),
    ):
        raise LaunchRefused("%s must be a finite float override" % name)
    token = repr(value)
    if "." not in token and "e" not in token and "E" not in token:
        raise LaunchRefused("%s override lost its float form" % name)
    return token


def _isaac_four_grid_manifest() -> dict[str, Any]:
    """Validate local matched settings against the single shared authority."""

    try:
        return _F.validate_runtime_match(
            wait_contract=_wait_contract(),
            formal_budgets={
                stage: BUDGETS[stage] for stage in FORMAL_GRID_STAGE_ORDER
            },
            action_id=ACTION_ID,
            action_uid=ACTION_UID,
            teacher_id=TEACHER_ID,
        )
    except _F.FourGridContractError as exc:
        raise LaunchRefused("A211 four-grid authority differs: %s" % exc) from exc


def _four_grid_cell(cell_id: str, *, task_family: str) -> dict[str, Any]:
    if task_family != "A211":
        raise LaunchRefused("A211 launcher cannot select another task family")
    _isaac_four_grid_manifest()
    try:
        return _F.cell_for_family(cell_id, "A211")
    except _F.FourGridContractError as exc:
        raise LaunchRefused("selector is not a formal A211 grid cell") from exc


# [已删除 2026-08-05 安全门精简] _base_question_binding(49 行):
# 只被已退役的 _validate_retired_exact_frame0_lineage 调用, 随之一并退役。
# 共享题面校验本身仍在 _F.validate_base_question(action_ball_211_four_grid_contract.py)。


# [已删除 2026-08-05 安全门精简] _selected_tape_variant_binding(37 行):
# 只被已退役的 _validate_retired_exact_frame0_lineage 调用, 随之一并退役。


def _actor_layout_identity() -> dict[str, Any]:
    offset = 0
    ordered = []
    for name, width in ACTOR_ORDERED_LAYOUT:
        ordered.append(
            {"name": name, "width": width, "slice": [offset, offset + width]}
        )
        offset += width
    unsigned = {
        "schema_version": 2,
        "kind": "action_ball_a211_actor_ordered_layout_v2",
        "actor_contract": ACTOR_CONTRACT,
        "total_dim": ACTOR_WIDTH,
        "ordered_terms": ordered,
        "sensor_sources": {
            "actual_base_pose_lin_vel_world": {
                "slice": [0, 12],
                "producer": "mdp.action_ball_actual_base_pose_lin_vel_world",
                "frame": "canonical_hope_world",
                "components": "position3_orientation6d_linear_velocity3",
                "angular_velocity_included": False,
            },
            "base_ang_vel_body": {
                "slice": [12, 15],
                "producer": "mdp.action_ball_base_ang_vel_body",
                "frame": "pelvis_body_imu",
                "components": "sim_root_gyro3_plus_configured_iid_robustness_noise",
                "deployment_source": "bias_corrected_pelvis_imu_gyro3",
                "noise_calibration_status": "robustness_baseline_not_sensor_calibrated",
            },
        },
        "forbidden_actor_terms": [
            "projected_gravity", "stage1_base_state_world",
            "root_ang_vel_world", "teacher_base_now_world",
        ],
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    return _B._exact_dict(value, tuple(keys), name=name)


def _finite_handoff_vector(
    value: Any, *, width: int, name: str
) -> list[float]:
    if (
        type(value) is not list
        or len(value) != width
        or any(
            type(item) not in (int, float)
            or type(item) is bool
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise LaunchRefused("%s must be %d finite numbers" % (name, width))
    return [float(item) for item in value]


def _whole_body_state_sha256(
    joint_pos: Sequence[float],
    root_pos: Sequence[float],
    root_quat: Sequence[float],
) -> str:
    digest = hashlib.sha256()
    for label, values in (
        ("joint_pos", joint_pos),
        ("root_pos_w", root_pos),
        ("root_quat_wxyz", root_quat),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        digest.update(label.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


# [已删除 2026-08-05 安全门精简] _exact_zero_handoff_semantics(280 行):
# 只被已退役的 _validate_retired_exact_frame0_lineage 调用。仍在用的同名实现分别活在
# materialize_action_ball_a211_lineage.py 与 launch_action_ball_c211_diagnostic.py 各自的副本里。


def _prelong_semantics_exec_environment(
    stage: str, runtime_reward_sha256: Any
) -> dict[str, str]:
    if stage != "scale4096":
        return {}
    reward_sha = _B._sha256(
        runtime_reward_sha256, name="A211 prelong runtime Reward recipe SHA"
    )
    return {
        REWARD_PPO_ECONOMY_ENABLE_ENV: "1",
        PRELONG_SEMANTICS_ENABLE_ENV: "1",
        PRELONG_REWARD_RECIPE_SHA_ENV: reward_sha,
    }


def _update_profile_exec_environment(
    environ: Mapping[str, str]
) -> dict[str, str]:
    """Pass only the exact diagnostic profiler switch across both execs."""

    value = environ.get(UPDATE_PROFILE_ENV)
    if value is None:
        return {}
    if value not in ("0", "1"):
        raise LaunchRefused(
            "%s must be exactly 0 or 1 when set" % UPDATE_PROFILE_ENV
        )
    return {UPDATE_PROFILE_ENV: value}


def _update_profile_contract(environ: Mapping[str, str]) -> dict[str, Any]:
    forwarded = _update_profile_exec_environment(environ)
    value = forwarded.get(UPDATE_PROFILE_ENV)
    mode = (
        "not_requested"
        if value is None
        else "profile_on_attribution_only"
        if value == "1"
        else "explicit_profiler_off"
    )
    return {
        "environment_variable": UPDATE_PROFILE_ENV,
        "forwarded_value": value,
        "mode": mode,
        "profile_json_prefix": UPDATE_PROFILE_JSON_PREFIX,
        "speed_evidence_eligible": False,
        "gpu_kernel_attribution_claimed": False,
        "gpu_attribution_reason": (
            "host perf-counter spans add no CUDA synchronization and cannot "
            "delimit asynchronous GPU kernels"
        ),
    }


def _deferred_speed_measurement_contract() -> dict[str, Any]:
    """State the unimplemented matched gate without minting fake evidence."""

    return {
        "status": "DEFERRED_UNTIL_FINITE",
        "implemented_by_this_launcher": False,
        "workload_kind": "fixed_question_exact_answer_cache_consumer",
        "cold_first_distinct_question_solver_calls_required": 1,
        "steady_identical_question_solver_calls_required": 0,
        "novel_question_producer_unit": "seconds_per_4096_novel_questions",
        "producer_evidence_must_be_separate": True,
        "num_envs": 4096,
        "steps_per_env": 24,
        "warmup_updates": 10,
        "minimum_measured_updates": 50,
        "isolation": "exclusive_single_process_same_gpu",
        "abba_order": ["current_A", "current_C", "current_C", "current_A"],
        "main_timing_mode": "profiler_off",
        "profile_run_is_separate_non_speed_evidence": True,
        "scale4096_is_speed_or_rate_evidence": False,
    }


def _curriculum_scope_contract() -> dict[str, Any]:
    return {
        "question_authority": "curriculum_sampler_rng_every_reset",
        "current_scope": "n1_center_until_success_gated_expansion",
        "permanent_single_question_curriculum": False,
        "final_curriculum_source": "online_curriculum_sampler",
        "reset_question_selection": "sample_current_domain_each_reset",
        "answer_reuse": "complete_semantic_question_sha256_exact_cache",
        "cold_first_distinct_question_inverse_solve_calls": 1,
        "identical_question_inverse_solve_calls": 0,
        "changed_question_inverse_solve_calls": 1,
        "immutable_tape_required": False,
    }


def _assert_no_retired_contract(value: Any, *, name: str) -> None:
    """Reject historical ABI/control-plane vocabulary at the plan boundary."""

    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) is not str:
                raise LaunchRefused("%s contains a non-string key" % name)
            lowered = key.lower()
            if any(token in lowered for token in FORBIDDEN_KEY_FRAGMENTS):
                raise LaunchRefused("%s contains retired key %s" % (name, key))
            # The exact code-owned four-grid registry intentionally names the
            # matched C211 comparison cells.  It is validated by bundle
            # equality and its own canonical seal, not treated as A runtime ABI.
            if lowered == "isaac_four_grid_manifest":
                continue
            # The shared pre-long gate returns a safe checkpoint audit as
            # observed evidence, never as a resume input.  It is recomputed
            # from the exact namespace before every long claim and sealed by
            # the terminal acceptance, so validate it through that authority
            # instead of the historical control-plane vocabulary filter.
            if lowered == "prelong_gate":
                continue
            # Content-address digests and Git identities are opaque hex, not
            # vocabulary.  A digest may contain ``c225`` by chance and must not
            # make an otherwise identical launch nondeterministically invalid.
            if lowered.endswith("sha256") or lowered == "commit_sha":
                continue
            _assert_no_retired_contract(child, name=name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_retired_contract(child, name=name)
    elif isinstance(value, str):
        lowered = value.lower()
        assignment_key, separator, assignment_value = lowered.partition("=")
        if (
            separator
            and (
                assignment_key.endswith("sha256")
                or assignment_key.endswith("commit_sha")
            )
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", assignment_value)
            is not None
        ):
            return
        if any(token in lowered for token in FORBIDDEN_VALUE_TOKENS):
            raise LaunchRefused("%s contains a retired ABI/arm token" % name)


def _pin(value: Any, *, name: str) -> dict[str, str]:
    row = _exact_dict(value, PIN_KEYS, name=name)
    path = _B._relative_path(row["path"], name="%s.path" % name)
    digest = _B._sha256(row["sha256"], name="%s.sha256" % name)
    return {"path": path, "sha256": digest}


def _four_grid_prelong_receipt_pin(value: Any) -> dict[str, str]:
    try:
        pin, _path = _Q._pin(value, name="four-grid scale4096 receipt")
    except _Q.BarrierRefused as exc:
        raise LaunchRefused("A211 four-grid pre-long receipt pin differs") from exc
    return pin


def _validate_four_grid_prelong_receipt(
    value: Any, *, checkout: Path
) -> dict[str, Any]:
    try:
        return _Q.validate_receipt(value, checkout=checkout)
    except _Q.BarrierRefused as exc:
        raise LaunchRefused("A211 four-grid pre-long barrier refused: %s" % exc) from exc


def _external_pin(value: Any, *, name: str) -> tuple[dict[str, str], Path]:
    row = _exact_dict(value, PIN_KEYS, name=name)
    path = _B._absolute_path(row["path"], name="%s.path" % name, must_exist=True)
    _B._stable_regular_file(path, name=name)
    digest = _B._sha256(row["sha256"], name="%s.sha256" % name)
    if _B.sha256_file(path) != digest:
        raise LaunchRefused("%s file SHA differs" % name)
    return {"path": str(path), "sha256": digest}, path


def _canonical_external_json(value: Any, *, name: str) -> tuple[dict[str, str], dict]:
    pin, path = _external_pin(value, name=name)
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name=name)
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("%s must be canonical JSON plus newline" % name)
    return pin, document


def _tracked_json(
    checkout: Path,
    commit: str,
    value: Any,
    *,
    name: str,
) -> tuple[dict[str, str], dict]:
    pin = _pin(value, name=name)
    normalized, path = _B._verify_tracked_file(
        checkout, commit, pin, name=name
    )
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name=name)
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("%s must be canonical JSON plus newline" % name)
    return normalized, document


def _verify_frame0_artifact_source_commit(
    checkout: Path, artifact_source_commit: str, artifact_pin: Mapping[str, str]
) -> None:
    committed_artifact = subprocess.run(
        [
            "git", "-C", str(checkout), "show",
            artifact_source_commit + ":" + artifact_pin["path"],
        ],
        check=False,
        capture_output=True,
    )
    if (
        committed_artifact.returncode != 0
        or hashlib.sha256(committed_artifact.stdout).hexdigest()
        != artifact_pin["sha256"]
    ):
        raise LaunchRefused("A211 frame0-exact artifact source commit differs")


def _verify_commit_ancestor(
    checkout: Path, ancestor_commit: str, descendant_commit: str, *, name: str
) -> None:
    result = subprocess.run(
        ["git", "-C", str(checkout), "merge-base", "--is-ancestor",
         ancestor_commit, descendant_commit],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise LaunchRefused("%s is not a launch ancestor" % name)


def _verify_frame0_probe_source_commit(
    checkout: Path, launch_commit: str, probe_source_commit: str
) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", probe_source_commit) is None:
        raise LaunchRefused("A211 frame0-exact probe source commit is malformed")
    ancestor = subprocess.run(
        ["git", "-C", str(checkout), "merge-base", "--is-ancestor",
         probe_source_commit, launch_commit],
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise LaunchRefused("A211 frame0-exact probe source is not a launch ancestor")
    for source_path in FRAME0_RECEIPT_PROBE_SOURCE_PATHS:
        present = subprocess.run(
            ["git", "-C", str(checkout), "cat-file", "-e",
             probe_source_commit + ":" + source_path],
            check=False,
            capture_output=True,
        )
        if present.returncode != 0:
            raise LaunchRefused(
                "A211 frame0-exact probe source closure is incomplete"
            )


def _validate_frame0_live_safety_evidence(
    receipt: Mapping[str, Any], artifact: Mapping[str, Any]
) -> None:
    live = receipt["live_safety_evidence"]
    if type(live) is not dict:
        raise LaunchRefused("A211 frame0 live safety evidence must be an object")
    live_unsigned = dict(live)
    live_content_sha = live_unsigned.pop("content_sha256", None)
    live_raw = _B._canonical_bytes(live)
    joint = live.get("joint_safety_telemetry")
    screenshots = live.get("screenshots")
    expected_labels = (
        "raw_env_reset", "physical_ready_after_reset_write",
        "after_step_1", "after_step_10", "final",
    )
    try:
        telemetry_finite = all(
            isinstance(live.get(key), (int, float))
            and not isinstance(live.get(key), bool)
            and math.isfinite(float(live[key]))
            for key in (
                "minimum_root_z_m", "maximum_root_tilt_rad",
                "both_feet_contact_fraction",
            )
        )
        minimum_gap = joint.get("final_minimum_hard_gap_rad")
        joint_vectors_finite = all(
            type(joint.get(key)) is list
            and len(joint[key]) == 31
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in joint[key]
            )
            for key in (
                "preterminal_joint_pos_rad", "preterminal_joint_vel_radps",
                "final_joint_pos_rad", "final_joint_vel_radps",
                "hard_lower_rad", "hard_upper_rad",
            )
        )
        birth = artifact["birth_horizon"]
        expected_duration = (
            birth["required_policy_ticks"] * artifact["policy_dt_s"]
        )
        requested_duration_exact = math.isclose(
            float(live.get("requested_duration_s", float("nan"))),
            expected_duration,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        completed_duration_exact = math.isclose(
            float(live.get("completed_duration_s", float("nan"))),
            expected_duration,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    except (AttributeError, TypeError, ValueError):
        telemetry_finite = False
        minimum_gap = None
        joint_vectors_finite = False
        requested_duration_exact = False
        completed_duration_exact = False
    if (
        live.get("schema_version") != 1
        or live.get("kind") != FRAME0_LIVE_RECEIPT_KIND
        or live.get("verdict") != "PASS"
        or live.get("action_id") != artifact["action_id"]
        or live.get("motion_sha256") != artifact["motion_sha256"]
        or live_content_sha != receipt["live_safety_evidence_content_sha256"]
        or live_content_sha != canonical_sha256(live_unsigned)
        or receipt["live_safety_evidence_file_sha256"]
        != hashlib.sha256(live_raw).hexdigest()
        or live.get("teacher_reference_unchanged") is not True
        or live.get("teacher_physical_birth_separated") is not False
        or live.get("candidate_physical_birth_written") is not True
        or live.get("candidate_hold_qdes_and_delay_history_installed") is not True
        or live.get("plant_contract_match") is not True
        or live.get("terminal_reasons") != []
        or live.get("generic_terminated") is not False
        or live.get("generic_truncated") is not False
        or live.get("completed_policy_steps")
        != artifact["birth_horizon"]["required_policy_ticks"]
        or live.get("completed_physics_steps")
        != receipt["birth_execution_horizon"]["required_physics_substeps"]
        or not requested_duration_exact
        or not completed_duration_exact
        or type(live.get("active_terminations")) is not list
        or any(name not in live["active_terminations"] for name in HARD_TERMINATION_UNION)
        or type(joint) is not dict
        or joint.get("schema_version") != 1
        or joint.get("complete") is not True
        or type(joint.get("joint_order")) is not list
        or len(joint["joint_order"]) != 31
        or len(set(joint["joint_order"])) != 31
        or joint.get("current_actual_hard_edge_joint_count") != 0
        or joint.get("current_actual_hard_edge_joint_names") != []
        or joint.get("substep_actual_hard_edge_joint_count") != 0
        or joint.get("substep_actual_hard_edge_joint_names") != []
        or not telemetry_finite
        or not joint_vectors_finite
        or not isinstance(minimum_gap, (int, float))
        or isinstance(minimum_gap, bool)
        or not math.isfinite(float(minimum_gap))
        or float(minimum_gap) <= 0.0
        or type(screenshots) is not list
        or tuple(
            row.get("label") for row in screenshots if type(row) is dict
        ) != expected_labels
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
            for row in screenshots if type(row) is dict
        )
    ):
        raise LaunchRefused(
            "A211 frame0-exact receipt lacks exact live safety evidence"
        )
    gate = receipt.get("dynamic_birth_gate_evidence")
    thresholds = {
        "table_contact_count_max": 0,
        "nonfinite_count_max": 0,
        "actual_hard_edge_joint_count_max": 0,
        "minimum_forward_hard_gap_rad_exclusive_min": 0.0,
    }
    observed = gate.get("observed") if type(gate) is dict else None
    scope = gate.get("nominal_scope") if type(gate) is dict else None
    headroom = observed.get("forward_headroom") if type(observed) is dict else None
    if (
        type(gate) is not dict
        or gate.get("schema_version") != 1
        or gate.get("kind") != "action_ball_frame0_dynamic_birth_gate_evidence_v1"
        or gate.get("thresholds_preregistered") != thresholds
        or type(observed) is not dict
        or observed.get("table_contact_count") != 0
        or observed.get("nonfinite_count") != 0
        or observed.get("current_actual_hard_edge_joint_count") != 0
        or observed.get("substep_actual_hard_edge_joint_count") != 0
        or type(headroom) is not list
        or [row.get("state") for row in headroom if type(row) is dict]
        != ["preterminal", "final"]
        or any(
            type(row.get("minimum_forward_hard_gap_rad")) not in (int, float)
            or type(row.get("minimum_forward_hard_gap_rad")) is bool
            or not math.isfinite(float(row["minimum_forward_hard_gap_rad"]))
            or float(row["minimum_forward_hard_gap_rad"]) <= 0.0
            or type(row.get("minimum_forward_hard_gap_joint_name")) is not str
            for row in headroom
        )
        or scope
        != {
            "actor_bias": "exact_frame0_normalized_action",
            "per_env_joint_default_offset_dr_preserved": True,
            "per_env_joint_default_offset_range_rad": [-0.01, 0.01],
            "full_dr_distribution_hold_pass_claimed": False,
        }
    ):
        raise LaunchRefused("A211 dynamic birth gate evidence is incomplete")


def _frame0_birth_gate_binding_sha256(
    *, artifact: Mapping[str, Any], receipt: Mapping[str, Any]
) -> str:
    """Seal the family-neutral timing, plant, and live birth authority."""

    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "action_ball_frame0_birth_gate_binding_v1",
            "motion_sha256": artifact["motion_sha256"],
            "wait_schedule_canonical_sha256": artifact[
                "wait_schedule_canonical_sha256"
            ],
            "frame0": artifact["frame0"],
            "timing_receipt": artifact["timing_receipt"],
            "birth_horizon": artifact["birth_horizon"],
            "plant_template_file_sha256": receipt[
                "plant_template_file_sha256"
            ],
            "plant_template_content_sha256": receipt[
                "plant_template_content_sha256"
            ],
            "birth_execution_horizon": receipt["birth_execution_horizon"],
            "dynamic_birth_gate_evidence": receipt[
                "dynamic_birth_gate_evidence"
            ],
            "live_safety_evidence_content_sha256": receipt[
                "live_safety_evidence_content_sha256"
            ],
        }
    )


# [已删除 2026-08-05 安全门精简] _validate_physical_ready_long_hold(282 行):
# 只被已退役的 _validate_retired_exact_frame0_lineage 调用, 随之一并退役。原实现见 git 历史。


# [已删除 2026-08-05 安全门精简] _validate_retired_exact_frame0_lineage(339 行):
# 退役的 frame0-exact 血统校验(schema_version 停在 2), 全仓零调用点。现役的正式血统
# 校验是下面的 _validate_lineage(schema_version=5), build_plan 与 _revalidate_claim_payload
# 两条 launch 路径都走它。原实现见 git 历史。


def _runtime_target_contract() -> dict[str, Any]:
    """Lineage-owned identity for the A target provider selected at rollout 0."""

    return {
        "schema_version": 1,
        "source": "online_solver",
        "recipe": "current_lm",
        "validity_mask": [True, True, True],
        "reuse_exact_question_until_semantics_change": True,
        "immutable_tape_consumed_by_runtime": False,
    }


def _initial_center_timing_authority(
    *,
    receipt: Mapping[str, Any],
    receipt_pin: Mapping[str, str],
    action_manifest: Mapping[str, Any],
    action_manifest_pin: Mapping[str, str],
    motion_sha256: str,
    family: str = "A",
) -> dict[str, Any]:
    """Bind literal level-zero center timing without making it a runtime tape."""

    if family not in ("A", "C"):
        raise LaunchRefused("initial-center timing family must be A or C")
    if type(receipt) is not dict:
        raise LaunchRefused("initial-center timing receipt must be a JSON object")
    receipt_pin = _pin(receipt_pin, name="initial-center timing receipt")
    action_manifest_pin = _pin(
        action_manifest_pin, name="initial-center action manifest"
    )
    receipt_unsigned = dict(receipt)
    receipt_content_sha256 = receipt_unsigned.pop("canonical_sha256", None)
    if (
        _B._sha256(
            receipt_content_sha256,
            name="initial-center timing receipt canonical SHA",
        )
        != canonical_sha256(receipt_unsigned)
        or receipt.get("schema_version") != 5
        or receipt.get("action_slot") != 0
        or receipt.get("action_uid") != ACTION_UID
        or receipt.get("motion_sha256") != motion_sha256
        or receipt.get("manifest_sha256") != action_manifest_pin["sha256"]
    ):
        raise LaunchRefused("initial-center timing receipt identity differs")

    actions = action_manifest.get("actions")
    action = actions[0] if type(actions) is list and len(actions) == 1 else None
    ball_profile = action.get("ball_profile") if type(action) is dict else None
    solver_profile_sha256 = action_manifest.get("solver_profile_sha256")
    physics_profile_sha256 = action_manifest.get("physics_profile_sha256")
    if (
        action_manifest.get("schema_version") != 3
        or action_manifest.get("action_order") != [ACTION_ID]
        or action_manifest.get("mobility_mode") != "no_move"
        or type(action) is not dict
        or action.get("action_id") != ACTION_ID
        or action.get("action_uid") != ACTION_UID
        or action.get("motion_sha256") != motion_sha256
        or type(ball_profile) is not dict
        or receipt.get("solver_sha256") != solver_profile_sha256
        or receipt.get("physics_sha256") != physics_profile_sha256
    ):
        raise LaunchRefused(
            "initial-center timing receipt manifest/solver closure differs"
        )

    zero_level_rows = (
        receipt.get("domain_levels"),
        receipt.get("sampling_levels"),
        receipt.get("birth_sampling_levels"),
    )
    if (
        receipt.get("domain_epoch") != 0
        or receipt.get("sampling_stratum") != "center"
        or receipt.get("birth_sampling_stratum") != "center"
        or receipt.get("frontier_arm") is not None
        or receipt.get("birth_frontier_arm") is not None
        or receipt.get("sample_draw_start") != 3
        or receipt.get("sample_draw_end") != 21
        or any(
            type(row) is not dict
            or len(row) != 32
            or any(type(level) is not float or level != 0.0 for level in row.values())
            for row in zero_level_rows
        )
    ):
        raise LaunchRefused(
            "initial-center timing receipt is not the literal all-zero center row"
        )

    try:
        contact_time_step_s = float(receipt["contact_time_step_s"])
        center_time_s = float(ball_profile["time_to_contact_center_s"])
        center_tick = math.floor(center_time_s / contact_time_step_s + 0.5)
        time_to_contact_tick = int(receipt["time_to_contact_tick"])
        time_to_contact_s = float(receipt["time_to_contact_s"])
        reference_t_hit_s = float(receipt["reference_t_hit_s"])
        teacher_rate = float(receipt["teacher_rate"])
        scaled_t_hit_s = float(receipt["scaled_t_hit_s"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise LaunchRefused("initial-center timing receipt clocks are malformed") from exc
    if family == "A":
        derived_wait_s = time_to_contact_s - scaled_t_hit_s
        derivation = "time_to_contact_s_minus_scaled_t_hit_s"
        timing_mode = "a_online_solver"
    else:
        derived_wait_s = time_to_contact_s - reference_t_hit_s
        derivation = "time_to_contact_s_minus_reference_t_hit_s"
        timing_mode = "c_direct_ball"
    center_identity = {
        "action_uid": ACTION_UID,
        "motion_sha256": motion_sha256,
        "time_to_contact_tick": center_tick,
        "time_to_contact_s": center_tick * contact_time_step_s,
        "incoming_speed_mps": ball_profile.get("incoming_speed_center_mps"),
        "incoming_direction_b_yaw": ball_profile.get(
            "incoming_direction_center_b_yaw"
        ),
        "spin_magnitude_radps": ball_profile.get("spin_magnitude_center_radps"),
        "spin_direction_b_yaw": ball_profile.get("spin_direction_center_b_yaw"),
        "base_spawn_center_w_xy_m": ball_profile.get("base_spawn_center_w_xy_m"),
        "base_travel_center_b_yaw_xy_m": ball_profile.get(
            "base_travel_center_b_yaw_xy_m"
        ),
        "contact_offset_center_b_yaw_m": ball_profile.get(
            "contact_offset_center_b_yaw_m"
        ),
    }
    if (
        not all(
            math.isfinite(value)
            for value in (
                contact_time_step_s,
                center_time_s,
                time_to_contact_s,
                reference_t_hit_s,
                teacher_rate,
                scaled_t_hit_s,
                derived_wait_s,
            )
        )
        or contact_time_step_s != POLICY_DT_S
        or center_tick <= 0
        or time_to_contact_tick != center_tick
        or time_to_contact_s != center_tick * contact_time_step_s
        or receipt.get("incoming_speed_mps")
        != ball_profile.get("incoming_speed_center_mps")
        or receipt.get("incoming_direction_b_yaw")
        != ball_profile.get("incoming_direction_center_b_yaw")
        or receipt.get("spin_magnitude_radps")
        != ball_profile.get("spin_magnitude_center_radps")
        or receipt.get("spin_direction_b_yaw")
        != ball_profile.get("spin_direction_center_b_yaw")
        or teacher_rate <= 0.0
        or reference_t_hit_s / teacher_rate != scaled_t_hit_s
        or receipt.get("pre_swing_wait_s") != derived_wait_s
        or derived_wait_s < 0.0
    ):
        raise LaunchRefused("initial-center timing derivation differs")

    unsigned = {
        "schema_version": 1,
        "kind": INITIAL_CENTER_TIMING_AUTHORITY_KIND,
        "role": "calibration_receipt_not_runtime_question_source",
        "family": family,
        "timing_mode": timing_mode,
        "literal_center_question_sha256": canonical_sha256(center_identity),
        "receipt": {
            **receipt_pin,
            "content_sha256": receipt_content_sha256,
        },
        "action_manifest": action_manifest_pin,
        "solver_profile_sha256": solver_profile_sha256,
        "physics_profile_sha256": physics_profile_sha256,
        "sampling_profile_sha256": receipt["profile_sha256"],
        "sample_receipt_sha256": receipt["sample_sha256"],
        "time_to_contact_tick": center_tick,
        "time_to_contact_s": time_to_contact_s,
        "reference_t_hit_s": reference_t_hit_s,
        "scaled_t_hit_s": scaled_t_hit_s,
        "initial_center_time_to_teacher_start_at_reveal_s": derived_wait_s,
        "derivation": derivation,
    }
    return {**unsigned, "claim_sha256": canonical_sha256(unsigned)}


def _dr_l0_manifest_binding(
    checkout: Path,
    commit: str,
    *,
    family: str,
    task_profile: str,
) -> dict[str, str]:
    """Bind one DR-L0 leaf to the exact resolved training finalizer bytes."""

    if family not in ("A", "C"):
        raise LaunchRefused("DR-L0 family must be A or C")
    pin, path = _B._verify_tracked_file(
        checkout,
        commit,
        {
            "path": DR_L0_MANIFEST_SOURCE,
            "sha256": _B.sha256_file(checkout / DR_L0_MANIFEST_SOURCE),
        },
        name="ActionBall DR-L0 launch manifest",
    )
    manifest = _B._strict_json_bytes(
        path.read_bytes(), name="ActionBall DR-L0 launch manifest"
    )
    try:
        training_contract = _OLD._load_training_contract_module(checkout)
        resolved = training_contract.action_ball_dr_l0_contract_payload()
        contract_sha256 = training_contract.action_ball_dr_l0_contract_sha256()
    except Exception as exc:
        raise LaunchRefused(
            "cannot resolve the code-owned DR-L0 finalizer contract"
        ) from exc
    authorization = manifest.get("authorization")
    resolved_manifest = manifest.get("resolved_finalizer_contract")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind")
        != "action_ball_211_dr_l0_learnability_candidate"
        or manifest.get("identity")
        != "action_ball_211_dr_l0_learnability_candidate_v1"
        or manifest.get("status") != "BOUND_FRESH_DIAGNOSTIC_LAUNCH"
        or manifest.get("runtime_integration_blockers") != []
        or type(authorization) is not dict
        or authorization.get("diagnostic_unauthorized") is not True
        or authorization.get("formal") is not False
        or authorization.get("runtime_launch") is not True
        or manifest.get("task_profiles", {}).get(family)
        != TASK_PROFILE_SOURCE.replace(
            "A211VendorV2N1DRL0", "%s211VendorV2N1DRL0" % family
        )
        or manifest.get("parent_profiles", {}).get(family)
        != task_profile.replace("DRL0Learnability", "Learnability")
        or type(resolved_manifest) is not dict
        or resolved_manifest.get("contract_sha256") != contract_sha256
        or resolved_manifest.get("hard_contract_identity")
        != "action_ball_dr_l0_exact_all_off_v1"
        or type(resolved) is not dict
        or resolved.get("identity")
        != resolved_manifest.get("hard_contract_identity")
        or canonical_sha256(resolved) != contract_sha256
    ):
        raise LaunchRefused("DR-L0 manifest/finalizer binding differs")
    return {
        "path": pin["path"],
        "file_sha256": pin["sha256"],
        "contract_sha256": contract_sha256,
        "hard_contract_identity": resolved["identity"],
        "task_profile": task_profile,
    }


def _validate_teacher_frame0_artifact(
    artifact: Any,
    *,
    motion_path: Path,
    motion_sha256: str,
) -> dict[str, Any]:
    """Validate measured frame0 as teacher bytes, never as physical reset proof."""

    row = _exact_dict(
        artifact,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "source_kind",
            "action_id",
            "motion_sha256",
            "task_close_ticks",
            "policy_dt_s",
            "wait_schedule_canonical_sha256",
            "frame0",
            "content_sha256",
        ),
        name="A211 teacher-frame0 artifact",
    )
    unsigned = dict(row)
    content_sha256 = unsigned.pop("content_sha256")
    frame0 = _exact_dict(
        row["frame0"],
        (
            "root_pos_w_m",
            "root_quat_wxyz",
            "root_lin_vel_w_mps",
            "root_ang_vel_w_radps",
            "joint_pos_rad",
            "joint_vel_radps",
        ),
        name="A211 teacher-frame0 payload",
    )
    if (
        row["schema_version"] != 1
        or row["kind"] != TEACHER_FRAME0_ARTIFACT_KIND
        or row["diagnostic_unauthorized"] is not True
        or row["source_kind"] != TEACHER_FRAME0_SOURCE_KIND
        or row["action_id"] != ACTION_ID
        or row["motion_sha256"] != motion_sha256
        or row["task_close_ticks"] != WAIT_SCHEDULE["required_active_ticks"]
        or row["policy_dt_s"] != POLICY_DT_S
        or row["wait_schedule_canonical_sha256"]
        != WAIT_SCHEDULE["canonical_sha256"]
        or content_sha256 != canonical_sha256(unsigned)
    ):
        raise LaunchRefused("A211 teacher-frame0 artifact binding differs")
    try:
        with np.load(str(motion_path), allow_pickle=False) as motion:
            expected = {
                "root_pos_w_m": np.asarray(motion["body_pos_w"])[0, 0].tolist(),
                "root_quat_wxyz": np.asarray(motion["body_quat_w"])[0, 0].tolist(),
                "root_lin_vel_w_mps": [0.0, 0.0, 0.0],
                "root_ang_vel_w_radps": [0.0, 0.0, 0.0],
                "joint_pos_rad": np.asarray(motion["joint_pos"])[0].tolist(),
                "joint_vel_radps": [0.0] * 31,
            }
    except (KeyError, OSError, ValueError) as exc:
        raise LaunchRefused("A211 measured motion lacks an exact frame0") from exc
    if frame0 != expected:
        raise LaunchRefused(
            "A211 teacher-frame0 artifact is not exact measured-motion frame0"
        )
    return {
        "content_sha256": content_sha256,
        "frame0": frame0,
        "physical_reset_authority": False,
        "role": "immutable_measured_teacher_reference_only",
    }


def _split_ready_reset_wait_semantics(
    *,
    dynamic: Mapping[str, Any],
    nominal: Mapping[str, Any],
    dynamic_pin: Mapping[str, str],
    nominal_pin: Mapping[str, str],
    teacher_frame0: Mapping[str, Any],
    motion_sha256: str,
    initial_center_timing_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Certify the physical-ready hold and measured-teacher reveal bridge."""

    timing_unsigned = dict(initial_center_timing_authority)
    timing_claim_sha256 = timing_unsigned.pop("claim_sha256", None)
    if (
        timing_unsigned.get("schema_version") != 1
        or timing_unsigned.get("kind") != INITIAL_CENTER_TIMING_AUTHORITY_KIND
        or timing_unsigned.get("role")
        != "calibration_receipt_not_runtime_question_source"
        or _B._sha256(
            timing_claim_sha256,
            name="initial-center timing authority claim SHA",
        )
        != canonical_sha256(timing_unsigned)
    ):
        raise LaunchRefused("initial-center timing authority is not reproducible")
    pre_swing_wait_s = timing_unsigned.get(
        "initial_center_time_to_teacher_start_at_reveal_s"
    )
    if (
        type(pre_swing_wait_s) is not float
        or not math.isfinite(pre_swing_wait_s)
        or pre_swing_wait_s < 0.0
    ):
        raise LaunchRefused("initial-center reveal timing is malformed")

    dynamic_unsigned = dict(dynamic)
    dynamic_content_sha256 = dynamic_unsigned.pop("content_sha256", None)
    nominal_unsigned = dict(nominal)
    nominal_content_sha256 = nominal_unsigned.pop("content_sha256", None)
    physical = dynamic.get("physical_ready")
    teacher = dynamic.get("teacher_reference")
    composition = dynamic.get("physical_birth_composition")
    runtime_plant = dynamic.get("runtime_plant")
    binding = nominal.get("artifact")
    joint = nominal.get("joint_safety_telemetry")
    if (
        _B._sha256(dynamic_content_sha256, name="split-ready content SHA")
        != canonical_sha256(dynamic_unsigned)
        or _B._sha256(
            nominal_content_sha256, name="split-ready hold content SHA"
        )
        != canonical_sha256(nominal_unsigned)
        or dynamic.get("schema_version") != 2
        or dynamic.get("kind") != "agibot_a3_action_dynamic_ready_candidate_v2"
        or dynamic.get("action_id") != ACTION_ID
        or type(physical) is not dict
        or type(teacher) is not dict
        or type(composition) is not dict
        or type(runtime_plant) is not dict
        or composition.get("semantics")
        != "teacher_yaw_aligned_full_seed_plus_exact_teacher_reference"
        or composition.get("teacher_and_physical_birth_differ") is not True
        or teacher.get("semantics") != "exact_motion_bytes_frame0_reference"
        or teacher.get("motion_sha256") != motion_sha256
        or teacher.get("frame_index") != 0
        or teacher.get("joint_pos_rad") != teacher_frame0["joint_pos_rad"]
        or teacher.get("root_pos_w_m") != teacher_frame0["root_pos_w_m"]
    ):
        raise LaunchRefused("split-ready physical/teacher authority differs")
    teacher_quat = _finite_handoff_vector(
        teacher.get("root_quat_wxyz"), width=4, name="split-ready teacher quaternion"
    )
    frame0_quat = _finite_handoff_vector(
        teacher_frame0.get("root_quat_wxyz"), width=4, name="measured frame0 quaternion"
    )
    teacher_norm = float(np.linalg.norm(np.asarray(teacher_quat, dtype=np.float64)))
    frame0_norm = float(np.linalg.norm(np.asarray(frame0_quat, dtype=np.float64)))
    orientation_dot = abs(float(np.dot(
        np.asarray(teacher_quat, dtype=np.float64) / teacher_norm,
        np.asarray(frame0_quat, dtype=np.float64) / frame0_norm,
    )))
    physical_joint = _finite_handoff_vector(
        physical.get("joint_pos_rad"), width=31, name="split-ready joint position"
    )
    physical_root = _finite_handoff_vector(
        physical.get("root_pos_w_m"), width=3, name="split-ready root position"
    )
    physical_quat = _finite_handoff_vector(
        physical.get("root_quat_wxyz"), width=4, name="split-ready root quaternion"
    )
    physical_vel = _finite_handoff_vector(
        physical.get("joint_vel_radps"), width=31, name="split-ready joint velocity"
    )
    physical_matches_teacher = (
        physical_joint == teacher_frame0["joint_pos_rad"]
        and physical_root == teacher_frame0["root_pos_w_m"]
        and physical_quat == teacher_frame0["root_quat_wxyz"]
    )
    decimation = runtime_plant.get("control_decimation")
    policy_steps = nominal.get("completed_policy_steps")
    physics_steps = nominal.get("completed_physics_steps")
    required_policy_steps = int(WAIT_SCHEDULE["max_wait_ticks"])
    expected_duration_s = (
        float(policy_steps) * POLICY_DT_S if type(policy_steps) is int else math.nan
    )
    try:
        receipt_metrics_finite = all(
            type(nominal.get(key)) in (int, float)
            and type(nominal.get(key)) is not bool
            and math.isfinite(float(nominal[key]))
            for key in ("minimum_root_z_m", "maximum_root_tilt_rad", "both_feet_contact_fraction")
        )
        joint_vectors_finite = all(
            type(joint.get(key)) is list
            and len(joint[key]) == 31
            and all(type(value) in (int, float) and type(value) is not bool and math.isfinite(float(value)) for value in joint[key])
            for key in ("preterminal_joint_pos_rad", "preterminal_joint_vel_radps", "final_joint_pos_rad", "final_joint_vel_radps", "hard_lower_rad", "hard_upper_rad")
        )
    except (AttributeError, TypeError):
        receipt_metrics_finite = False
        joint_vectors_finite = False
    if (
        not math.isfinite(teacher_norm)
        or not math.isfinite(frame0_norm)
        or teacher_norm <= 0.0
        or frame0_norm <= 0.0
        or orientation_dot < 1.0 - 1.0e-12
        or physical_vel != [0.0] * 31
        or physical_matches_teacher
        or type(decimation) is not int
        or decimation < 1
        or nominal.get("schema_version") != 1
        or nominal.get("kind") != FRAME0_LIVE_RECEIPT_KIND
        or nominal.get("verdict") != "PASS"
        or nominal.get("action_id") != ACTION_ID
        or nominal.get("motion_sha256") != motion_sha256
        or type(binding) is not dict
        or binding.get("sha256") != dynamic_pin["sha256"]
        or binding.get("content_sha256") != dynamic_content_sha256
        or nominal.get("teacher_reference_unchanged") is not True
        or nominal.get("teacher_physical_birth_separated") is not True
        or nominal.get("candidate_physical_birth_written") is not True
        or nominal.get("candidate_hold_qdes_and_delay_history_installed") is not True
        or nominal.get("plant_contract_match") is not True
        or nominal.get("terminal_reasons") != []
        or nominal.get("generic_terminated") is not False
        or nominal.get("generic_truncated") is not False
        or type(policy_steps) is not int
        or policy_steps < required_policy_steps
        or physics_steps != policy_steps * decimation
        or float(nominal.get("requested_duration_s", math.nan)) != expected_duration_s
        or float(nominal.get("completed_duration_s", math.nan)) != expected_duration_s
        or set(nominal.get("active_terminations", [])) != set(FULL_ACTIVE_TERMINATIONS)
        or nominal.get("both_feet_contact_fraction") != 1.0
        or type(joint) is not dict
        or joint.get("schema_version") != 1
        or joint.get("complete") is not True
        or joint.get("joint_order") != dynamic.get("robot", {}).get("joint_names")
        or joint.get("current_actual_hard_edge_joint_count") != 0
        or joint.get("current_actual_hard_edge_joint_names") != []
        or joint.get("substep_actual_hard_edge_joint_count") != 0
        or joint.get("substep_actual_hard_edge_joint_names") != []
        or type(joint.get("final_minimum_hard_gap_rad")) not in (int, float)
        or type(joint.get("final_minimum_hard_gap_rad")) is bool
        or float(joint.get("final_minimum_hard_gap_rad", math.nan)) <= 0.0
        or not receipt_metrics_finite
        or not joint_vectors_finite
    ):
        raise LaunchRefused(
            "split-ready receipt does not cover hidden WAIT with separated teacher"
        )
    unsigned = {
        "schema_version": 1,
        "kind": SPLIT_READY_RESET_WAIT_GATE_KIND,
        "diagnostic_unauthorized": True,
        "physical_reset_source": "dynamic_ready.physical_ready",
        "teacher_source": "measured_motion.frame0",
        "teacher_physical_birth_separated": True,
        "hidden_wait_required_policy_steps": required_policy_steps,
        "hidden_wait_required_physics_steps": required_policy_steps * decimation,
        "observed_policy_steps": policy_steps,
        "observed_physics_steps": physics_steps,
        "policy_dt_s": POLICY_DT_S,
        "control_decimation": decimation,
        "time_to_teacher_start_at_reveal_s": pre_swing_wait_s,
        "initial_center_timing_authority": {
            "claim_sha256": timing_claim_sha256,
            "family": timing_unsigned["family"],
            "timing_mode": timing_unsigned["timing_mode"],
            "literal_center_question_sha256": timing_unsigned[
                "literal_center_question_sha256"
            ],
            "receipt": timing_unsigned["receipt"],
            "action_manifest": timing_unsigned["action_manifest"],
            "solver_profile_sha256": timing_unsigned["solver_profile_sha256"],
            "physics_profile_sha256": timing_unsigned["physics_profile_sha256"],
            "sampling_profile_sha256": timing_unsigned["sampling_profile_sha256"],
            "sample_receipt_sha256": timing_unsigned["sample_receipt_sha256"],
        },
        "bridge_learning_signal": "dense_mimic_after_task_reveal",
        "passive_hold_after_reveal_required": False,
        "dynamic_ready": {"sha256": dynamic_pin["sha256"], "content_sha256": dynamic_content_sha256},
        "nominal_hold_receipt": {"sha256": nominal_pin["sha256"], "content_sha256": nominal_content_sha256},
    }
    return {**unsigned, "claim_sha256": canonical_sha256(unsigned)}


def _validate_lineage(
    checkout: Path, commit: str, value: Any
) -> dict[str, Any]:
    """Validate the active split-ready A lineage without retired tape births."""

    pin, row = _tracked_json(checkout, commit, value, name="A211 lineage")
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
            "actor_layout_identity",
            "task_profile",
            "gym_task",
            "target_semantics",
            "runtime_target_contract",
            "curriculum_scope",
            "action_id",
            "teacher_id",
            "seed",
            "motion",
            "action_manifest",
            "initial_center_task_receipt",
            "dynamic_ready_artifact",
            "dynamic_ready_nominal_receipt",
            "teacher_frame0_artifact",
            "dr_l0_manifest",
        ),
        name="A211 lineage",
    )
    _assert_no_retired_contract(row, name="A211 lineage")
    expected = {
        "schema_version": 5,
        "kind": LINEAGE_KIND,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "actor_layout_identity": _actor_layout_identity(),
        "task_profile": TASK_PROFILE_ID,
        "gym_task": GYM_TASK_ID,
        "target_semantics": TARGET_SEMANTICS,
        "runtime_target_contract": _runtime_target_contract(),
        "curriculum_scope": _curriculum_scope_contract(),
        "action_id": ACTION_ID,
        "teacher_id": TEACHER_ID,
        "seed": 0,
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("A211 split-ready lineage identity differs")
    pins: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for key in (
        "motion",
        "action_manifest",
        "initial_center_task_receipt",
        "dynamic_ready_artifact",
        "dynamic_ready_nominal_receipt",
        "teacher_frame0_artifact",
    ):
        normalized, path = _B._verify_tracked_file(
            checkout,
            commit,
            _pin(row[key], name="lineage.%s" % key),
            name="A211 %s" % key,
        )
        pins[key] = normalized
        paths[key] = path
    if (
        pins["dynamic_ready_artifact"]["sha256"]
        != SPLIT_READY_DYNAMIC_ARTIFACT_SHA256
        or pins["dynamic_ready_nominal_receipt"]["sha256"]
        != SPLIT_READY_NOMINAL_HOLD_SHA256
        or pins["teacher_frame0_artifact"]["sha256"]
        != SPLIT_READY_TEACHER_FRAME0_ARTIFACT_SHA256
    ):
        raise LaunchRefused("A211 split-ready authority bytes differ")
    documents = {
        key: _B._strict_json_bytes(path.read_bytes(), name="A211 %s" % key)
        for key, path in paths.items()
        if key != "motion"
    }
    teacher = _validate_teacher_frame0_artifact(
        documents["teacher_frame0_artifact"],
        motion_path=paths["motion"],
        motion_sha256=pins["motion"]["sha256"],
    )
    timing = _initial_center_timing_authority(
        receipt=documents["initial_center_task_receipt"],
        receipt_pin=pins["initial_center_task_receipt"],
        action_manifest=documents["action_manifest"],
        action_manifest_pin=pins["action_manifest"],
        motion_sha256=pins["motion"]["sha256"],
        family="A",
    )
    reset_wait = _split_ready_reset_wait_semantics(
        dynamic=documents["dynamic_ready_artifact"],
        nominal=documents["dynamic_ready_nominal_receipt"],
        dynamic_pin=pins["dynamic_ready_artifact"],
        nominal_pin=pins["dynamic_ready_nominal_receipt"],
        teacher_frame0=teacher["frame0"],
        motion_sha256=pins["motion"]["sha256"],
        initial_center_timing_authority=timing,
    )
    dr_l0_manifest = _dr_l0_manifest_binding(
        checkout, commit, family="A", task_profile=TASK_PROFILE_ID
    )
    if row["dr_l0_manifest"] != dr_l0_manifest:
        raise LaunchRefused("A211 DR-L0 lineage binding differs")
    return {
        **expected,
        **pins,
        "dr_l0_manifest": dr_l0_manifest,
        "teacher_frame0_artifact_content_sha256": teacher["content_sha256"],
        "initial_center_timing_authority": timing,
        "split_ready_reset_wait_authority": reset_wait,
        "artifact": pin,
        "lineage_sha256": pin["sha256"],
    }


def _arm_contract(arm_id: str) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise LaunchRefused("arm_id must select one of the two formal A211 grid cells")
    manifest = _isaac_four_grid_manifest()
    matched = manifest["matched_contract"]
    cell = _four_grid_cell(arm_id, task_family="A211")
    arm = json.loads(json.dumps(ARMS[arm_id]))
    # 探索包是本轮唯一的注册差异轴,所以它不能从 matched_contract 取(那里已经没有这三个键),
    # 必须逐字段对上本格 cell;本地手抄表与 sealed cell 任一处不同即拒。
    exploration = {key: arm.pop(key) for key in _F.EXPLORATION_CELL_KEYS if key in arm}
    if (
        arm["soft_weights"] != matched["soft_weights"]
        or arm["reference_guard_mode"] != matched["reference_guard_mode"]
        or arm["ppo"] != cell["ppo"]
        or arm["ppo"] != matched["ppo"]
        or set(exploration) != {"actor_init_mode", "init_noise_std", "noise_std_type"}
        or any(cell[key] != value for key, value in exploration.items())
    ):
        raise LaunchRefused("A211 arm differs from the sealed Isaac four-grid cell")
    try:
        _F.validate_exploration_package(cell)
    except _F.FourGridContractError as exc:
        raise LaunchRefused("A211 exploration package differs: %s" % exc) from exc
    payload = {
        "schema_version": 2,
        "kind": "action_ball_a211_learnability_arm_v2",
        "arm_id": arm_id,
        "four_grid_cell_id": arm_id,
        "isaac_four_grid_manifest_sha256": manifest["content_sha256"],
        "ppo_adaptation_axis": cell["ppo_adaptation_axis"],
        "contact_sigma_adaptation": False,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "fresh_normalizers_required": True,
        "exploration_axis": cell["exploration_axis"],
        "actor_init_mode": cell["actor_init_mode"],
        "four_sigma_hard_inner_gate_applies": cell[
            "four_sigma_hard_inner_gate_applies"
        ],
        "init_noise_std": cell["init_noise_std"],
        "noise_std_type": cell["noise_std_type"],
        "entropy_coef": matched["entropy_coef"],
        "actor_hidden_dims": matched["actor_hidden_dims"],
        "critic_hidden_dims": matched["critic_hidden_dims"],
        **arm,
    }
    return {**payload, "arm_contract_sha256": canonical_sha256(payload)}


def _planned_materialization(
    *, arm: Mapping[str, Any], lineage: Mapping[str, Any]
) -> dict[str, Any]:
    """Plan the code-owned arm and reward identity before runtime composition."""

    reward = {
        "soft_weights": arm["soft_weights"],
        "reference_guard_mode": arm["reference_guard_mode"],
        "weight_independent_projection_exposure_required": True,
    }
    unsigned = {
        "schema_version": 1,
        "kind": MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "reward_contract_sha256": canonical_sha256(reward),
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "dr_l0_manifest": lineage["dr_l0_manifest"],
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _runtime_reward_materialization(
    *,
    path: Path,
    planned: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the planned arm to the actually composed runtime reward recipe."""

    validated = _OLD._validate_reward_materialization(
        {"path": str(path), "sha256": _B.sha256_file(path)}
    )
    document = _B._strict_json_bytes(path.read_bytes(), name="A211 reward materialization")
    _require_effective_learnability_terms(document["terms"])
    observed_weights = _runtime_effective_soft_weights(
        document["terms"], arm=arm
    )
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
            "runtime_soft_weights": observed_weights,
        }
    )
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _runtime_effective_soft_weights(
    terms: Any, *, arm: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve the four arm-owned weights from effective RewardManager terms."""

    if type(terms) is not list:
        raise LaunchRefused("A211 runtime effective reward terms are malformed")
    runtime_terms = {}
    for term in terms:
        if type(term) is not dict or type(term.get("name")) is not str:
            raise LaunchRefused("A211 runtime effective reward term is malformed")
        name = term["name"]
        if name in runtime_terms:
            raise LaunchRefused("A211 runtime effective reward term name is duplicated")
        runtime_terms[name] = term
    expected_weights = {
        "death_penalty": arm["soft_weights"]["death_penalty"],
        "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
        "joint_limit": arm["soft_weights"]["joint_limit"],
    }
    observed_weights = {}
    for name in sorted(expected_weights):
        term = runtime_terms.get(name)
        if type(term) is not dict:
            observed_weights[name] = None
        elif name == "qdes_projection_penalty":
            params = term.get("params")
            manager_weight = term.get("weight")
            objective_weight = (
                params.get("objective_weight") if type(params) is dict else None
            )
            if (
                type(manager_weight) not in (int, float)
                or manager_weight != -1.0
                or type(objective_weight) not in (int, float)
            ):
                observed_weights[name] = None
            else:
                observed_weights[name] = objective_weight
        else:
            observed_weights[name] = term.get("weight")
    if observed_weights != {
        name: expected_weights[name] for name in sorted(expected_weights)
    }:
        raise LaunchRefused(
            "runtime effective reward soft weights differ from the selected A211 arm"
        )
    return observed_weights


def _require_effective_learnability_terms(terms: Any) -> None:
    """Fail closed unless the runtime recipe retained A211's learning signal."""

    if type(terms) is not list:
        raise LaunchRefused("A211 runtime effective reward terms are malformed")
    by_name = {}
    for term in terms:
        if type(term) is not dict or type(term.get("name")) is not str:
            raise LaunchRefused("A211 runtime effective reward term is malformed")
        name = term["name"]
        if name in by_name:
            raise LaunchRefused("A211 runtime effective reward term name is duplicated")
        by_name[name] = term
    for name, required in REQUIRED_EFFECTIVE_TERMS.items():
        term = by_name.get(name)
        if type(term) is not dict:
            raise LaunchRefused("A211 required effective term is absent: %s" % name)
        if term.get("callable") != required["callable"]:
            raise LaunchRefused("A211 required effective callable differs: %s" % name)
        if type(term.get("weight")) not in (int, float) or term["weight"] <= 0.0:
            raise LaunchRefused("A211 required effective term is not positive: %s" % name)
        params = term.get("params")
        if type(params) is not dict:
            raise LaunchRefused("A211 required effective params are malformed: %s" % name)
        for key, value in required["params"].items():
            if params.get(key) != value:
                raise LaunchRefused(
                    "A211 required effective gate identity differs: %s.%s" % (name, key)
                )


def _runtime_policy_materialization(
    *,
    path: Path,
    checkout: Path,
    lineage: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and bind the exact dynamic-ready PPO recipe emitted by train.py."""

    bundle = {
        "core": {
            "dynamic_ready": {
                "artifact": lineage["dynamic_ready_artifact"],
                "nominal_hold_receipt": lineage[
                    "dynamic_ready_nominal_receipt"
                ],
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
        raise LaunchRefused("runtime policy recipe validation failed") from exc
    document = _B._strict_json_bytes(
        path.read_bytes(), name="A211 policy materialization"
    )
    runner = document["action_ball_ppo_runner_recipe"]["recipe"]
    policy = runner.get("policy") if type(runner) is dict else None
    algorithm = runner.get("algorithm") if type(runner) is dict else None
    runner_settings = runner.get("runner") if type(runner) is dict else None
    expected_policy = {
        "actor_hidden_dims": arm["actor_hidden_dims"],
        "critic_hidden_dims": arm["critic_hidden_dims"],
        "init_noise_std": arm["init_noise_std"],
        "noise_std_type": arm["noise_std_type"],
    }
    expected_algorithm = {
        "entropy_coef": arm["entropy_coef"],
        **arm["ppo"],
    }
    if (
        type(policy) is not dict
        or any(policy.get(key) != value for key, value in expected_policy.items())
        or type(algorithm) is not dict
        or any(
            algorithm.get(key) != value
            for key, value in expected_algorithm.items()
        )
        or type(runner_settings) is not dict
        or runner_settings.get("empirical_normalization") is not True
        or runner_settings.get("init_at_random_ep_len") is not False
    ):
        raise LaunchRefused(
            "runtime policy recipe differs from the selected A211 PPO arm"
        )
    unsigned = {
        "schema_version": 1,
        "kind": POLICY_MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "runtime_policy_recipe_artifact": validated["artifact"],
        "runtime_policy_recipe_sha256": validated[
            "policy_contract_sha256"
        ],
        "dynamic_ready_binding_sha256": validated[
            "dynamic_ready_binding_sha256"
        ],
        "noise_std_type": validated["noise_std_type"],
        "configured_and_realized_init_noise_std": validated[
            "configured_and_realized_init_noise_std"
        ],
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _validated_stage_result(
    value: Any, *, expected_stage: str, name: str
) -> tuple[dict[str, str], dict[str, Any]]:
    pin, row = _canonical_external_json(value, name=name)
    keys = (
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
            "arm_materialization",
            "policy_recipe_materialization",
            "oracle32_receipt",
            "predecessor_result",
            "content_sha256",
    )
    has_terminal_acceptance = (
        type(row) is dict and "terminal_acceptance" in row
    )
    legacy_reward_only = (
        expected_stage == "materialize"
        and type(row) is dict
        and set(row) == set(keys) - {"policy_recipe_materialization"}
    )
    row = _exact_dict(
        row,
        tuple(
            key
            for key in keys
            if not (legacy_reward_only and key == "policy_recipe_materialization")
        ) + (("terminal_acceptance",) if has_terminal_acceptance else ()),
        name=name,
    )
    unsigned = dict(row)
    seal = unsigned.pop("content_sha256")
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
        or _B._sha256(seal, name="%s content SHA" % name)
        != canonical_sha256(unsigned)
    ):
        raise LaunchRefused("%s identity differs" % name)
    if expected_stage == "scale4096":
        if not has_terminal_acceptance or type(row["terminal_acceptance"]) is not dict:
            raise LaunchRefused(
                "A211 scale4096 result lacks terminal checkpoint/safety acceptance"
            )
    elif has_terminal_acceptance:
        raise LaunchRefused(
            "%s contains a scale4096-only terminal acceptance" % name
        )
    if legacy_reward_only:
        row = {**row, "policy_recipe_materialization": None}
    return pin, row


def _stable_artifact_bytes(path: Path, *, name: str, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    """Read one real regular file while binding its inode, size, and digest."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise LaunchRefused("%s is missing" % name) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > max_bytes
        or path.resolve(strict=True) != path
    ):
        raise LaunchRefused("%s must be a bounded real regular file" % name)
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise LaunchRefused("%s cannot be read stably" % name) from exc
    if (
        len(raw) != before.st_size
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise LaunchRefused("%s changed while it was audited" % name)
    return raw, {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _finite_tensor_tree(value: Any, *, name: str, torch_module: Any) -> dict[str, int]:
    """Count and check every tensor below one required RSL checkpoint subtree."""

    tensor_count = 0
    element_count = 0

    def visit(item: Any) -> None:
        nonlocal tensor_count, element_count
        if torch_module.is_tensor(item):
            tensor_count += 1
            element_count += int(item.numel())
            try:
                finite = bool(torch_module.isfinite(item).all().item())
            except (RuntimeError, TypeError, ValueError) as exc:
                raise LaunchRefused(
                    "A211 scale4096 checkpoint %s tensor cannot be audited" % name
                ) from exc
            if not finite:
                raise LaunchRefused(
                    "A211 scale4096 checkpoint %s contains a non-finite tensor" % name
                )
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if tensor_count <= 0 or element_count <= 0:
        raise LaunchRefused(
            "A211 scale4096 checkpoint %s contains no tensors" % name
        )
    return {"tensor_count": tensor_count, "element_count": element_count}


def _checkpoint_run_dir(
    *, log_raw: bytes, checkout: Path, namespace: Path
) -> Path:
    """Resolve the one RSL log directory printed by this exact namespace."""

    try:
        lines = log_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LaunchRefused("A211 scale4096 run log is not UTF-8") from exc
    marker = " | log: "
    candidates = []
    for line in lines:
        if line.startswith("[INFO] Task: ") and marker in line:
            candidates.append(line.rsplit(marker, 1)[1])
    if len(candidates) != 1:
        raise LaunchRefused(
            "A211 scale4096 run log lacks one exact RSL log directory"
        )
    run_dir = _B._absolute_path(candidates[0], name="scale4096 RSL log directory")
    root = checkout / _B.WBT_RELATIVE / "logs" / "rsl_rl" / EXPERIMENT_NAME
    expected_suffix = "_%s-DIAGNOSTIC_UNAUTHORIZED" % namespace.name
    if (
        not run_dir.name.endswith(expected_suffix)
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}"
            + re.escape(expected_suffix),
            run_dir.name,
        )
        is None
    ):
        raise LaunchRefused("A211 scale4096 RSL run name differs")
    try:
        if (
            root.resolve(strict=True) != root
            or run_dir.resolve(strict=True) != run_dir
            or run_dir.parent != root
            or not stat.S_ISDIR(run_dir.lstat().st_mode)
        ):
            raise LaunchRefused("A211 scale4096 RSL run directory escapes checkout")
    except OSError as exc:
        raise LaunchRefused("A211 scale4096 RSL run directory is missing") from exc
    return run_dir


def _terminal_json_events(log_raw: bytes, *, prefix: str, name: str) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = log_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LaunchRefused("A211 scale4096 run log is not UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant %s" % value)

    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            row = json.loads(line[len(prefix) :], parse_constant=reject_constant)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LaunchRefused("A211 scale4096 %s JSON is invalid" % name) from exc
        if type(row) is not dict:
            raise LaunchRefused("A211 scale4096 %s must be a JSON object" % name)
        rows.append(row)
    return rows


def _plain_counter(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise LaunchRefused("A211 scale4096 %s must be a nonnegative integer" % name)
    return value


def _ordered_terminal_events(
    rows: list[dict[str, Any]],
    *,
    event: str,
    schema_version: int,
    expected_updates: int,
    name: str,
) -> list[dict[str, Any]]:
    if (
        len(rows) != expected_updates
        or any(
            row.get("event") != event
            or row.get("schema_version") != schema_version
            or row.get("ppo_update") != index
            for index, row in enumerate(rows)
        )
    ):
        raise LaunchRefused(
            "A211 scale4096 %s lacks exactly %d contiguous terminal updates"
            % (name, expected_updates)
        )
    return rows


def _prelong_terminal_gate_binding(
    *,
    log_raw: bytes,
    run_log: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    safety_counters: Mapping[str, Any],
    launch_claim_sha256: str,
) -> dict[str, Any]:
    """Consume the five real semantic markers and bind PASS to exact artifacts."""

    if (
        _S.PRELONG_SEMANTICS_EVENT != _P.SEMANTIC_EVENT
        or _S.PRELONG_SEMANTICS_SCHEMA_VERSION != _P.SEMANTIC_SCHEMA_VERSION
    ):
        raise LaunchRefused("A211 pre-long producer and gate schemas differ")
    try:
        log_text = log_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LaunchRefused("A211 scale4096 run log is not UTF-8") from exc
    semantic_updates = _terminal_json_events(
        log_raw,
        prefix=_S.PRELONG_SEMANTICS_MARKER_PREFIX,
        name="pre-long semantic marker",
    )
    try:
        gate = _P.validate_prelong_gate(
            log_text=log_text,
            checkpoint_acceptance={
                "checkpoint": checkpoint,
                "safety_counters": safety_counters,
            },
            semantic_updates=semantic_updates,
        )
    except _P.PreLongGateRefused as exc:
        raise LaunchRefused("A211 scale4096 pre-long gate rejected: %s" % exc) from exc
    safe_gate = dict(gate)
    safe_gate["finite_model"] = safe_gate.pop("checkpoint")
    unsigned = {
        "schema_version": 1,
        "kind": "action_ball_a211_4096x5_prelong_gate_binding_v1",
        "diagnostic_unauthorized": True,
        "launch_claim_sha256": launch_claim_sha256,
        "run_log_sha256": run_log["sha256"],
        "finite_model_sha256": checkpoint["sha256"],
        "semantic_marker_prefix": _S.PRELONG_SEMANTICS_MARKER_PREFIX,
        "semantic_update_count": len(semantic_updates),
        "gate": safe_gate,
        "gate_sha256": canonical_sha256(safe_gate),
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _audit_scale4096_terminal(
    *,
    checkout: Path,
    namespace: Path,
    launch_claim_sha256: str,
) -> dict[str, Any]:
    """Recompute the pre-long checkpoint and observed-safety acceptance.

    ``torch.load(weights_only=True)`` still parses PyTorch's pickle container;
    it is not a general untrusted-file sandbox.  We therefore load only the
    exact checkpoint located below this clean checkout's exact RSL run
    directory, after binding its inode/size/SHA.  There is deliberately no
    fallback to ordinary pickle loading when weights-only rejects a file.
    """

    expected_updates = BUDGETS["scale4096"][1]
    log_path = namespace / "run.log"
    log_raw, log_artifact = _stable_artifact_bytes(
        log_path, name="A211 scale4096 terminal run log", max_bytes=512 << 20
    )
    run_dir = _checkpoint_run_dir(
        log_raw=log_raw, checkout=checkout, namespace=namespace
    )
    checkpoint_path = run_dir / ("model_%d.pt" % expected_updates)
    checkpoint_raw, checkpoint_artifact = _stable_artifact_bytes(
        checkpoint_path,
        name="A211 scale4096 exact checkpoint",
        max_bytes=32 << 30,
    )
    try:
        import torch as torch_module
    except ImportError as exc:  # pragma: no cover - exact Pod dependency
        raise LaunchRefused(
            "PyTorch is required to audit the A211 scale4096 checkpoint"
        ) from exc
    try:
        checkpoint = torch_module.load(
            io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise LaunchRefused(
            "A211 scale4096 checkpoint failed safe CPU weights-only load; "
            "ordinary pickle execution is forbidden"
        ) from exc
    if type(checkpoint) is not dict:
        raise LaunchRefused("A211 scale4096 checkpoint root must be a dict")
    embedded_iteration = checkpoint.get("iter")
    infos = checkpoint.get("infos")
    if (
        type(embedded_iteration) is not int
        or embedded_iteration != expected_updates
        or type(infos) is not dict
        or infos.get("training_launch_claim_sha256") != launch_claim_sha256
    ):
        raise LaunchRefused(
            "A211 scale4096 checkpoint iteration/launch-claim binding differs"
        )
    tensor_groups = {}
    for key, label in (
        ("model_state_dict", "model"),
        ("optimizer_state_dict", "optimizer"),
        ("obs_norm_state_dict", "actor_normalizer"),
        ("privileged_obs_norm_state_dict", "critic_normalizer"),
    ):
        subtree = checkpoint.get(key)
        if not isinstance(subtree, Mapping) or not subtree:
            raise LaunchRefused(
                "A211 scale4096 checkpoint lacks %s state" % label
            )
        tensor_groups[label] = _finite_tensor_tree(
            subtree, name=label, torch_module=torch_module
        )

    joint_rows = _ordered_terminal_events(
        _terminal_json_events(
            log_raw,
            prefix="HOPE_JOINT_SAFETY_UPDATE_JSON=",
            name="joint-safety counter",
        ),
        event="hope_joint_safety_diagnostic_compact_update",
        schema_version=1,
        expected_updates=expected_updates,
        name="joint-safety counters",
    )
    actual_rows = _ordered_terminal_events(
        _terminal_json_events(
            log_raw,
            prefix="HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON=",
            name="actual-hard counter",
        ),
        event="action_ball_actual_joint_forbidden_diagnostic_update",
        schema_version=2,
        expected_updates=expected_updates,
        name="actual-hard counters",
    )
    reward_rows = _ordered_terminal_events(
        _terminal_json_events(
            log_raw,
            prefix="HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=",
            name="reward-safety counter",
        ),
        event="hope_reward_safety_transition_update",
        schema_version=2,
        expected_updates=expected_updates,
        name="reward-safety counters",
    )
    behavior_rows = _ordered_terminal_events(
        _terminal_json_events(
            log_raw,
            prefix="HOPE_EXACT_BEHAVIOR_UPDATE_JSON=",
            name="exact-behavior counter",
        ),
        event="hope_exact_behavior_update",
        schema_version=1,
        expected_updates=expected_updates,
        name="exact-behavior counters",
    )

    actual_hard_edge_count = 0
    for index, row in enumerate(joint_rows):
        totals = row.get("counter_totals")
        if (
            row.get("status")
            != "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
            or type(totals) is not dict
            or "actual_hard_edge_events" not in totals
        ):
            raise LaunchRefused(
                "A211 scale4096 joint-safety terminal counter %d is incomplete"
                % index
            )
        actual_hard_edge_count += _plain_counter(
            totals["actual_hard_edge_events"],
            name="actual_hard_edge_events",
        )

    actual_hard_terminal_count = 0
    physics_nonfinite_count = 0
    for index, row in enumerate(actual_rows):
        if row.get("enabled") is not True or "total_hard_terminal_count" not in row:
            raise LaunchRefused(
                "A211 scale4096 actual-hard terminal counter %d is incomplete"
                % index
            )
        actual_hard_terminal_count += _plain_counter(
            row["total_hard_terminal_count"],
            name="actual hard terminal count",
        )
        control = row.get("physx_control_position_limits")
        if type(control) is not dict or control.get("enabled") is not True:
            raise LaunchRefused(
                "A211 scale4096 actual-hard control telemetry is missing"
            )
        by_joint = control.get("by_joint")
        if type(by_joint) is not list or not by_joint:
            raise LaunchRefused(
                "A211 scale4096 actual-hard control telemetry has no joints"
            )
        for joint in by_joint:
            sides = joint.get("sides") if type(joint) is dict else None
            if type(sides) is not dict or set(sides) != {"lower", "upper"}:
                raise LaunchRefused(
                    "A211 scale4096 actual-hard control side telemetry is incomplete"
                )
            for side in sides.values():
                if type(side) is not dict or type(
                    side.get("nonfinite_readback_observed")
                ) is not bool:
                    raise LaunchRefused(
                        "A211 scale4096 actual-hard nonfinite counter is missing"
                    )
                physics_nonfinite_count += int(
                    side["nonfinite_readback_observed"]
                )

    strict_hard_termination_count = 0
    table_contact_count = 0
    joint_qdes_terminal_count = 0
    joint_actual_terminal_count = 0
    reward_fall_by_reason = {reason: 0 for reason in PHYSICAL_FALL_REASONS}
    for index, row in enumerate(reward_rows):
        transitions = row.get("terminal_transitions")
        if row.get("coverage") != "complete_update" or type(transitions) is not list:
            raise LaunchRefused(
                "A211 scale4096 reward-safety terminal counter %d is incomplete"
                % index
            )
        for transition in transitions:
            terms = transition.get("termination_terms") if type(transition) is dict else None
            if (
                type(terms) is not list
                or not terms
                or any(type(term) is not str for term in terms)
            ):
                raise LaunchRefused(
                    "A211 scale4096 terminal transition lacks reason counters"
            )
            term_set = set(terms)
            strict_hard_termination_count += int(
                bool(term_set & set(STRICT_HARD_TERMINATION_UNION))
            )
            table_contact_count += int("robot_hit_table" in term_set)
            joint_qdes_terminal_count += int("joint_qdes_forbidden" in term_set)
            joint_actual_terminal_count += int("joint_actual_forbidden" in term_set)
            for reason in PHYSICAL_FALL_REASONS:
                reward_fall_by_reason[reason] += int(reason in term_set)

    required_nonfinite_counters = {
        "ready_nonfinite_value_count",
        "strike_window_entry_racket_target_distance_nonfinite_count",
        "virtual_contact_nonfinite_reject_count",
    }
    behavior_nonfinite_count = 0
    task_wait_started_by_update = []
    task_reveal_reached_by_update = []
    behavior_fall_by_reason = {reason: 0 for reason in PHYSICAL_FALL_REASONS}
    physical_fall_by_reason_phase = {
        reason: {phase: 0 for phase in PHYSICAL_FALL_PHASES}
        for reason in PHYSICAL_FALL_REASONS
    }
    table_contact_by_phase = {phase: 0 for phase in PHYSICAL_FALL_PHASES}
    behavior_table_contact_count = 0
    for index, row in enumerate(behavior_rows):
        counters = row.get("counters")
        required_balance_counters = {
            TASK_WAIT_STARTED_COUNTER,
            TASK_REVEAL_REACHED_COUNTER,
            "termination_reason_robot_hit_table_count",
            *(
                f"termination_reason_{reason}_count"
                for reason in PHYSICAL_FALL_REASONS
            ),
            *(
                f"termination_reason_{reason}_{phase}_count"
                for reason in PHYSICAL_FALL_REASONS
                for phase in PHYSICAL_FALL_PHASES
            ),
            *(
                f"termination_reason_robot_hit_table_{phase}_count"
                for phase in PHYSICAL_FALL_PHASES
            ),
        }
        if (
            type(counters) is not dict
            or not required_nonfinite_counters.issubset(counters)
            or not required_balance_counters.issubset(counters)
        ):
            raise LaunchRefused(
                "A211 scale4096 exact-behavior nonfinite counters and survival/fall counters %d are missing"
                % index
            )
        task_wait_started_by_update.append(
            _plain_counter(
                counters[TASK_WAIT_STARTED_COUNTER],
                name="task wait started count",
            )
        )
        task_reveal_reached_by_update.append(
            _plain_counter(
                counters[TASK_REVEAL_REACHED_COUNTER],
                name="task reveal reached count",
            )
        )
        for reason in PHYSICAL_FALL_REASONS:
            reason_total = _plain_counter(
                counters[f"termination_reason_{reason}_count"],
                name=f"{reason} terminal count",
            )
            phase_counts = {
                phase: _plain_counter(
                    counters[f"termination_reason_{reason}_{phase}_count"],
                    name=f"{reason} {phase} terminal count",
                )
                for phase in PHYSICAL_FALL_PHASES
            }
            if sum(phase_counts.values()) != reason_total:
                raise LaunchRefused(
                    f"A211 scale4096 {reason} reason-by-phase counters do not conserve "
                    f"in update {index}"
                )
            behavior_fall_by_reason[reason] += reason_total
            for phase, count in phase_counts.items():
                physical_fall_by_reason_phase[reason][phase] += count
        table_total = _plain_counter(
            counters["termination_reason_robot_hit_table_count"],
            name="robot_hit_table terminal count",
        )
        table_phase_counts = {
            phase: _plain_counter(
                counters[f"termination_reason_robot_hit_table_{phase}_count"],
                name=f"robot_hit_table {phase} terminal count",
            )
            for phase in PHYSICAL_FALL_PHASES
        }
        if sum(table_phase_counts.values()) != table_total:
            raise LaunchRefused(
                "A211 scale4096 robot_hit_table reason-by-phase counters do not "
                f"conserve in update {index}"
            )
        behavior_table_contact_count += table_total
        for phase, count in table_phase_counts.items():
            table_contact_by_phase[phase] += count
        for key, value in counters.items():
            if "nonfinite" in key:
                behavior_nonfinite_count += _plain_counter(
                    value, name="exact-behavior %s" % key
                )

    nonfinite_count = physics_nonfinite_count + behavior_nonfinite_count
    if behavior_fall_by_reason != reward_fall_by_reason:
        raise LaunchRefused(
            "A211 scale4096 physical-fall behavior and terminal-transition counts differ"
        )
    if behavior_table_contact_count != table_contact_count:
        raise LaunchRefused(
            "A211 scale4096 table-contact behavior and terminal-transition counts differ"
        )
    safety = {
        "observed_ppo_updates": expected_updates,
        "actual_hard_edge_event_count": actual_hard_edge_count,
        "actual_hard_terminal_count": actual_hard_terminal_count,
        "joint_qdes_forbidden_terminal_count": joint_qdes_terminal_count,
        "joint_actual_forbidden_terminal_count": joint_actual_terminal_count,
        "strict_hard_termination_count": strict_hard_termination_count,
        "table_contact_count": table_contact_count,
        "nonfinite_count": nonfinite_count,
        "base_fell_tilt_terminal_count": behavior_fall_by_reason["base_fell_tilt"],
        "base_too_low_terminal_count": behavior_fall_by_reason["base_too_low"],
        "physical_fall_by_reason_phase": physical_fall_by_reason_phase,
        "table_contact_by_phase": table_contact_by_phase,
        "task_wait_started_by_update": task_wait_started_by_update,
        "task_wait_started_count": sum(task_wait_started_by_update),
        "task_reveal_reached_by_update": task_reveal_reached_by_update,
        "task_reveal_reached_count": sum(task_reveal_reached_by_update),
    }
    strict_zero_keys = (
        "actual_hard_edge_event_count",
        "actual_hard_terminal_count",
        "joint_qdes_forbidden_terminal_count",
        "joint_actual_forbidden_terminal_count",
        "strict_hard_termination_count",
        "nonfinite_count",
    )
    if any(safety[key] != 0 for key in strict_zero_keys):
        raise LaunchRefused(
            "A211 scale4096 observed joint-qdes/joint-actual/nonfinite implementation counters are nonzero"
        )
    checkpoint_acceptance = {
        **checkpoint_artifact,
        "filename_iteration": expected_updates,
        "embedded_iteration": embedded_iteration,
        "map_location": "cpu",
        "load_mode": "torch_weights_only",
        "tensor_groups": tensor_groups,
        "all_tensors_finite": True,
    }
    prelong_gate = _prelong_terminal_gate_binding(
        log_raw=log_raw,
        run_log=log_artifact,
        checkpoint=checkpoint_acceptance,
        safety_counters=safety,
        launch_claim_sha256=launch_claim_sha256,
    )
    unsigned = {
        "schema_version": 1,
        "kind": SCALE4096_TERMINAL_ACCEPTANCE_KIND,
        "diagnostic_unauthorized": True,
        "launch_claim_sha256": launch_claim_sha256,
        "run_log": log_artifact,
        "checkpoint": checkpoint_acceptance,
        "safety_counters": safety,
        "prelong_gate": prelong_gate,
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _validate_materialization(value: Any, *, arm: Mapping[str, Any], lineage: Mapping[str, Any]) -> dict:
    pin, result = _validated_stage_result(
        value, expected_stage="materialize", name="A211 materialize result"
    )
    if (
        result["policy_recipe_materialization"] is not None
        or result["oracle32_receipt"] is not None
    ):
        raise LaunchRefused("A211 materialize result contains downstream receipts")
    row = result["arm_materialization"]
    materialization_keys = (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "arm_id",
            "lineage_sha256",
            "arm_contract_sha256",
            "reward_contract_sha256",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "trainability_contract",
            "dr_l0_manifest",
            "runtime_effective_reward_artifact",
            "runtime_effective_reward_sha256",
            "runtime_effective_reward_term_count",
            "runtime_soft_weights",
            "content_sha256",
    )
    legacy_planned_policy = (
        type(row) is dict
        and set(row) == set(materialization_keys) | {"policy_contract_sha256"}
    )
    row = _exact_dict(
        row,
        materialization_keys
        + (("policy_contract_sha256",) if legacy_planned_policy else ()),
        name="A211 arm materialization",
    )
    _assert_no_retired_contract(row, name="A211 arm materialization")
    unsigned = dict(row)
    seal = unsigned.pop("content_sha256")
    if _B._sha256(seal, name="materialization content SHA") != canonical_sha256(unsigned):
        raise LaunchRefused("A211 arm materialization content seal differs")
    if legacy_planned_policy:
        _B._sha256(
            row["policy_contract_sha256"], name="legacy planned policy contract SHA"
        )
    expected = {
        "schema_version": 1,
        "kind": MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "dr_l0_manifest": lineage["dr_l0_manifest"],
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("A211 arm materialization binding differs")
    reward_sha = _B._sha256(row["reward_contract_sha256"], name="reward contract SHA")
    runtime_reward_sha = _B._sha256(
        row["runtime_effective_reward_sha256"],
        name="runtime effective reward SHA",
    )
    runtime_artifact = row["runtime_effective_reward_artifact"]
    if (
        type(runtime_artifact) is not dict
        or set(runtime_artifact) != {"path", "sha256"}
        or type(runtime_artifact["path"]) is not str
        or not runtime_artifact["path"]
        or _B._sha256(
            runtime_artifact["sha256"], name="runtime reward artifact SHA"
        )
        != runtime_artifact["sha256"]
        or type(row["runtime_effective_reward_term_count"]) is not int
        or row["runtime_effective_reward_term_count"] <= 0
    ):
        raise LaunchRefused("A211 runtime reward materialization binding differs")
    expected_runtime_weights = {
        "death_penalty": arm["soft_weights"]["death_penalty"],
        "joint_limit": arm["soft_weights"]["joint_limit"],
        "qdes_limit_barrier": arm["soft_weights"]["qdes_limit"],
        "qdes_projection_penalty": arm["soft_weights"]["qdes_projection"],
    }
    if row["runtime_soft_weights"] != expected_runtime_weights:
        raise LaunchRefused("A211 runtime reward soft weights differ")
    return {
        "materialize_result": pin,
        **expected,
        "reward_contract_sha256": reward_sha,
        "runtime_effective_reward_artifact": runtime_artifact,
        "runtime_effective_reward_sha256": runtime_reward_sha,
        "runtime_effective_reward_term_count": row[
            "runtime_effective_reward_term_count"
        ],
        "runtime_soft_weights": expected_runtime_weights,
        "content_sha256": seal,
    }


def _validate_policy_recipe_materialization(
    value: Any,
    *,
    checkout: Path,
    arm: Mapping[str, Any],
    lineage: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    pin, result = _validated_stage_result(
        value, expected_stage="recipe", name="A211 recipe result"
    )
    if result["oracle32_receipt"] is not None:
        raise LaunchRefused("A211 recipe result contains an oracle32 receipt")
    row = result["policy_recipe_materialization"]
    prior = result["arm_materialization"]
    if (
        type(prior) is not dict
        or prior.get("content_sha256") != materialization["content_sha256"]
    ):
        raise LaunchRefused("A211 recipe reward materialization lineage differs")
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "arm_id",
            "lineage_sha256",
            "arm_contract_sha256",
            "runtime_policy_recipe_artifact",
            "runtime_policy_recipe_sha256",
            "dynamic_ready_binding_sha256",
            "noise_std_type",
            "configured_and_realized_init_noise_std",
            "content_sha256",
        ),
        name="A211 policy recipe materialization",
    )
    _assert_no_retired_contract(row, name="A211 policy recipe materialization")
    unsigned = dict(row)
    seal = unsigned.pop("content_sha256")
    if _B._sha256(seal, name="policy materialization content SHA") != canonical_sha256(
        unsigned
    ):
        raise LaunchRefused("A211 policy materialization content seal differs")
    expected = {
        "schema_version": 1,
        "kind": POLICY_MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("A211 policy materialization binding differs")
    artifact = row["runtime_policy_recipe_artifact"]
    if type(artifact) is not dict or set(artifact) != {"path", "sha256"}:
        raise LaunchRefused("A211 runtime policy artifact pin differs")
    runtime = _runtime_policy_materialization(
        path=_B._absolute_path(
            artifact["path"], name="runtime policy recipe artifact", must_exist=True
        ),
        checkout=checkout,
        lineage=lineage,
        arm=arm,
    )
    if any(
        row[key] != runtime[key]
        for key in (
            "runtime_policy_recipe_artifact",
            "runtime_policy_recipe_sha256",
            "dynamic_ready_binding_sha256",
            "noise_std_type",
            "configured_and_realized_init_noise_std",
        )
    ):
        raise LaunchRefused("A211 runtime policy materialization binding differs")
    return {
        "recipe_result": pin,
        **expected,
        "runtime_policy_recipe_artifact": runtime[
            "runtime_policy_recipe_artifact"
        ],
        "runtime_policy_recipe_sha256": runtime[
            "runtime_policy_recipe_sha256"
        ],
        "dynamic_ready_binding_sha256": runtime[
            "dynamic_ready_binding_sha256"
        ],
        "noise_std_type": runtime["noise_std_type"],
        "configured_and_realized_init_noise_std": runtime[
            "configured_and_realized_init_noise_std"
        ],
        "content_sha256": seal,
    }


def _validate_oracle32(
    value: Any,
    *,
    arm: Mapping[str, Any],
    lineage: Mapping[str, Any],
    materialization: Mapping[str, Any],
    policy_materialization: Mapping[str, Any],
) -> dict:
    pin, result = _validated_stage_result(
        value, expected_stage="oracle32", name="A211 oracle32 result"
    )
    result_policy = result["policy_recipe_materialization"]
    if (
        type(result_policy) is not dict
        or result_policy.get("content_sha256")
        != policy_materialization["content_sha256"]
    ):
        raise LaunchRefused("A211 oracle policy recipe lineage differs")
    row = result["oracle32_receipt"]
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "verdict",
            "episodes",
            "arm_id",
            "lineage_sha256",
            "arm_contract_sha256",
            "reward_contract_sha256",
            "runtime_effective_reward_sha256",
            "policy_contract_sha256",
            "runtime_policy_recipe_sha256",
            "actor_contract",
            "actor_width",
            "critic_contract",
            "critic_width",
            "trainability_contract",
            "seed",
            "raw_oracle_sha256",
            "content_sha256",
        ),
        name="A211 oracle32 receipt",
    )
    _assert_no_retired_contract(row, name="A211 oracle32 receipt")
    unsigned = dict(row)
    seal = unsigned.pop("content_sha256")
    if _B._sha256(seal, name="oracle32 content SHA") != canonical_sha256(unsigned):
        raise LaunchRefused("A211 oracle32 content seal differs")
    expected = {
        "schema_version": 1,
        "kind": ORACLE32_KIND,
        "diagnostic_unauthorized": True,
        "verdict": "PASS",
        "episodes": 32,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "reward_contract_sha256": materialization["reward_contract_sha256"],
        "runtime_effective_reward_sha256": _B._sha256(
            row["runtime_effective_reward_sha256"],
            name="runtime effective reward SHA",
        ),
        "policy_contract_sha256": policy_materialization[
            "runtime_policy_recipe_sha256"
        ],
        "runtime_policy_recipe_sha256": policy_materialization[
            "runtime_policy_recipe_sha256"
        ],
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "seed": lineage["seed"],
        "raw_oracle_sha256": _B._sha256(
            row["raw_oracle_sha256"], name="raw oracle SHA"
        ),
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("A211 oracle32 receipt binding differs")
    if (
        expected["runtime_effective_reward_sha256"]
        != materialization["runtime_effective_reward_sha256"]
    ):
        raise LaunchRefused(
            "A211 oracle runtime reward differs from materialized runtime reward"
        )
    if row["policy_contract_sha256"] != row["runtime_policy_recipe_sha256"]:
        raise LaunchRefused(
            "A211 oracle runtime policy differs from materialized runtime policy"
        )
    return {"oracle32_result": pin, **expected, "content_sha256": seal}


def _validate_predecessor_result(
    value: Any,
    *,
    checkout: Path,
    expected_stage: str,
    materialization: Mapping[str, Any],
    policy_materialization: Mapping[str, Any],
    oracle32: Mapping[str, Any],
) -> dict[str, Any]:
    pin, result = _validated_stage_result(
        value,
        expected_stage=expected_stage,
        name="A211 %s predecessor result" % expected_stage,
    )
    expected_materialization_sha = materialization["content_sha256"]
    expected_policy_sha = policy_materialization["content_sha256"]
    expected_oracle_sha = oracle32["content_sha256"]
    if (
        not isinstance(result["arm_materialization"], dict)
        or result["arm_materialization"].get("content_sha256")
        != expected_materialization_sha
        or not isinstance(result["policy_recipe_materialization"], dict)
        or result["policy_recipe_materialization"].get("content_sha256")
        != expected_policy_sha
        or not isinstance(result["oracle32_receipt"], dict)
        or result["oracle32_receipt"].get("content_sha256") != expected_oracle_sha
    ):
        raise LaunchRefused("A211 predecessor arm/oracle lineage differs")
    terminal_attestation = None
    if expected_stage == "scale4096":
        expected_completion = {
            "completion_exit_code": "0",
            "terminal_kind": "clean_completion",
            "terminal_exit_code": "0",
        }
        output_contract = result["output_contract"]
        if (
            result["completion"] != expected_completion
            or type(output_contract) is not dict
            or output_contract.get("ppo_update_count")
            != BUDGETS["scale4096"][1]
            or output_contract.get("finite_model_save_interval")
            != BUDGETS["scale4096"][2]
        ):
            raise LaunchRefused(
                "A211 scale4096 predecessor lacks an exact finite natural-exit receipt"
            )
        namespace = _B._absolute_path(
            result["namespace"],
            name="A211 scale4096 predecessor namespace",
            must_exist=True,
        )
        recomputed_terminal = _audit_scale4096_terminal(
            checkout=checkout,
            namespace=namespace,
            launch_claim_sha256=result["launch_claim_sha256"],
        )
        if result["terminal_acceptance"] != recomputed_terminal:
            raise LaunchRefused(
                "A211 scale4096 predecessor terminal checkpoint/safety acceptance differs"
            )
        terminal_attestation = {
            "completion": expected_completion,
            "ppo_update_count": BUDGETS["scale4096"][1],
            "finite_model_save_interval": BUDGETS["scale4096"][2],
            # "checkpoint" is retired resume-control vocabulary elsewhere in
            # the A211 claim.  This is an audited output artifact, not a
            # resume input, so retain its complete binding under an explicit
            # finite-model evidence name.
            "finite_model_artifact": recomputed_terminal["checkpoint"],
            "safety_counters": recomputed_terminal["safety_counters"],
            "prelong_gate": recomputed_terminal["prelong_gate"],
            "terminal_acceptance_content_sha256": recomputed_terminal[
                "content_sha256"
            ],
            "launch_result_content_sha256": result["content_sha256"],
        }
    return {
        "artifact": pin,
        "stage": expected_stage,
        "launch_claim_sha256": result["launch_claim_sha256"],
        "arm_materialization_content_sha256": expected_materialization_sha,
        "policy_recipe_materialization_content_sha256": expected_policy_sha,
        "oracle32_content_sha256": expected_oracle_sha,
        "terminal_attestation": terminal_attestation,
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
            "spec.gpu.require_empty must be %s when allow_vendor_v2_colocation=%s"
            % (str(expected_empty).lower(), str(allow_colocation).lower())
        )
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


def _wait_contract() -> dict[str, Any]:
    return {
        "policy_dt_s": POLICY_DT_S,
        "schedule": dict(WAIT_SCHEDULE),
        "in_loop_expansion_prohibited": True,
    }


def _validate_spec(document: Any, *, claimed: bool = False) -> dict[str, Any]:
    keys = (
            "schema_version",
            "kind",
            "source",
            "arm_id",
            "lineage",
            "arm_materialization",
            "policy_recipe_materialization",
            "oracle32_receipt",
            "predecessor_result",
            "four_grid_scale4096_receipt",
            "stage",
            "num_envs",
            "max_iterations",
            "save_interval",
            "wait_contract",
            "gpu",
            "namespace",
            "log_path",
    )
    actual = frozenset(document) if type(document) is dict else frozenset()
    required = frozenset(keys)
    optional = frozenset((COLOCATION_SPEC_KEY,))
    if not required.issubset(actual) or not actual.issubset(required | optional):
        raise LaunchRefused(
            "A211 launch spec keys differ: missing=%s extra=%s"
            % (sorted(required - actual), sorted(actual - required - optional))
        )
    row = dict(document)
    allow_colocation = row.get(COLOCATION_SPEC_KEY, False)
    if type(allow_colocation) is not bool:
        raise LaunchRefused("allow_vendor_v2_colocation must be a boolean")
    _assert_no_retired_contract(row, name="A211 launch spec")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused("A211 launch spec schema/kind differs")
    if row["wait_contract"] != _wait_contract():
        raise LaunchRefused("A211 launch wait schedule differs")
    source = _exact_dict(
        row["source"], ("checkout", "commit_sha", "isaac_python"), name="spec.source"
    )
    checkout = _B._absolute_path(source["checkout"], name="source.checkout", must_exist=True)
    commit = source["commit_sha"]
    if type(commit) is not str or _B.COMMIT_RE.fullmatch(commit) is None:
        raise LaunchRefused("source.commit_sha must be exact lowercase 40-hex")
    python = _isaac_python_entry(source["isaac_python"])
    arm = _arm_contract(row["arm_id"])
    stage = row["stage"]
    if stage not in BUDGETS:
        raise LaunchRefused("stage must be materialize, recipe, oracle32, smoke, probe512, long512, scale4096, or long4096")
    if allow_colocation and stage not in COLOCATED_STAGES:
        raise LaunchRefused(
            "VendorV2 colocation is restricted to scale4096/long4096 and is excluded from rate evidence"
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
        raise LaunchRefused("namespace parent must be the dedicated A211 experiment root")
    expected_parent = (
        Path(row["source"]["checkout"])
        / _B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / EXPERIMENT_NAME
    )
    if parent != expected_parent:
        raise LaunchRefused(
            "namespace parent must be the checkout-local A211 experiment root"
        )
    log_path = _B._absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must equal <namespace>/run.log")
    if stage == "materialize":
        if (
            row["arm_materialization"] is not None
            or row["policy_recipe_materialization"] is not None
            or row["oracle32_receipt"] is not None
        ):
            raise LaunchRefused("materialize stage must start without generated receipts")
    elif row["arm_materialization"] is None:
        raise LaunchRefused("stage requires its same-arm materialization receipt")
    if stage in ("materialize", "recipe"):
        if row["policy_recipe_materialization"] is not None:
            raise LaunchRefused(
                "%s stage must not predeclare a policy recipe receipt" % stage
            )
    elif row["policy_recipe_materialization"] is None:
        raise LaunchRefused("stage requires its same-arm policy recipe receipt")
    if stage in ("smoke", "probe512", "long512", "scale4096", "long4096"):
        if row["oracle32_receipt"] is None:
            raise LaunchRefused(
                "%s requires its same-arm oracle32 PASS receipt" % stage
            )
    elif row["oracle32_receipt"] is not None:
        raise LaunchRefused(
            "only smoke, probe512, long512, scale4096, and long4096 consume an oracle32 receipt"
        )
    expected_predecessor = {
        "probe512": "smoke",
        "long512": "probe512",
        "long4096": "scale4096",
    }.get(stage)
    if expected_predecessor is None:
        if row["predecessor_result"] is not None:
            raise LaunchRefused("stage must not consume a predecessor result")
    elif row["predecessor_result"] is None:
        raise LaunchRefused(
            "%s requires a completed %s result" % (stage, expected_predecessor)
        )
    if (stage == "long4096") is not (
        row["four_grid_scale4096_receipt"] is not None
    ):
        raise LaunchRefused(
            "%s four-grid scale4096 receipt requirement differs" % stage
        )
    four_grid_receipt = (
        _four_grid_prelong_receipt_pin(row["four_grid_scale4096_receipt"])
        if stage == "long4096"
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {"checkout": str(checkout), "commit_sha": commit, "isaac_python": str(python)},
        "arm_id": arm["arm_id"],
        "lineage": _pin(row["lineage"], name="spec.lineage"),
        "arm_materialization": row["arm_materialization"],
        "policy_recipe_materialization": row[
            "policy_recipe_materialization"
        ],
        "oracle32_receipt": row["oracle32_receipt"],
        "predecessor_result": row["predecessor_result"],
        "four_grid_scale4096_receipt": four_grid_receipt,
        "stage": stage,
        "num_envs": actual_budget[0],
        "max_iterations": actual_budget[1],
        "save_interval": actual_budget[2],
        "wait_contract": _wait_contract(),
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


def _training_argv(spec: Mapping[str, Any], lineage: Mapping[str, Any], arm: Mapping[str, Any]) -> list[str]:
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / _B.WBT_RELATIVE
    motion = checkout / lineage["motion"]["path"]
    manifest = checkout / lineage["action_manifest"]["path"]
    dynamic_ready = checkout / lineage["dynamic_ready_artifact"]["path"]
    dynamic_receipt = checkout / lineage["dynamic_ready_nominal_receipt"]["path"]
    action_list = json.dumps([lineage["action_id"]], separators=(",", ":"))
    motion_list = json.dumps([str(motion)], separators=(",", ":"))
    ppo = arm["ppo"]
    weights = arm["soft_weights"]
    materialization = (
        _planned_materialization(arm=arm, lineage=lineage)
        if spec["stage"] == "materialize"
        else _validate_materialization(
            spec["arm_materialization"], arm=arm, lineage=lineage
        )
    )
    policy_materialization = (
        None
        if spec["stage"] in ("materialize", "recipe")
        else _validate_policy_recipe_materialization(
            spec["policy_recipe_materialization"],
            checkout=checkout,
            arm=arm,
            lineage=lineage,
            materialization=materialization,
        )
    )
    policy_sha = (
        RECIPE_SENTINEL_POLICY_SHA256
        if policy_materialization is None
        else policy_materialization["runtime_policy_recipe_sha256"]
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
        # 探索包是本轮的注册差异轴(exp §5.6.2c):零权重格是 0.1/log,标准初始化格是
        # 1.0/scalar。三个 override 一起从选中的 cell 里发出,不再硬钉字面量。
        "algo.policy.init_noise_std=%s"
        % _float_override_token(
            arm["init_noise_std"], name="algo.policy.init_noise_std"
        ),
        "algo.policy.noise_std_type=%s" % arm["noise_std_type"],
        "action_ball_actor_init_mode=%s" % arm["actor_init_mode"],
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
        "task.domain_rand.stable_ready_plant=true",
        "task.motion.action_ball_diagnostic_split_ready_teacher=true",
        "action_ball_dynamic_ready_bootstrap=true",
        "action_ball_dynamic_ready_artifact_path=%s" % dynamic_ready,
        "action_ball_dynamic_ready_artifact_sha256=%s"
        % lineage["dynamic_ready_artifact"]["sha256"],
        "action_ball_dynamic_ready_nominal_receipt_path=%s" % dynamic_receipt,
        "action_ball_dynamic_ready_nominal_receipt_sha256=%s"
        % lineage["dynamic_ready_nominal_receipt"]["sha256"],
        "motion_file=%s" % motion_list,
        "task.racket.clip_names=%s" % action_list,
        "task.racket.action_ball_manifest_path=%s" % manifest,
        "task.racket.action_ball_manifest_sha256=%s"
        % lineage["action_manifest"]["sha256"],
        "task.racket.action_ball_policy_contract_sha256=%s"
        % policy_sha,
        "task.racket.action_ball_seed=%d" % lineage["seed"],
        "task.racket.action_ball_target_source=online_solver",
        "task.racket.action_ball_reuse_exact_question_until_semantics_change=true",
        "task.racket.action_ball_initial_center_single_question=true",
        "task.racket.action_ball_diagnostic_unauthorized=true",
        "task.racket.adaptive_sigma=false",
        "task.racket.adaptive_sigma_monotonic=false",
        "task.racket.adaptive_sigma_normal=false",
        "+task.racket.reference_guard_mode=%s" % arm["reference_guard_mode"],
        "task.rewards.death_penalty_weight=%s" % weights["death_penalty"],
        "task.rewards.qdes_limit_barrier_weight=%s" % weights["qdes_limit"],
        "+task.rewards.qdes_projection_penalty_weight=%s" % weights["qdes_projection"],
        "task.rewards.joint_limit_weight=%s" % weights["joint_limit"],
        "task.actions.control_step_action_delay_min=0",
        "task.actions.control_step_action_delay_max=0",
        "task.push.enable=false",
        "task.physical_ball=false",
        "task.racket.virtual_ball=true",
        "task.racket.action_ball_target_observation_noise=false",
        "task.racket.question_bank=",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
    ]
    if spec["stage"] == "oracle32":
        argv.extend(
            [
                "+action_ball_teacher_qdes_oracle_output_path=%s"
                % (Path(spec["namespace"]) / "teacher_qdes_oracle_32ep.json"),
                "+action_ball_teacher_qdes_oracle_episodes=32",
            ]
        )
    if spec["stage"] == "materialize":
        argv.extend(
            [
                "+n1_vendor_sigma_profile=%s" % REWARD_MATERIALIZATION_PROFILE,
                "+action_ball_effective_reward_recipe_output_path=%s"
                % (Path(spec["namespace"]) / REWARD_RECIPE_FILENAME),
            ]
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
    return argv


def _output_contract(
    spec: Mapping[str, Any], lineage: Mapping[str, Any]
) -> dict[str, Any]:
    stage = spec["stage"]
    profile_contract = _update_profile_contract(os.environ)
    if (
        profile_contract["forwarded_value"] is not None
        and spec[COLOCATION_SPEC_KEY]
    ):
        raise LaunchRefused(
            "ActionBall update profiling requires an exclusive GPU claim"
        )
    rate_isolation = (
        "excluded_colocated_diagnostic"
        if spec[COLOCATION_SPEC_KEY]
        else "excluded_scale_finite_gate"
        if stage == "scale4096"
        else "excluded_profile_instrumented_diagnostic"
        if profile_contract["mode"] == "profile_on_attribution_only"
        else "excluded_no_matched_abba_speed_stage"
    )
    output = {
        "ppo_update_count": spec["max_iterations"],
        "finite_model_save_interval": spec["save_interval"],
        "arm_materialization_embedded_in_claim": stage == "materialize",
        "policy_recipe_materialization_embedded_in_claim": stage == "recipe",
        "policy_recipe": None,
        "teacher_qdes_oracle32": None,
        "boot_marker": "Learning iteration",
        "iter500_quantitative_threshold_status": "UNSET",
        "iter500_action": "diagnostic_continue_only",
        "automatic_winner_selection_prohibited": True,
        "wait_contract": _wait_contract(),
        "teacher_frame0_artifact_sha256": lineage["teacher_frame0_artifact"][
            "sha256"
        ],
        "teacher_frame0_artifact_content_sha256": lineage[
            "teacher_frame0_artifact_content_sha256"
        ],
        "split_ready_reset_wait_claim_sha256": lineage[
            "split_ready_reset_wait_authority"
        ]["claim_sha256"],
        "physical_reset_semantics": "separate_safe_ready_zero_velocity",
        "teacher_reveal_semantics": "measured_frame0_with_public_countdown",
        "passive_hold_after_reveal_required": False,
        "dr_l0_manifest": lineage["dr_l0_manifest"],
        # No current finite/long stage implements the required matched
        # profiler-off A->C->C->A benchmark.  In particular, scale4096 owns a
        # five-update heavy pre-long ledger and can never be speed evidence.
        "speed_benchmark_eligible": False,
        "rate_evidence_eligible": False,
        "rate_evidence_isolation": rate_isolation,
        "update_profile": profile_contract,
        "deferred_matched_speed_measurement": (
            _deferred_speed_measurement_contract()
        ),
        "colocated_stage": (
            spec["stage"] if spec[COLOCATION_SPEC_KEY] else None
        ),
        "max_compute_processes_per_gpu": MAX_COLOCATED_PROCESSES_PER_GPU,
        "colocation_result_scope": "training_diagnostic_only",
    }
    if stage == "materialize":
        output["effective_reward_recipe"] = str(
            Path(spec["namespace"]) / REWARD_RECIPE_FILENAME
        )
        output["boot_marker"] = "ACTION_BALL_EFFECTIVE_REWARD_RECIPE_MATERIALIZED_JSON"
    elif stage == "recipe":
        output["policy_recipe"] = str(
            Path(spec["namespace"]) / POLICY_RECIPE_FILENAME
        )
        output["boot_marker"] = "ACTION_BALL_POLICY_RECIPE_MATERIALIZED"
    elif stage == "oracle32":
        output["teacher_qdes_oracle32"] = str(
            Path(spec["namespace"]) / "teacher_qdes_oracle_32ep.json"
        )
        output["boot_marker"] = "ACTION_BALL_TEACHER_QDES_ORACLE_COMPLETE_JSON"
    return output


def _normalizer_contract() -> dict[str, Any]:
    return {
        "actor": {"identity": ACTOR_NORMALIZER_IDENTITY, "state": "fresh_empty"},
        "critic": {"identity": CRITIC_NORMALIZER_IDENTITY, "state": "fresh_empty"},
        "distinct_objects_required": True,
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
        "strict_hard_termination_count_max": 0,
        "behavioral_termination_policy": (
            "fall_too_low_and_table_remain_terminal_but_are_reported_by_reason_phase_"
            "without_an_unvalidated_numeric_cutoff"
        ),
        "nonfinite_count_max": 0,
        "finite_model_required_when_updates_positive": True,
        "oracle32_pass_required_for_training_stages": True,
        "scale4096_required_for_long4096": True,
        "iter500_quantitative_threshold_status": "UNSET",
        "iter500_action": "diagnostic_continue_only",
        "automatic_winner_selection_prohibited": True,
    }


def _admission_training_argv(
    spec: Mapping[str, Any], bundle: Mapping[str, Any]
) -> list[str]:
    row = _exact_dict(
        bundle,
        (
            "lineage",
            "arm",
            "isaac_four_grid_manifest",
            "normalizers",
            "termination_contract",
            "continuation_stop_gate",
            "curriculum_scope",
        ),
        name="A211 claim bundle",
    )
    if row["isaac_four_grid_manifest"] != _isaac_four_grid_manifest():
        raise LaunchRefused("A211 claim four-grid manifest drifted")
    return _training_argv(spec, row["lineage"], row["arm"])


def _admission_output_contract(
    spec: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    bundle = payload.get("bundle")
    if type(bundle) is not dict or type(bundle.get("lineage")) is not dict:
        raise LaunchRefused("A211 co-resident claim lineage is malformed")
    return _output_contract(spec, bundle["lineage"])


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
    output_contract_from_payload=_admission_output_contract,
    physical_reservation_registry=True,
    forbidden_namespace_experiment_names=(
        "agibot_a3_action_ball_measured_vendor_v2_n1_diagnostic",
        "agibot_a3_action_ball_a225_four_arm_diagnostic",
        "agibot_a3_action_ball_c225_diagnostic",
        "agibot_a3_action_ball_c211_diagnostic",
    ),
)
_open_gpu_shared_lock = _ADMISSION._open_gpu_shared_lock
_lock_gpu_admission = _ADMISSION._lock_gpu_admission
_unlock_gpu_admission = _ADMISSION._unlock_gpu_admission
_query_gpu_processes = _ADMISSION._query_gpu_processes
_validate_runtime_gpu_process = _ADMISSION._validate_runtime_gpu_process
_live_reservations = _ADMISSION._live_reservations
_reservation_document = _ADMISSION._reservation_document
_write_reservation = _ADMISSION._write_reservation
_release_reservation = _ADMISSION._release_reservation
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


def _validate_raw_oracle32(
    path: Path,
    *,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the trainer's generic oracle and distill an A211-bound receipt."""

    _B._stable_regular_file(path, name="A211 raw oracle32")
    raw = path.read_bytes()
    row = _B._strict_json_bytes(raw, name="A211 raw oracle32")
    if raw != _B._canonical_bytes(row) + b"\n":
        raise LaunchRefused("A211 raw oracle32 must be canonical JSON plus newline")
    row = _exact_dict(
        row,
        (
            "schema_version",
            "kind",
            "diagnostic_unauthorized",
            "bindings",
            "completion",
            "phase_by_termination",
            "exact_strike",
            "capture_rejection",
            "measurement_contract",
            "safety_exposure",
            "teacher_qdes",
            "episodes",
        ),
        name="A211 raw oracle32",
    )
    if (
        row["schema_version"] != 2
        or row["kind"] != "action_ball_teacher_qdes_dynamic_oracle_v2"
        or row["diagnostic_unauthorized"] is not True
    ):
        raise LaunchRefused("A211 raw oracle32 schema/kind/authorization differs")
    bindings = _exact_dict(
        row["bindings"],
        (
            "source_sha256",
            "task_sha256",
            "hard_contract_sha256",
            "reward_sha256",
            "policy_sha256",
            "policy_contract_sha256",
            "dynamic_ready_sha256",
            "dynamic_ready_artifact_sha256",
            "dynamic_ready_nominal_hold_sha256",
            "manifest_sha256",
            "motion_sha256",
            "target_provider_contract_sha256",
        ),
        name="A211 raw oracle32 bindings",
    )
    for name, digest in bindings.items():
        _B._sha256(digest, name="A211 raw oracle32 %s" % name)
    bundle = claim["bundle"]
    lineage = bundle["lineage"]
    arm = bundle["arm"]
    materialization = claim["materialization_inputs"]["arm_materialization"]
    policy_materialization = claim["materialization_inputs"][
        "policy_recipe_materialization"
    ]
    materialized_reward = _runtime_reward_materialization(
        path=_B._absolute_path(
            materialization["runtime_effective_reward_artifact"]["path"],
            name="A211 materialized runtime reward artifact",
            must_exist=True,
        ),
        planned=_planned_materialization(arm=arm, lineage=lineage),
        arm=arm,
    )
    reward_materialization_fields = (
        "runtime_effective_reward_artifact",
        "runtime_effective_reward_sha256",
        "runtime_effective_reward_term_count",
        "runtime_soft_weights",
    )
    if any(
        materialized_reward[name] != materialization[name]
        for name in reward_materialization_fields
    ):
        raise LaunchRefused(
            "A211 raw oracle32 runtime reward differs from revalidated materialization"
        )
    sources = claim["runtime_sources"]
    expected_bindings = {
        "source_sha256": sources["training entrypoint"]["sha256"],
        "task_sha256": sources["A211 DR-L0 task profile"]["sha256"],
        "reward_sha256": materialized_reward[
            "runtime_effective_reward_sha256"
        ],
        "policy_sha256": policy_materialization[
            "runtime_policy_recipe_sha256"
        ],
        "policy_contract_sha256": policy_materialization[
            "runtime_policy_recipe_sha256"
        ],
        "dynamic_ready_sha256": policy_materialization[
            "dynamic_ready_binding_sha256"
        ],
        "dynamic_ready_artifact_sha256": lineage["dynamic_ready_artifact"]["sha256"],
        "dynamic_ready_nominal_hold_sha256": lineage[
            "dynamic_ready_nominal_receipt"
        ]["sha256"],
        "manifest_sha256": lineage["action_manifest"]["sha256"],
        "motion_sha256": lineage["motion"]["sha256"],
    }
    if any(bindings[key] != value for key, value in expected_bindings.items()):
        raise LaunchRefused("A211 raw oracle32 lineage bindings differ")
    spec = claim["spec"]
    checkout = Path(spec["source"]["checkout"])
    root = checkout / _B.WBT_RELATIVE / "logs/rsl_rl" / EXPERIMENT_NAME
    suffix = "_%s-DIAGNOSTIC_UNAUTHORIZED" % Path(spec["namespace"]).name
    candidates = (
        []
        if not root.is_dir()
        else [candidate for candidate in root.iterdir() if candidate.name.endswith(suffix)]
    )
    if len(candidates) != 1:
        raise LaunchRefused("A211 raw oracle32 has no unique runtime directory")
    hard_contract = candidates[0] / "params/training_contract.json"
    _B._stable_regular_file(hard_contract, name="A211 oracle32 hard contract")
    if _B.sha256_file(hard_contract) != bindings["hard_contract_sha256"]:
        raise LaunchRefused("A211 oracle32 hard-contract SHA differs")
    try:
        hard_document = _B._strict_json_bytes(
            hard_contract.read_bytes(), name="A211 oracle32 hard contract"
        )
        training_contract = _OLD._load_training_contract_module(checkout)
        training_contract.validate_schema3_contract_structure(hard_document)
        diagnostic = training_contract.validate_action_ball_training_authorization(
            hard_document
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise LaunchRefused(
            "A211 oracle32 hard contract is not an authorized schema-3 diagnostic"
        ) from exc
    expected_hard_identity = {
        "schema_version": 3,
        "target_mode": "action_ball",
        "actor_obs_contract": ACTOR_CONTRACT,
        "actor_obs_total_dim": ACTOR_WIDTH,
        "actor_obs_term_names": [name for name, _width in ACTOR_ORDERED_LAYOUT],
        "actor_obs_term_dims": [width for _name, width in ACTOR_ORDERED_LAYOUT],
        "critic_obs_contract": CRITIC_CONTRACT,
        "critic_obs_total_dim": CRITIC_WIDTH,
        "action_ball_211_trainability_contract": TRAINABILITY_CONTRACT,
        "action_ball_task_wait_contract": _wait_contract(),
        "actor_obs_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_obs_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
    }
    if diagnostic is not True or any(
        hard_document.get(key) != value
        for key, value in expected_hard_identity.items()
    ):
        raise LaunchRefused("A211 oracle32 hard-contract ABI/authorization differs")
    try:
        resolved_dr_l0 = training_contract.action_ball_dr_l0_contract_payload()
        resolved_dr_l0_sha256 = (
            training_contract.action_ball_dr_l0_contract_sha256()
        )
    except Exception as exc:
        raise LaunchRefused(
            "A211 oracle32 hard contract cannot resolve DR-L0 finalizer"
        ) from exc
    if (
        hard_document.get("action_ball_dr_l0") != resolved_dr_l0
        or canonical_sha256(resolved_dr_l0) != resolved_dr_l0_sha256
        or lineage.get("dr_l0_manifest", {}).get("contract_sha256")
        != resolved_dr_l0_sha256
        or lineage.get("dr_l0_manifest", {}).get("hard_contract_identity")
        != resolved_dr_l0.get("identity")
    ):
        raise LaunchRefused("A211 oracle32 hard-contract DR-L0 binding differs")
    try:
        runtime_target = hard_document["action_ball_training"]["runtime"][
            "target_provider"
        ]
    except (KeyError, TypeError) as exc:
        raise LaunchRefused(
            "A211 oracle32 hard-contract target provider is missing"
        ) from exc
    if (
        not isinstance(runtime_target, dict)
        or runtime_target.get("source") != "online_solver"
        or runtime_target.get("recipe") != "current_lm"
        or runtime_target.get("validity_mask") != [True, True, True]
        or runtime_target.get("target_observation_noise") is not False
        or runtime_target.get("immutable_tape") is not None
        or not isinstance(
            runtime_target.get("exact_question_answer_reuse"), dict
        )
        or runtime_target["exact_question_answer_reuse"].get("enabled") is not True
        or canonical_sha256(runtime_target)
        != bindings["target_provider_contract_sha256"]
    ):
        raise LaunchRefused(
            "A211 oracle32 target provider is not online exact-question reuse"
        )
    runtime_reward = hard_document.get("effective_reward_recipe")
    runtime_policy = hard_document.get("action_ball_ppo_runner_recipe")
    if (
        not isinstance(runtime_reward, dict)
        or runtime_reward.get("sha256") != bindings["reward_sha256"]
        or not isinstance(runtime_reward.get("terms"), list)
        or not isinstance(runtime_policy, dict)
        or runtime_policy.get("sha256") != bindings["policy_sha256"]
        or not isinstance(runtime_policy.get("recipe"), dict)
    ):
        raise LaunchRefused(
            "A211 oracle32 runtime reward/policy receipt differs from hard contract"
        )
    if (
        runtime_reward.get("schema_version") != 1
        or runtime_reward.get("sha256")
        != canonical_sha256(
            {
                "schema_version": runtime_reward.get("schema_version"),
                "terms": runtime_reward["terms"],
            }
        )
    ):
        raise LaunchRefused("A211 runtime effective reward semantic SHA differs")
    _runtime_effective_soft_weights(runtime_reward["terms"], arm=arm)
    policy_recipe = runtime_policy["recipe"]
    algorithm = policy_recipe.get("algorithm")
    policy = policy_recipe.get("policy")
    ppo = arm["ppo"]
    expected_algorithm = {
        "schedule": ppo["schedule"],
        "learning_rate": ppo["learning_rate"],
        "desired_kl": 0.01,
        "clip_param": 0.2,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "entropy_coef": arm["entropy_coef"],
    }
    expected_policy = {
        "actor_hidden_dims": arm["actor_hidden_dims"],
        "critic_hidden_dims": arm["critic_hidden_dims"],
        "init_noise_std": arm["init_noise_std"],
        "noise_std_type": arm["noise_std_type"],
    }
    if (
        not isinstance(algorithm, dict)
        or not isinstance(policy, dict)
        or any(algorithm.get(name) != value for name, value in expected_algorithm.items())
        or any(policy.get(name) != value for name, value in expected_policy.items())
    ):
        raise LaunchRefused("A211 runtime PPO recipe differs from arm")
    completion = row["completion"]
    exact = row["exact_strike"]
    capture = row["capture_rejection"]
    safety = row["safety_exposure"]
    teacher_qdes = row["teacher_qdes"]
    try:
        failures = _OLD._oracle32_acceptance_failures(
            completion=completion,
            observed=completion["exact_strike_observed_nonterminal"],
            exact_summary=exact,
            capture=capture,
            unknown=completion["pre_strike_or_same_step_unknown"],
            termination=safety["termination"],
            projection=safety["projection"],
            qdes=teacher_qdes,
            soft_limit=safety["soft_limit"],
            reference=safety["reference_guard"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchRefused("A211 raw oracle32 ledger is malformed") from exc
    if failures:
        raise LaunchRefused(
            "A211 oracle32 acceptance failed: %s" % ",".join(failures)
        )
    unsigned = {
        "schema_version": 1,
        "kind": ORACLE32_KIND,
        "diagnostic_unauthorized": True,
        "verdict": "PASS",
        "episodes": 32,
        "arm_id": arm["arm_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "arm_contract_sha256": arm["arm_contract_sha256"],
        "reward_contract_sha256": materialization["reward_contract_sha256"],
        "runtime_effective_reward_sha256": materialized_reward[
            "runtime_effective_reward_sha256"
        ],
        "policy_contract_sha256": policy_materialization[
            "runtime_policy_recipe_sha256"
        ],
        "runtime_policy_recipe_sha256": policy_materialization[
            "runtime_policy_recipe_sha256"
        ],
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "seed": lineage["seed"],
        "raw_oracle_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def build_plan(spec_path: Path) -> dict[str, Any]:
    path = _B._absolute_path(str(spec_path), name="--spec", must_exist=True)
    _B._stable_regular_file(path, name="A211 launch spec")
    raw = path.read_bytes()
    document = _B._strict_json_bytes(raw, name="A211 launch spec")
    if raw != _B._canonical_bytes(document) + b"\n":
        raise LaunchRefused("A211 launch spec must be canonical JSON plus newline")
    spec = _validate_spec(document)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    source = _B._verify_clean_source(checkout, commit)
    runtime_sources = _runtime_sources(checkout, commit)
    runtime_assets = _B._validate_runtime_asset_environment()
    lineage = _validate_lineage(checkout, commit, spec["lineage"])
    arm = _arm_contract(spec["arm_id"])
    materialization = (
        _planned_materialization(arm=arm, lineage=lineage)
        if spec["stage"] == "materialize"
        else _validate_materialization(
            spec["arm_materialization"], arm=arm, lineage=lineage
        )
    )
    policy_materialization = (
        _validate_policy_recipe_materialization(
            spec["policy_recipe_materialization"],
            checkout=checkout,
            arm=arm,
            lineage=lineage,
            materialization=materialization,
        )
        if spec["stage"] not in ("materialize", "recipe")
        else None
    )
    oracle32 = (
        _validate_oracle32(
            spec["oracle32_receipt"],
            arm=arm,
            lineage=lineage,
            materialization=materialization,
            policy_materialization=policy_materialization,
        )
        if spec["stage"] in (
            "smoke",
            "probe512",
            "long512",
            "scale4096",
            "long4096",
        )
        else None
    )
    expected_predecessor = {
        "probe512": "smoke",
        "long512": "probe512",
        "long4096": "scale4096",
    }.get(spec["stage"])
    predecessor = (
        _validate_predecessor_result(
            spec["predecessor_result"],
            checkout=checkout,
            expected_stage=expected_predecessor,
            materialization=materialization,
            policy_materialization=policy_materialization,
            oracle32=oracle32,
        )
        if expected_predecessor is not None
        else None
    )
    four_grid_receipt = (
        _validate_four_grid_prelong_receipt(
            spec["four_grid_scale4096_receipt"], checkout=checkout
        )
        if spec["stage"] == "long4096"
        else None
    )
    output_contract = _output_contract(spec, lineage)
    bundle = {
        "lineage": lineage,
        "arm": arm,
        "isaac_four_grid_manifest": _isaac_four_grid_manifest(),
        "normalizers": _normalizer_contract(),
        "termination_contract": _termination_contract(),
        "continuation_stop_gate": _continuation_stop_gate(),
        "curriculum_scope": _curriculum_scope_contract(),
    }
    materialization_inputs = {
        "arm_materialization": materialization,
        "policy_recipe_materialization": policy_materialization,
        "oracle32_receipt": oracle32,
        "predecessor_result": predecessor,
        "four_grid_scale4096_receipt": four_grid_receipt,
    }
    training_argv = _training_argv(spec, lineage, arm)
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
        "materialization_inputs": materialization_inputs,
        "output_contract": output_contract,
        "boot_marker": output_contract["boot_marker"],
        "training_argv": training_argv,
    }
    retired_check = dict(payload)
    retired_check.pop("resume_prohibited")
    _assert_no_retired_contract(retired_check, name="A211 launch claim")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(payload),
        "canonical_payload": payload,
    }


def _revalidate_claim_payload(
    payload: Mapping[str, Any], *, claimed: bool = True
) -> tuple[dict, dict, dict]:
    spec = _validate_spec(payload["spec"], claimed=claimed)
    payload = _ADMISSION._validate_claim_payload_safety(payload, spec)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    if _B._verify_clean_source(checkout, commit) != payload["source"]:
        raise LaunchRefused("clean source claim drifted")
    if _runtime_sources(checkout, commit) != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted")
    _B._validate_runtime_asset_claim(payload["runtime_assets"])
    lineage = _validate_lineage(checkout, commit, spec["lineage"])
    arm = _arm_contract(spec["arm_id"])
    materialization = (
        _planned_materialization(arm=arm, lineage=lineage)
        if spec["stage"] == "materialize"
        else _validate_materialization(
            spec["arm_materialization"], arm=arm, lineage=lineage
        )
    )
    policy_materialization = (
        _validate_policy_recipe_materialization(
            spec["policy_recipe_materialization"],
            checkout=checkout,
            arm=arm,
            lineage=lineage,
            materialization=materialization,
        )
        if spec["stage"] not in ("materialize", "recipe")
        else None
    )
    oracle32 = (
        _validate_oracle32(
            spec["oracle32_receipt"],
            arm=arm,
            lineage=lineage,
            materialization=materialization,
            policy_materialization=policy_materialization,
        )
        if spec["stage"] in (
            "smoke",
            "probe512",
            "long512",
            "scale4096",
            "long4096",
        )
        else None
    )
    predecessor_stage = {
        "probe512": "smoke",
        "long512": "probe512",
        "long4096": "scale4096",
    }.get(spec["stage"])
    predecessor = (
        _validate_predecessor_result(
            spec["predecessor_result"],
            checkout=checkout,
            expected_stage=predecessor_stage,
            materialization=materialization,
            policy_materialization=policy_materialization,
            oracle32=oracle32,
        )
        if predecessor_stage is not None
        else None
    )
    four_grid_receipt = (
        _validate_four_grid_prelong_receipt(
            spec["four_grid_scale4096_receipt"], checkout=checkout
        )
        if spec["stage"] == "long4096"
        else None
    )
    expected_bundle = {
        "lineage": lineage,
        "arm": arm,
        "isaac_four_grid_manifest": _isaac_four_grid_manifest(),
        "normalizers": _normalizer_contract(),
        "termination_contract": _termination_contract(),
        "continuation_stop_gate": _continuation_stop_gate(),
        "curriculum_scope": _curriculum_scope_contract(),
    }
    expected_inputs = {
        "arm_materialization": materialization,
        "policy_recipe_materialization": policy_materialization,
        "oracle32_receipt": oracle32,
        "predecessor_result": predecessor,
        "four_grid_scale4096_receipt": four_grid_receipt,
    }
    if (
        payload["spec"] != spec
        or payload["bundle"] != expected_bundle
        or payload["materialization_inputs"] != expected_inputs
        or payload["output_contract"] != _output_contract(spec, lineage)
        or payload["boot_marker"] != payload["output_contract"]["boot_marker"]
        or payload["training_argv"] != _training_argv(spec, lineage, arm)
    ):
        raise LaunchRefused("A211 claim lineage, output contract, or training argv drifted")
    return spec, lineage, arm


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
    spec, _lineage, _arm = _revalidate_claim_payload(payload)
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
                "kind": "action_ball_a211_pre_exec_gpu_admission_v1",
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
        **_update_profile_exec_environment(os.environ),
        **_prelong_semantics_exec_environment(
            spec["stage"],
            payload["materialization_inputs"]["arm_materialization"][
                "runtime_effective_reward_sha256"
            ],
        ),
        **_B._runtime_asset_exec_environment(payload["runtime_assets"]),
    }
    os.chdir(wbt)
    argv = payload["training_argv"]
    os.execve(argv[0], argv, environment)
    raise AssertionError("execve returned")


def _validate_completion_state(path: Path) -> dict[str, str]:
    _B._stable_regular_file(path, name="A211 completion state")
    observed: dict[str, str] = {}
    required = {"completion_exit_code", "terminal_kind", "terminal_exit_code"}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in required:
            if key in observed:
                raise LaunchRefused("A211 completion state has duplicate %s" % key)
            observed[key] = value
    if observed != {
        "completion_exit_code": "0",
        "terminal_kind": "clean_completion",
        "terminal_exit_code": "0",
    }:
        raise LaunchRefused("A211 workload did not exit cleanly and uniquely")
    return observed


def _completion_stage(stage: str) -> bool:
    return stage in (
        "materialize",
        "recipe",
        "oracle32",
        "smoke",
        "probe512",
        "scale4096",
    )


def execute(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    expected = _B._sha256(confirm_claim, name="--confirm-claim")
    outer = _exact_dict(
        plan,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="A211 launch plan",
    )
    payload = outer["canonical_payload"]
    if (
        outer["schema_version"] != SCHEMA_VERSION
        or outer["kind"] != CLAIM_KIND
        or expected != outer["launch_claim_sha256"]
        or type(payload) is not dict
        or canonical_sha256(payload) != expected
    ):
        raise LaunchRefused("--confirm-claim differs from freshly recomputed plan")
    spec, _lineage, _arm = _revalidate_claim_payload(payload, claimed=False)
    checkout = Path(spec["source"]["checkout"])
    _B._verify_clean_source(checkout, spec["source"]["commit_sha"])
    _B._validate_runtime_asset_claim(payload["runtime_assets"])
    lock_fd = _open_gpu_shared_lock(Path(spec["gpu"]["lock_path"]))
    namespace = None
    reservation_handle = None
    try:
        _lock_gpu_admission(lock_fd)
        try:
            first = _verify_gpu_admission(
                spec, phase="pre_launch", current_namespace=None
            )
            namespace = _B._claim_namespace(plan)
            reservation_handle = _write_reservation(spec, expected)
            _B._write_exclusive_json(
                namespace / "pre_launch_gpu_admission.json",
                {
                    "schema_version": 1,
                    "kind": "action_ball_a211_pre_launch_gpu_admission_v1",
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
            "KIT_BOOT_MARKER": plan["canonical_payload"]["boot_marker"],
            "KIT_BOOT_TIMEOUT_S": "2700",
            "KIT_BOOT_STALE_TIMEOUT_S": "1800",
            "KIT_BOOT_POLL_S": "5",
            "KIT_BOOT_STATE_FILE": str(state),
            **_update_profile_exec_environment(os.environ),
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
                _B._write_exclusive_json(
                    namespace
                    / (
                        "post_completion_gpu_admission.json"
                        if completion_stage
                        else "post_boot_gpu_admission.json"
                    ),
                    {
                        "schema_version": 1,
                        "kind": (
                            "action_ball_a211_post_completion_gpu_admission_v1"
                            if completion_stage
                            else "action_ball_a211_post_boot_gpu_admission_v1"
                        ),
                        "launch_claim_sha256": expected,
                        "gpu": final_gpu,
                    },
                )
            except (LaunchRefused, FileNotFoundError, ValueError, OSError) as exc:
                if completion_stage:
                    raise LaunchRefused(
                        "post-completion admission refused after exact clean exit"
                    ) from exc
                failure = _cleanup_post_boot_admission_failure(
                    namespace, state, expected, str(exc)
                )
                outcome = (
                    "completed"
                    if failure["cleanup"]["completed"] is True
                    else "incomplete"
                )
                raise LaunchRefused(
                    "post-boot admission refused; exact current-trainer cleanup %s; "
                    "failure receipt=%s" % (outcome, failure["path"])
                ) from exc
        finally:
            _unlock_gpu_admission(lock_fd)
        materialization = plan["canonical_payload"]["materialization_inputs"][
            "arm_materialization"
        ]
        policy_materialization = plan["canonical_payload"][
            "materialization_inputs"
        ]["policy_recipe_materialization"]
        oracle32 = plan["canonical_payload"]["materialization_inputs"][
            "oracle32_receipt"
        ]
        if spec["stage"] == "oracle32":
            oracle32 = _validate_raw_oracle32(
                Path(plan["canonical_payload"]["output_contract"]["teacher_qdes_oracle32"]),
                claim=plan["canonical_payload"],
            )
        elif spec["stage"] == "materialize":
            materialization = _runtime_reward_materialization(
                path=Path(
                    plan["canonical_payload"]["output_contract"][
                        "effective_reward_recipe"
                    ]
                ),
                planned=materialization,
                arm=plan["canonical_payload"]["bundle"]["arm"],
            )
        elif spec["stage"] == "recipe":
            policy_materialization = _runtime_policy_materialization(
                path=Path(
                    plan["canonical_payload"]["output_contract"][
                        "policy_recipe"
                    ]
                ),
                checkout=checkout,
                lineage=plan["canonical_payload"]["bundle"]["lineage"],
                arm=plan["canonical_payload"]["bundle"]["arm"],
            )
        terminal_acceptance = (
            _audit_scale4096_terminal(
                checkout=checkout,
                namespace=namespace,
                launch_claim_sha256=expected,
            )
            if spec["stage"] == "scale4096"
            else None
        )
        unsigned_result = {
            "schema_version": 1,
            "kind": RESULT_KIND,
            "diagnostic_unauthorized": True,
            "accepted": True,
            "launch_claim_sha256": expected,
            "stage": spec["stage"],
            "namespace": str(namespace),
            "completion": completion,
            "gpu_admission": final_gpu,
            "output_contract": plan["canonical_payload"]["output_contract"],
            "arm_materialization": materialization,
            "policy_recipe_materialization": policy_materialization,
            "oracle32_receipt": oracle32,
            "predecessor_result": plan["canonical_payload"][
                "materialization_inputs"
            ]["predecessor_result"],
        }
        if terminal_acceptance is not None:
            unsigned_result["terminal_acceptance"] = terminal_acceptance
        launch_result = {
            **unsigned_result,
            "content_sha256": canonical_sha256(unsigned_result),
        }
        _B._write_exclusive_json(namespace / "launch_result.json", launch_result)
        return launch_result
    finally:
        try:
            if reservation_handle is not None:
                _lock_gpu_admission(lock_fd)
                try:
                    _release_reservation(reservation_handle)
                finally:
                    _unlock_gpu_admission(lock_fd)
        finally:
            os.close(lock_fd)


def _write_template(args: argparse.Namespace) -> dict[str, Any]:
    budget = BUDGETS[args.stage]
    materialization_pair = (
        args.arm_materialization_path,
        args.arm_materialization_sha256,
    )
    policy_pair = (
        args.policy_recipe_materialization_path,
        args.policy_recipe_materialization_sha256,
    )
    oracle_pair = (args.oracle32_receipt_path, args.oracle32_receipt_sha256)
    predecessor_pair = (
        args.predecessor_result_path,
        args.predecessor_result_sha256,
    )
    four_grid_pair = (
        args.four_grid_scale4096_receipt_path,
        args.four_grid_scale4096_receipt_sha256,
    )
    if (materialization_pair[0] is None) != (materialization_pair[1] is None):
        raise LaunchRefused("arm materialization path/SHA must be supplied together")
    if (policy_pair[0] is None) != (policy_pair[1] is None):
        raise LaunchRefused("policy recipe result path/SHA must be supplied together")
    if (oracle_pair[0] is None) != (oracle_pair[1] is None):
        raise LaunchRefused("oracle32 receipt path/SHA must be supplied together")
    if (predecessor_pair[0] is None) != (predecessor_pair[1] is None):
        raise LaunchRefused("predecessor result path/SHA must be supplied together")
    if (four_grid_pair[0] is None) != (four_grid_pair[1] is None):
        raise LaunchRefused(
            "four-grid scale4096 receipt path/SHA must be supplied together"
        )
    if args.stage == "materialize":
        if (
            materialization_pair[0] is not None
            or policy_pair[0] is not None
            or oracle_pair[0] is not None
            or predecessor_pair[0] is not None
            or four_grid_pair[0] is not None
        ):
            raise LaunchRefused("materialize template accepts no generated receipt")
    elif materialization_pair[0] is None:
        raise LaunchRefused("stage requires an A211 materialize result path/SHA")
    if args.stage in ("materialize", "recipe"):
        if policy_pair[0] is not None:
            raise LaunchRefused(
                "%s template does not accept a policy recipe result" % args.stage
            )
    elif policy_pair[0] is None:
        raise LaunchRefused("stage requires an A211 policy recipe result path/SHA")
    if args.stage in ("smoke", "probe512", "long512", "scale4096", "long4096"):
        if oracle_pair[0] is None:
            raise LaunchRefused(
                "%s template requires an oracle32 result path/SHA" % args.stage
            )
    elif oracle_pair[0] is not None:
        raise LaunchRefused(
            "only smoke, probe512, long512, scale4096, and long4096 templates accept an oracle32 result"
        )
    needs_predecessor = args.stage in ("probe512", "long512", "long4096")
    if needs_predecessor is not (predecessor_pair[0] is not None):
        raise LaunchRefused(
            "%s template predecessor-result requirement differs" % args.stage
        )
    if (args.stage == "long4096") is not (four_grid_pair[0] is not None):
        raise LaunchRefused(
            "%s template four-grid scale4096 receipt requirement differs"
            % args.stage
        )
    namespace = Path(args.namespace).resolve(strict=False)
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(Path(args.checkout).resolve(strict=True)),
            "commit_sha": args.commit_sha,
            "isaac_python": str(_isaac_python_entry(args.isaac_python)),
        },
        "arm_id": args.arm_id,
        "lineage": {"path": args.lineage_path, "sha256": args.lineage_sha256},
        "arm_materialization": (
            None
            if args.arm_materialization_path is None
            else {
                "path": args.arm_materialization_path,
                "sha256": args.arm_materialization_sha256,
            }
        ),
        "policy_recipe_materialization": (
            None
            if args.policy_recipe_materialization_path is None
            else {
                "path": args.policy_recipe_materialization_path,
                "sha256": args.policy_recipe_materialization_sha256,
            }
        ),
        "oracle32_receipt": (
            None
            if args.oracle32_receipt_path is None
            else {
                "path": args.oracle32_receipt_path,
                "sha256": args.oracle32_receipt_sha256,
            }
        ),
        "predecessor_result": (
            None
            if args.predecessor_result_path is None
            else {
                "path": args.predecessor_result_path,
                "sha256": args.predecessor_result_sha256,
            }
        ),
        "four_grid_scale4096_receipt": (
            None
            if args.four_grid_scale4096_receipt_path is None
            else {
                "path": args.four_grid_scale4096_receipt_path,
                "sha256": args.four_grid_scale4096_receipt_sha256,
            }
        ),
        "stage": args.stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "wait_contract": _wait_contract(),
        "gpu": {
            "index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "owner": args.owner,
            "lock_path": "/tmp/hope_lean_queue_gpu%d.lock" % args.gpu_index,
            "require_empty": not args.allow_colocation,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }
    if args.allow_colocation:
        document[COLOCATION_SPEC_KEY] = True
    _assert_no_retired_contract(document, name="A211 launch spec")
    document = _validate_spec(document)
    if not args.allow_colocation:
        document.pop(COLOCATION_SPEC_KEY)
    output = Path(args.output).resolve(strict=False)
    _B._write_exclusive_json(output, document)
    return {"status": "CREATED", "spec": str(output), "arm_id": args.arm_id, "stage": args.stage}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--output", required=True)
    template.add_argument("--checkout", required=True)
    template.add_argument("--commit-sha", required=True)
    template.add_argument("--isaac-python", required=True)
    template.add_argument("--arm-id", required=True, choices=ARM_IDS)
    template.add_argument("--lineage-path", required=True)
    template.add_argument("--lineage-sha256", required=True)
    template.add_argument("--arm-materialization-path")
    template.add_argument("--arm-materialization-sha256")
    template.add_argument("--policy-recipe-materialization-path")
    template.add_argument("--policy-recipe-materialization-sha256")
    template.add_argument("--oracle32-receipt-path")
    template.add_argument("--oracle32-receipt-sha256")
    template.add_argument("--predecessor-result-path")
    template.add_argument("--predecessor-result-sha256")
    template.add_argument("--four-grid-scale4096-receipt-path")
    template.add_argument("--four-grid-scale4096-receipt-sha256")
    template.add_argument("--stage", required=True, choices=tuple(BUDGETS))
    template.add_argument("--gpu-index", required=True, type=int)
    template.add_argument("--gpu-uuid", required=True)
    template.add_argument("--owner", required=True)
    template.add_argument("--namespace", required=True)
    template.add_argument("--allow-colocation", action="store_true")
    for command in ("plan", "execute"):
        child = sub.add_parser(command)
        child.add_argument("--spec", required=True)
        if command == "execute":
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
            result = (
                plan
                if args.command == "plan"
                else execute(plan, confirm_claim=args.confirm_claim)
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    except (LaunchRefused, FileNotFoundError, ValueError, OSError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
