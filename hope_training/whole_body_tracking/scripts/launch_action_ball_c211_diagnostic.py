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
C211_EVIDENCE_FILE = SCRIPT_DIR / "action_ball_c211_oracle_evidence.py"
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
A211_LAUNCHER_FILE = SCRIPT_DIR / "launch_action_ball_a211_four_arm_diagnostic.py"


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
_C = _load_helper("_c211_diagnostic_oracle_evidence", C211_EVIDENCE_FILE)
_F = _load_helper("_a211_c211_four_grid_authority", FOUR_GRID_FILE)
_Q = _load_helper("_a211_c211_four_grid_prelong_barrier", FOUR_GRID_BARRIER_FILE)
_P = _load_helper("_c211_4096x5_prelong_gate", PRELONG_GATE_FILE)
_S = _load_helper("_c211_4096x5_prelong_semantics", PRELONG_SEMANTICS_FILE)
_W = _load_helper("_c211_task_wait_schedule", TASK_WAIT_FILE)
_FRAME0 = _load_helper("_c211_shared_frame0_authority", A211_LAUNCHER_FILE)

LaunchRefused = _B.LaunchRefused

SCHEMA_VERSION = 2
SPEC_KIND = "action_ball_c211_diagnostic_spec_v2"
CLAIM_KIND = "action_ball_c211_diagnostic_claim_v2"
LINEAGE_KIND = "action_ball_c211_direct_ball_split_ready_lineage_v4"
C211_BUNDLE_KIND = "action_ball_c211_direct_ball_split_ready_bundle_v4"
# 2026-08-05 v1 -> v2:recipe 合同新增 exploration_axis / actor_init_mode /
# four_sigma_hard_inner_gate_applies 三键,schema 随之 1 -> 2,kind 同批改名,
# 老 v1 收据不能冒充新配方。
# 2026-08-05 v2 -> v3(第二轴换成本体感观测噪声开关):recipe 合同再新增
# observation_noise_axis / policy_observation_corruption /
# proprioceptive_observation_noise_channels / task_channel_observation_noise /
# dr_level_identity 五键,schema 2 -> 3。v2 收据不带 DR 档身份,放它冒充新配方等于
# 让"跑的是哪一档"无法自陈,所以 kind 同批改名。
RECIPE_KIND = "action_ball_c211_matched_recipe_v3"
MATERIALIZATION_KIND = "action_ball_c211_reward_materialization_v1"
POLICY_MATERIALIZATION_KIND = "action_ball_c211_policy_materialization_v1"
ORACLE32_KIND = "action_ball_c211_oracle32_receipt_v3"
C211_RAW_ORACLE_KIND = "action_ball_c211_oracle_raw_evidence_v3"
C211_RUNNER_PREFLIGHT_KIND = "action_ball_c211_runner_preflight_evidence_v1"
C211_SELECTED_RUBBER_KIND = (
    "action_ball_c211_selected_rubber_contact_evidence_v1"
)
RESULT_KIND = "action_ball_c211_diagnostic_launch_result_v1"
# 2026-08-07 v2 -> v3:收据新增 reward_activation_evidence_scope /
# terminal_transition_source / behavior_strict_hard_* 四键 —— 「这一跑属于哪种
# reward 证据体制、那三项严格零计数到底是从哪一族读来的」从此写在收据里。
# v2 收据不带这块自陈,放它冒充新收据等于让读者无法判断数字的出处,所以同批改名。
SCALE4096_TERMINAL_ACCEPTANCE_KIND = (
    "action_ball_c211_scale4096_terminal_acceptance_v3"
)
# Retired 2026-08-05: FRAME0_EXACT_ARTIFACT_KIND / FRAME0_EXACT_RECEIPT_KIND /
# FRAME0_EXACT_SOURCE_KIND deleted with _validate_retired_exact_frame0_lineage;
# the live copies stay in launch_action_ball_a211_four_arm_diagnostic.py.
FRAME0_LIVE_RECEIPT_KIND = "isaac_action_ball_nominal_hold_v1"
FRAME0_RECEIPT_PROBE_SOURCE_PATHS = _FRAME0.FRAME0_RECEIPT_PROBE_SOURCE_PATHS
# Retired 2026-08-05: PASSIVE_HOLD_SOAK_GATE_KIND deleted with _validate_passive_hold_soak.
PRELONG_SEMANTICS_ENABLE_ENV = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_SEMANTICS"
)


def _verify_frame0_probe_source_commit(
    checkout: Path, lineage_commit: str, probe_commit: str
) -> None:
    try:
        _FRAME0._verify_frame0_probe_source_commit(
            checkout, lineage_commit, probe_commit
        )
    except _FRAME0.LaunchRefused as exc:
        raise LaunchRefused(str(exc)) from exc


def _verify_commit_ancestor(
    checkout: Path, ancestor: str, descendant: str, *, name: str
) -> None:
    try:
        _FRAME0._verify_commit_ancestor(
            checkout, ancestor, descendant, name=name
        )
    except _FRAME0.LaunchRefused as exc:
        raise LaunchRefused(str(exc)) from exc


# Retired 2026-08-05: _validate_frame0_live_safety_evidence and
# _frame0_birth_gate_binding_sha256 shims deleted with
# _validate_retired_exact_frame0_lineage; the underlying authorities remain in
# _FRAME0 (launch_action_ball_a211_four_arm_diagnostic.py).
PRELONG_REWARD_RECIPE_SHA_ENV = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_REWARD_RECIPE_SHA256"
)
REWARD_PPO_ECONOMY_ENABLE_ENV = "HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_GATE"
UPDATE_PROFILE_ENV = "HOPE_ACTION_BALL_UPDATE_PROFILE"
UPDATE_PROFILE_JSON_PREFIX = "HOPE_ACTION_BALL_UPDATE_PROFILE_JSON="
# Retired 2026-08-05: _FRAME0_HANDOFF_KEYS deleted with
# _exact_zero_handoff_semantics; A211 keeps the authoritative copy.
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
EXPERIMENT_NAME = "agibot_a3_action_ball_c211_diagnostic"

ISAAC_FOUR_GRID_KIND = _F.KIND
A_OBS_NOISE_OFF_CELL_ID = _F.A_OBS_NOISE_OFF_CELL_ID
A_OBS_NOISE_ON_CELL_ID = _F.A_OBS_NOISE_ON_CELL_ID
C_OBS_NOISE_OFF_CELL_ID = _F.C_OBS_NOISE_OFF_CELL_ID
C_OBS_NOISE_ON_CELL_ID = _F.C_OBS_NOISE_ON_CELL_ID
ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS = _F.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
ACTOR_INIT_MODE_DEFAULT = _F.ACTOR_INIT_MODE_DEFAULT
DR_LEVEL_IDENTITY_OBS_NOISE_OFF = _F.DR_LEVEL_IDENTITY_OBS_NOISE_OFF
DR_LEVEL_IDENTITY_OBS_NOISE_ON = _F.DR_LEVEL_IDENTITY_OBS_NOISE_ON
ISAAC_FOUR_GRID_CELL_IDS = _F.CELL_IDS
RECIPE_IDS = _F.FAMILY_CELL_IDS["C211"]
ACTOR_CONTRACT = "action_ball_c211"
ACTOR_WIDTH = 211
CRITIC_CONTRACT = "action_ball_c211_critic_v1"
CRITIC_WIDTH = 319
TRAINABILITY_CONTRACT = "action_ball_c211_fixed_midpoint_learnability_v2"
ACTOR_NORMALIZER_IDENTITY = "action_ball_c211_actor_norm_v2"
CRITIC_NORMALIZER_IDENTITY = "action_ball_c211_critic_norm_v1"
TASK_PROFILE_ID = "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability"
GYM_TASK_ID = "HOPE-PingPong-ActionBall-C211Learnability-AgibotA3-v0"
TARGET_SEMANTICS = "c211_incoming_ball_p_v_spin_outcome_dense_v1"
# [2026-08-06 载体错配核查结论: 对运行时无害, 只影响被记录的标定数字]
#
# `TARGET_RECIPE` 在运行时唯一的含义是选中全关的目标有效位:
# `hope_commands._action_ball_target_recipe_contract` 只用它查
# `_ACTION_BALL_TARGET_VALIDITY_BY_RECIPE`, 并强制 direct_ball 必须配
# outcome_dense_only + (False, False, False)。它不是收据选择器 —— 全仓没有任何一处
# 用 TARGET_RECIPE 去挑 `initial_center_task_receipt` 该是哪一份文件。
#
# 离线 producer 里 outcome_dense_only 的
# `algorithm_id = coherent_current_lm_carrier_mask_all_targets`, 即"借 current_lm 的
# 载体、把所有目标位掩掉"。实测这两份收据的字节完全相同(同一 sha256), recipe 之间的
# 区别只落在 tape 的 `target_producer_sha256` 上, 不落在收据内容上。
#
# C 的运行时在 direct_ball 分支(hope_commands.py 约 9401 行)用的是另一个载体:
# `_action_ball_reference_velocity_host_rows` / `_action_ball_reference_raw_normal_host_rows`,
# 也就是教师 FK 参考行, 不调任何 LM/解析逆解(`reset_inverse_solve=False`,
# `online_lm_calls=0`)。同一份题在两个载体下的教师速率: 教师 FK 载体≈1.0
# (producer 的 teacher_pos_face_no_velocity 实测 0.9999999716), current_lm 载体
# 0.8513505673, 差约 15%。
#
# 判定为无害的理由: 这份被钉的 `initial_center_task_receipt` 的角色是
# `calibration_receipt_not_runtime_question_source` —— 它不是运行时题源, 不进 argv,
# 运行时一个字段都不读它; C 的题由 `runtime_curriculum_sampler` 现场采样, 收据由运行时
# 自己吐、并被运行时自己的 derive_action_teacher_site_timing 逐字段复核。载体错配的
# 全部后果, 是发射凭据里记下的 `time_to_teacher_start_at_reveal_s` 取的是 current_lm
# 载体的 0.6924 s 而不是教师载体的 0.8600 s; 隐藏 WAIT 的真实长度由
# `WAIT_SCHEDULE["max_wait_ticks"]` 定, 与这个数字无关。
#
# 按 Franco 2026-08-06 裁定, 速率差交给 policy 泛化, 不改 TARGET_RECIPE、不新造原速配方。
TARGET_RECIPE = "outcome_dense_only"
TARGET_VALIDITY_MASK = (False, False, False)
TARGET_SOURCE = "direct_ball"
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
C211_OBSERVED_BUNDLE_FILENAME = "c211_observed_oracle_bundle.json"
RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64
POLICY_DT_S = 0.02
PASSIVE_HOLD_SOAK_POLICY_STEPS = 200
PASSIVE_HOLD_SOAK_PHYSICS_STEPS = 800
WAIT_SCHEDULE = _W.ActionBallTaskWaitSchedule(
    seed=20260804,
    min_wait_ticks=5,
    max_wait_ticks=25,
    episode_horizon_ticks=500,
    required_active_ticks=200,
).to_dict()
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
ORACLE_SINGLE_STROKE_REASON = "action_ball_single_stroke_complete"
# oracle32 里一集只可能停在这两个阶段之一(WAIT-only 复位根本不进这份证据)。
ORACLE_TERMINAL_PHASES = ("post_strike", "pre_strike_or_same_step_unknown")
# 人话:oracle32 跑的是**刚初始化、一步没训过**的 policy。它被允许的死法只有两类 ——
# 打完一整拍(single stroke),或者踩到五项硬安全终止之一。名单外的任何终止名
# (比如 hold 期禁用的 anchor_pos,或者某人新加的一项)一律拒收。
ORACLE_ALLOWED_TERMINATION_REASONS = (
    ORACLE_SINGLE_STROKE_REASON,
    *HARD_TERMINATION_UNION,
)
# 人话:这三项是"机器人没学会站着/挥拍",属于行为证据,按阶段计数上报,不拒收。
# 另外两项(STRICT_HARD_TERMINATION_UNION)是"实现坏了",必须严格零。
# §12.4 与 §8.3 都是这么写的,这里只是把门指到同一个对象上。
ORACLE_BEHAVIOR_TERMINATION_REASONS = (*PHYSICAL_FALL_REASONS, "robot_hit_table")
assert set(ORACLE_BEHAVIOR_TERMINATION_REASONS) | set(
    STRICT_HARD_TERMINATION_UNION
) == set(HARD_TERMINATION_UNION)
assert not set(ORACLE_BEHAVIOR_TERMINATION_REASONS) & set(
    STRICT_HARD_TERMINATION_UNION
)
PROHIBITED_HOLD_REFERENCE_TERMINATIONS = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)
# Retired 2026-08-05: FULL_ACTIVE_TERMINATIONS deleted with
# _validate_passive_hold_soak; A211 keeps the live copy.

REQUIRED_OUTCOME_TERMS: Mapping[str, Mapping[str, Any]] = {
    "c225_strike_ball_paddle_center_proximity": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.action_ball_c225_rewards."
            "c225_strike_ball_paddle_center_proximity"
        ),
        "weight": 240.0,
        "params": {"command_name": "racket_target", "std": 0.15},
    },
    "virtual_landing": {
        "callable": (
            "whole_body_tracking.tasks.tracking.mdp.action_ball_c225_rewards."
            "c225_landing_outcome_actual_contact"
        ),
        "weight": 700.0,
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
PROHIBITED_TASK_DIRECTED_TERMS = (
    "base_position",
    "racket_progress",
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
    "strike_capture_bonus",
    "virtual_pass_net",
    "virtual_landing_dense",
    "virtual_spin",
)
ALLOWED_TASK_DIRECTED_TERMS = frozenset(REQUIRED_OUTCOME_TERMS)
TASK_DIRECTED_PREFIXES = ("desired_", "racket_", "strike_", "virtual_", "c211_", "c225_")
assert not ALLOWED_TASK_DIRECTED_TERMS.intersection(PROHIBITED_TASK_DIRECTED_TERMS)

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
C211_EVIDENCE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_c211_oracle_evidence.py"
)
C211_LIVE_ORACLE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/action_ball_c211_live_oracle.py"
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
    "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability.yaml"
)
RETAINED_TASK_PROFILE_PARENT_SOURCE = (
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

RUNTIME_SOURCE_PATHS = (
    (LAUNCHER_SOURCE, "VendorV2 N1 launcher"),
    (FOUR_GRID_SOURCE, "Isaac A211/C211 four-grid authority"),
    (FOUR_GRID_BARRIER_SOURCE, "Isaac A211/C211 all-four pre-long barrier"),
    (PRELONG_GATE_SOURCE, "shared ActionBall 4096x5 pre-long terminal gate"),
    (PRELONG_SEMANTICS_SOURCE, "ActionBall 4096x5 semantic marker schema"),
    (ACTION_BALL_SAMPLING_SOURCE, "ActionBall curriculum question sampler"),
    (ACTION_BALL_COMMAND_SOURCE, "ActionBall runtime command integration"),
    (ADMISSION_SOURCE, "VendorV2 GPU admission"),
    (EXACT_GROUP_SOURCE, "exact process-group helper"),
    (BASE_SOURCE, "no-clobber base helper"),
    (KIT_LAUNCHER_SOURCE, "locked Kit launcher"),
    (TRAIN_SOURCE, "training entrypoint"),
    (OLD_VALIDATOR_SOURCE, "oracle32 acceptance validator"),
    (C211_EVIDENCE_SOURCE, "C211 observed-oracle sidecar publisher"),
    (C211_LIVE_ORACLE_SOURCE, "C211 live runtime oracle adapter"),
    (TRAINING_CONTRACT_SOURCE, "dynamic-ready policy contract"),
    (TASK_WAIT_SOURCE, "pre-task wait schedule contract"),
    (TASK_PROFILE_SOURCE, "C211 DR-L0 task profile"),
    (RETAINED_TASK_PROFILE_PARENT_SOURCE, "C211 inherited task-profile parent"),
    (_FRAME0.DR_L0_MANIFEST_SOURCE, "ActionBall DR-L0 launch manifest"),
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
FORMAL_GRID_STAGE_ORDER = _F.FORMAL_STAGE_ORDER
BLOCKED_RUNTIME_STAGES: tuple[str, ...] = ()
ORACLE_RUNTIME_BLOCKER = "C211_ORACLE_NOT_IMPLEMENTED"
ORACLE_RUNTIME_DEPENDENCIES: tuple[str, ...] = ()
C211_ORACLE_HOOK_SOURCE_MARKERS = {
    TRAIN_SOURCE: (
        b"action_ball_c211_oracle_bundle_output_path",
        b"_make_c211_live_runtime_step_adapter",
        b"collect_live_oracle_bundle(",
        b"racket._action_ball_ledger_payload()",
        b"racket.vb_fired[0]",
        b"env.step(actions)",
        b"ACTION_BALL_C211_OBSERVED_ORACLE_BUNDLE_JSON",
    ),
}
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
        raise LaunchRefused("C211 four-grid authority differs: %s" % exc) from exc


def _four_grid_cell(cell_id: str, *, task_family: str) -> dict[str, Any]:
    if task_family != "C211":
        raise LaunchRefused("C211 launcher cannot select another task family")
    _isaac_four_grid_manifest()
    try:
        return _F.cell_for_family(cell_id, "C211")
    except _F.FourGridContractError as exc:
        raise LaunchRefused("selector is not a formal C211 grid cell") from exc


def _exact_dict(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    return _B._exact_dict(value, tuple(keys), name=name)


# Retired 2026-08-05: _finite_handoff_vector deleted with
# _exact_zero_handoff_semantics (its only caller).


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


# Retired 2026-08-05: _exact_zero_handoff_semantics deleted with its only caller
# _validate_retired_exact_frame0_lineage; A211 keeps the live copy.


def _prelong_semantics_exec_environment(
    stage: str, reward_materialization: Any
) -> dict[str, str]:
    """Emit the prelong switches, and only for the stage that owns them.

    人话:这三条环境变量只有 scale4096 用得上,而 ``runtime_effective_reward_sha256``
    要等 reward 真的产出之后才存在。取值必须发生在 stage 判断**之后** —— 原来在调用
    点直接下标取,materialize / recipe / oracle32 三个阶段根本还没有这个键,于是 exec
    前当场 KeyError,整条流水线的第一站就过不去。scale4096 上缺这个键仍然是硬错。
    """

    if stage != "scale4096":
        return {}
    try:
        runtime_reward_sha256 = reward_materialization[
            "runtime_effective_reward_sha256"
        ]
    except (KeyError, TypeError) as exc:
        raise LaunchRefused(
            "C211 scale4096 requires the materialized runtime reward recipe SHA"
        ) from exc
    reward_sha = _B._sha256(
        runtime_reward_sha256, name="C211 prelong runtime Reward recipe SHA"
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
        "workload_kind": "direct_ball_sampler_consumer",
        "online_solver_calls_required": 0,
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


def _four_grid_prelong_receipt_pin(value: Any) -> dict[str, str]:
    try:
        pin, _path = _Q._pin(value, name="four-grid scale4096 receipt")
    except _Q.BarrierRefused as exc:
        raise LaunchRefused("C211 four-grid pre-long receipt pin differs") from exc
    return pin


def _validate_four_grid_prelong_receipt(
    value: Any, *, checkout: Path
) -> dict[str, Any]:
    try:
        return _Q.validate_receipt(value, checkout=checkout)
    except _Q.BarrierRefused as exc:
        raise LaunchRefused("C211 four-grid pre-long barrier refused: %s" % exc) from exc


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
    """Exact C211 sampler/provider scope emitted by the runtime contract."""

    return {
        "identity": "action_ball_211_question_source_scope_v5",
        "family": "C211",
        "question_sampler": {
            "source": "runtime_curriculum_sampler",
            "cadence": "every_episode_reset",
            "curriculum_domain_levels_consulted_every_reset": True,
            "sampler_runs_every_reset": True,
            "initial_center_single_question": True,
            "initial_center_activation": "all_32_domain_levels_exact_zero",
            "initial_center_physical_support": "literal_profile_center_point",
            "initial_center_rng_draws": "normal_fixed_budget",
            "post_promotion_support": "zero_to_manifest_max_width_per_promoted_arm",
            "sampler_rng_reused_by_target_provider": False,
            "physical_rng_draw_count_authority": (
                "sample_receipt_draw_end_minus_draw_start"
            ),
            "zero_physical_rng_draw_claim_permitted": False,
            "selection": "sample_current_domain_levels",
            "checkpoint_resume": "exact_sampler_and_curriculum_state",
        },
        "target_provider": {
            "source": "direct_ball",
            "desired_contact_inverse": False,
            "exact_question_answer_cache": {"enabled": False},
            "online_inverse_solves_per_reset": 0,
            "online_inverse_solves_per_step": 0,
        },
    }


def _question_rng_contract() -> dict[str, Any]:
    return {
        "owner": "runtime_curriculum_sampler",
        "cadence": "every_episode_reset",
        "draw_count_authority": "sample_receipt_draw_end_minus_draw_start",
        "zero_draw_claim_permitted": False,
        "checkpoint_resume": "exact_sampler_and_curriculum_state",
    }


def _curriculum_scope_contract() -> dict[str, Any]:
    return {
        "question_source": "runtime_curriculum_sampler",
        "domain_levels_authority": "runtime_action_ball_curriculum",
        "reset_question_selection": "sample_current_domain_levels",
        "question_rng": _question_rng_contract(),
        "desired_contact_inverse": False,
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
        "target_source": TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "target_observation_noise": False,
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "desired_contact_fields_observed": False,
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "question_rng": _question_rng_contract(),
    }


def _c211_reward_contract() -> dict[str, Any]:
    return {
        "identity": "action_ball_c211_achieved_outcome_reward_v3",
        "desired_contact_position_velocity_face_consumed": False,
        "task_valid_required": True,
        "strike_bridge": {
            "term": "c225_strike_ball_paddle_center_proximity",
            "callable": REQUIRED_OUTCOME_TERMS[
                "c225_strike_ball_paddle_center_proximity"
            ]["callable"],
            "weight": 240.0,
            "std_m": 0.15,
            "kernel": "cauchy_inverse_quadratic",
            "eligibility": "task_valid_active_swing_single_exact_strike_tick",
            "miss_retains_gradient": True,
        },
        "economics": {
            "policy_dt_s": 0.02,
            "task_valid_swing_mimic_undiscounted_cap": 2.8325,
            "task_reveal_discounted_gamma": 0.99,
            "task_reveal_contact_tick": 92,
            "task_valid_swing_mimic_discounted_cap": 1.7733077595610476,
            "strike_bridge_post_dt_peak": 4.8,
            "strike_bridge_discounted_at_contact": 1.9040534708257204,
            "legal_landing_post_dt_min": 8.4,
            "legal_landing_discounted_at_contact_min": 3.332093573945011,
            "ordering": (
                "task_valid_swing_mimic_lt_strike_peak_lt_legal_landing"
            ),
        },
        "landing": {
            "term": "virtual_landing",
            "callable": REQUIRED_OUTCOME_TERMS["virtual_landing"]["callable"],
            "weight": 700.0,
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


def _recipe_contract(recipe_id: str) -> dict[str, Any]:
    if recipe_id not in RECIPE_IDS:
        raise LaunchRefused("recipe_id must select one of the two formal C211 grid cells")
    manifest = _isaac_four_grid_manifest()
    matched = manifest["matched_contract"]
    cell = _four_grid_cell(recipe_id, task_family="C211")
    # 观测噪声开关是本轮唯一的注册差异轴(exp §5.6.2d),那几个键只能从本格 cell 取。
    # 探索包反过来:它已经全格相同,只能从 matched_contract 取 —— cell 上刻意没有同名
    # 键,旧写法会 KeyError 而不是读到一个"看起来是本格的"数字。
    exploration = matched["exploration_package"]
    if cell["ppo"] != matched["ppo"]:
        raise LaunchRefused("C211 grid cell PPO differs from the matched contract")
    if matched["exploration_axis_is_registered_difference"] is not False:
        raise LaunchRefused(
            "C211 grid still advertises the exploration package as a difference axis"
        )
    try:
        _F.validate_observation_noise_package(cell)
        _F.validate_exploration_package(exploration)
    except _F.FourGridContractError as exc:
        raise LaunchRefused("C211 four-grid cell package differs: %s" % exc) from exc
    unsigned = {
        "schema_version": 3,
        "kind": RECIPE_KIND,
        "recipe_id": recipe_id,
        "four_grid_cell_id": recipe_id,
        "isaac_four_grid_manifest_sha256": manifest["content_sha256"],
        "ppo_adaptation_axis": cell["ppo_adaptation_axis"],
        "contact_sigma_adaptation": False,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "fresh_normalizers_required": True,
        "foreign_checkpoint_reuse_prohibited": True,
        # 全格相同的探索包(标准 rsl_rl 初始化 + sigma 1.0 + scalar,4σ 门显式跳过)。
        "exploration_axis": exploration["exploration_axis"],
        "actor_init_mode": exploration["actor_init_mode"],
        "four_sigma_hard_inner_gate_applies": exploration[
            "four_sigma_hard_inner_gate_applies"
        ],
        "init_noise_std": exploration["init_noise_std"],
        "noise_std_type": exploration["noise_std_type"],
        # 本轮唯一的注册差异轴:本体感观测噪声开关及其 DR 档身份。
        "observation_noise_axis": cell["observation_noise_axis"],
        "policy_observation_corruption": cell["policy_observation_corruption"],
        "proprioceptive_observation_noise_channels": cell[
            "proprioceptive_observation_noise_channels"
        ],
        "task_channel_observation_noise": cell["task_channel_observation_noise"],
        "dr_level_identity": cell["dr_level_identity"],
        "entropy_coef": matched["entropy_coef"],
        "actor_hidden_dims": matched["actor_hidden_dims"],
        "critic_hidden_dims": matched["critic_hidden_dims"],
        "reference_guard_mode": matched["reference_guard_mode"],
        "soft_weights": matched["soft_weights"],
        "ppo": cell["ppo"],
    }
    return {**unsigned, "recipe_contract_sha256": canonical_sha256(unsigned)}


def _verify_c211_runtime_authorities(checkout: Path) -> None:
    """Require the live runner-before-oracle evidence hook."""

    missing = []
    for authority, sources in (
        ("runner-before-oracle", C211_ORACLE_HOOK_SOURCE_MARKERS),
    ):
        for relative, markers in sources.items():
            path = checkout / relative
            try:
                payload = path.read_bytes()
            except OSError:
                missing.append("%s:%s:missing" % (authority, relative))
                continue
            missing.extend(
                "%s:%s:%s"
                % (authority, relative, marker.decode("ascii"))
                for marker in markers
                if marker not in payload
            )
    if missing:
        raise LaunchRefused(
            "C211 live-oracle runtime authority is absent: " + ", ".join(missing)
        )


# Retired 2026-08-05: _validate_passive_hold_soak and
# _validate_retired_exact_frame0_lineage deleted -- both had zero call sites after
# the C211 lineage moved to the direct-ball split-ready contract validated by
# _validate_lineage below.


def _validate_lineage(
    checkout: Path, commit: str, value: Any
) -> dict[str, Any]:
    """Validate direct-ball C211 with separate physical and teacher births."""

    pin, row = _tracked_json(checkout, commit, value, name="C211 lineage")
    _verify_c211_runtime_authorities(checkout)
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
            "actor_normalizer_identity",
            "critic_normalizer_identity",
            "task_profile",
            "gym_task",
            "target_semantics",
            "curriculum_scope",
            "target_source",
            "question_source",
            "question_rng",
            "target_recipe",
            "target_validity_mask",
            "incoming_ball_fields",
            "reset_inverse_solve",
            "online_solver_calls",
            "online_lm_calls",
            "action_id",
            "action_uid",
            "teacher_id",
            "seed",
            "bundle",
            "motion",
            "action_manifest",
            "initial_center_task_receipt",
            "dynamic_ready_artifact",
            "dynamic_ready_nominal_receipt",
            "teacher_frame0_artifact",
            "dr_l0_manifest",
        ),
        name="C211 lineage",
    )
    _assert_c211_only(row, name="C211 lineage")
    expected = {
        "schema_version": 4,
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
        "target_source": TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng": _question_rng_contract(),
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "action_id": ACTION_ID,
        "action_uid": ACTION_UID,
        "teacher_id": TEACHER_ID,
        "seed": 0,
    }
    if any(row[key] != wanted for key, wanted in expected.items()):
        raise LaunchRefused("C211 split-ready lineage identity differs")

    pins: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for key in (
        "bundle",
        "motion",
        "action_manifest",
        "initial_center_task_receipt",
        "dynamic_ready_artifact",
        "dynamic_ready_nominal_receipt",
        "teacher_frame0_artifact",
    ):
        normalized, resolved = _B._verify_tracked_file(
            checkout,
            commit,
            _pin(row[key], name="lineage.%s" % key),
            name="C211 %s" % key,
        )
        pins[key] = normalized
        paths[key] = resolved
    if (
        pins["dynamic_ready_artifact"]["sha256"]
        != _FRAME0.SPLIT_READY_DYNAMIC_ARTIFACT_SHA256
        or pins["dynamic_ready_nominal_receipt"]["sha256"]
        != _FRAME0.SPLIT_READY_NOMINAL_HOLD_SHA256
        or pins["teacher_frame0_artifact"]["sha256"]
        != _FRAME0.SPLIT_READY_TEACHER_FRAME0_ARTIFACT_SHA256
    ):
        raise LaunchRefused("C211 split-ready authority bytes differ")

    documents: dict[str, dict[str, Any]] = {}
    for key in (
        "bundle",
        "action_manifest",
        "initial_center_task_receipt",
        "dynamic_ready_artifact",
        "dynamic_ready_nominal_receipt",
        "teacher_frame0_artifact",
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
            "target_source",
            "question_source",
            "question_rng",
            "target_recipe",
            "curriculum_scope",
            "target_validity_mask",
            "incoming_ball_fields",
            "reset_inverse_solve",
            "online_solver_calls",
            "online_lm_calls",
            "motion",
            "action_manifest",
            "initial_center_task_receipt",
            "dynamic_ready_artifact",
            "dynamic_ready_nominal_receipt",
            "teacher_frame0_artifact",
            "dr_l0_manifest",
        ),
        name="C211 split-ready bundle",
    )
    expected_bundle = {
        "schema_version": 4,
        "kind": C211_BUNDLE_KIND,
        "diagnostic_unauthorized": True,
        "action_id": ACTION_ID,
        "action_uid": ACTION_UID,
        "teacher_id": TEACHER_ID,
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "actor_normalizer_identity": ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": CRITIC_NORMALIZER_IDENTITY,
        "target_source": TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng": _question_rng_contract(),
        "target_recipe": TARGET_RECIPE,
        "curriculum_scope": _curriculum_scope_contract(),
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "motion": pins["motion"],
        "action_manifest": pins["action_manifest"],
        "initial_center_task_receipt": pins["initial_center_task_receipt"],
        "dynamic_ready_artifact": pins["dynamic_ready_artifact"],
        "dynamic_ready_nominal_receipt": pins[
            "dynamic_ready_nominal_receipt"
        ],
        "teacher_frame0_artifact": pins["teacher_frame0_artifact"],
    }
    try:
        dr_l0_manifest = _FRAME0._dr_l0_manifest_binding(
            checkout,
            commit,
            family="C",
            task_profile=TASK_PROFILE_ID,
        )
    except _FRAME0.LaunchRefused as exc:
        raise LaunchRefused(str(exc)) from exc
    expected_bundle["dr_l0_manifest"] = dr_l0_manifest
    if any(bundle[key] != wanted for key, wanted in expected_bundle.items()):
        raise LaunchRefused("C211 split-ready bundle closure differs")

    manifest = documents["action_manifest"]
    actions = manifest.get("actions")
    action = actions[0] if type(actions) is list and len(actions) == 1 else None
    if (
        manifest.get("schema_version") != 3
        or manifest.get("action_order") != [ACTION_ID]
        or manifest.get("mobility_mode") != "no_move"
        or type(action) is not dict
        or action.get("action_id") != ACTION_ID
        or action.get("action_uid") != ACTION_UID
        or action.get("motion_path") != pins["motion"]["path"]
        or action.get("motion_sha256") != pins["motion"]["sha256"]
    ):
        raise LaunchRefused("C211 action manifest closure differs")

    try:
        teacher = _FRAME0._validate_teacher_frame0_artifact(
            documents["teacher_frame0_artifact"],
            motion_path=paths["motion"],
            motion_sha256=pins["motion"]["sha256"],
        )
        timing = _FRAME0._initial_center_timing_authority(
            receipt=documents["initial_center_task_receipt"],
            receipt_pin=pins["initial_center_task_receipt"],
            action_manifest=documents["action_manifest"],
            action_manifest_pin=pins["action_manifest"],
            motion_sha256=pins["motion"]["sha256"],
            family="C",
        )
        reset_wait = _FRAME0._split_ready_reset_wait_semantics(
            dynamic=documents["dynamic_ready_artifact"],
            nominal=documents["dynamic_ready_nominal_receipt"],
            dynamic_pin=pins["dynamic_ready_artifact"],
            nominal_pin=pins["dynamic_ready_nominal_receipt"],
            teacher_frame0=teacher["frame0"],
            motion_sha256=pins["motion"]["sha256"],
            initial_center_timing_authority=timing,
        )
    except _FRAME0.LaunchRefused as exc:
        raise LaunchRefused(str(exc)) from exc
    if row["dr_l0_manifest"] != dr_l0_manifest:
        raise LaunchRefused("C211 DR-L0 lineage binding differs")
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


def _planned_materialization(
    *, recipe: Mapping[str, Any], lineage: Mapping[str, Any]
) -> dict[str, Any]:
    reward = {
        "question_contract": _question_contract(),
        "soft_weights": recipe["soft_weights"],
        "reference_guard_mode": recipe["reference_guard_mode"],
        "c211_reward_contract": _c211_reward_contract(),
        "required_positive_outcome_terms": list(REQUIRED_OUTCOME_TERMS),
        "allowed_task_directed_terms": sorted(ALLOWED_TASK_DIRECTED_TERMS),
        "prohibited_task_directed_terms": list(PROHIBITED_TASK_DIRECTED_TERMS),
        "unregistered_task_directed_terms_prohibited": True,
    }
    unsigned = {
        "schema_version": 1,
        "kind": MATERIALIZATION_KIND,
        "diagnostic_unauthorized": True,
        "recipe_id": recipe["recipe_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "recipe_contract_sha256": recipe["recipe_contract_sha256"],
        "reward_contract_sha256": canonical_sha256(reward),
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "dr_l0_manifest": lineage["dr_l0_manifest"],
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
    base_keys = (
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
    )
    has_terminal_acceptance = (
        type(row) is dict and "terminal_acceptance" in row
    )
    row = _sealed_row(
        row,
        base_keys + (("terminal_acceptance",) if has_terminal_acceptance else ()),
        name=name,
    )
    if (
        row["schema_version"] != 2
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
    if expected_stage == "scale4096":
        if not has_terminal_acceptance or type(row["terminal_acceptance"]) is not dict:
            raise LaunchRefused(
                "C211 scale4096 result lacks terminal checkpoint/safety acceptance"
            )
    elif has_terminal_acceptance:
        raise LaunchRefused(
            "%s contains a scale4096-only terminal acceptance" % name
        )
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
            "dr_l0_manifest",
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
        "dr_l0_manifest",
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
        "recipe_id": recipe["recipe_id"],
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


def _finite_vec2(value: Any, *, name: str) -> list[float | int]:
    if type(value) is not list or len(value) != 2:
        raise LaunchRefused("%s must be a length-2 list" % name)
    for component in value:
        if type(component) not in (int, float) or not math.isfinite(float(component)):
            raise LaunchRefused("%s must contain finite numbers" % name)
    return list(value)


def _validate_c211_hard_contract(
    value: Any,
    *,
    checkout: Path,
    oracle_namespace: Path,
    lineage: Mapping[str, Any],
    recipe: Mapping[str, Any],
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
    try:
        resolved_dr_l0 = module.action_ball_dr_l0_contract_payload()
        resolved_dr_l0_sha256 = module.action_ball_dr_l0_contract_sha256()
        resolved_dr_l0n = module.action_ball_dr_l0n_contract_payload()
    except Exception as exc:
        raise LaunchRefused(
            "C211 hard training contract cannot resolve DR-L0 finalizer"
        ) from exc
    # 谱系绑的是**共用的 DR-L0 leaf** 与它解析出来的字节 —— 这一条对四格都成立。
    # 真正跑的那一档由本格 cell 决定并写进 recipe 合同:噪声关 = L0,噪声开 = L0N。
    # 这里按本格身份挑一份 payload 去对拍硬合同,并要求另一档的键**不存在**。
    corruption = recipe["policy_observation_corruption"]
    if recipe["dr_level_identity"] != (
        DR_LEVEL_IDENTITY_OBS_NOISE_ON
        if corruption
        else DR_LEVEL_IDENTITY_OBS_NOISE_OFF
    ):
        raise LaunchRefused("C211 recipe DR level identity differs from its cell")
    expected_dr_key = "action_ball_dr_l0n" if corruption else "action_ball_dr_l0"
    forbidden_dr_key = "action_ball_dr_l0" if corruption else "action_ball_dr_l0n"
    expected_dr_payload = resolved_dr_l0n if corruption else resolved_dr_l0
    if (
        contract.get(expected_dr_key) != expected_dr_payload
        or forbidden_dr_key in contract
        or expected_dr_payload.get("identity") != recipe["dr_level_identity"]
        or expected_dr_payload.get("policy_observation_corruption") is not corruption
        or canonical_sha256(resolved_dr_l0) != resolved_dr_l0_sha256
        or lineage.get("dr_l0_manifest", {}).get("contract_sha256")
        != resolved_dr_l0_sha256
        or lineage.get("dr_l0_manifest", {}).get("hard_contract_identity")
        != resolved_dr_l0.get("identity")
    ):
        raise LaunchRefused("C211 hard training contract DR binding differs")
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
        # retired.  Its value identity is nevertheless the C211 reward-v3
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
        or actor_names[10:13] != list(INCOMING_BALL_FIELDS)
        or actor_dims[10:13] != [3, 3, 3]
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
    except (KeyError, TypeError) as exc:
        raise LaunchRefused("C211 hard training contract target provider is missing") from exc
    exact_reuse = (
        target.get("exact_question_answer_reuse")
        if type(target) is dict
        else None
    )
    if (
        type(target) is not dict
        or target.get("source") != TARGET_SOURCE
        or target.get("recipe") != TARGET_RECIPE
        or target.get("validity_mask") != list(TARGET_VALIDITY_MASK)
        or target.get("target_observation_noise") is not False
        or target.get("actor_width_unchanged") is not True
        or target.get("critic_width_unchanged") is not True
        or target.get("immutable_tape") is not None
        or type(exact_reuse) is not dict
        or exact_reuse.get("enabled") is not False
    ):
        raise LaunchRefused("C211 hard training contract direct-ball/000 differs")
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
            "task_valid_required",
            "task_wait_contract",
            "question_source_contract",
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
        "task_valid_required": True,
        "task_wait_contract": _hard_wait_contract(),
        "question_source_contract": _hard_question_source_contract(),
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
    if sum(buckets.values()) != 32:
        raise LaunchRefused("C211 selected-rubber denominator does not close")
    return {
        "artifact": pin,
        "content_sha256": row["content_sha256"],
        "row_sha256": row_sha256,
        "eligible_episode_denominator": len(episodes),
        "actual_selected_rubber_contact_count": buckets["selected_rubber"],
        "classifications": [item["classification"] for item in episodes],
    }


def _validate_c211_analytic_pair(flight_value: Any, prediction_value: Any, *, selected: bool):
    flight = _exact_dict(
        flight_value,
        (
            "evaluated", "finite", "landing_xy_m", "landing_valid",
            "net_crossed", "net_clear", "on_opponent_table", "source",
        ),
        name="C211 achieved analytic flight",
    )
    prediction = _exact_dict(
        prediction_value,
        (
            "evaluated", "predicted_net_clear", "predicted_legal_landing",
            "predicted_landing_xy_m", "source",
        ),
        name="C211 predicted analytic outcome",
    )
    if flight["evaluated"] is not selected or prediction["evaluated"] is not selected:
        raise LaunchRefused("C211 analytic evaluation differs from achieved contact")
    if selected:
        for key in (
            "finite", "landing_valid", "net_crossed", "net_clear",
            "on_opponent_table",
        ):
            if type(flight[key]) is not bool:
                raise LaunchRefused("C211 analytic flight boolean differs")
        if (
            flight["finite"] is not True
            or flight["source"]
            != "runtime_vb_one_shot_from_achieved_selected_rubber_contact"
            or prediction["source"]
            != "runtime_c225_achieved_flight_prediction_one_shot"
        ):
            raise LaunchRefused("C211 analytic authority differs")
        _finite_vec2(flight["landing_xy_m"], name="C211 achieved landing")
        _finite_vec2(
            prediction["predicted_landing_xy_m"], name="C211 predicted landing"
        )
        expected_legal = bool(
            flight["landing_valid"] and flight["net_crossed"]
            and flight["net_clear"] and flight["on_opponent_table"]
        )
        if (
            prediction["predicted_landing_xy_m"] != flight["landing_xy_m"]
            or prediction["predicted_net_clear"] is not flight["net_clear"]
            or prediction["predicted_legal_landing"] is not expected_legal
            or (flight["net_clear"] and not flight["net_crossed"])
            or (flight["on_opponent_table"] and not flight["landing_valid"])
        ):
            raise LaunchRefused("C211 predicted outcome differs from achieved flight")
    elif (
        flight != {
            "evaluated": False, "finite": False, "landing_xy_m": None,
            "landing_valid": False, "net_crossed": False, "net_clear": False,
            "on_opponent_table": False, "source": None,
        }
        or prediction != {
            "evaluated": False, "predicted_net_clear": None,
            "predicted_legal_landing": None,
            "predicted_landing_xy_m": None, "source": None,
        }
    ):
        raise LaunchRefused("C211 miss carries hypothetical analytic outcome")
    return flight, prediction


def _oracle_count(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise LaunchRefused(
            "C211 raw oracle %s must be a nonnegative integer" % name
        )
    return value


def _oracle_termination_reasons(value: Any, *, name: str) -> tuple[str, ...]:
    """One episode's terminal reasons, restricted to the declared vocabulary.

    人话:这一集为什么结束。允许"打完一拍"和五项硬安全终止;出现任何别的名字一律
    拒收 —— 重定范围只放行**已知的**行为证据,不放行"没人认识的死法"。
    """

    if (
        type(value) is not list
        or not value
        or any(type(reason) is not str for reason in value)
        or len(set(value)) != len(value)
        or any(
            reason not in ORACLE_ALLOWED_TERMINATION_REASONS for reason in value
        )
    ):
        raise LaunchRefused("%s is not one exact known terminal-reason set" % name)
    return tuple(value)


def _oracle_termination_census(
    rows: Sequence[tuple[str, tuple[str, ...]]],
) -> dict[str, Any]:
    """Recount phase x reason from the per-episode rows and prove it conserves.

    人话:不信收据自己写的总数,拿 32 集逐集重数一遍。之后每一个聚合数字(termination
    账、safety 账、completion 账)都必须和这份重数对上,谁也不能自己报一个好看的总数。
    """

    by_reason: dict[str, int] = {}
    phase_by_reason: dict[str, dict[str, int]] = {
        phase: {} for phase in ORACLE_TERMINAL_PHASES
    }
    episodes_by_phase = {phase: 0 for phase in ORACLE_TERMINAL_PHASES}
    for phase, reasons in rows:
        episodes_by_phase[phase] += 1
        for reason in reasons:
            by_reason[reason] = by_reason.get(reason, 0) + 1
            phase_by_reason[phase][reason] = (
                phase_by_reason[phase].get(reason, 0) + 1
            )
    if sum(episodes_by_phase.values()) != len(rows):
        raise LaunchRefused(
            "C211 raw oracle phase census does not cover every episode"
        )
    for reason, total in by_reason.items():
        if sum(
            table.get(reason, 0) for table in phase_by_reason.values()
        ) != total:
            raise LaunchRefused(
                "C211 raw oracle reason-by-phase census does not conserve for %s"
                % reason
            )
    return {
        "episodes": len(rows),
        "episodes_by_phase": episodes_by_phase,
        "by_reason": by_reason,
        "phase_by_reason": phase_by_reason,
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
    observed_bundle: Mapping[str, Any],
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
            "observed_oracle_bundle_content_sha256",
            "training_contract_artifact",
            "runner_preflight_artifact",
            "question_contract",
            "completion",
            "episodes",
            "desired_contact_metrics",
            "rollout_census",
            "termination",
            "safety",
            "selected_rubber_contact_artifact",
            "teacher_qdes",
        ),
        name="C211 raw oracle evidence",
    )
    if (
        row["schema_version"] != 3
        or row["kind"] != C211_RAW_ORACLE_KIND
        or row["diagnostic_unauthorized"] is not True
    ):
        raise LaunchRefused("C211 raw oracle evidence schema differs")
    if row["observed_oracle_bundle_content_sha256"] != canonical_sha256(observed_bundle):
        raise LaunchRefused("C211 raw oracle is not bound to the observed live bundle")
    # 人话:32 集"已关闭的挥拍"不是这一跑的全部。WAIT 期就死掉的复位既不算尝试、也
    # 不进这份证据 —— 但它必须被数出来,否则"32 集里只有 3 集摔倒"可能是从 300 次
    # WAIT 猝死里挑出来的。分母要看得见(§8.3:按 wait-start/reveal 作分母并守恒)。
    rollout = _exact_dict(
        row["rollout_census"],
        (
            "source_episodes_consumed",
            "wait_only_reset_excluded",
            "closed_attempts",
        ),
        name="C211 raw oracle rollout census",
    )
    rollout = {
        key: _oracle_count(value, name="rollout census " + key)
        for key, value in rollout.items()
    }
    if (
        rollout["closed_attempts"] != 32
        or rollout["source_episodes_consumed"]
        != rollout["closed_attempts"] + rollout["wait_only_reset_excluded"]
    ):
        raise LaunchRefused(
            "C211 raw oracle rollout census does not close over its own resets"
        )

    hard_contract = _validate_c211_hard_contract(
        row["training_contract_artifact"],
        checkout=checkout,
        oracle_namespace=oracle_namespace,
        lineage=lineage,
        recipe=recipe,
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
            "question_source",
            "target_recipe",
            "target_validity_mask",
            "target_observation_noise",
            "incoming_ball_fields",
            "desired_contact_fields_observed",
            "reset_inverse_solve",
            "online_solver_calls",
            "online_lm_calls",
            "question_rng",
        ),
        name="C211 raw oracle question contract",
    )
    expected_question = {
        "target_source": TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "target_observation_noise": False,
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "desired_contact_fields_observed": False,
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "question_rng": _question_rng_contract(),
    }
    if question != expected_question:
        raise LaunchRefused(
            "C211 raw oracle direct-ball/000/counter contract differs"
        )

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
    single_stroke = completion["single_stroke"]
    # 人话:oracle32 要的是"32 次已关闭的挥拍尝试跑完并留下证据",不是"未开训 policy
    # 已经会打 32 次完整的球"。single_stroke 因此只做取值域检查,真数字下面用逐集
    # 重数来核对(§8.3:不以"必须零次/必须满分"循环要求未开训 policy 已经学会)。
    if (
        completion["requested"] != 32
        or completion["terminal"] != 32
        or type(single_stroke) is not int
        or not 0 <= single_stroke <= 32
        or type(control_steps) is not int
        or control_steps < 32
    ):
        raise LaunchRefused("C211 raw oracle did not close 32 terminal attempts")

    episodes = row["episodes"]
    if type(episodes) is not list or len(episodes) != 32:
        raise LaunchRefused("C211 raw oracle episode ledger must contain 32 rows")
    episode_step_sum = 0
    sample_indices: set[int] = set()
    sample_sha256: set[str] = set()
    draw_intervals: set[tuple[int, int]] = set()
    census_rows: list[tuple[str, tuple[str, ...]]] = []
    for index, episode in enumerate(episodes):
        item = _exact_dict(
            episode,
            (
                "episode",
                "control_steps",
                "terminal_phase",
                "termination_reasons",
                "sampler_sample_index",
                "sampler_sample_sha256",
                "sampler_draw_start",
                "sampler_draw_end",
                "incoming_ball_observation",
                "selected_rubber_evidence_sha256",
                "achieved_analytic_flight",
                "predicted_outcome",
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
        reasons = _oracle_termination_reasons(
            item["termination_reasons"],
            name="C211 raw oracle episode %d termination reasons" % index,
        )
        if (
            item["episode"] != index
            or type(item["control_steps"]) is not int
            or item["control_steps"] <= 0
            or item["terminal_phase"] not in ORACLE_TERMINAL_PHASES
            or type(item["sampler_sample_index"]) is not int
            or item["sampler_sample_index"] < 0
            or _B._sha256(
                item["sampler_sample_sha256"],
                name="C211 sampler sample SHA",
            ) != item["sampler_sample_sha256"]
            or type(item["sampler_draw_start"]) is not int
            or type(item["sampler_draw_end"]) is not int
            or item["sampler_draw_start"] < 0
            or item["sampler_draw_end"] <= item["sampler_draw_start"]
            or incoming["source"]
            != "runtime_actor_and_critic_observation_terms"
            or actor != critic
            or item["selected_rubber_evidence_sha256"]
            != selected["row_sha256"][index]
        ):
            raise LaunchRefused("C211 raw oracle episode evidence differs")
        sample_index = item["sampler_sample_index"]
        sample_digest = item["sampler_sample_sha256"]
        draw_interval = (item["sampler_draw_start"], item["sampler_draw_end"])
        if (
            sample_index in sample_indices
            or sample_digest in sample_sha256
            or draw_interval in draw_intervals
        ):
            raise LaunchRefused(
                "C211 raw oracle does not prove one distinct sampler receipt per reset"
            )
        sample_indices.add(sample_index)
        sample_sha256.add(sample_digest)
        draw_intervals.add(draw_interval)
        selected_contact = selected["classifications"][index] == "selected_rubber"
        flight, prediction = _validate_c211_analytic_pair(
            item["achieved_analytic_flight"], item["predicted_outcome"],
            selected=selected_contact,
        )
        live_episode = observed_bundle["episodes"][index]
        if (
            flight != live_episode["achieved_analytic_flight"]
            or prediction != live_episode["predicted_outcome"]
        ):
            raise LaunchRefused("C211 raw analytic row differs from pinned live bundle")
        episode_step_sum += item["control_steps"]
        census_rows.append((item["terminal_phase"], reasons))
    if episode_step_sum != control_steps:
        raise LaunchRefused("C211 raw oracle episode/control-step ledger differs")
    census = _oracle_termination_census(census_rows)

    termination = _exact_dict(
        row["termination"],
        ("allowed_reason", "by_reason", "unexpected_by_reason", "phase_by_reason"),
        name="C211 raw oracle termination",
    )
    phases = termination.get("phase_by_reason")
    # 人话:发布出来的三张聚合表(总账 / 名单外账 / 阶段x原因账)必须和上面逐集重数
    # 的结果逐字相等。这是"普查守恒" —— 收据不能自己报一个跟自己 32 行对不上的总数。
    expected_unexpected = {
        reason: count
        for reason, count in census["by_reason"].items()
        if reason != ORACLE_SINGLE_STROKE_REASON
    }
    if (
        termination.get("allowed_reason") != ORACLE_SINGLE_STROKE_REASON
        or termination.get("by_reason") != census["by_reason"]
        or termination.get("unexpected_by_reason") != expected_unexpected
        or type(phases) is not dict
        or set(phases) != set(ORACLE_TERMINAL_PHASES)
        or any(
            phases[phase] != census["phase_by_reason"][phase]
            for phase in ORACLE_TERMINAL_PHASES
        )
    ):
        raise LaunchRefused(
            "C211 raw oracle termination ledger differs from its own 32 episodes"
        )
    if census["by_reason"].get(ORACLE_SINGLE_STROKE_REASON, 0) != single_stroke:
        raise LaunchRefused(
            "C211 raw oracle single-stroke completion count differs from its episodes"
        )

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
    published_hard = _exact_dict(
        safety["hard_termination_by_reason"], HARD_TERMINATION_UNION,
        name="C211 raw oracle hard termination ledger",
    )
    hard = {
        name: _oracle_count(
            published_hard[name], name="hard termination count %s" % name
        )
        for name in HARD_TERMINATION_UNION
    }
    table_contacts = _oracle_count(
        safety["robot_table_contact_count"], name="robot table contact count"
    )
    nonfinite = _oracle_count(
        safety["projection_nonfinite_count"], name="projection nonfinite count"
    )
    # 守恒:safety 那条独立通道(runtime 逐集 hard_termination_by_reason 累加)必须和
    # termination 那条通道(逐集 termination_reasons 重数)给出同一组数字。两条通道
    # 在运行时是分开写的,对不上就说明有一条在撒谎。
    if (
        safety["control_step_denominator"] != control_steps
        or any(
            hard[name] != census["by_reason"].get(name, 0)
            for name in HARD_TERMINATION_UNION
        )
        or table_contacts != census["by_reason"].get("robot_hit_table", 0)
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
        raise LaunchRefused(
            "C211 raw oracle safety ledger differs from its own termination census"
        )
    # 人话:到这里才是"能不能放行"的判据,而且只看两类**实现故障** ——
    #   * qdes-hard / actual-hard:指令或实测关节角越了硬边界,是控制实现坏了;
    #   * nonfinite:投影里出了 NaN/Inf,是数值实现坏了。
    # 摔倒 / 太低 / 撞桌**不在**这里,它们是未开训 policy 的行为证据,只上报计数
    # (§12.4「qdes-hard/actual-hard/nonfinite 才是实现 strict-zero」;
    #  §8.3「不以『必须零次』循环要求未开训 policy 已经学会平衡」)。
    implementation_strict_zero = {
        **{name: hard[name] for name in STRICT_HARD_TERMINATION_UNION},
        "projection_nonfinite_count": nonfinite,
    }
    if any(count != 0 for count in implementation_strict_zero.values()):
        raise LaunchRefused(
            "C211 raw oracle observed a qdes-hard/actual-hard/nonfinite "
            "implementation failure"
        )

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
        # 人话:这一跑到底"32 集里各阶段各死法分别多少集",直接写进收据。
        # 读的人不用去翻 32 行原始 JSON,也不用只看一个 PASS/FAIL。
        "termination_census": {
            "closed_attempt_episodes": census["episodes"],
            "source_episodes_consumed": rollout["source_episodes_consumed"],
            "wait_only_reset_excluded": rollout["wait_only_reset_excluded"],
            "episodes_by_terminal_phase": census["episodes_by_phase"],
            "terminal_reason_totals": census["by_reason"],
            "terminal_reason_by_phase": census["phase_by_reason"],
            "single_stroke_complete_count": single_stroke,
            "physical_fall_by_reason": {
                reason: census["by_reason"].get(reason, 0)
                for reason in PHYSICAL_FALL_REASONS
            },
            "robot_hit_table_count": table_contacts,
            "implementation_strict_zero": implementation_strict_zero,
        },
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
            "target_source",
            "question_source",
            "question_rng",
            "target_recipe",
            "target_validity_mask",
            "incoming_ball_fields",
            "reset_inverse_solve",
            "online_solver_calls",
            "online_lm_calls",
                "seed",
                "observed_oracle_bundle_artifact",
                "observed_oracle_bundle_content_sha256",
                "raw_oracle_artifact",
            "raw_oracle_kind",
            "raw_oracle_content_sha256",
            "control_step_denominator",
            "selected_rubber_episode_denominator",
            "actual_selected_rubber_contact_count",
            "termination_census",
        ),
        name="C211 oracle32 receipt",
    )
    expected = {
        "schema_version": 3,
        "kind": ORACLE32_KIND,
        "diagnostic_unauthorized": True,
        "verdict": "PASS",
        "episodes": 32,
        "recipe_id": recipe["recipe_id"],
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
        "target_source": TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng": _question_rng_contract(),
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
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
    _observed_pin, observed_path = _external_pin(
        row["observed_oracle_bundle_artifact"],
        name="C211 observed oracle bundle artifact",
    )
    expected_observed_path = oracle_namespace / C211_OBSERVED_BUNDLE_FILENAME
    if observed_path.resolve(strict=True) != expected_observed_path.resolve(strict=True):
        raise LaunchRefused("C211 observed oracle bundle artifact path differs")
    try:
        observed_bundle = _C._load_canonical(observed_path)
    except (_C.EvidenceError, OSError, ValueError) as exc:
        raise LaunchRefused("C211 observed oracle bundle is invalid") from exc
    if row["observed_oracle_bundle_content_sha256"] != canonical_sha256(
        observed_bundle
    ):
        raise LaunchRefused("C211 observed oracle bundle content SHA differs")
    raw_facts = _validate_c211_raw_oracle(
        _artifact_path,
        checkout=checkout,
        oracle_namespace=oracle_namespace,
        launch_claim_sha256=result["launch_claim_sha256"],
        recipe=recipe,
        lineage=lineage,
        materialization=materialization,
        policy=policy,
        observed_bundle=observed_bundle,
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
        "termination_census": raw_facts["termination_census"],
    }
    if any(row[key] != wanted for key, wanted in expected_raw.items()):
        raise LaunchRefused("C211 oracle32 receipt differs from parsed raw evidence")
    return {
        "oracle32_result": pin,
        **row,
        "raw_oracle_artifact": dict(artifact),
    }


def _stable_artifact_bytes(
    path: Path, *, name: str, max_bytes: int
) -> tuple[bytes, dict[str, Any]]:
    """Read one bounded real file while binding its identity and digest."""

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


def _finite_tensor_tree(
    value: Any, *, name: str, torch_module: Any
) -> dict[str, int]:
    """Recursively require a non-empty, entirely finite tensor subtree."""

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
                    "C211 scale4096 checkpoint %s tensor cannot be audited" % name
                ) from exc
            if not finite:
                raise LaunchRefused(
                    "C211 scale4096 checkpoint %s contains a non-finite tensor"
                    % name
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
            "C211 scale4096 checkpoint %s contains no tensors" % name
        )
    return {"tensor_count": tensor_count, "element_count": element_count}


def _checkpoint_run_dir(
    *, log_raw: bytes, checkout: Path, namespace: Path
) -> Path:
    """Resolve the unique RSL log directory printed by this exact namespace."""

    try:
        lines = log_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LaunchRefused("C211 scale4096 run log is not UTF-8") from exc
    marker = " | log: "
    candidates = [
        line.rsplit(marker, 1)[1]
        for line in lines
        if line.startswith("[INFO] Task: ") and marker in line
    ]
    if len(candidates) != 1:
        raise LaunchRefused(
            "C211 scale4096 run log lacks one exact RSL log directory"
        )
    run_dir = _B._absolute_path(
        candidates[0], name="C211 scale4096 RSL log directory"
    )
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
        raise LaunchRefused("C211 scale4096 RSL run name differs")
    try:
        if (
            root.resolve(strict=True) != root
            or run_dir.resolve(strict=True) != run_dir
            or run_dir.parent != root
            or not stat.S_ISDIR(run_dir.lstat().st_mode)
        ):
            raise LaunchRefused(
                "C211 scale4096 RSL run directory escapes checkout"
            )
    except OSError as exc:
        raise LaunchRefused(
            "C211 scale4096 RSL run directory is missing"
        ) from exc
    return run_dir


def _terminal_json_events(
    log_raw: bytes, *, prefix: str, name: str
) -> list[dict[str, Any]]:
    try:
        lines = log_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LaunchRefused("C211 scale4096 run log is not UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant %s" % value)

    rows = []
    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            row = json.loads(
                line[len(prefix) :], parse_constant=reject_constant
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LaunchRefused(
                "C211 scale4096 %s JSON is invalid" % name
            ) from exc
        if type(row) is not dict:
            raise LaunchRefused(
                "C211 scale4096 %s must be a JSON object" % name
            )
        rows.append(row)
    return rows


def _plain_counter(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise LaunchRefused(
            "C211 scale4096 %s must be a nonnegative integer" % name
        )
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
            "C211 scale4096 %s lacks exactly %d contiguous terminal updates"
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
        raise LaunchRefused("C211 pre-long producer and gate schemas differ")
    try:
        log_text = log_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LaunchRefused("C211 scale4096 run log is not UTF-8") from exc
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
        raise LaunchRefused("C211 scale4096 pre-long gate rejected: %s" % exc) from exc
    unsigned = {
        "schema_version": 1,
        "kind": "action_ball_c211_4096x5_prelong_gate_binding_v1",
        "diagnostic_unauthorized": True,
        "launch_claim_sha256": launch_claim_sha256,
        "run_log_sha256": run_log["sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "semantic_marker_prefix": _S.PRELONG_SEMANTICS_MARKER_PREFIX,
        "semantic_update_count": len(semantic_updates),
        "gate": gate,
        "gate_sha256": canonical_sha256(gate),
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _terminal_checkpoint_iteration(expected_updates: int) -> int:
    """Return the iteration index of the LAST checkpoint a finished run writes.

    人话:「跑满 5 个 update」的最后一份存档叫 ``model_4.pt``、里面的 ``iter`` 是 4,
    不是 5 —— RSL-RL 的迭代变量在循环体内取 ``0..N-1``,收尾存盘用的就是那个末值。
    这道门以前拿 ``expected_updates`` 当文件名和 ``iter``,**任何预算下都不可能满足**。

    终局编号只有 ``_P``(A211/C211 共用的 pre-long gate)一处定义,这里不再手抄;
    但仍然当场把它和本发射器自己的预算对一遍,免得两边各改各的还互相不知道。
    """

    terminal = _P.TERMINAL_CHECKPOINT_ITERATION
    if (
        type(expected_updates) is not int
        or expected_updates < 1
        or _P.EXPECTED_UPDATES != expected_updates
        or terminal != expected_updates - 1
        or _P.TERMINAL_CHECKPOINT_FILENAME != "model_%d.pt" % terminal
    ):
        raise LaunchRefused(
            "C211 scale4096 budget and the shared pre-long terminal "
            "checkpoint index disagree"
        )
    return terminal


def _audit_scale4096_terminal(
    *,
    checkout: Path,
    namespace: Path,
    launch_claim_sha256: str,
) -> dict[str, Any]:
    """Independently recompute the terminal gate consumed by ``long4096``.

    The checkpoint is accepted only from the exact RSL directory named in this
    namespace's stable run log.  It is loaded on CPU with ``weights_only=True``;
    ordinary pickle loading has no fallback path.
    """

    expected_updates = BUDGETS["scale4096"][1]
    terminal_iteration = _terminal_checkpoint_iteration(expected_updates)
    log_raw, log_artifact = _stable_artifact_bytes(
        namespace / "run.log",
        name="C211 scale4096 terminal run log",
        max_bytes=512 << 20,
    )
    run_dir = _checkpoint_run_dir(
        log_raw=log_raw, checkout=checkout, namespace=namespace
    )
    checkpoint_path = run_dir / _P.TERMINAL_CHECKPOINT_FILENAME
    checkpoint_raw, checkpoint_artifact = _stable_artifact_bytes(
        checkpoint_path,
        name="C211 scale4096 exact checkpoint",
        max_bytes=32 << 30,
    )
    try:
        import torch as torch_module
    except ImportError as exc:  # pragma: no cover - exact Pod dependency
        raise LaunchRefused(
            "PyTorch is required to audit the C211 scale4096 checkpoint"
        ) from exc
    try:
        checkpoint = torch_module.load(
            io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise LaunchRefused(
            "C211 scale4096 checkpoint failed safe CPU weights-only load; "
            "ordinary pickle execution is forbidden"
        ) from exc
    if type(checkpoint) is not dict:
        raise LaunchRefused("C211 scale4096 checkpoint root must be a dict")
    embedded_iteration = checkpoint.get("iter")
    infos = checkpoint.get("infos")
    if (
        type(embedded_iteration) is not int
        or embedded_iteration != terminal_iteration
        or type(infos) is not dict
        or infos.get("training_launch_claim_sha256") != launch_claim_sha256
    ):
        raise LaunchRefused(
            "C211 scale4096 checkpoint iteration/launch-claim binding differs"
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
                "C211 scale4096 checkpoint lacks %s state" % label
            )
        tensor_groups[label] = _finite_tensor_tree(
            subtree, name=label, torch_module=torch_module
        )

    # 这一跑属于哪种 reward 证据体制,由它自己的 joint-safety 收据自陈,不由发射器
    # 断言(见 action_ball_4096x5_prelong_gate 里那段三层查证)。诊断发射器只审诊断跑;
    # 正式跑的终局审计在 audit_reward_run.py 那条链上,不从这里借道。
    try:
        log_text = log_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LaunchRefused("C211 scale4096 run log is not UTF-8") from exc
    # 先定体制、再算范围:体制不对就当场拒,别让"正式跑缺证据"的理由盖住
    # "这台发射器根本不该审正式跑"这条更早的事实。
    try:
        reward_evidence_regime = _P.classify_reward_evidence_regime(log_text)
    except _P.PreLongGateRefused as exc:
        raise LaunchRefused(
            "C211 scale4096 reward-evidence regime is undetermined: %s" % exc
        ) from exc
    if reward_evidence_regime != _P.REWARD_EVIDENCE_REGIME_DIAGNOSTIC:
        raise LaunchRefused(
            "C211 scale4096 terminal audit only accepts diagnostic runs; "
            "observed reward-evidence regime %s" % reward_evidence_regime
        )
    try:
        reward_evidence_scope = _P.reward_activation_evidence_scope(
            log_text=log_text, expected_updates=expected_updates
        )
    except _P.PreLongGateRefused as exc:
        raise LaunchRefused(
            "C211 scale4096 reward-activation evidence scope rejected: %s" % exc
        ) from exc
    joint_rows = _ordered_terminal_events(
        _terminal_json_events(
            log_raw,
            prefix=_P.JOINT_SAFETY_PREFIX,
            name="joint-safety counter",
        ),
        event=_P.DIAGNOSTIC_JOINT_SAFETY_EVENT,
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
    # reward-safety 这一族由正式 reward activation ledger 铸,诊断跑结构上不产。
    # 适用就照旧要满 5 行(强度不变);不适用就是明账 0 行,而它承担的三项严格零
    # 计数改由 exact-behavior 的 termination_reason_* 供给 —— 不是记 0 跳过。
    reward_rows = (
        _ordered_terminal_events(
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
        if reward_evidence_scope["applicable"]
        else []
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
            row.get("status") != _P.DIAGNOSTIC_JOINT_SAFETY_STATUS
            or type(totals) is not dict
            or "actual_hard_edge_events" not in totals
        ):
            raise LaunchRefused(
                "C211 scale4096 joint-safety terminal counter %d is incomplete"
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
                "C211 scale4096 actual-hard terminal counter %d is incomplete"
                % index
            )
        actual_hard_terminal_count += _plain_counter(
            row["total_hard_terminal_count"], name="actual hard terminal count"
        )
        control = row.get("physx_control_position_limits")
        if type(control) is not dict or control.get("enabled") is not True:
            raise LaunchRefused(
                "C211 scale4096 actual-hard control telemetry is missing"
            )
        by_joint = control.get("by_joint")
        if type(by_joint) is not list or not by_joint:
            raise LaunchRefused(
                "C211 scale4096 actual-hard control telemetry has no joints"
            )
        for joint in by_joint:
            sides = joint.get("sides") if type(joint) is dict else None
            if type(sides) is not dict or set(sides) != {"lower", "upper"}:
                raise LaunchRefused(
                    "C211 scale4096 actual-hard control side telemetry is incomplete"
                )
            for side in sides.values():
                if type(side) is not dict or type(
                    side.get("nonfinite_readback_observed")
                ) is not bool:
                    raise LaunchRefused(
                        "C211 scale4096 actual-hard nonfinite counter is missing"
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
                "C211 scale4096 reward-safety terminal counter %d is incomplete"
                % index
            )
        for transition in transitions:
            terms = (
                transition.get("termination_terms")
                if type(transition) is dict
                else None
            )
            if (
                type(terms) is not list
                or not terms
                or any(type(term) is not str for term in terms)
            ):
                raise LaunchRefused(
                    "C211 scale4096 terminal transition lacks reason counters"
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
    # exact-behavior 的 termination_reason_* 是**每个活跃终止项一格**的固定 ABI,
    # 诊断跑照发。这两格就是 reward-safety 不在场时那三项严格零计数的真实出处。
    behavior_strict_hard_by_reason = {
        reason: 0 for reason in STRICT_HARD_TERMINATION_UNION
    }
    for index, row in enumerate(behavior_rows):
        counters = row.get("counters")
        required_balance_counters = {
            TASK_WAIT_STARTED_COUNTER,
            TASK_REVEAL_REACHED_COUNTER,
            "termination_reason_robot_hit_table_count",
            *(
                f"termination_reason_{reason}_count"
                for reason in STRICT_HARD_TERMINATION_UNION
            ),
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
                "C211 scale4096 exact-behavior nonfinite counters and survival/fall counters %d are missing"
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
                    f"C211 scale4096 {reason} reason-by-phase counters do not conserve "
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
                "C211 scale4096 robot_hit_table reason-by-phase counters do not "
                f"conserve in update {index}"
            )
        behavior_table_contact_count += table_total
        for phase, count in table_phase_counts.items():
            table_contact_by_phase[phase] += count
        for reason in STRICT_HARD_TERMINATION_UNION:
            behavior_strict_hard_by_reason[reason] += _plain_counter(
                counters[f"termination_reason_{reason}_count"],
                name=f"{reason} terminal count",
            )
        for key, value in counters.items():
            if "nonfinite" in key:
                behavior_nonfinite_count += _plain_counter(
                    value, name="exact-behavior %s" % key
                )

    # 每个 reason 各数各的,和为「至少有一项硬终止」的**上界**(同一集同时踩两项时
    # 会被数两次)。严格零这条性质不受影响:和为 0 当且仅当并集为 0。
    behavior_strict_hard_reason_sum = sum(
        behavior_strict_hard_by_reason.values()
    )
    if reward_evidence_scope["applicable"]:
        # 正式跑:两族都在,逐项对账,谁漂了都拒。
        if behavior_fall_by_reason != reward_fall_by_reason:
            raise LaunchRefused(
                "C211 scale4096 physical-fall behavior and terminal-transition counts differ"
            )
        if behavior_table_contact_count != table_contact_count:
            raise LaunchRefused(
                "C211 scale4096 table-contact behavior and terminal-transition counts differ"
            )
        if (
            behavior_strict_hard_by_reason["joint_qdes_forbidden"]
            != joint_qdes_terminal_count
            or behavior_strict_hard_by_reason["joint_actual_forbidden"]
            != joint_actual_terminal_count
        ):
            raise LaunchRefused(
                "C211 scale4096 joint-forbidden behavior and terminal-transition counts differ"
            )
        terminal_transition_source = "reward_safety_transition_markers"
    else:
        # 诊断跑:reward-safety 一族不在场,上面三项对账**两侧只剩一侧**,做不成。
        # 于是把它们的取值整体改由 exact-behavior 供给,并在收据里写清这一点 ——
        # 不允许出现「没观测到所以写 0」的数字。
        reward_fall_by_reason = dict(behavior_fall_by_reason)
        table_contact_count = behavior_table_contact_count
        joint_qdes_terminal_count = behavior_strict_hard_by_reason[
            "joint_qdes_forbidden"
        ]
        joint_actual_terminal_count = behavior_strict_hard_by_reason[
            "joint_actual_forbidden"
        ]
        strict_hard_termination_count = behavior_strict_hard_reason_sum
        terminal_transition_source = _P.DIAGNOSTIC_STRICT_ZERO_SOURCE_PREFIX
        # 守恒普查:重新取源之后,收据里那三个数必须跟它们自己的逐项出处对得上。
        # 这条挡的是「换了出处却漏接一项」——那样收据会报一个跟自己 5 行遥测
        # 对不上的总数,正是 oracle32 那次重定范围明令禁止的假收据。
        if (
            joint_qdes_terminal_count + joint_actual_terminal_count
            != strict_hard_termination_count
        ):
            raise LaunchRefused(
                "C211 scale4096 re-sourced strict-hard counters do not conserve "
                "against their exact-behavior reasons"
            )
    safety = {
        "observed_ppo_updates": expected_updates,
        "actual_hard_edge_event_count": actual_hard_edge_count,
        "actual_hard_terminal_count": actual_hard_terminal_count,
        "joint_qdes_forbidden_terminal_count": joint_qdes_terminal_count,
        "joint_actual_forbidden_terminal_count": joint_actual_terminal_count,
        "strict_hard_termination_count": strict_hard_termination_count,
        "table_contact_count": table_contact_count,
        "nonfinite_count": physics_nonfinite_count + behavior_nonfinite_count,
        "base_fell_tilt_terminal_count": behavior_fall_by_reason["base_fell_tilt"],
        "base_too_low_terminal_count": behavior_fall_by_reason["base_too_low"],
        "physical_fall_by_reason_phase": physical_fall_by_reason_phase,
        "table_contact_by_phase": table_contact_by_phase,
        "task_wait_started_by_update": task_wait_started_by_update,
        "task_wait_started_count": sum(task_wait_started_by_update),
        "task_reveal_reached_by_update": task_reveal_reached_by_update,
        "task_reveal_reached_count": sum(task_reveal_reached_by_update),
    }
    # 「哪几项必须严格零」是 pre-long gate 与四格 barrier 已经共用的一条策略,
    # 这里原来是它的**第三份手抄**。手抄的代价是真实的:同一条策略改了两处、
    # 漏了这一处,发射器就会在 barrier 还没看到这一跑之前先把它拒掉,另外两处
    # 的修改等于没生效。现在直接读共享出处,谁改都一起改。
    strict_zero_keys = tuple(_P.STRICT_ZERO_SAFETY_COUNTERS)
    if any(safety[key] != 0 for key in strict_zero_keys):
        raise LaunchRefused(
            "C211 scale4096 observed joint-qdes/joint-actual/nonfinite implementation counters are nonzero"
        )
    checkpoint_acceptance = {
        **checkpoint_artifact,
        "filename_iteration": terminal_iteration,
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
        # 收据自陈走了哪条分支。它**不放进** ``safety_counters``:那份 16 键
        # producer schema 被四格 barrier 的 ``_exact`` 逐键钉死,是两族共享的
        # 契约面;体制声明是范围说明,不是安全计数,放在这一层。
        # 四格聚合仍然拿得到它:pre-long gate 的结果里也有同一块,而 barrier 把
        # 那份结果整体做了内容寻址(``prelong_gate.content_sha256``)。
        "reward_activation_evidence_scope": reward_evidence_scope,
        "terminal_transition_source": terminal_transition_source,
        "behavior_strict_hard_by_reason": behavior_strict_hard_by_reason,
        "behavior_strict_hard_reason_sum": behavior_strict_hard_reason_sum,
        "safety_counters": safety,
        "prelong_gate": prelong_gate,
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _validate_scale_predecessor(
    value: Any,
    *,
    checkout: Path,
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
    namespace = _B._absolute_path(
        result["namespace"],
        name="C211 scale4096 predecessor namespace",
        must_exist=True,
    )
    recomputed_terminal = _audit_scale4096_terminal(
        checkout=checkout,
        namespace=namespace,
        launch_claim_sha256=result["launch_claim_sha256"],
    )
    if result["terminal_acceptance"] != recomputed_terminal:
        raise LaunchRefused(
            "C211 scale4096 predecessor terminal checkpoint/safety acceptance differs"
        )
    return {
        "artifact": pin,
        "stage": "scale4096",
        "launch_claim_sha256": result["launch_claim_sha256"],
        "content_sha256": result["content_sha256"],
        "completion": expected_completion,
        "finite_model_artifact": recomputed_terminal["checkpoint"],
        "safety_counters": recomputed_terminal["safety_counters"],
        "prelong_gate": recomputed_terminal["prelong_gate"],
        "terminal_acceptance_content_sha256": recomputed_terminal[
            "content_sha256"
        ],
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
    if row["recipe_id"] not in RECIPE_IDS:
        raise LaunchRefused("C211 launcher accepts only its two code-owned grid recipes")
    if row["wait_contract"] != _wait_contract():
        raise LaunchRefused("C211 launch wait schedule differs")
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
        raise LaunchRefused("namespace parent must be the dedicated C211 experiment root")
    expected_parent = (
        Path(row["source"]["checkout"])
        / _B.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / EXPERIMENT_NAME
    )
    if parent != expected_parent:
        raise LaunchRefused(
            "namespace parent must be the checkout-local C211 experiment root"
        )
    log_path = _B._absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must equal <namespace>/run.log")

    requirements = {
        "materialization_result": stage != "materialize",
        "recipe_result": stage not in ("materialize", "recipe"),
        "oracle32_result": stage in ("scale4096", "long4096"),
        "predecessor_result": stage == "long4096",
        "four_grid_scale4096_receipt": stage == "long4096",
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
        "recipe_id": row["recipe_id"],
        "lineage": _pin(row["lineage"], name="spec.lineage"),
        "materialization_result": row["materialization_result"],
        "recipe_result": row["recipe_result"],
        "oracle32_result": row["oracle32_result"],
        "predecessor_result": row["predecessor_result"],
        "four_grid_scale4096_receipt": (
            _four_grid_prelong_receipt_pin(row["four_grid_scale4096_receipt"])
            if stage == "long4096"
            else None
        ),
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
        "strict_hard_termination_count_max": 0,
        "behavioral_termination_policy": (
            "fall_too_low_and_table_remain_terminal_but_are_reported_by_reason_phase_"
            "without_an_unvalidated_numeric_cutoff"
        ),
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
        # 探索包本轮四格相同(标准 rsl_rl 初始化 + sigma 1.0 + scalar)。三个 override
        # 仍然从 recipe 合同里发出而不是硬钉字面量 —— 合同的值来自 matched_contract,
        # 所以"四格相同"这件事也由同一份封印背书。
        "algo.policy.init_noise_std=%s"
        % _float_override_token(
            recipe["init_noise_std"], name="algo.policy.init_noise_std"
        ),
        "algo.policy.noise_std_type=%s" % recipe["noise_std_type"],
        "action_ball_actor_init_mode=%s" % recipe["actor_init_mode"],
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
        # 注册 DR 元组整包写进 argv,不靠 leaf 记忆:三个键一起出现才是一档合法的
        # DR 档,train.py 见到半套会直接拒。四格里**唯一**变的就是最后这个布尔 ——
        # false = DR-L0(现状),true = DR-L0N(plant 与 L0 逐字节相同,只把
        # joint_pos/joint_vel/base_ang_vel 三路本体感通道的噪声打开)。
        "task.domain_rand.stable_ready_plant=true",
        "task.domain_rand.startup_physics_material=false",
        "task.domain_rand.startup_joint_default_pos=false",
        "task.domain_rand.policy_observation_corruption=%s"
        % ("true" if recipe["policy_observation_corruption"] else "false"),
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
        "task.racket.action_ball_target_source=direct_ball",
        "task.racket.action_ball_target_recipe=%s" % TARGET_RECIPE,
        "task.racket.action_ball_target_validity_mask=[false,false,false]",
        "task.racket.action_ball_target_observation_noise=false",
        "task.racket.action_ball_reuse_exact_question_until_semantics_change=false",
        "task.racket.action_ball_initial_center_single_question=true",
        "task.racket.adaptive_sigma=false",
        "task.racket.adaptive_sigma_monotonic=false",
        "task.racket.adaptive_sigma_normal=false",
        "task.racket.target_noise_white=0.0",
        "task.racket.target_noise_ar1_sigma=0.0",
        "task.racket.action_ball_diagnostic_unauthorized=true",
        "+task.racket.reference_guard_mode=%s" % recipe["reference_guard_mode"],
        "task.rewards.death_penalty_weight=%s" % weights["death_penalty"],
        "task.rewards.base_position_weight=0.0",
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
                "+action_ball_c211_oracle_bundle_output_path=%s"
                % (
                    Path(spec["namespace"])
                    / C211_OBSERVED_BUNDLE_FILENAME
                ),
                "+action_ball_c211_oracle_episodes=32",
                "+action_ball_c211_oracle_lineage_sha256=%s"
                % lineage["lineage_sha256"],
                "+action_ball_c211_oracle_recipe_contract_sha256=%s"
                % recipe["recipe_contract_sha256"],
                "+action_ball_c211_oracle_reward_contract_sha256=%s"
                % materialization["reward_contract_sha256"],
            )
        )
    return argv


def _output_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    stage = spec["stage"]
    runtime_blocked = stage in BLOCKED_RUNTIME_STAGES
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
        "ppo_update_count": 0 if runtime_blocked else spec["max_iterations"],
        "requested_ppo_update_count": spec["max_iterations"],
        "finite_model_save_interval": spec["save_interval"],
        "effective_reward_recipe": None,
        "policy_recipe": None,
        "c211_observed_oracle_bundle": None,
        "c211_raw_oracle32": None,
        "boot_marker": "Learning iteration",
        # Existing finite/long stages do not implement the matched
        # profiler-off A->C->C->A speed gate.  scale4096 additionally carries
        # the five-update pre-long evidence ledger and is never rate evidence.
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
        "diagnostic_unauthorized": True,
        "runtime_gate": ORACLE_RUNTIME_BLOCKER if runtime_blocked else "READY",
        "runtime_dependencies": (
            list(ORACLE_RUNTIME_DEPENDENCIES) if runtime_blocked else []
        ),
        "wait_contract": _wait_contract(),
        "physical_reset_semantics": "separate_safe_ready_zero_velocity",
        "teacher_reveal_semantics": "measured_frame0_with_public_countdown",
        "passive_hold_after_reveal_required": False,
    }
    namespace = Path(spec["namespace"])
    if stage == "materialize":
        output["effective_reward_recipe"] = str(namespace / REWARD_RECIPE_FILENAME)
        output["boot_marker"] = "ACTION_BALL_EFFECTIVE_REWARD_RECIPE_MATERIALIZED_JSON"
    elif stage == "recipe":
        output["policy_recipe"] = str(namespace / POLICY_RECIPE_FILENAME)
        output["boot_marker"] = "ACTION_BALL_POLICY_RECIPE_MATERIALIZED"
    elif stage == "oracle32":
        output["c211_observed_oracle_bundle"] = str(
            namespace / C211_OBSERVED_BUNDLE_FILENAME
        )
        output["c211_raw_oracle32"] = str(
            namespace / "teacher_qdes_oracle_32ep.json"
        )
        output["boot_marker"] = "ACTION_BALL_C211_OBSERVED_ORACLE_BUNDLE_JSON"
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
            checkout=Path(spec["source"]["checkout"]),
            materialization=materialization,
            policy=policy,
            oracle32=oracle32,
        )
        if spec["stage"] == "long4096"
        else None
    )
    four_grid_receipt = (
        _validate_four_grid_prelong_receipt(
            spec["four_grid_scale4096_receipt"],
            checkout=Path(spec["source"]["checkout"]),
        )
        if spec["stage"] == "long4096"
        else None
    )
    return {
        "reward_materialization": materialization,
        "policy_recipe_materialization": policy,
        "oracle32_receipt": oracle32,
        "predecessor_result": predecessor,
        "four_grid_scale4096_receipt": four_grid_receipt,
    }


def _admission_training_argv(
    spec: Mapping[str, Any], bundle: Mapping[str, Any]
) -> list[str]:
    row = _exact_dict(
        bundle,
        (
            "lineage",
            "recipe",
            "isaac_four_grid_manifest",
            "question_contract",
            "normalizers",
            "checkpoint_contract",
            "termination_contract",
            "continuation_stop_gate",
            "curriculum_scope",
        ),
        name="C211 claim bundle",
    )
    if row["isaac_four_grid_manifest"] != _isaac_four_grid_manifest():
        raise LaunchRefused("C211 claim four-grid manifest drifted")
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
    physical_reservation_registry=True,
    forbidden_namespace_experiment_names=(
        "agibot_a3_action_ball_measured_vendor_v2_n1_diagnostic",
        "agibot_a3_action_ball_a225_four_arm_diagnostic",
        "agibot_a3_action_ball_c225_diagnostic",
        "agibot_a3_action_ball_a211_four_arm_diagnostic",
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
    recipe = _recipe_contract(spec["recipe_id"])
    inputs = _materialization_inputs(
        spec, recipe=recipe, lineage=lineage
    )
    output_contract = _output_contract(spec)
    bundle = {
        "lineage": lineage,
        "recipe": recipe,
        "isaac_four_grid_manifest": _isaac_four_grid_manifest(),
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
    payload: Mapping[str, Any], *, claimed: bool = True
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    recipe = _recipe_contract(spec["recipe_id"])
    expected_bundle = {
        "lineage": lineage,
        "recipe": recipe,
        "isaac_four_grid_manifest": _isaac_four_grid_manifest(),
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
    for name, term in by_name.items():
        weight = term.get("weight")
        if (
            type(weight) not in (int, float)
            or not math.isfinite(float(weight))
        ):
            raise LaunchRefused("C211 runtime reward weight is malformed: %s" % name)
        if float(weight) == 0.0:
            continue
        if name in PROHIBITED_TASK_DIRECTED_TERMS:
            raise LaunchRefused(
                "C211 prohibited task-directed reward term is active: %s" % name
            )
        task_directed = name.startswith(TASK_DIRECTED_PREFIXES)
        if task_directed and name not in ALLOWED_TASK_DIRECTED_TERMS:
            raise LaunchRefused(
                "C211 unregistered task-directed reward term is active: %s" % name
            )


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
    # 借来的那台验证器要"发射方声明的探索包"和"运行时真吐出来的探索包"逐字节相等。
    # 它的默认值是 A225 那条 vendor-v2 诊断线自己的 log/0.02;C211 必须把**本格注册
    # 的**探索包传进去,否则默认值会把 C 夹在两条互斥的门中间:先在那里被要求
    # log/0.02,二十几行后又被下面的 expected_policy 要求等于 recipe 的 scalar/1.0,
    # 没有任何一份 recipe 能同时满足。传的是 _recipe_contract 里由
    # recipe_contract_sha256 封住的四格 matched exploration_package,不是字面量,
    # 所以这不是放宽门:包一旦和注册的格子不符,这里和下面都仍然当场拒。
    try:
        validated = _OLD._validate_policy_materialization(
            {"path": str(path), "sha256": _B.sha256_file(path)},
            checkout=checkout,
            bundle=bundle,
            expected_noise_std_type=recipe["noise_std_type"],
            expected_init_noise_std=recipe["init_noise_std"],
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
        "recipe_id": recipe["recipe_id"],
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


def _runtime_oracle32_receipt(
    *,
    bundle_path: Path,
    namespace: Path,
    checkout: Path,
    launch_claim_sha256: str,
    recipe: Mapping[str, Any],
    lineage: Mapping[str, Any],
    materialization: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the C sidecars from the runner-bound 000 observation bundle."""

    try:
        resolved_bundle = bundle_path.resolve(strict=True)
        if resolved_bundle != (namespace / C211_OBSERVED_BUNDLE_FILENAME).resolve(strict=True):
            raise LaunchRefused("C211 observed bundle path differs")
        observed = _C._load_canonical(resolved_bundle)
        publication = _C.publish_bundle(observed, namespace=namespace)
    except (_C.EvidenceError, FileNotFoundError, OSError, ValueError) as exc:
        raise LaunchRefused(
            "C211 runner-before-oracle bundle publication failed: %s" % exc
        ) from exc
    raw_pin = publication.get("raw_oracle_artifact")
    if type(raw_pin) is not dict or set(raw_pin) != set(PIN_KEYS):
        raise LaunchRefused("C211 oracle sidecar publication pin is malformed")
    raw_path = Path(raw_pin["path"])
    raw_facts = _validate_c211_raw_oracle(
        raw_path,
        checkout=checkout,
        oracle_namespace=namespace,
        launch_claim_sha256=launch_claim_sha256,
        recipe=recipe,
        lineage=lineage,
        materialization=materialization,
        policy=policy,
        observed_bundle=observed,
    )
    if raw_pin["sha256"] != raw_facts["file_sha256"]:
        raise LaunchRefused("C211 published raw-oracle file SHA differs")
    unsigned = {
        "schema_version": 3,
        "kind": ORACLE32_KIND,
        "diagnostic_unauthorized": True,
        "verdict": "PASS",
        "episodes": 32,
        "recipe_id": recipe["recipe_id"],
        "lineage_sha256": lineage["lineage_sha256"],
        "recipe_contract_sha256": recipe["recipe_contract_sha256"],
        "reward_contract_sha256": materialization["reward_contract_sha256"],
        "runtime_effective_reward_sha256": materialization[
            "runtime_effective_reward_sha256"
        ],
        "runtime_policy_recipe_sha256": policy[
            "runtime_policy_recipe_sha256"
        ],
        "actor_contract": ACTOR_CONTRACT,
        "actor_width": ACTOR_WIDTH,
        "critic_contract": CRITIC_CONTRACT,
        "critic_width": CRITIC_WIDTH,
        "trainability_contract": TRAINABILITY_CONTRACT,
        "target_source": TARGET_SOURCE,
        "question_source": "runtime_curriculum_sampler",
        "question_rng": _question_rng_contract(),
        "target_recipe": TARGET_RECIPE,
        "target_validity_mask": list(TARGET_VALIDITY_MASK),
        "incoming_ball_fields": list(INCOMING_BALL_FIELDS),
        "reset_inverse_solve": False,
        "online_solver_calls": 0,
        "online_lm_calls": 0,
        "seed": lineage["seed"],
        "observed_oracle_bundle_artifact": {
            "path": str(resolved_bundle),
            "sha256": hashlib.sha256(resolved_bundle.read_bytes()).hexdigest(),
        },
        "observed_oracle_bundle_content_sha256": canonical_sha256(observed),
        "raw_oracle_artifact": dict(raw_pin),
        "raw_oracle_kind": raw_facts["kind"],
        "raw_oracle_content_sha256": raw_facts["content_sha256"],
        "control_step_denominator": raw_facts["control_steps"],
        "selected_rubber_episode_denominator": raw_facts[
            "selected_rubber_episode_denominator"
        ],
        "actual_selected_rubber_contact_count": raw_facts[
            "actual_selected_rubber_contact_count"
        ],
        # 收据自陈:这一跑各阶段各原因分别多少集,直接写在 PASS 旁边。
        "termination_census": raw_facts["termination_census"],
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
        **_update_profile_exec_environment(os.environ),
        **_prelong_semantics_exec_environment(
            spec["stage"],
            payload["materialization_inputs"]["reward_materialization"],
        ),
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
    outer = _exact_dict(
        plan,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="C211 launch plan",
    )
    payload = outer["canonical_payload"]
    if expected != outer["launch_claim_sha256"]:
        raise LaunchRefused("--confirm-claim differs from freshly recomputed plan")
    if (
        outer["schema_version"] != SCHEMA_VERSION
        or outer["kind"] != CLAIM_KIND
        or type(payload) is not dict
        or canonical_sha256(payload) != expected
    ):
        raise LaunchRefused("launch plan payload seal differs before execution")
    spec, _lineage, _recipe = _revalidate_claim_payload(payload, claimed=False)
    checkout = Path(spec["source"]["checkout"])
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
        elif spec["stage"] == "oracle32":
            oracle32 = _runtime_oracle32_receipt(
                bundle_path=Path(
                    payload["output_contract"][
                        "c211_observed_oracle_bundle"
                    ]
                ),
                namespace=namespace,
                checkout=checkout,
                launch_claim_sha256=expected,
                recipe=payload["bundle"]["recipe"],
                lineage=payload["bundle"]["lineage"],
                materialization=materialization,
                policy=policy,
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
        unsigned = {
            "schema_version": 2,
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
        if terminal_acceptance is not None:
            unsigned["terminal_acceptance"] = terminal_acceptance
        launch_result = {**unsigned, "content_sha256": canonical_sha256(unsigned)}
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
        "recipe_id": args.recipe_id,
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
        "four_grid_scale4096_receipt": _paired_pin(
            args.four_grid_scale4096_receipt_path,
            args.four_grid_scale4096_receipt_sha256,
            name="four-grid scale4096 receipt",
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
    template.add_argument("--recipe-id", required=True, choices=RECIPE_IDS)
    template.add_argument("--lineage-path", required=True)
    template.add_argument("--lineage-sha256", required=True)
    template.add_argument("--stage", choices=STAGE_ORDER, required=True)
    template.add_argument("--materialization-result-path")
    template.add_argument("--materialization-result-sha256")
    template.add_argument("--recipe-result-path")
    template.add_argument("--recipe-result-sha256")
    template.add_argument("--oracle32-result-path")
    template.add_argument("--oracle32-result-sha256")
    template.add_argument("--predecessor-result-path")
    template.add_argument("--predecessor-result-sha256")
    template.add_argument("--four-grid-scale4096-receipt-path")
    template.add_argument("--four-grid-scale4096-receipt-sha256")
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
