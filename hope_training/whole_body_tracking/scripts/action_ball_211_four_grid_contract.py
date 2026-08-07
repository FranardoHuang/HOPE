#!/usr/bin/env python3
"""Single code-owned authority for the formal A211/C211 Isaac 2x2 grid.

The launchers may select only their own family cells.  They must not rebuild or
extend this manifest locally.  The module is intentionally dependency-free so
the exact same Python-3.8-compatible authority is loaded by both launchers and
its file SHA can be included in each launch claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence


# 2026-08-05 第二轴改版(第二次,exp §5.6.2d 裁决):
#
# 第一次改版把第二轴从 PPO schedule 换成了探索包(零权重 bootstrap + sigma 0.1 对
# 标准 rsl_rl 初始化 + sigma 1.0)。这一次把探索包也定死:**四格全部用标准初始化 +
# sigma 1.0**(对齐 BeyondMimic / build_1),第二轴换成**本体感观测噪声的开关**。
#
# 为什么是这一轴:
#   * 尽调 §22 判本体感噪声"D1 开满",证据是外部 9/9 库 day-1 全开 + 智元连 play 都
#     保留 + build_1 全开,**零反例**;
#   * DR-L0 的裁定正好相反 —— 它判这条"会改估计误差与终止率",所以为归因先关;
#   * 两边都是推理,**谁都没实测过**,而成本只是一个布尔。上一轮恢复的那批随机性
#     (摩擦/连杆质量/PD/CoM/关节零点/出生位姿)里,这是唯一有真冲突的一条。
#     两格测它 = 全表性价比最高的 A/B。
#
# 噪声幅度用通道里已经定义好的值(与智元、build_1 同区间),本轮不新增通道也不改数:
#     joint_pos ±0.01 rad / joint_vel ±0.5 rad·s⁻¹ / base_ang_vel ±0.2 rad·s⁻¹
# 任务通道(desired contact / incoming ball / 时间)**不加噪**:那会改支撑集,
# 等于换题而不是换传感器(§22 闸 1)。
#
# cell_id 随轴一起改名。留着 "zero-weight-bootstrap-sigma0p1" 而四格实际全跑标准
# 初始化,会让收据、namespace 与 barrier 布局表同时说谎。
KIND = "action_ball_211_isaac_four_grid_manifest_v4"
A_OBS_NOISE_OFF_CELL_ID = (
    "A0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off"
)
A_OBS_NOISE_ON_CELL_ID = (
    "A1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on"
)
C_OBS_NOISE_OFF_CELL_ID = (
    "C0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off"
)
C_OBS_NOISE_ON_CELL_ID = (
    "C1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on"
)
CELL_IDS = (
    A_OBS_NOISE_OFF_CELL_ID,
    A_OBS_NOISE_ON_CELL_ID,
    C_OBS_NOISE_OFF_CELL_ID,
    C_OBS_NOISE_ON_CELL_ID,
)
FAMILY_CELL_IDS = {
    "A211": (A_OBS_NOISE_OFF_CELL_ID, A_OBS_NOISE_ON_CELL_ID),
    "C211": (C_OBS_NOISE_OFF_CELL_ID, C_OBS_NOISE_ON_CELL_ID),
}
# 与 whole_body_tracking.utils.training_contract 的 ACTION_BALL_ACTOR_INIT_MODE_* 字面量
# 必须逐字相同。本模块刻意 dependency-free(两个 launcher 用 py3.8 直接 exec 它),所以
# 这里是手抄副本;跨模块一致性由 tests/test_action_ball_211_isaac_four_grid.py 断言。
ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS = "zero_weight_ready_bias"
ACTOR_INIT_MODE_DEFAULT = "default"
ACTOR_INIT_MODES = (
    ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS,
    ACTOR_INIT_MODE_DEFAULT,
)
# 4σ 硬内带门按 hold 姿态逐关节复算出的全局 sigma 上界(绑定关节 waist_pitch:
# 余量 0.4007 rad / action_scale 0.5900 / 4)。零权重路线的 sigma 高于它 = 直接拒。
ZERO_WEIGHT_READY_BIAS_SIGMA_CEILING = 0.1698
# 标准初始化路线不受 4σ 门约束(演员均值不再是那个常数 hold qdes,包络几何不成立),
# 上界改由 train.py 的 (0, 1] 区间校验承担。
STANDARD_INIT_SIGMA_CEILING = 1.0
EXPLORATION_PACKAGES = {
    ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS: {
        "exploration_axis": "zero_weight_ready_bias_bootstrap_sigma0p1_log",
        "actor_init_mode": ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS,
        "init_noise_std": 0.1,
        "noise_std_type": "log",
        "four_sigma_hard_inner_gate_applies": True,
    },
    ACTOR_INIT_MODE_DEFAULT: {
        "exploration_axis": "standard_rsl_rl_initialization_sigma1p0_scalar",
        "actor_init_mode": ACTOR_INIT_MODE_DEFAULT,
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
        "four_sigma_hard_inner_gate_applies": False,
    },
}
EXPLORATION_CELL_KEYS = (
    "exploration_axis",
    "actor_init_mode",
    "init_noise_std",
    "noise_std_type",
    "four_sigma_hard_inner_gate_applies",
)
# 2026-08-05:探索包不再是差异轴,四格全部取标准初始化这一包。它因此从 cells[i]
# 搬回 matched_contract —— 这不是搬家的偏好问题:上一版把它放在 cell 上,正是为了让
# "还去 matched_contract 取 init_noise_std 的旧代码"直接 KeyError;现在方向反过来,
# 还去 cell 上取 init_noise_std 的代码同样应该 KeyError,而不是读到一个骗人的数。
MATCHED_EXPLORATION_PACKAGE = ACTOR_INIT_MODE_DEFAULT

# --------------------------------------------------------------------------- #
# 第二轴:本体感观测噪声开关
# --------------------------------------------------------------------------- #
# 与 whole_body_tracking.utils.training_contract 的
# ACTION_BALL_DR_L0N_* / DR-L0 identity 必须逐字相同。本模块刻意 dependency-free
# (两个 launcher 用 py3.8 直接 exec 它),所以这里是手抄副本;跨模块一致性由
# tests/test_action_ball_211_isaac_four_grid.py 断言。
DR_LEVEL_IDENTITY_OBS_NOISE_OFF = "action_ball_dr_l0_exact_all_off_v1"
DR_LEVEL_IDENTITY_OBS_NOISE_ON = (
    "action_ball_dr_l0n_plant_all_off_proprio_obs_noise_on_v1"
)
# term 名 -> [n_min, n_max]。这三行是 hope_env_cfg 的 ActionBall{A,C}211PolicyCfg 里
# 已经写着的 Unoise 边界,本轮只决定它们生不生效,不改数值也不加通道。
PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS = {
    "base_ang_vel_body": [-0.2, 0.2],
    "joint_pos": [-0.01, 0.01],
    "joint_vel": [-0.5, 0.5],
}
OBSERVATION_NOISE_CELL_KEYS = (
    "observation_noise_axis",
    "policy_observation_corruption",
    "proprioceptive_observation_noise_channels",
    "task_channel_observation_noise",
    "dr_level_identity",
)
OBSERVATION_NOISE_PACKAGES = {
    False: {
        "observation_noise_axis": (
            "policy_observation_corruption_off_dr_l0_nominal_sensor"
        ),
        "policy_observation_corruption": False,
        "proprioceptive_observation_noise_channels": None,
        "task_channel_observation_noise": False,
        "dr_level_identity": DR_LEVEL_IDENTITY_OBS_NOISE_OFF,
    },
    True: {
        "observation_noise_axis": (
            "policy_observation_corruption_on_proprioceptive_channels_only"
        ),
        "policy_observation_corruption": True,
        "proprioceptive_observation_noise_channels": {
            name: list(bounds)
            for name, bounds in PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS.items()
        },
        "task_channel_observation_noise": False,
        "dr_level_identity": DR_LEVEL_IDENTITY_OBS_NOISE_ON,
    },
}
# 第二轴换掉 PPO schedule 之后,四格共用同一份 PPO;保留 A0/C0 原本的保守 fixed lr1e-4,
# 使对照组(零权重格)相对上一版四格一字未动,新增变量只有 A1/C1 的探索包。
SHARED_PPO = {
    "schedule": "fixed",
    "learning_rate": 1.0e-4,
    "desired_kl": 0.01,
    "clip_param": 0.2,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
}
FORMAL_STAGE_ORDER = (
    "materialize",
    "recipe",
    "oracle32",
    "scale4096",
    "long4096",
)
# [已退役 2026-08-06] CANONICAL_SOURCE_TAPE / CANONICAL_BASE_QUESTION[_SHA256] /
# CANONICAL_TEACHER_PROJECTION[_SHA256] 五个常量连同 validate_base_question /
# validate_teacher_projection 两个函数已整体删除:它们没有任何生产调用点,只是把下面
# 这份 tracked 磁带artifact 的字节在代码里又抄了一遍,而"同一事实两处存"迟早对不上。
# 题面/教师投影的唯一权威是磁带本体及其 task receipt:
#   configs/action_ball_n1_measured_20260803/fresh_592835dc_take061/
#     rematerialized_1d5d9d44/tape/immutable_n1_tape.v1.1eeccd2aa7b7.json
#     (question_sha256=9b9cf4d6..., canonical_sha256=6e4a502d...)
#     current_lm.target.task_receipt.v5.5e09858672ac.json (teacher 投影 22 字段)
# 注意:磁带题面的 time_to_contact_s=1.84(tick 92)是**现役**值,不要跟课程 level=0 的
# 初始中心 tick 混为一谈 —— 后者由 launcher 的 _initial_center_timing_authority 钉死在
# tick 91 / 1.82 / stratum="center"(635652f6 给 _contact_time_tick_grid 加
# allow_zero_initial 之后初始带真正塌到中心 tick)。两者是不同的量,都在现役。
CANONICAL_MOTION_SHA256 = (
    "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
)


class FourGridContractError(ValueError):
    """Raised when a launcher or manifest differs from the shared authority."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FourGridContractError("four-grid value is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def validate_exploration_package(value: Any) -> dict:
    """Cross-lock the three exploration fields so no half-set can be sealed.

    人话:初始化方式、sigma、std 参数化必须整包对上,而且 4σ 门开关必须与初始化方式
    严格对应。任一项对不上直接拒——两条路线各自 fail-closed,谁都不能借另一条放行。
    """

    if type(value) is not dict:
        raise FourGridContractError("four-grid exploration package must be a dict")
    mode = value.get("actor_init_mode")
    if type(mode) is not str or mode not in ACTOR_INIT_MODES:
        raise FourGridContractError("four-grid actor_init_mode is not a known mode")
    expected = EXPLORATION_PACKAGES[mode]
    observed = {key: value.get(key) for key in EXPLORATION_CELL_KEYS}
    if observed != expected:
        raise FourGridContractError(
            "four-grid exploration package differs from the sealed %s package" % mode
        )
    gate = observed["four_sigma_hard_inner_gate_applies"]
    sigma = observed["init_noise_std"]
    if type(sigma) is not float or sigma != sigma or not (0.0 < sigma):
        raise FourGridContractError("four-grid init_noise_std must be a positive float")
    if mode == ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS:
        # 零权重 + 钉死 bias:演员均值就是那个常数 hold qdes,4σ 包络几何成立,门必须开。
        if (
            gate is not True
            or observed["noise_std_type"] != "log"
            or sigma > ZERO_WEIGHT_READY_BIAS_SIGMA_CEILING
        ):
            raise FourGridContractError(
                "zero-weight bootstrap cell must keep the 4-sigma hard inner gate and "
                "stay at or below the recomputed sigma ceiling"
            )
    else:
        # 标准初始化:sealed-mean 前提不成立,4σ 门显式跳过(不是"忘了开")。
        if (
            gate is not False
            or observed["noise_std_type"] != "scalar"
            or sigma > STANDARD_INIT_SIGMA_CEILING
        ):
            raise FourGridContractError(
                "standard-initialization cell must declare the 4-sigma gate skipped, "
                "use the scalar std parameterization and stay within (0, 1]"
            )
    return copy.deepcopy(observed)


def validate_observation_noise_package(value: Any) -> dict:
    """Cross-lock the corruption switch, the channel table and the DR identity.

    人话:"开不开噪声"、"哪几路带噪"、"跑的是哪一档 DR"必须整包对上。半套一律拒 ——
    比如声称 corruption=true 却不给通道表,或者给了通道表却挂着 DR-L0 的身份。
    任务通道加噪在**两种取值下都拒**:那不是这根轴的取值,是换了一道题。
    """

    if type(value) is not dict:
        raise FourGridContractError(
            "four-grid observation-noise package must be a dict"
        )
    corruption = value.get("policy_observation_corruption")
    if type(corruption) is not bool:
        raise FourGridContractError(
            "four-grid policy_observation_corruption must be an explicit bool"
        )
    expected = OBSERVATION_NOISE_PACKAGES[corruption]
    observed = {key: value.get(key) for key in OBSERVATION_NOISE_CELL_KEYS}
    if observed != expected:
        raise FourGridContractError(
            "four-grid observation-noise package differs from the sealed "
            "corruption=%s package" % corruption
        )
    if observed["task_channel_observation_noise"] is not False:
        raise FourGridContractError(
            "four-grid task channels must never be noised: noising the desired "
            "contact / incoming ball / timing channels changes the support set"
        )
    channels = observed["proprioceptive_observation_noise_channels"]
    if corruption:
        if (
            type(channels) is not dict
            or sorted(channels) != sorted(PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS)
            or observed["dr_level_identity"] != DR_LEVEL_IDENTITY_OBS_NOISE_ON
        ):
            raise FourGridContractError(
                "the noise-on cell must declare exactly the proprioceptive "
                "channel table and the DR-L0N identity"
            )
        for name, bounds in channels.items():
            want = PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS[name]
            if (
                type(bounds) is not list
                or len(bounds) != 2
                or any(type(item) is not float for item in bounds)
                or bounds != list(want)
                or bounds[0] >= bounds[1]
            ):
                raise FourGridContractError(
                    "proprioceptive noise channel %s differs from the sealed "
                    "bounds" % name
                )
    elif (
        channels is not None
        or observed["dr_level_identity"] != DR_LEVEL_IDENTITY_OBS_NOISE_OFF
    ):
        raise FourGridContractError(
            "the noise-off cell must declare no channel table and the DR-L0 "
            "identity"
        )
    return copy.deepcopy(observed)


def _cell(
    cell_id: str,
    task_family: str,
    reward_semantics: str,
    policy_observation_corruption: bool,
) -> dict:
    package = copy.deepcopy(
        OBSERVATION_NOISE_PACKAGES[policy_observation_corruption]
    )
    row = {
        "cell_id": cell_id,
        "task_family": task_family,
        "task_reward_semantics": reward_semantics,
        "ppo": copy.deepcopy(SHARED_PPO),
        "ppo_adaptation_axis": "fixed_learning_rate",
        "learning_rate_role": "constant",
        "contact_sigma_adaptation": False,
        **package,
    }
    validate_observation_noise_package(row)
    return row


def _build_canonical_manifest() -> dict:
    matched = {
        "soft_weights": {
            # 2026-08-05 层级对齐(exp §5.6 第 7 条):-300 -> -10。post-dt 由 -6.0 降到 -0.2。
            # 原值是合法上台折扣下界 3.33209 的 180%,"打成一次再摔"净亏;外部三库与 build_1
            # 均无 death penalty 这一项。joint_actual_forbidden 改 telemetry 后本项触发面已从
            # "唯一死因"塌回"摔倒/撞桌/NaN"。
    # 2026-08-07 Franco 裁定二(形状照开源对齐):三条限位罚的核与量纲一起换成开源 rad 口径
    # (限位处磨圆的 L1 hinge、尾部线性无上界、地板退役),旧 -5 作用在归一 [0,1] 上、与新数不可比。
    # 采纳值:qdes/actual 两条 barrier 取上游 BeyondMimic 同族的 -10;投影罚取 -1
    # (同等策略水平下每步 -0.027~-0.099,与 build_1 当时整条 qdes 轴 -0.0635 同量级)。
            "death_penalty": -10.0,
            "qdes_limit": -10.0,
            "qdes_projection": -1.0,
            "joint_limit": -10.0,
        },
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        # 2026-08-05(第二次改版):探索包不再是差异轴 —— 四格全部标准 rsl_rl 初始化 +
        # sigma 1.0 + scalar,4σ 硬内带门显式跳过。它因此从 cells[i] 搬回 matched_contract。
        # cells[i] 里刻意不留同名键:任何还去 cell 上取 init_noise_std 的旧代码会直接
        # KeyError,而不是读到一个"看起来是本格的、其实全格相同"的数字。
        "exploration_axis_is_registered_difference": False,
        "exploration_package": copy.deepcopy(
            EXPLORATION_PACKAGES[MATCHED_EXPLORATION_PACKAGE]
        ),
        "ppo": copy.deepcopy(SHARED_PPO),
        "ppo_adaptation_axis": "fixed_learning_rate",
        "entropy_coef": 0.01,
        "reference_guard_mode": "metrics_only",
        "wait_contract": {
            "schedule": {
                "schema_version": 1,
                "kind": "action_ball_pre_task_wait_schedule",
                "seed": 20260804,
                "min_wait_ticks": 5,
                "max_wait_ticks": 25,
                "episode_horizon_ticks": 500,
                "required_active_ticks": 200,
                "unit": "policy_tick",
                "distribution": "uniform_integer_policy_ticks_inclusive",
                "counter_algorithm": "sha256_rejection_u64_v1",
                "canonical_sha256": (
                    "58aa7bb62406d301df619caf7026af8d595f4b8cd9594ea8441b4c89997d400e"
                ),
            },
            "policy_dt_s": 0.02,
            "in_loop_expansion_prohibited": True,
        },
        "seed": 0,
        "runtime_question_source": {
            "action_id": "take_061_unit04_bh",
            "action_uid": 5527597793770800,
            "teacher_id": "Take_061_unit04_BH",
            "motion_sha256": CANONICAL_MOTION_SHA256,
            "source": "runtime_curriculum_sampler",
            "cadence": "every_episode_reset",
            "selection": "sample_current_domain_levels",
            "curriculum_domain_levels_consulted_every_reset": True,
            "sampler_runs_every_reset": True,
            "physical_rng_draw_count_authority": (
                "sample_receipt_draw_end_minus_draw_start"
            ),
            "zero_physical_rng_draw_claim_permitted": False,
            "checkpoint_resume": "exact_sampler_and_curriculum_state",
            "family_target_providers": {
                "A211": "online_solver_with_complete_semantic_answer_cache",
                "C211": "direct_ball_no_inverse_no_answer_cache",
            },
            "shared_ac_question_claim": (
                "same_sampler_algorithm_seed_initial_domain_and_action_not_one_frozen_question"
            ),
        },
        "formal_budgets": {
            "materialize": [1, 0, 1],
            "recipe": [1, 0, 1],
            "oracle32": [1, 0, 1],
            "scale4096": [4096, 5, 1],
            "long4096": [4096, 1000, 100],
        },
        "control_step_action_delay": [0, 0],
        "contact_sigma_adaptation": False,
        "contact_sigma_contract": "static_rollout0_widths",
        # 出生位姿仍然是标准站位:上一轮落地的 start_pose_ramp 挂在 DR-L1 那两片 leaf
        # 上,本四格跑的是 DR-L0 / DR-L0N,斜坡不参与。写在这里是为了让"四格用的是
        # 标准初始化"这句话在收据里有据可查,而不是靠读者记得。
        "start_pose_ramp": None,
    }
    unsigned = {
        "schema_version": 4,
        "kind": KIND,
        "formal_cell_count": 4,
        "cell_order": list(CELL_IDS),
        "matched_contract": matched,
        "registered_difference_axes": [
            "task_semantics_and_reward",
            "policy_observation_corruption_cell",
        ],
        "deferred_difference_axes": [
            # exp §5.6.2c:在从未观测到一次接触前,LR schedule 的差异无法被任何指标分辨。
            "ppo_learning_rate_schedule_cell",
            # exp §5.6.2d:探索包这一轴本轮定死在标准初始化 + sigma 1.0(对齐
            # BeyondMimic / build_1),零权重 bootstrap 路线降级为 later。
            "actor_initialization_and_exploration_sigma_cell",
        ],
        "adaptive_term_disambiguation": {
            "adaptive_means": "ppo_kl_learning_rate_schedule",
            "ppo_kl_learning_rate_schedule": "disabled_fixed_learning_rate_all_cells",
            "contact_kernel_sigma_controller": "disabled_static_all_cells",
            "init_noise_std_is": (
                "static_ppo_action_distribution_initialization_not_a_controller"
            ),
            # 别把这两件事混成一个词:PPO 的 init_noise_std 是**动作分布**的探索噪声,
            # 由算法配方拥有;policy_observation_corruption 是**观测**噪声,由 DR 档拥有。
            # 本轮四格改的是后者,前者四格相同。
            "policy_observation_corruption_is": (
                "sensor_side_observation_noise_owned_by_the_dr_level_not_the_ppo_recipe"
            ),
        },
        "cells": [
            _cell(
                A_OBS_NOISE_OFF_CELL_ID,
                "A211",
                "desired_contact_dense",
                False,
            ),
            _cell(
                A_OBS_NOISE_ON_CELL_ID,
                "A211",
                "desired_contact_dense",
                True,
            ),
            _cell(
                C_OBS_NOISE_OFF_CELL_ID,
                "C211",
                "achieved_contact_outcome_only",
                False,
            ),
            _cell(
                C_OBS_NOISE_ON_CELL_ID,
                "C211",
                "achieved_contact_outcome_only",
                True,
            ),
        ],
    }
    _require_one_registered_difference_axis(unsigned["cells"])
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _require_one_registered_difference_axis(cells: Sequence[Any]) -> None:
    """Reject any grid where the two family cells differ outside the noise axis.

    人话:对照实验的前提是"只有本体感观测噪声的开关不同"。此处逐字段比对同族两格,
    除观测噪声包五个键之外任何差异(包括 PPO)一律拒;跨族只允许 task 语义/reward
    不同。另外硬性要求:探索包这一轴已经**不是**差异轴,所以四格 cell 上一个探索键
    都不许出现 —— 出现了就说明有人把已定死的轴又偷偷变回了变量。
    """

    if type(cells) not in (list, tuple) or len(cells) != len(CELL_IDS):
        raise FourGridContractError("four-grid must hold exactly four cells")
    for row in cells:
        stray = sorted(key for key in EXPLORATION_CELL_KEYS if key in row)
        if stray:
            raise FourGridContractError(
                "the exploration package is matched across all four cells and "
                "lives in matched_contract; cell %r still carries %r"
                % (row.get("cell_id"), stray)
            )
    for family, expected_ids in FAMILY_CELL_IDS.items():
        rows = [row for row in cells if row["task_family"] == family]
        if [row["cell_id"] for row in rows] != list(expected_ids):
            raise FourGridContractError("four-grid family cell order differs")
        first, second = rows
        varying = set(OBSERVATION_NOISE_CELL_KEYS) | {"cell_id"}
        if set(first) != set(second):
            raise FourGridContractError("four-grid family cells have different fields")
        for key in first:
            if key in varying:
                continue
            if first[key] != second[key]:
                raise FourGridContractError(
                    "%s cells differ outside the registered observation-noise "
                    "axis: %s" % (family, key)
                )
        switches = {row["policy_observation_corruption"] for row in rows}
        if switches != {False, True}:
            raise FourGridContractError(
                "%s cells must cover the observation-noise switch off and on "
                "exactly once" % family
            )


_CANONICAL_MANIFEST = _build_canonical_manifest()
CONTENT_SHA256 = _CANONICAL_MANIFEST["content_sha256"]


def validate_manifest(value: Any) -> dict:
    """Accept exactly the canonical four-cell manifest and nothing else."""

    if type(value) is not dict:
        raise FourGridContractError("four-grid manifest must be a plain dict")
    unsigned = dict(value)
    seal = unsigned.pop("content_sha256", None)
    if seal != canonical_sha256(unsigned):
        raise FourGridContractError("four-grid manifest content seal differs")
    if value != _CANONICAL_MANIFEST:
        raise FourGridContractError("four-grid manifest differs from authority")
    return copy.deepcopy(value)


def manifest() -> dict:
    """Return an isolated validated copy of the single canonical manifest."""

    return validate_manifest(copy.deepcopy(_CANONICAL_MANIFEST))


def validate_runtime_match(
    *,
    wait_contract: Any,
    formal_budgets: Mapping[str, Sequence[int]],
    action_id: Any,
    action_uid: Any,
    teacher_id: Any,
) -> dict:
    """Reject a launcher whose supposedly matched settings have drifted."""

    value = manifest()
    matched = value["matched_contract"]
    observed_budgets = {}
    if type(formal_budgets) is not dict:
        raise FourGridContractError("formal budgets must be a plain dict")
    for stage in FORMAL_STAGE_ORDER:
        budget = formal_budgets.get(stage)
        if type(budget) not in (list, tuple):
            raise FourGridContractError("formal budget %s is malformed" % stage)
        observed_budgets[stage] = list(budget)
    if set(formal_budgets) != set(FORMAL_STAGE_ORDER):
        raise FourGridContractError("formal budget stage set differs")
    if (
        wait_contract != matched["wait_contract"]
        or observed_budgets != matched["formal_budgets"]
        or action_id != matched["runtime_question_source"]["action_id"]
        or action_uid != matched["runtime_question_source"]["action_uid"]
        or teacher_id != matched["runtime_question_source"]["teacher_id"]
    ):
        raise FourGridContractError("launcher matched settings differ")
    return value


def cell_for_family(cell_id: Any, task_family: Any) -> dict:
    """Return one canonical cell only when it belongs to the requested family."""

    value = manifest()
    if type(task_family) is not str or task_family not in FAMILY_CELL_IDS:
        raise FourGridContractError("unknown four-grid task family")
    if type(cell_id) is not str or cell_id not in FAMILY_CELL_IDS[task_family]:
        raise FourGridContractError("selector is not a cell in the task family")
    matches = [row for row in value["cells"] if row["cell_id"] == cell_id]
    if len(matches) != 1 or matches[0]["task_family"] != task_family:
        raise FourGridContractError("four-grid family registry is inconsistent")
    validate_observation_noise_package(matches[0])
    # 探索包已经是全格相同的 matched 项;这里顺手复核一遍,免得有人只改了
    # matched_contract 的一半就把 manifest 重新封印。
    validate_exploration_package(value["matched_contract"]["exploration_package"])
    return copy.deepcopy(matches[0])


# 2026-08-05 重钉(第三次,先算后写):第二轴由探索包换成**本体感观测噪声开关**
# (exp §5.6.2d)。四格 cell_id 全部改名;探索包(exploration_axis / actor_init_mode /
# init_noise_std / noise_std_type / four_sigma_hard_inner_gate_applies)从每格 cells[i]
# 收回 matched_contract.exploration_package,四格统一为标准 rsl_rl 初始化 + sigma 1.0 +
# scalar;每格新增 observation_noise_axis / policy_observation_corruption /
# proprioceptive_observation_noise_channels / task_channel_observation_noise /
# dr_level_identity 五键;schema 3 -> 4、kind v3 -> v4。故 content seal 随之更新。
# 旧值 1bc1df34...1ca 只代签本次换轴之前的字节;更旧的 960fed56...c6e0 与
# 823d6d88...0709 分别只代签再往前两次改动之前的字节。
#
# 2026-08-07 重钉(第四次):Franco 裁定二把三条限位罚的核/量纲换成开源 rad 口径,
# soft_weights 的 qdes_limit / joint_limit -5 -> -10、qdes_projection -5 -> -1。
# 旧值 803144ef...445a 只代签本次改价之前的字节。
if CONTENT_SHA256 != "b31d894ea45010985f79abfacec97e723decca18d23784c6159cf017f4e5f44e":
    raise RuntimeError("formal A211/C211 Isaac four-grid manifest drifted")
