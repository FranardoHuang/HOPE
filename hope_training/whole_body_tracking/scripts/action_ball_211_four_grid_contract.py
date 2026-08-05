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


# 2026-08-05 第二轴改版(exp §5.6.2c 裁决):
# 旧的第二轴是 PPO schedule(A0/C0 fixed lr1e-4 对 A1/C1 adaptive-KL lr1e-3)。
# 在**从未观测到一次接触**的前提下,LR schedule 的差异无法被任何指标分辨,故该对照降级为
# later;第二轴换成**探索包**——零权重 bootstrap + 钉死 bias + sigma 0.1(现状) 对
# 标准 rsl_rl 初始化 + sigma 1.0(BeyondMimic / build_1 对齐)。探索包是一阶量:零权重
# actor 的初始策略是常数,梯度只能经由"探索产生了不同回报"传导。
# 因此 cell_id 也一并改名——留着 "adaptive-kl-initial-lr1e3" 而实际不跑 adaptive KL,
# 会让收据、namespace 与 barrier 布局表同时说谎。
KIND = "action_ball_211_isaac_four_grid_manifest_v3"
A_BOOTSTRAP_CELL_ID = "A0-base-safety-zero-weight-bootstrap-sigma0p1"
A_STANDARD_INIT_CELL_ID = "A1-base-safety-standard-init-sigma1p0"
C_BOOTSTRAP_CELL_ID = "C0-base-safety-zero-weight-bootstrap-sigma0p1"
C_STANDARD_INIT_CELL_ID = "C1-base-safety-standard-init-sigma1p0"
CELL_IDS = (
    A_BOOTSTRAP_CELL_ID,
    A_STANDARD_INIT_CELL_ID,
    C_BOOTSTRAP_CELL_ID,
    C_STANDARD_INIT_CELL_ID,
)
FAMILY_CELL_IDS = {
    "A211": (A_BOOTSTRAP_CELL_ID, A_STANDARD_INIT_CELL_ID),
    "C211": (C_BOOTSTRAP_CELL_ID, C_STANDARD_INIT_CELL_ID),
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
CANONICAL_SOURCE_TAPE = {
    "path": (
        "configs/action_ball_n1_measured_20260803/fresh_592835dc_take061/"
        "rematerialized_1d5d9d44/tape/"
        "immutable_n1_tape.v1.1eeccd2aa7b7.json"
    ),
    "file_sha256": (
        "1eeccd2aa7b7fbede5fb5d52356740f934664f20058b1af4237ae807655d94e6"
    ),
    "canonical_sha256": (
        "6e4a502d46df0ecfe7209b9d63327b67708d66d61fd0c7ca9803ea6d96011113"
    ),
}
CANONICAL_BASE_QUESTION_SHA256 = (
    "9b9cf4d614c0e31ead2754feee4f5ed2db81167cabdd66f7a94a7ba1ba2ad940"
)
CANONICAL_MOTION_SHA256 = (
    "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
)
CANONICAL_BASE_QUESTION = {
    "action_slot": 0,
    "action_uid": 5527597793770800,
    "ball_contact_w_m": [
        0.5163478872256125,
        -0.003197546078811897,
        1.0502659655327715,
    ],
    "base_goal_w_m": [-0.19223234, 0.28527880999999994, 1.0684000253677368],
    "base_quat_wxyz": [1.0, 0.0, 0.0, -2.710608404399295e-10],
    "base_spawn_latent_w_m": [
        -0.19223234,
        0.28527880999999994,
        1.0684000253677368,
    ],
    "base_spawn_w_m": [-0.19223234, 0.28527880999999994, 1.0684000253677368],
    "base_travel_latent_b_yaw_m": [0.0, 0.0, 0.0],
    "base_yaw_rad": -5.42121680879859e-10,
    "contact_offset_from_base_goal_b_yaw_m": [
        0.7085802273820018,
        -0.2884763556946751,
        -0.0181340598349653,
    ],
    "counter_rally_task": {
        "canonical_sha256": (
            "e1d07370111849bf852ec671e50d998de11722ad9be7cadd327259077f695a83"
        ),
        "objective_profile_sha256": (
            "7f490a9163fd5f45a2b4538cf711a03ce8d0a01288688897c4d7220d35a505ce"
        ),
        "return_direction_env_xy": [
            0.9375428134404594,
            -0.3478699081066772,
        ],
        "schema_version": 1,
        "target_baseline_speed_mps": 3.032258730715438,
    },
    "incoming_direction_b_yaw": [
        -0.9346040118217726,
        0.3467794819684754,
        0.07911594006471784,
    ],
    "incoming_speed_mps": 3.032258730715438,
    "incoming_spin_w_radps": [0.0, 0.0, 0.0],
    "incoming_velocity_w_mps": [
        -2.8339611740381896,
        1.0515251133682382,
        0.23989999999999997,
    ],
    "landing_aim_w_xy_m": [2.3, -0.6650114789147565],
    "mobility_mode": "no_move",
    "motion_sha256": CANONICAL_MOTION_SHA256,
    "physics_sha256": (
        "aa5c9085f9b48ca65b3a0ee2cbb35588a5e85a08e84dc3f2ce552d3ef4af85b7"
    ),
    "profile_sha256": (
        "ff30739979f76345e0b2fcc370eae6280207201192d68498ce88be725b5c3b39"
    ),
    "spin_direction_b_yaw": [0.0, 1.0, 0.0],
    "spin_magnitude_radps": 0.0,
    "time_to_contact_s": 1.84,
}
CANONICAL_TEACHER_PROJECTION_SHA256 = (
    "8fa2d28768c870c5dec9e867d598e1b49750f5d9ccc496d90e6980d4bc6de2ae"
)
CANONICAL_TEACHER_PROJECTION = {
    "desired_racket_site_w_m": [
        0.4992229370660459,
        0.0022324255145186878,
        1.0413862765896302,
    ],
    "desired_racket_face_center_velocity_w_mps": [
        1.4995014667510986,
        -0.258115291595459,
        0.527797520160675,
    ],
    "desired_racket_face_normal_w": [
        0.8406938314437866,
        -0.33081427216529846,
        0.42871421575546265,
    ],
    "runtime_target": {
        "geometry_source_sha256": (
            "2451e2fa1c29036d650d5ff4a1630a0d41c7ccb5730400270a2c69a6905ce29e"
        ),
        "mount_normal_sign": 1,
        "pre_swing_wait_s": 0.7123799138976297,
        "racket_command_angular_velocity_w_radps": [
            0.9587719532128958,
            0.1573498735067746,
            -0.8158117388007565,
        ],
        "racket_command_quat_wxyz": [
            0.1198626197378295,
            -0.6341276706887109,
            -0.5658849908747452,
            -0.5131171666970264,
        ],
        "racket_face_center_velocity_w_mps": [
            1.4995014667510986,
            -0.258115291595459,
            0.527797520160675,
        ],
        "racket_normal_w": [
            0.8406938314437866,
            -0.33081427216529846,
            0.42871421575546265,
        ],
        "racket_site_target_w_m": [
            0.4992229370660459,
            0.0022324255145186878,
            1.0413862765896302,
        ],
        "racket_site_velocity_w_mps": [
            1.498485602738144,
            -0.2575687000540188,
            0.526709063385955,
        ],
        "reaction_margin_s": 0.1,
        "reference_racket_angular_velocity_w_radps": [
            0.37466728687286377,
            0.479988694190979,
            -1.3601197004318237,
        ],
        "reference_racket_quat_wxyz": [
            0.3626406234330121,
            -0.7461820926331505,
            -0.42612779628720443,
            -0.36072034057028113,
        ],
        "reference_racket_site_speed_mps": 1.8900631416182667,
        "reference_t_cycle_s": 1.12,
        "reference_t_hit_s": 0.96,
        "required_racket_site_speed_mps": 1.6091063278459647,
        "scaled_t_cycle_s": 1.3155567671194324,
        "scaled_t_hit_s": 1.1276200861023704,
        "solver_residual_m": 0.00345757813192904,
        "teacher_rate": 0.8513505672981129,
        "teacher_rate_max": 1.01,
        "teacher_rate_min": 0.6,
    },
}


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


def _cell(
    cell_id: str,
    task_family: str,
    reward_semantics: str,
    actor_init_mode: str,
) -> dict:
    package = copy.deepcopy(EXPLORATION_PACKAGES[actor_init_mode])
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
    validate_exploration_package(row)
    return row


def _build_canonical_manifest() -> dict:
    matched = {
        "soft_weights": {
            # 2026-08-05 层级对齐(exp §5.6 第 7 条):-300 -> -10。post-dt 由 -6.0 降到 -0.2。
            # 原值是合法上台折扣下界 3.33209 的 180%,"打成一次再摔"净亏;外部三库与 build_1
            # 均无 death penalty 这一项。joint_actual_forbidden 改 telemetry 后本项触发面已从
            # "唯一死因"塌回"摔倒/撞桌/NaN"。
            "death_penalty": -10.0,
            "qdes_limit": -5.0,
            "qdes_projection": -5.0,
            "joint_limit": -5.0,
        },
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        # 2026-08-05 探索包上升为**注册差异轴**(exp §5.6.2c),因此 init_noise_std /
        # noise_std_type / actor_init_mode 三项从 matched_contract 移到每格 cells[i]。
        # 这里刻意不留同名键:任何仍去 matched_contract 里取 init_noise_std 的旧代码会
        # 直接 KeyError,而不是读到一个已经不再"全格相同"的数字。
        "exploration_axis_is_registered_difference": True,
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
    }
    unsigned = {
        "schema_version": 3,
        "kind": KIND,
        "formal_cell_count": 4,
        "cell_order": list(CELL_IDS),
        "matched_contract": matched,
        "registered_difference_axes": [
            "task_semantics_and_reward",
            "actor_initialization_and_exploration_sigma_cell",
        ],
        "deferred_difference_axes": [
            # exp §5.6.2c:在从未观测到一次接触前,LR schedule 的差异无法被任何指标分辨。
            "ppo_learning_rate_schedule_cell",
        ],
        "adaptive_term_disambiguation": {
            "adaptive_means": "ppo_kl_learning_rate_schedule",
            "ppo_kl_learning_rate_schedule": "disabled_fixed_learning_rate_all_cells",
            "contact_kernel_sigma_controller": "disabled_static_all_cells",
            "init_noise_std_is": (
                "static_ppo_action_distribution_initialization_not_a_controller"
            ),
        },
        "cells": [
            _cell(
                A_BOOTSTRAP_CELL_ID,
                "A211",
                "desired_contact_dense",
                ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS,
            ),
            _cell(
                A_STANDARD_INIT_CELL_ID,
                "A211",
                "desired_contact_dense",
                ACTOR_INIT_MODE_DEFAULT,
            ),
            _cell(
                C_BOOTSTRAP_CELL_ID,
                "C211",
                "achieved_contact_outcome_only",
                ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS,
            ),
            _cell(
                C_STANDARD_INIT_CELL_ID,
                "C211",
                "achieved_contact_outcome_only",
                ACTOR_INIT_MODE_DEFAULT,
            ),
        ],
    }
    _require_one_registered_difference_axis(unsigned["cells"])
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _require_one_registered_difference_axis(cells: Sequence[Any]) -> None:
    """Reject any grid where the two family cells differ outside the exploration axis.

    人话:对照实验的前提是"只有初始化与 sigma 不同"。此处逐字段比对同族两格,除探索包
    五个键之外任何差异(包括 PPO)一律拒;跨族只允许 task 语义/reward 不同。
    """

    if type(cells) not in (list, tuple) or len(cells) != len(CELL_IDS):
        raise FourGridContractError("four-grid must hold exactly four cells")
    for family, expected_ids in FAMILY_CELL_IDS.items():
        rows = [row for row in cells if row["task_family"] == family]
        if [row["cell_id"] for row in rows] != list(expected_ids):
            raise FourGridContractError("four-grid family cell order differs")
        first, second = rows
        varying = set(EXPLORATION_CELL_KEYS) | {"cell_id"}
        if set(first) != set(second):
            raise FourGridContractError("four-grid family cells have different fields")
        for key in first:
            if key in varying:
                continue
            if first[key] != second[key]:
                raise FourGridContractError(
                    "%s cells differ outside the registered exploration axis: %s"
                    % (family, key)
                )
        modes = {row["actor_init_mode"] for row in rows}
        if modes != set(ACTOR_INIT_MODES):
            raise FourGridContractError(
                "%s cells must cover both registered actor init modes exactly once"
                % family
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
    validate_exploration_package(matches[0])
    return copy.deepcopy(matches[0])


def validate_base_question(value: Any, *, motion_sha256: Any) -> dict:
    """Accept only the committed Take061 fixed counter-rally question."""

    if (
        type(motion_sha256) is not str
        or motion_sha256 != CANONICAL_MOTION_SHA256
        or type(value) is not dict
        or value != CANONICAL_BASE_QUESTION
        or canonical_sha256(value) != CANONICAL_BASE_QUESTION_SHA256
    ):
        raise FourGridContractError(
            "base question differs from the canonical Take061 fixed row"
        )
    return {
        "source_tape": copy.deepcopy(CANONICAL_SOURCE_TAPE),
        "question_sha256": CANONICAL_BASE_QUESTION_SHA256,
        "question": copy.deepcopy(CANONICAL_BASE_QUESTION),
    }


def validate_teacher_projection(value: Any, *, target_variant: Any) -> dict:
    """Bind A/C target variants to one exact Take061 teacher projection."""

    expected_variant = {
        "A211": "current_lm",
        "C211": "outcome_dense_only",
    }
    if (
        type(target_variant) is not tuple
        or len(target_variant) != 2
        or target_variant[0] not in expected_variant
        or target_variant[1] != expected_variant[target_variant[0]]
        or type(value) is not dict
    ):
        raise FourGridContractError("target variant is not a formal A/C grid variant")
    projection = {
        key: copy.deepcopy(value.get(key))
        for key in (
            "desired_racket_site_w_m",
            "desired_racket_face_center_velocity_w_mps",
            "desired_racket_face_normal_w",
            "runtime_target",
        )
    }
    if (
        projection != CANONICAL_TEACHER_PROJECTION
        or canonical_sha256(projection) != CANONICAL_TEACHER_PROJECTION_SHA256
    ):
        raise FourGridContractError(
            "target variant teacher projection differs from canonical Take061"
        )
    return {
        "target_variant": target_variant[1],
        "teacher_projection_sha256": CANONICAL_TEACHER_PROJECTION_SHA256,
        "teacher_projection": copy.deepcopy(CANONICAL_TEACHER_PROJECTION),
    }


if canonical_sha256(CANONICAL_BASE_QUESTION) != CANONICAL_BASE_QUESTION_SHA256:
    raise RuntimeError("canonical Take061 base-question bytes drifted")
if (
    canonical_sha256(CANONICAL_TEACHER_PROJECTION)
    != CANONICAL_TEACHER_PROJECTION_SHA256
):
    raise RuntimeError("canonical Take061 teacher projection drifted")
# 2026-08-05 重钉(第二次,先算后写):第二轴由 PPO schedule 换成探索包(exp §5.6.2c),
# 四格 cell_id 全部改名、init_noise_std/noise_std_type 从 matched_contract 下放到每格、
# 新增 actor_init_mode 与 four_sigma_hard_inner_gate_applies、schema 2 -> 3、kind v2 -> v3、
# 四格 PPO 统一为 fixed lr1e-4。故 content seal 随之更新。
# 旧值 960fed56...c6e0 只代签本次改名与下放之前的字节;更旧的 823d6d88...0709 只代签
# 2026-08-05 层级对齐之前的字节。
if CONTENT_SHA256 != "1bc1df349b3f66316c81f5b0b2a6a79b3b84735c4a489c0e910943fc751ab1ca":
    raise RuntimeError("formal A211/C211 Isaac four-grid manifest drifted")
