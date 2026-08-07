"""Schema and runtime ledger for the ActionBall 4096x5 pre-long marker.

The JSON formatter remains free of Torch and Isaac imports.  The optional
runtime ledger imports Torch lazily, snapshots true eligibility immediately
before ``env.step``, and consumes ``RewardManager._step_reward`` immediately
after that same step.  Eligibility is never inferred from whether a reward
happened to be non-zero or from how often RewardManager evaluated a term.
The same frozen ``task_valid`` mask partitions active mimic income into hidden
ready/wait and task-valid swing accounting without changing reward evaluation.

The split is deliberate: host tooling can consume the schema without importing
the simulator, while production has one implementation of the temporal and
reward-cache contracts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


PRELONG_SEMANTICS_EVENT = "hope_action_ball_4096x5_prelong_semantics_update"
PRELONG_SEMANTICS_SCHEMA_VERSION = 3
PRELONG_SEMANTICS_MARKER_PREFIX = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_SEMANTICS_UPDATE_JSON="
)
PRELONG_SEMANTICS_ENABLE_ENV = "HOPE_ACTION_BALL_4096X5_PRELONG_SEMANTICS"
PRELONG_SEMANTICS_RECIPE_SHA_ENV = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_REWARD_RECIPE_SHA256"
)

PRELONG_NUM_ENVS = 4096
PRELONG_ROLLOUT_STEPS = 24
PRELONG_ROLLOUT_SAMPLES = PRELONG_NUM_ENVS * PRELONG_ROLLOUT_STEPS
PRELONG_POLICY_DT_S = 0.02

PRELONG_PROFILE_A211 = "A211"
PRELONG_PROFILE_C211 = "C211"
PRELONG_PROFILES: Tuple[str, ...] = (
    PRELONG_PROFILE_A211,
    PRELONG_PROFILE_C211,
)

PRELONG_REWARD_GROUPS: Tuple[str, ...] = (
    "balance",
    "mimic",
    "strike",
    "target",
    "outcome",
)

# These names are the narrow producer ABI shared by the runtime ledger and its
# once-per-update runner consume.  Exact strike timing and closed-swing hit rate
# are intentionally distinct denominators: an exact tick in PPO update N can
# close in update N+1, so comparing their per-update counts would be false.
TASK_INVALID_OBSERVED_COUNTER = "prelong_task_invalid_observed_sample_count"
TASK_INVALID_REWARD_SUM_COUNTER = "prelong_task_invalid_task_reward_weighted_sum"
TASK_INVALID_REWARD_ELIGIBLE_COUNTER = (
    "prelong_task_invalid_task_reward_eligible_denominator"
)
READY_MIMIC_REWARD_SUM_COUNTER = (
    "prelong_task_invalid_ready_mimic_reward_weighted_sum"
)
READY_MIMIC_ELIGIBLE_COUNTER = (
    "prelong_task_invalid_ready_mimic_eligible_denominator"
)
SWING_MIMIC_REWARD_SUM_COUNTER = (
    "prelong_task_valid_swing_mimic_reward_weighted_sum"
)
SWING_MIMIC_ELIGIBLE_COUNTER = (
    "prelong_task_valid_swing_mimic_eligible_denominator"
)
EXACT_STRIKE_TIMING_COUNTER = "prelong_exact_strike_timing_count"
ELIGIBLE_CLOSED_SWING_COUNTER = "prelong_eligible_closed_swing_count"
ACTUAL_CONTACT_COUNTER = "prelong_actual_contact_count"
ACHIEVED_FLIGHT_COUNTER = "prelong_achieved_outgoing_flight_count"
UNKNOWN_ATTRIBUTION_COUNTER = "prelong_unknown_attribution_count"


# Exact non-zero compositions for the two admitted 4096x5 profiles.  Values are
# RewardManager weights, not per-step income; the runtime income is the manager
# cache (raw * weight) multiplied exactly once by policy dt.
_COMMON_SCIENTIFIC_TERM_WEIGHTS = {
    "balance": {
        # 2026-08-05 层级对齐,详见 exp §5.6 偏离记录第 5/6 条。这两个值此前从未进过
        # exp 的层级账,实测量级压过主层级:
        #   upright_exp 1.0 -> 每步无条件 +0.02,500 步 gamma=.99 折扣 +1.9869,
        #                      = task-valid mimic 1.77331 的 112% -> "站着不动"胜过"学动作";
        #   hit_unstable_support -10 -> 窗内单脚每步 -0.2,wide 窗 11 步最坏 -2.2,
        #                      > accepted window +1.85151 -> "进窗但重心转移"劣于"不挥拍",
        #                      而重心转移是击球必然发生的事。
        # 对齐后 upright_exp 折扣 +0.4967(mimic 的 28%)、hit_unstable 最坏 -0.22(window 的 12%)。
        "upright_exp": 0.25,
        "hit_unstable_support": -1.0,
        "foot_slip_sq": -0.1,
        "foot_velocity": -0.05,
        "foot_soft_landing": -0.003,
        "joint_torques": -3.0e-5,
        "undesired_contacts": -0.1,
        "base_ang_vel_xy": -0.05,
        "base_lin_vel_z": -0.5,
        "joint_vel": -1.0e-4,
        # 2026-08-08:封顶版下岗(σ=1.0 下 100% 时间在 clamp 上、零梯度),换回上游那条
        # 无封顶的 action_rate_l2 −0.1(BeyondMimic / mjlab / unitree-mimic 三家同值)。
        "action_rate_l2": -0.1,
    },
    "mimic": {
        "motion_global_anchor_ori": 0.075,
        "motion_body_pos": 0.15,
        "motion_body_ori": 0.15,
        "motion_body_lin_vel": 0.15,
        "motion_body_ang_vel": 0.15,
        "motion_racket_position": 0.20,
        "motion_racket_velocity": 0.20,
        "motion_racket_normal": 0.20,
        "motion_racket_long_axis": 0.10,
    },
}

PRELONG_SCIENTIFIC_TERM_WEIGHTS = {
    PRELONG_PROFILE_A211: {
        **_COMMON_SCIENTIFIC_TERM_WEIGHTS,
        "strike": {
            "strike_capture_bonus": 25.0,
        },
        "target": {
            "racket_progress": 10.0,
            "racket_position_coarse": 11.5,
            "racket_velocity_coarse": 11.5,
            "racket_normal_coarse": 5.75,
            "racket_position": 4.6,
            "racket_velocity": 0.575,
            "racket_normal": 0.575,
            "racket_position_precision": 0.575,
            "racket_velocity_precision": 0.2875,
            "racket_normal_precision": 0.575,
        },
        "outcome": {
            "virtual_pass_net": 20.0,
            "virtual_landing_dense": 20.0,
            "virtual_landing": 700.0,
        },
    },
    PRELONG_PROFILE_C211: {
        **_COMMON_SCIENTIFIC_TERM_WEIGHTS,
        "strike": {
            "c225_strike_ball_paddle_center_proximity": 240.0,
        },
        "target": {},
        "outcome": {
            "virtual_landing": 700.0,
        },
    },
}

PRELONG_EXCLUDED_SAFETY_TERM_WEIGHTS = {
    # 2026-08-05:-300(post-dt -6.0)是合法上台折扣下界 3.33209 的 180%,"打成一次再摔"净亏,
    # 正是 §5.1 要防的倒置。外部三库与 build_1 均无 death penalty 这一项(终止的代价就是失去
    # 未来收入),其最大单步罚 post-dt ≈ -0.2。取 -10 -> post-dt -0.2 = 上台下界的 6%。
    # 另:joint_actual_forbidden 已改 terminate=False,本项触发面从"唯一死因"塌回
    # "摔倒/撞桌/NaN"。详见 exp §5.6 第 7 条。
    # 2026-08-07 裁定二:两条 barrier 改开源 rad 口径 -> -10(旧 -5 是归一 [0,1] 口径,不可比)。
    "death_penalty": -10.0,
    "joint_limit": -10.0,
    "qdes_limit_barrier": -10.0,
    # The admitted scientific dose is carried by params.objective_weight;
    # RewardManager deliberately retains a fixed -1.0 exposure weight.
    "qdes_projection_penalty": -1.0,
}

PRELONG_EXCLUDED_PROBE_TERM_WEIGHTS = {
    "base_decel_activation_probe": 1.0,
    "qdes_limit_barrier_probe": 1.0,
    "actual_joint_limit_barrier_probe": 1.0,
}

PRELONG_REQUIRED_TERM_PARAMS = {
    # 2026-08-07 裁定二:核换成开源线性尾巴(rad 口径)后重算的采纳剂量,-5.0 -> -1.0。
    "qdes_projection_penalty": {
        "objective_weight": -1.0,
    },
}

_EFFECTIVE_RECIPE_TOOL_CACHE = None

_COMMON_EXPECTED_CALLABLE_NAMES = {
    "upright_exp": "upright_exp",
    "hit_unstable_support": "hit_unstable_support",
    "foot_slip_sq": "foot_slip_sq",
    "foot_velocity": "foot_velocity",
    "foot_soft_landing": "foot_soft_landing",
    "joint_torques": "joint_torques_l2",
    "undesired_contacts": "undesired_contacts",
    "base_ang_vel_xy": "ang_vel_xy_l2",
    "base_lin_vel_z": "lin_vel_z_l2",
    "joint_vel": "joint_vel_l2",
    # 上游 isaaclab 的 callable,不是我们自己的 —— 这是"形状照开源"的最强形式:
    # 我们连实现都不写第二份(见 mdp/__init__.py 的 `from isaaclab.envs.mdp import *`)。
    "action_rate_l2": "action_rate_l2",
    "motion_global_anchor_ori": "motion_global_anchor_orientation_error_exp",
    "motion_body_pos": "motion_body_pos_swing_only",
    "motion_body_ori": "motion_body_ori_swing_only",
    "motion_body_lin_vel": "motion_global_body_linear_velocity_error_exp",
    "motion_body_ang_vel": "motion_global_body_angular_velocity_error_exp",
    "motion_racket_position": "motion_racket_position_tracking_cauchy",
    "motion_racket_velocity": "motion_racket_velocity_tracking_cauchy",
    "motion_racket_normal": "motion_racket_normal_tracking_cauchy",
    "motion_racket_long_axis": "motion_racket_long_axis_tracking_cauchy",
    "death_penalty": "action_ball_safety_terminated",
    "joint_limit": "actual_joint_limit_barrier_v2",
    "qdes_limit_barrier": "qdes_limit_barrier_v2",
    "qdes_projection_penalty": "qdes_projection_penalty",
    "base_decel_activation_probe": "base_decel_activation_probe",
    "qdes_limit_barrier_probe": "qdes_limit_barrier_v2_probe",
    "actual_joint_limit_barrier_probe": "actual_joint_limit_barrier_v2_probe",
}

PRELONG_EXPECTED_CALLABLE_NAMES = {
    PRELONG_PROFILE_A211: {
        **_COMMON_EXPECTED_CALLABLE_NAMES,
        "strike_capture_bonus": "strike_capture_bonus",
        "racket_progress": "racket_progress",
        "racket_position_coarse": "racket_position_coarse_tracking_cauchy",
        "racket_velocity_coarse": "racket_velocity_coarse_tracking_cauchy",
        "racket_normal_coarse": "racket_normal_coarse_tracking_cauchy",
        "racket_position": "racket_position_tracking_exp",
        "racket_velocity": "racket_velocity_tracking_exp",
        "racket_normal": "racket_normal_tracking_exp",
        "racket_position_precision": "racket_position_tracking_exp",
        "racket_velocity_precision": "racket_velocity_tracking_exp",
        "racket_normal_precision": "racket_normal_tracking_exp",
        "virtual_pass_net": "virtual_pass_net",
        "virtual_landing_dense": "virtual_landing_dense_actual_contact",
        "virtual_landing": "virtual_landing",
    },
    PRELONG_PROFILE_C211: {
        **_COMMON_EXPECTED_CALLABLE_NAMES,
        "c225_strike_ball_paddle_center_proximity": (
            "c225_strike_ball_paddle_center_proximity"
        ),
        "virtual_landing": "c225_landing_outcome_actual_contact",
    },
}


def reward_group_sum_counter(group: str) -> str:
    """Return the canonical weighted-income counter for one fixed group."""

    if group not in PRELONG_REWARD_GROUPS:
        raise PrelongSemanticProducerError(
            "unknown pre-long reward group %r" % (group,)
        )
    return "prelong_%s_reward_weighted_sum" % group


def reward_group_eligible_counter(group: str) -> str:
    """Return the canonical true-eligibility counter for one fixed group."""

    if group not in PRELONG_REWARD_GROUPS:
        raise PrelongSemanticProducerError(
            "unknown pre-long reward group %r" % (group,)
        )
    return "prelong_%s_reward_eligible_denominator" % group


PRELONG_REWARD_GROUP_ELIGIBILITY_SEMANTICS = {
    "balance": (
        "union of samples eligible for registered physical-balance rewards; "
        "each simulator sample is counted at most once"
    ),
    "mimic": (
        "union of samples eligible for registered reference-imitation rewards; "
        "each simulator sample is counted at most once"
    ),
    "strike": (
        "task-valid exact strike-opportunity events eligible for registered "
        "strike rewards; each event is counted once"
    ),
    "target": (
        "task-valid contact-target opportunities eligible for registered target "
        "rewards; each event is counted once"
    ),
    "outcome": (
        "task-valid selected-rubber actual-contact outcome opportunities; downstream "
        "net/landing failures remain in the denominator with zero income, and each "
        "event is counted once"
    ),
}


class PrelongSemanticProducerError(ValueError):
    """Raised before emission when an update snapshot violates the schema."""


class PrelongSemanticLedgerError(RuntimeError):
    """Raised when live pre-long telemetry cannot preserve its contracts."""


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrelongSemanticLedgerError(
            "%s must be exactly 64 lowercase hexadecimal characters" % name
        )
    return value


def parse_prelong_runtime_request(
    environ: Mapping[str, str], *, reward_ppo_economy_requested: bool
) -> Any:
    """Return the sealed recipe SHA when the dedicated scale-only producer is requested.

    The pre-long producer deliberately does not inherit the older reward/PPO
    economy switch.  This keeps legacy economy probes usable while requiring
    the A211/C211 scale launcher to opt in and bind the fully normalized reward
    recipe that ``train.py`` already preregistered.
    """

    if not isinstance(environ, Mapping):
        raise PrelongSemanticLedgerError("pre-long environment must be a mapping")
    raw = environ.get(PRELONG_SEMANTICS_ENABLE_ENV)
    recipe = environ.get(PRELONG_SEMANTICS_RECIPE_SHA_ENV)
    if raw is None or raw == "0":
        if recipe is not None:
            raise PrelongSemanticLedgerError(
                "%s is present without %s=1"
                % (PRELONG_SEMANTICS_RECIPE_SHA_ENV, PRELONG_SEMANTICS_ENABLE_ENV)
            )
        return None
    if raw != "1":
        raise PrelongSemanticLedgerError(
            "%s must be exactly 0, 1, or absent" % PRELONG_SEMANTICS_ENABLE_ENV
        )
    if type(reward_ppo_economy_requested) is not bool:
        raise PrelongSemanticLedgerError(
            "reward_ppo_economy_requested must be a plain boolean"
        )
    if not reward_ppo_economy_requested:
        raise PrelongSemanticLedgerError(
            "%s=1 requires HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_GATE=1"
            % PRELONG_SEMANTICS_ENABLE_ENV
        )
    if recipe is None:
        raise PrelongSemanticLedgerError(
            "%s=1 requires %s"
            % (PRELONG_SEMANTICS_ENABLE_ENV, PRELONG_SEMANTICS_RECIPE_SHA_ENV)
        )
    return _require_sha256(recipe, name=PRELONG_SEMANTICS_RECIPE_SHA_ENV)


def _effective_recipe_tools():
    # Reuse the same complete parameter/callable normalizer that creates the
    # preregistered effective-reward receipt.  Import lazily so the pure marker
    # formatter remains dependency-light.
    global _EFFECTIVE_RECIPE_TOOL_CACHE
    if _EFFECTIVE_RECIPE_TOOL_CACHE is not None:
        return _EFFECTIVE_RECIPE_TOOL_CACHE
    try:
        from whole_body_tracking.utils.effective_reward_recipe import (
            EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION,
            _normalized_term,
            canonical_effective_reward_recipe_json,
        )
    except ImportError as package_exc:
        # Host schema tests load this file directly rather than installing the
        # package.  Resolve the adjacent dependency-light authority by path in
        # that one case; production resolves the package import above.
        path = Path(__file__).with_name("effective_reward_recipe.py")
        spec = importlib.util.spec_from_file_location(
            "_prelong_effective_reward_recipe", path
        )
        if spec is None or spec.loader is None:
            raise PrelongSemanticLedgerError(
                "pre-long semantics cannot load the effective-reward normalizer"
            ) from package_exc
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION = (
                module.EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION
            )
            _normalized_term = module._normalized_term
            canonical_effective_reward_recipe_json = (
                module.canonical_effective_reward_recipe_json
            )
        except (AttributeError, ImportError, OSError) as exc:
            raise PrelongSemanticLedgerError(
                "pre-long semantics cannot load the effective-reward normalizer"
            ) from exc
    _EFFECTIVE_RECIPE_TOOL_CACHE = (
        EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION,
        _normalized_term,
        canonical_effective_reward_recipe_json,
    )
    return _EFFECTIVE_RECIPE_TOOL_CACHE


def _normalize_runtime_term(name: str, cfg):
    _schema_version, normalize_term, _canonical_json = _effective_recipe_tools()
    try:
        return normalize_term(name, cfg)
    except Exception as exc:
        raise PrelongSemanticLedgerError(
            "reward term %r cannot be normalized exactly" % name
        ) from exc


def _normalized_runtime_reward_recipe(manager, names: Tuple[str, ...]):
    schema_version, _normalize_term, canonical_json = _effective_recipe_tools()
    get_term_cfg = getattr(manager, "get_term_cfg", None)
    if not callable(get_term_cfg):
        raise PrelongSemanticLedgerError(
            "pre-long semantics require RewardManager.get_term_cfg(name)"
        )
    normalized_by_name = {}
    active_rows = []
    for name in names:
        normalized = _normalize_runtime_term(name, get_term_cfg(name))
        normalized_by_name[name] = normalized
        if normalized is not None:
            active_rows.append(normalized)
    active_rows.sort(key=lambda row: row["name"])
    recipe = {"schema_version": schema_version, "terms": active_rows}
    try:
        canonical = canonical_json(recipe)
    except Exception as exc:
        raise PrelongSemanticLedgerError(
            "pre-long runtime reward recipe is not canonical"
        ) from exc
    receipt = {
        "schema_version": schema_version,
        "terms": active_rows,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    return receipt, normalized_by_name


def _without_runtime_entity_resolution(value):
    """Remove only manager-resolved SceneEntityCfg id caches before comparison.

    The preregistered recipe seals the original ids and names.  Isaac managers
    deterministically replace the ``*_ids`` slices with resolved integer lists
    after scene construction; names, selector mode, and every other parameter
    remain semantic configuration and must still compare exactly.
    """

    if isinstance(value, Mapping):
        config_type = value.get("__config_type__")
        fields = value.get("fields")
        if (
            type(config_type) is str
            and config_type.endswith(".SceneEntityCfg")
            and isinstance(fields, Mapping)
            and set(value) == {"__config_type__", "fields"}
        ):
            return {
                "__config_type__": config_type,
                "fields": {
                    key: _without_runtime_entity_resolution(item)
                    for key, item in fields.items()
                    if not key.endswith("_ids")
                },
            }
        return {
            key: _without_runtime_entity_resolution(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_without_runtime_entity_resolution(item) for item in value]
    return value


def _validate_preregistered_reward_recipe(
    expected: Any, observed: Mapping[str, Any]
) -> dict:
    _schema_version, _normalize_term, canonical_json = _effective_recipe_tools()
    if not isinstance(expected, Mapping) or set(expected) != {
        "schema_version",
        "terms",
        "sha256",
    }:
        raise PrelongSemanticLedgerError(
            "pre-long preregistered effective reward recipe is malformed"
        )
    expected_recipe = {
        "schema_version": expected.get("schema_version"),
        "terms": expected.get("terms"),
    }
    try:
        expected_sha = hashlib.sha256(
            canonical_json(expected_recipe).encode("utf-8")
        ).hexdigest()
    except Exception as exc:
        raise PrelongSemanticLedgerError(
            "pre-long preregistered effective reward recipe is not canonical"
        ) from exc
    if (
        _require_sha256(
            expected.get("sha256"),
            name="preregistered effective reward recipe SHA-256",
        )
        != expected_sha
    ):
        raise PrelongSemanticLedgerError(
            "pre-long preregistered effective reward recipe SHA-256 is invalid"
        )
    comparable_expected = _without_runtime_entity_resolution(expected_recipe)
    comparable_observed = _without_runtime_entity_resolution(
        {
            "schema_version": observed.get("schema_version"),
            "terms": observed.get("terms"),
        }
    )
    if comparable_expected != comparable_observed:
        raise PrelongSemanticLedgerError(
            "pre-long runtime reward terms differ from the complete preregistered recipe"
        )
    return {
        "schema_version": expected_recipe["schema_version"],
        "terms": list(expected_recipe["terms"]),
        "sha256": expected_sha,
    }


# [已删除 2026-08-06 过期结构清理] prelong_runtime_effective_reward_recipe_sha256(19 行)。
# 全仓零引用,而它的 docstring 写着 "for launcher binding" —— 没有任何 launcher 绑它,
# 这句话本身就是过期描述。函数体前 17 行与下面 _receipt 版逐字相同,只是末尾取 ["sha256"];
# 现役生产路径根本不走这两个模块级入口,而是类内的 _normalized_runtime_reward_recipe(:1725)。
# 也就是说"active_terms 怎么校验"这条规矩曾经存三份、跑一份。
def prelong_runtime_effective_reward_recipe_receipt(manager) -> dict:
    """Return the complete normalized runtime receipt for host/runtime tests."""

    raw_names = getattr(manager, "active_terms", None)
    if not isinstance(raw_names, (list, tuple)):
        raise PrelongSemanticLedgerError(
            "pre-long semantics require ordered RewardManager active_terms"
        )
    names = tuple(raw_names)
    if (
        not names
        or any(type(name) is not str or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise PrelongSemanticLedgerError(
            "pre-long RewardManager active_terms must be unique names"
        )
    receipt, _rows = _normalized_runtime_reward_recipe(manager, names)
    return {
        "schema_version": receipt["schema_version"],
        "terms": list(receipt["terms"]),
        "sha256": receipt["sha256"],
    }


def prelong_group_term_weights(
    profile: str,
) -> Dict[str, Dict[str, float]]:
    """Return a defensive copy of one profile's exact scientific taxonomy."""

    if profile not in PRELONG_PROFILES:
        raise PrelongSemanticLedgerError(
            "unknown pre-long reward profile %r" % (profile,)
        )
    return {
        group: dict(PRELONG_SCIENTIFIC_TERM_WEIGHTS[profile][group])
        for group in PRELONG_REWARD_GROUPS
    }


def expected_prelong_nonzero_reward_weights(profile: str) -> Dict[str, float]:
    """Return every admitted non-zero term, including excluded safety/probes."""

    result: Dict[str, float] = {}
    for terms in prelong_group_term_weights(profile).values():
        overlap = result.keys() & terms.keys()
        if overlap:
            raise PrelongSemanticLedgerError(
                "pre-long scientific taxonomy duplicates %r" % sorted(overlap)
            )
        result.update(terms)
    for excluded in (
        PRELONG_EXCLUDED_SAFETY_TERM_WEIGHTS,
        PRELONG_EXCLUDED_PROBE_TERM_WEIGHTS,
    ):
        overlap = result.keys() & excluded.keys()
        if overlap:
            raise PrelongSemanticLedgerError(
                "pre-long excluded taxonomy duplicates %r" % sorted(overlap)
            )
        result.update(excluded)
    return result


def expected_prelong_callable_names(profile: str) -> Dict[str, str]:
    """Return the exact runtime callable basename for every admitted term."""

    if profile not in PRELONG_PROFILES:
        raise PrelongSemanticLedgerError(
            "unknown pre-long reward profile %r" % (profile,)
        )
    result = dict(PRELONG_EXPECTED_CALLABLE_NAMES[profile])
    expected_names = set(expected_prelong_nonzero_reward_weights(profile))
    if set(result) != expected_names:
        raise PrelongSemanticLedgerError(
            "pre-long %s callable taxonomy differs from its non-zero term set" % profile
        )
    return result


def classify_prelong_reward_profile(reward_weights: Mapping[str, Any]) -> str:
    """Fail closed unless ``reward_weights`` is exactly admitted A211 or C211.

    Zero-weight RewardManager declarations are intentionally omitted by the
    caller.  Unknown non-zero terms, missing terms, and weight drift are all
    construction errors; this prevents an unclassified term from disappearing
    into the semantic marker's aggregate income.
    """

    if not isinstance(reward_weights, Mapping):
        raise PrelongSemanticLedgerError("pre-long reward weights must be a mapping")
    normalized: Dict[str, float] = {}
    for name, value in reward_weights.items():
        if type(name) is not str or not name:
            raise PrelongSemanticLedgerError(
                "pre-long reward term names must be non-empty strings"
            )
        if type(value) not in (int, float) or isinstance(value, bool):
            raise PrelongSemanticLedgerError(
                "pre-long reward weight for %r must be finite" % name
            )
        weight = float(value)
        if not math.isfinite(weight) or weight == 0.0:
            raise PrelongSemanticLedgerError(
                "pre-long classified reward weight for %r must be finite and non-zero"
                % name
            )
        normalized[name] = weight

    for profile in PRELONG_PROFILES:
        if normalized == expected_prelong_nonzero_reward_weights(profile):
            return profile

    preferred = (
        PRELONG_PROFILE_C211
        if "c225_strike_ball_paddle_center_proximity" in normalized
        else PRELONG_PROFILE_A211
    )
    expected = expected_prelong_nonzero_reward_weights(preferred)
    missing = sorted(expected.keys() - normalized.keys())
    unknown = sorted(normalized.keys() - expected.keys())
    drift = sorted(
        name
        for name in expected.keys() & normalized.keys()
        if expected[name] != normalized[name]
    )
    raise PrelongSemanticLedgerError(
        "pre-long %s non-zero reward composition differs: missing=%r unknown=%r "
        "weight_drift=%r" % (preferred, missing, unknown, drift)
    )


def required_prelong_counter_names() -> Tuple[str, ...]:
    """Return every scalar the simulator-side ledgers must supply."""

    names = [
        TASK_INVALID_OBSERVED_COUNTER,
        TASK_INVALID_REWARD_SUM_COUNTER,
        TASK_INVALID_REWARD_ELIGIBLE_COUNTER,
        READY_MIMIC_REWARD_SUM_COUNTER,
        READY_MIMIC_ELIGIBLE_COUNTER,
        SWING_MIMIC_REWARD_SUM_COUNTER,
        SWING_MIMIC_ELIGIBLE_COUNTER,
        EXACT_STRIKE_TIMING_COUNTER,
        ELIGIBLE_CLOSED_SWING_COUNTER,
        ACTUAL_CONTACT_COUNTER,
        ACHIEVED_FLIGHT_COUNTER,
        UNKNOWN_ATTRIBUTION_COUNTER,
    ]
    for group in PRELONG_REWARD_GROUPS:
        names.append(reward_group_sum_counter(group))
        names.append(reward_group_eligible_counter(group))
    return tuple(names)


def _counter(counters: Mapping[str, Any], name: str) -> int:
    if name not in counters:
        raise PrelongSemanticProducerError(
            "pre-long semantic ledger is missing %s" % name
        )
    value = counters[name]
    if type(value) is not int or value < 0:
        raise PrelongSemanticProducerError(
            "%s must be a nonnegative plain integer" % name
        )
    return value


def _finite_number(counters: Mapping[str, Any], name: str) -> float:
    if name not in counters:
        raise PrelongSemanticProducerError(
            "pre-long semantic ledger is missing %s" % name
        )
    value = counters[name]
    if type(value) not in (int, float) or isinstance(value, bool):
        raise PrelongSemanticProducerError("%s must be a finite plain number" % name)
    result = float(value)
    if not math.isfinite(result):
        raise PrelongSemanticProducerError("%s must be finite" % name)
    return result


def _bounded_count(value: int, *, name: str) -> None:
    if value > PRELONG_ROLLOUT_SAMPLES:
        raise PrelongSemanticProducerError(
            "%s exceeds the fixed %d-sample PPO window"
            % (name, PRELONG_ROLLOUT_SAMPLES)
        )


def build_prelong_semantics_update(
    *,
    ppo_update: int,
    counters: Mapping[str, Any],
    profile: Any = None,
    bridge_telemetry: Any = None,
) -> Dict[str, Any]:
    """Build one validated semantic record from one transactional snapshot.

    Extra counters are intentionally allowed for producer composition.  Every
    counter used by this schema is nevertheless mandatory and type checked.
    """

    if type(ppo_update) is not int or ppo_update < 0:
        raise PrelongSemanticProducerError(
            "ppo_update must be a nonnegative plain integer"
        )
    if not isinstance(counters, Mapping):
        raise PrelongSemanticProducerError("counters must be a mapping")

    invalid_samples = _counter(counters, TASK_INVALID_OBSERVED_COUNTER)
    invalid_income = _finite_number(counters, TASK_INVALID_REWARD_SUM_COUNTER)
    invalid_denominator = _counter(counters, TASK_INVALID_REWARD_ELIGIBLE_COUNTER)
    ready_mimic_income = _finite_number(counters, READY_MIMIC_REWARD_SUM_COUNTER)
    ready_mimic_denominator = _counter(counters, READY_MIMIC_ELIGIBLE_COUNTER)
    swing_mimic_income = _finite_number(counters, SWING_MIMIC_REWARD_SUM_COUNTER)
    swing_mimic_denominator = _counter(counters, SWING_MIMIC_ELIGIBLE_COUNTER)
    exact_strike_ticks = _counter(counters, EXACT_STRIKE_TIMING_COUNTER)
    closed = _counter(counters, ELIGIBLE_CLOSED_SWING_COUNTER)
    contacts = _counter(counters, ACTUAL_CONTACT_COUNTER)
    outcome_opportunities = _counter(counters, ACHIEVED_FLIGHT_COUNTER)
    unknown = _counter(counters, UNKNOWN_ATTRIBUTION_COUNTER)

    for name, value in (
        (TASK_INVALID_OBSERVED_COUNTER, invalid_samples),
        (READY_MIMIC_ELIGIBLE_COUNTER, ready_mimic_denominator),
        (SWING_MIMIC_ELIGIBLE_COUNTER, swing_mimic_denominator),
        (EXACT_STRIKE_TIMING_COUNTER, exact_strike_ticks),
        (ELIGIBLE_CLOSED_SWING_COUNTER, closed),
        (ACTUAL_CONTACT_COUNTER, contacts),
        (ACHIEVED_FLIGHT_COUNTER, outcome_opportunities),
    ):
        _bounded_count(value, name=name)

    if invalid_income != 0.0 or invalid_denominator != 0:
        raise PrelongSemanticProducerError(
            "task_valid=0 must have exactly zero task reward income and eligibility"
        )
    if ready_mimic_denominator != invalid_samples:
        raise PrelongSemanticProducerError(
            "task-invalid ready-mimic denominator must equal observed task-invalid samples"
        )
    for label, income, denominator in (
        ("task-invalid ready-mimic", ready_mimic_income, ready_mimic_denominator),
        ("task-valid swing-mimic", swing_mimic_income, swing_mimic_denominator),
    ):
        if denominator == 0 and income != 0.0:
            raise PrelongSemanticProducerError(
                "%s income is nonzero with a zero eligibility denominator" % label
            )
    if contacts > closed:
        raise PrelongSemanticProducerError(
            "actual contacts must be <= eligible closed swings"
        )
    if unknown != 0:
        raise PrelongSemanticProducerError(
            "unknown attribution must be explicitly present and zero"
        )

    reward_groups = []
    group_denominators = {}
    group_incomes = {}
    for group in PRELONG_REWARD_GROUPS:
        income_name = reward_group_sum_counter(group)
        eligible_name = reward_group_eligible_counter(group)
        income = _finite_number(counters, income_name)
        eligible = _counter(counters, eligible_name)
        _bounded_count(eligible, name=eligible_name)
        if eligible == 0 and income != 0.0:
            raise PrelongSemanticProducerError(
                "%s reward income is nonzero with a zero true-eligibility denominator"
                % group
            )
        group_denominators[group] = eligible
        group_incomes[group] = income
        reward_groups.append(
            {
                "group": group,
                "weighted_sum": income,
                "eligible_denominator": eligible,
                "eligibility_semantics": (
                    PRELONG_REWARD_GROUP_ELIGIBILITY_SEMANTICS[group]
                ),
            }
        )

    if group_denominators["strike"] > exact_strike_ticks:
        raise PrelongSemanticProducerError(
            "strike reward-group denominator cannot exceed exact-strike timing count"
        )
    if group_denominators["outcome"] != outcome_opportunities:
        raise PrelongSemanticProducerError(
            "outcome reward-group denominator must equal the outcome-opportunity count"
        )
    if (
        ready_mimic_denominator + swing_mimic_denominator
        != group_denominators["mimic"]
    ):
        raise PrelongSemanticProducerError(
            "ready/swing mimic denominators must exhaust the aggregate mimic denominator"
        )
    if not math.isclose(
        ready_mimic_income + swing_mimic_income,
        group_incomes["mimic"],
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise PrelongSemanticProducerError(
            "ready/swing mimic income must exhaust the aggregate mimic income"
        )
    if profile is None:
        # Compatibility for dependency-light callers that predate the explicit
        # profile argument.  The live ledger always supplies its recipe-classified
        # profile, so a formal marker never relies on this structural fallback.
        profile = (
            PRELONG_PROFILE_A211
            if group_denominators["target"] > 0
            else PRELONG_PROFILE_C211
        )
    if type(profile) is not str or profile not in PRELONG_PROFILES:
        raise PrelongSemanticProducerError(
            "pre-long semantic profile must be exactly A211 or C211"
        )
    if profile == PRELONG_PROFILE_C211 and (
        group_denominators["target"] != 0 or group_incomes["target"] != 0.0
    ):
        raise PrelongSemanticProducerError(
            "C211 target reward income and denominator must both be zero"
        )

    if bridge_telemetry is not None:
        if not isinstance(bridge_telemetry, Mapping):
            raise PrelongSemanticProducerError(
                "pre-long bridge telemetry must be a mapping or null"
            )
        try:
            # The producer owns the detailed field contract.  The dependency-
            # light formatter still proves that the object is finite canonical
            # JSON before it can enter the sole terminal marker.
            canonical_bridge = json.loads(
                json.dumps(
                    bridge_telemetry,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise PrelongSemanticProducerError(
                "pre-long bridge telemetry is not finite JSON"
            ) from exc
    else:
        canonical_bridge = None

    return {
        "event": PRELONG_SEMANTICS_EVENT,
        "schema_version": PRELONG_SEMANTICS_SCHEMA_VERSION,
        "profile": profile,
        "ppo_update": ppo_update,
        "window": {
            "num_envs": PRELONG_NUM_ENVS,
            "rollout_steps_per_env": PRELONG_ROLLOUT_STEPS,
            "rollout_sample_count": PRELONG_ROLLOUT_SAMPLES,
            "reset_boundary": (
                "prepared before PPO consumes the rollout and acknowledged only "
                "after that optimizer update succeeds"
            ),
        },
        "task_invalid": {
            "observed_sample_count": invalid_samples,
            "task_reward_weighted_sum": invalid_income,
            "task_reward_eligible_denominator": invalid_denominator,
            "eligibility_semantics": (
                "samples observed with the authoritative task_valid mask false; "
                "task reward eligibility is measured from true term masks"
            ),
        },
        "mimic_task_phase_split": {
            "task_invalid_ready": {
                "weighted_sum": ready_mimic_income,
                "eligible_denominator": ready_mimic_denominator,
                "eligibility_semantics": (
                    "active mimic-term income on samples whose authoritative "
                    "pre-step task_valid mask is false"
                ),
            },
            "task_valid_swing": {
                "weighted_sum": swing_mimic_income,
                "eligible_denominator": swing_mimic_denominator,
                "eligibility_semantics": (
                    "active mimic-term income on samples whose authoritative "
                    "pre-step task_valid mask is true"
                ),
            },
            "partition_semantics": (
                "the two masks are disjoint and exhaustive over the aggregate "
                "active-mimic sample union"
            ),
        },
        "strike_timing": {
            "exact_strike_tick_denominator": exact_strike_ticks,
            "denominator_semantics": (
                "task-valid nominal exact-strike timing ticks observed in this PPO "
                "window; this timing denominator is not a closed-attempt denominator"
            ),
        },
        "hit": {
            "eligible_closed_swing_count": closed,
            "actual_contact_numerator": contacts,
            "denominator_semantics": (
                "task-valid attempts transactionally closed in this PPO window; actual "
                "contact is latched selected-rubber contact on the same closure. A swing "
                "may exact-strike in the previous PPO window, so timing is reported separately"
            ),
        },
        "achieved_flight": {
            # Keep the v2 field name for consumers while correcting its
            # denominator semantics: an actual contact remains an outcome
            # opportunity even when the achieved flight fails net/landing gates.
            "eligible_denominator": outcome_opportunities,
            "eligibility_semantics": (
                "task-valid selected-rubber actual-contact outcome opportunities; "
                "net/landing failures remain zero-income denominator rows"
            ),
        },
        "reward_groups": reward_groups,
        "reveal_to_playback_bridge": canonical_bridge,
        "unknown_attribution_count": unknown,
    }


def prelong_semantics_marker_line(
    *,
    ppo_update: int,
    counters: Mapping[str, Any],
    profile: Any = None,
    bridge_telemetry: Any = None,
) -> str:
    """Return the canonical one-line terminal marker for one PPO update."""

    record = build_prelong_semantics_update(
        ppo_update=ppo_update,
        counters=counters,
        profile=profile,
        bridge_telemetry=bridge_telemetry,
    )
    return PRELONG_SEMANTICS_MARKER_PREFIX + json.dumps(
        record,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _runtime_torch():
    try:
        import torch
    except ImportError as exc:
        raise PrelongSemanticLedgerError(
            "runtime pre-long semantics require Torch"
        ) from exc
    return torch


def _require_bool_vector(torch, value, *, name: str, num_envs: int):
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or tuple(value.shape) != (num_envs,)
    ):
        raise PrelongSemanticLedgerError(
            "%s must be a bool tensor with shape [%d]" % (name, num_envs)
        )
    return value


def _require_tensor_vector(torch, value, *, name: str, num_envs: int):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (num_envs,):
        raise PrelongSemanticLedgerError(
            "%s must be a tensor with shape [%d]" % (name, num_envs)
        )
    return value


def _require_xyz_tensor(torch, value, *, name: str, num_envs: int):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
        num_envs,
        3,
    ):
        raise PrelongSemanticLedgerError(
            "%s must be a tensor with shape [%d,3]" % (name, num_envs)
        )
    return value


def prelong_eligibility_masks(command, profile: str) -> Dict[str, Any]:
    """Build true A/C group-union masks from one pre-``env.step`` state.

    The returned tensors are not cloned.  The runtime ledger clones them into
    its step token before the simulator can reveal a hidden task or reset an
    attempt.  This helper performs no mutation and does not read reward values.
    """

    if profile not in PRELONG_PROFILES:
        raise PrelongSemanticLedgerError(
            "unknown pre-long reward profile %r" % (profile,)
        )
    torch = _runtime_torch()
    task_valid = getattr(command, "_action_ball_task_valid", None)
    if not isinstance(task_valid, torch.Tensor) or tuple(task_valid.shape) != (
        PRELONG_NUM_ENVS,
    ):
        raise PrelongSemanticLedgerError(
            "pre-long semantics require authoritative task_valid [4096]"
        )
    num_envs = PRELONG_NUM_ENVS
    task_valid = _require_bool_vector(
        torch,
        task_valid,
        name="task_valid",
        num_envs=num_envs,
    )
    device = task_valid.device

    pre_strike = _require_bool_vector(
        torch,
        getattr(command, "pre_strike", None),
        name="pre_strike",
        num_envs=num_envs,
    )
    strike_window = _require_bool_vector(
        torch,
        getattr(command, "strike_window", None),
        name="strike_window",
        num_envs=num_envs,
    )
    strike_window_pos = getattr(command, "strike_window_pos", None)
    if strike_window_pos is None:
        strike_window_pos = strike_window
    strike_window_pos = _require_bool_vector(
        torch,
        strike_window_pos,
        name="strike_window_pos",
        num_envs=num_envs,
    )
    strike_window_wide = getattr(command, "strike_window_wide", None)
    if strike_window_wide is None:
        strike_window_wide = strike_window
    strike_window_wide = _require_bool_vector(
        torch,
        strike_window_wide,
        name="strike_window_wide",
        num_envs=num_envs,
    )
    for name, value in (
        ("pre_strike", pre_strike),
        ("strike_window", strike_window),
        ("strike_window_pos", strike_window_pos),
        ("strike_window_wide", strike_window_wide),
    ):
        if value.device != device:
            raise PrelongSemanticLedgerError(
                "%s must share the task_valid device" % name
            )

    metrics = getattr(command, "metrics", None)
    if not isinstance(metrics, Mapping):
        raise PrelongSemanticLedgerError("pre-long semantics require command metrics")
    exact = _require_tensor_vector(
        torch,
        metrics.get("exact_strike_hit_rate"),
        name="exact_strike_hit_rate",
        num_envs=num_envs,
    )
    active = _require_bool_vector(
        torch,
        getattr(command, "_action_ball_attempt_active", None),
        name="active attempt",
        num_envs=num_envs,
    )
    fired = _require_bool_vector(
        torch,
        getattr(command, "vb_fired", None),
        name="vb_fired",
        num_envs=num_envs,
    )
    if exact.device != device or active.device != device or fired.device != device:
        raise PrelongSemanticLedgerError(
            "strike/contact masks must share the task_valid device"
        )

    exact_active = (exact > 0.5) & active & task_valid
    if profile == PRELONG_PROFILE_A211:
        component_valid = getattr(command, "action_ball_target_component_valid", None)
        if not callable(component_valid):
            raise PrelongSemanticLedgerError(
                "A211 pre-long semantics require target-component validity"
            )
        component_values = tuple(
            component_valid(name) for name in ("position", "velocity", "face")
        )
        if any(type(value) is not bool for value in component_values):
            raise PrelongSemanticLedgerError(
                "target-component validity must return plain booleans"
            )
        if component_values != (True, True, True):
            raise PrelongSemanticLedgerError("A211 target components must all be valid")
        strike = exact_active
        target = task_valid & (pre_strike | strike_window_pos | strike_window_wide)
        outcome = fired & task_valid
    else:
        component_valid = getattr(command, "action_ball_target_component_valid", None)
        if not callable(component_valid):
            raise PrelongSemanticLedgerError(
                "C211 pre-long semantics require target-component validity"
            )
        component_values = tuple(
            component_valid(name) for name in ("position", "velocity", "face")
        )
        if any(type(value) is not bool for value in component_values):
            raise PrelongSemanticLedgerError(
                "target-component validity must return plain booleans"
            )
        if component_values != (False, False, False):
            raise PrelongSemanticLedgerError(
                "C211 desired-contact target components must all be invalid"
            )
        paddle = _require_xyz_tensor(
            torch,
            getattr(command, "racket_pos_w", None),
            name="C211 achieved paddle centre",
            num_envs=num_envs,
        )
        ball = _require_xyz_tensor(
            torch,
            getattr(command, "_action_ball_ball_contact_target_w", None),
            name="C211 incoming ball centre",
            num_envs=num_envs,
        )
        if paddle.device != device or ball.device != device:
            raise PrelongSemanticLedgerError(
                "C211 strike tensors must share the task_valid device"
            )
        distance = torch.linalg.vector_norm(paddle - ball, dim=-1)
        strike_kernel = torch.reciprocal(1.0 + torch.square(distance / 0.15))
        strike_finite = (
            torch.isfinite(paddle).all(dim=-1)
            & torch.isfinite(ball).all(dim=-1)
            & torch.isfinite(strike_kernel)
        )
        strike = exact_active & strike_finite
        target = torch.zeros_like(task_valid)

        landing = getattr(command, "vb_landing_xy", None)
        if not isinstance(landing, torch.Tensor) or tuple(landing.shape) != (
            num_envs,
            2,
        ):
            raise PrelongSemanticLedgerError(
                "C211 landing position must have shape [%d,2]" % num_envs
            )
        target_xy = getattr(command, "_vb_target_xy_per_env", None)
        if target_xy is None:
            base_target = getattr(command, "_vb_target_xy", None)
            if not isinstance(base_target, torch.Tensor) or tuple(
                base_target.shape
            ) != (2,):
                raise PrelongSemanticLedgerError(
                    "C211 landing target must be [2] or [num_envs,2]"
                )
            target_xy = base_target.unsqueeze(0)
        if (
            not isinstance(target_xy, torch.Tensor)
            or target_xy.ndim != 2
            or tuple(target_xy.shape[1:]) != (2,)
            or int(target_xy.shape[0]) not in (1, num_envs)
            or target_xy.device != device
            or landing.device != device
        ):
            raise PrelongSemanticLedgerError(
                "C211 landing target is not broadcastable to [num_envs,2]"
            )
        cfg = getattr(command, "cfg", None)
        sigma = getattr(cfg, "vb_landing_sigma", None)
        if type(sigma) not in (int, float) or isinstance(sigma, bool):
            raise PrelongSemanticLedgerError(
                "C211 landing sigma must be finite and positive"
            )
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise PrelongSemanticLedgerError(
                "C211 landing sigma must be finite and positive"
            )
        dist2 = torch.sum(torch.square(landing - target_xy), dim=-1)
        landing_kernel = torch.exp(-dist2 / (sigma**2))
        landing_valid = _require_bool_vector(
            torch,
            getattr(command, "vb_landing_valid", None),
            name="vb_landing_valid",
            num_envs=num_envs,
        )
        net_crossed = _require_bool_vector(
            torch,
            getattr(command, "vb_net_crossed", None),
            name="vb_net_crossed",
            num_envs=num_envs,
        )
        net_clear = _require_bool_vector(
            torch,
            getattr(command, "vb_net_clear", None),
            name="vb_net_clear",
            num_envs=num_envs,
        )
        for name, value in (
            ("vb_landing_valid", landing_valid),
            ("vb_net_crossed", net_crossed),
            ("vb_net_clear", net_clear),
        ):
            if value.device != device:
                raise PrelongSemanticLedgerError(
                    "%s must share the task_valid device" % name
                )
        outcome_opportunity = fired & task_valid
        outcome_finite = (
            torch.isfinite(landing).all(dim=-1)
            & torch.isfinite(landing_kernel)
        )
        if bool(torch.any(outcome_opportunity & ~outcome_finite).item()):
            raise PrelongSemanticLedgerError(
                "C211 actual-contact outcome state must be finite"
            )
        # Net/landing gates grade the income, not whether an actual contact was
        # an outcome opportunity.  Failed contacts therefore contribute 0/1.
        outcome = outcome_opportunity

    all_samples = torch.ones_like(task_valid)
    return {
        "task_valid": task_valid,
        "exact_strike_timing": exact_active,
        "groups": {
            "balance": all_samples,
            "mimic": all_samples,
            "strike": strike,
            "target": target,
            "outcome": outcome,
        },
    }


_PRELONG_BRIDGE_WAIT_MIN_TICKS = 5
_PRELONG_BRIDGE_WAIT_MAX_TICKS = 25
_PRELONG_BRIDGE_WAIT_COHORTS = tuple(
    range(_PRELONG_BRIDGE_WAIT_MIN_TICKS, _PRELONG_BRIDGE_WAIT_MAX_TICKS + 1)
)
_PRELONG_MIMIC_EXP_TERMS = frozenset(
    {
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    }
)
_PRELONG_MIMIC_CAUCHY_TERMS = frozenset(
    {
        "motion_racket_position",
        "motion_racket_velocity",
        "motion_racket_normal",
        "motion_racket_long_axis",
    }
)


def _canonical_payload_sha256(value: Any) -> str:
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PrelongSemanticLedgerError(
            "pre-long authority payload is not finite canonical JSON"
        ) from exc
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _single_question_cohort_sha256(
    receipts,
    births,
    *,
    payload_builder,
    sha256_builder,
) -> str:
    """Prove once that every row belongs to one exact semantic question."""

    if (
        not isinstance(receipts, list)
        or not isinstance(births, list)
        or len(receipts) != PRELONG_NUM_ENVS
        or len(births) != PRELONG_NUM_ENVS
    ):
        raise PrelongSemanticLedgerError(
            "pre-long bridge lacks the complete current-center task cohort"
        )
    cohort_sha = None
    for row, (receipt, birth) in enumerate(zip(receipts, births)):
        if receipt is None or birth is None:
            raise PrelongSemanticLedgerError(
                "pre-long bridge current-center task cohort has an empty row"
            )
        digest = _require_sha256(
            sha256_builder(
                payload_builder(
                    action_uid=int(receipt.action_uid),
                    action_slot=int(receipt.action_slot),
                    birth=birth,
                    sample=receipt,
                    mount_normal_sign=int(receipt.mount_normal_sign),
                )
            ),
            name="pre-long exact question SHA-256 at row %d" % row,
        )
        if cohort_sha is None:
            cohort_sha = digest
        elif digest != cohort_sha:
            raise PrelongSemanticLedgerError(
                "pre-long bridge current-center cohort contains multiple questions"
            )
    if cohort_sha is None:  # exact 4096-row validation above makes this unreachable
        raise PrelongSemanticLedgerError(
            "pre-long bridge current-center task cohort is empty"
        )
    return cohort_sha


def _prelong_bridge_authority(
    command,
    profile: str,
    reward_recipe_sha256: str,
    *,
    required: bool,
):
    """Bind the bridge marker to the already admitted N=1 task authorities.

    This runs once, at ledger construction.  Per-step accounting remains on
    device.  A narrow payload hook exists for dependency-light host tests; the
    production path derives every field from RacketTargetCommand's sealed hard
    contract and current-center receipt, and fails closed when that evidence is
    unavailable.
    """

    if type(required) is not bool:
        raise PrelongSemanticLedgerError(
            "pre-long bridge required flag must be boolean"
        )
    schedule = getattr(command, "_action_ball_task_wait_schedule", None)
    if schedule is None:
        if required:
            raise PrelongSemanticLedgerError(
                "pre-long runtime requires the ActionBall task-wait schedule"
            )
        return None
    if (
        type(getattr(schedule, "min_wait_ticks", None)) is not int
        or type(getattr(schedule, "max_wait_ticks", None)) is not int
        or schedule.min_wait_ticks != _PRELONG_BRIDGE_WAIT_MIN_TICKS
        or schedule.max_wait_ticks != _PRELONG_BRIDGE_WAIT_MAX_TICKS
    ):
        raise PrelongSemanticLedgerError(
            "pre-long reveal bridge requires the exact inclusive WAIT cohort 5..25"
        )
    wait_schedule_sha = _require_sha256(
        getattr(schedule, "canonical_sha256", None),
        name="pre-long WAIT schedule SHA-256",
    )

    hard_contract = getattr(command, "action_ball_hard_contract", None)
    if not callable(hard_contract):
        raise PrelongSemanticLedgerError(
            "pre-long bridge requires action_ball_hard_contract()"
        )
    hard = hard_contract()
    if not isinstance(hard, Mapping):
        raise PrelongSemanticLedgerError(
            "pre-long bridge hard contract is absent"
        )
    timing = hard.get("timing")
    target = hard.get("target_provider")
    profiles = hard.get("profiles")
    sampling = hard.get("sampling")
    if not all(
        isinstance(value, Mapping)
        for value in (timing, target, profiles, sampling)
    ):
        raise PrelongSemanticLedgerError(
            "pre-long bridge hard contract lacks timing/question/sampler authority"
        )
    if sampling.get("initial_center_single_question") is not True:
        raise PrelongSemanticLedgerError(
            "pre-long bridge scale run requires initial_center_single_question=true"
        )
    action_order = hard.get("action_order")
    if not isinstance(action_order, list) or len(action_order) != 1:
        raise PrelongSemanticLedgerError(
            "pre-long bridge scale run requires exactly one action"
        )
    target_source = target.get("source")
    target_recipe = target.get("recipe")
    timing_authority = timing.get("authority")
    sampler_sha = profiles.get("sampler_contract_sha256")
    if any(
        type(value) is not str or not value
        for value in (target_source, target_recipe, timing_authority)
    ):
        raise PrelongSemanticLedgerError(
            "pre-long bridge source/recipe/timing authority is malformed"
        )
    sampler_sha = _require_sha256(
        sampler_sha, name="pre-long sampler contract SHA-256"
    )

    # This is a one-time construction proof, not a per-step/per-row log.  The
    # hard-contract claim is insufficient on its own: every installed row is
    # independently re-hashed and must equal the same exact question.
    receipts = getattr(command, "_action_ball_task_by_env", None)
    births = getattr(command, "_action_ball_birth_by_env", None)
    try:
        from whole_body_tracking.tasks.tracking.mdp.action_ball_question_cache import (
            exact_question_sha256,
        )
        from whole_body_tracking.tasks.tracking.mdp.hope_commands import (
            _action_ball_exact_question_payload,
        )

        question_sha = _single_question_cohort_sha256(
            receipts,
            births,
            payload_builder=_action_ball_exact_question_payload,
            sha256_builder=exact_question_sha256,
        )
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise PrelongSemanticLedgerError(
            "pre-long bridge cannot derive the exact current-center question SHA"
        ) from exc

    action_family = profile
    manifest = getattr(command, "_action_ball_manifest", None)
    actions = getattr(manifest, "actions", None)
    if isinstance(actions, (list, tuple)) and len(actions) == 1:
        observed_family = getattr(actions[0], "family", None)
        if type(observed_family) is str and observed_family:
            action_family = observed_family
    authority = {
        "family": action_family,
        "target_source": target_source,
        "target_recipe": target_recipe,
        "timing_authority": timing_authority,
        "timing_contract_sha256": _canonical_payload_sha256(timing),
        "question_sha256": question_sha,
        "question_sha_semantics": (
            "exact current-center semantic solver question; all 4096 installed "
            "rows construction-time verified equal; no hot-path row logging"
        ),
        "sampler_contract_sha256": sampler_sha,
    }

    required = {
        "family",
        "target_source",
        "target_recipe",
        "timing_authority",
        "timing_contract_sha256",
        "question_sha256",
        "question_sha_semantics",
        "sampler_contract_sha256",
    }
    if set(authority) != required:
        raise PrelongSemanticLedgerError(
            "pre-long bridge authority has missing or unexpected fields"
        )
    for name in (
        "family",
        "target_source",
        "target_recipe",
        "timing_authority",
        "question_sha_semantics",
    ):
        if type(authority[name]) is not str or not authority[name]:
            raise PrelongSemanticLedgerError(
                "pre-long bridge authority %s must be a non-empty string" % name
            )
    for name in (
        "timing_contract_sha256",
        "question_sha256",
        "sampler_contract_sha256",
    ):
        authority[name] = _require_sha256(
            authority[name], name="pre-long bridge %s" % name
        )
    authority.update(
        {
            "profile": profile,
            "effective_reward_recipe_sha256": _require_sha256(
                reward_recipe_sha256,
                name="pre-long effective reward recipe SHA-256",
            ),
            "wait_schedule_sha256": wait_schedule_sha,
            "wait_cohort_ticks": list(_PRELONG_BRIDGE_WAIT_COHORTS),
            "policy_dt_s": PRELONG_POLICY_DT_S,
        }
    )
    return authority


class _PrelongStepToken:
    __slots__ = (
        "ledger",
        "sequence",
        "common_step_counter",
        "task_valid",
        "exact_strike_timing",
        "group_masks",
        "bridge_mask",
        "bridge_mimic_masks",
        "bridge_mimic_scales",
    )

    def __init__(
        self,
        *,
        ledger,
        sequence: int,
        common_step_counter: int,
        task_valid,
        exact_strike_timing,
        group_masks: Mapping[str, Any],
        bridge_mask,
        bridge_mimic_masks: Mapping[str, Any],
        bridge_mimic_scales: Mapping[str, Any],
    ):
        self.ledger = ledger
        self.sequence = sequence
        self.common_step_counter = common_step_counter
        self.task_valid = task_valid
        self.exact_strike_timing = exact_strike_timing
        self.group_masks = group_masks
        self.bridge_mask = bridge_mask
        self.bridge_mimic_masks = bridge_mimic_masks
        self.bridge_mimic_scales = bridge_mimic_scales


class _PrelongPreparedUpdate:
    __slots__ = (
        "ledger",
        "ppo_update",
        "counters",
        "record",
        "marker_line",
        "end_common_step_counter",
        "end_closure_totals",
    )

    def __init__(
        self,
        *,
        ledger,
        ppo_update: int,
        counters: Mapping[str, Any],
        record: Mapping[str, Any],
        marker_line: str,
        end_common_step_counter: int,
        end_closure_totals: Tuple[int, int],
    ):
        self.ledger = ledger
        self.ppo_update = ppo_update
        self.counters = dict(counters)
        self.record = dict(record)
        self.marker_line = marker_line
        self.end_common_step_counter = end_common_step_counter
        self.end_closure_totals = end_closure_totals


class _PrelongPreparedAcknowledgement:
    """Fully validated post-service commit; ``consume`` cannot fail."""

    __slots__ = (
        "ledger",
        "ppo_update",
        "marker_line",
        "common_step_counter",
        "closure_totals",
        "fresh_accumulators",
        "consumed",
    )

    def __init__(
        self,
        *,
        ledger,
        ppo_update: int,
        marker_line: str,
        common_step_counter: int,
        closure_totals: Tuple[int, int],
        fresh_accumulators: Mapping[str, Any],
    ):
        self.ledger = ledger
        self.ppo_update = ppo_update
        self.marker_line = marker_line
        self.common_step_counter = common_step_counter
        self.closure_totals = closure_totals
        self.fresh_accumulators = fresh_accumulators
        self.consumed = False

    def consume(self) -> str:
        """Install already-allocated state using assignment-only operations."""

        if self.consumed:
            return self.marker_line
        self.consumed = True
        ledger = self.ledger
        ledger._last_finished_update = self.ppo_update
        ledger._update_start_common_step_counter = self.common_step_counter
        ledger._closure_start_totals = self.closure_totals
        ledger._pending_update = None
        ledger._install_accumulators(self.fresh_accumulators)
        return self.marker_line


class ActionBallPrelongSemanticsLedger:
    """Transactionally accumulate one exact 4096x24 A211/C211 PPO window."""

    _CLOSURE_RTOL = 1.0e-5
    _CLOSURE_ATOL = 1.0e-7

    def __init__(
        self,
        env,
        *,
        preregistered_effective_reward_recipe: Mapping[str, Any],
        require_bridge_telemetry: bool = True,
    ):
        if type(require_bridge_telemetry) is not bool:
            raise PrelongSemanticLedgerError(
                "require_bridge_telemetry must be an exact boolean"
            )
        self._torch = _runtime_torch()
        self._env = env
        self._manager = getattr(env, "reward_manager", None)
        if self._manager is None:
            raise PrelongSemanticLedgerError(
                "pre-long semantics require reward_manager"
            )
        get_term_cfg = getattr(self._manager, "get_term_cfg", None)
        if not callable(get_term_cfg):
            raise PrelongSemanticLedgerError(
                "pre-long semantics require RewardManager.get_term_cfg(name)"
            )
        self._get_term_cfg = get_term_cfg
        raw_names = getattr(self._manager, "active_terms", None)
        if not isinstance(raw_names, (list, tuple)):
            raise PrelongSemanticLedgerError(
                "pre-long semantics require ordered RewardManager active_terms"
            )
        self._all_names = tuple(raw_names)
        if (
            not self._all_names
            or any(type(name) is not str or not name for name in self._all_names)
            or len(self._all_names) != len(set(self._all_names))
        ):
            raise PrelongSemanticLedgerError(
                "pre-long RewardManager active_terms must be unique names"
            )
        (
            runtime_recipe_receipt,
            normalized_recipe_terms,
        ) = _normalized_runtime_reward_recipe(self._manager, self._all_names)
        self.preregistered_effective_reward_recipe = (
            _validate_preregistered_reward_recipe(
                preregistered_effective_reward_recipe,
                runtime_recipe_receipt,
            )
        )
        self.effective_reward_recipe_sha256 = runtime_recipe_receipt["sha256"]
        step_reward = getattr(self._manager, "_step_reward", None)
        if not isinstance(step_reward, self._torch.Tensor) or tuple(
            step_reward.shape
        ) != (
            PRELONG_NUM_ENVS,
            len(self._all_names),
        ):
            raise PrelongSemanticLedgerError(
                "pre-long RewardManager _step_reward must have exact shape [%d,%d]"
                % (PRELONG_NUM_ENVS, len(self._all_names))
            )
        reward_buf = getattr(self._manager, "_reward_buf", None)
        if not isinstance(reward_buf, self._torch.Tensor) or tuple(
            reward_buf.shape
        ) != (PRELONG_NUM_ENVS,):
            raise PrelongSemanticLedgerError(
                "pre-long RewardManager _reward_buf must have shape [%d]"
                % PRELONG_NUM_ENVS
            )
        step_dt = getattr(env, "step_dt", None)
        if type(step_dt) not in (int, float) or isinstance(step_dt, bool):
            raise PrelongSemanticLedgerError(
                "pre-long environment step_dt must be 0.02 s"
            )
        self._step_dt = float(step_dt)
        if not math.isclose(
            self._step_dt,
            PRELONG_POLICY_DT_S,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise PrelongSemanticLedgerError(
                "pre-long environment step_dt must be 0.02 s"
            )

        bindings = []
        nonzero_weights = {}
        for index, name in enumerate(self._all_names):
            cfg = get_term_cfg(name)
            weight = getattr(cfg, "weight", None)
            if type(weight) not in (int, float) or isinstance(weight, bool):
                raise PrelongSemanticLedgerError(
                    "reward term %r weight must be finite" % name
                )
            weight = float(weight)
            if not math.isfinite(weight):
                raise PrelongSemanticLedgerError(
                    "reward term %r weight must be finite" % name
                )
            func = getattr(cfg, "func", None)
            if not callable(func):
                raise PrelongSemanticLedgerError(
                    "reward term %r has no callable" % name
                )
            required_params = PRELONG_REQUIRED_TERM_PARAMS.get(name, {})
            params = getattr(cfg, "params", None)
            if required_params and (
                not isinstance(params, Mapping)
                or any(
                    params.get(key) != value for key, value in required_params.items()
                )
            ):
                raise PrelongSemanticLedgerError(
                    "reward term %r required parameters differ" % name
                )
            bindings.append(
                {
                    "index": index,
                    "name": name,
                    "weight": weight,
                    "func": func,
                    "required_params": dict(required_params),
                    "recipe_term": normalized_recipe_terms[name],
                }
            )
            if weight != 0.0:
                nonzero_weights[name] = weight
        self._bindings = tuple(bindings)
        self.profile = classify_prelong_reward_profile(nonzero_weights)
        expected_callables = expected_prelong_callable_names(self.profile)
        for binding in self._bindings:
            if binding["weight"] == 0.0:
                continue
            observed_name = getattr(binding["func"], "__name__", None)
            expected_name = expected_callables[binding["name"]]
            if observed_name != expected_name:
                raise PrelongSemanticLedgerError(
                    "reward term %r callable differs: expected %r, observed %r"
                    % (binding["name"], expected_name, observed_name)
                )
        self._group_terms = prelong_group_term_weights(self.profile)
        name_to_index = {name: index for index, name in enumerate(self._all_names)}
        self._group_indices = {
            group: tuple(name_to_index[name] for name in terms)
            for group, terms in self._group_terms.items()
        }
        self._zero_weight_indices = tuple(
            binding["index"] for binding in self._bindings if binding["weight"] == 0.0
        )
        self._probe_indices = tuple(
            name_to_index[name] for name in PRELONG_EXCLUDED_PROBE_TERM_WEIGHTS
        )
        self._task_indices = tuple(
            sorted(
                {
                    index
                    for group in ("strike", "target", "outcome")
                    for index in self._group_indices[group]
                }
            )
        )

        command_manager = getattr(env, "command_manager", None)
        get_command = getattr(command_manager, "get_term", None)
        if not callable(get_command):
            raise PrelongSemanticLedgerError(
                "pre-long semantics require command_manager.get_term(name)"
            )
        self._command = get_command("racket_target")
        if not callable(getattr(self._command, "_action_ball_ledger_payload", None)):
            raise PrelongSemanticLedgerError(
                "pre-long semantics require the cumulative ActionBall C/H ledger"
            )
        # Validate the pre-step mask surface before the first simulator mutation.
        initial_masks = prelong_eligibility_masks(self._command, self.profile)
        if initial_masks["task_valid"].device != step_reward.device:
            raise PrelongSemanticLedgerError(
                "pre-long command and reward cache must share one device"
            )

        self._bridge_authority = _prelong_bridge_authority(
            self._command,
            self.profile,
            self.effective_reward_recipe_sha256,
            required=require_bridge_telemetry,
        )
        self._bridge_enabled = self._bridge_authority is not None
        self._mimic_bindings = tuple(
            binding
            for binding in self._bindings
            if binding["name"] in self._group_terms["mimic"]
        )
        if {binding["name"] for binding in self._mimic_bindings} != set(
            self._group_terms["mimic"]
        ):
            raise PrelongSemanticLedgerError(
                "pre-long bridge mimic bindings do not cover the admitted recipe"
            )
        if self._bridge_enabled:
            self._initialize_bridge_runtime(step_reward.device)

        self._last_finished_update = None
        self._pending_update = None
        self._open_step = None
        self._next_step_sequence = 0
        self._update_start_common_step_counter = self._read_common_step_counter()
        self._closure_start_totals = self._read_closure_totals()
        self._reset_accumulators(step_reward.device)
        if self._bridge_enabled:
            self._bridge_safety_snapshot(
                self._torch.zeros(
                    PRELONG_NUM_ENVS,
                    dtype=self._torch.bool,
                    device=step_reward.device,
                )
            )

    def _initialize_bridge_runtime(self, device) -> None:
        torch = self._torch
        n = PRELONG_NUM_ENVS
        command = self._command
        motion_getter = getattr(command, "_motion", None)
        if not callable(motion_getter):
            raise PrelongSemanticLedgerError(
                "pre-long bridge requires RacketTargetCommand._motion()"
            )
        self._bridge_motion = motion_getter()
        required_vectors = (
            ("_action_ball_task_valid", torch.bool),
            ("_action_ball_task_wait_total_ticks", torch.long),
            ("_action_ball_task_wait_elapsed_ticks", torch.long),
            ("_action_ball_reset_generation", torch.long),
        )
        for name, dtype in required_vectors:
            value = getattr(command, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (n,)
                or value.dtype != dtype
                or value.device != device
            ):
                raise PrelongSemanticLedgerError(
                    "pre-long bridge requires %s [%d] on the reward device"
                    % (name, n)
                )
        for name in (
            "_action_ball_task_age_s",
            "_action_ball_time_to_contact_s",
            "_action_ball_teacher_rate",
            "_action_ball_scaled_t_hit_s",
            "_action_ball_pre_swing_wait_s",
        ):
            value = getattr(self._bridge_motion, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (n,)
                or value.device != device
                or not value.is_floating_point()
            ):
                raise PrelongSemanticLedgerError(
                    "pre-long bridge requires MotionCommand.%s [%d] on the reward device"
                    % (name, n)
                )

        # One persistent device ledger spans PPO boundaries.  Only the compact
        # aggregate is copied to host once in prepare_update().
        cohort_count = len(_PRELONG_BRIDGE_WAIT_COHORTS)
        self._bridge_seen_generation = torch.full(
            (n,), -1, dtype=torch.long, device=device
        )
        self._bridge_active = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_active_generation = torch.full(
            (n,), -1, dtype=torch.long, device=device
        )
        self._bridge_wait_ticks = torch.zeros(n, dtype=torch.long, device=device)
        self._bridge_expected_ticks = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self._bridge_elapsed_ticks = torch.zeros(n, dtype=torch.long, device=device)
        self._bridge_lifetime_reveal = torch.zeros(
            cohort_count, dtype=torch.long, device=device
        )
        self._bridge_lifetime_start = torch.zeros_like(
            self._bridge_lifetime_reveal
        )
        self._bridge_lifetime_terminal = torch.zeros_like(
            self._bridge_lifetime_reveal
        )
        self._bridge_timing_names = (
            "time_to_contact_tick",
            "teacher_rate",
            "scaled_t_hit_s",
            "pre_swing_wait_s",
            "expected_bridge_ticks",
        )
        self._bridge_timing_sum = torch.zeros(
            len(self._bridge_timing_names), dtype=torch.float64, device=device
        )
        self._bridge_timing_min = torch.full_like(
            self._bridge_timing_sum, float("inf")
        )
        self._bridge_timing_max = torch.full_like(
            self._bridge_timing_sum, float("-inf")
        )
        self._bridge_timing_count = torch.zeros((), dtype=torch.long, device=device)

        if self.profile == PRELONG_PROFILE_A211:
            base_position = tuple(
                binding
                for binding in self._bindings
                if binding["name"] == "base_position"
            )
            if any(binding["weight"] != 0.0 for binding in base_position):
                raise PrelongSemanticLedgerError(
                    "A211 bridge requires base_position to be absent/weight zero"
                )
            self._bridge_allowed_task_indices = (
                self._all_names.index("racket_progress"),
            )
        else:
            self._bridge_allowed_task_indices = ()

    def _bridge_mimic_snapshot(self, bridge_mask):
        torch = self._torch
        if not self._bridge_enabled:
            return {}, {}
        command = self._command
        motion = self._bridge_motion
        imitation_eligible = getattr(motion, "imitation_eligible", None)
        if imitation_eligible is None:
            raise PrelongSemanticLedgerError(
                "pre-long bridge requires MotionCommand.imitation_eligible"
            )
        imitation_eligible = _require_bool_vector(
            torch,
            imitation_eligible,
            name="bridge mimic imitation_eligible",
            num_envs=PRELONG_NUM_ENVS,
        )
        wide = _require_bool_vector(
            torch,
            getattr(command, "strike_window_wide", None),
            name="bridge mimic strike_window_wide",
            num_envs=PRELONG_NUM_ENVS,
        ) & _require_bool_vector(
            torch,
            getattr(command, "_action_ball_task_valid", None),
            name="bridge mimic task_valid",
            num_envs=PRELONG_NUM_ENVS,
        )
        masks = {}
        scales = {}
        for binding in self._mimic_bindings:
            name = binding["name"]
            params = getattr(self._get_term_cfg(name), "params", None)
            if not isinstance(params, Mapping):
                raise PrelongSemanticLedgerError(
                    "bridge mimic term %r params must be a mapping" % name
                )
            std = params.get("std")
            if type(std) not in (int, float) or isinstance(std, bool):
                raise PrelongSemanticLedgerError(
                    "bridge mimic term %r requires finite positive std" % name
                )
            std = float(std)
            if not math.isfinite(std) or std <= 0.0:
                raise PrelongSemanticLedgerError(
                    "bridge mimic term %r requires finite positive std" % name
                )
            if name in {"motion_body_pos", "motion_body_ori"}:
                eligible = bridge_mask & imitation_eligible
            else:
                eligible = bridge_mask
            scale_value = params.get(
                "scale_in_strike_window", params.get("window_scale", 1.0)
            )
            if type(scale_value) not in (int, float) or isinstance(
                scale_value, bool
            ):
                raise PrelongSemanticLedgerError(
                    "bridge mimic term %r has malformed window scale" % name
                )
            scale_value = float(scale_value)
            if not math.isfinite(scale_value) or scale_value <= 0.0:
                raise PrelongSemanticLedgerError(
                    "bridge mimic term %r window scale must be finite and positive"
                    % name
                )
            scales[name] = torch.where(
                wide,
                torch.full_like(wide, scale_value, dtype=torch.float64),
                torch.ones_like(wide, dtype=torch.float64),
            )
            masks[name] = eligible
        return masks, scales

    def _bridge_safety_snapshot(self, bridge_mask) -> None:
        """Accumulate compact policy-boundary safety without per-row host IO."""

        torch = self._torch
        command = self._command
        robot = getattr(command, "robot", None)
        data = getattr(robot, "data", None)
        if data is None:
            raise PrelongSemanticLedgerError(
                "pre-long bridge requires RacketTargetCommand.robot.data"
            )
        q = getattr(data, "joint_pos", None)
        qvel = getattr(data, "joint_vel", None)
        limits = getattr(data, "joint_pos_limits", None)
        qvel_limits = getattr(data, "joint_vel_limits", None)
        root_pos = getattr(data, "root_pos_w", None)
        root_lin_vel = getattr(data, "root_lin_vel_w", None)
        if (
            not isinstance(q, torch.Tensor)
            or q.ndim != 2
            or int(q.shape[0]) != PRELONG_NUM_ENVS
            or not isinstance(qvel, torch.Tensor)
            or tuple(qvel.shape) != tuple(q.shape)
        ):
            raise PrelongSemanticLedgerError(
                "pre-long bridge requires robot joint_pos/joint_vel [4096,J]"
            )
        if not isinstance(limits, torch.Tensor) or limits.shape[-1] != 2:
            raise PrelongSemanticLedgerError(
                "pre-long bridge requires physical joint_pos_limits[...,2]"
            )
        if limits.ndim == 2:
            limits = limits.unsqueeze(0)
        if (
            limits.ndim != 3
            or int(limits.shape[0]) not in (1, PRELONG_NUM_ENVS)
            or int(limits.shape[1]) != int(q.shape[1])
        ):
            raise PrelongSemanticLedgerError(
                "pre-long bridge joint_pos_limits are not broadcastable"
            )
        if not isinstance(qvel_limits, torch.Tensor):
            raise PrelongSemanticLedgerError(
                "pre-long bridge requires physical joint_vel_limits"
            )
        if qvel_limits.ndim == 1:
            qvel_limits = qvel_limits.unsqueeze(0)
        if (
            qvel_limits.ndim != 2
            or int(qvel_limits.shape[0]) not in (1, PRELONG_NUM_ENVS)
            or int(qvel_limits.shape[1]) != int(q.shape[1])
        ):
            raise PrelongSemanticLedgerError(
                "pre-long bridge joint_vel_limits are not broadcastable"
            )
        if (
            not isinstance(root_pos, torch.Tensor)
            or tuple(root_pos.shape) != (PRELONG_NUM_ENVS, 3)
            or not isinstance(root_lin_vel, torch.Tensor)
            or tuple(root_lin_vel.shape) != (PRELONG_NUM_ENVS, 3)
        ):
            raise PrelongSemanticLedgerError(
                "pre-long bridge requires root_pos_w/root_lin_vel_w [4096,3]"
            )
        metrics = getattr(command, "metrics", None)
        if not isinstance(metrics, Mapping):
            raise PrelongSemanticLedgerError(
                "pre-long bridge requires command metrics"
            )
        upright = _require_tensor_vector(
            torch,
            metrics.get("base_upright"),
            name="bridge base_upright",
            num_envs=PRELONG_NUM_ENVS,
        )
        foot_contact = _require_tensor_vector(
            torch,
            metrics.get("foot_contact_frac"),
            name="bridge foot_contact_frac",
            num_envs=PRELONG_NUM_ENVS,
        )
        foot_slip = _require_tensor_vector(
            torch,
            metrics.get("foot_slip_speed"),
            name="bridge foot_slip_speed",
            num_envs=PRELONG_NUM_ENVS,
        )
        tensors = (
            q,
            qvel,
            limits,
            qvel_limits,
            root_pos,
            root_lin_vel,
            upright,
            foot_contact,
            foot_slip,
        )
        if any(value.device != bridge_mask.device for value in tensors):
            raise PrelongSemanticLedgerError(
                "pre-long bridge safety tensors must share the reward device"
            )
        with torch.inference_mode():
            lower_gap = q - limits[..., 0]
            upper_gap = limits[..., 1] - q
            hard_gap = torch.minimum(lower_gap, upper_gap)
            qvel_ratio = torch.abs(qvel) / qvel_limits
            root_speed_xy = torch.linalg.vector_norm(root_lin_vel[:, :2], dim=-1)
            row_finite = (
                torch.isfinite(hard_gap).all(dim=-1)
                & torch.isfinite(qvel_ratio).all(dim=-1)
                & torch.isfinite(root_pos[:, 2])
                & torch.isfinite(root_speed_xy)
                & torch.isfinite(upright)
                & torch.isfinite(foot_contact)
                & torch.isfinite(foot_slip)
                & (qvel_limits > 0.0).all(dim=-1)
            )
            self._bridge_violation_counts["nonfinite_or_invalid_safety"].add_(
                (bridge_mask & ~row_finite).sum(dtype=torch.long)
            )
            valid = bridge_mask & row_finite
            self._bridge_safety_count.add_(valid.sum(dtype=torch.long))
            joint_valid = valid[:, None]
            inf = torch.full((), float("inf"), dtype=q.dtype, device=q.device)
            neg_inf = torch.full(
                (), float("-inf"), dtype=q.dtype, device=q.device
            )
            self._bridge_safety_min[0].copy_(
                torch.minimum(
                    self._bridge_safety_min[0],
                    torch.where(joint_valid, hard_gap, inf).amin().to(torch.float64),
                )
            )
            self._bridge_safety_max[0].copy_(
                torch.maximum(
                    self._bridge_safety_max[0],
                    torch.where(joint_valid, qvel_ratio, neg_inf).amax().to(
                        torch.float64
                    ),
                )
            )
            scalar_inf = torch.full_like(root_pos[:, 2], float("inf"))
            scalar_neg_inf = torch.full_like(root_pos[:, 2], float("-inf"))
            self._bridge_safety_min[1].copy_(
                torch.minimum(
                    self._bridge_safety_min[1],
                    torch.where(valid, root_pos[:, 2], scalar_inf)
                    .amin()
                    .to(torch.float64),
                )
            )
            self._bridge_safety_max[1].copy_(
                torch.maximum(
                    self._bridge_safety_max[1],
                    torch.where(valid, root_pos[:, 2], scalar_neg_inf)
                    .amax()
                    .to(torch.float64),
                )
            )
            self._bridge_safety_min[2].copy_(
                torch.minimum(
                    self._bridge_safety_min[2],
                    torch.where(valid, upright, torch.full_like(upright, float("inf")))
                    .amin()
                    .to(torch.float64),
                )
            )
            self._bridge_safety_max[2].copy_(
                torch.maximum(
                    self._bridge_safety_max[2],
                    torch.where(
                        valid,
                        root_speed_xy,
                        torch.full_like(root_speed_xy, float("-inf")),
                    )
                    .amax()
                    .to(torch.float64),
                )
            )
            self._bridge_safety_sum[0].add_(
                torch.where(valid, foot_contact, torch.zeros_like(foot_contact)).sum(
                    dtype=torch.float64
                )
            )
            self._bridge_safety_sum[1].add_(
                torch.where(valid, foot_slip, torch.zeros_like(foot_slip)).sum(
                    dtype=torch.float64
                )
            )
            self._bridge_safety_max[3].copy_(
                torch.maximum(
                    self._bridge_safety_max[3],
                    torch.where(
                        valid,
                        foot_slip,
                        torch.full_like(foot_slip, float("-inf")),
                    )
                    .amax()
                    .to(torch.float64),
                )
            )

    def _begin_bridge_snapshot(self):
        torch = self._torch
        if not self._bridge_enabled:
            empty = torch.zeros(
                PRELONG_NUM_ENVS,
                dtype=torch.bool,
                device=getattr(self._manager, "_step_reward").device,
            )
            return empty, {}, {}
        command = self._command
        motion = self._bridge_motion
        task_valid = command._action_ball_task_valid
        total = command._action_ball_task_wait_total_ticks
        elapsed = command._action_ball_task_wait_elapsed_ticks
        generation = command._action_ball_reset_generation

        stale = self._bridge_active & (
            (generation != self._bridge_active_generation) | ~task_valid
        )
        self._bridge_violation_counts[
            "generation_or_validity_changed_without_terminal"
        ].add_(stale.sum(dtype=torch.long))
        reveal = (
            task_valid
            & (total >= _PRELONG_BRIDGE_WAIT_MIN_TICKS)
            & (total <= _PRELONG_BRIDGE_WAIT_MAX_TICKS)
            & (elapsed == total)
            & (generation != self._bridge_seen_generation)
        )
        malformed_reveal = task_valid & (elapsed == total) & (total > 0) & ~(
            (total >= _PRELONG_BRIDGE_WAIT_MIN_TICKS)
            & (total <= _PRELONG_BRIDGE_WAIT_MAX_TICKS)
        )
        self._bridge_violation_counts["wait_outside_sealed_cohort"].add_(
            malformed_reveal.sum(dtype=torch.long)
        )
        self._bridge_violation_counts["duplicate_reveal_before_close"].add_(
            (reveal & self._bridge_active).sum(dtype=torch.long)
        )

        base_prewait = (
            motion._action_ball_pre_swing_wait_s
            - total.to(dtype=torch.float64) * self._step_dt
        )
        base_ttc = (
            motion._action_ball_time_to_contact_s
            - total.to(dtype=torch.float64) * self._step_dt
        )
        timing_finite = (
            torch.isfinite(base_prewait)
            & torch.isfinite(base_ttc)
            & torch.isfinite(motion._action_ball_teacher_rate)
            & torch.isfinite(motion._action_ball_scaled_t_hit_s)
            & (base_prewait >= 0.0)
            & (base_ttc > 0.0)
            & (motion._action_ball_teacher_rate > 0.0)
            & (motion._action_ball_scaled_t_hit_s > 0.0)
        )
        self._bridge_violation_counts["invalid_timing_tuple"].add_(
            (reveal & ~timing_finite).sum(dtype=torch.long)
        )
        ttc_tick = torch.round(base_ttc / self._step_dt)
        self._bridge_violation_counts["ttc_off_policy_tick_grid"].add_(
            (
                reveal
                & (torch.abs(base_ttc - ttc_tick * self._step_dt) > 1.0e-7)
            ).sum(dtype=torch.long)
        )
        expected = torch.floor(
            base_prewait / self._step_dt + 1.0e-10
        ).to(dtype=torch.long) + 1
        self._bridge_violation_counts["invalid_expected_bridge_ticks"].add_(
            (reveal & (expected <= 0)).sum(dtype=torch.long)
        )
        cohort = total[reveal] - _PRELONG_BRIDGE_WAIT_MIN_TICKS
        self._bridge_lifetime_reveal.add_(
            torch.bincount(
                cohort,
                minlength=len(_PRELONG_BRIDGE_WAIT_COHORTS),
            )
        )
        self._bridge_seen_generation[reveal] = generation[reveal]
        self._bridge_active_generation[reveal] = generation[reveal]
        self._bridge_wait_ticks[reveal] = total[reveal]
        self._bridge_expected_ticks[reveal] = expected[reveal]
        self._bridge_elapsed_ticks[reveal] = 0
        self._bridge_active[reveal] = True
        timing_rows = torch.stack(
            (
                ttc_tick,
                motion._action_ball_teacher_rate.to(dtype=torch.float64),
                motion._action_ball_scaled_t_hit_s.to(dtype=torch.float64),
                base_prewait,
                expected.to(dtype=torch.float64),
            ),
            dim=1,
        )
        reveal_column = reveal[:, None]
        self._bridge_timing_sum.add_(
            torch.where(reveal_column, timing_rows, torch.zeros_like(timing_rows)).sum(
                dim=0
            )
        )
        self._bridge_timing_min.copy_(
            torch.minimum(
                self._bridge_timing_min,
                torch.where(
                    reveal_column,
                    timing_rows,
                    torch.full_like(timing_rows, float("inf")),
                ).amin(dim=0),
            )
        )
        self._bridge_timing_max.copy_(
            torch.maximum(
                self._bridge_timing_max,
                torch.where(
                    reveal_column,
                    timing_rows,
                    torch.full_like(timing_rows, float("-inf")),
                ).amax(dim=0),
            )
        )
        self._bridge_timing_count.add_(reveal.sum(dtype=torch.long))

        playback_started = self._bridge_active & (
            motion._action_ball_task_age_s
            > motion._action_ball_pre_swing_wait_s + 1.0e-12
        )
        bad_ticks = playback_started & (
            self._bridge_elapsed_ticks != self._bridge_expected_ticks
        )
        self._bridge_violation_counts["playback_start_tick_mismatch"].add_(
            bad_ticks.sum(dtype=torch.long)
        )
        cohort = (
            self._bridge_wait_ticks[playback_started]
            - _PRELONG_BRIDGE_WAIT_MIN_TICKS
        )
        self._bridge_lifetime_start.add_(
            torch.bincount(
                cohort,
                minlength=len(_PRELONG_BRIDGE_WAIT_COHORTS),
            )
        )
        self._bridge_active[playback_started] = False

        bridge_mask = self._bridge_active.clone()
        self._bridge_elapsed_ticks[bridge_mask] += 1
        self._bridge_safety_snapshot(bridge_mask)
        mimic_masks, mimic_scales = self._bridge_mimic_snapshot(bridge_mask)
        return bridge_mask, mimic_masks, mimic_scales

    def _read_common_step_counter(self) -> int:
        value = getattr(self._env, "common_step_counter", None)
        if type(value) is not int or value < 0:
            raise PrelongSemanticLedgerError(
                "pre-long semantics require nonnegative plain common_step_counter"
            )
        return value

    def _read_closure_totals(self) -> Tuple[int, int]:
        payload = self._command._action_ball_ledger_payload()
        if not isinstance(payload, Mapping) or not payload:
            raise PrelongSemanticLedgerError(
                "ActionBall cumulative closure ledger is absent"
            )
        closed = 0
        contacts = 0
        for action, row in payload.items():
            if type(action) is not str or not isinstance(row, Mapping):
                raise PrelongSemanticLedgerError(
                    "ActionBall cumulative closure ledger is malformed"
                )
            c_value = row.get("C")
            h_value = row.get("H")
            if (
                type(c_value) is not int
                or c_value < 0
                or type(h_value) is not int
                or h_value < 0
                or h_value > c_value
            ):
                raise PrelongSemanticLedgerError(
                    "ActionBall cumulative C/H row is malformed"
                )
            closed += c_value
            contacts += h_value
        return closed, contacts

    def _new_accumulators(self, device) -> Dict[str, Any]:
        torch = self._torch
        group_sums = {
            group: torch.zeros((), dtype=torch.float64, device=device)
            for group in PRELONG_REWARD_GROUPS
        }
        group_denominators = {
            group: torch.zeros((), dtype=torch.long, device=device)
            for group in PRELONG_REWARD_GROUPS
        }
        violation_counts = {
            "nonfinite_reward_cache": torch.zeros((), dtype=torch.long, device=device),
            "reward_buffer_closure": torch.zeros((), dtype=torch.long, device=device),
            "zero_weight_reward_cache_nonzero": torch.zeros(
                (), dtype=torch.long, device=device
            ),
            "diagnostic_probe_reward_cache_nonzero": torch.zeros(
                (), dtype=torch.long, device=device
            ),
            "task_income_outside_pre_step_eligibility": torch.zeros(
                (), dtype=torch.long, device=device
            ),
        }
        for group in PRELONG_REWARD_GROUPS:
            violation_counts[
                "%s_income_outside_pre_step_eligibility" % group
            ] = torch.zeros((), dtype=torch.long, device=device)
        bridge_values = None
        if self._bridge_enabled:
            bridge_values = {
                "mimic_eligible": {
                    binding["name"]: torch.zeros(
                        (), dtype=torch.long, device=device
                    )
                    for binding in self._mimic_bindings
                },
                "mimic_raw_sum": {
                    binding["name"]: torch.zeros(
                        (), dtype=torch.float64, device=device
                    )
                    for binding in self._mimic_bindings
                },
                "mimic_kernel_sum": {
                    binding["name"]: torch.zeros(
                        (), dtype=torch.float64, device=device
                    )
                    for binding in self._mimic_bindings
                },
                "mimic_error_sum": {
                    binding["name"]: torch.zeros(
                        (), dtype=torch.float64, device=device
                    )
                    for binding in self._mimic_bindings
                },
                "mimic_error_max": {
                    binding["name"]: torch.zeros(
                        (), dtype=torch.float64, device=device
                    )
                    for binding in self._mimic_bindings
                },
                "mimic_error_finite": {
                    binding["name"]: torch.zeros(
                        (), dtype=torch.long, device=device
                    )
                    for binding in self._mimic_bindings
                },
                "mimic_zero_kernel": {
                    binding["name"]: torch.zeros(
                        (), dtype=torch.long, device=device
                    )
                    for binding in self._mimic_bindings
                },
                "mimic_income": {
                    binding["name"]: torch.zeros(
                        (), dtype=torch.float64, device=device
                    )
                    for binding in self._mimic_bindings
                },
                "task_income": torch.zeros((), dtype=torch.float64, device=device),
                "progress_income": torch.zeros(
                    (), dtype=torch.float64, device=device
                ),
                "sample_count": torch.zeros((), dtype=torch.long, device=device),
                "safety_count": torch.zeros((), dtype=torch.long, device=device),
                # min hard gap, min root height, min upright
                "safety_min": torch.full(
                    (3,), float("inf"), dtype=torch.float64, device=device
                ),
                # max qvel ratio, max root height, max root xy speed, max foot slip
                "safety_max": torch.full(
                    (4,), float("-inf"), dtype=torch.float64, device=device
                ),
                # foot-contact sum, foot-slip sum
                "safety_sum": torch.zeros(
                    (2,), dtype=torch.float64, device=device
                ),
                "violation_counts": {
                    "nonfinite_or_invalid_safety": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "playback_start_tick_mismatch": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "bridge_task_income_not_allowed": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "mimic_raw_out_of_range": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "generation_or_validity_changed_without_terminal": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "wait_outside_sealed_cohort": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "duplicate_reveal_before_close": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "invalid_timing_tuple": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "ttc_off_policy_tick_grid": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                    "invalid_expected_bridge_ticks": torch.zeros(
                        (), dtype=torch.long, device=device
                    ),
                },
            }
        return {
            "group_sums": group_sums,
            "group_denominators": group_denominators,
            "invalid_samples": torch.zeros((), dtype=torch.long, device=device),
            "invalid_task_income": torch.zeros(
                (), dtype=torch.float64, device=device
            ),
            "invalid_task_denominator": torch.zeros(
                (), dtype=torch.long, device=device
            ),
            "ready_mimic_income": torch.zeros(
                (), dtype=torch.float64, device=device
            ),
            "ready_mimic_denominator": torch.zeros(
                (), dtype=torch.long, device=device
            ),
            "swing_mimic_income": torch.zeros(
                (), dtype=torch.float64, device=device
            ),
            "swing_mimic_denominator": torch.zeros(
                (), dtype=torch.long, device=device
            ),
            "exact_strike_timing_count": torch.zeros(
                (), dtype=torch.long, device=device
            ),
            "violation_counts": violation_counts,
            "bridge": bridge_values,
        }

    def _install_accumulators(self, values: Mapping[str, Any]) -> None:
        self._environment_step_count = 0
        self._group_sums = values["group_sums"]
        self._group_denominators = values["group_denominators"]
        self._invalid_samples = values["invalid_samples"]
        self._invalid_task_income = values["invalid_task_income"]
        self._invalid_task_denominator = values["invalid_task_denominator"]
        self._ready_mimic_income = values["ready_mimic_income"]
        self._ready_mimic_denominator = values["ready_mimic_denominator"]
        self._swing_mimic_income = values["swing_mimic_income"]
        self._swing_mimic_denominator = values["swing_mimic_denominator"]
        self._exact_strike_timing_count = values["exact_strike_timing_count"]
        self._violation_counts = values["violation_counts"]
        bridge = values["bridge"]
        if self._bridge_enabled:
            if not isinstance(bridge, Mapping):
                raise PrelongSemanticLedgerError(
                    "pre-long bridge accumulators are absent"
                )
            self._bridge_mimic_eligible = bridge["mimic_eligible"]
            self._bridge_mimic_raw_sum = bridge["mimic_raw_sum"]
            self._bridge_mimic_kernel_sum = bridge["mimic_kernel_sum"]
            self._bridge_mimic_error_sum = bridge["mimic_error_sum"]
            self._bridge_mimic_error_max = bridge["mimic_error_max"]
            self._bridge_mimic_error_finite = bridge["mimic_error_finite"]
            self._bridge_mimic_zero_kernel = bridge["mimic_zero_kernel"]
            self._bridge_mimic_income = bridge["mimic_income"]
            self._bridge_task_income = bridge["task_income"]
            self._bridge_progress_income = bridge["progress_income"]
            self._bridge_sample_count = bridge["sample_count"]
            self._bridge_safety_count = bridge["safety_count"]
            self._bridge_safety_min = bridge["safety_min"]
            self._bridge_safety_max = bridge["safety_max"]
            self._bridge_safety_sum = bridge["safety_sum"]
            self._bridge_violation_counts = bridge["violation_counts"]

    def _reset_accumulators(self, device) -> None:
        self._install_accumulators(self._new_accumulators(device))

    def _validate_runtime_bindings(self, step_reward) -> None:
        if tuple(getattr(self._manager, "active_terms", ())) != self._all_names:
            raise PrelongSemanticLedgerError(
                "RewardManager active_terms changed during a pre-long PPO window"
            )
        if tuple(step_reward.shape) != (
            PRELONG_NUM_ENVS,
            len(self._all_names),
        ):
            raise PrelongSemanticLedgerError(
                "RewardManager _step_reward shape changed during a pre-long PPO window"
            )
        for binding in self._bindings:
            cfg = self._get_term_cfg(binding["name"])
            weight = getattr(cfg, "weight", None)
            if type(weight) not in (int, float) or float(weight) != binding["weight"]:
                raise PrelongSemanticLedgerError(
                    "reward term %r weight changed during a PPO window"
                    % binding["name"]
                )
            if getattr(cfg, "func", None) is not binding["func"]:
                raise PrelongSemanticLedgerError(
                    "reward term %r callable changed during a PPO window"
                    % binding["name"]
                )
            if _normalize_runtime_term(binding["name"], cfg) != binding["recipe_term"]:
                raise PrelongSemanticLedgerError(
                    "reward term %r complete normalized parameters changed during a PPO window"
                    % binding["name"]
                )
            params = getattr(cfg, "params", None)
            if binding["required_params"] and (
                not isinstance(params, Mapping)
                or any(
                    params.get(key) != value
                    for key, value in binding["required_params"].items()
                )
            ):
                raise PrelongSemanticLedgerError(
                    "reward term %r required parameters changed during a PPO window"
                    % binding["name"]
                )

    def begin_environment_step(self):
        """Freeze all eligibility before RewardManager and command updates run."""

        if self._open_step is not None or self._pending_update is not None:
            raise PrelongSemanticLedgerError(
                "pre-long semantics already has an open/pending transaction"
            )
        if self._environment_step_count >= PRELONG_ROLLOUT_STEPS:
            raise PrelongSemanticLedgerError(
                "pre-long PPO window already contains 24 environment steps"
            )
        common_step = self._read_common_step_counter()
        expected = self._update_start_common_step_counter + self._environment_step_count
        if common_step != expected:
            raise PrelongSemanticLedgerError(
                "pre-long step token observed a skipped/foreign environment step"
            )
        masks = prelong_eligibility_masks(self._command, self.profile)
        task_valid = masks["task_valid"].detach().clone()
        group_masks = {
            group: masks["groups"][group].detach().clone()
            for group in PRELONG_REWARD_GROUPS
        }
        (
            bridge_mask,
            bridge_mimic_masks,
            bridge_mimic_scales,
        ) = self._begin_bridge_snapshot()
        token = _PrelongStepToken(
            ledger=self,
            sequence=self._next_step_sequence,
            common_step_counter=common_step,
            task_valid=task_valid,
            exact_strike_timing=(masks["exact_strike_timing"].detach().clone()),
            group_masks=group_masks,
            bridge_mask=bridge_mask.detach().clone(),
            bridge_mimic_masks={
                name: value.detach().clone()
                for name, value in bridge_mimic_masks.items()
            },
            bridge_mimic_scales={
                name: value.detach().clone()
                for name, value in bridge_mimic_scales.items()
            },
        )
        self._next_step_sequence += 1
        self._open_step = token
        return token

    def abort_environment_step(self, token=None) -> None:
        """Discard an interrupted step token; the runner re-raises the error."""

        if self._open_step is None:
            return
        if token is not None and token is not self._open_step:
            raise PrelongSemanticLedgerError("pre-long abort token is stale or foreign")
        self._open_step = None

    def _observe_bridge_after_step(self, token, step_reward) -> None:
        if not self._bridge_enabled:
            return
        torch = self._torch
        bridge = token.bridge_mask
        self._bridge_sample_count.add_(bridge.sum(dtype=torch.long))

        # Hidden WAIT is already guarded by the task-invalid zero-income
        # closure.  The public reveal->teacher-start bridge is stricter: A may
        # receive only racket_progress (base_position is absent/zero), while C
        # has no solved target and therefore receives no task income at all.
        if self._task_indices:
            task_columns = step_reward[:, list(self._task_indices)]
            task_rate = task_columns.sum(dim=1)
            self._bridge_task_income.add_(
                torch.sum(
                    task_rate * bridge.to(dtype=task_rate.dtype) * self._step_dt,
                    dtype=torch.float64,
                )
            )
            allowed = set(self._bridge_allowed_task_indices)
            disallowed_local = [
                local
                for local, global_index in enumerate(self._task_indices)
                if global_index not in allowed
            ]
            if disallowed_local:
                self._bridge_violation_counts[
                    "bridge_task_income_not_allowed"
                ].add_(
                    (
                        task_columns[:, disallowed_local][bridge] != 0.0
                    ).sum(dtype=torch.long)
                )
            if self._bridge_allowed_task_indices:
                progress = step_reward[
                    :, self._bridge_allowed_task_indices[0]
                ]
                self._bridge_progress_income.add_(
                    torch.sum(
                        progress
                        * bridge.to(dtype=progress.dtype)
                        * self._step_dt,
                        dtype=torch.float64,
                    )
                )

        for binding in self._mimic_bindings:
            name = binding["name"]
            mask = token.bridge_mimic_masks[name]
            scale = token.bridge_mimic_scales[name].to(
                dtype=step_reward.dtype
            )
            contribution = step_reward[:, binding["index"]]
            raw = contribution / float(binding["weight"])
            params = getattr(self._get_term_cfg(name), "params")
            std = float(params["std"])
            outside = bridge & ~mask & (raw != 0.0)
            malformed = mask & (
                ~torch.isfinite(raw)
                | (raw < 0.0)
                | (raw > scale + 1.0e-6)
            )
            self._bridge_violation_counts["mimic_raw_out_of_range"].add_(
                (outside | malformed).sum(dtype=torch.long)
            )
            self._bridge_mimic_eligible[name].add_(mask.sum(dtype=torch.long))
            self._bridge_mimic_income[name].add_(
                torch.sum(
                    contribution
                    * mask.to(dtype=contribution.dtype)
                    * self._step_dt,
                    dtype=torch.float64,
                )
            )
            safe_raw = torch.where(mask, raw, torch.zeros_like(raw))
            self._bridge_mimic_raw_sum[name].add_(
                safe_raw.sum(dtype=torch.float64)
            )
            kernel = torch.where(
                mask,
                raw / scale.clamp_min(1.0e-30),
                torch.zeros_like(raw),
            )
            kernel = kernel.clamp(min=0.0, max=1.0)
            self._bridge_mimic_kernel_sum[name].add_(
                kernel.sum(dtype=torch.float64)
            )
            positive = mask & (kernel > 0.0)
            self._bridge_mimic_zero_kernel[name].add_(
                (mask & ~positive).sum(dtype=torch.long)
            )
            kernel_tiny = torch.finfo(kernel.dtype).tiny
            if name in _PRELONG_MIMIC_EXP_TERMS:
                error = std * torch.sqrt(
                    torch.clamp(
                        -torch.log(kernel.clamp_min(kernel_tiny)), min=0.0
                    )
                )
            elif name in _PRELONG_MIMIC_CAUCHY_TERMS:
                error = std * torch.sqrt(
                    torch.clamp(
                        torch.reciprocal(kernel.clamp_min(kernel_tiny)) - 1.0,
                        min=0.0,
                    )
                )
            else:  # construction taxonomy should make this unreachable
                raise PrelongSemanticLedgerError(
                    "bridge mimic term %r has no error inversion" % name
                )
            finite_error = positive & torch.isfinite(error)
            self._bridge_mimic_error_finite[name].add_(
                finite_error.sum(dtype=torch.long)
            )
            self._bridge_mimic_error_sum[name].add_(
                torch.where(finite_error, error, torch.zeros_like(error)).sum(
                    dtype=torch.float64
                )
            )
            finite_max = torch.where(
                finite_error,
                error,
                torch.zeros_like(error),
            ).amax()
            self._bridge_mimic_error_max[name].copy_(
                torch.maximum(
                    self._bridge_mimic_error_max[name],
                    finite_max.to(dtype=torch.float64),
                )
            )

        manager = getattr(self._env, "termination_manager", None)
        terminated = getattr(manager, "terminated", None)
        terminated = _require_bool_vector(
            torch,
            terminated,
            name="bridge terminal mask",
            num_envs=PRELONG_NUM_ENVS,
        )
        if terminated.device != bridge.device:
            raise PrelongSemanticLedgerError(
                "bridge terminal mask must share the reward device"
            )
        terminal_before_start = bridge & terminated & self._bridge_active
        cohort = (
            self._bridge_wait_ticks[terminal_before_start]
            - _PRELONG_BRIDGE_WAIT_MIN_TICKS
        )
        self._bridge_lifetime_terminal.add_(
            torch.bincount(
                cohort,
                minlength=len(_PRELONG_BRIDGE_WAIT_COHORTS),
            )
        )
        self._bridge_active[terminal_before_start] = False

    def observe_after_environment_step(self, token) -> None:
        """Consume raw*weight cache income for the exact frozen pre-step masks."""

        if token is not self._open_step or getattr(token, "ledger", None) is not self:
            raise PrelongSemanticLedgerError(
                "pre-long environment step token is stale or foreign"
            )
        if self._read_common_step_counter() != token.common_step_counter + 1:
            raise PrelongSemanticLedgerError(
                "pre-long environment step did not advance common_step_counter once"
            )
        torch = self._torch
        step_reward = getattr(self._manager, "_step_reward", None)
        reward_buf = getattr(self._manager, "_reward_buf", None)
        if not isinstance(step_reward, torch.Tensor) or not isinstance(
            reward_buf, torch.Tensor
        ):
            raise PrelongSemanticLedgerError(
                "RewardManager cache disappeared during a pre-long PPO window"
            )
        self._validate_runtime_bindings(step_reward)
        if tuple(reward_buf.shape) != (PRELONG_NUM_ENVS,):
            raise PrelongSemanticLedgerError(
                "RewardManager _reward_buf shape changed during a PPO window"
            )
        if step_reward.device != token.task_valid.device or reward_buf.device != (
            token.task_valid.device
        ):
            raise PrelongSemanticLedgerError(
                "pre-long reward cache and eligibility token changed device"
            )

        with torch.inference_mode():
            step_reward = step_reward.detach()
            reward_buf = reward_buf.detach()
            nonfinite = (~torch.isfinite(step_reward)).sum(dtype=torch.long)
            nonfinite = nonfinite + (~torch.isfinite(reward_buf)).sum(dtype=torch.long)
            self._violation_counts["nonfinite_reward_cache"].add_(nonfinite)

            recomposed = torch.sum(step_reward, dim=1) * self._step_dt
            closure_error = torch.abs(recomposed - reward_buf)
            closure_scale = torch.sum(torch.abs(step_reward), dim=1) * (self._step_dt)
            closure_tolerance = closure_scale * self._CLOSURE_RTOL + self._CLOSURE_ATOL
            self._violation_counts["reward_buffer_closure"].add_(
                (closure_error > closure_tolerance).sum(dtype=torch.long)
            )
            if self._zero_weight_indices:
                self._violation_counts["zero_weight_reward_cache_nonzero"].add_(
                    (step_reward[:, list(self._zero_weight_indices)] != 0.0).sum(
                        dtype=torch.long
                    )
                )
            self._violation_counts["diagnostic_probe_reward_cache_nonzero"].add_(
                (step_reward[:, list(self._probe_indices)] != 0.0).sum(dtype=torch.long)
            )

            group_rates = {}
            for group in PRELONG_REWARD_GROUPS:
                indices = self._group_indices[group]
                if indices:
                    group_columns = step_reward[:, list(indices)]
                    rate = torch.sum(group_columns, dim=1)
                else:
                    group_columns = None
                    rate = torch.zeros_like(reward_buf)
                group_rates[group] = rate
                mask = token.group_masks[group]
                self._group_sums[group].add_(
                    torch.sum(rate * self._step_dt, dtype=torch.float64)
                )
                self._group_denominators[group].add_(mask.sum(dtype=torch.long))
                self._violation_counts[
                    "%s_income_outside_pre_step_eligibility" % group
                ].add_(
                    (
                        ((group_columns != 0.0) & (~mask).unsqueeze(1)).sum(
                            dtype=torch.long
                        )
                        if group_columns is not None
                        else torch.zeros((), dtype=torch.long, device=reward_buf.device)
                    )
                )

            # Mimic rewards remain active during both hidden ready/wait and
            # task-valid swing.  Freeze the split from the same pre-step
            # ``task_valid`` snapshot used by every other eligibility check;
            # this is accounting only and never masks or changes reward.
            ready_mimic_mask = ~token.task_valid
            swing_mimic_mask = token.task_valid
            mimic_rate = group_rates["mimic"]
            self._ready_mimic_income.add_(
                torch.sum(
                    mimic_rate
                    * ready_mimic_mask.to(dtype=mimic_rate.dtype)
                    * self._step_dt,
                    dtype=torch.float64,
                )
            )
            self._ready_mimic_denominator.add_(
                ready_mimic_mask.sum(dtype=torch.long)
            )
            self._swing_mimic_income.add_(
                torch.sum(
                    mimic_rate
                    * swing_mimic_mask.to(dtype=mimic_rate.dtype)
                    * self._step_dt,
                    dtype=torch.float64,
                )
            )
            self._swing_mimic_denominator.add_(
                swing_mimic_mask.sum(dtype=torch.long)
            )

            if self._task_indices:
                task_columns = step_reward[:, list(self._task_indices)]
                task_rate = torch.sum(task_columns, dim=1)
            else:
                task_columns = None
                task_rate = torch.zeros_like(reward_buf)
            task_union = (
                token.group_masks["strike"]
                | token.group_masks["target"]
                | token.group_masks["outcome"]
            )
            invalid = ready_mimic_mask
            self._invalid_samples.add_(invalid.sum(dtype=torch.long))
            self._exact_strike_timing_count.add_(
                token.exact_strike_timing.sum(dtype=torch.long)
            )
            self._invalid_task_income.add_(
                torch.sum(
                    task_rate * invalid.to(dtype=task_rate.dtype) * self._step_dt,
                    dtype=torch.float64,
                )
            )
            self._invalid_task_denominator.add_(
                (task_union & invalid).sum(dtype=torch.long)
            )
            self._violation_counts["task_income_outside_pre_step_eligibility"].add_(
                (
                    ((task_columns != 0.0) & (~task_union).unsqueeze(1)).sum(
                        dtype=torch.long
                    )
                    if task_columns is not None
                    else torch.zeros((), dtype=torch.long, device=reward_buf.device)
                )
            )
            self._observe_bridge_after_step(token, step_reward)

        self._environment_step_count += 1
        self._open_step = None

    def _prepare_bridge_telemetry(self):
        if not self._bridge_enabled:
            return {
                "status": "not_configured",
                "reason": "ActionBall task-wait schedule is absent",
            }
        torch = self._torch
        censored = torch.bincount(
            self._bridge_wait_ticks[self._bridge_active]
            - _PRELONG_BRIDGE_WAIT_MIN_TICKS,
            minlength=len(_PRELONG_BRIDGE_WAIT_COHORTS),
        )
        cohort_packet = (
            torch.stack(
                (
                    self._bridge_lifetime_reveal,
                    self._bridge_lifetime_start,
                    self._bridge_lifetime_terminal,
                    censored,
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        cohort_rows = []
        for wait_ticks, raw in zip(_PRELONG_BRIDGE_WAIT_COHORTS, cohort_packet):
            reveal, start, terminal, active = (int(value) for value in raw)
            if reveal != start + terminal + active:
                raise PrelongSemanticLedgerError(
                    "bridge conservation failed for WAIT=%d: %d != %d+%d+%d"
                    % (wait_ticks, reveal, start, terminal, active)
                )
            cohort_rows.append(
                {
                    "wait_ticks": wait_ticks,
                    "reveal_count": reveal,
                    "playback_start_count": start,
                    "terminal_before_start_count": terminal,
                    "censored_count": active,
                }
            )

        timing_count = int(self._bridge_timing_count.detach().cpu().item())
        reveal_total = sum(row["reveal_count"] for row in cohort_rows)
        if timing_count != reveal_total:
            raise PrelongSemanticLedgerError(
                "bridge timing tuple count differs from reveal count"
            )
        timing_rows = {}
        if timing_count:
            timing_packet = (
                torch.stack(
                    (
                        self._bridge_timing_sum,
                        self._bridge_timing_min,
                        self._bridge_timing_max,
                    ),
                    dim=1,
                )
                .detach()
                .cpu()
                .tolist()
            )
            for name, values in zip(self._bridge_timing_names, timing_packet):
                total, minimum, maximum = (float(value) for value in values)
                if not all(math.isfinite(value) for value in (total, minimum, maximum)):
                    raise PrelongSemanticLedgerError(
                        "bridge %s timing aggregate is non-finite" % name
                    )
                timing_rows[name] = {
                    "mean": total / timing_count,
                    "min": minimum,
                    "max": maximum,
                }
        else:
            timing_rows = {
                name: {"mean": None, "min": None, "max": None}
                for name in self._bridge_timing_names
            }

        mimic_rows = []
        for binding in self._mimic_bindings:
            name = binding["name"]
            packet = torch.stack(
                (
                    self._bridge_mimic_eligible[name].to(dtype=torch.float64),
                    self._bridge_mimic_raw_sum[name],
                    self._bridge_mimic_kernel_sum[name],
                    self._bridge_mimic_error_finite[name].to(dtype=torch.float64),
                    self._bridge_mimic_zero_kernel[name].to(dtype=torch.float64),
                    self._bridge_mimic_error_sum[name],
                    self._bridge_mimic_error_max[name],
                    self._bridge_mimic_income[name],
                )
            ).detach().cpu().tolist()
            (
                eligible_f,
                raw_sum,
                kernel_sum,
                error_finite_f,
                zero_kernel_f,
                error_sum,
                error_max,
                income,
            ) = (float(value) for value in packet)
            eligible = int(eligible_f)
            error_finite = int(error_finite_f)
            zero_kernel = int(zero_kernel_f)
            if error_finite + zero_kernel != eligible:
                raise PrelongSemanticLedgerError(
                    "bridge mimic %s error denominators do not conserve" % name
                )
            if not all(
                math.isfinite(value)
                for value in (raw_sum, kernel_sum, error_sum, error_max, income)
            ):
                raise PrelongSemanticLedgerError(
                    "bridge mimic %s aggregate is non-finite" % name
                )
            params = getattr(self._get_term_cfg(name), "params")
            mimic_rows.append(
                {
                    "term": name,
                    "kernel": (
                        "exp_negative_squared_error_over_std_squared"
                        if name in _PRELONG_MIMIC_EXP_TERMS
                        else "cauchy_one_over_one_plus_error_over_std_squared"
                    ),
                    "error_semantics": (
                        "std*sqrt(-ln(kernel))"
                        if name in _PRELONG_MIMIC_EXP_TERMS
                        else "std*sqrt(kernel^-1-1)"
                    ),
                    "std": float(params["std"]),
                    "eligible_denominator": eligible,
                    "raw_reward_sum_before_manager_weight": raw_sum,
                    "raw_kernel_sum_after_window_scale_removed": kernel_sum,
                    "finite_error_denominator": error_finite,
                    "zero_kernel_count": zero_kernel,
                    "error_mean": (
                        None if error_finite == 0 else error_sum / error_finite
                    ),
                    "error_max": None if error_finite == 0 else error_max,
                    "weighted_income_sum": income,
                    "income_semantics": "raw_reward_times_manager_weight_times_policy_dt",
                }
            )

        task_income = float(self._bridge_task_income.detach().cpu().item())
        progress_income = float(
            self._bridge_progress_income.detach().cpu().item()
        )
        if not math.isfinite(task_income) or not math.isfinite(progress_income):
            raise PrelongSemanticLedgerError(
                "bridge task-income aggregate is non-finite"
            )
        if self.profile == PRELONG_PROFILE_C211:
            if task_income != 0.0 or progress_income != 0.0:
                raise PrelongSemanticLedgerError(
                    "C211 reveal bridge must have exactly zero task income"
                )
            task_rule = "all_task_income_exact_zero"
        else:
            if not math.isclose(
                task_income, progress_income, rel_tol=1.0e-12, abs_tol=1.0e-9
            ):
                raise PrelongSemanticLedgerError(
                    "A211 reveal bridge task income must be racket_progress only"
                )
            task_rule = "racket_progress_only_base_position_absent_or_zero"

        safety_count = int(self._bridge_safety_count.detach().cpu().item())
        bridge_samples = int(self._bridge_sample_count.detach().cpu().item())
        if safety_count != bridge_samples:
            raise PrelongSemanticLedgerError(
                "bridge safety denominator differs from bridge sample count"
            )
        if safety_count:
            safety_min = self._bridge_safety_min.detach().cpu().tolist()
            safety_max = self._bridge_safety_max.detach().cpu().tolist()
            safety_sum = self._bridge_safety_sum.detach().cpu().tolist()
            if not all(
                math.isfinite(float(value))
                for value in (*safety_min, *safety_max, *safety_sum)
            ):
                raise PrelongSemanticLedgerError(
                    "bridge safety aggregate is non-finite"
                )
            safety = {
                "sample_count": safety_count,
                "minimum_physical_hard_gap_rad": float(safety_min[0]),
                "maximum_abs_qvel_over_physical_limit": float(safety_max[0]),
                "minimum_root_height_m": float(safety_min[1]),
                "maximum_root_height_m": float(safety_max[1]),
                "minimum_root_upright_cosine": float(safety_min[2]),
                "maximum_root_xy_speed_mps": float(safety_max[2]),
                "mean_foot_contact_fraction": float(safety_sum[0]) / safety_count,
                "mean_foot_slip_speed_mps": float(safety_sum[1]) / safety_count,
                "maximum_foot_slip_speed_mps": float(safety_max[3]),
                "sampling_semantics": (
                    "device-side policy-boundary state at every revealed pre-playback bridge step"
                ),
            }
        else:
            safety = {
                "sample_count": 0,
                "minimum_physical_hard_gap_rad": None,
                "maximum_abs_qvel_over_physical_limit": None,
                "minimum_root_height_m": None,
                "maximum_root_height_m": None,
                "minimum_root_upright_cosine": None,
                "maximum_root_xy_speed_mps": None,
                "mean_foot_contact_fraction": None,
                "mean_foot_slip_speed_mps": None,
                "maximum_foot_slip_speed_mps": None,
                "sampling_semantics": (
                    "device-side policy-boundary state at every revealed pre-playback bridge step"
                ),
            }

        return {
            "status": "active_fail_closed",
            "authority": dict(self._bridge_authority),
            "lifetime_conservation": {
                "equation": (
                    "reveal_count=playback_start_count+terminal_before_start_count+censored_count"
                ),
                "wait_cohorts": cohort_rows,
            },
            "timing_at_reveal": {
                "reveal_count": timing_count,
                "fields": timing_rows,
                "expected_bridge_tick_rule": (
                    "floor(pre_swing_wait_s/policy_dt_s)+1; playback starts on age>wait"
                ),
            },
            "window": {
                "bridge_sample_count": bridge_samples,
                "task_income_rule": task_rule,
                "task_weighted_income_sum": task_income,
                "racket_progress_weighted_income_sum": progress_income,
                "hidden_wait_task_income_required": 0.0,
                "mimic_terms": mimic_rows,
                "safety": safety,
            },
            "performance_contract": (
                "all per-step accumulation is device-side; one compact host transfer occurs at PPO prepare"
            ),
        }

    def prepare_update(self, ppo_update: int):
        """Validate and freeze one marker until the optimizer succeeds."""

        if type(ppo_update) is not int or ppo_update < 0:
            raise PrelongSemanticLedgerError(
                "pre-long PPO update must be a nonnegative plain integer"
            )
        if self._open_step is not None or self._pending_update is not None:
            raise PrelongSemanticLedgerError(
                "pre-long semantics has an open/pending transaction"
            )
        if self._last_finished_update is not None and ppo_update != (
            self._last_finished_update + 1
        ):
            raise PrelongSemanticLedgerError(
                "pre-long PPO update sequence is not contiguous"
            )
        if self._environment_step_count != PRELONG_ROLLOUT_STEPS:
            raise PrelongSemanticLedgerError(
                "pre-long ledger observed %d env.step calls, expected 24"
                % self._environment_step_count
            )
        end_common_step = self._read_common_step_counter()
        if end_common_step - self._update_start_common_step_counter != (
            PRELONG_ROLLOUT_STEPS
        ):
            raise PrelongSemanticLedgerError(
                "pre-long common_step_counter does not close the 24-step window"
            )

        violation_names = tuple(sorted(self._violation_counts))
        violation_values = (
            self._torch.stack(
                tuple(self._violation_counts[name] for name in violation_names)
            )
            .detach()
            .cpu()
            .tolist()
        )
        active_violations = {
            name: int(value)
            for name, value in zip(violation_names, violation_values)
            if int(value) != 0
        }
        if self._bridge_enabled:
            bridge_violation_names = tuple(sorted(self._bridge_violation_counts))
            bridge_violation_values = (
                self._torch.stack(
                    tuple(
                        self._bridge_violation_counts[name]
                        for name in bridge_violation_names
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
            active_violations.update(
                {
                    "bridge_%s" % name: int(value)
                    for name, value in zip(
                        bridge_violation_names, bridge_violation_values
                    )
                    if int(value) != 0
                }
            )
        if active_violations:
            raise PrelongSemanticLedgerError(
                "pre-long reward-cache/eligibility closure failed: %r"
                % active_violations
            )

        group_sums = (
            self._torch.stack(
                tuple(self._group_sums[group] for group in PRELONG_REWARD_GROUPS)
            )
            .detach()
            .cpu()
            .tolist()
        )
        group_denominators = (
            self._torch.stack(
                tuple(
                    self._group_denominators[group] for group in PRELONG_REWARD_GROUPS
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        invalid_packet = (
            self._torch.stack(
                (
                    self._invalid_samples,
                    self._invalid_task_denominator,
                    self._exact_strike_timing_count,
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        invalid_income = float(self._invalid_task_income.detach().cpu().item())
        if not math.isfinite(invalid_income) or invalid_income != 0.0:
            raise PrelongSemanticLedgerError(
                "task_valid=0 produced non-zero or non-finite task income"
            )
        mimic_phase_denominators = (
            self._torch.stack(
                (
                    self._ready_mimic_denominator,
                    self._swing_mimic_denominator,
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        mimic_phase_incomes = (
            self._torch.stack(
                (
                    self._ready_mimic_income,
                    self._swing_mimic_income,
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        ready_mimic_income = float(mimic_phase_incomes[0])
        swing_mimic_income = float(mimic_phase_incomes[1])
        if not math.isfinite(ready_mimic_income) or not math.isfinite(
            swing_mimic_income
        ):
            raise PrelongSemanticLedgerError(
                "pre-long ready/swing mimic income is non-finite"
            )
        mimic_group_index = PRELONG_REWARD_GROUPS.index("mimic")
        if int(mimic_phase_denominators[0]) != int(invalid_packet[0]):
            raise PrelongSemanticLedgerError(
                "task-invalid ready-mimic denominator lost task_valid alignment"
            )
        if (
            int(mimic_phase_denominators[0])
            + int(mimic_phase_denominators[1])
            != int(group_denominators[mimic_group_index])
        ):
            raise PrelongSemanticLedgerError(
                "ready/swing mimic denominators do not exhaust aggregate mimic eligibility"
            )
        if not math.isclose(
            ready_mimic_income + swing_mimic_income,
            float(group_sums[mimic_group_index]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ):
            raise PrelongSemanticLedgerError(
                "ready/swing mimic income does not exhaust aggregate mimic income"
            )

        closure_end = self._read_closure_totals()
        closed = closure_end[0] - self._closure_start_totals[0]
        contacts = closure_end[1] - self._closure_start_totals[1]
        if closed < 0 or contacts < 0 or contacts > closed:
            raise PrelongSemanticLedgerError(
                "ActionBall cumulative C/H ledger did not close monotonically"
            )

        counters: Dict[str, Any] = {
            TASK_INVALID_OBSERVED_COUNTER: int(invalid_packet[0]),
            TASK_INVALID_REWARD_SUM_COUNTER: (
                0.0 if invalid_income == 0.0 else invalid_income
            ),
            TASK_INVALID_REWARD_ELIGIBLE_COUNTER: int(invalid_packet[1]),
            READY_MIMIC_REWARD_SUM_COUNTER: (
                0.0 if ready_mimic_income == 0.0 else ready_mimic_income
            ),
            READY_MIMIC_ELIGIBLE_COUNTER: int(mimic_phase_denominators[0]),
            SWING_MIMIC_REWARD_SUM_COUNTER: (
                0.0 if swing_mimic_income == 0.0 else swing_mimic_income
            ),
            SWING_MIMIC_ELIGIBLE_COUNTER: int(mimic_phase_denominators[1]),
            EXACT_STRIKE_TIMING_COUNTER: int(invalid_packet[2]),
            ELIGIBLE_CLOSED_SWING_COUNTER: int(closed),
            ACTUAL_CONTACT_COUNTER: int(contacts),
            ACHIEVED_FLIGHT_COUNTER: int(
                group_denominators[PRELONG_REWARD_GROUPS.index("outcome")]
            ),
            UNKNOWN_ATTRIBUTION_COUNTER: 0,
        }
        for index, group in enumerate(PRELONG_REWARD_GROUPS):
            income = float(group_sums[index])
            if not math.isfinite(income):
                raise PrelongSemanticLedgerError(
                    "pre-long %s income is non-finite" % group
                )
            counters[reward_group_sum_counter(group)] = 0.0 if income == 0.0 else income
            counters[reward_group_eligible_counter(group)] = int(
                group_denominators[index]
            )

        bridge_telemetry = self._prepare_bridge_telemetry()
        record = build_prelong_semantics_update(
            ppo_update=ppo_update,
            counters=counters,
            profile=self.profile,
            bridge_telemetry=bridge_telemetry,
        )
        marker_line = prelong_semantics_marker_line(
            ppo_update=ppo_update,
            counters=counters,
            profile=self.profile,
            bridge_telemetry=bridge_telemetry,
        )
        prepared = _PrelongPreparedUpdate(
            ledger=self,
            ppo_update=ppo_update,
            counters=counters,
            record=record,
            marker_line=marker_line,
            end_common_step_counter=end_common_step,
            end_closure_totals=closure_end,
        )
        self._pending_update = {
            "token": prepared,
            "ppo_update": ppo_update,
            "counters": dict(counters),
            "marker_line": marker_line,
            "end_common_step_counter": end_common_step,
            "end_closure_totals": closure_end,
        }
        return prepared

    def _validated_pending_update(self, prepared):
        pending = self._pending_update
        if (
            not isinstance(pending, Mapping)
            or prepared is not pending.get("token")
            or getattr(prepared, "ledger", None) is not self
        ):
            raise PrelongSemanticLedgerError(
                "pre-long prepared update token is stale or foreign"
            )
        if (
            prepared.ppo_update != pending["ppo_update"]
            or prepared.counters != pending["counters"]
            or prepared.marker_line != pending["marker_line"]
            or prepared.end_common_step_counter != pending["end_common_step_counter"]
            or prepared.end_closure_totals != pending["end_closure_totals"]
        ):
            raise PrelongSemanticLedgerError(
                "prepared pre-long marker mutated before acknowledgement"
            )
        return pending

    def marker_line_for(self, prepared) -> str:
        """Read a frozen marker without consuming its post-optimizer transaction."""

        return self._validated_pending_update(prepared)["marker_line"]

    def prepare_acknowledgement(self, prepared):
        """Seal every fallible post-service check before marker emission."""

        pending = self._validated_pending_update(prepared)
        current_common_step = self._read_common_step_counter()
        if current_common_step != pending["end_common_step_counter"]:
            raise PrelongSemanticLedgerError(
                "environment stepped between pre-long preparation and acknowledgement"
            )
        current_closure_totals = self._read_closure_totals()
        boundary_closed = current_closure_totals[0] - pending["end_closure_totals"][0]
        boundary_contacts = current_closure_totals[1] - pending["end_closure_totals"][1]
        if (
            boundary_closed < 0
            or boundary_contacts < 0
            or boundary_contacts > boundary_closed
        ):
            raise PrelongSemanticLedgerError(
                "ActionBall C/H ledger changed non-monotonically after rollout preparation"
            )
        device = getattr(self._manager, "_step_reward").device
        fresh_accumulators = self._new_accumulators(device)
        return _PrelongPreparedAcknowledgement(
            ledger=self,
            ppo_update=pending["ppo_update"],
            marker_line=pending["marker_line"],
            common_step_counter=current_common_step,
            # Boundary services may close/reset an attempt after the optimizer.
            # It belongs to neither adjacent rollout, so the next baseline is
            # the fully validated current cumulative ledger.
            closure_totals=current_closure_totals,
            fresh_accumulators=fresh_accumulators,
        )

    def acknowledge_update(self, prepared) -> str:
        """Compatibility helper for non-emitting callers and host tests."""

        return self.prepare_acknowledgement(prepared).consume()
