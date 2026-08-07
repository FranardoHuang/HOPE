"""One canonical schema-v3 ``reveal_to_playback_bridge`` record, built in one place.

人话:自 2026-08-07 起,共享的 `4096x5` pre-long gate 严格消费这块记录,于是
**三个**测试模块都要喂它 —— 共享 gate 自己的测试,加上 A211 / C211 两个 launcher 的
终局门测试。三份手抄的下场是改一处忘两处,而这块记录正是"生产方与消费方必须同版本"
的那一块,漂了不会报错、只会让门悄悄换一个拒绝面。所以这里只留一份。

字段名/档位/核函数归类**从生产方取活值**(``action_ball_prelong_semantics``),不手抄:
这个 fixture 扮演的就是生产方,取值权威自然该是生产方。被测的是 gate 那边的常量表。

注意这是**输入**不是**期望值**:它只负责长得像一份真收据,断言写在各自的测试里。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


_SEMANTICS_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "action_ball_prelong_semantics.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_prelong_bridge_fixture_semantics", _SEMANTICS_SOURCE
)
assert _SPEC is not None and _SPEC.loader is not None
_SEMANTICS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SEMANTICS)

WAIT_COHORTS = tuple(_SEMANTICS._PRELONG_BRIDGE_WAIT_COHORTS)
MIMIC_TERMS = tuple(
    _SEMANTICS.prelong_group_term_weights(_SEMANTICS.PRELONG_PROFILE_A211)["mimic"]
)
CAUCHY_TERMS = frozenset(_SEMANTICS._PRELONG_MIMIC_CAUCHY_TERMS)

#: 一块窗口里被记账的样本数。桥的安全/mimic 分母都拴在它上面。
BRIDGE_SAMPLE_COUNT = 2048

#: ``motion_body_pos`` / ``motion_body_ori`` 是仅有的两项允许分母小于窗口的 mimic。
PARTIAL_ELIGIBILITY_TERMS = ("motion_body_pos", "motion_body_ori")
PARTIAL_ELIGIBLE_COUNT = 2000

#: 每个都带字母,这样"把一位十六进制改成大写"才是一次真的变异
#: (全数字的假 SHA 会让 ``.upper()`` 变成空操作,变异测试就成了自证)。
AUTHORITY_SHA256 = {
    "timing_contract_sha256": "1a" * 32,
    "question_sha256": "2b" * 32,
    "sampler_contract_sha256": "3c" * 32,
    "effective_reward_recipe_sha256": "4d" * 32,
    "wait_schedule_sha256": "5e" * 32,
}


def cohort_counts(update: int, cohort_index: int) -> dict[str, int]:
    """One WAIT 档在第 ``update`` 个 PPO update 上的四个计数。

    构造成逐 update 严格增长:门要求 ``reveal``/``start``/``terminal`` 跨 update 单调
    不减,且整窗至少有一行新揭示。恒定计数会被"没有新揭示的行"那条拒掉。
    """

    reveal = 10 + cohort_index + update
    start = 6 + cohort_index + update
    terminal = 2
    return {
        "wait_ticks": WAIT_COHORTS[cohort_index],
        "reveal_count": reveal,
        "playback_start_count": start,
        "terminal_before_start_count": terminal,
        "censored_count": reveal - start - terminal,
    }


def total_reveal_count(update: int) -> int:
    return sum(
        cohort_counts(update, index)["reveal_count"]
        for index in range(len(WAIT_COHORTS))
    )


def _mimic_row(term: str) -> dict[str, Any]:
    cauchy = term in CAUCHY_TERMS
    eligible = (
        PARTIAL_ELIGIBLE_COUNT
        if term in PARTIAL_ELIGIBILITY_TERMS
        else BRIDGE_SAMPLE_COUNT
    )
    zero_kernel = 8
    return {
        "term": term,
        "kernel": (
            "cauchy_one_over_one_plus_error_over_std_squared"
            if cauchy
            else "exp_negative_squared_error_over_std_squared"
        ),
        "error_semantics": (
            "std*sqrt(kernel^-1-1)" if cauchy else "std*sqrt(-ln(kernel))"
        ),
        "std": 0.3,
        "eligible_denominator": eligible,
        "raw_reward_sum_before_manager_weight": float(eligible) / 2.0,
        "raw_kernel_sum_after_window_scale_removed": float(eligible) / 2.0,
        "finite_error_denominator": eligible - zero_kernel,
        "zero_kernel_count": zero_kernel,
        "error_mean": 0.05,
        "error_max": 0.41,
        "weighted_income_sum": 12.5,
        "income_semantics": "raw_reward_times_manager_weight_times_policy_dt",
    }


def reveal_bridge(update: int, *, profile: str) -> dict[str, Any]:
    """Return one canonical ``active_fail_closed`` bridge record."""

    if profile == _SEMANTICS.PRELONG_PROFILE_A211:
        target_source, target_recipe = "online_solver", "current_lm"
        task_rule = "racket_progress_only_base_position_absent_or_zero"
        task_income = progress_income = 0.75
    elif profile == _SEMANTICS.PRELONG_PROFILE_C211:
        target_source, target_recipe = "direct_ball", "outcome_dense_only"
        task_rule = "all_task_income_exact_zero"
        task_income = progress_income = 0.0
    else:  # pragma: no cover - fixture misuse
        raise AssertionError(profile)
    return {
        "status": "active_fail_closed",
        "authority": {
            "family": "backhand",
            "target_source": target_source,
            "target_recipe": target_recipe,
            "timing_authority": "current_center_task_receipt",
            "question_sha_semantics": (
                "exact per-env installed question payload re-hashed per row"
            ),
            "profile": profile,
            "wait_cohort_ticks": list(WAIT_COHORTS),
            "policy_dt_s": 0.02,
            **AUTHORITY_SHA256,
        },
        "lifetime_conservation": {
            "equation": (
                "reveal_count=playback_start_count+terminal_before_start_count"
                "+censored_count"
            ),
            "wait_cohorts": [
                cohort_counts(update, index) for index in range(len(WAIT_COHORTS))
            ],
        },
        "timing_at_reveal": {
            "reveal_count": total_reveal_count(update),
            "fields": {
                "time_to_contact_tick": {"mean": 91.0, "min": 88.0, "max": 95.0},
                "teacher_rate": {"mean": 1.0, "min": 1.0, "max": 1.0},
                "scaled_t_hit_s": {"mean": 1.82, "min": 1.80, "max": 1.84},
                "pre_swing_wait_s": {
                    "mean": 0.6923799138976297,
                    "min": 0.6923799138976297,
                    "max": 0.6923799138976297,
                },
                "expected_bridge_ticks": {"mean": 35.0, "min": 35.0, "max": 35.0},
            },
            "expected_bridge_tick_rule": (
                "floor(pre_swing_wait_s/policy_dt_s)+1; playback starts on age>wait"
            ),
        },
        "window": {
            "bridge_sample_count": BRIDGE_SAMPLE_COUNT,
            "task_income_rule": task_rule,
            "task_weighted_income_sum": task_income,
            "racket_progress_weighted_income_sum": progress_income,
            "hidden_wait_task_income_required": 0.0,
            "mimic_terms": [_mimic_row(term) for term in MIMIC_TERMS],
            "safety": {
                "sample_count": BRIDGE_SAMPLE_COUNT,
                "minimum_physical_hard_gap_rad": 0.12,
                "maximum_abs_qvel_over_physical_limit": 0.63,
                "minimum_root_height_m": 0.71,
                "maximum_root_height_m": 0.79,
                "minimum_root_upright_cosine": 0.98,
                "maximum_root_xy_speed_mps": 0.21,
                "mean_foot_contact_fraction": 0.99,
                "mean_foot_slip_speed_mps": 0.004,
                "maximum_foot_slip_speed_mps": 0.031,
                "sampling_semantics": (
                    "device-side policy-boundary state at every revealed "
                    "pre-playback bridge step"
                ),
            },
        },
        "performance_contract": (
            "all per-step accumulation is device-side; one compact host transfer "
            "occurs at PPO prepare"
        ),
    }
