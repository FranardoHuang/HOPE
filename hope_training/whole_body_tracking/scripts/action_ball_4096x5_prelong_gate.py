#!/usr/bin/env python3
"""Fail-closed terminal telemetry gate for a 4096-env, five-update smoke.

The runtime emits optimizer-health, Reward-by-action, and a versioned
opportunity-semantic JSON marker.  The latter keeps exact-strike timing ticks
separate from transactionally closed attempts, so ``0 / closed_swings`` remains
visible without making a false same-update ordering claim.  Omitting that marker
is an explicit producer blocker; the validator never infers eligibility from
non-zero Reward samples.

The checkpoint argument is the safe, already-loaded audit returned by the
launcher checkpoint verifier.  This module deliberately does not load pickle or
PyTorch checkpoint bytes itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


NUM_ENVS = 4096
EXPECTED_UPDATES = 5
ROLLOUT_STEPS_PER_UPDATE = 24
ROLLOUT_SAMPLES_PER_UPDATE = NUM_ENVS * ROLLOUT_STEPS_PER_UPDATE
ROLLOUT_SAMPLES_FIVE_UPDATE_AGGREGATE = (
    ROLLOUT_SAMPLES_PER_UPDATE * EXPECTED_UPDATES
)

PROFILE_A211 = "A211"
PROFILE_C211 = "C211"
PRELONG_PROFILES = (PROFILE_A211, PROFILE_C211)

ECONOMY_PREFIX = "HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_UPDATE_JSON="
GROUP_PREFIX = "HOPE_EFFECTIVE_REWARD_BY_ACTION_UPDATE_JSON="

# Bind directly to the producer instead of maintaining a second hand-written
# schema/counter catalogue in the gate.
_SEMANTICS_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "action_ball_prelong_semantics.py"
)
_SEMANTICS_SPEC = importlib.util.spec_from_file_location(
    "_action_ball_4096x5_gate_semantics", _SEMANTICS_SOURCE
)
if _SEMANTICS_SPEC is None or _SEMANTICS_SPEC.loader is None:
    raise RuntimeError("pre-long gate cannot load the semantic producer contract")
_SEMANTICS = importlib.util.module_from_spec(_SEMANTICS_SPEC)
_SEMANTICS_SPEC.loader.exec_module(_SEMANTICS)

# The existing per-action group ``eligible_sample_count`` means
# RewardManager-evaluated samples, not task/opportunity eligibility.  The v3
# semantic event supplies exact timing/attempt ledgers plus the reveal bridge.
SEMANTIC_EVENT = _SEMANTICS.PRELONG_SEMANTICS_EVENT
SEMANTIC_SCHEMA_VERSION = _SEMANTICS.PRELONG_SEMANTICS_SCHEMA_VERSION
REQUIRED_SEMANTIC_COUNTER_NAMES = tuple(
    _SEMANTICS.required_prelong_counter_names()
)
if (
    SEMANTIC_SCHEMA_VERSION != 3
    or not REQUIRED_SEMANTIC_COUNTER_NAMES
    or len(REQUIRED_SEMANTIC_COUNTER_NAMES)
    != len(set(REQUIRED_SEMANTIC_COUNTER_NAMES))
):
    raise RuntimeError("pre-long gate semantic producer contract is malformed")

STRICT_ZERO_SAFETY_COUNTERS = (
    "actual_hard_edge_event_count",
    "actual_hard_terminal_count",
    "joint_qdes_forbidden_terminal_count",
    "joint_actual_forbidden_terminal_count",
    "strict_hard_termination_count",
    "nonfinite_count",
)
PHYSICAL_FALL_REASONS = ("base_fell_tilt", "base_too_low")
BEHAVIORAL_TERMINATION_REASONS = (
    *PHYSICAL_FALL_REASONS,
    "robot_hit_table",
)
PHYSICAL_FALL_PHASES = (
    "hidden_wait",
    "revealed_pre_strike",
    "post_strike",
)
REQUIRED_CHECKPOINT_GROUPS = (
    "model",
    "optimizer",
    "actor_normalizer",
    "critic_normalizer",
)
REQUIRED_OPPORTUNITY_REWARD_GROUPS = (
    "balance",
    "mimic",
    "strike",
    "target",
    "outcome",
)
BRIDGE_WAIT_COHORTS = tuple(range(5, 26))
BRIDGE_TIMING_FIELDS = (
    "time_to_contact_tick",
    "teacher_rate",
    "scaled_t_hit_s",
    "pre_swing_wait_s",
    "expected_bridge_ticks",
)
BRIDGE_MIMIC_TERMS = tuple(
    _SEMANTICS.prelong_group_term_weights(PROFILE_A211)["mimic"]
)
BRIDGE_CAUCHY_MIMIC_TERMS = (
    "motion_racket_position",
    "motion_racket_velocity",
    "motion_racket_normal",
    "motion_racket_long_axis",
)


class PreLongGateRefused(ValueError):
    """Raised when terminal evidence is absent, malformed, or unsafe."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _finite_number(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise PreLongGateRefused(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise PreLongGateRefused(f"{name} must be finite and nonnegative")
    return result


def _counter(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PreLongGateRefused(f"{name} must be a nonnegative integer")
    return value


def _rollout_counter(value: Any, *, name: str) -> int:
    result = _counter(value, name=name)
    if result > ROLLOUT_SAMPLES_PER_UPDATE:
        raise PreLongGateRefused(
            f"{name} exceeds the fixed {ROLLOUT_SAMPLES_PER_UPDATE}-sample window"
        )
    return result


def _marker_rows(log_text: str, *, prefix: str, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        if not line.startswith(prefix):
            continue
        try:
            row = json.loads(line[len(prefix) :], parse_constant=_reject_constant)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PreLongGateRefused(
                f"{name} marker at line {line_number} is not finite JSON"
            ) from exc
        if type(row) is not dict:
            raise PreLongGateRefused(f"{name} marker must be a JSON object")
        rows.append(row)
    return rows


def _ordered_updates(
    rows: Sequence[Mapping[str, Any]],
    *,
    event: str,
    schema_version: int,
    name: str,
) -> list[Mapping[str, Any]]:
    if len(rows) != EXPECTED_UPDATES:
        raise PreLongGateRefused(
            f"{name} must contain exactly {EXPECTED_UPDATES} updates; got {len(rows)}"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PreLongGateRefused(
                f"{name} updates must be contiguous 0..{EXPECTED_UPDATES - 1}"
            )
        observed_schema_version = row.get("schema_version")
        observed_ppo_update = row.get("ppo_update")
        if (
            row.get("event") != event
            or type(observed_schema_version) is not int
            or observed_schema_version != schema_version
            or type(observed_ppo_update) is not int
            or observed_ppo_update != index
        ):
            raise PreLongGateRefused(
                f"{name} updates must be contiguous 0..{EXPECTED_UPDATES - 1}"
            )
    return list(rows)


def validate_checkpoint_audit(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the launcher's safe checkpoint audit, not raw checkpoint bytes."""

    if not isinstance(checkpoint, Mapping):
        raise PreLongGateRefused("checkpoint audit is missing")
    if (
        checkpoint.get("filename_iteration") != EXPECTED_UPDATES
        or checkpoint.get("embedded_iteration") != EXPECTED_UPDATES
        or checkpoint.get("all_tensors_finite") is not True
        or checkpoint.get("load_mode") != "torch_weights_only"
    ):
        raise PreLongGateRefused(
            "checkpoint audit must bind model_5, embedded iter=5, and finite weights-only load"
        )
    groups = checkpoint.get("tensor_groups")
    if not isinstance(groups, Mapping) or set(groups) != set(REQUIRED_CHECKPOINT_GROUPS):
        raise PreLongGateRefused("checkpoint tensor-group coverage differs")
    summary: dict[str, dict[str, int]] = {}
    for name in REQUIRED_CHECKPOINT_GROUPS:
        row = groups[name]
        if not isinstance(row, Mapping):
            raise PreLongGateRefused(f"checkpoint {name} tensor audit is missing")
        tensors = _counter(row.get("tensor_count"), name=f"checkpoint {name} tensor_count")
        elements = _counter(row.get("element_count"), name=f"checkpoint {name} element_count")
        if tensors == 0 or elements == 0:
            raise PreLongGateRefused(f"checkpoint {name} tensor audit is empty")
        summary[name] = {"tensor_count": tensors, "element_count": elements}
    return {"iteration": EXPECTED_UPDATES, "all_tensors_finite": True, "groups": summary}


def validate_safety_audit(safety: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(safety, Mapping):
        raise PreLongGateRefused("terminal safety audit is missing")
    if safety.get("observed_ppo_updates") != EXPECTED_UPDATES:
        raise PreLongGateRefused("terminal safety audit does not cover exactly five updates")
    strict = {
        name: _counter(safety.get(name), name=f"safety {name}")
        for name in STRICT_ZERO_SAFETY_COUNTERS
    }
    if any(strict.values()):
        raise PreLongGateRefused(
            "joint-qdes/joint-actual/nonfinite implementation counters are nonzero"
        )

    by_reason = {
        reason: _counter(
            safety.get(f"{reason}_terminal_count"),
            name=f"safety {reason} terminal count",
        )
        for reason in PHYSICAL_FALL_REASONS
    }
    raw_reason_phase = safety.get("physical_fall_by_reason_phase")
    if not isinstance(raw_reason_phase, Mapping) or set(raw_reason_phase) != set(
        PHYSICAL_FALL_REASONS
    ):
        raise PreLongGateRefused(
            "physical-fall reason-by-phase coverage must contain exactly both balance reasons"
        )
    by_reason_phase: dict[str, dict[str, int]] = {}
    for reason in PHYSICAL_FALL_REASONS:
        raw_phases = raw_reason_phase[reason]
        if not isinstance(raw_phases, Mapping) or set(raw_phases) != set(
            PHYSICAL_FALL_PHASES
        ):
            raise PreLongGateRefused(
                f"physical-fall {reason} phase coverage must contain exactly "
                f"{PHYSICAL_FALL_PHASES!r}"
            )
        phases = {
            phase: _counter(
                raw_phases[phase],
                name=f"safety {reason} {phase} count",
            )
            for phase in PHYSICAL_FALL_PHASES
        }
        if sum(phases.values()) != by_reason[reason]:
            raise PreLongGateRefused(
                f"physical-fall {reason} reason-by-phase counts do not conserve"
            )
        by_reason_phase[reason] = phases

    raw_reveal_by_update = safety.get("task_reveal_reached_by_update")
    if (
        type(raw_reveal_by_update) is not list
        or len(raw_reveal_by_update) != EXPECTED_UPDATES
    ):
        raise PreLongGateRefused(
            "task-reveal survival denominator must cover exactly five updates"
        )
    reveal_by_update = [
        _counter(value, name=f"safety update {index} task reveal reached count")
        for index, value in enumerate(raw_reveal_by_update)
    ]
    if any(value == 0 for value in reveal_by_update):
        raise PreLongGateRefused(
            "every finite update requires a nonzero task-reveal survival denominator"
        )
    reveal_reached = sum(reveal_by_update)
    if safety.get("task_reveal_reached_count") != reveal_reached:
        raise PreLongGateRefused(
            "task-reveal per-update and aggregate counts do not conserve"
        )
    raw_wait_by_update = safety.get("task_wait_started_by_update")
    if type(raw_wait_by_update) is not list or len(raw_wait_by_update) != EXPECTED_UPDATES:
        raise PreLongGateRefused(
            "RESET_WAIT-start denominator must cover exactly five updates"
        )
    wait_by_update = [
        _counter(value, name=f"safety update {index} RESET_WAIT starts")
        for index, value in enumerate(raw_wait_by_update)
    ]
    if any(value == 0 for value in wait_by_update):
        raise PreLongGateRefused(
            "every finite update requires a nonzero RESET_WAIT-start denominator"
        )
    wait_started = sum(wait_by_update)
    if safety.get("task_wait_started_count") != wait_started:
        raise PreLongGateRefused(
            "RESET_WAIT-start per-update and aggregate counts do not conserve"
        )
    table_contact_count = _counter(
        safety.get("table_contact_count"), name="safety table contact count"
    )
    raw_table_phases = safety.get("table_contact_by_phase")
    if not isinstance(raw_table_phases, Mapping) or set(raw_table_phases) != set(
        PHYSICAL_FALL_PHASES
    ):
        raise PreLongGateRefused(
            "robot_hit_table phase coverage must contain exactly all three task phases"
        )
    table_contact_by_phase = {
        phase: _counter(
            raw_table_phases[phase], name=f"safety robot_hit_table {phase} count"
        )
        for phase in PHYSICAL_FALL_PHASES
    }
    if sum(table_contact_by_phase.values()) != table_contact_count:
        raise PreLongGateRefused(
            "robot_hit_table reason-by-phase counts do not conserve"
        )
    behavioral_by_reason = {
        **by_reason,
        "robot_hit_table": table_contact_count,
    }
    behavioral_by_reason_phase = {
        **by_reason_phase,
        "robot_hit_table": table_contact_by_phase,
    }
    return {
        "observed_ppo_updates": EXPECTED_UPDATES,
        "strict_zero_counters": strict,
        "balance_termination_counts": {
            "by_reason": by_reason,
            "by_reason_phase": by_reason_phase,
        },
        "behavioral_termination_counts": {
            "by_reason": behavioral_by_reason,
            "by_reason_phase": behavioral_by_reason_phase,
        },
        "task_reveal_reached_count": reveal_reached,
        "task_reveal_reached_by_update": reveal_by_update,
        "task_wait_started_count": wait_started,
        "task_wait_started_by_update": wait_by_update,
        "table_contact_count": table_contact_count,
        "table_contact_by_phase": table_contact_by_phase,
        "finite_balance_termination_policy": (
            "fall/too-low/table are behavioral termination evidence; "
            "finite acceptance has no unvalidated_numeric_cutoff, and the long-run health gate "
            "must preregister the tighter survival bound"
        ),
    }


def validate_survival_denominators(
    *, safety: Mapping[str, Any], semantics: Mapping[str, Any]
) -> dict[str, Any]:
    """Close the finite survival ladder without inventing a fall-rate threshold."""

    updates = semantics.get("updates")
    if type(updates) is not list or len(updates) != EXPECTED_UPDATES:
        raise PreLongGateRefused("pre-long semantic updates are missing")
    reveal_by_update = safety.get("task_reveal_reached_by_update")
    if type(reveal_by_update) is not list or len(reveal_by_update) != EXPECTED_UPDATES:
        raise PreLongGateRefused("per-update task-reveal denominators are missing")
    per_update = []
    for index, (row, reveal_value) in enumerate(zip(updates, reveal_by_update)):
        if not isinstance(row, Mapping):
            raise PreLongGateRefused(f"survival update {index} is missing")
        reveal_i = _counter(
            reveal_value, name=f"survival update {index} task reveal reached count"
        )
        nominal_i = _counter(
            row.get("exact_strike_tick_denominator"),
            name=f"survival update {index} nominal-strike reached count",
        )
        closed_i = _counter(
            row.get("eligible_closed_swing_count"),
            name=f"survival update {index} closed-swing count",
        )
        task_active_i = _counter(
            row.get("task_active_samples"),
            name=f"survival update {index} TASK_ACTIVE samples",
        )
        if task_active_i == 0 or reveal_i == 0 or nominal_i == 0 or closed_i == 0:
            raise PreLongGateRefused(
                f"finite survival update {index} requires nonzero TASK_ACTIVE, "
                "reveal, nominal-strike, and closed-swing denominators"
            )
        per_update.append(
            {
                "ppo_update": index,
                "task_active_sample_count": task_active_i,
                "task_reveal_reached_count": reveal_i,
                "nominal_strike_reached_count": nominal_i,
                "eligible_closed_swing_count": closed_i,
                "nominal_strike_per_reveal": nominal_i / reveal_i,
                "closed_swing_per_reveal": closed_i / reveal_i,
            }
        )

    aggregate = semantics.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise PreLongGateRefused("pre-long semantic aggregate is missing")
    reveal = _counter(
        safety.get("task_reveal_reached_count"),
        name="survival task reveal reached count",
    )
    nominal = _counter(
        aggregate.get("exact_strike_tick_denominator"),
        name="survival nominal-strike reached count",
    )
    closed = _counter(
        aggregate.get("eligible_closed_swing_count"),
        name="survival closed-swing count",
    )
    task_active = _counter(
        aggregate.get("task_active_observed_sample_count"),
        name="survival TASK_ACTIVE sample count",
    )
    if task_active != sum(row["task_active_sample_count"] for row in per_update):
        raise PreLongGateRefused("aggregate TASK_ACTIVE samples do not conserve per update")
    if reveal != sum(row["task_reveal_reached_count"] for row in per_update):
        raise PreLongGateRefused("aggregate task reveals do not conserve per update")
    if nominal != sum(row["nominal_strike_reached_count"] for row in per_update):
        raise PreLongGateRefused("aggregate nominal strikes do not conserve per update")
    if closed != sum(row["eligible_closed_swing_count"] for row in per_update):
        raise PreLongGateRefused("aggregate closed swings do not conserve per update")
    wait_started = _counter(
        safety.get("task_wait_started_count"),
        name="survival RESET_WAIT started count",
    )
    phase_exposure_denominators = {
        "hidden_wait": wait_started,
        "revealed_pre_strike": reveal,
        "post_strike": nominal,
    }
    raw_behavioral = safety.get("behavioral_termination_counts")
    raw_behavioral_by_reason = (
        raw_behavioral.get("by_reason")
        if isinstance(raw_behavioral, Mapping)
        else None
    )
    raw_behavioral_by_phase = (
        raw_behavioral.get("by_reason_phase")
        if isinstance(raw_behavioral, Mapping)
        else None
    )
    if (
        not isinstance(raw_behavioral_by_reason, Mapping)
        or set(raw_behavioral_by_reason) != set(BEHAVIORAL_TERMINATION_REASONS)
        or not isinstance(raw_behavioral_by_phase, Mapping)
        or set(raw_behavioral_by_phase) != set(BEHAVIORAL_TERMINATION_REASONS)
    ):
        raise PreLongGateRefused(
            "survival behavioral reason-by-phase evidence is missing"
        )
    behavioral: dict[str, dict[str, Any]] = {}
    for reason in BEHAVIORAL_TERMINATION_REASONS:
        total = _counter(
            raw_behavioral_by_reason[reason],
            name=f"survival {reason} total count",
        )
        phase_values = raw_behavioral_by_phase[reason]
        if not isinstance(phase_values, Mapping) or set(phase_values) != set(
            PHYSICAL_FALL_PHASES
        ):
            raise PreLongGateRefused(
                f"survival {reason} phase evidence is incomplete"
            )
        by_phase = {
            phase: _counter(
                phase_values[phase],
                name=f"survival {reason} {phase} count",
            )
            for phase in PHYSICAL_FALL_PHASES
        }
        if sum(by_phase.values()) != total:
            raise PreLongGateRefused(
                f"survival {reason} reason-by-phase counts do not conserve"
            )
        behavioral[reason] = {
            "total_count": total,
            "by_phase": by_phase,
            "phase_exposure_denominators": dict(phase_exposure_denominators),
            "phase_rates": {
                phase: by_phase[phase] / phase_exposure_denominators[phase]
                for phase in PHYSICAL_FALL_PHASES
            },
            "acceptance_threshold": None,
        }
    return {
        "per_update": per_update,
        "task_active_observed_sample_count": task_active,
        "task_reveal_reached_count": reveal,
        "nominal_strike_reached_count": nominal,
        "eligible_closed_swing_count": closed,
        "nominal_strike_per_reveal": nominal / reveal,
        "closed_swing_per_reveal": closed / reveal,
        "behavioral_terminations": behavioral,
        # Compatibility alias retained because existing receipt consumers use
        # this key.  It is the same audited row, not a second counter path.
        "robot_hit_table": behavioral["robot_hit_table"],
        "finite_acceptance": (
            "nonzero milestone denominators plus complete physical-fall attribution; "
            "no unvalidated finite fall-rate threshold"
        ),
    }


# 人话:一个奖励项如果在整个五-update 窗口里**逐位**吐同一个数,它就没有给策略任何可学的
# 信号 —— 它只是给每个样本加了一个常数。常数并不是无害的:每步一个负常数会让"早点终止"
# 严格优于"多活一步",所以一个既在收费、又没有信号的项,正好是把经济推向自杀的那一类。
#
# 现役实例:``action_rate_clamped`` 在 s15r1 的 C0/C1 两格、五个 update 上全是逐位相同的
# ``-3538.945068``(``raw_sum = 884736 = 98304 x 9``,即每个样本恒等于 ``value_clamp``)。
# 这不是实现 bug,是 ``action_rate_l2_clamped`` 的封顶被打满:31 维、std=1.0 的新策略
# ``E||Δa||² = 2 x 31 x std² = 62``,远在 ``9.0`` 之上,所以**每个**样本都被削到天花板。
# 旧版本的这道门只检查每项收入"是有限数"和分母对不对,因此它对"这一项从头到尾没动过"
# 完全没有意见 —— 连本模块自己的测试夹具都把 ``motion`` 写成五个 update 恒等于 ``1.0``。
#
# 允许一个项常数的唯一方式,是在下表里写清两件事:**是什么机制让它常数**,以及**这个常数
# 在什么条件下结束**。没写在表里的常数非零项一律拒收(fail closed)。
#
# 为什么不顺手把"恒零"也拒掉:A211 的 target/outcome 项在一次一球未碰的 5-update 冒烟里
# 本来就该恒零,拒掉它等于要求一个没训过的策略已经会打球(§5.6.8 同型的"门定错范围")。
# 恒零项如实进收据,不进拒收。
DECLARED_CONSTANT_REWARD_TERMS = {
    "action_rate_clamped": {
        "mechanism": (
            "action_rate_l2_clamped(value_clamp=9.0) is saturated: a fresh 31-D Gaussian "
            "policy at std 1.0 has E||da||^2 = 2 * 31 * std^2 = 62 >> 9.0, so every rollout "
            "sample is clipped to the ceiling and the term is a constant per-step offset."
        ),
        "ends_when": (
            "policy_std_max falls to roughly sqrt(9.0 / (2 * 31)) = 0.381, where ||da||^2 "
            "starts landing below value_clamp and the term regains sample-to-sample variation."
        ),
        "carries_no_learning_signal_while_constant": True,
    },
}


def _validate_declared_constant_reward_terms() -> None:
    """Refuse a silently blanked declaration table.

    A future "soft delete" that empties ``mechanism``/``ends_when`` would turn the
    allowlist into a blanket exemption without touching the refusal code path.
    """

    for term, declaration in DECLARED_CONSTANT_REWARD_TERMS.items():
        if type(term) is not str or not term:
            raise PreLongGateRefused("declared constant reward term names must be strings")
        if not isinstance(declaration, Mapping):
            raise PreLongGateRefused(
                f"declared constant reward term {term} has no declaration object"
            )
        for field in ("mechanism", "ends_when"):
            text = declaration.get(field)
            if type(text) is not str or not text.strip():
                raise PreLongGateRefused(
                    f"declared constant reward term {term} has an empty {field}"
                )


def _reward_term_signal_ledger(
    per_update_terms: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Report every term's cross-update variation and refuse undeclared frozen ones.

    Comparison is exact float equality against the first update, not a tolerance:
    a term that moves by a single float32 ulp between updates *is* responding to the
    policy and must not be reported as frozen.
    """

    _validate_declared_constant_reward_terms()
    names = sorted(per_update_terms[0])
    frozen_nonzero: list[dict[str, Any]] = []
    always_zero: list[str] = []
    varying: list[str] = []
    undeclared: list[str] = []
    for name in names:
        values = [row[name] for row in per_update_terms]
        first = values[0]
        constant = all(value == first for value in values)
        if not constant:
            varying.append(name)
            continue
        if first == 0.0:
            always_zero.append(name)
            continue
        declaration = DECLARED_CONSTANT_REWARD_TERMS.get(name)
        if declaration is None:
            undeclared.append(name)
            continue
        frozen_nonzero.append(
            {
                "term": name,
                "weighted_dt_sum_every_update": first,
                "declared_mechanism": declaration["mechanism"],
                "declared_ends_when": declaration["ends_when"],
            }
        )
    if undeclared:
        raise PreLongGateRefused(
            "reward term(s) %s are bitwise identical and non-zero across all %d updates: "
            "an undeclared constant reward term charges a price while carrying no learning "
            "signal; declare its mechanism in DECLARED_CONSTANT_REWARD_TERMS or fix the term"
            % (", ".join(sorted(undeclared)), len(per_update_terms))
        )
    return {
        "semantics": (
            "per_term_weighted_dt_sum compared by exact equality across every observed "
            "update; frozen means the term returned the identical total each time"
        ),
        "term_count": len(names),
        "varying_term_count": len(varying),
        "frozen_nonzero_terms": frozen_nonzero,
        "always_zero_terms": always_zero,
        "always_zero_is_blocking": False,
        "always_zero_rationale": (
            "a pre-long smoke with no ball contact is expected to pay exactly zero on the "
            "target/outcome tier; refusing it would demand an untrained policy already hit"
        ),
    }


def _income_inversion_ledger(
    per_update_terms: Sequence[Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Report the largest single cost against the whole positive income, per update.

    §5.4 item 6 requires that regularization/safety terms never swamp the three main
    tiers, but the pre-long taxonomy deliberately excludes the safety tier from its
    group accounting (``PRELONG_EXCLUDED_SAFETY_TERM_WEIGHTS``), so no receipt has ever
    carried the comparison.  This block is report-only: pricing is a design decision.
    """

    ledger = []
    for index, terms in enumerate(per_update_terms):
        positive = sum(value for value in terms.values() if value > 0.0)
        negatives = [(value, name) for name, value in terms.items() if value < 0.0]
        if negatives:
            worst_value, worst_name = min(negatives)
        else:
            worst_value, worst_name = 0.0, None
        ledger.append(
            {
                "ppo_update": index,
                "positive_income_sum": positive,
                "largest_cost_term": worst_name,
                "largest_cost_weighted_dt_sum": worst_value,
                "largest_cost_over_positive_income": (
                    abs(worst_value) / positive if positive > 0.0 else None
                ),
                "net_weighted_dt_sum": sum(terms.values()),
            }
        )
    return ledger


def validate_economy_updates(log_text: str) -> dict[str, Any]:
    rows = _ordered_updates(
        _marker_rows(log_text, prefix=ECONOMY_PREFIX, name="reward/PPO economy"),
        event="hope_action_ball_reward_ppo_economy_update",
        schema_version=1,
        name="reward/PPO economy",
    )
    summaries = []
    per_update_terms: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        if row.get("status") != "PASS" or row.get("gate") != {
            "num_envs": NUM_ENVS,
            "steps_per_env_per_update": ROLLOUT_STEPS_PER_UPDATE,
            "rollout_samples_per_update": ROLLOUT_SAMPLES_PER_UPDATE,
        }:
            raise PreLongGateRefused(f"reward/PPO economy update {index} gate differs")
        reward, ppo, gradient, policy = (
            row.get("reward"), row.get("ppo"), row.get("gradient"), row.get("policy")
        )
        if not all(isinstance(item, Mapping) for item in (reward, ppo, gradient, policy)):
            raise PreLongGateRefused(f"reward/PPO economy update {index} sections are missing")
        explained_variance = _finite_number(
            reward.get("explained_variance"), name=f"update {index} explained_variance"
        )
        weighted = reward.get("per_term_weighted_dt_sum")
        denominators = reward.get("per_term_eligible_denominator")
        if (
            not isinstance(weighted, Mapping)
            or not weighted
            or not isinstance(denominators, Mapping)
            or set(weighted) != set(denominators)
        ):
            raise PreLongGateRefused(f"reward/PPO economy update {index} term ledger differs")
        term_incomes: dict[str, float] = {}
        for term in weighted:
            term_incomes[term] = _finite_number(
                weighted[term], name=f"update {index} term {term} income"
            )
            if denominators[term] != ROLLOUT_SAMPLES_PER_UPDATE:
                raise PreLongGateRefused(
                    f"update {index} term {term} is not the existing whole-rollout denominator"
                )
        # The cross-update signal ledger below can only compare a stable term set.
        if per_update_terms and set(term_incomes) != set(per_update_terms[0]):
            raise PreLongGateRefused(
                f"reward/PPO economy update {index} term set differs from update 0"
            )
        per_update_terms.append(term_incomes)
        learning_rate = _finite_number(
            ppo.get("learning_rate"), name=f"update {index} learning_rate", nonnegative=True
        )
        if learning_rate <= 0.0:
            raise PreLongGateRefused(f"update {index} learning_rate must be positive")
        approx_kl = _finite_number(ppo.get("approx_kl"), name=f"update {index} approx_kl")
        clip_fraction = _finite_number(
            ppo.get("clip_fraction"), name=f"update {index} clip_fraction", nonnegative=True
        )
        if clip_fraction > 1.0:
            raise PreLongGateRefused(f"update {index} clip_fraction exceeds one")
        grad_norm = _finite_number(
            gradient.get("pre_clip_total_grad_norm"),
            name=f"update {index} pre_clip_total_grad_norm",
            nonnegative=True,
        )
        std_min = _finite_number(
            policy.get("policy_std_min"), name=f"update {index} policy_std_min"
        )
        std_mean = _finite_number(
            policy.get("policy_std_mean"), name=f"update {index} policy_std_mean"
        )
        std_max = _finite_number(
            policy.get("policy_std_max"), name=f"update {index} policy_std_max"
        )
        if not 0.0 < std_min <= std_mean <= std_max:
            raise PreLongGateRefused(f"update {index} policy std is not positive and ordered")
        summaries.append(
            {
                "ppo_update": index,
                "learning_rate": learning_rate,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "explained_variance": explained_variance,
                "pre_clip_total_grad_norm": grad_norm,
                "policy_std_min_mean_max": [std_min, std_mean, std_max],
                "term_count": len(weighted),
            }
        )
    return {
        "updates": summaries,
        "reward_term_signal": _reward_term_signal_ledger(per_update_terms),
        "income_inversion": _income_inversion_ledger(per_update_terms),
    }


def validate_group_income_updates(log_text: str) -> dict[str, Any]:
    rows = _ordered_updates(
        _marker_rows(log_text, prefix=GROUP_PREFIX, name="Reward group income"),
        event="hope_effective_reward_activation_by_action_update",
        schema_version=2,
        name="Reward group income",
    )
    summaries = []
    for index, row in enumerate(rows):
        actions = row.get("actions")
        if not isinstance(actions, list) or not actions:
            raise PreLongGateRefused(f"Reward group update {index} has no actions")
        action_summary = []
        for action in actions:
            if not isinstance(action, Mapping) or type(action.get("action_id")) is not str:
                raise PreLongGateRefused(f"Reward group update {index} action identity differs")
            groups = action.get("reward_groups")
            if not isinstance(groups, list) or not groups:
                raise PreLongGateRefused(f"Reward group update {index} has no group rows")
            names = []
            for group in groups:
                if not isinstance(group, Mapping) or type(group.get("group")) is not str:
                    raise PreLongGateRefused(f"Reward group update {index} group identity differs")
                if group.get("eligibility") != "reward_manager_evaluated_active_group_terms":
                    raise PreLongGateRefused(
                        f"Reward group update {index} changed existing evaluated-sample semantics"
                    )
                _counter(
                    group.get("eligible_sample_count"),
                    name=f"update {index} {group['group']} evaluated sample count",
                )
                _finite_number(
                    group.get("weighted_sum"),
                    name=f"update {index} {group['group']} weighted income",
                )
                names.append(group["group"])
            if len(names) != len(set(names)):
                raise PreLongGateRefused(f"Reward group update {index} has duplicate groups")
            action_summary.append({"action_id": action["action_id"], "groups": names})
        summaries.append({"ppo_update": index, "actions": action_summary})
    return {
        "updates": summaries,
        "eligibility_boundary": (
            "evaluated-sample counts validated; opportunity-conditioned eligibility "
            "must come from semantic evidence"
        ),
    }


def _exact_keys(value: Any, expected: Sequence[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise PreLongGateRefused(f"{name} fields differ")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise PreLongGateRefused(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreLongGateRefused(f"{name} must be lowercase SHA-256")
    return value


def _validate_reveal_bridge(
    value: Any,
    *,
    profile: str,
    update: int,
    previous_lifetime: Optional[Mapping[int, Mapping[str, int]]],
    expected_authority: Optional[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, int]], dict[str, Any]]:
    """Strictly consume one schema-v3 reveal-to-playback bridge record.

    **尚未接线(2026-08-06 核实):本函数全仓零调用点。**
    别把它读成"本 gate 已经严格消费 v3" —— 现役 gate 仍按 schema-v2 走,A launcher 在第 84
    项 fail-closed 拒的也是 v2 那条路。生产方 ``utils/action_ball_prelong_semantics.py``
    确实已经产出 ``reveal_to_playback_bridge``,消费方就差这一步接线。
    接线是独立一步(见 docs/experiments/2026-08/
    EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md 的"把 shared gate 升级为严格消费
    v3"一条),不是清理能顺手做的:它会改变 gate 的拒绝面。
    """

    bridge = _exact_keys(
        value,
        (
            "status",
            "authority",
            "lifetime_conservation",
            "timing_at_reveal",
            "window",
            "performance_contract",
        ),
        name=f"semantic update {update} reveal bridge",
    )
    if bridge.get("status") != "active_fail_closed":
        raise PreLongGateRefused(
            f"semantic update {update} reveal bridge is not active fail-closed"
        )
    authority = dict(
        _exact_keys(
            bridge.get("authority"),
            (
                "family",
                "target_source",
                "target_recipe",
                "timing_authority",
                "timing_contract_sha256",
                "question_sha256",
                "question_sha_semantics",
                "sampler_contract_sha256",
                "profile",
                "effective_reward_recipe_sha256",
                "wait_schedule_sha256",
                "wait_cohort_ticks",
                "policy_dt_s",
            ),
            name=f"semantic update {update} bridge authority",
        )
    )
    for field in (
        "family",
        "timing_authority",
        "question_sha_semantics",
    ):
        _nonempty_string(
            authority.get(field),
            name=f"semantic update {update} bridge authority {field}",
        )
    if authority.get("profile") != profile:
        raise PreLongGateRefused(
            f"semantic update {update} bridge authority profile differs"
        )
    expected_target = (
        ("online_solver", "current_lm")
        if profile == PROFILE_A211
        else ("direct_ball", "outcome_dense_only")
    )
    if (
        authority.get("target_source"),
        authority.get("target_recipe"),
    ) != expected_target:
        raise PreLongGateRefused(
            f"semantic update {update} bridge target source/recipe differs"
        )
    for field in (
        "timing_contract_sha256",
        "question_sha256",
        "sampler_contract_sha256",
        "effective_reward_recipe_sha256",
        "wait_schedule_sha256",
    ):
        _sha256(
            authority.get(field),
            name=f"semantic update {update} bridge authority {field}",
        )
    if authority.get("wait_cohort_ticks") != list(BRIDGE_WAIT_COHORTS):
        raise PreLongGateRefused(
            f"semantic update {update} bridge WAIT cohorts differ"
        )
    policy_dt = _finite_number(
        authority.get("policy_dt_s"),
        name=f"semantic update {update} bridge policy dt",
        nonnegative=True,
    )
    if not math.isclose(policy_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
        raise PreLongGateRefused(
            f"semantic update {update} bridge policy dt differs"
        )
    if expected_authority is not None and authority != expected_authority:
        raise PreLongGateRefused(
            f"semantic update {update} bridge authority drifted across updates"
        )

    conservation = _exact_keys(
        bridge.get("lifetime_conservation"),
        ("equation", "wait_cohorts"),
        name=f"semantic update {update} bridge lifetime conservation",
    )
    if conservation.get("equation") != (
        "reveal_count=playback_start_count+terminal_before_start_count+censored_count"
    ):
        raise PreLongGateRefused(
            f"semantic update {update} bridge conservation equation differs"
        )
    raw_cohorts = conservation.get("wait_cohorts")
    if type(raw_cohorts) is not list or len(raw_cohorts) != len(BRIDGE_WAIT_COHORTS):
        raise PreLongGateRefused(
            f"semantic update {update} bridge WAIT cohort coverage differs"
        )
    lifetime: dict[int, dict[str, int]] = {}
    reveal_delta = 0
    for expected_wait, raw in zip(BRIDGE_WAIT_COHORTS, raw_cohorts):
        cohort = _exact_keys(
            raw,
            (
                "wait_ticks",
                "reveal_count",
                "playback_start_count",
                "terminal_before_start_count",
                "censored_count",
            ),
            name=f"semantic update {update} bridge WAIT={expected_wait}",
        )
        if cohort.get("wait_ticks") != expected_wait:
            raise PreLongGateRefused(
                f"semantic update {update} bridge WAIT cohort order differs"
            )
        counts = {
            "reveal": _counter(
                cohort.get("reveal_count"),
                name=f"semantic update {update} WAIT={expected_wait} reveal",
            ),
            "start": _counter(
                cohort.get("playback_start_count"),
                name=f"semantic update {update} WAIT={expected_wait} playback start",
            ),
            "terminal": _counter(
                cohort.get("terminal_before_start_count"),
                name=f"semantic update {update} WAIT={expected_wait} terminal",
            ),
            "censored": _counter(
                cohort.get("censored_count"),
                name=f"semantic update {update} WAIT={expected_wait} censored",
            ),
        }
        if counts["reveal"] != (
            counts["start"] + counts["terminal"] + counts["censored"]
        ):
            raise PreLongGateRefused(
                f"semantic update {update} bridge WAIT={expected_wait} does not conserve"
            )
        previous = (
            {"reveal": 0, "start": 0, "terminal": 0, "censored": 0}
            if previous_lifetime is None
            else previous_lifetime[expected_wait]
        )
        for monotonic in ("reveal", "start", "terminal"):
            if counts[monotonic] < previous[monotonic]:
                raise PreLongGateRefused(
                    f"semantic update {update} bridge WAIT={expected_wait} lifetime regressed"
                )
        reveal_delta += counts["reveal"] - previous["reveal"]
        lifetime[expected_wait] = counts
    total_reveal = sum(row["reveal"] for row in lifetime.values())
    if reveal_delta <= 0:
        raise PreLongGateRefused(
            f"semantic update {update} bridge has no newly revealed rows"
        )

    timing = _exact_keys(
        bridge.get("timing_at_reveal"),
        ("reveal_count", "fields", "expected_bridge_tick_rule"),
        name=f"semantic update {update} bridge reveal timing",
    )
    timing_reveal = _counter(
        timing.get("reveal_count"),
        name=f"semantic update {update} bridge timing reveal count",
    )
    if timing_reveal != total_reveal:
        raise PreLongGateRefused(
            f"semantic update {update} bridge timing/reveal counts differ"
        )
    _nonempty_string(
        timing.get("expected_bridge_tick_rule"),
        name=f"semantic update {update} bridge expected-tick rule",
    )
    raw_fields = _exact_keys(
        timing.get("fields"),
        BRIDGE_TIMING_FIELDS,
        name=f"semantic update {update} bridge timing fields",
    )
    timing_summary = {}
    for field in BRIDGE_TIMING_FIELDS:
        stats = _exact_keys(
            raw_fields[field],
            ("mean", "min", "max"),
            name=f"semantic update {update} bridge timing {field}",
        )
        mean = _finite_number(
            stats.get("mean"), name=f"semantic update {update} bridge {field} mean"
        )
        minimum = _finite_number(
            stats.get("min"), name=f"semantic update {update} bridge {field} min"
        )
        maximum = _finite_number(
            stats.get("max"), name=f"semantic update {update} bridge {field} max"
        )
        if not minimum <= mean <= maximum:
            raise PreLongGateRefused(
                f"semantic update {update} bridge {field} min/mean/max are unordered"
            )
        lower_bound = 0.0 if field == "pre_swing_wait_s" else 0.0
        if minimum < lower_bound or (
            field != "pre_swing_wait_s" and minimum <= 0.0
        ):
            raise PreLongGateRefused(
                f"semantic update {update} bridge {field} is outside its physical domain"
            )
        if field in ("time_to_contact_tick", "expected_bridge_ticks") and (
            not float(minimum).is_integer() or not float(maximum).is_integer()
        ):
            raise PreLongGateRefused(
                f"semantic update {update} bridge {field} bounds are not ticks"
            )
        timing_summary[field] = {"mean": mean, "min": minimum, "max": maximum}

    window = _exact_keys(
        bridge.get("window"),
        (
            "bridge_sample_count",
            "task_income_rule",
            "task_weighted_income_sum",
            "racket_progress_weighted_income_sum",
            "hidden_wait_task_income_required",
            "mimic_terms",
            "safety",
        ),
        name=f"semantic update {update} bridge window",
    )
    bridge_samples = _rollout_counter(
        window.get("bridge_sample_count"),
        name=f"semantic update {update} bridge sample count",
    )
    if bridge_samples <= 0:
        raise PreLongGateRefused(
            f"semantic update {update} bridge sample count is zero"
        )
    task_income = _finite_number(
        window.get("task_weighted_income_sum"),
        name=f"semantic update {update} bridge task income",
    )
    progress_income = _finite_number(
        window.get("racket_progress_weighted_income_sum"),
        name=f"semantic update {update} bridge progress income",
    )
    if window.get("hidden_wait_task_income_required") != 0.0:
        raise PreLongGateRefused(
            f"semantic update {update} hidden WAIT task-income rule differs"
        )
    expected_task_rule = (
        "racket_progress_only_base_position_absent_or_zero"
        if profile == PROFILE_A211
        else "all_task_income_exact_zero"
    )
    if window.get("task_income_rule") != expected_task_rule:
        raise PreLongGateRefused(
            f"semantic update {update} bridge task-income rule differs"
        )
    if profile == PROFILE_A211:
        if not math.isclose(task_income, progress_income, rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise PreLongGateRefused(
                f"semantic update {update} A211 bridge income is not progress-only"
            )
    elif task_income != 0.0 or progress_income != 0.0:
        raise PreLongGateRefused(
            f"semantic update {update} C211 bridge task income is nonzero"
        )

    raw_mimic = window.get("mimic_terms")
    if type(raw_mimic) is not list or len(raw_mimic) != len(BRIDGE_MIMIC_TERMS):
        raise PreLongGateRefused(
            f"semantic update {update} bridge mimic coverage differs"
        )
    mimic_summary = []
    for expected_term, raw in zip(BRIDGE_MIMIC_TERMS, raw_mimic):
        term = _exact_keys(
            raw,
            (
                "term",
                "kernel",
                "error_semantics",
                "std",
                "eligible_denominator",
                "raw_reward_sum_before_manager_weight",
                "raw_kernel_sum_after_window_scale_removed",
                "finite_error_denominator",
                "zero_kernel_count",
                "error_mean",
                "error_max",
                "weighted_income_sum",
                "income_semantics",
            ),
            name=f"semantic update {update} bridge mimic {expected_term}",
        )
        if term.get("term") != expected_term:
            raise PreLongGateRefused(
                f"semantic update {update} bridge mimic purpose order differs"
            )
        cauchy = expected_term in BRIDGE_CAUCHY_MIMIC_TERMS
        expected_kernel = (
            "cauchy_one_over_one_plus_error_over_std_squared"
            if cauchy
            else "exp_negative_squared_error_over_std_squared"
        )
        expected_error = (
            "std*sqrt(kernel^-1-1)" if cauchy else "std*sqrt(-ln(kernel))"
        )
        if term.get("kernel") != expected_kernel or term.get("error_semantics") != expected_error:
            raise PreLongGateRefused(
                f"semantic update {update} bridge mimic {expected_term} kernel differs"
            )
        std = _finite_number(
            term.get("std"),
            name=f"semantic update {update} bridge mimic {expected_term} std",
        )
        if std <= 0.0:
            raise PreLongGateRefused(
                f"semantic update {update} bridge mimic {expected_term} std is not positive"
            )
        eligible = _rollout_counter(
            term.get("eligible_denominator"),
            name=f"semantic update {update} bridge mimic {expected_term} eligible",
        )
        if eligible > bridge_samples or (
            expected_term not in ("motion_body_pos", "motion_body_ori")
            and eligible != bridge_samples
        ):
            raise PreLongGateRefused(
                f"semantic update {update} bridge mimic {expected_term} eligibility differs"
            )
        raw_sum = _finite_number(
            term.get("raw_reward_sum_before_manager_weight"),
            name=f"semantic update {update} bridge mimic {expected_term} raw sum",
            nonnegative=True,
        )
        kernel_sum = _finite_number(
            term.get("raw_kernel_sum_after_window_scale_removed"),
            name=f"semantic update {update} bridge mimic {expected_term} kernel sum",
            nonnegative=True,
        )
        if raw_sum > eligible + 1.0e-6 or kernel_sum > eligible + 1.0e-6:
            raise PreLongGateRefused(
                f"semantic update {update} bridge mimic {expected_term} reward exceeds eligibility"
            )
        finite_errors = _counter(
            term.get("finite_error_denominator"),
            name=f"semantic update {update} bridge mimic {expected_term} finite errors",
        )
        zero_kernel = _counter(
            term.get("zero_kernel_count"),
            name=f"semantic update {update} bridge mimic {expected_term} zero kernels",
        )
        if finite_errors + zero_kernel != eligible:
            raise PreLongGateRefused(
                f"semantic update {update} bridge mimic {expected_term} errors do not conserve"
            )
        error_mean = term.get("error_mean")
        error_max = term.get("error_max")
        if finite_errors == 0:
            if error_mean is not None or error_max is not None:
                raise PreLongGateRefused(
                    f"semantic update {update} bridge mimic {expected_term} empty errors differ"
                )
        else:
            error_mean = _finite_number(
                error_mean,
                name=f"semantic update {update} bridge mimic {expected_term} error mean",
                nonnegative=True,
            )
            error_max = _finite_number(
                error_max,
                name=f"semantic update {update} bridge mimic {expected_term} error max",
                nonnegative=True,
            )
            if error_max < error_mean:
                raise PreLongGateRefused(
                    f"semantic update {update} bridge mimic {expected_term} errors are unordered"
                )
        income = _finite_number(
            term.get("weighted_income_sum"),
            name=f"semantic update {update} bridge mimic {expected_term} income",
            nonnegative=True,
        )
        if term.get("income_semantics") != (
            "raw_reward_times_manager_weight_times_policy_dt"
        ):
            raise PreLongGateRefused(
                f"semantic update {update} bridge mimic {expected_term} income semantics differ"
            )
        mimic_summary.append(
            {"term": expected_term, "eligible_denominator": eligible, "income": income}
        )

    safety = _exact_keys(
        window.get("safety"),
        (
            "sample_count",
            "minimum_physical_hard_gap_rad",
            "maximum_abs_qvel_over_physical_limit",
            "minimum_root_height_m",
            "maximum_root_height_m",
            "minimum_root_upright_cosine",
            "maximum_root_xy_speed_mps",
            "mean_foot_contact_fraction",
            "mean_foot_slip_speed_mps",
            "maximum_foot_slip_speed_mps",
            "sampling_semantics",
        ),
        name=f"semantic update {update} bridge safety",
    )
    safety_count = _rollout_counter(
        safety.get("sample_count"),
        name=f"semantic update {update} bridge safety samples",
    )
    if safety_count != bridge_samples:
        raise PreLongGateRefused(
            f"semantic update {update} bridge safety denominator differs"
        )
    hard_gap = _finite_number(
        safety.get("minimum_physical_hard_gap_rad"),
        name=f"semantic update {update} bridge hard gap",
    )
    qvel_ratio = _finite_number(
        safety.get("maximum_abs_qvel_over_physical_limit"),
        name=f"semantic update {update} bridge qvel ratio",
        nonnegative=True,
    )
    root_min = _finite_number(
        safety.get("minimum_root_height_m"),
        name=f"semantic update {update} bridge root min",
    )
    root_max = _finite_number(
        safety.get("maximum_root_height_m"),
        name=f"semantic update {update} bridge root max",
    )
    upright = _finite_number(
        safety.get("minimum_root_upright_cosine"),
        name=f"semantic update {update} bridge root upright",
    )
    root_speed = _finite_number(
        safety.get("maximum_root_xy_speed_mps"),
        name=f"semantic update {update} bridge root speed",
        nonnegative=True,
    )
    foot_contact = _finite_number(
        safety.get("mean_foot_contact_fraction"),
        name=f"semantic update {update} bridge foot contact",
        nonnegative=True,
    )
    foot_slip_mean = _finite_number(
        safety.get("mean_foot_slip_speed_mps"),
        name=f"semantic update {update} bridge foot slip mean",
        nonnegative=True,
    )
    foot_slip_max = _finite_number(
        safety.get("maximum_foot_slip_speed_mps"),
        name=f"semantic update {update} bridge foot slip max",
        nonnegative=True,
    )
    if (
        hard_gap < 0.0
        or root_min > root_max
        or not -1.0 <= upright <= 1.0
        or root_speed < 0.0
        or not 0.0 <= foot_contact <= 1.0
        or foot_slip_max < foot_slip_mean
    ):
        raise PreLongGateRefused(
            f"semantic update {update} bridge safety values are inconsistent"
        )
    _nonempty_string(
        safety.get("sampling_semantics"),
        name=f"semantic update {update} bridge safety semantics",
    )
    _nonempty_string(
        bridge.get("performance_contract"),
        name=f"semantic update {update} bridge performance contract",
    )
    return (
        {
            "reveal_count_delta": reveal_delta,
            "cumulative_reveal_count": total_reveal,
            "bridge_sample_count": bridge_samples,
            "task_weighted_income_sum": task_income,
            "racket_progress_weighted_income_sum": progress_income,
            "timing_at_reveal": timing_summary,
            "mimic_terms": mimic_summary,
            "safety": {
                "minimum_physical_hard_gap_rad": hard_gap,
                "maximum_abs_qvel_over_physical_limit": qvel_ratio,
                "minimum_root_height_m": root_min,
                "maximum_root_height_m": root_max,
                "minimum_root_upright_cosine": upright,
                "maximum_root_xy_speed_mps": root_speed,
                "mean_foot_contact_fraction": foot_contact,
                "mean_foot_slip_speed_mps": foot_slip_mean,
                "maximum_foot_slip_speed_mps": foot_slip_max,
            },
        },
        lifetime,
        authority,
    )


def validate_semantic_updates(
    rows: Optional[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Validate the missing producer's explicit opportunity semantics."""

    if rows is None:
        raise PreLongGateRefused(
            "MISSING_PRODUCER: per-update task-invalid/task-reward, closed-swing/contact, "
            "achieved-flight, opportunity-conditioned Reward-group, and unknown-attribution evidence"
        )
    ordered = _ordered_updates(
        rows,
        event=SEMANTIC_EVENT,
        schema_version=SEMANTIC_SCHEMA_VERSION,
        name="pre-long semantic evidence",
    )
    summaries = []
    total_invalid_samples = 0
    total_exact_strike_ticks = 0
    total_closed_swings = 0
    total_contacts = 0
    total_outcome_opportunities = 0
    aggregate_group_incomes = {
        group: 0.0 for group in REQUIRED_OPPORTUNITY_REWARD_GROUPS
    }
    aggregate_group_denominators = {
        group: 0 for group in REQUIRED_OPPORTUNITY_REWARD_GROUPS
    }
    profile: Optional[str] = None
    for index, row in enumerate(ordered):
        row_profile = row.get("profile")
        if type(row_profile) is not str or row_profile not in PRELONG_PROFILES:
            raise PreLongGateRefused(
                f"semantic update {index} profile must be exactly A211 or C211"
            )
        if profile is None:
            profile = row_profile
        elif row_profile != profile:
            raise PreLongGateRefused(
                "five-update semantic evidence mixes A211 and C211 profiles"
            )
        window = row.get("window")
        task_invalid = row.get("task_invalid")
        strike_timing = row.get("strike_timing")
        hit = row.get("hit")
        achieved_flight = row.get("achieved_flight")
        groups = row.get("reward_groups")
        if not all(
            isinstance(item, Mapping)
            for item in (
                window,
                task_invalid,
                strike_timing,
                hit,
                achieved_flight,
            )
        ):
            raise PreLongGateRefused(f"semantic update {index} opportunity sections are missing")
        if (
            window.get("num_envs") != NUM_ENVS
            or window.get("rollout_steps_per_env") != ROLLOUT_STEPS_PER_UPDATE
            or window.get("rollout_sample_count") != ROLLOUT_SAMPLES_PER_UPDATE
            or type(window.get("reset_boundary")) is not str
            or not window["reset_boundary"]
        ):
            raise PreLongGateRefused(
                f"semantic update {index} fixed rollout window differs"
            )
        invalid_samples = _rollout_counter(
            task_invalid.get("observed_sample_count"),
            name=f"semantic update {index} task-invalid samples",
        )
        task_active_samples = ROLLOUT_SAMPLES_PER_UPDATE - invalid_samples
        if task_active_samples <= 0:
            raise PreLongGateRefused(
                f"semantic update {index} has no TASK_ACTIVE samples"
            )
        total_invalid_samples += invalid_samples
        task_income = _finite_number(
            task_invalid.get("task_reward_weighted_sum"),
            name=f"semantic update {index} task-invalid task income",
        )
        task_denominator = _counter(
            task_invalid.get("task_reward_eligible_denominator"),
            name=f"semantic update {index} task-invalid task denominator",
        )
        if task_income != 0.0 or task_denominator != 0:
            raise PreLongGateRefused(
                f"semantic update {index} task_valid=0 leaked task reward or eligibility"
            )
        exact_strike_ticks = _rollout_counter(
            strike_timing.get("exact_strike_tick_denominator"),
            name=f"semantic update {index} exact-strike timing ticks",
        )
        closed = _rollout_counter(
            hit.get("eligible_closed_swing_count"),
            name=f"semantic update {index} eligible closed swings",
        )
        contacts = _rollout_counter(
            hit.get("actual_contact_numerator"),
            name=f"semantic update {index} actual contacts",
        )
        if contacts > closed:
            raise PreLongGateRefused(
                f"semantic update {index} requires contacts <= eligible closed swings"
            )
        total_exact_strike_ticks += exact_strike_ticks
        total_closed_swings += closed
        total_contacts += contacts
        flight_denominator = _rollout_counter(
            achieved_flight.get("eligible_denominator"),
            name=f"semantic update {index} achieved-flight denominator",
        )
        if not isinstance(groups, list) or not groups:
            raise PreLongGateRefused(f"semantic update {index} has no opportunity Reward groups")
        group_names = []
        group_incomes = {}
        group_denominators = {}
        for group in groups:
            if not isinstance(group, Mapping) or type(group.get("group")) is not str:
                raise PreLongGateRefused(f"semantic update {index} Reward group identity differs")
            income = _finite_number(
                group.get("weighted_sum"),
                name=f"semantic update {index} {group['group']} opportunity income",
            )
            denominator = _rollout_counter(
                group.get("eligible_denominator"),
                name=f"semantic update {index} {group['group']} opportunity denominator",
            )
            if denominator == 0 and income != 0.0:
                raise PreLongGateRefused(
                    f"semantic update {index} {group['group']} income is nonzero "
                    "with zero true eligibility"
                )
            if type(group.get("eligibility_semantics")) is not str or not group[
                "eligibility_semantics"
            ]:
                raise PreLongGateRefused(
                    f"semantic update {index} {group['group']} eligibility semantics missing"
                )
            group_names.append(group["group"])
            group_incomes[group["group"]] = income
            group_denominators[group["group"]] = denominator
        if len(group_names) != len(set(group_names)):
            raise PreLongGateRefused(f"semantic update {index} has duplicate Reward groups")
        if tuple(group_names) != REQUIRED_OPPORTUNITY_REWARD_GROUPS:
            raise PreLongGateRefused(
                f"semantic update {index} Reward groups must be exactly "
                f"{REQUIRED_OPPORTUNITY_REWARD_GROUPS!r} in purpose order"
            )
        for whole_rollout_group in ("balance", "mimic"):
            if (
                group_denominators[whole_rollout_group]
                != ROLLOUT_SAMPLES_PER_UPDATE
            ):
                raise PreLongGateRefused(
                    f"semantic update {index} {whole_rollout_group} denominator must "
                    f"equal {ROLLOUT_SAMPLES_PER_UPDATE}"
                )
        if group_denominators["strike"] > exact_strike_ticks:
            raise PreLongGateRefused(
                f"semantic update {index} strike-group denominator exceeds "
                "raw task-valid exact-strike timing"
            )
        if group_denominators["outcome"] != flight_denominator:
            raise PreLongGateRefused(
                f"semantic update {index} outcome-group denominator differs from "
                "achieved flight"
            )
        for group in REQUIRED_OPPORTUNITY_REWARD_GROUPS:
            aggregate_group_incomes[group] += group_incomes[group]
            aggregate_group_denominators[group] += group_denominators[group]
        total_outcome_opportunities += flight_denominator
        unknown = _counter(
            row.get("unknown_attribution_count"),
            name=f"semantic update {index} unknown attribution",
        )
        if unknown != 0:
            raise PreLongGateRefused(f"semantic update {index} unknown attribution is nonzero")
        summaries.append(
            {
                "ppo_update": index,
                "task_invalid_samples": invalid_samples,
                "task_active_samples": task_active_samples,
                "task_invalid_task_reward": "0/0",
                "hit": f"{contacts}/{closed}",
                "exact_strike_tick_denominator": exact_strike_ticks,
                "eligible_closed_swing_count": closed,
                "achieved_flight_eligible_denominator": flight_denominator,
                "reward_groups": group_names,
                "reward_group_ledger": {
                    group: {
                        "weighted_sum": group_incomes[group],
                        "eligible_denominator": group_denominators[group],
                    }
                    for group in group_names
                },
                "unknown_attribution_count": unknown,
            }
        )
    if total_invalid_samples <= 0:
        raise PreLongGateRefused(
            "five-update semantic evidence did not exercise task_valid=0"
        )
    if total_exact_strike_ticks <= 0 or total_closed_swings <= 0:
        raise PreLongGateRefused(
            "five-update semantic evidence lacks exact-strike timing or an eligible "
            "closed swing"
        )
    for group, income in aggregate_group_incomes.items():
        if not math.isfinite(income):
            raise PreLongGateRefused(
                f"five-update aggregate {group} income is non-finite"
            )
    for group in ("balance", "mimic"):
        if (
            aggregate_group_denominators[group]
            != ROLLOUT_SAMPLES_FIVE_UPDATE_AGGREGATE
        ):
            raise PreLongGateRefused(
                f"five-update aggregate {group} denominator must equal "
                f"{ROLLOUT_SAMPLES_FIVE_UPDATE_AGGREGATE}"
            )
    if aggregate_group_incomes["balance"] == 0.0:
        raise PreLongGateRefused(
            "five-update aggregate balance income must be nonzero"
        )
    if aggregate_group_incomes["mimic"] <= 0.0:
        raise PreLongGateRefused(
            "five-update aggregate mimic income must be positive"
        )
    if profile == PROFILE_A211:
        if (
            aggregate_group_denominators["target"] <= 0
            or aggregate_group_incomes["target"] <= 0.0
        ):
            raise PreLongGateRefused(
                "A211 five-update target denominator and income must both be positive"
            )
    elif profile == PROFILE_C211:
        if (
            aggregate_group_denominators["strike"] <= 0
            or aggregate_group_incomes["strike"] <= 0.0
        ):
            raise PreLongGateRefused(
                "C211 five-update strike denominator and income must both be positive"
            )
    else:  # pragma: no cover - guarded per row, retained for fail-closed maintenance.
        raise PreLongGateRefused("five-update semantic profile is absent")
    return {
        "updates": summaries,
        "aggregate": {
            "profile": profile,
            "task_invalid_observed_sample_count": total_invalid_samples,
            "task_active_observed_sample_count": (
                ROLLOUT_SAMPLES_FIVE_UPDATE_AGGREGATE - total_invalid_samples
            ),
            "exact_strike_tick_denominator": total_exact_strike_ticks,
            "eligible_closed_swing_count": total_closed_swings,
            "actual_contact_numerator": total_contacts,
            "outcome_opportunity_denominator": total_outcome_opportunities,
            "reward_groups": {
                group: {
                    "weighted_sum": aggregate_group_incomes[group],
                    "eligible_denominator": aggregate_group_denominators[group],
                }
                for group in REQUIRED_OPPORTUNITY_REWARD_GROUPS
            },
        },
    }


def validate_prelong_gate(
    *,
    log_text: str,
    checkpoint_acceptance: Mapping[str, Any],
    semantic_updates: Optional[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if not isinstance(checkpoint_acceptance, Mapping):
        raise PreLongGateRefused("checkpoint acceptance root must be an object")
    checkpoint = validate_checkpoint_audit(
        checkpoint_acceptance.get("checkpoint", {})
    )
    safety = validate_safety_audit(
        checkpoint_acceptance.get("safety_counters", {})
    )
    economy = validate_economy_updates(log_text)
    group_income = validate_group_income_updates(log_text)
    semantics = validate_semantic_updates(semantic_updates)
    survival = validate_survival_denominators(
        safety=safety,
        semantics=semantics,
    )
    return {
        "schema_version": 1,
        "kind": "action_ball_4096x5_prelong_terminal_gate",
        "status": "PASS",
        "diagnostic_unauthorized": True,
        "num_envs": NUM_ENVS,
        "ppo_updates": EXPECTED_UPDATES,
        "checkpoint": checkpoint,
        "optimizer_health": economy,
        "reward_group_income": group_income,
        "opportunity_semantics": semantics,
        "survival_denominators": survival,
        "safety": {**safety, "unknown_attribution_count": 0},
        "authorization": "pre_long_terminal_telemetry_only",
    }


def _read_json(path: Path, *, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PreLongGateRefused(f"{name} is not readable finite JSON") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--checkpoint-acceptance", type=Path, required=True)
    parser.add_argument("--semantic-updates", type=Path)
    args = parser.parse_args(argv)
    try:
        log_text = args.run_log.read_text(encoding="utf-8")
        checkpoint = _read_json(args.checkpoint_acceptance, name="checkpoint acceptance")
        semantics = (
            None
            if args.semantic_updates is None
            else _read_json(args.semantic_updates, name="semantic updates")
        )
        if semantics is not None and not isinstance(semantics, list):
            raise PreLongGateRefused("semantic updates root must be a list")
        result = validate_prelong_gate(
            log_text=log_text,
            checkpoint_acceptance=checkpoint,
            semantic_updates=semantics,
        )
    except (OSError, UnicodeError, PreLongGateRefused) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "action_ball_4096x5_prelong_terminal_gate",
                    "status": "BLOCKED",
                    "diagnostic_unauthorized": True,
                    "reason": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
