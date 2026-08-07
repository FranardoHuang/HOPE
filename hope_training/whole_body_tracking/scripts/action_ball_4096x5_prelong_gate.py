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
# 人话:跑满 N 个 PPO update 之后,最后一份存档的编号是 N-1,不是 N。
#
# RSL-RL 的 ``OnPolicyRunner.learn`` 用 ``for it in range(start_iter, start_iter +
# num_learning_iterations)`` 迭代,并且在**循环体内**执行
# ``self.current_learning_iteration = it``;循环结束后的收尾存盘用的就是这个末值。
# 所以 ``num_learning_iterations = N`` 且 ``save_interval = 1`` 时,落盘的是
# ``model_0.pt .. model_{N-1}.pt`` —— ``model_N.pt`` 这个文件在任何预算下都不存在。
#
# 这条常量是 A211/C211 两族**唯一**的终局编号出处:发射器不再各自手抄一份
# ``model_%d.pt % EXPECTED_UPDATES``。见 tests/test_action_ball_4096x5_terminal_index.py,
# 那份测试直接读 vendored RSL-RL 的活源码核对这个约定,而不是再抄第三遍。
TERMINAL_CHECKPOINT_ITERATION = EXPECTED_UPDATES - 1
TERMINAL_CHECKPOINT_FILENAME = "model_%d.pt" % TERMINAL_CHECKPOINT_ITERATION
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

# ---------------------------------------------------------------------------
# 诊断跑 / 正式跑:reward activation ledger 那一族证据的适用范围
# ---------------------------------------------------------------------------
# 人话:下面这一族遥测(``HOPE_EFFECTIVE_REWARD_*`` /
# ``HOPE_REWARD_SAFETY_TRANSITION_`` / ``HOPE_REWARD_EPISODE_SEGMENTED_CLOSURE_``)
# **只有正式跑才会发**。诊断跑按设计根本不建那本 reward activation ledger,所以
# 「诊断跑缺这几行」不是跑坏了,是这条证据在诊断跑里**结构上不存在**。
#
# 三层查证(2026-08-07,别再重查一遍):
#   1. 它是什么 —— ``utils/effective_reward_recipe.py`` 里的
#      ``ActionBoundRewardEvidenceLedger`` / ``EffectiveRewardActivationLedger``。
#      它是一笔**两段式提交**:optimizer 之前 prepare,落一份 durable artifact,
#      optimizer 成功后 commit,再 acknowledge。它给 PPO 上了一道证据栅栏,
#      并铸出可晋级(promotable)的正式收据。
#   2. 正式跑拿它做什么 —— 把 PPO 钉在 RewardManager 缓存合同上,并生成
#      ``audit_reward_run.py`` 那条正式审计链要读的四种事件。
#   3. 诊断跑不给它是有意还是遗漏 —— **有意**。
#      ``my_on_policy_runner._effective_reward_activation_task_kind()`` 对
#      ``action_ball_diagnostic_unauthorized`` 的跑直接 ``return None``,注释原话是
#      "Diagnostic reward screens deliberately cannot mint formal evidence or
#      promotion authority";引入提交是 ``790714b3``
#      "train(n1): keep formal audits off diagnostic PPO"(2026-07-29)。
#      而且诊断跑另有一本**替代**账(``reward_ppo_economy_ledger`` → ``ECONOMY_PREFIX``),
#      运行时还有一道硬门明确禁止两本账并存:
#      "reward/PPO economy diagnostic cannot share a formal Reward ledger"。
#      所以「给诊断跑接上正式账」不是补一处遗漏,而是推翻一条有名有姓的设计裁定,
#      并且会当场撞上那道互斥门。
#
# 因此这里做的是**重定范围**,不是放松:
#   * 正式跑 —— 照旧要求满 ``EXPECTED_UPDATES`` 行(等强,缺一行仍然拒收);
#   * 诊断跑 —— 明确标注「不适用」,并且**必须是 0 行**:诊断跑一旦发出这一族,
#     说明它越权铸了正式证据,同样拒收。记录与阻断同批,不是静默跳过。
JOINT_SAFETY_PREFIX = "HOPE_JOINT_SAFETY_UPDATE_JSON="
# 两种体制各自的 joint-safety 收据事件名(活值出处:my_on_policy_runner 的
# ``_JOINT_SAFETY_EVENT`` 与 ``_commit_diagnostic_joint_safety_update``)。
# 每一跑的 joint-safety 收据都自陈自己属于哪一种,所以体制不用外部声明,直接从
# 产物里读 —— 这也是收据能自陈走了哪条分支的原因。
FORMAL_JOINT_SAFETY_EVENT = "hope_joint_safety_update"
DIAGNOSTIC_JOINT_SAFETY_EVENT = "hope_joint_safety_diagnostic_compact_update"
FORMAL_JOINT_SAFETY_STATUS = "optimizer_committed_and_ledger_acknowledged"
DIAGNOSTIC_JOINT_SAFETY_STATUS = (
    "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
)
REWARD_EVIDENCE_REGIME_FORMAL = "formal_reward_activation_ledger"
REWARD_EVIDENCE_REGIME_DIAGNOSTIC = "diagnostic_no_reward_activation_ledger"
# 正式跑里**被阻断**的两条:今天已经有消费方在读,重定范围之后强度不变。
REWARD_ACTIVATION_LEDGER_REQUIRED_PREFIXES = (
    GROUP_PREFIX,
    "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=",
)
# 同族另外两条:只**记录**行数,不新增阻断。它们的发射条件比上面两条更窄
# (upper_safe 正式跑就不发 per-action/closure),这里不借重定范围之机偷偷加严。
REWARD_ACTIVATION_LEDGER_OBSERVED_PREFIXES = (
    "HOPE_EFFECTIVE_REWARD_ACTIVATION_UPDATE_JSON=",
    "HOPE_REWARD_EPISODE_SEGMENTED_CLOSURE_UPDATE_JSON=",
)
REWARD_ACTIVATION_LEDGER_PREFIXES = (
    *REWARD_ACTIVATION_LEDGER_REQUIRED_PREFIXES,
    *REWARD_ACTIVATION_LEDGER_OBSERVED_PREFIXES,
)
REWARD_EVIDENCE_SCOPE_KIND = "action_ball_reward_activation_evidence_scope_v1"
# 诊断跑里,这三项严格零安全计数改由哪一族供给。**不是**"没观测到所以记 0" ——
# 那正是 oracle32 那次重定范围明令禁止的假收据。
DIAGNOSTIC_STRICT_ZERO_SOURCE_PREFIX = "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
DIAGNOSTIC_STRICT_ZERO_SOURCE_COUNTERS = (
    "termination_reason_joint_qdes_forbidden_count",
    "termination_reason_joint_actual_forbidden_count",
)

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

# 2026-08-07 Franco 裁定三:**取消实际-q 硬超限那条门**(不是删遥测)。
# 依据是 build_1 的 ``Episode_Termination/actual_q_hard_limit_audit`` 从 iter 20 起恒 0 ——
# 策略学会之后本来就不硬越限,这条轴买不到东西,却会在 5 个 update 的新策略上直接拒收。
#
# 留在 STRICT_ZERO 里的四条**不是**"硬超限",逐条说明为什么留:
#   * joint_qdes_forbidden_terminal_count —— ActionBall 投影模式下它只对 NaN/Inf 的 q_des 触发。
#     那是数值 bug,不是学得会的行为。
#   * joint_actual_forbidden_terminal_count —— 这条**必须**是 0,因为该 DoneTerm 已配成
#     ``terminate=False``;它一旦非零,说明"只记账不 reset"的接线退化回了 reset。
#     它验的是我们自己的接线,不是机器人的行为。
#   * strict_hard_termination_count / nonfinite_count —— 同样是实现层数值健康。
STRICT_ZERO_SAFETY_COUNTERS = (
    "joint_qdes_forbidden_terminal_count",
    "joint_actual_forbidden_terminal_count",
    "strict_hard_termination_count",
    "nonfinite_count",
)
# 实际-q 机械硬边:**照记不照拦**。计数进收据、非零时进摘要 WARN,
# 但不再让这一跑被拒收。"取消 != 静默删除"。
REPORTED_HARD_EDGE_COUNTERS = (
    "actual_hard_edge_event_count",
    "actual_hard_terminal_count",
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

# 接线之后这三张表就是承重的,所以在导入期跟生产方对一次**活值**——不是文件 SHA。
# 指纹只能证明"字节没动",证明不了"两边说的是同一件事"(见 §5.6.13 的同型教训)。
# 各自漂了会怎样:
#   * WAIT 档位错位 -> 逐档守恒式照样成立,但比的是错档号,拒绝面整体走形;
#   * exp/cauchy 归类不一致 -> 核函数字符串检查会拒掉正确收据,或放过错的;
#   * mimic 项集合不一致 -> 生产方少写一项时长度才对不上,拒绝理由会指错地方。
# ``BRIDGE_TIMING_FIELDS`` 对不上——生产方那份 ``_bridge_timing_names`` 是实例属性,
# 导入期取不到;它靠 ``_exact_keys`` 的精确键集在运行时兜底(少一项/多一项都会拒)。
if (
    BRIDGE_WAIT_COHORTS != tuple(_SEMANTICS._PRELONG_BRIDGE_WAIT_COHORTS)
    or set(BRIDGE_MIMIC_TERMS)
    != (
        set(_SEMANTICS._PRELONG_MIMIC_EXP_TERMS)
        | set(_SEMANTICS._PRELONG_MIMIC_CAUCHY_TERMS)
    )
    or set(BRIDGE_CAUCHY_MIMIC_TERMS)
    != set(_SEMANTICS._PRELONG_MIMIC_CAUCHY_TERMS)
):
    raise RuntimeError(
        "pre-long gate reveal-bridge contract differs from the semantic producer"
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


def classify_reward_evidence_regime(log_text: str) -> str:
    """Read the run's own joint-safety receipts to decide which regime it ran in.

    人话:不接受外部声明「我是诊断跑」。每个 PPO update 的 joint-safety 收据里都写着
    自己是正式账还是诊断紧凑账;这里只认那份自陈。一跑里两种混着出现 = 拒收。
    """

    if type(log_text) is not str:
        raise PreLongGateRefused("run log text must be a string")
    rows = _marker_rows(log_text, prefix=JOINT_SAFETY_PREFIX, name="joint-safety")
    if not rows:
        raise PreLongGateRefused(
            "run log has no joint-safety receipt, so its reward-evidence regime "
            "cannot be established"
        )
    observed = {
        (row.get("event"), row.get("status"))
        for row in rows
    }
    formal = {(FORMAL_JOINT_SAFETY_EVENT, FORMAL_JOINT_SAFETY_STATUS)}
    diagnostic = {
        (DIAGNOSTIC_JOINT_SAFETY_EVENT, DIAGNOSTIC_JOINT_SAFETY_STATUS)
    }
    if observed == formal:
        return REWARD_EVIDENCE_REGIME_FORMAL
    if observed == diagnostic:
        return REWARD_EVIDENCE_REGIME_DIAGNOSTIC
    raise PreLongGateRefused(
        "joint-safety receipts do not declare exactly one reward-evidence "
        "regime; observed %r" % (sorted(map(repr, observed)),)
    )


def reward_activation_evidence_scope(
    *, log_text: str, expected_updates: int = EXPECTED_UPDATES
) -> dict[str, Any]:
    """Decide whether the reward-activation evidence family applies to this run.

    Formal runs keep the original strength: every required prefix must carry
    exactly ``expected_updates`` markers.  Diagnostic runs never build the
    ledger, so the family is declared **not applicable** — and must be
    completely absent, because a diagnostic screen emitting formal evidence
    would mean it minted promotion authority it is not allowed to have.
    """

    if type(expected_updates) is not int or expected_updates < 1:
        raise PreLongGateRefused(
            "reward-activation evidence scope needs a positive update budget"
        )
    regime = classify_reward_evidence_regime(log_text)
    observed_rows = {
        prefix: len(
            _marker_rows(
                log_text, prefix=prefix, name="reward-activation evidence"
            )
        )
        for prefix in REWARD_ACTIVATION_LEDGER_PREFIXES
    }
    if regime == REWARD_EVIDENCE_REGIME_FORMAL:
        applicable = True
        required_rows_per_prefix = expected_updates
        for prefix in REWARD_ACTIVATION_LEDGER_REQUIRED_PREFIXES:
            if observed_rows[prefix] != expected_updates:
                raise PreLongGateRefused(
                    "formal run reward-activation evidence %s lacks exactly %d "
                    "markers; got %d"
                    % (prefix, expected_updates, observed_rows[prefix])
                )
        reason = (
            "formal run: the reward activation ledger is live, so its terminal "
            "evidence is required"
        )
        strict_zero_source = "reward_safety_transition_markers"
    else:
        applicable = False
        required_rows_per_prefix = 0
        for prefix, count in sorted(observed_rows.items()):
            if count != 0:
                raise PreLongGateRefused(
                    "diagnostic run emitted formal reward-activation evidence "
                    "%s (%d markers); a diagnostic screen may not mint "
                    "promotion authority" % (prefix, count)
                )
        reason = (
            "这一跑没有 reward activation ledger(诊断跑按设计不建这本账,见 "
            "my_on_policy_runner._effective_reward_activation_task_kind 与提交 "
            "790714b3),因此 reward-activation 族终局证据不适用;它承担的严格零"
            "安全计数改由 %s 的 %s 供给,不是当作 0 跳过。"
            % (
                DIAGNOSTIC_STRICT_ZERO_SOURCE_PREFIX,
                ", ".join(DIAGNOSTIC_STRICT_ZERO_SOURCE_COUNTERS),
            )
        )
        strict_zero_source = "exact_behavior_termination_reason_counters"
    return {
        "schema_version": 1,
        "kind": REWARD_EVIDENCE_SCOPE_KIND,
        "regime": regime,
        "applicable": applicable,
        "reason": reason,
        "required_prefixes": list(REWARD_ACTIVATION_LEDGER_REQUIRED_PREFIXES),
        "observed_only_prefixes": list(
            REWARD_ACTIVATION_LEDGER_OBSERVED_PREFIXES
        ),
        "required_rows_per_prefix": required_rows_per_prefix,
        "observed_rows_per_prefix": dict(sorted(observed_rows.items())),
        "strict_zero_counter_source": strict_zero_source,
    }


def validate_checkpoint_audit(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the launcher's safe checkpoint audit, not raw checkpoint bytes."""

    if not isinstance(checkpoint, Mapping):
        raise PreLongGateRefused("checkpoint audit is missing")
    if (
        checkpoint.get("filename_iteration") != TERMINAL_CHECKPOINT_ITERATION
        or checkpoint.get("embedded_iteration") != TERMINAL_CHECKPOINT_ITERATION
        or checkpoint.get("all_tensors_finite") is not True
        or checkpoint.get("load_mode") != "torch_weights_only"
    ):
        raise PreLongGateRefused(
            "checkpoint audit must bind %s, embedded iter=%d, and finite "
            "weights-only load"
            % (TERMINAL_CHECKPOINT_FILENAME, TERMINAL_CHECKPOINT_ITERATION)
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
    return {
        "iteration": TERMINAL_CHECKPOINT_ITERATION,
        "all_tensors_finite": True,
        "groups": summary,
    }


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
    # 裁定三:硬边计数仍然必须**存在且是合法非负整数**(缺失/畸形照样拒收),
    # 只是它的值不再阻断。
    hard_edge = {
        name: _counter(safety.get(name), name=f"safety {name}")
        for name in REPORTED_HARD_EDGE_COUNTERS
    }
    hard_edge_warnings = [
        f"WARN actual-q hard edge observed: {name}={value}"
        for name, value in sorted(hard_edge.items())
        if value != 0
    ]

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
        "actual_hard_edge_counters": hard_edge,
        "actual_hard_edge_blocking": False,
        "actual_hard_edge_warnings": hard_edge_warnings,
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
# 2026-08-07 收紧一档:光写"什么条件下结束"是**一句没人核对的承诺**。
# ``action_rate_clamped`` 那条申报写的是"等 policy_std_max 掉到 0.381 就会解冻",
# 而同一份收据里 s15r1 两格的 ``policy_std_max`` 是
# ``1.00198 -> 1.00729``(C0)/ ``1.00191 -> 1.00661``(C1)—— **五个 update 一路在涨**,
# 因为 entropy_coef=0.01 的自适应 KL 就是在把 std 往上推。承诺的解冻点不但没靠近,
# 还在后退。真正的参照更狠:唯一已知能打球的实现 ``build_1`` 训到 21896 iter 收敛时,
# ``action_rate`` 每步仍是 ``-0.0216~-0.0241`` = 权重 -0.1 x dt 0.02 反推
# ``||Δa||² = 10.8~12.05``,**仍在 9.0 之上** —— 也就是说这个封顶从第 0 步到收敛
# 全程焊死,"会结束"这件事在这条谱系上根本不发生。
#
# 所以申报现在必须把结束条件写成**收据里真有的那个量**(``ends_when_metric`` /
# ``ends_when_threshold`` / ``ends_when_direction``),门会拿观测窗自己核对:
#   * 那个量在窗内朝阈值反方向走 -> 申报被证伪,拒收;
#   * 那个量已经越过阈值、项却还是常数 -> 申报被证伪,拒收;
#   * 朝阈值走 -> 放行,并把量的首末值写进收据。
# 这样"暂时饱和"和"永久死项"就不再靠自述区分。
#
# 为什么不顺手把"恒零"也拒掉:A211 的 target/outcome 项在一次一球未碰的 5-update 冒烟里
# 本来就该恒零,拒掉它等于要求一个没训过的策略已经会打球(§5.6.8 同型的"门定错范围")。
# 恒零项如实进收据,不进拒收。
# 2026-08-08 结案:上面那条现役实例已经不再是现役 —— Franco 裁定"超限形状照开源对齐",
# ``action_rate_clamped`` 退役(weight 0 = IsaacLab 直接跳过),一阶平滑换回上游无封顶的
# ``action_rate_l2`` −0.1(BeyondMimic / mjlab-tracking / unitree_rl_lab-mimic 三家同值同形)。
# 无封顶的 raw 随 ‖Δa‖² = 2σ²·χ²₃₁ 逐 update 变动,所以它不再是常数项,也就不需要任何申报。
# **本表因此仍然是空的,而且这道门一行没改** —— 它是把这件事逼出来的那道门,不是被它绕开的。
# 下面这套申报机制原样留着:下一个想"申报一个常数项"的人仍然要写出机制、写出收据里真有的
# 那个量、并接受门用观测窗核对。见 exp §5.6.25。
_ENDS_WHEN_DIRECTIONS = ("falls_to", "rises_to")

#: 结束条件可以引用的收据字段,必须是 ``policy`` 块里逐 update 都在的那三个标量。
#: 不许引用一个收据里不存在的量 —— 那等于回到无人核对的自述。
_ENDS_WHEN_METRICS = ("policy_std_min", "policy_std_mean", "policy_std_max")

DECLARED_CONSTANT_REWARD_TERMS: dict[str, dict[str, Any]] = {}


def _validate_declared_constant_reward_terms() -> None:
    """Refuse a silently blanked or unfalsifiable declaration table.

    A future "soft delete" that empties ``mechanism``/``ends_when`` would turn the
    allowlist into a blanket exemption without touching the refusal code path.  The
    telemetry triple is required for the same reason: an end condition that names no
    receipt field can never be contradicted by the run it exempts.
    """

    for term, declaration in DECLARED_CONSTANT_REWARD_TERMS.items():
        if type(term) is not str or not term:
            raise PreLongGateRefused("declared constant reward term names must be strings")
        if not isinstance(declaration, Mapping):
            raise PreLongGateRefused(
                f"declared constant reward term {term} has no declaration object"
            )
        metric = declaration.get("ends_when_metric")
        if metric not in _ENDS_WHEN_METRICS:
            raise PreLongGateRefused(
                f"declared constant reward term {term} must end on one of "
                f"{', '.join(_ENDS_WHEN_METRICS)}, got {metric!r}"
            )
        direction = declaration.get("ends_when_direction")
        if direction not in _ENDS_WHEN_DIRECTIONS:
            raise PreLongGateRefused(
                f"declared constant reward term {term} needs ends_when_direction in "
                f"{_ENDS_WHEN_DIRECTIONS}, got {direction!r}"
            )
        threshold = declaration.get("ends_when_threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise PreLongGateRefused(
                f"declared constant reward term {term} needs a numeric ends_when_threshold"
            )
        if not math.isfinite(float(threshold)):
            raise PreLongGateRefused(
                f"declared constant reward term {term} ends_when_threshold must be finite"
            )
        for field in ("mechanism", "ends_when"):
            text = declaration.get(field)
            if type(text) is not str or not text.strip():
                raise PreLongGateRefused(
                    f"declared constant reward term {term} has an empty {field}"
                )


def _audit_declared_end_condition(
    term: str,
    declaration: Mapping[str, Any],
    per_update_policy: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Check a constant term's declared end condition against the run's own telemetry.

    人话:申报说"等 X 掉到 T 这项就会解冻"。这里就拿收据里的 X 逐 update 核对。
    只有"X 正在朝 T 走"才算这个常数是暂时的;X 在后退、或者 X 已经越过 T 而项
    还是常数,都说明申报是错的,那这个豁免就不该给。
    """

    metric = declaration["ends_when_metric"]
    threshold = float(declaration["ends_when_threshold"])
    direction = declaration["ends_when_direction"]
    series = [float(row[metric]) for row in per_update_policy]
    first, last = series[0], series[-1]
    reached = last <= threshold if direction == "falls_to" else last >= threshold
    if reached:
        raise PreLongGateRefused(
            "declared constant reward term %s claims it unfreezes once %s %s %g, but %s is "
            "already %g at the last observed update and the term is still bitwise constant: "
            "the declared end condition is satisfied and nothing unfroze"
            % (term, metric, direction.replace("_", " "), threshold, metric, last)
        )
    approaching = last < first if direction == "falls_to" else last > first
    if not approaching:
        raise PreLongGateRefused(
            "declared constant reward term %s claims it unfreezes once %s %s %g, but %s moved "
            "%g -> %g across the observed window — away from the threshold, not toward it: a "
            "saturation whose declared exit is receding is a permanently dead term, not a "
            "temporary offset"
            % (term, metric, direction.replace("_", " "), threshold, metric, first, last)
        )
    return {
        "metric": metric,
        "threshold": threshold,
        "direction": direction,
        "first_observed": first,
        "last_observed": last,
    }


def _reward_term_signal_ledger(
    per_update_terms: Sequence[Mapping[str, float]],
    per_update_policy: Sequence[Mapping[str, float]],
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
                "declared_end_condition_audit": _audit_declared_end_condition(
                    name, declaration, per_update_policy
                ),
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
    per_update_policy: list[dict[str, float]] = []
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
        per_update_policy.append(
            {
                "policy_std_min": std_min,
                "policy_std_mean": std_mean,
                "policy_std_max": std_max,
            }
        )
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
        "reward_term_signal": _reward_term_signal_ledger(
            per_update_terms, per_update_policy
        ),
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

    **已接线(2026-08-07)。** ``validate_semantic_updates`` 逐 update 调用本函数;
    三个返回值依次是「本 update 的收据摘要」「逐 WAIT 档寿命(下一 update 的比较基准)」
    「权威身份(第一个 update 之后不许漂)」,汇总后写进
    ``aggregate.reveal_to_playback_bridge``,收据自陈这一步真跑过了。

    人话:这块记录管的是**揭示那一刻到 clip 真正开始推进之间**那段窗口的账 ——
    ``5..25`` 共 21 个 WAIT 档各自 `揭示 = 开始回放 + 开始前就终止 + 被截断`、
    七项权威 SHA 跨 5 个 update 不许漂、隐藏等待期的 task 收入必须恰好是 `0`、
    逐 mimic 项的核函数/分母/收入、以及这段窗口内的边界安全量。

    **别把它和 ``bridge_ramp_command_steps`` 混成一件事。** 那条 ramp(`34f8cf25`)
    在 ``scripts/train.py`` 的 teacher-q_des oracle 里,作用是把揭示那一 tick 的
    ``2.24 rad`` 阶跃摊到约 `35` 步(§5.6.6 实测:只多活 `1` tick)。**ramp 的存废跟着
    「出生姿态要不要改成 frame 0」走;本记录不跟** —— 它的六个块没有一个引用出生姿态
    或阶跃幅度。出生改成(接地后的)frame 0 只会换掉这里的**取值**
    (``pre_swing_wait_s`` / ``timing_contract_sha256`` 等),一条**检查**都不会失效;
    反而更承重,因为阶跃这个借口没了之后,回放开始前的每一次死亡都是纯平衡/plant 故障,
    而这是唯一按 WAIT 档把它们数出来的账。
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
    bridge_summaries: list[dict[str, Any]] = []
    bridge_lifetime: Optional[dict[int, dict[str, int]]] = None
    bridge_authority: Optional[dict[str, Any]] = None
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
        # 揭示->回放那段窗口的账。第一个 update 没有比较基准(两个 None),之后逐 update
        # 用上一轮的逐档寿命做单调性比较、用第一轮的 authority 做整块相等比较。
        bridge_summary, bridge_lifetime, observed_authority = _validate_reveal_bridge(
            row.get("reveal_to_playback_bridge"),
            profile=row_profile,
            update=index,
            previous_lifetime=bridge_lifetime,
            expected_authority=bridge_authority,
        )
        if bridge_authority is None:
            bridge_authority = observed_authority
        bridge_summaries.append(bridge_summary)
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
            # 收据自陈:这一步不是"只出结论、没人读"的位。authority 里带着七项权威 SHA
            # 与 WAIT 档表,读收据的人不必回头翻代码就知道这一跑的桥绑在哪份合同上。
            "reveal_to_playback_bridge": {
                "consumed_semantics": (
                    "every schema-v3 reveal-to-playback bridge block was strictly "
                    "consumed: exact key sets, active fail-closed status, per-WAIT "
                    "lifetime conservation and cross-update monotonicity, one "
                    "unchanging authority identity, and the hidden-WAIT income rule"
                ),
                "updates_consumed": len(bridge_summaries),
                "authority": bridge_authority,
                "cumulative_reveal_count": bridge_summaries[-1][
                    "cumulative_reveal_count"
                ],
                "newly_revealed_count": sum(
                    summary["reveal_count_delta"] for summary in bridge_summaries
                ),
                "final_wait_cohort_lifetime": [
                    {"wait_ticks": wait, **bridge_lifetime[wait]}
                    for wait in BRIDGE_WAIT_COHORTS
                ],
                "per_update": bridge_summaries,
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
    # Reward-group income is minted by the formal reward activation ledger, so
    # it shares the ledger's applicability.  The scope block below is emitted
    # either way: a diagnostic run must be able to show, in its own receipt,
    # that this evidence was declared not applicable rather than skipped.
    reward_evidence_scope = reward_activation_evidence_scope(log_text=log_text)
    group_income = (
        validate_group_income_updates(log_text)
        if reward_evidence_scope["applicable"]
        else {
            "updates": [],
            "applicable": False,
            "reward_activation_evidence_scope": reward_evidence_scope,
            "eligibility_boundary": (
                "reward-group income is minted by the formal reward activation "
                "ledger; this run has none, so no group-income claim is made"
            ),
        }
    )
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
        # 收据自陈走了哪条分支:这一跑属于哪种 reward 证据体制、这一族要不要、
        # 实际读到几行。诊断跑读到 0 行不再是"沉默的跳过",而是写进 PASS 收据里的
        # 一条明账。
        "reward_activation_evidence_scope": reward_evidence_scope,
        "reward_group_income": group_income,
        "opportunity_semantics": semantics,
        "survival_denominators": survival,
        "safety": {**safety, "unknown_attribution_count": 0},
        # 「WARN 必进摘要」:硬边观测直接抬到 gate 结果的顶层,不埋在 safety 子树里。
        "warnings": list(safety["actual_hard_edge_warnings"]),
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
