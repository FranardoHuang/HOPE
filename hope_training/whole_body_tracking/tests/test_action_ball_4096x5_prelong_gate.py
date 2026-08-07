"""Tests for the shared fail-closed 4096x5 pre-long telemetry gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "action_ball_4096x5_prelong_gate.py"
)
SPEC = importlib.util.spec_from_file_location("action_ball_4096x5_prelong_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)

# 揭示->回放那块记录的夹具只有一份(A/C launcher 的终局门测试共用同一份),见模块 docstring。
_BRIDGE_SPEC = importlib.util.spec_from_file_location(
    "prelong_bridge_fixture",
    Path(__file__).resolve().parent / "prelong_bridge_fixture.py",
)
assert _BRIDGE_SPEC is not None and _BRIDGE_SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(_BRIDGE_SPEC)
_BRIDGE_SPEC.loader.exec_module(BRIDGE)


def _checkpoint_acceptance():
    # 跑满 5 个 update 之后落盘的末位是 model_4.pt / iter=4:RSL-RL 的迭代变量
    # 在循环体内取 0..N-1,收尾存盘用的就是那个末值。这里刻意写死字面量 4 而不是
    # 从被测模块读常量(那样只会自证);字面量本身由
    # test_action_ball_4096x5_terminal_index.py 直接读 RSL-RL 活源码钉住。
    return {
        "checkpoint": {
            "filename_iteration": 4,
            "embedded_iteration": 4,
            "load_mode": "torch_weights_only",
            "all_tensors_finite": True,
            "tensor_groups": {
                name: {"tensor_count": 1, "element_count": 2}
                for name in (
                    "model",
                    "optimizer",
                    "actor_normalizer",
                    "critic_normalizer",
                )
            },
        },
        "safety_counters": {
            "observed_ppo_updates": 5,
            "actual_hard_edge_event_count": 0,
            "actual_hard_terminal_count": 0,
            "joint_qdes_forbidden_terminal_count": 0,
            "joint_actual_forbidden_terminal_count": 0,
            "strict_hard_termination_count": 0,
            "table_contact_count": 0,
            "nonfinite_count": 0,
            "base_fell_tilt_terminal_count": 0,
            "base_too_low_terminal_count": 0,
            "physical_fall_by_reason_phase": {
                reason: {phase: 0 for phase in GATE.PHYSICAL_FALL_PHASES}
                for reason in GATE.PHYSICAL_FALL_REASONS
            },
            "table_contact_by_phase": {
                phase: 0 for phase in GATE.PHYSICAL_FALL_PHASES
            },
            "task_wait_started_by_update": [12] * 5,
            "task_wait_started_count": 60,
            "task_reveal_reached_by_update": [10] * 5,
            "task_reveal_reached_count": 50,
        },
    }


def _economy(update, *, terms=None, policy=None):
    # 注:``motion`` 与 ``action_rate_l2`` 逐 update 变化都是**故意**的。本夹具原本
    # 把两项都写成五个 update 恒等于同一个数 —— 那正是 s15r1 的病灶形状,而把病灶写进
    # 默认夹具会让这个模块里的每一条测试都默认接受它。默认夹具现在是一份**健康**的跑,
    # 冻结项另有专门的测试(见下面"常数奖励项"一节)。
    if terms is None:
        terms = {
            "motion": 1.0 + 0.25 * update,
            "task": 0.0,
            "action_rate_l2": -3538.945068 + 1.5 * update,
        }
    if policy is None:
        policy = {
            "policy_std_min": 0.01,
            "policy_std_mean": 0.02,
            "policy_std_max": 0.03,
        }
    return {
        "event": "hope_action_ball_reward_ppo_economy_update",
        "schema_version": 1,
        "status": "PASS",
        "ppo_update": update,
        "gate": {
            "num_envs": 4096,
            "steps_per_env_per_update": 24,
            "rollout_samples_per_update": 98304,
        },
        "reward": {
            "explained_variance": 0.1,
            "per_term_weighted_dt_sum": dict(terms),
            "per_term_eligible_denominator": {name: 98304 for name in terms},
        },
        "ppo": {
            "learning_rate": 1.0e-4,
            "approx_kl": 0.01,
            "clip_fraction": 0.2,
        },
        "gradient": {"pre_clip_total_grad_norm": 0.5},
        "policy": dict(policy),
    }


def _groups(update):
    return {
        "event": "hope_effective_reward_activation_by_action_update",
        "schema_version": 2,
        "ppo_update": update,
        "actions": [
            {
                "action_id": "Take_061_unit04_BH",
                "reward_groups": [
                    {
                        "group": "motion",
                        "eligibility": "reward_manager_evaluated_active_group_terms",
                        "eligible_sample_count": 98304,
                        "weighted_sum": 4.0,
                    },
                    {
                        "group": "task",
                        "eligibility": "reward_manager_evaluated_active_group_terms",
                        "eligible_sample_count": 98304,
                        "weighted_sum": 0.0,
                    },
                ],
            }
        ],
    }


def _joint_safety(update, *, formal=True):
    """每一跑的 joint-safety 收据都自陈它属于哪种 reward 证据体制。"""

    if formal:
        return GATE.JOINT_SAFETY_PREFIX + json.dumps(
            {
                "event": GATE.FORMAL_JOINT_SAFETY_EVENT,
                "schema_version": 2,
                "status": GATE.FORMAL_JOINT_SAFETY_STATUS,
                "ppo_update": update,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return GATE.JOINT_SAFETY_PREFIX + json.dumps(
        {
            "event": GATE.DIAGNOSTIC_JOINT_SAFETY_EVENT,
            "schema_version": 1,
            "status": GATE.DIAGNOSTIC_JOINT_SAFETY_STATUS,
            "ppo_update": update,
            "formal_authority": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _reward_safety(update):
    return "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=" + json.dumps(
        {
            "event": "hope_reward_safety_transition_update",
            "schema_version": 2,
            "ppo_update": update,
            "coverage": "complete_update",
            "terminal_transitions": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _log():
    """一份**正式跑**的日志:它有 reward activation ledger,所以那一族全在。"""

    lines = []
    for update in range(5):
        lines.extend(
            (
                _joint_safety(update, formal=True),
                GATE.ECONOMY_PREFIX
                + json.dumps(_economy(update), sort_keys=True, separators=(",", ":")),
                GATE.GROUP_PREFIX
                + json.dumps(_groups(update), sort_keys=True, separators=(",", ":")),
                _reward_safety(update),
            )
        )
    return "\n".join(lines) + "\n"


def _diagnostic_log():
    """一份**诊断跑**的日志:结构上就没有 reward activation ledger 那一族。

    对照实测(pod1 ``c0_scale4096_s16r1/run.log``):economy / semantics / joint-safety
    各 5 行,``HOPE_EFFECTIVE_REWARD_*`` 与 ``HOPE_REWARD_SAFETY_TRANSITION_`` 各 0 行。
    """

    lines = []
    for update in range(5):
        lines.extend(
            (
                _joint_safety(update, formal=False),
                GATE.ECONOMY_PREFIX
                + json.dumps(_economy(update), sort_keys=True, separators=(",", ":")),
            )
        )
    return "\n".join(lines) + "\n"


def _semantic(
    update,
    *,
    profile=GATE.PROFILE_A211,
    contacts=0,
    closed=9,
    exact_strike_ticks=9,
    flight_denominator=0,
    invalid_samples=7,
):
    if profile == GATE.PROFILE_A211:
        strike_income = 0.0
        target_income = 2.0
        target_denominator = 3
    elif profile == GATE.PROFILE_C211:
        strike_income = 2.0
        target_income = 0.0
        target_denominator = 0
    else:
        raise AssertionError(profile)
    return {
        "event": GATE.SEMANTIC_EVENT,
        "schema_version": GATE.SEMANTIC_SCHEMA_VERSION,
        "profile": profile,
        "ppo_update": update,
        "window": {
            "num_envs": 4096,
            "rollout_steps_per_env": 24,
            "rollout_sample_count": 98304,
            "reset_boundary": "same once-per-PPO-update transaction",
        },
        "task_invalid": {
            "observed_sample_count": invalid_samples,
            "task_reward_weighted_sum": 0.0,
            "task_reward_eligible_denominator": 0,
        },
        "strike_timing": {
            "exact_strike_tick_denominator": exact_strike_ticks,
        },
        "hit": {
            "eligible_closed_swing_count": closed,
            "actual_contact_numerator": contacts,
        },
        "achieved_flight": {"eligible_denominator": flight_denominator},
        "reveal_to_playback_bridge": BRIDGE.reveal_bridge(update, profile=profile),
        "reward_groups": [
            {
                "group": "balance",
                "weighted_sum": 4.0,
                "eligible_denominator": 98304,
                "eligibility_semantics": "all_rollout_samples",
            },
            {
                "group": "mimic",
                "weighted_sum": 4.0,
                "eligible_denominator": 98304,
                "eligibility_semantics": "phase_eligible_mimic_samples",
            },
            {
                "group": "strike",
                "weighted_sum": strike_income,
                "eligible_denominator": exact_strike_ticks,
                "eligibility_semantics": "exact_strike_timing_ticks",
            },
            {
                "group": "target",
                "weighted_sum": target_income,
                "eligible_denominator": target_denominator,
                "eligibility_semantics": "task_valid_contact_target_opportunities",
            },
            {
                "group": "outcome",
                "weighted_sum": 0.0,
                "eligible_denominator": flight_denominator,
                "eligibility_semantics": "eligible_achieved_flights",
            },
        ],
        "unknown_attribution_count": 0,
    }


def _semantics(**kwargs):
    return [
        _semantic(
            update,
            invalid_samples=(7 if update == 0 else 0),
            **kwargs,
        )
        for update in range(5)
    ]


def test_gate_accepts_exact_five_updates_and_preserves_zero_over_c():
    result = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=_checkpoint_acceptance(),
        semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
    )
    assert result["status"] == "PASS"
    assert result["diagnostic_unauthorized"] is True
    assert result["ppo_updates"] == 5
    assert result["survival_denominators"]["task_reveal_reached_count"] == 50
    assert (
        result["survival_denominators"]["task_active_observed_sample_count"]
        == 4096 * 24 * 5 - 7
    )
    assert result["survival_denominators"]["nominal_strike_reached_count"] == 45
    assert result["survival_denominators"]["eligible_closed_swing_count"] == 45
    aggregate = result["opportunity_semantics"]["aggregate"]
    assert aggregate["profile"] == GATE.PROFILE_A211
    assert aggregate["actual_contact_numerator"] == 0
    assert aggregate["outcome_opportunity_denominator"] == 0
    assert (
        aggregate["reward_groups"]["balance"]["eligible_denominator"]
        == 4096 * 24 * 5
    )
    assert (
        aggregate["reward_groups"]["mimic"]["eligible_denominator"]
        == 4096 * 24 * 5
    )
    assert result["opportunity_semantics"]["updates"][0]["hit"] == "0/9"
    assert (
        result["opportunity_semantics"]["updates"][0][
            "achieved_flight_eligible_denominator"
        ]
        == 0
    )


# --------------------------------------------------------------------------------------------- #
# 2026-08-07 Franco 裁定三:实际-q 机械硬边"照记不照拦"
# --------------------------------------------------------------------------------------------- #
def test_actual_q_hard_edge_is_measured_reported_and_never_blocking():
    """制造一次真实的实际-q 硬越限:遥测必须记到、收据里看得见、而且**不再阻断**。

    依据是 build_1 自己的 ``actual_q_hard_limit_audit`` 从 iter 20 起恒 0 ——
    策略学会之后本来就不硬越限,拿它去卡一个 5 个 update 的新策略只会拒收正常的跑。
    """

    acceptance = _checkpoint_acceptance()
    acceptance["safety_counters"]["actual_hard_edge_event_count"] = 7
    acceptance["safety_counters"]["actual_hard_terminal_count"] = 3
    result = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=acceptance,
        semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
    )
    # 不阻断
    assert result["status"] == "PASS"
    # 记到了,而且是逐位的原值,不是一个布尔
    assert result["safety"]["actual_hard_edge_counters"] == {
        "actual_hard_edge_event_count": 7,
        "actual_hard_terminal_count": 3,
    }
    assert result["safety"]["actual_hard_edge_blocking"] is False
    # 「WARN 必进摘要」:抬到 gate 结果顶层,不埋在 safety 子树里
    assert result["warnings"] == [
        "WARN actual-q hard edge observed: actual_hard_edge_event_count=7",
        "WARN actual-q hard edge observed: actual_hard_terminal_count=3",
    ]
    # 而且它不能悄悄混进 strict-zero 集合
    assert "actual_hard_edge_event_count" not in result["safety"]["strict_zero_counters"]


def test_zero_hard_edge_still_reports_an_empty_warning_list():
    result = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=_checkpoint_acceptance(),
        semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
    )
    assert result["warnings"] == []
    assert result["safety"]["actual_hard_edge_counters"] == {
        "actual_hard_edge_event_count": 0,
        "actual_hard_terminal_count": 0,
    }


@pytest.mark.parametrize(
    "name",
    (
        "actual_hard_edge_event_count",
        "actual_hard_terminal_count",
    ),
)
def test_hard_edge_counter_may_stop_blocking_but_may_not_go_missing(name):
    """取消阻断 != 允许它消失。缺失/畸形照样 fail closed。"""

    acceptance = _checkpoint_acceptance()
    acceptance["safety_counters"].pop(name)
    with pytest.raises(GATE.PreLongGateRefused):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=acceptance,
            semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
        )
    acceptance = _checkpoint_acceptance()
    acceptance["safety_counters"][name] = -1
    with pytest.raises(GATE.PreLongGateRefused):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=acceptance,
            semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
        )


@pytest.mark.parametrize(
    "name",
    (
        "joint_qdes_forbidden_terminal_count",
        "joint_actual_forbidden_terminal_count",
        "strict_hard_termination_count",
        "nonfinite_count",
    ),
)
def test_the_four_implementation_counters_still_block(name):
    """变异测试:如果有人把这四条也一并"放宽",这里必须红。

    它们验的是数值健康与"terminate=False 有没有退化回 reset",不是可学会的行为。
    """

    acceptance = _checkpoint_acceptance()
    acceptance["safety_counters"][name] = 1
    with pytest.raises(GATE.PreLongGateRefused):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=acceptance,
            semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
        )


def test_c211_gate_requires_strike_signal_but_not_initial_contact_or_outcome():
    result = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=_checkpoint_acceptance(),
        semantic_updates=_semantics(
            profile=GATE.PROFILE_C211,
            contacts=0,
            closed=9,
            flight_denominator=0,
        ),
    )

    aggregate = result["opportunity_semantics"]["aggregate"]
    assert aggregate["profile"] == GATE.PROFILE_C211
    assert aggregate["actual_contact_numerator"] == 0
    assert aggregate["outcome_opportunity_denominator"] == 0
    assert aggregate["reward_groups"]["strike"] == {
        "weighted_sum": 10.0,
        "eligible_denominator": 45,
    }


def test_current_markers_alone_fail_closed_on_missing_semantic_producer():
    with pytest.raises(GATE.PreLongGateRefused, match="MISSING_PRODUCER"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=None,
        )


def test_gate_rejects_four_or_duplicate_economy_updates():
    lines = _log().splitlines()
    missing = "\n".join(
        line
        for line in lines
        if not (
            line.startswith(GATE.ECONOMY_PREFIX)
            and '"ppo_update":4' in line
        )
    )
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_economy_updates(missing)

    duplicated = _log() + GATE.ECONOMY_PREFIX + json.dumps(_economy(4)) + "\n"
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_economy_updates(duplicated)


@pytest.mark.parametrize(
    "marker_kind,field,value",
    (
        ("economy", "schema_version", True),
        ("economy", "ppo_update", False),
        ("groups", "schema_version", True),
        ("groups", "ppo_update", False),
        ("semantic", "schema_version", True),
        ("semantic", "ppo_update", False),
    ),
)
def test_gate_rejects_boolean_update_and_schema_version(
    marker_kind, field, value
):
    if marker_kind == "semantic":
        rows = _semantics()
        rows[0][field] = value
        with pytest.raises(GATE.PreLongGateRefused, match="contiguous"):
            GATE.validate_semantic_updates(rows)
        return

    lines = []
    for update in range(5):
        row = _economy(update) if marker_kind == "economy" else _groups(update)
        if update == 0:
            row[field] = value
        prefix = GATE.ECONOMY_PREFIX if marker_kind == "economy" else GATE.GROUP_PREFIX
        lines.append(prefix + json.dumps(row, separators=(",", ":")))
    validator = (
        GATE.validate_economy_updates
        if marker_kind == "economy"
        else GATE.validate_group_income_updates
    )
    with pytest.raises(GATE.PreLongGateRefused, match="contiguous"):
        validator("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    "section,key,value,match",
    (
        ("ppo", "approx_kl", float("nan"), "not finite JSON"),
        ("ppo", "clip_fraction", 1.1, "clip_fraction"),
        ("gradient", "pre_clip_total_grad_norm", float("inf"), "not finite JSON"),
        ("policy", "policy_std_min", 0.0, "policy std"),
    ),
)
def test_gate_rejects_nonfinite_or_invalid_optimizer_health(
    section, key, value, match
):
    rows = []
    for update in range(5):
        economy = _economy(update)
        if update == 2:
            economy[section][key] = value
        rows.append(
            GATE.ECONOMY_PREFIX
            + json.dumps(economy, allow_nan=True, separators=(",", ":"))
        )
    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_economy_updates("\n".join(rows))


# --------------------------------------------------------------------------- #
# 常数奖励项:该拦的要拦,误拦的不许拦
# --------------------------------------------------------------------------- #
# 背景:s15r1 的 C0/C1 两格、五个 update,``action_rate_clamped`` 全部是逐位相同的
# ``-3538.945068``(``raw_sum = 884736 = 98304 x 9``,每个样本恒等于 ``value_clamp=9.0``)。
# 旧版这道门只查"收入是有限数 + 分母是全 rollout",对"整窗没动过"零意见。
#
# 2026-08-07:豁免不再是一段自述,而是一条要拿收据核对的断言 —— 申报必须指名收据里的
# 哪个量、朝哪个方向、越过哪个阈值才解冻,门再看那个量在窗内是不是真的朝阈值走。
# 现役 ``action_rate_clamped`` 那条申报正是被这条新规矩证伪的(std 在涨),所以
# ``DECLARED_CONSTANT_REWARD_TERMS`` 现在是空表:没有任何项拿着豁免。


def _falls_to_declaration(**overrides):
    """一条格式合法的申报:std 掉到 0.381 就解冻。"""

    declaration = {
        "mechanism": "value_clamp is saturated at the fresh policy's action scale",
        "ends_when": "policy_std_max falls to 0.381",
        "ends_when_metric": "policy_std_max",
        "ends_when_threshold": 0.381,
        "ends_when_direction": "falls_to",
        "carries_no_learning_signal_while_constant": True,
    }
    declaration.update(overrides)
    return declaration


def _economy_log(per_update_terms, per_update_policy=None):
    lines = []
    for update, terms in enumerate(per_update_terms):
        policy = None if per_update_policy is None else per_update_policy[update]
        lines.append(
            GATE.ECONOMY_PREFIX
            + json.dumps(
                _economy(update, terms=terms, policy=policy),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + "\n"


def _std_series(values):
    return [
        {
            "policy_std_min": value - 0.002,
            "policy_std_mean": value - 0.001,
            "policy_std_max": value,
        }
        for value in values
    ]


def test_undeclared_constant_nonzero_reward_term_is_refused():
    """该拦的:一个没申报过的项整窗吐同一个非零数 -> 拒收。"""

    log = _economy_log(
        [{"motion": 1.0 + 0.25 * update, "mystery_cost": -7.5} for update in range(5)]
    )
    with pytest.raises(GATE.PreLongGateRefused, match="mystery_cost"):
        GATE.validate_economy_updates(log)


def test_no_reward_term_currently_holds_a_constant_exemption():
    """现役表必须是空的 —— 而且**结案方式是修机制,不是发豁免**。

    ``action_rate_clamped`` 当初写的解冻条件是 ``policy_std_max`` 掉到 ``0.381``;
    s15r1 两格实测 ``1.00198 -> 1.00729`` / ``1.00191 -> 1.00661``,五个 update 一路在涨。
    唯一已知能打球的 ``build_1`` 收敛后 ``||Δa||²`` 还有 ``10.8~12.05``,仍在
    ``value_clamp=9.0`` 之上 —— 这个封顶全程焊死,不是"暂时饱和"。

    2026-08-08 结案:该项退役(weight 0),一阶平滑换回上游无封顶的 ``action_rate_l2``
    −0.1。所以本表仍然是空的:**没有任何项拿着豁免,这道门一行没改。**
    """

    assert GATE.DECLARED_CONSTANT_REWARD_TERMS == {}


def test_the_s15r1_frozen_shape_is_still_refused_verbatim():
    """通用护栏仍然有效:把 s15r1 那条逐位常数原样喂回去,门必须照样拒收。

    这是 (d) 那条"确认它仍然有效"的可执行版本 —— 数值逐位取自 s15r1 的 C0/C1
    (``-3538.945068`` = ``98304 x 9.0 x 0.2 x 0.02``),项名用旧名字,因为万一有人
    把封顶版复活,门要以完全相同的方式拦住他。
    """

    log = _economy_log(
        [
            {"motion": 1.0 + 0.25 * update, "action_rate_clamped": -3538.945068}
            for update in range(5)
        ]
    )
    with pytest.raises(GATE.PreLongGateRefused, match="action_rate_clamped"):
        GATE.validate_economy_updates(log)


def test_declared_constant_reward_term_passes_and_self_reports(monkeypatch):
    """误拦的不许拦:申报机制**且解冻条件确实在逼近**的饱和项照常放行,并进收据。"""

    monkeypatch.setitem(
        GATE.DECLARED_CONSTANT_REWARD_TERMS, "saturated_term", _falls_to_declaration()
    )
    log = _economy_log(
        [{"motion": 1.0 + 0.25 * update, "saturated_term": -3538.945068, "task": 0.0}
         for update in range(5)],
        _std_series([1.00, 0.92, 0.81, 0.66, 0.49]),
    )
    signal = GATE.validate_economy_updates(log)["reward_term_signal"]
    frozen = {row["term"]: row for row in signal["frozen_nonzero_terms"]}
    assert set(frozen) == {"saturated_term"}
    assert frozen["saturated_term"]["weighted_dt_sum_every_update"] == -3538.945068
    assert "value_clamp" in frozen["saturated_term"]["declared_mechanism"]
    audit = frozen["saturated_term"]["declared_end_condition_audit"]
    assert audit["metric"] == "policy_std_max"
    assert audit["first_observed"] == pytest.approx(1.00)
    assert audit["last_observed"] == pytest.approx(0.49)
    assert signal["always_zero_terms"] == ["task"]
    assert signal["varying_term_count"] == 1


def test_declared_constant_is_refused_when_its_exit_is_receding(monkeypatch):
    """该拦的:解冻条件在**后退**的申报 = 永久死项,豁免作废。

    这就是 s15r1 的实测形状,``policy_std_max`` 逐 update 逐位取自那两格的 C0 序列。
    """

    monkeypatch.setitem(
        GATE.DECLARED_CONSTANT_REWARD_TERMS, "action_rate_clamped", _falls_to_declaration()
    )
    log = _economy_log(
        [{"motion": 1.0 + 0.25 * update, "action_rate_clamped": -3538.945068}
         for update in range(5)],
        _std_series(
            [1.0019794702529907, 1.0036075115203857, 1.0048182010650635,
             1.0059711933135986, 1.0072904825210571]
        ),
    )
    with pytest.raises(GATE.PreLongGateRefused, match="away from the threshold"):
        GATE.validate_economy_updates(log)


def test_declared_constant_is_refused_when_its_exit_already_happened(monkeypatch):
    """该拦的:解冻条件**已经满足**而项还没解冻 -> 申报被自己的收据打脸。"""

    monkeypatch.setitem(
        GATE.DECLARED_CONSTANT_REWARD_TERMS, "action_rate_clamped", _falls_to_declaration()
    )
    log = _economy_log(
        [{"motion": 1.0 + 0.25 * update, "action_rate_clamped": -3538.945068}
         for update in range(5)],
        _std_series([1.00, 0.90, 0.70, 0.50, 0.30]),
    )
    with pytest.raises(GATE.PreLongGateRefused, match="already 0.3"):
        GATE.validate_economy_updates(log)


def test_declaration_must_name_a_metric_the_receipt_actually_carries(monkeypatch):
    """该拦的:结束条件指向收据里没有的量 = 回到无人核对的自述。"""

    for bad in (
        {"ends_when_metric": "mean_episode_length"},
        {"ends_when_metric": None},
        {"ends_when_direction": "eventually"},
        {"ends_when_threshold": "0.381"},
        {"ends_when_threshold": float("inf")},
        {"ends_when_threshold": True},
    ):
        monkeypatch.setitem(
            GATE.DECLARED_CONSTANT_REWARD_TERMS,
            "action_rate_clamped",
            _falls_to_declaration(**bad),
        )
        with pytest.raises(GATE.PreLongGateRefused):
            GATE.validate_economy_updates(_log())


def test_always_zero_reward_term_is_reported_but_never_blocking():
    """恒零不是拒收理由:一次一球未碰的冒烟里 target/outcome 层本来就该是零。"""

    log = _economy_log(
        [
            {"motion": 1.0 + 0.25 * update, "virtual_landing": 0.0, "racket_progress": 0.0}
            for update in range(5)
        ]
    )
    signal = GATE.validate_economy_updates(log)["reward_term_signal"]
    assert signal["always_zero_terms"] == ["racket_progress", "virtual_landing"]
    assert signal["always_zero_is_blocking"] is False


def test_one_ulp_of_movement_is_not_frozen():
    """粗一档就过不了:改用容差比较的变体会把只差 1 ulp 的项误判成常数。"""

    # math.nextafter 是 py3.9+;host pytest 仍是 py3.8,所以用 struct 手工挪一个 ulp。
    import struct

    base = -3538.945068
    bits = struct.unpack("<Q", struct.pack("<d", base))[0]
    moved = struct.unpack("<d", struct.pack("<Q", bits + 1))[0]
    assert moved != base and abs(moved - base) < 1.0e-9
    log = _economy_log(
        [
            {
                "motion": 1.0 + 0.25 * update,
                "undeclared_but_moving": base if update else moved,
            }
            for update in range(5)
        ]
    )
    signal = GATE.validate_economy_updates(log)["reward_term_signal"]
    assert signal["frozen_nonzero_terms"] == []
    assert "undeclared_but_moving" not in signal["always_zero_terms"]


def test_a_term_that_only_wakes_up_on_the_last_update_is_not_frozen():
    """粗一档就过不了:只对比前两个 update 的变体会把这一项误判成常数并拒收。"""

    log = _economy_log(
        [
            {
                "motion": 1.0 + 0.25 * update,
                "late_waker": 0.0 if update < 4 else -5.5,
            }
            for update in range(5)
        ]
    )
    signal = GATE.validate_economy_updates(log)["reward_term_signal"]
    assert signal["frozen_nonzero_terms"] == []
    assert signal["always_zero_terms"] == []


def test_reward_term_set_must_not_change_between_updates():
    per_update = [{"motion": 1.0 + 0.25 * update} for update in range(5)]
    per_update[3]["surprise"] = -1.0
    with pytest.raises(GATE.PreLongGateRefused, match="term set differs"):
        GATE.validate_economy_updates(_economy_log(per_update))


def test_blanked_constant_declaration_fails_closed(monkeypatch):
    """静默把申报清空 = 把白名单变成免检,必须炸。"""

    monkeypatch.setitem(
        GATE.DECLARED_CONSTANT_REWARD_TERMS,
        "action_rate_clamped",
        _falls_to_declaration(mechanism="", ends_when=""),
    )
    with pytest.raises(GATE.PreLongGateRefused, match="empty mechanism"):
        GATE.validate_economy_updates(_log())


def test_income_inversion_is_reported_not_blocked():
    """记录:安全层被排除在层级会计之外,收据里至少要有"最大单项成本 vs 全部正收入"。"""

    inversion = GATE.validate_economy_updates(_log())["income_inversion"]
    assert [row["ppo_update"] for row in inversion] == [0, 1, 2, 3, 4]
    assert inversion[0]["largest_cost_term"] == "action_rate_l2"
    assert inversion[0]["positive_income_sum"] == pytest.approx(1.0)
    assert inversion[0]["largest_cost_over_positive_income"] == pytest.approx(3538.945068)
    assert inversion[0]["net_weighted_dt_sum"] == pytest.approx(-3537.945068)
    # 夹具里这一项逐 update +1.5,所以最后一格的成本比第一格轻 6.0 —— 一个**活着**的
    # 罚项本来就该这样动;冻结形状归上面那几条专门的测试。
    assert inversion[4]["largest_cost_weighted_dt_sum"] == pytest.approx(-3532.945068)


def test_task_invalid_reward_or_denominator_must_be_zero():
    rows = _semantics()
    rows[3]["task_invalid"]["task_reward_eligible_denominator"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="leaked task reward or eligibility"):
        GATE.validate_semantic_updates(rows)


def test_contact_numerator_must_not_exceed_closed_swing_denominator():
    with pytest.raises(GATE.PreLongGateRefused, match="contacts <= eligible closed"):
        GATE.validate_semantic_updates(_semantics(contacts=10, closed=9))


def test_exact_strike_timing_and_closed_swing_are_cross_update_denominators():
    rows = _semantics()
    rows[0]["strike_timing"]["exact_strike_tick_denominator"] = 9
    rows[0]["reward_groups"][2]["eligible_denominator"] = 9
    rows[0]["hit"]["eligible_closed_swing_count"] = 0
    rows[1]["strike_timing"]["exact_strike_tick_denominator"] = 0
    rows[1]["reward_groups"][2]["eligible_denominator"] = 0
    rows[1]["hit"]["eligible_closed_swing_count"] = 9
    accepted = GATE.validate_semantic_updates(rows)
    assert accepted["updates"][0]["hit"] == "0/0"
    assert accepted["updates"][1]["hit"] == "0/9"


def test_strike_group_may_finite_filter_raw_exact_strike_ticks():
    rows = _semantics()
    rows[0]["strike_timing"]["exact_strike_tick_denominator"] = 9
    rows[0]["reward_groups"][2]["eligible_denominator"] = 8

    accepted = GATE.validate_semantic_updates(rows)

    assert accepted["updates"][0]["exact_strike_tick_denominator"] == 9


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda row: row["reward_groups"][2].__setitem__(
                "eligible_denominator", 10
            ),
            "strike-group denominator",
        ),
        (
            lambda row: row["reward_groups"][4].__setitem__(
                "eligible_denominator", 1
            ),
            "outcome-group denominator",
        ),
        (
            lambda row: row["window"].__setitem__("num_envs", 4095),
            "fixed rollout window",
        ),
        (
            lambda row: row["reward_groups"][4].__setitem__(
                "weighted_sum", 0.5
            ),
            "zero true eligibility",
        ),
    ),
)
def test_semantic_window_and_denominator_conservation_are_fail_closed(
    mutation, match
):
    rows = _semantics()
    mutation(rows[2])
    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_semantic_updates(rows)


# --------------------------------------------------------------------------
# 揭示->回放桥(``reveal_to_playback_bridge``)。2026-08-07 之前这块记录**没有任何
# 消费方**:生产方每个 update 都在写,而门逐字段 ``row.get(...)`` 又不要求键集合精确,
# 所以带桥的行被原样收下、一个字段都没被看过。下面每条都刻意构造成
# "粗一个档次的检查会放行" —— 注释里写清粗版长什么样。
# --------------------------------------------------------------------------


def _bridge(row):
    return row["reveal_to_playback_bridge"]


def test_a_row_carrying_no_reveal_bridge_no_longer_passes_silently():
    """接线前的现状本身:v3 行把桥写成 ``null`` 也能过。现在必须拒。"""

    rows = _semantics()
    for row in rows:
        row["reveal_to_playback_bridge"] = None

    with pytest.raises(GATE.PreLongGateRefused, match="reveal bridge fields differ"):
        GATE.validate_semantic_updates(rows)


def test_bridge_status_must_be_active_fail_closed():
    """粗版:只要求 ``status`` 是个非空字符串 —— 那样 ``not_configured`` 会被放行。"""

    rows = _semantics()
    for row in rows:
        _bridge(row)["status"] = "not_configured"

    assert all(
        isinstance(_bridge(row)["status"], str) and _bridge(row)["status"]
        for row in rows
    )
    with pytest.raises(GATE.PreLongGateRefused, match="not active fail-closed"):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize("field", sorted(BRIDGE.AUTHORITY_SHA256))
def test_bridge_authority_sha_must_be_lowercase_hex(field):
    """粗版:只量长度 64 —— 那样把一位改成大写会被放行,而跨 update 的相等比较就废了。"""

    rows = _semantics()
    for row in rows:
        authority = _bridge(row)["authority"]
        authority[field] = authority[field][:-1] + authority[field][-1].upper()

    assert all(_bridge(row)["authority"][field] != BRIDGE.AUTHORITY_SHA256[field] for row in rows)
    assert all(len(_bridge(row)["authority"][field]) == 64 for row in rows)
    with pytest.raises(GATE.PreLongGateRefused, match="must be lowercase SHA-256"):
        GATE.validate_semantic_updates(rows)


def test_bridge_cohort_may_conserve_and_still_miss_the_reveal_total():
    """粗版:只逐档看守恒 —— 那样"某档少一次揭示、同时少一次截断"会被放行。"""

    rows = _semantics()
    for row in rows:
        cohort = _bridge(row)["lifetime_conservation"]["wait_cohorts"][0]
        cohort["reveal_count"] -= 1
        cohort["censored_count"] -= 1

    for row in rows:
        for cohort in _bridge(row)["lifetime_conservation"]["wait_cohorts"]:
            assert cohort["reveal_count"] == (
                cohort["playback_start_count"]
                + cohort["terminal_before_start_count"]
                + cohort["censored_count"]
            )
    with pytest.raises(
        GATE.PreLongGateRefused, match="bridge timing/reveal counts differ"
    ):
        GATE.validate_semantic_updates(rows)


def test_bridge_authority_may_not_drift_after_the_first_update():
    """粗版:只校验第一个 update 的 authority —— 那样第 3 个 update 换合同会被放行。"""

    rows = _semantics()
    _bridge(rows[2])["authority"]["question_sha256"] = "6f" * 32

    assert _bridge(rows[0])["authority"]["question_sha256"] == (
        BRIDGE.AUTHORITY_SHA256["question_sha256"]
    )
    assert all(
        len(_bridge(row)["authority"][name]) == 64
        and _bridge(row)["authority"][name].islower()
        for row in rows
        for name in BRIDGE.AUTHORITY_SHA256
    )
    with pytest.raises(
        GATE.PreLongGateRefused, match="authority drifted across updates"
    ):
        GATE.validate_semantic_updates(rows)


def test_bridge_playback_start_count_may_not_regress_between_updates():
    """粗版:只看单 update 快照 —— 那样"某档开始回放数比上一轮少"会被放行。"""

    rows = _semantics()
    cohort = _bridge(rows[3])["lifetime_conservation"]["wait_cohorts"][0]
    previous = _bridge(rows[2])["lifetime_conservation"]["wait_cohorts"][0]
    regressed = previous["playback_start_count"] - 1
    cohort["censored_count"] += cohort["playback_start_count"] - regressed
    cohort["playback_start_count"] = regressed

    assert cohort["reveal_count"] == (
        cohort["playback_start_count"]
        + cohort["terminal_before_start_count"]
        + cohort["censored_count"]
    )
    assert min(
        cohort["playback_start_count"],
        cohort["terminal_before_start_count"],
        cohort["censored_count"],
    ) >= 0
    with pytest.raises(GATE.PreLongGateRefused, match="lifetime regressed"):
        GATE.validate_semantic_updates(rows)


def test_accepted_receipt_self_reports_the_bridge_it_consumed():
    """记录与阻断同一批:门通过时,收据必须自陈它读过这块桥,而不是只留一个结论位。"""

    bridge = GATE.validate_semantic_updates(_semantics())["aggregate"][
        "reveal_to_playback_bridge"
    ]

    assert bridge["updates_consumed"] == 5
    assert bridge["authority"]["wait_cohort_ticks"] == list(GATE.BRIDGE_WAIT_COHORTS)
    assert bridge["authority"]["policy_dt_s"] == 0.02
    assert bridge["cumulative_reveal_count"] == BRIDGE.total_reveal_count(4)
    assert bridge["newly_revealed_count"] == BRIDGE.total_reveal_count(4)
    assert [row["wait_ticks"] for row in bridge["final_wait_cohort_lifetime"]] == list(
        GATE.BRIDGE_WAIT_COHORTS
    )
    assert bridge["final_wait_cohort_lifetime"][0]["reveal"] == (
        BRIDGE.cohort_counts(4, 0)["reveal_count"]
    )
    assert len(bridge["per_update"]) == 5


def test_semantic_markers_require_exactly_five_contiguous_updates():
    rows = _semantics()
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_semantic_updates(rows[:-1])
    duplicate = list(rows) + [dict(rows[-1])]
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_semantic_updates(duplicate)


def test_task_invalid_exercised_over_five_updates_not_every_update():
    rows = _semantics()
    assert rows[1]["task_invalid"]["observed_sample_count"] == 0
    accepted = GATE.validate_semantic_updates(rows)
    assert accepted["aggregate"]["task_invalid_observed_sample_count"] == 7
    for row in rows:
        row["task_invalid"]["observed_sample_count"] = 0
    with pytest.raises(GATE.PreLongGateRefused, match="did not exercise task_valid=0"):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize("group", ("balance", "mimic"))
def test_every_update_requires_full_balance_and_mimic_denominator(group):
    rows = _semantics()
    group_index = 0 if group == "balance" else 1
    rows[2]["reward_groups"][group_index]["eligible_denominator"] -= 1

    with pytest.raises(
        GATE.PreLongGateRefused,
        match=rf"{group} denominator must equal 98304",
    ):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize(
    "profile,group_index,match",
    (
        (GATE.PROFILE_A211, 3, "A211 five-update target"),
        (GATE.PROFILE_C211, 2, "C211 five-update strike"),
    ),
)
def test_profile_specific_learnability_signal_must_be_positive(
    profile, group_index, match
):
    rows = _semantics(profile=profile)
    for row in rows:
        row["reward_groups"][group_index]["weighted_sum"] = 0.0

    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize(
    "profile,group_index,match",
    (
        (GATE.PROFILE_A211, 3, "A211 five-update target"),
        (GATE.PROFILE_C211, 2, "C211 five-update strike"),
    ),
)
def test_profile_specific_learnability_denominator_must_be_positive(
    profile, group_index, match
):
    rows = _semantics(profile=profile)
    for row in rows:
        row["reward_groups"][group_index]["weighted_sum"] = 0.0
        row["reward_groups"][group_index]["eligible_denominator"] = 0

    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_semantic_updates(rows)


def test_all_zero_learning_signal_is_rejected():
    rows = _semantics()
    for row in rows:
        for group in row["reward_groups"]:
            group["weighted_sum"] = 0.0

    with pytest.raises(
        GATE.PreLongGateRefused,
        match="aggregate balance income must be nonzero",
    ):
        GATE.validate_semantic_updates(rows)


@pytest.mark.parametrize(
    "group_index,match",
    (
        (0, "aggregate balance income must be nonzero"),
        (1, "aggregate mimic income must be positive"),
    ),
)
def test_aggregate_balance_and_mimic_income_health_is_required(group_index, match):
    rows = _semantics()
    for row in rows:
        row["reward_groups"][group_index]["weighted_sum"] = 0.0

    with pytest.raises(GATE.PreLongGateRefused, match=match):
        GATE.validate_semantic_updates(rows)


def test_unknown_attribution_and_terminal_safety_are_zero_tolerance():
    rows = _semantics()
    rows[0]["unknown_attribution_count"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="unknown attribution is nonzero"):
        GATE.validate_semantic_updates(rows)

    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"]["joint_qdes_forbidden_terminal_count"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="implementation counters"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )


def test_fall_and_too_low_are_reported_but_not_zero_tolerance_in_finite_gate():
    checkpoint = _checkpoint_acceptance()
    safety = checkpoint["safety_counters"]
    safety["base_fell_tilt_terminal_count"] = 3
    safety["base_too_low_terminal_count"] = 2
    safety["physical_fall_by_reason_phase"] = {
        "base_fell_tilt": {
            "hidden_wait": 1,
            "revealed_pre_strike": 2,
            "post_strike": 0,
        },
        "base_too_low": {
            "hidden_wait": 0,
            "revealed_pre_strike": 1,
            "post_strike": 1,
        },
    }

    accepted = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=checkpoint,
        semantic_updates=_semantics(),
    )

    balance = accepted["safety"]["balance_termination_counts"]
    assert balance["by_reason"] == {"base_fell_tilt": 3, "base_too_low": 2}
    assert "unvalidated_numeric_cutoff" in accepted["safety"][
        "finite_balance_termination_policy"
    ]
    behavior = accepted["survival_denominators"]["behavioral_terminations"]
    assert behavior["base_fell_tilt"]["phase_exposure_denominators"] == {
        "hidden_wait": 60,
        "revealed_pre_strike": 50,
        "post_strike": 45,
    }
    assert behavior["base_fell_tilt"]["phase_rates"] == {
        "hidden_wait": 1 / 60,
        "revealed_pre_strike": 2 / 50,
        "post_strike": 0.0,
    }
    assert behavior["base_fell_tilt"]["acceptance_threshold"] is None


# 2026-08-07 Franco 裁定三:两条实际-q 硬边计数器已从这份 zero-tolerance 名单里移出,
# 改由上面的 test_actual_q_hard_edge_is_measured_reported_and_never_blocking 覆盖。
# 这里直接读生产常量,免得名单在源码里改了、测试还抄着旧的六条。
@pytest.mark.parametrize("counter", GATE.STRICT_ZERO_SAFETY_COUNTERS)
def test_every_strict_safety_counter_remains_zero_tolerance(counter):
    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"][counter] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="implementation counters"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )


def test_strict_zero_set_is_exactly_the_four_implementation_counters():
    """名单本身是被裁定过的对象,所以它必须被逐字钉住,不能默默增删。"""

    assert GATE.STRICT_ZERO_SAFETY_COUNTERS == (
        "joint_qdes_forbidden_terminal_count",
        "joint_actual_forbidden_terminal_count",
        "strict_hard_termination_count",
        "nonfinite_count",
    )
    assert GATE.REPORTED_HARD_EDGE_COUNTERS == (
        "actual_hard_edge_event_count",
        "actual_hard_terminal_count",
    )
    assert not set(GATE.STRICT_ZERO_SAFETY_COUNTERS) & set(
        GATE.REPORTED_HARD_EDGE_COUNTERS
    )


def test_reason_by_phase_and_reveal_denominators_fail_closed():
    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"]["base_fell_tilt_terminal_count"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="do not conserve"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )


def test_robot_hit_table_is_behavioral_phase_evidence_not_finite_strict_zero():
    checkpoint = _checkpoint_acceptance()
    safety = checkpoint["safety_counters"]
    safety["table_contact_count"] = 3
    safety["table_contact_by_phase"] = {
        "hidden_wait": 1,
        "revealed_pre_strike": 2,
        "post_strike": 0,
    }

    accepted = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=checkpoint,
        semantic_updates=_semantics(),
    )

    table = accepted["survival_denominators"]["robot_hit_table"]
    assert table["total_count"] == 3
    assert table["phase_exposure_denominators"] == {
        "hidden_wait": 60,
        "revealed_pre_strike": 50,
        "post_strike": 45,
    }
    assert table["acceptance_threshold"] is None


def test_robot_hit_table_phase_counts_must_conserve():
    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"]["table_contact_count"] = 1
    with pytest.raises(GATE.PreLongGateRefused, match="robot_hit_table.*conserve"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )

    checkpoint = _checkpoint_acceptance()
    checkpoint["safety_counters"]["task_reveal_reached_by_update"][2] = 0
    checkpoint["safety_counters"]["task_reveal_reached_count"] = 40
    with pytest.raises(GATE.PreLongGateRefused, match="every finite update"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=checkpoint,
            semantic_updates=_semantics(),
        )


@pytest.mark.parametrize("milestone", ("exact", "closed"))
def test_each_update_requires_nominal_strike_and_closed_swing_survival(milestone):
    rows = _semantics()
    if milestone == "exact":
        rows[2]["strike_timing"]["exact_strike_tick_denominator"] = 0
        rows[2]["reward_groups"][2]["eligible_denominator"] = 0
    else:
        rows[2]["hit"]["eligible_closed_swing_count"] = 0
    with pytest.raises(GATE.PreLongGateRefused, match="finite survival update 2"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=rows,
        )


def test_each_update_requires_task_active_samples():
    rows = _semantics()
    rows[2]["task_invalid"]["observed_sample_count"] = 4096 * 24
    with pytest.raises(GATE.PreLongGateRefused, match="no TASK_ACTIVE samples"):
        GATE.validate_prelong_gate(
            log_text=_log(),
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=rows,
        )


def test_checkpoint_audit_requires_all_finite_nonempty_state_groups():
    acceptance = _checkpoint_acceptance()
    del acceptance["checkpoint"]["tensor_groups"]["optimizer"]
    with pytest.raises(GATE.PreLongGateRefused, match="tensor-group coverage"):
        GATE.validate_checkpoint_audit(acceptance["checkpoint"])


@pytest.mark.parametrize(
    "field",
    ("filename_iteration", "embedded_iteration"),
)
@pytest.mark.parametrize("wrong", (5, 3))
def test_checkpoint_audit_binds_the_last_written_iteration_not_the_budget(
    field, wrong
):
    """跑满 5 个 update 的末位是 4;5(旧的差一格手抄)和 3(少跑一格)都要被拒。"""

    acceptance = _checkpoint_acceptance()
    assert acceptance["checkpoint"][field] == 4
    acceptance["checkpoint"][field] = wrong
    with pytest.raises(GATE.PreLongGateRefused, match="model_4.pt, embedded iter=4"):
        GATE.validate_checkpoint_audit(acceptance["checkpoint"])


def test_cli_reports_structured_blocker_when_semantic_input_is_absent(
    tmp_path, capsys
):
    log = tmp_path / "run.log"
    log.write_text(_log(), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(_checkpoint_acceptance()), encoding="utf-8")
    assert GATE.main(
        ["--run-log", str(log), "--checkpoint-acceptance", str(checkpoint)]
    ) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "BLOCKED"
    assert output["diagnostic_unauthorized"] is True
    assert output["reason"].startswith("MISSING_PRODUCER:")


# ---------------------------------------------------------------------------
# reward activation ledger 的适用范围:诊断跑没有这本账,正式跑必须有
# ---------------------------------------------------------------------------
# 变异证据分两类,必须同时成立:
#   * 正式跑缺那 5 行 -> 仍然拒收(等强,重定范围没有把正式跑的门放松);
#   * 诊断跑没有这本账 -> 不再误拒(重定范围确实起作用);
# 外加一条:收据必须自陈它走的是哪条分支。


def test_regime_comes_from_the_run_receipt_not_from_a_caller_claim():
    assert (
        GATE.classify_reward_evidence_regime(_log())
        == GATE.REWARD_EVIDENCE_REGIME_FORMAL
    )
    assert (
        GATE.classify_reward_evidence_regime(_diagnostic_log())
        == GATE.REWARD_EVIDENCE_REGIME_DIAGNOSTIC
    )


def test_regime_cannot_be_established_without_a_joint_safety_receipt():
    stripped = "\n".join(
        line
        for line in _log().splitlines()
        if not line.startswith(GATE.JOINT_SAFETY_PREFIX)
    )
    with pytest.raises(GATE.PreLongGateRefused, match="cannot be established"):
        GATE.classify_reward_evidence_regime(stripped + "\n")


def test_a_run_may_not_declare_two_reward_evidence_regimes_at_once():
    hybrid = _diagnostic_log() + _joint_safety(0, formal=True) + "\n"
    with pytest.raises(
        GATE.PreLongGateRefused, match="exactly one reward-evidence"
    ):
        GATE.classify_reward_evidence_regime(hybrid)


@pytest.mark.parametrize(
    "prefix", GATE.REWARD_ACTIVATION_LEDGER_REQUIRED_PREFIXES
)
def test_formal_run_missing_the_five_markers_is_still_refused(prefix):
    """等强:重定范围之后,正式跑缺这一族任何一条仍然拒收。"""

    dropped = "\n".join(
        line for line in _log().splitlines() if not line.startswith(prefix)
    )
    with pytest.raises(
        GATE.PreLongGateRefused, match="lacks exactly 5 markers"
    ):
        GATE.reward_activation_evidence_scope(log_text=dropped + "\n")

    partial = "\n".join(
        line
        for line in _log().splitlines()
        if not (line.startswith(prefix) and '"ppo_update":4' in line)
    )
    with pytest.raises(
        GATE.PreLongGateRefused, match="lacks exactly 5 markers"
    ):
        GATE.reward_activation_evidence_scope(log_text=partial + "\n")


def test_formal_run_missing_reward_activation_evidence_still_fails_the_gate():
    dropped = "\n".join(
        line
        for line in _log().splitlines()
        if not line.startswith("HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=")
    )
    with pytest.raises(GATE.PreLongGateRefused, match="lacks exactly 5 markers"):
        GATE.validate_prelong_gate(
            log_text=dropped + "\n",
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
        )


def test_diagnostic_run_is_no_longer_refused_for_evidence_it_cannot_produce():
    """重定范围有效:诊断跑 0 行不再是拒收理由。"""

    scope = GATE.reward_activation_evidence_scope(log_text=_diagnostic_log())
    assert scope["regime"] == GATE.REWARD_EVIDENCE_REGIME_DIAGNOSTIC
    assert scope["applicable"] is False
    assert scope["required_rows_per_prefix"] == 0
    assert set(scope["observed_rows_per_prefix"]) == set(
        GATE.REWARD_ACTIVATION_LEDGER_PREFIXES
    )
    assert all(
        count == 0 for count in scope["observed_rows_per_prefix"].values()
    )
    assert (
        scope["strict_zero_counter_source"]
        == "exact_behavior_termination_reason_counters"
    )


@pytest.mark.parametrize("prefix", GATE.REWARD_ACTIVATION_LEDGER_PREFIXES)
def test_diagnostic_run_may_not_mint_formal_reward_evidence(prefix):
    """阻断的另一半:诊断跑一旦发出这一族,同样拒收 —— 不是静默跳过。"""

    polluted = _diagnostic_log() + prefix + json.dumps({"ppo_update": 0}) + "\n"
    with pytest.raises(
        GATE.PreLongGateRefused,
        match="diagnostic run emitted formal reward-activation evidence",
    ):
        GATE.reward_activation_evidence_scope(log_text=polluted)


def test_receipt_self_declares_which_branch_it_took():
    formal = GATE.validate_prelong_gate(
        log_text=_log(),
        checkpoint_acceptance=_checkpoint_acceptance(),
        semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
    )
    formal_scope = formal["reward_activation_evidence_scope"]
    assert formal_scope["kind"] == GATE.REWARD_EVIDENCE_SCOPE_KIND
    assert formal_scope["regime"] == GATE.REWARD_EVIDENCE_REGIME_FORMAL
    assert formal_scope["applicable"] is True
    assert formal_scope["required_rows_per_prefix"] == 5
    assert formal["reward_group_income"]["updates"]

    diagnostic = GATE.validate_prelong_gate(
        log_text=_diagnostic_log(),
        checkpoint_acceptance=_checkpoint_acceptance(),
        semantic_updates=_semantics(contacts=0, closed=9, flight_denominator=0),
    )
    diagnostic_scope = diagnostic["reward_activation_evidence_scope"]
    assert diagnostic["status"] == "PASS"
    assert diagnostic_scope["regime"] == GATE.REWARD_EVIDENCE_REGIME_DIAGNOSTIC
    assert diagnostic_scope["applicable"] is False
    # 「不适用」必须写在收据里,而不是让 group income 悄悄变成空。
    assert diagnostic["reward_group_income"]["applicable"] is False
    assert (
        diagnostic["reward_group_income"]["reward_activation_evidence_scope"]
        == diagnostic_scope
    )
    assert "reward activation ledger" in diagnostic_scope["reason"]


def test_diagnostic_branch_does_not_weaken_anything_else_in_the_gate():
    """重定范围只摘掉这一族;economy / semantics / 存档那几段照旧硬。"""

    without_economy = "\n".join(
        line
        for line in _diagnostic_log().splitlines()
        if not line.startswith(GATE.ECONOMY_PREFIX)
    )
    with pytest.raises(GATE.PreLongGateRefused, match="exactly 5"):
        GATE.validate_prelong_gate(
            log_text=without_economy + "\n",
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=_semantics(
                contacts=0, closed=9, flight_denominator=0
            ),
        )
    with pytest.raises(GATE.PreLongGateRefused, match="MISSING_PRODUCER"):
        GATE.validate_prelong_gate(
            log_text=_diagnostic_log(),
            checkpoint_acceptance=_checkpoint_acceptance(),
            semantic_updates=None,
        )
    broken = _checkpoint_acceptance()
    broken["checkpoint"]["embedded_iteration"] = 5
    with pytest.raises(GATE.PreLongGateRefused):
        GATE.validate_prelong_gate(
            log_text=_diagnostic_log(),
            checkpoint_acceptance=broken,
            semantic_updates=_semantics(
                contacts=0, closed=9, flight_denominator=0
            ),
        )


def test_required_and_observed_only_prefixes_are_disjoint_and_complete():
    """名单不能自己漂:必需集与只记录集不重叠,并集就是全族。"""

    required = set(GATE.REWARD_ACTIVATION_LEDGER_REQUIRED_PREFIXES)
    observed_only = set(GATE.REWARD_ACTIVATION_LEDGER_OBSERVED_PREFIXES)
    assert not required & observed_only
    assert required | observed_only == set(
        GATE.REWARD_ACTIVATION_LEDGER_PREFIXES
    )
    assert GATE.GROUP_PREFIX in required
