"""Wave-Q qbar: all-joint q_des position-limit barrier (Jiayi V14 idea, top-k removed).

Idea credit: Jiayi's V14 all-joint top-k qdes barrier.  Franco's 2026-07-21 ruling removes the
top-k: EVERY one of the 31 deploy-space targets pays as soon as it enters the margin band next
to its position limit, on every control step in every phase (dense — matching V14's
whole-episode barrier).  The -0.65 weight / margin_frac 0.08 starting point comes from V14.
Aggregation is SUM, not mean (Franco: mean 是稀释器,单关节违规被 ÷31;sum 让违规几个关节
就罚几份——这也是去 top-k 的正当性所在;满违规单关节 tail≈1 → 每步约 -0.65×dt)。
人话:目标角贴到限位边上就罚,几个关节贴就扣几份,全身 31 个关节一视同仁、全程有效,
不挑"最狠的几个"。

Pinned here:

* the exact barrier math ``sum(1-exp(-square(relu(margin_frac-d)/margin_frac)))`` with
  ``d = min(qdes-lo, hi-qdes)/(hi-lo)``, hand-computed at depth 0.5 / 1 / 2 (beyond-limit);
* full 31-joint coverage: violating any single joint moves the sum undiluted; the source has
  no top-k and no joint-subset path, and no phase/command gate (dense);
* fail-closed on missing/malformed limits, non-identity joint order, and bad margin_frac;
* the weight-independent probe + idempotent per-step ledger (above-margin joint counts, max
  intrusion depth) and the InferenceMode-safe consume/reset;
* train.py fail-loud override translation and the schema-3 contract block roundtrip.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_qdes_limit_barrier.py -q
"""

from __future__ import annotations

import inspect
import math
import types
from pathlib import Path

import pytest
import torch

from test_reward_flags_mdp import hope_rewards_mod
from test_reward_flags_overrides import _Term, _apply_legacy_v1, _make_env_cfg, train_mod
from test_training_contract_schema3 import TC, _qdot_hinge_schema3_contract


JOINTS = list(hope_rewards_mod._A3_RUNTIME_JOINT_ORDER)
MARGIN = 0.08
TAIL_1 = 1.0 - math.exp(-1.0)

_QBAR_PARAMS = {"action_name": "joint_pos", "margin_frac": 0.08}


# --------------------------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------------------------- #
def _barrier_env(n=2, *, limits=None, joint_names=None, joint_ids=slice(None)):
    """Env with all 31 targets at the center of symmetric [-1, 1] limits (span 2)."""

    names = list(JOINTS) if joint_names is None else list(joint_names)
    processed = torch.zeros(n, len(names))
    if limits is None:
        limits = torch.stack(
            (
                torch.full((n, len(names)), -1.0),
                torch.full((n, len(names)), 1.0),
            ),
            dim=-1,
        )
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            soft_joint_pos_limits=limits,
            # 站姿豁免(2026-07-25)需要设计站姿;默认放在限位正中(d=0.5),即全部关节
            # margin_eff == margin_frac,既有手算期望逐字节不变。
            default_joint_pos=torch.zeros(n, len(names)),
        )
    )
    action = types.SimpleNamespace(
        processed_actions=processed,
        _asset=asset,
        _joint_names=list(names),
        _joint_ids=joint_ids,
    )
    env = types.SimpleNamespace(
        common_step_counter=23,
        action_manager=types.SimpleNamespace(get_term=lambda name: action),
    )
    return env, action, asset.data


def _barrier_env_cfg():
    cfg = _make_env_cfg()
    cfg.rewards.qdes_limit_barrier = _Term(weight=0.0, params=dict(_QBAR_PARAMS))
    cfg.rewards.qdes_limit_barrier_probe = _Term(weight=0.0, params=dict(_QBAR_PARAMS))
    return cfg


def _apply_qbar(task, cfg=None):
    # 2026-07-25 默认翻转后本套件仍测 legacy 翻译行为:钉 v1 + 滤 v1 记账行,原断言原样成立
    # (默认路径 = v2 展开,由 test_reward_flags_overrides JOB1 区块专测)。
    cfg = cfg if cfg is not None else _barrier_env_cfg()
    applied = _apply_legacy_v1(cfg, task)
    return cfg, applied


def _runtime_facts():
    return {
        "joint_names": list(JOINTS),
        "articulation_joint_names": list(JOINTS),
        "qdes_joint_pos_limits": [[-1.0, 1.0] for _ in JOINTS],
    }


# --------------------------------------------------------------------------------------------- #
# barrier math (hand-computed)
# --------------------------------------------------------------------------------------------- #
def test_all_targets_at_center_pay_exact_zero():
    env, *_ = _barrier_env(3)
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert torch.equal(value, torch.zeros(3))


def test_target_exactly_at_the_limit_pays_the_unit_tail():
    env, action, _ = _barrier_env(2)
    action.processed_actions[0, 5] = 1.0  # d = 0 -> t = 1
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert value.tolist() == pytest.approx([TAIL_1, 0.0], abs=1e-6)


def test_stance_exempt_margin_zeroes_the_designed_stance():
    """站姿豁免(2026-07-25):默认站姿贴限的关节(实机对应双肩 roll),站姿零罚零梯度;
    从站姿再向限位靠近仍照罚(罚带 = 收窄后的 m_eff)。"""
    env, action, data = _barrier_env(2)
    # 仿双肩 roll 几何:默认站姿离下限只有 0.03 x 跨度(< margin 0.08)
    data.default_joint_pos[:, 5] = -0.94  # d_default = (-0.94-(-1))/2 = 0.03
    action.processed_actions[0, 5] = -0.94  # q_des == 站姿 -> 免费(旧一刀切数学要罚 0.328)
    action.processed_actions[1, 5] = -0.985  # 更贴限位:d=0.0075 < m_eff=0.025 -> 照罚
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert value[0].item() == pytest.approx(0.0, abs=1e-7)
    t = (0.025 - 0.0075) / 0.025
    assert value[1].item() == pytest.approx(1.0 - math.exp(-(t**2)), abs=1e-6)


def test_stance_on_the_limit_fails_loud_instead_of_silencing():
    # 站姿本身贴死软限位 = 建模错误:宁可炸也不静默把 barrier 豁免成零带宽
    env, _, data = _barrier_env(1)
    data.default_joint_pos[0, 3] = -0.999  # d_default=0.0005 -> m_eff < MARGIN_FLOOR
    with pytest.raises(RuntimeError, match="default-stance"):
        hope_rewards_mod.qdes_limit_barrier(env)


def test_missing_default_joint_pos_fails_loud():
    env, _, data = _barrier_env(1)
    del data.default_joint_pos
    with pytest.raises(RuntimeError, match="default_joint_pos"):
        hope_rewards_mod.qdes_limit_barrier(env)


def test_half_depth_intrusion_hand_computed():
    env, action, _ = _barrier_env(2)
    # d = (1 - 0.92) / 2 = 0.04 = margin/2 -> t = 0.5 -> 1 - exp(-0.25).
    action.processed_actions[0, 7] = 0.92
    expected = 1.0 - math.exp(-0.25)
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert value.tolist() == pytest.approx([expected, 0.0], abs=1e-6)


def test_lower_limit_side_is_charged_symmetrically():
    env, action, _ = _barrier_env(2)
    action.processed_actions[0, 11] = -0.92  # same 0.04 normalized distance, lower side
    expected = 1.0 - math.exp(-0.25)
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert value.tolist() == pytest.approx([expected, 0.0], abs=1e-6)


def test_beyond_limit_target_keeps_growing_but_stays_bounded():
    env, action, _ = _barrier_env(1)
    # Unclamped legacy arm: q_des past hi.  d = (1 - 1.16)/2 = -0.08 -> t = 2.
    action.processed_actions[0, 3] = 1.16
    expected = 1.0 - math.exp(-4.0)
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert value.tolist() == pytest.approx([expected], abs=1e-6)
    assert value[0].item() < 1.0 + 1e-6  # bounded per joint (sum bound = joints in violation)


def test_exactly_at_the_margin_edge_is_free():
    env, action, _ = _barrier_env(1)
    action.processed_actions[0, 0] = 1.0 - 2.0 * MARGIN  # d = margin exactly
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert torch.equal(value, torch.zeros(1))


def test_asymmetric_limits_normalize_by_the_joint_range():
    n = 1
    limits = torch.stack(
        (torch.full((n, 31), -0.5), torch.full((n, 31), 1.5)), dim=-1
    )
    env, action, _ = _barrier_env(n, limits=limits)
    action.processed_actions[:] = 0.5  # center of [-0.5, 1.5]
    action.processed_actions[0, 9] = 1.5 - 0.08  # d = 0.08/2 = margin/2 -> t = 0.5
    expected = 1.0 - math.exp(-0.25)
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert value.tolist() == pytest.approx([expected], abs=1e-6)


# --------------------------------------------------------------------------------------------- #
# full 31-joint coverage, no top-k, no dilution, dense
# --------------------------------------------------------------------------------------------- #
def test_every_single_one_of_the_31_joints_pays_its_full_undiluted_tail():
    for joint in range(31):
        env, action, _ = _barrier_env(1)
        action.processed_actions[0, joint] = 1.0
        value = hope_rewards_mod.qdes_limit_barrier(env)
        assert value.tolist() == pytest.approx([TAIL_1], abs=1e-6), joint


def test_violations_add_up_one_share_per_violating_joint():
    # Franco's sum rationale: K joints in violation pay K shares (mean would divide by 31).
    env, action, _ = _barrier_env(1)
    for joint in (2, 5, 19):
        action.processed_actions[0, joint] = 1.0  # t = 1 each
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert value.tolist() == pytest.approx([3.0 * TAIL_1], abs=1e-6)


def test_source_has_no_topk_and_no_joint_subset_path():
    source = inspect.getsource(hope_rewards_mod._qdes_limit_barrier_values)
    source += inspect.getsource(hope_rewards_mod.qdes_limit_barrier)
    source += inspect.getsource(hope_rewards_mod.qdes_limit_barrier_probe)
    for forbidden in ("topk", "top_k", "selected", "RECOVERY_JOINT_NAMES"):
        assert forbidden not in source, forbidden


def test_dense_no_phase_gate_and_no_command_dependency():
    # The fake env deliberately has NO command_manager: a phase-gated implementation would crash.
    env, action, _ = _barrier_env(1)
    assert not hasattr(env, "command_manager")
    action.processed_actions[0, 5] = 1.0
    value = hope_rewards_mod.qdes_limit_barrier(env)
    assert value[0].item() > 0.0
    source = inspect.getsource(hope_rewards_mod._qdes_limit_barrier_values)
    assert "command_manager" not in source
    assert "post_strike" not in source


def test_docstring_records_jiayi_v14_provenance_and_the_frozen_start_point():
    doc = hope_rewards_mod.qdes_limit_barrier.__doc__
    assert "V14" in doc and "top-k" in doc
    assert "0.08" in doc and "-0.65" in doc
    assert inspect.signature(
        hope_rewards_mod.qdes_limit_barrier
    ).parameters["margin_frac"].default == 0.08


# --------------------------------------------------------------------------------------------- #
# fail-closed
# --------------------------------------------------------------------------------------------- #
def test_missing_position_limits_fail_closed():
    env, _, data = _barrier_env(1)
    del data.soft_joint_pos_limits
    with pytest.raises(RuntimeError, match="soft joint position limits"):
        hope_rewards_mod.qdes_limit_barrier(env)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda limits: limits.__setitem__((0, 4, 0), 1.0),  # lo == hi
        lambda limits: limits.__setitem__((0, 4, 1), -2.0),  # hi < lo
        lambda limits: limits.__setitem__((0, 4, 0), float("nan")),
        lambda limits: limits.__setitem__((0, 4, 1), float("inf")),
    ],
)
def test_degenerate_or_nonfinite_limits_fail_closed(mutate):
    env, _, data = _barrier_env(1)
    mutate(data.soft_joint_pos_limits)
    with pytest.raises(RuntimeError, match="lo < hi"):
        hope_rewards_mod.qdes_limit_barrier(env)


def test_misshapen_limits_tensor_fails_closed():
    env, _, data = _barrier_env(2)
    data.soft_joint_pos_limits = torch.zeros(2, 31)  # missing the [lo, hi] axis
    with pytest.raises(RuntimeError, match=r"\[31,2\]"):
        hope_rewards_mod.qdes_limit_barrier(env)


def test_non_identity_or_short_joint_order_fails_closed():
    env, *_ = _barrier_env(1, joint_ids=[1, 0, *range(2, 31)])
    with pytest.raises(RuntimeError, match="identity 31-joint"):
        hope_rewards_mod.qdes_limit_barrier(env)

    env, action, _ = _barrier_env(1)
    action._joint_names = list(reversed(JOINTS))
    with pytest.raises(RuntimeError, match="identity 31-joint"):
        hope_rewards_mod.qdes_limit_barrier(env)

    short = JOINTS[:30]
    limits = torch.stack(
        (torch.full((1, 30), -1.0), torch.full((1, 30), 1.0)), dim=-1
    )
    env, *_ = _barrier_env(1, joint_names=short, limits=limits)
    with pytest.raises(RuntimeError, match="identity 31-joint"):
        hope_rewards_mod.qdes_limit_barrier(env)


@pytest.mark.parametrize("margin", [True, 0.0, 0.5, -0.1, float("nan"), float("inf")])
def test_invalid_margin_frac_fails_closed(margin):
    env, *_ = _barrier_env(1)
    with pytest.raises(ValueError, match=r"\(0, 0.5\)"):
        hope_rewards_mod.qdes_limit_barrier(env, margin_frac=margin)


def test_wrong_action_name_fails_closed():
    env, *_ = _barrier_env(1)
    with pytest.raises(ValueError, match="exactly 'joint_pos'"):
        hope_rewards_mod.qdes_limit_barrier(env, action_name="arm_pos")


# --------------------------------------------------------------------------------------------- #
# probe + shared idempotent ledger
# --------------------------------------------------------------------------------------------- #
def test_probe_is_exact_zero_and_shares_one_hand_checked_ledger():
    env, action, _ = _barrier_env(4)
    action.processed_actions[1, 5] = 1.0  # t = 1
    action.processed_actions[2, 7] = 0.92  # t = 0.5
    action.processed_actions[3, 3] = 1.16  # t = 2 (beyond limit)
    probe = hope_rewards_mod.qdes_limit_barrier_probe(env)
    reward = hope_rewards_mod.qdes_limit_barrier(env)
    assert torch.equal(probe, torch.zeros(4))
    assert reward.tolist() == pytest.approx(
        [
            0.0,
            TAIL_1,
            1.0 - math.exp(-0.25),
            1.0 - math.exp(-4.0),
        ],
        abs=1e-6,
    )

    counters = hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(env)
    assert counters["observed_sample_count"].item() == 4  # probe+reward booked once
    assert counters["above_margin_joint_count"].item() == 3
    assert counters["above_margin_sample_count"].item() == 3
    assert counters["max_intrusion_depth_frac"].item() == pytest.approx(2.0)
    assert counters["barrier_value_sum"].item() == pytest.approx(
        reward.sum().item(), abs=1e-6
    )
    assert counters["reward_enabled_sample_count"].item() == 4
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(
            env
        ).values()
    )


def test_probe_alone_never_books_reward_enabled_samples():
    env, action, _ = _barrier_env(3)
    action.processed_actions[0, 5] = 1.0
    hope_rewards_mod.qdes_limit_barrier_probe(env)
    counters = hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(env)
    assert counters["observed_sample_count"].item() == 3
    assert counters["above_margin_joint_count"].item() == 1
    assert counters["reward_enabled_sample_count"].item() == 0


def test_probe_and_reward_with_different_parameters_in_one_step_raise():
    env, *_ = _barrier_env(2)
    hope_rewards_mod.qdes_limit_barrier_probe(env)
    with pytest.raises(RuntimeError, match="different parameters"):
        hope_rewards_mod.qdes_limit_barrier(env, margin_frac=0.1)


def test_consume_resets_inference_mode_counters_without_crash():
    """Regression: counters born under InferenceMode must reset via torch.inference_mode()."""

    env, action, _ = _barrier_env(2)
    action.processed_actions[0, 5] = 1.0
    with torch.inference_mode():
        hope_rewards_mod.qdes_limit_barrier(env)
    first = hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(env)
    assert first["observed_sample_count"].item() == 2
    second = hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(env)
    assert all(value.item() == 0 for value in second.values())
    source = inspect.getsource(
        hope_rewards_mod.consume_qdes_limit_barrier_activation_counters
    )
    assert "torch.inference_mode()" in source


# --------------------------------------------------------------------------------------------- #
# env cfg declaration
# --------------------------------------------------------------------------------------------- #
def test_env_cfg_declares_default_off_term_and_probe_with_explicit_params():
    source = (
        Path(__file__).resolve().parents[1]
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    ).read_text()
    for name, func in (
        ("qdes_limit_barrier = RewTerm(", "func=mdp.qdes_limit_barrier,"),
        ("qdes_limit_barrier_probe = RewTerm(", "func=mdp.qdes_limit_barrier_probe,"),
    ):
        start = source.index(name)
        block = source[start : source.index(")", source.index("}", start))]
        assert "weight=0.0" in block
        assert func in block
        assert '"action_name": "joint_pos"' in block
        assert '"margin_frac": 0.08' in block


# --------------------------------------------------------------------------------------------- #
# train.py override translation
# --------------------------------------------------------------------------------------------- #
def test_v1_baseline_task_leaves_barrier_terms_untouched():
    cfg, applied = _apply_qbar({})
    assert cfg.rewards.qdes_limit_barrier.weight == 0.0
    assert cfg.rewards.qdes_limit_barrier_probe.weight == 0.0
    assert not any("qdes_limit_barrier" in marker for marker in applied)
    assert train_mod._qdes_limit_barrier_reward_contract(cfg, _runtime_facts()) is None


def test_explicit_zero_weight_control_raises_the_probe_and_binds_the_contract():
    cfg, applied = _apply_qbar({"rewards": {"qdes_limit_barrier_weight": 0.0}})
    assert cfg.rewards.qdes_limit_barrier.weight == 0.0
    assert cfg.rewards.qdes_limit_barrier_probe.weight == 1.0
    assert any("qdes_limit_barrier_probe" in marker for marker in applied)
    block = train_mod._qdes_limit_barrier_reward_contract(cfg, _runtime_facts())
    assert block["enabled"] is False and block["probe_enabled"] is True
    assert block["activation_ledger"] == "weight_independent_control_step_counters"


def test_weight_and_margin_overrides_apply_and_keep_probe_params_synced():
    cfg, applied = _apply_qbar(
        {
            "rewards": {
                "qdes_limit_barrier_weight": -0.65,
                "qdes_limit_barrier_margin_frac": 0.1,
            }
        }
    )
    term = cfg.rewards.qdes_limit_barrier
    probe = cfg.rewards.qdes_limit_barrier_probe
    assert term.weight == pytest.approx(-0.65)
    assert term.params["margin_frac"] == pytest.approx(0.1)
    assert term.params["action_name"] == "joint_pos"
    assert probe.weight == 1.0 and probe.params == term.params
    assert "rewards.qdes_limit_barrier.weight=-0.65" in applied
    assert "rewards.qdes_limit_barrier.params.margin_frac=0.1" in applied
    block = train_mod._qdes_limit_barrier_reward_contract(cfg, _runtime_facts())
    assert block["enabled"] is True
    assert block["weight"] == pytest.approx(-0.65)
    assert block["margin_frac"] == pytest.approx(0.1)
    assert block["joint_count"] == 31
    assert block["joint_order"] == "runtime_articulation_identity"
    assert block["position_limit_source"] == "articulation.data.soft_joint_pos_limits"
    assert block["gate"] == "dense_every_control_step"


def test_margin_without_explicit_weight_is_refused():
    with pytest.raises(train_mod._OverrideError, match="qdes_limit_barrier_weight"):
        _apply_qbar({"rewards": {"qdes_limit_barrier_margin_frac": 0.1}})


@pytest.mark.parametrize("weight", [0.1, float("nan"), float("inf"), True, "bad"])
def test_invalid_or_positive_barrier_weight_is_refused(weight):
    with pytest.raises(train_mod._OverrideError, match="finite and <= 0"):
        _apply_qbar({"rewards": {"qdes_limit_barrier_weight": weight}})


@pytest.mark.parametrize("margin", [0.0, 0.5, -0.1, float("nan"), True, "bad"])
def test_invalid_barrier_margin_override_is_refused(margin):
    with pytest.raises(train_mod._OverrideError, match=r"\(0, 0.5\)"):
        _apply_qbar(
            {
                "rewards": {
                    "qdes_limit_barrier_weight": -0.65,
                    "qdes_limit_barrier_margin_frac": margin,
                }
            }
        )


def test_invalid_margin_does_not_partially_mutate_the_barrier_terms():
    cfg = _barrier_env_cfg()
    term = cfg.rewards.qdes_limit_barrier
    probe = cfg.rewards.qdes_limit_barrier_probe
    before = (term.weight, dict(term.params), probe.weight, dict(probe.params))
    with pytest.raises(train_mod._OverrideError):
        _apply_qbar(
            {
                "rewards": {
                    "qdes_limit_barrier_weight": -0.65,
                    "qdes_limit_barrier_margin_frac": 0.5,
                }
            },
            cfg=cfg,
        )
    assert (term.weight, dict(term.params), probe.weight, dict(probe.params)) == before


def test_reward_keys_whitelist_contains_exactly_the_two_qbar_keys_once():
    keys = [key for key in train_mod._REWARD_KEYS if key.startswith("qdes_limit_barrier")]
    assert sorted(keys) == ["qdes_limit_barrier_margin_frac", "qdes_limit_barrier_weight"]
    with pytest.raises(train_mod._OverrideError, match="qdes_limit_barrier_wieght"):
        _apply_qbar({"rewards": {"qdes_limit_barrier_wieght": -0.65}})


def test_contract_builder_requires_probe_pairing_and_matching_params():
    cfg = _barrier_env_cfg()
    cfg.rewards.qdes_limit_barrier_probe = None
    with pytest.raises(RuntimeError, match="declared together"):
        train_mod._qdes_limit_barrier_reward_contract(cfg, _runtime_facts())

    cfg2, _ = _apply_qbar({"rewards": {"qdes_limit_barrier_weight": -0.65}})
    cfg2.rewards.qdes_limit_barrier_probe.params["margin_frac"] = 0.2
    with pytest.raises(RuntimeError, match="exactly match"):
        train_mod._qdes_limit_barrier_reward_contract(cfg2, _runtime_facts())

    cfg3 = _barrier_env_cfg()
    cfg3.rewards.qdes_limit_barrier.weight = -0.65
    cfg3.rewards.qdes_limit_barrier_probe.weight = 0.0
    with pytest.raises(RuntimeError, match="weight-independent probe"):
        train_mod._qdes_limit_barrier_reward_contract(cfg3, _runtime_facts())


def test_contract_builder_requires_identity_runtime_order_and_sane_limits():
    cfg, _ = _apply_qbar({"rewards": {"qdes_limit_barrier_weight": -0.65}})
    facts = _runtime_facts()
    facts["joint_names"] = list(reversed(JOINTS))
    with pytest.raises(RuntimeError, match="identity 31-joint"):
        train_mod._qdes_limit_barrier_reward_contract(cfg, facts)

    facts = _runtime_facts()
    facts["qdes_joint_pos_limits"][4] = [1.0, 1.0]
    with pytest.raises(RuntimeError, match="lo < hi"):
        train_mod._qdes_limit_barrier_reward_contract(cfg, facts)

    facts = _runtime_facts()
    facts["qdes_joint_pos_limits"] = facts["qdes_joint_pos_limits"][:30]
    with pytest.raises(RuntimeError, match="31 runtime"):
        train_mod._qdes_limit_barrier_reward_contract(cfg, facts)


def test_hard_contract_builder_wires_the_conditional_barrier_block():
    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert "_qdes_limit_barrier_reward_contract(" in source
    assert '{"qdes_limit_barrier_reward": qdes_limit_barrier_contract}' in source


# --------------------------------------------------------------------------------------------- #
# schema-3 contract validation (roundtrip + tamper rejection)
# --------------------------------------------------------------------------------------------- #
def _barrier_contract_block(weight=-0.65):
    cfg, _ = _apply_qbar({"rewards": {"qdes_limit_barrier_weight": weight}})
    return train_mod._qdes_limit_barrier_reward_contract(cfg, _runtime_facts())


def _schema3_barrier_base():
    contract = _qdot_hinge_schema3_contract()
    contract.pop("joint_velocity_limit_hinge_reward", None)
    return contract


def test_schema3_roundtrips_enabled_and_zero_weight_control_blocks():
    for weight in (-0.65, 0.0):
        contract = _schema3_barrier_base()
        contract["qdes_limit_barrier_reward"] = _barrier_contract_block(weight)
        TC.validate_schema3_contract_structure(contract)
        TC.validate_schema3_contract(contract)
    # Absent block stays valid (legacy/default runs never acquire the subsection).
    TC.validate_schema3_contract_structure(_schema3_barrier_base())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda b: b.update(weight=0.1), "weight/enabled"),
        (lambda b: b.update(enabled=False), "weight/enabled"),
        (lambda b: b.update(probe_enabled=False), "flags"),
        (lambda b: b.update(margin_frac=0.5), r"\(0, 0.5\)"),
        (lambda b: b.update(margin_frac=True), "finite number"),
        (lambda b: b.update(schema_version=2), "schema_version"),
        (lambda b: b.update(joint_count=15), "identity 31-joint"),
        (lambda b: b.update(joint_order="topk_by_intrusion"), "joint_order"),
        (lambda b: b.update(formula="mean(topk(...))"), "formula"),
        (lambda b: b.update(gate="same_attempt_post_strike_age_s_inclusive"), "gate"),
        (lambda b: b.update(position_limit_source="urdf"), "position_limit_source"),
        (lambda b: b.update(extra_field=1.0), "unknown fields"),
        (lambda b: b.pop("margin_frac"), "missing fields"),
    ],
)
def test_schema3_refuses_drifted_barrier_blocks(mutate, match):
    contract = _schema3_barrier_base()
    block = _barrier_contract_block()
    mutate(block)
    contract["qdes_limit_barrier_reward"] = block
    with pytest.raises(ValueError, match=match):
        TC.validate_schema3_contract_structure(contract)


def test_schema3_refuses_a_barrier_block_on_a_non_identity_joint_order():
    contract = _schema3_barrier_base()
    contract["articulation_joint_names"] = list(
        reversed(contract["articulation_joint_names"])
    )
    contract["qdes_limit_barrier_reward"] = _barrier_contract_block()
    with pytest.raises(ValueError):
        TC.validate_schema3_contract_structure(contract)


# --------------------------------------------------------------------------------------------- #
# runner ledger wiring
# --------------------------------------------------------------------------------------------- #
def test_runner_consumes_the_probe_ledger_once_per_update():
    source = (
        Path(__file__).resolve().parents[1]
        / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
    ).read_text()
    assert '"qdes_limit_barrier_probe" in active_reward_terms' in source
    assert "consume_qdes_limit_barrier_activation_counters" in source
    assert "Live/qdes_limit_barrier/" in source
