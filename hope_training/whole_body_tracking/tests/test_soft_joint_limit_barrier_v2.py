"""Focused CPU tests for fresh ActionBall's non-grazeable soft-limit barrier v2.

The legacy q_des barrier remains covered by ``test_qdes_limit_barrier.py``.  This file pins the
fresh ActionBall-only v2 math, the distinct processed-q_des / actual-q activation channels, the
adopted dose, and the generic-death/table-specific invariants.
"""

from __future__ import annotations

import math
import types
from pathlib import Path

import pytest
import torch
import yaml

from test_reward_flags_mdp import hope_rewards_mod
from test_reward_flags_overrides import _NS, _Term, _apply_legacy_v1, _make_env_cfg, train_mod


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "hope_env_cfg.py"
)
ACTION_BALL_YAML = ROOT / "cfg/task/HOPEPingPongActionBall.yaml"
JOINTS = list(hope_rewards_mod._A3_RUNTIME_JOINT_ORDER)
# 2026-08-05 层级对齐(exp §5.6 第 9 条):两条 soft-limit v2 通道的带宽一起 0.08 -> 0.05。
# 2026-08-07 Franco 裁定二/附加条:再从 0.05 -> 0.02。0.05 恰好等于护栏的投影内沿,
# 带外沿与被钳关节的落点是同一点(实测 31 个关节里 29 个的 m_eff 正好命中),
# intrusion 由浮点舍入决定;0.02 让带边真正离开 clamp 边。
MARGIN = 0.02
# 地板没有删,是**挪到机械硬限位**:软带内连续,反利用性质由硬限位那个不连续点保住。
FLOOR = 0.25
WEIGHT = -10.0
PROJECTION_WEIGHT = -1.0
# 测试夹具的软限位是 [-1,1](span=2),硬限位取 soft/0.9 居中 => [-10/9, 10/9]。
SOFT_LO, SOFT_HI = -1.0, 1.0
HARD_LO, HARD_HI = -10.0 / 9.0, 10.0 / 9.0
SPAN = SOFT_HI - SOFT_LO
BAND = MARGIN * SPAN


def _env(num_envs: int = 2):
    limits = torch.stack(
        (
            torch.full((num_envs, 31), SOFT_LO),
            torch.full((num_envs, 31), SOFT_HI),
        ),
        dim=-1,
    )
    hard_limits = torch.stack(
        (
            torch.full((num_envs, 31), HARD_LO),
            torch.full((num_envs, 31), HARD_HI),
        ),
        dim=-1,
    )
    data = types.SimpleNamespace(
        joint_names=list(JOINTS),
        soft_joint_pos_limits=limits,
        joint_pos_limits=hard_limits,
        default_joint_pos=torch.zeros(num_envs, 31),
        joint_pos=torch.zeros(num_envs, 31),
    )
    asset = types.SimpleNamespace(data=data)
    action = types.SimpleNamespace(
        processed_actions=torch.zeros(num_envs, 31),
        _asset=asset,
        _joint_names=list(JOINTS),
        _joint_ids=slice(None),
    )
    env = types.SimpleNamespace(
        common_step_counter=41,
        action_manager=types.SimpleNamespace(get_term=lambda name: action),
        scene={"robot": asset},
    )
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    return env, action, data, asset_cfg


def _expected(depth_rad: float, *, hard_excess: float = 0.0) -> float:
    """The adopted kernel, written out independently of the implementation."""

    if depth_rad <= 0.0:
        return 0.0
    if depth_rad <= BAND:
        value = depth_rad * depth_rad / (2.0 * BAND)
    else:
        value = depth_rad - 0.5 * BAND
    if hard_excess > 0.0:
        value += FLOOR * BAND
    return value


def test_v2_is_zero_outside_band_and_continuous_at_the_band_edge():
    """人话:带外一分不收,而且刚踩进带里的那一步**不再是一个台阶**。

    这条就是"地板挪走"的验收点。旧核在带边是 0 -> 0.25 的跳变,所以贴着带边的 1 ulp
    浮点抖动也要付钱;新核在带边值和斜率都是 0。
    """

    edge = SOFT_HI - BAND
    env, action, _, _ = _env(3)
    action.processed_actions[0, 0] = edge - 1.0e-4
    action.processed_actions[1, 0] = edge + 1.0e-5
    action.processed_actions[2, 0] = edge + 1.0e-2
    values = hope_rewards_mod.qdes_limit_barrier_v2(env)
    assert values[0].item() == 0.0
    # 连续:刚过带边的收费必须比一个 FLOOR 台阶小若干个数量级。
    # 旧核在这一点是 0 -> FLOOR 的跳变,这条断言当时必然红。
    assert 0.0 < values[1].item() < FLOOR * BAND * 1.0e-4
    assert values[2].item() == pytest.approx(_expected(1.0e-2), rel=1.0e-4)


def test_v2_tail_is_linear_unbounded_and_matches_open_source_slope():
    """人话:越过软限位以后,每多 1 rad 就多罚 1 个单位 —— 和 IsaacLab 的 joint_pos_limits 同价。

    旧核每关节封顶 1,深处梯度衰减到边界处的 1/40;这条测试就是那件事的反面。
    """

    # 全部落在机械硬限位之内(HARD_HI - SOFT_HI = 0.1111),这样测的是纯尾巴,
    # 不混进硬限位那个反利用地板。
    env, action, _, _ = _env(4)
    excesses = (0.0, 0.02, 0.05, 0.08)
    for row, excess in enumerate(excesses):
        action.processed_actions[row, 3] = SOFT_HI + excess
    values = hope_rewards_mod.qdes_limit_barrier_v2(env)
    for row, excess in enumerate(excesses):
        assert values[row].item() == pytest.approx(
            _expected(BAND + excess), rel=1.0e-6
        )
    # 尾部斜率恒为 1 rad/rad:没有上界,也不衰减 —— 深处和浅处一模一样。
    assert values[2].item() - values[1].item() == pytest.approx(0.03, rel=1.0e-5)
    assert values[3].item() - values[2].item() == pytest.approx(0.03, rel=1.0e-5)
    # 无上界:再走 10 rad 仍然线性增长,不封顶在 1。
    env2, action2, _, _ = _env(1)
    action2.processed_actions[0, 3] = SOFT_HI + 10.0
    assert hope_rewards_mod.qdes_limit_barrier_v2(env2).item() == pytest.approx(
        _expected(BAND + 10.0, hard_excess=1.0), rel=1.0e-6
    )


def test_v2_anti_exploitation_floor_sits_on_the_mechanical_hard_edge():
    """人话:反利用地板没删,是挪到"撞上机械边"那一点。

    2026-08-07 裁定三取消的是实际-q 硬超限的**阻断**,不是这条兜底的定价。
    """

    env, action, _, _ = _env(2)
    just_inside = HARD_HI - 1.0e-6
    just_outside = HARD_HI + 1.0e-6
    action.processed_actions[0, 7] = just_inside
    action.processed_actions[1, 7] = just_outside
    values = hope_rewards_mod.qdes_limit_barrier_v2(env)
    step = values[1].item() - values[0].item()
    # 不连续点存在,而且大小就是 FLOOR*BAND —— "不存在一串正违规而罚金趋于零的路径"。
    assert step == pytest.approx(FLOOR * BAND, rel=1.0e-3)
    assert values[0].item() == pytest.approx(
        _expected(BAND + (just_inside - SOFT_HI)), rel=1.0e-6
    )


def test_v2_sum_aggregation_is_undiluted():
    env, action, _, _ = _env(1)
    action.processed_actions[0, [2, 5, 19]] = SOFT_HI + 0.05
    summed = hope_rewards_mod.qdes_limit_barrier_v2(env)
    assert summed.item() == pytest.approx(3.0 * _expected(BAND + 0.05), rel=1.0e-5)
    single = _env(1)
    single[1].processed_actions[0, 2] = SOFT_HI + 0.05
    one = hope_rewards_mod.qdes_limit_barrier_v2(single[0])
    # SUM,不是 mean、不是 top-k:三个关节就是单关节的三倍,一分不稀释。
    assert summed.item() == pytest.approx(3.0 * one.item(), rel=1.0e-5)


def test_v2_margin_must_stay_inside_the_projection_envelope_inset():
    """人话:带宽必须**严格小于**护栏投影内沿 0.05,否则被钳关节正好压在带边上。

    这是 2026-08-07 那条"29/31 个关节的 m_eff 恰等于 clamp 内缩"的回归门。
    """

    projection_inset = 0.05
    assert MARGIN < projection_inset
    clamped = SOFT_HI - projection_inset * SPAN
    env, action, _, _ = _env(1)
    action.processed_actions[0, :] = clamped
    assert hope_rewards_mod.qdes_limit_barrier_v2(env).item() == 0.0


def test_v2_stance_exemption_is_free_but_moving_toward_limit_is_not():
    env, action, data, _ = _env(2)
    # d(default) = 0.015,因此 m_eff = 0.010 (0.005 的呼吸间隙之后),带宽 = 0.020 rad。
    data.default_joint_pos[:, 5] = -0.97
    action.processed_actions[0, 5] = -0.97
    action.processed_actions[1, 5] = -0.99
    values = hope_rewards_mod.qdes_limit_barrier_v2(env)
    assert values[0].item() == 0.0
    assert values[1].item() > 0.0


def test_qdes_and_actual_q_activate_as_distinct_objectives():
    env, action, data, asset_cfg = _env(2)
    action.processed_actions[0, 7] = 0.99
    data.joint_pos[1, 9] = -0.99
    qdes = hope_rewards_mod.qdes_limit_barrier_v2(env)
    actual = hope_rewards_mod.actual_joint_limit_barrier_v2(env, asset_cfg)
    assert qdes[0].item() > 0.0 and qdes[1].item() == 0.0
    assert actual[0].item() == 0.0 and actual[1].item() > 0.0


def test_v2_probes_are_zero_and_ledger_keeps_qdes_actual_separate():
    env, action, data, asset_cfg = _env(2)
    action.processed_actions[0, 1] = 0.99
    data.joint_pos[1, 2] = -0.99
    assert torch.equal(
        hope_rewards_mod.qdes_limit_barrier_v2_probe(env), torch.zeros(2)
    )
    hope_rewards_mod.qdes_limit_barrier_v2(env)
    assert torch.equal(
        hope_rewards_mod.actual_joint_limit_barrier_v2_probe(env, asset_cfg),
        torch.zeros(2),
    )
    hope_rewards_mod.actual_joint_limit_barrier_v2(env, asset_cfg)

    counters = hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(env)
    assert counters["qdes_observed_sample_count"].item() == 2
    assert counters["actual_observed_sample_count"].item() == 2
    assert counters["qdes_intrusion_joint_count"].item() == 1
    assert counters["actual_intrusion_joint_count"].item() == 1
    assert counters["qdes_reward_enabled_sample_count"].item() == 2
    assert counters["actual_reward_enabled_sample_count"].item() == 2
    assert counters["qdes_barrier_value_sum"].item() > 0.0
    assert counters["actual_barrier_value_sum"].item() > 0.0
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(
            env
        ).values()
    )


def test_v2_activation_requires_step_identity_and_both_channels():
    env, _, _, _ = _env(1)
    del env.common_step_counter
    with pytest.raises(RuntimeError, match="common_step_counter"):
        hope_rewards_mod.qdes_limit_barrier_v2(env)

    env, _, _, _ = _env(1)
    hope_rewards_mod.qdes_limit_barrier_v2(env)
    with pytest.raises(RuntimeError, match="both qdes and actual channels"):
        hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(env)


def test_v2_activation_state_corruption_fails_closed():
    env, _, _, _ = _env(1)
    setattr(
        env,
        hope_rewards_mod._QDES_LIMIT_BARRIER_V2_ACTIVATION_ATTR,
        {"observed_sample_count": torch.zeros((), dtype=torch.long)},
    )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        hope_rewards_mod.qdes_limit_barrier_v2(env)


@pytest.mark.parametrize("source", ("qdes", "actual"))
@pytest.mark.parametrize(
    "field,value,match",
    (
        ("margin_frac", 0.0, r"\(0, 0.5\)"),
        ("margin_frac", 0.5, r"\(0, 0.5\)"),
        # 2026-08-07:地板下界改成闭区间(0.0 是合法的"不要硬边地板"消融),上界不变。
        ("penalty_floor", -0.001, r"\[0, 1\)"),
        ("penalty_floor", 1.0, r"\[0, 1\)"),
        ("penalty_floor", math.nan, r"\[0, 1\)"),
    ),
)
def test_v2_invalid_scalars_fail_closed(source, field, value, match):
    env, _, _, asset_cfg = _env(1)
    kwargs = {field: value}
    with pytest.raises(ValueError, match=match):
        if source == "qdes":
            hope_rewards_mod.qdes_limit_barrier_v2(env, **kwargs)
        else:
            hope_rewards_mod.actual_joint_limit_barrier_v2(env, asset_cfg, **kwargs)


def test_action_ball_config_uses_v2_callables_and_separate_probe_terms():
    source = ENV_CFG.read_text(encoding="utf-8")
    start = source.index("class HOPEActionBallRewardsCfg")
    end = source.index("class HOPEActionBallTerminationsCfg", start)
    block = source[start:end]
    assert "func=mdp.qdes_limit_barrier_v2," in block
    assert "func=mdp.qdes_limit_barrier_v2_probe," in block
    assert "func=mdp.actual_joint_limit_barrier_v2," in block
    assert "func=mdp.actual_joint_limit_barrier_v2_probe," in block
    assert block.count('"penalty_floor": 0.25') == 4
    assert block.count('"margin_frac": 0.02') == 4
    # 2026-08-07 裁定二:两条 barrier 换开源 rad 口径 -> -10;投影罚重算剂量 -> -1。
    assert block.count("weight=-10.0") == 2  # qdes, actual-q
    assert block.count("weight=-1.0") == 1  # projection
    assert "weight=-5.0" not in block
    assert '"knee_frac": 0.05' in block
    for term_name in (
        "qdes_limit_barrier_probe",
        "actual_joint_limit_barrier_probe",
    ):
        term_start = block.index(f"{term_name} = RewTerm(")
        term_end = block.index("\n    )", term_start)
        assert "weight=1.0" in block[term_start:term_end]


def test_adopted_scale_and_generic_death_invariants_are_pinned():
    task = yaml.safe_load(ACTION_BALL_YAML.read_text(encoding="utf-8"))
    rewards = task["rewards"]
    # 2026-08-07 Franco 裁定二:核/量纲换成开源 rad 口径,权重号码必须跟着换。
    # 旧 -5 作用在每关节归一 [0,1] 上,与新数**不可比**;-10 是上游 BeyondMimic /
    # mjlab-tracking / unitree_rl_lab-mimic 全身版的同一个数。
    assert rewards["qdes_limit_barrier_weight"] == pytest.approx(WEIGHT)
    assert rewards["joint_limit_weight"] == pytest.approx(WEIGHT)
    # v2 硬合同要求 q_des / actual-q 通道逐字段同权同带宽,所以断言的是"相等"本身。
    assert rewards["qdes_limit_barrier_margin_frac"] == pytest.approx(MARGIN)
    assert rewards["joint_limit_margin_frac"] == pytest.approx(MARGIN)
    assert (
        rewards["joint_limit_margin_frac"] == rewards["qdes_limit_barrier_margin_frac"]
    )
    assert rewards["joint_limit_weight"] == rewards["qdes_limit_barrier_weight"]
    # 2026-08-05 层级对齐(exp §5.6 第 7 条):death -300.0 -> -10.0(post-dt -6.0 -> -0.2)。
    assert rewards["death_penalty_weight"] == pytest.approx(-10.0)
    assert rewards["table_hit_penalty_weight"] == pytest.approx(0.0)

    # 带宽必须严格窄于护栏的投影内沿,否则罚的是护栏自己放进去的位置。
    assert rewards["qdes_limit_barrier_margin_frac"] < 0.05

    policy_dt = 0.02
    # 开源交叉验证:build_1 收敛态全身越软限位总量 0.003 rad。同一个核、同一个权重,
    # 每步剂量必须重现它日志里的 -0.0006。这条是"同等策略水平交叉验证"的回归门:
    # 权重、量纲、dt 任何一个搞错一档,这个数就差一个数量级。
    build1_converged_excess_rad = 0.003
    assert abs(WEIGHT) * build1_converged_excess_rad * policy_dt == pytest.approx(
        0.0006, abs=1.0e-9
    )
    # build_1 早期峰值 -0.0040/步 对应 0.02 rad 的全身越限量。
    assert abs(WEIGHT) * 0.02 * policy_dt == pytest.approx(0.0040, abs=1.0e-9)
    # 一个关节整整越出软限位 0.10 rad,单步单关节 -0.02;没有上界,再深就线性再涨。
    assert abs(WEIGHT) * 0.10 * policy_dt == pytest.approx(0.02, abs=1.0e-12)
    landing_max = 500.0 * policy_dt
    assert landing_max == pytest.approx(10.0)
    assert abs(-10.0 * policy_dt) == pytest.approx(0.2)


def test_projection_penalty_adopted_dose_is_the_recomputed_one():
    """裁定二:核换了就必须重算剂量,不能沿用 -5。"""

    source = ENV_CFG.read_text(encoding="utf-8")
    start = source.index("qdes_projection_penalty = RewTerm(")
    block = source[start : source.index("\n    )", start)]
    assert "weight=-1.0," in block
    assert '"knee_frac": 0.05,' in block
    assert "shape_rate" not in block


# --------------------------------------------------------------------------------------------- #
# 2026-08-07 裁定一的前置:深度遥测必须真的能被读到
# --------------------------------------------------------------------------------------------- #
def test_depth_telemetry_is_readable_and_does_not_clear_itself():
    """人话:按深度验收的前提是先能读到深度。

    这些计数器机制层每步都在算,但在此之前没有任何消费方把它们写进收据。
    ``peek_*`` 是只读的:连读两次必须给出同一份数,不能像 ``consume_*`` 那样清零。
    """

    env, action, data, asset_cfg = _env(2)
    action.processed_actions[0, 4] = SOFT_HI + 0.03
    data.joint_pos[1, 6] = -(SOFT_HI + 0.01)
    hope_rewards_mod.qdes_limit_barrier_v2(env)
    hope_rewards_mod.actual_joint_limit_barrier_v2(env, asset_cfg)

    first = hope_rewards_mod.peek_qdes_depth_telemetry(env)
    assert first is not None
    assert first["qdes_observed_sample_count"] == 2
    assert first["qdes_intrusion_joint_count"] == 1
    assert first["actual_intrusion_joint_count"] == 1
    # 深度口径:1.0 = 正好压在软限位上,>1.0 = 已经越过软限位(单位是带宽的倍数)。
    assert first["qdes_max_intrusion_depth_frac"] == pytest.approx(
        (BAND + 0.03) / BAND, rel=1.0e-4
    )
    assert first["actual_max_intrusion_depth_frac"] == pytest.approx(
        (BAND + 0.01) / BAND, rel=1.0e-4
    )
    # 只读:再读一次必须逐位相同。
    assert hope_rewards_mod.peek_qdes_depth_telemetry(env) == first


def test_depth_telemetry_is_absent_rather_than_faked_when_nothing_ran():
    env, _, _, _ = _env(1)
    assert hope_rewards_mod.peek_qdes_depth_telemetry(env) is None


def test_runner_writes_the_depth_block_into_the_economy_receipt():
    """变异测试:如果有人把 economy JSON 里的深度块删了,这里必须红。"""

    runner = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
    ).read_text(encoding="utf-8")
    assert '"joint_limit_depth": self._reward_ppo_economy_depth_telemetry()' in runner
    assert "peek_qdes_depth_telemetry" in runner
    assert '"joint_limit_depth_semantics"' in runner


def test_economy_receipt_raw_bounds_are_recomputed_from_the_urdf_not_hand_copied():
    """从 URDF + 站姿 + 采纳带宽重算两条通道的可达上界,和收据里的常数对拍。

    收据里那两个数是"物理上够得到的最大值",不是核的每关节上限 —— 新核没有上限。
    """

    import xml.etree.ElementTree as ET

    urdf = ROOT.parent.parent / "agi/URDF/a3_t2d5/urdf/model.urdf"
    if not urdf.exists():  # pragma: no cover - asset not present in this checkout
        pytest.skip("A3 URDF is not present in this checkout")
    defaults = {
        "hip_pitch": -0.1311, "knee": 0.2468, "ankle_pitch": -0.1204,
        "shoulder_pitch": 0.3, "elbow": 0.8,
    }
    exact = {
        "left_hip_roll_joint": 0.0056, "right_hip_roll_joint": -0.0056,
        "left_hip_yaw_joint": -0.0348, "right_hip_yaw_joint": 0.0348,
        "left_ankle_roll_joint": -0.0078, "right_ankle_roll_joint": 0.0078,
        "left_shoulder_roll_joint": 0.12, "right_shoulder_roll_joint": -0.12,
    }
    qdes_total = 0.0
    actual_total = 0.0
    count = 0
    for joint in ET.parse(str(urdf)).getroot().iter("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        count += 1
        name = joint.get("name")
        hard_lo, hard_hi = float(limit.get("lower")), float(limit.get("upper"))
        hard_span = hard_hi - hard_lo
        mid, half = 0.5 * (hard_lo + hard_hi), 0.45 * hard_span
        soft_lo, soft_hi = mid - half, mid + half
        soft_span = soft_hi - soft_lo
        default = exact.get(name, 0.0)
        if name not in exact:
            for suffix, value in defaults.items():
                if name.endswith(suffix + "_joint"):
                    default = value
        d_default = min(default - soft_lo, soft_hi - default) / soft_span
        m_eff = min(MARGIN, d_default - 0.005)
        band = m_eff * soft_span
        qdes_total += 0.5 * band
        actual_total += 0.5 * band + 0.05 * hard_span + FLOOR * band
    assert count == 31

    receipt = (
        ROOT / "scripts/materialize_action_ball_reward_ppo_economy_receipt.py"
    ).read_text(encoding="utf-8")
    namespace: dict = {}
    for line in receipt.splitlines():
        if line.startswith(("_QDES_LIMIT_BARRIER_RAW_MAX", "_ACTUAL_LIMIT_BARRIER_RAW_MAX")):
            exec(line, namespace)  # noqa: S102 - two float literals from our own file
    assert namespace["_QDES_LIMIT_BARRIER_RAW_MAX"] == pytest.approx(
        qdes_total, abs=1.0e-5
    )
    assert namespace["_ACTUAL_LIMIT_BARRIER_RAW_MAX"] == pytest.approx(
        actual_total, abs=1.0e-5
    )


# --------------------------------------------------------------------------------------------- #
# train.py override translation: the actual-q band travels with the q_des band
# --------------------------------------------------------------------------------------------- #
_ACTUAL_PARAMS = {
    "margin_frac": 0.08,
    "penalty_floor": 0.25,
    "expected_joint_count": 31,
}


def _actual_band_env_cfg():
    cfg = _make_env_cfg()
    cfg.rewards.joint_limit = _Term(
        weight=-5.0,
        params={"asset_cfg": _NS(name="robot", joint_ids=slice(None)), **_ACTUAL_PARAMS},
    )
    cfg.rewards.actual_joint_limit_barrier_probe = _Term(
        weight=1.0,
        params={"asset_cfg": _NS(name="robot", joint_ids=slice(None)), **_ACTUAL_PARAMS},
    )
    return cfg


def _apply_actual_band(task, cfg=None):
    cfg = cfg if cfg is not None else _actual_band_env_cfg()
    return cfg, _apply_legacy_v1(cfg, task)


def test_actual_band_override_moves_term_and_probe_together():
    cfg, applied = _apply_actual_band(
        {"rewards": {"joint_limit_weight": -5.0, "joint_limit_margin_frac": MARGIN}}
    )
    term = cfg.rewards.joint_limit
    probe = cfg.rewards.actual_joint_limit_barrier_probe
    assert term.params["margin_frac"] == pytest.approx(MARGIN)
    assert probe.params["margin_frac"] == pytest.approx(MARGIN)
    assert probe.weight == pytest.approx(1.0)
    assert term.weight == pytest.approx(-5.0)
    assert f"rewards.joint_limit.params.margin_frac={MARGIN}" in applied
    assert any(
        marker.startswith("rewards.actual_joint_limit_barrier_probe=")
        for marker in applied
    )


def test_actual_band_without_explicit_weight_is_refused():
    with pytest.raises(train_mod._OverrideError, match="joint_limit_weight"):
        _apply_actual_band({"rewards": {"joint_limit_margin_frac": MARGIN}})


@pytest.mark.parametrize("margin", [0.0, 0.5, -0.1, float("nan"), True, "bad"])
def test_invalid_actual_band_override_is_refused(margin):
    with pytest.raises(train_mod._OverrideError, match=r"\(0, 0.5\)"):
        _apply_actual_band(
            {
                "rewards": {
                    "joint_limit_weight": -5.0,
                    "joint_limit_margin_frac": margin,
                }
            }
        )


def test_invalid_actual_band_does_not_partially_mutate_either_term():
    cfg = _actual_band_env_cfg()
    term = cfg.rewards.joint_limit
    probe = cfg.rewards.actual_joint_limit_barrier_probe
    before = (dict(term.params), probe.weight, dict(probe.params))
    with pytest.raises(train_mod._OverrideError):
        _apply_actual_band(
            {
                "rewards": {
                    "joint_limit_weight": -5.0,
                    "joint_limit_margin_frac": 0.5,
                }
            },
            cfg=cfg,
        )
    assert (dict(term.params), probe.weight, dict(probe.params)) == before


def test_reward_keys_whitelist_carries_both_actual_band_keys_once():
    keys = [key for key in train_mod._REWARD_KEYS if key.startswith("joint_limit")]
    assert sorted(keys) == ["joint_limit_margin_frac", "joint_limit_weight"]
    with pytest.raises(train_mod._OverrideError, match="joint_limit_margin_farc"):
        _apply_actual_band({"rewards": {"joint_limit_margin_farc": MARGIN}})
