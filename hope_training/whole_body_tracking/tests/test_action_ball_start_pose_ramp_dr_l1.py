"""Fail-closed contract for the start-pose ramp and the DR-L1 restored plant.

Two things are under test here and they are deliberately in one file, because
the whole point of the change is that they are one story:

* the *ramp* is the only authorized way for the ActionBall birth pose to stop
  being a literal constant, and
* *DR-L1* is a NEW level that restores the day-1 plant randomization without
  touching DR-L0's ``exact all-off`` identity, which stays available as the
  attribution control.

Every "still refused" case below is a mutation test: it proves the old hard
gate is still standing whenever the new declaration is absent or incomplete,
rather than proving only that the new happy path works.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/train.py"
TRAINING_CONTRACT = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)
REPO_ROOT = ROOT.parents[1]
DR_L1_MANIFEST = (
    REPO_ROOT
    / "configs/action_ball_n1_measured_20260805"
    / "action_ball_211_dr_l1_restored_plant_candidate.v1.json"
)
DR_L0_MANIFEST = (
    REPO_ROOT
    / "configs/action_ball_n1_measured_20260803"
    / "action_ball_211_dr_l0_learnability_candidate.v1.json"
)
AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TC = _load_module("start_pose_ramp_training_contract_under_test", TRAINING_CONTRACT)


def _load_train_module():
    pytest.importorskip("hydra")
    source_root = str((ROOT / "source/whole_body_tracking").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return _load_module("start_pose_ramp_train_under_test", TRAIN)


# --------------------------------------------------------------------------- #
# (B) 换算:终点值必须还原成 Franco 说的那三句话
# --------------------------------------------------------------------------- #
def test_four_cell_endpoints_reproduce_the_requested_world_frame_region():
    """Endpoint offsets must map back onto the requested world-frame region.

    世界系:原点在近端左桌角,台面 2.74 x 1.525;机器人地面原点
    world [-0.5, -0.7625, -0.76]。把偏移量加回标准站位,必须正好落在
    "桌后 1 m / 左右各出界 0.5 m / 朝向 ±30 度"上。
    """

    ramp = TC.validate_action_ball_start_pose_ramp(
        TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL, name="four_cell"
    )
    standard_x, standard_y = -0.5, -0.7625
    table_length_x, table_width_y = 2.74, 1.525
    assert table_length_x > 0.0  # 世界系方向记号,防止有人把 X/Y 写反

    lo_x, hi_x = ramp["pose_range"]["x"]
    assert pytest.approx(standard_x + lo_x, abs=1e-12) == -1.5
    assert pytest.approx(standard_x + hi_x, abs=1e-12) == -0.5

    lo_y, hi_y = ramp["pose_range"]["y"]
    # 台宽占 y in [-1.525, 0];两侧各外扩 0.5 m
    assert pytest.approx(standard_y + lo_y, abs=1e-12) == -table_width_y - 0.5
    assert pytest.approx(standard_y + hi_y, abs=1e-12) == 0.5
    # 标准站位在中线上,所以左右偏移必须对称
    assert pytest.approx(-lo_y, abs=1e-12) == hi_y

    import math

    lo_yaw, hi_yaw = ramp["pose_range"]["yaw"]
    assert pytest.approx(math.degrees(hi_yaw), abs=1e-9) == 30.0
    assert pytest.approx(math.degrees(lo_yaw), abs=1e-9) == -30.0

    # 高度与滚转/俯仰归 ready 合同拥有,斜坡不许碰
    for axis in ("z", "roll", "pitch"):
        assert ramp["pose_range"][axis] == [0.0, 0.0]
    # 出生速度与关节复位噪声保持零
    assert all(ramp["velocity_range"][axis] == [0.0, 0.0] for axis in AXES)
    assert ramp["joint_position_range"] == [0.0, 0.0]
    # ramp 步数与 build_1 的站立位姿 ramp 同量级
    assert ramp["ramp_steps"] == 96000


def test_normalization_is_idempotent_but_rejects_a_foreign_schema():
    """train.py installs the normalized payload; the runtime re-validates it."""

    once = TC.validate_action_ball_start_pose_ramp(
        TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL, name="four_cell"
    )
    twice = TC.validate_action_ball_start_pose_ramp(once, name="four_cell")
    assert twice == once
    assert TC.action_ball_start_pose_ramp_sha256(
        twice
    ) == TC.action_ball_start_pose_ramp_sha256(once)
    foreign = deepcopy(once)
    foreign["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        TC.validate_action_ball_start_pose_ramp(foreign, name="foreign")
    foreign = deepcopy(once)
    foreign["kind"] = "action_ball_start_pose_ramp_v2"
    with pytest.raises(ValueError, match="kind"):
        TC.validate_action_ball_start_pose_ramp(foreign, name="foreign")
    # 未规范化的 payload 不许直接进摘要,免得把一份拼写当成法则绑进合同
    with pytest.raises(ValueError, match="normalized"):
        TC.action_ball_start_pose_ramp_sha256(
            dict(TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL)
        )


def test_disabled_ramp_is_the_literal_legacy_identity():
    disabled = TC.validate_action_ball_start_pose_ramp(None, name="absent")
    assert disabled["enabled"] is False
    assert disabled == TC.ACTION_BALL_START_POSE_RAMP_DISABLED
    assert TC.action_ball_start_pose_ramp_progress(disabled, 10**9) == 0.0
    for field in ("pose_range", "velocity_range"):
        for axis in AXES:
            assert TC.action_ball_start_pose_ramp_axis_range(
                disabled,
                field=field,
                axis=axis,
                static=(-0.3, 0.4),
                progress=1.0,
            ) == (-0.3, 0.4)
    assert TC.action_ball_start_pose_ramp_hold_window(
        disabled, static=(7, 11), progress=1.0
    ) == (7, 11)


def test_progress_is_monotone_clamped_and_starts_at_the_static_seed():
    ramp = TC.validate_action_ball_start_pose_ramp(
        TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL, name="four_cell"
    )
    steps = [0, 1, 100, 48000, 95999, 96000, 96001, 10**7]
    fractions = [
        TC.action_ball_start_pose_ramp_progress(ramp, step) for step in steps
    ]
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert all(0.0 <= value <= 1.0 for value in fractions)
    assert fractions == sorted(fractions)

    # progress=0 必须逐字节等于静态种子(这就是"第 0 步和过去一样"的含义)
    for axis in AXES:
        assert TC.action_ball_start_pose_ramp_axis_range(
            ramp, field="pose_range", axis=axis, static=(0.0, 0.0), progress=0.0
        ) == (0.0, 0.0)
    # progress=1 必须逐字节等于声明的终点
    for axis in AXES:
        assert list(
            TC.action_ball_start_pose_ramp_axis_range(
                ramp,
                field="pose_range",
                axis=axis,
                static=(0.0, 0.0),
                progress=1.0,
            )
        ) == ramp["pose_range"][axis]


@pytest.mark.parametrize(
    "mutate",
    (
        pytest.param(lambda spec: spec.pop("ramp_steps"), id="missing_key"),
        pytest.param(
            lambda spec: spec.update(extra_axis=1), id="unknown_key"
        ),
        pytest.param(
            lambda spec: spec.update(enabled="true"), id="stringly_bool"
        ),
        pytest.param(
            lambda spec: spec.update(ramp_steps=0), id="zero_length_ramp"
        ),
        pytest.param(
            lambda spec: spec.update(ramp_steps=-1), id="negative_ramp"
        ),
        pytest.param(
            lambda spec: spec["pose_range"].pop("yaw"), id="missing_axis"
        ),
        pytest.param(
            lambda spec: spec["pose_range"].update(yaw=[0.5, -0.5]),
            id="inverted_pair",
        ),
        pytest.param(
            lambda spec: spec["pose_range"].update(yaw=[float("nan"), 0.5]),
            id="non_finite",
        ),
        pytest.param(
            lambda spec: spec.update(hold_steps_range_start=[45, 60]),
            id="two_hold_clocks",
        ),
        pytest.param(
            lambda spec: spec.update(hold_clock_owner="whoever"),
            id="unknown_hold_owner",
        ),
    ),
)
def test_ramp_declaration_is_refused_when_incomplete_or_malformed(mutate):
    spec = deepcopy(dict(TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL))
    spec["pose_range"] = deepcopy(spec["pose_range"])
    mutate(spec)
    with pytest.raises(ValueError):
        TC.validate_action_ball_start_pose_ramp(spec, name="mutated")


def test_hold_lower_bound_may_only_contract_never_grow():
    spec = deepcopy(dict(TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL))
    spec["hold_clock_owner"] = TC.ACTION_BALL_START_POSE_RAMP_HOLD_OWNER_MOTION
    spec["hold_steps_range_start"] = [45, 60]
    spec["hold_steps_range_end"] = [20, 60]
    ramp = TC.validate_action_ball_start_pose_ramp(spec, name="motion_hold")
    assert TC.action_ball_start_pose_ramp_hold_window(
        ramp, static=(0, 0), progress=0.0
    ) == (45, 60)
    assert TC.action_ball_start_pose_ramp_hold_window(
        ramp, static=(0, 0), progress=1.0
    ) == (20, 60)
    windows = [
        TC.action_ball_start_pose_ramp_hold_window(
            ramp, static=(0, 0), progress=value / 8.0
        )
        for value in range(9)
    ]
    assert [lo for lo, _hi in windows] == sorted(
        (lo for lo, _hi in windows), reverse=True
    )
    assert all(lo <= hi for lo, hi in windows)
    # 下限"随熟练度扩张"是反向的,必须拒
    spec["hold_steps_range_end"] = [50, 60]
    with pytest.raises(ValueError, match="contract"):
        TC.validate_action_ball_start_pose_ramp(spec, name="growing_lower")
    # 归属是 motion 时,零窗口("两拍之间没有间隔")同样拒
    spec["hold_steps_range_start"] = [0, 0]
    spec["hold_steps_range_end"] = [0, 0]
    with pytest.raises(ValueError, match="no gap"):
        TC.validate_action_ball_start_pose_ramp(spec, name="zero_window")


def test_seed_must_stay_inside_zero_to_endpoint():
    ramp = TC.validate_action_ball_start_pose_ramp(
        TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL, name="four_cell"
    )
    inside = TC.action_ball_start_pose_ramp_seed_within_endpoint
    assert inside(ramp, field="pose_range", axis="y", static=(0.0, 0.0))
    assert inside(ramp, field="pose_range", axis="y", static=(-0.5, 0.5))
    assert inside(ramp, field="pose_range", axis="x", static=(-1.0, 0.0))
    # 超过终点
    assert not inside(ramp, field="pose_range", axis="x", static=(-1.5, 0.0))
    # 与终点反号(终点只许往后退,种子却往前挤)
    assert not inside(ramp, field="pose_range", axis="x", static=(0.0, 0.2))
    # z 终点是零,任何非零种子都越界
    assert not inside(ramp, field="pose_range", axis="z", static=(-0.01, 0.01))
    # 斜坡关闭时,唯一合法的种子就是逐字节零
    disabled = TC.validate_action_ball_start_pose_ramp(None, name="absent")
    assert inside(disabled, field="pose_range", axis="y", static=(0.0, 0.0))
    assert not inside(disabled, field="pose_range", axis="y", static=(0.0, 0.1))


# --------------------------------------------------------------------------- #
# (A) train.py 的硬门:未启用 = 一字不变;启用 = 按终点校验
# --------------------------------------------------------------------------- #
def _motion_cfg(**overrides):
    zero = {axis: (0.0, 0.0) for axis in AXES}
    base = dict(
        joint_position_range=(0.0, 0.0),
        pose_range=deepcopy(zero),
        velocity_range=deepcopy(zero),
        start_pose_ramp=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _run_reset_noise_gate(train, motion_cfg, applied=None):
    """Drive the EXACT production gate, never a mirror of it."""

    return train._validate_action_ball_reset_noise_against_ramp(
        motion_cfg, [] if applied is None else applied
    )


def test_absent_ramp_keeps_the_old_all_zero_hard_gate_bit_for_bit():
    train = _load_train_module()
    # 全零 + 无斜坡 = 通过(旧行为)
    _run_reset_noise_gate(train, _motion_cfg())
    # 非零 + 无斜坡 = 仍然当场拒(这是那道硬门,没被删掉)
    for mutation in (
        {"pose_range": {**{axis: (0.0, 0.0) for axis in AXES}, "x": (-0.1, 0.1)}},
        {
            "velocity_range": {
                **{axis: (0.0, 0.0) for axis in AXES},
                "yaw": (-0.2, 0.2),
            }
        },
        {"joint_position_range": (-0.01, 0.01)},
    ):
        with pytest.raises(train._OverrideError):
            _run_reset_noise_gate(train, _motion_cfg(**mutation))


def test_enabled_ramp_admits_seeds_inside_the_endpoint_and_refuses_beyond():
    train = _load_train_module()
    ramp_spec = deepcopy(dict(TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL))
    seeded = {axis: (0.0, 0.0) for axis in AXES}
    seeded["y"] = (-0.4, 0.4)
    applied = []
    _run_reset_noise_gate(
        train,
        _motion_cfg(pose_range=seeded, start_pose_ramp=ramp_spec),
        applied,
    )
    assert len(applied) == 1
    assert applied[0].startswith("task.motion.start_pose_ramp=enabled(")
    assert "ramp_steps=96000" in applied[0]

    beyond = {axis: (0.0, 0.0) for axis in AXES}
    beyond["y"] = (-2.0, 2.0)
    with pytest.raises(train._OverrideError):
        _run_reset_noise_gate(
            train, _motion_cfg(pose_range=beyond, start_pose_ramp=ramp_spec)
        )


def test_train_whitelists_start_pose_ramp_and_keeps_the_generic_speed_pinned():
    """The 0.85 floor is a teacher_rate ruling, not a generic-sampler licence.

    ``bind_action_ball_task_authority`` refuses any ActionBall run whose
    ``motion.speed_scale_range`` is not exactly ``(1.0, 1.0)`` — the per-swing
    rate belongs to the task receipt.  Relaxing the pre-flight gate here would
    only move a guaranteed runtime refusal later, so it stays pinned.
    """

    train = _load_train_module()
    assert "start_pose_ramp" in train._MOTION_KEYS
    assert train.ACTION_BALL_MIN_SPEED_SCALE_LOWER == 0.85
    source = TRAIN.read_text(encoding="utf-8")
    assert "if speed_range != (1.0, 1.0):" in source
    assert "teacher_rate" in source


# --------------------------------------------------------------------------- #
# (E) DR-L1:恢复 day-1 基线,同时不动 DR-L0 的身份
# --------------------------------------------------------------------------- #
def _event_term(mode: str, params: dict):
    return types.SimpleNamespace(mode=mode, params=dict(params))


def _restored_events():
    spec = TC.ACTION_BALL_DR_L1_ACTIVE_EVENTS
    return types.SimpleNamespace(
        physics_material=_event_term(
            "startup",
            {
                "static_friction_range": tuple(
                    spec["physics_material"]["static_friction_range"]
                ),
                "dynamic_friction_range": tuple(
                    spec["physics_material"]["dynamic_friction_range"]
                ),
                "restitution_range": tuple(
                    spec["physics_material"]["restitution_range"]
                ),
                "num_buckets": spec["physics_material"]["num_buckets"],
            },
        ),
        add_joint_default_pos=_event_term(
            "startup",
            {
                "pos_distribution_params": tuple(
                    spec["add_joint_default_pos"]["pos_distribution_params"]
                ),
                "operation": spec["add_joint_default_pos"]["operation"],
            },
        ),
        base_com=_event_term(
            "startup",
            {
                "com_range": {
                    axis: tuple(value)
                    for axis, value in spec["base_com"]["com_range"].items()
                }
            },
        ),
        randomize_link_mass=_event_term(
            "startup",
            {
                "mass_distribution_params": tuple(
                    spec["randomize_link_mass"]["mass_distribution_params"]
                ),
                "operation": spec["randomize_link_mass"]["operation"],
                "distribution": spec["randomize_link_mass"]["distribution"],
                "recompute_inertia": True,
            },
        ),
        randomize_pd_gains=_event_term(
            "startup",
            {
                "stiffness_distribution_params": tuple(
                    spec["randomize_pd_gains"]["stiffness_distribution_params"]
                ),
                "damping_distribution_params": tuple(
                    spec["randomize_pd_gains"]["damping_distribution_params"]
                ),
                "operation": spec["randomize_pd_gains"]["operation"],
                "distribution": spec["randomize_pd_gains"]["distribution"],
            },
        ),
        push_robot=None,
        force_push=None,
        force_push_sweep=None,
        combined_push=None,
        combined_push_sweep=None,
    )


def _l1_env(*, start_pose_ramp=None):
    zero = {axis: (0.0, 0.0) for axis in AXES}
    return types.SimpleNamespace(
        events=_restored_events(),
        observations=types.SimpleNamespace(
            policy=types.SimpleNamespace(enable_corruption=True)
        ),
        actions=types.SimpleNamespace(
            joint_pos=types.SimpleNamespace(
                control_step_action_delay_min=0,
                control_step_action_delay_max=0,
            )
        ),
        commands=types.SimpleNamespace(
            motion=types.SimpleNamespace(
                joint_position_range=(0.0, 0.0),
                stand_start_yaw_range=(0.0, 0.0),
                pose_range=deepcopy(zero),
                velocity_range=deepcopy(zero),
                start_pose_ramp=start_pose_ramp,
            ),
            racket_target=types.SimpleNamespace(
                achieved_target_mix_prob=0.0,
                midswing_resample_prob=0.0,
                target_delay_steps=0,
                target_jitter_pos_per_s=0.0,
                target_jitter_vel_per_s=0.0,
                target_noise_white=0.0,
                target_noise_ar1_sigma=0.0,
                target_dropout_prob=0.0,
                target_post_strike_dropout_s=0.0,
                target_bias_per_swing=0.0,
                action_ball_target_observation_noise=False,
            ),
        ),
        push=types.SimpleNamespace(enable=False),
        force_push=types.SimpleNamespace(enable=False),
    )


_L1_DR = {
    "stable_ready_plant": False,
    "startup_physics_material": True,
    "startup_joint_default_pos": True,
    "policy_observation_corruption": False,
}


def test_dr_l1_finalizer_accepts_the_restored_day_one_plant():
    train = _load_train_module()
    env_cfg = _l1_env(
        start_pose_ramp=deepcopy(dict(TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL))
    )
    applied = []
    assert (
        train._apply_action_ball_dr_l1_finalizer(env_cfg, _L1_DR, applied)
        is True
    )
    # 五条 plant 事件必须仍然活着(这是 L1 与 L0 的全部差别)
    for name in train._ACTION_BALL_DR_L1_REQUIRED_EVENTS:
        assert getattr(env_cfg.events, name) is not None
    # 观测腐蚀与执行器延迟不在这一批,保持 L0 的位置
    assert env_cfg.observations.policy.enable_corruption is False
    assert env_cfg.actions.joint_pos.control_step_action_delay_max == 0
    assert len(applied) == 1
    assert TC.ACTION_BALL_DR_L1_IDENTITY in applied[0]
    marker = getattr(env_cfg, train._ACTION_BALL_DR_L1_RUNTIME_ATTR)
    assert marker["identity"] == TC.ACTION_BALL_DR_L1_IDENTITY
    assert marker["startup_offset_delta"]["lower"] == [-0.01] * 31
    assert marker["startup_offset_delta"]["upper"] == [0.01] * 31
    assert marker["motion_reset_noise"]["start_pose_ramp"]["enabled"] is True
    assert train._action_ball_dr_l1_runtime_contract(env_cfg) == marker


def test_dr_l1_refuses_a_disabled_or_retuned_plant_event():
    train = _load_train_module()
    for mutate in (
        lambda cfg: setattr(cfg.events, "add_joint_default_pos", None),
        lambda cfg: cfg.events.add_joint_default_pos.params.update(
            pos_distribution_params=(-0.05, 0.05)
        ),
        lambda cfg: cfg.events.randomize_pd_gains.params.update(
            damping_distribution_params=(0.9, 1.1)
        ),
        lambda cfg: cfg.events.base_com.params.update(
            com_range={"x": (-0.5, 0.5), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
        ),
        lambda cfg: cfg.events.physics_material.params.update(num_buckets=8),
        lambda cfg: setattr(cfg.events, "push_robot", object()),
        lambda cfg: setattr(
            cfg.actions.joint_pos, "control_step_action_delay_max", 2
        ),
        lambda cfg: cfg.commands.motion.pose_range.update(x=(-0.1, 0.1)),
    ):
        env_cfg = _l1_env()
        mutate(env_cfg)
        with pytest.raises(train._OverrideError, match="DR-L1"):
            train._apply_action_ball_dr_l1_finalizer(env_cfg, _L1_DR, [])


def test_dr_l1_requires_stable_ready_plant_false_and_is_disjoint_from_l0():
    train = _load_train_module()
    wrong = dict(_L1_DR, stable_ready_plant=True)
    with pytest.raises(train._OverrideError, match="DR-L1"):
        train._apply_action_ball_dr_l1_finalizer(_l1_env(), wrong, [])
    # L0 的元组绝不能被 L1 认领,反之亦然
    l0 = {
        "stable_ready_plant": True,
        "startup_physics_material": False,
        "startup_joint_default_pos": False,
        "policy_observation_corruption": False,
    }
    assert train._resolve_action_ball_dr_l1_request(l0) is False
    assert train._resolve_action_ball_dr_l0_request(_L1_DR) is False


def test_dr_l0_identity_and_digest_are_untouched_by_the_l1_addition():
    train = _load_train_module()
    payload = TC.action_ball_dr_l0_contract_payload()
    assert payload["identity"] == "action_ball_dr_l0_exact_all_off_v1"
    # L0 的事件槽仍然是"全部缺席",一个都没被 L1 顺手打开
    assert all(value is None for value in payload["event_slots"].values())
    assert payload["startup_offset_delta"]["lower"] == [0.0] * 31
    assert "start_pose_ramp" not in payload["motion_reset_noise"]
    assert train._action_ball_dr_l0_contract_payload() == payload
    assert (
        TC.action_ball_dr_l1_contract_sha256()
        != TC.action_ball_dr_l0_contract_sha256()
    )


def test_dr_l1_digest_moves_with_the_declared_ramp():
    """The ramp is part of the level's identity, not a free-floating knob."""

    without = TC.action_ball_dr_l1_contract_sha256()
    with_ramp = TC.action_ball_dr_l1_contract_sha256(
        start_pose_ramp=TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL
    )
    assert without != with_ramp
    widened = deepcopy(dict(TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL))
    widened["ramp_steps"] = 48000
    assert (
        TC.action_ball_dr_l1_contract_sha256(start_pose_ramp=widened)
        != with_ramp
    )


# --------------------------------------------------------------------------- #
# Hydra 叶子:DR-L1 是新叶,不是对 DR-L0 的原地修改
# --------------------------------------------------------------------------- #
DR_L1_LEAVES = {
    "A": (
        "HOPEPingPongActionBallA211VendorV2N1DRL1Learnability",
        "HOPEPingPongActionBallA211VendorV2N1Learnability",
    ),
    "C": (
        "HOPEPingPongActionBallC211VendorV2N1DRL1Learnability",
        "HOPEPingPongActionBallC211VendorV2N1Learnability",
    ),
}


def _compose(leaf_name: str):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str((ROOT / "cfg").resolve())
    ):
        return hydra.compose(
            config_name="train", overrides=[f"task={leaf_name}"]
        ).task


@pytest.mark.parametrize("side", ("A", "C"))
def test_dr_l1_leaf_is_new_and_leaves_the_dr_l0_leaf_untouched(side: str):
    import yaml

    leaf_name, parent_name = DR_L1_LEAVES[side]
    leaf = yaml.safe_load(
        (ROOT / "cfg/task" / f"{leaf_name}.yaml").read_text(encoding="utf-8")
    )
    assert leaf["name"] == leaf_name
    assert leaf["defaults"] == [f"{parent_name}@_here_", "_self_"]
    assert leaf["domain_rand"] == {
        "stable_ready_plant": False,
        "startup_physics_material": True,
        "startup_joint_default_pos": True,
        "policy_observation_corruption": False,
    }
    # DR-L0 的叶子仍然在,而且仍然是那个 exact all-off 元组
    l0 = yaml.safe_load(
        (
            ROOT
            / "cfg/task"
            / f"{leaf_name.replace('DRL1', 'DRL0')}.yaml"
        ).read_text(encoding="utf-8")
    )
    assert l0["domain_rand"] == {
        "stable_ready_plant": True,
        "startup_physics_material": False,
        "startup_joint_default_pos": False,
        "policy_observation_corruption": False,
    }


@pytest.mark.parametrize("side", ("A", "C"))
def test_dr_l1_leaf_ramp_mirrors_the_code_owned_endpoint_constant(side: str):
    """The YAML may spell the ramp, but the code owns its values."""

    leaf_name, _parent = DR_L1_LEAVES[side]
    task = _compose(leaf_name)
    from omegaconf import OmegaConf

    spelled = OmegaConf.to_container(task.motion.start_pose_ramp, resolve=True)
    assert TC.validate_action_ball_start_pose_ramp(
        spelled, name=leaf_name
    ) == TC.validate_action_ball_start_pose_ramp(
        TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL, name="code"
    )
    # 通用播放速率与旧 hold 时钟必须保持钉死:两者都归任务收据所有
    assert tuple(task.motion.speed_scale_range) == (1.0, 1.0)
    assert tuple(task.motion.hold_steps_range) == (0, 0)
    # 静态复位噪声仍然全零 —— 出生扰动只有斜坡这一条路
    for axis in AXES:
        assert tuple(task.motion.pose_range[axis]) == (0.0, 0.0)
        assert tuple(task.motion.velocity_range[axis]) == (0.0, 0.0)
    assert tuple(task.motion.joint_position_range) == (0.0, 0.0)
    # 复位分支保持 100% 站立(RSI 与 split-ready 的交互未验证)
    assert float(task.motion.stand_start_prob) == 1.0
    assert float(task.motion.post_swing_start_prob) == 0.0


@pytest.mark.parametrize("side", ("A", "C"))
def test_composed_dr_l1_leaf_drives_the_finalizer_to_the_restored_plant(side):
    train = _load_train_module()
    leaf_name, _parent = DR_L1_LEAVES[side]
    task = _compose(leaf_name)
    from omegaconf import OmegaConf

    env_cfg = _l1_env(
        start_pose_ramp=OmegaConf.to_container(
            task.motion.start_pose_ramp, resolve=True
        )
    )
    applied = []
    assert (
        train._apply_action_ball_dr_l1_finalizer(
            env_cfg, task.domain_rand, applied
        )
        is True
    )
    # 同一份 domain_rand 绝不能同时被 L0 认领
    assert train._resolve_action_ball_dr_l0_request(task.domain_rand) is False
    marker = getattr(env_cfg, train._ACTION_BALL_DR_L1_RUNTIME_ATTR)
    assert marker["motion_reset_noise"]["start_pose_ramp"]["ramp_steps"] == 96000


# --------------------------------------------------------------------------- #
# manifest:摘要必须是算出来的,状态必须自陈还缺什么
# --------------------------------------------------------------------------- #
def test_dr_l1_manifest_binds_the_live_resolved_contract_bytes():
    import json

    manifest = json.loads(DR_L1_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "action_ball_211_dr_l1_restored_plant_candidate"
    resolved = manifest["resolved_finalizer_contract"]
    assert resolved["hard_contract_identity"] == TC.ACTION_BALL_DR_L1_IDENTITY
    assert resolved["contract_sha256"] == TC.action_ball_dr_l1_contract_sha256(
        start_pose_ramp=TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL
    )
    ramp = TC.validate_action_ball_start_pose_ramp(
        TC.ACTION_BALL_START_POSE_RAMP_FOUR_CELL, name="four_cell"
    )
    assert manifest["start_pose_ramp"]["payload_sha256"] == (
        TC.action_ball_start_pose_ramp_sha256(ramp)
    )
    assert manifest["start_pose_ramp"]["ramp_steps"] == ramp["ramp_steps"]

    # 恢复的五条必须逐字节等于 day-1 基线常量
    restored = manifest["restored_axes"]
    spec = TC.ACTION_BALL_DR_L1_ACTIVE_EVENTS
    assert restored["add_joint_default_pos"]["pos_distribution_params"] == (
        spec["add_joint_default_pos"]["pos_distribution_params"]
    )
    assert restored["physics_material"]["static_friction_range"] == (
        spec["physics_material"]["static_friction_range"]
    )
    assert restored["randomize_pd_gains"]["damping_distribution_params"] == (
        spec["randomize_pd_gains"]["damping_distribution_params"]
    )
    assert restored["randomize_link_mass"]["mass_distribution_params"] == (
        spec["randomize_link_mass"]["mass_distribution_params"]
    )
    assert restored["base_com"]["com_range"] == spec["base_com"]["com_range"]


def test_dr_l1_manifest_names_the_untouched_dr_l0_control_and_its_blockers():
    """A manifest that claims a launch it cannot make would be the worst outcome."""

    import json

    manifest = json.loads(DR_L1_MANIFEST.read_text(encoding="utf-8"))
    control = manifest["attribution_control"]
    assert control["dr_l0_unchanged"] is True
    assert control["dr_l0_contract_sha256"] == (
        TC.action_ball_dr_l0_contract_sha256()
    )
    l0_manifest = json.loads(DR_L0_MANIFEST.read_text(encoding="utf-8"))
    assert l0_manifest["resolved_finalizer_contract"]["contract_sha256"] == (
        control["dr_l0_contract_sha256"]
    )
    # 还没有 DR-L1 的 launcher lineage,manifest 必须自己说出来
    blockers = manifest["runtime_integration_blockers"]
    assert blockers and any(
        "launcher_lineage_not_materialized" in entry for entry in blockers
    )
    assert manifest["status"] == "BOUND_MECHANISM_LANDED_LAUNCH_LINEAGE_PENDING"
    # 被推迟的那几条必须点名说明,而不是消失
    deferred = manifest["deferred_axes"]
    for key in (
        "action_delay",
        "hold_or_wait_window",
        "motion_speed_scale",
        "observation_corruption",
        "reset_branch_split",
    ):
        assert key in deferred and deferred[key]


# 运行时那两条(MotionCommand 绑定同一条法则 / 有效范围从种子走到终点)住在
# tests/test_reward_flags_mdp.py —— 那里已经有 commands.py 需要的 isaaclab 桩,
# 在这里再造一份只会多一份会漂移的桩。
