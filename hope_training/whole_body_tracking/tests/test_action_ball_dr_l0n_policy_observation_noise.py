"""Fail-closed contract for DR-L0N — the DR-L0 plant with the sensor turned on.

人话:这一档是四格实验的第二根轴(尽调 §22),不是一档更强的 DR。被控对象与 DR-L0
逐字节相同,唯一的差别是 actor 那三路本体感通道叠了噪声:
    joint_pos ±0.01 rad / joint_vel ±0.5 rad·s⁻¹ / base_ang_vel ±0.2 rad·s⁻¹
任务通道(desired contact / incoming ball / 时间)绝不加噪 —— 那会改支撑集,
等于换题而不是换传感器。本文件把这两句话都变成会红的断言。
"""

from __future__ import annotations

import ast
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
HOPE_ENV_CFG = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
    / "config/agibot_a3/hope_env_cfg.py"
)
FOUR_GRID = ROOT / "scripts/action_ball_211_four_grid_contract.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TC = _load_module("action_ball_dr_l0n_training_contract", TRAINING_CONTRACT)
FG = _load_module("action_ball_dr_l0n_four_grid", FOUR_GRID)


def _load_train_module():
    pytest.importorskip("hydra")
    source_root = str((ROOT / "source/whole_body_tracking").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return _load_module("action_ball_dr_l0n_train_under_test", TRAIN)


class _Unoise:
    """Stand-in with the exact class name and fields train.py reads."""

    __name__ = "UniformNoiseCfg"

    def __init__(self, n_min, n_max, operation="add"):
        self.n_min = n_min
        self.n_max = n_max
        self.operation = operation


# 让 type(noise).__name__ 真的等于 "UniformNoiseCfg" —— train.py 是按类名认的。
UniformNoiseCfg = type("UniformNoiseCfg", (_Unoise,), {})


def _term(noise=None):
    return types.SimpleNamespace(func=lambda env: None, noise=noise)


def _policy_group(*, enable_corruption: bool, channels=None):
    """A policy obs group shaped like ActionBall{A,C}211PolicyCfg."""

    if channels is None:
        channels = {
            name: list(bounds)
            for name, bounds in TC.ACTION_BALL_DR_L0N_PROPRIO_NOISE_CHANNELS.items()
        }
    group = types.SimpleNamespace(
        enable_corruption=enable_corruption,
        concatenate_terms=True,
    )
    # 无噪的本体感/教师通道
    for name in (
        "actual_base_pose_lin_vel_world",
        "actions",
        "racket_site_achieved_now_heading",
        "teacher_joint_pos",
        "teacher_joint_vel",
        "racket_site_teacher_now_heading",
        "racket_site_teacher_at_reference_hit_heading",
    ):
        setattr(group, name, _term())
    # 任务通道:永远无噪
    for name in (
        "task_desired_contact_position_heading",
        "task_desired_contact_velocity_heading",
        "task_desired_contact_face_heading",
        "desired_base_xy_world",
        "time_to_contact",
        "time_to_teacher_start",
        "task_valid",
    ):
        setattr(group, name, _term())
    for name, bounds in channels.items():
        setattr(group, name, _term(UniformNoiseCfg(bounds[0], bounds[1])))
    return group


def _finalizer_env(*, enable_corruption=True, channels=None):
    sentinel = object()
    zero_ranges = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    return types.SimpleNamespace(
        events=types.SimpleNamespace(
            physics_material=sentinel,
            add_joint_default_pos=sentinel,
            base_com=None,
            randomize_link_mass=None,
            randomize_pd_gains=None,
            push_robot=None,
            force_push=None,
            force_push_sweep=None,
            combined_push=None,
            combined_push_sweep=None,
        ),
        observations=types.SimpleNamespace(
            policy=_policy_group(
                enable_corruption=enable_corruption, channels=channels
            ),
            # 非对称 actor-critic:critic 看的是干净的特权观测。
            critic=types.SimpleNamespace(
                enable_corruption=False, concatenate_terms=True
            ),
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
                pose_range=deepcopy(zero_ranges),
                velocity_range=deepcopy(zero_ranges),
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


L0N_DR = {
    "stable_ready_plant": True,
    "startup_physics_material": False,
    "startup_joint_default_pos": False,
    "policy_observation_corruption": True,
}
L0_DR = dict(L0N_DR, policy_observation_corruption=False)


# --------------------------------------------------------------------------- #
# 1. 这一档只准是"DR-L0 + 传感器"
# --------------------------------------------------------------------------- #
def test_dr_l0n_payload_is_dr_l0_plus_exactly_the_declared_sensor():
    l0 = TC.action_ball_dr_l0_contract_payload()
    l0n = TC.action_ball_dr_l0n_contract_payload()
    differing = sorted(key for key in set(l0) | set(l0n) if l0.get(key) != l0n.get(key))
    assert differing == sorted(TC.ACTION_BALL_DR_L0N_DECLARED_DIFFERENCES)
    # plant 那一半逐字节相同 —— 这是"只换传感器"的全部含义。
    for key in ("event_slots", "motion_reset_noise", "target_transport_noise",
                "control_step_action_delay", "push_flags", "startup_offset_delta"):
        assert l0[key] == l0n[key]
    assert l0["policy_observation_corruption"] is False
    assert l0n["policy_observation_corruption"] is True
    assert l0n["identity"] == TC.ACTION_BALL_DR_L0N_IDENTITY
    noise = l0n["proprioceptive_observation_noise"]
    assert noise["channels"] == TC.ACTION_BALL_DR_L0N_PROPRIO_NOISE_CHANNELS
    assert noise["operation"] == "add"
    assert noise["task_channel_observation_noise"] is False
    assert noise["unlisted_policy_channel_noise_forbidden"] is True


def test_dr_l0_identity_and_digest_are_untouched_by_this_batch():
    """DR-L0 是归因对照,它的字节不许因为新开一档而动。"""

    assert (
        TC.action_ball_dr_l0_contract_sha256()
        == "fd22321e3371a81b1f979dc4ecfb79e76c44c8d7734fffb2eebdc92620cb7ed9"
    )
    assert (
        TC.action_ball_dr_l0_contract_payload()["identity"]
        == "action_ball_dr_l0_exact_all_off_v1"
    )
    assert (
        TC.action_ball_dr_l0n_contract_sha256()
        != TC.action_ball_dr_l0_contract_sha256()
    )


def test_declared_channels_match_the_live_a211_and_c211_policy_groups():
    """通道表不是新编的:必须逐字等于 hope_env_cfg 里已经写着的 Unoise。

    用 AST 静态读源码,不需要 Isaac。少一路、多一路、改幅度,这里都会红。
    """

    tree = ast.parse(HOPE_ENV_CFG.read_text(encoding="utf-8"), filename=str(HOPE_ENV_CFG))
    wanted = {
        "ActionBallA211PolicyCfg",
        "ActionBallC211PolicyCfg",
    }
    seen = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in wanted:
            continue
        noised = {}
        unnoised = []
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            call = stmt.value
            if not isinstance(call, ast.Call):
                continue
            noise_kw = next(
                (kw for kw in call.keywords if kw.arg == "noise"), None
            )
            if noise_kw is None:
                unnoised.append(target.id)
                continue
            bounds = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in noise_kw.value.keywords
                if kw.arg in ("n_min", "n_max")
            }
            noised[target.id] = [bounds["n_min"], bounds["n_max"]]
        seen[node.name] = (noised, unnoised)

    assert set(seen) == wanted
    for name, (noised, unnoised) in seen.items():
        assert noised == TC.ACTION_BALL_DR_L0N_PROPRIO_NOISE_CHANNELS, name
        # 任务通道必须在"无噪"那一堆里出现,不能只是"碰巧没写 noise="。
        assert "time_to_contact" in unnoised, name
        assert "desired_base_xy_world" in unnoised, name


def test_four_grid_mirror_matches_the_runtime_contract():
    assert (
        FG.PROPRIOCEPTIVE_OBSERVATION_NOISE_CHANNELS
        == TC.ACTION_BALL_DR_L0N_PROPRIO_NOISE_CHANNELS
    )
    assert FG.DR_LEVEL_IDENTITY_OBS_NOISE_ON == TC.ACTION_BALL_DR_L0N_IDENTITY
    assert (
        FG.DR_LEVEL_IDENTITY_OBS_NOISE_OFF
        == TC.action_ball_dr_l0_contract_payload()["identity"]
    )


# --------------------------------------------------------------------------- #
# 2. finalizer:达成的最终状态
# --------------------------------------------------------------------------- #
def test_l0n_finalizer_reaches_the_l0_plant_with_the_sensor_on():
    train = _load_train_module()
    assert train._ACTION_BALL_DR_L0N_NOISE_OPERATION == (
        TC.ACTION_BALL_DR_L0N_NOISE_OPERATION
    )
    env_cfg = _finalizer_env(enable_corruption=False)
    applied = []
    assert train._apply_action_ball_dr_l0n_finalizer(env_cfg, L0N_DR, applied) is True
    assert env_cfg.events.physics_material is None
    assert env_cfg.events.add_joint_default_pos is None
    assert env_cfg.events.base_com is None
    assert env_cfg.events.randomize_link_mass is None
    assert env_cfg.events.randomize_pd_gains is None
    assert env_cfg.observations.policy.enable_corruption is True
    assert len(applied) == 1
    assert TC.ACTION_BALL_DR_L0N_IDENTITY in applied[0]
    assert getattr(env_cfg, train._ACTION_BALL_DR_L0N_RUNTIME_ATTR) == (
        TC.action_ball_dr_l0n_contract_payload()
    )
    # L0 的 marker 绝不能同时出现。
    assert not hasattr(env_cfg, train._ACTION_BALL_DR_L0_RUNTIME_ATTR)


def test_l0_and_l0n_are_mutually_exclusive_resolvers():
    train = _load_train_module()
    assert train._resolve_action_ball_dr_l0_request(L0N_DR) is False
    assert train._resolve_action_ball_dr_l0n_request(L0N_DR) is True
    assert train._resolve_action_ball_dr_l0_request(L0_DR) is True
    assert train._resolve_action_ball_dr_l0n_request(L0_DR) is False
    # L0 的 finalizer 见到 L0N 元组必须不认领,而不是当成"混合的 L0"报错。
    env_cfg = _finalizer_env()
    assert train._apply_action_ball_dr_l0_finalizer(env_cfg, L0N_DR, []) is False


def test_l0n_requires_the_stable_ready_plant_spelling():
    train = _load_train_module()
    mixed = dict(L0N_DR, stable_ready_plant=False)
    with pytest.raises(train._OverrideError, match="DR-L0N"):
        train._resolve_action_ball_dr_l0n_request(mixed)


@pytest.mark.parametrize(
    "present",
    (
        {"stable_ready_plant": True, "policy_observation_corruption": True},
        {
            "stable_ready_plant": True,
            "startup_physics_material": False,
            "policy_observation_corruption": True,
        },
    ),
)
def test_half_a_tuple_is_still_refused_not_read_as_l0n(present):
    """只写一半仍然是未注册的混合 plant,必须由 L0 解析器当场拒。"""

    train = _load_train_module()
    assert train._resolve_action_ball_dr_l0n_request(present) is False
    with pytest.raises(train._OverrideError, match="ActionBall DR-L0"):
        train._resolve_action_ball_dr_l0_request(present)


# --------------------------------------------------------------------------- #
# 3. 通道表的 fail-closed:多一路 / 少一路 / 改幅度 / 任务通道带噪
# --------------------------------------------------------------------------- #
def _mutated_channels(**changes):
    channels = {
        name: list(bounds)
        for name, bounds in TC.ACTION_BALL_DR_L0N_PROPRIO_NOISE_CHANNELS.items()
    }
    channels.update(changes)
    return channels


@pytest.mark.parametrize(
    "mutation",
    (
        "widen_joint_vel",
        "drop_joint_pos",
        "noise_a_task_channel",
        "noise_an_unlisted_proprio_channel",
        "wrong_operation",
        "corruption_forced_off",
        "critic_also_noised",
    ),
)
def test_l0n_finalizer_refuses_any_other_noise_layout(mutation: str):
    train = _load_train_module()
    if mutation == "widen_joint_vel":
        env_cfg = _finalizer_env(channels=_mutated_channels(joint_vel=[-5.0, 5.0]))
    elif mutation == "drop_joint_pos":
        channels = _mutated_channels()
        channels.pop("joint_pos")
        env_cfg = _finalizer_env(channels=channels)
    elif mutation == "noise_a_task_channel":
        env_cfg = _finalizer_env()
        env_cfg.observations.policy.time_to_contact = _term(
            UniformNoiseCfg(-0.01, 0.01)
        )
    elif mutation == "noise_an_unlisted_proprio_channel":
        env_cfg = _finalizer_env()
        env_cfg.observations.policy.actual_base_pose_lin_vel_world = _term(
            UniformNoiseCfg(-0.1, 0.1)
        )
    elif mutation == "wrong_operation":
        env_cfg = _finalizer_env()
        env_cfg.observations.policy.joint_pos = _term(
            UniformNoiseCfg(-0.01, 0.01, operation="abs")
        )
    elif mutation == "critic_also_noised":
        # payload 明写 critic_group_corruption=False;critic 也带噪 = 换成"两边都退化"。
        env_cfg = _finalizer_env()
        env_cfg.observations.critic.enable_corruption = True
    else:
        env_cfg = _finalizer_env()

        class _Stubborn(types.SimpleNamespace):
            """腐蚀开关写不进去 —— 模拟"最终状态没达成"。"""

            def __setattr__(self, name, value):
                if name == "enable_corruption":
                    value = False
                super().__setattr__(name, value)

        policy = env_cfg.observations.policy
        stubborn = _Stubborn(**vars(policy))
        stubborn.enable_corruption = False
        env_cfg.observations.policy = stubborn

    with pytest.raises(train._OverrideError, match="DR-L0N"):
        train._apply_action_ball_dr_l0n_finalizer(env_cfg, L0N_DR, [])


@pytest.mark.parametrize(
    "mutation", ("push", "reset_state", "target_transport", "action_delay", "plant_event")
)
def test_l0n_finalizer_still_refuses_every_non_l0_plant_axis(mutation: str):
    """L0N 只放开传感器;plant 那一半的 fail-closed 一条都不许松。"""

    train = _load_train_module()
    env_cfg = _finalizer_env()
    if mutation == "push":
        env_cfg.events.push_robot = object()
        env_cfg.push.enable = True
    elif mutation == "reset_state":
        env_cfg.commands.motion.pose_range["x"] = (-0.1, 0.1)
    elif mutation == "target_transport":
        env_cfg.commands.racket_target.target_dropout_prob = 0.01
    elif mutation == "action_delay":
        env_cfg.actions.joint_pos.control_step_action_delay_max = 1
    else:
        env_cfg.events.combined_push = object()
    with pytest.raises(train._OverrideError, match="DR-L0N"):
        train._apply_action_ball_dr_l0n_finalizer(env_cfg, L0N_DR, [])


# --------------------------------------------------------------------------- #
# 4. 运行期再开一次:marker 不算证据
# --------------------------------------------------------------------------- #
def _zero_bootstrap(train):
    zero = TC.action_ball_dr_l0n_contract_payload()["startup_offset_delta"]
    return {
        "decoder": {
            "startup_offset_delta_source": zero["source"],
            "startup_offset_delta_lower": list(zero["lower"]),
            "startup_offset_delta_upper": list(zero["upper"]),
            "startup_offset_delta_identity": {
                "startup_offset_delta": list(zero["lower"])
            },
        }
    }


def test_l0n_runtime_contract_reopens_the_state_and_requires_a_zero_decoder():
    train = _load_train_module()
    env_cfg = _finalizer_env()
    assert train._apply_action_ball_dr_l0n_finalizer(env_cfg, L0N_DR, []) is True
    contract = train._action_ball_dr_l0n_runtime_contract(
        env_cfg, policy_bootstrap=_zero_bootstrap(train)
    )
    assert contract == TC.action_ball_dr_l0n_contract_payload()

    # 事后把腐蚀关掉 -> 运行期复核必须抓到。
    env_cfg.observations.policy.enable_corruption = False
    with pytest.raises(RuntimeError, match="DR-L0N runtime state drifted"):
        train._action_ball_dr_l0n_runtime_contract(
            env_cfg, policy_bootstrap=_zero_bootstrap(train)
        )

    # 没有 fresh bootstrap -> 拒。
    env_cfg.observations.policy.enable_corruption = True
    with pytest.raises(RuntimeError, match="fresh policy"):
        train._action_ball_dr_l0n_runtime_contract(env_cfg, policy_bootstrap=None)


def test_l0n_runtime_contract_is_absent_without_the_marker():
    train = _load_train_module()
    env_cfg = _finalizer_env()
    assert (
        train._action_ball_dr_l0n_runtime_contract(env_cfg, policy_bootstrap=None)
        is None
    )
