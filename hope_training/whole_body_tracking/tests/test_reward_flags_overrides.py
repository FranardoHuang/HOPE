"""reward-flags-0709 — train.py override-translation unit tests (NO Isaac imports).

Covers the eight default-OFF flag sets of docs/research/reward_staged_design_2026-07-08.md plus
the new task.rewards fail-loud whitelist, at the TRANSLATION layer (_apply_task_overrides):

* task.rewards unknown-key fail-loud (_REWARD_KEYS) + regression that every rewards/motion/racket/
  terminations key in the real cfg/task/*.yaml files is whitelisted (so the new check cannot brick
  an existing task).
* V1  rewards.free_wrist_vel_mimic       — wrist dropped from motion_body_lin_vel only.
* V2  rewards.motion_scale_in_window     — window_scale params wired onto the (non-None) motion terms.
* 1c  racket.strike_window_pos_s/wide_s  — split-window cfg fields set.
* C2a rewards.face_gate_by_pos+radius    — pos_gate_radius param on racket_velocity/normal;
                                           half-configured gate raises either way.
* B2  rewards.racket_guidance_weight     — weight set; positive weight raises.
* R-a task.actor_leg_ref_mask            — actor command obs func swapped; critic untouched.
* R-c motion.rsi_skip_settle_frames / rsi_hold_root_stand_z — MotionCommandCfg fields set.
* R-b terminations.envelope_as_penalty(+envelope_penalty_weight) — anchor_pos/ee_body_pos removed,
      tracking_envelope weight set (default -1.0), track_envelope_violation switched on;
      weight-without-flag and weight>=0 raise; absolute terminations untouched.
* R9  terminations.anchor_pos_off / ee_upper_only (lower-body-free ablation, franco 2026-07-08) —
      anchor_pos_off removes ONLY the torso-z leash termination; ee_upper_only narrows the
      ee_body_pos body list to the wrists (ankles freed); absolutes + anchor_ori stay; either
      flag combined with envelope_as_penalty raises; an unexpected body list raises.
* default path: a DeployParity-like task node without any new key leaves every new attr untouched.
* YAML-null 删参 (jiayi 8ee2e82a -> main) — rewards 下显式 `some_key: null` 把继承来的
      term.params 项 pop 掉并记账;没写的键、表外键、已不存在的参数一律不动。
* B2 rewards.reward_pack — v2 蓝图一键成套换装(reward_redesign_20260725 §3/§3.5):键控注入
      走现有翻译层、direct-cfg 项逐条记账,包先展开显式键后写后赢;motion_scale_in_window
      与包冲突、未知包值、缺 adaptive_sigma、血统缺项全部 fail-loud。2026-07-25 Franco 裁定
      默认翻转:缺席 = 按 v2 展开(applied 记 defaulted 标记);显式 v1 = legacy 兜底 flag,
      逐字节不变。本文件所有 legacy 翻译断言经 _apply 显式钉 v1 维持原语义;默认路径(v2
      展开)在 JOB1 区块用 _apply_default 单独测。
* B2 task.venue_profile — 场地档案一键展开(Wave-1 utils/venue_profile.py loader):
      mocap/transport 注入 task.racket 走翻译层,physics 直写 events params;显式键赢;
      applied 标记带档案名+sha 前缀;未知档案/缺 events 旋钮 fail-loud;缺席 = 逐字节 no-op。

train.py is imported directly (its top-level imports are hydra/omegaconf only — no Isaac); the
env cfg is a plain-namespace fake, exactly the level _apply_task_overrides operates on.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py -q
"""

from __future__ import annotations

import os
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
CFG_TASK_DIR = os.path.abspath(os.path.join(HERE, "..", "cfg", "task"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import train as train_mod  # noqa: E402  (hydra/omegaconf only at import time)


# --------------------------------------------------------------------------------------------- #
# fakes: the minimal env-cfg surface _apply_task_overrides touches
# --------------------------------------------------------------------------------------------- #
class _Term:
    """RewTerm/ObsTerm/DoneTerm stand-in: weight + params dict + func slot."""

    def __init__(self, weight=1.0, params=None, func="orig_func", body_names=None):
        self.weight = weight
        self.params = dict(params) if params is not None else {}
        if body_names is not None:
            self.params["body_names"] = list(body_names)
        self.func = func


class _NS(types.SimpleNamespace):
    pass


_WRIST = "right_wrist_yaw_Link"
_LEFT_NON_STRIKING = [
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
]
_RIGHT_STRIKING = ["right_shoulder_roll_Link", "right_elbow_Link", _WRIST]
_UPPER = ["torso_Link", *_LEFT_NON_STRIKING, *_RIGHT_STRIKING]


def _make_env_cfg(anchor_pos_none=True):
    """DeployParity-shaped fake env cfg (motion_global_anchor_pos removed, like the real cfg)."""
    rewards = _NS(
        racket_position=_Term(weight=14.0, params={"std": 0.2}),
        racket_position_coarse=_Term(weight=0.0, params={"std": 0.30}),
        racket_velocity=_Term(weight=10.0, params={"std": 1.0}),
        racket_normal=_Term(weight=5.0, params={"std": 0.30}),
        hold_ready=_Term(weight=0.0, params={"std": 1.5, "reach": 0.2, "reach_mode": "station"}),
        post_strike_brake=_Term(weight=0.0, params={"std": 0.5}),
        hold_heading=_Term(weight=0.0, params={"std": 0.6}),
        foot_orientation=_Term(weight=0.0, params={"hold_gate": False}),
        base_decel_activation_probe=_Term(
            weight=0.0,
            params={"command_name": "racket_target", "v_gain": 2.0, "v_max": 1.6, "std": 0.4},
        ),
        base_decel=_Term(
            weight=0.0,
            params={"command_name": "racket_target", "v_gain": 2.0, "v_max": 1.6, "std": 0.4},
        ),
        joint_velocity_limit_hinge=_Term(
            weight=0.0,
            params={
                "asset_cfg": _NS(name="robot", joint_ids=slice(None)),
                "margin": 0.85,
                "expected_joint_count": 31,
            },
        ),
        joint_velocity_limit_hinge_probe=_Term(
            weight=0.0,
            params={
                "asset_cfg": _NS(name="robot", joint_ids=slice(None)),
                "margin": 0.85,
                "expected_joint_count": 31,
            },
        ),
        processed_qdes_slew_hinge=_Term(
            weight=0.0,
            params={
                "action_name": "joint_pos",
                "command_name": "racket_target",
                "margin": 0.85,
                "recovery_start_s": 0.20,
                "recovery_end_s": 1.55,
            },
        ),
        processed_qdes_slew_hinge_probe=_Term(
            weight=0.0,
            params={
                "action_name": "joint_pos",
                "command_name": "racket_target",
                "margin": 0.85,
                "recovery_start_s": 0.20,
                "recovery_end_s": 1.55,
            },
        ),
        lower_body_pose_imitation=_Term(
            weight=0.0,
            params={
                "racket_command_name": "racket_target",
                "motion_command_name": "motion",
                "std": 0.35,
                "support_pre_s": 0.30,
                "support_post_s": 0.40,
            },
        ),
        lower_body_pose_imitation_probe=_Term(
            weight=0.0,
            params={
                "racket_command_name": "racket_target",
                "motion_command_name": "motion",
                "std": 0.35,
                "support_pre_s": 0.30,
                "support_post_s": 0.40,
            },
        ),
        lower_body_stability_bundle=_Term(
            weight=0.0,
            params={
                "racket_command_name": "racket_target",
                "motion_command_name": "motion",
                "min_stance_width_m": 0.22,
                "stance_scale_m": 0.05,
                "leg_velocity_margin_radps": 1.0,
                "leg_velocity_scale_radps": 0.5,
                "support_pre_s": 0.30,
                "support_post_s": 0.40,
            },
        ),
        lower_body_stability_bundle_probe=_Term(
            weight=0.0,
            params={
                "racket_command_name": "racket_target",
                "motion_command_name": "motion",
                "min_stance_width_m": 0.22,
                "stance_scale_m": 0.05,
                "leg_velocity_margin_radps": 1.0,
                "leg_velocity_scale_radps": 0.5,
                "support_pre_s": 0.30,
                "support_post_s": 0.40,
            },
        ),
        racket_guidance=_Term(weight=0.0, params={"command_name": "racket_target", "d_max": 0.5}),
        racket_face_guidance=_Term(weight=0.0, params={"command_name": "racket_target", "theta_max": 1.5707963}),
        racket_face_conditional_guidance=_Term(
            weight=0.0,
            params={
                "command_name": "racket_target",
                "theta_free": 0.262,
                "theta_max": 3.141592653589793,
                "pos_full": 0.075,
                "pos_zero": 0.095,
                "vel_full": 0.5,
                "vel_zero": 1.0,
            },
        ),
        tracking_envelope=_Term(weight=0.0, params={"command_name": "motion", "threshold": 0.25}),
        # v2 奖励包(reward_pack=v2)会动的 direct-cfg 项:窗内站稳四件套 + 反作弊小税 +
        # 税型/收入型站正 + PACE 单条击球稳定(权重取真 cfg 的现役值,证明包真的改了它们)。
        foot_slip_sq=_Term(weight=-1.0, params={"command_name": "racket_target"}),
        foot_drag=_Term(weight=-0.5, params={"command_name": "racket_target"}),
        # 真 cfg 默认 0(07-26 前包也不动它=科学臂静默跑丢);v2 包现在落 -0.003
        foot_soft_landing=_Term(weight=0.0, params={"force_threshold_n": 300.0}),
        arm_overreach=_Term(weight=-0.5, params={"command_name": "racket_target"}),
        prestrike_waist_twist=_Term(weight=-1.0, params={"command_name": "racket_target"}),
        prestrike_upright=_Term(weight=-1.0, params={"command_name": "racket_target"}),
        strike_upright=_Term(weight=-2.0, params={"command_name": "racket_target"}),
        strike_ang_vel=_Term(weight=-0.5, params={"command_name": "racket_target"}),
        strike_foot_vel=_Term(weight=-0.5, params={"command_name": "racket_target"}),
        strike_vbob=_Term(weight=-1.0, params={"command_name": "racket_target"}),
        upright=_Term(weight=-1.0),
        upright_exp=_Term(weight=0.0, params={"std": 0.4472135954999579}),
        hit_unstable_support=_Term(weight=0.0, params={"command_name": "racket_target"}),
        # v2 击中层 direct-cfg 项(§3.5):三核乘法加强层(真 cfg 现役 5.0)+ one-shot 击中
        # 大奖(VirtualBall 谱系 weight=0 待命)。
        racket_strike_success=_Term(
            weight=5.0,
            params={
                "command_name": "racket_target",
                "std_pos": 0.075,
                "std_vel": 0.5,
                "std_normal": 0.262,
            },
        ),
        strike_capture_bonus=_Term(weight=0.0, params={"command_name": "racket_target"}),
        action_rate_clamped=_Term(weight=0.0, params={"value_clamp": 9.0}),
        death_penalty=_Term(weight=0.0),
        action_acc_l2=_Term(weight=0.0, params={"action_name": "joint_pos"}),
        # v2.1(Franco 07-25 裁定):上台组扛"击中+打好"主奖;假 cfg 按 VirtualBall 谱系默认声明
        virtual_pass_net=_Term(weight=20.0, params={"command_name": "racket_target"}),
        virtual_landing=_Term(weight=30.0, params={"command_name": "racket_target"}),
        virtual_spin=_Term(weight=5.0, params={"command_name": "racket_target"}),
        motion_global_anchor_pos=None if anchor_pos_none else _Term(weight=0.5, params={"std": 0.3}),
        motion_global_anchor_ori=_Term(weight=0.5, params={"std": 0.4}),
        motion_body_pos=_Term(weight=1.0, params={"std": 0.3}, body_names=_UPPER),
        motion_body_ori=_Term(weight=1.0, params={"std": 0.4}, body_names=_UPPER),
        motion_body_lin_vel=_Term(weight=1.0, params={"std": 1.0}, body_names=_UPPER),
        motion_body_ang_vel=_Term(weight=1.0, params={"std": 3.14}, body_names=_UPPER),
        motion_racket_position=_Term(
            weight=0.0,
            params={"std": 0.70, "scale_in_strike_window": 1.0},
        ),
        motion_racket_velocity=_Term(
            weight=0.0,
            params={"std": 4.0, "scale_in_strike_window": 1.0},
        ),
        motion_racket_normal=_Term(
            weight=0.0,
            params={"std": 3.141592653589793, "scale_in_strike_window": 1.0},
        ),
        motion_racket_long_axis=_Term(
            weight=0.0,
            params={"std": 1.0, "scale_in_strike_window": 1.0},
        ),
        action_rate_l2=_Term(weight=-0.1),
        joint_torques=_Term(weight=-3e-5),
    )
    racket_target = _NS(
        strike_phase=0.46,
        strike_window_s=0.12,
        strike_window_pos_s=None,
        strike_window_wide_s=None,
        strike_success_pos_thresh=0.075,
        track_envelope_violation=False,
        target_mode="uniform",
        adaptive_sigma=False,
        adaptive_sigma_normal=False,
        adaptive_sigma_monotonic=False,
        target_delay_steps=0,
        target_delay_tts_mode="live",
        target_noise_white=0.0,
        target_noise_ar1_sigma=0.0,
        target_noise_ar1_rho=0.0,
        target_dropout_prob=0.0,
        midswing_resample_prob=0.0,
        midswing_resample_tts_floor=0.3,
        question_bank="",
        question_bank_allow_legacy=False,
        face_command=False,
        face_command_pairing="shared_plus_y",
    )
    motion = _NS(
        wrap_teleport=False,
        stand_start_prob=0.25,
        hold_steps_range=(0, 100),
        stand_start_min_hold=25,
        stand_start_yaw_range=(0.0, 0.0),
        post_swing_start_prob=0.0,
        post_swing_buffer_size=4096,
        post_swing_min_fill=256,
        post_swing_min_hold=25,
        post_swing_teacher_receipt="",
        post_swing_teacher_receipt_sha256="",
        post_swing_teacher_retry_authorization="",
        post_swing_teacher_retry_authorization_sha256="",
        post_swing_teacher_root_linear_velocity_limit_mps=0.0,
        post_swing_teacher_root_angular_velocity_limit_radps=0.0,
        post_swing_require_ready_at_init=False,
        post_swing_fail_fast_first_reset=False,
        post_swing_first_reset_min_adopted_count=1,
        post_swing_first_reset_min_adopted_fraction=0.0,
        post_swing_first_reset_selection_tolerance=1.0,
        post_swing_first_reset_require_readback=False,
        post_swing_capture_output_dir="",
        post_swing_capture_target_count=0,
        clip_switch_prob=0.0,
        speed_scale_range=(1.0, 1.0),
        speed_scale_per_clip=None,
        rsi_skip_settle_frames=0,
        rsi_hold_root_stand_z=False,
        allow_legacy_link_origin_velocity=False,
    )
    observations = _NS(
        policy=_NS(
            command=_Term(func="generated_commands", params={"command_name": "motion"}),
            time_to_strike=_Term(func="live_time_to_strike", params={"command_name": "racket_target"}),
        ),
        critic=_NS(
            command=_Term(func="generated_commands", params={"command_name": "motion"}),
            time_to_strike=_Term(func="live_time_to_strike", params={"command_name": "racket_target"}),
        ),
    )
    terminations = _NS(
        time_out="TIME_OUT",
        anchor_pos="ANCHOR_POS_TERM",
        anchor_ori="ANCHOR_ORI_TERM",
        ee_body_pos="EE_BODY_POS_TERM",
        base_fell_tilt="BASE_FELL_TILT_TERM",
        base_too_low="BASE_TOO_LOW_TERM",
    )
    actuators = {
        "legs": _NS(friction={"hip": 1.2, "knee": 2.4}),
        "arms": _NS(friction=0.1),
    }
    # venue_profile 物理 section 的落点(初值故意与 franco_rig 档案不同,证明档案真的落了地)。
    events = _NS(
        physics_material=_Term(
            params={
                "static_friction_range": (1.0, 1.0),
                "dynamic_friction_range": (1.0, 1.0),
                "restitution_range": (0.9, 0.9),
            }
        ),
        randomize_link_mass=_Term(
            params={"mass_distribution_params": (1.0, 1.0), "operation": "scale"}
        ),
    )
    return _NS(
        rewards=rewards,
        commands=_NS(motion=motion, racket_target=racket_target),
        observations=observations,
        terminations=terminations,
        events=events,
        scene=_NS(env_spacing=2.5, robot=_NS(actuators=actuators)),
        episode_length_s=10.0,
        sim=_NS(dt=0.005),
        decimation=4,
        actions=_NS(
            joint_pos=_NS(
                clamp=False,
                pre_apply_limit_guard=True,
                pre_apply_guard_margin_fraction=0.05,
                pre_apply_guard_brake_mode="velocity_horizon_v1",
                project_finite_preclamp_qdes_without_termination=True,
                physx_control_position_limit_inset_fraction=0.0,
                control_step_action_delay_min=0,
                control_step_action_delay_max=0,
            )
        ),
    )


# 2026-07-25 默认翻转(Franco 裁定):reward_pack 缺席 = v2 全套展开。本文件绝大多数用例测
# 的是【翻译层 legacy 行为】,断言全部建立在"没写 = 不动"的 v1 基线上——所以 _apply 把没写
# reward_pack 的任务节点显式钉成 v1(兜底 flag),再滤掉那一条 v1 记账行,原断言原样成立。
# 默认路径(v2 展开)的语义由 JOB1 区块用 _apply_default 单独测。
_V1_MARKER = "rewards.reward_pack=v1 (legacy baseline)"


def _pin_reward_pack_v1(task):
    """任务节点没写 reward_pack 就显式钉 v1;写了(v1/v2/非法值)原样放行。dict 与
    OmegaConf 节点都支持这里的 __setitem__。"""
    if train_mod._get(task, "rewards") is None:
        task["rewards"] = {"reward_pack": "v1"}
    elif train_mod._get(task["rewards"], "reward_pack") is None:
        task["rewards"]["reward_pack"] = "v1"
    return task


def _apply_legacy_v1(env_cfg, task):
    """邻居测试套件共用:钉 v1 + 滤 v1 记账行,返回 applied(legacy 断言零改动)。"""
    task = _pin_reward_pack_v1(task)
    applied = train_mod._apply_task_overrides(env_cfg, task, clip_name=None)
    return [marker for marker in applied if marker != _V1_MARKER]


def _apply(task, env_cfg=None):
    env_cfg = env_cfg if env_cfg is not None else _make_env_cfg()
    applied = _apply_legacy_v1(env_cfg, task)
    return env_cfg, applied


def _apply_default(task, env_cfg=None):
    """默认路径入口:不钉 v1,reward_pack 缺席走真实默认(v2 全套展开)。"""
    env_cfg = env_cfg if env_cfg is not None else _make_env_cfg()
    applied = train_mod._apply_task_overrides(env_cfg, task, clip_name=None)
    return env_cfg, applied


# --------------------------------------------------------------------------------------------- #
# fail-loud whitelist
# --------------------------------------------------------------------------------------------- #
def test_empty_task_applies_nothing():
    env_cfg, applied = _apply({})
    assert applied == []
    assert env_cfg.rewards.racket_guidance.weight == 0.0
    assert env_cfg.rewards.racket_face_conditional_guidance.weight == 0.0
    assert env_cfg.commands.racket_target.strike_window_pos_s is None
    assert not hasattr(env_cfg, train_mod._LATERAL_TRAINING_SPEC_ATTR)


def test_table_attribution_override_uses_fail_loud_explicit_boolean_parser():
    name = "task.table_contact_attribution_diagnostic"
    assert train_mod._as_explicit_bool(True, name) is True
    assert train_mod._as_explicit_bool("true", name) is True
    assert train_mod._as_explicit_bool(False, name) is False
    assert train_mod._as_explicit_bool("false", name) is False
    with pytest.raises(train_mod._OverrideError, match=name):
        train_mod._as_explicit_bool("tru", name)


def _task_first_table_params(*, enabled=False):
    params = {key: None for key in train_mod._TASK_FIRST_TABLE_PARAM_KEYS}
    params["attribution_diagnostic"] = enabled
    params["attribution_command_name"] = "racket_target"
    return params


def test_task_first_table_attribution_exact_params_accept_default_and_probe():
    train_mod._validate_task_first_table_attribution_params(
        _task_first_table_params(enabled=False),
        _NS(table_contact_attribution_diagnostic=False),
        action_ball=False,
    )
    train_mod._validate_task_first_table_attribution_params(
        _task_first_table_params(enabled=True),
        _NS(table_contact_attribution_diagnostic=True),
        action_ball=True,
    )


@pytest.mark.parametrize(
    "mutation,env_value,action_ball,match",
    [
        (lambda params: params.pop("attribution_diagnostic"), False, True, "requires exactly"),
        (lambda params: params.pop("attribution_command_name"), False, True, "requires exactly"),
        (lambda params: params.__setitem__("attribution_diagnostic_extra", False), False, True, "requires exactly"),
        (lambda params: params.__setitem__("attribution_diagnostic", "false"), False, True, "diagnostic attribution"),
        (lambda params: params.__setitem__("attribution_command_name", "motion"), False, True, "diagnostic attribution"),
        (lambda params: None, "false", True, "must be an exact boolean"),
        (lambda params: params.__setitem__("attribution_diagnostic", True), False, True, "diagnostic attribution"),
        (lambda params: params.__setitem__("attribution_diagnostic", True), True, False, "diagnostic attribution"),
    ],
)
def test_task_first_table_attribution_exact_params_reject_drift(
    mutation, env_value, action_ball, match
):
    params = _task_first_table_params(enabled=False)
    mutation(params)
    with pytest.raises(train_mod._OverrideError, match=match):
        train_mod._validate_task_first_table_attribution_params(
            params,
            _NS(table_contact_attribution_diagnostic=env_value),
            action_ball=action_ball,
        )


def test_racket_position_coarse_override_is_independent_and_default_off():
    env_cfg, applied = _apply(
        {
            "rewards": {
                "racket_position_coarse_weight": 1.0,
                "racket_position_coarse_std": 0.30,
            }
        }
    )
    assert env_cfg.rewards.racket_position.weight == pytest.approx(14.0)
    assert env_cfg.rewards.racket_position.params["std"] == pytest.approx(0.2)
    assert env_cfg.rewards.racket_position_coarse.weight == pytest.approx(1.0)
    assert env_cfg.rewards.racket_position_coarse.params["std"] == pytest.approx(0.30)
    assert "rewards.racket_position_coarse.weight=1.0" in applied
    assert "rewards.racket_position_coarse.params.std=0.3" in applied


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), "bad", None])
def test_racket_position_coarse_rejects_invalid_weight(weight):
    with pytest.raises(train_mod._OverrideError, match="coarse.*(?:>= 0|non-null)"):
        _apply({"rewards": {"racket_position_coarse_weight": weight}})


@pytest.mark.parametrize("std", [0.0, -0.1, float("nan"), float("inf"), "bad", None])
def test_racket_position_coarse_rejects_invalid_std(std):
    with pytest.raises(train_mod._OverrideError, match="coarse.*(?:> 0|non-null)"):
        _apply({"rewards": {"racket_position_coarse_std": std}})


def test_arm_torque_request_is_not_claimed_applied_before_backend_resolution():
    env_cfg = _make_env_cfg()
    env_cfg.rewards.arm_torque_saturation = _Term(weight=0.0)
    env_cfg, applied = _apply(
        {"rewards": {"arm_torque_saturation_weight": -0.5}},
        env_cfg=env_cfg,
    )

    assert env_cfg.rewards.arm_torque_saturation.weight == -0.5
    assert not any("arm_torque_saturation" in marker for marker in applied)
    line = train_mod._reward_backend_compatibility_log_line(
        {
            "name": "arm_torque_saturation",
            "requested_weight": -0.5,
            "effective_weight": 0.0,
            "status": "disabled_incompatible_actuator_backend",
            "reason_code": (
                "implicit_actuator_has_no_proven_explicit_preclip_demand"
            ),
        }
    )
    assert "requested_weight=-0.5" in line
    assert "effective_weight=0.0" in line
    assert "status=disabled_incompatible_actuator_backend" in line
    assert ".weight=-0.5" not in line
    assert "applied" not in line


def test_lateral_trainer_default_off_is_historical_no_hook_path():
    env_cfg, applied = _apply({"lateral_perturbation": {"enabled": False}})
    assert not hasattr(env_cfg, train_mod._LATERAL_TRAINING_SPEC_ATTR)
    assert applied == ["lateral_perturbation.enabled=False (historical no-hook path)"]
    assert train_mod._resolve_lateral_training_runtime(
        _NS(cfg=env_cfg, step_dt=0.02)
    ) is None


@pytest.mark.parametrize("cell", ["L0", "L1"])
def test_lateral_trainer_translation_accepts_only_frozen_cells(cell):
    env_cfg, applied = _apply(
        {
            "lateral_perturbation": {
                "enabled": True,
                "cell": cell,
                "seed": 20260715,
            }
        }
    )
    assert getattr(env_cfg, train_mod._LATERAL_TRAINING_SPEC_ATTR) == {
        "schema_version": 1,
        "cell": cell,
        "seed": 20260715,
    }
    assert len(applied) == 1
    assert f"cell={cell}" in applied[0]
    assert "recovery_hold_only" in applied[0]


@pytest.mark.parametrize(
    "node, match",
    [
        ({"enabled": False, "cell": "L0"}, "require enabled=true"),
        ({"enabled": True, "cell": "anytime", "seed": 1}, "exactly 'L0' or 'L1'"),
        ({"enabled": True, "cell": "L1"}, "exact uint32"),
        ({"enabled": True, "cell": "L1", "seed": True}, "exact uint32"),
        ({"enabled": True, "cell": "L1", "seed": -1}, "exact uint32"),
        ({"enabled": True, "cell": "L1", "seed": 1, "body": "pelvis"}, "body"),
    ],
)
def test_lateral_trainer_translation_fails_closed(node, match):
    with pytest.raises(train_mod._OverrideError, match=match):
        _apply({"lateral_perturbation": node})


def test_lateral_trainer_is_conditionally_checkpoint_hard_contract_bound():
    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert "_resolve_lateral_training_runtime(env)" in source
    assert '{"lateral_perturbation": lateral_training[1]}' in source


def test_unknown_rewards_key_fails_loud():
    with pytest.raises(train_mod._OverrideError, match="face_gate_radiuss"):
        _apply({"rewards": {"face_gate_radiuss": 0.15}})  # typo'd key: must raise, never no-op


def _qdot_runtime_facts():
    names = [f"joint_{index:02d}" for index in range(31)]
    return {
        "articulation_joint_names": list(names),
        "joint_names": list(names),
        "joint_velocity_limits": [10.0 + index for index in range(31)],
    }


_A3_RUNTIME_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def _qdes_runtime_facts():
    return {
        "articulation_joint_names": list(_A3_RUNTIME_JOINTS),
        "joint_names": list(_A3_RUNTIME_JOINTS),
        "joint_velocity_limits": [10.0 + index for index in range(31)],
    }


def test_qdot_limit_hinge_default_off_and_override_markers():
    env_cfg, applied = _apply({})
    assert env_cfg.rewards.joint_velocity_limit_hinge.weight == 0.0
    assert env_cfg.rewards.joint_velocity_limit_hinge.params["margin"] == pytest.approx(0.85)
    assert env_cfg.rewards.joint_velocity_limit_hinge_probe.weight == 0.0
    assert not any("joint_velocity_limit_hinge" in item for item in applied)

    env_cfg, applied = _apply({
        "rewards": {
            "joint_velocity_limit_hinge_weight": -0.25,
            "joint_velocity_limit_hinge_margin": 0.8,
        }
    })
    assert env_cfg.rewards.joint_velocity_limit_hinge.weight == pytest.approx(-0.25)
    assert env_cfg.rewards.joint_velocity_limit_hinge.params["margin"] == pytest.approx(0.8)
    assert "rewards.joint_velocity_limit_hinge.weight=-0.25" in applied
    assert "rewards.joint_velocity_limit_hinge.params.margin=0.8" in applied
    assert env_cfg.rewards.joint_velocity_limit_hinge_probe.weight == 1.0
    assert env_cfg.rewards.joint_velocity_limit_hinge_probe.params["margin"] == pytest.approx(0.8)
    assert any("rewards.joint_velocity_limit_hinge_probe=" in item for item in applied)


@pytest.mark.parametrize("weight", [0.1, float("nan"), float("inf"), True, "bad"])
def test_qdot_limit_hinge_rejects_non_penalty_weight(weight):
    with pytest.raises(train_mod._OverrideError, match="finite and <= 0"):
        _apply({"rewards": {"joint_velocity_limit_hinge_weight": weight}})


@pytest.mark.parametrize("margin", [0.0, 1.0, -0.1, float("nan"), True, "bad"])
def test_qdot_limit_hinge_rejects_invalid_margin(margin):
    with pytest.raises(train_mod._OverrideError, match=r"finite and in \(0, 1\)"):
        _apply({"rewards": {"joint_velocity_limit_hinge_margin": margin}})


def test_qdot_limit_hinge_is_a_checkpoint_hard_contract_fact():
    control = _make_env_cfg()
    treatment, _ = _apply({
        "rewards": {
            "joint_velocity_limit_hinge_weight": -0.25,
            "joint_velocity_limit_hinge_margin": 0.8,
        }
    })
    facts = _qdot_runtime_facts()
    control_contract = train_mod._joint_velocity_limit_hinge_reward_contract(control, facts)
    treatment_contract = train_mod._joint_velocity_limit_hinge_reward_contract(treatment, facts)
    assert control_contract["enabled"] is False
    assert control_contract["weight"] == 0.0
    assert treatment_contract["enabled"] is True
    assert treatment_contract["weight"] == pytest.approx(-0.25)
    assert treatment_contract["margin"] == pytest.approx(0.8)
    assert treatment_contract["joint_order"] == "runtime_articulation_identity"
    assert treatment_contract["velocity_limit_source"].endswith("joint_velocity_limits")
    assert train_mod._contract_diff(control_contract, treatment_contract)


@pytest.mark.parametrize("bad_limit", [0.0, -1.0, float("nan"), float("inf")])
def test_qdot_limit_hinge_contract_rejects_bad_runtime_limit(bad_limit):
    facts = _qdot_runtime_facts()
    facts["joint_velocity_limits"][9] = bad_limit
    with pytest.raises(RuntimeError, match="finite and positive"):
        train_mod._joint_velocity_limit_hinge_reward_contract(_make_env_cfg(), facts)


def test_qdot_limit_hinge_contract_rejects_runtime_order_or_count_drift():
    facts = _qdot_runtime_facts()
    facts["joint_names"][0], facts["joint_names"][1] = (
        facts["joint_names"][1], facts["joint_names"][0]
    )
    with pytest.raises(RuntimeError, match="identity 31-joint"):
        train_mod._joint_velocity_limit_hinge_reward_contract(_make_env_cfg(), facts)

    facts = _qdot_runtime_facts()
    facts["joint_velocity_limits"].pop()
    with pytest.raises(RuntimeError, match="31 runtime joint_velocity_limits"):
        train_mod._joint_velocity_limit_hinge_reward_contract(_make_env_cfg(), facts)

    env_cfg = _make_env_cfg()
    env_cfg.rewards.joint_velocity_limit_hinge.params["asset_cfg"].joint_ids = [
        1, 0, *range(2, 31)
    ]
    with pytest.raises(RuntimeError, match="identity 31-joint order"):
        train_mod._joint_velocity_limit_hinge_reward_contract(
            env_cfg, _qdot_runtime_facts()
        )

    env_cfg = _make_env_cfg()
    env_cfg.rewards.joint_velocity_limit_hinge.params["expected_joint_count"] = 31.0
    with pytest.raises(RuntimeError, match="must be exactly 31"):
        train_mod._joint_velocity_limit_hinge_reward_contract(
            env_cfg, _qdot_runtime_facts()
        )


def test_processed_qdes_slew_default_off_and_explicit_overrides_enable_probe():
    env_cfg, applied = _apply({})
    assert env_cfg.rewards.processed_qdes_slew_hinge.weight == 0.0
    assert env_cfg.rewards.processed_qdes_slew_hinge_probe.weight == 0.0
    assert train_mod._processed_qdes_slew_hinge_reward_contract(
        env_cfg, _qdes_runtime_facts()
    ) is None
    assert not any("processed_qdes_slew" in marker for marker in applied)

    env_cfg, applied = _apply(
        {
            "rewards": {
                "processed_qdes_slew_hinge_weight": -0.4,
                "processed_qdes_slew_hinge_margin": 0.8,
                "processed_qdes_slew_hinge_recovery_start_s": 0.25,
                "processed_qdes_slew_hinge_recovery_end_s": 1.25,
            }
        }
    )
    term = env_cfg.rewards.processed_qdes_slew_hinge
    probe = env_cfg.rewards.processed_qdes_slew_hinge_probe
    assert term.weight == pytest.approx(-0.4)
    assert term.params["action_name"] == "joint_pos"
    assert term.params["command_name"] == "racket_target"
    assert term.params["margin"] == pytest.approx(0.8)
    assert term.params["recovery_start_s"] == pytest.approx(0.25)
    assert term.params["recovery_end_s"] == pytest.approx(1.25)
    assert probe.weight == 1.0
    assert probe.params == term.params
    assert any("processed_qdes_slew_hinge_probe" in marker for marker in applied)

    contract = train_mod._processed_qdes_slew_hinge_reward_contract(
        env_cfg, _qdes_runtime_facts()
    )
    assert contract["enabled"] is True
    assert contract["joint_count"] == 15
    assert contract["control_dt_s"] == pytest.approx(0.02)
    assert contract["control_dt_source"] == "env_cfg.sim.dt_times_decimation"
    assert contract["age_source"] == "per_env_exact_strike_control_tick_latch"
    assert contract["joint_names"] == [
        name
        for name in _A3_RUNTIME_JOINTS
        if "waist_" in name
        or any(part in name for part in ("_hip_", "_knee_", "_ankle_"))
    ]


def test_processed_qdes_slew_explicit_zero_control_is_hard_contract_bound():
    env_cfg, _ = _apply(
        {"rewards": {"processed_qdes_slew_hinge_weight": 0.0}}
    )
    contract = train_mod._processed_qdes_slew_hinge_reward_contract(
        env_cfg, _qdes_runtime_facts()
    )
    assert contract["enabled"] is False
    assert contract["weight"] == 0.0
    assert env_cfg.rewards.processed_qdes_slew_hinge_probe.weight == 1.0


@pytest.mark.parametrize("weight", [True, 0.1, float("nan"), float("inf"), float("-inf"), "bad"])
def test_processed_qdes_slew_rejects_non_penalty_weight(weight):
    with pytest.raises(train_mod._OverrideError, match="finite and <= 0"):
        _apply({"rewards": {"processed_qdes_slew_hinge_weight": weight}})


@pytest.mark.parametrize("margin", [True, 0.0, 1.0, -0.1, float("nan"), "bad"])
def test_processed_qdes_slew_rejects_bad_margin(margin):
    with pytest.raises(train_mod._OverrideError, match="margin/window"):
        _apply({"rewards": {"processed_qdes_slew_hinge_margin": margin}})


@pytest.mark.parametrize(
    "rewards",
    [
        {"processed_qdes_slew_hinge_recovery_start_s": -0.1},
        {"processed_qdes_slew_hinge_recovery_start_s": 1.55},
        {"processed_qdes_slew_hinge_recovery_end_s": 0.20},
        {"processed_qdes_slew_hinge_recovery_end_s": float("inf")},
        {"processed_qdes_slew_hinge_recovery_start_s": True},
    ],
)
def test_processed_qdes_slew_rejects_bad_recovery_window(rewards):
    with pytest.raises(train_mod._OverrideError, match="margin/window"):
        _apply({"rewards": rewards})


def test_processed_qdes_slew_contract_rejects_control_dt_drift():
    env_cfg, _ = _apply(
        {"rewards": {"processed_qdes_slew_hinge_weight": -0.4}}
    )
    env_cfg.decimation = 2
    with pytest.raises(RuntimeError, match=r"dt \* decimation == 0\.02"):
        train_mod._processed_qdes_slew_hinge_reward_contract(
            env_cfg, _qdes_runtime_facts()
        )


@pytest.mark.parametrize("weight", [True, 0.1, float("nan"), float("inf"), float("-inf"), "bad"])
def test_action_rate_weight_rejects_bool_nonfinite_and_positive(weight):
    with pytest.raises(train_mod._OverrideError, match="finite and <= 0"):
        _apply({"rewards": {"action_rate_weight": weight}})


@pytest.mark.parametrize("weight", [0.0, -0.05])
def test_action_rate_weight_accepts_finite_nonpositive_values(weight):
    env_cfg, applied = _apply({"rewards": {"action_rate_weight": weight}})
    assert env_cfg.rewards.action_rate_l2.weight == pytest.approx(weight)
    assert f"rewards.action_rate_l2.weight={weight}" in applied


def test_vendor_max_inward_brake_mode_is_exact_and_requires_guard():
    env_cfg, applied = _apply(
        {
            "actions": {
                "pre_apply_guard_brake_mode": (
                    "max_inward_until_nonoutward_v1"
                )
            }
        }
    )
    assert (
        env_cfg.actions.joint_pos.pre_apply_guard_brake_mode
        == "max_inward_until_nonoutward_v1"
    )
    assert any(
        line
        == "actions.joint_pos.pre_apply_guard_brake_mode="
        "max_inward_until_nonoutward_v1"
        for line in applied
    )

    with pytest.raises(train_mod._OverrideError, match="must be exactly"):
        _apply({"actions": {"pre_apply_guard_brake_mode": "velocity_horizon_v1"}})

    disabled = _make_env_cfg()
    disabled.actions.joint_pos.pre_apply_limit_guard = False
    with pytest.raises(train_mod._OverrideError, match="enabled pre-apply guard"):
        _apply(
            {
                "actions": {
                    "pre_apply_guard_brake_mode": (
                        "max_inward_until_nonoutward_v1"
                    )
                }
            },
            env_cfg=disabled,
        )


def test_vendor_guard_margin_fraction_is_exact_provenanced_and_default_isolated():
    baseline = _make_env_cfg()
    baseline_value = baseline.actions.joint_pos.pre_apply_guard_margin_fraction
    env_cfg, applied = _apply(
        {"actions": {"pre_apply_guard_margin_fraction": 0.06}},
        env_cfg=baseline,
    )
    assert baseline_value == pytest.approx(0.05)
    assert env_cfg.actions.joint_pos.pre_apply_guard_margin_fraction == pytest.approx(
        0.06
    )
    assert (
        "actions.joint_pos.pre_apply_guard_margin_fraction=0.06 "
        "(source=task.actions)"
    ) in applied

    ordinary, ordinary_applied = _apply({})
    assert ordinary.actions.joint_pos.pre_apply_guard_margin_fraction == pytest.approx(
        0.05
    )
    assert not any("pre_apply_guard_margin_fraction" in row for row in ordinary_applied)


def test_real_vendor_compose_translates_six_percent_guard_margin():
    import hydra
    from omegaconf import OmegaConf

    cfg_dir = os.path.abspath(os.path.join(HERE, "..", "cfg"))
    with hydra.initialize_config_dir(version_base=None, config_dir=cfg_dir):
        composed = hydra.compose(
            config_name="train",
            overrides=["task=HOPEPingPongActionBallA3VendorV1"],
        ).task
    task = OmegaConf.create(
        {
            "actions": OmegaConf.to_container(composed.actions, resolve=True),
            "rewards": {"reward_pack": "v1"},
        }
    )
    env_cfg, applied = _apply(task)
    assert env_cfg.actions.joint_pos.pre_apply_guard_margin_fraction == pytest.approx(
        0.06
    )
    assert env_cfg.actions.joint_pos.control_step_action_delay_min == 0
    assert env_cfg.actions.joint_pos.control_step_action_delay_max == 2
    assert env_cfg.actions.joint_pos.pre_apply_guard_brake_mode == (
        "max_inward_until_nonoutward_v1"
    )
    assert (
        env_cfg.actions.joint_pos.physx_control_position_limit_inset_fraction
        == pytest.approx(0.02)
    )
    assert (
        "actions.joint_pos.pre_apply_guard_margin_fraction=0.06 "
        "(source=task.actions)"
    ) in applied
    assert (
        "actions.joint_pos.physx_control_position_limit_inset_fraction="
        "0.02 (H_ctrl only; H_mech ledger unchanged)"
    ) in applied


@pytest.mark.parametrize(
    "value", [None, True, False, 0, 0.0, 0.01, 0.03, float("nan"), "0.02"]
)
def test_vendor_physx_control_inset_rejects_anything_except_exact_float_0p02(value):
    with pytest.raises(
        train_mod._OverrideError,
        match="exact finite float|exact vendor-only value",
    ):
        _apply(
            {
                "actions": {
                    "pre_apply_guard_margin_fraction": 0.06,
                    "physx_control_position_limit_inset_fraction": value,
                }
            }
        )


def test_vendor_physx_control_inset_requires_six_percent_guard_and_projection():
    wrong_guard = _make_env_cfg()
    with pytest.raises(train_mod._OverrideError, match="six-percent guard"):
        _apply(
            {
                "actions": {
                    "physx_control_position_limit_inset_fraction": 0.02,
                }
            },
            env_cfg=wrong_guard,
        )

    no_projection = _make_env_cfg()
    no_projection.actions.joint_pos.pre_apply_guard_margin_fraction = 0.06
    no_projection.actions.joint_pos.project_finite_preclamp_qdes_without_termination = False
    with pytest.raises(train_mod._OverrideError, match="finite projection"):
        _apply(
            {
                "actions": {
                    "physx_control_position_limit_inset_fraction": 0.02,
                }
            },
            env_cfg=no_projection,
        )


@pytest.mark.parametrize("value", [0.0, 0.499999])
def test_vendor_guard_margin_fraction_accepts_closed_open_range(value):
    env_cfg, applied = _apply(
        {"actions": {"pre_apply_guard_margin_fraction": value}}
    )
    assert env_cfg.actions.joint_pos.pre_apply_guard_margin_fraction == pytest.approx(
        value
    )
    assert any(
        row.startswith("actions.joint_pos.pre_apply_guard_margin_fraction=")
        for row in applied
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        1,
        -0.01,
        0.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        "0.06",
    ],
)
def test_vendor_guard_margin_fraction_rejects_nonexact_or_out_of_range(value):
    with pytest.raises(
        train_mod._OverrideError,
        match="exact finite float",
    ):
        _apply({"actions": {"pre_apply_guard_margin_fraction": value}})


def test_vendor_guard_margin_fraction_requires_enabled_guard_and_cfg_field():
    disabled = _make_env_cfg()
    disabled.actions.joint_pos.pre_apply_limit_guard = False
    with pytest.raises(train_mod._OverrideError, match="enabled pre-apply guard margin"):
        _apply(
            {"actions": {"pre_apply_guard_margin_fraction": 0.06}},
            env_cfg=disabled,
        )

    missing = _make_env_cfg()
    del missing.actions.joint_pos.pre_apply_guard_margin_fraction
    with pytest.raises(train_mod._OverrideError, match="enabled pre-apply guard margin"):
        _apply(
            {"actions": {"pre_apply_guard_margin_fraction": 0.06}},
            env_cfg=missing,
        )


def test_zero_joint_friction_is_explicit_and_all_actuators_are_zeroed():
    env_cfg, applied = _apply({"plant": {"zero_joint_friction": True}})
    assert {act.friction for act in env_cfg.scene.robot.actuators.values()} == {0.0}
    assert any("zero-friction plant control" in item for item in applied)


def test_zero_joint_friction_false_is_byte_preserving_noop():
    env_cfg, applied = _apply({"plant": {"zero_joint_friction": False}})
    assert env_cfg.scene.robot.actuators["legs"].friction == {"hip": 1.2, "knee": 2.4}
    assert env_cfg.scene.robot.actuators["arms"].friction == 0.1
    assert applied == []


def test_zero_joint_friction_fails_loud_on_bad_cfg_or_unknown_key():
    with pytest.raises(train_mod._OverrideError, match="scene.robot.actuators"):
        _apply({"plant": {"zero_joint_friction": True}}, _NS(scene=_NS()))
    with pytest.raises(train_mod._OverrideError, match="zero_joint_frction"):
        _apply({"plant": {"zero_joint_frction": True}})
    with pytest.raises(train_mod._OverrideError, match="explicit boolean"):
        _apply({"plant": {"zero_joint_friction": "treu"}})
    with pytest.raises(train_mod._OverrideError, match="must be a mapping"):
        _apply({"plant": True})


def test_zero_joint_friction_runtime_contract_must_be_31_exact_zeros():
    contract = {
        "joint_names": [f"j{i}" for i in range(31)],
        "joint_friction_coefficients": [0.0] * 31,
    }
    train_mod._require_zero_joint_friction_contract(contract)

    bad = dict(contract)
    bad["joint_friction_coefficients"] = [0.0] * 30 + [0.1]
    with pytest.raises(RuntimeError, match="non-zero coefficients"):
        train_mod._require_zero_joint_friction_contract(bad)

    short = {"joint_names": ["j0"], "joint_friction_coefficients": [0.0]}
    with pytest.raises(RuntimeError, match="exactly 31"):
        train_mod._require_zero_joint_friction_contract(short)


def test_rally_v3_recovery_overrides_are_wired_and_validated():
    env_cfg, applied = _apply({
        "motion": {"stand_start_yaw_range": [-0.35, 0.35]},
        "rewards": {
            "post_strike_brake_weight": 0.0,
            "post_strike_brake_std": 0.5,
            "hold_heading_weight": 0.1,
            "hold_heading_std": 0.6,
            "foot_orientation_hold_gate": True,
        },
    })
    assert env_cfg.commands.motion.stand_start_yaw_range == (-0.35, 0.35)
    assert env_cfg.rewards.hold_heading.weight == pytest.approx(0.1)
    assert env_cfg.rewards.hold_heading.params["std"] == pytest.approx(0.6)
    assert env_cfg.rewards.foot_orientation.params["hold_gate"] is True
    assert any("stand_start_yaw_range" in item for item in applied)


@pytest.mark.parametrize("yaw_range", ([0.4, -0.4], [float("nan"), 0.4], [-4.0, 0.0]))
def test_rally_v3_invalid_yaw_range_fails_loud(yaw_range):
    with pytest.raises(train_mod._OverrideError, match="stand_start_yaw_range"):
        _apply({"motion": {"stand_start_yaw_range": yaw_range}})


def test_reward_std_must_be_positive():
    with pytest.raises(train_mod._OverrideError, match="finite and > 0"):
        _apply({"rewards": {"hold_heading_std": 0.0}})


def test_question_bank_rejects_hitter_pure_target_mode():
    with pytest.raises(train_mod._OverrideError, match="incompatible.*hitter_pure"):
        _apply({"racket": {"question_bank": "/tmp/bank.npz", "target_mode": "hitter_pure"}})


def test_question_bank_rejects_midswing_question_redraw():
    with pytest.raises(train_mod._OverrideError, match="midswing_resample_prob"):
        _apply({"racket": {"question_bank": "/tmp/bank.npz", "midswing_resample_prob": 0.01}})


def test_question_bank_allows_speed_range_retiming_and_keeps_speed_contract_fields():
    env_cfg, applied = _apply({
        "motion": {"speed_scale_range": [0.8, 1.2]},
        "racket": {"question_bank": "/tmp/bank.npz"},
    })
    assert env_cfg.commands.motion.speed_scale_range == (0.8, 1.2)
    assert env_cfg.commands.racket_target.question_bank == "/tmp/bank.npz"
    assert any("speed_scale_range" in item for item in applied)
    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert '"motion_speed_scale_range"' in source
    assert '"motion_speed_scale_per_clip"' in source


def test_question_bank_allows_fixed_per_clip_retiming_override():
    env_cfg, applied = _apply({
        "motion": {"speed_scale_per_clip": [1.25, 0.9]},
        "racket": {"question_bank": "/tmp/bank.npz"},
    })
    assert env_cfg.commands.motion.speed_scale_per_clip == (1.25, 0.9)
    assert env_cfg.commands.racket_target.question_bank == "/tmp/bank.npz"
    assert any("speed_scale_per_clip" in item for item in applied)


@pytest.mark.parametrize(
    "mode", ["source_timestamp_compensated", "uncompensated"]
)
def test_atomic_tts_mode_override_swaps_only_policy_and_is_hard_contract_bound(
    mode, monkeypatch
):
    fake_mdp = _stub_mdp_module(monkeypatch)
    env_cfg, applied = _apply({
        "racket": {"target_delay_steps": 2, "target_delay_tts_mode": mode}
    })
    assert env_cfg.commands.racket_target.target_delay_steps == 2
    assert env_cfg.commands.racket_target.target_delay_tts_mode == mode
    assert env_cfg.observations.policy.time_to_strike.func is fake_mdp.actor_time_to_strike
    assert env_cfg.observations.critic.time_to_strike.func == "live_time_to_strike"
    assert any("critic remains live" in item for item in applied)
    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert '"racket_target_delay_tts_mode"' in source
    assert '"racket_target_delay_steps"' in source


def test_atomic_tts_mode_live_is_default_noop_and_unknown_mode_fails_loud():
    env_cfg, applied = _apply({})
    assert env_cfg.commands.racket_target.target_delay_tts_mode == "live"
    assert env_cfg.observations.policy.time_to_strike.func == "live_time_to_strike"
    assert not any("actor_time_to_strike" in item for item in applied)
    with pytest.raises(train_mod._OverrideError, match="target_delay_tts_mode"):
        _apply({"racket": {"target_delay_tts_mode": "timestamp-ish"}})


@pytest.mark.parametrize("pairing", ["shared_plus_y", "legacy_signed_vs_A"])
def test_face_command_pairing_override_is_strict_and_audited(pairing):
    env_cfg, applied = _apply({
        "racket": {"face_command": True, "face_command_pairing": pairing}
    })
    assert env_cfg.commands.racket_target.face_command_pairing == pairing
    if pairing == "shared_plus_y":
        assert any("kernel frame=+Y(A/bank)" in item for item in applied)
    else:
        assert any("legacy_signed_vs_A" in item for item in applied)


def test_face_command_pairing_unknown_value_fails_loud():
    with pytest.raises(train_mod._OverrideError, match="face_command_pairing"):
        _apply({"racket": {"face_command_pairing": "legacy-ish"}})


def test_legacy_motion_velocity_override_requires_explicit_bool_and_marks_inexact():
    env_cfg, applied = _apply({
        "motion": {"allow_legacy_link_origin_velocity": True}
    })
    assert env_cfg.commands.motion.allow_legacy_link_origin_velocity is True
    assert any("motion_kinematics_exact=false" in item for item in applied)

    env_cfg, applied = _apply({
        "motion": {"allow_legacy_link_origin_velocity": False}
    })
    assert env_cfg.commands.motion.allow_legacy_link_origin_velocity is False
    assert not any("motion_kinematics_exact=false" in item for item in applied)

    with pytest.raises(train_mod._OverrideError, match="explicit boolean"):
        _apply({"motion": {"allow_legacy_link_origin_velocity": "treu"}})


def test_all_real_task_yaml_keys_are_whitelisted():
    """Regression: the new task.rewards/_TERMINATION_KEYS whitelists must accept every key the
    real task YAMLs already set (otherwise the fail-loud check bricks existing tasks)."""
    yaml = pytest.importorskip("yaml")
    checked = 0
    for fn in sorted(os.listdir(CFG_TASK_DIR)):
        if not fn.endswith(".yaml"):
            continue
        with open(os.path.join(CFG_TASK_DIR, fn)) as fh:
            doc = yaml.safe_load(fh) or {}
        for node_key, whitelist in (
            ("plant", ("zero_joint_friction",)),
            ("rewards", train_mod._REWARD_KEYS),
            ("motion", train_mod._MOTION_KEYS),
            ("racket", train_mod._RACKET_KEYS),
            ("terminations", train_mod._TERMINATION_KEYS),
        ):
            node = doc.get(node_key)
            if node is None:
                continue
            unknown = sorted(str(k) for k in node.keys() if str(k) not in whitelist)
            assert not unknown, f"{fn}: {node_key} keys {unknown} missing from the whitelist"
            checked += 1
    assert checked >= 4  # the four tasks with rewards blocks at minimum


def test_implicit_a3_baselines_do_not_request_explicit_only_torque_reward():
    """Config truth: active A3 YAML must say OFF before backend composition.

    The compatibility receipt remains the fail-closed guard for an explicitly
    overridden non-zero request, but the two implicit-production baselines must
    not advertise a scientific weight that composition will necessarily zero.
    """

    yaml = pytest.importorskip("yaml")
    for filename in (
        "HOPEPingPongHitter.yaml",
        "HOPEPingPongDeployParity.yaml",
    ):
        path = os.path.join(CFG_TASK_DIR, filename)
        with open(path, encoding="utf-8") as stream:
            task = yaml.safe_load(stream)
        assert task["rewards"]["arm_torque_saturation_weight"] == 0.0, filename


def test_hitter_yaml_pins_legacy_reward_pack_while_action_ball_overrides_v2():
    """The missing-key default is v2, which bare Hitter cannot structurally host."""

    yaml = pytest.importorskip("yaml")
    expected = {
        "HOPEPingPongHitter.yaml": "v1",
        "HOPEPingPongActionBall.yaml": "v2",
    }
    for filename, reward_pack in expected.items():
        path = os.path.join(CFG_TASK_DIR, filename)
        with open(path, encoding="utf-8") as stream:
            task = yaml.safe_load(stream)
        assert task["rewards"]["reward_pack"] == reward_pack, filename


def test_pd_randomization_comments_match_split_vendor_ranges():
    """Prevent the old single +/-15% prose from regaining config authority."""

    env_cfg_path = os.path.abspath(
        os.path.join(
            HERE,
            "..",
            "source",
            "whole_body_tracking",
            "whole_body_tracking",
            "tasks",
            "tracking",
            "config",
            "agibot_a3",
            "hope_env_cfg.py",
        )
    )
    with open(env_cfg_path, encoding="utf-8") as stream:
        env_cfg_source = stream.read()
    assert "DR keeps PD ±15%" not in env_cfg_source
    assert "startup Kp scale (0.8,1.2)" in env_cfg_source
    assert "Kd scale (0.7,1.3)" in env_cfg_source

    with open(
        os.path.join(CFG_TASK_DIR, "HOPEPingPongHitter.yaml"),
        encoding="utf-8",
    ) as stream:
        hitter_source = stream.read()
    assert "old rationale text about" not in hitter_source
    assert "Gain DR" in hitter_source
    assert "does not depend on an IdealPD/explicit" in hitter_source


def test_virtual_ball_yaml_pins_outcome_dominant_effective_weights():
    """The composed task must not inherit DeployParity's historical 14/10/5 by accident."""
    yaml = pytest.importorskip("yaml")
    path = os.path.join(CFG_TASK_DIR, "HOPEPingPongVirtualBall.yaml")
    with open(path) as fh:
        task = yaml.safe_load(fh)
    rw = task["rewards"]
    expected = {
        "racket_position_weight": 4.0,
        "racket_position_std": 0.075,
        "racket_velocity_weight": 0.5,
        "racket_velocity_std": 0.5,
        "racket_normal_weight": 0.5,
        "racket_normal_std": 0.262,
        "foot_orientation_weight": 0.0,
    }
    assert {key: float(rw[key]) for key in expected} == expected

    # Exercise the same train.py translation that Hydra's resolved task node enters. Starting
    # from DeployParity-like 14/10/5 proves the child task actually wins, not merely that the
    # keys are present in text.
    env_cfg = _make_env_cfg()
    env_cfg.rewards.racket_position.weight = 14.0
    env_cfg.rewards.racket_position.params["std"] = 0.20
    env_cfg.rewards.racket_velocity.weight = 10.0
    env_cfg.rewards.racket_velocity.params["std"] = 1.0
    env_cfg.rewards.racket_normal.weight = 5.0
    env_cfg.rewards.racket_normal.params["std"] = 0.30
    env_cfg.rewards.foot_orientation.weight = -0.3
    env_cfg, _ = _apply({"rewards": rw}, env_cfg)
    for term_name, prefix in (
        ("racket_position", "racket_position"),
        ("racket_velocity", "racket_velocity"),
        ("racket_normal", "racket_normal"),
    ):
        term = getattr(env_cfg.rewards, term_name)
        assert term.weight == pytest.approx(expected[f"{prefix}_weight"])
        assert term.params["std"] == pytest.approx(expected[f"{prefix}_std"])
    assert env_cfg.rewards.foot_orientation.weight == pytest.approx(0.0)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (False, False),
        (0, False),
        ("false", False),
        ("0", False),
        ("no", False),
        (True, True),
        (1, True),
        ("true", True),
        ("1", True),
        ("yes", True),
    ],
)
def test_action_ball_boolean_overrides_require_explicit_values(
    raw,
    expected,
    monkeypatch,
):
    monkeypatch.setattr(
        train_mod,
        "_assert_physical_validity_guards_present",
        lambda cfg: None,
    )
    env_cfg = _make_env_cfg()
    env_cfg.commands.racket_target.action_ball_diagnostic_unauthorized = (
        not expected
    )
    env_cfg.commands.racket_target.virtual_ball = not expected

    env_cfg, _ = _apply(
        {
            "racket": {
                "action_ball_diagnostic_unauthorized": raw,
                "virtual_ball": raw,
            }
        },
        env_cfg,
    )

    assert (
        env_cfg.commands.racket_target.action_ball_diagnostic_unauthorized
        is expected
    )
    assert env_cfg.commands.racket_target.virtual_ball is expected


@pytest.mark.parametrize("raw", ["", "treu", "off", 2])
def test_action_ball_boolean_overrides_reject_ambiguous_values(
    raw,
    monkeypatch,
):
    monkeypatch.setattr(
        train_mod,
        "_assert_physical_validity_guards_present",
        lambda cfg: None,
    )
    env_cfg = _make_env_cfg()
    env_cfg.commands.racket_target.action_ball_diagnostic_unauthorized = False
    env_cfg.commands.racket_target.virtual_ball = False
    with pytest.raises(train_mod._OverrideError, match="explicit boolean"):
        _apply(
            {
                "racket": {
                    "action_ball_diagnostic_unauthorized": raw,
                    "virtual_ball": False,
                }
            },
            env_cfg,
        )


# --------------------------------------------------------------------------------------------- #
# base-decel weight-independent RewardManager-stage activation probe
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_base_decel_override_enables_same_exact_reward_stage_probe_for_both_arms(weight):
    env_cfg, applied = _apply(
        {
            "rewards": {
                "base_decel_weight": weight,
                "base_decel_v_gain": 1.7,
                "base_decel_v_max": 1.2,
                "base_decel_std": 0.35,
            }
        }
    )
    probe = env_cfg.rewards.base_decel_activation_probe
    assert probe.weight == pytest.approx(1.0)
    assert probe.params["v_gain"] == pytest.approx(1.7)
    assert probe.params["v_max"] == pytest.approx(1.2)
    assert probe.params["std"] == pytest.approx(0.35)
    assert probe.params["command_name"] == "racket_target"
    assert env_cfg.rewards.base_decel.weight == pytest.approx(weight)
    assert any("rewards.base_decel_activation_probe=" in item for item in applied)


def test_base_decel_probe_remains_manager_inactive_without_explicit_override():
    env_cfg, _ = _apply({"rewards": {"racket_position_weight": 14.0}})
    probe = env_cfg.rewards.base_decel_activation_probe
    assert probe.weight == pytest.approx(0.0)
    assert probe.params == {
        "command_name": "racket_target",
        "v_gain": 2.0,
        "v_max": 1.6,
        "std": 0.4,
    }


# --------------------------------------------------------------------------------------------- #
# V1 free_wrist_vel_mimic
# --------------------------------------------------------------------------------------------- #
def test_v1_free_wrist_vel_mimic_drops_wrist_from_lin_vel_only():
    env_cfg, applied = _apply({"rewards": {"free_wrist_vel_mimic": True}})
    assert _WRIST not in env_cfg.rewards.motion_body_lin_vel.params["body_names"]
    assert env_cfg.commands.motion.v1_free_wrist_vel_mimic_activation is True
    # the orientation/ang-vel mimic lists are free_wrist_ori_mimic's business — untouched here
    assert _WRIST in env_cfg.rewards.motion_body_ori.params["body_names"]
    assert _WRIST in env_cfg.rewards.motion_body_ang_vel.params["body_names"]
    assert _WRIST in env_cfg.rewards.motion_body_pos.params["body_names"]
    assert any("motion_body_lin_vel.body_names" in a for a in applied)


def test_v1_false_is_noop():
    env_cfg, applied = _apply({"rewards": {"free_wrist_vel_mimic": False}})
    assert _WRIST in env_cfg.rewards.motion_body_lin_vel.params["body_names"]
    assert not hasattr(env_cfg.commands.motion, "v1_free_wrist_vel_mimic_activation")
    assert applied == []


def test_v1_requires_explicit_body_list():
    env_cfg = _make_env_cfg()
    del env_cfg.rewards.motion_body_lin_vel.params["body_names"]
    with pytest.raises(train_mod._OverrideError):
        _apply({"rewards": {"free_wrist_vel_mimic": True}}, env_cfg)


# --------------------------------------------------------------------------------------------- #
# A0/A1 non-striking-arm imitation mask
# --------------------------------------------------------------------------------------------- #
def test_a1_non_striking_arm_mask_changes_only_four_imitation_body_lists():
    env_cfg = _make_env_cfg()
    rewards_before = {
        name: (term.weight, dict(term.params), term.func)
        for name, term in vars(env_cfg.rewards).items()
        if term is not None
    }
    terminations_before = dict(vars(env_cfg.terminations))
    action_clamp_before = env_cfg.actions.joint_pos.clamp
    actuator_friction_before = {
        name: actuator.friction for name, actuator in env_cfg.scene.robot.actuators.items()
    }

    env_cfg, applied = _apply(
        {"rewards": {"free_non_striking_arm_mimic": True}}, env_cfg
    )

    expected = ["torso_Link", *_RIGHT_STRIKING]
    changed = {
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    }
    for name, term in vars(env_cfg.rewards).items():
        if term is None:
            continue
        before_weight, before_params, before_func = rewards_before[name]
        assert term.weight == before_weight
        assert term.func == before_func
        if name in changed:
            assert term.params["body_names"] == expected
            assert {k: v for k, v in term.params.items() if k != "body_names"} == {
                k: v for k, v in before_params.items() if k != "body_names"
            }
        else:
            assert term.params == before_params
    assert dict(vars(env_cfg.terminations)) == terminations_before
    assert env_cfg.actions.joint_pos.clamp == action_clamp_before
    assert {
        name: actuator.friction for name, actuator in env_cfg.scene.robot.actuators.items()
    } == actuator_friction_before
    assert len([item for item in applied if "left non-striking arm imitation removed" in item]) == 4


def test_a0_non_striking_arm_false_is_byte_preserving_noop():
    env_cfg, applied = _apply(
        {"rewards": {"free_non_striking_arm_mimic": False}}
    )
    for name in (
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    ):
        assert getattr(env_cfg.rewards, name).params["body_names"] == _UPPER
    assert applied == []


def test_a1_non_striking_arm_mask_fails_closed_on_contract_drift_or_bad_boolean():
    env_cfg = _make_env_cfg()
    env_cfg.rewards.motion_body_pos.params["body_names"] = list(reversed(_UPPER))
    with pytest.raises(train_mod._OverrideError, match="exact reviewed"):
        _apply({"rewards": {"free_non_striking_arm_mimic": True}}, env_cfg)

    env_cfg = _make_env_cfg()
    del env_cfg.rewards.motion_body_ang_vel.params["body_names"]
    with pytest.raises(train_mod._OverrideError, match="explicit body list"):
        _apply({"rewards": {"free_non_striking_arm_mimic": True}}, env_cfg)

    with pytest.raises(train_mod._OverrideError, match="explicit boolean"):
        _apply({"rewards": {"free_non_striking_arm_mimic": "ture"}})


def test_a1_composes_with_current_free_racket_wrist_orientation_recipe():
    env_cfg, _ = _apply(
        {
            "rewards": {
                "free_wrist_ori_mimic": True,
                "free_non_striking_arm_mimic": True,
            }
        }
    )
    assert env_cfg.rewards.motion_body_pos.params["body_names"] == [
        "torso_Link", *_RIGHT_STRIKING
    ]
    assert env_cfg.rewards.motion_body_lin_vel.params["body_names"] == [
        "torso_Link", *_RIGHT_STRIKING
    ]
    for name in ("motion_body_ori", "motion_body_ang_vel"):
        assert getattr(env_cfg.rewards, name).params["body_names"] == [
            "torso_Link", "right_shoulder_roll_Link", "right_elbow_Link"
        ]


# --------------------------------------------------------------------------------------------- #
# 全身模仿开关(Franco 2026-07-25 裁定:下半身也应全局模仿,flag 可开关)
# --------------------------------------------------------------------------------------------- #
_LOWER_BODY = [
    "pelvis_link",
    "left_hip_roll_Link",
    "left_knee_Link",
    "left_ankle_roll_Link",
    "right_hip_roll_Link",
    "right_knee_Link",
    "right_ankle_roll_Link",
]


def test_full_body_mimic_restores_pelvis_and_legs_to_four_lists_only():
    env_cfg = _make_env_cfg()
    rewards_before = {
        name: (term.weight, dict(term.params), term.func)
        for name, term in vars(env_cfg.rewards).items()
        if term is not None
    }
    env_cfg, applied = _apply({"rewards": {"full_body_mimic": True}}, env_cfg)
    changed = {
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    }
    expected = [*_LOWER_BODY, *_UPPER]
    for name, term in vars(env_cfg.rewards).items():
        if term is None:
            continue
        before_weight, before_params, before_func = rewards_before[name]
        assert term.weight == before_weight
        assert term.func == before_func
        if name in changed:
            assert term.params["body_names"] == expected
        else:
            assert term.params == before_params
    assert len([m for m in applied if "full-body mimic" in m]) == 4


def test_full_body_mimic_false_is_byte_preserving_noop():
    env_cfg, applied = _apply({"rewards": {"full_body_mimic": False}})
    for name in (
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    ):
        assert getattr(env_cfg.rewards, name).params["body_names"] == _UPPER
    assert applied == []


def test_full_body_mimic_composes_with_wrist_and_left_arm_removals():
    # 组合语义:腿加回,被摘的照旧不学(ori/ang_vel 只剩 torso+右肩+右肘 = 最小 3 件合同)
    env_cfg, _ = _apply(
        {
            "rewards": {
                "free_wrist_ori_mimic": True,
                "free_non_striking_arm_mimic": True,
                "full_body_mimic": True,
            }
        }
    )
    assert env_cfg.rewards.motion_body_pos.params["body_names"] == [
        *_LOWER_BODY, "torso_Link", *_RIGHT_STRIKING
    ]
    for name in ("motion_body_ori", "motion_body_ang_vel"):
        assert getattr(env_cfg.rewards, name).params["body_names"] == [
            *_LOWER_BODY, "torso_Link", "right_shoulder_roll_Link", "right_elbow_Link"
        ]


def test_action_ball_wrist_position_and_paddle_window_master_translation():
    env_cfg, applied = _apply(
        {
            "rewards": {
                "free_wrist_pos_mimic": True,
                "motion_racket_position_weight": 0.2,
                "motion_racket_position_std": 0.7,
                "motion_racket_velocity_weight": 0.2,
                "motion_racket_velocity_std": 4.0,
                "motion_racket_normal_weight": 0.2,
                "motion_racket_normal_std": 3.141592653589793,
                "motion_racket_scale_in_strike_window": 0.0,
                "motion_racket_long_axis_weight": 0.1,
                "motion_racket_long_axis_std": 1.0,
                "motion_racket_long_axis_scale_in_strike_window": 1.0,
            }
        }
    )
    assert _WRIST not in env_cfg.rewards.motion_body_pos.params["body_names"]
    for name in (
        "motion_racket_position",
        "motion_racket_velocity",
        "motion_racket_normal",
    ):
        term = getattr(env_cfg.rewards, name)
        assert term.weight == pytest.approx(0.2)
        assert term.params["scale_in_strike_window"] == 0.0
    assert env_cfg.rewards.motion_racket_long_axis.weight == pytest.approx(0.1)
    assert (
        env_cfg.rewards.motion_racket_long_axis.params["scale_in_strike_window"]
        == 1.0
    )
    assert any("motion_body_pos.body_names-=" in marker for marker in applied)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_motion_racket_window_scale_rejects_out_of_range(value):
    with pytest.raises(train_mod._OverrideError, match="must be finite in"):
        _apply({"rewards": {"motion_racket_scale_in_strike_window": value}})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_motion_racket_long_axis_window_scale_rejects_out_of_range(value):
    with pytest.raises(train_mod._OverrideError, match="long_axis.*finite in"):
        _apply(
            {
                "rewards": {
                    "motion_racket_long_axis_scale_in_strike_window": value
                }
            }
        )


def test_full_body_mimic_fails_closed_on_drift_or_double_apply():
    env_cfg = _make_env_cfg()
    env_cfg.rewards.motion_body_pos.params["body_names"] = [*_LOWER_BODY, *_UPPER]
    with pytest.raises(train_mod._OverrideError, match="not already contain lower-body"):
        _apply({"rewards": {"full_body_mimic": True}}, env_cfg)

    env_cfg = _make_env_cfg()
    env_cfg.rewards.motion_body_pos.params["body_names"] = ["mystery_Link", *_UPPER]
    with pytest.raises(train_mod._OverrideError, match="reviewed A3 upper contract"):
        _apply({"rewards": {"full_body_mimic": True}}, env_cfg)

    with pytest.raises(train_mod._OverrideError, match="explicit boolean"):
        _apply({"rewards": {"full_body_mimic": "ture"}})


def test_a0_a1_post_override_masks_are_checkpoint_hard_contract_facts():
    a0, _ = _apply(
        {
            "rewards": {
                "free_wrist_ori_mimic": True,
                "free_wrist_vel_mimic": False,
                "free_non_striking_arm_mimic": False,
            }
        }
    )
    a1, _ = _apply(
        {
            "rewards": {
                "free_wrist_ori_mimic": True,
                "free_wrist_vel_mimic": False,
                "free_non_striking_arm_mimic": True,
            }
        }
    )
    a0_fact = train_mod._motion_imitation_body_names_contract(a0)
    a1_fact = train_mod._motion_imitation_body_names_contract(a1)
    assert tuple(a0_fact) == train_mod._MOTION_IMITATION_BODY_TERMS
    assert tuple(a1_fact) == train_mod._MOTION_IMITATION_BODY_TERMS
    assert a0_fact != a1_fact
    for name in train_mod._MOTION_IMITATION_BODY_TERMS:
        assert [body for body in a0_fact[name] if body not in _LEFT_NON_STRIKING] == a1_fact[name]


def test_imitation_body_names_hard_contract_rejects_forged_or_ambiguous_lists():
    env_cfg = _make_env_cfg()
    env_cfg.rewards.motion_body_pos.params["body_names"] = ["torso_Link", "torso_Link"]
    with pytest.raises(RuntimeError, match="non-empty, unique"):
        train_mod._motion_imitation_body_names_contract(env_cfg)

    env_cfg = _make_env_cfg()
    env_cfg.rewards.motion_body_ori.params["body_names"] = "torso_Link"
    with pytest.raises(RuntimeError, match="body_names list"):
        train_mod._motion_imitation_body_names_contract(env_cfg)

    env_cfg = _make_env_cfg()
    del env_cfg.rewards.motion_body_ang_vel.params
    with pytest.raises(RuntimeError, match="params mapping"):
        train_mod._motion_imitation_body_names_contract(env_cfg)


# --------------------------------------------------------------------------------------------- #
# V2 motion_scale_in_window
# --------------------------------------------------------------------------------------------- #
def test_v2_wires_window_scale_onto_non_none_motion_terms():
    env_cfg, applied = _apply({"rewards": {"motion_scale_in_window": 0.25}})
    for name in ("motion_global_anchor_ori", "motion_body_pos", "motion_body_ori",
                 "motion_body_lin_vel", "motion_body_ang_vel"):
        term = getattr(env_cfg.rewards, name)
        assert term.params["window_scale"] == 0.25, name
        assert term.params["window_command_name"] == "racket_target", name
    assert env_cfg.rewards.motion_global_anchor_pos is None  # removed term skipped, no crash
    assert env_cfg.commands.motion.v2_motion_scale_in_window_activation == 0.25
    assert any("motion_scale_in_window=0.25" in a for a in applied)
    # weights themselves untouched (the scaling is per-env inside the reward funcs)
    assert env_cfg.rewards.motion_body_lin_vel.weight == 1.0


def test_v2_absent_leaves_params_untouched():
    env_cfg, _ = _apply({"rewards": {"motion_scale": 1.0}})
    assert "window_scale" not in env_cfg.rewards.motion_body_lin_vel.params
    assert not hasattr(env_cfg.commands.motion, "v2_motion_scale_in_window_activation")


def test_v2_requires_racket_command():
    env_cfg = _make_env_cfg()
    env_cfg.commands = _NS(motion=env_cfg.commands.motion)  # no racket_target
    with pytest.raises(train_mod._OverrideError):
        _apply({"rewards": {"motion_scale_in_window": 0.25}}, env_cfg)


# --------------------------------------------------------------------------------------------- #
# 1c split strike windows
# --------------------------------------------------------------------------------------------- #
def test_1c_split_windows_set_cfg_fields():
    env_cfg, applied = _apply(
        {"racket": {"strike_window_pos_s": 0.02, "strike_window_wide_s": 0.10}})
    C = env_cfg.commands.racket_target
    assert C.strike_window_pos_s == 0.02
    assert C.strike_window_wide_s == 0.10
    assert C.strike_window_s == 0.12  # the base window stays the yaml value
    assert any("strike_window_pos_s=0.02" in a for a in applied)
    assert any("strike_window_wide_s=0.1" in a for a in applied)


# --------------------------------------------------------------------------------------------- #
# C2a proximity power-gate
# --------------------------------------------------------------------------------------------- #
def test_face_gate_sets_radius_param_on_vel_and_normal_only():
    env_cfg, applied = _apply({"rewards": {"face_gate_by_pos": True, "face_gate_radius": 0.15}})
    assert env_cfg.rewards.racket_velocity.params["pos_gate_radius"] == 0.15
    assert env_cfg.rewards.racket_normal.params["pos_gate_radius"] == 0.15
    assert "pos_gate_radius" not in env_cfg.rewards.racket_position.params
    assert any("face_gate_by_pos=true" in a for a in applied)


def test_face_gate_half_configured_raises_both_ways():
    with pytest.raises(train_mod._OverrideError, match="face_gate_radius"):
        _apply({"rewards": {"face_gate_by_pos": True}})
    with pytest.raises(train_mod._OverrideError, match="face_gate_by_pos"):
        _apply({"rewards": {"face_gate_radius": 0.15}})


# --------------------------------------------------------------------------------------------- #
# B2 constant guidance
# --------------------------------------------------------------------------------------------- #
def test_guidance_weight_set_and_sign_checked():
    env_cfg, applied = _apply({"rewards": {"racket_guidance_weight": -0.75}})
    assert env_cfg.rewards.racket_guidance.weight == -0.75
    assert any("racket_guidance.weight=-0.75" in a for a in applied)
    with pytest.raises(train_mod._OverrideError, match="must be <= 0"):
        _apply({"rewards": {"racket_guidance_weight": 0.5}})


def test_face_guidance_weight_and_theta_max_wired():
    env_cfg, applied = _apply({"rewards": {"racket_face_guidance_weight": -0.4,
                                           "racket_face_guidance_theta_max": 3.14159265}})
    assert env_cfg.rewards.racket_face_guidance.weight == -0.4
    assert env_cfg.rewards.racket_face_guidance.params["theta_max"] == 3.14159265
    assert any("racket_face_guidance.weight=-0.4" in a for a in applied)
    assert any("racket_face_guidance.params.theta_max=3.14159265" in a for a in applied)
    with pytest.raises(train_mod._OverrideError, match="must be <= 0"):
        _apply({"rewards": {"racket_face_guidance_weight": 0.4}})
    with pytest.raises(train_mod._OverrideError, match=r"in \(0, pi\]"):
        _apply({"rewards": {"racket_face_guidance_theta_max": 3.5}})  # > pi: clamp would lie
    with pytest.raises(train_mod._OverrideError, match=r"in \(0, pi\]"):
        _apply({"rewards": {"racket_face_guidance_theta_max": 0.0}})  # zero: gradient dead everywhere


def test_conditional_face_guidance_is_one_flag_with_fixed_budget_contract():
    env_cfg, applied = _apply(
        {"rewards": {"racket_face_conditional_guidance_weight": -0.4}}
    )
    term = env_cfg.rewards.racket_face_conditional_guidance
    assert term.weight == -0.4
    assert term.params == {
        "command_name": "racket_target",
        "theta_free": 0.262,
        "theta_max": 3.141592653589793,
        "pos_full": 0.075,
        "pos_zero": 0.095,
        "vel_full": 0.5,
        "vel_zero": 1.0,
    }
    assert any("racket_face_conditional_guidance.weight=-0.4" in item for item in applied)
    for invalid in (0.1, float("nan"), True, "wrong"):
        with pytest.raises(train_mod._OverrideError, match="finite number <= 0"):
            _apply({"rewards": {"racket_face_conditional_guidance_weight": invalid}})

    fact = train_mod._racket_guidance_reward_contract(env_cfg, racket_task=True)
    assert fact["conditional_signed_face"] == {"weight": -0.4, **term.params}


def test_c2_d2_guidance_recipe_is_a_checkpoint_hard_contract_fact():
    c2, _ = _apply({
        "rewards": {
            "racket_guidance_weight": 0.0,
            "racket_face_guidance_weight": 0.0,
            "racket_face_guidance_theta_max": 3.141592653589793,
        }
    })
    d2, _ = _apply({
        "rewards": {
            "racket_guidance_weight": 0.0,
            "racket_face_guidance_weight": -0.4,
            "racket_face_guidance_theta_max": 3.141592653589793,
        }
    })
    c2_fact = train_mod._racket_guidance_reward_contract(c2, racket_task=True)
    d2_fact = train_mod._racket_guidance_reward_contract(d2, racket_task=True)
    assert c2_fact != d2_fact
    assert c2_fact["signed_face"]["weight"] == 0.0
    assert d2_fact["signed_face"]["weight"] == -0.4

    c2_without_axis = {
        **c2_fact,
        "signed_face": {**c2_fact["signed_face"], "weight": "<causal-axis>"},
    }
    d2_without_axis = {
        **d2_fact,
        "signed_face": {**d2_fact["signed_face"], "weight": "<causal-axis>"},
    }
    assert c2_without_axis == d2_without_axis
    assert train_mod._racket_guidance_reward_contract(c2, racket_task=False) is None
    with open(train_mod.__file__, encoding="utf-8") as stream:
        assert '"racket_guidance_reward"' in stream.read()


@pytest.mark.parametrize(
    "term_name,attribute,value,message",
    [
        ("racket_guidance", "weight", float("nan"), "finite and <= 0"),
        ("racket_guidance", "weight", 0.1, "finite and <= 0"),
        ("racket_face_guidance", "weight", True, "finite number"),
        ("racket_face_conditional_guidance", "weight", True, "finite number"),
    ],
)
def test_guidance_hard_contract_rejects_ambiguous_weights(
    term_name, attribute, value, message
):
    env_cfg = _make_env_cfg()
    setattr(getattr(env_cfg.rewards, term_name), attribute, value)
    with pytest.raises(RuntimeError, match=message):
        train_mod._racket_guidance_reward_contract(env_cfg, racket_task=True)


def test_guidance_hard_contract_rejects_wrong_command_or_bounds():
    env_cfg = _make_env_cfg()
    env_cfg.rewards.racket_face_guidance.params["command_name"] = "motion"
    with pytest.raises(RuntimeError, match="exactly 'racket_target'"):
        train_mod._racket_guidance_reward_contract(env_cfg, racket_task=True)

    env_cfg = _make_env_cfg()
    env_cfg.rewards.racket_face_guidance.params["theta_max"] = 3.5
    with pytest.raises(RuntimeError, match=r"finite and \(0, pi\]"):
        train_mod._racket_guidance_reward_contract(env_cfg, racket_task=True)

    for field, value, message in (
        ("theta_free", 3.2, "theta_free < theta_max"),
        ("pos_full", 0.1, "pos_full < pos_zero"),
        ("vel_zero", 0.4, "vel_full < vel_zero"),
    ):
        env_cfg = _make_env_cfg()
        env_cfg.rewards.racket_face_conditional_guidance.params[field] = value
        with pytest.raises(RuntimeError, match=message):
            train_mod._racket_guidance_reward_contract(env_cfg, racket_task=True)


# --------------------------------------------------------------------------------------------- #
# R-a actor leg-reference mask
# --------------------------------------------------------------------------------------------- #
def _stub_mdp_module(monkeypatch):
    """train.py lazily imports whole_body_tracking...mdp inside the R-a branch — stub the chain."""
    fake_mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    fake_mdp.generated_commands_actor_leg_masked = "LEG_MASKED_FUNC"
    fake_mdp.actor_time_to_strike = object()
    fake_tracking = types.ModuleType("whole_body_tracking.tasks.tracking")
    fake_tracking.mdp = fake_mdp
    fake_tasks = types.ModuleType("whole_body_tracking.tasks")
    fake_tasks.tracking = fake_tracking
    fake_root = types.ModuleType("whole_body_tracking")
    fake_root.tasks = fake_tasks
    for name, m in (
        ("whole_body_tracking", fake_root),
        ("whole_body_tracking.tasks", fake_tasks),
        ("whole_body_tracking.tasks.tracking", fake_tracking),
        ("whole_body_tracking.tasks.tracking.mdp", fake_mdp),
    ):
        monkeypatch.setitem(sys.modules, name, m)
    return fake_mdp


def test_ra_swaps_actor_command_func_only(monkeypatch):
    fake_mdp = _stub_mdp_module(monkeypatch)
    env_cfg, applied = _apply({"actor_leg_ref_mask": True})
    assert env_cfg.observations.policy.command.func is fake_mdp.generated_commands_actor_leg_masked
    assert env_cfg.observations.critic.command.func == "generated_commands"  # critic untouched
    assert any("generated_commands_actor_leg_masked" in a for a in applied)


def test_ra_false_is_noop(monkeypatch):
    _stub_mdp_module(monkeypatch)
    env_cfg, applied = _apply({"actor_leg_ref_mask": False})
    assert env_cfg.observations.policy.command.func == "generated_commands"
    assert applied == []


# --------------------------------------------------------------------------------------------- #
# R-c RSI birth fixes
# --------------------------------------------------------------------------------------------- #
def test_rc_motion_flags_set():
    env_cfg, applied = _apply(
        {"motion": {"rsi_skip_settle_frames": 6, "rsi_hold_root_stand_z": True}})
    M = env_cfg.commands.motion
    assert M.rsi_skip_settle_frames == 6 and isinstance(M.rsi_skip_settle_frames, int)
    assert M.rsi_hold_root_stand_z is True
    assert any("rsi_skip_settle_frames=6" in a for a in applied)
    assert any("rsi_hold_root_stand_z=True" in a for a in applied)


def test_rc_absent_keeps_defaults():
    env_cfg, _ = _apply({"motion": {"hold_steps_range": [0, 100]}})
    assert env_cfg.commands.motion.rsi_skip_settle_frames == 0
    assert env_cfg.commands.motion.rsi_hold_root_stand_z is False


def test_post_swing_teacher_cold_start_overrides_are_explicit_and_typed():
    env_cfg, applied = _apply(
        {
            "motion": {
                "post_swing_teacher_receipt": "/ignored/teacher/receipt.json",
                "post_swing_teacher_receipt_sha256": "a" * 64,
                "post_swing_teacher_retry_authorization": "/ignored/teacher/retry.json",
                "post_swing_teacher_retry_authorization_sha256": "b" * 64,
                "post_swing_teacher_root_linear_velocity_limit_mps": 2.0,
                "post_swing_teacher_root_angular_velocity_limit_radps": 4.0,
                "post_swing_require_ready_at_init": True,
                "post_swing_fail_fast_first_reset": True,
                "post_swing_first_reset_min_adopted_count": 700,
                "post_swing_first_reset_min_adopted_fraction": 0.20,
                "post_swing_first_reset_selection_tolerance": 0.03,
                "post_swing_first_reset_require_readback": True,
            }
        }
    )
    motion = env_cfg.commands.motion
    assert motion.post_swing_teacher_receipt.endswith("receipt.json")
    assert motion.post_swing_teacher_receipt_sha256 == "a" * 64
    assert motion.post_swing_teacher_retry_authorization.endswith("retry.json")
    assert motion.post_swing_teacher_retry_authorization_sha256 == "b" * 64
    assert motion.post_swing_require_ready_at_init is True
    assert motion.post_swing_fail_fast_first_reset is True
    assert motion.post_swing_teacher_root_linear_velocity_limit_mps == 2.0
    assert motion.post_swing_first_reset_min_adopted_count == 700
    assert motion.post_swing_first_reset_min_adopted_fraction == 0.20
    assert motion.post_swing_first_reset_selection_tolerance == 0.03
    assert motion.post_swing_first_reset_require_readback is True
    assert any("post_swing_require_ready_at_init=True" in item for item in applied)

    with pytest.raises(train_mod._OverrideError, match="explicit boolean"):
        _apply({"motion": {"post_swing_require_ready_at_init": "treu"}})


# --------------------------------------------------------------------------------------------- #
# R-b envelope-as-penalty
# --------------------------------------------------------------------------------------------- #
def test_rb_softens_envelope_and_wires_penalty_and_accounting():
    env_cfg, applied = _apply(
        {"terminations": {"envelope_as_penalty": True, "envelope_penalty_weight": -1.0}})
    T = env_cfg.terminations
    assert T.anchor_pos is None and T.ee_body_pos is None  # envelope no longer terminates
    # absolute terminations + anchor_ori stay
    assert T.anchor_ori == "ANCHOR_ORI_TERM"
    assert T.base_fell_tilt == "BASE_FELL_TILT_TERM"
    assert T.base_too_low == "BASE_TOO_LOW_TERM"
    assert env_cfg.rewards.tracking_envelope.weight == -1.0
    assert env_cfg.rewards.tracking_envelope.params["ignore_hold"] is True
    assert env_cfg.commands.racket_target.track_envelope_violation is True
    assert any("envelope_as_penalty" in a for a in applied)
    assert any("tracking_envelope.weight=-1.0" in a for a in applied)


def test_rb_default_weight_is_minus_one():
    env_cfg, _ = _apply({"terminations": {"envelope_as_penalty": True}})
    assert env_cfg.rewards.tracking_envelope.weight == -1.0
    assert env_cfg.rewards.tracking_envelope.params["ignore_hold"] is True


def test_rb_guards():
    with pytest.raises(train_mod._OverrideError, match="envelope_as_penalty"):
        _apply({"terminations": {"envelope_penalty_weight": -1.0}})  # weight without the flag
    with pytest.raises(train_mod._OverrideError, match="must be < 0"):
        _apply({"terminations": {"envelope_as_penalty": True, "envelope_penalty_weight": 1.0}})
    with pytest.raises(train_mod._OverrideError, match="does not\n?.*consume|translation layer"):
        _apply({"terminations": {"unknown_termination_knob": 1}})


def test_rb_off_leaves_terminations_untouched():
    env_cfg, applied = _apply({"terminations": {"envelope_as_penalty": False}})
    assert env_cfg.terminations.anchor_pos == "ANCHOR_POS_TERM"
    assert env_cfg.terminations.ee_body_pos == "EE_BODY_POS_TERM"
    assert env_cfg.rewards.tracking_envelope.weight == 0.0
    assert applied == []


# --------------------------------------------------------------------------------------------- #
# R9 lower-body-free: terminations.anchor_pos_off / ee_upper_only
# --------------------------------------------------------------------------------------------- #
# Real A3 ee_body_pos list after the flat_env_cfg __post_init__ re-pin (feet first, then hands).
_EE_BODIES = ["left_ankle_roll_Link", "right_ankle_roll_Link",
              "left_wrist_yaw_Link", "right_wrist_yaw_Link"]
_EE_WRISTS = ["left_wrist_yaw_Link", "right_wrist_yaw_Link"]


def _make_env_cfg_with_ee_term():
    """Fake env cfg whose ee_body_pos is a real-shaped DoneTerm (params + body list)."""
    env_cfg = _make_env_cfg()
    env_cfg.terminations.ee_body_pos = _Term(
        params={"command_name": "motion", "threshold": 0.25}, body_names=_EE_BODIES)
    return env_cfg


def test_r9_anchor_pos_off_removes_torso_leash_only():
    env_cfg, applied = _apply({"terminations": {"anchor_pos_off": True}})
    T = env_cfg.terminations
    assert T.anchor_pos is None  # the torso-z leash is gone
    # everything else — including the OTHER envelope termination — stays
    assert T.ee_body_pos == "EE_BODY_POS_TERM"
    assert T.anchor_ori == "ANCHOR_ORI_TERM"
    assert T.base_fell_tilt == "BASE_FELL_TILT_TERM"
    assert T.base_too_low == "BASE_TOO_LOW_TERM"
    # no penalty swap: this is a pure removal
    assert env_cfg.rewards.tracking_envelope.weight == 0.0
    assert env_cfg.commands.racket_target.track_envelope_violation is False
    assert any("anchor_pos_off" in a for a in applied)


def test_r9_anchor_pos_off_false_is_noop():
    env_cfg, applied = _apply({"terminations": {"anchor_pos_off": False}})
    assert env_cfg.terminations.anchor_pos == "ANCHOR_POS_TERM"
    assert applied == []


def test_r9_ee_upper_only_keeps_wrists_drops_ankles():
    env_cfg, applied = _apply({"terminations": {"ee_upper_only": True}},
                              _make_env_cfg_with_ee_term())
    term = env_cfg.terminations.ee_body_pos
    assert term.params["body_names"] == _EE_WRISTS  # ankles freed, wrist order preserved
    assert term.params["threshold"] == 0.25  # threshold/command untouched
    assert term.params["command_name"] == "motion"
    # the torso leash and absolutes are not this flag's business
    assert env_cfg.terminations.anchor_pos == "ANCHOR_POS_TERM"
    assert env_cfg.terminations.base_fell_tilt == "BASE_FELL_TILT_TERM"
    assert any("ee_upper_only" in a for a in applied)


def test_r9_ee_upper_only_false_is_noop():
    env_cfg, applied = _apply({"terminations": {"ee_upper_only": False}},
                              _make_env_cfg_with_ee_term())
    assert env_cfg.terminations.ee_body_pos.params["body_names"] == _EE_BODIES
    assert applied == []


def test_r9_full_lowerbody_free_pack(monkeypatch):
    """The R9 arm config: anchor_pos_off + ee_upper_only + actor_leg_ref_mask, all in one task."""
    fake_mdp = _stub_mdp_module(monkeypatch)
    env_cfg, applied = _apply(
        {"actor_leg_ref_mask": True,
         "terminations": {"anchor_pos_off": True, "ee_upper_only": True}},
        _make_env_cfg_with_ee_term())
    T = env_cfg.terminations
    assert T.anchor_pos is None
    assert T.ee_body_pos.params["body_names"] == _EE_WRISTS
    assert T.anchor_ori == "ANCHOR_ORI_TERM"
    assert T.base_fell_tilt == "BASE_FELL_TILT_TERM"
    assert T.base_too_low == "BASE_TOO_LOW_TERM"
    assert env_cfg.observations.policy.command.func is fake_mdp.generated_commands_actor_leg_masked
    assert env_cfg.observations.critic.command.func == "generated_commands"
    assert sum(("anchor_pos_off" in a) + ("ee_upper_only" in a) for a in applied) == 2


def test_r9_conflicts_with_envelope_as_penalty_raise():
    with pytest.raises(train_mod._OverrideError, match="anchor_pos_off"):
        _apply({"terminations": {"envelope_as_penalty": True, "anchor_pos_off": True}})
    with pytest.raises(train_mod._OverrideError, match="ee_upper_only"):
        _apply({"terminations": {"envelope_as_penalty": True, "ee_upper_only": True}},
               _make_env_cfg_with_ee_term())


def test_r9_ee_upper_only_unexpected_body_list_raises():
    # a torso body in the list: refusing to guess beats silently freeing it
    env_cfg = _make_env_cfg()
    env_cfg.terminations.ee_body_pos = _Term(
        params={"command_name": "motion", "threshold": 0.25},
        body_names=_EE_BODIES + ["torso_Link"])
    with pytest.raises(train_mod._OverrideError, match="wrists\\+ankles"):
        _apply({"terminations": {"ee_upper_only": True}}, env_cfg)
    # no explicit body list at all (params missing): fail loud, never no-op
    env_cfg2 = _make_env_cfg()  # ee_body_pos is a plain string stand-in, no params
    with pytest.raises(train_mod._OverrideError, match="body_names"):
        _apply({"terminations": {"ee_upper_only": True}}, env_cfg2)


# --------------------------------------------------------------------------------------------- #
# default-path guard: a realistic flag-free task node touches none of the new machinery
# --------------------------------------------------------------------------------------------- #
def test_deployparity_like_task_without_new_flags_is_untouched():
    task = {
        "rewards": {
            "racket_position_weight": 14.0, "racket_position_std": 0.20,
            "racket_velocity_weight": 10.0, "racket_velocity_std": 1.0,
            "racket_normal_weight": 5.0, "racket_normal_std": 0.30,
            "free_wrist_ori_mimic": False,
            "action_rate_weight": -0.10,
        },
        "racket": {"strike_window_s": 0.12},
        "motion": {"hold_steps_range": [0, 100], "post_swing_start_prob": 0.0},
    }
    env_cfg, applied = _apply(task)
    C = env_cfg.commands.racket_target
    R = env_cfg.rewards
    assert C.strike_window_pos_s is None and C.strike_window_wide_s is None
    assert C.track_envelope_violation is False
    assert "pos_gate_radius" not in R.racket_velocity.params
    assert "window_scale" not in R.motion_body_lin_vel.params
    assert R.racket_guidance.weight == 0.0 and R.tracking_envelope.weight == 0.0
    assert R.racket_face_conditional_guidance.weight == 0.0
    assert env_cfg.commands.motion.rsi_skip_settle_frames == 0
    assert env_cfg.commands.motion.rsi_hold_root_stand_z is False
    assert env_cfg.observations.policy.command.func == "generated_commands"
    assert env_cfg.terminations.anchor_pos == "ANCHOR_POS_TERM"
    assert env_cfg.terminations.ee_body_pos == "EE_BODY_POS_TERM"
    # sanity: the plain overrides did land
    assert R.racket_position.weight == 14.0 and R.racket_position.params["std"] == 0.20
    assert R.racket_strike_success.params == {
        "command_name": "racket_target",
        "std_pos": pytest.approx(0.075),
        "std_vel": pytest.approx(0.5),
        "std_normal": pytest.approx(0.262),
    }
    assert not any("adaptive-sigma lockstep" in row for row in applied)
    assert len(applied) > 0


def test_adaptive_tracking_stds_lock_additive_and_strike_success_kernels():
    env_cfg = _make_env_cfg()
    env_cfg.rewards.racket_strike_success.params.update(
        {"std_pos": 9.0, "std_vel": 8.0, "std_normal": 7.0}
    )

    env_cfg, applied = _apply(
        {
            "rewards": {
                "reward_pack": "v1",
                "racket_position_std": 0.20,
                "racket_velocity_std": 1.0,
                "racket_normal_std": 0.52,
            },
            "racket": {
                "adaptive_sigma": True,
                "adaptive_sigma_monotonic": True,
                "adaptive_sigma_normal": True,
            },
        },
        env_cfg,
    )

    assert env_cfg.rewards.racket_position.params["std"] == pytest.approx(0.20)
    assert env_cfg.rewards.racket_velocity.params["std"] == pytest.approx(1.0)
    assert env_cfg.rewards.racket_normal.params["std"] == pytest.approx(0.52)
    assert env_cfg.rewards.racket_strike_success.params == {
        "command_name": "racket_target",
        "std_pos": pytest.approx(0.20),
        "std_vel": pytest.approx(1.0),
        "std_normal": pytest.approx(0.52),
    }
    assert sum("adaptive-sigma lockstep with rewards.racket_" in row for row in applied) == 3


def test_static_explicit_tracking_stds_preserve_legacy_strike_success_kernels():
    env_cfg = _make_env_cfg()
    before = dict(env_cfg.rewards.racket_strike_success.params)

    env_cfg, applied = _apply(
        {
            "rewards": {
                "reward_pack": "v1",
                "racket_position_std": 0.20,
                "racket_velocity_std": 1.0,
                "racket_normal_std": 0.30,
            }
        },
        env_cfg,
    )

    assert env_cfg.rewards.racket_position.params["std"] == pytest.approx(0.20)
    assert env_cfg.rewards.racket_velocity.params["std"] == pytest.approx(1.0)
    assert env_cfg.rewards.racket_normal.params["std"] == pytest.approx(0.30)
    assert env_cfg.rewards.racket_strike_success.params == before
    assert not any("adaptive-sigma lockstep" in row for row in applied)


def test_absent_tracking_stds_do_not_touch_strike_success_kernels():
    env_cfg = _make_env_cfg()
    before = dict(env_cfg.rewards.racket_strike_success.params)

    env_cfg, applied = _apply(
        {"rewards": {"reward_pack": "v1"}}, env_cfg
    )

    assert env_cfg.rewards.racket_strike_success.params == before
    assert not any("adaptive-sigma lockstep" in row for row in applied)


def test_effective_reward_hash_only_request_requires_absolute_canary_profile(
    tmp_path: Path,
):
    target = tmp_path / "reward.json"
    cfg = {
        "action_ball_effective_reward_recipe_output_path": str(target),
        "n1_vendor_sigma_profile": "monotonic_fresh_canary_v1",
    }
    assert train_mod._resolve_action_ball_effective_reward_materialization_request(
        cfg, action_ball_launch_requested=True
    ) == str(target)
    with pytest.raises(RuntimeError, match="ActionBall-only"):
        train_mod._resolve_action_ball_effective_reward_materialization_request(
            cfg, action_ball_launch_requested=False
        )
    wrong = dict(cfg, n1_vendor_sigma_profile="static_v1")
    with pytest.raises(RuntimeError, match="profile marker"):
        train_mod._resolve_action_ball_effective_reward_materialization_request(
            wrong, action_ball_launch_requested=True
        )
    measured = dict(
        cfg, n1_vendor_sigma_profile="measured_vendor_v2_n1_static_v1"
    )
    assert train_mod._resolve_action_ball_effective_reward_materialization_request(
        measured, action_ball_launch_requested=True
    ) == str(target)
    relative = dict(
        cfg, action_ball_effective_reward_recipe_output_path="reward.json"
    )
    with pytest.raises(RuntimeError, match="absolute"):
        train_mod._resolve_action_ball_effective_reward_materialization_request(
            relative, action_ball_launch_requested=True
        )


def test_reward_hash_profiles_accept_only_exact_legacy_or_measured_vendor_v2():
    def env(
        *, flags, widths, success, schedule
    ):
        racket = types.SimpleNamespace(
            adaptive_sigma=flags[0],
            adaptive_sigma_monotonic=flags[1],
            adaptive_sigma_normal=flags[2],
            sigma_update_every=schedule[0],
            sigma_ema_scale=schedule[1],
            sigma_pos_min=schedule[2],
            sigma_pos_max=schedule[3],
            sigma_vel_min=schedule[4],
            sigma_vel_max=schedule[5],
            sigma_normal_min=schedule[6],
            sigma_normal_max=schedule[7],
        )
        rewards = types.SimpleNamespace(
            racket_position=types.SimpleNamespace(params={"std": widths[0]}),
            racket_velocity=types.SimpleNamespace(params={"std": widths[1]}),
            racket_normal=types.SimpleNamespace(params={"std": widths[2]}),
            racket_strike_success=types.SimpleNamespace(
                params={
                    "std_pos": success[0],
                    "std_vel": success[1],
                    "std_normal": success[2],
                }
            ),
        )
        return types.SimpleNamespace(
            commands=types.SimpleNamespace(racket_target=racket),
            rewards=rewards,
        )

    legacy_schedule = (500, 1.0, 0.075, 0.20, 0.5, 1.0, 0.262, 0.52)
    legacy = env(
        flags=(True, True, True),
        widths=(0.20, 1.0, 0.52),
        success=(0.20, 1.0, 0.52),
        schedule=legacy_schedule,
    )
    assert train_mod._validate_action_ball_effective_reward_materialization_profile(
        legacy, {"n1_vendor_sigma_profile": "monotonic_fresh_canary_v1"}
    ) == "monotonic_fresh_canary_v1"

    measured_schedule = (500, 1.0, 0.075, 0.50, 0.5, 3.0, 0.262, 2.10)
    measured = env(
        flags=(False, False, False),
        widths=(0.50, 3.0, 2.10),
        success=(0.075, 0.5, 0.262),
        schedule=measured_schedule,
    )
    assert train_mod._validate_action_ball_effective_reward_materialization_profile(
        measured,
        {"n1_vendor_sigma_profile": "measured_vendor_v2_n1_static_v1"},
    ) == "measured_vendor_v2_n1_static_v1"

    measured.rewards.racket_velocity.params["std"] = 2.99
    with pytest.raises(RuntimeError, match="profile differs"):
        train_mod._validate_action_ball_effective_reward_materialization_profile(
            measured,
            {"n1_vendor_sigma_profile": "measured_vendor_v2_n1_static_v1"},
        )
    with pytest.raises(RuntimeError, match="not code-owned"):
        train_mod._validate_action_ball_effective_reward_materialization_profile(
            legacy, {"n1_vendor_sigma_profile": "unknown"}
        )


def test_effective_reward_hash_only_receipt_is_canonical_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package = types.ModuleType("whole_body_tracking")
    package.__path__ = []
    utils = types.ModuleType("whole_body_tracking.utils")
    utils.__path__ = []
    module = types.ModuleType(
        "whole_body_tracking.utils.effective_reward_recipe"
    )
    module.canonical_effective_reward_recipe_json = lambda value: json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    monkeypatch.setitem(sys.modules, "whole_body_tracking", package)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.utils", utils)
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        module,
    )
    payload = {"schema_version": 1, "terms": []}
    digest = hashlib.sha256(
        module.canonical_effective_reward_recipe_json(payload).encode("utf-8")
    ).hexdigest()
    receipt = {**payload, "sha256": digest}
    target = tmp_path / "reward.json"
    result = train_mod._materialize_effective_reward_recipe_receipt(
        target, receipt
    )
    expected = (
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert target.read_bytes() == expected
    assert result["effective_reward_recipe_sha256"] == digest
    assert result["file_sha256"] == hashlib.sha256(expected).hexdigest()
    with pytest.raises(RuntimeError, match="already exists"):
        train_mod._materialize_effective_reward_recipe_receipt(target, receipt)


# --------------------------------------------------------------------------------------------- #
# YAML 显式 null 删参 (jiayi 8ee2e82a -> main): `some_key: null` 把继承来的 reward 参数从
# term.params 里删掉;没写的键一律不动。
# --------------------------------------------------------------------------------------------- #
def test_yaml_null_removes_inherited_reward_params_and_logs():
    env_cfg, applied = _apply({"rewards": {
        "hold_ready_reach": None,                    # ad-hoc setter 路径
        "processed_qdes_slew_hinge_margin": None,    # "requested"-block 路径
        "racket_position_std": None,                 # _set_reward std 路径
    }})
    hr = env_cfg.rewards.hold_ready.params
    assert "reach" not in hr
    assert hr["reach_mode"] == "station" and hr["std"] == 1.5  # 邻居参数不受影响
    slew = env_cfg.rewards.processed_qdes_slew_hinge.params
    assert "margin" not in slew
    assert slew["recovery_start_s"] == 0.20 and slew["recovery_end_s"] == 1.55
    # null 的 margin 不算 "requested":slew 探针保持休眠,不被激活块拉起
    assert env_cfg.rewards.processed_qdes_slew_hinge_probe.weight == 0.0
    assert "std" not in env_cfg.rewards.racket_position.params
    assert env_cfg.rewards.racket_position.weight == 14.0  # weight 不是 params 项,不动
    assert sorted(applied) == [
        "rewards.hold_ready.params.reach=<removed by YAML null>",
        "rewards.processed_qdes_slew_hinge.params.margin=<removed by YAML null>",
        "rewards.racket_position.params.std=<removed by YAML null>",
    ]


def test_yaml_null_removal_is_idempotent_when_param_or_term_absent():
    env_cfg = _make_env_cfg()
    del env_cfg.rewards.hold_ready.params["reach"]  # 这个血统本来就没带 reach
    env_cfg, applied = _apply(
        {"rewards": {
            "hold_ready_reach": None,                # params 里没有这个键
            "qdes_limit_barrier_margin_frac": None,  # fake 血统根本没有这个 term
        }},
        env_cfg=env_cfg,
    )
    assert applied == []  # "确保不存在"已成立 -> 静默无记账
    assert "reach" not in env_cfg.rewards.hold_ready.params


def test_yaml_null_outside_removal_table_still_means_absent():
    # weight/开关类键不在删参表里:null 仍等价于"没写",什么都不动(与 jiayi 语义一致)。
    env_cfg, applied = _apply({"rewards": {"racket_position_weight": None}})
    assert env_cfg.rewards.racket_position.weight == 14.0
    assert env_cfg.rewards.racket_position.params["std"] == 0.2
    assert applied == []


def test_yaml_non_null_values_on_removable_keys_still_set_params_normally():
    env_cfg, applied = _apply({"rewards": {"hold_ready_reach": 0.3}})
    assert env_cfg.rewards.hold_ready.params["reach"] == 0.3
    assert "rewards.hold_ready.params.reach=0.3" in applied


def test_yaml_null_removal_works_on_omegaconf_nodes():
    # 运行时 task 是 OmegaConf 节点而非 dict;"key in node" + null 值的判定必须同样成立。
    from omegaconf import OmegaConf

    task = OmegaConf.create({"rewards": {"hold_ready_reach": None}})
    env_cfg, applied = _apply(task)
    assert "reach" not in env_cfg.rewards.hold_ready.params
    assert applied == ["rewards.hold_ready.params.reach=<removed by YAML null>"]


def test_null_removal_table_is_inside_reward_whitelist():
    # 表键必须都在 _REWARD_KEYS 里,否则 _check_unknown_keys 先拒掉,表项成死代码
    # (train.py import 时也有同款自检,这里再守一道回归)。
    strays = sorted(set(train_mod._REWARD_NULL_REMOVABLE_PARAMS) - set(train_mod._REWARD_KEYS))
    assert strays == []


# --------------------------------------------------------------------------------------------- #
# B2 JOB1 — task.rewards.reward_pack(reward_redesign_20260725 §3/§3.5 一键成套换装;
# 2026-07-25 Franco 裁定:缺席 = 默认 v2,显式 v1 = legacy 兜底)
# --------------------------------------------------------------------------------------------- #
# v2 包展开后每一项的期望终值(蓝图钉死;与 train._REWARD_PACK_V2_* 表逐项对齐。
# 击球三通道 60/45/35 与击中层 30/850 是 §3.5 名义值,probe 校准后冻结 prereg)。
_PACK_V2_WEIGHTS = {
    "strike_upright": 0.0,
    "strike_ang_vel": 0.0,
    "strike_foot_vel": 0.0,
    "strike_vbob": 0.0,
    "hit_unstable_support": -10.0,
    "upright": 0.0,
    "upright_exp": 1.0,
    "arm_overreach": 0.0,
    "hold_ready": 0.0,
    "foot_orientation": 0.0,
    "prestrike_upright": 0.0,
    "prestrike_waist_twist": 0.0,
    "foot_slip_sq": -0.1,
    "foot_drag": 0.0,
    # 07-26 配方审计补漏:落地冲击罚此前既不在包里也不在臂 argv 里(=0 静默跑丢)
    "foot_soft_landing": -0.003,
    "racket_position": 4.0,
    "racket_velocity": 0.5,
    "racket_normal": 0.5,
    # v2.1(Franco 07-25):两个人造 AND 代理删除,上台组扛大奖,spin 先验删除
    "racket_strike_success": 0.0,
    "strike_capture_bonus": 0.0,
    # v2.2:上台只留 landing(过网+落台=先决条件 gate,pass_net 塑形下岗)
    "virtual_pass_net": 0.0,
    "virtual_landing": 500.0,
    "virtual_spin": 0.0,
    # 值封顶平滑(fresh 自杀区间解;冻结档位)
    "action_rate_l2": 0.0,
    "action_rate_clamped": -0.2,
    "action_acc_l2": -0.05,
    "death_penalty": -300.0,
}

_PACK_DEFAULTED_MARKER = (
    "rewards.reward_pack defaulted to v2 (2026-07-25 Franco ruling; set "
    "reward_pack=v1 for legacy baseline)"
)


def _pack_v2_task(rewards_extra=None, racket_extra=None):
    rewards = {"reward_pack": "v2"}
    rewards.update(rewards_extra or {})
    racket = {"adaptive_sigma": True}  # 第三通道必须搭在 adaptive_sigma 上(翻译层就校验)
    racket.update(racket_extra or {})
    return {"rewards": rewards, "racket": racket}


def test_reward_pack_v1_is_byte_preserving_legacy_baseline():
    # (b) 显式 reward_pack=v1 = legacy 兜底:所有 v2 会动的项保持 fake 初值,applied 只有
    # 一条 v1 标记(用 _apply_default 走真实入口,不经 _apply 的钉 v1/滤标记)。
    env_cfg, applied = _apply_default({"rewards": {"reward_pack": "v1"}})
    fresh = _make_env_cfg()
    for name in _PACK_V2_WEIGHTS:
        assert getattr(env_cfg.rewards, name).weight == getattr(fresh.rewards, name).weight
        assert getattr(env_cfg.rewards, name).params == getattr(fresh.rewards, name).params
    for name in ("motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"):
        assert getattr(env_cfg.rewards, name).params["body_names"] == _UPPER
    assert env_cfg.commands.racket_target.adaptive_sigma is False
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is False
    assert env_cfg.commands.motion.stand_start_prob == pytest.approx(0.25)  # v1 不动 motion
    assert applied == [_V1_MARKER]


def test_reward_pack_defaults_to_v2_full_expansion():
    # (a) 默认路径:reward_pack 缺席 = §3.5 全套数值落地 + defaulted 标记。用户没显式表态
    # adaptive_sigma 时默认包代为置 true(默认自洽)。
    env_cfg, applied = _apply_default({})
    R = env_cfg.rewards
    for name, weight in _PACK_V2_WEIGHTS.items():
        assert getattr(R, name).weight == pytest.approx(weight), name
    # full_body_mimic 生效:pelvis+6 腿回四个名单;v2 废除窗内模仿折扣
    for name in ("motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"):
        assert getattr(R, name).params["body_names"] == [*_LOWER_BODY, *_UPPER]
        assert "window_scale" not in getattr(R, name).params
    assert "window_scale" not in R.motion_global_anchor_ori.params
    # 07-26:adaptive sigma 退役——包不再触碰 racket cfg(静态验收 σ;显式键仍可开=消融 flag)
    assert env_cfg.commands.racket_target.adaptive_sigma is False
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is False
    # 07-26:stand starts 进包(防挥拍中段 RSI 空降刷分)
    assert env_cfg.commands.motion.stand_start_prob == pytest.approx(1.0)
    assert env_cfg.commands.motion.post_swing_start_prob == pytest.approx(0.0)
    assert "motion.stand_start_prob=1.0 (reward_pack=v2 stand starts)" in applied
    assert _PACK_DEFAULTED_MARKER in applied
    # 逐项走的仍是 v2 包机器:defaulted 标记之外,每条包改动照旧带 reward_pack=v2
    assert "rewards.hit_unstable_support.weight=-10.0 (reward_pack=v2)" in applied
    assert "rewards.virtual_landing.weight=500.0 (reward_pack=v2)" in applied
    assert not any("strike_capture_bonus" in m for m in applied)  # v2.1:代理不进包
    assert "rewards.racket_position.weight=4.0" in applied


def test_reward_pack_default_applies_when_rewards_node_is_absent_entirely():
    # 连 rewards 节点都没有的任务照样吃默认包(线上大量任务节点只配 racket/motion)。
    env_cfg, applied = _apply_default({"motion": {"hold_steps_range": [0, 100]}})
    assert env_cfg.rewards.upright_exp.weight == pytest.approx(1.0)
    assert env_cfg.rewards.virtual_landing.weight == pytest.approx(500.0)
    assert env_cfg.rewards.virtual_landing.params["mode"] == "legal_base"
    # 07-26 裁决:延付默认关(消融 flag);臂级显式键单测见 test_settle_delay_flag_*
    assert env_cfg.rewards.virtual_landing.params["settle_delay_s"] == 0.0
    assert env_cfg.rewards.strike_capture_bonus.weight == pytest.approx(0.0)
    assert _PACK_DEFAULTED_MARKER in applied


def test_reward_pack_default_explicit_keys_still_win():
    # (c) 显式键仍压过默认包:现役波的 17/7/5 三通道显式键继续生效。
    env_cfg, applied = _apply_default(
        {
            "rewards": {
                "racket_position_weight": 17.0,
                "racket_velocity_weight": 7.0,
                "racket_normal_weight": 5.0,
            }
        }
    )
    R = env_cfg.rewards
    assert R.racket_position.weight == pytest.approx(17.0)
    assert R.racket_velocity.weight == pytest.approx(7.0)
    assert R.racket_normal.weight == pytest.approx(5.0)
    # 没被压过的包项照常落地
    assert R.hit_unstable_support.weight == pytest.approx(-10.0)
    assert R.virtual_landing.weight == pytest.approx(500.0)
    assert R.strike_capture_bonus.weight == pytest.approx(0.0)  # v2.1:代理不进包
    assert len([m for m in applied if "user override wins" in m]) == 3


def test_vendor_bang_bang_explicitly_disables_action_acc_only():
    env_cfg, applied = _apply_default(
        {"rewards": {"reward_pack": "v2", "action_acc_weight": 0.0}}
    )
    assert env_cfg.rewards.action_acc_l2.weight == pytest.approx(0.0)
    assert env_cfg.rewards.action_rate_l2.weight == pytest.approx(0.0)
    assert env_cfg.rewards.action_rate_clamped.weight == pytest.approx(-0.2)
    assert "rewards.action_acc_l2.weight=0.0" in applied
    assert any(
        "rewards.action_acc_weight explicitly set" in row
        and "user override wins" in row
        for row in applied
    )


def test_landing_scale_keys_translate_and_win_over_pack():
    # scale 消融键:显式 weight/base_frac 压过包冻结值;越界拒收
    env_cfg, applied = _apply(
        {"rewards": {"reward_pack": "v2", "virtual_landing_weight": 791.9,
                     "virtual_landing_base_frac": 0.3}}
    )
    assert env_cfg.rewards.virtual_landing.weight == pytest.approx(791.9)
    assert env_cfg.rewards.virtual_landing.params["base_frac"] == pytest.approx(0.3)
    assert any("virtual_landing.weight=791.9" in m for m in applied)
    with pytest.raises(train_mod._OverrideError, match="base_frac"):
        _apply({"rewards": {"reward_pack": "v2", "virtual_landing_base_frac": 1.5}})
    with pytest.raises(train_mod._OverrideError, match="virtual_landing_weight"):
        _apply({"rewards": {"reward_pack": "v2", "virtual_landing_weight": -5.0}})


def test_death_penalty_weight_key_translates_and_wins_over_pack():
    # 硬安全终止罚消融键:显式键压包;
    # 只许 <=0(0=消融关闭),正数拒收
    env_cfg, applied = _apply(
        {"rewards": {"reward_pack": "v2", "death_penalty_weight": -900.0}}
    )
    assert env_cfg.rewards.death_penalty.weight == pytest.approx(-900.0)
    assert any("death_penalty.weight=-900.0" in m for m in applied)
    env_cfg, _ = _apply({"rewards": {"reward_pack": "v2", "death_penalty_weight": 0.0}})
    assert env_cfg.rewards.death_penalty.weight == pytest.approx(0.0)
    with pytest.raises(train_mod._OverrideError, match="death_penalty_weight"):
        _apply({"rewards": {"reward_pack": "v2", "death_penalty_weight": 5.0}})


def test_settle_delay_flag_translates_and_defaults_off():
    # 默认包 = 0(立发);显式 0.24 = 延付消融臂;负数/非法拒收
    env_cfg, applied = _apply(
        {"rewards": {"reward_pack": "v2", "virtual_landing_settle_delay_s": 0.24},
         "racket": {"adaptive_sigma": True}}
    )
    assert env_cfg.rewards.virtual_landing.params["settle_delay_s"] == pytest.approx(0.24)
    assert any("settle_delay_s=0.24" in m for m in applied)
    with pytest.raises(train_mod._OverrideError, match="settle_delay"):
        _apply({"rewards": {"reward_pack": "v2", "virtual_landing_settle_delay_s": -1.0},
                "racket": {"adaptive_sigma": True}})


def test_reward_pack_v2_strips_unclamped_action_rate_with_marker():
    # v2 用封顶版;谱系基线普遍带 action_rate_weight → 包剥离+记账,不拒收(防双计费)
    env_cfg, applied = _apply(
        {"rewards": {"reward_pack": "v2", "action_rate_weight": -0.2},
         "racket": {"adaptive_sigma": True}}
    )
    assert env_cfg.rewards.action_rate_l2.weight == 0.0
    assert env_cfg.rewards.action_rate_clamped.weight == pytest.approx(-0.2)
    assert any("action_rate_weight" in m and "dropped" in m for m in applied)


def test_reward_pack_default_with_motion_scale_in_window_fails_loud():
    # (d) 默认包 + 显式 motion_scale_in_window = legacy 配方撞新默认:响亮失败,报错文案
    # 直接指路 reward_pack=v1(现役 wave yaml 全配了这个键——有意的响亮失败)。
    with pytest.raises(
        train_mod._OverrideError, match=r"must declare\s+reward_pack=v1"
    ):
        _apply_default({"rewards": {"motion_scale_in_window": 0.25}})


def test_reward_pack_default_leaves_sigma_choices_to_the_recipe():
    # 07-26 退役:显式 adaptive_sigma=false 不再与默认包相撞(σ 静态验收档=新常态);
    # 显式 true 仍可开(消融 flag),包一概不动。
    env_cfg, _ = _apply_default({"racket": {"adaptive_sigma": False}})
    assert env_cfg.commands.racket_target.adaptive_sigma is False
    env_cfg, _ = _apply_default({"racket": {"adaptive_sigma": True, "adaptive_sigma_normal": True}})
    assert env_cfg.commands.racket_target.adaptive_sigma is True
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is True


def test_action_ball_real_compose_and_translator_expose_monotonic_sigma_default_off():
    import hydra
    from omegaconf import OmegaConf

    cfg_dir = os.path.abspath(os.path.join(HERE, "..", "cfg"))
    with hydra.initialize_config_dir(version_base=None, config_dir=cfg_dir):
        baseline = hydra.compose(
            config_name="train",
            overrides=["task=HOPEPingPongActionBall"],
        ).task
        enabled = hydra.compose(
            config_name="train",
            overrides=[
                "task=HOPEPingPongActionBall",
                "task.racket.adaptive_sigma=true",
                "task.racket.adaptive_sigma_normal=true",
                "task.racket.adaptive_sigma_monotonic=true",
            ],
        ).task

    assert baseline.racket.adaptive_sigma_monotonic is False
    assert enabled.racket.adaptive_sigma is True
    assert enabled.racket.adaptive_sigma_normal is True
    assert enabled.racket.adaptive_sigma_monotonic is True

    # Feed the values produced by the real Hydra compose through the actual translator seam.
    composed_racket = OmegaConf.to_container(enabled.racket, resolve=True)
    env_cfg, applied = _apply(
        {
            "rewards": {
                "reward_pack": "v1",
                "racket_position_std": float(enabled.rewards.racket_position_std),
                "racket_velocity_std": float(enabled.rewards.racket_velocity_std),
                "racket_normal_std": float(enabled.rewards.racket_normal_std),
            },
            "racket": {
                key: composed_racket[key]
                for key in (
                    "adaptive_sigma",
                    "adaptive_sigma_normal",
                    "adaptive_sigma_monotonic",
                )
            }
        }
    )
    assert env_cfg.commands.racket_target.adaptive_sigma is True
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is True
    assert env_cfg.commands.racket_target.adaptive_sigma_monotonic is True
    assert "racket_target.adaptive_sigma_monotonic=True" in applied


def test_monotonic_sigma_override_rejects_unknown_boolean_spelling_and_null_is_noop():
    with pytest.raises(train_mod._OverrideError, match="must be an explicit boolean"):
        _apply({"racket": {"adaptive_sigma_monotonic": "not-a-bool"}})
    env_cfg, _ = _apply({"racket": {"adaptive_sigma_monotonic": None}})
    assert env_cfg.commands.racket_target.adaptive_sigma_monotonic is False


def test_reward_pack_default_optout_normal_channel_keeps_sigma_machinery_off():
    # 显式 adaptive_sigma_normal=false 退订第三通道:默认包不再碰 adaptive_sigma,其余照落。
    env_cfg, applied = _apply_default({"racket": {"adaptive_sigma_normal": False}})
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is False
    assert env_cfg.commands.racket_target.adaptive_sigma is False
    assert env_cfg.rewards.upright_exp.weight == pytest.approx(1.0)
    assert not any("racket_target.adaptive_sigma=True" in m for m in applied)


def test_reward_pack_v2_expands_every_blueprint_mutation_with_markers():
    # (b) 全套展开:权重/flag/参数逐项核对 + reward_pack=v2 标记条数钉死。
    env_cfg, applied = _apply(_pack_v2_task())
    R = env_cfg.rewards
    for name, weight in _PACK_V2_WEIGHTS.items():
        assert getattr(R, name).weight == pytest.approx(weight), name
    # full_body_mimic=true:pelvis+6 腿回四个名单
    for name in ("motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"):
        assert getattr(R, name).params["body_names"] == [*_LOWER_BODY, *_UPPER]
        # v2 废除窗内模仿折扣:六个 motion 项一律不带 window_scale 参数
        assert "window_scale" not in getattr(R, name).params
    assert "window_scale" not in R.motion_global_anchor_ori.params
    # racket 侧:拍面 sigma 第三通道开,搭在显式 adaptive_sigma=true 上
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is False  # 07-26 退役:包不动 sigma
    assert env_cfg.commands.racket_target.adaptive_sigma is True
    # 每条包改动的 applied 标记都带 reward_pack=v2；optional table 缺席也必须留下
    # 一条 skipped receipt，所以当前 fake lineage 共 33 条。
    pack_markers = [m for m in applied if "reward_pack=v2" in m]
    assert len(pack_markers) == 33
    # 键控注入的项同时会有覆写层自己的标准记账(证明真走了现有翻译层)
    assert "rewards.hold_ready.weight=0.0" in applied
    assert "rewards.foot_slip_sq.weight=-0.1" in applied
    assert "rewards.racket_position.weight=4.0" in applied
    assert "rewards.racket_velocity.weight=0.5" in applied
    assert "rewards.racket_normal.weight=0.5" in applied
    assert len([m for m in applied if "full-body mimic" in m]) == 4
    # direct 项的标记逐条在
    assert "rewards.hit_unstable_support.weight=-10.0 (reward_pack=v2)" in applied
    assert "rewards.upright_exp.weight=1.0 (reward_pack=v2)" in applied
    assert "rewards.death_penalty.weight=-300.0 (reward_pack=v2)" in applied
    assert "rewards.racket_strike_success.weight=0.0 (reward_pack=v2)" in applied
    assert "rewards.virtual_pass_net.weight=0.0 (reward_pack=v2)" in applied
    assert "rewards.virtual_spin.weight=0.0 (reward_pack=v2)" in applied
    assert (
        "rewards.table_hit_penalty ABSENT -> skipped (reward_pack=v2 optional)"
        in applied
    )
    assert any("virtual_landing.params" in m and "legal_base" in m for m in applied)
    assert "motion.stand_start_prob=1.0 (reward_pack=v2 stand starts)" in applied
    assert not any("adaptive_sigma" in m and "reward_pack" in m for m in applied)  # 07-26 退役:包署名的 sigma 标记不复存在(用户显式 racket 键的翻译标记不在此列)
    # 显式 v2(非默认)绝不带 defaulted 标记
    assert _PACK_DEFAULTED_MARKER not in applied


def test_reward_pack_v2_uses_one_generic_catastrophe_price_when_table_term_exists():
    env_cfg = _make_env_cfg()
    env_cfg.rewards.table_hit_penalty = _Term(weight=-1800.0)

    env_cfg, applied = _apply_default(
        {"rewards": {"reward_pack": "v2"}},
        env_cfg,
    )

    assert env_cfg.rewards.death_penalty.weight == pytest.approx(-300.0)
    assert env_cfg.rewards.table_hit_penalty.weight == pytest.approx(0.0)
    assert (
        "rewards.table_hit_penalty.weight=0.0 (reward_pack=v2 optional)"
        in applied
    )


# --------------------------------------------------------------------------------------------- #
# r6 低剂量冻结表:普通偏离带数值记账 + WARNING,strict 升级为 fail-loud;
# 已退役的 393.4/295.1/229.5 则无条件拒绝。
# --------------------------------------------------------------------------------------------- #
_LIVE_QUALITY_YAML = {  # cfg/task/HOPEPingPongVirtualBall.yaml 现役值,逐字
    "racket_position_weight": 4.0,
    "racket_velocity_weight": 0.5,
    "racket_normal_weight": 0.5,
}


def test_calibrated_override_marker_carries_both_values_and_ratio():
    env_cfg, applied = _apply_default(
        {"rewards": {"racket_position_weight": 17.0}}
    )
    assert env_cfg.rewards.racket_position.weight == pytest.approx(17.0)
    marker = next(m for m in applied if m.startswith("rewards.racket_position_weight="))
    assert "17.0" in marker and "4.0" in marker and "FROZEN" in marker
    assert "4.25x" in marker
    # grep 兼容:现役日志审计仍按这句话找覆写行
    assert "user override wins" in marker
    assert len([m for m in applied if "user override wins" in m]) == 1


def test_calibrated_override_prints_warning_to_stdout(capsys):
    _apply_default(
        {
            "rewards": {
                "racket_position_weight": 17.0,
                "racket_velocity_weight": 7.0,
                "racket_normal_weight": 5.0,
            }
        }
    )
    out = capsys.readouterr().out
    warnings = [line for line in out.splitlines() if "WARNING" in line and "FROZEN" in line]
    assert len(warnings) == 3
    assert any("racket_position_weight" in w and "4.0" in w for w in warnings)


def test_non_calibrated_pack_key_override_keeps_original_wording_and_is_quiet(capsys):
    # 布尔/清零类包键不是标定数,措辞与静默度一个字不变(不误报警)。
    _, applied = _apply_default({"rewards": {"hold_ready_weight": 3.0}})
    assert (
        "rewards.hold_ready_weight explicitly set — user override wins "
        "(reward_pack=v2 keeps hands off)"
    ) in applied
    assert "FROZEN" not in capsys.readouterr().out


def test_override_equal_to_frozen_value_is_not_flagged(capsys):
    # 显式写成与冻结值相同 = 没有被打败,不报警(避免"狼来了"淹掉真事故)。
    _, applied = _apply_default({"rewards": {"racket_position_weight": 4.0}})
    marker = next(m for m in applied if m.startswith("rewards.racket_position_weight="))
    assert "[same value]" in marker
    assert "FROZEN" not in capsys.readouterr().out


def test_reward_pack_strict_turns_a_defeated_frozen_value_into_fail_loud():
    # prereg 冻结臂的护栏:开了 strict 就不可能再"以为在跑冻结表"。
    with pytest.raises(train_mod._OverrideError, match=r"reward_pack_strict=true forbids"):
        _apply_default(
            {
                "rewards": {
                    "reward_pack_strict": True,
                    "racket_position_weight": 17.0,
                }
            }
        )


def test_reward_pack_strict_passes_when_nothing_defeats_the_frozen_table():
    env_cfg, applied = _apply_default({"rewards": {"reward_pack_strict": True}})
    assert env_cfg.rewards.racket_position.weight == pytest.approx(4.0)
    assert env_cfg.rewards.racket_velocity.weight == pytest.approx(0.5)
    assert env_cfg.rewards.racket_normal.weight == pytest.approx(0.5)
    assert (
        "rewards.reward_pack_strict=True (frozen pack values may not be overridden)" in applied
    )


def test_reward_pack_strict_default_off_keeps_default_boot_byte_identical():
    # 7263464b 的教训:无条件 raise 会炸掉每一次 default-v2 boot。默认必须是"警告不拦"。
    env_cfg, _ = _apply_default({"rewards": dict(_LIVE_QUALITY_YAML)})
    assert env_cfg.rewards.racket_position.weight == pytest.approx(4.0)


@pytest.mark.parametrize(
    "key,value",
    (
        ("racket_position_weight", 393.4),
        ("racket_velocity_weight", 295.1),
        ("racket_normal_weight", 229.5),
    ),
)
def test_retired_high_dose_quality_weights_fail_closed(key, value):
    with pytest.raises(train_mod._OverrideError, match="retired v2 high-dose"):
        _apply_default({"rewards": {key: value}})


def test_every_task_yaml_declares_the_quality_keys_so_the_pack_is_dead_code_there():
    # 这条测试保留旧七个 task 的缺陷存档:它们全都显式写了三键,所以包里的冻结质量表
    # 在那些谱系上仍是死码。ActionBall 现在有意采用已验证的 Tier-1 低 tracking:
    # pack 仍供应 v2 outcome/safety，其显式 4/0.5/0.5 按真实翻译顺序后写后赢。
    import glob
    import re

    declaring = []
    for path in sorted(glob.glob(os.path.join(CFG_TASK_DIR, "*.yaml"))):
        text = open(path, encoding="utf-8").read()
        if re.search(r"^\s*racket_position_weight\s*:", text, re.M):
            declaring.append(os.path.basename(path))
    assert "HOPEPingPongVirtualBall.yaml" in declaring
    action_ball = "HOPEPingPongActionBall.yaml"
    assert action_ball in declaring
    stage1_profiles = {
        "HOPEPingPongStage1NaturalClipA3VendorV1.yaml",
        "HOPEPingPongStage1NaturalClipA3VendorV2.yaml",
    }
    assert stage1_profiles.issubset(declaring)
    action_ball_successors = {"HOPEPingPongActionBallA3VendorV2.yaml"}
    assert action_ball_successors.issubset(declaring)
    assert len(
        [
            name
            for name in declaring
            if name != action_ball
            and name not in stage1_profiles
            and name not in action_ball_successors
        ]
    ) == 7, declaring

    import yaml

    with open(os.path.join(CFG_TASK_DIR, action_ball), encoding="utf-8") as fh:
        action_ball_rewards = yaml.safe_load(fh)["rewards"]
    assert action_ball_rewards["reward_pack"] == "v2"
    assert action_ball_rewards["reward_pack_strict"] is False
    assert {
        key: float(action_ball_rewards[key])
        for key in (
            "racket_position_weight",
            "racket_velocity_weight",
            "racket_normal_weight",
        )
    } == {
        "racket_position_weight": 4.0,
        "racket_velocity_weight": 0.5,
        "racket_normal_weight": 0.5,
    }


def test_action_ball_adopted_rewards_win_after_real_v2_pack_expansion():
    """Exercise the actual pack-first, explicit-task-values-last translation order."""

    import hydra
    from omegaconf import OmegaConf

    cfg_dir = os.path.abspath(os.path.join(HERE, "..", "cfg"))
    with hydra.initialize_config_dir(version_base=None, config_dir=cfg_dir):
        composed = hydra.compose(
            config_name="train",
            overrides=["task=HOPEPingPongActionBall"],
        ).task
    rewards_node = OmegaConf.to_container(composed.rewards, resolve=True)

    env_cfg = _make_env_cfg()
    # Add the Hitter/ActionBall-only reward terms to the dependency-light DeployParity fake.
    env_cfg.rewards.base_position = _Term(weight=1.0, params={"std": 0.3})
    env_cfg.rewards.joint_limit = _Term(weight=-5.0)
    env_cfg.rewards.undesired_contacts = _Term(weight=-0.1)
    env_cfg.rewards.pre_strike_foot_slip = _Term(weight=-0.2)
    env_cfg.rewards.arm_torque_saturation = _Term(weight=-0.5)
    env_cfg.rewards.table_hit_penalty = _Term(weight=-1800.0)
    env_cfg.rewards.qdes_limit_barrier = _Term(
        weight=0.0,
        params={
            "action_name": "joint_pos",
            "margin_frac": 0.08,
            "penalty_floor": 0.25,
        },
    )
    env_cfg.rewards.qdes_limit_barrier_probe = _Term(
        weight=0.0,
        params={
            "action_name": "joint_pos",
            "margin_frac": 0.08,
            "penalty_floor": 0.25,
        },
    )

    applied = train_mod._apply_task_overrides(
        env_cfg,
        {"rewards": rewards_node},
        clip_name=None,
    )
    R = env_cfg.rewards
    assert {
        "racket_position": R.racket_position.weight,
        "racket_velocity": R.racket_velocity.weight,
        "racket_normal": R.racket_normal.weight,
        "virtual_landing": R.virtual_landing.weight,
        "death_penalty": R.death_penalty.weight,
        "table_hit_penalty": R.table_hit_penalty.weight,
        "qdes_limit_barrier": R.qdes_limit_barrier.weight,
        "joint_limit": R.joint_limit.weight,
    } == pytest.approx(
        {
            "racket_position": 4.0,
            "racket_velocity": 0.5,
            "racket_normal": 0.5,
            "virtual_landing": 500.0,
            "death_penalty": -300.0,
            "table_hit_penalty": 0.0,
            "qdes_limit_barrier": -5.0,
            "joint_limit": -5.0,
        }
    )
    assert R.racket_position.params["std"] == pytest.approx(0.075)
    assert R.racket_velocity.params["std"] == pytest.approx(0.5)
    assert R.racket_normal.params["std"] == pytest.approx(0.262)
    assert R.qdes_limit_barrier.params["margin_frac"] == pytest.approx(0.08)
    assert R.qdes_limit_barrier_probe.weight == pytest.approx(1.0)

    # Prove the explicit ActionBall values agree exactly with the adopted r6 table.
    for key, low in (
        ("racket_position", 4.0),
        ("racket_velocity", 0.5),
        ("racket_normal", 0.5),
    ):
        marker = next(
            item
            for item in applied
            if item.startswith(f"rewards.{key}_weight=")
            and "FROZEN" in item
        )
        assert repr(low) in marker
        assert "[same value]" in marker


def _soft_limit_v2_contract_fixture(monkeypatch):
    def qdes_limit_barrier_v2():
        pass

    def qdes_limit_barrier_v2_probe():
        pass

    def actual_joint_limit_barrier_v2():
        pass

    def actual_joint_limit_barrier_v2_probe():
        pass

    callables = {
        func.__name__: func
        for func in (
            qdes_limit_barrier_v2,
            qdes_limit_barrier_v2_probe,
            actual_joint_limit_barrier_v2,
            actual_joint_limit_barrier_v2_probe,
        )
    }
    monkeypatch.setattr(
        train_mod,
        "_authoritative_mdp_reward_callable",
        callables.__getitem__,
    )
    shared = {"margin_frac": 0.08, "penalty_floor": 0.25}
    asset_cfg = _NS(name="robot", joint_ids=slice(None))
    rewards = _NS(
        qdes_limit_barrier=_Term(
            weight=-5.0,
            func=qdes_limit_barrier_v2,
            params={"action_name": "joint_pos", **shared},
        ),
        qdes_limit_barrier_probe=_Term(
            weight=1.0,
            func=qdes_limit_barrier_v2_probe,
            params={"action_name": "joint_pos", **shared},
        ),
        joint_limit=_Term(
            weight=-5.0,
            func=actual_joint_limit_barrier_v2,
            params={
                "asset_cfg": asset_cfg,
                **shared,
                "expected_joint_count": 31,
            },
        ),
        actual_joint_limit_barrier_probe=_Term(
            weight=1.0,
            func=actual_joint_limit_barrier_v2_probe,
            params={
                "asset_cfg": asset_cfg,
                **shared,
                "expected_joint_count": 31,
            },
        ),
    )
    facts = {
        "joint_names": [f"j{i}" for i in range(31)],
        "articulation_joint_names": [f"j{i}" for i in range(31)],
        "qdes_joint_pos_limits": [[-1.0, 1.0] for _ in range(31)],
    }
    return _NS(rewards=rewards), facts, callables


def test_soft_limit_v2_training_contract_binds_real_callables_and_two_channels(
    monkeypatch,
):
    cfg, facts, _ = _soft_limit_v2_contract_fixture(monkeypatch)
    qdes = train_mod._qdes_limit_barrier_reward_contract(cfg, facts)
    actual = train_mod._actual_joint_limit_barrier_reward_contract(
        cfg, facts, qdes_contract=qdes
    )
    assert qdes["schema_version"] == actual["schema_version"] == 2
    assert qdes["term_callable"].endswith(".qdes_limit_barrier_v2")
    assert qdes["probe_callable"].endswith(".qdes_limit_barrier_v2_probe")
    assert actual["term_callable"].endswith(".actual_joint_limit_barrier_v2")
    assert actual["probe_callable"].endswith(
        ".actual_joint_limit_barrier_v2_probe"
    )
    for key, expected in (
        ("weight", -5.0),
        ("margin_frac", 0.08),
        ("penalty_floor", 0.25),
        ("shape_rate", 4.0),
        ("stance_eps", 0.005),
        ("margin_floor", 0.005),
        ("per_joint_cap", 1.0),
    ):
        assert qdes[key] == pytest.approx(expected)
        assert actual[key] == pytest.approx(expected)
    assert qdes["aggregation"] == actual["aggregation"] == "sum_all_31_joints"
    assert qdes["gate"] == actual["gate"] == "dense_every_control_step"


def test_soft_limit_v2_contract_rejects_callable_floor_and_channel_drift(
    monkeypatch,
):
    cfg, facts, callables = _soft_limit_v2_contract_fixture(monkeypatch)
    cfg.rewards.qdes_limit_barrier_probe.func = callables[
        "actual_joint_limit_barrier_v2_probe"
    ]
    with pytest.raises(RuntimeError, match="authoritative callable"):
        train_mod._qdes_limit_barrier_reward_contract(cfg, facts)

    cfg, facts, _ = _soft_limit_v2_contract_fixture(monkeypatch)
    cfg.rewards.qdes_limit_barrier.params.pop("penalty_floor")
    cfg.rewards.qdes_limit_barrier_probe.params.pop("penalty_floor")
    with pytest.raises(RuntimeError, match="penalty_floor"):
        train_mod._qdes_limit_barrier_reward_contract(cfg, facts)

    cfg, facts, _ = _soft_limit_v2_contract_fixture(monkeypatch)
    qdes = train_mod._qdes_limit_barrier_reward_contract(cfg, facts)
    cfg.rewards.joint_limit.weight = -20.0
    with pytest.raises(RuntimeError, match="weight must match"):
        train_mod._actual_joint_limit_barrier_reward_contract(
            cfg, facts, qdes_contract=qdes
        )

    cfg, facts, _ = _soft_limit_v2_contract_fixture(monkeypatch)
    qdes = train_mod._qdes_limit_barrier_reward_contract(cfg, facts)
    cfg.rewards.joint_limit.params["margin_frac"] = 0.1
    cfg.rewards.actual_joint_limit_barrier_probe.params["margin_frac"] = 0.1
    with pytest.raises(RuntimeError, match="margin_frac must match"):
        train_mod._actual_joint_limit_barrier_reward_contract(
            cfg, facts, qdes_contract=qdes
        )


def test_reward_pack_v2_works_on_omegaconf_nodes():
    from omegaconf import OmegaConf

    env_cfg, applied = _apply(OmegaConf.create(_pack_v2_task()))
    assert env_cfg.rewards.hit_unstable_support.weight == pytest.approx(-10.0)
    assert env_cfg.rewards.foot_slip_sq.weight == pytest.approx(-0.1)
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is False  # 07-26 退役:包不动 sigma


def test_reward_pack_v2_explicit_user_keys_win():
    # (c) 包先展开,显式同名键后写后赢(rewards 键控项 + racket 第三通道都要赢;
    # 现役波的 17/7/5 三通道显式键必须继续生效)。
    env_cfg, applied = _apply(
        _pack_v2_task(
            rewards_extra={
                "hold_ready_weight": 1.5,
                "foot_slip_sq_weight": -0.3,
                "full_body_mimic": False,
                "racket_position_weight": 17.0,
                "racket_velocity_weight": 7.0,
                "racket_normal_weight": 5.0,
            },
            racket_extra={"adaptive_sigma_normal": False},
        )
    )
    R = env_cfg.rewards
    assert R.hold_ready.weight == pytest.approx(1.5)
    assert R.foot_slip_sq.weight == pytest.approx(-0.3)
    assert R.motion_body_pos.params["body_names"] == _UPPER  # 用户显式 false:腿不回名单
    assert R.racket_position.weight == pytest.approx(17.0)
    assert R.racket_velocity.weight == pytest.approx(7.0)
    assert R.racket_normal.weight == pytest.approx(5.0)
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is False
    # 没被用户压过的包项照常落地
    assert R.hit_unstable_support.weight == pytest.approx(-10.0)
    assert R.virtual_landing.weight == pytest.approx(500.0)
    assert R.strike_capture_bonus.weight == pytest.approx(0.0)
    assert R.foot_drag.weight == 0.0
    assert len([m for m in applied if "user override wins" in m]) == 6  # 07-26:sigma user-win 标记随退役消失


def test_reward_pack_v2_optout_normal_channel_lifts_adaptive_sigma_requirement():
    # 用户显式 adaptive_sigma_normal=false -> 第三通道不开,adaptive_sigma 也就不强制。
    env_cfg, _ = _apply(
        {"rewards": {"reward_pack": "v2"}, "racket": {"adaptive_sigma_normal": False}}
    )
    assert env_cfg.commands.racket_target.adaptive_sigma_normal is False
    assert env_cfg.rewards.upright_exp.weight == pytest.approx(1.0)


def test_reward_pack_v2_conflicting_motion_scale_in_window_fails_loud():
    # (d) v2 的定义就是"窗内不打折":与包同时显式配 motion_scale_in_window 拒收。
    with pytest.raises(train_mod._OverrideError, match="motion_scale_in_window"):
        _apply(_pack_v2_task(rewards_extra={"motion_scale_in_window": 0.25}))


@pytest.mark.parametrize("bad", ["V2", "V1", "v3", True, 2])
def test_reward_pack_rejects_unknown_pack_values(bad):
    # (e) 未知包值绝不静默兜底("v1" 自 2026-07-25 起是合法兜底 flag,不再在此列)。
    with pytest.raises(train_mod._OverrideError, match="reward_pack"):
        _apply({"rewards": {"reward_pack": bad}})


def test_reward_pack_v2_fails_loud_when_lineage_lacks_a_v2_term():
    # cfg 血统缺 v2 要动的项(例如没有 hit_unstable_support 声明)-> 拒绝静默半套换装。
    env_cfg = _make_env_cfg()
    env_cfg.rewards.hit_unstable_support = None
    with pytest.raises(train_mod._OverrideError, match="hit_unstable_support"):
        _apply(_pack_v2_task(), env_cfg)
    # 上台主奖同理:非 VirtualBall 谱系(没有 virtual_landing 声明)不许半套 v2。
    env_cfg = _make_env_cfg()
    env_cfg.rewards.virtual_landing = None
    with pytest.raises(train_mod._OverrideError, match="virtual_landing"):
        _apply(_pack_v2_task(), env_cfg)


def test_reward_pack_keyed_table_is_inside_reward_whitelist():
    # 键控注入表的键必须都在 _REWARD_KEYS 白名单里(train.py import 时也有同款自检)。
    strays = sorted(
        {key for key, _ in train_mod._REWARD_PACK_V2_KEYED} - set(train_mod._REWARD_KEYS)
    )
    assert strays == []
    assert "reward_pack" in train_mod._REWARD_KEYS


# --------------------------------------------------------------------------------------------- #
# B2 JOB2 — task.venue_profile(场地档案一键展开,Wave-1 loader 消费)
# --------------------------------------------------------------------------------------------- #
_FRANCO_RIG = "franco_rig_20260725"


def _franco_rig_sha8():
    import hashlib

    path = os.path.abspath(
        os.path.join(HERE, "..", "..", "..", "configs", "venue_profiles", _FRANCO_RIG + ".json")
    )
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def test_venue_profile_absent_is_byte_preserving_noop():
    env_cfg, applied = _apply({})
    assert env_cfg.events.physics_material.params["static_friction_range"] == (1.0, 1.0)
    assert env_cfg.events.randomize_link_mass.params["mass_distribution_params"] == (1.0, 1.0)
    assert env_cfg.commands.racket_target.target_noise_white == 0.0
    assert applied == []


def test_venue_profile_franco_rig_lands_all_three_sections(monkeypatch):
    fake_mdp = _stub_mdp_module(monkeypatch)
    env_cfg, applied = _apply({"venue_profile": _FRANCO_RIG})
    C = env_cfg.commands.racket_target
    # mocap_noise
    assert C.target_noise_white == pytest.approx(0.0019)
    assert C.target_noise_ar1_sigma == pytest.approx(0.0052)
    assert C.target_noise_ar1_rho == pytest.approx(0.717)
    # transport(非 live tts 模式必须触发现有翻译层的 actor 观测 func 换装副作用)
    assert C.target_delay_steps == 0
    assert C.target_dropout_prob == 0.0
    assert C.target_delay_tts_mode == "source_timestamp_compensated"
    assert env_cfg.observations.policy.time_to_strike.func is fake_mdp.actor_time_to_strike
    assert env_cfg.observations.critic.time_to_strike.func == "live_time_to_strike"
    # physics
    pm = env_cfg.events.physics_material.params
    assert pm["static_friction_range"] == (0.3, 1.6)
    assert pm["dynamic_friction_range"] == (0.3, 1.2)
    assert pm["restitution_range"] == (0.0, 0.5)
    assert env_cfg.events.randomize_link_mass.params["mass_distribution_params"] == (0.85, 1.15)
    # applied 标记带档案名 + sha 前缀,注入键走翻译层留下标准记账
    tag = f"venue_profile={_FRANCO_RIG}@{_franco_rig_sha8()}"
    assert any(tag in m and "loaded" in m for m in applied)
    assert all(tag in m for m in applied if "venue_profile=" in m)
    assert f"task.racket.target_noise_white=0.0019 ({tag})" in applied
    assert "racket_target.target_noise_white=0.0019" in applied
    assert f"events.physics_material.params.restitution_range=(0.0, 0.5) ({tag})" in applied


def test_venue_profile_explicit_user_keys_win(monkeypatch):
    _stub_mdp_module(monkeypatch)
    env_cfg, applied = _apply(
        {
            "venue_profile": _FRANCO_RIG,
            "racket": {"target_noise_white": 0.005},
            "plant": {"robot_material_static_friction_range": [0.5, 1.0]},
        }
    )
    C = env_cfg.commands.racket_target
    assert C.target_noise_white == pytest.approx(0.005)  # 显式 racket 键赢
    assert C.target_noise_ar1_sigma == pytest.approx(0.0052)  # 没写的键档案照落
    pm = env_cfg.events.physics_material.params
    assert pm["static_friction_range"] == (0.5, 1.0)  # 显式 plant 键赢
    assert pm["dynamic_friction_range"] == (0.3, 1.2)
    assert len([m for m in applied if "user override wins" in m]) == 2


def test_venue_profile_unknown_name_fails_loud():
    with pytest.raises(ValueError, match="no_such_venue"):
        _apply({"venue_profile": "no_such_venue_20990101"})


def test_venue_profile_fails_loud_when_events_knob_missing():
    env_cfg = _make_env_cfg()
    del env_cfg.events.randomize_link_mass.params["mass_distribution_params"]
    with pytest.raises(train_mod._OverrideError, match="mass_distribution_params"):
        _apply({"venue_profile": _FRANCO_RIG}, env_cfg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
