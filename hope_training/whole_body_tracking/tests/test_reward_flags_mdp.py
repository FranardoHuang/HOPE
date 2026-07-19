"""reward-flags-0709 — mdp BEHAVIOR unit tests (CPU, isaaclab STUBBED).

The mdp modules import isaaclab at the top, so this file installs a minimal isaaclab stub into
sys.modules and then loads the REAL repo modules under their canonical dotted names
(commands.py / rewards.py / terminations.py / hope_commands.py / hope_rewards.py /
hope_observations.py). What is tested is therefore the actual shipped math, not a re-derivation:

* V2  rewards._apply_window_scale through the real motion_* funcs: in-window x k, out-window x 1,
      default params = byte-identical.
* 1c  RacketTargetCommand._compute_strike_timing: split pos/wide masks; None defaults alias the
      legacy strike_window; hope_rewards racket_position gates on the TIGHT window while
      racket_velocity/racket_normal gate on the WIDE one.
* C2a hope_rewards._pos_gate: sigmoid((r - pos_err)/0.05) values at pos_err = r (0.5) and far
      out of reach (~0); None = exactly 1.0.
* B2  hope_rewards.racket_guidance: min(dist, d_max) clamp + (pre_strike | strike_window) mask.
* R-b hope_rewards.tracking_envelope_violation == the union of the two removed termination
      expressions (anchor z / any-envelope-body z over threshold).
* R-a hope_observations.generated_commands_actor_leg_masked: exactly the 24 runtime-derived leg
      dims become default-stand pos + zero vel, every other dim byte-identical to
      cat([joint_pos, joint_vel]); a wrong find_joints count refuses loudly.
* H0  HitterPure's four body-imitation terms and three reference-relative terminations are
      swing-only; held envs receive no teacher-velocity income / hidden reference death while
      absolute safety remains a separate config concern.
* A1  target position/velocity/demanded-normal/swing-sign/TTS remain one atomic actor message
      through delay/dropout; source-timestamp compensation and its stale-TTS negative control are
      distinct, and a true reset clears/backfills every stateful sensor-defect buffer.
* P2.4 base_decel is zero during a frozen hold; debug raw/gated telemetry records the real mask.
* D6 qdot-limit hinge uses realized joint velocity divided by the actual 31 runtime-ordered
      articulation limits; reordered joints and zero/non-finite limits fail closed.
* R-c commands.MotionCommand on a synthetic 2-clip npz pair: rsi_skip_settle_frames offsets the
      swing-entry frame (multiseg + single-clip clamp, incl. the short-clip clamp) and
      rsi_hold_root_stand_z rewrites ONLY the held-RSI birth root z to the default stand height.
* A8 post-swing replay activation: true-reset counts distinguish buffer-not-ready, eligible,
      random-not-selected, selected and state-write-started paths; per-update consumption resets
      exactly, and the disabled mechanism stays all-zero.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_reward_flags_mdp.py -q
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import tempfile
import types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MDP_DIR = os.path.abspath(os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking", "tasks", "tracking", "mdp"))


# --------------------------------------------------------------------------------------------- #
# isaaclab stub — the minimal surface the mdp modules import at module scope
# --------------------------------------------------------------------------------------------- #
def _module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _sample_uniform(lo, hi, size, device=None):
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=device)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=device)
    shape = tuple(size) if isinstance(size, (tuple, list, torch.Size)) else tuple(size.shape) if torch.is_tensor(size) else (int(size),)
    return lo_t + (hi_t - lo_t) * torch.rand(*shape, device=device)


def _install_isaaclab_stub():
    if "isaaclab" in sys.modules and getattr(sys.modules["isaaclab"], "_reward_flags_stub", False):
        return
    isaaclab = _module("isaaclab")
    isaaclab._reward_flags_stub = True

    assets = _module("isaaclab.assets")
    for cls_name in ("Articulation", "RigidObject", "ArticulationCfg", "AssetBaseCfg", "RigidObjectCfg"):
        setattr(assets, cls_name, type(cls_name, (), {}))
    isaaclab.assets = assets

    managers = _module("isaaclab.managers")

    class CommandTerm:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.num_envs = env.num_envs
            self.device = env.device
            self.metrics = {}

    class _KwInit:
        def __init__(self, *args, **kwargs):
            for a, name in zip(args, ("name",)):
                setattr(self, name, a)
            for k, v in kwargs.items():
                setattr(self, k, v)

    managers.CommandTerm = CommandTerm
    managers.CommandTermCfg = type("CommandTermCfg", (), {"__init__": _KwInit.__init__})
    managers.SceneEntityCfg = type("SceneEntityCfg", (_KwInit,), {})
    for cfg_name in ("RewardTermCfg", "ObservationTermCfg", "TerminationTermCfg", "EventTermCfg",
                     "ObservationGroupCfg"):
        setattr(managers, cfg_name, type(cfg_name, (_KwInit,), {}))
    isaaclab.managers = managers

    markers = _module("isaaclab.markers")

    class VisualizationMarkers:
        def __init__(self, cfg):
            self.cfg = cfg

    class _MarkerCfg:
        def __init__(self):
            self.markers = {"frame": types.SimpleNamespace(scale=None)}

        def replace(self, **kwargs):
            return copy.deepcopy(self)

    markers.VisualizationMarkers = VisualizationMarkers
    markers.VisualizationMarkersCfg = _MarkerCfg
    markers_config = _module("isaaclab.markers.config")
    markers_config.FRAME_MARKER_CFG = _MarkerCfg()
    markers.config = markers_config
    isaaclab.markers = markers

    sensors = _module("isaaclab.sensors")
    sensors.ContactSensor = type("ContactSensor", (), {})
    sensors.ContactSensorCfg = type("ContactSensorCfg", (), {})
    isaaclab.sensors = sensors

    envs = _module("isaaclab.envs")
    envs.ManagerBasedRLEnv = type("ManagerBasedRLEnv", (), {})
    envs.ManagerBasedEnv = type("ManagerBasedEnv", (), {})
    isaaclab.envs = envs

    envs_mdp = _module("isaaclab.envs.mdp")
    envs_actions = _module("isaaclab.envs.mdp.actions")
    actions_cfg = _module("isaaclab.envs.mdp.actions.actions_cfg")
    joint_actions = _module("isaaclab.envs.mdp.actions.joint_actions")

    class JointPositionAction:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._asset = env.scene[cfg.asset_name]
            self.num_envs = env.num_envs
            self.device = env.device
            self._joint_ids = slice(None)
            self._joint_names = list(self._asset.data.joint_names)
            self._raw_actions = torch.zeros(
                self.num_envs, len(self._joint_names), device=self.device
            )
            self._processed_actions = torch.zeros_like(self._raw_actions)
            self._scale = float(getattr(cfg, "scale", 1.0))
            self._offset = (
                self._asset.data.default_joint_pos.clone()
                if getattr(cfg, "use_default_offset", False)
                else 0.0
            )

        @property
        def processed_actions(self):
            return self._processed_actions

        def process_actions(self, actions):
            self._raw_actions[:] = actions
            self._processed_actions = self._raw_actions * self._scale + self._offset

        def reset(self, env_ids=None):
            self._raw_actions[env_ids] = 0.0

    actions_cfg.JointPositionActionCfg = type("JointPositionActionCfg", (), {})
    joint_actions.JointPositionAction = JointPositionAction
    envs_actions.actions_cfg = actions_cfg
    envs_actions.joint_actions = joint_actions
    envs_mdp.actions = envs_actions
    envs.mdp = envs_mdp

    utils = _module("isaaclab.utils")

    def configclass(cls):
        def __init__(self, **kwargs):  # class attributes double as defaults
            for k, v in kwargs.items():
                setattr(self, k, v)

        cls.__init__ = __init__
        return cls

    utils.configclass = configclass

    math_mod = _module("isaaclab.utils.math")
    math_mod.sample_uniform = _sample_uniform
    math_mod.quat_error_magnitude = lambda a, b: torch.norm(a - b, dim=-1)
    math_mod.quat_apply = lambda q, v: v
    math_mod.quat_rotate_inverse = lambda q, v: v
    math_mod.quat_inv = lambda q: q
    math_mod.quat_mul = lambda a, b: b.clone() if torch.is_tensor(b) else b
    math_mod.yaw_quat = lambda q: q

    def quat_from_euler_xyz(r, p, y):
        q = torch.zeros(r.shape[0], 4)
        q[:, 0] = 1.0
        return q

    math_mod.quat_from_euler_xyz = quat_from_euler_xyz
    math_mod.matrix_from_quat = lambda q: torch.eye(3, device=q.device).expand(q.shape[0], 3, 3)
    math_mod.euler_xyz_from_quat = lambda q: (torch.zeros(q.shape[0]),) * 3
    utils.math = math_mod
    isaaclab.utils = utils


def _load(dotted, filename):
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, os.path.join(MDP_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


_install_isaaclab_stub()
_PKG = "whole_body_tracking.tasks.tracking.mdp"
for _p in ("whole_body_tracking", "whole_body_tracking.tasks", "whole_body_tracking.tasks.tracking", _PKG):
    sys.modules.setdefault(_p, types.ModuleType(_p))
_load(f"{_PKG}.event_timing", "event_timing.py")
_load(f"{_PKG}.post_swing_teacher", "post_swing_teacher.py")
planner_revision_mod = _load(f"{_PKG}.planner_revision", "planner_revision.py")
commands_mod = _load(f"{_PKG}.commands", "commands.py")
rewards_mod = _load(f"{_PKG}.rewards", "rewards.py")
terminations_mod = _load(f"{_PKG}.terminations", "terminations.py")
_load(f"{_PKG}.stage1_question_bank", "stage1_question_bank.py")
hope_commands_mod = _load(f"{_PKG}.hope_commands", "hope_commands.py")
hope_rewards_mod = _load(f"{_PKG}.hope_rewards", "hope_rewards.py")
hope_observations_mod = _load(f"{_PKG}.hope_observations", "hope_observations.py")
hope_actions_mod = _load(f"{_PKG}.hope_actions", "hope_actions.py")


# --------------------------------------------------------------------------------------------- #
# shared fakes
# --------------------------------------------------------------------------------------------- #
def _fake_env(**terms):
    return types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: terms[name]))


def test_stand_start_yaw_degenerate_nonzero_range_is_applied_deterministically():
    fixed = commands_mod._stand_start_yaw_samples((0.35, 0.35), 4, "cpu")
    assert torch.equal(fixed, torch.full((4,), 0.35))
    # The legacy default remains a true no-op and consumes no random draw.
    assert commands_mod._stand_start_yaw_samples((0.0, 0.0), 4, "cpu") is None


def _yaw_quats(yaws):
    y = torch.as_tensor(yaws, dtype=torch.float32)
    q = torch.zeros(len(y), 4)
    q[:, 0] = torch.cos(0.5 * y)
    q[:, 3] = torch.sin(0.5 * y)
    return q


def _recovery_cmd(n):
    rt = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    rt.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(root_quat_w=_yaw_quats([0.0] * n))
    )
    rt._hold_edge_pending = torch.zeros(n, dtype=torch.bool)
    rt._previous_in_hold = torch.zeros(n, dtype=torch.bool)
    rt._hold_start_yaw = torch.zeros(n)
    rt._heading_expiry_sum_acc = 0.0
    rt._heading_expiry_n_acc = 0.0
    rt._recovery_spawn_sum_acc = 0.0
    rt._recovery_expiry_sum_acc = 0.0
    rt._recovery_n_acc = 0.0
    return rt


def test_recovery_edges_condition_on_spawn_yaw_and_handle_negative_wrap_yaw():
    rt = _recovery_cmd(3)
    rt.robot.data.root_quat_w = _yaw_quats([0.5, 0.1, -0.6])
    rt._update_hold_recovery_metrics(torch.tensor([True, True, True]))
    assert torch.allclose(rt._hold_start_yaw, torch.tensor([0.5, 0.1, 0.6]), atol=1e-6)

    rt.robot.data.root_quat_w = _yaw_quats([0.2, 0.05, -0.1])
    rt._update_hold_recovery_metrics(torch.tensor([False, False, False]))
    assert rt._heading_expiry_n_acc == pytest.approx(3.0)
    assert rt._heading_expiry_sum_acc == pytest.approx(0.35, abs=1e-6)
    # Only spawn yaw > 0.30 rad belongs to the conditioned recovery ledger.
    assert rt._recovery_n_acc == pytest.approx(2.0)
    assert rt._recovery_spawn_sum_acc == pytest.approx(1.1, abs=1e-6)
    assert rt._recovery_expiry_sum_acc == pytest.approx(0.3, abs=1e-6)


def test_recovery_reset_while_held_restarts_edge_and_zero_length_hold_is_not_counted():
    rt = _recovery_cmd(1)
    rt.robot.data.root_quat_w = _yaw_quats([0.8])
    rt._update_hold_recovery_metrics(torch.tensor([True]))

    # A reset/wrap replaces the held state without a False edge. Pending must discard the old
    # 0.8-rad spawn and stamp the new 0.4-rad state, not book a false expiry.
    rt._hold_edge_pending[:] = True
    rt._previous_in_hold[:] = False
    rt._hold_start_yaw[:] = 0.0
    rt.robot.data.root_quat_w = _yaw_quats([0.4])
    rt._update_hold_recovery_metrics(torch.tensor([True]))
    assert rt._hold_start_yaw.item() == pytest.approx(0.4, abs=1e-6)
    assert rt._recovery_n_acc == 0.0

    rt.robot.data.root_quat_w = _yaw_quats([0.1])
    rt._update_hold_recovery_metrics(torch.tensor([False]))
    assert rt._recovery_n_acc == pytest.approx(1.0)
    assert rt._recovery_spawn_sum_acc == pytest.approx(0.4, abs=1e-6)
    assert rt._recovery_expiry_sum_acc == pytest.approx(0.1, abs=1e-6)

    before = (rt._heading_expiry_n_acc, rt._recovery_n_acc)
    rt._hold_edge_pending[:] = True
    rt._previous_in_hold[:] = False
    rt._update_hold_recovery_metrics(torch.tensor([False]))
    assert (rt._heading_expiry_n_acc, rt._recovery_n_acc) == before


def _fake_racket_cmd(n=4, window=None, window_pos=None, window_wide=None, pre_strike=None):
    window = torch.zeros(n, dtype=torch.bool) if window is None else window
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(face_command=False, debug_reward_logging=False),
        num_envs=n, device="cpu", metrics={},
        strike_window=window,
        strike_window_pos=window if window_pos is None else window_pos,
        strike_window_wide=window if window_wide is None else window_wide,
        pre_strike=torch.zeros(n, dtype=torch.bool) if pre_strike is None else pre_strike,
        time_to_strike=torch.zeros(n),
        racket_pos_w=torch.zeros(n, 3),
        racket_target_pos_w=torch.zeros(n, 3),
        racket_lin_vel_w=torch.zeros(n, 3),
        racket_target_vel_w=torch.zeros(n, 3),
        racket_normal_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).clone(),
        # raw (+Y/A-frame) twin — face_command 通道读它(hope_rewards._face_pair);无符号表的
        # 环境里 raw ≡ signed,fake 保持同值同语义。
        racket_normal_raw_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).clone(),
        racket_target_normal_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).clone(),
        target_normal_cmd=torch.zeros(n, 3),
    )


# --------------------------------------------------------------------------------------------- #
# V2 in-window imitation yield (rewards._apply_window_scale through the real funcs)
# --------------------------------------------------------------------------------------------- #
def _fake_motion_for_rewards(n=4, n_bodies=2):
    zeros_b = torch.zeros(n, n_bodies, 3)
    quat = torch.zeros(n, n_bodies, 4)
    quat[..., 0] = 1.0
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            body_names=["torso_Link", "right_wrist_yaw_Link"],
            v1_free_wrist_vel_mimic_activation=False,
            v2_motion_scale_in_window_activation=None,
        ),
        anchor_pos_w=torch.zeros(n, 3), robot_anchor_pos_w=torch.zeros(n, 3),
        anchor_quat_w=torch.zeros(n, 4), robot_anchor_quat_w=torch.zeros(n, 4),
        body_pos_relative_w=zeros_b.clone(), robot_body_pos_w=zeros_b.clone(),
        body_quat_relative_w=quat.clone(), robot_body_quat_w=quat.clone(),
        body_lin_vel_w=zeros_b.clone(), robot_body_lin_vel_w=zeros_b.clone(),
        body_ang_vel_w=zeros_b.clone(), robot_body_ang_vel_w=zeros_b.clone(),
        in_hold=torch.zeros(n, dtype=torch.bool),
        record_v1_velocity_mimic_activation=lambda *args, **kwargs: None,
        record_v2_strike_window_scale_activation=lambda *args, **kwargs: None,
    )


def _activation_motion_for_rewards(n=4, *, v1=False, v2_scale=None):
    motion = _fake_motion_for_rewards(n)
    motion.cfg.v1_free_wrist_vel_mimic_activation = v1
    motion.cfg.v2_motion_scale_in_window_activation = v2_scale
    motion._post_swing_activation_counters = {}
    motion._reward_activation_counters = {
        name: torch.zeros((), dtype=torch.long)
        for name in (
            "v1_velocity_mimic_eligible_sample_count",
            "v1_held_wrist_excluded_sample_count",
            "v2_strike_window_eligible_imitation_sample_count",
            "v2_quarter_scaled_strike_window_imitation_sample_count",
        )
    }
    motion.record_v1_velocity_mimic_activation = types.MethodType(
        commands_mod.MotionCommand.record_v1_velocity_mimic_activation, motion
    )
    motion.record_v2_strike_window_scale_activation = types.MethodType(
        commands_mod.MotionCommand.record_v2_strike_window_scale_activation,
        motion,
    )
    motion.consume_training_activation_counters = types.MethodType(
        commands_mod.MotionCommand.consume_training_activation_counters, motion
    )
    return motion


def test_v2_window_scale_scales_only_windowed_envs():
    n = 4
    win = torch.tensor([True, False, True, False])
    racket = _fake_racket_cmd(n, window=torch.zeros(n, dtype=torch.bool), window_wide=win)
    env = _fake_env(motion=_fake_motion_for_rewards(n), racket_target=racket)
    for func in (
        rewards_mod.motion_global_anchor_position_error_exp,
        rewards_mod.motion_global_anchor_orientation_error_exp,
    ):
        r = func(env, "motion", std=1.0, window_scale=0.25, window_command_name="racket_target")
        assert torch.allclose(r, torch.tensor([0.25, 1.0, 0.25, 1.0])), func.__name__
    for func in (
        rewards_mod.motion_relative_body_position_error_exp,
        rewards_mod.motion_relative_body_orientation_error_exp,
        rewards_mod.motion_global_body_linear_velocity_error_exp,
        rewards_mod.motion_global_body_angular_velocity_error_exp,
    ):
        r = func(env, "motion", std=1.0, body_names=None,
                 window_scale=0.25, window_command_name="racket_target")
        assert torch.allclose(r, torch.tensor([0.25, 1.0, 0.25, 1.0])), func.__name__


def test_v2_default_params_are_byte_identical():
    n = 3
    env = _fake_env(motion=_fake_motion_for_rewards(n))  # no racket cmd needed on the default path
    r_default = rewards_mod.motion_global_body_linear_velocity_error_exp(env, "motion", std=1.0)
    assert torch.allclose(r_default, torch.ones(n))
    # k=1.0 short-circuits before the command lookup too
    r_k1 = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env, "motion", std=1.0, window_scale=1.0, window_command_name="racket_target")
    assert torch.equal(r_default, r_k1)


def test_v2_uses_wide_window_and_forwards_through_swing_only_wrappers():
    n = 2
    motion = _fake_motion_for_rewards(n)
    wide = torch.tensor([True, False])
    racket = _fake_racket_cmd(n, window=torch.zeros(n, dtype=torch.bool), window_wide=wide)
    env = _fake_env(motion=motion, racket_target=racket)
    r = hope_rewards_mod.motion_body_pos_swing_only(
        env, "motion", std=1.0, body_names=None, window_scale=0.0, window_command_name="racket_target")
    assert torch.allclose(r, torch.tensor([0.0, 1.0]))  # teacher fully silent in-window
    motion.in_hold = torch.tensor([False, True])  # hold gate still wins
    r = hope_rewards_mod.motion_body_ori_swing_only(
        env, "motion", std=1.0, body_names=None, window_scale=0.0, window_command_name="racket_target")
    assert torch.allclose(r, torch.tensor([0.0, 0.0]))


def test_v1_execution_counts_multi_env_samples_and_resolved_wrist_exclusion():
    motion = _activation_motion_for_rewards(3, v1=True)
    env = _fake_env(motion=motion)
    kept_bodies = ["torso_Link"]

    first = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env, "motion", std=1.0, body_names=kept_bodies
    )
    second = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env, "motion", std=1.0, body_names=kept_bodies
    )
    assert torch.equal(first, torch.ones(3))
    assert torch.equal(second, first)

    snapshot = motion.consume_training_activation_counters()
    assert snapshot["v1_velocity_mimic_eligible_sample_count"].item() == 6
    assert snapshot["v1_held_wrist_excluded_sample_count"].item() == 6
    assert snapshot["v1_velocity_mimic_eligible_sample_count"].dtype == torch.long
    assert all(value.item() == 0 for value in motion._reward_activation_counters.values())


def test_v1_counterexample_keeps_denominator_when_wrist_was_not_excluded():
    motion = _activation_motion_for_rewards(4, v1=True)
    env = _fake_env(motion=motion)
    rewards_mod.motion_global_body_linear_velocity_error_exp(
        env,
        "motion",
        std=1.0,
        body_names=["torso_Link", "right_wrist_yaw_Link"],
    )
    snapshot = motion.consume_training_activation_counters()
    assert snapshot["v1_velocity_mimic_eligible_sample_count"].item() == 4
    assert snapshot["v1_held_wrist_excluded_sample_count"].item() == 0


def test_v2_execution_counts_each_real_windowed_reward_application():
    motion = _activation_motion_for_rewards(4, v2_scale=0.25)
    wide = torch.tensor([True, False, True, False])
    racket = _fake_racket_cmd(
        4, window=torch.zeros(4, dtype=torch.bool), window_wide=wide
    )
    env = _fake_env(motion=motion, racket_target=racket)

    anchor = rewards_mod.motion_global_anchor_position_error_exp(
        env,
        "motion",
        std=1.0,
        window_scale=0.25,
        window_command_name="racket_target",
    )
    velocity = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env,
        "motion",
        std=1.0,
        body_names=["torso_Link"],
        window_scale=0.25,
        window_command_name="racket_target",
    )
    expected = torch.tensor([0.25, 1.0, 0.25, 1.0])
    assert torch.equal(anchor, expected)
    assert torch.equal(velocity, expected)

    snapshot = motion.consume_training_activation_counters()
    # Two imitation terms x two in-window env samples.  This deliberately measures reward
    # applications, not unique environment steps.
    assert snapshot["v2_strike_window_eligible_imitation_sample_count"].item() == 4
    assert snapshot[
        "v2_quarter_scaled_strike_window_imitation_sample_count"
    ].item() == 4


def test_v2_counterexample_records_window_but_not_wrong_scale_as_quarter():
    motion = _activation_motion_for_rewards(4, v2_scale=0.25)
    wide = torch.tensor([True, False, True, False])
    env = _fake_env(
        motion=motion,
        racket_target=_fake_racket_cmd(4, window_wide=wide),
    )
    result = rewards_mod.motion_global_anchor_position_error_exp(
        env,
        "motion",
        std=1.0,
        window_scale=0.5,
        window_command_name="racket_target",
    )
    assert torch.equal(result, torch.tensor([0.5, 1.0, 0.5, 1.0]))
    snapshot = motion.consume_training_activation_counters()
    assert snapshot["v2_strike_window_eligible_imitation_sample_count"].item() == 2
    assert snapshot[
        "v2_quarter_scaled_strike_window_imitation_sample_count"
    ].item() == 0


def test_v1_v2_disabled_ledgers_are_zero_without_rng_or_reward_change(monkeypatch):
    motion = _activation_motion_for_rewards(3, v1=False, v2_scale=None)
    env = _fake_env(motion=motion)

    def _unexpected_rng(*args, **kwargs):
        raise AssertionError("disabled V1/V2 reward instrumentation must not sample RNG")

    monkeypatch.setattr(torch, "rand", _unexpected_rng)
    result = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env, "motion", std=1.0, body_names=["torso_Link"]
    )
    assert torch.equal(result, torch.ones(3))
    snapshot = motion.consume_training_activation_counters()
    assert all(value.item() == 0 for value in snapshot.values())


def test_hitter_pure_velocity_imitation_is_swing_only():
    motion = _fake_motion_for_rewards(3)
    motion.in_hold = torch.tensor([False, True, False])
    # Make env2 non-perfect so the test proves normal swing reward math is preserved rather
    # than all rows coincidentally returning one.
    motion.robot_body_lin_vel_w[2, :, 0] = 1.0
    motion.robot_body_ang_vel_w[2, :, 1] = 1.0
    env = _fake_env(motion=motion)
    lin = hope_rewards_mod.motion_body_lin_vel_swing_only(env, "motion", std=1.0)
    ang = hope_rewards_mod.motion_body_ang_vel_swing_only(env, "motion", std=1.0)
    expected = torch.tensor([1.0, 0.0, torch.exp(torch.tensor(-1.0))])
    assert torch.allclose(lin, expected)
    assert torch.allclose(ang, expected)


def test_motion_in_hold_keeps_the_final_post_decrement_step_held():
    cmd = commands_mod.MotionCommand.__new__(commands_mod.MotionCommand)
    cmd.hold_counter = torch.tensor([1, 0], dtype=torch.long)
    cmd.metrics = {"in_hold": torch.zeros(2)}
    held = cmd.hold_counter > 0
    cmd.hold_counter = torch.clamp(cmd.hold_counter - 1, min=0)
    cmd.metrics["in_hold"] = held.float()
    # Env0's clock was frozen on this step, so its teacher/termination contract must still be
    # held even though the remaining counter is now zero.
    assert cmd.in_hold.tolist() == [True, False]
    cmd.metrics["in_hold"].zero_()  # next control step releases normally
    assert cmd.in_hold.tolist() == [False, False]


def test_reference_envelope_terminations_ignore_hold_only_when_explicit():
    n = 3
    body_names = ["left_wrist_yaw_link"]
    motion = types.SimpleNamespace(
        cfg=types.SimpleNamespace(body_names=body_names),
        anchor_pos_w=torch.zeros(n, 3),
        robot_anchor_pos_w=torch.zeros(n, 3),
        body_pos_relative_w=torch.zeros(n, 1, 3),
        robot_body_pos_w=torch.zeros(n, 1, 3),
        in_hold=torch.tensor([True, False, True]),
    )
    # Every env violates both reference envelopes. The hold-aware wrappers must retain only
    # env1; ignore_hold=False must be exactly the original behavior.
    motion.anchor_pos_w[:, 2] = 0.5
    motion.body_pos_relative_w[:, 0, 2] = 0.5
    env = _fake_env(motion=motion)
    for wrapped, raw in (
        (hope_rewards_mod.bad_anchor_pos_z_only_hold_aware,
         terminations_mod.bad_anchor_pos_z_only),
        (hope_rewards_mod.bad_motion_body_pos_z_only_hold_aware,
         terminations_mod.bad_motion_body_pos_z_only),
    ):
        kwargs = {"body_names": body_names} if "body_pos" in wrapped.__name__ else {}
        assert wrapped(env, "motion", 0.25, ignore_hold=True, **kwargs).tolist() == [False, True, False]
        assert torch.equal(
            wrapped(env, "motion", 0.25, ignore_hold=False, **kwargs),
            raw(env, "motion", 0.25, **kwargs),
        )


def test_ignore_hold_contract_fails_loud_without_mask():
    motion = types.SimpleNamespace(
        cfg=types.SimpleNamespace(body_names=[]),
        anchor_pos_w=torch.zeros(1, 3), robot_anchor_pos_w=torch.zeros(1, 3),
    )
    with pytest.raises(RuntimeError, match="in_hold"):
        hope_rewards_mod.bad_anchor_pos_z_only_hold_aware(
            _fake_env(motion=motion), "motion", 0.25, ignore_hold=True
        )


def _torque_cmd(tau=None, limits=None, indices=(0, 1)):
    robot = types.SimpleNamespace(
        find_joints=lambda expr: (list(indices), None),
        data=types.SimpleNamespace(computed_torque=tau, joint_effort_limits=limits),
    )
    return types.SimpleNamespace(
        robot=robot, num_envs=2, device="cpu", metrics={},
    )


def test_arm_torque_saturation_uses_heterogeneous_joint_limits():
    # Models the important A3 asymmetry directly: 24 Nm elbow/yaw vs 60 Nm shoulder pitch/roll.
    tau = torch.tensor([[24.0, 120.0], [48.0, 30.0]])
    limits = torch.tensor([[24.0, 60.0], [24.0, 60.0]])
    cmd = _torque_cmd(tau, limits)
    value = hope_rewards_mod.arm_torque_saturation(_fake_env(racket_target=cmd), "racket_target")
    assert torch.allclose(value, torch.tensor([0.5, 0.5]))
    assert torch.equal(cmd.metrics["arm_torque_sat_frac"], value)


@pytest.mark.parametrize(
    "cmd,match",
    [
        (_torque_cmd(None, None), "requires robot.data.computed_torque"),
        (_torque_cmd(torch.zeros(2, 2), torch.tensor([[24.0, float("nan")]] * 2)),
         "non-finite"),
        (_torque_cmd(torch.zeros(2, 2), torch.tensor([[24.0, 0.0]] * 2)),
         "strictly positive"),
        (_torque_cmd(torch.zeros(2, 2), torch.ones(2, 2), indices=()),
         "resolved zero"),
    ],
)
def test_arm_torque_saturation_fails_loud_when_mechanism_unavailable(cmd, match):
    with pytest.raises(RuntimeError, match=match):
        hope_rewards_mod.arm_torque_saturation(
            _fake_env(racket_target=cmd), "racket_target"
        )


# --------------------------------------------------------------------------------------------- #
# 1c split windows — timing masks + per-channel gating
# --------------------------------------------------------------------------------------------- #
def _timing_cmd(time_steps, pos_s=None, wide_s=None, window_s=0.12):
    rt = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    rt.cfg = types.SimpleNamespace(
        strike_phase=0.5, strike_phase_per_clip=(), strike_window_s=window_s,
        strike_window_pos_s=pos_s, strike_window_wide_s=wide_s)
    rt._env = types.SimpleNamespace(step_dt=0.02)
    fake_motion = types.SimpleNamespace(
        _multiseg=False, motion=types.SimpleNamespace(time_step_total=101),
        retiming_active=False, time_steps=torch.tensor(time_steps))
    rt._motion = lambda: fake_motion
    rt._strike_phase_per_clip_t = None
    rt._compute_strike_timing()
    return rt


def test_1c_split_masks():
    # strike step = round(0.5 * 100) = 50; tts = (50 - ts) * 0.02
    rt = _timing_cmd([50, 49, 45, 30], pos_s=0.02, wide_s=0.10)
    assert rt.strike_window_pos.tolist() == [True, True, False, False]     # |tts| <= 0.02
    assert rt.strike_window_wide.tolist() == [True, True, True, False]     # |tts| <= 0.10
    assert rt.strike_window.tolist() == [True, True, True, False]          # base 0.12 window


def test_1c_default_masks_alias_legacy_window():
    rt = _timing_cmd([50, 49, 45, 30])
    assert rt.strike_window_pos is rt.strike_window
    assert rt.strike_window_wide is rt.strike_window


def _timing_cmd_multiseg(spc):
    """Multiseg twin of _timing_cmd: 2 segments (20 + 15 frames), per-clip strike phase table."""
    rt = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    rt.cfg = types.SimpleNamespace(
        strike_phase=0.5, strike_phase_per_clip=spc, strike_window_s=0.12,
        strike_window_pos_s=None, strike_window_wide_s=None)
    rt._env = types.SimpleNamespace(step_dt=0.02)
    rt.device = "cpu"
    rt._strike_phase_per_clip_t = None
    fake_ml = types.SimpleNamespace(
        num_segments=2, seg_start=torch.tensor([0, 20]), seg_len=torch.tensor([20, 15]))
    fake_motion = types.SimpleNamespace(
        _multiseg=True, motion=fake_ml, retiming_active=False,
        time_steps=torch.tensor([0, 25]), clip_id=torch.tensor([0, 1]))
    rt._motion = lambda: fake_motion
    return rt


def test_strike_phase_per_clip_length_mismatch_fails_loud():
    """人话:每 clip 击球点表和加载的 clip 数对不上,必须当场报错——以前会悄悄退回全局 strike_phase,
    每个 env 都在错误帧上找击球点还不吭声(fail-loud 文化,取证副产品 2026-07-09)。"""
    rt = _timing_cmd_multiseg((0.4, 0.5, 0.6))  # 3 entries, 2 loaded segments
    with pytest.raises(ValueError, match="strike_phase_per_clip"):
        rt._compute_strike_timing()
    with pytest.raises(ValueError, match="strike_phase_per_clip"):
        rt._strike_frame_for_clip(rt._motion().motion, 0)


def test_strike_phase_per_clip_empty_falls_back_and_matched_table_is_used():
    rt = _timing_cmd_multiseg(())  # () stays the documented "use global strike_phase" default
    rt._compute_strike_timing()
    assert torch.allclose(rt._strike_phase_per_clip_t, torch.tensor([0.5, 0.5]))
    step, phase, seg_start, seg_len = rt._strike_frame_for_clip(rt._motion().motion, 1)
    assert (step, phase, seg_start, seg_len) == (20 + round(0.5 * 14), 0.5, 20, 15)
    rt2 = _timing_cmd_multiseg((0.4, 0.6))  # matched table is taken verbatim
    rt2._compute_strike_timing()
    assert torch.allclose(rt2._strike_phase_per_clip_t, torch.tensor([0.4, 0.6]))
    step2, phase2, _, _ = rt2._strike_frame_for_clip(rt2._motion().motion, 0)
    assert (step2, phase2) == (round(0.4 * 19), 0.4)


def test_1c_rewards_gate_per_channel():
    n = 3
    pos_win = torch.tensor([True, False, False])
    wide_win = torch.tensor([True, True, False])
    cmd = _fake_racket_cmd(n, window=wide_win, window_pos=pos_win, window_wide=wide_win)
    env = _fake_env(racket_target=cmd)
    rp = hope_rewards_mod.racket_position_tracking_exp(env, "racket_target", std=0.2)
    rv = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0)
    rn = hope_rewards_mod.racket_normal_tracking_exp(env, "racket_target", std=0.3)
    assert torch.allclose(rp, torch.tensor([1.0, 0.0, 0.0]))   # tight window
    assert torch.allclose(rv, torch.tensor([1.0, 1.0, 0.0]))   # wide window
    assert torch.allclose(rn, torch.tensor([1.0, 1.0, 0.0]))   # wide window
    # multiplicative success = the LEGACY window (R3b fix: the bonus channel must NOT be narrowed
    # to the window intersection when the split windows are active). Here strike_window == wide_win.
    rs = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.3)
    assert torch.allclose(rs, torch.tensor([1.0, 1.0, 0.0]))


def test_1c_strike_success_keeps_legacy_window_not_intersection():
    """R3b zero-return regression (2026-07-08): with split windows on, racket_strike_success must pay
    across the FULL legacy strike window — not just the ±0.02 s tight intersection. Before the fix the
    expected value below read [1, 0, 0, 0] (support collapsed to the tight window)."""
    n = 4
    legacy = torch.tensor([True, True, True, False])     # strike_window_s (e.g. ±0.12 s)
    pos_win = torch.tensor([True, False, False, False])  # strike_window_pos_s = 0.02
    wide_win = torch.tensor([True, True, False, False])  # strike_window_wide_s = 0.10
    cmd = _fake_racket_cmd(n, window=legacy, window_pos=pos_win, window_wide=wide_win)
    env = _fake_env(racket_target=cmd)
    rs = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.3)
    assert torch.allclose(rs, torch.tensor([1.0, 1.0, 1.0, 0.0]))
    # the additive channels keep their own (narrower) windows — the fix must not widen them
    rp = hope_rewards_mod.racket_position_tracking_exp(env, "racket_target", std=0.2)
    rv = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0)
    assert torch.allclose(rp, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(rv, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_1c_strike_success_default_path_byte_identical_and_kernels_match():
    """No split windows (pos/wide alias the legacy window): the fixed product must equal the OLD
    semantics rp*rv*rn exactly, including nonzero kernel values and the face_command re-anchor."""
    n = 2
    win = torch.tensor([True, False])
    cmd = _fake_racket_cmd(n, window=win)
    # nonzero errors so the kernels are not trivially 1
    cmd.racket_pos_w = torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    cmd.racket_lin_vel_w = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cmd.racket_normal_w = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cmd.racket_normal_raw_w = cmd.racket_normal_w.clone()  # 无符号表:raw ≡ signed
    cmd.racket_target_normal_w = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    env = _fake_env(racket_target=cmd)
    rp = hope_rewards_mod.racket_position_tracking_exp(env, "racket_target", std=0.2)
    rv = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0)
    rn = hope_rewards_mod.racket_normal_tracking_exp(env, "racket_target", std=0.3)
    rs = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.3)
    assert torch.allclose(rs, rp * rv * rn)
    assert rs[0].item() > 0.0 and rs[1].item() == 0.0
    # face_command=True: the product's normal factor re-anchors to target_normal_cmd
    cmd.cfg.face_command = True
    cmd.target_normal_cmd = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])  # aligned -> kernel 1
    rs_fc = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.3)
    assert torch.allclose(rs_fc, rp * rv)  # normal factor == 1 when re-anchored and aligned


# --------------------------------------------------------------------------------------------- #
# C2a proximity power-gate
# --------------------------------------------------------------------------------------------- #
def test_pos_gate_sigmoid_values_and_default():
    n = 3
    win = torch.ones(n, dtype=torch.bool)
    cmd = _fake_racket_cmd(n, window=win)
    cmd.racket_pos_w = torch.tensor([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.65, 0.0, 0.0]])
    env = _fake_env(racket_target=cmd)
    r = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0,
                                                      pos_gate_radius=0.15)
    assert abs(r[0].item() - torch.sigmoid(torch.tensor(3.0)).item()) < 1e-6  # in reach: ~0.95
    assert abs(r[1].item() - 0.5) < 1e-6                                       # at the gate edge
    assert r[2].item() < 1e-4                                                  # far out of reach
    # default = exactly ungated
    r_off = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0)
    assert torch.allclose(r_off, torch.ones(n))
    # normal channel gets the same gate; position never does
    rn = hope_rewards_mod.racket_normal_tracking_exp(env, "racket_target", std=0.3,
                                                     pos_gate_radius=0.15)
    assert abs(rn[1].item() - 0.5) < 1e-6
    rp = hope_rewards_mod.racket_position_tracking_exp(env, "racket_target", std=10.0)
    assert rp[2].item() > 0.99  # position channel keeps paying at distance (reach gradient)


# --------------------------------------------------------------------------------------------- #
# B2 constant guidance
# --------------------------------------------------------------------------------------------- #
def test_racket_guidance_clamp_and_mask():
    n = 4
    cmd = _fake_racket_cmd(
        n,
        window=torch.tensor([False, False, True, False]),
        pre_strike=torch.tensor([True, True, False, False]),
    )
    cmd.racket_pos_w = torch.tensor(
        [[0.3, 0.0, 0.0], [0.9, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
    env = _fake_env(racket_target=cmd)
    r = hope_rewards_mod.racket_guidance(env, "racket_target", d_max=0.5)
    # env0 pre-strike: dist 0.3; env1 pre-strike far: clamped 0.5; env2 in-window: 0.2;
    # env3 post-strike/post-window: unpaid (follow-through untouched)
    assert torch.allclose(r, torch.tensor([0.3, 0.5, 0.2, 0.0]))


def test_racket_face_guidance_clamp_mask_and_target_selection():
    """Face-angle guidance (M3c 死区解药): min(angle, theta_max) × (pre_strike|window) mask;
    face_command=True 时目标取需求面(target_normal_cmd),False 时取 clip 参考面。"""
    import math
    n = 4
    cmd = _fake_racket_cmd(
        n,
        window=torch.tensor([False, False, True, False]),
        pre_strike=torch.tensor([True, True, False, False]),
    )
    # achieved normals: 0° / 180°(反面,应被 theta_max=pi/2 截断)/ 90° / 90°(mask 外不付钱)
    cmd.racket_normal_w = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cmd.racket_normal_raw_w = cmd.racket_normal_w.clone()  # 无符号表:raw ≡ signed(_face_pair 读 raw)
    cmd.target_normal_cmd = torch.tensor([[1.0, 0.0, 0.0]] * n)
    cmd.racket_target_normal_w = torch.tensor([[0.0, 0.0, 1.0]] * n)
    cmd.cfg.face_command = True
    env = _fake_env(racket_target=cmd)
    r = hope_rewards_mod.racket_face_guidance(env, "racket_target", theta_max=math.pi / 2)
    assert torch.allclose(
        r, torch.tensor([0.0, math.pi / 2, math.pi / 2, 0.0]), atol=1e-5), r
    # face_command=False → 换 clip 参考面为目标:env0 的 achieved [1,0,0] vs [0,0,1] = 90°
    cmd.cfg.face_command = False
    r2 = hope_rewards_mod.racket_face_guidance(env, "racket_target", theta_max=math.pi / 2)
    assert abs(float(r2[0]) - math.pi / 2) < 1e-5, r2


def test_conditional_face_guidance_has_readiness_preserving_fixed_budget():
    """The term is [0,1], window-only, with no face gradient outside either readiness margin."""
    import math

    n = 7
    win = torch.tensor([True, True, True, True, False, True, True])
    cmd = _fake_racket_cmd(n, window=win, window_wide=win)
    cmd.cfg.face_command = True
    cmd.target_normal_cmd = torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3).clone()
    angles = torch.tensor([
        math.pi,
        0.1,
        math.pi,
        math.pi,
        math.pi,
        (0.262 + math.pi) / 2.0,
        math.pi,
    ])
    cmd.racket_normal_raw_w = torch.stack(
        [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
    ).detach().requires_grad_()
    cmd.racket_normal_w = cmd.racket_normal_raw_w.clone()

    # env2 is exactly at the 9.5cm outer position margin; env5 halfway through both compact
    # readiness ramps; env6 proves position is measured against the swing-through target-now.
    cmd.racket_pos_w[2, 0] = 0.095
    cmd.racket_pos_w[5, 0] = 0.085
    cmd.racket_lin_vel_w[3, 0] = 1.0
    cmd.racket_lin_vel_w[5, 0] = 0.75
    cmd.time_to_strike[6] = 0.1
    cmd.racket_target_vel_w[6, 0] = 1.0
    cmd.racket_lin_vel_w[6, 0] = 1.0
    cmd.racket_pos_w[6, 0] = -0.1

    r = hope_rewards_mod.racket_face_conditional_guidance(_fake_env(racket_target=cmd), "racket_target")
    # Within the time window an unready state keeps the fixed cost 1 instead of escaping the face
    # penalty.  A fully ready state pays only face_fraction; the halfway-ready/half-face case is .875.
    assert torch.allclose(r, torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.875, 1.0]), atol=1e-5), r
    assert torch.all((r >= 0.0) & (r <= 1.0))
    assert torch.allclose(
        cmd.metrics["face_conditional_guidance_gate"],
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.25, 1.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        cmd.metrics["face_conditional_guidance_error_fraction"],
        torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.5, 1.0]),
        atol=1e-5,
    )
    assert torch.allclose(cmd.metrics["face_conditional_guidance_cost_fraction"], r)
    r.sum().backward()
    assert torch.isfinite(cmd.racket_normal_raw_w.grad).all()
    # Below-angle-floor and every compact-gate-zero row have no face gradient to compete with
    # position/velocity acquisition.  Exact 180 degrees is finite (its turn axis is ambiguous).
    assert torch.equal(
        cmd.racket_normal_raw_w.grad[[1, 2, 3, 4]],
        torch.zeros_like(cmd.racket_normal_raw_w.grad[[1, 2, 3, 4]]),
    )


def test_conditional_face_guidance_never_rewards_abandoning_readiness():
    """The scalar reward, not merely autograd, must prefer .094m/.94mps over escaping the outer gate."""
    errors = torch.tensor([0.070, 0.075, 0.085, 0.094, 0.095, 0.096])

    for dimension in ("position", "velocity"):
        cmd = _fake_racket_cmd(len(errors), window=torch.ones(len(errors), dtype=torch.bool))
        cmd.cfg.face_command = True
        cmd.target_normal_cmd = torch.tensor([[1.0, 0.0, 0.0]]).expand(len(errors), 3).clone()
        angle = torch.full((len(errors),), math.pi / 2.0)
        cmd.racket_normal_raw_w = torch.stack(
            [torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1
        )
        cmd.racket_normal_w = cmd.racket_normal_raw_w.clone()
        if dimension == "position":
            cmd.racket_pos_w[:, 0] = errors
        else:
            # Map the same ordered fractions across the 0.5 -> 1.0 m/s velocity readiness band.
            cmd.racket_lin_vel_w[:, 0] = (errors - 0.075) / 0.02 * 0.5 + 0.5

        cost = hope_rewards_mod.racket_face_conditional_guidance(
            _fake_env(racket_target=cmd), "racket_target"
        )
        weighted_reward = -0.4 * cost
        # As either error worsens, cost may rise but must never fall; a negative weight therefore
        # never makes a deliberately less-ready state preferable.
        assert torch.all(cost[1:] >= cost[:-1] - 1e-7), (dimension, cost)
        assert torch.all(weighted_reward[:-1] >= weighted_reward[1:] - 1e-7), (
            dimension,
            weighted_reward,
        )
        assert weighted_reward[3] > weighted_reward[5], (dimension, weighted_reward)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"theta_free": math.pi}, "theta_free < theta_max"),
        ({"pos_full": 0.1, "pos_zero": 0.095}, "pos_full < pos_zero"),
        ({"vel_full": 1.0, "vel_zero": 1.0}, "vel_full < vel_zero"),
        ({"pos_zero": float("nan")}, "finite bounds"),
    ],
)
def test_conditional_face_guidance_rejects_invalid_source_bounds(kwargs, message):
    cmd = _fake_racket_cmd(1, window=torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError, match=message):
        hope_rewards_mod.racket_face_conditional_guidance(
            _fake_env(racket_target=cmd), "racket_target", **kwargs
        )


# --------------------------------------------------------------------------------------------- #
# R-b envelope violation indicator
# --------------------------------------------------------------------------------------------- #
def test_tracking_envelope_violation_matches_removed_terminations():
    n = 3
    body_names = ["left_ankle_roll_Link", "right_ankle_roll_Link",
                  "left_wrist_yaw_Link", "right_wrist_yaw_Link"]
    all_bodies = ["pelvis_link"] + body_names
    nb = len(all_bodies)
    motion = types.SimpleNamespace(
        cfg=types.SimpleNamespace(body_names=all_bodies),
        anchor_pos_w=torch.zeros(n, 3), robot_anchor_pos_w=torch.zeros(n, 3),
        body_pos_relative_w=torch.zeros(n, nb, 3), robot_body_pos_w=torch.zeros(n, nb, 3),
    )
    motion.anchor_pos_w[0, 2] = 0.30              # env0: anchor z blows the 0.25 envelope
    motion.body_pos_relative_w[1, 2, 2] = 0.30    # env1: one envelope body z blows it
    motion.body_pos_relative_w[2, 0, 2] = 0.30    # env2: pelvis is NOT in the envelope list
    env = _fake_env(motion=motion)
    viol = hope_rewards_mod.tracking_envelope_violation(env, "motion", 0.25, body_names)
    assert viol.tolist() == [1.0, 1.0, 0.0]
    # exact agreement with the termination funcs it replaces
    t = terminations_mod.bad_anchor_pos_z_only(env, "motion", 0.25) | \
        terminations_mod.bad_motion_body_pos_z_only(env, "motion", 0.25, body_names)
    assert torch.equal(viol, t.float())
    motion.in_hold = torch.tensor([True, False, True])
    hold_aware = hope_rewards_mod.tracking_envelope_violation(
        env, "motion", 0.25, body_names, ignore_hold=True
    )
    assert hold_aware.tolist() == [0.0, 1.0, 0.0]


# --------------------------------------------------------------------------------------------- #
# R-a actor leg-reference mask
# --------------------------------------------------------------------------------------------- #
_A3_JOINTS = [
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
]  # 31 joints, A3-style interleaved order


class _MaskRobot:
    def __init__(self, num_envs, joint_names):
        self.joint_names = list(joint_names)
        gen = torch.Generator().manual_seed(7)
        self.data = types.SimpleNamespace(
            default_joint_pos=torch.randn(num_envs, len(joint_names), generator=gen))

    def find_joints(self, exprs):
        ids, names = [], []
        for i, n in enumerate(self.joint_names):
            if any(re.fullmatch(e, n) for e in exprs):
                ids.append(i)
                names.append(n)
        return ids, names


def _mask_cmd(n=3, joints=_A3_JOINTS):
    gen = torch.Generator().manual_seed(11)
    nj = len(joints)
    return types.SimpleNamespace(
        joint_pos=torch.randn(n, nj, generator=gen),
        joint_vel=torch.randn(n, nj, generator=gen),
        robot=_MaskRobot(n, joints),
        device="cpu",
    )


def test_leg_mask_masks_exactly_24_dims():
    n = 3
    cmd = _mask_cmd(n)
    env = _fake_env(motion=cmd)
    out = hope_observations_mod.generated_commands_actor_leg_masked(env, "motion")
    nj = len(_A3_JOINTS)
    assert out.shape == (n, 2 * nj) == (n, 62)
    leg_ids, _ = cmd.robot.find_joints([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
    assert len(leg_ids) == 12
    raw = torch.cat([cmd.joint_pos, cmd.joint_vel], dim=1)
    for j in range(nj):
        if j in leg_ids:
            assert torch.equal(out[:, j], cmd.robot.data.default_joint_pos[:, j]), j
            assert torch.all(out[:, j + nj] == 0.0), j
        else:
            assert torch.equal(out[:, j], raw[:, j]), j          # non-leg pos byte-identical
            assert torch.equal(out[:, j + nj], raw[:, j + nj]), j  # non-leg vel byte-identical
    # second call hits the cache and agrees
    out2 = hope_observations_mod.generated_commands_actor_leg_masked(env, "motion")
    assert torch.equal(out, out2)
    assert hasattr(cmd, "_actor_leg_mask_ids")


def test_leg_mask_refuses_wrong_joint_count():
    joints = [j for j in _A3_JOINTS if j != "left_ankle_roll_joint"]  # 11 leg joints only
    cmd = _mask_cmd(3, joints=joints)
    env = _fake_env(motion=cmd)
    with pytest.raises(RuntimeError, match="expected 12 leg joints"):
        hope_observations_mod.generated_commands_actor_leg_masked(env, "motion")


# --------------------------------------------------------------------------------------------- #
# A1 atomic target-message contract + reward gate diagnostics
# --------------------------------------------------------------------------------------------- #
def _make_a1_cmd(
    delay_steps=0,
    dropout_prob=0.0,
    ar1_sigma=0.0,
    tts_mode="live",
    planner_revision=False,
):
    cmd = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    cmd.num_envs = 1
    cmd.device = "cpu"
    cmd._env = types.SimpleNamespace(
        step_dt=0.02,
        termination_manager=types.SimpleNamespace(active_terms=()),
    )
    cmd.metrics = {"actor_time_to_strike_s": torch.zeros(1)}
    cmd._actor_view_active = True
    cmd._delay_steps = delay_steps
    cmd._delay_tts_mode = tts_mode
    cmd._delay_tts_active = tts_mode != "live"
    cmd.planner_revision_enabled = planner_revision
    cmd._atomic_tts_active = cmd._delay_tts_active or planner_revision
    cmd._delay_ptr = 0
    cmd._jitter_pos = 0.0
    cmd._jitter_vel = 0.0
    cmd._mnoise_white = 0.0
    cmd._mnoise_ar1_sigma = ar1_sigma
    cmd._mnoise_ar1_rho = 0.9
    cmd._mnoise_ar1_state = torch.full((1, 3), 7.0)
    cmd._drop_prob = dropout_prob
    cmd._post_strike_drop_steps = 0
    cmd._bias_per_swing = 0.0
    cmd._a1v2_active = dropout_prob > 0.0
    cmd.time_to_strike = torch.tensor([0.5])
    cmd.pre_strike = torch.tensor([True])
    cmd.racket_target_pos_w = torch.tensor([[1.0, 2.0, 3.0]])
    cmd.racket_target_vel_w = torch.tensor([[4.0, 5.0, 6.0]])
    cmd.target_normal_cmd = torch.tensor([[0.0, 1.0, 0.0]])
    cmd.swing_sign = torch.tensor([1.0])
    cmd.delayed_racket_target_pos_w = cmd.racket_target_pos_w.clone()
    cmd.delayed_racket_target_vel_w = cmd.racket_target_vel_w.clone()
    cmd.delayed_target_normal_cmd = cmd.target_normal_cmd.clone()
    cmd.delayed_swing_sign = cmd.swing_sign.clone()
    if cmd._atomic_tts_active:
        cmd.delayed_time_to_strike = cmd.time_to_strike.clone()
    if planner_revision:
        cmd._planner_initial_tts_mixture = None
        cmd._planner_control_epoch = torch.tensor([7], dtype=torch.long)
        cmd._planner_task_id = torch.tensor([10], dtype=torch.long)
        cmd._planner_task_revision = torch.tensor([1], dtype=torch.long)
        cmd._planner_visible_pos = cmd.racket_target_pos_w.clone()
        cmd._planner_visible_vel = cmd.racket_target_vel_w.clone()
        cmd._planner_visible_normal = cmd.target_normal_cmd.clone()
        cmd._planner_visible_tts = cmd.time_to_strike.clone()
        cmd._planner_visible_last_precontact = torch.zeros(1, dtype=torch.bool)
        cmd._planner_actor_control_epoch = cmd._planner_control_epoch.clone()
        cmd._planner_actor_task_id = cmd._planner_task_id.clone()
        cmd._planner_actor_task_revision = cmd._planner_task_revision.clone()
    if delay_steps > 0:
        length = delay_steps + 1
        cmd._delay_buf_pos = cmd.racket_target_pos_w.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_vel = cmd.racket_target_vel_w.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_normal = cmd.target_normal_cmd.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_sign = cmd.swing_sign.unsqueeze(0).repeat(length, 1)
        if cmd._atomic_tts_active:
            cmd._delay_buf_tts = cmd.time_to_strike.unsqueeze(0).repeat(length, 1)
        if planner_revision:
            cmd._delay_buf_planner_epoch = cmd._planner_control_epoch.unsqueeze(0).repeat(length, 1)
            cmd._delay_buf_planner_task = cmd._planner_task_id.unsqueeze(0).repeat(length, 1)
            cmd._delay_buf_planner_revision = cmd._planner_task_revision.unsqueeze(0).repeat(length, 1)
            cmd._delay_buf_planner_last_precontact = torch.zeros(
                length, 1, dtype=torch.bool
            )
    if cmd._a1v2_active:
        cmd._swing_bias = torch.zeros(1, 3)
        cmd._drop_cd = torch.zeros(1, dtype=torch.long)
        cmd._prev_pre_strike = torch.ones(1, dtype=torch.bool)
        cmd._held_pos = cmd.racket_target_pos_w.clone()
        cmd._held_vel = cmd.racket_target_vel_w.clone()
        cmd._held_normal = cmd.target_normal_cmd.clone()
        cmd._held_sign = cmd.swing_sign.clone()
        if cmd._atomic_tts_active:
            cmd._held_tts = cmd.time_to_strike.clone()
        if planner_revision:
            cmd._held_planner_epoch = cmd._planner_control_epoch.clone()
            cmd._held_planner_task = cmd._planner_task_id.clone()
            cmd._held_planner_revision = cmd._planner_task_revision.clone()
            cmd._held_planner_last_precontact = torch.zeros(1, dtype=torch.bool)
    return cmd


def _set_question_b(cmd):
    cmd.racket_target_pos_w[:] = torch.tensor([[10.0, 20.0, 30.0]])
    cmd.racket_target_vel_w[:] = torch.tensor([[40.0, 50.0, 60.0]])
    cmd.target_normal_cmd[:] = torch.tensor([[1.0, 0.0, 0.0]])
    cmd.swing_sign[:] = -1.0


def _assert_actor_question(cmd, which):
    if which == "a":
        assert torch.equal(cmd.delayed_racket_target_pos_w, torch.tensor([[1.0, 2.0, 3.0]]))
        assert torch.equal(cmd.delayed_racket_target_vel_w, torch.tensor([[4.0, 5.0, 6.0]]))
        assert torch.equal(cmd.delayed_target_normal_cmd, torch.tensor([[0.0, 1.0, 0.0]]))
        assert torch.equal(cmd.delayed_swing_sign, torch.tensor([1.0]))
    else:
        assert torch.equal(cmd.delayed_racket_target_pos_w, torch.tensor([[10.0, 20.0, 30.0]]))
        assert torch.equal(cmd.delayed_racket_target_vel_w, torch.tensor([[40.0, 50.0, 60.0]]))
        assert torch.equal(cmd.delayed_target_normal_cmd, torch.tensor([[1.0, 0.0, 0.0]]))
        assert torch.equal(cmd.delayed_swing_sign, torch.tensor([-1.0]))


def test_a1_delay_keeps_face_command_atomic_with_pos_vel_and_side():
    cmd = _make_a1_cmd(delay_steps=1)
    _set_question_b(cmd)
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    # The face observation must read the same delayed message, not the live question B normal.
    obs = hope_observations_mod.racket_target_normal_cmd(
        _fake_env(racket_target=cmd), "racket_target"
    )
    assert torch.equal(obs, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    cmd._push_actor_target()
    _assert_actor_question(cmd, "b")


@pytest.mark.parametrize(
    ("mode", "expected_first_tts", "expected_delayed_tts"),
    [
        ("source_timestamp_compensated", 0.46, 0.44),
        ("uncompensated", 0.50, 0.48),
    ],
)
def test_a1_delay_two_keeps_tts_atomic_and_distinguishes_timestamp_compensation(
    mode, expected_first_tts, expected_delayed_tts
):
    cmd = _make_a1_cmd(delay_steps=2, tts_mode=mode)
    _set_question_b(cmd)

    # The two backfilled source tuples both carry question A/TTS=0.50.  Compensation advances
    # that source TTS by the known 2*20 ms transport age; the negative control exposes 0.50 stale.
    cmd.time_to_strike[:] = 0.48
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    assert cmd.actor_time_to_strike().item() == pytest.approx(expected_first_tts)

    cmd.time_to_strike[:] = 0.46
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")

    # Third push reads the first question-B source tuple (source TTS=0.48) atomically with B.
    cmd.time_to_strike[:] = 0.44
    cmd._push_actor_target()
    _assert_actor_question(cmd, "b")
    assert cmd.actor_time_to_strike().item() == pytest.approx(expected_delayed_tts)
    assert cmd.metrics["actor_time_to_strike_s"].item() == pytest.approx(expected_delayed_tts)


def test_a1_default_live_tts_is_the_live_tensor_alias_even_when_targets_are_delayed():
    cmd = _make_a1_cmd(delay_steps=2, tts_mode="live")
    assert cmd.actor_time_to_strike() is cmd.time_to_strike
    cmd.time_to_strike[:] = 0.37
    _set_question_b(cmd)
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    assert cmd.actor_time_to_strike().item() == pytest.approx(0.37)
    obs = hope_observations_mod.actor_time_to_strike(
        _fake_env(racket_target=cmd), "racket_target"
    )
    assert obs.item() == pytest.approx(0.37)


@pytest.mark.parametrize("mode", ["source_timestamp_compensated", "uncompensated"])
def test_a1_atomic_tts_zero_delay_is_numerically_live(mode):
    cmd = _make_a1_cmd(delay_steps=0, tts_mode=mode)
    cmd.time_to_strike[:] = 0.31
    cmd._push_actor_target()
    assert cmd.actor_time_to_strike().item() == pytest.approx(0.31)


def test_planner_revision_actor_delivery_counts_only_after_atomic_ring_materializes():
    cmd = _make_a1_cmd(
        delay_steps=2,
        tts_mode="source_timestamp_compensated",
        planner_revision=True,
    )
    cmd._planner_task_revision[:] = 2
    cmd._planner_visible_pos[:] = torch.tensor([[10.0, 20.0, 30.0]])
    cmd._planner_visible_vel[:] = torch.tensor([[40.0, 50.0, 60.0]])
    cmd._planner_visible_normal[:] = torch.tensor([[1.0, 0.0, 0.0]])
    cmd._planner_visible_tts[:] = 0.02
    cmd._planner_visible_last_precontact[:] = True

    cmd._push_actor_target()
    cmd._push_actor_target()
    before_delivery = cmd.consume_exact_behavior_decision_counters()
    assert before_delivery["planner_revision_actor_visible_count"].item() == 0
    assert before_delivery["planner_revision_last_precontact_actor_visible_count"].item() == 0

    cmd._push_actor_target()
    delivered = cmd.consume_exact_behavior_decision_counters()
    assert delivered["planner_revision_actor_visible_count"].item() == 1
    assert delivered["planner_revision_last_precontact_actor_visible_count"].item() == 1
    assert cmd.actor_time_to_strike().item() == 0.0

    cmd._push_actor_target()
    duplicate = cmd.consume_exact_behavior_decision_counters()
    assert duplicate["planner_revision_actor_visible_count"].item() == 0
    assert duplicate["planner_revision_last_precontact_actor_visible_count"].item() == 0


def test_planner_revision_delay_zero_counts_same_step_and_clamps_postcontact_tts():
    cmd = _make_a1_cmd(
        delay_steps=0,
        tts_mode="source_timestamp_compensated",
        planner_revision=True,
    )
    cmd._planner_task_revision[:] = 2
    cmd._planner_visible_tts[:] = -0.04
    cmd._planner_visible_last_precontact[:] = True
    cmd._push_actor_target()
    delivered = cmd.consume_exact_behavior_decision_counters()
    assert delivered["planner_revision_actor_visible_count"].item() == 1
    assert delivered["planner_revision_last_precontact_actor_visible_count"].item() == 1
    assert cmd.actor_time_to_strike().item() == 0.0


def test_delayed_final_revision_arriving_after_contact_is_not_precontact_visible():
    cmd = _make_a1_cmd(
        delay_steps=2,
        tts_mode="source_timestamp_compensated",
        planner_revision=True,
    )
    cmd._planner_task_revision[:] = 2
    cmd._planner_visible_tts[:] = 0.02
    cmd._planner_visible_last_precontact[:] = True
    cmd._push_actor_target()
    cmd._push_actor_target()
    cmd.pre_strike[:] = False
    cmd._push_actor_target()
    delivered = cmd.consume_exact_behavior_decision_counters()
    assert delivered["planner_revision_actor_visible_count"].item() == 1
    assert delivered["planner_revision_last_precontact_actor_visible_count"].item() == 0


def test_a1_dropout_holds_the_entire_target_message():
    cmd = _make_a1_cmd(dropout_prob=1.0)
    _set_question_b(cmd)
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    cmd._drop_prob = 0.0
    cmd._push_actor_target()
    _assert_actor_question(cmd, "b")


def test_a1_true_reset_clears_state_and_backfills_all_message_fields():
    cmd = _make_a1_cmd(
        delay_steps=2,
        dropout_prob=1.0,
        ar1_sigma=0.2,
        tts_mode="source_timestamp_compensated",
    )
    _set_question_b(cmd)
    cmd.time_to_strike[:] = 0.63
    cmd._drop_cd[:] = 9
    cmd._swing_bias[:] = 3.0
    cmd._held_pos[:] = -3.0
    cmd._held_vel[:] = -4.0
    cmd._held_normal[:] = -5.0
    cmd._held_sign[:] = 0.0
    cmd._held_tts[:] = -6.0
    cmd.delayed_time_to_strike[:] = -7.0
    cmd._reset_actor_target_state(torch.tensor([0]))
    _assert_actor_question(cmd, "b")
    assert torch.all(cmd._delay_buf_pos == cmd.racket_target_pos_w.unsqueeze(0))
    assert torch.all(cmd._delay_buf_vel == cmd.racket_target_vel_w.unsqueeze(0))
    assert torch.all(cmd._delay_buf_normal == cmd.target_normal_cmd.unsqueeze(0))
    assert torch.all(cmd._delay_buf_sign == cmd.swing_sign.unsqueeze(0))
    assert torch.all(cmd._delay_buf_tts == cmd.time_to_strike.unsqueeze(0))
    assert torch.equal(cmd.delayed_time_to_strike, cmd.time_to_strike)
    assert torch.all(cmd._mnoise_ar1_state == 0.0)
    assert torch.all(cmd._swing_bias == 0.0)
    assert torch.all(cmd._drop_cd == 0)
    assert torch.equal(cmd._held_normal, cmd.target_normal_cmd)
    assert torch.equal(cmd._held_sign, cmd.swing_sign)
    assert torch.equal(cmd._held_tts, cmd.time_to_strike)


def test_debug_reward_log_records_real_gate_instead_of_identity():
    cmd = types.SimpleNamespace(
        cfg=types.SimpleNamespace(debug_reward_logging=True),
        metrics={"dbg_probe_raw": torch.zeros(2), "dbg_probe_gated": torch.zeros(2)},
    )
    raw = torch.tensor([0.25, 0.75])
    hope_rewards_mod._dbg_log(cmd, "probe", raw, torch.tensor([True, False]))
    assert torch.equal(cmd.metrics["dbg_probe_raw"], raw)
    assert torch.equal(cmd.metrics["dbg_probe_gated"], torch.tensor([0.25, 0.0]))


def test_base_decel_is_disabled_during_frozen_hold():
    motion = types.SimpleNamespace(in_hold=torch.tensor([True, False]))
    cmd = types.SimpleNamespace(
        racket_target_pos_w=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        racket_pos_w=torch.zeros(2, 3),
        robot=types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_w=torch.zeros(2, 3))),
        pre_strike=torch.tensor([True, True]),
        _motion=lambda: motion,
    )
    reward = hope_rewards_mod.base_decel_tracking(
        _fake_env(racket_target=cmd), "racket_target", v_gain=1.0, v_max=2.0, std=1.0
    )
    assert reward[0] == 0.0
    assert reward[1] > 0.0


def test_base_decel_observer_is_a_real_reward_term_not_a_command_stage_hook():
    # Isaac 2.1 executes reward -> reset -> command.  Both paired arms therefore must enter through
    # RewardManager; any command-stage observer would compare a next-state control against an
    # old-state treatment.  Keep this regression tied to the shipped class/config source.
    command_source = inspect.getsource(hope_commands_mod.RacketTargetCommand)
    assert "_observe_base_decel_activation" not in command_source
    assert "base_decel_activation_enabled" not in command_source

    cfg_path = os.path.abspath(
        os.path.join(MDP_DIR, "..", "config", "agibot_a3", "hope_env_cfg.py")
    )
    with open(cfg_path, encoding="utf-8") as handle:
        config_source = handle.read()
    probe_decl = config_source.index("base_decel_activation_probe = RewTerm(")
    treatment_decl = config_source.index("base_decel = RewTerm(", probe_decl)
    assert probe_decl < treatment_decl
    assert "func=mdp.base_decel_activation_probe, weight=0.0" in config_source[
        probe_decl:treatment_decl
    ]


# --------------------------------------------------------------------------------------------- #
# R-c MotionCommand birth fixes (real MotionCommand on synthetic clips)
# --------------------------------------------------------------------------------------------- #
_BODY_NAMES = ["pelvis_link", "torso_Link"]
_N_JOINTS = 31
_STAND_Z = 1.0684
_CROUCH_Z = 0.78


def _write_motion_npz(
    path,
    frames,
    root_z=_CROUCH_Z,
    *,
    fps=50,
    metadata_body_names=None,
):
    quat = np.zeros((frames, len(_BODY_NAMES), 4), dtype=np.float32)
    quat[..., 0] = 1.0
    body_pos = np.zeros((frames, len(_BODY_NAMES), 3), dtype=np.float32)
    body_pos[:, :, 2] = root_z
    np.savez(
        path,
        fps=fps,
        joint_pos=np.zeros((frames, _N_JOINTS), dtype=np.float32),
        joint_vel=np.zeros((frames, _N_JOINTS), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((frames, len(_BODY_NAMES), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, len(_BODY_NAMES), 3), dtype=np.float32),
        kinematics_schema_version=np.array([2], dtype=np.int64),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
        body_names=np.asarray(metadata_body_names or _BODY_NAMES),
    )
    return path


class _Scene:
    def __init__(self, robot, num_envs):
        self._robot = robot
        self.env_origins = torch.zeros(num_envs, 3)

    def __getitem__(self, key):
        assert key == "robot"
        return self._robot


class _CmdRobot:
    def __init__(self, num_envs):
        self.body_names = list(_BODY_NAMES)
        self.joint_names = list(_A3_JOINTS)
        default_root = torch.zeros(num_envs, 13)
        default_root[:, 2] = _STAND_Z
        default_root[:, 3] = 1.0
        self.data = types.SimpleNamespace(
            joint_names=list(_A3_JOINTS),
            joint_pos=torch.zeros(num_envs, _N_JOINTS),
            joint_vel=torch.zeros(num_envs, _N_JOINTS),
            joint_vel_limits=torch.full((_N_JOINTS,), 5.0),
            root_state_w=default_root.clone(),
            default_joint_pos=torch.zeros(num_envs, _N_JOINTS),
            default_joint_vel=torch.zeros(num_envs, _N_JOINTS),
            default_root_state=default_root,
            soft_joint_pos_limits=torch.stack(
                [-torch.ones(num_envs, _N_JOINTS) * 3.14, torch.ones(num_envs, _N_JOINTS) * 3.14], dim=-1),
        )
        self.calls = []

    def find_bodies(self, names, preserve_order=True):
        return [self.body_names.index(n) for n in names], list(names)

    def write_root_state_to_sim(self, root, env_ids=None):
        self.calls.append(("root", root.clone(), None if env_ids is None else env_ids.clone()))
        self.data.root_state_w[env_ids] = root

    def write_joint_state_to_sim(self, jp, jv, env_ids=None):
        self.calls.append(("joint", jp.clone(), jv.clone()))
        self.data.joint_pos[env_ids] = jp
        self.data.joint_vel[env_ids] = jv


def _make_motion_command(motion_files, num_envs=8, **cfg_overrides):
    robot = _CmdRobot(num_envs)
    env = types.SimpleNamespace(
        num_envs=num_envs, device="cpu", step_dt=0.02,
        scene=_Scene(robot, num_envs),
        cfg=types.SimpleNamespace(decimation=4, sim=types.SimpleNamespace(dt=0.005)),
        termination_manager=types.SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
    )
    cfg_kwargs = dict(
        asset_name="robot",
        motion_file=motion_files if len(motion_files) > 1 else motion_files[0],
        anchor_body_name="torso_Link",
        body_names=list(_BODY_NAMES),
        pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0),
        stand_start_prob=0.0, post_swing_start_prob=0.0,
        hold_steps_range=(5, 5),  # every birth held -> deterministic held-RSI path
    )
    cfg_kwargs.update(cfg_overrides)
    cfg = commands_mod.MotionCommandCfg(**cfg_kwargs)
    return commands_mod.MotionCommand(cfg, env), robot


@pytest.fixture(scope="module")
def clips():
    tmp = tempfile.mkdtemp(prefix="reward_flags_clips_")
    return (
        _write_motion_npz(os.path.join(tmp, "forehand.npz"), frames=20),
        _write_motion_npz(os.path.join(tmp, "backhand.npz"), frames=15),
        _write_motion_npz(os.path.join(tmp, "single.npz"), frames=30),
    )


def test_motion_loader_rejects_wrong_body_order_and_fps_contracts(tmp_path):
    wrong_order = _write_motion_npz(
        tmp_path / "wrong_order.npz",
        frames=12,
        metadata_body_names=list(reversed(_BODY_NAMES)),
    )
    with pytest.raises(ValueError, match="body_names/order"):
        _make_motion_command([wrong_order])

    clip_50 = _write_motion_npz(tmp_path / "clip_50.npz", frames=12, fps=50)
    clip_49 = _write_motion_npz(tmp_path / "clip_49.npz", frames=12, fps=49)
    with pytest.raises(ValueError, match="unequal fps"):
        _make_motion_command([clip_50, clip_49])

    clip_60 = _write_motion_npz(tmp_path / "clip_60.npz", frames=12, fps=60)
    with pytest.raises(ValueError, match="motion fps must equal the policy rate"):
        _make_motion_command([clip_60])


def test_motion_loader_rejects_non_scalar_or_nonfinite_fps(tmp_path):
    nonscalar = _write_motion_npz(
        tmp_path / "nonscalar_fps.npz", frames=12, fps=np.array([50.0, 50.0])
    )
    with pytest.raises(ValueError, match="fps must be scalar"):
        _make_motion_command([nonscalar])

    nonfinite = _write_motion_npz(tmp_path / "nonfinite_fps.npz", frames=12, fps=np.nan)
    with pytest.raises(ValueError, match="finite and positive"):
        _make_motion_command([nonfinite])


def test_rc_skip_settle_frames_multiseg(clips):
    cmd, _ = _make_motion_command([clips[0], clips[1]], rsi_skip_settle_frames=6)
    ids = torch.arange(cmd.num_envs)
    cmd._adaptive_sampling(ids)
    seg_start = cmd.motion.seg_start[cmd.clip_id]
    assert torch.all(cmd.time_steps == seg_start + 6)  # every swing entry starts at frame N
    # default 0 = byte-identical entry at seg_start
    cmd0, _ = _make_motion_command([clips[0], clips[1]])
    cmd0._adaptive_sampling(ids)
    assert torch.all(cmd0.time_steps == cmd0.motion.seg_start[cmd0.clip_id])


def test_rc_skip_settle_frames_clamps_short_clips(clips):
    cmd, _ = _make_motion_command([clips[0], clips[1]], rsi_skip_settle_frames=50)
    ids = torch.arange(cmd.num_envs)
    cmd._adaptive_sampling(ids)
    seg_start = cmd.motion.seg_start[cmd.clip_id]
    seg_last = seg_start + cmd.motion.seg_len[cmd.clip_id] - 1
    assert torch.all(cmd.time_steps == seg_last)  # clamped inside the segment, never out of range


def test_rc_skip_settle_frames_single_clip(clips):
    torch.manual_seed(0)
    cmd, _ = _make_motion_command([clips[2]], rsi_skip_settle_frames=6)
    ids = torch.arange(cmd.num_envs)
    cmd._adaptive_sampling(ids)
    assert torch.all(cmd.time_steps >= 6)
    assert torch.all(cmd.time_steps <= cmd.motion.time_step_total - 1)


def test_rc_hold_root_stand_z_rewrites_held_rsi_births(clips):
    torch.manual_seed(0)
    cmd, robot = _make_motion_command([clips[2]], rsi_hold_root_stand_z=True)
    cmd._resample_command(torch.arange(cmd.num_envs))
    roots = [c for c in robot.calls if c[0] == "root"]
    assert len(roots) == 1
    root = roots[0][1]
    assert torch.all(cmd.hold_counter[roots[0][2]] > 0)  # the births under test are HELD
    assert torch.allclose(root[:, 2], torch.full((root.shape[0],), _STAND_Z))  # stand height
    assert torch.allclose(root[:, 7:], torch.zeros_like(root[:, 7:]))  # hold-zeroed velocities


def test_rc_hold_root_stand_z_off_keeps_reference_z(clips):
    torch.manual_seed(0)
    cmd, robot = _make_motion_command([clips[2]])  # flag off (default)
    cmd._resample_command(torch.arange(cmd.num_envs))
    root = [c for c in robot.calls if c[0] == "root"][0][1]
    assert torch.allclose(root[:, 2], torch.full((root.shape[0],), _CROUCH_Z))  # crouch frame-0 z


def test_rc_hold_root_stand_z_leaves_unheld_births_alone(clips):
    torch.manual_seed(0)
    cmd, robot = _make_motion_command(
        [clips[2]], rsi_hold_root_stand_z=True, hold_steps_range=(0, 0))  # hold=0: never held
    cmd._resample_command(torch.arange(cmd.num_envs))
    root = [c for c in robot.calls if c[0] == "root"][0][1]
    assert torch.allclose(root[:, 2], torch.full((root.shape[0],), _CROUCH_Z))  # untouched


def _prime_post_swing_buffer(cmd):
    count = int(cmd.cfg.post_swing_min_fill)
    cmd._post_swing_count = count
    cmd._post_swing_ptr = 0
    cmd._post_swing_root = torch.zeros(count, 13)
    cmd._post_swing_root[:, 3] = 1.0
    cmd._post_swing_joint_pos = torch.zeros(count, _N_JOINTS)
    cmd._post_swing_joint_vel = torch.zeros(count, _N_JOINTS)


def _counter_values(snapshot):
    return {name: int(value.item()) for name, value in snapshot.items()}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_post_swing_teacher_receipt(tmp_path, motion_files, *, count=4):
    payload = tmp_path / "post_swing_teacher_states.npz"
    root = np.zeros((count, 13), dtype=np.float32)
    root[:, 2] = 1.0
    root[:, 3] = 1.0
    np.savez(
        payload,
        root_state_origin_relative=root,
        joint_pos=np.zeros((count, _N_JOINTS), dtype=np.float32),
        joint_vel=np.zeros((count, _N_JOINTS), dtype=np.float32),
    )
    receipt = {
        "schema_version": 2,
        "artifact_kind": "hope_post_swing_teacher_state_receipt",
        "capture_contract": {
            "event": "natural_clip_wrap",
            "wrap_teleport": False,
            "clip_switch_aborted_states_included": False,
            "root_position_frame": "environment_origin_relative",
            "root_state_layout": "pos3_quat_wxyz4_linear_velocity_com3_angular_velocity3",
            "joint_state_order": "runtime_articulation_joint_names",
        },
        "teacher": {
            "source_commit": "1" * 40,
            "checkpoint_sha256": "2" * 64,
            "training_contract_sha256": "3" * 64,
            "training_contract_schema_version": 3,
            "fresh_lineage": True,
        },
        "motion_clips": [
            {"index": index, "sha256": _sha256(path)}
            for index, path in enumerate(motion_files)
        ],
        "states": {
            "relative_path": payload.name,
            "sha256": _sha256(payload),
            "count": count,
            "root_shape": [count, 13],
            "joint_pos_shape": [count, _N_JOINTS],
            "joint_vel_shape": [count, _N_JOINTS],
            "joint_names": list(_A3_JOINTS),
            "velocity_limits": {
                "root_linear_norm_max_mps": 2.0,
                "root_angular_norm_max_radps": 4.0,
                "joint_abs_max_radps": [5.0] * _N_JOINTS,
            },
        },
        "attestation": {
            "schema_version": 2,
            "artifact_kind": "hope_post_swing_teacher_capture_attestation",
            "capture_result_sha256": "4" * 64,
            "capture_result_relative_path": "natural_wrap_capture.json",
            "capture_claim_sha256": "a" * 64,
            "capture_claim_relative_path": "natural_wrap_capture.claim.json",
            "checkpoint": {
                "sha256": "2" * 64,
                "training_contract_schema_version": 3,
                "training_contract_sha256": "3" * 64,
                "training_contract_lineage_exact": True,
                "training_launch_claim_sha256": "5" * 64,
            },
            "hard_contract": {"sha256": "3" * 64, "schema_version": 3},
            "checkpoint_source": {
                "commit": "1" * 40,
                "launch_claim_content_sha256": "5" * 64,
            },
            "capture_source": {
                "commit": "6" * 40,
                "clean": True,
                "producer_source_sha256": "8" * 64,
            },
            "attestor_source": {
                "commit": "7" * 40,
                "clean": True,
                "attestor_source_sha256": "9" * 64,
            },
            "retry_authorization": {
                "authorization_id": "test-v3-attestor-attempt2",
                "file_sha256": "b" * 64,
                "v3_plan_file_sha256": "c" * 64,
            },
        },
    }
    capture_claim = {
        "schema_version": 1,
        "artifact_kind": "hope_post_swing_natural_wrap_exclusive_claim",
        "producer_source_sha256": "8" * 64,
        "runtime_hard_contract_sha256": "3" * 64,
        "target_count": count,
        "motion_clips": receipt["motion_clips"],
        "joint_names": list(_A3_JOINTS),
        "exclusive_create": True,
    }
    claim_path = tmp_path / "natural_wrap_capture.claim.json"
    claim_path.write_text(
        json.dumps(capture_claim, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["capture_claim_sha256"] = _sha256(claim_path)
    capture_result = {
        "schema_version": 2,
        "artifact_kind": "hope_post_swing_natural_wrap_capture_result",
        "capture_contract": receipt["capture_contract"],
        "evidence": {
            "producer_source_sha256": "8" * 64,
            "runtime_hard_contract_sha256": "3" * 64,
            "exclusive_claim_sha256": _sha256(claim_path),
            "exclusive_claim_relative_path": "natural_wrap_capture.claim.json",
            "no_clobber": True,
        },
        "motion_clips": receipt["motion_clips"],
        "states": {
            key: value for key, value in receipt["states"].items() if key != "velocity_limits"
        },
    }
    capture_path = tmp_path / "natural_wrap_capture.json"
    capture_path.write_text(
        json.dumps(capture_result, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["capture_result_sha256"] = _sha256(capture_path)
    receipt_path = tmp_path / "post_swing_teacher_receipt.json"
    authorization = {
        "schema_version": 1,
        "artifact_kind": "hope_post_swing_teacher_attestor_retry_authorization",
        "authorization_id": "test-v3-attestor-attempt2",
        "v3_plan": {"plan_id": tmp_path.name, "file_sha256": "c" * 64},
        "capture": {
            "output_directory": str(tmp_path.absolute()),
            "output_receipt": str(receipt_path.absolute()),
            "capture_claim_sha256": _sha256(claim_path),
            "states_sha256": _sha256(payload),
            "result_sha256": _sha256(capture_path),
            "state_count": count,
        },
        "teacher": {
            "checkpoint_sha256": "2" * 64,
            "hard_contract_sha256": "3" * 64,
            "launch_claim_content_sha256": "5" * 64,
        },
        "capture_source": {
            "commit": "6" * 40,
            "producer_source_sha256": "8" * 64,
        },
        "attestor_source": {
            "commit": "7" * 40,
            "attestor_source_sha256": "9" * 64,
        },
        "decision": {
            "capture_retry_authorized": False,
            "attestor_attempt2_authorized": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
        },
    }
    authorization_path = tmp_path / "retry_authorization.json"
    authorization_path.write_text(
        json.dumps(authorization, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["retry_authorization"]["file_sha256"] = _sha256(
        authorization_path
    )
    receipt_path.write_text(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path, _sha256(receipt_path), authorization_path, _sha256(
        authorization_path
    )


def test_post_swing_activation_disabled_stays_zero_without_replay_adoption(clips):
    cmd, robot = _make_motion_command(
        [clips[0], clips[1]], num_envs=4, post_swing_start_prob=0.0
    )
    _prime_post_swing_buffer(cmd)
    cmd._resample_command(torch.arange(4))

    assert _counter_values(cmd.consume_post_swing_activation_counters()) == {
        "post_swing_replay_buffer_not_ready_reset_count": 0,
        "post_swing_replay_eligible_reset_count": 0,
        "post_swing_replay_random_not_selected_reset_count": 0,
        "post_swing_replay_selected_reset_count": 0,
        "post_swing_replay_started_reset_count": 0,
    }
    # The legacy RSI path still writes exactly one joint/root batch; enabling instrumentation
    # alone must not turn a disabled post-swing mechanism into a replay state write.
    assert [call[0] for call in robot.calls] == ["joint", "root"]


def test_post_swing_activation_counts_buffer_not_ready_as_ineligible(clips):
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]], num_envs=5, post_swing_start_prob=0.5
    )
    assert cmd._post_swing_count < int(cmd.cfg.post_swing_min_fill)
    cmd._resample_command(torch.arange(5))

    assert _counter_values(cmd.consume_post_swing_activation_counters()) == {
        "post_swing_replay_buffer_not_ready_reset_count": 5,
        "post_swing_replay_eligible_reset_count": 0,
        "post_swing_replay_random_not_selected_reset_count": 0,
        "post_swing_replay_selected_reset_count": 0,
        "post_swing_replay_started_reset_count": 0,
    }


def test_post_swing_activation_multi_env_selection_and_per_update_reset(clips, monkeypatch):
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        stand_start_prob=0.25,
        post_swing_start_prob=0.5,
    )
    _prime_post_swing_buffer(cmd)
    real_rand = torch.rand

    def controlled_rand(*shape, **kwargs):
        if shape == (4,):
            # stand, replay, replay, RSI: the replay interval is [0.25, 0.75).
            return torch.tensor([0.10, 0.30, 0.70, 0.90], device=kwargs.get("device"))
        return real_rand(*shape, **kwargs)

    monkeypatch.setattr(torch, "rand", controlled_rand)
    cmd._resample_command(torch.arange(4))
    cmd._resample_command(torch.arange(4))
    assert _counter_values(cmd.consume_post_swing_activation_counters()) == {
        "post_swing_replay_buffer_not_ready_reset_count": 0,
        "post_swing_replay_eligible_reset_count": 8,
        "post_swing_replay_random_not_selected_reset_count": 4,
        "post_swing_replay_selected_reset_count": 4,
        "post_swing_replay_started_reset_count": 4,
    }
    # A second logger consumption is the next PPO update's empty window, not a cumulative total.
    assert all(
        value == 0
        for value in _counter_values(
            cmd.consume_post_swing_activation_counters()
        ).values()
    )


def test_post_swing_selected_is_not_started_until_state_write_returns(clips, monkeypatch):
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]], num_envs=3, post_swing_start_prob=1.0
    )
    _prime_post_swing_buffer(cmd)

    def fail_before_adoption(_env_ids):
        raise RuntimeError("synthetic replay state write failure")

    monkeypatch.setattr(cmd, "_write_post_swing_states", fail_before_adoption)
    with pytest.raises(RuntimeError, match="synthetic replay state write failure"):
        cmd._resample_command(torch.arange(3))
    snapshot = _counter_values(cmd.consume_post_swing_activation_counters())
    assert snapshot["post_swing_replay_eligible_reset_count"] == 3
    assert snapshot["post_swing_replay_selected_reset_count"] == 3
    assert snapshot["post_swing_replay_started_reset_count"] == 0
    assert snapshot["post_swing_replay_random_not_selected_reset_count"] == 0


def test_post_swing_teacher_cold_start_is_ready_and_activates_before_learning(
    clips, tmp_path, monkeypatch
):
    receipt, receipt_sha, authorization, authorization_sha = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    cmd, robot = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        post_swing_start_prob=1.0,
        post_swing_min_fill=4,
        post_swing_buffer_size=8,
        post_swing_teacher_receipt=str(receipt),
        post_swing_teacher_receipt_sha256=receipt_sha,
        post_swing_teacher_retry_authorization=str(authorization),
        post_swing_teacher_retry_authorization_sha256=authorization_sha,
        post_swing_teacher_root_linear_velocity_limit_mps=2.0,
        post_swing_teacher_root_angular_velocity_limit_radps=4.0,
        post_swing_require_ready_at_init=True,
        post_swing_fail_fast_first_reset=True,
    )
    assert cmd._post_swing_count == 4
    assert cmd._post_swing_ptr == 4
    contract = cmd.post_swing_replay_hard_contract()
    assert contract["teacher_distribution"] == "immutable"
    assert contract["teacher_receipt"]["receipt_sha256"] == receipt_sha

    cmd._resample_command(torch.arange(4))
    snapshot = _counter_values(cmd.consume_training_activation_counters())
    assert snapshot["post_swing_replay_buffer_not_ready_reset_count"] == 0
    assert snapshot["post_swing_replay_eligible_reset_count"] == 4
    assert snapshot["post_swing_replay_selected_reset_count"] == 4
    assert snapshot["post_swing_replay_started_reset_count"] == 4
    assert [call[0] for call in robot.calls] == ["root", "joint"]

    # A frozen controlled-direct-effect buffer ignores later live captures.  The state count and
    # pointer remain bound to the immutable receipt instead of becoming treatment-dependent.
    cmd._capture_post_swing_states(torch.arange(4))
    assert cmd._post_swing_count == 4
    assert cmd._post_swing_ptr == 4


def test_post_swing_teacher_fail_fast_refuses_zero_activation_before_first_update(
    clips, tmp_path, monkeypatch
):
    receipt, receipt_sha, authorization, authorization_sha = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        post_swing_start_prob=0.25,
        post_swing_min_fill=4,
        post_swing_buffer_size=8,
        post_swing_teacher_receipt=str(receipt),
        post_swing_teacher_receipt_sha256=receipt_sha,
        post_swing_teacher_retry_authorization=str(authorization),
        post_swing_teacher_retry_authorization_sha256=authorization_sha,
        post_swing_teacher_root_linear_velocity_limit_mps=2.0,
        post_swing_teacher_root_angular_velocity_limit_radps=4.0,
        post_swing_require_ready_at_init=True,
        post_swing_fail_fast_first_reset=True,
    )
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.full(shape, 0.9))
    with pytest.raises(RuntimeError, match="adopted count below"):
        cmd._resample_command(torch.arange(4))


def test_post_swing_first_reset_checks_count_fraction_and_state_readback(clips, tmp_path):
    receipt, receipt_sha, authorization, authorization_sha = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    cmd, robot = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        post_swing_start_prob=1.0,
        post_swing_min_fill=4,
        post_swing_buffer_size=8,
        post_swing_teacher_receipt=str(receipt),
        post_swing_teacher_receipt_sha256=receipt_sha,
        post_swing_teacher_retry_authorization=str(authorization),
        post_swing_teacher_retry_authorization_sha256=authorization_sha,
        post_swing_teacher_root_linear_velocity_limit_mps=2.0,
        post_swing_teacher_root_angular_velocity_limit_radps=4.0,
        post_swing_require_ready_at_init=True,
        post_swing_fail_fast_first_reset=True,
        post_swing_first_reset_min_adopted_count=4,
        post_swing_first_reset_min_adopted_fraction=1.0,
        post_swing_first_reset_selection_tolerance=0.0,
        post_swing_first_reset_require_readback=True,
    )
    cmd._resample_command(torch.arange(4))
    assert cmd._post_swing_first_reset_checked is True
    assert torch.allclose(robot.data.root_state_w[:, 2], torch.ones(4))


def test_post_swing_first_reset_readback_cannot_be_forged_by_selected_count(
    clips, tmp_path, monkeypatch
):
    receipt, receipt_sha, authorization, authorization_sha = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    cmd, robot = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        post_swing_start_prob=1.0,
        post_swing_min_fill=4,
        post_swing_buffer_size=8,
        post_swing_teacher_receipt=str(receipt),
        post_swing_teacher_receipt_sha256=receipt_sha,
        post_swing_teacher_retry_authorization=str(authorization),
        post_swing_teacher_retry_authorization_sha256=authorization_sha,
        post_swing_teacher_root_linear_velocity_limit_mps=2.0,
        post_swing_teacher_root_angular_velocity_limit_radps=4.0,
        post_swing_require_ready_at_init=True,
        post_swing_fail_fast_first_reset=True,
        post_swing_first_reset_min_adopted_count=4,
        post_swing_first_reset_min_adopted_fraction=1.0,
        post_swing_first_reset_selection_tolerance=0.0,
        post_swing_first_reset_require_readback=True,
    )
    monkeypatch.setattr(
        robot,
        "write_root_state_to_sim",
        lambda root, env_ids=None: robot.calls.append(("root", root.clone(), env_ids.clone())),
    )
    with pytest.raises(RuntimeError, match="root readback differs"):
        cmd._resample_command(torch.arange(4))


def test_post_swing_ready_at_init_requires_exact_teacher_receipt(clips):
    with pytest.raises(ValueError, match="require an immutable"):
        _make_motion_command(
            [clips[0], clips[1]],
            post_swing_start_prob=0.25,
            post_swing_require_ready_at_init=True,
        )


def test_post_swing_teacher_receipt_requires_frozen_retry_authorization(
    clips, tmp_path
):
    receipt, receipt_sha, _, _ = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    with pytest.raises(ValueError, match="receipt and retry authorization"):
        _make_motion_command(
            [clips[0], clips[1]],
            post_swing_start_prob=1.0,
            post_swing_min_fill=4,
            post_swing_teacher_receipt=str(receipt),
            post_swing_teacher_receipt_sha256=receipt_sha,
        )


def test_rally_hold_heading_is_hold_only_and_monotone():
    yaw = torch.tensor([0.0, 0.6, 0.6])
    quat = torch.zeros(3, 4)
    quat[:, 0] = torch.cos(yaw / 2.0)
    quat[:, 3] = torch.sin(yaw / 2.0)
    motion = types.SimpleNamespace(in_hold=torch.tensor([True, True, False]))
    cmd = types.SimpleNamespace(
        base_quat_w=quat,
        num_envs=3,
        device="cpu",
        _motion=lambda: motion,
    )
    reward = hope_rewards_mod.hold_heading(_fake_env(racket_target=cmd), "racket_target", std=0.6)
    assert reward[0] == pytest.approx(1.0)
    assert reward[1] == pytest.approx(float(torch.exp(torch.tensor(-1.0))))
    assert reward[2] == 0.0
    with pytest.raises(ValueError, match="finite and > 0"):
        hope_rewards_mod.hold_heading(_fake_env(racket_target=cmd), "racket_target", std=0.0)


def test_rally_foot_orientation_gate_only_zeros_hold_rows():
    motion = types.SimpleNamespace(
        in_hold=torch.tensor([True, False]),
        joint_pos=torch.zeros(2, 2),
    )
    asset = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=torch.tensor([[1.0, 2.0], [1.0, 2.0]])))
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda _name: motion),
        scene={"robot": asset},
    )
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=[0, 1])
    always = hope_rewards_mod.foot_orientation_discipline(env, "motion", asset_cfg, hold_gate=False)
    gated = hope_rewards_mod.foot_orientation_discipline(env, "motion", asset_cfg, hold_gate=True)
    assert torch.equal(always, torch.tensor([3.0, 3.0]))
    assert torch.equal(gated, torch.tensor([0.0, 3.0]))


def _qdot_limit_env(limits=None):
    names = [f"joint_{index:02d}" for index in range(31)]
    joint_vel = torch.zeros(2, 31)
    joint_vel[0, :] = 8.5
    joint_vel[1, 0] = 15.0
    if limits is None:
        limits = torch.full((2, 31), 10.0)
    asset = types.SimpleNamespace(
        joint_names=list(names),
        data=types.SimpleNamespace(
            joint_names=list(names),
            joint_vel=joint_vel,
            joint_vel_limits=limits,
        ),
    )
    env = types.SimpleNamespace(scene={"robot": asset}, common_step_counter=1)
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=list(range(31)))
    return env, asset_cfg


def test_qdot_limit_hinge_uses_actual_runtime_limits_and_normalized_mean():
    env, asset_cfg = _qdot_limit_env()
    result = hope_rewards_mod.joint_velocity_limit_hinge(
        env, asset_cfg, margin=0.85, expected_joint_count=31
    )
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx((1.5 - 0.85) ** 2 / 31.0)


def test_qdot_probe_and_hinge_share_one_observation_but_book_activation_separately():
    env, asset_cfg = _qdot_limit_env()
    probe = hope_rewards_mod.joint_velocity_limit_hinge_probe(env, asset_cfg)
    assert torch.equal(probe, torch.zeros(2))
    reward = hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)
    assert reward[1] > 0.0

    snapshot = hope_rewards_mod.consume_joint_velocity_limit_hinge_activation_counters(env)
    assert snapshot["observed_sample_count"].item() == 2
    assert snapshot["hinge_active_sample_count"].item() == 2
    assert snapshot["excess_sample_count"].item() == 1
    assert snapshot["normalized_excess_square_sum"].item() == pytest.approx(
        reward.sum().item()
    )
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_joint_velocity_limit_hinge_activation_counters(env).values()
    )


def test_sparse_virtual_reward_ledger_is_exact_per_action_and_resets_once():
    command = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    command._clip_names = {0: "forehand", 1: "backhand"}
    names = (
        "strike_opportunity_count",
        "virtual_capture_count",
        "virtual_net_clear_count",
        "virtual_landing_valid_count",
        "virtual_legal_return_count",
    )
    command._sparse_reward_eligibility_counters = {
        name: torch.zeros((), dtype=torch.long) for name in names
    }
    for family in command._clip_names.values():
        for name in names:
            command._sparse_reward_eligibility_counters[f"{name}_{family}"] = torch.zeros(
                (), dtype=torch.long
            )
    command._motion_term = types.SimpleNamespace(
        _multiseg=True, clip_id=torch.tensor([0, 0, 1, 1])
    )
    command._event_timing_bound = True

    command._book_sparse_reward_eligibility(
        exact_strike=torch.tensor([True, True, True, True]),
        capture=torch.tensor([True, False, True, False]),
        net_clear=torch.tensor([True, False, False, False]),
        landing_valid=torch.tensor([True, False, True, False]),
        legal_return=torch.tensor([True, False, False, False]),
    )
    snapshot = command.consume_sparse_reward_eligibility_counters()
    assert snapshot["strike_opportunity_count"].item() == 4
    assert snapshot["virtual_capture_count"].item() == 2
    assert snapshot["virtual_net_clear_count"].item() == 1
    assert snapshot["virtual_landing_valid_count"].item() == 2
    assert snapshot["virtual_legal_return_count"].item() == 1
    assert snapshot["strike_opportunity_count_forehand"].item() == 2
    assert snapshot["strike_opportunity_count_backhand"].item() == 2
    assert snapshot["virtual_capture_count_forehand"].item() == 1
    assert snapshot["virtual_capture_count_backhand"].item() == 1
    assert snapshot["virtual_legal_return_count_forehand"].item() == 1
    assert snapshot["virtual_legal_return_count_backhand"].item() == 0
    assert all(
        value.item() == 0
        for value in command.consume_sparse_reward_eligibility_counters().values()
    )


def test_qdot_limit_hinge_reloads_runtime_limits_after_first_call():
    limits = torch.full((2, 31), 10.0)
    env, asset_cfg = _qdot_limit_env(limits)
    first = hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)
    assert first[1] > 0.0

    # A runtime limit update must affect the very next reward; a first-call cache would keep
    # charging against 10 rad/s and fail this assertion.
    limits.fill_(20.0)
    second = hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)
    assert torch.equal(second, torch.zeros_like(second))

    # The same live path must fail closed if a later update introduces an invalid limit.
    limits[:, 4] = 0.0
    with pytest.raises(RuntimeError, match="finite, positive"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)


def test_qdot_limit_hinge_rejects_order_zero_nonfinite_and_per_env_drift():
    env, asset_cfg = _qdot_limit_env()
    with pytest.raises(ValueError, match="exact 31-joint"):
        hope_rewards_mod.joint_velocity_limit_hinge(
            env, asset_cfg, expected_joint_count=31.0
        )

    asset_cfg.joint_ids[0], asset_cfg.joint_ids[1] = 1, 0
    with pytest.raises(RuntimeError, match="identity 31-joint"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)

    limits = torch.full((2, 31), 10.0)
    limits[:, 7] = 0.0
    env, asset_cfg = _qdot_limit_env(limits)
    with pytest.raises(RuntimeError, match="finite, positive"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)

    limits = torch.full((2, 31), 10.0)
    limits[:, 8] = float("nan")
    env, asset_cfg = _qdot_limit_env(limits)
    with pytest.raises(RuntimeError, match="finite, positive"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)

    limits = torch.full((2, 31), 10.0)
    limits[1, 8] = 11.0
    env, asset_cfg = _qdot_limit_env(limits)
    with pytest.raises(RuntimeError, match="identical articulation velocity limits"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)


def test_clamped_action_keeps_reset_aware_previous_processed_qdes():
    n = 2
    names = list(_A3_JOINTS)
    limits = torch.stack(
        (
            torch.full((n, len(names)), -0.5),
            torch.full((n, len(names)), 0.5),
        ),
        dim=-1,
    )
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            default_joint_pos=torch.zeros(n, len(names)),
            soft_joint_pos_limits=limits,
        )
    )
    cfg = types.SimpleNamespace(
        asset_name="robot", scale=1.0, use_default_offset=True, clamp=True
    )
    env = types.SimpleNamespace(
        scene={"robot": asset}, num_envs=n, device="cpu"
    )
    action = hope_actions_mod.ClampedJointPositionAction(cfg, env)

    first = torch.zeros(n, len(names))
    first[:, 0] = torch.tensor([0.8, -0.8])
    action.process_actions(first)
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.5, -0.5])
    assert action.previous_processed_qdes_valid.tolist() == [False, False]

    second = torch.full((n, len(names)), 0.1)
    action.process_actions(second)
    assert action.previous_processed_qdes[:, 0].tolist() == pytest.approx([0.5, -0.5])
    assert action.previous_processed_qdes_valid.tolist() == [True, True]

    action.reset(env_ids=torch.tensor([0]))
    third = torch.full((n, len(names)), 0.2)
    action.process_actions(third)
    assert action.previous_processed_qdes[:, 0].tolist() == pytest.approx([0.1, 0.1])
    assert action.previous_processed_qdes_valid.tolist() == [False, True]


def test_post_strike_clock_counts_control_ticks_and_closes_on_wrap_or_reveal():
    cmd = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    cmd.num_envs = 4
    cmd.device = "cpu"
    cmd._env = types.SimpleNamespace(step_dt=0.02, common_step_counter=10)
    cmd.pre_strike = torch.zeros(4, dtype=torch.bool)
    cmd._post_strike_elapsed_s = torch.zeros(4)
    cmd._post_strike_elapsed_valid = torch.zeros(4, dtype=torch.bool)
    cmd._post_strike_elapsed_last_step = -1
    motion = types.SimpleNamespace(
        retiming_active=True,
        time_steps_f=torch.full((4,), 60.0),
        # Deliberately time-varying and irrelevant: wall time must not be inferred from it.
        speed_scale=torch.tensor([0.1, 0.5, 2.0, 4.0]),
        just_resampled=torch.zeros(4, dtype=torch.bool),
        event_just_installed=torch.zeros(4, dtype=torch.bool),
    )
    cmd._motion_term = motion
    exact = torch.tensor([True, False, True, False])
    hope_commands_mod.RacketTargetCommand._advance_post_strike_elapsed(cmd, exact)
    age, same_attempt = cmd.post_strike_age_and_same_attempt()
    assert age.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert same_attempt.tolist() == [True, False, True, False]

    # A duplicate call in one manager step is idempotent; the next unique tick adds exactly 20 ms.
    hope_commands_mod.RacketTargetCommand._advance_post_strike_elapsed(
        cmd, torch.zeros(4, dtype=torch.bool)
    )
    assert age.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    cmd._env.common_step_counter = 11
    motion.just_resampled[0] = True
    motion.event_just_installed[2] = True
    hope_commands_mod.RacketTargetCommand._advance_post_strike_elapsed(
        cmd, torch.zeros(4, dtype=torch.bool)
    )
    assert age.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert same_attempt.tolist() == [False, False, False, False]


def test_post_strike_clock_advances_exactly_twenty_ms_per_unique_tick():
    cmd = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    cmd.num_envs = 1
    cmd.device = "cpu"
    cmd._env = types.SimpleNamespace(step_dt=0.02, common_step_counter=1)
    cmd.pre_strike = torch.zeros(1, dtype=torch.bool)
    cmd._post_strike_elapsed_s = torch.zeros(1)
    cmd._post_strike_elapsed_valid = torch.zeros(1, dtype=torch.bool)
    cmd._post_strike_elapsed_last_step = -1
    cmd._motion_term = types.SimpleNamespace(
        just_resampled=torch.zeros(1, dtype=torch.bool),
        event_just_installed=torch.zeros(1, dtype=torch.bool),
        speed_scale=torch.tensor([4.0]),
    )
    cmd._advance_post_strike_elapsed(torch.ones(1, dtype=torch.bool))
    for step in range(2, 12):
        cmd._env.common_step_counter = step
        cmd._motion_term.speed_scale.fill_(0.1 if step % 2 else 4.0)
        cmd._advance_post_strike_elapsed(torch.zeros(1, dtype=torch.bool))
    age, valid = cmd.post_strike_age_and_same_attempt()
    assert age.item() == pytest.approx(0.20)
    assert valid.item() is True


def test_post_strike_clock_arms_on_positive_half_tick_exact_opportunity():
    cmd = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    cmd.num_envs = 1
    cmd.device = "cpu"
    cmd._env = types.SimpleNamespace(step_dt=0.02, common_step_counter=1)
    # exact_strike may fire at +8 ms, before the strict tts>0 pre-strike flag falls.
    cmd.pre_strike = torch.ones(1, dtype=torch.bool)
    cmd._post_strike_elapsed_s = torch.zeros(1)
    cmd._post_strike_elapsed_valid = torch.zeros(1, dtype=torch.bool)
    cmd._post_strike_elapsed_last_step = -1
    cmd._motion_term = types.SimpleNamespace(
        just_resampled=torch.zeros(1, dtype=torch.bool),
        event_just_installed=torch.zeros(1, dtype=torch.bool),
    )
    cmd._advance_post_strike_elapsed(torch.ones(1, dtype=torch.bool))
    age, valid = cmd.post_strike_age_and_same_attempt()
    assert age.item() == 0.0
    assert valid.item() is True

    cmd._env.common_step_counter = 2
    cmd.pre_strike.zero_()
    cmd._advance_post_strike_elapsed(torch.zeros(1, dtype=torch.bool))
    assert age.item() == pytest.approx(0.02)
    assert valid.item() is True


def _processed_qdes_slew_env(*, step_dt=0.02):
    n = 4
    names = list(_A3_JOINTS)
    selected = [
        index
        for index, name in enumerate(names)
        if name in hope_rewards_mod._PROCESSED_QDES_RECOVERY_JOINT_NAMES
    ]
    previous = torch.zeros(n, len(names))
    processed = torch.zeros_like(previous)
    # qdot limit 10 rad/s * 0.02 s = 0.2 rad allowance. env0 stays below margin;
    # env1 has u=1 on every selected joint; env2 is reset-invalid; env3 is off-window.
    processed[0, selected] = 0.1
    processed[1, selected] = 0.2
    processed[2, selected] = 2.0
    processed[3, selected] = 0.2
    # An arbitrarily large arm target must not enter the 15-joint mean.
    arm_index = names.index("right_elbow_joint")
    processed[:, arm_index] = 100.0
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            joint_vel_limits=torch.full((n, len(names)), 10.0),
        )
    )
    action = types.SimpleNamespace(
        processed_actions=processed,
        previous_processed_qdes=previous,
        previous_processed_qdes_valid=torch.tensor([True, True, False, True]),
        _asset=asset,
        _joint_names=names,
        _joint_ids=slice(None),
    )
    command = types.SimpleNamespace(
        post_strike_age_and_same_attempt=lambda: (
            torch.tensor([0.20, 1.55, 0.50, 0.50]),
            torch.tensor([True, True, True, False]),
        )
    )
    return types.SimpleNamespace(
        step_dt=step_dt,
        common_step_counter=17,
        action_manager=types.SimpleNamespace(get_term=lambda name: action),
        command_manager=types.SimpleNamespace(get_term=lambda name: command),
    )


def test_processed_qdes_slew_formula_gate_invalid_first_step_and_update_ledger():
    env = _processed_qdes_slew_env()
    probe = hope_rewards_mod.processed_qdes_slew_hinge_probe(env)
    reward = hope_rewards_mod.processed_qdes_slew_hinge(env)
    expected_tail = 1.0 - math.exp(-1.0)
    assert torch.equal(probe, torch.zeros(4))
    assert reward.tolist() == pytest.approx([0.0, expected_tail, 0.0, 0.0])

    counters = (
        hope_rewards_mod.consume_processed_qdes_slew_hinge_activation_counters(env)
    )
    assert counters["observed_sample_count"].item() == 4
    assert counters["previous_qdes_valid_sample_count"].item() == 3
    assert counters["previous_qdes_invalid_first_step_sample_count"].item() == 1
    assert counters["recovery_eligible_sample_count"].item() == 2
    assert counters["reward_enabled_eligible_sample_count"].item() == 2
    assert counters["tail_active_sample_count"].item() == 1
    assert counters["above_margin_joint_count"].item() == 15
    assert counters["gated_tail_value_sum"].item() == pytest.approx(expected_tail)
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_processed_qdes_slew_hinge_activation_counters(
            env
        ).values()
    )


def test_processed_qdes_slew_rejects_control_dt_drift():
    with pytest.raises(RuntimeError, match=r"step_dt == 0\.02"):
        hope_rewards_mod.processed_qdes_slew_hinge(
            _processed_qdes_slew_env(step_dt=0.01)
        )


def _exact_behavior_command(termination_manager, num_envs=4):
    command = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    command.device = "cpu"
    command.num_envs = num_envs
    command._env = types.SimpleNamespace(termination_manager=termination_manager)
    command._exact_behavior_decision_counters = {}
    command._exact_attempt_active = torch.zeros(num_envs, dtype=torch.bool)
    command._exact_attempt_completed = torch.zeros(num_envs, dtype=torch.bool)
    command._exact_pending_completion = torch.zeros(num_envs, dtype=torch.bool)
    command._clip_names = {0: "forehand", 1: "backhand"}
    sparse_names = (
        "strike_opportunity_count",
        "virtual_capture_count",
        "virtual_net_clear_count",
        "virtual_landing_valid_count",
        "virtual_legal_return_count",
    )
    command._sparse_reward_eligibility_counters = {
        name: torch.zeros((), dtype=torch.long) for name in sparse_names
    }
    for family in command._clip_names.values():
        for name in sparse_names:
            command._sparse_reward_eligibility_counters[f"{name}_{family}"] = torch.zeros(
                (), dtype=torch.long
            )
    command._ensure_exact_behavior_decision_counters()
    return command


def test_exact_behavior_terminal_reset_separates_physical_pre_post_and_guards():
    reasons = {
        "base_fell_tilt": torch.tensor([True, False, False, False]),
        "base_too_low": torch.tensor([False, False, True, True]),
        "anchor_pos": torch.tensor([False, True, False, False]),
        "time_out": torch.zeros(4, dtype=torch.bool),
    }
    tm = types.SimpleNamespace(
        active_terms=list(reasons),
        get_term=lambda name: reasons[name],
        terminated=torch.ones(4, dtype=torch.bool),
        time_outs=torch.zeros(4, dtype=torch.bool),
    )
    command = _exact_behavior_command(tm)
    command._exact_attempt_active[:] = True
    command._close_exact_swing_attempts(torch.arange(4))
    command._book_exact_behavior_terminal_reset(
        torch.arange(4),
        pre_strike=torch.tensor([True, True, False, True]),
        recovering=torch.tensor([False, False, False, True]),
    )

    first = command.consume_exact_behavior_decision_counters()
    assert first["terminal_reset_count"].item() == 4
    assert first["swing_outcome_count"].item() == 4
    assert first["swing_completion_count"].item() == 0
    assert first["physical_fall_count"].item() == 3
    assert first["pre_strike_physical_fall_count"].item() == 1
    assert first["post_strike_physical_fall_count"].item() == 2
    assert first["non_physical_terminal_reset_count"].item() == 1
    assert first["termination_reason_base_fell_tilt_count"].item() == 1
    assert first["termination_reason_base_too_low_count"].item() == 2
    assert first["termination_reason_anchor_pos_count"].item() == 1
    # Consume is a hard PPO-update boundary: the next transaction cannot inherit any event.
    second = command.consume_exact_behavior_decision_counters()
    assert all(value.item() == 0 for value in second.values())


def test_exact_swing_closeout_pairs_window_denominator_and_transfers_wrap_strike():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=2)
    # Two attempts started before this decision window.  Only the first reached its strike;
    # a defensive exact strike on the current wrap step is parked for the second env's NEW task.
    command._exact_attempt_active[:] = True
    command._exact_attempt_completed[:] = torch.tensor([True, False])
    command._exact_pending_completion[:] = torch.tensor([False, True])

    command._close_exact_swing_attempts(torch.arange(2))
    first = command.consume_exact_behavior_decision_counters()
    assert first["swing_start_count"].item() == 0
    assert first["strike_opportunity_count"].item() == 0
    assert first["swing_outcome_count"].item() == 2
    assert first["swing_completion_count"].item() == 1
    assert torch.equal(command._exact_attempt_completed, torch.tensor([False, True]))
    assert not bool(command._exact_pending_completion.any())

    # The parked wrap strike is paired only when its new attempt later closes; no prior-window
    # start/strike counter is needed to reconstruct this window's bounded completion ratio.
    command._close_exact_swing_attempts(torch.tensor([1]))
    second = command.consume_exact_behavior_decision_counters()
    assert second["swing_outcome_count"].item() == 1
    assert second["swing_completion_count"].item() == 1


def test_exact_initial_tts_buckets_pair_closeout_and_sparse_outcomes_per_update():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=4)
    command.planner_revision_enabled = True
    command._motion_term = types.SimpleNamespace(
        _multiseg=False,
        just_resampled=torch.zeros(4, dtype=torch.bool),
    )
    command._exact_attempt_active[:] = True
    with torch.inference_mode():
        command._assign_exact_attempt_initial_tts(
            torch.arange(4), torch.tensor([0.49, 0.50, 0.90, 0.91])
        )
        command._book_sparse_reward_eligibility(
            exact_strike=torch.tensor([True, True, True, False]),
            capture=torch.tensor([True, False, True, False]),
            net_clear=torch.tensor([True, False, True, False]),
            landing_valid=torch.tensor([True, False, True, False]),
            legal_return=torch.tensor([False, False, True, False]),
        )
        command._exact_attempt_completed[:] = torch.tensor([True, False, True, False])
        command._close_exact_swing_attempts(torch.arange(4))
    snapshot = command.consume_exact_behavior_decision_counters()

    for bucket in ("lt_0p5", "eq_0p5", "gt_0p5_le_0p9", "gt_0p9"):
        assert snapshot[f"planner_initial_tts_{bucket}_swing_outcome_count"].item() == 1
    assert snapshot["planner_initial_tts_lt_0p5_swing_completion_count"].item() == 1
    assert snapshot["planner_initial_tts_eq_0p5_swing_completion_count"].item() == 0
    assert snapshot["planner_initial_tts_gt_0p5_le_0p9_swing_completion_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_swing_completion_count"].item() == 0
    assert snapshot["planner_initial_tts_lt_0p5_strike_opportunity_count"].item() == 1
    assert snapshot["planner_initial_tts_eq_0p5_strike_opportunity_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p5_le_0p9_strike_opportunity_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_strike_opportunity_count"].item() == 0
    assert snapshot["planner_initial_tts_lt_0p5_virtual_capture_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p5_le_0p9_virtual_capture_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p5_le_0p9_virtual_legal_return_count"].item() == 1
    assert snapshot["planner_initial_tts_lt_0p5_virtual_legal_return_count"].item() == 0

    second = command.consume_exact_behavior_decision_counters()
    assert all(
        value.item() == 0
        for name, value in second.items()
        if name.startswith("planner_initial_tts_")
    )


def test_exact_initial_tts_bucket_defers_same_wrap_sparse_outcome_to_new_task():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=2)
    command.planner_revision_enabled = True
    command._motion_term = types.SimpleNamespace(
        _multiseg=False,
        just_resampled=torch.tensor([True, False]),
    )
    command._exact_attempt_active[:] = True
    command._assign_exact_attempt_initial_tts(
        torch.arange(2), torch.tensor([0.49, 0.50])
    )
    command._book_sparse_reward_eligibility(
        exact_strike=torch.tensor([True, True]),
        capture=torch.tensor([True, True]),
        net_clear=torch.tensor([True, True]),
        landing_valid=torch.tensor([True, True]),
        legal_return=torch.tensor([True, True]),
    )

    # Env 0's strike belongs to the task that starts on this wrap.  Close the old <0.5 task,
    # then assign the new >0.9 task and flush the parked outcomes into that new bucket.
    command._close_exact_swing_attempts(torch.tensor([0]))
    command._assign_exact_attempt_initial_tts(torch.tensor([0]), torch.tensor([1.10]))
    snapshot = command.consume_exact_behavior_decision_counters()
    assert snapshot["planner_initial_tts_lt_0p5_swing_outcome_count"].item() == 1
    assert snapshot["planner_initial_tts_lt_0p5_strike_opportunity_count"].item() == 0
    assert snapshot["planner_initial_tts_eq_0p5_strike_opportunity_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_strike_opportunity_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_virtual_capture_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_virtual_legal_return_count"].item() == 1


def test_exact_terminal_reason_masks_reject_numeric_truthiness():
    reasons = {"base_fell_tilt": torch.tensor([0.0, 0.2])}
    tm = types.SimpleNamespace(
        active_terms=list(reasons),
        get_term=lambda name: reasons[name],
        terminated=torch.tensor([False, True]),
        time_outs=torch.zeros(2, dtype=torch.bool),
    )
    command = _exact_behavior_command(tm, num_envs=2)
    with pytest.raises(TypeError, match="boolean dtype"):
        command._book_exact_behavior_terminal_reset(
            torch.arange(2),
            pre_strike=torch.ones(2, dtype=torch.bool),
            recovering=torch.zeros(2, dtype=torch.bool),
        )
    snapshot = command.consume_exact_behavior_decision_counters()
    assert snapshot["terminal_reset_count"].item() == 0
    assert snapshot["physical_fall_count"].item() == 0


def test_exact_terminal_reason_masks_reject_broadcastable_matrix_shape():
    reasons = {"base_fell_tilt": torch.zeros(2, 1, dtype=torch.bool)}
    tm = types.SimpleNamespace(
        active_terms=list(reasons),
        get_term=lambda name: reasons[name],
        terminated=torch.tensor([False, True]),
        time_outs=torch.zeros(2, dtype=torch.bool),
    )
    command = _exact_behavior_command(tm, num_envs=2)
    with pytest.raises(ValueError, match="one-dimensional"):
        command._book_exact_behavior_terminal_reset(
            torch.arange(2),
            pre_strike=torch.ones(2, dtype=torch.bool),
            recovering=torch.zeros(2, dtype=torch.bool),
        )


def test_exact_ready_balance_aggregates_have_explicit_phase_and_sensor_denominators():
    tm = types.SimpleNamespace(active_terms=())
    command = _exact_behavior_command(tm)
    command._motion_term = types.SimpleNamespace(
        event_timing_enabled=False,
        in_hold=torch.tensor([True, False, True, True]),
    )
    command._event_timing_bound = True
    command.metrics = {
        "base_upright": torch.cos(torch.tensor([0.0, 0.5, 0.2, 0.4])),
        "foot_contact_frac": torch.tensor([1.0, 0.0, 0.5, 1.0]),
        "foot_slip_speed": torch.tensor([0.1, 9.0, 0.2, 0.3]),
    }
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor(
                [[0.0, 0.0, 1.0], [9.0, 9.0, 1.0], [0.3, 0.4, 1.0], [0.0, 1.0, 1.0]]
            ),
            root_lin_vel_w=torch.tensor(
                [[0.3, 0.4, 0.0], [9.0, 9.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
        )
    )
    command.base_target_pos_w = torch.zeros(4, 2)
    command._foot_idx_robot = [0, 1]
    command._foot_idx_contact = [0, 1]
    command._contact_sensor = object()

    command._book_exact_ready_behavior_samples()
    first = command.consume_exact_behavior_decision_counters()
    assert first["ready_phase_sample_count"].item() == 3
    assert first["ready_planner_task_entry_sample_count"].item() == 0
    assert first["ready_planner_legacy_hold_violation_count"].item() == 0
    assert first["ready_foot_sensor_unavailable_sample_count"].item() == 0
    for key in (
        "ready_tilt_eligible_sample_count",
        "ready_base_speed_eligible_sample_count",
        "ready_station_offset_eligible_sample_count",
        "ready_foot_contact_eligible_sample_count",
        "ready_foot_slip_eligible_sample_count",
    ):
        assert first[key].item() == 3
    assert first["ready_tilt_rad_sum"].item() == pytest.approx(0.6, abs=1e-5)
    assert first["ready_base_speed_xy_mps_sum"].item() == pytest.approx(1.5)
    assert first["ready_station_offset_m_sum"].item() == pytest.approx(1.5)
    assert first["ready_foot_contact_fraction_sum"].item() == pytest.approx(2.5)
    assert first["ready_foot_slip_speed_mps_sum"].item() == pytest.approx(0.6)

    # No hold eligibility means no fabricated zero-valued observation in the next update.
    command._motion_term.in_hold.zero_()
    command._book_exact_ready_behavior_samples()
    second = command.consume_exact_behavior_decision_counters()
    assert second["ready_tilt_eligible_sample_count"].item() == 0
    assert second["ready_foot_contact_eligible_sample_count"].item() == 0


def test_exact_ready_planner_first_post_install_sample_has_nonzero_denominators_once():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()))
    command.planner_revision_enabled = True
    command._motion_term = types.SimpleNamespace(
        event_timing_enabled=False,
        planner_revision_enabled=True,
        in_hold=torch.zeros(4, dtype=torch.bool),
        _planner_active=torch.ones(4, dtype=torch.bool),
    )
    command._event_timing_bound = True
    command._planner_control_epoch = torch.tensor([1, 1, 2, 2])
    command._planner_task_id = torch.tensor([1, 2, 1, 3])
    command._planner_task_revision = torch.ones(4, dtype=torch.long)
    command.metrics = {
        "base_upright": torch.cos(torch.tensor([0.1, 0.2, 0.3, 0.4])),
        "foot_contact_frac": torch.tensor([1.0, 0.5, 1.0, 0.5]),
        "foot_slip_speed": torch.tensor([0.0, 0.1, 0.2, 0.3]),
    }
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.2, 1.0], [0.3, 0.4, 1.0]]
            ),
            root_lin_vel_w=torch.tensor(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.3, 0.4, 0.0]]
            ),
        )
    )
    command.base_target_pos_w = torch.zeros(4, 2)
    command._foot_idx_robot = [0, 1]
    command._foot_idx_contact = [0, 1]
    command._contact_sensor = object()

    command._book_exact_ready_behavior_samples()
    first = command.consume_exact_behavior_decision_counters()
    assert first["ready_phase_sample_count"].item() == 4
    assert first["ready_planner_task_entry_sample_count"].item() == 4
    assert first["ready_planner_legacy_hold_violation_count"].item() == 0
    for key in (
        "ready_tilt_eligible_sample_count",
        "ready_base_speed_eligible_sample_count",
        "ready_station_offset_eligible_sample_count",
        "ready_foot_contact_eligible_sample_count",
        "ready_foot_slip_eligible_sample_count",
    ):
        assert first[key].item() == 4
    assert first["ready_foot_sensor_unavailable_sample_count"].item() == 0

    # A same-ball revision changes only task_revision and must not create another ready sample.
    command._planner_task_revision += 17
    command._book_exact_ready_behavior_samples()
    second = command.consume_exact_behavior_decision_counters()
    assert second["ready_phase_sample_count"].item() == 0
    assert second["ready_tilt_eligible_sample_count"].item() == 0

    # A same-episode next ball changes task_id and produces one new task-entry sample.
    command._planner_task_id[2] += 1
    command._book_exact_ready_behavior_samples()
    third = command.consume_exact_behavior_decision_counters()
    assert third["ready_phase_sample_count"].item() == 1
    assert third["ready_planner_task_entry_sample_count"].item() == 1
    assert third["ready_foot_contact_eligible_sample_count"].item() == 1

    # A partial vector-env true reset reuses task_id=1 under a new epoch.  Only the reset env is
    # new; unchanged neighbors must keep their one-shot latches.
    command._planner_control_epoch[1] += 1
    command._planner_task_id[1] = 1
    command._book_exact_ready_behavior_samples()
    fourth = command.consume_exact_behavior_decision_counters()
    assert fourth["ready_phase_sample_count"].item() == 1
    assert fourth["ready_planner_task_entry_sample_count"].item() == 1
    assert fourth["ready_tilt_eligible_sample_count"].item() == 1

    # An installed identity is not sampled while inactive and is not prematurely latched.  It is
    # sampled exactly once when the task becomes active on a later metrics tick.
    command._planner_task_id[3] += 1
    command._motion_term._planner_active[3] = False
    command._book_exact_ready_behavior_samples()
    inactive = command.consume_exact_behavior_decision_counters()
    assert inactive["ready_phase_sample_count"].item() == 0
    command._motion_term._planner_active[3] = True
    command._book_exact_ready_behavior_samples()
    activated = command.consume_exact_behavior_decision_counters()
    assert activated["ready_phase_sample_count"].item() == 1
    assert activated["ready_planner_task_entry_sample_count"].item() == 1


def test_exact_ready_planner_legacy_hold_is_violation_not_eligibility():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=3)
    command.planner_revision_enabled = True
    command._motion_term = types.SimpleNamespace(
        event_timing_enabled=False,
        planner_revision_enabled=True,
        in_hold=torch.tensor([True, True, False]),
        _planner_active=torch.tensor([True, False, True]),
    )
    command._event_timing_bound = True
    command._planner_control_epoch = torch.ones(3, dtype=torch.long)
    command._planner_task_id = torch.ones(3, dtype=torch.long)
    command.metrics = {
        "base_upright": torch.ones(3),
        "foot_contact_frac": torch.ones(3),
        "foot_slip_speed": torch.zeros(3),
    }
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor(
                [[0.0, 0.0, 1.0], [9.0, 9.0, 1.0], [0.1, 0.2, 1.0]]
            ),
            root_lin_vel_w=torch.zeros(3, 3),
        )
    )
    command.base_target_pos_w = torch.zeros(3, 2)
    command._foot_idx_robot = [0, 1]
    command._foot_idx_contact = [0, 1]
    command._contact_sensor = object()

    command._book_exact_ready_behavior_samples()
    first = command.consume_exact_behavior_decision_counters()
    # Both unexpected hold envs are witnessed, but only the two active new task identities enter
    # the ready denominator.  In particular, inactive env 1's hold cannot leak into the sums.
    assert first["ready_planner_legacy_hold_violation_count"].item() == 2
    assert first["ready_phase_sample_count"].item() == 2
    assert first["ready_planner_task_entry_sample_count"].item() == 2
    assert first["ready_station_offset_eligible_sample_count"].item() == 2
    assert first["ready_station_offset_m_sum"].item() == pytest.approx(
        math.sqrt(0.1**2 + 0.2**2)
    )

    # The same identities stay one-shot.  A persistent illegal hold remains visible as a fresh
    # violation each metrics tick but never creates another ready observation.
    command._book_exact_ready_behavior_samples()
    second = command.consume_exact_behavior_decision_counters()
    assert second["ready_planner_legacy_hold_violation_count"].item() == 2
    assert second["ready_phase_sample_count"].item() == 0
    assert second["ready_tilt_eligible_sample_count"].item() == 0


def test_exact_ready_missing_foot_sensor_is_explicitly_unavailable_not_zero_observation():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=2)
    command._motion_term = types.SimpleNamespace(
        event_timing_enabled=False,
        in_hold=torch.tensor([True, True]),
    )
    command._event_timing_bound = True
    command.metrics = {
        "base_upright": torch.ones(2),
        # Historical zero-filled metric buffers may exist even when no sensor resolved.  They must
        # not become eligible observations.
        "foot_contact_frac": torch.zeros(2),
        "foot_slip_speed": torch.zeros(2),
    }
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]]),
            root_lin_vel_w=torch.zeros(2, 3),
        )
    )
    command.base_target_pos_w = torch.zeros(2, 2)
    command._foot_idx_robot = []
    command._foot_idx_contact = []
    command._contact_sensor = None

    command._book_exact_ready_behavior_samples()
    snapshot = command.consume_exact_behavior_decision_counters()
    assert snapshot["ready_phase_sample_count"].item() == 2
    assert snapshot["ready_tilt_eligible_sample_count"].item() == 2
    assert snapshot["ready_foot_contact_eligible_sample_count"].item() == 0
    assert snapshot["ready_foot_slip_eligible_sample_count"].item() == 0
    assert snapshot["ready_foot_contact_fraction_sum"].item() == 0.0
    assert snapshot["ready_foot_slip_speed_mps_sum"].item() == 0.0
    assert snapshot["ready_foot_sensor_unavailable_sample_count"].item() == 2


def test_exact_planner_revision_activation_counters_book_and_reset_per_update():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()))
    command.planner_revision_enabled = True
    command._planner_revision_profile = types.SimpleNamespace(
        min_tts_s=0.02, early_deadline_tolerance_s=1.0e-6
    )
    command._planner_initial_tts_mixture = None

    command._book_planner_revision_decisions(
        attempted_tts=torch.tensor([0.10, 0.02000001, 0.02, 0.08]),
        accepted=torch.tensor([True, True, False, False]),
    )
    first = command.consume_exact_behavior_decision_counters()
    assert first["planner_revision_attempt_count"].item() == 4
    assert first["planner_revision_accepted_count"].item() == 2
    assert first["planner_revision_rejected_count"].item() == 2
    assert first["planner_revision_last_precontact_attempt_count"].item() == 2
    assert first["planner_revision_last_precontact_accepted_count"].item() == 1

    second = command.consume_exact_behavior_decision_counters()
    for name in (
        "planner_revision_attempt_count",
        "planner_revision_accepted_count",
        "planner_revision_rejected_count",
        "planner_revision_last_precontact_attempt_count",
        "planner_revision_last_precontact_accepted_count",
        "planner_revision_actor_visible_count",
        "planner_revision_last_precontact_actor_visible_count",
    ):
        assert second[name].item() == 0


def test_training_governor_keeps_final_actor_interval_for_point_zero_two_revision():
    motion = commands_mod.MotionCommand.__new__(commands_mod.MotionCommand)
    motion.device = "cpu"
    motion.planner_revision_enabled = True
    motion._planner_revision_profile = planner_revision_mod.PhaseGovernorProfile(
        policy_dt_s=0.02,
        min_tts_s=0.02,
        max_tts_s=2.0,
        max_phase_rate_per_s=4.0,
        max_phase_acceleration_per_s2=20.0,
        max_deadline_revision_delta_s=0.25,
        max_position_revision_delta_m=0.10,
        max_velocity_revision_delta_mps=0.50,
        max_normal_revision_delta_rad=0.20,
        early_deadline_tolerance_s=1.0e-6,
    )
    motion.time_steps_f = torch.zeros(1)
    motion.speed_scale = torch.ones(1)
    motion.metrics = {
        "planner_revision_accepted": torch.zeros(1),
        "planner_revision_rejected": torch.zeros(1),
    }
    motion._planner_active = torch.zeros(1, dtype=torch.bool)
    motion._planner_control_epoch = torch.zeros(1, dtype=torch.long)
    motion._planner_task_id = torch.zeros(1, dtype=torch.long)
    motion._planner_task_revision = torch.full((1,), -1, dtype=torch.long)
    motion._planner_start_step = torch.zeros(1)
    motion._planner_strike_step = torch.zeros(1)
    motion._planner_phase_rate = torch.zeros(1)
    motion._planner_slow_only_next = torch.zeros(1, dtype=torch.bool)
    motion._planner_desired_tts = torch.zeros(1)
    motion._planner_begin_tts = torch.zeros(1)
    motion._planner_truth_tts = torch.zeros(1)
    motion._planner_begin_target_pos = torch.zeros(1, 3)
    motion._planner_begin_target_vel = torch.zeros(1, 3)
    motion._planner_begin_target_normal = torch.zeros(1, 3)

    ids = torch.tensor([0], dtype=torch.long)
    position = torch.tensor([[0.2, -0.1, 0.9]])
    velocity = torch.tensor([[-2.0, 0.0, -0.2]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    motion.begin_planner_task(
        ids,
        control_epoch=torch.tensor([7]),
        task_id=torch.tensor([10]),
        strike_step=torch.tensor([1.0]),
        initial_tts=torch.tensor([0.5]),
        target_position=position,
        target_velocity=velocity,
        target_normal=normal,
    )

    accepted_tts = []
    revision = 1
    for tick in range(1, 25):
        delta = motion._advance_planner_phase(torch.tensor([False]))
        motion.time_steps_f += delta
        if tick >= 20:
            revision += 1
            # Exercise the actual float32 countdown path; 0.50 - 24*0.02 is
            # 0.0199999101 on the training grid, not the rounded decimal 0.02.
            tts = motion._planner_truth_tts[ids].clone()
            accepted = motion.submit_planner_revision(
                ids,
                control_epoch=torch.tensor([7]),
                task_id=torch.tensor([10]),
                task_revision=torch.tensor([revision]),
                desired_tts=tts,
                target_position=position,
                target_velocity=velocity,
                target_normal=normal,
            )
            assert accepted.tolist() == [True]
            accepted_tts.append(tts.item())

    assert accepted_tts == pytest.approx([0.10, 0.08, 0.06, 0.04, 0.02], abs=1.0e-6)
    assert torch.equal(
        motion._planner_desired_tts[ids], torch.full((1,), 0.02)
    )
    assert motion.time_steps_f.item() < 1.0

    delta = motion._advance_planner_phase(torch.tensor([False]))
    motion.time_steps_f += delta
    assert motion.time_steps_f.item() == pytest.approx(1.0)
    post_contact = motion.submit_planner_revision(
        ids,
        control_epoch=torch.tensor([7]),
        task_id=torch.tensor([10]),
        task_revision=torch.tensor([revision + 1]),
        desired_tts=torch.tensor([0.02]),
        target_position=position,
        target_velocity=velocity,
        target_normal=normal,
    )
    assert post_contact.tolist() == [False]


def test_planner_revision_metrics_remain_full_env_after_eligible_set_shrinks():
    """Compact revision decisions must remain safe for CommandTerm.reset global indexing."""

    num_envs = 4096
    motion = commands_mod.MotionCommand.__new__(commands_mod.MotionCommand)
    motion.num_envs = num_envs
    motion.device = "cpu"
    motion.planner_revision_enabled = True
    motion._planner_revision_profile = planner_revision_mod.PhaseGovernorProfile(
        policy_dt_s=0.02,
        min_tts_s=0.02,
        max_tts_s=2.0,
        max_phase_rate_per_s=4.0,
        max_phase_acceleration_per_s2=20.0,
        max_deadline_revision_delta_s=0.25,
        max_position_revision_delta_m=0.10,
        max_velocity_revision_delta_mps=0.50,
        max_normal_revision_delta_rad=0.20,
        early_deadline_tolerance_s=1.0e-6,
    )
    motion.time_steps_f = torch.zeros(num_envs)
    motion.speed_scale = torch.ones(num_envs)
    motion.metrics = {
        "planner_revision_accepted": torch.ones(num_envs),
        "planner_revision_rejected": torch.ones(num_envs),
        "planner_phase_rate_per_s": torch.zeros(num_envs),
        "planner_truth_tts_s": torch.zeros(num_envs),
    }
    motion._planner_active = torch.ones(num_envs, dtype=torch.bool)
    motion._planner_control_epoch = torch.full((num_envs,), 7, dtype=torch.long)
    motion._planner_task_id = torch.full((num_envs,), 10, dtype=torch.long)
    motion._planner_task_revision = torch.ones(num_envs, dtype=torch.long)
    motion._planner_start_step = torch.zeros(num_envs)
    motion._planner_strike_step = torch.full((num_envs,), 100.0)
    motion._planner_phase_rate = torch.zeros(num_envs)
    motion._planner_slow_only_next = torch.zeros(num_envs, dtype=torch.bool)
    motion._planner_desired_tts = torch.full((num_envs,), 0.50)
    motion._planner_begin_tts = torch.full((num_envs,), 0.50)
    motion._planner_truth_tts = torch.full((num_envs,), 0.50)
    motion._planner_begin_target_pos = torch.zeros(num_envs, 3)
    motion._planner_begin_target_vel = torch.zeros(num_envs, 3)
    motion._planner_begin_target_normal = torch.zeros(num_envs, 3)
    motion._planner_begin_target_normal[:, 0] = 1.0

    # The phase step clears stale decisions for every environment before the currently eligible
    # subset submits its compact revision rows.
    motion._advance_planner_phase(torch.zeros(num_envs, dtype=torch.bool))
    ids = torch.tensor([1, 4001], dtype=torch.long)
    accepted = motion.submit_planner_revision(
        ids,
        control_epoch=torch.tensor([7, 7]),
        task_id=torch.tensor([10, 999]),  # one accept, one deliberate reject
        task_revision=torch.tensor([2, 2]),
        desired_tts=torch.tensor([0.48, 0.48]),
        target_position=torch.zeros(2, 3),
        target_velocity=torch.zeros(2, 3),
        target_normal=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert torch.equal(accepted, torch.tensor([True, False]))
    assert motion.metrics["planner_revision_accepted"].shape == (num_envs,)
    assert motion.metrics["planner_revision_rejected"].shape == (num_envs,)
    assert motion.metrics["planner_revision_accepted"].sum().item() == 1
    assert motion.metrics["planner_revision_rejected"].sum().item() == 1
    assert motion.metrics["planner_revision_accepted"][1].item() == 1
    assert motion.metrics["planner_revision_rejected"][4001].item() == 1

    # Reproduce the reset pattern from the A4 failure: the last three reset positions carry high
    # global env ids. Every registered metric must remain safely gatherable by those ids.
    reset_env_ids = torch.tensor(list(range(68)) + [4001, 4050, 4095], dtype=torch.long)
    for metric in motion.metrics.values():
        assert metric.shape[0] == num_envs
        assert metric[reset_env_ids].shape[0] == len(reset_env_ids)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
