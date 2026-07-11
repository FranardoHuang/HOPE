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
* A1  target position/velocity/demanded-normal/swing-sign remain one atomic actor message through
      delay/dropout, and a true reset clears every stateful sensor-defect buffer.
* P2.4 base_decel is zero during a frozen hold; debug raw/gated telemetry records the real mask.
* R-c commands.MotionCommand on a synthetic 2-clip npz pair: rsi_skip_settle_frames offsets the
      swing-entry frame (multiseg + single-clip clamp, incl. the short-clip clamp) and
      rsi_hold_root_stand_z rewrites ONLY the held-RSI birth root z to the default stand height.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_reward_flags_mdp.py -q
"""

from __future__ import annotations

import copy
import importlib.util
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
commands_mod = _load(f"{_PKG}.commands", "commands.py")
rewards_mod = _load(f"{_PKG}.rewards", "rewards.py")
terminations_mod = _load(f"{_PKG}.terminations", "terminations.py")
_load(f"{_PKG}.stage1_question_bank", "stage1_question_bank.py")
hope_commands_mod = _load(f"{_PKG}.hope_commands", "hope_commands.py")
hope_rewards_mod = _load(f"{_PKG}.hope_rewards", "hope_rewards.py")
hope_observations_mod = _load(f"{_PKG}.hope_observations", "hope_observations.py")


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
        cfg=types.SimpleNamespace(body_names=["torso_Link", "right_wrist_yaw_Link"]),
        anchor_pos_w=torch.zeros(n, 3), robot_anchor_pos_w=torch.zeros(n, 3),
        anchor_quat_w=torch.zeros(n, 4), robot_anchor_quat_w=torch.zeros(n, 4),
        body_pos_relative_w=zeros_b.clone(), robot_body_pos_w=zeros_b.clone(),
        body_quat_relative_w=quat.clone(), robot_body_quat_w=quat.clone(),
        body_lin_vel_w=zeros_b.clone(), robot_body_lin_vel_w=zeros_b.clone(),
        body_ang_vel_w=zeros_b.clone(), robot_body_ang_vel_w=zeros_b.clone(),
        in_hold=torch.zeros(n, dtype=torch.bool),
    )


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
def _make_a1_cmd(delay_steps=0, dropout_prob=0.0, ar1_sigma=0.0):
    cmd = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    cmd.num_envs = 1
    cmd.device = "cpu"
    cmd._actor_view_active = True
    cmd._delay_steps = delay_steps
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
    if delay_steps > 0:
        length = delay_steps + 1
        cmd._delay_buf_pos = cmd.racket_target_pos_w.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_vel = cmd.racket_target_vel_w.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_normal = cmd.target_normal_cmd.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_sign = cmd.swing_sign.unsqueeze(0).repeat(length, 1)
    if cmd._a1v2_active:
        cmd._swing_bias = torch.zeros(1, 3)
        cmd._drop_cd = torch.zeros(1, dtype=torch.long)
        cmd._prev_pre_strike = torch.ones(1, dtype=torch.bool)
        cmd._held_pos = cmd.racket_target_pos_w.clone()
        cmd._held_vel = cmd.racket_target_vel_w.clone()
        cmd._held_normal = cmd.target_normal_cmd.clone()
        cmd._held_sign = cmd.swing_sign.clone()
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


def test_a1_dropout_holds_the_entire_target_message():
    cmd = _make_a1_cmd(dropout_prob=1.0)
    _set_question_b(cmd)
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    cmd._drop_prob = 0.0
    cmd._push_actor_target()
    _assert_actor_question(cmd, "b")


def test_a1_true_reset_clears_state_and_backfills_all_message_fields():
    cmd = _make_a1_cmd(delay_steps=2, dropout_prob=1.0, ar1_sigma=0.2)
    _set_question_b(cmd)
    cmd._drop_cd[:] = 9
    cmd._swing_bias[:] = 3.0
    cmd._held_pos[:] = -3.0
    cmd._held_vel[:] = -4.0
    cmd._held_normal[:] = -5.0
    cmd._held_sign[:] = 0.0
    cmd._reset_actor_target_state(torch.tensor([0]))
    _assert_actor_question(cmd, "b")
    assert torch.all(cmd._delay_buf_pos == cmd.racket_target_pos_w.unsqueeze(0))
    assert torch.all(cmd._delay_buf_vel == cmd.racket_target_vel_w.unsqueeze(0))
    assert torch.all(cmd._delay_buf_normal == cmd.target_normal_cmd.unsqueeze(0))
    assert torch.all(cmd._delay_buf_sign == cmd.swing_sign.unsqueeze(0))
    assert torch.all(cmd._mnoise_ar1_state == 0.0)
    assert torch.all(cmd._swing_bias == 0.0)
    assert torch.all(cmd._drop_cd == 0)
    assert torch.equal(cmd._held_normal, cmd.target_normal_cmd)
    assert torch.equal(cmd._held_sign, cmd.swing_sign)


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
        default_root = torch.zeros(num_envs, 13)
        default_root[:, 2] = _STAND_Z
        default_root[:, 3] = 1.0
        self.data = types.SimpleNamespace(
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

    def write_joint_state_to_sim(self, jp, jv, env_ids=None):
        self.calls.append(("joint", jp.clone(), jv.clone()))


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
