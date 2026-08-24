"""Host tests for the no-ball Stage-1 natural-clip official-racket-site reward."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_rewards.py"
)


def _load_rewards():
    canonical = "whole_body_tracking.tasks.tracking.mdp.hope_commands"
    mdp_canonical = "whole_body_tracking.tasks.tracking.mdp"
    geometry_canonical = f"{mdp_canonical}.racket_contact_geometry"
    old = sys.modules.get(canonical)
    old_mdp = sys.modules.get(mdp_canonical)
    old_geometry = sys.modules.get(geometry_canonical)
    stub = types.ModuleType(canonical)
    stub.RacketTargetCommand = object
    stub.face_tracking_pair = lambda command: (
        command.racket_normal_w,
        command.racket_target_normal_w,
    )
    mdp_stub = types.ModuleType(mdp_canonical)
    mdp_stub.__path__ = []
    geometry_spec = importlib.util.spec_from_file_location(
        geometry_canonical, REWARD_PATH.with_name("racket_contact_geometry.py")
    )
    assert geometry_spec is not None and geometry_spec.loader is not None
    geometry = importlib.util.module_from_spec(geometry_spec)
    sys.modules[mdp_canonical] = mdp_stub
    sys.modules[geometry_canonical] = geometry
    geometry_spec.loader.exec_module(geometry)
    mdp_stub.racket_contact_geometry = geometry
    sys.modules[canonical] = stub
    try:
        spec = importlib.util.spec_from_file_location("stage1_clip_site_rewards_under_test", REWARD_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            sys.modules.pop(canonical, None)
        else:
            sys.modules[canonical] = old
        if old_mdp is None:
            sys.modules.pop(mdp_canonical, None)
        else:
            sys.modules[mdp_canonical] = old_mdp
        if old_geometry is None:
            sys.modules.pop(geometry_canonical, None)
        else:
            sys.modules[geometry_canonical] = old_geometry


@pytest.fixture(scope="module")
def rewards():
    return _load_rewards()


def _yaw_quat(angle: float) -> torch.Tensor:
    return torch.tensor([torch.cos(torch.tensor(angle / 2)), 0.0, 0.0, torch.sin(torch.tensor(angle / 2))])


def test_pure_helper_differentiates_the_offset_site_not_body_com(rewards):
    theta = 0.2
    previous_quat = _yaw_quat(-theta).unsqueeze(0)
    current_quat = _yaw_quat(0.0).unsqueeze(0)
    next_quat = _yaw_quat(theta).unsqueeze(0)
    body_pos = torch.zeros(1, 3)

    site, normal, velocity = rewards.stage1_clip_site_target_from_aligned_body_pose(
        body_pos,
        previous_quat,
        body_pos,
        current_quat,
        body_pos,
        next_quat,
        mount_offset_body=torch.tensor([1.0, 0.0, 0.0]),
        mount_quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        normal_axis=1,
        normal_sign=-1.0,
        central_difference_span_s=0.2,
    )

    torch.testing.assert_close(site, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(normal, torch.tensor([[0.0, -1.0, 0.0]]))
    # The body origin is stationary, but the official site moves under angular motion.  Reading a
    # stored wrist/body COM velocity would incorrectly produce zero here.
    torch.testing.assert_close(
        velocity,
        torch.tensor([[0.0, 2.0 * torch.sin(torch.tensor(theta)) / 0.2, 0.0]]),
    )


def test_measured_racket_helper_uses_physical_blade_channel_directly(rewards):
    position, normal, velocity = (
        rewards.stage1_clip_site_target_from_aligned_measured_racket(
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.2, 0.4, 0.6]]),
            torch.tensor([[0.5, 0.8, 1.1]]),
            torch.tensor([[0.0, 3.0, 0.0]]),
            central_difference_span_s=0.2,
        )
    )
    torch.testing.assert_close(position, torch.tensor([[0.2, 0.4, 0.6]]))
    torch.testing.assert_close(normal, torch.tensor([[0.0, 1.0, 0.0]]))
    torch.testing.assert_close(velocity, torch.tensor([[2.0, 3.0, 4.0]]))


def test_measured_racket_helper_rejects_zero_face_normal(rewards):
    with pytest.raises(RuntimeError):
        rewards.stage1_clip_site_target_from_aligned_measured_racket(
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.ones(1, 3),
            torch.zeros(1, 3),
            central_difference_span_s=0.2,
        )


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "racket_target"
        return self.command


def _fake_command():
    env_count = 2
    frames = 5
    body_pos = torch.zeros(frames, 3, 3)
    body_quat = torch.zeros(frames, 3, 4)
    body_quat[..., 0] = 1.0
    for frame in range(frames):
        # raw body 0 is deliberately not the tracked anchor.  The historical
        # local-index bug read it when motion_anchor_body_index=0.
        body_pos[frame, 0] = torch.tensor([100.0 + frame, 0.0, 4.0])
        body_pos[frame, 1] = torch.tensor([1.0 + 0.3 * frame, 0.2, 1.5])
        body_pos[frame, 2] = torch.tensor([0.2 * frame, 0.0, 1.0])
    loader = types.SimpleNamespace(
        _body_pos_w=body_pos,
        _body_quat_w=body_quat,
        # Poisoned by design: the Stage-1 target must never consume this COM-point channel.
        _body_lin_vel_w=torch.full_like(body_pos, float("nan")),
        seg_start=torch.tensor([0]),
        seg_len=torch.tensor([frames]),
        num_segments=1,
        time_step_total=frames,
    )
    motion = types.SimpleNamespace(
        motion=loader,
        _multiseg=False,
        _action_ball_continuous_motion_mutation_version=0,
        motion_anchor_body_index=0,
        robot_anchor_body_index=2,
        robot_anchor_pos_w=torch.tensor([[10.0, 0.0, 1.0], [20.0, 0.0, 1.0]]),
        robot_anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(env_count, 1),
        speed_scale=torch.ones(env_count),
        in_hold=torch.zeros(env_count, dtype=torch.bool),
        _pose_reference_steps=lambda: torch.tensor([2, 2]),
    )
    owner_env = types.SimpleNamespace(
        common_step_counter=7,
        step_dt=0.02,
        scene=types.SimpleNamespace(env_origins=torch.zeros(env_count, 3)),
    )
    command = types.SimpleNamespace(
        _env=owner_env,
        device=torch.device("cpu"),
        num_envs=env_count,
        _motion=lambda: motion,
        _racket_mode="body",
        _racket_body_index=1,
        _wrist_body_index=-1,
        cfg=types.SimpleNamespace(
            mount_normal_sign_per_clip=(),
            mount_normal_sign=1.0,
            mount_normal_axis=1,
            debug_reward_logging=False,
            strike_phase=0.75,
            strike_phase_per_clip=(),
        ),
        _strike_phases_cfg=lambda count: (),
        strike_window=torch.tensor([True, True]),
        strike_window_pos=torch.tensor([True, False]),
        strike_window_wide=torch.tensor([True, False]),
        metrics={},
    )
    command._strike_steps_for_envs = lambda ids: torch.full(
        (len(ids),),
        float(round(command.cfg.strike_phase * (frames - 1))),
        device=ids.device,
    )
    return command


def _fresh_measured_command():
    command = _fake_command()
    motion = command._motion()
    loader = motion.motion
    loader.measured_racket_available = True
    loader._measured_racket_site_pos_w = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.3, 0.0, 1.0],
            [0.6, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    # This artifact channel is already signed for the selected action.
    loader._measured_racket_normal_w = torch.tensor(
        [[0.0, -1.0, 0.0]]
    ).repeat(loader.time_step_total, 1)
    loader._measured_racket_long_axis_w = torch.tensor(
        [[1.0, 0.0, 0.0]]
    ).repeat(loader.time_step_total, 1)
    command.cfg.motion_teacher_racket_source = "measured_channel"

    motion._action_ball_continuous_fresh_motion_lane_bound = True
    motion.action_ball_diagnostic_split_ready_teacher = False
    motion.ready_wait = torch.ones(command.num_envs, dtype=torch.bool)
    motion.safe_steps = torch.zeros(command.num_envs, dtype=torch.long)
    motion._action_ball_safe_ready_wait_mask = lambda: motion.ready_wait
    motion._action_ball_full_mdp_safe_pose_reference_steps = (
        lambda: motion.safe_steps
    )
    motion.robot = types.SimpleNamespace(
        body_names=("root", "racket", "anchor")
    )
    motion.cfg = types.SimpleNamespace(
        body_names=("root", "racket", "anchor")
    )
    motion.body_pos_relative_w = torch.zeros(command.num_envs, 3, 3)
    motion.body_pos_relative_w[:, 1] = torch.tensor(
        [[30.0, 0.0, 1.0], [40.0, 0.0, 1.0]]
    )
    motion.body_quat_relative_w = torch.zeros(command.num_envs, 3, 4)
    motion.body_quat_relative_w[..., 0] = 1.0
    command.racket_normal_raw_w = torch.tensor(
        [[0.0, 1.0, 0.0]]
    ).repeat(command.num_envs, 1)
    command.racket_normal_w = -command.racket_normal_raw_w
    return command


def test_reward_wrappers_use_full_phase_teacher_and_separate_precision_windows(rewards):
    command = _fake_command()
    target_pos, target_normal, target_velocity = rewards._stage1_aligned_clip_site_target(command)
    torch.testing.assert_close(
        target_pos,
        torch.tensor([[11.2, 0.2, 1.5], [21.2, 0.2, 1.5]]),
    )
    command.racket_pos_w = target_pos.clone()
    command.racket_normal_w = target_normal.clone()
    command.racket_lin_vel_w = target_velocity.clone()
    env = types.SimpleNamespace(command_manager=_CommandManager(command))

    pos = rewards.stage1_clip_racket_position_tracking_exp(env, "racket_target", 0.30)
    normal = rewards.stage1_clip_racket_normal_tracking_exp(env, "racket_target", 0.60)
    velocity = rewards.stage1_clip_racket_velocity_tracking_exp(env, "racket_target", 1.0)
    coarse_pos = rewards.stage1_clip_racket_position_coarse_tracking_exp(
        env, "racket_target", 0.70
    )
    coarse_normal = rewards.stage1_clip_racket_normal_coarse_tracking_exp(
        env, "racket_target", math.pi
    )
    coarse_velocity = rewards.stage1_clip_racket_velocity_coarse_tracking_exp(
        env, "racket_target", 4.0
    )
    precision_pos = rewards.stage1_clip_racket_position_precision_tracking_exp(
        env, "racket_target", 0.075
    )
    precision_normal = rewards.stage1_clip_racket_normal_precision_tracking_exp(
        env, "racket_target", 0.262
    )
    precision_velocity = rewards.stage1_clip_racket_velocity_precision_tracking_exp(
        env, "racket_target", 0.50
    )

    torch.testing.assert_close(pos, torch.ones(2))
    torch.testing.assert_close(normal, torch.ones(2))
    torch.testing.assert_close(velocity, torch.ones(2))
    torch.testing.assert_close(coarse_pos, torch.ones(2))
    torch.testing.assert_close(coarse_normal, torch.ones(2))
    torch.testing.assert_close(coarse_velocity, torch.ones(2))
    torch.testing.assert_close(precision_pos, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(precision_normal, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(precision_velocity, torch.tensor([1.0, 0.0]))
    assert torch.isfinite(target_pos).all()
    assert torch.isfinite(target_velocity).all()


def test_mixed_hold_zeroes_teacher_velocity_without_changing_pose_or_moving_rate(rewards):
    command = _fake_command()
    motion = command._motion()
    motion.in_hold = torch.tensor([True, False])
    motion.speed_scale = torch.tensor([0.0, 0.5])

    position, normal, velocity = rewards._stage1_aligned_clip_site_target_at_steps(
        command, torch.tensor([2, 2])
    )

    torch.testing.assert_close(
        position,
        torch.tensor([[11.2, 0.2, 1.5], [21.2, 0.2, 1.5]]),
    )
    torch.testing.assert_close(normal, torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1))
    torch.testing.assert_close(velocity[0], torch.zeros(3))
    # Source body 1 moves 0.6 m over the two-frame 40 ms stencil: 15 m/s at native rate,
    # therefore 7.5 m/s at a 0.5 playback rate.
    torch.testing.assert_close(velocity[1], torch.tensor([7.5, 0.0, 0.0]))


def test_zero_speed_outside_hold_fails_closed(rewards):
    command = _fake_command()
    command._motion().speed_scale[0] = 0.0

    with pytest.raises(RuntimeError):
        rewards._stage1_aligned_clip_site_target_at_steps(command, torch.tensor([2, 2]))


def test_measured_channel_hold_zeroes_velocity_but_preserves_measured_geometry(rewards):
    command = _fake_command()
    motion = command._motion()
    loader = motion.motion
    loader.measured_racket_available = True
    loader._measured_racket_site_pos_w = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.3, 0.0, 1.0], [0.6, 0.0, 1.0], [1.0, 0.0, 1.0]]
    )
    loader._measured_racket_normal_w = torch.tensor([[0.0, -1.0, 0.0]]).repeat(5, 1)
    loader._measured_racket_long_axis_w = torch.tensor(
        [[1.0, 0.0, 0.0]]
    ).repeat(5, 1)
    command.cfg.motion_teacher_racket_source = "measured_channel"
    motion.in_hold = torch.tensor([True, False])
    motion.speed_scale = torch.tensor([0.0, 0.5])

    position, normal, velocity = rewards._stage1_aligned_clip_site_target_at_steps(
        command, torch.tensor([2, 2])
    )

    # The fake reference anchor is at x=.4 on row 2, so its aligned measured x=.3 site is -.1
    # relative to the robot anchors at x=10/20.
    torch.testing.assert_close(position, torch.tensor([[9.9, 0.0, 1.0], [19.9, 0.0, 1.0]]))
    torch.testing.assert_close(normal, torch.tensor([[0.0, -1.0, 0.0]]).repeat(2, 1))
    torch.testing.assert_close(velocity[0], torch.zeros(3))
    # Measured source moves 0.5 m over the two-frame stencil: 12.5 m/s native, 6.25 m/s at 0.5x.
    torch.testing.assert_close(velocity[1], torch.tensor([6.25, 0.0, 0.0]))


def test_motion_racket_teacher_is_masked_inside_wide_strike_window(rewards):
    command = _fake_command()
    target_pos, target_normal, target_velocity = rewards._stage1_aligned_clip_site_target(
        command
    )
    command.racket_pos_w = target_pos.clone()
    command.racket_normal_w = target_normal.clone()
    command.racket_lin_vel_w = target_velocity.clone()
    # The first environment is in the strike window; the second is outside it.
    command.strike_window_wide = torch.tensor([True, False])
    env = types.SimpleNamespace(command_manager=_CommandManager(command))

    for reward in (
        rewards.motion_racket_position_tracking_cauchy,
        rewards.motion_racket_velocity_tracking_cauchy,
        rewards.motion_racket_normal_tracking_cauchy,
    ):
        value = reward(
            env,
            "racket_target",
            std=1.0,
            scale_in_strike_window=0.0,
        )
        torch.testing.assert_close(value, torch.tensor([0.0, 1.0]))


def test_fullmdp_motion_prior_cauchy_wrappers_have_exact_half_height(
    rewards, monkeypatch
):
    command = _fake_command()
    position = torch.zeros(2, 3)
    normal = torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1)
    velocity = torch.zeros(2, 3)
    long_axis = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)
    monkeypatch.setattr(
        rewards,
        "stage1_aligned_clip_racket_target_now",
        lambda _cmd: (position, normal, velocity, long_axis),
    )
    command.racket_pos_w = position + torch.tensor([0.70, 0.0, 0.0])
    command.racket_lin_vel_w = velocity + torch.tensor([4.0, 0.0, 0.0])
    command.racket_normal_w = -normal
    command.racket_long_axis_w = torch.tensor(
        [[math.cos(1.0), math.sin(1.0), 0.0]]
    ).repeat(2, 1)
    command.strike_window_wide = torch.zeros(2, dtype=torch.bool)
    env = types.SimpleNamespace(command_manager=_CommandManager(command))

    values = (
        rewards.motion_racket_position_tracking_cauchy(
            env, "racket_target", 0.70, 1.0
        ),
        rewards.motion_racket_velocity_tracking_cauchy(
            env, "racket_target", 4.0, 1.0
        ),
        rewards.motion_racket_normal_tracking_cauchy(
            env, "racket_target", math.pi, 1.0
        ),
        rewards.motion_racket_long_axis_tracking_cauchy(
            env, "racket_target", 1.0, 1.0
        ),
    )
    for value in values:
        torch.testing.assert_close(value, torch.full((2,), 0.5))
    torch.testing.assert_close(
        rewards._cauchy_tracking_kernel(torch.tensor([0.0, 1.0, 2.0]), 1.0),
        torch.tensor([1.0, 0.5, 0.2]),
    )


def test_fresh_same_token_reselects_ready_to_signed_measured_frame0(
    rewards, monkeypatch
):
    command = _fresh_measured_command()
    calls = []
    original = rewards._stage1_aligned_clip_racket_target_at_steps

    def counted(cmd, steps):
        calls.append(steps.clone())
        return original(cmd, steps)

    monkeypatch.setattr(
        rewards, "_stage1_aligned_clip_racket_target_at_steps", counted
    )
    first = rewards._stage1_aligned_clip_racket_target(command)
    measured = rewards._stage1_aligned_clip_measured_racket_target(command)
    torch.testing.assert_close(
        first[0], command._motion().body_pos_relative_w[:, 1]
    )
    torch.testing.assert_close(
        first[1], command.racket_normal_raw_w
    )
    torch.testing.assert_close(first[2], torch.zeros_like(first[2]))
    env = types.SimpleNamespace(command_manager=_CommandManager(command))
    torch.testing.assert_close(
        rewards.motion_racket_normal_tracking_cauchy(
            env, "racket_target", math.pi, 1.0
        ),
        torch.ones(2),
    )

    # ACCEPT changes only Motion's selector; the public step token remains 7.
    command._motion().ready_wait[0] = False
    second = rewards._stage1_aligned_clip_racket_target(command)
    for channel in range(4):
        torch.testing.assert_close(second[channel][0], measured[channel][0])
        torch.testing.assert_close(second[channel][1], first[channel][1])
    torch.testing.assert_close(second[1][0], torch.tensor([0.0, -1.0, 0.0]))
    torch.testing.assert_close(
        rewards.motion_racket_normal_tracking_cauchy(
            env, "racket_target", math.pi, 1.0
        ),
        torch.ones(2),
    )
    assert len(calls) == 1


def test_current_four_channel_target_uses_quarantined_motion_steps(rewards):
    command = _fresh_measured_command()
    motion = command._motion()
    motion.ready_wait.zero_()
    motion.safe_steps = torch.tensor([0, 2], dtype=torch.long)

    def forbidden_raw_steps():
        raise AssertionError("fresh reward bypassed Motion quarantine")

    motion._pose_reference_steps = forbidden_raw_steps
    actual = rewards._stage1_aligned_clip_racket_target(command)
    expected_raw = rewards._stage1_aligned_clip_racket_target_at_steps(
        command, motion.safe_steps
    )
    expected = rewards._stage1_select_split_ready_site_target(
        command, expected_raw
    )
    assert len(actual) == 4
    for actual_channel, expected_channel in zip(actual, expected):
        assert torch.isfinite(actual_channel).all()
        torch.testing.assert_close(actual_channel, expected_channel)


def test_public_now_target_reuses_the_shared_per_step_teacher_cache(rewards):
    command = _fake_command()
    first = rewards.stage1_aligned_clip_site_target_now(command)
    cache = command._stage1_clip_site_target_cache
    second = rewards.stage1_aligned_clip_site_target_now(command)

    assert command._stage1_clip_site_target_cache is cache
    for lhs, rhs in zip(first, second):
        torch.testing.assert_close(lhs, rhs)


def test_same_motion_generation_reuses_teacher_across_reward_terms(
    rewards, monkeypatch
):
    command = _fresh_measured_command()
    command._motion().ready_wait.zero_()
    command.racket_pos_w = torch.zeros(command.num_envs, 3)
    command.racket_lin_vel_w = torch.zeros(command.num_envs, 3)
    command.racket_long_axis_w = torch.tensor(
        [[1.0, 0.0, 0.0]]
    ).repeat(command.num_envs, 1)
    env = types.SimpleNamespace(command_manager=_CommandManager(command))
    calls = []
    original = rewards._stage1_aligned_clip_racket_target_at_steps

    def counted(cmd, steps):
        calls.append(steps.clone())
        return original(cmd, steps)

    monkeypatch.setattr(
        rewards, "_stage1_aligned_clip_racket_target_at_steps", counted
    )
    for reward in (
        rewards.motion_racket_position_tracking_cauchy,
        rewards.motion_racket_velocity_tracking_cauchy,
        rewards.motion_racket_normal_tracking_cauchy,
        rewards.motion_racket_long_axis_tracking_cauchy,
    ):
        value = reward(env, "racket_target", std=1.0)
        assert torch.isfinite(value).all()

    assert len(calls) == 1
    assert command._stage1_clip_site_target_cache[0] == (7, 0)


def test_motion_generation_invalidates_reward_cache_before_observation(
    rewards, monkeypatch
):
    command = _fresh_measured_command()
    motion = command._motion()
    motion.ready_wait.zero_()
    motion.safe_steps.zero_()
    motion._action_ball_continuous_motion_mutation_version = 23
    command.racket_pos_w = torch.zeros(command.num_envs, 3)
    env = types.SimpleNamespace(command_manager=_CommandManager(command))
    calls = []
    original = rewards._stage1_aligned_clip_racket_target_at_steps

    def counted(cmd, steps):
        calls.append(steps.clone())
        return original(cmd, steps)

    monkeypatch.setattr(
        rewards, "_stage1_aligned_clip_racket_target_at_steps", counted
    )
    # Isaac reward fills generation 23 at common_step 7.
    reward = rewards.motion_racket_position_tracking_cauchy(
        env, "racket_target", std=1.0
    )
    assert torch.isfinite(reward).all()
    before = command._stage1_clip_site_target_cache[1][0].clone()
    assert len(calls) == 1
    assert command._stage1_clip_site_target_cache[0] == (7, 23)

    # Motion advances in the same public step, then observation must see its
    # new selected rows rather than RewardManager's cached teacher.
    motion.safe_steps.fill_(2)
    motion._action_ball_continuous_motion_mutation_version += 1
    after = rewards.stage1_aligned_clip_racket_target_now(command)[0]

    assert len(calls) == 2
    torch.testing.assert_close(calls[0], torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(calls[1], torch.full((2,), 2, dtype=torch.long))
    assert command._stage1_clip_site_target_cache[0] == (7, 24)
    assert not torch.equal(before, after)


def test_fixed_coarse_kernels_cover_reviewed_cold_start_envelope(rewards):
    command = _fake_command()
    target_pos, target_normal, target_velocity = rewards._stage1_aligned_clip_site_target(command)
    command.racket_pos_w = target_pos + torch.tensor([0.70, 0.0, 0.0])
    command.racket_lin_vel_w = target_velocity + torch.tensor([4.0, 0.0, 0.0])
    command.racket_normal_w = -target_normal
    env = types.SimpleNamespace(command_manager=_CommandManager(command))

    expected = torch.full((2,), math.exp(-1.0))
    torch.testing.assert_close(
        rewards.stage1_clip_racket_position_coarse_tracking_exp(
            env, "racket_target", 0.70
        ),
        expected,
    )
    torch.testing.assert_close(
        rewards.stage1_clip_racket_velocity_coarse_tracking_exp(
            env, "racket_target", 4.0
        ),
        expected,
    )
    torch.testing.assert_close(
        rewards.stage1_clip_racket_normal_coarse_tracking_exp(
            env, "racket_target", math.pi
        ),
        expected,
    )

    # The adaptive fine kernels also begin inside a usable band at the same reviewed edge; the
    # fixed coarse kernels are the permanent backstop after fine contracts to its precision floor.
    fine = (
        rewards.stage1_clip_racket_position_tracking_exp(
            env, "racket_target", 0.50
        ),
        rewards.stage1_clip_racket_velocity_tracking_exp(
            env, "racket_target", 3.0
        ),
        rewards.stage1_clip_racket_normal_tracking_exp(
            env, "racket_target", 2.10
        ),
    )
    for value in fine:
        assert torch.all(value >= 0.10)


def test_weighted_dual_kernels_pull_every_reviewed_edge_toward_zero():
    error = torch.tensor(
        [0.70, 4.0, math.pi], dtype=torch.float64, requires_grad=True
    )
    coarse_sigma = torch.tensor([0.70, 4.0, math.pi], dtype=torch.float64)
    fine_sigma = torch.tensor([0.50, 3.0, 2.10], dtype=torch.float64)
    coarse_weight = torch.tensor([0.30, 0.15, 0.30], dtype=torch.float64)
    fine_weight = torch.tensor([0.90, 0.45, 0.90], dtype=torch.float64)
    reward = coarse_weight * torch.exp(-torch.square(error / coarse_sigma))
    reward = reward + fine_weight * torch.exp(-torch.square(error / fine_sigma))

    reward.sum().backward()

    # For positive scalar errors, reward ascent must reduce every error.  These are not merely
    # non-zero float64 crumbs: the smallest reviewed weighted slope (velocity) still exceeds .05.
    assert error.grad is not None
    assert torch.all(error.grad < -0.05)


def test_public_reference_hit_target_uses_configured_clip_phase_without_ball_target(rewards):
    command = _fake_command()
    command._motion()._pose_reference_steps = lambda: torch.tensor([1, 1])
    # 0.75 * (5 - 1) -> absolute row 3, independent of the current reference row above.
    site, normal, velocity = rewards.stage1_aligned_clip_site_target_at_reference_hit(command)

    torch.testing.assert_close(
        site,
        torch.tensor([[11.3, 0.2, 1.5], [21.3, 0.2, 1.5]]),
    )
    assert torch.isfinite(normal).all()
    assert torch.isfinite(velocity).all()
    assert not hasattr(command, "racket_target_pos_w")

    cached = rewards.stage1_aligned_clip_site_target_at_reference_hit(command)
    assert cached[0] is site


def test_public_reference_hit_target_uses_strike_clock_half_even_rounding(rewards):
    command = _fake_command()
    command.cfg.strike_phase = 0.625
    # The live strike clock uses torch.round(phase * (seg_len - 1)); 0.625 * 4 = 2.5 and
    # half-to-even therefore selects row 2, not row 3.
    site, _, _ = rewards.stage1_aligned_clip_site_target_at_reference_hit(command)

    torch.testing.assert_close(
        site,
        torch.tensor([[11.2, 0.2, 1.5], [21.2, 0.2, 1.5]]),
    )


def test_shape_contract_fails_loudly(rewards):
    with pytest.raises(ValueError, match="body quaternion 2"):
        rewards.stage1_clip_site_target_from_aligned_body_pose(
            torch.zeros(2, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            torch.zeros(2, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            torch.zeros(2, 3),
            torch.zeros(1, 4),
            mount_offset_body=torch.zeros(3),
            mount_quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            normal_axis=1,
            normal_sign=1.0,
            central_difference_span_s=0.04,
        )
