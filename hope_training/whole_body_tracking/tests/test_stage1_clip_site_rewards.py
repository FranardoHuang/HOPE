"""Host tests for the no-ball Stage-1 natural-clip official-racket-site reward."""

from __future__ import annotations

import importlib.util
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
    old = sys.modules.get(canonical)
    stub = types.ModuleType(canonical)
    stub.RacketTargetCommand = object
    stub.face_tracking_pair = lambda command: (
        command.racket_normal_w,
        command.racket_target_normal_w,
    )
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
        motion_anchor_body_index=0,
        robot_anchor_body_index=2,
        robot_anchor_pos_w=torch.tensor([[10.0, 0.0, 1.0], [20.0, 0.0, 1.0]]),
        robot_anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(env_count, 1),
        speed_scale=torch.ones(env_count),
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
        ),
        strike_window=torch.tensor([True, True]),
        strike_window_pos=torch.tensor([True, False]),
        strike_window_wide=torch.tensor([True, True]),
        metrics={},
    )
    return command


def test_reward_wrappers_use_phase_continuous_teacher_and_split_windows(rewards):
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

    torch.testing.assert_close(pos, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(normal, torch.ones(2))
    torch.testing.assert_close(velocity, torch.ones(2))
    assert torch.isfinite(target_pos).all()
    assert torch.isfinite(target_velocity).all()


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
