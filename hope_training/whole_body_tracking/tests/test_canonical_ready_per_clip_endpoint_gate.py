"""Per-clip canonical-ready endpoint gate (coordinator ruling 2026-07-28).

人话:`_validate_canonical_ready_clips` 只按"每件剪辑自己的首末帧"判卷——
首末帧姿势 float32 逐位相等、端点六通道速度 literal zero、frame-0 根四元数为
单位四元数(yaw 投影即出生帧,真实 ready 站姿带 roll/pitch,不要求 yaw-only)、端点值有限。跨件"共享同一世界系 ready 姿势"条款已删:runtime 的
逐 slot ready 机制(_action_ball_ready_yaw/quat/z)本来就按每件剪辑自己的
frame-0 取值,瞄准旋转过的 canonical 剪辑在世界系里合法地互不相同。

Run on a CPU Torch environment:

    python -m pytest \
      hope_training/whole_body_tracking/tests/test_canonical_ready_per_clip_endpoint_gate.py -q
"""

from __future__ import annotations

import math
import types
from pathlib import Path

import pytest
import torch

from test_reward_flags_mdp import commands_mod as C  # noqa: E402


_JOINT_COUNT = 31
_BODY_COUNT = 3
_FRAMES = 9

_MDP = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
)
COMMANDS = (_MDP / "commands.py").read_text(encoding="utf-8")
HOPE_COMMANDS = (_MDP / "hope_commands.py").read_text(encoding="utf-8")


def _ready_quat(yaw_rad: float, pitch_rad: float = math.radians(-11.19)) -> torch.Tensor:
    """R_z(yaw) * R_y(pitch): the fivebind-like ready root (pitched, NOT yaw-only)."""

    half_yaw, half_pitch = 0.5 * yaw_rad, 0.5 * pitch_rad
    wz, zz = math.cos(half_yaw), math.sin(half_yaw)
    wy, yy = math.cos(half_pitch), math.sin(half_pitch)
    return torch.tensor(
        [wz * wy, -zz * yy, wz * yy, zz * wy],
        dtype=torch.float32,
    )


def _clip(yaw_deg: float, frames: int = _FRAMES) -> dict:
    """One canonical-ready clip whose world channels are aim-rotated by ``yaw_deg``.

    The ready pose (frame 0) equals the final pose exactly; middle frames move and
    carry non-zero velocities; endpoint velocities are literal zero.  The root pivot
    stays fixed while the other bodies rotate about it — the fivebind aim-rotation
    shape.
    """

    yaw = math.radians(yaw_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    root = torch.tensor([0.154225, -0.179536, 0.920683])
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.1],
            [0.0, 0.3, 0.2],
        ]
    )
    rot = torch.tensor(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ready_body_pos = (root[None, :] + offsets @ rot.T).to(torch.float32)
    ready_body_quat = torch.stack(
        [_ready_quat(yaw), _ready_quat(yaw + 0.3), _ready_quat(yaw - 0.2)]
    )
    ready_joint = (
        torch.arange(_JOINT_COUNT, dtype=torch.float32) / 100.0
    )

    joint_pos = ready_joint[None, :].repeat(frames, 1).clone()
    body_pos = ready_body_pos[None, :, :].repeat(frames, 1, 1).clone()
    body_quat = ready_body_quat[None, :, :].repeat(frames, 1, 1).clone()
    joint_vel = torch.zeros(frames, _JOINT_COUNT)
    body_lin = torch.zeros(frames, _BODY_COUNT, 3)
    body_ang = torch.zeros(frames, _BODY_COUNT, 3)
    for frame in range(1, frames - 1):
        joint_pos[frame] += 0.05 * math.sin(frame)
        body_pos[frame, 1:, 2] += 0.02 * frame
        joint_vel[frame] = 0.3
        body_lin[frame, :, 0] = 0.2
        body_ang[frame, :, 2] = 0.1
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
    }


def _command(clips: list[dict]):
    seg_lens = [clip["joint_pos"].shape[0] for clip in clips]
    starts = []
    total = 0
    for length in seg_lens:
        starts.append(total)
        total += length
    motion = types.SimpleNamespace(
        num_segments=len(clips),
        seg_start=torch.tensor(starts, dtype=torch.long),
        seg_len=torch.tensor(seg_lens, dtype=torch.long),
        joint_pos=torch.cat([clip["joint_pos"] for clip in clips], dim=0),
        joint_vel=torch.cat([clip["joint_vel"] for clip in clips], dim=0),
        _body_pos_w=torch.cat([clip["body_pos_w"] for clip in clips], dim=0),
        _body_quat_w=torch.cat([clip["body_quat_w"] for clip in clips], dim=0),
        _body_lin_vel_w=torch.cat(
            [clip["body_lin_vel_w"] for clip in clips], dim=0
        ),
        _body_ang_vel_w=torch.cat(
            [clip["body_ang_vel_w"] for clip in clips], dim=0
        ),
    )
    command = C.MotionCommand.__new__(C.MotionCommand)
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            default_joint_pos=torch.zeros(1, _JOINT_COUNT)
        )
    )
    command.body_indexes = torch.tensor([0, 1, 2], dtype=torch.long)
    command.motion = motion
    command.action_ball_diagnostic_split_ready_teacher = False
    return command


_FIVEBIND_LIKE_YAWS = (-113.660813, 84.633843, 34.809414, 42.244359)


def test_four_clips_differing_only_in_world_yaw_pass():
    clips = [_clip(yaw) for yaw in _FIVEBIND_LIKE_YAWS]
    command = _command(clips)
    # The set genuinely exercises the removed cross-clip clause: world-frame
    # frame-0 poses differ between clips ...
    assert not torch.equal(
        clips[0]["body_pos_w"][0], clips[1]["body_pos_w"][0]
    )
    assert not torch.equal(
        clips[0]["body_quat_w"][0], clips[1]["body_quat_w"][0]
    )
    # ... and the per-clip gate accepts it.
    C.MotionCommand._validate_canonical_ready_clips(command)


def test_clip_end_pose_differing_from_its_own_start_fails():
    clips = [_clip(yaw) for yaw in _FIVEBIND_LIKE_YAWS]
    clips[2]["joint_pos"][-1, 5] += 1.0e-3
    with pytest.raises(ValueError, match="start and end on one"):
        C.MotionCommand._validate_canonical_ready_clips(_command(clips))


def test_clip_end_world_pose_differing_from_its_own_start_fails():
    clips = [_clip(yaw) for yaw in _FIVEBIND_LIKE_YAWS]
    clips[1]["body_pos_w"][-1, 2, 0] += 1.0e-3
    with pytest.raises(ValueError, match="channel=body_pos_w"):
        C.MotionCommand._validate_canonical_ready_clips(_command(clips))


def test_nonzero_endpoint_velocity_fails():
    clips = [_clip(yaw) for yaw in _FIVEBIND_LIKE_YAWS]
    clips[3]["body_lin_vel_w"][-1, 0, 1] = 1.0e-6
    with pytest.raises(
        ValueError, match="literal zero endpoint velocities"
    ):
        C.MotionCommand._validate_canonical_ready_clips(_command(clips))


def test_pitched_ready_root_quat_passes():
    # Real curated data: the shared ready root has ~-11.2 deg pitch. The gate
    # must accept it (the action-ball birth frame is the yaw projection).
    clips = [_clip(yaw) for yaw in _FIVEBIND_LIKE_YAWS]
    quat0 = clips[0]["body_quat_w"][0, 0]
    assert float(torch.min(torch.abs(quat0[1:3]))) > 1.0e-3  # genuinely not yaw-only
    C.MotionCommand._validate_canonical_ready_clips(_command(clips))


def test_non_unit_frame0_root_quat_fails():
    clips = [_clip(yaw) for yaw in _FIVEBIND_LIKE_YAWS]
    for frame in (0, _FRAMES - 1):
        clips[0]["body_quat_w"][frame, 0] *= 1.001
    with pytest.raises(ValueError, match="unit frame-0 root quaternion"):
        C.MotionCommand._validate_canonical_ready_clips(_command(clips))


def test_negated_hemisphere_yaw_only_root_quat_passes():
    clips = [_clip(yaw) for yaw in _FIVEBIND_LIKE_YAWS]
    for frame in (0, _FRAMES - 1):
        clips[1]["body_quat_w"][frame, 0] = -clips[1]["body_quat_w"][
            frame, 0
        ]
    C.MotionCommand._validate_canonical_ready_clips(_command(clips))


def test_single_clip_behavior_unchanged():
    C.MotionCommand._validate_canonical_ready_clips(
        _command([_clip(-113.660813)])
    )
    bad = _clip(-113.660813)
    bad["joint_pos"][-1, 0] += 1.0e-3
    with pytest.raises(ValueError, match="start and end on one"):
        C.MotionCommand._validate_canonical_ready_clips(_command([bad]))
    fast = _clip(-113.660813)
    fast["joint_vel"][0, 0] = 1.0e-6
    with pytest.raises(
        ValueError, match="literal zero endpoint velocities"
    ):
        C.MotionCommand._validate_canonical_ready_clips(_command([fast]))


def test_non_finite_boundary_fails():
    clips = [_clip(yaw) for yaw in _FIVEBIND_LIKE_YAWS]
    clips[0]["body_pos_w"][0, 0, 2] = float("nan")
    clips[0]["body_pos_w"][-1, 0, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        C.MotionCommand._validate_canonical_ready_clips(_command(clips))


def test_scoped_measured_n1_accepts_nonloop_end_but_not_moving_start():
    clip = _clip(0.0)
    clip["joint_pos"][-1, 7] += 0.4
    clip["body_pos_w"][-1, 1, 2] += 0.08
    clip["joint_vel"][-1, 7] = 2.4
    command = _command([clip])
    command.action_ball_diagnostic_split_ready_teacher = True
    C.MotionCommand._validate_canonical_ready_clips(command)

    clip["joint_vel"][0, 7] = 1.0e-5
    with pytest.raises(ValueError, match="teacher-start velocities"):
        moving_start = _command([clip])
        moving_start.action_ball_diagnostic_split_ready_teacher = True
        C.MotionCommand._validate_canonical_ready_clips(moving_start)


def test_formal_default_still_rejects_same_nonloop_measured_shape():
    clip = _clip(0.0)
    clip["joint_pos"][-1, 7] += 0.4
    clip["joint_vel"][-1, 7] = 2.4
    with pytest.raises(ValueError, match="start and end on one"):
        C.MotionCommand._validate_canonical_ready_clips(_command([clip]))


def test_split_ready_external_install_accepts_only_teacher_start():
    command = _command([_clip(0.0)])
    command.canonical_ready_mode = True
    command.action_ball_diagnostic_split_ready_teacher = True
    command.clip_id = torch.tensor([0], dtype=torch.long)
    command.time_steps = torch.tensor([0], dtype=torch.long)
    env_ids = torch.tensor([0], dtype=torch.long)

    C.MotionCommand._require_canonical_ready_boundary(
        command, env_ids, "external install"
    )
    command.time_steps[0] = _FRAMES - 1
    with pytest.raises(ValueError, match="legal canonical ready boundary"):
        C.MotionCommand._require_canonical_ready_boundary(
            command, env_ids, "external install"
        )


def test_formal_ready_to_ready_external_install_still_accepts_clip_end():
    command = _command([_clip(0.0)])
    command.canonical_ready_mode = True
    command.action_ball_diagnostic_split_ready_teacher = False
    command.clip_id = torch.tensor([0], dtype=torch.long)
    command.time_steps = torch.tensor([_FRAMES - 1], dtype=torch.long)

    C.MotionCommand._require_canonical_ready_boundary(
        command, torch.tensor([0], dtype=torch.long), "external install"
    )


def test_cross_clip_clause_stays_removed_in_source():
    assert "requires all clip starts/ends to share one" not in COMMANDS
    # source text splits the message across literals; pin each piece
    assert "requires each clip to start and end on one " in COMMANDS
    assert "exact runtime-float32 ready pose" in COMMANDS
    assert "unit frame-0 root quaternion" in COMMANDS
    assert "must be yaw-only" not in COMMANDS
    # hope_commands stores the yaw PROJECTION of the ready root; the hard
    # yaw-only root demand (which rejected every curated clip) stays removed.
    assert "YAW PROJECTION of the canonical ready" in HOPE_COMMANDS
    assert "must be yaw-only so the" not in HOPE_COMMANDS
