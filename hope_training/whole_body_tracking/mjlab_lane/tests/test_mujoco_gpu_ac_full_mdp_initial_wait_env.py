"""Direct tests for the current MuJoCo 203/219 initial-WAIT slice."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env


P = wait_env.observation_contract


# Independent literal offsets.  These are deliberately not derived from the
# production layout: a reordered producer must fail even when its total width
# remains 203/219.
V2_ACTOR_SLICES = {
    "projected_gravity_b": (0, 3),
    "base_ang_vel_b": (3, 6),
    "base_position_table": (6, 9),
    "base_heading_table_xy": (9, 11),
    "base_com_lin_vel_heading": (11, 14),
    "joint_pos_rel": (14, 45),
    "joint_vel": (45, 76),
    "last_action": (76, 107),
    "teacher_joint_pos_rel": (107, 138),
    "teacher_joint_vel": (138, 169),
    "motion_anchor_pos_b": (169, 172),
    "motion_anchor_ori_b6": (172, 178),
    "motion_phase_one_hot": (178, 183),
    "racket_target_pos_error_heading": (183, 186),
    "racket_target_vel_error_heading": (186, 189),
    "racket_target_normal_error_heading": (189, 192),
    "base_goal_error_heading_xy": (192, 194),
    "time_to_contact_s": (194, 195),
    "time_to_teacher_start_s": (195, 196),
    "time_to_next_opportunity_s": (196, 197),
    "epoch_learning_phase_one_hot": (197, 202),
    "task_valid": (202, 203),
}
V2_CRITIC_SLICES = {
    "episode_time_remaining_s": (203, 204),
    "live_ball_center_rel_root_heading": (204, 207),
    "live_ball_lin_vel_heading": (207, 210),
    "live_ball_ang_vel_heading": (210, 213),
    "selected_rubber_contact_latched": (213, 214),
    "net_crossed_latched": (214, 215),
    "net_clear_latched": (215, 216),
    "foot_supported_lr": (216, 218),
    "cadence_ready_dwell_fraction": (218, 219),
}


def _offset(layout, name):
    offset = 0
    for row_name, width in layout:
        if row_name == name:
            return offset, width
        offset += width
    raise AssertionError(name)


def test_shared_layout_has_the_live_full_mdp_widths_and_one_order():
    assert P.ACTOR_WIDTH_V1 == 229
    assert P.CRITIC_WIDTH_V1 == 399
    assert P.TASK_F32_WIDTH == 45
    assert P.OWNER_FACT_F32_WIDTH == 32
    assert P.REWARD_CONSUMER_COUNT == 14

    rows = {
        name: torch.full((1, width), float(index))
        for index, (name, width) in enumerate(P.ACTOR_LAYOUT_V1, start=1)
    }
    packed = P.concatenate_layout_rows(P.ACTOR_LAYOUT_V1, rows)
    assert packed.shape == (1, 229)
    offset = 0
    for index, (_name, width) in enumerate(P.ACTOR_LAYOUT_V1, start=1):
        assert torch.equal(
            packed[:, offset : offset + width],
            torch.full((1, width), float(index)),
        )
        offset += width


def test_semantic_v2_literal_offsets_are_exact_and_not_padding():
    assert P.ACTOR_WIDTH_V2 == 203
    assert P.CRITIC_WIDTH_V2 == 219
    assert tuple(
        (name, end - start) for name, (start, end) in V2_ACTOR_SLICES.items()
    ) == P.ACTOR_LAYOUT_V2
    assert tuple(
        (name, end - start)
        for name, (start, end) in V2_CRITIC_SLICES.items()
    ) == P.CRITIC_EXTENSION_LAYOUT_V2
    assert tuple(V2_ACTOR_SLICES.values())[-1][1] == 203
    assert tuple(V2_CRITIC_SLICES.values())[-1][1] == 219


def test_full_a_semantic_v2_uses_native_com_frames_clocks_and_current_shot():
    n, dtype = 2, torch.float64
    origins = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=dtype)
    root_scene = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
    root_world = root_scene + origins
    half = 2.0**-0.5
    root_quat = torch.tensor(
        [[half, 0.0, 0.0, half], [1.0, 0.0, 0.0, 0.0]], dtype=dtype
    )
    ball_qpos = torch.zeros((n, 7), dtype=dtype)
    ball_qpos[:, :3] = root_world + torch.tensor([[1.0, 0.0, 0.0], [9.0, 9.0, 9.0]], dtype=dtype)
    ball_qpos[0, 3:7] = root_quat[0]
    ball_qpos[1, 3] = 1.0
    ball_qvel = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [8.0, 8.0, 8.0, 8.0, 8.0, 8.0]],
        dtype=dtype,
    )
    live_anchor_pos = torch.tensor(
        [[[10.0, 20.0, 30.0]], [[40.0, 50.0, 60.0]]], dtype=dtype
    )
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    live_anchor_quat = identity.reshape(1, 1, 4).expand(n, 1, 4).clone()
    teacher_anchor_pos = live_anchor_pos[:, 0] + torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]], dtype=dtype
    )
    teacher_anchor_quat = identity.reshape(1, 4).expand(n, 4).clone()

    joint = torch.arange(n * 31, dtype=dtype).reshape(n, 31) / 10.0
    joint_vel = joint + 0.25
    offset = torch.linspace(-0.3, 0.3, 31, dtype=dtype)
    teacher_joint = joint + 1.0
    teacher_joint_vel = joint_vel + 2.0
    task = torch.zeros((n, 45), dtype=dtype)
    task[0, 5:8] = torch.tensor([2.0, 3.0, 4.0], dtype=dtype)
    task[0, 8:11] = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    task[0, 11:14] = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    task[0, 24:26] = torch.tensor([3.0, 5.0], dtype=dtype)
    task[1, 5:14] = 99.0
    task[1, 24:26] = 99.0
    racket_position = torch.tensor([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0]], dtype=dtype)
    racket_velocity = torch.tensor([[1.0, 0.0, 0.0], [9.0, 9.0, 9.0]], dtype=dtype)
    racket_raw_normal = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)
    com_velocity = torch.tensor(
        [[[1.0, 0.0, 0.0]], [[4.0, 5.0, 6.0]]], dtype=dtype
    )
    cvel = torch.zeros((n, 1, 6), dtype=dtype)
    cvel[0, 0, :3] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    cvel[0, 0, 3:] = torch.tensor([1.0, -1.0, 0.0], dtype=dtype)
    cvel[1, 0, 3:] = com_velocity[1, 0]
    xipos = torch.zeros((n, 1, 3), dtype=dtype)
    xipos[0, 0, 0] = 1.0
    subtree_com = torch.zeros((n, 1, 3), dtype=dtype)

    env = types.SimpleNamespace(
        _torch=torch,
        qpos_init=torch.zeros(1, dtype=dtype),
        sim=types.SimpleNamespace(
            data=types.SimpleNamespace(
                qpos=ball_qpos,
                qvel=ball_qvel,
                xpos=live_anchor_pos,
                xquat=live_anchor_quat,
                cvel=cvel,
                xipos=xipos,
                subtree_com=subtree_com,
            )
        ),
        env=types.SimpleNamespace(scene=types.SimpleNamespace(env_origins=origins)),
        num_envs=n,
        device=torch.device("cpu"),
        b_q=0,
        b_v=0,
        _fullmdp_body_ids=torch.tensor([0]),
        _fullmdp_anchor_index=0,
        _fullmdp_anchor_body_id=0,
        _fullmdp_pelvis_body_id=0,
        _fullmdp_pelvis_root_id=0,
        _teacher_body_pos=teacher_anchor_pos[:, None, :],
        _teacher_body_quat=teacher_anchor_quat[:, None, :],
        _aligned_teacher_body_pos=torch.full((n, 1, 3), -777.0, dtype=dtype),
        _aligned_teacher_body_quat=torch.full((n, 1, 4), -777.0, dtype=dtype),
        _full_a_table_surface_center_scene=torch.tensor([0.5, 1.5, 0.75], dtype=dtype),
        # Row one retains the Epoch current shot after retirement, but its
        # READY_HOLD Motion phase must hide every actor task field.
        _epoch_task_valid=torch.tensor([True, True]),
        _epoch_task_f32=task,
        _epoch_clock_ticks=torch.tensor([[10, 25, 0, 0, 0], [-1, -1, -1, -1, -1]]),
        _full_a_pre_swing_wait_s=torch.tensor([0.5, 7.0], dtype=dtype),
        common_step_counter=20,
        step_dt=0.02,
        _full_a_next_reveal_tick=torch.tensor([30, 40]),
        episode_length_buf=torch.tensor([10, 20]),
        max_episode_length=100,
        _epoch_phase=torch.tensor(
            [wait_env.FULL_A_PHASE_LAUNCH_SETTLED, wait_env.FULL_A_PHASE_RETIRED]
        ),
        _epoch_launch_succeeded=torch.tensor([True, True]),
        _full_a_motion_phase_code=torch.tensor([1, 4]),
        action_offset=offset,
        actions=joint + 3.0,
        _full_a_teacher_joint_pos=teacher_joint,
        _full_a_teacher_joint_vel=teacher_joint_vel,
        _full_a_mount_normal_sign=torch.tensor([-1, 1], dtype=torch.int8),
        _full_a_selected_racket_contact=torch.tensor([True, True]),
        _full_a_net_crossed=torch.tensor([True, True]),
        _full_a_net_clear=torch.tensor([True, True]),
        _full_a_foot_supported_lr=torch.tensor([[True, False], [False, True]]),
        _full_a_cadence_ready_streak=torch.tensor([1, 7]),
        _full_a_actor_scale_v2=torch.tensor(P.ACTOR_SCALE_FLAT_V2, dtype=dtype),
        _full_a_critic_extension_scale_v2=torch.tensor(
            P.CRITIC_EXTENSION_SCALE_FLAT_V2, dtype=dtype
        ),
        _qpos_act=lambda: joint,
        _qvel_act=lambda: joint_vel,
        _body_com_velocities_from_cvel=(
            wait_env.FullMdpInitialWaitVecEnv._body_com_velocities_from_cvel
        ),
        _full_a_racket_kinematics=lambda: (
            racket_position,
            racket_velocity,
            racket_raw_normal,
        ),
    )
    st = {
        "base_pos": root_world,
        "base_quat": root_quat,
        # Deliberately disagree with the inertial-COM producer.
        "base_lin_w": torch.full((n, 3), 1234.0, dtype=dtype),
        "base_ang_b": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=dtype),
        "proj_g": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=dtype),
    }
    policy = wait_env.FullMdpInitialWaitVecEnv._full_a_semantic_observation_v2(
        env, st
    )
    critic = env._critic_obs_buf
    assert policy.shape == (2, 203)
    assert critic.shape == (2, 219)

    def actor(name):
        start, end = V2_ACTOR_SLICES[name]
        return policy[:, start:end]

    def privileged(name):
        start, end = V2_CRITIC_SLICES[name]
        return critic[:, start:end]

    assert torch.allclose(actor("base_position_table")[0], torch.tensor([0.5, 0.5, 2.25], dtype=dtype))
    assert torch.allclose(actor("base_heading_table_xy")[0], torch.tensor([0.0, 1.0], dtype=dtype), atol=1e-15)
    assert torch.allclose(actor("base_ang_vel_b")[0], torch.tensor([0.025, 0.05, 0.075], dtype=dtype), atol=1e-15)
    assert torch.allclose(actor("base_com_lin_vel_heading")[0], torch.tensor([0.0, -0.5, 0.0], dtype=dtype), atol=1e-15)
    assert not torch.any(actor("base_com_lin_vel_heading").eq(1234.0))
    torch.testing.assert_close(actor("joint_vel"), joint_vel * 0.05)
    torch.testing.assert_close(actor("teacher_joint_vel"), teacher_joint_vel * 0.05)
    assert torch.allclose(actor("motion_anchor_pos_b")[0], torch.tensor([10.0 / 3.0, 20.0 / 3.0, 10.0], dtype=dtype))
    assert not torch.any(actor("motion_anchor_pos_b").eq(-777.0))
    assert torch.allclose(actor("racket_target_pos_error_heading")[0], torch.tensor([10.0, -5.0, 15.0], dtype=dtype), atol=1e-15)
    assert torch.allclose(actor("racket_target_vel_error_heading")[0], torch.tensor([2.0, 0.0, 3.0], dtype=dtype), atol=1e-15)
    # mount sign=-1 selects the contacted rubber but does not reverse the raw
    # A/+Y normal shared by the FullMDP question, observation and R03 reward.
    assert torch.allclose(actor("racket_target_normal_error_heading")[0], torch.tensor([2.0, 2.0, 0.0], dtype=dtype), atol=1e-15)
    assert torch.allclose(actor("base_goal_error_heading_xy")[0], torch.tensor([15.0, -10.0], dtype=dtype), atol=1e-15)
    torch.testing.assert_close(
        actor("time_to_contact_s")[:, 0],
        torch.tensor([0.1 / 2.42, 0.0], dtype=dtype),
    )
    assert torch.equal(actor("time_to_teacher_start_s")[:, 0], torch.tensor([0.3, 0.0], dtype=dtype))
    torch.testing.assert_close(
        actor("time_to_next_opportunity_s")[:, 0],
        torch.tensor([0.4 / 5.86, 0.4 / 5.86], dtype=dtype),
    )
    assert torch.count_nonzero(policy[1, 183:196]) == 0
    assert torch.equal(actor("epoch_learning_phase_one_hot"), torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=dtype))
    assert torch.equal(actor("task_valid")[:, 0], torch.tensor([1.0, 0.0], dtype=dtype))

    assert torch.allclose(privileged("live_ball_center_rel_root_heading")[0], torch.tensor([0.0, -1.0, 0.0], dtype=dtype), atol=1e-15)
    assert torch.allclose(privileged("live_ball_lin_vel_heading")[0], torch.tensor([0.1, 0.0, 0.0], dtype=dtype), atol=1e-15)
    # Ball spin is body-local in qvel: ball yaw maps +X to world +Y, then base
    # inverse yaw maps it back to heading +X before static 1/60 scaling.
    assert torch.allclose(privileged("live_ball_ang_vel_heading")[0], torch.tensor([1.0 / 60.0, 0.0, 0.0], dtype=dtype), atol=1e-15)
    assert torch.count_nonzero(critic[1, 204:213]) == 0
    # RETIRED retains the Epoch payload for identity/history, but no live
    # flight remains.  Contact/net critic state must therefore be neutral.
    assert torch.count_nonzero(critic[1, 213:216]) == 0
    assert torch.equal(privileged("foot_supported_lr"), torch.tensor([[1, 0], [0, 1]], dtype=dtype))
    assert torch.equal(privileged("cadence_ready_dwell_fraction")[:, 0], torch.tensor([0.5, 1.0], dtype=dtype))
    torch.testing.assert_close(
        privileged("episode_time_remaining_s")[:, 0],
        torch.tensor([1.8 / 30.0, 1.6 / 30.0], dtype=dtype),
    )
    assert torch.equal(critic[:, :203], policy)


def test_initial_wait_requires_positive_world_count_and_deterministic_reset():
    with pytest.raises(ValueError, match="requires positive nworld"):
        wait_env.FullMdpInitialWaitVecEnv(
            sim_cfg=wait_env.SimCfg(nworld=0),
            task_cfg=wait_env.TaskCfg(),
            device="cpu",
        )
    with pytest.raises(ValueError, match="requires deterministic reset"):
        wait_env.FullMdpInitialWaitVecEnv(
            sim_cfg=wait_env.SimCfg(nworld=1),
            task_cfg=wait_env.TaskCfg(reset_joint_noise_rad=0.01),
            device="cpu",
        )


def test_fullmdp_forwards_ready_pose_bytes_to_the_real_base(monkeypatch):
    class _StopAtBase(Exception):
        pass

    captured = {}

    def _base_init(_self, *_args, **kwargs):
        captured.update(kwargs)
        raise _StopAtBase

    monkeypatch.setattr(wait_env.A3ReadyBallVecEnv, "__init__", _base_init)
    with pytest.raises(_StopAtBase):
        wait_env.FullMdpInitialWaitVecEnv(
            sim_cfg=wait_env.SimCfg(nworld=1),
            task_cfg=wait_env.TaskCfg(
                reset_joint_noise_rad=0.0,
                reset_joint_vel_noise=0.0,
                reset_root_xy_noise_m=0.0,
                reset_root_yaw_noise_rad=0.0,
            ),
            device="cpu",
            ready_pose_payload=b"frozen-pose",
            ready_pose_source="/frozen/ready_pose.json",
        )
    assert captured["ready_pose_path"] is None
    assert captured["ready_pose_payload"] == b"frozen-pose"
    assert captured["ready_pose_source"] == "/frozen/ready_pose.json"


def test_real_base_parses_ready_pose_bytes_before_build_and_never_fallbacks(
    monkeypatch, tmp_path
):
    train = sys.modules[wait_env.A3ReadyBallVecEnv.__module__]
    monkeypatch.setitem(sys.modules, "mujoco", types.ModuleType("mujoco"))
    trace = []

    def _parse(payload, source):
        trace.append(("parse", payload, source))
        return {"source": source}

    class _StopAtBuild(Exception):
        pass

    def _build(*_args, **_kwargs):
        trace.append(("build",))
        raise _StopAtBuild

    monkeypatch.setattr(train.court, "load_ready_pose_bytes", _parse)
    monkeypatch.setattr(train.court, "load_ready_pose", lambda *_: pytest.fail("fallback"))
    monkeypatch.setattr(train.court, "build_court_env", _build)
    with pytest.raises(_StopAtBuild):
        wait_env.A3ReadyBallVecEnv(
            wait_env.SimCfg(nworld=1),
            wait_env.TaskCfg(),
            "cpu",
            ready_pose_payload=b"frozen-pose",
            ready_pose_source="/frozen/ready_pose.json",
        )
    assert trace == [
        ("parse", b"frozen-pose", "/frozen/ready_pose.json"),
        ("build",),
    ]

    trace.clear()
    with pytest.raises(FileNotFoundError, match="explicit ready pose"):
        wait_env.A3ReadyBallVecEnv(
            wait_env.SimCfg(nworld=1),
            wait_env.TaskCfg(),
            "cpu",
            ready_pose_path=tmp_path / "missing.json",
        )
    assert trace == []

    with pytest.raises(ValueError, match="exclusive source pair"):
        wait_env.A3ReadyBallVecEnv(
            wait_env.SimCfg(nworld=1),
            wait_env.TaskCfg(),
            "cpu",
            ready_pose_source="/partial/source.json",
        )


@pytest.mark.skipif(
    os.environ.get("ACTIONBALL_RUN_MUJOCO_GPU_DIRECT") != "1",
    reason="requires the exact MuJoCo-Warp GPU environment and A3 assets",
)
def test_real_n1_reset_forward_returns_stock_rsl_tensordict():
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("mjlab")
    tensordict = pytest.importorskip("tensordict")

    task = wait_env.TaskCfg(
        action_scale_mode="vendor",
        reset_joint_noise_rad=0.0,
        reset_joint_vel_noise=0.0,
        reset_root_xy_noise_m=0.0,
        reset_root_yaw_noise_rad=0.0,
    )
    sim = wait_env.SimCfg(nworld=1)
    ready_path = os.environ.get("ACTIONBALL_READY_POSE")
    env = wait_env.FullMdpInitialWaitVecEnv(
        sim_cfg=sim,
        task_cfg=task,
        device=os.environ.get("ACTIONBALL_MUJOCO_DEVICE", "cuda:0"),
        ready_pose_path=Path(ready_path) if ready_path else None,
    )
    try:
        observations, extras = env.reset()
        assert env.physics_dt == 0.001
        assert env.decimation == 20
        assert env.step_dt == 0.02
        # MuJoCo-Warp has no noslip pass; this is the tracked, registered
        # backend deviation rather than an attach warning being ignored.
        assert int(env.mj_model.opt.noslip_iterations) == 0
        assert isinstance(observations, tensordict.TensorDictBase)
        assert extras == {}
        policy = observations["policy"]
        critic = observations["critic"]
        assert policy.shape == (1, 229)
        assert critic.shape == (1, 399)
        assert torch.isfinite(policy).all()
        assert torch.isfinite(critic).all()

        live = env._state()
        offset, width = _offset(P.ACTOR_LAYOUT_V1, "projected_gravity_b")
        assert torch.equal(policy[:, offset : offset + width], live["proj_g"])
        offset, width = _offset(P.ACTOR_LAYOUT_V1, "joint_pos_rel")
        expected_q = env._qpos_act() - env.q_ready.unsqueeze(0)
        assert torch.equal(policy[:, offset : offset + width], expected_q)
        phase_offset, phase_width = _offset(
            P.ACTOR_LAYOUT_V1, "motion_phase_one_hot"
        )
        expected_phase = torch.zeros_like(
            policy[:, phase_offset : phase_offset + phase_width]
        )
        expected_phase[:, wait_env.READY_HOLD_PHASE_INDEX] = 1.0
        assert torch.equal(
            policy[:, phase_offset : phase_offset + phase_width], expected_phase
        )
        epoch_offset, epoch_width = _offset(
            P.ACTOR_LAYOUT_V1, "epoch_phase_one_hot"
        )
        expected_epoch_phase = torch.zeros_like(
            policy[:, epoch_offset : epoch_offset + epoch_width]
        )
        expected_epoch_phase[:, P.EPOCH_IDLE_PHASE_INDEX] = 1.0
        assert torch.equal(
            policy[:, epoch_offset : epoch_offset + epoch_width],
            expected_epoch_phase,
        )
        for name in (
            "epoch_task_f32",
            "epoch_clock_remaining_s",
            "epoch_task_valid",
            "epoch_selected",
            "epoch_launch_succeeded",
        ):
            offset, width = _offset(P.ACTOR_LAYOUT_V1, name)
            assert torch.count_nonzero(policy[:, offset : offset + width]) == 0
        assert torch.count_nonzero(critic[:, P.ACTOR_WIDTH_V1 :]) == 0

        expected_park = torch.tensor(
            wait_env.WAIT_BALL_PARK_HOPE,
            dtype=env.qpos_init.dtype,
            device=env.device,
        ) + env.hope_to_scene
        assert torch.equal(
            env.sim.data.qpos[0, env.b_q : env.b_q + 3], expected_park
        )
        assert torch.count_nonzero(
            env.sim.data.qvel[0, env.b_v : env.b_v + 6]
        ) == 0
        contact = env._con_geom[:]
        valid = env._con_idx < env._nacon[0]
        ball_contact = valid & (
            (contact[:, 0] == env._ball_gid)
            | (contact[:, 1] == env._ball_gid)
        )
        assert not bool(ball_contact.any())
    finally:
        env.close()
