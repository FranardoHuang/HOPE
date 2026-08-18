"""Direct tests for the first real MuJoCo 229/399 initial-WAIT slice."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env


P = wait_env.observation_contract


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


def test_initial_wait_step_fails_instead_of_claiming_missing_producers():
    with pytest.raises(ValueError, match="requires nworld=1"):
        wait_env.FullMdpInitialWaitVecEnv(
            sim_cfg=wait_env.SimCfg(nworld=2),
            task_cfg=wait_env.TaskCfg(),
            device="cpu",
        )
    with pytest.raises(RuntimeError, match="no FullMDP step/reward/termination"):
        wait_env.FullMdpInitialWaitVecEnv.step(None, torch.zeros((1, 31)))


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
