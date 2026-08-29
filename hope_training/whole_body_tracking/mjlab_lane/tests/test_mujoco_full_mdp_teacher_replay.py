"""Focused construction tests for the opt-in frozen-teacher diagnostic."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_full_mdp_teacher_replay as replay
import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env
import mujoco_gpu_ac_full_mdp_wait_rsl3 as runner


def test_teacher_replay_wait_bridge_and_affine_decoder_are_exact():
    hold = torch.tensor([[1.0, 2.0]])
    frame0 = torch.tensor([[5.0, 10.0]])
    previous = hold.clone()
    commands = []
    for valid, frozen in ((False, 4), (True, 3), (True, 2), (True, 1), (True, 0)):
        previous = replay.frozen_teacher_qdes(
            torch=torch,
            task_valid=torch.tensor([valid]),
            hold_qdes=hold,
            previous_qdes=previous,
            teacher_qdes=frame0,
            frozen_steps=torch.tensor([frozen]),
            bridge=wait_env.portable_question.step_diagnostic_split_ready_qdes_bridge,
        )
        commands.append(previous.clone())
    assert torch.equal(commands[0], hold)
    assert torch.equal(commands[-1], frame0)
    offset = torch.tensor([0.5, -1.0])
    scale = torch.tensor([0.5, 2.0])
    action = replay.decode_teacher_qdes_to_action(
        torch=torch, qdes=frame0, action_offset=offset, action_scale=scale
    )
    torch.testing.assert_close(offset + scale * action[0], frame0[0])


def test_teacher_replay_frozen_counter_and_hash_are_content_exact():
    frozen = replay.remaining_teacher_frozen_steps(
        torch=torch,
        common_step=12,
        reveal_tick=torch.tensor([10]),
        pre_swing_wait_s=torch.tensor([0.10]),
        step_dt=0.02,
    )
    assert frozen.tolist() == [3]
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    expected = hashlib.sha256(
        np.asarray([[1.0, 2.0]], dtype="<f4").tobytes()
    ).hexdigest()
    assert replay.tensor_f32_sha256(value) == expected


def test_contact_patch_is_absent_until_explicitly_enabled():
    bare = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="not enabled"):
        wait_env.FullMdpInitialWaitVecEnv.diagnostic_first_generic_contact_patch(
            bare
        )
    source = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts
    )
    assert 'getattr(self, "_diagnostic_contact_patch_consumer", None)' in source
    assert "patch_consumer is not None" in source


def test_teacher_replay_cli_is_n1_zero_ppo_and_reuses_live_owner():
    signature = inspect.signature(runner.main)
    assert signature.parameters["diagnostic_teacher_replay"].default is False
    assert signature.parameters["diagnostic_teacher_replay_steps"].default == 180
    source = inspect.getsource(runner._run_teacher_replay)
    assert "env.step(action)" in source
    assert "ppo_update_calls\": 0" in source
    assert "enable_diagnostic_first_generic_contact_patch" in source
    assert "_epoch_task_f32" in source
    assert "_full_a_launch_state_f32" in source
