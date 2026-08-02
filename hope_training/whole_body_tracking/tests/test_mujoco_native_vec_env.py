"""Contracts for the no-reward native MuJoCo diagnostic VecEnv adapter."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import n1_ball_core as n1  # noqa: E402
from mujoco_native import single_env  # noqa: E402
from mujoco_native import vec_env  # noqa: E402


CONTRACT = (
    REPO_ROOT
    / "configs/a3_vendor_runtime_authority_20260802_r8"
    / "bh_loop_c.shared_ready.training_contract.json"
)


def _groups(offset: float = 0.0):
    return {
        name: np.arange(width, dtype=np.float64) + offset
        for name, width in vec_env.OBSERVATION_LAYOUT
    }


def test_flattened_layout_is_explicit_76d_and_fail_closed():
    flat = vec_env.flatten_observation_groups(_groups())
    assert vec_env.OBSERVATION_WIDTH == 76
    assert flat.shape == (76,)
    cursor = 0
    for name, width in vec_env.OBSERVATION_LAYOUT:
        np.testing.assert_array_equal(flat[cursor : cursor + width], _groups()[name])
        cursor += width

    missing = _groups()
    missing.pop("validity")
    with pytest.raises(vec_env.VecEnvContractError, match="groups differ"):
        vec_env.flatten_observation_groups(missing)
    wrong = _groups()
    wrong["landing_aim_xy_w_m"] = np.zeros(3)
    with pytest.raises(vec_env.VecEnvContractError, match="must be 2"):
        vec_env.flatten_observation_groups(wrong)


def test_reward_blocker_prohibits_fake_ppo_checkpoint_and_resume():
    receipt = vec_env.reward_blocker_receipt()
    assert receipt["status"] == "PPO_BLOCKED_MISSING_REAL_REWARD_CONTRACT"
    assert receipt["reward_available"] is False
    assert receipt["zero_reward_allowed"] is False
    assert receipt["improvised_proxy_reward_allowed"] is False
    assert set(receipt["blockers"]) == set(vec_env.REWARD_BLOCKERS)
    assert "optimizer_update" in receipt["prohibited_scope"]
    assert "cold_load_resume" in receipt["prohibited_scope"]
    assert receipt["enforcement_scope"]["vecenv_step_raises_before_physics"] is True
    assert receipt["enforcement_scope"]["upstream_runner_save_load_intercepted"] is False
    assert not any(receipt["authorization"].values())


def test_rsl_step_raises_before_touching_physics():
    instance = object.__new__(vec_env.MujocoN1DiagnosticVecEnv)
    with pytest.raises(vec_env.RewardContractMissing, match="before physics"):
        instance.step(object())


class _FakeCore:
    def __init__(self, index: int):
        self.index = index
        self.binding = SimpleNamespace(
            binding_sha256="a" * 64,
            policy_step_dt_s=0.02,
        )
        self.scene_binding_sha256 = "b" * 64
        self.ticks = 0

    def reset(self, *, robot_tape, question):
        assert robot_tape.plant_binding_sha256 == "a" * 64
        assert question.scene_binding_sha256 == "b" * 64
        self.ticks = 0
        return _groups(float(self.index))

    def step(self, action):
        assert np.asarray(action).shape == (31,)
        self.ticks += 1
        return {
            "observation_groups": _groups(float(self.index + self.ticks)),
            "new_events": [],
        }


def test_torch_vecenv_n8_reset_rollout_is_deterministic_and_no_reward():
    torch = pytest.importorskip("torch")
    tape = SimpleNamespace(
        plant_binding_sha256="a" * 64,
        actions=np.zeros((5, 31), dtype=np.float64),
        source_sha256="c" * 64,
    )
    question = SimpleNamespace(
        scene_binding_sha256="b" * 64,
        source_sha256="d" * 64,
    )
    cores = tuple(_FakeCore(index) for index in range(8))
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=cores,
        robot_tape=tape,
        questions=(question,) * 8,
    )
    # The rsl_rl runner fetches observations from the fresh instance without
    # calling reset first.
    initial, _ = env.get_observations()
    assert initial.shape == (8, 76)
    obs, extras = env.reset()
    assert obs.shape == (8, 76)
    assert extras["observations"]["critic"].shape == (8, 76)
    step = env.diagnostic_step(torch.zeros((8, 31)))
    assert step.observations.shape == (8, 76)
    assert not step.time_outs.any()
    assert env.episode_length_buf.tolist() == [1] * 8

    actions = torch.zeros((3, 8, 31))
    trace_a, receipt_a = env.run_diagnostic_rollout(actions)
    trace_b, receipt_b = env.run_diagnostic_rollout(actions)
    np.testing.assert_array_equal(trace_a, trace_b)
    assert receipt_a["trace_and_event_sha256"] == receipt_b[
        "trace_and_event_sha256"
    ]
    assert receipt_a["status"] == "DIAGNOSTIC_NO_REWARD_ROLLOUT_COMPLETE"
    assert receipt_a["reward_blocker"]["reward_available"] is False
    before = [core.ticks for core in cores]
    with pytest.raises(vec_env.RewardContractMissing):
        env.step(torch.zeros((8, 31)))
    assert [core.ticks for core in cores] == before


def test_real_mujoco_n1_vecenv_finite_rollout_and_reset(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    binding = single_env.load_plant_binding(CONTRACT)
    core = n1.MujocoN1BallCore(binding)
    robot_payload = single_env.build_probe_tape(binding, delay_steps=0)
    robot_path = tmp_path / "robot.json"
    single_env.write_fixed_tape(robot_path, robot_payload)
    robot_tape = single_env.load_fixed_tape(robot_path, binding)
    payload = n1.build_question_payload(
        question_id="vecenv_flight_probe",
        scene_binding_sha256=core.scene_binding_sha256,
        birth_position_w_m=(2.3, 0.0, 1.5),
        birth_linear_velocity_w_mps=(-1.0, 0.0, 0.0),
        landing_aim_xy_w_m=(2.3, 0.0),
        nominal_time_to_contact_s=0.5,
    )
    question_path = tmp_path / "question.json"
    n1.write_question(question_path, payload)
    question = n1.load_question(
        question_path,
        expected_file_sha256=hashlib.sha256(question_path.read_bytes()).hexdigest(),
        scene_binding_sha256=core.scene_binding_sha256,
    )
    env = vec_env.MujocoN1DiagnosticVecEnv(
        cores=(core,), robot_tape=robot_tape, questions=(question,)
    )
    actions = torch.zeros((3, 1, 31))
    trace_a, receipt_a = env.run_diagnostic_rollout(actions)
    trace_b, receipt_b = env.run_diagnostic_rollout(actions)
    np.testing.assert_array_equal(trace_a, trace_b)
    assert trace_a.shape == (4, 1, 76)
    assert receipt_a["trace_and_event_sha256"] == receipt_b[
        "trace_and_event_sha256"
    ]
    assert receipt_a["reward_blocker"]["reward_available"] is False
    sim_time = float(core.data.time)
    with pytest.raises(vec_env.RewardContractMissing):
        env.step(torch.zeros((1, 31)))
    assert float(core.data.time) == sim_time
