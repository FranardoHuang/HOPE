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

    termination = vec_env.termination_blocker_receipt()
    assert termination["formal_termination_available"] is False
    assert termination["terminated_tensor_available"] is False
    assert termination["exact_time_out_latch_available"] is True
    assert set(termination["blockers"]) == set(
        vec_env.FORMAL_TERMINATION_BLOCKERS
    )
    assert termination["reward_paid"] is False
    assert not any(termination["authorization"].values())


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
            control_decimation=4,
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
            "plant": _plant_row(),
            "observation_groups": _groups(float(self.index + self.ticks)),
            "new_events": [],
        }


def _plant_row(**overrides):
    row = {
        "qdes_clamp_joint_events": 0,
        "effort_clip_joint_events": 0,
        "velocity_limit_joint_events": 0,
        "table_contact_pairs": 0,
        "self_contact_pairs": 0,
        "table_contact_substeps": 0,
        "self_contact_substeps": 0,
        "max_table_penetration_m": 0.0,
        "max_self_penetration_m": 0.0,
        "max_joint_velocity_ratio": 0.25,
        "pelvis_height_m": 1.0,
        "pelvis_up_world_z": 1.0,
        "first_table_contact_pair": None,
        "first_self_contact_pair": None,
    }
    row.update(overrides)
    return row


def test_event_ledger_validates_substeps_latches_facts_and_refuses_termination():
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    first = ledger.record_step(
        plant=_plant_row(
            table_contact_pairs=2,
            table_contact_substeps=1,
            max_table_penetration_m=0.001,
            first_table_contact_pair="robot~table",
        ),
        events=(
            {
                "policy_tick": 0,
                "physics_substep": 2,
                "time_s": 0.015,
                "event": "racket",
            },
        ),
        time_out=False,
    )
    assert first["policy_ticks"] == 1
    assert first["physics_substeps"] == 4
    assert first["contact_edge_counts"]["racket"] == 1
    assert first["latches"]["ball_racket_contact_seen"] is True
    assert first["latches"]["robot_obstacle_contact_seen"] is True
    assert first["first_robot_obstacle_contact"] == {
        "policy_tick": 0,
        "pair": "robot~table",
    }
    assert first["termination"] == {
        "exact_time_out_latched": False,
        "formal_hard_termination_available": False,
        "formal_hard_terminated": None,
        "blocker_sha256": vec_env.termination_blocker_receipt()[
            "content_sha256"
        ],
    }
    assert first["reward_paid"] is False

    second = ledger.record_step(
        plant=_plant_row(velocity_limit_joint_events=1),
        events=(),
        time_out=True,
    )
    assert second["policy_ticks"] == 2
    assert second["physics_substeps"] == 8
    assert second["termination"]["exact_time_out_latched"] is True
    assert second["latches"]["joint_velocity_limit_seen"] is True
    assert second["termination"]["formal_hard_terminated"] is None


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            (
                {
                    "policy_tick": 1,
                    "physics_substep": 0,
                    "time_s": 0.005,
                    "event": "racket",
                },
            ),
            "policy tick differs",
        ),
        (
            (
                {
                    "policy_tick": 0,
                    "physics_substep": 4,
                    "time_s": 0.005,
                    "event": "racket",
                },
            ),
            "index is out of range",
        ),
        (
            (
                {
                    "policy_tick": 0,
                    "physics_substep": 0,
                    "time_s": 0.005,
                    "event": "wall",
                },
            ),
            "label is unsupported",
        ),
    ],
)
def test_event_ledger_rejects_malformed_substep_evidence_without_commit(
    events, message
):
    ledger = vec_env.DiagnosticEventLedger(control_decimation=4)
    with pytest.raises(vec_env.VecEnvContractError, match=message):
        ledger.record_step(plant=_plant_row(), events=events, time_out=False)
    assert ledger.policy_ticks == 0
    assert ledger.physics_substeps == 0
    assert not any(ledger.contact_edge_counts.values())


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
    assert len(step.per_env_ledgers) == 8
    assert all(row["policy_ticks"] == 1 for row in step.per_env_ledgers)
    assert all(
        row["termination"]["formal_hard_terminated"] is None
        for row in step.per_env_ledgers
    )

    actions = torch.zeros((3, 8, 31))
    trace_a, receipt_a = env.run_diagnostic_rollout(actions)
    trace_b, receipt_b = env.run_diagnostic_rollout(actions)
    np.testing.assert_array_equal(trace_a, trace_b)
    assert receipt_a["trace_and_event_sha256"] == receipt_b[
        "trace_and_event_sha256"
    ]
    assert receipt_a["status"] == "DIAGNOSTIC_NO_REWARD_ROLLOUT_COMPLETE"
    assert receipt_a["reward_blocker"]["reward_available"] is False
    assert receipt_a["termination_blocker"]["formal_termination_available"] is False
    assert len(receipt_a["event_ledger_transcript"]) == 3
    assert all(
        row["termination"]["formal_hard_terminated"] is None
        for row in receipt_a["final_event_ledgers"]
    )
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
    assert receipt_a["termination_blocker"]["formal_termination_available"] is False
    assert receipt_a["final_event_ledgers"][0]["physics_substeps"] == 12
    assert receipt_a["final_event_ledgers"][0]["reward_paid"] is False
    sim_time = float(core.data.time)
    with pytest.raises(vec_env.RewardContractMissing):
        env.step(torch.zeros((1, 31)))
    assert float(core.data.time) == sim_time
