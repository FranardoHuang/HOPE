"""Host-small contracts for the fixed-centre diagnostic MuJoCo recipe."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

try:
    # Import torch before NumPy-backed native modules on macOS.  The inverse
    # order can load libomp before torch's libiomp and abort the interpreter.
    import torch
except ImportError:
    torch = None

import numpy as np
import pytest

from hope_training.whole_body_tracking.mujoco_native import checkpoint
from hope_training.whole_body_tracking.mujoco_native import (
    fixed_center_recipe as recipe,
)
from hope_training.whole_body_tracking.mujoco_native import n1_reward_event_kernel
from hope_training.whole_body_tracking.mujoco_native import trainer
from hope_training.whole_body_tracking.mujoco_native import vec_env


def _teacher() -> recipe.Frame0JointTeacher:
    return recipe.Frame0JointTeacher(
        joint_pos=tuple(0.0 for _ in range(31)),
        pelvis_height_m=1.0,
        source_motion_sha256="1" * 64,
        source_motion_uid="take061-frame0",
        source_frame_index=0,
        hold_candidate_content_sha256="2" * 64,
    )


def _eligibility(*, target: bool) -> n1_reward_event_kernel.N1RewardEligibility:
    return n1_reward_event_kernel.N1RewardEligibility(
        motion_mimic_denominator=True,
        contact_target_denominator=target,
        closed_swing_denominator=target,
        actual_contact_numerator=False,
        achieved_outgoing_flight_denominator=False,
        predicted_outcome_denominator=False,
        predicted_net_clear_numerator=False,
        predicted_legal_landing_numerator=False,
        observed_outcome_denominator=False,
        observed_net_clear_numerator=False,
        observed_legal_landing_numerator=False,
        unresolved_achieved_flight=False,
        motion_mimic_pay_eligible=True,
        contact_target_pay_eligible=target,
        actual_contact_pay_eligible=False,
        predicted_outcome_pay_eligible=False,
        observed_outcome_pay_eligible=False,
    )


class _FakeCLiteBase:
    def __init__(
        self,
        torch,
        *,
        num_envs: int,
        hidden_contact: bool = False,
        hidden_nonracket_event: bool = False,
    ):
        self.torch = torch
        self.num_envs = num_envs
        self.num_observations = 76
        self.num_actions = 31
        self.device = torch.device("cpu")
        self.c_lite_reward_enabled = True
        self.max_episode_length = 2
        self.step_dt = 0.02
        self.questions = tuple(
            SimpleNamespace(
                nominal_time_to_contact_s=0.04,
                source_sha256="7" * 64,
                birth_position_w_m=np.ones(3),
                birth_linear_velocity_w_mps=np.ones(3),
            )
            for _ in range(num_envs)
        )
        self.cfg = {"kind": "fake_exact_c_lite_base"}
        self._c_lite_reward_receipt = {"content_sha256": "9" * 64}
        self._ticks = np.zeros(num_envs, dtype=np.int64)
        self._boundary = True
        self.hidden_contact = hidden_contact
        self.hidden_nonracket_event = hidden_nonracket_event
        self.reset_calls = 0

    def _row(self, offset: float = 0.0):
        row = self.torch.zeros(76, dtype=self.torch.float32)
        row[62:76] = 1.0 + offset
        return row

    def reset(self, *, seed=None):
        del seed
        self.reset_calls += 1
        self._ticks[:] = 0
        self._boundary = True
        observations = self.torch.stack(
            [self._row(float(index)) for index in range(self.num_envs)]
        )
        return observations, {"observations": {"critic": observations.clone()}}

    def is_reset_boundary(self):
        return bool(self._boundary)

    def diagnostic_training_identity(self):
        return {
            "contract_sha256": "3" * 64,
            "observation_contract_sha256": "4" * 64,
            "action_contract_sha256": "5" * 64,
            "reward_contract_sha256": "9" * 64,
        }

    def diagnostic_training_receipt(self):
        return {
            "ppo_ready": True,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
        }

    def _facts(self, policy_tick: int):
        contact = self.hidden_contact and policy_tick == 1
        return {
            "policy_tick": policy_tick,
            "racket_contact_edge_count_total": int(contact),
            "first_racket_contact_stamp": (
                {"policy_tick": policy_tick, "physics_substep": 0} if contact else None
            ),
            "outgoing_flight": None,
            "observed_outcome_snapshot": {
                "outcome_resolved": False,
                "status": "unarmed_no_outgoing_flight",
                "observed_net_clear": None,
            },
        }

    def diagnostic_step(self, actions):
        assert tuple(actions.shape) == (self.num_envs, 31)
        self._boundary = False
        terminal_rows = []
        observation_rows = []
        facts = []
        physical = []
        ledgers = []
        dones = []
        reset_ids = []
        policy_ticks = []
        for index in range(self.num_envs):
            self._ticks[index] += 1
            tick = int(self._ticks[index])
            policy_ticks.append(tick)
            done = tick == self.max_episode_length
            terminal_rows.append(self._row(float(index)))
            facts.append(self._facts(tick))
            physical.append(
                {
                    "miss_sample_eligible": True,
                    "selected_rubber_center_w_m": (0.0, 0.0, 0.0),
                    "ball_center_w_m": (0.0, 0.0, 0.0),
                    "sample_time_s": tick * self.step_dt,
                }
            )
            ledgers.append(
                {
                    "latest_pelvis_samples": {
                        "height_m": 1.0,
                        "up_world_z": 1.0,
                    }
                }
            )
            dones.append(done)
            if done:
                self._ticks[index] = 0
                reset_ids.append(index)
            observation_rows.append(self._row(float(index)))
        self._boundary = len(reset_ids) == self.num_envs
        done_tensor = self.torch.as_tensor(dones, dtype=self.torch.bool)
        terminal = self.torch.stack(terminal_rows)
        observations = self.torch.stack(observation_rows)
        return vec_env.DiagnosticBatchStep(
            observations=observations,
            terminal_observations=terminal,
            terminal_observation_mask=done_tensor.clone(),
            episode_dones=done_tensor.clone(),
            episode_done_reasons=tuple(
                "time_out" if value else None for value in dones
            ),
            reset_env_ids=tuple(reset_ids),
            exact_phase_fidelity_runtime_available=False,
            per_env_phase_fidelity_samples=(None,) * self.num_envs,
            native_physical_event_runtime_available=True,
            per_env_native_physical_event_facts=tuple(facts),
            per_env_c_lite_physical_samples=tuple(physical),
            per_env_events=tuple(
                ({"event": "ball_hit_net"},)
                if self.hidden_nonracket_event and tick == 1
                else ()
                for tick in policy_ticks
            ),
            per_env_ledgers=tuple(ledgers),
            time_outs=done_tensor.clone(),
            exact_hard_terminations=self.torch.zeros_like(done_tensor),
            exact_hard_termination_reasons=(None,) * self.num_envs,
        )

    def _c_lite_event_eligibility(self, **_kwargs):
        return _eligibility(target=True)


def _wrapped(torch, *, num_envs=1, hidden_contact=False, hidden_nonracket_event=False):
    spec = recipe.FixedCenterRecipeSpec(reset_wait_steps=1)
    base = _FakeCLiteBase(
        torch,
        num_envs=num_envs,
        hidden_contact=hidden_contact,
        hidden_nonracket_event=hidden_nonracket_event,
    )
    base._fixed_center_continuous_wait_preparation = recipe.ContinuousWaitPreparation(
        spec_sha256=spec.content_sha256,
        wait_policy_steps=1,
        wait_physics_substeps=4,
        physics_step_dt_s=0.005,
        per_env=tuple(
            {
                "env_index": index,
                "parent_question_source_sha256": "7" * 64,
                "launch_content_sha256": "8" * 64,
                "wait_s": 0.02,
                "gravity_w_mps2": [0.0, 0.0, -9.81],
                "wind_w_mps": [0.0, 0.0, 0.0],
                "fluid_density": 0.0,
                "fluid_viscosity": 0.0,
                "ball_dof_damping": [0.0] * 6,
                "ball_body_gravcomp": 0.0,
                "ball_qfrc_applied": [0.0] * 6,
                "ball_xfrc_applied": [0.0] * 6,
                "reset_ball_position_w_m": [1.0, 1.0, 1.0],
                "reset_ball_linear_velocity_w_mps": [1.0, 1.0, 1.0],
                "reveal_ball_position_w_m": [1.0 + index] * 3,
                "reveal_ball_linear_velocity_w_mps": [1.0 + index] * 3,
                "parent_nominal_time_to_contact_s": 0.02,
                "derived_nominal_time_to_contact_s": 0.04,
            }
            for index in range(num_envs)
        ),
    )
    return recipe.FixedCenterDiagnosticVecEnv(
        base_env=base,
        teacher_reference=_teacher(),
        spec=spec,
    )


def _torch_or_skip():
    if torch is None:
        pytest.skip("torch is not installed")
    return torch


def test_recipe_receipt_is_explicitly_diagnostic_and_not_full_body():
    spec = recipe.FixedCenterRecipeSpec(reset_wait_steps=2)
    assert len(spec.content_sha256) == 64
    teacher = _teacher()
    assert len(teacher.content_sha256) == 64
    assert "frame0_joint_teacher_is_not_full_body_measured_mimic" in (
        recipe.FORMAL_BLOCKERS
    )
    assert "cpu_sequential_vecenv_has_no_4096_matched_workload_receipt" in (
        recipe.FORMAL_BLOCKERS
    )
    assert "continuous_native_gravity_wait_is_not_cross_engine_launch_parity" in (
        recipe.FORMAL_BLOCKERS
    )
    assert len(recipe.RECIPE_SOURCE_SHA256) == 64


def test_wait_prepared_base_cannot_bypass_outer_identity_or_step():
    base = object.__new__(vec_env.MujocoN1DiagnosticVecEnv)
    base._fixed_center_requires_outer_wrapper = True
    receipt = base.diagnostic_training_receipt()
    assert receipt["ppo_ready"] is False
    assert receipt["blockers"] == ["fixed_center_outer_wrapper_identity_required"]
    with pytest.raises(vec_env.RewardContractMissing, match="private"):
        base.diagnostic_training_identity()
    with pytest.raises(vec_env.RewardContractMissing, match="fixed-center wrapper"):
        base.step(None)


def test_reverse_euler_wait_prefix_reaches_original_fixed_question_birth():
    reveal_position = np.asarray([1.2, -0.3, 1.1], dtype=np.float64)
    reveal_velocity = np.asarray([-2.8, 1.0, 0.24], dtype=np.float64)
    gravity = np.asarray([0.0, 0.0, -9.81], dtype=np.float64)
    reset_position, reset_velocity = recipe._reverse_euler_ballistic_prefix(
        reveal_position=reveal_position,
        reveal_velocity=reveal_velocity,
        gravity=gravity,
        physics_dt=0.005,
        substeps=4,
    )
    position = reset_position.copy()
    velocity = reset_velocity.copy()
    for _ in range(4):
        velocity += gravity * 0.005
        position += velocity * 0.005
    assert np.allclose(position, reveal_position, rtol=0.0, atol=1.0e-15)
    assert np.allclose(velocity, reveal_velocity, rtol=0.0, atol=1.0e-15)


def test_wait_preparation_preserves_external_question_sha_and_seals_launch(
    monkeypatch,
):
    parent_sha = "a" * 64
    question = SimpleNamespace(
        source_path="/authority/question.json",
        source_sha256=parent_sha,
        question_id="fixed-centre",
        scene_binding_sha256="b" * 64,
        birth_position_w_m=np.asarray([1.2, 0.0, 1.1]),
        birth_linear_velocity_w_mps=np.asarray([-2.8, 1.0, 0.24]),
        birth_spin_w_radps=np.zeros(3),
        landing_aim_xy_w_m=np.asarray([2.3, -0.6]),
        nominal_time_to_contact_s=1.84,
        spin_valid=False,
        authority={"kind": "source-bound"},
        selected_rubber_action_lineage={"content_sha256": "c" * 64},
    )
    core = SimpleNamespace(
        model=SimpleNamespace(
            opt=SimpleNamespace(
                integrator=0,
                timestep=0.005,
                gravity=np.asarray([0.0, 0.0, -9.81]),
                wind=np.zeros(3),
                density=0.0,
                viscosity=0.0,
            ),
            dof_damping=np.zeros(6),
            body_gravcomp=np.zeros(1),
        ),
        scene=SimpleNamespace(ball_dof_adr=0, ball_body_id=0),
        data=SimpleNamespace(qfrc_applied=np.zeros(6), xfrc_applied=np.zeros((1, 6))),
    )
    base = SimpleNamespace(
        c_lite_reward_enabled=True,
        cores=(core,),
        questions=(question,),
        num_envs=1,
        control_decimation=4,
        step_dt=0.02,
        robot_tape=object(),
        max_episode_length=100,
    )

    class _Rebuilt:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def diagnostic_training_identity(self):
            if getattr(self, "_fixed_center_requires_outer_wrapper", False):
                raise RuntimeError("outer wrapper required")
            return {
                "contract_sha256": "1" * 64,
                "observation_contract_sha256": "2" * 64,
                "action_contract_sha256": "3" * 64,
                "reward_contract_sha256": "4" * 64,
            }

        def diagnostic_training_receipt(self):
            return {
                "ppo_ready": not getattr(
                    self, "_fixed_center_requires_outer_wrapper", False
                )
            }

    monkeypatch.setattr(recipe.vec_env, "MujocoN1DiagnosticVecEnv", _Rebuilt)
    prepared = recipe.prepare_continuous_wait_base(
        base, recipe.FixedCenterRecipeSpec(reset_wait_steps=1)
    )
    assert prepared.questions[0].source_path == question.source_path
    assert prepared.questions[0].source_sha256 == parent_sha
    launch_sha = prepared._fixed_center_continuous_wait_preparation.per_env[0][
        "launch_content_sha256"
    ]
    assert launch_sha != parent_sha
    assert len(launch_sha) == 64
    assert prepared.diagnostic_training_receipt()["ppo_ready"] is False
    with pytest.raises(ValueError):
        prepared.questions[0].birth_position_w_m[0] = 99.0


def test_preparation_mutation_fails_closed_before_physics():
    torch_module = _torch_or_skip()
    env = _wrapped(torch_module)
    env.continuous_wait_preparation.per_env[0][
        "derived_nominal_time_to_contact_s"
    ] = 0.06
    with pytest.raises(recipe.FixedCenterRecipeError, match="preparation mutated"):
        env.step(torch_module.zeros((1, 31)))


def test_wait_masks_task_then_atomically_reveals_and_keeps_mimic_balance():
    torch_module = _torch_or_skip()
    env = _wrapped(torch_module)
    assert env.base_env.reset_calls == 1
    observations, extras = env.get_observations()
    assert env.task_valid == (False,)
    assert torch_module.equal(
        observations[0, recipe.TASK_SLICE], torch_module.zeros(14)
    )
    assert extras["formal_authorized"] is False

    observations, rewards, dones, extras = env.step(torch_module.zeros((1, 31)))
    assert env.task_valid == (True,)
    assert extras["task_valid_transition"] == [False]
    assert extras["task_valid_next"] == [True]
    assert torch_module.equal(observations[0, recipe.TASK_SLICE], torch_module.ones(14))
    terms = extras["reward_terms"][0]
    assert terms["motion_reward"] == 0.25
    assert terms["balance_reward"] == 0.1
    assert terms["miss_proximity_reward"] == 0.0
    assert terms["nominal_strike_sampled_now"] is False
    assert terms["exact_strike_timing_tick_count"] == 0
    assert terms["contact_target_denominator"] is False
    assert terms["closed_swing_denominator"] is False
    assert terms["observed_outcome_denominator"] is False
    assert rewards.tolist() == pytest.approx([0.35])
    assert dones.tolist() == [False]
    assert env.base_env.reset_calls == 1

    observations, rewards, dones, extras = env.step(torch_module.zeros((1, 31)))
    assert extras["task_valid_transition"] == [True]
    assert extras["reward_terms"][0]["miss_proximity_reward"] == 1.0
    assert extras["reward_terms"][0]["nominal_strike_sampled_now"] is True
    assert extras["reward_terms"][0]["exact_strike_timing_tick_count"] == 1
    assert extras["reward_terms"][0]["strike_opportunity_count"] == 1
    assert extras["reward_terms"][0]["contact_target_denominator"] is True
    assert rewards.tolist() == pytest.approx([1.35])
    assert dones.tolist() == [True]
    assert env.task_valid == (False,)
    assert torch_module.equal(
        observations[0, recipe.TASK_SLICE], torch_module.zeros(14)
    )
    assert env.is_reset_boundary() is True


def test_hidden_contact_before_reveal_fails_closed():
    torch_module = _torch_or_skip()
    env = _wrapped(torch_module, hidden_contact=True)
    with pytest.raises(
        recipe.FixedCenterRecipeError,
        match="hidden task produced contact/outcome",
    ):
        env.step(torch_module.zeros((1, 31)))
    with pytest.raises(recipe.FixedCenterRecipeError, match="must be reset"):
        env.step(torch_module.zeros((1, 31)))


def test_any_hidden_ball_contact_event_before_reveal_fails_closed():
    torch_module = _torch_or_skip()
    env = _wrapped(torch_module, hidden_nonracket_event=True)
    with pytest.raises(
        recipe.FixedCenterRecipeError,
        match="hidden task produced contact/outcome",
    ):
        env.step(torch_module.zeros((1, 31)))


def test_host_small_batched_ppo_and_boundary_cold_load_parity(tmp_path: Path):
    torch_module = _torch_or_skip()

    def build():
        env = _wrapped(torch_module, num_envs=4)
        identity = trainer.TrainerIdentity(**env.diagnostic_training_identity())
        config = trainer.DiagnosticPPOConfig(
            observation_dim=76,
            action_dim=31,
            rollout_steps=2,
            hidden_dims=(8,),
            seed=41,
            learning_rate=1.0e-3,
            initial_action_std=0.02,
        )
        return env, identity, config

    env, identity, config = build()
    source = trainer.MujocoDiagnosticPPOTrainer(
        env=env, identity=identity, config=config
    )
    first = source.run_update()
    assert first["update_counter"] == 1
    assert first["at_reset_boundary"] is True
    path = tmp_path / "fixed_center_boundary.pt"
    checkpoint.ResetBoundaryCheckpoint().save(path, source)
    expected = source.run_update()
    expected_state = copy.deepcopy(source.model.state_dict())

    cold_env, cold_identity, cold_config = build()
    assert cold_identity == identity
    assert cold_config == config
    cold = trainer.MujocoDiagnosticPPOTrainer(
        env=cold_env, identity=cold_identity, config=cold_config
    )
    checkpoint.ResetBoundaryCheckpoint().load(path, cold)
    actual = cold.run_update()
    assert actual == expected
    for name, value in cold.model.state_dict().items():
        assert torch_module.equal(value, expected_state[name])
