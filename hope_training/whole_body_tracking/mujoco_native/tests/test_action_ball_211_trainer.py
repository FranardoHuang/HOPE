"""Asymmetric trainer/checkpoint tests for the strict 211/319 MuJoCo ABI."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hope_training.whole_body_tracking.mujoco_native import action_ball_211_abi as abi
from hope_training.whole_body_tracking.mujoco_native import checkpoint
from hope_training.whole_body_tracking.mujoco_native import trainer


def _digest(character: str) -> str:
    return character * 64


def _identity(profile: abi.ActionBall211Profile) -> trainer.TrainerIdentity:
    return trainer.TrainerIdentity(
        contract_sha256=_digest("a"),
        observation_contract_sha256=profile.observation_contract_sha256,
        action_contract_sha256=_digest("c"),
        reward_contract_sha256=_digest("d"),
    )


class FakeActionBall211Env:
    """Deterministic WAIT->ACTIVE->compact-reset protocol fake."""

    def __init__(
        self,
        profile: abi.ActionBall211Profile,
        *,
        num_envs: int = 4,
        expose_critic: bool = True,
        horizon: int = 2,
    ) -> None:
        self.profile = profile
        self.identity = _identity(profile)
        self.num_envs = num_envs
        self.num_observations = profile.actor.width
        self.num_privileged_observations = profile.critic.width
        self.num_actions = 3
        self.device = "cpu"
        self.expose_critic = expose_critic
        self.horizon = horizon
        self.reset_calls = 0
        self.step_calls = 0
        self._tick = 0
        self._boundary = True

    def diagnostic_training_receipt(self):
        return {
            "kind": trainer.DIAGNOSTIC_TRAINER_RECEIPT_KIND,
            "ppo_ready": True,
            "reward_available": True,
            "normal_step_available": True,
            "reset_boundary_checkpoint_available": True,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
            "blockers": [],
            "normalizer_binding": trainer.asymmetric_normalizer_binding(
                profile_observation_contract_sha256=(
                    self.profile.observation_contract_sha256
                ),
                actor_width=self.profile.actor.width,
                critic_width=self.profile.critic.width,
                actor_normalizer_identity=(
                    self.profile.actor_normalizer_identity
                ),
                critic_normalizer_identity=(
                    self.profile.critic_normalizer_identity
                ),
                actor_task_mask_indices=self.profile.actor.task_mask_indices,
                critic_task_mask_indices=self.profile.critic.task_mask_indices,
                actor_task_valid_index=self.profile.actor.task_valid_index,
                critic_task_valid_index=self.profile.critic.task_valid_index,
                epsilon=1.0e-5,
            ),
            **self.identity.as_dict(),
        }

    def is_reset_boundary(self):
        return self._boundary

    def _lane(self, lane: abi.ObservationLane, *, active: bool, phase: float):
        values = torch.arange(self.num_envs * lane.width, dtype=torch.float32).reshape(
            self.num_envs, lane.width
        )
        values = values * 1.0e-4 + phase
        if active:
            values[:, lane.task_mask_indices] = 0.25 + phase
            values[:, lane.task_valid_index] = 1.0
        else:
            values[:, lane.task_mask_indices] = 0.0
            values[:, lane.task_valid_index] = 0.0
        return values

    def _pair(self, *, active: bool):
        actor = self._lane(self.profile.actor, active=active, phase=0.1)
        critic = self._lane(self.profile.critic, active=active, phase=0.6)
        extras = {}
        if self.expose_critic:
            extras["observations"] = {"critic": critic}
        return actor, extras

    def reset(self, *, seed: int):
        assert type(seed) is int
        self.reset_calls += 1
        self._tick = 0
        self._boundary = False
        return self._pair(active=False)

    def step(self, actions):
        self.step_calls += 1
        self._tick += 1
        done = self._tick >= self.horizon
        actor, extras = self._pair(active=not done)
        rewards = -actions.square().sum(dim=-1) + self._tick * 0.01
        dones = torch.full((self.num_envs,), done, dtype=torch.bool)
        extras["time_outs"] = dones.clone()
        self._boundary = done
        if done:
            # Mirror the native VecEnv compact reset: the returned row is an
            # inactive reset boundary and the next step starts a fresh episode.
            self._tick = 0
        return actor, rewards, dones, extras


class ExactTerminalTelemetryEnv(FakeActionBall211Env):
    """Script independent hard/timeout axes across exact WAIT/ACTIVE ticks."""

    def __init__(
        self,
        profile: abi.ActionBall211Profile,
        *,
        forbidden_ticks_by_env: tuple[int, ...] = (0, 0, 0, 0),
    ) -> None:
        super().__init__(profile, num_envs=4, horizon=2)
        self.forbidden_ticks_by_env = forbidden_ticks_by_env

    def diagnostic_training_receipt(self):
        receipt = super().diagnostic_training_receipt()
        receipt["terminal_row_telemetry_available"] = True
        receipt["terminal_row_telemetry_contract"] = (
            trainer.terminal_row_telemetry_contract()
        )
        return receipt

    @staticmethod
    def _hard_event(*, tick: int, reason: str, physics_substep=None):
        timing = (
            "physics_substep"
            if physics_substep is not None
            else "post_control_step"
        )
        return {
            "policy_tick": tick - 1,
            "sample_timing": timing,
            "physics_substep": physics_substep,
            "reason": reason,
            "all_reasons": [reason],
        }

    @classmethod
    def _ledger(
        cls,
        *,
        tick: int,
        hard_reason=None,
        physics_substep=None,
        forbidden_ticks: int = 0,
    ):
        # The reason string comes from the shared constant, not a fourth spelling
        # of it: a fake that hand-copies the vocabulary stops testing agreement.
        return {
            "policy_ticks": tick,
            "termination": {
                "exact_time_out_latched": False,
                "exact_hard_terminated": hard_reason is not None,
                "exact_hard_reason": hard_reason,
            },
            "first_exact_hard_termination": (
                None
                if hard_reason is None
                else cls._hard_event(
                    tick=tick,
                    reason=hard_reason,
                    physics_substep=physics_substep,
                )
            ),
            "joint_actual_forbidden_observed_ticks": forbidden_ticks,
            "promotion_blocking_evidence": {
                "promotion_blocked": forbidden_ticks > 0,
                "reasons": (
                    [trainer.PROMOTION_BLOCKING_REASON]
                    if forbidden_ticks > 0
                    else []
                ),
                "semantics": "fake env mirror of the live ledger conclusion",
            },
        }

    def step(self, actions):
        self.step_calls += 1
        self._tick += 1
        if self._tick == 1:
            active = [False, True, True, True]
            ticks = [1, 92, 104, 104]
            reasons = ["base_too_low", None, None, None]
            hard_reasons = ["base_too_low", None, None, None]
            time_out_values = [False, False, False, False]
            waits = [25, 25, 25, 25]
            generations = [1, 1, 1, 1]
        else:
            active = [True, True, True, True]
            ticks = [93, 105, 105, 93]
            reasons = [
                "anchor_ori",
                "action_ball_single_stroke_complete",
                "action_ball_single_stroke_complete",
                "anchor_ori",
            ]
            hard_reasons = ["anchor_ori", None, "robot_hit_table", "anchor_ori"]
            time_out_values = [False, True, True, False]
            waits = [1, 25, 25, 1]
            generations = [2, 1, 1, 1]

        dones = torch.as_tensor(
            [reason is not None for reason in reasons], dtype=torch.bool
        )
        time_outs = torch.as_tensor(time_out_values, dtype=torch.bool)
        exact_hard = torch.as_tensor(
            [reason is not None for reason in hard_reasons], dtype=torch.bool
        )
        actor = self._lane(
            self.profile.actor, active=self._tick == 1, phase=0.1
        )
        critic = self._lane(
            self.profile.critic, active=self._tick == 1, phase=0.6
        )
        if self._tick == 1:
            actor[0, self.profile.actor.task_mask_indices] = 0.0
            actor[0, self.profile.actor.task_valid_index] = 0.0
            critic[0, self.profile.critic.task_mask_indices] = 0.0
            critic[0, self.profile.critic.task_valid_index] = 0.0
        else:
            actor[:, self.profile.actor.task_mask_indices] = 0.0
            actor[:, self.profile.actor.task_valid_index] = 0.0
            critic[:, self.profile.critic.task_mask_indices] = 0.0
            critic[:, self.profile.critic.task_valid_index] = 0.0
        rewards = -actions.square().sum(dim=-1) + self._tick * 0.01
        ledgers = []
        for index, (tick, hard_reason) in enumerate(zip(ticks, hard_reasons)):
            ledger = self._ledger(
                tick=tick,
                hard_reason=hard_reason,
                physics_substep=(2 if index == 2 else None),
                forbidden_ticks=self.forbidden_ticks_by_env[index],
            )
            ledger["termination"]["exact_time_out_latched"] = bool(
                time_out_values[index]
            )
            ledgers.append(ledger)
        extras = {
            "observations": {"critic": critic},
            "time_outs": time_outs,
            "episode_done_reasons": reasons,
            "task_valid_transition": active,
            "wait_assignment_transition": [
                {
                    "env_id": index,
                    "reset_generation": generations[index],
                    "wait_ticks": waits[index],
                }
                for index in range(self.num_envs)
            ],
            "reward_terms": [
                {
                    "task_valid": active[index],
                    "sample_policy_tick_1based": ticks[index],
                }
                for index in range(self.num_envs)
            ],
            "diagnostic_event_ledgers": ledgers,
            "diagnostic_exact_hard_terminations": exact_hard,
            "diagnostic_exact_hard_termination_reasons": hard_reasons,
        }
        self._boundary = bool(torch.all(dones).item())
        if self._boundary:
            self._tick = 0
        return actor, rewards, dones, extras


def _config(profile: abi.ActionBall211Profile):
    return trainer.DiagnosticPPOConfig(
        **profile.trainer_config_kwargs(),
        action_dim=3,
        rollout_steps=2,
        hidden_dims=(8,),
        seed=53,
        learning_rate=1.0e-3,
    )


def _build_exact(profile: abi.ActionBall211Profile):
    env = ExactTerminalTelemetryEnv(profile)
    return env, trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=env.identity,
        config=_config(profile),
    )


def _build(profile: abi.ActionBall211Profile, *, expose_critic=True):
    env = FakeActionBall211Env(profile, expose_critic=expose_critic)
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=env.identity,
        config=_config(profile),
    )
    return env, instance


@pytest.mark.parametrize("profile", (abi.A211_PROFILE, abi.C211_PROFILE))
def test_networks_and_normalizers_are_real_asymmetric_objects(profile):
    env, instance = _build(profile)
    assert instance.model.actor[0].in_features == 211
    assert instance.model.critic[0].in_features == 319
    assert instance.actor_normalizer is not instance.critic_normalizer
    receipt = instance.run_update()
    assert env.reset_calls == 1
    assert env.step_calls == 2
    assert receipt["actor_observation_dim"] == 211
    assert receipt["critic_observation_dim"] == 319
    assert receipt["actor_normalizer_identity"] == profile.actor_normalizer_identity
    assert receipt["critic_normalizer_identity"] == profile.critic_normalizer_identity
    assert (
        receipt["actor_normalizer_state_sha256"]
        != receipt["critic_normalizer_state_sha256"]
    )
    assert receipt["at_reset_boundary"] is True


def test_fresh_actor_hold_bootstrap_is_exact_and_vecenv_bound(monkeypatch):
    profile = abi.C211_PROFILE
    env = FakeActionBall211Env(profile)
    bias = (2.5, -1.25, 0.75)
    bootstrap = trainer.fresh_actor_bootstrap_contract(
        bias, initial_action_std=0.02
    )
    original_receipt = env.diagnostic_training_receipt

    def receipt_with_bootstrap():
        receipt = original_receipt()
        receipt["fresh_actor_bootstrap"] = copy.deepcopy(bootstrap)
        return receipt

    monkeypatch.setattr(env, "diagnostic_training_receipt", receipt_with_bootstrap)
    config = trainer.DiagnosticPPOConfig(
        **profile.trainer_config_kwargs(),
        action_dim=3,
        rollout_steps=2,
        hidden_dims=(8,),
        seed=53,
        learning_rate=1.0e-3,
        initial_action_std=0.02,
        fresh_actor_output_bias=bias,
        fresh_actor_bootstrap_authority_sha256=bootstrap["content_sha256"],
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env, identity=env.identity, config=config
    )
    output = instance.model.actor[-1]
    assert torch.count_nonzero(output.weight).item() == 0
    assert torch.equal(output.bias, torch.tensor(bias, dtype=output.bias.dtype))
    rows = torch.randn(7, profile.actor.width)
    assert torch.equal(
        instance.model.actor(rows),
        torch.tensor(bias, dtype=rows.dtype).expand(7, -1),
    )
    receipt = instance.run_update()
    assert receipt["fresh_actor_bootstrap"] == bootstrap


def test_fresh_actor_bootstrap_requires_exact_bias_width_and_std():
    kwargs = abi.C211_PROFILE.trainer_config_kwargs()
    with pytest.raises(trainer.DiagnosticPPOContractError, match="one bias per action"):
        trainer.DiagnosticPPOConfig(
            **kwargs,
            action_dim=3,
            initial_action_std=0.02,
            fresh_actor_output_bias=(1.0, 2.0),
            fresh_actor_bootstrap_authority_sha256=_digest("e"),
        )
    with pytest.raises(trainer.DiagnosticPPOContractError, match="initial_action_std"):
        trainer.DiagnosticPPOConfig(
            **kwargs,
            action_dim=3,
            initial_action_std=0.2,
            fresh_actor_output_bias=(1.0, 2.0, 3.0),
            fresh_actor_bootstrap_authority_sha256=_digest("e"),
        )


def test_missing_privileged_critic_fails_before_first_physics_step():
    env, instance = _build(abi.A211_PROFILE, expose_critic=False)
    with pytest.raises(trainer.DiagnosticPPOContractError, match="observations.critic"):
        instance.run_update()
    assert env.reset_calls == 1
    assert env.step_calls == 0
    assert instance.update_counter == 0


@pytest.mark.parametrize(
    "env_profile,config_profile",
    (
        (abi.A211_PROFILE, abi.C211_PROFILE),
        (abi.C211_PROFILE, abi.A211_PROFILE),
    ),
)
def test_same_width_cross_profile_normalizer_config_is_rejected_before_reset(
    env_profile, config_profile
):
    env = FakeActionBall211Env(env_profile)
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=env.identity,
        config=_config(config_profile),
    )
    with pytest.raises(trainer.DiagnosticPPOBlocked, match="normalizer/profile/WAIT"):
        instance.run_update()
    assert env.reset_calls == env.step_calls == 0
    assert instance.update_counter == 0


def test_empty_wait_masks_cannot_hide_behind_correct_widths_and_identities():
    profile = abi.C211_PROFILE
    env = FakeActionBall211Env(profile)
    kwargs = profile.trainer_config_kwargs()
    kwargs.update(
        {
            "actor_task_mask_indices": (),
            "critic_task_mask_indices": (),
            "actor_task_valid_index": None,
            "critic_task_valid_index": None,
        }
    )
    config = trainer.DiagnosticPPOConfig(
        **kwargs,
        action_dim=3,
        rollout_steps=2,
        hidden_dims=(8,),
        seed=53,
        learning_rate=1.0e-3,
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env, identity=env.identity, config=config
    )
    with pytest.raises(trainer.DiagnosticPPOBlocked, match="normalizer/profile/WAIT"):
        instance.run_update()
    assert env.reset_calls == env.step_calls == 0
    assert instance.update_counter == 0


def test_nonterminal_bootstrap_normalizes_without_double_updating_moments():
    profile = abi.C211_PROFILE
    env = FakeActionBall211Env(profile, num_envs=2, horizon=20)
    config = trainer.DiagnosticPPOConfig(
        **profile.trainer_config_kwargs(),
        action_dim=3,
        rollout_steps=4,
        hidden_dims=(8,),
        seed=53,
        learning_rate=1.0e-3,
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env, identity=env.identity, config=config
    )
    first = instance.run_update()
    assert first["at_reset_boundary"] is False
    assert instance.actor_normalizer.count.item() == 8.0
    assert instance.critic_normalizer.count.item() == 8.0
    second = instance.run_update()
    assert second["at_reset_boundary"] is False
    assert instance.actor_normalizer.count.item() == 16.0
    assert instance.critic_normalizer.count.item() == 16.0


def test_rollout_extension_keeps_collecting_on_policy_until_reset_boundary():
    profile = abi.C211_PROFILE
    env = FakeActionBall211Env(profile, num_envs=2, horizon=3)
    config = trainer.DiagnosticPPOConfig(
        **profile.trainer_config_kwargs(),
        action_dim=3,
        rollout_steps=4,
        rollout_reset_boundary_extension_steps=3,
        hidden_dims=(8,),
        seed=53,
        learning_rate=1.0e-3,
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env, identity=env.identity, config=config
    )
    receipt = instance.run_update()
    assert env.step_calls == 6
    assert receipt["rollout_steps"] == 6
    assert receipt["minimum_rollout_steps"] == 4
    assert receipt["maximum_rollout_steps"] == 7
    assert receipt["reset_boundary_extension_steps_used"] == 2
    assert receipt["batch_size"] == 12
    assert receipt["at_reset_boundary"] is True
    assert instance.actor_normalizer.count.item() == 12.0


def test_rollout_extension_fails_if_boundary_not_reached_before_cap():
    profile = abi.A211_PROFILE
    env = FakeActionBall211Env(profile, num_envs=2, horizon=20)
    config = trainer.DiagnosticPPOConfig(
        **profile.trainer_config_kwargs(),
        action_dim=3,
        rollout_steps=4,
        rollout_reset_boundary_extension_steps=3,
        hidden_dims=(8,),
        seed=53,
        learning_rate=1.0e-3,
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env, identity=env.identity, config=config
    )
    with pytest.raises(trainer.ResetBoundaryRequired, match="extension exhausted"):
        instance.run_update()
    assert env.step_calls == 7
    assert instance.update_counter == 0


def test_rollout_extension_width_must_be_nonnegative_plain_integer():
    kwargs = abi.C211_PROFILE.trainer_config_kwargs()
    for value in (-1, 1.5, True):
        with pytest.raises(
            trainer.DiagnosticPPOContractError, match="non-negative plain integer"
        ):
            trainer.DiagnosticPPOConfig(
                **kwargs,
                action_dim=3,
                rollout_reset_boundary_extension_steps=value,
            )


def test_timeout_bootstrap_matches_rsl_rl_pre_step_value_rule():
    profile = abi.C211_PROFILE
    env = FakeActionBall211Env(profile, num_envs=2, horizon=1)
    config = trainer.DiagnosticPPOConfig(
        **profile.trainer_config_kwargs(),
        action_dim=3,
        rollout_steps=1,
        hidden_dims=(8,),
        seed=53,
        learning_rate=1.0e-3,
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env, identity=env.identity, config=config
    )
    output = instance.model.critic[-1]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.fill_(2.0)
    receipt = instance.run_update()
    expected_bootstrap = 2 * config.gamma * 2.0
    assert receipt["timeout_bootstrap_rule"] == trainer.TIMEOUT_BOOTSTRAP_RULE
    assert receipt["timeout_row_count"] == 2
    assert receipt["hard_terminal_row_count"] == 0
    assert receipt["timeout_bootstrap_reward_sum"] == pytest.approx(
        expected_bootstrap
    )
    assert receipt["ppo_reward_sum"] == pytest.approx(
        receipt["raw_reward_sum"] + expected_bootstrap
    )


def test_timeout_mask_must_be_bool_shape_and_subset_of_done(monkeypatch):
    env, instance = _build(abi.A211_PROFILE)
    original_step = env.step

    def invalid_step(actions):
        actor, rewards, dones, extras = original_step(actions)
        extras["time_outs"] = ~dones
        return actor, rewards, dones, extras

    monkeypatch.setattr(env, "step", invalid_step)
    with pytest.raises(trainer.DiagnosticPPOContractError, match="subset of dones"):
        instance.run_update()
    assert instance.update_counter == 0


@pytest.mark.parametrize("profile", (abi.A211_PROFILE, abi.C211_PROFILE))
def test_exact_terminal_rows_preserve_hard_timeout_overlap_phase_and_ticks(profile):
    env = ExactTerminalTelemetryEnv(profile)
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=env.identity,
        config=_config(profile),
    )
    receipt = instance.run_update()
    telemetry = receipt["terminal_row_telemetry"]
    assert receipt["terminal_row_telemetry_available"] is True
    assert receipt["hard_terminal_row_count"] == 3
    assert receipt["timeout_row_count"] == 2
    assert telemetry["terminal_row_count"] == 5
    assert telemetry["hard_only_row_count"] == 3
    assert telemetry["exact_hard_termination_row_count"] == 4
    assert telemetry["timeout_row_count"] == 2
    assert telemetry["timeout_only_row_count"] == 1
    assert telemetry["single_stroke_timeout_row_count"] == 2
    assert telemetry["horizon_timeout_row_count"] == 0
    assert telemetry["coincident_exact_hard_and_timeout_row_count"] == 1
    assert telemetry["reason_histogram"] == {
        "action_ball_single_stroke_complete": 2,
        "anchor_ori": 2,
        "base_too_low": 1,
    }
    assert telemetry["exact_hard_reason_histogram"] == {
        "anchor_ori": 2,
        "base_too_low": 1,
        "robot_hit_table": 1,
    }
    assert telemetry["phase_histogram"] == {"RESET_WAIT": 1, "TASK_ACTIVE": 4}
    assert telemetry["reason_phase_tick_histogram"] == [
        {
            "reason": "action_ball_single_stroke_complete",
            "termination_class": "single_stroke_timeout",
            "exact_hard_termination": False,
            "time_out": True,
            "exact_hard_reason": None,
            "phase": "TASK_ACTIVE",
            "episode_transition_tick_1based": 105,
            "phase_transition_tick_1based": 80,
            "count": 1,
        },
        {
            "reason": "action_ball_single_stroke_complete",
            "termination_class": "single_stroke_timeout",
            "exact_hard_termination": True,
            "time_out": True,
            "exact_hard_reason": "robot_hit_table",
            "phase": "TASK_ACTIVE",
            "episode_transition_tick_1based": 105,
            "phase_transition_tick_1based": 80,
            "count": 1,
        },
        {
            "reason": "anchor_ori",
            "termination_class": "hard",
            "exact_hard_termination": True,
            "time_out": False,
            "exact_hard_reason": "anchor_ori",
            "phase": "TASK_ACTIVE",
            "episode_transition_tick_1based": 93,
            "phase_transition_tick_1based": 92,
            "count": 2,
        },
        {
            "reason": "base_too_low",
            "termination_class": "hard",
            "exact_hard_termination": True,
            "time_out": False,
            "exact_hard_reason": "base_too_low",
            "phase": "RESET_WAIT",
            "episode_transition_tick_1based": 1,
            "phase_transition_tick_1based": 1,
            "count": 1,
        },
    ]
    assert len(telemetry["content_sha256"]) == 64
    assert receipt["at_reset_boundary"] is True


@pytest.mark.parametrize(
    "fault",
    (
        "missing_sideband",
        "missing_hard_reason",
        "episode_reason_missing",
        "reward_phase_mismatch",
        "reward_tick_mismatch",
        "done_without_hard_or_timeout",
    ),
)
def test_exact_terminal_telemetry_faults_fail_before_optimizer(monkeypatch, fault):
    env = ExactTerminalTelemetryEnv(abi.A211_PROFILE)
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=env.identity,
        config=_config(abi.A211_PROFILE),
    )
    before = copy.deepcopy(instance.model.state_dict())
    original_step = env.step

    def invalid_step(actions):
        actor, rewards, dones, extras = original_step(actions)
        if fault == "missing_sideband":
            extras.pop("diagnostic_event_ledgers")
        elif fault == "missing_hard_reason":
            extras["diagnostic_exact_hard_termination_reasons"][0] = None
        elif fault == "episode_reason_missing":
            extras["episode_done_reasons"][0] = None
        elif fault == "reward_phase_mismatch":
            extras["reward_terms"][0]["task_valid"] = True
        elif fault == "reward_tick_mismatch":
            extras["reward_terms"][0]["sample_policy_tick_1based"] += 1
        elif fault == "done_without_hard_or_timeout":
            extras["diagnostic_exact_hard_terminations"][0] = False
            extras["diagnostic_exact_hard_termination_reasons"][0] = None
            extras["diagnostic_event_ledgers"][0]["termination"][
                "exact_hard_terminated"
            ] = False
            extras["diagnostic_event_ledgers"][0]["termination"][
                "exact_hard_reason"
            ] = None
            extras["diagnostic_event_ledgers"][0][
                "first_exact_hard_termination"
            ] = None
        return actor, rewards, dones, extras

    monkeypatch.setattr(env, "step", invalid_step)
    with pytest.raises(trainer.DiagnosticPPOContractError, match="terminal"):
        instance.run_update()
    assert instance.update_counter == 0
    for name, value in instance.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_clean_update_publishes_an_unblocked_promotion_conclusion():
    env, instance = _build_exact(abi.A211_PROFILE)
    receipt = instance.run_update()
    evidence = receipt["promotion_blocking_evidence"]
    assert evidence["kind"] == trainer.PROMOTION_BLOCKING_EVIDENCE_KIND
    assert evidence["promotion_blocked"] is False
    assert evidence["reasons"] == []
    assert evidence["blocked_sample_count"] == 0
    assert evidence["blocked_env_indices"] == []
    assert evidence["first_blocked_sample"] is None
    # It must have actually looked at every env on every rollout step; an
    # evidence block computed over nothing would also say "not blocked".
    assert evidence["checked_sample_count"] == (
        receipt["rollout_steps"] * receipt["num_envs"]
    )
    assert len(evidence["content_sha256"]) == 64


def test_observed_hard_edge_reaches_the_update_receipt_conclusion():
    env = ExactTerminalTelemetryEnv(
        abi.A211_PROFILE, forbidden_ticks_by_env=(0, 0, 3, 0)
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=env.identity,
        config=_config(abi.A211_PROFILE),
    )
    receipt = instance.run_update()
    evidence = receipt["promotion_blocking_evidence"]
    assert evidence["promotion_blocked"] is True
    assert evidence["reasons"] == [trainer.PROMOTION_BLOCKING_REASON]
    assert evidence["blocked_env_indices"] == [2]
    assert evidence["blocked_sample_count"] == receipt["rollout_steps"]
    first = evidence["first_blocked_sample"]
    assert first["rollout_step_1based"] == 1
    assert first["env_index"] == 2
    assert first["joint_actual_forbidden_observed_ticks"] == 3


@pytest.mark.parametrize(
    "mutation",
    (
        "evidence_removed",
        "conclusion_hardcoded_false",
        "conclusion_hardcoded_true",
        "conclusion_is_a_truthy_int",
        "reasons_emptied_while_blocked",
        "counter_removed",
        "evidence_is_not_a_mapping",
    ),
)
def test_promotion_conclusion_mutations_fail_before_the_optimizer(
    monkeypatch, mutation
):
    """Every way the conclusion bit can rot must stop the update.

    人话:这些变异体就是"一个粗一个档次的检查会漏掉的"那些——把结论写死 False、
    把结论删掉、把结论和它自己的计数拆开各说各话。只断言"字段还在"是抓不到的。
    """

    env = ExactTerminalTelemetryEnv(
        abi.A211_PROFILE, forbidden_ticks_by_env=(0, 0, 3, 0)
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=env.identity,
        config=_config(abi.A211_PROFILE),
    )
    before = copy.deepcopy(instance.model.state_dict())
    original_step = env.step

    def mutated_step(actions):
        actor, rewards, dones, extras = original_step(actions)
        ledger = extras["diagnostic_event_ledgers"][2]
        if mutation == "evidence_removed":
            ledger.pop("promotion_blocking_evidence")
        elif mutation == "conclusion_hardcoded_false":
            ledger["promotion_blocking_evidence"]["promotion_blocked"] = False
        elif mutation == "conclusion_hardcoded_true":
            for row in extras["diagnostic_event_ledgers"]:
                row["promotion_blocking_evidence"]["promotion_blocked"] = True
        elif mutation == "conclusion_is_a_truthy_int":
            ledger["promotion_blocking_evidence"]["promotion_blocked"] = 1
        elif mutation == "reasons_emptied_while_blocked":
            ledger["promotion_blocking_evidence"]["reasons"] = []
        elif mutation == "counter_removed":
            ledger.pop("joint_actual_forbidden_observed_ticks")
        elif mutation == "evidence_is_not_a_mapping":
            ledger["promotion_blocking_evidence"] = [
                trainer.PROMOTION_BLOCKING_REASON
            ]
        return actor, rewards, dones, extras

    monkeypatch.setattr(env, "step", mutated_step)
    with pytest.raises(trainer.DiagnosticPPOContractError, match="promotion"):
        instance.run_update()
    assert instance.update_counter == 0
    for name, value in instance.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_checkpoint_receipts_carry_the_promotion_conclusion(tmp_path: Path):
    env = ExactTerminalTelemetryEnv(
        abi.A211_PROFILE, forbidden_ticks_by_env=(0, 0, 3, 0)
    )
    instance = trainer.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=env.identity,
        config=_config(abi.A211_PROFILE),
    )
    instance.run_update()
    blocked_path = tmp_path / "blocked.pt"
    save = checkpoint.ResetBoundaryCheckpoint().save(blocked_path, instance)
    assert save["promotion_blocked"] is True

    clean_env, clean = _build_exact(abi.A211_PROFILE)
    clean.run_update()
    clean_path = tmp_path / "clean.pt"
    clean_save = checkpoint.ResetBoundaryCheckpoint().save(clean_path, clean)
    assert clean_save["promotion_blocked"] is False
    load = checkpoint.ResetBoundaryCheckpoint().load(clean_path, clean)
    assert load["promotion_blocked"] is False


def test_checkpoint_without_an_update_receipt_reports_blocked(tmp_path: Path):
    """No evidence is not the same as no fault; it is the same as blocked."""

    _env, instance = _build_exact(abi.A211_PROFILE)
    save = checkpoint.ResetBoundaryCheckpoint().save(
        tmp_path / "never_updated.pt", instance
    )
    assert save["promotion_blocked"] is True


def test_promotion_evidence_receipt_refuses_a_vacuous_sample_count():
    with pytest.raises(trainer.DiagnosticPPOContractError):
        trainer.promotion_blocking_evidence_receipt([], checked_sample_count=0)
    with pytest.raises(trainer.DiagnosticPPOContractError):
        trainer.promotion_blocking_evidence_receipt(
            [
                {
                    "rollout_step_1based": 1,
                    "env_index": 0,
                    "joint_actual_forbidden_observed_ticks": 1,
                    "reasons": [trainer.PROMOTION_BLOCKING_REASON],
                }
            ],
            checked_sample_count=0,
        )


@pytest.mark.parametrize("profile", (abi.A211_PROFILE, abi.C211_PROFILE))
def test_exact_terminal_telemetry_is_cold_load_exact(profile, tmp_path: Path):
    source_env = ExactTerminalTelemetryEnv(profile)
    source = trainer.MujocoDiagnosticPPOTrainer(
        env=source_env,
        identity=source_env.identity,
        config=_config(profile),
    )
    source.run_update()
    path = tmp_path / f"{profile.label.lower()}_terminal.pt"
    checkpoint.ResetBoundaryCheckpoint().save(path, source)
    expected = source.run_update()

    cold_env = ExactTerminalTelemetryEnv(profile)
    cold = trainer.MujocoDiagnosticPPOTrainer(
        env=cold_env,
        identity=cold_env.identity,
        config=_config(profile),
    )
    checkpoint.ResetBoundaryCheckpoint().load(path, cold)
    actual = cold.run_update()
    assert actual["terminal_row_telemetry"] == expected["terminal_row_telemetry"]
    assert actual == expected


def test_wait_task_columns_stay_zero_after_normalizer_has_seen_active_values():
    env, instance = _build(abi.C211_PROFILE)
    active_actor, active_extras = env._pair(active=True)
    active_critic = active_extras["observations"]["critic"]
    instance._normalized_pair(active_actor, active_critic)
    wait_actor, wait_extras = env._pair(active=False)
    wait_critic = wait_extras["observations"]["critic"]
    normalized_actor, normalized_critic = instance._normalized_pair(
        wait_actor, wait_critic
    )
    assert torch.all(
        normalized_actor[:, abi.C211_PROFILE.actor.task_mask_indices] == 0.0
    )
    assert torch.all(
        normalized_critic[:, abi.C211_PROFILE.critic.task_mask_indices] == 0.0
    )


def test_raw_wait_nonzero_task_column_is_rejected_not_silently_remasked():
    env, instance = _build(abi.A211_PROFILE)
    actor, extras = env._pair(active=False)
    actor[:, abi.A211_PROFILE.actor.task_mask_indices[0]] = 1.0
    with pytest.raises(trainer.DiagnosticPPOContractError, match="exact zero"):
        instance._observation_pair(actor, extras, name="test")


@pytest.mark.parametrize("profile", (abi.A211_PROFILE, abi.C211_PROFILE))
def test_two_normalizer_checkpoint_cold_load_matches_exact_next_update(
    profile, tmp_path: Path
):
    _env, source = _build(profile)
    source.run_update()
    path = tmp_path / f"{profile.label.lower()}.pt"
    checkpoint.ResetBoundaryCheckpoint().save(path, source)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 3
    assert set(payload) >= {
        "actor_normalizer_state_dict",
        "critic_normalizer_state_dict",
        "environment_state",
    }
    assert payload["normalizer_identities"] == {
        "actor": profile.actor_normalizer_identity,
        "critic": profile.critic_normalizer_identity,
    }
    expected_receipt = source.run_update()
    expected_model = copy.deepcopy(source.model.state_dict())
    expected_actor_norm = source.actor_normalizer.state_dict()
    expected_critic_norm = source.critic_normalizer.state_dict()

    cold_env, cold = _build(profile)
    checkpoint.ResetBoundaryCheckpoint().load(path, cold)
    assert cold_env.reset_calls == cold_env.step_calls == 0
    actual_receipt = cold.run_update()
    assert actual_receipt == expected_receipt
    for name, value in cold.model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    for name in ("mean", "m2", "count"):
        assert torch.equal(
            cold.actor_normalizer.state_dict()[name], expected_actor_norm[name]
        )
        assert torch.equal(
            cold.critic_normalizer.state_dict()[name], expected_critic_norm[name]
        )


def test_same_width_a211_checkpoint_cannot_load_into_c211(tmp_path: Path):
    _env, source = _build(abi.A211_PROFILE)
    source.run_update()
    path = tmp_path / "a211.pt"
    checkpoint.ResetBoundaryCheckpoint().save(path, source)
    target_env, target = _build(abi.C211_PROFILE)
    before = copy.deepcopy(target.model.state_dict())
    with pytest.raises(checkpoint.CheckpointRefused, match="SHA differs"):
        checkpoint.ResetBoundaryCheckpoint().load(path, target)
    assert target_env.reset_calls == target_env.step_calls == 0
    for name, value in target.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_v1_single_normalizer_checkpoint_is_explicitly_rejected(tmp_path: Path):
    _env, source = _build(abi.C211_PROFILE)
    path = tmp_path / "v2.pt"
    checkpoint.ResetBoundaryCheckpoint().save(path, source)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 1
    payload["kind"] = "a3_mujoco_controlled_diagnostic_reset_boundary_checkpoint_v1"
    payload["normalizer_state_dict"] = payload.pop("actor_normalizer_state_dict")
    payload.pop("critic_normalizer_state_dict")
    legacy = tmp_path / "v1.pt"
    torch.save(payload, legacy)

    _target_env, target = _build(abi.C211_PROFILE)
    with pytest.raises(checkpoint.CheckpointRefused, match="field set|kind/schema"):
        checkpoint.ResetBoundaryCheckpoint().load(legacy, target)


def test_swapped_actor_critic_normalizers_are_rejected_before_mutation(
    tmp_path: Path,
):
    _env, source = _build(abi.A211_PROFILE)
    path = tmp_path / "source.pt"
    checkpoint.ResetBoundaryCheckpoint().save(path, source)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor_state = payload["actor_normalizer_state_dict"]
    payload["actor_normalizer_state_dict"] = payload["critic_normalizer_state_dict"]
    payload["critic_normalizer_state_dict"] = actor_state
    swapped = tmp_path / "swapped.pt"
    torch.save(payload, swapped)

    _target_env, target = _build(abi.A211_PROFILE)
    before = copy.deepcopy(target.model.state_dict())
    with pytest.raises(checkpoint.CheckpointRefused, match="width/epsilon"):
        checkpoint.ResetBoundaryCheckpoint().load(swapped, target)
    for name, value in target.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_checkpoint_binds_exact_normalizer_identity_strings(tmp_path: Path):
    _env, source = _build(abi.C211_PROFILE)
    path = tmp_path / "source.pt"
    checkpoint.ResetBoundaryCheckpoint().save(path, source)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["normalizer_identities"][
        "actor"
    ] = abi.A211_PROFILE.actor_normalizer_identity
    tampered = tmp_path / "identity-drift.pt"
    torch.save(payload, tampered)

    _target_env, target = _build(abi.C211_PROFILE)
    with pytest.raises(checkpoint.CheckpointRefused, match="normalizer identity"):
        checkpoint.ResetBoundaryCheckpoint().load(tampered, target)


def test_critic_only_change_changes_value_but_not_actor_mean():
    _env, instance = _build(abi.A211_PROFILE)
    actor = torch.zeros((2, 211), dtype=torch.float32)
    critic_a = torch.zeros((2, 319), dtype=torch.float32)
    critic_b = torch.ones((2, 319), dtype=torch.float32)
    mean_a = instance.model.actor(actor)
    mean_b = instance.model.actor(actor)
    value_a = instance.model.critic(critic_a)
    value_b = instance.model.critic(critic_b)
    assert torch.equal(mean_a, mean_b)
    assert not torch.equal(value_a, value_b)
