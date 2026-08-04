"""Controlled tests for the diagnostic-only native MuJoCo PPO shell."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

T = importlib.import_module("hope_training.whole_body_tracking.mujoco_native.trainer")
C = importlib.import_module(
    "hope_training.whole_body_tracking.mujoco_native.checkpoint"
)


def _digest(character: str) -> str:
    return character * 64


IDENTITY = T.TrainerIdentity(
    contract_sha256=_digest("a"),
    observation_contract_sha256=_digest("b"),
    action_contract_sha256=_digest("c"),
    reward_contract_sha256=_digest("d"),
)


class FakeDiagnosticVecEnv:
    """Small deterministic protocol fake; reset randomness comes from its seed."""

    def __init__(
        self,
        *,
        identity=IDENTITY,
        num_envs: int = 2,
        observation_dim: int = 3,
        action_dim: int = 2,
        horizon: int = 4,
        blockers=(),
    ) -> None:
        self.identity = identity
        self.num_envs = num_envs
        self.num_observations = observation_dim
        self.num_actions = action_dim
        self.device = "cpu"
        self.horizon = horizon
        self.blockers = tuple(blockers)
        self.reset_calls = 0
        self.step_calls = 0
        self._tick = 0
        self._boundary = True
        self.last_reset_seed = None

    def diagnostic_training_receipt(self):
        return {
            "kind": T.DIAGNOSTIC_TRAINER_RECEIPT_KIND,
            "ppo_ready": not self.blockers,
            "reward_available": True,
            "normal_step_available": True,
            "reset_boundary_checkpoint_available": True,
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
            "blockers": list(self.blockers),
            **self.identity.as_dict(),
        }

    def is_reset_boundary(self):
        return self._boundary

    def reset(self, *, seed: int):
        self.reset_calls += 1
        self.last_reset_seed = seed
        self._tick = 0
        self._boundary = False
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        observations = torch.randn(
            self.num_envs,
            self.num_observations,
            generator=generator,
            dtype=torch.float32,
        )
        return observations, {"reset_seed": seed}

    def step(self, actions):
        self.step_calls += 1
        self._tick += 1
        padded = torch.zeros(self.num_envs, self.num_observations, dtype=torch.float32)
        padded[:, : self.num_actions] = actions
        observations = padded + self._tick * 0.05
        rewards = -actions.square().sum(dim=-1) + 0.1 * self._tick
        done = self._tick >= self.horizon
        dones = torch.full((self.num_envs,), done, dtype=torch.bool)
        self._boundary = done
        return observations, rewards, dones, {
            "tick": self._tick,
            "time_outs": dones.clone(),
        }


def _config(*, rollout_steps=4):
    return T.DiagnosticPPOConfig(
        observation_dim=3,
        action_dim=2,
        rollout_steps=rollout_steps,
        hidden_dims=(8,),
        seed=17,
        learning_rate=1.0e-3,
    )


def _trainer(env, *, identity=IDENTITY, rollout_steps=4):
    return T.MujocoDiagnosticPPOTrainer(
        env=env,
        identity=identity,
        config=_config(rollout_steps=rollout_steps),
    )


def _state_clone(trainer):
    return {
        key: value.detach().clone() for key, value in trainer.model.state_dict().items()
    }


def test_blocked_receipt_is_rejected_before_reset_step_or_optimizer_update():
    env = FakeDiagnosticVecEnv(blockers=("real_reward_not_bound",))
    trainer = _trainer(env)
    before = _state_clone(trainer)

    with pytest.raises(T.DiagnosticPPOBlocked, match="ppo_ready"):
        trainer.run_update()

    assert env.reset_calls == 0
    assert env.step_calls == 0
    assert trainer.update_counter == 0
    assert trainer.optimizer.state == {}
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize(
    "field,value",
    [
        ("diagnostic_unauthorized", False),
        ("formal_authorized", True),
        ("mid_episode_resume", True),
    ],
)
def test_receipt_cannot_expand_diagnostic_authority(field, value):
    env = FakeDiagnosticVecEnv()
    original = env.diagnostic_training_receipt

    def changed():
        receipt = original()
        receipt[field] = value
        return receipt

    env.diagnostic_training_receipt = changed
    trainer = _trainer(env)
    with pytest.raises(T.DiagnosticPPOBlocked, match=field):
        trainer.run_update()
    assert env.reset_calls == env.step_calls == 0


def test_one_minimal_update_is_finite_and_stays_explicitly_diagnostic():
    env = FakeDiagnosticVecEnv()
    trainer = _trainer(env)
    before = _state_clone(trainer)

    receipt = trainer.run_update()

    assert receipt["status"] == "CONTROLLED_DIAGNOSTIC_PPO_UPDATE_COMPLETE"
    assert receipt["update_counter"] == 1
    assert receipt["batch_size"] == 8
    assert receipt["at_reset_boundary"] is True
    assert receipt["diagnostic_unauthorized"] is True
    assert receipt["formal_authorized"] is False
    assert receipt["mid_episode_resume"] is False
    assert all(
        torch.isfinite(torch.tensor(receipt[name]))
        for name in (
            "loss",
            "surrogate_loss",
            "value_loss",
            "entropy",
            "pre_clip_grad_norm",
        )
    )
    assert any(
        not torch.equal(value, before[key])
        for key, value in trainer.model.state_dict().items()
    )


def test_checkpoint_refuses_mid_episode_and_does_not_create_file(tmp_path):
    env = FakeDiagnosticVecEnv(horizon=5)
    trainer = _trainer(env, rollout_steps=4)
    trainer.run_update()
    destination = tmp_path / "mid_episode.pt"

    with pytest.raises(T.ResetBoundaryRequired, match="mid-episode"):
        C.ResetBoundaryCheckpoint().save(destination, trainer)

    assert not destination.exists()


def test_blocked_checkpoint_save_refuses_before_file_write(tmp_path):
    env = FakeDiagnosticVecEnv()
    trainer = _trainer(env)
    destination = tmp_path / "blocked.pt"
    env.blockers = ("reward_receipt_revoked",)

    with pytest.raises(T.DiagnosticPPOBlocked):
        C.ResetBoundaryCheckpoint().save(destination, trainer)

    assert not destination.exists()
    assert env.reset_calls == env.step_calls == 0


def test_checkpoint_contains_complete_state_and_cold_load_matches_next_update(
    tmp_path,
):
    uninterrupted_env = FakeDiagnosticVecEnv()
    uninterrupted = _trainer(uninterrupted_env)
    first = uninterrupted.run_update()
    path = tmp_path / "boundary.pt"
    save_receipt = C.ResetBoundaryCheckpoint().save(path, uninterrupted)
    assert save_receipt["update_counter"] == 1
    assert save_receipt["mid_episode_resume"] is False

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) == {
        "schema_version",
        "kind",
        "identity",
        "config_sha256",
        "normalizer_identities",
        "model_state_dict",
        "optimizer_state_dict",
        "actor_normalizer_state_dict",
        "critic_normalizer_state_dict",
        "rng_state",
        "update_counter",
        "last_update_receipt",
        "environment_state",
        "boundary",
    }
    # The generic fake VecEnv has no continuation-state producer.  Schema v3
    # still carries the field so C211 can seal its 5--25 tick WAIT schedule.
    assert payload["environment_state"] is None
    assert payload["update_counter"] == first["update_counter"]
    assert payload["normalizer_identities"] == {
        "actor": uninterrupted.config.actor_normalizer_identity,
        "critic": uninterrupted.config.critic_normalizer_identity,
    }
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu"}
    expected_second = uninterrupted.run_update()
    expected_state = _state_clone(uninterrupted)
    expected_actor_normalizer = uninterrupted.actor_normalizer.state_dict()
    expected_critic_normalizer = uninterrupted.critic_normalizer.state_dict()

    cold_env = FakeDiagnosticVecEnv()
    cold = _trainer(cold_env)
    load_receipt = C.ResetBoundaryCheckpoint().load(path, cold)
    assert cold_env.reset_calls == cold_env.step_calls == 0
    assert load_receipt["at_reset_boundary"] is True
    actual_second = cold.run_update()

    assert actual_second == expected_second
    for key, value in cold.model.state_dict().items():
        assert torch.equal(value, expected_state[key])
    for key in ("mean", "m2", "count"):
        assert torch.equal(
            cold.actor_normalizer.state_dict()[key], expected_actor_normalizer[key]
        )
        assert torch.equal(
            cold.critic_normalizer.state_dict()[key], expected_critic_normalizer[key]
        )
    assert (
        cold.optimizer.state_dict()["param_groups"]
        == uninterrupted.optimizer.state_dict()["param_groups"]
    )
    for parameter_id, state in cold.optimizer.state_dict()["state"].items():
        expected = uninterrupted.optimizer.state_dict()["state"][parameter_id]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, expected[key])
            else:
                assert value == expected[key]


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("contract_sha256", _digest("1")),
        ("observation_contract_sha256", _digest("2")),
        ("action_contract_sha256", _digest("3")),
        ("reward_contract_sha256", _digest("4")),
    ],
)
def test_load_identity_mismatch_refuses_before_env_step(tmp_path, field, replacement):
    source_env = FakeDiagnosticVecEnv()
    source = _trainer(source_env)
    source.run_update()
    path = tmp_path / f"{field}.pt"
    C.ResetBoundaryCheckpoint().save(path, source)

    values = IDENTITY.as_dict()
    values[field] = replacement
    other_identity = T.TrainerIdentity(**values)
    target_env = FakeDiagnosticVecEnv(identity=other_identity)
    target = _trainer(target_env, identity=other_identity)
    before = _state_clone(target)

    with pytest.raises(C.CheckpointRefused, match="SHA differs"):
        C.ResetBoundaryCheckpoint().load(path, target)

    assert target_env.reset_calls == target_env.step_calls == 0
    assert target.update_counter == 0
    for key, value in target.model.state_dict().items():
        assert torch.equal(value, before[key])


def test_checkpoint_save_is_no_clobber(tmp_path):
    env = FakeDiagnosticVecEnv()
    trainer = _trainer(env)
    path = tmp_path / "checkpoint.pt"
    first = C.ResetBoundaryCheckpoint().save(path, trainer)
    original = path.read_bytes()

    with pytest.raises(C.CheckpointRefused, match="no-clobber"):
        C.ResetBoundaryCheckpoint().save(path, trainer)

    assert path.read_bytes() == original
    assert first["update_counter"] == 0
