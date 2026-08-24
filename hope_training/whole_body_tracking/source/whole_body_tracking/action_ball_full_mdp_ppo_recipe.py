"""Typed, dependency-free PPO V5 recipe for the continuous FullMDP lanes.

The shared ``cfg/algo/ppo.yaml`` remains the legacy/default recipe for every
other task.  FullMDP Isaac and MuJoCo consumers derive their effective values
from the frozen object below so rollout, GAE and update cadence cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class ActionBallFullMdpPpoRecipe:
    """Complete FullMDP PPO recipe, including its finite run schedule."""

    # V5 is a latency/refresh tradeoff, not a fixed-tape-equivalent speedup.
    # Halving N and doubling the update count preserves the total transition
    # budget, while halving the minibatch count preserves minibatch size and
    # the total number of optimizer steps.  It deliberately refreshes the
    # policy twice as often, so N belongs to the scientific learning identity.
    kind: str = "action_ball_full_mdp_ppo_v5"
    num_envs: int = 2_048
    num_steps_per_env: int = 48
    max_iterations: int = 25_000
    save_interval: int = 500
    empirical_normalization: bool = False

    policy_class_name: str = "ActorCritic"
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 0.05
    noise_std_type: str = "log"

    algorithm_class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.98
    value_loss_coef: float = 1.0
    # ``log_std`` is already learned by PPO.  A permanent positive entropy
    # coefficient adds the same upward gradient to every action dimension on
    # every minibatch, independent of task quality.  In the v2 long runs that
    # drove mean std past 1.0 and destroyed an already learned
    # balance policy.  Keep exploration in the learned distribution instead
    # of paying an unconditional lifetime bonus for more noise.
    entropy_coef: float = 0.0
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    normalize_advantage_per_mini_batch: bool = False

    def policy(self) -> dict:
        """Return the backend-neutral RSL actor/critic configuration."""

        return {
            "class_name": self.policy_class_name,
            "actor_hidden_dims": list(self.actor_hidden_dims),
            "critic_hidden_dims": list(self.critic_hidden_dims),
            "activation": self.activation,
            "init_noise_std": self.init_noise_std,
            "noise_std_type": self.noise_std_type,
        }

    def algorithm(self) -> dict:
        """Return every effective RSL PPO coefficient used by FullMDP."""

        return {
            "class_name": self.algorithm_class_name,
            "num_learning_epochs": self.num_learning_epochs,
            "num_mini_batches": self.num_mini_batches,
            "clip_param": self.clip_param,
            "gamma": self.gamma,
            "lam": self.lam,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "use_clipped_value_loss": self.use_clipped_value_loss,
            "schedule": self.schedule,
            "desired_kl": self.desired_kl,
            "normalize_advantage_per_mini_batch": (
                self.normalize_advantage_per_mini_batch
            ),
            "rnd_cfg": None,
            "symmetry_cfg": None,
        }

    def learning_recipe(self) -> dict:
        """Return the existing training-contract scientific recipe shape.

        Total budget and save cadence are intentionally absent because the
        existing serializer treats them as operational.  They remain frozen on
        this same object and are consumed directly by both runners.
        """

        return {
            "schema_version": 2,
            "runner": {
                "num_envs": self.num_envs,
                "num_steps_per_env": self.num_steps_per_env,
                "empirical_normalization": self.empirical_normalization,
            },
            "policy": self.policy(),
            "algorithm": self.algorithm(),
        }

    def learning_recipe_sha256(self) -> str:
        return self._sha256(self.learning_recipe())

    def execution_recipe(self) -> dict:
        """Return the complete finite execution identity for run artifacts."""

        return {
            "schema_version": 2,
            "kind": self.kind,
            "runner": {
                "num_envs": self.num_envs,
                "num_steps_per_env": self.num_steps_per_env,
                "max_iterations": self.max_iterations,
                "save_interval": self.save_interval,
            },
            "learning_recipe": self.learning_recipe(),
        }

    def recipe_sha256(self) -> str:
        """Hash the learning recipe together with budget and save cadence."""

        return self._sha256(self.execution_recipe())

    @staticmethod
    def _sha256(value: dict) -> str:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def isaac_overrides(self) -> dict:
        """Return the FullMDP-only Hydra/RSL override mapping."""

        return {
            "runner": {
                "num_steps_per_env": self.num_steps_per_env,
                "max_iterations": self.max_iterations,
                "save_interval": self.save_interval,
                "empirical_normalization": self.empirical_normalization,
            },
            "policy": self.policy(),
            "algorithm": self.algorithm(),
        }

    def mujoco_train_cfg(self) -> dict:
        """Return the upstream RSL-RL 3 config used by the MuJoCo lane."""

        return {
            "num_steps_per_env": self.num_steps_per_env,
            "save_interval": self.save_interval,
            "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
            "policy": {
                **self.policy(),
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
            },
            "algorithm": self.algorithm(),
        }


ACTION_BALL_FULL_MDP_PPO_RECIPE = ActionBallFullMdpPpoRecipe()
