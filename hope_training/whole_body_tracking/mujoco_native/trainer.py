"""Controlled, diagnostic-only PPO shell for the native MuJoCo trainer lane.

This module intentionally does not make ``MujocoN1DiagnosticVecEnv`` trainable.
It consumes a small VecEnv protocol so that reward/termination successors can
prove one finite PPO update and checkpoint continuity without importing
``rsl_rl``.  A runtime receipt must explicitly authorize that narrow operation
while retaining all formal/promotion/deployment prohibitions.

The shell supports reset-boundary continuation only.  It neither captures nor
restores an in-flight environment episode.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DIAGNOSTIC_TRAINER_RECEIPT_KIND = "a3_mujoco_controlled_diagnostic_ppo_readiness_v1"
DIAGNOSTIC_UPDATE_RECEIPT_KIND = "a3_mujoco_controlled_diagnostic_ppo_update_v1"

_IDENTITY_FIELDS = (
    "contract_sha256",
    "observation_contract_sha256",
    "action_contract_sha256",
    "reward_contract_sha256",
)


class DiagnosticPPOError(RuntimeError):
    """Base class for controlled diagnostic trainer failures."""


class DiagnosticPPOBlocked(DiagnosticPPOError):
    """The runtime receipt does not authorize even a diagnostic PPO update."""


class DiagnosticPPOContractError(DiagnosticPPOError):
    """The trainer, VecEnv, or tensor ABI differs from the frozen contract."""


class ResetBoundaryRequired(DiagnosticPPOError):
    """Checkpointing was requested while an episode was in flight."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DiagnosticPPOError(
            "torch is required for the diagnostic PPO shell"
        ) from exc
    return torch


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DiagnosticPPOContractError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise DiagnosticPPOContractError(f"{name} must be a positive plain integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DiagnosticPPOContractError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DiagnosticPPOContractError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class TrainerIdentity:
    """Content identities that must match across VecEnv, update, and load."""

    contract_sha256: str
    observation_contract_sha256: str
    action_contract_sha256: str
    reward_contract_sha256: str

    def __post_init__(self) -> None:
        for field in _IDENTITY_FIELDS:
            _sha256(getattr(self, field), field)

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _IDENTITY_FIELDS}


@dataclass(frozen=True)
class DiagnosticPPOConfig:
    """Small fixed PPO recipe used only for a finite plumbing diagnostic."""

    observation_dim: int
    action_dim: int
    rollout_steps: int = 4
    hidden_dims: tuple[int, ...] = (32, 32)
    seed: int = 0
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    initial_action_std: float = 0.2
    normalizer_epsilon: float = 1.0e-5

    def __post_init__(self) -> None:
        _positive_int(self.observation_dim, "observation_dim")
        _positive_int(self.action_dim, "action_dim")
        _positive_int(self.rollout_steps, "rollout_steps")
        if not self.hidden_dims or any(
            type(width) is not int or width < 1 for width in self.hidden_dims
        ):
            raise DiagnosticPPOContractError(
                "hidden_dims must be a non-empty tuple of positive integers"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise DiagnosticPPOContractError("seed must be a non-negative integer")
        _finite_positive(self.learning_rate, "learning_rate")
        _finite_positive(self.gamma, "gamma")
        _finite_positive(self.gae_lambda, "gae_lambda")
        _finite_positive(self.clip_param, "clip_param")
        _finite_positive(self.value_loss_coef, "value_loss_coef")
        if not math.isfinite(float(self.entropy_coef)) or self.entropy_coef < 0.0:
            raise DiagnosticPPOContractError(
                "entropy_coef must be finite and non-negative"
            )
        _finite_positive(self.max_grad_norm, "max_grad_norm")
        _finite_positive(self.initial_action_std, "initial_action_std")
        _finite_positive(self.normalizer_epsilon, "normalizer_epsilon")
        if self.gamma > 1.0 or self.gae_lambda > 1.0 or self.clip_param >= 1.0:
            raise DiagnosticPPOContractError(
                "gamma/gae_lambda must be <=1 and clip_param must be <1"
            )

    @property
    def content_sha256(self) -> str:
        return _canonical_json_sha256(asdict(self))


def validate_diagnostic_readiness_receipt(
    receipt: Mapping[str, Any], identity: TrainerIdentity
) -> dict[str, Any]:
    """Fail closed unless the receipt permits only controlled diagnostic PPO."""

    if not isinstance(receipt, Mapping):
        raise DiagnosticPPOBlocked("diagnostic PPO readiness receipt is absent")
    required = {
        "kind": DIAGNOSTIC_TRAINER_RECEIPT_KIND,
        "ppo_ready": True,
        "reward_available": True,
        "normal_step_available": True,
        "reset_boundary_checkpoint_available": True,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
        "mid_episode_resume": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise DiagnosticPPOBlocked(
                f"diagnostic PPO readiness receipt field {key!r} must equal {expected!r}"
            )
    expected_identity = identity.as_dict()
    for field, expected in expected_identity.items():
        try:
            actual = _sha256(receipt.get(field), f"receipt.{field}")
        except DiagnosticPPOContractError as exc:
            raise DiagnosticPPOBlocked(str(exc)) from exc
        if actual != expected:
            raise DiagnosticPPOBlocked(
                f"diagnostic PPO readiness receipt {field} differs from trainer identity"
            )
    blockers = receipt.get("blockers")
    if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
        raise DiagnosticPPOBlocked(
            "diagnostic PPO readiness blockers must be a sequence"
        )
    if list(blockers):
        raise DiagnosticPPOBlocked(
            "diagnostic PPO is blocked: " + ",".join(str(item) for item in blockers)
        )
    return copy.deepcopy(dict(receipt))


class _RunningNormalizer:
    """Deterministic CPU running moments with an explicit state dictionary."""

    def __init__(self, width: int, epsilon: float) -> None:
        torch = _require_torch()
        self.width = width
        self.epsilon = float(epsilon)
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.m2 = torch.zeros(width, dtype=torch.float64)
        self.count = torch.zeros((), dtype=torch.float64)

    def update(self, observations: Any) -> None:
        torch = _require_torch()
        batch = observations.detach().to(dtype=torch.float64, device="cpu")
        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
        if float(self.count.item()) == 0.0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            self.count.fill_(batch_count)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean.add_(delta * (batch_count / total))
        self.m2.add_(batch_m2 + delta.square() * self.count * batch_count / total)
        self.count.copy_(total)

    def normalize(self, observations: Any) -> Any:
        torch = _require_torch()
        denominator = torch.clamp(self.count, min=1.0)
        variance = self.m2 / denominator
        scale = torch.sqrt(torch.clamp(variance, min=self.epsilon))
        return (
            (observations.to(dtype=torch.float64, device="cpu") - self.mean)
            .div(scale)
            .to(dtype=torch.float32)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "epsilon": self.epsilon,
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
            "count": self.count.clone(),
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        torch = _require_torch()
        if not isinstance(state, Mapping) or set(state) != {
            "width",
            "epsilon",
            "mean",
            "m2",
            "count",
        }:
            raise DiagnosticPPOContractError("normalizer checkpoint schema differs")
        if state["width"] != self.width or float(state["epsilon"]) != self.epsilon:
            raise DiagnosticPPOContractError("normalizer width/epsilon differs")
        if tuple(state["mean"].shape) != (self.width,) or tuple(state["m2"].shape) != (
            self.width,
        ):
            raise DiagnosticPPOContractError("normalizer moment shape differs")
        if tuple(state["count"].shape) != ():
            raise DiagnosticPPOContractError("normalizer count must be scalar")
        for name in ("mean", "m2", "count"):
            if (
                not isinstance(state[name], torch.Tensor)
                or not torch.isfinite(state[name]).all()
            ):
                raise DiagnosticPPOContractError(
                    f"normalizer checkpoint {name} is not a finite tensor"
                )
        if float(state["count"].item()) < 0.0 or bool(torch.any(state["m2"] < 0.0)):
            raise DiagnosticPPOContractError("normalizer moments are invalid")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.validate_state_dict(state)
        self.mean.copy_(state["mean"].to(dtype=self.mean.dtype, device="cpu"))
        self.m2.copy_(state["m2"].to(dtype=self.m2.dtype, device="cpu"))
        self.count.copy_(state["count"].to(dtype=self.count.dtype, device="cpu"))


def _mlp(widths: Sequence[int]) -> Any:
    torch = _require_torch()
    layers = []
    for index, (input_width, output_width) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(input_width, output_width))
        if index < len(widths) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def _build_actor_critic(config: DiagnosticPPOConfig) -> Any:
    torch = _require_torch()
    module = torch.nn.Module()
    module.actor = _mlp(
        (config.observation_dim, *config.hidden_dims, config.action_dim)
    )
    module.critic = _mlp((config.observation_dim, *config.hidden_dims, 1))
    module.register_parameter(
        "log_std",
        torch.nn.Parameter(
            torch.full(
                (config.action_dim,),
                math.log(config.initial_action_std),
                dtype=torch.float32,
            )
        ),
    )
    return module


def _tensor_digest(state: Mapping[str, Any]) -> str:
    torch = _require_torch()
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class MujocoDiagnosticPPOTrainer:
    """One-update-at-a-time CPU PPO runner with a fail-closed VecEnv receipt."""

    def __init__(
        self,
        *,
        env: Any,
        identity: TrainerIdentity,
        config: DiagnosticPPOConfig,
    ) -> None:
        torch = _require_torch()
        self.env = env
        self.identity = identity
        self.config = config
        self.num_envs = _positive_int(getattr(env, "num_envs", None), "env.num_envs")
        for attribute, expected in (
            ("num_observations", config.observation_dim),
            ("num_actions", config.action_dim),
        ):
            actual = getattr(env, attribute, None)
            if actual != expected:
                raise DiagnosticPPOContractError(
                    f"env.{attribute}={actual!r} differs from configured {expected}"
                )
        if getattr(env, "device", "cpu") not in ("cpu", torch.device("cpu")):
            raise DiagnosticPPOContractError("diagnostic PPO shell is CPU-only")
        if not callable(getattr(env, "diagnostic_training_receipt", None)):
            raise DiagnosticPPOContractError(
                "VecEnv must expose diagnostic_training_receipt()"
            )
        if not callable(getattr(env, "is_reset_boundary", None)):
            raise DiagnosticPPOContractError("VecEnv must expose is_reset_boundary()")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.model = _build_actor_critic(config).to(device="cpu")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.normalizer = _RunningNormalizer(
            config.observation_dim, config.normalizer_epsilon
        )
        self.update_counter = 0
        self._observations = None
        self._last_update_receipt: dict[str, Any] | None = None

    def _validated_readiness(self) -> dict[str, Any]:
        try:
            receipt = self.env.diagnostic_training_receipt()
        except Exception as exc:  # noqa: BLE001 - external protocol boundary
            raise DiagnosticPPOBlocked(
                "VecEnv diagnostic readiness receipt could not be read"
            ) from exc
        return validate_diagnostic_readiness_receipt(receipt, self.identity)

    def is_reset_boundary(self) -> bool:
        try:
            env_boundary = self.env.is_reset_boundary()
        except Exception as exc:  # noqa: BLE001 - external protocol boundary
            raise ResetBoundaryRequired(
                "VecEnv reset boundary could not be read"
            ) from exc
        if type(env_boundary) is not bool:
            raise ResetBoundaryRequired("VecEnv reset boundary must be a plain boolean")
        return env_boundary and self._observations is None

    def assert_reset_boundary(self) -> None:
        if not self.is_reset_boundary():
            raise ResetBoundaryRequired(
                "only an explicit full reset boundary can be checkpointed; "
                "mid-episode resume is unsupported"
            )

    def _observations_tensor(self, value: Any, name: str) -> Any:
        torch = _require_torch()
        if not isinstance(value, torch.Tensor):
            raise DiagnosticPPOContractError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu" or tuple(value.shape) != (
            self.num_envs,
            self.config.observation_dim,
        ):
            raise DiagnosticPPOContractError(
                f"{name} must be finite CPU [{self.num_envs}, {self.config.observation_dim}]"
            )
        value = value.to(dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise DiagnosticPPOContractError(f"{name} contains non-finite values")
        return value

    def _reset(self) -> Any:
        torch = _require_torch()
        reset_seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
        result = self.env.reset(seed=reset_seed)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[1], Mapping)
        ):
            raise DiagnosticPPOContractError(
                "VecEnv reset must return (observations, extras)"
            )
        return self._observations_tensor(result[0], "reset observations")

    def _distribution_terms(
        self, normalized_observations: Any, actions: Any | None = None
    ) -> tuple[Any, Any, Any, Any]:
        torch = _require_torch()
        means = self.model.actor(normalized_observations)
        std = torch.exp(self.model.log_std).expand_as(means)
        if actions is None:
            actions = means + torch.randn_like(means) * std
        variance = std.square()
        log_prob = -0.5 * (
            ((actions - means).square() / variance)
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1)
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(std)).sum(dim=-1)
        values = self.model.critic(normalized_observations).squeeze(-1)
        return actions, log_prob, entropy, values

    def run_update(self) -> dict[str, Any]:
        """Run exactly one finite rollout and one full-batch PPO optimizer step."""

        torch = _require_torch()
        readiness = self._validated_readiness()
        observations = (
            self._observations if self._observations is not None else self._reset()
        )
        normalized_rows = []
        action_rows = []
        old_log_prob_rows = []
        reward_rows = []
        done_rows = []
        value_rows = []

        for _step in range(self.config.rollout_steps):
            self.normalizer.update(observations)
            normalized = self.normalizer.normalize(observations)
            with torch.no_grad():
                actions, log_prob, _entropy, values = self._distribution_terms(
                    normalized
                )
            result = self.env.step(actions.detach().clone())
            if not isinstance(result, tuple) or len(result) != 4:
                raise DiagnosticPPOContractError(
                    "VecEnv step must return (observations, rewards, dones, extras)"
                )
            next_observations, rewards, dones, extras = result
            next_observations = self._observations_tensor(
                next_observations, "step observations"
            )
            if (
                not isinstance(rewards, torch.Tensor)
                or rewards.device.type != "cpu"
                or tuple(rewards.shape) != (self.num_envs,)
                or not torch.isfinite(rewards).all()
            ):
                raise DiagnosticPPOContractError(
                    "VecEnv rewards must be finite CPU [num_envs]"
                )
            if (
                not isinstance(dones, torch.Tensor)
                or dones.dtype != torch.bool
                or dones.device.type != "cpu"
                or tuple(dones.shape) != (self.num_envs,)
            ):
                raise DiagnosticPPOContractError(
                    "VecEnv dones must be CPU bool [num_envs]"
                )
            if not isinstance(extras, Mapping):
                raise DiagnosticPPOContractError("VecEnv step extras must be a mapping")
            normalized_rows.append(normalized)
            action_rows.append(actions.detach())
            old_log_prob_rows.append(log_prob.detach())
            reward_rows.append(rewards.to(dtype=torch.float32).detach())
            done_rows.append(dones.detach().clone())
            value_rows.append(values.detach())
            observations = next_observations

        all_rows_done = bool(torch.all(done_rows[-1]).item())
        if all_rows_done:
            if self.env.is_reset_boundary() is not True:
                raise DiagnosticPPOContractError(
                    "all done rows must expose an explicit VecEnv reset boundary"
                )
            self._observations = None
            next_value = torch.zeros(self.num_envs, dtype=torch.float32)
        else:
            self._observations = observations.detach().clone()
            self.normalizer.update(observations)
            with torch.no_grad():
                normalized_next = self.normalizer.normalize(observations)
                next_value = self.model.critic(normalized_next).squeeze(-1)

        advantages = []
        gae = torch.zeros(self.num_envs, dtype=torch.float32)
        for index in reversed(range(self.config.rollout_steps)):
            nonterminal = (~done_rows[index]).to(dtype=torch.float32)
            delta = (
                reward_rows[index]
                + self.config.gamma * next_value * nonterminal
                - value_rows[index]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages.append(gae.clone())
            next_value = value_rows[index]
        advantages.reverse()

        flat_observations = torch.cat(normalized_rows, dim=0)
        flat_actions = torch.cat(action_rows, dim=0)
        flat_old_log_prob = torch.cat(old_log_prob_rows, dim=0)
        flat_advantages = torch.cat(advantages, dim=0)
        flat_returns = flat_advantages + torch.cat(value_rows, dim=0)
        advantage_mean = flat_advantages.mean()
        advantage_std = flat_advantages.std(unbiased=False)
        flat_advantages = (flat_advantages - advantage_mean) / torch.clamp(
            advantage_std, min=1.0e-8
        )

        _actions, new_log_prob, entropy, new_values = self._distribution_terms(
            flat_observations, flat_actions
        )
        ratio = torch.exp(new_log_prob - flat_old_log_prob)
        unclipped = ratio * flat_advantages
        clipped = (
            torch.clamp(
                ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
            )
            * flat_advantages
        )
        surrogate_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = (new_values - flat_returns).square().mean()
        entropy_mean = entropy.mean()
        loss = (
            surrogate_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy_mean
        )
        if not torch.isfinite(loss):
            raise DiagnosticPPOError("diagnostic PPO loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        if not torch.isfinite(pre_clip_grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise DiagnosticPPOError("diagnostic PPO gradient norm is non-finite")
        self.optimizer.step()
        if any(
            not torch.isfinite(parameter).all() for parameter in self.model.parameters()
        ):
            raise DiagnosticPPOError("diagnostic PPO produced non-finite parameters")

        self.update_counter += 1
        rollout_digest = hashlib.sha256()
        for tensor in (
            flat_observations,
            flat_actions,
            torch.cat(reward_rows, dim=0),
            torch.cat(done_rows, dim=0),
        ):
            rollout_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        receipt = {
            "schema_version": 1,
            "kind": DIAGNOSTIC_UPDATE_RECEIPT_KIND,
            "status": "CONTROLLED_DIAGNOSTIC_PPO_UPDATE_COMPLETE",
            "update_counter": self.update_counter,
            "num_envs": self.num_envs,
            "rollout_steps": self.config.rollout_steps,
            "batch_size": self.num_envs * self.config.rollout_steps,
            **self.identity.as_dict(),
            "config_sha256": self.config.content_sha256,
            "readiness_receipt_sha256": _canonical_json_sha256(readiness),
            "loss": float(loss.detach().item()),
            "surrogate_loss": float(surrogate_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "entropy": float(entropy_mean.detach().item()),
            "pre_clip_grad_norm": float(pre_clip_grad_norm.detach().item()),
            "rollout_sha256": rollout_digest.hexdigest(),
            "model_state_sha256": _tensor_digest(self.model.state_dict()),
            "normalizer_state_sha256": _tensor_digest(self.normalizer.state_dict()),
            "at_reset_boundary": self.is_reset_boundary(),
            "diagnostic_unauthorized": True,
            "formal_authorized": False,
            "mid_episode_resume": False,
            "authorization": {
                "formal_training": False,
                "promotion": False,
                "deployment": False,
                "hardware": False,
            },
        }
        receipt["content_sha256"] = _canonical_json_sha256(receipt)
        self._last_update_receipt = copy.deepcopy(receipt)
        return receipt

    def checkpoint_state(self) -> dict[str, Any]:
        """Return complete trainer state, only at an explicit reset boundary."""

        self._validated_readiness()
        self.assert_reset_boundary()
        torch = _require_torch()
        return {
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "normalizer_state_dict": self.normalizer.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state().clone(),
            },
            "update_counter": self.update_counter,
            "last_update_receipt": copy.deepcopy(self._last_update_receipt),
        }


__all__ = [
    "DIAGNOSTIC_TRAINER_RECEIPT_KIND",
    "DIAGNOSTIC_UPDATE_RECEIPT_KIND",
    "DiagnosticPPOBlocked",
    "DiagnosticPPOConfig",
    "DiagnosticPPOContractError",
    "DiagnosticPPOError",
    "MujocoDiagnosticPPOTrainer",
    "ResetBoundaryRequired",
    "TrainerIdentity",
    "validate_diagnostic_readiness_receipt",
]
